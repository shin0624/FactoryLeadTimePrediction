using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using Tensorflow;

namespace FactoryLeadTimePrediction.DeepLearning.A_CastingDefectClassification
{
    public class DeepLearning_CastingDefect // 주조품 양/불량 이진분류 딥러닝 실습의 메인 실행 클래스. 
    {
        private static DeepLearning_CastingDefect? _instance; //static 필드는 CLR이 null로 초기화 하므로 "?"가 맞는 선언
        public static DeepLearning_CastingDefect Instance()
        {
            if(_instance == null)
            {
                _instance = new DeepLearning_CastingDefect();
            }
            return _instance;
        }// 반환형은 non-nullable.

        // 경로 설정
        private const string DatasetRoot = @"C:\GitHubClone\FactoryLeadTimePrediction\data\Image_Classifiacation\casting_data"; // @""(verbatim stirng)  : 백슬래시를 이스케이프하지 않아도 되며, 한글/공백이 그대로 들어갈 수 있음

        private const string TrainFolder = "train"; // 6633장(불량 3758, 정상 2875)
        private const string TestFolder = "test"; // 715장(불량 453, 정상 262)

        private const string WorkspacePath = "workspace_casting"; // 트레이너가 피처 캐시(.tsv)와 임시 파일을 쓰는 폴더.
        private const string ModelSavePath = "models/CastingDefect_Model.zip"; // 학습된 모델.

        // 하이퍼 파라미터
        private const int Epochs = 20; // 에폭 수
        private const int BatchSize = 16; // 배치크기. cpu 학습이고 메모리 여유가 적다면 줄임
        private const float ValidationFraction = 0.1f;// train에서 validation용으로 떼어낼 비율 (10% = 약 663장)
        private const int Seed = 42; // 재현성

        public void StartCastingDefectClassification() // 데이터 로딩 -> train/val분할 -> 학습 -> 평가 -> 오분류 분석 -> 저장 파이프라인 실행 메서드 
        {
            System.Console.WriteLine("==========[주조품 양/불량 분류] 시작==========");

            var mlContext = new MLContext(seed : Seed);// seed를 고정하면 TrainTestSplit의 분할과 내부 난수가 고정되어 실행마다 같은 결과가 나온다.

            // 데이터 로딩 + 검산
            var loader = new CastingImageLoader();

            System.Console.WriteLine();
            System.Console.WriteLine("----------train 폴더 로딩----------");
            var trainAll = loader.LoadFromFolder(Path.Combine(DatasetRoot, TrainFolder));

            System.Console.WriteLine();
            System.Console.WriteLine("----------test 폴더 로딩----------");
            var testData = loader.LoadFromFolder(Path.Combine(DatasetRoot, TestFolder));

            loader.PrintSummary("train 전체 (기대 : 6633장 / 불량 3758 / 정상 2875)", trainAll);
            loader.PrintSummary("test (기대 : 715장 / 불량 453 / 정상 262 / 베이스라인 63.36%)", testData);

            if(trainAll.Count == 0 || testData.Count == 0)
            {
                System.Console.WriteLine("[중단] 데이터를 읽지 못했습니다. DatasetRoot 경로를 확인하세요.");
                return;
            }

            // IDataView 변환 및 train/val 분할
            IDataView trainAllView = mlContext.Data.LoadFromEnumerable(trainAll);
            IDataView testView = mlContext.Data.LoadFromEnumerable(testData);
            // List<T> -> IDataView 형으로 변환. 이미지 픽셀이 아니라 경로 문자열만 들어있으므로 ,이 단계는 매우 가벼움

            var split = mlContext.Data.TrainTestSplit(trainAllView, testFraction : ValidationFraction, seed : Seed);
            IDataView trainView = split.TrainSet; // 실제 학습용 (약 5970장)
            IDataView valView = split.TestSet; // 에폭별 검증용(약 663장)
            /*
            왜 Train / Test / Val 인가?
            - ImageClassificationTrainer는 에폭마다 검증 정확도를 출력하고, 조기 종료 판단에도 사용
            - 여기에 test를 넣으면 test가 학습 과정에 관여하게 되어버림(최종 성능을 과대평가)
            - test는 모든 튜닝이 끝난 뒤 단 한번만 보는 것이 원칙
            - LightGBM은 2분할로 충분했으나, 딥러닝은 에폭 단위 검증이 필요하여 3분할로 수행.
            */

            Console.WriteLine();
            Console.WriteLine($"---------- 3분할 결과 ----------");
            Console.WriteLine($"  train : {mlContext.Data.CreateEnumerable<CastingImageData>(trainView, false).Count()}장 (학습)");
            Console.WriteLine($"  val   : {mlContext.Data.CreateEnumerable<CastingImageData>(valView, false).Count()}장 (에폭별 검증)");
            Console.WriteLine($"  test  : {testData.Count}장 (최종 평가 1회)");

            // 파이프라인 구성
            Directory.CreateDirectory(WorkspacePath); // 트레이너가 캐시를 쓸 폴더를 미리 생성
            
            // 전처리 : 라벨 문자열을 Key로 ,이미지 경로를 바이트로
            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName : "LabelKey", inputColumnName : nameof(CastingImageData.Label))
                                                    .Append(mlContext.Transforms.LoadRawImageBytes(
                                                        outputColumnName : "ImageBytes",
                                                        imageFolder : null, // ImagePath가 이미 절대경로이므로 기준 폴더 불필요
                                                        inputColumnName : nameof(CastingImageData.ImagePath)
                                                    ));
            /*
            MapValueToKey : 문자열 정답 라벨("defect" / "ok")을 내부 Key 타입으로 변환하는 메서드
            - 분류 알고리즘은 문자열을 직접 다루지 못하므로, 이 변환이 반드시 필요
            - Key에 매겨지는 번호는 데이터에 먼저 등장한 순서.
            - 위 이유로 인해 Score 배열의 인덱스 의미를 추측해서는 안되며, 스키마에서 확인해야 함

            LoadRawImageBytes : 파일 경로를 읽어 원본 바이트배열로 만드는 메서드
            - 디코딩, 리사이즈 , 정규화는 ImageClassificationTrainer가 내부에서 알아서 처리함(사전학습 모델이 요구하는 입력 크기로 자동 변환)
            - 위 이유로 인해 LightGBM의 Concatenate("Features", ...)에 해당하는 단계가 없음. CNN이 픽셀에서 피처를 직접 학습하므로 ,사람이 피처를 조립할 필요 X
            */

            var preprocessingTransformer = preprocessingPipeline.Fit(trainAllView);// MapValueToKey는 "학습이 필요한" 변환기임.
            IDataView trainTransformed = preprocessingTransformer.Transform(trainView);
            IDataView valTransformed = preprocessingTransformer.Transform(valView);
             /*
            왜 ValidationSet은 직접 Transform 해야 하는가?
            - EstimatorChain.Fit(trainView)는 넘겨받은 trainView에만 변환을 순차 적용한다.
            - 하지만 options.ValidationSet은 체인 바깥에서 트레이너에게 직접 꽂히는 IDataView다. 체인이 "아, 이것도 변환해줘야겠네"라고 알아서 처리해주지 않는다.
            - 그래서 원본 valView를 넘기면 트레이너가 LabelKey 컬럼을 찾지 못해 ArgumentOutOfRangeException: Column 'LabelKey' not found 가 난다.
            */




            // 트레이너 옵션
            var options = new ImageClassificationTrainer.Options
            {
                FeatureColumnName = "ImageBytes", // LoadRawImageBytes의 출력 컬럼명과 일치해야 함
                LabelColumnName = "LabelKey", // MapValueToKey의 출력 컬럼명과 일치해야 함
                Arch = ImageClassificationTrainer.Architecture.MobilenetV2,
                Epoch = Epochs,
                BatchSize = BatchSize,
                LearningRate = 0.01f,
                ValidationSet = valTransformed, 
                MetricsCallback = metrics => System.Console.WriteLine($"    {metrics}"),
                WorkspacePath = WorkspacePath, // 백본이 추출한 피처를 .tsv로 캐싱하는 폴더.
                ReuseTrainSetBottleneckCachedValues = true,
                ReuseValidationSetBottleneckCachedValues = true,
                EarlyStoppingCriteria = new ImageClassificationTrainer.EarlyStopping(
                    minDelta : 0.001f, // 이보다 작은 개선은 "개선 없음"으로 본다.
                    patience : 5, // 5에폭 연속 개선이없으면 중단
                    metric : ImageClassificationTrainer.EarlyStoppingMetric.Accuracy
                )
            };
            /*
            Arch(백본(사전학습 모델) 종류)
            - MobilenetV2 : 가볍고 빠름. CPU 학습에 적합
            - ResnetV2101 : 더 정확할 수는 있으나, cpu에서는 에폭 당 수 분 소요, 첫 실행 시 사전학습 모델을 자동 다운로드 함.
            - InceptionV3
            - ResnetV250
            MetricsCallback을 반드시 설정해야 하는 이유
            - 이것이 없으면 학습 중 아무 출력이 없어 수십 분간 멈춘 것처럼 보인다. (에폭별 train/val 정확도를 보려면 필수)
            - LightGBM에서 학습 R²와 검증 R²의 격차로 과적합을 판별했던 것처럼, 여기서도 train 정확도와 val 정확도의 격차를 에폭별로 관찰한다.

            전이학습은 백본을 1회만 통과시켜 피처를 뽑고, 그 피처로 마지막 층만 반복 학습.

            병목 캐시 재사용
            - true로 두면 두 번째 실행부터 피처 추출 단계를 건너뛰어 훨씬 빠름
            - 데이터가 바뀌었는데 true로 두면 옛 캐시를 재사용해 사용하여 잘못된 결과가 나올 수 있음에 주의
            - 데이터나 Arch를 바꿀 때에는 WorkspacePath폴더를 비우거나 false로 두어야 함

            조기종료(EarlyStoppingCriteria)
            - 검증 정확도가 더 오르지 않으면 학습을 멈춘다. 
            */

            var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(options).Append(mlContext.Transforms.Conversion.MapKeyToValue(
                outputColumnName : "PredictedLabel",
                inputColumnName : "PredictedLabel"
            ));
            /*
            이진 분류인데 왜 MulticlassClassification인가
            - ImageClassificationTrainer는 다중분류 카탈로그에만 존재한다.
            - 클래스가 2개면 이진 분류와 수학적으로 동일하게 동작한다.
            - 다만 평가 지표가 BinaryClassification이 아니라 Multiclass 쪽(MicroAccuracy 등)으로 나온다.
                      -> Precision/Recall은 혼동행렬에서 직접 계산한다(아래 EvaluateWithConfusionMatrix).
            */

            // 학습
            System.Console.WriteLine();
            System.Console.WriteLine($"==========학습 시작 (Arch = {options.Arch}, Epoch = {Epochs}, Batch = {BatchSize})==========");
            System.Console.WriteLine("  첫 실행은 사전학습 모델 다운로드 + 피처 추출로 오래걸림");

            var stopwatch = Stopwatch.StartNew(); //학습시간 측정
            var trainedModel = trainingPipeline.Fit(trainTransformed);

            stopwatch.Stop();
            System.Console.WriteLine($"==========학습 완로 : {stopwatch.Elapsed.TotalMinutes:F1}분 소요");

            // 평가
            PrintClassOrder(mlContext, trainedModel,  preprocessingTransformer.Transform(testView)); //Score 배열의 인덱스 의미를 먼저 확인
            EvaluateWithConfusionMatrix(mlContext, trainedModel, trainTransformed, "train (과적합 확인용)");
            EvaluateWithConfusionMatrix(mlContext, trainedModel, preprocessingTransformer.Transform(testView), "test (최종 성능)");

            //전처리 변환기와 학습된 모델을 하나로 합친다,.
            var fullModel = preprocessingTransformer.Append(trainedModel); // PredictionEngine<CastingImageData, CastingPrediction>은 원본 DTO를 입력으로 받는데, 전처리가 없는 trainedModel에 넣으면 ImageBytes 컬럼이 없어서 오류가 남. 


            // 오분류 분석
            AnalyzeMisclassified(mlContext, fullModel, testData);

            //모델 저장
            Directory.CreateDirectory(Path.GetDirectoryName(ModelSavePath)!);
            mlContext.Model.Save(fullModel, trainView.Schema, ModelSavePath);

            System.Console.WriteLine(); 
            Console.WriteLine($"모델 저장 완료 : {ModelSavePath}");

            Console.WriteLine();
            Console.WriteLine("========== [주조품 양/불량 분류] 완료 ==========");

        }

        private void PrintClassOrder(MLContext mlContext, ITransformer model, IDataView data) // Score 배열의 인덱스가 어느 클래스인지 확인하는 메서드
        {
            // MapValueToKey가 매기는 Key 번호는 데이터에 등장한 순서임(알파벳 순 x)
            // 따라서 Score[0]이 defect인지 ok인지 추측할 수 없음
            // 스키마이 SlotNames 주석에 실제 순서가 들어있으므로 그것을 읽는다.           

            var transformed = model.Transform(data);
            var scoreColumn = transformed.Schema["Score"];

            VBuffer<ReadOnlyMemory<char>> slotNames = default; // ML.NET에서 피처 데이터나 예측 결과 벡터를 다룰 때 내부적으로 VBuffer<ReadOnlyMemory<char>>나 VBuffer<float> 형태로 데이터를 주고받음.  
            scoreColumn.Annotations.GetValue("SlotNames", ref slotNames); // Score 각 칸의 이름(=클래스명)을 꺼낸다.

            Console.WriteLine();
            Console.WriteLine("---------- Score 배열 인덱스 <-> 클래스 대응 ----------");

            var names = slotNames.DenseValues().Select(x => x.ToString()).ToArray();

            for (int i = 0; i < names.Length; i++)
            {
                Console.WriteLine($"  Score[{i}] = \"{names[i]}\"");
            }
        }

        private void EvaluateWithConfusionMatrix(MLContext mLContext, ITransformer model, IDataView data, string title) // 평가 + 혼동행렬 + Precision/Recall 집계 메서드
        {
            var predictions = model.Transform(data);

            var metrics = mLContext.MulticlassClassification.Evaluate(
                predictions, labelColumnName : "LabelKey", predictedLabelColumnName : "PredictedLabel"
            ); 
            /*
            MultiClassClassification.Evaluate()
                - labelColumnName에 LabelKey(Key타입)을 넘겨야 함. 원본 Label(문자열)을 넘기면 타입 에러
                - MicroAccuracy : 전체 샘플 기준 정확도. 다수 클래스에 유리하게 나옴.
                - MacroAccuracy : 클래스별 정확도의 평균. 불균형 데이터에서 더 정직한 지표임. 
                - LogLoss : 확률 예측의 품질. 낮을 수록 좋음.
            */
            Console.WriteLine();
            Console.WriteLine($"---------- 평가 : {title} ----------");
            Console.WriteLine($"  MicroAccuracy : {metrics.MicroAccuracy,8:P2}   (전체 기준, 다수 클래스에 유리)");
            Console.WriteLine($"  MacroAccuracy : {metrics.MacroAccuracy,8:P2}   (클래스별 평균, 불균형에 정직)");
            Console.WriteLine($"  LogLoss       : {metrics.LogLoss,8:F4}   (낮을수록 좋음)");

            // 혼동 행렬 집계
            //metrics.ConfusionMatrix도 있으나, 어느 칸이 어느 클래스인지 혼동되기 쉬움. 품질검사에서는 FN(불량유출)을 정확히 세는 것이 가장 중요하므로, 문자열 라벨로 직접 집계하여 명확하게 만든다.
            var rows = mLContext.Data.CreateEnumerable<CastingPredictionWithLabel>(predictions, reuseRowObject : false).ToList();

            int tp = rows.Count(r => r.Label == CastingImageLoader.LabelDefect && r.PredictedLabel == CastingImageLoader.LabelDefect); // 불량을 불량으로
            int fn = rows.Count(r => r.Label == CastingImageLoader.LabelDefect && r.PredictedLabel == CastingImageLoader.LabelOk); // 불량을 정상으로
            int fp = rows.Count(r => r.Label == CastingImageLoader.LabelOk && r.PredictedLabel == CastingImageLoader.LabelDefect); // 정상을 불량으로
            int tn = rows.Count(r => r.Label == CastingImageLoader.LabelOk && r.PredictedLabel == CastingImageLoader.LabelOk); // 정상을 정상으로
            // 품질검사의 주 목적은 불량을 찾아내는 것이므로, 불량을 Positive로 잡아야 함.
            
            Console.WriteLine();
            Console.WriteLine("  [혼동행렬]            예측 : 불량   예측 : 정상");
            Console.WriteLine($"    실제 : 불량          {tp,8} (TP) {fn,7} (FN) ★불량유출");
            Console.WriteLine($"    실제 : 정상          {fp,8} (FP) {tn,7} (TN)");

            // precision, recall, f1 계산
            double precision = tp + fp > 0 ? (double)tp / (tp + fp) : 0; // 불량이라 예측한 것 중 실제 불량의 비율
            double recall = tp + fn > 0 ? (double) tp / (tp + fn) : 0 ; // 실제 불량을 찾아낸 비율

            double f1 = precision + recall > 0 ?  2* precision*recall / (precision + recall) : 0;
            double accuracy = (double) (tp+tn) / rows.Count;

            Console.WriteLine();
            Console.WriteLine($"  Accuracy          : {accuracy,8:P2}");
            Console.WriteLine($"  Precision(불량)   : {precision,8:P2}   (불량 판정 중 실제 불량 비율)");
            Console.WriteLine($"  Recall(불량)      : {recall,8:P2}   ★(실제 불량 중 검출 비율)");
            Console.WriteLine($"  F1(불량)          : {f1,8:P2}");

        }

        private void AnalyzeMisclassified(MLContext mLContext, ITransformer model, List<CastingImageData> testData) // 오분류 사례 분석 메서드
        {
            System.Console.WriteLine();
            System.Console.WriteLine("---------- 오분류 분석 (FN)----------");

            var engine = mLContext.Model.CreatePredictionEngine<CastingImageData, CastingPrediction>(model); // PredictionEngine은 단일샘플 예측용. Thread-Safety하지 않지만, 위 파이프라인의 순차 루프에서는 괜찮음.
            // predictionEngine은 사용이 필요한 곳에서 모두 인스턴스를 만들어야 함. 여러 스레드에서 단일 PredictionEngine의 인스턴스의 Predict()를 호출하는 경우라면 위험하므로, PredictionEnginePool을 사용함

            var falseNegatives = new List<(string path, float[] score)>();
            var falsePositive = new List<(string path, float[] score)>();

            foreach(var sample in testData)
            {
                var p = engine.Predict(sample);

                if(sample.Label == CastingImageLoader.LabelDefect && p.PredictedLabel == CastingImageLoader.LabelOk) // 실제 이미지가 불량 && AI 예측 결과는 정상일 떄
                {
                    falseNegatives.Add((sample.ImagePath, p.Score)); // 불량 이미지를 보고 정상이라 예측함 -> 유출
                }
                else if (sample.Label == CastingImageLoader.LabelOk && p.PredictedLabel == CastingImageLoader.LabelDefect) // 실제 이미지가 정상 && ai 예측 결과는 불량일 때
                {
                    falsePositive.Add((sample.ImagePath, p.Score));
                    // 정상품을 불량이라고 예측 -> 과잉 폐기
                }
            }

            System.Console.WriteLine($" FN(불량 유출, 불량 이미지를 정상이라고 예측) : {falseNegatives.Count}건");
            foreach(var (path, score) in falseNegatives.Take(10))
            {
                System.Console.WriteLine($" {Path.GetFileName(path), -28} Score = [{string.Join(", ", score.Select(s => s.ToString("F3")))}]"); // 확률을 함께 출력 -> 0.49처럼 간신히 틀린 케이스인지, 0.05처럼 확인하며 틀린 케이스이지 구분. 전자면 임계값 조정으로 잡을 수 있는데 ,후자는 모델이 근본적으로 놓친 것.
            }

            System.Console.WriteLine($" FP(과잉 폐기, 정상 이미지를 불량이라고 예측) : {falsePositive.Count}건");
            foreach(var (path, score) in falsePositive.Take(5))
            {
                System.Console.WriteLine($" {Path.GetFileName(path), -28} Score = [{string.Join(", ", score.Select(s => s.ToString("F3")))}]");
            }

            Console.WriteLine();
            Console.WriteLine("  * FN 이미지를 직접 열어보면 결함이 작거나 애매한 경우가 많습니다.");
            Console.WriteLine("    이것이 '모델의 한계가 어디인가'를 체감하는 가장 좋은 방법입니다.");

        }

        private class CastingPredictionWithLabel
        {
            public string Label {get;set;} = string.Empty; // 원본 문자열 라벨
            public string PredictedLabel {get;set;} = string.Empty; // MapKeyToValue로 되돌린 예측 라벨
            /*
            CastingPrediction과 따로 두는 이유
            - CastingPrediction 타입은 PredictionEngine의 출력용으로, 입력에 정답이 없음.
            - 혼동행렬은 "정답과 예측을 한 행에서" 읽어야 하므로, Label 컬럼이 추가로 필요.
            - CreateEnumerable은 이름이 일치하는 컬럼만 채우므로, 필요한 두 개만 선언하면 됨.
            */


        } // 정답과 예측을 함께 읽기 위한 혼동행렬 집계용 내부 DTO
    }
}