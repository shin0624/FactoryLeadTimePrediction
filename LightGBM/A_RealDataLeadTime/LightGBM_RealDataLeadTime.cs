using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Data;

namespace FactoryLeadTimePrediction.LightGBM.A_RealDataLeadTime
{
    public class LightGBM_RealDataLeadTime // 실데이터(KAMP 표면처리 공급망최적화 데이터셋) 기반 리드타임 예측 실습의 메인 실행 클래스.
    {
        private static LightGBM_RealDataLeadTime? _instance; // static 필드는 CLR이 반드시 null로 초기화하므로 '?'가 사실에 맞는 선언. '?'는 컴파일 타임 정적 분석용 힌트일 뿐 런타임 동작/성능에 영향이 없다.
        public static LightGBM_RealDataLeadTime Instance() // 반환형은 non-nullable : "Instance()를 부르면 절대 null이 아닌 것을 받는다"는 계약을 표현한다.
        {
            if(_instance == null)
            {
                _instance = new LightGBM_RealDataLeadTime();
            }
            return _instance;
        }

        private const string CsvPath = "data/Supply_Chain_Optimization/data/DataSet.csv"; // 원본 데이터 경로 (프로젝트 루트 기준 상대경로)

        public void StartRealDataLeadTime() // 1단계 : CSV 전처리 -> Feature 변환 -> 검산 -> IDataView 적재
        {
            var mlContext = new MLContext(seed:42); // mlcontext 초기화. seed를 고정하면 실행마다 동일한 결과가 나온다.

            Console.WriteLine("========== [1단계] 실데이터 CSV 전처리 시작 ==========");

            // ---------- (1) CSV 파싱 : CSV -> List<RawProductionRecord> ----------
            string fullPath = ResolveCsvPath(CsvPath);
            var loader = new CsvLoader();
            var rawRecords = loader.LoadRawProductionRecord(fullPath);

            // ---------- (2) 피처 엔지니어링 : List<RawProductionRecord> -> List<LeadTimeFeature> ----------
            var features = loader.TransformToFeatures(rawRecords);

            // ---------- (3) 검산 출력 ----------
            PrintValidationSummary(rawRecords, features);

            // ---------- (4) IDataView 적재 및 스키마 확인 ----------
            IDataView dataView = mlContext.Data.LoadFromEnumerable(features);
            // LoadFromEnumerable()은 List<T>를 ML.NET 내부 데이터 형식인 IDataView로 변환한다.
            // [LoadColumn] 없이도 프로퍼티 이름이 그대로 컬럼명이 되고, [ColumnName("Label")]이 붙은 프로퍼티는 "Label" 컬럼이 된다.

            PrintSchema(dataView);

            Console.WriteLine("========== [1단계] 완료 : 학습(Fit)은 2단계에서 진행 ==========");
        }

        public void StartRealDataTraining() // 2단계(학습->평가->기여도  분석)를 실행하는 메서드
        {
            var mlContext = new MLContext(seed : 42);

            Console.WriteLine("========== [2단계] 실데이터 LightGBM 학습 시작 ==========");

            // 1단계와 동일한 전처리
            string fullPath = ResolveCsvPath(CsvPath);
            var loader = new CsvLoader();
            var rawRecords = loader.LoadRawProductionRecord(fullPath);
            var features = loader.TransformToFeatures(rawRecords);

            IDataView allData = mlContext.Data.LoadFromEnumerable(features);

            // 학습/검증 데이터 분할
            var split = mlContext.Data.TrainTestSplit(allData, testFraction : 0.2, seed:42);
            IDataView trainData = split.TrainSet;
            IDataView testData = split.TestSet;

        // TrainTestSplit() : 데이터를 학습용/검증용으로 무작위 분할 하는 메서드
        // testFraction 0.2 : 8대 2로 분할(약 141행 학습, 35행 검증)
        // seed를 고정해야 실행마다 같은 분할률이 적용되어 비교가 가능.

        Console.WriteLine($"학습 데이터 : {mlContext.Data.CreateEnumerable<LeadTimeFeature>(trainData, reuseRowObject: false).Count()}행");
        Console.WriteLine($"검증 데이터 : {mlContext.Data.CreateEnumerable<LeadTimeFeature>(testData, reuseRowObject: false).Count()}행");
        
        
        // 전처리 파이프라인(피처 6개를 Features 벡터로 묶는다.)
        //nameof를 쓰면 프로퍼티 명을 바꾸어도 에러로 간주되지 않으므로, 문자열 직접 입력보다 안전.
        var dataProcessPipeline = mlContext.Transforms.Concatenate("Features",
            nameof(LeadTimeFeature.ProductionKg),
            nameof(LeadTimeFeature.ProcessMinutes),
            nameof(LeadTimeFeature.WeekdayIndex),
            nameof(LeadTimeFeature.Month),
            nameof(LeadTimeFeature.SameDayOrderCount),
            nameof(LeadTimeFeature.StartHour)
        ); // Label은 피처가 아니라 예측 대상이므로 Concatenate에 넣지 않음.

        // LightGBM 트레이너 설정
        var options = new LightGbmRegressionTrainer.Options
        {
            NumberOfLeaves = 16, // 트리 노드 갯수
            MinimumExampleCountPerLeaf = 15,//노드당 최소 데이터 수. 노드 당 15개의 행이 있어야 분할이 허용된다.(최소 15건의 공통 패턴을 학습한다.) 더미데이터는 2였으나, 실데이터(176행)에서 2행만 있어도 분할을 허용하면 트리가 개별 행을 외워버리는 과적합 발생 가능
            // 이 값을 2, 5, 15, 30으로 바꿔가며 RMSE를 비교하는 것이 하이퍼파라미터 튜닝 연습이 된다.
            LearningRate = 0.05, // 학습률. 낮추면 학습 속도가 늘어나지만 ,일반화 성능이 향상됨
            NumberOfIterations = 200,//부스팅 방복 횟수
            Booster = new GradientBooster.Options{L2Regularization = 1.0}, // l2 규제
            FeatureColumnName = "Features",
            LabelColumnName = "Label"
        };

        var trainer = mlContext.Regression.Trainers.LightGbm(options);
        var trainingPipeline = dataProcessPipeline.Append(trainer);

        //학습
        System.Console.WriteLine("========= 학습 중 =========");
        var trainedModel = trainingPipeline.Fit(trainData);
        System.Console.WriteLine("========= 학습 완료 =========");

        // 평가
        EvaluateModel(mlContext, trainedModel, trainData, "학습 데이터(과적합 확인용)");
        EvaluateModel(mlContext, trainedModel, testData, "검증 데이터(실 성능)");

        // 교차검증
        RunCrossValidation(mlContext, trainingPipeline, allData);

        // 기여도 분석
        AnalyzeContribution(mlContext, trainedModel, trainData, features);

        Console.WriteLine("========== [2단계] 완료 ==========");
        }

        private void EvaluateModel(MLContext mlContext, ITransformer model, IDataView data, string label )// 평가 메서드
        {
            var predictions = model.Transform(data);// 예측 수행(Score 컬럼이 추가된 IDataView를 반환)
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName : "Label", scoreColumnName : "Score");
            
            // RegressionCatalog.Evaluate(IDataview, string labelColumName, string scoreColumnName) : 회귀모델의 성능 지표를 계산하는 메서드.
            // labelColumnName 기본 값이 "Label"이므로, LeadTimeFeature에서 [ColumnName("Label")]의 이름과 동일한 지 확인해야 함
            //산출값
            // -  RSquared : 1에 가까울 수록 좋음. 0이면 평균 찍기와 동일, 음수이면 평균보다 못함
            // - RootMeanSquaredError(RMSE) : 예측 오차의 크기(DAY). 작을수록 좋음. 기준일 (9.9일)과 비교해야 할 값
            // - MeanAbsoluteError(MAE) : 평균 절대오차(DAY). 이상치에 덜 민감함.

            Console.WriteLine();
            Console.WriteLine($"---------- 평가 : {label} ----------");
            Console.WriteLine($"  R²   (1에 가까울수록 좋음) : {metrics.RSquared,8:F4}");
            Console.WriteLine($"  RMSE (작을수록 좋음)       : {metrics.RootMeanSquaredError,8:F2}일");
            Console.WriteLine($"  MAE  (작을수록 좋음)       : {metrics.MeanAbsoluteError,8:F2}일");
            Console.WriteLine($"  * 베이스라인(평균값으로 예측) RMSE = 9.9일 / MAE = 7.39일");

            if(metrics.RootMeanSquaredError > 9.9)
            {
                Console.WriteLine("  -> [주의] 기준일(9.9일)보다 RMSE가 큽니다. 즉 항상 평균값으로 예측하는 것보다 못합니다.");
            }
        }  

        private void RunCrossValidation(MLContext mLContext, IEstimator<ITransformer> pipeline, IDataView allData) // 교차검증 메서드
        {
            Console.WriteLine();
            Console.WriteLine("---------- 5-Fold 교차검증 ----------");
            /*
            교차검증을 쓰는 이유
            - 원본데이터 176행에서 20%(35행)만 검증에 쓰면 어느 35행이 뽑혔는지에 따라 결과가 크게 변함.
            - k-fold는 데이터를 k등분해서 각 조각을 한 번씩 검증용으로 쓰고 k번 학습함.
            즉, 모든 행이 한 번씩 검증에 쓰이므로 결과가 훨씬 안정적임. 소규모 데이터에서는 단일 분할보다 교차검증이 거의 항상 나음.
            */

            var cvResults = mLContext.Regression.CrossValidate(allData, pipeline, numberOfFolds : 5, labelColumnName : "Label" );

            foreach(var (result, idx) in cvResults.Select((r, i) => (r, i+1)))
            {
                Console.WriteLine($" Fold{idx} : R²  = {result.Metrics.RSquared, 7:F4} / RMSE = {result.Metrics.RootMeanSquaredError, 6:F2}일");
                
                double avgR2 = cvResults.Average(r=>r.Metrics.RSquared);
                double avgRmse = cvResults.Average(r => r.Metrics.RootMeanSquaredError);
                double stdRmse = Math.Sqrt(cvResults.Average(r => Math.Pow(r.Metrics.RootMeanSquaredError - avgRmse, 2)));

            Console.WriteLine($"  ----------------------------------------");
            Console.WriteLine($"  평균 R²   : {avgR2,7:F4}");
            Console.WriteLine($"  평균 RMSE : {avgRmse,7:F2}일 (표준편차 {stdRmse:F2})");
            Console.WriteLine($"  * 표준편차가 크면 폴드마다 성능이 들쭉날쭉하다는 뜻 = 데이터가 부족하거나 패턴이 불안정하다는 신호");
            }

        }

        private void AnalyzeContribution(MLContext mLContext, ITransformer trainedModel, IDataView trainData, List<LeadTimeFeature> features)// 기여도 분석 메서드. 
        {
            Console.WriteLine();
            Console.WriteLine("---------- 기여도 분석 ----------");

            var chain = (TransformerChain<RegressionPredictionTransformer<LightGbmRegressionModelParameters>>)trainedModel;
            // trainedModel은 TransformerChain이므로, LastTransformer로 예측기만 꺼낸다.
            //CalculateFeatureContribution()은 ISingleFeaturePredictionTransformer<TModel>을 요구하므로 전체 체인을 넘길 수 없음.

            var transformedData = chain.Transform(trainData);// 기여도 계산기를 Fit()하려면 Feature 컬럼이 있는 데이터가 필요
            
            var contributionCalc = mLContext.Transforms.CalculateFeatureContribution(chain.LastTransformer, normalize:false).Fit(transformedData);// normalize : false => true이면 -1~1로 정규화되어 비중 비교용이 됨.

            var scoringPipeline = chain.Append(contributionCalc);// 학습된 모델 체인에 기여도 계산기 넣음
            var engine = mLContext.Model.CreatePredictionEngine<LeadTimeFeature, A_LeadTimePrediction>(scoringPipeline);// 예측 엔진 생성

            string[] featureNames = {"생산량(kg)", "공정시간(분)", "요일", "월", "같은날 건수", "시작 시각"}; // 기여도 배열의 순서는 Concatenate에 넣은 순서와 같음

            foreach(var sample in features.Take(3))// 검증용으로 실제데이터 3건을 뽑아 예측 근거를 출력
            {
                var prediction = engine.Predict(sample);

                Console.WriteLine();
                Console.WriteLine($"  [입력] 생산량 {sample.ProductionKg:F0}Kg / 공정 {sample.ProcessMinutes:F0}분 / 요일 {sample.WeekdayIndex:F0} / {sample.Month:F0}월 / 같은날 {sample.SameDayOrderCount:F0}건 / {sample.StartHour:F0}시");
                Console.WriteLine($"  [실제] {sample.LeadTimeDays:F0}일   [예측] {prediction.PredictedLeadTime:F2}일   [오차] {Math.Abs(sample.LeadTimeDays - prediction.PredictedLeadTime):F2}일");

                if(prediction.FeaturesContributions!=null)
                {
                    //기여도 합 + 기준값(bias) = Score 가 성립. 검산해보면 기여도를 제대로 읽고 있는지 확인 가능
                    float sum = prediction.FeaturesContributions.Sum();
                    Console.WriteLine($"  [근거] 기준값 {prediction.PredictedLeadTime - sum:F2}일 에서 출발");

                    for(int i=0; i<featureNames.Length && i<prediction.FeaturesContributions.Length; i++)
                    {
                        float c = prediction.FeaturesContributions[i];
                        string arrow = c > 0 ? "▲ 늘림" : (c < 0 ? "▼ 줄임" : "- 미사용");
                        Console.WriteLine($"         {featureNames[i],-12} : {c,+7:F2}일  {arrow}");
                    }

                }
            }
        }

        

        private string ResolveCsvPath(string relativePath) // 실행 위치(bin/Debug/net8.0)와 프로젝트 루트가 다르므로 상위로 올라가며 데이터 파일을 찾는다.
        {
            if(File.Exists(relativePath))
            {
                return relativePath;
            }

            // dotnet run 시 작업 디렉터리가 프로젝트 루트가 아닐 수 있어, 최대 5단계까지 상위 폴더를 탐색한다.
            var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
            for(int i=0; i<5 && dir != null; i++)
            {
                string candidate = Path.Combine(dir.FullName, relativePath);
                if(File.Exists(candidate))
                {
                    return candidate;
                }
                dir = dir.Parent;
            }

            return relativePath; // 못 찾으면 원래 경로를 반환하고, CsvLoader가 FileNotFoundException을 던지도록 둔다.
        }

        private void PrintValidationSummary(List<RawProductionRecord> raws, List<LeadTimeFeature> features)
        {
            /*
            검산의 목적
                파싱이 조용히 실패해도 프로그램은 돌아간다. 특히 따옴표 분할이 틀리면 생산량이 0이 되는데,
                학습까지 다 돌려놓고 나서야 이상함을 알게 된다. 그래서 숫자를 먼저 눈으로 확인한다.

            기대값 (사전에 데이터를 실측한 값)
                원본 행 수      : 177
                유효 행 수      : 176   (1차 출고날짜 결측 1건 제외)
                리드타임 평균    : 19.4일  (min 2 / max 55)
                생산량 평균      : 14056.3
                공정시간 평균    : 193.6분
            -> 생산량 평균이 14,056이 아니면(특히 0이면) 따옴표 분할이 실패한 것이다.
            */
            Console.WriteLine();
            Console.WriteLine("---------- 검산 결과 (괄호 안은 기대값) ----------");
            Console.WriteLine($"원본 행 수       : {raws.Count,8}      (177)");
            Console.WriteLine($"유효 행 수       : {features.Count,8}      (176)");

            if(features.Count == 0)
            {
                Console.WriteLine("[오류] 유효 행이 0개입니다. 파싱 로직을 확인하세요.");
                return;
            }

            var lead = features.Select(f => f.LeadTimeDays).ToList();
            var kg = features.Select(f => f.ProductionKg).ToList();
            var min = features.Select(f => f.ProcessMinutes).ToList();

            Console.WriteLine($"리드타임 평균     : {lead.Average(),8:F1}일    (19.4)");
            Console.WriteLine($"리드타임 min/max : {lead.Min(),8:F0} / {lead.Max():F0}   (2 / 55)");
            Console.WriteLine($"생산량 평균       : {kg.Average(),8:F1}      (14056.3)");
            Console.WriteLine($"공정시간 평균     : {min.Average(),8:F1}분    (193.6)");
            Console.WriteLine($"제품코드 고유값   : {raws.Select(r => r.ItemCode).Distinct().Count(),8}      (1 = M21 뿐)");

            // 리드타임이 음수인 행은 데이터 오류(출고가 생산보다 빠름)를 뜻한다. 실측상 0건이어야 한다.
            int negative = lead.Count(v => v < 0);
            Console.WriteLine($"리드타임 음수 행  : {negative,8}      (0)");

            // 베이스라인 : "항상 평균값으로 예측"했을 때의 RMSE. 3단계에서 모델이 이 값을 이겨야 의미가 있다.
            float mean = lead.Average();
            double rmse = Math.Sqrt(lead.Average(v => Math.Pow(v - mean, 2)));
            Console.WriteLine($"베이스라인 RMSE   : {rmse,8:F1}일    (9.9)  <- 3단계에서 모델이 이겨야 하는 기준");
            Console.WriteLine();
        }

        private void PrintSchema(IDataView dataView) // IDataView에 의도한 컬럼이 전부 Single(float)로 들어갔는지 눈으로 확인하는 단계.
        {
            Console.WriteLine("---------- IDataView 스키마 ----------");
            foreach(var col in dataView.Schema)
            {
                Console.WriteLine($"  [{col.Index}] {col.Name,-20} : {col.Type}");
            }
            Console.WriteLine();
        }
    }
}
