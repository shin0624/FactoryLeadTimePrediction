using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

namespace FactoryLeadTimePrediction.LightGBM.B_SyntheticLeadTime // 합성 데이터의 싱글톤 메인 스크립트(학습 -> 평가 -> 기여도 분석 -> 행수 실험)
{
    public class LightGBM_SyntheticLeadTime
    {
        private static LightGBM_SyntheticLeadTime? _instance; // static 필드는 CLR이 null로 초기화하므로 '?'가 사실에 맞는 선언. '?'는 컴파일타임 힌트일 뿐 런타임 동작/성능 영향 없음.
        public static LightGBM_SyntheticLeadTime Instance()// 반환형은 non-nullable : "Instance()를 부르면 null이 아닌 것을 받는다"는 계약.
        {
            if(_instance == null)
            {
                _instance = new LightGBM_SyntheticLeadTime();
            }
            return _instance;
        }

        private const int DefaultRowCount = 176; // 실데이터와 동일한 행 수.
        private const float DefaultSigma = 3.0f; // 기본 노이즈. 이때 RSquared 상한은 0.877
        private const int Seed = 42; // 난수 생성 시드

        // 기여도 배열 / 정답 계수와 순서를 맞추기 위한 피처 명.
        private static readonly string[] FeatureNames = {"납기여유일", "거래처재고", "운송빈도", "생산량(kg)", "공정시간(분)", "요일"};

        public void StartSyntheticLeadTime() // 합성 데이터 실습 전체를 실행하는 메서드
        {
            Console.WriteLine("========== [합성데이터] 파이프라인 검증 시작 ==========");

            RunSingleExperiment(DefaultRowCount, DefaultSigma); // 기본 조건(176행, sigma 3.0)으로 학습 + 평가 + 기여도 분석 수행
            RunRowCountExperiment();// 행 수를 바꾸어가며 과적합 변화 관찰
            RunSigmaExperiment(); // 노이즈 크기를 바꾸어가며 RSquared 상한 변화 관찰

            Console.WriteLine("========== [합성데이터] 파이프라인 검증 종료 ==========");
        }

        private void RunSingleExperiment(int rowCount, float sigma) // 단일 조건 실험 메서드 : 학습 -> 평가 -> 교차검증 -> 기여도 분석 
        {
            var mlContext = new MLContext(seed : Seed); // seed 고정 시 TrainTestSplit, LightGBM 내부 난수가 고정되어 실행마다 같은 결과가 나온다.

            Console.WriteLine();
            Console.WriteLine($"------------기본 실험 : {rowCount}행 / sigma={sigma}--------------");

            // 데이터 생성
            var generator = new SyntheticDataGenerator();
            var data = generator.Generate(rowCount, sigma, Seed);

            IDataView allData = mlContext.Data.LoadFromEnumerable(data); // LoadFromEnumerable()을 사용하여 List<T>를 ML.NET 내부 형식인 IDataView로 변환. [LoadColumn] 없이 프로퍼티 명이 그대로 컬럼 명이 된다.

            // 학습셋 / 검증셋 분할
            var split = mlContext.Data.TrainTestSplit(allData, testFraction : 0.2, seed : Seed); // 학습:검증 = 8:2로 분할. 

            // 파이프라인구성
            var pipeline = BuildPipeline(mlContext);

            //학습
            Console.WriteLine("---------- 학습 중 ----------");
            var model = pipeline.Fit(split.TrainSet);
            Console.WriteLine("---------- 학습 완료 ----------");

            // 평가
            var trainMetrics = Evaluate(mlContext, model, split.TrainSet, "학습 데이터");
            var testMetrics = Evaluate(mlContext, model, split.TestSet, "검증 데이터");

            // 검증 체크리스트 1, 2번을 코드로 자동 판정
            Console.WriteLine();
            Console.WriteLine("---------- 파이프라인 검증 판정 ----------");
            Console.WriteLine($"  [기준1] 검증 R² >= 0.8  : {(testMetrics.RSquared >= 0.8 ? "통과" : "실패")} (실측 {testMetrics.RSquared:F4})");
            Console.WriteLine($"  [기준2] 학습 R² > 검증 R² : {(trainMetrics.RSquared > testMetrics.RSquared ? "통과(정상 과적합)" : "이상")}");
            Console.WriteLine($"  [기준3] 검증 R² <= 0.877 : {(testMetrics.RSquared <= 0.877 ? "통과" : "이상(누출 의심)")} <- 이론 상한 초과 여부");
            // 기준3이 중요한 이유 : 이론 상한을 넘으면 노이즈까지 맞혔다는 뜻 = 정답이 입력에 섞인 것(데이터 누출).

            // 교차검증
            RunCrossValidation(mlContext, pipeline, allData);

            // 기여도 분석
            AnalyzeContribution(mlContext, model, split.TrainSet, data, generator);
        }

        private IEstimator<ITransformer> BuildPipeline(MLContext mlContext) // 전처리 및 트레이너 파이프라인을 만드는 메서드
        {
            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features",
                nameof(SyntheticLeadTimeFeature.CustomerDueDays), // [0] 납기여유일
                nameof(SyntheticLeadTimeFeature.CustomerStockLevel), // [1] 거래처재고
                nameof(SyntheticLeadTimeFeature.ShippingFrequency), // [2] 운송빈도
                nameof(SyntheticLeadTimeFeature.ProductionKg), // [3] 생산량
                nameof(SyntheticLeadTimeFeature.ProcessMinutes), // [4] 공정시간
                nameof(SyntheticLeadTimeFeature.WeekdayIndex) // [5] 요일
            );
            // Concatenate()는 개별 프로퍼티들을 LightGBM이 행렬 연산할 수 있도록 "Features"라는 하나의 float 벡터로 만드는 메서드.
            // 이 순서가 기여도 배열의 인덱스 순서이자 GetTrueCoefficients()의 순서이므로, 동일하게 맞춰야 함
            // Label은 LightGBM의 예측 대상이므로, Concatenate()에 넣지 않는다.

            var options = new LightGbmRegressionTrainer.Options
            {
               NumberOfLeaves = 16,
               MinimumExampleCountPerLeaf = 15, // 노드 하나에 최소 15행
               LearningRate = 0.05,
               NumberOfIterations = 200,
               Booster = new GradientBooster.Options{L2Regularization = 1.0},
               FeatureColumnName = "Features",
               LabelColumnName = "Label"
            }; // 하이퍼 파라미터는 실데이터 사용 시와 동일하게 맞춘다.

            var trainer = mlContext.Regression.Trainers.LightGbm(options); // 하이퍼 파라미터 옵션을 LightGBM 트레이너 객체에 넣는다.
            return dataProcessPipeline.Append(trainer); // 데이터 파이프라인에 트레이너 객체를 추가하여 파이프라인을 리턴
        }

        private RegressionMetrics Evaluate(MLContext mlContext, ITransformer model, IDataView data, string label) // 학습된 모델을 평가하는 메서드
        {
            var predictions = model.Transform(data); // 예측을 수행하여 Score 컬럼이 추가된 IDataView를 반환한다.(지연 평가)

            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName : "Label", scoreColumnName : "Score"); // Regression.Evaluate()는 Label(정답)과 Score(예측)을 비교하여 회귀 지표를 계산하는 메서드.
            Console.WriteLine();
            Console.WriteLine($"---------- 평가 : {label} ----------");
            Console.WriteLine($"  R²   : {metrics.RSquared,8:F4}   (1에 가까울수록 좋음 / 음수 = 평균보다 못함)");
            Console.WriteLine($"  RMSE : {metrics.RootMeanSquaredError,8:F2}일");
            Console.WriteLine($"  MAE  : {metrics.MeanAbsoluteError,8:F2}일");

            return metrics; // 호출부에서 기준 판정에 사용하도록 metrics 객체를 반환
        }

        private void RunCrossValidation(MLContext mlContext, IEstimator<ITransformer> pipeline, IDataView allData) // 교차검증을 수행하는 메서드
        {
            Console.WriteLine();
            Console.WriteLine("---------- 5-Fold 교차검증 ----------");

            var cvResults = mlContext.Regression.CrossValidate(allData, pipeline, numberOfFolds : 5, labelColumnName : "Label"); // 데이터를 5등분하여 각 조각을 한 번씩 검증용으로 사용하고 5번 학습. 모든 행이 한 번씩 검증에 쓰여 결과가 안정적임.

            foreach(var (result, idx) in cvResults.Select((r, i) => (r, i + 1)))
            {
                Console.WriteLine($"    Fold{idx} : R² = {result.Metrics.RSquared, 7:F4} / RMSE = {result.Metrics.RootMeanSquaredError, 6:F2}일");
            }

            double avgR2 = cvResults.Average(r => r.Metrics.RSquared);
            double avgRmse = cvResults.Average(r => r.Metrics.RootMeanSquaredError);
            double stdRmse = Math.Sqrt(cvResults.Average(r => Math.Pow(r.Metrics.RootMeanSquaredError - avgRmse, 2)));//표준편차 : √편차 제곱의 평균 -> 폴드별 성능이 얼마나 흔들리는지 나타냄

            Console.WriteLine($"  ----------------------------------------");
            Console.WriteLine($"  평균 R²   : {avgR2,7:F4}");
            Console.WriteLine($"  평균 RMSE : {avgRmse,7:F2}일 (표준편차 {stdRmse:F2})");
            Console.WriteLine($"  * 실데이터 사용 시 표준편차는 1.24-> 이보다 작으면 패턴 안정적");

        }

        private void AnalyzeContribution(MLContext mlContext, ITransformer model, IDataView trainData, List<SyntheticLeadTimeFeature> data, SyntheticDataGenerator generator)// 기여도 분석 및 정답 대조 메서드
        {
            Console.WriteLine();
            Console.WriteLine("---------- 기여도 분석 (정답 대조) ----------");

            var chain = (TransformerChain<RegressionPredictionTransformer<LightGbmRegressionModelParameters>>)model;
            // 타입 캐스팅을 왜이렇게 길게하나? --> model은 Concatenate() + LightGBM이 결합된 TransformerChain 형태의 객체임. 기여도분석 메서드(CalculateFeatureContribution())는 ISingleFeaturePredictionTransformer<TModel>을 요구하므로 전체체인을 넘길 수 없고, LastTransformer를 통해 예측기 부분만 꺼내야 하기에 구체적인 타입 캐스팅이 필요.

            var transformedData = chain.Transform(trainData); // 기여도 계산기를 Fit() 하려면 "Feature" 컬럼이 이미 만들어진 데이터가 필요함. 원본 trainData를 그대로 기여도 계산기에 넣으면 Features 컬럼이 없어서 에러가 날 수 있음.
            
            var contributionCalc = mlContext.Transforms.CalculateFeatureContribution(chain.LastTransformer, normalize : false).Fit(transformedData);
            // normalize : false -> "며칠 늘렸는지?" 는 절대 수치로 받음. true이면 -1~1로 정규화되어 비중 비교용이 되어버림.
            // 기여도 계산기도 Fit()을 거쳐야 ITransformer 형태가 된다.

            var scoringPipeline = chain.Append(contributionCalc); // 학습된 체인 끝에 기여도 계산기를 붙여 "Score + FeatureContributions"를 모두 출력하는 새 모델을 만든다. 

            var engine = mlContext.Model.CreatePredictionEngine<SyntheticLeadTimeFeature, SyntheticLeadTimePrediction>(scoringPipeline); // 위의 새 모델로 예측 엔진을 생성. 기여도 컬럼이 없는 원본 model로 엔진을 만들면 FetureContributions = null이 된다.
            // CreatePredictionEngine<입력, 출력>(ITransformer형 객체)

            float [] trueCoeff = generator.GetTrueCoefficients(); // 정답 계수 배열 {+0.8, +1.5, -1.5, 0.0, 0.0, 0.0}

            // 샘플 3건의 예측 근거 출력
            var biasList = new List<float>(); // 기준값(바이어스)이 샘플마다 일관되는지 확인하기 위함

            foreach(var sample in data.Take(3))
            {
                var prediction = engine.Predict(sample);
                if(prediction.FeatureContributions == null) continue; // 기여도 계산기가 안 붙었다면 null이 되도록 방어.

                float sum = prediction.FeatureContributions.Sum();
                float bias = prediction.PredictedLeadTime - sum; // Score - 기여도 합 = 기준값 (이론상)
                biasList.Add(bias);

                Console.WriteLine();
                Console.WriteLine($"  [실제] {sample.LeadTimeDays:F2}일   [예측] {prediction.PredictedLeadTime:F2}일   [오차] {Math.Abs(sample.LeadTimeDays - prediction.PredictedLeadTime):F2}일");
                Console.WriteLine($"  [기준값] {bias:F2}일");

                for(int i=0; i<FeatureNames.Length && i < prediction.FeatureContributions.Length; i++)
                {
                    float c = prediction.FeatureContributions[i];
                    string expected = trueCoeff[i] == 0.0f ? "정답 : 무관 (0)" : $"정답 : {trueCoeff[i]:+0.0;-0.0}"; // 정답 계수가 0인 피처는 기여도도 0 근처여야 함.  (c#의 서식지정자 : {값:서식}형태이며, 콜론 앞뒤에 공백이 있으면 부호가 붙어 나오지 않음에 주의)
                    Console.WriteLine($"    {FeatureNames[i],-12} : {c,+8:F2}일   {expected}");
                }
            }

            // 검증 체크리스트 판정
            Console.WriteLine();
            Console.WriteLine("---------- 기여도 검증 판정 ----------");

            // 샘플 전체의 기여도 절대값 평균을 구하여 피처별 영향력 순위를 매긴다.
            // 한 샘플만 보면 우연히 순서가 뒤집힐 수도 있으므로, 여러 건을 평균한다.

            var absMean = new double[FeatureNames.Length]; 
            int n = 0;

            foreach(var sample in data.Take(30))
            {
                var p = engine.Predict(sample);

                if(p.FeatureContributions == null) continue;

                for(int i=0; i<absMean.Length && i < p.FeatureContributions.Length; i++)
                {
                    absMean[i] += Math.Abs(p.FeatureContributions[i]); // 기여도 절대값들을 넣는다.
                }
                n++;
            }

            for(int i=0; i<absMean.Length; i++)
            {
                absMean[i] /= Math.Max(1, n); // 0으로 나누는 것을 방지하며 기여도 절대값들의 평균을 구한다.
            }

            Console.WriteLine("  피처별 평균 기여 크기 (30건 평균) :");

            for(int i=0; i<FeatureNames.Length; i++)
            {
                string tag = trueCoeff[i] == 0.0f ? "← 0에 가까워야 정상" : "← 커야 정상";
                Console.WriteLine($"    {FeatureNames[i],-12} : {absMean[i],7:F2}  {tag}");
            }

            // 진짜 피처 3개의 크기 순서가 정답과 일치하는지 체크(6.70 > 4.33 > 2.08)
            bool orderOk = absMean[0] > absMean[1] && 
                           absMean[1] > absMean[2];
            Console.WriteLine($"  크기 순서 : 납기여유일 > 거래처재고 > 운송빈도 : {(orderOk ? "통과" : "실패")}");

            // 노이즈 3개가 진짜 피처 중 가장 작은것 보다 작은지 체크
            double maxNoise = Math.Max(absMean[3], Math.Max(absMean[4], absMean[5]));
            bool noiseOk = maxNoise < absMean[2];
            Console.WriteLine($"  노이즈 피처 기여도 < 진짜 피처 최소값 : {(noiseOk ? "통과" : "실패")} (노이즈 최대 {maxNoise:F2} vs 운송빈도 {absMean[2]:F2})");

            // 기준값이 샘플마다 일관되는지 체크
            if(biasList.Count == 0)
            {
                Console.WriteLine("  [경고] 기여도를 읽지 못했습니다. [ColumnName]이 \"FeatureContributions\"인지 확인하세요.");
            }
            else
            {
                float biasRange = biasList.Max() - biasList.Min();
                Console.WriteLine($"  기준값 편차 : {biasRange:F2}일");
                Console.WriteLine(biasRange < 1.0f
                    ? "    -> 기준값이 일관됨. 실데이터에서 42.5/54.2/39.2로 흔들린 것은 '과적합' 때문이었다는 결론."
                    : "    -> 합성데이터에서도 기준값이 흔들림. ML.NET 기여도가 SHAP처럼 bias+합=Score 항등식을 보장하지 않는 듯함");
            }


            
        }

        private void RunRowCountExperiment()// 행 수를 바꿔가며 과적합 변화를 관찰하는 실험 메서드
        {
            Console.WriteLine();
            Console.WriteLine("########## 실험 A : 행 수 효과 (sigma=3 고정) ##########");
            Console.WriteLine("  행수 | 학습 R² | 검증 R² | 격차");
            Console.WriteLine("  -----+---------+---------+------");

            foreach (int rowCount in new[] { 176, 500, 1000, 3000 })
            {
                var mlContext = new MLContext(seed: Seed);
                var data = new SyntheticDataGenerator().Generate(rowCount, DefaultSigma, Seed);

                IDataView allData = mlContext.Data.LoadFromEnumerable(data);

                var split = mlContext.Data.TrainTestSplit(allData, testFraction: 0.2, seed: Seed);

                var model = BuildPipeline(mlContext).Fit(split.TrainSet);
                // 파이프라인 구성을 메서드로 분리해둔 덕에 조건만 바꿔 반복 실험할 수 있다.

                var trainR2 = mlContext.Regression.Evaluate(model.Transform(split.TrainSet), labelColumnName: "Label").RSquared;
                var testR2  = mlContext.Regression.Evaluate(model.Transform(split.TestSet),  labelColumnName: "Label").RSquared;

                Console.WriteLine($"  {rowCount,4} | {trainR2,7:F3} | {testR2,7:F3} | {trainR2 - testR2,5:F3}");
            }

            Console.WriteLine();
            Console.WriteLine("  * 기대 패턴 : 행수가 늘면 학습 R²는 내려가고 검증 R²는 올라가며 격차가 좁혀진다.");
            Console.WriteLine("    = 데이터가 많아지면 모델이 '외우기'에서 '학습하기'로 바뀐다는 뜻.");
            Console.WriteLine("  * 핵심 : 176행에서도 검증 R²가 0.8 이상 나온다.");
            Console.WriteLine("    실데이터 176행은 R² -0.35였으므로, 실패 원인은 '행이 적어서'가 아니라 '정보가 없어서'다.");
            Console.WriteLine("    -> 해결책은 데이터를 더 모으는 것이 아니라 거래처 납기 같은 컬럼을 수집하는 것.");
            // 이 결론이 결과분석 문서의 '원인 ③ 176행 부족' 가설을 반증한다. 실무에서는 투자 방향을 바꾸는 판단이 된다.
        }

        private void RunSigmaExperiment() // 노이즈 크기를 바꿔가며 RSquared 상한 변화를 관찰하는 실험 메서드
        {
            Console.WriteLine();
            Console.WriteLine("########## 실험 B : 노이즈 효과 (176행 고정) ##########");
            Console.WriteLine("  sigma | 검증 R² | 이론상한 | 비고");
            Console.WriteLine("  ------+---------+----------+------");

            var theoretical = new Dictionary<float, double> { { 1f, 0.981 }, { 3f, 0.877 }, { 5f, 0.732 }, { 8f, 0.533 } };
            // 사전에 계산해둔 이론 상한값. 노이즈가 차지하는 분산 비율만큼은 어떤 모델도 설명할 수 없다.

            foreach (float sigma in new[] { 1f, 3f, 5f, 8f })
            {
                var mlContext = new MLContext(seed: Seed);
                var data = new SyntheticDataGenerator().Generate(DefaultRowCount, sigma, Seed);
                IDataView allData = mlContext.Data.LoadFromEnumerable(data);
                var split = mlContext.Data.TrainTestSplit(allData, testFraction: 0.2, seed: Seed);

                var model = BuildPipeline(mlContext).Fit(split.TrainSet);
                var testR2 = mlContext.Regression.Evaluate(model.Transform(split.TestSet), labelColumnName: "Label").RSquared;

                string note = testR2 > theoretical[sigma] ? "상한 초과(누출 의심)" : "정상";
                Console.WriteLine($"  {sigma,5:F0} | {testR2,7:F3} | {theoretical[sigma],8:F3} | {note}");
            }

            Console.WriteLine();
            Console.WriteLine("  * 노이즈가 커지면 R²가 떨어진다. 이것은 모델의 한계가 아니라 '설명 불가능한 변동'이 늘어난 것이다.");
            Console.WriteLine("  * 현실 데이터에는 항상 노이즈가 있으므로, R² 1.0은 목표가 될 수 없다.");
            Console.WriteLine("    이론 상한을 넘으면 오히려 데이터 누출을 의심해야 한다.");
        }

        
    }
}