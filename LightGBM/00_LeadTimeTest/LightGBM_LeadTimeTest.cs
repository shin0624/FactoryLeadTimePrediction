using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

namespace FactoryLeadTimePrediction
{
    // ML.NET을 사용한 제조현장 데이터 기반 예측모델 개발 테스트 
    // 1. LightGBM 개발

    public class LightGBM_LeadTimeTest
    {
        private static readonly LightGBM_LeadTimeTest _instance = new LightGBM_LeadTimeTest();
        public static LightGBM_LeadTimeTest Instance() => _instance;

        public void StartLightGBM_LeadTimeTest()
        {
            var mlContext = new MLContext(seed:42);// mlcontext 초기화

            var trainingData = new List<ProductionData>
            {
                new ProductionData {InputQuantity = 100.0f, MachineTemperature = 75.5f, WorkerCount =3.0f, LeadTime = 45.0f },
                new ProductionData { InputQuantity = 150.0f, MachineTemperature = 78.2f, WorkerCount = 4.0f, LeadTime = 55.0f },
                new ProductionData { InputQuantity = 200.0f, MachineTemperature = 80.0f, WorkerCount = 5.0f, LeadTime = 62.0f },
                new ProductionData { InputQuantity = 80.0f,  MachineTemperature = 74.0f, WorkerCount = 2.0f, LeadTime = 40.0f },
                new ProductionData { InputQuantity = 120.0f, MachineTemperature = 76.1f, WorkerCount = 3.0f, LeadTime = 48.0f }  
            };// 테스트용 가상 제조 데이터 생성

            IDataView trainDataView = mlContext.Data.LoadFromEnumerable(trainingData);// 데이터를 ML.NET 내부 데이터 형식인 IDataView 형식으로 변환.

            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features",
                nameof(ProductionData.InputQuantity),
                nameof(ProductionData.MachineTemperature),
                nameof(ProductionData.WorkerCount)
            ); 
            // 데이터 전처리 파이프라인 구축. LightGBM은 입력 Features를 하나의 벡터 배열인 Features로 묶어주어야 함.
            //mlContext.Transforms.Concatenate()는 C# 데이터모델의 개별 프로퍼티 필드들을 LightGBM이 한번에 행렬 연산할 수 있도록 Features라는 이름의 하나의 벡터 컬럼으로 묶어주는 전처리 메서드.


            /*
            만약 WorkerCount를 Int로 선언했을 경우
            var dataProcessPipeline = mlContext.Transforms.Conversion.ConvertType("WorkerCountFloat", nameof(ProductionData.WorkerCount), DataKine.Single)
                                        .Append(mlContext.Transforms.Concatenate("Features",
                                                nameof(ProductionData.InputQuantity),
                                                nameof(ProductionData.MachineTemperature),
                                                "WorkerCountFloat"));
                                                --> 이처럼 파이프라인 구성 시 타입캐스팅 단계를 추가하여 "WorkerCount(int)"컬럼을 "WorkerCountFloat(float)"컬럼으로 변환하고, 변환된 float 컬럼을 포함하여 Features벡터로 병합해야 함.
                                                이때, int형 변수명 대신 float 컬럼명을 넣어야함
                                                bool, double 등 float 이외의 데이터도 위와 같이 캐스팅해야 함
            */

            var options = new LightGbmRegressionTrainer.Options
            {
                NumberOfLeaves = 4,// 트리 노드(잎사귀) 갯수
                MinimumExampleCountPerLeaf = 2, // 노드 당 최소 데이터 수
                LearningRate = 0.1, // 학습률
                NumberOfIterations = 100, // 부스팅 반복 횟수(Tree 갯수)
                Booster = new GradientBooster.Options{L2Regularization = 0.5}, // L2 규제 적용(L2 규제 : 과적합 방지를 위한 손실함수에 가중치 제곱합을 패널티로 더해주는 기법)
                FeatureColumnName = "Features",
                LabelColumnName = "LeadTime"
            };
            // LightGBM 회귀 트레이너 추가 및 하이퍼 파라미터 설정.
            // LightGbmRegressionTrainer.Options는 C# 코드 안에서 LightGBM의 핵심 파라미터를 세밀하게 튜닝할 수 있는 옵션 객체. 이상치에 강한 손실함수 등 이 옵션 내에서 설정 조절이 가능.

            var trainer = mlContext.Regression.Trainers.LightGbm(options);// 하이퍼 파라미터를 설정한 LightGBM 옵션을 트레이너에 적용

            var trainingPipeline = dataProcessPipeline.Append(trainer);// 데이터 전처리 파이프라인과 트레이너를 결합

            Console.WriteLine("---------LightGBM 모델 학습 시작---------");

            var trainedModel = trainingPipeline.Fit(trainDataView);//IDataView 형식으로 변환된 학습용 데이터들을 주입하고 학습을 시작.
            // Fit()메서드는 매개변수로 들어온 IDataView 형식의 데이터로 학습을 시작하고, 종료 시 ITransformer 형태의 학습 모델을 반환하는 메서드. 

            Console.WriteLine("---------모델 학습 완료---------");

            
            var transformedTrainedModel = trainedModel.Transform(trainDataView); // 기여도 계산기에 넣기 위한 Features컬럼이 있는 데이터를 만든다. 


            // -- 예측결과의 근거 산출을 위한 기여도 계산기 삽입 과정 추가 ---
            var contributionCalcPipeline = mlContext.Transforms.CalculateFeatureContribution(trainedModel.LastTransformer, normalize : false).Fit(transformedTrainedModel);
            /*
                ExplainabilityCatalog.CalculateFeatureContribution(매개변수 너무 김) :  개별 데이터 예측 시, Feature가 예측값에 어떤 긍정적/부정적 영향을 미쳤는지(기여도)를 계산하는 설명 도구 메서드.
                - 즉, 각 Features(투입수량, 기계온도, 작업자수)에 어느정도 기여도가 가해져서 LightGBM의 최종 예측 리드타임이 62분이 나왔는지를 확인할 수 있다.
                - 예를 들어, 투입 수량 점수가 +12.5, 작업자 수 점수가 -5.0이라면
                    "투입 수량이 많아서 리드타임이 12.5분 늘어났고, 작업자를 많이 배치해서 5분이 단축된 결과, 리드타임을 MM분이라고 예측할 수 있었다"라는 예측결과 도출의 근거를 알 수 있음.
                - 주요 매개변수
                    ISingleFeaturePredictionTransformer<TModel> predictionTransformer : 예측 모델
                    int numberOfPositiveContributions : 긍정적인 기여도의 수. 가장 높은 크기에서 가장 낮은 크기로 정렬. 미지정 시 10
                    int numberOfNegativeContributions : 부정적인 기여도의 수. 가장 높은 크기에서 가장 낮은 크기로 정렬. 미지정시 10
                    bool normalize : 기여도를 -1~1간격으로 정규화해야 하는지 여부
                - 기여도 계산기도 Fit()을 한번 거쳐야 ITransformer형이 된다. (공식문서 예시에서도 계산기 뒤에 Fit()을 붙인 것이 이것때문인듯)
            */

            var scoringPipeline = trainedModel.Append(contributionCalcPipeline);// 기여도 컬럼을 만들어주는 변환기가 붙은 새 모델.

            var predictionEngine_contributionCalcResult = mlContext.Model.CreatePredictionEngine<ProductionData, LeadTimePrediction>(scoringPipeline);// 기여도 컬럼이 붙은 새 모델로 예측엔진 생성

            // 잠시 비활성화
            // var predictionEngine = mlContext.Model.CreatePredictionEngine<ProductionData, LeadTimePrediction>(trainedModel);// 학습 완료된 trainedModel을 넣어서 실시간 예측 엔진 객체 생성
            /*
            ModelOperationsCatalog.CreatePredictionEngine<TSrc, TDst>(ITransformer, DataViewSchema) : 일회성 예측을 위한 예측 엔진을 만드는 메서드. 
            TSrc : 입력 데이터(제조 데이터)
            TDst : 출력 데이터 (모델이 예측한 리드타임)
            매개변수 ITransformer : 예측에 사용할 ITransformer형 모델
            매개변수 DataViewSchema : 입력 스키마
            리턴 : PredictionEngine<TSrc, TDst>
            CreatePredictionEngine()을 사용하면 학습 완료된 모델을 메모리에 올려놓고 공장에서 새로운 데이터가 들어올 때 마다 N밀리초(ms)만에 실시간으로 생산량이나 리드타임을 찍어내는 고성능 예측엔진을 빌드할 수 있음.(ML.NET IDataView의 지연평가 기반 구조 + c#의 런타임속도 )        
            */
           
            var newSituation = new ProductionData
            {
                InputQuantity =180.0f,
                MachineTemperature = 74.0f,
                WorkerCount = 8.0f
            }; // 예측 수행에 사용될 새로운 공정 환경 데이터 객체

            var prediction = predictionEngine_contributionCalcResult.Predict(newSituation);// 새로운 공정환경 데이터를 예측 엔진에 주입 후 예측 시작
            // PredictionEngine.Predict(TSrc) : 예측 파이프라인을 실행. 매개변수 TSrc는 예측을 실행할 소스(예제), 리턴값은 TDst 타입의 예측 결과 개체
            // PredictionEngine.Predcit(TSrc, TDst) : 예측 파이프라인 실행. 매개변수 TSrc는 예측을 수행할 소스(예제)이며, TDst는 예측 결과를 저장할 개체. null로 둘 경우 새 항목이 만들어지고, 명시하면 이전 항목이 다시 사용됨.

            Console.WriteLine("---------실시간 리드타임 예측 결과---------");
            Console.WriteLine($"[입력 조건]\n1. 투입량 : {newSituation.InputQuantity}개\n2. 온도 : {newSituation.MachineTemperature}도\n3. 작업자 수 : {newSituation.WorkerCount}명");
            Console.WriteLine($"[예측 결과]예상 공정 리드타임 : {prediction.PredictedLeadTime:F2}분");


            string[] featuresNames = new[] {"투입량", "온도", "작업자 수"};// 기여도 배열의 라벨 순서는 Features와 같음.
            for(int i=0; i< featuresNames.Length; i++)
            {
                Console.WriteLine($"- {featuresNames[i]} 기여도 : {prediction.FeatureContributions?[i]:F2}");
            }

            //mlContext.Model.Save(trainedModel, trainDataView.Schema, "models/LightGBM_Model.zip");
        }

    }
}