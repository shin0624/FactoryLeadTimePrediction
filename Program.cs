using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

namespace FactoryLeadTimePrediction
{
    // ML.NET을 사용한 제조현장 데이터 기반 예측모델 개발 테스트 
    // 1. LightGBM 개발
    public class ProductionData // 입력 데이터 구조 정의(feature)
    {
        [LoadColumn(0)] public float InputQuantity {get;set;}// 투입 수량
        [LoadColumn(1)] public float MachineTemperature {get;set;}// 설비 가동 온도
        [LoadColumn(2)] public float WorkerCount {get;set;}// 투입 작업자 수 --> 모델 학습 메서드인 Fit() 실행 시, LightGBM을 포함한 ML.NET의 대부분의 학습 알고리즘은 Int32가 섞여있는 Feature 컬럼을 받지 못함.(학습데이터에 int형 컬럼이 있으면 IOE에러발생)
        // ML.NET 알고리즘의 Features컬럼은 반드시 모든 요소가 Single(float)타입인 단일 벡터 배열 형태여야 함.
        /*
        왜?
        - ML.NET에서 여러 피처를 하나로 묶어주는 Concatenate 변환기는 결합하려는 피처들의 데이터 타입이 모두 같아야 하거나, 내부적으로 하나의 고정된 데이터 타입 벡터(VBuffer<float>)로 변환해야 함.
        - int와 float를 무작위로 섞어서 묶으려고 하면 파이프라인 구성이나 Fit() 시점에 타입 불일치 에러가 발생.
        - 학습 알고리즘의 수치 연산 제약
            LightGBM의 내부 코어 엔진은 하드웨어 연산 최적화를 위해 입력 데이터를 32비트 부동소수점(float / Single) 또는 64비트 부동소수점(double)으로 변환하여 트리를 분할하기에, int타입을 그대로 전달하면 인식 불가.
        -> 해결법
            파이프라인 내에서 int를 float로 캐스팅하고 Concatenate()를 수행해야 함.
            ML.NET은 이를 위해 Transforms.Conversion.ConvertType 메서드를 제공.
        */
        [LoadColumn(3), ColumnName("Label")] public float LeadTime{get;set;}// 실제 리드타임(LightGBM이 예측할 대상)
    }

    public class LeadTimePrediction// 예측 결과 데이터의 구조 정의
    {
        [ColumnName("Score")] public float PredictedLeadTime{get;set;}//모델이 예측한 리드타임
    }

    class Program
    {
        static void Main(string[] args)
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
                LabelColumnName = "Label"
            };
            // LightGBM 회귀 트레이너 추가 및 하이퍼 파라미터 설정.
            // LightGbmRegressionTrainer.Options는 C# 코드 안에서 LightGBM의 핵심 파라미터를 세밀하게 튜닝할 수 있는 옵션 객체. 이상치에 강한 손실함수 등 이 옵션 내에서 설정 조절이 가능.

            var trainer = mlContext.Regression.Trainers.LightGbm(options);// 하이퍼 파라미터를 설정한 LightGBM 옵션을 트레이너에 적용

            var trainingPipeline = dataProcessPipeline.Append(trainer);// 데이터 전처리 파이프라인과 트레이너를 결합

            System.Console.WriteLine("---------LightGBM 모델 학습 시작---------");
            var trainedModel = trainingPipeline.Fit(trainDataView);//IDataView 형식으로 변환된 학습용 데이터들을 주입하고 학습을 시작.
            // Fit()메서드는 매개변수로 들어온 IDataView 형식의 데이터로 학습을 시작하고, 종료 시 ITransformer 형태의 학습 모델을 반환하는 메서드. 
            System.Console.WriteLine("---------모델 학습 완료---------");

            var predictionEngine = mlContext.Model.CreatePredictionEngine<ProductionData, LeadTimePrediction>(trainedModel);// 학습 완료된 trainedModel을 넣어서 실시간 예측 엔진 객체 생성
            // ModelOperationsCatalog.CreatePredictionEngine<TSrc, TDst>(ITransformer, DataViewSchema) : 일회성 예측을 위한 예측 엔진을 만드는 메서드. 
            // TSrc : 입력 데이터(제조 데이터)
            // TDst : 출력 데이터 (모델이 예측한 리드타임)
            // 매개변수 ITransformer : 예측에 사용할 ITransformer형 모델
            // 매개변수 DataViewSchema : 입력 스키마
            // 리턴 : PredictionEngine<TSrc, TDst>
            // CreatePredictionEngine()을 사용하면 학습 완료된 모델을 메모리에 올려놓고 공장에서 새로운 데이터가 들어올 때 마다 N밀리초(ms)만에 실시간으로 생산량이나 리드타임을 찍어내는 고성능 예측엔진을 빌드할 수 있음.

            System.Console.Write("투입량 : ");
            string iqInput = Console.ReadLine();
            System.Console.Write("온도 : ");
            string mtInput = Console.ReadLine();
            System.Console.Write("작업자 수 : ");
            string wcInput = Console.ReadLine();
                        
            if(float.TryParse(iqInput, out float iq))
            {
                System.Console.WriteLine("형변환 성공 1");
            }
            else
            {
                System.Console.WriteLine("형변환 실패 1");
            }

            if(float.TryParse(mtInput, out float mt))
            {
                System.Console.WriteLine("형변환 성공 2");
            }
            else
            {
                System.Console.WriteLine("형변환 실패 2");
            }

            if(float.TryParse(wcInput, out float wc))
            {
                System.Console.WriteLine("형변환 성공 3");
            }
            else
            {
                System.Console.WriteLine("형변환 실패 3");
            }
            
        
            var newSituation = new ProductionData
            {
                InputQuantity =iq,
                MachineTemperature = mt,
                WorkerCount = wc
            }; // 예측 수행에 사용될 새로운 공정 환경 데이터 객체

            var prediction = predictionEngine.Predict(newSituation);// 새로운 공정환경 데이터를 예측 엔진에 주입 후 예측 시작
            // PredictionEngine.Predict(TSrc) : 예측 파이프라인을 실행. 매개변수 TSrc는 예측을 실행할 소스(예제), 리턴값은 TDst 타입의 예측 결과 개체
            // PredictionEngine.Predcit(TSrc, TDst) : 예측 파이프라인 실행. 매개변수 TSrc는 예측을 수행할 소스(예제)이며, TDst는 예측 결과를 저장할 개체. null로 둘 경우 새 항목이 만들어지고, 명시하면 이전 항목이 다시 사용됨.

            System.Console.WriteLine("---------실시간 리드타임 예측 결과---------");
            System.Console.WriteLine($"[입력 조건]\n1. 투입량 : {newSituation.InputQuantity}개\n2. 온도 : {newSituation.MachineTemperature}도\n3. 작업자 수 : {newSituation.WorkerCount}명");
            System.Console.WriteLine($"[예측 결과]예상 공정 리드타임 : {prediction.PredictedLeadTime:F2}분");

            mlContext.Model.Save(trainedModel, trainDataView.Schema, "D:/261008_LightGBM_Test/FactoryLeadTimePrediction/models/LightGBM_Model_TrainingTest.zip");
    }
}
}