using Microsoft.ML.Data;

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
        [LoadColumn(3), ColumnName("LeadTime")] public float LeadTime{get;set;}// 실제 리드타임(LightGBM이 예측할 대상)
    }
}