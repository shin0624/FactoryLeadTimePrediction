using Microsoft.ML.Data;


namespace FactoryLeadTimePrediction
{
    public class LeadTimePrediction// 예측 결과 데이터의 구조 정의
    {
        [ColumnName("Score")] public float PredictedLeadTime{get;set;}//모델이 예측한 리드타임

        [ColumnName("FeatureContributions")] public float[]? FeatureContributions {get;set;} // 모델 예측 결과의 근거를 Feature에 대한 기여도 점수로 출력하기 위한 기여도점수 배열 선언.
    }
}