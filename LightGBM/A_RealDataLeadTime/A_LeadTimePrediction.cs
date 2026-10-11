using Microsoft.ML.Data;

namespace FactoryLeadTimePrediction.LightGBM.A_RealDataLeadTime
{
    public class A_LeadTimePrediction // 실데이터 기반 리드타임 예측값(AI 출력값)을 담는 DTO
    {
        [ColumnName("Score")] public float PredictedLeadTime {get;set;} // LightGBM이 예측한 리드타임
        [ColumnName("FeatureContributions")] public float[]? FeaturesContributions {get;set;} // 모델 예측 결과 근거(기여도)
    }
}