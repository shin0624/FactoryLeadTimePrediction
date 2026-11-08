using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace FactoryLeadTimePrediction.LightGBM.B_SyntheticLeadTime
{
    public class SyntheticLeadTimePrediction // 합성 데이터 예측 결과를 담는 DTO
    {
        [ColumnName("Score")] public float PredictedLeadTime {get;set;} // 예측된 리드타임. ML.NET 회귀모델의 예측값은 항상 Score라는 컬럼명으로 나옴. 프로퍼티 명은 자유롭게 하되, [ColumnName]으로 매핑한다.

        [ColumnName("FeatureContributions")] public float[]? FeatureContributions {get;set;} // 기여도 배열. 
    }
}