using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace FactoryLeadTimePrediction.DeepLearning.A_CastingDefectClassification
{
    public class CastingPrediction // 이미지 분류 모델 출력 DTO
    {
        [ColumnName("PredictedLabel")] public string PredictedLabel {get;set;} = string.Empty; // 모델이 예측한 클래스 명(defect / ok)
        // 파이프라인 끝의 MapKeyToValue 변환기가 내부 Key(0,1)를 다시 원래 문자열로 변환. 이게 없으면 "1"이라는 숫자만 나오기에, 어느 클래스인지 알 수 없음
        // string -> MapValueToKey(Label) -> Inference -> MapKeyToValue(PredictedLabel) -> string

        public float[] Score {get;set;} = Array.Empty<float>(); // 클래스별 확률(softmax 출력)을 담는 배열. 길이는 클래스 수(defect / ok 2개)
        // Regression의 Score는 float 하나(예측값)였으나, Classification의 Score는 float형 배열.
        // 배열 순서는 MapValueToKey가 매긴 Key 순서이며, 알파벳순이 아니라 데이터에 등장한 순서. -> 그래서 어느 인덱스가 어느 클래스인지 스키마에서 직접 확인해야 함.

        /*
        제조 품질검사 핵심 지표 
        - AI 예측 정상 && 실제 이미지 정상 : TN
        - AI 예측 정상 && 실제 이미지 불량 : FN
        - AI 예측 불량 && 실제 이미지 정상 : FP
        - AI 예측 불량 && 실제 이미지 불량 : TP
        --> FN(불량인데 정상이라 예측한 것 ,불량 유출)률이 낮아야 함. 즉, Recall(TP / (TP+FN))-즉, 불량 검출률을 주 지표로 삼아야 함. 

        Score 배열이 중요한 이유
        - PredictedLabel은 단순히 Score가 가장 큰 클래스를 고른 결과(argmax)에 해당함.(임계값 0.5)
        - 품질검사에서는 임계값 0.5가 최적이 아니며, 불량유출(FN)을 줄이려면 임계값을 낮춰야 함
        - 그 조정을 하려면 PredictedLabel이 아니라 Score 원본 확률이 필요.
        CalculateFeatureContribution을 사용할 수 없음
        - 이미지 분류 모델에서는 위 기여도 계산 메서드 사용 불가
        - Vision 모델의 설명가능성은 Grad-CAM / Saliency Map과 같은 기법을 사용하며, ML.NET에 내장 지원이 없음.
        - 이번에는 오분류 사례 분석으로 모델 한계를 파악한다.
        
        */
    }
}