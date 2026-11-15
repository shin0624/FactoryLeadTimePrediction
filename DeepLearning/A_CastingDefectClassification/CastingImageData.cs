using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace FactoryLeadTimePrediction.DeepLearning.A_CastingDefectClassification
{
    public class CastingImageData // 이미지 분류 모델 입력 DTO
    {
        /*
        Image Classification은 CNN이 픽셀에서 피처를 스스로 학습하므로, 개발자의 피처 엔지니어링이 필요 없음.
        --> LightGBM 등 전통적인 방식의 ml(수동 피처 설계)과 딥러닝(표현 학습)의 가장 큰 차이점
        */

        public string ImagePath {get;set;} = string.Empty; // 이미지 파일 절대 경로 -> 파이프라인의 LoadRawImageBytes 변환기가 이 경로를 읽어 바이트로 변환
        // 이미지 데이터셋을 모두 메모리에 올리지 않고 경로만 들고 있다가, 배치 단위로 읽으므로 메모리 효율 측면에서 좋음

        public string Label {get;set;} = string.Empty; // 정답 라벨(defect / ok 문자열) -> Classification은 LightGBM 등의 Regression과 달리 범주 형태의 Label 사용. 파이프라인의 MapValueToKey 변환기가 이 문자열을 내부 Key 타입(0,1)으로 변환

        // Regression과 Classification의 차이
        // Regression : [ColumnName("Label")] public float LeadTimeDays -> 연속값. 해당 값이 그대로 학습에 투입됨
        // Classification : public string Label -> 범주값. MapValueToKey로 Key 변환 필수 -> ColumnName을 붙이지않는 이유 : 프로퍼티 이름 자체가 이미 "Label"이기 때문
    }
}