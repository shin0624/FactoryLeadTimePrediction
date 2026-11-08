using Microsoft.ML.Data;

namespace FactoryLeadTimePrediction.LightGBM.B_SyntheticLeadTime// 합성 데이터의 ml.net 입력 DTO
{
    public class SyntheticLeadTimeFeature
    {
        // 라벨(예측 대상) -> IDataView 내에서 컬럼 명을 Label로 지정하는 이유 : Evaluate(), CrossValidate() 등에서 라벨을 기본값 Label으로 인식하기 때문
        [ColumnName("Label")] public float LeadTimeDays {get;set;} // 일단위 리드타임 (생성 규칙 : 8.0 + 0.8*CustomerDueDays + 1.5*CustomerStockLevel - 1.2*ShippingFrequency + N(0, sigma), 이후 max(1.0, ...)으로 클리핑)

        // 리드타임을 실제로 결정할 피처 3개
        public float CustomerDueDays {get;set;}// 거래처가 요구한 납기 여유일. Uniform(1,30) 범위. (납기가 여유로우면 창고에 제품이 더 오래 머물게 됨. 즉, 리드타임 증가 요인) (계수 +0.8 / 기여 표준편차 6.70)

        public float CustomerStockLevel {get;set;} // 거래처 재고 수준. Uniform(0, 10) 범위. (거래처에 재고가 많이 있으면 공장에서 급히 받을 필요 없음. 즉, 리드타임 증가 요인) (계수 +1.5 / 기여 표준편차 4.33)

        public float ShippingFrequency {get;set;} // 주당 운송 편성 횟수. Uniform(1, 7) 범위. (운송편이 잦으면 제품이 공장에서 더 빨리 실려나감. 즉, 리드타임 감소 요인(유일한 음수 계수)) (계수 -1.2 / 기여 표준편차 2.08)

        /*
        계수와 분포 범위 조절 이유
         - 계수만 바꾸면 입력값의 산포가 같아, 세 피처의 실제 영향력이 비슷해짐
         - 기여 표준편차는 |계수| * 입력 표준편차 --> 분포까지 조절하여 6.70 / 4.33 / 2.08로 뚜렷히 차등
         - 이렇게 해야 기여도 분석 결과가 DueDays > StockLevel > ShippingFreq 순서를 재현하는지 정답 대조가 가능.
        */

        // 리드타임과 무관한 노이즈 피처 2개
        public float ProductionKg {get;set;} // 생산량(kg). Normal(14000, 1000) 범위, 계수 0 -> 리드타임과 무관
        public float ProcessMinutes {get;set;} // 공정 소요시간(분 단위).  Normal(194, 26) 범위, 계수 0 -> 리드타임과 무관
        public float WeekdayIndex {get;set;} // 요일(0~4 = 월~금). Uniform(0, 4), 계수 0 -> 리드타임과 무관 (실데이터에서 주말 0건이므로, 0~4만 생성)
        /*
        노이즈 피처 추가 이유
        - 위 3개 피처의 기여도가 0 근처로 나와야 기여도 도구가 정상이라는 의미임.
        - 위 3개 피처에서 큰 값이 나오면, 기여도 해석을 신뢰할 수 없다는 신호임(과적합 등)
        - 실데이터 실습에서는 기여도가 맞는지 틀린지 판단할 기준이 없었음.
        - 피처 개수를 6개로 맞춘 것도, 실데이터 실습 때와 조건을 갖게 하기 위함
        
        */
    }
}