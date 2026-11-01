using Microsoft.ML.Data;

namespace FactoryLeadTimePrediction.LightGBM.A_RealDataLeadTime
{
    public class LeadTimeFeature // RawProductionRecord를 변환하여 LightGBM에 실제로 투입하는 ML.NET 입력 DTO.
    {
        /*
        이 클래스의 역할 2가지
        1. 타입 변환의 종착점 : DateTime, string, 배열을 전부 float 스칼라로 평탄화한다.
           ML.NET 알고리즘의 Features 벡터는 모든 요소가 Single(float)이어야 하므로 int/DateTime을 섞으면 Fit()에서 에러.
        2. 데이터 누출 차단벽 : 이 클래스에 없는 컬럼은 모델이 볼 수 없다.
           Raw DTO에 출고량/출고횟수가 담겨 있어도 여기에 넣지 않으면 학습에 쓰이지 않는다. 설계로 누출을 막는 방식.

        [LoadColumn]을 붙이지 않는 이유
            - [LoadColumn]은 LoadFromTextFile()로 CSV를 직접 읽을 때 컬럼 인덱스를 알려주는 속성.
            - 여기서는 CsvLoader가 직접 파싱한 List를 LoadFromEnumerable()로 넣으므로, 프로퍼티 이름이 그대로 컬럼명이 된다.
        */

        [ColumnName("Label")] public float LeadTimeDays { get; set; }
        /*
        예측 대상(Label). 리드타임 = ReleaseDates[0] - ProductionDate (일 단위)
        - 컬럼명을 "Label"로 두는 이유 : Evaluate(), CrossValidate()가 라벨 컬럼을 기본값 "Label"로 찾는다.
          "LeadTime" 등으로 두면 평가/교차검증 호출마다 labelColumnName을 명시해야 함.
        - 프로퍼티 이름에는 단위(Days)를 넣는다. 00_LeadTimeTest의 더미데이터는 "분" 단위였고 이번은 "일" 단위이므로,
          두 실습 결과를 비교할 때 단위 혼동을 막기 위함.
        */

        // ---------- 이하 Features (생산 시점까지 확정된 정보만) ----------

        public float ProductionKg { get; set; } // 생산량(Kg). 실측 7,500 ~ 17,250 (평균 14,056)
        public float ProcessMinutes { get; set; } // 공정 소요시간 = EndTime - StartTime (분). 실측 100 ~ 385 (평균 193.6)
        /*
        시간(h) 단위가 아니라 분(min) 단위로 두는 이유
            - 시간 단위면 1.67 ~ 6.42로 범위가 좁아 트리 분할 지점이 뭉친다. 분 단위가 분할에 유리.
        */

        public float WeekdayIndex { get; set; } // 생산일자의 요일. (int)DateTime.DayOfWeek -> .NET은 일요일=0 기준
        /*
        이름을 DayOfWeek로 하지 않는 이유 : System.DayOfWeek 열거형과 이름이 겹쳐 혼동됨.
        요일/월을 숫자로 넣는 것의 한계
            - 0~6 숫자로 넣으면 트리가 "요일 <= 2.5" 같은 분할을 하므로 순서가 있는 값처럼 취급된다. 실제로는 범주(category)임.
            - 제대로 하려면 One-Hot Encoding(mlContext.Transforms.Categorical.OneHotEncoding)이 맞다.
            - 다만 트리 모델은 범주를 숫자로 넣어도 선형 모델과 달리 꽤 잘 작동하고,
              실데이터는 요일이 월~금 5개뿐(주말 생산 0건)이므로 1단계에서는 숫자로 두고 One-Hot은 이후 비교 실험으로 남긴다.
        */

        public float Month { get; set; } // 생산일자의 월 (1~12). 계절성 확인용
        public float SameDayOrderCount { get; set; } // 같은 생산일자에 몇 건이 지시되었는지. 실측 1 ~ 6 (평균 3.2)
        /*
        이 피처만 성격이 다름
            - 나머지 피처는 한 행만 보면 계산되지만, 이건 전체 데이터를 봐야 계산된다(2-pass 필요).
            - 실전 배포 시 주의 : "오늘 총 몇 건 생산하는지"를 생산 시작 시점에 알 수 있어야 누출이 아니다.
              생산 계획이 전일에 확정되면 OK, 당일 추가되면 누출. 누출 판단은 피처마다 따로 해야 한다는 예시.
        */

        public float StartHour { get; set; } // 생산 시작 시각의 시(0~23). 교대조 영향 확인용
        /*
        리드타임과의 상관계수가 실측 -0.000 으로 사실상 무의미한 피처.
        일부러 넣어두는 이유 : 기여도 계산기가 이 피처에 0.00을 주는 것을 확인하기 위함.
            - 00_LeadTimeTest에서 온도/작업자수가 0.00이었던 것은 "다중공선성(피처 간 완벽 상관)" 때문이고,
            - 이 피처가 0.00이 나오는 것은 "라벨과 상관이 없어서"다. 원인이 다른 두 경우를 비교하면 기여도 해석이 깊어진다.
        */

        /*
        ---------- 의도적으로 제외한 컬럼 ----------
        1) 1~10차 출고량, 출고 분할 횟수 : 데이터 누출(Data Leakage)
           - 생산 시점에 "몇 번에 나눠 나갈지", "1차에 몇 Kg 나갈지"는 알 수 없다. 리드타임과 동시에 결정되는 값.
           - 실측 상관계수가 출고횟수 +0.295로 가장 높지만, 그래서 더 위험하다. 성능 좋은 피처 != 쓸 수 있는 피처.
        2) OrderNumber(작업지시번호) : ID 컬럼
           - 숫자로 넣으면 트리가 "번호 <= 2103500000이면 15일" 식으로 학습 데이터를 암기한다(과적합).
           - 게다가 앞 6자리가 날짜(210113 = 2021-01-13)라서 시간 추세를 간접 학습해버린다.
        3) ItemCode(제품코드) : 177행 전부 "M21" 이라 분산이 0. 트리가 분할할 수 없으므로 정보가 없다.
           - 단, 향후 다른 제품 데이터가 추가되면 그때는 피처가 된다.

        ---------- 베이스라인 (3단계에서 비교할 기준) ----------
        평균값으로 전부 예측했을 때 RMSE = 9.9일 / 중앙값으로 전부 예측했을 때 MAE = 7.39일
        -> Evaluate() 결과가 RMSE 9.9보다 크면 모델이 "평균값으로 예측"보다 못한 것이다.
        */
    }
}
