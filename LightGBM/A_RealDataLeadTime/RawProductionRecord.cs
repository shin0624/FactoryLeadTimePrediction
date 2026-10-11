using System;

namespace FactoryLeadTimePrediction.LightGBM.A_RealDataLeadTime
{
    public class RawProductionRecord // 원본데이터 csv를 파싱해서 담을 DTO. ML.NET의 DTO가 아니라 순수 C# DTO라서 [ColumnName]을 쓰면 안되고, 원본 csv를 그대로 담는 RawDTO 역할.
    {
        public DateTime ProductionDate {get;set;}// 생산일자(생산 지시가 배정된 날짜.)
        public long OrderNumber{get;set;} // 작업지시번호 (10자리라서 int면 오버플로우남)
        public string ItemCode {get;set;} = "M21"; // 제품코드
        public DateTime StartTime{get;set;} // 생산 시작 시각 (20xx - 01 - 01 HH : MM) 날짜 + 시각 (설비가 실제로 돌기 시작한 시각)
        public DateTime EndTime {get;set;} // 생산 종료 시각 (20xx - 01 - 01 HH : MM) 날짜 + 시각 --> 공정 소요 시간 피처를 뽑을 떄, End - Start로 시각까지 포함하여 계산할것.
        public float Production{get;set;} // 생산량(kg)
        public DateTime?[] ReleaseDates{get;set;} = new DateTime?[10];// 1~10차 출고날짜 (배열로 csv의 1~10차 출고날짜를 받는다.)
        public float[] ReleaseAmounts {get;set;} = new float[10];// 1~10차 출고량 (배열로 csv의 1~10차 출고량을 받는다.). -> [0] :1차/ [1] : 2차 [2] 3차 출고일 -> ReleaseDates와 한 쌍이 됨
        
        /*
        DateTime[]? : 배열 전체가 null일 수 있음. 각 원소는 null 불가
        DateTime?[] : 배열은 존재하고, 각 원소가 null일 수 있음.
        DateTime?[]? : 둘 다 가능함. 
        --> 왜 DateTime[]으로는 안되나?        
            - 출고 회차는 1~10차로 가변임. 실데이터에서도 5차 이후로는 대부분 공백. 그래서 원소별로 null이 필요.
            - DateTime[]으로 하면 빈 칸에 DateTime.MinValue(0001-01-01)같은 값을 넣어야 하는데, 이건 출고 없음과 구분이 안되어서 버그 발생 가능.
            - ReleaseAmounts는 float[]?으로 해도 됨. 빈칸을 0으로 채우면 출고량 0과 의미가 같으니.
        --> 그냥 깔끔하게 10으로 초기화. ReleaseDates는 전부 10칸 전부 null로 초기화되고, ReleaseAmounts는 10칸 전부 0으로 초기화. 차수는 정해져 있으니 이게 깔끔함. 
        */

        /*
        Raw DTO : 원본 데이터를 그대로 담는 역할
        Feature DTO : AI 모델 입력으로 사용할 데이터를 파싱해서 담는 역할.
            - 1차 출고량, 출고 분할 횟수는 Feature로 쓸 수 없음.
            - 생산이 끝난 시점에서 이 물량이 몇번에 나누어 출고될지, 1차 출고에 몇 kg이 나갈지는 모르며, 리드타임과 동시에 결정되는 값임./
            - 이를 데이터 누출(Date Leakage)라고 하며, 예측시점에 알 수 없는 정보를 피처로 쓰는 것을 의미함.
            - 출고 관련 컬럼은 Label 계산에만 쓰고, Feature에는 넣지 않는다.
            - Feature에 넣는 것은 생산 시점까지 확정된 것만        
        
        리드타임 정의 : 1차 출고 날짜 - 생산 일자 (ReleaseDates[0] - ProductionDate)
        */
    }
}