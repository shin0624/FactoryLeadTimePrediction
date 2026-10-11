using Microsoft.ML.Data;

namespace FactoryLeadTimePrediction.LightGBM.A_RealDataLeadTime
{
    public class LeadTimeFeature// LightGBM에 입력할 값들을 담는 DTO. RawProductionRecord.cs의 데이터들을 전부 float로 변환
    {
        /*
        1. ML.NET 알고리즘은 전부 float만 받기에, 원본 데이터 DTO의 값들을 float 스칼라로 평탄화 해야 함. 
        2. 데이터 누출을 유발하는 출고 관련 컬럼들(생산 시점에 확인 불가한 데이터들)은 넣지 않아야 함.
        AI 입력 Feature : 공정 소요시간(EndTime - StartTime), 월, 요일, 생산량, 같은 날 생산 건수 + Label(ai가 예측할 대상)
        [LoadColumn]은 붙이지 않는다.-> [LoadColumn]은 LoadFromTextFile()로 CSV를 직접 읽을 때 컬럼 인덱스를 알려주는 속성임. 
        LoadFromTextFile()을 쓰면 프로퍼티 이름이 그대로 컬럼 명이 됨.  라벨에는 [ColumnName]을 붙여야 함. -> Evaluate(), CrossValidate()가 Label이라 이름붙은 프로퍼티를 기본값으로 찾아주기 때문.

        - 원본 csv는 아래의 문제때문에 LoadFromTextFile()을 쓸 수 없어서, LoadFromEnumerable()으로 읽어야 함.
            - 숫자에 콤마 + 따옴표가 있는 방식임("13,550") : csv 구분자가 콤마인데, 값 안에도 콤마가 있어서 따옴표로 감싼 구조가 됨.
            - 시간 포맷이 섞여있음 : DateTime.ParseExact에 "yyyy-MM-dd H:mm"을 쓰면 되긴 함
            - 생산 일자 != 시작 날짜 : 야간 교대로 날짜가 넘어가는 등의 경우가 있어서, 생산 일자와 시작 날짜가 다름.
            - 빈 문자열이 많음 : 3차 출고 이후 대부분의 컬럼이 비어있음. null이 아니라 빈 문자열임 ML.NET이 빈 값을 float로 읽으면 0이 아니라 NaN이 될 수 있는데, 0(출고 없음)과 NaN(데이터 없음)은 의미가 다르기에 LightGBM은 NaN을 결측으로 처리해버림.
        -> 해결 : CSV 파일 -> StreamReader 파싱 후 List<RawProductionRecord>형태로 만들고 LoadFromEnumerable()해야 함
        */

        public float ProductionKg {get;set;} // 생산량(kg)
        public float ProcessMinutes {get;set;} // 공정 소요 시간 (EndTime - StartTime).TotalMinutes
        public float DayOfWeek {get;set;} // 요일( (int)ProductionDate.DatOfWeek)
        public float Month {get;set;} // 월 (ProductionDate.Month)
        public float? SameDayOrderCount {get;set;} // 같은 날 생산 건수 (같은 생산일자 행 갯수)
        [ColumnName("Label")] public float LeadTimeDays {get;set;} // AI가 예측할 리드타임
    }
}