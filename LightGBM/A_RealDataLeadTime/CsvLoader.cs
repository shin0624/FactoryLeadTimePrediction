using System;
using System.Collections.Generic;
using System.Linq;
using System.Globalization;
using System.IO;
using System.Text;


namespace FactoryLeadTimePrediction.LightGBM.A_RealDataLeadTime
{
    public class CsvLoader // 원본 csv 파일을 StreamReader로 파싱하여 List<RawDTO>형태로 만들어 LightGBM_RealDataLeadTime.cs에 던지는 클래스.
    {
        private const int ExpectedColumnCount = 26; // 원본 csv의 컬럼 수. 파싱 결과가 이 값과 다르면 분할이 잘못된 것이므로 즉시 잡아낸다.
        private const int ReleaseDateStartIdx = 6;  // 1차 출고날짜 컬럼 인덱스 (0-base)
        private const int ReleaseAmountStartIdx = 16; // 1차 출고량 컬럼 인덱스 (0-base)
        private const int ReleaseCount = 10; // 출고 차수 (1~10차)

        private readonly string[] _dateFormats = {"yyyy-MM-dd H:mm", "yyyy-MM-dd HH:mm", "yyyy-MM-dd"}; // 공통 날짜 포맷 정의(생산일자, 시작종료일, 출고날짜) -> TryParseExact()으로 안전한 파싱을 하기 위해, 공통 포맷을 정의한다.
        // "H:mm"(1자리 시) 와 "HH:mm"(2자리 시)를 모두 넣은 이유 : 실데이터에 "2021-01-13 7:10"(149건)과 "2021-01-13 10:40"(205건)이 섞여 있음.
        private readonly CultureInfo _culture = CultureInfo.InvariantCulture; // 특정 문화권에 종속되지 않는 중립적·독립적인 표준 문화권 날짜표기방식
        private readonly NumberStyles _numStyle = NumberStyles.Number | NumberStyles.AllowThousands; // 문자열을 숫자로 변환할 때, 해당 문자열에 어떤 형식의 요소가 포함될 수 있는지 지정하는 NumberStyles 열거형. 일반적인 숫자와 천단위 구분기호(콤마) 허용

        public List<RawProductionRecord> LoadRawProductionRecord(string filePath)//csv 원본 파일을 읽어 파싱하는 메서드
        {
            var rawList = new List<RawProductionRecord>();

            if(!File.Exists(filePath)) // 파일 경로 오류 처리
            {
                throw new FileNotFoundException($"CSV파일을 찾을 수 없습니다. : {filePath}");
            }

            int lineNumber = 1; // 헤더가 1번째 줄. 경고 메시지에 파일상의 실제 줄번호를 찍기 위한 카운터.
            int skipped = 0;

            using (var reader = new StreamReader(filePath)) // using 블록으로 StreamReader 객체의 사용이 끝난 후 안전하게 해제한다.
            // StreamReader는 UTF-8 BOM(EF BB BF)을 자동 감지하여 제거한다. 원본은 CP949였으나 UTF-8(BOM)으로 재저장했으므로 별도 인코딩 지정이 불필요.
            {
                string? header = reader.ReadLine(); // ReadLine()의 반환형은 string? (파일 끝에서 null)
                if(header == null)
                {
                    return rawList;
                }

                string? line;

                while((line = reader.ReadLine())!=null)
                {
                    lineNumber++;

                    if(string.IsNullOrWhiteSpace(line)) continue;

                    var cols = SplitCsvLine(line); // 따옴표를 인식하는 분할. line.Split(',')을 쓰면 안 되는 이유는 아래 SplitCsvLine() 주석 참고.

                    if(cols.Length != ExpectedColumnCount) // 느슨한 검사(< 6) 대신 정확히 26개인지 확인한다.
                    {
                        Console.WriteLine($"[경고] {lineNumber}번째 줄 : 컬럼 수가 {cols.Length}개 (기대값 {ExpectedColumnCount}개) -> 건너뜀");
                        skipped++;
                        continue;
                    }

                    var raw = new RawProductionRecord();

                    // 기본 필드 파싱
                    raw.ProductionDate = ParseDateTime(cols[0], lineNumber, "생산일자"); // 생산일자
                    raw.OrderNumber = long.TryParse(cols[1].Trim(), out long orderNum) ? orderNum : 0L;
                    raw.ItemCode = string.IsNullOrWhiteSpace(cols[2]) ? "UNKNOWN" : cols[2].Trim();
                    raw.StartTime = ParseDateTime(cols[3], lineNumber, "시작");//시작시각
                    raw.EndTime = ParseDateTime(cols[4], lineNumber, "종료"); // 종료시각
                    raw.Production = ParseFloat(cols[5], 0.0f);// 생산량(kg)

                    // 1~10차 출고날짜 배열 파싱
                    for(int i=0; i<ReleaseCount; i++)
                    {
                        int dateColIdx = ReleaseDateStartIdx + i;

                        if(dateColIdx < cols.Length) // 배열 인덱스는 0 ~ Length-1 이므로 '<' 가 맞다. '<=' 면 cols[cols.Length]에 접근해 IndexOutOfRangeException.
                        {
                            raw.ReleaseDates[i] = ParseNullableDateTime(cols[dateColIdx]);
                        }
                    }

                    // 1~10차 출고량 배열 파싱
                    for(int i=0; i<ReleaseCount; i++)
                    {
                        int amountColIdx = ReleaseAmountStartIdx + i;

                        if(amountColIdx < cols.Length) // 검사 대상과 실제 접근 인덱스를 동일하게 맞춘다. (상수를 비교하면 검사가 무의미해짐)
                        {
                            raw.ReleaseAmounts[i] = ParseFloat(cols[amountColIdx], 0.0f);
                        }
                    }
                    rawList.Add(raw);
                }

            }
            Console.WriteLine($"csv 파싱 완료 : {rawList.Count}행 적재" + (skipped > 0 ? $" (건너뛴 행 {skipped}개)" : ""));
            return rawList; // 파싱된 리스트를 리턴.

        }

        public List<LeadTimeFeature> TransformToFeatures(List<RawProductionRecord> raws) // 파싱된 List<RawProductionRecord>를 기반으로 당일 생산 건수를 집계하고, LightGBM 모델 입력용 Feature으로 변환하는 메서드.
        {
            var featureList = new List<LeadTimeFeature>();

            // [1차 패스] 같은 날 생산 건수(SameDayOrderCount)를 구하기 위한 딕셔너리 필드 선언 -> ProductionDate의 Date 부분만 추출하여 그룹화 후 개수를 센다.
            // SameDayOrderCount는 한 행만 봐서는 계산할 수 없고 전체를 훑어야 하므로 2-pass 구조가 된다.
            var orderCountDictionary = raws.GroupBy(r => r.ProductionDate.Date).ToDictionary(g => g.Key, g => g.Count());

            int droppedNoRelease = 0; // 1차 출고날짜가 없어 리드타임을 계산할 수 없는 행 수

            // [2차 패스] 각 행을 Feature로 변환
            foreach(var raw in raws)
            {
                var firstRelease = raw.ReleaseDates[0]; // 리드타임 정의 : 생산일자 -> 1차 출고

                if(firstRelease == null) // 1차 출고날짜가 비어 있으면 Label을 만들 수 없으므로 학습 대상에서 제외한다. (실데이터에 1건 존재 : 작업지시번호 2205040847)
                {
                    droppedNoRelease++;
                    continue;
                }

                var feature = new LeadTimeFeature
                {
                    // Label : 리드타임(일). 날짜끼리의 차이이므로 .Date로 시각을 떼고 계산해 소수점이 섞이지 않게 한다.
                    LeadTimeDays = (float)(firstRelease.Value.Date - raw.ProductionDate.Date).TotalDays,

                    // Features : 생산 시점까지 확정된 정보만
                    ProductionKg = raw.Production,
                    ProcessMinutes = (float)(raw.EndTime - raw.StartTime).TotalMinutes, // 시각까지 포함해 계산해야 값이 나온다. 날짜만 빼면 전부 0.
                    WeekdayIndex = (float)(int)raw.ProductionDate.DayOfWeek, // DayOfWeek 열거형 -> int -> float. .NET은 일요일=0.
                    Month = raw.ProductionDate.Month,
                    SameDayOrderCount = orderCountDictionary[raw.ProductionDate.Date], // 1차 패스에서 만든 딕셔너리 참조
                    StartHour = raw.StartTime.Hour
                };

                featureList.Add(feature);
            }

            Console.WriteLine($"Feature 변환 완료 : {featureList.Count}행" + (droppedNoRelease > 0 ? $" (1차 출고날짜 결측으로 제외한 행 {droppedNoRelease}개)" : ""));
            return featureList;
        }


        private string[] SplitCsvLine(string line) // 따옴표(")로 감싸인 필드 안의 콤마를 구분자로 보지 않는 CSV 분할 메서드.
        {
            /*
            왜 line.Split(',') 을 쓸 수 없는가
                - 원본 csv의 숫자 필드는 천단위 콤마 때문에 따옴표로 감싸여 있다. 예: "13,550"
                - Split(',')으로 첫 행을 쪼개면 26개가 아니라 31개 필드가 나온다.
                    cols[5] = "\"13"  /  cols[6] = "550\""   <- 생산량이 둘로 쪼개지고
                    cols[7] 이후 모든 출고날짜/출고량이 밀려 엉뚱한 값이 들어간다.
                - 결과적으로 생산량은 파싱 실패로 0이 되고, Label(리드타임)도 완전히 틀어진다.

            동작 원리
                - 문자를 하나씩 순회하며 inQuotes 상태를 토글한다.
                - 따옴표 "밖"의 콤마에서만 필드를 끊는다. 따옴표 "안"의 콤마는 그냥 글자로 취급.
                - 연속된 두 개의 따옴표("")는 CSV 표준에서 escape된 따옴표 한 글자를 의미하므로 그렇게 처리한다.
            */
            var fields = new List<string>();
            var sb = new StringBuilder();
            bool inQuotes = false;

            for(int i=0; i<line.Length; i++)
            {
                char c = line[i];

                if(c == '"')
                {
                    if(inQuotes && i + 1 < line.Length && line[i+1] == '"') // "" -> escape된 따옴표 한 글자
                    {
                        sb.Append('"');
                        i++; // 두 번째 따옴표를 건너뛴다
                    }
                    else
                    {
                        inQuotes = !inQuotes; // 따옴표 구간 진입/이탈
                    }
                }
                else if(c == ',' && !inQuotes) // 따옴표 밖의 콤마 = 진짜 필드 구분자
                {
                    fields.Add(sb.ToString());
                    sb.Clear();
                }
                else
                {
                    sb.Append(c);
                }
            }
            fields.Add(sb.ToString()); // 마지막 필드

            return fields.ToArray();
        }

        private DateTime ParseDateTime(string input, int lineNumber, string columnName) // 데이트타임 파싱 헬퍼메서드
        {
            if(DateTime.TryParseExact(input?.Trim(), _dateFormats, _culture,  DateTimeStyles.None, out DateTime dt)) // TryParseExact()는 문자열을 지정한 날짜 형식에 맞춰 안전하게 DateTime 객체로 변환하는 메서드. 변환 성공 시 true, 실패해도 예외를 던지지 않고 false를 반환하여 프로그램이 멈추지 않음.
            {
                return dt;
            }

            /*
            생산일자 / 시작 / 종료는 실데이터에 결측이 없는 컬럼이다. 즉 여기서 실패하면 포맷 정의나 분할 로직이 틀린 것.
            조용히 DateTime.MinValue(0001-01-01)를 반환하면 리드타임이 -737000일 같은 값이 되어 원인 추적이 어려워지므로,
            값은 반환하되 경고를 남겨 즉시 알아차릴 수 있게 한다.
            */
            Console.WriteLine($"[경고] {lineNumber}번째 줄 : '{columnName}' 날짜 파싱 실패 (입력값 : \"{input}\")");
            return DateTime.MinValue; // 기본값 리턴
        }

        private DateTime? ParseNullableDateTime(string input) // Null 파싱 헬퍼 메서드
        {
            if(string.IsNullOrWhiteSpace(input)) // 출고날짜는 빈 문자열(",,,,")이 정상이다. 5차 이후는 대부분 공백.
            {
                return null;
            }

            if(DateTime.TryParseExact(input.Trim(), _dateFormats, _culture, DateTimeStyles.None, out DateTime dt))
            {
                return dt;
            }
            return null;
        }

        private float ParseFloat(string input, float defaultValue) // float 파싱 헬퍼 메서드
        {
            if(string.IsNullOrWhiteSpace(input))
            {
                return defaultValue; // 빈 출고량은 0. "출고 없음"과 "출고량 0"은 의미가 같으므로 0으로 둬도 안전하다.
            }
            if(float.TryParse(input.Trim(), _numStyle, _culture, out float result)) // _numStyle에 AllowThousands가 있으므로 "13,550"을 Replace 없이 바로 파싱한다.
            {
                return result;
            }
            return defaultValue;
        }
    }
}
