using System;
using System.Collections.Generic;
using System.Linq;
using System.Globalization;
using System.IO;


namespace FactoryLeadTimePrediction.LightGBM.A_RealDataLeadTime
{
    public class CsvLoader // 원본 csv 파일을 StreamReader로 파싱하여 List<RawDTO>형태로 만들어 LightGBM_RealDataLeadTime.cs에 던지는 클래스.
    {
        private readonly string[] _dataFormats = {"yyyy-MM-dd H:mm", "yyyy-MM-dd HH:mm", "yyyy-MM-dd"}; // 공통 날짜 포맷 정의(생산일자, 시작종료일, 출고날짜) -> TryParseExact()으로 안전한 파싱을 하기 위해, 공통 포맷을 정의한다.
        private readonly CultureInfo _culture = CultureInfo.InvariantCulture; // 특정 문화권에 종속되지 않는 중립적·독립적인 표준 문화권 날짜표기방식
        private readonly NumberStyles _numStyle = NumberStyles.Number | NumberStyles.AllowThousands; // 문자열을 숫자로 변환할 때, 해당 문자열에 어떤 형식의 요소가 포함될 수 있는지 지정하는 NumberStyles 열거형. 일반적인 숫자와 천단위 구분기호(콤마) 허용

        public List<RawProductionRecord> LoadRawProductionRecord(string filePath)//csv 원본 파일을 읽어 파싱하는 메서드
        {
            var rawList = new List<RawProductionRecord>();

            if(!File.Exists(filePath)) // 파일 경로 오류 처리
            {
                throw new FileNotFoundException($"CSV파일을 찾을 수 없습니다. : {filePath}");
            }

            using (var reader = new StreamReader(filePath)) // using 블록으로 StreamReader 객체의 사용이 끝난 후 안전하게 해제한다.
            {
                string header = reader.ReadLine();
                if(header == null)
                {
                    return rawList;
                }

                string line;

                while((line = reader.ReadLine())!=null)
                {
                    if(string.IsNullOrWhiteSpace(line)) continue;

                    var cols = line.Split(',');//콤마 기준 분리 및 공백 제거
                    if(cols.Length<6) continue;

                    var raw = new RawProductionRecord();

                    // 기본 필드 파싱
                    raw.ProductionDate = ParseDateTime(cols[0]); // 생산일자
                    raw.OrderNumber = long.TryParse(cols[1].Trim(), out long orderNum) ? orderNum : 0L;
                    raw.ItemCode = string.IsNullOrWhiteSpace(cols[2]) ? "UNKNOWN" : cols[2].Trim();
                    raw.StartTime = ParseDateTime(cols[3]);//시작시각
                    raw.EndTime = ParseDateTime(cols[4]); // 종료시각
                    raw.Production = ParseFloat(cols[5], 0.0f);// 생산량(kg)

                    // 1~10차 출고날짜 배열 파싱
                    int releaseDateStartIdx = 6;
                    for(int i=0; i<10; i++)
                    {
                        int dateColIdx =releaseDateStartIdx + i;
                        
                        if(dateColIdx <= cols.Length)
                        {
                            raw.ReleaseDates[i] = ParseNullableDateTime(cols[dateColIdx]);
                        }
                    }

                    // 1~10차 출고량 배열 파싱
                    int releaseAmountsStartIdx = 16;
                    for(int i=0; i<10; i++)
                    {
                        int amountColIdx = releaseAmountsStartIdx + i;
                        if(releaseAmountsStartIdx < cols.Length)
                        {
                            raw.ReleaseAmounts[i] = ParseFloat(cols[amountColIdx], 0.0f);
                        }
                    }
                    rawList.Add(raw);
                }

            }
            Console.WriteLine($"csv 파싱 완료");
            return rawList; // 파싱된 리스트를 리턴.

        }

        // public List<LeadTimeFeature> TransformToFeatures(List<RawProductionRecord> raws) // 파싱된 List<RawProductionRecord>를 기반으로 당일 생산 건수를 집계하고, LightGBM 모델 입력용 Feature으로 변환하는 메서드,.
        // {
        //     var featureList = new List<LeadTimeFeature>();

        //     // 같은 날 생산 건수(SameDayOrderCount)를 구하기 위한 딕셔너리 필드 선언 -> ProductionDate의 Date 부분만 추출하여 그룹화 후 개수를 센다.
        //     var orderCountDictionary = raws.GroupBy(r => r.ProductionDate.Date).ToDictionary(g => g.Key, g => g.Count());

        //     foreach(var raw in raws)
        //     {
                
        //     }
        // }



        private DateTime ParseDateTime(string input) // 데이트타임 파싱 헬퍼메서드
        {
            if(DateTime.TryParseExact(input?.Trim(), _dataFormats, _culture,  DateTimeStyles.None, out DateTime dt)) // TryParseExact()는 문자열을 지정한 날짜 형식에 맞춰 안전하게 DateTime 객체로 변환하는 메서드. 변환 성공 시 true, 실패해도 예외를 던지지 않고 false를 반환하여 프로그램이 멈추지 않음.
            {
                return dt;
            }
            return DateTime.MinValue; // 기본값 리턴
        }

        private DateTime? ParseNullableDateTime(string input) // Null 파싱 헬퍼 메서드
        {
            if(string.IsNullOrWhiteSpace(input))
            {
                return null;
            }

            if(DateTime.TryParseExact(input.Trim(), _dataFormats, _culture, DateTimeStyles.None, out DateTime dt))
            {
                return dt;
            }
            return null;
        }

        private float ParseFloat(string input, float defaultValue) // float 파싱 헬퍼 메서드
        {
            if(string.IsNullOrWhiteSpace(input))
            {
                return defaultValue;
            }
            if(float.TryParse(input.Trim(), _numStyle, _culture, out float result))
            {
                return result;
            }
            return defaultValue;
        }
    }
}