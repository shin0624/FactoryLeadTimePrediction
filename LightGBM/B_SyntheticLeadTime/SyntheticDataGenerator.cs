using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace FactoryLeadTimePrediction.LightGBM.B_SyntheticLeadTime//규칙에 따른 합성 데이터 생성 스크립트.
{
    public class SyntheticDataGenerator
    {
        // 생성 규칙 상수 (기여도 분석 결과와 대조할 정답. )
        // 초기 절편 15, 계수 0.6 / 1.2 / -1.5는 sigma가 커질 때 리드타임이 -30일까지 내려갔었기에, 클리핑이 과하게 발동하였었음. 이에 리드타임 최소 신호값이 2.2일(+)이 되도록 절편을 8.0으로 조정.
        private const float Intercept = 8.0f; // 절편(기준값). 모든 피처가 0일 때의 리드타임 출발점
        private const float CoefDueDays = 0.8f;// CustomerDueDays(거래처 납기 여유일)의 계수 (+)
        private const float CoefStockLevel = 1.5f; // CustomerStockLevel(거래처 재고 수준)의 계수 (+)
        private const float CoefShipFreq = -1.2f; // ShippingFrequency(주당 운송 편성 횟수)의 계수 (-)
        private const float MinLeadTime = 1.0f; // 리드타임 최소값. 음수 리드타임은 불가하므로 클리핑해야 함.

        public List<SyntheticLeadTimeFeature> Generate(int rowCount, float sigma, int seed)//합성 데이터 생성 메서드
        {
            var list = new List<SyntheticLeadTimeFeature>(rowCount); // 용량을 rowCount로 미리 지정하면 List 내부 배열 재할당이 줄어들어 속도 향상. (List<T>는 내부적으로 배열을 사용하며, 용량이 부족하면 새 배열을 할당하고 기존 데이터를 복사함)

            var rnd = new Random(seed); // seed를 받아 고정하여 재현 가능한 난수 생성. (동일 seed로 동일 난수열 생성)

            for(int i=0; i<rowCount; i++)
            {
                // 진짜 피처 3개 생성
                float dueDays = (float)(1.0 + rnd.NextDouble() * 29.0); // Uniform(1,30) 범위. NextDouble()은 0.0 이상 ~ 1.0 미만을 반환하므로, 1 + (0~29) = 1~30 범위가 됨.
                float stockLevel = (float)(rnd.NextDouble() * 10.0); // Uniform(0, 10) 범위. 하한이 0이므로 곱셈만 하면 됨.
                float shipFreq = (float)(1.0 + rnd.NextDouble() * 6.0); // Uniform(1,7) 범위. 주당 운송 편성 횟수이므로, 최소 1회는 있다고 가정함.

                // 노이즈 피처 3개 생성
                float productionKg = (float)NextGaussian(rnd, 14000.0, 1000.0); // Normal(14000, 1000) 범위. 평균 14000, 표준편차 1000인 정규분포 난수 생성
                float processMinutes = (float)NextGaussian(rnd, 194.0, 26.0); // Normal(194, 26) 범위. 평균 194, 표준편차 26인 정규분포 난수 생성
                float weekdayIndex = rnd.Next(0, 5); // Uniform(0, 4) 범위. Next(min, max)는 min 이상 max 미만 범위의 정수 난수를 반환하므로, 0~4 범위를 얻으려면 Next(0, 5)로 호출해야 함.

                // 생성 규칙에 따라 Label 계산
                float signal = Intercept + (CoefDueDays * dueDays) + (CoefStockLevel * stockLevel) + (CoefShipFreq * shipFreq); // 노이즈를 뺀 순수 신호 부분. 어떤 모델이든 이론상 이 부분까지만 설명 가능. 노이즈 3개는 식에 등장하면 안됨(계수 0)

                float noise = (float)NextGaussian(rnd, 0.0, sigma); // 평균 0, 표준편차 sigma인 정규분포 노이즈. 현실의 설명되지 않는 변동을 모사함. 이 노이즈때문에 RSquared에 이론적 상한이 생김.

                float leadTime = Math.Max(MinLeadTime, signal + noise); // 신호 + 노이즈를 합산한 값이 리드타임. Math.Max로 1.0일 미만은 잘라냄. 클리핑이 자주 발생하면 분포가 왜곡되므로, 앞서 계수를 조정하여 신호 최소값이 2.2일이 되게 만들었음.

                // DTO에 담아 List에 추가
                list.Add(new SyntheticLeadTimeFeature
                {
                    LeadTimeDays = leadTime,
                    CustomerDueDays = dueDays,
                    CustomerStockLevel = stockLevel,
                    ShippingFrequency = shipFreq,
                    ProductionKg = productionKg,
                    ProcessMinutes = processMinutes,
                    WeekdayIndex = weekdayIndex
                });
            }
        
            Console.WriteLine($"합성 데이터 생성 완료 : {list.Count}행 (sigma={sigma}, seed={seed})");
            return list;
        }

        private double NextGaussian(Random rnd, double mean, double stdDev)// 전규분포 난수를 생성하는 메서드.
        {
            // .NET의 Random 클래스에는 정규분포 난수 메서드가 없음(균등분포 NextDouble()만 존재)
            // 그렇기에, Box-Muller 변환을 사용하여 균등분포 난수 2개를 정규분포난수 1개로 만든다.
            // z = sqrt(-2 * ln(u1)) * cos(2 * pi * u2) -> z ~ N(0,1)

            double u1 = 1.0 - rnd.NextDouble(); // (0, 1] 범위의 균등분포 난수. Nextdouble()는 [0, 1) 범위인데, 0.0을 반환할 수 있음. 그렇게 되면 "ln(0) = -무한대"가 되어버려 계산 깨짐
            double u2 = 1.0 - rnd.NextDouble();

            double standardNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); // Box-Muller 공식. Sin이나 Cos 둘 중 어느쪽을 써도 N(0,1) 표준정규분포 난수로 나옴.

            return mean + stdDev * standardNormal; // 표준정규분포 N(0,1)을 원하는 평균(mean) , 표준편차(stdDev)로 선형 변환
        }

        public float[] GetTrueCoefficients() // 기여도 검증용으로 정답 계수를 넘겨주는 메서드.
        {
            return new[] {CoefDueDays, CoefStockLevel, CoefShipFreq, 0.0f, 0.0f, 0.0f};// 순서는 Concatenate()에 넣을 피처 순서와 반드시 일치시켜야 함. 뒤에 3개 0.0f는 노이즈.
            // 이 배열이 있어야 기여도 부호 및 크기 순서가 정답과 맞는지 확인 가능.
        }


    }
}