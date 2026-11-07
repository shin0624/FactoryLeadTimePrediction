using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

namespace FactoryLeadTimePrediction.LightGBM.B_SyntheticLeadTime // 합성 데이터의 싱글톤 메인 스크립트(학습 -> 평가 -> 기여도 분석 -> 행수 실험)
{
    public class LightGBM_SyntheticLeadTime
    {
        private static LightGBM_SyntheticLeadTime? _instance; // static 필드는 CLR이 null로 초기화하므로 '?'가 사실에 맞는 선언. '?'는 컴파일타임 힌트일 뿐 런타임 동작/성능 영향 없음.
        private static LightGBM_SyntheticLeadTime Instance()// 반환형은 non-nullable : "Instance()를 부르면 null이 아닌 것을 받는다"는 계약.
        {
            if(_instance == null)
            {
                _instance = new LightGBM_SyntheticLeadTime();
            }
            return _instance;
        }

        private const int DefaultRowCount = 176; // 실데이터와 동일한 행 수.
        private const float DefaultSigma = 3.0f; // 기본 노이즈. 이때 RSquared 상한은 0.877
        private const int Seed = 42; // 난수 생성 시드

        // 기여도 배열 / 정답 계수와 순서를 맞추기 위한 피처 명.
        private static readonly string[] FeatureNames = {"납기여유일", "거래처재고", "운송빈도", "생산량(kg)", "공정시간(분)", "요일"};

        public void StartSyntheticLeadTime() // 합성 데이터 실습 전체를 실행하는 메서드
        {
            
        }

        private void RunSingleExperiment(int rowCount, float sigma) // 단일 조건 실험 메서드 : 학습 -> 평가 -> 교차검증 -> 기여도 분석 
        {
            
        }

        private IEstimator<ITransformer> BuildPipeline(MLContext mlContext) // 전처리 및 트레이너 파이프라인을 만드는 메서드
        {
            
        }

        private RegressionMetrics Evaluate(MLContext mlContext, ITransformer model, IDataView data, string label) // 학습된 모델을 평가하는 메서드
        {
            
        }

        private void RunCrossValidation(MLContext mlContext, IEstimator<ITransformer> pipeline, IDataView allData) // 교차검증을 수행하는 메서드
        {
            
        }

        private void AnalyzeContribution(MLContext mlContext, ITransformer model, IDataView trainData, List<SyntheticLeadTimeFeature> data, SyntheticDataGenerator generator)// 기여도 분석 및 정답 대조 메서드
        {
            
        }

        private void RunRowCountExperiment()// 행 수를 바꿔가며 과적합 변화를 관찰하는 실험 메서드
        {
            
        }

        private void RunSigmaExperiment() // 노이즈 크기를 바꿔가며 RSquared 상한 변화를 관찰하는 실험 메서드
        {
            
        }

        
    }
}