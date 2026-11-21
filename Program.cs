using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using FactoryLeadTimePrediction.LightGBM.A_RealDataLeadTime;
using FactoryLeadTimePrediction.LightGBM.B_SyntheticLeadTime;
using FactoryLeadTimePrediction.DeepLearning.A_CastingDefectClassification;
using TorchSharp;
using static TorchSharp.torch;

namespace FactoryLeadTimePrediction
{

    class Program
    {
        static void Main(string[] args)
        {
            // ---------- 00_LeadTimeTest : 더미데이터 LightGBM 학습/예측/기여도 (완료) ----------
            // var lightGBMInstance01 = LightGBM_LeadTimeTest.Instance();
            // lightGBMInstance01.StartLightGBM_LeadTimeTest();

            // ---------- A_RealDataLeadTime : 실데이터 CSV 전처리 (1단계) ----------
            //var realDataInstance = LightGBM_RealDataLeadTime.Instance();
            //realDataInstance.StartRealDataLeadTime();

            
            // ---------- A_RealDataLeadTime : 학습 및 평가 (2단계) ----------
            //var realDataInstance = LightGBM_RealDataLeadTime.Instance();
            //realDataInstance.StartRealDataTraining();

            // ---------- B_SyntheticLeadTime : 파이프라인 검증 ----------
            //var syntheticInstance = LightGBM_SyntheticLeadTime.Instance();
            //syntheticInstance.StartSyntheticLeadTime();

            // ---------- DeepLearning A : 주조품 양/불량 분류
            //var castingInstance = DeepLearning_CastingDefect.Instance();
            //castingInstance.StartCastingDefectClassification();

            // ---------- DeepLearning A : 512x512 추가 검증 ----------
            var castingInstance = DeepLearning_CastingDefect.Instance();
            castingInstance.StartExternalValidation();



        }
    }
}