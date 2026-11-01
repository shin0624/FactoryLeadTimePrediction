using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using FactoryLeadTimePrediction.LightGBM.A_RealDataLeadTime;

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
            var realDataInstance = LightGBM_RealDataLeadTime.Instance();
            realDataInstance.StartRealDataTraining();
        }
    }
}