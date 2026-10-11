using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

namespace FactoryLeadTimePrediction
{

    class Program
    {
        static void Main(string[] args)
        {
            var lightGBMInstance01 = LightGBM_LeadTimeTest.Instance();
            lightGBMInstance01.StartLightGBM_LeadTimeTest();
        }
    }
}