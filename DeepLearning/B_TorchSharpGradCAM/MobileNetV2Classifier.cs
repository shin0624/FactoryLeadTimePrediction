using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using System.IO;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;


namespace FactoryLeadTimePrediction.DeepLearning.B_TorchSharpGradCAM
{
    public class MobileNetV2Classifier : Module<Tensor, Tensor> // Module<입력타입, 출력타입>을 상속하는 모델 정의 클래스.
    {
         /*
        사전학습 가중치 출처
            https://huggingface.co/yueyinqiu/vision-TorchSharp
              -> MobileNet_V2_Weights.IMAGENET1K_V2  (13.5MB, TorchSharp .dat 포맷)
            - torchvision 공식 가중치를 TorchSharp 포맷으로 변환한 것. 원본은 BSD-3-Clause.
            - 변환본 자체에는 라이선스 표기가 없으므로, 재배포하지 않고 각자 다운로드하는 전제로 쓴다.
            - 가중치 파일은 .gitignore(models/ 폴더).
        */

        private readonly Module<Tensor, Tensor> features; // 마지막 컨볼루션레이어까지의 특징 추출부. Grad-CAM이 여기를 봄.
        private readonly Sequential classifier;           // 2클래스 분류층 (유일하게 학습되는 부분)

        public Module<Tensor, Tensor> Features => features;
        /*
        전체 백본(_backbone) 필드를 두지 않는 이유  ★진단으로 드러난 문제
            처음에는 _backbone 필드와 Features 자동프로퍼티를 함께 두었더니 진단 출력이 이렇게 나왔다 :
                _backbone                 : 파라미터 텐서 158개   <- features(156) + 1000클래스 분류층(2)
                <Features>k__BackingField : 파라미터 텐서 156개   <- _backbone 내부와 동일 객체 (중복 등록!)
                _classifier               : 파라미터 텐서   2개
                전체 파라미터 : 3,507,434   (기대 2,226,434보다 128만 개 많음)

            문제 1 : RegisterComponents()는 리플렉션으로 '필드'를 스캔하는데,
                     자동 구현 프로퍼티의 백킹 필드(<Features>k__BackingField)까지 읽는다.
                     그래서 같은 텐서가 _backbone 경로와 Features 경로로 두 번 등록됐다.
            문제 2 : forward()에서 호출조차 하지 않는 1000클래스 분류층(약 128만 파라미터)이
                     학습 대상에 남았다. 옵티마이저가 쓸모없는 그래디언트를 계산하게 된다.

            해결 : 가중치 로딩이 끝나면 features만 필드로 보관하고 전체 모델 참조를 놓는다.
                   외부 노출은 식 본문 프로퍼티(=> features)로 한다. 이것은 백킹 필드를 만들지 않으므로
                   중복 등록이 생기지 않는다. TorchSharp 모델 작성의 관례다.
        */

        public MobileNetV2Classifier(string? pretrainedPath, int numClasses = 2) : base(nameof(MobileNetV2Classifier))
        {
            // 사전학습 가중치의 classifier 층이 [1000, 1280]이므로, 1000클래스 구조로 만들어 가중치 로딩.
            // 처음부터 num_classes:2로 만들면 [2, 1280]이라 shape가 안맞아 로딩에 실패함. 처음에는 1000으로 만들어서 로딩한 뒤, 분류층만 2클래스로 교체하는 전이학습 표준절차를 따른다. ML.NET에서는 Trainer 객체가 알아서 했던 부분임
            var net = torchvision.models.mobilenet_v2(num_classes:1000);

            if(!string.IsNullOrEmpty(pretrainedPath) && File.Exists(pretrainedPath))
            {
                net.load(pretrainedPath);
                System.Console.WriteLine($" 사전학습 가중치 로딩 : {Path.GetFileName(pretrainedPath)}");
            }
            else
            {
                System.Console.WriteLine($" [경고] 사전학습 가중치 없음 -> 랜덤 초기화로 진행");
            }

            // features만 꺼내어 보관하고, net(전체 모델) 참조는 놓는다.
            features = (Module<Tensor, Tensor>)net.named_children().First(c=>c.name == "features").module;
            /*
            왜 Features와 Classifier를 분리하는가? -> Grad-CAM의 전제
            - Grad-CAM은 마지막 컨볼루션 레이어의 출력(feature map)과 그 출력에 대한 그라디언트가 필요
            - feature의 출력이 [batch, 1280, 7, 7]형태인 feature map에 해당
            - ML.NET의 ImageClassificationTrainer는 이 중간 출력에 접근할 방법이 없음.

            net을 필드에 담지 않는 이유
            - net은 지역 변수이므로 생성자가 끝나면 참조가 사라진다.
            - features는 net의 자식이지만 우리가 따로 참조를 들고 있으므로 살아남고,
              쓰지 않는 1000클래스 분류층은 GC 회수 대상이 된다.
            - 결과적으로 RegisterComponents()가 등록하는 것은 features와 classifier 두 개뿐이다.
            */

            // 분류층을 2클래스로 교체
            classifier = Sequential(
                ("dropout", Dropout(0.2)),
                ("fc", Linear(1280, numClasses))
            );
            /*
            1280 : MobileNetV2 features의 출력 채널 수. 아키텍처가 고정한 값
            Dropout(0.2) : torchvision의 MobileNetV2 원본 분류층과 같은 구성. 학습 시에만 작동하고, eval() 에서는 자동으로 꺼짐
            새로 만든 Linear는 학습으로 채워야 하는 유일한 부분(랜덤 초기화 상태)이며, 백본을 동결하면 이 층만 학습된다.
            */

            RegisterComponents(); 
            //TorchSharp 필수 호출 메서드.
            // 이 메서드를 빼면 parameters()가 비어서 옵티마이저가 아무것도 학습하지 않는다.
            // 예외가 나지 않고 loss가 전혀 줄지 않는 증상으로 나타날 수 있음.
            
        }

        public override Tensor forward(Tensor input) // 순전파 메서드 구현(입력 이미지 -> 클래스 로짓)
        {
            var feat = features.forward(input); // [B, 1280, 7, 7]
            var pooled = functional.adaptive_avg_pool2d(feat, new long[] {1, 1}); // [B, 1280, 1, 1]
            // 공간 차원을 1*1로 평균 폴링. 이미지 크기가 달라도 분류층 입력 크기가 고정됨.

            var flat = pooled.flatten(1); // [B, 1280]
            return classifier.forward(flat); // [B, 2]
            /*
            forward를 직접 쓰는 이유
            - 백본의 기본 forward를 쓰면 1000클래스 분류층을 통과해버림.
            - features -> pool -> 2클래스 분류층으로의 경로를 명시해야 2클래스 출력이 나옴.
            - 이렇게 쪼개어 두면 Grad-CAM에서 feat를 가로채기 쉬움 
            
            */
        }

        public Tensor ClassifierForward(Tensor pooledFlat) // 분류층만 통과시키는 메서드 [B, 1280] -> [B, 2]
        {
            return classifier.forward(pooledFlat);
            /*
            Grad-CAM 사용에 필요한 메서드
            - Grad-CAM은 features의 출력 텐서를 붙잡아두고 그 텐서에 흐른 그라디언트를 읽는다. 그런데 forward(input)을 호출하면 features를 다시 실행하므로, 분석을 위해 입력한 이미지의 텐서와는 다른 객체에 그라디언트가 흐른다.
            - 그래서 features 이후 경로(분류층)만 따로 호출할 수 있어야 함.
            
            */
        }


        public void FreezeBackbone(bool freeze = true)// 백본 동결/해제 메서드
        {
            foreach(var p in features.parameters())
            {
                p.requires_grad = !freeze;
                // freeze ? false : true 는 !freeze 와 같다. 짧은 쪽이 읽기 쉽다.
                /*
                동결 시
                - 역전파가 features를 건너뛰어 학습이 빠름(ml.net의 bottleneck 캐싱과 같은 효과)
                - 파라미터가 적어서 과적합이 어려움. 
                미동결 시
                - 주조품 이미지에 백본을 맞출 수 있어 성능이 더 오를 수 있음
                - 단, 학습 시간이 크게 늘고 과적합 위험이 생기므로, lr을 낮추어야 함
                먼저 동결로 ML.NET 결과 99.02%를 재현하고, 그 다음 해제를 비교하도록 함.
                
                */
            }
            System.Console.WriteLine($"  백본 {(freeze ? "동결" : "해제")} : features 텐서 {features.parameters().Count()}개 requires_grad = {!freeze}");
            // ★루프 밖에 둔다. 루프 안에 두면 파라미터 텐서 개수(156)만큼 같은 줄이 반복 출력된다.
        }
    }
}