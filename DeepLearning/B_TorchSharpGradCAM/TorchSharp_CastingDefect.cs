using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using System.Reflection;


namespace FactoryLeadTimePrediction.DeepLearning.B_TorchSharpGradCAM
{
    public class TorchSharp_CastingDefect // Grad-CAM 구현을 위한 싱글톤 메인 클래스
    {
        private static TorchSharp_CastingDefect? _instance;
        public static TorchSharp_CastingDefect Instance() => _instance ??= new TorchSharp_CastingDefect(); // ??=는 "null 이면 대입"하라는 연산자. 기존 if(null) 패턴과 동일

        private const string DatasetRoot = @"C:\GitHubClone\FactoryLeadTimePrediction\data\Image_Classifiacation\casting_data";
        private const string PretrainedPath = "models/MobileNet_V2_Weights.IMAGENET1K_V2";
        private const int ImageSize = 300;
        /*
        리사이즈 제거 (300x300 원본 그대로)
            - MobileNetV2는 ImageNet 224x224로 학습됐지만, 전결합층이 없고 adaptive_avg_pool2d를 쓰므로 다른 입력 크기도 받는다.
            - feature map이 [1,1280,7,7] -> [1,1280,10,10] 이 된다. Grad-CAM 히트맵 해상도도 49칸 -> 100칸으로 올라간다.
            - 학습 시간은 (300/224)^2 = 1.8배 늘어 약 12분 예상.

            - 학습률을 10배 올리고(1e-3->1e-2), 감쇠를 넣고, R채널 버그를 고쳐도 test FN이 22건에서 움직이지 않았다.
            - val(96.53%)이 train(95.96%)보다 높다 = 과적합이 아니다. 즉 '더 학습하면 되는' 문제가 아니라 '입력에 정보가 없는' 문제다.
            - 주조 결함은 작은 점 형태다. 300->224 축소(면적 44% 손실)에서 그 점이 뭉개졌을 가능성이 가장 크다.
*/
        private const int Seed = 42;

        public void Step1_VerifyPipeline() // 데이터로더와 모델이 맞물려 도는지 일단 확인하는 메서드
        {
            System.Console.WriteLine("===========[TorchSharp] 파이프라인 검증===========");

            torch.manual_seed(Seed);

            var device = torch.CPU;
            System.Console.WriteLine($"디바이스 : {device.type}");
            System.Console.WriteLine($"CUDA 사용 가능 : {torch.cuda_is_available()}");
            // CPU 버전을 설치했기 때문에 GPU가 있어도 FALSE가 뜰 것. 이후 쿠다툴킷 환경 세팅 후 교체예정

            // 데이터셋 로딩
            System.Console.WriteLine();
            System.Console.WriteLine("========= test 폴더 로딩 ==========");
            var testSet = new CastingDataset(Path.Combine(DatasetRoot, "test"), ImageSize);

            var (defect, ok) = testSet.CountByLabel();
            System.Console.WriteLine($" 총 {testSet.Count}장 (불량 {defect} / 정상 {ok}");
            System.Console.WriteLine($" ML.NET 실습 기준 : 715장 (불량 453 / 정상 262)");

            // 텐서 1건 꺼내서 검산
            System.Console.WriteLine();
            System.Console.WriteLine("========== 텐서 변환 검산 ==========");
            var sample = testSet.GetTensor(0);
            var img = sample["data"];

            Console.WriteLine($"  shape : [{string.Join(",", img.shape)}]   (기대 : [3,{ImageSize},{ImageSize}])");
            Console.WriteLine($"  dtype : {img.dtype}   (기대 : Float32)");
            Console.WriteLine($"  범위  : {img.min().ToSingle():F3} ~ {img.max().ToSingle():F3}");
            Console.WriteLine($"  평균  : {img.mean().ToSingle():F4}  표준편차 : {img.std().ToSingle():F4}");
            Console.WriteLine($"  라벨  : {sample["label"].ToInt64()}  (0 = defect, 1 = ok)");
            /*
            정규화가 제대로 되었는지 읽는 법
            - ImageNet 정규화 후일 경우, 값의 범위가 대략 -2.2 ~ +2.7 사이 (0이 -mean/std ≈ -2.1이고, 1이 (1 - mean) / std ≈ +2.6)
            - 범위가 0 ~ 1이라면 정규화를 안한 것이고, 0 ~ 255이면 255로 나누기도 안한 것임
            - 주조품 이미지는 회색조에 가까워서 평균이 -1 근처로 치우칠 수 있음.(정상)
            - 정규화가 틀려도 예외가 나지 않고 조용히 성능이 떨어질 수 있으므로, 위 검산이 필요
            */

            // 모델 생성
            System.Console.WriteLine();
            System.Console.WriteLine("========== 모델 생성 ==========");
            var model = new MobileNetV2Classifier(PretrainedPath, numClasses : 2);
            model.to(device);

            // ---------- 파라미터 등록 진단 ----------
            Console.WriteLine();
            Console.WriteLine("---------- 파라미터 등록 진단 ----------");

            Console.WriteLine($"  parameters() 텐서 개수 : {model.parameters().Count()}   (기대: 158)");
            // 158 = features 156 + classifier 2.
            // 0이면 RegisterComponents()가 아무것도 등록하지 못한 것이고,
            // 160이면 1000클래스 분류층이 남아 중복 등록된 것이다(수정 전 상태).

            Console.WriteLine($"  named_children() 목록 : (기대: features / classifier 2개만)");
            foreach (var c in model.named_children())
            {
                int cnt = ((torch.nn.Module)c.module).parameters().Count();
                Console.WriteLine($"    - {c.name,-12} : 파라미터 텐서 {cnt}개");
            }
            /*
            무엇을 보려는가
                - RegisterComponents()는 리플렉션으로 '필드'를 스캔해 하위 모듈을 등록한다.
                  이때 자동 구현 프로퍼티의 백킹 필드(<Features>k__BackingField)까지 읽는다.
                - 수정 전에는 _backbone(158) + <Features>k__BackingField(156) + _classifier(2)가 모두 잡혀
                  같은 텐서가 두 경로로 등록되고, 쓰지 않는 1000클래스 분류층까지 학습 대상이 되었다.
                - 지금은 features / classifier 두 개만 나와야 정상이다.
            */

            long total = model.parameters().Sum(p => p.numel());
            Console.WriteLine($"  전체 파라미터     : {total:N0}   (기대: 2,226,434)");
            // 2,226,434 = features 2,223,872 + 우리 분류층 2,562
            // 3,507,434가 나오면 1000클래스 분류층(1,281,000)이 아직 붙어 있는 것이다.

            Console.WriteLine($"  학습가능(동결 전) : {model.parameters().Where(p => p.requires_grad).Sum(p => p.numel()):N0}   (기대: 2,226,434)");

            model.FreezeBackbone(true);

            Console.WriteLine($"  학습가능(동결 후) : {model.parameters().Where(p => p.requires_grad).Sum(p => p.numel()):N0}   (기대: 2,562)");
            /*
            동결 전후를 나란히 찍는 이유
                - 동결 후 값만 보면 FreezeBackbone이 실제로 걸렸는지 알 수 없다.
                  (애초에 전체가 2,562였다면 동결이 무의미하게 성공한 것처럼 보인다)
                - 2,226,434 -> 2,562 로 줄어드는 것을 확인해야 동결이 작동했다고 말할 수 있다.

            2,562가 나와야 하는 이유
                - 교체한 Linear(1280, 2)의 가중치 1280 x 2 = 2560 + bias 2 = 2562
                - 이것이 ML.NET 로그의 'Froze 2 variables'가 의미한 바와 같다.
                  전체 222만 파라미터 중 2,562개만 학습한다. 그래서 CPU로도 빠르고 과적합이 어렵다.
                - 숫자가 다르면 FreezeBackbone이나 RegisterComponents가 제대로 되지 않은 것이다.

            long으로 받는 이유 : p.numel()이 long을 반환하므로 (int) 캐스팅은 불필요하고 오버플로 위험도 있다.
            */

            // 순전파
            System.Console.WriteLine();
            System.Console.WriteLine("========== 순전파 테스트 ===========");
            
            model.eval();
            using (torch.no_grad())
            {
                var batch = img.unsqueeze(0).to(device); // [3, 224, 224] -> [1, 3, 224, 224] 배치 차원을 추가
                var logits = model.forward(batch);
                var probs = torch.nn.functional.softmax(logits, dim : 1);
                /*
                softmax : 주로 3개 이상의 선택지 중 정답을 고르는 다중 클래스 분류(Multi-class Classification) 문제의 최종 출력층에서 사용하는, 인공지능과 딥러닝에서 여러 개의 입력값을 '총합이 1이 되는 확률 분포'로 변환해 주는 수학 함수
                - 확률로 변환 : 입력된 임의의 실수(로짓, Logit)들을 모두 0과 1 사이의 값으로 바꿈
                - 총합은 항상 1임 :  변환된 모든 출력값을 더하면 정확히 1(100%)이 되므로, 각 항목의 결과값을 직관적인 '확률'로 해석할 수 있음
                - 큰 값 강조 (부드러운 최댓값) :  지수 함수(e^{x})를 사용하기 때문에 큰 입력값은 더 큰 확률로, 작은 입력값은 더 작은 확률로 차이를 벌려줌. 가장 큰 값 하나만 1로 고르는 일반 max 함수와 달리, 모든 후보의 비율을 부드럽게 반영한다는 뜻에서 'Softmax'라는 이름임
                */

                Console.WriteLine($"  로짓 shape : [{string.Join(",", logits.shape)}]   (기대 : [1,2])");
                Console.WriteLine($"  확률 : defect = {probs[0, 0].ToSingle():F4}  ok={probs[0, 1].ToSingle():F4}");
                Console.WriteLine($"  합계 : {probs.sum().ToSingle():F4}   (기대 : 1.0000)");
                Console.WriteLine($"  분류층이 랜덤 초기화 상태이므로 0.5 근처가 나오는 것이 정상");
                Console.WriteLine($"  학습하지 않았으므로 이 확률에는 의미가 없고, 구조가 도는지만 확인하는 단계");
            }

                // 백본이 살아있는지 확인
                System.Console.WriteLine();
                System.Console.WriteLine("========== 백본 feature map 확인 ==========");

                using (torch.no_grad())
                {
                    var batch = img.unsqueeze(0).to(device);
                    var feat = model.Features.forward(batch);

                    Console.WriteLine($"  feature map shape : [{string.Join(",", feat.shape)}]   (기대 : [1,1280,7,7])");
                    Console.WriteLine($"  평균 {feat.mean().ToSingle():F4}  표준편차 {feat.std().ToSingle():F4}  최대 {feat.max().ToSingle():F4}");
                    Console.WriteLine($"  표준편차가 0이면 백본이 죽은 것. 0.3 ~ 1.5 범위면 정상.");
                    Console.WriteLine($"  이 [1280, 7, 7] 텐서가 Grad-CAM의 입력. 7x7 격자가 히트맵의 해상도.");
                }


            Console.WriteLine();
            Console.WriteLine("========== [단계 1] 완료 ==========");
        }

        // ==================== 단계 2 : 학습 / 평가 ====================

        private const int Epochs = 20;          // 
        private const int BatchSize = 32;       // CPU 메모리와 속도의 타협. 배치가 크면 빠르지만 메모리를 더 쓴다.
        private const double LearningRate = 1e-2; // ML.NET의 LearningRate=0.01 과 동일하게 맞췄다.
        private const double LrDecayGamma = 0.94; // 에폭마다 학습률에 곱할 비율


        /*
        LrDecayGamma : 에폭마다 학습률에 곱할 비율 (0.01 -> 0.0094 -> 0.00884 -> ...)
        - ML.NET 학습 로그에 실제로 찍혀 있던 감쇠 :
                Epoch 0: Learning Rate 0.01
                Epoch 1: Learning Rate 0.0094
                Epoch 3: Learning Rate 0.008836
                Epoch 5: Learning Rate 0.008305839
              0.0094/0.01 = 0.94, 0.008836/0.0094 = 0.94 -> gamma 0.94 가 맞다.

        왜 감쇠가 필요한가  ★앞선 두 실험의 진단
            lr 1e-3 고정 : 10에폭 써도 val 97.44% (학습 부족, 걸음이 너무 작다)
            lr 1e-2 고정 : Epoch 0~1에 최고점 찍고 계속 하락 (최적점 주변 진동, 걸음이 너무 크다)
            -> 초기에는 큰 걸음으로 빠르게, 후반에는 작은 걸음으로 미세조정해야 한다.
               고정 학습률로는 둘을 동시에 만족할 수 없다.

        1e-3 -> 1e-2 로 올린 이유  ★실측 근거가 있는 변경
            1e-3 실행 결과 : 10에폭에 val 97.44% / test 97.76% / FN 12건
                             (ML.NET : Epoch 1에 이미 99.01%, 최종 test 99.02% / FN 2건)
            진단 : train acc 97.42% vs val acc 97.13% 로 거의 붙어 있었다.
                   과적합(train >> val)이 아니라 '학습 부족(underfitting)'이다.
                   2,562개 파라미터가 아직 수렴하지 못한 상태에서 에폭이 끝난 것.
            근거 : ML.NET 로그에는 LearningRate 0.01 -> 0.0073 으로 감쇠하는 기록이 있었다.
                   즉 ML.NET은 우리보다 10배 큰 학습률로 시작했다.
            Adam 1e-3은 '전체 네트워크 fine-tuning'의 표준값이다. 분류층만 학습할 때는 보수적이다.

        한 번에 하나만 바꾼다
            - Dropout 제거나 전처리 변경도 후보였지만 동시에 바꾸면 무엇이 기여했는지 알 수 없다.
            - B 실습에서 행 수와 sigma를 따로 실험했던 것과 같은 원칙이다.
        */
        private const double ValFraction = 0.1; // ML.NET 실습과 동일 (6,633 -> 5,970 / 663)
        private const int Patience = 3;         // 검증 정확도가 3에폭 연속 개선되지 않으면 중단
        private const string ModelSavePath = "models/TorchSharp_CastingDefect.dat";

        public void Step2_Train() // 단계 2 : 학습 루프를 직접 구현해 ML.NET의 99.02%를 재현한다.
        {
            Console.WriteLine("========== [TorchSharp 단계 2] 학습 및 평가 ==========");

            torch.manual_seed(Seed);
            var device = torch.CPU;

            // ---------- (1) 데이터 로딩 및 분할 ----------
            Console.WriteLine();
            Console.WriteLine("---------- train 폴더 로딩 ----------");
            var trainAll = new CastingDataset(Path.Combine(DatasetRoot, "train"), ImageSize);
            var (trainSet, valSet) = trainAll.SplitTrainVal(ValFraction, Seed);

            Console.WriteLine();
            Console.WriteLine("---------- test 폴더 로딩 ----------");
            var testSet = new CastingDataset(Path.Combine(DatasetRoot, "test"), ImageSize);

            Console.WriteLine();
            Console.WriteLine($"---------- 3분할 결과 ----------");
            Console.WriteLine($"  train : {trainSet.Count,5}장 (학습)        * ML.NET 기준 5,970");
            Console.WriteLine($"  val   : {valSet.Count,5}장 (에폭별 검증)   * ML.NET 기준   663");
            Console.WriteLine($"  test  : {testSet.Count,5}장 (최종 1회)     * ML.NET 기준   715");

            var (trDef, trOk) = trainSet.CountByLabel();
            var (vaDef, vaOk) = valSet.CountByLabel();
            Console.WriteLine($"  train 구성 : 불량 {trDef} / 정상 {trOk}");
            Console.WriteLine($"  val   구성 : 불량 {vaDef} / 정상 {vaOk}   <- 양쪽에 두 클래스가 모두 있어야 정상(셔플 확인)");
            /*
            val 구성을 반드시 확인하는 이유
                - 셔플을 빼먹으면 val이 전부 한 클래스가 된다. 그 상태로도 학습은 '돌아간다'.
                - 검증 정확도가 0% 또는 100%로 고정되는 증상으로 나타나며, 원인을 찾기 어렵다.
                - 숫자로 먼저 확인하는 것이 디버깅보다 싸다. A 실습의 검산 습관과 같다.
            */

            // ---------- (2) DataLoader ----------
            var trainLoader = new DataLoader(trainSet, BatchSize, shuffle: true, device: device, seed: Seed, disposeDataset: false);
            var valLoader = new DataLoader(valSet, BatchSize, shuffle: false, device: device, disposeDataset: false);
            var testLoader = new DataLoader(testSet, BatchSize, shuffle: false, device: device, disposeDataset: false);
            /*
            DataLoader가 하는 일 (ML.NET의 IDataView가 감췄던 부분)
                - Dataset.GetTensor()를 배치 크기만큼 호출해 [B,3,224,224]로 쌓아준다(collate).
                - shuffle: true면 에폭마다 순서를 섞는다. 같은 순서로 반복 학습하면
                  배치 단위 그래디언트에 주기성이 생겨 수렴이 나빠진다.
                - 학습셋만 셔플한다. 검증/테스트는 순서가 결과에 영향을 주지 않으므로 섞을 이유가 없고,
                  섞지 않으면 디버깅 시 같은 순서로 재현된다.

            타입이 TorchSharp.Modules.DataLoader 인 점에 주의
                - 처음에 utils.data.DataLoader 로 썼더니 CS0426(그 타입이 없음)이 났다.
                - PyTorch는 torch.utils.data.DataLoader 이지만, TorchSharp는 Modules 네임스페이스에 둔다.
                  (Dataset은 torch.utils.data.Dataset 이므로 둘의 위치가 다르다. 혼동하기 쉬운 지점)

            disposeDataset: false 를 반드시 넘긴다  ★기본값이 true라서 생기는 함정
                - DataLoader는 자신이 Dispose될 때 Dataset까지 함께 Dispose한다(기본값 true).
                - trainSet/valSet은 trainAll에서 분할한 것이고, 평가에서 trainLoader를 재사용하므로
                  어느 한쪽이 데이터셋을 해제하면 다른 쪽이 해제된 객체를 쓰게 된다.
                - 예외 메시지가 불친절해서 원인을 찾기 어렵다. 명시적으로 끄는 편이 안전하다.

            seed: Seed 를 넘기는 이유 : 셔플 순서까지 고정해 실행마다 같은 결과가 나오게 한다.
            */

            // ---------- (3) 모델 ----------
            Console.WriteLine();
            Console.WriteLine("---------- 모델 준비 ----------");
            var model = new MobileNetV2Classifier(PretrainedPath, numClasses: 2);
            model.to(device);
            model.FreezeBackbone(true);

            long trainable = model.parameters().Where(p => p.requires_grad).Sum(p => p.numel());
            Console.WriteLine($"  학습가능 파라미터 : {trainable:N0}   (기대: 2,562)");

            // ---------- (4) 손실함수와 옵티마이저 ----------
            var lossFn = torch.nn.CrossEntropyLoss();
            /*
            CrossEntropyLoss를 쓰는 이유
                - 내부에서 softmax + negative log likelihood를 함께 계산한다.
                  따라서 모델은 softmax를 거치지 않은 '로짓'을 그대로 출력해야 한다.
                  forward()에서 softmax를 걸면 이중 적용이 되어 학습이 느려진다. 흔한 실수다.
                - 라벨은 Int64 타입의 클래스 인덱스(0 또는 1)를 받는다. one-hot이 아니다.
            */

            var optimizer = torch.optim.Adam(
                model.parameters().Where(p => p.requires_grad), lr: LearningRate);
            /*
            requires_grad가 true인 파라미터만 넘기는 이유
                - 동결한 백본을 옵티마이저에 넘기면 쓸모없는 상태(모멘텀 등)를 유지해 메모리를 낭비한다.
                - 일부 옵티마이저는 requires_grad=false인 파라미터에서 예외를 던진다.
                - 결과적으로 2,562개만 옵티마이저가 관리한다.

            Adam을 쓰는 이유 : 학습률 민감도가 낮아 하이퍼파라미터 탐색 없이 쓰기 쉽다.
                               ML.NET의 ImageClassificationTrainer도 내부적으로 유사한 최적화를 쓴다.
            */

            var scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma: LrDecayGamma);
            /*
            ExponentialLR : 에폭마다 lr <- lr * gamma
                - StepLR(N에폭마다 감쇠)이나 MultiStepLR(지정 시점에 감쇠)도 있지만,
                  ML.NET 로그가 매 에폭 일정 비율로 줄어드는 형태였으므로 이것이 맞다.
                - scheduler는 optimizer를 참조해 그 안의 lr 값을 직접 바꾼다.
                  따라서 optimizer보다 먼저 만들 수 없다.
            */

            // ---------- (5) 학습 루프 ----------
            Console.WriteLine();
            Console.WriteLine($"---------- 학습 시작 (Epoch={Epochs}, Batch={BatchSize}, lr={LearningRate}) ----------");

            var sw = System.Diagnostics.Stopwatch.StartNew();
            double bestValAcc = 0;
            int epochsNoImprove = 0;

            for (int epoch = 0; epoch < Epochs; epoch++)
            {
                // ===== 학습 =====
                model.train();
                /*
                train() 모드가 바꾸는 것
                    - Dropout이 활성화된다(확률적으로 뉴런을 끈다).
                    - BatchNorm이 배치 통계로 정규화하고 running statistics를 갱신한다.
                    - eval()로 바꾸지 않으면 검증에서도 Dropout이 걸려 정확도가 낮게 나온다.
                      ★예외가 나지 않는 조용한 실패. 반드시 모드를 전환해야 한다.
                */

                double trainLoss = 0;
                long trainCorrect = 0, trainTotal = 0;
                int batchIdx = 0;

                foreach (var batch in trainLoader)
                {
                    using var d = torch.NewDisposeScope();
                    /*
                    DisposeScope가 필수인 이유  ★TorchSharp 고유의 문제
                        - 모든 Tensor는 네이티브 메모리를 쓰며 GC가 즉시 회수하지 않는다.
                        - 중간 텐서를 해제하지 않으면 수십 배치 안에 메모리가 수 GB로 불어난다.
                        - 이 스코프를 벗어날 때 안에서 만들어진 텐서가 모두 해제된다.
                        - ML.NET에서는 이런 수동 관리가 전혀 필요하지 않았다. 전환의 또 다른 비용이다.
                    */

                    var data = batch["data"].to(device);
                    var label = batch["label"].to(device);

                    optimizer.zero_grad();
                    // 그래디언트를 0으로 초기화한다. PyTorch/TorchSharp는 그래디언트를 '누적'하므로
                    // 이것을 빼면 이전 배치의 그래디언트가 더해져 학습이 발산한다.

                    var output = model.forward(data);       // [B, 2] 로짓
                    var loss = lossFn.forward(output, label);

                    loss.backward();   // 역전파 : 각 파라미터의 그래디언트를 계산
                    optimizer.step();  // 그래디언트 방향으로 파라미터를 갱신
                    /*
                    이 네 줄(zero_grad -> forward -> backward -> step)이 ML.NET의 Fit() 한 줄에 해당한다.
                    감춰져 있던 학습의 실체가 드러나는 지점이다.
                    */

                    trainLoss += loss.ToDouble() * data.shape[0];
                    var pred = output.argmax(1);
                    trainCorrect += pred.eq(label).sum().ToInt64();
                    trainTotal += data.shape[0];

                    batchIdx++;
                    if (batchIdx % 20 == 0)
                    {
                        Console.WriteLine($"\r    Epoch {epoch}: batch {batchIdx}/{(trainSet.Count + BatchSize - 1) / BatchSize}   ");
                        // \r로 같은 줄을 덮어쓴다. 진행 상황을 보여주되 로그가 길어지지 않게 한다.
                    }
                }

                double trainAcc = (double)trainCorrect / trainTotal;
                trainLoss /= trainTotal;

                // ===== 검증 =====
                var (valAcc, valLoss) = Evaluate(model, valLoader, lossFn, device);

                Console.Write("\r");
                Console.WriteLine($"    Epoch {epoch,2} | train loss {trainLoss:F4} acc {trainAcc:P2} | val loss {valLoss:F4} acc {valAcc:P2}");

                // ===== 조기 종료 =====
                if (valAcc > bestValAcc + 0.001)
                {
                    bestValAcc = valAcc;
                    epochsNoImprove = 0;
                    model.save(ModelSavePath);
                    Console.WriteLine($"             -> 검증 정확도 개선. 모델 저장.");
                    /*
                    개선될 때마다 저장하는 이유 (checkpointing)
                        - 마지막 에폭의 모델이 최선이라는 보장이 없다. 과적합이 시작되면 오히려 나빠진다.
                        - 검증 정확도가 가장 좋았던 시점의 가중치를 남기는 것이 정석이다.
                        - ML.NET의 EarlyStopping은 이것을 내부에서 처리했다.
                    */
                }
                else
                {
                    epochsNoImprove++;
                    if (epochsNoImprove >= Patience)
                    {
                        Console.WriteLine($"             -> {Patience}에폭 연속 개선 없음. 조기 종료.");
                        break;
                    }
                }

                scheduler.step();
                Console.WriteLine($"             lr -> {scheduler.get_last_lr().First():F6}");
                /*
                호출 위치가 중요
                    - 반드시 검증이 끝난 뒤에 호출. 루프 시작에 두면 첫 에폭부터 감쇠가 적용되어 초기 학습률이 0.01이 아니라 0.0094가 된다.
                    - optimizer.step()(배치마다, 파라미터 갱신)과 scheduler.step()(에폭마다, 학습률 갱신)은 완전히 다른 것. 이름이 같아 혼동하기 쉽다.
                    - break 다음에 두었으므로 조기 종료 시에는 호출되지 않는다.

                lr을 매 에폭 출력하는 이유
                    - 감쇠가 실제로 걸리는지 눈으로 확인해야 한다.
                    - scheduler.step()을 빼먹어도 예외가 나지 않고 lr이 그대로 유지된다.
                      ★또 하나의 조용한 실패 지점.
                */
            }

            sw.Stop();
            Console.WriteLine($"---------- 학습 완료 : {sw.Elapsed.TotalMinutes:F1}분 ----------");
            Console.WriteLine($"  * ML.NET 기준 2.5분. 느린 이유는 bottleneck 캐싱이 없어 매 에폭 백본 전체를 통과하기 때문이다.");

            // ---------- (6) 최종 평가 ----------
            model.load(ModelSavePath);
            // 검증 최고 시점의 가중치를 복원한다. 마지막 에폭 상태로 평가하면 안 된다.
            Console.WriteLine($"  최고 검증 정확도 시점의 모델 복원 (val acc {bestValAcc:P2})");

            EvaluateWithConfusionMatrix(model, trainLoader, device, "train (과적합 확인용)");
            EvaluateWithConfusionMatrix(model, testLoader, device, "test (최종 성능)");

            Console.WriteLine();
            Console.WriteLine("========== [단계 2] 완료 ==========");
        }

        private (double acc, double loss) Evaluate(MobileNetV2Classifier model, DataLoader loader,
            torch.nn.Module<Tensor, Tensor, Tensor> lossFn, Device device) // 정확도와 손실만 빠르게 계산
        {
            model.eval(); // Dropout 비활성화, BatchNorm을 running statistics 사용으로 전환
            long correct = 0, total = 0;
            double lossSum = 0;

            using (torch.no_grad())
            {
                /*
                no_grad()가 필요한 이유
                    - 추론에는 그래디언트가 필요 없다. 계산 그래프를 만들지 않으면 메모리와 시간이 절약된다.
                    - 이것을 빼면 평가 중에도 그래프가 쌓여 메모리가 급증한다.
                */
                foreach (var batch in loader)
                {
                    using var d = torch.NewDisposeScope();

                    var data = batch["data"].to(device);
                    var label = batch["label"].to(device);

                    var output = model.forward(data);
                    lossSum += lossFn.forward(output, label).ToDouble() * data.shape[0];

                    correct += output.argmax(1).eq(label).sum().ToInt64();
                    total += data.shape[0];
                }
            }

            return ((double)correct / total, lossSum / total);
        }

        private void EvaluateWithConfusionMatrix(MobileNetV2Classifier model, DataLoader loader,
            Device device, string title) // 혼동행렬 + Precision/Recall/F1 (ML.NET 실습과 같은 지표)
        {
            model.eval();
            long tp = 0, fn = 0, fp = 0, tn = 0;

            using (torch.no_grad())
            {
                foreach (var batch in loader)
                {
                    using var d = torch.NewDisposeScope();

                    var data = batch["data"].to(device);
                    var label = batch["label"].to(device);
                    var pred = model.forward(data).argmax(1);

                    // "불량"(label 0)을 Positive로 잡는다. 품질검사의 관심사가 불량 검출이기 때문이다.
                    var isDefectTrue = label.eq(CastingDataset.LabelDefect);
                    var isDefectPred = pred.eq(CastingDataset.LabelDefect);

                    tp += (isDefectTrue & isDefectPred).sum().ToInt64();
                    fn += (isDefectTrue & ~isDefectPred).sum().ToInt64();
                    fp += (~isDefectTrue & isDefectPred).sum().ToInt64();
                    tn += (~isDefectTrue & ~isDefectPred).sum().ToInt64();
                    /*
                    텐서 연산으로 집계하는 이유
                        - 배치 단위로 한 번에 세므로 샘플별 루프보다 빠르다.
                        - & 는 元소별 AND, ~ 는 元소별 NOT이다(bool 텐서 연산).
                        - ML.NET에서는 CreateEnumerable로 행을 꺼내 LINQ로 셌다. 결과는 같다.
                    */
                }
            }

            long totalCount = tp + fn + fp + tn;
            double precision = tp + fp > 0 ? (double)tp / (tp + fp) : 0;
            double recall = tp + fn > 0 ? (double)tp / (tp + fn) : 0;
            double f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
            double accuracy = (double)(tp + tn) / totalCount;

            Console.WriteLine();
            Console.WriteLine($"---------- 평가 : {title} ----------");
            Console.WriteLine("  [혼동행렬]            예측:불량   예측:정상");
            Console.WriteLine($"    실제:불량          {tp,8} (TP) {fn,7} (FN) <- 불량유출");
            Console.WriteLine($"    실제:정상          {fp,8} (FP) {tn,7} (TN)");
            Console.WriteLine();
            Console.WriteLine($"  Accuracy          : {accuracy,8:P2}");
            Console.WriteLine($"  Precision(불량)   : {precision,8:P2}");
            Console.WriteLine($"  Recall(불량)      : {recall,8:P2}   <- 주 지표");
            Console.WriteLine($"  F1(불량)          : {f1,8:P2}");
            Console.WriteLine($"  * ML.NET 기준(test) : Accuracy 99.02% / Recall 99.56% / FN 2건");
            Console.WriteLine($"    베이스라인(전부 불량) : 63.36%");
        }
    
        // ==================== 단계 3 : 오분류(FN/FP) 샘플 분석 ====================

        public void Step3_AnalyzeMisclassified() // 저장된 모델로 test를 1장씩 추론해 FN/FP의 파일명과 확률을 출력한다.
        {
            Console.WriteLine("========== [TorchSharp 단계 3] 오분류 샘플 분석 ==========");

            torch.manual_seed(Seed);
            var device = torch.CPU;

            // ---------- (1) 저장된 모델 로딩 ----------
            if (!File.Exists(ModelSavePath))
            {
                Console.WriteLine($"[중단] 모델 파일이 없습니다 : {ModelSavePath}");
                Console.WriteLine("       먼저 Step2_Train()을 실행하세요.");
                return;
            }

            var model = new MobileNetV2Classifier(PretrainedPath, numClasses: 2);
            model.load(ModelSavePath);
            model.to(device);
            model.eval();
            /*
            학습을 다시 하지 않는 이유
                - 목적은 "이미 만든 모델이 어떤 샘플을 놓치는가"를 보는 것이다. 재학습하면 질문이 달라진다.
                - eval() 호출이 중요하다. Dropout이 켜진 상태로 추론하면 같은 이미지에 매번 다른 확률이 나와
                  오분석 목록이 실행마다 바뀐다. ★예외가 나지 않는 조용한 실패.

            생성자에 PretrainedPath를 그대로 넘기는 이유
                - 구조를 먼저 만들어야 load()가 가중치를 넣을 자리를 찾는다.
                - 어차피 바로 덮어쓰므로 사전학습 가중치 값 자체는 쓰이지 않는다. 구조만 필요하다.
            */

            // ---------- (2) test 데이터를 1장씩 순회 ----------
            var testSet = new CastingDataset(Path.Combine(DatasetRoot, "test"), ImageSize);
            Console.WriteLine();
            Console.WriteLine($"test {testSet.Count}장 추론 시작 (배치가 아니라 1장씩 - 파일명을 매칭해야 하므로)");

            var falseNegatives = new List<(string path, float defectScore)>(); // 불량인데 정상이라 판정 = 유출
            var falsePositives = new List<(string path, float defectScore)>(); // 정상인데 불량이라 판정 = 과잉 폐기
            int correct = 0;

            var sw = System.Diagnostics.Stopwatch.StartNew();

            using (torch.no_grad())
            {
                for (long i = 0; i < testSet.Count; i++)
                {
                    using var d = torch.NewDisposeScope();
                    // 715회 반복이므로 스코프가 없으면 중간 텐서가 쌓여 메모리가 불어난다.

                    var sample = testSet.GetTensor(i);
                    var input = sample["data"].unsqueeze(0).to(device); // [3,224,224] -> [1,3,224,224]
                    long trueLabel = sample["label"].ToInt64();

                    var logits = model.forward(input);
                    var probs = torch.nn.functional.softmax(logits, dim: 1);
                    /*
                    여기서 softmax를 직접 적용하는 이유
                        - forward()는 로짓(정규화되지 않은 점수)을 반환한다. CrossEntropyLoss가 내부에서
                          softmax를 하기 때문에 모델에는 넣지 않았다.
                        - 하지만 사람이 읽을 확률이 필요하므로 추론 시에는 직접 적용한다.
                        - 학습 때 넣으면 이중 적용이 되고, 추론 때 빼면 확률로 읽을 수 없다. 둘을 구분해야 한다.
                    */

                    float defectScore = probs[0, CastingDataset.LabelDefect].ToSingle();
                    long predLabel = logits.argmax(1).ToInt64();

                    string path = testSet.GetPath(i);

                    if (trueLabel == CastingDataset.LabelDefect && predLabel == CastingDataset.LabelOk)
                    {
                        falseNegatives.Add((path, defectScore));
                    }
                    else if (trueLabel == CastingDataset.LabelOk && predLabel == CastingDataset.LabelDefect)
                    {
                        falsePositives.Add((path, defectScore));
                    }
                    else
                    {
                        correct++;
                    }
                }
            }

            sw.Stop();
            Console.WriteLine($"추론 완료 : {sw.ElapsedMilliseconds}ms  (장당 {sw.ElapsedMilliseconds / (double)testSet.Count:F1}ms)");
            Console.WriteLine($"  정답 {correct} / FN {falseNegatives.Count} / FP {falsePositives.Count}");
            Console.WriteLine($"  * Step2 평가와 숫자가 일치해야 정상이다(배치 추론과 단건 추론의 결과가 같아야 함).");
            /*
            Step2 결과와 대조하는 이유
                - 배치로 돌린 EvaluateWithConfusionMatrix와 여기 단건 추론의 FN/FP 개수가 달라지면
                  BatchNorm 처리나 eval() 모드에 문제가 있다는 뜻이다.
                - 숫자가 같아야 이 목록을 신뢰할 수 있다. 검산 먼저, 분석 나중.
            */

            // ---------- (3) FN 목록 : 확률 오름차순 ----------
            Console.WriteLine();
            Console.WriteLine($"---------- FN (불량 유출) {falseNegatives.Count}건 ----------");
            Console.WriteLine("  불량 확률이 낮은 순 = 모델이 '정상'이라고 확신한 순");
            Console.WriteLine();
            Console.WriteLine($"  {"파일명",-26} {"불량확률",8}  {"판정",-20}");
            Console.WriteLine("  " + new string('-', 60));

            foreach (var (path, score) in falseNegatives.OrderBy(x => x.defectScore))
            {
                string verdict = score >= 0.45f ? "임계값 0.45로 구제 가능"
                               : score >= 0.30f ? "임계값 0.30까지 낮추면 구제"
                               : "확신하며 틀림 - 구제 불가";
                Console.WriteLine($"  {Path.GetFileName(path),-26} {score,8:F3}  {verdict,-20}");
            }
            /*
            확률 오름차순으로 정렬하는 이유  ★ML.NET 실습에서 얻은 교훈의 적용
                - 같은 FN이라도 성격이 전혀 다르다.
                    0.498 : 모델이 사실상 "모르겠다"고 말한 것. 임계값만 내려도 잡힌다.
                    0.086 : 모델이 "정상"이라고 91.4% 확신하며 틀린 것. 임계값으로는 못 잡는다.
                - 정렬하면 "임계값으로 해결될 몇 건"과 "모델의 근본 한계 몇 건"이 한눈에 갈린다.
                - ML.NET 실습에서 cast_def_0_150(0.086)과 cast_def_0_2483(0.498)의 차이를
                  발견한 것이 이 방식이었다.
            */

            // ---------- (4) ML.NET이 놓친 2장과 대조 ----------
            Console.WriteLine();
            Console.WriteLine("---------- ML.NET 결과와 대조 ----------");
            var mlnetFn = new[] { "cast_def_0_150.jpeg", "cast_def_0_2483.jpeg" };
            // ML.NET(Accuracy 99.02%)이 놓친 FN 2건. Score=[0.086,0.914] / [0.498,0.502] 였다.

            var ourFnNames = falseNegatives.Select(x => Path.GetFileName(x.path)).ToHashSet(StringComparer.OrdinalIgnoreCase);

            foreach (var name in mlnetFn)
            {
                bool alsoOurs = ourFnNames.Contains(name);
                Console.WriteLine($"  {name,-26} : TorchSharp도 놓침 = {(alsoOurs ? "예" : "아니오")}");
            }

            int overlap = mlnetFn.Count(n => ourFnNames.Contains(n));
            Console.WriteLine();
            Console.WriteLine($"  겹침 {overlap}/2건, TorchSharp만 놓친 것 {falseNegatives.Count - overlap}건");
            /*
            이 대조로 무엇을 알 수 있는가  ★단계 3의 핵심 질문
                학습률을 1e-3 -> 1e-2 로 10배 바꿨는데도 test FN이 12건으로 동일했다.
                같은 12장을 놓친다면 원인은 최적화가 아니라 그 앞 단계에 있다.

                시나리오 A : ML.NET의 2장을 포함 + 다른 10장
                    -> 그 10장은 TorchSharp 경로에서만 손실된 정보 때문이다.
                       가장 의심스러운 것은 전처리다. 300x300 -> 224x224 리사이즈
                       (SKFilterQuality.Medium)에서 결함 텍스처가 뭉개졌을 가능성.
                       ML.NET은 내부에서 다른 크기/보간으로 처리했다.
                    -> 다음 실험 : ImageSize를 300으로 올려(리사이즈 없이) 재학습해 FN이 줄는지 본다.

                시나리오 B : 전혀 다른 12장
                    -> 전처리가 아니라 학습된 결정경계 자체가 다르다는 뜻이다.
                       분류층 구조(Dropout)나 옵티마이저 차이를 봐야 한다.

                시나리오 C : 겹침이 2건이고 우리 FN도 2~3건
                    -> 애초에 Step2 평가와 수치가 안 맞는 것이므로 코드 문제다.
            */

            // ---------- (5) FP 목록 ----------
            Console.WriteLine();
            Console.WriteLine($"---------- FP (과잉 폐기) {falsePositives.Count}건 ----------");
            foreach (var (path, score) in falsePositives.OrderByDescending(x => x.defectScore))
            {
                Console.WriteLine($"  {Path.GetFileName(path),-26} {score,8:F3}");
            }

            // ---------- (6) 임계값별 FN/FP 변화 ----------
            Console.WriteLine();
            Console.WriteLine("---------- 임계값 조정 효과 (FN만 대상) ----------");
            Console.WriteLine("  * FP 쪽 확률까지 봐야 정확한 트레이드오프가 나오지만,");
            Console.WriteLine("    여기서는 'FN 중 몇 건이 임계값으로 구제되는가'만 본다.");
            Console.WriteLine();
            foreach (float th in new[] { 0.5f, 0.45f, 0.40f, 0.35f, 0.30f, 0.20f })
            {
                int rescued = falseNegatives.Count(x => x.defectScore >= th);
                int remaining = falseNegatives.Count - rescued;
                Console.WriteLine($"  임계값 {th:F2} : FN {remaining,2}건 남음 (구제 {rescued,2}건)");
            }
            Console.WriteLine();
            Console.WriteLine("  * 임계값을 낮추면 FP가 늘어난다. 기대비용(FnCostRatio)으로 판단해야 한다.");
            Console.WriteLine("    ML.NET 실습에서 'FN 최소화' 규칙이 FN 9건 줄이려 FP 83건을 늘린 사례를 참고.");

            Console.WriteLine();
            Console.WriteLine("========== [단계 3] 완료 ==========");
            Console.WriteLine();
            Console.WriteLine("다음 : 위 FN 목록의 이미지를 직접 열어 결함이 보이는지 확인할 것.");
            Console.WriteLine("       '확신하며 틀림'으로 분류된 건이 Grad-CAM으로 분석할 가치가 가장 크다.");
            Console.WriteLine("       (모델이 어디를 보고 '정상'이라 판단했는지가 드러난다)");
        }

    }
}
