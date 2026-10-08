# FactoryLeadTimePrediction
- ML.NET을 사용한 제조현장 데이터 기반 예측 AI 모델 개발 테스트
- AI 자율제조 연구에 투입되어 .NET 기반 AI 플랫폼을 개발하는 과정에서, Python 환경을 쓰지 않고 AI 파이프라인을 구축하는 연습을 위해 작성
- 예측만 간단히 해보려고 LeadTimePrediction이라고 써놓긴 했는데, 머신러닝부터 딥러닝까지 다 테스트 해볼 예정

## Python 기반 vs C# 기반 AI 개발 장단점 비교 
<details>
<summary>Python vs C# 기반 AI 개발 비교</summary>

- 연구원에서 여러 국책, 민간수탁 R&D 사업에 참여하며 느낀점을 AI 모델 개발의 전 과정(데이터 전처리, 모델 학습, 추론 성능, AI 서비스 통합 개발)을 기준으로 비교


| 개발 단계 | Python | C#  |
| :--- | :--- | :--- |
| **데이터 전처리** | **[장점] 압도적인 생태계**<br>Pandas, NumPy, Scikit-learn 등 방대한 라이브러리가 존재하여 어떤 형태의 데이터(정형, 비정형, 오디오, 3D 등)든 몇 줄로 가공 가능.<br><br>**[단점]** 대용량 데이터 처리 시 GIL로 인한 멀티스레딩 한계, 고질적인 메모리 관리 비효율. | **[장점] 고성능 타입 안정성**<br>`IDataView` 구조와 LINQ를 활용해 지연 평가(Lazy Evaluation) 방식으로 메모리를 극도로 아끼며 대용량 처리가 가능. 강력한 멀티스레딩 성능.<br><br>**[단점] 전처리 라이브러리 부족**<br>Pandas만큼 직관적이고 풍부한 시각화/분석 라이브러리가 부족해 복잡한 비정형 데이터 가공 코드가 길어짐. |
| **모델 학습** | **[장점] AI 트렌드의 중심**<br>PyTorch, TensorFlow, Hugging Face 등 전 세계 최신 논문과 AI 아키텍처(LLM, Diffusion 등)가 파이썬으로 최초 공개됨.<br><br>**[단점]** 학습 파이프라인의 완성도를 높이려면 엔지니어링 리소스가 많이 들고, 코드 수정 시 런타임 에러(동적 타이핑) 리스크가 존재. | **[장점] .NET 에코시스템 통합**<br>AutoML, Model Builder(GUI)를 통해 머신러닝 모델 학습이 매우 직관적임. `TorchSharp`으로 C# 네이티브 딥러닝 학습도 가능.<br><br>**[단점] 최첨단 딥러닝 학습의 한계**<br>최신 파이토치 기반 거대 모델(LLM, 최신 멀티모달)을 C# 코드로만 처음부터 학습시키는 것은 커뮤니티 지원 부족으로 사실상 불가능에 가까움. |
| **모델 추론** | **[장점]** Hugging Face 런타임, Triton Inference Server 등 추론 최적화 서빙 도구가 풍부함.<br><br>**[단점] 오버헤드와 속도 저하**<br>인터프리터 언어 특유의 속도 한계가 있으며, C# 메인 시스템과 연동 시 인터페이스 통신(REST API, gRPC) 오버헤드 발생. | **[장점] 압도적인 런타임 속도 및 보안**<br>컴파일 언어의 성능과 `ONNX Runtime` 네이티브 연동을 통해 **밀리초 단위의 초고속 추론** 가능. 외부 프로세스 통신이 없어 메모리 카피 비용 제로, 강력한 소스코드 및 모델 보안 보장. |
| **AI 서비스 통합** | **[장점] 빠른 프로토타이핑**<br>FastAPI, Flask 등으로 AI SW나 마이크로서비스 API를 몇 분 만에 빌드 가능.<br><br>**[단점] 개발환경 관리가 어려움**<br>의존성 꼬임 문제(Conda, Pip 패키지 충돌), Electron이나 Pyinstaller 등으로 백엔드/프론트엔드 따로 빌드하려면 시간도 많이 소요되고, 오류도 잦음 | **[장점] 단일 스택의 청정 인프라**<br>기존 ASP.NET Core 웹서버나 WinForms 제조 장비 앱에 **AI 모델을 라이브러리(.dll)처럼 직접 임베딩**. 인프라 복잡도가 절반으로 줄어들고 DevOps 배포가 극도로 단순해짐.<br><br>**[단점]** 기존에 구동되던 .NET 환경이 아닐 경우, 오직 AI 서비스를 만들기 위해 C# 인프라를 처음부터 도입하기엔 진입장벽이 있음. |

</details>

## 261008_LightGBM
<details>
<summary>LightGBM 구현</summary>

### 입력 데이터
```
ProductionData
{
    public float InputQuantity // 투입 수량
    public float MachineTemperature // 설비 가동 온도
    public float WorkerCount // 투입 작업자 수
    public float LeadTime // 리드타임
}
```
### 출력 데이터(LightGBM이 도출해야 할 예측 결과)
```
LeadTimePrediction
{
    public float PredictedLeadTime // 예측 리드타임
    public float[] FeatureContributions // 예측결과 근거(기여도 점수)
}
```
### 테스트 방법
1. 가상 제조 데이터 생성
```
var trainingData = new List<ProductionData>
            {
                new ProductionData {InputQuantity = 100.0f, MachineTemperature = 75.5f, WorkerCount =3.0f, LeadTime = 45.0f },
                new ProductionData { InputQuantity = 150.0f, MachineTemperature = 78.2f, WorkerCount = 4.0f, LeadTime = 55.0f },
                new ProductionData { InputQuantity = 200.0f, MachineTemperature = 80.0f, WorkerCount = 5.0f, LeadTime = 62.0f },
                new ProductionData { InputQuantity = 80.0f,  MachineTemperature = 74.0f, WorkerCount = 2.0f, LeadTime = 40.0f },
                new ProductionData { InputQuantity = 120.0f, MachineTemperature = 76.1f, WorkerCount = 3.0f, LeadTime = 48.0f }  
            };

IDataView trainDataView = mlContext.Data.LoadFromEnumerable(trainingData); // 입력 데이터를 ML.NET의 데이터 형식인 IDataView로 변환.
```
2. 데이터 전처리 파이프라인 구축
```
var dataProcessPipeline = mlContext.Transforms.Concatenate("Features",
                nameof(ProductionData.InputQuantity),
                nameof(ProductionData.MachineTemperature),
                nameof(ProductionData.WorkerCount)
            ); 
```
- LightGBM은 입력데이터들을 하나의 벡터배열인 Features로 묶어주어야 하며, float형태만 받는다. int, bool 등의 자료형은 캐스팅 필요

3. LightGBM 하이퍼파라미터 설정
```
var options = new LightGbmRegressionTrainer.Options
            {
                NumberOfLeaves = 4,// 트리 노드(잎사귀) 갯수
                MinimumExampleCountPerLeaf = 2, // 노드 당 최소 데이터 수
                LearningRate = 0.1, // 학습률
                NumberOfIterations = 100, // 부스팅 반복 횟수(Tree 갯수)
                Booster = new GradientBooster.Options{L2Regularization = 0.5}, // L2 규제 적용(L2 규제 : 과적합 방지를 위한 손실함수에 가중치 제곱합을 패널티로 더해주는 기법)
                FeatureColumnName = "Features",
                LabelColumnName = "Label"
            };

var trainer =mlContext.Regression.Trainers.LightGbm(options); // 하이퍼파라미터를 설정한 LightGBM옵션을 학습기에 적용
```
4. 데이터 전처리 파이프라인과 학습기 결합
```
var trainingPipeline = dataPipeline = dataProcessPipeline.Append(trainer);
```
5. 학습 시작
```
var trainedModel = trainingPipeline.Fit(trainDataView);// ML.NET은 주입된 IDataView 형 데이터를 Fit()메서드로 학습하며, 학습 완료 시 ITransformer형 모델을 반환.
```
6. 예측근거(기여도) 계산
- Features가 예측값에 어떤 긍정적/부정적 기여를 했는지 계산하는 ExplainabilityCatalog.CalculateFeatureContribution()를 사용.
- 예를 들어, 투입 수량 기여도가 +12.5, 작업자 수 기여도가 -5.0 으로 나왔다면, "투입 수량이 많아서 리드타임이 12.5분 늘어났고, 작업자를 많이 배치해서 5분이 단축된 결과, 리드타임을 MM분이라고 예측할 수 있었다"라는 예측결과 도출의 근거를 알 수 있음.
- 기여도 계산기도 Fit()을 한번 거쳐야 ITransformer 형이 된다.
- **기여도 계산기는 모델이 아니라 변환기(Transformer)이다.**
- 즉 trainedModel은 Concatenate + LightGBM이 결합된 ITransformer 체인으로, 예측 엔진은 그 체인이 출력하는 컬럼만 "도출된 예측값"으로 채우는 것.

```
 var transformedTrainedModel = trainedModel.Transform(trainDataView); // 기여도 계산기에 넣기 위한 Features컬럼이 있는 데이터를 만든다. 

var contributionCalcPipeline = mlContext.Transforms.CalculateFeatureContribution(trainedModel.LastTransformer, normalize : false).Fit(transformedTrainedModel);

var scoringPipeline = trainedModel.Append(contributionCalcPipeline);// 기여도 컬럼을 만들어주는 변환기가 붙은 새 모델.

```
- LastTransformer()를 통해 trainedModel에서 마지막 트랜스포머(예측기)만 추출
- Transform()을 수행한 데이터로 Fit()을 한번 더 수행 후 Append()로 새 모델을 만든다. 이를 통해 체인 끝에 기여도 컬럼을 만들어주는 변환기를 한 칸 더 붙일 수 있고, 이것을 새 모델(contributionCalcPipeline)로 만들어 예측을 돌린다.

7. 예측 수행
```
var predictionEngine_contributionCalcResult = mlContext.Model.CreatePredictionEngine<ProductionData, LeadTimePrediction>(scoringPipeline);// 기여도 컬럼이 붙은 새 모델로 예측엔진 생성

var newSituation = new ProductionData
            {
                InputQuantity =230.0f,
                MachineTemperature = 120.0f,
                WorkerCount = 10.0f
            }; // 예측모델에 넣을 새로운 데이터

var prediction = predictionEngine_contributionCalcResult.Predict(newSituation);// 새로운 공정환경 데이터를 예측 엔진에 주입 후 예측 시작
```
8. 결과 출력
```
Console.WriteLine($"[입력 조건]\n1. 투입량 : {newSituation.InputQuantity}개\n
                                2. 온도 : {newSituation.MachineTemperature}도\n
                                3. 작업자 수 : {newSituation.WorkerCount}명");

Console.WriteLine($"[예측 결과]예상 공정 리드타임 : {prediction.PredictedLeadTime:F2}분");

string[] featuresNames = new[] {"투입량", "온도", "작업자 수"}; // 기여도 배열의 라벨 순서는 Features와 같음.
            for(int i=0; i< featuresNames.Length; i++)
            {
                Console.WriteLine($"- {featuresNames[i]} 기여도 : {prediction.FeatureContributions?[i]:F2}");
            }
```

### 결과
```
---------LightGBM 모델 학습 시작---------
---------모델 학습 완료---------
---------실시간 리드타임 예측 결과---------
[입력 조건]
1. 투입량 : 230개
2. 온도 : 120도
3. 작업자 수 : 10명
[예측 결과]예상 공정 리드타임 : 58.50분
- 투입량 기여도 : 14.16
- 온도 기여도 : 0.00
- 작업자 수 기여도 : 0.00
```

### 해석
1. 기여도의 핵심 성질은 [기준값(bias) + 모든 기여도 합 = 예측 리드타임]
    - 기여도 합 : 14.16 + 0.0 + 0.0 = 14.16
    - 예측 리드타임 : 58.50
    - 기준값 : 58.50 - 14.16 = 44.34
2. 학습데이터 리드타임 평균은 (45+55+62+40+48)/5 = 50인데, 44.34은 LightGBM의 초기 예측값(부스팅 시작점)임.
    - 즉 "기본 44.34분에서 출발해 투입량이 많아서 14.16분 늘어나 58.5분" 이라는 해석이 됨.
3. 더미로 넣어둔 학습데이터를 보면,

|투입량|온도|작업자|리드타임|
|---|---|---|---|
|80|74.0|2|40|
|100|75.5|3|45|
|120|76.1|3|48|
|150|78.2|4|55|
|200|80.0|5|62|

- 세 Feature가 선형으로 증가한다. 트리 모델은 분기 시 Feature 하나만 고르는데, 어느걸 골라도 결과가 동일하니까 투입량 하나만 보고 나머지는 한번도 안봤기 때문에 온도, 작업자 수 기여도가 0이 나온것.

### 심화 테스트 설계
- 기초적인 LightGBM 학습-예측-결과 도출-근거 확인을 수행했으니, 실제 대용량 제조 데이터를 사용하여 파이프라인을 수행해본다.
- 제조 데이터는 KAMP(https://www.kamp-ai.kr/main)에서 찾은 "표면처리 공급망 최적화 AI 데이터셋"(https://www.kamp-ai.kr/aidataDetail?AI_SEARCH=&page=1&DATASET_SEQ=36&DISPLAY_MODE_SEL=CARD&EQUIP_SEL=&GUBUN_SEL=C004016&FILE_TYPE_SEL=&WDATE_SEL=)을 사용해보기로 한다.

</details> 