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

<details>
<summary>LightGBM 심화테스트(합성데이터 사용 및 파이프라인 검증 결과)</summary>

### 합성데이터 파이프라인 검증 시작
```
------------기본 실험 : 176행 / sigma=3--------------
합성 데이터 생성 완료 : 176행 (sigma=3, seed=42)

---------- 평가 : 학습 데이터 ----------
  R²   :   0.9713   (1에 가까울수록 좋음 / 음수 = 평균보다 못함)
  RMSE :     1.37일
  MAE  :     1.08일

---------- 평가 : 검증 데이터 ----------
  R²   :   0.7484   (1에 가까울수록 좋음 / 음수 = 평균보다 못함)
  RMSE :     3.76일
  MAE  :     3.25일

---------- 파이프라인 검증 판정 ----------
  [기준1] 검증 R² >= 0.8  : 실패 (실측 0.7484)
  [기준2] 학습 R² > 검증 R² : 통과(정상 과적합)
  [기준3] 검증 R² <= 0.877 : 통과 <- 이론 상한 초과 여부

---------- 5-Fold 교차검증 ----------
    Fold1 : R² =  0.7484 / RMSE =   3.76일
    Fold2 : R² =  0.8654 / RMSE =   3.40일
    Fold3 : R² =  0.7797 / RMSE =   3.57일
    Fold4 : R² =  0.7456 / RMSE =   3.94일
    Fold5 : R² =  0.7919 / RMSE =   3.27일
  ----------------------------------------
  평균 R²   :  0.7862
  평균 RMSE :    3.59일 (표준편차 0.24)
  * 실데이터 사용 시 표준편차는 1.24-> 이보다 작으면 패턴 안정적

---------- 기여도 분석 (정답 대조) ----------

  [실제] 26.16일   [예측] 26.43일   [오차] 0.27일
  [기준값] 14.13일
    납기여유일        :     8.85일   정답 : +0.8
    거래처재고        :    -5.47일   정답 : +1.5
    운송빈도         :     6.88일   정답 : -1.2
    생산량(kg)      :     1.12일   정답 : 무관 (0)
    공정시간(분)      :     1.24일   정답 : 무관 (0)
    요일           :    -0.33일   정답 : 무관 (0)

  [실제] 15.83일   [예측] 15.76일   [오차] 0.07일
  [기준값] 22.81일
    납기여유일        :    -5.91일   정답 : +0.8
    거래처재고        :    -4.93일   정답 : +1.5
    운송빈도         :     1.23일   정답 : -1.2
    생산량(kg)      :     1.10일   정답 : 무관 (0)
    공정시간(분)      :     1.93일   정답 : 무관 (0)
    요일           :    -0.47일   정답 : 무관 (0)

  [실제] 19.40일   [예측] 19.14일   [오차] 0.26일
  [기준값] 32.01일
    납기여유일        :    -3.71일   정답 : +0.8
    거래처재고        :   -10.44일   정답 : +1.5
    운송빈도         :     4.33일   정답 : -1.2
    생산량(kg)      :    -3.34일   정답 : 무관 (0)
    공정시간(분)      :    -0.04일   정답 : 무관 (0)
    요일           :     0.35일   정답 : 무관 (0)

---------- 기여도 검증 판정 ----------
  피처별 평균 기여 크기 (30건 평균) :
    납기여유일        :    9.48  ← 커야 정상
    거래처재고        :    7.33  ← 커야 정상
    운송빈도         :    3.43  ← 커야 정상
    생산량(kg)      :    1.67  ← 0에 가까워야 정상
    공정시간(분)      :    1.51  ← 0에 가까워야 정상
    요일           :    0.43  ← 0에 가까워야 정상
  크기 순서 : 납기여유일 > 거래처재고 > 운송빈도 : 통과
  노이즈 피처 기여도 < 진짜 피처 최소값 : 통과 (노이즈 최대 1.67 vs 운송빈도 3.43)
  기준값 편차 : 17.88일
    -> 합성데이터에서도 기준값이 흔들림. ML.NET 기여도가 SHAP처럼 bias+합=Score 항등식을 보장하지 않는 듯함
```

### 실험 A : 행 수 효과 (sigma=3 고정)

|행수|학습 R²|검증 R²|격차|
|---|---|---|---|
|176 |0.971|0.748|0.223|
|500 |0.978|0.851|0.128|
|1000 |0.964|0.879|0.084|
|3000 |0.922|0.877|0.045|

- 기대 패턴 : 행수가 늘면 학습 R²는 내려가고 검증 R²는 올라가며 격차가 좁혀진다. = 데이터가 많아지면 모델이 '외우기'에서 '학습하기'로 바뀐다는 뜻.
- 핵심 : 176행에서도 검증 R²가 0.8 이상 나온다. 실데이터 176행은 R² -0.35였으므로, 실패 원인은 '행이 적어서'가 아니라 '정보가 없어서'다. -> 해결책은 데이터를 더 모으는 것이 아니라 거래처 납기 같은 컬럼을 수집하는 것

### 실험 B : 노이즈 효과 (176행 고정)
|sigma|검증 R²|이론상한|비고|
|---|---|---|---|
|1|0.961|0.981|정상|
|3|0.748|0.877|정상|
|5|0.441|0.732|정상|
|8|-0.030|0.533|정상|

- 노이즈가 커지면 R²가 떨어진다. 이것은 모델의 한계가 아니라 '설명 불가능한 변동'이 늘어난 것이다.
- 현실 데이터에는 항상 노이즈가 있으므로, R² 1.0은 목표가 될 수 없다. 이론 상한을 넘으면 오히려 데이터 누출을 의심해야 한다.
</details> 

## 261115_Deep Learning : 주조품 외관 양/불량 이진분류
<details>
<summary>ML.NET ImageClassificationTrainer 기반 전이학습 (Accuracy 99.02% / 학습 2.5분)</summary>

### 목적
- 머신러닝(LightGBM, 정형 데이터)에서 딥러닝(CNN, 이미지)으로 넘어가 **C#만으로 비전 품질검사 모델**을 만들 것
- 전이학습(Transfer Learning)이 ML.NET에서 어떻게 구현되는지, Python/PyTorch와 무엇이 다른지 확인

### 사용 데이터셋
| 항목 | 내용 |
| :--- | :--- |
| 이름 | Casting Product Image Data for Quality Inspection |
| 출처 | Kaggle (제출자 : Ravirajsinh Dabhi) |
| 대상 | 펌프 임펠러(submersible pump impeller) 주조품 상면 |
| 형식 | JPEG, 300x300, 3채널 (안정된 조명 환경에서 촬영, 배경 통제됨) |
| 라이선스 | **CC BY-NC-ND 4.0** |

> **라이선스 준수 사항**
> `ND(No Derivatives)` 조항 때문에 **원본 이미지와 학습된 모델 가중치를 본 저장소에 포함하지 않는다.**
> 데이터셋은 저장소 외부 경로에 두고, `.gitignore`로 `*.jpg / *.jpeg / *.png / *.zip / /models / /workspace_casting`을 차단했다.
> 따라서 이 README는 **수치, 혼동행렬, 로그만으로** 학습 과정과 결과를 기록한다. 재현하려면 Kaggle에서 직접 데이터셋을 받아 `DatasetRoot` 경로를 수정해야 한다.

#### 데이터 분할
| 분할 | 불량(def_front) | 정상(ok_front) | 합계 | 용도 |
| :--- | ---: | ---: | ---: | :--- |
| train | 3,758 | 2,875 | 6,633 | 아래에서 90:10으로 재분할 |
| &nbsp;&nbsp;└ train | — | — | 5,970 | 실제 학습 |
| &nbsp;&nbsp;└ validation | — | — | 663 | 에폭별 검증 / 조기 종료 판단 |
| test | 453 | 262 | 715 |*최종 평가 1회만 |

> **왜 3분할?** : `ImageClassificationTrainer`는 에폭마다 검증 정확도를 계산하고 조기 종료 판단에 씀. 여기에 test를 넣으면 test가 학습 과정에 관여해 오염됨

#### 베이스라인
```
test 기준 : 불량 453 / 715 = 63.36%
-> 아무것도 학습하지 않고 "전부 불량"이라고만 답해도 Accuracy 63.36%
```

---

### 입력 / 출력 데이터
```csharp
CastingImageData // 입력 : 프로퍼티 2개
{
    public string ImagePath // 이미지 파일의 절대 경로 (픽셀이 아니라 경로)
    public string Label     // "defect" 또는 "ok"
}

CastingPrediction // 출력
{
    [ColumnName("PredictedLabel")] public string PredictedLabel // 예측 클래스명
    public float[] Score  // 클래스별 확률 (길이 2)
}
```

> **LightGBM과의 차이** : `LeadTimeFeature`는 피처 6개를 사람이 직접 설계해 float로 선언.( **CNN이 픽셀에서 피처를 스스로 학습**하기 때문)
> 그래서 `Concatenate("Features", ...)`에 해당하는 단계가 아예 없고, `CalculateFeatureContribution()`도 쓸 수 없음. 비전 모델의 설명가능성은 Grad-CAM 같은 완전히 다른 기법이며 ML.NET에 내장 지원이 없다. 대신 **오분류 사례의 확률값 분석**으로 모델의 한계를 파악

---

### 학습 방법

#### 1. 전처리 파이프라인
```csharp
var preprocessingPipeline = mlContext.Transforms.Conversion
    .MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label")
    .Append(mlContext.Transforms.LoadRawImageBytes(
        outputColumnName: "ImageBytes", imageFolder: null, inputColumnName: "ImagePath"));

var preprocessingTransformer = preprocessingPipeline.Fit(trainAllView);
IDataView trainTransformed = preprocessingTransformer.Transform(trainView);
IDataView valTransformed   = preprocessingTransformer.Transform(valView); // 필수
```
- `MapValueToKey` : 문자열 라벨("defect"/"ok")을 내부 Key 타입(0, 1)으로 변환. 분류 알고리즘은 문자열을 직접 못 다룸
- `LoadRawImageBytes` : 경로를 읽어 원본 바이트로 변환. 디코딩/리사이즈/정규화는 트레이너가 내부에서 알아서 처리
- **`ValidationSet`은 반드시 직접 `Transform()` 해야 함** -> 아래 트러블슈팅 참조.

#### 2. 트레이너 옵션
```csharp
var options = new ImageClassificationTrainer.Options
{
    FeatureColumnName = "ImageBytes",
    LabelColumnName   = "LabelKey",
    Arch              = ImageClassificationTrainer.Architecture.MobilenetV2,
    Epoch = 20, BatchSize = 16, LearningRate = 0.01f,
    ValidationSet = valTransformed,                              // 전처리 거친 것
    MetricsCallback = metrics => Console.WriteLine($"    {metrics}"),
    WorkspacePath = WorkspacePath,
    ReuseTrainSetBottleneckCachedValues = true,
    ReuseValidationSetBottleneckCachedValues = true,
    EarlyStoppingCriteria = new ImageClassificationTrainer.EarlyStopping(
        minDelta: 0.001f, patience: 5,
        metric: ImageClassificationTrainer.EarlyStoppingMetric.Accuracy),
};

var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(options)
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));
```

| 옵션 | 역할 |
| :--- | :--- |
| `Arch = MobilenetV2` | ImageNet 사전학습 백본. 가볍고 빨라 CPU 학습에 적합. 첫 실행 시 자동 다운로드. |
| `WorkspacePath` | **병목(bottleneck) 피처 캐시 폴더.** 전이학습은 백본을 1회만 통과시켜 피처를 뽑고 그 피처로 마지막 층만 반복 학습하므로 CPU로도 현실적인 시간에 끝남 |
| `MetricsCallback` |  없으면 학습 중 아무 출력이 없어 멈춘 것처럼 보임.|
| `EarlyStoppingCriteria` | 검증 정확도가 더 오르지 않으면 중단 |

> **이진 분류인데 왜 `MulticlassClassification`인가?** : `ImageClassificationTrainer`는 다중분류 카탈로그에만 존재. 클래스가 2개면 이진 분류와 수학적으로 동일하게 동작하지만 평가 지표가 `MicroAccuracy` 등 다중분류 쪽으로 나온다. 그래서 Precision/Recall은 혼동행렬에서 직접 계산

#### 3. 전처리 + 모델 재결합
```csharp
var fullModel = preprocessingTransformer.Append(trainedModel);
mlContext.Model.Save(fullModel, trainAllView.Schema, ModelSavePath);
```
전처리를 체인에서 분리했으므로 `trainedModel`은 원본 DTO를 받지 못함. `PredictionEngine`과 `Model.Save`에는 반드시 재결합한 `fullModel`을 써야 함.  그러지 않으면 나중에 Load한 모델이 "경로 -> 바이트" 변환을 못 해 쓸 수 없는 모델이 된다

---

### 결과

#### 학습 로그 (일부)
```
Phase: Bottleneck Computation, Dataset used: Train, Image Index: 5941
    Epoch: 0, Train Accuracy: 0.8668, Cross-Entropy: 0.5157 | Validation Accuracy: 0.8835
    Epoch: 1, Train Accuracy: 0.9679, Cross-Entropy: 0.0923 | Validation Accuracy: 0.9901
    Epoch: 2, Train Accuracy: 0.9857, Cross-Entropy: 0.0547 | Validation Accuracy: 0.9929
    Epoch: 3, Train Accuracy: 0.9881, Cross-Entropy: 0.0461 | Validation Accuracy: 0.9943
    Epoch: 4, Train Accuracy: 0.9906, Cross-Entropy: 0.0405 | Validation Accuracy: 0.9957
    Epoch: 5~9 ................................................ Validation Accuracy: 0.9957 (변화 없음)
Froze 2 variables. Converted 2 variables to const ops.
========== 학습 완료 : 2.5분 소요 ==========
```

#### 최종 성능
```
---------- Score 배열 인덱스 <-> 클래스 대응 ----------
  Score[0] = "defect"
  Score[1] = "ok"
```

| 지표 | train (5,970장) | **test (715장)** | 베이스라인 |
| :--- | ---: | ---: | ---: |
| MicroAccuracy | 99.50% | **99.02%** | 63.36% |
| MacroAccuracy | 99.50% | **98.83%** | — |
| LogLoss | 0.0272 | **0.0346** | — |
| Precision(불량) | 99.64% | **98.90%** | — |
| **Recall(불량)** | 99.47% | **99.56%** | — |
| F1(불량) | 99.56% | **99.23%** | — |

**혼동행렬 (test / "불량"을 Positive로 설정)**

| | 예측 : 불량 | 예측 : 정상 |
| :--- | ---: | ---: |
| **실제 : 불량** | 451 (TP) | **2 (FN) <- 불량 유출** |
| **실제 : 정상** | 5 (FP) | 257 (TN) |

> **왜 Recall을 주 지표로 삼는가**
> ```
>              예측:정상            예측:불량
> 실제:정상    TN(정상 출하)        FP(멀쩡한 걸 폐기)
> 실제:불량    FN(불량 유출)        TP(정상 검출)
>              
> ```

---

### 해석

#### 1. 과적합이 거의 없음
```
train Accuracy 99.50%  ->  test 99.02%   격차 0.48%p
```
LightGBM B는 학습 R² 0.971 / 검증 0.748로 **격차 0.223**이었다. 여기는 격차가 거의 0이다. 이유는 아래와 같다.
- **데이터가 많음** : 5,970장 vs 176행
- **학습 파라미터가 적음** :  MobilenetV2 백본은 동결되고 마지막 분류층만 학습한다. **외울 수 있는 용량 자체가 작아서 과적합하기 어려움**

Epoch 1에서 이미 검증 99.0%에 도달한 것도 같은 이유로, 백본이 ImageNet에서 모양/질감/엣지를 이미 다 배워와서 그 피처를 2개 클래스로 가르는 분류기만 학습시킨 것

#### 2. 오분류 사례 분석
`Score` 배열을 함께 출력한 것에서 "간신히 틀린 것"과 "확신하며 틀린 것"이 구분됨

```
FN cast_def_0_2483.jpeg   Score=[0.498, 0.502]   <- 간신히 틀림 (임계값으로 구제 가능)
FN cast_def_0_150.jpeg    Score=[0.086, 0.914]   <- 확신하며 틀림 = 모델의 진짜 한계

FP cast_ok_0_1368.jpeg    Score=[0.500, 0.500]   <- 완전히 모름
FP cast_ok_0_2311.jpeg    Score=[0.506, 0.494]   <- 간신히 틀림
FP cast_ok_0_259.jpeg     Score=[0.538, 0.462]
FP cast_ok_0_1022.jpeg    Score=[0.626, 0.374]
FP cast_ok_0_10.jpeg      Score=[0.731, 0.269]   <- 확신하며 틀림
```

**임계값을 0.5 -> 0.45로 낮추면** `cast_def_0_2483`(0.498)을 잡아 FN이 2 -> 1건이 된다. 반면 FP 쪽 0.494 / 0.462는 여전히 틀리므로 **FN만 줄고 FP는 늘지 않는다.** 

> `PredictedLabel`은 단순히 Score가 가장 큰 쪽을 고른 결과(argmax = 임계값 0.5)다. 품질검사에서 0.5는 최적이 아님. 이 조정을 하려면 `PredictedLabel`이 아니라 **`Score` 원본 확률**이 필요. 그래서 출력 DTO에 `float[] Score`를 둔 것이 중요

반면 `cast_def_0_150`(0.086)은 모델이 "정상"이라고 **91.4% 확신**하며 틀림. 임계값으로 잡을 수 없는 모델의 근본적 한계선

---

---

### 추가 실습 1 : 임계값 튜닝

#### 목적
`PredictedLabel`은 Score가 큰 쪽을 고른 결과(argmax)이며, 이진 분류에서는 **임계값 0.5를 쓴 것과 같음.** 그런데 0.5는 "FN과 FP의 비용이 같다"는 가정 위에 존재함

**임계값 조정은 모델을 재학습하지 않음.** 이미 뽑힌 확률을 다르게 해석할 뿐이므로 비용이 0

```csharp
bool predictedDefect = defectScore >= th;
```

#### 결과 (512 미사용 808장 기준)

| 임계값 | TP | FN | FP | TN | Recall | Precision | F1 | Accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.30 | 423 | 23 | 184 | 178 | 94.8% | 69.7% | 80.3% | 74.4% |
| 0.35 | 422 | 24 | 167 | 195 | 94.6% | 71.6% | 81.5% | 76.4% |
| 0.40 | 421 | 25 | 157 | 205 | 94.4% | 72.8% | 82.2% | 77.5% |
| 0.45 | 421 | 25 | 151 | 211 | 94.4% | 73.6% | 82.7% | 78.2% |
| **0.50** | 421 | 25 | 136 | 226 | 94.4% | 75.6% | 83.9% | **80.1%** (기본값) |
| 0.55 | 420 | 26 | 125 | 237 | 94.2% | 77.1% | 84.8% | 81.3% |
| 0.60 | 417 | 29 | 119 | 243 | 93.5% | 77.8% | 84.9% | 81.7% |
| 0.70 | 414 | 32 | 101 | 261 | 92.8% | 80.4% | 86.2% | **83.5%** |


최초 구현은 **"FN 최소화를 1순위"** 로 하였음
```csharp
if (fn < bestFn || (fn == bestFn && f1 > bestF1AtBestFn))  // 잘못된 규칙
```
결과는 **임계값 0.30 권장**이었다. 그런데 표를 보면
```
0.30 : FN 23  FP 184  Accuracy 74.4%   <- 코드가 권장한 값
0.70 : FN 32  FP 101  Accuracy 83.5%   <- 실제로 나은 값
       차이 : FN +9건,  FP -83건
```
**FN 9건을 줄이려고 FP 83건을 늘렸다.** 비율이 9:1

#### 수정 : 기대비용 최소화

```csharp
private const double FnCostRatio = 10.0;
// FN(불량 유출) 1건이 FP(정상품 폐기) 1건보다 몇 배 비싼가
//   FP 비용 = 그 제품의 제조원가 (멀쩡한 걸 버린 손해)
//   FN 비용 = 클레임 처리 + 교체 + 신뢰도 손상 (+ 최악의 경우 리콜)

double expectedCost = fn * FnCostRatio + fp * 1.0;
if (expectedCost < bestCost) { ... }
```

비용비 `k = FN비용/FP비용` 에 따른 최적점 (FP 1건 = 1 단위로 정규화) :

| 임계값 | FN | FP | k=5 | k=10 | k=20 | k=50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.30 | 23 | 184 | 299 | 414 | 644 | **1334** |
| 0.50 | 25 | 136 | 261 | **386** | **636** | 1386 |
| 0.55 | 26 | 125 | **255** | 385 | 645 | 1425 |
| 0.70 | 32 | 101 | 261 | 421 | 741 | 1701 |

- **k=5~10** (현실적인 펌프 임펠러 가정) -> **0.50~0.55가 최적**
- **0.30이 최적이 되려면 k > 40** 이 필요하다. 브레이크나 항공 부품처럼 **리콜/안전사고로 번지는 부품**이 아니면 성립하지 않는다.

> **결론** : 임계값은 모델의 속성이 아니라 **[모델 + 데이터 분포 + 비용구조]의 함수**.
---

### 추가 실습 2 : casting_512x512 외부 검증 (도메인 시프트 실측)

#### 설계 : 파일명 중복 제외

```
casting_512x512 : 1,300장
  ├─ train 폴더와 파일명이 같은 것 : 460장 (35.4%)  <- 학습에 쓴 이미지
  ├─ test 폴더와 파일명이 같은 것  :  44장
  └─ 어느 쪽에도 없는 것          : 796장 (61.2%)  <- 진짜 미사용
```

1,300장 전체로 평가하면 **35%가 학습에 쓴 이미지**라 성능이 부풀려진다. 그래서 `HashSet<string>`으로 파일명을 걸러 **808장**(불량 446 / 정상 362)만 따로 평가.

> `List.Contains`는 O(n)이라 1,300 × 7,348 = 955만 번 비교. `HashSet`은 O(1)이므로 1,300번으로 끝난다.
>

학습은 하지 않고 **저장된 `models/CastingDefect_Model.zip`을 `Model.Load()`로 불러와** 추론만 했다. 

#### 결과

| | test (300x300) | 512 전체 1,300장 | **512 미사용 808장** |
| :--- | ---: | ---: | ---: |
| Accuracy | 99.02% | 80.69% | **80.07%** |
| **Recall(불량)** | 99.56% | 93.09% | **94.39%** |
| **Precision(불량)** | 98.90% | 78.68% | **75.58%** |
| F1(불량) | 99.23% | 85.28% | **83.95%** |
| 베이스라인 | 63.36% | 60.08% | 55.20% |

**혼동행렬 (512 미사용 808장)**

| | 예측 : 불량 | 예측 : 정상 |
| :--- | ---: | ---: |
| **실제 : 불량** | 421 (TP) | **25 (FN)** |
| **실제 : 정상** | **136 (FP)** | 226 (TN) |

#### 해석 1 : 누출 제외 효과가 거의 없음

```
512 전체 1,300장  : Accuracy 80.69% / Recall 93.09%
512 미사용  808장 : Accuracy 80.07% / Recall 94.39%  <- Recall은 오히려 상승
                    차이 0.6%p
```

누출이 성능을 부풀리고 있었다면, 중복 492장은 거의 다 맞췄을 테니 제외 시 수치가 뚝 떨어져야 한다. **그렇지 않았음.** 

> **모델이 "이 임펠러 개체"를 외운 것이 아니라, "이 해상도의 픽셀 패턴"에 의존**
> 같은 부품의 사진인데도 해상도가 바뀌면 못 알아봄

파일명 중복 걱정은 결과적으로 기우였지만, **확인해야만 알 수 있는 것**이었다. 확인 없이 "1,300장으로 80%"라고만 적었다면 운 좋게 맞은 셈

#### 해석 2 : Recall이 아니라 Precision이 무너짐

```
Recall    99.56% -> 94.39%   (-5.2%p)
Precision 98.90% -> 75.58%   (-23.3%p) 
```
```
FP(정상을 불량으로) : 136건 / 정상 362장 = 37.6%가 오판
FN(불량을 정상으로) :  25건 / 불량 446장 =  5.6%가 오판
```

**정상품 3개 중 1개 이상을 불량으로 버리고 있다.** 모델이 "애매하면 불량"으로 쏠렸다. 512 -> 300 다운샘플링에서 생긴 리샘플링 노이즈(미세한 격자/링잉 패턴)를 결함 텍스처로 오인하는 것으로 추정

품질검사 관점에서 이는 FN이 늘어난 것보다는 덜 치명적(불량 유출은 94.4%). 하지만 **정상품 37%를 폐기하는 라인은 경제적으로 성립하지 않음**

#### 해석 3 : 최적 임계값의 방향이 뒤집힘

| | 300x300 test | 512 미사용 808장 |
| :--- | :--- | :--- |
| FN / FP | FN 2 / FP 5 | FN 25 / FP 136 |
| 문제 | 균형 잡힘 | **FP가 과도** |
| 임계값 방향 | **내려야** (0.45) | **올려야** (0.55~0.70) |

**같은 모델, 같은 가중치인데 입력 분포가 바뀌니 최적 판정 문턱이 반대로 간다.**  운영 환경에서 카메라나 조명을 바꾸면 다시 잡아야 한다.

#### 결론 : 99.02%는 이 조건에서만 유효한 숫자

이것이 **도메인 시프트(domain shift)** . 학습 데이터는 안정된 조명·고정 배경·300x300으로 통제되어 있었고, 모델은 그 조건에 맞춰 최적화되었다. 조건이 하나(해상도) 바뀌자 Accuracy가 **19%p** 떨어졌다.

---

### 추가 실습 3 : WPF 추론 앱 (CastingInspector.Wpf)

학습된 모델을 가벼운 앱에 넣어 돌림. 이미지를 선택하면 양/불량 판정과 확률을 표시.

#### 구조
```
FactoryLeadTimePrediction.sln
├── FactoryLeadTimePrediction.csproj      (net8.0, 콘솔 : 학습/실습)
└── CastingInspector.Wpf.csproj           (net8.0-windows, WPF : 추론 전용)
        └── ProjectReference -> 콘솔 프로젝트 (DTO 재사용)
```

#### UI

![Image](image.png)
![DefImage](image-1.png)

#### WPF 유의점
```csharp
await Task.Run(() => { var model = mlContext.Model.Load(path, out _); ... });
```
모델 파일이 22.8MB이고 TensorFlow 그래프 복원까지 하므로 수 초가 걸린다. **UI 스레드에서 하면 창이 흰색으로 멈추고 "응답 없음"이 뜬다.** `Task.Run`으로 스레드풀에 보내고, `await` 이후 UI 스레드로 돌아와 UI 요소를 갱신. 추론도 같은 이유로 래핑.

> `PredictionEngine`은 **스레드 안전하지 않음.** 버튼을 비활성화해 한 번에 한 건만 처리하는 구조라 안전하지만 동시 요청을 처리하려면 `PredictionEnginePool`을 써야 함.

`BitmapCacheOption.OnLoad`도 필수. 기본값(`OnDemand`)은 파일 핸들을 계속 열어 두어 해당 파일을 삭제·덮어쓸 수 없고, 같은 파일을 다시 열면 옛 이미지가 캐시에서 나온다.

#### 앱으로 확인한 모델의 한계

콘솔 실습에서 오분류로 나온 파일들을 직접 넣어 봄

| 파일 | 판정 | 의미 |
| :--- | :--- | :--- |
| `cast_def_0_2483.jpeg` | 정상 50.2% + **경계 경고** | 모델도 햇갈림 |
| `cast_def_0_150.jpeg` | **정상 91.4%** (경고 없음) | 불량인데 자신 있게 틀림 |
| 512x512 이미지들 | 정상품을 불량으로 다수 오판 | 모델 한계? |

**`cast_def_0_150` 사례가 중요.** UI가 "정상, 확신도 91.4%"를 자신 있게 표시하는데 실제로는 불량. 경계 경고도 뜨지 않는다. 확률을 보여주는 것은 **"모델이 흔들리는 구간"** 을 드러내는 데 유용하지만, **"모델이 확신하며 틀리는 구간"** 은 잡아내지 못한다. 후자를 다루려면 OOD 탐지나 앙상블 불확실도 같은 별개의 기법이 필요.

</details>

## 261122_TorchSharp : ML.NET 재현 + Grad-CAM XAI 구현
<details>
<summary>ML.NET의 블랙박스를 벗어나기 위한 TorchSharp 전환</summary>

### 왜 TorchSharp로 옮겼는가

ML.NET 실습(99.02%)에서 남은 질문은 **"불량이라면 어디가 불량인가"** 였다. 이미지에 마스크나 바운딩박스를 쳐서 근거를 보여주고 싶었다.

#### 먼저 확인한 것 : 세그멘테이션 모델은 쓸 수 없다

| 방법 | 필요 라벨 | 가능 여부 |
| :--- | :--- | :--- |
| DeepLabV3 / SegFormer | **픽셀 단위 마스크** | ❌ 데이터셋에 마스크가 없음 |
| Grad-CAM | 없음 (학습된 모델만) | ⚠️ ML.NET에서는 불가 |
| Occlusion Sensitivity | 없음 | ✅ 가능하나 느림 |

```
현재 라벨   : cast_def_0_150.jpeg -> "defect"      (이미지 1장당 문자열 1개)
세그멘테이션 : cast_def_0_150.jpeg -> 300x300 마스크 (픽셀 1개당 0/1)
```

불량 이미지 3,758장에 결함 영역을 손으로 칠해야 한다. 장당 1분이면 60시간이고, 주조 결함(blow hole, pin hole, shrinkage) 구분에는 전문 지식이 필요하다. **제조 AI에서 라벨링 비용이 모델 개발보다 큰 전형적 상황.** ImageNet/ADE20K 사전학습 가중치를 가져와도 "주조 결함"이라는 클래스를 가르치려면 결국 마스크가 필요하다.

#### Grad-CAM이 정석이지만 ML.NET에서는 불가능하다

Grad-CAM은 **마지막 컨볼루션층의 출력과 그 출력에 대한 그래디언트**가 필요하다. `ImageClassificationTrainer`는 내부 TensorFlow 그래프를 봉인하므로 중간층 출력도 그래디언트도 꺼낼 수 없다.

> 이 프로젝트에서 **같은 벽에 세 번째로 부딪혔다.** ONNX export 불가, 배포 크기 수백 MB, 그리고 XAI 불가. 모두 "모델이 블랙박스로 봉인되어 있다"는 한 가지 원인이다. 이것이 TorchSharp로 옮긴 유일하고 충분한 이유다.

---

### 사전학습 가중치 조달

TorchSharp의 TorchVision은 **아키텍처 정의만** 제공하고 ImageNet 가중치는 들어있지 않다. Python에서 export해야 하는데, 그러면 "파이썬 없이 C#으로"라는 전제가 깨진다.

**해결 : HuggingFace에 TorchSharp 포맷 변환본이 있었다.**

```
https://huggingface.co/yueyinqiu/vision-TorchSharp
  -> MobileNet_V2_Weights.IMAGENET1K_V2   (13.5MB)
```

| 확인 항목 | 결과 |
| :--- | :--- |
| 포맷 | TorchSharp 자체 `.dat` (헤더 확인) |
| 라이선스 표기 | **없음** (원본 torchvision은 BSD-3-Clause) |
| 업로드 시점 | 약 2년 전 |
| TorchSharp 0.107.0 호환성 | ✅ 그대로 로딩됨 |

```
헤더 16바이트 : BA-02-13-66-65-61-74-75-72-65-73-2E-30-2E-30-2E
                ^^^^^ ^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                314   19  "features.0.0."
```
`0xBA02` = 314 (텐서 개수), 이후 텐서 이름이 평문 ASCII. pickle(`0x80`)도 ZIP(`PK`)도 아니다.

> 가중치 파일은 저장소에 포함하지 않는다(`.gitignore`의 `/models`). 라이선스 표기가 없는 변환본이므로 재배포하지 않고 각자 다운로드하는 전제로 쓴다.

#### 로딩 검증
```
✅ 로딩 성공
   텐서 개수 : 314
   첫 텐서 : classifier.1.weight  shape=[1000,1280]
   평균=0.000001  표준편차=0.052431
```
**표준편차 0.0524가 핵심이다.** `load()`가 예외 없이 끝나도 키 이름이 안 맞으면 조용히 건너뛰어 랜덤 초기화 상태로 남을 수 있다. 학습된 가중치라면 표준편차가 0.1~0.5 범위에 들고, 전부 0이거나 1에 가까우면 실질적으로 실패한 것이다.

---

### 단계 1 : 데이터로더 + 모델 정의

#### ML.NET이 감춰줬던 것들이 전부 드러난다

| 단계 | ML.NET | TorchSharp |
| :--- | :--- | :--- |
| 이미지 로딩 | `LoadRawImageBytes` 한 줄 | `SKBitmap.Decode` + 리사이즈 + 정규화 직접 구현 |
| 배치/셔플 | `IDataView`가 처리 | `Dataset` 상속 + `DataLoader` |
| 전이학습 | `Arch = MobilenetV2` 한 줄 | 1000클래스로 로딩 -> 분류층 교체 |
| 백본 동결 | 자동 (`Froze 2 variables`) | `requires_grad = false` 직접 설정 |
| 학습 | `Fit()` 한 줄 | `zero_grad -> forward -> backward -> step` |
| 메모리 관리 | 불필요 | `NewDisposeScope()` 필수 |

#### 검산 결과
```
parameters() 텐서 개수 : 158   (기대: 158 = features 156 + classifier 2)
named_children() : features / classifier 2개만
전체 파라미터     : 2,226,434
학습가능(동결 전) : 2,226,434
학습가능(동결 후) : 2,562        <- Linear(1280,2) = 1280x2 + 2
feature map shape : [1,1280,7,7]  <- Grad-CAM의 입력
```

**`2,562`가 ML.NET 로그의 `Froze 2 variables`와 정확히 대응한다.** 222만 중 0.12%만 학습한다.

#### 트러블슈팅 : 모듈이 중복 등록되는 문제

처음 진단 출력이 이랬다.
```
_backbone                 : 파라미터 텐서 158개   <- features(156) + 1000클래스 분류층(2)
<Features>k__BackingField : 파라미터 텐서 156개   <- _backbone 내부와 동일 객체 (중복!)
_classifier               : 파라미터 텐서   2개
전체 파라미터 : 3,507,434   (기대보다 128만 개 많음)
```

**원인 두 가지**
1. `RegisterComponents()`는 리플렉션으로 **필드**를 스캔하는데, 자동 구현 프로퍼티의 백킹 필드(`<Features>k__BackingField`)까지 읽는다. 같은 텐서가 두 경로로 등록됐다.
2. 전체 모델(`_backbone`)을 필드로 들고 있어, `forward()`에서 호출조차 하지 않는 **1000클래스 분류층(128만 파라미터)이 학습 대상에 남았다.**

**해결**
```csharp
// ❌ 기존
private readonly Module<Tensor,Tensor> _backbone;
public Module<Tensor,Tensor> Features { get; }      // 백킹 필드 생성 -> 중복 등록

// ✅ 수정
private readonly Module<Tensor,Tensor> features;    // 필드로 선언
private readonly Sequential classifier;
public Module<Tensor,Tensor> Features => features;  // 식 본문 -> 백킹 필드 없음
```
가중치 로딩이 끝나면 `features`만 보관하고 전체 모델 참조를 놓는다. 

---

### 단계 2 : 학습 루프

#### ML.NET의 `Fit()` 한 줄이 펼쳐진다
```csharp
foreach (var batch in trainLoader)
{
    using var d = torch.NewDisposeScope();   // 없으면 메모리가 수 GB로 폭증

    optimizer.zero_grad();                   // 그래디언트는 누적된다. 안 비우면 발산
    var output = model.forward(data);
    var loss = lossFn.forward(output, label);
    loss.backward();                         // 역전파
    optimizer.step();                        // 파라미터 갱신
}
```

#### 셔플을 빼면 학습이 통째로 망가진다
`_items`는 폴더 순서(불량 전부 -> 정상 전부)로 담긴다. 셔플 없이 앞 90%를 train으로 자르면 **train에 정상이 거의 없고 val은 전부 정상**이 된다. 그래도 학습은 "돌아가고", val 정확도가 0% 또는 100%로 고정되는 증상으로만 나타난다.

```
train 구성 : 불량 3381 / 정상 2589
val   구성 : 불량  377 / 정상  286   <- 양쪽에 두 클래스가 있어야 정상
```
ML.NET의 `TrainTestSplit()`이 해줬던 일이다. 

#### TorchSharp 고유의 조용한 실패 지점들

| 실수 | 증상 (예외는 나지 않는다) |
| :--- | :--- |
| `RegisterComponents()` 누락 | `parameters()`가 비어 loss가 전혀 줄지 않음 |
| `NewDisposeScope()` 누락 | 메모리가 수 GB로 폭증 |
| `train()`/`eval()` 미전환 | 검증에도 Dropout 적용되어 정확도 저하 |
| `zero_grad()` 누락 | 그래디언트 누적 -> 발산 |
| `no_grad()` 누락 | 평가 중 계산 그래프가 쌓임 |
| `forward`에 softmax | `CrossEntropyLoss`와 이중 적용 |

ML.NET에서는 **존재하지 않던 실수 지점들**이다. 이것이 전환의 실제 비용이다.

---

### 성능 재현 시도

목표는 ML.NET의 **test Accuracy 99.02% / FN 2건** 재현이었다.

| # | 설정 | 최고 val | test Acc | **test FN** | FP | 시간 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | lr 1e-3, 224px | 97.44% | 97.76% | 12 | 4 | 17.1분 |
| 2 | lr 1e-2, 224px (R채널 버그) | 98.64% | 97.62% | 12 | 5 | 8.5분 |
| 3 | lr 1e-2, 224px (버그 수정) | 96.83% | 96.36% | 22 | 4 | 6.9분 |
| 4 | lr 1e-2 + 감쇠, 300px | 97.13% | 96.08% | **27** | 1 | 22.2분 |
| — | **ML.NET** | 99.57% | **99.02%** | **2** | 5 | 2.5분 |

#### 실험 1 -> 2 : 학습률 (가설 부분 적중)
`train acc 97.42%` vs `val acc 97.13%`로 거의 붙어 있었다. **과적합이 아니라 학습 부족(underfitting)** 이다. Adam 1e-3은 '전체 네트워크 fine-tuning'의 표준값이고, 분류층만 학습할 때는 보수적이다. ML.NET 로그에는 `LearningRate 0.01 -> 0.0073` 감쇠 기록이 있었다.

1e-2로 올리니 **Epoch 0에 이미 98.04%** (이전 93.97%). 수렴 속도 문제는 해결됐지만 **test FN은 12건 그대로.**

#### 실험 2 -> 3 : R채널 버그 (수정했더니 더 나빠짐)
전처리 코드에서 오타를 발견했다.
```csharp
data[0 + plane + i] = ...   // ❌ '*' 가 '+' 로 되어 있었다
data[1 + plane + i] = ...
data[2 + plane + i] = ...
```
인덱스가 `plane+i`, `plane+1+i`, `plane+2+i`가 되어 **R채널 영역이 전부 0으로 남고 G/B가 1칸씩 밀려 겹쳐 쓰인다.** 즉 R채널이 통째로 사라진 입력으로 학습하고 있었다.

**고쳤더니 FN이 12 -> 22건으로 나빠졌다.** 주조품이 회색조(R≈G≈B)라 R채널 손실이 치명적이지 않았고, 오히려 "상수 채널 1개 + 변별력 있는 2채널" 구성이 우연히 유리했을 가능성이 있다. 단정할 수는 없고, **둘 다 노이즈 범위**라는 설명이 더 그럴듯하다.

> 예외가 나지 않고 정확도만 바뀌는 전형적인 버그다. 발견 경로는 Grad-CAM 코드를 쓰면서 같은 전처리를 다시 읽은 것이었다.

#### 실험 3 -> 4 : 학습률 감쇠 (작동했지만 무의미)
```
lr -> 0.009400 -> 0.008836 -> 0.008306   <- ML.NET 로그와 소수점까지 일치
```
감쇠는 정확히 걸렸다. 그런데 **혼동행렬 네 숫자가 감쇠 전과 완전히 동일**했다.

원인 : 최고 val이 **Epoch 0**이었다. 감쇠는 Epoch 0이 끝난 뒤 처음 적용되므로, **저장된 모델에는 감쇠가 한 번도 적용되지 않았다.** `Patience 3`이 먼저 걸려 감쇠가 효과를 낼 구간에 도달하지 못했다. 스케줄러 추가 자체가 무의미했다.

#### 실험 4 : 입력 해상도 300px (val은 올랐으나 test는 악화)
주조 결함은 작은 점 형태이므로 300->224 축소(면적 44% 손실)에서 뭉개졌으리라 추정했다. 300px로 올리니 **val은 97.13%로 개선**됐고 감쇠도 Epoch 2까지 효과를 냈다. 그런데 **test FN은 27건으로 더 나빠졌다.**

---

### 재현 실패 원인 : val과 test의 분포가 다르다

단계 3(오분류 분석)에서 발견한 것이 답이었다.

```
test/def_front 파일명 조사
  cast_def_0_1591.jpeg    <- 기존 규칙
  new__0_5223.jpeg        <- ???
```

| 폴더 | `cast_` | **`new__`** |
| :--- | ---: | ---: |
| train/def_front | 3,758 | **0** |
| train/ok_front | 2,875 | **0** |
| **test/def_front** | 416 | **37** |
| test/ok_front | 262 | **0** |

**`new__` 유형은 test의 불량에만 37장 있고, 학습 데이터에는 한 장도 없다.**

```
val은 train에서 분할 -> new__ 0장
test만 new__ 37장 포함
```

**그래서 val을 기준으로 최적화하면 test의 `new__`에 대해서는 아무것도 개선되지 않는다.** 네 번의 실험에서 val이 오를 때마다 test FN이 따라 내려오지 않은 이유가 이것이다.

FN 12건 시점의 유형별 분석 :

| 유형 | test 불량 | FN | **FN율** |
| :--- | ---: | ---: | ---: |
| `cast_def_*` (학습된 유형) | 416 | 8 | **1.9%** |
| `new__*` (미학습 유형) | 37 | 4 | **10.8%** |

`new__`는 test의 5%뿐인데 FN의 33%를 차지한다. **512x512 검증에서 본 도메인 시프트가 원래 test 폴더 안에 이미 섞여 있었다.**

> ML.NET은 FN 2건에 `new__`가 없었다. 즉 미학습 유형에도 더 강건했다. **왜 그런지는 ML.NET 내부를 볼 수 없어 특정할 수 없다.**

#### 오분류 샘플의 성격
```
불량확률 오름차순 = 모델이 '정상'이라고 확신한 순
  cast_def_0_1591.jpeg   0.000   확신하며 틀림 - 구제 불가
  new__0_5223.jpeg       0.007   확신하며 틀림 - 구제 불가
  cast_def_0_150.jpeg    0.030   확신하며 틀림 - 구제 불가   <- ML.NET도 놓친 것
  ...
  cast_def_0_9921.jpeg   0.376   임계값 0.30까지 낮추면 구제
  cast_def_0_1413.jpeg   0.483   임계값 0.45로 구제 가능

임계값 0.50 -> 0.20 으로 낮춰도 FN 12 -> 9건 (3건만 구제)
```
**`cast_def_0_1591`은 불량확률 0.000이다.** 모델이 "정상"이라고 완전히 확신한다. 임계값 조정으로는 손댈 수 없는 영역이다.

---

### 결론

| 조정 축 | 결과 |
| :--- | :--- |
| 학습률 (1e-3 ↔ 1e-2) | val은 바뀌지만 test FN은 12~22건 |
| 학습률 감쇠 | val 개선, test FN 변화 없음 |
| R채널 버그 수정 | FN 오히려 증가 |
| 입력 해상도 (224 ↔ 300) | val 개선, test FN 악화 |

**모든 최적화 조정이 val만 개선하고 test FN은 개선하지 못했다.** 남은 축은 데이터 쪽인데, `new__` 유형을 학습에 포함시키면 test가 오염된다.

> **정직한 결론** : TorchSharp로 ML.NET 파이프라인을 재현해 test 96~98%를 얻었다. ML.NET의 99.02%와의 차이는 **test에만 존재하는 미학습 유형(`new__` 37장)에 대한 강건성 차이**이며, ML.NET 내부를 볼 수 없어 원인을 특정할 수 없다. 무한정 쫓을 사안이 아니다.

성능 수치를 쫓다가 **데이터셋 자체의 구성 문제를 발견한 것**이 이 과정의 실제 수확이다.

---

### 단계 3 : Grad-CAM 구현

#### 원리
```
1) 마지막 컨볼루션층의 출력 A를 얻는다.              -> [1, 1280, 7, 7]
2) 관심 클래스 점수를 A로 미분한 그래디언트 G를 얻는다.  -> [1, 1280, 7, 7]
3) G를 공간 방향으로 평균해 채널별 가중치 w를 만들고,
   A를 w로 가중합한 뒤 ReLU를 건다.                  -> [7, 7]
```
- `w[k]`가 크다 = "k번째 채널이 켜지면 이 클래스 점수가 올라간다"
- `A[k]`가 크다 = "k번째 채널이 이 위치에서 켜졌다"
- 둘을 곱해 더하면 **이 클래스 판단에 기여한 위치**가 나온다

> LightGBM 실습의 `CalculateFeatureContribution`과 발상이 같다. 기여도를 '피처'가 아니라 **'공간 위치'** 에 대해 계산하는 것이다.

#### 구현상 핵심 

**① `enable_grad()` : 추론인데 그래디언트를 켠다**
```csharp
using (torch.enable_grad())
{
    featureMap = _model.Features.forward(input);
    featureMap.retain_grad();
```
보통 추론에는 `no_grad()`를 쓰지만 **Grad-CAM은 그래디언트 자체가 목적이다.** `no_grad()`로 감싸면 계산 그래프가 없어 `backward()`가 실패한다.

**② `retain_grad()` : 중간 텐서의 그래디언트를 남긴다**

PyTorch/TorchSharp는 말단(leaf) 텐서에만 `.grad`를 보관한다. `featureMap`은 중간 결과이므로 기본적으로 버려진다. 이것을 빼면 `featureMap.grad`가 `null`이고 원인을 찾기 어렵다.

**③ `ClassifierForward()` : 모델을 쪼개 둔 이유**
```csharp
var pooled = adaptive_avg_pool2d(featureMap, new long[]{1,1});
logits = _model.ClassifierForward(pooled.flatten(1));   // forward()를 쓰지 않는다
```
`model.forward(input)`를 호출하면 features를 다시 실행하므로, 우리가 붙잡은 `featureMap`과 다른 텐서 객체에 그래디언트가 흐른다. 그러면 `featureMap.grad`가 `null`이 된다. 

> hook(`register_forward_hook`)으로도 가능하지만 등록/해제 관리가 필요하고 TorchSharp에서 시그니처가 까다롭다.

#### 히트맵 해상도 한계
```
입력 224x224 -> feature map 7x7   (한 칸 = 원본 32x32 픽셀)
입력 300x300 -> feature map 10x10 (한 칸 = 원본 30x30 픽셀)
```
Bilinear 보간으로 224x224까지 늘려 부드럽게 보이지만 **정보량은 여전히 49칸(또는 100칸)이다.** 늘려도 정밀도가 생기지는 않는다. 바운딩박스 경계가 뭉툭한 이유이기도 하다.

> **"결함 영역"이 아니라 "주목 영역"이다.** Grad-CAM은 마스크 라벨로 학습된 것이 아니라 "모델이 이 클래스로 판단할 때 기여한 영역"을 보여준다. UI 문구를 과장하면 나중에 본인도 헷갈린다.

#### 히트맵은 이미지마다 재정규화된다
```csharp
cam = (cam - camMin) / denom;   // 0~1로 정규화
```
**정상품 이미지에도 반드시 빨간 영역이 생긴다.** 기여도가 전반적으로 낮아도 그중 최대값이 1.0이 되기 때문이다. 서로 다른 이미지의 빨간색끼리 비교하면 안 된다. 흔한 오해 지점.

---

### 단계 4 : WPF GUI 통합

#### 화면 구성
```
[이미지 선택] [바운딩박스 ☑] [임계값 ━━●━━ 0.50] [표시: 오버레이 ▼] [컬러바 ☐]
┌────────────────────┬────────────────────┐
│       원본          │  Grad-CAM (주목영역) │
│                    │  + 바운딩박스        │
└────────────────────┴────────────────────┘
[ 불  량 ]  불량확률 ████████░░ 87.3%
 확신도 87.3%  정상확률 ██░░░░░░░░ 12.7%
```

| 표시 모드 | 용도 |
| :--- | :--- |
| 오버레이 | 원본 위에 반투명 합성. 기여도 낮은 곳은 원본이 보인다 |
| 히트맵만 | matplotlib `imshow` 스타일. 기여도 분포 자체를 읽을 때 |
| 원본+히트맵 | 좌우 비교. 실제 결함 위치와 대조할 때 가장 유용 |

#### 설계 의도
- **판정은 `Score`로 직접 한다.** `PredictedLabel`은 임계값 0.5에 고정되어 조정이 불가능하다
- **임계값 슬라이더는 재추론하지 않는다.** 히트맵을 보관해 두고 바운딩박스만 다시 그려 즉시 반응한다
- **"판정 경계" 경고** : `|Score - 0.5| < 0.05`이면 표시. 모델이 사실상 "모르겠다"고 말한 건을 자동 판정하지 말고 사람에게 넘기기 위함(reject option)
- **히트맵 해상도를 UI에 명시** : "7x7 (한 칸 = 원본 32x32px)". 적어두지 않으면 픽셀 단위로 결함을 찾았다고 오해한다

#### WPF에서 반드시 지킬 것
```csharp
result = await Task.Run(() => _GradCAM!.Explain(imagePath));
```
Grad-CAM은 역전파를 포함해 추론보다 2~3배 느리다. UI 스레드에서 하면 창이 멈춘듯 보인다. 448x448 픽셀을 `SetPixel`로 그리는 렌더링도 20만 번 호출이라 분리했다.

```csharp
bmp.Freeze();   // ToBitmapSource 안에서
```
**없으면 "다른 스레드가 소유한 개체" 예외가 난다.** 백그라운드 스레드에서 만든 `BitmapSource`를 UI에서 쓰려면 필수다.

#### jet 컬러맵 재현
처음 구현한 삼각파 근사는 값 0에서 파랑이 1.0으로 나와, matplotlib의 **진한 남색 `(0,0,0.5)`** 과 달랐다. 구간별 정의로 다시 썼다.
```
0.0 -> (0,0,0.5) 남색 / 0.25 -> (0,1,1) 청록 / 0.5 -> (0.5,1,0.5)
0.75 -> (1,1,0) 노랑 / 1.0 -> (0.5,0,0) 진한 빨강
```
> jet은 지각적으로 균일하지 않아 청록 구간에서 값 차이가 과장되고, 색각 이상이 있으면 구분이 어렵다. 정량 비교가 목적이면 viridis가 낫다. 여기서는 Grad-CAM 논문의 관례를 따랐다.

---

#### 환경 설정
```bash
dotnet add package TorchSharp --version 0.107.0
dotnet add package libtorch-cpu-win-x64 --version 2.10.0   # 76MB, Windows만
dotnet add package TorchVision --version 0.107.0
```
> `TorchSharp-cpu` 메타패키지는 Linux/macOS 네이티브까지 받아 **약 265MB**가 된다. Windows만 쓸 거라면 `libtorch-cpu-win-x64`를 직접 지정해 **76MB**로 줄인다.

GPU는 포기했다. `libtorch-cuda-12.8`이 필요한데 CUDA Toolkit이 설치되어 있지 않고, ML.NET 실습에서 이미 같은 결론에 도달했다. **CPU로 7~22분이면 실습에 충분하다.**

---

### 이 전환에서 얻은 것

| # | 발견 | 근거 |
| :--- | :--- | :--- |
| 1 | **XAI를 위해서는 프레임워크를 바꿔야 했다** | ML.NET은 중간층/그래디언트 접근 불가. Grad-CAM 자체가 성립하지 않음 |
| 2 | **추상화의 비용은 유연성** | `Fit()` 한 줄이 6단계로 펼쳐졌고, '조용한 실패' 지점이 6개 생겼다 |
| 3 | **성능은 재현했으나 완전하지 않다** | 96~98% vs 99.02%. 차이의 원인을 특정할 수 없다 |
| 4 | **val 개선이 test 개선이 아니다** | 네 번의 실험 모두 val만 올랐다. 분포가 달랐기 때문 |
| 5 | **데이터셋 구성을 먼저 확인했어야 했다** | `new__` 37장이 test에만 있었다. 파일명만 세어봐도 알 수 있었다 |
| 6 | **버그 수정이 성능을 올리지는 않는다** | R채널 복구 후 FN이 12->22건. 옳은 수정과 좋은 결과는 별개 |
</details>
