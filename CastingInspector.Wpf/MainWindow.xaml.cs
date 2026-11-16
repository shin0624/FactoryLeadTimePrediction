using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Microsoft.ML;
using Microsoft.Win32;
using FactoryLeadTimePrediction.DeepLearning.A_CastingDefectClassification;
// 콘솔 프로젝트의 DTO를 그대로 재사용

namespace CastingInspector.Wpf
{
    public partial class MainWindow : Window
    {
        private const string ModelPath = @"..\..\..\..\models\CastingDefect_Model.zip";
        // bin\Debug\net8.0-windows 에서 4단계 위로 올라가면 저장소 루트
        // 상대경로라 실행 위치가 바뀌면 깨진다. 아래 ResolveModelPath()에서 보정

        private const int DefectScoreIndex = 0; // Score[0] = "defect" (콘솔의 PrintClassOrder로 확인)
        private const float DefectThreshold = 0.50f;
        // 임계값 실습 결과에 따라 0.45 등으로 바꾸면 된다.
        // 모델을 재학습하지 않고 이 상수 하나로 판정 성향이 바뀜

        private PredictionEngine<CastingImageData, CastingPrediction>? _engine;

        public MainWindow()
        {
            InitializeComponent();
            _ = LoadModelAsync(); // await하지 않고 띄운다. 생성자는 async가 될 수 없기 때문
            // '_ =' 는 "반환값을 의도적으로 버린다"는 표시. 없으면 CS4014 경고가 난다
        }

        private async Task LoadModelAsync() // 모델을 백그라운드 스레드에서 로딩
        {
            try
            {
                string path = ResolveModelPath();

                await Task.Run(() =>
                {
                    var mlContext = new MLContext();
                    var model = mlContext.Model.Load(path, out _);
                    _engine = mlContext.Model.CreatePredictionEngine<CastingImageData, CastingPrediction>(model);
                });
                /*
                왜 Task.Run으로 감싸는가 (WPF의 핵심 원칙)
                    - 모델 파일이 22.8MB이고 TensorFlow 그래프 복원까지 하므로 수 초가 걸린다.
                    - 이것을 UI 스레드에서 하면 창이 흰색으로 멈추고 "응답 없음"이 뜬다.
                    - Task.Run은 이 작업을 스레드풀로 보내고, await가 끝나면 다시 UI 스레드로 돌아온다.
                      그래서 아래 TxtStatus.Text 대입이 안전하다(UI 요소는 UI 스레드에서만 건드릴 수 있다).
                */

                TxtStatus.Text = $"모델 로딩 완료 - {Path.GetFileName(path)}";
                BtnOpen.IsEnabled = true;
            }
            catch (Exception ex)
            {
                TxtStatus.Text = "모델 로딩 실패";
                MessageBox.Show(
                    $"모델을 불러오지 못했습니다.\n\n{ex.Message}\n\n" +
                    "콘솔 프로젝트에서 StartCastingDefectClassification()을 먼저 실행해 모델을 생성하세요.",
                    "오류", MessageBoxButton.OK, MessageBoxImage.Error);
                // 예외를 삼키지 않고 사용자에게 원인과 해결책을 함께 알린다.
                // 모델 파일이 .gitignore로 제외되므로, 저장소를 clone한 사람은 반드시 이 상황을 만난다.
            }
        }

        private string ResolveModelPath() // 실행 위치에 상관없이 모델 파일을 찾는다.
        {
            string dir = AppDomain.CurrentDomain.BaseDirectory;

            for (int i = 0; i < 6; i++) // 최대 6단계까지 상위로 올라가며 탐색
            {
                string candidate = Path.Combine(dir, "models", "CastingDefect_Model.zip");
                if (File.Exists(candidate)) return candidate;
                dir = Path.Combine(dir, "..");
            }
            throw new FileNotFoundException("models/CastingDefect_Model.zip 을 찾을 수 없습니다.");
            /*
            콘솔 실습의 ResolveCsvPath()와 같은 패턴
                - bin\Debug\net8.0-windows\ 에서 실행되므로 상대경로가 직관과 어긋난다.
                - 하드코딩 대신 위로 올라가며 찾으면 Debug/Release, dotnet run/직접실행 모두에서 동작한다.
            */
        }

        private async void BtnOpen_Click(object sender, RoutedEventArgs e) // 파일 선택 -> 미리보기 -> 추론 -> 표시
        {
            var dialog = new OpenFileDialog
            {
                Title = "검사할 주조품 이미지 선택",
                Filter = "이미지 파일|*.jpeg;*.jpg;*.png|모든 파일|*.*"
            };

            if (dialog.ShowDialog() != true) return; // 사용자가 취소한 경우
            string imagePath = dialog.FileName;

            // ---------- (1) 미리보기 표시 ----------
            var bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.UriSource = new Uri(imagePath);
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.EndInit();
            ImgPreview.Source = bitmap;
            /*
            CacheOption = OnLoad 가 중요한 이유
                - 기본값(OnDemand)은 파일 핸들을 계속 열어 둔다. 그러면 그 파일을 삭제하거나
                  덮어쓸 수 없고, 같은 파일을 다시 열 때 옛 이미지가 캐시에서 나온다.
                - OnLoad는 즉시 메모리로 읽고 핸들을 닫는다. 파일을 다루는 앱에서는 사실상 필수다.
            */

            TxtFileName.Text = $"파일 : {Path.GetFileName(imagePath)}";
            TxtStatus.Text = "추론 중...";
            BtnOpen.IsEnabled = false;

            // ---------- (2) 추론 ----------
            var input = new CastingImageData { ImagePath = imagePath, Label = string.Empty };
            // Label은 비워 둔다. 추론 시점에는 정답을 모르는 것이 당연하다.
            // 모델이 MapValueToKey를 포함하지만, 학습 때 만든 사전을 쓰므로 빈 값이어도 동작한다.

            var sw = Stopwatch.StartNew();
            CastingPrediction prediction = await Task.Run(() => _engine!.Predict(input));
            sw.Stop();
            /*
            여기도 Task.Run으로 감싸는 이유
                - 단일 이미지 추론은 수십~수백 ms지만, UI 스레드를 막으면 그동안 창이 얼어 보인다.
                - PredictionEngine은 스레드 안전하지 않다. 지금처럼 버튼이 비활성화되어
                  한 번에 한 건만 처리하는 구조에서는 안전하지만,
                  여러 요청을 동시에 처리하려면 PredictionEnginePool을 써야 한다.

            _engine! 의 '!' 는 null 아님 단언이다.
                - 버튼이 모델 로딩 완료 후에만 활성화되므로 이 시점에 null일 수 없다.
                - 다만 '!'는 컴파일러를 침묵시킬 뿐 런타임 보장은 아니다. UI 상태로 보장하고 있는 것.
            */

            // ---------- (3) 결과 표시 ----------
            float defectScore = prediction.Score[DefectScoreIndex];
            float okScore = prediction.Score[1 - DefectScoreIndex];

            bool isDefect = defectScore >= DefectThreshold;
            // 판정 규칙. PredictedLabel을 쓰지 않고 Score로 직접 판정한다.
            //  그래야 임계값을 바꿀 수 있다. PredictedLabel은 임계값 0.5에 고정되어 있다.

            TxtVerdict.Text = isDefect ? "불  량" : "정  상";
            BorderVerdict.Background = isDefect
                ? new SolidColorBrush(Color.FromRgb(0xB0, 0x30, 0x30))   // 불량 : 붉은색
                : new SolidColorBrush(Color.FromRgb(0x2E, 0x7D, 0x32));  // 정상 : 녹색

            float confidence = isDefect ? defectScore : okScore;
            TxtConfidence.Text = $"확신도 {confidence:P1}";

            // 확률 막대 길이 갱신 (부모 Grid의 실제 폭에 비례)
            double barMaxWidth = ((FrameworkElement)BarDefect.Parent).ActualWidth;
            BarDefect.Width = barMaxWidth * defectScore;
            BarOk.Width = barMaxWidth * okScore;
            TxtDefectPct.Text = $"{defectScore:P1}";
            TxtOkPct.Text = $"{okScore:P1}";

            TxtElapsed.Text = $"추론 시간 : {sw.ElapsedMilliseconds} ms";
            TxtThreshold.Text = $"판정 임계값 : {DefectThreshold:F2}" +
                                (Math.Abs(DefectThreshold - 0.5f) < 0.001f ? " (기본값)" : " (조정됨)");

            // ---------- (4) 애매한 경우 경고 ----------
            if (Math.Abs(defectScore - 0.5f) < 0.05f)
            {
                TxtConfidence.Text += "  ⚠ 판정 경계에 가까움";
            }
            /*
            왜 이 경고가 중요한가
                - 콘솔 실습의 오분류 분석에서 Score=[0.498, 0.502] 같은 사례가 나왔다.
                  모델이 사실상 "모르겠다"고 말한 것인데, argmax는 그것을 단정적인 답으로 바꿔버린다.
                - 실제 검사 라인이라면 이런 건은 자동 판정하지 말고 사람에게 넘기는 것이 맞다.
                  (reject option / human-in-the-loop)
                - 확률을 보여주는 UI의 가장 큰 가치가 여기에 있다.
            */

            TxtStatus.Text = "판정 완료";
            BtnOpen.IsEnabled = true;
        }
    }
}
