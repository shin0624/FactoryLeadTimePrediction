using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Microsoft.Win32;
using SkiaSharp;
using FactoryLeadTimePrediction.DeepLearning.B_TorchSharpGradCAM;

namespace CastingInspector.Wpf
{
    public partial class MainWindow : Window
    {
        private const string ModelFileName = "TorchSharp_CastingDefect.dat";
        private const string PretrainedFileName = "MobileNet_V2_Weights.IMAGENET1K_V2";
        private const int ImageSize = 224;
        private const int OverlaySize = 448; // 화면 표시용. 히트맵 정보량은 여전히 7x7이다.

        private const float DefectThreshold = 0.50f;
        // 임계값 실습 결과에 따라 조정한다. 모델을 재학습하지 않고 이 상수 하나로 판정 성향이 바뀐다.

        private MobileNetV2Classifier? _model;
        private GradCAM? _GradCAM;
        private float[,]? _lastHeatmap;   // 슬라이더로 박스만 다시 그리기 위해 보관
        private string? _lastImagePath;

        public MainWindow()
        {
            InitializeComponent();
            SldThreshold.ValueChanged += (s, e) =>
            {
                TxtThresholdVal.Text = $"{SldThreshold.Value:F2}";
                RedrawBoundingBox();   // 재추론 없이 박스만 갱신. 히트맵은 이미 있으므로 즉시 반응한다.
            };
            ChkBox.Checked += (s, e) => RedrawBoundingBox();
            ChkBox.Unchecked += (s, e) => RedrawBoundingBox();



            CmbMode.SelectionChanged += async (s, e) => await RerenderAsync();
            ChkColorBar.Checked += async (s, e) => await RerenderAsync();
            ChkColorBar.Unchecked += async (s, e) => await RerenderAsync();
            // 모드만 바꿀 때는 재추론이 필요 없다. 보관한 히트맵으로 다시 그리기만 한다.
            
            _ = LoadModelAsync();
        }

        private async Task LoadModelAsync()
        {
            try
            {
                string modelPath = ResolveFile(ModelFileName);
                string pretrainedPath = ResolveFile(PretrainedFileName);

                await Task.Run(() =>
                {
                    var m = new MobileNetV2Classifier(pretrainedPath, numClasses: 2);
                    m.load(modelPath);
                    m.eval();
                    _model = m;
                    _GradCAM = new GradCAM(m, ImageSize);
                });
                /*
                Task.Run으로 감싸는 이유
                    - 모델 로딩과 libtorch 초기화는 수 초가 걸린다.
                    - UI 스레드에서 하면 창이 흰색으로 멈추고 "응답 없음"이 뜬다.
                    - await 이후에는 UI 스레드로 돌아오므로 아래 UI 갱신이 안전하다.

                사전학습 가중치 파일이 왜 또 필요한가
                    - MobileNetV2Classifier 생성자가 구조를 만들 때 쓴다.
                    - 그 값은 곧 load()로 덮어쓰이므로 실제로는 '구조 생성용'이다.
                    - 생성자에 null을 넘겨도 되지만, 경고 메시지가 나오므로 그대로 넘긴다.
                */

                TxtStatus.Text = $"모델 로딩 완료 — {ModelFileName}";
                BtnOpen.IsEnabled = true;
            }
            catch (Exception ex)
            {
                TxtStatus.Text = "모델 로딩 실패";
                MessageBox.Show(
                    $"모델을 불러오지 못했습니다.\n\n{ex.Message}\n\n" +
                    "콘솔 프로젝트에서 Step2_Train()을 먼저 실행해 models/ 에 모델을 생성하세요.",
                    "오류", MessageBoxButton.OK, MessageBoxImage.Error);
                // 모델 파일은 .gitignore 대상이므로, 저장소를 clone한 사람은 반드시 이 상황을 만난다.
            }
        }

        private string ResolveFile(string fileName) // 실행 위치에 상관없이 models/ 아래 파일을 찾는다.
        {
            string dir = AppDomain.CurrentDomain.BaseDirectory;
            for (int i = 0; i < 7; i++)
            {
                string candidate = Path.Combine(dir, "models", fileName);
                if (File.Exists(candidate)) return candidate;
                dir = Path.Combine(dir, "..");
            }
            throw new FileNotFoundException($"models/{fileName} 을 찾을 수 없습니다.");
            // bin\Debug\net8.0-windows 에서 실행되므로 상대경로가 직관과 어긋난다.
            // 콘솔 실습의 ResolveCsvPath()와 같은 패턴이다.
        }

        private async void BtnOpen_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFileDialog
            {
                Title = "검사할 주조품 이미지 선택",
                Filter = "이미지 파일|*.jpeg;*.jpg;*.png|모든 파일|*.*"
            };
            if (dialog.ShowDialog() != true) return;

            string imagePath = dialog.FileName;
            _lastImagePath = imagePath;

            TxtStatus.Text = "추론 + Grad-CAM 계산 중...";
            BtnOpen.IsEnabled = false;
            BoxCanvas.Children.Clear();

            // ---------- (1) 원본 표시 ----------
            ImgOriginal.Source = LoadBitmap(imagePath);

            // ---------- (2) 추론 + Grad-CAM ----------
            GradCAM.Result result;

            try
            {
                result = await Task.Run(() => _GradCAM!.Explain(imagePath));
                /*
                Task.Run 안에서는 Explain() 하나만 호출한다.
                    - Grad-CAM은 역전파를 포함하므로 추론보다 2~3배 느리다. UI 스레드를 막으면 창이 얼어 보인다.
                    - 렌더링(히트맵 -> 비트맵)은 RerenderAsync()로 분리했다.
                      표시 모드를 바꿀 때 재추론 없이 다시 그리기만 하면 되기 때문이다.

                _GradCAM 은 스레드 안전하지 않다.
                    - 버튼을 비활성화해 한 번에 한 건만 처리하므로 안전하다.
                */

                _lastHeatmap = result.Heatmap;
                // 보관해 둔다. 표시 모드/임계값을 바꿀 때 이것으로 다시 그린다.

                await RerenderAsync();
                // 현재 선택된 모드(오버레이 / 히트맵만 / 원본+히트맵)로 ImgHeatmap에 그린다.
            }
            catch (Exception ex)
            {
                TxtStatus.Text = "계산 실패";
                MessageBox.Show($"Grad-CAM 계산 중 오류:\n\n{ex.Message}", "오류");
                BtnOpen.IsEnabled = true;
                return;
            }

            // ---------- (3) 히트맵 표시 ----------
            _lastHeatmap = result.Heatmap;

            // ---------- (4) 판정 ----------
            bool isDefect = result.DefectScore >= DefectThreshold;
            // PredictedLabel(argmax)이 아니라 Score로 직접 판정한다. 그래야 임계값을 바꿀 수 있다.

            TxtVerdict.Text = isDefect ? "불  량" : "정  상";
            BorderVerdict.Background = isDefect
                ? new SolidColorBrush(Color.FromRgb(0xB0, 0x30, 0x30))
                : new SolidColorBrush(Color.FromRgb(0x2E, 0x7D, 0x32));

            float confidence = isDefect ? result.DefectScore : result.OkScore;
            TxtConfidence.Text = $"확신도 {confidence:P1}";

            if (Math.Abs(result.DefectScore - 0.5f) < 0.05f)
            {
                TxtConfidence.Text += "  ⚠ 판정 경계";
                /*
                콘솔 실습에서 Score=[0.498, 0.502] 같은 사례가 나왔다.
                모델이 사실상 "모르겠다"고 말한 것인데 argmax는 단정적인 답으로 바꿔버린다.
                실제 검사 라인이라면 이런 건은 사람에게 넘기는 것이 맞다(reject option).
                */
            }

            double barMax = ((FrameworkElement)BarDefect.Parent).ActualWidth;
            BarDefect.Width = barMax * result.DefectScore;
            BarOk.Width = barMax * result.OkScore;
            TxtDefectPct.Text = $"{result.DefectScore:P1}";
            TxtOkPct.Text = $"{result.OkScore:P1}";

            // ---------- (5) 바운딩박스 ----------
            RedrawBoundingBox();

            TxtStatus.Text = "완료";
            BtnOpen.IsEnabled = true;
        }

        private async Task RerenderAsync() // 보관한 히트맵으로 다시 그린다 (재추론 없음)
        {
            if (_lastHeatmap is null || _lastImagePath is null) return;

            var mode = CmbMode.SelectedIndex switch
            {
                1 => GradCAM.RenderMode.HeatmapOnly,
                2 => GradCAM.RenderMode.SideBySide,
                _ => GradCAM.RenderMode.Overlay
            };
            bool bar = ChkColorBar.IsChecked == true;
            string path = _lastImagePath;
            var heat = _lastHeatmap;

            var bmp = await Task.Run(() =>
            {
                using var sk = GradCAM.Render(path, heat, mode, alpha: 0.5f, outSize: 448, colorBar: bar);
                return ToBitmapSource(sk);
            });
            /*
            Task.Run으로 감싸는 이유
                - 448x448 픽셀을 하나씩 SetPixel 하므로 20만 번 호출이다. UI 스레드에서는 버벅인다.
                - 재추론(Grad-CAM)은 하지 않으므로 훨씬 가볍지만, 그래도 분리하는 편이 안전하다.

            ToBitmapSource 안에서 Freeze()를 호출하므로 다른 스레드에서 만들어도 UI에서 쓸 수 있다.
            Freeze()가 없으면 "다른 스레드가 소유한 개체" 예외가 난다. ★WPF의 흔한 함정
            */

            ImgHeatmap.Source = bmp;
            RedrawBoundingBox();
        }

        private void RedrawBoundingBox() // 보관한 히트맵으로 박스만 다시 그린다 (재추론 없음)
        {
            BoxCanvas.Children.Clear();
            if (_lastHeatmap is null || ChkBox.IsChecked != true) { UpdateDetail(null); return; }

            float th = (float)SldThreshold.Value;
            var box = GradCAM.FindBoundingBox(_lastHeatmap, th);
            if (box is null) { UpdateDetail(null); return; }

            var (bx, by, bw, bh) = box.Value;
            int hSize = _lastHeatmap.GetLength(0);

            // 히트맵 좌표 -> 실제 표시 영역 좌표로 변환
            double dispW = ImgHeatmap.ActualWidth;
            double dispH = ImgHeatmap.ActualHeight;
            if (dispW <= 0 || dispH <= 0) { UpdateDetail(box); return; }

            double scaleX = dispW / hSize;
            double scaleY = dispH / hSize;
            /*
            좌표 변환의 한계
                - Image의 Stretch="Uniform" 때문에 실제 그려진 이미지가 컨트롤보다 작을 수 있다
                  (레터박스). 정밀하게 맞추려면 그 여백을 계산해야 한다.
                - 여기서는 정사각형 이미지(224x224 -> 448x448)만 다루므로 왜곡이 작다.
                  다른 비율의 이미지를 넣으면 박스가 어긋난다. 알고 쓸 것.
            */

            var rect = new System.Windows.Shapes.Rectangle
            {
                Width = bw * scaleX,
                Height = bh * scaleY,
                Stroke = new SolidColorBrush(Colors.Yellow),
                StrokeThickness = 2,
                StrokeDashArray = new DoubleCollection { 4, 2 }
            };
            Canvas.SetLeft(rect, bx * scaleX);
            Canvas.SetTop(rect, by * scaleY);
            BoxCanvas.Children.Add(rect);

            UpdateDetail(box);
        }

        private void UpdateDetail((int x, int y, int w, int h)? box)
        {
            string boxText = box is null
                ? "없음 (임계값 이상 영역 없음)"
                : $"{box.Value.w}x{box.Value.h} @ ({box.Value.x},{box.Value.y})";

            string file = _lastImagePath is null ? "—" : Path.GetFileName(_lastImagePath);
            TxtDetail.Text = $"파일 : {file}   |   주목영역 : {boxText}   |   " +
                             $"히트맵 해상도 : 7x7 (한 칸 = 원본 32x32px)";
            /*
            히트맵 해상도를 UI에 명시하는 이유
                - 448x448로 부드럽게 보이지만 실제 정보량은 49칸뿐이다.
                - 이것을 적어두지 않으면 "픽셀 단위로 결함을 찾았다"고 오해하게 된다.
                - 세그멘테이션 모델과의 차이를 사용자가 알아야 한다.
            */
        }

        private static BitmapImage LoadBitmap(string path)
        {
            var bmp = new BitmapImage();
            bmp.BeginInit();
            bmp.UriSource = new Uri(path);
            bmp.CacheOption = BitmapCacheOption.OnLoad;
            bmp.EndInit();
            return bmp;
            // OnLoad : 즉시 메모리로 읽고 파일 핸들을 닫는다.
            //          기본값(OnDemand)은 핸들을 유지해 파일 삭제/덮어쓰기를 막고 캐시 문제를 만든다.
        }

        private static BitmapSource ToBitmapSource(SKBitmap skBitmap) // SkiaSharp -> WPF 변환
        {
            using var image = SKImage.FromBitmap(skBitmap);
            using var data = image.Encode(SKEncodedImageFormat.Png, 100);
            using var stream = new MemoryStream(data.ToArray());

            var bmp = new BitmapImage();
            bmp.BeginInit();
            bmp.StreamSource = stream;
            bmp.CacheOption = BitmapCacheOption.OnLoad;
            bmp.EndInit();
            bmp.Freeze();
            return bmp;
            /*
            PNG로 인코딩해 거치는 이유
                - SKBitmap의 픽셀 버퍼를 WriteableBitmap에 직접 복사하는 것이 더 빠르지만,
                  색 순서(BGRA/RGBA)와 스트라이드를 맞춰야 해 실수가 잦다.
                - 장당 1회 변환이므로 PNG 경유로도 충분히 빠르다.

            Freeze()
                - 다른 스레드에서도 안전하게 쓸 수 있게 하고 성능도 약간 올라간다.
                - StreamSource는 using으로 스트림이 닫히기 전에 OnLoad로 다 읽어야 한다.
            */
        }
    }
}
