using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using TorchSharp;
using static TorchSharp.torch;
using SkiaSharp;

namespace FactoryLeadTimePrediction.DeepLearning.B_TorchSharpGradCAM
{
    public class GradCAM // Grad-CAM 구현 클래스. 모델이 어느 영역을 보고 판단했는지 히트맵으로 시각화한다.
    {
        /*
        Grad-CAM 원리
        - 마지막 컨볼루션 레이어의 출력 A를 얻는다. -> [1, 1280, 7, 7]
        - 관심 클래스의 점수를 A로 미분한 그라디언트 G를 얻는다. -> [1, 1280, 7, 7]
        - G를 공간 방향으로 평균하여 채널별 가중치 w를 만들고, A를 w로 가중합 한 뒤 ReLU를 걸어 히트맵을 만든다. -> [7, 7]

        - w[k]가 크다 : k번째 채널이 켜지면 이 클래스 점수가 올라간다.
        - A[k]가 크다 : k번째 채널이 이 위치에서 켜졌다.
        - 둘을 곱해 더하면 "이 클래스 판단에 기여한 위치"가 나온다.
        - LightGBM 때의 기여도를 피처가 아니라 공간 위치에 계산하는 것.

        ML.NET으로 불가능했던 이유 : Grad-CAM을 위해 중간 층과 그라디언트에 접근해야 하는데, ImageClassificationTrainer는 내부 Tensorflow 그래프에 접근할 수 없게 되어 있음.
        
        */

        private readonly MobileNetV2Classifier _model;
        private readonly int _imageSize;

        // CastingDataset과 동일한 ImageNet 정규화 상수. 학습 때와 추론 때가 반드시 같아야 한다.
        private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
        private static readonly float[] Std = { 0.229f, 0.224f, 0.225f };

        public GradCAM(MobileNetV2Classifier model, int imageSize = 300)
        {
            _model = model;
            _imageSize = imageSize;
        }

        public class Result // Grad-CAM 1건의 결과를 담을 클래스
        {
            public float DefectScore; // 불량 확률(softmax 후)
            public float OkScore; // 정상 확률
            public long PredictedLabel; // argmax 결과 (0 = defect / 1 = ok)
            public float[,] Heatmap = new float[0,0]; // 0 ~ 1로 정규화된 히트맵(7*7을 업샘플한 크기)
            public long ElapsedMs; // 소요시간
        }

        public Result Explain(string imagePath, long? targetClass = null) // 이미지 1장의 Grad-CAM을 계산하는 메서드.
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var result = new Result();

            // 전처리
            var input = LoadImageAsTensor(imagePath).unsqueeze(0); // [1, 3, 300, 300]

            _model.eval(); // 드롭아웃이 켜진 상태면 같은 이미지에 매번 다른 히트맵이 나오게 되어 재현이 불가능하기에 eval()호출이 필요.

            // 순전파 (feature map을 붙잡음)
            Tensor featureMap;
            Tensor logits;

            using(torch.enable_grad())
            {
                /*
                추론에서 왜 그라디언트를 켜는가?
                - 보통 추론에는 그라디언트가 필요 없어서 no_grad()를 쓰는데, Grad-CAM은 그라디언트 자체가 목적이므로.
                - no_grad()로 감싸면 계산 그래프가 만들어지지 않아서 역전파에 실패
                
                */

                featureMap = _model.Features.forward(input); // [1, 1280, 7, 7]
                featureMap.retain_grad();
                /*
                retain_grad()가 필요한 이유
                - TorchSharp, Pytorch는 말단(leaf)텐서에만 .grad를 보관
                - featureMap은 중간 결과이므로, 기본적으로 역전파 후 그라디언트가 버려진다.
                - retain_grad()를 호출하며 이 중간 텐서의 .grad를 남겨준다, 이것을 빼면 featureMap.grad가 null이고, 원인을 찾기 어렵다,.
                */

                var pooled = torch.nn.functional.adaptive_avg_pool2d(featureMap, new long[] {1,1});
                logits = _model.ClassifierForward(pooled.flatten(1)); // [1,2]
                /*
                model.forward(input)을 그대로 쓰지 않는 이유
                - forward()는 내부에서 features를 다시 호출하므로, 붙잡은 featureMap과 다른 텐서 객체에 그라디언트가 흐르게 된다.
                */

            }

            var probs = torch.nn.functional.softmax(logits, dim : 1);
            result.DefectScore = probs[0, CastingDataset.LabelDefect].ToSingle();
            result.OkScore = probs[0, CastingDataset.LabelOk].ToSingle();
            result.PredictedLabel = logits.argmax(1).ToInt64();

            // 역전파 : 관심 클래스 점수를 features map으로 미분
            long cls = targetClass ?? result.PredictedLabel; 
            /*
            어떤 클래스를 설명하는가?
            - 기본값 : 예측된 클래스
            - 불량 판정일 경우 :어디가 결함으로 보였는가?
            - 불량 이미지를 정상으로 판정한 경우(FN) : targetClass = 0 (defect)를 넘겨서 결함을 찾으려면 어디를 봤었어야 했는가를 볼 수도 있음
            
            
            */
            _model.zero_grad(); // 이전 호출의 그라디언트가 남아있으면 더해지기 때문에, 반드시 초기화

            var score = logits[0, cls];
            score.backward(retain_graph:false);
            /*
            softmax 확률이 아니라 로짓을 미분하는 이유
                - 원 논문(Selvaraju et al.)이 softmax 이전 점수를 쓴다.
                - softmax는 클래스 간 경쟁을 포함하므로, 확률을 미분하면 다른 클래스의 영향이 섞인다.
                - 로짓을 쓰면 "이 클래스 점수를 올리는 데 기여한 영역"이 더 깨끗하게 나온다.
            */

            var grad = featureMap.grad;

            if(grad is null)
            {
                throw new InvalidOperationException("featureMap.grad 가 null 입니다. retain_grad() 호출과 enable_grad() 범위를 확인하세요.");
                // 가장 흔한 실패 지점
            }

            // 가중치 계산과 가중합
            var weights = grad.mean(new long[] {2, 3}, keepdim : true); // [1, 1280, 1, 1]
            /*
            공간 방향(H, W)으로 평균하는 의미
            - grad[k, i, j] = 이 위치의 활성값이 1 늘면 클래스 점수가 얼마나 오르는가>?
            - 그것을 위치 전체에 대해평균하면 채널 k 전체의 중요도가 나옴.
            - 이것이 Grad-CAM의 alpha_k에 해당.
            
            */

            var cam = (weights * featureMap).sum(new long[] {1}, keepdim : true); // [1, 1, 7, 7]
            cam = torch.nn.functional.relu(cam);
            /*ReLU를 거는 이유
            - 음수 기여는 "이 클래스가 아니라는 근거"임.
            - 보고자 하는 것은 "이 클래스라는 근거"이므로 음수를 버리기 위해 활성함수 ReLU를 걸어줌.
            - ReLU를 빼면 히트맵이 전체적으로 뿌옇게 번져 해석이 어려워짐.
            
            */

            // 업샘플링과 0~1 정규화
            cam = torch.nn.functional.interpolate(cam,
                        size : new long[] {_imageSize, _imageSize},
                        mode : InterpolationMode.Bilinear,
                        align_corners : false); // [1, 1, 300, 300]
             /*
            7x7 -> 300*300으로 늘린다
                - 정보량은 여전히 7x7(= 49칸)이다. 늘려도 정밀도가 생기지는 않는다.
                - 한 칸이 원본 32x32 픽셀에 해당한다. 바운딩박스 경계가 뭉툭해지는 이유다.
                - Bilinear 보간으로 부드럽게 만들어 사람이 보기 좋게 하는 것이 목적이다.
                - 더 정밀하게 보려면 입력 해상도를 올리거나(=feature map이 커진다) Grad-CAM++ / LayerCAM 같은 변형을 써야 한다.
            */

            var camMin = cam.min().ToSingle();
            var camMax = cam.max().ToSingle();
            var denom = Math.Max(camMax - camMin, 1e-8f); // 0으로 나누는 것 방지. ReLU 때문에 전부 0이 되는 경우(모든 기여가 음수가 되는 경우)가 있음

            cam = (cam - camMin) / denom;

            // Tensor -> float[,]으로 변환
            var flat = cam.reshape(_imageSize * _imageSize).data<float>().ToArray();
            var heat = new float[_imageSize, _imageSize];

            for(int y = 0; y < _imageSize; y++)
            {
                for(int x = 0; x < _imageSize; x++)
                {
                    heat[y, x] = flat[y * _imageSize + 1];
                }
            }
            result.Heatmap = heat;

            sw.Stop();
            result.ElapsedMs = sw.ElapsedMilliseconds;

            return result;
            
        }

        private Tensor LoadImageAsTensor(string path) //  CastingDataset과 동일한 전처리 메서드
        {
            using var bitmap = SKBitmap.Decode(path);
            if (bitmap == null) throw new InvalidDataException($"이미지 디코딩 실패 : {path}");

            using var resized = bitmap.Resize(
                new SKImageInfo(_imageSize, _imageSize),
                new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.None));

            var data = new float[3 * _imageSize * _imageSize];
            int plane = _imageSize * _imageSize;

            for (int y = 0; y < _imageSize; y++)
            {
                for (int x = 0; x < _imageSize; x++)
                {
                    var c = resized.GetPixel(x, y);
                    int i = y * _imageSize + x;

                    data[0 * plane + i] = (c.Red / 255.0f - Mean[0]) / Std[0];
                    data[1 * plane + i] = (c.Green / 255.0f - Mean[1]) / Std[1];
                    data[2 * plane + i] = (c.Blue / 255.0f - Mean[2]) / Std[2];
                }
            }

            return torch.tensor(data, new long[] { 3, _imageSize, _imageSize });
            /*
            CastingDataset과 코드가 중복
                - 학습용 Dataset은 Dataset 상속 구조에 묶여 있어 그대로 재사용하기 번거롭다.
                - 다만 중복은 위험. 한쪽만 고치면 학습과 추론의 전처리가 어긋나고, 그것은 예외 없이 정확도만 떨어뜨림
                - 실무라면 전처리를 static 유틸로 빼내 양쪽이 같은 함수를 쓰게 해야함
            */
        }

        public static SKBitmap Render(string imagePath, float[,] heatmap,
            RenderMode mode = RenderMode.Overlay, float alpha = 0.5f, int outSize = 448, bool colorBar = false)
        {
            switch (mode)
            {
                case RenderMode.HeatmapOnly:
                    return RenderHeatmapOnly(heatmap, outSize, colorBar);

                case RenderMode.SideBySide:
                    {
                        using var left = RenderPlain(imagePath, outSize);
                        using var right = RenderHeatmapOnly(heatmap, outSize, colorBar);
                        return Concat(left, right);
                    }

                default:
                    return RenderOverlay(imagePath, heatmap, alpha, outSize, colorBar);
            }
        }

        private static SKBitmap RenderHeatmapOnly(float[,] heatmap, int outSize, bool colorBar)
        // 히트맵만 그린다. 값 0이 진한 남색으로 꽉 찬다 (matplotlib imshow와 같은 모양).
        {
            int hSize = heatmap.GetLength(0);
            int barW = colorBar ? 64 : 0;
            var output = new SKBitmap(outSize + barW, outSize);

            for (int y = 0; y < outSize; y++)
            {
                for (int x = 0; x < outSize; x++)
                {
                    int hy = Math.Min(hSize - 1, y * hSize / outSize);
                    int hx = Math.Min(hSize - 1, x * hSize / outSize);
                    var (r, g, b) = JetColor(heatmap[hy, hx]);
                    output.SetPixel(x, y, new SKColor(r, g, b));
                }
            }
            /*
            오버레이와 결정적으로 다른 점
                - 원본 픽셀을 전혀 섞지 않는다. 히트맵 값만으로 색을 정한다.
                - 그래서 기여도 0인 영역이 '원본이 비치는 곳'이 아니라 '남색'이 된다.
                - 기여도 분포 자체를 보고 싶을 때 이쪽이 읽기 쉽다.
                  단, 결함이 실제로 어디 있는지와 대조하려면 원본이 함께 있어야 한다.
                  -> SideBySide 모드가 그 용도다.
            */

            if (colorBar) DrawColorBar(output, outSize, barW);
            return output;
        }

        private static SKBitmap RenderPlain(string imagePath, int outSize) // 원본을 지정 크기로만 변환
        {
            using var src = SKBitmap.Decode(imagePath);
            return src.Resize(new SKImageInfo(outSize, outSize),
                new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.None));
        }

        private static SKBitmap RenderOverlay(string imagePath, float[,] heatmap,
            float alpha, int outSize, bool colorBar)
        {
            using var src = SKBitmap.Decode(imagePath);
            using var baseImg = src.Resize(new SKImageInfo(outSize, outSize),
                new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.None));

            int hSize = heatmap.GetLength(0);
            int barW = colorBar ? 64 : 0;
            var output = new SKBitmap(outSize + barW, outSize);

            for (int y = 0; y < outSize; y++)
            {
                for (int x = 0; x < outSize; x++)
                {
                    var bc = baseImg.GetPixel(x, y);

                    int hy = Math.Min(hSize - 1, y * hSize / outSize);
                    int hx = Math.Min(hSize - 1, x * hSize / outSize);
                    float v = heatmap[hy, hx];

                    var (r, g, b) = JetColor(v);

                    float a = alpha * v;
                    /*
                    alpha에 v를 곱하는 이유
                        - 고정 alpha면 기여도 0인 영역까지 남색으로 덮여 원본이 안 보인다.
                        - v를 곱하면 기여도가 높은 곳만 강조되고 나머지는 원본이 남는다.
                        - 히트맵 전체를 덮는 모양을 원하면 HeatmapOnly 모드를 쓸 것.
                    */

                    output.SetPixel(x, y, new SKColor(
                        (byte)(bc.Red * (1 - a) + r * a),
                        (byte)(bc.Green * (1 - a) + g * a),
                        (byte)(bc.Blue * (1 - a) + b * a)));
                }
            }

            if (colorBar) DrawColorBar(output, outSize, barW);
            return output;
        }

        private static SKBitmap Concat(SKBitmap left, SKBitmap right) // 두 비트맵을 가로로 잇는다
        {
            int gap = 8;
            var output = new SKBitmap(left.Width + gap + right.Width, Math.Max(left.Height, right.Height));
            using var canvas = new SKCanvas(output);
            canvas.Clear(new SKColor(30, 30, 30));
            canvas.DrawBitmap(left, 0, 0);
            canvas.DrawBitmap(right, left.Width + gap, 0);
            return output;
        }

        private static void DrawColorBar(SKBitmap target, int xOffset, int barW) // 오른쪽에 범례를 그린다
        {
            using var canvas = new SKCanvas(target);
            //    Clear는 캔버스 '전체'를 덮으므로 이미 그려둔 히트맵이 지워진다.
            //    배경이 필요한 것은 컬러바 영역뿐이므로 아래에서 그 영역만 사각형으로 칠한다.

            int pad = 10;
            int barX = xOffset + pad;
            int barWidth = 20;
            int barTop = pad;
            int barBottom = target.Height - pad - 14;

            // 배경
            using (var bg = new SKPaint { Color = new SKColor(255, 255, 255) })
                canvas.DrawRect(xOffset, 0, barW, target.Height, bg);

            // 그라데이션 (위가 1.0, 아래가 0.0)
            for (int y = barTop; y < barBottom; y++)
            {
                float v = 1f - (y - barTop) / (float)(barBottom - barTop);
                var (r, g, b) = JetColor(v);
                using var p = new SKPaint { Color = new SKColor(r, g, b) };
                canvas.DrawRect(barX, y, barWidth, 1, p);
            }

            // 눈금 레이블
            using var text = new SKPaint { Color = SKColors.Black, IsAntialias = true };
            using var font = new SKFont(SKTypeface.Default, 11);

            foreach (float tick in new[] { 0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f })
            {
                float ty = barBottom - tick * (barBottom - barTop);
                canvas.DrawText($"{tick:F1}", barX + barWidth + 4, ty + 4, SKTextAlign.Left, font, text);
            }
            /*
            컬러바가 중요한 이유
                - 히트맵 색만 보면 "빨간 곳이 중요하다"는 것은 알아도 그 값이 얼마인지 모른다.
                - 바운딩박스 임계값(예: 0.5)이 색으로 어디쯤인지 범례가 있어야 판단된다.
                - 다만 이 값은 '0~1로 정규화된 상대값'이다. 이미지마다 최대값이 1이 되도록
                  스케일되므로, 서로 다른 이미지의 빨간색끼리 비교하면 안 된다. ★흔한 오해
            */
        }

        private static (byte r, byte g, byte b) JetColor(float v) // matplotlib 'jet' 컬러맵 재현
        {
            v = Math.Clamp(v, 0f, 1f);

            // matplotlib jet의 실제 구간별 정의
            float r = Clamp01(v < 0.35f ? 0f
                            : v < 0.66f ? (v - 0.35f) / 0.31f
                            : v < 0.89f ? 1f
                            : 1f - (v - 0.89f) / 0.11f * 0.5f);

            float g = Clamp01(v < 0.125f ? 0f
                            : v < 0.375f ? (v - 0.125f) / 0.25f
                            : v < 0.64f ? 1f
                            : v < 0.91f ? 1f - (v - 0.64f) / 0.27f
                            : 0f);

            float b = Clamp01(v < 0.11f ? 0.5f + v / 0.11f * 0.5f
                            : v < 0.34f ? 1f
                            : v < 0.65f ? 1f - (v - 0.34f) / 0.31f
                            : 0f);

            return ((byte)(r * 255), (byte)(g * 255), (byte)(b * 255));
            /*
            이전 구현(삼각파 근사)을 버린 이유
                - 값 0에서 파랑이 0.5가 아니라 1.0으로 나와, matplotlib의 '진한 남색(0,0,0.5)'과 달랐다.
                - 값 1에서도 빨강이 1.0으로 끝나 'dark red(0.5,0,0)'로 어두워지는 부분이 없었다.
                - 위 구간 정의는 matplotlib의 _jet_data를 선형 보간한 것이다.
                  0.0 -> (0,0,0.5) 남색 / 0.25 -> (0,1,1) 청록 / 0.5 -> (0.5,1,0.5)
                  0.75 -> (1,1,0) 노랑 / 1.0 -> (0.5,0,0) 진한 빨강

            jet의 한계는 알고 쓸 것
                - 지각적으로 균일하지 않아 청록 구간에서 값 차이가 과장된다.
                - 색각 이상이 있으면 구분이 어렵다.
                - 정량 비교가 목적이면 viridis가 낫다. 여기서는 Grad-CAM 논문의 관례를 따랐다.
            */
        }

        private static float Clamp01(float x) => Math.Clamp(x, 0f, 1f);

        public static (int x, int y, int w, int h)? FindBoundingBox(float[,] heatmap, float threshold = 0.5f) // 히트맵에서 임계값 이상인 영역을 감싸는 바운딩박스를 찾는 메서드
        {
            int size = heatmap.GetLength(0);
            int minX = size, minY = size, maxX = -1, maxY = -1;

            for(int y = 0; y < size; y++)
            {
                for(int x = 0; x < size; x++)
                {
                    if(heatmap[y, x] >= threshold)
                    {
                        if(x < minX)
                        {
                            minX = x;
                        }
                        if(y < minY)
                        {
                            minY = y;
                        }
                        if(x > maxX)
                        {
                            maxX = x;
                        }
                        if(y > maxY)
                        {
                            maxY = y;
                        }
                    }
                }
            }

            if(maxX < 0)// 임계값을 넘는 영역이 없다면 null 리턴
            {
                return null;
            }

            return (minX, minY, maxX - minY+1, maxY - minY + 1);
             /*
            이 방식의 한계
                - 임계값 이상인 모든 픽셀을 하나의 박스로 감싼다. 결함이 두 곳에 떨어져 있으면 그 사이 빈 공간까지 포함하는 큰 박스가 된다.
                - 분리된 영역을 각각 잡으려면 연결요소 분석(connected components)이 필요하다.
                - 또한 이 박스는 '결함의 위치'가 아니라 '모델이 주목한 위치'다. 세그멘테이션 모델의 마스크와 달리 정답 라벨로 학습된 것이 아니다. 따라서 UI 문구를 "결함 영역"이 아니라 "주목 영역"으로 쓰는 편이 정확하다.
            */
        }

        public enum RenderMode
        {
            Overlay,     // 원본 위에 히트맵을 반투명 합성 (기본)
            HeatmapOnly, // 히트맵만 단독 표시 (matplotlib imshow 스타일)
            SideBySide   // 왼쪽 원본 + 오른쪽 히트맵 단독
        }
    }
}