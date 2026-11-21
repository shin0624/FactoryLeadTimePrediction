using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using TorchSharp;
using static TorchSharp.torch;
using SkiaSharp;
using System.Linq.Expressions;

namespace FactoryLeadTimePrediction.DeepLearning.B_TorchSharpGradCAM
{
    public class CastingDataset : utils.data.Dataset // Grad-CAM 히트맵 구현을 위한 데이터 로더. TorchSharp의 Dataset클래스를 상속한다. ML.NET의 IDataView에 대응하는 개념.
    {
        /*
        ML.NET과 TorchSharp이 무엇이 다른가?
        - ML.NET : LoadRawImageBytes 변환기가 경로 -> 바이트 -> 디코딩 -> 리사이즈 -> 정규화를 전부 수행하였고, 개발자는 매개변수에 경로만 넘기면 되었음.
        - TorchSharp : 위 변환 로직을 직접 구현해야 함. 아래 GetTensor()메서드에서 수행.
            ML.NET의 LoadRawImageBytes()에 래핑되어있던 전처리 단계를 직접 구현.
        */

        private readonly List<(string path, long label)> _items;
        private readonly int _imageSize;

        // ImageNet 정규화 상수
        private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f};
        private static readonly float[] Std = { 0.229f, 0.224f, 0.225f};
        /*
        - ImageNet 학습 데이터 전체의 채널별 평균/표준편차 상수(torchvision의 표준값)
        - 사전학습 가중치는 위와같이 정규화된 입력을 전제로 학습되었음
        - 다른 값으로 정규화 하면 입력 분포가 어긋나 백본이 엉뚱한 피처를 뽑는다. 예외가 나지 않고 정확도만 떨어져서 발견이 어려움. ml.net은 이것을 내부에서 처리해 주었었음.
        */

        public const long LabelDefect = 0; // ML.NET에서 썼던 Score[0] = "defect"와 순서를 맞춘다.
        public const long LabelOk = 1;

        public CastingDataset(string rootFolder, int imageSize = 224)
        {
            _imageSize = imageSize;//224 : MobileNetV2가 ImageNet에서 학습된 입력 크기. 원본이 300*300이므로 줄여야 함. 사전학습 가중치를 쓰려면 이 크기를 맞추는 게 유리함

            _items = new List<(string, long)> ();

            var map = new Dictionary<string, long> {{"def_front", LabelDefect}, {"ok_front", LabelOk} };

            foreach(var (folder, label)in map)
            {
                string dir = Path.Combine(rootFolder, folder);
                if(!Directory.Exists(dir))
                {
                    System.Console.WriteLine($"[경고] 폴더 없음 : {dir}");
                    continue;
                }

                var files = Directory.EnumerateFiles(dir).Where(f => f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase)).ToList();

                foreach(var f in files)
                {
                    _items.Add((Path.GetFullPath(f), label));
                }
                Console.WriteLine($"  {folder,-10} -> label {label} : {files.Count,5}장");
            }
        }

        private CastingDataset(List<(string path, long label)> items, int imageSize) // 분할용 private 생성자
        {
            _items = items;
            _imageSize = imageSize;
        }

        public (CastingDataset train, CastingDataset val) SplitTrainVal(double valFraction, int seed) // train/val 2분할
        {
            var rng = new Random(seed);
            var shuffled = _items.OrderBy(_ => rng.Next()).ToList();
            /*
            셔플이 반드시 필요한 이유  ★이것을 빼면 학습이 통째로 망가진다
                - _items는 폴더 순서대로 담겨 있다. 앞쪽 전부 불량(label 0), 뒤쪽 전부 정상(label 1).
                - 셔플 없이 앞 90%를 train으로 자르면, train에 정상이 거의 없고 val은 전부 정상이 된다.
                - 그러면 모델은 "항상 불량"이라고 답하는 것을 배우고, val 정확도는 0%가 된다.
                - ML.NET의 TrainTestSplit()은 내부에서 셔플을 해줬다. 직접 짤 때는 빼먹기 쉬운 지점이다.

            seed를 받는 이유 : 실행마다 같은 분할이 나와야 결과를 비교할 수 있다. A/B 실습과 같은 원칙.
            */

            int valCount = (int)(shuffled.Count * valFraction);
            var valItems = shuffled.Take(valCount).ToList();
            var trainItems = shuffled.Skip(valCount).ToList();

            return (new CastingDataset(trainItems, _imageSize), new CastingDataset(valItems, _imageSize));
        }

        public override long Count => _items.Count; // Dataset의 추상 멤버. DataLoader가 이 값으로 배치 개수를 계산한다.

        public override Dictionary<string, Tensor> GetTensor(long index) // 인덱스 1개에 해당하는 샘플을 Tensor로 만든다.
        {
            var (path, label) = _items[(int)index];

            return new Dictionary<string, Tensor>
            {
                {"data", LoadImageAsTensor(path)},
                {"label", torch.tensor(label, ScalarType.Int64)}
                // CrossEntropyLoss가 반드시 int64(long)를 요구하기에, 라벨은 반드시 int64로 해야 함. int32를 넣으면 런타임에 오류 발생.    
            };
            /*
            Dictionary로 반환하는 것이 TorchSharp Dataset의 규칙으로, 여기서 쓴 키 "data/label"은 학습 루프에서 꺼낼 떄 그대로 써야 함
            */
        }

        private Tensor LoadImageAsTensor(string path)// 이미지 파일 하나를 [3, H, W] 정규화된 Tensor로 변환하는 메서드.
        {
            using var bitmap = SKBitmap.Decode(path); // SkiaSharp으로 디코딩. TorchSharp이 의존성으로 이미 이를 가져오므로, 추가 패키지는 불필요

            using var resized = bitmap.Resize(new SKImageInfo(_imageSize, _imageSize), new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.None)); // 300*300을 224*224로 리사이징. 

            // 픽셀 -> float 배열로 변환
            var data = new float[3*_imageSize * _imageSize];
            int plane = _imageSize * _imageSize;

            for(int y = 0; y<_imageSize; y++)
            {
                for(int x=0; x < _imageSize; x++)
                {
                    var c = resized.GetPixel(x,y);
                    int i = y*_imageSize + x;

                    data[0 * plane + i] = (c.Red / 255.0f - Mean[0]) / Std[0];
                    data[1 * plane + i] = (c.Green / 255.0f - Mean[1]) / Std[1];
                    data[2 * plane + i] = (c.Blue / 255.0f - Mean[2]) / Std[2];
                    /*
                    ★ 여기는 반드시 0*plane / 1*plane / 2*plane 이다 (+ 가 아니라 *).
                       '+' 로 쓰면 인덱스가 plane+i, plane+1+i, plane+2+i 가 되어
                         - 0번 채널(R) 영역 [0, plane) 이 전부 0으로 남고
                         - 1번 채널 영역에 R/G/B 값이 1칸씩 밀려 겹쳐 쓰인다
                       즉 R 채널이 통째로 비고 G/B가 뒤섞인 입력이 모델에 들어간다.
                       예외가 나지 않고 정확도만 떨어지므로 발견이 매우 어렵다.
                       (실제로 이 오타 상태에서 test 97.62% / FN 12건이 나왔다)
                    */
                    /*
                    0~255 정수를 0~1 실수로 변환 (/255.0f) -> ImageNet의 평균/표준편차로 정규화 ((x-mean) / std)
                    배치 레이아웃이 [C, H, W]인 이유?
                        - Pytorch/TorchSharp은 [채널, 높이, 너비] 순서.
                        - 이미지 파일은 보통 픽셀마다 RGB가 붙어있는 HWC라서 plane 단위로 채널을 분리하여 담는다. 이 순서를 틀리면 색이 섞인 입력이 들어가는데, 예외는 나지 않기에 조용히 실패한다.
                    
                    */
                }
            }
            return torch.tensor(data, new long[] {3, _imageSize, _imageSize}); // shape를 명시하여 [3, 224, 224] 텐서로 만든다. 배치 차원은 DataLoader가 붙여준다.
        }

        public (int defect, int ok)CountByLabel() // 불량, 정상 라벨 수 검산용 메서드
        {
            return (_items.Count(i => i.label == LabelDefect), _items.Count(i=>i.label ==LabelOk));
        }

        public string GetPath(long index) => _items[(int)index].path; // Grad-CAM에서 원본 이미지를 다시 읽기 위해 필요.
    }
}