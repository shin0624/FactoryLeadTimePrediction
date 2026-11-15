using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Tensorflow;

namespace FactoryLeadTimePrediction.DeepLearning.A_CastingDefectClassification
{
    public class CastingImageLoader // 데이터셋 폴더를 순회하며 List<CastingImageData>로 만들고, 검산 수치를 출력하는 클래스
    {
        // 라벨 문자열을 상수로 뽑아서 혼동행렬 계싼, 임계값 튜닝, 출력 메시지 등 여러 곳에서 이 문자열만 비교하게 해야 함.
        public const string LabelDefect = "defect" ; // 불량 (def_front 폴더)
        public const string LabelOk = "ok"; // 정상(ok_front 폴더)

        private static readonly string[] ImageExtensions = {".jpeg", ".jpg", ".png"}; // 확장자가 섞여도 받을 수 있도록 함

        public List<CastingImageData> LoadFromFolder(string rootFolder)// 지정 폴더 아래의 def_front, ok_front를 읽어 목록을 만드는 메서드
        {
            var list = new List<CastingImageData>();

            if (!Directory.Exists(rootFolder))// 경로 오류 처리
            {
                throw new DirectoryNotFoundException($"데이터셋 폴더를 찾을 수 없습니다. : {rootFolder}");
            }

            var folderToLabel = new Dictionary<string, string>
            {
                {"def_front", LabelDefect},
                {"ok_front", LabelOk}
            }; // 폴더명을 라벨로 매핑하여 저장. (폴더명을 기반으로 하면  추후 클래스를 추가할 때 Dictionary 한 줄만 늘리면 됨)

            foreach(var (folderName, label) in folderToLabel)
            {
                string classFolder = Path.Combine(rootFolder, folderName); // 경로에 한글이나 공백이 있으면문자열 연결(+ "\\")은 실수가 나기 쉬우므로, Path.Combine으로 연결

                if(!Directory.Exists(classFolder))
                {
                    System.Console.WriteLine($"[경고] 클래스 폴더가 없습니다. : {classFolder}");
                    continue;
                }

                var files = Directory.EnumerateFiles(classFolder).Where(f => ImageExtensions.Contains(Path.GetExtension(f).ToLowerInvariant())).ToList();
                // EnumerateFiles : GetFiles()와 달리, 지연 열거 방식으로 작동하므로, 대량 파일에서 메모리 사용량이 적음
                // ToLoweInvarient : 확장자 대소문자를 커버할 수 있음.

                foreach(var file in files)
                {
                    list.add(new CastingImageData
                    {
                       ImagePath = Path.GetFullPath(file), // 절대경로(FullPath) 반환. 상대경로이면 실행위치가 바뀔 때 깨짐.
                       Label = label 
                    });
                }
                System.Console.WriteLine($" {folderName, -10} -> {label, -7} : {files.Count, 5}장");
            }
            return list;
            
        }

        public void PrintSummary(string title, List<CastingImageData> data) // 클래스별 개수와 불균형 비율을 출력하는 검산 메서드
        {
            System.Console.WriteLine();
            System.Console.WriteLine($"----------{title}----------");

            if(data.Count == 0)
            {
                System.Console.WriteLine(   "[오류] 이미지가 0장입니다. 경로를 확인하세요.");
                return;
            }

            int defect = data.Count(d => d.Label == LabelDefect); // 불량인 이미지
            int ok = data.Count(d => d.Label == LabelOk); // 정상 이미지

            Console.WriteLine($"  총 {data.Count}장  (불량 {defect} / 정상 {ok})");
            Console.WriteLine($"  클래스 비율 : 불량 {defect * 100.0 / data.Count:F1}% / 정상 {ok * 100.0 / data.Count:F1}%");
            // 이번 학습용 데이터 중 Test 데이터셋의 불량 데이터 비중은 63.4%이므로, 아무것도 학습하지 않고 전부 불량이라고 답해도 Accuracy = 63.4%가 나온다.
            // 즉, 이번 학습 모델의 베이스라인은 63.4%

            int majority = Math.Max(defect, ok);
            string majorityLabel = defect >= ok ? LabelDefect : LabelOk;

            Console.WriteLine($"  [베이스라인] 전부 \"{majorityLabel}\"이라고 답할 때 Accuracy = {majority * 100.0 / data.Count:F2}%");
            Console.WriteLine($"               <- 모델이 반드시 이겨야 하는 기준선");
        }

    }
}