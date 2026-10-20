import sys

# Đảm bảo mã hóa UTF-8 cho console Windows tránh UnicodeEncodeError khi in tiếng Việt
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from torch.nn import functional as F

# Ánh xạ mã cảm xúc trong tên file RAVDESS sang 6 nhãn chuẩn (tương đương bộ CASIA: k = 6)
# Cấu trúc tên file RAVDESS: 03-01-[EMOTION]-01-01-01-[ACTOR].wav
RAVDESS_EMOTION_CODE = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "08": "surprise",
}

EMOTION_TO_IDX = {
    "angry": 0,
    "happy": 1,
    "fear": 2,
    "sad": 3,
    "neutral": 4,
    "surprise": 5,
}


class FreeSERDataCrawler:
    """
    Lớp Vật lý & Lưu trữ (Physical Layer & EMO-DB Storing):
    Tự động tải (crawl) bộ dữ liệu âm thanh cảm xúc mở RAVDESS về máy cục bộ,
    lọc 6 nhãn cảm xúc và tạo phân hoạch Leave-One-Speaker-Out (LOSO).
    """

    ZENODO_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"

    def __init__(self, root_dir: str = "./data/RAVDESS"):
        self.root_dir = Path(root_dir)
        self.zip_path = self.root_dir / "Audio_Speech_Actors_01-24.zip"
        self.num_classes = len(EMOTION_TO_IDX)

    def download_and_extract(self) -> None:
        """
        Tự động tải file .zip (~200MB) từ Zenodo và giải nén ra các thư mục Actor_01..Actor_24.
        """
        self.root_dir.mkdir(parents=True, exist_ok=True)

        # Kiểm tra nếu đã giải nén file .wav thì không cần tải lại
        existing_wavs = list(self.root_dir.rglob("*.wav"))
        if len(existing_wavs) > 0:
            print(f"[INFO] Đã tìm thấy {len(existing_wavs)} file .wav trong '{self.root_dir}'. Bỏ qua bước tải.")
            return

        if not self.zip_path.exists():
            print(f"[CRAWLING] Đang tải bộ dữ liệu âm thanh mở từ:\n  {self.ZENODO_URL}")
            print("  Vui lòng đợi 1-2 phút (dung lượng ~200MB)...")

            def _progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100.0, downloaded * 100.0 / total_size)
                    print(f"\r  Tiến độ tải: {percent:5.1f}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end="")

            urllib.request.urlretrieve(self.ZENODO_URL, self.zip_path, reporthook=_progress)
            print("\n[DONE] Tải hoàn tất!")

        print(f"[EXTRACTING] Đang giải nén dữ liệu vào '{self.root_dir}'...")
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            zf.extractall(self.root_dir)
        print("[DONE] Giải nén thành công!")

    def scan_and_filter_samples(self) -> List[Dict]:
        """
        Quét toàn bộ file .wav trên ổ cứng và lọc lấy 6 nhãn cảm xúc mục tiêu:
        angry, happy, fear, sad, neutral, surprise.
        """
        self.download_and_extract()
        wav_files = sorted(list(self.root_dir.rglob("*.wav")))
        filtered_samples = []

        for wav_path in wav_files:
            parts = wav_path.stem.split("-")
            if len(parts) != 7:
                continue

            emo_code = parts[2]
            actor_code = parts[6]

            # Chỉ giữ lại các mẫu thuộc 6 nhãn cảm xúc chuẩn
            if emo_code in RAVDESS_EMOTION_CODE:
                emo_str = RAVDESS_EMOTION_CODE[emo_code]
                filtered_samples.append({
                    "audio_path": str(wav_path),
                    "speaker_id": f"Actor_{actor_code}",
                    "emotion_str": emo_str,
                    "label_idx": EMOTION_TO_IDX[emo_str],
                })

        return filtered_samples

    def generate_loso_splits(
        self, samples: List[Dict]
    ) -> Dict[str, Tuple[List[Dict], List[Dict]]]:
        """
        Chia dữ liệu theo giao thức Leave-One-Speaker-Out (LOSO):
        1 người nói làm tập Test, các người nói còn lại làm tập Train.
        """
        speakers = sorted(list({s["speaker_id"] for s in samples}))
        loso_folds = {}

        for test_speaker in speakers:
            train_set = [s for s in samples if s["speaker_id"] != test_speaker]
            test_set = [s for s in samples if s["speaker_id"] == test_speaker]
            loso_folds[test_speaker] = (train_set, test_set)

        return loso_folds

    def encode_true_labels(self, label_indices: List[int]) -> torch.Tensor:
        """
        Mã hóa nhãn cảm xúc thực z thuộc R^(b x k) để tính CrossEntropyLoss.
        """
        labels_tensor = torch.tensor(label_indices, dtype=torch.long)
        return F.one_hot(labels_tensor, num_classes=self.num_classes).float()


if __name__ == "__main__":
    crawler = FreeSERDataCrawler(root_dir="./data/RAVDESS")

    # 1. Tự động tải dữ liệu thật về máy và lọc 6 nhãn cảm xúc
    samples = crawler.scan_and_filter_samples()
    print(f"\n[1] Tổng số file .wav hợp lệ thu thập được (6 cảm xúc): {len(samples)} mẫu")
    if samples:
        print(f"    Mẫu thực tế đầu tiên trên ổ cứng: {samples[0]}")

    # 2. Chia dữ liệu theo giao thức độc lập người nói (LOSO)
    loso_folds = crawler.generate_loso_splits(samples)
    print(f"\n[2] Tổng số Folds theo người nói (LOSO): {len(loso_folds)} folds")
    first_spk = list(loso_folds.keys())[0]
    train_s, test_s = loso_folds[first_spk]
    print(f"    Ví dụ Fold '{first_spk}': Train = {len(train_s)} file .wav | Test = {len(test_s)} file .wav")

    # 3. Kiểm tra tensor nhãn thực z thuộc R^(b x k) với b = 64, k = 6
    batch_labels = [s["label_idx"] for s in samples[:64]]
    z = crawler.encode_true_labels(batch_labels)
    print(f"\n[3] Kích thước Tensor nhãn thực z (b=64, k=6): {z.shape}")