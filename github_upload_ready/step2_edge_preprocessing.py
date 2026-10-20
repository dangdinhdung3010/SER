import sys

# Đảm bảo mã hóa UTF-8 cho console Windows tránh UnicodeEncodeError khi in tiếng Việt
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from typing import List, Dict, Tuple
import numpy as np
from scipy.io import wavfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.functional as AF

from step1_data_collector import FreeSERDataCrawler


class EdgeAudioPreprocessor(nn.Module):
    """Biến đổi sóng âm 16kHz -> Log-Mel-Spectrogram -> Chuỗi 1D S (T=128, C=32)."""

    def __init__(self, sample_rate: int = 16000, n_fft: int = 512, n_mels: int = 32, target_frames: int = 128):
        super().__init__()
        self.sample_rate = sample_rate
        self.target_frames = target_frames
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=int(sample_rate * 0.02),  # 320 samples (20 ms)
            hop_length=int(sample_rate * 0.01),  # 160 samples (10 ms)
            window_fn=torch.hamming_window,
            n_mels=n_mels,
            power=2.0,
        )

    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Đọc file .wav, chuyển về 1 kênh (c=1) và chuẩn hóa tần số 16.000 Hz."""
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception:
            sr, data = wavfile.read(audio_path)
            data = data.astype(np.float32) / (32768.0 if data.dtype == np.int16 else 1.0)
            waveform = torch.from_numpy(data).unsqueeze(0) if data.ndim == 1 else torch.from_numpy(data.T)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = AF.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)
        return waveform

    def forward(self, waveform: torch.Tensor, is_train: bool = True) -> torch.Tensor:
        # 1. Tiền nhấn (Pre-emphasis 0.97) & Log-Mel-Spectrogram
        emphasized = AF.preemphasis(waveform, coeff=0.97)
        mel = self.mel_spectrogram(emphasized)
        log_mel = torch.log(torch.clamp(mel, min=1e-10)).squeeze(0)  # (32, d_raw)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

        # 2. Cắt phân đoạn (Slicing) cố định d = 128 frames
        total_frames = log_mel.shape[-1]
        if total_frames > self.target_frames:
            start = (
                torch.randint(0, total_frames - self.target_frames + 1, (1,)).item()
                if is_train
                else (total_frames - self.target_frames) // 2
            )
            log_mel = log_mel[:, start : start + self.target_frames]
        elif total_frames < self.target_frames:
            log_mel = F.pad(log_mel, (0, self.target_frames - total_frames))

        # 3. Chuyển vị sang Chuỗi thời gian 1D S: (T=128, C=32)
        return log_mel.transpose(0, 1).contiguous()


class SEREdgeDataset(Dataset):
    """Dataset kết nối danh sách file .wav của Bước 1 với bộ tiền xử lý Bước 2."""

    def __init__(self, samples: List[Dict], num_classes: int = 6, is_train: bool = True):
        self.samples = samples
        self.num_classes = num_classes
        self.is_train = is_train
        self.preprocessor = EdgeAudioPreprocessor()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        record = self.samples[idx]
        waveform = self.preprocessor.load_audio(record["audio_path"])
        s_1d = self.preprocessor(waveform, is_train=self.is_train)
        label_idx = int(record["label_idx"])
        z_onehot = F.one_hot(torch.tensor(label_idx), num_classes=self.num_classes).float()
        return s_1d, label_idx, z_onehot


def create_edge_dataloaders(
    train_samples: List[Dict], test_samples: List[Dict], batch_size: int = 64, num_classes: int = 6
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(SEREdgeDataset(train_samples, num_classes, is_train=True), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(SEREdgeDataset(test_samples, num_classes, is_train=False), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    crawler = FreeSERDataCrawler(root_dir="./data/RAVDESS")
    samples = crawler.scan_and_filter_samples()
    loso_folds = crawler.generate_loso_splits(samples)

    first_spk = list(loso_folds.keys())[0]
    train_s, test_s = loso_folds[first_spk]
    train_loader, _ = create_edge_dataloaders(train_s, test_s, batch_size=64)

    s_batch, y_idx, z_batch = next(iter(train_loader))
    print(f"[STEP 2 OK] Fold '{first_spk}' | S shape: {s_batch.shape} | z shape: {z_batch.shape}")