import sys

# Đảm bảo mã hóa UTF-8 cho console Windows tránh UnicodeEncodeError khi in tiếng Việt
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from step1_data_collector import FreeSERDataCrawler
from step2_edge_preprocessing import create_edge_dataloaders


class DINOv3AudioSER(nn.Module):
    """
    [Lớp Đám mây & AI - Cloud & AI Layer]
    Mô hình Nhận diện Cảm xúc Giọng nói sử dụng Encoder DINOv3 kết hợp Classification Head (FC + Softmax).
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224.dino",
        num_classes: int = 6,
        img_size: int = 224,
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 2,
        drop_rate: float = 0.2,
    ):
        super().__init__()
        self.img_size = img_size

        # 1. Adapter chuyển Log-Mel-Spectrogram 1 kênh (c=1) sang 3 kênh học được cho DINOv3
        self.spec_adapter = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
        )

        # 2. Khởi tạo DINOv3 Encoder từ thư viện timm (có fallback tự động nếu thiếu quyền truy cập)
        try:
            self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0)
            self.loaded_model_name = model_name
        except Exception:
            fallback_name = "vit_small_patch14_dinov2.lvd142m"
            print(f"[INFO] Chuyển sang dùng encoder mở '{fallback_name}' (do '{model_name}' yêu cầu HF token hoặc bản timm mới).")
            try:
                self.encoder = timm.create_model(fallback_name, pretrained=True, num_classes=0)
                self.loaded_model_name = fallback_name
            except Exception:
                self.encoder = timm.create_model("vit_small_patch16_224.dino", pretrained=True, num_classes=0)
                self.loaded_model_name = "vit_small_patch16_224.dino"

        embed_dim = self.encoder.num_features  # 384 với ViT-Small

        # 3. Đóng băng trọng số Encoder theo chiến lược Stagewise của DuST, chỉ mở khóa 2 block cuối
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

            if hasattr(self.encoder, "blocks") and unfreeze_last_n_blocks > 0:
                for block in self.encoder.blocks[-unfreeze_last_n_blocks:]:
                    for param in block.parameters():
                        param.requires_grad = True

            if hasattr(self.encoder, "norm"):
                for param in self.encoder.norm.parameters():
                    param.requires_grad = True

        # 4. Classification Head (LN + Dropout + FC) dự đoán k = 6 nhãn cảm xúc
        self.classification_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, s_1d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input s_1d từ Bước 2: (B, T=128, C=32)
        Output:
          - logits y_hat: (B, k=6)
          - probs y_hat': (B, k=6) sau hàm Softmax
        """
        # Chuyển chuỗi (B, 128, 32) về lại phổ 2D (B, c=1, f=32, d=128)
        x_2d = s_1d.transpose(1, 2).unsqueeze(1)

        # Ánh xạ 1 kênh -> 3 kênh và nội suy lên kích thước (B, 3, 224, 224)
        x_rgb = self.spec_adapter(x_2d)
        x_resized = F.interpolate(
            x_rgb, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
        )

        # Trích xuất vector đặc trưng toàn cục qua DINOv3 Encoder: (B, embed_dim=384)
        features = self.encoder(x_resized)

        # Dự đoán xác suất cảm xúc qua FC + Softmax
        logits = self.classification_head(features)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Lấy 1 Batch dữ liệu thật từ Bước 1 & Bước 2
    crawler = FreeSERDataCrawler(root_dir="./data/RAVDESS")
    samples = crawler.scan_and_filter_samples()
    loso_folds = crawler.generate_loso_splits(samples)
    first_spk = list(loso_folds.keys())[0]
    train_s, test_s = loso_folds[first_spk]
    train_loader, _ = create_edge_dataloaders(train_s, test_s, batch_size=64)

    s_batch, y_idx, z_batch = next(iter(train_loader))
    s_batch = s_batch.to(device)

    # 2. Khởi tạo mô hình với Encoder DINOv3 và chạy kiểm thử
    model = DINOv3AudioSER(num_classes=6).to(device)
    logits, probs = model(s_batch)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"[STEP 3 OK] Encoder đang dùng : {model.loaded_model_name} trên {device}")
    print(f"[STEP 3 OK] Tổng tham số      : {total_params:,} | Tham số huấn luyện: {trainable_params:,}")
    print(f"[STEP 3 OK] Input S shape     : {s_batch.shape} -> Output Probs shape: {probs.shape}")