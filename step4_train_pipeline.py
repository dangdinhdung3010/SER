import sys

# Đảm bảo mã hóa UTF-8 cho console Windows tránh UnicodeEncodeError khi in tiếng Việt
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

# Import các bước trước
from step1_data_collector import FreeSERDataCrawler, EMOTION_TO_IDX
from step2_edge_preprocessing import create_edge_dataloaders
from step3_1d_swin_model import DINOv3AudioSER


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Huấn luyện mô hình qua 1 Epoch."""
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    for s_batch, label_indices, z_batch in dataloader:
        s_batch = s_batch.to(device)
        z_batch = z_batch.to(device)
        label_indices = label_indices.to(device)

        optimizer.zero_grad()
        logits, probs = model(s_batch)

        # Tính hàm mất mát Cross-Entropy Loss giữa xác suất dự đoán và nhãn thực z
        loss = criterion(logits, label_indices)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * s_batch.size(0)
        _, preds = torch.max(probs, dim=1)
        correct_preds += (preds == label_indices).sum().item()
        total_samples += s_batch.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds / total_samples
    return epoch_loss, epoch_acc


def evaluate_model(model, dataloader, criterion, device):
    """Đánh giá mô hình trên tập kiểm thử (Test / Validation set)."""
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():
        for s_batch, label_indices, z_batch in dataloader:
            s_batch = s_batch.to(device)
            label_indices = label_indices.to(device)

            logits, probs = model(s_batch)
            loss = criterion(logits, label_indices)

            running_loss += loss.item() * s_batch.size(0)
            _, preds = torch.max(probs, dim=1)
            correct_preds += (preds == label_indices).sum().item()
            total_samples += s_batch.size(0)

    val_loss = running_loss / total_samples
    val_acc = correct_preds / total_samples
    return val_loss, val_acc


if __name__ == "__main__":
    print("=== BẮT ĐẦU BƯỚC 4: HUẤN LUYỆN MÔ HÌNH TRÊN GPU/CPU ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Thiết bị thực thi: {device}")

    # 1. Chuẩn bị thư mục lưu trọng số checkpoint trên EMO-DB Server
    checkpoint_dir = Path("./emo_db_storage/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "best_dinov3_ser.pth"

    # 2. Lấy dữ liệu RAVDESS từ Bước 1 và tạo DataLoader từ Bước 2 (batch_size = 64)
    crawler = FreeSERDataCrawler(root_dir="./data/RAVDESS")
    samples = crawler.scan_and_filter_samples()
    loso_folds = crawler.generate_loso_splits(samples)

    # Chọn thử Fold đầu tiên theo giao thức LOSO (Test trên Actor_01)
    first_spk = list(loso_folds.keys())[0]
    train_samples, test_samples = loso_folds[first_spk]
    train_loader, test_loader = create_edge_dataloaders(
        train_samples, test_samples, batch_size=64, num_classes=6
    )

    # 3. Khởi tạo mô hình DINOv3AudioSER từ Bước 3
    model = DINOv3AudioSER(num_classes=6).to(device)

    # 4. Thiết lập hàm mất mát (CrossEntropy) và bộ tối ưu Adam
    criterion = nn.CrossEntropyLoss()
    # Chỉ tối ưu hóa các tham số mở khóa (requires_grad = True)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-4,
    )

    # 5. Vòng lặp huấn luyện qua các epochs
    num_epochs = 5  # Bạn có thể tăng lên 50-100 epochs khi train chính thức
    best_val_acc = 0.0

    print(f"\n[TRAINING] Bắt đầu huấn luyện Fold Test: '{first_spk}'")
    print(f"           Số mẫu Train: {len(train_samples)} | Số mẫu Test: {len(test_samples)}")
    print("-" * 65)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)

        print(
            f"Epoch [{epoch+1:02d}/{num_epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:6.2f}% | "
            f"Test Loss: {val_loss:.4f} - Test Acc: {val_acc*100:6.2f}%"
        )

        # Lưu lại checkpoint có độ chính xác Test tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> [SAVED] Đã lưu mô hình tốt nhất với Test Acc: {val_acc*100:.2f}% tại {best_model_path}")

    print("\n[STEP 4 OK] Hoàn thành quy trình huấn luyện và lưu checkpoint thành công!")