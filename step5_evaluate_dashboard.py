import sys

# Đảm bảo mã hóa UTF-8 cho console Windows tránh UnicodeEncodeError khi in tiếng Việt
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)

# Import các module từ Bước 1, Bước 2 và Bước 3
from step1_data_collector import FreeSERDataCrawler, EMOTION_TO_IDX
from step2_edge_preprocessing import create_edge_dataloaders, EdgeAudioPreprocessor
from step3_1d_swin_model import DINOv3AudioSER

# Danh sách tên 6 nhãn cảm xúc theo đúng thứ tự chỉ số 0..5
IDX_TO_EMOTION = {v: k for k, v in EMOTION_TO_IDX.items()}
CLASS_NAMES = [IDX_TO_EMOTION[i] for i in range(len(EMOTION_TO_IDX))]


def run_inference_on_testset(model, test_loader, device):
    """
    Thực hiện suy luận (Inference) trên toàn bộ tập Test,
    thu thập nhãn thực, nhãn dự đoán và vector xác suất Softmax.
    """
    model.eval()
    all_Labels = []
    all_Preds = []
    all_Probs = []

    with torch.no_grad():
        for s_batch, label_indices, _ in test_loader:
            s_batch = s_batch.to(device)
            _, probs = model(s_batch)
            preds = torch.argmax(probs, dim=1)

            all_Labels.extend(label_indices.numpy())
            all_Preds.extend(preds.cpu().numpy())
            all_Probs.extend(probs.cpu().numpy())

    return np.array(all_Labels), np.array(all_Preds), np.array(all_Probs)


def compute_and_save_metrics(y_true, y_pred, y_prob, output_dir: Path):
    """
    Tính toán WAR, UAR (theo bài báo SER) cùng Precision, Recall, F1, Ave. AUC (theo bài DuST)
    và lưu báo cáo chi tiết ra file text trên Dashboard.
    """
    # 1. Chỉ số WAR (tương đương Accuracy toàn cục) và UAR (Macro Recall)
    war = accuracy_score(y_true, y_pred)
    uar = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # 2. Chỉ số Precision, Recall, F1-score cho từng lớp cảm xúc
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )

    # 3. Chỉ số Average AUC (One-vs-Rest)
    try:
        ave_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        ave_auc = 0.0

    # Định dạng nội dung báo cáo
    report_lines = [
        "=" * 70,
        "   BÁO CÁO ĐÁNH GIÁ HỆ THỐNG NHẬN DIỆN CẢM XÚC GIỌNG NÓI (SER DASHBOARD)",
        "=" * 70,
        f"1. Chỉ số Tổng hợp (Overall Metrics):",
        f"   - WAR (Weighted Average Recall / Accuracy) : {war * 100:6.2f}%",
        f"   - UAR (Unweighted Average Recall)          : {uar * 100:6.2f}%",
        f"   - Average AUC (Macro ROC-AUC)              : {ave_auc:6.4f}",
        "-" * 70,
        "2. Chỉ số Chi tiết theo từng Cảm xúc (Per-Class Metrics):",
        f"   {'Emotion':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Samples':<8}",
        "   " + "-" * 58,
    ]

    for i, emo_name in enumerate(CLASS_NAMES):
        report_lines.append(
            f"   {emo_name:<12} | {precisions[i]:<10.4f} | {recalls[i]:<10.4f} | {f1s[i]:<10.4f} | {supports[i]:<8d}"
        )
    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_path = output_dir / "classification_metrics_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"[SAVED] Đã lưu báo cáo số liệu tại: {report_path}")


def plot_confusion_matrix_dashboard(y_true, y_pred, output_dir: Path):
    """
    Vẽ Ma trận nhầm lẫn chuẩn hóa theo hàng (Normalized Confusion Matrix)
    giống định dạng Hình 2 trong bài báo Speech Swin-Transformer.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    cm_normalized = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    im = ax.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Greys)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        title="Normalized Confusion Matrix (SER - DINOv3)",
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Hiển thị giá trị tỷ lệ trong từng ô
    thresh = cm_normalized.max() / 2.0
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_normalized[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm_normalized[i, j] > thresh else "black",
                fontsize=10,
            )

    fig.tight_layout()
    cm_path = output_dir / "confusion_matrix_ravdess.png"
    plt.savefig(cm_path)
    plt.close(fig)
    print(f"[SAVED] Đã xuất ảnh Ma trận nhầm lẫn tại: {cm_path}")


def predict_single_audio_dashboard(model, sample_record: dict, device, output_dir: Path):
    """
    Mô phỏng Quy trình Suy luận (Inference Process) từ Lớp Biên lên Lớp Ứng dụng:
    Nhận 1 file .wav -> Tiền xử lý Log-Mel (128x32) -> DINOv3 dự đoán -> Vẽ Dashboard xác suất.
    """
    preprocessor = EdgeAudioPreprocessor()
    waveform = preprocessor.load_audio(sample_record["audio_path"])
    s_1d = preprocessor(waveform, is_train=False)  # (128, 32)

    model.eval()
    with torch.no_grad():
        s_input = s_1d.unsqueeze(0).to(device)  # (1, 128, 32)
        _, probs = model(s_input)
        probs_np = probs.squeeze(0).cpu().numpy()

    pred_idx = int(np.argmax(probs_np))
    pred_emotion = IDX_TO_EMOTION[pred_idx]
    true_emotion = sample_record["emotion_str"]

    # Vẽ Dashboard gồm 2 khung: Phổ Log-Mel-Spectrogram và Biểu đồ xác suất 6 cảm xúc
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    # Khung 1: Log-Mel-Spectrogram tại Lớp Biên (f=32, d=128)
    spec_2d = s_1d.transpose(0, 1).numpy()
    im1 = ax1.imshow(spec_2d, aspect="auto", origin="lower", cmap="viridis")
    ax1.set_title(f"Edge Layer: Log-Mel-Spectrogram (32x128)\nFile: {Path(sample_record['audio_path']).name}")
    ax1.set_xlabel("Time domain (128 frames)")
    ax1.set_ylabel("Frequency domain (32 Mel-bands)")
    fig.colorbar(im1, ax=ax1)

    # Khung 2: Phân bố xác suất dự đoán tại Lớp Ứng dụng
    bars = ax2.barh(CLASS_NAMES, probs_np * 100, color="#4C72B0")
    bars[pred_idx].set_color("#DD8452")
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Confidence Probability (%)")
    ax2.set_title(f"Application Layer: Emotion Prediction\nTrue: [{true_emotion.upper()}] | Predicted: [{pred_emotion.upper()}] ({probs_np[pred_idx]*100:.1f}%)")

    for bar, p in zip(bars, probs_np):
        ax2.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2, f"{p*100:.1f}%", va="center")

    fig.tight_layout()
    demo_path = output_dir / "single_audio_inference_dashboard.png"
    plt.savefig(demo_path)
    plt.close(fig)
    print(f"[SAVED] Đã xuất biểu đồ Dashboard suy luận mẫu đơn tại: {demo_path}")


if __name__ == "__main__":
    print("=== BẮT ĐẦU BƯỚC 5: ĐÁNH GIÁ & XUẤT KẾT QUẢ LÊN APPLICATION LAYER ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Tạo thư mục xuất kết quả Dashboard
    output_dir = Path("./dashboard_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path("./emo_db_storage/checkpoints/best_dinov3_ser.pth")

    # 2. Chuẩn bị tập dữ liệu kiểm thử từ Bước 1 & Bước 2
    crawler = FreeSERDataCrawler(root_dir="./data/RAVDESS")
    samples = crawler.scan_and_filter_samples()
    loso_folds = crawler.generate_loso_splits(samples)

    first_spk = list(loso_folds.keys())[0]
    train_samples, test_samples = loso_folds[first_spk]
    _, test_loader = create_edge_dataloaders(train_samples, test_samples, batch_size=64, num_classes=6)

    # 3. Khởi tạo mô hình và nạp trọng số đã huấn luyện từ Bước 4
    model = DINOv3AudioSER(num_classes=6).to(device)
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[INFO] Đã nạp trọng số mô hình từ: {checkpoint_path}")
    else:
        print(f"[WARN] Chưa tìm thấy file {checkpoint_path}. Đang chạy với trọng số hiện tại.")

    # 4. Đánh giá trên tập Test, tính WAR / UAR / F1 / AUC và vẽ Confusion Matrix
    y_true, y_pred, y_prob = run_inference_on_testset(model, test_loader, device)
    compute_and_save_metrics(y_true, y_pred, y_prob, output_dir)
    plot_confusion_matrix_dashboard(y_true, y_pred, output_dir)

    # 5. Chạy thử Quy trình Suy luận (Inference Process) trên 1 file .wav cụ thể cho Chuyên gia
    predict_single_audio_dashboard(model, test_samples[0], device, output_dir)

    print("\n[STEP 5 OK] Toàn bộ 5 bước của dự án SER với DINOv3 đã hoàn tất!")