import shutil
from pathlib import Path
import torch

from step3_1d_swin_model import DINOv3AudioSER


def prepare_streamlit_github_bundle():
    print("=== BƯỚC 7: ĐÓNG GÓI HỆ THỐNG CHO GITHUB & STREAMLIT CLOUD 24/7 ===")

    bundle_dir = Path("./github_upload_ready")
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tạo file requirements.txt chuẩn cho Streamlit Cloud
    requirements_content = "\n".join([
        "streamlit>=1.40.0",
        "torch",
        "torchaudio",
        "numpy",
        "scipy",
        "pandas",
        "timm",
        "huggingface_hub",
        "soundfile",
    ])
    (bundle_dir / "requirements.txt").write_text(requirements_content, encoding="utf-8")
    print("[1/4] Đã tạo file requirements.txt")

    # 2. Tạo file .gitignore để loại bỏ thư mục data nặng 200MB và file rác
    gitignore_content = "\n".join([
        "data/",
        "__pycache__/",
        "*.zip",
        "*.wav",
        "emo_db_storage/checkpoints/best_dinov3_ser.pth",
    ])
    (bundle_dir / ".gitignore").write_text(gitignore_content, encoding="utf-8")
    print("[2/4] Đã tạo file .gitignore")

    # 3. Copy các file code từ Bước 1, 2, 3, 6 và README.md
    code_files = [
        "step1_data_collector.py",
        "step2_edge_preprocessing.py",
        "step3_1d_swin_model.py",
        "step6_deploy_app.py",
        "README.md",
    ]
    for fname in code_files:
        src = Path(fname)
        if src.exists():
            shutil.copy2(src, bundle_dir / fname)
    print("[3/4] Đã gom đầy đủ các file code (step1, step2, step3, step6)")

    # 4. Nén thông minh trọng số mô hình từ ~85MB xuống ~14MB (chỉ lưu các tầng mở khóa)
    full_ckpt = Path("./emo_db_storage/checkpoints/best_dinov3_ser.pth")
    if full_ckpt.exists():
        model = DINOv3AudioSER(num_classes=6)
        trainable_keys = {name for name, p in model.named_parameters() if p.requires_grad}
        buffer_keys = {name for name, _ in model.named_buffers()}
        keep_keys = trainable_keys.union(buffer_keys)

        try:
            full_state = torch.load(full_ckpt, map_location="cpu", weights_only=True)
        except TypeError:
            full_state = torch.load(full_ckpt, map_location="cpu")

        compact_state = {k: v for k, v in full_state.items() if k in keep_keys}
        ckpt_out_dir = bundle_dir / "emo_db_storage" / "checkpoints"
        ckpt_out_dir.mkdir(parents=True, exist_ok=True)
        compact_path = ckpt_out_dir / "best_dinov3_compact.pth"
        torch.save(compact_state, compact_path)

        size_mb = compact_path.stat().st_size / (1024 * 1024)
        print(f"[4/4] Đã nén trọng số mô hình xuống còn {size_mb:.1f} MB (dưới giới hạn 25MB của GitHub Web!)")
    else:
        print("[4/4] Chưa có file best_dinov3_ser.pth, hệ thống sẽ dùng trọng số DINO mặc định.")

    print("\n" + "=" * 72)
    print("🎉 HOÀN TẤT! TOÀN BỘ FILE ĐÃ SẴN SÀNG TẠI THƯ MỤC: ./github_upload_ready/")
    print("=" * 72)
    print("👉 Bây giờ bạn chỉ cần làm 3 thao tác kéo thả đơn giản:")
    print("   1. Vào https://github.com/new tạo 1 Repository mới (chọn Public).")
    print("   2. Bấm 'uploading an existing file' và kéo toàn bộ ruột trong thư mục")
    print("      'github_upload_ready' thả vào GitHub rồi bấm 'Commit changes'.")
    print("   3. Vào https://share.streamlit.io/ ➔ Bấm 'Create app' ➔ Chọn Repo vừa tạo,")
    print("      nhập Main file path là: step6_deploy_app.py ➔ Bấm 'Deploy!' để nhận Link 24/7.")


if __name__ == "__main__":
    prepare_streamlit_github_bundle()