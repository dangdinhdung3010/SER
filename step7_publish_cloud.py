import sys

# Đảm bảo mã hóa UTF-8 cho console Windows tránh lỗi Unicode
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import shutil
from pathlib import Path
from huggingface_hub import HfApi, login

# ==============================================================================
# CẤU HÌNH THÔNG TIN DEPLOY LÊN CLOUD 24/7
# ==============================================================================
HF_TOKEN = ""

# Tên đường link ứng dụng của bạn (chỉ dùng chữ thường, số và dấu gạch ngang)
SPACE_NAME = "tro-ly-cam-xuc-giong-noi"
# ==============================================================================


def get_valid_token() -> str:
    """Lấy mã Hugging Face Write Token từ cấu hình hoặc hỏi trực tiếp trên Terminal."""
    token = HF_TOKEN.strip()
    if not token or token.startswith("hf_DAN_MA"):
        print("\n[HƯỚNG DẪN LẤY TOKEN MIỄN PHÍ TRONG 30 GIÂY]")
        print("1. Truy cập: https://huggingface.co/settings/tokens")
        print("2. Bấm 'Create new token' ➔ Chọn tab 'Write' ➔ Đặt tên bất kỳ ➔ Bấm 'Create token'")
        token = input("👉 Hãy dán mã Token (hf_...) của bạn vào đây rồi nhấn Enter: ").strip()

    if not token.startswith("hf_"):
        raise ValueError("Mã Token không hợp lệ! Mã Token của Hugging Face phải bắt đầu bằng 'hf_'.")
    return token


def build_cloud_bundle() -> Path:
    """Gom toàn bộ code Bước 1, 2, 3, 6 và trọng số Bước 4 vào thư mục chuẩn Cloud."""
    bundle_dir = Path("./cloud_deploy_bundle")
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # 1. Kiểm tra các file bắt buộc phải có
    required_files = [
        "step1_data_collector.py",
        "step2_edge_preprocessing.py",
        "step3_1d_swin_model.py",
        "step6_deploy_app.py",
    ]
    for req_file in required_files:
        if not Path(req_file).exists():
            raise FileNotFoundError(f"Thiếu file '{req_file}'! Hãy đảm bảo bạn đang chạy đúng thư mục gốc.")

    # 2. Tạo file requirements.txt cho máy chủ Cloud tự cài đặt thư viện
    requirements_txt = "\n".join([
        "torch",
        "torchaudio",
        "numpy",
        "scipy",
        "pandas",
        "timm",
        "huggingface_hub",
        "matplotlib",
    ])
    (bundle_dir / "requirements.txt").write_text(requirements_txt, encoding="utf-8")

    # 3. Copy các module lõi Bước 1, 2, 3
    for fname in ["step1_data_collector.py", "step2_edge_preprocessing.py", "step3_1d_swin_model.py"]:
        shutil.copy2(fname, bundle_dir / fname)

    # 4. Copy step6_deploy_app.py thành app.py (file khởi chạy mặc định trên Cloud)
    shutil.copy2("step6_deploy_app.py", bundle_dir / "app.py")
    print("[OK] Đã chuyển đổi step6_deploy_app.py ➔ app.py")

    # 5. Đóng gói kèm trọng số mô hình đã train ở Bước 4 (nếu có)
    ckpt_src = Path("./emo_db_storage/checkpoints/best_dinov3_ser.pth")
    if ckpt_src.exists():
        ckpt_dst_dir = bundle_dir / "emo_db_storage" / "checkpoints"
        ckpt_dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ckpt_src, ckpt_dst_dir / "best_dinov3_ser.pth")
        print(f"[OK] Đã đính kèm trọng số mô hình đã huấn luyện: {ckpt_src}")
    else:
        print("[INFO] Chưa có file best_dinov3_ser.pth, Cloud sẽ tự động dùng trọng số DINO mặc định.")

    return bundle_dir


def deploy_to_cloud_24_7():
    print("=== BẮT ĐẦU BƯỚC 7: ĐẨY HỆ THỐNG LÊN MÁY CHỦ CLOUD HOẠT ĐỘNG 24/7 ===")

    # 1. Xác thực tài khoản Hugging Face
    token = get_valid_token()
    login(token=token, add_to_git_credential=False)
    api = HfApi(token=token)

    user_info = api.whoami()
    username = user_info["name"]
    repo_id = f"{username}/{SPACE_NAME}"

    # 2. Đóng gói toàn bộ dự án
    print("\n[1/3] Đang đóng gói mã nguồn và mô hình AI...")
    bundle_dir = build_cloud_bundle()

    # 3. Khởi tạo Space trên Hugging Face Cloud
    print(f"[2/3] Đang khởi tạo máy chủ Cloud Space: {repo_id} ...")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
            private=False,
        )
    except Exception as e:
        err_msg = str(e)
        if "402" in err_msg or "PRO" in err_msg:
            print("\n" + "!" * 72)
            print("[THÔNG BÁO TỪ HUGGING FACE CLOUD]")
            print("Hugging Face vừa cập nhật chính sách toàn cầu:")
            print("Việc khởi tạo máy chủ chạy Gradio Space yêu cầu tài khoản đăng ký gói PRO ($9/tháng).")
            print("Tài khoản miễn phí hiện chỉ hỗ trợ các web tĩnh (Static Spaces).")
            print("-" * 72)
            print("💡 GIẢI PHÁP MIỄN PHÍ 100% THAY THẾ:")
            print("   Toàn bộ mã nguồn đã được đóng gói sẵn trong thư mục './cloud_deploy_bundle'.")
            print("   Để chia sẻ ứng dụng ngay cho người khác mà KHÔNG TỐN CHI PHÍ,")
            print("   bạn chỉ cần chạy Bước 6 (step6_deploy_app.py) — hệ thống sẽ tự động tạo")
            print("   đường link công khai toàn cầu (dạng https://xxxx.gradio.live) để gửi cho mọi người!")
            print("!" * 72 + "\n")
            return
        raise

    # 4. Tải toàn bộ thư mục lên Cloud (tự động xử lý file lớn .pth qua LFS)
    print("[3/3] Đang tải hệ thống lên máy chủ (vui lòng đợi khoảng 30 - 60 giây)...")
    api.upload_folder(
        folder_path=str(bundle_dir),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy 24/7 Friendly SER Web App with DINOv3",
    )

    direct_app_url = f"https://{username.lower()}-{SPACE_NAME.lower()}.hf.space"
    space_dashboard_url = f"https://huggingface.co/spaces/{repo_id}"

    print("\n" + "=" * 72)
    print("🎉 CHÚC MỪNG! HỆ THỐNG ĐÃ ĐƯỢC DEPLOY LÊN CLOUD 24/7 THÀNH CÔNG!")
    print("-" * 72)
    print(f"🔗 1. Link giao diện Web trực tiếp (Gửi cho người dùng):")
    print(f"      {direct_app_url}")
    print(f"\n🔗 2. Link quản lý trạng thái máy chủ (Hugging Face Space):")
    print(f"      {space_dashboard_url}")
    print("=" * 72)
    print("💡 LƯU Ý QUAN TRỌNG:")
    print("   - Lần đầu tiên sau khi tải lên, máy chủ Cloud cần 2 - 3 phút để cài đặt")
    print("     thư viện PyTorch & Gradio (hiển thị trạng thái 'Building' màu xanh dương).")
    print("   - Ngay khi trạng thái chuyển sang 'Running' (màu xanh lá), đường link trên")
    print("     sẽ hoạt động 24/7 vĩnh viễn kể cả khi bạn tắt máy tính!")


if __name__ == "__main__":
    deploy_to_cloud_24_7()