import os
import sys
from pathlib import Path

# 1. Khóa xung đột đa luồng DLL (OpenMP / MKL) trên Windows trước khi import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pytest
import timm
import torch

torch.set_num_threads(1)

# 2. Thêm thư mục gốc vào sys.path để import các file step1 -> step6
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from step1_data_collector import EMOTION_TO_IDX
from step2_edge_preprocessing import EdgeAudioPreprocessor
from step3_1d_swin_model import DINOv3AudioSER


@pytest.fixture(scope="module")
def preprocessor():
    return EdgeAudioPreprocessor()


@pytest.fixture(scope="module")
def ser_model():
    # Tạm thời chuyển pretrained=False trong lúc chạy Unit Test để tránh lỗi
    # Windows mmap access violation và giúp CI/CD chạy nhanh không cần tải mạng
    orig_create_model = timm.create_model

    def _safe_create_model(model_name, *args, **kwargs):
        kwargs["pretrained"] = False
        return orig_create_model(model_name, *args, **kwargs)

    timm.create_model = _safe_create_model
    try:
        model = DINOv3AudioSER(num_classes=len(EMOTION_TO_IDX))
        model.eval()
    finally:
        timm.create_model = orig_create_model

    return model


def test_emotion_labels_count():
    """1. Kiểm tra bộ nhãn cảm xúc luôn giữ đúng 6 lớp tiêu chuẩn."""
    assert len(EMOTION_TO_IDX) == 6
    expected_keys = {"angry", "happy", "fear", "sad", "neutral", "surprise"}
    assert set(EMOTION_TO_IDX.keys()) == expected_keys


def test_edge_preprocessing_output_shape(preprocessor):
    """2. Unit Test: Kiểm tra sóng âm 16kHz đi qua tầng Edge ra đúng Tensor (128, 32)."""
    sr = 16000
    t = torch.linspace(0, 3.0, sr * 3)
    dummy_waveform = 0.5 * torch.sin(2 * torch.pi * 220.0 * t).unsqueeze(0)

    s_1d = preprocessor(dummy_waveform, is_train=False)

    assert isinstance(s_1d, torch.Tensor)
    assert s_1d.shape == (128, 32), f"Sai kích thước đầu ra tầng Biên: {s_1d.shape}"
    assert not torch.isnan(s_1d).any(), "Tensor đầu ra chứa giá trị NaN"


def test_dinov3_adapter_and_forward_pass(ser_model):
    """3. Integration Test: Kiểm tra luồng Tensor 1D (2, 128, 32) qua Adapter + DINOv3 ra 6 xác suất."""
    dummy_s1d = torch.randn(2, 128, 32)

    with torch.no_grad():
        logits, probs = ser_model(dummy_s1d)

    assert logits.shape == (2, 6), f"Logits phải có shape (2, 6), nhận được {logits.shape}"
    assert probs.shape == (2, 6), f"Probs phải có shape (2, 6), nhận được {probs.shape}"

    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-4)


def test_step6_app_syntax_compile():
    """4. Kiểm tra file giao diện step6_deploy_app.py không có lỗi cú pháp Python."""
    app_path = ROOT_DIR / "step6_deploy_app.py"
    assert app_path.exists(), "Không tìm thấy file step6_deploy_app.py"
    source_code = app_path.read_text(encoding="utf-8")
    compile(source_code, str(app_path), "exec")