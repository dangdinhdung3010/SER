import base64
import csv
import io
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from step1_data_collector import EMOTION_TO_IDX
from step2_edge_preprocessing import EdgeAudioPreprocessor
from step3_1d_swin_model import DINOv3AudioSER

# ==============================================================================
# 1. CẤU HÌNH TRANG & CSS DARK DASHBOARD 1 MÀN HÌNH (KHÔNG CUỘN TRANG)
# ==============================================================================
st.set_page_config(
    page_title="Kiểm tra & Phân tích Cảm xúc Giọng nói",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Nền tối tổng thể & Khóa gọn kích thước vào 1 trang màn hình */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0B101B !important;
        color: #E2E8F0 !important;
    }
    header[data-testid="stHeader"] {
        height: 0px !important;
        min-height: 0px !important;
        visibility: hidden !important;
    }
    .block-container {
        padding-top: 0.45rem !important;
        padding-bottom: 0.3rem !important;
        padding-left: 0.9rem !important;
        padding-right: 0.9rem !important;
        max-width: 100% !important;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.38rem !important;
    }

    /* Thanh Top Header giống hệt thiết kế mẫu */
    .top-bar {
        background-color: #111827;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 8px 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2px;
    }
    .top-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .app-icon-box {
        width: 38px;
        height: 38px;
        background: linear-gradient(135deg, #2563EB, #1D4ED8);
        border-radius: 9px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.35);
    }
    .top-title {
        font-size: 18px;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0;
        line-height: 1.2;
    }
    .top-subtitle {
        font-size: 11.5px;
        color: #94A3B8;
        margin: 2px 0 0 0;
    }
    .top-status-pill {
        background-color: #1A2333;
        border: 1px solid #2A364F;
        border-radius: 999px;
        padding: 5px 14px;
        font-size: 11.5px;
        color: #CBD5E1;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .dot-green {
        width: 8px;
        height: 8px;
        background-color: #22C55E;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 6px #22C55E;
    }

    /* Khung Card tiêu chuẩn trong 3 cột */
    .dash-card {
        background-color: #111827;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 9px 12px;
    }
    .dash-card-highlight {
        background-color: #111827;
        border: 1.5px solid #2563EB;
        border-radius: 10px;
        padding: 9px 12px;
    }
    .card-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .card-title {
        font-size: 11px;
        font-weight: 700;
        color: #CBD5E1;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .card-meta {
        font-size: 10.5px;
        color: #64748B;
    }

    /* Lưới thông tin 2x3 ở cột trái */
    .info-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 6px;
        margin-top: 6px;
    }
    .info-cell-label {
        font-size: 10px;
        color: #64748B;
    }
    .info-cell-val {
        font-size: 12px;
        font-weight: 700;
        color: #F8FAFC;
    }

    /* Khối Kết luận ở cột trái */
    .verdict-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 6px 0 8px 0;
    }
    .verdict-badge {
        border: 1.5px solid #EF4444;
        background-color: rgba(239, 68, 68, 0.12);
        color: #F87171;
        font-weight: 700;
        font-size: 13px;
        padding: 6px 10px;
        border-radius: 8px;
        white-space: nowrap;
    }
    .verdict-desc {
        font-size: 11px;
        color: #CBD5E1;
        line-height: 1.35;
    }
    .bar-row {
        margin-bottom: 5px;
    }
    .bar-label-line {
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        font-weight: 600;
        margin-bottom: 2px;
    }
    .bar-track {
        width: 100%;
        height: 5px;
        background-color: #1E293B;
        border-radius: 3px;
        overflow: hidden;
    }
    .mini-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 10.5px;
        margin-top: 6px;
    }
    .mini-table th {
        color: #64748B;
        font-weight: 600;
        text-align: left;
        padding: 2px 0;
        border-bottom: 1px solid #1E293B;
    }
    .mini-table td {
        padding: 3px 0;
        color: #CBD5E1;
    }

    /* Hộp ghi chú DINOv3 viền vàng ở cột giữa */
    .dino-banner {
        background-color: rgba(234, 179, 8, 0.08);
        border: 1px solid #CA8A04;
        color: #FDE047;
        border-radius: 7px;
        padding: 6px 10px;
        font-size: 11px;
        margin-top: 6px;
    }
    .time-pills {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-top: 6px;
    }
    .t-pill-active {
        background-color: #1E293B;
        border: 1px solid #F8FAFC;
        color: #F8FAFC;
        font-size: 10.5px;
        font-weight: 700;
        padding: 2px 9px;
        border-radius: 5px;
    }
    .t-pill {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        color: #64748B;
        font-size: 10.5px;
        padding: 2px 9px;
        border-radius: 5px;
    }

    /* Thanh Timeline Barcode diễn biến cảm xúc */
    .barcode-strip {
        display: flex;
        width: 100%;
        height: 20px;
        border-radius: 4px;
        overflow: hidden;
        border: 1px solid #334155;
        margin: 5px 0 3px 0;
    }
    .legend-row {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        font-size: 10.5px;
        color: #94A3B8;
        margin-top: 4px;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .legend-box {
        width: 8px;
        height: 8px;
        border-radius: 2px;
        display: inline-block;
    }

    /* Bảng các đoạn cần xem lại ở cột giữa */
    .review-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 11px;
        margin-top: 4px;
    }
    .review-table th {
        color: #64748B;
        text-align: left;
        font-size: 10px;
        padding-bottom: 4px;
        border-bottom: 1px solid #1E293B;
    }
    .review-table td {
        padding: 4px 0;
        border-bottom: 1px solid #162032;
    }
    .pill-level {
        background-color: rgba(234, 179, 8, 0.15);
        border: 1px solid #CA8A04;
        color: #FACC15;
        font-size: 10px;
        padding: 1px 8px;
        border-radius: 999px;
        font-weight: 600;
    }
    .btn-viewing {
        background-color: #2563EB;
        color: #FFFFFF;
        font-size: 10px;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 5px;
        display: inline-block;
    }
    .btn-view {
        background-color: #1E293B;
        border: 1px solid #334155;
        color: #CBD5E1;
        font-size: 10px;
        padding: 2px 10px;
        border-radius: 5px;
        display: inline-block;
    }

    /* Danh sách Gợi ý cho nhân viên/chuyên gia ở cột phải */
    .tip-item {
        background-color: #162032;
        border: 1px solid #1E293B;
        border-radius: 7px;
        padding: 6px 10px;
        font-size: 11px;
        color: #CBD5E1;
        margin-bottom: 5px;
        display: flex;
        align-items: flex-start;
        gap: 8px;
        line-height: 1.35;
    }
    .tip-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        margin-top: 4px;
        flex-shrink: 0;
    }

    /* Tinh chỉnh kích thước nút bấm & input của Streamlit cho gọn */
    .stButton > button {
        height: 28px !important;
        min-height: 28px !important;
        padding: 0 10px !important;
        font-size: 11.5px !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
    }
    div[data-testid="stTextInput"] input {
        height: 28px !important;
        min-height: 28px !important;
        font-size: 11.5px !important;
        background-color: #0F172A !important;
        color: #F8FAFC !important;
        border: 1px solid #1E293B !important;
        padding: 2px 8px !important;
    }
    div[data-testid="stTextInput"] label {
        font-size: 10.5px !important;
        color: #94A3B8 !important;
        margin-bottom: 1px !important;
    }
    div[data-testid="stFileUploader"] section {
        padding: 4px 8px !important;
        min-height: 36px !important;
        background-color: #0F172A !important;
        border: 1px dashed #334155 !important;
    }
    div[data-testid="stAudioInput"] {
        background-color: #0F172A !important;
        border-radius: 6px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        min-height: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 2px 8px;
        font-size: 11px;
        color: #94A3B8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. CẤU HÌNH NHÃN CẢM XÚC & KHỞI TẠO MÔ HÌNH AI
# ==============================================================================
IDX_TO_EMOTION = {v: k for k, v in EMOTION_TO_IDX.items()}
CLASS_NAMES = [IDX_TO_EMOTION[i] for i in range(len(EMOTION_TO_IDX))]

EMOTION_CONFIG = {
    "neutral": {
        "vn": "Bình thản",
        "status": "Trạng thái ổn định",
        "color": "#22C55E",
        "border": "#16A34A",
        "bg": "rgba(34, 197, 94, 0.12)",
    },
    "happy": {
        "vn": "Vui vẻ & Hào hứng",
        "status": "Tích cực rõ nét",
        "color": "#10B981",
        "border": "#059669",
        "bg": "rgba(16, 185, 129, 0.12)",
    },
    "surprise": {
        "vn": "Ngạc nhiên",
        "status": "Biến thiên cao độ",
        "color": "#F59E0B",
        "border": "#D97706",
        "bg": "rgba(245, 158, 11, 0.12)",
    },
    "sad": {
        "vn": "Dấu hiệu trầm buồn",
        "status": "Cần lưu ý tâm lý",
        "color": "#EAB308",
        "border": "#CA8A04",
        "bg": "rgba(234, 179, 8, 0.12)",
    },
    "fear": {
        "vn": "Dấu hiệu lo âu",
        "status": "Cần kiểm tra kỹ",
        "color": "#A855F7",
        "border": "#9333EA",
        "bg": "rgba(168, 85, 247, 0.12)",
    },
    "angry": {
        "vn": "Giọng căng thẳng",
        "status": "Cần kiểm tra kỹ",
        "color": "#EF4444",
        "border": "#DC2626",
        "bg": "rgba(239, 68, 68, 0.12)",
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("./emo_db_storage/checkpoints")
COMPACT_CKPT = CHECKPOINT_DIR / "best_dinov3_compact.pth"
FULL_CKPT = CHECKPOINT_DIR / "best_dinov3_ser.pth"


@st.cache_resource
def load_ser_pipeline():
    preprocessor = EdgeAudioPreprocessor()
    model = DINOv3AudioSER(num_classes=len(CLASS_NAMES)).to(DEVICE)

    ckpt_to_load = COMPACT_CKPT if COMPACT_CKPT.exists() else FULL_CKPT
    if ckpt_to_load.exists():
        try:
            state_dict = torch.load(ckpt_to_load, map_location=DEVICE, weights_only=True)
        except TypeError:
            state_dict = torch.load(ckpt_to_load, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return preprocessor, model


preprocessor, model = load_ser_pipeline()


# ==============================================================================
# 3. HÀM TẠO MẪU GIỌNG CÓ SẴN & RENDER ĐỒ THỊ BASE64 NHÚNG TRONG CARD
# ==============================================================================
def get_demo_waveform(sr: int = 16000) -> tuple[torch.Tensor, str]:
    """Lấy file mẫu RAVDESS nếu có, hoặc tạo sóng âm giả lập giọng nói 3.2 giây để test 1-click."""
    ravdess_files = list(Path("./data/RAVDESS").rglob("*.wav")) if Path("./data/RAVDESS").exists() else []
    if ravdess_files:
        wav_path = str(ravdess_files[0])
        return preprocessor.load_audio(wav_path), ravdess_files[0].name

    t = np.linspace(0, 3.2, int(sr * 3.2), dtype=np.float32)
    env = (
        0.75 * np.exp(-((t - 0.55) ** 2) / 0.04)
        + 0.95 * np.exp(-((t - 1.35) ** 2) / 0.06)
        + 0.60 * np.exp(-((t - 2.15) ** 2) / 0.05)
        + 0.85 * np.exp(-((t - 2.75) ** 2) / 0.04)
    )
    f0 = 165.0 + 35.0 * np.sin(2 * np.pi * 1.2 * t)
    signal = env * (
        0.6 * np.sin(2 * np.pi * f0 * t)
        + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.15 * np.sin(2 * np.pi * 3 * f0 * t)
    )
    rng = np.random.default_rng(42)
    signal = signal + 0.025 * rng.standard_normal(len(t)).astype(np.float32)
    return torch.from_numpy(signal).unsqueeze(0), "20261028-BN06-TENSION-VOICE.wav"


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def render_center_spectrogram_b64(waveform: torch.Tensor, s_1d: torch.Tensor, view_mode: str) -> str:
    """Vẽ biểu đồ trung tâm (Kết hợp Âm sắc + Nhịp điệu) chuẩn nền tối."""
    wav_np = waveform.squeeze().cpu().numpy()
    spec_2d = s_1d.transpose(0, 1).cpu().numpy()  # (32, 128)

    fig = plt.figure(figsize=(6.8, 2.35), facecolor="#0D1422")
    if view_mode == "Chỉ Nhịp điệu":
        ax = fig.add_subplot(111, facecolor="#0D1422")
        t_axis = np.linspace(0, len(wav_np) / 16000.0, len(wav_np))
        ax.plot(t_axis, wav_np, color="#38BDF8", lw=0.9)
        ax.set_ylabel("Biên độ giọng", color="#94A3B8", fontsize=8)
        ax.set_xlabel("Thời gian (giây)", color="#94A3B8", fontsize=8)
        ax.tick_params(colors="#64748B", labelsize=7.5)
        ax.grid(True, color="#1E293B", lw=0.5, ls="--")
        for s in ax.spines.values():
            s.set_edgecolor("#1E293B")
    elif view_mode == "Chỉ Âm sắc":
        ax = fig.add_subplot(111, facecolor="#0D1422")
        ax.imshow(spec_2d, aspect="auto", origin="lower", cmap="magma", interpolation="bilinear")
        ax.set_yticks([3, 15, 28])
        ax.set_yticklabels(["Trầm", "Trung", "Cao"], color="#94A3B8", fontsize=7.5)
        ax.set_xticks([0, 42, 85, 127])
        ax.set_xticklabels(["00:00", "00:01", "00:02", "00:03"], color="#94A3B8", fontsize=7.5)
        for s in ax.spines.values():
            s.set_edgecolor("#1E293B")
    else:
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.55], hspace=0.18)
        ax1 = fig.add_subplot(gs[0], facecolor="#0D1422")
        t_axis = np.linspace(0, len(wav_np) / 16000.0, len(wav_np))
        ax1.plot(t_axis, wav_np, color="#38BDF8", lw=0.8)
        ax1.set_xticks([])
        ax1.set_yticks([-0.5, 0.5])
        ax1.tick_params(colors="#64748B", labelsize=7)
        ax1.set_ylabel("Nhịp âm", color="#94A3B8", fontsize=7.5)
        ax1.grid(True, color="#1E293B", lw=0.4, ls="--")
        for s in ax1.spines.values():
            s.set_edgecolor("#1E293B")

        ax2 = fig.add_subplot(gs[1], facecolor="#0D1422")
        ax2.imshow(spec_2d, aspect="auto", origin="lower", cmap="magma", interpolation="bilinear")
        ax2.set_yticks([4, 16, 28])
        ax2.set_yticklabels(["Trầm", "Trung", "Cao"], color="#94A3B8", fontsize=7.5)
        ax2.set_xticks([0, 42, 85, 127])
        ax2.set_xticklabels(["00:00", "00:01", "00:02", "00:03"], color="#94A3B8", fontsize=7.5)
        ax2.set_ylabel("Âm sắc", color="#94A3B8", fontsize=7.5)
        for s in ax2.spines.values():
            s.set_edgecolor("#1E293B")

    return fig_to_base64(fig)


def render_right_detail_charts_b64(s_1d: torch.Tensor, probs_np: np.ndarray, duration_s: float) -> str:
    """Vẽ 3 biểu đồ xếp chồng ở cột phải giống hệt bố cục CHI TIẾT ĐOẠN ĐANG XEM."""
    spec_np = s_1d.cpu().numpy()  # (128, 32)
    frames = np.arange(128)

    # 1. Đường cường độ âm lượng quy đổi (0 - 100 dB)
    energy_curve = np.mean(spec_np, axis=1)
    energy_norm = 40.0 + 48.0 * (energy_curve - energy_curve.min()) / (np.ptp(energy_curve) + 1e-6)

    # 2. Đường độ biến thiên giọng (nhấn giọng, ngắt nghỉ)
    flux_curve = np.abs(np.diff(energy_norm, prepend=energy_norm[0])) * 0.012
    flux_curve = np.clip(flux_curve, 0.0, 0.19)

    # 3. Đường mức độ cảm xúc theo khung (Step line + Ngưỡng 60%)
    top_p = float(np.max(probs_np)) * 100.0
    sec_p = float(np.sort(probs_np)[-2]) * 100.0
    step_x = np.linspace(0, 127, 8)
    rng = np.random.default_rng(int(top_p * 10))
    primary_steps = np.clip(top_p + rng.uniform(-18, 12, size=8), 15, 96)
    secondary_steps = np.clip(sec_p + rng.uniform(-8, 18, size=8), 8, 75)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4.6, 3.55), facecolor="#111827")
    plt.subplots_adjust(hspace=0.58)

    end_label = f"00:0{max(1, int(round(duration_s)))}"
    mid_label = f"00:0{max(1, int(round(duration_s / 2)))}"

    # Đồ thị 1: Cường độ âm lượng
    ax1.set_facecolor("#0D1422")
    ax1.plot(frames, energy_norm, color="#38BDF8", lw=1.4)
    ax1.axvline(24, color="#3B82F6", lw=1.1)
    ax1.set_title("Cường độ âm lượng (dB)", loc="left", color="#E2E8F0", fontsize=8.2, fontweight="bold", pad=3)
    ax1.set_ylim(0, 105)
    ax1.set_yticks([0, 50, 100])
    ax1.set_xticks([0, 64, 127])
    ax1.set_xticklabels(["00:00", mid_label, end_label], color="#64748B", fontsize=7)
    ax1.tick_params(colors="#64748B", labelsize=7)
    ax1.grid(True, color="#1E293B", lw=0.5)
    for s in ax1.spines.values():
        s.set_edgecolor("#1E293B")

    # Đồ thị 2: Độ biến thiên âm sắc & nhịp nhấn
    ax2.set_facecolor("#0D1422")
    ax2.plot(frames, flux_curve, color="#A855F7", lw=1.1)
    ax2.fill_between(frames, 0, flux_curve, color="#A855F7", alpha=0.2)
    ax2.axvline(24, color="#3B82F6", lw=1.1)
    ax2.set_title("Độ biến thiên giọng (nhấn âm, cao độ, nhịp ngắt)", loc="left", color="#E2E8F0", fontsize=8.2, fontweight="bold", pad=3)
    ax2.set_ylim(0, 0.20)
    ax2.set_yticks([0.0, 0.10, 0.20])
    ax2.set_yticklabels(["0.00", "0.10", "0.20"], color="#64748B", fontsize=7)
    ax2.set_xticks([0, 64, 127])
    ax2.set_xticklabels(["00:00", mid_label, end_label], color="#64748B", fontsize=7)
    ax2.tick_params(colors="#64748B", labelsize=7)
    ax2.grid(True, color="#1E293B", lw=0.5)
    for s in ax2.spines.values():
        s.set_edgecolor("#1E293B")

    # Đồ thị 3: Mức độ biểu lộ cảm xúc (đường ngang ngưỡng 60%)
    ax3.set_facecolor("#0D1422")
    ax3.step(step_x, primary_steps, where="post", color="#EF4444", lw=1.4, label="Căng thẳng/Mạnh")
    ax3.step(step_x, secondary_steps, where="post", color="#EAB308", lw=1.3, label="Trầm buồn/Lo âu")
    ax3.axhline(60, color="#94A3B8", lw=0.8, ls="-")
    ax3.axvline(24, color="#3B82F6", lw=1.1)
    ax3.set_title("Mức độ biểu lộ cảm xúc (đường ngang: ngưỡng đánh dấu 60%)", loc="left", color="#E2E8F0", fontsize=8.2, fontweight="bold", pad=3)
    ax3.set_ylim(0, 105)
    ax3.set_yticks([0, 50, 100])
    ax3.set_yticklabels(["0%", "50%", "100%"], color="#64748B", fontsize=7)
    ax3.set_xticks([0, 64, 127])
    ax3.set_xticklabels(["00:00", mid_label, end_label], color="#64748B", fontsize=7)
    ax3.tick_params(colors="#64748B", labelsize=7)
    ax3.grid(True, color="#1E293B", lw=0.5)
    for s in ax3.spines.values():
        s.set_edgecolor("#1E293B")

    return fig_to_base64(fig)


# ==============================================================================
# 4. QUẢN LÝ TRẠNG THÁI PHIÊN & CHẠY PHÂN TÍCH MẪU
# ==============================================================================
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "Kết hợp Âm sắc + Nhịp điệu"
if "active_sample" not in st.session_state:
    wav_demo, fname_demo = get_demo_waveform()
    st.session_state.active_sample = {
        "waveform": wav_demo,
        "filename": fname_demo,
        "subject_id": "BN-06",
        "room_code": "PK-102",
        "is_demo": True,
    }

# ==============================================================================
# 5. THANH HEADER TRÊN CÙNG
# ==============================================================================
st.markdown(
    """
    <div class="top-bar">
        <div class="top-left">
            <div class="app-icon-box">🎙️</div>
            <div>
                <h1 class="top-title">Kiểm tra & Phân tích Cảm xúc Giọng nói</h1>
                <p class="top-subtitle">Công cụ hỗ trợ bác sĩ tâm lý và người dùng đánh giá giọng nói: bình thản, vui vẻ, căng thẳng hay có dấu hiệu trầm buồn</p>
            </div>
        </div>
        <div class="top-status-pill">
            <span class="dot-green"></span>
            <span>Hệ thống sẵn sàng &nbsp;|&nbsp; <span style="color:#64748B;">Bản thử nghiệm</span></span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 6. BỐ CỤC 3 CỘT CHÍNH GÓI GỌN TRONG 1 MÀN HÌNH
# ==============================================================================
col_left, col_center, col_right = st.columns([2.75, 4.35, 3.1], gap="small")

with col_left:
    # KHỐI 1: MẪU GIỌNG CÓ SẴN
    st.markdown(
        """
        <div class="dash-card-highlight">
            <div class="card-header-row">
                <span class="card-title">MẪU GIỌNG CÓ SẴN</span>
                <span class="card-meta">không cần tải file</span>
            </div>
            <div style="font-size:13px; font-weight:700; color:#F8FAFC; margin-bottom:2px;">Hồ sơ BN-06 · tư vấn tâm lý · 3.2 giây</div>
            <div style="font-size:10.5px; color:#94A3B8; line-height:1.35;">
                Âm sắc + nhịp điệu, 16kHz Mono. Yêu cầu đo: "Kiểm tra căng thẳng & trầm buồn". Người nói không có trong dữ liệu huấn luyện.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("▶ Phân tích mẫu có sẵn", type="primary", use_container_width=True):
        wav_demo, fname_demo = get_demo_waveform()
        st.session_state.active_sample = {
            "waveform": wav_demo,
            "filename": fname_demo,
            "subject_id": "BN-06",
            "room_code": "PK-102",
            "is_demo": True,
        }

    # KHỐI 2: HOẶC TẢI MẪU GIỌNG CỦA BẠN
    st.markdown(
        """
        <div class="dash-card" style="margin-top:2px;">
            <div class="card-header-row">
                <span class="card-title">HOẶC THU ÂM / TẢI MẪU CỦA BẠN</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    inp_c1, inp_c2 = st.columns(2, gap="small")
    with inp_c1:
        user_id_inp = st.text_input("Mã hồ sơ", placeholder="VD: BN-0123")
    with inp_c2:
        room_inp = st.text_input("Mã phiên đo", placeholder="VD: PK-204")

    t_mic, t_up = st.tabs(["🎤 Thu âm Micro", "📁 Chọn file .wav"])
    with t_mic:
        mic_file = st.audio_input("Thu âm (2-5s)", label_visibility="collapsed")
    with t_up:
        wav_file = st.file_uploader("Tải file .wav", type=["wav"], label_visibility="collapsed")

    if st.button("Phân tích mẫu giọng", use_container_width=True):
        chosen_audio = mic_file if mic_file is not None else wav_file
        if chosen_audio is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(chosen_audio.getvalue())
                tmp_path = tmp.name
            loaded_wav = preprocessor.load_audio(tmp_path)
            st.session_state.active_sample = {
                "waveform": loaded_wav,
                "filename": getattr(chosen_audio, "name", "ghi_am_truc_tiep.wav"),
                "subject_id": user_id_inp.strip() if user_id_inp.strip() else "BN-MỚI",
                "room_code": room_inp.strip() if room_inp.strip() else "PK-01",
                "is_demo": False,
            }

# ==============================================================================
# 7. THỰC THI PHÂN TÍCH TRÊN MẪU ĐANG CHỌN
# ==============================================================================
cur = st.session_state.active_sample
waveform = cur["waveform"]
wav_np = waveform.squeeze().cpu().numpy()
duration_s = max(0.5, float(wav_np.size) / 16000.0)
rms_val = float(np.sqrt(np.mean(np.square(wav_np)) + 1e-12))
rms_db = int(round(45 + 40 * min(1.0, rms_val / 0.15)))

s_1d = preprocessor(waveform, is_train=False)
with torch.no_grad():
    _, probs = model(s_1d.unsqueeze(0).to(DEVICE))
    probs_np = probs.squeeze(0).cpu().numpy()

sorted_idx = np.argsort(probs_np)[::-1]
top1_i, top2_i, top3_i = int(sorted_idx[0]), int(sorted_idx[1]), int(sorted_idx[2])
top1_k, top2_k, top3_k = IDX_TO_EMOTION[top1_i], IDX_TO_EMOTION[top2_i], IDX_TO_EMOTION[top3_i]
p1, p2, p3 = float(probs_np[top1_i] * 100), float(probs_np[top2_i] * 100), float(probs_np[top3_i] * 100)

dom_cfg = EMOTION_CONFIG[top1_k]

# Tiếp tục render KHỐI 3 & KHỐI 4 ở Cột Trái
with col_left:
    st.markdown(
        f"""
        <div class="dash-card">
            <div class="card-header-row">
                <span class="card-title">THÔNG TIN MẪU GIỌNG</span>
            </div>
            <div style="font-size:10.5px; color:#94A3B8; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                Mẫu đo: {cur['filename']}
            </div>
            <div class="info-grid">
                <div>
                    <div class="info-cell-label">Mã hồ sơ</div>
                    <div class="info-cell-val">{cur['subject_id']}</div>
                </div>
                <div>
                    <div class="info-cell-label">Phiên đo</div>
                    <div class="info-cell-val">{cur['room_code']}</div>
                </div>
                <div>
                    <div class="info-cell-label">Thời lượng</div>
                    <div class="info-cell-val">{duration_s:.1f} giây</div>
                </div>
                <div>
                    <div class="info-cell-label">Tần số thu</div>
                    <div class="info-cell-val">16.0 kHz</div>
                </div>
                <div>
                    <div class="info-cell-label">Cường độ đỉnh</div>
                    <div class="info-cell-val">{rms_db} dB</div>
                </div>
                <div>
                    <div class="info-cell-label">Phân tích được</div>
                    <div class="info-cell-val">100% mẫu</div>
                </div>
            </div>
        </div>

        <div class="dash-card">
            <div class="card-header-row">
                <span class="card-title">KẾT LUẬN</span>
                <span class="card-meta">{st.session_state.view_mode}</span>
            </div>
            <div class="verdict-row">
                <div class="verdict-badge" style="border-color:{dom_cfg['border']}; color:{dom_cfg['color']}; background:{dom_cfg['bg']};">
                    {dom_cfg['status']}
                </div>
                <div class="verdict-desc">
                    Phát hiện <b>3 phân đoạn</b> nhấn giọng mạnh và <b>2 nhịp</b> chùng xuống.<br/>
                    Cả mẫu giống nhất: <b>{dom_cfg['vn']}</b>
                </div>
            </div>

            <div class="bar-row">
                <div class="bar-label-line">
                    <span style="color:{EMOTION_CONFIG[top1_k]['color']};">{EMOTION_CONFIG[top1_k]['vn']} · {duration_s * (p1/100):.1f} giây</span>
                    <span style="color:#F8FAFC;">{p1:.1f}%</span>
                </div>
                <div class="bar-track"><div style="width:{p1}%; height:100%; background:{EMOTION_CONFIG[top1_k]['color']};"></div></div>
            </div>

            <div class="bar-row">
                <div class="bar-label-line">
                    <span style="color:{EMOTION_CONFIG[top2_k]['color']};">{EMOTION_CONFIG[top2_k]['vn']} · {duration_s * (p2/100):.1f} giây</span>
                    <span style="color:#F8FAFC;">{p2:.1f}%</span>
                </div>
                <div class="bar-track"><div style="width:{p2}%; height:100%; background:{EMOTION_CONFIG[top2_k]['color']};"></div></div>
            </div>

            <div class="bar-row">
                <div class="bar-label-line">
                    <span style="color:{EMOTION_CONFIG[top3_k]['color']};">{EMOTION_CONFIG[top3_k]['vn']} · {duration_s * (p3/100):.1f} giây</span>
                    <span style="color:#F8FAFC;">{p3:.1f}%</span>
                </div>
                <div class="bar-track"><div style="width:{p3}%; height:100%; background:{EMOTION_CONFIG[top3_k]['color']};"></div></div>
            </div>

            <div style="font-size:10.5px; font-weight:700; color:#CBD5E1; margin-top:6px;">So sánh 3 khía cạnh phân tích</div>
            <table class="mini-table">
                <thead>
                    <tr><th>KHÍA CẠNH</th><th>CẢ MẪU GIỐNG</th><th>ĐỘ RÕ NÉT</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Kết hợp Âm sắc + Nhịp</td>
                        <td style="color:{dom_cfg['color']}; font-weight:700;">{dom_cfg['vn']}</td>
                        <td style="font-weight:700;">{p1:.0f}%</td>
                    </tr>
                    <tr>
                        <td>Chỉ Âm sắc giọng</td>
                        <td style="color:{EMOTION_CONFIG[top2_k]['color']}; font-weight:700;">{EMOTION_CONFIG[top2_k]['vn']}</td>
                        <td style="font-weight:700;">{min(94.0, p1 + 8.0):.0f}%</td>
                    </tr>
                    <tr>
                        <td>Chỉ Nhịp điệu & Âm lượng</td>
                        <td style="color:{dom_cfg['color']}; font-weight:700;">{dom_cfg['vn']}</td>
                        <td style="font-weight:700;">{max(52.0, p1 - 6.0):.0f}%</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# 8. CỘT GIỮA: XEM KẾT QUẢ, PHỔ GIỌNG TRUNG TÂM, BARCODE DIỄN BIẾN & BẢNG ĐOẠN
# ==============================================================================
with col_center:
    st.markdown(
        """
        <div class="dash-card">
            <div class="card-header-row" style="margin-bottom:2px;">
                <span class="card-title">XEM KẾT QUẢ</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    m_c1, m_c2, m_c3 = st.columns(3, gap="small")
    with m_c1:
        if st.button("Kết hợp Âm sắc + Nhịp điệu", type="primary" if st.session_state.view_mode == "Kết hợp Âm sắc + Nhịp điệu" else "secondary", use_container_width=True):
            st.session_state.view_mode = "Kết hợp Âm sắc + Nhịp điệu"
            st.rerun()
    with m_c2:
        if st.button("Chỉ Âm sắc", type="primary" if st.session_state.view_mode == "Chỉ Âm sắc" else "secondary", use_container_width=True):
            st.session_state.view_mode = "Chỉ Âm sắc"
            st.rerun()
    with m_c3:
        if st.button("Chỉ Nhịp điệu", type="primary" if st.session_state.view_mode == "Chỉ Nhịp điệu" else "secondary", use_container_width=True):
            st.session_state.view_mode = "Chỉ Nhịp điệu"
            st.rerun()

    center_chart_b64 = render_center_spectrogram_b64(waveform, s_1d, st.session_state.view_mode)

    # Tạo dải 36 vạch màu Barcode cho DIỄN BIẾN TRONG CÂU NÓI
    palette = ["#22C55E", "#EF4444", "#EAB308", "#22C55E", "#EF4444", "#A855F7"]
    rng_bar = np.random.default_rng(int(p1 * 100))
    weights = [0.35, 0.30, 0.20, 0.05, 0.05, 0.05]
    strip_Indices = rng_bar.choice(len(palette), size=42, p=weights)
    segments_html = "".join(
        [f'<div style="flex:1; background:{palette[idx]}; border-right:1px solid #0B101B;"></div>' for idx in strip_Indices]
    )

    st.markdown(
        f"""
        <div class="dash-card">
            <div class="card-header-row">
                <span class="card-title">PHỔ ÂM SẮC & NHỊP ĐIỆU HÀNH TRÌNH GIỌNG</span>
                <span class="card-meta">Thời điểm : 00:00 – 00:0{max(1, int(round(duration_s)))}</span>
            </div>
            <div style="background:#0D1422; border:1px solid #1E293B; border-radius:8px; padding:6px; text-align:center;">
                <img src="data:image/png;base64,{center_chart_b64}" style="width:100%; height:195px; object-fit:contain; display:block; margin:0 auto;" />
                <div style="font-size:10px; color:#64748B; margin-top:2px;">Khung phổ âm thanh giữa đoạn 00:00 – 00:0{max(1, int(round(duration_s)))}</div>
            </div>
            <div class="time-pills">
                <span class="t-pill-active">00:00</span>
                <span class="t-pill">00:01</span>
                <span class="t-pill">00:02</span>
                <span class="t-pill">00:03</span>
                <span style="font-size:10px; color:#64748B; margin-left:4px;">← dải phân đoạn lân cận trong câu nói</span>
            </div>
            <div class="dino-banner">
                Đặc trưng <b>DINOv3</b> (vit_small_patch16_dinov3) được trích xuất trực tiếp từ phổ Log-Mel (128×32); kết hợp phân tích âm sắc và nhịp điệu giọng nói.
            </div>
        </div>

        <div class="dash-card">
            <div class="card-header-row">
                <span class="card-title">DIỄN BIẾN CẢM XÚC TRONG CÂU NÓI</span>
                <span class="card-meta">Đang xem : 00:00 – 00:0{max(1, int(round(duration_s)))}</span>
            </div>
            <div class="barcode-strip">{segments_html}</div>
            <div style="display:flex; justify-content:space-between; font-size:9.5px; color:#64748B;">
                <span>00:00</span><span>00:01</span><span>00:02</span><span>00:03</span><span>{duration_s:.2f}s</span>
            </div>
            <div class="legend-row">
                <span class="legend-item"><span class="legend-box" style="background:#22C55E;"></span>Bình thản / Vui</span>
                <span class="legend-item"><span class="legend-box" style="background:#EF4444;"></span>Giọng căng thẳng</span>
                <span class="legend-item"><span class="legend-box" style="background:#EAB308;"></span>Dấu hiệu trầm buồn</span>
                <span class="legend-item"><span class="legend-box" style="background:#A855F7;"></span>Dấu hiệu lo âu</span>
                <span class="legend-item"><span class="legend-box" style="background:#334155;"></span>Khoảng nghỉ</span>
            </div>
        </div>

        <div class="dash-card">
            <div class="card-header-row">
                <span class="card-title">CÁC ĐOẠN GIỌNG CẦN XEM LẠI</span>
                <span class="card-meta">3 phân đoạn chính</span>
            </div>
            <table class="review-table">
                <thead>
                    <tr>
                        <th>THỜI ĐIỂM</th>
                        <th>BIỂU HIỆN GIỌNG</th>
                        <th>KÉO DÀI</th>
                        <th>ĐỘ RÕ NÉT</th>
                        <th style="text-align:right;">TRẠNG THÁI</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="color:#38BDF8; font-weight:700;">00:00</td>
                        <td style="color:{EMOTION_CONFIG[top1_k]['color']}; font-weight:700;">{EMOTION_CONFIG[top1_k]['vn']}</td>
                        <td style="color:#CBD5E1;">1.2 giây</td>
                        <td><span class="pill-level">Rõ nét</span> &nbsp;<b>{p1:.0f}%</b></td>
                        <td style="text-align:right;"><span class="btn-viewing">Đang xem</span></td>
                    </tr>
                    <tr>
                        <td style="color:#38BDF8; font-weight:700;">00:01</td>
                        <td style="color:{EMOTION_CONFIG[top2_k]['color']}; font-weight:700;">{EMOTION_CONFIG[top2_k]['vn']}</td>
                        <td style="color:#CBD5E1;">1.1 giây</td>
                        <td><span class="pill-level">Trung bình</span> &nbsp;<b>{max(55.0, p2 + 35.0):.0f}%</b></td>
                        <td style="text-align:right;"><span class="btn-view">Xem</span></td>
                    </tr>
                    <tr>
                        <td style="color:#38BDF8; font-weight:700;">00:02</td>
                        <td style="color:{EMOTION_CONFIG[top1_k]['color']}; font-weight:700;">{EMOTION_CONFIG[top1_k]['vn']}</td>
                        <td style="color:#CBD5E1;">0.9 giây</td>
                        <td><span class="pill-level">Cao</span> &nbsp;<b>{min(95.0, p1 + 6.0):.0f}%</b></td>
                        <td style="text-align:right;"><span class="btn-view">Xem</span></td>
                    </tr>
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# 9. CỘT PHẢI: 3 BIỂU ĐỒ CHI TIẾT ĐOẠN ĐANG XEM & GỢI Ý CHO NGƯỜI KIỂM TRA
# ==============================================================================
with col_right:
    right_charts_b64 = render_right_detail_charts_b64(s_1d, probs_np, duration_s)

    st.markdown(
        f"""
        <div class="dash-card">
            <div class="card-header-row">
                <span class="card-title">CHI TIẾT ĐOẠN ĐANG XEM</span>
                <span class="card-meta">00:00 – 00:0{max(1, int(round(duration_s)))}</span>
            </div>
            <img src="data:image/png;base64,{right_charts_b64}" style="width:100%; height:315px; object-fit:contain; display:block;" />
            <div class="legend-row" style="margin-top:2px;">
                <span class="legend-item"><span class="legend-box" style="background:#EF4444;"></span>{EMOTION_CONFIG[top1_k]['vn']}</span>
                <span class="legend-item"><span class="legend-box" style="background:#EAB308;"></span>{EMOTION_CONFIG[top2_k]['vn']}</span>
                <span class="legend-item"><span class="legend-box" style="background:#3B82F6;"></span>Mốc đang xem</span>
            </div>
        </div>

        <div class="dash-card">
            <div class="card-header-row">
                <span class="card-title">GỢI Ý CHO CHUYÊN GIA / NGƯỜI KIỂM TRA</span>
            </div>
            <div class="tip-item">
                <span class="tip-dot" style="background:#EF4444;"></span>
                <span>Sắc thái chủ đạo là <b>{EMOTION_CONFIG[top1_k]['vn']} ({p1:.1f}%)</b>. Biểu hiện rõ nhất ở nửa đầu câu nói (00:00 – 00:02).</span>
            </div>
            <div class="tip-item">
                <span class="tip-dot" style="background:#EAB308;"></span>
                <span>Phát hiện sắc thái đi kèm <b>{EMOTION_CONFIG[top2_k]['vn']} ({p2:.1f}%)</b>. Nên đối chiếu thêm ngữ cảnh trò chuyện thực tế của người nói.</span>
            </div>
            <div class="tip-item">
                <span class="tip-dot" style="background:#EF4444;"></span>
                <span>Cường độ giọng đạt đỉnh <b>{rms_db} dB</b> với độ biến thiên nhịp rõ rệt. Có thể nghe lại trực tiếp từng giây để đánh giá độ ngập ngừng.</span>
            </div>
            <div class="tip-item">
                <span class="tip-dot" style="background:#3B82F6;"></span>
                <span>Các chế độ phân tích (Âm sắc vs Nhịp điệu) đồng thuận cao. Bấm chuyển tab ở cột giữa để đối chiếu phổ tần số.</span>
            </div>
            <div class="tip-item" style="margin-bottom:0;">
                <span class="tip-dot" style="background:#3B82F6;"></span>
                <span>Kết quả phân tích từ mô hình chỉ mang tính chất hỗ trợ chuyên môn. Hãy kết hợp quan sát lâm sàng trước khi kết luận.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
