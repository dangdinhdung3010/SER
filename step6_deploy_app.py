import csv
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
# 1. CẤU HÌNH TRANG & GIAO DIỆN HIỆN ĐẠI (CLEAN CLINICAL & ANALYTICS DESIGN)
# ==============================================================================
st.set_page_config(
    page_title="Hệ thống Phân tích Cảm xúc Giọng nói",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1380px;
    }
    /* Thanh tiêu đề chính */
    .app-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #FFFFFF;
        padding: 22px 28px;
        border-radius: 14px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.08);
    }
    .app-title {
        font-size: 22px;
        font-weight: 700;
        margin: 0 0 4px 0;
        color: #F8FAFC;
    }
    .app-subtitle {
        font-size: 14px;
        color: #CBD5E1;
        margin: 0;
    }
    .status-pill {
        background-color: rgba(16, 185, 129, 0.15);
        border: 1px solid #10B981;
        color: #34D399;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
    }
    /* Tiêu đề từng khu vực */
    .section-header {
        font-size: 15px;
        font-weight: 700;
        color: #0F172A;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding-bottom: 8px;
        border-bottom: 2px solid #E2E8F0;
        margin-bottom: 16px;
    }
    /* Lưới 4 thẻ kết quả chính */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 14px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .metric-label {
        font-size: 12px;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 700;
        color: #0F172A;
        margin: 0 0 4px 0;
    }
    .metric-sub {
        font-size: 13px;
        color: #475569;
        margin: 0;
    }
    /* Khung đánh giá chất lượng mẫu thu */
    .quality-card {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 16px 18px;
        margin-top: 14px;
        margin-bottom: 16px;
        font-size: 14px;
        color: #1E293B;
        line-height: 1.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. DANH MỤC 6 SẮC THÁI CẢM XÚC (NGÔN NGỮ RÕ RÀNG, KHÁCH QUAN)
# ==============================================================================
IDX_TO_EMOTION = {v: k for k, v in EMOTION_TO_IDX.items()}
CLASS_NAMES = [IDX_TO_EMOTION[i] for i in range(len(EMOTION_TO_IDX))]

EMOTION_METADATA = {
    "happy": {
        "name": "😊 Vui vẻ & Hào hứng",
        "short_name": "Vui vẻ",
        "group": "Nhóm cảm xúc Tích cực",
        "color": "#059669",
        "voice_trait": "Âm sắc sáng, độ mở âm lượng lớn và giàu năng lượng",
    },
    "neutral": {
        "name": "😌 Bình thản & Điềm tĩnh",
        "short_name": "Bình thản",
        "group": "Nhóm trạng thái Cân bằng",
        "color": "#2563EB",
        "voice_trait": "Nhịp độ đều đặn, âm lượng ổn định, không có biến thiên đột ngột",
    },
    "surprise": {
        "name": "😲 Ngạc nhiên & Bất ngờ",
        "short_name": "Ngạc nhiên",
        "group": "Nhóm cảm xúc Hưng phấn cao",
        "color": "#D97706",
        "voice_trait": "Cao độ giọng vút lên nhanh, nhịp nhấn chuyển biến rõ rệt",
    },
    "sad": {
        "name": "😢 Trầm lắng & Buồn bã",
        "short_name": "Trầm buồn",
        "group": "Nhóm cảm xúc Trầm lắng",
        "color": "#4F46E5",
        "voice_trait": "Tông giọng trầm, năng lượng âm thanh giảm, nhịp nói chậm",
    },
    "fear": {
        "name": "😨 Lo âu & Hồi hộp",
        "short_name": "Lo âu",
        "group": "Nhóm cảm xúc Hồi hộp / Lo lắng",
        "color": "#7C3AED",
        "voice_trait": "Tần số giọng hơi cao, nhịp thở và âm tiết có độ ngập ngừng",
    },
    "angry": {
        "name": "😠 Căng thẳng & Gay gắt",
        "short_name": "Căng thẳng",
        "group": "Nhóm cảm xúc Cường độ mạnh",
        "color": "#DC2626",
        "voice_trait": "Cường độ âm thanh mạnh, lực phát âm dứt khoát và dồn dập",
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("./emo_db_storage/checkpoints")
COMPACT_CKPT = CHECKPOINT_DIR / "best_dinov3_compact.pth"
FULL_CKPT = CHECKPOINT_DIR / "best_dinov3_ser.pth"

LOG_DIR = Path("./emo_db_storage/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_LOG_FILE = LOG_DIR / "emotion_analysis_records.csv"


@st.cache_resource
def load_analysis_engine():
    """Khởi tạo bộ xử lý âm thanh và mô hình nhận diện cảm xúc trên máy chủ."""
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


preprocessor, model = load_analysis_engine()


# ==============================================================================
# 3. CÁC HÀM PHÂN TÍCH ĐẶC TRƯNG GIỌNG NÓI (CHUYỂN HÓA SANG NGÔN NGỮ ĐỜI SỐNG)
# ==============================================================================
def analyze_voice_characteristics(waveform: torch.Tensor, sr: int = 16000) -> dict:
    """Chuyển đổi các chỉ số vật lý sóng âm thành nhận định âm lượng và nhịp điệu dễ hiểu."""
    wav_np = waveform.squeeze().cpu().numpy()
    num_samples = int(wav_np.size)
    duration_s = num_samples / float(sr)
    peak_amp = float(np.max(np.abs(wav_np))) if num_samples > 0 else 0.0
    rms = float(np.sqrt(np.mean(np.square(wav_np)) + 1e-12))

    # Phân loại cường độ giọng nói
    if rms < 0.03:
        volume_desc = "Nhỏ & Trầm nhẹ"
    elif rms < 0.12:
        volume_desc = "Vừa phải, rõ tiếng"
    else:
        volume_desc = "Lớn & Giàu năng lượng"

    # Tính tỷ lệ khoảng phát âm có tiếng so với khoảng nghỉ
    frame_len = 320
    num_frames = num_samples // frame_len
    if num_frames > 0:
        frames = wav_np[: num_frames * frame_len].reshape(num_frames, frame_len)
        frame_energy = np.sqrt(np.mean(frames ** 2, axis=1))
        active_ratio = float(np.mean(frame_energy > (rms * 0.45)))
    else:
        active_ratio = 0.0

    if active_ratio > 0.72:
        pacing_desc = "Liên tục, nhịp độ nhanh"
    elif active_ratio > 0.45:
        pacing_desc = "Đều đặn, ngắt nghỉ tự nhiên"
    else:
        pacing_desc = "Chậm rãi, nhiều khoảng lặng"

    return {
        "samples": num_samples,
        "duration_s": duration_s,
        "peak_amp": peak_amp,
        "volume_desc": volume_desc,
        "pacing_desc": pacing_desc,
    }


def render_friendly_voice_charts(waveform: torch.Tensor, s_1d: torch.Tensor, sr: int = 16000):
    """Vẽ biểu đồ Nhịp nói và Bản đồ Âm sắc bằng chú thích tiếng Việt dễ hiểu."""
    wav_np = waveform.squeeze().cpu().numpy()
    time_axis = np.linspace(0, len(wav_np) / sr, num=len(wav_np))
    spec_2d = s_1d.transpose(0, 1).cpu().numpy()  # (32, 128)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 3.3), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")

    # Biểu đồ 1: Biến thiên âm lượng và nhịp nói theo thời gian
    ax1.set_facecolor("#F8FAFC")
    ax1.plot(time_axis, wav_np, color="#2563EB", linewidth=0.9, alpha=0.85)
    ax1.set_title("Biểu đồ Nhịp điệu & Độ mạnh nhẹ của Giọng nói", color="#0F172A", fontsize=10.5, fontweight="bold", pad=10)
    ax1.set_xlabel("Thời gian câu nói (Giây)", color="#475569", fontsize=9)
    ax1.set_ylabel("Độ mở âm lượng", color="#475569", fontsize=9)
    ax1.tick_params(colors="#64748B", labelsize=8)
    ax1.grid(True, linestyle="--", linewidth=0.5, color="#CBD5E1", alpha=0.7)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#CBD5E1")

    # Biểu đồ 2: Bản đồ sắc thái âm sắc (Trầm - Trung - Cao)
    ax2.set_facecolor("#F8FAFC")
    im = ax2.imshow(spec_2d, aspect="auto", origin="lower", cmap="YlOrRd", interpolation="bilinear")
    ax2.set_title("Bản đồ Sắc thái Âm sắc (Màu đậm thể hiện nơi nhấn giọng)", color="#0F172A", fontsize=10.5, fontweight="bold", pad=10)
    ax2.set_xlabel("Diễn tiến câu nói (Từ đầu câu ➔ cuối câu)", color="#475569", fontsize=9)
    ax2.set_ylabel("Tông giọng", color="#475569", fontsize=9)
    ax2.set_yticks([2, 15, 29])
    ax2.set_yticklabels(["Trầm", "Trung", "Cao"], fontsize=8.5, color="#334155", fontweight="bold")
    ax2.set_xticks([])
    for spine in ax2.spines.values():
        spine.set_edgecolor("#CBD5E1")

    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Năng lượng giọng", fontsize=8, color="#475569")
    cbar.ax.yaxis.set_tick_params(color="#64748B", labelsize=7.5)

    plt.tight_layout()
    return fig


def save_analysis_record(subject_id: str, duration_s: float, volume_desc: str, top1_name: str, top1_pct: float, pattern_type: str, top2_summary: str):
    """Lưu lịch sử phiên đo phục vụ theo dõi của bác sĩ hoặc cá nhân."""
    file_exists = HISTORY_LOG_FILE.exists()
    with open(HISTORY_LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Thời gian đo",
                "Mã hồ sơ / Người đo",
                "Thời lượng (giây)",
                "Cường độ giọng",
                "Cảm xúc chủ đạo",
                "Độ rõ nét (%)",
                "Tính chất biểu cảm",
                "Sắc thái đi kèm",
            ])
        writer.writerow([
            datetime.now().strftime("%H:%M:%S - %d/%m/%Y"),
            subject_id.strip() if subject_id.strip() else "Khách / Tự kiểm tra",
            f"{duration_s:.1f}s",
            volume_desc,
            top1_name,
            f"{top1_pct:.1f}%",
            pattern_type,
            top2_summary,
        ])


# ==============================================================================
# 4. HIỂN THỊ GIAO DIỆN CHÍNH (BỐ CỤC 2 CỘT TRỰC QUAN)
# ==============================================================================
st.markdown(
    """
    <div class="app-header">
        <div>
            <h1 class="app-title">🎙️ Hệ thống Phân tích Cảm xúc Giọng nói</h1>
            <p class="app-subtitle">Đánh giá khách quan sắc thái biểu cảm qua âm lượng, nhịp điệu và âm sắc giọng nói — Phục vụ Chuyên gia Tâm lý & Theo dõi Cá nhân</p>
        </div>
        <div class="status-pill">● Hệ thống sẵn sàng phân tích</div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_input, col_result = st.columns([4.5, 6.5], gap="large")

with col_input:
    st.markdown('<div class="section-header">1. Thu âm & Kiểm tra Mẫu giọng nói</div>', unsafe_allow_html=True)

    subject_code = st.text_input(
        "Mã hồ sơ / Tên người thực hiện (Tùy chọn)",
        placeholder="Ví dụ: BN-012 hoặc Nguyễn Văn A...",
    )

    tab_mic, tab_upload = st.tabs(["🎤 Ghi âm trực tiếp", "📁 Tải file ghi âm (.wav)"])
    with tab_mic:
        mic_audio = st.audio_input("Bấm biểu tượng Micro và nói một câu tự nhiên (2 – 5 giây)")
    with tab_upload:
        uploaded_audio = st.file_uploader("Chọn file âm thanh (.wav) từ máy tính hoặc điện thoại", type=["wav"])

    active_audio = mic_audio if mic_audio is not None else uploaded_audio
    analyze_btn = st.button("🔍 Thực hiện Phân tích Cảm xúc", type="primary", use_container_width=True)

analysis_result = None

if active_audio is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(active_audio.getvalue())
        tmp_wav_path = tmp.name

    waveform = preprocessor.load_audio(tmp_wav_path)
    voice_info = analyze_voice_characteristics(waveform)

    with col_input:
        is_valid = voice_info["samples"] >= 1600 and voice_info["peak_amp"] >= 1e-4
        status_text = "✅ Đạt chuẩn phân tích" if is_valid else "⚠️ Âm thanh quá nhỏ hoặc quá ngắn"
        status_color = "#059669" if is_valid else "#DC2626"

        st.markdown(
            f"""
            <div class="quality-card">
                <div style="font-weight:700; color:#0F172A; margin-bottom:6px;">📋 ĐÁNH GIÁ CHẤT LƯỢNG MẪU THU</div>
                • <b>Trạng thái mẫu:</b> <span style="color:{status_color}; font-weight:700;">{status_text}</span><br/>
                • <b>Thời lượng câu nói:</b> {voice_info['duration_s']:.2f} giây<br/>
                • <b>Cường độ âm lượng:</b> {voice_info['volume_desc']}<br/>
                • <b>Nhịp điệu phát âm:</b> {voice_info['pacing_desc']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not is_valid:
        with col_result:
            st.warning("Mẫu ghi âm chưa đủ độ dài hoặc quá nhỏ. Vui lòng nói rõ tiếng hơn trong khoảng 2 đến 5 giây để hệ thống phân tích chính xác.")
    else:
        s_1d = preprocessor(waveform, is_train=False)
        with torch.no_grad():
            _, probs = model(s_1d.unsqueeze(0).to(DEVICE))
            probs_np = probs.squeeze(0).cpu().numpy()

        sorted_idx = np.argsort(probs_np)[::-1]
        top1_idx, top2_idx = int(sorted_idx[0]), int(sorted_idx[1])
        top1_key, top2_key = IDX_TO_EMOTION[top1_idx], IDX_TO_EMOTION[top2_idx]

        top1_pct = float(probs_np[top1_idx] * 100.0)
        top2_pct = float(probs_np[top2_idx] * 100.0)
        diff_pct = top1_pct - top2_pct

        # Quy đổi Entropy sang Độ ổn định cảm xúc (0 - 100%) dễ hiểu cho bác sĩ & người dùng
        entropy = float(-np.sum(probs_np * np.log2(np.clip(probs_np, 1e-12, 1.0))))
        max_entropy = float(np.log2(len(CLASS_NAMES)))
        stability_pct = max(0.0, min(100.0, (1.0 - entropy / max_entropy) * 100.0))

        if diff_pct >= 25.0:
            pattern_type = "Biểu cảm Đơn nhất (Rõ ràng)"
            pattern_sub = f"Độ ổn định sắc thái: {stability_pct:.0f}%"
        else:
            pattern_type = "Cảm xúc Pha trộn"
            pattern_sub = f"Giao thoa giữa {EMOTION_METADATA[top1_key]['short_name']} & {EMOTION_METADATA[top2_key]['short_name']}"

        if top1_pct >= 70.0:
            clarity_desc = "Mức độ biểu hiện: Rất rõ nét"
        elif top1_pct >= 45.0:
            clarity_desc = "Mức độ biểu hiện: Khá rõ ràng"
        else:
            clarity_desc = "Mức độ biểu hiện: Nhẹ / Đa sắc thái"

        top2_summary = f"{EMOTION_METADATA[top2_key]['name']} ({top2_pct:.1f}%)"

        analysis_result = {
            "waveform": waveform,
            "s_1d": s_1d,
            "probs": probs_np,
            "sorted_idx": sorted_idx,
            "top1_key": top1_key,
            "top1_pct": top1_pct,
            "clarity_desc": clarity_desc,
            "pattern_type": pattern_type,
            "pattern_sub": pattern_sub,
            "top2_summary": top2_summary,
            "voice_info": voice_info,
        }

        if analyze_btn:
            save_analysis_record(
                subject_code,
                voice_info["duration_s"],
                voice_info["volume_desc"],
                EMOTION_METADATA[top1_key]["short_name"],
                top1_pct,
                pattern_type,
                f"{EMOTION_METADATA[top2_key]['short_name']} ({top2_pct:.1f}%)",
            )

with col_result:
    st.markdown('<div class="section-header">2. Kết quả Phân tích Sắc thái Cảm xúc</div>', unsafe_allow_html=True)

    if analysis_result is None:
        st.markdown(
            """
            <div style="background-color:#F8FAFC; border:2px dashed #CBD5E1; border-radius:14px; padding:48px 24px; text-align:center; color:#475569;">
                <div style="font-size:38px; margin-bottom:10px;">🎧</div>
                <h3 style="margin:0 0 6px 0; color:#0F172A; font-size:18px;">Chưa có mẫu giọng nói đầu vào</h3>
                <p style="margin:0; font-size:14px; color:#64748B;">
                    Hãy ghi âm trực tiếp hoặc tải một file .wav ở khung bên trái để xem bảng phân tích chi tiết 6 sắc thái cảm xúc.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        dom_meta = EMOTION_METADATA[analysis_result["top1_key"]]
        st.markdown(
            f"""
            <div class="metric-grid">
                <div class="metric-card" style="border-left: 5px solid {dom_meta['color']};">
                    <div class="metric-label">CẢM XÚC CHỦ ĐẠO</div>
                    <div class="metric-value" style="color: {dom_meta['color']};">{dom_meta['name']}</div>
                    <div class="metric-sub">{dom_meta['group']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">ĐỘ RÕ NÉT CỦA CẢM XÚC</div>
                    <div class="metric-value">{analysis_result['top1_pct']:.1f}%</div>
                    <div class="metric-sub">{analysis_result['clarity_desc']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">TÍNH CHẤT BIỂU CẢM</div>
                    <div class="metric-value" style="font-size:19px;">{analysis_result['pattern_type']}</div>
                    <div class="metric-sub">{analysis_result['pattern_sub']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">SẮC THÁI ĐI KÈM (THỨ CẤP)</div>
                    <div class="metric-value" style="font-size:19px;">{analysis_result['top2_summary']}</div>
                    <div class="metric-sub">Đặc trưng: {dom_meta['voice_trait']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Bảng tỷ lệ thành phần 6 sắc thái cảm xúc
        st.markdown("**Tỷ lệ thành phần 6 sắc thái cảm xúc nhận diện được trong giọng nói:**")
        table_rows = []
        for rank, idx in enumerate(analysis_result["sorted_idx"], start=1):
            emo_key = CLASS_NAMES[idx]
            meta = EMOTION_METADATA[emo_key]
            pct = float(analysis_result["probs"][idx] * 100.0)
            table_rows.append({
                "Thứ tự": f"#{rank}",
                "Sắc thái cảm xúc": meta["name"],
                "Nhóm biểu cảm": meta["group"],
                "Tỷ lệ tương ứng (%)": pct,
            })

        df_dist = pd.DataFrame(table_rows)
        st.dataframe(
            df_dist,
            column_config={
                "Tỷ lệ tương ứng (%)": st.column_config.ProgressColumn(
                    "Mức độ hiện diện (%)",
                    format="%.1f%%",
                    min_value=0.0,
                    max_value=100.0,
                ),
            },
            hide_index=True,
            use_container_width=True,
        )

# ==============================================================================
# 5. KHU VỰC 3: BIỂU ĐỒ TRỰC QUAN HÓA NHỊP ĐIỆU & ÂM SẮC GIỌNG NÓI
# ==============================================================================
st.markdown('<div class="section-header" style="margin-top:20px;">3. Biểu đồ Trực quan hóa Nhịp điệu & Âm sắc Giọng nói</div>', unsafe_allow_html=True)

if analysis_result is not None:
    voice_fig = render_friendly_voice_charts(analysis_result["waveform"], analysis_result["s_1d"])
    st.pyplot(voice_fig, use_container_width=True)
    plt.close(voice_fig)
else:
    st.caption("Biểu đồ nhịp nói và bản đồ âm sắc (Trầm – Trung – Cao) sẽ hiển thị tự động ngay khi bạn ghi âm hoặc chọn file âm thanh.")

# ==============================================================================
# 6. KHU VỰC 4: BẢNG LỊCH SỬ CÁC PHIÊN PHÂN TÍCH (XUẤT BÁO CÁO CSV)
# ==============================================================================
with st.expander("📁 4. Lịch sử các Phiên Phân tích Cảm xúc (Dành cho Bác sĩ & Theo dõi Cá nhân)", expanded=False):
    if HISTORY_LOG_FILE.exists():
        hist_df = pd.read_csv(HISTORY_LOG_FILE).tail(20).iloc[::-1]
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        csv_data = hist_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇ Tải Báo cáo Lịch sử Phân tích (.CSV / Excel)",
            data=csv_data,
            file_name="bao_cao_phan_tich_cam_xuc_giong_noi.csv",
            mime="text/csv",
        )
    else:
        st.info("Chưa có phiên lưu trữ nào. Hãy bấm nút 'Thực hiện Phân tích Cảm xúc' để lưu kết quả vào bảng theo dõi.")
