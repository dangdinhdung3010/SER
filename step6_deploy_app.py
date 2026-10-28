import csv
import tempfile
import time
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
# 1. CẤU HÌNH TRANG & DESIGN SYSTEM (DARK ENTERPRISE AI STUDIO)
# ==============================================================================
st.set_page_config(
    page_title="SER AI Inference Studio | DINOv3 Audio Analytics",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Tổng thể khoảng cách & Typography */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1440px;
    }
    /* Thanh Header trạng thái hệ thống */
    .studio-header {
        background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }
    .studio-title {
        color: #F8FAFC;
        font-size: 20px;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .status-dot {
        height: 10px;
        width: 10px;
        background-color: #10B981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #10B981;
    }
    .spec-badges {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .spec-badge {
        background-color: #0F172A;
        border: 1px solid #475569;
        color: #94A3B8;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    /* Thẻ KPI kết quả suy luận AI */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 16px;
    }
    .kpi-card {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 16px;
    }
    .kpi-label {
        color: #94A3B8;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #F8FAFC;
        font-family: 'Courier New', monospace;
        font-size: 22px;
        font-weight: 700;
        margin: 0;
    }
    .kpi-sub {
        color: #64748B;
        font-size: 11px;
        margin-top: 4px;
    }
    /* Bảng thông số tín hiệu tại biên */
    .signal-box {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 12px;
        margin-bottom: 14px;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        color: #E2E8F0;
    }
    .section-label {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748B;
        margin-bottom: 10px;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. CẤU HÌNH NHÃN KỸ THUẬT & KHỞI TẠO PIPELINE AI
# ==============================================================================
IDX_TO_EMOTION = {v: k for k, v in EMOTION_TO_IDX.items()}
CLASS_NAMES = [IDX_TO_EMOTION[i] for i in range(len(EMOTION_TO_IDX))]

CLASS_SPECS = {
    "angry": {"code": "ANG", "label": "ANGRY (Giận dữ)", "color": "#EF4444"},
    "happy": {"code": "HAP", "label": "HAPPY (Vui vẻ)", "color": "#10B981"},
    "fear": {"code": "FEA", "label": "FEAR (Sợ hãi)", "color": "#8B5CF6"},
    "sad": {"code": "SAD", "label": "SAD (Buồn bã)", "color": "#3B82F6"},
    "neutral": {"code": "NEU", "label": "NEUTRAL (Trung tính)", "color": "#64748B"},
    "surprise": {"code": "SUR", "label": "SURPRISE (Ngạc nhiên)", "color": "#F59E0B"},
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("./emo_db_storage/checkpoints")
COMPACT_CKPT = CHECKPOINT_DIR / "best_dinov3_compact.pth"
FULL_CKPT = CHECKPOINT_DIR / "best_dinov3_ser.pth"

LOG_DIR = Path("./emo_db_storage/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
TELEMETRY_LOG_FILE = LOG_DIR / "inference_telemetry.csv"


@st.cache_resource
def init_ser_engine():
    """Khởi tạo bộ tiền xử lý tín hiệu biên và mô hình DINOv3 trên RAM."""
    preprocessor = EdgeAudioPreprocessor()
    model = DINOv3AudioSER(num_classes=len(CLASS_NAMES)).to(DEVICE)

    ckpt_Source = "Pretrained-Base"
    ckpt_to_load = COMPACT_CKPT if COMPACT_CKPT.exists() else FULL_CKPT
    if ckpt_to_load.exists():
        try:
            state_dict = torch.load(ckpt_to_load, map_location=DEVICE, weights_only=True)
        except TypeError:
            state_dict = torch.load(ckpt_to_load, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
        ckpt_Source = ckpt_to_load.name

    model.eval()
    return preprocessor, model, ckpt_Source


preprocessor, model, active_ckpt_name = init_ser_engine()


# ==============================================================================
# 3. CÁC HÀM TÍNH TOÁN CHỈ SỐ TÍN HIỆU & TRỰC QUAN HÓA KỸ THUẬT
# ==============================================================================
def compute_signal_telemetry(waveform: torch.Tensor, sr: int = 16000) -> dict:
    """Tính toán các thông số vật lý của tín hiệu âm thanh tại tầng Edge."""
    wav_np = waveform.squeeze().cpu().numpy()
    num_samples = int(wav_np.size)
    duration_sec = num_samples / float(sr)
    peak_amp = float(np.max(np.abs(wav_np))) if num_samples > 0 else 0.0
    rms = float(np.sqrt(np.mean(np.square(wav_np)) + 1e-12))
    rms_db = float(20.0 * np.log10(max(rms, 1e-9)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(wav_np)))) / 2.0) if num_samples > 1 else 0.0
    return {
        "samples": num_samples,
        "duration_s": duration_sec,
        "peak_amp": peak_amp,
        "rms_db": rms_db,
        "zcr": zcr,
    }


def render_acoustic_plots(waveform: torch.Tensor, s_1d: torch.Tensor, sr: int = 16000):
    """Vẽ đồ thị Sóng âm miền thời gian và Phổ Log-Mel-Spectrogram (32×128) chuẩn Dark Studio."""
    wav_np = waveform.squeeze().cpu().numpy()
    time_axis = np.linspace(0, len(wav_np) / sr, num=len(wav_np))
    spec_2d = s_1d.transpose(0, 1).cpu().numpy()  # (32, 128)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 3.2), dpi=160)
    fig.patch.set_facecolor("#0F172A")

    # Đồ thị 1: 16kHz Waveform
    ax1.set_facecolor("#0F172A")
    ax1.plot(time_axis, wav_np, color="#38BDF8", linewidth=0.85, alpha=0.9)
    ax1.set_title("TIME-DOMAIN WAVEFORM (16kHz Mono)", color="#E2E8F0", fontsize=9.5, fontweight="bold", pad=8)
    ax1.set_xlabel("Time (seconds)", color="#94A3B8", fontsize=8)
    ax1.set_ylabel("Amplitude", color="#94A3B8", fontsize=8)
    ax1.tick_params(colors="#64748B", labelsize=7.5)
    ax1.grid(True, linestyle="--", linewidth=0.4, color="#334155", alpha=0.6)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#334155")

    # Đồ thị 2: Log-Mel Spectrogram (32 Mel bins x 128 Frames)
    ax2.set_facecolor("#0F172A")
    im = ax2.imshow(spec_2d, aspect="auto", origin="lower", cmap="magma", interpolation="nearest")
    ax2.set_title("Z-SCORE LOG-MEL SPECTROGRAM (32 Bins × 128 Frames)", color="#E2E8F0", fontsize=9.5, fontweight="bold", pad=8)
    ax2.set_xlabel("Temporal Frames (d = 128 | 10ms hop)", color="#94A3B8", fontsize=8)
    ax2.set_ylabel("Mel Frequency Bins (f = 32)", color="#94A3B8", fontsize=8)
    ax2.tick_params(colors="#64748B", labelsize=7.5)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#334155")

    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#64748B", labelsize=7)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94A3B8")

    plt.tight_layout()
    return fig


def append_telemetry_log(source_type: str, sig_meta: dict, pred_code: str, conf: float, margin: float, entropy: float, latency_ms: float):
    """Lưu log suy luận phục vụ giám sát mô hình (MLOps Telemetry)."""
    exists = TELEMETRY_LOG_FILE.exists()
    with open(TELEMETRY_LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow([
                "Timestamp", "Input_Source", "Duration_s", "RMS_dBFS",
                "Pred_Class", "Top1_Conf_%", "Margin_T1_T2_%", "Entropy_bits", "Latency_ms"
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source_type,
            f"{sig_meta['duration_s']:.2f}",
            f"{sig_meta['rms_db']:.1f}",
            pred_code,
            f"{conf * 100:.2f}",
            f"{margin * 100:.2f}",
            f"{entropy:.3f}",
            f"{latency_ms:.1f}",
        ])


# ==============================================================================
# 4. HIỂN THỊ HEADER & BỐ CỤC CHÍNH (MAIN WORKSPACE)
# ==============================================================================
backbone_name = getattr(model, "loaded_model_name", "vit_small_patch16_dinov3")
st.markdown(
    f"""
    <div class="studio-header">
        <div class="studio-title">
            <span class="status-dot"></span>
            SER AI INFERENCE & ACOUSTIC ANALYTICS CONSOLE
        </div>
        <div class="spec-badges">
            <span class="spec-badge">Backbone: {backbone_name}</span>
            <span class="spec-badge">Input Tensor: 1D (128×32)</span>
            <span class="spec-badge">Weights: {active_ckpt_name}</span>
            <span class="spec-badge">Compute: {str(DEVICE).upper()}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([4.5, 6.5], gap="large")

with col_left:
    st.markdown('<div class="section-label">1. AUDIO SIGNAL ACQUISITION (EDGE INPUT)</div>', unsafe_allow_html=True)
    tab_mic, tab_file = st.tabs(["🎙️ Live WebRTC Stream", "📂 Upload WAV File"])
    with tab_mic:
        mic_stream = st.audio_input("Record 16kHz Mono/Stereo Speech Signal")
    with tab_file:
        uploaded_wav = st.file_uploader("Select uncompressed .wav file", type=["wav"])

    active_audio = mic_stream if mic_stream is not None else uploaded_wav
    source_label = "WebRTC_Mic" if mic_stream is not None else ("WAV_Upload" if uploaded_wav is not None else "None")

    run_inference = st.button("⚡ EXECUTE AI INFERENCE PIPELINE", type="primary", use_container_width=True)

# Biến lưu trạng thái kết quả để vẽ đồ thị ở Zone 3
inference_payload = None

if active_audio is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(active_audio.getvalue())
        tmp_wav_path = tmp.name

    t0 = time.perf_counter()
    waveform = preprocessor.load_audio(tmp_wav_path)
    sig_meta = compute_signal_telemetry(waveform)

    with col_left:
        st.markdown(
            f"""
            <div class="signal-box">
                <div style="color:#38BDF8; font-weight:700; margin-bottom:6px;">[EDGE SIGNAL TELEMETRY]</div>
                • Sample Rate : 16,000 Hz (Mono, c=1)<br/>
                • Duration    : {sig_meta['duration_s']:.3f} sec ({sig_meta['samples']:,} samples)<br/>
                • RMS Energy  : {sig_meta['rms_db']:.2f} dBFS | Peak: {sig_meta['peak_amp']:.4f}<br/>
                • Zero-Cross  : {sig_meta['zcr']:.4f} | Pre-emphasis: α = 0.97
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Kiểm tra ngưỡng tín hiệu hợp lệ (VAD check)
    if sig_meta["samples"] < 1600 or sig_meta["peak_amp"] < 1e-4:
        with col_right:
            st.error("SIGNAL REJECTED (VAD TRIGGERED): Độ dài mẫu < 100ms hoặc biên độ tín hiệu dưới ngưỡng nhiễu nền (1e-4).")
    else:
        s_1d = preprocessor(waveform, is_train=False)  # (128, 32)
        with torch.no_grad():
            s_tensor = s_1d.unsqueeze(0).to(DEVICE)    # (1, 128, 32)
            logits, probs = model(s_tensor)
            logits_np = logits.squeeze(0).cpu().numpy()
            probs_np = probs.squeeze(0).cpu().numpy()

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Tính toán các chỉ số thống kê suy luận
        sorted_indices = np.argsort(probs_np)[::-1]
        top1_idx, top2_idx = int(sorted_indices[0]), int(sorted_indices[1])
        pred_key = IDX_TO_EMOTION[top1_idx]
        top1_prob = float(probs_np[top1_idx])
        margin_prob = float(probs_np[top1_idx] - probs_np[top2_idx])
        shannon_entropy = float(-np.sum(probs_np * np.log2(np.clip(probs_np, 1e-12, 1.0))))

        inference_payload = {
            "waveform": waveform,
            "s_1d": s_1d,
            "logits": logits_np,
            "probs": probs_np,
            "sorted_indices": sorted_indices,
            "pred_key": pred_key,
            "top1_prob": top1_prob,
            "margin": margin_prob,
            "entropy": shannon_entropy,
            "latency_ms": latency_ms,
        }

        if run_inference:
            append_telemetry_log(
                source_label,
                sig_meta,
                CLASS_SPECS[pred_key]["code"],
                top1_prob,
                margin_prob,
                shannon_entropy,
                latency_ms,
            )

with col_right:
    st.markdown('<div class="section-label">2. CLOUD AI MODEL INFERENCE & PROBABILITY DISTRIBUTION</div>', unsafe_allow_html=True)

    if inference_payload is None:
        st.markdown(
            """
            <div style="background-color:#0F172A; border:1px dashed #334155; border-radius:12px; padding:48px 24px; text-align:center; color:#64748B; font-family:monospace;">
                [WAITING FOR AUDIO STREAM]<br/>
                Provide a 16kHz audio signal on the left panel to inspect Logits, Softmax Probabilities, and Spectral Features.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        spec_info = CLASS_SPECS[inference_payload["pred_key"]]
        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card" style="border-left: 4px solid {spec_info['color']};">
                    <div class="kpi-label">TOP-1 PREDICTION</div>
                    <div class="kpi-value" style="color: {spec_info['color']};">{spec_info['code']}</div>
                    <div class="kpi-sub">{spec_info['label']}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">SOFTMAX CONFIDENCE</div>
                    <div class="kpi-value">{inference_payload['top1_prob'] * 100:.2f}%</div>
                    <div class="kpi-sub">Margin Δ(T1-T2): +{inference_payload['margin'] * 100:.1f}%</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">SHANNON ENTROPY (H)</div>
                    <div class="kpi-value">{inference_payload['entropy']:.3f}</div>
                    <div class="kpi-sub">Max Uncertainty: 2.585 bits</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">E2E LATENCY</div>
                    <div class="kpi-value">{inference_payload['latency_ms']:.1f} ms</div>
                    <div class="kpi-sub">Edge STFT + DINOv3 Forward</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Bảng & Thanh phân phối xác suất Softmax + Logits cho cả 6 lớp
        dist_rows = []
        for rank, idx in enumerate(inference_payload["sorted_indices"], start=1):
            cname = CLASS_NAMES[idx]
            prob_val = float(inference_payload["probs"][idx])
            logit_val = float(inference_payload["logits"][idx])
            dist_rows.append({
                "Rank": f"#{rank}",
                "Class Code": CLASS_SPECS[cname]["code"],
                "Emotion Class": CLASS_SPECS[cname]["label"],
                "Raw Logit (ŷ)": f"{logit_val:+.4f}",
                "Softmax Prob (ŷ')": prob_val * 100.0,
            })

        dist_df = pd.DataFrame(dist_rows)
        st.dataframe(
            dist_df,
            column_config={
                "Softmax Prob (ŷ')": st.column_config.ProgressColumn(
                    "Softmax Probability (%)",
                    format="%.2f%%",
                    min_value=0.0,
                    max_value=100.0,
                ),
            },
            hide_index=True,
            use_container_width=True,
        )

# ==============================================================================
# 5. ZONE 3: TRỰC QUAN HÓA ĐẶC TRƯNG TÍN HIỆU (WAVEFORM & LOG-MEL SPECTROGRAM)
# ==============================================================================
st.markdown('<div class="section-label" style="margin-top:16px;">3. ACOUSTIC FEATURE INSPECTOR (WAVEFORM & 32×128 LOG-MEL TENSOR)</div>', unsafe_allow_html=True)

if inference_payload is not None:
    acoustic_fig = render_acoustic_plots(inference_payload["waveform"], inference_payload["s_1d"])
    st.pyplot(acoustic_fig, use_container_width=True)
    plt.close(acoustic_fig)
else:
    st.caption("Biểu đồ miền thời gian (Waveform) và miền tần số - thời gian (Log-Mel-Spectrogram 32×128) sẽ tự động xuất hiện ngay khi có tín hiệu đầu vào.")

# ==============================================================================
# 6. ZONE 4: BẢNG TELEMETRY LOG SUY LUẬN (MLOPS OBSERVABILITY)
# ==============================================================================
with st.expander("📊 4. SESSION INFERENCE TELEMETRY LOG (MLOPS AUDIT TABLE)", expanded=False):
    if TELEMETRY_LOG_FILE.exists():
        log_df = pd.read_csv(TELEMETRY_LOG_FILE).tail(15).iloc[::-1]
        st.dataframe(log_df, use_container_width=True, hide_index=True)
        csv_bytes = log_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Export Telemetry CSV",
            data=csv_bytes,
            file_name="ser_inference_telemetry.csv",
            mime="text/csv",
        )
    else:
        st.info("Chưa có bản ghi Telemetry. Bấm nút 'EXECUTE AI INFERENCE PIPELINE' để lưu log suy luận.")
