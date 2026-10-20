import csv
import tempfile
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Kết nối ngầm với các module xử lý âm thanh và mô hình AI từ Bước 1, 2, 3
from step1_data_collector import EMOTION_TO_IDX
from step2_edge_preprocessing import EdgeAudioPreprocessor
from step3_1d_swin_model import DINOv3AudioSER

st.set_page_config(
    page_title="Trợ lý Thấu hiểu Cảm xúc Giọng nói",
    page_icon="🎙️",
    layout="wide",
)

IDX_TO_EMOTION = {v: k for k, v in EMOTION_TO_IDX.items()}
CLASS_NAMES = [IDX_TO_EMOTION[i] for i in range(len(EMOTION_TO_IDX))]

# Bộ giao diện tiếng Việt thân thiện cho người dùng cuối
EMOTION_UI_INFO = {
    "happy": {
        "vn_name": "😊 Vui vẻ & Hào hứng",
        "badge": "Năng lượng tích cực",
        "color": "#059669",
        "bg_color": "#ECFDF5",
        "border_color": "#6EE7B7",
        "message": "Giọng nói của bạn rất sáng, giàu sức sống và lan tỏa sự vui tươi!",
        "tip": "Hãy lưu giữ khoảnh khắc này hoặc chia sẻ niềm vui với những người xung quanh nhé.",
    },
    "neutral": {
        "vn_name": "😌 Bình thản & Điềm tĩnh",
        "badge": "Trạng thái cân bằng",
        "color": "#2563EB",
        "bg_color": "#EFF6FF",
        "border_color": "#93C5FD",
        "message": "Thanh âm của bạn rất ổn định, nhẹ nhàng và cho thấy sự thư thái.",
        "tip": "Đây là trạng thái lý tưởng để bạn tập trung làm việc, đọc sách hoặc lên kế hoạch mới.",
    },
    "surprise": {
        "vn_name": "😲 Ngạc nhiên & Phấn khích",
        "badge": "Cảm xúc dâng trào",
        "color": "#D97706",
        "bg_color": "#FFFBEB",
        "border_color": "#FCD34D",
        "message": "Giọng của bạn có nhịp điệu biến thiên mạnh, thể hiện sự bất ngờ rõ nét!",
        "tip": "Có điều gì thú vị vừa xảy ra phải không? Hãy ghi chú lại vào nhật ký bên dưới nhé.",
    },
    "sad": {
        "vn_name": "😢 Trầm lắng & Tâm tư",
        "badge": "Cần sự vỗ về",
        "color": "#4F46E5",
        "bg_color": "#EEF2FF",
        "border_color": "#A5B4FC",
        "message": "Giọng nói của bạn hơi trầm và chậm lại, có vẻ bạn đang mệt hoặc có chuyện suy tư.",
        "tip": "Hãy tạm gác lại công việc một chút, nghe một bản nhạc dịu nhẹ và cho phép bản thân nghỉ ngơi nhé.",
    },
    "fear": {
        "vn_name": "😨 Lo lắng & Hồi hộp",
        "badge": "Cần thư giãn",
        "color": "#7C3AED",
        "bg_color": "#F5F3FF",
        "border_color": "#C4B5FD",
        "message": "Nhịp giọng của bạn hơi ngập ngừng và căng nhẹ, cho thấy bạn đang có chút áp lực.",
        "tip": "Hãy thử hít vào thật sâu trong 4 giây, giữ nhẹ và thở ra chậm rãi để cơ thể thả lỏng hơn.",
    },
    "angry": {
        "vn_name": "😠 Căng thẳng & Bức bối",
        "badge": "Cần hạ nhiệt",
        "color": "#DC2626",
        "bg_color": "#FEF2F2",
        "border_color": "#FCA5A5",
        "message": "Âm lượng và nhịp độ giọng nói cho thấy bạn đang khá căng thẳng hoặc bực bội.",
        "tip": "Hãy uống một ngụm nước mát, rời mắt khỏi màn hình vài phút để lấy lại sự bình tĩnh nhé.",
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("./emo_db_storage/checkpoints")
COMPACT_CKPT = CHECKPOINT_DIR / "best_dinov3_compact.pth"
FULL_CKPT = CHECKPOINT_DIR / "best_dinov3_ser.pth"

LOG_DIR = Path("./emo_db_storage/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "user_emotion_diary.csv"


@st.cache_resource
def load_ai_pipeline():
    """Khởi tạo bộ tiền xử lý và mô hình AI một lần duy nhất trên máy chủ."""
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


preprocessor, model = load_ai_pipeline()

# Phần đầu trang giao diện thân thiện
st.markdown(
    """
    <div style="text-align: center; max-width: 720px; margin: 0 auto 20px auto;">
        <h1 style="margin-bottom: 8px;">🎙️ Trợ lý Lắng nghe & Thấu hiểu Cảm xúc</h1>
        <p style="font-size: 16px; color: #475569; margin: 0;">
            Mỗi thanh âm đều mang một câu chuyện riêng. Hãy nói một câu ngắn qua Micro để khám phá sắc thái cảm xúc hiện tại trong giọng nói của bạn.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([5, 6], gap="large")

with col1:
    st.subheader("1. Chia sẻ giọng nói của bạn")
    tab_mic, tab_upload = st.tabs(["🎤 Thu âm trực tiếp", "📁 Tải file ghi âm (.wav)"])
    with tab_mic:
        mic_audio = st.audio_input("Bấm vào biểu tượng Micro để ghi âm (2 - 5 giây)")
    with tab_upload:
        uploaded_audio = st.file_uploader("Chọn file âm thanh (.wav) từ thiết bị", type=["wav"])

    user_note = st.text_area(
        "Đôi dòng tâm sự hôm nay (Không bắt buộc)",
        placeholder="Ví dụ: Hôm nay mình vừa hoàn thành xong một công việc quan trọng...",
        height=85,
    )
    submit_btn = st.button("✨ Khám phá Cảm xúc Ngay", type="primary", use_container_width=True)

with col2:
    st.subheader("2. Cảm nhận từ giọng nói của bạn")
    audio_source = mic_audio if mic_audio is not None else uploaded_audio

    if submit_btn:
        if audio_source is None:
            st.warning("🎙️ Bạn chưa ghi âm giọng nói! Vui lòng bấm nút Micro để thu âm hoặc tải một file .wav ở khung bên trái nhé.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_source.getvalue())
                tmp_path = tmp.name

            waveform = preprocessor.load_audio(tmp_path)
            if waveform.numel() < 1600 or torch.max(torch.abs(waveform)).item() < 1e-4:
                st.error("🔇 Âm thanh quá nhỏ hoặc quá ngắn. Hãy nói to rõ hơn trong khoảng 2 - 4 giây nhé!")
            else:
                s_1d = preprocessor(waveform, is_train=False)
                with torch.no_grad():
                    _, probs = model(s_1d.unsqueeze(0).to(DEVICE))
                    probs_np = probs.squeeze(0).cpu().numpy()

                pred_idx = int(np.argmax(probs_np))
                pred_key = IDX_TO_EMOTION[pred_idx]
                confidence = float(probs_np[pred_idx])
                ui = EMOTION_UI_INFO[pred_key]

                # Hiển thị Thẻ Cảm Xúc trực quan
                st.markdown(
                    f"""
                    <div style="padding: 24px; border-radius: 16px; background-color: {ui['bg_color']}; border: 2px solid {ui['border_color']}; margin-bottom: 16px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                            <span style="padding: 4px 12px; border-radius: 999px; background-color: white; color: {ui['color']}; font-size: 13px; font-weight: 700; border: 1px solid {ui['border_color']};">
                                ✨ {ui['badge']}
                            </span>
                            <span style="font-size: 14px; color: #475569; font-weight: 600;">
                                Độ rõ nét: {confidence * 100:.1f}%
                            </span>
                        </div>
                        <h2 style="margin: 14px 0 8px 0; font-size: 28px; color: {ui['color']};">{ui['vn_name']}</h2>
                        <p style="margin: 0 0 14px 0; font-size: 16px; color: #1E293B; line-height: 1.5;">{ui['message']}</p>
                        <div style="padding: 12px 16px; border-radius: 12px; background-color: rgba(255,255,255,0.8); color: #334155; font-size: 15px; line-height: 1.5;">
                            💡 <b>Gợi ý dành cho bạn:</b> {ui['tip']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("**Sự pha trộn các sắc thái cảm xúc trong giọng nói:**")
                for idx in np.argsort(probs_np)[::-1]:
                    emo_label = EMOTION_UI_INFO[CLASS_NAMES[idx]]["vn_name"]
                    pct = float(probs_np[idx])
                    st.progress(pct, text=f"{emo_label}: {pct * 100:.1f}%")

                # Lưu vào nhật ký người dùng
                file_exists = LOG_FILE.exists()
                with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["Thời gian", "Sắc thái cảm xúc", "Mức độ rõ nét", "Lời nhắn của bạn"])
                    writer.writerow([
                        datetime.now().strftime("%H:%M - %d/%m/%Y"),
                        ui["vn_name"],
                        f"{confidence * 100:.1f}%",
                        user_note.strip() if user_note else "—",
                    ])
    else:
        st.markdown(
            """
            <div style="padding: 36px 24px; border-radius: 16px; background-color: #F8FAFC; border: 2px dashed #CBD5E1; text-align: center;">
                <div style="font-size: 42px; margin-bottom: 10px;">🎧</div>
                <h3 style="margin: 0; color: #334155; font-size: 20px;">Sẵn sàng lắng nghe bạn</h3>
                <p style="margin: 8px 0 0 0; color: #64748B; font-size: 15px;">
                    Hãy thu âm một câu nói bất kỳ ở khung bên trái và bấm nút <b>"Khám phá Cảm xúc Ngay"</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.expander("📖 Nhật ký cảm xúc gần đây"):
    if LOG_FILE.exists():
        st.dataframe(pd.read_csv(LOG_FILE).tail(8).iloc[::-1], use_container_width=True)
    else:
        st.info("Chưa có bản ghi nhật ký nào.")