import sys

# Đảm bảo mã hóa UTF-8 cho console Windows tránh UnicodeEncodeError khi in tiếng Việt
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import csv
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import gradio as gr

# Kết nối ngầm với các module xử lý âm thanh và mô hình AI từ Bước 1, 2, 3
from step1_data_collector import EMOTION_TO_IDX
from step2_edge_preprocessing import EdgeAudioPreprocessor
from step3_1d_swin_model import DINOv3AudioSER

IDX_TO_EMOTION = {v: k for k, v in EMOTION_TO_IDX.items()}
CLASS_NAMES = [IDX_TO_EMOTION[i] for i in range(len(EMOTION_TO_IDX))]

# Bộ giao diện tiếng Việt thân thiện kèm lời nhắn và gợi ý chăm sóc tinh thần
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

# Khởi tạo thiết bị và đường dẫn lưu trữ
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = Path("./emo_db_storage/checkpoints/best_dinov3_ser.pth")
LOG_DIR = Path("./emo_db_storage/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "user_emotion_diary.csv"

# Nạp bộ tiền xử lý và mô hình chạy ngầm
preprocessor = EdgeAudioPreprocessor()
model = DINOv3AudioSER(num_classes=len(CLASS_NAMES)).to(DEVICE)

if CHECKPOINT_PATH.exists():
    try:
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    except TypeError:
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print(f"[READY] Đã nạp mô hình từ: {CHECKPOINT_PATH}")
else:
    print("[INFO] Khởi chạy với trọng số mặc định (chưa có file checkpoint riêng).")
model.eval()


def get_default_welcome_card() -> str:
    """Thẻ giao diện mặc định khi người dùng mới mở trang web."""
    return """
    <div style="padding: 36px 24px; border-radius: 16px; background-color: #F8FAFC; border: 2px dashed #CBD5E1; text-align: center;">
        <div style="font-size: 42px; margin-bottom: 10px;">🎧</div>
        <h3 style="margin: 0; color: #334155; font-size: 20px;">Sẵn sàng lắng nghe bạn</h3>
        <p style="margin: 8px 0 0 0; color: #64748B; font-size: 15px;">
            Hãy thu âm một câu nói bất kỳ ở khung bên trái và bấm nút <b>"Khám phá Cảm xúc Ngay"</b>.
        </p>
    </div>
    """


def save_diary_entry(emotion_vn: str, confidence: float, user_note: str):
    """Lưu nhật ký cảm xúc của người dùng."""
    file_exists = LOG_FILE.exists()
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Thời gian", "Sắc thái cảm xúc", "Mức độ rõ nét", "Lời nhắn của bạn"])
        writer.writerow([
            datetime.now().strftime("%H:%M - %d/%m/%Y"),
            emotion_vn,
            f"{confidence * 100:.1f}%",
            user_note.strip() if user_note else "—",
        ])


def load_diary_history() -> pd.DataFrame:
    """Đọc 8 lần kiểm tra gần nhất để hiển thị trong bảng nhật ký."""
    if LOG_FILE.exists():
        try:
            return pd.read_csv(LOG_FILE).tail(8).iloc[::-1]
        except Exception:
            pass
    return pd.DataFrame(columns=["Thời gian", "Sắc thái cảm xúc", "Mức độ rõ nét", "Lời nhắn của bạn"])


def analyze_user_voice(audio_filepath: str, user_note: str):
    """Hàm xử lý chính: Nhận giọng nói -> Phân tích ngầm -> Trả về thẻ giao diện thân thiện."""
    if not audio_filepath:
        warning_card = """
        <div style="padding: 24px; border-radius: 16px; background-color: #FFFBEB; border: 1px solid #FDE68A; text-align: center;">
            <h3 style="color: #B45309; margin: 0; font-size: 18px;">🎙️ Bạn chưa ghi âm giọng nói!</h3>
            <p style="color: #92400E; margin: 8px 0 0 0; font-size: 15px;">
                Vui lòng bấm nút <b>Record</b> để nói một câu ngắn hoặc chọn một file âm thanh có sẵn trên máy nhé.
            </p>
        </div>
        """
        return warning_card, {}, load_diary_history()

    try:
        waveform = preprocessor.load_audio(audio_filepath)
        # Kiểm tra nếu bản thu quá ngắn hoặc hoàn toàn im lặng
        if waveform.numel() < 1600 or torch.max(torch.abs(waveform)).item() < 1e-4:
            silent_card = """
            <div style="padding: 24px; border-radius: 16px; background-color: #FEF2F2; border: 1px solid #FECACA; text-align: center;">
                <h3 style="color: #DC2626; margin: 0; font-size: 18px;">🔇 Âm thanh quá nhỏ hoặc quá ngắn</h3>
                <p style="color: #991B1B; margin: 8px 0 0 0; font-size: 15px;">
                    Hệ thống chưa nghe rõ giọng của bạn. Hãy nói to hơn một chút trong khoảng 2 - 4 giây nhé!
                </p>
            </div>
            """
            return silent_card, {}, load_diary_history()

        s_1d = preprocessor(waveform, is_train=False)

        with torch.no_grad():
            s_input = s_1d.unsqueeze(0).to(DEVICE)
            _, probs = model(s_input)
            probs_np = probs.squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs_np))
        pred_key = IDX_TO_EMOTION[pred_idx]
        confidence = float(probs_np[pred_idx])
        ui = EMOTION_UI_INFO[pred_key]

        # Thiết kế Thẻ Cảm Xúc trực quan cho người dùng cuối
        result_card_html = f"""
        <div style="padding: 24px; border-radius: 16px; background-color: {ui['bg_color']}; border: 2px solid {ui['border_color']}; box-shadow: 0 4px 12px rgba(0,0,0,0.04);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <span style="padding: 4px 12px; border-radius: 999px; background-color: white; color: {ui['color']}; font-size: 13px; font-weight: 700; border: 1px solid {ui['border_color']};">
                    ✨ {ui['badge']}
                </span>
                <span style="font-size: 14px; color: #475569; font-weight: 600;">
                    Độ rõ nét: {confidence * 100:.1f}%
                </span>
            </div>
            <h2 style="margin: 14px 0 8px 0; font-size: 28px; color: {ui['color']};">{ui['vn_name']}</h2>
            <p style="margin: 0 0 14px 0; font-size: 16px; color: #1E293B; line-height: 1.5;">
                {ui['message']}
            </p>
            <div style="padding: 12px 16px; border-radius: 12px; background-color: rgba(255,255,255,0.75); color: #334155; font-size: 15px; line-height: 1.5;">
                💡 <b>Gợi ý dành cho bạn:</b> {ui['tip']}
            </div>
        </div>
        """

        prob_dict_vn = {
            EMOTION_UI_INFO[CLASS_NAMES[i]]["vn_name"]: float(probs_np[i])
            for i in range(len(CLASS_NAMES))
        }

        save_diary_entry(ui["vn_name"], confidence, user_note)
        return result_card_html, prob_dict_vn, load_diary_history()

    except Exception:
        error_card = """
        <div style="padding: 24px; border-radius: 16px; background-color: #FEF2F2; border: 1px solid #FECACA; text-align: center;">
            <h3 style="color: #DC2626; margin: 0;">⚠️ Không thể đọc được tệp âm thanh này</h3>
            <p style="color: #7F1D1D; margin-top: 8px;">Vui lòng thử thu âm lại trực tiếp bằng Micro hoặc chọn một file .wav khác nhé.</p>
        </div>
        """
        return error_card, {}, load_diary_history()


def reset_interface():
    """Làm mới giao diện để người dùng thu âm lượt tiếp theo."""
    return None, "", get_default_welcome_card(), {}


# Xây dựng Giao diện Người dùng (Friendly Web UI)
with gr.Blocks(theme=gr.themes.Soft(), title="Trợ lý Thấu hiểu Cảm xúc Giọng nói") as demo:
    gr.Markdown(
        """
        <div style="text-align: center; max-width: 720px; margin: 0 auto 12px auto;">
            <h1 style="margin-bottom: 8px;">🎙️ Trợ lý Lắng nghe & Thấu hiểu Cảm xúc</h1>
            <p style="font-size: 16px; color: #475569; margin: 0;">
                Mỗi thanh âm đều mang một câu chuyện riêng. Hãy nói một câu ngắn qua Micro để khám phá sắc thái cảm xúc hiện tại trong giọng nói của bạn.
            </p>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        # Cột trái: Thu âm & Ghi chú
        with gr.Column(scale=5):
            gr.Markdown("### 1. Chia sẻ giọng nói của bạn")
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Bấm vào biểu tượng Micro để ghi âm (2 - 5 giây) hoặc tải file ghi âm lên",
            )
            user_note = gr.Textbox(
                label="Đôi dòng tâm sự hôm nay (Không bắt buộc)",
                placeholder="Ví dụ: Hôm nay mình vừa hoàn thành xong một công việc quan trọng...",
                lines=2,
            )
            with gr.Row():
                clear_btn = gr.Button("🔄 Thu âm lại", variant="secondary")
                submit_btn = gr.Button("✨ Khám phá Cảm xúc Ngay", variant="primary", scale=2)

        # Cột phải: Thẻ Cảm xúc & Biểu đồ sắc thái
        with gr.Column(scale=6):
            gr.Markdown("### 2. Cảm nhận từ giọng nói của bạn")
            result_card = gr.HTML(value=get_default_welcome_card())
            prob_output = gr.Label(
                num_top_classes=6,
                label="Sự pha trộn các sắc thái cảm xúc trong giọng nói",
            )

    with gr.Accordion("📖 Nhật ký cảm xúc gần đây", open=False):
        history_table = gr.Dataframe(value=load_diary_history(), interactive=False)

    # Sự kiện bấm nút Phân tích và nút Làm mới
    submit_btn.click(
        fn=analyze_user_voice,
        inputs=[audio_input, user_note],
        outputs=[result_card, prob_output, history_table],
    )
    clear_btn.click(
        fn=reset_interface,
        inputs=[],
        outputs=[audio_input, user_note, result_card, prob_output],
    )


if __name__ == "__main__":
    # Tự động nhận diện: Nếu đang chạy trên Cloud 24/7 (Hugging Face Spaces) thì chạy chế độ Server chuẩn,
    # nếu chạy trên máy cá nhân (Antigravity IDE) thì tự động tạo link chia sẻ share=True.
    IS_CLOUD_SPACE = bool(os.environ.get("SPACE_ID"))

    if IS_CLOUD_SPACE:
        demo.launch()
    else:
        print("=== ĐANG KHỞI ĐỘNG GIAO DIỆN NGƯỜI DÙNG & TẠO LINK CHIA SẺ ===", flush=True)
        _, local_url, share_url = demo.launch(server_name="127.0.0.1", server_port=7860, share=True, inbrowser=True)
        print(f"\n" + "=" * 72, flush=True)
        print(f"👉 LINK TRUY CẬP CỤC BỘ: {local_url}", flush=True)
        print(f"👉 LINK CÔNG KHAI TOÀN CẦU: {share_url}", flush=True)
        print("=" * 72 + "\n", flush=True)