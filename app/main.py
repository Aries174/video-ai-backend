from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
    Form,
)
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess, json
import os
import shutil
import json
import uuid
import urllib.parse
import subprocess
import re

from dotenv import load_dotenv

import whisper
import edge_tts
from openai import OpenAI

from moviepy import (
    VideoFileClip,
    AudioFileClip,
    CompositeAudioClip,
)

# Load .env (if present) so API keys / provider settings work
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
FONT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fonts"))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

WHISPER_MODEL = "small"

# ---------------------------------------------------------------------------
# LLM provider config (chọn động qua env var TRANSLATE_PROVIDER)
#
# Hỗ trợ:
#   - ollama  (local, free vô hạn)              → cần Ollama đang chạy
#   - gemini  (Google AI Studio, free 250/day)  → cần GEMINI_API_KEY
#   - groq    (Groq Cloud, free 14k tok/phút)   → cần GROQ_API_KEY
#
# Cách đổi: set biến môi trường (hoặc bỏ vào file .env ở project root)
#   TRANSLATE_PROVIDER=gemini
#   GEMINI_API_KEY=AIza...
# ---------------------------------------------------------------------------
PROVIDER_PRESETS = {
    "ollama": {
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "api_key_env": "OLLAMA_API_KEY",
        "default_api_key": "ollama",
        "default_model": "qwen2.5:14b",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "default_api_key": "AIzaSyDE_lgkYgzqTcZE7_sPWhIFY18A2Ti8cOA",  # bắt buộc set GEMINI_API_KEY
        "default_model": "gemini-2.5-flash",
    },
}

# ---- TTS speed tuning -----------------------------------------------------
# TTS_MAX_SPEED_RATIO: nén tối đa bao nhiêu lần khi TTS dài hơn segment.
#   1.0 = không bao giờ nén (audio sẽ tràn vào segment sau)
#   1.4 = nén tối đa 1.4x — tự nhiên, gần như không nhận ra (mặc định)
#   2.0 = nén tối đa 2x   — bắt đầu nghe "vội"
#   2.5 = nén tối đa 2.5x — rất nhanh, chói
# TTS_OVERFLOW_FACTOR: cho phép audio kéo dài tối đa bao nhiêu lần segment Whisper.
#   2.5 = chặt, hay phải nén
#   4.0 = nới — câu kế tiếp có thể bị đè 1 chút nhưng giọng tự nhiên hơn (mặc định)
TTS_MAX_SPEED_RATIO = float(os.getenv("TTS_MAX_SPEED_RATIO", "1.8"))
TTS_OVERFLOW_FACTOR = float(os.getenv("TTS_OVERFLOW_FACTOR", "3.5"))
print(f"[Config] TTS speed: max_ratio={TTS_MAX_SPEED_RATIO}, overflow_factor={TTS_OVERFLOW_FACTOR}")

TRANSLATE_PROVIDER = os.getenv("TRANSLATE_PROVIDER", "ollama").lower().strip()

if TRANSLATE_PROVIDER not in PROVIDER_PRESETS:
    print(f"[Config] Unknown TRANSLATE_PROVIDER='{TRANSLATE_PROVIDER}', falling back to 'ollama'")
    TRANSLATE_PROVIDER = "ollama"

_preset = PROVIDER_PRESETS[TRANSLATE_PROVIDER]
_api_key = os.getenv(_preset["api_key_env"], _preset["default_api_key"])

if not _api_key:
    print(
        f"[Config] WARNING: {_preset['api_key_env']} not set for provider "
        f"'{TRANSLATE_PROVIDER}'. Falling back to Ollama."
    )
    TRANSLATE_PROVIDER = "ollama"
    _preset = PROVIDER_PRESETS["ollama"]
    _api_key = os.getenv(_preset["api_key_env"], _preset["default_api_key"])

# Cho phép override model độc lập với provider
TRANSLATE_MODEL = os.getenv("TRANSLATE_MODEL", _preset["default_model"])

# Giữ lại tên cũ OLLAMA_MODEL để các log/khóa khác trong code không vỡ
OLLAMA_MODEL = TRANSLATE_MODEL

print(
    f"[Config] Translate provider: {TRANSLATE_PROVIDER} | "
    f"model: {TRANSLATE_MODEL} | base_url: {_preset['base_url']}"
)

DEFAULT_VOICE = "vi-VN-HoaiMyNeural"
DEFAULT_RATE = "+0%"

# ---------------------------------------------------------------------------
# VOICE_OPTIONS — multi-language để khớp với FE LANGUAGES dropdown
# Mỗi entry có:
#   - value: edge-tts voice id
#   - label: hiển thị FE
#   - lang : mã ngôn ngữ ngắn (vi/en/zh/ja/ko) để filter theo dropdown
# ---------------------------------------------------------------------------
VOICE_OPTIONS = [
    # Vietnamese
    {"value": "vi-VN-HoaiMyNeural",  "label": "HN - Hoài My (Nữ)",   "lang": "vi"},
    {"value": "vi-VN-NamMinhNeural", "label": "HN - Nam Minh (Nam)", "lang": "vi"},
    # English
    {"value": "en-US-AriaNeural",     "label": "US - Aria (Nữ)",      "lang": "en"},
    {"value": "en-US-GuyNeural",      "label": "US - Guy (Nam)",      "lang": "en"},
    {"value": "en-GB-SoniaNeural",    "label": "UK - Sonia (Nữ)",     "lang": "en"},
    # Chinese
    {"value": "zh-CN-XiaoxiaoNeural", "label": "CN - Xiaoxiao (Nữ)",  "lang": "zh"},
    {"value": "zh-CN-YunxiNeural",    "label": "CN - Yunxi (Nam)",    "lang": "zh"},
    # Japanese
    {"value": "ja-JP-NanamiNeural",   "label": "JP - Nanami (Nữ)",    "lang": "ja"},
    {"value": "ja-JP-KeitaNeural",    "label": "JP - Keita (Nam)",    "lang": "ja"},
    # Korean
    {"value": "ko-KR-SunHiNeural",    "label": "KR - Sun-Hi (Nữ)",    "lang": "ko"},
    {"value": "ko-KR-InJoonNeural",   "label": "KR - In-Joon (Nam)",  "lang": "ko"},
]

DEFAULT_VOICE = "vi-VN-HoaiMyNeural"
ALLOWED_VOICES = [v["value"] for v in VOICE_OPTIONS]

# Mã ngôn ngữ FE → ngôn ngữ Whisper dùng để transcribe
LANGUAGE_TO_WHISPER = {
    "vi": "vi", "en": "en", "zh": "zh", "ja": "ja", "ko": "ko",
}

model = whisper.load_model(WHISPER_MODEL)

client = OpenAI(
    base_url=_preset["base_url"],
    api_key=_api_key,
)

jobs = {}


# ---------------------------------------------------------------------------
# Demucs vocal separator: tách audio gốc thành (vocal, background)
# - vocal:      giọng người, dùng cho Whisper transcribe (chính xác hơn)
# - background: nhạc nền + sfx, dùng để giữ lại trong video output
#
# Lazy-load model lần đầu chạy (~250MB), cache lại cho lần sau.
# ---------------------------------------------------------------------------
_demucs_model = None


def _get_demucs_model():
    global _demucs_model
    if _demucs_model is None:
        import torch
        from demucs.pretrained import get_model

        # htdemucs là model mặc định mới nhất, chất lượng cao
        # Dùng htdemucs_ft cho chất lượng tốt nhất (nhưng nặng hơn).
        # CPU thì htdemucs đủ dùng, ~1.5x realtime.
        print("[Demucs] Loading model 'htdemucs' (lần đầu sẽ tải ~250MB)...")
        _demucs_model = get_model("htdemucs")
        _demucs_model.cpu()
        _demucs_model.eval()
        print("[Demucs] Model ready")
    return _demucs_model


def separate_vocals_and_background(audio_path, vocal_out_path, bg_out_path):
    """
    Tách audio_path thành 2 file:
    - vocal_out_path: chỉ giọng người
    - bg_out_path:    nhạc nền + tiếng động (drums + bass + other)

    Chạy CPU mất khoảng 0.5–1.5x thời lượng audio.
    """
    import torch
    import torchaudio
    import numpy as np
    from demucs.apply import apply_model
    from demucs.audio import convert_audio

    model = _get_demucs_model()

    # torchaudio 2.11+ yêu cầu torchcodec để load mp3.
    # Dùng ffmpeg trực tiếp → numpy → tensor để né dependency này.
    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"[Demucs] torchaudio.load failed ({e}), using ffmpeg fallback")
        # Decode bằng ffmpeg ra raw PCM s16le, sample rate 44100, stereo
        target_sr = model.samplerate
        target_ch = model.audio_channels
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", audio_path,
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ac", str(target_ch),
            "-ar", str(target_sr),
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, check=True)
        raw = np.frombuffer(proc.stdout, dtype=np.int16)
        # (samples * channels,) → (channels, samples)
        raw = raw.reshape(-1, target_ch).T.astype(np.float32) / 32768.0
        waveform = torch.from_numpy(raw)
        sr = target_sr

    waveform = convert_audio(waveform, sr, model.samplerate, model.audio_channels)

    print(f"[Demucs] separating {audio_path} ({waveform.shape[-1] / model.samplerate:.1f}s)...")

    with torch.no_grad():
        # apply_model expect (batch, channels, samples)
        sources = apply_model(
            model,
            waveform.unsqueeze(0),
            device="cpu",
            shifts=1,        # chất lượng OK, shifts=2 cao hơn nhưng chậm gấp đôi
            split=True,      # chia segment để đỡ tốn RAM
            overlap=0.25,
            progress=False,
        )[0]  # (sources, channels, samples)

    # model.sources là list ['drums', 'bass', 'other', 'vocals'] (htdemucs)
    sources_dict = {name: sources[i] for i, name in enumerate(model.sources)}

    vocals = sources_dict.get("vocals")
    if vocals is None:
        raise Exception("Demucs did not produce vocals stem")

    # Background = tổng của (drums + bass + other), giữ nguyên nhạc/sfx
    bg_parts = [sources_dict[k] for k in model.sources if k != "vocals"]
    background = sum(bg_parts) if bg_parts else torch.zeros_like(vocals)

    # Lưu ra WAV bằng ffmpeg để né torchcodec dependency
    def _save_wav(tensor, path, sr):
        try:
            torchaudio.save(path, tensor, sr)
        except Exception:
            # Fallback: pipe raw PCM qua ffmpeg
            arr = tensor.cpu().numpy()
            arr = np.clip(arr, -1.0, 1.0)
            arr_int = (arr * 32767).astype(np.int16)
            # (channels, samples) → interleaved
            interleaved = arr_int.T.flatten().tobytes()
            ch = tensor.shape[0]
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ac", str(ch),
                "-ar", str(sr),
                "-i", "-",
                path,
            ]
            subprocess.run(cmd, input=interleaved, check=True)

    _save_wav(vocals, vocal_out_path, model.samplerate)
    _save_wav(background, bg_out_path, model.samplerate)

    print(f"[Demucs] done → vocal={vocal_out_path}, bg={bg_out_path}")
    return vocal_out_path, bg_out_path


def update_job(job_id, progress, status, result=None):
    if job_id not in jobs:
        return

    jobs[job_id]["progress"] = progress
    jobs[job_id]["status"] = status

    if result is not None:
        jobs[job_id]["result"] = result


def contains_chinese(text):
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text))


def contains_bad_english(text):
    text = str(text or "").lower()
    banned = [
        "the",
        "this",
        "that",
        "you",
        "your",
        "what",
        "why",
        "how",
        "hello",
        "video",
        "follow",
        "like",
        "share",
        "comment",
        "subscribe",
        "douyin",
        "tiktok",
    ]
    return any(re.search(rf"\b{word}\b", text) for word in banned)


def clean_text(text):
    text = str(text or "").strip()
    text = text.replace("```json", "").replace("```", "")
    text = text.replace("{", "").replace("}", "")
    text = text.replace("[", "").replace("]", "")
    text = text.replace('"', "")
    text = text.replace("'", "")
    text = text.replace("...", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def shorten_words(text, max_words=10):
    words = str(text or "").split()

    if len(words) <= max_words:
        return str(text).strip()

    return " ".join(words[:max_words]).strip()


def is_bad_translation(text, target_lang="vi"):
    text = str(text or "").strip()

    if not text:
        return True

    if text.lower() in ["none", "null", "nan"]:
        return True

    # Nếu target KHÔNG phải tiếng Trung mà output có ký tự Trung → là dịch sót
    if target_lang != "zh" and contains_chinese(text):
        return True

    # Chỉ check lỗi "leak English" nếu target là tiếng Việt (đặc thù lỗi cũ)
    if target_lang == "vi" and contains_bad_english(text):
        return True

    return False


FALLBACK_TEXT = "Không rõ"
BATCH_SIZE = 10  # số câu mỗi lần gửi Ollama
CONTEXT_TAIL = 2  # số câu trước được đưa làm ngữ cảnh cho batch kế tiếp

# Whisper hay "ảo giác" những câu này khi gặp nhạc nền hoặc đoạn im lặng.
# QUAN TRỌNG: chỉ flag khi pattern là TOÀN BỘ câu (không phải substring),
# nếu không sẽ lọc oan câu thoại thật chứa particle phổ biến như "啊", "嗯".
WHISPER_HALLUCINATION_PATTERNS = [
    "请订阅", "请点赞", "感谢观看", "谢谢观看", "谢谢大家",
    "请关注", "请分享", "字幕由", "字幕组", "中文字幕",
    "感谢您的观看", "下次再见", "关注我", "点赞订阅",
    "明镜与点点", "由我提供",
]

# Câu chỉ gồm interjection thuần ("嗯", "啊啊", "哦哦哦") = noise
_PURE_INTERJECTION_RE = re.compile(r"^[嗯啊哦唔呃哎哈嘿嘻呀哇\s，。！？.,\!\?]+$")
# Bỏ dấu câu để so khớp pattern
_PUNCT_RE = re.compile(r"[，。！？、,.\!\?\s]+")


def is_whisper_hallucination(text):
    text = str(text or "").strip()
    if not text:
        return True

    # Câu chỉ chứa particle/interjection (sau khi strip dấu câu)
    if _PURE_INTERJECTION_RE.match(text):
        return True

    # 1-2 ký tự lặp ("嗯嗯", "啊啊啊") thường là noise
    if len(text) <= 3 and len(set(text.replace(" ", ""))) <= 1:
        return True

    # Chỉ flag pattern khi nó CHIẾM gần như toàn bộ câu, không phải nằm trong câu dài
    cleaned = _PUNCT_RE.sub("", text)
    if not cleaned:
        return True

    for pat in WHISPER_HALLUCINATION_PATTERNS:
        # Pattern là toàn bộ câu, hoặc chiếm ≥ 80% độ dài câu
        if cleaned == pat:
            return True
        if pat in cleaned and len(pat) >= 0.8 * len(cleaned):
            return True

    return False


SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[。！？；;!?,，])\s*')
MAX_TRANSLATION_CHUNK_CHARS = 100


def split_text_into_chunks(text, max_chars=MAX_TRANSLATION_CHUNK_CHARS):
    text = str(text or "").strip()
    if not text:
        return []

    parts = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(text) if part.strip()]
    if not parts:
        return [text]

    chunks = []
    for part in parts:
        if len(part) <= max_chars:
            chunks.append(part)
            continue

        if " " in part:
            current = []
            current_len = 0
            for word in part.split():
                if current and current_len + len(word) + 1 > max_chars:
                    chunks.append(" ".join(current).strip())
                    current = [word]
                    current_len = len(word)
                else:
                    current.append(word)
                    current_len += len(word) + (1 if current_len else 0)
            if current:
                chunks.append(" ".join(current).strip())
        else:
            for i in range(0, len(part), max_chars):
                chunks.append(part[i : i + max_chars].strip())

    if not chunks:
        return [text]
    return chunks


def split_segments_for_translation(segments):
    output = []

    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            output.append(seg)
            continue

        chunks = split_text_into_chunks(text)
        if len(chunks) <= 1:
            output.append(seg)
            continue

        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        duration = max(0.0, end - start)
        total_chars = sum(max(1, len(chunk)) for chunk in chunks)
        current = start

        for idx, chunk in enumerate(chunks):
            if idx == len(chunks) - 1:
                chunk_end = end
            elif duration > 0:
                ratio = len(chunk) / total_chars
                chunk_end = current + duration * ratio
            else:
                chunk_end = current

            output.append({
                "start": current,
                "end": chunk_end,
                "text": chunk,
                "original_text": chunk,
            })
            current = chunk_end

    return output


LANG_NAME_VI = {
    "vi": "tiếng Việt", "en": "tiếng Anh", "zh": "tiếng Trung",
    "ja": "tiếng Nhật", "ko": "tiếng Hàn", "fr": "tiếng Pháp",
    "es": "tiếng Tây Ban Nha", "de": "tiếng Đức",
}


def _translate_one_batch(batch_segments, prev_context_pairs=None,
                         source_lang="zh", target_lang="vi"):
    """Dịch 1 batch nhỏ (≤ BATCH_SIZE) câu. Trả về list[str] cùng độ dài batch.

    prev_context_pairs: list[(src, tgt)] các câu trước đã dịch, để giữ mạch.
    """
    src_name = LANG_NAME_VI.get(source_lang, source_lang)
    tgt_name = LANG_NAME_VI.get(target_lang, target_lang)
    # Mỗi câu kèm budget = thời lượng segment (giây) → LLM biết phải dịch ngắn cỡ nào
    numbered_lines = []
    for i, seg in enumerate(batch_segments):
        try:
            dur = max(0.4, float(seg.get("end", 0)) - float(seg.get("start", 0)))
        except Exception:
            dur = 2.0
        numbered_lines.append(
            f"{i+1}. [budget={dur:.1f}s] {seg['text'].strip()}"
        )
    numbered = "\n".join(numbered_lines)

    if prev_context_pairs:
        ctx_lines = "\n".join(
            f"- 「{zh}」 → 「{vi}」"
            for zh, vi in prev_context_pairs
            if zh and vi
        )
        context_block = f"""NGỮ CẢNH (các câu ngay trước batch này, đã được dịch — dùng để giữ mạch văn, KHÔNG dịch lại):
{ctx_lines}

"""
    else:
        context_block = ""

    # Phần guidelines tiếng Việt chỉ có ý nghĩa khi đang dịch SANG tiếng Việt.
    # Nếu target khác (en/ja/...), bỏ qua các rule và glossary chuyên Việt.
    is_target_vi = (target_lang == "vi")
    is_source_zh = (source_lang == "zh")

    vi_specific_rules = ""
    vi_specific_glossary = ""
    vi_examples = ""

    if is_target_vi:
        vi_specific_rules = """
3. KHÔNG dùng Hán-Việt cứng khi tiếng Việt thuần đủ dùng:
   • 进行 → "đang/làm", KHÔNG "tiến hành"
   • 实施 → "làm/triển khai", KHÔNG "thực thi"
   • 非常 → "rất/cực", KHÔNG luôn "vô cùng"
   • 之前 → "trước/lúc trước", KHÔNG "trước đó"
"""
        if is_source_zh:
            vi_specific_glossary = """
<topic_detection>
TRƯỚC tiên đoán THỂ LOẠI video qua các câu phía trên:
• Có tên động vật (海獭/狮子/老虎/海豚/...) + 妈妈/幼崽/捕食/栖息地 → PHIM TÀI LIỆU thiên nhiên
• Có tên mèo (橘子/小橘/大橘) + 老板/兄弟/赚钱/茶水费 → DRAMA TIKTOK mèo AI
• Có 我/你/今天/咱们 + đời thường → VLOG / TIKTOK
Chọn glossary tương ứng. KHÔNG dùng glossary drama mèo cho phim động vật và ngược lại.
</topic_detection>

<glossary_wildlife>
Phim tài liệu động vật/thiên nhiên — TỪ VỰNG BẮT BUỘC:

ĐỘNG VẬT BIỂN:
• 海獭 = "rái cá biển" (KHÔNG phải "tháp nước"/"sếp"/"ông chủ"!)
• 海豚 = "cá heo"
• 鲸鱼 = "cá voi"
• 海豹 = "hải cẩu"
• 海狮 = "sư tử biển"
• 海象 = "hải mã"
• 海星 = "sao biển"
• 海胆 = "nhím biển"
• 鲨鱼 = "cá mập"
• 章鱼 = "bạch tuộc"
• 海龟 = "rùa biển"
• 螃蟹 / 蟹 = "cua"

ĐỘNG VẬT TRÊN CẠN:
• 狮子 = "sư tử"
• 老虎 = "hổ"
• 大象 = "voi"
• 猴子 = "khỉ"
• 鳄鱼 = "cá sấu"
• 长颈鹿 = "hươu cao cổ"

THUẬT NGỮ TỰ NHIÊN:
• 海藻林 / 海藻森林 = "rừng tảo bẹ" (KHÔNG "rừng biển xanh")
• 珊瑚礁 = "rạn san hô"
• 栖息地 = "môi trường sống"
• 庇护 / 庇护所 = "che chở" / "nơi trú ẩn"
• 食物链 = "chuỗi thức ăn"
• 捕食者 = "kẻ săn mồi"
• 猎物 = "con mồi"
• 哺乳动物 = "động vật có vú"
• 物种 = "loài"
• 灭绝 = "tuyệt chủng"
• 阿留申群岛 = "quần đảo Aleutian" (KHÔNG "Schiedam")

QUAN HỆ GIA ĐÌNH ĐỘNG VẬT:
• 妈妈 / 母 (trong phim tài liệu) = "(con) mẹ" hoặc "rái cá mẹ", "sư tử mẹ"... — GHÉP với loài
   ví dụ: 母海獭 = "rái cá mẹ", 母狮 = "sư tử cái"
• 幼崽 / 宝宝 / 小崽 = "con non" / "đứa con" (KHÔNG "cậu bé"/"bé nhỏ"!)
• 三个孩子 (trong phim động vật) = "ba đứa con" / "ba con non"

XƯNG HÔ TRONG PHIM TÀI LIỆU:
• Lời dẫn (narrator) → trang trọng, dùng "chúng" cho động vật, KHÔNG "anh ấy/cô ấy/sếp"
• Tránh "mình", "tao/mày", "sếp" trong phim tài liệu

VÍ DỤ ĐÚNG (phim tài liệu):
- "海獭妈妈" → "rái cá mẹ"
- "海藻森林是它们的家" → "Rừng tảo bẹ là nhà của chúng"
- "幼崽完全依赖母亲" → "Con non phụ thuộc hoàn toàn vào mẹ"
- "猎食很困难" → "Săn mồi rất khó khăn"
</glossary_wildlife>

<glossary_names>
Tên nhân vật meme TikTok DRAMA (CHỈ dùng khi rõ là drama mèo/chó AI):
• 橘子 / 橘猫 / 小橘 / 大橘 = "Quýt" (mèo cam)
• 大黄 / 小黄 = "Vàng" (chó vàng)
• 小白 / 白白 = "Bạch" / "Trắng"
• 黑炭 / 小黑 = "Đen"
• 鼠 / 老鼠 (TÊN nhân vật drama) = "Chuột"
• 鼠猫 / 鼠和猫 (cặp đôi) = "Chuột và Mèo"
• 老板 = "sếp" / "ông chủ" / "boss"
• 兄弟 / 兄弟们 = "anh em" / "đám"
• 哥们 = "huynh đệ" / "ông bạn"
• 你们 (đám đông không thân) = "các người" / "bọn bay"
• 我们 (nhóm thân) = "tụi mình" / "bọn mình"

⚠️ QUAN TRỌNG: Trong PHIM TÀI LIỆU, 老板 KHÔNG có nghĩa "sếp" mà thường là Whisper transcribe nhầm — bỏ qua hoặc dịch context-aware.
</glossary_names>

<glossary_idioms>
Idiom & slang TUYỆT ĐỐI không dịch chữ-một:
• 干完了 / 火干完了 = "xong việc" / "làm xong" (KHÔNG dịch "tắt lửa")
• 不愧是 = "Quả không hổ danh" / "Quả là"
• 头脑灵活 = "đầu óc nhanh nhạy"
• 凭啥 = "dựa vào đâu" / "lấy gì cớ"
• 茶水费 = "tiền trà nước" / "tiền lót tay"
• 别吵了 = "đừng cãi nữa"
• 不对啊 = "không đúng rồi"
• 这一千都给你们 = "ngàn này cho tụi bay"
• 这样才对 = "vậy mới đúng"
• 哈哈 = "haha"
• 上车 = "lên xe"
• 等等我们 = "đợi tụi tao với"
• 整天拉着 = "suốt ngày lôi kéo"
• 油钱 = "tiền xăng"
• 喂不熟的狗 = "đám chó vô ơn"
• 自己走路回去 = "tự đi bộ về"
• 怎么走了 = "sao đi rồi"
• 咋回事 = "có chuyện gì" / "sao thế"
• 不知道啊 = "không biết nữa"
• 发了工资 = "nhận lương"
• 一起去吃饭 = "đi ăn cùng"
• 你晚上过来 = "tối qua đây" (晚上 = tối)
• 顺便聚聚 = "tiện thể tụ tập"
• 订房间 = "đặt phòng"
• 不懂事 = "không biết điều"
• 重新找别人 = "tìm người khác đi"
• 离谱 = "vô lý"
• 笑死 = "cười xỉu"
• 绝了 = "đỉnh thật"
• 真的假的 = "thật hay đùa"
• 白眼狼 = "đồ vong ơn"
• 加油 = "cố lên"
• 服了 = "chịu rồi"
• 没事 / 没事儿 = "không sao"
• 谁能徒手扛起钢卷 = "ai cầm nổi cuộn thép này"
• 钢卷就是谁的 = "cuộn thép thuộc về người đó"
• 你个傻狗给我闭嘴 = "Đồ chó ngu ngốc, câm miệng đi!"
• 这钢卷不错,拿去卖了赚点零花钱。 = "Cuộn thép này tốt đấy;bán nó để kiếm chút tiền tiêu vặt."
• 你个傻狗给我闭嘴 = "Đồ chó ngu ngốc, câm miệng đi!"
• 水壶里加一把白糖,再倒上白醋,然后多加一点自来水。 = "Cho một nắm đường trắng vào ấm, sau đó đổ giấm trắng vào, rồi thêm một ít nước lọc nữa."
• 把水烧开以后,再静置十分钟,时间到了,直接把水倒出来。 = "Sau khi đun sôi nước, để yên trong mười phút. Khi hết thời gian, đổ nước đi."
• 你看,水里面夹杂着水购,水壶的内胆也干净如新了 = "Hãy nhìn xem, nước có lẫn những giọt nước nhỏ, nhưng bên trong ấm vẫn sạch như mới."
• 最后倒60度的开水,这样每个月清理一次,下水道就不会堵了,更不会有异味和小飞虫。 = "Cuối cùng, đổ nước sôi ở 60 độ C vào. Vệ sinh cống rãnh mỗi tháng một lần sẽ giúp ngăn ngừa tắc nghẽn."
• 5000万的单子,你把客户手打断了? = "Hợp đồng 50 triệu mà cậu lại đánh gãy tay khách hàng à?"
• 起洁精加白醋,喷在油烟机上,油汙一擦就掉。 = "Pha nước rửa chén với giấm trắng, xịt lên máy hút mùi thì vết dầu mỡ lau một cái là sạch ngay."
• • 起洁精加白醋,喷在油烟机上,油汙一擦就掉。 = "Ahhhhhhhhhhh"
• 盆里挤上点牙膏,撒上点食盐,加点洗衣液,倒上白醋。 = "Bóp một ít kem đánh răng vào chậu, thêm chút muối, nước giặt và đổ thêm giấm trắng."
• 然后把裤子拿出来,裤子干干净净。 = "Sau đó lấy quần ra, chiếc quần sẽ sạch sẽ như mới."
• 加入热水,把氧化发黄的白色裤子放入盆中浸泡一会儿, = "Đổ nước nóng vào, rồi ngâm chiếc quần trắng bị ố vàng do oxy hóa vào chậu một lúc."
• 新来的,你贴钻多钱一天? = "Người mới à, một ngày em nhận dán đá bao nhiêu tiền?"
• 老板,三百一天,这么贵,你别干了,我外包出去。 = "Sếp, 300 tệ một ngày. Đắt vậy à? Thôi em nghỉ đi, tôi thuê ngoài cho rồi."
• 三百还贵,你外包更不划算啊。 = "300 mà còn chê đắt à? Thuê ngoài còn tốn hơn đấy."
• 这不用你管,你明天别来了 = "Không cần em lo, mai khỏi tới nữa."
• 你小声点,接个电话,是表哥呀。 = "Nói nhỏ thôi, tôi nghe điện thoại cái. À anh họ đấy à."
• 我这儿有个贴钻的活打算包给你,你要不要做?= "Đúng là tự chuốc lấy mà."
• 造孽呀 = "Đúng là tự chuốc lấy mà."
• 遇到这样很小的防水圈套不上怎么办。= “Gặp loại vòng chống nước nhỏ như thế này mà không lắp vào được thì làm sao?”
• 就像我这样,用一条绳子穿过防水圈,拉扯两边的绳子, = “Giống như tôi đây, dùng một sợi dây luồn qua vòng chống nước rồi kéo hai đầu dây,”
• 防水圈就会自己跑进水管的槽里面。 = "vòng chống nước sẽ tự chui vào rãnh của ống nước."
• 老婆,我们工地缺人,你带着小菊一起过来吧。 = "Vợ ơi, công trường đang thiếu người, em dẫn Tiểu Cúc qua đây nhé."
• 太好了,我们俩一起赚钱,这样你也轻松一点。 = "Tuyệt quá, hai vợ chồng mình cùng kiếm tiền, như vậy anh cũng đỡ vất vả hơn."
• 耶,可以去爸爸那里啦。 = "Yay, cuối cùng cũng được đến chỗ bố rồi."
• 妈妈,还有多久到爸爸那儿? = "Mẹ ơi, còn bao lâu nữa mới đến chỗ bố vậy?"
• 马上就要到了。 = "Sắp tới nơi rồi."
• 妈妈,爸爸在那儿。 = "Mẹ ơi, bố ở bên kia kìa."
• 爸爸,爸爸,我好想你。 = "Bố ơi, con nhớ bố lắm."
• 小菊,爸爸也想你。 = "Tiểu Cúc, bố cũng nhớ con lắm."
• 老婆,我们以后就这儿了,报哥特意安排了单间宿舍。 = "Vợ à, sau này chúng ta sẽ sống ở đây, anh Bá đã đặc biệt sắp xếp cho mình một phòng riêng."
• 真好,我们一家人可以住一起了。 = "Tốt quá, cuối cùng cả gia đình cũng được ở cùng nhau."
• 太好了 我太喜欢这里了 = "Tuyệt quá, em thích nơi này lắm."
• 可以每天和爸爸妈妈在一起了 = "Giờ ngày nào cũng được ở cùng bố mẹ rồi."
• 胖菊 这工地就你最卖力 = "Phát Triển à, ở công trường này cậu là người chăm chỉ nhất đấy."
• 这么拼干嘛 = "Sao phải cố gắng đến vậy?"
• 为了老婆孩子嘛 = "Vì vợ con thôi mà."
• 我得努力挣钱 = "Anh phải cố gắng kiếm nhiều tiền hơn."
• 我努力一点 = "Em cũng sẽ cố gắng thêm một chút."
• 老公压力也小一点 = "Để chồng em bớt áp lực hơn."
• 我的作业快写完了 = "Bài tập của con sắp làm xong rồi."
• 爸爸妈妈也快下班了 = "Bố mẹ cũng sắp tan làm rồi."
• 老婆 小菊 = "Vợ à, Tiểu Cúc à."
• 这个盒饭吃不吃的惯啊 = "Không biết hai mẹ con có ăn quen cơm hộp này không?"
• 吃的惯 有菜有肉挺好的 = "Ăn quen mà, có rau có thịt thế này là tốt lắm rồi."
• 香喷喷的 挺好吃 = "Thơm quá, ngon lắm."
• 爸爸妈妈这里好漂亮 真好玩 = "Bố mẹ ơi, nơi này đẹp quá, vui thật đấy."
• 以后爸爸妈妈可以每天陪你一起玩了 = "Sau này bố mẹ có thể chơi với con mỗi ngày rồi."
• 老婆 = "Vợ à."
• 这宿舍有点小 = "Phòng ký túc hơi nhỏ một chút."
• 让你们受苦了 = "Làm mẹ con em phải chịu khổ rồi."
• 老公 = "Chồng à."
• 只要一家人在一起 = "Chỉ cần cả gia đình ở bên nhau."
• 住哪儿都可以 = "Ở đâu cũng được."
• 我也喜欢这里 = "Em cũng thích nơi này."
• 每天都可以和爸爸妈妈在一起 = "Ngày nào cũng được ở cạnh bố mẹ."
• 果然高手还是在农村,收集六个拉环,将它全部摆断。 = "Đúng là cao thủ ở trong dân gian, gom sáu cái khoen lon rồi bẻ thẳng hết ra."
• 在一条木棍上面找出中心位置。 = "Tìm vị trí chính giữa trên một thanh gỗ."
• 用电钻在中间位置打一个洞。 = "Dùng máy khoan khoan một lỗ ở vị trí trung tâm."
• 突然塞一个膨胀螺丝挂钩进去。 = "Sau đó gắn một móc vít nở vào bên trong."
• 拧紧上面的螺母,将所有拉环合理安装在木棍上面。 = "Siết chặt đai ốc phía trên rồi lắp tất cả các khoen lon lên thanh gỗ cho hợp lý."
• 家里的衣架不够用就用它。 = "Nếu trong nhà thiếu móc treo quần áo thì dùng cách này."
• 没有橡皮筋也能自制检疫功,方法超简单,一看就会。 = "Không cần dây thun cũng có thể tự làm cung đồ chơi, cách cực kỳ đơn giản, nhìn là biết làm."
• 实习剩下的筷子别扔,中性比取下防滑套。 = "Đừng vứt đôi đũa còn dư, tháo phần bọc chống trượt ra."
• 先把一根筷子从防滑套的一端穿进去,让筷子坚露出,接着把另一根筷子使劲而挤进去,用力拉下来。 = "Trước tiên luồn một chiếc đũa vào một đầu của lớp chống trượt để lộ phần đầu đũa ra, sau đó nhét mạnh chiếc đũa còn lại vào rồi kéo xuống."
• 然后在这一端也穿上防滑套,再并入一根筷子,最好也是从肩肩这边穿入,兼具自己看着调整就好。 = "Tiếp theo cũng lắp lớp chống trượt ở đầu bên này rồi ghép thêm một chiếc đũa nữa, tốt nhất nên luồn từ cùng một phía để dễ tự điều chỉnh."
• 随便拿根绳子,掐好绑在筷子上的圆环凹槽里当弓弦。 = "Lấy đại một sợi dây rồi buộc cố định vào rãnh tròn trên đũa để làm dây cung."
• 把弦的长度调整合适,另一头也绑紧。 = "Điều chỉnh độ dài dây cung cho phù hợp rồi buộc chặt đầu còn lại."
• 一把超酷的弓就这么轻松搞定了。 = "Vậy là một chiếc cung siêu ngầu đã hoàn thành cực kỳ dễ dàng."
• 不用的这种电源线别扔,这端简断,使用价值立马翻倍。 = "Đừng vứt loại dây nguồn cũ này đi, chỉ cần cắt đầu này ra là giá trị sử dụng tăng gấp đôi."
• 里面有三根线,棕色接火线,蓝色接零线,黄绿双色接地线。 = "Bên trong có ba sợi dây: dây màu nâu nối dây nóng, dây màu xanh dương nối dây nguội, dây vàng xanh nối dây tiếp đất."
• 找一个不用的五孔插座,要接火线,n接零线,中间位置接地线。 = "Lấy một ổ cắm 5 lỗ không dùng nữa, nối dây nóng vào cổng L, dây nguội vào cổng N và dây tiếp đất vào vị trí ở giữa."
• 接好以后就是一个移动插座,哪里需要放哪里?真的太实用了。 = "Sau khi nối xong sẽ thành một ổ cắm di động, cần dùng ở đâu thì mang tới đó, thật sự rất tiện lợi."
• 你是咋进来的? = "Mày vào đây bằng cách nào?"
• 唉,我孩子生了重病,我找路人借钱,就进来了,你咋进来的? = "Haiz, con tao bị bệnh nặng, tao đi vay tiền người qua đường nên mới vào đây, còn mày thì sao?"
• 孩子被人欺负,我跟别人讲道理,讲着讲着把人打成重伤。 = "Con tao bị bắt nạt, tao đi nói lý lẽ với người ta, nói một hồi lại đánh người ta trọng thương."
• 为了孩子,咋咱们好好改造,争取早点出去。 = "Vì con cái, tụi mình cố gắng cải tạo tốt để được ra ngoài sớm."
• 看什么看?没进过house啊! = "Nhìn cái gì? Chưa từng vào tù à!"
• 你为啥进来的? = "Mày vào đây vì chuyện gì?"
• 慧琦,老子专门贩卖小孩的,可这次失手了。 = "Tao chuyên buôn bán trẻ em, nhưng lần này thất bại rồi."
• 不过我还记得那孩子住哪儿,等我出去,绝对饶不了他们全家。 = "Nhưng tao vẫn nhớ nhà thằng bé đó ở đâu, đợi tao ra ngoài nhất định sẽ không tha cho cả nhà nó."
• 领导,我能问问那个人贩子,弄了几个孩子吗? = "Báo cáo cán bộ, tôi có thể hỏi tên buôn người đó đã bắt bao nhiêu đứa trẻ không?"
• 唉,听说有十几个,最小的才刚满月。 = "Nghe nói có hơn chục đứa, đứa nhỏ nhất mới đầy tháng thôi."
• 报告领导 我们房间的监控好像坏了 = "Báo cáo cán bộ, camera trong phòng chúng tôi hình như bị hỏng rồi."
• 要不你们停电检查一下 = "Hay là cắt điện để kiểm tra thử đi."
• 三分钟后停电检修 估计一个小时 = "Ba phút nữa sẽ cắt điện để sửa chữa, dự kiến khoảng một tiếng."
• 怎么停电了 = "Sao lại mất điện rồi?"
• 时间差不多了 快看看 = "Tới giờ rồi, mau vào xem đi."
• 对孩子下手 你禽兽不如 = "Ra tay với trẻ con, mày còn không bằng súc vật."
• 老子最看不起你们这种人贩子 = "Tao khinh nhất loại buôn người như tụi mày."
• 新来的情绪就是不稳定 = "Người mới vào đúng là cảm xúc không ổn định."
• 启动酷孔器在水管的两边开孔。 = "Dùng dụng cụ khoan lỗ để khoan hai bên ống nước."
• 女朋友的鞋带把一条下来穿过两个孔。 = "Lấy một sợi dây giày luồn qua hai lỗ vừa khoan."
• 鞋带的一头打上一个截,堵住一边的孔。 = "Thắt nút một đầu dây giày để chặn một bên lỗ lại."
• 假如我们不小心把物品落进了狭小空间, = "Nếu chẳng may làm rơi đồ vào khe hẹp,"
• 用它来去任何物品都没问题。 = "Thì dùng nó để lấy đồ vật ra sẽ rất dễ dàng."
• 真是个天才啊,把齿轮焊接到钢筋手柄上固定住。 = "Đúng là thiên tài, hàn bánh răng lên tay cầm bằng thép để cố định lại."
• 接着再用切割机把齿轮先平着切割,再挟着切割掉。 = "Tiếp theo dùng máy cắt cắt ngang bánh răng trước rồi cắt chéo tiếp."
• 然后再把链条压到齿轮上再焊接上去。 = "Sau đó ép sợi xích lên bánh răng rồi hàn cố định lại."
• 这样一把可调节多功能齿轮搬手就制作好了。 = "Vậy là đã chế tạo xong một chiếc cờ lê bánh răng đa năng có thể điều chỉnh."
• 可以用来卡住生锈的水管街头,再把链条柴绕上去锁住水管街头。 = "Có thể dùng nó để kẹp chặt đầu ống nước bị gỉ rồi quấn xích quanh để khóa cố định đầu ống."
• 这样转动就可以把水管街头拧出来。 = "Sau đó xoay là có thể vặn đầu ống nước ra."
• 你们觉得这个发明怎么样? = "Mọi người thấy phát minh này thế nào?"
• 真是的天才啊,把报废轮胎切割开制作四料制造机,将焊接制作材料涂入口与轮胎放平。 = "Đúng là thiên tài, cắt lốp xe cũ để chế tạo máy băm nguyên liệu, rồi hàn phần nạp liệu ngang bằng với mặt lốp."
• 接着在周围打上螺丝固定住,然后就可以把切割机电机装上去,用螺丝固定了。 = "Tiếp theo bắt vít cố định xung quanh, sau đó lắp động cơ máy cắt lên và siết chặt bằng vít."
• 再接着把轮胎转过来换一个面,然后再把打磨制作好的切割刀片用螺丝固定装到轮胎里面的切割机上。 = "Sau đó lật lốp sang mặt còn lại rồi dùng vít cố định lưỡi cắt đã mài sẵn vào máy cắt bên trong lốp."
• 再把另一块钢板同样用螺丝固定到轮胎侧面。 = "Tiếp tục dùng vít cố định thêm một tấm thép khác vào bên hông lốp."
• 这样就可以把整个轮胎用螺丝固定到架子上了。 = "Vậy là có thể dùng vít để cố định toàn bộ lốp lên khung máy."
• 最后启动切割机再把草料放进去搅拌切割制作成丝料 = "Cuối cùng khởi động máy cắt rồi cho cỏ vào để băm và cắt thành sợi."
• 你们说这个发明好用吗 = "Mọi người thấy phát minh này có hữu ích không?"
• 毛刷用眉毛的不要扔,脚模机的固定螺母放在上面。 = "Đừng vứt chổi quét sơn cữ, đặt đai ốc cố định của máy lên phía trên."
• 画一下记号,用电钻在上面打两个孔。 = "Đánh dấu vị trí rồi dùng máy khoan điện khoan hai lỗ lên đai ốc."
• 拧两个平头螺丝进孔里面,用它来拆脚模机很方便。 = "Vặn hai con vít đầu dẹt vào các lỗ vừa khoan, dùng nó để tháo máy rất tiện lợi."
• 太厉害了,把气感固定在电钻上再用切割机切割开,接着再把气感帽子焊接到钻头上,然后再把气感两边切割掉。 = "Quá đỉnh rồi, cố định đầu tuýp lên máy khoan rồi dùng máy cắt để cắt ra, sau đó hàn phần đầu tuýp lên mũi khoan và cắt bỏ hai bên."
• 再接着切割掉上面的气感帽子再打磨掉两边,再然后用高温加热气感再放到机油里面脆弧增加硬度。 = "Tiếp theo cắt bỏ phần đầu phía trên rồi mài nhẵn hai bên, sau đó nung nóng bằng nhiệt độ cao và nhúng vào dầu máy để tăng độ cứng."
• 这样就可以把它装到充机钻上用来切割生锈的螺丝。你们说这个发明怎么样? = "Vậy là có thể lắp nó lên máy khoan điện để cắt những con ốc bị gỉ rồi. Mọi người thấy phát minh này thế nào?"
• 在扫揪上放一个硬币,再把扫揪毛按在上面,用胶部将其固定,再剪下来另一端,也用同样的方法固定。 = "Đặt một đồng xu lên đầu chổi quét sơn, sau đó ép phần lông chổi lên trên rồi dùng keo cố định lại, tiếp tục cắt đầu còn lại và làm tương tự."
• 这样就是一个穿针线神器,用它来穿线无论多小的针孔,你一穿就过。 = "Vậy là đã có một dụng cụ xỏ chỉ cực tiện lợi, dù lỗ kim nhỏ đến đâu cũng xỏ qua dễ dàng."
• 太厉害了,螺丝断在发动机里不要怕,可以喷上除锈机在高温加热让除锈机渗透进去。 = "Quá đỉnh rồi, nếu ốc vít bị gãy trong động cơ thì đừng lo, hãy xịt dung dịch tẩy gỉ rồi gia nhiệt để chất tẩy gỉ thấm vào bên trong."
• 接着再放上一枚电片,再把电片焊接上去。 = "Tiếp theo đặt thêm một miếng thép lên trên rồi hàn cố định lại."
• 然后在电片上焊接上一颗螺母,这样就可以用套筒领动螺母再把断掉的螺丝带出来了。 = "Sau đó hàn thêm một con đai ốc lên miếng thép, như vậy có thể dùng đầu tuýp để xoay và tháo phần ốc bị gãy ra ngoài."
• 你们说这个方法好用吗? = "Mọi người thấy cách này có hiệu quả không?"
• 太厉害了,螺丝断在发动机里不要怕,可以喷上除锈机在高温加热让除锈机渗透进去。 = "Quá đỉnh rồi, nếu ốc vít bị gãy trong động cơ thì đừng lo, hãy xịt dung dịch tẩy gỉ rồi gia nhiệt để dung dịch thấm sâu vào bên trong."
• 接着再放上一枚电片,再把电片焊接上去。 = "Tiếp theo đặt một miếng thép lên trên rồi hàn cố định lại."
• 然后在电片上焊接上一颗螺母,这样就可以用套筒领动螺母再把断掉的螺丝带出来了。 = "Sau đó hàn thêm một con đai ốc lên miếng thép, như vậy có thể dùng đầu tuýp để xoay đai ốc và kéo phần ốc bị gãy ra ngoài."
• 你们说这个方法好用吗? = "Mọi người thấy cách này có hiệu quả không?"
• 太厉害了,将轴承内圈套在尖嘴前上焊接切割开, = "Quá đỉnh rồi, lồng vòng trong của vòng bi vào đầu kìm nhọn rồi hàn cố định, sau đó cắt ra."
• 接着再将轴承内圈从中间切割开两半, = "Tiếp theo cắt vòng trong của vòng bi từ giữa thành hai nửa."
• 然后再把尖嘴前的固定转轴和侧面打孔, = "Sau đó khoan lỗ ở trục cố định và phần bên hông của đầu kìm."
• 再然后把侧面打好的孔打上罗文,接着拧上一颗螺丝, = "Tiếp tục taro ren cho lỗ vừa khoan ở bên hông rồi vặn một con vít vào."
• 最后再把控制提手通过刚刚打好的孔安装到尖嘴前上。 = "Cuối cùng lắp tay cầm điều khiển vào đầu kìm thông qua các lỗ vừa khoan."
• 这样一把轴承提取工具就制作好了。 = "Vậy là đã chế tạo xong một dụng cụ tháo vòng bi."
• 把它插在轴承里面拧紧螺丝就可以将卡住的轴承提取出来。 = "Chỉ cần đưa nó vào bên trong vòng bi rồi siết chặt vít là có thể kéo vòng bi bị kẹt ra."
• 你们说这个发明好用吗? = "Mọi người thấy phát minh này có hữu ích không?"
• 太厉害了,将轴承内圈套在尖嘴前上焊接切割开, = "Quá đỉnh rồi, lồng vòng trong của vòng bi vào đầu kìm nhọn rồi hàn cố định, sau đó cắt tách ra."
• 接着再将轴承内圈从中间切割开两半, = "Tiếp theo cắt vòng trong của vòng bi từ giữa thành hai nửa."
• 然后再把尖嘴前的固定转轴和侧面打孔, = "Sau đó khoan lỗ ở trục cố định và phần bên hông của đầu kìm."
• 再然后把侧面打好的孔打上罗文,接着拧上一颗螺丝, = "Tiếp tục taro ren cho lỗ vừa khoan ở bên hông rồi vặn một con vít vào."
• 最后再把控制提手通过刚刚打好的孔安装到尖嘴前上。 = "Cuối cùng lắp tay cầm điều khiển vào đầu kìm thông qua các lỗ vừa khoan."
• 这样一把轴承提取工具就制作好了。 = "Vậy là đã chế tạo xong một dụng cụ tháo vòng bi."
• 把它插在轴承里面拧紧螺丝就可以将卡住的轴承提取出来。 = "Chỉ cần đưa nó vào bên trong vòng bi rồi siết chặt vít là có thể kéo vòng bi bị kẹt ra."
• 你们说这个发明好用吗? = "Mọi người thấy phát minh này có hữu ích không?"
• 真是太有才了, 刚管中垂进一颗轴城,再往另一头汉街上板压。 = "Quả là tài tình, đưa một vòng bi vào bên trong ống thép rồi ép chặt đầu còn lại."
• 接着再往刚管上汉街一根铁棍制作手柄。 = "Tiếp theo hàn thêm một thanh sắt lên ống thép để làm tay cầm."
• 将轴城套在铁棍上面,再抓住铁棍顺时针转动。 = "Lắp vòng bi vào thanh sắt đã chuẩn bị rồi nắm lấy tay cầm và xoay theo chiều kim đồng hồ."
• 罗文就开好了。 = "Vậy là ren đã được tạo xong."
• 摩托车抽汽油妙照,洗手液用完后,剪断抽水泵的吸管,用1米长的软管套住。 = "Mẹo hút xăng xe máy: sau khi dùng hết chai nước rửa tay, cắt lấy ống hút của bơm rồi nối với một đoạn ống mềm dài khoảng 1 mét."
• 把软管放进摩托车油箱,挤压挤下抽水泵引水出来,拔开吸管就能快速抽油。 = "Đưa ống mềm vào bình xăng xe máy, bóp đầu bơm vài lần để tạo lực hút, sau đó tháo ống ra là có thể hút xăng ra nhanh chóng."
• 太厉害了,将水管接头焊接到钢板上,在焊接上一颗螺丝,接着把铁圈焊接到螺帽上。 = "Quá đỉnh rồi, hàn đầu nối ống nước lên tấm thép, sau đó hàn thêm một con vít cố định rồi tiếp tục hàn vòng sắt lên đai ốc."

• 然后用螺丝将钢板固定到一起在焊接上钢圈,这样装上水笔一把原规就制作好了,真是太聪明了。 = "Sau đó dùng vít để cố định các tấm thép lại với nhau rồi hàn thêm vòng thép. Cuối cùng lắp bút lấy dấu vào là hoàn thành một chiếc compa tự chế, đúng là quá thông minh."
• 用针筒自制电焊机,较成开始,铁钉放在主筒中间用钳子敲击进去。 = "Tự chế máy hàn điện từ ống tiêm: đầu tiên đặt đinh sắt vào giữa thân chính rồi dùng kìm gõ cố định vào bên trong."

• 塞主筒进针筒,用胶水固定住,突然放两个电池进针筒里面。 = "Đưa phần thân chính vào trong ống tiêm rồi cố định bằng keo, sau đó đặt hai viên pin vào bên trong ống."

• 红色电线穿过活塞固定住一个罗母,另外一边也固定一个罗母。 = "Luồn dây điện màu đỏ qua pít-tông và cố định một đầu bằng đai ốc, đầu còn lại cũng cố định tương tự."

• 两个活塞安装回针筒上面,一个针筒电焊机就制作好了。 = "Lắp lại hai pít-tông vào ống tiêm là hoàn thành mô hình máy hàn điện tự chế từ ống tiêm."
• 真是太惊人了,网络刷上安装一个手柄,再从中间切断。 = "Quá sáng tạo rồi, gắn một tay cầm lên khung rồi cắt đôi ở chính giữa."

• 网上面拧上一个短罗母和一个打孔的家常罗母,再插接一颗自公罗斯丁。 = "Tiếp theo vặn vào một đai ốc ngắn và một đai ốc đã khoan lỗ, sau đó lắp thêm một con vít ren."
• 活了半辈子才知道原来酱油醋的拉环取下来不是让你扔掉的。 = "Sống nửa đời người mới biết rằng vòng nhựa trên nắp chai nước tương và giấm không phải tháo ra để vứt đi."

• 正确的用法是你把它反转过来,然后回到原来的位置。 = "Cách dùng đúng là lật ngược vòng nhựa đó lại rồi gắn trở về vị trí ban đầu."
• 太厉害了,在钢管上用尺子画出跟视频一样的角度切开。 = "Quá đỉnh rồi, dùng thước đánh dấu góc cắt trên ống thép giống như trong video rồi tiến hành cắt."

• 然后再把钢管拼接焊接,这样一个完美的90度角焊接就完成了。 = "Sau đó ghép các đoạn ống lại và hàn cố định, vậy là hoàn thành một mối hàn góc 90 độ rất chuẩn."
• 真是个天才啊,将发动机七刚活塞改造成台式切割机,先将七刚活塞拆卸分开来,然后再把七刚活塞打孔。 = "Đúng là thiên tài, biến piston động cơ thành máy cắt bàn. Đầu tiên tháo rời piston rồi khoan các lỗ cần thiết."

• 接着将七刚活塞用螺丝固定到钢板上,再将活塞摇壁安装上去,再把切割机安装到摇壁上,然后拧上螺丝安装固定弹簧。 = "Tiếp theo dùng bu lông cố định piston lên tấm thép, lắp tay đòn rồi gắn máy cắt vào đó, sau cùng bắt vít và lắp lò xo cố định."

• 接着把轴承外圈拧到切割机上当做控制把手。 = "Sau đó lắp vòng ngoài của vòng bi lên máy cắt để làm tay cầm điều khiển."

• 最后再将制作好的台式切割机启动切割钢筋。 = "Cuối cùng khởi động chiếc máy cắt bàn vừa chế tạo để cắt thanh thép."

• 你们觉得这个便费维保的制作实用吗? = "Mọi người thấy ý tưởng tự chế này có thực dụng không?"
• 你们说这强度怎么样? = "Mọi người thấy độ bền của cách làm này thế nào?"
• 这样倒出来的酱油醋不会喷到处都是。 = "Làm như vậy khi rót nước tương hoặc giấm sẽ không bị bắn tung tóe ra xung quanh."
• 最后将罗母顶住木板,垫子安装套筒,转动罗斯丁横向移动,就可以进行切割或者开槽。真是脑洞大开。 = "Cuối cùng dùng đai ốc tì lên tấm gỗ, lắp thêm ống đệm rồi xoay vít để dịch chuyển ngang, từ đó có thể thực hiện cắt hoặc tạo rãnh. Đúng là ý tưởng quá độc đáo."
• 在油桶的底部粘一块垫工胶布,再把一根钢钉砸到胶布上,然后将棉签剪一个尖,再拔出钉子,用棉签堵住小孔。 = "Dán một miếng cao su hoặc keo chống thấm vào đáy can nhựa, sau đó đóng một chiếc đinh xuyên qua. Tiếp theo cắt vát đầu tăm bông, rút đinh ra rồi dùng tăm bông bịt vào lỗ nhỏ vừa tạo."
• 长期出门时,这样家里种的花就不会干汗了。 = "Khi phải đi xa trong thời gian dài, cách này sẽ giúp cây hoa ở nhà luôn được cấp nước và không bị khô héo."
• 注意看,从这里剪到第四条线,再从第三条线这里剪开,然后再把上面一层按第一条线修剪掉。 = "Chú ý nhé, cắt từ vị trí này đến đường thứ tư, sau đó cắt theo đường thứ ba. Tiếp theo cắt bỏ phần lớp trên theo đường thứ nhất."

• 再把下面一层宽出来的部分对折,再把上面一层包起来直接在上面压一条线,压好线之后打开铺平,把这个脚履顺。 = "Gấp đôi phần dư của lớp dưới, bọc lớp trên lại rồi may một đường cố định. Sau khi may xong thì mở ra và vuốt phẳng các nếp gấp."

• 再把多出来的部分对折,再把这个脚包起来直接在上面压一条线打开来挥针。 = "Tiếp tục gấp đôi phần thừa còn lại, bọc góc vải vào trong rồi may thêm một đường cố định, sau đó mở ra để chỉnh lại."
• 真是个天才啊。 = "Đúng là một ý tưởng quá thông minh."

• 楼梯辅手采用不锈钢连接处,汉阶有缝隙,还特别容易开裂,不要怕。 = "Khi lắp tay vịn cầu thang bằng ống inox, nếu chỗ tiếp giáp với bậc thang có khe hở hoặc dễ bị nứt thì cũng đừng lo."

• 可以按照视频中先给钢管画线做记号。 = "Hãy làm theo video, trước tiên dùng bút đánh dấu các đường cắt trên ống thép."

• 接着再把画线做记号的钢管切割开。 = "Sau đó cắt ống thép theo các đường đã đánh dấu."

• 这样就可以完美无缝拼接楼梯辅手在汉阶牢固。 = "Như vậy có thể lắp ghép tay vịn cầu thang một cách khít đẹp, chắc chắn và gần như không thấy mối nối."

• 你们觉得这个汉阶小技巧实用吗? = "Mọi người thấy mẹo thi công cầu thang này có hữu ích không?"
• 不穿的T恤衫这样对折,从袖子的中间裁剪下来,下摆这里也裁剪掉。 = "Gấp đôi chiếc áo thun cũ không còn mặc nữa, sau đó cắt từ giữa phần tay áo và cắt bỏ phần gấu áo."

• 再给它打开铺平,从中间的位置拎起来,下摆的位置就是腰头。 = "Mở áo ra và trải phẳng, nhấc phần giữa lên, lúc này phần gấu áo sẽ trở thành cạp quần."

• 袖子这里缝起来就是当缝,再把裤脚往里面折个边。 = "May kín phần tay áo để tạo đường may đáy quần, sau đó gấp mép ống quần vào trong và may cố định."
• 太厉害了,遇到钢板拼接角度过大时,不要怕。 = "Quá đỉnh rồi, khi gặp trường hợp các tấm thép cần ghép với nhau ở góc quá lớn thì cũng đừng lo."

• 可以把其中一块钢板两头画线做记号标记处要切割的区域。 = "Hãy đánh dấu bằng cách vẽ các đường cắt ở hai đầu của một tấm thép để xác định phần cần gia công."

• 接着再把标记的区域切割开。 = "Sau đó dùng máy cắt để cắt theo các vị trí đã đánh dấu."

• 这样就可以把钢管捏扁插到另外两条钢管里面,打上钢钉固定在一起了。 = "Tiếp theo ép dẹt đầu ống thép rồi luồn vào giữa hai ống thép còn lại, sau đó dùng chốt hoặc đinh thép cố định lại với nhau."

• 你们说这个方法实用吗? = "Mọi người thấy phương pháp này có thực dụng không?"
• 腰头再包一条相机,一条好看又舒服的短裤就做好了。 = "May thêm một lớp bo ở phần cạp, vậy là đã hoàn thành một chiếc quần short vừa đẹp vừa thoải mái."
• 再把这个脚放到正面来看效果,一个官边立体直角就做好了。做各种收纳包包都可以用这种方法。 = "Lật sang mặt trước để xem thành quả, vậy là đã hoàn thiện một góc vuông 3D rất gọn đẹp. Phương pháp này có thể áp dụng để may nhiều loại túi đựng và túi lưu trữ khác nhau."
• 我们在洗这种漏网石的时候用水冲 = "Khi rửa loại rổ lưới này bằng nước,"
• 冲半天也不干净 = "xả mãi vẫn không sạch,"
• 老是一件 = "cặn bẩn cứ bám lại."
• 这个时候我们就拿一个勺子 = "Lúc này chỉ cần lấy một chiếc thìa,"
• 看这样 = "Nhìn này,"
• 它的面积就大了 = "diện tích tiếp xúc với nước sẽ lớn hơn,"
• 冲得也快了 = "nên rửa sạch nhanh hơn nhiều."
• 不穿的短袖对着铺平 = "Trải phẳng chiếc áo thun cũ ra."
• 抵当1.5寸 = "Chừa lại khoảng 4cm,"
• 转45度脚才下来 = "rồi cắt theo góc 45 độ."
• 展开铺平 = "Mở ra và trải phẳng."
• 不会画线的 = "Nếu không biết vẽ đường cong,"
• 拿个祖传的盘子扣在上面画弧线 = "hãy dùng một chiếc đĩa để làm mẫu rồi vẽ theo."
• 按照画线捡下来 = "Cắt theo các đường đã đánh dấu."
• 你学会了吗? = "Bạn đã học được cách làm chưa?"
• 注意看 = "Chú ý nhé."
• 把裁编两头对齐 = "Canh hai đầu mép vải cho thẳng hàng."
• 再把相机重新一公分包起来 = "Sau đó gấp mép thêm khoảng 1cm để may."
• 注意下面一层要长出来0.7到0.8左右 = "Lớp vải phía dưới nên dài hơn khoảng 0,7 đến 0,8cm."
• 这样相机腰头一次性拼接好 = "Như vậy phần cạp chun sẽ được ráp hoàn chỉnh trong một lần."
• 记得打个来回针 = "Nhớ may lùi mũi để cố định đường chỉ."
• 再把腰头打开 = "Sau đó mở phần cạp ra."
• 你看这个腰头有高低层 = "Bạn sẽ thấy phần cạp có độ cao thấp giữa các lớp."
• 这样拼接的相机不管你拉不拉线它都不会卷 = "May như vậy thì dù kéo chun hay không, mép vải cũng không bị cuộn."
• 然后把裙子或者裤子的大片放在上面 = "Tiếp theo đặt thân váy hoặc thân quần lên trên."
• 缝对缝对齐 = "Canh các đường may cho thật khớp."
• 注意包相机的腰头一直要保持高低缝 = "Khi ráp cạp, luôn giữ đúng độ cao thấp của các lớp vải."
• 下面一层要宽出来0.7 0.8左右 = "Lớp dưới vẫn phải rộng hơn khoảng 0,7 đến 0,8cm."
• 大片的直口 = "Đối với mép thẳng của thân chính,"
• 拼到上面一层的直口平齐 = "hãy canh bằng với mép thẳng của lớp trên cùng."
• 这样三合一一次先把相机包上去 = "Như vậy có thể ráp ba lớp cùng lúc."
• 包相机我们就用单边压脚 = "Khi ráp chun, dùng chân vịt một bên."
• 这样就不会踩到里面的相机 = "Như vậy sẽ không may đè lên phần chun bên trong."
• 相机包好之后把下面一层宽出来的直口碗里面折 = "Sau khi ráp xong, gấp phần mép thừa của lớp dưới vào trong."
• 再换一个反高的压脚 = "Sau đó đổi sang chân vịt phù hợp."
• 直接在上面压一个0.1的明线 = "May một đường chỉ nổi cách mép khoảng 0,1cm."
• 一边折一边压线 = "Vừa gấp mép vừa may theo đường chỉ."
• 注意直口要折均匀 = "Lưu ý phải gấp mép thật đều."
• 这样压好的线反面是0.1正面是0.6 = "Sau khi may xong, mặt trong cách mép 0,1cm, mặt ngoài khoảng 0,6cm."
• 这样包好的相机不用烤边机 = "Cách làm này không cần máy vắt sổ."
• 也不用包边条 = "Cũng không cần dùng băng viền."
• 里面是官边的 = "Mặt trong vẫn rất gọn gàng."
• 不累呀 = "Thao tác cũng không phức tạp."
• 外面也好看 = "Mặt ngoài nhìn cũng rất đẹp."
• 把家里的叉子放在火上面烧红 = "Hơ nóng một chiếc nĩa trên lửa cho đỏ lên."
• 趁热赶紧在瓶盖上捅出八个小孔 = "Khi còn nóng, nhanh tay đục 8 lỗ nhỏ trên nắp chai."
• 再割下来瓶口 = "Sau đó cắt phần cổ chai ra."
• 扣在水杯上 = "Úp phần cổ chai lên miệng cốc."
• 然后再捅一圈小孔 = "Rồi đục thêm một vòng lỗ nhỏ xung quanh."
• 这样做出来的洗菜盆过滤网 = "Vậy là đã có một chiếc lưới lọc rau củ tự chế."
• 太好用了 = "Dùng cực kỳ tiện lợi."
</glossary_idioms>

<pronoun_consistency>
QUAN TRỌNG: Drama nhiều nhân vật → xưng hô NHẤT QUÁN trong cùng 1 mạch hội thoại.
- Sếp ↔ nhân viên: "tôi/anh/em" (sếp gọi cấp dưới = "em" hoặc "tụi bay" tùy mức thân)
- Đồng nghiệp với nhau: "tao/mày" hoặc "tôi/cậu" (đám trẻ TikTok hay dùng)
- Tránh chuyển đổi giữa "tôi/anh/cậu/ngươi/các người" trong cùng đoạn
- "ngươi" CHỈ dùng khi không có lựa chọn khác (cổ trang) — KHÔNG dùng cho drama hiện đại
</pronoun_consistency>
"""
            vi_examples = """
<good_examples>

[PHIM TÀI LIỆU ĐỘNG VẬT — narrator trang trọng]

INPUT  ["1. [budget=3.0s] 在海藻森林的庇护下"]
OUTPUT ["Dưới sự che chở của rừng tảo bẹ"]

INPUT  ["1. [budget=4.0s] 海獭妈妈才得以养活三个孩子"]
OUTPUT ["Rái cá mẹ mới nuôi nổi ba đứa con"]
(海獭 = rái cá biển, KHÔNG phải "tháp nước"/"sếp")

INPUT  ["1. [budget=2.5s] 它们以海胆为食"]
OUTPUT ["Chúng săn nhím biển làm mồi"]
(narrator dùng "chúng", không "anh ấy")

INPUT  ["1. [budget=3.0s] 阿留申群岛的这只小海獭"]
OUTPUT ["Chú rái cá nhỏ trên quần đảo Aleutian"]
(KHÔNG dịch tên đảo thành tên khác như Schiedam)

INPUT  ["1. [budget=2.0s] 幼崽完全依赖母亲"]
OUTPUT ["Con non phụ thuộc hoàn toàn vào mẹ"]
(幼崽 = con non, KHÔNG "cậu bé")

INPUT  ["1. [budget=2.5s] 母海獭一刻不停地照顾"]
OUTPUT ["Rái cá mẹ chăm sóc không ngừng"]


[DRAMA TIKTOK MÈO AI — đối thoại đời thường]

INPUT  ["1. [budget=2.4s] 老板,火干完了,可以结账了吗?"]
OUTPUT ["Sếp, em xong rồi, tính tiền nhé"]

INPUT  ["1. [budget=4.0s] 不愧是鼠猫的,头脑真灵活,下次还带着你"]
OUTPUT ["Quả là Chuột với Mèo, đầu óc nhanh thật"]

INPUT  ["1. [budget=1.6s] 我们干活,凭啥给他茶水费?"]
OUTPUT ["Tụi mình làm, sao phải lót tay nó"]

INPUT  ["1. [budget=1.3s] 哈哈,这样才对"]
OUTPUT ["Haha, vậy mới đúng"]

INPUT  ["1. [budget=1.0s] 你晚上过来"]
OUTPUT ["Tối em qua nhé"]

INPUT  ["1. [budget=1.0s] 他俩很不懂事"]
OUTPUT ["Hai đứa không biết điều"]

INPUT  ["1. [budget=0.8s] 咋回事"]
OUTPUT ["Sao thế"]
</good_examples>

<bad_examples>
[PHIM ĐỘNG VẬT — TUYỆT ĐỐI tránh]
❌ "海獭" → "tháp nước Oyster"  (sai nghiêm trọng — là RÁI CÁ BIỂN)
❌ "海獭" → "sếp" / "ông chủ"  (chỉ dùng cho drama mèo, KHÔNG cho phim động vật)
❌ "海藻森林" → "rừng biển xanh"  (đúng phải là "rừng tảo bẹ")
❌ "阿留申群岛" → "Schiedam" / "Scherling Islands"  (đó là Aleutian)
❌ "幼崽" → "cậu bé"  (con non động vật, KHÔNG phải người)
❌ "母海獭" → "mẹ nước cao" / "mẹ sếp"  (rái cá mẹ)
❌ "三个孩子" trong phim động vật → "ba đứa trẻ"  (đúng: "ba đứa con")

[DRAMA — TUYỆT ĐỐI tránh]
❌ "老板,火干完了" → "Sếp, lửa tắt cả rồi"  (火干完了 = xong việc)
❌ "你说给了一千" → "Ngươi nói đưa một nghìn"  (KHÔNG "ngươi" trong drama hiện đại)
❌ "你晚上过来" → "Em tối qua đây"  (晚上 = tối nay)
❌ "他俩很不懂事" → "Họ rất vô lý"  (不懂事 = không biết điều)
❌ "嗯嗯" → "Vâng vâng tôi hiểu rồi"  (interjection — phải trả "")
</bad_examples>
"""

    prompt = f"""<role>
Bạn là dịch giả {src_name}→{tgt_name} chuyên dịch phụ đề video TikTok, phim ngắn, vlog, phim tài liệu. Bạn dịch tự nhiên như người bản ngữ viết — KHÔNG phải dịch máy. Ưu tiên: NGHĨA đúng > tự NHIÊN > NGẮN gọn vừa khung subtitle.
</role>

<task>
Dịch CHÍNH XÁC {len(batch_segments)} câu phụ đề từ {src_name} sang {tgt_name}.
Trả về JSON array đúng {len(batch_segments)} phần tử, không giải thích.
</task>

<workflow>
Trước khi dịch, hãy tự nhẩm trong đầu (không in ra):
1. Đọc HẾT {len(batch_segments)} câu + ngữ cảnh phía trên → đoán chủ đề video.
2. Nhận diện speaker và xưng hô nhất quán xuyên suốt batch.
3. Đánh dấu tên riêng, số tiền, idiom.
4. Mỗi câu: ƯU TIÊN giữ Ý CHÍNH, dịch CỰC NGẮN. Cắt bỏ TẤT CẢ chi tiết phụ.
5. Mỗi câu xem `[budget=Xs]` ở cuối câu gốc — đó là giây tối đa khi đọc thành lời.
   Quy tắc: 1 giây = ~3 từ tiếng Việt. Phải dịch không quá `budget × 3` từ.
</workflow>

<core_rules>
1. KHÔNG bịa, KHÔNG suy diễn. Câu mơ hồ → dịch trung tính, không thêm thông tin.
2. KHÔNG thêm chủ ngữ mà câu gốc đã bỏ.{vi_specific_rules}
4. KHÔNG dịch chữ-một idiom. Tìm idiom tương đương hoặc paraphrase.
5. Câu chỉ là interjection (嗯/啊/哦/uh/um/...) hoặc nguyên âm lặp → trả "" (rỗng).
6. Giữ TONE gốc: hỏi vẫn là hỏi, mắng vẫn là mắng, đùa vẫn là đùa.
7. ✦ NGẮN LÀ TỐT NHẤT. Nguyên tắc subtitle: nói được trong thời lượng cảnh.
   - budget 1s → tối đa 3 từ
   - budget 2s → tối đa 6 từ
   - budget 3s → tối đa 9 từ
   - budget ≥4s → tối đa 12 từ
   QUAN TRỌNG: nếu vượt budget, cảnh phim và voice sẽ LỆCH NHAU. Thà bỏ chi tiết
   phụ còn hơn để câu dài. Người xem ưu tiên thấy mắt nhân vật mở miệng khớp lời.
   Câu gốc dài mà budget ngắn → CHỈ giữ ý cốt lõi, BỎ trợ từ, BỎ chi tiết phụ.
8. KHÔNG kết thúc bằng dấu "." nếu câu gốc không có dấu kết câu.
9. KHÔNG mở đầu bằng "Đây là...", "Tôi muốn nói...", "Sự thật là..." — bỏ filler.
10. RÚT GỌN từ thừa: "rất là" → "rất", "cho rồi" → "rồi", "đã từng" → "từng".
    "anh ấy" → "anh", "cô ấy" → "cô" khi context đã rõ.
</core_rules>
{vi_specific_glossary}{vi_examples}
{context_block}<input>
{numbered}
</input>

<output_format>
JSON array đúng {len(batch_segments)} phần tử, theo thứ tự, không markdown, không số thứ tự, không giải thích:
["...", "...", "..."]
</output_format>"""

    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Bạn là dịch giả {src_name}→{tgt_name} cho phụ đề video. "
                    "Quy tắc tuyệt đối: (1) đúng nghĩa, không bịa; "
                    f"(2) {tgt_name} tự nhiên như người bản ngữ; "
                    "(3) trả về DUY NHẤT 1 JSON array với đúng số phần tử yêu cầu; "
                    "(4) KHÔNG markdown, KHÔNG giải thích, KHÔNG số thứ tự, KHÔNG comment. "
                    "Câu mơ hồ → dịch trung tính. Câu chỉ là interjection → trả \"\"."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.15,
        top_p=0.9,
    )

    raw_text = response.choices[0].message.content.strip()
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    translated = []

    # Thử parse là JSON array hoàn chỉnh
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, list):
            translated = [str(x) for x in parsed]
    except Exception:
        pass

    # Fallback: mỗi dòng có thể là 1 array hoặc 1 string
    if not translated:
        for line in raw_text.splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                arr = json.loads(line)
                if isinstance(arr, list):
                    translated.extend(str(x) for x in arr)
                elif isinstance(arr, str):
                    translated.append(arr)
            except Exception:
                match = re.search(r'"(.+?)"', line)
                if match:
                    translated.append(match.group(1))

    # Pad/truncate đúng size
    if len(translated) < len(batch_segments):
        translated.extend([""] * (len(batch_segments) - len(translated)))
    return translated[: len(batch_segments)]


def translate_segments_with_ollama(segments, source_lang="zh", target_lang="vi"):
    """Chia segments thành batch nhỏ để Ollama không bị cắt response.
    Truyền 2 câu vừa dịch xong làm context cho batch sau, để giữ mạch văn.

    source_lang/target_lang: mã ngôn ngữ ngắn (zh, vi, en, ja, ko...).
    Nếu source == target → trả nguyên text gốc, không gọi LLM.
    """
    # Cùng ngôn ngữ → không cần dịch, chỉ cần dùng text gốc cho TTS
    if source_lang == target_lang:
        print(f"[Translate] source==target=={source_lang}, skip LLM, using original text")
        return [
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": clean_text(seg.get("text", "")),
                "original_text": seg.get("text", ""),
            }
            for seg in segments
        ]

    # Chia câu dài thành các phần nhỏ hơn trước khi dịch,
    # đặc biệt hữu ích khi Whisper trả về một segment dài không ngắt câu.
    segments = split_segments_for_translation(segments)

    # Đánh dấu trước câu nào là Whisper hallucination → không gửi đi dịch
    skip_flags = [is_whisper_hallucination(seg.get("text", "")) for seg in segments]
    skipped_count = sum(skip_flags)

    # Chỉ gửi các segment hợp lệ sang Ollama
    valid_segments = [seg for seg, skip in zip(segments, skip_flags) if not skip]
    print(f"[Translate] {len(segments)} segments → skip {skipped_count} hallucination → translate {len(valid_segments)} ({source_lang}→{target_lang})")

    valid_translations = []
    total_batches = (len(valid_segments) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        batch = valid_segments[start : start + BATCH_SIZE]

        # Lấy 2 cặp (src, tgt) cuối cùng từ batch trước làm ngữ cảnh
        ctx_pairs = []
        if valid_translations:
            tail_src = [s["text"].strip() for s in valid_segments[max(0, start - CONTEXT_TAIL):start]]
            tail_tgt = [t for t in valid_translations[-CONTEXT_TAIL:]]
            ctx_pairs = list(zip(tail_src, tail_tgt))

        batch_result = []
        for attempt in range(2):
            try:
                batch_result = _translate_one_batch(
                    batch,
                    prev_context_pairs=ctx_pairs,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )
                valid_count = sum(
                    1 for t in batch_result
                    if t and not is_bad_translation(clean_text(t), target_lang=target_lang)
                )
                if valid_count >= max(1, len(batch) // 2):
                    break
                print(f"[Translate] batch {batch_idx+1}/{total_batches} attempt {attempt+1}: only {valid_count}/{len(batch)} valid, retrying...")
            except Exception as e:
                print(f"[Translate] batch {batch_idx+1} attempt {attempt+1} error: {e}")
                batch_result = [""] * len(batch)

        valid_translations.extend(batch_result)
        print(f"[Translate] batch {batch_idx+1}/{total_batches} done")

    print(f"[Translate] got {len(valid_translations)} / {len(valid_segments)} translations")

    # Map kết quả ngược lại theo thứ tự gốc
    translated_segments = []
    valid_iter = iter(valid_translations)

    for seg, skip in zip(segments, skip_flags):
        if skip:
            text = ""  # câu này sẽ bị bỏ qua khi TTS
        else:
            raw = next(valid_iter, "")
            text = clean_text(raw)

        if is_bad_translation(text, target_lang=target_lang):
            # Câu hallucination → để rỗng để TTS skip
            # Câu dịch fail thật → fallback ngắn
            text = "" if skip else FALLBACK_TEXT

        # Safety cap: LLM đã thấy [budget=Xs] trong prompt nên dịch tự rút.
        # Đây chỉ là chốt chặn nếu LLM trả câu quá dài (>18 từ).
        # KHÔNG cắt theo budget chính xác — sẽ làm mất chữ cuối câu hợp lệ.
        text = shorten_words(text, max_words=18)

        translated_segments.append(
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": text,
                "original_text": seg.get("text", ""),
            }
        )

    return translated_segments


def ass_time(seconds):
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def hex_to_ass_color(hex_color, alpha="00"):
    """First definition — kept for backward compat, delegates to the canonical one below."""
    val = str(hex_color or "#FFFFFF").strip()

    if val.lower() == "transparent":
        return "&HFF000000"

    val = val.replace("#", "")
    if len(val) != 6:
        val = "FFFFFF"

    r = val[0:2]
    g = val[2:4]
    b = val[4:6]

    return f"&H{alpha}{b}{g}{r}"


def safe_ass_text(text):
    text = str(text or "")
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text

def get_video_dimensions_ffprobe(video_path):
    """Dùng ffprobe để lấy đúng display dimensions, xử lý rotation metadata."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
 
        for stream in data.get("streams", []):
            if stream.get("codec_type") != "video":
                continue
 
            w = stream.get("width", 0)
            h = stream.get("height", 0)
 
            # Đọc rotation từ side_data_list (ffprobe mới)
            rotation = 0
            for sd in stream.get("side_data_list", []):
                if sd.get("side_data_type") == "Display Matrix":
                    rotation = abs(int(sd.get("rotation", 0)))
 
            # Fallback: đọc từ tags (ffprobe cũ)
            if rotation == 0:
                tags = stream.get("tags", {})
                rotation = abs(int(tags.get("rotate", 0)))
 
            # Swap width/height nếu video bị xoay 90° hoặc 270°
            if rotation in (90, 270):
                w, h = h, w
 
            if w > 0 and h > 0:
                print(f"[ffprobe] dims: {w}x{h}, rotation={rotation}°")
                return int(w), int(h)
 
    except Exception as e:
        print(f"[ffprobe] Error: {e}, fallback to VideoFileClip")
 
    clip = VideoFileClip(video_path)
    w, h = int(clip.w), int(clip.h)
    clip.close()
    print(f"[fallback] dims: {w}x{h}")
    return w, h
 
 
def ass_time(seconds):
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"
 
 
def hex_to_ass_color(hex_color, alpha="00"):
    """
    Convert "#RRGGBB" → "&HAABBGGRR" (ASS format).
    "transparent" → fully transparent (&HFF000000).
    """
    val = str(hex_color or "#FFFFFF").strip()

    if val.lower() == "transparent":
        return "&HFF000000"

    val = val.replace("#", "")
    if len(val) != 6:
        val = "FFFFFF"

    r = val[0:2]
    g = val[2:4]
    b = val[4:6]
    return f"&H{alpha}{b}{g}{r}"
 
 
def safe_ass_text(text):
    text = str(text or "")
    text = text.replace("{", "").replace("}", "")
    # Convert newline thật (\n) → ASS newline (\N) — hỗ trợ xuống dòng từ FE
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", "\\N")   # ← đã có, giữ nguyên
    return text
 
 
def _ass_alpha_from_opacity(opacity):
    """Convert FE opacity 0..1 → ASS alpha hex (00 = đặc, FF = trong suốt)."""
    try:
        op = float(opacity)
    except Exception:
        op = 1.0
    op = max(0.0, min(1.0, op))
    alpha = int(round((1.0 - op) * 255))
    return f"{alpha:02X}"


# Cache font đã load để khỏi mở file mỗi câu
_FONT_CACHE = {}


def _get_pil_font(font_family, font_size):
    """Lazy-load Pillow font. Fallback nếu font_family không tìm được."""
    key = (font_family, font_size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    from PIL import ImageFont

    normalized = str(font_family or "").strip()
    font_candidates = []

    # Common bundled font families in fonts/ folder
    family_map = {
        "roboto": "Roboto-Regular.ttf",
        "open sans": "OpenSans-Regular.ttf",
        "opensans": "OpenSans-Regular.ttf",
        "montserrat": "Montserrat-Regular.ttf",
        "bevietnampro": "BeVietnamPro-Regular.ttf",
        "be vietnam pro": "BeVietnamPro-Regular.ttf",
        "arial": "arial.ttf",
        "times new roman": "Times New Roman.ttf",
        "timesnewroman": "Times New Roman.ttf",
        "times": "Times.ttf",
    }

    file_candidate = family_map.get(normalized.lower())
    if file_candidate:
        font_candidates.append(os.path.join(FONT_DIR, file_candidate))
        font_candidates.append(file_candidate)
    else:
        # Try exact font family or .ttf variants
        font_candidates.append(normalized)
        font_candidates.append(f"{normalized}.ttf")
        if normalized.lower() == "times new roman":
            font_candidates.append("Times.ttf")
            font_candidates.append("Times New Roman.ttf")
            font_candidates.append("Times New Roman")

    # Search bundled fonts directory for a matching base name
    if os.path.isdir(FONT_DIR):
        for entry in os.listdir(FONT_DIR):
            if entry.lower().startswith(normalized.lower().replace(" ", "")) and entry.lower().endswith(".ttf"):
                font_candidates.insert(0, os.path.join(FONT_DIR, entry))

    # Fallback candidates
    font_candidates.extend([
        "arial.ttf",
        "Arial.ttf",
    ])

    font = None
    for name in font_candidates:
        try:
            font = ImageFont.truetype(name, font_size)
            break
        except Exception:
            continue

    if font is None:
        font = ImageFont.load_default()

    _FONT_CACHE[key] = font
    return font


def _measure_text_width(text, font):
    """Đo width pixel của 1 dòng text với font đã cho."""
    try:
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0]
    except Exception:
        return font.getlength(text) if hasattr(font, "getlength") else len(text) * 8


def wrap_text_to_box(text, font, max_width):
    """
    Wrap 1 dòng text thành nhiều dòng vừa max_width pixel.
    - Word-aware: cắt theo space khi có thể.
    - Char-aware: từ dài hơn box → cắt giữa từ.
    Trả về list[str] — mỗi phần tử là 1 dòng.
    """
    text = str(text or "").strip()
    if not text:
        return [""]

    # Đảm bảo có khoảng đệm bên trong box (8% mỗi bên)
    inner_width = max(1, int(max_width * 0.92))

    # Tách theo whitespace nhưng giữ nguyên ký tự (CJK không có space)
    has_space = any(ch.isspace() for ch in text)
    if has_space:
        words = text.split()
    else:
        # CJK: mỗi ký tự là 1 "word"
        words = list(text)

    lines = []
    current = ""
    sep = " " if has_space else ""

    for w in words:
        candidate = (current + sep + w) if current else w
        if _measure_text_width(candidate, font) <= inner_width:
            current = candidate
            continue

        # Push current line nếu có
        if current:
            lines.append(current)
            current = ""

        # Nếu 1 từ đơn lẻ vẫn dài hơn box → cắt giữa từ theo character
        if _measure_text_width(w, font) > inner_width:
            buf = ""
            for ch in w:
                test = buf + ch
                if _measure_text_width(test, font) <= inner_width:
                    buf = test
                else:
                    if buf:
                        lines.append(buf)
                    buf = ch
            current = buf
        else:
            current = w

    if current:
        lines.append(current)

    return lines or [""]


def create_ass_file(video_path, subtitles, ass_path, preview_w=0, preview_h=0,
                    subtitle_config=None):
    """
    Tạo ASS subtitle file khớp PIXEL-PERFECT với preview FE.

    Mỗi subtitle render bằng 2 Dialogue lines:
      - Layer 0: vẽ background box bằng drawing primitive (rectangle) — kích thước
        cố định = boxWidth × boxHeight px (đã scale từ preview → video).
      - Layer 1: vẽ text trong box với fontSize chính xác.

    Cách này đảm bảo position + size khớp tuyệt đối với những gì user đã kéo/resize
    trên FE preview, không bị ASS auto-fit text như BorderStyle=3.

    subtitle_config: dict từ FE chứa fontSize, boxWidth, boxHeight, bgColor,
                     textColor, bgOpacity, showBg, fontFamily, borderRadius
    """
    cfg = subtitle_config or {}

    video_w, video_h = get_video_dimensions_ffprobe(video_path)

    # Scale từ preview pixel → video pixel
    if preview_w > 0 and preview_h > 0:
        scale_x = video_w / preview_w
        scale_y = video_h / preview_h
        print(f"[ASS] preview={preview_w}x{preview_h}, video={video_w}x{video_h}, scale=({scale_x:.3f}, {scale_y:.3f})")
    else:
        # FE quên gửi previewWidth/Height → heuristic: giả định preview rộng ~1/3 video
        # (typical: video 1080x1920, preview ~360x640 trong app)
        # Nếu user bỏ qua step gửi preview, ít nhất subtitle còn nhìn thấy được
        is_portrait = video_h > video_w
        guess_pw = video_w / 3 if is_portrait else video_w / 3
        guess_ph = video_h / 3
        scale_x = video_w / guess_pw
        scale_y = video_h / guess_ph
        print(
            f"[ASS] WARN: previewWidth/Height = 0 from FE — using heuristic scale={scale_x:.2f}x. "
            f"Subtitle có thể không khớp preview. Sửa FE gửi previewWidth/previewHeight thật."
        )

    # Lấy giá trị từ config (slider FE) — đây là PIXEL TRONG PREVIEW
    cfg_font_size = int(cfg.get("fontSize", 32))
    cfg_box_w_px = int(cfg.get("boxWidth", 300))
    cfg_box_h_px = int(cfg.get("boxHeight", 50))
    cfg_text_color = cfg.get("textColor", "#FFFFFF")
    cfg_bg_color = cfg.get("bgColor", "#000000")
    cfg_show_bg = bool(cfg.get("showBg", True))
    cfg_bg_opacity = float(cfg.get("bgOpacity", 1.0))
    cfg_font_family = cfg.get("fontFamily", "Times New Roman")
    cfg_border_radius = int(cfg.get("borderRadius", 4))

    # Scale sang video pixel
    box_w = max(1, int(cfg_box_w_px * scale_x))
    box_h = max(1, int(cfg_box_h_px * scale_y))
    font_size = max(8, int(cfg_font_size * scale_y))
    border_radius = max(0, int(cfg_border_radius * scale_y))

    # Colors
    text_color_ass = hex_to_ass_color(cfg_text_color, alpha="00")
    bg_alpha = _ass_alpha_from_opacity(cfg_bg_opacity)
    bg_color_ass = hex_to_ass_color(cfg_bg_color, alpha=bg_alpha)

    is_bg_transparent = (
        not cfg_show_bg
        or str(cfg_bg_color).strip().lower() == "transparent"
        or cfg_bg_opacity <= 0.01
    )

    # Drawing rectangle: vẽ từ (0,0) đến (box_w, box_h)
    # Dùng \an7 (anchor top-left) + \pos(x-hw, y-hh) để đặt center của box tại (x, y)
    # Cách này tránh quirk của libass khi drawing trong \an5
    hw = box_w // 2
    hh = box_h // 2
    r = min(border_radius, hw, hh)

    if r > 0:
        # Rounded rectangle: bezier ở 4 góc
        rect_drawing = (
            f"m {r} 0 "
            f"l {box_w - r} 0 "
            f"b {box_w} 0 {box_w} 0 {box_w} {r} "
            f"l {box_w} {box_h - r} "
            f"b {box_w} {box_h} {box_w} {box_h} {box_w - r} {box_h} "
            f"l {r} {box_h} "
            f"b 0 {box_h} 0 {box_h} 0 {box_h - r} "
            f"l 0 {r} "
            f"b 0 0 0 0 {r} 0"
        )
    else:
        rect_drawing = f"m 0 0 l {box_w} 0 l {box_w} {box_h} l 0 {box_h}"

    header = f"""[Script Info]
Title: AI Subtitle
ScriptType: v4.00+
PlayResX: {video_w}
PlayResY: {video_h}
ScaledBorderAndShadow: yes
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: BoxStyle,{cfg_font_family},{font_size},{text_color_ass},&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1
Style: TextStyle,{cfg_font_family},{font_size},{text_color_ass},&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # Pre-load font để wrap text theo box width
    pil_font = _get_pil_font(cfg_font_family, font_size)
    # Inner width = box_w trừ padding 8% mỗi bên (đã có trong wrap_text_to_box)

    events = []

    # Gap giữa các subtitle liền kề (giây). 0.05 = 50ms — đủ để 2 dialogue
    # không hiện chồng lên nhau ở frame chuyển, không quá rõ với người xem.
    SUBTITLE_GAP = 0.05

    # Build danh sách câu hợp lệ trước (đã filter rỗng) để biết câu kế tiếp
    valid_subs = [s for s in subtitles if safe_ass_text(s.get("text", "")).strip()]

    for idx, s in enumerate(valid_subs):
        text = safe_ass_text(s.get("text", ""))

        seg_start = float(s.get("start", 0))
        seg_end = float(s.get("end", 0))

        # Nếu có câu kế tiếp và end gần (hoặc trùng) start câu kế → cắt sớm 50ms
        if idx + 1 < len(valid_subs):
            next_start = float(valid_subs[idx + 1].get("start", 0))
            if seg_end >= next_start - SUBTITLE_GAP:
                seg_end = max(seg_start + 0.1, next_start - SUBTITLE_GAP)

        start = ass_time(seg_start)
        end = ass_time(seg_end)

        # x, y là tỉ lệ 0→1 từ FE (vị trí TÂM của box trên video)
        cx = int(float(s.get("x", 0.5)) * video_w)
        cy = int(float(s.get("y", 0.85)) * video_h)

        # Top-left của box (dùng cho \an7\pos)
        box_x = cx - hw
        box_y = cy - hh

        # Wrap text trong box bằng PIL → mỗi dòng vừa box_w
        # Convert "\N" cũ về "\n" để wrap, sau đó dùng "\\N" cho ASS
        plain_text = text.replace("\\N", "\n")
        all_lines = []
        for paragraph in plain_text.split("\n"):
            wrapped = wrap_text_to_box(paragraph, pil_font, box_w)
            all_lines.extend(wrapped)
        wrapped_text = "\\N".join(all_lines) if all_lines else text

        # ─── Layer 0: BACKGROUND BOX ──────────────────────────────────
        if not is_bg_transparent:
            bg_bgr = bg_color_ass[4:]  # "BBGGRR"
            box_override = (
                f"{{\\an7\\pos({box_x},{box_y})"
                f"\\fs1"
                f"\\1c&H{bg_bgr}"
                f"\\1a&H{bg_alpha}&"
                f"\\3a&HFF&"
                f"\\4a&HFF&"
                f"\\bord0\\shad0"
                f"\\p1}}"
                f"{rect_drawing}"
                f"{{\\p0}}"
            )
            events.append(
                f"Dialogue: 0,{start},{end},BoxStyle,,0,0,0,,{box_override}"
            )

        # ─── Layer 1: TEXT ─────────────────────────────────────────────
        # \an5\pos(cx, cy) đặt CENTER của text block tại tâm box
        # text đã được wrap thủ công nên không cần MarginL/R
        text_override = (
            f"{{\\an5\\pos({cx},{cy})"
            f"\\fs{font_size}"
            f"\\fn{cfg_font_family}"
            f"\\1c{text_color_ass}"
            f"\\3c&H000000&"
            f"\\bord1\\shad0"
            f"\\q2}}"
        )
        events.append(
            f"Dialogue: 1,{start},{end},TextStyle,,0,0,0,,{text_override}{wrapped_text}"
        )

    with open(ass_path, "w", encoding="utf-8-sig") as f:
        f.write(header + "\n".join(events))

    print(f"[ASS] Created: {ass_path} ({video_w}x{video_h}), box={box_w}x{box_h}, font={font_size}, {len(events)} dialogue lines")


def append_text_boxes_to_ass(video_path, text_boxes, ass_path, preview_w=0, preview_h=0):
    """
    Ghi thêm các free text box (từ FE) vào cuối file ASS đã có.

    Mỗi text box render bằng 2 Dialogue lines suốt cả video:
      - Layer 2: background rectangle (nếu showBg=True)
      - Layer 3: text

    text_boxes: list[dict] với keys:
        id, text, x, y, width, height  ← tỉ lệ 0..1
        fontSize, color, fontFamily
        bgColor, bgOpacity, showBg     ← background settings
    """
    if not text_boxes:
        return

    video_w, video_h = get_video_dimensions_ffprobe(video_path)

    if preview_w > 0 and preview_h > 0:
        scale_x = video_w / preview_w
        scale_y = video_h / preview_h
    else:
        scale_x = 3.0
        scale_y = 3.0

    # Thời lượng video để set end time cho text box permanent
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-print_format", "json", video_path],
            capture_output=True, text=True, timeout=15,
        )
        dur = float(json.loads(result.stdout)["format"]["duration"])
    except Exception:
        dur = 9999.0

    end_time_ass = ass_time(dur)
    start_time_ass = "0:00:00.00"

    new_lines = []

    for tb in text_boxes:
        text = safe_ass_text(str(tb.get("text", "")).strip())
        if not text:
            continue

        # Vị trí tâm box (tỉ lệ → pixel video)
        cx = int(float(tb.get("x", 0.5)) * video_w)
        cy = int(float(tb.get("y", 0.3)) * video_h)

        # Kích thước box (tỉ lệ → pixel video)
        box_w = max(40, int(float(tb.get("width", 0.25)) * video_w))
        box_h = max(20, int(float(tb.get("height", 0.08)) * video_h))

        # Font
        font_size = max(8, int(float(tb.get("fontSize", 18)) * scale_y))
        font_family = str(tb.get("fontFamily", "Times New Roman"))
        text_color = hex_to_ass_color(tb.get("color", "#FFFFFF"), alpha="00")

        # Background
        show_bg = bool(tb.get("showBg", True))
        bg_color_hex = tb.get("bgColor", "#000000")
        bg_opacity = float(tb.get("bgOpacity", 0.7))
        bg_alpha = _ass_alpha_from_opacity(bg_opacity)
        bg_color_ass = hex_to_ass_color(bg_color_hex, alpha=bg_alpha)
        is_bg_transparent = (
            not show_bg
            or str(bg_color_hex).strip().lower() == "transparent"
            or bg_opacity <= 0.01
        )

        hw = box_w // 2
        hh = box_h // 2

        # Rounded rect (bo góc cố định 6px sau khi scale)
        r = min(6, hw, hh)
        if r > 0:
            rect_drawing = (
                f"m {r} 0 "
                f"l {box_w - r} 0 "
                f"b {box_w} 0 {box_w} 0 {box_w} {r} "
                f"l {box_w} {box_h - r} "
                f"b {box_w} {box_h} {box_w} {box_h} {box_w - r} {box_h} "
                f"l {r} {box_h} "
                f"b 0 {box_h} 0 {box_h} 0 {box_h - r} "
                f"l 0 {r} "
                f"b 0 0 0 0 {r} 0"
            )
        else:
            rect_drawing = f"m 0 0 l {box_w} 0 l {box_w} {box_h} l 0 {box_h}"

        box_x = cx - hw
        box_y = cy - hh

        # Wrap text
        pil_font = _get_pil_font(font_family, font_size)
        plain_text = text.replace("\\N", "\n")
        all_lines = []
        for paragraph in plain_text.split("\n"):
            wrapped = wrap_text_to_box(paragraph, pil_font, box_w)
            all_lines.extend(wrapped)
        wrapped_text = "\\N".join(all_lines) if all_lines else text

        # Layer 2: background box
        if not is_bg_transparent:
            bg_bgr = bg_color_ass[4:]  # drop "&H"
            box_override = (
                f"{{\\an7\\pos({box_x},{box_y})"
                f"\\fs1"
                f"\\1c&H{bg_bgr}"
                f"\\1a&H{bg_alpha}&"
                f"\\3a&HFF&"
                f"\\4a&HFF&"
                f"\\bord0\\shad0"
                f"\\p1}}"
                f"{rect_drawing}"
                f"{{\\p0}}"
            )
            new_lines.append(
                f"Dialogue: 2,{start_time_ass},{end_time_ass},BoxStyle,,0,0,0,,{box_override}"
            )

        # Layer 3: text
        text_override = (
            f"{{\\an5\\pos({cx},{cy})"
            f"\\fs{font_size}"
            f"\\fn{font_family}"
            f"\\1c{text_color}"
            f"\\3c&H000000&"
            f"\\bord1\\shad0"
            f"\\q2}}"
        )
        new_lines.append(
            f"Dialogue: 3,{start_time_ass},{end_time_ass},TextStyle,,0,0,0,,{text_override}{wrapped_text}"
        )

    if new_lines:
        with open(ass_path, "a", encoding="utf-8-sig") as f:
            f.write("\n" + "\n".join(new_lines))
        print(f"[ASS] Appended {len(new_lines)} text-box dialogue lines to {ass_path}")


def ffmpeg_escape_path(path):
    path = os.path.abspath(path)
    path = path.replace("\\", "/")
    path = path.replace(":", "\\:")
    return path


def atempo_compress_audio(input_path, output_path, target_duration, max_ratio=2.5):
    """
    Dùng ffmpeg atempo để nén audio cho vừa target_duration mà không méo tiếng.
    atempo chỉ chấp nhận 0.5–2.0 trong 1 filter, ratio > 2.0 cần chain nhiều filter.
    Trả về True nếu nén thành công.
    """
    try:
        clip = AudioFileClip(input_path)
        src_duration = clip.duration
        clip.close()

        if src_duration <= 0 or target_duration <= 0:
            return False

        # Cho phép lệch 5% - không cần nén
        if src_duration <= target_duration * 1.05:
            return False

        ratio = src_duration / target_duration
        ratio = min(ratio, max_ratio)  # cap ratio để không bị quá nhanh

        # Chain atempo cho ratio > 2.0
        filters = []
        remaining = ratio
        while remaining > 2.0:
            filters.append("atempo=2.0")
            remaining /= 2.0
        if remaining > 1.0:
            filters.append(f"atempo={remaining:.4f}")

        if not filters:
            return False

        filter_str = ",".join(filters)

        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-filter:a", filter_str,
            "-vn",
            output_path,
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60,
            encoding="utf-8", errors="replace",
        )

        if result.returncode != 0:
            print(f"  [atempo] ffmpeg failed: {result.stderr[-300:]}")
            return False

        return os.path.exists(output_path) and os.path.getsize(output_path) >= 100

    except Exception as e:
        print(f"  [atempo] error: {e}")
        return False


def _sanitize_tts_text(text):
    """
    Loại bỏ ký tự gây edge-tts reject:
    - Emoji
    - Tab/control characters
    - Tiếng nước ngoài lạc giữa câu (romaji nhỏ giọt, ký tự Trung sót)
    - Dấu nháy/ngoặc không cần thiết
    """
    if not text:
        return ""
    text = str(text)
    # Xóa control chars
    text = "".join(ch for ch in text if ch == "\n" or ch == " " or 0x20 <= ord(ch) < 0xD800 or 0xE000 <= ord(ch) < 0xFFFE)
    # Xóa emoji range
    text = re.sub(r"[\U0001F300-\U0001FAFF\U0001F000-\U0001F2FF\u2600-\u27BF]", "", text)
    # Xóa ký tự Trung Quốc còn sót (tránh edge-tts cố đọc tiếng Trung)
    text = re.sub(r"[\u4e00-\u9fff]+", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Fallback voices theo ngôn ngữ (dùng khi voice chính fail liên tục)
_FALLBACK_VOICES = {
    "vi": ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"],
    "en": ["en-US-AriaNeural", "en-US-GuyNeural"],
    "zh": ["zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural"],
    "ja": ["ja-JP-NanamiNeural", "ja-JP-KeitaNeural"],
    "ko": ["ko-KR-SunHiNeural", "ko-KR-InJoonNeural"],
}


async def make_tts_audio(text, voice, rate, output_path, retries=3):
    import asyncio

    text = clean_text(text)
    text = _sanitize_tts_text(text)
    if not text or len(text) < 2:
        text = FALLBACK_TEXT

    # Clamp rate trong giới hạn edge-tts chấp nhận được thực tế
    rate_int = parse_rate_int(rate)
    rate_int = max(-30, min(35, rate_int))
    rate = f"+{rate_int}%" if rate_int >= 0 else f"{rate_int}%"

    def _normalize_for_tts(t):
        t = t.strip()
        # Thêm dấu chấm nếu không có dấu kết câu
        if t and t[-1] not in ".!?…":
            t = t + "."
        # Câu cực ngắn → pad bằng từ đệm để engine không reject
        if len(t) < 5:
            t = t + " ạ."
        return t

    # Build danh sách voice thử: voice chính + fallback cùng ngôn ngữ
    voice_lang = next((v["lang"] for v in VOICE_OPTIONS if v["value"] == voice), "vi")
    fallback_list = _FALLBACK_VOICES.get(voice_lang, [])
    # Thử voice chính trước, sau đó fallback (loại trùng)
    voices_to_try = [voice] + [v for v in fallback_list if v != voice]

    last_err = None
    for attempt in range(retries):
        # Vòng cuối thử fallback voice nếu vẫn còn voice khác
        current_voice = voices_to_try[min(attempt, len(voices_to_try) - 1)]

        try:
            # Xóa file cũ nếu có
            if os.path.exists(output_path):
                os.remove(output_path)

            tts_text = _normalize_for_tts(text)

            communicate = edge_tts.Communicate(
                text=tts_text,
                voice=current_voice,
                rate=rate,
            )
            await communicate.save(output_path)

            if os.path.exists(output_path) and os.path.getsize(output_path) >= 100:
                if current_voice != voice:
                    print(f"  [TTS] success with fallback voice: {current_voice}")
                return output_path

            last_err = Exception("TTS output is empty or too small")
            print(f"  [TTS] attempt {attempt + 1}/{retries} ({current_voice}): file too small, retrying...")

        except Exception as e:
            last_err = e
            print(f"  [TTS] attempt {attempt + 1}/{retries} ({current_voice}) failed: {e}")

        if attempt < retries - 1:
            await asyncio.sleep(1.0 + attempt * 0.5)  # backoff: 1s, 1.5s, 2s

    raise last_err

def parse_rate_int(rate_str):
    """Parse rate string '+10%' → 10, '-20%' → -20"""
    rate_str = str(rate_str or "+0%").strip()
    match = re.search(r"([+-]?\d+)", rate_str)
    return int(match.group(1)) if match else 0


def calc_dynamic_rate(tts_duration, segment_duration, base_rate_str="+0%"):
    base = parse_rate_int(base_rate_str)

    if segment_duration <= 0 or tts_duration <= 0:
        return f"+{base}%" if base >= 0 else f"{base}%"

    speed_factor = tts_duration / segment_duration
    needed = (speed_factor - 1) * 100
    total = int(round(base + needed))
    total = max(-30, min(35, total))  # ← giảm từ 80 → 35

    return f"+{total}%" if total >= 0 else f"{total}%"
async def process_video(
    job_id: str,
    filename: str,
    voice: str = DEFAULT_VOICE,
    rate: str = DEFAULT_RATE,
    *,
    language: str = "vi",
    aspect_ratio: str = "9 / 16",
    keep_original_audio: bool = True,
    audio_mode: str = "all",          # "all" = giữ tất cả, "music" = chỉ nhạc nền
    original_volume: float = 0.1,
    zoom_crop: float = 1.0,
    subtitle_config: dict | None = None,
    text_boxes: list | None = None,
    logo_path: str | None = None,
    music_path: str | None = None,
):
    try:
        if voice not in ALLOWED_VOICES:
            voice = DEFAULT_VOICE

        # Nếu voice không khớp target language → tự đổi voice mặc định của target
        # (vd FE chọn language=en nhưng quên đổi voice từ vi → fallback voice EN)
        voice_lang = next((v["lang"] for v in VOICE_OPTIONS if v["value"] == voice), None)
        if voice_lang and voice_lang != language:
            target_voices = [v["value"] for v in VOICE_OPTIONS if v["lang"] == language]
            if target_voices:
                old_voice = voice
                voice = target_voices[0]
                print(f"[Voice] auto-corrected: {old_voice} ({voice_lang}) → {voice} (matches target {language})")

        # Whisper sẽ tự detect ngôn ngữ nguồn ở dưới.
        # FE "language" = ngôn ngữ ĐÍCH (cho dịch + TTS), không phải nguồn.
        target_language_fe = language

        video_path = f"{UPLOAD_FOLDER}/{filename}"
        base_name = os.path.splitext(filename)[0]

        update_job(job_id, 10, "Extracting audio...")

        video = VideoFileClip(video_path)
        audio_path = f"{UPLOAD_FOLDER}/{base_name}.mp3"

        if video.audio is None:
            raise Exception("Video has no audio")

        video.audio.write_audiofile(audio_path)

        # ----- Tách vocal & background bằng Demucs -----
        update_job(job_id, 20, "Separating vocals from background...")

        vocal_path = f"{UPLOAD_FOLDER}/{base_name}_vocal.wav"
        bg_path = f"{UPLOAD_FOLDER}/{base_name}_bg.wav"

        try:
            separate_vocals_and_background(audio_path, vocal_path, bg_path)
            transcribe_input = vocal_path
            has_background = True
        except Exception as e:
            # Demucs fail → fallback dùng audio gốc, video output không có nhạc nền
            print(f"[Demucs] failed ({e}), fallback to original audio")
            transcribe_input = audio_path
            has_background = False

        # ─── Auto-detect ngôn ngữ nguồn (KHÔNG dùng FE language) ───────────
        # FE "language" = ngôn ngữ ĐÍCH (target để dịch + TTS).
        # Whisper cần ngôn ngữ NGUỒN — phải tự detect từ audio.
        # Cố ép sai source language → Whisper dịch âm sai bét.
        update_job(job_id, 25, "Detecting language...")

        try:
            import whisper as _whisper_pkg
            mel = _whisper_pkg.log_mel_spectrogram(
                _whisper_pkg.pad_or_trim(_whisper_pkg.load_audio(transcribe_input)),
                n_mels=model.dims.n_mels,
            ).to(model.device)
            _, lang_probs = model.detect_language(mel)
            detected_lang = max(lang_probs, key=lang_probs.get)
            print(f"[Whisper] detected source language: {detected_lang} (prob={lang_probs[detected_lang]:.2f})")
        except Exception as e:
            print(f"[Whisper] language detect failed ({e}), letting whisper auto-detect")
            detected_lang = None

        # Initial prompt theo từng ngôn ngữ giúp Whisper bám đúng register/từ vựng
        initial_prompts = {
            "zh": "以下是普通话视频的字幕,内容自然口语化。",
            "vi": "Đây là phụ đề video tiếng Việt, nội dung tự nhiên đời thường.",
            "en": "The following is a casual conversational video subtitle in English.",
            "ja": "以下は日常会話の日本語ビデオの字幕です。",
            "ko": "다음은 일상 대화 한국어 비디오의 자막입니다.",
        }
        initial_prompt = initial_prompts.get(detected_lang, "")

        update_job(job_id, 30, "Transcribing audio...")

        # transcribe_kwargs: nếu auto-detect không chắc thì để Whisper tự đoán
        transcribe_kwargs = dict(
            task="transcribe",
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            verbose=False,
        )
        if detected_lang:
            transcribe_kwargs["language"] = detected_lang
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt

        result = model.transcribe(transcribe_input, **transcribe_kwargs)

        # Lưu lại detected language để dùng cho bước dịch
        source_lang = result.get("language", detected_lang or "zh")
        target_lang = language  # từ FE
        print(f"[Whisper] source={source_lang} → target={target_lang}")

        raw_segments = result.get("segments", [])
        print(f"[Whisper] {len(raw_segments)} raw segments")

        # Lọc thêm 1 lớp ở Python: bỏ segment có dấu hiệu hallucinate
        segments = []
        for seg in raw_segments:
            no_speech = float(seg.get("no_speech_prob", 0.0))
            avg_logprob = float(seg.get("avg_logprob", 0.0))
            text = str(seg.get("text", "")).strip()
            duration = float(seg.get("end", 0)) - float(seg.get("start", 0))

            # Xác suất "không có giọng" cao → bỏ
            if no_speech > 0.6:
                print(f"[Whisper] skip no_speech={no_speech:.2f}: {text[:40]}")
                continue

            # Model không tự tin → bỏ
            if avg_logprob < -1.0:
                print(f"[Whisper] skip low_logprob={avg_logprob:.2f}: {text[:40]}")
                continue

            # Segment cực ngắn (< 0.3s) ở đầu video thường là hallucinate
            if duration < 0.3:
                print(f"[Whisper] skip tiny duration={duration:.2f}s: {text[:40]}")
                continue

            # Lọc câu bịa cố định (đã có sẵn)
            if is_whisper_hallucination(text):
                print(f"[Whisper] skip hallucination pattern: {text[:40]}")
                continue

            segments.append(seg)

        print(f"[Whisper] kept {len(segments)} / {len(raw_segments)} segments after filter")

        if not segments:
            raise Exception("No valid speech detected after filtering hallucinations")

        update_job(job_id, 45, "Translating...")

        translated_segments = translate_segments_with_ollama(
            segments,
            source_lang=source_lang,
            target_lang=target_language_fe,
        )

        json_path = f"{UPLOAD_FOLDER}/{filename}_translated.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(translated_segments, f, ensure_ascii=False, indent=2)

        update_job(job_id, 60, f"Generating voice: {voice}...")

        audio_clips = []
        final_segments = []
        current_end = 0.0  # thời điểm câu trước kết thúc, để tránh overlap

        video_duration = float(video.duration)

        for i, seg in enumerate(translated_segments):
            try:
                text = clean_text(seg.get("text", ""))

                # Câu rỗng = Whisper hallucination đã lọc → skip hoàn toàn
                if not text:
                    print(f"TTS SKIP {i}: hallucination filtered (zh={seg.get('original_text','')[:30]})")
                    continue

                if is_bad_translation(text):
                    text = FALLBACK_TEXT

                # Safety cap: LLM đã rút theo budget rồi, đây chỉ là chốt chặn
                text = shorten_words(text, max_words=18)

                tts_path = f"{UPLOAD_FOLDER}/tts_{job_id}_{i}.mp3"
                tts_path_fast = f"{UPLOAD_FOLDER}/tts_{job_id}_{i}_fast.mp3"

                print(f"TTS TEXT {i}: {text}")

                # Generate TTS với rate người dùng chọn
                try:
                    await make_tts_audio(
                        text=text,
                        voice=voice,
                        rate=rate,
                        output_path=tts_path,
                    )
                except Exception as e:
                    print(f"  [TTS] segment {i} skipped (TTS failed): {e}")
                    continue

                # Tính segment duration từ whisper timestamps
                seg_start = float(seg["start"])
                seg_end = float(seg["end"])
                segment_duration = seg_end - seg_start

                # Khoảng tới câu kế (hoặc tới hết video)
                if i + 1 < len(translated_segments):
                    next_start = float(translated_segments[i + 1]["start"])
                    gap_to_next = next_start - seg_start
                else:
                    gap_to_next = video_duration - seg_start

                # max_allowed: cho audio kéo dài tới TTS_OVERFLOW_FACTOR lần segment
                # nhưng KHÔNG vượt quá khoảng tới câu kế
                max_allowed = min(gap_to_next, segment_duration * TTS_OVERFLOW_FACTOR)
                max_allowed = max(max_allowed, segment_duration)

                # Đo độ dài TTS gốc
                probe_clip = AudioFileClip(tts_path)
                tts_duration = probe_clip.duration
                probe_clip.close()

                # Nén bằng atempo CHỈ KHI thật sự cần — và tối đa MAX_SPEED_RATIO
                use_path = tts_path
                if tts_duration > max_allowed * 1.05:
                    compressed = atempo_compress_audio(
                        tts_path, tts_path_fast, max_allowed, max_ratio=TTS_MAX_SPEED_RATIO
                    )
                    if compressed:
                        use_path = tts_path_fast
                        print(f"  [seg {i}] tts={tts_duration:.2f}s → compressed to fit {max_allowed:.2f}s")
                    else:
                        print(f"  [seg {i}] tts={tts_duration:.2f}s, can't fit {max_allowed:.2f}s — kept full audio")

                audio_clip = AudioFileClip(use_path)

                if audio_clip.duration <= 0:
                    audio_clip.close()
                    continue

                # ⭐ ƯU TIÊN ĐỒNG BỘ HÌNH: bám CHÍNH XÁC seg_start của Whisper.
                # KHÔNG đẩy lùi câu sau khi câu trước dài → tránh drift tích lũy.
                # Nếu câu trước tràn sang start câu này, audio sẽ overlap vài phần
                # mười giây (CompositeAudioClip tự mix) — chấp nhận được vì tốt hơn
                # nhiều so với drift hẳn cảnh.
                start_time = seg_start

                # Nếu start đã vượt video → bỏ
                if start_time >= video_duration:
                    audio_clip.close()
                    continue

                # Cắt audio nếu nó tràn vào câu sau quá xa (>30% câu sau)
                # Cho phép overlap nhẹ với câu sau, không cho overlap nhiều
                if i + 1 < len(translated_segments):
                    next_start = float(translated_segments[i + 1]["start"])
                    next_dur = float(translated_segments[i + 1]["end"]) - next_start
                    overlap_budget = next_start + max(0.4, next_dur * 0.3)
                    max_end = overlap_budget
                else:
                    max_end = video_duration

                if start_time + audio_clip.duration > max_end:
                    audio_clip = audio_clip.subclipped(0, max(0.3, max_end - start_time))

                # Cắt phần thừa ngoài video
                if start_time + audio_clip.duration > video_duration:
                    audio_clip = audio_clip.subclipped(0, video_duration - start_time)

                end_time = start_time + audio_clip.duration
                current_end = end_time  # giữ để theo dõi nhưng không dùng push start

                audio_clip = audio_clip.with_start(start_time)
                audio_clips.append(audio_clip)

                # Apply subtitle_config từ FE (override mặc định)
                sc = subtitle_config or {}
                final_segments.append(
                    {
                        "start": start_time,
                        "end": end_time,
                        "text": text,
                        "x": 0.5,
                        "y": 0.82,
                        "fontSize": int(sc.get("fontSize", 32)),
                        "bg": sc.get("bgColor", "#000000") if sc.get("showBg", True) else "transparent",
                        "color": sc.get("textColor", "#ffffff"),
                        "strokeColor": "#000000",
                        "strokeWidth": 2,
                        "fontFamily": sc.get("fontFamily", "Times New Roman"),
                        "borderRadius": int(sc.get("borderRadius", 4)),
                        "bgOpacity": float(sc.get("bgOpacity", 1.0)),
                        "width": 0.75,
                        "boxWidth": 0.75,
                        "boxHeight": 0.08,
                        "paddingX": 16,
                        "paddingY": 8,
                    }
                )

            except Exception as e:
                print(f"TTS ERROR {i}: {e}")
                continue

        if not audio_clips:
            raise Exception("No valid TTS audio generated")

        # ────────────────────────────────────────────────────────────────────
        # MERGE: render full audio track ra WAV trước, RỒI giải phóng moviepy,
        # cuối cùng dùng ffmpeg trực tiếp để mux video (KHÔNG re-encode video).
        # Cách này tránh x264 OOM khi Demucs/Whisper còn giữ RAM.
        # ────────────────────────────────────────────────────────────────────

        # ────────────────────────────────────────────────────────────────────
        # AUDIO MIX: tổng hợp 3 nguồn theo lựa chọn của FE
        #   1. TTS tiếng Việt (luôn có)
        #   2. Audio gốc — chỉ khi keep_original_audio=True
        #      • audio_mode="all"   → gốc đầy đủ (giọng + nhạc)
        #      • audio_mode="music" → chỉ nhạc nền (đã tách bỏ vocal tiếng Trung)
        #   3. Nhạc nền user upload — luôn loop nếu user gửi music_path
        # ────────────────────────────────────────────────────────────────────
        layers = list(audio_clips)  # TTS layer

        # Layer 2: original audio
        if keep_original_audio:
            try:
                if audio_mode == "music" and has_background and os.path.exists(bg_path):
                    base_clip = AudioFileClip(bg_path)
                    print("[Merge] keeping original audio: music only (vocals removed)")
                else:
                    base_clip = AudioFileClip(audio_path)
                    print("[Merge] keeping original audio: all (voice + music)")

                base_clip = base_clip.with_volume_scaled(float(original_volume))
                if base_clip.duration > video_duration:
                    base_clip = base_clip.subclipped(0, video_duration)
                layers.insert(0, base_clip)
            except Exception as e:
                print(f"[Merge] failed to keep original audio ({e}), skipping")

        # Layer 3: user-uploaded background music
        if music_path and os.path.exists(music_path):
            try:
                user_music = AudioFileClip(music_path)
                user_music = user_music.with_volume_scaled(0.25)
                if user_music.duration > video_duration:
                    user_music = user_music.subclipped(0, video_duration)
                layers.insert(0, user_music)
                print(f"[Merge] added user music: {os.path.basename(music_path)}")
            except Exception as e:
                print(f"[Merge] user music failed ({e})")

        final_audio = CompositeAudioClip(layers)

        update_job(job_id, 75, "Merging video...")

        # Render audio track ra WAV (mp3 cũng được, WAV không re-encode lossy nên rõ hơn)
        final_audio_path = f"{UPLOAD_FOLDER}/final_audio_{job_id}.wav"
        if final_audio.duration > video_duration:
            final_audio = final_audio.subclipped(0, video_duration)
        final_audio.write_audiofile(
            final_audio_path,
            fps=44100,
            codec="pcm_s16le",
            logger=None,
        )

        # Đóng tất cả moviepy clip để giải phóng RAM trước khi gọi ffmpeg
        bg_clip = None
        try:
            video.close()
            final_audio.close()
            for c in audio_clips:
                c.close()
            if has_background and bg_clip is not None:
                # bg_clip có thể đã bị composite consume, không thiệt hại nếu fail
                try:
                    bg_clip.close()
                except Exception:
                    pass
        except Exception:
            pass

        # Mux video gốc + audio mới bằng ffmpeg
        merged_filename = f"final_{filename}"
        merged_path = f"{UPLOAD_FOLDER}/{merged_filename}"

        # ────────────────────────────────────────────────────────────────────
        # NOTE: KHÔNG burn-in subtitle ở đây.
        # Lý do: FE muốn user kéo/thả + resize subtitle trước khi xuất.
        # /upload-video chỉ tạo video có audio mới (TTS + bg).
        # FE gọi /render-final với subtitle đã edit để burn-in cuối cùng.
        # ────────────────────────────────────────────────────────────────────

        # ────────────────────────────────────────────────────────────────────
        # Build filter graph cho video stream:
        #   - aspect_ratio: crop về 9:16, 16:9, 1:1 nếu khác tỉ lệ gốc
        #   - zoom_crop:    phóng to crop để tránh trùng bản quyền
        #   - logo overlay: chèn logo người dùng upload góc phải trên
        # Có bất kỳ filter nào → phải re-encode (không -c:v copy được)
        # ────────────────────────────────────────────────────────────────────
        video_filters = []
        zoom_factor = max(0.5, min(2.0, float(zoom_crop or 1.0)))

        # Crop theo aspect ratio FE chọn
        ar_map = {"9 / 16": (9, 16), "16 / 9": (16, 9), "1 / 1": (1, 1)}
        ar_target = ar_map.get(str(aspect_ratio), None)
        if ar_target:
            ar_w, ar_h = ar_target
            video_filters.append(
                f"crop='min(iw,ih*{ar_w}/{ar_h})':'min(ih,iw*{ar_h}/{ar_w})'"
            )

        # Zoom crop (sau aspect crop)
        if abs(zoom_factor - 1.0) > 0.001:
            video_filters.append(
                f"scale=iw*{zoom_factor}:ih*{zoom_factor},crop=iw/{zoom_factor}:ih/{zoom_factor}"
            )

        inputs = ["-i", video_path, "-i", final_audio_path]
        v_label = "[0:v]"
        filter_complex_parts = []

        if video_filters:
            chain = ",".join(video_filters)
            filter_complex_parts.append(f"{v_label}{chain}[vfilt]")
            v_label = "[vfilt]"

        # Logo overlay (sau crop/zoom)
        if logo_path and os.path.exists(logo_path):
            inputs.extend(["-i", logo_path])
            filter_complex_parts.append(f"[2:v]scale=iw*0.12:-1[logo]")
            filter_complex_parts.append(f"{v_label}[logo]overlay=W-w-20:20[vout]")
            v_label = "[vout]"

        cmd_base = ["ffmpeg", "-y", "-loglevel", "error"] + inputs

        if filter_complex_parts:
            cmd = cmd_base + [
                "-filter_complex", ";".join(filter_complex_parts),
                "-map", v_label,
                "-map", "1:a:0",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-threads", "2",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                merged_path,
            ]
            print(f"[Merge] running ffmpeg with filters (aspect={aspect_ratio}, zoom={zoom_factor:.2f}, logo={'yes' if logo_path else 'no'})")
        else:
            cmd = cmd_base + [
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                merged_path,
            ]
            print(f"[Merge] running ffmpeg mux (copy video, encode AAC audio)")

        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            print(f"[Merge] ffmpeg failed:\n{proc.stderr[-800:]}")
            print(f"[Merge] retrying with safe re-encode (libx264 ultrafast, low memory)...")
            cmd_retry = cmd_base + [
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-threads", "2",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                merged_path,
            ]
            proc2 = subprocess.run(cmd_retry, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if proc2.returncode != 0:
                raise Exception(f"ffmpeg merge failed: {proc2.stderr[-500:]}")

        # Dọn file audio tạm
        try:
            os.remove(final_audio_path)
        except Exception:
            pass

        update_job(
            job_id,
            100,
            "Completed!",
            {
                "video": merged_filename,
                "subtitles": final_segments,
                "voice": voice,
                "rate": rate,
            },
        )

    except Exception as e:
        print(e)
        update_job(job_id, 0, f"Failed: {str(e)}")


@app.get("/")
def root():
    return {
        "message": "video route working",
        "translate_provider": TRANSLATE_PROVIDER,
        "translate_model": TRANSLATE_MODEL,
        "default_voice": DEFAULT_VOICE,
        "languages": list(LANGUAGE_TO_WHISPER.keys()),
    }


@app.get("/voices")
def get_voices(language: str | None = None):
    """
    Trả về danh sách giọng đọc.
    - Không truyền language: trả về tất cả
    - Truyền ?language=vi → chỉ trả về voice tiếng Việt
    Endpoint khớp với FE useEffect Load voices.
    """
    if language:
        filtered = [v for v in VOICE_OPTIONS if v.get("lang") == language]
    else:
        filtered = VOICE_OPTIONS

    return {
        "defaultVoice": DEFAULT_VOICE,
        "voices": filtered,
    }


@app.post("/preview-voice")
async def preview_voice(
    voice: str = Form(DEFAULT_VOICE),
    text: str | None = Form(None),
    rate: str = Form(DEFAULT_RATE),
):
    """
    Phát thử 1 đoạn TTS ngắn để FE bấm "Nghe thử giọng".
    Trả về file mp3 mỗi lần gọi (không cache).
    """
    if voice not in ALLOWED_VOICES:
        voice = DEFAULT_VOICE

    if not text:
        # Câu mẫu cho từng ngôn ngữ
        sample_by_lang = {
            "vi": "Xin chào, tôi là giọng đọc cho video của bạn.",
            "en": "Hello, this is the narrator voice for your video.",
            "zh": "你好,这是您视频的旁白声音。",
            "ja": "こんにちは、これはあなたのビデオのナレーターです。",
            "ko": "안녕하세요. 영상의 내레이터 목소리입니다.",
        }
        # Tìm lang theo voice value
        voice_lang = next((v["lang"] for v in VOICE_OPTIONS if v["value"] == voice), "vi")
        text = sample_by_lang.get(voice_lang, sample_by_lang["vi"])

    preview_filename = f"preview_{uuid.uuid4().hex}.mp3"
    preview_path = os.path.join(UPLOAD_FOLDER, preview_filename)

    try:
        await make_tts_audio(text=text, voice=voice, rate=rate, output_path=preview_path, retries=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS preview failed: {e}")

    return FileResponse(
        path=preview_path,
        media_type="audio/mpeg",
        filename=preview_filename,
    )


@app.post("/upload-video")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    voice: str = Form(DEFAULT_VOICE),
    rate: str = Form(DEFAULT_RATE),
    # Tham số khớp 1-1 với FE handleExport()
    language: str = Form("vi"),
    aspectRatio: str = Form("9 / 16"),
    keepOriginalAudio: str = Form("true"),
    audioMode: str = Form("all"),
    originalVolume: str = Form("0.1"),
    zoomCrop: str = Form("1.0"),
    subtitleConfig: str = Form("{}"),
    textBoxes: str = Form("[]"),
    logo: UploadFile | None = File(None),
    music: UploadFile | None = File(None),
):
    """
    Endpoint chính FE gọi sau khi user bấm "Lưu & Xuất video".
    Nhận TẤT CẢ thiết lập trong 1 request multipart, lưu file, kích background job.

    Returns: { job_id, voice, rate }
    FE poll /job-status/{job_id} đến khi progress=100.
    """
    if voice not in ALLOWED_VOICES:
        voice = DEFAULT_VOICE

    # Parse các JSON field
    try:
        sub_cfg = json.loads(subtitleConfig) if subtitleConfig else {}
    except Exception:
        sub_cfg = {}
    try:
        text_boxes_data = json.loads(textBoxes) if textBoxes else []
    except Exception:
        text_boxes_data = []

    # Parse boolean / float
    keep_audio = str(keepOriginalAudio).lower() in ("true", "1", "yes")
    try:
        orig_vol = float(originalVolume)
    except Exception:
        orig_vol = 0.1
    try:
        zoom_val = float(zoomCrop)
    except Exception:
        zoom_val = 1.0

    # Lưu video chính
    safe_filename = (file.filename or f"video_{uuid.uuid4().hex}.mp4").replace("/", "_").replace("\\", "_")
    file_path = os.path.join(UPLOAD_FOLDER, safe_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Lưu logo (optional)
    logo_path = None
    if logo and logo.filename:
        logo_filename = f"logo_{uuid.uuid4().hex}_{logo.filename}"
        logo_path = os.path.join(UPLOAD_FOLDER, logo_filename)
        with open(logo_path, "wb") as f:
            shutil.copyfileobj(logo.file, f)

    # Lưu music (optional)
    music_path = None
    if music and music.filename:
        music_filename = f"music_{uuid.uuid4().hex}_{music.filename}"
        music_path = os.path.join(UPLOAD_FOLDER, music_filename)
        with open(music_path, "wb") as f:
            shutil.copyfileobj(music.file, f)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "progress": 0,
        "status": "Queued...",
        "result": None,
        "voice": voice,
        "rate": rate,
        "language": language,
    }

    background_tasks.add_task(
        process_video,
        job_id,
        safe_filename,
        voice,
        rate,
        language=language,
        aspect_ratio=aspectRatio,
        keep_original_audio=keep_audio,
        audio_mode=audioMode,
        original_volume=orig_vol,
        zoom_crop=zoom_val,
        subtitle_config=sub_cfg,
        text_boxes=text_boxes_data,
        logo_path=logo_path,
        music_path=music_path,
    )

    return {
        "job_id": job_id,
        "voice": voice,
        "rate": rate,
        "language": language,
    }


@app.get("/job-status/{job_id}")
def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.delete("/job/{job_id}")
def delete_job(job_id: str):
    """Xóa job sau khi FE tải video xong, giải phóng memory dict."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    jobs.pop(job_id, None)
    return {"deleted": job_id}


@app.get("/download/{filename}")
async def download_file(filename: str):
    filename = urllib.parse.unquote(filename)
    file_path = f"{UPLOAD_FOLDER}/{filename}"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=os.path.basename(file_path),
    )


@app.post("/render-final")
async def render_final(data: dict):
    video = data["video"]
    subtitles = data["subtitles"]
    images = data.get("images", [])
    subtitle_config = data.get("subtitleConfig", {}) or {}
    text_boxes = data.get("textBoxes", []) or []

    preview_w = float(data.get("previewWidth", 0))
    preview_h = float(data.get("previewHeight", 0))

    input_path = f"{UPLOAD_FOLDER}/{video}"

    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Video not found")

    safe_video = urllib.parse.unquote(video).replace(" ", "_")
    base_name = os.path.splitext(safe_video)[0]

    ass_filename = f"{base_name}_{uuid.uuid4().hex}.ass"
    ass_path = f"{UPLOAD_FOLDER}/{ass_filename}"

    output_filename = f"custom_{safe_video}"
    output_path = f"{UPLOAD_FOLDER}/{output_filename}"

    # Inject subtitle_config (slider settings từ FE) vào MỖI subtitle
    # để create_ass_file dùng đúng fontSize/boxWidth/boxHeight/bgOpacity từ slider
    enriched_subs = []
    for s in subtitles:
        enriched_subs.append({**subtitle_config, **s})

    create_ass_file(input_path, enriched_subs, ass_path, preview_w, preview_h, subtitle_config)

    # Ghi thêm free text boxes (nếu có) vào cùng file ASS
    if text_boxes:
        append_text_boxes_to_ass(input_path, text_boxes, ass_path, preview_w, preview_h)

    video_w, video_h = get_video_dimensions_ffprobe(input_path)

    inputs = ["-i", input_path]

    filter_parts = []

    # subtitle layer
    ass_path_for_ffmpeg = ffmpeg_escape_path(ass_path)
    current_stream = "[0:v]"

    filter_parts.append(
        f"{current_stream}ass='{ass_path_for_ffmpeg}'[v0]"
    )
    current_stream = "[v0]"

    # overlay images
    for i, img in enumerate(images):
        image_path = os.path.join(UPLOAD_FOLDER, img["filename"])

        if not os.path.exists(image_path):
            continue

        inputs.extend(["-i", image_path])

        img_w = int(img["width"] * video_w)
        img_h = int(img["height"] * video_h)

        x = int(img["x"] * video_w - img_w / 2)
        y = int(img["y"] * video_h - img_h / 2)

        scale_label = f"[img{i}]"
        out_label = f"[v{i+1}]"

        filter_parts.append(
            f"[{i+1}:v]scale={img_w}:{img_h}{scale_label}"
        )

        filter_parts.append(
            f"{current_stream}{scale_label}overlay={x}:{y}{out_label}"
        )

        current_stream = out_label

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        current_stream,
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-c:a",
        "copy",
        output_path,
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        raise HTTPException(status_code=500, detail=e.stderr[-1000:])

    return {"filename": output_filename}
@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": filename,
        "url": f"/download-image/{filename}"
    }
@app.get("/download-image/{filename}")
async def download_image(filename: str):
    file_path = f"{UPLOAD_FOLDER}/{filename}"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(file_path)