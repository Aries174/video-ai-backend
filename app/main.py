from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
    Form,
)
from fastapi.concurrency import run_in_threadpool
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
from app.services.capcut_tts import capcut_tts_generate, get_resource_id as capcut_get_resource_id
from app.services.chinese_text_remover import remove_chinese_text_from_video
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
WHISPER_MIN_LOGPROB = float(os.getenv("WHISPER_MIN_LOGPROB", "-10.0"))
WHISPER_KEEP_ALL_SEGMENTS = os.getenv("WHISPER_KEEP_ALL_SEGMENTS", "").lower() in ["true", "1", "yes"]
if WHISPER_KEEP_ALL_SEGMENTS:
    WHISPER_MIN_LOGPROB = -100.0  # Vô hiệu hóa filter logprob
print(f"[Config] Whisper min logprob: {WHISPER_MIN_LOGPROB}") 
if WHISPER_KEEP_ALL_SEGMENTS:
    print(f"[Config] WARNING: WHISPER_KEEP_ALL_SEGMENTS=true → keeping ALL segments (including potential hallucinations)")

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
#   - value: edge-tts voice id (hoặc capcut:<voice_type> cho CapCut voices)
#   - label: hiển thị FE
#   - lang : mã ngôn ngữ ngắn (vi/en/zh/ja/ko) để filter theo dropdown
# ---------------------------------------------------------------------------
VOICE_OPTIONS = [
    # Vietnamese — Edge TTS
    {"value": "vi-VN-HoaiMyNeural",  "label": "HN - Hoài My (Nữ)",   "lang": "vi"},
    {"value": "vi-VN-NamMinhNeural", "label": "HN - Nam Minh (Nam)", "lang": "vi"},

    # Vietnamese — CapCut TTS (API không chính thức)
    {"value": "capcut:BV074_streaming",     "label": "🎬 Cô Gái Hoạt Ngôn (CapCut)",    "lang": "vi"},
    {"value": "capcut:BV075_streaming",     "label": "🎬 Thanh Niên Tự Tin (CapCut)",    "lang": "vi"},
    {"value": "capcut:BV421_vivn_streaming", "label": "🎬 Nhỏ Ngọt Ngào (CapCut)",       "lang": "vi"},
    {"value": "capcut:BV562_streaming",     "label": "🎬 Mai (CapCut)",                  "lang": "vi"},
    {"value": "capcut:vi_female_huong",     "label": "🎬 Giọng Nữ Phổ Thông (CapCut)",  "lang": "vi"},
    {"value": "capcut:BV074_streaming_dsp", "label": "🎬 Giọng Bé (CapCut)",             "lang": "vi"},
    {"value": "capcut:BV075_streaming_vibrato_dsp", "label": "🎬 Việt Méo (CapCut)",     "lang": "vi"},
    {"value": "capcut:BV075_streaming_demon_dsp",   "label": "🎬 Kenny Đại Đế (CapCut)", "lang": "vi"},
    {"value": "capcut:multi_female_richgirl_uranus_bigtts",  "label": "🎬 Review Phim (CapCut)",  "lang": "vi"},
    {"value": "capcut:multi_female_yangguangnv_uranus_bigtts", "label": "🎬 Ban Mai (CapCut)",   "lang": "vi"},
    {"value": "capcut:multi_female_sisi_uranus_bigtts",       "label": "🎬 Bản Tin Nữ (CapCut)", "lang": "vi"},
    {"value": "capcut:multi_female_peiqi_uranus_bigtts",      "label": "🎬 Gái Mới Lớn (CapCut)", "lang": "vi"},
    {"value": "capcut:multi_male_felipe_uranus_bigtts",       "label": "🎬 Giọng Nam Trầm (CapCut)", "lang": "vi"},

    # English — Edge TTS
    {"value": "en-US-AriaNeural",     "label": "US - Aria (Nữ)",      "lang": "en"},
    {"value": "en-US-GuyNeural",      "label": "US - Guy (Nam)",      "lang": "en"},
    {"value": "en-GB-SoniaNeural",    "label": "UK - Sonia (Nữ)",     "lang": "en"},
    # Chinese — Edge TTS
    {"value": "zh-CN-XiaoxiaoNeural", "label": "CN - Xiaoxiao (Nữ)",  "lang": "zh"},
    {"value": "zh-CN-YunxiNeural",    "label": "CN - Yunxi (Nam)",    "lang": "zh"},
    # Japanese — Edge TTS
    {"value": "ja-JP-NanamiNeural",   "label": "JP - Nanami (Nữ)",    "lang": "ja"},
    {"value": "ja-JP-KeitaNeural",    "label": "JP - Keita (Nam)",    "lang": "ja"},
    # Korean — Edge TTS
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
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))  # số câu mỗi lần gửi Ollama
print(f"[Config] Batch size for translation: {BATCH_SIZE}")
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
# Ký tự lặp noise thuần để lọc "嗯嗯", "啊啊" nhưng không bỏ "妈" hay "你"
_REPEATED_NOISE_CHARS = set("嗯啊哦唔呃哎哈嘿嘻呀哇")
# Bỏ dấu câu để so khớp pattern
_PUNCT_RE = re.compile(r"[，。！？、,.\!\?\s]+")


def is_whisper_hallucination(text):
    text = str(text or "").strip()
    if not text:
        return True

    # Câu chỉ chứa particle/interjection (sau khi strip dấu câu)
    if _PURE_INTERJECTION_RE.match(text):
        return True

    cleaned = _PUNCT_RE.sub("", text)
    if not cleaned:
        return True

    # 1-2 ký tự lặp của noise thuần ("嗯嗯", "啊啊啊") thường là noise
    if len(cleaned) <= 3 and len(set(cleaned)) == 1 and cleaned in _REPEATED_NOISE_CHARS:
        return True

    # Chỉ flag pattern khi nó CHIẾM gần như toàn bộ câu, không phải nằm trong câu dài
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
• 太厉害了,想周成固定起来再把周成外圈切割开一个缺口取出里面的钢珠和周成内圈。 = "Quá đỉnh rồi, cố định vòng bi lại rồi cắt một khe trên vòng ngoài để lấy các viên bi và vòng trong ra."
• 接着焊接上罗文感和罗母。 = "Tiếp theo hàn một đoạn ty ren và một đai ốc vào."
• 然后再把钢管切割一段下来。 = "Sau đó cắt một đoạn ống thép phù hợp."
• 再然后把周成焊接到钢管上和周成内圈焊接上钢板。 = "Tiếp tục hàn vòng bi vào ống thép và hàn một tấm thép vào vòng trong của vòng bi."
• 这样一把可调节班手就制作好了。 = "Vậy là đã chế tạo xong một chiếc cờ lê điều chỉnh đa năng."
• 真是太聪明了。 = "Đúng là một ý tưởng rất thông minh."
• 注意看。 = "Chú ý xem nhé."
• 把门禁对折再反折一厘米。 = "Gấp đôi mép vải rồi gấp ngược vào trong khoảng 1 cm."
• 再把铃子内贴放上面面对面。 = "Đặt lớp vải lót úp mặt vào nhau."
• 车分起来。 = "Sau đó may cố định lại."
• 这样做圆铃特别简单。 = "Làm phần bo cong theo cách này rất đơn giản."
• 在有弧度的地方打上几个剪刀口。 = "Ở những đoạn cong, dùng kéo cắt vài đường nhỏ."
• 这样做的铃子更圆顺。 = "Nhờ vậy phần bo sẽ tròn đều và mềm mại hơn."
• 然后在内贴的上面压一道边线。 = "Tiếp theo may một đường chỉ diễu trên lớp vải lót."
• 这样防止铃子外翻。 = "Cách này giúp mép vải không bị lật ra ngoài."
• 再放到正面来看效果。 = "Cuối cùng lật mặt phải ra để xem thành phẩm."
• 是不是非常的平整又圆顺。 = "Có phải trông rất phẳng phiu và tròn đẹp không?"
• 反面也浮贴。 = "Mặt trái cũng được xử lý rất gọn gàng."
• 你学会了吗? = "Bạn đã học được cách làm này chưa?"
• 自从有了这个小熊电砂锅,包汤炖肉都不需要盯着火了。 = "Từ khi có chiếc nồi đất điện này, việc nấu canh hay hầm thịt trở nên nhàn hơn rất nhiều."
• 分体式的设计三升的黄金容量。 = "Thiết kế tách rời với dung tích vàng 3 lít, vừa đủ cho cả gia đình."
• 材质是双层设计用着也放心。 = "Lòng nồi bằng chất liệu an toàn, sử dụng yên tâm hơn."
• 冷热骤变不开裂。 = "Chịu được thay đổi nhiệt độ nóng lạnh, hạn chế nứt vỡ."
• 温火慢炖。 = "Chế độ hầm lửa nhỏ giúp món ăn chín đều và đậm vị."
• 不溢锅,防干烧。 = "Tự động chống trào và chống cháy khô an toàn."
• 砂锅煲的汤能更好地锁住食材的营养和味道。 = "Nồi đất giúp giữ trọn dưỡng chất và hương vị tự nhiên của thực phẩm."
• 五大菜单自带预约保温功能。 = "Tích hợp 5 chế độ nấu cùng chức năng hẹn giờ và giữ ấm."
• 煲煮焖炖炒,一锅都能搞定。 = "Từ nấu canh, nấu cháo đến hầm thực phẩm đều có thể thực hiện trong một nồi."
• 早上煲个砂锅粥。 = "Buổi sáng chỉ cần nấu một nồi cháo thơm ngon."
• 煮完就能直接端上桌。 = "Nấu xong có thể mang lên bàn ăn hoặc tiếp tục hầm đều rất tiện."
• 洗的时候用水一冲就干净了。 = "Khi vệ sinh chỉ cần xả nước là sạch."
• 也不用担心进水真的很方便。 = "Không phải lo nước vào bộ phận điện, cực kỳ tiện lợi khi sử dụng."
• 太厉害了。 = "Quá đỉnh rồi!"
• 油管断在接头螺纹里拧不出来，不要怕。 = "Nếu ống dẫn dầu bị gãy và kẹt trong phần ren nối, đừng lo."
• 可以用螺纹钻头插进去。 = "Hãy dùng mũi tháo ren ngược cắm vào bên trong đoạn ống bị gãy."
• 卡住油管。 = "Khi mũi tháo đã bám chặt vào ống dẫn dầu,"
• 再把油管拧出来。 = "chỉ cần xoay là có thể vặn đoạn ống bị kẹt ra ngoài."
• 最后再用老虎钳把断掉的油管取出来。 = "Sau đó dùng kìm để lấy hoàn toàn phần ống dầu bị gãy ra."
• 这样就可以更换新油管了。 = "Như vậy là có thể thay thế bằng một ống dẫn dầu mới."
• 你们觉得这个方法实用吗? = "Mọi người thấy mẹo này có hữu ích không?"
• 太厉害了。 = "Quá đỉnh rồi!"
• 报废的齿轮和老虎钳不要扔。 = "Đừng vội vứt những bánh răng cũ và kìm chết đã hỏng."
• 可以制作成多功能轴承提取工具。 = "Chúng có thể được tận dụng để chế tạo một dụng cụ tháo vòng bi đa năng."
• 先把老虎钳焊接到齿轮上。 = "Đầu tiên hàn chiếc kìm chết vào bánh răng."
• 再把齿轮切割下来。 = "Sau đó cắt và gia công bánh răng theo kích thước phù hợp."
• 然后再把另一个老虎钳的前嘴焊接到齿轮上切割下来。 = "Tiếp tục hàn phần đầu của một chiếc kìm chết khác vào bánh răng rồi cắt chỉnh lại."
• 接着把螺纹杆焊接到老虎钳的控制手柄上。 = "Tiếp theo hàn một đoạn ty ren vào tay cầm điều chỉnh của kìm."
• 这样就可以通过调节拧紧老虎钳上的螺母来卡住轴承。 = "Khi siết đai ốc điều chỉnh, dụng cụ sẽ kẹp chặt vòng bi hoặc bạc đạn."
• 最后再把轴承提取出来。 = "Cuối cùng chỉ cần kéo ra là có thể tháo vòng bi bị kẹt."
• 你们说这个发明好用吗? = "Mọi người thấy phát minh này có hữu ích không?"
• 这个杯子好啊。 = "Chiếc cốc này thật sự rất tiện."
• 能热牛奶还能煮茶炖甜品。 = "Vừa hâm sữa, pha trà, lại còn nấu được các món tráng miệng."
• 不仅好看还特别实用。 = "Không chỉ đẹp mắt mà còn cực kỳ hữu dụng."
• 内胆是法郎又不粘的。 = "Lòng cốc được phủ lớp chống dính cao cấp."
• 用起来特别方便。 = "Sử dụng và vệ sinh đều rất dễ dàng."
• 功能也齐全。 = "Tích hợp đầy đủ nhiều chức năng."
• 烧水、花茶、煲汤泡面、焖饭煮好还自动保温。 = "Có thể đun nước, pha trà, hâm canh, nấu mì và nấu cơm, xong còn tự động giữ ấm."
• 早餐用它热个牛奶。 = "Buổi sáng dùng để hâm nóng sữa rất tiện."
• 不温不烫。 = "Nhiệt độ vừa phải, không quá nóng."
• 55度刚刚好。 = "Mức giữ ấm 55°C rất dễ uống."
• 平时用它烧水煮养生茶都很方便。 = "Bình thường dùng để đun nước hay nấu trà thảo mộc cũng rất phù hợp."
• 你们如果喜欢喝茶。 = "Nếu bạn là người thích uống trà."
• 还能用这个茶漏煮茶。 = "Có thể dùng lõi lọc đi kèm để nấu trà."
• 就不担心喝到茶渣了。 = "Không lo uống phải bã trà nữa."
• 这小巧精致的造型。 = "Thiết kế nhỏ gọn và tinh tế."
• 不管你们在家或办公室用。 = "Dù dùng ở nhà hay tại văn phòng."
• 还是出差旅行。 = "Hoặc mang theo khi đi công tác, du lịch."
• 都很方便。 = "Đều rất tiện lợi."
• 每天早上都要吃鸡蛋的姐妹。 = "Những ai sáng nào cũng ăn trứng thì đừng bỏ qua nhé."
• 这个制高大品牌蒸蛋器真的可以安排上。 = "Chiếc máy hấp trứng của thương hiệu Zhigao này thực sự rất đáng mua."
• 用它煮鸡蛋不用看时间也不用早起了。 = "Dùng nó luộc trứng không cần canh giờ, cũng không phải dậy sớm."
• 双层的设计特别实用。 = "Thiết kế hai tầng cực kỳ tiện lợi."
• 煮鸡蛋的同时还能蒸点包子馒头。 = "Vừa luộc trứng vừa có thể hấp bánh bao hoặc màn thầu."
• 晚上可以提前预约好时间。 = "Buổi tối chỉ cần hẹn giờ trước."
• 早上起来就能吃到热乎乎的早餐了。 = "Sáng thức dậy là có ngay bữa sáng nóng hổi."
• 功能也很齐全。 = "Các chức năng cũng rất đầy đủ."
• 不仅能做早餐。 = "Không chỉ dùng để chuẩn bị bữa sáng."
• 炖个甜品或者给杯子高温清洁都没问题。 = "Dùng để chưng món ngọt hoặc khử trùng cốc bằng nhiệt độ cao đều rất tiện."
• 还是制高大品牌。 = "Lại còn đến từ thương hiệu Zhigao uy tín."
• 品质也没得说。 = "Chất lượng thì hoàn toàn yên tâm."
• 小小一个。 = "Thiết kế nhỏ gọn."
• 放厨房还不占空间。 = "Đặt trong bếp cũng không chiếm nhiều diện tích."
• 这个设计好啊。 = "Thiết kế này thật sự rất thông minh."
• 蒜瓣一放。 = "Chỉ cần cho tép tỏi vào."
• 一按蒜丁就切好了。 = "Ấn nhẹ một cái là tỏi đã được băm nhỏ."
• 一点都不费手。 = "Không tốn chút sức nào."
• 里面是不锈钢的刀片。 = "Bên trong sử dụng lưỡi dao bằng thép không gỉ."
• 按压下去就把大蒜压成大小均匀的蒜丁了。 = "Chỉ cần nhấn xuống là tỏi được cắt thành những miếng đều nhau."
• 也不用拿刀切半天。 = "Không cần mất thời gian dùng dao băm thủ công."
• 还手上都是味了。 = "Tay cũng không bị ám mùi tỏi."
• 倒点生抽跟醋还能秒变一个酱料底。 = "Cho thêm chút nước tương và giấm là có ngay phần sốt chấm thơm ngon."
• 别看那么便宜。 = "Đừng thấy giá rẻ mà xem thường."
• 质感可好了。 = "Chất lượng hoàn thiện rất tốt."
• 还能拆卸。 = "Các bộ phận còn có thể tháo rời."
• 用完水一冲就干净了。 = "Dùng xong chỉ cần rửa qua nước là sạch."
• 自从买了这个带转盘的杂粮收纳盒。 = "Từ khi mua hộp đựng ngũ cốc xoay 360 độ này."
• 每次煮个粥或者做个杂粮饭。 = "Mỗi lần nấu cháo hay làm cơm ngũ cốc."
• 真的太方便了。 = "Thực sự rất tiện lợi."
• 再也不用各种袋子摆的满柜子都是。 = "Không còn cảnh đủ loại túi đựng chất đầy trong tủ nữa."
• 就这样集中收纳在一块。 = "Tất cả được sắp xếp gọn gàng trong cùng một hộp."
• 需要哪个就这么一转。 = "Cần loại nào chỉ việc xoay nhẹ một vòng."
• 可省事了。 = "Vừa nhanh vừa tiện."
• 它有六个收纳罐。 = "Hộp được chia thành 6 ngăn chứa riêng biệt."
• 平身都是透明的。 = "Thiết kế trong suốt từ trên xuống dưới."
• 想要哪个一目了然。 = "Nhìn một cái là biết ngay bên trong có gì."
• 而且它旋转起来真的超顺滑。 = "Cơ chế xoay rất mượt mà."
• 一点都不卡顿。 = "Hoàn toàn không bị kẹt hay giật."
• 密封性也很好。 = "Khả năng kín khí cũng rất tốt."
• 就不怕会受潮长虫。 = "Không lo ngũ cốc bị ẩm mốc hay mọc mọt."
• 开口的设计就很好。 = "Thiết kế miệng lấy thực phẩm rất tiện dụng."
• 控量也不会太多。 = "Dễ kiểm soát lượng lấy ra, tránh lãng phí."
• 不管放在台面上还是柜子里都可以。 = "Đặt trên bàn bếp hay trong tủ đều phù hợp."
• 家里的杯子怎么摆都是乱糟糕的。 = "Ly cốc trong nhà lúc nào cũng bày bừa lộn xộn."

• 你就像我安排这个杯架。 = "Bạn nên sắm ngay chiếc kệ để cốc này."

• 分层收纳的设计一目了然。 = "Thiết kế phân tầng giúp sắp xếp gọn gàng và dễ quan sát."

• 拿取超方便。 = "Lấy ra hay cất vào đều rất thuận tiện."

• 保温杯、玻璃杯、普通水杯全都能放。 = "Bình giữ nhiệt, cốc thủy tinh hay cốc uống nước thông thường đều để được."

• 波浪卡槽的设计特别稳当。 = "Các khe đỡ dạng gợn sóng giúp cố định cốc rất chắc chắn."

• 而且是多巴胺的配色。 = "Màu sắc phong cách Dopamine trẻ trung và bắt mắt."

• 不管放储柜还是台面。 = "Dù đặt trong tủ hay trên mặt bàn."

• 杯子都能摆放得整整齐齐。 = "Mọi loại cốc đều được sắp xếp ngăn nắp."

• 看着都特别舒服。 = "Nhìn vào cũng thấy gọn gàng và dễ chịu hơn."

• 最关键的是再也不用翻箱倒柜找杯子了。 = "Quan trọng nhất là không còn phải lục tung tủ kệ để tìm cốc
• 太厉害了。 = "Quá đỉnh luôn."

• 将报废的大力钳的钳嘴焊接到钢板上。 = "Hàn phần đầu của chiếc kìm cộng lực cũ lên một tấm thép."

• 接着装上螺丝。 = "Sau đó lắp bu lông vào."

• 用螺母把两块钳嘴固定起来。 = "Dùng đai ốc để cố định hai đầu lưỡi cắt lại với nhau."

• 然后再把摇杆手柄穿上螺丝固定上去。 = "Tiếp theo lắp tay đòn và cố định bằng bu lông."

• 再接着把连杆用螺丝固定安装到钳嘴和摇杆手柄上。 = "Gắn thanh truyền nối giữa đầu kìm và tay đòn bằng bu lông."

• 这样一台不用大力就可以把钢筋轻松剪断的大力钳就制作好了。 = "Như vậy là đã hoàn thành một chiếc kìm cắt thép trợ lực, giúp cắt thép rất nhẹ tay."

• 可以用来剪钢筋铁管。 = "Có thể dùng để cắt thép cây và ống sắt."

• 你们说这个发明怎么样？ = "Các bạn thấy phát minh này thế nào?"
• 太厉害了。 = "Quá đỉnh luôn."

• 像这样的钢管拼接在一起是焊接不上去的。 = "Những đoạn ống thép như thế này rất khó hàn nối trực tiếp với nhau."

• 可以把钢管用尺子按照视频中画线做记号再切割掉。 = "Hãy dùng thước đánh dấu theo đường cắt như trong video rồi cắt bỏ phần đã đánh dấu."

• 这样就可以完美无缝拼接焊接在一起了。 = "Sau đó các đoạn ống sẽ được ghép và hàn lại một cách khít đẹp, gần như không có khe hở."
• 真是个天才。 = "Đúng là một ý tưởng quá thông minh."

• 把螺母焊接到另一个方形螺母上。 = "Hàn một đai ốc vào một đai ốc vuông khác."

• 接着在螺母的侧面用电钻打孔。 = "Tiếp theo dùng máy khoan tạo một lỗ ở bên hông đai ốc."

• 然后再把打好的孔攻螺纹。 = "Sau đó taro ren cho lỗ vừa khoan."

• 最后再拧上一颗螺丝进去。 = "Cuối cùng vặn một con vít vào."

• 这样一个电池接线柱固定器就制作好了。 = "Như vậy là đã hoàn thành một bộ cố định cọc đấu dây bình ắc quy."

• 你们觉得这个发明好用吗？ = "Các bạn thấy phát minh này có hữu ích không?"
• 这才是桌面办公需要的神器。 = "Đây mới đúng là món đồ không thể thiếu cho dân văn phòng."

• 一架多用。 = "Một sản phẩm nhưng có thể sử dụng theo nhiều cách."

• 磁吸面可拆卸手提。 = "Mặt nam châm có thể tháo rời và mang theo dễ dàng."

• 展开托脚架平板。 = "Mở ra là có thể dùng làm giá đỡ máy tính bảng."

• 支架还能架笔记本。 = "Thậm chí còn có thể đỡ được cả laptop."

• 脚架带有自动弹簧架。 = "Chân đế được trang bị cơ chế lò xo tự động."

• 调节高度很适合。 = "Có thể điều chỉnh độ cao linh hoạt theo nhu cầu."

• 桌面办公时。 = "Khi làm việc tại bàn."

• 再也不用低头了。 = "Không còn phải cúi đầu nhìn màn hình nữa."

• 收纳起来小小一个。 = "Gấp gọn lại chỉ còn một kích thước rất nhỏ."

• 随身携带也很方便。 = "Mang theo bên mình cũng vô cùng tiện lợi."
• 哈哈哈哈。 = "Haha, buồn cười thật."

• 我要是不说哪里看得出是个纸巾盒啊。 = "Nếu tôi không nói thì chẳng ai nghĩ đây là một hộp đựng khăn giấy đâu."

• 还是卡皮巴拉和熊帽头的造型。 = "Lại còn được thiết kế theo hình Capybara đội mũ gấu siêu dễ thương."

• 试过超多种纸巾盒。 = "Tôi đã dùng thử rất nhiều loại hộp đựng khăn giấy."

• 还是这种磁吸开盖的罐子方便啊。 = "Nhưng loại nắp nam châm mở đóng như thế này vẫn tiện nhất."

• 抽纸湿滑不卡顿。 = "Rút khăn rất mượt, không bị kẹt hay kéo rối."

• 摆在桌子上啊。 = "Đặt trên bàn nhìn rất đẹp mắt."

• 精致又百搭。 = "Vừa tinh tế vừa dễ phối với mọi không gian."

• 随便一放都好看啊。 = "Đặt ở đâu cũng thấy xinh xắn."
• 真是个天才啊。 = "Đúng là một ý tưởng quá thông minh."

• 把自行车的车头钢管切割下来。 = "Cắt lấy phần ống cổ lái của một chiếc xe đạp cũ."

• 固定到瓶口前上。 = "Sau đó cố định vào phần đầu của dụng cụ."

• 接着将轴承外圈焊接到钢管上。 = "Tiếp theo hàn vòng ngoài của ổ bi lên ống thép."

• 然后再把轴承外圈切割掉留下一半。 = "Sau đó cắt bỏ một nửa vòng ngoài của ổ bi."

• 再然后就可以将链条的螺纹焊到钢管里面。 = "Tiếp tục hàn đoạn xích có ren vào bên trong ống thép."

• 再将链条焊接到轴承外圈上。 = "Rồi hàn phần xích đó vào vòng ngoài của ổ bi."

• 这样就可以把钢管放平。 = "Như vậy là có thể đặt ống thép nằm ngang để sử dụng."

• 再把螺母拧到螺纹上。 = "Cuối cùng vặn đai ốc vào phần ren đã chuẩn bị."

• 这样一个机油滤芯拆卸神器就制作好了。 = "Vậy là đã hoàn thành dụng cụ tháo lọc dầu động cơ."

• 可以用它来拆卸安装发动机的机油滤芯。 = "Dụng cụ này có thể dùng để tháo lắp lọc dầu trên động cơ."

• 真是太好用了。 = "Thực sự rất tiện dụng."
• 剥线就能发家致富的方法。 = "Đây là một mẹo giúp bóc dây điện nhanh và hiệu quả hơn."

• 首先。 = "Đầu tiên."

• 要自己动手像废品站老板一样。 = "Hãy tự chế một dụng cụ giống như những người thu mua phế liệu chuyên nghiệp."

• 用合页自制出一把锋利的剥线小工具。 = "Dùng một chiếc bản lề để làm thành dụng cụ tuốt dây điện sắc bén."

• 回收来的电线就这样去剥皮。 = "Những cuộn dây điện thu gom về chỉ cần dùng dụng cụ này để tách lớp vỏ."

• 取铜线。 = "Lấy phần lõi đồng bên trong ra."

• 不用剪刀。 = "Không cần dùng kéo."

• 快几倍。 = "Tốc độ nhanh hơn gấp nhiều lần."

• 一个人顶几个。 = "Một người làm việc bằng vài người."

• 想不发家致富都难。 = "Làm việc hiệu quả thế này thì khó mà không kiếm thêm được tiền."
• 自动封盖的鸡蛋盒。 = "Hộp đựng trứng tự động đóng nắp."

• 每一层拿掉最后一个鸡蛋。 = "Khi lấy quả trứng cuối cùng ở một tầng,"

• 它就会自动翻上去。 = "ngăn phía trên sẽ tự động lật xuống."

• 容量大。 = "Dung tích lớn."

• 可以装30个鸡蛋。 = "Có thể chứa tới 30 quả trứng."

• 它的材质很厚实。 = "Chất liệu dày dặn và chắc chắn."

• 尺寸设计得很合理。 = "Kích thước được thiết kế rất hợp lý."

• 可以放到冰箱侧门。 = "Có thể đặt vừa vặn ở cánh cửa tủ lạnh."

• 正好利用了侧边的空间。 = "Tận dụng hiệu quả không gian bên hông tủ."

• 吃完一层拿一层。 = "Dùng hết một tầng thì lấy tiếp tầng kế tiếp."

• 特别方便。 = "Cực kỳ tiện lợi."

• 而且放在厨房的角落也可以。 = "Ngoài ra cũng có thể đặt gọn ở góc bếp."
• 太厉害了。 = "Thật là một ý tưởng tuyệt vời."

• 将钢管焊接到轴承制作的底座上固定起来。 = "Hàn một đoạn ống thép vào phần đế được làm từ vòng bi để cố định lại."

• 接着在大钢管上焊接上一条小钢管。 = "Tiếp theo hàn thêm một đoạn ống thép nhỏ lên ống thép lớn."

• 然后再清理掉焊接留下的焊渣。 = "Sau đó làm sạch phần xỉ hàn còn sót lại."

• 这样一个切割片放置工具就制作好了。 = "Như vậy là đã hoàn thành một giá đỡ để cất giữ lưỡi cắt."

• 你们说这个发明怎么样? = "Các bạn thấy phát minh này có hữu ích không?"
• 太厉害了。 = "Quá thông minh rồi."

• 将钢管焊接到轴承制作的底座上固定起来。 = "Đầu tiên hàn một đoạn ống thép lên phần đế được chế tạo từ vòng bi để cố định chắc chắn."
• 接着在大钢管上焊接上一条小钢管。 = "Tiếp theo hàn thêm một đoạn ống thép nhỏ lên thân ống thép lớn."
• 然后再清理掉焊接留下的焊渣。 = "Sau đó vệ sinh sạch phần xỉ hàn còn lại sau khi gia công."
• 这样一个切割片放置工具就制作好了。 = "Như vậy là đã hoàn thành một giá đỡ chuyên dùng để cất giữ các lưỡi cắt."
• 你们说这个发明怎么样? = "Các bạn thấy phát minh này có hay không?"
• 嫂子。 = "Chị dâu."
• 你好香啊。 = "Chị thơm quá."
• 张建国。 = "Trương Kiến Quốc."
• 你怎么在这里。 = "Sao anh lại ở đây?"
• 我不是被冻死了吗。 = "Chẳng phải tôi đã chết cóng rồi sao?"
• 求求你们。 = "Xin mọi người."
• 你们不要赶我走。 = "Đừng đuổi tôi đi."
• 我23岁嫁到张家。 = "Tôi gả vào nhà họ Trương từ năm 23 tuổi."
• 这20年里没有功劳也有苦劳。 = "Hai mươi năm qua, dù không có công lao lớn thì cũng có biết bao vất vả."
• 大嫂 早几年看你长得俊巧。 = "Chị dâu, mấy năm trước chị xinh đẹp biết bao."
• 那时候跟了我多好。 = "Giá như lúc đó chị theo tôi thì tốt biết mấy."
• 现在大哥死了。 = "Giờ anh cả đã mất rồi."
• 你又是一个半摊吃白饭。 = "Chị lại trở thành người ăn không ngồi rồi trong nhà."
• 不赶你赶谁。 = "Không đuổi chị thì đuổi ai?"
• 我们张家可没多余的娘娘。 = "Nhà họ Trương chúng tôi không nuôi người vô dụng."
• 你跟他说这些干啥。 = "Nói những chuyện đó với cô ta làm gì?"
• 赶紧滚 多亏一切。 = "Cút ngay đi."
• 妈 这二十年我洗衣做饭下地干活。 = "Mẹ à, suốt hai mươi năm qua con giặt giũ, nấu cơm và làm việc đồng áng."
• 连过娶媳妇儿 侄子上大学都是我出钱出力啊。 = "Ngay cả việc cưới vợ cho em trai và cho cháu đi học đại học cũng đều do con bỏ tiền bỏ sức."
• 这种房子我也可以不要。 = "Căn nhà này con cũng có thể không cần."
• 只要你不看我走。 = "Chỉ cần mọi người để con rời đi."
• 我呸。 = "Phi!"
• 你生个丫头骗子。 = "Mày chỉ sinh ra một đứa con gái vô dụng."
• 我儿子才三十五岁。 = "Con trai tao mới ba mươi lăm tuổi."
• 就被你和那个骚把刑克死了。 = "Vậy mà bị mày và con hồ ly tinh kia hại chết."
• 你还有脸说侧侯这个家。 = "Mày còn mặt mũi nhắc đến chuyện cống hiến cho cái nhà này sao?"
• 真不要脸。 = "Đúng là không biết xấu hổ."
• 难不成 我重生了。 = "Chẳng lẽ tôi đã sống lại?"
• 重生到了二十年前被小数字逼见的那个晚上。 = "Tôi đã sống lại đúng vào đêm hai mươi năm trước, khi bị cả nhà ép đến đường cùng."
• 太厉害了。 = "Đúng là một mẹo quá hay.
• 维修水泵遇到螺丝损坏打滑拆不出来不要把水泵直接扔掉。 = "Khi sửa máy bơm nước mà gặp ốc vít bị tòe ren, trượt ren không tháo ra được thì đừng vội bỏ cả máy bơm."
• 可以先给螺帽焊接上一颗钢珠。 = "Hãy hàn một viên bi thép lên đầu con ốc bị hỏng trước."
• 接着再把一颗螺母焊接到钢珠上。 = "Sau đó hàn thêm một đai ốc vào viên bi thép đó."
• 然后再把机油倒在螺母里面给螺母降温。 = "Tiếp theo đổ dầu máy vào để làm nguội phần đai ốc vừa hàn."
• 这样。 = "Làm như vậy là được.
• 再用套筒拧螺母再把打滑的螺丝带出来。 = "Cuối cùng dùng khẩu tuýp vặn đai ốc để kéo luôn con ốc bị trượt ren ra ngoài."
• 真是太聪明了。 = "Đúng là một cách xử lý rất thông minh."
• 太厉害了。 = "Đúng là một ý tưởng quá hay."
• 将钢筋焊接到轴承钢圈上。 = "Hàn một thanh thép lên vòng ngoài của ổ bi."
• 然后在钢筋上面焊接上齿轮。 = "Sau đó hàn một bánh răng lên thanh thép vừa lắp."
• 接着再焊接一条掰弯的钢筋。 = "Tiếp tục hàn thêm một thanh thép đã được uốn cong."
• 这样一台轴承卡死提取神器就制作好了。 = "Như vậy là đã hoàn thành dụng cụ chuyên dùng để tháo ổ bi bị kẹt."
• 可以用来提取卡死的发动机轴承。 = "Dụng cụ này có thể dùng để tháo các ổ bi bị kẹt trên động cơ."
• 你们说实用吗? = "Các bạn thấy phát minh này có hữu ích không?"
• 嗨。 = "Chào."

• 嫂子。 = "Chị dâu."

• 你好香啊。 = "Chị thơm quá."

• 张静谷。 = "Trương Tĩnh Cúc."

• 你怎么在这里? = "Sao anh lại ở đây?"

• 我不是被冻死了吗? = "Chẳng phải tôi đã chết cóng rồi sao?"

• 求求你们。 = "Xin mọi người."

• 你们不要赶我走。 = "Đừng đuổi tôi đi."

• 我23岁嫁到张家。 = "Tôi gả vào nhà họ Trương từ năm 23 tuổi."

• 这20年里没有功劳也有苦劳。 = "Hai mươi năm qua, dù không có công lao lớn thì cũng đã bỏ ra biết bao vất vả."

• 大嫂 早几年看你长得俊巧。 = "Chị dâu, mấy năm trước chị xinh đẹp và duyên dáng biết bao."

• 那时候跟了我多好。 = "Giá mà khi đó chị theo tôi thì tốt biết mấy."

• 现在大哥死了。 = "Bây giờ anh cả đã mất rồi."

• 你又是一个白吃白喝的人。 = "Giờ chị chỉ là người ăn không ngồi rồi trong nhà."

• 不赶你赶谁。 = "Không đuổi chị thì đuổi ai?"

• 我们张家可没多余的粮食养闲人。 = "Nhà họ Trương chúng tôi không nuôi người vô dụng."

• 你跟他说这些干啥。 = "Nói những điều đó với cô ta làm gì?"

• 赶紧滚。 = "Cút đi ngay."

• 妈 这二十年我洗衣做饭下地干活。 = "Mẹ à, suốt hai mươi năm qua con giặt giũ, nấu cơm và làm việc đồng áng."

• 连过去娶媳妇儿 侄子上大学。 = "Ngay cả chuyện cưới vợ cho người trong nhà và cho cháu đi học đại học."

• 都是我出钱出力啊。 = "Đều là con bỏ tiền và bỏ công sức."

• 这种房子我也可以不要。 = "Ngôi nhà này con cũng có thể không cần."

• 只要你不赶我走。 = "Chỉ cần mẹ đừng đuổi con đi."

• 我呸。 = "Phi!"

• 你生个丫头骗子。 = "Mày đúng là đồ lừa đảo."

• 我儿子才三十五岁。 = "Con trai tao mới chỉ 35 tuổi."

• 就被你和那个贱女人害死了。 = "Đã bị mày và con đàn bà đó hại chết."

• 你还有脸说伺候这个家。 = "Mày còn mặt mũi nào nhắc đến chuyện chăm lo cho cái nhà này?"

• 真不要脸。 = "Đúng là không biết xấu hổ."

• 难不成我重生了。 = "Chẳng lẽ tôi đã sống lại?"

• 重生到了二十年前被逼走的那个晚上。 = "Tôi đã sống lại đúng vào đêm bị ép rời khỏi nhà cách đây hai mươi năm."
• 太厉害了。 = "Đúng là một ý tưởng quá hay."

• 把角钢焊接上铁棒和锁止机构制作成门锁。 = "Hàn thanh thép góc với thanh sắt và cơ cấu chốt khóa để chế tạo một bộ khóa cửa."

• 接着再把制作好的门锁焊接到门上。 = "Sau đó hàn bộ khóa vừa chế tạo lên cánh cửa."

• 这样一个简易实用的门锁就制作好了。 = "Như vậy là đã hoàn thành một chiếc khóa cửa đơn giản nhưng rất thực dụng."

• 你们说这个门锁怎么样? = "Các bạn thấy chiếc khóa cửa này có hữu ích không?"
• 机油盖你能找到吧？ = "Nắp châm dầu động cơ thì chắc bạn tìm được chứ?"

• 那你就能找到旁边这个长4厘米、宽3厘米的黑色小零件。 = "Vậy thì bạn cũng sẽ thấy bộ phận nhỏ màu đen bên cạnh, dài khoảng 4cm và rộng khoảng 3cm."

• 你手不自觉地拔掉了它的插头。 = "Bạn rút giắc cắm của nó ra."

• 接着你去大舅那儿借了一个10件套套筒工具。 = "Sau đó đi mượn một bộ tuýp 10 món để tháo lắp."

• 把它上面的固定螺丝拆了下来。 = "Tháo con ốc cố định trên bộ phận đó ra."

• 双手不停地左右晃动。 = "Lắc nhẹ qua lại bằng hai tay."

• 一不小心把它抖了出来。 = "Chỉ một lúc là có thể rút nó ra ngoài."

• 左眼一睁，就能发现它上面全是油泥和积碳。 = "Lúc này sẽ thấy bề mặt của nó bám đầy cặn dầu và muội bẩn."

• 已经把滤网完全堵死了。 = "Lưới lọc bên trong gần như đã bị tắc hoàn toàn."

• 你又一个不小心。 = "Tiếp theo bạn lấy dung dịch vệ sinh chuyên dụng."

• 在我的橱窗里面买来了一瓶化清剂。 = "Xịt dung dịch làm sạch lên toàn bộ bề mặt."

• 朝着它上面就是一顿猛喷。 = "Phun kỹ vào các khe và vị trí bám bẩn."

• 没想到清洗效果超乎想象。 = "Không ngờ hiệu quả làm sạch lại tốt đến vậy."

• 油泥和积碳都被溶解下来了。 = "Cặn dầu và các chất bẩn đều được hòa tan."

• 滤网都全部露出来了。 = "Lưới lọc sạch sẽ và hiện rõ trở lại."

• 清洗干净以后。 = "Sau khi vệ sinh xong."

• 用纸巾擦一擦。 = "Dùng khăn giấy lau khô."

• 恢复了原厂状态。 = "Bộ phận này gần như trở lại trạng thái ban đầu."

• 聪明的你又把它装了回去。 = "Sau đó lắp lại đúng vị trí ban đầu."

• 那么恭喜你。 = "Xin chúc mừng."

• 原来你的车发动机。 = "Những hiện tượng trên xe của bạn như:"

• 冷车热车启动困难。 = "Khó khởi động khi máy nguội hoặc máy nóng."

• 动力下降。 = "Động cơ yếu, giảm công suất."

• 油耗飙升的故障情况。 = "Mức tiêu hao nhiên liệu tăng cao."

• 居然全部消失了。 = "Đều có thể được cải thiện đáng kể."

• 你一脸开心地跑过来。 = "Lúc đó bạn sẽ rất vui."

• 给我点了一个小心心。 = "Và nhớ ủng hộ mình một lượt thích nhé."

• 才知道它是VVT电磁阀。 = "Thì ra bộ phận này chính là van điện từ VVT."

• 它的作用是通过发动机电脑控制它通断机油。 = "Nó được ECU điều khiển để kiểm soát dòng dầu động cơ."

• 驱动VVT机构改变气门正时。 = "Từ đó điều chỉnh thời điểm đóng mở xupap."

• 让车子怠速稳、提速快、更省油。 = "Giúp xe chạy êm hơn, tăng tốc tốt hơn và tiết kiệm nhiên liệu hơn."

• 如果它脏了或者坏了。 = "Nếu bộ phận này bị bẩn hoặc hư hỏng."

• 就会出现发动机异响。 = "Động cơ có thể xuất hiện tiếng kêu bất thường."

• 油耗升高，动力下降。 = "Xe sẽ hao nhiên liệu hơn và bị giảm công suất."

• 启动困难的故障现象。 = "Đồng thời gây ra hiện tượng khó khởi động."

• 看到这里的大哥大姐能不能给我点一个关注。 = "Nếu thấy video hữu ích, hãy cho mình một lượt theo dõi nhé."

• 我会每天讲解汽车知识传递正能量视频。 = "Mình sẽ tiếp tục chia sẻ kiến thức ô tô hữu ích mỗi ngày."
• 真是个天才啊。 = "Đúng là một thiên tài."

• 在卡死的轴承上打上热熔胶。 = "Bôi keo nóng chảy lên vòng bi đang bị kẹt."

• 接着用热熔胶给轴承高温加热。 = "Tiếp tục dùng keo nóng chảy để gia nhiệt cho vòng bi."

• 让轴承热胀冷缩。 = "Làm cho vòng bi giãn nở khi nóng và co lại khi nguội."

• 然后再将螺纹杆和螺丝刀插到轴承里卡住轴承。 = "Sau đó đưa thanh ren và tua vít vào bên trong để giữ chặt vòng bi."

• 最后再用电钻转动螺纹杆把轴承提取出来。 = "Cuối cùng dùng máy khoan xoay thanh ren để kéo vòng bi bị kẹt ra ngoài."

• 真是太好用了。 = "Cách này thực sự rất hiệu quả và tiện lợi."
• 太厉害了。 = "Đúng là quá thông minh."

• 螺丝生锈腐蚀不要怕。 = "Đừng lo khi gặp những con ốc bị gỉ sét hoặc ăn mòn."

• 可以先把原来的螺帽喷上除锈剂。 = "Trước tiên hãy xịt dung dịch tẩy gỉ lên phần ốc cũ."

• 用钢丝刷打磨干净。 = "Dùng bàn chải sắt chà sạch lớp gỉ bám bên ngoài."

• 接着再用打磨机把螺帽最上面一层生锈打磨掉。 = "Sau đó dùng máy mài loại bỏ lớp gỉ sét ở phần trên cùng của đai ốc."

• 然后把螺母放上去。 = "Tiếp theo đặt một đai ốc mới lên vị trí đó."

• 再然后就可以在螺母里面倒上吸热液。 = "Sau đó đổ dung dịch làm mát hoặc hấp thụ nhiệt vào bên trong đai ốc."

• 接着等吸热液冷却就可以用扳手卡住螺母。 = "Đợi cho đai ốc nguội xuống rồi dùng cờ lê kẹp chặt vào."

• 这样就可以把断掉的螺丝拧出来了。 = "Như vậy có thể dễ dàng vặn phần ốc bị gãy ra ngoài."

• 你们觉得这个维修小技巧怎么样？ = "Các bạn thấy mẹo sửa chữa này có hữu ích không?"
• 机油盖你能找到吧？ = "Nắp châm dầu động cơ thì chắc bạn tìm được chứ?"

• 那你就能找到旁边这个长4厘米、宽3厘米的黑色小零件。 = "Ngay bên cạnh đó sẽ có một chi tiết nhỏ màu đen, dài khoảng 4cm và rộng khoảng 3cm."

• 你手不自觉地拔掉了它的插头。 = "Hãy rút giắc cắm của bộ phận này ra."

• 接着你去大舅那儿借了一个10件套套筒工具。 = "Sau đó chuẩn bị một bộ tuýp hoặc dụng cụ tháo lắp phù hợp."

• 把它上面的固定螺丝拆了下来。 = "Tháo con ốc cố định phía trên ra."

• 双手不停地左右晃动。 = "Lắc nhẹ qua lại bằng hai tay."

• 一不小心把它抖了出来。 = "Rồi từ từ rút bộ phận đó ra ngoài."

• 左眼一睁，就能发现它上面全是油泥和积碳。 = "Lúc này bạn sẽ thấy nó bám đầy cặn dầu và muội bẩn."

• 已经把滤网完全堵完了。 = "Lưới lọc bên trong gần như đã bị tắc kín."

• 你又一个不小心。 = "Tiếp theo hãy lấy dung dịch vệ sinh chuyên dụng."

• 在我的橱窗里面买来了一瓶化清剂。 = "Chuẩn bị một chai dung dịch làm sạch chi tiết động cơ."

• 朝着它上面就是一顿猛喷。 = "Xịt trực tiếp lên toàn bộ bề mặt cần vệ sinh."

• 没想到清洗效果超乎想象。 = "Hiệu quả làm sạch thật sự vượt ngoài mong đợi."

• 油泥和积碳都被溶解下来了。 = "Các lớp cặn dầu và chất bẩn được hòa tan nhanh chóng."

• 滤网都全部露出来了。 = "Phần lưới lọc sạch sẽ và hiện rõ trở lại."

• 清洗干净以后。 = "Sau khi vệ sinh sạch hoàn toàn."

• 用纸巾擦一擦。 = "Dùng khăn giấy lau khô."

• 恢复了原厂状态。 = "Bộ phận này gần như trở lại trạng thái như mới."

• 聪明的你又把它装了回去。 = "Sau đó lắp lại đúng vị trí ban đầu."

• 那么恭喜你。 = "Và xin chúc mừng."

• 原来你的车发动机。 = "Những hiện tượng khó chịu trên động cơ như:"

• 冷车热车启动困难。 = "Khó khởi động khi máy nguội hoặc máy nóng."

• 动力下降。 = "Xe bị giảm công suất."

• 油耗飙升的故障情况。 = "Mức tiêu hao nhiên liệu tăng cao."

• 居然全部消失了。 = "Đều có thể được cải thiện đáng kể."

• 你一脸开心地跑过来。 = "Lúc đó chắc chắn bạn sẽ rất hài lòng."

• 给我点了一个小心心。 = "Và đừng quên để lại một lượt yêu thích nhé."

• 才知道它是VVT电磁阀。 = "Thì ra bộ phận này chính là van điện từ VVT."

• 它的作用是通过发动机电脑控制它通断机油。 = "Nó được ECU điều khiển để kiểm soát dòng dầu đi qua hệ thống."

• 驱动VVT机构改变气门正时。 = "Qua đó điều chỉnh thời điểm đóng mở xupap của động cơ."

• 让车子怠速稳、提速快、更省油。 = "Giúp xe chạy êm hơn, tăng tốc tốt hơn và tiết kiệm nhiên liệu hơn."

• 如果它脏了或者坏了。 = "Nếu bộ phận này bị bẩn hoặc hư hỏng."

• 就会出现发动机异响。 = "Động cơ có thể xuất hiện tiếng kêu bất thường."

• 油耗升高，动力下降。 = "Mức tiêu hao nhiên liệu tăng và công suất giảm."

• 启动困难的故障现象。 = "Đồng thời gây hiện tượng khó khởi động."

• 看到这里的大哥大姐能不能给我点一个关注。 = "Nếu thấy hữu ích, hãy cho mình một lượt theo dõi nhé."

• 我会每天讲解汽车知识传递正能量视频。 = "Mình sẽ tiếp tục chia sẻ kiến thức ô tô hữu ích mỗi ngày."
• 太厉害了。 = "Đúng là quá thông minh."

• 摩托车启动器打滑不用换。 = "Bộ khởi động xe máy bị trượt cũng không cần thay mới."

• 把启动器拆下来再把转轴切割开缝隙。 = "Tháo bộ khởi động ra rồi cắt một rãnh nhỏ trên trục quay."

• 接着把脚踏装回去用电片插进去。 = "Sau đó lắp lại cần đạp và chèn một miếng thép mỏng vào khe vừa cắt."

• 卡住在拧上螺丝固定防止打滑。 = "Siết chặt bằng ốc vít để cố định, giúp chống trượt hiệu quả."

• 你们说这个维修方法如何? = "Các bạn thấy cách sửa chữa này có hữu ích không?"
• 淘米水千万别倒了。 = "Nước vo gạo đừng vội đổ đi."

• 它可还有大用处。 = "Nó vẫn còn rất nhiều công dụng hữu ích."

• 倒上白醋。 = "Cho thêm một ít giấm trắng."

• 再来一勺小苏打。 = "Tiếp theo cho một muỗng baking soda."

• 家里抹布用时间长了。 = "Khăn lau trong nhà dùng lâu ngày."

• 油油腻腻洗不干净。 = "Bám đầy dầu mỡ, rất khó giặt sạch."

• 还有一股怪味。 = "Lại còn có mùi khó chịu."

• 你把它放进去。 = "Hãy cho chiếc khăn vào dung dịch đó."

• 浸泡十来分钟。 = "Ngâm khoảng mười phút."

• 然后再轻轻揉一揉搓一搓。 = "Sau đó vò và chà nhẹ vài lần."

• 抹布这样洗完跟新的一样。 = "Giặt theo cách này, chiếc khăn sẽ sạch như mới."

• 还没有异味。 = "Đồng thời cũng không còn mùi hôi nữa."
• 咱们烧水的水壶总是有水垢该怎么办。 = "Ấm đun nước trong nhà bị bám cặn vôi thì phải làm sao?"

• 今天学会一招。 = "Hôm nay mình sẽ chia sẻ cho các bạn một mẹo đơn giản."

• 非常地简单。 = "Cách làm cực kỳ dễ."

• 在水壶中加上生姜和鸡蛋壳。 = "Cho vài lát gừng và vỏ trứng vào trong ấm."

• 再倒一点白醋。 = "Sau đó đổ thêm một ít giấm trắng."

• 加一点自来水。 = "Tiếp theo cho nước sạch vào."

• 把水烧开后静置15分钟倒掉就可以了。 = "Đun sôi nước rồi để yên khoảng 15 phút, sau đó đổ bỏ."

• 这样水壶很长时间都不会再生水垢了。 = "Làm như vậy, ấm nước sẽ lâu ngày không còn bị đóng cặn nữa."
• 真是个天才。 = "Đúng là một thiên tài."

• 像这样的大钢管如何快速无缝焊接到两条小钢管上。 = "Làm thế nào để hàn nhanh một ống thép lớn vào hai ống thép nhỏ mà không bị hở mối nối?"

• 先用尺子给大钢管画两条直线。 = "Trước tiên dùng thước kẻ hai đường thẳng trên ống thép lớn."

• 接着再沿着画好的线做标记。 = "Tiếp theo đánh dấu theo các đường đã vẽ."

• 然后再把做好标记的钢管切割掉。 = "Sau đó cắt bỏ phần ống thép đã được đánh dấu."

• 这样就可以完美无缝地把大钢管和小钢管焊接在一起了。 = "Như vậy có thể hàn ống thép lớn và ống thép nhỏ lại với nhau một cách khít và đẹp mắt."

• 你们说怎么样？ = "Các bạn thấy cách này thế nào?"
• 在轮胎破损的位置穿一些绑带。 = "Luồn vài sợi dây rút nhựa qua vị trí lốp bị rách."

• 用火把绑带头完全烤融化。 = "Dùng nhiệt làm nóng phần đầu dây rút cho nhựa nóng chảy hoàn toàn."

• 然后再用刀片将其刮平。 = "Sau đó dùng dao gạt phẳng bề mặt vừa nóng chảy."

• 紧接着再次加热。 = "Tiếp tục gia nhiệt thêm một lần nữa."

• 使其和车胎完全融为一体。 = "Để phần nhựa hòa quyện hoàn toàn với bề mặt lốp."

• 最后再喷一些溶剂。 = "Cuối cùng phun thêm một lớp dung dịch xử lý bề mặt."

• 便可修复破损的车胎。 = "Như vậy là có thể khắc phục phần lốp bị hư hỏng."
• 很多人往这么小的方管里面安装螺母。 = "Nhiều người gặp khó khăn khi lắp đai ốc vào bên trong những thanh thép hộp có kích thước nhỏ như thế này."

• 手塞不进去的时候。 = "Khi không thể đưa tay vào bên trong."

• 钳子也够不着。 = "Ngay cả kìm cũng không với tới được."

• 那就可以用这个小妙招。 = "Lúc đó bạn có thể áp dụng mẹo nhỏ này."

• 在螺丝刀头上放一根螺栓。 = "Đặt một con bu lông lên đầu tua vít."

• 将其固定到上面。 = "Cố định nó chắc chắn vào đầu tua vít."

• 再卡住需要拧的螺母。 = "Sau đó giữ lấy đai ốc cần lắp."

• 这样就算再小的空间。 = "Nhờ vậy, dù không gian có chật hẹp đến đâu."

• 也能快速拧上。 = "Bạn vẫn có thể lắp và siết đai ốc một cách nhanh chóng."

• 非常方便。 = "Cực kỳ tiện lợi."
这具对时1996年口碑炸裂的经典科幻巨智, = "Đây là siêu phẩm khoa học viễn tưởng kinh điển năm 1996,"
以7500万美元的制作成本, = "Với kinh phí 75 triệu USD,"
黄蓝全球8.17亿美元票房, = "Thu về 817 triệu USD toàn cầu,"
一举登顶当年全球票房绑手, = "Đứng đầu doanh thu toàn cầu năm đó,"
铸救了影史上一段不可复制的商业审划。 = "Tạo nên huyền thoại thương mại độc nhất lịch sử điện ảnh."
它拉开了外星入侵题材的黄金时代, = "Mở ra kỷ nguyên vàng cho thể loại xâm lược ngoài hành tinh,"
用一连串震撼镜头刻下了永恒烙印。 = "Loạt cảnh ấn tượng đã để lại dấu ấn vĩnh cửu."
其中剧情光束轰然摧毁核心地标建筑的震撼画面, = "Cảnh tia sáng hủy diệt các tòa nhà biểu tượng gây chấn động,"
至今仍是在那天领域争相致敬的标杆镜头。 = "Đến nay vẫn là cảnh quay chuẩn mực được nhiều phim học hỏi."
這不做品用鐵一般的實力證明 = "Tác phẩm này đã chứng minh sức mạnh vượt trội."
真正的經典既能稱霸票房榜 = "Kinh điển thực sự không chỉ thống trị phòng vé"
也能無懼實況沖刷 = "mà còn vượt thời gian"
化作整整一代人共同的一幕記憶 = "thành ký ức chung của cả thế hệ."
今天就讓我們一起深度拆解這部不朽科幻災難大片 = "Hôm nay, hãy cùng phân tích siêu phẩm khoa học viễn tưởng bất hủ này:"
獨立日 = "Ngày Độc Lập."
巫視開篇 = "Mở đầu,"
全世界技術陷入巨型外星造物籠罩的無盡黃孔之中 = "Thế giới chìm trong hoảng loạn dưới bóng vật thể ngoài hành tinh."
一艘直定近60公里的巨型外星巨劍衝破天際 = "Phi thuyền khổng lồ, đường kính gần 60km, xé toạc bầu trời,"
闖入地球大氣層 = "Xâm nhập khí quyển Trái Đất,"
漫天裂火籠罩整片天空 = "Lửa dữ bao trùm cả bầu trời."
從未見過這般震撼意象的民眾瞬間慌亂不已 = "Người dân chưa từng thấy cảnh này lập tức hoảng loạn."
城市街道上車輛接連相當, = "Xe cộ trên phố va chạm liên tiếp."
望著這天必日的龐大外星飛船 = "Nhìn phi thuyền khổng lồ che kín bầu trời,"
眾人全都帶領原地, = "Mọi người đều đứng sững,"
反應過來後紛紛瘋狂逃竄 = "Định thần lại, họ điên cuồng bỏ chạy."
整座城市瞬間陷入巨大恐慌 = "Cả thành phố chìm trong hoảng loạn tột độ."
想關高層敘述開啟全網公開講話, = "Giới chức nhanh chóng phát biểu công khai,"
盡力安撫黃恐怖案的大眾 = "Cố trấn an công chúng đang sợ hãi."
可面對這般氣勢龐薄的外星戰艦 = "But trước chiến hạm ngoài hành tinh hùng vĩ,"
恐懼早已四處蔓延, = "Nỗi sợ đã lan rộng khắp nơi."
不少人急忙收拾行囊 = "Nhiều người vội vã dọn hành lý,"
打算遠離危險地帶 = "Định rời khỏi vùng nguy hiểm."
房間流傳多年的外星萊克傳聞就此成真 = "Tin đồn về người ngoài hành tinh bấy lâu nay đã thành sự thật."
人群之中心態各不相同, = "Tâm lý đám đông có sự khác nhau,"
有人慌亂逃命 = "Kẻ hoảng loạn bỏ chạy,"
也有部分人毫不畏懼, = "Nhưng một số người không hề sợ hãi,"
聚集在高樓天灘, = "Họ tụ tập trên các sân thượng tòa nhà chọc trời,"
高距標派迎接天外生靈降臨 = "Giơ biểu ngữ chào đón sinh vật ngoài hành tinh."
計算及工程師大衛友人破譯出隱藏在衛星信號裡的秘密訊息 = "Lúc này, kỹ sư David Levinson giải mã thông điệp bí mật ẩn trong tín hiệu vệ tinh,"
這才得知降臨地球的外星艦船艘足足有實物搜之多 = "Mới biết chiến hạm đổ bộ Trái Đất có tới hơn mười chiếc,"
技術停靠在全球各大繁華城市上空 = "Chúng đã vào vị trí ngay trên các thành phố lớn sầm uất toàn cầu."
一旦他們完成全部部署, = "Một khi hoàn tất việc triển khai,"
就會對整片大地發起全面攻勢 = "Chúng sẽ tấn công toàn diện Trái Đất."
既是人類賴以生存的家園文明 = "Nền văn minh và quê hương của nhân loại"
或將迎來滅頂之災, = "Đứng trước nguy cơ bị diệt vong."
大衛心急如焚 = "David nóng lòng như lửa đốt,"
立刻帶的家人趕往核心指揮重地 = "Anh lập tức đưa gia đình đến trung tâm chỉ huy đầu não,"
想要把這場巨大危機如實上報, = "Muốn báo cáo khẩn cấp nguy cơ cực lớn này,"
可惜一切都為十一晚 = "Nhưng mọi thứ dường như đã quá muộn."
地面防衛力量抱著友好溝通的想法 = "Lực lượng phòng vệ mặt đất muốn giao tiếp hòa bình,"
派出專用通航飛行器靠近外星巨劍 = "Cử máy bay chuyên dụng tiếp cận phi thuyền ngoài hành tinh,"
試圖藉助光影訊號建立交流 = "Thử dùng tín hiệu ánh sáng để thiết lập liên lạc."
可這份滿槍善意並未得到回應 = "Nhưng thiện chí của loài người không được đáp lại,"
外星戰艦直接發射高能激光 = "Chiến hạm ngoài hành tinh bắn thẳng tia laser năng lượng cao,"
瞬間將飛行器徹底摧毀, = "Phá hủy hoàn toàn chiếc máy bay trong chớp mắt,"
敵意展露無遺 = "Lộ rõ sự thù địch."
人類與天外文明的首次大戰正式拉開帷幕 = "Cuộc chiến đầu tiên giữa nhân loại và nền văn minh ngoài hành tinh chính thức bắt đầu."
外星飞船底部舱门环环开启, = "Cửa khoang đáy của phi thuyền khổng lồ từ từ mở ra,"
幽兰色强光铺满大地, = "Ánh sáng xanh tím bao phủ mặt đất,"
一道毁灭性光著著然落下 = "Một luồng sáng hủy diệt giáng xuống."
树千尺高的摩天楼与如同纸片般轟然倒塌, = "Những tòa nhà chọc trời cao hàng nghìn thước đổ sập như giấy,"
熊熊烈火如同潮水般席卷整座城市 = "Biển lửa dữ dội như sóng thần càn quét qua thành phố."
强悍的冲击波似已摧毁周遭一切, = "Sóng xung kích cực mạnh phá hủy mọi thứ xung quanh,"
人们拼尽全力四处逃亡, = "Người dân cố gắng chạy trốn bằng mọi giá,"
却依旧难以躲过灾难降临 = "Nhưng vẫn khó lòng thoát khỏi thảm họa kinh hoàng."
有人奮力掙扎求生, = "Có người vật lộn để tìm sự sống,"
有人滿心絕望坦然接受 = "Có người lại tuyệt vọng chấp nhận số phận."
緊接著又遇到強力激光轟擊核心指揮建築中心 = "Ngay sau đó, tia laser cực mạnh nhắm thẳng vào trung tâm chỉ huy đầu não,"
建築當場畫子飛揮 = "Tòa nhà lập tức bị san phẳng thành tro bụi."
極右專屬應急專機腳性躲過衝擊 = "Chiếc chuyên cơ khẩn cấp của Tổng thống may mắn thoát nạn,"
倉皇緊急撤離險境, = "Vội vã sơ tán khỏi vùng nguy hiểm."
端端數秒之內 = "Chỉ trong vòng vài giây,"
半鬥城市技術淪為斷壁殘源 = "Nửa thành phố đã biến thành đống đổ nát hoang tàn."
慌亂奔逃的人群裡, = "Giới dòng người hoảng loạn chạy trốn,"
一位母親緊緊抱著孩子拼命前行 = "Một người mẹ ôm chặt con mình chạy thục mạng."
危险步步逼近, = "Nguy hiểm rình rập ngay sau lưng,"
他集中生日踹開尚未完工的隧道圍擋, = "Cô dồn sức đạp đổ hàng rào của một đường hầm đang thi công,"
帶著孩子敘述藏身其中, = "Đưa con trốn vào bên trong,"
還急忙呼喊自家狗狗快跑, = "Đồng thời cuống quýt gọi chú chó của gia đình chạy theo."
就在危難降臨的岔那, = "Ngay khoảnh khắc ngọn lửa ập đến,"
小狗也縱身跳進隧道躲了起來。 = "Chú chó cũng kịp phóng người vào đường hầm ẩn nấp."
等到天地間的動靜徹底平息, = "Khi mọi chấn động bên ngoài hoàn toàn lắng xuống,"
繁華都市早已滿目創意, = "Đô thị phồn hoa giờ đã hoang tàn,"
整片區域僅是廢墟, = "Khắp nơi chỉ còn là đống đổ nát,"
唯有正對母子和一隻小狗僥倖存活下來。 = "Chỉ có hai mẹ con và chú chó là may mắn sống sót."
地面防衛隊伍敘述集結地量準備迎戰, = "Lực lượng phòng vệ mặt đất nhanh chóng tập hợp để chuẩn bị nghênh chiến,"
人類和天外一族的對戰正式開啟, = "Cuộc đối đầu chính thức khai màn."
轰入驾驶战机的人员望的眼前遮天闭日的巨型飞行器 = "Các phi công chiến đấu nhìn phi thuyền khổng lồ che kín bầu trời,"
内心满是震撼, = "Trong lòng không khỏi chấn động tột cùng,"
可危及局势容不得半分迟疑 = "Nhưng tình thế nguy cấp không cho phép họ chần chừ."
随着前线指挥下达指令, = "Ngay khi lệnh tấn công từ tiền tuyến được phát ra,"
大批作战 = "Đội hình"
战机迅速瞄准敌方目标 = "Các chiến cơ lập tức khóa mục tiêu địch,"
无数弹药潮着巨型外星飞行器很很康惧 = "Vô số tên lửa bắn xối xả vào phi thuyền khổng lồ ngoài hành tinh."
可战况远没有预想中顺利, = "Tuy nhiên, trận chiến không diễn ra suôn sẻ như mong đợi,"
所有攻击落在对方身上 = "Tất cả các đòn tấn công dội vào phi thuyền địch"
就像雨滴击打在坚硬玻璃上一般, = "Chỉ như những giọt nước đập vào tấm kính cường lực cứng nhắc,"
起不到丝毫作用 = "Hoàn toàn vô tác dụng."
原來這艘巨型飛行器外層 = "Hóa ra, lớp vỏ ngoài của con tàu khổng lồ này"
籠罩了一層肩不可催的能量防護屏障 = "Được bảo vệ bởi một lá chắn năng lượng vô cùng kiên cố."
首輪進攻徹底失利 = "Đợt tấn công đầu tiên thất bại hoàn toàn,"
戰機編隊立刻調整方位 = "Biển đội chiến cơ lập tức điều chỉnh hướng"
再度發起攻勢 = "Để tiếp tục mở đợt công kích mới,"
接連射出多枚彈藥 = "Liên tiếp bắn ra nhiều loạt tên lửa,"
結局依舊毫無改變 = "Nhưng kết cục vẫn không hề thay đổi."
Come on, = "Cố lên,"
come on! = "bắn đi chứ!"
眼看距離敵方飛行器越來越近 = "Nhận thấy khoảng cách với phi thuyền địch ngày càng thu hẹp,"
飛行員聯盟操控戰機向上攀升 = "Các phi công kéo cần điều khiển cho máy bay bay vút lên cao."
其中一架戰機沒能及時避讓 = "Một chiếc chiến cơ trong số đó do không kịp né tránh"
直直撞上防護屏障 = "Đã đâm sầm vào lá chắn bảo vệ,"
請客間機身損毀 = "Nổ tung thành trăm mảnh,"
人員不幸遇難, = "Phi công hy sinh anh dũng."
其餘戰機不敢再有絲毫鬆懈 = "Những máy bay khác không dám lơ là một giây nào nữa,"
急忙想要撤離這片危險空域, = "Vội vã tìm đường rút lui khỏi vùng không phận nguy hiểm này,"
卻早已措施最佳時機 = "Nhưng họ đã để lỡ mất thời cơ tốt nhất."
無數小型歪行戰機從巨型母艦中封用而出 = "Vô số máy bay tiêm kích mini ùa ra từ tàu mẹ khổng lồ,"
朝著人類戰機展開猛烈反擊 = "Tấn công dồn dập vào phi đội của loài người."
雙方交鋒瞬間, = "Ngay khi hai bên giáp lá cà,"
人類戰機接連受損落敗 = "Chiến cơ của loài người liên tục bị bắn hạ,"
敵方戰機卻依舊玩毫無損 = "Trong khi máy bay địch không hề hấn gì."
就連這些小型飛行器, = "Hóa ra ngay cả những chiếc tiêm kích nhỏ đó"
也都配備著穩固的防護解藉 = "Cũng được trang bị lớp lá chắn năng lượng kiên cố."
雙方科技實力差距懸殊 = "Khoảng cách về trình độ công nghệ quá lớn,"
人類這邊的作戰力量根本難以抗衡 = "Lực lượng phòng vệ của loài người hoàn toàn bất lực không thể kháng cự."
後方統籌據點指能眼睜睜看著前線戰機接連受損 = "Bộ chỉ huy ở hậu phương chỉ biết trố mắt nhìn các chiến cơ tiền tuyến rụng rơi,"
卻沒有有效的應對辦法 = "Mà chẳng có bất kỳ phương án ứng phó hiệu quả nào."
高層建廠當即下達指令 = "Cấp cao ngay lập tức ra lệnh,"
讓前線作戰人員儘快撤離戰場 = "Yêu cầu toàn bộ lực lượng tiền tuyến nhanh chóng rút khỏi chiến trường."
和面對數量遠超人類的星際戰機 = "Nhưng trước số lượng áp đảo của tiêm kích không gian địch,"
撤退之路依舊艱難 = "Đường rút lui vô cùng gian nan."
大批戰機接連被擊毀 = "Phần lớn chiến cơ đều bị bắn nổ,"
僅有兩架戰機拼死突圍成功 = "Chỉ có hai chiếc là liều chết mở được đường máu thoát ra."
身後卻依舊有兩架敵方戰機緊追不捨 = "Nhưng phía sau vẫn có hai chiếc tiêm kích địch bám đuổi gắt gao."
同行了戰友為了引開追兵 = "Một phi công đồng đội để đánh lạc hướng quân truy đuổi,"
獨自操控戰機極速拉升高度 = "Đã một mình lái máy bay vút lên độ cao tối đa,"
最容易高空環境影響失去飛行平衡 = "Dưới áp lực của bầu khí quyển cao, máy bay mất thăng bằng,"
很快就被敵方戰機擊中損毀 = "Và nhanh chóng bị tiêm kích địch bắn hạ."
另一邊, = "Ở một diễn biến khác,"
大批外星戰機精直衝向大型防空助手據點, = "Rất nhiều chiến cơ địch lao thẳng vào các cứ điểm phòng không mặt đất,"
對據點展開密集猛烈的襲擊 = "Dội bom tàn phá dữ dội."
紅大的對戰場面, = "Cảnh tượng giao tranh vô cùng hoành tráng,"
宛如真實上演的星際機戰一般, = "Giống hệt như một trận chiến giữa các vì sao trong thực tế."
面對擁有頂尖科技的天外飛行器 = "Trước những phi thuyền công nghệ đỉnh cao của đối phương,"
藍布級出身撤離的前線人員傷亡慘重, = "Lực lượng mặt đất chưa kịp rút lui phải chịu thương vong vô cùng nặng nề,"
整片空中作戰 = "Toàn bộ khu vực chiến đấu trên không"
助地轉眼就被徹底移平 = "Và cứ điểm phòng không bị san phẳng trong tích tắc."
另一邊, = "Lúc này,"
史蒂夫巧是計測, = "Đại úy Steven Hiller đã dùng mưu kế,"
把一架小型外星飛行器引至幽生峽谷之中, = "Dụ một chiếc tiêm kích ngoài hành tinh bay vào hẻm núi Grand Canyon,"
敵方飛行器在身後死死緊追 = "Chiếc máy bay địch vẫn bám đuổi sát nút phía sau."
通知峽谷地形走势, = "Nhờ thông thuộc địa hình hiểm trở của hẻm núi,"
數次靈活閃躲, = "Anh thực hiện vài cú lượn ngoạn mục,"
避開了對方攜來的攻勢 = "Né tránh thành công các loạt đạn từ kẻ thù."
眼看戰機人員即將耗盡, = "Nhận thấy nhiên liệu sắp cạn kiệt,"
史蒂夫立刻想出脫身執法, = "Steven lập tức nghĩ ra kế thoát thân,"
果斷開啟戰機, = "Anh dứt khoát kích hoạt thiết bị bung dù"
減速裝置 = "Hãm tốc của máy bay."
飄落的減速散瞬間遮擋住後方推動的視線 = "Tấm dù hãm tốc bung ra bản lớn che khuất hoàn toàn tầm nhìn của kẻ địch,"
緊接著他果斷彈射, = "Ngay lập tức, anh nhấn nút ghế phóng"
跳散脫離坐艙 = "Để nhảy dù thoát khỏi buồng lái."
施工的戰機直直撞向前方山崖, = "Chiếc chiến cơ mất lái lao thẳng vào vách đá,"
後方緊追的外星飛行起來不及自動 = "Máy bay ngoài hành tinh bám theo phía sau không kịp phản ứng"
也狠狠撞击地面损毁, = "Nên cũng đâm sầm xuống đất nổ tơi tả."
使地夫平稳落地, = "Steven tiếp đất an toàn,"
緩步走向损毁的外星飛行器, = "Anh chậm rãi đi tới mảnh vỡ của chiếc phi thuyền địch,"
隨手掀開艙門 = "Dùng tay cạy nắp khoang lái."
裡面一隻外星奇特的一星生靈證準備衝出, = "Bên trong, tên quái vật ngoài hành tinh định nhảy xổ ra tấn công,"
被他抬手一拳直接打暈在地 = "Nhưng liền bị anh tặng cho một cú đấm knock-out bất tỉnh nhân sự."
之後使地夫拖著陷入昏迷的一星生靈, = "Sau đó, Steven kéo lê gã người ngoài hành tinh đang hôn mê,"
在茫茫荒漠裡艱難趕路 = "Một mình lầm lũi đi bộ vượt qua sa mạc mênh mông."
屠住正巧遇上一對逃難車隊, = "May mắn thay, anh bắt gặp một đoàn xe tị nạn,"
跟著眾人一同前行, = "Nên đã đi cùng họ"
抵達了一處隱秘的地下研究據點 = "Đến Khu vực 51 - căn cứ nghiên cứu ngầm tuyệt mật."
他亮出抓獲的一星生靈後, = "Sau khi cho lực lượng cảnh vệ xem món quà là sinh vật ngoài hành tinh,"
直手人員立刻放行, = "Họ lập tức mở cổng cho đoàn xe đi vào."
這裡正是一直流傳在外的神秘隱秘試驗區 = "Nơi đây chính là khu nghiên cứu bí ẩn luôn bị chính phủ giấu kín bấy lâu."
此前一星勢力發起的首輪大範圍襲擊 = "Đợt tấn công hủy diệt diện rộng trước đó của kẻ địch"
讓地球上眾多繁華、城市競速遭殃 = "Đã khiến vô vàn đô thị phồn hoa trên Trái Đất tan hoang,"
維度這片隱秘、據點安穩無事 = "Duy chỉ có khu căn cứ ngầm này là vẫn an toàn vô sự."
僥倖脫險了一眾核心人員也紛紛聚集到此 = "Những nhân sự cốt cán thoát chết từ Nhà Trắng cũng tập trung về đây."
眾人望著據點內停放的天外飛行器 = "Mọi người nhìn chiếc phi thuyền đậu sẵn bên trong căn cứ"
滿心震驚 = "Mà lòng đầy sửng sốt."
原來很早之前 = "Hóa ra từ rất lâu về trước,"
人類就已經知曉天外生靈真實存在 = "Chính phủ loài người đã biết sinh vật ngoài hành tinh thực sự tồn tại."
多年前一場天外飛行器意外墜落世界 = "Nhiều thập kỷ trước, một vụ tai nạn rơi phi thuyền tại Roswell"
就讓人類首次接觸到外星文明 = "Đã giúp con người lần đầu tiên tiếp xúc với công nghệ ngoài Trái Đất."
相關一星生靈依還也一直被妥善留存 = "Xác sinh vật ngoài hành tinh và mảnh vỡ phi thuyền luôn được bảo quản nghiêm ngặt."
數十年來 = "Suốt mấy chục năm qua,"
相關科研團隊史中在案中 = "Đội ngũ khoa học tại đây vẫn âm thầm nghiên cứu"
前夕研究外星飞行器与外星生命体 = "Về cấu trúc chiếc phi thuyền cũng như các mẫu sinh vật này."
这类天外生灵身批特制生物防护开架 = "Loài sinh vật này vốn mặc một lớp giáp sinh học đặc biệt,"
身体构造和人类有着不少相似之处 = "Cơ thể thực bên trong của chúng có cấu trúc khá giống con người."
同样属于肉身屈体 = "Chúng cũng có xương thịt,"
屈体本身并不算强悍脆弱异伤 = "Nhưng thể xác khá mỏng manh và dễ bị tổn thương."
他们没有发生器官 = "Chúng không có cơ quan phát thanh,"
彼此之间依靠类似精神意念的方式 = "Mà giao tiếp với nhau bằng năng lực thần giao cách cảm,"
传递信息互相沟通 = "Truyền tải thông tin trực tiếp vào não bộ."
话音刚落 = "Vừa dứt lời,"
学者突然接到消息 = "Tiến sĩ Okun bất ngờ nhận được tin báo"
由一名存活的天外生灵被送到了剧点内 = "Có một mẫu vật sống vừa được đưa vào căn cứ."
他联盟快不出门迎接 = "Ông vội vàng ra cửa chào đón,"
正是史蒂夫压缩过来的这名异性生灵 = "Đó chính là tên quái vật mà Steven vừa kéo về."
很快就被送入专属隔离观测室, = "Nó nhanh chóng được đưa vào phòng cách ly đặc biệt,"
准备开展细致的躯体研究 = "Để tiến hành cuộc phẫu thuật nghiên cứu chi tiết."
学着手持精密气泄, = "Vị tiến sĩ cầm dụng cụ y tế,"
小心玻璃异性生灵外层的防护铠甲 = "Cẩn thận rạch bỏ lớp giáp sinh học bên ngoài của nó."
就在众人凝神官望之时突发小状况 = "Trong lúc mọi người đang nín thở theo dõi, một sự cố bất ngờ xảy ra."
好在只是虚惊异场, = "May mắn thay, đó chỉ là một phen hú vía."
随着外层铠甲逐步拆解 = "Khi lớp giáp ngoài được bóc tách ra hết,"
天外生灵真正的样貌, = "Hình dạng thật sự của sinh vật ngoài hành tinh"
终于完整展露在所有人眼前 = "Cuối cùng đã lộ diện hoàn toàn trước mắt mọi người."
就在这时, = "Ngay vào lúc này,"
那名异性生灵的手臂悄悄缓缓挪动 = "Cánh tay của sinh vật kia đột nhiên khẽ cử động,"
在场众人丝毫没有察觉危机依然逼近 = "Tất cả những người có mặt vẫn chưa hề hay biết nguy hiểm đang cận kề."
下一秒, = "Giây tiếp theo,"
兩道人影印進他驟然睜開了眼眸裡 = "Nó đột ngột mở trừng mắt, phản chiếu bóng hình của các bác sĩ."
科研學者只覺得腦中突然襲賴一股莫名的精神衝擊 = "Vị tiến sĩ chỉ cảm thấy đầu óc quay cuồng bởi một luồng xung kích tinh thần mạnh mẽ."
異星聲靈環環站起身, = "Tên quái vật từ từ gượng dậy,"
當場朝著實驗室裡的眾人發起突襲 = "Ngay lập tức mở cuộc tấn công bất ngờ vào mọi người trong phòng."
等基地負責人匆匆趕來時, = "Đến khi Tổng thống và đội hộ tống vội vã chạy tới,"
整間實驗室早已被濃密煙霧籠罩 = "Toàn bộ phòng thí nghiệm đã bị bao phủ bởi làn khói đặc."
異星人僅是緩不靠近 = "Sinh vật ngoài hành tinh từ từ tiến lại gần tấm kính."
有人想要救出被困的學者, = "Một người lính định lao vào cứu vị tiến sĩ đang bị kẹt,"
卻被身旁同伴及時攔下 = "Nhưng đồng đội đã kịp thời giữ lại."
帶到看清常常在學者勃梗處的詭異處虛 = "Khi nhìn rõ chiếc xúc tu ma quái đang quấn chặt lấy cổ vị tiến sĩ,"
才明白學者早已被對方暗中操控, = "Họ mới hiểu ra ông ấy đã bị gã quái vật khống chế như một con rối."
漫天濃煙慢慢消散, = "Làn khói dày đặc dần tan ra,"
異星生靈發出古怪詭異的聲響, = "Gã người ngoài hành tinh phát ra những âm thanh kỳ quái rợn người,"
無形的精神聲波瞬間擴散開來, = "Một làn sóng xung kích tinh thần vô hình lập tức lan tỏa."
在場眾人瞬間頭痛男人倒地全縮, = "Khiến những người có mặt đau đầu dữ dội, ngã quỵ xuống đất ôm đầu."
現場帶隊之人立刻下達指令, = "Chỉ huy đội bảo vệ liền hét lớn ra lệnh,"
眾人紛紛對著隔離倉內的異星生靈權力發起攻擊, = "Các binh lính đồng loạt xả súng điên cuồng vào sinh vật trong phòng cách ly."
一陣劇烈的精神刺痛過後, = "Thông qua bộ não của vị tiến sĩ trong cơn đau đớn đó,"
眾人腦海中都接受到了外星族群奔赴地球的真正圖謀, = "Tổng thống đã thấu thị được dã tâm tàn bạo của lũ quái vật: chúng muốn tận diệt loài người."
現場人員聯盟拿出武器, = "Các đặc vụ lập tức rút súng,"
對著失去反抗之力的異星生靈補給, = "Bắn bồi thêm nhiều phát vào sinh vật đã mất khả năng kháng cự,"
徹底消除了這份隱患。 = "Tiêu diệt hoàn toàn mối hiểm họa này."
为了彻底击退外来入侵者, = "Để đánh bại hoàn toàn thế lực xâm lược này,"
助人商议过后, = "Sau khi bàn bạc kỹ lưỡng,"
决定动用中级强力武器展开反击 = "Quyết định sử dụng vũ khí hạt nhân để phản công."
很怪, = "Rất nhanh sau đó,"
搭在重型作战武器的飞行战机奔赴战场上空 = "Chiếc máy bay ném bom mang theo vũ khí hạng nặng tiến vào vùng trời chiến sự."
接到指令后, = "Nhận được lệnh khai hỏa,"
重爆武器禁止朝着剧情外星主建社区 = "Đầu đạn tầm nhiệt nhắm thẳng vào phi thuyền khổng lồ ngoài hành tinh phóng đi."
到计时结束后精准命路目标 = "Sau khi đếm ngược kết thúc, tên lửa đánh trúng mục tiêu một cách chính xác."
就在所有人以为战局营来转机, = "Ngay khi mọi người cứ ngỡ cục diện trận chiến đã xoay chuyển,"
满心振恶之时 = "Và đang vô cùng phấn khích"
一則令人徹底心寒的消息傳來 = "Thì một tin tức dội gáo nước lạnh truyền đến:"
劇情外星逐漸已久完好無損 = "Tàu mẹ khổng lồ ngoài hành tinh vẫn vẹn toàn vô sự,"
絲毫沒有受到半點損傷 = "Không chịu bất kỳ tổn hại nào từ vụ nổ hạt nhân."
面對不僅逼 = "Đối mặt với kẻ thù"
妄睹侵賴整片大敵的外來勢力 = "Đang từng bước muốn nuốt chửng toàn bộ Trái Đất,"
難道人類文明真的要就此走向落幕 = "Liệu nền văn minh nhân loại có thực sự phải đi đến hồi kết?"
絕望的情緒瞬間弄著所有人 = "Sự tuyệt vọng bủa vây lấy tâm trí của tất cả mọi người."
大衛也整日借酒消愁 = "David cũng mượn rượu giải sầu suốt ngày,"
滿心懊惱自己空用學識 = "Anh vô cùng bất mãn khi bản thân học rộng tài cao"
無力扭轉眼下的絕境 = "Nhưng lại bất lực trước cảnh ngộ bế tắc hiện tại."
好在家人耐心開導勸位 = "May mắn là nhờ có người cha kiên trì khuyên bảo và khai sáng,"
讓讓重新振作起來 = "Đã giúp anh vực dậy tinh thần."
腦海中猛然浮現出 = "Một ý tưởng lóe lên trong đầu,"
克制外星戰艦的破解執法 = "Anh đã tìm ra cách phá giải lá chắn năng lượng của chiến hạm địch."
他立刻叫請休息中的科研人員 = "Anh gọi ngay các nhà khoa học đang nghỉ ngơi dậy,"
趙吉基地內所有人員一同商議對策, = "Tập hợp tất cả mọi người trong căn cứ để thảo luận đối sách,"
還特意拿出灌體道具 = "Anh còn lấy ra một mô hình thực nghiệm"
模擬對外星飛行器發起攻擊做實地測試 = "Để mô phỏng cuộc tấn công thực địa thử nghiệm trên phi thuyền địch."
初次試驗時, = "Trong lần thử nghiệm đầu tiên,"
子彈撞擊在能量防護層上直接被彈開 = "Viên đạn bắn vào lớp phòng thủ năng lượng lập tức bị bật văng ra,"
場面驚險不已 = "Cảnh tượng vô cùng hú vía."
隨後大衛將特質干擾程序植入模擬信號中 = "Sau đó, David cài đặt một virus máy tính đặc biệt vào tín hiệu mô phỏng"
再次進行射擊測試, = "Rồi tiến hành bắn thử lại lần nữa,"
這一次攻擊順利命入目標 = "Lần này đòn tấn công đã bắn trúng mục tiêu một cách dễ dàng."
原本堅不可摧的防護屏障徹底失效 = "Lớp bảo vệ vốn kiên cố bất khả xâm phạm nay đã hoàn toàn bị vô hiệu hóa."
原來大衛想出的辦法, = "Hóa ra, phương pháp mà David nghĩ ra"
開就是用特質信號病毒干擾對方系統 = "Chính là dùng một loại virus mã hóa để làm nhiễu loạn hệ thống của đối phương,"
如同讓飛行器陷入程序紊亂一般 = "Khiến hệ thống điều khiển của phi thuyền rơi vào trạng thái tê liệt."
在借助太空信號網絡 = "Sau đó mượn mạng lưới tín hiệu không gian liên lạc của chúng,"
把病毒擴散到所有外星母艦之中 = "Để phát tán virus này đến tất cả các tàu mẹ ngoài hành tinh,"
瓦解它們賴以自保的能量護盾 = "Đánh sập lớp khiên năng lượng tự vệ của chúng."
這也是人類唯一能正面發起反擊的突破口 = "Đây chính là bước đột phá và là cơ hội duy nhất để loài người phản công trực diện."
只是想要完成這項計畫 = "Tuy nhiên, để hoàn thành kế hoạch này,"
必須挑選兩名勇士駕駛小型飛行器奔赴浩漢太空 = "Cần phải tuyển chọn hai dũng sĩ lái chiếc phi thuyền nhỏ thu được vào không gian sâu thẳm,"
悄悄潛入外星主艦內部植入干擾程序 = "Âm thầm thâm nhập vào bên trong tàu mẹ của địch để cấy virus."
即便是計畫順利成功 = "Ngay cả khi kế hoạch thành công tốt đẹp,"
也只能短暫破除防護 = "Thì lớp phòng thủ cũng chỉ bị phá vỡ"
爭取短短片刻的作戰時機 = "Và mang lại một khoảng thời gian tác chiến vô cùng ngắn ngủi."
這已是眼下僅存的求生希望 = "Nhưng đây đã là tia hy vọng sống sót cuối cùng của Trái Đất."
為了守護家園 = "Để bảo vệ quê hương,"
大衛與史密私主動挺身而出接下來這項兇險任務 = "David cùng Đại úy Steven đã chủ động đứng ra nhận nhiệm vụ cực kỳ nguy hiểm này,"
自行前路秒忙, = "Dẫn thân vào hành trình lành ít dữ nhiều,"
升環機率維護其位 = "Xác suất sống sót trở về là vô cùng thấp."
與此同時, = "Cùng lúc đó,"
以密據點接觸專屬加密電波 = "Căn cứ bí mật sử dụng tần số vô tuyến mã hóa đặc biệt"
向全球各處傳遞聯合反擊的訊息 = "Để truyền thông điệp tổng lực phản công tới các lực lượng còn lại trên toàn cầu."
二人順利抵達太空領域 = "Hai người thuận lợi lái phi thuyền tiến vào không gian."
一艘體量左右1.4月球大小的巨型外星母艦赫然出現在視野之中 = "Một trạm không gian đầu não có kích thước bằng 1/4 Mặt Trăng hiện ra sừng sững trước mắt."
可就在飛行機逐步靠近時, = "Nhưng khi chiếc phi thuyền nhỏ tiến lại gần,"
機身突然不受操控 = "Hệ thống lái đột ngột bị mất kiểm soát,"
自動切換成自主航行模式 = "Tự động chuyển sang chế độ tự hành kết nối với tàu mẹ."
這樣的突發狀況其實早已在大衛的預料之內 = "Tình huống bất ngờ này thực chất đã nằm trong dự tính của David."
在万星母舰 of 引力迁引下 = "Dưới lực hút dẫn đường của tàu mẹ,"
兩人乘坐的飛行器緩緩使肉艦體內部 = "Chiếc phi thuyền của hai người từ từ tiến vào sâu bên trong khoang tàu."
眼前灰红奇特的內部構造 = "Kiến trúc khổng lồ và kỳ dị bên trong"
難以想像的竟是一艘星機航行戰艦 = "Khiến người ta khó lòng tưởng tượng nổi đây lại là một siêu chiến hạm vũ trụ."
飛行機從一種外星生靈頭頂緩緩練過 = "Phi thuyền lướt nhẹ qua ngay trên đầu của hàng loạt sinh vật ngoài hành tinh."
下方寬闊的巨型平台上 = "Dưới nền sân ga rộng lớn,"
數值不盡的外星作戰隊伍排列的整整機器 = "Vô số quân đoàn viễn chinh ngoài hành tinh đã tập hợp chỉnh tề,"
仍然是在集結待命 = "Chúng đang đứng chờ lệnh"
準備發起最後的全面清晰 = "Để phát động cuộc tổng tấn công hủy diệt cuối cùng."
沒多久 = "Không lâu sau,"
小型飛行器便被母艦內部的固定支架牢牢鎖住 = "Chiếc phi thuyền nhỏ bị hệ thống càng kẹp của tàu mẹ khóa chặt cố định."
抓緊時間上傳特製干擾程序 = "Họ tranh thủ từng giây để tải lên chương trình virus đặc chế."
統一時間, = "Cùng thời điểm đó,"
地球這邊也收到行動就緒的信號 = "Phía Trái Đất cũng nhận được tín hiệu báo sẵn sàng hành động."
所有空中作戰、飛行機全部正中待命 = "Toàn bộ phi đội máy bay tiêm kích đều vào vị trí chờ lệnh,"
終極對決一觸及發 = "Trận chiến sinh tử cuối cùng chuẩn bị bùng nổ."
很快干擾程序順利植入完成 = "Rất nhanh, virus máy tính đã được cấy thành công,"
混亂信號迅速擴散 = "Tín hiệu hỗn loạn nhanh chóng lan rộng,"
連艦內員們安穩修整的外星族群都受到了明顯影響 = "Khiến toàn bộ hệ thống của lũ quái vật trong tàu mẹ bị tác động rõ rệt."
前線指揮人員當機下達進攻指令 = "Chỉ huy mặt đất lập tức phát lệnh tấn công,"
大批戰基利克朝著巨星外星主艦發起攻勢 = "Hàng loạt chiến cơ lao thẳng về phía các phi thuyền khổng lồ của địch."
可對方依舊迅速撐起防護屏障 = "Nhưng đối phương vẫn kịp thời bật lá chắn bảo vệ,"
起來的攻擊技術被阻攔在外 = "Loạt tên lửa bắn ra đều bị chặn đứng bên ngoài."
眾人見撞都以為計劃落空, = "Mọi người thấy vậy cứ ngỡ kế hoạch đã đổ bể,"
戰機編隊只能暫時向後撤離 = "Biển đội máy bay đành phải tạm thời bay dạt ra sau."
可所有人都不願就此放棄, = "Nhưng các phi công không cam chịu bỏ cuộc,"
再度發起進攻嘗試 = "Họ tiếp tục thử nổ súng công kích lần nữa."
這一次攻擊成功突破阻礙機動艦體 = "Lần này, loạt đạn đã xuyên qua được lớp lá chắn và bắn trúng vỏ tàu."
地面上的人們瞬間歡呼雀躍, = "Mọi người ở mặt đất lập tức hò reo ăn mừng phấn khích,"
戰機隊伍迅速重等陣行 = "Đội hình chiến cơ nhanh chóng xếp lại trận thế,"
集中所有火力對外星戰艦展開輪返本宮 = "Tập trung toàn bộ hỏa lực luân phiên oanh tạc vào chiến hạm địch."
針對戰艦體型太過龐大 = "Tuy nhiên do kích thước con tàu quá khổng lồ,"
普通攻勢依舊難以造成毀滅性重瘡 = "Những đòn tấn công thông thường vẫn chưa thể gây ra thiệt hại mang tính hủy diệt."
請客間秘密麻麻的歪行小型戰機星潮而出展開反撲 = "Trong nháy mắt, đàn tiêm kích nhỏ dày đặc của địch ùa ra như ong vỡ tổ để phản công."
如今對方防護平蕩時效, = "Giờ đây khi lớp khiên bảo vệ của đối phương đã mất hiệu lực,"
雙方實力趨於對等 = "Cán cân thực lực giữa hai bên đã trở nên ngang ngửa."
前線作戰人員全 đều 拼盡全力奮勇迎戰 = "Các phi công tiền tuyến đều dốc toàn lực chiến đấu kiên cường."
戰火橫跨蔓延至地面易秘據點 = "Khói lửa chiến tranh nhanh chóng lan rộng xuống khu vực căn cứ mật dưới mặt đất,"
據點遭到大批外星飛行器突襲 = "Nơi này bị một toán máy bay địch bất ngờ tập kích,"
助手人員傷亡慘重 = "Lực lượng đồn trú chịu thương vong không nhỏ."
眾人荒芒躲進地下避險攻勢之內 = "Mọi người hoảng loạn kéo nhau trốn vào boong-ke ngầm để lánh nạn."
危機關頭 = "Giữa thời khắc nguy kịch,"
劇情外星主鍵再度啟動殺傷力極強的高能光束武器 = "Khẩu đại pháo laser hủy diệt dưới bụng tàu địch bắt đầu tích tụ năng lượng chuẩn bị khai hỏa."
眾人萬分交集 = "Ai nấy đều vô cùng lo sốt vó."
聯盟發射彈藥想要提前摧毀發射裝置 = "Biển đội máy bay cố bắn tên lửa nhằm phá hủy sớm họng súng phát xạ,"
卻被艦艇底部艙門紋紋擋住 = "Nhưng trớ trêu thay tầm bắn lại bị cửa khoang che chắn mất."
就在光束即將蓄能發射的生死瞬間 = "Ngay vào khoảnh khắc sinh tử khi tia laser hủy diệt chuẩn bị phóng ra,"
因為曾親力過外星危機的勞者依然卸下防護裝備 = "Ông lão phi công Russell Casse, người từng bị quái vật bắt cóc năm xưa, đã tháo bỏ mặt nạ dưỡng khí,"
禁止衝向巨型戰艦底部 = "Lái chiếc máy bay lao thẳng về phía bụng chiến hạm khổng lồ,"
分立撞向光束發射口 = "Liều mình đâm sầm vào họng súng laser đang mở rộng."
劇烈衝擊瞬間引發連鎖爆炸 = "Cú va chạm cực mạnh kích nổ kho vũ khí, gây ra chuỗi nổ liên hoàn."
龐大的外星戰艦人起衝天烈火 = "Con tàu ngoài hành tinh khổng lồ bốc cháy ngùn ngụt,"
禁止朝著地面墜落 = "Rồi đâm sầm xuống đất nổ tung thành tro bụi."
勞著不僅了結了過網恩怨 = "Ông lão không chỉ giải quyết được mối thù hận cá nhân trong quá khứ,"
更挺身而出守護了世間眾人 = "Mà còn dùng cả tính mạng để bảo vệ nhân loại."
成為了萬眾敬仰的英雄 = "Trở thành vị anh hùng vĩ đại được muôn người kính trọng."
而身處太空母艦內部的大衛羽同伴 = "Trong khi đó, David và đồng đội đang ở trong không gian,"
完成任務後卻發現大城的飛行器徹底失靈 = "Sau khi hoàn thành nhiệm vụ thì phát hiện phi thuyền của mình đã bị liệt hoàn toàn hệ thống khởi động,"
兩人瞬間陷入絕境 = "Cả hai rơi vào tình cảnh ngàn cân treo sợi tóc."
二人朝著敵方指揮區域試意過後 = "Sau cái nhìn từ biệt hướng về phía phòng điều hành của kẻ địch,"
果斷朝著母艦核心區域投放了重型殺傷武器 = "Họ dứt khoát phóng một đầu đạn hạt nhân hạng nặng vào thẳng lõi năng lượng của tàu mẹ."
千軍一發之際 = "Ngay trong khoảnh khắc định mệnh đó,"
所住飛行器的固定裝置意外鬆動 = "Khóa chốt giữ chiếc phi thuyền đột nhiên bị rung lắc chấn động làm lỏng ra,"
同伴立刻啟動設備全速撤離 = "Steven lập tức nhấn ga, điều khiển phi thuyền tháo chạy hết tốc lực."
身後數架敵方飛行棄警隨其後窮追不捨 = "Phía sau, vài chiếc tiêm kích địch vẫn điên cuồng đuổi theo không buông."
最終二人趕在艙門閉合的最後一順 = "Cuối cùng, hai người đã kịp lao ra ngoài"
成功衝出危險區域順利拖身 = "Ngay vào giây cuối cùng trước khi cửa khoang đóng sập lại, thoát hiểm ngoạn mục."
留在母艦之內的外星 = "Lũ quái vật ngoài hành tinh"
足群無力阻攔 = "Bị bỏ lại bên trong tàu mẹ hoàn toàn bất lực,"
只能眼睜睜看著到計時結束 = "Chúng chỉ biết tuyệt vọng nhìn đồng hồ đếm ngược kết thúc."
弱大的太空母艦轟然炸裂 = "Trạm không gian khổng lồ nổ tung kinh hoàng,"
花索漫天碎片消散在宇宙之中 = "Biến thành hàng triệu mảnh vụn tiêu tán vào không gian vũ trụ bao la."
人類終於成功擊退外敵 = "Nhân loại cuối cùng đã đánh đuổi được thế lực xâm lược,"
贏得了這場星際之戰的勝利 = "Giành chiến thắng oanh liệt trong cuộc chiến bảo vệ Trái Đất."
只是所有人都未曾察觉, = "Tuy nhiên, không một ai hay biết rằng,"
就在巨型母舰崩塌前, = "Ngay trước khi trạm không gian khổng lồ sụp đổ,"
一到来自一星升临的紧急球员讯号, = "Một tín hiệu cầu cứu khẩn cấp từ bọn chúng"
早已巧然向着遥远星际船递了出去。 = "Đã kịp thời được gửi đi hướng về phía dải ngân hà xa xôi."
值得一提的是, = "Một thông tin bên lề thú vị là,"
本片开机到沙清仅仅只用了72天, = "Bộ phim này từ lúc bấm máy cho đến khi đóng máy chỉ mất vỏn vẹn 72 ngày."
制作期间打造3978架支持战机, = "Trong quá trình sản xuất, họ đã chế tạo 3.978 mô hình chiến cơ của Trái Đất,"
3931架一星战机, = "3.931 mô hình tiêm kích ngoài hành tinh,"
52278组建筑场景碎片, = "52.278 mảnh vỡ bối cảnh công trình kiến trúc,"
还制作了9米长巨型星际母舰, = "Cùng một mô hình tàu mẹ không gian dài tới 9 mét,"
以即一米十二復刻, = "Và một mô hình phục chế tỉ lệ 1:12,"
六米長的核心地標鍵柱經系模型, = "Đó là mô hình chi tiết của các tòa nhà biểu tượng cốt lõi dài 6 mét."
平跌龐大威梭道具規模, = "Nhờ quy mô đạo cụ mô hình khổng lồ và hoành tráng này,"
斬獲了吉尼斯世界紀錄的認證, = "Bộ phim đã được tổ chức Kỷ lục Thế giới Guinness công nhận,"
成為同期使用威梭模型數量最多的影片。 = "Trở thành tác phẩm điện ảnh sử dụng số lượng mô hình thu nhỏ nhiều nhất trong lịch sử thời bấy giờ."
• 进尘三房包裹法有哪些？ = "Có 3 cách quấn khăn cho bé mà mẹ nào cũng nên biết."

• 第一种 防着凉包裹法。 = "Cách thứ nhất: Quấn chống lạnh."

• 这种包裹法适合外出打疫苗。 = "Phù hợp khi đưa bé ra ngoài hoặc đi tiêm phòng."

• 既能遮光挡风又能防着凉。 = "Vừa che nắng, cản gió, vừa giúp bé tránh bị nhiễm lạnh."

• 适合各种外出环境。 = "Thích hợp trong hầu hết các trường hợp khi ra ngoài."

• 第二种 防惊跳包裹法。 = "Cách thứ hai: Quấn chống giật mình."

• 小月龄宝宝往往有惊跳反射。 = "Trẻ sơ sinh thường có phản xạ giật mình."

• 睡觉时一惊一乍，睡不踏实。 = "Khi ngủ bé dễ giật mình, ngủ không sâu giấc."

• 这种包裹法模拟宝宝在妈妈肚子里的环境。 = "Cách quấn này mô phỏng cảm giác bé còn ở trong bụng mẹ."

• 增加安全感，让宝宝睡得香不闹觉。 = "Giúp bé cảm thấy an toàn hơn, ngủ ngon và ít quấy khóc."

• 第三种 防吐奶包裹法。 = "Cách thứ ba: Quấn chống trớ sữa."

• 把包被折成一个圆枕。 = "Gấp khăn quấn thành một chiếc gối tròn."

• 宝宝侧躺时把小脚搭在圆枕上。 = "Khi cho bé nằm nghiêng, đặt chân bé lên chiếc gối tròn."

• 背部靠着小圆枕。 = "Phần lưng tựa vào gối."

• 双手也包裹固定好。 = "Hai tay cũng được quấn cố định gọn gàng."

• 这种包裹法能有效防止宝宝吐奶。 = "Cách quấn này giúp hạn chế tình trạng bé bị trớ sữa rất hiệu quả."
• 错误姿势起不来。 = "Sai tư thế nên bé rất khó ngẩng đầu."

• 额不前伸。 = "Đầu không vươn về phía trước."

• 不发力。 = "Bé không thể dùng lực."

• 正确姿势共五步。 = "Tư thế đúng gồm 5 bước."

• 小腿一搭。 = "Đặt một chân của bé lên trước."

• 小手一举。 = "Nâng một cánh tay của bé lên."

• 屁屁一按。 = "Dùng tay đỡ nhẹ phần hông của bé."

• 整理两只小手。 = "Điều chỉnh hai tay của bé đúng vị trí."

• 护住后背。 = "Đỡ và bảo vệ phần lưng của bé."

• 缓解一下宝宝的背部肌张力。 = "Giúp giảm tình trạng căng cơ vùng lưng của bé."

• 让宝宝的头抬得更高、抬得更好。 = "Giúp bé ngẩng đầu cao hơn và vững hơn."

• 锻炼宝宝的手部力量、颈部力量和背部力量。 = "Rèn luyện sức mạnh cho tay, cổ và lưng của bé."

• 为宝宝的抬头翻身做准备。 = "Chuẩn bị nền tảng để bé ngẩng đầu và tập lật an toàn."
• 你看这个手啊。 = "Nhìn cánh tay của bé này."

• 它就开始扭了。 = "Bé bắt đầu cựa quậy."

• 马上就把自己扭醒了。 = "Chỉ một lúc là tự làm mình tỉnh giấc."

• 那怎么办呢？ = "Vậy phải làm thế nào?"

• 来。 = "Nào."

• 抓住。 = "Giữ nhẹ tay bé."

• 给它包过来。 = "Quấn khăn ôm gọn cánh tay lại."

• 一个手在下面。 = "Đặt một tay của bé ở phía dưới."

• 一个手在上面。 = "Tay còn lại đặt ở phía trên."

• 包过来。 = "Tiếp tục quấn khăn cố định lại."

• 把它的两个小脚给它抓住。 = "Nhẹ nhàng giữ hai bàn chân của bé."

• 拍它。 = "Vỗ về bé nhẹ nhàng."

• 一二三四五六七八九十。 = "Đếm từ một đến mười."

• 放松。 = "Thả lỏng cơ thể bé."

• 睡着了吧。 = "Thế là bé lại ngủ ngon rồi."
为什么坏掉的充电器里面全都是宝，却没人告诉我。 = "Tại sao bên trong những cục sạc hỏng lại toàn là đồ có giá trị mà chẳng ai nói cho tôi biết?"
如果你也有这种废旧的充电器。 = "Nếu bạn cũng có những cục sạc cũ đã hỏng,"
千万别当破烂卖。 = "Thì đừng vội bán cho người thu mua ve chai."
收废品的几毛钱收走。 = "Vì họ chỉ mua với giá rẻ mạt."
在里面随便拆一个零件。 = "Chỉ cần tháo ra một linh kiện bên trong,"
转手就能翻起十倍。 = "Là giá trị có thể tăng lên gấp nhiều lần."
我们瞅瞅这两根线。 = "Hãy nhìn hai sợi dây này."
里面是正宗的纯紫铜。 = "Bên trong là đồng đỏ nguyên chất."
这可是硬通货。 = "Đây là vật liệu rất có giá trị."
家里很多地方都能用得到。 = "Có thể tận dụng cho rất nhiều việc trong gia đình."
再看这个小风扇。 = "Tiếp theo là chiếc quạt nhỏ này."
12伏的给家里小孩儿。 = "Chỉ cần cấp nguồn 12 vôn là có thể sử dụng."
改个迷你小风扇。 = "Chế thành một chiếc quạt mini."
省电又耐用。 = "Vừa tiết kiệm điện, vừa bền."
看一下主板上的变压器。 = "Còn đây là biến áp trên bo mạch."
里头全都是紫铜丝的。 = "Bên trong quấn toàn bằng dây đồng đỏ."
稍微改一下就是个小型电焊机。 = "Chỉ cần cải tạo một chút là có thể làm thành máy hàn mini."
比买的还好用。 = "Thậm chí còn dùng hiệu quả hơn nhiều loại bán sẵn."
还有这个大电容放电电流特别猛。 = "Tụ điện lớn này có khả năng phóng điện rất mạnh."
专门修复那些饿死的坏电瓶特别方便。 = "Rất hữu ích khi phục hồi những bình ắc quy bị cạn kiệt."
这几个二极管更是好东西。 = "Mấy con điốt này cũng rất đáng giá."
不仅能整流。 = "Không chỉ dùng để chỉnh lưu dòng điện."
还可以接在充电口上。 = "Mà còn có thể lắp vào cổng sạc."
就可以防止小孩子不小心碰到触电。 = "Giúp hạn chế nguy cơ trẻ nhỏ vô tình bị điện giật."
这两个发光二极管。 = "Còn hai đèn LED này."
那是维修师傅的最爱。 = "Là linh kiện rất được thợ sửa chữa ưa dùng."
测电动车控制器。 = "Có thể dùng để kiểm tra bộ điều khiển xe điện."
查全车线路。 = "Hoặc kiểm tra hệ thống dây điện trên xe."
一测一个准。 = "Kiểm tra rất nhanh và chính xác."
还有这个黄色的方形小电容。 = "Còn tụ điện màu vàng hình vuông này."
如果家里晚上关了灯还发暗光。 = "Nếu đèn trong nhà đã tắt mà vẫn còn sáng mờ,"
接上去立马就能搞定。 = "Chỉ cần lắp thêm linh kiện này là xử lý được ngay."
最后看这个三级管。 = "Cuối cùng là con tranzito này."
用它做的迷你型拆焊台。 = "Có thể tận dụng để chế thành một bộ tháo hàn mini."
自己修灯。 = "Tự sửa chữa đèn."
换灯珠方便的很。 = "Thay bóng LED rất nhanh và thuận tiện."
瞧见了吧。 = "Thấy chưa?"
这里面全都是好东西。 = "Bên trong toàn là những linh kiện có giá trị."
想跟着做的。 = "Nếu muốn tự làm theo,"
我来一步一步的教你该怎么做。 = "Tôi sẽ hướng dẫn từng bước thật chi tiết."
把她这样拳起来。 = "Ôm gọn bé như thế này."

• 哎呀。 = "Đấy nhé!"

• 听到没。 = "Nghe thấy chưa?"

• 你说宝宝不好戴。 = "Nếu bé khó xì hơi,"

• 那是因为你不知道这四种排气法。 = "Là vì bạn chưa biết 4 cách giúp bé xì hơi hiệu quả."

• 第一种。 = "Cách thứ nhất."

• 我们让宝宝趴过来。 = "Cho bé nằm sấp."

• 腿托把它拳起来。 = "Đỡ hai chân rồi ôm gọn bé lại."

• 这个就叫飞机拳抱。 = "Đây gọi là tư thế bế máy bay."

• 这种姿势她的腿托。 = "Ở tư thế này, hai chân bé được nâng đỡ."

• 肚子有个脊啊。 = "Phần bụng sẽ được tạo một áp lực nhẹ."

• 就会帮助排气。 = "Nhờ đó bé sẽ dễ xì hơi hơn."

• 第二种。 = "Cách thứ hai."

• 我们把宝宝以飞机抱的姿势放在妈妈的腿上面。 = "Đặt bé ở tư thế bế máy bay trên đùi của mẹ."

• 其跨在你的小腿上。 = "Cho bé nằm vắt ngang trên cẳng chân."

• 要把这个手把它捏起来。 = "Một tay giữ chắc cơ thể của bé."

• 重点是我们要把她腿托起。 = "Điều quan trọng là phải nâng phần chân của bé lên."

• 跨在你的小腿上。 = "Để bé tựa trên cẳng chân."

• 这样子可以增加她肚子的脊啊。 = "Tư thế này sẽ tạo thêm áp lực lên vùng bụng."

• 然后就可以帮助排气了。 = "Giúp bé xì hơi dễ dàng hơn."

• 第三种 妈妈把腿全起来。 = "Cách thứ ba: Mẹ co hai chân lại."

• 把宝宝放在两腿中间。 = "Đặt bé vào giữa hai chân."

• 和妈妈面对面。 = "Để bé quay mặt về phía mẹ."

• 然后把腿头灯在妈妈的肚子上面。 = "Cho hai chân bé tì nhẹ vào bụng mẹ."

• 这样子可以增加敷衙。 = "Cách này giúp tăng áp lực lên vùng bụng."

• 就像我们灯坑的感觉。 = "Nhờ đó bé sẽ dễ xì hơi hơn."

• 跟这个动作一定要做到爸爸做。 = "Bố cũng có thể thực hiện động tác này."

• 第四个。 = "Cách thứ tư."

• 我们把宝宝包起来。 = "Quấn khăn cho bé."

• 两只小脚含起来。 = "Co nhẹ hai chân của bé."

• 抵住肚子。 = "Để đầu gối áp sát vào bụng."

• 这样就可以增加敷衙。 = "Cách này sẽ tạo thêm áp lực lên vùng bụng."

• 帮助排气了。 = "Giúp bé xì hơi nhanh và dễ dàng hơn."
• 大修发动机拆出来的活塞不要扔。 = "Pít-tông tháo ra khi đại tu động cơ đừng vội vứt đi."

• 可以焊接上钢板改造成木板打磨工具。 = "Chỉ cần hàn thêm một tấm thép là có thể chế thành dụng cụ chà nhám gỗ."

• 先把废旧活塞焊接上钢板固定配件。 = "Trước tiên, hàn cố định pít-tông cũ lên tấm thép."

• 接着再把磨砂纸打上胶水贴在活塞上压紧。 = "Tiếp theo, bôi keo lên giấy nhám, dán lên bề mặt pít-tông rồi ép chặt."

• 然后再把弹簧安装到活塞两边的钢板上用螺丝固定。 = "Sau đó, lắp lò xo vào hai tấm thép ở hai bên pít-tông và cố định bằng bu lông."

• 再接着把钢板装上去压住弹簧拧紧螺丝。 = "Tiếp tục lắp tấm thép phía trên để ép lò xo, rồi siết chặt các bu lông."

• 再然后把滚动轴承装到打磨机上。 = "Tiếp theo, lắp ổ bi vào máy chà nhám."

• 接着将打磨机装到活塞上拧紧固定制作成抛磨机。 = "Sau đó gắn máy chà nhám vào cụm pít-tông và siết chặt để hoàn thiện dụng cụ."

• 这样就可以启动打磨机带动活塞来回活动。 = "Khi khởi động, máy sẽ truyền động giúp pít-tông chuyển động tịnh tiến qua lại."

• 把木板表面打磨光滑去掉毛刺。 = "Nhờ vậy có thể chà nhẵn bề mặt gỗ và loại bỏ hoàn toàn các dăm gỗ."

• 你们觉得这个发明实用吗？ = "Các bạn thấy sáng kiến này có thực sự hữu ích không?"
• 你还见过这种老物件吗？ = "Bạn đã từng thấy món đồ cũ này chưa?"

• 可惜现在的很多年轻人都不知道。 = "Đáng tiếc là bây giờ nhiều bạn trẻ không còn biết đến nó."

• 把线往上一缠。 = "Quấn dây đồng quanh lõi."

• 加两块小磁铁。 = "Sau đó gắn thêm hai viên nam châm nhỏ."

• 立马变成高科技。 = "Lập tức biến thành một thiết bị phát điện mini."

• 不用电池盒。 = "Không cần dùng hộp đựng pin."

• 磁铁往里电池上一吸。 = "Chỉ cần gắn nam châm vào hai đầu viên pin."

• 绝了。 = "Quá hay luôn."

• 根本不用管正负极。 = "Thậm chí không cần quan tâm đến cực âm hay cực dương."

• 只要灯一闪就开始充电。 = "Chỉ cần đèn báo sáng là thiết bị đã bắt đầu sạc."

• 是不是比现在的充电器还香？ = "Các bạn thấy cách này có tiện hơn cả bộ sạc thông thường không?"
• 颠一颠。 = "Nhẹ nhàng nhún bé lên xuống."

• 听到没？ = "Nghe rõ chưa?"

• 你不想做排气操，我给你讲一个简单排气法。 = "Nếu bạn không muốn tập bài tập xì hơi cho bé thì mình sẽ chỉ một cách đơn giản hơn."

• 拿个枕头过来。 = "Lấy một chiếc gối."

• 垫在腿下面。 = "Đặt gối kê dưới chân."

• 我们把宝宝以飞机抱的姿势。 = "Bế bé theo tư thế máy bay."

• 趴在你的腿上。 = "Đặt bé nằm sấp trên đùi."

• 我们把宝宝的手捏到一起。 = "Khép hai tay của bé lại."

• 这样它就不会散开了。 = "Như vậy tay bé sẽ không bị dang ra."

• 把宝宝的腿这样往上面一推。 = "Nhẹ nhàng nâng hai chân bé lên."

• 它的肚子是不是就骑跨在你的腿上面。 = "Lúc này bụng bé sẽ tì lên đùi của bạn."

• 这样就感觉到宝宝肚子和你的小腿之间。 = "Nhờ vậy bụng bé sẽ được tì nhẹ vào cẳng chân của bạn."

• 就一个小小的挤压，就可以帮助排气了。 = "Chỉ cần một lực ép nhẹ là đã giúp bé xì hơi dễ hơn."

• 这种姿势的话，比我们飞机抱排气更轻松。 = "Tư thế này còn giúp bé xì hơi hiệu quả và nhẹ nhàng hơn cả tư thế bế máy bay thông thường."
• 这个小东西叫微动开关，几分钱一个。 = "Linh kiện nhỏ này gọi là Nút bấm 4 chân, giá chỉ vài chục nghìn một cái."

• 如果家里的电风扇或其他电器只是灯亮。 = "Nếu quạt điện hoặc các thiết bị điện trong nhà chỉ sáng đèn báo,"

• 按开关没任何反应打不开。 = "nhưng bấm công tắc mà không có phản ứng, thiết bị không hoạt động,"

• 大部分都是这个微动开关的问题。 = "thì đa phần nguyên nhân là do Nút bấm 4 chân bị hỏng."

• 只需打开外壳拆除里面的主板。 = "Chỉ cần mở vỏ thiết bị và tháo bo mạch bên trong ra."

• 找到相对应的那个开关。 = "Sau đó xác định đúng Nút bấm 4 chân cần thay."

• 用烙铁把它烫下来。 = "Dùng mỏ hàn tháo công tắc cũ ra."

• 再安装一个新的就能解决。 = "Lắp công tắc mới vào là thiết bị sẽ hoạt động bình thường."

• 你学会了吗？ = "Bạn đã biết cách làm chưa?"
• 注意看。 = "Chú ý nhé."

• 这就是最简单好用的风口器。 = "Đây là đầu thổi gió tự chế đơn giản nhưng cực kỳ hiệu quả."

• 把一个不要的衣服夹拆开。 = "Tháo rời một chiếc kẹp quần áo cũ không còn sử dụng."

• 拽出来这个U型的金属丝。 = "Lấy phần dây kim loại hình chữ U ở bên trong ra."

• 然后塞到打火机的防风罩里面。 = "Luồn dây kim loại vào bên trong tấm chắn gió của bật lửa."

• 只需加热一下就可以使用。 = "Chỉ cần châm lửa làm nóng là có thể sử dụng ngay."

• 我们来看看这样的效果。 = "Cùng xem hiệu quả của mẹo này nhé."

• 那是真不错。 = "Kết quả thực sự rất ấn tượng."
• 注意看。 = "Chú ý nhé."

• 鱼肉买回来不要总是红烧了。 = "Mua cá về đừng lúc nào cũng chỉ đem kho nữa."

• 试试这个蒜香鱼片。 = "Hãy thử món cá phi lê sốt tỏi này."

• 保证你吃上一次就会念念不忘了。 = "Đảm bảo chỉ cần ăn một lần là sẽ nhớ mãi hương vị."

• 首先。 = "Đầu tiên,"

• 选用10块一斤的黑鱼。 = "Chuẩn bị cá lóc, phi lê thành từng lát mỏng."

• 切成片后放葱姜。 = "Cho hành lá và gừng vào cùng cá."

• 一小勺盐。 = "Thêm một thìa nhỏ muối."

• 少许胡椒粉。 = "Rắc một ít tiêu xay."

• 一个鸡蛋清。 = "Cho thêm một lòng trắng trứng."

• 抓拌均匀腌制10分钟。 = "Trộn đều rồi ướp khoảng 10 phút."

• 水开后。 = "Khi nước sôi,"

• 把鱼肉一片一片放进去。 = "Thả từng lát cá vào nồi."

• 避免成坨了。 = "Không cho vào cùng lúc để cá không bị dính vào nhau."

• 焯烫鱼肉微微卷起就捞出备用。 = "Khi miếng cá vừa cong nhẹ thì vớt ra để ráo."

• 起锅油热放蒜末。 = "Làm nóng chảo với ít dầu rồi cho tỏi băm vào phi thơm."

• 小米辣炒出蒜香。 = "Thêm ớt hiểm băm và xào cho dậy mùi."

• 调味放生抽。 = "Nêm thêm nước tương."

• 蚝油少许鸡精炒出酱香味。 = "Cho dầu hào và một ít bột ngọt, đảo đến khi nước sốt thơm."

• 有缘看到视频的叔叔阿姨哥哥姐姐。 = "Cảm ơn mọi người đã xem đến đây."

• 祝你们事事顺心 心想事成。 = "Chúc mọi người luôn gặp nhiều may mắn, vạn sự như ý."

• 我的祝福你记得查收哦。 = "Nhớ nhận lời chúc tốt đẹp này nhé."

• 接着加入鱼肉。 = "Cho phần cá đã chần vào chảo."

• 快速翻炒均匀。 = "Đảo nhanh tay để cá thấm đều gia vị."

• 撒点葱花。 = "Rắc thêm hành lá cắt nhỏ."

• 开大火翻炒几下就可以出锅了。 = "Đảo trên lửa lớn vài lần rồi tắt bếp."

• 这样做出来的蒜香鱼片。 = "Món cá phi lê sốt tỏi hoàn thành."

• 鲜香滑嫩。 = "Thịt cá mềm ngọt, thơm ngon."

• 蒜香十足。 = "Mùi tỏi thơm nức."

• 三碗米饭都不够吃。 = "Ngon đến mức ăn hết ba bát cơm vẫn còn thèm."

• 喜欢的话赶紧试试看哦。 = "Nếu thích thì hãy thử làm ngay nhé."
• 注意看。 = "Chú ý nhé."

• 买回来的草鱼不要总是红烧清蒸了。 = "Mua cá trắm về đừng chỉ đem kho hay hấp nữa."

• 试试这个草鱼新吃法。 = "Hãy thử ngay cách chế biến cá trắm mới này."

• 简单好吃又美味。 = "Vừa đơn giản lại thơm ngon khó cưỡng."

• 你可要看好了。 = "Nhớ xem kỹ cách làm nhé."

• 8块一斤的草鱼买回来。 = "Chuẩn bị một con cá trắm tươi."

• 像我这样改上花刀方便入味。 = "Khứa vài đường lên thân cá như thế này để cá thấm gia vị hơn."

• 切好之后放葱姜。 = "Sau đó cho hành lá và gừng vào."

• 一勺料酒。 = "Thêm một thìa rượu nấu ăn."

• 清洗下这样能除去鱼腥味。 = "Xoa đều rồi rửa sạch để khử mùi tanh của cá."

• 然后放姜末、蒜末、小米椒。 = "Tiếp theo cho gừng băm, tỏi băm và ớt hiểm băm."

• 一勺料酒。 = "Thêm một thìa rượu nấu ăn."

• 少许生抽。 = "Cho một ít nước tương."

• 少许盐。 = "Nêm thêm một ít muối."

• 一点鸡精、胡椒粉、豆豉。 = "Cho thêm một ít bột ngọt, tiêu xay và đậu đen lên men."

• 最后放一勺食用油。 = "Cuối cùng thêm một thìa dầu ăn."

• 全部翻拌均匀腌制15分钟。 = "Trộn đều tất cả rồi ướp khoảng 15 phút."

• 看到这如果你还记不住的话。 = "Nếu xem đến đây mà vẫn chưa nhớ cách làm,"

• 可以点赞收藏起来哦。 = "thì nhớ lưu lại để xem lại nhé."

• 接着把腌制好的鱼肉整齐摆入盘中。 = "Xếp cá đã ướp ngay ngắn lên đĩa."

• 水开扣盖大火蒸8分钟。 = "Đợi nước sôi rồi hấp lửa lớn khoảng 8 phút."

• 出锅之前撒点葱花。 = "Trước khi lấy ra, rắc thêm ít hành lá."

• 这样做出来的草鱼。 = "Món cá trắm làm theo cách này"

• 鱼肉鲜香入味。 = "thịt cá mềm ngọt, thấm đẫm gia vị."

• 下饭好吃。 = "Ăn với cơm cực kỳ ngon."

• 赶紧试试吧。 = "Lưu lại và thử ngay nhé."
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

    return _parse_ollama_response(raw_text, len(batch_segments))


def _parse_ollama_response(raw_text, expected_count):
    raw_text = str(raw_text or "").strip()
    if not raw_text:
        return [""] * expected_count

    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    def normalize_item(item):
        if item is None:
            return ""
        return str(item).strip()

    # 1) Try strict JSON parse
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, list):
            return [normalize_item(x) for x in parsed][:expected_count] + [""] * max(0, expected_count - len(parsed))
    except Exception:
        pass

    # 2) Try to extract the first JSON array from the text
    try:
        m = re.search(r"(\[.*\])", raw_text, re.S)
        if m:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, list):
                return [normalize_item(x) for x in parsed][:expected_count] + [""] * max(0, expected_count - len(parsed))
    except Exception:
        pass

    translated = []
    for line in raw_text.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue

        # Remove numbered or bullet prefixes like "1.", "-", "*"
        line = re.sub(r"^\s*\d+\.\s*", "", line)
        line = re.sub(r"^\s*[-*]\s*", "", line)

        try:
            item = json.loads(line)
            if isinstance(item, list):
                translated.extend(normalize_item(x) for x in item)
                continue
            if isinstance(item, str):
                translated.append(item)
                continue
        except Exception:
            pass

        if line.startswith('"') and line.endswith('"'):
            translated.append(line[1:-1].strip())
            continue

        quoted = re.findall(r'"([^"]+)"', line)
        if quoted:
            translated.extend(q.strip() for q in quoted)
            continue

        translated.append(line)

    if not translated:
        # Last resort: split by plain newlines and take nonempty text
        translated = [part.strip() for part in raw_text.splitlines() if part.strip()]

    if len(translated) < expected_count:
        translated.extend([""] * (expected_count - len(translated)))
    return translated[:expected_count]


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

    # Dịch tất cả segments (hallucination filter đã tắt)
    valid_segments = segments
    print(f"[Translate] {len(segments)} segments → translate all ({source_lang}→{target_lang})")

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

    # Map kết quả theo thứ tự gốc
    translated_segments = []

    for seg, raw in zip(segments, valid_translations):
        text = clean_text(raw)

        if is_bad_translation(text, target_lang=target_lang):
            # Dịch không rõ / sai → bỏ qua để TTS skip
            text = ""

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


async def make_tts_audio(text, voice, rate, output_path, retries=3):
    import asyncio

    text = clean_text(text)
    text = _sanitize_tts_text(text)
    if not text or len(text) < 2:
        text = FALLBACK_TEXT

    # -----------------------------------------------------------------------
    # CapCut TTS routing — voice bắt đầu bằng "capcut:" → gọi CapCut API
    # -----------------------------------------------------------------------
    if voice.startswith("capcut:"):
        capcut_voice_type = voice.replace("capcut:", "")
        resource_id = capcut_get_resource_id(capcut_voice_type)

        # Chuyển rate từ "+10%" → "1.1" cho CapCut API
        rate_int = parse_rate_int(rate)
        capcut_rate = str(round(1.0 + rate_int / 100.0, 2))

        last_err = None
        for attempt in range(retries):
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
                result = await capcut_tts_generate(
                    text=text,
                    voice=capcut_voice_type,
                    resource_id=resource_id,
                    rate=capcut_rate,
                    output_path=output_path,
                    timeout=30.0,
                )
                if os.path.exists(result) and os.path.getsize(result) >= 100:
                    return result
                last_err = Exception("CapCut TTS output is empty or too small")
            except Exception as e:
                last_err = e
                print(f"  [CapCut TTS] attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2.0 + attempt * 1.0)
        raise last_err

    # -----------------------------------------------------------------------
    # Edge TTS (default path — giữ nguyên logic cũ)
    # -----------------------------------------------------------------------
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

    # Chỉ retry cùng voice đã chọn, không tự động chuyển voice khi gặp lỗi.
    voices_to_try = [voice]

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
    remove_chinese: bool = False,
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

        # Giữ tất cả segments từ Whisper (hallucination filter đã tắt)
        segments = raw_segments
        print(f"[Whisper] keeping all {len(segments)} segments (no hallucination filter)")

        if not segments:
            raise Exception("No valid speech detected")

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

        tts_cutoff = video_duration  # TTS segments can use full video duration

        for i, seg in enumerate(translated_segments):
            try:
                text = clean_text(seg.get("text", ""))

                # Câu rỗng = Whisper hallucination đã lọc → skip hoàn toàn
                if not text:
                    print(f"TTS SKIP {i}: hallucination filtered (zh={seg.get('original_text','')[:30]})")
                    continue

                if is_bad_translation(text):
                    print(f"TTS SKIP {i}: bad translation or unclear text (zh={seg.get('original_text','')[:30]})")
                    continue

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
                if start_time >= tts_cutoff:
                    audio_clip.close()
                    print(f"TTS SKIP {i}: start={start_time:.1f}s >= tts_cutoff={tts_cutoff:.1f}s")
                    continue

                # Cắt audio nếu nó tràn vào câu sau quá xa (>30% câu sau)
                # Cho phép overlap nhẹ với câu sau, không cho overlap nhiều
                if i + 1 < len(translated_segments):
                    next_start = float(translated_segments[i + 1]["start"])
                    next_dur = float(translated_segments[i + 1]["end"]) - next_start
                    overlap_budget = next_start + max(0.4, next_dur * 0.3)
                    max_end = overlap_budget
                else:
                    max_end = tts_cutoff  # Câu cuối: không cho vượt tts_cutoff

                if start_time + audio_clip.duration > max_end:
                    audio_clip = audio_clip.subclipped(0, max(0.3, max_end - start_time))



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
        layers = list(audio_clips)  # TTS layer (including closing message)

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

        # ────────────────────────────────────────────────────────────────────
        # XÓA TEXT TIẾNG TRUNG: nếu FE bật removeChinese=true,
        # dùng EasyOCR + cv2.inpaint để xóa text Trung trên video.
        # Chạy SAU merge audio để video output đã có audio TTS mới.
        # ────────────────────────────────────────────────────────────────────
        if remove_chinese:
            update_job(job_id, 85, "Removing Chinese text from video...")
            print(f"[RemoveChinese] Starting text removal for {merged_path}")

            clean_filename = f"clean_{merged_filename}"
            clean_path = f"{UPLOAD_FOLDER}/{clean_filename}"

            try:
                def _progress_cb(current, total):
                    if total > 0:
                        pct = 85 + int(10 * current / total)  # 85% → 95%
                        pct = min(pct, 95)
                        update_job(job_id, pct, f"Removing Chinese text... {current}/{total} frames")

                remove_chinese_text_from_video(
                    input_video=merged_path,
                    output_video=clean_path,
                    ocr_interval=5,
                    inpaint_radius=7,
                    dilate_px=12,
                    progress_callback=_progress_cb,
                )

                # Thay output bằng file đã clean
                merged_filename = clean_filename
                print(f"[RemoveChinese] Done → {clean_filename}")

            except Exception as e:
                # Nếu text removal fail, vẫn giữ video gốc (có text Trung)
                print(f"[RemoveChinese] Failed ({e}), keeping original video")

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


def _download_video_from_url(video_url: str) -> str:
    try:
        import yt_dlp
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Video URL support requires yt-dlp. Install yt-dlp in requirements.txt.",
        )

    safe_filename = f"video_{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(UPLOAD_FOLDER, safe_filename)
    
    is_douyin = "douyin.com" in video_url or "dy.isi.art" in video_url

    # First attempt: Try with browser cookies
    ydl_opts = {
        "outtmpl": output_path,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        "ignoreerrors": False,
        "restrictfilenames": True,
        "socket_timeout": 30,
        "retries": 3,
    }
    
    # For Douyin, try to extract cookies from browser
    if is_douyin:
        # Try multiple browsers in order of preference
        for browser in ["firefox", "chrome", "edge"]:
            ydl_opts["cookiesfrombrowser"] = (browser, None)
            try:
                print(f"[yt-dlp] Attempting Douyin download with {browser} cookies...")
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
                if os.path.exists(output_path):
                    print(f"[yt-dlp] Successfully downloaded from Douyin using {browser}")
                    return output_path
            except Exception as e:
                error_msg = str(e)
                print(f"[yt-dlp] Failed with {browser}: {error_msg}")
                # Clean up partial file
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except:
                        pass
                continue
        
        # If all browser methods failed
        raise HTTPException(
            status_code=403,
            detail=(
                "Douyin/TikTok China requires fresh login cookies. Please:\n"
                "1. Log into douyin.com in your browser (Firefox, Chrome, or Edge)\n"
                "2. Keep the browser open and try again\n"
                "Or use a YouTube URL instead."
            ),
        )
    
    # For other platforms (YouTube, etc)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    except Exception as e:
        error_msg = str(e)
        print(f"[yt-dlp] Download error: {error_msg}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to download video: {error_msg[:100]}"
        )

    if not os.path.exists(output_path):
        raise HTTPException(status_code=500, detail="Failed to download video from URL")

    return output_path


@app.post("/upload-video")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(None),
    videoUrl: str | None = Form(None),
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
    removeChinese: str = Form("false"),
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
    remove_chinese_flag = str(removeChinese).lower() in ("true", "1", "yes")

    try:
        zoom_val = float(zoomCrop)
    except Exception:
        zoom_val = 1.0

    # Nếu có videoUrl thì tải video từ URL trước khi xử lý.
    if videoUrl:
        videoUrl = videoUrl.strip()
        if not videoUrl:
            raise HTTPException(status_code=400, detail="videoUrl is empty")
        file_path = await run_in_threadpool(_download_video_from_url, videoUrl)
        safe_filename = os.path.basename(file_path)
    else:
        if file is None or not getattr(file, "filename", None):
            raise HTTPException(status_code=400, detail="No video file or videoUrl provided")

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
        remove_chinese=remove_chinese_flag,
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
    # Tùy chọn hiển thị/ẩn subtitle — mặc định True (hiển thị)
    show_subtitles = bool(data.get("showSubtitles", True))

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

    current_stream = "[0:v]"

    # subtitle layer — chỉ burn-in khi showSubtitles=True
    if show_subtitles:
        ass_path_for_ffmpeg = ffmpeg_escape_path(ass_path)
        filter_parts.append(
            f"{current_stream}ass='{ass_path_for_ffmpeg}'[v0]"
        )
        current_stream = "[v0]"
    else:
        print(f"[Render] showSubtitles=False → bỏ qua burn-in subtitle")

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

    # Nếu không có filter nào (showSubtitles=False và không có image),
    # dùng -c:v copy để không re-encode (nhanh hơn nhiều)
    if filter_parts:
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
    else:
        # Không có filter → copy nguyên video, không re-encode
        cmd = [
            "ffmpeg",
            "-y",
            *inputs,
            "-map", "0:v:0",
            "-map", "0:a?",
            "-c:v", "copy",
            "-c:a", "copy",
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