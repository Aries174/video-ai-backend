"""
CapCut TTS Service — wrapper cho CapCut reverse-engineered TTS API.

⚠️  API KHÔNG CHÍNH THỨC: có thể ngừng hoạt động bất cứ lúc nào
    nếu ByteDance thay đổi endpoint/signing.

Dựa trên: https://github.com/K07VN/capcut-tts-api
"""

import asyncio
import base64
import hashlib
import json
import os
import random
import secrets
import time
import uuid
from copy import deepcopy
from urllib.parse import urlencode

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE = "https://editor-api-sg.capcutapi.com"

DEFAULT_DEVICE = {
    "aid": "359289",
    "app_name": "CapCut",
    "appvr": "8.7.0",
    "version_name": "8.7.0",
    "version_code": "8.7.0",
    "channel": "capcutpc_google",
    "device_platform": "mac",
    "device_type": "MacBookPro17,1",
    "device_brand": "MacBookPro17,1",
    "os_version": "15.7.4",
    "device_id": str(random.randint(7000000000000000000, 7999999999999999999)),
    "iid": str(random.randint(7000000000000000000, 7999999999999999999)),
    "region": "VN",
    "loc": "VN",
    "lan": "vi-VN",
    "pf": "3",
    "tdid": str(random.randint(7000000000000000000, 7999999999999999999)),
}

TTS_SIGN_PUBLIC_KEY_PEM = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAmTd34Lw4b7IuldSXh/zY
CMla+ITdGG5TeWz6ad+OySd4r+IrY45AoqrYUxhQ2dl+7z+i7r/5vEa8rr39BYfB
8AGMQLmZA8HmgpWBsqrn/V6daUALkKnkLb70Fn32CJigIuGXAYqxUdGuI340aC+0
v5Es3puJsHyzf01/AelE4Cdc6bZhQrASJLBh8R3BQToYClmDVSDUQk28o8sl/guA
Z4n303Vj+6Siv1HayPCdV6kpVVnMBAG4+umUbwGmn132N3fgpzLarFF3XyWmS1zh
D/J07iM/rP8GDO9IskHNHd2phrO0G6KzrcFAnTBHjVv+hCBEfzN/no3FNA9AuC36
mwIDAQAB
-----END PUBLIC KEY-----"""

# ---------------------------------------------------------------------------
# CapCut voice registry — load từ Voice.json
# ---------------------------------------------------------------------------
_VOICE_JSON_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "capcut-tts-api", "Voice.json"
)

_capcut_voices_cache = None


def load_capcut_voices():
    """Load danh sách voice từ Voice.json (lazy, cache)."""
    global _capcut_voices_cache
    if _capcut_voices_cache is not None:
        return _capcut_voices_cache

    path = os.path.abspath(_VOICE_JSON_PATH)
    if not os.path.exists(path):
        print(f"[CapCut TTS] Voice.json not found at {path}")
        _capcut_voices_cache = []
        return _capcut_voices_cache

    with open(path, "r", encoding="utf-8") as f:
        _capcut_voices_cache = json.load(f)
    print(f"[CapCut TTS] Loaded {len(_capcut_voices_cache)} voices from Voice.json")
    return _capcut_voices_cache


def get_resource_id(voice_type: str) -> str:
    """Tìm resource_id theo voice_type."""
    for v in load_capcut_voices():
        if v.get("voice_type") == voice_type:
            return v["resource_id"]
    # Default fallback — Cô Gái Hoạt Ngôn
    return "7102355709945188865"


# ---------------------------------------------------------------------------
# Crypto / Signing helpers (extracted from capcut_common_task_client.py)
# ---------------------------------------------------------------------------
def _compact_json(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _make_x_ss_stub(body_text):
    return hashlib.md5(body_text.encode("utf-8")).hexdigest()


def _der_len(data, pos):
    first = data[pos]
    pos += 1
    if first < 0x80:
        return first, pos
    nbytes = first & 0x7F
    return int.from_bytes(data[pos: pos + nbytes], "big"), pos + nbytes


def _der_value(data, pos, tag):
    if data[pos] != tag:
        raise ValueError(f"bad DER tag: expected 0x{tag:02x}, got 0x{data[pos]:02x}")
    length, pos = _der_len(data, pos + 1)
    return data[pos: pos + length], pos + length


def _der_int(data, pos):
    raw, pos = _der_value(data, pos, 0x02)
    return int.from_bytes(raw.lstrip(b"\x00"), "big"), pos


def _rsa_public_numbers_from_pem(pem):
    b64 = "".join(line for line in pem.splitlines() if not line.startswith("-----"))
    der = base64.b64decode(b64)
    outer, pos = _der_value(der, 0, 0x30)
    if pos != len(der):
        raise ValueError("trailing data in public key")
    _, pos = _der_value(outer, 0, 0x30)
    bit_string, pos = _der_value(outer, pos, 0x03)
    if pos != len(outer) or not bit_string or bit_string[0] != 0:
        raise ValueError("bad subjectPublicKeyInfo")
    rsa_seq, pos = _der_value(bit_string[1:], 0, 0x30)
    if pos != len(bit_string[1:]):
        raise ValueError("trailing data in RSA public key")
    modulus, pos = _der_int(rsa_seq, 0)
    exponent, pos = _der_int(rsa_seq, pos)
    if pos != len(rsa_seq):
        raise ValueError("trailing integer data in RSA public key")
    return modulus, exponent


def _rsa_encrypt_pkcs1v15(message, pem=TTS_SIGN_PUBLIC_KEY_PEM):
    modulus, exponent = _rsa_public_numbers_from_pem(pem)
    key_len = (modulus.bit_length() + 7) // 8
    msg = message.encode("utf-8") if isinstance(message, str) else bytes(message)
    if len(msg) > key_len - 11:
        raise ValueError("message too long for RSA PKCS#1 v1.5")
    ps_len = key_len - len(msg) - 3
    ps = bytearray()
    while len(ps) < ps_len:
        chunk = secrets.token_bytes(ps_len - len(ps))
        ps.extend(b for b in chunk if b != 0)
    encoded = b"\x00\x02" + bytes(ps[:ps_len]) + b"\x00" + msg
    encrypted = pow(
        int.from_bytes(encoded, "big"), exponent, modulus
    ).to_bytes(key_len, "big")
    return base64.b64encode(encrypted).decode("ascii")


def _escape_xml(text):
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _make_tts_payload_sign(ssml, extra_info, device_id, app_id):
    ssml_md5 = hashlib.md5(ssml.encode("utf-8")).hexdigest()
    sign_input = f"appid:{app_id}&did:{device_id}&creditDisable:false&ssml:{ssml_md5}"
    if extra_info is not None:
        sign_input += f"&extraInfo:{extra_info}"
    return _rsa_encrypt_pkcs1v15(sign_input)


def _make_sign_header(url, appvr, device_time, tdid):
    path = url.split("?", 1)[0]
    sign_str = f"9e2c|{path[-7:]}|3|{appvr}|{device_time}|{tdid}|11ac"
    return hashlib.md5(sign_str.encode("utf-8")).hexdigest()


def _make_trace_id():
    seed = uuid.uuid4().hex[:32]
    return f"00-{seed}-{seed[:16]}-01"


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------
def _common_query(device, babi_param=None, include_region=True):
    q = {
        "app_name": device["app_name"],
        "device_type": device["device_type"],
        "os_version": device["os_version"],
        "channel": device["channel"],
        "version_name": device["version_name"],
        "device_brand": device["device_brand"],
        "device_id": device["device_id"],
        "iid": device["iid"],
        "version_code": device["version_code"],
        "device_platform": device["device_platform"],
        "aid": device["aid"],
    }
    if include_region:
        q["region"] = device["region"]
    if babi_param is not None:
        q["babi_param"] = _compact_json(babi_param)
    return q


def _base_headers(device, body_text, appid=False):
    now = str(int(time.time()))
    headers = {
        "content-type": "application/json",
        "appvr": device["appvr"],
        "ch": device["channel"],
        "device-time": now,
        "lan": device["lan"],
        "loc": device["loc"],
        "pf": device["pf"],
        "sign-ver": "1",
        "tdid": device["tdid"],
        "x-ss-stub": _make_x_ss_stub(body_text),
        "x-ss-dp": device["aid"],
        "x-khronos": now,
        "x-tt-trace-id": _make_trace_id(),
        "user-agent": "Cronet/TTNetVersion:1d7cc3b1 2025-07-16 QuicVersion:52c2b40d 2025-04-03",
        "accept-encoding": "gzip, deflate",
        "store-country-code": device["loc"].lower(),
        "store-country-code-src": "did",
        "is-dispatch-us-ttp": "0",
        "is-app-region-us-ttp": "0",
    }
    if appid:
        headers["app-sdk-version"] = device["appvr"]
        headers["appid"] = device["aid"]
    return headers


def _tts_new_body(text, voice, resource_id, rate, device):
    """Build request body cho TTS task creation."""
    babi = {
        "feature_entrance": "editor",
        "feature_entrance_detail": "editor-feature-text_to_speech",
        "feature_key": "text_to_speech",
        "scenario": "video_editor",
    }
    ssml = (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">\n'
        f'    <voice name="{voice}" mock_tone_info="" platform="sami" '
        f'resource_id="{resource_id}" emotion="" emotion_scale="0" style="" role="" '
        f'moyin_emotion="" is_clone_tone="false" need_subtitle_timestamp="false">\n'
        f'        <prosody rate="{rate}">{_escape_xml(text)}</prosody>\n'
        f'    </voice>\n'
        '</speak>'
    )
    extra_info = _compact_json({"benefit_info": {}})
    payload = {
        "audio_format": "mp3",
        "babi_param": _compact_json(babi),
        "credit_disable": False,
        "extra_info": extra_info,
        "need_merge_voice": False,
        "need_subtitle_timestamp": False,
        "scene": "text_to_speech",
        "ssml": ssml,
    }
    payload["sign"] = _make_tts_payload_sign(
        ssml, extra_info, device["device_id"], device["aid"]
    )
    body = {
        "bind_id": str(uuid.uuid4()),
        "can_queue": True,
        "enter_from": "text_to_speech",
        "tasks": [
            {
                "context": str(uuid.uuid4()),
                "payload": _compact_json(payload),
                "req_key": "sami_text_to_speech",
                "task_version": "v3",
            }
        ],
    }
    return babi, body


# ---------------------------------------------------------------------------
# Main TTS generation function
# ---------------------------------------------------------------------------
async def capcut_tts_generate(
    text: str,
    voice: str = "BV074_streaming",
    resource_id: str = "7102355709945188865",
    rate: str = "1.0",
    output_path: str = "output.mp3",
    timeout: float = 30.0,
    device_override: dict | None = None,
) -> str:
    """
    Tạo audio TTS bằng CapCut API.

    1. Gửi request tạo TTS task
    2. Poll query cho đến khi task hoàn thành
    3. Download audio → lưu vào output_path

    Returns: output_path nếu thành công
    Raises: Exception nếu thất bại hoặc timeout
    """
    device = deepcopy(DEFAULT_DEVICE)
    if device_override:
        device.update(device_override)

    # --- Step 1: Create TTS task ---
    babi, body = _tts_new_body(text, voice, resource_id, rate, device)
    body_text = _compact_json(body)

    path = "/lv/v1/common_task/new"
    query = _common_query(device, babi, include_region=True)
    url = BASE + path + "?" + urlencode(query)

    headers = _base_headers(device, body_text, appid=True)
    lower_headers = {k.lower(): v for k, v in headers.items()}
    if "sign" not in lower_headers:
        headers["sign"] = _make_sign_header(
            url, device["appvr"], lower_headers["device-time"], device["tdid"]
        )

    print(f"[CapCut TTS] Creating task for voice={voice}, text={text[:50]}...")

    # Run blocking requests in thread pool
    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(
        None,
        lambda: requests.post(
            url,
            headers=headers,
            data=body_text.encode("utf-8"),
            timeout=15,
        ),
    )

    if resp.status_code != 200:
        raise Exception(f"[CapCut TTS] Create task failed: HTTP {resp.status_code} — {resp.text[:300]}")

    resp_data = resp.json()
    tasks = resp_data.get("data", {}).get("tasks", [])
    if not tasks:
        raise Exception(f"[CapCut TTS] No tasks in response: {resp_data}")

    task = tasks[0]
    task_id = task.get("id")
    token = task.get("token")

    if not task_id or not token:
        raise Exception(f"[CapCut TTS] Missing task_id/token: {task}")

    print(f"[CapCut TTS] Task created: id={task_id[:20]}...")

    # --- Step 2: Poll for completion ---
    query_body = {
        "tasks": [
            {
                "bind_id": "",
                "id": task_id,
                "req_key": "sami_text_to_speech",
                "task_version": "v3",
                "token": token,
            }
        ]
    }
    query_body_text = _compact_json(query_body)
    query_path = "/lv/v1/common_task/query"
    query_qs = _common_query(device, None, include_region=False)
    query_url = BASE + query_path + "?" + urlencode(query_qs)

    start_time = time.time()
    audio_url = None

    while time.time() - start_time < timeout:
        q_headers = _base_headers(device, query_body_text, appid=True)
        q_lower = {k.lower(): v for k, v in q_headers.items()}
        if "sign" not in q_lower:
            q_headers["sign"] = _make_sign_header(
                query_url, device["appvr"], q_lower["device-time"], device["tdid"]
            )

        q_resp = await loop.run_in_executor(
            None,
            lambda: requests.post(
                query_url,
                headers=q_headers,
                data=query_body_text.encode("utf-8"),
                timeout=15,
            ),
        )

        if q_resp.status_code != 200:
            print(f"[CapCut TTS] Query failed: HTTP {q_resp.status_code}")
            await asyncio.sleep(1)
            continue

        q_data = q_resp.json()
        q_tasks = q_data.get("data", {}).get("tasks", [])
        if not q_tasks:
            await asyncio.sleep(1)
            continue

        q_task = q_tasks[0]
        status = q_task.get("status", "")

        if status in ("done", "succeed"):
            # Tìm audio URL trong output
            output_str = q_task.get("output", "")
            payload_str = q_task.get("payload", "")

            def _find_url(json_str):
                if not json_str:
                    return None
                try:
                    data = json.loads(json_str) if isinstance(json_str, str) else json_str
                    # Check output format
                    url = (
                        data.get("audio_url")
                        or data.get("url")
                        or data.get("audio", {}).get("url")
                    )
                    if url:
                        return url
                    # Check payload format
                    subs = data.get("audio_subtitles", [])
                    if subs and isinstance(subs, list) and len(subs) > 0:
                        return subs[0].get("speech_url")
                except Exception:
                    pass
                return None

            audio_url = _find_url(output_str) or _find_url(payload_str)

            if not audio_url:
                # Thử tìm trong resp_url hoặc trường khác
                audio_url = q_task.get("resp_url") or q_task.get("audio_url")

            if not audio_url:
                raise Exception(
                    f"[CapCut TTS] Task done but no audio URL found. Task output: {str(q_task)[:500]}"
                )
            break

        elif status in ("failed", "error"):
            raise Exception(f"[CapCut TTS] Task failed: {q_task}")

        # Chưa xong, chờ rồi poll tiếp
        print(f"[CapCut TTS] Task status: {status}, waiting...")
        await asyncio.sleep(1.0)

    if not audio_url:
        raise Exception(f"[CapCut TTS] Timeout after {timeout}s waiting for task {task_id[:20]}...")

    # --- Step 3: Download audio ---
    print(f"[CapCut TTS] Downloading audio from {audio_url[:80]}...")

    dl_resp = await loop.run_in_executor(
        None,
        lambda: requests.get(audio_url, timeout=30),
    )

    if dl_resp.status_code != 200:
        raise Exception(f"[CapCut TTS] Download failed: HTTP {dl_resp.status_code}")

    with open(output_path, "wb") as f:
        f.write(dl_resp.content)

    file_size = os.path.getsize(output_path)
    if file_size < 100:
        raise Exception(f"[CapCut TTS] Downloaded file too small: {file_size} bytes")

    print(f"[CapCut TTS] ✓ Saved {file_size} bytes → {output_path}")
    return output_path
