"""
chinese_text_remover.py
-----------------------
Phát hiện và xóa text tiếng Trung khỏi video gốc.

Pipeline:
  1. EasyOCR detect text tiếng Trung (ch_sim) trên frame
  2. Lọc chỉ giữ bbox chứa ký tự Hán (\\u4e00-\\u9fff)
  3. Tạo mask (dilate để cover rộng hơn)
  4. cv2.inpaint xóa text, thay bằng nền xung quanh
  5. Ghi frame đã clean → video output (không có audio)
  6. Mux audio gốc vào video output bằng ffmpeg

Lazy-load EasyOCR reader lần đầu gọi (~200MB models).
"""

import os
import re
import subprocess
import numpy as np

# Lazy imports — chỉ load khi thực sự cần
cv2 = None
easyocr = None

_ocr_reader = None

# Regex cho ký tự Hán (CJK Unified Ideographs)
_CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")


def _ensure_cv2():
    """Lazy import OpenCV."""
    global cv2
    if cv2 is None:
        import cv2 as _cv2
        cv2 = _cv2
    return cv2


def _get_ocr_reader():
    """Lazy-load EasyOCR reader cho Simplified Chinese.

    Chỉ load model lần đầu gọi, cache lại cho lần sau.
    """
    global _ocr_reader, easyocr
    if _ocr_reader is None:
        import easyocr as _easyocr
        easyocr = _easyocr
        print("[ChineseRemover] Loading EasyOCR model 'ch_sim' (lần đầu sẽ tải ~200MB)...")
        _ocr_reader = easyocr.Reader(
            ["ch_sim"],
            gpu=False,       # CPU cho tương thích; set True nếu có CUDA
            verbose=False,
        )
        print("[ChineseRemover] EasyOCR ready")
    return _ocr_reader


def _contains_chinese(text: str) -> bool:
    """Kiểm tra string có chứa ký tự Hán không."""
    return bool(_CHINESE_CHAR_RE.search(str(text or "")))


def detect_chinese_text_regions(frame, reader=None):
    """Detect vùng chứa text tiếng Trung trên 1 frame.

    Returns:
        list[tuple]: Mỗi phần tử là ((x1, y1, x2, y2), text, confidence)
                     trong đó (x1,y1) = top-left, (x2,y2) = bottom-right.
    """
    if reader is None:
        reader = _get_ocr_reader()

    # EasyOCR trả về list[ (bbox, text, confidence) ]
    # bbox = [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]  (4 corners)
    results = reader.readtext(frame, detail=1, paragraph=False)

    chinese_regions = []
    for bbox, text, conf in results:
        # Chỉ giữ text chứa ký tự Trung
        if not _contains_chinese(text):
            continue

        # Chuyển 4-corner bbox thành rectangle (x1,y1,x2,y2)
        pts = np.array(bbox, dtype=np.int32)
        x1 = int(pts[:, 0].min())
        y1 = int(pts[:, 1].min())
        x2 = int(pts[:, 0].max())
        y2 = int(pts[:, 1].max())

        chinese_regions.append(((x1, y1, x2, y2), text, conf))

    return chinese_regions


def create_inpaint_mask(frame_shape, bboxes, dilate_px=12):
    """Tạo binary mask từ danh sách bounding box.

    Args:
        frame_shape: (h, w) hoặc (h, w, c)
        bboxes: list[(x1, y1, x2, y2)]
        dilate_px: số pixel dilate mask để cover sạch hơn

    Returns:
        np.ndarray mask (h, w), dtype=uint8, giá trị 0 hoặc 255
    """
    cv = _ensure_cv2()

    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for (x1, y1, x2, y2) in bboxes:
        # Clamp to frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        cv.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # Dilate để mở rộng mask, cover viền text còn sót
    if dilate_px > 0:
        kernel = cv.getStructuringElement(
            cv.MORPH_RECT,
            (dilate_px * 2 + 1, dilate_px * 2 + 1),
        )
        mask = cv.dilate(mask, kernel, iterations=1)

    return mask


def remove_chinese_text_from_video(
    input_video: str,
    output_video: str,
    ocr_interval: int = 5,
    inpaint_radius: int = 7,
    dilate_px: int = 12,
    progress_callback=None,
):
    """Xóa text tiếng Trung khỏi video.

    Args:
        input_video: đường dẫn video gốc
        output_video: đường dẫn video output (không có audio)
        ocr_interval: chạy OCR mỗi N frame (mặc định 5)
        inpaint_radius: bán kính inpaint của cv2.inpaint
        dilate_px: dilate mask bao nhiêu pixel
        progress_callback: function(current_frame, total_frames) để báo progress

    Returns:
        str: đường dẫn output_video đã tạo

    Raises:
        Exception: nếu không mở được video hoặc lỗi xử lý
    """
    cv = _ensure_cv2()
    reader = _get_ocr_reader()

    cap = cv.VideoCapture(input_video)
    if not cap.isOpened():
        raise Exception(f"Không mở được video: {input_video}")

    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    print(f"[ChineseRemover] {width}x{height} @ {fps:.1f}fps, {total_frames} frames, "
          f"OCR mỗi {ocr_interval} frame")

    # Video writer — ghi ra file tạm không audio
    # Dùng mp4v codec (H.264 cần thêm ffmpeg backend)
    temp_video = output_video + ".tmp.mp4"
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(temp_video, fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise Exception(f"Không tạo được video writer: {temp_video}")

    current_mask = None
    frame_idx = 0
    frames_with_chinese = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Chạy OCR mỗi ocr_interval frame
            if frame_idx % ocr_interval == 0:
                regions = detect_chinese_text_regions(frame, reader)

                if regions:
                    bboxes = [r[0] for r in regions]
                    current_mask = create_inpaint_mask(
                        frame.shape, bboxes, dilate_px=dilate_px
                    )
                    frames_with_chinese += 1

                    if frame_idx % (ocr_interval * 10) == 0:
                        texts = [r[1] for r in regions]
                        print(f"[ChineseRemover] frame {frame_idx}: "
                              f"found {len(regions)} Chinese text regions: "
                              f"{texts[:3]}{'...' if len(texts) > 3 else ''}")
                else:
                    current_mask = None

            # Inpaint frame nếu có mask
            if current_mask is not None:
                frame = cv.inpaint(
                    frame, current_mask, inpaint_radius, cv.INPAINT_NS
                )

            writer.write(frame)
            frame_idx += 1

            # Report progress
            if progress_callback and frame_idx % 50 == 0:
                progress_callback(frame_idx, total_frames)

    finally:
        cap.release()
        writer.release()

    print(f"[ChineseRemover] Xử lý xong {frame_idx} frames, "
          f"{frames_with_chinese} lần detect Chinese text")

    # Mux audio từ video gốc vào video đã clean bằng ffmpeg
    print(f"[ChineseRemover] Muxing audio from original video...")
    mux_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", temp_video,       # video đã clean (không audio)
        "-i", input_video,      # video gốc (lấy audio)
        "-map", "0:v:0",        # video từ file clean
        "-map", "1:a?",         # audio từ file gốc (nếu có)
        "-c:v", "copy",         # không re-encode video
        "-c:a", "copy",         # không re-encode audio
        "-shortest",
        output_video,
    ]

    proc = subprocess.run(
        mux_cmd, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )

    # Dọn file tạm
    try:
        os.remove(temp_video)
    except OSError:
        pass

    if proc.returncode != 0:
        raise Exception(
            f"ffmpeg mux failed: {proc.stderr[-500:]}"
        )

    print(f"[ChineseRemover] Done → {output_video}")
    return output_video
