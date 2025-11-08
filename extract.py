"""
extract.py
-----------
Core extraction logic for Exam Form Scanner.
This module handles:
- Image preprocessing, deskew, and perspective correction
- Zone-based cropping
- Google Vision OCR
- Postprocessing and normalization of extracted fields
"""

import cv2
import numpy as np
import re
import json
import os
from google.cloud import vision

# Default field zones (fallback)
DEFAULT_ZONES = {
    "full_name_line": [0.06, 0.68, 0.90, 0.735],
    "first_name_box": [0.06, 0.695, 0.30, 0.735],
    "middle_name_box": [0.30, 0.695, 0.56, 0.735],
    "surname_box": [0.56, 0.695, 0.90, 0.735],
    "class_box": [0.06, 0.755, 0.18, 0.795],
    "phone_box": [0.46, 0.755, 0.78, 0.795],
    "school_line": [0.06, 0.815, 0.90, 0.865],
    "english_medium_checkbox": [0.06, 0.645, 0.10, 0.675],
    "vernacular_checkbox": [0.40, 0.645, 0.44, 0.675],
    "total_fee_box": [0.86, 0.695, 0.98, 0.82]
}

TARGET_LONG = 2400  # target max side for resize
VISION_CLIENT = None


# ======================
# Vision initialization
# ======================
def init_vision():
    global VISION_CLIENT
    if VISION_CLIENT is None:
        VISION_CLIENT = vision.ImageAnnotatorClient()
    return VISION_CLIENT


# ======================
# Image utilities
# ======================
def load_and_resize(path_or_bytes, target_long=TARGET_LONG):
    """Load image (bytes or path), resize long side to target."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        arr = np.frombuffer(path_or_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(path_or_bytes)
    if img is None:
        raise ValueError("Could not load image")
    h, w = img.shape[:2]
    scale = target_long / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def detect_page_corners(image):
    """Detect outermost page corners for perspective correction."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 21, 10)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype("float32")
    return None


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def crop_zone(page, zone):
    """Crop region based on normalized coordinates."""
    h, w = page.shape[:2]
    x1 = int(zone[0] * w)
    y1 = int(zone[1] * h)
    x2 = int(zone[2] * w)
    y2 = int(zone[3] * h)
    return page[y1:y2, x1:x2]


def preprocess_for_ocr(img):
    """Enhance contrast and denoise."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.medianBlur(enhanced, 3)


# ======================
# OCR wrapper
# ======================
def ocr_image_np(img_np):
    """Send numpy image to Google Vision for text detection."""
    client = init_vision()
    _, buf = cv2.imencode('.png', img_np)
    image = vision.Image(content=buf.tobytes())
    resp = client.document_text_detection(image=image)

    text = ""
    confs = []
    try:
        if resp.full_text_annotation and resp.full_text_annotation.text:
            text = resp.full_text_annotation.text.strip()
        for page in resp.full_text_annotation.pages:
            for block in page.blocks:
                for para in block.paragraphs:
                    for word in para.words:
                        if hasattr(word, "confidence"):
                            confs.append(word.confidence)
    except Exception:
        pass

    avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
    return text, avg_conf


# ======================
# Normalizers
# ======================
def normalize_phone(raw):
    if not raw:
        return None
    s = raw.replace("O", "0").replace("o", "0").replace("l", "1")
    digits = re.sub(r"\\D", "", s)
    if len(digits) >= 10:
        return digits[-10:]
    return None


def normalize_class(raw):
    if not raw:
        return None
    s = raw.lower()
    m = re.search(r"(\\d+)", s)
    if m:
        return m.group(1)
    mapping = {
        "first": "1",
        "second": "2",
        "third": "3",
        "fourth": "4"
    }
    for k, v in mapping.items():
        if k in s:
            return v
    return s.upper()


# ======================
# Main pipeline
# ======================
def extract_fields_from_image_bytes(image_bytes, zones=None):
    """Main entry — extract all fields from a single image."""
    zones = zones or DEFAULT_ZONES
    img = load_and_resize(image_bytes)
    pts = detect_page_corners(img)
    page = four_point_transform(img, pts) if pts is not None else img

    results = {}
    confidences = {}

    for key, zone in zones.items():
        crop = crop_zone(page, zone)
        crop = cv2.resize(crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        prep = preprocess_for_ocr(crop)
        text, conf = ocr_image_np(prep)
        results[key] = text.strip()
        confidences[key] = conf

    # Apply normalization
    if "phone_box" in results:
        results["phone"] = normalize_phone(results["phone_box"])
    if "class_box" in results:
        results["class"] = normalize_class(results["class_box"])

    results["_conf_avg"] = sum(confidences.values()) / max(1, len(confidences))
    results["_per_field_conf"] = confidences
    return results


if __name__ == "__main__":
    print("This module is meant to be imported by app.py or batch_processor.py.")
