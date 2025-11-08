# extract.py
import os
import re
import json
import cv2
import requests
import streamlit as st
from PIL import Image
import numpy as np

# Optional Google Vision client libraries
try:
    from google.cloud import vision
    from google.oauth2 import service_account
    HAVE_GOOGLE = True
except Exception:
    HAVE_GOOGLE = False

# ------------------ Google Vision helper ------------------
def vision_text_from_image(image_path, service_account_info):
    """
    Uses Google Cloud Vision to get text from an image.
    service_account_info: Python dict parsed from service account JSON.
    Returns (text, error)
    """
    try:
        credentials = service_account.Credentials.from_service_account_info(service_account_info)
        client = vision.ImageAnnotatorClient(credentials=credentials)
        with open(image_path, "rb") as f:
            content = f.read()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        if response.error.message:
            return "", f"Google Vision error: {response.error.message}"
        texts = response.text_annotations
        if not texts:
            return "", None
        return texts[0].description, None
    except Exception as e:
        return "", f"Google Vision exception: {e}"

# ------------------ OCR.space helper (fallback) ------------------
def ocr_space_file(image_path, api_key, language='eng', max_size_kb=800):
    url_api = "https://api.ocr.space/parse/image"
    try:
        im = Image.open(image_path)
        size_kb = os.path.getsize(image_path) / 1024
        if size_kb > max_size_kb:
            compressed_path = image_path + "_compressed.jpg"
            quality = 80 if size_kb < 3000 else 60
            im.convert("RGB").save(compressed_path, "JPEG", optimize=True, quality=quality)
            image_path = compressed_path
    except Exception:
        pass

    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {"apikey": api_key, "language": language, "isOverlayRequired": False, "scale": True}
        for attempt in range(2):
            try:
                resp = requests.post(url_api, files=files, data=data, timeout=120)
                result = resp.json()
                if result.get("IsErroredOnProcessing"):
                    err = result.get("ErrorMessage")
                    if isinstance(err, list):
                        err = err[0] if err else "Unknown OCR.space error"
                    return "", f"OCR.space error: {err}"
                parsed = result.get("ParsedResults")[0]
                return parsed.get("ParsedText", ""), None
            except requests.exceptions.ReadTimeout:
                if attempt == 0:
                    continue
                return "", "OCR.space request timed out twice."
            except Exception as e:
                return "", f"OCR.space request failed: {e}"
    return "", "Unexpected OCR failure"

# ------------------ Preprocess + crop helpers ------------------
def crop_bottom_area(image_path, crop_ratio=0.45):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path
        h, w = img.shape[:2]
        keep_h = int(h * crop_ratio)
        start_y = h - keep_h
        cropped = img[start_y:h, 0:w]
        out_path = image_path + ".crop.jpg"
        cv2.imwrite(out_path, cropped)
        return out_path
    except Exception:
        return image_path

def preprocess_for_ocr(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        blur = cv2.GaussianBlur(norm, (3,3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = np.ones((2,2), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        h, w = morph.shape
        morph = cv2.resize(morph, (w*2, h*2))
        out = image_path + ".proc.jpg"
        cv2.imwrite(out, morph)
        return out
    except Exception:
        return image_path

# ------------------ small parsing helpers ------------------
def safe_search(pattern, text, flags=0):
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else ""

def normalize_name(full_name):
    parts = re.split(r"\s+", (full_name or "").strip())
    parts = [p for p in parts if p]
    first = middle = surname = ""
    if len(parts) >= 3:
        first, middle, surname = parts[0], parts[1], " ".join(parts[2:])
    elif len(parts) == 2:
        first, surname = parts[0], parts[1]
    elif len(parts) == 1:
        first = parts[0]
    return first, middle, surname

# ------------------ main extraction for one image ------------------
def extract_from_image_path(img_path):
    # crop + preprocess
    cropped = crop_bottom_area(img_path, crop_ratio=0.45)
    proc = preprocess_for_ocr(cropped)

    # Try Google Vision if secret is provided and library is present
    google_secret = st.secrets.get("GCP_SERVICE_ACCOUNT_JSON", None)
    text = ""
    if google_secret and HAVE_GOOGLE:
        try:
            service_info = json.loads(google_secret)
            text, err = vision_text_from_image(proc, service_info)
            if err:
                # fallback to OCR.space
                text = ""
        except Exception as e:
            text = ""  # fallback path

    # If Google not used or returned empty, try OCR.space if key exists
    if not text:
        ocr_key = st.secrets.get("OCRSPACE_API_KEY", None)
        if ocr_key:
            text, err = ocr_space_file(proc, ocr_key)
            if err:
                return None, f"OCR.space failed: {err}"
        else:
            return None, "Neither Google Vision configured nor OCR.space key set."

    txt = " ".join(text.split())
    txt_lower = txt.lower()

    # parse Medium
    medium = "English" if re.search(r"\benglish\b", txt_lower) else ("Vernacular" if re.search(r"\bvernacular\b|\bmarathi\b|\bhindi\b", txt_lower) else "")

    # parse mobile
    mobile = ""
    m = re.search(r"(?:\+?91[\-\s]?|0)?\s*(\d{10})", txt)
    if m:
        mobile = m.group(1)
    else:
        m2 = re.search(r"(\d{8,10})", txt)
        if m2:
            mobile = m2.group(1)

    # class
    class_std = safe_search(r"(?:Class|Std|Standard)[:\-\s]*([A-Za-z0-9\s\/\-]{1,12})", txt, flags=re.IGNORECASE)
    if not class_std:
        c_match = re.search(r"\b(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|XI|XII|XI|X)\b", txt, re.IGNORECASE)
        class_std = c_match.group(0) if c_match else ""

    # name
    full_name = safe_search(r"Name of the Student[:\-\s]*([A-Z\s]{3,80})", txt)
    if not full_name:
        full_name = safe_search(r"Name[:\-\s]*([A-Z\s]{3,80})", txt)
    first = middle = surname = ""
    if full_name:
        cleaned = re.sub(r"[^A-Z\s]", " ", full_name).strip()
        parts = re.split(r"\s+", cleaned)
        if len(parts) >= 3:
            first, middle, surname = parts[0], parts[1], " ".join(parts[2:])
        elif len(parts) == 2:
            first, surname = parts[0], parts[1]
        elif len(parts) == 1:
            first = parts[0]

    # school
    school = safe_search(r"(?:Name of School|School|Institution)[:\-\s]*([A-Za-z0-9\.\'\-\s]{5,120})", txt)
    if not school:
        m_school = re.search(r"([A-Z][A-Z0-9\.\'\-\s]{6,120})\s+(?:Medium|Total|Fee|₹)", txt)
        if m_school:
            school = m_school.group(1).strip()

    confidence = "High"
    if not mobile or not first:
        confidence = "Low"

    result = {
        "First Name": first,
        "Middle Name": middle,
        "Surname": surname,
        "Class": class_std,
        "Mobile": mobile,
        "School": school,
        "Medium": medium,
        "Confidence": confidence,
        "RawTextSample": txt[:400]
    }
    return result, None

def extract_data(file_path):
    records = []
    errors = []
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        errors.append("PDF not supported in this build. Upload page images (jpg/png).")
        return records, errors

    rec, err = extract_from_image_path(file_path)
    if rec:
        rec["File"] = os.path.basename(file_path)
        records.append(rec)
    else:
        errors.append(err or f"Unknown processing error for {file_path}")
    return records, errors
