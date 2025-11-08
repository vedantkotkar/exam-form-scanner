# extract.py
import cv2
import re
import os
import requests
import streamlit as st
from PIL import Image
import numpy as np

# ---------- OCR.space API ----------
def ocr_space_file(image_path, api_key, language='eng'):
    url_api = "https://api.ocr.space/parse/image"
    with open(image_path, 'rb') as f:
        payload = {'apikey': api_key, 'language': language}
        files = {'file': f}
        r = requests.post(url_api, files=files, data=payload, timeout=60)
        result = r.json()
        if result.get("IsErroredOnProcessing"):
            return "", "OCR.space error"
        parsed = result.get("ParsedResults")[0]
        return parsed.get("ParsedText", ""), None

# ---------- IMAGE PREPROCESS ----------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return image_path

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    # remove background tint (white balance)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Threshold to pure black/white for boxed handwriting
    thresh = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Dilation to join broken letters in boxes
    kernel = np.ones((2,2), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Optional upscale (helps boxed text)
    scale = 2
    h, w = morph.shape
    morph = cv2.resize(morph, (w*scale, h*scale))

    out = image_path + ".clean.jpg"
    cv2.imwrite(out, morph)
    return out

# ---------- FIELD PARSING ----------
def safe_search(pattern, text):
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else ""

def extract_data(file_path):
    records, errors = [], []

    api_key = st.secrets.get("OCRSPACE_API_KEY")
    if not api_key:
        errors.append("No OCR API key set.")
        return records, errors

    clean_path = preprocess_image(file_path)
    text, err = ocr_space_file(clean_path, api_key)
    if err:
        errors.append(err)
        return records, errors

    txt = " ".join(text.split())
    txt_lower = txt.lower()

    # ---- MEDIUM ----
    medium = "English" if "english" in txt_lower else ("Vernacular" if "vernacular" in txt_lower else "")

    # ---- MOBILE ----
    mobile = safe_search(r"(?:\+?91[\-\s]?|0)?\s*(\d{10})", txt)

    # ---- CLASS ----
    class_std = safe_search(r"(?:Class|Std|Standard)[:\-\s]*([A-Za-z0-9\s]+)", txt)
    if not class_std:
        # fallback: match typical standards
        c_match = re.search(r"\b(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth)\b", txt, re.IGNORECASE)
        class_std = c_match.group(1).capitalize() if c_match else ""

    # ---- NAME ----
    name_block = safe_search(r"Name of the Student.*?([A-Z\s]{5,40}) Class", txt)
    if not name_block:
        name_block = safe_search(r"Name[:\-\s]*([A-Z\s]{5,40})", txt)

    first = middle = surname = ""
    if name_block:
        parts = re.split(r"\s+", name_block.strip())
        if len(parts) >= 3:
            first, middle, surname = parts[0], parts[1], " ".join(parts[2:])
        elif len(parts) == 2:
            first, surname = parts[0], parts[1]
        elif len(parts) == 1:
            first = parts[0]

    # ---- SCHOOL ----
    school = safe_search(r"Name of School[:\-\s]*([A-Za-z0-9\s\.\'\-]{5,60})", txt)
    if not school:
        school = safe_search(r"School[:\-\s]*([A-Za-z0-9\s\.\'\-]{5,60})", txt)

    # ---- Confidence ----
    confidence = "High"
    if not mobile or not first:
        confidence = "Low"

    record = {
        "File": os.path.basename(file_path),
        "First Name": first,
        "Middle Name": middle,
        "Surname": surname,
        "Class": class_std,
        "Mobile": mobile,
        "School": school,
        "Medium": medium,
        "Confidence": confidence,
        "RawTextSample": txt[:250]
    }

    records.append(record)
    return records, errors
