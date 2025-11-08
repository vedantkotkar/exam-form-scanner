# extract.py
import os
import re
import io
import json
import cv2
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account

# ---------- CONFIG ----------
CROP_RATIO = 0.70   # keep bottom 70% (increase if fields are higher on form)
MIN_UPPERCASE_WORDS_FOR_NAME = 2
# ---------- /CONFIG ----------

# ---------- Google Vision client ----------
def get_vision_client():
    try:
        service_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_JSON"])
        credentials = service_account.Credentials.from_service_account_info(service_info)
        client = vision.ImageAnnotatorClient(credentials=credentials)
        return client
    except Exception as e:
        st.error(f"Vision client setup failed: {e}")
        return None

# ---------- Image preprocessing helpers ----------
def crop_bottom(image: np.ndarray, ratio=CROP_RATIO):
    h = image.shape[0]
    start = int(h * (1.0 - ratio))
    return image[start:, :]

def enhance_image_for_ocr(pil_img: Image.Image):
    # convert to numpy RGB
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # crop bottom area where fields are (keeps more region)
    cropped = crop_bottom(gray, CROP_RATIO)

    # resize modestly if small
    if cropped.shape[0] < 800:
        scale = int(800 / max(1, cropped.shape[0]))
        cropped = cv2.resize(cropped, (cropped.shape[1]*scale, cropped.shape[0]*scale), interpolation=cv2.INTER_LINEAR)

    # slight denoise
    denoised = cv2.fastNlMeansDenoising(cropped, None, 7, 7, 21)

    # adaptive threshold (works better for mixed lighting / handwriting)
    th = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)

    # morphological close to join broken characters in boxes
    kernel = np.ones((2,2), np.uint8)
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    # final upscale
    h, w = closed.shape
    if h < 1200:
        closed = cv2.resize(closed, (w*2, h*2), interpolation=cv2.INTER_LINEAR)

    return closed

# ---------- OCR with Google Vision ----------
def ocr_with_vision(np_img):
    client = get_vision_client()
    if client is None:
        raise RuntimeError("Google Vision not configured")

    success, encoded = cv2.imencode(".jpg", np_img)
    if not success:
        raise RuntimeError("Failed to encode image for OCR")

    content = encoded.tobytes()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")

    texts = response.text_annotations
    if not texts:
        return ""
    return texts[0].description

# ---------- Text cleaning helpers ----------
BOILERPLATE_PATTERNS = [
    r"practice book fees", r"total", r"pune", r"representative", r"edufit", r"administrative office",
    r"contact", r"phone", r"mobile", r"email", r"www\.", r"http"
]

def remove_boilerplate(text):
    t = text
    for p in BOILERPLATE_PATTERNS:
        t = re.sub(p, " ", t, flags=re.IGNORECASE)
    # remove long numeric sequences that are not useful (e.g., receipt numbers)
    t = re.sub(r"\b\d{6,}\b", " ", t)
    # collapse whitespace
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

# ---------- Parsing heuristics ----------
def split_name(full_name):
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

def extract_fields_from_text(raw_text):
    txt = raw_text.replace("\n", " ").strip()
    txt_up = txt.upper()

    # remove repeated footer/boilerplate which confuses parsing
    txt_clean = remove_boilerplate(txt_up)

    result = {
        "First Name": "",
        "Middle Name": "",
        "Surname": "",
        "Class": "",
        "Mobile": "",
        "School Name": "",
        "Medium": "",
    }

    # mobile: robust extraction of 10-digit numbers
    m = re.search(r"(?:\+?91[\-\s]?)?(\b\d{10}\b)", txt_clean)
    if m:
        result["Mobile"] = m.group(1)

    # medium detection (English / Marathi / Hindi)
    if re.search(r"\bENGLISH\b", txt_clean):
        result["Medium"] = "English"
    elif re.search(r"\bMARATHI\b", txt_clean):
        result["Medium"] = "Marathi"
    elif re.search(r"\bHINDI\b", txt_clean):
        result["Medium"] = "Hindi"

    # Class extraction
    c = re.search(r"(?:CLASS|STD|STANDARD)[:\-\s]*([A-Z0-9\/\s]{1,8})", txt_clean)
    if c:
        result["Class"] = c.group(1).strip()

    # School extraction - try label first, then fallback to capitalized phrase
    s = re.search(r"(?:NAME\s*OF\s*SCHOOL|SCHOOL|INSTITUTION)[:\-\s]*([A-Z0-9\.\'\-\s]{4,80})", txt_clean)
    if s:
        result["School Name"] = s.group(1).strip()
    else:
        # heuristic: pick long capitalized phrase (not purely numbers) before Medium or Mobile
        heur = re.search(r"([A-Z][A-Z\.\'\-\s]{6,80})\s+(?:ENGLISH|MARATHI|HINDI|\d{10}|CLASS|STD)", txt_clean)
        if heur:
            result["School Name"] = heur.group(1).strip()

    # Name extraction: label first
    name_patterns = [
        r"NAME\s*OF\s*THE\s*STUDENT[:\-\s]*([A-Z\s]{3,80})",
        r"STUDENT\s*NAME[:\-\s]*([A-Z\s]{3,80})",
        r"NAME[:\-\s]*([A-Z\s]{3,80})"
    ]
    full_name = ""
    for p in name_patterns:
        mname = re.search(p, txt_clean)
        if mname:
            full_name = mname.group(1).strip()
            break

    # fallback: find nearest uppercase sequence with multiple words (likely name)
    if not full_name:
        # find sequences of 2-4 uppercase words of length>=2
        caps = re.findall(r"\b([A-Z]{2,}\s+[A-Z]{2,}(?:\s+[A-Z]{2,})?)\b", txt_clean)
        # filter out ones that are likely to be school/headers by length and presence of common words
        candidates = []
        for cand in caps:
            if len(cand) < 120 and not re.search(r"LEVEL|COMPETITION|MARATHON|ENGLISH|SCHOOL|PRACTICE|FEES", cand):
                candidates.append(cand.strip())
        if candidates:
            # pick the candidate nearest to the word "CLASS" or "STD" if possible
            if re.search(r"(CLASS|STD)", txt_clean):
                # locate index of CLASS
                idx = txt_clean.find("CLASS")
                best = None
                best_dist = None
                for cand in candidates:
                    pos = txt_clean.find(cand)
                    if pos >= 0:
                        dist = abs(pos - idx)
                        if best is None or dist < best_dist:
                            best = cand
                            best_dist = dist
                if best:
                    full_name = best
            else:
                full_name = candidates[0]

    # set name fields
    if full_name:
        first, middle, surname = split_name(full_name)
        result["First Name"] = first
        result["Middle Name"] = middle
        result["Surname"] = surname

    return result, txt_clean

# ---------- Core extraction for a single file ----------
def extract_data(file_path):
    """
    Input: local file path to an image (jpg/png)
    Output: (records_list, error_or_None)
    """
    try:
        pil = Image.open(file_path).convert("RGB")
    except Exception as e:
        return [], f"Failed to open image: {e}"

    # preprocess and OCR
    proc = enhance_image_for_ocr(pil)
    try:
        raw_text = ocr_with_vision(proc)
    except Exception as e:
        return [], f"OCR error: {e}"

    fields, cleaned_text = extract_fields_from_text(raw_text)
    fields["RawTextSample"] = (raw_text[:350] + "...") if len(raw_text) > 350 else raw_text
    # confidence rules
    confidence = "High"
    if not fields["Mobile"] or not fields["First Name"]:
        confidence = "Low"
    fields["Confidence"] = confidence

    # ensure File column
    fields["File"] = os.path.basename(file_path)
    return [fields], None

# ---------- Robust batch processor ----------
def process_files(uploaded_items):
    """Accepts list of file-path strings or Streamlit UploadedFile objects."""
    all_records = []
    errors = []

    for item in uploaded_items:
        # item can be a path string or UploadedFile
        if isinstance(item, str):
            file_path = item
            display_name = os.path.basename(file_path)
            temp_created = False
        else:
            # streamlit uploaded file-like object
            display_name = getattr(item, "name", "uploaded_file")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(display_name)[1] or ".jpg")
            try:
                tmp.write(item.read())
                tmp.close()
                file_path = tmp.name
                temp_created = True
            except Exception as e:
                errors.append({"file": display_name, "error": f"Failed to save upload: {e}"})
                continue

        # extract
        try:
            recs, err = extract_data(file_path)
            if err:
                errors.append({"file": display_name, "error": err})
            else:
                for r in recs:
                    # prefer display name
                    r["File"] = display_name
                    all_records.append(r)
        except Exception as e:
            errors.append({"file": display_name, "error": str(e)})

        # cleanup temp file created for UploadedFile
        if not isinstance(item, str) and temp_created:
            try:
                os.remove(file_path)
            except Exception:
                pass

    df = pd.DataFrame(all_records) if all_records else pd.DataFrame()
    return df, errors
