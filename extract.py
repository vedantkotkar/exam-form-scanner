# extract.py
import re
import os
import cv2
import requests
import streamlit as st

# ---------- OCR.space helper ----------
def ocr_space_file(image_path, api_key, language="eng"):
    """Call OCR.space API to extract text from an image file. Returns (text, error)."""
    url_api = "https://api.ocr.space/parse/image"
    with open(image_path, "rb") as f:
        payload = {
            'isOverlayRequired': False,
            'apikey': api_key,
            'language': language
        }
        files = {'file': f}
        try:
            r = requests.post(url_api, files=files, data=payload, timeout=60)
            result = r.json()
        except Exception as e:
            return "", f"ocr.space request failed: {e}"

        if result.get("IsErroredOnProcessing"):
            err = result.get("ErrorMessage")
            if isinstance(err, list):
                err = err[0] if err else "Unknown OCR.space error"
            return "", f"ocr.space error: {err}"
        parsed = result.get("ParsedResults")[0]
        return parsed.get("ParsedText", ""), None

# ---------- small helpers ----------
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

# ---------- very small preprocessing to improve OCR.space results ----------
def preprocess_image_cv(image_path):
    """Read image with OpenCV, convert to grayscale and write a temp file for upload."""
    try:
        import cv2
    except Exception:
        return image_path
    img = cv2.imread(image_path)
    if img is None:
        return image_path
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gentle blur + Otsu
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    out = image_path + ".proc.jpg"
    cv2.imwrite(out, thresh)
    return out

# ---------- main extraction ----------
def extract_from_image_path(img_path):
    """Extract text from image using OCR.space and parse fields. Returns (record, error)."""
    # Preprocess for a cleaner image
    proc = preprocess_image_cv(img_path)

    api_key = None
    try:
        api_key = st.secrets["OCRSPACE_API_KEY"]
    except Exception:
        return None, "OCRSPACE_API_KEY not set in Streamlit secrets."

    text, err = ocr_space_file(proc, api_key)
    if err:
        return None, err

    txt = " ".join(text.split())
    txt_lower = txt.lower()

    # Name patterns
    name_patterns = [
        r"Name of the Student[:\-]?\s*(.+?)\s+(?:Class|Std|School|Mob|Tel)",
        r"Name[:\-]?\s*(.+?)\s+(?:Class|Std|School|Mob|Tel)",
        r"Student Name[:\-]?\s*(.+?)\s+(?:Class|Std|School|Mob|Tel)",
    ]
    full_name = ""
    for p in name_patterns:
        full_name = safe_search(p, txt, flags=re.IGNORECASE)
        if full_name:
            break
    first, middle, surname = normalize_name(full_name or "")

    class_match = safe_search(r"(?:Class|Std\.?|Standard)[:\-]?\s*([A-Za-z0-9\-\s]+)", txt, flags=re.IGNORECASE)
    class_std = class_match.replace("Class", "").strip() if class_match else ""

    mobile = safe_search(r"(?:Mob(?:il)?e|Mob\.|Mob\/Tel|Tel|Mobile No|Ph\.?)[:\-]?\s*(\d{10})", txt, flags=re.IGNORECASE)
    if not mobile:
        mobile = safe_search(r"(?:Mob(?:il)?e|Mob\.|Tel|Ph\.?)[:\-]?\s*(\d{8,10})", txt, flags=re.IGNORECASE)

    school = safe_search(r"(?:School|Name of School|Institution)[:\-]?\s*(.+?)\s+(?:Medium|Mob|Tel|Fee|Total)", txt, flags=re.IGNORECASE)

    medium = "English" if re.search(r"\benglish\b", txt_lower) else "Vernacular"

    result = {
        "First Name": first,
        "Middle Name": middle,
        "Surname": surname,
        "Class": class_std,
        "Mobile": mobile,
        "School": school,
        "Medium": medium,
        "RawTextSample": txt[:350]
    }
    return result, None

def extract_data(file_path):
    """Top-level: accepts a path to an image file. Returns ([records], [errors])"""
    records = []
    errors = []
    # If PDF, ask user to upload page-images for now. (We removed PDF handling to keep things stable.)
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        errors.append("PDFs are not supported in this simplified build. Please upload page images (jpg/png).")
        return records, errors

    rec, err = extract_from_image_path(file_path)
    if rec:
        rec["File"] = os.path.basename(file_path)
        records.append(rec)
    else:
        errors.append(err or f"Unknown error processing {file_path}")
    return records, errors
