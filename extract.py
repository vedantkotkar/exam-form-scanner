# extract.py
import re
import os
import cv2
import requests
import pytesseract
from PIL import Image, UnidentifiedImageError
import fitz  # PyMuPDF
import streamlit as st

# ---------- Image / PDF helpers ----------
def pdf_to_images(pdf_path):
    """Convert each PDF page to a JPG and return list of image paths."""
    images = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Could not open PDF: {e}")
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)
        out = f"{pdf_path}.page{i}.jpg"
        pix.save(out)
        images.append(out)
    doc.close()
    return images

def preprocess_image(file_path):
    """Simple preprocessing: read, convert to grayscale and threshold."""
    img = cv2.imread(file_path)
    if img is None:
        # If cv2 fails, return original path and let downstream handle
        return file_path
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    temp_path = file_path + ".proc.jpg"
    cv2.imwrite(temp_path, thresh)
    return temp_path

# ---------- OCR.space helper ----------
def ocr_space_file(image_path, api_key, language="eng"):
    """
    Calls OCR.space API to extract text from image file. Returns (text, error).
    """
    url_api = "https://api.ocr.space/parse/image"
    with open(image_path, "rb") as f:
        payload = {
            'isOverlayRequired': False,
            'apikey': api_key,
            'language': language
        }
        files = {
            'file': f
        }
        try:
            r = requests.post(url_api, files=files, data=payload, timeout=60)
            result = r.json()
        except Exception as e:
            return "", f"ocr.space request failed: {e}"

        if result.get("IsErroredOnProcessing"):
            # Try to return error message if available
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

# ---------- main extraction from a single image path ----------
def extract_from_image_path(img_path):
    """
    Extract text from one image path and parse the required fields.
    Returns (record_dict, error_message)
    """
    proc = preprocess_image(img_path)

    # Try local pytesseract first (will work only if binary exists)
    text = ""
    try:
        # PIL Image open may raise UnidentifiedImageError; handle it
        try:
            text = pytesseract.image_to_string(Image.open(proc))
        except UnidentifiedImageError:
            # fallback: use cv2 image directly
            img_cv = cv2.imread(proc)
            if img_cv is not None:
                text = pytesseract.image_to_string(img_cv)
            else:
                text = ""
    except pytesseract.TesseractNotFoundError:
        text = ""  # will trigger cloud OCR fallback

    # If pytesseract gave empty or very short text, try OCR.space using API key from secrets
    if not text or len(text.strip()) < 20:
        api_key = None
        try:
            api_key = st.secrets["OCRSPACE_API_KEY"]
        except Exception:
            api_key = None
        if api_key:
            text, err = ocr_space_file(proc, api_key)
            if err:
                return None, f"OCR.space failed: {err}"
        else:
            return None, "Tesseract not available in environment and OCRSPACE_API_KEY not set in Streamlit secrets."

    txt = " ".join(text.split())
    txt_lower = txt.lower()

    # Name extraction (several patterns)
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

    # Class / Std capture
    class_match = safe_search(r"(?:Class|Std\.?|Standard)[:\-]?\s*([A-Za-z0-9\-\s]+)", txt, flags=re.IGNORECASE)
    class_std = class_match.replace("Class", "").strip() if class_match else ""

    # Mobile extraction (10 digit preferred)
    mobile = safe_search(r"(?:Mob(?:il)?e|Mob\.|Mob\/Tel|Tel|Mobile No|Ph\.?)[:\-]?\s*(\d{10})", txt, flags=re.IGNORECASE)
    if not mobile:
        mobile = safe_search(r"(?:Mob(?:il)?e|Mob\.|Tel|Ph\.?)[:\-]?\s*(\d{8,10})", txt, flags=re.IGNORECASE)

    # School name
    school = safe_search(r"(?:School|Name of School|Institution)[:\-]?\s*(.+?)\s+(?:Medium|Mob|Tel|Fee|Total)", txt, flags=re.IGNORECASE)

    # Medium detection
    medium = "English" if re.search(r"\benglish\b", txt_lower) else "Vernacular"

    result = {
        "First Name": first,
        "Middle Name": middle,
        "Surname": surname,
        "Class": class_std,
        "Mobile": mobile,
        "School": school,
        "Medium": medium,
        "RawTextSample": txt[:300]
    }
    return result, None

# ---------- top-level entry ----------
def extract_data(file_path):
    """
    Accept a path to uploaded file. If it's a PDF, convert to images and process each page.
    Returns (records_list, errors_list)
    """
    records = []
    errors = []
    ext = os.path.splitext(file_path)[1].lower()
    image_paths = []
    if ext == ".pdf":
        try:
            image_paths = pdf_to_images(file_path)
        except Exception as e:
            errors.append(f"PDF conversion failed for {file_path}: {e}")
            return records, errors
    else:
        image_paths = [file_path]

    for img in image_paths:
        rec, err = extract_from_image_path(img)
        if rec:
            rec["File"] = os.path.basename(file_path)
            records.append(rec)
        else:
            errors.append(err or f"Unknown error processing {img}")

    return records, errors
