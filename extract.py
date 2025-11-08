# add at top
import requests
import streamlit as st

# new helper: call OCR.space
def ocr_space_file(image_path, api_key):
    """Send image file to OCR.space and return extracted text or error."""
    url_api = "https://api.ocr.space/parse/image"
    with open(image_path, "rb") as f:
        payload = {
            'isOverlayRequired': False,
            'apikey': api_key,
            'language': 'eng'
        }
        files = {
            'file': f
        }
        try:
            r = requests.post(url_api, files=files, data=payload, timeout=60)
            result = r.json()
            if result.get("IsErroredOnProcessing"):
                return "", "ocr.space error: " + result.get("ErrorMessage", ["Unknown"])[0]
            parsed = result.get("ParsedResults")[0]
            return parsed.get("ParsedText", ""), None
        except Exception as e:
            return "", f"ocr.space request failed: {e}"

# modify extract_from_image_path to try pytesseract then fallback to OCR.space
def extract_from_image_path(img_path):
    """Extract text from one image path and parse fields."""
    proc = preprocess_image(img_path)
    text = ""
    # Try pytesseract first (works if binary exists) - safe try
    try:
        text = pytesseract.image_to_string(Image.open(proc))
    except Exception:
        text = ""
    # If empty or very short, use OCR.space (via streamlit secrets)
    if not text or len(text.strip()) < 20:
        api_key = None
        try:
            api_key = st.secrets["OCRSPACE_API_KEY"]
        except Exception:
            api_key = None
        if api_key:
            text, err = ocr_space_file(proc, api_key)
            if err:
                # return error to be shown
                return None, err
        else:
            # no API key set; return error so user can set secrets
            return None, "Tesseract not available and OCR API key not set. Add OCRSPACE_API_KEY in Streamlit secrets."
    # proceed with parsing as before
    txt = " ".join(text.split())
    txt_lower = txt.lower()

    # name extraction
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
        "RawTextSample": txt[:300]
    }
    return result, None
