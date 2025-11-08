# extract.py
import pytesseract, re
from PIL import Image
import cv2, os

def preprocess_image(file_path):
    # Basic preprocessing for OCR: grayscale + Otsu threshold
    img = cv2.imread(file_path)
    if img is None:
        return file_path
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    temp_path = file_path + ".proc.jpg"
    cv2.imwrite(temp_path, thresh)
    return temp_path

def safe_search(pattern, text, flags=0):
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else ""

def normalize_name(full_name):
    # Split name into First, Middle, Surname (best effort)
    parts = re.split(r"\s+", full_name.strip())
    parts = [p for p in parts if p]
    first = middle = surname = ""
    if len(parts) >= 3:
        first, middle, surname = parts[0], parts[1], " ".join(parts[2:])
    elif len(parts) == 2:
        first, surname = parts[0], parts[1]
    elif len(parts) == 1:
        first = parts[0]
    return first, middle, surname

def extract_data(file_path):
    proc = preprocess_image(file_path)
    try:
        text = pytesseract.image_to_string(Image.open(proc))
    except Exception:
        text = pytesseract.image_to_string(Image.open(file_path))

    # Normalize whitespace for easier regex
    txt = " ".join(text.split())
    txt_lower = txt.lower()

    # Name: look for common labels
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

    # Class / Std (capture things like '4', '4A', 'Std 4', '4-B')
    class_match = safe_search(r"(?:Class|Std\.?|Standard)[:\-]?\s*([A-Za-z0-9\-\s]+)", txt, flags=re.IGNORECASE)
    class_std = class_match.replace("Class", "").strip() if class_match else ""

    # Mobile: 10 digit numbers (India); fallback to 8- or 9-digit
    mobile = safe_search(r"(?:Mob(?:il)?e|Mob\.|Mob\/Tel|Tel|Mobile No|Ph\.?)[:\-]?\s*(\d{10})", txt, flags=re.IGNORECASE)
    if not mobile:
        mobile = safe_search(r"(?:Mob(?:il)?e|Mob\.|Tel|Ph\.?)[:\-]?\s*(\d{8,10})", txt, flags=re.IGNORECASE)

    # School name
    school = safe_search(r"(?:School|Name of School|Institution)[:\-]?\s*(.+?)\s+(?:Medium|Mob|Tel|Fee|Total)", txt, flags=re.IGNORECASE)

    # Medium detection
    medium = "English" if re.search(r"\benglish\b", txt_lower) else "Vernacular"

    return {
        "File": os.path.basename(file_path),
        "First Name": first,
        "Middle Name": middle,
        "Surname": surname,
        "Class": class_std,
        "Mobile": mobile,
        "School": school,
        "Medium": medium,
        "RawTextSample": txt[:200]  # short sample for debugging
    }
