# extract.py
import pytesseract, re
from PIL import Image, UnidentifiedImageError
import cv2, os
import fitz  # PyMuPDF

def pdf_to_images(pdf_path):
    """Convert each page of pdf to a JPG file, return list of image paths."""
    images = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)  # reasonable DPI
        out = f"{pdf_path}.page{i}.jpg"
        pix.save(out)
        images.append(out)
    doc.close()
    return images

def preprocess_image(file_path):
    img = cv2.imread(file_path)
    if img is None:
        # If cv2 can't read, return original path (PIL might still work)
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

def extract_from_image_path(img_path):
    """Extract text from one image path and parse fields."""
    proc = preprocess_image(img_path)
    # try PIL first
    try:
        text = pytesseract.image_to_string(Image.open(proc))
    except UnidentifiedImageError:
        # fallback to cv2 read + pytesseract (less reliable but better than crash)
        img = cv2.imread(proc)
        if img is None:
            return None, f"Could not read image {img_path}"
        text = pytesseract.image_to_string(img)
    except Exception as e:
        # generic fallback
        img = cv2.imread(proc)
        if img is None:
            return None, f"OCR error for {img_path}: {e}"
        text = pytesseract.image_to_string(img)

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

def extract_data(file_path):
    """
    Accept a path to an uploaded file. If PDF -> convert pages to images and process each.
    Returns a list of records (one per processed image/page) and list of errors.
    """
    records = []
    errors = []

    # If file is PDF, convert to images first
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
