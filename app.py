# app.py
import streamlit as st
import io, re, numpy as np, pandas as pd
from PIL import Image
import cv2

st.set_page_config(page_title="Exam Form Scanner (Batch)", layout="wide")
st.title("Exam Form Scanner — Batch OCR Demo")
st.write("Drag & drop multiple filled form images. App extracts fields and shows them in a table. Edit fields inline and download CSV/Excel.")

# Try EasyOCR first, fallback to pytesseract
USE_EASYOCR = False
reader = None
try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)
    USE_EASYOCR = True
except Exception:
    try:
        import pytesseract
        USE_EASYOCR = False
    except Exception:
        st.error("No OCR engine available. Please include EasyOCR or pytesseract in requirements.")
        st.stop()

def run_easyocr(img_np):
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    res = reader.readtext(img_rgb, detail=1)
    texts = [r[1] for r in res]
    confs = [r[2] for r in res]
    return "\n".join(texts), (float(np.mean(confs)) if confs else 0.0)

def run_pytesseract(img_np):
    import pytesseract
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    try:
        gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    except Exception:
        pass
    txt = pytesseract.image_to_string(gray, lang='eng')
    # approximate confidence omitted (pytesseract needs image_to_data)
    return txt, 0.0

def normalize_phone(text):
    if not text: return ""
    digits = re.sub(r'\D','', text)
    if len(digits)==10: return digits
    if len(digits)==12 and digits.startswith("91"): return digits[2:]
    return digits

def split_name_line(line):
    if not line: return "","",""
    parts = [p for p in re.split(r'\s+', line.strip()) if p]
    if len(parts)==1: return parts[0],"",""
    if len(parts)==2: return parts[0],"",parts[1]
    return parts[0], " ".join(parts[1:-1]), parts[-1]

def heuristic_extract(full_text):
    txt = full_text.replace('\r','\n')
    txt_u = txt.upper()
    out = {"Full Name":"", "First Name":"", "Middle Name":"", "Last Name":"", "Mobile":"", "School":"", "Class":"", "Medium":"", "OCR_CONF":0.0"}
    # mobile
    m = re.search(r'\b[6-9]\d{9}\b', txt_u)
    if m: out["Mobile"] = normalize_phone(m.group(0))
    # medium
    med = re.search(r'(ENGLISH|MARATHI|HINDI)\s*MEDIUM', txt_u)
    if med: out["Medium"] = med.group(0).title()
    # class
    cls = re.search(r'\b(?:CLASS|STD|STANDARD)\b[^A-Z0-9]{0,6}([A-Z0-9]{1,5})', txt_u)
    if cls: out["Class"] = cls.group(1).title()
    # school: look for lines containing SCHOOL or common tokens
    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    sc
