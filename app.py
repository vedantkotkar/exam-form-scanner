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
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    except Exception:
        pass
    txt = pytesseract.image_to_string(gray, lang='eng')
    return txt, 0.0

def normalize_phone(text):
    if not text:
        return ""
    digits = re.sub(r'\D', '', text)
    if len(digits) == 10:
        return digits
    if len(digits) == 12 and digits.startswith("91"):
        return digits[2:]
    return digits

def split_name_line(line):
    if not line:
        return "", "", ""
    parts = [p for p in re.split(r'\s+', line.strip()) if p]
    if len(parts) == 1:
        return parts[0], "", ""
    if len(parts) == 2:
        return parts[0], "", parts[1]
    return parts[0], " ".join(parts[1:-1]), parts[-1]

def heuristic_extract(full_text):
    txt = full_text.replace('\r', '\n')
    txt_u = txt.upper()
    out = {
        "Full Name": "",
        "First Name": "",
        "Middle Name": "",
        "Last Name": "",
        "Mobile": "",
        "School": "",
        "Class": "",
        "Medium": "",
        "OCR_CONF": 0.0
    }

    # mobile
    m = re.search(r'\b[6-9]\d{9}\b', txt_u)
    if m:
        out["Mobile"] = normalize_phone(m.group(0))

    # medium
    med = re.search(r'(ENGLISH|MARATHI|HINDI)\s*MEDIUM', txt_u)
    if med:
        out["Medium"] = med.group(0).title()

    # class
    cls = re.search(r'\b(?:CLASS|STD|STANDARD)\b[^A-Z0-9]{0,6}([A-Z0-9]{1,5})', txt_u)
    if cls:
        out["Class"] = cls.group(1).title()

    # school
    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    school_line = None
    for ln in lines:
        if any(k in ln.upper() for k in ["SCHOOL", "HIGH SCHOOL", "MOUNT", "MOUNT ST", "MOUNT STANN", "MOUNT ST. ANN", "SAINT"]):
            school_line = ln
            break
    if school_line:
        out["School"] = school_line.title()

    # name
    candidate = None
    for ln in lines[:6]:
        if ln and not re.search(r'\d', ln) and 2 <= len(ln.split()) <= 6:
            candidate = ln
            break
    if not candidate and lines:
        candidate = lines[0]

    out["Full Name"] = candidate or ""
    f, mn, l = split_name_line(candidate or "")
    out["First Name"], out["Middle Name"], out["Last Name"] = f, mn, l

    return out

uploaded_files = st.file_uploader(
    "Drag & drop images (multiple) — JPG/PNG", 
    type=['jpg', 'jpeg', 'png'], 
    accept_multiple_files=True
)

if uploaded_files:
    rows = []
    progress_bar = st.progress(0)
    total = len(uploaded_files)

    for idx, up in enumerate(uploaded_files, start=1):
        raw = up.read()
        img_pil = Image.open(io.BytesIO(raw)).convert("RGB")
        img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        if USE_EASYOCR:
            full_text, conf = run_easyocr(img_np)
        else:
            full_text, conf = run_pytesseract(img_np)

        extracted = heuristic_extract(full_text)
        extracted["OCR_CONF"] = conf
        extracted["source_file"] = up.name
        extracted["raw_text_preview"] = (full_text[:600] + "...") if len(full_text) > 600 else full_text
        rows.append(extracted)

        progress_bar.progress(idx / total)

    progress_bar.empty()
    df = pd.DataFrame(rows)
    cols = ["source_file", "Full Name", "First Name", "Middle Name", "Last Name",
            "Mobile", "School", "Class", "Medium", "OCR_CONF", "raw_text_preview"]
    df = df[cols]

    st.markdown("### Extracted results")
    st.write("You can edit values in the table below before downloading.")
    edited = st.experimental_data_editor(df, num_rows="dynamic")

    csv_bytes = edited.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv_bytes, "extracted_forms.csv", "text/csv")

    towrite = io.BytesIO()
    with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
        edited.to_excel(writer, index=False, sheet_name="extracted")
    towrite.seek(0)
    st.download_button(
        "Download Excel",
        towrite.read(),
        "extracted_forms.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.success(f"Processed {len(edited)} images. Inspect/edit table and download.")
else:
    st.info("Drop multiple form images above to process. Use clear scans for best results.")
