# app.py
import streamlit as st
import json, io, re, numpy as np, pandas as pd
from PIL import Image
import cv2

st.set_page_config(page_title="Template OCR (Percent)", layout="wide")
st.title("Exam Form Scanner — Template-based (percent coords)")

# Load template.json (percent-based)
TEMPLATE_JSON = "template.json"
def load_template():
    try:
        with open(TEMPLATE_JSON, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

template = load_template()
if not template:
    st.error("template.json not found in repo. Upload template.json (I provided one) to the repo root or upload via file uploader below.")
else:
    st.success("Loaded template.json")

# OCR engine selection
USE_EASYOCR = False
reader = None
try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)
    USE_EASYOCR = True
except Exception:
    try:
        import pytesseract  # noqa
        USE_EASYOCR = False
    except Exception:
        st.error("No OCR engine available. Add easyocr or pytesseract to requirements.")
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

st.markdown("**Step 1 — (Optional) upload a blank template image** — best if it is a clear scan/phone-photo of the blank form. If not provided, the app will treat the first uploaded filled form as template reference.")
blank_template_file = st.file_uploader("Upload blank template (optional)", type=['jpg','jpeg','png'])

st.markdown("**Step 2 — Upload filled form images (multiple)**")
filled_files = st.file_uploader("Drag & drop filled forms (multiple)", type=['jpg','jpeg','png'], accept_multiple_files=True)

if blank_template_file:
    bbytes = blank_template_file.read()
    timg = Image.open(io.BytesIO(bbytes)).convert("RGB")
    tpl_width, tpl_height = timg.size
    st.image(timg, caption="Uploaded blank template (used for scaling)", use_column_width=False)
else:
    # We'll infer size from first filled image later
    tpl_width = tpl_height = None

if filled_files:
    # if no blank template, use first filled file to compute pixel sizes
    if not tpl_width:
        first = filled_files[0]
        fb = first.read()
        timg = Image.open(io.BytesIO(fb)).convert("RGB")
        tpl_width, tpl_height = timg.size
    # compute pixel boxes from percent template
    fields = template['fields']
    # convert percent coords -> pixel coords
    pixel_boxes = {}
    for fname, val in fields.items():
        x1 = int(val['x1_pct'] * tpl_width)
        y1 = int(val['y1_pct'] * tpl_height)
        x2 = int(val['x2_pct'] * tpl_width)
        y2 = int(val['y2_pct'] * tpl_height)
        # clamp
        x1 = max(0, min(x1, tpl_width-1)); x2 = max(0, min(x2, tpl_width-1))
        y1 = max(0, min(y1, tpl_height-1)); y2 = max(0, min(y2, tpl_height-1))
        pixel_boxes[fname] = (x1,y1,x2,y2)
    st.write("Computed pixel boxes (example):")
    st.json({k: pixel_boxes[k] for k in pixel_boxes})

    rows=[]
    progress = st.progress(0)
    total = len(filled_files)
    for i, f in enumerate(filled_files, start=1):
        raw = f.read()
        img_pil = Image.open(io.BytesIO(raw)).convert("RGB")
        img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        h,w = img_np.shape[:2]
        # If uploaded filled image size differs from template size, scale image to template size
        if (w, h) != (tpl_width, tpl_height):
            warped = cv2.resize(img_np, (tpl_width, tpl_height), interpolation=cv2.INTER_AREA)
        else:
            warped = img_np.copy()

        rec = {"source_file": f.name}
        for fname, (x1,y1,x2,y2) in pixel_boxes.items():
            crop = warped[y1:y2, x1:x2]
            if crop.size==0:
                text, conf = "", 0.0
            else:
                if USE_EASYOCR:
                    text, conf = run_easyocr(crop)
                else:
                    text, conf = run_pytesseract(crop)
            rec[fname] = text.strip()
            rec[fname + "_conf"] = float(conf)
        # post-process common fields
        if "mobile" in rec:
            rec["mobile_parsed"]=normalize_phone(rec.get("mobile",""))
        if "full_name" in rec and ("first_name" not in rec or not rec.get("first_name")):
            f1, m1, l1 = split_name_line(rec.get("full_name",""))
            rec["first_name_parsed"]=f1; rec["middle_name_parsed"]=m1; rec["last_name_parsed"]=l1
        rows.append(rec)
        progress.progress(i/total)
    progress.empty()

    df = pd.DataFrame(rows)
    # reorder columns for display
    display_cols = ["source_file","full_name","first_name","middle_name","last_name","mobile","mobile_parsed","class","school","medium"]
    display_cols = [c for c in display_cols if c in df.columns]
    st.markdown("### Extracted (by-crop) — edit inline below if needed")
    # display dataframe
    st.dataframe(df[display_cols], use_container_width=True)

    # Let user download
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv_bytes, "extracted_by_template.csv", "text/csv")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='extracted')
    buf.seek(0)
    st.download_button("Download Excel", buf.read(), "extracted_by_template.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.info("Upload filled forms to process. If results are a bit off, you can move boxes a little in template.json (change pct by 0.01) and re-run.")
