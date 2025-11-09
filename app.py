# app.py
import streamlit as st
import io, os, json, re
import numpy as np, pandas as pd
from PIL import Image, UnidentifiedImageError
import cv2

st.set_page_config(page_title="Exam Form Scanner — Final", layout="wide")
st.title("Exam Form Scanner — Template-based OCR (final)")

# ----------------- OCR engine init -----------------
USE_EASYOCR = False
reader = None
try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)
    USE_EASYOCR = True
    st.info("Using EasyOCR engine.")
except Exception:
    try:
        import pytesseract  # noqa: F401
        USE_EASYOCR = False
        st.info("EasyOCR not available, using pytesseract instead.")
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
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    except Exception:
        pass
    text = pytesseract.image_to_string(gray, lang='eng')
    return text, 0.0


# ----------------- helpers -----------------
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


def pct_to_pixels_field(pct_field, width, height):
    x1 = int(pct_field['x1_pct'] * width)
    y1 = int(pct_field['y1_pct'] * height)
    x2 = int(pct_field['x2_pct'] * width)
    y2 = int(pct_field['y2_pct'] * height)
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def load_json_if_exists(name):
    if os.path.exists(name):
        with open(name, 'r') as f:
            return json.load(f)
    return None


def safe_open_image_bytes(b):
    try:
        return Image.open(io.BytesIO(b)).convert("RGB")
    except UnidentifiedImageError:
        return None


# ----------------- Template loading logic -----------------
pixel_template = load_json_if_exists("template_pixels.json")
percent_template = load_json_if_exists("template.json")

if pixel_template:
    st.success("Found template_pixels.json — using pixel boxes.")
elif percent_template:
    st.info("Found percent-based template.json — will convert to pixels using uploaded blank or first image.")
else:
    st.warning("No template found. Upload template.json (percent) or template_pixels.json (pixel).")

# UI for optional template uploads
st.markdown("### Optional: upload template files")
colA, colB = st.columns(2)
with colA:
    upload_pixel = st.file_uploader("Upload template_pixels.json", type=['json'], key="up_pix")
with colB:
    upload_percent = st.file_uploader("Upload template.json", type=['json'], key="up_pct")

if upload_pixel:
    tplp = json.load(upload_pixel)
    with open("template_pixels.json", "w") as f:
        json.dump(tplp, f, indent=2)
    pixel_template = tplp
    st.success("template_pixels.json uploaded successfully.")

if upload_percent:
    tplpct = json.load(upload_percent)
    with open("template.json", "w") as f:
        json.dump(tplpct, f, indent=2)
    percent_template = tplpct
    st.success("template.json uploaded successfully.")

blank_template_file = st.file_uploader("Upload blank (clean) template image (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

st.markdown("---")
st.markdown("### Upload filled form images (multiple)")
uploaded = st.file_uploader("Drag & drop filled forms", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

# ----------------- Process uploads -----------------
if uploaded:
    filled_bytes = [{"name": f.name, "bytes": f.read()} for f in uploaded]
    pixel_boxes = None
    tpl_w = tpl_h = None

    # 1️⃣ Use pixel template if available
    if os.path.exists("template_pixels.json"):
        tplp = load_json_if_exists("template_pixels.json")
        pixel_boxes = tplp["fields"]
        tpl_w = tplp.get("width")
        tpl_h = tplp.get("height")
        st.info("Using template_pixels.json.")

    # 2️⃣ Else convert percent template using blank or first image
    elif os.path.exists("template.json"):
        tplpct = load_json_if_exists("template.json")
        if blank_template_file:
            blank_bytes = blank_template_file.read()
            img_tpl = safe_open_image_bytes(blank_bytes)
            if not img_tpl:
                st.error("Invalid blank template image.")
                st.stop()
            tpl_w, tpl_h = img_tpl.size
        else:
            first_valid = next((it["bytes"] for it in filled_bytes if it["bytes"]), None)
            if not first_valid:
                st.error("No valid filled images to infer template size.")
                st.stop()
            img_tpl = safe_open_image_bytes(first_valid)
            if not img_tpl:
                st.error("Could not read first image to infer size.")
                st.stop()
            tpl_w, tpl_h = img_tpl.size
        pixel_boxes = {fn: pct_to_pixels_field(fc, tpl_w, tpl_h) for fn, fc in tplpct["fields"].items()}
        with open("template_pixels.json", "w") as f:
            json.dump({"width": tpl_w, "height": tpl_h, "fields": pixel_boxes}, f, indent=2)
        st.success("Converted percent template to pixel boxes.")

    else:
        st.error("No template found.")
        st.stop()

    st.json(pixel_boxes)
    rows = []
    total = len(filled_bytes)
    pb = st.progress(0)
    show_preview = st.checkbox("Show crop previews for first image", value=True)

    for idx, itm in enumerate(filled_bytes, start=1):
        name, b = itm["name"], itm["bytes"]
        if not b:
            pb.progress(idx / total)
            continue
        img_pil = safe_open_image_bytes(b)
        if not img_pil:
            pb.progress(idx / total)
            continue

        img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        h, w = img_np.shape[:2]
        if (w, h) != (tpl_w, tpl_h):
            warped = cv2.resize(img_np, (tpl_w, tpl_h), interpolation=cv2.INTER_AREA)
        else:
            warped = img_np

        rec = {"source_file": name}
        preview_crops = {}

        for fname, box in pixel_boxes.items():
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            crop = warped[y1:y2, x1:x2]
            if crop.size == 0:
                text, conf = "", 0.0
            else:
                text, conf = run_easyocr(crop) if USE_EASYOCR else run_pytesseract(crop)
            rec[fname] = text.strip()
            rec[fname + "_conf"] = float(conf)
            if idx == 1 and show_preview and crop.size > 0:
                _, buf_png = cv2.imencode(".png", crop)
                preview_crops[fname] = buf_png.tobytes()

        if "mobile" in rec:
            rec["mobile_parsed"] = normalize_phone(rec["mobile"])
        if "full_name" in rec and not rec.get("first_name"):
            f1, m1, l1 = split_name_line(rec["full_name"])
            rec["first_name_parsed"], rec["middle_name_parsed"], rec["last_name_parsed"] = f1, m1, l1

        rows.append(rec)
        pb.progress(idx / total)

    pb.empty()

    if not rows:
        st.error("No valid rows processed.")
        st.stop()

    df = pd.DataFrame(rows)

    # Optional: show crop previews for first image
    if show_preview and 'preview_crops' in locals() and preview_crops:
        st.markdown("#### Crop previews for first file")
        cols = st.columns(min(4, len(preview_crops)))
        i = 0
        for k, imgbytes in preview_crops.items():
            with cols[i % len(cols)]:
                st.markdown(f"**{k}**")
                st.image(Image.open(io.BytesIO(imgbytes)), use_column_width=True)
            i += 1

    # Display & download
    display_cols = [c for c in df.columns if not c.endswith("_conf")]
    conf_cols = [c for c in df.columns if c.endswith("_conf")]
    display_cols += conf_cols

    st.markdown("### Extracted results — edit then download")
    if hasattr(st, "experimental_data_editor"):
        try:
            edited = st.experimental_data_editor(df[display_cols], num_rows="dynamic")
        except Exception:
            edited = df[display_cols]
    else:
        edited = df[display_cols]

    out_df = edited.copy() if isinstance(edited, pd.DataFrame) else df.copy()

    csv_bytes = out_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv_bytes, "extracted_forms.csv", "text/csv")

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        out_df.to_excel(writer, index=False, sheet_name="extracted")
    buf.seek(0)
    st.download_button(
        "Download Excel",
        buf.read(),
        "extracted_forms.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.success(f"Processed {len(out_df)} files. Download available above.")
else:
    st.info("Upload filled form images above to begin processing.")
