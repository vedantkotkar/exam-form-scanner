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
        st.info("EasyOCR not available, falling back to pytesseract (requires Tesseract binary if run locally).")
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
    text = pytesseract.image_to_string(gray, lang='eng')
    # We return 0.0 for confidence because computing it needs image_to_data parsing; acceptable fallback.
    return text, 0.0

# ----------------- helpers -----------------
def normalize_phone(text):
    if not text: return ""
    digits = re.sub(r'\D', '', text)
    if len(digits) == 10: return digits
    if len(digits) == 12 and digits.startswith("91"): return digits[2:]
    return digits

def split_name_line(line):
    if not line: return "","",""
    parts = [p for p in re.split(r'\s+', line.strip()) if p]
    if len(parts)==1: return parts[0],"",""
    if len(parts)==2: return parts[0],"",parts[1]
    return parts[0], " ".join(parts[1:-1]), parts[-1]

def pct_to_pixels_field(pct_field, width, height):
    x1 = int(pct_field['x1_pct'] * width)
    y1 = int(pct_field['y1_pct'] * height)
    x2 = int(pct_field['x2_pct'] * width)
    y2 = int(pct_field['y2_pct'] * height)
    x1 = max(0, min(x1, width-1)); x2 = max(0, min(x2, width-1))
    y1 = max(0, min(y1, height-1)); y2 = max(0, min(y2, height-1))
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
# Priority:
# 1) template_pixels.json (pixel coords)
# 2) template.json (percent coords) + blank template image (or first filled image) -> conversion
pixel_template = load_json_if_exists("template_pixels.json")
percent_template = load_json_if_exists("template.json")

if pixel_template:
    st.success("Found template_pixels.json — using pixel boxes.")
elif percent_template:
    st.info("Found percent-based template.json — will convert to pixels using uploaded blank or first image.")
else:
    st.warning("No template_pixels.json or template.json found in repo. Place one in repo root or upload percent template below.")

# UI: allow user to upload pixel-template or percent-template / blank template as a fallback
st.markdown("### Optional: upload template files here")
colA, colB = st.columns(2)
with colA:
    upload_pixel = st.file_uploader("Upload template_pixels.json (optional)", type=['json'], key="up_pix")
with colB:
    upload_percent = st.file_uploader("Upload template.json (percent coords) (optional)", type=['json'], key="up_pct")

if upload_pixel:
    try:
        tplp = json.load(upload_pixel)
        with open("template_pixels.json","w") as f:
            json.dump(tplp, f, indent=2)
        pixel_template = tplp
        st.success("Saved template_pixels.json to repo root (workspace).")
    except Exception as e:
        st.error(f"Failed to read uploaded pixel template: {e}")

if upload_percent:
    try:
        tplpct = json.load(upload_percent)
        with open("template.json","w") as f:
            json.dump(tplpct, f, indent=2)
        percent_template = tplpct
        st.success("Saved template.json (percent) to repo root.")
    except Exception as e:
        st.error(f"Failed to read uploaded percent template: {e}")

# Allow blank template image upload for percent->pixel conversion
st.markdown("---")
st.markdown("### Template image (optional, recommended for percent->pixel conversion)")
blank_template_file = st.file_uploader("Upload blank (clean) template image (JPG/PNG) — optional but recommended", type=['jpg','jpeg','png'], key="blank")

# ----------------- File upload (filled images) -----------------
st.markdown("---")
st.markdown("### Upload filled form images (multiple)")
uploaded = st.file_uploader("Drag & drop filled forms (multiple)", type=['jpg','jpeg','png'], accept_multiple_files=True, key="forms")

# ----------------- Process uploads -----------------
if uploaded:
    # Read uploaded filled images into memory once
    filled_bytes = []
    for f in uploaded:
        try:
            b = f.read()
            if not b:
                st.warning(f"Uploaded file {f.name} is empty")
            filled_bytes.append({"name": f.name, "bytes": b})
        except Exception as e:
            st.warning(f"Failed reading {f.name}: {e}")
            filled_bytes.append({"name": f.name, "bytes": b""})

    # prepare template boxes (pixel)
    pixel_boxes = None
    tpl_w = tpl_h = None

    # 1) If pixel template exists in workspace, use it
    if os.path.exists("template_pixels.json"):
        tplp = load_json_if_exists("template_pixels.json")
        pixel_boxes = tplp["fields"]
        tpl_w = tplp.get("width")
        tpl_h = tplp.get("height")
        st.info("Using template_pixels.json from workspace.")
    # 2) else if percent template exists -> convert using blank_template_file or first uploaded image
    elif os.path.exists("template.json"):
        tplpct = load_json_if_exists("template.json")
        # determine width/height using blank template if provided
        if blank_template_file:
            try:
                blank_bytes = blank_template_file.read()
                img_tpl = safe_open_image_bytes(blank_bytes)
                if img_tpl is None:
                    st.error("Uploaded blank template couldn't be read as image.")
                    st.stop()
                tpl_w, tpl_h = img_tpl.size
                st.info(f"Using uploaded blank template size {tpl_w}x{tpl_h}")
            except Exception as e:
                st.error(f"Failed processing blank template: {e}")
                st.stop()
        else:
            # infer from first valid filled image bytes
            first_valid = None
            for it in filled_bytes:
                if it["bytes"]:
                    first_valid = it["bytes"]; break
            if not first_valid:
                st.error("No valid filled images to infer template size. Upload a blank template or a valid filled image.")
                st.stop()
            img_tpl = safe_open_image_bytes(first_valid)
            if img_tpl is None:
                st.error("Could not read first filled image to infer size. Upload a blank template image.")
                st.stop()
            tpl_w, tpl_h = img_tpl.size
            st.info(f"Inferred template size from first uploaded image: {tpl_w}x{tpl_h}")

        # convert percent coords -> pixel coords
        pixel_boxes = {}
        for fname, coords in tplpct["fields"].items():
            px = pct_to_pixels_field(coords, tpl_w, tpl_h)
            pixel_boxes[fname] = px
        # save pixel template for future runs
        out = {"description": tplpct.get("description","pixel template"), "width": tpl_w, "height": tpl_h, "fields": pixel_boxes}
        with open("template_pixels.json","w") as f:
            json.dump(out, f, indent=2)
        st.success("Converted percent template -> template_pixels.json and saved it.")
    else:
        st.error("No template.json or template_pixels.json available. Upload one to repo or via the upload widgets above.")
        st.stop()

    # Now we have pixel_boxes and tpl_w, tpl_h
    if not pixel_boxes or not tpl_w or not tpl_h:
        st.error("Template boxes or template size missing. Cannot proceed.")
        st.stop()

    # show boxes for verification (small)
    st.markdown("Computed pixel boxes (fields):")
    st.json(pixel_boxes)

    # Process each file: open from bytes, resize to template size if needed, crop fields, OCR
    rows = []
    # show crop preview area for the first file
    show_preview = st.checkbox("Show crop previews for first processed image", value=True)

    total = len(filled_bytes)
    pb = st.progress(0)
    for idx, itm in enumerate(filled_bytes, start=1):
        name = itm["name"]
        b = itm["bytes"]
        if not b:
            st.warning(f"Skipping empty file {name}")
            pb.progress(idx/total); continue
        img_pil = safe_open_image_bytes(b)
        if img_pil is None:
            st.warning(f"Skipping invalid image {name}")
            pb.progress(idx/total); continue

        img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        h,w = img_np.shape[:2]
        # scale to template size if different
        if (w,h) != (tpl_w, tpl_h):
            warped = cv2.resize(img_np, (tpl_w, tpl_h), interpolation=cv2.INTER_AREA)
        else:
            warped = img_np.copy()

        rec = {"source_file": name}
        # for preview only, store first image crops in a dict
        preview_crops = {}
        for fname, box in pixel_boxes.items():
            x1,y1,x2,y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            # guard
            x1 = max(0, min(x1, tpl_w-1)); x2 = max(0, min(x2, tpl_w-1))
            y1 = max(0, min(y1, tpl_h-1)); y2 = max(0, min(y2, tpl_h-1))
            if x2<=x1 or y2<=y1:
                crop_text, conf = "", 0.0
            else:
                crop = warped[y1:y2, x1:x2]
                if USE_EASYOCR:
                    crop_text, conf = run_easyocr(crop)
                else:
                    crop_text, conf = run_pytesseract(crop)
                crop_text = crop_text.strip()
            rec[fname] = crop_text
            rec[fname + "_conf"] = float(conf)
            if idx==1 and show_preview:
                try:
                    # convert crop to PNG bytes for display later
                    if x2>x1 and y2>y1:
                        cimg = crop
                        _, png = cv2.imencode('.png', cimg)
                        preview_crops[fname] = png.tobytes()
                except Exception:
                    preview_crops[fname] = None

        # postprocessing
        if "mobile" in rec:
            rec["mobile_parsed"] = normalize_phone(rec.get("mobile",""))
        if "full_name" in rec and (not rec.get("first_name")):
            f1,m1,l1 = split_name_line(rec.get("full_name",""))
            rec["first_name_parsed"]=f1; rec["middle_name_parsed"]=m1; rec["last_name_parsed"]=l1

        rows.append(rec)
        pb.progress(idx/total)

    pb.empty()

    if not rows:
        st.error("No valid rows processed.")
        st.stop()

    df = pd.DataFrame(rows)
    # Display crop previews for the first image if requested
    if show_preview and 'preview_crops' in locals() and preview_crops:
        st.markdown("#### Crop previews for first file (verify boxes)")
        cols = st.columns(min(4, len(preview_crops)))
        i = 0
        for k, imgbytes in preview_crops.items():
            with cols[i % len(cols)]:
                st.markdown(f"**{k}**")
                if imgbytes:
                    st.image(Image.open(io.BytesIO(imgbytes)), use_column_width=True)
                else:
                    st.write("no preview")
            i += 1

    # Choose display columns
    display_cols = ["source_file"]
    # attempt to show name split fields if present; otherwise the crop full_name
    if "full_name" in df.columns:
        display_cols += ["full_name"]
    if "first_name" in df.columns:
        display_cols += ["first_name","middle_name","last_name"]
    if "mobile" in df.columns:
        display_cols += ["mobile","mobile_parsed"]
    if "class" in df.columns:
        display_cols += ["class"]
    if "school" in df.columns:
        display_cols += ["school"]
    if "medium" in df.columns:
        display_cols += ["medium"]
    # always include confidences if present
    conf_cols = [c for c in df.columns if c.endswith("_conf")]
    display_cols += conf_cols

    # Show results and allow editing
    st.markdown("### Extracted results — edit then download")
    # Prefer experimental_data_editor if available
    if hasattr(st, "experimental_data_editor"):
        try:
            edited = st.experimental_data_editor(df[display_cols], num_rows="dynamic")
        except Exception:
            st.warning("Inline editor not available in this runtime — showing read-only table with row editor below.")
            edited = df[display_cols]
    else:
        st.warning("Inline editor not available in this Streamlit runtime — showing read-only table with row editor below.")
        edited = df[display_cols]

    # Provide downloads: use the edited table if available, else df
    if isinstance(edited, pd.DataFrame):
        out_df = edited.copy()
    else:
        out_df = df.copy()

    csv_bytes = out_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv_bytes, "extracted_forms.csv", "text/csv")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        out_df.to_excel(writer, index=False, sheet_name="extracted")
    buf.
