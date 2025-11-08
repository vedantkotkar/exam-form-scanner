import streamlit as st
from PIL import Image
import io, os, json, tempfile, cv2, pandas as pd
from extract import extract_fields_from_image_bytes, load_and_resize, detect_page_corners, four_point_transform, DEFAULT_ZONES

# --- Streamlit Cloud credential helper ---
try:
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") and st.secrets.get("GCP_SERVICE_ACCOUNT_JSON", None):
        sa_json = st.secrets["GCP_SERVICE_ACCOUNT_JSON"]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp.write(sa_json.encode("utf-8"))
        tmp.flush()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
except Exception:
    pass
# -----------------------------------------

st.set_page_config(page_title='Exam Form Scanner', layout='wide')
st.title('🧾 Exam Form Scanner (PeethMart MVP)')

# Session zones
if "zones" not in st.session_state:
    st.session_state["zones"] = DEFAULT_ZONES

# Sidebar
mode = st.sidebar.radio("Choose mode", ["Single Image", "Batch", "Zone Editor"])

# =====================
# MODE: SINGLE IMAGE
# =====================
if mode == "Single Image":
    st.header("📄 Single Image Extraction")
    uploaded = st.file_uploader("Upload form image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img_bytes = uploaded.read()
        st.image(Image.open(io.BytesIO(img_bytes)), caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Extract Fields"):
            with st.spinner("Processing..."):
                try:
                    results = extract_fields_from_image_bytes(img_bytes, st.session_state["zones"])
                    st.success("✅ Extraction complete!")

                    left, right = st.columns([2, 1])
                    with left:
                        st.subheader("Extracted Fields")
                        editable = {}
                        for k, v in results.items():
                            if k.startswith("_"):  # Skip metadata
                                continue
                            editable[k] = st.text_input(k, value=str(v) if v else "")
                        st.write("**Average Confidence:**", round(results.get("_conf_avg", 0), 3))
                        if st.button("💾 Save to CSV"):
                            df = pd.DataFrame([editable])
                            csv_path = "extracted_single.csv"
                            df.to_csv(csv_path, index=False)
                            with open(csv_path, "rb") as f:
                                st.download_button("Download CSV", f, file_name=csv_path)

                    with right:
                        st.subheader("Preview Crops")
                        page = load_and_resize(img_bytes)
                        pts = detect_page_corners(page)
                        aligned = four_point_transform(page, pts) if pts is not None else page
                        for name, zone in st.session_state["zones"].items():
                            x1, y1, x2, y2 = zone
                            h, w = aligned.shape[:2]
                            crop = aligned[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
                            st.image(crop, caption=name, width=300)

                except Exception as e:
                    st.error(f"❌ Error: {e}")

# =====================
# MODE: BATCH
# =====================
elif mode == "Batch":
    st.header("🗃️ Batch Processing")
    uploaded_files = st.file_uploader("Upload multiple images or a ZIP", type=["jpg", "jpeg", "png", "zip"], accept_multiple_files=True)
    output_rows = []

    if uploaded_files and st.button("🚀 Start Batch Extraction"):
        with st.spinner("Processing batch..."):
            import zipfile
            tempdir = tempfile.mkdtemp()
            all_images = []

            for f in uploaded_files:
                if f.name.endswith(".zip"):
                    with zipfile.ZipFile(f, "r") as zf:
                        zf.extractall(tempdir)
                        for root, _, files in os.walk(tempdir):
                            for file in files:
                                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                                    all_images.append(os.path.join(root, file))
                else:
                    tmp_path = os.path.join(tempdir, f.name)
                    with open(tmp_path, "wb") as tempf:
                        tempf.write(f.read())
                    all_images.append(tmp_path)

            for path in all_images:
                try:
                    with open(path, "rb") as f:
                        b = f.read()
                    res = extract_fields_from_image_bytes(b, st.session_state["zones"])
                    res["_source"] = os.path.basename(path)
                    output_rows.append(res)
                    st.write(f"✅ Processed {os.path.basename(path)}")
                except Exception as e:
                    st.warning(f"❌ Error on {path}: {e}")

            if output_rows:
                df = pd.DataFrame(output_rows)
                st.dataframe(df)
                csv_path = "batch_output.csv"
                df.to_csv(csv_path, index=False)
                with open(csv_path, "rb") as f:
                    st.download_button("📥 Download Batch CSV", f, file_name=csv_path)
            else:
                st.warning("No images processed.")

# =====================
# MODE: ZONE EDITOR
# =====================
elif mode == "Zone Editor":
    st.header("🎯 Zone Editor")
    sample = st.file_uploader("Upload a sample image", type=["jpg", "jpeg", "png"])

    if sample:
        img_bytes = sample.read()
        page = load_and_resize(img_bytes)
        pts = detect_page_corners(page)
        aligned = four_point_transform(page, pts) if pts is not None else page
        st.image(Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)), caption="Aligned Page")

        new_zones = {}
        for key, val in st.session_state["zones"].items():
            st.subheader(key)
            x1 = st.slider(f"{key} x1", 0.0, 1.0, val[0], 0.001)
            y1 = st.slider(f"{key} y1", 0.0, 1.0, val[1], 0.001)
            x2 = st.slider(f"{key} x2", 0.0, 1.0, val[2], 0.001)
            y2 = st.slider(f"{key} y2", 0.0, 1.0, val[3], 0.001)
            new_zones[key] = [x1, y1, x2, y2]

        if st.button("👁️ Preview Zones"):
            viz = aligned.copy()
            h, w = viz.shape[:2]
            for k, z in new_zones.items():
                x1, y1, x2, y2 = [int(z[0]*w), int(z[1]*h), int(z[2]*w), int(z[3]*h)]
                cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(viz, k, (x1+5, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            st.image(viz, caption="Zones Preview")

        if st.button("💾 Save Zones"):
            st.session_state["zones"] = new_zones
            with open("zones.json", "w") as f:
                json.dump(new_zones, f, indent=2)
            st.success("Zones saved to zones.json!")
            with open("zones.json", "rb") as f:
                st.download_button("Download zones.json", f, file_name="zones.json")

# =====================
# FOOTER
# =====================
st.markdown("---")
st.caption("PeethMart Exam Form Scanner — Built with ❤️ using Streamlit + Google Vision OCR.")
