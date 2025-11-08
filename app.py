import streamlit as st
import pandas as pd
import tempfile
import os
from extract import extract_data, process_files

# ---------------------------------
# App Configuration
# ---------------------------------
st.set_page_config(
    page_title="Exam Form Scanner",
    page_icon="🧾",
    layout="centered",
)

st.title("🧾 Exam Form Scanner (Prototype)")
st.write(
    "Upload clear photos (JPG/PNG) of the filled registration form. "
    "For best results: flat, daylight, no heavy shadows. "
    "This build crops the bottom region automatically."
)

# Debug: confirm key loaded
import json
try:
    creds = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_JSON"])
    st.success(f"✅ Google key loaded. Project: {creds['project_id']}")
except Exception as e:
    st.error("❌ Google Vision not configured correctly.")
    st.code(str(e))

# ---------------------------------
# File Upload
# ---------------------------------
uploaded_files = st.file_uploader(
    "Upload form images (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.info(f"Processing {len(uploaded_files)} file(s)... Please wait ⏳")

    with st.spinner("Extracting data from images..."):
        # Use temporary files for processing
        temp_paths = []
        for uploaded_file in uploaded_files:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.read())
            temp_paths.append(temp_file.name)
            temp_file.close()

        df, errors = process_files(temp_paths)

        if not df.empty:
            st.success("✅ Extraction complete!")
            st.dataframe(df)

            # Option to download results
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="exam_forms_data.csv",
                mime="text/csv",
            )
        else:
            st.warning("⚠️ No records extracted. Check logs or try clearer images.")

        # Show errors if any
        if errors:
            st.error("Some files had issues:")
            for e in errors:
                st.write(f"- {e.get('file', '?')}: {e.get('error', 'Unknown error')}")

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("---")
st.caption(
    "Built by **Vedant Kotkar** | Powered by Google Vision AI | Prototype v1.0"
)
