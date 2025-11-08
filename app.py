# app.py
import streamlit as st
import pandas as pd
import os
from extract import extract_data

st.set_page_config(page_title="Exam Form Scanner", layout="centered")
st.title("🧾 Exam Form Scanner (Prototype)")
st.markdown("Upload scanned registration forms (JPG/PNG/PDF). The app will extract: First, Middle, Surname, Class, Mobile, School, Medium.")

uploaded_files = st.file_uploader("Upload files", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)

if uploaded_files:
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    results = []
    for file in uploaded_files:
        file_path = os.path.join("uploads", file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        st.info(f"Processing {file.name} ...")
        data = extract_data(file_path)
        results.append(data)

    df = pd.DataFrame(results)
    st.success("✅ Extraction complete")
    st.dataframe(df)

    csv_path = os.path.join("outputs", "exam_forms_output.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(csv_path, "rb") as f:
        st.download_button("⬇️ Download CSV", f, file_name="exam_forms_output.csv")
else:
    st.info("Upload 1-20 scanned form images to see extraction results.")
