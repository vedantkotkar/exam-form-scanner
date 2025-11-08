# app.py
import streamlit as st
import pandas as pd
import os

from extract import extract_data

st.set_page_config(page_title="Exam Form Scanner", layout="centered")
st.title("🧾 Exam Form Scanner (Prototype)")
st.markdown("Upload clear photos (JPG/PNG) of the filled registration form. For best results: flat, daylight, no heavy shadows. This build crops the bottom region automatically.")

uploaded_files = st.file_uploader("Upload form images (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    all_results = []
    all_errors = []

    for file in uploaded_files:
        file_path = os.path.join("uploads", file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())

        st.info(f"Processing {file.name} ...")
        records, errors = extract_data(file_path)
        if records:
            all_results.extend(records)
        if errors:
            all_errors.extend(errors)

    if all_results:
        df = pd.DataFrame(all_results)
        st.success("✅ Extraction complete")
        st.dataframe(df)
        csv_path = os.path.join("outputs", "exam_forms_output.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        with open(csv_path, "rb") as f:
            st.download_button("⬇️ Download CSV", f, file_name="exam_forms_output.csv")
    else:
        st.warning("No records extracted. See errors below.")

    if all_errors:
        st.error("Some files had issues:")
        for e in all_errors:
            st.write("- " + str(e))
else:
    st.info("Upload clear JPG/PNG images of each filled form (one page per image).")
