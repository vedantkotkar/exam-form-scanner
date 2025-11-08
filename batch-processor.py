"""
Batch Processor — Exam Form Scanner
-----------------------------------
This script automatically processes all images in `data/incoming/` using the same OCR logic as the Streamlit app,
then saves extracted results to `data/output/batch_output.csv`.

Usage (local or Windows):
    python batch_processor.py
"""

import os
import cv2
import pandas as pd
from app import extract_fields_from_image_bytes, load_and_resize, detect_page_corners, four_point_transform, FIELD_ZONES
import json

# Ensure folders exist
os.makedirs("data/incoming", exist_ok=True)
os.makedirs("data/output", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Load zones.json if available
if os.path.exists("zones.json"):
    with open("zones.json", "r") as f:
        FIELD_ZONES = json.load(f)

def process_all_images():
    input_dir = "data/incoming"
    output_csv = "data/output/batch_output.csv"
    rows = []
    files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"Found {len(files)} files to process...")

    for fname in files:
        fpath = os.path.join(input_dir, fname)
        try:
            with open(fpath, "rb") as f:
                image_bytes = f.read()
            res = extract_fields_from_image_bytes(image_bytes, FIELD_ZONES)
            res["_source"] = fname
            rows.append(res)

            # Move processed file to /processed/
            os.rename(fpath, os.path.join("data/processed", fname))

            print(f"✅ Processed {fname}")
        except Exception as e:
            print(f"❌ Error with {fname}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        print(f"\nAll done! Saved {len(rows)} rows → {output_csv}")
    else:
        print("No files processed or extracted.")

if __name__ == "__main__":
    process_all_images()
