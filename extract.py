import io
import os
import re
import cv2
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account


# --------------------------
# 1. Google Vision Setup
# --------------------------
def get_vision_client():
    """Safely load Google Vision client using Streamlit secrets."""
    try:
        service_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_JSON"])
        credentials = service_account.Credentials.from_service_account_info(service_info)
        client = vision.ImageAnnotatorClient(credentials=credentials)
        return client
    except Exception as e:
        st.error(f"❌ Vision client setup failed: {e}")
        return None


# --------------------------
# 2. Image Preprocessing
# --------------------------
def preprocess_for_ocr(pil_image):
    """Convert image to grayscale, crop bottom, and apply threshold."""
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # crop bottom 35% of image (where form data is)
    height = gray.shape[0]
    crop_start = int(height * 0.60)
    cropped = gray[crop_start:, :]

    # apply threshold for clarity
    _, thresh = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


# --------------------------
# 3. OCR Extraction
# --------------------------
def extract_text_google(img):
    """Use Google Vision to extract text."""
    client = get_vision_client()
    if client is None:
        raise RuntimeError("❌ Google Vision not configured")

    # Encode as bytes for Vision
    success, encoded_img = cv2.imencode(".jpg", img)
    if not success:
        raise ValueError("Could not encode image")

    content = encoded_img.tobytes()
    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    if response.error.message:
        raise Exception(response.error.message)

    texts = response.text_annotations
    if not texts:
        return ""

    return texts[0].description


# --------------------------
# 4. Parse Extracted Text
# --------------------------
def parse_exam_form_text(text):
    """Extract key fields from OCR text."""
    result = {
        "First Name": "",
        "Middle Name": "",
        "Surname": "",
        "Class": "",
        "Mobile": "",
        "School Name": "",
        "Medium": ""
    }

    # Clean text
    clean_text = text.replace("\n", " ").upper()

    # Detect medium
    if "ENGLISH" in clean_text:
        result["Medium"] = "English"
    elif "VERNACULAR" in clean_text or "MARATHI" in clean_text:
        result["Medium"] = "Marathi"

    # Extract name fields
    # Example: "Name of the Student: SWARUPA VIJAY KHOT"
    name_match = re.search(r"NAME\s*OF\s*THE\s*STUDENT[:\s]+([A-Z\s]+)", clean_text)
    if name_match:
        full_name = name_match.group(1).strip()
        parts = full_name.split()
        if len(parts) == 3:
            result["First Name"], result["Middle Name"], result["Surname"] = parts
        elif len(parts) == 2:
            result["First Name"], result["Surname"] = parts
        elif len(parts) == 1:
            result["First Name"] = parts[0]

    # Extract class
    class_match = re.search(r"CLASS[:\s]*([A-Z0-9 ]+)", clean_text)
    if class_match:
        result["Class"] = class_match.group(1).strip()

    # Extract school name
    school_match = re.search(r"NAME\s*OF\s*SCHOOL[:\s]*([A-Z0-9 ]+)", clean_text)
    if school_match:
        result["School Name"] = school_match.group(1).strip()

    # Extract mobile
    mob_match = re.search(r"(\b\d{10}\b)", clean_text)
    if mob_match:
        result["Mobile"] = mob_match.group(1).strip()

    return result


# --------------------------
# 5. Main Extraction Pipeline
# --------------------------
def extract_data(file_path):
    """Main processing pipeline for a single file."""
    errors = None
    try:
        pil_img = Image.open(file_path).convert("RGB")
        processed = preprocess_for_ocr(pil_img)
        raw_text = extract_text_google(processed)
        record = parse_exam_form_text(raw_text)
        record["RawTextSample"] = raw_text[:150] + "..." if len(raw_text) > 150 else raw_text
        return [record], None
    except Exception as e:
        errors = str(e)
        return [], errors


# --------------------------
# 6. Batch Handler (for multiple files)
# --------------------------
def process_files(uploaded_files):
    """
    Accepts either:
      - list of Streamlit UploadedFile objects (with .name and .read()), or
      - list of local file path strings
      - or a mixed list of both.
    Returns: (pandas.DataFrame, errors_list)
    """
    all_records = []
    error_list = []

    for item in uploaded_files:
        # Determine path
        if isinstance(item, str):
            # it's already a path on disk
            file_path = item
            display_name = os.path.basename(item)
        else:
            # assume it's a file-like object (UploadedFile)
            # write it to a temp file and use that path
            try:
                display_name = getattr(item, "name", "uploaded_file")
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(display_name)[1] or ".jpg")
                tmp.write(item.read())
                tmp.close()
                file_path = tmp.name
            except Exception as e:
                error_list.append({"file": getattr(item, "name", str(item)), "error": f"Failed to save uploaded file: {e}"})
                continue

        # Now call extract_data(file_path)
        try:
            records, err = extract_data(file_path)
            if err:
                error_list.append({"file": display_name, "error": err})
            else:
                # append records (list of dicts)
                for r in records:
                    # ensure 'File' column exists and uses the display name (not temp path)
                    if "File" not in r or not r["File"]:
                        r["File"] = display_name
                    all_records.append(r)
        except Exception as e:
            error_list.append({"file": display_name, "error": str(e)})

        # If we created a temp file from an UploadedFile, we can optionally remove it
        if not isinstance(item, str):
            try:
                os.remove(file_path)
            except Exception:
                pass

    df = pd.DataFrame(all_records) if all_records else pd.DataFrame()
    return df, error_list

