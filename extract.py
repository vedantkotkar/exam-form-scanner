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
