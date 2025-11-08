import streamlit as st
from PIL import Image
import cv2
import numpy as np
import io
import os
import csv
import pandas as pd
import json
import tempfile
import re
from google.cloud import vision

# --- Streamlit Cloud credential helper ---
import os, tempfile
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

TARGET_LONG = 2400
FIELD_ZONES = {
    "full_name_line": (0.06, 0.68, 0.90, 0.735),
    "first_name_box": (0.06, 0.695, 0.30, 0.735),
    "middle_name_box": (0.30, 0.695, 0.56, 0.735),
    "surname_box": (0.56, 0.695, 0.90, 0.735),
    "class_box": (0.06, 0.755, 0.18, 0.795),
    "phone_box": (0.46, 0.755, 0.78, 0.795),
    "school_line": (0.06, 0.815, 0.90, 0.865),
    "english_medium_checkbox": (0.06, 0.645, 0.10, 0.675),
    "vernacular_checkbox": (0.40, 0.645, 0.44, 0.675),
    "total_fee_box": (0.86, 0.695, 0.98, 0.82)
}
AUTO_ACCEPT_CONF = 0.92
VISION_CLIENT = None

def init_vision():
    global VISION_CLIENT
    if VISION_CLIENT is None:
        VISION_CLIENT = vision.ImageAnnotatorClient()
    return VISION_CLIENT

def load_and_resize(path_or_bytes, target_long=TARGET_LONG):
    if isinstance(path_or_bytes, (bytes, bytearray)):
        arr = np.frombuffer(path_or_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(path_or_bytes)
    if img is None:
        raise ValueError("Could not load image")
    h, w = img.shape[:2]
    scale = target_long / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def detect_page_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype("float32")
    return None

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def crop_zone(page, zone):
    h, w = page.shape[:2]
    x1, y1, x2, y2 = [int(v * w) if i % 2 == 0 else int(v * h) for i, v in enumerate(zone)]
    return page[y1:y2, x1:x2]

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return cv2.medianBlur(clahe.apply(gray), 3)

def ocr_image_np(img_np):
    client = init_vision()
    _, im_buf_arr = cv2.imencode('.png', img_np)
    content = im_buf_arr.tobytes()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    full_text = response.full_text_annotation.text.strip() if response.full_text_annotation.text else ""
    confs = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    if hasattr(word, 'confidence'):
                        confs.append(word.confidence)
    avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
    return full_text, avg_conf

def extract_fields_from_image_bytes(image_bytes, field_zones=None):
    field_zones = field_zones or FIELD_ZONES
    img = load_and_resize(image_bytes)
    pts = detect_page_corners(img)
    page = four_point_transform(img, pts) if pts is not None else img
    results = {}
    for name, zone in field_zones.items():
        crop = crop_zone(page, zone)
        crop = cv2.resize(crop, None, fx=1.5, fy=1.5)
        prep = preprocess_for_ocr(crop)
        text, conf = ocr_image_np(prep)
        results[name] = text.strip()
        results[f"conf_{name}"] = conf
    return results

st.set_page_config(page_title='Exam Form Scanner', layout='wide')
st.title('Exam Form Scanner - Streamlit')

mode = st.sidebar.selectbox('Mode', ['Single Image', 'Zone Editor'])

if 'field_zones' not in st.session_state:
    st.session_state['field_zones'] = FIELD_ZONES

if mode == 'Single Image':
    uploaded = st.file_uploader('Upload form image', type=['jpg', 'jpeg', 'png'])
    if uploaded and st.button('Extract fields'):
        with st.spinner('Processing...'):
            res = extract_fields_from_image_bytes(uploaded.read(), st.session_state['field_zones'])
            st.write(res)

elif mode == 'Zone Editor':
    st.header('Zone Editor')
    sample = st.file_uploader('Upload sample image', type=['jpg', 'jpeg', 'png'])
    if sample:
        image_bytes = sample.read()
        img = load_and_resize(image_bytes)
        pts = detect_page_corners(img)
        page = four_point_transform(img, pts) if pts is not None else img
        st.image(Image.fromarray(cv2.cvtColor(page, cv2.COLOR_BGR2RGB)), caption='Sample Page')
        zones = st.session_state['field_zones']
        new_zones = {}
        for k, v in zones.items():
            st.subheader(k)
            x1 = st.slider(f'{k} x1', 0.0, 1.0, v[0], 0.001)
            y1 = st.slider(f'{k} y1', 0.0, 1.0, v[1], 0.001)
            x2 = st.slider(f'{k} x2', 0.0, 1.0, v[2], 0.001)
            y2 = st.slider(f'{k} y2', 0.0, 1.0, v[3], 0.001)
            new_zones[k] = (x1, y1, x2, y2)
        if st.button('Preview zones'):
            viz = page.copy()
            h, w = viz.shape[:2]
            for k, z in new_zones.items():
                cv2.rectangle(viz, (int(z[0]*w), int(z[1]*h)), (int(z[2]*w), int(z[3]*h)), (0,255,0), 2)
            st.image(Image.fromarray(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)))
        if st.button('Save zones'):
            st.session_state['field_zones'] = new_zones
            with open('zones.json', 'w') as f:
                json.dump(new_zones, f, indent=2)
            st.success('Zones saved to zones.json')
