# main.py
import os
import base64
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI  # pip install openai
import uvicorn
import asyncio
import logging

# Config
API_TOKEN_HEADER = "x-backend-token"  # simple auth between Streamlit and backend
EXPECTED_BACKEND_TOKEN = os.environ.get("BACKEND_TOKEN")  # set on Render
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # change as available

if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY env var not set")

client = OpenAI(api_key=OPENAI_KEY)
app = FastAPI()
logger = logging.getLogger("uvicorn.error")

class ExtractResponse(BaseModel):
    first_name: Optional[str] = ""
    middle_name: Optional[str] = ""
    surname: Optional[str] = ""
    class_std: Optional[str] = ""
    mobile: Optional[str] = ""
    school: Optional[str] = ""
    medium: Optional[str] = ""
    raw_text: Optional[str] = ""


def build_system_prompt():
    return (
        "You are a precise form-extraction assistant. "
        "Given a single image of a fixed printed school registration form, extract the following fields "
        "in JSON only: first_name, middle_name, surname, class, mobile, school, medium. "
        "Return a single JSON object with these keys (use empty string for missing values). "
        "Do NOT add commentary or extra fields. Mobile should be digits only (10 digits preferred). "
        "If unsure, return the best guess but do not fabricate unrelated values."
    )


async def call_openai_with_image_b64(image_b64: str):
    # Build the input payload for the responses API (vision-enabled)
    system_prompt = build_system_prompt()

    # Create a multimodal user message: a short prompt + the image as a data URI
    user_input = [
        {"type": "input_text", "text": "Extract these fields from the attached registration form."},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"}
    ]

    # Use the Responses API via client.responses.create (structure depends on client version).
    # The 'responses' call returns 'output' with content blocks.
    try:
        resp = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            max_output_tokens=500
        )
    except Exception as e:
        logger.exception("OpenAI call failed")
        raise

    # Parse the returned text. The structure can vary; try to get textual JSON from output_text
    # New client often exposes resp.output_text or resp.output[0].content
    text_out = ""
    try:
        # sometimes API provides a helper
        text_out = getattr(resp, "output_text", None)
        if not text_out:
            # fallback: scan resp.output
            outputs = resp.output if hasattr(resp, "output") else resp.get("output", [])
            # outputs is list of content blocks; find the text block
            parts = []
            for o in outputs:
                if isinstance(o, dict) and "content" in o:
                    for c in o["content"]:
                        if c.get("type") in ("output_text", "text"):
                            parts.append(c.get("text") or c.get("data") or "")
                        elif c.get("type") == "message":
                            parts.append(c.get("content", ""))
            text_out = "\n".join(parts).strip()
    except Exception:
        text_out = str(resp)

    # If we got text that is JSON, try to parse it; otherwise return raw_text
    parsed = {}
    try:
        # sometimes the model replies with a JSON block inside text; extract first {...}
        start = text_out.find("{")
        end = text_out.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_blob = text_out[start:end+1]
            parsed = json.loads(json_blob)
    except Exception:
        parsed = {}

    return text_out, parsed


@app.post("/extract", response_model=ExtractResponse)
async def extract(file: UploadFile = File(...), x_backend_token: Optional[str] = Header(None)):
    # Simple auth between Streamlit and backend
    if EXPECTED_BACKEND_TOKEN and x_backend_token != EXPECTED_BACKEND_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid backend token")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    # limit file size: e.g., 5 MB
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")

    # encode base64
    image_b64 = base64.b64encode(contents).decode("utf-8")

    # Call OpenAI
    try:
        raw_text, parsed_json = await call_openai_with_image_b64(image_b64)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {e}")

    # Map parsed keys to our output schema
    def get_val(k):
        return parsed_json.get(k) if isinstance(parsed_json, dict) else None

    out = ExtractResponse(
        first_name = get_val("first_name") or get_val("firstName") or "",
        middle_name = get_val("middle_name") or get_val("middleName") or "",
        surname = get_val("surname") or get_val("last_name") or get_val("lastName") or "",
        class_std = get_val("class") or get_val("class_std") or "",
        mobile = (get_val("mobile") or "").strip(),
        school = get_val("school") or "",
        medium = get_val("medium") or "",
        raw_text = raw_text
    )

    # Basic mobile normalization (digits only)
    if out.mobile:
        out.mobile = "".join([c for c in out.mobile if c.isdigit()])[:10]

    return out


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
