import os
import tempfile
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import cv2
import pytesseract
import requests
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
TESSERACT_PATH = os.getenv("TESSERACT_PATH")
POPPLER_PATH = os.getenv("POPPLER_PATH")

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Gemini API endpoint
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

# Preprocessing - Adaptive Threshold
def preprocess_adaptive(gray_img):
    blurred = cv2.medianBlur(gray_img, 3)
    adaptive = cv2.adaptiveThreshold(blurred, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 15, 10)
    return cv2.bitwise_not(adaptive)

# Preprocessing - Otsu Threshold
def preprocess_otsu(gray_img):
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu

# OCR using Tesseract
def extract_text(preprocessed_img):
    config = '--oem 1 --psm 6'
    return pytesseract.image_to_string(preprocessed_img, config=config).strip()

# Convert PDF to list of PIL Images
def convert_pdf_to_images(pdf_path):
    return convert_from_path(pdf_path, poppler_path=POPPLER_PATH, dpi=300)

# Convert PIL Image to OpenCV format
def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Call Gemini API
def call_gemini_api(adaptive_text, otsu_text, question):
    combined_input = f"Adaptive OCR Output:\n{adaptive_text}\n\nOtsu OCR Output:\n{otsu_text}"
    prompt = ("just merge these two (combine it) and give out the error corrected deciphered version alone "
              "(give everything given there, but dont add extra content, just decipher and only get help from context). "
              "add nothing else in your response. the text is an answer for the question: " + question + 
              ". The text is: " + combined_input)

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(f"{API_URL}?key={API_KEY}", headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Gemini API Error: {response.status_code} - {response.text}")

# Flask App Setup
app = Flask(__name__)

@app.route("/ocr", methods=["POST"])
def ocr_api():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    if "question" not in request.form:
        return jsonify({"error": "No question provided"}), 400

    file = request.files["file"]
    question = request.form["question"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        file.save(tmp_file.name)
        pdf_path = tmp_file.name

    try:
        pages = convert_pdf_to_images(pdf_path)
        all_adaptive_text = ""
        all_otsu_text = ""

        for i, pil_img in enumerate(pages, 1):
            img_cv2 = pil_to_cv2(pil_img)
            gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

            adaptive_img = preprocess_adaptive(gray)
            otsu_img = preprocess_otsu(gray)

            adaptive_text = extract_text(adaptive_img)
            otsu_text = extract_text(otsu_img)

            all_adaptive_text += f"\n\n[Page {i}]\n{adaptive_text}"
            all_otsu_text += f"\n\n[Page {i}]\n{otsu_text}"

        gemini_response = call_gemini_api(all_adaptive_text, all_otsu_text, question)

        if 'candidates' in gemini_response:
            final_text = gemini_response['candidates'][0]['content']['parts'][0]['text']
            return jsonify({"result": final_text})
        else:
            return jsonify({"error": "No valid response from Gemini."}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(pdf_path)

if __name__ == "__main__":
    app.run(debug=True)
