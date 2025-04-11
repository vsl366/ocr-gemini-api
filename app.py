import os
from dotenv import load_dotenv
import cv2
import pytesseract
import requests
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

# Load environment variables from .env
load_dotenv()

# Get values from environment variables
API_KEY = os.getenv("GEMINI_API_KEY")
TESSERACT_PATH = os.getenv("TESSERACT_PATH")
POPPLER_PATH = os.getenv("POPPLER_PATH")

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# API URL for Gemini
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

# Preprocessing Methods
def preprocess_adaptive(gray_img):
    blurred = cv2.medianBlur(gray_img, 3)
    adaptive = cv2.adaptiveThreshold(blurred, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 15, 10)
    return cv2.bitwise_not(adaptive)

def preprocess_otsu(gray_img):
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu

# OCR Function
def extract_text(preprocessed_img):
    config = '--oem 1 --psm 6'
    return pytesseract.image_to_string(preprocessed_img, config=config).strip()

# Convert PDF to image list
def convert_pdf_to_images(pdf_path):
    return convert_from_path(pdf_path, poppler_path=POPPLER_PATH, dpi=300)

# Convert PIL image to OpenCV
def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Call Gemini API
def call_gemini_api(adaptive_text, otsu_text):
    combined_input = f"Adaptive OCR Output:\n{adaptive_text}\n\nOtsu OCR Output:\n{otsu_text}"
    question = "1)Why and for What POSIX is used for2)talk about join in sql3)list classical and romantic era composers"
    prompt = ("just merge these two (combine it) and give out the error corrected deciphered version alone"
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

# Main Driver
def main(pdf_path):
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

    # Send to Gemini
    try:
        gemini_response = call_gemini_api(all_adaptive_text, all_otsu_text)
        if 'candidates' in gemini_response:
            final_text = gemini_response['candidates'][0]['content']['parts'][0]['text']
            print(final_text)
        else:
            print("No valid response from Gemini.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    pdf_path = "ocrpdf.pdf"
    main(pdf_path)
