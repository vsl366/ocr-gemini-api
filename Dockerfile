FROM python:3.10-slim

# Install system packages: poppler (for pdf2image), tesseract (for OCR)
RUN apt-get update && \
    apt-get install -y poppler-utils tesseract-ocr && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy everything into the container
COPY . .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 5000

# Run your app
CMD ["python", "app.py"]
