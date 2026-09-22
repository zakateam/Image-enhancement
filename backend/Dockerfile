# Start from an official Python image (this is the base "environment")
FROM python:3.10-slim

# Some image/AI libraries (opencv, torch) need these system tools installed
# curl is needed to download the model files below
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a folder inside the container for our app
WORKDIR /app

# Copy just the requirements file first (this makes rebuilds faster)
COPY backend/requirements.txt ./backend/requirements.txt

# Install all Python packages your backend needs
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the backend code AND the BasicSR folder, keeping them as siblings
# (main.py expects BasicSR to sit next to the backend folder, not inside it)
COPY backend/ ./backend/
COPY BasicSR/ ./BasicSR/

# --------------------------------------------------------------
# Download the model weight files.
# --------------------------------------------------------------
RUN mkdir -p backend/models && \
    curl -L "https://huggingface.co/zakateam/image-enhancement-weights/resolve/main/RealESRGAN_x4plus.pth" -o backend/models/RealESRGAN_x4plus.pth && \
    curl -L "https://huggingface.co/zakateam/image-enhancement-weights/resolve/main/realesrgan_finetuned_2000.pth" -o backend/models/realesrgan_finetuned_2000.pth

WORKDIR /app/backend

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
