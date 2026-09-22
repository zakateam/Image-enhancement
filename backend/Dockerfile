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
COPY requirements.txt .

# Install all Python packages your backend needs
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your backend code (main.py, BasicSR folder, etc.)
COPY . .

# --------------------------------------------------------------
# Download the model weight files.
# These are too large for GitHub, so they're hosted on Hugging Face
# instead, and downloaded here every time the image is built.
# --------------------------------------------------------------
RUN mkdir -p models && \
    curl -L "https://huggingface.co/zakateam/image-enhancement-weights/resolve/main/RealESRGAN_x4plus.pth" -o models/RealESRGAN_x4plus.pth && \
    curl -L "https://huggingface.co/zakateam/image-enhancement-weights/resolve/main/realesrgan_finetuned_2000.pth" -o models/realesrgan_finetuned_2000.pth

# Tell Docker which port the app will run on
EXPOSE 8000

# The command that actually starts your server when the container runs
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
