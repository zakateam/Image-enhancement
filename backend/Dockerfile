# Start from an official Python image (this is the base "environment")
FROM python:3.10-slim

# Some image/AI libraries (opencv, torch) need these system tools installed
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a folder inside the container for our app
WORKDIR /app

# Copy just the requirements file first (this makes rebuilds faster)
COPY requirements.txt .

# Install all Python packages your backend needs
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your backend code (main.py, model files, etc.)
COPY . .

# Tell Docker which port the app will run on
EXPOSE 8000

# The command that actually starts your server when the container runs
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
