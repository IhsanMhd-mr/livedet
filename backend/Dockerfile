FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code and model
COPY . .

# Set model path
ENV BEST_MODEL_PATH=/app/yolov8s.pt
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=8000

EXPOSE 8000

CMD ["python", "app.py"]
