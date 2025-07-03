FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the necessary application files
COPY car_verification.py .
COPY car_verification_api.py .
COPY database.py .

# Create necessary directories
RUN mkdir -p static/visualizations uploads/reference uploads/verification

# Copy the YOLO model file if it exists locally
COPY yolo11x.pt* ./

# Create a script to ensure model setup
RUN echo '#!/bin/bash\n\
if [ ! -f yolo11x.pt ]; then\n\
    echo "ERROR: yolo11x.pt model file not found. Please ensure the model file is available."\n\
    exit 1\n\
fi\n\
\n\
# Start the API server\n\
uvicorn car_verification_api:app --host 0.0.0.0 --port 8000' > start.sh && \
    chmod +x start.sh

# Expose the port
EXPOSE 8000

# Run the application
CMD ["./start.sh"] 