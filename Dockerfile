# 1. Use NVIDIA Base Image (Guarantees GPU Drivers are present)
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# 2. Prevent Python from buffering stdout/stderr (better logs)
ENV PYTHONUNBUFFERED=1

# 3. Install Python 3.11 and System Dependencies
# libgl1 and libglib2.0-0 are strictly required for OpenCV
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Alias python3.11 to python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# 5. Install Python Dependencies
COPY requirements.txt .

# Upgrade pip first to avoid errors with newer wheels
RUN pip install --no-cache-dir --upgrade pip

# Install requirements (Includes GPU versions of rembg & torch)
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Code
COPY . .

# 7. Expose Port 8002 (Updated)
EXPOSE 8002

# 8. Run the Service on Port 8002 (Updated)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]