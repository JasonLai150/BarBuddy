# BarBuddy 3D Pose Pipeline v2
# CUDA runtime for GPU inference (PyTorch, YOLO, RTMPose, VideoPose3D)
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    ffmpeg \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Install Python deps (layer-cached; put before COPY src)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Pre-download YOLO model weights into the image so first run is fast
RUN python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"

COPY src ./src

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "src.worker_entry"]
