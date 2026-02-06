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

# Pre-download all model weights into the image so first run has zero cold-start
ENV BARBUDDY_MODEL_CACHE=/root/.cache/barbuddy/models
RUN mkdir -p $BARBUDDY_MODEL_CACHE

# YOLOv8m — person detection (Stage 1)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"

# RTMPose-m ONNX — 2D pose estimation (Stage 2)
RUN python -c "
import urllib.request, zipfile, os
cache = os.environ['BARBUDDY_MODEL_CACHE']
url = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip'
zip_path = os.path.join(cache, 'rtmpose-m.zip')
urllib.request.urlretrieve(url, zip_path)
with zipfile.ZipFile(zip_path, 'r') as zf:
    onnx = [m for m in zf.namelist() if m.endswith('.onnx')][0]
    zf.extract(onnx, cache)
    extracted = os.path.join(cache, onnx)
    target = os.path.join(cache, 'rtmpose-m.onnx')
    if extracted != target:
        os.rename(extracted, target)
os.remove(zip_path)
print('RTMPose ONNX:', target)
"

# VideoPose3D — temporal 3D lifter (Stage 3)
RUN python -c "
import urllib.request, os
cache = os.environ['BARBUDDY_MODEL_CACHE']
url = 'https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin'
out = os.path.join(cache, 'videopose3d_h36m_cpn.bin')
urllib.request.urlretrieve(url, out)
print('VideoPose3D:', out)
"

COPY src ./src

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "src.worker_entry"]
