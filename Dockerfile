# SnapFind Studio - Copyright (c) 2026 Perarasu Boopathirajan. All rights reserved.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run as user 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Face models (OpenCV Zoo: YuNet detector + SFace recognizer)
RUN mkdir -p models && \
    curl -fsSL -o models/face_detection_yunet_2023mar.onnx \
      https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx && \
    curl -fsSL -o models/face_recognition_sface_2021dec.onnx \
      https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx && \
    chown -R user models

COPY --chown=user app ./app
RUN mkdir -p data && chown -R user data

USER user
ENV PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
