# Dockerfile for ppe-compliance-monitor-data
# Uploads model and video files to MinIO (idempotent - skips if already exists)
#
# Build context must be the ``app`` directory (see Makefile ``build-push-data``):
#   podman build -f app/data-image/Dockerfile app
#
# Weights: add or replace files under **app/models/** (e.g. ``bird.pt``, ``ppe.pt``)
# *before* building. Stage 2 runs ``export_models.py``, which exports every ``*.pt`` in
# that folder to OpenVINO IR and ONNX, except stems listed in ``EXPORT_EXCLUDE_STEMS``
# (default: ``custome_ppe`` only).

# Stage 1: Extract compressed archives + stage raw .pt weights
FROM alpine:3.19 AS extractor

RUN apk add --no-cache p7zip

WORKDIR /extract

# Copy video archives and extract
COPY data/combined-video-no-gap-rooftop.mp4.7z.* ./data/
RUN cd data && 7z x combined-video-no-gap-rooftop.mp4.7z.001

# Sample MP4 for bird demo (same bucket as rooftop clip; see upload.sh)
COPY data/bluejayclear.mp4 ./data/

COPY models /extract/models/

# Stage 2: Export each *.pt to OpenVINO IR + ONNX
FROM docker.io/ultralytics/ultralytics:latest-cpu AS exporter

COPY --from=extractor /extract/models /weights
COPY data-image/export_models.py /tmp/export_models.py

RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git && \
    python3 /tmp/export_models.py

# Stage 3: Create uploader image with mc CLI
FROM docker.io/minio/mc:latest

WORKDIR /upload

COPY --from=extractor /extract/data/combined-video-no-gap-rooftop.mp4 /upload/data/
COPY --from=extractor /extract/data/bluejayclear.mp4 /upload/data/
COPY --from=extractor /extract/models /upload/models-pt/
COPY --from=exporter /exported-openvino/ /upload/models/
# ONNX for Triton: triton/<name>/1/model.onnx; OpenVINO IR: ovms/<name>/1/
COPY --from=exporter /exported-onnx/ /upload/models/
COPY data-image/config/config.pbtxt /upload/triton-config/config.pbtxt
COPY data-image/upload.sh /upload/upload.sh
RUN chmod +x /upload/upload.sh

ENV MINIO_ENDPOINT=http://minio:9000 \
    MINIO_ACCESS_KEY=minioadmin \
    MINIO_SECRET_KEY=minioadmin

ENTRYPOINT ["/bin/sh", "/upload/upload.sh"]
