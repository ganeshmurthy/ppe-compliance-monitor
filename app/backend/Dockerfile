FROM --platform=linux/amd64 registry.access.redhat.com/ubi9/python-311:1-77

USER 0
RUN dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm && \
    dnf install -y mesa-libGL glib2 && \
    dnf clean all
USER 1001

WORKDIR /app/backend

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /bin/

COPY backend/pyproject.toml backend/uv.lock ./

# Install dependencies
RUN uv sync --locked


COPY backend/ ./

WORKDIR /app

# ENV PYTHONPATH=/app/backend
ENV PATH="/app/backend/.venv/bin:$PATH"
ENV TRANSFORMERS_CACHE=/tmp/.cache/huggingface
ENV HF_HOME=/tmp/.cache/huggingface
ENV XDG_CACHE_HOME=/tmp/.cache
ENV MPLCONFIGDIR=/tmp/.cache/matplotlib
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

RUN mkdir -p /tmp/.cache/huggingface /tmp/.cache/matplotlib /tmp/Ultralytics && \
    chmod 777 /tmp/.cache/huggingface /tmp/.cache/matplotlib /tmp/Ultralytics

EXPOSE 8888

ENTRYPOINT ["python"]
CMD ["/app/backend/app.py"]
