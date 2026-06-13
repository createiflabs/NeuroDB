# syntax=docker/dockerfile:1

# ---- builder: install dependencies and the package into an isolated venv ----
# For reproducible builds, pin the base by digest (managed by Dependabot):
#   FROM python:3.12-slim@sha256:<digest> AS builder
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install third-party deps first so this layer is cached across code changes.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Install NeuroDB itself (deps already satisfied above).
COPY pyproject.toml README.md ./
COPY neurodb ./neurodb
RUN pip install --no-deps .

# ---- runtime: slim image with only the venv and a non-root user ----
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="NeuroDB" \
      org.opencontainers.image.description="Content-addressable store powered by Modern Hopfield networks: pattern completion, per-field anomaly detection, similarity search, single-file persistence." \
      org.opencontainers.image.source="https://github.com/createiflabs/NeuroDB" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.vendor="createif labs"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    NEURODB_DATA_FILE=/data/neurodb.npz \
    NEURODB_HOST=0.0.0.0 \
    NEURODB_PORT=8000

COPY --from=builder /opt/venv /opt/venv

RUN useradd --create-home --uid 10001 neurodb \
    && mkdir -p /data \
    && chown -R neurodb:neurodb /data

USER neurodb
WORKDIR /home/neurodb
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=4s --start-period=5s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('NEURODB_PORT','8000'))" || exit 1

CMD ["python", "-m", "neurodb"]
