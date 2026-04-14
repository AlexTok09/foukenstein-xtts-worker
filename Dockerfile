# syntax=docker/dockerfile:1.6
# ─────────────────────────────────────────────────────────────────────────────
# Foukenstein XTTS — Runpod Serverless worker
#
# Les poids XTTS NE SONT PAS dans l'image : ils vivent sur le Network Volume
# srvvzpk6jj (EU-RO-1), monté à /runpod-volume/ par Runpod. Voir
# deployment_v1/NETWORK_VOLUME.md pour les chemins exacts.
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        libsndfile1 \
        ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch d'abord (wheels CUDA depuis l'index PyTorch) — gros layer, caché.
RUN pip install --no-cache-dir \
        torch==2.1.0 torchaudio==2.1.0 \
        --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# Pas de HTTP server — runpod.serverless.start() ouvre son propre listener.
CMD ["python", "-u", "handler.py"]
