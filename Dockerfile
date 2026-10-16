# syntax=docker/dockerfile:1@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
FROM scratch AS wheels
COPY docker/wheels/ /

FROM python:3.10.20-slim-bookworm@sha256:6dafe336b64dbf76f7ec872d07e58b5c7de0e2ea5d4c0853a05c3b6be3278b00 AS runtime
ARG APT_MIRROR=https://deb.debian.org
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG PIP_NO_INDEX=0
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy \
    WANDB_MODE=disabled \
    OMP_NUM_THREADS=8 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN sed -i "s|http://deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential git libegl1 libgl1 libglib2.0-0 libgomp1
WORKDIR /opt/diffusion-policy
COPY docker/requirements.lock.txt /tmp/requirements.lock.txt
RUN --mount=type=bind,from=wheels,source=/,target=/wheels \
    pip install --no-cache-dir --find-links=/wheels pip==23.3.2 setuptools==65.5.0 wheel==0.38.4 cmake==3.24.3 && \
    pip install --no-cache-dir --no-build-isolation --find-links=/wheels -r /tmp/requirements.lock.txt && \
    pip check
COPY third_party/diffusion_policy/ third_party/diffusion_policy/
RUN pip install --no-cache-dir --no-build-isolation --no-deps -e third_party/diffusion_policy
COPY scripts/ scripts/

FROM runtime AS evaluation
LABEL org.opencontainers.image.title="Diffusion Policy Push-T reproduction" \
      org.opencontainers.image.source="https://github.com/LAOBIEXL/diffusion-policy-reproduction" \
      org.opencontainers.image.description="Official epoch=0500 EMA policy, 50 Push-T visual test episodes"
COPY data/docker/ data/docker/
ENTRYPOINT ["python", "/opt/diffusion-policy/scripts/container_evaluate.py"]
