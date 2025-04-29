ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
ARG MODEL_SIZE=base_plus

FROM ${BASE_IMAGE}

# Gunicorn environment variables
ENV GUNICORN_WORKERS=1
ENV GUNICORN_THREADS=2
ENV GUNICORN_PORT=5000

# SAM 2 environment variables
ENV APP_ROOT=/opt/sam2
ENV PYTHONUNBUFFERED=1
ENV SAM2_BUILD_CUDA=0
ENV MODEL_SIZE=${MODEL_SIZE}

# Install system requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavutil-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    pkg-config \
    build-essential \
    libffi-dev \
    git \
    wget \
    curl


RUN git clone https://github.com/facebookresearch/sam2 && \
    cd sam2  && \
    # git checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4 && \
    python3 -m pip install  --upgrade-strategy only-if-needed -e . -v && \
    python3 -m pip install  --upgrade-strategy only-if-needed flask imageio[ffmpeg] nibabel APScheduler gunicorn pydicom && \
    python3 -m pip install   --upgrade-strategy only-if-needed -e ".[demo]" && \
    cd checkpoints && ./download_ckpts.sh && cd ..

#RUN usermod -aG dialout user
#USER user
#STOPSIGNAL SIGTERM
RUN python3 -m pip install matplotlib
CMD sudo service ssh start && /bin/bash
