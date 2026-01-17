# Use an NVIDIA CUDA image as the base
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
# nvidia/cuda:12.6.3-cudnn-runtime-ubuntu20.04

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="${PATH}:/home/user/.local/bin"

# We love UTF!
ENV LANG C.UTF-8

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Set the nvidia container runtime environment variables
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV CUDA_HOME="/usr/local/cuda"
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX 8.9"

# Install some handy tools. Even Guvcview for webcam support!
RUN set -x \
        && apt-get update \
        && apt-get install -y apt-transport-https ca-certificates \
        && apt-get install -y git vim tmux nano htop sudo curl wget gnupg2 \
        && apt-get install -y bash-completion \
        && apt-get install -y guvcview \
        && rm -rf /var/lib/apt/lists/* \
        && useradd -ms /bin/bash user \
        && echo "user:user" | chpasswd && adduser user sudo \
        && echo "user ALL=(ALL) NOPASSWD: ALL " >> /etc/sudoers

RUN set -x \
    && apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN set -x \
    && apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-venv python3.11-dev \
    && apt-get install -y python3.11-tk

#RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2

RUN apt-get install -y python3 python3-pip python3-venv

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /home/user


RUN git clone https://github.com/facebookresearch/sam2 && \
    cd sam2 && \
    # git checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4 && \
    git checkout 0f6515ae853c40420ea8e3dd250f8031bbf03023 && \
    python3 -m pip install -e . -v --break-system-packages --ignore-installed && \
    python3 -m pip install flask imageio[ffmpeg] nibabel APScheduler gunicorn pydicom --break-system-packages --ignore-installed && \
    python3 -m pip install -e ".[demo]" --break-system-packages --ignore-installed && \
    cd checkpoints && ./download_ckpts.sh && cd ..

RUN usermod -aG dialout user
USER user
STOPSIGNAL SIGTERM


CMD sudo service ssh start && /bin/bash