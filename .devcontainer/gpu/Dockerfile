FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

RUN apt-get -y update \
    && apt-get install -y --no-install-recommends \
    git \
    make \
    wget \
    vim \
    build-essential \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=ubuntu

USER $USERNAME
WORKDIR /home/$USERNAME
RUN ["/bin/bash", "-c", "set -o pipefail && wget -qO- https://astral.sh/uv/install.sh | sh"]
ENV PATH="/home/$USERNAME/.local/bin/:$PATH"
RUN ls -alh
RUN pwd
RUN --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-install-project --no-install-workspace