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


SHELL ["/bin/bash", "-c"]
RUN set -o pipefail \
    && wget -qO- https://astral.sh/uv/install.sh \
    | sh

ENV PATH="/home/$USERNAME/.local/bin/:$PATH"

RUN --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv python install

RUN --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --no-install-project --no-install-workspace --no-group cpu --group cu128

RUN --mount=type=bind,source=benchmarks/download_models.sh,target=download_models.sh \
    bash download_models.sh

COPY --chown=ubuntu:ubuntu pyproject.toml pyproject.toml
COPY --chown=ubuntu:ubuntu .python-version .python-version
COPY --chown=ubuntu:ubuntu benchmarks benchmarks
COPY --chown=ubuntu:ubuntu src src
RUN touch README.md LICENSE
WORKDIR /home/$USERNAME/benchmarks

ENTRYPOINT ["bash", "./run_benchmark.sh"]