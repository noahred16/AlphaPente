# Pure-Python image, deliberately not building any C++/LibTorch here (see
# the root Dockerfile's own comment on the same tradeoff for api/worker).
# `replay` (apps/Replay.cpp) is dynamically linked against the CUDA-enabled
# LibTorch vendored at libs/libtorch/ — a 4.1GB, gitignored directory not
# available in a Docker build context — so this container never compiles it;
# it's bind-mounted in from the host's own `build/` at run time instead
# (see docker-compose.yml). The one real runtime dependency that isn't
# covered by that mount is libcudart.so.12 itself (the plain, unmangled one —
# LibTorch bundles its own separately renamed copy, but still depends on this
# system one too), which is why this uses an Ubuntu base: `libcudart12` is a
# small (~700KB) package in Ubuntu's own default `multiverse` component, no
# NVIDIA apt repo needed. A Debian-based image (e.g. python:slim) doesn't
# carry that package.
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libcudart12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY server.py ./
COPY static/ static/

EXPOSE 8010
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8010"]
