# Builds the pieces the API/worker actually need from the C++ engine: the
# pente binary (api/engine.py shells out to it) and the pente_native pybind11
# extension (api/zobrist.py). Deliberately builds just those two targets, not
# the whole project (train/generate/unit_tests/etc. aren't needed here, and
# skipping LibTorch keeps this build fast and the image small - revisit if
# the API ever needs an NN evaluator).
FROM python:3.12-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY CMakeLists.txt ./
COPY include/ include/
COPY src/ src/
COPY apps/ apps/
# tests/ isn't built here (only the pente/pente_native --target below are),
# but CMakeLists.txt's add_executable(unit_tests tests/...) etc. are
# evaluated for every target at configure time regardless, so the listed
# source files still need to exist on disk or `cmake -B build` itself fails.
COPY tests/ tests/
COPY requirements.txt ./

RUN pip install --no-cache-dir pybind11

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release . \
    && cmake --build build --target pente pente_native -j"$(nproc)"

# ─── Runtime ────────────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=build /app/build/pente build/pente
COPY --from=build /app/build/pente_native*.so build/
COPY api/ api/

# Run as a non-root user rather than root (celery warns loudly about this
# otherwise) - shared by both the api and worker services, since they're the
# same image. /data is chowned here so the named volume mounted over it (see
# docker-compose.yml) inherits these permissions on first creation - if
# you're updating an existing setup that already ran as root,
# `docker compose down -v` first (recreates the volume) or the worker will
# hit a permission error writing to it.
RUN useradd -m celeryuser && mkdir -p /data && chown -R celeryuser:celeryuser /app /data
USER celeryuser

ENV BOOK_DB_PATH=/data/book.db
VOLUME /data

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
