# syntax=docker/dockerfile:1.7
# =============================================================================
# EvalPipeline Pro — multi-stage image
#
#   base      -> Python 3.9 slim + non-root user (shared by every stage)
#   api-deps  -> CPU-only torch + runtime requirements (cached layer, rebuilt only
#                when requirements.txt changes)
#   api       -> production FastAPI image            (docker build --target api .)
#   dev       -> api + pytest/ruff + hot reload      (used by docker-compose.yml)
#   dashboard -> Streamlit UI, no ML deps           (docker build --target dashboard .)
# =============================================================================

ARG PYTHON_VERSION=3.9

# ----------------------------------------------------------------------------- base
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # HuggingFace / sentence-transformers model cache (mounted as a named volume in compose)
    HF_HOME=/cache/hf \
    SENTENCE_TRANSFORMERS_HOME=/cache/hf

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system app \
 && useradd --system --gid app --create-home --home-dir /home/app app \
 && mkdir -p /cache/hf \
 && chown -R app:app /app /cache

# ----------------------------------------------------------------------------- api-deps
FROM base AS api-deps

# Keep pip itself current (the slim image ships an older pip whose resolver is slower).
RUN pip install --upgrade pip

# 1) CPU-only PyTorch from the official CPU index. Pulling torch via requirements.txt
#    on linux/amd64 would drag ~2 GB of NVIDIA CUDA libraries into the image.
#    The index serves x86_64 (+cpu), aarch64 (Apple-silicon Docker) and cp39 wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

# 2) Everything else. `torch==2.5.1` in requirements.txt is already satisfied by step 1.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# ----------------------------------------------------------------------------- api (prod)
FROM api-deps AS api

COPY --chown=app:app pyproject.toml ./
COPY --chown=app:app app ./app
COPY --chown=app:app scripts ./scripts

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# TODO(hosting): behind a real load balancer add `--proxy-headers --forwarded-allow-ips=*`
#                and tune `--workers` to the CPU count of the host/task.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ----------------------------------------------------------------------------- dev
FROM api-deps AS dev

COPY requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

# Source is bind-mounted by docker-compose for hot reload; copy once so the image
# is still runnable on its own (e.g. `docker run ... pytest`).
COPY --chown=app:app pyproject.toml ./
COPY --chown=app:app app ./app
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app tests ./tests

# Bind mounts are owned by the host user; stop pytest writing a cache dir into them.
ENV PYTEST_ADDOPTS="-p no:cacheprovider"

USER app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ----------------------------------------------------------------------------- dashboard
FROM base AS dashboard

RUN pip install --upgrade pip
COPY requirements-dashboard.txt ./
RUN pip install -r requirements-dashboard.txt

# dashboard.py is created in Step 5; the compose service sits behind the `ui`
# profile until then so a default `docker compose up` never tries to build this stage.
COPY --chown=app:app dashboard.py ./

USER app
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
