# syntax=docker/dockerfile:1
# Lean CPU image. torch==*+cpu resolves to native wheels on aarch64 and x86_64.

FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt ./
RUN pip install -U pip && pip install -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps -e .

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:$PATH"
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 make \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/src ./src
COPY pyproject.toml Makefile README.md ./
COPY config ./config
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser
CMD ["python", "-m", "nanogpt_lab.train", "--config", "config/smoke.yaml"]
