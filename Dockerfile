# syntax=docker/dockerfile:1

# IsoFATE for AWS Batch on Fargate. Build with --platform linux/amd64; jobs run, for example,
#     python -m isofate.mc_timeseries s3://BUCKET/outputs/RUN/shard-{shard:05d}.npz --n-planets 100

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
# dependencies first, so code changes reuse this layer
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-dev --extra aws --no-install-project
COPY pyproject.toml uv.lock ./
COPY isofate isofate
# uv reuses a cached isofate wheel until pyproject.toml changes, so rebuild it from the copied source
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra aws --no-editable --reinstall-package isofate

FROM python:3.13-slim-bookworm
RUN useradd --create-home isofate
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
USER isofate
WORKDIR /home/isofate
CMD ["python", "-m", "isofate.mc_timeseries", "--help"]
