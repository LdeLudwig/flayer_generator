FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN mkdir /app

COPY requirements.txt /app/requirements.txt

# install the pytorch lib for gpu processing
RUN uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN uv pip install --system --no-cache -r /app/requirements.txt

COPY api /app/api
COPY db /app/db
COPY utils /app/utils
COPY *py /app

WORKDIR /app/