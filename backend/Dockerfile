FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

RUN mkdir /app

COPY requirements.txt /app/requirements.txt

# install the pytorch lib for gpu processing
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY api /app/api
COPY db /app/db
COPY utils /app/utils
COPY *py /app

WORKDIR /app/