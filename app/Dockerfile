FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    ML_ENGINE_KEY=your-secret-key \
    API_MODE=local \
    LOG_LEVEL=info

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libglvnd0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker_requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade --no-cache-dir \
    -r /tmp/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY . /app

ARG VERSION=0.0.0
RUN sed -i "s/0.0.0/$VERSION/" /app/src/__init__.py

EXPOSE 90

CMD ["uvicorn", "src.run_server:app", "--host", "0.0.0.0", "--port", "90", "--reload"]
