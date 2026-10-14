FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    bash \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
       -e ".[all]" onnx onnxruntime alkaid

COPY . .

WORKDIR /app/tests
CMD ["bash", "run_tests.sh"]
