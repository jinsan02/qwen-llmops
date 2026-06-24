FROM python:3.12-slim-bookworm AS cpu-runtime

RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# ARM64/RPi5 포함 범용 CPU 런타임: onnxruntime-gpu → onnxruntime 교체
RUN sed 's/^onnxruntime-gpu==/onnxruntime==/' requirements.txt > requirements.cpu.txt \
    && pip install --no-cache-dir --timeout 300 -r requirements.cpu.txt

COPY . .

CMD ["python", "service/qwen_service.py"]
