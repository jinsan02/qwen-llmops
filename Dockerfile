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


# ── GGUF(llama.cpp) 런타임 ─────────────────────────────────────────────────
# 대전제 변경: M5를 llama.cpp Q4_K_M로 구동. RPi5(ARM64)는 llama-cpp-python을
# 소스 빌드하므로 cmake/컴파일러 필요. ONNX Runtime은 설치하지 않는다.
FROM python:3.12-slim-bookworm AS gguf-runtime

RUN apt-get update && apt-get install -y \
    build-essential cmake libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# ONNX 계열 제거(qwen_gguf는 onnxruntime 불필요) + llama-cpp-python 추가
RUN grep -viE '^onnxruntime|^optimum' requirements.txt > requirements.gguf.txt \
    && pip install --no-cache-dir --timeout 600 -r requirements.gguf.txt \
    && pip install --no-cache-dir --timeout 600 "llama-cpp-python>=0.3.0"

COPY . .

ENV SLM_BACKEND=gguf \
    SLM_MODEL=qwen_15b_gguf \
    SLM_TOKENIZER=qwen_15b

CMD ["python", "service/qwen_service.py"]
