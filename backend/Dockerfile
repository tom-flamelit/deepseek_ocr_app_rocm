# Backend Dockerfile - FastAPI + DeepSeek-OCR (ROCm edition)
FROM rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.5.1

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY main.py .

EXPOSE 8000

# Use uvicorn with reasonable workers for GPU workload
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
