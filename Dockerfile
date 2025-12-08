FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

WORKDIR /app

# Python 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# requirements.txt 복사
COPY requirements.txt .

# llama-cpp-python (wheel)
RUN pip install --no-cache-dir \
    llama-cpp-python==0.2.90 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# 나머지 패키지
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY . .

EXPOSE 7860

CMD ["python3", "app.py"]