FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 기본 패키지만 설치 (빌드 도구 불필요)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# requirements.txt에서 llama-cpp-python 제외한 것들 먼저 설치
COPY requirements.txt .

# ✅ 핵심: 사전 빌드된 CUDA wheel 설치
RUN pip3 install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 && \
    grep -v "llama-cpp-python" requirements.txt > /tmp/requirements_no_llama.txt && \
    pip3 install --no-cache-dir -r /tmp/requirements_no_llama.txt

COPY . .

# Streamlit 설정
RUN mkdir -p ~/.streamlit && \
    echo "[server]" > ~/.streamlit/config.toml && \
    echo "headless = true" >> ~/.streamlit/config.toml && \
    echo "port = 7860" >> ~/.streamlit/config.toml && \
    echo "enableCORS = false" >> ~/.streamlit/config.toml && \
    echo "enableXsrfProtection = false" >> ~/.streamlit/config.toml

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]