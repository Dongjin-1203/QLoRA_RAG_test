FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# 환경 변수
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python을 python으로 심볼릭 링크
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# 작업 디렉토리
WORKDIR /app

# 의존성 먼저 설치 (캐싱 최적화)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 앱 파일 복사
COPY . .

# Streamlit 설정
RUN mkdir -p ~/.streamlit && \
    echo "[server]" > ~/.streamlit/config.toml && \
    echo "headless = true" >> ~/.streamlit/config.toml && \
    echo "port = 7860" >> ~/.streamlit/config.toml && \
    echo "enableCORS = false" >> ~/.streamlit/config.toml && \
    echo "enableXsrfProtection = false" >> ~/.streamlit/config.toml

# 포트 노출
EXPOSE 7860

# Streamlit 실행 (이게 핵심!)
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]