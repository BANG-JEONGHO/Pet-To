# 1. 파이썬 3.11 Slim 버전 사용 (가볍고 빠름)
FROM python:3.11-slim

# 2. 컨테이너 내부의 작업 디렉토리 설정
WORKDIR /app

# 3. 시스템 필수 패키지 설치 (OpenCV 구동 등)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. 파이썬 라이브러리 설치
# (캐시를 사용하지 않아 이미지 용량을 최적화합니다)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 및 weights 폴더 복사
# 🚨 이 단계에서 미리 다운받은 weights 폴더 안의 6~7GB 파일들이 도커 이미지 안으로 쏙 들어갑니다!
COPY . .

# 6. 서버 포트 오픈 (Cloud Run 기본 포트)
EXPOSE 8080

# 7. FastAPI 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]