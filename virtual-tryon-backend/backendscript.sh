# ============================================================================
# 1. requirements.txt - Python 의존성
# ============================================================================

# FastAPI 및 웹 서버
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6

# 이미지 처리
Pillow==10.1.0
opencv-python==4.8.1.78
scikit-image==0.22.0

# 딥러닝 및 AI
torch==2.1.0
torchvision==0.16.0
numpy==1.24.3
scipy==1.11.4
scikit-learn==1.3.2

# Computer Vision
mediapipe==0.10.7
# detectron2 설치는 별도 스크립트로 (선택사항)

# 데이터 검증
pydantic==2.5.0
pydantic-settings==2.1.0

# 파일 처리
aiofiles==23.2.1
python-dotenv==1.0.0

# 로깅 및 모니터링
structlog==23.1.0
prometheus-client==0.19.0

# 캐싱 (선택사항)
redis==5.0.0

# 유틸리티
tqdm==4.66.1
requests==2.31.0

# ============================================================================
# 2. requirements-dev.txt - 개발 의존성
# ============================================================================

# 테스팅
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
httpx==0.25.2

# 코드 품질
black==23.11.0
flake8==6.1.0
isort==5.12.0
mypy==1.7.1

# 개발 도구
jupyter==1.0.0
ipython==8.17.2

# ============================================================================
# 3. .env.example - 환경변수 예시
# ============================================================================

# Application Settings
APP_NAME=MyCloset AI Backend
APP_VERSION=1.0.0
DEBUG=false
HOST=0.0.0.0
PORT=8000

# CORS Settings
CORS_ORIGINS=http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000

# File Upload Settings
MAX_UPLOAD_SIZE=52428800  # 50MB
ALLOWED_EXTENSIONS=jpg,jpeg,png,webp,bmp

# AI Model Settings
DEFAULT_MODEL=ootd
USE_GPU=true
DEVICE=cuda
IMAGE_SIZE=512
MAX_WORKERS=4
BATCH_SIZE=1

# Paths (자동으로 설정되지만 필요시 오버라이드 가능)
# PROJECT_ROOT=/app
# AI_MODELS_DIR=/app/ai_models
# STATIC_DIR=/app/static
# UPLOAD_DIR=/app/static/uploads
# RESULTS_DIR=/app/static/results
# TEMP_DIR=/app/temp
# LOGS_DIR=/app/logs

# Performance Settings
CACHE_TTL=3600

# Logging
LOG_LEVEL=INFO
LOG_FILE=

# Redis (선택사항)
REDIS_URL=redis://localhost:6379/0

# ============================================================================
# 4. .gitignore
# ============================================================================

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Virtual environments
env/
venv/
ENV/
env.bak/
venv.bak/
.venv/

# IDEs
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Environment variables
.env
.env.local
.env.development
.env.test
.env.production

# 업로드된 파일들
static/uploads/*
static/results/*
temp/*
logs/*

# 예외: .gitkeep 파일들은 유지
!static/uploads/.gitkeep
!static/results/.gitkeep
!temp/.gitkeep
!logs/.gitkeep

# AI 모델 파일들 (용량이 큰 경우)
*.pth
*.onnx
*.h5
*.pb
*.safetensors
ai_models/*/checkpoints/*
!ai_models/*/checkpoints/.gitkeep

# 테스트 및 커버리지
.pytest_cache/
.coverage
htmlcov/
.tox/
.nox/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# ============================================================================
# 5. Dockerfile
# ============================================================================

FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Python 의존성 설치
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# 선택적으로 Detectron2 설치 (GPU 환경에서)
# RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git'

# 애플리케이션 코드 복사
COPY . .

# 필요한 디렉토리 생성
RUN mkdir -p static/uploads static/results temp logs

# .gitkeep 파일 생성
RUN touch static/uploads/.gitkeep static/results/.gitkeep temp/.gitkeep logs/.gitkeep

# 권한 설정
RUN chmod -R 755 /app

# 포트 노출
EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 서버 시작
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ============================================================================
# 6. docker-compose.yml
# ============================================================================

version: '3.8'

services:
  backend:
    build: .
    container_name: mycloset-backend
    ports:
      - "8000:8000"
    volumes:
      - ./static/uploads:/app/static/uploads
      - ./static/results:/app/static/results
      - ./logs:/app/logs
      - ./ai_models:/app/ai_models  # AI 모델 디렉토리 마운트
    environment:
      - PYTHONUNBUFFERED=1
    env_file:
      - .env
    restart: unless-stopped
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  redis:
    image: redis:7-alpine
    container_name: mycloset-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

  # 선택사항: 모니터링
  prometheus:
    image: prom/prometheus
    container_name: mycloset-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

volumes:
  redis_data:

# ============================================================================
# 7. Makefile - 개발 편의 스크립트
# ============================================================================

.PHONY: help setup install dev-install format lint test run docker-build docker-run clean models

help:
	@echo "MyCloset AI Backend - 개발 명령어"
	@echo ""
	@echo "설정 명령어:"
	@echo "  setup        - 가상환경 생성"
	@echo "  install      - 기본 의존성 설치"
	@echo "  dev-install  - 개발 의존성 설치"
	@echo "  models       - AI 모델 다운로드"
	@echo ""
	@echo "개발 명령어:"
	@echo "  format       - 코드 포맷팅"
	@echo "  lint         - 코드 린트 검사"
	@echo "  test         - 테스트 실행"
	@echo "  run          - 개발 서버 실행"
	@echo ""
	@echo "Docker 명령어:"
	@echo "  docker-build - Docker 이미지 빌드"
	@echo "  docker-run   - Docker 컨테이너 실행"
	@echo "  docker-stop  - Docker 컨테이너 중지"
	@echo ""
	@echo "유틸리티:"
	@echo "  clean        - 캐시 파일 정리"
	@echo "  dirs         - 필요한 디렉토리 생성"

# 환경 설정
setup:
	python3 -m venv venv
	@echo "가상환경이 생성되었습니다. 다음 명령어로 활성화하세요:"
	@echo "source venv/bin/activate  # Linux/Mac"
	@echo "venv\\Scripts\\activate    # Windows"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

dev-install: install
	pip install -r requirements-dev.txt

models:
	@echo "AI 모델 다운로드 스크립트 실행..."
	python scripts/download_models.py

# 개발 도구
format:
	black app tests
	isort app tests

lint:
	flake8 app tests
	mypy app

test:
	pytest tests/ -v --cov=app --cov-report=html

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Docker
docker-build:
	docker build -t mycloset-ai-backend .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f backend

# 유틸리티
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache
	rm -rf build dist *.egg-info

dirs:
	mkdir -p static/uploads static/results temp logs
	touch static/uploads/.gitkeep static/results/.gitkeep temp/.gitkeep logs/.gitkeep

# ============================================================================
# 8. scripts/download_models.py - 모델 다운로드 스크립트
# ============================================================================

"""
AI 모델 다운로드 스크립트
"""

import os
import requests
import subprocess
from pathlib import Path
from tqdm import tqdm
import zipfile
import tarfile

def download_file(url, destination, desc=None):
    """파일 다운로드 with 진행률 표시"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=desc,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def clone_repo(repo_url, destination):
    """Git 저장소 클론"""
    if not os.path.exists(destination):
        print(f"Cloning {repo_url} to {destination}")
        subprocess.run(['git', 'clone', repo_url, destination], check=True)
    else:
        print(f"Repository already exists: {destination}")

def download_huggingface_model(model_name, destination):
    """Hugging Face 모델 다운로드"""
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_name, local_dir=destination)
        print(f"Downloaded {model_name} to {destination}")
    except ImportError:
        print("huggingface_hub not installed. Skipping Hugging Face models.")

def main():
    """메인 다운로드 함수"""
    ai_models_dir = Path("ai_models")
    ai_models_dir.mkdir(exist_ok=True)
    
    print("🤖 AI 모델 다운로드 시작...")
    
    # 1. OOTDiffusion
    ootd_dir = ai_models_dir / "OOTDiffusion"
    if not ootd_dir.exists():
        print("📥 OOTDiffusion 다운로드 중...")
        clone_repo(
            "https://github.com/levihsu/OOTDiffusion.git",
            str(ootd_dir)
        )
    
    # 2. VITON-HD
    viton_dir = ai_models_dir / "VITON-HD"
    if not viton_dir.exists():
        print("📥 VITON-HD 다운로드 중...")
        clone_repo(
            "https://github.com/shadow2496/VITON-HD.git",
            str(viton_dir)
        )
    
    # 3. Stable Diffusion (Hugging Face)
    sd_dir = ai_models_dir / "stable-diffusion-v1-5"
    if not sd_dir.exists():
        print("📥 Stable Diffusion v1.5 다운로드 중...")
        download_huggingface_model(
            "runwayml/stable-diffusion-v1-5",
            str(sd_dir)
        )
    
    # 4. 체크포인트 디렉토리 생성
    checkpoint_dirs = [
        ootd_dir / "checkpoints",
        viton_dir / "checkpoints",
        ai_models_dir / "DeepFashion_Try_On" / "ACGPN_inference" / "checkpoints"
    ]
    
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / ".gitkeep").touch()
    
    print("✅ 모델 다운로드 완료!")
    print("📝 참고: 일부 모델의 사전 훈련된 가중치는 별도로 다운로드해야 합니다.")
    print("🔗 모델별 다운로드 링크:")
    print("   - OOTDiffusion: https://github.com/levihsu/OOTDiffusion#checkpoints")
    print("   - VITON-HD: https://github.com/shadow2496/VITON-HD#checkpoints")

if __name__ == "__main__":
    main()

# ============================================================================
# 9. scripts/setup.sh - 초기 설정 스크립트
# ============================================================================

#!/bin/bash

echo "🚀 MyCloset AI Backend 초기 설정 시작..."

# 1. 가상환경 생성
echo "📦 가상환경 생성 중..."
python3 -m venv venv
source venv/bin/activate

# 2. 의존성 설치
echo "📚 Python 의존성 설치 중..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. 개발 의존성 설치 (선택사항)
read -p "개발 의존성도 설치하시겠습니까? (y/N): " install_dev
if [[ $install_dev =~ ^[Yy]$ ]]; then
    pip install -r requirements-dev.txt
fi

# 4. 환경변수 파일 설정
if [ ! -f .env ]; then
    echo "⚙️ 환경변수 파일 생성 중..."
    cp .env.example .env
    echo "📝 .env 파일을 편집하여 설정을 조정하세요."
fi

# 5. 필요한 디렉토리 생성
echo "📁 디렉토리 구조 생성 중..."
mkdir -p static/uploads static/results temp logs
touch static/uploads/.gitkeep static/results/.gitkeep temp/.gitkeep logs/.gitkeep

# 6. AI 모델 다운로드
read -p "AI 모델을 다운로드하시겠습니까? (y/N): " download_models
if [[ $download_models =~ ^[Yy]$ ]]; then
    echo "🤖 AI 모델 다운로드 중..."
    python scripts/download_models.py
fi

# 7. GPU 확인
echo "🔍 GPU 확인 중..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo "✅ 초기 설정 완료!"
echo ""
echo "🚀 서버 실행 방법:"
echo "   make run              # 개발 서버"
echo "   make docker-run       # Docker로 실행"
echo ""
echo "📚 추가 명령어:"
echo "   make help             # 전체 명령어 목록"
echo "   make test             # 테스트 실행"
echo "   make format           # 코드 포맷팅"

# ============================================================================
# 10. tests/conftest.py - 테스트 설정
# ============================================================================

"""
테스트 설정 파일
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
import sys

# 테스트용 임포트
sys.path.append(str(Path(__file__).parent.parent))
from app.main import app
from app.config import settings

@pytest.fixture(scope="session")
def event_loop():
    """이벤트 루프 설정"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def client():
    """테스트 클라이언트"""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def temp_dir():
    """임시 디렉토리"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_person_image():
    """테스트용 사람 이미지"""
    import cv2
    import numpy as np
    
    # 가짜 이미지 생성 (512x512, 3채널)
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    _, encoded = cv2.imencode('.jpg', img)
    return encoded.tobytes()

@pytest.fixture
def sample_clothing_image():
    """테스트용 의류 이미지"""
    import cv2
    import numpy as np
    
    # 가짜 이미지 생성 (512x512, 3채널)
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    _, encoded = cv2.imencode('.jpg', img)
    return encoded.tobytes()

# ============================================================================
# 11. tests/test_api.py - API 테스트
# ============================================================================

"""
API 엔드포인트 테스트
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import io

def test_root_endpoint(client):
    """루트 엔드포인트 테스트"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "MyCloset AI Backend" in data["message"]
    assert "available_models" in data

def test_health_check(client):
    """헬스 체크 테스트"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "MyCloset AI"

def test_models_status(client):
    """모델 상태 확인 테스트"""
    response = client.get("/api/models/status")
    assert response.status_code == 200

@patch('app.services.virtual_tryon.VirtualTryOnService.process_virtual_tryon')
def test_virtual_tryon_endpoint(mock_process, client, sample_person_image, sample_clothing_image):
    """가상 피팅 엔드포인트 테스트"""
    # Mock 응답 설정
    mock_response = {
        "success": True,
        "session_id": "test-session-123",
        "result_image_url": "/static/results/test-session-123.jpg",
        "processing_time": 2.5,
        "model_used": "ootd",
        "confidence_score": 0.85,
        "recommendations": ["이 의류가 잘 어울립니다!"]
    }
    mock_process.return_value = mock_response
    
    # 테스트 요청
    files = {
        "person_image": ("person.jpg", io.BytesIO(sample_person_image), "image/jpeg"),
        "clothing_image": ("clothing.jpg", io.BytesIO(sample_clothing_image), "image/jpeg")
    }
    data = {
        "height": 170.0,
        "weight": 60.0,
        "model_type": "ootd",
        "category": "upper_body",
        "quality": "high"
    }
    
    response = client.post("/api/virtual-tryon", files=files, data=data)
    assert response.status_code == 200
    
    result = response.json()
    assert result["success"] == True
    assert "session_id" in result

def test_virtual_tryon_invalid_files(client):
    """잘못된 파일 업로드 테스트"""
    files = {
        "person_image": ("test.txt", io.BytesIO(b"not an image"), "text/plain"),
        "clothing_image": ("test.txt", io.BytesIO(b"not an image"), "text/plain")
    }
    data = {
        "height": 170.0,
        "weight": 60.0
    }
    
    response = client.post("/api/virtual-tryon", files=files, data=data)
    assert response.status_code == 400

# ============================================================================
# 12. tests/test_services.py - 서비스 테스트
# ============================================================================

"""
서비스 로직 테스트
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import asyncio

from app.services.virtual_tryon import VirtualTryOnService
from app.api.schemas import VirtualTryOnRequest, ModelType, ClothingCategory, QualityLevel

@pytest.mark.asyncio
async def test_virtual_tryon_service_initialization():
    """가상 피팅 서비스 초기화 테스트"""
    service = VirtualTryOnService()
    assert service.device is not None
    assert service.models == {}
    assert service.is_initialized == False

@pytest.mark.asyncio
async def test_preprocess_images():
    """이미지 전처리 테스트"""
    service = VirtualTryOnService()
    
    # 가짜 이미지 데이터 생성
    import cv2
    fake_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    _, person_bytes = cv2.imencode('.jpg', fake_img)
    _, clothing_bytes = cv2.imencode('.jpg', fake_img)
    
    person_img, clothing_img = await service._preprocess_images(
        person_bytes.tobytes(), 
        clothing_bytes.tobytes()
    )
    
    assert person_img.shape == (512, 512, 3)
    assert clothing_img.shape == (512, 512, 3)

@pytest.mark.asyncio
async def test_analyze_body():
    """신체 분석 테스트"""
    service = VirtualTryOnService()
    
    # 가짜 이미지 생성
    fake_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    measurements = await service._analyze_body(fake_img, 170.0, 60.0)
    
    assert measurements.chest > 0
    assert measurements.waist > 0
    assert measurements.hip > 0
    assert 0 <= measurements.confidence <= 1

@pytest.mark.asyncio
async def test_analyze_clothing():
    """의류 분석 테스트"""
    service = VirtualTryOnService()
    
    # 가짜 의류 이미지 생성
    fake_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    analysis = await service._analyze_clothing(fake_img)
    
    assert analysis.category in ["shirt", "pants", "dress"]
    assert analysis.style in ["casual", "formal", "smart-casual"]
    assert len(analysis.dominant_colors) > 0
    assert 0 <= analysis.confidence <= 1

# ============================================================================
# 13. tests/test_utils.py - 유틸리티 테스트
# ============================================================================

"""
유틸리티 함수 테스트
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile

from app.utils.image_utils import ImageProcessor
from app.utils.file_utils import FileManager

def test_image_decode_encode():
    """이미지 인코딩/디코딩 테스트"""
    # 테스트 이미지 생성
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # 인코딩
    img_bytes = ImageProcessor.encode_image(img)
    
    # 디코딩
    decoded_img = ImageProcessor.decode_image(img_bytes)
    
    assert decoded_img.shape == img.shape
    assert decoded_img.dtype == img.dtype

def test_image_resize():
    """이미지 리사이즈 테스트"""
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    resized = ImageProcessor.resize_image(img, (100, 100))
    
    assert resized.shape == (100, 100, 3)

def test_extract_dominant_colors():
    """주요 색상 추출 테스트"""
    # 단색 이미지 생성
    img = np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8)  # 빨간색
    
    colors = ImageProcessor.extract_dominant_colors(img, k=1)
    
    assert len(colors) == 1
    assert len(colors[0]) == 3  # RGB

@pytest.mark.asyncio
async def test_file_manager_temp_file():
    """파일 매니저 임시 파일 테스트"""
    manager = FileManager()
    
    test_data = b"test data"
    filepath = await manager.create_temp_file(test_data)
    
    assert Path(filepath).exists()
    assert manager.temp_files
    
    # 정리
    await manager.cleanup_temp_files()
    assert not Path(filepath).exists()

def test_validate_file_extension():
    """파일 확장자 검증 테스트"""
    assert FileManager.validate_file_extension("image.jpg") == True
    assert FileManager.validate_file_extension("image.png") == True
    assert FileManager.validate_file_extension("image.txt") == False
    assert FileManager.validate_file_extension("image.exe") == False

# ============================================================================
# 14. monitoring/prometheus.yml - 모니터링 설정
# ============================================================================

global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'mycloset-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

# ============================================================================
# 15. scripts/start_dev.sh - 개발 서버 시작 스크립트
# ============================================================================

#!/bin/bash

echo "🚀 MyCloset AI 개발 서버 시작..."

# 가상환경 활성화 확인
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️ 가상환경이 활성화되지 않았습니다."
    echo "다음 명령어로 활성화하세요:"
    echo "source venv/bin/activate"
    exit 1
fi

# 환경변수 파일 확인
if [ ! -f .env ]; then
    echo "⚠️ .env 파일이 없습니다. .env.example을 복사하여 .env를 생성하세요."
    cp .env.example .env
fi

# 필요한 디렉토리 생성
mkdir -p static/uploads static/results temp logs
touch static/uploads/.gitkeep static/results/.gitkeep temp/.gitkeep logs/.gitkeep

# AI 모델 디렉토리 확인
if [ ! -d "ai_models" ]; then
    echo "⚠️ ai_models 디렉토리가 없습니다."
    echo "다음 명령어로 모델을 다운로드하세요:"
    echo "make models"
    
    read -p "지금 모델을 다운로드하시겠습니까? (y/N): " download_now
    if [[ $download_now =~ ^[Yy]$ ]]; then
        python scripts/download_models.py
    fi
fi

echo "📚 Python 의존성 확인 중..."
pip install -r requirements.txt > /dev/null 2>&1

echo "🔄 서버 시작 중..."
echo "📱 서버 주소: http://localhost:8000"
echo "📖 API 문서: http://localhost:8000/docs"
echo "🔍 헬스체크: http://localhost:8000/health"
echo ""
echo "중지하려면 Ctrl+C를 누르세요."
echo ""

# 서버 실행
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# ============================================================================
# 16. scripts/production_deploy.sh - 프로덕션 배포 스크립트
# ============================================================================

#!/bin/bash

echo "🚀 MyCloset AI 프로덕션 배포 스크립트"

# 환경 확인
if [ "$1" != "production" ]; then
    echo "⚠️ 프로덕션 배포는 다음과 같이 실행하세요:"
    echo "./scripts/production_deploy.sh production"
    exit 1
fi

echo "🔄 프로덕션 환경 설정 중..."

# 1. 환경변수 확인
if [ ! -f .env.production ]; then
    echo "❌ .env.production 파일이 필요합니다."
    exit 1
fi

# 2. Docker 이미지 빌드
echo "🏗️ Docker 이미지 빌드 중..."
docker build -t mycloset-ai-backend:latest .

# 3. 기존 컨테이너 중지
echo "🛑 기존 컨테이너 중지 중..."
docker-compose -f docker-compose.prod.yml down

# 4. 프로덕션 환경 시작
echo "🚀 프로덕션 환경 시작 중..."
docker-compose -f docker-compose.prod.yml up -d

# 5. 헬스체크
echo "🔍 서비스 상태 확인 중..."
sleep 10

curl -f http://localhost:8000/health || {
    echo "❌ 서비스 시작 실패"
    docker-compose -f docker-compose.prod.yml logs backend
    exit 1
}

echo "✅ 프로덕션 배포 완료!"
echo "📱 서비스 URL: http://your-domain.com"
echo "📊 모니터링: http://your-domain.com:9090"

# ============================================================================
# 17. README.md - 프로젝트 문서
# ============================================================================

# MyCloset AI Backend

🎽 AI 기반 가상 피팅 시스템의 백엔드 서버

## 🌟 주요 기능

- **다중 AI 모델 지원**: OOTDiffusion, ACGPN, VITON-HD
- **실시간 가상 피팅**: 고품질 이미지 생성
- **신체 분석**: MediaPipe 기반 자동 치수 측정
- **의류 분석**: 색상, 스타일, 패턴 자동 인식
- **핏 분석**: AI 기반 착용감 평가
- **RESTful API**: FastAPI 기반 고성능 API

## 🏗️ 시스템 아키텍처

```
MyCloset AI Backend
├── 🖼️ 이미지 전처리 (OpenCV, PIL)
├── 🤖 AI 모델 통합
│   ├── OOTDiffusion (Diffusion Model)
│   ├── ACGPN (GAN Model)
│   ├── VITON-HD (High Definition)
│   └── Human Parsing (Detectron2)
├── 📊 분석 엔진
│   ├── 신체 치수 측정
│   ├── 의류 카테고리 분류
│   └── 핏 스코어 계산
└── 🚀 FastAPI 서버
    ├── RESTful API
    ├── 실시간 처리
    └── 자동 문서화
```

## 🚀 빠른 시작

### 1. 프로젝트 클론

```bash
git clone https://github.com/your-username/mycloset-ai-backend.git
cd mycloset-ai-backend
```

### 2. 자동 설정 (권장)

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### 3. 수동 설정

```bash
# 가상환경 생성
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일을 편집하여 설정 조정

# 디렉토리 구조 생성
make dirs

# AI 모델 다운로드
python scripts/download_models.py
```

### 4. 서버 실행

```bash
# 개발 서버
make run
# 또는
./scripts/start_dev.sh

# Docker로 실행
make docker-run
```

## 📋 시스템 요구사항

### 최소 요구사항
- Python 3.9+
- 8GB RAM
- 10GB 저장공간

### 권장 요구사항
- Python 3.11+
- NVIDIA GPU (8GB+ VRAM)
- 16GB RAM
- 50GB 저장공간 (AI 모델 포함)

## 🔧 설정

### 환경변수 (.env)

```bash
# 기본 설정
APP_NAME=MyCloset AI Backend
DEBUG=false
PORT=8000

# AI 모델 설정
DEFAULT_MODEL=ootd
USE_GPU=true
DEVICE=cuda

# 성능 설정
MAX_UPLOAD_SIZE=52428800  # 50MB
IMAGE_SIZE=512
MAX_WORKERS=4
```

### AI 모델 선택

```python
# API 요청 시 모델 지정
{
  "model_type": "ootd",      # ootd, acgpn, viton-hd
  "category": "upper_body",   # upper_body, lower_body, dresses
  "quality": "high"          # low, medium, high
}
```

## 📖 API 문서

### 메인 엔드포인트

#### `POST /api/virtual-tryon`

가상 피팅 실행

**요청:**
```python
files = {
    'person_image': open('person.jpg', 'rb'),
    'clothing_image': open('clothing.jpg', 'rb')
}
data = {
    'height': 170.0,
    'weight': 60.0,
    'model_type': 'ootd',
    'category': 'upper_body'
}
```

**응답:**
```json
{
  "success": true,
  "session_id": "uuid-string",
  "result_image_url": "/static/results/uuid.jpg",
  "processing_time": 2.34,
  "confidence_score": 0.87,
  "recommendations": ["이 의류가 잘 어울립니다!"]
}
```

### 기타 엔드포인트

- `GET /health` - 헬스체크
- `GET /api/models/status` - AI 모델 상태
- `POST /api/preprocess` - 이미지 전처리만

## 🧪 테스트

```bash
# 전체 테스트 실행
make test

# 특정 테스트 실행
pytest tests/test_api.py -v

# 커버리지 리포트
pytest --cov=app --cov-report=html
```

## 📦 배포

### Docker 배포

```bash
# 이미지 빌드
make docker-build

# 컨테이너 실행
make docker-run

# 로그 확인
make docker-logs
```

### 프로덕션 배포

```bash
# 프로덕션 환경 배포
./scripts/production_deploy.sh production
```

## 🛠️ 개발

### 코드 스타일

```bash
# 코드 포맷팅
make format

# 린트 검사
make lint
```

### 개발 명령어

```bash
make help          # 전체 명령어 목록
make run           # 개발 서버 실행
make test          # 테스트 실행
make clean         # 캐시 파일 정리
```

## 📊 성능 최적화

### GPU 메모리 최적화

```python
# config.py에서 설정
USE_GPU = True
DEVICE = "cuda"
BATCH_SIZE = 1  # GPU 메모리에 따라 조정
```

### 캐싱 활용

```python
# Redis 캐싱 활성화
REDIS_URL = "redis://localhost:6379/0"
CACHE_TTL = 3600
```

## 🔍 모니터링

### 로그 확인

```bash
# 실시간 로그
tail -f logs/app.log

# Docker 로그
docker-compose logs -f backend
```

### 메트릭스

- Prometheus: `http://localhost:9090`
- API 메트릭스: `http://localhost:8000/metrics`

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 🙏 감사의 말

- [OOTDiffusion](https://github.com/levihsu/OOTDiffusion)
- [VITON-HD](https://github.com/shadow2496/VITON-HD)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [MediaPipe](https://mediapipe.dev/)

## 📞 지원

- 📧 Email: support@mycloset-ai.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/mycloset-ai-backend/issues)
- 📖 Documentation: [Full Documentation](https://docs.mycloset-ai.com)

---

💡 **팁**: 개발 환경에서는 `make run`으로 시작하고, 프로덕션에서는 Docker를 사용하는 것을 권장합니다.