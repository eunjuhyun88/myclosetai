#!/bin/bash

echo "🚀 MyCloset AI 완전 자동 설정 시작..."
echo "=================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

PROJECT_ROOT=$(pwd)
log_info "프로젝트 루트: $PROJECT_ROOT"

# 1. 백엔드 설정
log_info "백엔드 설정 중..."
cd backend

# Python 가상환경 생성
if [ ! -d "venv" ]; then
    log_info "Python 가상환경 생성 중..."
    python3 -m venv venv
    log_success "가상환경 생성 완료"
fi

# 가상환경 활성화
source venv/bin/activate

# 기본 의존성 설치
log_info "기본 의존성 설치 중..."
pip install --upgrade pip
pip install fastapi uvicorn python-multipart pydantic pydantic-settings
pip install aiofiles python-dotenv requests

# __init__.py 파일들 생성
touch app/__init__.py
touch app/api/__init__.py
touch app/core/__init__.py
touch app/models/__init__.py
touch app/services/__init__.py
touch app/utils/__init__.py

# app/main.py 수정 (경로 문제 해결)
cat > app/main.py << 'MAINEOF'
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

try:
    from app.core.config import settings
    from app.api.routes import router as api_router
except ImportError as e:
    print(f"임포트 오류: {e}")
    print("기본 설정으로 실행합니다.")
    
    class Settings:
        APP_NAME = "MyCloset AI Backend"
        APP_VERSION = "1.0.0"
        DEBUG = True
        HOST = "0.0.0.0"
        PORT = 8000
        CORS_ORIGINS = ["http://localhost:3000", "http://localhost:5173"\]
    
    settings = Settings()

# FastAPI 앱 생성
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered virtual try-on platform"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "MyCloset AI Backend is running!", 
        "version": settings.APP_VERSION,
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "debug": settings.DEBUG
    }

@app.post("/api/virtual-tryon")
async def virtual_tryon_test():
    return {
        "success": True,
        "message": "테스트 엔드포인트가 정상 작동 중입니다!",
        "fitted_image": "",
        "confidence": 0.85,
        "fit_score": 0.88,
        "recommendations": ["서버가 정상 작동 중입니다!"]
    }

# 라우터 등록 시도
try:
    app.include_router(api_router, prefix="/api")
except:
    log_warning("API 라우터 로드 실패 - 기본 엔드포인트 사용")

if __name__ == "__main__":
    print(f"🚀 서버 시작 중...")
    print(f"📍 주소: http://localhost:{settings.PORT}")
    print(f"📖 API 문서: http://localhost:{settings.PORT}/docs")
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
MAINEOF

# 간단한 설정 파일 생성
mkdir -p app/core
cat > app/core/config.py << 'CONFIGEOF'
from typing import List

class Settings:
    APP_NAME: str = "MyCloset AI Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"\]

settings = Settings()
CONFIGEOF

# 간단한 라우터 생성
mkdir -p app/api
cat > app/api/routes.py << 'ROUTESEOF'
from fastapi import APIRouter

router = APIRouter()

@router.get("/test")
async def test_endpoint():
    return {"message": "API 테스트 성공!", "status": "working"}

@router.get("/models/status")
async def models_status():
    return {
        "models": {"ootd": False, "viton": False},
        "status": "development"
    }
ROUTESEOF

# .env 파일 생성
cat > .env << 'ENVEOF'
APP_NAME=MyCloset AI Backend
DEBUG=true
PORT=8000
ENVEOF

cd ..

# Makefile 생성
cat > Makefile << 'MAKEEOF'
.PHONY: run-backend test clean

run-backend:
	cd backend && source venv/bin/activate && python app/main.py

test:
	curl -s http://localhost:8000/health || echo "서버가 실행되지 않았습니다"

clean:
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
MAKEEOF

log_success "설정 완료!"
echo ""
echo "🚀 서버 실행 방법:"
echo "make run-backend"
echo ""
echo "또는:"
echo "cd backend && source venv/bin/activate && python app/main.py"
