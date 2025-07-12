# MyCloset AI 완전 설정 및 문제 해결 스크립트

## 🚀 1. 전체 자동 설정 스크립트

### `scripts/complete_setup.sh` - 완전 자동화 설정

```bash
#!/bin/bash

echo "🚀 MyCloset AI 완전 자동 설정 시작..."
echo "=================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수들
log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

# 에러 처리
set -e
trap 'log_error "스크립트 실행 중 오류가 발생했습니다."' ERR

# 프로젝트 루트 확인
PROJECT_ROOT=$(pwd)
log_info "프로젝트 루트: $PROJECT_ROOT"

# 1. 시스템 요구사항 확인
log_info "시스템 요구사항 확인 중..."

# Python 버전 확인
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ $(echo "$PYTHON_VERSION 3.9" | tr ' ' '\n' | sort -V | head -n1) == "3.9" ]]; then
        log_success "Python $PYTHON_VERSION 확인됨"
    else
        log_error "Python 3.9+ 필요. 현재: $PYTHON_VERSION"
        exit 1
    fi
else
    log_error "Python3가 설치되지 않았습니다."
    exit 1
fi

# Node.js 확인
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
    if [[ $NODE_VERSION -ge 18 ]]; then
        log_success "Node.js $(node --version) 확인됨"
    else
        log_warning "Node.js 18+ 권장. 현재: $(node --version)"
    fi
else
    log_error "Node.js가 설치되지 않았습니다."
    exit 1
fi

# Git 확인
if command -v git &> /dev/null; then
    log_success "Git 확인됨"
else
    log_error "Git이 설치되지 않았습니다."
    exit 1
fi

# 2. 프로젝트 구조 생성
log_info "프로젝트 구조 생성 중..."

# 기본 디렉토리 생성
mkdir -p {backend,frontend,scripts,docker}
mkdir -p backend/{app,ai_models,static,tests,logs}
mkdir -p backend/app/{api,core,models,services,utils}
mkdir -p backend/static/{uploads,results}
mkdir -p backend/ai_models/{checkpoints,OOTDiffusion,VITON-HD}
mkdir -p frontend/src/{components,pages,hooks,types,utils}
mkdir -p frontend/src/components/{ui,features}
mkdir -p frontend/public

# .gitkeep 파일 생성
find backend/static -type d -exec touch {}/.gitkeep \;
find backend/ai_models -type d -exec touch {}/.gitkeep \;

log_success "프로젝트 구조 생성 완료"

# 3. 백엔드 설정
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

# requirements.txt 생성 (호환성 개선)
cat > requirements.txt << 'EOF'
# FastAPI 및 웹 서버
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6

# 이미지 처리
Pillow>=10.0.0
opencv-python>=4.8.0
scikit-image>=0.20.0

# 데이터 검증
pydantic>=2.5.0
pydantic-settings>=2.1.0

# 파일 처리
aiofiles>=23.0.0
python-dotenv>=1.0.0

# 유틸리티
tqdm>=4.65.0
requests>=2.31.0
structlog>=23.0.0

# Computer Vision (선택적)
# mediapipe>=0.10.0

# NumPy (호환성 중요)
numpy>=1.24.0,<2.0.0
scipy>=1.10.0
EOF

# 기본 의존성 설치
log_info "Python 의존성 설치 중..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# PyTorch 설치 (시스템에 맞게)
log_info "PyTorch 설치 중..."
if command -v nvidia-smi &> /dev/null; then
    log_info "CUDA 감지됨 - GPU 버전 PyTorch 설치"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    log_info "CPU 버전 PyTorch 설치"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# MediaPipe 별도 설치 (문제 발생 시 스킵)
log_info "MediaPipe 설치 시도 중..."
pip install mediapipe || log_warning "MediaPipe 설치 실패 - 나중에 수동 설치 필요"

log_success "백엔드 의존성 설치 완료"

# 4. 백엔드 파일들 생성
log_info "백엔드 핵심 파일들 생성 중..."

# __init__.py 파일들 생성
touch app/__init__.py
touch app/{api,core,models,services,utils}/__init__.py

# app/main.py 수정 (경로 문제 해결)
cat > app/main.py << 'MAINEOF'
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from app.core.config import settings
from app.api.routes import router as api_router

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

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

# 라우터 등록
app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "MyCloset AI Backend is running!", "version": settings.APP_VERSION}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "debug": settings.DEBUG
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
MAINEOF

# app/core/config.py
cat > app/core/config.py << 'CONFIGEOF'
from pydantic_settings import BaseSettings
from typing import List
import os
from pathlib import Path

class Settings(BaseSettings):
    # App
    APP_NAME: str = "MyCloset AI Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "webp"]
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    UPLOAD_DIR: Path = PROJECT_ROOT / "static" / "uploads"
    RESULTS_DIR: Path = PROJECT_ROOT / "static" / "results"
    AI_MODELS_DIR: Path = PROJECT_ROOT / "ai_models"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# 디렉토리 생성
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
settings.AI_MODELS_DIR.mkdir(parents=True, exist_ok=True)
CONFIGEOF

# app/api/routes.py
cat > app/api/routes.py << 'ROUTESEOF'
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uuid
import time
from typing import Optional

router = APIRouter()

@router.post("/virtual-tryon")
async def virtual_tryon(
    person_image: UploadFile = File(...),
    clothing_image: UploadFile = File(...),
    height: float = Form(170),
    weight: float = Form(65),
    model_type: str = Form("ootd")
):
    """Virtual try-on endpoint"""
    session_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # 파일 검증
        if not person_image.filename or not clothing_image.filename:
            raise HTTPException(400, "파일이 업로드되지 않았습니다.")
        
        # 더미 처리 (실제 AI 처리 전까지)
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "session_id": session_id,
            "fitted_image": "",  # Base64 이미지 (나중에 구현)
            "processing_time": round(processing_time, 2),
            "confidence": 0.87,
            "fit_score": 0.88,
            "recommendations": [
                "테스트 서버가 정상 작동 중입니다!",
                f"모델: {model_type}",
                f"신장: {height}cm, 체중: {weight}kg"
            ]
        }
    except Exception as e:
        raise HTTPException(500, f"처리 중 오류 발생: {str(e)}")

@router.get("/models/status")
async def get_models_status():
    """AI 모델 상태 확인"""
    return {
        "models": {
            "ootd": {"loaded": False, "ready": False},
            "viton-hd": {"loaded": False, "ready": False}
        },
        "gpu_available": False,
        "status": "development"
    }

@router.get("/test")
async def test_endpoint():
    """테스트 엔드포인트"""
    return {"message": "API가 정상 작동 중입니다!", "timestamp": time.time()}
ROUTESEOF

# .env 파일 생성
cat > .env << 'ENVEOF'
# Application Settings
APP_NAME=MyCloset AI Backend
APP_VERSION=1.0.0
DEBUG=true
HOST=0.0.0.0
PORT=8000

# CORS Settings
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# File Upload Settings
MAX_UPLOAD_SIZE=52428800
ALLOWED_EXTENSIONS=jpg,jpeg,png,webp
ENVEOF

# 서버 실행 스크립트 생성
cat > run_server.py << 'RUNEOF'
#!/usr/bin/env python3
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import uvicorn
from app.main import app

if __name__ == "__main__":
    print("🚀 MyCloset AI 백엔드 서버 시작...")
    print("📍 서버 주소: http://localhost:8000")
    print("📖 API 문서: http://localhost:8000/docs")
    print("🔍 헬스체크: http://localhost:8000/health")
    print("\n중지하려면 Ctrl+C를 누르세요.\n")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        access_log=True
    )
RUNEOF

chmod +x run_server.py

cd ..

log_success "백엔드 설정 완료"

# 5. 프론트엔드 설정
log_info "프론트엔드 설정 중..."

cd frontend

# package.json 확인 후 설치
if [ -f "package.json" ]; then
    log_info "Node.js 의존성 설치 중..."
    npm install
    log_success "프론트엔드 의존성 설치 완료"
else
    log_warning "package.json이 없습니다. Vite 프로젝트를 생성해야 합니다."
fi

cd ..

# 6. 루트 레벨 설정 파일들
log_info "루트 설정 파일들 생성 중..."

# Makefile
cat > Makefile << 'MAKEEOF'
.PHONY: help setup install run-backend run-frontend dev clean test

help:
	@echo "MyCloset AI 개발 명령어:"
	@echo "  setup        - 초기 환경 설정"
	@echo "  install      - 의존성 설치"
	@echo "  run-backend  - 백엔드 서버 실행"
	@echo "  run-frontend - 프론트엔드 서버 실행"
	@echo "  dev          - 개발 모드 (백엔드만)"
	@echo "  test         - 연결 테스트"
	@echo "  clean        - 정리"

setup:
	@echo "🔧 초기 설정 중..."
	cd backend && python3 -m venv venv
	@echo "✅ 설정 완료"

install:
	@echo "📦 의존성 설치 중..."
	cd backend && source venv/bin/activate && pip install -r requirements.txt
	cd frontend && npm install
	@echo "✅ 의존성 설치 완료"

run-backend:
	@echo "🚀 백엔드 서버 시작..."
	cd backend && source venv/bin/activate && python run_server.py

run-frontend:
	@echo "🎨 프론트엔드 서버 시작..."
	cd frontend && npm run dev

dev: 
	@echo "🚀 개발 모드 시작 (백엔드)"
	@echo "프론트엔드는 별도 터미널에서 'make run-frontend' 실행"
	$(MAKE) run-backend

test:
	@echo "🔍 서버 연결 테스트..."
	@sleep 2
	@curl -s http://localhost:8000/health || echo "❌ 백엔드 서버가 실행되지 않았습니다."
	@curl -s http://localhost:8000/api/test || echo "❌ API가 응답하지 않습니다."

clean:
	@echo "🧹 정리 중..."
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".DS_Store" -delete 2>/dev/null || true
	@echo "✅ 정리 완료"
MAKEEOF

# README.md 업데이트
cat > README.md << 'READMEEOF'
# MyCloset AI - 가상 피팅 플랫폼

🎽 AI 기반 가상 피팅 시스템

## 🚀 빠른 시작

### 1. 전체 자동 설정
```bash
bash scripts/complete_setup.sh
```

### 2. 수동 설정
```bash
# 백엔드 설정
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 프론트엔드 설정
cd frontend
npm install
```

### 3. 서버 실행

**방법 1: Makefile 사용 (권장)**
```bash
make run-backend    # 백엔드 실행
make run-frontend   # 프론트엔드 실행 (새 터미널)
```

**방법 2: 직접 실행**
```bash
# 백엔드
cd backend && source venv/bin/activate && python run_server.py

# 프론트엔드 (새 터미널)
cd frontend && npm run dev
```

## 📱 접속 주소

- **프론트엔드**: http://localhost:5173
- **백엔드 API**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs
- **헬스체크**: http://localhost:8000/health

## 🔧 개발 명령어

```bash
make help           # 전체 명령어 보기
make setup          # 초기 설정
make install        # 의존성 설치
make dev            # 개발 모드
make test           # 연결 테스트
make clean          # 정리
```

## 📁 프로젝트 구조

```
mycloset-ai/
├── frontend/           # React + TypeScript
├── backend/           # FastAPI + Python
├── scripts/           # 설정 스크립트
└── ai_models/         # AI 모델들
```

## 🤖 AI 모델

- **OOTDiffusion**: 고품질 가상 피팅
- **VITON-HD**: 고해상도 피팅

## 🆘 문제 해결

실행 중 문제가 발생하면:
```bash
bash scripts/fix_common_issues.sh
```
READMEEOF

log_success "루트 설정 파일들 생성 완료"

# 7. 최종 확인
log_info "설정 완료 확인 중..."

cd backend
if source venv/bin/activate && python -c "import fastapi; print('FastAPI 설치 확인됨')"; then
    log_success "백엔드 설정 정상"
else
    log_error "백엔드 설정에 문제가 있습니다."
fi
cd ..

# 완료 메시지
echo ""
echo "🎉 MyCloset AI 설정 완료!"
echo "=========================="
log_success "프로젝트가 성공적으로 설정되었습니다!"
echo ""
echo "📋 다음 단계:"
echo "1. 백엔드 실행: make run-backend"
echo "2. 프론트엔드 실행: make run-frontend (새 터미널)"
echo "3. 브라우저에서 http://localhost:5173 접속"
echo ""
echo "🔧 문제 발생 시:"
echo "bash scripts/fix_common_issues.sh"
```

## 🛠️ 2. 문제 해결 스크립트

### `scripts/fix_common_issues.sh` - 일반적인 문제 해결

```bash
#!/bin/bash

echo "🛠️ MyCloset AI 문제 해결 스크립트"
echo "================================"

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

# 1. Python 모듈 경로 문제 해결
fix_python_path() {
    log_info "Python 모듈 경로 문제 해결 중..."
    
    cd backend
    
    # PYTHONPATH 설정 스크립트 생성
    cat > set_python_path.sh << 'EOF'
#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "PYTHONPATH 설정됨: $PYTHONPATH"
EOF
    chmod +x set_python_path.sh
    
    # __init__.py 파일들 확인 및 생성
    touch app/__init__.py
    touch app/{api,core,models,services,utils}/__init__.py
    
    # app/main.py 경로 수정
    cat > app/main.py << 'MAINEOF'
import sys
import os
from pathlib import Path

# 현재 스크립트의 절대 경로 가져오기
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

# Python 경로에 프로젝트 루트 추가
sys.path.insert(0, str(project_root))

# 환경 변수로도 설정
os.environ['PYTHONPATH'] = str(project_root)

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    
    from app.core.config import settings
    from app.api.routes import router as api_router
    
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
    
    # 정적 파일 서빙
    static_dir = project_root / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # 라우터 등록
    app.include_router(api_router, prefix="/api")
    
    @app.get("/")
    async def root():
        return {
            "message": "MyCloset AI Backend is running!",
            "version": settings.APP_VERSION,
            "python_path": sys.path[:3]
        }
    
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "version": settings.APP_VERSION,
            "debug": settings.DEBUG,
            "project_root": str(project_root)
        }
    
    if __name__ == "__main__":
        print(f"🚀 서버 시작 - 프로젝트 루트: {project_root}")
        uvicorn.run(
            "app.main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=settings.DEBUG
        )

except ImportError as e:
    print(f"❌ 모듈 임포트 오류: {e}")
    print(f"프로젝트 루트: {project_root}")
    print(f"Python 경로: {sys.path}")
    sys.exit(1)
MAINEOF
    
    # 직접 실행 스크립트 생성
    cat > run_app.py << 'RUNEOF'
#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# 프로젝트 루트 설정
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))
os.chdir(project_root)

print(f"📁 작업 디렉토리: {os.getcwd()}")
print(f"🐍 Python 경로에 추가됨: {project_root}")

# 앱 실행
exec(open('app/main.py').read())
RUNEOF
    chmod +x run_app.py
    
    cd ..
    log_success "Python 모듈 경로 문제 해결 완료"
}

# 2. 의존성 문제 해결
fix_dependencies() {
    log_info "의존성 문제 해결 중..."
    
    cd backend
    source venv/bin/activate
    
    # pip 업그레이드
    pip install --upgrade pip setuptools wheel
    
    # 문제가 되는 패키지 개별 설치
    log_info "기본 패키지 설치 중..."
    pip install fastapi uvicorn python-multipart
    
    log_info "이미지 처리 패키지 설치 중..."
    pip install Pillow opencv-python
    
    log_info "데이터 검증 패키지 설치 중..."
    pip install pydantic pydantic-settings
    
    log_info "파일 처리 패키지 설치 중..."
    pip install aiofiles python-dotenv
    
    log_info "유틸리티 패키지 설치 중..."
    pip install tqdm requests structlog
    
    # NumPy 호환성 문제 해결
    log_info "NumPy 호환성 문제 해결 중..."
    pip install "numpy>=1.24.0,<2.0.0"
    
    # MediaPipe 별도 처리
    log_info "MediaPipe 설치 시도 중..."
    pip install mediapipe || log_warning "MediaPipe 설치 실패 - 선택적 패키지입니다"
    
    cd ..
    log_success "의존성 문제 해결 완료"
}

# 3. 포트 충돌 해결
fix_port_conflicts() {
    log_info "포트 충돌 확인 중..."
    
    # 8000번 포트 사용 프로세스 확인
    if lsof -i :8000 > /dev/null 2>&1; then
        log_warning "8000번 포트가 사용 중입니다."
        echo "사용 중인 프로세스:"
        lsof -i :8000
        
        read -p "이 프로세스를 종료하시겠습니까? (y/N): " kill_process
        if [[ $kill_process =~ ^[Yy]$ ]]; then
            pkill -f "uvicorn\|python.*8000" || true
            log_success "포트 8000번 정리 완료"
        else
            log_info "다른 포트를 사용하려면 backend/.env에서 PORT를 변경하세요"
        fi
    else
        log_success "포트 8000번 사용 가능"
    fi
}

# 4. 권한 문제 해결
fix_permissions() {
    log_info "파일 권한 문제 해결 중..."
    
    # 실행 권한 부여
    find scripts -name "*.sh" -exec chmod +x {} \;
    find backend -name "*.py" -exec chmod +r {} \;
    
    # 디렉토리 권한 설정
    chmod -R 755 backend/static
    chmod -R 755 backend/ai_models
    
    log_success "파일 권한 문제 해결 완료"
}

# 5. 환경 설정 문제 해결
fix_environment() {
    log_info "환경 설정 문제 해결 중..."
    
    cd backend
    
    # .env 파일 생성/수정
    cat > .env << 'ENVEOF'
# Application Settings
APP_NAME=MyCloset AI Backend
APP_VERSION=1.0.0
DEBUG=true
HOST=0.0.0.0
PORT=8000

# CORS Settings
CORS_ORIGINS=http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173

# File Upload Settings
MAX_UPLOAD_SIZE=52428800
ALLOWED_EXTENSIONS=jpg,jpeg,png,webp

# Paths
PYTHONPATH=.
ENVEOF
    
    cd ..
    log_success "환경 설정 문제 해결 완료"
}

# 6. 서버 연결 테스트
test_server() {
    log_info "서버 연결 테스트 중..."
    
    cd backend
    source venv/bin/activate
    
    # 백그라운드에서 서버 시작
    python run_app.py &
    SERVER_PID=$!
    
    # 서버 시작 대기
    sleep 5
    
    # 헬스체크 테스트
    if curl -s http://localhost:8000/health > /dev/null; then
        log_success "서버가 정상적으로 실행되고 있습니다!"
        curl -s http://localhost:8000/health | python -m json.tool
    else
        log_error "서버 연결 실패"
    fi
    
    # 서버 종료
    kill $SERVER_PID 2>/dev/null || true
    
    cd ..
}

# 메인 실행
main() {
    echo "🔧 어떤 문제를 해결하시겠습니까?"
    echo "1. 모든 문제 자동 해결"
    echo "2. Python 모듈 경로 문제"
    echo "3. 의존성 문제"
    echo "4. 포트 충돌 문제"
    echo "5. 권한 문제"
    echo "6. 환경 설정 문제"
    echo "7. 서버 연결 테스트"
    echo ""
    
    read -p "선택하세요 (1-7): " choice
    
    case $choice in
        1)
            log_info "모든 문제 자동 해결 시작..."
            fix_python_path
            fix_dependencies
            fix_port_conflicts
            fix_permissions
            fix_environment
            test_server
            ;;
        2) fix_python_path ;;
        3) fix_dependencies ;;
        4) fix_port_conflicts ;;
        5) fix_permissions ;;
        6) fix_environment ;;
        7) test_server ;;
        *)
            log_error "잘못된 선택입니다."
            exit 1
            ;;
    esac
    
    echo ""
    log_success "문제 해결 완료!"
    echo ""
    echo "📋 서버 실행 방법:"
    echo "cd backend && source venv/bin/activate && python run_app.py"
    echo ""
    echo "또는:"
    echo "make run-backend"
}

# 스크립트 실행
main "$@"
```

## 🚀 3. 사용 방법

### 전체 자동 설정 실행
```bash
# 스크립트에 실행 권한 부여
chmod +x scripts/complete_setup.sh
chmod +x scripts/fix_common_issues.sh

# 전체 자동 설정 실행
bash scripts/complete_setup.sh
```

### 문제 해결 스크립트 실행
```bash
# 일반적인 문제들 해결
bash scripts/fix_common_issues.sh
```

### 개별 실행 방법
```bash
# 백엔드만 실행 (경로 문제 해결됨)
cd backend
source venv/bin/activate
python run_app.py

# 또는 Makefile 사용
make run-backend
```

이 스크립트들을 실행하면 모든 설정과 일반적인 문제들이 자동으로 해결됩니다!