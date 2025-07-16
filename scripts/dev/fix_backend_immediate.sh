#!/bin/bash

echo "🔧 MyCloset AI 백엔드 즉시 수정 중..."

cd backend

# 1. pydantic-settings 설치 (config 로드 문제 해결)
echo "📦 1. 누락된 패키지 설치 중..."
conda install -c conda-forge pydantic-settings python-dotenv -y
pip install python-multipart aiofiles

# 2. 간단하고 안정적인 config.py 생성
echo "⚙️ 2. 안정적인 config.py 생성 중..."

cat > app/core/config.py << 'EOF'
from typing import List
import os

class Settings:
    """간단하고 안정적인 설정 클래스"""
    
    # App 기본 설정
    APP_NAME: str = "MyCloset AI Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS 설정 (안전한 기본값)
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173", 
        "http://localhost:8080"
    ]
    
    # 파일 업로드 설정
    MAX_UPLOAD_SIZE: int = 52428800  # 50MB
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "webp", "bmp"]
    
    # AI 모델 설정
    DEFAULT_MODEL: str = "demo"
    USE_GPU: bool = False  # 안정성을 위해 False
    DEVICE: str = "cpu"
    IMAGE_SIZE: int = 512
    MAX_WORKERS: int = 2
    BATCH_SIZE: int = 1
    
    # 로깅
    LOG_LEVEL: str = "INFO"
    
    # 경로
    UPLOAD_PATH: str = "static/uploads"
    RESULT_PATH: str = "static/results"
    MODEL_PATH: str = "ai_models"
    
    def __init__(self):
        # 환경변수에서 값 읽기 (있으면)
        self.DEBUG = os.getenv("DEBUG", "true").lower() == "true"
        self.HOST = os.getenv("HOST", "0.0.0.0")
        self.PORT = int(os.getenv("PORT", 8000))
        
        # CORS_ORIGINS 환경변수 처리
        cors_env = os.getenv("CORS_ORIGINS")
        if cors_env:
            # 쉼표로 구분된 문자열을 리스트로 변환
            self.CORS_ORIGINS = [origin.strip() for origin in cors_env.split(",")]

# 전역 설정 객체
settings = Settings()
EOF

echo "✅ 안정적인 config.py 생성 완료"

# 3. 작동하는 main.py 생성
echo "🔧 3. 작동하는 main.py 생성 중..."

cat > app/main.py << 'EOF'
import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import platform

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# 안전한 설정 로드
try:
    from app.core.config import settings
    CONFIG_LOADED = True
    print("✅ 설정 로드 성공")
except Exception as e:
    print(f"⚠️ 설정 로드 실패: {e}")
    CONFIG_LOADED = False
    
    # 폴백 설정
    class FallbackSettings:
        APP_NAME = "MyCloset AI Backend"
        APP_VERSION = "1.0.0"
        DEBUG = True
        CORS_ORIGINS = ["http://localhost:3000", "http://localhost:5173"]
        MAX_UPLOAD_SIZE = 52428800
        ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp"]
    
    settings = FallbackSettings()

# PyTorch 안전 확인
TORCH_AVAILABLE = False
DEVICE_TYPE = "cpu"
try:
    import torch
    TORCH_AVAILABLE = True
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEVICE_TYPE = "mps"
        DEVICE_INFO = "Apple Silicon (MPS)"
    elif torch.cuda.is_available():
        DEVICE_TYPE = "cuda"
        DEVICE_INFO = "NVIDIA GPU"
    else:
        DEVICE_TYPE = "cpu"
        DEVICE_INFO = "CPU"
except ImportError:
    DEVICE_INFO = "PyTorch 없음"
    print("⚠️ PyTorch가 설치되지 않았습니다")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="MyCloset AI Backend",
    description="AI 가상 피팅 시스템 (안정 버전)",
    version="1.0.0"
)

# CORS 미들웨어
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip 압축
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 정적 파일 서빙
static_path = project_root / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.get("/")
async def root():
    return {
        "message": "MyCloset AI Backend이 정상 작동 중입니다! ✨",
        "version": "1.0.0",
        "environment": "Conda",
        "conda_env": os.getenv("CONDA_DEFAULT_ENV", "unknown"),
        "python_version": platform.python_version(),
        "status": "healthy",
        "docs": "/docs",
        "config_loaded": CONFIG_LOADED,
        "torch_available": TORCH_AVAILABLE,
        "device": DEVICE_TYPE,
        "features": {
            "virtual_fitting": True,
            "image_upload": True,
            "api_docs": True
        }
    }

@app.get("/api/health")
async def health_check():
    """상세 헬스체크"""
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "conda_env": os.getenv("CONDA_DEFAULT_ENV", "unknown"),
            "architecture": platform.machine()
        },
        "ai": {
            "torch_available": TORCH_AVAILABLE,
            "device_type": DEVICE_TYPE,
            "device_info": DEVICE_INFO
        },
        "config": {
            "loaded": CONFIG_LOADED,
            "cors_origins": len(settings.CORS_ORIGINS) if hasattr(settings, 'CORS_ORIGINS') else 0
        },
        "services": {
            "virtual_fitting": "available",
            "image_processing": "available",
            "file_upload": "available"
        }
    }

@app.post("/api/virtual-tryon")
async def virtual_tryon_endpoint(
    person_image: UploadFile = File(..., description="사용자 사진"),
    clothing_image: UploadFile = File(..., description="의류 사진"),
    height: float = Form(..., description="신장 (cm)"),
    weight: float = Form(..., description="체중 (kg)")
):
    """가상 피팅 API"""
    
    # 파일 검증
    if not person_image.content_type.startswith("image/"):
        raise HTTPException(400, "사용자 이미지 파일이 아닙니다.")
    
    if not clothing_image.content_type.startswith("image/"):
        raise HTTPException(400, "의류 이미지 파일이 아닙니다.")
    
    # 파일 크기 체크
    max_size = getattr(settings, 'MAX_UPLOAD_SIZE', 52428800)  # 50MB
    
    try:
        # 간단한 처리 시뮬레이션
        import time
        import uuid
        
        session_id = str(uuid.uuid4())
        start_time = time.time()
        
        # 처리 시뮬레이션 (1-3초)
        await asyncio.sleep(2)
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "가상 피팅이 성공적으로 완료되었습니다!",
            "data": {
                "person_image": {
                    "filename": person_image.filename,
                    "content_type": person_image.content_type,
                    "size": f"{person_image.size if hasattr(person_image, 'size') else 'unknown'} bytes"
                },
                "clothing_image": {
                    "filename": clothing_image.filename,
                    "content_type": clothing_image.content_type,
                    "size": f"{clothing_image.size if hasattr(clothing_image, 'size') else 'unknown'} bytes"
                },
                "measurements": {
                    "height": f"{height}cm",
                    "weight": f"{weight}kg",
                    "bmi": round(weight / ((height/100) ** 2), 1)
                }
            },
            "processing": {
                "time_seconds": round(processing_time, 2),
                "device": DEVICE_TYPE,
                "torch_available": TORCH_AVAILABLE
            },
            "result": {
                "confidence": 0.92,
                "fit_score": 0.88,
                "recommendation": "좋은 핏입니다!" if 18.5 <= weight / ((height/100) ** 2) <= 25 else "사이즈를 확인해보세요."
            },
            "note": "현재 데모 모드로 실행 중입니다. 실제 AI 모델은 곧 통합될 예정입니다."
        }
        
    except Exception as e:
        raise HTTPException(500, f"처리 중 오류 발생: {str(e)}")

@app.get("/api/models")
async def get_available_models():
    """사용 가능한 AI 모델 목록"""
    return {
        "models": [
            {
                "id": "demo",
                "name": "데모 모드",
                "status": "available",
                "device": DEVICE_TYPE,
                "description": "빠른 테스트용 가상 피팅",
                "features": ["빠른 처리", "기본 합성", "테스트 모드"]
            },
            {
                "id": "ootd_diffusion",
                "name": "OOT-Diffusion",
                "status": "preparing",
                "device": DEVICE_TYPE,
                "description": "고품질 Diffusion 기반 가상 피팅",
                "features": ["고해상도", "자연스러운 합성", "정확한 피팅"]
            }
        ],
        "default": "demo",
        "environment": {
            "torch_available": TORCH_AVAILABLE,
            "device": DEVICE_TYPE,
            "conda_env": os.getenv("CONDA_DEFAULT_ENV")
        },
        "status": "모든 기본 기능이 정상 작동합니다"
    }

@app.get("/api/status")
async def get_system_status():
    """시스템 상태 확인"""
    return {
        "backend": "running",
        "database": "not_required",
        "ai_models": "demo_mode",
        "file_storage": "available",
        "config": "loaded" if CONFIG_LOADED else "fallback",
        "torch": "available" if TORCH_AVAILABLE else "not_installed",
        "timestamp": datetime.now().isoformat()
    }

# 추가로 필요한 import
import asyncio

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 MyCloset AI Backend 시작됨")
    logger.info(f"🐍 Conda 환경: {os.getenv('CONDA_DEFAULT_ENV', 'unknown')}")
    logger.info(f"🔧 설정 로드: {'성공' if CONFIG_LOADED else '폴백 사용'}")
    logger.info(f"🔥 PyTorch: {'사용 가능' if TORCH_AVAILABLE else '없음'}")
    logger.info(f"💻 디바이스: {DEVICE_TYPE}")
    logger.info("📁 필수 디렉토리 확인 중...")
    
    # 필수 디렉토리 생성
    directories = ["static/uploads", "static/results", "logs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("✅ 모든 시스템이 준비되었습니다!")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 MyCloset AI Backend 종료됨")

if __name__ == "__main__":
    import uvicorn
    print("🚀 서버를 시작합니다...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
EOF

echo "✅ 작동하는 main.py 생성 완료"

# 4. 필수 디렉토리 생성
echo "📁 4. 필수 디렉토리 생성 중..."
mkdir -p static/{uploads,results}
mkdir -p logs
touch static/uploads/.gitkeep
touch static/results/.gitkeep
touch logs/.gitkeep

# 5. 간단한 실행 스크립트 생성
echo "📜 5. 간단한 실행 스크립트 생성 중..."

cat > run_fixed.sh << 'EOF'
#!/bin/bash

echo "🚀 MyCloset AI Backend - 수정된 버전 실행"
echo "=========================================="

# Conda 환경 확인
if [[ "$CONDA_DEFAULT_ENV" == "" ]]; then
    echo "❌ Conda 환경이 활성화되지 않았습니다."
    echo "conda activate mycloset"
    exit 1
fi

echo "✅ Conda 환경: $CONDA_DEFAULT_ENV"

# 패키지 확인
echo "📦 필수 패키지 확인 중..."

# FastAPI 확인
python -c "import fastapi; print(f'✅ FastAPI: {fastapi.__version__}')" 2>/dev/null || {
    echo "❌ FastAPI가 없습니다. 설치: conda install fastapi uvicorn -y"
    exit 1
}

# 서버 시작
echo ""
echo "🌐 서버 시작 중..."
echo "📱 접속 주소: http://localhost:8000"
echo "📚 API 문서: http://localhost:8000/docs"
echo "🔧 헬스체크: http://localhost:8000/api/health"
echo "🧪 가상 피팅 테스트: http://localhost:8000/api/virtual-tryon"
echo ""
echo "⏹️ 종료하려면 Ctrl+C를 누르세요"
echo ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
EOF

chmod +x run_fixed.sh

# 6. 현재 환경 정보 출력
echo "📊 6. 현재 환경 정보"
echo "===================="

echo "🐍 Conda 환경: ${CONDA_DEFAULT_ENV:-'없음'}"
echo "🐍 Python: $(python --version 2>/dev/null || echo '확인불가')"

# 패키지 확인
echo "📦 설치된 패키지:"
python -c "import fastapi; print(f'  ✅ FastAPI: {fastapi.__version__}')" 2>/dev/null || echo "  ❌ FastAPI 없음"
python -c "import uvicorn; print(f'  ✅ Uvicorn: {uvicorn.__version__}')" 2>/dev/null || echo "  ❌ Uvicorn 없음"
python -c "import pydantic; print(f'  ✅ Pydantic: {pydantic.__version__}')" 2>/dev/null || echo "  ❌ Pydantic 없음"
python -c "import torch; print(f'  ✅ PyTorch: {torch.__version__}')" 2>/dev/null || echo "  ℹ️ PyTorch 없음 (선택사항)"

echo ""
echo "🎉 수정 완료!"
echo ""
echo "🚀 실행 방법:"
echo "   ./run_fixed.sh"
echo ""
echo "🔧 문제가 있다면:"
echo "   conda install fastapi uvicorn python-multipart -y"
echo "   pip install pydantic-settings python-dotenv"
echo ""
echo "📱 실행 후 접속: http://localhost:8000"   