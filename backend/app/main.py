# backend/app/main.py
"""
MyCloset AI Backend - FastAPI 메인 애플리케이션
M3 Max 최적화 가상 피팅 시스템
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# 로깅 설정
from app.core.logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

# GPU 설정 임포트
from app.core.gpu_config import gpu_config, startup_gpu_check

# API 라우터 임포트
from app.api.health import router as health_router
from app.api.virtual_tryon import router as virtual_tryon_router
from app.api.models import router as models_router

# 설정 임포트
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    
    # 시작 시 실행
    logger.info("🚀 MyCloset AI Backend 시작...")
    
    # 필수 디렉토리 생성
    directories = [
        settings.UPLOAD_DIR,
        settings.RESULTS_DIR,
        settings.LOGS_DIR,
        PROJECT_ROOT / "ai_models" / "checkpoints",
        PROJECT_ROOT / "ai_models" / "temp",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # GPU 설정 및 테스트
    startup_gpu_check()
    
    # AI 모델 로드 (백그라운드에서)
    try:
        from app.services.model_manager import model_manager
        # 사용 가능한 모델 목록 업데이트
        model_manager.available_models = model_manager.get_available_models()
        logger.info(f"✅ {sum(model_manager.available_models.values())}/{len(model_manager.available_models)} 모델 감지됨")
        
        # 선택적 모델 로드 (백그라운드에서)
        await model_manager.load_models()
        logger.info("✅ AI 모델 매니저 초기화 완료")
    except Exception as e:
        logger.warning(f"⚠️ AI 모델 로드 실패: {e}")
        logger.info("ℹ️ 모델을 먼저 다운로드하세요: python scripts/download_ai_models.py")

    logger.info("✅ 애플리케이션 초기화 완료")
    
    yield
    
    # 종료 시 실행
    logger.info("🛑 MyCloset AI Backend 종료...")
    
    # 메모리 정리
    gpu_config.optimize_memory()
    
    # 모델 언로드
    try:
        await model_manager.unload_models()
        logger.info("✅ 모델 언로드 완료")
    except:
        pass
    
    logger.info("👋 애플리케이션 종료 완료")


# FastAPI 애플리케이션 생성
app = FastAPI(
    title="MyCloset AI Backend",
    description="M3 Max 최적화 가상 피팅 시스템 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Gzip 압축 미들웨어
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """요청 로깅 미들웨어"""
    
    import time
    
    # 요청 시작 시간
    start_time = time.time()
    
    # 요청 정보 로깅
    logger.info(f"📥 {request.method} {request.url.path}")
    
    try:
        # 요청 처리
        response = await call_next(request)
        
        # 처리 시간 계산
        process_time = time.time() - start_time
        
        # 응답 로깅
        logger.info(f"📤 {request.method} {request.url.path} - {response.status_code} ({process_time:.3f}s)")
        
        # 응답 헤더에 처리 시간 추가
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        # 에러 로깅
        process_time = time.time() - start_time
        logger.error(f"❌ {request.method} {request.url.path} - ERROR ({process_time:.3f}s): {e}")
        raise


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP 예외 처리"""
    
    logger.warning(f"⚠️ HTTP {exc.status_code}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": request.url.path
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """일반 예외 처리"""
    
    logger.error(f"❌ 예상치 못한 오류 {request.url.path}: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error" if not settings.DEBUG else str(exc),
            "status_code": 500,
            "path": request.url.path
        }
    )


# API 라우터 등록
app.include_router(health_router, prefix="/health", tags=["Health"])
app.include_router(virtual_tryon_router, prefix="/api/virtual-tryon", tags=["Virtual Try-On"])
app.include_router(models_router, prefix="/api/models", tags=["Models"])


@app.get("/")
async def root():
    """루트 엔드포인트"""
    
    return {
        "service": "MyCloset AI Backend",
        "version": "1.0.0",
        "status": "running",
        "gpu_device": gpu_config.device,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/info")
async def get_system_info():
    """시스템 정보 조회"""
    
    from app.core.gpu_config import DEVICE_INFO, MODEL_CONFIG
    import psutil
    import torch
    
    # 메모리 정보
    memory = psutil.virtual_memory()
    
    # GPU 정보
    gpu_info = {
        "device": gpu_config.device,
        "available": True,
        "details": DEVICE_INFO
    }
    
    if gpu_config.device == "mps":
        gpu_info["mps_available"] = torch.backends.mps.is_available()
        gpu_info["optimization"] = "Apple M3 Max Metal"
    elif gpu_config.device == "cuda":
        gpu_info["cuda_version"] = torch.version.cuda
        gpu_info["devices_count"] = torch.cuda.device_count()
    
    # 모델 상태
    model_status = {}
    try:
        from app.services.model_manager import model_manager
        model_status = {
            "loaded_models": list(model_manager.loaded_models.keys()),
            "total_models": len(model_manager.available_models),
        }
    except:
        model_status = {"status": "model_manager_not_ready"}
    
    return {
        "system": {
            "platform": DEVICE_INFO["platform"],
            "machine": DEVICE_INFO["machine"],
            "python_version": DEVICE_INFO["python_version"],
            "pytorch_version": DEVICE_INFO["pytorch_version"],
        },
        "memory": {
            "total_gb": round(memory.total / (1024**3), 1),
            "available_gb": round(memory.available / (1024**3), 1),
            "used_percent": memory.percent,
        },
        "gpu": gpu_info,
        "models": model_status,
        "config": {
            "debug": settings.DEBUG,
            "max_upload_size_mb": settings.MAX_UPLOAD_SIZE // (1024*1024),
            "allowed_extensions": settings.ALLOWED_EXTENSIONS,
            "device": MODEL_CONFIG["device"],
            "batch_size": MODEL_CONFIG["batch_size"],
        }
    }


if __name__ == "__main__":
    # 직접 실행 시 서버 시작
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning",
        access_log=settings.DEBUG,
    )