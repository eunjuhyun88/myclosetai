"""
MyCloset AI Backend - Main Application
VirtualFitter 제거 후 PipelineManager 중심 구조
✅ 기존 함수명/클래스명 유지
✅ step_routes.py ↔ PipelineManager 직접 연결
✅ M3 Max 최적화
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

# FastAPI 및 미들웨어
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

# 설정 및 로깅
try:
    from app.core.config import settings
    # Settings 객체에서 속성 접근 방식 확인
    if hasattr(settings, 'APP_NAME'):
        APP_NAME = settings.APP_NAME
    elif hasattr(settings, 'app_name'):
        APP_NAME = settings.app_name
    else:
        APP_NAME = "MyCloset AI Backend"
    
    if hasattr(settings, 'APP_VERSION'):
        APP_VERSION = settings.APP_VERSION
    elif hasattr(settings, 'app') and hasattr(settings.app, 'get'):
        APP_VERSION = settings.app.get('app_version', '1.0.0')
    else:
        APP_VERSION = "1.0.0"
        
    if hasattr(settings, 'DEBUG'):
        DEBUG = settings.DEBUG
    elif hasattr(settings, 'app') and hasattr(settings.app, 'get'):
        DEBUG = settings.app.get('debug', True)
    else:
        DEBUG = True
        
    if hasattr(settings, 'HOST'):
        HOST = settings.HOST
    elif hasattr(settings, 'app') and hasattr(settings.app, 'get'):
        HOST = settings.app.get('host', '0.0.0.0')
    else:
        HOST = "0.0.0.0"
        
    if hasattr(settings, 'PORT'):
        PORT = settings.PORT
    elif hasattr(settings, 'app') and hasattr(settings.app, 'get'):
        PORT = settings.app.get('port', 8000)
    else:
        PORT = 8000
        
except ImportError:
    APP_NAME = "MyCloset AI Backend"
    APP_VERSION = "1.0.0"
    DEBUG = True
    HOST = "0.0.0.0"
    PORT = 8000
from app.core.logging_config import setup_logging

# 로깅 설정
setup_logging()
logger = logging.getLogger(__name__)

# ============================================
# 🔥 API 라우터들 안전한 Import
# ============================================

# 핵심 라우터들
api_routers = {}

# 1. Health 라우터 (기본)
try:
    from app.api.health import router as health_router
    api_routers['health'] = health_router
    logger.info("✅ Health 라우터 로드 성공")
except ImportError as e:
    logger.warning(f"⚠️ Health 라우터 로드 실패: {e}")

# 2. Step Routes 라우터 (메인 - PipelineManager 연결)
try:
    from app.api.step_routes import router as step_routes_router
    api_routers['step_routes'] = step_routes_router
    logger.info("🔥 Step Routes 라우터 로드 성공 (PipelineManager 연결)")
except ImportError as e:
    logger.warning(f"⚠️ Step Routes 라우터 로드 실패: {e}")

# 3. WebSocket 라우터 (실시간 통신)
try:
    from app.api.websocket_routes import router as websocket_router
    api_routers['websocket'] = websocket_router
    logger.info("✅ WebSocket 라우터 로드 성공")
except ImportError as e:
    logger.warning(f"⚠️ WebSocket 라우터 로드 실패: {e}")

# 4. Models 라우터 (모델 관리)
try:
    from app.api.models import router as models_router
    api_routers['models'] = models_router
    logger.info("✅ Models 라우터 로드 성공")
except ImportError as e:
    logger.warning(f"⚠️ Models 라우터 로드 실패: {e}")

# 5. Pipeline 라우터 (전체 파이프라인 실행)
try:
    from app.api.pipeline_routes import router as pipeline_router
    api_routers['pipeline'] = pipeline_router
    logger.info("✅ Pipeline 라우터 로드 성공")
except ImportError as e:
    logger.warning(f"⚠️ Pipeline 라우터 로드 실패: {e}")

# ============================================
# 🚀 FastAPI 앱 생성
# ============================================

app = FastAPI(
    title=APP_NAME,
    description="AI-powered virtual try-on platform (PipelineManager 중심)",
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================
# 🛡️ 미들웨어 설정
# ============================================

# CORS 미들웨어
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React 개발 서버
        "http://localhost:5173",  # Vite 개발 서버
        "http://localhost:8080",  # 추가 개발 서버
        "https://mycloset-ai.vercel.app"  # 배포용
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
    allow_headers=["*"],
)

# Gzip 압축 미들웨어
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ============================================
# 📁 정적 파일 서빙
# ============================================

# 정적 파일 디렉토리 설정
static_dir = project_root / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info(f"✅ 정적 파일 서빙 설정: {static_dir}")

# 업로드 디렉토리 설정
uploads_dir = static_dir / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)

results_dir = static_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# ============================================
# 🔥 API 라우터 등록 (PipelineManager 중심)
# ============================================

# 1. Health 라우터 (기본 헬스체크)
if api_routers.get('health'):
    try:
        app.include_router(api_routers['health'], prefix="/api", tags=["health"])
        logger.info("✅ Health 라우터 등록 완료")
    except Exception as e:
        logger.warning(f"Health 라우터 등록 실패: {e}")

# 2. Step Routes 라우터 (메인 - 8단계 AI 파이프라인)
if api_routers.get('step_routes'):
    try:
        app.include_router(api_routers['step_routes'], prefix="/api/step", tags=["step-pipeline"])
        logger.info("🔥 Step Routes 라우터 등록 완료 (PipelineManager 연결)")
        logger.info("   📋 엔드포인트:")
        logger.info("     - POST /api/step/1/upload-validation")
        logger.info("     - POST /api/step/3/human-parsing")
        logger.info("     - POST /api/step/7/virtual-fitting")
        logger.info("     - GET /api/step/health")
    except Exception as e:
        logger.warning(f"Step Routes 라우터 등록 실패: {e}")

# 3. WebSocket 라우터 (실시간 진행 상황)
if api_routers.get('websocket'):
    try:
        app.include_router(api_routers['websocket'], prefix="/api/ws", tags=["websocket"])
        logger.info("✅ WebSocket 라우터 등록 완료")
    except Exception as e:
        logger.warning(f"WebSocket 라우터 등록 실패: {e}")

# 4. Models 라우터 (AI 모델 관리)
if api_routers.get('models'):
    try:
        app.include_router(api_routers['models'], prefix="/api/models", tags=["models"])
        logger.info("✅ Models 라우터 등록 완료")
    except Exception as e:
        logger.warning(f"Models 라우터 등록 실패: {e}")

# 5. Pipeline 라우터 (전체 파이프라인 실행)
if api_routers.get('pipeline'):
    try:
        app.include_router(api_routers['pipeline'], prefix="/api/pipeline", tags=["pipeline"])
        logger.info("✅ Pipeline 라우터 등록 완료")
    except Exception as e:
        logger.warning(f"Pipeline 라우터 등록 실패: {e}")

# ============================================
# 🌐 기본 엔드포인트
# ============================================

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "MyCloset AI Backend is running!",
        "version": APP_VERSION,
        "architecture": "PipelineManager-centered",
        "ai_pipeline": "8-step AI pipeline",
        "optimization": "M3 Max optimized",
        "docs": "/docs",
        "health": "/api/health"
    }

@app.get("/health")
async def health_check():
    """기본 헬스체크"""
    return {
        "status": "healthy",
        "version": APP_VERSION,
        "architecture": "PipelineManager-centered",
        "debug": DEBUG
    }

@app.get("/api/status")
async def api_status():
    """API 상태 확인"""
    return {
        "status": "operational",
        "loaded_routers": list(api_routers.keys()),
        "total_routes": len(api_routers),
        "ai_pipeline_ready": "step_routes" in api_routers,
        "websocket_ready": "websocket" in api_routers
    }

# ============================================
# 🔧 애플리케이션 이벤트
# ============================================

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행"""
    logger.info("🚀 MyCloset AI Backend 시작됨")
    logger.info(f"🏗️ 아키텍처: PipelineManager 중심")
    logger.info(f"🔧 설정: {APP_NAME} v{APP_VERSION}")
    logger.info(f"🤖 AI 파이프라인: 8단계 통합")
    logger.info(f"📊 로드된 라우터: {len(api_routers)}개")
    
    # PipelineManager 초기화는 step_routes.py에서 자동으로 처리됨

@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 실행"""
    logger.info("🛑 MyCloset AI Backend 종료됨")
    # PipelineManager 정리는 step_routes.py에서 자동으로 처리됨

# ============================================
# 🎯 메인 실행
# ============================================

if __name__ == "__main__":
    logger.info("🚀 MyCloset AI Backend 서버 시작 중...")
    logger.info(f"📍 주소: http://{HOST}:{PORT}")
    logger.info(f"📖 API 문서: http://{HOST}:{PORT}/docs")
    logger.info(f"🏗️ 아키텍처: PipelineManager 중심 (VirtualFitter 제거)")
    
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level="info"
    )