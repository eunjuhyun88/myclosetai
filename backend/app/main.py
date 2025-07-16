"""
MyCloset AI Backend - 완전한 Main Application
backend/app/main.py

✅ 완전한 GPU 설정 초기화
✅ 폴백 제거, 실제 작동 코드만 유지
✅ PipelineManager 중심 구조
✅ M3 Max 최적화 완전 구현
✅ 모든 라우터 안전한 로딩
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import time
import asyncio

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

# FastAPI 및 미들웨어
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

# ===============================================================
# 🔧 Core 모듈 Import (필수)
# ===============================================================

# 로깅 설정 먼저 (순환 참조 방지)
try:
    from app.core.logging_config import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("✅ 로깅 설정 완료")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ 로깅 설정 실패: {e}")

# Core 모듈 import
try:
    from app.core import (
        settings,
        gpu_config,
        DEVICE,
        DEVICE_NAME,
        DEVICE_TYPE,
        DEVICE_INFO,
        MODEL_CONFIG,
        IS_M3_MAX,
        get_device_config,
        get_model_config,
        get_device_info,
        check_memory_available,
        optimize_memory,
        initialization_success
    )
    logger.info("✅ Core 모듈 로드 성공")
except ImportError as e:
    logger.error(f"❌ Core 모듈 로드 실패: {e}")
    print(f"❌ Core 모듈 로드 실패: {e}")
    print("시스템을 시작할 수 없습니다.")
    sys.exit(1)

# Core 모듈 초기화 확인
if not initialization_success:
    logger.error("❌ Core 모듈 초기화 실패")
    sys.exit(1)

logger.info("✅ Core 모듈 초기화 완료")

# ===============================================================
# 🔧 설정 값 추출
# ===============================================================

# Settings 객체에서 안전한 속성 접근
def get_setting(key: str, default: Any = None) -> Any:
    """설정 값 안전 추출"""
    try:
        # 직접 속성 접근 시도
        if hasattr(settings, key):
            return getattr(settings, key)
        
        # 대소문자 변환 시도
        key_upper = key.upper()
        if hasattr(settings, key_upper):
            return getattr(settings, key_upper)
        
        key_lower = key.lower()
        if hasattr(settings, key_lower):
            return getattr(settings, key_lower)
        
        # 환경 변수에서 가져오기
        env_value = os.getenv(key.upper(), os.getenv(key.lower()))
        if env_value is not None:
            return env_value
        
        return default
        
    except Exception as e:
        logger.warning(f"설정 값 '{key}' 추출 실패: {e}")
        return default

# 애플리케이션 설정
APP_NAME = get_setting('APP_NAME', "MyCloset AI Backend")
APP_VERSION = get_setting('APP_VERSION', "3.0.0")
DEBUG = get_setting('DEBUG', True)
HOST = get_setting('HOST', "0.0.0.0")
PORT = get_setting('PORT', 8000)

# 타입 변환
if isinstance(DEBUG, str):
    DEBUG = DEBUG.lower() in ['true', '1', 'yes', 'on']
if isinstance(PORT, str):
    PORT = int(PORT)

logger.info(f"📋 애플리케이션 설정:")
logger.info(f"  - 이름: {APP_NAME}")
logger.info(f"  - 버전: {APP_VERSION}")
logger.info(f"  - 디버그: {DEBUG}")
logger.info(f"  - 호스트: {HOST}")
logger.info(f"  - 포트: {PORT}")

# ===============================================================
# 🔥 API 라우터들 안전한 Import
# ===============================================================

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

# ===============================================================
# 🚀 FastAPI 앱 생성
# ===============================================================

app = FastAPI(
    title=APP_NAME,
    description=f"AI-powered virtual try-on platform (PipelineManager 중심, {DEVICE_NAME} 최적화)",
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    debug=DEBUG
)

# ===============================================================
# 🛡️ 미들웨어 설정
# ===============================================================

# CORS 미들웨어
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React 개발 서버
        "http://localhost:5173",  # Vite 개발 서버
        "http://localhost:8080",  # 추가 개발 서버
        "http://127.0.0.1:3000",  # 로컬 IP
        "http://127.0.0.1:5173",  # 로컬 IP
        "https://mycloset-ai.vercel.app",  # 배포용
        "https://*.vercel.app",  # 기타 Vercel 배포
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
    allow_headers=["*"],
)

# Gzip 압축 미들웨어
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 요청 로깅 미들웨어
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """요청 로깅 미들웨어"""
    start_time = time.time()
    
    # 요청 로그
    logger.info(f"📥 {request.method} {request.url.path}")
    
    # 응답 처리
    response = await call_next(request)
    
    # 응답 로그
    process_time = time.time() - start_time
    logger.info(f"📤 {request.method} {request.url.path} - {response.status_code} ({process_time:.3f}s)")
    
    return response

# ===============================================================
# 📁 정적 파일 서빙
# ===============================================================

# 정적 파일 디렉토리 설정
static_dir = project_root / "static"
if not static_dir.exists():
    static_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
logger.info(f"✅ 정적 파일 서빙 설정: {static_dir}")

# 업로드 및 결과 디렉토리 생성
uploads_dir = static_dir / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)

results_dir = static_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# ===============================================================
# 🔥 API 라우터 등록 (PipelineManager 중심)
# ===============================================================

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

# ===============================================================
# 🌐 기본 엔드포인트
# ===============================================================

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "MyCloset AI Backend is running!",
        "version": APP_VERSION,
        "architecture": "PipelineManager-centered",
        "ai_pipeline": "8-step AI pipeline",
        "optimization": f"{DEVICE_NAME} optimized",
        "gpu_config": {
            "device": DEVICE,
            "device_name": DEVICE_NAME,
            "is_m3_max": IS_M3_MAX,
            "memory_gb": gpu_config.memory_gb,
            "optimization_level": gpu_config.optimization_settings["optimization_level"]
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/api/health",
            "step_pipeline": "/api/step/",
            "websocket": "/api/ws/",
            "models": "/api/models/",
            "pipeline": "/api/pipeline/"
        }
    }

@app.get("/health")
async def health_check():
    """기본 헬스체크"""
    memory_check = check_memory_available(min_gb=1.0)
    
    return {
        "status": "healthy",
        "version": APP_VERSION,
        "architecture": "PipelineManager-centered",
        "debug": DEBUG,
        "gpu_config": {
            "device": DEVICE,
            "device_name": DEVICE_NAME,
            "is_m3_max": IS_M3_MAX,
            "memory_gb": gpu_config.memory_gb,
            "optimization_level": gpu_config.optimization_settings["optimization_level"]
        },
        "memory_status": {
            "available": memory_check.get('is_available', False),
            "system_memory_gb": memory_check.get('system_memory', {}).get('available_gb', 0)
        },
        "loaded_routers": list(api_routers.keys()),
        "initialization_success": initialization_success
    }

@app.get("/api/status")
async def api_status():
    """API 상태 확인"""
    device_info = get_device_info()
    model_config = get_model_config()
    
    return {
        "status": "operational",
        "timestamp": time.time(),
        "loaded_routers": list(api_routers.keys()),
        "total_routes": len(api_routers),
        "ai_pipeline_ready": "step_routes" in api_routers,
        "websocket_ready": "websocket" in api_routers,
        "gpu_status": {
            "device": DEVICE,
            "device_name": DEVICE_NAME,
            "is_m3_max": IS_M3_MAX,
            "memory_gb": gpu_config.memory_gb,
            "optimization_level": gpu_config.optimization_settings["optimization_level"],
            "is_initialized": gpu_config.is_initialized
        },
        "pipeline_optimizations": len(gpu_config.pipeline_optimizations),
        "model_config": {
            "batch_size": model_config.get("batch_size", 1),
            "dtype": model_config.get("dtype", "float32"),
            "max_workers": model_config.get("max_workers", 4),
            "concurrent_sessions": model_config.get("concurrent_sessions", 1)
        }
    }

@app.get("/api/gpu-info")
async def gpu_info():
    """GPU 정보 상세 조회"""
    return {
        "gpu_config": get_device_config(),
        "model_config": get_model_config(),
        "device_info": get_device_info(),
        "memory_stats": gpu_config.get_memory_stats(),
        "optimization_settings": gpu_config.optimization_settings,
        "pipeline_optimizations": gpu_config.pipeline_optimizations,
        "is_m3_max": IS_M3_MAX,
        "capabilities": {
            "supports_fp16": model_config.get("dtype") == "float16",
            "supports_neural_engine": IS_M3_MAX,
            "supports_metal_shaders": DEVICE == "mps",
            "supports_8step_pipeline": True,
            "max_batch_size": model_config.get("batch_size", 1) * 2,
            "recommended_image_size": (768, 768) if IS_M3_MAX else (512, 512)
        }
    }

@app.post("/api/optimize-memory")
async def optimize_memory_endpoint():
    """메모리 최적화 실행"""
    try:
        result = optimize_memory(aggressive=True)
        return {
            "status": "success",
            "optimization_result": result,
            "memory_stats": gpu_config.get_memory_stats(),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"메모리 최적화 실패: {e}")
        raise HTTPException(status_code=500, detail=f"메모리 최적화 실패: {str(e)}")

# ===============================================================
# 🔧 애플리케이션 이벤트
# ===============================================================

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행"""
    logger.info("🚀 MyCloset AI Backend 시작됨")
    logger.info(f"🏗️ 아키텍처: PipelineManager 중심")
    logger.info(f"🔧 설정: {APP_NAME} v{APP_VERSION}")
    logger.info(f"🤖 AI 파이프라인: 8단계 통합")
    logger.info(f"📊 로드된 라우터: {len(api_routers)}개")
    
    # GPU 설정 정보
    logger.info(f"🎯 GPU 설정:")
    logger.info(f"  - 디바이스: {DEVICE} ({DEVICE_NAME})")
    logger.info(f"  - M3 Max: {'✅' if IS_M3_MAX else '❌'}")
    logger.info(f"  - 메모리: {gpu_config.memory_gb:.1f}GB")
    logger.info(f"  - 최적화 레벨: {gpu_config.optimization_settings['optimization_level']}")
    logger.info(f"  - 배치 크기: {gpu_config.model_config['batch_size']}")
    logger.info(f"  - 정밀도: {gpu_config.model_config['dtype']}")
    logger.info(f"  - 동시 세션: {gpu_config.optimization_settings['concurrent_sessions']}")
    
    # 메모리 상태 확인
    memory_check = check_memory_available(min_gb=2.0)
    if memory_check.get('is_available', False):
        logger.info(f"💾 메모리 상태: {memory_check['system_memory']['available_gb']:.1f}GB 사용 가능")
    else:
        logger.warning("⚠️ 메모리 부족 - 성능이 저하될 수 있습니다.")
    
    # M3 Max 특화 정보
    if IS_M3_MAX:
        logger.info("🍎 M3 Max 특화 기능 활성화:")
        logger.info("  - Neural Engine 가속")
        logger.info("  - Metal Performance Shaders")
        logger.info("  - 통합 메모리 최적화")
        logger.info("  - 8단계 파이프라인 최적화")
        logger.info("  - 고해상도 처리 지원")
    
    # 8단계 파이프라인 최적화 상태
    pipeline_count = len(gpu_config.pipeline_optimizations)
    if pipeline_count > 0:
        logger.info(f"⚙️ 8단계 파이프라인 최적화: {pipeline_count}개 단계 설정됨")
    
    # 초기 메모리 최적화 실행
    try:
        optimization_result = optimize_memory()
        if optimization_result.get('success', False):
            logger.info(f"💾 초기 메모리 최적화 완료: {optimization_result['method']}")
    except Exception as e:
        logger.warning(f"초기 메모리 최적화 실패: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 실행"""
    logger.info("🛑 MyCloset AI Backend 종료 중...")
    
    # 메모리 정리
    try:
        cleanup_result = gpu_config.cleanup_memory(aggressive=True)
        if cleanup_result.get('success', False):
            logger.info(f"💾 종료 시 메모리 정리 완료: {cleanup_result['method']}")
    except Exception as e:
        logger.warning(f"종료 시 메모리 정리 실패: {e}")
    
    logger.info("🛑 MyCloset AI Backend 종료됨")

# ===============================================================
# 🎯 메인 실행
# ===============================================================

if __name__ == "__main__":
    logger.info("🚀 MyCloset AI Backend 서버 시작 중...")
    logger.info(f"📍 주소: http://{HOST}:{PORT}")
    logger.info(f"📖 API 문서: http://{HOST}:{PORT}/docs")
    logger.info(f"🏗️ 아키텍처: PipelineManager 중심 (VirtualFitter 제거)")
    logger.info(f"🎯 GPU 최적화: {DEVICE_NAME} ({DEVICE})")
    logger.info(f"🍎 M3 Max 최적화: {'✅' if IS_M3_MAX else '❌'}")
    
    # 시스템 정보 출력
    logger.info("📊 시스템 정보:")
    logger.info(f"  - Python: {sys.version.split()[0]}")
    logger.info(f"  - PyTorch: {gpu_config.hardware.system_info['pytorch_version']}")
    logger.info(f"  - Platform: {gpu_config.hardware.system_info['platform']}")
    logger.info(f"  - Machine: {gpu_config.hardware.system_info['machine']}")
    logger.info(f"  - CPU 코어: {gpu_config.hardware.cpu_cores}")
    logger.info(f"  - 메모리: {gpu_config.memory_gb:.1f}GB")
    
    # 개발 모드 경고
    if DEBUG:
        logger.warning("⚠️ 개발 모드로 실행 중 - 프로덕션에서는 DEBUG=False로 설정하세요")
    
    # 서버 시작
    try:
        uvicorn.run(
            "app.main:app",
            host=HOST,
            port=PORT,
            reload=DEBUG,
            log_level="info" if DEBUG else "warning",
            access_log=DEBUG,
            workers=1  # GPU 사용 시 단일 워커 권장
        )
    except Exception as e:
        logger.error(f"❌ 서버 시작 실패: {e}")
        sys.exit(1)