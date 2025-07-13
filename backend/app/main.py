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

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
import time
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

# API 라우터 임포트 (에러가 있을 수 있으니 try-except로 감쌈)
from app.api.health import router as health_router
try:
    from app.api.virtual_tryon import router as virtual_tryon_router
    VIRTUAL_TRYON_ROUTER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ virtual_tryon 라우터 임포트 실패: {e}")
    VIRTUAL_TRYON_ROUTER_AVAILABLE = False

try:
    from app.api.models import router as models_router
    MODELS_ROUTER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ models 라우터 임포트 실패: {e}")
    MODELS_ROUTER_AVAILABLE = False

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


# =============================================================================
# 직접 구현한 가상 피팅 API 엔드포인트
# =============================================================================

@app.post("/api/virtual-tryon")
async def virtual_tryon_direct(
    person_image: UploadFile = File(..., description="사용자 이미지"),
    clothing_image: UploadFile = File(..., description="의류 이미지"),
    height: float = Form(..., description="키 (cm)"),
    weight: float = Form(..., description="몸무게 (kg)"),
):
    """
    AI 가상 피팅 API (직접 구현)
    """
    logger.info(f"🎽 가상 피팅 요청: height={height}cm, weight={weight}kg")
    
    try:
        # 1. 이미지 파일 검증
        if not person_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="사용자 이미지가 올바르지 않습니다")
        
        if not clothing_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="의류 이미지가 올바르지 않습니다")
        
        # 2. 이미지 로드
        person_image_data = await person_image.read()
        clothing_image_data = await clothing_image.read()
        
        person_pil = Image.open(io.BytesIO(person_image_data))
        clothing_pil = Image.open(io.BytesIO(clothing_image_data))
        
        logger.info(f"📸 이미지 로드 완료: 사용자={person_pil.size}, 의류={clothing_pil.size}")
        
        # 3. AI 가상 피팅 실행
        start_time = time.time()
        
        # 실제 AI 모델 사용 시도
        try:
            from app.services.virtual_fitter import virtual_fitter
            result_image, metadata = await virtual_fitter.complete_ai_fitting(
                person_pil, clothing_pil, height, weight
            )
            logger.info("✅ 실제 AI 모델 사용")
        except Exception as e:
            logger.warning(f"⚠️ AI 모델 사용 실패, 데모 모드로 전환: {e}")
            # 데모 모드: 기본 합성
            result_image = create_demo_result(person_pil, clothing_pil)
            metadata = {"mode": "demo", "error": str(e)}
        
        processing_time = time.time() - start_time
        
        # 4. 결과 이미지를 Base64로 인코딩
        result_buffer = io.BytesIO()
        result_image.save(result_buffer, format='JPEG', quality=85)
        result_base64 = base64.b64encode(result_buffer.getvalue()).decode()
        
        # 5. BMI 계산
        bmi = weight / ((height / 100) ** 2)
        
        # 6. 응답 데이터 구성
        response_data = {
            "success": True,
            "fitted_image": result_base64,
            "processing_time": round(processing_time, 2),
            "confidence": metadata.get("confidence", 0.85),
            "measurements": {
                "chest": 90,  # 실제로는 AI가 측정
                "waist": 75,
                "hip": 95,
                "bmi": round(bmi, 1)
            },
            "clothing_analysis": {
                "category": "shirt",  # 실제로는 AI가 분석
                "style": "casual",
                "dominant_color": [120, 150, 200]  # RGB
            },
            "fit_score": metadata.get("fit_score", 0.88),
            "recommendations": [
                "이 옷은 당신의 체형에 잘 어울립니다",
                "어깨 라인이 자연스럽게 맞습니다",
                "전체적인 핏이 우수합니다"
            ]
        }
        
        logger.info(f"✅ 가상 피팅 완료: {processing_time:.2f}초")
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 가상 피팅 오류: {e}")
        raise HTTPException(status_code=500, detail=f"가상 피팅 처리 중 오류가 발생했습니다: {str(e)}")


@app.get("/api/virtual-tryon/test")
async def virtual_tryon_test():
    """가상 피팅 테스트 엔드포인트"""
    return {
        "success": True,
        "message": "Virtual Try-On API is working!",
        "timestamp": time.time(),
        "endpoints": {
            "virtual_tryon": "/api/virtual-tryon",
            "test": "/api/virtual-tryon/test",
            "docs": "/docs"
        }
    }


def create_demo_result(person_image: Image.Image, clothing_image: Image.Image) -> Image.Image:
    """데모용 기본 합성"""
    logger.info("🎭 데모 모드: 기본 이미지 합성")
    
    # 사용자 이미지 복사
    result = person_image.copy()
    
    # 의류를 축소하여 적절한 위치에 배치
    clothing_resized = clothing_image.resize((200, 250))
    
    # 이미지 중앙 상단에 의류 배치
    person_width, person_height = result.size
    clothing_x = (person_width - 200) // 2
    clothing_y = person_height // 4
    
    # 경계 체크
    if clothing_x < 0:
        clothing_x = 0
    if clothing_y < 0:
        clothing_y = 0
    if clothing_x + 200 > person_width:
        clothing_x = person_width - 200
    if clothing_y + 250 > person_height:
        clothing_y = person_height - 250
    
    # 의류가 투명도를 지원하는 경우
    if clothing_resized.mode == 'RGBA':
        result.paste(clothing_resized, (clothing_x, clothing_y), clothing_resized)
    else:
        result.paste(clothing_resized, (clothing_x, clothing_y))
    
    return result


# =============================================================================
# 기존 라우터 등록 (사용 가능한 경우에만)
# =============================================================================

# API 라우터 등록
app.include_router(health_router, prefix="/health", tags=["Health"])

if VIRTUAL_TRYON_ROUTER_AVAILABLE:
    app.include_router(virtual_tryon_router, prefix="/api/virtual-tryon-router", tags=["Virtual Try-On Router"])
    logger.info("✅ virtual_tryon 라우터 등록됨")
else:
    logger.info("ℹ️ virtual_tryon 라우터를 직접 구현으로 대체")

if MODELS_ROUTER_AVAILABLE:
    app.include_router(models_router, prefix="/api/models", tags=["Models"])
    logger.info("✅ models 라우터 등록됨")


@app.get("/")
async def root():
    """루트 엔드포인트"""
    
    return {
        "service": "MyCloset AI Backend",
        "version": "1.0.0",
        "status": "running",
        "gpu_device": gpu_config.device,
        "docs": "/docs",
        "health": "/health",
        "virtual_tryon_api": "/api/virtual-tryon",
        "test_api": "/api/virtual-tryon/test"
    }


@app.get("/routes")
async def list_routes():
    """등록된 라우터 목록 확인"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods) if route.methods else [],
                "name": getattr(route, 'name', 'unknown')
            })
    return {"routes": routes}


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
        },
        "api_status": {
            "virtual_tryon_router": VIRTUAL_TRYON_ROUTER_AVAILABLE,
            "models_router": MODELS_ROUTER_AVAILABLE,
            "direct_api": True
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