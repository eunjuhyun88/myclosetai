"""
MyCloset AI FastAPI 메인 애플리케이션
8단계 파이프라인과 프론트엔드를 연결하는 백엔드 서버
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import logging
import time
import psutil
from datetime import datetime
from pathlib import Path

# 로컬 임포트
from .api.pipeline_routes import router as pipeline_router
from .core.config import settings
from .models.schemas import HealthCheck, SystemStats, MonitoringData

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="MyCloset AI Backend",
    description="8단계 AI 파이프라인 기반 가상 피팅 시스템",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 시작 시간 기록
start_time = time.time()
request_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "processing_times": [],
    "quality_scores": []
}

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZIP 압축 미들웨어
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

# 라우터 등록
app.include_router(pipeline_router)

# 요청 처리 미들웨어
@app.middleware("http")
async def process_request_middleware(request, call_next):
    """요청 처리 미들웨어 - 통계 수집"""
    start_time_req = time.time()
    request_stats["total_requests"] += 1
    
    try:
        response = await call_next(request)
        
        # 성공한 요청 기록
        if response.status_code < 400:
            request_stats["successful_requests"] += 1
            processing_time = time.time() - start_time_req
            request_stats["processing_times"].append(processing_time)
            
            # 처리 시간이 너무 많이 쌓이지 않도록 제한
            if len(request_stats["processing_times"]) > 1000:
                request_stats["processing_times"] = request_stats["processing_times"][-500:]
        
        return response
        
    except Exception as e:
        logger.error(f"요청 처리 중 오류: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "detail": str(e)}
        )

# 시작/종료 이벤트
@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행"""
    logger.info("🚀 MyCloset AI Backend 시작")
    logger.info(f"⚙️  설정: {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"🖥️  디바이스: {settings.DEVICE}")
    logger.info(f"💾 메모리 한계: {settings.MEMORY_LIMIT_GB}GB")
    
    # 필요한 디렉토리 생성
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.RESULT_DIR).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    logger.info("✅ 백엔드 시작 완료")

@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 실행"""
    logger.info("🛑 MyCloset AI Backend 종료 중...")
    
    # 파이프라인 정리
    try:
        from .api.pipeline_routes import pipeline_instance
        if pipeline_instance:
            pipeline_instance.cleanup()
            logger.info("🧹 파이프라인 리소스 정리 완료")
    except Exception as e:
        logger.error(f"파이프라인 정리 중 오류: {e}")
    
    logger.info("✅ 백엔드 종료 완료")

# 기본 엔드포인트
@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "MyCloset AI Backend API",
        "version": "1.0.0",
        "status": "운영 중",
        "docs": "/docs",
        "pipeline_status": "/api/pipeline/status"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """헬스 체크 엔드포인트"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        uptime=time.time() - start_time
    )

@app.get("/stats", response_model=SystemStats)
async def get_system_stats():
    """시스템 통계 조회"""
    avg_processing_time = (
        sum(request_stats["processing_times"]) / len(request_stats["processing_times"])
        if request_stats["processing_times"] else 0.0
    )
    
    avg_quality_score = (
        sum(request_stats["quality_scores"]) / len(request_stats["quality_scores"])
        if request_stats["quality_scores"] else 0.0
    )
    
    # 메모리 사용량
    memory_info = psutil.virtual_memory()
    peak_memory = memory_info.used / (1024**3)  # GB
    
    return SystemStats(
        total_requests=request_stats["total_requests"],
        successful_requests=request_stats["successful_requests"],
        average_processing_time=avg_processing_time,
        average_quality_score=avg_quality_score,
        peak_memory_usage=peak_memory,
        uptime=time.time() - start_time,
        last_request_time=datetime.now() if request_stats["total_requests"] > 0 else None
    )

@app.get("/monitoring", response_model=MonitoringData)
async def get_monitoring_data():
    """실시간 모니터링 데이터"""
    
    # 시스템 리소스 정보
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    net_io = psutil.net_io_counters()
    
    return MonitoringData(
        cpu_usage=cpu_percent,
        memory_usage=memory.percent,
        disk_usage=(disk.used / disk.total) * 100,
        network_io={
            "bytes_sent": float(net_io.bytes_sent),
            "bytes_recv": float(net_io.bytes_recv)
        },
        active_requests=0,  # 실제 구현시 활성 요청 수 추적
        queue_size=0,       # 실제 구현시 대기열 크기 추적
        timestamp=datetime.now()
    )

# 개발용 엔드포인트
@app.get("/dev/info")
async def development_info():
    """개발 정보 (개발 모드에서만)"""
    if not settings.DEBUG:
        raise HTTPException(status_code=404, detail="Not found")
    
    import torch
    import platform
    
    return {
        "system": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "architecture": platform.machine()
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        },
        "settings": {
            "app_name": settings.APP_NAME,
            "device": settings.DEVICE,
            "debug": settings.DEBUG,
            "memory_limit": f"{settings.MEMORY_LIMIT_GB}GB"
        }
    }

# 글로벌 예외 처리기
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """글로벌 예외 처리"""
    logger.error(f"예상치 못한 오류: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "서버에서 예상치 못한 오류가 발생했습니다.",
            "request_id": str(id(request))
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP 예외 처리"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

# 커스텀 엔드포인트들
@app.post("/api/feedback")
async def submit_feedback(feedback: dict):
    """사용자 피드백 수집"""
    logger.info(f"사용자 피드백: {feedback}")
    
    # 실제 구현에서는 데이터베이스에 저장
    return {
        "success": True,
        "message": "피드백이 접수되었습니다.",
        "feedback_id": str(time.time())
    }

@app.get("/api/examples")
async def get_example_images():
    """예시 이미지 목록 조회"""
    examples_dir = Path("static/examples")
    examples_dir.mkdir(exist_ok=True)
    
    example_files = []
    for file_path in examples_dir.glob("*.jpg"):
        example_files.append({
            "name": file_path.stem,
            "url": f"/static/examples/{file_path.name}",
            "type": "person" if "person" in file_path.name else "clothing"
        })
    
    return {
        "examples": example_files,
        "total_count": len(example_files)
    }

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )