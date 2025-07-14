# app/main.py
"""
MyCloset AI Backend - 완전한 메인 애플리케이션
M3 Max 128GB 최적화, 안정적인 import 처리, 프로덕션 레벨 구현
"""

import sys
import os
import logging
import time
import asyncio
import traceback
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

# Python 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(project_root))

print("🐍 Python 경로 설정:")
print(f"  - App Dir: {current_dir}")
print(f"  - Project Root: {project_root}")

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, Request, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import JSONResponse, HTMLResponse
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException
    from pydantic import BaseModel
except ImportError as e:
    print(f"❌ FastAPI import 실패: {e}")
    sys.exit(1)

# 로깅 설정
def setup_logging():
    """로깅 시스템 초기화"""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 파일 핸들러
    file_handler = logging.FileHandler(
        log_dir / f"mycloset-ai-{datetime.now().strftime('%Y%m%d')}.log",
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)

# 로깅 초기화
logger = setup_logging()

# ============================================
# 안전한 컴포넌트 Import 시스템
# ============================================

class ComponentImporter:
    """안전한 컴포넌트 import 매니저"""
    
    def __init__(self):
        self.components = {}
        self.import_errors = []
        self.fallback_mode = False
    
    def safe_import_gpu_config(self):
        """GPU 설정 안전 import"""
        try:
            from app.core.gpu_config import (
                gpu_config, DEVICE, MODEL_CONFIG, 
                DEVICE_INFO, get_device_config,
                get_device, get_model_config, get_device_info  # 추가된 함수들
            )
            
            self.components['gpu_config'] = {
                'instance': gpu_config,
                'device': DEVICE,
                'model_config': MODEL_CONFIG,
                'device_info': DEVICE_INFO,
                'get_config': get_device_config,
                'get_device': get_device,
                'get_model_config': get_model_config,
                'get_device_info': get_device_info
            }
            
            logger.info("✅ GPU 설정 import 성공")
            return True
            
        except ImportError as e:
            error_msg = f"GPU 설정 import 실패: {e}"
            self.import_errors.append(error_msg)
            logger.warning(f"⚠️ {error_msg}")
            
            # 폴백 설정
            self.components['gpu_config'] = {
                'instance': None,
                'device': "cpu",
                'model_config': {"device": "cpu", "dtype": "float32"},
                'device_info': {
                    "device": "cpu",
                    "name": "CPU",
                    "memory_gb": 0,
                    "is_m3_max": False,
                    "pytorch_version": "unknown",
                    "mps_available": False
                },
                'get_config': lambda: {"device": "cpu", "name": "CPU"},
                'get_device': lambda: "cpu",
                'get_model_config': lambda: {"device": "cpu"},
                'get_device_info': lambda: {"device": "cpu", "name": "CPU"}
            }
            return False
    
    def safe_import_memory_manager(self):
        """메모리 매니저 안전 import"""
        try:
            from app.ai_pipeline.utils.memory_manager import (
                get_memory_manager, 
                optimize_memory_usage,
                check_memory,
                MemoryManager,
                get_global_memory_manager,  # 추가된 함수들
                create_memory_manager,
                get_default_memory_manager
            )
            
            self.components['memory_manager'] = {
                'get_manager': get_memory_manager,
                'optimize': optimize_memory_usage,
                'check': check_memory,
                'class': MemoryManager,
                'get_global': get_global_memory_manager,
                'create': create_memory_manager,
                'get_default': get_default_memory_manager
            }
            
            logger.info("✅ 메모리 매니저 import 성공")
            return True
            
        except ImportError as e:
            error_msg = f"메모리 매니저 import 실패: {e}"
            self.import_errors.append(error_msg)
            logger.warning(f"⚠️ {error_msg}")
            
            # 폴백 함수들
            def fallback_get_memory_manager():
                return None
            
            def fallback_optimize_memory_usage(device=None, aggressive=False):
                return {
                    "success": False, 
                    "error": "Memory manager not available",
                    "device": device or "unknown"
                }
            
            def fallback_check_memory():
                return {
                    "status": "unknown", 
                    "error": "Memory manager not available"
                }
            
            self.components['memory_manager'] = {
                'get_manager': fallback_get_memory_manager,
                'optimize': fallback_optimize_memory_usage,
                'check': fallback_check_memory,
                'class': None,
                'get_global': fallback_get_memory_manager,
                'create': fallback_get_memory_manager,
                'get_default': fallback_get_memory_manager
            }
            return False
    
    def safe_import_m3_optimizer(self):
        """M3 Max 최적화 안전 import"""
        try:
            from app.core.m3_optimizer import (
                get_m3_optimizer,
                is_m3_max_optimized,
                get_optimal_config,
                M3MaxOptimizer,
                create_m3_optimizer,  # 추가된 함수들
                get_m3_config,
                optimize_for_m3_max
            )
            
            self.components['m3_optimizer'] = {
                'get_optimizer': get_m3_optimizer,
                'is_optimized': is_m3_max_optimized,
                'get_config': get_optimal_config,
                'class': M3MaxOptimizer,
                'create': create_m3_optimizer,
                'get_m3_config': get_m3_config,
                'optimize': optimize_for_m3_max
            }
            
            logger.info("✅ M3 최적화 import 성공")
            return True
            
        except ImportError as e:
            error_msg = f"M3 최적화 import 실패: {e}"
            self.import_errors.append(error_msg)
            logger.warning(f"⚠️ {error_msg}")
            
            # 폴백 함수들
            def fallback_get_m3_optimizer():
                class FallbackOptimizer:
                    def __init__(self):
                        self.is_m3_max = False
                        self.device = "cpu"
                return FallbackOptimizer()
            
            def fallback_is_m3_max_optimized():
                return False
            
            def fallback_get_optimal_config(model_type="diffusion"):
                return {"device": "cpu", "batch_size": 1}
            
            self.components['m3_optimizer'] = {
                'get_optimizer': fallback_get_m3_optimizer,
                'is_optimized': fallback_is_m3_max_optimized,
                'get_config': fallback_get_optimal_config,
                'class': None,
                'create': fallback_get_m3_optimizer,
                'get_m3_config': lambda: {"device": "cpu"},
                'optimize': fallback_is_m3_max_optimized
            }
            return False
    
    def safe_import_pipeline_manager(self):
        """파이프라인 매니저 안전 import"""
        try:
            from app.ai_pipeline.pipeline_manager import (
                PipelineManager, PipelineMode,
                get_pipeline_manager,  # 추가된 함수들  
                create_pipeline_manager,
                get_available_modes
            )
            
            self.components['pipeline_manager'] = {
                'class': PipelineManager,
                'modes': PipelineMode,
                'get_manager': get_pipeline_manager,
                'create': create_pipeline_manager,
                'get_modes': get_available_modes
            }
            
            logger.info("✅ 파이프라인 매니저 import 성공")
            return True
            
        except ImportError as e:
            error_msg = f"파이프라인 매니저 import 실패: {e}"
            self.import_errors.append(error_msg)
            logger.warning(f"⚠️ {error_msg}")
            
            # 시뮬레이션 파이프라인 클래스
            class SimulationPipelineManager:
                def __init__(self, mode="simulation", **kwargs):
                    self.mode = mode
                    self.is_initialized = False
                    self.device = kwargs.get('device', 'cpu')
                
                async def initialize(self):
                    logger.info("🎭 시뮬레이션 파이프라인 초기화...")
                    await asyncio.sleep(1)  # 초기화 시뮬레이션
                    self.is_initialized = True
                    logger.info("✅ 시뮬레이션 파이프라인 준비 완료")
                    return True
                
                async def cleanup(self):
                    logger.info("🎭 시뮬레이션 파이프라인 정리 완료")
                    self.is_initialized = False
                
                def get_status(self):
                    return {
                        "mode": self.mode,
                        "initialized": self.is_initialized,
                        "device": self.device,
                        "simulation": True
                    }
            
            # 시뮬레이션 모드 enum
            class SimulationMode:
                SIMULATION = "simulation"
                PRODUCTION = "production"
                HYBRID = "hybrid"
            
            def fallback_get_pipeline_manager():
                return None
            
            def fallback_create_pipeline_manager(mode="simulation"):
                return SimulationPipelineManager(mode=mode)
            
            def fallback_get_available_modes():
                return {
                    "simulation": "simulation",
                    "production": "production", 
                    "hybrid": "hybrid"
                }
            
            self.components['pipeline_manager'] = {
                'class': SimulationPipelineManager,
                'modes': SimulationMode,
                'get_manager': fallback_get_pipeline_manager,
                'create': fallback_create_pipeline_manager,
                'get_modes': fallback_get_available_modes
            }
            self.fallback_mode = True
            return False
    
    def safe_import_api_routers(self):
        """API 라우터들 안전 import"""
        routers = {}
        
        # Health router
        try:
            from app.api.health import router as health_router
            routers['health'] = health_router
            logger.info("✅ Health 라우터 import 성공")
        except ImportError as e:
            logger.warning(f"⚠️ Health 라우터 import 실패: {e}")
            routers['health'] = None
        
        # Virtual try-on router
        try:
            from app.api.virtual_tryon import router as virtual_tryon_router
            routers['virtual_tryon'] = virtual_tryon_router
            logger.info("✅ Virtual Try-on 라우터 import 성공")
        except ImportError as e:
            logger.warning(f"⚠️ Virtual Try-on 라우터 import 실패: {e}")
            routers['virtual_tryon'] = None
        
        # Models router
        try:
            from app.api.models import router as models_router
            routers['models'] = models_router
            logger.info("✅ Models 라우터 import 성공")
        except ImportError as e:
            logger.warning(f"⚠️ Models 라우터 import 실패: {e}")
            routers['models'] = None
        
        # Pipeline routes
        try:
            from app.api.pipeline_routes import router as pipeline_router
            routers['pipeline'] = pipeline_router
            logger.info("✅ Pipeline 라우터 import 성공")
        except ImportError as e:
            logger.warning(f"⚠️ Pipeline 라우터 import 실패: {e}")
            routers['pipeline'] = None
        
        # WebSocket routes - 안전하게 처리
        try:
            # Pydantic V2 호환성 문제로 인해 조건부 import
            import pydantic
            pydantic_version = pydantic.version.VERSION
            
            if pydantic_version.startswith('2.'):
                # Pydantic V2를 사용하는 경우에만 import 시도
                from app.api.websocket_routes import router as websocket_router
                routers['websocket'] = websocket_router
                logger.info("✅ WebSocket 라우터 import 성공")
            else:
                logger.warning("⚠️ Pydantic V1 감지 - WebSocket 라우터 비활성화")
                routers['websocket'] = None
                
        except ImportError as e:
            logger.warning(f"⚠️ WebSocket 라우터 import 실패: {e}")
            routers['websocket'] = None
        except Exception as e:
            logger.warning(f"⚠️ WebSocket 라우터 오류: {e}")
            routers['websocket'] = None
        
        self.components['routers'] = routers
        return routers
    
    def initialize_all_components(self):
        """모든 컴포넌트 초기화"""
        logger.info("🔄 AI 파이프라인 로딩 시도...")
        
        # AI 파이프라인 디렉토리 확인
        ai_pipeline_dir = current_dir / "ai_pipeline"
        if ai_pipeline_dir.exists():
            logger.info(f"✅ AI 파이프라인 디렉토리 발견: {ai_pipeline_dir}")
            
            # Step 파일들 확인
            steps_dir = ai_pipeline_dir / "steps"
            if steps_dir.exists():
                step_files = list(steps_dir.glob("step_*.py"))
                logger.info(f"📊 Step 파일들 발견: {len(step_files)}개")
        
        # 필요한 디렉토리 생성
        directories_to_create = [
            project_root / "logs",
            project_root / "static" / "uploads",
            project_root / "static" / "results",
            project_root / "temp",
            current_dir / "ai_pipeline" / "cache",
            current_dir / "ai_pipeline" / "models" / "checkpoints"
        ]
        
        created_count = 0
        for directory in directories_to_create:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                created_count += 1
        
        if created_count > 0:
            print(f"✅ 필요한 디렉토리 생성 완료: {created_count}개")
        
        # 컴포넌트별 import
        success_count = 0
        
        if self.safe_import_gpu_config():
            success_count += 1
        
        if self.safe_import_memory_manager():
            success_count += 1
        
        if self.safe_import_m3_optimizer():
            success_count += 1
        
        if self.safe_import_pipeline_manager():
            success_count += 1
        
        self.safe_import_api_routers()
        
        logger.info(f"📊 컴포넌트 import 완료: {success_count}/4 성공")
        
        if self.import_errors:
            logger.warning("⚠️ Import 오류 목록:")
            for error in self.import_errors:
                logger.warning(f"  - {error}")
        
        return success_count >= 2  # 최소 절반 이상 성공

# 컴포넌트 importer 초기화
importer = ComponentImporter()
import_success = importer.initialize_all_components()

# 컴포넌트 참조 설정
gpu_config = importer.components['gpu_config']
memory_manager = importer.components['memory_manager']
m3_optimizer = importer.components['m3_optimizer']
pipeline_manager_info = importer.components['pipeline_manager']
api_routers = importer.components['routers']

# 전역 변수들
pipeline_manager = None
app_state = {
    "initialized": False,
    "startup_time": None,
    "import_success": import_success,
    "fallback_mode": importer.fallback_mode,
    "device": gpu_config['device'],
    "pipeline_mode": "simulation",
    "total_sessions": 0,
    "successful_sessions": 0,
    "errors": importer.import_errors.copy(),
    "performance_metrics": {
        "average_response_time": 0.0,
        "total_requests": 0,
        "error_rate": 0.0
    }
}

# ============================================
# Pydantic 모델들
# ============================================

class SystemStatus(BaseModel):
    """시스템 상태 모델"""
    status: str
    initialized: bool
    device: str
    pipeline_mode: str
    fallback_mode: bool
    import_success: bool
    errors: List[str]

class VirtualTryOnRequest(BaseModel):
    """가상 피팅 요청 모델"""
    person_image_url: Optional[str] = None
    clothing_image_url: Optional[str] = None
    clothing_type: str = "shirt"
    fabric_type: str = "cotton"
    quality_target: float = 0.8

class PerformanceMetrics(BaseModel):
    """성능 메트릭 모델"""
    total_requests: int
    successful_requests: int
    average_response_time: float
    error_rate: float
    uptime_seconds: float

# ============================================
# 미들웨어 및 예외 처리
# ============================================

async def add_process_time_header(request: Request, call_next):
    """요청 처리 시간 추가 미들웨어"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    response.headers["X-Process-Time"] = str(round(process_time, 4))
    
    # 성능 메트릭 업데이트
    app_state["performance_metrics"]["total_requests"] += 1
    current_avg = app_state["performance_metrics"]["average_response_time"]
    total_requests = app_state["performance_metrics"]["total_requests"]
    
    # 이동 평균 계산
    app_state["performance_metrics"]["average_response_time"] = (
        (current_avg * (total_requests - 1) + process_time) / total_requests
    )
    
    return response

# ============================================
# 애플리케이션 라이프사이클
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    global pipeline_manager, app_state
    
    # ==========================================
    # 시작 로직
    # ==========================================
    logger.info("🚀 MyCloset AI Backend 시작...")
    startup_start = time.time()
    
    try:
        # 파이프라인 매니저 초기화
        PipelineManagerClass = pipeline_manager_info['class']
        
        if PipelineManagerClass:
            logger.info("🎭 시뮬레이션 파이프라인 초기화 중...")
            
            # 디바이스 설정
            device = gpu_config['device']
            
            # 파이프라인 매니저 생성
            if importer.fallback_mode:
                logger.info("🎭 시뮬레이션 파이프라인 초기화...")
                pipeline_manager = PipelineManagerClass(mode="simulation", device=device)
            else:
                logger.info("🤖 실제 파이프라인 초기화 시도...")
                pipeline_manager = PipelineManagerClass(mode="simulation", device=device)
            
            # 초기화 시도
            initialization_success = await pipeline_manager.initialize()
            
            if initialization_success:
                app_state["initialized"] = True
                app_state["pipeline_mode"] = getattr(pipeline_manager, 'mode', 'simulation')
                logger.info("✅ 파이프라인 초기화 완료")
            else:
                logger.warning("⚠️ 파이프라인 초기화 부분 실패")
                app_state["errors"].append("Pipeline initialization partially failed")
        
        else:
            logger.error("❌ 파이프라인 매니저 클래스를 찾을 수 없음")
            app_state["errors"].append("Pipeline manager class not found")
        
        app_state["startup_time"] = time.time() - startup_start
        
        # 시스템 상태 로깅
        logger.info("=" * 60)
        logger.info("🏥 MyCloset AI Backend 시스템 상태")
        logger.info("=" * 60)
        logger.info(f"🔧 디바이스: {app_state['device']}")
        logger.info(f"🎭 파이프라인 모드: {app_state['pipeline_mode']}")
        logger.info(f"✅ 초기화 성공: {app_state['initialized']}")
        logger.info(f"🚨 폴백 모드: {app_state['fallback_mode']}")
        logger.info(f"⏱️ 시작 시간: {app_state['startup_time']:.2f}초")
        
        if app_state['errors']:
            logger.warning(f"⚠️ 오류 목록 ({len(app_state['errors'])}개):")
            for error in app_state['errors']:
                logger.warning(f"  - {error}")
        
        logger.info("✅ 백엔드 초기화 완료")
        logger.info("=" * 60)
        
    except Exception as e:
        error_msg = f"Startup error: {str(e)}"
        logger.error(f"❌ 시작 중 치명적 오류: {error_msg}")
        logger.error(f"📋 스택 트레이스: {traceback.format_exc()}")
        app_state["errors"].append(error_msg)
        app_state["initialized"] = False
    
    yield  # 애플리케이션 실행
    
    # ==========================================
    # 종료 로직
    # ==========================================
    logger.info("🛑 MyCloset AI Backend 종료 중...")
    
    try:
        if pipeline_manager and hasattr(pipeline_manager, 'cleanup'):
            await pipeline_manager.cleanup()
            logger.info("✅ 파이프라인 리소스 정리 완료")
        
        # 메모리 정리
        if memory_manager['optimize']:
            result = memory_manager['optimize'](aggressive=True)
            if result.get('success'):
                logger.info("✅ 메모리 정리 완료")
        
        logger.info("✅ 정리 완료")
        
    except Exception as e:
        logger.warning(f"⚠️ 정리 중 오류: {e}")

# ============================================
# FastAPI 애플리케이션 생성
# ============================================

app = FastAPI(
    title="MyCloset AI Backend",
    description="M3 Max 최적화 가상 피팅 AI 백엔드 서비스",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ============================================
# 미들웨어 설정
# ============================================

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 성능 측정 미들웨어
app.middleware("http")(add_process_time_header)

# ============================================
# 예외 처리
# ============================================

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """HTTP 예외 처리"""
    app_state["performance_metrics"]["total_requests"] += 1
    
    error_response = {
        "success": False,
        "error": {
            "type": "http_error",
            "status_code": exc.status_code,
            "message": exc.detail,
            "timestamp": datetime.now().isoformat()
        },
        "request_info": {
            "method": request.method,
            "url": str(request.url),
            "client": request.client.host if request.client else "unknown"
        }
    }
    
    logger.warning(f"HTTP 예외: {exc.status_code} - {exc.detail} - {request.url}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """요청 검증 예외 처리"""
    app_state["performance_metrics"]["total_requests"] += 1
    
    error_response = {
        "success": False,
        "error": {
            "type": "validation_error",
            "message": "Request validation failed",
            "details": exc.errors(),
            "timestamp": datetime.now().isoformat()
        }
    }
    
    logger.warning(f"검증 오류: {exc.errors()} - {request.url}")
    
    return JSONResponse(
        status_code=422,
        content=error_response
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """일반 예외 처리"""
    app_state["performance_metrics"]["total_requests"] += 1
    
    error_msg = str(exc)
    error_type = type(exc).__name__
    
    error_response = {
        "success": False,
        "error": {
            "type": error_type,
            "message": error_msg,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    logger.error(f"일반 예외: {error_type} - {error_msg} - {request.url}")
    logger.error(f"스택 트레이스: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content=error_response
    )

# ============================================
# API 라우터 등록
# ============================================

# Health router
if api_routers.get('health'):
    app.include_router(api_routers['health'], prefix="/health", tags=["health"])
    logger.info("✅ Health 라우터 등록됨")

# Virtual try-on router
if api_routers.get('virtual_tryon'):
    app.include_router(api_routers['virtual_tryon'], prefix="/api", tags=["virtual-tryon"])
    logger.info("✅ Virtual Try-on 라우터 등록됨")

# Models router
if api_routers.get('models'):
    app.include_router(api_routers['models'], prefix="/api", tags=["models"])
    logger.info("✅ Models 라우터 등록됨")

# Pipeline router
if api_routers.get('pipeline'):
    app.include_router(api_routers['pipeline'], prefix="/api/pipeline", tags=["pipeline"])
    logger.info("✅ Pipeline 라우터 등록됨")

# WebSocket router
if api_routers.get('websocket'):
    app.include_router(api_routers['websocket'], prefix="/api/ws", tags=["websocket"])
    logger.info("✅ WebSocket 라우터 등록됨")

# ============================================
# 정적 파일 서빙
# ============================================

# 정적 파일 디렉토리 설정
static_dir = project_root / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info("✅ 정적 파일 서빙 설정됨")

# ============================================
# 핵심 API 엔드포인트들
# ============================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """루트 엔드포인트 - HTML 대시보드"""
    device_emoji = "🍎" if gpu_config['device'] == "mps" else "🖥️" if gpu_config['device'] == "cuda" else "💻"
    status_emoji = "✅" if app_state["initialized"] else "⚠️"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MyCloset AI Backend</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }}
            .status {{ padding: 15px; border-radius: 5px; margin: 15px 0; }}
            .status.success {{ background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }}
            .status.warning {{ background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }}
            .status.error {{ background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
            .metric {{ background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }}
            .metric h3 {{ margin: 0; color: #666; font-size: 0.9em; }}
            .metric p {{ margin: 5px 0 0 0; font-size: 1.4em; font-weight: bold; color: #333; }}
            .links {{ margin-top: 30px; }}
            .links a {{ display: inline-block; margin: 5px 10px 5px 0; padding: 10px 15px; background: #007acc; color: white; text-decoration: none; border-radius: 5px; }}
            .links a:hover {{ background: #005a9e; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{device_emoji} MyCloset AI Backend v3.0</h1>
            
            <div class="status {'success' if app_state['initialized'] else 'warning'}">
                <strong>{status_emoji} 시스템 상태:</strong> 
                {'정상 운영 중' if app_state['initialized'] else '초기화 중 또는 제한적 운영'}
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <h3>디바이스</h3>
                    <p>{gpu_config['device'].upper()}</p>
                </div>
                <div class="metric">
                    <h3>파이프라인 모드</h3>
                    <p>{app_state['pipeline_mode']}</p>
                </div>
                <div class="metric">
                    <h3>총 요청 수</h3>
                    <p>{app_state['performance_metrics']['total_requests']}</p>
                </div>
                <div class="metric">
                    <h3>평균 응답 시간</h3>
                    <p>{app_state['performance_metrics']['average_response_time']:.3f}s</p>
                </div>
                <div class="metric">
                    <h3>가동 시간</h3>
                    <p>{(time.time() - (app_state['startup_time'] or time.time())):.0f}s</p>
                </div>
                <div class="metric">
                    <h3>Import 성공</h3>
                    <p>{'✅' if app_state['import_success'] else '⚠️'}</p>
                </div>
            </div>
            
            {f'<div class="status error"><strong>⚠️ 오류:</strong><br>{"<br>".join(app_state["errors"][:3])}</div>' if app_state['errors'] else ''}
            
            <div class="links">
                <a href="/docs">📚 API 문서</a>
                <a href="/status">📊 상세 상태</a>
                <a href="/health">💊 헬스체크</a>
                <a href="/api/system/performance">📈 성능 메트릭</a>
            </div>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.get("/status", response_model=Dict[str, Any])
async def get_detailed_status():
    """상세 시스템 상태 조회"""
    memory_status = memory_manager['check']()
    
    # 파이프라인 상태
    pipeline_status = {}
    if pipeline_manager and hasattr(pipeline_manager, 'get_status'):
        try:
            pipeline_status = pipeline_manager.get_status()
        except Exception as e:
            pipeline_status = {"error": str(e)}
    
    # 디바이스 정보
    device_info = gpu_config['device_info'].copy()
    
    # M3 최적화 상태
    m3_status = {}
    if m3_optimizer['get_optimizer']:
        try:
            optimizer = m3_optimizer['get_optimizer']()
            m3_status = {
                "is_m3_max": getattr(optimizer, 'is_m3_max', False),
                "device": getattr(optimizer, 'device', 'unknown'),
                "optimized": m3_optimizer['is_optimized']()
            }
        except Exception as e:
            m3_status = {"error": str(e)}
    
    uptime = time.time() - (app_state['startup_time'] or time.time())
    
    return {
        "application": {
            "name": "MyCloset AI Backend",
            "version": "3.0.0",
            "initialized": app_state["initialized"],
            "fallback_mode": app_state["fallback_mode"],
            "import_success": app_state["import_success"],
            "uptime_seconds": uptime,
            "startup_time": app_state["startup_time"],
            "errors": app_state["errors"]
        },
        "system": {
            "device": gpu_config["device"],
            "device_info": device_info,
            "memory_status": memory_status,
            "m3_optimization": m3_status
        },
        "pipeline": {
            "mode": app_state["pipeline_mode"],
            "status": pipeline_status,
            "available": pipeline_manager is not None
        },
        "performance": app_state["performance_metrics"],
        "component_status": {
            "gpu_config": gpu_config['instance'] is not None,
            "memory_manager": memory_manager['class'] is not None,
            "m3_optimizer": m3_optimizer['class'] is not None,
            "pipeline_manager": pipeline_manager_info['class'] is not None
        },
        "api_routers": {
            name: router is not None 
            for name, router in api_routers.items()
        }
    }

@app.get("/health")
async def health_check():
    """간단한 헬스체크"""
    return {
        "status": "healthy" if app_state["initialized"] else "degraded",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "device": gpu_config["device"],
        "uptime": time.time() - (app_state["startup_time"] or time.time())
    }

# ============================================
# 시스템 관리 엔드포인트들
# ============================================

@app.post("/api/system/optimize-memory")
async def optimize_memory_endpoint():
    """메모리 최적화 엔드포인트"""
    try:
        start_time = time.time()
        result = memory_manager['optimize'](device=gpu_config['device'], aggressive=False)
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "optimization_result": result,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"메모리 최적화 API 오류: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/system/performance")
async def get_performance_metrics():
    """성능 메트릭 조회"""
    uptime = time.time() - (app_state["startup_time"] or time.time())
    
    return PerformanceMetrics(
        total_requests=app_state["performance_metrics"]["total_requests"],
        successful_requests=app_state["successful_sessions"],
        average_response_time=app_state["performance_metrics"]["average_response_time"],
        error_rate=app_state["performance_metrics"]["error_rate"],
        uptime_seconds=uptime
    )

@app.post("/api/system/restart-pipeline")
async def restart_pipeline():
    """파이프라인 재시작"""
    global pipeline_manager
    
    try:
        if pipeline_manager and hasattr(pipeline_manager, 'cleanup'):
            await pipeline_manager.cleanup()
        
        PipelineManagerClass = pipeline_manager_info['class']
        if PipelineManagerClass:
            pipeline_manager = PipelineManagerClass(
                mode="simulation", 
                device=gpu_config['device']
            )
            success = await pipeline_manager.initialize()
            
            if success:
                app_state["initialized"] = True
                return {
                    "success": True,
                    "message": "파이프라인 재시작 완료",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "message": "파이프라인 재시작 실패",
                    "timestamp": datetime.now().isoformat()
                }
        else:
            return {
                "success": False,
                "message": "파이프라인 매니저 클래스를 찾을 수 없음",
                "timestamp": datetime.now().isoformat()
            }
    
    except Exception as e:
        logger.error(f"파이프라인 재시작 오류: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/system/logs")
async def get_recent_logs(lines: int = 50):
    """최근 로그 조회"""
    try:
        log_file = project_root / "logs" / f"mycloset-ai-{datetime.now().strftime('%Y%m%d')}.log"
        
        if not log_file.exists():
            return {
                "success": False,
                "message": "로그 파일을 찾을 수 없음"
            }
        
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return {
            "success": True,
            "logs": [line.strip() for line in recent_lines],
            "total_lines": len(all_lines),
            "returned_lines": len(recent_lines),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"로그 조회 오류: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ============================================
# 메인 실행부
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🍎 M3 Max 최적화된 MyCloset AI Backend v3.0.0 시작...")
    logger.info(f"🤖 AI 파이프라인: {'시뮬레이션 모드' if importer.fallback_mode else '실제 모드'}")
    logger.info(f"🔧 디바이스: {gpu_config['device']}")
    logger.info(f"📊 Import 성공: {import_success}")
    
    # 환경별 설정
    if os.getenv("ENVIRONMENT") == "production":
        # 프로덕션 설정
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            workers=1,  # M3 Max 환경에서는 단일 워커 권장
            log_level="info",
            access_log=True
        )
    else:
        # 개발 설정
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # import 문제로 인해 reload 비활성화
            log_level="info",
            access_log=True
        )

# ============================================
# 시작 시 자동 실행 코드
# ============================================

# 시작 시 메모리 상태 로깅
if memory_manager['check']:
    try:
        memory_status = memory_manager['check']()
        logger.info(f"💾 초기 메모리 상태: {memory_status['status']}")
    except Exception as e:
        logger.warning(f"메모리 상태 확인 실패: {e}")

# M3 Max 최적화 상태 로깅
if m3_optimizer['is_optimized']:
    try:
        is_optimized = m3_optimizer['is_optimized']()
        logger.info(f"🍎 M3 Max 최적화: {'✅ 활성화됨' if is_optimized else '❌ 비활성화됨'}")
    except Exception as e:
        logger.warning(f"M3 최적화 상태 확인 실패: {e}")

logger.info("🚀 MyCloset AI Backend 메인 모듈 로드 완료")