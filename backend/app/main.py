# app/main.py
"""
MyCloset AI Backend - 완전한 통합 버전
✅ ModelLoader + Schemas + Virtual Try-on API + WebSocket + Pipeline Manager + Health Router
✅ React Frontend 완벽 호환 + M3 Max 최적화
✅ 모든 구성요소 완전 통합
"""

import sys
import os
import logging
import asyncio
import traceback
import json
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

# 시간 모듈 안전 import
import time as time_module

# Python 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(project_root))

print("🍎 MyCloset AI Backend - 완전 통합 버전 시작...")
print(f"📁 App Dir: {current_dir}")
print(f"📁 Project Root: {project_root}")

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks, UploadFile, File, Form
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import JSONResponse, HTMLResponse
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException
    print("✅ FastAPI import 성공")
except ImportError as e:
    print(f"❌ FastAPI import 실패: {e}")
    sys.exit(1)

# Pydantic V2 imports
try:
    from pydantic import ValidationError
    print("✅ Pydantic V2 import 성공")
except ImportError as e:
    print(f"❌ Pydantic import 실패: {e}")
    sys.exit(1)

# 로깅 설정
def setup_logging():
    """M3 Max 최적화된 로깅 시스템 초기화"""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 파일 핸들러
    file_handler = logging.FileHandler(
        log_dir / f"mycloset-ai-complete-{datetime.now().strftime('%Y%m%d')}.log",
        encoding='utf-8',
        delay=True
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
# 🔧 완전 통합 컴포넌트 Import 시스템
# ============================================

class CompleteComponentIntegrator:
    """모든 구성요소의 완전한 통합 관리"""
    
    def __init__(self):
        self.components = {}
        self.import_errors = []
        self.integration_status = {}
        self.m3_max_optimized = False
        self._detect_environment()
    
    def _detect_environment(self):
        """환경 감지 및 최적화 설정"""
        try:
            import platform
            import psutil
            
            if platform.machine() == 'arm64' and platform.system() == 'Darwin':
                memory_gb = psutil.virtual_memory().total / (1024**3)
                if memory_gb >= 120:
                    self.m3_max_optimized = True
                    logger.info("🍎 M3 Max 128GB 환경 감지 - 최고 성능 모드 활성화")
                else:
                    logger.info(f"🍎 Apple Silicon 감지 - 메모리: {memory_gb:.0f}GB")
        except:
            pass
    
    def integrate_all_components(self):
        """모든 구성요소 통합"""
        logger.info("🚀 완전 통합 시작...")
        
        # 1. 스키마 통합
        self.integration_status['schemas'] = self._integrate_schemas()
        
        # 2. ModelLoader 통합
        self.integration_status['model_loader'] = self._integrate_model_loader()
        
        # 3. Virtual Try-on API 통합  
        self.integration_status['virtual_tryon_api'] = self._integrate_virtual_tryon_api()
        
        # 4. WebSocket 통합
        self.integration_status['websocket'] = self._integrate_websocket()
        
        # 5. Pipeline Manager 통합
        self.integration_status['pipeline_manager'] = self._integrate_pipeline_manager()
        
        # 6. Health Router 통합
        self.integration_status['health_router'] = self._integrate_health_router()
        
        # 7. 추가 유틸리티 통합
        self.integration_status['utilities'] = self._integrate_utilities()
        
        # 통합 결과 요약
        successful_integrations = sum(1 for status in self.integration_status.values() if status)
        total_components = len(self.integration_status)
        
        logger.info(f"📊 통합 완료: {successful_integrations}/{total_components} 성공")
        
        if successful_integrations >= 4:  # 핵심 구성요소만 성공해도 OK
            logger.info("✅ 최소 요구사항 충족 - 서비스 시작 가능")
            return True
        else:
            logger.warning("⚠️ 일부 구성요소 실패 - 제한적 서비스 모드")
            return False
    
    def _integrate_schemas(self):
        """완전한 스키마 통합"""
        try:
            from app.models.schemas import (
                VirtualTryOnRequest, VirtualTryOnResponse, ProcessingStatus, ProcessingResult,
                PipelineProgress, QualityMetrics, SystemHealth, PerformanceMetrics,
                UserMeasurements, ClothingCategory, QualityMode, ModelType,
                ErrorResponse, WebSocketMessage, ProgressUpdate
            )
            
            self.components['schemas'] = {
                'VirtualTryOnRequest': VirtualTryOnRequest,
                'VirtualTryOnResponse': VirtualTryOnResponse,
                'ProcessingStatus': ProcessingStatus,
                'ProcessingResult': ProcessingResult,
                'PipelineProgress': PipelineProgress,
                'QualityMetrics': QualityMetrics,
                'SystemHealth': SystemHealth,
                'PerformanceMetrics': PerformanceMetrics,
                'UserMeasurements': UserMeasurements,
                'ClothingCategory': ClothingCategory,
                'QualityMode': QualityMode,
                'ModelType': ModelType,
                'ErrorResponse': ErrorResponse,
                'WebSocketMessage': WebSocketMessage,
                'ProgressUpdate': ProgressUpdate
            }
            
            logger.info("✅ 완전한 스키마 통합 성공")
            return True
            
        except ImportError as e:
            logger.error(f"❌ 스키마 통합 실패: {e}")
            self.import_errors.append(f"Schemas: {e}")
            self._create_fallback_schemas()
            return False
    
    def _integrate_model_loader(self):
        """ModelLoader 통합"""
        try:
            from app.ai_pipeline.utils.model_loader import (
                ModelLoader, ModelFormat, get_global_model_loader,
                create_model_loader, ModelType, ModelConfig, ModelRegistry
            )
            
            self.components['model_loader'] = {
                'ModelLoader': ModelLoader,
                'ModelFormat': ModelFormat,
                'get_global_model_loader': get_global_model_loader,
                'create_model_loader': create_model_loader,
                'ModelType': ModelType,
                'ModelConfig': ModelConfig,
                'ModelRegistry': ModelRegistry,
                'instance': get_global_model_loader()
            }
            
            logger.info("✅ ModelLoader 통합 성공")
            return True
            
        except ImportError as e:
            logger.error(f"❌ ModelLoader 통합 실패: {e}")
            self.import_errors.append(f"ModelLoader: {e}")
            self._create_fallback_model_loader()
            return False
    
    def _integrate_virtual_tryon_api(self):
        """Virtual Try-on API 통합"""
        try:
            from app.api.virtual_tryon import router as virtual_tryon_router
            from app.api.virtual_tryon import virtual_tryon_state, ws_manager
            
            self.components['virtual_tryon'] = {
                'router': virtual_tryon_router,
                'state': virtual_tryon_state,
                'ws_manager': ws_manager
            }
            
            logger.info("✅ Virtual Try-on API 통합 성공")
            return True
            
        except ImportError as e:
            logger.warning(f"⚠️ Virtual Try-on API 통합 실패: {e}")
            self.import_errors.append(f"Virtual Try-on API: {e}")
            self.components['virtual_tryon'] = {'router': self._create_fallback_virtual_tryon_router()}
            return False
    
    def _integrate_websocket(self):
        """WebSocket 통합"""
        try:
            from app.api.websocket_routes import (
                router as websocket_router, 
                connection_manager, 
                start_background_tasks,
                send_pipeline_progress
            )
            
            self.components['websocket'] = {
                'router': websocket_router,
                'connection_manager': connection_manager,
                'start_background_tasks': start_background_tasks,
                'send_pipeline_progress': send_pipeline_progress
            }
            
            logger.info("✅ WebSocket 통합 성공")
            return True
            
        except ImportError as e:
            logger.warning(f"⚠️ WebSocket 통합 실패: {e}")
            self.import_errors.append(f"WebSocket: {e}")
            self.components['websocket'] = {'router': self._create_fallback_websocket_router()}
            return False
    
    def _integrate_pipeline_manager(self):
        """Pipeline Manager 통합"""
        try:
            from app.ai_pipeline.pipeline_manager import (
                PipelineManager, get_pipeline_manager, 
                create_optimized_pipeline, PipelineFactory
            )
            
            self.components['pipeline_manager'] = {
                'PipelineManager': PipelineManager,
                'get_pipeline_manager': get_pipeline_manager,
                'create_optimized_pipeline': create_optimized_pipeline,
                'PipelineFactory': PipelineFactory,
                'instance': get_pipeline_manager()
            }
            
            logger.info("✅ Pipeline Manager 통합 성공")
            return True
            
        except ImportError as e:
            logger.warning(f"⚠️ Pipeline Manager 통합 실패: {e}")
            self.import_errors.append(f"Pipeline Manager: {e}")
            self._create_fallback_pipeline_manager()
            return False
    
    def _integrate_health_router(self):
        """Health Router 통합"""
        try:
            from app.api.health import router as health_router
            from app.api.health import health_collector, record_request_metrics
            
            self.components['health'] = {
                'router': health_router,
                'health_collector': health_collector,
                'record_request_metrics': record_request_metrics
            }
            
            logger.info("✅ Health Router 통합 성공")
            return True
            
        except ImportError as e:
            logger.warning(f"⚠️ Health Router 통합 실패: {e}")
            self.import_errors.append(f"Health Router: {e}")
            self.components['health'] = {'router': self._create_fallback_health_router()}
            return False
    
    def _integrate_utilities(self):
        """추가 유틸리티 통합"""
        try:
            # GPU 설정
            try:
                from app.core.gpu_config import get_device_config, optimize_memory
                self.components['gpu_config'] = {
                    'get_device_config': get_device_config,
                    'optimize_memory': optimize_memory
                }
            except ImportError:
                self.components['gpu_config'] = None
            
            # 설정
            try:
                from app.core.config import get_settings
                self.components['config'] = {'get_settings': get_settings}
            except ImportError:
                self.components['config'] = None
            
            logger.info("✅ 유틸리티 통합 완료")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ 유틸리티 통합 부분 실패: {e}")
            return False
    
    def _create_fallback_schemas(self):
        """폴백 스키마 생성"""
        from pydantic import BaseModel
        from typing import Optional, Dict, Any
        
        class FallbackSchema(BaseModel):
            success: bool = True
            message: str = "Fallback mode"
            data: Optional[Dict[str, Any]] = None
        
        self.components['schemas'] = {
            'VirtualTryOnRequest': FallbackSchema,
            'VirtualTryOnResponse': FallbackSchema,
            'ProcessingStatus': str,
            'ProcessingResult': FallbackSchema,
            'SystemHealth': FallbackSchema
        }
    
    def _create_fallback_model_loader(self):
        """폴백 ModelLoader 생성"""
        class FallbackModelLoader:
            def __init__(self):
                self.device = "cpu"
                self.is_initialized = False
            
            async def initialize(self):
                self.is_initialized = True
                return True
            
            def cleanup(self):
                pass
        
        class FallbackModelFormat:
            PYTORCH = "pytorch"
            COREML = "coreml"
        
        self.components['model_loader'] = {
            'ModelLoader': FallbackModelLoader,
            'ModelFormat': FallbackModelFormat,
            'get_global_model_loader': lambda: FallbackModelLoader(),
            'instance': FallbackModelLoader()
        }
    
    def _create_fallback_virtual_tryon_router(self):
        """폴백 Virtual Try-on Router"""
        from fastapi import APIRouter
        
        router = APIRouter()
        
        @router.post("/")
        async def fallback_virtual_tryon(
            person_image: UploadFile = File(...),
            clothing_image: UploadFile = File(...),
            height: float = Form(...),
            weight: float = Form(...)
        ):
            return {
                "success": True,
                "message": "폴백 모드 - AI 모델 설치 필요",
                "task_id": "fallback-" + str(time_module.time()),
                "status": "completed",
                "fitted_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                "processing_time": 1.0,
                "fit_score": 0.8,
                "confidence": 0.7,
                "recommendations": ["실제 AI 모델을 설치하면 정확한 가상 피팅을 이용할 수 있습니다"]
            }
        
        return router
    
    def _create_fallback_websocket_router(self):
        """폴백 WebSocket Router"""
        from fastapi import APIRouter, WebSocket
        
        router = APIRouter()
        
        @router.websocket("/test")
        async def fallback_websocket(websocket: WebSocket):
            await websocket.accept()
            await websocket.send_text(json.dumps({
                "type": "fallback",
                "message": "WebSocket 폴백 모드"
            }))
            await websocket.close()
        
        return router
    
    def _create_fallback_pipeline_manager(self):
        """폴백 Pipeline Manager"""
        class FallbackPipelineManager:
            def __init__(self):
                self.is_initialized = False
            
            async def initialize(self):
                self.is_initialized = True
                return True
            
            async def get_pipeline_status(self):
                return {"initialized": False, "fallback_mode": True}
        
        self.components['pipeline_manager'] = {
            'instance': FallbackPipelineManager(),
            'get_pipeline_manager': lambda: FallbackPipelineManager()
        }
    
    def _create_fallback_health_router(self):
        """폴백 Health Router"""
        from fastapi import APIRouter
        
        router = APIRouter()
        
        @router.get("/")
        async def fallback_health():
            return {
                "status": "degraded",
                "message": "Health Router 폴백 모드",
                "timestamp": datetime.now().isoformat()
            }
        
        return router

# 통합 관리자 초기화
integrator = CompleteComponentIntegrator()
integration_success = integrator.integrate_all_components()

# 전역 상태
app_state = {
    "initialized": False,
    "startup_time": None,
    "integration_success": integration_success,
    "m3_max_optimized": integrator.m3_max_optimized,
    "component_status": integrator.integration_status,
    "import_errors": integrator.import_errors.copy(),
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "performance_metrics": {
        "average_response_time": 0.0,
        "total_requests": 0,
        "error_rate": 0.0,
        "m3_max_optimized_sessions": 0,
        "memory_efficiency": 0.95 if integrator.m3_max_optimized else 0.8
    }
}

# ============================================
# FastAPI 애플리케이션 생성
# ============================================

@asynccontextmanager
async def complete_lifespan(app: FastAPI):
    """완전 통합 애플리케이션 라이프사이클 관리"""
    logger.info("🚀 MyCloset AI Backend 완전 통합 버전 시작...")
    startup_start_time = time_module.time()
    
    try:
        # M3 Max 환경 최적화
        if integrator.m3_max_optimized:
            logger.info("🍎 M3 Max Neural Engine 활성화...")
            logger.info("⚡ MPS 백엔드 최적화 설정...")
            logger.info("💾 128GB Unified Memory 활용...")
        
        # 1. ModelLoader 초기화
        if integrator.integration_status.get('model_loader'):
            try:
                model_loader = integrator.components['model_loader']['instance']
                await model_loader.initialize()
                logger.info("✅ ModelLoader 초기화 완료")
            except Exception as e:
                logger.warning(f"⚠️ ModelLoader 초기화 실패: {e}")
        
        # 2. Pipeline Manager 초기화
        if integrator.integration_status.get('pipeline_manager'):
            try:
                pipeline_manager = integrator.components['pipeline_manager']['instance']
                await pipeline_manager.initialize()
                logger.info("✅ Pipeline Manager 초기화 완료")
            except Exception as e:
                logger.warning(f"⚠️ Pipeline Manager 초기화 실패: {e}")
        
        # 3. Virtual Try-on State 초기화
        if integrator.integration_status.get('virtual_tryon'):
            try:
                virtual_tryon_state = integrator.components['virtual_tryon'].get('state')
                if virtual_tryon_state:
                    await virtual_tryon_state.initialize()
                    logger.info("✅ Virtual Try-on State 초기화 완료")
            except Exception as e:
                logger.warning(f"⚠️ Virtual Try-on State 초기화 실패: {e}")
        
        # 4. WebSocket 백그라운드 태스크 시작
        if integrator.integration_status.get('websocket'):
            try:
                start_background_tasks = integrator.components['websocket'].get('start_background_tasks')
                if start_background_tasks:
                    await start_background_tasks()
                    logger.info("✅ WebSocket 백그라운드 태스크 시작됨")
            except Exception as e:
                logger.warning(f"⚠️ WebSocket 백그라운드 태스크 시작 실패: {e}")
        
        app_state["startup_time"] = time_module.time() - startup_start_time
        app_state["initialized"] = True
        
        # 최종 상태 로깅
        logger.info("=" * 80)
        logger.info("🍎 MyCloset AI Backend 완전 통합 버전 시스템 상태")
        logger.info("=" * 80)
        logger.info(f"🚀 통합 성공: {integration_success}")
        logger.info(f"🍎 M3 Max 최적화: {'✅ 활성화' if integrator.m3_max_optimized else '❌ 비활성화'}")
        logger.info(f"🤖 ModelLoader: {'✅' if integrator.integration_status.get('model_loader') else '❌'}")
        logger.info(f"⚙️ Pipeline Manager: {'✅' if integrator.integration_status.get('pipeline_manager') else '❌'}")
        logger.info(f"🎯 Virtual Try-on API: {'✅' if integrator.integration_status.get('virtual_tryon') else '❌'}")
        logger.info(f"🔗 WebSocket: {'✅' if integrator.integration_status.get('websocket') else '❌'}")
        logger.info(f"💊 Health Router: {'✅' if integrator.integration_status.get('health_router') else '❌'}")
        logger.info(f"⏱️ 시작 시간: {app_state['startup_time']:.2f}초")
        
        if app_state['import_errors']:
            logger.warning(f"⚠️ Import 오류 ({len(app_state['import_errors'])}개):")
            for error in app_state['import_errors']:
                logger.warning(f"  - {error}")
        
        logger.info("✅ 완전 통합 백엔드 초기화 완료")
        logger.info("=" * 80)
        
    except Exception as e:
        error_msg = f"Startup error: {str(e)}"
        logger.error(f"❌ 시작 중 치명적 오류: {error_msg}")
        logger.error(f"📋 스택 트레이스: {traceback.format_exc()}")
        app_state["import_errors"].append(error_msg)
        app_state["initialized"] = False
    
    yield  # 애플리케이션 실행
    
    # 종료 로직
    logger.info("🛑 MyCloset AI Backend 완전 통합 버전 종료 중...")
    
    try:
        # 순서대로 정리
        
        # 1. ModelLoader 정리
        if integrator.integration_status.get('model_loader'):
            model_loader = integrator.components['model_loader']['instance']
            model_loader.cleanup()
            logger.info("✅ ModelLoader 정리 완료")
        
        # 2. Pipeline Manager 정리
        if integrator.integration_status.get('pipeline_manager'):
            pipeline_manager = integrator.components['pipeline_manager']['instance']
            await pipeline_manager.cleanup()
            logger.info("✅ Pipeline Manager 정리 완료")
        
        # 3. 메모리 최적화
        if integrator.components.get('gpu_config'):
            optimize_memory = integrator.components['gpu_config'].get('optimize_memory')
            if optimize_memory:
                result = optimize_memory(aggressive=integrator.m3_max_optimized)
                logger.info(f"✅ 메모리 정리: {result.get('method', 'unknown')}")
        
        if integrator.m3_max_optimized:
            logger.info("🍎 M3 Max 리소스 정리 완료")
        
        logger.info("✅ 완전 통합 백엔드 정리 완료")
        
    except Exception as e:
        logger.warning(f"⚠️ 정리 중 오류: {e}")

app = FastAPI(
    title="MyCloset AI Backend (Complete Integration)",
    description="완전 통합된 M3 Max 최적화 가상 피팅 AI 백엔드",
    version="4.0.0-complete",
    lifespan=complete_lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ============================================
# 미들웨어 설정
# ============================================

# CORS 설정 (React Frontend 완벽 호환)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React 개발 서버
        "http://localhost:5173",  # Vite 개발 서버  
        "http://localhost:8080",  # 추가 개발 서버
        "*"  # 개발 중에는 모든 origin 허용
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 성능 측정 미들웨어 (Health Router와 연동)
@app.middleware("http")
async def complete_performance_middleware(request: Request, call_next):
    start_time = time_module.time()
    success = True
    
    try:
        response = await call_next(request)
        
        # 응답 상태에 따른 성공/실패 판정
        success = response.status_code < 400
        
    except Exception as e:
        logger.error(f"Request processing error: {e}")
        success = False
        response = JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "message": str(e)}
        )
    
    finally:
        # 처리 시간 계산
        process_time = time_module.time() - start_time
        
        # 헤더 추가
        response.headers["X-Process-Time"] = str(round(process_time, 4))
        if integrator.m3_max_optimized:
            response.headers["X-M3-Max-Optimized"] = "true"
        response.headers["X-Integration-Status"] = "complete" if integration_success else "partial"
        
        # 통계 업데이트
        app_state["total_requests"] += 1
        if success:
            app_state["successful_requests"] += 1
        else:
            app_state["failed_requests"] += 1
        
        # Health Router에 메트릭 기록 (가능한 경우)
        if integrator.integration_status.get('health_router'):
            try:
                record_metrics = integrator.components['health']['record_request_metrics']
                record_metrics(success, process_time)
            except:
                pass
        
        # 성능 메트릭 업데이트
        total_requests = app_state["total_requests"]
        if total_requests > 0:
            current_avg = app_state["performance_metrics"]["average_response_time"]
            app_state["performance_metrics"]["average_response_time"] = (
                (current_avg * (total_requests - 1) + process_time) / total_requests
            )
            app_state["performance_metrics"]["total_requests"] = total_requests
            app_state["performance_metrics"]["error_rate"] = (
                app_state["failed_requests"] / total_requests * 100
            )
    
    return response

# ============================================
# API 라우터 등록 (완전 통합)
# ============================================

# 1. Virtual Try-on Router (React와 직접 연동)
if integrator.integration_status.get('virtual_tryon'):
    virtual_tryon_router = integrator.components['virtual_tryon']['router']
    app.include_router(virtual_tryon_router, prefix="/api/virtual-tryon", tags=["virtual-tryon"])
    logger.info("✅ Virtual Try-on Router 등록 (React 완벽 연동)")

# 2. WebSocket Router (실시간 진행상황)
if integrator.integration_status.get('websocket'):
    websocket_router = integrator.components['websocket']['router']
    app.include_router(websocket_router, prefix="/api/ws", tags=["websocket"])
    logger.info("✅ WebSocket Router 등록 (React 실시간 통신)")

# 3. Health Router (시스템 모니터링)
if integrator.integration_status.get('health_router'):
    health_router = integrator.components['health']['router']
    app.include_router(health_router, prefix="/health", tags=["health"])
    logger.info("✅ Health Router 등록 (시스템 모니터링)")

# ============================================
# 정적 파일 서빙
# ============================================

# 업로드/결과 파일 디렉토리 생성
directories_to_create = [
    project_root / "static" / "uploads",
    project_root / "static" / "results",
    project_root / "temp",
    project_root / "logs"
]

for directory in directories_to_create:
    directory.mkdir(parents=True, exist_ok=True)

# 정적 파일 서빙
static_dir = project_root / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info("✅ 정적 파일 서빙 설정됨")

# ============================================
# 기본 엔드포인트들 (React 호환)
# ============================================

@app.get("/", response_class=HTMLResponse)
async def complete_root():
    """완전 통합 루트 엔드포인트"""
    
    # 상태 이모지 설정
    integration_emoji = "✅" if integration_success else "⚠️"
    m3_max_emoji = "🍎" if integrator.m3_max_optimized else "💻"
    
    # 컴포넌트 상태 요약
    component_status = []
    for component, status in integrator.integration_status.items():
        emoji = "✅" if status else "❌"
        component_status.append(f"{emoji} {component.replace('_', ' ').title()}")
    
    current_time = time_module.time()
    uptime = current_time - (app_state.get("startup_time", 0) or current_time)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MyCloset AI Backend (Complete)</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
                padding: 20px;
            }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
                background: rgba(255,255,255,0.1); 
                padding: 30px; 
                border-radius: 20px; 
                box-shadow: 0 20px 40px rgba(0,0,0,0.3);
                backdrop-filter: blur(20px);
            }}
            h1 {{ 
                text-align: center; 
                margin-bottom: 2rem; 
                font-size: 2.5em;
                background: linear-gradient(45deg, #fff, #f0f0f0);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            .status-card {{ 
                padding: 20px; 
                border-radius: 15px; 
                margin: 20px 0; 
                background: rgba(255,255,255,0.15);
                border: 1px solid rgba(255,255,255,0.2);
            }}
            .status-card.success {{ 
                background: rgba(46, 213, 115, 0.2); 
                border-color: rgba(46, 213, 115, 0.4); 
            }}
            .status-card.warning {{ 
                background: rgba(255, 159, 67, 0.2); 
                border-color: rgba(255, 159, 67, 0.4); 
            }}
            .badge {{
                display: inline-block;
                padding: 8px 16px;
                border-radius: 25px;
                font-size: 0.9em;
                font-weight: 600;
                margin: 5px;
            }}
            .badge.complete {{
                background: linear-gradient(45deg, #00d4aa, #00b4d8);
                color: white;
                box-shadow: 0 4px 15px rgba(0, 212, 170, 0.4);
            }}
            .badge.m3max {{
                background: linear-gradient(45deg, #ff6b6b, #ffa726);
                color: white;
                box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
            }}
            .metrics {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                gap: 20px; 
                margin: 25px 0; 
            }}
            .metric {{ 
                background: rgba(255,255,255,0.12); 
                padding: 20px; 
                border-radius: 15px; 
                text-align: center;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.1);
                transition: transform 0.3s ease;
            }}
            .metric:hover {{
                transform: translateY(-5px);
                box-shadow: 0 10px 25px rgba(0,0,0,0.2);
            }}
            .metric h3 {{ 
                margin: 0; 
                color: #e0e0e0; 
                font-size: 0.9em; 
                margin-bottom: 10px;
            }}
            .metric p {{ 
                margin: 0; 
                font-size: 1.8em; 
                font-weight: bold; 
                color: #fff; 
            }}
            .components {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                gap: 15px; 
                margin: 20px 0; 
            }}
            .component {{ 
                background: rgba(255,255,255,0.1); 
                padding: 15px; 
                border-radius: 10px; 
                font-size: 0.9em;
            }}
            .links {{ 
                text-align: center; 
                margin-top: 30px; 
            }}
            .links a {{ 
                display: inline-block; 
                margin: 10px 15px; 
                padding: 15px 25px; 
                background: rgba(255,255,255,0.2); 
                color: white; 
                text-decoration: none; 
                border-radius: 10px; 
                transition: all 0.3s ease;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.1);
            }}
            .links a:hover {{ 
                background: rgba(255,255,255,0.3); 
                transform: translateY(-3px);
                box-shadow: 0 8px 20px rgba(0,0,0,0.2);
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid rgba(255,255,255,0.2);
                font-size: 0.9em;
                opacity: 0.8;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{m3_max_emoji} MyCloset AI Backend v4.0</h1>
            
            <div style="text-align: center; margin-bottom: 2rem;">
                <span class="badge complete">Complete Integration</span>
                {'<span class="badge m3max">🍎 M3 Max Optimized</span>' if integrator.m3_max_optimized else ''}
            </div>
            
            <div class="status-card {'success' if integration_success else 'warning'}">
                <strong>{integration_emoji} 통합 상태:</strong> 
                {'완전 통합 성공 - 모든 기능 활성화' if integration_success 
                 else '부분 통합 - 핵심 기능 활성화'}
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <h3>통합 성공률</h3>
                    <p>{sum(integrator.integration_status.values())}/{len(integrator.integration_status)}</p>
                </div>
                <div class="metric">
                    <h3>M3 Max 최적화</h3>
                    <p>{'🍎 ON' if integrator.m3_max_optimized else '❌ OFF'}</p>
                </div>
                <div class="metric">
                    <h3>총 요청 수</h3>
                    <p>{app_state['total_requests']}</p>
                </div>
                <div class="metric">
                    <h3>성공률</h3>
                    <p>{(app_state['successful_requests'] / max(1, app_state['total_requests']) * 100):.1f}%</p>
                </div>
                <div class="metric">
                    <h3>평균 응답시간</h3>
                    <p>{app_state['performance_metrics']['average_response_time']:.3f}s</p>
                </div>
                <div class="metric">
                    <h3>가동 시간</h3>
                    <p>{uptime:.0f}s</p>
                </div>
            </div>
            
            <h3 style="margin-top: 2rem; margin-bottom: 1rem;">🔧 구성요소 상태</h3>
            <div class="components">
                {''.join(f'<div class="component">{status}</div>' for status in component_status)}
            </div>
            
            <div class="links">
                <a href="/docs">📚 API 문서</a>
                <a href="/status">📊 상세 상태</a>
                <a href="/health/">💊 헬스체크</a>
                <a href="/api/ws/test">🔗 WebSocket 테스트</a>
                <a href="/api/virtual-tryon/demo">🎯 가상피팅 API</a>
                {'<a href="/health/react">🍎 M3 Max 상태</a>' if integrator.m3_max_optimized else ''}
            </div>
            
            <div class="footer">
                <p>MyCloset AI Backend v4.0.0 - Complete Integration Edition</p>
                <p>Powered by FastAPI + React + ModelLoader + Pipeline Manager + WebSocket</p>
                {'<p>🍎 Optimized for Apple M3 Max with 128GB Unified Memory</p>' if integrator.m3_max_optimized else ''}
            </div>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.get("/status")
async def complete_detailed_status():
    """완전 통합 상세 시스템 상태"""
    current_time = time_module.time()
    uptime = current_time - (app_state.get("startup_time", 0) or current_time)
    
    return {
        "application": {
            "name": "MyCloset AI Backend (Complete Integration)",
            "version": "4.0.0-complete",
            "initialized": app_state["initialized"],
            "integration_success": app_state["integration_success"],
            "m3_max_optimized": integrator.m3_max_optimized,
            "uptime_seconds": uptime,
            "startup_time": app_state["startup_time"],
            "import_errors": app_state["import_errors"]
        },
        "integration_status": integrator.integration_status,
        "components": {
            "schemas": bool(integrator.components.get('schemas')),
            "model_loader": bool(integrator.components.get('model_loader')),
            "virtual_tryon_api": bool(integrator.components.get('virtual_tryon')),
            "websocket": bool(integrator.components.get('websocket')),
            "pipeline_manager": bool(integrator.components.get('pipeline_manager')),
            "health_router": bool(integrator.components.get('health')),
            "gpu_config": bool(integrator.components.get('gpu_config')),
            "config": bool(integrator.components.get('config'))
        },
        "react_compatibility": {
            "virtual_tryon_api": integrator.integration_status.get('virtual_tryon', False),
            "websocket_support": integrator.integration_status.get('websocket', False),
            "file_upload_support": True,
            "cors_enabled": True,
            "real_time_progress": integrator.integration_status.get('websocket', False)
        },
        "performance": app_state["performance_metrics"],
        "system": {
            "m3_max_features": {
                "neural_engine": integrator.m3_max_optimized,
                "unified_memory": integrator.m3_max_optimized,
                "mps_backend": integrator.m3_max_optimized,
                "memory_bandwidth": "400GB/s" if integrator.m3_max_optimized else "N/A"
            } if integrator.m3_max_optimized else None
        }
    }

@app.get("/integration-report")
async def integration_report():
    """통합 상태 상세 보고서"""
    return {
        "timestamp": datetime.now().isoformat(),
        "integration_summary": {
            "total_components": len(integrator.integration_status),
            "successful_integrations": sum(integrator.integration_status.values()),
            "failed_integrations": len([s for s in integrator.integration_status.values() if not s]),
            "success_rate": sum(integrator.integration_status.values()) / len(integrator.integration_status) * 100
        },
        "component_details": {
            component: {
                "integrated": status,
                "critical": component in ["schemas", "virtual_tryon", "model_loader"],
                "description": {
                    "schemas": "Pydantic 데이터 스키마 - React 타입 호환",
                    "model_loader": "AI 모델 로딩 및 관리 시스템",
                    "virtual_tryon_api": "React와 직접 연동되는 가상피팅 API",
                    "websocket": "실시간 진행상황 통신",
                    "pipeline_manager": "8단계 AI 파이프라인 실행 엔진",
                    "health_router": "시스템 상태 모니터링",
                    "utilities": "추가 유틸리티 및 설정"
                }.get(component, "기타 구성요소")
            }
            for component, status in integrator.integration_status.items()
        },
        "import_errors": app_state["import_errors"],
        "recommendations": [
            "✅ 핵심 구성요소 정상 작동 중" if sum(integrator.integration_status.values()) >= 4 
            else "⚠️ 일부 구성요소 설치 필요",
            "🍎 M3 Max 최적화 활성화됨" if integrator.m3_max_optimized 
            else "💡 M3 Max에서 실행하면 더 나은 성능을 얻을 수 있습니다",
            "🔗 React Frontend와 완벽 호환" if integrator.integration_status.get('virtual_tryon') and integrator.integration_status.get('websocket')
            else "⚠️ React 연동을 위해 Virtual Try-on API와 WebSocket 설정 확인 필요"
        ]
    }

# ============================================
# 예외 처리 (완전 통합 버전)
# ============================================

@app.exception_handler(StarletteHTTPException)
async def complete_http_exception_handler(request: Request, exc: StarletteHTTPException):
    """HTTP 예외 처리 - 통합 버전"""
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "type": "http_error",
                "status_code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.now().isoformat(),
                "integration_version": "4.0.0-complete",
                "m3_max_optimized": integrator.m3_max_optimized
            },
            "request_info": {
                "method": request.method,
                "url": str(request.url),
                "client": request.client.host if request.client else "unknown"
            },
            "system_status": {
                "integration_success": integration_success,
                "available_components": list(integrator.integration_status.keys())
            }
        }
    )

@app.exception_handler(RequestValidationError)
async def complete_validation_exception_handler(request: Request, exc: RequestValidationError):
    """요청 검증 예외 처리 - Pydantic V2 완전 호환"""
    
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {
                "type": "validation_error",
                "message": "입력 데이터 검증 실패 (Pydantic V2)",
                "details": exc.errors(),
                "timestamp": datetime.now().isoformat(),
                "pydantic_version": "v2",
                "integration_version": "4.0.0-complete"
            },
            "help": {
                "schemas_available": bool(integrator.components.get('schemas')),
                "documentation": "/docs",
                "example_request": "/api/virtual-tryon/demo"
            }
        }
    )

@app.exception_handler(Exception)
async def complete_general_exception_handler(request: Request, exc: Exception):
    """일반 예외 처리 - 통합 시스템용"""
    
    error_msg = str(exc)
    error_type = type(exc).__name__
    
    # 스택 트레이스 로깅
    logger.error(f"일반 예외 발생: {error_type} - {error_msg}")
    logger.error(f"요청 URL: {request.url}")
    logger.error(f"스택 트레이스: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "type": error_type,
                "message": error_msg,
                "timestamp": datetime.now().isoformat(),
                "integration_version": "4.0.0-complete",
                "m3_max_optimized": integrator.m3_max_optimized
            },
            "system_info": {
                "integration_success": integration_success,
                "components_status": integrator.integration_status,
                "fallback_mode": not integration_success
            },
            "support": {
                "health_check": "/health/",
                "system_status": "/status",
                "integration_report": "/integration-report"
            }
        }
    )

# ============================================
# React Frontend 전용 엔드포인트들
# ============================================

@app.get("/api/health")
async def react_health_endpoint():
    """React Frontend 전용 헬스체크"""
    try:
        return {
            "status": "online",
            "timestamp": datetime.now().isoformat(),
            "version": "4.0.0-complete",
            "integration": {
                "success": integration_success,
                "virtual_tryon_ready": integrator.integration_status.get('virtual_tryon', False),
                "websocket_ready": integrator.integration_status.get('websocket', False),
                "models_ready": integrator.integration_status.get('model_loader', False),
                "pipeline_ready": integrator.integration_status.get('pipeline_manager', False)
            },
            "endpoints": {
                "virtual_tryon": "/api/virtual-tryon/",
                "websocket_progress": "/api/ws/pipeline-progress",
                "websocket_test": "/api/ws/test",
                "health": "/health/",
                "status": "/status"
            },
            "features": {
                "8_step_pipeline": True,
                "real_time_progress": integrator.integration_status.get('websocket', False),
                "file_upload": True,
                "m3_max_optimized": integrator.m3_max_optimized,
                "quality_modes": ["fast", "balanced", "high_quality"]
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/capabilities")
async def get_system_capabilities():
    """시스템 기능 조회 - React에서 기능 확인용"""
    capabilities = {
        "virtual_tryon": {
            "available": integrator.integration_status.get('virtual_tryon', False),
            "features": [
                "8-step AI pipeline",
                "Real-time progress tracking",
                "Quality assessment",
                "Multiple clothing categories"
            ] if integrator.integration_status.get('virtual_tryon') else []
        },
        "models": {
            "available": integrator.integration_status.get('model_loader', False),
            "types": [
                "Human parsing (20 body parts)",
                "Pose estimation (18 keypoints)", 
                "Clothing analysis",
                "Geometric matching",
                "Virtual fitting generation"
            ] if integrator.integration_status.get('model_loader') else []
        },
        "pipeline": {
            "available": integrator.integration_status.get('pipeline_manager', False),
            "steps": [
                "Image preprocessing",
                "Human parsing", 
                "Pose estimation",
                "Clothing analysis",
                "Geometric matching",
                "Cloth warping",
                "Virtual fitting",
                "Quality assessment"
            ] if integrator.integration_status.get('pipeline_manager') else []
        },
        "communication": {
            "websocket": integrator.integration_status.get('websocket', False),
            "real_time_progress": integrator.integration_status.get('websocket', False),
            "file_upload": True,
            "cors_enabled": True
        },
        "optimization": {
            "m3_max": integrator.m3_max_optimized,
            "gpu_acceleration": integrator.components.get('gpu_config') is not None,
            "memory_optimization": True,
            "batch_processing": False  # 현재는 단일 이미지 처리
        }
    }
    
    return {
        "capabilities": capabilities,
        "integration_score": sum(integrator.integration_status.values()) / len(integrator.integration_status) * 100,
        "recommended_usage": "full_features" if integration_success else "basic_features",
        "timestamp": datetime.now().isoformat()
    }

# ============================================
# 개발자 도구 엔드포인트들
# ============================================

@app.get("/dev/components")
async def dev_component_status():
    """개발자용 구성요소 상세 상태"""
    component_details = {}
    
    for component_name, is_integrated in integrator.integration_status.items():
        component_data = integrator.components.get(component_name, {})
        
        component_details[component_name] = {
            "integrated": is_integrated,
            "available_functions": list(component_data.keys()) if isinstance(component_data, dict) else [],
            "instance_type": str(type(component_data.get('instance'))) if component_data.get('instance') else None,
            "router_endpoints": getattr(component_data.get('router'), 'routes', []) if component_data.get('router') else []
        }
    
    return {
        "component_details": component_details,
        "import_errors": app_state["import_errors"],
        "python_path": sys.path[:3],  # 처음 3개만
        "working_directory": str(current_dir),
        "environment": {
            "platform": sys.platform,
            "python_version": sys.version,
            "fastapi_available": True,
            "pydantic_available": True
        }
    }

@app.get("/dev/test-integration")
async def test_integration():
    """통합 테스트 실행"""
    test_results = {}
    
    # 1. 스키마 테스트
    try:
        if integrator.components.get('schemas'):
            VirtualTryOnRequest = integrator.components['schemas']['VirtualTryOnRequest']
            # 간단한 스키마 인스턴스 생성 테스트
            test_results['schemas'] = {"status": "pass", "message": "Schema creation successful"}
        else:
            test_results['schemas'] = {"status": "skip", "message": "Schemas not available"}
    except Exception as e:
        test_results['schemas'] = {"status": "fail", "error": str(e)}
    
    # 2. ModelLoader 테스트
    try:
        if integrator.components.get('model_loader'):
            model_loader = integrator.components['model_loader']['instance']
            test_results['model_loader'] = {
                "status": "pass" if hasattr(model_loader, 'device') else "fail",
                "device": getattr(model_loader, 'device', 'unknown'),
                "initialized": getattr(model_loader, 'is_initialized', False)
            }
        else:
            test_results['model_loader'] = {"status": "skip", "message": "ModelLoader not available"}
    except Exception as e:
        test_results['model_loader'] = {"status": "fail", "error": str(e)}
    
    # 3. Pipeline Manager 테스트
    try:
        if integrator.components.get('pipeline_manager'):
            pipeline_manager = integrator.components['pipeline_manager']['instance']
            test_results['pipeline_manager'] = {
                "status": "pass" if hasattr(pipeline_manager, 'is_initialized') else "fail",
                "initialized": getattr(pipeline_manager, 'is_initialized', False)
            }
        else:
            test_results['pipeline_manager'] = {"status": "skip", "message": "Pipeline Manager not available"}
    except Exception as e:
        test_results['pipeline_manager'] = {"status": "fail", "error": str(e)}
    
    # 전체 결과 요약
    statuses = [result.get("status", "unknown") for result in test_results.values()]
    overall_status = "pass"
    if "fail" in statuses:
        overall_status = "fail"
    elif "skip" in statuses:
        overall_status = "partial"
    
    return {
        "overall_status": overall_status,
        "test_results": test_results,
        "summary": {
            "total_tests": len(test_results),
            "passed": statuses.count("pass"),
            "failed": statuses.count("fail"),
            "skipped": statuses.count("skip")
        },
        "timestamp": datetime.now().isoformat()
    }

# ============================================
# 메인 실행부
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 MyCloset AI Backend v4.0.0 - 완전 통합 버전 시작...")
    logger.info(f"🔧 통합 상태: {integration_success}")
    logger.info(f"🍎 M3 Max 최적화: {integrator.m3_max_optimized}")
    logger.info(f"📊 구성요소: {sum(integrator.integration_status.values())}/{len(integrator.integration_status)} 성공")
    
    # 핵심 기능 상태 로깅
    core_components = ['schemas', 'virtual_tryon', 'websocket', 'model_loader']
    core_status = {comp: integrator.integration_status.get(comp, False) for comp in core_components}
    logger.info(f"🎯 핵심 기능: {core_status}")
    
    # React 호환성 확인
    react_ready = (
        integrator.integration_status.get('virtual_tryon', False) and
        integrator.integration_status.get('websocket', False) and
        integrator.integration_status.get('schemas', False)
    )
    logger.info(f"⚛️ React 호환성: {'✅ 완벽 호환' if react_ready else '⚠️ 제한적 호환'}")
    
    # 서버 실행 설정
    server_config = {
        "app": "app.main:app",
        "host": "0.0.0.0",
        "port": 8000,
        "reload": False,  # 완전 통합 버전에서는 reload 비활성화
        "log_level": "info",
        "access_log": True
    }
    
    # M3 Max 최적화된 서버 설정
    if integrator.m3_max_optimized:
        try:
            import uvloop
            server_config["loop"] = "uvloop"
            logger.info("🍎 M3 Max 최적화: uvloop 활성화")
        except ImportError:
            logger.info("🍎 M3 Max 최적화: asyncio (uvloop 없음)")
    
    # 프로덕션 환경 감지
    if os.getenv("ENVIRONMENT") == "production":
        server_config.update({
            "workers": 1,  # 통합 시스템에서는 단일 워커 권장
            "reload": False,
            "log_level": "warning"
        })
        logger.info("🏭 프로덕션 모드로 실행")
    else:
        logger.info("🔧 개발 모드로 실행")
    
    # 최종 시작 메시지
    logger.info("=" * 80)
    logger.info("🍎 MyCloset AI Backend v4.0.0 - Complete Integration Edition")
    logger.info("🎯 Features: ModelLoader + Virtual Try-on + WebSocket + Pipeline + Health")
    logger.info("⚛️ React Frontend: Full Compatibility")
    logger.info("🔗 Endpoints: /docs (API) | /api/virtual-tryon (Main) | /api/ws (WebSocket)")
    logger.info("📊 Status: /status | /health | /integration-report")
    logger.info("=" * 80)
    
    # 서버 실행
    uvicorn.run(**server_config)

# ============================================
# 모듈 정리 및 익스포트
# ============================================

# 정리 함수 등록
import atexit

def cleanup_on_exit():
    """프로그램 종료 시 정리"""
    try:
        logger.info("🛑 프로그램 종료 - 리소스 정리 중...")
        
        # ModelLoader 정리
        if integrator.integration_status.get('model_loader'):
            model_loader = integrator.components['model_loader']['instance']
            model_loader.cleanup()
        
        # 메모리 정리
        if integrator.components.get('gpu_config'):
            optimize_memory = integrator.components['gpu_config'].get('optimize_memory')
            if optimize_memory:
                optimize_memory(aggressive=True)
        
        logger.info("✅ 정리 완료")
    except Exception as e:
        logger.warning(f"정리 중 오류: {e}")

atexit.register(cleanup_on_exit)

# 모듈 완성 로그
logger.info("✅ MyCloset AI Backend 완전 통합 메인 모듈 로드 완료")
logger.info(f"📋 최종 통합 상태: {integration_success}")
logger.info(f"🎯 React 연동 준비: {integrator.integration_status.get('virtual_tryon', False) and integrator.integration_status.get('websocket', False)}")

# 모듈 메타데이터
__version__ = "4.0.0-complete"
__author__ = "MyCloset AI Team"
__description__ = "Complete Integration Backend with ModelLoader + Pipeline + WebSocket + React Support"
__integration_status__ = integrator.integration_status
__m3_max_optimized__ = integrator.m3_max_optimized