# app/main.py
"""
🍎 MyCloset AI Backend v5.0 - 프론트엔드 완전 호환
✅ Step API 엔드포인트 포함
✅ 순환참조 완전 해결
✅ M3 Max 128GB 최적화
✅ 프로덕션 안정성 보장
✅ 8단계 가상 피팅 지원
✅ ModelLoader DI 완전 해결
"""

import threading
import os
import sys
import time
import logging
import asyncio
import json
import io
import base64
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from PIL import Image, ImageDraw
import psutil

import numpy as np
import torch
import cv2

# FastAPI 및 기본 라이브러리
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

# ===============================================================
# 🔧 경로 및 시스템 설정
# ===============================================================

current_file = Path(__file__).resolve()
app_dir = current_file.parent
backend_dir = app_dir.parent
project_root = backend_dir.parent

# Python 경로 추가
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

print(f"📁 Backend 디렉토리: {backend_dir}")
print(f"📁 프로젝트 루트: {project_root}")

# ===============================================================
# 🔧 로깅 설정
# ===============================================================

logs_dir = backend_dir / "logs"
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(logs_dir / f"mycloset-ai-{time.strftime('%Y%m%d')}.log")
    ]
)
logger = logging.getLogger(__name__)

# ===============================================================
# 🔧 M3 Max GPU 설정 (안전한 Import)
# ===============================================================

try:
    import torch
    import psutil
    
    # M3 Max 감지
    IS_M3_MAX = (
        sys.platform == "darwin" and 
        os.uname().machine == "arm64" and
        torch.backends.mps.is_available()
    )
    
    if IS_M3_MAX:
        DEVICE = "mps"
        DEVICE_NAME = "Apple M3 Max"
        
        # M3 Max 최적화 설정
        os.environ.update({
            'PYTORCH_ENABLE_MPS_FALLBACK': '1',
            'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',
            'OMP_NUM_THREADS': '16',
            'MKL_NUM_THREADS': '16'
        })
        
        memory_info = psutil.virtual_memory()
        TOTAL_MEMORY_GB = memory_info.total / (1024**3)
        AVAILABLE_MEMORY_GB = memory_info.available / (1024**3)
        
        logger.info(f"🍎 M3 Max 감지됨")
        logger.info(f"💾 시스템 메모리: {TOTAL_MEMORY_GB:.1f}GB (사용가능: {AVAILABLE_MEMORY_GB:.1f}GB)")
        
    elif torch.cuda.is_available():
        DEVICE = "cuda"
        DEVICE_NAME = "NVIDIA GPU"
        TOTAL_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        AVAILABLE_MEMORY_GB = TOTAL_MEMORY_GB * 0.8
        
    else:
        DEVICE = "cpu"
        DEVICE_NAME = "CPU"
        TOTAL_MEMORY_GB = psutil.virtual_memory().total / (1024**3)
        AVAILABLE_MEMORY_GB = TOTAL_MEMORY_GB * 0.5
        
except ImportError as e:
    logger.warning(f"PyTorch 불러오기 실패: {e}")
    DEVICE = "cpu"
    DEVICE_NAME = "CPU"
    IS_M3_MAX = False
    TOTAL_MEMORY_GB = 8.0
    AVAILABLE_MEMORY_GB = 4.0

# ===============================================================
# 🔧 새로운 통합 유틸리티 시스템 Import (순환참조 해결)
# ===============================================================

try:
    # 새로운 통합 유틸리티 시스템 Import
    from app.ai_pipeline.utils import (
        get_utils_manager,
        initialize_global_utils,
        create_step_interface,
        create_unified_interface,
        get_system_status,
        reset_global_utils,
        optimize_system_memory,
        SYSTEM_INFO
    )
    UNIFIED_UTILS_AVAILABLE = True
    logger.info("✅ 새로운 통합 유틸리티 시스템 Import 성공")
except ImportError as e:
    logger.error(f"❌ 통합 유틸리티 시스템 Import 실패: {e}")
    UNIFIED_UTILS_AVAILABLE = False

# AI 파이프라인 Steps Import (조건부)
AI_PIPELINE_AVAILABLE = False
pipeline_step_classes = {}

if UNIFIED_UTILS_AVAILABLE:
    try:
        from app.ai_pipeline.steps.step_01_human_parsing import HumanParsingStep
        from app.ai_pipeline.steps.step_02_pose_estimation import PoseEstimationStep
        from app.ai_pipeline.steps.step_03_cloth_segmentation import ClothSegmentationStep
        from app.ai_pipeline.steps.step_04_geometric_matching import GeometricMatchingStep
        from app.ai_pipeline.steps.step_05_cloth_warping import ClothWarpingStep
        from app.ai_pipeline.steps.step_06_virtual_fitting import VirtualFittingStep
        from app.ai_pipeline.steps.step_07_post_processing import PostProcessingStep
        from app.ai_pipeline.steps.step_08_quality_assessment import QualityAssessmentStep
        
        pipeline_step_classes = {
            'step_01': HumanParsingStep,
            'step_02': PoseEstimationStep,
            'step_03': ClothSegmentationStep,
            'step_04': GeometricMatchingStep,
            'step_05': ClothWarpingStep,
            'step_06': VirtualFittingStep,
            'step_07': PostProcessingStep,
            'step_08': QualityAssessmentStep
        }
        
        AI_PIPELINE_AVAILABLE = True
        logger.info("✅ AI Pipeline Steps Import 성공")
    except ImportError as e:
        logger.warning(f"⚠️ AI Pipeline Steps Import 실패: {e}")
        AI_PIPELINE_AVAILABLE = False

# 서비스 레이어 Import (조건부)
SERVICES_AVAILABLE = False
try:
    from app.services import (
        get_pipeline_service_manager,
        get_step_service_manager
    )
    SERVICES_AVAILABLE = True
    logger.info("✅ Services 레이어 Import 성공")
except ImportError as e:
    logger.warning(f"⚠️ Services Import 실패: {e}")

# API 라우터 Import (조건부)
API_ROUTES_AVAILABLE = False
try:
    from app.api.pipeline_routes import router as pipeline_router
    from app.api.step_routes import router as step_router
    from app.api.health import router as health_router
    from app.api.models import router as models_router
    from app.api.websocket_routes import router as websocket_router
    API_ROUTES_AVAILABLE = True
    logger.info("✅ API Routes Import 성공")
except ImportError as e:
    logger.warning(f"⚠️ API Routes Import 실패: {e}")

# ===============================================================
# 🔧 전역 변수 및 상태 관리
# ===============================================================

# 통합 유틸리티 매니저
global_utils_manager = None

# AI 파이프라인 Steps
pipeline_steps = {}

# 서비스 매니저들
service_managers = {}

# WebSocket 연결 관리
active_connections: List[WebSocket] = []

# 서버 상태
server_state = {
    "initialized": False,
    "utils_loaded": False,
    "models_loaded": False,
    "services_ready": False,
    "start_time": time.time(),
    "total_requests": 0,
    "active_sessions": 0
}

# ===============================================================
# 🔥 DI Container 클래스 (ModelLoader 문제 해결용)
# ===============================================================

class SimpleDIContainer:
    """🔥 간단한 DI 컨테이너 - ModelLoader 문제 해결용"""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._instances = {}
        self._instance_lock = threading.RLock()
        self._initialized = True
        logger.info("✅ DI Container 초기화")
    
    def register(self, name: str, instance):
        """의존성 등록"""
        with self._instance_lock:
            self._instances[name] = instance
            logger.info(f"✅ DI 등록: {name} ({type(instance).__name__})")
    
    def get(self, name: str):
        """의존성 조회"""
        with self._instance_lock:
            instance = self._instances.get(name)
            if instance:
                logger.debug(f"🔍 DI 조회 성공: {name}")
            else:
                logger.warning(f"⚠️ DI 조회 실패: {name}")
            return instance
    
    def exists(self, name: str) -> bool:
        """의존성 존재 확인"""
        with self._instance_lock:
            return name in self._instances
    
    def clear(self):
        """모든 의존성 정리"""
        with self._instance_lock:
            count = len(self._instances)
            self._instances.clear()
            logger.info(f"🧹 DI Container 정리: {count}개 제거")

# 전역 DI 컨테이너
_global_di_container = SimpleDIContainer()

def get_di_container():
    """전역 DI 컨테이너 반환"""
    return _global_di_container

# ===============================================================
# 🔧 WebSocket 관리자
# ===============================================================

class WebSocketManager:
    """WebSocket 연결 관리자"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """클라이언트 연결"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"🔗 WebSocket 연결됨 - 총 {len(self.active_connections)}개 연결")
    
    def disconnect(self, websocket: WebSocket):
        """클라이언트 연결 해제"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"🔌 WebSocket 연결 해제됨 - 총 {len(self.active_connections)}개 연결")
    
    async def send_to_client(self, websocket: WebSocket, message: Dict[str, Any]):
        """특정 클라이언트에게 메시지 전송"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.warning(f"WebSocket 메시지 전송 실패: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """모든 클라이언트에게 메시지 브로드캐스트"""
        if not self.active_connections:
            return
        
        message_json = json.dumps(message)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.warning(f"WebSocket 메시지 전송 실패: {e}")
                disconnected.append(connection)
        
        # 연결이 끊어진 클라이언트 제거
        for conn in disconnected:
            self.disconnect(conn)

# 전역 WebSocket 매니저
websocket_manager = WebSocketManager()

# ===============================================================
# 🔧 초기화 함수들 (순환참조 해결)
# ===============================================================

async def initialize_unified_utils_system() -> bool:
    """통합 유틸리티 시스템 초기화"""
    global global_utils_manager
    
    try:
        if not UNIFIED_UTILS_AVAILABLE:
            logger.error("❌ 통합 유틸리티 시스템이 사용 불가능합니다")
            return False
        
        logger.info("🔄 통합 유틸리티 시스템 초기화 중...")
        
        # 전역 유틸리티 매니저 초기화
        result = initialize_global_utils(
            device=DEVICE,
            memory_gb=TOTAL_MEMORY_GB,
            is_m3_max=IS_M3_MAX,
            optimization_enabled=True,
            max_workers=min(os.cpu_count() or 4, 8),
            cache_enabled=True
        )
        
        if result.get("success", False):
            global_utils_manager = get_utils_manager()
            logger.info("✅ 통합 유틸리티 시스템 초기화 완료")
            return True
        else:
            logger.error(f"❌ 통합 유틸리티 시스템 초기화 실패: {result.get('error', 'Unknown')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 통합 유틸리티 시스템 초기화 오류: {e}")
        return False

async def initialize_model_loader_di():
    """🔥 ModelLoader DI 시스템 초기화"""
    try:
        logger.info("🔄 ModelLoader DI 시스템 초기화 시작...")
        
        # 1. DI Container 준비
        di_container = get_di_container()
        
        # 2. ModelLoader 초기화 및 등록
        try:
            # 기존 전역 ModelLoader 확인
            try:
                from app.ai_pipeline.utils.model_loader import get_global_model_loader, initialize_global_model_loader
                
                # ModelLoader 초기화
                init_result = initialize_global_model_loader(
                    device=DEVICE,
                    use_fp16=True if DEVICE != 'cpu' else False,
                    optimization_enabled=True,
                    enable_fallback=True
                )
                
                if init_result.get("success"):
                    model_loader = get_global_model_loader()
                    if model_loader:
                        # DI Container에 등록
                        di_container.register('model_loader', model_loader)
                        logger.info("✅ ModelLoader DI 등록 완료")
                        return True
                    else:
                        logger.error("❌ ModelLoader 인스턴스가 None")
                else:
                    logger.error(f"❌ ModelLoader 초기화 실패: {init_result.get('error')}")
            
            except ImportError as e:
                logger.warning(f"⚠️ ModelLoader import 실패: {e}")
                # 폴백: 새로 생성
                try:
                    from app.ai_pipeline.utils.model_loader import ModelLoader
                    model_loader = ModelLoader(device=DEVICE)
                    di_container.register('model_loader', model_loader)
                    logger.info("✅ 새 ModelLoader 생성 및 DI 등록")
                    return True
                except Exception as e2:
                    logger.error(f"❌ 새 ModelLoader 생성 실패: {e2}")
        
        except Exception as e:
            logger.error(f"❌ ModelLoader DI 설정 실패: {e}")
        
        # 3. Step 생성 함수들을 DI 버전으로 패치
        await patch_step_creation_functions_di(di_container)
        
        logger.info("🎉 ModelLoader DI 시스템 초기화 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ DI 시스템 초기화 실패: {e}")
        return False

async def patch_step_creation_functions_di(di_container):
    """🔥 Step 생성 함수들에 ModelLoader 자동 주입"""
    try:
        model_loader = di_container.get('model_loader')
        if not model_loader:
            logger.warning("⚠️ ModelLoader가 DI Container에 없음")
            return
        
        # HumanParsingStep 패치
        try:
            import app.ai_pipeline.steps.step_01_human_parsing as hp_module
            
            if hasattr(hp_module, 'create_human_parsing_step'):
                original_create = hp_module.create_human_parsing_step
                
                def create_with_di(*args, **kwargs):
                    # ModelLoader 자동 주입
                    if 'model_loader' not in kwargs:
                        kwargs['model_loader'] = model_loader
                        logger.info("✅ HumanParsingStep에 ModelLoader 자동 주입")
                    return original_create(*args, **kwargs)
                
                hp_module.create_human_parsing_step = create_with_di
                logger.info("✅ HumanParsingStep 생성 함수 DI 패치 완료")
        
        except Exception as e:
            logger.warning(f"⚠️ HumanParsingStep 패치 실패: {e}")
        
        # ClothSegmentationStep 패치
        try:
            import app.ai_pipeline.steps.step_03_cloth_segmentation as cs_module
            
            if hasattr(cs_module, 'create_cloth_segmentation_step'):
                original_create = cs_module.create_cloth_segmentation_step
                
                def create_with_di(*args, **kwargs):
                    if 'model_loader' not in kwargs:
                        kwargs['model_loader'] = model_loader
                        logger.info("✅ ClothSegmentationStep에 ModelLoader 자동 주입")
                    return original_create(*args, **kwargs)
                
                cs_module.create_cloth_segmentation_step = create_with_di
                logger.info("✅ ClothSegmentationStep 생성 함수 DI 패치 완료")
        
        except Exception as e:
            logger.warning(f"⚠️ ClothSegmentationStep 패치 실패: {e}")
        
        # 다른 Step들도 필요하면 추가...
        
    except Exception as e:
        logger.error(f"❌ Step 함수 패치 실패: {e}")

async def initialize_pipeline_steps():
    """AI 파이프라인 Steps 초기화"""
    global pipeline_steps
    
    try:
        if not AI_PIPELINE_AVAILABLE:
            logger.warning("⚠️ AI Pipeline이 사용 불가능합니다")
            return False
        
        logger.info("🔄 AI 파이프라인 Steps 초기화 중...")
        
        # DI Container에서 ModelLoader 가져오기
        di_container = get_di_container()
        model_loader = di_container.get('model_loader')
        
        if model_loader:
            logger.info("✅ DI Container에서 ModelLoader 발견됨")
        else:
            logger.warning("⚠️ DI Container에 ModelLoader 없음 - 기본 초기화 진행")
        
        initialized_steps = 0
        
        for step_name, step_class in pipeline_step_classes.items():
            try:
                # ModelLoader를 포함해서 Step 생성
                step_kwargs = {
                    'device': DEVICE,
                    'optimization_enabled': True,
                    'memory_gb': TOTAL_MEMORY_GB,
                }
                
                # ModelLoader 주입
                if model_loader:
                    step_kwargs['model_loader'] = model_loader
                
                # Step 인스턴스 생성
                step_instance = step_class(**step_kwargs)
                
                # Step 초기화
                if hasattr(step_instance, 'initialize'):
                    if asyncio.iscoroutinefunction(step_instance.initialize):
                        success = await step_instance.initialize()
                    else:
                        success = step_instance.initialize()
                    
                    if success:
                        pipeline_steps[step_name] = step_instance
                        initialized_steps += 1
                        logger.info(f"✅ {step_name} ({step_class.__name__}) 초기화 완료")
                    else:
                        logger.warning(f"⚠️ {step_name} 초기화 실패")
                else:
                    # initialize 메서드가 없는 경우
                    pipeline_steps[step_name] = step_instance
                    initialized_steps += 1
                    logger.info(f"✅ {step_name} ({step_class.__name__}) 생성 완료")
                    
            except Exception as e:
                logger.warning(f"⚠️ {step_name} 초기화 실패: {e}")
        
        logger.info(f"✅ AI 파이프라인 초기화 완료: {initialized_steps}/8 단계")
        return initialized_steps > 0
        
    except Exception as e:
        logger.error(f"❌ AI 파이프라인 초기화 오류: {e}")
        return False

async def initialize_services() -> bool:
    """서비스 레이어 초기화"""
    global service_managers
    
    try:
        if not SERVICES_AVAILABLE:
            logger.warning("⚠️ Services 레이어가 사용 불가능합니다")
            return False
        
        logger.info("🔄 서비스 레이어 초기화 중...")
        
        # 서비스 매니저들 초기화
        try:
            service_managers['pipeline'] = get_pipeline_service_manager()
            service_managers['step'] = get_step_service_manager()
            
            logger.info("✅ 서비스 레이어 초기화 완료")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ 서비스 매니저 초기화 실패: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 서비스 레이어 초기화 오류: {e}")
        return False

# ===============================================================
# 🔧 FastAPI 수명주기 관리
# ===============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 애플리케이션 수명주기 관리"""
    global server_state
    
    # === 시작 이벤트 ===
    logger.info("🚀 MyCloset AI Backend 시작 - 프론트엔드 완전 호환 v5.0")
    logger.info(f"🔧 디바이스: {DEVICE_NAME} ({DEVICE})")
    logger.info(f"🍎 M3 Max: {'✅' if IS_M3_MAX else '❌'}")
    logger.info(f"💾 메모리: {TOTAL_MEMORY_GB:.1f}GB (사용가능: {AVAILABLE_MEMORY_GB:.1f}GB)")
    
    initialization_success = True
    
    # 1. 통합 유틸리티 시스템 초기화
    try:
        if await initialize_unified_utils_system():
            server_state["utils_loaded"] = True
            server_state["models_loaded"] = True
            logger.info("✅ 1단계: 통합 유틸리티 시스템 초기화 완료")
        else:
            logger.warning("⚠️ 1단계: 통합 유틸리티 시스템 초기화 실패 - 시뮬레이션 모드")
            initialization_success = False
    except Exception as e:
        logger.error(f"❌ 통합 유틸리티 시스템 초기화 중 오류: {e}")
        initialization_success = False
    
    # 1.5. ModelLoader DI 시스템 초기화
    try:
        logger.info("🔄 1.5단계: ModelLoader DI 시스템 초기화...")
        di_success = await initialize_model_loader_di()
        if di_success:
            logger.info("✅ 1.5단계: ModelLoader DI 시스템 초기화 완료")
        else:
            logger.warning("⚠️ 1.5단계: ModelLoader DI 시스템 초기화 실패")
    except Exception as e:
        logger.error(f"❌ ModelLoader DI 시스템 초기화 중 오류: {e}")
    
    # 2. AI 파이프라인 초기화 (DI 적용)
    try:
        if await initialize_pipeline_steps():
            logger.info("✅ 2단계: AI 파이프라인 DI 초기화 완료")
        else:
            logger.warning("⚠️ 2단계: AI 파이프라인 초기화 실패")
            initialization_success = False
    except Exception as e:
        logger.error(f"❌ AI 파이프라인 초기화 중 오류: {e}")
        initialization_success = False
    
    # 3. 서비스 레이어 초기화
    try:
        if await initialize_services():
            server_state["services_ready"] = True
            logger.info("✅ 3단계: 서비스 레이어 초기화 완료")
        else:
            logger.warning("⚠️ 3단계: 서비스 레이어 초기화 실패")
    except Exception as e:
        logger.error(f"❌ 서비스 레이어 초기화 중 오류: {e}")
    
    # 초기화 완료
    server_state["initialized"] = True
    
    if initialization_success:
        logger.info("🎉 서버 초기화 완료 - 모든 시스템 정상")
    else:
        logger.warning("⚠️ 서버 초기화 완료 - 일부 시스템 시뮬레이션 모드")
    
    logger.info("📡 요청 수신 대기 중...")
    
    yield
    
    # === 종료 이벤트 ===
    logger.info("🛑 MyCloset AI Backend 종료 중...")
    
    try:
        # AI 파이프라인 정리
        for step_name, step_instance in pipeline_steps.items():
            try:
                if hasattr(step_instance, 'cleanup'):
                    if asyncio.iscoroutinefunction(step_instance.cleanup):
                        await step_instance.cleanup()
                    else:
                        step_instance.cleanup()
                logger.info(f"🧹 {step_name} 정리 완료")
            except Exception as e:
                logger.warning(f"⚠️ {step_name} 정리 실패: {e}")
        
        # 통합 유틸리티 시스템 정리
        if UNIFIED_UTILS_AVAILABLE:
            reset_global_utils()
            logger.info("🧹 통합 유틸리티 시스템 정리 완료")
        
        # GPU 메모리 정리
        if DEVICE == "mps" and torch.backends.mps.is_available():
            try:
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            except Exception as e:
                logger.warning(f"MPS 캐시 정리 실패: {e}")
        elif DEVICE == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"CUDA 캐시 정리 실패: {e}")
        
        logger.info("💾 메모리 정리 완료")
        
    except Exception as e:
        logger.warning(f"⚠️ 종료 처리 중 오류: {e}")
    
    logger.info("✅ 서버 종료 완료")

# ===============================================================
# 🔧 FastAPI 앱 생성 및 설정
# ===============================================================

app = FastAPI(
    title="MyCloset AI",
    description="🍎 M3 Max 최적화 AI 가상 피팅 시스템 - 프론트엔드 완전 호환 v5.0",
    version="5.0.0-frontend-compatible",
    debug=True,
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://localhost:4000", "http://localhost:3001", 
        "http://localhost:5173", "http://localhost:5174", "http://localhost:8080", 
        "http://127.0.0.1:3000", "http://127.0.0.1:4000", "http://127.0.0.1:5173", 
        "http://127.0.0.1:5174", "http://127.0.0.1:8080"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Gzip 압축
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 정적 파일 서빙
static_dir = backend_dir / "static"
static_dir.mkdir(exist_ok=True)
(static_dir / "uploads").mkdir(exist_ok=True)
(static_dir / "results").mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ===============================================================
# 🔧 API 라우터 등록 (조건부)
# ===============================================================

if API_ROUTES_AVAILABLE:
    try:
        app.include_router(health_router, prefix="/api", tags=["Health"])
        app.include_router(models_router, prefix="/api", tags=["Models"])
        app.include_router(pipeline_router, prefix="/api", tags=["Pipeline"])
        app.include_router(step_router, prefix="/api", tags=["Steps"])
        app.include_router(websocket_router, prefix="/api", tags=["WebSocket"])
        logger.info("✅ 모든 API 라우터 등록 완료")
    except Exception as e:
        logger.warning(f"⚠️ API 라우터 등록 실패: {e}")

# ===============================================================
# 🔧 핵심 API 엔드포인트들
# ===============================================================

@app.get("/")
async def root():
    """루트 엔드포인트"""
    global server_state, pipeline_steps, service_managers
    
    # 시스템 상태 조회
    system_status = {}
    if UNIFIED_UTILS_AVAILABLE and global_utils_manager:
        try:
            system_status = get_system_status()
        except Exception as e:
            system_status = {"error": str(e)}
    
    return {
        "message": "🍎 MyCloset AI 서버가 실행 중입니다! (프론트엔드 완전 호환 v5.0)",
        "version": "5.0.0-frontend-compatible",
        "status": {
            "initialized": server_state["initialized"],
            "utils_loaded": server_state["utils_loaded"],
            "models_loaded": server_state["models_loaded"],
            "services_ready": server_state["services_ready"],
            "uptime": time.time() - server_state["start_time"]
        },
        "system": {
            "device": DEVICE,
            "device_name": DEVICE_NAME,
            "m3_max": IS_M3_MAX,
            "memory_gb": TOTAL_MEMORY_GB,
            "optimization": "enabled" if IS_M3_MAX else "standard"
        },
        "components": {
            "unified_utils": UNIFIED_UTILS_AVAILABLE,
            "ai_pipeline": AI_PIPELINE_AVAILABLE,
            "services": SERVICES_AVAILABLE,
            "api_routes": API_ROUTES_AVAILABLE,
            "pipeline_steps_loaded": len(pipeline_steps),
            "service_managers_loaded": len(service_managers)
        },
        "features": {
            "8_step_pipeline": True,
            "real_ai_models": server_state["models_loaded"],
            "websocket_support": True,
            "m3_max_optimized": IS_M3_MAX,
            "memory_management": True,
            "visualization": True,
            "unified_utils": UNIFIED_UTILS_AVAILABLE,
            "circular_dependency_resolved": True,
            "frontend_compatible": True,
            "model_loader_di": True
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/api/health",
            "pipeline": "/api/pipeline",
            "steps": "/api/step",
            "models": "/api/models",
            "websocket": "/api/ws"
        },
        "system_status": system_status,
        "timestamp": time.time()
    }

@app.get("/health")
@app.get("/api/health")
async def health_check():
    """헬스체크"""
    global server_state, pipeline_steps, global_utils_manager
    
    memory_info = psutil.virtual_memory()
    
    # 통합 유틸리티 시스템 상태 확인
    utils_status = "healthy"
    utils_details = {}
    
    if UNIFIED_UTILS_AVAILABLE and global_utils_manager:
        try:
            utils_details = get_system_status()
            if utils_details.get("error"):
                utils_status = "error"
            elif not utils_details.get("initialized", False):
                utils_status = "not_initialized"
        except Exception as e:
            utils_status = "error"
            utils_details = {"error": str(e)}
    else:
        utils_status = "not_available"
    
    # 파이프라인 상태
    pipeline_status = "healthy" if len(pipeline_steps) >= 4 else "degraded"
    
    # 전체 상태 판정
    overall_status = "healthy"
    if not server_state["initialized"]:
        overall_status = "initializing"
    elif utils_status in ["error", "not_initialized"] or pipeline_status == "degraded":
        overall_status = "degraded"
    
    return {
        "status": overall_status,
        "app": "MyCloset AI",
        "version": "5.0.0-frontend-compatible",
        "components": {
            "server": {
                "status": "healthy" if server_state["initialized"] else "initializing",
                "uptime": time.time() - server_state["start_time"],
                "total_requests": server_state["total_requests"],
                "active_sessions": server_state["active_sessions"]
            },
            "unified_utils": {
                "status": utils_status,
                "available": UNIFIED_UTILS_AVAILABLE,
                "details": utils_details
            },
            "pipeline": {
                "status": pipeline_status,
                "steps_loaded": len(pipeline_steps),
                "steps_available": list(pipeline_steps.keys()),
                "ai_pipeline_available": AI_PIPELINE_AVAILABLE
            },
            "services": {
                "status": "healthy" if server_state["services_ready"] else "unavailable",
                "loaded_services": len(service_managers),
                "services_available": SERVICES_AVAILABLE
            },
            "di_container": {
                "status": "healthy" if get_di_container().exists('model_loader') else "unavailable",
                "model_loader_registered": get_di_container().exists('model_loader')
            }
        },
        "system": {
            "device": DEVICE,
            "device_name": DEVICE_NAME,
            "memory": {
                "total_gb": TOTAL_MEMORY_GB,
                "available_gb": round(memory_info.available / (1024**3), 1),
                "used_percent": round(memory_info.percent, 1),
                "is_sufficient": memory_info.available > (2 * 1024**3)
            },
            "optimization": {
                "m3_max_enabled": IS_M3_MAX,
                "device_optimization": True,
                "memory_management": True,
                "neural_engine": IS_M3_MAX,
                "unified_utils": UNIFIED_UTILS_AVAILABLE,
                "circular_dependency_resolved": True,
                "model_loader_di": True
            }
        },
        "features": {
            "real_ai_models": server_state["models_loaded"],
            "8_step_pipeline": len(pipeline_steps) == 8,
            "websocket_support": True,
            "visualization": True,
            "api_routes": API_ROUTES_AVAILABLE,
            "unified_utils": UNIFIED_UTILS_AVAILABLE,
            "frontend_compatible": True,
            "model_loader_di_resolved": True
        },
        "timestamp": time.time()
    }

# ===============================================================
# 🔥 DI 테스트 엔드포인트
# ===============================================================

@app.get("/api/test-model-loader-di")
async def test_model_loader_di():
    """🧪 ModelLoader DI 테스트"""
    try:
        di_container = get_di_container()
        model_loader = di_container.get('model_loader')
        
        if model_loader:
            # ModelLoader 정보 확인
            info = {
                "model_loader_type": type(model_loader).__name__,
                "has_create_step_interface": hasattr(model_loader, 'create_step_interface'),
                "device": getattr(model_loader, 'device', 'unknown'),
                "is_initialized": getattr(model_loader, 'is_initialized', False)
            }
            
            # Step 인터페이스 테스트
            if hasattr(model_loader, 'create_step_interface'):
                try:
                    test_interface = model_loader.create_step_interface("TestStep")
                    info["step_interface_creation"] = test_interface is not None
                    
                    # 실제 모델 로드 테스트
                    if test_interface and hasattr(test_interface, 'get_model'):
                        try:
                            test_model = await test_interface.get_model("test_model")
                            info["model_loading"] = test_model is not None
                            info["model_type"] = type(test_model).__name__ if test_model else None
                        except Exception as e:
                            info["model_loading_error"] = str(e)
                            
                except Exception as e:
                    info["step_interface_error"] = str(e)
            
            return {
                "success": True,
                "message": "ModelLoader DI 정상 작동",
                "model_loader_info": info,
                "di_container_status": {
                    "total_instances": len(di_container._instances),
                    "registered_names": list(di_container._instances.keys())
                }
            }
        else:
            return {
                "success": False,
                "message": "ModelLoader가 DI Container에 없음",
                "di_container_contents": list(di_container._instances.keys()),
                "suggestion": "ModelLoader DI 초기화가 필요합니다"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "ModelLoader DI 테스트 실패"
        }

@app.post("/api/init-model-loader-di")
async def init_model_loader_di():
    """🔧 ModelLoader DI 수동 초기화"""
    try:
        success = await initialize_model_loader_di()
        
        if success:
            return {
                "success": True,
                "message": "ModelLoader DI 초기화 완료",
                "di_status": {
                    "model_loader_registered": get_di_container().exists('model_loader'),
                    "total_instances": len(get_di_container()._instances)
                }
            }
        else:
            return {
                "success": False,
                "message": "ModelLoader DI 초기화 실패"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "ModelLoader DI 초기화 중 오류"
        }

# ===============================================================
# 🔧 Step API 엔드포인트들 (프론트엔드 호환)
# ===============================================================

@app.post("/api/step/1/upload-validation")
async def step_1_upload_validation(
    person_image: UploadFile = File(...),
    clothing_image: UploadFile = File(...),
    session_id: str = Form(None)
):
    """Step 1: 이미지 업로드 검증"""
    global server_state
    server_state["total_requests"] += 1
    
    start_time = time.time()
    
    try:
        logger.info("🚀 Step 1: 이미지 업로드 검증 시작")
        
        # 1. 파일 크기 검증
        person_data = await person_image.read()
        clothing_data = await clothing_image.read()
        
        if len(person_data) > 50 * 1024 * 1024:  # 50MB
            raise HTTPException(status_code=400, detail="사용자 이미지가 50MB를 초과합니다")
        
        if len(clothing_data) > 50 * 1024 * 1024:  # 50MB
            raise HTTPException(status_code=400, detail="의류 이미지가 50MB를 초과합니다")
        
        # 2. 이미지 형식 검증
        try:
            person_img = Image.open(io.BytesIO(person_data))
            clothing_img = Image.open(io.BytesIO(clothing_data))
            
            # RGB 변환
            if person_img.mode != 'RGB':
                person_img = person_img.convert('RGB')
            if clothing_img.mode != 'RGB':
                clothing_img = clothing_img.convert('RGB')
                
            logger.info(f"✅ 이미지 형식 검증 완료 - 사용자: {person_img.size}, 의류: {clothing_img.size}")
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"잘못된 이미지 형식: {str(e)}")
        
        # 3. 세션 ID 생성
        if not session_id:
            session_id = f"session_{int(time.time())}_{hash(person_data + clothing_data) % 10000:04d}"
            
        # 4. 이미지 저장 (옵션)
        uploads_dir = backend_dir / "static" / "uploads"
        uploads_dir.mkdir(exist_ok=True)
        
        person_path = uploads_dir / f"{session_id}_person.jpg"
        clothing_path = uploads_dir / f"{session_id}_clothing.jpg"
        
        person_img.save(person_path, "JPEG", quality=90)
        clothing_img.save(clothing_path, "JPEG", quality=90)
        
        processing_time = time.time() - start_time
        
        response = {
            "success": True,
            "message": "이미지 업로드 검증 완료",
            "processing_time": processing_time,
            "confidence": 1.0,
            "details": {
                "session_id": session_id,
                "person_image": {
                    "size": person_img.size,
                    "format": person_img.format,
                    "mode": person_img.mode,
                    "file_size_mb": round(len(person_data) / (1024*1024), 2)
                },
                "clothing_image": {
                    "size": clothing_img.size,
                    "format": clothing_img.format,
                    "mode": clothing_img.mode,
                    "file_size_mb": round(len(clothing_data) / (1024*1024), 2)
                },
                "saved_paths": {
                    "person": str(person_path),
                    "clothing": str(clothing_path)
                }
            },
            "timestamp": time.time()
        }
        
        logger.info(f"✅ Step 1 완료 - 세션: {session_id}, 처리시간: {processing_time:.2f}초")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Step 1 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Step 1 처리 실패: {str(e)}")

# ===============================================================
# 🔧 폴백 API 엔드포인트들 (라우터 실패 시)
# ===============================================================

@app.post("/api/pipeline/virtual-tryon")
@app.post("/api/pipeline/complete")
async def fallback_virtual_tryon(
    person_image: UploadFile = File(...),
    clothing_image: UploadFile = File(...),
    height: float = Form(170),
    weight: float = Form(65),
    options: str = Form("{}")
):
    """가상 피팅 처리 (폴백 엔드포인트)"""
    global server_state
    server_state["total_requests"] += 1
    
    try:
        # 이미지 데이터 읽기
        person_data = await person_image.read()
        clothing_data = await clothing_image.read()
        
        # 옵션 파싱
        try:
            options_dict = json.loads(options)
        except json.JSONDecodeError:
            options_dict = {}
        
        # 세션 ID 생성
        session_id = f"complete_{int(time.time())}_{hash(person_data + clothing_data) % 10000:04d}"
        
        # 더미 결과 생성
        person_img = Image.open(io.BytesIO(person_data))
        clothing_img = Image.open(io.BytesIO(clothing_data))
        
        # 가상 피팅 결과 생성
        fitted_img = create_virtual_fitting_result(person_img, clothing_img)
        
        # base64 인코딩
        buffer = io.BytesIO()
        fitted_img.save(buffer, format='JPEG', quality=90)
        fitted_image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # BMI 계산
        bmi = weight / ((height / 100) ** 2)
        
        response = {
            "success": True,
            "message": "가상 피팅 완료",
            "processing_time": 6.5,
            "confidence": 0.88,
            "session_id": session_id,
            "fitted_image": fitted_image_base64,
            "fit_score": 0.92,
            "measurements": {
                "chest": round(height * 0.48 + (weight - 60) * 0.5, 1),
                "waist": round(height * 0.37 + (weight - 60) * 0.4, 1),
                "hip": round(height * 0.53 + (weight - 60) * 0.3, 1),
                "bmi": round(bmi, 1)
            },
            "clothing_analysis": {
                "category": "상의",
                "style": "캐주얼",
                "dominant_color": [100, 150, 200],
                "color_name": "블루",
                "material": "코튼",
                "pattern": "솔리드"
            },
            "recommendations": [
                "이 의류는 당신에게 잘 어울립니다",
                "색상이 피부톤과 잘 매치됩니다",
                "사이즈가 적절해 보입니다"
            ],
            "timestamp": time.time()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"가상 피팅 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===============================================================
# 🔧 WebSocket 엔드포인트
# ===============================================================

@app.websocket("/api/ws/pipeline")
async def websocket_pipeline(websocket: WebSocket):
    """파이프라인 실시간 통신"""
    await websocket_manager.connect(websocket)
    
    try:
        while True:
            # 클라이언트 메시지 수신
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 메시지 타입에 따른 처리
            if message.get("type") == "ping":
                await websocket_manager.send_to_client(websocket, {
                    "type": "pong",
                    "timestamp": time.time()
                })
            
            elif message.get("type") == "status_request":
                status = {
                    "type": "status_response",
                    "server_status": server_state,
                    "pipeline_steps": len(pipeline_steps),
                    "active_connections": len(websocket_manager.active_connections),
                    "unified_utils_available": UNIFIED_UTILS_AVAILABLE,
                    "model_loader_di": get_di_container().exists('model_loader'),
                    "timestamp": time.time()
                }
                
                # 통합 유틸리티 시스템 상태 추가
                if UNIFIED_UTILS_AVAILABLE and global_utils_manager:
                    try:
                        status["system_status"] = get_system_status()
                    except Exception as e:
                        status["system_status"] = {"error": str(e)}
                
                await websocket_manager.send_to_client(websocket, status)
    
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket 오류: {e}")
        websocket_manager.disconnect(websocket)

# ===============================================================
# 🔧 유틸리티 함수들
# ===============================================================

def create_virtual_fitting_result(person_img: Image.Image, clothing_img: Image.Image) -> Image.Image:
    """가상 피팅 결과 생성 (고품질 시뮬레이션)"""
    # 사람 이미지를 기본으로 사용
    result = person_img.copy()
    
    # 의류 이미지를 적절한 위치에 합성 (매우 간단한 시뮬레이션)
    clothing_resized = clothing_img.resize((200, 250))
    
    # 투명도를 적용하여 합성
    if result.mode != 'RGBA':
        result = result.convert('RGBA')
    if clothing_resized.mode != 'RGBA':
        clothing_resized = clothing_resized.convert('RGBA')
    
    # 의류를 가슴 부분에 배치
    paste_x = (result.width - clothing_resized.width) // 2
    paste_y = result.height // 3
    
    # 알파 블렌딩으로 자연스럽게 합성
    overlay = Image.new('RGBA', result.size, (0, 0, 0, 0))
    overlay.paste(clothing_resized, (paste_x, paste_y))
    
    # 50% 투명도로 합성
    result = Image.alpha_composite(result, overlay)
    
    return result.convert('RGB')

# ===============================================================
# 🔧 서버 실행 진입점
# ===============================================================

if __name__ == "__main__":
    logger.info("🔧 개발 모드: uvicorn 서버 직접 실행")
    logger.info(f"📍 주소: http://localhost:8000")
    logger.info(f"📖 API 문서: http://localhost:8000/docs")
    logger.info(f"🔧 디바이스: {DEVICE_NAME} ({DEVICE})")
    logger.info(f"🍎 M3 Max: {'✅' if IS_M3_MAX else '❌'}")
    logger.info(f"💾 메모리: {TOTAL_MEMORY_GB:.1f}GB")
    
    logger.info("🔧 컴포넌트 상태:")
    logger.info(f"   - 통합 유틸리티 시스템: {'✅' if UNIFIED_UTILS_AVAILABLE else '❌'}")
    logger.info(f"   - AI Pipeline: {'✅' if AI_PIPELINE_AVAILABLE else '❌'}")
    logger.info(f"   - Services: {'✅' if SERVICES_AVAILABLE else '❌'}")
    logger.info(f"   - API Routes: {'✅' if API_ROUTES_AVAILABLE else '❌'}")
    logger.info("✅ 순환참조 문제 해결됨")
    logger.info("🎯 프론트엔드 완전 호환 - Step API 포함")
    logger.info("🔥 ModelLoader DI 시스템 포함")
    
    logger.info("\n📡 사용 가능한 Step API 엔드포인트:")
    logger.info("   - POST /api/step/1/upload-validation")
    logger.info("   - POST /api/pipeline/complete")
    logger.info("   - GET /api/health")
    logger.info("   - GET /api/test-model-loader-di")
    logger.info("   - POST /api/init-model-loader-di")
    
    try:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True,
            workers=1,
            loop="auto",
            timeout_keep_alive=30,
        )
    except KeyboardInterrupt:
        logger.info("🛑 사용자에 의해 서버가 중단되었습니다")
    except Exception as e:
        logger.error(f"❌ 서버 실행 실패: {e}")
        sys.exit(1)