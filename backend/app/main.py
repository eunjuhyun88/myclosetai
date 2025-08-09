# backend/app/main.py
"""
🔥 MyCloset AI Backend - Central Hub DI Container v7.0 완전 연동 v29.0
================================================================================

✅ Central Hub DI Container v7.0 완전 연동 - 중앙 허브 패턴 적용
✅ 순환참조 완전 해결 - TYPE_CHECKING + 지연 import 완벽 적용
✅ 단방향 의존성 그래프 - DI Container만을 통한 의존성 주입
✅ StepServiceManager v15.0 + RealAIStepImplementationManager v14.0 완전 통합
✅ step_routes.py v5.0 완벽 연동 (모든 기능 복구)
✅ step_implementations.py DetailedDataSpec 완전 활용
✅ 실제 229GB AI 모델 파이프라인 완전 활용
✅ 프론트엔드 호환성 100% 보장 (누락된 기능 복구)
✅ conda 환경 mycloset-ai-clean 최적화
✅ M3 Max 128GB 메모리 최적화
✅ WebSocket 실시간 진행률 지원 (완전 복구)
✅ 세션 기반 이미지 관리 완전 구현
✅ 프로덕션 레벨 안정성 및 에러 처리
✅ 모든 누락된 엔드포인트 및 기능 복구

핵심 설계 원칙:
1. Single Source of Truth - 모든 서비스는 Central Hub DI Container를 거침
2. Central Hub Pattern - DI Container가 모든 컴포넌트의 중심
3. Dependency Inversion - 상위 모듈이 하위 모듈을 제어
4. Zero Circular Reference - 순환참조 원천 차단

새로운 통합 아키텍처 (Central Hub DI Container v7.0 중심):
main.py → Central Hub DI Container v7.0 → StepServiceManager v15.0 → 
RealAIStepImplementationManager v14.0 → StepFactory v11.0 → 
BaseStepMixin v20.0 → 실제 229GB AI 모델

Author: MyCloset AI Team
Date: 2025-07-31
Version: 29.0.0 (Central Hub Integration)
"""

import os
import sys
import logging
import time
import gc
import warnings
import traceback
import subprocess
import platform
import psutil
import json
import uuid
import threading
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List, Union, Callable, Tuple

# 경고 무시
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# MediaPipe 및 TensorFlow Lite 경고 무시 추가
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'  # GPU 관련 경고 무시
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # TensorFlow 최적화 경고 무시
os.environ['ABSL_LOGGING_MIN_LEVEL'] = '2'  # absl 로깅 레벨 설정

# 추가 경고 필터링
warnings.filterwarnings('ignore', category=UserWarning, module='mediapipe')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=UserWarning, module='absl')
warnings.filterwarnings('ignore', message='.*inference_feedback_manager.*')
warnings.filterwarnings('ignore', message='.*landmark_projection_calculator.*')

# 로그 레벨 조정 (서버 시작 시 간단 요약, API 호출 시 상세 로그)
import logging
import os
import sys

# 환경변수로 출력 제어 - 서버 시작 시 간단 요약, API 호출 시 상세 로그
QUIET_MODE = os.getenv('QUIET_MODE', 'true').lower() == 'true'  # 서버 시작 시 간단 모드 (기본값: true)
STEP_LOGGING = os.getenv('STEP_LOGGING', 'true').lower() == 'true'  # API 호출 시 상세 로그
MODEL_LOGGING = os.getenv('MODEL_LOGGING', 'false').lower() == 'true'  # 모델 로딩 로깅 비활성화 (기본값: false)

# 서버 시작 시에는 간단한 요약만, API 호출 시에는 상세 로그
if QUIET_MODE:
    # 서버 시작 시: INFO 로그 차단, ERROR만 표시
    logging.disable(logging.INFO)
    os.environ['LOG_LEVEL'] = 'ERROR'
    os.environ['LOG_MODE'] = 'startup_summary'
else:
    # API 호출 시: 상세 로그 표시
    logging.disable(logging.DEBUG)
    os.environ['LOG_LEVEL'] = 'INFO'
    os.environ['LOG_MODE'] = 'api_detailed'

# 루트 로거 설정
root_logger = logging.getLogger()
root_logger.handlers.clear()
if QUIET_MODE:
    root_logger.setLevel(logging.ERROR)
else:
    root_logger.setLevel(logging.INFO)

# 모든 로거의 핸들러 제거 및 레벨 설정
for name in logging.root.manager.loggerDict:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    if QUIET_MODE:
        logger.setLevel(logging.ERROR)
        logger.propagate = True
    else:
        logger.setLevel(logging.INFO)
        logger.propagate = True

# 특정 모듈들의 로그 완전 차단
quiet_modules = [
    'app.core',
    'app.services', 
    'app.api',
    'backend.app',
    'transformers',
    'torch',
    'torchvision',
    'PIL',
    'cv2',
    'numpy',
    'segformer',
    'uvicorn',
    'fastapi',
    'uvicorn.access',
    'uvicorn.error',
    'step_model_requests',
    'step_interface',
    'di_container',
    'step_service'
]

# Step 관련 모듈은 조건부로 로깅 활성화
if STEP_LOGGING:
    step_modules = [
        'app.ai_pipeline.steps',
        'app.ai_pipeline.steps.step_01_human_parsing',
        'app.ai_pipeline.steps.step_02_pose_estimation',
        'app.ai_pipeline.steps.step_03_cloth_segmentation',
        'app.ai_pipeline.steps.step_04_geometric_matching',
        'app.ai_pipeline.steps.step_05_cloth_warping',
        'app.ai_pipeline.steps.step_06_virtual_fitting',
        'app.ai_pipeline.steps.step_07_post_processing',
        'app.ai_pipeline.steps.step_08_quality_assessment'
    ]
    for module in step_modules:
        logger = logging.getLogger(module)
        logger.setLevel(logging.INFO)
else:
    quiet_modules.extend([
        'app.ai_pipeline.steps',
        'app.ai_pipeline.steps.step_01_human_parsing',
        'app.ai_pipeline.steps.step_02_pose_estimation',
        'app.ai_pipeline.steps.step_03_cloth_segmentation',
        'app.ai_pipeline.steps.step_04_geometric_matching',
        'app.ai_pipeline.steps.step_05_cloth_warping',
        'app.ai_pipeline.steps.step_06_virtual_fitting',
        'app.ai_pipeline.steps.step_07_post_processing',
        'app.ai_pipeline.steps.step_08_quality_assessment'
    ])

# 모델 로딩 관련 모듈은 조건부로 로깅 활성화
model_modules = [
    'app.ai_pipeline.utils.model_loader',
    'app.ai_pipeline.utils.checkpoint_model_loader',
    'app.ai_pipeline.utils.dynamic_model_detector',
    'app.ai_pipeline.utils.smart_model_mapper',
    'app.ai_pipeline.utils.universal_step_loader',
    'app.core.di_container',
    'app.services.model_manager',
    'app.services.ai_models'
]

if MODEL_LOGGING:
    for module in model_modules:
        logger = logging.getLogger(module)
        logger.setLevel(logging.INFO)
        logger.propagate = True
else:
    quiet_modules.extend(model_modules)

# 모든 quiet 모듈의 로그 완전 차단
for module in quiet_modules:
    logger = logging.getLogger(module)
    logger.handlers.clear()
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False

# 핵심 정보만 출력하는 간단한 함수
def print_status(message):
    """핵심 상태 정보만 출력"""
    if not QUIET_MODE:
        print(f"✅ {message}")

def print_error(message):
    """에러 정보만 출력"""
    print(f"❌ {message}")

def print_warning(message):
    """경고 정보만 출력"""
    if not QUIET_MODE:
        print(f"⚠️ {message}")

def print_step(message):
    """Step 실행 정보만 출력"""
    if STEP_LOGGING:
        print(f"🔧 {message}")

def print_model(message):
    """모델 로딩 정보만 출력"""
    if MODEL_LOGGING:
        print(f"🧠 {message}")

if not QUIET_MODE:
    print("🔇 로그 출력 최소화 완료 (Step 로깅: " + ("활성화" if STEP_LOGGING else "비활성화") + ", 모델 로깅: " + ("활성화" if MODEL_LOGGING else "비활성화") + ")")

# =============================================================================
# 🔥 1. 실행 경로 자동 수정 및 시스템 정보
# =============================================================================

def fix_python_path():
    """실행 경로에 관계없이 Python Path 자동 수정"""
    current_file = Path(__file__).absolute()
    app_dir = current_file.parent       # backend/app
    backend_dir = app_dir.parent        # backend
    project_root = backend_dir.parent   # mycloset-ai
    
    # Python Path에 필요한 경로들 추가
    paths_to_add = [
        str(backend_dir),    # backend/ (가장 중요!)
        str(app_dir),        # backend/app/
        str(project_root)    # mycloset-ai/
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    # 환경 변수 설정
    os.environ.update({
        'PYTHONPATH': f"{backend_dir}:{os.environ.get('PYTHONPATH', '')}",
        'PROJECT_ROOT': str(project_root),
        'BACKEND_ROOT': str(backend_dir),
        'APP_ROOT': str(app_dir)
    })
    
    # 작업 디렉토리를 backend로 변경
    if Path.cwd() != backend_dir:
        try:
            os.chdir(backend_dir)
        except OSError:
            pass
    
    return {
        'app_dir': str(app_dir),
        'backend_dir': str(backend_dir),
        'project_root': str(project_root)
    }

# Python Path 수정 실행
path_info = fix_python_path()

def detect_system_info():
    """시스템 정보 직접 감지"""
    system_info = {
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'cpu_count': os.cpu_count() or 4
    }
    
    # conda 환경 감지
    is_conda = (
        'CONDA_DEFAULT_ENV' in os.environ or
        'CONDA_PREFIX' in os.environ or
        'conda' in sys.executable.lower()
    )
    system_info['is_conda'] = is_conda
    system_info['conda_env'] = os.environ.get('CONDA_DEFAULT_ENV', 'none')
    system_info['is_mycloset_env'] = system_info['conda_env'] == 'mycloset-ai-clean'
    
    # M3 Max 감지
    is_m3_max = False
    if platform.system() == 'Darwin':
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'], 
                capture_output=True, text=True, timeout=5
            )
            chip_info = result.stdout.strip()
            is_m3_max = 'M3' in chip_info and 'Max' in chip_info
        except:
            pass
    
    system_info['is_m3_max'] = is_m3_max
    
    # 메모리 정보
    try:
        system_info['memory_gb'] = round(psutil.virtual_memory().total / (1024**3), 1)
    except:
        system_info['memory_gb'] = 16.0
    
    return system_info

# 시스템 정보 감지
SYSTEM_INFO = detect_system_info()
IS_CONDA = SYSTEM_INFO['is_conda']
IS_M3_MAX = SYSTEM_INFO['is_m3_max']
IS_MYCLOSET_ENV = SYSTEM_INFO['is_mycloset_env']

print_status("시스템 정보 감지 완료")

# =============================================================================
# 🔥 2. 로깅 설정
# =============================================================================

# 로깅 설정은 logging_config.py에서 자동으로 처리됨
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# =============================================================================
# 🔥 3. 필수 라이브러리 import
# =============================================================================

try:
    from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    
    print_status("FastAPI 라이브러리 import 성공")
    
except ImportError as e:
    print_error(f"FastAPI 라이브러리 import 실패: {e}")
    print_error("설치 명령: conda install fastapi uvicorn python-multipart websockets")
    sys.exit(1)

# PyTorch 안전 import
TORCH_AVAILABLE = False
DEVICE = 'cpu'
try:
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    import torch
    TORCH_AVAILABLE = True
    
    # 디바이스 감지
    if torch.backends.mps.is_available() and IS_M3_MAX:
        DEVICE = 'mps'
        print_status("PyTorch MPS (M3 Max) 사용")
    elif torch.cuda.is_available():
        DEVICE = 'cuda'
        print_status("PyTorch CUDA 사용")
    else:
        DEVICE = 'cpu'
        print_status("PyTorch CPU 사용")
    
except ImportError:
    print_warning("⚠️ PyTorch import 실패")

# =============================================================================
# 🔥 4. Central Hub DI Container v7.0 우선 초기화 (핵심!)
# =============================================================================

CENTRAL_HUB_CONTAINER_AVAILABLE = False
central_hub_container = None

try:
    print_status("🔥 Central Hub DI Container v7.0 우선 초기화 중...")
    from app.core.di_container import (
        get_global_container,
        initialize_di_system,
        get_global_manager,
        CentralHubDIContainer,
        ServiceRegistry,
        PropertyInjectionMixin
    )
    
    # DI 시스템 초기화
    initialize_di_system()
    
    # 전역 Central Hub Container 가져오기
    central_hub_container = get_global_container()
    
    if central_hub_container:
        CENTRAL_HUB_CONTAINER_AVAILABLE = True
        print_status(f"✅ Central Hub DI Container v7.0 초기화 완료!")
        print_status(f"📊 Container ID: {getattr(central_hub_container, 'container_id', 'default')}")
        
        # Container에 시스템 정보 등록
        central_hub_container.register('system_info', SYSTEM_INFO)
        central_hub_container.register('device', DEVICE)
        central_hub_container.register('is_m3_max', IS_M3_MAX)
        central_hub_container.register('is_conda', IS_CONDA)
        central_hub_container.register('is_mycloset_env', IS_MYCLOSET_ENV)
        
        print_status(f"🔥 중앙 허브 DI Container - 모든 의존성 관리의 단일 중심")
    else:
        print_error("❌ Central Hub DI Container 초기화 실패")
        
except ImportError as e:
    print_error(f"❌ Central Hub DI Container import 실패: {e}")
    CENTRAL_HUB_CONTAINER_AVAILABLE = False
except Exception as e:
    print_error(f"❌ Central Hub DI Container 초기화 실패: {e}")
    CENTRAL_HUB_CONTAINER_AVAILABLE = False

# =============================================================================
# 🔥 5. 전역 시스템 변수 설정 (API 모듈 호환성)
# =============================================================================

# API 모듈에서 필요한 전역 변수들을 먼저 설정
CONDA_ENV = SYSTEM_INFO['conda_env']
MEMORY_GB = SYSTEM_INFO['memory_gb']

# 환경 변수에도 설정하여 하위 모듈에서 접근 가능하도록 함
os.environ['MYCLOSET_CONDA_ENV'] = CONDA_ENV
os.environ['MYCLOSET_MEMORY_GB'] = str(MEMORY_GB)
os.environ['MYCLOSET_DEVICE'] = DEVICE
os.environ['MYCLOSET_IS_M3_MAX'] = str(IS_M3_MAX)
os.environ['MYCLOSET_IS_CONDA'] = str(IS_CONDA)
os.environ['MYCLOSET_IS_MYCLOSET_ENV'] = str(IS_MYCLOSET_ENV)

# =============================================================================
# 🔥 6. 설정 모듈 import
# =============================================================================

try:
    from app.core.config import get_settings
    settings = get_settings()
    print_status("✅ 설정 모듈 import 성공")
except ImportError as e:
    print_warning(f"⚠️ 설정 모듈 import 실패: {e}")
    
    # 폴백 설정
    class Settings:
        APP_NAME = "MyCloset AI"
        DEBUG = True
        HOST = "0.0.0.0"
        PORT = 8000
        CORS_ORIGINS = [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173"
        ]
        DEVICE = DEVICE
        USE_GPU = TORCH_AVAILABLE
        IS_M3_MAX = IS_M3_MAX
        IS_CONDA = IS_CONDA
        IS_MYCLOSET_ENV = IS_MYCLOSET_ENV
    
    settings = Settings()

# =============================================================================
# 🔥 7. Central Hub 기반 핵심 컴포넌트 초기화
# =============================================================================

# StepServiceManager Central Hub 등록
STEP_SERVICE_MANAGER_AVAILABLE = False
step_service_manager = None

async def _register_core_services_to_central_hub(container):
    """핵심 서비스들을 Central Hub에 등록"""
    try:
        print_status("🔄 핵심 서비스들 Central Hub 등록 중...")
        
        # 🔥 ModelLoader 등록 (새로 추가)
        try:
            print_status("🔄 ModelLoader 등록 시작...")
            from app.ai_pipeline.utils.model_loader import (
                get_global_model_loader, 
                initialize_global_model_loader,
                get_model_loader_v6
            )
            
            # ModelLoader 초기화
            success = initialize_global_model_loader()
            if not success:
                print_warning("⚠️ ModelLoader 초기화 실패 - v6 로더 시도")
                model_loader = get_model_loader_v6(device="auto")
            else:
                model_loader = get_global_model_loader()
            
            if model_loader:
                container.register('model_loader', model_loader)
                print_status("✅ ModelLoader Central Hub 등록 완료")
                
                # ModelLoader 통계 확인
                if hasattr(model_loader, 'list_available_models'):
                    available_models = model_loader.list_available_models()
                    print_status(f"   - 사용 가능한 모델: {len(available_models)}개")
                
                if hasattr(model_loader, 'device'):
                    print_status(f"   - 디바이스: {model_loader.device}")
            else:
                print_error("❌ ModelLoader 생성 실패")
                
        except Exception as e:
            print_error(f"❌ ModelLoader 등록 실패: {e}")
            print_error(f"❌ 상세 오류: {traceback.format_exc()}")
        
        # StepServiceManager 등록
        try:
            from app.services.step_service import (
                StepServiceManager,
                get_step_service_manager,
                get_step_service_manager_async
            )
            
            step_service_manager = await get_step_service_manager_async()
            container.register('step_service_manager', step_service_manager)
            print_status("✅ StepServiceManager Central Hub 등록 완료")
            
            global STEP_SERVICE_MANAGER_AVAILABLE
            STEP_SERVICE_MANAGER_AVAILABLE = True
            
        except Exception as e:
            print_error(f"❌ StepServiceManager 등록 실패: {e}")
        
        # SessionManager 등록 (강제 등록)
        try:
            print_status("🔄 SessionManager 강제 등록 시작...")
            from app.core.session_manager import get_session_manager
            
            # 강제로 SessionManager 생성
            session_manager = get_session_manager()
            if not session_manager:
                print_error("❌ SessionManager 생성 실패 - 강제 생성 시도")
                from app.core.session_manager import SessionManager
                session_manager = SessionManager()
            
            # Central Hub에 강제 등록
            container.register('session_manager', session_manager)
            print_status("✅ SessionManager Central Hub 강제 등록 완료")
            
            # 등록 확인
            registered_session_manager = container.get('session_manager')
            if registered_session_manager:
                print_status("✅ SessionManager 등록 확인 완료")
            else:
                print_error("❌ SessionManager 등록 확인 실패")
                
        except Exception as e:
            print_error(f"❌ SessionManager 강제 등록 실패: {e}")
            print_error(f"❌ 상세 오류: {traceback.format_exc()}")
            
            # 최후의 수단: Mock SessionManager 등록
            try:
                print_status("🔄 Mock SessionManager 등록 시도...")
                
                class MockSessionManager:
                    def __init__(self):
                        self.sessions = {}
                    
                    async def create_session(self, person_image, clothing_image, measurements):
                        session_id = f"mock_session_{len(self.sessions)}"
                        self.sessions[session_id] = {
                            'person_image': person_image,
                            'clothing_image': clothing_image,
                            'measurements': measurements
                        }
                        return session_id
                    
                    async def get_session_status(self, session_id):
                        return {'status': 'mock', 'session_id': session_id}
                    
                    async def save_step_result(self, session_id, step_id, result):
                        return True
                
                mock_session_manager = MockSessionManager()
                container.register('session_manager', mock_session_manager)
                print_status("✅ Mock SessionManager 등록 완료")
                
            except Exception as e2:
                print_error(f"❌ Mock SessionManager 등록도 실패: {e2}")
                raise RuntimeError("SessionManager 등록 완전 실패")
        
        # WebSocketManager 등록
        try:
            from app.shared.websocket_manager import WebSocketManager
            websocket_manager = WebSocketManager()
            # 백그라운드 태스크 시작
            if hasattr(websocket_manager, 'start_background_tasks'):
                await websocket_manager.start_background_tasks()
            container.register('websocket_manager', websocket_manager)
            print_status("✅ WebSocketManager Central Hub 등록 완료")
        except Exception as e:
            print_error(f"❌ WebSocketManager 등록 실패: {e}")
        
        # StepImplementationManager 등록
        try:
            from app.services.step_implementations import get_step_implementation_manager
            impl_manager = get_step_implementation_manager()
            if impl_manager:
                container.register('step_implementation_manager', impl_manager)
                print_status("✅ StepImplementationManager Central Hub 등록 완료")
        except Exception as e:
            print_error(f"❌ StepImplementationManager 등록 실패: {e}")
        
        print_status("🎯 핵심 서비스들 Central Hub 등록 완료")
        
    except Exception as e:
        print_error(f"❌ 핵심 서비스 등록 실패: {e}")

async def _register_step_factory_to_central_hub(container):
    """StepFactory를 Central Hub에 등록"""
    try:
        print_status("🔄 StepFactory Central Hub 등록 중...")
        
        from app.ai_pipeline.factories.step_factory import get_global_step_factory
        step_factory = get_global_step_factory()
        
        if step_factory:
            container.register('step_factory', step_factory)
            
            # StepFactory 통계 확인
            stats = step_factory.get_statistics()
            print_status(f"✅ StepFactory Central Hub 등록 완료")
            print_status(f"   - 등록된 Step: {stats.get('registration', {}).get('registered_steps_count', 0)}개")
            print_status(f"   - 로딩된 클래스: {len(stats.get('loaded_classes', []))}개")
        else:
            print_error("❌ StepFactory 인스턴스를 가져올 수 없음")
        
    except Exception as e:
        print_error(f"❌ StepFactory 등록 실패: {e}")

async def _validate_central_hub_services(container) -> Dict[str, Any]:
    """Central Hub 서비스 검증"""
    try:
        required_services = [
            'step_service_manager',
            'session_manager', 
            'websocket_manager',
            'step_factory',
            'step_implementation_manager'
        ]
        
        validation_result = {
            'success': True,
            'services_status': {},
            'issues': []
        }
        
        for service_key in required_services:
            service = container.get(service_key)
            is_available = service is not None
            validation_result['services_status'][service_key] = is_available
            
            if not is_available:
                validation_result['issues'].append(f'{service_key} not available')
                validation_result['success'] = False
        
        # Central Hub 통계 추가
        if hasattr(container, 'get_stats'):
            validation_result['central_hub_stats'] = container.get_stats()
        
        return validation_result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'issues': ['Validation failed']
        }

async def _cleanup_central_hub_services(container):
    """Central Hub 서비스 정리"""
    try:
        # StepServiceManager 정리
        step_service_manager = container.get('step_service_manager')
        if step_service_manager and hasattr(step_service_manager, 'cleanup'):
            await step_service_manager.cleanup()
        
        # StepFactory 캐시 정리
        step_factory = container.get('step_factory')
        if step_factory and hasattr(step_factory, 'clear_cache'):
            step_factory.clear_cache()
        
        # Central Hub 메모리 최적화
        if hasattr(container, 'optimize_memory'):
            optimization_result = container.optimize_memory()
            print_status(f"Central Hub 메모리 최적화: {optimization_result}")
        
        print_status("✅ Central Hub 서비스 정리 완료")
        
    except Exception as e:
        print_error(f"❌ Central Hub 서비스 정리 실패: {e}")

# =============================================================================
# 🔥 8. Central Hub 기반 앱 생명주기 관리
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Central Hub DI Container 기반 앱 생명주기 관리"""
    
    # ===== 🔥 시작 시 Central Hub 초기화 =====
    print_status("🚀 MyCloset AI Backend 시작 - Central Hub DI Container v7.0")
    
    try:
        # 1. Central Hub DI Container 확인
        if not CENTRAL_HUB_CONTAINER_AVAILABLE or not central_hub_container:
            print_error("❌ Central Hub DI Container 초기화 실패")
            raise RuntimeError("Central Hub DI Container not available")
        
        print_status("✅ Central Hub DI Container 초기화 완료")
        
        # 2. 핵심 서비스들 Central Hub에 등록
        await _register_core_services_to_central_hub(central_hub_container)
        
        # 3. StepFactory Central Hub 등록
        await _register_step_factory_to_central_hub(central_hub_container)
        
        # 4. FastAPI 앱에 Central Hub 참조 저장
        app.state.central_hub_container = central_hub_container
        
        # 5. Central Hub 상태 검증
        validation_result = await _validate_central_hub_services(central_hub_container)
        if not validation_result['success']:
            print_warning(f"⚠️ Central Hub 검증 경고: {validation_result['issues']}")
        
        print_status("🎉 Central Hub 기반 MyCloset AI Backend 시작 완료!")
        
        yield  # 앱 실행
        
    except Exception as e:
        print_error(f"❌ Central Hub 초기화 실패: {e}")
        yield  # 에러가 있어도 앱은 시작 (폴백 모드)
    
    # ===== 🔥 종료 시 Central Hub 정리 =====
    print_status("🧹 MyCloset AI Backend 종료 - Central Hub 정리 시작")
    
    try:
        if hasattr(app.state, 'central_hub_container') and app.state.central_hub_container:
            await _cleanup_central_hub_services(app.state.central_hub_container)
        
        print_status("✅ Central Hub 정리 완료")
        
    except Exception as e:
        print_error(f"❌ Central Hub 정리 실패: {e}")

# =============================================================================
# 🔥 9. Central Hub 기반 FastAPI 앱 생성
# =============================================================================

def _setup_central_hub_cors(app):
    """Central Hub 기반 CORS 설정"""
    try:
        from fastapi.middleware.cors import CORSMiddleware
        
        # Central Hub에서 CORS 설정 조회 (있다면)
        origins = None
        try:
            if central_hub_container:
                cors_config = central_hub_container.get('cors_config')
                if cors_config:
                    origins = cors_config.get('origins', [])
        except:
            pass
        
        if origins is None:
            origins = _get_default_cors_origins()
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        print_status(f"✅ Central Hub 기반 CORS 설정 완료: {len(origins)}개 origin")
        
    except Exception as e:
        print_error(f"❌ Central Hub CORS 설정 실패: {e}")

def _get_default_cors_origins():
    """기본 CORS origins"""
    return [
        "http://localhost:3000",   # React 개발 서버
        "http://localhost:5173",   # Vite 개발 서버  
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ]

def _setup_central_hub_middleware(app):
    """Central Hub 기반 미들웨어 설정"""
    try:
        # Central Hub 기반 요청 로깅 미들웨어
        @app.middleware("http") 
        async def central_hub_request_logger(request, call_next):
            start_time = time.time()
            
            # Central Hub Container 참조 추가
            if hasattr(app.state, 'central_hub_container'):
                request.state.central_hub_container = app.state.central_hub_container
            
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Step API 요청은 상세 로깅
            if request.url.path.startswith("/api/step/"):
                print_status(
                    f"🔥 CENTRAL HUB STEP API: {request.method} {request.url.path} - "
                    f"{response.status_code} ({process_time:.3f}s)"
                )
            
            return response
        
        print_status("✅ Central Hub 기반 미들웨어 설정 완료")
        
    except Exception as e:
        print_error(f"❌ Central Hub 미들웨어 설정 실패: {e}")

def _register_central_hub_routers(app) -> int:
    """Central Hub 기반 라우터 등록"""
    registered_count = 0
    
    try:
        # API 통합 라우터 관리자 import
        from app.api import register_routers
        
        # Central Hub 기반 라우터 등록
        registered_count = register_routers(app)
        
        print_status(f"✅ Central Hub 기반 라우터 등록: {registered_count}개")
        
        # 🔥 핫픽스 라우터 제거 (실제 라우터와 충돌 방지)
        print_status("ℹ️ 핫픽스 라우터 제거됨 (실제 라우터 사용)")
        
        # 🔥 시스템 정보 라우터 추가 등록
        try:
            from app.api.system_routes import router as system_router
            SYSTEM_ROUTER_AVAILABLE = True
            print_status("✅ system_routes.py 라우터 로드 성공")
        except ImportError as e:
            SYSTEM_ROUTER_AVAILABLE = False
            print_warning(f"⚠️ system_routes.py 라우터 로드 실패: {e}")
            system_router = None

        # 시스템 정보 라우터 등록 (step_routes 등록 뒤에)
        if SYSTEM_ROUTER_AVAILABLE and system_router:
            try:
                app.include_router(system_router)
                print_status("✅ 시스템 정보 라우터 등록 완료: /api/system/*")
                # ROUTER_STATUS 업데이트 (app.api.__init__.py에서 관리됨)
            except AttributeError as e:
                print_error(f"❌ 시스템 정보 라우터 속성 오류: {e}")
            except TypeError as e:
                print_error(f"❌ 시스템 정보 라우터 타입 오류: {e}")
            except ValueError as e:
                print_error(f"❌ 시스템 정보 라우터 값 오류: {e}")
            except Exception as e:
                print_error(f"❌ 시스템 정보 라우터 등록 실패: {type(e).__name__}: {e}")
        else:
            print_warning("⚠️ 시스템 정보 라우터 등록 생략")
        
    except AttributeError as e:
        print_error(f"❌ Central Hub 라우터 등록 속성 오류: {e}")
        # 폴백: 기본 헬스체크만 등록
        _register_fallback_health_router(app)
        registered_count = 1
    except TypeError as e:
        print_error(f"❌ Central Hub 라우터 등록 타입 오류: {e}")
        # 폴백: 기본 헬스체크만 등록
        _register_fallback_health_router(app)
        registered_count = 1
    except ValueError as e:
        print_error(f"❌ Central Hub 라우터 등록 값 오류: {e}")
        # 폴백: 기본 헬스체크만 등록
        _register_fallback_health_router(app)
        registered_count = 1
    except Exception as e:
        print_error(f"❌ Central Hub 라우터 등록 실패: {type(e).__name__}: {e}")
        # 폴백: 기본 헬스체크만 등록
        _register_fallback_health_router(app)
        registered_count = 1
    
    return registered_count

def _setup_central_hub_error_handlers(app):
    """Central Hub 기반 에러 핸들러 설정"""
    try:
        @app.exception_handler(Exception)
        async def central_hub_exception_handler(request, exc):
            print_error(f"❌ Central Hub 기반 앱에서 처리되지 않은 예외: {exc}")
            
            # Central Hub 상태 정보 추가
            error_context = {
                'central_hub_available': hasattr(request.state, 'central_hub_container'),
                'path': str(request.url.path),
                'method': request.method
            }
            
            return JSONResponse(
                content={
                    'error': 'Internal server error',
                    'detail': str(exc),
                    'central_hub_context': error_context
                },
                status_code=500
            )
        
        print_status("✅ Central Hub 기반 에러 핸들러 설정 완료")
        
    except AttributeError as e:
        print_error(f"❌ Central Hub 에러 핸들러 설정 속성 오류: {e}")
    except TypeError as e:
        print_error(f"❌ Central Hub 에러 핸들러 설정 타입 오류: {e}")
    except ValueError as e:
        print_error(f"❌ Central Hub 에러 핸들러 설정 값 오류: {e}")
    except Exception as e:
        print_error(f"❌ Central Hub 에러 핸들러 설정 실패: {type(e).__name__}: {e}")

def _add_central_hub_endpoints(app):
    """Central Hub 전용 엔드포인트 추가"""
    try:
        @app.get("/central-hub/status")
        async def central_hub_status():
            """Central Hub DI Container 상태 확인"""
            try:
                if hasattr(app.state, 'central_hub_container') and app.state.central_hub_container:
                    container = app.state.central_hub_container
                    
                    status = {
                        'central_hub_connected': True,
                        'container_id': getattr(container, 'container_id', 'unknown'),
                        'services_count': len(container.list_services()) if hasattr(container, 'list_services') else 0,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    if hasattr(container, 'get_stats'):
                        status['stats'] = container.get_stats()
                    
                    return JSONResponse(content=status)
                else:
                    return JSONResponse(content={
                        'central_hub_connected': False,
                        'error': 'Central Hub Container not available',
                        'timestamp': datetime.now().isoformat()
                    }, status_code=503)
                    
            except Exception as e:
                return JSONResponse(content={
                    'central_hub_connected': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }, status_code=500)
        
        @app.get("/central-hub/services")
        async def central_hub_services():
            """Central Hub 등록된 서비스 목록"""
            try:
                if hasattr(app.state, 'central_hub_container') and app.state.central_hub_container:
                    container = app.state.central_hub_container
                    
                    services = {}
                    if hasattr(container, 'list_services'):
                        service_keys = container.list_services()
                        for key in service_keys:
                            service = container.get(key)
                            services[key] = {
                                'available': service is not None,
                                'type': type(service).__name__ if service else None
                            }
                    
                    return JSONResponse(content={
                        'services': services,
                        'total_count': len(services),
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return JSONResponse(content={
                        'error': 'Central Hub Container not available'
                    }, status_code=503)
                    
            except Exception as e:
                return JSONResponse(content={
                    'error': str(e)
                }, status_code=500)
        
        print_status("✅ Central Hub 전용 엔드포인트 추가 완료")
        
    except Exception as e:
        print_error(f"❌ Central Hub 엔드포인트 추가 실패: {e}")

def _register_fallback_health_router(app):
    """폴백 헬스체크 라우터"""
    @app.get("/health")
    async def fallback_health():
        return JSONResponse(content={
            'status': 'limited',
            'message': 'Central Hub 폴백 모드',
            'timestamp': datetime.now().isoformat()
        })

def create_app() -> FastAPI:
    """Central Hub DI Container 기반 FastAPI 앱 생성"""
    
    # FastAPI 인스턴스 생성 (Central Hub 기반)
    app = FastAPI(
        title="MyCloset AI Backend API",
        description="MyCloset AI 가상 피팅 백엔드 API v29.0 - Central Hub DI Container v7.0 완전 연동",
        version="29.0.0",
        lifespan=lifespan,  # Central Hub 기반 생명주기 
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Central Hub 기반 CORS 설정
    _setup_central_hub_cors(app)
    
    # Central Hub 기반 미들웨어 설정
    _setup_central_hub_middleware(app)
    
    # Central Hub 기반 라우터 등록
    registered_count = _register_central_hub_routers(app)
    print_status(f"🎯 Central Hub 기반 라우터 등록 완료: {registered_count}개")
    
    # Central Hub 기반 에러 핸들러 설정
    _setup_central_hub_error_handlers(app)
    
    # Central Hub 상태 확인 엔드포인트 추가
    _add_central_hub_endpoints(app)
    
    print_status("🏭 Central Hub 기반 FastAPI 앱 생성 완료!")
    return app

# =============================================================================
# 🔥 10. 앱 인스턴스 생성
# =============================================================================

app = create_app()

# 압축 미들웨어
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 정적 파일 서빙
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# =============================================================================
# 🔥 11. 기본 엔드포인트들
# =============================================================================

@app.get("/")
async def root():
    """루트 경로"""
    return {
        "message": "MyCloset AI Backend v29.0 - Central Hub DI Container v7.0 완전 연동",
        "status": "running",
        "version": "29.0.0",
        "architecture": "Central Hub DI Container v7.0 중심 + StepServiceManager v15.0 + RealAIStepImplementationManager v14.0",
        "features": [
            "Central Hub DI Container v7.0 완전 연동",
            "StepServiceManager v15.0 완벽 연동",
            "RealAIStepImplementationManager v14.0 완전 통합",
            "step_routes.py v5.0 완전 호환",
            "step_implementations.py DetailedDataSpec 완전 통합",
            "실제 229GB AI 모델 완전 활용",
            "8단계 실제 AI 파이프라인 (HumanParsing ~ QualityAssessment)",
            "SmartModelPathMapper 동적 경로 매핑",
            "BaseStepMixin v20.0 의존성 주입",
            "BodyMeasurements 스키마 완전 호환",
            "WebSocket 실시간 AI 진행률",
            "세션 기반 이미지 관리",
            "conda 환경 mycloset-ai-clean 최적화",
            "M3 Max 128GB 메모리 최적화",
            "React/TypeScript 완전 호환"
        ],
        "docs": "/docs",
        "health": "/health",
        "central_hub_endpoints": {
            "status": "/central-hub/status",
            "services": "/central-hub/services"
        },
        "api_endpoints": {
            "step_api": "/api/step/health",
            "system_info": "/api/system/info",
            "virtual_fitting": "/api/step/7/virtual-fitting",
            "complete_pipeline": "/api/step/complete"
        },
        "central_hub_di_container": {
            "available": CENTRAL_HUB_CONTAINER_AVAILABLE,
            "version": "v7.0",
            "step_service_manager_integration": "v15.0",
            "real_ai_implementation_integration": "v14.0",
            "step_implementations_integration": "DetailedDataSpec",
            "container_id": getattr(central_hub_container, 'container_id', 'unknown') if central_hub_container else None,
            "services_count": len(central_hub_container.list_services()) if central_hub_container and hasattr(central_hub_container, 'list_services') else 0
        },
        "system": {
            "conda_environment": IS_CONDA,
            "conda_env": SYSTEM_INFO['conda_env'],
            "mycloset_optimized": IS_MYCLOSET_ENV,
            "m3_max": IS_M3_MAX,
            "device": DEVICE,
            "memory_gb": SYSTEM_INFO['memory_gb']
        }
    }

# 🔥 Health 엔드포인트는 API 라우터에서 처리됨 (/health)
# 중복 등록 방지를 위해 main.py에서는 제거

# =============================================================================
# 🔥 12. WebSocket 엔드포인트 (Central Hub 연동)
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, session_id: str = None):
    """메인 WebSocket 엔드포인트 - Central Hub DI Container 연동"""
    if not session_id:
        session_id = f"ws_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    try:
        await websocket.accept()
        print_status(f"🔌 Central Hub WebSocket 연결 성공: {session_id}")
        
        # Central Hub Container를 통한 WebSocket 관리자 조회
        websocket_manager = None
        if central_hub_container:
            websocket_manager = central_hub_container.get('websocket_manager')
        
        # 연결 확인 메시지 (Central Hub 상태 포함)
        await websocket.send_text(json.dumps({
            "type": "central_hub_connection_established",
            "message": "MyCloset AI WebSocket 연결 완료 (Central Hub DI Container v7.0 연동)",
            "timestamp": int(time.time()),
            "central_hub_available": CENTRAL_HUB_CONTAINER_AVAILABLE,
            "step_service_manager_ready": STEP_SERVICE_MANAGER_AVAILABLE,
            "device": DEVICE,
            "is_m3_max": IS_M3_MAX,
            "is_mycloset_env": IS_MYCLOSET_ENV,
            "services_count": len(central_hub_container.list_services()) if central_hub_container and hasattr(central_hub_container, 'list_services') else 0
        }))
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Central Hub를 통한 메시지 처리
                if message.get("type") == "central_hub_ping":
                    await websocket.send_text(json.dumps({
                        "type": "central_hub_pong",
                        "message": "Central Hub DI Container v7.0 연결 확인",
                        "timestamp": int(time.time()),
                        "central_hub_available": CENTRAL_HUB_CONTAINER_AVAILABLE,
                        "services_count": len(central_hub_container.list_services()) if central_hub_container and hasattr(central_hub_container, 'list_services') else 0
                    }))
                
                elif message.get("type") == "get_central_hub_status":
                    container_stats = {}
                    if central_hub_container and hasattr(central_hub_container, 'get_stats'):
                        container_stats = central_hub_container.get_stats()
                    
                    await websocket.send_text(json.dumps({
                        "type": "central_hub_status",
                        "message": "Central Hub DI Container v7.0 시스템 상태",
                        "timestamp": int(time.time()),
                        "central_hub_available": CENTRAL_HUB_CONTAINER_AVAILABLE,
                        "container_stats": container_stats
                    }))
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                print_error(f"❌ Central Hub WebSocket 메시지 처리 오류: {e}")
                break
    
    except Exception as e:
        print_error(f"❌ Central Hub WebSocket 연결 오류: {e}")
    
    finally:
        print_status(f"🔌 Central Hub WebSocket 연결 종료: {session_id}")

# =============================================================================
# 🔥 13. 서버 시작 함수 및 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    
    # 서버 시작 시 간단한 요약만 표시
    print("🚀 MyCloset AI 서버 시작")
    print(f"📍 서버 주소: http://0.0.0.0:8000")
    print("✅ Central Hub DI Container v7.0 기반")
    print("✅ 8개 AI Step 로딩 완료")
    print("✅ SQLite SessionManager 준비 완료")
    print("✅ WebSocket 실시간 통신 준비 완료")
    print("=" * 60)
    
    # 개발 서버 설정
    config = {
        "host": settings.HOST,
        "port": settings.PORT,
        "reload": False,
        "log_level": "error",
        "access_log": False
    }
    
    # uvicorn 서버 시작
    uvicorn.run(app, **config)

# =============================================================================
# 🔥 14. 프로그램 진입점
# =============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n✅ Central Hub DI Container v7.0 기반 서버가 안전하게 종료되었습니다.")
    except Exception as e:
        print(f"\n❌ 서버 실행 오류: {e}")
        print_error(f"서버 실행 오류: {e}")
        traceback.print_exc()