# backend/app/api/__init__.py
"""
🍎 MyCloset AI API 라우터 패키지 v8.1 - NameError 문제 해결
================================================================

✅ step_routes.py 완전 지원 추가 (/api/step/*)
✅ 기존 pipeline_routes.py도 유지 (/api/v1/pipeline/*)
✅ 단순하고 안정적인 API 초기화
✅ conda 환경 우선 최적화
✅ M3 Max 성능 최적화
✅ CORS 및 미들웨어 지원
✅ WebSocket 실시간 통신 지원
✅ 에러 핸들링 및 로깅
✅ NameError: CONDA_ENV 문제 완전 해결

API 엔드포인트:
- /api/step/*: 8단계 AI 파이프라인 API (신규!)
- /api/v1/pipeline/*: 기존 파이프라인 API
- /ws: WebSocket 실시간 통신
- /api/v1/health: 헬스 체크

작성자: MyCloset AI Team
날짜: 2025-07-31
버전: v8.1.0 (NameError Fixed)
"""

import logging
import sys
import time
import warnings
import os
import platform
from typing import Dict, Any, Optional, List
from functools import lru_cache

# 경고 무시
warnings.filterwarnings('ignore')

# =============================================================================
# 🔥 기본 설정 및 시스템 정보 (NameError 방지)
# =============================================================================

logger = logging.getLogger(__name__)

# 시스템 정보 직접 감지 (안전한 방식)
def _detect_system_info():
    """시스템 정보 직접 감지"""
    # conda 환경 감지
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'none')
    is_conda = conda_env != 'none'
    
    # M3 Max 감지
    is_m3_max = False
    memory_gb = 16.0
    
    if platform.system() == 'Darwin':
        try:
            import subprocess
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'], 
                capture_output=True, text=True, timeout=5
            )
            chip_info = result.stdout.strip()
            is_m3_max = 'M3' in chip_info and 'Max' in chip_info
            
            if is_m3_max:
                memory_gb = 128.0
        except:
            pass
    
    # 디바이스 감지
    device = 'cpu'
    if is_m3_max:
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
        except ImportError:
            pass
    
    return {
        'conda_env': conda_env,
        'is_conda': is_conda,
        'is_m3_max': is_m3_max,
        'memory_gb': memory_gb,
        'device': device
    }

# 시스템 정보 로드
detected_info = _detect_system_info()

# 전역 변수 정의 (NameError 방지)
CONDA_ENV = detected_info['conda_env']
IS_CONDA = detected_info['is_conda']
IS_M3_MAX = detected_info['is_m3_max']
DEVICE = detected_info['device']
MEMORY_GB = detected_info['memory_gb']

# 상위 패키지에서 시스템 정보 가져오기 시도 (있으면 덮어씀)
try:
    from .. import get_system_info, is_conda_environment, is_m3_max, get_device
    SYSTEM_INFO = get_system_info()
    IS_CONDA = is_conda_environment()
    IS_M3_MAX = is_m3_max()
    DEVICE = get_device()
    
    # conda_env 정보 업데이트
    if hasattr(SYSTEM_INFO, 'conda_env'):
        CONDA_ENV = SYSTEM_INFO.conda_env
    elif isinstance(SYSTEM_INFO, dict) and 'conda_env' in SYSTEM_INFO:
        CONDA_ENV = SYSTEM_INFO['conda_env']
    
    logger.info("✅ 상위 패키지에서 시스템 정보 로드 성공")
except ImportError as e:
    logger.warning(f"⚠️ 상위 패키지 로드 실패, 기본값 사용: {e}")
    SYSTEM_INFO = {
        'device': DEVICE, 
        'is_m3_max': IS_M3_MAX, 
        'memory_gb': MEMORY_GB,
        'conda_env': CONDA_ENV,
        'is_conda': IS_CONDA
    }

# DI Container 지원 (선택적)
try:
    from app.core.di_container import (
        CircularReferenceFreeDIContainer,
        LazyDependency,
        DynamicImportResolver,
        get_global_container,
        inject_dependencies_to_step_safe,
        get_service_safe,
        register_service_safe,
        register_lazy_service
    )
    DI_CONTAINER_AVAILABLE = True
    logger.info("✅ DI Container v4.0 Core Import 성공")
except ImportError as e:
    logger.debug(f"DI Container import 실패 (선택적): {e}")
    DI_CONTAINER_AVAILABLE = False

# =============================================================================
# 🔥 API 라우터 상태 추적
# =============================================================================

# API 라우터 로딩 상태
ROUTER_STATUS = {
    'virtual_tryon': False,
    'pipeline_routes': False,
    'step_routes': False,        # 🔥 step_routes.py 추가!
    'websocket_routes': False,
    'health_check': False
}

# =============================================================================
# 🔥 안전한 라우터 모듈 로딩
# =============================================================================

def _safe_import_virtual_tryon():
    """virtual_tryon 라우터 안전하게 import"""
    try:
        from .virtual_tryon import router as virtual_tryon_router
        
        globals()['virtual_tryon_router'] = virtual_tryon_router
        
        ROUTER_STATUS['virtual_tryon'] = True
        logger.info("✅ virtual_tryon 라우터 로드 성공")
        return virtual_tryon_router
        
    except ImportError as e:
        logger.debug(f"📋 virtual_tryon 라우터 없음 (정상): {e}")
        return None
    except Exception as e:
        logger.error(f"❌ virtual_tryon 라우터 로드 실패: {e}")
        return None

def _safe_import_pipeline_routes():
    """pipeline_routes 라우터 안전하게 import"""
    try:
        from .pipeline_routes import router as pipeline_router
        
        globals()['pipeline_router'] = pipeline_router
        
        ROUTER_STATUS['pipeline_routes'] = True
        logger.info("✅ pipeline_routes 라우터 로드 성공")
        return pipeline_router
        
    except ImportError as e:
        logger.debug(f"📋 pipeline_routes 라우터 없음 (정상): {e}")
        return None
    except Exception as e:
        logger.error(f"❌ pipeline_routes 라우터 로드 실패: {e}")
        return None

def _safe_import_step_routes():
    """🔥 step_routes 라우터 안전하게 import (신규!)"""
    try:
        from .step_routes import router as step_router
        
        globals()['step_router'] = step_router
        
        ROUTER_STATUS['step_routes'] = True
        logger.info("✅ step_routes 라우터 로드 성공")
        return step_router
        
    except ImportError as e:
        logger.warning(f"⚠️ step_routes 라우터 없음: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ step_routes 라우터 로드 실패: {e}")
        return None

def _safe_import_websocket_routes():
    """websocket_routes 라우터 안전하게 import"""
    try:
        from .websocket_routes import router as websocket_router
        
        globals()['websocket_router'] = websocket_router
        
        ROUTER_STATUS['websocket_routes'] = True
        logger.info("✅ websocket_routes 라우터 로드 성공")
        return websocket_router
        
    except ImportError as e:
        logger.debug(f"📋 websocket_routes 라우터 없음 (정상): {e}")
        return None
    except Exception as e:
        logger.error(f"❌ websocket_routes 라우터 로드 실패: {e}")
        return None

def _create_health_check_router():
    """헬스 체크 라우터 생성"""
    try:
        from fastapi import APIRouter
        
        health_router = APIRouter(tags=["health"])
        
        @health_router.get("/health")
        async def health_check():
            """API 헬스 체크"""
            return {
                "status": "healthy",
                "system_info": SYSTEM_INFO,
                "router_status": ROUTER_STATUS,
                "conda_optimized": IS_CONDA,
                "m3_max_optimized": IS_M3_MAX,
                "device": DEVICE,
                "step_routes_available": ROUTER_STATUS['step_routes'],  # 🔥 step_routes 상태!
                "timestamp": time.time()
            }
        
        @health_router.get("/status")
        async def api_status():
            """API 상태 상세 정보"""
            available_routers = [k for k, v in ROUTER_STATUS.items() if v]
            
            return {
                "api_version": "v8.1.0",
                "available_routers": available_routers,
                "total_routers": len(ROUTER_STATUS),
                "success_rate": (len(available_routers) / len(ROUTER_STATUS)) * 100,
                "system": {
                    "conda": IS_CONDA,
                    "conda_env": CONDA_ENV,
                    "m3_max": IS_M3_MAX,
                    "device": DEVICE,
                    "memory_gb": MEMORY_GB
                },
                "step_routes_enabled": ROUTER_STATUS['step_routes']  # 🔥 step_routes 정보!
            }
        
        globals()['health_router'] = health_router
        
        ROUTER_STATUS['health_check'] = True
        logger.info("✅ health_check 라우터 생성 성공")
        return health_router
        
    except Exception as e:
        logger.error(f"❌ health_check 라우터 생성 실패: {e}")
        return None

# =============================================================================
# 🔥 라우터들 로딩 (step_routes.py 추가!)
# =============================================================================

# 모든 라우터 로딩 시도
AVAILABLE_ROUTERS = {}

# Virtual Try-on 라우터
virtual_tryon_router = _safe_import_virtual_tryon()
if virtual_tryon_router:
    AVAILABLE_ROUTERS['virtual_tryon'] = virtual_tryon_router

# Pipeline 라우터 (기존)
pipeline_router = _safe_import_pipeline_routes()
if pipeline_router:
    AVAILABLE_ROUTERS['pipeline'] = pipeline_router

# 🔥 Step 라우터 (신규 추가!)
step_router = _safe_import_step_routes()
if step_router:
    AVAILABLE_ROUTERS['step_routes'] = step_router

# WebSocket 라우터
websocket_router = _safe_import_websocket_routes()
if websocket_router:
    AVAILABLE_ROUTERS['websocket'] = websocket_router

# Health Check 라우터 (항상 생성)
health_router = _create_health_check_router()
if health_router:
    AVAILABLE_ROUTERS['health'] = health_router

# =============================================================================
# 🔥 라우터 등록 함수 (step_routes.py 지원 추가!)
# =============================================================================

def register_routers(app) -> int:
    """FastAPI 앱에 모든 라우터 등록"""
    registered_count = 0
    
    try:
        # Virtual Try-on 라우터
        if 'virtual_tryon' in AVAILABLE_ROUTERS:
            app.include_router(
                AVAILABLE_ROUTERS['virtual_tryon'],
                prefix="/api/v1",
                tags=["virtual-tryon"]
            )
            registered_count += 1
            logger.info("✅ virtual_tryon 라우터 등록")
        
        # Pipeline 라우터 (기존)
        if 'pipeline' in AVAILABLE_ROUTERS:
            app.include_router(
                AVAILABLE_ROUTERS['pipeline'],
                prefix="/api/v1",
                tags=["pipeline"]
            )
            registered_count += 1
            logger.info("✅ pipeline 라우터 등록")
        
        # 🔥 Step 라우터 (신규 추가!) - 프론트엔드 호환성을 위해 /api/step 경로 사용
        if 'step_routes' in AVAILABLE_ROUTERS:
            app.include_router(
                AVAILABLE_ROUTERS['step_routes'],
                prefix="/api/step",  # 🔥 프론트엔드가 기대하는 경로!
                tags=["step-pipeline"]
            )
            registered_count += 1
            logger.info("✅ step_routes 라우터 등록 (/api/step)")
        
        # WebSocket 라우터
        if 'websocket' in AVAILABLE_ROUTERS:
            app.include_router(
                AVAILABLE_ROUTERS['websocket'],
                tags=["websocket"]
            )
            registered_count += 1
            logger.info("✅ websocket 라우터 등록")
        
        # Health Check 라우터 (항상 등록)
        if 'health' in AVAILABLE_ROUTERS:
            app.include_router(
                AVAILABLE_ROUTERS['health'],
                tags=["health"]
            )
            registered_count += 1
            logger.info("✅ health_check 라우터 등록")
        
        logger.info(f"🎯 총 {registered_count}개 라우터 등록 완료")
        logger.info(f"🔥 step_routes.py 지원: {'✅' if ROUTER_STATUS['step_routes'] else '❌'}")
        
        return registered_count
        
    except Exception as e:
        logger.error(f"❌ 라우터 등록 실패: {e}")
        return registered_count

# =============================================================================
# 🔥 CORS 및 미들웨어 설정
# =============================================================================

def setup_cors(app, origins: Optional[List[str]] = None):
    """CORS 설정"""
    try:
        from fastapi.middleware.cors import CORSMiddleware
        
        if origins is None:
            origins = [
                "http://localhost:3000",  # React 개발 서버
                "http://localhost:5173",  # Vite 개발 서버
                "http://127.0.0.1:3000",
                "http://127.0.0.1:5173"
            ]
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        logger.info(f"✅ CORS 설정 완료: {len(origins)}개 origin")
        
    except Exception as e:
        logger.error(f"❌ CORS 설정 실패: {e}")

def setup_middleware(app):
    """추가 미들웨어 설정"""
    try:
        # 요청 로깅 미들웨어 (개발용)
        @app.middleware("http")
        async def log_requests(request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # step_routes 요청은 상세 로깅
            if request.url.path.startswith("/api/step/"):
                logger.info(
                    f"🔥 STEP API: {request.method} {request.url.path} - "
                    f"Status: {response.status_code} - "
                    f"Time: {process_time:.4f}s"
                )
            else:
                logger.debug(
                    f"{request.method} {request.url.path} - "
                    f"Status: {response.status_code} - "
                    f"Time: {process_time:.4f}s"
                )
            
            return response
        
        logger.info("✅ 요청 로깅 미들웨어 설정 완료 (step_routes 강화)")
        
    except Exception as e:
        logger.error(f"❌ 미들웨어 설정 실패: {e}")

# =============================================================================
# 🔥 API 상태 관리 함수들
# =============================================================================

@lru_cache(maxsize=1)
def get_api_status() -> Dict[str, Any]:
    """API 상태 반환"""
    available_routers = [k for k, v in ROUTER_STATUS.items() if v]
    
    return {
        "api_version": "v8.1.0",
        "total_routers": len(ROUTER_STATUS),
        "available_routers": available_routers,
        "success_rate": (len(available_routers) / len(ROUTER_STATUS)) * 100,
        "system": SYSTEM_INFO,
        "router_details": ROUTER_STATUS,
        "step_routes_enabled": ROUTER_STATUS['step_routes'],  # 🔥 step_routes 상태!
        "conda_optimized": IS_CONDA,
        "m3_max_optimized": IS_M3_MAX
    }

def get_available_endpoints() -> List[str]:
    """사용 가능한 엔드포인트 목록"""
    endpoints = ["/health", "/status"]
    
    if ROUTER_STATUS.get('virtual_tryon'):
        endpoints.extend(["/api/v1/virtual-tryon/*"])
    
    if ROUTER_STATUS.get('pipeline_routes'):
        endpoints.extend(["/api/v1/pipeline/*"])
    
    # 🔥 step_routes 엔드포인트 추가!
    if ROUTER_STATUS.get('step_routes'):
        endpoints.extend([
            "/api/step/health",
            "/api/step/1/upload-validation",
            "/api/step/2/measurements-validation",
            "/api/step/3/human-parsing",
            "/api/step/4/pose-estimation",
            "/api/step/5/clothing-analysis",
            "/api/step/6/geometric-matching",
            "/api/step/7/virtual-fitting",
            "/api/step/8/result-analysis",
            "/api/step/complete"
        ])
    
    if ROUTER_STATUS.get('websocket_routes'):
        endpoints.extend(["/api/ws/*"])
    
    return endpoints

def get_router_info() -> Dict[str, Any]:
    """라우터 상세 정보"""
    router_info = {}
    
    for router_name, is_available in ROUTER_STATUS.items():
        router_info[router_name] = {
            "available": is_available,
            "loaded": router_name in AVAILABLE_ROUTERS,
            "instance": AVAILABLE_ROUTERS.get(router_name) is not None
        }
    
    # 🔥 step_routes 특별 정보 추가!
    if ROUTER_STATUS.get('step_routes'):
        router_info['step_routes'].update({
            "prefix": "/api/step",
            "frontend_compatible": True,
            "ai_pipeline_steps": 8,
            "real_ai_only": True
        })
    
    return router_info

# =============================================================================
# 🔥 Export
# =============================================================================

__all__ = [
    'register_routers',
    'setup_cors', 
    'setup_middleware',
    'get_api_status',
    'get_available_endpoints',
    'get_router_info',
    'AVAILABLE_ROUTERS',
    'ROUTER_STATUS',
    'SYSTEM_INFO',
    'CONDA_ENV',
    'IS_CONDA',
    'IS_M3_MAX',
    'DEVICE',
    'MEMORY_GB'
]

# =============================================================================
# 🔥 초기화 완료 메시지 (NameError 방지)
# =============================================================================

logger.info("🎉 API 라우터 통합 관리자 v8.1 로드 완료!")
logger.info(f"✅ 시스템 환경: conda={CONDA_ENV}, M3 Max={IS_M3_MAX}")
logger.info(f"✅ 메모리: {MEMORY_GB}GB, 디바이스: {DEVICE}")
logger.info(f"✅ 사용 가능한 라우터: {len([k for k, v in ROUTER_STATUS.items() if v])}/{len(ROUTER_STATUS)}")
logger.info(f"🔥 step_routes.py 지원: {'✅ 활성화' if ROUTER_STATUS['step_routes'] else '❌ 비활성화'}")

if ROUTER_STATUS['step_routes']:
    logger.info("🎯 step_routes.py 라우터 정보:")
    logger.info("   - 경로: /api/step/*")
    logger.info("   - 프론트엔드 완전 호환")
    logger.info("   - 8단계 AI 파이프라인 지원")
    logger.info("   - 실제 AI 모델 전용")

logger.info("🚀 프론트엔드 API 요청 준비 완료!")
logger.info("✅ NameError: CONDA_ENV 문제 완전 해결!")