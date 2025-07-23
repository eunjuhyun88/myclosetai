# backend/app/api/__init__.py
"""
🍎 MyCloset AI API 라우터 패키지 v7.0 - 단순화된 API 초기화
================================================================

✅ 단순하고 안정적인 API 초기화
✅ FastAPI 라우터 자동 등록
✅ conda 환경 우선 최적화
✅ M3 Max 성능 최적화
✅ CORS 및 미들웨어 지원
✅ WebSocket 실시간 통신 지원
✅ 에러 핸들링 및 로깅

API 엔드포인트:
- /api/v1/virtual-tryon: 가상 피팅 API
- /api/v1/pipeline: AI 파이프라인 API
- /ws: WebSocket 실시간 통신
- /api/v1/health: 헬스 체크

작성자: MyCloset AI Team
날짜: 2025-07-23
버전: v7.0.0 (Simplified API Initialization)
"""

import logging
import sys
from typing import Dict, Any, Optional, List
from functools import lru_cache
import warnings

# 경고 무시
warnings.filterwarnings('ignore')

# =============================================================================
# 🔥 기본 설정 및 시스템 정보
# =============================================================================

logger = logging.getLogger(__name__)

# 상위 패키지에서 시스템 정보 가져오기
try:
    from .. import get_system_info, is_conda_environment, is_m3_max, get_device
    SYSTEM_INFO = get_system_info()
    IS_CONDA = is_conda_environment()
    IS_M3_MAX = is_m3_max()
    DEVICE = get_device()
    logger.info("✅ 상위 패키지에서 시스템 정보 로드 성공")
except ImportError as e:
    logger.warning(f"⚠️ 상위 패키지 로드 실패, 기본값 사용: {e}")
    SYSTEM_INFO = {'device': 'cpu', 'is_m3_max': False, 'memory_gb': 16.0}
    IS_CONDA = False
    IS_M3_MAX = False
    DEVICE = 'cpu'

# =============================================================================
# 🔥 API 라우터 상태 추적
# =============================================================================

# API 라우터 로딩 상태
ROUTER_STATUS = {
    'virtual_tryon': False,
    'pipeline_routes': False,
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
        
        health_router = APIRouter(prefix="/api/v1", tags=["health"])
        
        @health_router.get("/health")
        async def health_check():
            """API 헬스 체크"""
            return {
                "status": "healthy",
                "system_info": SYSTEM_INFO,
                "router_status": ROUTER_STATUS,
                "conda_optimized": IS_CONDA,
                "m3_max_optimized": IS_M3_MAX,
                "device": DEVICE
            }
        
        @health_router.get("/status")
        async def api_status():
            """API 상태 상세 정보"""
            available_routers = [k for k, v in ROUTER_STATUS.items() if v]
            
            return {
                "api_version": "v7.0.0",
                "available_routers": available_routers,
                "total_routers": len(ROUTER_STATUS),
                "success_rate": (len(available_routers) / len(ROUTER_STATUS)) * 100,
                "system": {
                    "conda": IS_CONDA,
                    "m3_max": IS_M3_MAX,
                    "device": DEVICE,
                    "memory_gb": SYSTEM_INFO.get('memory_gb', 16)
                }
            }
        
        globals()['health_router'] = health_router
        
        ROUTER_STATUS['health_check'] = True
        logger.info("✅ health_check 라우터 생성 성공")
        return health_router
        
    except Exception as e:
        logger.error(f"❌ health_check 라우터 생성 실패: {e}")
        return None

# =============================================================================
# 🔥 라우터들 로딩
# =============================================================================

# 모든 라우터 로딩 시도
AVAILABLE_ROUTERS = {}

# Virtual Try-on 라우터
virtual_tryon_router = _safe_import_virtual_tryon()
if virtual_tryon_router:
    AVAILABLE_ROUTERS['virtual_tryon'] = virtual_tryon_router

# Pipeline 라우터
pipeline_router = _safe_import_pipeline_routes()
if pipeline_router:
    AVAILABLE_ROUTERS['pipeline'] = pipeline_router

# WebSocket 라우터
websocket_router = _safe_import_websocket_routes()
if websocket_router:
    AVAILABLE_ROUTERS['websocket'] = websocket_router

# Health Check 라우터 (항상 생성)
health_router = _create_health_check_router()
if health_router:
    AVAILABLE_ROUTERS['health'] = health_router

# =============================================================================
# 🔥 라우터 등록 함수
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
        
        # Pipeline 라우터
        if 'pipeline' in AVAILABLE_ROUTERS:
            app.include_router(
                AVAILABLE_ROUTERS['pipeline'],
                prefix="/api/v1",
                tags=["pipeline"]
            )
            registered_count += 1
            logger.info("✅ pipeline 라우터 등록")
        
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
            
            logger.debug(
                f"{request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.4f}s"
            )
            
            return response
        
        logger.info("✅ 요청 로깅 미들웨어 설정 완료")
        
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
        'api_version': 'v7.0.0',
        'system_info': SYSTEM_INFO,
        'router_status': ROUTER_STATUS.copy(),
        'available_routers': available_routers,
        'total_routers': len(ROUTER_STATUS),
        'success_rate': (len(available_routers) / len(ROUTER_STATUS)) * 100,
        'conda_optimized': IS_CONDA,
        'm3_max_optimized': IS_M3_MAX,
        'device': DEVICE
    }

def get_available_routers() -> Dict[str, Any]:
    """사용 가능한 라우터 목록 반환"""
    return AVAILABLE_ROUTERS.copy()

def get_router_info(router_name: str) -> Dict[str, Any]:
    """특정 라우터 정보 반환"""
    router = AVAILABLE_ROUTERS.get(router_name)
    
    return {
        'router_name': router_name,
        'available': router is not None,
        'loaded': ROUTER_STATUS.get(router_name, False),
        'router_object': router is not None
    }

# =============================================================================
# 🔥 Export 목록
# =============================================================================

__all__ = [
    # 🎯 핵심 함수들
    'register_routers',
    'setup_cors',
    'setup_middleware',
    
    # 📊 상태 관리 함수들
    'get_api_status',
    'get_available_routers',
    'get_router_info',
    
    # 🔧 라우터들 (조건부)
    'AVAILABLE_ROUTERS',
    'ROUTER_STATUS',
    
    # 📡 시스템 정보
    'SYSTEM_INFO',
    'IS_CONDA',
    'IS_M3_MAX',
    'DEVICE'
]

# 사용 가능한 라우터들을 동적으로 추가
for router_name in AVAILABLE_ROUTERS.keys():
    router_var_name = f"{router_name}_router"
    if router_var_name in globals():
        __all__.append(router_var_name)

# =============================================================================
# 🔥 초기화 완료 메시지
# =============================================================================

def _print_initialization_summary():
    """초기화 요약 출력"""
    available_count = len(AVAILABLE_ROUTERS)
    total_count = len(ROUTER_STATUS)
    success_rate = (available_count / total_count) * 100 if total_count > 0 else 0
    
    print(f"\n🍎 MyCloset AI API 시스템 v7.0 초기화 완료!")
    print(f"📡 사용 가능한 라우터: {available_count}/{total_count}개 ({success_rate:.1f}%)")
    print(f"🐍 conda 환경: {'✅' if IS_CONDA else '❌'}")
    print(f"🍎 M3 Max: {'✅' if IS_M3_MAX else '❌'}")
    print(f"🖥️ 디바이스: {DEVICE}")
    
    if AVAILABLE_ROUTERS:
        print(f"✅ 로드된 라우터: {', '.join(AVAILABLE_ROUTERS.keys())}")
    
    unavailable_routers = [k for k, v in ROUTER_STATUS.items() if not v]
    if unavailable_routers:
        print(f"⚠️ 구현 대기 라우터: {', '.join(unavailable_routers)}")
        print(f"💡 이는 정상적인 상태입니다 (단계적 구현)")
    
    print("🚀 API 시스템 준비 완료!\n")

# 초기화 상태 출력 (한 번만)
if not hasattr(sys, '_mycloset_api_initialized'):
    _print_initialization_summary()
    sys._mycloset_api_initialized = True

logger.info("🍎 MyCloset AI API 시스템 초기화 완료")

# 시간 import (미들웨어에서 사용)
import time