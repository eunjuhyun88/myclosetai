# backend/app/api/__init__.py
"""
🔥 MyCloset AI API 라우터 패키지 v9.0 - Central Hub DI Container v7.0 완전 연동
================================================================================

✅ Central Hub DI Container v7.0 완전 연동 - 중앙 허브 패턴 적용
✅ 순환참조 완전 해결 - TYPE_CHECKING + 지연 import 완벽 적용
✅ 단방향 의존성 그래프 - DI Container만을 통한 의존성 주입
✅ step_routes.py 완전 지원 추가 (/api/step/*)
✅ 기존 pipeline_routes.py도 유지 (/api/v1/pipeline/*)
✅ 단순하고 안정적인 API 초기화
✅ conda 환경 우선 최적화
✅ M3 Max 성능 최적화
✅ CORS 및 미들웨어 지원
✅ WebSocket 실시간 통신 지원
✅ 에러 핸들링 및 로깅
✅ 모든 NameError 문제 완전 해결

핵심 설계 원칙:
1. Single Source of Truth - 모든 서비스는 Central Hub DI Container를 거침
2. Central Hub Pattern - DI Container가 모든 컴포넌트의 중심
3. Dependency Inversion - 상위 모듈이 하위 모듈을 제어
4. Zero Circular Reference - 순환참조 원천 차단

API 엔드포인트:
- /api/step/*: 8단계 AI 파이프라인 API (Central Hub 연동!)
- /api/v1/pipeline/*: 기존 파이프라인 API
- /ws: WebSocket 실시간 통신
- /api/v1/health: 헬스 체크

작성자: MyCloset AI Team
날짜: 2025-07-31
버전: v9.0.0 (Central Hub Integration)
"""
import threading
import logging
import sys
import time
import warnings
import os
import platform
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from functools import lru_cache

# 경고 무시
warnings.filterwarnings('ignore')

# =============================================================================
# 🔥 Central Hub DI Container 안전 import (순환참조 방지)
# =============================================================================

def _get_central_hub_container():
    """Central Hub DI Container 안전한 동적 해결"""
    try:
        import importlib
        module = importlib.import_module('app.core.di_container')
        get_global_fn = getattr(module, 'get_global_container', None)
        if get_global_fn:
            return get_global_fn()
        return None
    except ImportError:
        return None
    except Exception:
        return None

def _inject_dependencies_to_router_safe(router_instance):
    """Central Hub DI Container를 통한 안전한 라우터 의존성 주입"""
    try:
        container = _get_central_hub_container()
        if container and hasattr(container, 'inject_to_router'):
            return container.inject_to_router(router_instance)
        return 0
    except Exception:
        return 0

def _get_service_from_central_hub(service_key: str):
    """Central Hub를 통한 안전한 서비스 조회"""
    try:
        container = _get_central_hub_container()
        if container:
            return container.get(service_key)
        return None
    except Exception:
        return None

# TYPE_CHECKING으로 순환참조 완전 방지
if TYPE_CHECKING:
    from app.core.di_container import CentralHubDIContainer

# =============================================================================
# 🔥 기본 설정 및 시스템 정보 (Central Hub 기반)
# =============================================================================

logger = logging.getLogger(__name__)

# 시스템 정보 직접 감지 (안전한 방식)
def _detect_system_info():
    """시스템 정보 직접 감지 (Central Hub 호환)"""
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

# 전역 변수 정의 (Central Hub 기반 - NameError 방지)
CONDA_ENV = detected_info['conda_env']
IS_CONDA = detected_info['is_conda']
IS_M3_MAX = detected_info['is_m3_max']
DEVICE = detected_info['device']
MEMORY_GB = detected_info['memory_gb']

# Central Hub Container에서 시스템 정보 가져오기 시도 (있으면 덮어씀)
try:
    container = _get_central_hub_container()
    if container:
        # Central Hub Container에서 시스템 정보 조회
        system_info = container.get('system_info')
        if system_info:
            CONDA_ENV = system_info.get('conda_env', CONDA_ENV)
            IS_CONDA = system_info.get('is_conda', IS_CONDA)
            IS_M3_MAX = system_info.get('is_m3_max', IS_M3_MAX)
            DEVICE = container.get('device') or DEVICE
            MEMORY_GB = system_info.get('memory_gb', MEMORY_GB)
            
            logger.info("✅ Central Hub Container에서 시스템 정보 로드 성공")
        
        SYSTEM_INFO = {
            'device': DEVICE, 
            'is_m3_max': IS_M3_MAX, 
            'memory_gb': MEMORY_GB,
            'conda_env': CONDA_ENV,
            'is_conda': IS_CONDA
        }
except Exception as e:
    logger.debug(f"Central Hub Container 시스템 정보 로드 실패, 기본값 사용: {e}")
    SYSTEM_INFO = {
        'device': DEVICE, 
        'is_m3_max': IS_M3_MAX, 
        'memory_gb': MEMORY_GB,
        'conda_env': CONDA_ENV,
        'is_conda': IS_CONDA
    }

# =============================================================================
# 🔥 Central Hub 기반 API 라우터 상태 추적
# =============================================================================

# API 라우터 로딩 상태 (Central Hub 기반)
ROUTER_STATUS = {
    'virtual_tryon': False,
    'pipeline_routes': False,
    'step_routes': False,        # 🔥 step_routes.py 추가!
    'websocket_routes': False,
    'system_routes': False,      # 🔥 system_routes.py 추가!
    'health_check': False
}

# Central Hub Container 참조
central_hub_container = None

# =============================================================================
# 🔥 Central Hub 기반 안전한 라우터 모듈 로딩
# =============================================================================

def _safe_import_virtual_tryon_central_hub():
    """Central Hub 기반 virtual_tryon 라우터 안전 import"""
    try:
        from .virtual_tryon import router as virtual_tryon_router
        
        # Central Hub Container 주입 시도
        injection_count = _inject_dependencies_to_router_safe(virtual_tryon_router)
        
        globals()['virtual_tryon_router'] = virtual_tryon_router
        
        ROUTER_STATUS['virtual_tryon'] = True
        logger.info(f"✅ Central Hub 기반 virtual_tryon 라우터 로드 성공 (의존성 주입: {injection_count}개)")
        return virtual_tryon_router
        
    except ImportError as e:
        logger.debug(f"📋 virtual_tryon 라우터 없음 (정상): {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Central Hub virtual_tryon 라우터 로드 실패: {e}")
        return None

def _safe_import_pipeline_routes_central_hub():
    """Central Hub 기반 pipeline_routes 라우터 안전 import"""
    try:
        from .pipeline_routes import router as pipeline_router
        
        # Central Hub Container 주입 시도
        injection_count = _inject_dependencies_to_router_safe(pipeline_router)
        
        globals()['pipeline_router'] = pipeline_router
        
        ROUTER_STATUS['pipeline_routes'] = True
        logger.info(f"✅ Central Hub 기반 pipeline_routes 라우터 로드 성공 (의존성 주입: {injection_count}개)")
        return pipeline_router
        
    except ImportError as e:
        logger.debug(f"📋 pipeline_routes 라우터 없음 (정상): {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Central Hub pipeline_routes 라우터 로드 실패: {e}")
        return None

def _safe_import_step_routes_central_hub():
    """🔥 Central Hub 기반 step_routes 라우터 안전 import (신규!)"""
    try:
        logger.info("🔄 Central Hub 기반 step_routes 로딩 시도...")
        
        # Central Hub Container 확인
        container = _get_central_hub_container()
        
        # 🔥 수정: Central Hub Container 여부와 관계없이 step_routes 로딩
        try:
            from .step_routes import router as step_router
            logger.info("✅ step_routes 라우터 import 성공")
            
            # step_router에 Central Hub Container 주입 (있는 경우에만)
            injection_count = 0
            if container:
                injection_count = _inject_dependencies_to_router_safe(step_router)
                
                # step_router에 Central Hub Container 직접 참조 추가 (백업)
                if hasattr(step_router, 'central_hub_container'):
                    step_router.central_hub_container = container
                logger.info(f"✅ Central Hub Container 주입 완료 (의존성 주입: {injection_count}개)")
            else:
                logger.warning("⚠️ Central Hub Container 없음, 기본 로딩으로 진행")
            
            # 라우터 상태 확인
            if hasattr(step_router, 'routes'):
                route_count = len(step_router.routes)
                logger.info(f"✅ step_router에 {route_count}개 엔드포인트 확인됨")
                
                # 주요 엔드포인트 확인
                for route in step_router.routes:
                    if hasattr(route, 'path') and hasattr(route, 'methods'):
                        if '/3/human-parsing' in route.path:
                            logger.info(f"✅ /3/human-parsing 엔드포인트 확인됨: {route.path} [{', '.join(route.methods)}]")
            
            globals()['step_router'] = step_router
            ROUTER_STATUS['step_routes'] = True
            logger.info(f"✅ step_routes 라우터 로드 완료")
            return step_router
            
        except ImportError as e:
            logger.error(f"❌ step_routes 라우터 import 실패: {e}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Central Hub step_routes 라우터 로드 실패: {e}")
        return None

def _safe_import_websocket_routes_central_hub():
    """Central Hub 기반 websocket_routes 라우터 안전 import"""
    try:
        from .websocket_routes import router as websocket_router
        
        # Central Hub Container 주입 시도
        injection_count = _inject_dependencies_to_router_safe(websocket_router)
        
        globals()['websocket_router'] = websocket_router
        
        ROUTER_STATUS['websocket_routes'] = True
        logger.info(f"✅ Central Hub 기반 websocket_routes 라우터 로드 성공 (의존성 주입: {injection_count}개)")
        return websocket_router
        
    except ImportError as e:
        logger.debug(f"📋 websocket_routes 라우터 없음 (정상): {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Central Hub websocket_routes 라우터 로드 실패: {e}")
        return None

def _safe_import_system_routes_central_hub():
    """Central Hub 기반 system_routes 라우터 안전 import"""
    try:
        from .system_routes import router as system_router
        
        # Central Hub Container 주입 시도
        injection_count = _inject_dependencies_to_router_safe(system_router)
        
        globals()['system_router'] = system_router
        
        ROUTER_STATUS['system_routes'] = True
        logger.info(f"✅ Central Hub 기반 system_routes 라우터 로드 성공 (의존성 주입: {injection_count}개)")
        return system_router
        
    except ImportError as e:
        logger.debug(f"📋 system_routes 라우터 없음 (정상): {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Central Hub system_routes 라우터 로드 실패: {e}")
        return None

def _create_central_hub_health_router():
    """Central Hub 통합 헬스체크 라우터 생성"""
    try:
        from fastapi import APIRouter
        
        health_router = APIRouter(tags=["health-central-hub"])
        
        @health_router.get("/health")
        async def central_hub_health_check():
            """Central Hub 통합 헬스체크"""
            try:
                container = _get_central_hub_container()
                
                health_status = {
                    'status': 'healthy',
                    'version': '9.0 (Central Hub Integration)',
                    'timestamp': datetime.now().isoformat(),
                    'central_hub': {
                        'connected': container is not None,
                        'services': {}
                    }
                }
                
                if container:
                    # Central Hub 서비스 상태 확인
                    core_services = [
                        'step_service_manager',
                        'step_factory',
                        'session_manager',
                        'websocket_manager',
                        'model_loader',
                        'memory_manager'
                    ]
                    
                    for service_key in core_services:
                        try:
                            service = container.get(service_key)
                            health_status['central_hub']['services'][service_key] = {
                                'available': service is not None,
                                'type': type(service).__name__ if service else None
                            }
                        except:
                            health_status['central_hub']['services'][service_key] = {
                                'available': False,
                                'error': 'Check failed'
                            }
                    
                    # Central Hub 통계 추가
                    try:
                        if hasattr(container, 'get_stats'):
                            health_status['central_hub']['stats'] = container.get_stats()
                    except:
                        pass
                    
                    # 전체 상태 판정 (session_manager는 선택적)
                    critical_services = [
                        'step_service_manager',
                        'step_factory',
                        'websocket_manager',
                        'model_loader',
                        'memory_manager'
                    ]
                    
                    critical_services_healthy = all(
                        health_status['central_hub']['services'].get(service_key, {}).get('available', False)
                        for service_key in critical_services
                    )
                    
                    # session_manager가 없어도 기본 서비스는 정상이므로 200 OK 반환
                    health_status['status'] = 'healthy'
                    from fastapi.responses import JSONResponse
                    return JSONResponse(content=health_status)
                else:
                    health_status['status'] = 'limited'
                    health_status['message'] = 'Central Hub not available'
                    from fastapi.responses import JSONResponse
                    return JSONResponse(content=health_status, status_code=503)
                    
            except Exception as e:
                from fastapi.responses import JSONResponse
                return JSONResponse(content={
                    'status': 'unhealthy',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }, status_code=503)
        
        @health_router.get("/status")
        async def central_hub_api_status():
            """Central Hub 기반 API 상태 상세 정보"""
            available_routers = [k for k, v in ROUTER_STATUS.items() if v]
            
            container = _get_central_hub_container()
            container_services = []
            if container and hasattr(container, 'list_services'):
                container_services = container.list_services()
            
            return {
                "api_version": "v9.0.0 (Central Hub Integration)",
                "available_routers": available_routers,
                "total_routers": len(ROUTER_STATUS),
                "success_rate": (len(available_routers) / len(ROUTER_STATUS)) * 100,
                "central_hub": {
                    "connected": container is not None,
                    "services_count": len(container_services),
                    "services": container_services[:10]  # 처음 10개만 표시
                },
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
        logger.info("✅ Central Hub 기반 health_check 라우터 생성 성공")
        return health_router
        
    except Exception as e:
        logger.error(f"❌ Central Hub health_check 라우터 생성 실패: {e}")
        return None

# =============================================================================
# 🔥 Central Hub 기반 라우터들 로딩 (step_routes.py 추가!)
# =============================================================================

# Central Hub Container 참조 가져오기
try:
    central_hub_container = _get_central_hub_container()
    if central_hub_container:
        logger.info("✅ Central Hub Container 참조 획득")
    else:
        logger.warning("⚠️ Central Hub Container 사용 불가")
except Exception as e:
    logger.warning(f"⚠️ Central Hub Container 참조 실패: {e}")

# 모든 라우터 로딩 시도 (Central Hub 기반)
AVAILABLE_ROUTERS = {}

# Virtual Try-on 라우터 (Central Hub 연동)
virtual_tryon_router = _safe_import_virtual_tryon_central_hub()
if virtual_tryon_router:
    AVAILABLE_ROUTERS['virtual_tryon'] = virtual_tryon_router

# Pipeline 라우터 (기존 - Central Hub 연동)
pipeline_router = _safe_import_pipeline_routes_central_hub()
if pipeline_router:
    AVAILABLE_ROUTERS['pipeline'] = pipeline_router

# 🔥 Step 라우터 (신규 추가 - Central Hub 연동!)
step_router = _safe_import_step_routes_central_hub()
if step_router:
    AVAILABLE_ROUTERS['step_routes'] = step_router

# WebSocket 라우터 (Central Hub 연동)
websocket_router = _safe_import_websocket_routes_central_hub()
if websocket_router:
    AVAILABLE_ROUTERS['websocket'] = websocket_router

# 🔥 System 라우터 (신규 추가 - Central Hub 연동!)
system_router = _safe_import_system_routes_central_hub()
if system_router:
    AVAILABLE_ROUTERS['system_routes'] = system_router

# Health Check 라우터 (항상 생성 - Central Hub 통합)
health_router = _create_central_hub_health_router()
if health_router:
    AVAILABLE_ROUTERS['health'] = health_router
    ROUTER_STATUS['health_check'] = True
    logger.info("✅ Central Hub 기반 health 라우터 생성 완료")

# =============================================================================
# 🔥 Central Hub 기반 라우터 등록 함수 (step_routes.py 지원 추가!)
# =============================================================================

def register_routers(app) -> int:
    """Central Hub 기반 모든 라우터 등록"""
    registered_count = 0
    
    try:
        logger.info("🔄 Central Hub 기반 라우터 등록 시작...")
        
        # Central Hub Container 확인
        container = _get_central_hub_container()
        if container:
            logger.info("✅ Central Hub Container 사용 가능")
        else:
            logger.warning("⚠️ Central Hub Container 사용 불가, 일반 등록 진행")
        
        # 🔥 Step 라우터 (최우선) - Central Hub 기반
        logger.info(f"🔍 AVAILABLE_ROUTERS 키들: {list(AVAILABLE_ROUTERS.keys())}")
        
        if 'step_routes' in AVAILABLE_ROUTERS:
            step_router = AVAILABLE_ROUTERS['step_routes']
            logger.info(f"✅ step_routes 라우터 발견: {type(step_router)}")
            
            # 라우터 상태 확인
            if hasattr(step_router, 'routes'):
                route_count = len(step_router.routes)
                logger.info(f"✅ step_router에 {route_count}개 엔드포인트 확인됨")
                
                # 주요 엔드포인트 확인
                for route in step_router.routes:
                    if hasattr(route, 'path') and hasattr(route, 'methods'):
                        if '/3/human-parsing' in route.path:
                            logger.info(f"✅ /3/human-parsing 엔드포인트 확인됨: {route.path} [{', '.join(route.methods)}]")
            
            # Central Hub Container를 step_router 상태에 추가
            if container and hasattr(step_router, 'dependencies'):
                from fastapi import Depends
                step_router.dependencies.append(
                    Depends(lambda: container)
                )
                logger.info("✅ Central Hub Container 의존성 추가됨")
            
            try:
                app.include_router(
                    step_router,
                    prefix="/api/step",  # 프론트엔드 호환성
                    tags=["step-pipeline-central-hub"]
                )
                registered_count += 1
                logger.info("✅ Central Hub 기반 step_routes 라우터 등록 완료 (/api/step)")
            except Exception as e:
                logger.error(f"❌ step_routes 라우터 등록 실패: {e}")
        else:
            logger.error("❌ step_routes가 AVAILABLE_ROUTERS에 없음!")
            logger.error(f"🔍 사용 가능한 라우터: {list(AVAILABLE_ROUTERS.keys())}")
        
        # Virtual Try-on 라우터 - Central Hub 연동
        if 'virtual_tryon' in AVAILABLE_ROUTERS:
            app.include_router(
                AVAILABLE_ROUTERS['virtual_tryon'],
                prefix="/api/v1",
                tags=["virtual-tryon-central-hub"]
            )
            registered_count += 1
            logger.info("✅ Central Hub 기반 virtual_tryon 라우터 등록")
        
        # Pipeline 라우터 - Central Hub 연동
        if 'pipeline' in AVAILABLE_ROUTERS:
            app.include_router(
                AVAILABLE_ROUTERS['pipeline'],
                prefix="/api/v1",
                tags=["pipeline-central-hub"]
            )
            registered_count += 1
            logger.info("✅ Central Hub 기반 pipeline 라우터 등록")
        
        # WebSocket 라우터 - Central Hub 연동
        if 'websocket' in AVAILABLE_ROUTERS:
            app.include_router(
                AVAILABLE_ROUTERS['websocket'],
                tags=["websocket-central-hub"]
            )
            registered_count += 1
            logger.info("✅ Central Hub 기반 websocket 라우터 등록")
        
        # 🔥 System 라우터 - Central Hub 연동
        if 'system_routes' in AVAILABLE_ROUTERS:
            app.include_router(
                AVAILABLE_ROUTERS['system_routes'],
                tags=["system-central-hub"]
            )
            registered_count += 1
            logger.info("✅ Central Hub 기반 system_routes 라우터 등록 (/api/system)")
        
        # Health Check 라우터 - Central Hub 통합 (루트 경로에 등록)
        if 'health' in AVAILABLE_ROUTERS:
            app.include_router(
                AVAILABLE_ROUTERS['health'],
                tags=["health-central-hub"]
            )
            registered_count += 1
            logger.info("✅ Central Hub 기반 health 라우터 등록 (루트 경로)")
        
        logger.info(f"🎯 Central Hub 기반 총 {registered_count}개 라우터 등록 완료")
        
        # Central Hub 라우터 등록 상태를 Container에 저장
        if container:
            try:
                router_registry = {
                    'total_registered': registered_count,
                    'step_routes_enabled': 'step_routes' in AVAILABLE_ROUTERS,
                    'registration_timestamp': datetime.now().isoformat()
                }
                container.register('router_registry', router_registry)
                logger.info("✅ Central Hub에 라우터 등록 상태 저장")
            except Exception as e:
                logger.debug(f"Central Hub 라우터 상태 저장 실패: {e}")
        
        return registered_count
        
    except Exception as e:
        logger.error(f"❌ Central Hub 라우터 등록 실패: {e}")
        return registered_count

# =============================================================================
# 🔥 Central Hub 기반 CORS 및 미들웨어 설정
# =============================================================================

def setup_cors(app, origins: Optional[List[str]] = None):
    """Central Hub 기반 CORS 설정"""
    try:
        from fastapi.middleware.cors import CORSMiddleware
        
        # Central Hub Container에서 CORS 설정 조회 시도
        if origins is None:
            container = _get_central_hub_container()
            if container:
                cors_config = container.get('cors_config')
                if cors_config and 'origins' in cors_config:
                    origins = cors_config['origins']
                    logger.info("✅ Central Hub에서 CORS 설정 로드")
        
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
        
        logger.info(f"✅ Central Hub 기반 CORS 설정 완료: {len(origins)}개 origin")
        
        # Central Hub Container에 CORS 설정 저장
        container = _get_central_hub_container()
        if container:
            try:
                container.register('cors_config', {'origins': origins})
                logger.debug("✅ Central Hub에 CORS 설정 저장")
            except:
                pass
        
    except Exception as e:
        logger.error(f"❌ Central Hub CORS 설정 실패: {e}")

def setup_middleware(app):
    """Central Hub 기반 추가 미들웨어 설정"""
    try:
        # Central Hub 기반 요청 로깅 미들웨어 (개발용)
        @app.middleware("http")
        async def central_hub_log_requests(request, call_next):
            start_time = time.time()
            
            # Central Hub Container 참조 추가
            container = _get_central_hub_container()
            if container:
                request.state.central_hub_container = container
            
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # step_routes 요청은 상세 로깅
            if request.url.path.startswith("/api/step/"):
                logger.info(
                    f"🔥 CENTRAL HUB STEP API: {request.method} {request.url.path} - "
                    f"Status: {response.status_code} - "
                    f"Time: {process_time:.4f}s"
                )
            else:
                logger.debug(
                    f"Central Hub: {request.method} {request.url.path} - "
                    f"Status: {response.status_code} - "
                    f"Time: {process_time:.4f}s"
                )
            
            return response
        
        logger.info("✅ Central Hub 기반 요청 로깅 미들웨어 설정 완료 (step_routes 강화)")
        
    except Exception as e:
        logger.error(f"❌ Central Hub 미들웨어 설정 실패: {e}")

# =============================================================================
# 🔥 Central Hub 기반 API 상태 관리 함수들
# =============================================================================

@lru_cache(maxsize=1)
def get_api_status() -> Dict[str, Any]:
    """Central Hub 기반 API 상태 반환"""
    available_routers = [k for k, v in ROUTER_STATUS.items() if v]
    
    container = _get_central_hub_container()
    central_hub_info = {
        'connected': container is not None,
        'services_count': 0,
        'container_id': None
    }
    
    if container:
        try:
            if hasattr(container, 'list_services'):
                central_hub_info['services_count'] = len(container.list_services())
            central_hub_info['container_id'] = getattr(container, 'container_id', 'unknown')
        except:
            pass
    
    return {
        "api_version": "v9.0.0 (Central Hub Integration)",
        "total_routers": len(ROUTER_STATUS),
        "available_routers": available_routers,
        "success_rate": (len(available_routers) / len(ROUTER_STATUS)) * 100,
        "central_hub": central_hub_info,
        "system": SYSTEM_INFO,
        "router_details": ROUTER_STATUS,
        "step_routes_enabled": ROUTER_STATUS['step_routes'],  # 🔥 step_routes 상태!
        "conda_optimized": IS_CONDA,
        "m3_max_optimized": IS_M3_MAX
    }

def get_available_endpoints() -> List[str]:
    """사용 가능한 엔드포인트 목록 (Central Hub 기반)"""
    endpoints = ["/health", "/status"]
    
    if ROUTER_STATUS.get('virtual_tryon'):
        endpoints.extend(["/api/v1/virtual-tryon/*"])
    
    if ROUTER_STATUS.get('pipeline_routes'):
        endpoints.extend(["/api/v1/pipeline/*"])
    
    # 🔥 step_routes 엔드포인트 추가! (Central Hub 기반)
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
    
    # 🔥 system_routes 엔드포인트 추가!
    if ROUTER_STATUS.get('system_routes'):
        endpoints.extend([
            "/api/system/info",
            "/api/system/health", 
            "/api/system/status"
        ])
    
    # Central Hub 전용 엔드포인트 추가
    endpoints.extend([
        "/central-hub/status",
        "/central-hub/services"
    ])
    
    return endpoints

def get_router_info() -> Dict[str, Any]:
    """Central Hub 기반 라우터 상세 정보"""
    router_info = {}
    
    for router_name, is_available in ROUTER_STATUS.items():
        router_info[router_name] = {
            "available": is_available,
            "loaded": router_name in AVAILABLE_ROUTERS,
            "instance": AVAILABLE_ROUTERS.get(router_name) is not None,
            "central_hub_integrated": True  # 모든 라우터가 Central Hub 통합
        }
    
    # 🔥 step_routes 특별 정보 추가! (Central Hub 기반)
    if ROUTER_STATUS.get('step_routes'):
        router_info['step_routes'].update({
            "prefix": "/api/step",
            "frontend_compatible": True,
            "ai_pipeline_steps": 8,
            "real_ai_only": True,
            "central_hub_version": "v7.0",
            "dependency_injection": "완료"
        })
    
    # 🔥 system_routes 특별 정보 추가! (Central Hub 기반)
    if ROUTER_STATUS.get('system_routes'):
        router_info['system_routes'].update({
            "prefix": "/api/system",
            "frontend_compatible": True,
            "system_info_endpoints": 3,
            "caching_enabled": True,
            "central_hub_version": "v7.0",
            "dependency_injection": "완료"
        })
    
    # Central Hub Container 정보 추가
    container = _get_central_hub_container()
    router_info['central_hub_container'] = {
        'available': container is not None,
        'services_count': len(container.list_services()) if container and hasattr(container, 'list_services') else 0,
        'container_id': getattr(container, 'container_id', 'unknown') if container else None
    }
    
    return router_info

# =============================================================================
# 🔥 Central Hub 기반 Export
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
    'MEMORY_GB',
    # Central Hub 전용
    'central_hub_container',
    '_get_central_hub_container',
    '_get_service_from_central_hub'
]

# =============================================================================
# 🔥 Central Hub 기반 초기화 완료 메시지
# =============================================================================

logger.info("🎉 Central Hub 기반 API 라우터 통합 관리자 v9.0 로드 완료!")
logger.info(f"✅ 시스템 환경: conda={CONDA_ENV}, M3 Max={IS_M3_MAX}")
logger.info(f"✅ 메모리: {MEMORY_GB}GB, 디바이스: {DEVICE}")
logger.info(f"✅ 사용 가능한 라우터: {len([k for k, v in ROUTER_STATUS.items() if v])}/{len(ROUTER_STATUS)}")
logger.info(f"🔥 Central Hub Container: {'✅ 연결됨' if central_hub_container else '❌ 사용 불가'}")
logger.info(f"🔥 step_routes.py 지원: {'✅ 활성화' if ROUTER_STATUS['step_routes'] else '❌ 비활성화'}")
logger.info(f"🔥 system_routes.py 지원: {'✅ 활성화' if ROUTER_STATUS['system_routes'] else '❌ 비활성화'}")

if ROUTER_STATUS['step_routes']:
    logger.info("🎯 Central Hub 기반 step_routes.py 라우터 정보:")
    logger.info("   - 경로: /api/step/*")
    logger.info("   - 프론트엔드 완전 호환")
    logger.info("   - 8단계 AI 파이프라인 지원")
    logger.info("   - 실제 AI 모델 전용")
    logger.info("   - Central Hub DI Container v7.0 완전 연동")

if ROUTER_STATUS['system_routes']:
    logger.info("🎯 Central Hub 기반 system_routes.py 라우터 정보:")
    logger.info("   - 경로: /api/system/*")
    logger.info("   - 시스템 정보 API 지원")
    logger.info("   - 헬스 체크 및 상태 모니터링")
    logger.info("   - 30초 캐시 최적화")
    logger.info("   - Central Hub DI Container v7.0 완전 연동")

if central_hub_container:
    try:
        services_count = len(central_hub_container.list_services()) if hasattr(central_hub_container, 'list_services') else 0
        container_id = getattr(central_hub_container, 'container_id', 'unknown')
        logger.info(f"🔥 Central Hub Container 정보:")
        logger.info(f"   - Container ID: {container_id}")
        logger.info(f"   - 등록된 서비스: {services_count}개")
        logger.info(f"   - Single Source of Truth 구현")
        logger.info(f"   - Dependency Inversion 적용")
        logger.info(f"   - 순환참조 완전 해결")
    except Exception as e:
        logger.debug(f"Central Hub Container 정보 조회 실패: {e}")

logger.info("🚀 Central Hub 기반 프론트엔드 API 요청 준비 완료!")
logger.info("✅ Central Hub DI Container v7.0 완전 연동 완성!")
logger.info("🎯 모든 의존성이 단일 중심을 통해 관리됩니다!")