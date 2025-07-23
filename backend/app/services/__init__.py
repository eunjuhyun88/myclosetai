
# ============================================================================
# 📁 backend/app/services/__init__.py - 서비스 레이어 관리
# ============================================================================

"""
🔧 MyCloset AI Services 모듈 - conda 환경 우선 서비스 레이어 관리
============================================================

✅ conda 환경 우선 최적화
✅ 순환참조 완전 방지 (지연 로딩 패턴)
✅ 비즈니스 로직 서비스들 안전한 로딩
✅ AI 파이프라인 서비스 통합
✅ 세션 기반 처리 서비스
✅ main.py 완벽 호환성

역할: 비즈니스 로직과 AI 처리 서비스들의 로딩과 관리를 담당
"""

import logging
import threading
from typing import Dict, Any, Optional, Type

# 상위 패키지에서 시스템 정보 가져오기
try:
    from .. import SYSTEM_INFO, IS_CONDA, CONDA_ENV, _lazy_loader
except ImportError:
    SYSTEM_INFO = {'device': 'cpu', 'is_m3_max': False}
    IS_CONDA = 'CONDA_DEFAULT_ENV' in os.environ
    CONDA_ENV = os.environ.get('CONDA_DEFAULT_ENV', 'none')
    _lazy_loader = None

# 로거 설정
logger = logging.getLogger(__name__)

# =============================================================================
# 🔥 서비스 모듈 정보
# =============================================================================

SERVICE_MODULES = {
    'ai_pipeline': 'ai_pipeline',
    'pipeline_service': 'pipeline_service',
    'session_service': 'session_service',
    'step_implementations': 'step_implementations',
    'step_utils': 'step_utils',
    'websocket_service': 'websocket_service'
}

SERVICE_CLASSES = {
    'ai_pipeline': 'AIPipelineService',
    'pipeline_service': 'PipelineService',
    'session_service': 'SessionService', 
    'step_implementations': 'StepImplementationService',
    'step_utils': 'StepUtilsService',
    'websocket_service': 'WebSocketService'
}

# =============================================================================
# 🔥 지연 로딩 함수들 (순환참조 방지)
# =============================================================================

def get_ai_pipeline_service_class():
    """AIPipelineService 클래스 지연 로딩"""
    if _lazy_loader:
        return _lazy_loader.get_class('ai_pipeline', 'AIPipelineService', 'app.services')
    
    try:
        from .ai_pipeline import AIPipelineService
        return AIPipelineService
    except ImportError as e:
        logger.warning(f"AIPipelineService 클래스 로딩 실패: {e}")
        return None

def get_pipeline_service_class():
    """PipelineService 클래스 지연 로딩"""
    if _lazy_loader:
        return _lazy_loader.get_class('pipeline_service', 'PipelineService', 'app.services')
    
    try:
        from .pipeline_service import PipelineService
        return PipelineService
    except ImportError as e:
        logger.warning(f"PipelineService 클래스 로딩 실패: {e}")
        return None

def get_session_service_class():
    """SessionService 클래스 지연 로딩"""
    if _lazy_loader:
        return _lazy_loader.get_class('session_service', 'SessionService', 'app.services')
    
    try:
        from .session_service import SessionService
        return SessionService
    except ImportError as e:
        logger.warning(f"SessionService 클래스 로딩 실패: {e}")
        return None

def get_websocket_service_class():
    """WebSocketService 클래스 지연 로딩"""
    if _lazy_loader:
        return _lazy_loader.get_class('websocket_service', 'WebSocketService', 'app.services')
    
    try:
        from .websocket_service import WebSocketService
        return WebSocketService
    except ImportError as e:
        logger.warning(f"WebSocketService 클래스 로딩 실패: {e}")
        return None

# =============================================================================
# 🔥 팩토리 함수들 (conda 환경 최적화)
# =============================================================================

def create_ai_pipeline_service(**kwargs) -> Optional[Any]:
    """AIPipelineService 인스턴스 생성 (conda 환경 최적화)"""
    AIPipelineService = get_ai_pipeline_service_class()
    if AIPipelineService:
        # conda 환경 설정 추가
        service_config = {
            'device': SYSTEM_INFO.get('device', 'cpu'),
            'is_m3_max': SYSTEM_INFO.get('is_m3_max', False),
            'memory_gb': SYSTEM_INFO.get('memory_gb', 16.0),
            'conda_optimized': IS_CONDA,
            'conda_env': CONDA_ENV
        }
        service_config.update(kwargs)
        
        try:
            return AIPipelineService(**service_config)
        except Exception as e:
            logger.error(f"AIPipelineService 생성 실패: {e}")
            return None
    return None

def create_pipeline_service(**kwargs) -> Optional[Any]:
    """PipelineService 인스턴스 생성 (conda 환경 최적화)"""
    PipelineService = get_pipeline_service_class()
    if PipelineService:
        service_config = {
            'device': SYSTEM_INFO.get('device', 'cpu'),
            'conda_optimized': IS_CONDA
        }
        service_config.update(kwargs)
        
        try:
            return PipelineService(**service_config)
        except Exception as e:
            logger.error(f"PipelineService 생성 실패: {e}")
            return None
    return None

def create_session_service(**kwargs) -> Optional[Any]:
    """SessionService 인스턴스 생성"""
    SessionService = get_session_service_class()
    if SessionService:
        try:
            return SessionService(**kwargs)
        except Exception as e:
            logger.error(f"SessionService 생성 실패: {e}")
            return None
    return None

def create_websocket_service(**kwargs) -> Optional[Any]:
    """WebSocketService 인스턴스 생성"""
    WebSocketService = get_websocket_service_class()
    if WebSocketService:
        try:
            return WebSocketService(**kwargs)
        except Exception as e:
            logger.error(f"WebSocketService 생성 실패: {e}")
            return None
    return None

# =============================================================================
# 🔥 전역 서비스 인스턴스 관리 (싱글톤 패턴)
# =============================================================================

_global_services = {}
_service_lock = threading.RLock()

def get_global_ai_pipeline_service():
    """전역 AIPipelineService 인스턴스 반환"""
    with _service_lock:
        if 'ai_pipeline' not in _global_services:
            _global_services['ai_pipeline'] = create_ai_pipeline_service()
        return _global_services['ai_pipeline']

def get_global_pipeline_service():
    """전역 PipelineService 인스턴스 반환"""
    with _service_lock:
        if 'pipeline' not in _global_services:
            _global_services['pipeline'] = create_pipeline_service()
        return _global_services['pipeline']

def get_global_session_service():
    """전역 SessionService 인스턴스 반환"""
    with _service_lock:
        if 'session' not in _global_services:
            _global_services['session'] = create_session_service()
        return _global_services['session']

def get_global_websocket_service():
    """전역 WebSocketService 인스턴스 반환"""
    with _service_lock:
        if 'websocket' not in _global_services:
            _global_services['websocket'] = create_websocket_service()
        return _global_services['websocket']

# =============================================================================
# 🔥 서비스 상태 관리
# =============================================================================

def get_services_status() -> Dict[str, Any]:
    """서비스 상태 반환"""
    return {
        'conda_environment': IS_CONDA,
        'conda_env_name': CONDA_ENV,
        'device': SYSTEM_INFO.get('device', 'cpu'),
        'is_m3_max': SYSTEM_INFO.get('is_m3_max', False),
        'availability': {
            'ai_pipeline_service': get_ai_pipeline_service_class() is not None,
            'pipeline_service': get_pipeline_service_class() is not None,
            'session_service': get_session_service_class() is not None,
            'websocket_service': get_websocket_service_class() is not None,
        },
        'global_services': {
            'ai_pipeline': 'ai_pipeline' in _global_services,
            'pipeline': 'pipeline' in _global_services,
            'session': 'session' in _global_services,
            'websocket': 'websocket' in _global_services,
        },
        'total_global_services': len(_global_services)
    }

def cleanup_services():
    """서비스 정리"""
    try:
        logger.info("🧹 서비스 정리 시작...")
        
        with _service_lock:
            for name, service in _global_services.items():
                try:
                    if hasattr(service, 'cleanup'):
                        service.cleanup()
                    elif hasattr(service, 'close'):
                        service.close()
                except Exception as e:
                    logger.warning(f"서비스 정리 실패 {name}: {e}")
            
            _global_services.clear()
        
        logger.info("✅ 서비스 정리 완료")
        
    except Exception as e:
        logger.error(f"❌ 서비스 정리 실패: {e}")

# =============================================================================
# 🔥 Services 모듈 Export
# =============================================================================

__all__ = [
    # 🔗 지연 로딩 함수들
    'get_ai_pipeline_service_class',
    'get_pipeline_service_class',
    'get_session_service_class',
    'get_websocket_service_class',
    
    # 🏭 팩토리 함수들
    'create_ai_pipeline_service',
    'create_pipeline_service',
    'create_session_service',
    'create_websocket_service',
    
    # 🌍 전역 서비스 함수들
    'get_global_ai_pipeline_service',
    'get_global_pipeline_service',
    'get_global_session_service',
    'get_global_websocket_service',
    
    # 🔧 상태 관리 함수들
    'get_services_status',
    'cleanup_services',
]

# 초기화 정보 출력
logger.info("🔧 MyCloset AI Services 모듈 초기화 완료")
logger.info(f"🐍 conda 최적화: {IS_CONDA}")
logger.info(f"🍎 M3 Max: {SYSTEM_INFO.get('is_m3_max', False)}")
logger.info(f"🔗 지연 로딩: 활성화")

