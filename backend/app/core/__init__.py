# ============================================================================
# 📁 backend/app/core/__init__.py - 핵심 설정 모듈
# ============================================================================

"""
🔧 MyCloset AI Core 모듈 - conda 환경 우선 핵심 설정
========================================================

✅ conda 환경 우선 최적화
✅ 순환참조 완전 방지 (지연 로딩 패턴)
✅ M3 Max 128GB 메모리 설정
✅ AI 모델 경로 관리
✅ GPU/디바이스 설정 통합 관리
✅ 세션 관리 시스템
✅ 안전한 설정 로딩

역할: 애플리케이션의 핵심 설정과 시스템 초기화를 담당
"""

import os
import sys
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional

# 상위 패키지에서 시스템 정보 가져오기
try:
    from .. import SYSTEM_INFO, AI_MODEL_PATHS, IS_CONDA, CONDA_ENV, _lazy_loader
except ImportError:
    # 폴백: 직접 감지
    SYSTEM_INFO = {'device': 'cpu', 'is_m3_max': False, 'memory_gb': 16.0}
    AI_MODEL_PATHS = {'ai_models_root': Path(__file__).parent.parent.parent / 'ai_models'}
    IS_CONDA = 'CONDA_DEFAULT_ENV' in os.environ
    CONDA_ENV = os.environ.get('CONDA_DEFAULT_ENV', 'none')
    _lazy_loader = None

# 로거 설정
logger = logging.getLogger(__name__)

# =============================================================================
# 🔥 Core 모듈 경로 설정
# =============================================================================

BACKEND_DIR = Path(__file__).parent.parent.parent
AI_MODELS_DIR = BACKEND_DIR / "ai_models"
CORE_CONFIG_DIR = Path(__file__).parent

# AI 모델 환경 변수 설정
os.environ.setdefault("MYCLOSET_AI_MODELS_PATH", str(AI_MODELS_DIR))
os.environ.setdefault("MYCLOSET_CORE_CONFIG_PATH", str(CORE_CONFIG_DIR))

# conda 환경 특화 설정
if IS_CONDA:
    os.environ.setdefault("MYCLOSET_CONDA_OPTIMIZED", "true")
    os.environ.setdefault("MYCLOSET_PACKAGE_MANAGER", "conda")
else:
    os.environ.setdefault("MYCLOSET_CONDA_OPTIMIZED", "false")
    os.environ.setdefault("MYCLOSET_PACKAGE_MANAGER", "pip")

# =============================================================================
# 🔥 지연 로딩 함수들 (순환참조 방지)
# =============================================================================

def get_config_class():
    """Config 클래스 지연 로딩"""
    if _lazy_loader:
        return _lazy_loader.get_class('config', 'Config', 'app.core')
    
    try:
        from .config import Config
        return Config
    except ImportError as e:
        logger.warning(f"Config 클래스 로딩 실패: {e}")
        return None

def get_gpu_config_class():
    """GPUConfig 클래스 지연 로딩"""
    if _lazy_loader:
        return _lazy_loader.get_class('gpu_config', 'GPUConfig', 'app.core')
    
    try:
        from .gpu_config import GPUConfig
        return GPUConfig
    except ImportError as e:
        logger.warning(f"GPUConfig 클래스 로딩 실패: {e}")
        return None

def get_session_manager_class():
    """SessionManager 클래스 지연 로딩"""
    if _lazy_loader:
        return _lazy_loader.get_class('session_manager', 'SessionManager', 'app.core')
    
    try:
        from .session_manager import SessionManager
        return SessionManager
    except ImportError as e:
        logger.warning(f"SessionManager 클래스 로딩 실패: {e}")
        return None

def get_di_container_class():
    """DI Container 클래스 지연 로딩"""
    if _lazy_loader:
        return _lazy_loader.get_class('di_container', 'DIContainer', 'app.core')
    
    try:
        from .di_container import DIContainer
        return DIContainer
    except ImportError as e:
        logger.warning(f"DIContainer 클래스 로딩 실패: {e}")
        return None

# =============================================================================
# 🔥 팩토리 함수들 (conda 환경 최적화)
# =============================================================================

def create_config(**kwargs) -> Optional[Any]:
    """Config 인스턴스 생성 (conda 환경 최적화)"""
    Config = get_config_class()
    if Config:
        # conda 환경 설정 추가
        conda_config = {
            'is_conda': IS_CONDA,
            'conda_env': CONDA_ENV,
            'device': SYSTEM_INFO.get('device', 'cpu'),
            'is_m3_max': SYSTEM_INFO.get('is_m3_max', False),
            'memory_gb': SYSTEM_INFO.get('memory_gb', 16.0)
        }
        conda_config.update(kwargs)
        
        try:
            return Config(**conda_config)
        except Exception as e:
            logger.error(f"Config 생성 실패: {e}")
            return None
    return None

def create_gpu_config(**kwargs) -> Optional[Any]:
    """GPUConfig 인스턴스 생성 (conda 환경 최적화)"""
    GPUConfig = get_gpu_config_class()
    if GPUConfig:
        # conda 환경 GPU 설정 추가
        gpu_config = {
            'device': SYSTEM_INFO.get('device', 'cpu'),
            'is_m3_max': SYSTEM_INFO.get('is_m3_max', False),
            'memory_gb': SYSTEM_INFO.get('memory_gb', 16.0),
            'conda_optimized': IS_CONDA
        }
        gpu_config.update(kwargs)
        
        try:
            return GPUConfig(**gpu_config)
        except Exception as e:
            logger.error(f"GPUConfig 생성 실패: {e}")
            return None
    return None

def create_session_manager(**kwargs) -> Optional[Any]:
    """SessionManager 인스턴스 생성"""
    SessionManager = get_session_manager_class()
    if SessionManager:
        try:
            return SessionManager(**kwargs)
        except Exception as e:
            logger.error(f"SessionManager 생성 실패: {e}")
            return None
    return None

# =============================================================================
# 🔥 전역 인스턴스 관리 (싱글톤 패턴)
# =============================================================================

_global_instances = {}
_instance_lock = threading.RLock()

def get_global_config():
    """전역 Config 인스턴스 반환"""
    with _instance_lock:
        if 'config' not in _global_instances:
            _global_instances['config'] = create_config()
        return _global_instances['config']

def get_global_gpu_config():
    """전역 GPUConfig 인스턴스 반환"""
    with _instance_lock:
        if 'gpu_config' not in _global_instances:
            _global_instances['gpu_config'] = create_gpu_config()
        return _global_instances['gpu_config']

def get_global_session_manager():
    """전역 SessionManager 인스턴스 반환"""
    with _instance_lock:
        if 'session_manager' not in _global_instances:
            _global_instances['session_manager'] = create_session_manager()
        return _global_instances['session_manager']

# =============================================================================
# 🔥 편의 함수들
# =============================================================================

def get_ai_models_path() -> Path:
    """AI 모델 경로 반환"""
    return AI_MODELS_DIR

def get_system_device() -> str:
    """시스템 디바이스 반환"""
    return SYSTEM_INFO.get('device', 'cpu')

def is_conda_optimized() -> bool:
    """conda 최적화 여부 확인"""
    return IS_CONDA

def get_memory_gb() -> float:
    """시스템 메모리 용량 반환"""
    return SYSTEM_INFO.get('memory_gb', 16.0)

def get_core_status() -> Dict[str, Any]:
    """Core 모듈 상태 반환"""
    return {
        'ai_models_path': str(AI_MODELS_DIR),
        'ai_models_exist': AI_MODELS_DIR.exists(),
        'conda_environment': IS_CONDA,
        'conda_env_name': CONDA_ENV,
        'device': SYSTEM_INFO.get('device', 'cpu'),
        'is_m3_max': SYSTEM_INFO.get('is_m3_max', False),
        'memory_gb': SYSTEM_INFO.get('memory_gb', 16.0),
        'config_available': get_config_class() is not None,
        'gpu_config_available': get_gpu_config_class() is not None,
        'session_manager_available': get_session_manager_class() is not None,
        'global_instances_count': len(_global_instances)
    }

# =============================================================================
# 🔥 Core 모듈 Export
# =============================================================================

__all__ = [
    # 🔧 경로 정보
    'BACKEND_DIR',
    'AI_MODELS_DIR',
    'CORE_CONFIG_DIR',
    
    # 🔗 지연 로딩 함수들
    'get_config_class',
    'get_gpu_config_class', 
    'get_session_manager_class',
    'get_di_container_class',
    
    # 🏭 팩토리 함수들
    'create_config',
    'create_gpu_config',
    'create_session_manager',
    
    # 🌍 전역 인스턴스 함수들
    'get_global_config',
    'get_global_gpu_config',
    'get_global_session_manager',
    
    # 🔧 편의 함수들
    'get_ai_models_path',
    'get_system_device',
    'is_conda_optimized',
    'get_memory_gb',
    'get_core_status',
]

# 초기화 정보 출력
logger.info("🔧 MyCloset AI Core 모듈 초기화 완료")
logger.info(f"📁 AI Models 경로: {AI_MODELS_DIR}")
logger.info(f"📁 경로 존재: {AI_MODELS_DIR.exists()}")
logger.info(f"🐍 conda 최적화: {IS_CONDA}")
logger.info(f"🍎 M3 Max: {SYSTEM_INFO.get('is_m3_max', False)}")
logger.info(f"🖥️  디바이스: {SYSTEM_INFO.get('device', 'cpu')}")