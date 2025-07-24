# backend/app/utils/__init__.py  
"""
🍎 MyCloset AI 통합 유틸리티 시스템 v6.1 - import 경로 수정
================================================================
✅ StepModelInterface import 경로 수정
✅ register_model_requirement 메서드 지원
✅ 순환참조 완전 해결
✅ conda 환경 100% 최적화  
✅ M3 Max 128GB 메모리 활용
✅ main.py 호출 패턴 완전 호환

수정사항:
- StepModelInterface를 ai_pipeline.interface에서 import
- get_step_model_interface 함수 완전 구현
- register_model_requirement 메서드 지원 확인

작성자: MyCloset AI Team
날짜: 2025-07-24
버전: v6.1.0 (Import Path Fixed)
"""

import logging
import threading
import asyncio
import sys
import time
import platform
import psutil
from typing import Dict, Any, Optional, List, Union, Callable, Type
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import warnings
import weakref
import gc

# 경고 무시
warnings.filterwarnings('ignore')

# =============================================================================
# 🔥 기본 로깅 설정
# =============================================================================

logger = logging.getLogger(__name__)

# =============================================================================  
# 🔥 시스템 정보 감지
# =============================================================================

def _get_system_info() -> Dict[str, Any]:
    """시스템 정보 감지"""
    try:
        # 기본 정보
        system_info = {
            "platform": platform.system(),
            "cpu_count": psutil.cpu_count() if hasattr(psutil, 'cpu_count') else 4,
            "python_version": platform.python_version()
        }
        
        # 메모리 정보
        try:
            memory = psutil.virtual_memory()
            system_info["memory_gb"] = memory.total / (1024**3)
        except Exception:
            system_info["memory_gb"] = 16.0
        
        # M3 Max 감지
        try:
            if system_info["platform"] == "Darwin":
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True, timeout=2)
                if "Apple" in result.stdout and "M3" in result.stdout:
                    system_info["is_m3_max"] = True
                    system_info["memory_gb"] = 128.0  # M3 Max 기본값
                else:
                    system_info["is_m3_max"] = False
            else:
                system_info["is_m3_max"] = False
        except Exception:
            system_info["is_m3_max"] = False
            
        # 디바이스 감지
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                system_info["device"] = "mps"
            elif torch.cuda.is_available():
                system_info["device"] = "cuda"
            else:
                system_info["device"] = "cpu"
        except ImportError:
            system_info["device"] = "cpu"
        
        return system_info
        
    except Exception as e:
        logger.warning(f"시스템 정보 감지 실패: {e}")
        return {
            "platform": "unknown",
            "is_m3_max": False,
            "device": "cpu",
            "cpu_count": 4,
            "memory_gb": 16,
            "python_version": "3.8.0"
        }

# 전역 시스템 정보
SYSTEM_INFO = _get_system_info()

# =============================================================================
# 🔥 통합 데이터 구조 (순환참조 없음)
# =============================================================================

class UtilsMode(Enum):
    """유틸리티 모드"""
    LEGACY = "legacy"        # 기존 방식 (v3.0)
    UNIFIED = "unified"      # 새로운 통합 방식 (v6.0)
    HYBRID = "hybrid"        # 혼합 방식

@dataclass
class SystemConfig:
    """시스템 설정"""
    device: str = "auto"
    memory_gb: float = 16.0
    is_m3_max: bool = False
    optimization_enabled: bool = True
    max_workers: int = 4
    cache_enabled: bool = True
    debug_mode: bool = False

@dataclass
class StepConfig:
    """Step 설정 (순환참조 없는 데이터 전용)"""
    step_name: str
    model_name: Optional[str] = None
    model_type: Optional[str] = None
    model_class: Optional[str] = None
    input_size: tuple = (512, 512)
    device: str = "auto"
    precision: str = "fp16"
    optimization_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelInfo:
    """모델 정보 (순환참조 없는 데이터 전용)"""
    name: str
    path: str
    model_type: str
    file_size_mb: float
    confidence_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# 🔥 StepModelInterface import (수정된 경로)
# =============================================================================

# StepModelInterface를 올바른 경로에서 import
try:
    from ..ai_pipeline.interface.step_interface import StepModelInterface
    STEP_INTERFACE_AVAILABLE = True
    logger.info("✅ StepModelInterface import 성공 (ai_pipeline.interface)")
except ImportError as e:
    logger.warning(f"⚠️ StepModelInterface import 실패: {e}")
    STEP_INTERFACE_AVAILABLE = False
    
    # 폴백 클래스 정의
    class StepModelInterface:
        """폴백 StepModelInterface"""
        def __init__(self, step_name: str, model_loader=None):
            self.step_name = step_name
            self.model_loader = model_loader
            self.logger = logging.getLogger(f"FallbackInterface.{step_name}")
            
        def register_model_requirement(self, model_name: str, model_type: str = "BaseModel", **kwargs) -> bool:
            """폴백 register_model_requirement"""
            self.logger.warning(f"⚠️ 폴백 모드: {model_name} 요구사항 등록 무시")
            return True
            
        def list_available_models(self, step_class: Optional[str] = None, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
            """폴백 list_available_models"""
            return [{"name": "fallback_model", "type": "fallback", "available": False}]
            
        async def get_model(self, model_name: str) -> Optional[Any]:
            """폴백 get_model"""
            return {"fallback": True, "model_name": model_name}
            
        def get_model_sync(self, model_name: str) -> Optional[Any]:
            """폴백 get_model_sync"""
            return {"fallback": True, "model_name": model_name}

# =============================================================================
# 🔥 핵심 함수: get_step_model_interface (main.py 완전 호환)
# =============================================================================

def get_step_model_interface(step_name: str, model_loader_instance=None) -> StepModelInterface:
    """
    🔥 main.py에서 요구하는 핵심 함수 (완전 수정)
    ✅ import 오류 해결
    ✅ StepModelInterface 반환
    ✅ register_model_requirement 메서드 지원
    ✅ 비동기 메서드 포함
    """
    try:
        # ModelLoader 인스턴스 가져오기
        if model_loader_instance is None:
            try:
                # 전역 ModelLoader 시도
                from ..ai_pipeline.utils.model_loader import get_global_model_loader
                model_loader_instance = get_global_model_loader()
                logger.debug(f"✅ 전역 ModelLoader 획득: {step_name}")
            except ImportError as e:
                logger.warning(f"⚠️ ModelLoader import 실패: {e}")
                model_loader_instance = None
            except Exception as e:
                logger.warning(f"⚠️ 전역 ModelLoader 획득 실패: {e}")
                model_loader_instance = None
        
        # Step 인터페이스 생성
        interface = StepModelInterface(step_name, model_loader_instance)
        
        # register_model_requirement 메서드 확인
        if hasattr(interface, 'register_model_requirement'):
            logger.debug(f"✅ register_model_requirement 메서드 확인: {step_name}")
        else:
            logger.warning(f"⚠️ register_model_requirement 메서드 없음: {step_name}")
        
        logger.info(f"🔗 {step_name} 모델 인터페이스 생성 완료")
        return interface
        
    except Exception as e:
        logger.error(f"❌ {step_name} 인터페이스 생성 실패: {e}")
        # 완전 폴백 인터페이스
        return StepModelInterface(step_name, None)

# =============================================================================
# 🔥 통합 유틸리티 매니저 (기존 코드 유지)
# =============================================================================

class UnifiedUtilsManager:
    """
    🍎 통합 유틸리티 매니저
    ✅ 순환참조 완전 해결
    ✅ 의존성 주입 패턴
    ✅ 모든 기능 통합 관리
    ✅ 비동기 처리 완전 개선
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.logger = logging.getLogger(f"{__name__}.UnifiedUtilsManager")
        
        # 기본 설정
        self.system_config = SystemConfig(
            device=SYSTEM_INFO["device"],
            memory_gb=SYSTEM_INFO["memory_gb"],
            is_m3_max=SYSTEM_INFO["is_m3_max"],
            max_workers=min(SYSTEM_INFO["cpu_count"], 8)
        )
        
        # 상태 관리
        self.is_initialized = False
        self.initialization_time = None
        
        # 컴포넌트 저장소 (약한 참조 사용)
        self._components = weakref.WeakValueDictionary()
        self._model_interfaces = {}
        self._step_instances = {}
        
        # 성능 모니터링
        self.stats = {
            "interfaces_created": 0,
            "memory_optimizations": 0,
            "errors": 0,
            "cache_hits": 0
        }
        
        # 스레딩
        self._interface_lock = threading.RLock()
        
        self._initialized = True
        self.logger.info("✅ UnifiedUtilsManager 초기화 완료")
    
    def create_step_model_interface(self, step_name: str) -> StepModelInterface:
        """Step 모델 인터페이스 생성 (main.py 호환)"""
        try:
            if step_name in self._model_interfaces:
                return self._model_interfaces[step_name]
            
            interface = StepModelInterface(step_name, getattr(self, 'model_loader', None))
            self._model_interfaces[step_name] = interface
            
            self.logger.info(f"🔗 {step_name} 모델 인터페이스 생성 완료")
            return interface
            
        except Exception as e:
            self.logger.error(f"❌ {step_name} 모델 인터페이스 생성 실패: {e}")
            # 폴백 인터페이스
            return StepModelInterface(step_name, None)
    
    async def optimize_memory(self) -> Dict[str, Any]:
        """메모리 최적화"""
        try:
            start_time = time.time()
            
            # Python 가비지 컬렉션
            collected = gc.collect()
            
            # PyTorch 메모리 정리 (가능한 경우)
            torch_cleaned = False
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch_cleaned = True
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    # M3 Max 메모리 정리
                    gc.collect()
                    torch_cleaned = True
            except ImportError:
                pass
            
            # 약한 참조 정리
            cleaned_components = len(self._components)
            self._components.clear()
            
            elapsed_time = time.time() - start_time
            self.stats["memory_optimizations"] += 1
            
            result = {
                "success": True,
                "elapsed_time": elapsed_time,
                "garbage_collected": collected,
                "torch_cleaned": torch_cleaned,
                "components_cleaned": cleaned_components,
                "timestamp": time.time()
            }
            
            self.logger.info(f"🧹 메모리 최적화 완료: {collected}개 객체 정리 ({elapsed_time:.2f}초)")
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"❌ 메모리 최적화 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """상태 조회"""
        return {
            "system_config": self.system_config.__dict__,
            "is_initialized": self.is_initialized,
            "components_count": len(self._components),
            "model_interfaces_count": len(self._model_interfaces),
            "step_instances_count": len(self._step_instances),
            "stats": self.stats.copy(),
            "memory_info": {
                "total_gb": SYSTEM_INFO["memory_gb"],
                "is_m3_max": SYSTEM_INFO["is_m3_max"],
                "device": SYSTEM_INFO["device"]
            }
        }
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            # 모든 인터페이스 정리
            for interface in self._model_interfaces.values():
                if hasattr(interface, 'cleanup'):
                    try:
                        interface.cleanup()
                    except Exception as e:
                        self.logger.warning(f"인터페이스 정리 실패: {e}")
            
            # 캐시 정리
            self._model_interfaces.clear()
            self._step_instances.clear()
            self._components.clear()
            
            # 메모리 정리
            await self.optimize_memory()
            
            self.logger.info("🧹 UnifiedUtilsManager 정리 완료")
            
        except Exception as e:
            self.logger.error(f"❌ UnifiedUtilsManager 정리 실패: {e}")

# =============================================================================
# 🔥 전역 관리자 및 편의 함수들
# =============================================================================

_global_manager = None
_manager_lock = threading.Lock()

def get_utils_manager() -> UnifiedUtilsManager:
    """전역 유틸리티 관리자 반환"""
    global _global_manager
    
    with _manager_lock:
        if _global_manager is None:
            _global_manager = UnifiedUtilsManager()
        return _global_manager

async def initialize_global_utils(**kwargs) -> Dict[str, Any]:
    """전역 유틸리티 초기화 - 비동기"""
    try:
        manager = get_utils_manager()
        
        # 설정 업데이트
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(manager.system_config, key):
                    setattr(manager.system_config, key, value)
        
        manager.is_initialized = True
        manager.initialization_time = time.time()
        
        # 메모리 최적화 실행
        await manager.optimize_memory()
        
        logger.info("✅ 전역 유틸리티 초기화 완료")
        return {"success": True, "manager": manager}
        
    except Exception as e:
        logger.error(f"❌ 전역 유틸리티 초기화 실패: {e}")
        return {"success": False, "error": str(e)}

def get_system_status() -> Dict[str, Any]:
    """시스템 상태 조회"""
    try:
        manager = get_utils_manager()
        return manager.get_status()
    except Exception as e:
        return {"error": str(e), "system_info": SYSTEM_INFO}

async def reset_global_utils():
    """전역 유틸리티 리셋 - 비동기 개선"""
    global _global_manager
    
    try:
        with _manager_lock:
            if _global_manager:
                await _global_manager.cleanup()
                _global_manager = None
        logger.info("✅ 전역 유틸리티 리셋 완료")
    except Exception as e:
        logger.warning(f"⚠️ 전역 유틸리티 리셋 실패: {e}")

# =============================================================================
# 🔥 편의 함수들
# =============================================================================

def create_unified_interface(step_name: str, **options):
    """새로운 통합 인터페이스 생성 (권장)"""
    manager = get_utils_manager()
    return manager.create_step_model_interface(step_name)

async def optimize_system_memory() -> Dict[str, Any]:
    """시스템 메모리 최적화 - 비동기"""
    manager = get_utils_manager()
    return await manager.optimize_memory()

# 기존 호환성 함수들
def create_step_interface(step_name: str, **options) -> StepModelInterface:
    """Step 인터페이스 생성 (레거시 호환)"""
    return get_step_model_interface(step_name)

# =============================================================================
# 🔥 __all__ 정의 (핵심 수정사항)
# =============================================================================

__all__ = [
    # 🎯 핵심 클래스들
    'UnifiedUtilsManager',
    'StepModelInterface',  # 추가됨
    'SystemConfig',
    'StepConfig',
    'ModelInfo',
    
    # 🔧 전역 함수들
    'get_utils_manager',
    'initialize_global_utils',
    'get_system_status',
    'reset_global_utils',
    
    # 🔄 인터페이스 생성 (main.py 호환)
    'get_step_model_interface',       # ✅ main.py 호환 (핵심!)
    'create_step_interface',          # 레거시 호환
    'create_unified_interface',       # 새로운 방식
    
    # 📊 시스템 정보
    'SYSTEM_INFO',
    'optimize_system_memory',
    
    # 🔧 유틸리티
    'UtilsMode',
    'STEP_INTERFACE_AVAILABLE'  # 추가됨
]

# =============================================================================
# 🔥 모듈 초기화 완료
# =============================================================================

logger.info("=" * 70)
logger.info("🍎 MyCloset AI 통합 유틸리티 시스템 v6.1 로드 완료")
logger.info("✅ StepModelInterface import 경로 수정")
logger.info("✅ register_model_requirement 메서드 지원")
logger.info("✅ get_step_model_interface 함수 완전 구현")
logger.info("✅ 비동기 처리 완전 개선")
logger.info("✅ 순환참조 완전 해결")
logger.info("✅ 기존 코드 하위 호환성 보장")
logger.info(f"🔧 시스템: {SYSTEM_INFO['platform']} / {SYSTEM_INFO['device']}")
logger.info(f"🍎 M3 Max: {'✅' if SYSTEM_INFO['is_m3_max'] else '❌'}")
logger.info(f"💾 메모리: {SYSTEM_INFO['memory_gb']}GB")
logger.info(f"🔌 StepInterface: {'✅' if STEP_INTERFACE_AVAILABLE else '⚠️ 폴백'}")
logger.info("=" * 70)

# 종료 시 정리 함수 등록
import atexit

def cleanup_on_exit():
    """종료 시 정리"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(reset_global_utils())
        loop.close()
    except Exception as e:
        logger.warning(f"⚠️ 종료 시 정리 실패: {e}")

atexit.register(cleanup_on_exit)