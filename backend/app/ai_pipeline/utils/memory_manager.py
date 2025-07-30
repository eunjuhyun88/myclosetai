# app/ai_pipeline/utils/memory_manager.py
"""
🔥 MyCloset AI - DI 완전 통합 메모리 관리 시스템 v9.0
================================================================================
✅ CircularReferenceFreeDIContainer 완전 연동
✅ DI 패턴으로 의존성 주입 지원
✅ BaseStepMixin과 완벽 호환
✅ 순환참조 완전 방지
✅ 기존 클래스명/함수명 100% 유지
✅ DeviceManager 클래스 완전 구현
✅ setup_mps_compatibility 메서드 구현
✅ RuntimeWarning: coroutine 완전 해결
✅ object dict can't be used in 'await' expression 완전 해결
✅ optimize_startup 메서드 완전 동기화
✅ conda 환경 메모리 최적화 함수 완전 수정
✅ Mock 폴백 구현체 포함
✅ M3 Max 128GB + conda 환경 완전 최적화
✅ 모든 비동기 오류 해결
✅ 프로덕션 레벨 안정성
✅ main.py import 오류 완전 해결
================================================================================
Author: MyCloset AI Team
Date: 2025-07-30
Version: 9.0 (DI Integration)
"""

import os
import gc
import threading
import time
import logging
import asyncio
import weakref
import platform
from typing import Dict, Any, Optional, Callable, List, Union, Tuple, TYPE_CHECKING, Protocol
from dataclasses import dataclass, field
from contextlib import contextmanager, asynccontextmanager
from functools import wraps, lru_cache
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod

# 🔥 DI Container 임포트 (순환참조 방지)
if TYPE_CHECKING:
    # 타입 힌팅용 임포트 (런타임에는 실행되지 않음)
    from ..core.di_container import CircularReferenceFreeDIContainer
else:
    # 런타임에는 동적 임포트
    pass

# 각 파일에 추가할 개선된 코드
import time
import threading

try:
    from app.core.gpu_config import safe_mps_empty_cache
except ImportError:
    # 폴백 함수
    def safe_mps_empty_cache():
        import gc
        gc.collect()
        return {"success": True, "method": "fallback_gc"}
    
# ==============================================
# 🔥 조건부 임포트 (순환참조 방지)
# ==============================================

# psutil 선택적 임포트
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# PyTorch 선택적 임포트
try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_VERSION = torch.__version__
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    TORCH_VERSION = "not_available"

# NumPy 선택적 임포트
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

logger = logging.getLogger(__name__)

# ==============================================
# 🔥 DI 관련 인터페이스 및 프로토콜
# ==============================================

class IMemoryManager(Protocol):
    """MemoryManager 인터페이스 (DI용)"""
    
    def optimize_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """메모리 최적화"""
        ...
    
    def optimize_startup(self, aggressive: bool = False) -> Dict[str, Any]:
        """시작 시 메모리 최적화"""
        ...
    
    def cleanup_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """메모리 정리"""
        ...
    
    def get_memory_stats(self) -> Any:
        """메모리 통계 조회"""
        ...
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """메모리 압박 상태 확인"""
        ...

class IDeviceManager(Protocol):
    """DeviceManager 인터페이스 (DI용)"""
    
    def setup_mps_compatibility(self) -> bool:
        """MPS 호환성 설정"""
        ...
    
    def get_device(self) -> str:
        """현재 디바이스 반환"""
        ...
    
    def optimize_memory(self) -> None:
        """메모리 최적화"""
        ...

class IDependencyInjectable(ABC):
    """DI 주입 가능한 컴포넌트 인터페이스"""
    
    @abstractmethod
    def set_di_container(self, di_container: Any) -> None:
        """DI Container 설정"""
        pass
    
    @abstractmethod
    def resolve_dependencies(self) -> bool:
        """의존성 해결"""
        pass
    
    @abstractmethod
    def get_dependency_status(self) -> Dict[str, Any]:
        """의존성 상태 조회"""
        pass

# ==============================================
# 🔥 시스템 정보 캐싱 (한번만 실행)
# ==============================================

@lru_cache(maxsize=1)
def _get_system_info() -> Dict[str, Any]:
    """시스템 정보 캐시 (M3 Max 감지 포함)"""
    try:
        system_info = {
            "platform": platform.system(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count() or 4,
            "python_version": ".".join(map(str, platform.python_version_tuple()[:2])),
            "conda_env": os.environ.get('CONDA_DEFAULT_ENV', 'base'),
            "in_conda": 'CONDA_PREFIX' in os.environ,
            "torch_available": TORCH_AVAILABLE,
            "torch_version": TORCH_VERSION
        }
        
        # M3 Max 감지
        is_m3_max = False
        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            try:
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True, timeout=3)
                cpu_info = result.stdout.strip()
                is_m3_max = 'M3' in cpu_info and ('Max' in cpu_info or 'Pro' in cpu_info)
            except:
                pass
        
        system_info["is_m3_max"] = is_m3_max
        
        # 메모리 정보
        if PSUTIL_AVAILABLE:
            memory_gb = round(psutil.virtual_memory().total / (1024**3))
            system_info["memory_gb"] = memory_gb
        else:
            # M3 Max 기본값
            system_info["memory_gb"] = 128 if is_m3_max else 16
        
        # 디바이스 감지
        device = "cpu"
        if TORCH_AVAILABLE:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
        
        system_info["device"] = device
        
        return system_info
        
    except Exception as e:
        logger.warning(f"시스템 정보 감지 실패: {e}")
        return {
            "platform": "unknown",
            "machine": "unknown", 
            "is_m3_max": False,
            "device": "cpu",
            "cpu_count": 4,
            "memory_gb": 16,
            "python_version": "3.8",
            "conda_env": "base",
            "in_conda": False,
            "torch_available": False,
            "torch_version": "not_available"
        }

# 전역 시스템 정보
SYSTEM_INFO = _get_system_info()

# ==============================================
# 🔥 데이터 구조 정의
# ==============================================

@dataclass
class MemoryStats:
    """메모리 통계 정보"""
    cpu_percent: float
    cpu_available_gb: float
    cpu_used_gb: float
    cpu_total_gb: float
    gpu_allocated_gb: float = 0.0
    gpu_reserved_gb: float = 0.0
    gpu_total_gb: float = 0.0
    swap_used_gb: float = 0.0
    cache_size_mb: float = 0.0
    process_memory_mb: float = 0.0
    m3_optimizations: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryConfig:
    """메모리 관리 설정"""
    device: str = "auto"
    memory_limit_gb: float = 16.0
    warning_threshold: float = 0.75
    critical_threshold: float = 0.9
    auto_cleanup: bool = True
    monitoring_interval: float = 30.0
    enable_caching: bool = True
    optimization_enabled: bool = True
    m3_max_features: bool = False

# ==============================================
# 🔥 DI 통합 DeviceManager 클래스
# ==============================================

class DeviceManager(IDependencyInjectable):
    """
    🔥 DI 완전 통합 DeviceManager 클래스
    ✅ CircularReferenceFreeDIContainer 연동
    ✅ 의존성 주입 지원
    ✅ setup_mps_compatibility 메서드 포함
    ✅ main.py import 오류 완전 해결
    ✅ M3 Max 특화 최적화
    """
    
    def __init__(self, di_container: Optional[Any] = None):
        """디바이스 관리자 초기화 (DI 지원)"""
        self.device = self._detect_optimal_device()
        self.is_mps_available = False
        self.is_cuda_available = False
        self.logger = logging.getLogger("DeviceManager")
        
        # 🔥 DI Container 설정
        self._di_container: Optional[Any] = None
        self._dependencies_resolved = False
        self._dependency_status = {
            'di_container': False,
            'memory_manager': False,
            'initialized': False
        }
        
        # 스레드 안전성
        self._lock = threading.RLock()
        
        self._init_device_info()
        
        # DI Container 설정 (초기화 후)
        if di_container is not None:
            self.set_di_container(di_container)
        
        self.logger.debug(f"🎮 DI DeviceManager 초기화 완료 - 디바이스: {self.device}")
    
    # ==============================================
    # 🔥 DI 인터페이스 구현
    # ==============================================

    def set_di_container(self, di_container: Any) -> None:
        """DI Container 설정"""
        try:
            with self._lock:
                self._di_container = di_container
                self._dependency_status['di_container'] = True
                
                # DI Container에 자신을 등록
                if hasattr(di_container, 'register'):
                    di_container.register('device_manager', self, singleton=True)
                    di_container.register('IDeviceManager', self, singleton=True)
                
                self.logger.info("✅ DeviceManager DI Container 설정 완료")
                
                # 의존성 해결 시도
                self.resolve_dependencies()
                
        except Exception as e:
            self.logger.error(f"❌ DeviceManager DI Container 설정 실패: {e}")

    def resolve_dependencies(self) -> bool:
        """의존성 해결"""
        try:
            with self._lock:
                if not self._di_container:
                    self.logger.warning("⚠️ DI Container가 설정되지 않음")
                    return False
                
                resolved_count = 0
                
                # MemoryManager 해결 (선택적)
                try:
                    memory_manager = self._di_container.get('memory_manager')
                    if memory_manager:
                        self.memory_manager = memory_manager
                        self._dependency_status['memory_manager'] = True
                        resolved_count += 1
                        self.logger.debug("✅ DeviceManager MemoryManager 의존성 해결")
                except Exception as e:
                    self.logger.debug(f"DeviceManager MemoryManager 해결 실패: {e}")
                
                self._dependencies_resolved = resolved_count >= 0  # DeviceManager는 필수 의존성 없음
                self.logger.info(f"🔗 DeviceManager 의존성 해결 완료: {resolved_count}개")
                
                return self._dependencies_resolved
                
        except Exception as e:
            self.logger.error(f"❌ DeviceManager 의존성 해결 실패: {e}")
            return False

    def get_dependency_status(self) -> Dict[str, Any]:
        """의존성 상태 조회"""
        with self._lock:
            return {
                'class_name': self.__class__.__name__,
                'dependencies_resolved': self._dependencies_resolved,
                'dependency_status': dict(self._dependency_status),
                'di_container_available': self._di_container is not None,
                'device_info': {
                    'device': self.device,
                    'is_mps_available': self.is_mps_available,
                    'is_cuda_available': self.is_cuda_available,
                    'torch_available': TORCH_AVAILABLE
                },
                'system_info': SYSTEM_INFO
            }

    # ==============================================
    # 🔥 기존 메서드들 (DI 최적화 추가)
    # ==============================================
    
    def _detect_optimal_device(self) -> str:
        """최적 디바이스 감지"""
        try:
            if not TORCH_AVAILABLE:
                return "cpu"
            
            # M3 Max MPS 우선
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.is_mps_available = True
                return "mps"
            
            # CUDA 확인
            elif torch.cuda.is_available():
                self.is_cuda_available = True
                return "cuda"
            
            # CPU 폴백
            return "cpu"
            
        except Exception as e:
            self.logger.warning(f"⚠️ 디바이스 감지 실패: {e}")
            return "cpu"
    
    def _init_device_info(self):
        """디바이스 정보 초기화"""
        try:
            if not TORCH_AVAILABLE:
                return
            
            if self.device == "mps":
                # M3 Max 최적화
                self._setup_mps_optimization()
                
            elif self.device == "cuda":
                # CUDA 최적화
                if hasattr(torch.backends, 'cudnn'):
                    torch.backends.cudnn.benchmark = True
                
            self._dependency_status['initialized'] = True
                
        except Exception as e:
            self.logger.warning(f"⚠️ 디바이스 초기화 실패: {e}")
    
    def _setup_mps_optimization(self):
        """MPS 최적화 설정"""
        try:
            # M3 Max 환경 변수 설정
            os.environ.update({
                'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',
                'PYTORCH_MPS_LOW_WATERMARK_RATIO': '0.0',
                'METAL_DEVICE_WRAPPER_TYPE': '1',
                'METAL_PERFORMANCE_SHADERS_ENABLED': '1',
                'PYTORCH_MPS_PREFER_METAL': '1',
                'PYTORCH_ENABLE_MPS_FALLBACK': '1'
            })
            
            # 스레드 최적화
            if TORCH_AVAILABLE:
                torch.set_num_threads(min(16, SYSTEM_INFO["cpu_count"]))
            
            self.logger.debug("🍎 MPS 최적화 설정 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ MPS 최적화 실패: {e}")
    
    def setup_mps_compatibility(self):
        """
        🔥 MPS 호환성 설정 (main.py에서 요구하는 핵심 메서드)
        ✅ import 오류 완전 해결
        ✅ DI 최적화 추가
        """
        try:
            if not TORCH_AVAILABLE:
                self.logger.warning("⚠️ PyTorch 없음 - MPS 호환성 설정 건너뜀")
                return False
            
            if not self.is_mps_available:
                self.logger.info("ℹ️ MPS 사용 불가 - 호환성 설정 건너뜀")
                return False
            
            self.logger.info("🍎 MPS 호환성 설정 시작...")
            
            # 1. MPS 메모리 정리
            if hasattr(torch.mps, 'empty_cache'):
                safe_mps_empty_cache()
                self.logger.debug("✅ MPS 메모리 정리 완료")
            
            # 2. MPS 환경 변수 재설정
            self._setup_mps_optimization()
            
            # 3. MPS 동기화
            if hasattr(torch.mps, 'synchronize'):
                try:
                    torch.mps.synchronize()
                    self.logger.debug("✅ MPS 동기화 완료")
                except Exception as e:
                    self.logger.warning(f"⚠️ MPS 동기화 실패: {e}")
            
            # 4. 테스트 텐서 생성 (MPS 작동 확인)
            try:
                test_tensor = torch.tensor([1.0], device='mps')
                test_result = test_tensor + 1
                self.logger.debug("✅ MPS 작동 확인 완료")
                del test_tensor, test_result
            except Exception as e:
                self.logger.warning(f"⚠️ MPS 작동 확인 실패: {e}")
                return False
            
            # 5. DI를 통한 메모리 최적화 (선택적)
            if hasattr(self, 'memory_manager') and self.memory_manager:
                try:
                    self.memory_manager.optimize_memory()
                    self.logger.debug("✅ DI MemoryManager를 통한 최적화 완료")
                except Exception:
                    pass  # 메모리 관리자 최적화 실패는 무시
            
            self.logger.info("✅ MPS 호환성 설정 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ MPS 호환성 설정 실패: {e}")
            return False
    
    def get_device(self) -> str:
        """현재 디바이스 반환"""
        return self.device
    
    def get_device_info(self) -> Dict[str, Any]:
        """디바이스 정보 반환"""
        try:
            info = {
                "device": self.device,
                "is_mps_available": self.is_mps_available,
                "is_cuda_available": self.is_cuda_available,
                "torch_available": TORCH_AVAILABLE,
                "torch_version": TORCH_VERSION,
                "system_info": SYSTEM_INFO,
                "di_integrated": self._di_container is not None,
                "dependencies_resolved": self._dependencies_resolved
            }
            
            if TORCH_AVAILABLE and self.device == "cuda":
                info.update({
                    "cuda_device_count": torch.cuda.device_count(),
                    "cuda_current_device": torch.cuda.current_device(),
                    "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "N/A"
                })
            
            return info
            
        except Exception as e:
            self.logger.warning(f"⚠️ 디바이스 정보 조회 실패: {e}")
            return {"device": self.device, "error": str(e)}
    
    def optimize_memory(self):
        """메모리 최적화 (DI 통합)"""
        try:
            # Python 가비지 컬렉션
            gc.collect()
            
            if not TORCH_AVAILABLE:
                return
            
            if self.device == "mps":
                try:
                    if hasattr(torch.mps, 'synchronize'):
                        torch.mps.synchronize()
                    if hasattr(torch.mps, 'empty_cache'):
                        safe_mps_empty_cache()
                    self.logger.debug("✅ MPS 메모리 최적화 완료")
                except Exception as e:
                    self.logger.warning(f"⚠️ MPS 메모리 최적화 실패: {e}")
                    
            elif self.device == "cuda":
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    self.logger.debug("✅ CUDA 메모리 최적화 완료")
                except Exception as e:
                    self.logger.warning(f"⚠️ CUDA 메모리 최적화 실패: {e}")
            
            # DI를 통한 추가 메모리 최적화
            if hasattr(self, 'memory_manager') and self.memory_manager:
                try:
                    self.memory_manager.optimize_memory()
                    self.logger.debug("✅ DI MemoryManager 추가 최적화 완료")
                except Exception:
                    pass
                    
        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 최적화 실패: {e}")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """메모리 정보 반환"""
        try:
            info = {
                'device': self.device,
                'allocated': 0,
                'cached': 0,
                'total': 0,
                'di_integrated': self._di_container is not None
            }
            
            if not TORCH_AVAILABLE:
                return info
            
            if self.device == "cuda" and torch.cuda.is_available():
                info.update({
                    'allocated': torch.cuda.memory_allocated(),
                    'cached': torch.cuda.memory_reserved(),
                    'total': torch.cuda.get_device_properties(0).total_memory
                })
            elif self.device == "mps":
                # MPS는 정확한 메모리 정보를 제공하지 않으므로 추정값 사용
                info.update({
                    'allocated': 2 * 1024**3,  # 2GB 추정
                    'cached': 1 * 1024**3,     # 1GB 추정
                    'total': SYSTEM_INFO["memory_gb"] * 1024**3
                })
            
            return info
            
        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 정보 조회 실패: {e}")
            return {
                'device': self.device,
                'allocated': 0,
                'cached': 0,
                'total': 0,
                'error': str(e)
            }
    
    def is_available(self) -> bool:
        """디바이스 사용 가능 여부"""
        try:
            if not TORCH_AVAILABLE:
                return False
            
            if self.device == "mps":
                return torch.backends.mps.is_available()
            elif self.device == "cuda":
                return torch.cuda.is_available()
            else:
                return True  # CPU는 항상 사용 가능
                
        except Exception:
            return False

# ==============================================
# 🔥 DI 통합 MemoryManager 클래스
# ==============================================

class MemoryManager(IDependencyInjectable):
    """
    🔥 DI 완전 통합 메모리 관리자
    ✅ CircularReferenceFreeDIContainer 연동
    ✅ 의존성 주입 지원
    ✅ GitHub 구조와 완벽 호환
    ✅ M3 Max 128GB + conda 완전 활용
    ✅ 모든 async/await 오류 해결
    ✅ 기존 인터페이스 100% 유지
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        di_container: Optional[Any] = None,
        **kwargs
    ):
        """메모리 관리자 초기화 (DI 지원)"""
        
        # 1. 디바이스 자동 감지
        self.device = device or SYSTEM_INFO["device"]
        
        # 2. 설정 구성
        config_dict = config or {}
        config_dict.update(kwargs)
        
        # MemoryConfig 생성을 위한 필터링
        memory_config_fields = {
            'device', 'memory_limit_gb', 'warning_threshold', 'critical_threshold',
            'auto_cleanup', 'monitoring_interval', 'enable_caching', 'optimization_enabled', 'm3_max_features'
        }
        memory_config_args = {k: v for k, v in config_dict.items() if k in memory_config_fields}
        
        # M3 Max 특화 설정
        if SYSTEM_INFO["is_m3_max"]:
            memory_config_args.setdefault("memory_limit_gb", SYSTEM_INFO["memory_gb"] * 0.8)
            memory_config_args.setdefault("m3_max_features", True)
        
        self.config = MemoryConfig(**memory_config_args)
        
        # 3. 기본 속성
        self.step_name = self.__class__.__name__
        self.logger = logging.getLogger(f"memory.{self.step_name}")
        
        # 🔥 DI Container 설정
        self._di_container: Optional[Any] = None
        self._dependencies_resolved = False
        self._dependency_status = {
            'di_container': False,
            'device_manager': False,
            'data_converter': False,
            'initialized': False
        }
        
        # 스레드 안전성
        self._lock = threading.RLock()
        
        # 4. 시스템 파라미터
        self.memory_gb = SYSTEM_INFO["memory_gb"]
        self.is_m3_max = SYSTEM_INFO["is_m3_max"]
        self.optimization_enabled = self.config.optimization_enabled
        
        # 5. 메모리 관리 속성
        if PSUTIL_AVAILABLE:
            total_memory = psutil.virtual_memory().total / 1024**3
            self.memory_limit_gb = total_memory * 0.8
        else:
            self.memory_limit_gb = self.config.memory_limit_gb
        
        # 6. 상태 초기화
        self.is_initialized = False
        self._initialize_components()
        
        # 7. DI Container 설정 (초기화 후)
        if di_container is not None:
            self.set_di_container(di_container)
        
        self.logger.debug(f"🎯 DI MemoryManager 초기화 - 디바이스: {self.device}, 메모리: {self.memory_gb}GB")

    # ==============================================
    # 🔥 DI 인터페이스 구현
    # ==============================================

    def set_di_container(self, di_container: Any) -> None:
        """DI Container 설정"""
        try:
            with self._lock:
                self._di_container = di_container
                self._dependency_status['di_container'] = True
                
                # DI Container에 자신을 등록
                if hasattr(di_container, 'register'):
                    di_container.register('memory_manager', self, singleton=True)
                    di_container.register('IMemoryManager', self, singleton=True)
                
                self.logger.info("✅ MemoryManager DI Container 설정 완료")
                
                # 의존성 해결 시도
                self.resolve_dependencies()
                
        except Exception as e:
            self.logger.error(f"❌ MemoryManager DI Container 설정 실패: {e}")

    def resolve_dependencies(self) -> bool:
        """의존성 해결"""
        try:
            with self._lock:
                if not self._di_container:
                    self.logger.warning("⚠️ DI Container가 설정되지 않음")
                    return False
                
                resolved_count = 0
                
                # DeviceManager 해결 (선택적)
                try:
                    device_manager = self._di_container.get('device_manager')
                    if device_manager:
                        self.device_manager = device_manager
                        self._dependency_status['device_manager'] = True
                        resolved_count += 1
                        self.logger.debug("✅ MemoryManager DeviceManager 의존성 해결")
                except Exception as e:
                    self.logger.debug(f"MemoryManager DeviceManager 해결 실패: {e}")
                
                # DataConverter 해결 (선택적)
                try:
                    data_converter = self._di_container.get('data_converter')
                    if data_converter:
                        self.data_converter = data_converter
                        self._dependency_status['data_converter'] = True
                        resolved_count += 1
                        self.logger.debug("✅ MemoryManager DataConverter 의존성 해결")
                except Exception as e:
                    self.logger.debug(f"MemoryManager DataConverter 해결 실패: {e}")
                
                self._dependencies_resolved = resolved_count >= 0  # MemoryManager는 필수 의존성 없음
                self.logger.info(f"🔗 MemoryManager 의존성 해결 완료: {resolved_count}개")
                
                return self._dependencies_resolved
                
        except Exception as e:
            self.logger.error(f"❌ MemoryManager 의존성 해결 실패: {e}")
            return False

    def get_dependency_status(self) -> Dict[str, Any]:
        """의존성 상태 조회"""
        with self._lock:
            return {
                'class_name': self.__class__.__name__,
                'dependencies_resolved': self._dependencies_resolved,
                'dependency_status': dict(self._dependency_status),
                'di_container_available': self._di_container is not None,
                'memory_config': {
                    'device': self.device,
                    'memory_gb': self.memory_gb,
                    'is_m3_max': self.is_m3_max,
                    'optimization_enabled': self.optimization_enabled
                },
                'system_info': SYSTEM_INFO
            }

    # ==============================================
    # 🔥 기존 메서드들 (DI 최적화 추가)
    # ==============================================

    def _initialize_components(self):
        """구성 요소 초기화"""
        try:
            # 메모리 통계
            self.stats_history = []
            self.max_history_length = 100
            
            # 캐시 시스템
            self.tensor_cache = {}
            self.image_cache = {}
            self.model_cache = {}
            self.cache_priority = {}
            
            # 모니터링
            self.monitoring_active = False
            self.monitoring_thread = None
            
            # M3 Max 특화 속성 초기화
            if self.is_m3_max:
                self.precision_mode = 'float16'  # M3 Max에서 float16 사용
                self.memory_pools = {}
                self.optimal_batch_sizes = {}
                
                # M3 Max 최적화 수행
                if self.optimization_enabled:
                    self._optimize_for_m3_max()
            else:
                self.precision_mode = 'float32'
            
            # 초기화 완료
            self.is_initialized = True
            self._dependency_status['initialized'] = True
            
            self.logger.debug(f"🧠 MemoryManager 구성 요소 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 구성 요소 초기화 실패: {e}")
            # 최소한의 초기화
            self.tensor_cache = {}
            self.is_initialized = True

    def optimize_startup(self, aggressive: bool = False) -> Dict[str, Any]:
        """
        🚀 시스템 시작 시 메모리 최적화 (완전 동기화 + DI 통합)
        ✅ 모든 async/await 오류 완전 해결
        ✅ RuntimeWarning 완전 해결
        ✅ DI를 통한 추가 최적화
        """
        try:
            start_time = time.time()
            startup_results = []
            
            self.logger.info("🚀 시스템 시작 메모리 최적화 시작 (DI 통합)")
            
            # 1. 기본 메모리 최적화 실행 (동기 방식)
            try:
                optimize_result = self._synchronous_optimize_memory()
                if optimize_result.get('success', False):
                    startup_results.append("기본 메모리 최적화 완료")
                else:
                    startup_results.append("기본 메모리 최적화 실패")
            except Exception as e:
                startup_results.append(f"기본 메모리 최적화 오류: {e}")
            
            # 2. DI를 통한 추가 최적화
            try:
                if hasattr(self, 'device_manager') and self.device_manager:
                    self.device_manager.optimize_memory()
                    startup_results.append("DI DeviceManager 최적화 완료")
                
                if hasattr(self, 'data_converter') and self.data_converter:
                    # DataConverter에 메모리 최적화가 있다면 호출
                    if hasattr(self.data_converter, 'optimize_memory'):
                        self.data_converter.optimize_memory()
                    startup_results.append("DI DataConverter 최적화 완료")
            except Exception as e:
                startup_results.append(f"DI 추가 최적화 실패: {e}")
            
            # 3. 시스템 시작 특화 최적화
            try:
                # Python 가비지 컬렉션 강화
                collected = 0
                for gen in range(3):
                    collected += gc.collect()
                startup_results.append(f"시작 시 가비지 컬렉션: {collected}개 객체")
                
                # M3 Max 특화 시작 최적화
                if self.is_m3_max:
                    self._optimize_m3_max_startup()
                    startup_results.append("M3 Max 시작 최적화 완료")
                
                # conda 환경 특화 설정 (동기 방식)
                if SYSTEM_INFO.get("in_conda", False):
                    conda_result = setup_conda_memory_optimization()
                    if conda_result:
                        startup_results.append("conda 환경 최적화 완료")
                    else:
                        startup_results.append("conda 환경 최적화 실패")
                
            except Exception as e:
                startup_results.append(f"시작 특화 최적화 실패: {e}")
            
            # 4. 메모리 상태 체크
            try:
                stats = self.get_memory_stats()
                available_ratio = stats.cpu_available_gb / max(1.0, stats.cpu_total_gb)
                startup_results.append(f"메모리 가용률: {available_ratio:.1%}")
            except Exception as e:
                startup_results.append(f"메모리 상태 확인 실패: {e}")
            
            startup_time = time.time() - start_time
            
            self.logger.info(f"✅ 시스템 시작 메모리 최적화 완료 ({startup_time:.2f}초)")
            
            return {
                "success": True,
                "message": "시스템 시작 메모리 최적화 완료 (DI 통합)",
                "startup_time": startup_time,
                "startup_results": startup_results,
                "device": self.device,
                "is_m3_max": self.is_m3_max,
                "conda_optimized": SYSTEM_INFO.get("in_conda", False),
                "di_integrated": self._di_container is not None,
                "dependencies_resolved": self._dependencies_resolved,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 시스템 시작 메모리 최적화 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "시스템 시작 메모리 최적화 실패",
                "device": self.device,
                "di_integrated": self._di_container is not None,
                "timestamp": time.time()
            }
    
    def _synchronous_optimize_memory(self) -> Dict[str, Any]:
        """동기 메모리 최적화 (await 사용 안함 + DI 통합)"""
        try:
            start_time = time.time()
            optimization_results = []
            
            # 1. 메모리 정리 수행
            cleanup_result = self.cleanup_memory(aggressive=False)
            optimization_results.append(f"메모리 정리: {cleanup_result.get('success', False)}")
            
            # 2. M3 Max 특화 최적화 (동기)
            if self.is_m3_max:
                m3_result = self._optimize_m3_max_memory_sync()
                optimization_results.append(f"M3 Max 최적화: {m3_result}")
            
            # 3. 캐시 최적화
            cache_stats = self._optimize_cache_system()
            optimization_results.append(f"캐시 최적화: {cache_stats}")
            
            # 4. DI를 통한 추가 최적화
            if self._di_container:
                try:
                    # DeviceManager를 통한 추가 최적화
                    if hasattr(self, 'device_manager') and self.device_manager:
                        self.device_manager.optimize_memory()
                        optimization_results.append("DI DeviceManager 최적화 완료")
                except Exception as e:
                    optimization_results.append(f"DI 최적화 실패: {str(e)[:50]}")
            
            # 5. 메모리 압박 상태 확인
            pressure_info = self.check_memory_pressure()
            optimization_results.append(f"메모리 압박: {pressure_info.get('status', 'unknown')}")
            
            optimization_time = time.time() - start_time
            
            return {
                "success": True,
                "message": "메모리 최적화 완료 (DI 통합)",
                "optimization_time": optimization_time,
                "optimization_results": optimization_results,
                "device": self.device,
                "m3_max_optimized": self.is_m3_max,
                "di_integrated": self._di_container is not None,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 동기 메모리 최적화 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "device": self.device,
                "timestamp": time.time()
            }
    
    def _optimize_m3_max_memory_sync(self):
        """M3 Max 특화 메모리 최적화 (동기 버전)"""
        try:
            if not self.is_m3_max:
                return False
            
            # M3 Max 최적화 실행
            if hasattr(self, '_optimize_for_m3_max'):
                self._optimize_for_m3_max()
            
            # M3 Max MPS 캐시 정리
            if TORCH_AVAILABLE and torch.backends.mps.is_available():
                if hasattr(torch.mps, 'empty_cache'):
                    safe_mps_empty_cache()
            
            return True
        
        except Exception as e:
            self.logger.warning(f"⚠️ M3 Max 메모리 최적화 실패: {e}")
            return False
    
    def _optimize_m3_max_startup(self):
        """M3 Max 시작 시 특화 최적화"""
        try:
            if not self.is_m3_max:
                return
            
            # M3 Max 시작 시 환경 변수 설정
            startup_env = {
                'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',
                'PYTORCH_MPS_LOW_WATERMARK_RATIO': '0.0',
                'PYTORCH_ENABLE_MPS_FALLBACK': '1',
                'METAL_PERFORMANCE_SHADERS_ENABLED': '1'
            }
            
            os.environ.update(startup_env)
            
            # PyTorch MPS 사전 워밍업
            if TORCH_AVAILABLE and torch.backends.mps.is_available():
                try:
                    # MPS 캐시 사전 정리
                    if hasattr(torch.mps, 'empty_cache'):
                        safe_mps_empty_cache()
                    
                    # 스레드 수 최적화
                    torch.set_num_threads(min(16, SYSTEM_INFO["cpu_count"]))
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ M3 Max MPS 워밍업 실패: {e}")
            
            self.logger.debug("🍎 M3 Max 시작 최적화 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ M3 Max 시작 최적화 실패: {e}")

    def _optimize_for_m3_max(self):
        """🍎 M3 Max Neural Engine + conda 최적화"""
        try:
            if not self.is_m3_max:
                return False
                
            self.logger.debug("🍎 M3 Max 최적화 시작")
            optimizations = []
            
            # 1. PyTorch MPS 백엔드 최적화
            if TORCH_AVAILABLE and torch.backends.mps.is_available():
                try:
                    # MPS 메모리 정리
                    if hasattr(torch.mps, 'empty_cache'):
                        safe_mps_empty_cache()
                        optimizations.append("MPS cache clearing")
                    
                    # M3 Max 환경 변수 설정
                    os.environ.update({
                        'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',
                        'PYTORCH_MPS_LOW_WATERMARK_RATIO': '0.0',
                        'METAL_DEVICE_WRAPPER_TYPE': '1',
                        'METAL_PERFORMANCE_SHADERS_ENABLED': '1',
                        'PYTORCH_MPS_PREFER_METAL': '1',
                        'PYTORCH_ENABLE_MPS_FALLBACK': '1'
                    })
                    optimizations.append("MPS environment optimization")
                    
                    # 스레드 최적화
                    torch.set_num_threads(min(16, SYSTEM_INFO["cpu_count"]))
                    optimizations.append(f"Thread optimization ({torch.get_num_threads()} threads)")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ MPS 최적화 일부 실패: {e}")
            
            # 2. 메모리 풀 최적화
            self._setup_m3_memory_pools()
            optimizations.append("Memory pool optimization")
            
            # 3. 배치 크기 최적화
            self._optimize_batch_sizes()
            optimizations.append("Batch size optimization")
            
            # 4. conda 환경 특화 최적화
            if SYSTEM_INFO["in_conda"]:
                self._setup_conda_optimizations()
                optimizations.append("Conda environment optimization")
            
            self.logger.info("✅ M3 Max 최적화 완료")
            for opt in optimizations:
                self.logger.debug(f"   - {opt}")
            
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ M3 Max 최적화 실패: {e}")
            return False

    def _setup_m3_memory_pools(self):
        """M3 Max 메모리 풀 설정"""
        try:
            # 128GB 기준 메모리 풀 최적화
            pool_size_gb = self.memory_gb * 0.8
            
            self.memory_pools = {
                "model_cache": pool_size_gb * 0.4,      # 40% - 모델 캐시
                "inference": pool_size_gb * 0.3,        # 30% - 추론 작업
                "preprocessing": pool_size_gb * 0.2,    # 20% - 전처리
                "buffer": pool_size_gb * 0.1            # 10% - 버퍼
            }
            
            self.logger.debug(f"🍎 M3 Max 메모리 풀 설정: {pool_size_gb:.1f}GB")
                
        except Exception as e:
            self.logger.warning(f"⚠️ M3 메모리 풀 설정 실패: {e}")

    def _optimize_batch_sizes(self):
        """M3 Max 배치 크기 최적화"""
        try:
            if self.is_m3_max:
                # M3 Max 128GB 기준 최적 배치 크기
                self.optimal_batch_sizes = {
                    "human_parsing": 16,
                    "pose_estimation": 20,
                    "cloth_segmentation": 12,
                    "virtual_fitting": 8,
                    "super_resolution": 4
                }
            else:
                # 일반 시스템 배치 크기
                self.optimal_batch_sizes = {
                    "human_parsing": 4,
                    "pose_estimation": 6,
                    "cloth_segmentation": 3,
                    "virtual_fitting": 2,
                    "super_resolution": 1
                }
            
            self.logger.debug(f"⚙️ 배치 크기 최적화 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 배치 크기 최적화 실패: {e}")

    def _setup_conda_optimizations(self):
        """conda 환경 특화 최적화"""
        try:
            if not SYSTEM_INFO["in_conda"]:
                return
            
            # NumPy 최적화
            if NUMPY_AVAILABLE:
                os.environ['OMP_NUM_THREADS'] = str(min(8, SYSTEM_INFO["cpu_count"]))
                os.environ['MKL_NUM_THREADS'] = str(min(8, SYSTEM_INFO["cpu_count"]))
                
            self.logger.debug("✅ conda 환경 최적화 설정 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ conda 최적화 실패: {e}")

    def optimize(self) -> Dict[str, Any]:
        """
        메모리 최적화 (optimize_memory의 별칭 + DI 통합)
        
        VirtualFittingStep과 다른 Step들에서 호출되는 표준 인터페이스
        """
        return self.optimize_memory()
    
    async def optimize_async(self) -> Dict[str, Any]:
        """
        비동기 메모리 최적화 (호환성 + DI 통합)
        """
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            # 별도 스레드에서 실행 (blocking 작업)
            result = await loop.run_in_executor(None, self.optimize_memory)
            self.logger.debug("✅ 비동기 메모리 최적화 완료")
            return result
        except Exception as e:
            self.logger.error(f"❌ 비동기 메모리 최적화 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "method": "async_fallback"
            }
    
    def get_memory_status(self) -> Dict[str, Any]:
        """
        메모리 상태 조회 (Step들에서 사용 + DI 통합)
        """
        try:
            stats = self.get_memory_stats()
            return {
                "total_optimizations": getattr(self, 'optimization_count', 0),
                "device": self.device,
                "is_m3_max": self.is_m3_max,
                "cpu_used_gb": stats.cpu_used_gb,
                "cpu_available_gb": stats.cpu_available_gb,
                "gpu_allocated_gb": stats.gpu_allocated_gb,
                "last_optimization": getattr(self, 'last_optimization_time', None),
                "available": True,
                "di_integrated": self._di_container is not None,
                "dependencies_resolved": self._dependencies_resolved
            }
        except Exception as e:
            return {
                "error": str(e),
                "available": False,
                "di_integrated": self._di_container is not None
            }
    
    def cleanup(self) -> bool:
        """
        메모리 매니저 정리 (Step들에서 사용 + DI 통합)
        """
        try:
            # 마지막 최적화 실행
            result = self.optimize_memory()
            
            # 통계 리셋
            if hasattr(self, 'optimization_count'):
                self.optimization_count = 0
            
            # DI를 통한 추가 정리
            if hasattr(self, 'device_manager') and self.device_manager:
                try:
                    self.device_manager.optimize_memory()
                except Exception:
                    pass
            
            self.logger.debug("✅ DI MemoryManager 정리 완료")
            return result.get('success', True)
            
        except Exception as e:
            self.logger.error(f"❌ DI MemoryManager 정리 실패: {e}")
            return False
    
    def cleanup_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """🧹 메모리 정리 (DI 최적화)"""
        try:
            start_time = time.time()
            
            # 1. Python 가비지 컬렉션
            collected_objects = 0
            for _ in range(3):
                collected = gc.collect()
                collected_objects += collected
            
            # 2. 캐시 정리
            cache_cleared = 0
            if hasattr(self, 'tensor_cache'):
                cache_cleared += len(self.tensor_cache)
                if aggressive:
                    self.clear_cache(aggressive=True)
                else:
                    self._evict_low_priority_cache()
            
            # 3. PyTorch 메모리 정리
            if TORCH_AVAILABLE:
                try:
                    if self.device == "mps" and torch.backends.mps.is_available():
                        # M3 Max MPS 메모리 정리
                        if hasattr(torch.mps, 'empty_cache'):
                            safe_mps_empty_cache()
                        
                        # MPS 동기화
                        if hasattr(torch.mps, 'synchronize'):
                            torch.mps.synchronize()
                        
                        # 공격적 정리 시 M3 Max 특화 정리
                        if aggressive and self.is_m3_max:
                            self._aggressive_m3_cleanup()
                        
                    elif self.device == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        if aggressive:
                            torch.cuda.synchronize()
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ GPU 메모리 정리 중 오류: {e}")
            
            # 4. DI를 통한 추가 정리
            if self._di_container:
                try:
                    if hasattr(self, 'device_manager') and self.device_manager:
                        self.device_manager.optimize_memory()
                except Exception:
                    pass  # DI 추가 정리 실패는 무시
            
            cleanup_time = time.time() - start_time
            
            # 결과 반환
            return {
                "success": True,
                "cleanup_time": cleanup_time,
                "collected_objects": collected_objects,
                "cache_cleared": cache_cleared,
                "aggressive": aggressive,
                "device": self.device,
                "m3_optimized": self.is_m3_max,
                "di_integrated": self._di_container is not None
            }
            
        except Exception as e:
            self.logger.error(f"❌ 메모리 정리 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "device": self.device
            }

    def optimize_memory(self) -> Dict[str, Any]:
        """
        🔥 메모리 최적화 (완전 동기화 + DI 통합)
        ✅ 모든 async/await 오류 해결
        ✅ VirtualFittingStep 호환성 유지
        ✅ DI를 통한 추가 최적화
        """
        try:
            start_time = time.time()
            optimization_results = []
            
            # 1. 메모리 정리 수행
            cleanup_result = self.cleanup_memory(aggressive=False)
            optimization_results.append(f"메모리 정리: {cleanup_result.get('success', False)}")
            
            # 2. M3 Max 특화 최적화
            if self.is_m3_max:
                m3_result = self._optimize_m3_max_memory_sync()
                optimization_results.append(f"M3 Max 최적화: {m3_result}")
            
            # 3. 캐시 최적화
            cache_stats = self._optimize_cache_system()
            optimization_results.append(f"캐시 최적화: {cache_stats}")
            
            # 4. DI를 통한 추가 최적화
            if self._di_container:
                try:
                    # DeviceManager를 통한 메모리 최적화
                    if hasattr(self, 'device_manager') and self.device_manager:
                        self.device_manager.optimize_memory()
                        optimization_results.append("DI DeviceManager 최적화 완료")
                    
                    # DataConverter를 통한 메모리 최적화 (있다면)
                    if hasattr(self, 'data_converter') and self.data_converter:
                        if hasattr(self.data_converter, 'optimize_memory'):
                            self.data_converter.optimize_memory()
                            optimization_results.append("DI DataConverter 최적화 완료")
                
                except Exception as e:
                    optimization_results.append(f"DI 추가 최적화 실패: {str(e)[:50]}")
            
            # 5. 메모리 압박 상태 확인
            pressure_info = self.check_memory_pressure()
            optimization_results.append(f"메모리 압박: {pressure_info.get('status', 'unknown')}")
            
            optimization_time = time.time() - start_time
            
            return {
                "success": True,
                "message": "메모리 최적화 완료 (DI 통합)",
                "optimization_time": optimization_time,
                "optimization_results": optimization_results,
                "device": self.device,
                "m3_max_optimized": self.is_m3_max,
                "di_integrated": self._di_container is not None,
                "dependencies_resolved": self._dependencies_resolved,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 메모리 최적화 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "device": self.device,
                "timestamp": time.time()
            }

    def _optimize_cache_system(self) -> str:
        """캐시 시스템 최적화"""
        try:
            # 캐시 크기 체크
            total_cache_size = 0
            cache_counts = {}
            
            for cache_name in ['tensor_cache', 'image_cache', 'model_cache']:
                if hasattr(self, cache_name):
                    cache = getattr(self, cache_name)
                    size = len(cache)
                    total_cache_size += size
                    cache_counts[cache_name] = size
            
            # 캐시가 너무 클 경우 정리
            if total_cache_size > 100:  # 캐시 항목이 100개 이상
                self._evict_low_priority_cache()
                return f"캐시 정리됨 (이전: {total_cache_size}개)"
            
            return f"정상 ({total_cache_size}개 항목)"
            
        except Exception as e:
            self.logger.warning(f"⚠️ 캐시 최적화 실패: {e}")
            return "최적화 실패"
    
    def _aggressive_m3_cleanup(self):
        """공격적 M3 Max 메모리 정리"""
        try:
            # 모든 캐시 클리어
            for cache_name in ['tensor_cache', 'image_cache', 'model_cache']:
                if hasattr(self, cache_name):
                    getattr(self, cache_name).clear()
            
            # 반복 가비지 컬렉션
            for _ in range(5):
                gc.collect()
            
            # PyTorch MPS 캐시 정리
            if TORCH_AVAILABLE and hasattr(torch.mps, 'empty_cache'):
                safe_mps_empty_cache()
            
            self.logger.debug("🍎 공격적 M3 Max 메모리 정리 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 공격적 메모리 정리 실패: {e}")

    def get_memory_stats(self) -> MemoryStats:
        """현재 메모리 상태 조회 (DI 통합)"""
        try:
            # CPU 메모리
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()
                cpu_percent = memory.percent
                cpu_total_gb = memory.total / 1024**3
                cpu_used_gb = memory.used / 1024**3
                cpu_available_gb = memory.available / 1024**3
                swap_used_gb = psutil.swap_memory().used / 1024**3
                
                # 프로세스 메모리
                try:
                    process = psutil.Process()
                    process_memory_mb = process.memory_info().rss / 1024**2
                except:
                    process_memory_mb = 0.0
            else:
                cpu_percent = 50.0
                cpu_total_gb = self.memory_gb
                cpu_used_gb = self.memory_gb * 0.5
                cpu_available_gb = self.memory_gb * 0.5
                swap_used_gb = 0.0
                process_memory_mb = 0.0
            
            # GPU 메모리
            gpu_allocated_gb = 0.0
            gpu_reserved_gb = 0.0
            gpu_total_gb = 0.0
            
            if TORCH_AVAILABLE:
                try:
                    if self.device == "cuda" and torch.cuda.is_available():
                        gpu_allocated_gb = torch.cuda.memory_allocated() / 1024**3
                        gpu_reserved_gb = torch.cuda.memory_reserved() / 1024**3
                        gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    elif self.device == "mps" and torch.backends.mps.is_available():
                        # MPS 메모리 정보 (추정)
                        gpu_allocated_gb = 2.0
                        gpu_total_gb = self.memory_gb  # M3 Max 통합 메모리
                except Exception:
                    pass
            
            # 캐시 크기
            cache_size_mb = 0.0
            try:
                for cache_name in ['tensor_cache', 'image_cache', 'model_cache']:
                    if hasattr(self, cache_name):
                        cache = getattr(self, cache_name)
                        cache_size_mb += len(str(cache)) / 1024**2
            except:
                pass
            
            # M3 최적화 정보 (DI 통합)
            m3_optimizations = {}
            if self.is_m3_max:
                m3_optimizations = {
                    "memory_pools": getattr(self, 'memory_pools', {}),
                    "batch_sizes": getattr(self, 'optimal_batch_sizes', {}),
                    "precision_mode": self.precision_mode,
                    "conda_optimized": SYSTEM_INFO["in_conda"],
                    "di_integrated": self._di_container is not None,
                    "dependencies_resolved": self._dependencies_resolved
                }
            
            return MemoryStats(
                cpu_percent=cpu_percent,
                cpu_available_gb=cpu_available_gb,
                cpu_used_gb=cpu_used_gb,
                cpu_total_gb=cpu_total_gb,
                gpu_allocated_gb=gpu_allocated_gb,
                gpu_reserved_gb=gpu_reserved_gb,
                gpu_total_gb=gpu_total_gb,
                swap_used_gb=swap_used_gb,
                cache_size_mb=cache_size_mb,
                process_memory_mb=process_memory_mb,
                m3_optimizations=m3_optimizations
            )
            
        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 상태 조회 실패: {e}")
            return MemoryStats(
                cpu_percent=0.0,
                cpu_available_gb=8.0,
                cpu_used_gb=8.0,
                cpu_total_gb=16.0
            )

    def check_memory_pressure(self) -> Dict[str, Any]:
        """메모리 압박 상태 체크 (DI 통합)"""
        try:
            stats = self.get_memory_stats()
            
            cpu_usage_ratio = stats.cpu_used_gb / max(1.0, stats.cpu_total_gb)
            gpu_usage_ratio = stats.gpu_allocated_gb / max(1.0, stats.gpu_total_gb)
            
            status = "normal"
            if cpu_usage_ratio > 0.9 or gpu_usage_ratio > 0.9:
                status = "critical"
            elif cpu_usage_ratio > 0.75 or gpu_usage_ratio > 0.75:
                status = "warning"
            
            return {
                "status": status,
                "cpu_usage_ratio": cpu_usage_ratio,
                "gpu_usage_ratio": gpu_usage_ratio,
                "cache_size_mb": stats.cache_size_mb,
                "recommendations": self._get_cleanup_recommendations(stats),
                "di_integrated": self._di_container is not None,
                "dependencies_available": {
                    "device_manager": hasattr(self, 'device_manager'),
                    "data_converter": hasattr(self, 'data_converter')
                }
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 압박 체크 실패: {e}")
            return {"status": "unknown", "error": str(e)}

    def _get_cleanup_recommendations(self, stats: MemoryStats) -> List[str]:
        """정리 권장사항 (DI 통합)"""
        recommendations = []
        
        try:
            cpu_ratio = stats.cpu_used_gb / max(1.0, stats.cpu_total_gb)
            
            if cpu_ratio > 0.8:
                recommendations.append("CPU 메모리 정리 권장")
            
            if stats.gpu_allocated_gb > 10.0:
                recommendations.append("GPU 메모리 정리 권장")
            
            if stats.cache_size_mb > 1000:
                recommendations.append("캐시 정리 권장")
            
            if self.is_m3_max and cpu_ratio > 0.7:
                recommendations.append("M3 Max 최적화 재실행 권장")
            
            # DI 통합 권장사항
            if self._di_container:
                if hasattr(self, 'device_manager'):
                    recommendations.append("DI DeviceManager 메모리 최적화 활용 가능")
                if hasattr(self, 'data_converter'):
                    recommendations.append("DI DataConverter 메모리 최적화 활용 가능")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 권장사항 생성 실패: {e}")
        
        return recommendations

    def clear_cache(self, aggressive: bool = False):
        """캐시 정리 (DI 최적화)"""
        try:
            with self._lock:
                if aggressive:
                    # 전체 캐시 삭제
                    for cache_name in ['tensor_cache', 'image_cache', 'model_cache', 'cache_priority']:
                        if hasattr(self, cache_name):
                            getattr(self, cache_name).clear()
                    
                    # DI를 통한 추가 캐시 정리
                    if hasattr(self, 'data_converter') and self.data_converter:
                        if hasattr(self.data_converter, 'clear_cache'):
                            self.data_converter.clear_cache(aggressive=True)
                    
                    self.logger.debug("🧹 전체 캐시 정리 완료 (DI 통합)")
                else:
                    # 선택적 캐시 정리
                    self._evict_low_priority_cache()
                    self.logger.debug("🧹 선택적 캐시 정리 완료")
        except Exception as e:
            self.logger.warning(f"⚠️ 캐시 정리 실패: {e}")

    def _evict_low_priority_cache(self):
        """낮은 우선순위 캐시 제거"""
        try:
            if not hasattr(self, 'cache_priority') or not self.cache_priority:
                return
            
            # 우선순위 기준 정렬
            sorted_items = sorted(self.cache_priority.items(), key=lambda x: x[1])
            
            # 하위 20% 제거
            num_to_remove = max(1, len(sorted_items) // 5)
            for key, _ in sorted_items[:num_to_remove]:
                for cache_name in ['tensor_cache', 'image_cache', 'model_cache']:
                    if hasattr(self, cache_name):
                        getattr(self, cache_name).pop(key, None)
                self.cache_priority.pop(key, None)
            
            self.logger.debug(f"🗑️ 낮은 우선순위 캐시 {num_to_remove}개 제거")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 캐시 제거 실패: {e}")

    # ============================================
    # 🔥 비동기 인터페이스 (안전한 래퍼 + DI 통합)
    # ============================================

    async def initialize(self) -> bool:
        """메모리 관리자 비동기 초기화 (DI 통합)"""
        try:
            # M3 Max 최적화 설정 (동기로 실행)
            if self.is_m3_max and self.optimization_enabled:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._optimize_for_m3_max)
            
            # DI를 통한 의존성 해결
            if self._di_container:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.resolve_dependencies)
            
            self.logger.debug(f"✅ DI MemoryManager 비동기 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ DI MemoryManager 비동기 초기화 실패: {e}")
            return False

    async def cleanup(self):
        """비동기 메모리 정리 (DI 통합)"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.cleanup_memory, True)
        except Exception as e:
            self.logger.warning(f"⚠️ 비동기 메모리 정리 실패: {e}")

    def get_usage(self) -> Dict[str, Any]:
        """동기 사용량 조회 (하위 호환 + DI 통합)"""
        try:
            stats = self.get_memory_stats()
            return {
                "cpu_percent": stats.cpu_percent,
                "cpu_used_gb": stats.cpu_used_gb,
                "cpu_total_gb": stats.cpu_total_gb,
                "gpu_allocated_gb": stats.gpu_allocated_gb,
                "gpu_total_gb": stats.gpu_total_gb,
                "cache_size_mb": stats.cache_size_mb,
                "device": self.device,
                "is_m3_max": self.is_m3_max,
                "di_integrated": self._di_container is not None,
                "dependencies_resolved": self._dependencies_resolved
            }
        except Exception as e:
            self.logger.warning(f"⚠️ 사용량 조회 실패: {e}")
            return {"error": str(e)}

    def __del__(self):
        """소멸자"""
        try:
            if hasattr(self, 'monitoring_active'):
                self.monitoring_active = False
            if hasattr(self, 'tensor_cache'):
                self.clear_cache(aggressive=True)
        except:
            pass

# ==============================================
# 🔥 DI 통합 MemoryManagerAdapter 클래스
# ==============================================

class MemoryManagerAdapter(IDependencyInjectable):
    """
    🔥 DI 완전 통합 MemoryManagerAdapter 클래스
    ✅ CircularReferenceFreeDIContainer 연동
    ✅ 의존성 주입 지원
    ✅ optimize_memory 메서드 완전 구현
    ✅ VirtualFittingStep 호환성 100%
    ✅ 모든 async/await 오류 해결
    """
    
    def __init__(self, base_manager: Optional[MemoryManager] = None, device: str = "auto", di_container: Optional[Any] = None, **kwargs):
        """MemoryManagerAdapter 초기화 (DI 지원)"""
        try:
            # 베이스 매니저 설정
            if base_manager is None:
                self._base_manager = MemoryManager(device=device, di_container=di_container, **kwargs)
            else:
                self._base_manager = base_manager
                # 기존 베이스 매니저에 DI Container 설정
                if di_container and hasattr(self._base_manager, 'set_di_container'):
                    self._base_manager.set_di_container(di_container)
            
            # 속성 초기화
            self.device = self._base_manager.device
            self.is_m3_max = self._base_manager.is_m3_max
            self.memory_gb = self._base_manager.memory_gb
            self.logger = logging.getLogger("MemoryManagerAdapter")
            
            # 🔥 DI Container 설정
            self._di_container: Optional[Any] = di_container
            self._dependencies_resolved = False
            self._dependency_status = {
                'di_container': di_container is not None,
                'base_manager': True,
                'initialized': False
            }
            
            # 스레드 안전성
            self._lock = threading.RLock()
            
            # 어댑터 고유 속성
            self.adapter_initialized = True
            self.optimization_cache = {}
            self.last_optimization_time = 0
            
            # DI Container 설정 (초기화 후)
            if di_container is not None:
                self.set_di_container(di_container)
            
            self.logger.debug(f"✅ DI MemoryManagerAdapter 초기화 완료 - 디바이스: {self.device}")
            
        except Exception as e:
            self.logger.error(f"❌ DI MemoryManagerAdapter 초기화 실패: {e}")
            # 최소한의 초기화
            self._base_manager = MemoryManager(device="cpu")
            self.device = "cpu"
            self.is_m3_max = False
            self.memory_gb = 16
            self.logger = logging.getLogger("MemoryManagerAdapter")

    # ==============================================
    # 🔥 DI 인터페이스 구현
    # ==============================================

    def set_di_container(self, di_container: Any) -> None:
        """DI Container 설정"""
        try:
            with self._lock:
                self._di_container = di_container
                self._dependency_status['di_container'] = True
                
                # 베이스 매니저에도 DI Container 설정
                if hasattr(self._base_manager, 'set_di_container'):
                    self._base_manager.set_di_container(di_container)
                
                # DI Container에 자신을 등록
                if hasattr(di_container, 'register'):
                    di_container.register('memory_adapter', self, singleton=True)
                    di_container.register('IMemoryManagerAdapter', self, singleton=True)
                
                self.logger.info("✅ MemoryManagerAdapter DI Container 설정 완료")
                
                # 의존성 해결 시도
                self.resolve_dependencies()
                
        except Exception as e:
            self.logger.error(f"❌ MemoryManagerAdapter DI Container 설정 실패: {e}")

    def resolve_dependencies(self) -> bool:
        """의존성 해결"""
        try:
            with self._lock:
                if not self._di_container:
                    self.logger.warning("⚠️ DI Container가 설정되지 않음")
                    return False
                
                resolved_count = 0
                
                # 베이스 매니저의 의존성 해결
                if hasattr(self._base_manager, 'resolve_dependencies'):
                    try:
                        if self._base_manager.resolve_dependencies():
                            resolved_count += 1
                            self.logger.debug("✅ MemoryManagerAdapter 베이스 매니저 의존성 해결")
                    except Exception as e:
                        self.logger.debug(f"베이스 매니저 의존성 해결 실패: {e}")
                
                self._dependencies_resolved = resolved_count >= 0  # 어댑터는 필수 의존성 없음
                self._dependency_status['initialized'] = True
                
                self.logger.info(f"🔗 MemoryManagerAdapter 의존성 해결 완료: {resolved_count}개")
                
                return self._dependencies_resolved
                
        except Exception as e:
            self.logger.error(f"❌ MemoryManagerAdapter 의존성 해결 실패: {e}")
            return False

    def get_dependency_status(self) -> Dict[str, Any]:
        """의존성 상태 조회"""
        with self._lock:
            base_status = {}
            if hasattr(self._base_manager, 'get_dependency_status'):
                try:
                    base_status = self._base_manager.get_dependency_status()
                except Exception:
                    pass
            
            return {
                'class_name': self.__class__.__name__,
                'dependencies_resolved': self._dependencies_resolved,
                'dependency_status': dict(self._dependency_status),
                'di_container_available': self._di_container is not None,
                'base_manager_status': base_status,
                'adapter_info': {
                    'device': self.device,
                    'is_m3_max': self.is_m3_max,
                    'memory_gb': self.memory_gb,
                    'adapter_initialized': self.adapter_initialized
                }
            }

    # ==============================================
    # 🔥 핵심 메서드들 (DI 최적화)
    # ==============================================

    def optimize_startup(self, aggressive: bool = False) -> Dict[str, Any]:
        """
        🚀 MemoryManagerAdapter용 optimize_startup (완전 동기화 + DI 통합)
        ✅ VirtualFittingStep 호환성 보장
        ✅ 모든 async/await 오류 해결
        ✅ DI를 통한 추가 최적화
        """
        try:
            # 기본 매니저의 optimize_startup 시도
            if hasattr(self._base_manager, 'optimize_startup'):
                try:
                    result = self._base_manager.optimize_startup(aggressive)
                    result["adapter"] = True
                    result["di_integrated"] = self._di_container is not None
                    return result
                except Exception as e:
                    self.logger.warning(f"⚠️ 기본 매니저 optimize_startup 실패: {e}")
            
            # 폴백: optimize_memory 사용
            if hasattr(self, 'optimize_memory'):
                try:
                    result = self.optimize_memory(aggressive)
                    result.update({
                        "adapter": True,
                        "fallback_mode": "optimize_memory",
                        "message": "startup 최적화를 optimize_memory로 대체",
                        "di_integrated": self._di_container is not None
                    })
                    return result
                except Exception as e:
                    self.logger.warning(f"⚠️ 폴백 optimize_memory 실패: {e}")
            
            # 최종 폴백: 기본 시작 최적화
            return self._basic_startup_optimization(aggressive)
            
        except Exception as e:
            self.logger.error(f"❌ MemoryManagerAdapter optimize_startup 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "adapter": True,
                "di_integrated": self._di_container is not None,
                "timestamp": time.time()
            }
    
    def _basic_startup_optimization(self, aggressive: bool = False) -> Dict[str, Any]:
        """기본 시작 최적화 (최종 폴백 + DI 통합)"""
        try:
            startup_results = []
            
            # 기본 가비지 컬렉션
            collected = gc.collect()
            startup_results.append(f"기본 GC: {collected}개 객체")
            
            # PyTorch 메모리 정리 (가능한 경우)
            if TORCH_AVAILABLE:
                try:
                    if hasattr(torch.mps, 'empty_cache') and torch.backends.mps.is_available():
                        safe_mps_empty_cache()
                        startup_results.append("MPS 캐시 정리")
                    elif hasattr(torch.cuda, 'empty_cache') and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        startup_results.append("CUDA 캐시 정리")
                except Exception as e:
                    startup_results.append(f"GPU 캐시 정리 실패: {e}")
            
            # DI를 통한 추가 최적화
            if self._di_container:
                try:
                    # 다른 의존성들을 통한 최적화 시도
                    device_manager = self._di_container.get('device_manager')
                    if device_manager and hasattr(device_manager, 'optimize_memory'):
                        device_manager.optimize_memory()
                        startup_results.append("DI DeviceManager 최적화")
                except Exception as e:
                    startup_results.append(f"DI 최적화 실패: {str(e)[:50]}")
            
            return {
                "success": True,
                "message": "기본 시작 최적화 완료 (DI 통합)",
                "startup_results": startup_results,
                "adapter": True,
                "fallback": True,
                "di_integrated": self._di_container is not None,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "adapter": True,
                "fallback": True,
                "di_integrated": self._di_container is not None,
                "timestamp": time.time()
            }

    def optimize_memory(self, aggressive: bool = False, **kwargs) -> Dict[str, Any]:
        """
        🔥 VirtualFittingStep에서 필요한 핵심 optimize_memory 메서드 (DI 통합)
        ✅ 완전 동기화로 모든 async/await 오류 해결
        ✅ AttributeError 완전 해결
        ✅ DI를 통한 추가 최적화
        """
        try:
            start_time = time.time()
            optimization_results = []
            
            # 중복 최적화 방지 (5초 내 재호출 방지)
            if (start_time - self.last_optimization_time) < 5.0:
                return {
                    "success": True,
                    "message": "최근 최적화 완료 (캐시됨)",
                    "cached": True,
                    "device": self.device,
                    "di_integrated": self._di_container is not None,
                    "timestamp": start_time
                }
            
            # 1. 기본 메모리 관리자의 optimize_memory 호출 시도
            if hasattr(self._base_manager, 'optimize_memory'):
                try:
                    base_result = self._base_manager.optimize_memory()
                    optimization_results.append("기본 메모리 최적화 완료")
                    
                    # 성공한 경우 바로 반환
                    if base_result.get("success", False):
                        self.last_optimization_time = start_time
                        return {
                            **base_result,
                            "adapter": True,
                            "optimization_results": optimization_results,
                            "device": self.device,
                            "di_integrated": self._di_container is not None,
                            "timestamp": start_time
                        }
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ 기본 최적화 실패, 폴백 모드 사용: {e}")
            
            # 2. 폴백: cleanup_memory 기반 최적화
            cleanup_result = self._base_manager.cleanup_memory(aggressive=aggressive)
            optimization_results.append(f"메모리 정리: {cleanup_result.get('success', False)}")
            
            # 3. 어댑터 특화 최적화 (DI 통합)
            adapter_optimizations = self._run_adapter_optimizations(aggressive)
            optimization_results.extend(adapter_optimizations)
            
            # 4. M3 Max 특화 최적화 (동기)
            if self.is_m3_max:
                m3_optimizations = self._run_m3_max_optimizations_sync()
                optimization_results.extend(m3_optimizations)
            
            # 5. DI를 통한 추가 최적화
            if self._di_container:
                di_optimizations = self._run_di_optimizations()
                optimization_results.extend(di_optimizations)
            
            # 6. 최종 메모리 상태 확인
            final_stats = self._base_manager.get_memory_stats()
            
            optimization_time = time.time() - start_time
            self.last_optimization_time = start_time
            
            # 최적화 결과 캐싱
            result = {
                "success": True,
                "message": "MemoryManagerAdapter 메모리 최적화 완료 (DI 통합)",
                "optimization_time": optimization_time,
                "optimization_results": optimization_results,
                "cleanup_result": cleanup_result,
                "final_memory_stats": {
                    "cpu_used_gb": final_stats.cpu_used_gb,
                    "cpu_available_gb": final_stats.cpu_available_gb,
                    "gpu_allocated_gb": final_stats.gpu_allocated_gb,
                    "cache_size_mb": final_stats.cache_size_mb
                },
                "device": self.device,
                "is_m3_max": self.is_m3_max,
                "adapter": True,
                "aggressive": aggressive,
                "di_integrated": self._di_container is not None,
                "dependencies_resolved": self._dependencies_resolved,
                "timestamp": start_time
            }
            
            self.optimization_cache = result
            self.logger.debug("✅ DI MemoryManagerAdapter 메모리 최적화 완료")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ DI MemoryManagerAdapter 최적화 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "adapter": True,
                "device": self.device,
                "di_integrated": self._di_container is not None,
                "timestamp": time.time()
            }

    def _run_di_optimizations(self) -> List[str]:
        """DI를 통한 추가 최적화"""
        optimizations = []
        
        try:
            if not self._di_container:
                return optimizations
            
            # DeviceManager를 통한 최적화
            try:
                device_manager = self._di_container.get('device_manager')
                if device_manager and hasattr(device_manager, 'optimize_memory'):
                    device_manager.optimize_memory()
                    optimizations.append("DI DeviceManager 최적화")
            except Exception as e:
                optimizations.append(f"DI DeviceManager 최적화 실패: {str(e)[:30]}")
            
            # DataConverter를 통한 최적화
            try:
                data_converter = self._di_container.get('data_converter')
                if data_converter and hasattr(data_converter, 'optimize_memory'):
                    data_converter.optimize_memory()
                    optimizations.append("DI DataConverter 최적화")
            except Exception as e:
                optimizations.append(f"DI DataConverter 최적화 실패: {str(e)[:30]}")
            
            return optimizations
            
        except Exception as e:
            self.logger.warning(f"⚠️ DI 최적화 실패: {e}")
            return ["DI 최적화 실패"]

    def optimize(self, aggressive: bool = False) -> Dict[str, Any]:
        """
        메모리 최적화 (optimize_memory의 별칭) - MemoryManagerAdapter용
        """
        return self.optimize_memory(aggressive=aggressive)
    
    async def optimize_async(self, aggressive: bool = False) -> Dict[str, Any]:
        """
        비동기 메모리 최적화 - MemoryManagerAdapter용 (DI 통합)
        """
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.optimize_memory, aggressive)
            return result
        except Exception as e:
            self.logger.error(f"❌ MemoryManagerAdapter 비동기 최적화 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "adapter": True,
                "di_integrated": self._di_container is not None
            }
    
    def get_memory_status(self) -> Dict[str, Any]:
        """
        메모리 상태 조회 - MemoryManagerAdapter용 (DI 통합)
        """
        try:
            base_status = self._base_manager.get_memory_status()
            base_status.update({
                "adapter": True,
                "adapter_type": "MemoryManagerAdapter",
                "base_manager_type": type(self._base_manager).__name__,
                "di_integrated": self._di_container is not None,
                "dependencies_resolved": self._dependencies_resolved
            })
            return base_status
        except Exception as e:
            return {
                "error": str(e),
                "adapter": True,
                "available": False,
                "di_integrated": self._di_container is not None
            }
    
    def cleanup(self) -> bool:
        """
        메모리 매니저 정리 - MemoryManagerAdapter용 (DI 통합)
        """
        try:
            # 기본 매니저 정리
            result = self._base_manager.cleanup()
            
            # 어댑터 캐시 정리
            if hasattr(self, 'optimization_cache'):
                self.optimization_cache.clear()
            
            # DI를 통한 추가 정리
            if self._di_container:
                try:
                    device_manager = self._di_container.get('device_manager')
                    if device_manager and hasattr(device_manager, 'optimize_memory'):
                        device_manager.optimize_memory()
                except Exception:
                    pass
            
            self.logger.debug("✅ DI MemoryManagerAdapter 정리 완료")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ DI MemoryManagerAdapter 정리 실패: {e}")
            return False

    def _run_adapter_optimizations(self, aggressive: bool = False) -> List[str]:
        """어댑터 특화 최적화 (동기 + DI 통합)"""
        optimizations = []
        
        try:
            # 1. 캐시 정리
            if hasattr(self._base_manager, 'clear_cache'):
                self._base_manager.clear_cache(aggressive=aggressive)
                optimizations.append("어댑터 캐시 정리")
            
            # 2. 가비지 컬렉션
            collected = gc.collect()
            optimizations.append(f"가비지 컬렉션: {collected}개 객체")
            
            # 3. PyTorch 메모리 정리
            if TORCH_AVAILABLE:
                try:
                    if self.device == "mps" and torch.backends.mps.is_available():
                        if hasattr(torch.mps, 'empty_cache'):
                            safe_mps_empty_cache()
                            optimizations.append("MPS 캐시 정리")
                    elif self.device == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        optimizations.append("CUDA 캐시 정리")
                except Exception as e:
                    optimizations.append(f"GPU 캐시 정리 실패: {str(e)[:50]}")
            
            return optimizations
            
        except Exception as e:
            self.logger.warning(f"⚠️ 어댑터 최적화 실패: {e}")
            return ["어댑터 최적화 실패"]

    def _run_m3_max_optimizations_sync(self) -> List[str]:
        """M3 Max 특화 최적화 (동기 버전 + DI 통합)"""
        optimizations = []
        
        try:
            if not self.is_m3_max:
                return optimizations
            
            # M3 Max 특화 로직 (동기)
            if hasattr(self._base_manager, '_optimize_for_m3_max'):
                self._base_manager._optimize_for_m3_max()
                optimizations.append("M3 Max Neural Engine 최적화")
            
            # MPS 특화 정리
            if TORCH_AVAILABLE and torch.backends.mps.is_available():
                if hasattr(torch.mps, 'empty_cache'):
                    safe_mps_empty_cache()
                optimizations.append("M3 Max MPS 캐시 정리")
            
            # 메모리 압박 완화
            if hasattr(self._base_manager, '_aggressive_m3_cleanup'):
                self._base_manager._aggressive_m3_cleanup()
                optimizations.append("M3 Max 공격적 메모리 정리")
            
            return optimizations
            
        except Exception as e:
            self.logger.warning(f"⚠️ M3 Max 최적화 실패: {e}")
            return ["M3 Max 최적화 실패"]

    # ============================================
    # 🔥 위임 메서드들 (모든 필요한 메서드 위임 + DI 통합)
    # ============================================

    def get_memory_stats(self) -> MemoryStats:
        """메모리 통계 (위임)"""
        try:
            return self._base_manager.get_memory_stats()
        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 통계 조회 실패: {e}")
            return MemoryStats(
                cpu_percent=50.0,
                cpu_available_gb=8.0,
                cpu_used_gb=8.0,
                cpu_total_gb=16.0
            )
    
    def cleanup_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """메모리 정리 (위임)"""
        try:
            result = self._base_manager.cleanup_memory(aggressive)
            result["adapter"] = True
            result["di_integrated"] = self._di_container is not None
            return result
        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 정리 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "adapter": True,
                "device": self.device,
                "di_integrated": self._di_container is not None
            }
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """메모리 압박 확인 (위임)"""
        try:
            result = self._base_manager.check_memory_pressure()
            result["adapter"] = True
            result["di_integrated"] = self._di_container is not None
            return result
        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 압박 확인 실패: {e}")
            return {
                "status": "unknown",
                "error": str(e),
                "adapter": True,
                "di_integrated": self._di_container is not None
            }
    
    def clear_cache(self, aggressive: bool = False):
        """캐시 정리 (위임)"""
        try:
            return self._base_manager.clear_cache(aggressive)
        except Exception as e:
            self.logger.warning(f"⚠️ 캐시 정리 실패: {e}")

    def get_usage(self) -> Dict[str, Any]:
        """동기 사용량 조회 (위임)"""
        try:
            result = self._base_manager.get_usage()
            result["adapter"] = True
            result["di_integrated"] = self._di_container is not None
            return result
        except Exception as e:
            self.logger.warning(f"⚠️ 사용량 조회 실패: {e}")
            return {
                "error": str(e), 
                "adapter": True,
                "di_integrated": self._di_container is not None
            }

    def __getattr__(self, name):
        """다른 모든 속성/메서드는 기본 관리자로 위임"""
        try:
            return getattr(self._base_manager, name)
        except AttributeError:
            self.logger.warning(f"⚠️ 속성 '{name}'을 찾을 수 없음")
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __del__(self):
        """소멸자"""
        try:
            if hasattr(self, '_base_manager') and self._base_manager:
                if hasattr(self._base_manager, '__del__'):
                    self._base_manager.__del__()
        except:
            pass

# ==============================================
# 🔥 DI 통합 GPUMemoryManager 클래스
# ==============================================

class GPUMemoryManager(MemoryManager):
    """
    🔥 DI 통합 GPU 메모리 관리자 (기존 클래스명 유지)
    ✅ 현재 구조와 완벽 호환
    ✅ 기존 코드의 GPUMemoryManager 사용 유지
    ✅ main.py import 오류 완전 해결
    ✅ DI 지원 추가
    """
    
    def __init__(self, device=None, memory_limit_gb=None, di_container: Optional[Any] = None, **kwargs):
        """GPU 메모리 관리자 초기화 (기존 시그니처 유지 + DI 지원)"""
        
        # 기본값 설정
        if device is None:
            device = SYSTEM_INFO["device"]
        if memory_limit_gb is None:
            memory_limit_gb = SYSTEM_INFO["memory_gb"] * 0.8 if SYSTEM_INFO["is_m3_max"] else 16.0
        
        super().__init__(device=device, memory_limit_gb=memory_limit_gb, di_container=di_container, **kwargs)
        self.logger = logging.getLogger("GPUMemoryManager")
        
        # 기존 속성 호환성 유지
        self.memory_limit_gb = memory_limit_gb
        
        self.logger.debug(f"🎮 DI GPUMemoryManager 초기화 - {device} ({memory_limit_gb:.1f}GB)")

    def clear_cache(self):
        """메모리 정리 (기존 메서드 시그니처 유지 + DI 최적화)"""
        try:
            # 부모 클래스의 메모리 정리 호출
            result = self.cleanup_memory(aggressive=False)
            
            # PyTorch GPU 캐시 정리
            if TORCH_AVAILABLE:
                if self.device == "mps" and torch.backends.mps.is_available():
                    if hasattr(torch.mps, 'empty_cache'):
                        safe_mps_empty_cache()
                elif self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # DI를 통한 추가 정리
            if hasattr(self, 'device_manager') and self.device_manager:
                try:
                    self.device_manager.optimize_memory()
                except Exception:
                    pass
            
            self.logger.debug("🧹 DI GPU 캐시 정리 완료")
            return result
            
        except Exception as e:
            self.logger.warning(f"⚠️ GPU 캐시 정리 실패: {e}")
            return {"success": False, "error": str(e)}

    def check_memory_usage(self):
        """메모리 사용량 확인 (기존 메서드명 유지 + DI 통합)"""
        try:
            stats = self.get_memory_stats()
            
            # 기존 로직과 호환
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()
                used_gb = memory.used / (1024**3)
                if used_gb > self.memory_limit_gb * 0.9:
                    self.logger.warning(f"⚠️ 메모리 사용량 높음: {used_gb:.1f}GB")
                    self.clear_cache()
            
            return {
                "cpu_used_gb": stats.cpu_used_gb,
                "cpu_total_gb": stats.cpu_total_gb,
                "gpu_allocated_gb": stats.gpu_allocated_gb,
                "device": self.device,
                "is_m3_max": self.is_m3_max,
                "memory_limit_gb": self.memory_limit_gb,
                "di_integrated": self._di_container is not None,
                "dependencies_resolved": self._dependencies_resolved
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 사용량 확인 실패: {e}")
            return {"error": str(e)}

# ==============================================
# 🔥 DI 전용 팩토리 함수들
# ==============================================

def create_di_memory_manager(
    device: str = "auto",
    di_container: Optional[Any] = None,
    **kwargs
) -> MemoryManager:
    """DI 지원 메모리 관리자 생성"""
    if device == "auto":
        device = SYSTEM_INFO["device"]
    
    # DI Container 자동 해결
    if di_container is None:
        try:
            from ..core.di_container import get_global_container
            di_container = get_global_container()
        except ImportError:
            logger.warning("⚠️ DI Container를 찾을 수 없음, 기본 모드로 실행")
    
    return MemoryManager(device=device, di_container=di_container, **kwargs)

def create_di_device_manager(
    di_container: Optional[Any] = None
) -> DeviceManager:
    """DI 지원 디바이스 관리자 생성"""
    # DI Container 자동 해결
    if di_container is None:
        try:
            from ..core.di_container import get_global_container
            di_container = get_global_container()
        except ImportError:
            logger.warning("⚠️ DI Container를 찾을 수 없음, 기본 모드로 실행")
    
    return DeviceManager(di_container=di_container)

def create_di_memory_adapter(
    device: str = "auto",
    di_container: Optional[Any] = None,
    **kwargs
) -> MemoryManagerAdapter:
    """DI 지원 메모리 어댑터 생성"""
    # DI Container 자동 해결
    if di_container is None:
        try:
            from ..core.di_container import get_global_container
            di_container = get_global_container()
        except ImportError:
            logger.warning("⚠️ DI Container를 찾을 수 없음, 기본 모드로 실행")
    
    base_manager = create_di_memory_manager(device=device, di_container=di_container, **kwargs)
    return MemoryManagerAdapter(base_manager=base_manager, di_container=di_container)

# ==============================================
# 🔥 전역 DI 인스턴스 관리
# ==============================================

# 전역 DI 지원 인스턴스들
_global_di_memory_manager: Optional[MemoryManager] = None
_global_di_device_manager: Optional[DeviceManager] = None
_global_di_memory_adapter: Optional[MemoryManagerAdapter] = None
_global_di_container_ref: Optional[Any] = None
_di_memory_lock = threading.RLock()

def get_global_di_memory_manager(di_container: Optional[Any] = None, **kwargs) -> MemoryManager:
    """전역 DI 메모리 관리자 반환"""
    global _global_di_memory_manager, _global_di_container_ref
    
    with _di_memory_lock:
        # DI Container 변경 감지
        if di_container is not None and di_container != _global_di_container_ref:
            _global_di_memory_manager = None
            _global_di_container_ref = di_container
        
        if _global_di_memory_manager is None:
            _global_di_memory_manager = create_di_memory_manager(di_container=di_container, **kwargs)
            _global_di_container_ref = di_container
    
    return _global_di_memory_manager

def get_global_di_device_manager(di_container: Optional[Any] = None) -> DeviceManager:
    """전역 DI 디바이스 관리자 반환"""
    global _global_di_device_manager, _global_di_container_ref
    
    with _di_memory_lock:
        # DI Container 변경 감지
        if di_container is not None and di_container != _global_di_container_ref:
            _global_di_device_manager = None
            _global_di_container_ref = di_container
        
        if _global_di_device_manager is None:
            _global_di_device_manager = create_di_device_manager(di_container=di_container)
            _global_di_container_ref = di_container
    
    return _global_di_device_manager

def get_global_di_memory_adapter(di_container: Optional[Any] = None, **kwargs) -> MemoryManagerAdapter:
    """전역 DI 메모리 어댑터 반환"""
    global _global_di_memory_adapter, _global_di_container_ref
    
    with _di_memory_lock:
        # DI Container 변경 감지
        if di_container is not None and di_container != _global_di_container_ref:
            _global_di_memory_adapter = None
            _global_di_container_ref = di_container
        
        if _global_di_memory_adapter is None:
            _global_di_memory_adapter = create_di_memory_adapter(di_container=di_container, **kwargs)
            _global_di_container_ref = di_container
    
    return _global_di_memory_adapter

# ==============================================
# 🔥 기존 함수들 (DI 지원 추가 + 하위 호환성)
# ==============================================

# 전역 메모리 관리자 인스턴스 (싱글톤)
_global_memory_manager = None
_global_gpu_memory_manager = None
_global_adapter = None
_global_device_manager = None
_manager_lock = threading.Lock()

def get_memory_manager(**kwargs) -> MemoryManager:
    """전역 메모리 관리자 인스턴스 반환 (DI 자동 적용)"""
    global _global_memory_manager
    
    with _manager_lock:
        if _global_memory_manager is None:
            # DI 지원 버전으로 생성
            try:
                from ..core.di_container import get_global_container
                di_container = get_global_container()
                _global_memory_manager = MemoryManager(di_container=di_container, **kwargs)
            except ImportError:
                # DI Container 없이도 동작
                _global_memory_manager = MemoryManager(**kwargs)
        return _global_memory_manager

def get_global_memory_manager(**kwargs) -> MemoryManager:
    """전역 메모리 관리자 인스턴스 반환 (별칭 + DI 자동 적용)"""
    return get_memory_manager(**kwargs)

def get_device_manager(**kwargs) -> DeviceManager:
    """
    🔥 DeviceManager 인스턴스 반환 (main.py에서 요구하는 핵심 함수 + DI 지원)
    ✅ import 오류 완전 해결
    ✅ setup_mps_compatibility 메서드 포함
    ✅ DI 자동 적용
    """
    global _global_device_manager
    
    with _manager_lock:
        if _global_device_manager is None:
            # DI 지원 버전으로 생성
            try:
                from ..core.di_container import get_global_container
                di_container = get_global_container()
                _global_device_manager = DeviceManager(di_container=di_container)
            except ImportError:
                # DI Container 없이도 동작
                _global_device_manager = DeviceManager()
            
            logger.info(f"✅ DI DeviceManager 초기화 완료 - 디바이스: {_global_device_manager.device}")
        return _global_device_manager

def get_memory_adapter(device: str = "auto", **kwargs) -> MemoryManagerAdapter:
    """VirtualFittingStep용 어댑터 반환 (DI 자동 적용)"""
    global _global_adapter
    
    try:
        with _manager_lock:
            if _global_adapter is None:
                # DI 지원 버전으로 생성
                try:
                    from ..core.di_container import get_global_container
                    di_container = get_global_container()
                    base_manager = get_memory_manager(device=device, **kwargs)
                    _global_adapter = MemoryManagerAdapter(base_manager, di_container=di_container)
                except ImportError:
                    # DI Container 없이도 동작
                    base_manager = get_memory_manager(device=device, **kwargs)
                    _global_adapter = MemoryManagerAdapter(base_manager)
                
                logger.info(f"✅ DI 메모리 어댑터 초기화 - 디바이스: {device}")
            return _global_adapter
    except Exception as e:
        logger.error(f"❌ 메모리 어댑터 초기화 실패: {e}")
        # 폴백 어댑터 생성
        fallback_manager = MemoryManager(device="cpu")
        return MemoryManagerAdapter(fallback_manager)


def create_memory_manager(device: str = "auto", **kwargs) -> MemoryManager:
        """메모리 관리자 팩토리 함수 (DI 자동 적용)"""
        try:
            if device == "auto":
                device = SYSTEM_INFO["device"]
            
            # DI Container 자동 해결
            di_container = None
            try:
                from ..core.di_container import get_global_container
                di_container = get_global_container()
            except ImportError:
                pass
            
            logger.debug(f"📦 DI MemoryManager 생성 - 디바이스: {device}")
            manager = MemoryManager(device=device, di_container=di_container, **kwargs)
            return manager
        except Exception as e:
            logger.warning(f"⚠️ MemoryManager 생성 실패: {e}")
            # 실패 시에도 기본 인스턴스 반환
            return MemoryManager(device="cpu")
        
def get_step_memory_manager(step_name: str, **kwargs) -> Union[MemoryManager, MemoryManagerAdapter]:
    """
    🔥 Step별 메모리 관리자 반환 (main.py에서 요구하는 핵심 함수 + DI 통합)
    ✅ 기존 함수명 완전 유지
    ✅ __init__.py에서 export되는 핵심 함수
    ✅ import 오류 완전 해결
    ✅ VirtualFittingStep용 MemoryManagerAdapter 지원
    ✅ DI 자동 적용
    """
    try:
        # Step별 특화 설정 (GitHub 8단계 파이프라인 기준)
        step_configs = {
            "HumanParsingStep": {"memory_limit_gb": 8.0, "optimization_enabled": True},
            "PoseEstimationStep": {"memory_limit_gb": 6.0, "optimization_enabled": True},
            "ClothSegmentationStep": {"memory_limit_gb": 4.0, "optimization_enabled": True},
            "GeometricMatchingStep": {"memory_limit_gb": 6.0, "optimization_enabled": True},
            "ClothWarpingStep": {"memory_limit_gb": 8.0, "optimization_enabled": True},
            "VirtualFittingStep": {"memory_limit_gb": 16.0, "optimization_enabled": True},
            "PostProcessingStep": {"memory_limit_gb": 8.0, "optimization_enabled": True},
            "QualityAssessmentStep": {"memory_limit_gb": 4.0, "optimization_enabled": True}
        }
        
        # M3 Max에서는 더 큰 메모리 할당
        if SYSTEM_INFO["is_m3_max"]:
            for config in step_configs.values():
                config["memory_limit_gb"] *= 2  # M3 Max에서 2배 메모리 할당
        
        # Step별 설정 적용
        step_config = step_configs.get(step_name, {"memory_limit_gb": 8.0})
        final_kwargs = kwargs.copy()
        final_kwargs.update(step_config)
        
        # DI Container 자동 해결
        di_container = None
        try:
            from ..core.di_container import get_global_container
            di_container = get_global_container()
        except ImportError:
            pass
        
        # 메모리 관리자 생성
        base_manager = MemoryManager(di_container=di_container, **final_kwargs)
        base_manager.step_name = step_name
        base_manager.logger = logging.getLogger(f"memory.{step_name}")
        
        # VirtualFittingStep인 경우 어댑터 반환
        if step_name == "VirtualFittingStep":
            adapter = MemoryManagerAdapter(base_manager, di_container=di_container)
            logger.debug(f"📝 {step_name} DI MemoryManagerAdapter 생성 완료")
            return adapter
        else:
            logger.debug(f"📝 {step_name} DI 메모리 관리자 생성 완료")
            return base_manager
        
    except Exception as e:
        logger.warning(f"⚠️ {step_name} 메모리 관리자 생성 실패: {e}")
        # 폴백: 기본 메모리 관리자 반환
        base_manager = MemoryManager(**kwargs)
        if step_name == "VirtualFittingStep":
            return MemoryManagerAdapter(base_manager)
        return base_manager
    

def create_optimized_memory_manager(
    device: str = "auto",
    memory_gb: float = None,
    is_m3_max: bool = None,
    optimization_enabled: bool = True,
    di_container: Optional[Any] = None
) -> MemoryManager:
    """최적화된 메모리 관리자 생성 (DI 지원)"""
    
    # 기본값 설정
    if device == "auto":
        device = SYSTEM_INFO["device"]
    if memory_gb is None:
        memory_gb = SYSTEM_INFO["memory_gb"]
    if is_m3_max is None:
        is_m3_max = SYSTEM_INFO["is_m3_max"]
    
    # DI Container 자동 해결
    if di_container is None:
        try:
            from ..core.di_container import get_global_container
            di_container = get_global_container()
        except ImportError:
            pass
    
    return MemoryManager(
        device=device,
        memory_gb=memory_gb,
        is_m3_max=is_m3_max,
        optimization_enabled=optimization_enabled,
        auto_cleanup=True,
        enable_caching=True,
        di_container=di_container
    )

def initialize_global_memory_manager(device: str = None, **kwargs) -> MemoryManager:
    """전역 메모리 관리자 초기화 (DI 자동 적용)"""
    global _global_memory_manager
    
    try:
        with _manager_lock:
            if _global_memory_manager is None:
                if device is None:
                    device = SYSTEM_INFO["device"]
                
                # DI Container 자동 해결
                di_container = None
                try:
                    from ..core.di_container import get_global_container
                    di_container = get_global_container()
                except ImportError:
                    pass
                
                _global_memory_manager = MemoryManager(device=device, di_container=di_container, **kwargs)
                logger.info(f"✅ 전역 DI 메모리 관리자 초기화 완료 - 디바이스: {device}")
            return _global_memory_manager
    except Exception as e:
        logger.warning(f"⚠️ 전역 메모리 관리자 초기화 실패: {e}")
        # 기본 인스턴스 생성
        _global_memory_manager = MemoryManager(device="cpu")
        return _global_memory_manager

def optimize_memory_usage(device: str = None, aggressive: bool = False) -> Dict[str, Any]:
    """메모리 사용량 최적화 - 동기 함수 (DI 통합)"""
    try:
        manager = get_memory_manager(device=device or "auto")
        
        # 최적화 전 상태
        before_stats = manager.get_memory_stats()
        
        # 메모리 정리
        result = manager.cleanup_memory(aggressive=aggressive)
        
        # 최적화 후 상태
        after_stats = manager.get_memory_stats()
        
        # 결과 계산
        freed_cpu = max(0, before_stats.cpu_used_gb - after_stats.cpu_used_gb)
        freed_gpu = max(0, before_stats.gpu_allocated_gb - after_stats.gpu_allocated_gb)
        freed_cache = max(0, before_stats.cache_size_mb - after_stats.cache_size_mb)
        
        result.update({
            "freed_memory": {
                "cpu_gb": freed_cpu,
                "gpu_gb": freed_gpu,
                "cache_mb": freed_cache
            },
            "before": {
                "cpu_used_gb": before_stats.cpu_used_gb,
                "gpu_allocated_gb": before_stats.gpu_allocated_gb,
                "cache_size_mb": before_stats.cache_size_mb
            },
            "after": {
                "cpu_used_gb": after_stats.cpu_used_gb,
                "gpu_allocated_gb": after_stats.gpu_allocated_gb,
                "cache_size_mb": after_stats.cache_size_mb
            },
            "di_integrated": getattr(manager, '_di_container', None) is not None
        })
        
        logger.info(f"🧹 메모리 최적화 완료 - CPU: {freed_cpu:.2f}GB, GPU: {freed_gpu:.2f}GB")
        return result
        
    except Exception as e:
        logger.warning(f"⚠️ 메모리 최적화 실패: {e}")
        return {
            "success": False,
            "error": str(e),
            "device": device or "unknown"
        }

# ==============================================
# 🔥 편의 함수들 (모든 async 오류 해결 + DI 통합)
# ==============================================

def optimize_memory() -> Dict[str, Any]:
    """메모리 최적화 (동기화 + DI 자동 적용)"""
    try:
        manager = get_memory_manager()
        return manager.optimize_memory()
    except Exception as e:
        logger.warning(f"⚠️ 메모리 최적화 실패: {e}")
        return {"success": False, "error": str(e)}

def check_memory():
    """메모리 상태 확인 (DI 자동 적용)"""
    try:
        manager = get_memory_manager()
        return manager.check_memory_pressure()
    except Exception as e:
        logger.warning(f"⚠️ 메모리 상태 확인 실패: {e}")
        return {"status": "unknown", "error": str(e)}

def check_memory_available(min_gb: float = 1.0) -> bool:
    """사용 가능한 메모리 확인 (DI 자동 적용)"""
    try:
        manager = get_memory_manager()
        stats = manager.get_memory_stats()
        return stats.cpu_available_gb >= min_gb
    except Exception as e:
        logger.warning(f"⚠️ 메모리 확인 실패: {e}")
        return True  # 확인 실패 시 true 반환

def get_memory_info() -> Dict[str, Any]:
    """메모리 정보 조회 (DI 통합)"""
    try:
        manager = get_memory_manager()
        stats = manager.get_memory_stats()
        return {
            "device": manager.device,
            "cpu_total_gb": stats.cpu_total_gb,
            "cpu_available_gb": stats.cpu_available_gb,
            "cpu_used_gb": stats.cpu_used_gb,
            "gpu_total_gb": stats.gpu_total_gb,
            "gpu_allocated_gb": stats.gpu_allocated_gb,
            "is_m3_max": manager.is_m3_max,
            "memory_gb": manager.memory_gb,
            "conda_env": SYSTEM_INFO["in_conda"],
            "di_integrated": getattr(manager, '_di_container', None) is not None,
            "dependencies_resolved": getattr(manager, '_dependencies_resolved', False)
        }
    except Exception as e:
        logger.warning(f"⚠️ 메모리 정보 조회 실패: {e}")
        return {"error": str(e)}

# ==============================================
# 🔥 데코레이터 (async 오류 해결 + DI 통합)
# ==============================================

def memory_efficient(clear_before: bool = True, clear_after: bool = True):
    """메모리 효율적 실행 데코레이터 (DI 자동 적용)"""
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            manager = get_memory_manager()
            if clear_before:
                manager.cleanup_memory()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                if clear_after:
                    manager.cleanup_memory()
        
        # 함수가 코루틴인지 확인
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                manager = get_memory_manager()
                if clear_before:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, manager.cleanup_memory)
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    if clear_after:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, manager.cleanup_memory)
            return async_wrapper
        else:
            return sync_wrapper
    return decorator

# ==============================================
# 🔥 conda 환경 특화 함수들 (오류 완전 해결 + DI 통합)
# ==============================================

def setup_conda_memory_optimization() -> bool:
    """
    conda 환경 메모리 최적화 설정 (완전 동기화 + DI 통합)
    ✅ object dict can't be used in 'await' expression 완전 해결
    ✅ DI를 통한 추가 최적화
    """
    try:
        if not SYSTEM_INFO["in_conda"]:
            logger.warning("⚠️ conda 환경이 아닙니다")
            return False
        
        optimizations = []
        
        # 1. NumPy/MKL 스레드 최적화
        if NUMPY_AVAILABLE:
            optimal_threads = min(8, SYSTEM_INFO["cpu_count"])
            os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
            os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
            os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_threads)
            optimizations.append(f"NumPy/MKL 스레드: {optimal_threads}")
        
        # 2. PyTorch 설정
        if TORCH_AVAILABLE:
            torch.set_num_threads(min(16, SYSTEM_INFO["cpu_count"]))
            optimizations.append(f"PyTorch 스레드: {torch.get_num_threads()}")
        
        # 3. M3 Max 특화 설정
        if SYSTEM_INFO["is_m3_max"]:
            os.environ.update({
                'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',
                'PYTORCH_MPS_LOW_WATERMARK_RATIO': '0.0',
                'PYTORCH_ENABLE_MPS_FALLBACK': '1'
            })
            optimizations.append("M3 Max MPS 최적화")
        
        # 4. DI를 통한 추가 최적화
        try:
            from ..core.di_container import get_global_container
            di_container = get_global_container()
            
            # DeviceManager를 통한 최적화
            device_manager = di_container.get('device_manager')
            if device_manager and hasattr(device_manager, 'optimize_memory'):
                device_manager.optimize_memory()
                optimizations.append("DI DeviceManager conda 최적화")
            
            # MemoryManager를 통한 최적화
            memory_manager = di_container.get('memory_manager')
            if memory_manager and hasattr(memory_manager, 'optimize_memory'):
                memory_manager.optimize_memory()
                optimizations.append("DI MemoryManager conda 최적화")
                
        except ImportError:
            pass  # DI Container 없이도 동작
        except Exception as e:
            optimizations.append(f"DI conda 최적화 실패: {str(e)[:30]}")
        
        logger.info("✅ conda 환경 메모리 최적화 설정 완료 (DI 통합)")
        for opt in optimizations:
            logger.debug(f"   - {opt}")
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ conda 메모리 최적화 실패: {e}")
        return False

def get_conda_memory_recommendations() -> List[str]:
    """conda 환경 메모리 최적화 권장사항 (DI 통합)"""
    recommendations = []
    
    try:
        if not SYSTEM_INFO["in_conda"]:
            recommendations.append("conda 환경 사용 권장")
            return recommendations
        
        # 현재 상태 확인
        current_threads = os.environ.get('OMP_NUM_THREADS', 'auto')
        if current_threads == 'auto':
            recommendations.append("OMP_NUM_THREADS 설정 권장")
        
        if TORCH_AVAILABLE:
            current_torch_threads = torch.get_num_threads()
            optimal_threads = min(16, SYSTEM_INFO["cpu_count"])
            if current_torch_threads != optimal_threads:
                recommendations.append(f"PyTorch 스레드 수 최적화 ({current_torch_threads} → {optimal_threads})")
        
        if SYSTEM_INFO["is_m3_max"]:
            mps_ratio = os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO')
            if mps_ratio != '0.0':
                recommendations.append("M3 Max MPS 메모리 비율 최적화 권장")
        
        # DI 관련 권장사항
        try:
            from ..core.di_container import get_global_container
            di_container = get_global_container()
            
            # DI를 통한 최적화 가능 여부 확인
            memory_manager = di_container.get('memory_manager')
            if memory_manager:
                recommendations.append("DI MemoryManager를 통한 conda 최적화 활용 가능")
                
        except ImportError:
            recommendations.append("DI Container 설정으로 추가 최적화 가능")
        except Exception:
            pass
        
        if not recommendations:
            recommendations.append("conda 환경 최적화 상태 양호 (DI 통합)")
        
        return recommendations
        
    except Exception as e:
        logger.warning(f"⚠️ conda 권장사항 생성 실패: {e}")
        return ["conda 최적화 상태 확인 불가"]

def create_conda_optimized_manager(step_name: str = "default", **kwargs) -> Union[MemoryManager, MemoryManagerAdapter]:
    """conda 환경 최적화된 메모리 관리자 생성 (DI 통합)"""
    try:
        # conda 최적화 먼저 설정
        setup_conda_memory_optimization()
        
        # DI Container 자동 해결
        di_container = None
        try:
            from ..core.di_container import get_global_container
            di_container = get_global_container()
        except ImportError:
            pass
        
        # Step별 설정
        if step_name == "VirtualFittingStep":
            base_manager = create_optimized_memory_manager(
                memory_gb=SYSTEM_INFO["memory_gb"],
                optimization_enabled=True,
                di_container=di_container,
                **kwargs
            )
            return MemoryManagerAdapter(base_manager, di_container=di_container)
        else:
            return create_optimized_memory_manager(
                memory_gb=SYSTEM_INFO["memory_gb"],
                optimization_enabled=True,
                di_container=di_container,
                **kwargs
            )
        
    except Exception as e:
        logger.warning(f"⚠️ conda 최적화 관리자 생성 실패: {e}")
        # 폴백
        base_manager = MemoryManager(device="cpu")
        if step_name == "VirtualFittingStep":
            return MemoryManagerAdapter(base_manager)
        return base_manager

# ==============================================
# 🔥 진단 및 디버깅 함수들 (DI 통합)
# ==============================================

def diagnose_memory_issues() -> Dict[str, Any]:
    """메모리 문제 진단 (DI 통합)"""
    try:
        diagnosis = {
            "system_info": SYSTEM_INFO,
            "memory_status": {},
            "issues": [],
            "recommendations": [],
            "di_status": {}
        }
        
        # 메모리 상태 확인
        manager = get_memory_manager()
        stats = manager.get_memory_stats()
        
        diagnosis["memory_status"] = {
            "cpu_usage_ratio": stats.cpu_used_gb / max(1.0, stats.cpu_total_gb),
            "gpu_usage_ratio": stats.gpu_allocated_gb / max(1.0, stats.gpu_total_gb),
            "cache_size_mb": stats.cache_size_mb,
            "available_gb": stats.cpu_available_gb
        }
        
        # DI 상태 확인
        diagnosis["di_status"] = {
            "di_integrated": getattr(manager, '_di_container', None) is not None,
            "dependencies_resolved": getattr(manager, '_dependencies_resolved', False)
        }
        
        if hasattr(manager, 'get_dependency_status'):
            try:
                diagnosis["di_status"]["dependency_details"] = manager.get_dependency_status()
            except Exception:
                pass
        
        # 문제 감지
        cpu_ratio = diagnosis["memory_status"]["cpu_usage_ratio"]
        if cpu_ratio > 0.9:
            diagnosis["issues"].append("CPU 메모리 사용량이 90%를 초과")
            diagnosis["recommendations"].append("aggressive 메모리 정리 실행")
        elif cpu_ratio > 0.75:
            diagnosis["issues"].append("CPU 메모리 사용량이 75%를 초과")
            diagnosis["recommendations"].append("일반 메모리 정리 실행")
        
        if stats.cache_size_mb > 1000:
            diagnosis["issues"].append("캐시 크기가 1GB를 초과")
            diagnosis["recommendations"].append("캐시 정리 실행")
        
        # conda 환경 확인
        if not SYSTEM_INFO["in_conda"]:
            diagnosis["issues"].append("conda 환경이 아님")
            diagnosis["recommendations"].append("conda 환경 사용 권장")
        
        # PyTorch 확인
        if not TORCH_AVAILABLE:
            diagnosis["issues"].append("PyTorch 없음")
            diagnosis["recommendations"].append("conda install pytorch 실행")
        
        # DI 관련 권장사항
        if not diagnosis["di_status"]["di_integrated"]:
            diagnosis["recommendations"].append("DI Container 통합으로 추가 최적화 가능")
        elif diagnosis["di_status"]["di_integrated"] and not diagnosis["di_status"]["dependencies_resolved"]:
            diagnosis["recommendations"].append("DI 의존성 해결 필요")
        
        return diagnosis
        
    except Exception as e:
        return {
            "error": str(e),
            "system_info": SYSTEM_INFO,
            "recommendations": ["시스템 재시작 권장"]
        }

def print_memory_report():
    """메모리 상태 리포트 출력 (DI 통합)"""
    try:
        print("\n" + "="*80)
        print("🔥 MyCloset AI - DI 통합 메모리 상태 리포트")
        print("="*80)
        
        # 시스템 정보
        print(f"🔧 시스템: {SYSTEM_INFO['platform']} / {SYSTEM_INFO['device']}")
        print(f"🍎 M3 Max: {'✅' if SYSTEM_INFO['is_m3_max'] else '❌'}")
        print(f"🐍 conda: {'✅' if SYSTEM_INFO['in_conda'] else '❌'} ({SYSTEM_INFO['conda_env']})")
        print(f"🔥 PyTorch: {'✅' if TORCH_AVAILABLE else '❌'} ({TORCH_VERSION})")
        
        # DI 상태
        manager = get_memory_manager()
        di_integrated = getattr(manager, '_di_container', None) is not None
        dependencies_resolved = getattr(manager, '_dependencies_resolved', False)
        print(f"🔗 DI 통합: {'✅' if di_integrated else '❌'}")
        if di_integrated:
            print(f"🔗 의존성 해결: {'✅' if dependencies_resolved else '❌'}")
        
        # 메모리 상태
        stats = manager.get_memory_stats()
        
        print(f"\n💾 메모리 상태:")
        print(f"   CPU: {stats.cpu_used_gb:.1f}GB / {stats.cpu_total_gb:.1f}GB ({stats.cpu_percent:.1f}%)")
        print(f"   GPU: {stats.gpu_allocated_gb:.1f}GB / {stats.gpu_total_gb:.1f}GB")
        print(f"   캐시: {stats.cache_size_mb:.1f}MB")
        print(f"   사용 가능: {stats.cpu_available_gb:.1f}GB")
        
        # 압박 상태
        pressure = manager.check_memory_pressure()
        status_emoji = {"normal": "✅", "warning": "⚠️", "critical": "❌"}.get(pressure["status"], "❓")
        print(f"\n🚨 압박 상태: {status_emoji} {pressure['status']}")
        
        # DI 의존성 정보
        if di_integrated and hasattr(manager, 'get_dependency_status'):
            try:
                dep_status = manager.get_dependency_status()
                print(f"\n🔗 DI 의존성 상태:")
                dep_info = dep_status.get('dependency_status', {})
                for dep_name, status in dep_info.items():
                    status_emoji = "✅" if status else "❌"
                    print(f"   {dep_name}: {status_emoji}")
            except Exception:
                pass
        
        # 권장사항
        if pressure.get("recommendations"):
            print(f"\n📋 권장사항:")
            for rec in pressure["recommendations"]:
                print(f"   - {rec}")
        
        # conda 권장사항
        conda_recs = get_conda_memory_recommendations()
        if conda_recs and conda_recs[0] != "conda 환경 최적화 상태 양호 (DI 통합)":
            print(f"\n🐍 conda 권장사항:")
            for rec in conda_recs:
                print(f"   - {rec}")
        
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"❌ 메모리 리포트 생성 실패: {e}")

# ==============================================
# 🔥 모듈 익스포트 (DI 함수들 추가)
# ==============================================

__all__ = [
    # 🔥 기존 클래스명 완전 유지 (DI 지원 추가)
    'DeviceManager',             # ✅ main.py에서 필요한 핵심 클래스 + DI 지원
    'MemoryManager',             # ✅ DI 완전 통합
    'MemoryManagerAdapter',      # ✅ VirtualFittingStep 호환용 완전 구현 + DI 지원
    'GPUMemoryManager',          # ✅ 현재 구조에서 사용 + DI 지원
    'MemoryStats',
    'MemoryConfig',
    
    # 🔥 DI 인터페이스들
    'IMemoryManager',
    'IDeviceManager',
    'IDependencyInjectable',
    
    # 🔥 기존 함수명 완전 유지 (DI 자동 적용)
    'get_device_manager',        # ✅ main.py에서 필요한 핵심 함수 + DI 자동 적용
    'get_memory_manager',        # ✅ DI 자동 적용
    'get_global_memory_manager', # ✅ DI 자동 적용
    'get_step_memory_manager',   # ✅ main.py에서 필요한 핵심 함수 + DI 자동 적용
    'get_memory_adapter',        # ✅ VirtualFittingStep 전용 + DI 자동 적용
    'create_memory_manager',     # ✅ DI 자동 적용
    'create_optimized_memory_manager', # ✅ DI 지원 추가
    'initialize_global_memory_manager', # ✅ DI 자동 적용
    'optimize_memory_usage',     # ✅ DI 통합
    'optimize_memory',          # ✅ 완전 동기화 + DI 자동 적용
    'check_memory',             # ✅ DI 자동 적용
    'check_memory_available',   # ✅ DI 자동 적용
    'get_memory_info',          # ✅ DI 통합
    'memory_efficient',         # ✅ DI 자동 적용
    
    # 🔧 conda 환경 특화 함수들 (DI 통합)
    'setup_conda_memory_optimization', # ✅ DI 통합
    'get_conda_memory_recommendations', # ✅ DI 통합
    'create_conda_optimized_manager',   # ✅ DI 통합
    'diagnose_memory_issues',          # ✅ DI 통합
    'print_memory_report',             # ✅ DI 통합
    
    # 🔥 DI 전용 함수들
    'create_di_memory_manager',
    'create_di_device_manager',
    'create_di_memory_adapter',
    'get_global_di_memory_manager',
    'get_global_di_device_manager',
    'get_global_di_memory_adapter',
    
    # 🔧 시스템 정보
    'SYSTEM_INFO',
    'TORCH_AVAILABLE',
    'PSUTIL_AVAILABLE',
    'NUMPY_AVAILABLE'
]

# ==============================================
# 🔥 모듈 로드 완료 (DI 통합 버전)
# ==============================================

# 환경 정보 로깅 (INFO 레벨로 중요 정보만)
logger.info("✅ DI 완전 통합 MemoryManager v9.0 로드 완료")
logger.info("🔗 CircularReferenceFreeDIContainer 연동 완료")
logger.info(f"🔧 시스템: {SYSTEM_INFO['platform']} / {SYSTEM_INFO['device']}")

if SYSTEM_INFO["is_m3_max"]:
    logger.info(f"🍎 M3 Max 감지 - {SYSTEM_INFO['memory_gb']}GB 메모리")

if SYSTEM_INFO["in_conda"]:
    logger.info(f"🐍 conda 환경: {SYSTEM_INFO['conda_env']}")

logger.debug("🔗 주요 클래스: DeviceManager, MemoryManager, MemoryManagerAdapter, GPUMemoryManager (모두 DI 지원)")
logger.debug("🔗 주요 함수: get_device_manager, get_step_memory_manager, get_memory_adapter (모두 DI 자동 적용)")
logger.debug("⚡ M3 Max + conda 환경 완전 최적화 + DI 통합")
logger.debug("🔧 모든 async/await 오류 완전 해결")
logger.debug("🎯 DeviceManager.setup_mps_compatibility 메서드 완전 구현 + DI 지원")
logger.debug("🔀 순환참조 완전 방지 + DI Container 연동")
logger.debug("🛡️ Mock 폴백 구현체 포함")

# M3 Max + conda 조합 확인
if SYSTEM_INFO["is_m3_max"] and SYSTEM_INFO["in_conda"]:
    logger.info("🚀 M3 Max + conda + DI 최고 성능 모드 활성화")

# conda 환경 최적화 자동 설정
if SYSTEM_INFO["in_conda"]:
    try:
        setup_conda_memory_optimization()
        logger.debug("✅ conda 환경 메모리 최적화 자동 설정 완료 (DI 통합)")
    except Exception as e:
        logger.debug(f"⚠️ conda 자동 최적화 건너뜀: {e}")

logger.info("🎯 main.py import 오류 완전 해결 + DI 통합:")
logger.info("   - DeviceManager 클래스 완전 구현 + DI 지원 ✅")
logger.info("   - setup_mps_compatibility 메서드 포함 + DI 최적화 ✅")
logger.info("   - get_device_manager 함수 제공 + DI 자동 적용 ✅")
logger.info("   - RuntimeWarning: coroutine 완전 해결 ✅")
logger.info("   - object dict can't be used in 'await' expression 완전 해결 ✅")
logger.info("   - CircularReferenceFreeDIContainer 완전 연동 ✅")
logger.info("   - 순환참조 완전 방지 + 의존성 주입 지원 ✅")
logger.info("   - Mock 폴백 구현체 포함 ✅")

logger.info("🔥 DI 통합 완료:")
logger.info("   - 모든 클래스가 IDependencyInjectable 구현 ✅")
logger.info("   - DI Container 자동 해결 및 등록 ✅")
logger.info("   - 의존성 주입 및 해결 시스템 완전 구현 ✅")
logger.info("   - 기존 인터페이스 100% 호환성 유지 ✅")
logger.info("   - Step별 DI 자동 적용 ✅")