"""
🍎 MyCloset AI - GPU 설정 매니저 (완전 개선 버전)
=======================================================

✅ Clean Architecture 적용
✅ 단일 책임 원칙 준수
✅ Type Safety 완벽 적용
✅ Conda 환경 완전 지원
✅ M3 Max 최적화
✅ 순환 참조 방지
✅ 에러 복구 메커니즘
✅ 성능 모니터링
✅ PyTorch 2.6+ 호환

프로젝트 구조:
backend/app/core/gpu_config.py (이 파일)
    ↓ 사용됨
backend/app/core/config.py
backend/app/api/pipeline_routes.py
backend/app/ai_pipeline/utils/model_loader.py
backend/app/ai_pipeline/steps/*.py
"""

import os
import gc
import time
import platform
import threading
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Protocol
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import logging
import warnings

# =============================================================================
# 📦 조건부 임포트 (안전한 처리)
# =============================================================================

try:
    import psutil
    PSUTIL_AVAILABLE = True
    PSUTIL_VERSION = psutil.__version__
except ImportError:
    PSUTIL_AVAILABLE = False
    PSUTIL_VERSION = "not_available"

try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_VERSION = torch.__version__
    TORCH_MAJOR, TORCH_MINOR = map(int, TORCH_VERSION.split('.')[:2])
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_VERSION = "not_available"
    TORCH_MAJOR, TORCH_MINOR = 0, 0

try:
    import numpy as np
    NUMPY_AVAILABLE = True
    NUMPY_VERSION = np.__version__
except ImportError:
    NUMPY_AVAILABLE = False
    NUMPY_VERSION = "not_available"

# =============================================================================
# 🔧 로깅 설정 (노이즈 최소화)
# =============================================================================

# 경고 필터링
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # 로그 노이즈 감소

# =============================================================================
# 🔧 상수 및 열거형
# =============================================================================

class DeviceType(Enum):
    """디바이스 타입"""
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"
    AUTO = "auto"

class OptimizationLevel(Enum):
    """최적화 레벨"""
    SAFE = "safe"
    BALANCED = "balanced"
    PERFORMANCE = "performance"
    ULTRA = "ultra"

class PerformanceClass(Enum):
    """성능 등급"""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA_HIGH = "ultra_high"

# =============================================================================
# 📊 데이터 클래스
# =============================================================================

@dataclass(frozen=True)
class SystemInfo:
    """시스템 정보"""
    platform: str
    machine: str
    processor: str
    python_version: str
    pytorch_version: str
    numpy_version: str
    is_m3_max: bool
    memory_gb: float
    cpu_cores: int
    detection_time: float = field(default_factory=time.time)

@dataclass(frozen=True)
class CondaEnvironment:
    """Conda 환경 정보"""
    is_conda: bool
    env_name: Optional[str] = None
    prefix: Optional[str] = None
    package_manager: str = "pip"
    python_version: str = ""
    optimization_level: str = "standard"

@dataclass(frozen=True)
class DeviceCapabilities:
    """디바이스 기능"""
    device_type: str
    name: str
    memory_gb: float
    supports_fp16: bool = False
    supports_fp32: bool = True
    unified_memory: bool = False
    max_batch_size: int = 1
    recommended_image_size: Tuple[int, int] = (512, 512)
    tensor_cores: bool = False
    neural_engine: bool = False

@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    memory_cleanup_count: int = 0
    memory_cleanup_avg_ms: float = 0.0
    error_count: int = 0
    uptime_seconds: float = 0.0
    last_reset: float = field(default_factory=time.time)

@dataclass(frozen=True)
class OptimizationProfile:
    """최적화 프로필"""
    batch_size: int
    max_workers: int
    memory_fraction: float
    quality_level: str
    dtype: str = "float32"
    mixed_precision: bool = False
    enable_checkpointing: bool = False

# =============================================================================
# 🔧 프로토콜 (인터페이스)
# =============================================================================

class MemoryManagerProtocol(Protocol):
    """메모리 관리자 프로토콜"""
    def cleanup_memory(self, aggressive: bool = False) -> Dict[str, Any]: ...
    def get_memory_info(self) -> Dict[str, Any]: ...

class DeviceDetectorProtocol(Protocol):
    """디바이스 감지기 프로토콜"""
    def detect_device(self) -> str: ...
    def get_device_capabilities(self) -> DeviceCapabilities: ...

# =============================================================================
# 🔧 메모리 관리자 (단일 책임)
# =============================================================================

class MemoryManager:
    """메모리 관리 전담 클래스"""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._last_cleanup = 0.0
        self._cleanup_interval = 1.0  # 1초 간격
        self._failure_count = 0
        self._max_failures = 3
    
    def cleanup_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """메모리 정리 (스레드 안전)"""
        with self._lock:
            current_time = time.time()
            
            # 연속 실패 체크
            if self._failure_count >= self._max_failures and not aggressive:
                return self._create_result(
                    success=True,
                    method="failure_threshold",
                    message=f"연속 실패 {self._failure_count}회로 스킵"
                )
            
            # 호출 간격 체크
            if current_time - self._last_cleanup < self._cleanup_interval and not aggressive:
                return self._create_result(
                    success=True,
                    method="throttled",
                    message=f"호출 제한 ({self._cleanup_interval}초 간격)"
                )
            
            self._last_cleanup = current_time
            
            try:
                return self._perform_cleanup(aggressive)
            except Exception as e:
                self._failure_count += 1
                return self._create_result(
                    success=False,
                    method="error",
                    message=str(e)[:200]
                )
    
    def _perform_cleanup(self, aggressive: bool) -> Dict[str, Any]:
        """실제 메모리 정리 수행"""
        start_time = time.time()
        methods = []
        
        # 기본 가비지 컬렉션
        collected = gc.collect()
        if collected > 0:
            methods.append(f"gc_collected_{collected}")
        
        if not TORCH_AVAILABLE:
            return self._create_result(
                success=True,
                method="gc_only",
                message=f"가비지 컬렉션 완료 ({collected}개)",
                duration=time.time() - start_time,
                methods=methods
            )
        
        # PyTorch 메모리 정리 시도
        cleanup_success = False
        
        # MPS 메모리 정리
        if self._try_mps_cleanup():
            methods.append("mps_cleanup")
            cleanup_success = True
        
        # CUDA 메모리 정리
        elif self._try_cuda_cleanup():
            methods.append("cuda_cleanup")
            cleanup_success = True
        
        # 적극적 정리
        if aggressive:
            for i in range(3):
                additional = gc.collect()
                if additional > 0:
                    collected += additional
                    methods.append(f"aggressive_gc_round_{i+1}")
                if i < 2:
                    time.sleep(0.05)  # 50ms 대기
        
        if cleanup_success:
            self._failure_count = 0  # 성공 시 실패 카운터 리셋
        
        return self._create_result(
            success=True,
            method="comprehensive",
            message=f"메모리 정리 완료 (총 {collected}개)",
            duration=time.time() - start_time,
            methods=methods,
            aggressive=aggressive
        )
    
    def _try_mps_cleanup(self) -> bool:
        """MPS 메모리 정리 시도"""
        try:
            # torch.mps.empty_cache() 시도
            if (hasattr(torch, 'mps') and 
                hasattr(torch.mps, 'empty_cache') and
                callable(torch.mps.empty_cache)):
                torch.mps.empty_cache()
                return True
            
            # torch.backends.mps.empty_cache() 시도
            if (hasattr(torch.backends, 'mps') and 
                hasattr(torch.backends.mps, 'empty_cache') and
                callable(torch.backends.mps.empty_cache)):
                torch.backends.mps.empty_cache()
                return True
            
            # torch.mps.synchronize() 시도
            if (hasattr(torch, 'mps') and 
                hasattr(torch.mps, 'synchronize') and
                callable(torch.mps.synchronize)):
                torch.mps.synchronize()
                return True
                
        except (AttributeError, RuntimeError, TypeError):
            pass
        
        return False
    
    def _try_cuda_cleanup(self) -> bool:
        """CUDA 메모리 정리 시도"""
        try:
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
                return True
        except Exception:
            pass
        return False
    
    def _create_result(self, success: bool, method: str, message: str, 
                      duration: float = 0.0, **kwargs) -> Dict[str, Any]:
        """결과 딕셔너리 생성"""
        return {
            "success": success,
            "method": method,
            "message": message,
            "duration": round(duration, 4),
            "timestamp": time.time(),
            **kwargs
        }
    
    def get_memory_info(self) -> Dict[str, Any]:
        """메모리 정보 조회"""
        info = {
            "timestamp": time.time(),
            "psutil_available": PSUTIL_AVAILABLE
        }
        
        if PSUTIL_AVAILABLE:
            try:
                memory = psutil.virtual_memory()
                info.update({
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "used_percent": round(memory.percent, 1),
                    "free_gb": round(memory.free / (1024**3), 2)
                })
            except Exception as e:
                info["error"] = str(e)[:100]
        else:
            info.update({
                "total_gb": 16.0,
                "available_gb": 12.0,
                "used_percent": 25.0,
                "fallback_mode": True
            })
        
        return info

# =============================================================================
# 🔧 하드웨어 감지기 (단일 책임)
# =============================================================================

class HardwareDetector:
    """하드웨어 정보 감지 전담 클래스"""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._detection_time = time.time()
    
    @lru_cache(maxsize=1)
    def get_system_info(self) -> SystemInfo:
        """시스템 정보 수집 (캐시됨)"""
        return SystemInfo(
            platform=platform.system(),
            machine=platform.machine(),
            processor=platform.processor(),
            python_version=platform.python_version(),
            pytorch_version=TORCH_VERSION,
            numpy_version=NUMPY_VERSION,
            is_m3_max=self.is_m3_max(),
            memory_gb=self.get_memory_gb(),
            cpu_cores=self.get_cpu_cores(),
            detection_time=self._detection_time
        )
    
    def is_m3_max(self) -> bool:
        """M3 Max 감지 (정밀 검사)"""
        if "m3_max" in self._cache:
            return self._cache["m3_max"]
        
        with self._lock:
            try:
                # 1차: 플랫폼 체크
                if platform.system() != "Darwin" or platform.machine() != "arm64":
                    result = False
                else:
                    score = 0
                    
                    # 2차: 메모리 기반 감지 (M3 Max는 96GB/128GB)
                    memory_gb = self.get_memory_gb()
                    if memory_gb >= 120:  # 128GB
                        score += 3
                    elif memory_gb >= 90:  # 96GB
                        score += 2
                    
                    # 3차: CPU 코어 수 (M3 Max는 16코어)
                    cpu_cores = self.get_cpu_cores()
                    if cpu_cores >= 16:
                        score += 2
                    elif cpu_cores >= 14:
                        score += 1
                    
                    # 4차: MPS 지원 체크
                    if self._check_mps_support():
                        score += 1
                    
                    result = score >= 3
                
                self._cache["m3_max"] = result
                return result
                
            except Exception as e:
                logger.debug(f"M3 Max 감지 실패: {e}")
                self._cache["m3_max"] = False
                return False
    
    def get_memory_gb(self) -> float:
        """시스템 메모리 용량 (GB)"""
        if "memory_gb" in self._cache:
            return self._cache["memory_gb"]
        
        try:
            if PSUTIL_AVAILABLE:
                memory_gb = round(psutil.virtual_memory().total / (1024**3), 2)
            else:
                # macOS sysctl 폴백
                if platform.system() == "Darwin":
                    result = subprocess.run(
                        ['sysctl', 'hw.memsize'], 
                        capture_output=True, 
                        text=True, 
                        timeout=5
                    )
                    if result.returncode == 0:
                        memory_bytes = int(result.stdout.split(':')[1].strip())
                        memory_gb = round(memory_bytes / (1024**3), 2)
                    else:
                        memory_gb = 16.0
                else:
                    memory_gb = 16.0
            
            self._cache["memory_gb"] = memory_gb
            return memory_gb
            
        except Exception as e:
            logger.debug(f"메모리 정보 수집 실패: {e}")
            self._cache["memory_gb"] = 16.0
            return 16.0
    
    def get_cpu_cores(self) -> int:
        """CPU 코어 수"""
        if "cpu_cores" in self._cache:
            return self._cache["cpu_cores"]
        
        try:
            if PSUTIL_AVAILABLE:
                physical = psutil.cpu_count(logical=False) or 8
                logical = psutil.cpu_count(logical=True) or 8
                cores = max(physical, logical)
            else:
                cores = os.cpu_count() or 8
            
            self._cache["cpu_cores"] = cores
            return cores
            
        except Exception:
            self._cache["cpu_cores"] = 8
            return 8
    
    def detect_conda_environment(self) -> CondaEnvironment:
        """Conda 환경 감지"""
        conda_info = {
            'is_conda': False,
            'env_name': None,
            'prefix': None,
            'package_manager': 'pip',
            'python_version': platform.python_version(),
            'optimization_level': 'standard'
        }
        
        try:
            # CONDA_DEFAULT_ENV 확인
            conda_env = os.environ.get('CONDA_DEFAULT_ENV')
            if conda_env and conda_env != 'base':
                conda_info.update({
                    'is_conda': True,
                    'env_name': conda_env,
                    'package_manager': 'conda',
                    'optimization_level': 'conda_optimized'
                })
            
            # CONDA_PREFIX 확인
            conda_prefix = os.environ.get('CONDA_PREFIX')
            if conda_prefix:
                conda_info['prefix'] = conda_prefix
                if not conda_info['is_conda']:
                    conda_info.update({
                        'is_conda': True,
                        'env_name': Path(conda_prefix).name,
                        'package_manager': 'conda',
                        'optimization_level': 'conda_optimized'
                    })
            
            # Mamba 감지
            if conda_prefix and 'mamba' in str(conda_prefix).lower():
                conda_info['package_manager'] = 'mamba'
                conda_info['optimization_level'] = 'mamba_optimized'
        
        except Exception as e:
            logger.debug(f"Conda 환경 감지 실패: {e}")
        
        return CondaEnvironment(**conda_info)
    
    def detect_device(self) -> str:
        """최적 디바이스 감지"""
        if not TORCH_AVAILABLE:
            return "cpu"
        
        try:
            # 환경 변수 우선 확인
            env_device = os.environ.get('DEVICE', '').lower()
            if env_device in ['cpu', 'mps', 'cuda'] and self._is_device_available(env_device):
                return env_device
            
            # 자동 감지: MPS > CUDA > CPU
            if self._check_mps_support():
                return "mps"
            elif hasattr(torch, 'cuda') and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
                
        except Exception as e:
            logger.debug(f"디바이스 감지 실패: {e}")
            return "cpu"
    
    def get_device_capabilities(self, device: str) -> DeviceCapabilities:
        """디바이스 기능 정보 생성"""
        system_info = self.get_system_info()
        
        if device == "mps" and system_info.is_m3_max:
            return DeviceCapabilities(
                device_type=device,
                name="Apple M3 Max",
                memory_gb=system_info.memory_gb,
                supports_fp16=False,  # MPS FP16 제한으로 안전하게 False
                supports_fp32=True,
                unified_memory=True,
                max_batch_size=8 if system_info.memory_gb >= 120 else 4,
                recommended_image_size=(1024, 1024) if system_info.memory_gb >= 120 else (768, 768),
                neural_engine=True
            )
        elif device == "mps":
            return DeviceCapabilities(
                device_type=device,
                name="Apple Silicon",
                memory_gb=system_info.memory_gb,
                supports_fp32=True,
                unified_memory=True,
                max_batch_size=4,
                recommended_image_size=(768, 768)
            )
        elif device == "cuda" and TORCH_AVAILABLE:
            try:
                props = torch.cuda.get_device_properties(0)
                return DeviceCapabilities(
                    device_type=device,
                    name=props.name,
                    memory_gb=round(props.total_memory / (1024**3), 2),
                    supports_fp16=props.major >= 7,
                    supports_fp32=True,
                    max_batch_size=4,
                    tensor_cores=props.major >= 7
                )
            except Exception:
                pass
        
        # CPU 폴백
        return DeviceCapabilities(
            device_type="cpu",
            name="CPU",
            memory_gb=system_info.memory_gb,
            supports_fp32=True,
            max_batch_size=1
        )
    
    def _check_mps_support(self) -> bool:
        """MPS 지원 여부 확인"""
        if not TORCH_AVAILABLE:
            return False
        
        try:
            return (hasattr(torch.backends, 'mps') and 
                   hasattr(torch.backends.mps, 'is_available') and
                   torch.backends.mps.is_available())
        except Exception:
            return False
    
    def _is_device_available(self, device: str) -> bool:
        """디바이스 사용 가능 여부 확인"""
        if device == "cpu":
            return True
        elif device == "mps":
            return self._check_mps_support()
        elif device == "cuda":
            return TORCH_AVAILABLE and torch.cuda.is_available()
        return False

# =============================================================================
# 🔧 최적화 프로필 매니저 (단일 책임)
# =============================================================================

class OptimizationProfileManager:
    """최적화 프로필 관리 전담 클래스"""
    
    def __init__(self, hardware_detector: HardwareDetector):
        self.hardware_detector = hardware_detector
        self._profiles = self._create_base_profiles()
    
    def _create_base_profiles(self) -> Dict[PerformanceClass, OptimizationProfile]:
        """기본 최적화 프로필 생성"""
        return {
            PerformanceClass.ULTRA_HIGH: OptimizationProfile(
                batch_size=8,
                max_workers=20,
                memory_fraction=0.85,
                quality_level="ultra",
                dtype="float32",
                mixed_precision=False,
                enable_checkpointing=False
            ),
            PerformanceClass.HIGH: OptimizationProfile(
                batch_size=6,
                max_workers=16,
                memory_fraction=0.8,
                quality_level="high",
                dtype="float32",
                mixed_precision=False,
                enable_checkpointing=False
            ),
            PerformanceClass.MEDIUM: OptimizationProfile(
                batch_size=4,
                max_workers=12,
                memory_fraction=0.75,
                quality_level="balanced",
                dtype="float32",
                mixed_precision=False,
                enable_checkpointing=True
            ),
            PerformanceClass.LOW: OptimizationProfile(
                batch_size=2,
                max_workers=8,
                memory_fraction=0.6,
                quality_level="balanced",
                dtype="float32",
                mixed_precision=False,
                enable_checkpointing=True
            ),
            PerformanceClass.MINIMAL: OptimizationProfile(
                batch_size=1,
                max_workers=4,
                memory_fraction=0.5,
                quality_level="fast",
                dtype="float32",
                mixed_precision=False,
                enable_checkpointing=True
            )
        }
    
    def get_performance_class(self) -> PerformanceClass:
        """시스템 성능 등급 결정"""
        system_info = self.hardware_detector.get_system_info()
        
        if system_info.is_m3_max and system_info.memory_gb >= 120:
            return PerformanceClass.ULTRA_HIGH
        elif system_info.is_m3_max or (system_info.memory_gb >= 64 and system_info.cpu_cores >= 12):
            return PerformanceClass.HIGH
        elif system_info.memory_gb >= 32 and system_info.cpu_cores >= 8:
            return PerformanceClass.MEDIUM
        elif system_info.memory_gb >= 16:
            return PerformanceClass.LOW
        else:
            return PerformanceClass.MINIMAL
    
    def get_optimization_profile(self, optimization_level: OptimizationLevel) -> OptimizationProfile:
        """최적화 프로필 생성"""
        performance_class = self.get_performance_class()
        base_profile = self._profiles[performance_class]
        
        # 최적화 레벨 조정
        multipliers = {
            OptimizationLevel.SAFE: {"batch_size": 0.5, "memory_fraction": 0.6},
            OptimizationLevel.BALANCED: {"batch_size": 1.0, "memory_fraction": 1.0},
            OptimizationLevel.PERFORMANCE: {"batch_size": 1.2, "memory_fraction": 1.1},
            OptimizationLevel.ULTRA: {"batch_size": 1.5, "memory_fraction": 1.2}
        }
        
        multiplier = multipliers.get(optimization_level, multipliers[OptimizationLevel.BALANCED])
        
        return OptimizationProfile(
            batch_size=max(1, int(base_profile.batch_size * multiplier["batch_size"])),
            max_workers=base_profile.max_workers,
            memory_fraction=min(0.95, base_profile.memory_fraction * multiplier["memory_fraction"]),
            quality_level=base_profile.quality_level,
            dtype=base_profile.dtype,
            mixed_precision=base_profile.mixed_precision,
            enable_checkpointing=base_profile.enable_checkpointing
        )

# =============================================================================
# 🔧 메인 GPU 설정 클래스 (조합 패턴)
# =============================================================================

class GPUConfig:
    """GPU 설정 메인 클래스 - Clean Architecture 적용"""
    
    def __init__(self, 
                 device: Optional[str] = None, 
                 optimization_level: Optional[str] = None,
                 **kwargs):
        """
        GPU 설정 초기화
        
        Args:
            device: 사용할 디바이스 ("cpu", "mps", "cuda", "auto")
            optimization_level: 최적화 레벨 ("safe", "balanced", "performance", "ultra")
        """
        # 의존성 주입
        self._hardware_detector = HardwareDetector()
        self._memory_manager = MemoryManager()
        self._profile_manager = OptimizationProfileManager(self._hardware_detector)
        
        # 초기화 시간
        self._initialization_time = time.time()
        
        # 시스템 정보 수집
        self._system_info = self._hardware_detector.get_system_info()
        self._conda_env = self._hardware_detector.detect_conda_environment()
        
        # 디바이스 설정
        self._device = self._determine_device(device)
        self._device_capabilities = self._hardware_detector.get_device_capabilities(self._device)
        
        # 최적화 설정
        self._optimization_level = self._determine_optimization_level(optimization_level)
        self._optimization_profile = self._profile_manager.get_optimization_profile(self._optimization_level)
        
        # 성능 메트릭
        self._metrics = PerformanceMetrics()
        
        # 환경 최적화 적용
        self._is_initialized = False
        try:
            self._apply_environment_optimizations()
            self._is_initialized = True
        except Exception as e:
            logger.warning(f"환경 최적화 적용 실패: {e}")
    
    # =========================================================================
    # 🔧 초기화 메서드들
    # =========================================================================
    
    def _determine_device(self, device: Optional[str]) -> str:
        """디바이스 결정"""
        if device and device != "auto":
            if self._hardware_detector._is_device_available(device):
                return device
            else:
                logger.warning(f"요청된 디바이스 '{device}' 사용 불가, 자동 선택")
        
        return self._hardware_detector.detect_device()
    
    def _determine_optimization_level(self, level: Optional[str]) -> OptimizationLevel:
        """최적화 레벨 결정"""
        if level:
            try:
                return OptimizationLevel(level.lower())
            except ValueError:
                logger.warning(f"잘못된 최적화 레벨 '{level}', 기본값 사용")
        
        # 환경 변수 확인
        env_level = os.environ.get('OPTIMIZATION_LEVEL', '').lower()
        try:
            return OptimizationLevel(env_level)
        except ValueError:
            pass
        
        # 성능 클래스 기반 자동 결정
        performance_class = self._profile_manager.get_performance_class()
        level_mapping = {
            PerformanceClass.ULTRA_HIGH: OptimizationLevel.ULTRA,
            PerformanceClass.HIGH: OptimizationLevel.PERFORMANCE,
            PerformanceClass.MEDIUM: OptimizationLevel.BALANCED,
            PerformanceClass.LOW: OptimizationLevel.BALANCED,
            PerformanceClass.MINIMAL: OptimizationLevel.SAFE
        }
        
        return level_mapping.get(performance_class, OptimizationLevel.BALANCED)
    
    def _apply_environment_optimizations(self):
        """환경 최적화 적용"""
        if not TORCH_AVAILABLE:
            return
        
        try:
            # PyTorch 스레드 설정
            torch.set_num_threads(self._optimization_profile.max_workers)
            
            # 디바이스별 환경 변수
            if self._device == "mps":
                env_vars = {
                    'PYTORCH_ENABLE_MPS_FALLBACK': '1',
                    'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0'
                }
                
                if self._system_info.is_m3_max:
                    env_vars.update({
                        'OMP_NUM_THREADS': '16',
                        'MKL_NUM_THREADS': '16',
                        'PYTORCH_MPS_PREFER_METAL': '1'
                    })
                
                os.environ.update(env_vars)
                
            elif self._device == "cuda":
                os.environ.update({
                    'CUDA_LAUNCH_BLOCKING': '0',
                    'CUDA_CACHE_DISABLE': '0'
                })
                
                # CUDNN 최적화
                if hasattr(torch.backends.cudnn, 'benchmark'):
                    torch.backends.cudnn.benchmark = True
                if hasattr(torch.backends.cudnn, 'deterministic'):
                    torch.backends.cudnn.deterministic = False
            
            else:  # CPU
                os.environ.update({
                    'OMP_NUM_THREADS': str(self._optimization_profile.max_workers),
                    'MKL_NUM_THREADS': str(self._optimization_profile.max_workers)
                })
            
            # Conda 환경 최적화
            if self._conda_env.is_conda:
                conda_vars = {
                    'CONDA_DEFAULT_ENV': self._conda_env.env_name or 'base',
                    'PYTHONUNBUFFERED': '1'
                }
                if self._conda_env.prefix:
                    conda_vars['CONDA_PREFIX'] = self._conda_env.prefix
                os.environ.update(conda_vars)
            
            # 초기 메모리 정리
            self._memory_manager.cleanup_memory()
            
        except Exception as e:
            logger.warning(f"환경 최적화 적용 실패: {e}")
    
    # =========================================================================
    # 🔧 공개 인터페이스
    # =========================================================================
    
    @property
    def device(self) -> str:
        """현재 디바이스"""
        return self._device
    
    @property
    def device_name(self) -> str:
        """디바이스 이름"""
        return self._device_capabilities.name
    
    @property
    def device_type(self) -> str:
        """디바이스 타입"""
        return self._device
    
    @property
    def memory_gb(self) -> float:
        """시스템 메모리 (GB)"""
        return self._system_info.memory_gb
    
    @property
    def is_m3_max(self) -> bool:
        """M3 Max 여부"""
        return self._system_info.is_m3_max
    
    @property
    def optimization_level(self) -> str:
        """최적화 레벨"""
        return self._optimization_level.value
    
    @property
    def is_initialized(self) -> bool:
        """초기화 완료 여부"""
        return self._is_initialized
    
    @property
    def float_compatibility_mode(self) -> bool:
        """Float 호환성 모드 (항상 True)"""
        return True
    
    def get_device(self) -> str:
        """디바이스 반환 (호환성)"""
        return self.device
    
    def get_device_name(self) -> str:
        """디바이스 이름 반환 (호환성)"""
        return self.device_name
    
    def get_memory_info(self) -> Dict[str, Any]:
        """메모리 정보 조회"""
        return self._memory_manager.get_memory_info()
    
    def cleanup_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """메모리 정리"""
        result = self._memory_manager.cleanup_memory(aggressive)
        
        # 메트릭 업데이트
        if result["success"]:
            self._update_metrics("memory_cleanup", result.get("duration", 0) * 1000, True)
        else:
            self._update_metrics("memory_cleanup", 0, False)
        
        return result
    
    def get_optimal_settings(self) -> Dict[str, Any]:
        """최적화된 설정 반환"""
        return {
            "device_config": self._create_model_config(),
            "optimization_settings": self._create_optimization_settings(),
            "device_info": self._create_device_info(),
            "last_updated": time.time()
        }
    
    def get_device_capabilities(self) -> Dict[str, Any]:
        """디바이스 기능 정보 반환"""
        caps = self._device_capabilities
        return {
            "device": caps.device_type,
            "name": caps.name,
            "memory_gb": caps.memory_gb,
            "supports_fp16": caps.supports_fp16,
            "supports_fp32": caps.supports_fp32,
            "unified_memory": caps.unified_memory,
            "max_batch_size": caps.max_batch_size,
            "recommended_image_size": caps.recommended_image_size,
            "tensor_cores": caps.tensor_cores,
            "neural_engine": caps.neural_engine,
            "optimization_level": self.optimization_level,
            "performance_class": self._profile_manager.get_performance_class().value,
            "pytorch_version": TORCH_VERSION,
            "float_compatibility_mode": True,
            "conda_environment": self._conda_env.is_conda,
            "last_updated": time.time()
        }
    
    def benchmark_device(self, duration_seconds: int = 10) -> Dict[str, Any]:
        """디바이스 벤치마크"""
        if not TORCH_AVAILABLE:
            return {
                "error": "PyTorch not available",
                "device": self.device
            }
        
        try:
            start_time = time.time()
            device = torch.device(self.device)
            
            # 메모리 할당 테스트
            memory_tests = []
            for size in [100, 500, 1000]:
                try:
                    test_start = time.time()
                    tensor = torch.randn(size, size, device=device)
                    alloc_time = (time.time() - test_start) * 1000
                    memory_mb = (tensor.nelement() * tensor.element_size()) / (1024**2)
                    
                    memory_tests.append({
                        "size": f"{size}x{size}",
                        "allocation_time_ms": round(alloc_time, 2),
                        "memory_mb": round(memory_mb, 2)
                    })
                    del tensor
                except Exception as e:
                    memory_tests.append({
                        "size": f"{size}x{size}",
                        "error": str(e)[:100]
                    })
            
            # 연산 속도 테스트
            compute_tests = []
            if memory_tests and "error" not in memory_tests[0]:
                test_tensor = torch.randn(500, 500, device=device)
                
                operations = [
                    ("matrix_multiply", lambda x: torch.mm(x, x.t())),
                    ("elementwise_ops", lambda x: x * 2 + 1),
                    ("reduction_ops", lambda x: torch.sum(x, dim=0))
                ]
                
                for op_name, op_func in operations:
                    times = []
                    try:
                        for _ in range(3):  # 3회 반복
                            op_start = time.time()
                            result = op_func(test_tensor)
                            times.append((time.time() - op_start) * 1000)
                            del result
                        
                        compute_tests.append({
                            "operation": op_name,
                            "avg_time_ms": round(sum(times) / len(times), 2),
                            "min_time_ms": round(min(times), 2),
                            "max_time_ms": round(max(times), 2)
                        })
                    except Exception as e:
                        compute_tests.append({
                            "operation": op_name,
                            "error": str(e)[:100]
                        })
                
                del test_tensor
            
            # 메모리 정리 성능 테스트
            cleanup_start = time.time()
            cleanup_result = self.cleanup_memory(aggressive=True)
            cleanup_time = (time.time() - cleanup_start) * 1000
            
            total_duration = time.time() - start_time
            
            return {
                "device": self.device,
                "device_name": self.device_name,
                "benchmark_duration_seconds": round(total_duration, 2),
                "memory_tests": memory_tests,
                "compute_tests": compute_tests,
                "memory_cleanup_time_ms": round(cleanup_time, 2),
                "cleanup_result": cleanup_result,
                "memory_info": self.get_memory_info(),
                "device_capabilities": self.get_device_capabilities(),
                "timestamp": time.time(),
                "benchmark_success": True
            }
            
        except Exception as e:
            return {
                "device": self.device,
                "benchmark_success": False,
                "error": str(e)[:300],
                "timestamp": time.time()
            }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """전체 성능 리포트"""
        return {
            "system_info": {
                "platform": self._system_info.platform,
                "machine": self._system_info.machine,
                "python_version": self._system_info.python_version,
                "pytorch_version": self._system_info.pytorch_version,
                "memory_gb": self._system_info.memory_gb,
                "cpu_cores": self._system_info.cpu_cores,
                "is_m3_max": self._system_info.is_m3_max
            },
            "conda_environment": {
                "is_conda": self._conda_env.is_conda,
                "env_name": self._conda_env.env_name,
                "package_manager": self._conda_env.package_manager,
                "optimization_level": self._conda_env.optimization_level
            },
            "device_info": {
                "device": self.device,
                "name": self.device_name,
                "capabilities": self.get_device_capabilities()
            },
            "optimization": {
                "level": self.optimization_level,
                "performance_class": self._profile_manager.get_performance_class().value,
                "profile": {
                    "batch_size": self._optimization_profile.batch_size,
                    "max_workers": self._optimization_profile.max_workers,
                    "memory_fraction": self._optimization_profile.memory_fraction,
                    "quality_level": self._optimization_profile.quality_level
                }
            },
            "metrics": {
                "memory_cleanup_count": self._metrics.memory_cleanup_count,
                "memory_cleanup_avg_ms": round(self._metrics.memory_cleanup_avg_ms, 2),
                "error_count": self._metrics.error_count,
                "uptime_hours": round((time.time() - self._metrics.last_reset) / 3600, 2)
            },
            "memory_info": self.get_memory_info(),
            "initialization_time": self._initialization_time,
            "is_initialized": self._is_initialized,
            "generation_time": time.time()
        }
    
    # =========================================================================
    # 🔧 딕셔너리 스타일 인터페이스 (호환성)
    # =========================================================================
    
    def get(self, key: str, default: Any = None) -> Any:
        """딕셔너리 스타일 접근"""
        # 직접 속성 매핑
        direct_attrs = {
            'device': self.device,
            'device_name': self.device_name,
            'device_type': self.device_type,
            'memory_gb': self.memory_gb,
            'is_m3_max': self.is_m3_max,
            'optimization_level': self.optimization_level,
            'is_initialized': self.is_initialized,
            'float_compatibility_mode': self.float_compatibility_mode,
            'pytorch_version': TORCH_VERSION,
            'numpy_version': NUMPY_VERSION,
            'conda_environment': self._conda_env.is_conda,
            'conda_env_name': self._conda_env.env_name,
            'performance_class': self._profile_manager.get_performance_class().value
        }
        
        if key in direct_attrs:
            return direct_attrs[key]
        
        # 설정 딕셔너리에서 검색
        try:
            model_config = self._create_model_config()
            if key in model_config:
                return model_config[key]
        except Exception:
            pass
        
        # 객체 속성에서 검색
        if hasattr(self, key):
            attr = getattr(self, key)
            # 메서드는 호출하지 않음
            if callable(attr):
                return default
            return attr
        
        return default
    
    def __getitem__(self, key: str) -> Any:
        """딕셔너리 스타일 접근"""
        result = self.get(key)
        if result is None:
            raise KeyError(f"Key '{key}' not found in GPUConfig")
        return result
    
    def __contains__(self, key: str) -> bool:
        """in 연산자 지원"""
        return self.get(key) is not None
    
    def keys(self) -> List[str]:
        """사용 가능한 키 목록"""
        return [
            'device', 'device_name', 'device_type', 'memory_gb', 'is_m3_max',
            'optimization_level', 'is_initialized', 'float_compatibility_mode',
            'pytorch_version', 'numpy_version', 'conda_environment', 'conda_env_name',
            'performance_class', 'batch_size', 'max_workers', 'memory_fraction',
            'quality_level', 'dtype'
        ]
    
    def items(self) -> List[Tuple[str, Any]]:
        """키-값 쌍 반환"""
        return [(key, self.get(key)) for key in self.keys()]
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return dict(self.items())
    
    # =========================================================================
    # 🔧 내부 헬퍼 메서드들
    # =========================================================================
    
    def _create_model_config(self) -> Dict[str, Any]:
        """모델 설정 딕셔너리 생성"""
        return {
            "device": self.device,
            "device_name": self.device_name,
            "dtype": self._optimization_profile.dtype,
            "batch_size": self._optimization_profile.batch_size,
            "max_workers": self._optimization_profile.max_workers,
            "memory_fraction": self._optimization_profile.memory_fraction,
            "optimization_level": self.optimization_level,
            "quality_level": self._optimization_profile.quality_level,
            "float_compatibility_mode": True,
            "mps_fallback_enabled": self.device == "mps",
            "pytorch_version": TORCH_VERSION,
            "numpy_version": NUMPY_VERSION,
            "m3_max_optimized": self.is_m3_max,
            "conda_environment": self._conda_env.is_conda
        }
    
    def _create_optimization_settings(self) -> Dict[str, Any]:
        """최적화 설정 딕셔너리 생성"""
        profile = self._optimization_profile
        return {
            "batch_size": profile.batch_size,
            "max_workers": profile.max_workers,
            "memory_fraction": profile.memory_fraction,
            "quality_level": profile.quality_level,
            "dtype": profile.dtype,
            "mixed_precision": profile.mixed_precision,
            "enable_checkpointing": profile.enable_checkpointing,
            "optimization_level": self.optimization_level,
            "performance_class": self._profile_manager.get_performance_class().value
        }
    
    def _create_device_info(self) -> Dict[str, Any]:
        """디바이스 정보 딕셔너리 생성"""
        return {
            "device": self.device,
            "device_name": self.device_name,
            "device_type": self.device_type,
            "memory_gb": self.memory_gb,
            "is_m3_max": self.is_m3_max,
            "capabilities": self.get_device_capabilities(),
            "system_info": {
                "platform": self._system_info.platform,
                "machine": self._system_info.machine,
                "cpu_cores": self._system_info.cpu_cores,
                "python_version": self._system_info.python_version
            },
            "conda_environment": {
                "is_conda": self._conda_env.is_conda,
                "env_name": self._conda_env.env_name,
                "package_manager": self._conda_env.package_manager
            },
            "pytorch_available": TORCH_AVAILABLE,
            "pytorch_version": TORCH_VERSION,
            "numpy_available": NUMPY_AVAILABLE,
            "numpy_version": NUMPY_VERSION,
            "psutil_available": PSUTIL_AVAILABLE
        }
    
    def _update_metrics(self, operation: str, duration_ms: float, success: bool):
        """메트릭 업데이트"""
        try:
            if operation == "memory_cleanup" and success:
                count = self._metrics.memory_cleanup_count
                avg = self._metrics.memory_cleanup_avg_ms
                self._metrics.memory_cleanup_avg_ms = (avg * count + duration_ms) / (count + 1)
                self._metrics.memory_cleanup_count += 1
            elif not success:
                self._metrics.error_count += 1
        except Exception:
            pass  # 메트릭 업데이트 실패는 무시

# =============================================================================
# 🔧 유틸리티 함수들
# =============================================================================

@lru_cache(maxsize=1)
def get_gpu_config(**kwargs) -> GPUConfig:
    """GPU 설정 싱글톤 팩토리"""
    try:
        return GPUConfig(**kwargs)
    except Exception as e:
        logger.error(f"GPUConfig 생성 실패: {e}")
        return _create_fallback_gpu_config()

def _create_fallback_gpu_config():
    """폴백 GPU 설정 객체"""
    class FallbackGPUConfig:
        def __init__(self):
            self.device = "cpu"
            self.device_name = "CPU (Fallback)"
            self.device_type = "cpu"
            self.memory_gb = 16.0
            self.is_m3_max = False
            self.optimization_level = "safe"
            self.is_initialized = False
            self.float_compatibility_mode = True
        
        def get(self, key, default=None):
            return getattr(self, key, default)
        
        def get_device(self):
            return self.device
        
        def get_device_name(self):
            return self.device_name
        
        def get_memory_info(self):
            return {"total_gb": self.memory_gb, "device": self.device, "fallback_mode": True}
        
        def cleanup_memory(self, aggressive=False):
            collected = gc.collect()
            return {"success": True, "method": "fallback_gc", "device": "cpu", "collected_objects": collected}
        
        def get_optimal_settings(self):
            return {"device_config": {"device": "cpu", "batch_size": 1, "fallback_mode": True}}
        
        def get_device_capabilities(self):
            return {"device": "cpu", "fallback_mode": True, "error": "GPUConfig initialization failed"}
        
        def __getitem__(self, key):
            return self.get(key)
        
        def __contains__(self, key):
            return self.get(key) is not None
    
    return FallbackGPUConfig()

# 편의 함수들
def get_device_config() -> Dict[str, Any]:
    """디바이스 설정 반환"""
    try:
        config = get_gpu_config()
        return config._create_model_config()
    except Exception as e:
        return {"error": str(e)[:200], "device": "cpu", "fallback_mode": True}

def get_model_config() -> Dict[str, Any]:
    """모델 설정 반환"""
    return get_device_config()

def get_device_info() -> Dict[str, Any]:
    """디바이스 정보 반환"""
    try:
        config = get_gpu_config()
        return config._create_device_info()
    except Exception as e:
        return {"error": str(e)[:200], "device": "cpu", "fallback_mode": True}

def get_device() -> str:
    """현재 디바이스 반환"""
    try:
        return get_gpu_config().device
    except:
        return "cpu"

def get_device_name() -> str:
    """디바이스 이름 반환"""
    try:
        return get_gpu_config().device_name
    except:
        return "CPU (Fallback)"

def is_m3_max() -> bool:
    """M3 Max 여부 확인"""
    try:
        return get_gpu_config().is_m3_max
    except:
        return False

def get_optimal_settings() -> Dict[str, Any]:
    """최적화된 설정 반환"""
    try:
        return get_gpu_config().get_optimal_settings()
    except Exception as e:
        return {"error": str(e)[:200], "fallback_config": {"device": "cpu", "batch_size": 1}}

def get_device_capabilities() -> Dict[str, Any]:
    """디바이스 기능 정보 반환"""
    try:
        return get_gpu_config().get_device_capabilities()
    except Exception as e:
        return {"error": str(e)[:200], "device": "cpu", "fallback_mode": True}

def cleanup_device_memory(aggressive: bool = False) -> Dict[str, Any]:
    """디바이스 메모리 정리"""
    try:
        return get_gpu_config().cleanup_memory(aggressive=aggressive)
    except Exception as e:
        # 폴백: 기본 가비지 컬렉션
        collected = gc.collect()
        return {"success": True, "method": "fallback_gc", "device": "cpu", "collected_objects": collected, "error": str(e)[:100]}

def benchmark_device(duration_seconds: int = 10) -> Dict[str, Any]:
    """디바이스 벤치마크"""
    try:
        return get_gpu_config().benchmark_device(duration_seconds)
    except Exception as e:
        return {"benchmark_success": False, "error": str(e)[:200], "device": "cpu"}

def get_performance_report() -> Dict[str, Any]:
    """전체 성능 리포트"""
    try:
        return get_gpu_config().get_performance_report()
    except Exception as e:
        return {
            "error": str(e)[:200],
            "basic_info": {
                "pytorch_available": TORCH_AVAILABLE,
                "numpy_available": NUMPY_AVAILABLE,
                "psutil_available": PSUTIL_AVAILABLE
            },
            "timestamp": time.time()
        }

def reset_gpu_config():
    """GPU 설정 리셋"""
    try:
        get_gpu_config.cache_clear()
        return {"success": True, "message": "GPU config reset"}
    except Exception as e:
        return {"success": False, "error": str(e)[:200]}

# =============================================================================
# 🔧 전역 GPU 설정 매니저 생성
# =============================================================================

try:
    # 메인 GPU 설정 객체 생성
    gpu_config = get_gpu_config()
    
    # 전역 변수 설정
    DEVICE = gpu_config.device
    DEVICE_NAME = gpu_config.device_name
    DEVICE_TYPE = gpu_config.device_type
    MODEL_CONFIG = gpu_config._create_model_config()
    DEVICE_INFO = gpu_config._create_device_info()
    IS_M3_MAX = gpu_config.is_m3_max
    OPTIMIZATION_LEVEL = gpu_config.optimization_level
    CONDA_ENV = gpu_config._conda_env
    
    # 성공 메시지
    if IS_M3_MAX:
        print(f"🍎 M3 Max ({DEVICE}) 완전 최적화 모드 활성화")
        print(f"💾 통합 메모리: {gpu_config.memory_gb}GB | 최적화: {OPTIMIZATION_LEVEL}")
        if CONDA_ENV.is_conda:
            print(f"🐍 conda 환경: {CONDA_ENV.env_name} | 패키지 관리자: {CONDA_ENV.package_manager}")
    else:
        print(f"✅ GPU 설정 완전 로드 - 디바이스: {DEVICE} | 최적화: {OPTIMIZATION_LEVEL}")

except Exception as e:
    print(f"⚠️ GPU 설정 초기화 실패: {str(e)[:100]}")
    
    # 폴백 설정
    DEVICE = "cpu"
    DEVICE_NAME = "CPU (Fallback)"
    DEVICE_TYPE = "cpu"
    MODEL_CONFIG = {"device": "cpu", "dtype": "float32", "batch_size": 1, "fallback_mode": True}
    DEVICE_INFO = {"device": "cpu", "error": "GPU config initialization failed", "fallback_mode": True}
    IS_M3_MAX = False
    OPTIMIZATION_LEVEL = "safe"
    CONDA_ENV = CondaEnvironment(is_conda=False)
    
    # 폴백 GPU 설정 객체
    gpu_config = _create_fallback_gpu_config()

# =============================================================================
# 🔧 Export 리스트
# =============================================================================

__all__ = [
    # 메인 클래스들
    'GPUConfig', 'HardwareDetector', 'MemoryManager', 'OptimizationProfileManager',
    
    # 데이터 클래스들
    'SystemInfo', 'CondaEnvironment', 'DeviceCapabilities', 'PerformanceMetrics', 'OptimizationProfile',
    
    # 열거형들
    'DeviceType', 'OptimizationLevel', 'PerformanceClass',
    
    # 전역 인스턴스 및 변수들
    'gpu_config', 'DEVICE', 'DEVICE_NAME', 'DEVICE_TYPE', 
    'MODEL_CONFIG', 'DEVICE_INFO', 'IS_M3_MAX', 'OPTIMIZATION_LEVEL', 'CONDA_ENV',
    
    # 팩토리 및 설정 함수들
    'get_gpu_config', 'get_device_config', 'get_model_config', 'get_device_info',
    'get_device', 'get_device_name', 'is_m3_max', 'get_optimal_settings', 
    'get_device_capabilities', 'reset_gpu_config',
    
    # 메모리 관리 함수들
    'cleanup_device_memory',
    
    # 성능 및 벤치마크 함수들
    'benchmark_device', 'get_performance_report',
    
    # 상수 및 버전 정보
    'TORCH_AVAILABLE', 'TORCH_VERSION', 'TORCH_MAJOR', 'TORCH_MINOR',
    'NUMPY_AVAILABLE', 'NUMPY_VERSION', 'PSUTIL_AVAILABLE', 'PSUTIL_VERSION'
]

# 모듈 완료 메시지
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🍎 MyCloset AI GPU Config 시스템 완전 로드 완료")
    print("="*80)
    
    # 간단한 시스템 정보 출력
    try:
        config = get_gpu_config()
        print(f"디바이스: {config.device} ({config.device_name})")
        print(f"메모리: {config.memory_gb}GB")
        print(f"최적화 레벨: {config.optimization_level}")
        print(f"M3 Max: {'✅' if config.is_m3_max else '❌'}")
        print(f"conda 환경: {'✅' if config._conda_env.is_conda else '❌'}")
        print(f"PyTorch: {TORCH_VERSION if TORCH_AVAILABLE else '❌'}")
        print(f"초기화 완료: {'✅' if config.is_initialized else '❌'}")
        
        # 성능 클래스 출력
        performance_class = config._profile_manager.get_performance_class()
        print(f"성능 클래스: {performance_class.value}")
        
        # 최적화 프로필 요약
        profile = config._optimization_profile
        print(f"배치 크기: {profile.batch_size} | 워커: {profile.max_workers} | 메모리: {profile.memory_fraction:.1%}")
        
    except Exception as e:
        print(f"❌ 시스템 정보 출력 실패: {e}")
    
    print("="*80)