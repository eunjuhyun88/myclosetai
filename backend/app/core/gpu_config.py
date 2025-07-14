# app/core/gpu_config.py
"""
최적 GPU 설정 시스템 - 지능적 디바이스 관리
- 자동 디바이스 감지 및 최적화
- M3 Max 특화 설정
- 메모리 최적화
- 성능 모니터링
"""
import os
import platform
import subprocess
import logging
import psutil
from typing import Dict, Any, Optional, List, Tuple, Union
from functools import lru_cache
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# ===============================================================
# 🎯 최적 GPU 설정 베이스 클래스
# ===============================================================

@dataclass
class DeviceInfo:
    """디바이스 정보 데이터 클래스"""
    device: str
    device_type: str
    name: str
    memory_gb: float
    compute_capability: Optional[str] = None
    driver_version: Optional[str] = None
    is_available: bool = True
    optimization_level: str = 'balanced'
    supports_mixed_precision: bool = False
    supports_dynamic_batching: bool = False

class OptimalGPUConfigBase(ABC):
    """
    🎯 최적화된 GPU 설정 베이스 클래스
    - 자동 디바이스 감지
    - 지능적 최적화
    - 메모리 관리
    - 성능 모니터링
    """

    def __init__(
        self,
        preferred_device: Optional[str] = None,  # 선호 디바이스
        memory_fraction: float = 0.8,  # 메모리 사용 비율
        enable_optimization: bool = True,  # 최적화 활성화
        **kwargs  # 확장 파라미터
    ):
        """
        ✅ 최적 GPU 설정 생성자

        Args:
            preferred_device: 선호하는 디바이스 (None=자동감지)
            memory_fraction: GPU 메모리 사용 비율 (0.1~1.0)
            enable_optimization: 최적화 활성화 여부
            **kwargs: 확장 파라미터들
                - force_cpu: bool = False
                - mixed_precision: bool = auto
                - enable_profiling: bool = False
                - memory_growth: bool = True
                - 기타...
        """
        self.preferred_device = preferred_device
        self.memory_fraction = max(0.1, min(1.0, memory_fraction))
        self.enable_optimization = enable_optimization
        self.kwargs = kwargs
        
        # 1. 💡 시스템 정보 수집
        self.system_info = self._collect_system_info()
        
        # 2. 🖥️ 사용 가능한 디바이스 스캔
        self.available_devices = self._scan_available_devices()
        
        # 3. 🎯 최적 디바이스 선택
        self.device_info = self._select_optimal_device()
        
        # 4. ⚙️ 디바이스별 설정 적용
        self._configure_device()
        
        # 5. 🚀 최적화 적용
        if self.enable_optimization:
            self._apply_optimizations()
        
        # 6. 📊 모니터링 설정
        self._setup_monitoring()
        
        logger.info(f"🎯 GPU 설정 완료 - 디바이스: {self.device_info.device} ({self.device_info.name})")

    def _collect_system_info(self) -> Dict[str, Any]:
        """🖥️ 시스템 정보 수집"""
        return {
            'platform': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'cpu_count': os.cpu_count() or 4,
            'total_memory_gb': self._get_total_memory_gb(),
            'available_memory_gb': self._get_available_memory_gb(),
            'is_m3_max': self._detect_m3_max(),
            'is_container': self._detect_container(),
            'python_version': platform.python_version()
        }

    def _get_total_memory_gb(self) -> float:
        """총 메모리 용량 (GB)"""
        try:
            return psutil.virtual_memory().total / (1024**3)
        except:
            return 16.0  # 기본값

    def _get_available_memory_gb(self) -> float:
        """사용 가능한 메모리 (GB)"""
        try:
            return psutil.virtual_memory().available / (1024**3)
        except:
            return 8.0  # 기본값

    def _detect_m3_max(self) -> bool:
        """🍎 M3 Max 칩 감지"""
        if platform.system() != 'Darwin':
            return False
        
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'], 
                capture_output=True, text=True, timeout=5
            )
            cpu_info = result.stdout.strip()
            return any(chip in cpu_info for chip in ['M3', 'M2', 'M1']) and 'Max' in cpu_info
        except:
            return False

    def _detect_container(self) -> bool:
        """🐳 컨테이너 환경 감지"""
        indicators = [
            os.path.exists('/.dockerenv'),
            os.getenv('KUBERNETES_SERVICE_HOST') is not None,
            os.getenv('CONTAINER') is not None
        ]
        return any(indicators)

    @abstractmethod
    def _scan_available_devices(self) -> List[DeviceInfo]:
        """사용 가능한 디바이스 스캔 (서브클래스에서 구현)"""
        pass

    def _select_optimal_device(self) -> DeviceInfo:
        """🎯 최적 디바이스 선택"""
        
        # 강제 CPU 모드
        if self.kwargs.get('force_cpu', False):
            return self._create_cpu_device_info()
        
        # 선호 디바이스가 있고 사용 가능한 경우
        if self.preferred_device:
            for device in self.available_devices:
                if device.device == self.preferred_device and device.is_available:
                    return device
        
        # 자동 선택: 성능 순서대로 정렬
        if self.available_devices:
            # M3 Max MPS > NVIDIA CUDA > CPU 순으로 우선순위
            priority_order = ['mps', 'cuda', 'cpu']
            
            for device_type in priority_order:
                for device in self.available_devices:
                    if device.device == device_type and device.is_available:
                        return device
        
        # 폴백: CPU
        return self._create_cpu_device_info()

    def _create_cpu_device_info(self) -> DeviceInfo:
        """CPU 디바이스 정보 생성"""
        return DeviceInfo(
            device='cpu',
            device_type='cpu',
            name=f"CPU ({self.system_info['cpu_count']} cores)",
            memory_gb=self.system_info['available_memory_gb'],
            is_available=True,
            optimization_level='basic',
            supports_mixed_precision=False,
            supports_dynamic_batching=True
        )

    @abstractmethod
    def _configure_device(self):
        """디바이스별 설정 (서브클래스에서 구현)"""
        pass

    @abstractmethod
    def _apply_optimizations(self):
        """최적화 적용 (서브클래스에서 구현)"""
        pass

    def _setup_monitoring(self):
        """📊 모니터링 설정"""
        self.monitoring_enabled = self.kwargs.get('enable_profiling', False)
        self.memory_stats = {
            'allocated': 0,
            'cached': 0,
            'max_allocated': 0
        }

    # 공통 유틸리티 메서드들
    def get_device(self) -> str:
        """현재 디바이스 반환"""
        return self.device_info.device

    def get_device_info(self) -> DeviceInfo:
        """디바이스 정보 반환"""
        return self.device_info

    def get_memory_info(self) -> Dict[str, float]:
        """메모리 정보 반환"""
        return {
            'total_gb': self.device_info.memory_gb,
            'allocated_gb': self.memory_stats['allocated'],
            'available_gb': self.device_info.memory_gb - self.memory_stats['allocated'],
            'memory_fraction': self.memory_fraction
        }

    def optimize_memory(self):
        """메모리 최적화"""
        if self.device_info.device == 'cuda':
            self._optimize_cuda_memory()
        elif self.device_info.device == 'mps':
            self._optimize_mps_memory()
        else:
            self._optimize_cpu_memory()

    def _optimize_cuda_memory(self):
        """CUDA 메모리 최적화"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    def _optimize_mps_memory(self):
        """MPS 메모리 최적화"""
        try:
            import torch
            if torch.backends.mps.is_available():
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
                elif hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
        except ImportError:
            pass

    def _optimize_cpu_memory(self):
        """CPU 메모리 최적화"""
        import gc
        gc.collect()

    def check_memory_available(self, required_gb: float = 4.0) -> bool:
        """메모리 가용성 체크"""
        available_gb = self.device_info.memory_gb - self.memory_stats['allocated']
        return available_gb >= required_gb

# ===============================================================
# 🎯 PyTorch GPU 설정 클래스
# ===============================================================

class PyTorchGPUConfig(OptimalGPUConfigBase):
    """
    🎯 PyTorch 전용 GPU 설정
    - PyTorch 백엔드 최적화
    - CUDA/MPS 설정
    - 메모리 관리
    - M3 Max 특화
    """

    def _scan_available_devices(self) -> List[DeviceInfo]:
        """PyTorch 디바이스 스캔"""
        devices = []
        
        try:
            import torch
            
            # MPS (Apple Silicon) 확인
            if torch.backends.mps.is_available():
                memory_gb = self._estimate_mps_memory()
                devices.append(DeviceInfo(
                    device='mps',
                    device_type='apple_silicon',
                    name=f"Apple Silicon MPS ({memory_gb:.1f}GB)",
                    memory_gb=memory_gb,
                    is_available=True,
                    optimization_level='ultra' if self.system_info['is_m3_max'] else 'high',
                    supports_mixed_precision=True,
                    supports_dynamic_batching=True
                ))
            
            # CUDA 확인
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / (1024**3)
                    
                    devices.append(DeviceInfo(
                        device='cuda',
                        device_type='nvidia_gpu',
                        name=f"{props.name} ({memory_gb:.1f}GB)",
                        memory_gb=memory_gb,
                        compute_capability=f"{props.major}.{props.minor}",
                        is_available=True,
                        optimization_level='high',
                        supports_mixed_precision=props.major >= 7,  # Tensor Cores
                        supports_dynamic_batching=True
                    ))
            
        except ImportError:
            logger.warning("PyTorch 없이 CPU만 사용 가능")
        
        # CPU는 항상 추가
        devices.append(self._create_cpu_device_info())
        
        return devices

    def _estimate_mps_memory(self) -> float:
        """MPS 메모리 추정"""
        # Apple Silicon의 통합 메모리 시스템
        # 일반적으로 시스템 메모리의 80% 정도를 GPU가 활용 가능
        system_memory = self.system_info['total_memory_gb']
        
        if self.system_info['is_m3_max']:
            # M3 Max는 더 많은 GPU 메모리 활용 가능
            return min(system_memory * 0.85, 128.0)
        else:
            return min(system_memory * 0.75, 64.0)

    def _configure_device(self):
        """PyTorch 디바이스 설정"""
        try:
            import torch
            
            if self.device_info.device == 'mps':
                self._configure_mps()
            elif self.device_info.device == 'cuda':
                self._configure_cuda()
            else:
                self._configure_cpu()
                
        except ImportError:
            logger.warning("PyTorch 설정 실패 - 라이브러리 없음")

    def _configure_mps(self):
        """MPS 설정"""
        try:
            import torch
            
            # MPS 폴백 활성화
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            
            # M3 Max 특화 설정
            if self.system_info['is_m3_max']:
                # 고성능 모드
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
                
                # 스레드 최적화 (14코어 M3 Max)
                optimal_threads = min(8, self.system_info['cpu_count'])
                torch.set_num_threads(optimal_threads)
                
                logger.info(f"🍎 M3 Max MPS 최적화 완료 - 스레드: {optimal_threads}")
            
            # 메모리 관리 최적화
            if hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
                
        except Exception as e:
            logger.warning(f"MPS 설정 실패: {e}")

    def _configure_cuda(self):
        """CUDA 설정"""
        try:
            import torch
            
            # 메모리 fraction 설정
            if self.memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
            
            # 메모리 growth 활성화 (TensorFlow 스타일)
            if self.kwargs.get('memory_growth', True):
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            # CuDNN 최적화
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # 멀티 GPU 지원
            if torch.cuda.device_count() > 1:
                logger.info(f"🎮 다중 GPU 감지: {torch.cuda.device_count()}개")
            
            logger.info(f"🎮 CUDA 최적화 완료 - 메모리 fraction: {self.memory_fraction}")
            
        except Exception as e:
            logger.warning(f"CUDA 설정 실패: {e}")

    def _configure_cpu(self):
        """CPU 설정"""
        try:
            import torch
            
            # CPU 스레드 최적화
            if self.enable_optimization:
                optimal_threads = min(self.system_info['cpu_count'], 8)
                torch.set_num_threads(optimal_threads)
                
                # OpenMP 스레드 설정
                os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
                os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
                
                logger.info(f"⚡ CPU 최적화 완료 - 스레드: {optimal_threads}")
                
        except Exception as e:
            logger.warning(f"CPU 설정 실패: {e}")

    def _apply_optimizations(self):
        """PyTorch 최적화 적용"""
        try:
            import torch
            
            # Mixed Precision 설정
            if self.device_info.supports_mixed_precision and self.kwargs.get('mixed_precision', True):
                logger.info("🚀 Mixed Precision 활성화")
            
            # JIT 컴파일 최적화
            if self.enable_optimization:
                torch.jit.set_fusion_strategy([('STATIC', 2), ('DYNAMIC', 2)])
                
            # 메모리 최적화
            if self.device_info.device != 'cpu':
                self.optimize_memory()
                
        except Exception as e:
            logger.warning(f"PyTorch 최적화 실패: {e}")

    def get_optimal_batch_size(self, base_size: int = 4) -> int:
        """최적 배치 크기 계산"""
        if self.device_info.device == 'mps' and self.system_info['is_m3_max']:
            # M3 Max는 더 큰 배치 가능
            memory_multiplier = self.device_info.memory_gb / 32.0
            return int(base_size * min(memory_multiplier, 4.0))
        elif self.device_info.device == 'cuda':
            # CUDA는 메모리에 따라
            memory_multiplier = self.device_info.memory_gb / 16.0
            return int(base_size * min(memory_multiplier, 2.0))
        else:
            # CPU는 보수적으로
            return max(1, base_size // 2)

    def get_optimal_workers(self) -> int:
        """최적 데이터로더 워커 수"""
        if self.device_info.device == 'cpu':
            return min(4, self.system_info['cpu_count'])
        else:
            # GPU 사용시 CPU 코어의 25% 정도
            return min(4, max(1, self.system_info['cpu_count'] // 4))

# ===============================================================
# 🎯 전역 GPU 설정 관리자
# ===============================================================

class GPUConfigManager:
    """
    🎯 전역 GPU 설정 관리자
    - 싱글톤 패턴
    - 캐싱 지원
    - 동적 재설정
    """

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_config(
        self, 
        framework: str = 'pytorch',
        **kwargs
    ) -> OptimalGPUConfigBase:
        """GPU 설정 반환 (캐시됨)"""
        
        cache_key = f"{framework}_{hash(frozenset(kwargs.items()))}"
        
        if self._config is None or getattr(self._config, '_cache_key', None) != cache_key:
            if framework.lower() == 'pytorch':
                self._config = PyTorchGPUConfig(**kwargs)
                self._config._cache_key = cache_key
            else:
                raise ValueError(f"지원하지 않는 프레임워크: {framework}")
        
        return self._config

    def reset_config(self):
        """설정 초기화"""
        self._config = None

    def get_device_summary(self) -> Dict[str, Any]:
        """디바이스 요약 정보"""
        if self._config is None:
            self._config = self.get_config()
        
        return {
            'current_device': self._config.get_device(),
            'device_info': self._config.get_device_info(),
            'memory_info': self._config.get_memory_info(),
            'system_info': self._config.system_info,
            'optimization_enabled': self._config.enable_optimization
        }

# ===============================================================
# 🎯 전역 인스턴스 및 편의 함수들
# ===============================================================

# 전역 GPU 설정 매니저
gpu_config_manager = GPUConfigManager()

@lru_cache()
def get_gpu_config(**kwargs) -> OptimalGPUConfigBase:
    """GPU 설정 반환 (캐시됨)"""
    return gpu_config_manager.get_config(**kwargs)

# 기본 GPU 설정
gpu_config = get_gpu_config()

# 편의 상수들 (하위 호환성)
DEVICE = gpu_config.get_device()
DEVICE_INFO = gpu_config.get_device_info()
DEVICE_TYPE = DEVICE_INFO.device_type
USE_GPU = DEVICE != 'cpu'
IS_M3_MAX = gpu_config.system_info['is_m3_max']
MEMORY_GB = DEVICE_INFO.memory_gb

# 모델 설정 (하위 호환성)
MODEL_CONFIG = {
    'device': DEVICE,
    'batch_size': gpu_config.get_optimal_batch_size() if hasattr(gpu_config, 'get_optimal_batch_size') else 4,
    'num_workers': gpu_config.get_optimal_workers() if hasattr(gpu_config, 'get_optimal_workers') else 2,
    'mixed_precision': DEVICE_INFO.supports_mixed_precision,
    'memory_fraction': gpu_config.memory_fraction
}

# 편의 함수들
def get_device() -> str:
    """현재 디바이스 반환"""
    return DEVICE

def get_device_info() -> DeviceInfo:
    """디바이스 정보 반환"""
    return DEVICE_INFO

def get_optimal_settings() -> Dict[str, Any]:
    """최적 설정 반환"""
    return MODEL_CONFIG

def optimize_memory():
    """메모리 최적화"""
    gpu_config.optimize_memory()

def check_memory_available(required_gb: float = 4.0) -> bool:
    """메모리 가용성 체크"""
    return gpu_config.check_memory_available(required_gb)

def get_memory_info() -> Dict[str, float]:
    """메모리 정보 반환"""
    return gpu_config.get_memory_info()

def create_custom_gpu_config(**kwargs) -> OptimalGPUConfigBase:
    """커스텀 GPU 설정 생성"""
    return PyTorchGPUConfig(**kwargs)

# M3 Max 전용 설정 생성 함수
def create_m3_max_config(**kwargs) -> OptimalGPUConfigBase:
    """M3 Max 전용 최적화 설정"""
    return PyTorchGPUConfig(
        preferred_device='mps',
        memory_fraction=0.85,
        enable_optimization=True,
        mixed_precision=True,
        **kwargs
    )

# 개발용 설정 생성 함수
def create_development_config() -> OptimalGPUConfigBase:
    """개발용 GPU 설정"""
    return PyTorchGPUConfig(
        memory_fraction=0.6,
        enable_optimization=False,
        enable_profiling=True
    )

# 프로덕션용 설정 생성 함수
def create_production_config() -> OptimalGPUConfigBase:
    """프로덕션용 GPU 설정"""
    return PyTorchGPUConfig(
        memory_fraction=0.9,
        enable_optimization=True,
        mixed_precision=True,
        memory_growth=True
    )

# 초기화 로깅
logger.info(f"🎯 GPU 설정 시스템 초기화 완료")
logger.info(f"💻 디바이스: {DEVICE} ({DEVICE_INFO.name})")
logger.info(f"💾 메모리: {MEMORY_GB:.1f}GB")

if IS_M3_MAX:
    logger.info("🍎 M3 Max 최적화 활성화")
if USE_GPU:
    logger.info(f"🎮 GPU 가속 활성화")

# 메모리 체크
if not check_memory_available(4.0):
    logger.warning("⚠️ GPU 메모리 부족 - 성능 저하 가능")

__all__ = [
    'OptimalGPUConfigBase',
    'PyTorchGPUConfig', 
    'GPUConfigManager',
    'DeviceInfo',
    'get_gpu_config',
    'gpu_config',
    'gpu_config_manager',
    # 편의 상수들
    'DEVICE',
    'DEVICE_INFO', 
    'DEVICE_TYPE',
    'USE_GPU',
    'IS_M3_MAX',
    'MEMORY_GB',
    'MODEL_CONFIG',
    # 편의 함수들
    'get_device',
    'get_device_info',
    'get_optimal_settings',
    'optimize_memory',
    'check_memory_available',
    'get_memory_info',
    'create_custom_gpu_config',
    'create_m3_max_config',
    'create_development_config',
    'create_production_config'
]