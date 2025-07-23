# app/core/pipeline_config.py
"""
🔥 MyCloset AI - 완전한 PipelineConfig 클래스 (conda 환경 우선)
================================================================

✅ main.py PipelineManager 초기화 오류 완전 해결
✅ conda 환경 최적화 + M3 Max 128GB 최적화 
✅ 모든 필수 속성 완전 구현
✅ SafeConfigMixin 상속으로 .get() 메서드 지원
✅ 순환참조 완전 방지
✅ 프로덕션 레벨 안정성

파일 위치: backend/app/core/pipeline_config.py
"""

import os
import sys
import logging
import platform
import subprocess
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# 안전한 import
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# 로깅 설정
logger = logging.getLogger(__name__)

# ==============================================
# 🔥 열거형 정의 (Enum Classes)
# ==============================================

class DeviceType(Enum):
    """지원되는 디바이스 타입"""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    OPENCL = "opencl"

class QualityLevel(Enum):
    """품질 레벨"""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"
    ULTRA = "ultra"
    MAXIMUM = "maximum"

class PipelineMode(Enum):
    """파이프라인 동작 모드"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"
    SIMULATION = "simulation"
    HYBRID = "hybrid"

class ProcessingStrategy(Enum):
    """처리 전략"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    BATCH = "batch"

class MemoryStrategy(Enum):
    """메모리 관리 전략"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

# ==============================================
# 🍎 시스템 정보 클래스
# ==============================================

@dataclass
class SystemInfo:
    """시스템 하드웨어 정보"""
    platform: str = field(default_factory=lambda: platform.system())
    architecture: str = field(default_factory=lambda: platform.machine())
    cpu_cores: int = field(default_factory=lambda: psutil.cpu_count() if PSUTIL_AVAILABLE else 4)
    cpu_name: str = ""
    memory_gb: float = field(default_factory=lambda: psutil.virtual_memory().total / (1024**3) if PSUTIL_AVAILABLE else 16.0)
    available_memory_gb: float = field(default_factory=lambda: psutil.virtual_memory().available / (1024**3) if PSUTIL_AVAILABLE else 12.0)
    is_m3_max: bool = False
    is_apple_silicon: bool = False
    gpu_available: bool = False
    gpu_memory_gb: float = 0.0
    is_conda: bool = False
    conda_env_name: str = ""
    
    def __post_init__(self):
        """시스템 정보 자동 감지"""
        # Apple Silicon 감지
        if self.platform == "Darwin" and self.architecture == "arm64":
            self.is_apple_silicon = True
            # M3 Max 감지
            try:
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True, text=True, timeout=5
                )
                if 'M3' in result.stdout:
                    self.is_m3_max = True
                    self.memory_gb = 128.0  # M3 Max 기본 메모리
                    self.gpu_available = True
                    self.gpu_memory_gb = 40.0  # M3 Max GPU 메모리
            except:
                pass
        
        # conda 환경 감지
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if conda_env:
            self.is_conda = True
            self.conda_env_name = conda_env

# ==============================================
# 🔧 SafeConfigMixin - get() 메서드 지원
# ==============================================

class SafeConfigMixin:
    """
    🔧 SafeConfigMixin - dict 스타일 접근 지원
    
    PipelineManager에서 config.get() 호출 시 필요한 mixin
    """
    
    def get(self, key: str, default: Any = None) -> Any:
        """딕셔너리 스타일 접근 지원"""
        return getattr(self, key, default)
    
    def __getitem__(self, key: str) -> Any:
        """딕셔너리 스타일 접근 지원 []"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in config")
    
    def __contains__(self, key: str) -> bool:
        """'in' 연산자 지원"""
        return hasattr(self, key)
    
    def keys(self):
        """딕셔너리 스타일 키 목록 반환"""
        return [attr for attr in dir(self) if not attr.startswith('_') and not callable(getattr(self, attr))]
    
    def items(self):
        """딕셔너리 스타일 아이템 반환"""
        for key in self.keys():
            yield key, getattr(self, key)

# ==============================================
# 🔥 PipelineConfig 클래스 (완전한 구현)
# ==============================================

class PipelineConfig(SafeConfigMixin):
    """
    🔥 완전한 PipelineConfig 클래스 - main.py 오류 해결
    
    ✅ SafeConfigMixin 상속으로 .get() 메서드 지원
    ✅ 모든 필수 속성 완전 구현
    ✅ conda 환경 우선 최적화
    ✅ M3 Max 128GB 메모리 완전 활용
    ✅ 순환참조 완전 방지
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        quality_level: Optional[str] = None,
        mode: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_workers: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        max_retries: Optional[int] = None,
        enable_caching: Optional[bool] = None,
        memory_optimization: Optional[bool] = None,
        **kwargs
    ):
        """PipelineConfig 초기화"""
        
        # SafeConfigMixin 초기화
        super().__init__()
        
        # 시스템 정보 수집
        self.system_info = SystemInfo()
        
        # 🔥 디바이스 설정 (자동 감지)
        if device == "auto" or device is None:
            if self.system_info.is_m3_max:
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # 🔥 품질 레벨 설정
        if isinstance(quality_level, str):
            self.quality_level = quality_level
        elif hasattr(quality_level, 'value'):
            self.quality_level = quality_level.value
        else:
            self.quality_level = "balanced"
        
        # 🔥 모드 설정
        if isinstance(mode, str):
            self.mode = mode
        elif hasattr(mode, 'value'):
            self.mode = mode.value
        else:
            self.mode = "production"
        
        # 🔥 conda 환경 최적화 적용
        if self.system_info.is_conda and self.system_info.is_m3_max:
            # conda + M3 Max: 안정성 우선
            self.batch_size = batch_size or 1
            self.max_workers = max_workers or 2
            self.timeout_seconds = timeout_seconds or 300
            self.memory_optimization = True
            self.use_fp16 = False  # conda에서는 FP16 비활성화
        elif self.system_info.is_m3_max:
            # M3 Max: 성능 우선
            self.batch_size = batch_size or 2
            self.max_workers = max_workers or 4
            self.timeout_seconds = timeout_seconds or 180
            self.memory_optimization = True
            self.use_fp16 = True
        else:
            # 일반 환경
            self.batch_size = batch_size or 1
            self.max_workers = max_workers or 2
            self.timeout_seconds = timeout_seconds or 120
            self.memory_optimization = memory_optimization or True
            self.use_fp16 = False
        
        # 🔥 기본 설정들
        self.max_retries = max_retries or 2
        self.enable_caching = enable_caching if enable_caching is not None else True
        
        # 🔥 시스템 정보 복사
        self.is_m3_max = self.system_info.is_m3_max
        self.is_conda = self.system_info.is_conda
        self.conda_env_name = self.system_info.conda_env_name
        self.memory_gb = self.system_info.memory_gb
        self.cpu_cores = self.system_info.cpu_cores
        
        # 🔥 성능 설정
        self.parallel_processing = True
        self.model_cache_size = 16 if self.is_m3_max else 8
        self.lazy_loading = True
        self.preload_models = False  # conda에서는 비활성화
        
        # 🔥 디버그 설정
        self.debug_mode = self.mode == "development"
        self.verbose_logging = self.debug_mode
        self.save_intermediate_results = self.debug_mode
        
        # 🔥 추가 kwargs 적용
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        logger.info(f"✅ PipelineConfig 초기화 완료")
        logger.info(f"   🔧 디바이스: {self.device}")
        logger.info(f"   📊 품질 레벨: {self.quality_level}")
        logger.info(f"   🏭 모드: {self.mode}")
        logger.info(f"   🍎 M3 Max: {self.is_m3_max}")
        logger.info(f"   🐍 conda: {self.conda_env_name if self.is_conda else '비활성화'}")
        logger.info(f"   💾 메모리: {self.memory_gb:.1f}GB")
        logger.info(f"   👥 워커: {self.max_workers}개")
        logger.info(f"   📦 배치 크기: {self.batch_size}")
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            key: getattr(self, key) 
            for key in self.keys() 
            if not key.startswith('system_info')
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """딕셔너리에서 생성"""
        return cls(**config_dict)
    
    def enable_debug_mode(self):
        """디버그 모드 활성화"""
        self.debug_mode = True
        self.verbose_logging = True
        self.save_intermediate_results = True
        self.mode = "development"
    
    def enable_production_mode(self):
        """프로덕션 모드 활성화"""
        self.debug_mode = False
        self.verbose_logging = False
        self.save_intermediate_results = False
        self.mode = "production"
        self.memory_optimization = True
        self.enable_caching = True

# ==============================================
# 🏭 팩토리 함수들
# ==============================================

def create_pipeline_config(
    device: Optional[str] = None,
    quality_level: Optional[str] = None,
    mode: Optional[str] = None,
    **kwargs
) -> PipelineConfig:
    """파이프라인 설정 생성 팩토리"""
    return PipelineConfig(
        device=device,
        quality_level=quality_level,
        mode=mode,
        **kwargs
    )

def create_development_config() -> PipelineConfig:
    """개발용 설정 생성"""
    config = PipelineConfig(
        mode="development",
        quality_level="fast",
        debug_mode=True,
        verbose_logging=True,
        save_intermediate_results=True
    )
    return config

def create_production_config() -> PipelineConfig:
    """프로덕션용 설정 생성"""
    config = PipelineConfig(
        mode="production",
        quality_level="high",
        memory_optimization=True,
        enable_caching=True,
        preload_models=False  # conda 안정성 우선
    )
    return config

def create_m3_max_config() -> PipelineConfig:
    """M3 Max 최적화 설정 생성"""
    return PipelineConfig(
        device="mps",
        quality_level="maximum",
        mode="production",
        batch_size=2,
        max_workers=4,
        memory_optimization=True,
        use_fp16=True
    )

def create_conda_optimized_config() -> PipelineConfig:
    """conda 환경 최적화 설정 생성"""
    return PipelineConfig(
        device="auto",
        quality_level="balanced",
        mode="production",
        batch_size=1,
        max_workers=2,
        memory_optimization=True,
        use_fp16=False,  # conda 안정성
        preload_models=False
    )

# ==============================================
# 🔧 전역 설정 관리
# ==============================================

_global_pipeline_config: Optional[PipelineConfig] = None

def get_global_pipeline_config() -> PipelineConfig:
    """전역 파이프라인 설정 반환"""
    global _global_pipeline_config
    if _global_pipeline_config is None:
        _global_pipeline_config = create_conda_optimized_config()
    return _global_pipeline_config

def set_global_pipeline_config(config: PipelineConfig):
    """전역 파이프라인 설정 설정"""
    global _global_pipeline_config
    _global_pipeline_config = config

def reset_global_pipeline_config():
    """전역 파이프라인 설정 초기화"""
    global _global_pipeline_config
    _global_pipeline_config = None

# ==============================================
# 🎯 환경변수 기반 설정 로드
# ==============================================

def load_config_from_env() -> PipelineConfig:
    """환경변수에서 설정 로드"""
    config_kwargs = {}
    
    # 기본 설정
    if os.getenv('MYCLOSET_DEVICE'):
        config_kwargs['device'] = os.getenv('MYCLOSET_DEVICE')
    if os.getenv('MYCLOSET_QUALITY'):
        config_kwargs['quality_level'] = os.getenv('MYCLOSET_QUALITY')
    if os.getenv('MYCLOSET_MODE'):
        config_kwargs['mode'] = os.getenv('MYCLOSET_MODE')
    
    # 성능 설정
    if os.getenv('MYCLOSET_BATCH_SIZE'):
        config_kwargs['batch_size'] = int(os.getenv('MYCLOSET_BATCH_SIZE'))
    if os.getenv('MYCLOSET_MAX_WORKERS'):
        config_kwargs['max_workers'] = int(os.getenv('MYCLOSET_MAX_WORKERS'))
    
    # 디버그 설정
    if os.getenv('MYCLOSET_DEBUG', '').lower() in ['true', '1']:
        config_kwargs['debug_mode'] = True
    
    return PipelineConfig(**config_kwargs)

# ==============================================
# 📋 모듈 Export
# ==============================================

__all__ = [
    # Enums
    'DeviceType', 'QualityLevel', 'PipelineMode', 'ProcessingStrategy', 'MemoryStrategy',
    
    # Classes
    'SystemInfo', 'PipelineConfig', 'SafeConfigMixin',
    
    # Factory Functions
    'create_pipeline_config', 'create_development_config', 'create_production_config', 
    'create_m3_max_config', 'create_conda_optimized_config',
    
    # Global Config Management
    'get_global_pipeline_config', 'set_global_pipeline_config', 'reset_global_pipeline_config',
    
    # Environment Config
    'load_config_from_env'
]

# ==============================================
# 🎉 모듈 초기화
# ==============================================

logger.info("✅ Pipeline Config 모듈 로드 완료")
system_info = SystemInfo()
logger.info(f"🍎 시스템: {system_info.platform} ({system_info.architecture})")
logger.info(f"💾 메모리: {system_info.memory_gb:.1f}GB")
if system_info.is_m3_max:
    logger.info("🔥 M3 Max 감지됨 - 최적화 모드 활성화")
if system_info.is_conda:
    logger.info(f"🐍 conda 환경 감지됨: {system_info.conda_env_name}")