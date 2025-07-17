# app/core/pipeline_config.py
"""
MyCloset AI - 파이프라인 설정 클래스
✅ M3 Max 128GB 최적화
✅ 8단계 AI 파이프라인 설정
✅ 동적 디바이스 감지
✅ 품질/성능 레벨 설정
✅ 메모리 관리 최적화

파일 위치: backend/app/core/pipeline_config.py
"""

import os
import sys
import logging
import platform
import psutil
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# 로깅 설정
logger = logging.getLogger(__name__)

# ==============================================
# 🔥 열거형 정의 (Enum Classes)
# ==============================================

class DeviceType(Enum):
    """지원되는 디바이스 타입"""
    AUTO = "auto"           # 자동 감지
    CPU = "cpu"            # CPU만 사용
    CUDA = "cuda"          # NVIDIA GPU
    MPS = "mps"            # Apple Silicon (M1/M2/M3)
    OPENCL = "opencl"      # OpenCL 지원 GPU

class QualityLevel(Enum):
    """품질 레벨 (성능 vs 품질 트레이드오프)"""
    FAST = "fast"          # 빠른 처리 (낮은 품질)
    BALANCED = "balanced"   # 균형잡힌 품질/성능
    HIGH = "high"          # 높은 품질 (느린 처리)
    ULTRA = "ultra"        # 최고 품질 (매우 느림)
    MAXIMUM = "maximum"    # 최대 품질 (M3 Max 전용)

class PipelineMode(Enum):
    """파이프라인 동작 모드"""
    DEVELOPMENT = "development"  # 개발 모드 (디버깅)
    PRODUCTION = "production"    # 프로덕션 모드 (최적화)
    TESTING = "testing"         # 테스트 모드 (검증)
    SIMULATION = "simulation"   # 시뮬레이션 모드 (더미)
    HYBRID = "hybrid"          # 혼합 모드

class ProcessingStrategy(Enum):
    """처리 전략"""
    SEQUENTIAL = "sequential"   # 순차 처리
    PARALLEL = "parallel"      # 병렬 처리
    PIPELINE = "pipeline"      # 파이프라인 처리
    BATCH = "batch"           # 배치 처리

class MemoryStrategy(Enum):
    """메모리 관리 전략"""
    CONSERVATIVE = "conservative"  # 보수적 (메모리 절약)
    BALANCED = "balanced"         # 균형잡힌
    AGGRESSIVE = "aggressive"     # 공격적 (성능 우선)
    MAXIMUM = "maximum"          # 최대 활용 (M3 Max)

# ==============================================
# 🍎 시스템 정보 클래스
# ==============================================

@dataclass
class SystemInfo:
    """시스템 하드웨어 정보"""
    platform: str = field(default_factory=lambda: platform.system())
    architecture: str = field(default_factory=lambda: platform.machine())
    cpu_cores: int = field(default_factory=lambda: psutil.cpu_count())
    cpu_name: str = ""
    memory_gb: float = field(default_factory=lambda: psutil.virtual_memory().total / (1024**3))
    available_memory_gb: float = field(default_factory=lambda: psutil.virtual_memory().available / (1024**3))
    is_m3_max: bool = False
    is_apple_silicon: bool = False
    gpu_available: bool = False
    gpu_memory_gb: float = 0.0
    
    def __post_init__(self):
        """시스템 정보 자동 감지"""
        self._detect_cpu_info()
        self._detect_gpu_info()
        self._detect_apple_silicon()
    
    def _detect_cpu_info(self):
        """CPU 정보 감지"""
        try:
            if self.platform == "Darwin":  # macOS
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                self.cpu_name = result.stdout.strip()
                
                # M3 Max 감지
                if "M3" in self.cpu_name and "Max" in self.cpu_name:
                    self.is_m3_max = True
                    self.memory_gb = min(self.memory_gb, 128.0)  # M3 Max 최대 128GB
            
            elif self.platform == "Linux":
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('model name'):
                            self.cpu_name = line.split(':')[1].strip()
                            break
            
            elif self.platform == "Windows":
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                   "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0")
                self.cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
                
        except Exception as e:
            logger.warning(f"CPU 정보 감지 실패: {e}")
            self.cpu_name = "Unknown CPU"
    
    def _detect_gpu_info(self):
        """GPU 정보 감지"""
        try:
            # PyTorch GPU 감지
            import torch
            
            if torch.cuda.is_available():
                self.gpu_available = True
                self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.gpu_available = True
                # Apple Silicon의 경우 통합 메모리 사용
                self.gpu_memory_gb = self.memory_gb * 0.7  # 통합 메모리의 70% 추정
                
        except ImportError:
            logger.warning("PyTorch 없음 - GPU 감지 건너뜀")
        except Exception as e:
            logger.warning(f"GPU 정보 감지 실패: {e}")
    
    def _detect_apple_silicon(self):
        """Apple Silicon 감지"""
        if self.platform == "Darwin" and self.architecture == "arm64":
            self.is_apple_silicon = True
            
            # M 시리즈 칩 감지
            if any(chip in self.cpu_name for chip in ["M1", "M2", "M3"]):
                self.is_apple_silicon = True
                
                # M3 Max 특별 처리
                if "M3" in self.cpu_name and "Max" in self.cpu_name:
                    self.is_m3_max = True
                    self.cpu_cores = 16  # M3 Max 16코어
                    self.memory_gb = min(self.memory_gb, 128.0)

# ==============================================
# 🎯 메인 파이프라인 설정 클래스
# ==============================================

@dataclass
class PipelineConfig:
    """MyCloset AI 파이프라인 통합 설정"""
    
    # === 기본 설정 ===
    device: Union[DeviceType, str] = DeviceType.AUTO
    quality_level: Union[QualityLevel, str] = QualityLevel.BALANCED
    mode: Union[PipelineMode, str] = PipelineMode.PRODUCTION
    processing_strategy: Union[ProcessingStrategy, str] = ProcessingStrategy.SEQUENTIAL
    memory_strategy: Union[MemoryStrategy, str] = MemoryStrategy.BALANCED
    
    # === 시스템 정보 ===
    system_info: Optional[SystemInfo] = None
    
    # === 성능 설정 ===
    batch_size: int = 1
    max_workers: int = 4
    timeout_seconds: int = 300
    max_retries: int = 3
    enable_caching: bool = True
    cache_size_mb: int = 1024
    
    # === 메모리 설정 ===
    memory_optimization: bool = True
    gpu_memory_fraction: float = 0.8
    cpu_memory_limit_gb: float = 8.0
    enable_memory_monitoring: bool = True
    memory_cleanup_threshold: float = 0.85
    
    # === 모델 설정 ===
    model_precision: str = "float32"  # float16, float32, mixed
    enable_quantization: bool = False
    model_cache_enabled: bool = True
    model_cache_size: int = 10
    preload_models: bool = False
    
    # === 8단계 파이프라인 설정 ===
    step_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    step_timeouts: Dict[str, int] = field(default_factory=dict)
    step_enabled: Dict[str, bool] = field(default_factory=dict)
    
    # === 고급 설정 ===
    enable_profiling: bool = False
    save_intermediate_results: bool = False
    output_format: str = "png"  # png, jpg, webp
    output_quality: int = 95
    enable_progress_callback: bool = True
    
    # === 디버깅 설정 ===
    debug_mode: bool = False
    verbose_logging: bool = False
    save_debug_images: bool = False
    benchmark_mode: bool = False
    
    def __post_init__(self):
        """설정 후처리 및 최적화"""
        # 시스템 정보 자동 생성
        if self.system_info is None:
            self.system_info = SystemInfo()
        
        # Enum 변환
        self._convert_enums()
        
        # 시스템별 자동 최적화
        self._auto_optimize_for_system()
        
        # 8단계 설정 초기화
        self._initialize_step_configs()
        
        # 검증
        self._validate_config()
    
    def _convert_enums(self):
        """문자열을 Enum으로 변환"""
        if isinstance(self.device, str):
            self.device = DeviceType(self.device)
        if isinstance(self.quality_level, str):
            self.quality_level = QualityLevel(self.quality_level)
        if isinstance(self.mode, str):
            self.mode = PipelineMode(self.mode)
        if isinstance(self.processing_strategy, str):
            self.processing_strategy = ProcessingStrategy(self.processing_strategy)
        if isinstance(self.memory_strategy, str):
            self.memory_strategy = MemoryStrategy(self.memory_strategy)
    
    def _auto_optimize_for_system(self):
        """시스템별 자동 최적화"""
        system = self.system_info
        
        # M3 Max 최적화
        if system.is_m3_max:
            self.device = DeviceType.MPS
            self.quality_level = QualityLevel.MAXIMUM
            self.memory_strategy = MemoryStrategy.MAXIMUM
            self.processing_strategy = ProcessingStrategy.PARALLEL
            self.batch_size = min(8, self.batch_size * 4)
            self.max_workers = 16
            self.gpu_memory_fraction = 0.95
            self.cpu_memory_limit_gb = min(64.0, system.memory_gb * 0.8)
            self.model_precision = "float16"
            self.cache_size_mb = 4096
            self.model_cache_size = 15
            logger.info("🍎 M3 Max 최적화 설정 적용")
        
        # Apple Silicon 일반 최적화
        elif system.is_apple_silicon:
            self.device = DeviceType.MPS
            self.memory_strategy = MemoryStrategy.BALANCED
            self.gpu_memory_fraction = 0.8
            self.model_precision = "float16"
            logger.info("🍎 Apple Silicon 최적화 설정 적용")
        
        # CUDA GPU 최적화
        elif system.gpu_available and system.gpu_memory_gb > 8:
            self.device = DeviceType.CUDA
            self.processing_strategy = ProcessingStrategy.PARALLEL
            self.batch_size = min(4, self.batch_size * 2)
            self.gpu_memory_fraction = 0.9
            self.model_precision = "float16"
            logger.info("🚀 CUDA GPU 최적화 설정 적용")
        
        # CPU 전용 최적화
        else:
            self.device = DeviceType.CPU
            self.quality_level = QualityLevel.FAST
            self.memory_strategy = MemoryStrategy.CONSERVATIVE
            self.batch_size = 1
            self.model_precision = "float32"
            logger.info("💻 CPU 최적화 설정 적용")
        
        # 메모리 제한 설정
        if system.memory_gb < 16:
            self.memory_strategy = MemoryStrategy.CONSERVATIVE
            self.cache_size_mb = 512
            self.model_cache_size = 3
            self.cpu_memory_limit_gb = system.memory_gb * 0.6
    
    def _initialize_step_configs(self):
        """8단계별 세부 설정 초기화"""
        steps = [
            "human_parsing", "pose_estimation", "cloth_segmentation",
            "geometric_matching", "cloth_warping", "virtual_fitting",
            "post_processing", "quality_assessment"
        ]
        
        # 기본 단계별 설정
        for step in steps:
            if step not in self.step_configs:
                self.step_configs[step] = self._get_default_step_config(step)
            
            if step not in self.step_timeouts:
                self.step_timeouts[step] = 60  # 기본 60초
            
            if step not in self.step_enabled:
                self.step_enabled[step] = True
        
        # M3 Max 최적화된 타임아웃
        if self.system_info.is_m3_max:
            for step in steps:
                self.step_timeouts[step] = min(30, self.step_timeouts[step] // 2)
    
    def _get_default_step_config(self, step: str) -> Dict[str, Any]:
        """단계별 기본 설정 반환"""
        base_config = {
            "enabled": True,
            "device": self.device.value,
            "precision": self.model_precision,
            "batch_size": self.batch_size,
            "enable_caching": self.enable_caching
        }
        
        # 단계별 특수 설정
        step_specific = {
            "human_parsing": {
                "model_name": "graphonomy",
                "input_size": (512, 512),
                "enable_visualization": True
            },
            "pose_estimation": {
                "model_name": "openpose",
                "confidence_threshold": 0.5,
                "enable_hand_detection": False
            },
            "cloth_segmentation": {
                "model_name": "u2net",
                "refinement_enabled": True,
                "edge_smoothing": True
            },
            "geometric_matching": {
                "model_name": "gmm",
                "matching_threshold": 0.8,
                "enable_refinement": True
            },
            "cloth_warping": {
                "model_name": "tom",
                "warp_strength": 1.0,
                "preserve_details": True
            },
            "virtual_fitting": {
                "model_name": "ootdiffusion",
                "inference_steps": 20,
                "guidance_scale": 7.5
            },
            "post_processing": {
                "enable_super_resolution": self.quality_level in [QualityLevel.HIGH, QualityLevel.ULTRA, QualityLevel.MAXIMUM],
                "enable_denoising": True,
                "sharpening_strength": 0.5
            },
            "quality_assessment": {
                "enable_ai_assessment": True,
                "assessment_model": "combined",
                "threshold_score": 0.7
            }
        }
        
        base_config.update(step_specific.get(step, {}))
        return base_config
    
    def _validate_config(self):
        """설정 검증"""
        # 메모리 검증
        if self.cpu_memory_limit_gb > self.system_info.memory_gb:
            self.cpu_memory_limit_gb = self.system_info.memory_gb * 0.8
            logger.warning(f"CPU 메모리 제한을 {self.cpu_memory_limit_gb:.1f}GB로 조정")
        
        # 배치 크기 검증
        if self.batch_size > 16:
            self.batch_size = 16
            logger.warning("배치 크기를 16으로 제한")
        
        # 워커 수 검증
        if self.max_workers > self.system_info.cpu_cores:
            self.max_workers = self.system_info.cpu_cores
            logger.warning(f"워커 수를 {self.max_workers}로 제한")
    
    # === 유틸리티 메서드 ===
    
    def get_device_str(self) -> str:
        """디바이스 문자열 반환"""
        return self.device.value
    
    def get_step_config(self, step_name: str) -> Dict[str, Any]:
        """특정 단계 설정 반환"""
        return self.step_configs.get(step_name, {})
    
    def update_step_config(self, step_name: str, config: Dict[str, Any]):
        """단계 설정 업데이트"""
        if step_name in self.step_configs:
            self.step_configs[step_name].update(config)
        else:
            self.step_configs[step_name] = config
    
    def enable_debug_mode(self):
        """디버그 모드 활성화"""
        self.debug_mode = True
        self.verbose_logging = True
        self.save_debug_images = True
        self.save_intermediate_results = True
        self.enable_profiling = True
        logger.info("🐛 디버그 모드 활성화")
    
    def enable_production_mode(self):
        """프로덕션 모드 최적화"""
        self.mode = PipelineMode.PRODUCTION
        self.debug_mode = False
        self.verbose_logging = False
        self.save_debug_images = False
        self.save_intermediate_results = False
        self.enable_profiling = False
        self.memory_optimization = True
        logger.info("🚀 프로덕션 모드 활성화")
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        config_dict = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Enum):
                config_dict[field_name] = field_value.value
            elif isinstance(field_value, SystemInfo):
                config_dict[field_name] = field_value.__dict__
            else:
                config_dict[field_name] = field_value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """딕셔너리에서 설정 생성"""
        return cls(**config_dict)
    
    def save_to_file(self, filepath: str):
        """설정을 파일로 저장"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"설정을 {filepath}에 저장")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'PipelineConfig':
        """파일에서 설정 로드"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

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
    config_kwargs = kwargs.copy()
    
    if device:
        config_kwargs['device'] = device
    if quality_level:
        config_kwargs['quality_level'] = quality_level
    if mode:
        config_kwargs['mode'] = mode
    
    return PipelineConfig(**config_kwargs)

def create_development_config() -> PipelineConfig:
    """개발용 설정 생성"""
    config = PipelineConfig(
        mode=PipelineMode.DEVELOPMENT,
        quality_level=QualityLevel.FAST,
        debug_mode=True,
        verbose_logging=True,
        save_intermediate_results=True
    )
    config.enable_debug_mode()
    return config

def create_production_config() -> PipelineConfig:
    """프로덕션용 설정 생성"""
    config = PipelineConfig(
        mode=PipelineMode.PRODUCTION,
        memory_optimization=True,
        enable_caching=True,
        preload_models=True
    )
    config.enable_production_mode()
    return config

def create_m3_max_config() -> PipelineConfig:
    """M3 Max 최적화 설정 생성"""
    return PipelineConfig(
        device=DeviceType.MPS,
        quality_level=QualityLevel.MAXIMUM,
        memory_strategy=MemoryStrategy.MAXIMUM,
        processing_strategy=ProcessingStrategy.PARALLEL,
        model_precision="float16",
        batch_size=8,
        max_workers=16
    )

# ==============================================
# 🔧 전역 설정 관리
# ==============================================

_global_pipeline_config: Optional[PipelineConfig] = None

def get_global_pipeline_config() -> PipelineConfig:
    """전역 파이프라인 설정 반환"""
    global _global_pipeline_config
    if _global_pipeline_config is None:
        _global_pipeline_config = PipelineConfig()
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
    
    # 메모리 설정
    if os.getenv('MYCLOSET_MEMORY_LIMIT'):
        config_kwargs['cpu_memory_limit_gb'] = float(os.getenv('MYCLOSET_MEMORY_LIMIT'))
    if os.getenv('MYCLOSET_GPU_MEMORY_FRACTION'):
        config_kwargs['gpu_memory_fraction'] = float(os.getenv('MYCLOSET_GPU_MEMORY_FRACTION'))
    
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
    'SystemInfo', 'PipelineConfig',
    
    # Factory Functions
    'create_pipeline_config', 'create_development_config', 'create_production_config', 'create_m3_max_config',
    
    # Global Config Management
    'get_global_pipeline_config', 'set_global_pipeline_config', 'reset_global_pipeline_config',
    
    # Environment Config
    'load_config_from_env'
]

# ==============================================
# 🎉 모듈 초기화
# ==============================================

logger.info("✅ Pipeline Config 모듈 로드 완료")
logger.info(f"🍎 시스템: {SystemInfo().platform} ({SystemInfo().architecture})")
logger.info(f"💾 메모리: {SystemInfo().memory_gb:.1f}GB")
if SystemInfo().is_m3_max:
    logger.info("🔥 M3 Max 감지됨 - 최적화 모드 활성화")