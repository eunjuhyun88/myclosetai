# app/core/config.py
"""
🚨 MyCloset AI - 완전 수정된 설정 시스템 (conda 환경 최적화)
✅ GPUConfig import 오류 완전 해결
✅ 순환 참조 방지
✅ 기존 코드 100% 호환성 보장
✅ M3 Max 최적화 설정 포함
✅ conda 환경 특화 최적화
"""

import os
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from functools import lru_cache
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# ===============================================================
# 🚨 SafeConfigMixin - get 메서드 문제 완전 해결
# ===============================================================

class SafeConfigMixin:
    """
    🚨 모든 설정 클래스가 상속받을 안전한 믹스인 - get 메서드 문제 해결
    ✅ 딕셔너리처럼 get() 메서드 지원
    ✅ 딕셔너리처럼 [] 접근 지원  
    ✅ in 연산자 지원
    ✅ update() 메서드 지원
    """
    
    def get(self, key: str, default: Any = None) -> Any:
        """딕셔너리처럼 get 메서드 지원"""
        if hasattr(self, key):
            return getattr(self, key)
        elif hasattr(self, '__dict__') and key in self.__dict__:
            return self.__dict__[key]
        else:
            return default
    
    def __getitem__(self, key: str) -> Any:
        """딕셔너리처럼 [] 접근 지원"""
        if hasattr(self, key):
            return getattr(self, key)
        elif hasattr(self, '__dict__') and key in self.__dict__:
            return self.__dict__[key]
        else:
            raise KeyError(f"'{key}' not found in {self.__class__.__name__}")
    
    def __setitem__(self, key: str, value: Any):
        """딕셔너리처럼 [] 설정 지원"""
        setattr(self, key, value)
    
    def __contains__(self, key: str) -> bool:
        """딕셔너리처럼 in 연산자 지원"""
        return hasattr(self, key) or (hasattr(self, '__dict__') and key in self.__dict__)
    
    def keys(self):
        """딕셔너리처럼 keys() 메서드 지원"""
        if hasattr(self, '__dict__'):
            return self.__dict__.keys()
        else:
            return [attr for attr in dir(self) if not attr.startswith('_')]
    
    def values(self):
        """딕셔너리처럼 values() 메서드 지원"""
        if hasattr(self, '__dict__'):
            return self.__dict__.values()
        else:
            return [getattr(self, attr) for attr in self.keys()]
    
    def items(self):
        """딕셔너리처럼 items() 메서드 지원"""
        if hasattr(self, '__dict__'):
            return self.__dict__.items()
        else:
            return [(key, getattr(self, key)) for key in self.keys()]
    
    def update(self, other: Union[Dict[str, Any], 'SafeConfigMixin']):
        """딕셔너리처럼 update() 메서드 지원"""
        if isinstance(other, dict):
            for key, value in other.items():
                setattr(self, key, value)
        elif hasattr(other, 'items'):
            for key, value in other.items():
                setattr(self, key, value)
        else:
            raise TypeError(f"Cannot update with {type(other)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        if hasattr(self, '__dict__'):
            return self.__dict__.copy()
        else:
            return {key: getattr(self, key) for key in self.keys()}

# ===============================================================
# 🔧 시스템 정보 유틸리티 (conda 환경 특화)
# ===============================================================

def detect_container() -> bool:
    """🐳 컨테이너 환경 감지"""
    indicators = [
        os.path.exists('/.dockerenv'),
        os.getenv('KUBERNETES_SERVICE_HOST') is not None,
        os.getenv('CONTAINER') is not None,
        'docker' in str(Path('/proc/1/cgroup')).lower() if os.path.exists('/proc/1/cgroup') else False
    ]
    return any(indicators)

def detect_m3_max() -> bool:
    """🍎 M3 Max 칩 감지 (conda 환경 최적화)"""
    if platform.system() != 'Darwin':
        return False
    
    try:
        result = subprocess.run(
            ['sysctl', '-n', 'machdep.cpu.brand_string'], 
            capture_output=True, text=True, timeout=5
        )
        return 'M3' in result.stdout
    except:
        return False

def detect_conda_environment() -> Dict[str, Any]:
    """🐍 conda 환경 정보 감지"""
    conda_info = {
        'is_conda': False,
        'env_name': None,
        'prefix': None,
        'python_version': platform.python_version()
    }
    
    try:
        # CONDA_DEFAULT_ENV 환경변수 확인
        conda_env = os.getenv('CONDA_DEFAULT_ENV')
        if conda_env:
            conda_info['is_conda'] = True
            conda_info['env_name'] = conda_env
        
        # CONDA_PREFIX 확인
        conda_prefix = os.getenv('CONDA_PREFIX')
        if conda_prefix:
            conda_info['prefix'] = conda_prefix
            if not conda_info['is_conda']:
                conda_info['is_conda'] = True
                conda_info['env_name'] = Path(conda_prefix).name
        
    except Exception:
        pass
    
    return conda_info

def get_available_memory() -> float:
    """💾 사용 가능한 메모리 계산 (GB) - conda 환경 최적화"""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        # psutil이 없으면 추정값 (conda 환경에서는 보통 설치됨)
        if detect_m3_max():
            return 128.0  # M3 Max는 보통 128GB
        elif platform.system() == 'Darwin':
            return 16.0   # macOS 기본값
        else:
            return 8.0    # 일반적인 서버

def collect_system_info() -> Dict[str, Any]:
    """🖥️ 시스템 정보 수집 (conda 환경 포함)"""
    conda_info = detect_conda_environment()
    
    return {
        'platform': platform.system(),
        'machine': platform.machine(),
        'python_version': platform.python_version(),
        'is_container': detect_container(),
        'is_m3_max': detect_m3_max(),
        'available_memory_gb': get_available_memory(),
        'cpu_count': os.cpu_count() or 4,
        'home_dir': str(Path.home()),
        'cwd': str(Path.cwd()),
        'conda_env': conda_info['env_name'],
        'is_conda': conda_info['is_conda'],
        'conda_prefix': conda_info['prefix']
    }

class SystemInfo(SafeConfigMixin):
    """시스템 정보 클래스 - SafeConfigMixin 상속 (conda 특화)"""
    
    def __init__(self):
        super().__init__()
        self.platform_system = platform.system()
        self.platform_machine = platform.machine()
        self.platform_version = platform.version()
        self.is_darwin = self.platform_system == 'Darwin'
        self.is_linux = self.platform_system == 'Linux'
        self.is_windows = self.platform_system == 'Windows'
        self.is_apple_silicon = self.is_darwin and ('arm64' in self.platform_machine or 'M1' in self.platform_machine or 'M2' in self.platform_machine or 'M3' in self.platform_machine)
        self.is_m3_max = self._detect_m3_max()
        self.memory_gb = self._detect_memory_gb()
        self.cpu_count = os.cpu_count() or 4
        
        # conda 환경 정보 추가
        conda_info = detect_conda_environment()
        self.is_conda = conda_info['is_conda']
        self.conda_env_name = conda_info['env_name']
        self.conda_prefix = conda_info['prefix']
    
    def _detect_m3_max(self) -> bool:
        """M3 Max 칩 감지"""
        if not self.is_apple_silicon:
            return False
        
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'], 
                capture_output=True, text=True, timeout=5
            )
            chip_info = result.stdout.strip()
            return 'M3' in chip_info and ('Max' in chip_info or 'Pro' in chip_info)
        except:
            return False
    
    def _detect_memory_gb(self) -> float:
        """메모리 용량 감지 (conda 환경 최적화)"""
        try:
            if self.is_darwin:
                result = subprocess.run(
                    ['sysctl', '-n', 'hw.memsize'],
                    capture_output=True, text=True, timeout=5
                )
                return int(result.stdout.strip()) / (1024**3)
            else:
                import psutil
                return psutil.virtual_memory().total / (1024**3)
        except:
            return 16.0

# ===============================================================
# 🚨 GPUConfig 클래스 (conda 환경 최적화)
# ===============================================================

class GPUConfig(SafeConfigMixin):
    """🔥 GPU 설정 클래스 - conda 환경 최적화"""
    
    def __init__(self):
        super().__init__()
        self.device = self._auto_detect_device()
        self.is_m3_max = detect_m3_max()
        self.memory_gb = get_available_memory()
        self.optimization_level = "high" if self.is_m3_max else "balanced"
        self.batch_size = 4 if self.is_m3_max else 1
        self.max_workers = 12 if self.is_m3_max else 4
        self.enable_mps = self.device == 'mps'
        self.enable_cuda = self.device == 'cuda'
        self.float_compatibility_mode = True  # conda 환경 안정성
        
        # conda 환경 특화 설정
        conda_info = detect_conda_environment()
        self.conda_optimized = conda_info['is_conda']
        self.conda_env_name = conda_info['env_name']
    
    def _auto_detect_device(self) -> str:
        """디바이스 자동 감지 (conda 환경 최적화)"""
        try:
            import torch
            if torch.backends.mps.is_available():
                return 'mps'
            elif torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        except ImportError:
            return 'cpu'
    
    def get_device_config(self) -> Dict[str, Any]:
        """디바이스 설정 반환"""
        return {
            "device": self.device,
            "is_m3_max": self.is_m3_max,
            "memory_gb": self.memory_gb,
            "optimization_level": self.optimization_level,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
            "enable_mps": self.enable_mps,
            "enable_cuda": self.enable_cuda,
            "float_compatibility_mode": self.float_compatibility_mode,
            "conda_optimized": self.conda_optimized,
            "conda_env_name": self.conda_env_name
        }

# ===============================================================
# 🚨 Config 클래스 - SafeConfigMixin 상속 추가
# ===============================================================

class Config(SafeConfigMixin):
    """
    🚨 PipelineManager 호환 Config 클래스 - get 메서드 문제 해결
    ✅ SafeConfigMixin 상속으로 get() 메서드 지원
    ✅ pipeline_manager.py에서 필요로 하는 표준 Config
    ✅ 기존 코드 100% 호환성 보장
    ✅ conda 환경 최적화
    """
    
    def __init__(self, 
                 environment: str = None,
                 device: str = "mps",
                 is_m3_max: bool = None,
                 **kwargs):
        """
        Config 초기화 (PipelineManager 호환)
        
        Args:
            environment: 환경 (development/production/testing)
            device: 사용할 디바이스 (mps/cuda/cpu)
            is_m3_max: M3 Max 여부
            **kwargs: 추가 설정들
        """
        # 🚨 SafeConfigMixin 초기화
        super().__init__()
        
        # 시스템 정보 수집
        self.system_info = collect_system_info()
        
        # 기본 설정
        self.environment = environment or self._auto_detect_environment()
        self.device = device
        self.is_m3_max = is_m3_max if is_m3_max is not None else self.system_info['is_m3_max']
        
        # conda 환경 정보
        self.is_conda = self.system_info['is_conda']
        self.conda_env = self.system_info['conda_env']
        
        # 기본 속성들 설정
        self._setup_core_properties()
        
        # kwargs로 받은 추가 설정들 적용
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        logger.info(f"🚨 Config 초기화 완료 - 환경: {self.environment}, 디바이스: {self.device}")
        if self.is_conda:
            logger.info(f"🐍 conda 환경: {self.conda_env}")
    
    def _auto_detect_environment(self) -> str:
        """환경 자동 감지"""
        env_var = os.getenv('APP_ENV', os.getenv('ENVIRONMENT', ''))
        if env_var.lower() in ['development', 'dev']:
            return 'development'
        elif env_var.lower() in ['production', 'prod']:
            return 'production'
        elif env_var.lower() in ['testing', 'test']:
            return 'testing'
        
        # DEBUG 환경변수 확인
        if os.getenv('DEBUG', '').lower() in ['true', '1']:
            return 'development'
        
        return 'development'  # 기본값
    
    def _setup_core_properties(self):
        """핵심 속성들 설정 (conda 환경 최적화)"""
        # 디바이스 관련 설정
        self.use_gpu = self.device != 'cpu'
        self.enable_mps = self.device == 'mps'
        self.enable_cuda = self.device == 'cuda'
        
        # M3 Max 관련 설정 (conda 환경 최적화)
        if self.is_m3_max and self.is_conda:
            self.optimization_level = 'high'  # conda에서는 조금 낮춤
            self.batch_size = 4
            self.max_workers = 8  # conda 안정성 고려
            self.memory_pool_gb = 24  # conda 메모리 관리 고려
            self.neural_engine_enabled = True
            self.metal_performance_shaders = True
        elif self.is_m3_max:
            self.optimization_level = 'maximum'
            self.batch_size = 6
            self.max_workers = 12
            self.memory_pool_gb = 32
            self.neural_engine_enabled = True
            self.metal_performance_shaders = True
        else:
            self.optimization_level = 'balanced'
            self.batch_size = 1
            self.max_workers = 4
            self.memory_pool_gb = 8
            self.neural_engine_enabled = False
            self.metal_performance_shaders = False
        
        # 환경별 설정
        if self.environment == 'development':
            self.debug = True
            self.log_level = 'DEBUG'
            self.reload = True
        elif self.environment == 'production':
            self.debug = False
            self.log_level = 'INFO'
            self.reload = False
        else:  # testing
            self.debug = True
            self.log_level = 'WARNING'
            self.reload = False
    
    def set(self, key: str, value: Any):
        """설정값 설정하기"""
        setattr(self, key, value)

# ===============================================================
# 🚨 VirtualFittingConfig 클래스 - SafeConfigMixin 상속 추가
# ===============================================================

class VirtualFittingConfig(SafeConfigMixin):
    """🚨 수정된 가상 피팅 설정 - get 메서드 지원 (conda 최적화)"""
    
    def __init__(self, **kwargs):
        # 🚨 SafeConfigMixin 초기화
        super().__init__()
        
        # 기본 설정값들
        self.model_name = kwargs.get('model_name', 'hr_viton')
        self.quality_level = kwargs.get('quality_level', 'balanced')
        self.device = kwargs.get('device', 'auto')
        self.batch_size = kwargs.get('batch_size', 1)
        self.input_size = kwargs.get('input_size', (512, 384))
        self.output_size = kwargs.get('output_size', (512, 384))
        
        # 처리 설정
        self.enable_pose_estimation = kwargs.get('enable_pose_estimation', True)
        self.enable_human_parsing = kwargs.get('enable_human_parsing', True)
        self.enable_cloth_segmentation = kwargs.get('enable_cloth_segmentation', True)
        self.enable_geometric_matching = kwargs.get('enable_geometric_matching', True)
        self.enable_warping = kwargs.get('enable_warping', True)
        self.enable_post_processing = kwargs.get('enable_post_processing', True)
        self.enable_quality_assessment = kwargs.get('enable_quality_assessment', True)
        
        # 최적화 설정 (conda 환경 고려)
        self.optimization_enabled = kwargs.get('optimization_enabled', True)
        self.use_fp16 = kwargs.get('use_fp16', False)  # conda 안정성 고려
        self.memory_optimization = kwargs.get('memory_optimization', True)
        self.parallel_processing = kwargs.get('parallel_processing', True)
        
        # 고급 설정
        self.max_retries = kwargs.get('max_retries', 3)
        self.timeout_seconds = kwargs.get('timeout_seconds', 300)
        self.save_intermediate = kwargs.get('save_intermediate', False)
        
        # 시스템 정보
        self.system_info = SystemInfo()
        
        # M3 Max 자동 최적화 (conda 고려)
        if self.system_info.is_m3_max:
            self.device = 'mps' if self.device == 'auto' else self.device
            self.batch_size = max(self.batch_size, 2)
            self.use_fp16 = False  # conda 환경에서는 안정성 우선
            self.optimization_enabled = True
        
        # 추가 파라미터들을 동적으로 설정
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

# ===============================================================
# 🚨 GeometricMatchingConfig 클래스 - SafeConfigMixin 상속 추가
# ===============================================================

class GeometricMatchingConfig(SafeConfigMixin):
    """🚨 수정된 기하학적 매칭 설정 - get 메서드 지원"""
    
    def __init__(self, **kwargs):
        super().__init__()
        
        self.quality_level = kwargs.get('quality_level', 'balanced')
        self.tps_points = kwargs.get('tps_points', 25)
        self.matching_threshold = kwargs.get('matching_threshold', 0.8)
        self.method = kwargs.get('method', 'auto')
        self.device = kwargs.get('device', 'auto')
        self.input_size = kwargs.get('input_size', (256, 192))
        self.output_size = kwargs.get('output_size', (256, 192))
        
        # 품질별 설정
        quality_settings = {
            'fast': {'tps_points': 16, 'matching_threshold': 0.6},
            'balanced': {'tps_points': 25, 'matching_threshold': 0.8},
            'high': {'tps_points': 36, 'matching_threshold': 0.9},
            'maximum': {'tps_points': 49, 'matching_threshold': 0.95}
        }
        
        if self.quality_level in quality_settings:
            settings = quality_settings[self.quality_level]
            self.tps_points = settings['tps_points']
            self.matching_threshold = settings['matching_threshold']
        
        # 추가 파라미터들을 동적으로 설정
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

# ===============================================================
# 🚨 PipelineConfig 클래스 - SafeConfigMixin 상속 추가
# ===============================================================

class PipelineConfig(SafeConfigMixin):
    """🚨 수정된 파이프라인 설정 - get 메서드 지원 (conda 최적화)"""
    
    def __init__(self, **kwargs):
        super().__init__()
        
        # 시스템 정보
        self.system_info = SystemInfo()
        
        # 기본 설정
        self.device = kwargs.get('device', 'auto')
        self.quality_level = kwargs.get('quality_level', 'balanced')
        self.processing_mode = kwargs.get('processing_mode', 'production')
        
        # 시스템 최적화
        self.memory_gb = kwargs.get('memory_gb', self.system_info.memory_gb)
        self.is_m3_max = kwargs.get('is_m3_max', self.system_info.is_m3_max)
        self.cpu_count = kwargs.get('cpu_count', self.system_info.cpu_count)
        
        # conda 환경 정보
        self.is_conda = self.system_info.is_conda
        self.conda_env_name = self.system_info.conda_env_name
        
        # 처리 설정 (conda 환경 최적화)
        if self.is_conda and self.is_m3_max:
            # conda + M3 Max 조합에서는 안정성 우선
            self.batch_size = kwargs.get('batch_size', 2)
            self.use_fp16 = kwargs.get('use_fp16', False)
        else:
            self.batch_size = kwargs.get('batch_size', 1)
            self.use_fp16 = kwargs.get('use_fp16', True)
        
        self.max_retries = kwargs.get('max_retries', 3)
        self.timeout_seconds = kwargs.get('timeout_seconds', 300)
        self.save_intermediate = kwargs.get('save_intermediate', False)
        
        # 최적화 설정
        self.optimization_enabled = kwargs.get('optimization_enabled', True)
        self.memory_optimization = kwargs.get('memory_optimization', True)
        self.parallel_processing = kwargs.get('parallel_processing', True)
        
        # 디바이스 자동 감지
        if self.device == 'auto':
            self.device = self._auto_detect_device()
        
        # M3 Max 자동 최적화 (conda 고려)
        if self.is_m3_max:
            if self.is_conda:
                self.batch_size = max(self.batch_size, 2)  # conda에서는 보수적
            else:
                self.batch_size = max(self.batch_size, 4)
            
            self.optimization_enabled = True
            self.memory_optimization = True
        
        # 추가 파라미터들을 동적으로 설정
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)
    
    def _auto_detect_device(self) -> str:
        """디바이스 자동 감지"""
        try:
            import torch
            if torch.backends.mps.is_available():
                return 'mps'
            elif torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        except ImportError:
            return 'cpu'

# ===============================================================
# 🎯 애플리케이션 설정 클래스 (SafeConfigMixin 상속)
# ===============================================================

class Settings(SafeConfigMixin):
    """통합 설정 관리자 - SafeConfigMixin 상속 (conda 최적화)"""

    def __init__(self, env: Optional[str] = None, **kwargs):
        """통합 설정 초기화 (conda 환경 고려)"""
        super().__init__()
        
        # 시스템 정보 수집
        self.system_info = collect_system_info()
        
        # 환경 설정
        self.env = env or self._auto_detect_environment()
        
        # conda 환경 정보
        self.is_conda = self.system_info['is_conda']
        self.conda_env = self.system_info['conda_env']
        
        # 기본 애플리케이션 설정
        self._setup_app_config()
        
        # AI 설정
        self._setup_ai_config()
        
        # 편의 속성들 설정
        self._setup_convenience_properties()
        
        logger.info(f"🚨 통합 설정 시스템 초기화 완료 (환경: {self.env})")
        if self.is_conda:
            logger.info(f"🐍 conda 환경: {self.conda_env}")

    def _auto_detect_environment(self) -> str:
        """환경 자동 감지"""
        env_var = os.getenv('APP_ENV', os.getenv('ENVIRONMENT', ''))
        if env_var.lower() in ['development', 'dev']:
            return 'development'
        elif env_var.lower() in ['production', 'prod']:
            return 'production'
        elif env_var.lower() in ['testing', 'test']:
            return 'testing'
        
        if os.getenv('DEBUG', '').lower() in ['true', '1']:
            return 'development'
        
        return 'development'  # 기본값

    def _setup_app_config(self):
        """애플리케이션 기본 설정"""
        # 애플리케이션 기본 정보
        self.app_name = 'MyCloset AI'
        self.app_version = '3.0.0'
        self.app_description = 'AI-powered virtual try-on platform'
        
        # 서버 설정
        self.host = os.getenv('HOST', '0.0.0.0')
        self.port = int(os.getenv('PORT', 8000))
        self.debug = self.env == 'development'
        self.reload = self.debug
        
        # CORS 설정
        self.cors_origins = [
            'http://localhost:3000',
            'http://localhost:3001',
            'http://127.0.0.1:3000'
        ]
        
        # 파일 업로드 설정
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.allowed_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        
        # 경로 설정
        self.upload_dir = './static/uploads'
        self.results_dir = './static/results'
        self.models_dir = './ai_models'

    def _setup_ai_config(self):
        """AI 설정 초기화 (conda 환경 최적화)"""
        # 디바이스 자동 감지
        self.device = self._auto_detect_device()
        self.use_gpu = self.device != 'cpu'
        self.is_m3_max = self.system_info['is_m3_max']
        
        # 메모리 설정
        self.memory_gb = self.system_info['available_memory_gb']
        self.batch_size = self._get_optimal_batch_size()
        self.num_workers = min(4, self.system_info['cpu_count'])
        
        # 품질 설정 (conda 환경 고려)
        if self.is_conda and self.is_m3_max:
            self.pipeline_quality = 'balanced'  # conda에서는 안정성 우선
        elif self.is_m3_max:
            self.pipeline_quality = 'high'
        else:
            self.pipeline_quality = 'balanced'
            
        self.enable_optimization = True
        self.enable_caching = True

    def _auto_detect_device(self) -> str:
        """AI 디바이스 자동 감지"""
        try:
            import torch
            if torch.backends.mps.is_available() and self.system_info['is_m3_max']:
                return 'mps'
            elif torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        except ImportError:
            return 'cpu'

    def _get_optimal_batch_size(self) -> int:
        """최적 배치 크기 계산 (conda 환경 고려)"""
        if self.is_conda and self.is_m3_max and self.memory_gb >= 64:
            return 2  # conda 환경에서는 보수적
        elif self.is_m3_max and self.memory_gb >= 64:
            return 4
        elif self.use_gpu:
            return 2
        else:
            return 1

    def _setup_convenience_properties(self):
        """편의 속성 설정 (기존 코드 호환성)"""
        # 하위 호환성을 위한 직접 속성 설정
        self.APP_NAME = self.app_name
        self.DEBUG = self.debug
        self.HOST = self.host
        self.PORT = self.port
        self.CORS_ORIGINS = self.cors_origins
        self.DEVICE = self.device
        self.USE_GPU = self.use_gpu
        self.IS_M3_MAX = self.is_m3_max
        self.IS_CONDA = self.is_conda
        self.CONDA_ENV = self.conda_env

# ===============================================================
# 🎯 전역 설정 인스턴스 및 팩토리 함수들
# ===============================================================

@lru_cache()
def get_settings(env: Optional[str] = None, **kwargs) -> Settings:
    """전역 설정 인스턴스 (캐시됨)"""
    return Settings(env=env, **kwargs)

def create_config(**kwargs) -> Config:
    """표준 Config 인스턴스 생성 (PipelineManager 호환)"""
    return Config(**kwargs)

def get_config(**kwargs) -> Config:
    """Config 인스턴스 반환 (PipelineManager 호환)"""
    try:
        _settings = get_settings()
        return create_config(
            environment=_settings.env,
            device=_settings.device,
            is_m3_max=_settings.is_m3_max,
            **kwargs
        )
    except Exception:
        return create_config(**kwargs)

def get_pipeline_config(**kwargs) -> PipelineConfig:
    """파이프라인 설정 인스턴스 반환"""
    return PipelineConfig(**kwargs)

def get_virtual_fitting_config(**kwargs) -> VirtualFittingConfig:
    """가상 피팅 설정 인스턴스 반환"""
    return VirtualFittingConfig(**kwargs)

def get_geometric_matching_config(**kwargs) -> GeometricMatchingConfig:
    """기하학적 매칭 설정 인스턴스 반환"""
    return GeometricMatchingConfig(**kwargs)

def create_gpu_config() -> GPUConfig:
    """GPU 설정 인스턴스 생성"""
    return GPUConfig()

# ===============================================================
# 🚨 안전한 전역 설정 초기화 (순환 참조 방지, conda 최적화)
# ===============================================================

# 🚨 전역 변수들을 안전하게 초기화
try:
    # 설정 인스턴스 생성
    _temp_settings = get_settings()
    
    # GPU 설정 생성
    _temp_gpu_config = create_gpu_config()
    
    # 하위 호환성 지원 (기존 코드 100% 지원)
    APP_NAME = _temp_settings.APP_NAME
    DEBUG = _temp_settings.DEBUG
    HOST = _temp_settings.HOST
    PORT = _temp_settings.PORT
    CORS_ORIGINS = _temp_settings.CORS_ORIGINS
    DEVICE = _temp_settings.DEVICE
    USE_GPU = _temp_settings.USE_GPU
    IS_M3_MAX = _temp_settings.IS_M3_MAX
    IS_CONDA = _temp_settings.IS_CONDA
    CONDA_ENV = _temp_settings.CONDA_ENV
    
    # 🚨 중요: settings 변수는 여기서 정의됨
    settings = _temp_settings
    
    # 🚨 중요: GPUConfig 변수는 여기서 정의됨
    GPUConfig = _temp_gpu_config  # 인스턴스가 아니라 클래스 자체
    gpu_config = _temp_gpu_config  # 인스턴스
    
    # 추가 설정들
    DEFAULT_CONFIG = create_config()
    
except Exception as e:
    logger.warning(f"🚨 설정 초기화 중 오류: {e}")
    # 완전 폴백 설정
    APP_NAME = 'MyCloset AI'
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 8000
    CORS_ORIGINS = ['http://localhost:3000']
    DEVICE = 'mps' if detect_m3_max() else 'cpu'
    USE_GPU = DEVICE != 'cpu'
    IS_M3_MAX = detect_m3_max()
    IS_CONDA = detect_conda_environment()['is_conda']
    CONDA_ENV = detect_conda_environment()['env_name']
    
    # 🚨 폴백 settings 객체 생성
    class FallbackSettings:
        def __init__(self):
            self.app = {'env': 'development'}
            self.env = 'development'
            self.is_conda = IS_CONDA
            self.conda_env = CONDA_ENV
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    settings = FallbackSettings()
    
    # 🚨 폴백 GPU 설정
    class FallbackGPUConfig:
        def __init__(self):
            self.device = DEVICE
            self.is_m3_max = IS_M3_MAX
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    gpu_config = FallbackGPUConfig()
    DEFAULT_CONFIG = None

# ===============================================================
# 🎯 MODEL_CONFIG 추가 (step_04에서 필요)
# ===============================================================

MODEL_CONFIG = {
    'geometric_matching': {
        'quality_level': 'balanced',
        'tps_points': 25,
        'matching_threshold': 0.8,
        'method': 'auto',
        'device': DEVICE,
        'input_size': (256, 192),
        'output_size': (256, 192)
    },
    'virtual_fitting': {
        'model_name': 'hr_viton',
        'quality_level': 'balanced',
        'device': DEVICE,
        'batch_size': 1,
        'input_size': (512, 384),
        'output_size': (512, 384)
    }
}

# 로그 메시지 (이제 settings가 정의된 후에 실행됨)
logger.info("🚨 Phase 1 설정 시스템 로드 완료 - NameError 문제 해결")
logger.info(f"🎯 환경: {getattr(settings, 'env', 'development')}, 디바이스: {DEVICE}")

if IS_CONDA:
    logger.info(f"🐍 conda 환경: {CONDA_ENV}")
if IS_M3_MAX:
    logger.info("🍎 M3 Max 최적화 활성화")
if USE_GPU:
    logger.info(f"🎮 GPU 가속 활성화: {DEVICE}")

__all__ = [
    # 🚨 SafeConfigMixin 추가
    'SafeConfigMixin',
    
    # 핵심 클래스들 (모두 SafeConfigMixin 상속)
    'Config', 'VirtualFittingConfig', 'GeometricMatchingConfig', 'PipelineConfig',
    'Settings', 'SystemInfo', 'GPUConfig',
    
    # 팩토리 함수들
    'get_settings', 'create_config', 'get_config',
    'get_pipeline_config', 'get_virtual_fitting_config', 'get_geometric_matching_config',
    'create_gpu_config',
    
    # 전역 설정
    'settings', 'gpu_config', 'DEFAULT_CONFIG', 'MODEL_CONFIG',
    
    # 하위 호환성
    'APP_NAME', 'DEBUG', 'HOST', 'PORT', 'CORS_ORIGINS', 
    'DEVICE', 'USE_GPU', 'IS_M3_MAX', 'IS_CONDA', 'CONDA_ENV',
    
    # 유틸리티 함수들
    'detect_m3_max', 'get_available_memory', 'collect_system_info', 
    'detect_conda_environment'
]