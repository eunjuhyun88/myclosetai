# app/core/config.py
"""
최적 설정 시스템 - 순환 참조 수정 및 M3 Max 최적화
"""
import os
import platform
import subprocess
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from functools import lru_cache
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# ===============================================================
# 🔧 시스템 정보 유틸리티 (독립 함수들)
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
    """🍎 M3 Max 칩 감지 (독립 함수)"""
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

def get_available_memory() -> float:
    """💾 사용 가능한 메모리 계산 (GB)"""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        # psutil이 없으면 추정값
        if detect_m3_max():
            return 128.0  # M3 Max는 보통 128GB
        elif platform.system() == 'Darwin':
            return 16.0   # macOS 기본값
        else:
            return 8.0    # 일반적인 서버

def collect_system_info() -> Dict[str, Any]:
    """🖥️ 시스템 정보 수집 (독립 함수)"""
    return {
        'platform': platform.system(),
        'machine': platform.machine(),
        'python_version': platform.python_version(),
        'is_container': detect_container(),
        'is_m3_max': detect_m3_max(),
        'available_memory_gb': get_available_memory(),
        'cpu_count': os.cpu_count() or 4,
        'home_dir': str(Path.home()),
        'cwd': str(Path.cwd())
    }

# ===============================================================
# 🎯 최적 설정 베이스 클래스
# ===============================================================

class OptimalConfigBase(ABC):
    """
    🎯 최적화된 설정 베이스 클래스
    - 자동 환경 감지
    - 지능적 기본값
    - 확장성
    - 일관성
    """

    def __init__(
        self,
        env: Optional[str] = None,  # 환경 (None=자동감지)
        config_path: Optional[str] = None,  # 설정 파일 경로
        **kwargs  # 확장 파라미터
    ):
        """
        ✅ 최적 설정 생성자 - 순환 참조 해결

        Args:
            env: 환경 (None=자동감지, 'development', 'production', 'testing')
            config_path: 외부 설정 파일 경로
            **kwargs: 확장 파라미터들
        """
        # 1. 📋 시스템 정보 수집 (먼저 수집)
        self.system_info = collect_system_info()
        
        # 2. 💡 지능적 환경 자동 감지
        self.env = self._auto_detect_environment(env)
        
        # 3. 🔧 기본 설정 생성
        self._config = self._create_base_config()
        
        # 4. ⚙️ kwargs 파라미터 병합
        self._merge_kwargs_config(kwargs)
        
        # 5. 📁 외부 설정 파일 로드
        if config_path and os.path.exists(config_path):
            self._load_external_config(config_path)
        
        # 6. 🌍 환경변수 오버라이드
        self._apply_environment_overrides()
        
        # 7. ✨ 환경별 최적화 적용
        self._apply_environment_optimizations()
        
        logger.info(f"🎯 {self.__class__.__name__} 초기화 완료 - 환경: {self.env}")

    def _auto_detect_environment(self, preferred_env: Optional[str]) -> str:
        """💡 지능적 환경 자동 감지"""
        if preferred_env:
            return preferred_env

        # 환경변수 확인
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
        
        # 개발 환경 감지 (일반적인 개발 도구들)
        dev_indicators = [
            '.git',
            'requirements-dev.txt',
            'docker-compose.yml',
            'Dockerfile.dev'
        ]
        
        current_dir = Path.cwd()
        for indicator in dev_indicators:
            if (current_dir / indicator).exists():
                return 'development'
        
        # 기본값: production (안전한 선택)
        return 'production'

    @abstractmethod
    def _create_base_config(self) -> Dict[str, Any]:
        """기본 설정 생성 (서브클래스에서 구현)"""
        pass

    def _merge_kwargs_config(self, kwargs: Dict[str, Any]):
        """⚙️ kwargs 파라미터 병합"""
        for key, value in kwargs.items():
            if key in self._config:
                self._config[key] = value
            else:
                # 새로운 설정 추가
                self._config[key] = value

    def _load_external_config(self, config_path: str):
        """📁 외부 설정 파일 로드"""
        try:
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                external_config = json.load(f)
                self._deep_merge(self._config, external_config)
            logger.info(f"📁 외부 설정 로드: {config_path}")
        except Exception as e:
            logger.warning(f"외부 설정 로드 실패 ({config_path}): {e}")

    def _apply_environment_overrides(self):
        """🌍 환경변수 오버라이드"""
        env_mappings = {
            'APP_NAME': 'app_name',
            'HOST': 'host',
            'PORT': 'port',
            'DEBUG': 'debug',
            'LOG_LEVEL': 'log_level',
            'DATABASE_URL': 'database_url',
            'REDIS_URL': 'redis_url',
            'SECRET_KEY': 'secret_key',
            'CORS_ORIGINS': 'cors_origins'
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # 타입 변환
                if config_key == 'port':
                    self._config[config_key] = int(env_value)
                elif config_key == 'debug':
                    self._config[config_key] = env_value.lower() in ['true', '1']
                elif config_key == 'cors_origins':
                    self._config[config_key] = [origin.strip() for origin in env_value.split(',')]
                else:
                    self._config[config_key] = env_value

    def _apply_environment_optimizations(self):
        """✨ 환경별 최적화"""
        if self.env == 'development':
            self._apply_development_optimizations()
        elif self.env == 'production':
            self._apply_production_optimizations()
        elif self.env == 'testing':
            self._apply_testing_optimizations()

    def _apply_development_optimizations(self):
        """🔧 개발 환경 최적화"""
        self._config.update({
            'debug': True,
            'log_level': 'DEBUG',
            'reload': True,
            'workers': 1,
            'enable_profiling': True,
            'enable_hot_reload': True,
            'cors_origins': ['*']  # 개발시 모든 origin 허용
        })

    def _apply_production_optimizations(self):
        """🚀 프로덕션 환경 최적화"""
        optimal_workers = min(self.system_info['cpu_count'], 8)
        
        self._config.update({
            'debug': False,
            'log_level': 'INFO',
            'reload': False,
            'workers': optimal_workers,
            'enable_profiling': False,
            'enable_hot_reload': False,
            'timeout': 300,
            'keepalive': 2
        })
        
        # M3 Max 프로덕션 최적화
        if self.system_info['is_m3_max']:
            self._config.update({
                'workers': min(12, optimal_workers * 2),  # M3 Max는 더 많은 워커
                'max_memory_usage': '64GB',
                'enable_mps_optimization': True
            })

    def _apply_testing_optimizations(self):
        """🧪 테스트 환경 최적화"""
        self._config.update({
            'debug': True,
            'log_level': 'WARNING',  # 테스트시 로그 줄이기
            'workers': 1,
            'timeout': 30,
            'database_url': 'sqlite:///:memory:',  # 인메모리 DB
            'cache_enabled': False
        })

    def _deep_merge(self, base_dict: Dict, merge_dict: Dict):
        """🔗 딕셔너리 깊은 병합"""
        for key, value in merge_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value

    # 설정 접근 메서드들
    def get(self, key: str, default: Any = None) -> Any:
        """설정값 가져오기"""
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        """설정값 설정하기"""
        self._config[key] = value

    def update(self, **kwargs):
        """여러 설정값 업데이트"""
        self._config.update(kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 반환"""
        return self._config.copy()

    @property
    def is_development(self) -> bool:
        return self.env == 'development'

    @property
    def is_production(self) -> bool:
        return self.env == 'production'

    @property
    def is_testing(self) -> bool:
        return self.env == 'testing'

# ===============================================================
# 🎯 애플리케이션 설정 클래스
# ===============================================================

class AppConfig(OptimalConfigBase):
    """
    🎯 애플리케이션 메인 설정
    - FastAPI 설정
    - 서버 설정
    - 보안 설정
    - CORS 설정
    """

    def _create_base_config(self) -> Dict[str, Any]:
        """기본 애플리케이션 설정 생성"""
        return {
            # 애플리케이션 기본 정보
            'app_name': 'MyCloset AI',
            'app_version': '2.0.0',
            'app_description': 'AI-powered virtual try-on platform',
            'api_prefix': '/api',
            
            # 서버 설정
            'host': '0.0.0.0',
            'port': 8000,
            'workers': 4,
            'timeout': 300,
            'keepalive': 2,
            'max_requests': 1000,
            'max_requests_jitter': 100,
            
            # 개발/디버그 설정
            'debug': False,
            'reload': False,
            'log_level': 'INFO',
            'enable_profiling': False,
            'enable_hot_reload': False,
            
            # 보안 설정
            'secret_key': self._generate_secret_key(),
            'algorithm': 'HS256',
            'access_token_expire_minutes': 1440,  # 24시간
            'allowed_hosts': ['*'],
            
            # CORS 설정
            'cors_origins': [
                'http://localhost:3000',
                'http://localhost:3001',
                'http://127.0.0.1:3000'
            ],
            'cors_allow_credentials': True,
            'cors_allow_methods': ['*'],
            'cors_allow_headers': ['*'],
            
            # 데이터베이스 설정
            'database_url': 'sqlite:///./mycloset.db',
            'database_echo': False,
            'database_pool_size': 20,
            'database_max_overflow': 0,
            
            # 캐시 설정
            'redis_url': 'redis://localhost:6379/0',
            'cache_enabled': True,
            'cache_ttl': 3600,  # 1시간
            
            # 파일 업로드 설정
            'upload_dir': './static/uploads',
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'allowed_extensions': ['.jpg', '.jpeg', '.png', '.webp'],
            
            # 시스템 정보
            'system_info': self.system_info,
            'environment': self.env
        }

    def _generate_secret_key(self) -> str:
        """🔐 시크릿 키 생성"""
        import secrets
        return secrets.token_urlsafe(32)

    # 편의 속성들
    @property
    def database_url(self) -> str:
        return self.get('database_url')

    @property
    def redis_url(self) -> str:
        return self.get('redis_url')

    @property
    def cors_origins(self) -> List[str]:
        return self.get('cors_origins', [])

    @property
    def is_debug(self) -> bool:
        return self.get('debug', False)

    @property
    def log_level(self) -> str:
        return self.get('log_level', 'INFO')

# ===============================================================
# 🎯 AI 설정 클래스
# ===============================================================

class AIConfig(OptimalConfigBase):
    """
    🎯 AI 모델 및 파이프라인 설정
    - 모델 경로
    - 디바이스 설정
    - 성능 최적화
    - M3 Max 특화
    """

    def _create_base_config(self) -> Dict[str, Any]:
        """기본 AI 설정 생성"""
        
        # 디바이스 자동 감지
        device = self._auto_detect_device()
        
        return {
            # 디바이스 설정
            'device': device,
            'device_type': self._get_device_type(device),
            'enable_mps': device == 'mps',
            'enable_cuda': device == 'cuda',
            'mixed_precision': device != 'cpu',
            
            # 메모리 설정
            'memory_gb': self.system_info['available_memory_gb'],
            'max_memory_usage': self._get_optimal_memory_usage(),
            'memory_efficient': True,
            'enable_memory_monitoring': True,
            
            # 모델 경로
            'models_dir': './models/ai_models',
            'checkpoints_dir': './models/ai_models/checkpoints',
            'cache_dir': './models/cache',
            
            # 성능 설정
            'batch_size': self._get_optimal_batch_size(device),
            'num_workers': min(4, self.system_info['cpu_count']),
            'pin_memory': device != 'cpu',
            'non_blocking': True,
            
            # 파이프라인 설정
            'pipeline_quality': self._get_optimal_quality_level(),
            'enable_optimization': True,
            'enable_caching': True,
            'enable_parallel': True,
            'max_concurrent_requests': self._get_optimal_concurrent_requests(),
            
            # M3 Max 특화 설정
            'is_m3_max': self.system_info['is_m3_max'],
            'm3_max_optimizations': self.system_info['is_m3_max'],
            'metal_performance_shaders': self.system_info['is_m3_max'] and device == 'mps',
            
            # 모델별 설정
            'models': {
                'human_parsing': {
                    'model_name': 'graphonomy',
                    'input_size': (512, 512),
                    'num_classes': 20
                },
                'pose_estimation': {
                    'model_complexity': 2,
                    'min_detection_confidence': 0.7
                },
                'cloth_segmentation': {
                    'method': 'auto',
                    'quality_threshold': 0.7
                },
                'geometric_matching': {
                    'method': 'auto',
                    'max_iterations': 1500 if self.system_info['is_m3_max'] else 1000
                },
                'cloth_warping': {
                    'physics_enabled': True,
                    'deformation_strength': 0.7,
                    'enable_fabric_physics': True
                }
            }
        }

    def _auto_detect_device(self) -> str:
        """🖥️ AI 디바이스 자동 감지"""
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

    def _get_device_type(self, device: str) -> str:
        """🔧 디바이스 타입 결정"""
        if device == 'mps':
            return 'apple_silicon'
        elif device == 'cuda':
            return 'nvidia_gpu'
        else:
            return 'cpu'

    def _get_optimal_memory_usage(self) -> str:
        """💾 최적 메모리 사용량 계산"""
        available_gb = self.system_info['available_memory_gb']
        
        if available_gb >= 128:  # M3 Max 128GB
            return '64GB'
        elif available_gb >= 64:
            return '32GB'
        elif available_gb >= 32:
            return '16GB'
        elif available_gb >= 16:
            return '8GB'
        else:
            return '4GB'

    def _get_optimal_batch_size(self, device: str) -> int:
        """📦 최적 배치 크기 계산"""
        memory_gb = self.system_info['available_memory_gb']
        
        if device == 'mps' and self.system_info['is_m3_max']:
            # M3 Max는 더 큰 배치 크기 가능
            if memory_gb >= 128:
                return 8
            elif memory_gb >= 64:
                return 4
            else:
                return 2
        elif device == 'cuda':
            # NVIDIA GPU
            return 4
        else:
            # CPU
            return 1

    def _get_optimal_quality_level(self) -> str:
        """🎨 최적 품질 레벨 결정"""
        if self.system_info['is_m3_max']:
            return 'ultra'
        elif self.system_info['available_memory_gb'] >= 32:
            return 'high'
        elif self.system_info['available_memory_gb'] >= 16:
            return 'medium'
        else:
            return 'basic'

    def _get_optimal_concurrent_requests(self) -> int:
        """🚀 최적 동시 요청 수 계산"""
        memory_gb = self.system_info['available_memory_gb']
        cpu_count = self.system_info['cpu_count']
        
        if self.system_info['is_m3_max']:
            return min(8, cpu_count)
        elif memory_gb >= 32:
            return min(4, cpu_count)
        else:
            return min(2, cpu_count)

# ===============================================================
# 🎯 통합 설정 관리자
# ===============================================================

class Settings:
    """
    🎯 통합 설정 관리자
    - 모든 설정을 하나로 통합
    - 편의 속성 제공
    - 캐싱 지원
    """

    def __init__(
        self,
        env: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs
    ):
        """
        통합 설정 초기화
        
        Args:
            env: 환경 (None=자동감지)
            config_path: 설정 파일 경로
            **kwargs: 추가 설정
        """
        self.app = AppConfig(env=env, config_path=config_path, **kwargs)
        self.ai = AIConfig(env=env, config_path=config_path, **kwargs)
        
        # 편의 속성들 (하위 호환성)
        self._setup_convenience_properties()
        
        logger.info("🎯 통합 설정 시스템 초기화 완료")

    def _setup_convenience_properties(self):
        """편의 속성 설정 (기존 코드 호환성)"""
        # property로 정의된 속성들은 제외
        reserved_properties = {
            'APP_NAME', 'DEBUG', 'HOST', 'PORT', 'DATABASE_URL', 
            'CORS_ORIGINS', 'DEVICE', 'USE_GPU', 'IS_M3_MAX'
        }
        
        # 앱 설정 직접 접근 (property 충돌 방지)
        for key, value in self.app.to_dict().items():
            attr_name = key.upper()
            if attr_name not in reserved_properties:
                setattr(self, attr_name, value)
        
        # AI 설정 직접 접근 (AI_ 접두사, property 충돌 방지)
        for key, value in self.ai.to_dict().items():
            attr_name = f'AI_{key.upper()}'
            if attr_name not in reserved_properties:
                setattr(self, attr_name, value)

    # 주요 속성들 (하위 호환성)
    @property
    def APP_NAME(self) -> str:
        return self.app.get('app_name')

    @property
    def DEBUG(self) -> bool:
        return self.app.get('debug')

    @property
    def HOST(self) -> str:
        return self.app.get('host')

    @property
    def PORT(self) -> int:
        return self.app.get('port')

    @property
    def DATABASE_URL(self) -> str:
        return self.app.get('database_url')

    @property
    def CORS_ORIGINS(self) -> List[str]:
        return self.app.get('cors_origins')

    @property
    def DEVICE(self) -> str:
        return self.ai.get('device')

    @property
    def USE_GPU(self) -> bool:
        return self.ai.get('device') != 'cpu'

    @property
    def IS_M3_MAX(self) -> bool:
        return self.ai.get('is_m3_max', False)

# ===============================================================
# 🎯 전역 설정 인스턴스 및 팩토리 함수들
# ===============================================================

@lru_cache()
def get_settings(
    env: Optional[str] = None,
    config_path: Optional[str] = None,
    **kwargs
) -> Settings:
    """전역 설정 인스턴스 (캐시됨)"""
    return Settings(env=env, config_path=config_path, **kwargs)

# 기본 설정 인스턴스
settings = get_settings()

# 편의 함수들
def get_app_config() -> AppConfig:
    """앱 설정 반환"""
    return settings.app

def get_ai_config() -> AIConfig:
    """AI 설정 반환"""
    return settings.ai

# 하위 호환성 지원 (기존 코드 100% 지원)
APP_NAME = settings.APP_NAME
DEBUG = settings.DEBUG
HOST = settings.HOST
PORT = settings.PORT
DATABASE_URL = settings.DATABASE_URL
DEVICE = settings.DEVICE
USE_GPU = settings.USE_GPU
IS_M3_MAX = settings.IS_M3_MAX

logger.info(f"🎯 최적 설정 시스템 로드 완료 - 환경: {settings.app.env}, 디바이스: {DEVICE}")

if IS_M3_MAX:
    logger.info("🍎 M3 Max 최적화 활성화")
if USE_GPU:
    logger.info(f"🎮 GPU 가속 활성화: {DEVICE}")

__all__ = [
    'OptimalConfigBase', 'AppConfig', 'AIConfig', 'Settings',
    'get_settings', 'settings', 'get_app_config', 'get_ai_config',
    'APP_NAME', 'DEBUG', 'HOST', 'PORT', 'DATABASE_URL', 
    'DEVICE', 'USE_GPU', 'IS_M3_MAX'
]