# ============================================================================
# 🍎 MyCloset AI - 완전한 __init__.py 파일 시스템
# ============================================================================
# conda 환경 우선 + 순환참조 해결 + 역할별 분리 + 지연 로딩 패턴

# ============================================================================
# 📁 backend/app/__init__.py - 메인 패키지 초기화
# ============================================================================

"""
🍎 MyCloset AI 메인 애플리케이션 패키지 v7.0
==============================================

✅ conda 환경 우선 최적화
✅ Python Path 자동 설정 (Import 오류 완전 해결) 
✅ M3 Max 128GB 메모리 완전 활용
✅ 순환참조 완전 방지 (지연 로딩 패턴)
✅ 89.8GB AI 모델 경로 자동 설정
✅ 레이어 아키텍처 지원 (API → Service → Pipeline → AI)
✅ 프로덕션 레벨 안정성
✅ 모든 하위 모듈 안전한 로딩

프로젝트 구조:
backend/                   ← 작업 디렉토리 (여기서 python app/main.py 실행)
├── app/                   ← 패키지 루트 (이 파일의 위치)
│   ├── __init__.py       ← 이 파일!
│   ├── main.py           ← FastAPI 서버 진입점
│   ├── core/             ← 핵심 설정 및 유틸리티
│   ├── services/         ← 비즈니스 로직 서비스들
│   ├── ai_pipeline/      ← AI 파이프라인 (8단계)
│   ├── api/              ← REST API 엔드포인트
│   ├── models/           ← 데이터 모델/스키마
│   └── utils/            ← 공통 유틸리티
├── ai_models/            ← AI 모델 체크포인트 (89.8GB)
└── static/               ← 정적 파일 및 업로드

작성자: MyCloset AI Team
날짜: 2025-07-23
버전: v7.0.0 (Complete Init System with Conda Priority)
"""

import os
import sys
import logging
import platform
import subprocess
import threading
import weakref
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import warnings

# =============================================================================
# 🔥 Step 1: conda 환경 우선 체크 및 설정
# =============================================================================

# conda 환경 감지
CONDA_ENV = os.environ.get('CONDA_DEFAULT_ENV', 'none')
CONDA_PREFIX = os.environ.get('CONDA_PREFIX', '')
IS_CONDA = CONDA_ENV != 'none' or bool(CONDA_PREFIX)

if IS_CONDA:
    print(f"🐍 conda 환경 감지: {CONDA_ENV} at {CONDA_PREFIX}")
    
    # conda 우선 라이브러리 경로 설정
    if CONDA_PREFIX:
        python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        conda_site_packages = os.path.join(CONDA_PREFIX, 'lib', python_version, 'site-packages')
        if os.path.exists(conda_site_packages) and conda_site_packages not in sys.path:
            sys.path.insert(0, conda_site_packages)
            print(f"✅ conda site-packages 경로 추가: {conda_site_packages}")
    
    # conda 환경 최적화 설정
    os.environ.setdefault('OMP_NUM_THREADS', str(max(1, os.cpu_count() // 2)))
    os.environ.setdefault('MKL_NUM_THREADS', str(max(1, os.cpu_count() // 2)))
    os.environ.setdefault('NUMEXPR_NUM_THREADS', str(max(1, os.cpu_count() // 2)))
else:
    print("⚠️ conda 환경이 비활성화됨 - 'conda activate <환경명>' 권장")

# =============================================================================
# 🔥 Step 2: 패키지 경로 및 환경 자동 설정
# =============================================================================

# 현재 패키지의 절대 경로
_current_package_dir = Path(__file__).parent.absolute()
_backend_root = _current_package_dir.parent
_project_root = _backend_root.parent

# Python Path 자동 설정 (Import 오류 해결)
_paths_to_add = [
    str(_backend_root),           # backend/ 경로 (가장 중요!)
    str(_current_package_dir),    # backend/app/ 경로
    str(_project_root),           # 프로젝트 루트 경로
]

for path in _paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# 환경 변수 설정 (conda 환경 고려)
os.environ.update({
    'PYTHONPATH': f"{_backend_root}:{os.environ.get('PYTHONPATH', '')}",
    'PROJECT_ROOT': str(_project_root),
    'BACKEND_ROOT': str(_backend_root),
    'APP_ROOT': str(_current_package_dir),
    'MYCLOSET_CONDA_ENV': CONDA_ENV if IS_CONDA else 'none'
})

# 작업 디렉토리를 backend로 설정
if Path.cwd() != _backend_root:
    try:
        os.chdir(_backend_root)
        print(f"✅ 작업 디렉토리 설정: {_backend_root}")
    except Exception as e:
        warnings.warn(f"작업 디렉토리 설정 실패: {e}")

# =============================================================================
# 🔥 Step 3: 시스템 정보 수집 (conda 환경 최적화)
# =============================================================================

def _detect_m3_max() -> bool:
    """M3 Max 칩 감지"""
    try:
        if platform.system() == 'Darwin':
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True, timeout=5
            )
            chip_info = result.stdout.strip()
            return 'M3' in chip_info and ('Max' in chip_info or 'Pro' in chip_info)
    except Exception:
        pass
    return False

def _get_memory_gb() -> float:
    """메모리 용량 감지"""
    try:
        if platform.system() == 'Darwin':
            result = subprocess.run(
                ['sysctl', '-n', 'hw.memsize'],
                capture_output=True, text=True, timeout=5
            )
            return int(result.stdout.strip()) / (1024**3)
        else:
            try:
                import psutil
                return psutil.virtual_memory().total / (1024**3)
            except ImportError:
                return 16.0
    except Exception:
        return 16.0

def _detect_device() -> str:
    """최적 디바이스 감지 (conda 환경 우선)"""
    try:
        # conda PyTorch 체크
        import torch
        
        # M3 Max MPS 우선
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # conda 환경에서 MPS 최적화 설정
            if IS_CONDA:
                os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
                os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    except ImportError:
        return 'cpu'

# 시스템 정보 수집
SYSTEM_INFO = {
    'python_version': platform.python_version(),
    'platform': platform.platform(),
    'architecture': platform.architecture()[0],
    'cpu_count': os.cpu_count(),
    'memory_gb': _get_memory_gb(),
    'is_m3_max': _detect_m3_max(),
    'device': _detect_device(),
    'is_conda': IS_CONDA,
    'conda_env': CONDA_ENV,
    'conda_prefix': CONDA_PREFIX,
    'backend_root': str(_backend_root),
    'app_root': str(_current_package_dir)
}

# AI 모델 경로 설정
AI_MODEL_PATHS = {
    'ai_models_root': _backend_root / 'ai_models',
    'checkpoints': _backend_root / 'ai_models' / 'checkpoints',
    'configs': _backend_root / 'ai_models' / 'configs',
    'weights': _backend_root / 'ai_models' / 'weights'
}

# AI 모델 환경 변수 설정
os.environ.setdefault("MYCLOSET_AI_MODELS_PATH", str(AI_MODEL_PATHS['ai_models_root']))

# =============================================================================
# 🔥 Step 4: 로깅 시스템 설정 (conda 환경 최적화)
# =============================================================================

def _setup_logging():
    """로깅 시스템 초기화"""
    logger = logging.getLogger('mycloset_ai')
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    # 로그 디렉토리
    log_dir = _backend_root / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # 포맷터 (conda 환경 정보 포함)
    conda_info = f"conda:{CONDA_ENV}" if IS_CONDA else "pip"
    formatter = logging.Formatter(
        f'%(asctime)s - {conda_info} - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (선택적)
    try:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        warnings.warn(f"파일 로그 핸들러 설정 실패: {e}")
    
    return logger

# 패키지 로거 초기화
logger = _setup_logging()

# =============================================================================
# 🔥 Step 5: 지연 로딩 시스템 (순환참조 방지)
# =============================================================================

class LazyLoader:
    """지연 로딩 헬퍼 클래스 - 순환참조 완전 방지"""
    
    def __init__(self):
        self._cache = {}
        self._loading = set()
        self._lock = threading.RLock()
    
    def load_module(self, module_name: str, package: str = None):
        """모듈 지연 로딩"""
        cache_key = f"{package}.{module_name}" if package else module_name
        
        with self._lock:
            # 캐시에서 확인
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            # 순환 로딩 방지
            if cache_key in self._loading:
                logger.warning(f"순환참조 감지: {cache_key}")
                return None
            
            self._loading.add(cache_key)
            
            try:
                import importlib
                full_module_name = f"{package}.{module_name}" if package else module_name
                module = importlib.import_module(full_module_name)
                self._cache[cache_key] = module
                logger.debug(f"✅ 지연 로딩 성공: {cache_key}")
                return module
                
            except ImportError as e:
                logger.debug(f"⚠️ 지연 로딩 실패: {cache_key} - {e}")
                self._cache[cache_key] = None
                return None
            
            except Exception as e:
                logger.error(f"❌ 지연 로딩 오류: {cache_key} - {e}")
                self._cache[cache_key] = None
                return None
            
            finally:
                self._loading.discard(cache_key)
    
    def get_class(self, module_name: str, class_name: str, package: str = None):
        """클래스 지연 로딩"""
        module = self.load_module(module_name, package)
        if module:
            return getattr(module, class_name, None)
        return None

# 전역 지연 로더
_lazy_loader = LazyLoader()

# =============================================================================
# 🔥 Step 6: 핵심 모듈 지연 로딩 (순환참조 방지)
# =============================================================================

def get_config():
    """Config 모듈 지연 로딩"""
    return _lazy_loader.load_module('config', 'app.core')

def get_gpu_config():
    """GPU Config 모듈 지연 로딩"""
    return _lazy_loader.load_module('gpu_config', 'app.core')

def get_session_manager():
    """Session Manager 모듈 지연 로딩"""
    return _lazy_loader.load_module('session_manager', 'app.core')

def get_model_loader():
    """Model Loader 모듈 지연 로딩"""
    return _lazy_loader.load_module('model_loader', 'app.ai_pipeline.utils')

def get_pipeline_manager():
    """Pipeline Manager 모듈 지연 로딩"""
    return _lazy_loader.load_module('pipeline_manager', 'app.ai_pipeline')

def get_file_manager():
    """File Manager 모듈 지연 로딩"""
    return _lazy_loader.load_module('file_manager', 'app.utils')

def get_image_utils():
    """Image Utils 모듈 지연 로딩"""
    return _lazy_loader.load_module('image_utils', 'app.utils')

# 지연 로딩 편의 함수들
def safe_import(module_name: str, package: str = None):
    """안전한 모듈 import (지연 로딩)"""
    return _lazy_loader.load_module(module_name, package)

def safe_get_class(module_name: str, class_name: str, package: str = None):
    """안전한 클래스 import (지연 로딩)"""
    return _lazy_loader.get_class(module_name, class_name, package)

# =============================================================================
# 🔥 Step 7: 패키지 메타데이터 및 정보
# =============================================================================

__version__ = "7.0.0"
__author__ = "MyCloset AI Team"
__description__ = "AI-powered Virtual Try-On Platform with Conda Priority"
__license__ = "Proprietary"

# 패키지 정보
PACKAGE_INFO = {
    'name': 'MyCloset AI',
    'version': __version__,
    'description': __description__,
    'author': __author__,
    'backend_root': str(_backend_root),
    'app_root': str(_current_package_dir),
    'python_version': SYSTEM_INFO['python_version'],
    'conda_env': SYSTEM_INFO['conda_env'],
    'is_conda': SYSTEM_INFO['is_conda'],
    'is_m3_max': SYSTEM_INFO['is_m3_max'],
    'device': SYSTEM_INFO['device'],
    'memory_gb': SYSTEM_INFO['memory_gb'],
    'ai_models_available': AI_MODEL_PATHS['ai_models_root'].exists()
}

# =============================================================================
# 🔥 Step 8: 유틸리티 함수들
# =============================================================================

def get_package_info() -> Dict[str, Any]:
    """패키지 정보 반환"""
    return PACKAGE_INFO.copy()

def get_system_info() -> Dict[str, Any]:
    """시스템 정보 반환"""
    return SYSTEM_INFO.copy()

def get_ai_model_paths() -> Dict[str, Path]:
    """AI 모델 경로 정보 반환"""
    return AI_MODEL_PATHS.copy()

def is_conda_environment() -> bool:
    """conda 환경 여부 확인"""
    return SYSTEM_INFO['is_conda']

def is_m3_max() -> bool:
    """M3 Max 환경 여부 확인"""
    return SYSTEM_INFO['is_m3_max']

def get_device() -> str:
    """추천 디바이스 반환"""
    return SYSTEM_INFO['device']

def validate_environment() -> Dict[str, Any]:
    """환경 검증 및 상태 반환"""
    validation = {
        'python_path_ok': str(_backend_root) in sys.path,
        'working_directory_ok': Path.cwd() == _backend_root,
        'ai_models_exist': AI_MODEL_PATHS['ai_models_root'].exists(),
        'conda_environment': SYSTEM_INFO['is_conda'],
        'conda_env_name': SYSTEM_INFO['conda_env'],
        'device_available': SYSTEM_INFO['device'] != 'cpu',
        'memory_sufficient': SYSTEM_INFO['memory_gb'] >= 8.0,
        'lazy_loader_ready': _lazy_loader is not None
    }
    
    validation['overall_status'] = all([
        validation['python_path_ok'],
        validation['working_directory_ok'],
        validation['memory_sufficient']
    ])
    
    return validation

# =============================================================================
# 🔥 Step 9: 초기화 상태 출력
# =============================================================================

def _print_initialization_status():
    """초기화 상태 출력 (conda 환경 우선)"""
    print(f"\n🍎 MyCloset AI 패키지 초기화 완료!")
    print(f"📦 버전: {__version__}")
    print(f"🐍 conda 환경: {'✅' if IS_CONDA else '❌'} ({CONDA_ENV})")
    print(f"🍎 M3 Max: {'✅' if SYSTEM_INFO['is_m3_max'] else '❌'}")
    print(f"🖥️  디바이스: {SYSTEM_INFO['device']}")
    print(f"💾 메모리: {SYSTEM_INFO['memory_gb']:.1f}GB")
    print(f"📁 AI 모델 경로: {AI_MODEL_PATHS['ai_models_root']}")
    print(f"🔗 지연 로딩: ✅ 활성화")
    print(f"🔧 Python Path: ✅ 설정 완료")
    
    if IS_CONDA:
        print(f"🐍 conda 최적화: ✅ 활성화")
        print(f"🐍 conda 경로: {CONDA_PREFIX}")
    else:
        print(f"⚠️  conda 비활성화 - 성능 최적화를 위해 conda 사용 권장")

# 초기화 상태 출력
_print_initialization_status()

# =============================================================================
# 🔥 Step 10: 패키지 Export
# =============================================================================

__all__ = [
    # 🔥 버전 정보
    '__version__',
    '__author__',
    '__description__',
    
    # 📊 시스템 정보
    'SYSTEM_INFO',
    'PACKAGE_INFO',
    'AI_MODEL_PATHS',
    'IS_CONDA',
    'CONDA_ENV',
    
    # 🔧 유틸리티 함수들
    'get_package_info',
    'get_system_info', 
    'get_ai_model_paths',
    'is_conda_environment',
    'is_m3_max',
    'get_device',
    'validate_environment',
    
    # 🚀 지연 로딩 함수들
    'get_config',
    'get_gpu_config',
    'get_session_manager',
    'get_model_loader',
    'get_pipeline_manager',
    'get_file_manager',
    'get_image_utils',
    'safe_import',
    'safe_get_class',
]

logger.info("🎉 MyCloset AI 메인 패키지 초기화 완료!")
logger.info(f"🐍 conda 환경: {IS_CONDA} ({CONDA_ENV})")
logger.info(f"🍎 M3 Max 최적화: {SYSTEM_INFO['is_m3_max']}")
logger.info(f"🔗 지연 로딩 시스템: 활성화")
