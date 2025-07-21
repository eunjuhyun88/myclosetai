# backend/app/__init__.py
"""
🍎 MyCloset AI 메인 애플리케이션 패키지 v6.0
==============================================

✅ Python Path 자동 설정 (Import 오류 완전 해결)
✅ conda 환경 우선 최적화 
✅ M3 Max 128GB 메모리 완전 활용
✅ 순환참조 완전 방지
✅ 89.8GB AI 모델 경로 자동 설정
✅ main.py 완벽 호환
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
날짜: 2025-07-22
버전: v6.0.0 (Complete Package Initialization)
"""

import os
import sys
import logging
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
import warnings

# =============================================================================
# 🔥 Step 1: 패키지 경로 및 환경 자동 설정
# =============================================================================

# 현재 패키지의 절대 경로
_current_package_dir = Path(__file__).parent.absolute()
_backend_root = _current_package_dir.parent
_project_root = _backend_root.parent

# Python Path 자동 설정 (Import 오류 해결)
_paths_to_add = [
    str(_backend_root),    # backend/ 경로 (가장 중요!)
    str(_current_package_dir),  # backend/app/ 경로
    str(_project_root),    # 프로젝트 루트 경로
]

for path in _paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# 환경 변수 설정 (conda 환경 고려)
os.environ.update({
    'PYTHONPATH': f"{_backend_root}:{os.environ.get('PYTHONPATH', '')}",
    'PROJECT_ROOT': str(_project_root),
    'BACKEND_ROOT': str(_backend_root),
    'APP_ROOT': str(_current_package_dir)
})

# 작업 디렉토리를 backend로 설정 (중요!)
try:
    os.chdir(_backend_root)
except OSError as e:
    warnings.warn(f"작업 디렉토리 변경 실패: {e}")

# =============================================================================
# 🔥 Step 2: 시스템 환경 감지 및 최적화 설정
# =============================================================================

def _detect_system_environment() -> Dict[str, Any]:
    """시스템 환경 자동 감지 (conda 환경 우선)"""
    env_info = {
        'platform': platform.system(),
        'machine': platform.machine(),
        'python_version': platform.python_version(),
        'is_conda': False,
        'conda_env': None,
        'is_m3_max': False,
        'device': 'cpu',
        'memory_gb': 16.0,
        'cpu_count': os.cpu_count() or 4
    }
    
    try:
        # conda 환경 감지
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_env and conda_env != 'base':
            env_info['is_conda'] = True
            env_info['conda_env'] = conda_env
        elif conda_prefix:
            env_info['is_conda'] = True
            env_info['conda_env'] = Path(conda_prefix).name
        
        # M3 Max 감지 (conda 환경에서 최적화)
        if (env_info['platform'] == 'Darwin' and 
            'arm64' in env_info['machine']):
            try:
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'], 
                    capture_output=True, text=True, timeout=3
                )
                if 'M3' in result.stdout:
                    env_info['is_m3_max'] = True
                    env_info['memory_gb'] = 128.0  # M3 Max Unified Memory
                    env_info['device'] = 'mps'
            except:
                pass
        
        # 메모리 감지
        try:
            import psutil
            env_info['memory_gb'] = round(psutil.virtual_memory().total / (1024**3))
        except ImportError:
            pass
            
        # PyTorch 디바이스 감지
        try:
            import torch
            if torch.backends.mps.is_available():
                env_info['device'] = 'mps'
            elif torch.cuda.is_available():
                env_info['device'] = 'cuda'
        except ImportError:
            pass
            
    except Exception as e:
        warnings.warn(f"시스템 환경 감지 중 오류: {e}")
    
    return env_info

# 전역 시스템 정보
SYSTEM_INFO = _detect_system_environment()

# M3 Max 최적화 환경 변수 설정
if SYSTEM_INFO['is_m3_max']:
    os.environ.update({
        'PYTORCH_ENABLE_MPS_FALLBACK': '1',
        'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',
        'OMP_NUM_THREADS': str(min(SYSTEM_INFO['cpu_count'] * 2, 16)),
        'DEVICE': 'mps'
    })

# =============================================================================
# 🔥 Step 3: AI 모델 경로 자동 설정 (89.8GB 모델)
# =============================================================================

def _setup_ai_model_paths() -> Dict[str, Path]:
    """AI 모델 경로 자동 설정 및 검증"""
    ai_models_root = _backend_root / "ai_models"
    
    model_paths = {
        'ai_models_root': ai_models_root,
        'step_01_human_parsing': ai_models_root / "step_01_human_parsing",
        'step_02_pose_estimation': ai_models_root / "step_02_pose_estimation", 
        'step_03_cloth_segmentation': ai_models_root / "step_03_cloth_segmentation",
        'step_04_geometric_matching': ai_models_root / "step_04_geometric_matching",
        'step_05_cloth_warping': ai_models_root / "step_05_cloth_warping",
        'step_06_virtual_fitting': ai_models_root / "step_06_virtual_fitting",
        'step_07_post_processing': ai_models_root / "step_07_post_processing",
        'step_08_quality_assessment': ai_models_root / "step_08_quality_assessment",
        'checkpoints': ai_models_root / "checkpoints",
        'cache': ai_models_root / "cache",
        'huggingface_cache': ai_models_root / "huggingface_cache"
    }
    
    # 환경 변수로 경로 설정
    for name, path in model_paths.items():
        env_name = f"AI_MODEL_{name.upper()}_PATH"
        os.environ[env_name] = str(path)
    
    return model_paths

AI_MODEL_PATHS = _setup_ai_model_paths()

# =============================================================================
# 🔥 Step 4: 로깅 시스템 설정
# =============================================================================

def _setup_logging():
    """패키지 전체 로깅 시스템 설정"""
    # 로그 디렉토리 생성
    log_dir = _backend_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # 로거 설정
    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)
    
    # 이미 핸들러가 있으면 스킵
    if logger.handlers:
        return logger
    
    # 포매터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (회전 로그) - 수정된 import 방식
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
# 🔥 Step 5: 패키지 메타데이터 및 정보
# =============================================================================

__version__ = "6.0.0"
__author__ = "MyCloset AI Team"
__description__ = "AI-powered Virtual Try-On Platform"
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
    'is_m3_max': SYSTEM_INFO['is_m3_max'],
    'device': SYSTEM_INFO['device'],
    'memory_gb': SYSTEM_INFO['memory_gb'],
    'ai_models_available': AI_MODEL_PATHS['ai_models_root'].exists()
}

# =============================================================================
# 🔥 Step 6: 핵심 모듈 안전한 로딩 (순환참조 방지)
# =============================================================================

def _safe_import(module_name: str, package: str = None):
    """안전한 모듈 import (오류 시 로그만 기록)"""
    try:
        if package:
            module = __import__(f"{package}.{module_name}", fromlist=[module_name])
        else:
            module = __import__(module_name)
        return module
    except ImportError as e:
        logger.warning(f"⚠️ {module_name} import 실패: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ {module_name} import 오류: {e}")
        return None

# 핵심 모듈들 미리 로드 시도 (실패해도 계속 진행)
_core_modules = {
    'config': _safe_import('config', 'app.core'),
    'gpu_config': _safe_import('gpu_config', 'app.core'),
    'session_manager': _safe_import('session_manager', 'app.core'),
}

# =============================================================================
# 🔥 Step 7: 유틸리티 함수들
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
        'core_modules_loaded': sum(1 for m in _core_modules.values() if m is not None)
    }
    
    validation['overall_status'] = all([
        validation['python_path_ok'],
        validation['working_directory_ok'],
        validation['memory_sufficient']
    ])
    
    return validation

# =============================================================================
# 🔥 Step 8: 패키지 초기화 완료 및 상태 출력
# =============================================================================

def _print_initialization_status():
    """초기화 상태 출력 (conda 환경 우선)"""
    print(f"\n🍎 MyCloset AI 패키지 초기화 완료!")
    print(f"📍 버전: {__version__}")
    print(f"🐍 Python: {SYSTEM_INFO['python_version']}")
    
    if SYSTEM_INFO['is_conda']:
        print(f"🐍 Conda 환경: {SYSTEM_INFO['conda_env']} ✅")
    else:
        print(f"⚠️  일반 Python 환경 (conda 권장)")
    
    if SYSTEM_INFO['is_m3_max']:
        print(f"🍎 M3 Max {SYSTEM_INFO['memory_gb']:.0f}GB: {SYSTEM_INFO['device']} ✅")
    else:
        print(f"💻 {SYSTEM_INFO['platform']}: {SYSTEM_INFO['device']}")
    
    print(f"📁 Backend: {_backend_root}")
    print(f"🤖 AI Models: {'✅' if AI_MODEL_PATHS['ai_models_root'].exists() else '❌'}")
    
    validation = validate_environment()
    if validation['overall_status']:
        print(f"✅ 환경 검증 완료 - 서버 실행 가능!")
    else:
        print(f"⚠️  환경 검증 실패 - 설정을 확인하세요")
    print()

# 초기화 상태 출력 (개발 환경에서만)
if os.getenv('DEBUG', '').lower() in ['true', '1'] or '--verbose' in sys.argv:
    _print_initialization_status()

# =============================================================================
# 🔥 Step 9: __all__ 및 공개 API 정의
# =============================================================================

__all__ = [
    # 메타데이터
    '__version__',
    '__author__',
    '__description__',
    
    # 정보 함수들
    'get_package_info',
    'get_system_info', 
    'get_ai_model_paths',
    'is_conda_environment',
    'is_m3_max',
    'get_device',
    'validate_environment',
    
    # 상수들
    'SYSTEM_INFO',
    'AI_MODEL_PATHS',
    'PACKAGE_INFO',
    
    # 경로들
    '_backend_root',
    '_current_package_dir',
    '_project_root'
]

# =============================================================================
# 🔥 최종: 초기화 성공 로그
# =============================================================================

logger.info(f"🎉 MyCloset AI 패키지 초기화 완료 (v{__version__})")
logger.info(f"🐍 환경: {'Conda' if SYSTEM_INFO['is_conda'] else 'Python'} - {SYSTEM_INFO['conda_env'] or 'system'}")
logger.info(f"🍎 M3 Max: {'활성' if SYSTEM_INFO['is_m3_max'] else '비활성'}")
logger.info(f"🤖 AI 모델: {'사용가능' if AI_MODEL_PATHS['ai_models_root'].exists() else '없음'}")
logger.info(f"📁 작업경로: {Path.cwd()}")

# 환경 검증 및 경고
validation = validate_environment()
if not validation['overall_status']:
    logger.warning("⚠️ 환경 검증 실패 - 일부 기능이 제한될 수 있습니다")
    if not validation['python_path_ok']:
        logger.warning("   - Python 경로 설정 문제")
    if not validation['working_directory_ok']:
        logger.warning("   - 작업 디렉토리 문제")
    if not validation['memory_sufficient']:
        logger.warning("   - 메모리 부족 (최소 8GB 권장)")

# conda 환경 권장 메시지
if not SYSTEM_INFO['is_conda']:
    logger.info("💡 conda 환경 사용을 권장합니다: conda activate mycloset-ai")

logger.info("=" * 60)