# backend/app/__init__.py
"""
🍎 MyCloset AI 메인 애플리케이션 패키지 v7.0
==============================================

✅ 단순화된 패키지 초기화 (복잡성 제거)
✅ conda 환경 우선 최적화 
✅ M3 Max 128GB 메모리 완전 활용
✅ 순환참조 완전 방지
✅ Python Path 자동 설정
✅ 시스템 정보 중앙 관리
✅ 프로덕션 레벨 안정성

작성자: MyCloset AI Team
날짜: 2025-07-23
버전: v7.0.0 (Simplified Package Initialization)
"""

import os
import sys
import logging
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

# 경고 무시 설정
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# =============================================================================
# 🔥 패키지 경로 자동 설정
# =============================================================================

# 현재 패키지의 절대 경로
_CURRENT_DIR = Path(__file__).parent.absolute()
_BACKEND_ROOT = _CURRENT_DIR.parent
_PROJECT_ROOT = _BACKEND_ROOT.parent

# Python Path 설정 (Import 오류 해결)
_paths_to_add = [
    str(_BACKEND_ROOT),      # backend/ 경로 (최우선)
    str(_CURRENT_DIR),       # backend/app/ 경로
]

for path in _paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# 환경 변수 설정
os.environ.update({
    'PROJECT_ROOT': str(_PROJECT_ROOT),
    'BACKEND_ROOT': str(_BACKEND_ROOT),
    'APP_ROOT': str(_CURRENT_DIR),
    'PYTHONPATH': f"{_BACKEND_ROOT}:{os.environ.get('PYTHONPATH', '')}"
})

# 작업 디렉토리를 backend로 설정
try:
    os.chdir(_BACKEND_ROOT)
except OSError:
    pass

# =============================================================================
# 🔥 시스템 정보 감지 (중앙 관리)
# =============================================================================

def _detect_conda_environment() -> bool:
    """conda 환경 감지"""
    return (
        'CONDA_DEFAULT_ENV' in os.environ or
        'CONDA_PREFIX' in os.environ or
        sys.executable.find('conda') != -1 or
        sys.executable.find('miniconda') != -1
    )

def _detect_m3_max() -> bool:
    """M3 Max 감지"""
    try:
        if platform.system() == 'Darwin':
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'], 
                capture_output=True, text=True, timeout=5
            )
            chip_info = result.stdout.strip()
            return 'M3' in chip_info and 'Max' in chip_info
    except:
        pass
    return False

def _get_memory_gb() -> float:
    """시스템 메모리 조회 (GB)"""
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024**3), 1)
    except:
        return 16.0  # 기본값

def _get_device() -> str:
    """최적 디바이스 감지"""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    except:
        return "cpu"

# 시스템 정보 수집
_SYSTEM_INFO = {
    'platform': platform.system(),
    'python_version': platform.python_version(),
    'is_conda': _detect_conda_environment(),
    'conda_env': os.environ.get('CONDA_DEFAULT_ENV', 'none'),
    'is_m3_max': _detect_m3_max(),
    'memory_gb': _get_memory_gb(),
    'cpu_count': os.cpu_count() or 4,
    'device': _get_device(),
    'project_root': str(_PROJECT_ROOT),
    'backend_root': str(_BACKEND_ROOT),
    'app_root': str(_CURRENT_DIR)
}

# =============================================================================
# 🔥 공용 함수들 (Export용)
# =============================================================================

def get_system_info() -> Dict[str, Any]:
    """시스템 정보 반환"""
    return _SYSTEM_INFO.copy()

def is_conda_environment() -> bool:
    """conda 환경인지 확인"""
    return _SYSTEM_INFO['is_conda']

def is_m3_max() -> bool:
    """M3 Max인지 확인"""
    return _SYSTEM_INFO['is_m3_max']

def get_device() -> str:
    """최적 디바이스 반환"""
    return _SYSTEM_INFO['device']

def get_project_paths() -> Dict[str, str]:
    """프로젝트 경로들 반환"""
    return {
        'project_root': _SYSTEM_INFO['project_root'],
        'backend_root': _SYSTEM_INFO['backend_root'],
        'app_root': _SYSTEM_INFO['app_root'],
        'models_dir': str(_PROJECT_ROOT / 'backend' / 'ai_models'),
        'static_dir': str(_PROJECT_ROOT / 'backend' / 'static'),
        'upload_dir': str(_PROJECT_ROOT / 'backend' / 'static' / 'uploads'),
        'results_dir': str(_PROJECT_ROOT / 'backend' / 'static' / 'results')
    }

# =============================================================================
# 🔥 로깅 설정
# =============================================================================

def setup_logging(level: str = "INFO") -> None:
    """로깅 설정 (중복 방지)"""
    # 이미 설정된 경우 스킵
    if logging.getLogger().handlers:
        return
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

# 기본 로깅 설정 (중복 방지)
setup_logging()
logger = logging.getLogger(__name__)

# =============================================================================
# 🔥 패키지 정보 출력
# =============================================================================

def print_system_info():
    """시스템 정보 출력"""
    print("\n" + "="*80)
    print("🍎 MyCloset AI 애플리케이션 패키지 v7.0")
    print("="*80)
    print(f"🔧 시스템: {_SYSTEM_INFO['platform']} / {_SYSTEM_INFO['device']}")
    print(f"🍎 M3 Max: {'✅' if _SYSTEM_INFO['is_m3_max'] else '❌'}")
    print(f"🐍 conda: {'✅' if _SYSTEM_INFO['is_conda'] else '❌'} ({_SYSTEM_INFO['conda_env']})")
    print(f"💾 메모리: {_SYSTEM_INFO['memory_gb']}GB")
    print(f"🧠 CPU: {_SYSTEM_INFO['cpu_count']}코어")
    print(f"🐍 Python: {_SYSTEM_INFO['python_version']}")
    print("="*80)
    print(f"📁 프로젝트 루트: {_SYSTEM_INFO['project_root']}")
    print(f"📁 백엔드 루트: {_SYSTEM_INFO['backend_root']}")
    print(f"📁 앱 루트: {_SYSTEM_INFO['app_root']}")
    print("="*80)
    print("✅ 패키지 초기화 완료!")
    print("="*80 + "\n")

# 시스템 정보 출력 (한 번만)
if not hasattr(sys, '_mycloset_app_initialized'):
    print_system_info()
    sys._mycloset_app_initialized = True

# =============================================================================
# 🔥 Export 목록
# =============================================================================

__all__ = [
    # 시스템 정보 함수들
    'get_system_info',
    'is_conda_environment', 
    'is_m3_max',
    'get_device',
    'get_project_paths',
    
    # 유틸리티 함수들
    'setup_logging',
    'print_system_info'
]

# 초기화 완료 로깅
logger.info("🍎 MyCloset AI 메인 패키지 초기화 완료")