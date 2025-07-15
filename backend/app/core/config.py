# backend/app/core/config.py
"""
MyCloset AI 설정 시스템 (단순하고 실용적인 버전)
- get_settings() 함수 완전 지원 ✅
- M3 Max 128GB 최적화 ✅
- 복잡한 상속 구조 제거, 단순하고 명확함 ✅
"""

import os
import platform
import subprocess
import psutil
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from functools import lru_cache

# 로깅 설정
logger = logging.getLogger(__name__)

# ===============================================================
# 🔧 시스템 감지 함수들 (간단하게)
# ===============================================================

def detect_m3_max_info() -> Tuple[str, float, bool, str]:
    """M3 Max 환경 정보 감지"""
    try:
        memory_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        is_m3_max = False
        device = "cpu"
        chip_name = "Unknown"
        
        # macOS 감지
        if platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True, text=True, timeout=5
                )
                chip_name = result.stdout.strip()
                
                # M3 Max 감지
                if 'M3' in chip_name and ('Max' in chip_name or memory_gb >= 64):
                    is_m3_max = True
                    device = "mps"
                elif 'M3' in chip_name or 'M2' in chip_name or 'M1' in chip_name:
                    device = "mps"
                    
            except:
                # 메모리 기반 추정
                if memory_gb >= 64:
                    is_m3_max = True
                    device = "mps"
        
        # GPU 감지
        if device == "cpu":
            try:
                import torch
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
            except ImportError:
                pass
        
        # 최적화 레벨 결정
        if is_m3_max and memory_gb >= 128:
            quality = "ultra"
        elif memory_gb >= 64:
            quality = "high"
        elif memory_gb >= 32:
            quality = "balanced"
        else:
            quality = "fast"
            
        return device, memory_gb, is_m3_max, quality
        
    except Exception as e:
        logger.warning(f"시스템 감지 실패: {e}")
        return "cpu", 8.0, False, "fast"

def detect_environment() -> str:
    """환경 감지"""
    env = os.getenv('MYCLOSET_ENV', os.getenv('APP_ENV', '')).lower()
    
    if env in ['prod', 'production']:
        return 'production'
    elif env in ['test', 'testing']:
        return 'testing'
    elif env in ['dev', 'development']:
        return 'development'
    elif os.getenv('DEBUG', '').lower() in ['true', '1']:
        return 'development'
    else:
        return 'development'

# ===============================================================
# 🎯 Settings 클래스 (간단하고 명확함)
# ===============================================================

class Settings:
    """MyCloset AI 설정 클래스 - 간단하고 실용적"""
    
    def __init__(self):
        # 환경 감지
        self.env = detect_environment()
        
        # 시스템 정보 감지
        self.device, self.memory_gb, self.is_m3_max, self.quality_level = detect_m3_max_info()
        
        # 기본 앱 설정
        self._setup_app_config()
        
        # AI 설정
        self._setup_ai_config()
        
        # 경로 설정
        self._setup_paths()
        
        # 디렉토리 생성
        self._ensure_directories()
        
        logger.info(f"🎯 Settings 초기화 완료 - 환경: {self.env}, 디바이스: {self.device}, M3 Max: {self.is_m3_max}")
    
    def _setup_app_config(self):
        """앱 설정"""
        self.APP_NAME = "MyCloset AI Backend"
        self.APP_VERSION = "3.0.0"
        self.DEBUG = self.env == 'development'
        self.HOST = "0.0.0.0"
        self.PORT = 8000
        self.LOG_LEVEL = "DEBUG" if self.DEBUG else "INFO"
        
        # CORS 설정
        self.CORS_ORIGINS = [
            "http://localhost:3000",
            "http://localhost:3001", 
            "http://localhost:5173",
            "http://127.0.0.1:3000"
        ]
        
        # 데이터베이스
        self.DATABASE_URL = "sqlite:///./mycloset_ai.db"
        
        # 파일 처리
        self.MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
        self.ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    
    def _setup_ai_config(self):
        """AI 설정"""
        # 디바이스 설정
        self.DEVICE = self.device
        self.USE_GPU = self.device != "cpu"
        self.IS_M3_MAX = self.is_m3_max
        self.ENABLE_MPS = self.device == "mps"
        self.ENABLE_CUDA = self.device == "cuda"
        
        # 성능 설정
        self.QUALITY_LEVEL = self.quality_level
        self.IMAGE_SIZE = 512
        self.MAX_WORKERS = min(4, psutil.cpu_count() or 4)
        
        # 배치 크기 (M3 Max 최적화)
        if self.is_m3_max and self.memory_gb >= 128:
            self.BATCH_SIZE = 8
        elif self.is_m3_max and self.memory_gb >= 64:
            self.BATCH_SIZE = 4
        elif self.memory_gb >= 32:
            self.BATCH_SIZE = 2
        else:
            self.BATCH_SIZE = 1
    
    def _setup_paths(self):
        """경로 설정"""
        self.PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
        self.STATIC_DIR = os.path.join(self.PROJECT_ROOT, "static")
        self.UPLOAD_DIR = os.path.join(self.PROJECT_ROOT, "static", "uploads")
        self.RESULTS_DIR = os.path.join(self.PROJECT_ROOT, "static", "results")
        self.CACHE_DIR = os.path.join(self.PROJECT_ROOT, "cache")
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, "ai_models")
        self.LOGS_DIR = os.path.join(self.PROJECT_ROOT, "logs")
    
    def _ensure_directories(self):
        """필요한 디렉토리 생성"""
        directories = [
            self.STATIC_DIR, self.UPLOAD_DIR, self.RESULTS_DIR,
            self.CACHE_DIR, self.MODELS_DIR, self.LOGS_DIR
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

# ===============================================================
# 🎯 호환성 함수들
# ===============================================================

# 전역 설정 인스턴스
_global_settings: Optional[Settings] = None

@lru_cache()
def get_settings() -> Settings:
    """기존 코드와 완전 호환되는 설정 반환 함수"""
    global _global_settings
    
    if _global_settings is None:
        _global_settings = Settings()
    
    return _global_settings

# 기본 설정 인스턴스 (하위 호환성)
settings = get_settings()

# 하위 호환성을 위한 전역 변수들 (기존 코드가 이걸 사용함)
APP_NAME = settings.APP_NAME
APP_VERSION = settings.APP_VERSION
DEBUG = settings.DEBUG
HOST = settings.HOST
PORT = settings.PORT
LOG_LEVEL = settings.LOG_LEVEL
DATABASE_URL = settings.DATABASE_URL
CORS_ORIGINS = settings.CORS_ORIGINS

DEVICE = settings.DEVICE
USE_GPU = settings.USE_GPU
IS_M3_MAX = settings.IS_M3_MAX
ENABLE_MPS = settings.ENABLE_MPS
ENABLE_CUDA = settings.ENABLE_CUDA
QUALITY_LEVEL = settings.QUALITY_LEVEL
BATCH_SIZE = settings.BATCH_SIZE

# M3 Max 최적화 상태 로깅
if IS_M3_MAX:
    logger.info("🍎 M3 Max 128GB 최적화 활성화")
    logger.info(f"   - Neural Engine: {settings.is_m3_max}")
    logger.info(f"   - Metal Performance Shaders: {ENABLE_MPS}")
    logger.info(f"   - 통합 메모리: {settings.memory_gb}GB")

if USE_GPU:
    logger.info(f"🎮 GPU 가속 활성화: {DEVICE}")

logger.info(f"🎯 MyCloset AI 설정 시스템 로드 완료 - 환경: {settings.env}, 디바이스: {DEVICE}")

# __all__ export (기존 코드가 필요로 하는 것들만)
__all__ = [
    'get_settings', 'settings', 'Settings',
    'APP_NAME', 'APP_VERSION', 'DEBUG', 'HOST', 'PORT', 'LOG_LEVEL', 
    'DATABASE_URL', 'CORS_ORIGINS', 'DEVICE', 'USE_GPU', 'IS_M3_MAX', 
    'ENABLE_MPS', 'ENABLE_CUDA', 'QUALITY_LEVEL', 'BATCH_SIZE'
]