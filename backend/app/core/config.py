# backend/app/core/config.py
"""
MyCloset AI Backend - 애플리케이션 설정
환경변수 기반 설정 관리
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import validator
from pydantic_settings import BaseSettings  # 수정된 부분

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent

class Settings(BaseSettings):
    """애플리케이션 설정 클래스"""
    
    # 기본 애플리케이션 설정
    APP_NAME: str = "MyCloset AI Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS 설정
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001"
    ]
    
    # 파일 업로드 설정
    MAX_UPLOAD_SIZE: int = 52428800  # 50MB
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "webp", "bmp"]
    
    # AI 모델 설정
    DEFAULT_MODEL: str = "ootd"
    USE_GPU: bool = True
    DEVICE: str = "mps"  # mps, cuda, cpu
    BATCH_SIZE: int = 1
    MAX_MEMORY_FRACTION: float = 0.8
    
    # 이미지 처리 설정
    IMAGE_SIZE: int = 512
    NUM_INFERENCE_STEPS: int = 20
    GUIDANCE_SCALE: float = 7.5
    
    # 경로 설정
    PROJECT_ROOT: Path = PROJECT_ROOT
    STATIC_DIR: Path = PROJECT_ROOT / "static"
    UPLOAD_DIR: Path = PROJECT_ROOT / "static" / "uploads"
    RESULTS_DIR: Path = PROJECT_ROOT / "static" / "results"
    AI_MODELS_DIR: Path = PROJECT_ROOT / "ai_models"
    TEMP_DIR: Path = PROJECT_ROOT / "ai_models" / "temp"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    
    # 성능 설정
    MAX_WORKERS: int = 4
    CACHE_TTL: int = 3600  # 1시간
    
    # 로깅 설정
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    
    # Redis 설정 (선택사항)
    REDIS_URL: Optional[str] = None
    
    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v) -> List[str]:
        """CORS origins 설정 파싱"""
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list):
            return v
        raise ValueError("CORS_ORIGINS must be a string or list")
    
    @validator("ALLOWED_EXTENSIONS", pre=True)
    def assemble_allowed_extensions(cls, v) -> List[str]:
        """허용된 파일 확장자 파싱"""
        if isinstance(v, str):
            return [i.strip().lower() for i in v.split(",")]
        elif isinstance(v, list):
            return [ext.lower() for ext in v]
        raise ValueError("ALLOWED_EXTENSIONS must be a string or list")
    
    @validator("DEVICE")
    def validate_device(cls, v) -> str:
        """디바이스 설정 검증"""
        allowed_devices = ["mps", "cuda", "cpu", "auto"]
        if v.lower() not in allowed_devices:
            raise ValueError(f"DEVICE must be one of {allowed_devices}")
        return v.lower()
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v) -> str:
        """로그 레벨 검증"""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"LOG_LEVEL must be one of {allowed_levels}")
        return v.upper()
    
    @validator("IMAGE_SIZE")
    def validate_image_size(cls, v) -> int:
        """이미지 크기 검증"""
        if v not in [256, 512, 768, 1024]:
            raise ValueError("IMAGE_SIZE must be one of [256, 512, 768, 1024]")
        return v
    
    @validator("MAX_UPLOAD_SIZE")
    def validate_upload_size(cls, v) -> int:
        """업로드 크기 검증"""
        if v > 100 * 1024 * 1024:  # 100MB 제한
            raise ValueError("MAX_UPLOAD_SIZE cannot exceed 100MB")
        return v
    
    def create_directories(self):
        """필요한 디렉토리 생성"""
        directories = [
            self.STATIC_DIR,
            self.UPLOAD_DIR,
            self.RESULTS_DIR,
            self.AI_MODELS_DIR,
            self.AI_MODELS_DIR / "checkpoints",
            self.TEMP_DIR,
            self.LOGS_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# 전역 설정 인스턴스
settings = Settings()

# 시작 시 디렉토리 생성
settings.create_directories()