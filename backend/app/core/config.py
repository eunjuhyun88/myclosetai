# backend/app/core/config.py
"""
MyCloset AI 백엔드 설정 모듈 - LOGS 디렉토리 추가
get_settings 함수 포함한 완전한 설정 관리
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """애플리케이션 설정 클래스"""
    
    # 앱 기본 정보
    APP_NAME: str = "MyCloset AI Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS 설정
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ]
    
    # 파일 업로드 설정
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "webp", "bmp"]
    
    # AI 모델 설정
    USE_GPU: bool = True
    DEVICE: str = "auto"  # auto, cpu, cuda, mps
    MODEL_PRECISION: str = "fp16"  # fp32, fp16, int8
    BATCH_SIZE: int = 1
    MAX_WORKERS: int = 4
    
    # 경로 설정
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    STATIC_DIR: str = os.path.join(PROJECT_ROOT, "static")
    UPLOAD_DIR: str = os.path.join(PROJECT_ROOT, "static", "uploads")
    RESULTS_DIR: str = os.path.join(PROJECT_ROOT, "static", "results")
    MODELS_DIR: str = os.path.join(PROJECT_ROOT, "models", "ai_models")
    CACHE_DIR: str = os.path.join(PROJECT_ROOT, "cache")
    # LOGS 디렉토리 추가
    LOGS_DIR: Path = Path(PROJECT_ROOT) / "logs"
    
    # 로깅 설정 확장
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5
    LOG_TO_CONSOLE: bool = True
    LOG_TO_FILE: bool = True
    
    # 보안 설정
    SECRET_KEY: str = "mycloset-ai-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # API 설정
    API_V1_STR: str = "/api/v1"
    API_TIMEOUT: int = 300  # 5분
    
    # AI 파이프라인 설정
    PIPELINE_STEPS: List[str] = [
        "human_parsing",
        "pose_estimation", 
        "cloth_segmentation",
        "geometric_matching",
        "cloth_warping",
        "virtual_fitting",
        "post_processing",
        "quality_assessment"
    ]
    
    # 품질 설정
    DEFAULT_QUALITY: str = "high"  # low, medium, high
    QUALITY_THRESHOLDS: dict = {
        "low": 0.6,
        "medium": 0.7,
        "high": 0.8
    }
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"

# 싱글톤 설정 인스턴스
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """
    설정 인스턴스를 반환하는 함수
    싱글톤 패턴으로 구현하여 메모리 효율성 확보
    """
    global _settings
    if _settings is None:
        _settings = Settings()
        
        # 필요한 디렉토리 생성
        for directory in [
            _settings.STATIC_DIR,
            _settings.UPLOAD_DIR, 
            _settings.RESULTS_DIR,
            _settings.CACHE_DIR,
            _settings.LOGS_DIR  # LOGS 디렉토리 추가
        ]:
            os.makedirs(directory, exist_ok=True)
    
    return _settings

# 기존 코드 호환성을 위한 settings 객체
settings = get_settings()

# 자동 디렉토리 생성
def ensure_directories():
    """필요한 디렉토리가 존재하는지 확인하고 생성"""
    settings = get_settings()
    directories = [
        settings.STATIC_DIR,
        settings.UPLOAD_DIR,
        settings.RESULTS_DIR, 
        settings.CACHE_DIR,
        settings.MODELS_DIR,
        settings.LOGS_DIR  # LOGS 디렉토리 추가
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
    print(f"✅ 필요한 디렉토리 생성 완료: {len(directories)}개")

# 앱 시작 시 디렉토리 자동 생성
ensure_directories()

# 설정 검증
def validate_settings():
    """설정 값들이 유효한지 검증"""
    settings = get_settings()
    
    # 포트 범위 확인
    if not (1024 <= settings.PORT <= 65535):
        raise ValueError(f"PORT must be between 1024 and 65535, got {settings.PORT}")
    
    # 업로드 크기 확인
    if settings.MAX_UPLOAD_SIZE <= 0:
        raise ValueError("MAX_UPLOAD_SIZE must be positive")
    
    # 배치 크기 확인  
    if settings.BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be positive")
    
    # LOGS 디렉토리 권한 확인
    if not os.access(settings.LOGS_DIR, os.W_OK):
        print(f"⚠️ LOGS 디렉토리 쓰기 권한 없음: {settings.LOGS_DIR}")
        
    print("✅ 설정 검증 완료")

# 개발 환경에서 설정 정보 출력
if __name__ == "__main__":
    settings = get_settings()
    print("🔧 MyCloset AI 설정 정보")
    print("=" * 30)
    print(f"앱 이름: {settings.APP_NAME}")
    print(f"버전: {settings.APP_VERSION}")
    print(f"디버그 모드: {settings.DEBUG}")
    print(f"호스트: {settings.HOST}")
    print(f"포트: {settings.PORT}")
    print(f"프로젝트 루트: {settings.PROJECT_ROOT}")
    print(f"업로드 디렉토리: {settings.UPLOAD_DIR}")
    print(f"결과 디렉토리: {settings.RESULTS_DIR}")
    print(f"로그 디렉토리: {settings.LOGS_DIR}")  # LOGS 디렉토리 추가
    validate_settings()