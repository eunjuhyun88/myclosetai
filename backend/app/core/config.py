from typing import List
import os

class Settings:
    """간단하고 안정적인 설정 클래스"""
    
    # App 기본 설정
    APP_NAME: str = "MyCloset AI Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS 설정 (안전한 기본값)
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173", 
        "http://localhost:8080"
    ]
    
    # 파일 업로드 설정
    MAX_UPLOAD_SIZE: int = 52428800  # 50MB
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "webp", "bmp"]
    
    # AI 모델 설정
    DEFAULT_MODEL: str = "demo"
    USE_GPU: bool = False  # 안정성을 위해 False
    DEVICE: str = "cpu"
    IMAGE_SIZE: int = 512
    MAX_WORKERS: int = 2
    BATCH_SIZE: int = 1
    
    # 로깅
    LOG_LEVEL: str = "INFO"
    
    # 경로
    UPLOAD_PATH: str = "static/uploads"
    RESULT_PATH: str = "static/results"
    MODEL_PATH: str = "ai_models"
    
    def __init__(self):
        # 환경변수에서 값 읽기 (있으면)
        self.DEBUG = os.getenv("DEBUG", "true").lower() == "true"
        self.HOST = os.getenv("HOST", "0.0.0.0")
        self.PORT = int(os.getenv("PORT", 8000))
        
        # CORS_ORIGINS 환경변수 처리
        cors_env = os.getenv("CORS_ORIGINS")
        if cors_env:
            # 쉼표로 구분된 문자열을 리스트로 변환
            self.CORS_ORIGINS = [origin.strip() for origin in cors_env.split(",")]

# 전역 설정 객체
settings = Settings()
