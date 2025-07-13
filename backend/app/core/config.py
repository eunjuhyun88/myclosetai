from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # 앱 설정
    APP_NAME: str = "MyCloset AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS 설정
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]
    
    # 파일 업로드 설정
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "webp"]
    
    # 경로 설정
    UPLOAD_DIR: str = "static/uploads"
    RESULT_DIR: str = "static/results"
    MODEL_DIR: str = "models/checkpoints"
    
    # AI 설정
    DEVICE: str = "mps"  # M3 Max
    USE_FP16: bool = True
    MEMORY_LIMIT_GB: float = 16.0
    
    # 로깅 설정
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    class Config:
        env_file = ".env"

settings = Settings()