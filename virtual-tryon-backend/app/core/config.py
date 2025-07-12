# core/config.py
from pathlib import Path
import torch

class Settings:
    # 프로젝트 설정
    APP_NAME = "Virtual Try-On API"
    VERSION = "1.0.0"
    
    # API 설정
    API_PREFIX = "/api/v1"
    
    # 파일 업로드 설정
    MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
    
    # 경로 설정
    BASE_DIR = Path(__file__).parent.parent
    UPLOAD_DIR = BASE_DIR / "static" / "uploads"
    RESULT_DIR = BASE_DIR / "static" / "results"
    MODEL_DIR = BASE_DIR / "models"
    
    # 모델 설정
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # CORS 설정
    CORS_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "*"  # 개발용, 프로덕션에서는 제거
    ]
    
    def __init__(self):
        # 디렉토리 생성
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.RESULT_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()