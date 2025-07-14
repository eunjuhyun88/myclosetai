from pydantic_settings import BaseSettings
from typing import Optional
from pydantic import ConfigDict

class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # Pydantic V2 설정 - extra 필드 허용
    model_config = ConfigDict(
        extra="allow",  # 추가 필드 허용
        env_file=".env"
    )
    
    # 기본 설정
    app_name: str = "MyCloset AI"
    version: str = "3.0.0"
    debug: bool = False
    
    # M3 Max 최적화 설정
    device: str = "mps"
    batch_size: int = 4
    max_workers: int = 6
    memory_limit_gb: float = 102.4
    
    # API 설정
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # 파일 경로
    upload_dir: str = "static/uploads"
    results_dir: str = "static/results"
    models_dir: str = "models/ai_models"

# 전역 설정 인스턴스
settings = Settings()

def get_settings() -> Settings:
    """설정 반환"""
    return settings

def get_optimal_settings():
    """M3 Max 최적 설정 반환"""
    import torch
    
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    return {
        "device": device,
        "batch_size": 4 if device == "mps" else 2,
        "workers": 6 if device == "mps" else 4,
        "memory_fraction": 0.8,
        "fp16": device == "mps"
    }
