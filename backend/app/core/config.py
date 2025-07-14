# app/core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """애플리케이션 설정"""
    
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
    
    class Config:
        env_file = ".env"

# 전역 설정 인스턴스
_settings = None

def get_settings() -> Settings:
    """설정 싱글톤 패턴"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings