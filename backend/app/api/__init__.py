"""
API 엔드포인트 모듈
FastAPI 라우터들을 관리합니다
"""

from fastapi import APIRouter

# 메인 API 라우터
api_router = APIRouter(prefix="/api", tags=["api"])

# 버전 정보
API_VERSION = "v1"
API_TITLE = "MyCloset AI API"
