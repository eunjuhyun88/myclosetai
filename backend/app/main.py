
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import os
from pathlib import Path

from app.core.config import settings
from app.api.routes import router as api_router

# FastAPI 앱 생성
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered virtual try-on platform",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙
static_path = Path("static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# 라우터 등록
app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    return {
        "message": "MyCloset AI Backend is running!",
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "cors_origins": settings.CORS_ORIGINS
    }

# 개발 서버 실행용 (python app/main.py로 실행할 때만)
if __name__ == "__main__":
    print("🚀 MyCloset AI 백엔드 서버 시작...")
    print(f"📍 서버 주소: http://{settings.HOST}:{settings.PORT}")
    print(f"📖 API 문서: http://{settings.HOST}:{settings.PORT}/docs")
    print(f"🔍 헬스체크: http://{settings.HOST}:{settings.PORT}/health")
    print("\n중지하려면 Ctrl+C를 누르세요.\n")
    
    uvicorn.run(
        "app.main:app",  # 문자열로 import 경로 지정
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
