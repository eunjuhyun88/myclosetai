# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from api.routes import router
from core.config import settings
from core.middleware import TimingMiddleware

# FastAPI 앱 생성
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 커스텀 미들웨어
app.add_middleware(TimingMiddleware)

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

# 라우터 등록
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행"""
    print(f"🚀 {settings.app_name} v{settings.version} starting...")
    print(f"📍 API docs: http://localhost:8000/docs")
    print(f"🔧 Device: {settings.device}")

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 실행"""
    print("👋 Shutting down...")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        workers=1 if settings.debug else 4
    )