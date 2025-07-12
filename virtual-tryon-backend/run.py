# run.py
import uvicorn
from app.main import app
from app.core.config import settings

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 MyCloset AI Virtual Try-On 서버 시작")
    print("=" * 60)
    print(f"📱 API 문서: http://localhost:{settings.PORT}/docs")
    print(f"🏥 헬스체크: http://localhost:{settings.PORT}/api/health")
    print(f"🔗 가상 피팅: http://localhost:{settings.PORT}/api/virtual-tryon")
    print("=" * 60)
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )