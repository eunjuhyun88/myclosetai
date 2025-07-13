"""
MyCloset AI Backend - 안전한 실행 버전
PyTorch segfault 방지를 위한 안전한 시작점
"""

import sys
import os
import logging
from pathlib import Path

# 안전 환경변수 설정
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import platform

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="MyCloset AI Backend (Safe Mode)",
    description="안전 모드로 실행되는 AI 가상 피팅 시스템",
    version="1.0.0-safe"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PyTorch 안전 로드 함수
def safe_load_pytorch():
    """PyTorch를 안전하게 로드"""
    try:
        import torch
        logger.info(f"✅ PyTorch 로드 성공: {torch.__version__}")
        
        # 디바이스 확인 (안전모드)
        device = "cpu"  # 안정성을 위해 CPU 강제 사용
        
        return {
            "available": True,
            "version": torch.__version__,
            "device": device,
            "mode": "safe_cpu"
        }
    except Exception as e:
        logger.warning(f"⚠️ PyTorch 로드 실패: {e}")
        return {
            "available": False,
            "error": str(e),
            "device": "none",
            "mode": "no_pytorch"
        }

# 시작시 PyTorch 상태 확인
pytorch_status = safe_load_pytorch()

@app.get("/")
async def root():
    return {
        "message": "MyCloset AI Backend (Safe Mode) 🛡️",
        "version": "1.0.0-safe",
        "status": "running",
        "environment": {
            "conda_env": os.getenv("CONDA_DEFAULT_ENV", "unknown"),
            "python_version": platform.python_version(),
            "platform": platform.system(),
            "architecture": platform.machine()
        },
        "pytorch": pytorch_status,
        "docs": "/docs",
        "health": "/api/health"
    }

@app.get("/api/health")
async def health_check():
    """상세 헬스체크"""
    
    # 환경변수 확인
    env_vars = {
        "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS"),
        "PYTORCH_ENABLE_MPS_FALLBACK": os.getenv("PYTORCH_ENABLE_MPS_FALLBACK")
    }
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mode": "safe",
        "pytorch": pytorch_status,
        "environment_variables": env_vars,
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "conda_env": os.getenv("CONDA_DEFAULT_ENV")
        }
    }

@app.get("/api/test-pytorch")
async def test_pytorch():
    """PyTorch 테스트 엔드포인트"""
    
    if not pytorch_status["available"]:
        raise HTTPException(
            status_code=503, 
            detail="PyTorch를 사용할 수 없습니다"
        )
    
    try:
        import torch
        
        # 간단한 텐서 연산 테스트
        x = torch.tensor([1.0, 2.0, 3.0])
        y = x * 2
        
        return {
            "status": "success",
            "test_tensor": x.tolist(),
            "result": y.tolist(),
            "device": pytorch_status["device"],
            "message": "PyTorch가 정상적으로 작동합니다"
        }
        
    except Exception as e:
        logger.error(f"PyTorch 테스트 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"PyTorch 테스트 실패: {str(e)}"
        )

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 MyCloset AI Backend (Safe Mode) 시작됨")
    logger.info(f"🐍 Conda 환경: {os.getenv('CONDA_DEFAULT_ENV', 'unknown')}")
    logger.info(f"🔥 PyTorch: {'사용 가능' if pytorch_status['available'] else '사용 불가'}")

if __name__ == "__main__":
    import uvicorn
    print("🛡️ 안전 모드로 서버를 시작합니다...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
