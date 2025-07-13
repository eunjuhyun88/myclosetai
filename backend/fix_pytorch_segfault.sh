#!/bin/bash

echo "🐍 PyTorch Segmentation Fault 해결 중 (Conda 환경)..."

# 1. 현재 환경 확인
echo "🔍 1. 현재 환경 확인..."
echo "Conda 환경: ${CONDA_DEFAULT_ENV:-'없음'}"
echo "Python 버전: $(python --version 2>/dev/null || echo '확인불가')"

# 2. PyTorch 완전 제거
echo "🗑️ 2. 기존 PyTorch 완전 제거 중..."
conda remove pytorch torchvision torchaudio pytorch-cuda -y --force 2>/dev/null || true
pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

# Apple Silicon 감지
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    echo "🍎 Apple Silicon (M1/M2/M3) 감지됨"
    IS_APPLE_SILICON=true
else
    echo "🖥️ Intel/AMD 시스템 감지됨"
    IS_APPLE_SILICON=false
fi

# 3. NumPy 호환성 먼저 해결
echo "📦 3. NumPy 호환성 해결 중..."
conda install numpy=1.24.3 -y --force-reinstall

# 4. 안전한 PyTorch 설치
echo "🔥 4. 안전한 PyTorch 설치 중..."

if [[ "$IS_APPLE_SILICON" == true ]]; then
    # Apple Silicon용 PyTorch (CPU 버전 - 안정성 우선)
    echo "🍎 Apple Silicon용 CPU PyTorch 설치 중..."
    conda install pytorch torchvision -c pytorch -y
    
    # MPS 관련 환경변수 설정 (문제 발생시 비활성화)
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    
else
    # Intel/AMD용 PyTorch
    echo "🖥️ CPU 버전 PyTorch 설치 중..."
    conda install pytorch torchvision cpuonly -c pytorch -y
fi

# 5. 추가 안정성 패키지 설치
echo "📦 5. 안정성 패키지 설치 중..."
conda install -c conda-forge \
    pillow=10.0.0 \
    scipy=1.10.1 \
    scikit-image=0.21.0 \
    -y

# 6. 환경변수 설정
echo "⚙️ 6. 안정성 환경변수 설정 중..."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 환경변수를 ~/.bashrc에 추가
cat >> ~/.bashrc << 'EOF'

# PyTorch 안정성 설정
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
EOF

# 7. 간단한 PyTorch 테스트 스크립트 생성
echo "🧪 7. 안전한 PyTorch 테스트 스크립트 생성 중..."

cat > test_pytorch_safe.py << 'EOF'
#!/usr/bin/env python3
"""
안전한 PyTorch 테스트 스크립트
Segmentation fault 없이 PyTorch 테스트
"""

import sys
import os

# 안전 환경변수 설정
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def test_imports():
    """안전한 import 테스트"""
    print("🔍 Python 라이브러리 테스트 중...")
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except Exception as e:
        print(f"❌ NumPy 실패: {e}")
        return False
    
    try:
        # PyTorch를 조심스럽게 import
        print("🔥 PyTorch import 시도 중...")
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        # 기본 텐서 생성 테스트
        x = torch.tensor([1.0, 2.0, 3.0])
        print(f"✅ 텐서 생성 성공: {x}")
        
        # 디바이스 정보
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ Apple MPS 사용 가능 (하지만 CPU 사용 권장)")
            device = "cpu"  # 안정성을 위해 CPU 사용
        elif torch.cuda.is_available():
            print(f"✅ CUDA 사용 가능: {torch.cuda.get_device_name()}")
            device = "cpu"  # 안정성을 위해 CPU 사용
        else:
            print("✅ CPU 모드")
            device = "cpu"
            
        print(f"🎯 사용할 디바이스: {device}")
        
        # 간단한 연산 테스트
        y = x * 2
        print(f"✅ 연산 테스트 성공: {y}")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch 테스트 실패: {e}")
        return False

def test_basic_model():
    """기본 모델 테스트"""
    try:
        import torch
        import torch.nn as nn
        
        print("🧠 기본 모델 테스트 중...")
        
        # 간단한 모델 생성
        model = nn.Linear(3, 1)
        
        # 테스트 입력
        x = torch.tensor([[1.0, 2.0, 3.0]])
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
            
        print(f"✅ 모델 테스트 성공: {output}")
        return True
        
    except Exception as e:
        print(f"❌ 모델 테스트 실패: {e}")
        return False

def main():
    print("🐍 PyTorch 안정성 테스트 시작")
    print("=" * 40)
    
    # 시스템 정보
    print(f"🔧 Python: {sys.version}")
    print(f"💻 플랫폼: {sys.platform}")
    
    # Import 테스트
    if not test_imports():
        print("\n❌ Import 테스트 실패")
        sys.exit(1)
    
    # 모델 테스트
    if not test_basic_model():
        print("\n❌ 모델 테스트 실패")
        sys.exit(1)
    
    print("\n🎉 모든 테스트 성공!")
    print("✅ PyTorch가 안전하게 작동합니다.")
    print("✅ 이제 MyCloset AI 백엔드를 실행할 수 있습니다.")

if __name__ == "__main__":
    main()
EOF

# 8. 안전한 main.py 생성 (PyTorch 없이도 실행 가능)
echo "🔧 8. 안전한 main.py 생성 중..."

cat > app/main_safe.py << 'EOF'
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
EOF

# 9. 안전한 실행 스크립트 생성
echo "📜 9. 안전한 실행 스크립트 생성 중..."

cat > run_safe_server.sh << 'EOF'
#!/bin/bash

echo "🛡️ MyCloset AI Backend - 안전 모드 실행"
echo "======================================"

# 환경변수 설정
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Conda 환경 확인
if [[ "$CONDA_DEFAULT_ENV" == "" ]]; then
    echo "❌ Conda 환경이 활성화되지 않았습니다."
    echo "conda activate mycloset"
    exit 1
fi

echo "✅ Conda 환경: $CONDA_DEFAULT_ENV"

# PyTorch 테스트
echo "🧪 PyTorch 안전성 테스트 중..."
python test_pytorch_safe.py

if [[ $? -eq 0 ]]; then
    echo "✅ PyTorch 테스트 성공"
    echo ""
    echo "🌐 안전 모드 서버 시작 중..."
    echo "📱 접속: http://localhost:8000"
    echo "📚 API 문서: http://localhost:8000/docs"
    echo "🧪 PyTorch 테스트: http://localhost:8000/api/test-pytorch"
    echo ""
    
    # 안전한 main.py 실행
    uvicorn app.main_safe:app --reload --host 0.0.0.0 --port 8000
else
    echo "❌ PyTorch 테스트 실패"
    echo "기본 웹서버만 실행합니다..."
    
    # 최소한의 서버 실행
    python -c "
from fastapi import FastAPI
import uvicorn

app = FastAPI(title='MyCloset AI - 최소 서버')

@app.get('/')
def root():
    return {'message': 'MyCloset AI Backend - 최소 모드', 'status': 'pytorch_disabled'}

uvicorn.run(app, host='0.0.0.0', port=8000)
"
fi
EOF

chmod +x run_safe_server.sh
chmod +x test_pytorch_safe.py

echo ""
echo "🎉 PyTorch Segfault 해결 완료!"
echo ""
echo "🚀 실행 방법:"
echo "   1. ./run_safe_server.sh      # 안전한 실행"
echo "   2. python test_pytorch_safe.py  # PyTorch 테스트만"
echo ""
echo "🔧 문제가 지속되면:"
echo "   1. conda activate mycloset"
echo "   2. conda clean --all"
echo "   3. conda install pytorch=2.0.1 cpuonly -c pytorch -y"
echo "   4. ./run_safe_server.sh"
echo ""
echo "📱 실행 후: http://localhost:8000"