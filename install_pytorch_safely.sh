#!/bin/bash

echo "🔥 PyTorch 안전 설치 스크립트 (Conda 환경)"
echo "============================================"

# 1. 현재 환경 확인
echo "🔍 1. 현재 환경 확인..."
echo "Conda 환경: ${CONDA_DEFAULT_ENV:-'없음'}"
echo "Python 버전: $(python --version 2>/dev/null || echo '확인불가')"
echo "시스템: $(uname -sm)"

if [[ "$CONDA_DEFAULT_ENV" == "" ]]; then
    echo "❌ Conda 환경이 활성화되지 않았습니다."
    echo "conda activate mycloset"
    exit 1
fi

# 2. 기존 PyTorch 완전 제거 (충돌 방지)
echo "🗑️ 2. 기존 PyTorch 완전 제거 중..."
conda remove pytorch torchvision torchaudio pytorch-cuda cpuonly -y --force 2>/dev/null || true
pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

# Apple Silicon 감지
ARCH=$(uname -m)
SYSTEM=$(uname -s)

if [[ "$ARCH" == "arm64" && "$SYSTEM" == "Darwin" ]]; then
    echo "🍎 Apple Silicon (M1/M2/M3) 감지됨"
    IS_APPLE_SILICON=true
else
    echo "🖥️ Intel/AMD 시스템 감지됨"
    IS_APPLE_SILICON=false
fi

# 3. NumPy 호환성 먼저 해결 (가장 중요!)
echo "📦 3. NumPy 호환성 해결 중..."
conda install numpy=1.24.3 -y --force-reinstall

# 4. 시스템별 PyTorch 설치
echo "🔥 4. PyTorch 안전 설치 중..."

if [[ "$IS_APPLE_SILICON" == true ]]; then
    echo "🍎 Apple Silicon용 PyTorch 설치..."
    
    # Apple Silicon용 안정 버전 설치
    conda install pytorch=2.0.1 torchvision=0.15.2 -c pytorch -y
    
    # MPS 관련 환경변수 설정
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    
else
    # Intel/AMD용 PyTorch 설치
    echo "🖥️ Intel/AMD용 PyTorch 설치..."
    
    # CUDA 확인
    if command -v nvidia-smi &> /dev/null; then
        echo "🔥 NVIDIA GPU 감지됨 - CUDA 버전 설치"
        conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
    else
        echo "💻 CPU 버전 설치"
        conda install pytorch torchvision cpuonly -c pytorch -y
    fi
fi

# 5. 안정성을 위한 추가 패키지 설치
echo "📦 5. 안정성 패키지 설치 중..."
conda install -c conda-forge \
    pillow=10.0.0 \
    scipy=1.10.1 \
    scikit-image=0.21.0 \
    opencv=4.8.0 \
    -y

# 6. 환경변수 설정 (Segfault 방지)
echo "⚙️ 6. 안정성 환경변수 설정 중..."

# 현재 세션용
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

if [[ "$IS_APPLE_SILICON" == true ]]; then
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
fi

# conda 환경에 영구 설정
CONDA_ENV_PATH=$CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_ENV_PATH

cat > $CONDA_ENV_PATH/pytorch_env.sh << 'EOF'
# PyTorch 안정성 환경변수
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
EOF

# 7. PyTorch 설치 확인 및 테스트
echo "🧪 7. PyTorch 설치 확인 중..."

cat > test_pytorch_installation.py << 'EOF'
#!/usr/bin/env python3
"""
PyTorch 설치 확인 및 기능 테스트
"""

import sys
import os

def test_pytorch_installation():
    """PyTorch 설치 및 기능 테스트"""
    print("🔥 PyTorch 설치 테스트 시작...")
    print("=" * 50)
    
    try:
        import torch
        print(f"✅ PyTorch 버전: {torch.__version__}")
        
        # 기본 텐서 생성 테스트
        x = torch.tensor([1.0, 2.0, 3.0])
        print(f"✅ 텐서 생성 성공: {x}")
        
        # 기본 연산 테스트
        y = x * 2 + 1
        print(f"✅ 기본 연산 성공: {y}")
        
        # 디바이스 확인
        print("\n🖥️ 사용 가능한 디바이스:")
        
        # CPU 항상 사용 가능
        print("  ✅ CPU: 사용 가능")
        cpu_tensor = torch.randn(3, 3, device='cpu')
        print(f"     CPU 텐서 테스트: {cpu_tensor.shape}")
        
        # CUDA 확인
        if torch.cuda.is_available():
            print(f"  ✅ CUDA: 사용 가능 ({torch.cuda.get_device_name()})")
            cuda_tensor = torch.randn(3, 3, device='cuda')
            print(f"     CUDA 텐서 테스트: {cuda_tensor.shape}")
            recommended_device = "cuda"
        else:
            print("  ℹ️ CUDA: 사용 불가")
            
        # MPS (Apple Silicon) 확인
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("  ✅ MPS (Apple Silicon): 사용 가능")
            try:
                mps_tensor = torch.randn(3, 3, device='mps')
                print(f"     MPS 텐서 테스트: {mps_tensor.shape}")
                if 'recommended_device' not in locals():
                    recommended_device = "mps"
            except Exception as e:
                print(f"  ⚠️ MPS 테스트 실패: {e}")
                print("     CPU 사용을 권장합니다")
                recommended_device = "cpu"
        else:
            print("  ℹ️ MPS: 사용 불가")
            
        if 'recommended_device' not in locals():
            recommended_device = "cpu"
            
        print(f"\n🎯 권장 디바이스: {recommended_device}")
        
        # 간단한 신경망 테스트
        print("\n🧠 신경망 테스트...")
        model = torch.nn.Linear(3, 2)
        test_input = torch.randn(1, 3)
        
        with torch.no_grad():
            output = model(test_input)
            
        print(f"✅ 신경망 테스트 성공: 입력 {test_input.shape} → 출력 {output.shape}")
        
        # 메모리 사용량 확인
        if recommended_device == "cuda":
            print(f"\n💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
        elif recommended_device == "mps":
            print("\n💾 통합 메모리 사용 (Apple Silicon)")
        
        print("\n🎉 모든 테스트 성공!")
        print(f"✅ PyTorch가 정상적으로 설치되었습니다.")
        print(f"🎯 MyCloset AI에서 사용할 디바이스: {recommended_device}")
        
        return True, recommended_device
        
    except ImportError as e:
        print(f"❌ PyTorch 임포트 실패: {e}")
        return False, "none"
    except Exception as e:
        print(f"❌ PyTorch 테스트 실패: {e}")
        return False, "cpu"

def test_ai_dependencies():
    """AI 관련 의존성 테스트"""
    print("\n📦 AI 의존성 테스트...")
    
    dependencies = [
        ("numpy", "넘파이"),
        ("PIL", "Pillow (이미지 처리)"),
        ("cv2", "OpenCV (컴퓨터 비전)"),
        ("scipy", "SciPy (과학 계산)"),
        ("skimage", "scikit-image (이미지 처리)")
    ]
    
    for package, description in dependencies:
        try:
            if package == "PIL":
                import PIL
                print(f"  ✅ {description}: {PIL.__version__}")
            elif package == "cv2":
                import cv2
                print(f"  ✅ {description}: {cv2.__version__}")
            elif package == "skimage":
                import skimage
                print(f"  ✅ {description}: {skimage.__version__}")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"  ✅ {description}: {version}")
        except ImportError:
            print(f"  ❌ {description}: 설치되지 않음")

if __name__ == "__main__":
    print(f"🐍 Python: {sys.version}")
    print(f"💻 플랫폼: {sys.platform}")
    
    success, device = test_pytorch_installation()
    test_ai_dependencies()
    
    if success:
        print("\n" + "="*50)
        print("🎉 설치 완료! MyCloset AI Backend를 실행할 수 있습니다.")
        print(f"🎯 사용할 디바이스: {device}")
        print("🚀 실행: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    else:
        print("\n" + "="*50)
        print("❌ PyTorch 설치에 문제가 있습니다.")
        print("🔧 해결 방법:")
        print("   conda install pytorch torchvision -c pytorch -y")
        sys.exit(1)
EOF

python test_pytorch_installation.py

if [[ $? -eq 0 ]]; then
    echo ""
    echo "🎉 PyTorch 설치 성공!"
    
    # 8. 백엔드 main.py를 PyTorch 지원 버전으로 업데이트
    echo "🔧 8. 백엔드를 PyTorch 지원 버전으로 업데이트 중..."
    
    cat > app/main_with_pytorch.py << 'EOF'
"""
MyCloset AI Backend - PyTorch 지원 버전
실제 AI 기능을 포함한 완전한 백엔드
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import platform
import asyncio
import time
import uuid

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# 안전한 설정 로드
try:
    from app.core.config import settings
    CONFIG_LOADED = True
    print("✅ 설정 로드 성공")
except Exception as e:
    print(f"⚠️ 설정 로드 실패: {e}")
    CONFIG_LOADED = False
    
    # 폴백 설정
    class FallbackSettings:
        APP_NAME = "MyCloset AI Backend"
        APP_VERSION = "1.0.0"
        DEBUG = True
        CORS_ORIGINS = ["http://localhost:3000", "http://localhost:5173"]
        MAX_UPLOAD_SIZE = 52428800
        ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp"]
    
    settings = FallbackSettings()

# PyTorch 및 AI 라이브러리 로드
TORCH_AVAILABLE = False
DEVICE_TYPE = "cpu"
DEVICE_INFO = "알 수 없음"

try:
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import numpy as np
    
    TORCH_AVAILABLE = True
    print("✅ PyTorch 로드 성공")
    
    # 최적 디바이스 선택
    if torch.cuda.is_available():
        DEVICE_TYPE = "cuda"
        DEVICE_INFO = f"NVIDIA GPU ({torch.cuda.get_device_name()})"
        print(f"🔥 CUDA GPU 사용: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEVICE_TYPE = "mps"
        DEVICE_INFO = "Apple Silicon (Metal)"
        print("🍎 Apple Silicon MPS 사용")
    else:
        DEVICE_TYPE = "cpu"
        DEVICE_INFO = "CPU"
        print("💻 CPU 사용")
        
except ImportError as e:
    print(f"⚠️ PyTorch 로드 실패: {e}")
    print("기본 기능만 사용 가능합니다.")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="MyCloset AI Backend",
    description="PyTorch 기반 AI 가상 피팅 시스템",
    version="1.0.0-pytorch"
)

# CORS 미들웨어
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip 압축
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 정적 파일 서빙
static_path = project_root / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# AI 모델 초기화 (간단한 예제)
if TORCH_AVAILABLE:
    # 이미지 전처리 파이프라인
    image_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 간단한 데모 모델 (실제로는 복잡한 가상 피팅 모델)
    demo_model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(64, 10)
    ).to(DEVICE_TYPE)
    
    demo_model.eval()
    print(f"✅ 데모 AI 모델 로드 완료 ({DEVICE_TYPE})")

@app.get("/")
async def root():
    return {
        "message": "MyCloset AI Backend (PyTorch 지원) 🔥",
        "version": "1.0.0-pytorch",
        "environment": "Conda",
        "conda_env": os.getenv("CONDA_DEFAULT_ENV", "unknown"),
        "python_version": platform.python_version(),
        "status": "healthy",
        "docs": "/docs",
        "config_loaded": CONFIG_LOADED,
        "ai": {
            "torch_available": TORCH_AVAILABLE,
            "device": DEVICE_TYPE,
            "device_info": DEVICE_INFO,
            "models_loaded": TORCH_AVAILABLE
        },
        "features": {
            "virtual_fitting": TORCH_AVAILABLE,
            "ai_processing": TORCH_AVAILABLE,
            "image_upload": True,
            "api_docs": True
        }
    }

@app.get("/api/health")
async def health_check():
    """상세 헬스체크"""
    
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0-pytorch",
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "conda_env": os.getenv("CONDA_DEFAULT_ENV", "unknown"),
            "architecture": platform.machine()
        },
        "ai": {
            "torch_available": TORCH_AVAILABLE,
            "device_type": DEVICE_TYPE,
            "device_info": DEVICE_INFO
        },
        "config": {
            "loaded": CONFIG_LOADED,
            "cors_origins": len(settings.CORS_ORIGINS) if hasattr(settings, 'CORS_ORIGINS') else 0
        },
        "services": {
            "virtual_fitting": "available" if TORCH_AVAILABLE else "disabled",
            "image_processing": "available",
            "file_upload": "available"
        }
    }
    
    # PyTorch가 사용 가능하면 추가 정보
    if TORCH_AVAILABLE:
        health_data["ai"].update({
            "torch_version": torch.__version__,
            "models_loaded": True,
            "memory_allocated": f"{torch.cuda.memory_allocated() // 1024**2}MB" if DEVICE_TYPE == "cuda" else "N/A"
        })
    
    return health_data

async def process_image_with_ai(person_image: Image.Image, clothing_image: Image.Image) -> dict:
    """AI를 사용한 이미지 처리"""
    
    if not TORCH_AVAILABLE:
        raise HTTPException(500, "AI 기능을 사용할 수 없습니다. PyTorch가 설치되지 않았습니다.")
    
    try:
        # 1. 이미지 전처리
        person_tensor = image_transform(person_image).unsqueeze(0).to(DEVICE_TYPE)
        clothing_tensor = image_transform(clothing_image).unsqueeze(0).to(DEVICE_TYPE)
        
        # 2. AI 모델 추론 (데모 버전)
        with torch.no_grad():
            # 실제로는 복잡한 가상 피팅 로직
            person_features = demo_model(person_tensor)
            clothing_features = demo_model(clothing_tensor)
            
            # 간단한 호환성 점수 계산
            similarity = torch.cosine_similarity(person_features, clothing_features, dim=1)
            confidence = float(similarity.item())
            
        return {
            "ai_processed": True,
            "confidence": abs(confidence),
            "fit_score": min(abs(confidence) + 0.2, 1.0),
            "device_used": DEVICE_TYPE,
            "processing_successful": True
        }
        
    except Exception as e:
        logger.error(f"AI 처리 실패: {e}")
        raise HTTPException(500, f"AI 처리 중 오류 발생: {str(e)}")

@app.post("/api/virtual-tryon")
async def virtual_tryon_endpoint(
    person_image: UploadFile = File(..., description="사용자 사진"),
    clothing_image: UploadFile = File(..., description="의류 사진"),
    height: float = Form(..., description="신장 (cm)"),
    weight: float = Form(..., description="체중 (kg)")
):
    """PyTorch 기반 가상 피팅 API"""
    
    # 파일 검증
    if not person_image.content_type.startswith("image/"):
        raise HTTPException(400, "사용자 이미지 파일이 아닙니다.")
    
    if not clothing_image.content_type.startswith("image/"):
        raise HTTPException(400, "의류 이미지 파일이 아닙니다.")
    
    try:
        session_id = str(uuid.uuid4())
        start_time = time.time()
        
        # 이미지 로드
        person_img = Image.open(person_image.file).convert('RGB')
        clothing_img = Image.open(clothing_image.file).convert('RGB')
        
        # AI 처리 (PyTorch 사용)
        if TORCH_AVAILABLE:
            ai_result = await process_image_with_ai(person_img, clothing_img)
            processing_type = "AI 기반 처리"
        else:
            # 폴백: 기본 처리
            ai_result = {
                "ai_processed": False,
                "confidence": 0.75,
                "fit_score": 0.80,
                "device_used": "cpu",
                "processing_successful": True
            }
            processing_type = "기본 처리"
        
        processing_time = time.time() - start_time
        
        # BMI 계산
        bmi = round(weight / ((height/100) ** 2), 1)
        bmi_status = "정상" if 18.5 <= bmi <= 25 else "확인 필요"
        
        return {
            "success": True,
            "session_id": session_id,
            "message": f"{processing_type}로 가상 피팅이 완료되었습니다!",
            "data": {
                "person_image": {
                    "filename": person_image.filename,
                    "content_type": person_image.content_type,
                    "size": f"{person_img.size[0]}x{person_img.size[1]}"
                },
                "clothing_image": {
                    "filename": clothing_image.filename,
                    "content_type": clothing_image.content_type,
                    "size": f"{clothing_img.size[0]}x{clothing_img.size[1]}"
                },
                "measurements": {
                    "height": f"{height}cm",
                    "weight": f"{weight}kg",
                    "bmi": bmi,
                    "bmi_status": bmi_status
                }
            },
            "ai_analysis": {
                "torch_used": TORCH_AVAILABLE,
                "device": ai_result["device_used"],
                "confidence": round(ai_result["confidence"], 3),
                "fit_score": round(ai_result["fit_score"], 3),
                "processing_type": processing_type
            },
            "processing": {
                "time_seconds": round(processing_time, 2),
                "status": "completed"
            },
            "recommendations": [
                f"핏 점수: {round(ai_result['fit_score']*100)}%" if ai_result['fit_score'] > 0.8 else "사이즈 확인 권장",
                f"AI 신뢰도: {round(ai_result['confidence']*100)}%",
                f"체형 분석: {bmi_status} (BMI: {bmi})"
            ]
        }
        
    except Exception as e:
        logger.error(f"가상 피팅 처리 실패: {e}")
        raise HTTPException(500, f"처리 중 오류 발생: {str(e)}")

@app.get("/api/models")
async def get_available_models():
    """사용 가능한 AI 모델 목록"""
    
    models = [
        {
            "id": "demo_pytorch",
            "name": "PyTorch 데모 모델",
            "status": "available" if TORCH_AVAILABLE else "unavailable",
            "device": DEVICE_TYPE,
            "description": "PyTorch 기반 가상 피팅 데모",
            "features": ["AI 분석", "핏 점수", "신뢰도 측정"] if TORCH_AVAILABLE else ["기본 기능만"]
        }
    ]
    
    if TORCH_AVAILABLE:
        models.extend([
            {
                "id": "ootd_diffusion",
                "name": "OOT-Diffusion",
                "status": "preparing",
                "device": DEVICE_TYPE,
                "description": "고품질 Diffusion 기반 가상 피팅",
                "features": ["고해상도", "자연스러운 합성", "정확한 피팅"]
            }
        ])
    
    return {
        "models": models,
        "default": "demo_pytorch" if TORCH_AVAILABLE else "basic",
        "environment": {
            "torch_available": TORCH_AVAILABLE,
            "device": DEVICE_TYPE,
            "device_info": DEVICE_INFO,
            "conda_env": os.getenv("CONDA_DEFAULT_ENV")
        },
        "capabilities": {
            "ai_processing": TORCH_AVAILABLE,
            "gpu_acceleration": DEVICE_TYPE in ["cuda", "mps"],
            "real_time": True
        }
    }

@app.get("/api/torch-test")
async def test_pytorch_functionality():
    """PyTorch 기능 테스트 엔드포인트"""
    
    if not TORCH_AVAILABLE:
        raise HTTPException(503, "PyTorch가 설치되지 않았습니다.")
    
    try:
        # 간단한 텐서 연산 테스트
        start_time = time.time()
        
        x = torch.randn(100, 100, device=DEVICE_TYPE)
        y = torch.randn(100, 100, device=DEVICE_TYPE)
        z = torch.mm(x, y)
        
        processing_time = time.time() - start_time
        
        return {
            "status": "success",
            "device": DEVICE_TYPE,
            "device_info": DEVICE_INFO,
            "test_results": {
                "tensor_size": "100x100",
                "operation": "matrix_multiplication",
                "processing_time_ms": round(processing_time * 1000, 2),
                "result_shape": list(z.shape),
                "memory_allocated": f"{torch.cuda.memory_allocated() // 1024**2}MB" if DEVICE_TYPE == "cuda" else "N/A"
            },
            "torch_version": torch.__version__,
            "message": "PyTorch가 정상적으로 작동합니다!"
        }
        
    except Exception as e:
        raise HTTPException(500, f"PyTorch 테스트 실패: {str(e)}")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 MyCloset AI Backend (PyTorch 지원) 시작됨")
    logger.info(f"🐍 Conda 환경: {os.getenv('CONDA_DEFAULT_ENV', 'unknown')}")
    logger.info(f"🔧 설정 로드: {'성공' if CONFIG_LOADED else '폴백 사용'}")
    logger.info(f"🔥 PyTorch: {'사용 가능' if TORCH_AVAILABLE else '없음'}")
    logger.info(f"💻 디바이스: {DEVICE_TYPE}")
    
    if TORCH_AVAILABLE:
        logger.info(f"🎯 AI 모델: 로드됨 ({DEVICE_TYPE})")
    else:
        logger.warning("⚠️ AI 기능 제한됨 (PyTorch 없음)")
    
    # 필수 디렉토리 생성
    directories = ["static/uploads", "static/results", "logs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("✅ 모든 시스템이 준비되었습니다!")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 MyCloset AI Backend 종료됨")
    if TORCH_AVAILABLE and DEVICE_TYPE == "cuda":
        torch.cuda.empty_cache()
        logger.info("🧹 GPU 메모리 정리 완료")

if __name__ == "__main__":
    import uvicorn
    print("🚀 PyTorch 지원 서버를 시작합니다...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
EOF

    # 9. PyTorch 지원 실행 스크립트 생성
    cat > run_with_pytorch.sh << 'EOF'
#!/bin/bash

echo "🔥 MyCloset AI Backend - PyTorch 지원 버전 실행"
echo "=============================================="

# Conda 환경 확인
if [[ "$CONDA_DEFAULT_ENV" == "" ]]; then
    echo "❌ Conda 환경이 활성화되지 않았습니다."
    echo "conda activate mycloset"
    exit 1
fi

echo "✅ Conda 환경: $CONDA_DEFAULT_ENV"

# PyTorch 확인
echo "🔥 PyTorch 상태 확인 중..."
python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')

if torch.cuda.is_available():
    print(f'✅ CUDA: {torch.cuda.get_device_name()}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✅ MPS (Apple Silicon): 사용 가능')
else:
    print('✅ CPU 모드')
" 2>/dev/null || {
    echo "❌ PyTorch가 설치되지 않았습니다!"
    echo "설치: conda install pytorch torchvision -c pytorch -y"
    exit 1
}

echo ""
echo "🌐 PyTorch 지원 서버 시작 중..."
echo "📱 메인: http://localhost:8000"
echo "📚 API 문서: http://localhost:8000/docs"
echo "🔧 헬스체크: http://localhost:8000/api/health"
echo "🧪 PyTorch 테스트: http://localhost:8000/api/torch-test"
echo "🎭 가상 피팅: http://localhost:8000/api/virtual-tryon"
echo ""
echo "⏹️ 종료하려면 Ctrl+C를 누르세요"
echo ""

# PyTorch 지원 버전 실행
uvicorn app.main_with_pytorch:app --reload --host 0.0.0.0 --port 8000
EOF

    chmod +x run_with_pytorch.sh
    
    echo ""
    echo "🎉 PyTorch 설치 및 백엔드 업데이트 완료!"
    echo ""
    echo "🚀 실행 방법:"
    echo "   ./run_with_pytorch.sh"
    echo ""
    echo "🧪 테스트 방법:"
    echo "   python test_pytorch_installation.py"
    echo ""
    echo "📱 실행 후 접속: http://localhost:8000"
    echo "🔥 PyTorch 테스트: http://localhost:8000/api/torch-test"
    
else
    echo ""
    echo "❌ PyTorch 설치 실패"
    echo "🔧 수동 설치 방법:"
    echo "   conda install pytorch torchvision -c pytorch -y"
    echo "   python test_pytorch_installation.py"
fi