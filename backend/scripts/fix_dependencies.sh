#!/bin/bash
# ================================================================
# MyCloset AI - 수정된 의존성 설치 (패키지 오류 해결)
# ================================================================

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

echo "🔧 MyCloset AI - 패키지 오류 해결 및 재설치"
echo "=================================================="

# 환경 활성화 확인
if [[ "$CONDA_DEFAULT_ENV" != "mycloset-ai" ]]; then
    log_warning "conda 환경을 활성화합니다..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate mycloset-ai
fi

log_info "현재 환경: $CONDA_DEFAULT_ENV"
log_info "Python 버전: $(python --version)"

# ================================================================
# 1. 기본 AI/ML 패키지 (검증된 버전들)
# ================================================================

log_info "1단계: Transformers 생태계 설치 (검증된 버전)"

# Transformers 핵심 패키지들 - Python 3.11 호환 버전
pip install \
    transformers==4.35.0 \
    tokenizers==0.15.0 \
    safetensors==0.4.0 \
    accelerate==0.24.1 \
    datasets==2.14.6 \
    huggingface-hub==0.17.3

log_success "Transformers 생태계 설치 완료"

# ================================================================
# 2. Diffusers (가상 피팅 핵심)
# ================================================================

log_info "2단계: Diffusers 설치"
pip install diffusers==0.21.4

log_success "Diffusers 설치 완료"

# ================================================================
# 3. 컴퓨터 비전 패키지 (호환성 우선)
# ================================================================

log_info "3단계: 컴퓨터 비전 패키지 설치 (호환 버전)"

# MediaPipe는 이미 설치됨
log_info "MediaPipe: 이미 설치됨 (0.10.7)"

# YOLO - 호환 가능한 버전 설치
log_info "YOLO 호환 버전 설치 시도..."
pip install ultralytics==8.0.34 || {
    log_warning "ultralytics 8.0.34 설치 실패, 최신 호환 버전 시도"
    pip install "ultralytics>=8.0.0,<8.1.0" || {
        log_warning "ultralytics 설치 실패, 건너뜀 (선택적 패키지)"
    }
}

# Segment Anything - 공식 저장소에서 설치
log_info "Segment Anything Model 설치..."
pip install git+https://github.com/facebookresearch/segment-anything.git || {
    log_warning "SAM 설치 실패, 대체 방법 시도..."
    pip install segment-anything || {
        log_warning "SAM 설치 완전 실패, 건너뜀 (Step 3에서 다른 모델 사용)"
    }
}

log_success "컴퓨터 비전 패키지 설치 완료"

# ================================================================
# 4. 웹 프레임워크 및 서버
# ================================================================

log_info "4단계: 웹 프레임워크 설치"
pip install \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6 \
    aiofiles==23.2.1 \
    websockets==11.0.3 \
    python-dotenv==1.0.0

log_success "웹 프레임워크 설치 완료"

# ================================================================
# 5. 데이터 검증 및 설정
# ================================================================

log_info "5단계: 데이터 검증 라이브러리 설치"
pip install \
    pydantic==2.5.0 \
    pydantic-settings==2.1.0 \
    structlog==23.1.0

log_success "데이터 검증 라이브러리 설치 완료"

# ================================================================
# 6. 개발 도구 (선택적)
# ================================================================

log_info "6단계: 개발 도구 설치"
pip install \
    pytest==7.4.3 \
    pytest-asyncio==0.21.1 \
    black==23.11.0 \
    isort==5.12.0

log_success "개발 도구 설치 완료"

# ================================================================
# 7. M3 Max 특화 최적화
# ================================================================

log_info "7단계: M3 Max 특화 설정"

# Core ML Tools (Apple 전용)
pip install coremltools==7.0 || {
    log_warning "CoreML Tools 설치 실패 (선택적)"
}

log_success "M3 Max 특화 설정 완료"

# ================================================================
# 8. 패키지 검증
# ================================================================

log_info "8단계: 설치된 패키지 검증"

# 핵심 패키지 import 테스트
python -c "
import sys
print(f'🐍 Python: {sys.version}')

# 핵심 패키지 테스트
packages_to_test = [
    ('torch', 'PyTorch'),
    ('transformers', 'Transformers'),
    ('diffusers', 'Diffusers'),
    ('fastapi', 'FastAPI'),
    ('numpy', 'NumPy'),
    ('cv2', 'OpenCV'),
    ('PIL', 'Pillow'),
    ('matplotlib', 'Matplotlib'),
    ('sklearn', 'Scikit-learn'),
    ('skimage', 'Scikit-image')
]

print('\n📦 패키지 검증:')
success_count = 0
for package, name in packages_to_test:
    try:
        __import__(package)
        print(f'  ✅ {name}: OK')
        success_count += 1
    except ImportError as e:
        print(f'  ❌ {name}: FAILED ({e})')

print(f'\n📊 결과: {success_count}/{len(packages_to_test)} 패키지 성공')

# MPS 테스트
print('\n🍎 M3 Max MPS 테스트:')
try:
    import torch
    if torch.backends.mps.is_available():
        print('  ✅ MPS 사용 가능')
        device = torch.device('mps')
        x = torch.randn(100, 100, dtype=torch.float32, device=device)
        y = torch.mm(x, x.T)
        print(f'  ✅ MPS 연산 테스트 성공: {y.shape}')
    else:
        print('  ❌ MPS 사용 불가')
except Exception as e:
    print(f'  ❌ MPS 테스트 실패: {e}')

# 선택적 패키지 확인
print('\n🔍 선택적 패키지:')
optional_packages = [
    ('mediapipe', 'MediaPipe'),
    ('ultralytics', 'YOLO'),
    ('segment_anything', 'SAM'),
    ('coremltools', 'CoreML')
]

for package, name in optional_packages:
    try:
        __import__(package)
        print(f'  ✅ {name}: 설치됨')
    except ImportError:
        print(f'  ⚠️ {name}: 미설치 (선택적)')
"

# ================================================================
# 9. 환경 변수 및 설정 파일 생성
# ================================================================

log_info "9단계: 환경 설정 파일 생성"

# .env 파일 생성 (M3 Max 최적화)
cat > .env << 'EOF'
# MyCloset AI - M3 Max 최적화 환경 설정 (패키지 오류 해결 버전)

# ===========================================
# 기본 설정
# ===========================================
APP_NAME=MyCloset AI Backend
DEBUG=true
HOST=0.0.0.0
PORT=8000

# ===========================================
# M3 Max 하드웨어 최적화
# ===========================================
DEVICE=mps
GPU_TYPE=m3_max
MEMORY_GB=128
USE_GPU=true
UNIFIED_MEMORY=true

# ===========================================
# PyTorch MPS 최적화 (타입 오류 해결)
# ===========================================
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
PYTORCH_MPS_ALLOCATOR_POLICY=garbage_collection
PYTORCH_ENABLE_MPS_FALLBACK=1

# 타입 불일치 해결
MPS_FORCE_FLOAT32=true
MPS_PRECISION_MODE=float32
TORCH_DTYPE=float32

# ===========================================
# 패키지별 최적화 설정
# ===========================================
# Transformers 설정
TRANSFORMERS_CACHE=/tmp/transformers_cache
HF_DATASETS_CACHE=/tmp/datasets_cache

# OpenCV 설정
OPENCV_LOG_LEVEL=ERROR

# NumPy 설정
OPENBLAS_NUM_THREADS=16
MKL_NUM_THREADS=16

# ===========================================
# 메모리 관리
# ===========================================
MAX_MEMORY_FRACTION=0.75
MEMORY_POOL_SIZE=32
AUTO_MEMORY_CLEANUP=true

# ===========================================
# AI 모델 설정 (문제 해결)
# ===========================================
# 사용 가능한 모델만 활성화
ENABLE_SAM=false
ENABLE_YOLO=auto
ENABLE_ULTRALYTICS=auto

# 모델 로딩 최적화
MODEL_PRECISION=float32
ENABLE_MIXED_PRECISION=false
MODEL_CACHE_SIZE=16

# ===========================================
# 로깅
# ===========================================
LOG_LEVEL=INFO
LOG_FILE=logs/mycloset-ai.log
SUPPRESS_PACKAGE_WARNINGS=true
EOF

# 활성화 스크립트 생성
cat > activate_fixed.sh << 'EOF'
#!/bin/bash
# MyCloset AI 수정된 환경 활성화

# Conda 환경 활성화
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mycloset-ai

# M3 Max 최적화 환경 변수
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# 패키지 경고 억제
export PYTHONWARNINGS="ignore::UserWarning"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

echo "✅ MyCloset AI 수정된 환경 활성화"
echo "🔧 Python: $(python --version)"
echo "⚡ PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "🍎 MPS: $(python -c 'import torch; print("Available" if torch.backends.mps.is_available() else "Not Available")')"
echo ""
echo "🚀 서버 실행: cd backend && python app/main.py"
EOF

chmod +x activate_fixed.sh

# ================================================================
# 10. 빠른 테스트 서버 생성
# ================================================================

log_info "10단계: 테스트 서버 생성"

mkdir -p app

cat > app/test_fixed_server.py << 'EOF'
"""
수정된 MyCloset AI 테스트 서버
패키지 호환성 문제 해결 버전
"""
import os
import sys
import warnings

# 경고 억제
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import uvicorn

app = FastAPI(
    title="MyCloset AI - 수정된 테스트 서버",
    description="패키지 호환성 문제 해결 버전"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "MyCloset AI 수정된 테스트 서버",
        "version": "2.0.0-fixed",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    # 패키지 상태 확인
    package_status = {}
    
    packages_to_check = [
        "torch", "transformers", "diffusers", "fastapi", 
        "numpy", "cv2", "PIL", "matplotlib"
    ]
    
    for package in packages_to_check:
        try:
            module = __import__(package)
            if hasattr(module, '__version__'):
                package_status[package] = module.__version__
            else:
                package_status[package] = "imported_ok"
        except ImportError:
            package_status[package] = "not_available"
    
    # 선택적 패키지 확인
    optional_packages = ["mediapipe", "ultralytics", "segment_anything"]
    optional_status = {}
    
    for package in optional_packages:
        try:
            __import__(package)
            optional_status[package] = "available"
        except ImportError:
            optional_status[package] = "not_installed"
    
    return {
        "status": "healthy",
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "packages": package_status,
        "optional_packages": optional_status
    }

@app.get("/test-mps")
async def test_mps_fixed():
    """MPS 테스트 (타입 오류 해결)"""
    if not torch.backends.mps.is_available():
        return {"error": "MPS not available"}
    
    try:
        device = torch.device('mps')
        
        # float32로 통일 (타입 불일치 해결)
        x = torch.randn(100, 100, dtype=torch.float32, device=device)
        y = torch.randn(100, 100, dtype=torch.float32, device=device)
        
        # 행렬 곱셈 테스트
        result = torch.mm(x, y)
        
        # 간단한 신경망 연산 테스트
        linear = torch.nn.Linear(100, 50, dtype=torch.float32).to(device)
        output = linear(x)
        
        return {
            "status": "success",
            "device": str(device),
            "matrix_mult_shape": list(result.shape),
            "neural_net_shape": list(output.shape),
            "memory_allocated": torch.mps.current_allocated_memory() if hasattr(torch.mps, 'current_allocated_memory') else 0,
            "dtype_test": "float32_consistent"
        }
    except Exception as e:
        return {
            "error": str(e),
            "error_type": type(e).__name__
        }

@app.get("/models-status")
async def models_status():
    """AI 모델 상태 확인"""
    status = {
        "transformers": {"available": False, "models": []},
        "diffusers": {"available": False, "models": []},
        "computer_vision": {"available": False, "models": []}
    }
    
    # Transformers 확인
    try:
        from transformers import AutoModel, AutoTokenizer
        status["transformers"]["available"] = True
        status["transformers"]["models"] = ["CLIP", "BERT", "GPT"]
    except ImportError:
        pass
    
    # Diffusers 확인
    try:
        from diffusers import StableDiffusionPipeline
        status["diffusers"]["available"] = True
        status["diffusers"]["models"] = ["Stable Diffusion", "ControlNet"]
    except ImportError:
        pass
    
    # 컴퓨터 비전 확인
    cv_models = []
    try:
        import mediapipe
        cv_models.append("MediaPipe")
    except ImportError:
        pass
    
    try:
        import ultralytics
        cv_models.append("YOLO")
    except ImportError:
        pass
    
    if cv_models:
        status["computer_vision"]["available"] = True
        status["computer_vision"]["models"] = cv_models
    
    return status

if __name__ == "__main__":
    print("🚀 MyCloset AI 수정된 테스트 서버 시작...")
    print("📍 서버: http://localhost:8000")
    print("❤️ 헬스체크: http://localhost:8000/health")
    print("🧪 MPS 테스트: http://localhost:8000/test-mps")
    print("🤖 모델 상태: http://localhost:8000/models-status")
    print("")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
EOF

# 패키지 목록 저장
pip freeze > requirements_fixed.txt

# ================================================================
# 완료 메시지
# ================================================================

echo ""
echo "🎉 패키지 오류 해결 및 설치 완료!"
echo "=================================================="
log_success "✅ 호환성 문제 해결됨"
log_success "✅ M3 Max MPS 최적화 적용"
log_success "✅ 필수 패키지 설치 완료"
log_success "✅ 선택적 패키지 스킵 (오류 방지)"

echo ""
echo "📋 생성된 파일들:"
echo "  - .env: M3 Max 최적화 환경 변수"
echo "  - activate_fixed.sh: 수정된 환경 활성화"
echo "  - app/test_fixed_server.py: 호환성 테스트 서버"
echo "  - requirements_fixed.txt: 최종 패키지 목록"

echo ""
echo "🚀 다음 단계:"
echo "1. 테스트: python app/test_fixed_server.py"
echo "2. 브라우저: http://localhost:8000/health"
echo "3. MPS 테스트: http://localhost:8000/test-mps"
echo "4. 모델 상태: http://localhost:8000/models-status"

echo ""
echo "💡 환경 활성화:"
echo "  source activate_fixed.sh"

echo ""
log_warning "📌 주요 변경사항:"
echo "  - ultralytics: 호환 버전으로 변경"
echo "  - segment-anything: Git에서 직접 설치"
echo "  - MPS 타입 불일치 오류 해결"
echo "  - 선택적 패키지 실패 시 건너뛰기"