#!/bin/bash
# ==============================================================================
# MyCloset AI - 기존 프로젝트용 M3 Max 최적화 Conda 환경 설정
# 이미 만들어진 디렉토리 구조를 활용하여 환경만 설정
# ==============================================================================

set -e  # 에러 발생시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# 로그 함수들
log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }
log_header() { echo -e "${PURPLE}🚀 $1${NC}"; }

# 프로젝트 정보
PROJECT_NAME="mycloset-ai"
CONDA_ENV_NAME="mycloset-m3"
PYTHON_VERSION="3.11"

log_header "MyCloset AI - 기존 프로젝트 M3 Max 최적화 설정"
echo "=================================================================="
log_info "프로젝트: $(pwd)"
log_info "Python: $PYTHON_VERSION"
log_info "Conda 환경명: $CONDA_ENV_NAME"
echo ""

# 1. 시스템 체크
log_header "Step 1: 시스템 요구사항 체크"

# macOS 및 Apple Silicon 체크
if [[ "$(uname -s)" != "Darwin" ]]; then
    log_error "이 스크립트는 macOS 전용입니다."
    exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
    log_error "Apple Silicon (M1/M2/M3) Mac이 필요합니다."
    exit 1
fi

log_success "Apple Silicon Mac 확인됨"

# Conda 설치 체크
if ! command -v conda &> /dev/null; then
    log_error "Conda가 설치되어 있지 않습니다."
    log_info "Miniconda 설치: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

log_success "Conda 설치 확인됨: $(conda --version)"

# 기존 프로젝트 구조 확인
log_info "기존 프로젝트 구조 확인 중..."

required_dirs=("backend" "backend/app" "backend/app/core")
missing_dirs=()

for dir in "${required_dirs[@]}"; do
    if [[ -d "$dir" ]]; then
        log_success "디렉토리 확인: $dir"
    else
        missing_dirs+=("$dir")
        log_warning "디렉토리 누락: $dir"
    fi
done

# 누락된 핵심 디렉토리만 생성
if [[ ${#missing_dirs[@]} -gt 0 ]]; then
    log_info "누락된 핵심 디렉토리 생성 중..."
    for dir in "${missing_dirs[@]}"; do
        mkdir -p "$dir"
        log_info "생성: $dir"
    done
fi

# 2. Conda 환경 생성
log_header "Step 2: Conda 환경 생성 및 설정"

# Conda 초기화
source "$(conda info --base)/etc/profile.d/conda.sh"

# 기존 환경 제거 (있다면)
if conda env list | grep -q "$CONDA_ENV_NAME"; then
    log_warning "기존 환경 '$CONDA_ENV_NAME' 제거 중..."
    conda env remove -n "$CONDA_ENV_NAME" -y
fi

# 새 환경 생성
log_info "Conda 환경 '$CONDA_ENV_NAME' 생성 중..."
conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y

# 환경 활성화
log_info "환경 활성화 중..."
conda activate "$CONDA_ENV_NAME"

log_success "Conda 환경 '$CONDA_ENV_NAME' 생성 완료"

# 3. M3 Max 최적화 패키지 설치
log_header "Step 3: M3 Max 최적화 패키지 설치"

# conda-forge 채널 추가
conda config --add channels conda-forge
conda config --add channels pytorch
conda config --add channels apple

# PyTorch M3 Max 최적화 버전 설치
log_info "PyTorch (M3 Max Metal 최적화) 설치 중..."
conda install pytorch torchvision torchaudio -c pytorch -y

# 기본 패키지 설치
log_info "기본 패키지 설치 중..."
conda install -y \
    numpy=1.24.3 \
    scipy=1.11.4 \
    pillow=10.1.0 \
    opencv=4.8.1 \
    scikit-image=0.22.0 \
    scikit-learn=1.3.2 \
    pydantic=2.5.0 \
    requests=2.31.0 \
    tqdm=4.66.1

# pip로 추가 패키지 설치
log_info "FastAPI 및 웹 서버 패키지 설치 중..."
pip install \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6 \
    aiofiles==23.2.1 \
    python-dotenv==1.0.0 \
    pydantic-settings==2.1.0

log_info "AI/ML 패키지 설치 중..."
pip install \
    transformers==4.35.0 \
    diffusers==0.21.4 \
    accelerate==0.24.1 \
    mediapipe==0.10.7

log_info "개발 도구 설치 중..."
pip install \
    structlog==23.1.0 \
    pytest==7.4.3 \
    pytest-asyncio==0.21.1 \
    black==23.11.0 \
    isort==5.12.0

log_success "패키지 설치 완료"

# 4. 필요한 디렉토리 보완 (있는 것은 건드리지 않음)
log_header "Step 4: 프로젝트 디렉토리 보완"

# 필요한 하위 디렉토리들 생성 (없는 것만)
additional_dirs=(
    "backend/ai_models/checkpoints"
    "backend/ai_models/temp"
    "backend/static/uploads"
    "backend/static/results"
    "backend/logs"
    "backend/scripts"
    "backend/app/api"
    "backend/app/services"
    "backend/app/utils"
    "backend/app/models"
    "backend/tests"
)

for dir in "${additional_dirs[@]}"; do
    if [[ ! -d "$dir" ]]; then
        mkdir -p "$dir"
        log_info "생성: $dir"
    fi
done

# .gitkeep 파일 생성 (빈 디렉토리 유지)
gitkeep_dirs=(
    "backend/static/uploads"
    "backend/static/results"
    "backend/ai_models/checkpoints"
    "backend/ai_models/temp"
    "backend/logs"
)

for dir in "${gitkeep_dirs[@]}"; do
    if [[ -d "$dir" && ! -f "$dir/.gitkeep" ]]; then
        touch "$dir/.gitkeep"
        log_info ".gitkeep 생성: $dir"
    fi
done

log_success "프로젝트 디렉토리 보완 완료"

# 5. M3 Max GPU 테스트
log_header "Step 5: M3 Max GPU (Metal) 테스트"

python3 -c "
import torch
import platform
import sys

print(f'Python 버전: {sys.version}')
print(f'PyTorch 버전: {torch.__version__}')
print(f'플랫폼: {platform.platform()}')
print(f'아키텍처: {platform.machine()}')
print()

# MPS 체크
if torch.backends.mps.is_available():
    print('✅ MPS (Metal Performance Shaders) 사용 가능')
    device = torch.device('mps')
    
    # 간단한 연산 테스트
    print('🧪 GPU 연산 테스트 중...')
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    
    import time
    start = time.time()
    z = torch.mm(x, y)
    end = time.time()
    
    print(f'✅ M3 Max GPU 연산 성공: {end-start:.4f}초')
    print(f'✅ 텐서 크기: {z.shape}')
    print(f'✅ 디바이스: {z.device}')
    
else:
    print('⚠️ MPS 사용 불가 - CPU 모드로 실행됩니다')
    print('   macOS 12.3+ 및 PyTorch 1.12+ 필요')
"

# 6. 환경 설정 파일 생성/업데이트
log_header "Step 6: 환경 설정 파일 생성"

# backend/.env 파일 생성 (없는 경우만)
if [[ ! -f "backend/.env" ]]; then
    log_info "backend/.env 파일 생성 중..."
    cat > backend/.env << 'EOF'
# MyCloset AI Backend - M3 Max 최적화 설정
APP_NAME=MyCloset AI Backend
APP_VERSION=1.0.0
DEBUG=true
HOST=0.0.0.0
PORT=8000

# CORS Settings
CORS_ORIGINS=http://localhost:3000,http://localhost:3001

# M3 Max GPU 설정
USE_GPU=true
DEVICE=mps
BATCH_SIZE=1
MAX_MEMORY_FRACTION=0.8

# File Upload Settings
MAX_UPLOAD_SIZE=52428800
ALLOWED_EXTENSIONS=jpg,jpeg,png,webp

# AI Model Settings
DEFAULT_MODEL=ootd
IMAGE_SIZE=512
NUM_INFERENCE_STEPS=20
GUIDANCE_SCALE=7.5

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
EOF
    log_success "backend/.env 파일 생성 완료"
else
    log_info "backend/.env 파일이 이미 존재합니다."
fi

# backend/requirements.txt 업데이트 (백업 후 생성)
if [[ -f "backend/requirements.txt" ]]; then
    cp backend/requirements.txt backend/requirements.txt.backup
    log_info "기존 requirements.txt 백업 생성"
fi

log_info "backend/requirements.txt 업데이트 중..."
cat > backend/requirements.txt << 'EOF'
# MyCloset AI - M3 Max 최적화 패키지 목록
# Conda 환경에서 설치된 패키지들의 참조용

# Core ML/AI
torch>=2.1.0
torchvision>=0.16.0
numpy==1.24.3
scipy==1.11.4

# FastAPI & Web
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6

# Image Processing
Pillow==10.1.0
opencv-python==4.8.1.78
scikit-image==0.22.0

# AI/ML
transformers==4.35.0
diffusers==0.21.4
accelerate==0.24.1
mediapipe==0.10.7

# Utils
pydantic==2.5.0
pydantic-settings==2.1.0
aiofiles==23.2.1
python-dotenv==1.0.0
structlog==23.1.0
requests==2.31.0
tqdm==4.66.1

# Development
pytest==7.4.3
pytest-asyncio==0.21.1
black==23.11.0
isort==5.12.0
EOF

log_success "requirements.txt 업데이트 완료"

# 7. 활성화 스크립트 생성
log_header "Step 7: 편의 스크립트 생성"

# Conda 환경 활성화 스크립트
cat > activate_env.sh << 'EOF'
#!/bin/bash
# MyCloset AI Conda 환경 활성화 스크립트

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mycloset-m3

echo "✅ MyCloset AI M3 Max 환경 활성화됨"
echo "🔧 Python: $(python --version)"
echo "⚡ PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "🖥️ GPU: $(python -c 'import torch; print("MPS" if torch.backends.mps.is_available() else "CPU")')"
echo ""
echo "사용법:"
echo "  개발 서버 실행: cd backend && python app/main.py"
echo "  AI 모델 다운로드: cd backend && python scripts/download_ai_models.py"
echo ""
EOF

chmod +x activate_env.sh

# 개발 서버 실행 스크립트
cat > run_dev.sh << 'EOF'
#!/bin/bash
# MyCloset AI 개발 서버 실행

# Conda 환경 활성화
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mycloset-m3

# 백엔드 디렉토리로 이동
cd backend

echo "🚀 MyCloset AI 개발 서버 시작..."
echo "📍 서버 주소: http://localhost:8000"
echo "📚 API 문서: http://localhost:8000/docs"
echo "❤️ 헬스체크: http://localhost:8000/health"
echo ""

# GPU 상태 출력
python -c "
import torch
if torch.backends.mps.is_available():
    print('✅ M3 Max GPU (Metal) 활성화됨')
else:
    print('⚠️ CPU 모드로 실행')
print()
"

# 서버 실행 (main.py가 있으면 실행, 없으면 메시지 출력)
if [[ -f "app/main.py" ]]; then
    python app/main.py
else
    echo "⚠️ backend/app/main.py 파일이 없습니다."
    echo "먼저 FastAPI 애플리케이션을 생성하세요."
fi
EOF

chmod +x run_dev.sh

log_success "편의 스크립트 생성 완료"

# 8. 완료 메시지
log_header "🎉 M3 Max 최적화 환경 설정 완료!"
echo ""
log_success "Conda 환경 '$CONDA_ENV_NAME' 준비 완료"
log_success "M3 Max GPU (Metal) 최적화 적용됨"
log_success "기존 프로젝트 구조 보완 완료"
echo ""

echo "🚀 다음 단계:"
echo "1. 환경 활성화: source activate_env.sh  (또는 conda activate $CONDA_ENV_NAME)"
echo "2. GPU 설정 파일 추가: backend/app/core/gpu_config.py"
echo "3. FastAPI 앱 생성: backend/app/main.py"
echo "4. AI 모델 다운로드: python backend/scripts/download_ai_models.py"
echo "5. 개발 서버 실행: ./run_dev.sh"
echo ""

echo "📚 생성된 파일:"
echo "- activate_env.sh : 환경 활성화"
echo "- run_dev.sh : 개발 서버 실행"
echo "- backend/.env : 환경변수 설정"
echo "- backend/requirements.txt : 패키지 목록"
echo ""

echo "📋 현재 환경:"
echo "- Conda 환경: $CONDA_ENV_NAME"
echo "- Python: $(python --version 2>/dev/null || echo '환경 활성화 필요')"
echo "- 프로젝트: $(pwd)"
echo ""

log_warning "현재 세션에서 환경을 사용하려면:"
echo "conda activate $CONDA_ENV_NAME"