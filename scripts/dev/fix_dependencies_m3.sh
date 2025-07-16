#!/bin/bash

# MyCloset AI - M3 Max 128GB + Conda 전용 환경 설정
# 실제 하드웨어 스펙에 최적화된 설정

set -e

echo "🍎 M3 Max 128GB + Conda 환경 설정"
echo "================================================"
echo "🖥️  Target: Apple M3 Max 128GB Unified Memory"
echo "🐍 Package Manager: Conda (miniforge/mambaforge)"
echo "⚡ Optimization: Metal Performance Shaders"
echo ""

# 색상 출력 함수들
log_info() { echo -e "\033[34m[INFO]\033[0m $1"; }
log_success() { echo -e "\033[32m[SUCCESS]\033[0m $1"; }
log_warning() { echo -e "\033[33m[WARNING]\033[0m $1"; }
log_error() { echo -e "\033[31m[ERROR]\033[0m $1"; }
log_header() { echo -e "\033[35m\n=== $1 ===\033[0m"; }

# M3 Max 환경 확인
check_m3_max_environment() {
    log_header "M3 Max 128GB 환경 확인"
    
    # macOS 확인
    if [[ "$OSTYPE" != "darwin"* ]]; then
        log_error "이 스크립트는 macOS 전용입니다."
        exit 1
    fi
    
    # M3 Max 칩셋 확인
    CHIP_INFO=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
    echo "🔍 CPU: $CHIP_INFO"
    
    # 메모리 확인
    MEMORY_GB=$(echo "$(sysctl -n hw.memsize) / 1024 / 1024 / 1024" | bc)
    echo "💾 메모리: ${MEMORY_GB}GB"
    
    if [[ $MEMORY_GB -lt 64 ]]; then
        log_warning "128GB 메모리가 아닌 것 같습니다. 계속하시겠습니까? (y/N)"
        read -r confirm
        if [[ ! $confirm =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Conda 확인
    if ! command -v conda &> /dev/null; then
        log_error "Conda가 설치되지 않았습니다."
        echo "miniforge를 설치하세요: https://github.com/conda-forge/miniforge"
        exit 1
    fi
    
    CONDA_VERSION=$(conda --version)
    echo "🐍 $CONDA_VERSION"
    
    log_success "M3 Max 128GB 환경 확인 완료"
}

# Conda 환경 생성
create_conda_environment() {
    log_header "Conda 환경 생성 (M3 Max 최적화)"
    
    ENV_NAME="mycloset-m3max"
    PYTHON_VERSION="3.11"  # M3 Max에서 가장 안정적
    
    # 기존 환경 확인
    if conda env list | grep -q "$ENV_NAME"; then
        log_warning "환경 '$ENV_NAME'이 이미 존재합니다. 재생성하시겠습니까? (y/N)"
        read -r recreate
        if [[ $recreate =~ ^[Yy]$ ]]; then
            log_info "기존 환경 제거 중..."
            conda env remove -n "$ENV_NAME" -y
        else
            log_info "기존 환경을 사용합니다."
            return 0
        fi
    fi
    
    log_info "Conda 환경 생성: $ENV_NAME (Python $PYTHON_VERSION)"
    
    # M3 Max 최적화 환경 생성
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    
    log_success "Conda 환경 '$ENV_NAME' 생성 완료"
    echo "💡 환경 활성화: conda activate $ENV_NAME"
}

# M3 Max 최적화 패키지 설치
install_m3_max_packages() {
    log_header "M3 Max 128GB 최적화 패키지 설치"
    
    ENV_NAME="mycloset-m3max"
    
    # 환경 활성화
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    
    log_info "활성 환경: $(conda info --envs | grep '*' | awk '{print $1}')"
    
    # 1. 기본 시스템 패키지 (conda-forge)
    log_info "🔧 기본 시스템 패키지 설치 중..."
    conda install -c conda-forge -y \
        numpy=1.24.3 \
        scipy=1.11.4 \
        matplotlib=3.7.2 \
        pillow=10.0.1 \
        opencv=4.8.1 \
        scikit-image=0.21.0 \
        psutil=5.9.5 \
        tqdm=4.66.1
    
    # 2. PyTorch 생태계 (M3 Max MPS 지원)
    log_info "⚡ PyTorch MPS 지원 버전 설치 중..."
    conda install -c pytorch -c nvidia -y \
        pytorch=2.1.0 \
        torchvision=0.16.0 \
        torchaudio=2.1.0
    
    # MPS 지원 확인
    python -c "
import torch
print(f'🔍 PyTorch 버전: {torch.__version__}')
if torch.backends.mps.is_available():
    print('✅ Metal Performance Shaders (MPS) 사용 가능')
    device = torch.device('mps')
    x = torch.randn(1000, 1000, device=device)
    y = torch.mm(x, x.T)
    print(f'✅ M3 Max GPU 연산 테스트 성공: {y.shape}')
else:
    print('❌ MPS 사용 불가')
    exit(1)
    "
    
    # 3. AI/ML 패키지들
    log_info "🤖 AI/ML 패키지 설치 중..."
    
    # pip로 최신 버전 설치 (conda보다 빠름)
    pip install --upgrade pip
    
    # Transformers 생태계
    pip install \
        transformers==4.35.0 \
        tokenizers==0.15.0 \
        safetensors==0.4.0 \
        accelerate==0.24.1 \
        datasets==2.14.6 \
        huggingface-hub==0.17.3
    
    # Diffusers (가상 피팅용)
    pip install diffusers==0.21.4
    
    # 4. 웹 프레임워크
    log_info "🌐 웹 프레임워크 설치 중..."
    pip install \
        fastapi==0.104.1 \
        uvicorn[standard]==0.24.0 \
        python-multipart==0.0.6 \
        aiofiles==23.2.1 \
        websockets==11.0.3
    
    # 5. 데이터 검증 및 설정
    pip install \
        pydantic==2.5.0 \
        pydantic-settings==2.1.0 \
        python-dotenv==1.0.0 \
        structlog==23.1.0
    
    # 6. M3 Max 특화 패키지들
    log_info "🍎 M3 Max 특화 패키지 설치 중..."
    
    # MediaPipe (M3 Max 최적화 버전)
    pip install mediapipe==0.10.7
    
    # Core ML Tools (Apple 전용)
    pip install coremltools==7.0
    
    # Metal Performance Shaders Python 바인딩 (있다면)
    pip install metal-python || log_warning "metal-python 설치 실패 (선택적)"
    
    log_success "M3 Max 128GB 최적화 패키지 설치 완료"
}

# 환경 설정 파일 생성
create_environment_config() {
    log_header "M3 Max 환경 설정 파일 생성"
    
    # 프로젝트 루트로 이동
    cd "$(dirname "$0")"
    
    # backend/.env 파일 생성
    mkdir -p backend
    cat > backend/.env << 'EOF'
# MyCloset AI - M3 Max 128GB 최적화 설정
# Generated automatically for Conda environment

# ===========================================
# 애플리케이션 기본 설정
# ===========================================
APP_NAME=MyCloset AI Backend
APP_VERSION=1.0.0
DEBUG=true
HOST=0.0.0.0
PORT=8000

# ===========================================
# M3 Max 128GB 하드웨어 최적화
# ===========================================
# GPU 설정
USE_GPU=true
DEVICE=mps
GPU_TYPE=m3_max
MEMORY_GB=128
UNIFIED_MEMORY=true

# Neural Engine 설정
NEURAL_ENGINE_ENABLED=true
METAL_PERFORMANCE_SHADERS=true

# 메모리 관리
MAX_MEMORY_FRACTION=0.75
MEMORY_POOL_SIZE=32
AUTO_MEMORY_CLEANUP=true
MEMORY_THRESHOLD=0.85

# ===========================================
# AI 모델 최적화 설정
# ===========================================
# 배치 크기 (128GB 메모리 활용)
DEFAULT_BATCH_SIZE=8
MAX_BATCH_SIZE=16
INFERENCE_BATCH_SIZE=4

# 모델 정밀도
MODEL_PRECISION=float32
ENABLE_MIXED_PRECISION=false

# 파이프라인 설정
PIPELINE_WORKERS=4
PARALLEL_PROCESSING=true
ASYNC_PROCESSING=true

# ===========================================
# PyTorch MPS 최적화
# ===========================================
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
PYTORCH_MPS_ALLOCATOR_POLICY=garbage_collection
PYTORCH_ENABLE_MPS_FALLBACK=1

# ===========================================
# 성능 최적화
# ===========================================
# CPU 설정
OMP_NUM_THREADS=12
MKL_NUM_THREADS=12
VECLIB_MAXIMUM_THREADS=12

# I/O 최적화
MAX_WORKERS=8
ASYNC_CONCURRENCY=16

# ===========================================
# 파일 처리 설정
# ===========================================
MAX_UPLOAD_SIZE=104857600
ALLOWED_EXTENSIONS=jpg,jpeg,png,webp,heic
TEMP_DIR=/tmp/mycloset_ai

# 이미지 처리
DEFAULT_IMAGE_SIZE=1024
MAX_IMAGE_SIZE=2048
IMAGE_QUALITY=95

# ===========================================
# AI 모델 경로
# ===========================================
MODELS_DIR=./ai_models
CACHE_DIR=./models_cache
RESULTS_DIR=./static/results
UPLOADS_DIR=./static/uploads

# ===========================================
# 로깅 설정
# ===========================================
LOG_LEVEL=INFO
LOG_FILE=logs/mycloset_m3max.log
LOG_ROTATION=true
LOG_MAX_SIZE=100MB

# ===========================================
# CORS 및 보안
# ===========================================
CORS_ORIGINS=http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000
CORS_ALLOW_CREDENTIALS=true

# ===========================================
# 개발 도구
# ===========================================
RELOAD=true
ACCESS_LOG=true
DEBUG_TOOLBAR=true
PROFILING=false
EOF

    # conda 환경 활성화 스크립트
    cat > activate_m3max.sh << 'EOF'
#!/bin/bash
# M3 Max 128GB MyCloset AI 환경 활성화

echo "🍎 M3 Max 128GB MyCloset AI 환경 활성화 중..."

# Conda 환경 활성화
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mycloset-m3max

# 환경 변수 설정
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# M3 Max 최적화 확인
echo "✅ 환경 활성화 완료"
echo "🔧 Python: $(python --version)"
echo "⚡ PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "🍎 MPS: $(python -c 'import torch; print("Available" if torch.backends.mps.is_available() else "Not Available")')"
echo "💾 메모리: $(python -c 'import psutil; print(f"{psutil.virtual_memory().total/1024**3:.0f}GB")')"
echo ""
echo "🚀 사용법:"
echo "  cd backend && python app/main.py"
echo "  또는"
echo "  ./run_m3max.sh"
EOF

    chmod +x activate_m3max.sh

    # 실행 스크립트
    cat > run_m3max.sh << 'EOF'
#!/bin/bash
# M3 Max MyCloset AI 개발 서버 실행

# 환경 활성화
source activate_m3max.sh

# 백엔드 디렉토리로 이동
cd backend

echo "🚀 M3 Max MyCloset AI 백엔드 시작..."
echo "📡 서버: http://localhost:8000"
echo "📚 API 문서: http://localhost:8000/docs"
echo "❤️ 헬스체크: http://localhost:8000/health"
echo ""

# 서버 실행
if [[ -f "app/main.py" ]]; then
    python app/main.py
else
    echo "⚠️ backend/app/main.py가 없습니다."
    echo "FastAPI 애플리케이션을 먼저 생성하세요."
fi
EOF

    chmod +x run_m3max.sh

    log_success "환경 설정 파일 생성 완료"
}

# 성능 벤치마크
run_m3_max_benchmark() {
    log_header "M3 Max 성능 벤치마크"
    
    ENV_NAME="mycloset-m3max"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    
    python -c "
import torch
import time
import numpy as np
import psutil

print('🍎 M3 Max 128GB 성능 벤치마크')
print('=' * 50)

# 시스템 정보
print(f'💾 총 메모리: {psutil.virtual_memory().total / 1024**3:.1f}GB')
print(f'💾 사용 가능: {psutil.virtual_memory().available / 1024**3:.1f}GB')
print(f'🔧 CPU 코어: {psutil.cpu_count(logical=False)}개 (논리: {psutil.cpu_count()}개)')

# PyTorch 정보
print(f'⚡ PyTorch: {torch.__version__}')
print(f'🍎 MPS: {\"✅ 사용 가능\" if torch.backends.mps.is_available() else \"❌ 사용 불가\"}')

if torch.backends.mps.is_available():
    device = torch.device('mps')
    print(f'🎯 디바이스: {device}')
    
    # 메모리 대역폭 테스트
    print('\\n🚀 메모리 대역폭 테스트...')
    sizes = [1000, 2000, 4000, 8000]
    
    for size in sizes:
        # CPU
        start = time.time()
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        c_cpu = torch.mm(a_cpu, b_cpu)
        cpu_time = time.time() - start
        
        # MPS
        start = time.time()
        a_mps = torch.randn(size, size, device=device)
        b_mps = torch.randn(size, size, device=device)
        c_mps = torch.mm(a_mps, b_mps)
        torch.mps.synchronize()
        mps_time = time.time() - start
        
        speedup = cpu_time / mps_time
        print(f'  {size}x{size}: CPU {cpu_time:.3f}s vs MPS {mps_time:.3f}s (가속: {speedup:.1f}x)')
    
    # 대용량 배치 테스트 (128GB 활용)
    print('\\n💪 대용량 배치 처리 테스트...')
    batch_sizes = [1, 4, 8, 16, 32]
    
    for batch in batch_sizes:
        try:
            start = time.time()
            images = torch.randn(batch, 3, 512, 512, device=device)
            # 간단한 컨볼루션 연산
            conv = torch.nn.Conv2d(3, 64, 3, padding=1).to(device)
            output = conv(images)
            torch.mps.synchronize()
            elapsed = time.time() - start
            
            memory_used = psutil.virtual_memory().percent
            print(f'  배치 {batch}: {elapsed:.3f}s (메모리: {memory_used:.1f}%)')
            
        except Exception as e:
            print(f'  배치 {batch}: 실패 - {e}')
            break
    
    print('\\n✅ M3 Max 벤치마크 완료')
    print(f'💡 권장 배치 크기: 8-16 (메모리 여유분 고려)')
    print(f'🍎 128GB 통합 메모리의 장점을 최대한 활용하세요!')

else:
    print('❌ MPS를 사용할 수 없습니다.')
    print('PyTorch와 macOS 버전을 확인하세요.')
"
}

# 패키지 목록 저장
save_conda_requirements() {
    log_header "패키지 목록 저장"
    
    ENV_NAME="mycloset-m3max"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    
    # conda 환경 export
    conda env export > environment_m3max.yml
    
    # pip requirements
    pip freeze > requirements_m3max.txt
    
    # 요약 정보
    cat > package_summary.md << 'EOF'
# MyCloset AI - M3 Max 128GB 패키지 목록

## 환경 정보
- **하드웨어**: Apple M3 Max 128GB
- **OS**: macOS (Apple Silicon)
- **Python**: 3.11
- **패키지 관리자**: Conda + pip

## 핵심 패키지
- **PyTorch**: 2.1.0 (MPS 지원)
- **Transformers**: 4.35.0
- **Diffusers**: 0.21.4
- **FastAPI**: 0.104.1
- **OpenCV**: 4.8.1

## 재현 방법
```bash
# 1. 환경 생성
conda env create -f environment_m3max.yml

# 2. 환경 활성화
conda activate mycloset-m3max

# 3. 서버 실행
./run_m3max.sh
```

## 성능 최적화
- Metal Performance Shaders (MPS) 활성화
- 128GB 통합 메모리 최대 활용
- Neural Engine 연동 준비
- 배치 크기: 8-16 권장
EOF

    log_success "패키지 목록 저장 완료"
    echo "📦 environment_m3max.yml - Conda 환경 파일"
    echo "📦 requirements_m3max.txt - pip 패키지 목록"
    echo "📦 package_summary.md - 요약 정보"
}

# 메인 실행 함수
main() {
    echo "🍎 M3 Max 128GB + Conda 환경 설정을 시작합니다."
    echo ""
    echo "다음 단계들이 실행됩니다:"
    echo "1. M3 Max 128GB 환경 확인"
    echo "2. Conda 환경 생성 (mycloset-m3max)"
    echo "3. M3 Max 최적화 패키지 설치"
    echo "4. 환경 설정 파일 생성"
    echo "5. 성능 벤치마크 실행"
    echo "6. 패키지 목록 저장"
    echo ""
    
    read -p "계속하시겠습니까? (y/N): " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        echo "설정이 취소되었습니다."
        exit 0
    fi
    
    # 단계별 실행
    check_m3_max_environment
    create_conda_environment
    install_m3_max_packages
    create_environment_config
    
    # 벤치마크 실행 여부 확인
    echo ""
    read -p "성능 벤치마크를 실행하시겠습니까? (y/N): " benchmark
    if [[ $benchmark =~ ^[Yy]$ ]]; then
        run_m3_max_benchmark
    fi
    
    save_conda_requirements
    
    # 완료 메시지
    log_header "🎉 M3 Max 128GB 환경 설정 완료!"
    echo ""
    log_success "Conda 환경 'mycloset-m3max' 준비 완료"
    log_success "M3 Max 128GB 최적화 설정 적용"
    log_success "Metal Performance Shaders (MPS) 활성화"
    echo ""
    
    echo "🚀 다음 단계:"
    echo "1. 환경 활성화: source activate_m3max.sh"
    echo "2. 개발 서버 실행: ./run_m3max.sh"
    echo "3. API 문서 확인: http://localhost:8000/docs"
    echo ""
    
    echo "📋 생성된 파일들:"
    echo "- activate_m3max.sh: 환경 활성화 스크립트"
    echo "- run_m3max.sh: 서버 실행 스크립트"
    echo "- backend/.env: M3 Max 최적화 환경 변수"
    echo "- environment_m3max.yml: Conda 환경 백업"
    echo "- requirements_m3max.txt: pip 패키지 목록"
    echo ""
    
    echo "💡 사용법:"
    echo "  source activate_m3max.sh && ./run_m3max.sh"
    echo ""
    
    log_info "M3 Max 128GB 환경 설정 완료! 🎉"
}

# 스크립트 실행
main "$@"