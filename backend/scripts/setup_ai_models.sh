#!/bin/bash
# scripts/setup_ai_models.sh
# MyCloset AI 모델 완전 자동 설정 스크립트

set -e  # 에러 발생 시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수들
log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

echo "🤖 MyCloset AI 모델 완전 설정 시작"
echo "=================================="

# 1. 환경 확인
log_info "시스템 환경 확인 중..."

# Python 버전 확인
if ! command -v python3 &> /dev/null; then
    log_error "Python3가 설치되지 않았습니다."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
log_success "Python $PYTHON_VERSION 감지됨"

# GPU 확인
log_info "GPU 환경 확인 중..."
if command -v nvidia-smi &> /dev/null; then
    log_success "NVIDIA GPU 감지됨"
    DEVICE="cuda"
elif python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
    log_success "Apple Silicon (MPS) 감지됨"
    DEVICE="mps"
else
    log_warning "GPU를 감지할 수 없음. CPU 모드로 실행됩니다."
    DEVICE="cpu"
fi

# 2. 가상환경 설정
log_info "Python 가상환경 설정 중..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    log_success "가상환경 생성 완료"
fi

# 가상환경 활성화
source venv/bin/activate || {
    log_error "가상환경 활성화 실패"
    exit 1
}

log_success "가상환경 활성화됨"

# 3. 기본 의존성 설치
log_info "기본 패키지 설치 중..."
pip install --upgrade pip setuptools wheel

# 4. AI 의존성 설치
log_info "AI 라이브러리 설치 중... (시간이 걸릴 수 있습니다)"

# PyTorch 설치 (디바이스별)
if [ "$DEVICE" = "cuda" ]; then
    log_info "CUDA용 PyTorch 설치 중..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
elif [ "$DEVICE" = "mps" ]; then
    log_info "Apple Silicon용 PyTorch 설치 중..."
    pip install torch torchvision torchaudio
else
    log_info "CPU용 PyTorch 설치 중..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 나머지 AI 라이브러리 설치
log_info "추가 AI 라이브러리 설치 중..."
pip install -r requirements-ai.txt

# 5. 기본 백엔드 의존성 설치
log_info "백엔드 의존성 설치 중..."
pip install -r requirements.txt

# 6. 디렉토리 구조 생성
log_info "AI 모델 디렉토리 구조 생성 중..."

mkdir -p ai_models/{checkpoints,configs,temp}
mkdir -p ai_models/checkpoints/{ootdiffusion,viton_hd,human_parsing,background_removal}
mkdir -p static/{uploads,results}
mkdir -p logs

# .gitkeep 파일 생성
find ai_models -type d -exec touch {}/.gitkeep \;
find static -type d -exec touch {}/.gitkeep \;
touch logs/.gitkeep

log_success "디렉토리 구조 생성 완료"

# 7. AI 모델 다운로드
log_info "AI 모델 다운로드 시작..."

if [ ! -f "scripts/download_ai_models.py" ]; then
    log_error "download_ai_models.py 스크립트를 찾을 수 없습니다."
    exit 1
fi

python scripts/download_ai_models.py

# 8. 설정 파일 생성
log_info "설정 파일 생성 중..."

# GPU 설정 파일 생성
cat > app/core/gpu_config.py << EOF
"""
GPU 및 디바이스 설정
"""
import torch

# 디바이스 자동 감지
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"🚀 CUDA GPU 사용: {torch.cuda.get_device_name()}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps" 
    print("🍎 Apple Silicon MPS 사용")
else:
    DEVICE = "cpu"
    print("💻 CPU 사용")

# 모델 설정
MODEL_CONFIG = {
    "device": DEVICE,
    "dtype": torch.float32 if DEVICE == "mps" else torch.float16,
    "memory_fraction": 0.8,
    "enable_attention_slicing": True,
    "enable_memory_efficient_attention": DEVICE != "mps"
}

# GPU 메모리 최적화
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
elif DEVICE == "mps":
    # MPS 최적화 설정
    torch.backends.mps.empty_cache()
EOF

# 이미지 유틸리티 생성
log_info "유틸리티 함수 생성 중..."

cat > app/utils/image_utils.py << 'EOF'
"""
이미지 처리 유틸리티 함수들
"""
import io
from typing import Tuple
from PIL import Image, ImageEnhance, ImageFilter

def resize_image(image: Image.Image, target_size: Tuple[int, int], maintain_ratio: bool = True) -> Image.Image:
    """이미지 크기 조정"""
    if maintain_ratio:
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # 정사각형으로 패딩
        new_image = Image.new('RGB', target_size, (255, 255, 255))
        paste_x = (target_size[0] - image.width) // 2
        paste_y = (target_size[1] - image.height) // 2
        new_image.paste(image, (paste_x, paste_y))
        return new_image
    else:
        return image.resize(target_size, Image.Resampling.LANCZOS)

def enhance_image_quality(image: Image.Image) -> Image.Image:
    """이미지 품질 향상"""
    # 선명도 향상
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.1)
    
    # 대비 향상
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.05)
    
    return image

def convert_to_rgb(image: Image.Image) -> Image.Image:
    """RGB로 변환"""
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image

async def validate_image_content(image_bytes: bytes) -> bool:
    """이미지 내용 검증"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        
        # 최소/최대 크기 검사
        if width < 100 or height < 100:
            return False
        if width > 4096 or height > 4096:
            return False
            
        return True
    except Exception:
        return False
EOF

# 9. 테스트 스크립트 생성
log_info "테스트 스크립트 생성 중..."

cat > scripts/test_models.py << 'EOF'
#!/usr/bin/env python3
"""
AI 모델 테스트 스크립트
"""
import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from app.services.ai_models import model_manager
from app.services.real_working_ai_fitter import RealWorkingAIFitter
from PIL import Image

async def test_models():
    print("🧪 AI 모델 테스트 시작...")
    
    try:
        # 1. 모델 관리자 초기화
        await model_manager.initialize_models()
        print("✅ 모델 관리자 초기화 완료")
        
        # 2. 사용 가능한 모델 확인
        available_models = model_manager.get_available_models()
        print(f"📋 사용 가능한 모델: {available_models}")
        
        # 3. 더미 이미지로 테스트
        dummy_person = Image.new('RGB', (512, 512), color='white')
        dummy_clothing = Image.new('RGB', (512, 512), color='blue')
        
        if available_models:
            print("🎨 가상 피팅 테스트 중...")
            result_image, metadata = await model_manager.generate_virtual_fitting(
                dummy_person, dummy_clothing
            )
            print(f"✅ 가상 피팅 테스트 성공: {metadata}")
        
        # 4. AI 서비스 테스트
        ai_fitter = RealWorkingAIFitter()
        status = await ai_fitter.get_model_status()
        print(f"📊 AI 서비스 상태: {status}")
        
        print("🎉 모든 테스트 통과!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False
    
    return True

if __name__ == "__main__":
    result = asyncio.run(test_models())
    sys.exit(0 if result else 1)
EOF

chmod +x scripts/test_models.py

# 10. 서비스 시작 스크립트 생성
cat > scripts/start_server.sh << 'EOF'
#!/bin/bash
# AI 모델 서버 시작 스크립트

echo "🚀 MyCloset AI 서버 시작..."

# 가상환경 활성화
source venv/bin/activate

# 모델 테스트
echo "🧪 모델 상태 확인 중..."
python scripts/test_models.py

if [ $? -eq 0 ]; then
    echo "✅ 모델 상태 정상"
    echo "🌐 서버 시작 중..."
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
else
    echo "❌ 모델 상태 이상. 서버 시작을 중단합니다."
    exit 1
fi
EOF

chmod +x scripts/start_server.sh

# 11. 최종 검증
log_info "설치 검증 중..."

# Python 패키지 검증
python3 -c "
import torch
import PIL
import cv2
import numpy as np
print('✅ 필수 패키지 임포트 성공')
print(f'PyTorch 버전: {torch.__version__}')
print(f'디바이스: {torch.device(\"$DEVICE\")}')
"

# 12. 완료 메시지
echo ""
log_success "MyCloset AI 모델 설정 완료!"
echo ""
echo "📋 설치된 구성요소:"
echo "   ✅ PyTorch ($DEVICE 지원)"
echo "   ✅ Hugging Face Transformers"
echo "   ✅ Computer Vision 라이브러리들"
echo "   ✅ AI 모델 다운로드 스크립트"
echo "   ✅ 통합 서비스 클래스들"
echo ""
echo "🚀 다음 단계:"
echo "   1. 모델 테스트: python scripts/test_models.py"
echo "   2. 서버 시작: ./scripts/start_server.sh"
echo "   3. API 테스트: curl http://localhost:8000/api/models"
echo ""
echo "📚 문서:"
echo "   - API 문서: http://localhost:8000/docs"
echo "   - 모델 상태: http://localhost:8000/api/status"
echo ""
echo "⚠️  주의사항:"
echo "   - 첫 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다"
echo "   - GPU 메모리가 부족하면 CPU 모드로 전환됩니다"
echo "   - 일부 모델은 별도의 라이센스가 필요할 수 있습니다"