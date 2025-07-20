#!/bin/bash

# =============================================================================
# MyCloset AI - M3 Max libjpeg 최적화 스크립트
# Conda 환경에서 이미지 처리 성능 100% 활용
# =============================================================================

echo "🍎 M3 Max libjpeg 최적화 시작..."

# 현재 conda 환경 확인
if [[ "$CONDA_DEFAULT_ENV" != "mycloset-ai" ]]; then
    echo "⚠️ mycloset-ai conda 환경을 먼저 활성화하세요"
    echo "실행: conda activate mycloset-ai"
    exit 1
fi

echo "✅ conda 환경: $CONDA_DEFAULT_ENV"

# 1. 기존 이미지 라이브러리 완전 제거 후 재설치
echo "🔧 기존 이미지 라이브러리 제거 중..."
conda remove --yes libjpeg-turbo libpng zlib pillow opencv -q 2>/dev/null || true

# 2. M3 Max 최적화 이미지 라이브러리 설치
echo "📦 M3 Max 최적화 이미지 라이브러리 설치 중..."
conda install --yes -c conda-forge \
    libjpeg-turbo=3.0.0 \
    libpng=1.6.39 \
    zlib=1.2.13 \
    pillow=10.1.0 \
    opencv=4.8.1

# 3. PyTorch 이미지 확장 모듈 재설치
echo "🔥 PyTorch 이미지 확장 재설치 중..."
pip uninstall -y torchvision
conda install --yes pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 2>/dev/null || \
conda install --yes pytorch torchvision torchaudio -c pytorch

# 4. 라이브러리 링크 확인
echo "🔗 라이브러리 링크 확인 중..."
python3 -c "
import torchvision
print('✅ torchvision import 성공')
try:
    from torchvision.io import read_image
    print('✅ torchvision.io 이미지 확장 사용 가능')
except Exception as e:
    print(f'⚠️ torchvision.io 확장 문제: {e}')

import cv2
print(f'✅ OpenCV 버전: {cv2.__version__}')

from PIL import Image
print(f'✅ PIL 버전: {Image.__version__}')
"

# 5. M3 Max GPU 이미지 처리 테스트
echo "🧪 M3 Max GPU 이미지 처리 테스트..."
python3 -c "
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('🍎 M3 Max MPS 디바이스 사용')
    
    # 더미 이미지로 GPU 처리 테스트
    dummy_image = Image.new('RGB', (512, 512), (255, 0, 0))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(dummy_image).unsqueeze(0).to(device)
    print(f'✅ GPU 텐서 생성 성공: {tensor.shape}, 디바이스: {tensor.device}')
    
    # 간단한 이미지 변환 연산
    resized = torch.nn.functional.interpolate(tensor, size=(256, 256), mode='bilinear')
    print(f'✅ GPU 이미지 리사이즈 성공: {resized.shape}')
else:
    print('⚠️ MPS 사용 불가')
"

echo ""
echo "🎉 M3 Max libjpeg 최적화 완료!"
echo "이제 서버를 재시작하세요:"
echo "python3 backend/app/main.py"