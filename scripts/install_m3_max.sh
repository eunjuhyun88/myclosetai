#!/bin/bash
echo "🍎 M3 Max 전용 설치 시작..."

cd backend

# 기존 venv 제거 (있다면)
if [ -d "venv" ]; then
    echo "기존 가상환경 제거 중..."
    rm -rf venv
fi

# 새 가상환경 생성
echo "📦 M3 Max 최적화 가상환경 생성..."
python3 -m venv venv
source venv/bin/activate

# pip 업그레이드
pip install --upgrade pip

# M3 Max 최적화 패키지 설치
echo "⚡ M3 Max 최적화 패키지 설치 중..."
pip install -r requirements-mac.txt

# PyTorch MPS 테스트
python3 -c "
import torch
print(f'MPS 사용 가능: {torch.backends.mps.is_available()}')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    z = torch.mm(x, y)
    print('✅ M3 Max GPU 정상 동작')
else:
    print('⚠️ CPU 모드로 동작')
"

echo "✅ M3 Max 최적화 설치 완료!"
