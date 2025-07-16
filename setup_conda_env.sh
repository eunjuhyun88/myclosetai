#!/bin/bash
# MyCloset AI Conda 환경 설정 스크립트

echo "🐍 MyCloset AI Conda 환경 설정 시작..."

# 1. Conda 환경 생성
conda create -n mycloset_env python=3.11 -y

# 2. 환경 활성화
conda activate mycloset_env

# 3. PyTorch MPS 설치 (M3 Max 최적화)
conda install pytorch torchvision torchaudio -c pytorch -y

# 4. 필수 패키지 설치
conda install numpy scipy scikit-learn scikit-image -y
conda install opencv pillow -y
pip install transformers diffusers accelerate
pip install fastapi uvicorn websockets aiofiles
pip install pydantic pydantic-settings

# 5. M3 Max 최적화 설정
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export OMP_NUM_THREADS=16

echo "✅ Conda 환경 설정 완료!"
echo "사용법: conda activate mycloset_env"
