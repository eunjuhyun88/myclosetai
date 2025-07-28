#!/bin/bash
# 🔥 MyCloset AI - Conda 환경 필수 패키지 설치 스크립트
# M3 Max + conda 환경 최적화

echo "🔥 MyCloset AI - Conda 패키지 설치 시작"
echo "================================================"

# 현재 conda 환경 확인
if [[ "$CONDA_DEFAULT_ENV" != "mycloset-ai-clean" ]]; then
    echo "⚠️  mycloset-ai-clean 환경을 활성화해주세요:"
    echo "conda activate mycloset-ai-clean"
    exit 1
fi

echo "✅ conda 환경: $CONDA_DEFAULT_ENV"
echo "================================================"

# 1. 기본 채널 설정 (M3 Max 최적화)
echo "🔧 conda 채널 설정..."
conda config --add channels conda-forge
conda config --add channels pytorch  
conda config --add channels huggingface
conda config --set channel_priority flexible

# 2. PyTorch + MPS 지원 (M3 Max 최적화)
echo "🧠 PyTorch MPS 지원 설치..."
conda install pytorch torchvision torchaudio -c pytorch -y

# 3. Transformers (Hugging Face)
echo "🤖 Transformers 설치..."
conda install transformers -c huggingface -y

# 4. Diffusers 
echo "🎨 Diffusers 설치..."
pip install diffusers[torch]

# 또는 conda로 시도
# conda install diffusers -c conda-forge -y

# 5. 추가 AI 라이브러리들
echo "📚 추가 AI 라이브러리 설치..."
conda install -c conda-forge -y \
    safetensors \
    accelerate \
    xformers \
    tokenizers

# 6. 컴퓨터 비전 라이브러리들  
echo "👁️ 컴퓨터 비전 라이브러리 설치..."
conda install -c conda-forge -y \
    opencv \
    scikit-image \
    imageio \
    pillow

# 7. 과학 계산 라이브러리들
echo "🔬 과학 계산 라이브러리 설치..."
conda install -c conda-forge -y \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn

# 8. 개발 도구들
echo "🛠️ 개발 도구 설치..."
conda install -c conda-forge -y \
    jupyter \
    ipython \
    tqdm \
    wandb

# 9. pip로 추가 패키지 설치
echo "📦 pip 패키지 설치..."
pip install --upgrade \
    timm \
    controlnet-aux \
    invisible-watermark \
    clip-by-openai \
    open-clip-torch \
    segment-anything \
    rembg[new] \
    onnxruntime

# 10. M3 Max 최적화 환경 변수 설정
echo "🍎 M3 Max 최적화 설정..."
cat >> ~/.zshrc << 'EOF'

# MyCloset AI M3 Max 최적화
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

EOF

# 11. conda 환경 정리
echo "🧹 conda 환경 정리..."
conda clean --all -y

echo "================================================"
echo "✅ MyCloset AI 패키지 설치 완료!"
echo "================================================"

# 12. 설치 확인 스크립트
echo "🔍 설치 확인 중..."
python << 'EOF'
import sys
print(f"🐍 Python: {sys.version}")

# 필수 패키지 확인
packages = {
    'torch': 'PyTorch',
    'torchvision': 'TorchVision', 
    'transformers': 'Transformers',
    'diffusers': 'Diffusers',
    'PIL': 'Pillow',
    'cv2': 'OpenCV',
    'scipy': 'SciPy',
    'numpy': 'NumPy',
    'safetensors': 'SafeTensors'
}

print("\n📊 패키지 설치 상태:")
for package, name in packages.items():
    try:
        __import__(package)
        print(f"✅ {name}")
    except ImportError:
        print(f"❌ {name}")

# MPS 지원 확인 (M3 Max)
try:
    import torch
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✅ MPS (M3 Max) 가속 지원")
    else:
        print("❌ MPS 가속 미지원")
except:
    print("❌ PyTorch MPS 확인 실패")

print("\n🎉 설치 확인 완료!")
EOF

echo "================================================"
echo "🚀 MyCloset AI 준비 완료!"
echo "이제 VirtualFittingStep v13.0을 실행할 수 있습니다."
echo "================================================"