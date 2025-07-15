#!/bin/bash
# M3 Max MyCloset AI 환경 활성화

echo "🍎 M3 Max MyCloset AI 환경 활성화 중..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mycloset-m3max

echo "✅ 환경 활성화 완료"
echo "🔧 Python: $(python --version)"
echo "⚡ PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "🍎 MPS: $(python -c 'import torch; print("Available" if torch.backends.mps.is_available() else "Not Available")')"
echo ""
