#!/bin/bash
# 🔍 현재 모델 체크포인트 위치 자동 탐지

echo "🔍 MyCloset AI 체크포인트 현재 위치 탐지..."
echo "현재 디렉토리: $(pwd)"
echo "Conda 환경: ${CONDA_DEFAULT_ENV:-없음}"
echo ""

# 1. 기본 PyTorch 모델 파일들 찾기
echo "📁 PyTorch 모델 파일 (.pth, .pt) 탐지:"
find . -name "*.pth" -o -name "*.pt" 2>/dev/null | head -10

echo ""
echo "📁 기타 AI 모델 파일 (.bin, .safetensors) 탐지:"
find . -name "*.bin" -o -name "*.safetensors" 2>/dev/null | head -10

echo ""
echo "📊 모델 파일 크기 순 정렬 (상위 10개):"
find . \( -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" \) -exec ls -lh {} \; 2>/dev/null | sort -k5 -hr | head -10

echo ""
echo "🎯 Step별 특화 모델 탐지:"

echo "  Human Parsing 관련:"
find . -name "*.pth" 2>/dev/null | grep -i -E "(schp|atr|graphonomy|human|parsing)" | head -3

echo "  Pose Estimation 관련:"
find . -name "*.pth" 2>/dev/null | grep -i -E "(openpose|pose|body)" | head -3

echo "  U2Net/Segmentation 관련:"
find . -name "*.pth" 2>/dev/null | grep -i -E "(u2net|segmentation|cloth)" | head -3

echo "  Diffusion/OOTD 관련:"
find . \( -name "*.bin" -o -name "*.safetensors" -o -name "*.pth" \) 2>/dev/null | grep -i -E "(diffusion|ootd|stable|unet)" | head -3

echo ""
echo "📂 주요 디렉토리 구조:"
find . -type d -name "*model*" -o -name "*checkpoint*" -o -name "*ai_*" 2>/dev/null | head -10

echo ""
echo "✅ 탐지 완료!"