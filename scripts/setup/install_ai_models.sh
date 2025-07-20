#!/bin/bash
# AI 모델 자동 설치 스크립트

echo '🚀 MyCloset AI 모델 설치 시작'
echo '================================'

# Python 환경 확인
if ! command -v python3 &> /dev/null; then
    echo '❌ Python3가 설치되지 않았습니다'
    exit 1
fi

# 필요한 라이브러리 설치
echo '📦 라이브러리 설치 중...'
pip install transformers diffusers torch torchvision onnxruntime
pip install mediapipe opencv-python pillow

# 모델 다운로드
echo '📥 모델 다운로드 중...'
python3 -c "
import sys
sys.path.append('.')
from complete_model_check import AIModelChecker
checker = AIModelChecker()
results = checker.check_all_models()
checker.download_missing_models(results, ['critical', 'high'])
"

echo '✅ 설치 완료!'