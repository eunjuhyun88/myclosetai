#!/bin/bash

echo "🔥 PyTorch 테스트 서버 실행"
echo "========================="

# Conda 환경 확인
if [[ "$CONDA_DEFAULT_ENV" != "mycloset" ]]; then
    echo "❌ mycloset conda 환경이 활성화되지 않았습니다."
    echo "conda activate mycloset"
    exit 1
fi

# PyTorch 간단 확인
python -c "import torch; print(f'✅ PyTorch {torch.__version__} 준비됨')" || {
    echo "❌ PyTorch가 제대로 설치되지 않았습니다."
    exit 1
}

echo "🚀 테스트 서버 시작 중..."
python test_pytorch_server.py
