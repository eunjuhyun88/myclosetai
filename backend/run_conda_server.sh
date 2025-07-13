#!/bin/bash
echo "🚀 Conda 환경에서 MyCloset AI 서버 시작..."

# Conda 환경 확인
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "❌ Conda 환경이 활성화되지 않았습니다."
    echo "📝 실행: conda activate mycloset"
    exit 1
fi

echo "✅ Conda 환경: $CONDA_DEFAULT_ENV"

# 모델 설정 실행
echo "🔧 모델 설정 확인 중..."
python scripts/download_models_conda.py

# 서버 시작
echo "🌐 서버 시작 중..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
