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
