#!/bin/bash

echo "🚀 MyCloset AI Backend - 수정된 버전 실행"
echo "=========================================="

# Conda 환경 확인
if [[ "$CONDA_DEFAULT_ENV" == "" ]]; then
    echo "❌ Conda 환경이 활성화되지 않았습니다."
    echo "conda activate mycloset"
    exit 1
fi

echo "✅ Conda 환경: $CONDA_DEFAULT_ENV"

# 패키지 확인
echo "📦 필수 패키지 확인 중..."

# FastAPI 확인
python -c "import fastapi; print(f'✅ FastAPI: {fastapi.__version__}')" 2>/dev/null || {
    echo "❌ FastAPI가 없습니다. 설치: conda install fastapi uvicorn -y"
    exit 1
}

# 서버 시작
echo ""
echo "🌐 서버 시작 중..."
echo "📱 접속 주소: http://localhost:8000"
echo "📚 API 문서: http://localhost:8000/docs"
echo "🔧 헬스체크: http://localhost:8000/api/health"
echo "🧪 가상 피팅 테스트: http://localhost:8000/api/virtual-tryon"
echo ""
echo "⏹️ 종료하려면 Ctrl+C를 누르세요"
echo ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
