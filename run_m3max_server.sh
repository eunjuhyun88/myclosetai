#!/bin/bash
# M3 Max MyCloset AI 서버 실행

source activate_m3max.sh
cd backend

echo "🚀 M3 Max MyCloset AI 서버 시작..."
echo "📡 서버: http://localhost:8000"
echo "📚 API 문서: http://localhost:8000/docs"
echo ""

if [[ -f "app/main.py" ]]; then
    python app/main.py
else
    echo "⚠️ backend/app/main.py가 없습니다."
    echo "FastAPI 애플리케이션을 먼저 생성하세요."
fi
