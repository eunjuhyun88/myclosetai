#!/bin/bash
# 🔧 포트 8000 통일 스크립트

echo "🔧 포트 8000으로 통일 중..."

# 1. 실행 중인 서버들 모두 중지
echo "🛑 기존 서버 프로세스 중지 중..."
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "8001" 2>/dev/null || true
pkill -f "8000" 2>/dev/null || true
pkill -f "python app/main.py" 2>/dev/null || true

# PID 파일들도 정리
rm -f server.pid frontend.pid 2>/dev/null

sleep 3

# 2. 백엔드 main.py에서 포트 8000으로 변경
echo "📝 백엔드 포트 설정 변경 중..."
cd backend

# main.py에서 8001을 8000으로 변경
sed -i.backup 's/port=8001/port=8000/g' app/main.py
sed -i.backup 's/localhost:8001/localhost:8000/g' app/main.py
sed -i.backup 's/:8001/:8000/g' app/main.py

echo "✅ 백엔드 포트 설정 8000으로 변경 완료"

# 3. 포트 8000이 사용 중인지 확인
if lsof -i :8000 >/dev/null 2>&1; then
    echo "⚠️ 포트 8000이 사용 중입니다. 해당 프로세스를 종료합니다..."
    lsof -ti :8000 | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# 4. 백엔드 서버 시작 (포트 8000)
echo "🚀 백엔드 서버 시작 (포트 8000)..."

# 가상환경 확인 및 활성화
if [ -d "mycloset_env" ]; then
    source mycloset_env/bin/activate
    echo "✅ mycloset_env 가상환경 활성화"
elif [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ venv 가상환경 활성화"
else
    echo "⚠️ 가상환경을 찾을 수 없습니다. 전역 Python 사용"
fi

# 서버 시작 (직접 포트 8000 지정)
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > ../server.log 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > ../server.pid

echo "✅ 백엔드 서버 시작됨 (PID: $SERVER_PID, 포트: 8000)"

cd ..

# 5. 프론트엔드도 재시작
echo "🎨 프론트엔드 재시작 중..."
cd frontend

# 기존 프론트엔드 프로세스 종료
pkill -f "npm run dev" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true

# 프론트엔드 시작
if [ -d "node_modules" ]; then
    nohup npm run dev > ../frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > ../frontend.pid
    echo "✅ 프론트엔드 서버 시작됨 (PID: $FRONTEND_PID)"
else
    echo "⚠️ node_modules가 없습니다. npm install을 먼저 실행하세요"
fi

cd ..

# 6. 연결 테스트 (15초 대기)
echo "🧪 연결 테스트 중... (15초 대기)"
sleep 15

# 백엔드 테스트
echo "📡 백엔드 연결 테스트..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ 백엔드 서버 정상 동작 (http://localhost:8000)"
    
    # API 상세 테스트
    if curl -s http://localhost:8000/api/system/info > /dev/null; then
        echo "✅ API 엔드포인트 정상 동작"
    else
        echo "⚠️ API 엔드포인트 확인 필요"
    fi
else
    echo "❌ 백엔드 서버 연결 실패"
    echo "🔍 서버 로그 확인:"
    tail -10 server.log
    exit 1
fi

# 프론트엔드 테스트
echo "🎨 프론트엔드 연결 테스트..."
if curl -s http://localhost:5173 > /dev/null; then
    echo "✅ 프론트엔드 서버 정상 동작 (http://localhost:5173)"
else
    echo "⚠️ 프론트엔드 연결 확인 필요"
fi

# 7. 포트 사용 현황 확인
echo "📊 포트 사용 현황:"
echo "포트 8000: $(lsof -i :8000 2>/dev/null | wc -l | xargs)개 프로세스"
echo "포트 5173: $(lsof -i :5173 2>/dev/null | wc -l | xargs)개 프로세스"

echo ""
echo "🎉 포트 통일 완료!"
echo "======================="
echo ""
echo "📱 접속 주소:"
echo "   프론트엔드: http://localhost:5173"
echo "   백엔드: http://localhost:8000"
echo "   API 문서: http://localhost:8000/docs"
echo ""
echo "🔍 로그 확인:"
echo "   백엔드: tail -f server.log"
echo "   프론트엔드: tail -f frontend.log"
echo ""
echo "🛑 서버 중지:"
echo "   kill \$(cat server.pid frontend.pid) 2>/dev/null"
echo ""
echo "🧪 테스트 방법:"
echo "   1. 브라우저에서 http://localhost:5173 접속"
echo "   2. 이미지 업로드 후 'Complete Pipeline' 클릭"
echo "   3. 개발자 도구 (F12) → Network 탭 확인"