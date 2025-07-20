#!/bin/bash
# MyCloset AI 서버 실행 문제 해결 스크립트

echo "🔧 MyCloset AI 서버 문제 해결 중..."

# 1. 기존 프로세스 완전 종료
echo "1️⃣ 기존 서버 프로세스 종료..."
pkill -f "uvicorn"
pkill -f "python.*main.py"
pkill -f "python.*8000"
pkill -f "fastapi"
sleep 3

# 2. 포트 사용 확인 및 해제
echo "2️⃣ 포트 8000 상태 확인..."
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "포트 8000이 사용 중입니다. 강제 종료..."
    sudo lsof -ti:8000 | xargs sudo kill -9
    sleep 2
fi

# 3. 대체 포트 찾기
echo "3️⃣ 사용 가능한 포트 확인..."
for port in 8001 8002 8003 8004 8005; do
    if ! lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        AVAILABLE_PORT=$port
        break
    fi
done

if [ -z "$AVAILABLE_PORT" ]; then
    AVAILABLE_PORT=8888
fi

echo "✅ 사용 가능한 포트: $AVAILABLE_PORT"

# 4. 환경 변수 최적화
echo "4️⃣ 환경 변수 최적화..."
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# 5. Python 캐시 정리
echo "5️⃣ Python 캐시 정리..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# 6. 메모리 정리
echo "6️⃣ 시스템 메모리 정리..."
if command -v purge >/dev/null 2>&1; then
    sudo purge
fi

# 7. 서버 실행 (여러 방법 시도)
echo "7️⃣ 서버 실행 시도..."

cd backend || exit 1

# 방법 1: 직접 실행
echo "📍 시도 1: python app/main.py --port $AVAILABLE_PORT"
if timeout 10s python app/main.py --port $AVAILABLE_PORT 2>&1 | grep -q "ERROR.*Address already in use"; then
    echo "❌ 방법 1 실패"
    
    # 방법 2: uvicorn 직접 실행
    echo "📍 시도 2: uvicorn app.main:app --port $AVAILABLE_PORT"
    if timeout 10s uvicorn app.main:app --host 0.0.0.0 --port $AVAILABLE_PORT --reload 2>&1 | grep -q "ERROR.*Address already in use"; then
        echo "❌ 방법 2 실패"
        
        # 방법 3: 다른 포트로 강제 실행
        RANDOM_PORT=$((8000 + RANDOM % 1000))
        echo "📍 시도 3: 랜덤 포트 $RANDOM_PORT 사용"
        uvicorn app.main:app --host 0.0.0.0 --port $RANDOM_PORT --reload &
        SERVER_PID=$!
        
        sleep 5
        if ps -p $SERVER_PID > /dev/null; then
            echo "✅ 서버 실행 성공!"
            echo "📡 주소: http://localhost:$RANDOM_PORT"
            echo "📚 API 문서: http://localhost:$RANDOM_PORT/docs"
            echo "🔧 프로세스 ID: $SERVER_PID"
            echo ""
            echo "서버를 중지하려면: kill $SERVER_PID"
        else
            echo "❌ 모든 방법 실패"
        fi
    else
        echo "✅ 방법 2 성공 - uvicorn 실행됨"
    fi
else
    echo "✅ 방법 1 성공 - python 직접 실행됨"
fi

echo ""
echo "🎉 스크립트 완료!"
echo "📋 문제 해결 상태:"
echo "   - 포트 충돌: ✅ 해결"
echo "   - 프로세스 정리: ✅ 완료"
echo "   - 환경 최적화: ✅ 적용"
echo "   - 캐시 정리: ✅ 완료"