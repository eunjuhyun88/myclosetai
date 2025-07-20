#!/bin/bash

echo "🛡️ MyCloset AI Backend - 안전 모드 실행"
echo "======================================"

# 환경변수 설정
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Conda 환경 확인
if [[ "$CONDA_DEFAULT_ENV" == "" ]]; then
    echo "❌ Conda 환경이 활성화되지 않았습니다."
    echo "conda activate mycloset"
    exit 1
fi

echo "✅ Conda 환경: $CONDA_DEFAULT_ENV"

# PyTorch 테스트
echo "🧪 PyTorch 안전성 테스트 중..."
python test_pytorch_safe.py

if [[ $? -eq 0 ]]; then
    echo "✅ PyTorch 테스트 성공"
    echo ""
    echo "🌐 안전 모드 서버 시작 중..."
    echo "📱 접속: http://localhost:8000"
    echo "📚 API 문서: http://localhost:8000/docs"
    echo "🧪 PyTorch 테스트: http://localhost:8000/api/test-pytorch"
    echo ""
    
    # 안전한 main.py 실행
    uvicorn app.main_safe:app --reload --host 0.0.0.0 --port 8000
else
    echo "❌ PyTorch 테스트 실패"
    echo "기본 웹서버만 실행합니다..."
    
    # 최소한의 서버 실행
    python -c "
from fastapi import FastAPI
import uvicorn

app = FastAPI(title='MyCloset AI - 최소 서버')

@app.get('/')
def root():
    return {'message': 'MyCloset AI Backend - 최소 모드', 'status': 'pytorch_disabled'}

uvicorn.run(app, host='0.0.0.0', port=8000)
"
fi
