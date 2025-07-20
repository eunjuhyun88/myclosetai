#!/bin/bash

echo "🔧 MyCloset AI 경로 수정 및 서버 재시작"
echo "======================================"

# 1. 현재 위치 확인
echo "📍 현재 위치: $(pwd)"
echo "📁 파일 구조 확인:"
ls -la

# 2. 올바른 디렉토리로 이동 (이미 backend에 있다면 그대로, 아니면 이동)
if [[ $(basename $(pwd)) == "backend" ]]; then
    echo "✅ 이미 backend 디렉토리에 있습니다"
    BACKEND_DIR="."
else
    echo "📂 backend 디렉토리로 이동..."
    cd backend
    BACKEND_DIR="."
fi

# 3. 필요한 파일들 존재 확인
echo "📋 필수 파일 확인:"
if [ -f "app/main.py" ]; then
    echo "✅ app/main.py 존재"
else
    echo "❌ app/main.py 없음"
    find . -name "main.py" -type f
fi

if [ -f "app/ai_pipeline/utils/model_loader.py" ]; then
    echo "✅ model_loader.py 존재"
else
    echo "❌ model_loader.py 없음"
    find . -name "model_loader.py" -type f
fi

# 4. 서버 종료
echo "📋 기존 서버 프로세스 종료..."
pkill -f "python.*main.py" 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true
sleep 3

# 5. Python 캐시 정리
echo "🧹 Python 캐시 정리..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# 6. PYTHONPATH 설정 및 import 테스트
echo "🔍 PYTHONPATH 설정 및 import 테스트..."
export PYTHONPATH="$(pwd):$PYTHONPATH"
echo "📊 PYTHONPATH: $PYTHONPATH"

# Python 모듈 경로 테스트
python3 -c "
import sys
print('🐍 Python 실행 경로:', sys.executable)
print('📦 sys.path:')
for p in sys.path:
    print(f'  - {p}')

# 현재 디렉토리를 sys.path에 추가
import os
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    print(f'✅ 현재 디렉토리 추가: {current_dir}')

# import 테스트
try:
    print('🔍 model_loader import 시도...')
    from app.ai_pipeline.utils.model_loader import preprocess_image
    print('✅ preprocess_image import 성공!')
except ImportError as e:
    print(f'❌ ImportError: {e}')
    
    # 대안 경로로 시도
    try:
        import app.ai_pipeline.utils.model_loader as ml
        if hasattr(ml, 'preprocess_image'):
            print('✅ 대안 방법으로 preprocess_image 찾음!')
        else:
            print('❌ preprocess_image 함수가 모듈에 없음')
            print('📋 모듈의 __all__ 속성:', getattr(ml, '__all__', 'None'))
    except Exception as e2:
        print(f'❌ 대안 방법도 실패: {e2}')
        
except Exception as e:
    print(f'❌ 기타 오류: {e}')
"

# 7. 서버 시작 (올바른 경로로)
echo ""
echo "🚀 서버 시작..."
echo "=================================="

# PYTHONPATH 환경변수와 함께 서버 실행
PYTHONPATH="$(pwd):$PYTHONPATH" python3 app/main.py