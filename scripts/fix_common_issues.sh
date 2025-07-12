#!/bin/bash

echo "🛠️ MyCloset AI 문제 해결"
echo "===================="

cd backend

# 가상환경 활성화
source venv/bin/activate

# Python 경로 문제 해결
echo "🔧 Python 경로 문제 해결 중..."

# 모든 __init__.py 파일 생성
touch app/__init__.py
touch app/api/__init__.py
touch app/core/__init__.py

# 실행 스크립트 생성
cat > run_server.py << 'RUNEOF'
#!/usr/bin/env python3
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print(f"🐍 Python 경로: {sys.path[0]}")
print(f"📁 작업 디렉토리: {project_root}")

# 앱 실행
exec(open('app/main.py').read())
RUNEOF

chmod +x run_server.py

echo "✅ 문제 해결 완료!"
echo ""
echo "🚀 다음 명령어로 서버 실행:"
echo "python run_server.py"
