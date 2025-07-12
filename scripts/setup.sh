#!/bin/bash

echo "🚀 MyCloset AI 완전 개발 환경 설정 시작..."

# 현재 디렉토리 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "📁 프로젝트 루트: $PROJECT_ROOT"

# 1. 시스템 요구사항 확인
echo "\n🔍 시스템 요구사항 확인 중..."

# Python 버전 확인
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_python="3.9"

if [ "$(printf '%s\n' "$required_python" "$python_version" | sort -V | head -n1)" = "$required_python" ]; then
    echo "✅ Python $python_version (>= $required_python 필요)"
else
    echo "❌ Python $required_python 이상이 필요합니다. 현재: $python_version"
    exit 1
fi

# Node.js 확인
if command -v node > /dev/null 2>&1; then
    node_version=$(node --version | cut -d'v' -f2 | cut -d. -f1)
    if [ "$node_version" -ge 18 ]; then
        echo "✅ Node.js $(node --version)"
    else
        echo "⚠️ Node.js 18+ 권장. 현재: $(node --version)"
    fi
else
    echo "❌ Node.js가 설치되어 있지 않습니다."
    echo "Node.js 설치: https://nodejs.org/"
    exit 1
fi

# Git 확인
if ! command -v git > /dev/null 2>&1; then
    echo "❌ Git이 설치되어 있지 않습니다."
    exit 1
fi

# 2. 백엔드 설정
echo "\n🔧 백엔드 설정 중..."
cd "$PROJECT_ROOT/backend"

# 가상환경 생성
if [ ! -d "venv" ]; then
    echo "📦 Python 가상환경 생성 중..."
    python3 -m venv venv
fi

# 가상환경 활성화
source venv/bin/activate

# 의존성 설치
echo "📚 Python 의존성 설치 중..."
pip install --upgrade pip
pip install -r requirements.txt

# 환경설정 파일 생성
if [ ! -f ".env" ]; then
    echo "⚙️ 환경설정 파일 생성 중..."
    cp .env.example .env
fi

# 필요한 디렉토리 생성
echo "📁 디렉토리 구조 생성 중..."
mkdir -p static/{uploads,results} ai_models logs
touch static/uploads/.gitkeep static/results/.gitkeep logs/.gitkeep

# 3. 프론트엔드 설정
echo "\n🎨 프론트엔드 설정 중..."
cd "$PROJECT_ROOT/frontend"

# Node.js 의존성 설치
echo "📚 Node.js 의존성 설치 중..."
npm install

# Tailwind CSS 설정
echo "🎨 Tailwind CSS 설정 중..."
if [ ! -f "tailwind.config.js" ]; then
    npx tailwindcss init -p
fi

# 4. AI 모델 다운로드
cd "$PROJECT_ROOT"
read -p "🤖 AI 모델을 다운로드하시겠습니까? (Y/n): " download_models
if [[ $download_models =~ ^[Yy]$ ]] || [[ -z $download_models ]]; then
    echo "🤖 AI 모델 다운로드 중..."
    python scripts/download_models.py
fi

# 5. Docker 설정 (선택적)
read -p "🐳 Docker 설정을 구성하시겠습니까? (y/N): " setup_docker
if [[ $setup_docker =~ ^[Yy]$ ]]; then
    echo "🐳 Docker 설정 중..."
    # Docker 설정 스크립트 실행
    # (여기에 Docker 설정 로직 추가)
fi

echo "\n✅ 설정 완료!"
echo "\n🚀 서버 시작 방법:"
echo "1. 백엔드 서버:"
echo "   cd backend && source venv/bin/activate && python -m uvicorn app.main:app --reload"
echo ""
echo "2. 프론트엔드 서버:"
echo "   cd frontend && npm run dev"
echo ""
echo "📱 접속 주소:"
echo "   프론트엔드: http://localhost:5173"
echo "   백엔드 API: http://localhost:8000"
echo "   API 문서: http://localhost:8000/docs"
