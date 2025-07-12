#!/bin/bash

echo "🏗️ MyCloset AI 프로젝트 구조 확인"
echo "================================="

PROJECT_ROOT=$(pwd)
echo "📁 프로젝트 루트: $PROJECT_ROOT"
echo ""

# 1. 루트 레벨 확인
echo "📂 루트 디렉토리:"
ls -la | grep -E '^d|^-.*\.(md|json|yml|yaml|txt|sh)$' | head -10
echo ""

# 2. 백엔드 구조 확인
if [ -d "backend" ]; then
    echo "🔧 백엔드 구조:"
    echo "  backend/"
    find backend -type d -maxdepth 2 | sort | sed 's/^/    /'
    echo ""
    
    echo "  주요 파일들:"
    find backend -name "*.py" -maxdepth 2 | head -10 | sed 's/^/    /'
    echo ""
else
    echo "❌ backend 디렉토리 없음"
fi

# 3. 프론트엔드 구조 확인
if [ -d "frontend" ]; then
    echo "🎨 프론트엔드 구조:"
    echo "  frontend/"
    find frontend -type d -maxdepth 2 | sort | sed 's/^/    /'
    echo ""
    
    if [ -f "frontend/package.json" ]; then
        echo "  ✅ package.json 존재"
    else
        echo "  ❌ package.json 없음"
    fi
else
    echo "❌ frontend 디렉토리 없음"
fi

# 4. AI 모델 디렉토리 확인
if [ -d "backend/ai_models" ]; then
    echo "🤖 AI 모델 구조:"
    echo "  backend/ai_models/"
    find backend/ai_models -type d | head -10 | sed 's/^/    /'
    echo ""
else
    echo "❌ ai_models 디렉토리 없음"
fi

# 5. 스크립트 디렉토리 확인
if [ -d "scripts" ]; then
    echo "📜 스크립트 디렉토리:"
    ls -la scripts/ | sed 's/^/    /'
    echo ""
else
    echo "❌ scripts 디렉토리 없음"
fi

# 6. 설정 파일들 확인
echo "⚙️ 중요 설정 파일들:"
files=(".gitignore" "README.md" "backend/.env" "backend/requirements.txt" "frontend/package.json")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (누락)"
    fi
done
echo ""

# 7. 가상환경 확인
if [ -d "backend/venv" ]; then
    echo "✅ Python 가상환경 존재"
else
    echo "❌ Python 가상환경 없음"
fi

if [ -d "frontend/node_modules" ]; then
    echo "✅ Node.js 의존성 설치됨"
else
    echo "❌ Node.js 의존성 미설치"
fi

echo ""
echo "🎯 구조 확인 완료!"
