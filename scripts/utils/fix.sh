#!/bin/bash

echo "🚨 긴급 상태 확인 시작..."
echo "=================================================="

# 현재 위치 확인
echo "📍 현재 위치:"
pwd

echo -e "\n📁 현재 디렉토리 구조:"
find . -maxdepth 3 -type d | head -20

echo -e "\n🔧 Git 상태 확인:"
git status --short

echo -e "\n📊 Git 로그 (최근 5개 커밋):"
git log --oneline -5

echo -e "\n🌿 Git 브랜치 확인:"
git branch

echo -e "\n📋 중요 파일들 존재 여부:"
echo "backend/app/main.py: $([ -f backend/app/main.py ] && echo '✅ 존재' || echo '❌ 없음')"
echo "backend/app/services/: $([ -d backend/app/services ] && echo '✅ 존재' || echo '❌ 없음')"
echo "frontend/: $([ -d frontend ] && echo '✅ 존재' || echo '❌ 없음')"
echo ".gitignore: $([ -f .gitignore ] && echo '✅ 존재' || echo '❌ 없음')"

echo -e "\n💾 저장소 크기:"
du -sh . 2>/dev/null || echo "크기 측정 실패"

echo -e "\n🔍 백업 파일들 확인:"
find . -name "*.backup" -o -name "*.bak" | head -10

echo -e "\n⚠️ 문제가 될 수 있는 파일들:"
find . -name "__pycache__" -o -name "*.pyc" -o -name ".DS_Store" | head -10

echo -e "\n=================================================="
echo "🎯 상태 확인 완료"