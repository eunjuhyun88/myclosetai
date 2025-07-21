#!/bin/bash

echo "🔍 MyCloset AI 모델 파일 구조 분석"
echo "======================================"

# conda 환경 확인
echo "🐍 Conda 환경 확인:"
echo "   - 현재 환경: $CONDA_DEFAULT_ENV"
echo "   - Python 경로: $(which python)"
echo ""

# 프로젝트 루트 찾기
PROJECT_ROOT="/Users/gimdudeul/MVP/mycloset-ai"
echo "📁 프로젝트 루트: $PROJECT_ROOT"
echo ""

# AI 모델 디렉토리들 확인
echo "🤖 AI 모델 디렉토리 검색:"

# 1. backend/ai_models 확인
if [ -d "$PROJECT_ROOT/backend/ai_models" ]; then
    echo "✅ backend/ai_models 존재"
    du -sh "$PROJECT_ROOT/backend/ai_models" 2>/dev/null || echo "   - 크기 측정 실패"
    echo "   - 하위 디렉토리:"
    find "$PROJECT_ROOT/backend/ai_models" -maxdepth 2 -type d | sort
else
    echo "❌ backend/ai_models 없음"
fi

echo ""

# 2. ai_models (루트) 확인
if [ -d "$PROJECT_ROOT/ai_models" ]; then
    echo "✅ ai_models (루트) 존재"
    du -sh "$PROJECT_ROOT/ai_models" 2>/dev/null || echo "   - 크기 측정 실패"
    echo "   - 하위 디렉토리:"
    find "$PROJECT_ROOT/ai_models" -maxdepth 2 -type d | sort
else
    echo "❌ ai_models (루트) 없음"
fi

echo ""

# 3. 실제 .pth, .pt, .bin 파일들 찾기
echo "🔍 실제 AI 모델 파일 검색 (.pth, .pt, .bin, .safetensors):"
echo ""

# 전체 프로젝트에서 모델 파일 검색
MODEL_FILES=$(find "$PROJECT_ROOT" -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" \) 2>/dev/null | head -20)

if [ -n "$MODEL_FILES" ]; then
    echo "📦 발견된 모델 파일들:"
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            size=$(ls -lh "$file" | awk '{print $5}')
            echo "   - $(basename "$file") ($size)"
            echo "     위치: $file"
        fi
    done <<< "$MODEL_FILES"
else
    echo "❌ 모델 파일을 찾을 수 없습니다."
fi

echo ""

# 4. Hugging Face 캐시 확인
echo "🤗 Hugging Face 캐시 확인:"
HF_CACHE_DIRS=(
    "$HOME/.cache/huggingface"
    "$PROJECT_ROOT/backend/ai_models/huggingface_cache"
    "$PROJECT_ROOT/ai_models/huggingface_cache"
)

for cache_dir in "${HF_CACHE_DIRS[@]}"; do
    if [ -d "$cache_dir" ]; then
        echo "✅ $cache_dir 존재"
        du -sh "$cache_dir" 2>/dev/null || echo "   - 크기 측정 실패"
    else
        echo "❌ $cache_dir 없음"
    fi
done

echo ""

# 5. 큰 파일들 찾기 (1GB 이상)
echo "📊 큰 파일들 검색 (1GB 이상):"
find "$PROJECT_ROOT" -type f -size +1G 2>/dev/null | head -10 | while read file; do
    if [ -f "$file" ]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "   - $(basename "$file") ($size)"
        echo "     위치: $file"
    fi
done

echo ""

# 6. 권장 해결책
echo "🛠 권장 해결책:"
echo "   1. 모델 파일이 없다면: 다운로드 스크립트 실행"
echo "   2. 경로가 다르다면: 심볼릭 링크 생성"
echo "   3. 권한 문제라면: chmod 755 적용"
echo ""

echo "✅ 검사 완료"