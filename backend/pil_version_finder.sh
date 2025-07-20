#!/bin/bash

# ============================================================================
# PIL.Image.VERSION 오류 찾기 및 수정 스크립트
# PIL.Image.VERSION은 최신 Pillow에서 제거되었음 (PIL.__version__ 사용해야 함)
# ============================================================================

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }
log_header() { echo -e "${PURPLE}🔍 $1${NC}"; }

log_header "PIL.Image.VERSION 오류 찾기 스크립트"
echo "=========================================="

# 현재 작업 디렉토리 확인
CURRENT_DIR=$(pwd)
log_info "작업 디렉토리: $CURRENT_DIR"

# 1. PIL.Image.VERSION 사용하는 파일들 찾기
log_header "1. PIL.Image.VERSION 사용 파일 검색"
echo ""

# Python 파일에서 PIL.Image.VERSION 검색
log_info "Python 파일에서 PIL.Image.VERSION 검색 중..."
echo ""

PIL_VERSION_FILES=()

# backend 디렉토리에서 검색
if [ -d "backend" ]; then
    log_info "backend 디렉토리 검색 중..."
    while IFS= read -r -d '' file; do
        if grep -l "PIL\.Image\.VERSION\|Image\.VERSION" "$file" 2>/dev/null; then
            PIL_VERSION_FILES+=("$file")
            echo "  📁 발견: $file"
            # 해당 라인 표시
            grep -n "PIL\.Image\.VERSION\|Image\.VERSION" "$file" | head -3 | while read line; do
                echo "    └─ $line"
            done
            echo ""
        fi
    done < <(find backend -name "*.py" -type f -print0 2>/dev/null)
fi

# frontend 디렉토리에서도 검색 (혹시 Python 파일이 있을 수 있음)
if [ -d "frontend" ]; then
    log_info "frontend 디렉토리 검색 중..."
    while IFS= read -r -d '' file; do
        if grep -l "PIL\.Image\.VERSION\|Image\.VERSION" "$file" 2>/dev/null; then
            PIL_VERSION_FILES+=("$file")
            echo "  📁 발견: $file"
            grep -n "PIL\.Image\.VERSION\|Image\.VERSION" "$file" | head -3 | while read line; do
                echo "    └─ $line"
            done
            echo ""
        fi
    done < <(find frontend -name "*.py" -type f -print0 2>/dev/null)
fi

# 루트 디렉토리의 Python 파일들도 검색
log_info "루트 디렉토리 Python 파일 검색 중..."
while IFS= read -r -d '' file; do
    if grep -l "PIL\.Image\.VERSION\|Image\.VERSION" "$file" 2>/dev/null; then
        PIL_VERSION_FILES+=("$file")
        echo "  📁 발견: $file"
        grep -n "PIL\.Image\.VERSION\|Image\.VERSION" "$file" | head -3 | while read line; do
            echo "    └─ $line"
        done
        echo ""
    fi
done < <(find . -maxdepth 1 -name "*.py" -type f -print0 2>/dev/null)

# 2. 더 넓은 범위 검색 (다른 패턴들)
log_header "2. 관련 패턴 추가 검색"
echo ""

log_info "PIL 관련 VERSION 패턴 검색 중..."
ADDITIONAL_PATTERNS=(
    "from PIL import.*VERSION"
    "PIL.*__version__"
    "Image.*__version__"
    "\.VERSION"
    "version.*PIL"
    "PIL.*version"
)

for pattern in "${ADDITIONAL_PATTERNS[@]}"; do
    echo "🔍 패턴: $pattern"
    
    if [ -d "backend" ]; then
        while IFS= read -r -d '' file; do
            if grep -l "$pattern" "$file" 2>/dev/null; then
                echo "  📁 $file"
                grep -n "$pattern" "$file" | head -2 | while read line; do
                    echo "    └─ $line"
                done
            fi
        done < <(find backend -name "*.py" -type f -print0 2>/dev/null)
    fi
    echo ""
done

# 3. 특정 파일들 상세 검사
log_header "3. 의심 파일 상세 검사"
echo ""

SUSPECT_FILES=(
    "backend/app/ai_pipeline/utils/__init__.py"
    "backend/app/ai_pipeline/utils/utils.py"
    "backend/app/ai_pipeline/utils/data_converter.py"
    "backend/app/ai_pipeline/utils/image_processor.py"
    "backend/app/core/m3_optimizer.py"
    "backend/app/main.py"
)

for file in "${SUSPECT_FILES[@]}"; do
    if [ -f "$file" ]; then
        log_info "검사 중: $file"
        
        # PIL import 확인
        if grep -n "from PIL\|import PIL" "$file" 2>/dev/null; then
            echo "  ✅ PIL import 발견"
            
            # VERSION 사용 확인
            if grep -n "VERSION\|__version__" "$file" 2>/dev/null; then
                echo "  ⚠️  VERSION 관련 코드 발견:"
                grep -n "VERSION\|__version__" "$file" | while read line; do
                    echo "    └─ $line"
                done
            fi
        fi
        echo ""
    fi
done

# 4. 로그 파일에서 오류 추적
log_header "4. 로그에서 오류 위치 추적"
echo ""

log_info "로그 파일에서 PIL.Image.VERSION 오류 검색 중..."

# 일반적인 로그 파일 위치들
LOG_LOCATIONS=(
    "backend/logs/"
    "logs/"
    "backend/"
    "./"
)

for log_dir in "${LOG_LOCATIONS[@]}"; do
    if [ -d "$log_dir" ]; then
        echo "📂 $log_dir 검색 중..."
        while IFS= read -r -d '' file; do
            if grep -l "PIL\.Image.*VERSION\|module 'PIL.Image' has no attribute 'VERSION'" "$file" 2>/dev/null; then
                echo "  📄 로그 발견: $file"
                echo "  📋 오류 컨텍스트:"
                grep -A 5 -B 5 "PIL\.Image.*VERSION\|module 'PIL.Image' has no attribute 'VERSION'" "$file" | head -10
                echo ""
            fi
        done < <(find "$log_dir" -name "*.log" -o -name "*.txt" -type f -print0 2>/dev/null)
    fi
done

# 5. 스택 트레이스에서 파일 위치 찾기
log_header "5. Python 코드에서 PIL.Image.VERSION 직접 검색"
echo ""

log_info "더 정확한 검색을 위해 정규표현식 사용..."

# 정확한 패턴으로 재검색
find . -name "*.py" -type f -exec grep -l "PIL\.Image\.VERSION\|Image\.VERSION" {} \; 2>/dev/null | while read file; do
    echo "🎯 정확한 매치 발견: $file"
    echo "📄 해당 코드:"
    grep -n -A 2 -B 2 "PIL\.Image\.VERSION\|Image\.VERSION" "$file"
    echo ""
done

# 6. 수정 스크립트 생성
log_header "6. 자동 수정 스크립트 생성"
echo ""

cat > fix_pil_version.py << 'EOF'
#!/usr/bin/env python3
"""
PIL.Image.VERSION을 PIL.__version__으로 자동 수정하는 스크립트
"""

import os
import re
import shutil
from pathlib import Path

def fix_pil_version_in_file(file_path):
    """파일에서 PIL.Image.VERSION을 수정"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # PIL.Image.VERSION -> PIL.__version__ 으로 변경
        patterns_to_fix = [
            (r'PIL\.Image\.VERSION', 'PIL.__version__'),
            (r'Image\.VERSION', 'PIL.__version__'),
            (r'from PIL import.*Image.*\n.*Image\.VERSION', 'import PIL\nPIL.__version__'),
        ]
        
        changes_made = []
        for pattern, replacement in patterns_to_fix:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                changes_made.append(f"{pattern} -> {replacement}")
        
        if content != original_content:
            # 백업 생성
            backup_path = f"{file_path}.backup_pil_fix"
            shutil.copy2(file_path, backup_path)
            
            # 수정된 내용 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ 수정됨: {file_path}")
            print(f"📝 백업: {backup_path}")
            for change in changes_made:
                print(f"   - {change}")
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ 오류: {file_path} - {e}")
        return False

def main():
    """메인 함수"""
    print("🔧 PIL.Image.VERSION 자동 수정 시작...")
    
    fixed_files = []
    
    # Python 파일들 검색 및 수정
    for py_file in Path('.').rglob('*.py'):
        if fix_pil_version_in_file(py_file):
            fixed_files.append(str(py_file))
    
    print(f"\n🎉 수정 완료: {len(fixed_files)}개 파일")
    for file in fixed_files:
        print(f"  - {file}")
    
    if fixed_files:
        print("\n⚠️  백업 파일들이 생성되었습니다.")
        print("수정 후 테스트가 완료되면 백업 파일들을 삭제하세요:")
        print("find . -name '*.backup_pil_fix' -delete")

if __name__ == "__main__":
    main()
EOF

chmod +x fix_pil_version.py

log_success "자동 수정 스크립트 생성됨: fix_pil_version.py"
echo ""

# 7. 결과 요약
log_header "7. 검색 결과 요약"
echo ""

if [ ${#PIL_VERSION_FILES[@]} -gt 0 ]; then
    log_warning "발견된 문제 파일들:"
    for file in "${PIL_VERSION_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    log_info "수정 방법:"
    echo "1. 자동 수정: python3 fix_pil_version.py"
    echo "2. 수동 수정: PIL.Image.VERSION → PIL.__version__"
    echo ""
else
    log_success "PIL.Image.VERSION 사용하는 파일을 찾지 못했습니다."
    log_info "로그에서 오류가 발생했다면 다른 라이브러리에서 사용하는 것일 수 있습니다."
    echo ""
fi

# 8. 추가 디버깅 정보
log_header "8. 추가 디버깅 명령어"
echo ""

echo "🔍 더 자세한 검색을 원한다면:"
echo "grep -r \"PIL.*VERSION\" backend/ 2>/dev/null || echo '검색 결과 없음'"
echo "grep -r \"Image.*VERSION\" backend/ 2>/dev/null || echo '검색 결과 없음'"
echo ""

echo "🐍 Python에서 직접 확인:"
echo "python3 -c \"import PIL; print('PIL version:', PIL.__version__)\""
echo "python3 -c \"from PIL import Image; print('Image module:', dir(Image))\""
echo ""

log_success "PIL.Image.VERSION 오류 찾기 완료!"
echo "=========================================="