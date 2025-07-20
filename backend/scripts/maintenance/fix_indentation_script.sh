#!/bin/bash
# 🔥 MyCloset AI 들여쓰기 오류 완전 수정 스크립트 v2.0
# 모든 Step 파일들의 IndentationError 해결

set -e

echo "🔧 MyCloset AI 들여쓰기 오류 완전 수정 시작..."
echo "=" * 60

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

# 현재 디렉토리 확인
if [ ! -d "app/ai_pipeline/steps" ]; then
    log_error "Step 디렉토리를 찾을 수 없습니다. backend 디렉토리에서 실행해주세요."
    exit 1
fi

log_info "Step 1: 백업 생성"
mkdir -p backup_steps
cp -r app/ai_pipeline/steps/*.py backup_steps/ 2>/dev/null || true
log_success "백업 완료"

# Python 스크립트로 들여쓰기 수정
log_info "Step 2: Python 스크립트로 들여쓰기 수정"

python3 << 'PYTHON_SCRIPT'
import os
import re
import sys
from pathlib import Path

def fix_indentation_in_file(file_path):
    """파일의 들여쓰기 문제 수정"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 백업 생성
        backup_path = f"{file_path}.indent_backup"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 들여쓰기 수정
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            # 빈 줄은 그대로
            if not line.strip():
                fixed_lines.append(line)
                continue
            
            # 들여쓰기 감지
            leading_spaces = len(line) - len(line.lstrip())
            
            # 잘못된 들여쓰기 수정 (홀수 공백을 4의 배수로)
            if leading_spaces % 4 != 0 and leading_spaces > 0:
                corrected_indent = (leading_spaces // 4) * 4
                if leading_spaces % 4 >= 2:
                    corrected_indent += 4
                
                new_line = ' ' * corrected_indent + line.lstrip()
                fixed_lines.append(new_line)
                print(f"  라인 {i+1}: {leading_spaces}→{corrected_indent} 공백 수정")
            else:
                fixed_lines.append(line)
        
        # 수정된 내용 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(fixed_lines))
        
        return True
        
    except Exception as e:
        print(f"❌ {file_path} 수정 실패: {e}")
        return False

def fix_specific_patterns(file_path):
    """특정 패턴의 들여쓰기 문제 수정"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 문제가 되는 특정 패턴들 수정
        patterns_to_fix = [
            # 클래스 정의 후 들여쓰기 문제
            (r'^(\s*)class\s+(\w+).*:\s*\n(\s*)([^#\s])', r'\1class \2:\n\1    \4'),
            
            # 함수 정의 후 들여쓰기 문제  
            (r'^(\s*)def\s+(\w+).*:\s*\n(\s*)([^#\s])', r'\1def \2:\n\1    \4'),
            
            # try/except 블록 들여쓰기
            (r'^(\s*)try:\s*\n(\s*)([^#\s])', r'\1try:\n\1    \3'),
            (r'^(\s*)except.*:\s*\n(\s*)([^#\s])', r'\1except:\n\1    \3'),
            
            # if/else 블록 들여쓰기
            (r'^(\s*)if\s+.*:\s*\n(\s*)([^#\s])', r'\1if:\n\1    \3'),
            (r'^(\s*)else:\s*\n(\s*)([^#\s])', r'\1else:\n\1    \3'),
        ]
        
        for pattern, replacement in patterns_to_fix:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return True
        
    except Exception as e:
        print(f"❌ {file_path} 패턴 수정 실패: {e}")
        return False

# Step 파일들 수정
step_files = [
    'app/ai_pipeline/steps/step_01_human_parsing.py',
    'app/ai_pipeline/steps/step_02_pose_estimation.py', 
    'app/ai_pipeline/steps/step_03_cloth_segmentation.py',
    'app/ai_pipeline/steps/step_04_geometric_matching.py',
    'app/ai_pipeline/steps/step_05_cloth_warping.py',
    'app/ai_pipeline/steps/step_06_virtual_fitting.py',
    'app/ai_pipeline/steps/step_07_post_processing.py',
    'app/ai_pipeline/steps/step_08_quality_assessment.py'
]

fixed_count = 0
for file_path in step_files:
    if os.path.exists(file_path):
        print(f"🔧 {file_path} 수정 중...")
        if fix_indentation_in_file(file_path):
            fix_specific_patterns(file_path)
            fixed_count += 1
            print(f"✅ {file_path} 수정 완료")
        else:
            print(f"❌ {file_path} 수정 실패")
    else:
        print(f"⚠️ {file_path} 파일을 찾을 수 없음")

print(f"\n✅ 총 {fixed_count}개 파일 수정 완료")
PYTHON_SCRIPT

log_info "Step 3: 특정 문제 파일 수동 수정"

# step_04_geometric_matching.py의 특정 문제 수정
if [ -f "app/ai_pipeline/steps/step_04_geometric_matching.py" ]; then
    log_info "step_04_geometric_matching.py 특정 문제 수정..."
    
    # GeometricMatchingMixin 클래스 들여쓰기 수정
    python3 << 'PYTHON_FIX_04'
import re

file_path = 'app/ai_pipeline/steps/step_04_geometric_matching.py'
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 문제가 되는 라인 60 근처 수정
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        if 'class GeometricMatchingMixin' in line:
            # 클래스 정의 라인의 들여쓰기를 0으로 설정
            lines[i] = line.lstrip()
            
            # 다음 라인들의 들여쓰기도 적절히 조정
            j = i + 1
            while j < len(lines) and (lines[j].strip() == '' or lines[j].startswith('    ') or lines[j].startswith('\t')):
                if lines[j].strip() != '':
                    # 클래스 내부 메서드나 속성은 4칸 들여쓰기
                    lines[j] = '    ' + lines[j].lstrip()
                j += 1
            break
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print("✅ step_04_geometric_matching.py 수정 완료")
    
except Exception as e:
    print(f"❌ step_04_geometric_matching.py 수정 실패: {e}")
PYTHON_FIX_04
fi

# step_06_virtual_fitting.py의 특정 문제 수정
if [ -f "app/ai_pipeline/steps/step_06_virtual_fitting.py" ]; then
    log_info "step_06_virtual_fitting.py 특정 문제 수정..."
    
    python3 << 'PYTHON_FIX_06'
import re

file_path = 'app/ai_pipeline/steps/step_06_virtual_fitting.py'
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 문제가 되는 라인 134 근처 수정
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        if 'def ensure_step_initialization' in line:
            # 함수 정의 라인의 들여쓰기를 0으로 설정
            lines[i] = line.lstrip()
            
            # 다음 라인들의 들여쓰기도 적절히 조정
            j = i + 1
            while j < len(lines) and (lines[j].strip() == '' or lines[j].startswith('    ') or lines[j].startswith('\t')):
                if lines[j].strip() != '':
                    # 함수 내부는 4칸 들여쓰기
                    lines[j] = '    ' + lines[j].lstrip()
                j += 1
            break
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print("✅ step_06_virtual_fitting.py 수정 완료")
    
except Exception as e:
    print(f"❌ step_06_virtual_fitting.py 수정 실패: {e}")
PYTHON_FIX_06
fi

log_info "Step 4: 구문 검사"

# 모든 Step 파일들의 구문 검사
error_count=0
for file in app/ai_pipeline/steps/step_*.py; do
    if [ -f "$file" ]; then
        log_info "구문 검사: $file"
        if python3 -m py_compile "$file" 2>/dev/null; then
            log_success "✅ $file 구문 정상"
        else
            log_error "❌ $file 구문 오류"
            error_count=$((error_count + 1))
        fi
    fi
done

log_info "Step 5: 권한 설정"
chmod +x app/ai_pipeline/steps/*.py

if [ $error_count -eq 0 ]; then
    log_success "🎉 모든 들여쓰기 오류 수정 완료!"
    echo ""
    echo "✅ 수정 완료 사항:"
    echo "   - 모든 Step 파일 들여쓰기 정규화"
    echo "   - 클래스/함수 정의 들여쓰기 수정"
    echo "   - 특정 문제 파일 수동 수정"
    echo "   - 구문 검사 통과"
    echo ""
    echo "🚀 다음 단계:"
    echo "   python app/main.py"
else
    log_warning "⚠️ $error_count 개 파일에 여전히 문제가 있습니다."
    echo ""
    echo "🔧 수동 확인이 필요한 파일들:"
    for file in app/ai_pipeline/steps/step_*.py; do
        if [ -f "$file" ]; then
            if ! python3 -m py_compile "$file" 2>/dev/null; then
                echo "   - $file"
            fi
        fi
    done
fi

echo ""
echo "📁 백업 파일 위치: backup_steps/"
echo "🔄 복원 명령어: cp backup_steps/*.py app/ai_pipeline/steps/"