#!/bin/bash
# MyCloset AI 프로젝트 파일 분석 및 정리 스크립트
# M3 Max 128GB 최적화

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
log_header() { echo -e "${PURPLE}🚀 $1${NC}"; }

echo "🔍 MyCloset AI 프로젝트 분석 및 정리"
echo "======================================"
echo "M3 Max 128GB 최적화 | $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 1. 프로젝트 구조 분석
log_header "Phase 1: 프로젝트 구조 분석"

# 디렉토리별 크기 체크
check_directory_size() {
    local dir=$1
    if [ -d "$dir" ]; then
        local size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        local files=$(find "$dir" -type f 2>/dev/null | wc -l)
        echo "📁 $dir: $size ($files files)"
    else
        echo "❌ $dir: 존재하지 않음"
    fi
}

log_info "디렉토리별 크기 분석:"
check_directory_size "backend"
check_directory_size "backend/ai_models"
check_directory_size "backend/app"
check_directory_size "frontend"
check_directory_size "scripts"
check_directory_size "configs"

echo ""

# 2. 큰 파일들 찾기 (100MB 이상)
log_header "Phase 2: 대용량 파일 분석"

log_info "100MB 이상 파일 검색 중..."
find . -type f -size +100M 2>/dev/null | while read file; do
    size=$(du -sh "$file" | cut -f1)
    log_warning "대용량 파일: $file ($size)"
done

# AI 모델 파일들 체크
log_info "AI 모델 파일 분석:"
if [ -d "backend/ai_models" ]; then
    find backend/ai_models -name "*.pth" -o -name "*.bin" -o -name "*.safetensors" -o -name "*.ckpt" 2>/dev/null | while read model; do
        if [ -f "$model" ]; then
            size=$(du -sh "$model" | cut -f1)
            echo "🧠 AI Model: $(basename "$model") ($size)"
        fi
    done
else
    log_warning "AI 모델 디렉토리가 없습니다."
fi

echo ""

# 3. 불필요한 파일 탐지
log_header "Phase 3: 불필요한 파일 탐지"

log_info "임시 파일 및 캐시 검색:"

# Python 캐시 파일
cache_files=$(find . -name "__pycache__" -type d 2>/dev/null | wc -l)
if [ $cache_files -gt 0 ]; then
    log_warning "Python 캐시 디렉토리: $cache_files개"
fi

# .pyc 파일
pyc_files=$(find . -name "*.pyc" 2>/dev/null | wc -l)
if [ $pyc_files -gt 0 ]; then
    log_warning ".pyc 파일: $pyc_files개"
fi

# 백업 파일
backup_files=$(find . -name "*.backup*" -o -name "*.bak" -o -name "*~" 2>/dev/null | wc -l)
if [ $backup_files -gt 0 ]; then
    log_warning "백업 파일: $backup_files개"
fi

# Node.js 모듈
if [ -d "frontend/node_modules" ]; then
    node_size=$(du -sh frontend/node_modules 2>/dev/null | cut -f1)
    log_warning "Node.js 모듈: $node_size"
fi

# 가상환경들
for env in venv mycloset_env venv_py311 env; do
    if [ -d "backend/$env" ]; then
        env_size=$(du -sh "backend/$env" 2>/dev/null | cut -f1)
        log_warning "Python 가상환경 ($env): $env_size"
    fi
done

echo ""

# 4. .gitignore 분석
log_header "Phase 4: .gitignore 최적화 분석"

log_info ".gitignore 파일 체크:"
if [ -f ".gitignore" ]; then
    log_success ".gitignore 존재"
    
    # AI 모델 제외 규칙 확인
    if grep -q "ai_models" .gitignore; then
        log_success "AI 모델 제외 규칙 존재"
    else
        log_warning "AI 모델 제외 규칙 누락"
    fi
    
    # 가상환경 제외 규칙 확인
    if grep -q "venv" .gitignore; then
        log_success "가상환경 제외 규칙 존재"
    else
        log_warning "가상환경 제외 규칙 누락"
    fi
    
else
    log_error ".gitignore 파일 없음"
fi

echo ""

# 5. 자동 정리 옵션 제시
log_header "Phase 5: 정리 옵션"

echo "🛠️ 자동 정리 가능한 항목들:"
echo "1. Python 캐시 파일 (__pycache__, *.pyc)"
echo "2. 백업 파일 (*.backup, *.bak, *~)"
echo "3. Node.js 캐시 (node_modules - 재설치 가능)"
echo "4. 로그 파일 (30일 이상 된 것)"
echo "5. 임시 파일 (temp/, cache/ 등)"
echo ""

echo "⚠️ 주의 사항:"
echo "- AI 모델 파일은 수동으로 확인 후 삭제"
echo "- 가상환경은 백업 후 정리"
echo "- 중요 설정 파일은 보존"
echo ""

# 6. 권장 명령어 생성
log_header "Phase 6: 권장 정리 명령어"

cat << 'EOF'

🧹 안전한 정리 명령어들:

# 1. Python 캐시 정리
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -type f -delete 2>/dev/null || true

# 2. 백업 파일 정리  
find . -name "*.backup*" -type f -delete 2>/dev/null || true
find . -name "*.bak" -type f -delete 2>/dev/null || true
find . -name "*~" -type f -delete 2>/dev/null || true

# 3. 30일 이상 된 로그 파일 정리
find backend/logs -name "*.log" -type f -mtime +30 -delete 2>/dev/null || true

# 4. Node.js 모듈 재설치 (선택적)
# cd frontend && rm -rf node_modules && npm install

# 5. 사용하지 않는 가상환경 정리 (신중하게)
# rm -rf backend/venv_backup_*

🔧 conda 환경 정리:
conda clean --all --yes
conda env list  # 사용하지 않는 환경 확인

📊 Git 저장소 정리:
git gc --aggressive --prune=now
git remote prune origin

EOF

echo ""
log_success "분석 완료! 위 명령어를 선택적으로 실행하세요."
echo ""

# 7. 시스템 최적화 정보
log_header "Phase 7: M3 Max 최적화 권장사항"

echo "🍎 M3 Max 128GB 시스템 최적화:"
echo ""
echo "메모리 관리:"
echo "- AI 모델 배치 크기: 8-16 (128GB 활용)"
echo "- MPS 캐시 주기적 정리: torch.mps.empty_cache()"
echo "- 멀티프로세싱: 16코어 활용"
echo ""
echo "저장공간 관리:"
echo "- AI 모델은 외부 SSD 고려"
echo "- 로그 로테이션 설정"
echo "- 정기적 프로젝트 정리"
echo ""

echo "🚀 다음 단계:"
echo "1. 위 분석 결과 검토"
echo "2. 필요한 정리 명령어 실행"
echo "3. .gitignore 최적화"
echo "4. AI 모델 관리 전략 수립"
echo "5. 개발 환경 재구성"