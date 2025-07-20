#!/bin/bash

# 🏗️ MyCloset AI Backend 디렉토리 완전 정리 스크립트
# 복잡하게 얽힌 backend 폴더를 깔끔하게 정리합니다

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }
log_header() { echo -e "${PURPLE}🚀 $1${NC}"; }
log_section() { echo -e "${CYAN}📂 $1${NC}"; }

log_header "MyCloset AI Backend 디렉토리 완전 정리"
echo "============================================================="
log_info "현재 디렉토리: $(pwd)"
log_info "정리 시작 시간: $(date)"
echo ""

# backend 디렉토리인지 확인
if [[ ! -d "app" ]] || [[ ! -f "requirements.txt" ]]; then
    log_error "backend 디렉토리에서 실행해주세요!"
    log_info "backend 디렉토리로 이동: cd backend"
    exit 1
fi

# 현재 상태 분석
log_info "📊 현재 backend 상태 분석..."
total_files=$(find . -type f | wc -l)
total_dirs=$(find . -type d | wc -l)
backend_size=$(du -sh . 2>/dev/null | cut -f1)

echo "   📁 총 디렉토리: $total_dirs개"
echo "   📄 총 파일: $total_files개"
echo "   💾 backend 크기: $backend_size"
echo ""

# 1. 백업 생성
log_section "1. 중요 파일 백업 생성"
BACKUP_DIR="backend_cleanup_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# 중요 설정 파일들 백업
important_files=(
    "requirements*.txt"
    "Makefile"
    "README.md"
    ".env*"
    "run_server.py"
    "main.py"
)

for file_pattern in "${important_files[@]}"; do
    for file in $file_pattern; do
        if [[ -f "$file" ]]; then
            cp "$file" "$BACKUP_DIR/"
            log_info "백업: $file"
        fi
    done
done

log_success "백업 생성 완료: $BACKUP_DIR"

# 2. 임시 파일 및 캐시 정리
log_section "2. 임시 파일 및 캐시 정리"

# Python 캐시 정리
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true
find . -name "*.pyd" -delete 2>/dev/null || true

# 백업 파일 정리
find . -name "*.backup*" -delete 2>/dev/null || true
find . -name "*.bak" -delete 2>/dev/null || true
find . -name "*~" -delete 2>/dev/null || true

# 시스템 파일 정리
find . -name ".DS_Store" -delete 2>/dev/null || true
find . -name "Thumbs.db" -delete 2>/dev/null || true

log_success "임시 파일 정리 완료"

# 3. 스크립트 파일 정리
log_section "3. 스크립트 파일 정리"

# scripts 디렉토리 구조 생성
mkdir -p scripts/{setup,dev,maintenance,models,utils,tests}

# 스크립트 분류 및 이동
script_categories=(
    # 설정/설치 관련
    "setup:*setup*.py,*install*.py,*setup*.sh,*install*.sh"
    # 개발/테스트 관련  
    "dev:*test*.py,*debug*.py,*check*.py,*verify*.py"
    # 유지보수 관련
    "maintenance:*fix*.py,*fix*.sh,*patch*.py,*emergency*.py"
    # 모델 관련
    "models:*model*.py,*scanner*.py,*detector*.py,*relocate*.py"
    # 유틸리티
    "utils:*util*.py,*helper*.py,*tool*.py"
    # 테스트
    "tests:test_*.py,*_test.py"
)

for category_info in "${script_categories[@]}"; do
    IFS=':' read -r category patterns <<< "$category_info"
    
    IFS=',' read -ra pattern_array <<< "$patterns"
    for pattern in "${pattern_array[@]}"; do
        for file in $pattern; do
            if [[ -f "$file" ]]; then
                mv "$file" "scripts/$category/" 2>/dev/null || true
                log_info "스크립트 이동: $file -> scripts/$category/"
            fi
        done
    done
done

log_success "스크립트 파일 정리 완료"

# 4. 로그 파일 정리
log_section "4. 로그 파일 정리"

# logs 디렉토리 구조 생성
mkdir -p logs/{pipeline,models,server,tests,optimization}

# 로그 파일 분류 및 이동
log_patterns=(
    "pipeline:pipeline*.log,*pipeline*.log"
    "models:*model*.log,*detection*.log,*scanner*.log"
    "server:*server*.log,mycloset*.log"
    "tests:test*.log,*test*.log"
    "optimization:*optimization*.log,*report*.log"
)

for log_info in "${log_patterns[@]}"; do
    IFS=':' read -r log_category log_patterns <<< "$log_info"
    
    IFS=',' read -ra pattern_array <<< "$log_patterns"
    for pattern in "${pattern_array[@]}"; do
        for file in $pattern; do
            if [[ -f "$file" ]]; then
                mv "$file" "logs/$log_category/" 2>/dev/null || true
                log_info "로그 이동: $file -> logs/$log_category/"
            fi
        done
    done
done

# 일반 로그 파일들
for file in *.log; do
    if [[ -f "$file" ]]; then
        mv "$file" "logs/" 2>/dev/null || true
        log_info "로그 이동: $file -> logs/"
    fi
done

log_success "로그 파일 정리 완료"

# 5. 리포트 및 분석 파일 정리
log_section "5. 리포트 및 분석 파일 정리"

# reports 디렉토리 구조 생성
mkdir -p reports/{analysis,optimization,testing,model_detection}

# 리포트 파일 분류
report_patterns=(
    "analysis:*analysis*.json,project_analysis.json"
    "optimization:*optimization*.json,*optimization*.txt"
    "testing:*test_report*.json,*test_results*.json"
    "model_detection:*detection*.json,*relocate*.json,*scanner*.json"
)

for report_info in "${report_patterns[@]}"; do
    IFS=':' read -r report_category report_patterns <<< "$report_info"
    
    IFS=',' read -ra pattern_array <<< "$report_patterns"
    for pattern in "${pattern_array[@]}"; do
        for file in $pattern; do
            if [[ -f "$file" ]]; then
                mv "$file" "reports/$report_category/" 2>/dev/null || true
                log_info "리포트 이동: $file -> reports/$report_category/"
            fi
        done
    done
done

# 일반 JSON 및 TXT 리포트
for file in *.json *.txt; do
    if [[ -f "$file" ]] && [[ "$file" != "requirements.txt" ]] && [[ "$file" != "README.txt" ]]; then
        mv "$file" "reports/" 2>/dev/null || true
        log_info "리포트 이동: $file -> reports/"
    fi
done

log_success "리포트 파일 정리 완료"

# 6. 데이터베이스 파일 정리
log_section "6. 데이터베이스 파일 정리"

# cache 디렉토리에 데이터베이스 파일들 이동
mkdir -p cache/databases

for file in *.db; do
    if [[ -f "$file" ]]; then
        mv "$file" "cache/databases/" 2>/dev/null || true
        log_info "DB 이동: $file -> cache/databases/"
    fi
done

log_success "데이터베이스 파일 정리 완료"

# 7. 테스트 이미지 및 에셋 정리
log_section "7. 테스트 이미지 및 에셋 정리"

# static 구조 확인 및 정리
mkdir -p static/{uploads,results,test_images,assets}

# 테스트 이미지들 이동
test_image_patterns=(
    "test_*.png"
    "test_*.jpg"
    "test_*.jpeg"
    "*_test.png"
    "*_test.jpg"
)

for pattern in "${test_image_patterns[@]}"; do
    for file in $pattern; do
        if [[ -f "$file" ]]; then
            mv "$file" "static/test_images/" 2>/dev/null || true
            log_info "테스트 이미지 이동: $file -> static/test_images/"
        fi
    done
done

# 모델 파일들 정리 (개별 파일들)
model_file_patterns=(
    "*.pth"
    "*.pt"
    "*.bin"
    "*.safetensors"
    "*.ckpt"
    "*.h5"
)

mkdir -p ai_models/individual_models

for pattern in "${model_file_patterns[@]}"; do
    for file in $pattern; do
        if [[ -f "$file" ]]; then
            mv "$file" "ai_models/individual_models/" 2>/dev/null || true
            log_info "모델 파일 이동: $file -> ai_models/individual_models/"
        fi
    done
done

log_success "에셋 파일 정리 완료"

# 8. 백업 디렉토리 정리
log_section "8. 기존 백업 디렉토리 정리"

# 기존 백업들을 하나의 디렉토리로 통합
mkdir -p backups/old_backups

for backup_dir in backup_* backups_*; do
    if [[ -d "$backup_dir" ]] && [[ "$backup_dir" != "$BACKUP_DIR" ]]; then
        mv "$backup_dir" "backups/old_backups/" 2>/dev/null || true
        log_info "백업 이동: $backup_dir -> backups/old_backups/"
    fi
done

log_success "백업 디렉토리 정리 완료"

# 9. 최종 디렉토리 구조 생성 및 확인
log_section "9. 최종 디렉토리 구조 생성"

# 표준 backend 구조 생성
standard_dirs=(
    "app/api"
    "app/core" 
    "app/models"
    "app/services"
    "app/utils"
    "app/ai_pipeline/steps"
    "app/ai_pipeline/utils"
    "app/ai_pipeline/models"
    "ai_models/checkpoints"
    "ai_models/downloads"
    "ai_models/cache"
    "static/uploads"
    "static/results"
    "tests"
    "configs"
)

for dir in "${standard_dirs[@]}"; do
    if [[ ! -d "$dir" ]]; then
        mkdir -p "$dir"
        touch "$dir/.gitkeep"
        log_info "생성: $dir"
    fi
done

log_success "표준 디렉토리 구조 생성 완료"

# 10. .gitignore 업데이트
log_section "10. Backend .gitignore 업데이트"

cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so

# Virtual environments
venv/
env/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log

# Cache
cache/
.cache/

# AI Models (large files)
ai_models/individual_models/
*.pth
*.pt
*.bin
*.safetensors
*.ckpt
*.h5

# Environment variables
.env
.env.local

# Static files
static/uploads/*
static/results/*
!static/uploads/.gitkeep
!static/results/.gitkeep

# Database
*.db
*.sqlite

# Temporary
temp/
tmp/
*.tmp

# Reports and backups
reports/
backups/
EOF

log_success "Backend .gitignore 업데이트 완료"

# 11. 정리 결과 확인
log_section "11. 정리 결과 확인"

echo ""
log_header "🎉 Backend 정리 완료!"
echo "============================================================="

# 정리 후 상태
after_files=$(find . -type f | wc -l)
after_dirs=$(find . -type d | wc -l)
after_size=$(du -sh . 2>/dev/null | cut -f1)

echo ""
log_info "📊 정리 후 상태:"
echo "   📁 총 디렉토리: $after_dirs개 (이전: $total_dirs개)"
echo "   📄 총 파일: $after_files개 (이전: $total_files개)"
echo "   💾 backend 크기: $after_size (이전: $backend_size)"
echo ""

log_info "📂 새로운 backend 구조:"
echo "   backend/"
echo "   ├── app/               # 메인 애플리케이션 코드"
echo "   │   ├── api/           # API 라우터"
echo "   │   ├── core/          # 핵심 로직"
echo "   │   ├── models/        # 데이터 모델"
echo "   │   ├── services/      # 비즈니스 로직"
echo "   │   ├── utils/         # 유틸리티"
echo "   │   └── ai_pipeline/   # AI 처리 파이프라인"
echo "   ├── ai_models/         # AI 모델 파일들"
echo "   ├── static/            # 정적 파일들"
echo "   ├── scripts/           # 스크립트들 (카테고리별)"
echo "   ├── logs/              # 로그 파일들"
echo "   ├── reports/           # 리포트 파일들"
echo "   ├── cache/             # 캐시 파일들"
echo "   ├── backups/           # 백업 파일들"
echo "   └── tests/             # 테스트 코드"
echo ""

log_info "💾 백업 위치: $BACKUP_DIR"
echo ""

log_header "📋 다음 단계 권장사항:"
echo "1. git status로 변경사항 확인"
echo "2. python run_server.py (서버 테스트)"
echo "3. 앞에서 만든 모델 경로 수정 스크립트 실행"
echo "4. python -m pytest tests/ (테스트 실행)"
echo ""

log_success "✨ Backend가 깔끔하게 정리되었습니다!"
echo ""

# 12. 선택적 추가 작업
echo ""
log_warning "🔧 추가 작업을 진행하시겠습니까?"
echo "1. 모델 경로 수정 스크립트 실행"
echo "2. 서버 시작 테스트"
echo "3. 정리만 하고 종료"
echo ""
read -p "선택하세요 (1-3): " choice

case $choice in
    1)
        log_info "모델 경로 수정 스크립트 실행 중..."
        if [[ -f "../fix_model_paths.sh" ]]; then
            cd .. && ./fix_model_paths.sh && cd backend
        else
            log_warning "모델 경로 수정 스크립트를 찾을 수 없습니다."
        fi
        ;;
    2)
        log_info "서버 시작 테스트 중..."
        python run_server.py &
        SERVER_PID=$!
        sleep 5
        if kill -0 $SERVER_PID 2>/dev/null; then
            log_success "서버가 성공적으로 시작되었습니다!"
            kill $SERVER_PID
        else
            log_error "서버 시작에 실패했습니다."
        fi
        ;;
    3)
        log_info "정리 작업만 완료했습니다."
        ;;
    *)
        log_info "잘못된 선택입니다. 정리 작업만 완료했습니다."
        ;;
esac

echo ""
log_success "🎊 모든 작업이 완료되었습니다!"