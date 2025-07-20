#!/bin/bash

# 🧹 MyCloset AI 프로젝트 완전 정리 스크립트
# 복잡하게 얽힌 루트 디렉토리를 깔끔하게 정리합니다

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

log_header "MyCloset AI 프로젝트 완전 정리 시작"
echo "============================================================="
log_info "현재 디렉토리: $(pwd)"
log_info "정리 시작 시간: $(date)"
echo ""

# 현재 상태 확인
log_info "📊 현재 프로젝트 상태 분석..."
total_files=$(find . -type f | wc -l)
total_dirs=$(find . -type d | wc -l)
project_size=$(du -sh . 2>/dev/null | cut -f1)

echo "   📁 총 디렉토리: $total_dirs개"
echo "   📄 총 파일: $total_files개"
echo "   💾 프로젝트 크기: $project_size"
echo ""

# 1. 백업 생성
log_section "1. 중요 파일 백업 생성"
BACKUP_DIR="cleanup_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# 중요 설정 파일들 백업
important_files=(
    ".env"
    "Makefile" 
    "README.md"
    "requirements.txt"
    "package.json"
    ".gitignore"
    "verification_results.json"
)

for file in "${important_files[@]}"; do
    if [[ -f "$file" ]]; then
        cp "$file" "$BACKUP_DIR/"
        log_info "백업: $file"
    fi
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

# 로그 파일 정리 (크기가 큰 것들)
find . -name "*.log" -size +10M -delete 2>/dev/null || true

# 시스템 파일 정리
find . -name ".DS_Store" -delete 2>/dev/null || true
find . -name "Thumbs.db" -delete 2>/dev/null || true

log_success "임시 파일 정리 완료"

# 3. scripts 디렉토리 정리
log_section "3. scripts 디렉토리 정리"

# scripts 구조 생성
mkdir -p scripts/{setup,dev,maintenance,models,utils,run}

# 설정 관련 스크립트
setup_scripts=(
    "*setup*.sh"
    "*install*.sh" 
    "*conda*.sh"
    "complete_setup.sh"
    "setup_conda_m3.sh"
)

# 개발 관련 스크립트
dev_scripts=(
    "*check*.sh"
    "*test*.py"
    "*debug*.py"
    "*verify*.py"
    "syntax_check.py"
    "improve_verify_script.py"
)

# 유지보수 관련 스크립트
maintenance_scripts=(
    "*fix*.sh"
    "*fix*.py"
    "*cleanup*.sh"
    "*patch*.py"
    "fix_conda_env.sh"
    "fix_indent_error.py"
    "patch_modelloader.py"
)

# 모델 관련 스크립트
model_scripts=(
    "*model*.py"
    "*model*.sh"
    "*download*.py"
    "*download*.sh"
    "auto_detect.py"
    "comprehensive_finder.py"
    "consolidate*.py"
    "enhanced_model_detection_cache.db"
    "find_unet_vton.py"
)

# 실행 관련 스크립트
run_scripts=(
    "*run*.sh"
    "start_*.sh"
)

# 스크립트들 이동
move_scripts() {
    local pattern=$1
    local target_dir=$2
    
    for script in $pattern; do
        if [[ -f "$script" ]]; then
            mv "$script" "$target_dir/" 2>/dev/null || true
            log_info "이동: $script -> $target_dir/"
        fi
    done
}

for script in "${setup_scripts[@]}"; do
    move_scripts "$script" "scripts/setup"
done

for script in "${dev_scripts[@]}"; do
    move_scripts "$script" "scripts/dev"
done

for script in "${maintenance_scripts[@]}"; do
    move_scripts "$script" "scripts/maintenance"
done

for script in "${model_scripts[@]}"; do
    move_scripts "$script" "scripts/models"
done

for script in "${run_scripts[@]}"; do
    move_scripts "$script" "scripts/run"
done

# 유틸리티 스크립트
util_scripts=(
    "smart_cleanup.py"
    "project_structure_analyzer.py"
    "complete_consolidator.py"
    "integrate_virtual_fitting_v2.py"
)

for script in "${util_scripts[@]}"; do
    if [[ -f "$script" ]]; then
        mv "$script" "scripts/utils/" 2>/dev/null || true
        log_info "이동: $script -> scripts/utils/"
    fi
done

log_success "scripts 디렉토리 정리 완료"

# 4. 백업 디렉토리 정리
log_section "4. 백업 디렉토리 정리"

# 기존 백업들을 하나의 디렉토리로 통합
mkdir -p backups/old_backups

backup_dirs=(
    "backup_*"
    "backup_ai_models_*"
    "temp"
    "cache"
)

for backup_pattern in "${backup_dirs[@]}"; do
    for backup_dir in $backup_pattern; do
        if [[ -d "$backup_dir" ]] && [[ "$backup_dir" != "$BACKUP_DIR" ]]; then
            mv "$backup_dir" "backups/old_backups/" 2>/dev/null || true
            log_info "백업 이동: $backup_dir -> backups/old_backups/"
        fi
    done
done

log_success "백업 디렉토리 정리 완료"

# 5. 로그 및 리포트 정리
log_section "5. 로그 및 리포트 정리"

# logs 디렉토리 생성 및 정리
mkdir -p logs/{pipeline,models,server,cleanup}

# 로그 파일들 이동
log_files=(
    "*.log"
    "*_log.txt"
    "pipeline.log"
    "model_finder.log"
)

for log_pattern in "${log_files[@]}"; do
    for log_file in $log_pattern; do
        if [[ -f "$log_file" ]]; then
            mv "$log_file" "logs/" 2>/dev/null || true
            log_info "로그 이동: $log_file -> logs/"
        fi
    done
done

# 리포트 파일들 정리
mkdir -p reports/{cleanup,verification,analysis}

report_files=(
    "*report*.txt"
    "*report*.json"
    "verification_results.json"
    "cleanup_report_*.txt"
    "complete_consolidation_report_*.json"
    "huggingface_consolidation_report_*.json"
)

for report_pattern in "${report_files[@]}"; do
    for report_file in $report_pattern; do
        if [[ -f "$report_file" ]]; then
            mv "$report_file" "reports/" 2>/dev/null || true
            log_info "리포트 이동: $report_file -> reports/"
        fi
    done
done

log_success "로그 및 리포트 정리 완료"

# 6. AI 모델 경로 정리
log_section "6. AI 모델 경로 정리"

# 기본 ai_models 구조가 있으면 backend로 이동
if [[ -d "ai_models" ]]; then
    if [[ ! -d "backend/ai_models" ]]; then
        mv "ai_models" "backend/"
        log_info "ai_models -> backend/ai_models 이동"
    else
        log_warning "backend/ai_models가 이미 존재함. 수동 병합 필요"
    fi
fi

# huggingface_cache 정리
if [[ -d "huggingface_cache" ]]; then
    mkdir -p "backend/ai_models/"
    mv "huggingface_cache" "backend/ai_models/"
    log_info "huggingface_cache -> backend/ai_models/ 이동"
fi

log_success "AI 모델 경로 정리 완료"

# 7. 단일 파일들 정리
log_section "7. 단일 파일들 정리"

# 개별 모델 파일들
model_files=(
    "*.pth"
    "*.bin"
    "*.safetensors"
    "*.ckpt"
    "*.h5"
)

mkdir -p "backend/ai_models/individual_models"

for model_pattern in "${model_files[@]}"; do
    for model_file in $model_pattern; do
        if [[ -f "$model_file" ]]; then
            mv "$model_file" "backend/ai_models/individual_models/" 2>/dev/null || true
            log_info "모델 파일 이동: $model_file -> backend/ai_models/individual_models/"
        fi
    done
done

# 설정 파일들 configs로 이동
config_files=(
    "*.yaml"
    "*.yml"
    "*.toml"
    "*.ini"
)

mkdir -p configs

for config_pattern in "${config_files[@]}"; do
    for config_file in $config_pattern; do
        if [[ -f "$config_file" ]] && [[ "$config_file" != "pyproject.toml" ]]; then
            mv "$config_file" "configs/" 2>/dev/null || true
            log_info "설정 파일 이동: $config_file -> configs/"
        fi
    done
done

log_success "단일 파일들 정리 완료"

# 8. .gitignore 업데이트
log_section "8. .gitignore 업데이트"

if [[ -f ".gitignore" ]]; then
    # 기존 .gitignore 백업
    cp ".gitignore" "$BACKUP_DIR/.gitignore.backup"
    
    # 새로운 항목들 추가
    cat >> .gitignore << 'EOF'

# 정리된 디렉토리들
backups/
logs/
reports/
temp/
cache/

# AI 모델 파일들
backend/ai_models/individual_models/
*.pth
*.bin
*.safetensors
*.ckpt
*.h5

# 시스템 파일들
.DS_Store
Thumbs.db
*.pyc
__pycache__/

# 로그 파일들
*.log
pipeline.log

# 백업 파일들
*.backup*
*.bak
*~
EOF

    log_success ".gitignore 업데이트 완료"
fi

# 9. 최종 디렉토리 구조 생성
log_section "9. 최종 디렉토리 구조 생성"

# 필수 디렉토리들 확인/생성
mkdir -p {backend,frontend,scripts,configs,docs,backups,logs,reports}
mkdir -p backend/{app,ai_models,static,tests}
mkdir -p backend/app/{api,core,models,services,utils,ai_pipeline}
mkdir -p frontend/{src,public}
mkdir -p scripts/{setup,dev,maintenance,models,utils,run}
mkdir -p docs/{api,guides,examples}

# .gitkeep 파일 생성 (빈 디렉토리 보존)
touch backend/static/uploads/.gitkeep
touch backend/static/results/.gitkeep
touch logs/.gitkeep
touch reports/.gitkeep

log_success "최종 디렉토리 구조 생성 완료"

# 10. 정리 결과 확인
log_section "10. 정리 결과 확인"

echo ""
log_header "🎉 프로젝트 정리 완료!"
echo "============================================================="

# 정리 후 상태
after_files=$(find . -type f | wc -l)
after_dirs=$(find . -type d | wc -l)
after_size=$(du -sh . 2>/dev/null | cut -f1)

echo ""
log_info "📊 정리 후 상태:"
echo "   📁 총 디렉토리: $after_dirs개 (이전: $total_dirs개)"
echo "   📄 총 파일: $after_files개 (이전: $total_files개)"
echo "   💾 프로젝트 크기: $after_size (이전: $project_size)"
echo ""

log_info "📂 새로운 프로젝트 구조:"
echo "   mycloset-ai/"
echo "   ├── backend/           # 백엔드 소스코드"
echo "   ├── frontend/          # 프론트엔드 소스코드"
echo "   ├── scripts/           # 모든 스크립트들 (카테고리별 정리)"
echo "   ├── configs/           # 설정 파일들"
echo "   ├── docs/              # 문서"
echo "   ├── backups/           # 백업 파일들"
echo "   ├── logs/              # 로그 파일들"
echo "   └── reports/           # 리포트 파일들"
echo ""

log_info "💾 백업 위치: $BACKUP_DIR"
echo ""

log_header "📋 다음 단계 권장사항:"
echo "1. git status로 변경사항 확인"
echo "2. git add . && git commit -m '프로젝트 구조 정리'"
echo "3. conda activate mycloset-ai (또는 현재 환경)"
echo "4. cd backend && python run_server.py (서버 테스트)"
echo "5. cd frontend && npm install && npm run dev (프론트엔드 테스트)"
echo ""

log_success "✨ 이제 깔끔하게 정리된 프로젝트 구조입니다!"
echo ""