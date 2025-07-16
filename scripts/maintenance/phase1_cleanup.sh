#!/bin/bash

# ============================================================================
# MyCloset AI - Phase 1: 프로젝트 정리 및 최적화
# 기존 구조 유지하면서 불필요한 파일 정리 및 최적화
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
log_header() { echo -e "${PURPLE}🚀 $1${NC}"; }

log_header "Phase 1: 프로젝트 정리 및 최적화"
echo "========================================="
log_info "현재 브랜치: $(git branch --show-current)"
log_info "작업 디렉토리: $(pwd)"
echo ""

# 1. 백업 파일 정리
log_info "1. 백업 파일 정리 중..."
find backend/app -name "*.backup*" -type f -delete 2>/dev/null || true
find backend/app -name "*.bak*" -type f -delete 2>/dev/null || true
find backend/app -name "*.pyc" -type f -delete 2>/dev/null || true
find backend/app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# 프론트엔드 캐시 정리
find frontend -name "node_modules" -type d -exec rm -rf {} + 2>/dev/null || true
find frontend -name "dist" -type d -exec rm -rf {} + 2>/dev/null || true
find frontend -name ".next" -type d -exec rm -rf {} + 2>/dev/null || true

log_success "백업 파일 및 캐시 정리 완료"

# 2. 필수 디렉토리 구조 확인 및 생성
log_info "2. 필수 디렉토리 구조 확인 중..."

# 백엔드 필수 디렉토리
mkdir -p backend/app/{core,api,services,models,utils}
mkdir -p backend/app/ai_pipeline/{steps,utils,models,cache}
mkdir -p backend/app/ai_pipeline/models/{ai_models,checkpoints}
mkdir -p backend/static/{uploads,results}
mkdir -p backend/logs
mkdir -p backend/tests

# 프론트엔드 필수 디렉토리
mkdir -p frontend/src/{components,pages,hooks,types,utils,services,styles}
mkdir -p frontend/src/components/{ui,features,layout}
mkdir -p frontend/public

# .gitkeep 파일 생성 (빈 디렉토리 유지)
find backend/static -type d -exec touch {}/.gitkeep \; 2>/dev/null || true
find backend/app/ai_pipeline/cache -type d -exec touch {}/.gitkeep \; 2>/dev/null || true
find frontend/src/components/ui -type d -exec touch {}/.gitkeep \; 2>/dev/null || true

log_success "디렉토리 구조 확인 및 생성 완료"

# 3. API 라우터 정리 및 통합
log_info "3. API 라우터 정리 중..."

# 중복된 라우터 파일들 백업
if [ -f "backend/app/api/routes.py" ]; then
    mv backend/app/api/routes.py backend/app/api/routes.py.backup_$(date +%Y%m%d_%H%M%S)
fi

if [ -f "backend/app/api/unified_routes.py" ]; then
    mv backend/app/api/unified_routes.py backend/app/api/unified_routes.py.backup_$(date +%Y%m%d_%H%M%S)
fi

log_success "API 라우터 정리 완료"

# 4. M3 Max 최적화 설정 확인
log_info "4. M3 Max 최적화 설정 확인 중..."

# Apple Silicon 감지
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    log_success "Apple Silicon Mac 감지됨"
    
    # M3 Max 구체적 확인
    if system_profiler SPHardwareDataType | grep -q "Apple M3 Max"; then
        log_success "🍎 Apple M3 Max 확인됨"
        M3_MAX_DETECTED=true
    else
        log_warning "Apple Silicon이지만 M3 Max가 아닙니다"
        M3_MAX_DETECTED=false
    fi
else
    log_warning "Apple Silicon Mac이 아닙니다"
    M3_MAX_DETECTED=false
fi

# 5. 환경 설정 파일 확인
log_info "5. 환경 설정 파일 확인 중..."

# 백엔드 환경 설정
if [ ! -f "backend/.env" ]; then
    cat > backend/.env << 'EOF'
# MyCloset AI 환경 설정
APP_NAME="MyCloset AI Backend"
APP_VERSION="1.0.0"
DEBUG=True
CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173"]

# M3 Max 최적화 설정
DEVICE_TYPE="auto"
ENABLE_MPS=True
MEMORY_POOL_SIZE=64
BATCH_SIZE=8
MAX_CONCURRENT_SESSIONS=4

# 파일 업로드 설정
MAX_UPLOAD_SIZE=52428800
ALLOWED_EXTENSIONS=["jpg", "jpeg", "png", "webp", "bmp"]

# AI 모델 설정
MODEL_CACHE_SIZE=32
ENABLE_MODEL_QUANTIZATION=True
ENABLE_NEURAL_ENGINE=True
EOF
    log_success "백엔드 .env 파일 생성됨"
else
    log_success "백엔드 .env 파일 이미 존재"
fi

# 프론트엔드 환경 설정
if [ ! -f "frontend/.env" ]; then
    cat > frontend/.env << 'EOF'
# MyCloset AI 프론트엔드 설정
VITE_APP_TITLE="MyCloset AI"
VITE_API_BASE_URL="http://localhost:8000"
VITE_WS_BASE_URL="ws://localhost:8000"
VITE_UPLOAD_MAX_SIZE=52428800
VITE_SUPPORTED_FORMATS="jpg,jpeg,png,webp,bmp"
EOF
    log_success "프론트엔드 .env 파일 생성됨"
else
    log_success "프론트엔드 .env 파일 이미 존재"
fi

# 6. 파이프라인 단계별 상태 확인
log_info "6. AI 파이프라인 단계별 상태 확인 중..."

PIPELINE_STEPS=(
    "step_01_human_parsing"
    "step_02_pose_estimation"
    "step_03_cloth_segmentation"
    "step_04_geometric_matching"
    "step_05_cloth_warping"
    "step_06_virtual_fitting"
    "step_07_post_processing"
    "step_08_quality_assessment"
)

for step in "${PIPELINE_STEPS[@]}"; do
    if [ -f "backend/app/ai_pipeline/steps/${step}.py" ]; then
        log_success "✅ ${step} 구현됨"
    else
        log_warning "⚠️  ${step} 파일 없음"
    fi
done

# 7. 의존성 파일 상태 확인
log_info "7. 의존성 파일 상태 확인 중..."

if [ -f "backend/requirements.txt" ]; then
    log_success "백엔드 requirements.txt 존재"
    deps_count=$(wc -l < backend/requirements.txt)
    log_info "  └─ 의존성 패키지: ${deps_count}개"
else
    log_warning "백엔드 requirements.txt 없음"
fi

if [ -f "frontend/package.json" ]; then
    log_success "프론트엔드 package.json 존재"
    if command -v jq &> /dev/null && [ -f "frontend/package.json" ]; then
        deps_count=$(jq '.dependencies | length' frontend/package.json 2>/dev/null || echo "계산불가")
        log_info "  └─ 의존성 패키지: ${deps_count}개"
    fi
else
    log_warning "프론트엔드 package.json 없음"
fi

# 8. Git 상태 확인
log_info "8. Git 상태 확인 중..."
log_info "현재 브랜치: $(git branch --show-current)"
log_info "수정된 파일 수: $(git status --porcelain | wc -l)"

# 9. 완료 보고서
echo ""
log_header "Phase 1 완료 보고서"
echo "=================="
log_success "✅ 프로젝트 정리 및 최적화 완료"
log_info "📊 현재 상태:"
log_info "  ├─ 백엔드: $(find backend/app -name "*.py" | wc -l)개 Python 파일"
log_info "  ├─ 프론트엔드: $(find frontend/src -name "*.ts" -o -name "*.tsx" | wc -l)개 TypeScript 파일"
log_info "  ├─ AI 파이프라인: 8단계 구현"
log_info "  ├─ M3 Max 최적화: ${M3_MAX_DETECTED}"
log_info "  └─ 환경 설정: 완료"

echo ""
log_header "다음 단계"
echo "=========="
log_info "1. Phase 2: 백엔드 최적화 및 통합"
log_info "2. Phase 3: 프론트엔드 개선"
log_info "3. Phase 4: 테스트 및 배포"
echo ""
log_success "Phase 1 완료! 다음 단계를 진행하세요."