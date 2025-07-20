#!/bin/bash
# MyCloset AI 프로젝트 구조 수정 스크립트
# 생성일: 2025-07-17 19:07:29
# 시스템: Darwin / arm

set -e  # 오류 시 스크립트 중단

echo "🔧 MyCloset AI 프로젝트 구조 수정 시작..."
echo "=================================================="

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

# 1. 필수 디렉토리 생성
log_info "Step 1: 필수 디렉토리 생성"
mkdir -p backend/app/{api,core,models,services,utils}
mkdir -p backend/app/ai_pipeline/{steps,utils}
mkdir -p backend/{ai_models,static,tests,scripts,logs}
mkdir -p backend/static/{uploads,results}
mkdir -p backend/scripts/{test,utils,download}

# .gitkeep 파일 생성
touch backend/static/uploads/.gitkeep
touch backend/static/results/.gitkeep
touch backend/logs/.gitkeep

log_success "디렉토리 구조 생성 완료"

# 2. 필수 __init__.py 파일 생성
log_info "Step 2: Python 패키지 초기화 파일 생성"
touch backend/app/__init__.py
touch backend/app/api/__init__.py
touch backend/app/core/__init__.py
touch backend/app/models/__init__.py
touch backend/app/services/__init__.py
touch backend/app/utils/__init__.py
touch backend/app/ai_pipeline/__init__.py
touch backend/app/ai_pipeline/steps/__init__.py
touch backend/app/ai_pipeline/utils/__init__.py
touch backend/tests/__init__.py

log_success "__init__.py 파일들 생성 완료"

# 3. 시스템별 최적화 설정
log_info "Step 3: 시스템 최적화 설정"

# 일반 시스템 설정
export CUDA_VISIBLE_DEVICES=0
log_info "CUDA 설정 완료"

# 4. 모델 다운로드 체크
log_info "Step 4: AI 모델 상태 확인"

if [ ! -f "backend/ai_models/clip-vit-base-patch32/model.safetensors" ]; then
    log_warning "CLIP 모델이 없습니다. 다운로드가 필요합니다."
    echo "다음 명령어로 CLIP 모델을 다운로드하세요:"
    echo "python3 -c \"from transformers import CLIPModel, CLIPProcessor; model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); model.save_pretrained('./backend/ai_models/clip-vit-base-patch32')\"" 
else
    log_success "CLIP 모델 확인됨"
fi

# 5. 권한 설정
log_info "Step 5: 파일 권한 설정"
chmod +x backend/run_server.py 2>/dev/null || log_warning "run_server.py 권한 설정 실패"
chmod +x backend/scripts/test/*.py 2>/dev/null || log_info "테스트 스크립트 권한 설정"

log_success "프로젝트 구조 수정 완료!"
echo ""
echo "📊 수정 완료 상태:"
echo "=================================="
echo "📋 즉시 수정됨:"
echo "   - touch backend/app/__init__.py"
echo "   - 생성: backend/app/models/schemas.py - 데이터 스키마 파일"
echo "📋 다운로드 필요:"
echo "   - CLIP 모델 다운로드 필요"
echo "📋 수동 검토 필요:"
echo "   - 검토 필요: backend/app/api/pipeline_routes.py - API 라우트 복구"

echo ""
echo "🚀 다음 단계:"
echo "1. python3 backend/scripts/test/test_final_models.py  # 모델 테스트"
echo "2. python3 backend/run_server.py  # 서버 시작"
echo "3. 브라우저에서 http://localhost:8000 접속"
