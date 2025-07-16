#!/bin/bash

# ============================================================================
# 현재 상태 점검 스크립트 - 무엇을 먼저 수정해야 할지 파악
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

# 문제점 수집
ISSUES=()
PRIORITIES=()

log_header "MyCloset AI - 현재 상태 점검"
echo "================================="
log_info "현재 디렉토리: $(pwd)"
log_info "현재 브랜치: $(git branch --show-current)"
echo ""

# 1. 백엔드 서버 실행 가능성 테스트
log_info "1. 백엔드 서버 실행 가능성 테스트"
echo "================================="

cd backend

# 가상환경 확인
if [[ -d "venv" ]]; then
    log_success "가상환경 존재"
    source venv/bin/activate
    
    # main.py 실행 테스트
    log_info "main.py 실행 테스트 중..."
    if timeout 5 python app/main.py --help &>/dev/null; then
        log_success "main.py 기본 실행 가능"
    else
        log_error "main.py 실행 불가"
        ISSUES+=("main.py 실행 불가")
        PRIORITIES+=("HIGH")
    fi
    
    # 필수 모듈 import 테스트
    log_info "필수 모듈 import 테스트 중..."
    
    # FastAPI 테스트
    if python -c "from fastapi import FastAPI; print('FastAPI OK')" 2>/dev/null; then
        log_success "FastAPI 사용 가능"
    else
        log_error "FastAPI 사용 불가"
        ISSUES+=("FastAPI 설치 필요")
        PRIORITIES+=("HIGH")
    fi
    
    # 설정 파일 테스트
    if python -c "from app.core.config import get_settings; print('Config OK')" 2>/dev/null; then
        log_success "설정 파일 로드 가능"
    else
        log_warning "설정 파일 로드 불가"
        ISSUES+=("설정 파일 수정 필요")
        PRIORITIES+=("MEDIUM")
    fi
    
    # API 라우터 테스트
    if python -c "from app.api.step_routes import router; print('Step routes OK')" 2>/dev/null; then
        log_success "Step routes 사용 가능"
    else
        log_warning "Step routes 수정 필요"
        ISSUES+=("Step routes 수정 필요")
        PRIORITIES+=("MEDIUM")
    fi
    
    # AI 파이프라인 테스트
    if python -c "from app.ai_pipeline.pipeline_manager import PipelineManager; print('Pipeline OK')" 2>/dev/null; then
        log_success "AI 파이프라인 사용 가능"
    else
        log_warning "AI 파이프라인 수정 필요"
        ISSUES+=("AI 파이프라인 수정 필요")
        PRIORITIES+=("LOW")
    fi
    
else
    log_error "가상환경 없음"
    ISSUES+=("가상환경 생성 필요")
    PRIORITIES+=("HIGH")
fi

cd ..

# 2. 프론트엔드 상태 확인
log_info "2. 프론트엔드 상태 확인"
echo "========================"

cd frontend

# package.json 확인
if [[ -f "package.json" ]]; then
    log_success "package.json 존재"
    
    # node_modules 확인
    if [[ -d "node_modules" ]]; then
        log_success "node_modules 존재"
        
        # 빌드 테스트
        if npm run build --dry-run &>/dev/null; then
            log_success "빌드 설정 정상"
        else
            log_warning "빌드 설정 점검 필요"
            ISSUES+=("프론트엔드 빌드 설정 점검")
            PRIORITIES+=("LOW")
        fi
    else
        log_warning "node_modules 없음"
        ISSUES+=("npm install 필요")
        PRIORITIES+=("MEDIUM")
    fi
else
    log_error "package.json 없음"
    ISSUES+=("프론트엔드 설정 필요")
    PRIORITIES+=("HIGH")
fi

cd ..

# 3. 중요 파일들 상태 확인
log_info "3. 중요 파일들 상태 확인"
echo "========================"

CRITICAL_FILES=(
    "backend/app/main.py:백엔드 메인 파일"
    "backend/app/core/config.py:설정 파일"
    "backend/app/api/step_routes.py:API 라우터"
    "backend/requirements.txt:Python 의존성"
    "frontend/package.json:프론트엔드 의존성"
    "frontend/src/App.tsx:프론트엔드 메인"
)

for file_info in "${CRITICAL_FILES[@]}"; do
    IFS=':' read -r file_path description <<< "$file_info"
    
    if [[ -f "$file_path" ]]; then
        # 파일 크기 확인
        file_size=$(stat -f%z "$file_path" 2>/dev/null || stat -c%s "$file_path" 2>/dev/null || echo "0")
        if [[ $file_size -gt 100 ]]; then
            log_success "$description (${file_size} bytes)"
        else
            log_warning "$description (파일이 너무 작음: ${file_size} bytes)"
            ISSUES+=("$description 내용 점검 필요")
            PRIORITIES+=("MEDIUM")
        fi
    else
        log_error "$description 없음"
        ISSUES+=("$description 생성 필요")
        PRIORITIES+=("HIGH")
    fi
done

# 4. AI 모델 상태 확인
log_info "4. AI 모델 상태 확인"
echo "===================="

AI_MODEL_PATHS=(
    "backend/app/ai_pipeline/models/ai_models"
    "backend/app/ai_pipeline/models/checkpoints"
    "backend/app/models/ai_models"
)

AI_MODELS_FOUND=false
for model_path in "${AI_MODEL_PATHS[@]}"; do
    if [[ -d "$model_path" ]]; then
        model_count=$(find "$model_path" -type f -name "*.py" -o -name "*.pth" -o -name "*.onnx" 2>/dev/null | wc -l)
        if [[ $model_count -gt 0 ]]; then
            log_success "AI 모델 파일 발견: $model_path ($model_count 개)"
            AI_MODELS_FOUND=true
        else
            log_warning "AI 모델 디렉토리 비어있음: $model_path"
        fi
    fi
done

if [[ "$AI_MODELS_FOUND" == false ]]; then
    log_warning "AI 모델 파일 없음"
    ISSUES+=("AI 모델 다운로드 필요")
    PRIORITIES+=("LOW")
fi

# 5. 환경 설정 확인
log_info "5. 환경 설정 확인"
echo "================"

# .env 파일 확인
if [[ -f "backend/.env" ]]; then
    log_success "백엔드 .env 파일 존재"
else
    log_warning "백엔드 .env 파일 없음"
    ISSUES+=("백엔드 .env 파일 생성 필요")
    PRIORITIES+=("MEDIUM")
fi

if [[ -f "frontend/.env" ]]; then
    log_success "프론트엔드 .env 파일 존재"
else
    log_warning "프론트엔드 .env 파일 없음"
    ISSUES+=("프론트엔드 .env 파일 생성 필요")
    PRIORITIES+=("MEDIUM")
fi

# 6. 결과 요약 및 우선순위 제시
echo ""
log_header "문제점 요약 및 수정 우선순위"
echo "=============================="

if [[ ${#ISSUES[@]} -eq 0 ]]; then
    log_success "🎉 모든 검사 통과! 시스템이 정상 상태입니다."
else
    log_warning "발견된 문제점: ${#ISSUES[@]}개"
    echo ""
    
    # 우선순위별 분류
    HIGH_PRIORITY=()
    MEDIUM_PRIORITY=()
    LOW_PRIORITY=()
    
    for i in "${!ISSUES[@]}"; do
        case "${PRIORITIES[$i]}" in
            "HIGH")
                HIGH_PRIORITY+=("${ISSUES[$i]}")
                ;;
            "MEDIUM")
                MEDIUM_PRIORITY+=("${ISSUES[$i]}")
                ;;
            "LOW")
                LOW_PRIORITY+=("${ISSUES[$i]}")
                ;;
        esac
    done
    
    # 높은 우선순위 (즉시 수정 필요)
    if [[ ${#HIGH_PRIORITY[@]} -gt 0 ]]; then
        log_error "🔥 즉시 수정 필요 (HIGH PRIORITY):"
        for issue in "${HIGH_PRIORITY[@]}"; do
            echo "   - $issue"
        done
        echo ""
    fi
    
    # 중간 우선순위 (곧 수정 필요)
    if [[ ${#MEDIUM_PRIORITY[@]} -gt 0 ]]; then
        log_warning "⚠️ 곧 수정 필요 (MEDIUM PRIORITY):"
        for issue in "${MEDIUM_PRIORITY[@]}"; do
            echo "   - $issue"
        done
        echo ""
    fi
    
    # 낮은 우선순위 (나중에 수정)
    if [[ ${#LOW_PRIORITY[@]} -gt 0 ]]; then
        log_info "📋 나중에 수정 (LOW PRIORITY):"
        for issue in "${LOW_PRIORITY[@]}"; do
            echo "   - $issue"
        done
        echo ""
    fi
fi

# 7. 다음 단계 제안
log_header "추천 수정 순서"
echo "=============="

if [[ ${#HIGH_PRIORITY[@]} -gt 0 ]]; then
    echo "1️⃣ 즉시 수정 (HIGH PRIORITY):"
    echo "   bash fix_high_priority.sh"
    echo ""
fi

if [[ ${#MEDIUM_PRIORITY[@]} -gt 0 ]]; then
    echo "2️⃣ 시스템 안정화 (MEDIUM PRIORITY):"
    echo "   bash fix_medium_priority.sh"
    echo ""
fi

if [[ ${#LOW_PRIORITY[@]} -gt 0 ]]; then
    echo "3️⃣ 기능 확장 (LOW PRIORITY):"
    echo "   bash fix_low_priority.sh"
    echo ""
fi

echo "4️⃣ 전체 테스트 및 최적화:"
echo "   bash final_optimization.sh"
echo ""

log_success "상태 점검 완료! 위 순서대로 진행하시면 됩니다."