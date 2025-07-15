#!/bin/bash

# ==============================================================================
# MyCloset AI - 순수 상태 체크 스크립트 (읽기 전용)
# 아무것도 삭제하거나 수정하지 않고 현재 상태만 체크
# ==============================================================================

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 로그 함수들
log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }
log_header() { echo -e "${PURPLE}🚀 $1${NC}"; }
log_section() { echo -e "${CYAN}📂 $1${NC}"; }

# 체크 결과 저장
MISSING_PACKAGES=()
MISSING_MODELS=()
MISSING_FILES=()
WARNINGS=()
RECOMMENDATIONS=()

PROJECT_ROOT=$(pwd)
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
AI_MODELS_DIR="$BACKEND_DIR/ai_models"

log_header "MyCloset AI - 현재 상태 체크 (읽기 전용)"
echo "=================================================================="
log_info "프로젝트 루트: $PROJECT_ROOT"
log_info "체크 시작 시간: $(date)"
log_warning "⚠️  이 스크립트는 아무것도 수정하지 않습니다 (읽기 전용)"
echo ""

# ==============================================================================
# 1. 시스템 환경 체크
# ==============================================================================

log_section "Step 1: 시스템 환경 체크"

# 운영체제 및 하드웨어 정보
OS_TYPE=$(uname -s)
ARCH_TYPE=$(uname -m)
log_info "운영체제: $OS_TYPE ($ARCH_TYPE)"

# M3 Max 감지
if [[ "$OS_TYPE" == "Darwin" && "$ARCH_TYPE" == "arm64" ]]; then
    # M3 Max 구체적 감지
    CHIP_INFO=$(system_profiler SPHardwareDataType | grep "Chip" | head -1)
    if echo "$CHIP_INFO" | grep -q "Apple M3 Max"; then
        log_success "🍎 Apple M3 Max 감지됨"
        log_info "  └─ 칩: $(echo $CHIP_INFO | cut -d':' -f2 | xargs)"
        
        # 메모리 정보
        MEMORY_GB=$(system_profiler SPHardwareDataType | grep "Memory" | awk '{print $2}')
        log_info "  └─ 메모리: ${MEMORY_GB}GB"
        
        if [[ "$MEMORY_GB" == "128" ]]; then
            log_success "  └─ 최대 메모리 구성 (128GB) ✨"
        fi
        
        GPU_TYPE="mps"
    else
        log_success "Apple Silicon Mac 감지됨"
        GPU_TYPE="mps"
    fi
elif command -v nvidia-smi &> /dev/null; then
    log_success "NVIDIA GPU 감지됨"
    GPU_TYPE="cuda"
    # GPU 정보 출력
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while read line; do
        log_info "  └─ GPU: $line"
    done
else
    log_warning "GPU 가속 없음 - CPU 모드"
    GPU_TYPE="cpu"
fi

echo ""

# ==============================================================================
# 2. Python 환경 체크
# ==============================================================================

log_section "Step 2: Python 환경 체크"

# Python 버전
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    log_success "Python 설치됨: $PYTHON_VERSION"
else
    log_error "Python3 미설치"
    MISSING_PACKAGES+=("python3")
fi

# 가상환경 체크
if [[ -d "$BACKEND_DIR/mycloset_env" ]]; then
    log_success "Python 가상환경 존재: mycloset_env"
    
    # 가상환경이 활성화되었는지 체크
    if [[ "$VIRTUAL_ENV" == *"mycloset_env"* ]]; then
        log_success "가상환경 활성화됨"
    else
        log_warning "가상환경이 활성화되지 않음"
        WARNINGS+=("가상환경 미활성화")
    fi
elif [[ -d "$BACKEND_DIR/mycloset-ai" ]]; then
    log_success "Python 가상환경 존재: mycloset-ai"
else
    log_warning "Python 가상환경 없음"
    RECOMMENDATIONS+=("Python 가상환경 생성 권장")
fi

echo ""

# ==============================================================================
# 3. 백엔드 패키지 체크 (현재 환경에서)
# ==============================================================================

log_section "Step 3: 백엔드 패키지 체크"

# 핵심 패키지들 체크
CORE_PACKAGES=(
    "fastapi:FastAPI 웹 프레임워크"
    "uvicorn:ASGI 서버"
    "torch:PyTorch 머신러닝"
    "torchvision:PyTorch 비전"
    "transformers:Hugging Face Transformers"
    "diffusers:Stable Diffusion"
    "opencv-python:OpenCV 이미지 처리"
    "pillow:PIL 이미지 라이브러리"
    "numpy:수치 연산"
    "pandas:데이터 처리"
)

log_info "핵심 패키지 상태:"
for package_info in "${CORE_PACKAGES[@]}"; do
    IFS=':' read -r package_name description <<< "$package_info"
    
    if python3 -c "import $package_name" 2>/dev/null; then
        # 버전 정보 가져오기
        version=$(python3 -c "import $package_name; print(getattr($package_name, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
        log_success "$description ($package_name==$version)"
    else
        log_error "$description ($package_name) - 미설치"
        MISSING_PACKAGES+=("$package_name")
    fi
done

echo ""

# 로그에서 언급된 누락 패키지들 체크
MISSING_FROM_LOG=(
    "rembg:배경 제거 라이브러리"
    "onnxruntime:ONNX 런타임"
    "scikit-image:이미지 처리 확장"
    "mediapipe:Google MediaPipe"
)

log_info "로그에서 감지된 누락 패키지들:"
for package_info in "${MISSING_FROM_LOG[@]}"; do
    IFS=':' read -r package_name description <<< "$package_info"
    
    if python3 -c "import $package_name" 2>/dev/null; then
        version=$(python3 -c "import $package_name; print(getattr($package_name, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
        log_success "$description ($package_name==$version)"
    else
        log_warning "$description ($package_name) - 권장 설치"
        RECOMMENDATIONS+=("pip install $package_name")
    fi
done

echo ""

# ==============================================================================
# 4. 프론트엔드 환경 체크
# ==============================================================================

log_section "Step 4: 프론트엔드 환경 체크"

# Node.js 체크
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    log_success "Node.js 설치됨: $NODE_VERSION"
else
    log_error "Node.js 미설치"
    MISSING_PACKAGES+=("node.js")
fi

# 프론트엔드 프로젝트 체크
if [[ -f "$FRONTEND_DIR/package.json" ]]; then
    log_success "package.json 존재"
    
    # 의존성 설치 상태
    if [[ -d "$FRONTEND_DIR/node_modules" ]]; then
        log_success "node_modules 설치됨"
        
        # 패키지 개수 확인
        package_count=$(find "$FRONTEND_DIR/node_modules" -maxdepth 1 -type d | wc -l)
        log_info "  └─ 설치된 패키지: 약 $package_count개"
    else
        log_warning "node_modules 미설치"
        RECOMMENDATIONS+=("cd frontend && npm install")
    fi
else
    log_warning "package.json 없음"
    RECOMMENDATIONS+=("프론트엔드 프로젝트 초기화 필요")
fi

echo ""

# ==============================================================================
# 5. AI 모델 상태 체크
# ==============================================================================

log_section "Step 5: AI 모델 상태 체크"

# AI 모델 디렉토리 구조
AI_MODEL_DIRS=(
    "$AI_MODELS_DIR:AI 모델 루트"
    "$AI_MODELS_DIR/checkpoints:체크포인트"
    "$AI_MODELS_DIR/OOTDiffusion:OOT-Diffusion"
    "$AI_MODELS_DIR/VITON-HD:VITON-HD"
)

log_info "AI 모델 디렉토리 상태:"
for dir_info in "${AI_MODEL_DIRS[@]}"; do
    IFS=':' read -r dir_path description <<< "$dir_info"
    
    if [[ -d "$dir_path" ]]; then
        file_count=$(find "$dir_path" -type f 2>/dev/null | wc -l)
        if [[ $file_count -gt 0 ]]; then
            # 디렉토리 크기 계산
            size=$(du -sh "$dir_path" 2>/dev/null | cut -f1)
            log_success "$description - $file_count개 파일 (크기: $size)"
        else
            log_warning "$description - 빈 디렉토리"
            MISSING_MODELS+=("$description")
        fi
    else
        log_error "$description - 디렉토리 없음"
        MISSING_MODELS+=("$description")
    fi
done

echo ""

# 큰 모델 파일들 체크
log_info "대용량 모델 파일 검색 중..."
large_files=$(find "$AI_MODELS_DIR" -type f -size +100M 2>/dev/null | head -10)

if [[ -n "$large_files" ]]; then
    log_success "대용량 모델 파일 발견:"
    echo "$large_files" | while read file; do
        size=$(du -sh "$file" 2>/dev/null | cut -f1)
        filename=$(basename "$file")
        log_info "  └─ $filename ($size)"
    done
else
    log_warning "대용량 모델 파일 없음 (100MB+)"
    MISSING_MODELS+=("대용량 AI 모델 파일들")
fi

echo ""

# ==============================================================================
# 6. 서버 상태 체크
# ==============================================================================

log_section "Step 6: 서버 상태 체크"

# 포트 사용 확인
PORTS=(8000 3000 5173)
for port in "${PORTS[@]}"; do
    if lsof -i :$port &>/dev/null; then
        process_info=$(lsof -ti:$port | xargs ps -p 2>/dev/null | tail -n +2 | awk '{print $4}')
        log_success "포트 $port 사용 중 - 프로세스: $process_info"
    else
        log_info "포트 $port 사용 가능"
    fi
done

# 백엔드 헬스체크 (실행 중인 경우)
if lsof -i :8000 &>/dev/null; then
    log_info "백엔드 헬스체크 시도 중..."
    if curl -s http://localhost:8000/health &>/dev/null; then
        log_success "백엔드 서버 정상 응답"
    else
        log_warning "백엔드 서버 응답 없음"
    fi
fi

echo ""

# ==============================================================================
# 7. 설정 파일 체크
# ==============================================================================

log_section "Step 7: 설정 파일 체크"

# 백엔드 설정 파일들
BACKEND_CONFIG_FILES=(
    "$BACKEND_DIR/.env:환경 변수"
    "$BACKEND_DIR/requirements.txt:Python 의존성"
    "$BACKEND_DIR/app/main.py:메인 애플리케이션"
    "$BACKEND_DIR/app/core/config.py:핵심 설정"
)

log_info "백엔드 설정 파일:"
for file_info in "${BACKEND_CONFIG_FILES[@]}"; do
    IFS=':' read -r file_path description <<< "$file_info"
    
    if [[ -f "$file_path" ]]; then
        size=$(du -sh "$file_path" 2>/dev/null | cut -f1)
        log_success "$description ($size)"
    else
        log_warning "$description - 파일 없음"
        MISSING_FILES+=("$(basename $file_path)")
    fi
done

echo ""

# 프론트엔드 설정 파일들
FRONTEND_CONFIG_FILES=(
    "$FRONTEND_DIR/.env:환경 변수"
    "$FRONTEND_DIR/package.json:프로젝트 설정"
    "$FRONTEND_DIR/vite.config.ts:Vite 설정"
    "$FRONTEND_DIR/tailwind.config.js:Tailwind 설정"
)

log_info "프론트엔드 설정 파일:"
for file_info in "${FRONTEND_CONFIG_FILES[@]}"; do
    IFS=':' read -r file_path description <<< "$file_info"
    
    if [[ -f "$file_path" ]]; then
        size=$(du -sh "$file_path" 2>/dev/null | cut -f1)
        log_success "$description ($size)"
    else
        log_warning "$description - 파일 없음"
        MISSING_FILES+=("$(basename $file_path)")
    fi
done

echo ""

# ==============================================================================
# 8. 로그 분석 (백엔드가 실행 중인 경우)
# ==============================================================================

log_section "Step 8: 로그 분석"

# 로그 파일 확인
if [[ -d "$BACKEND_DIR/logs" ]]; then
    latest_log=$(find "$BACKEND_DIR/logs" -name "*.log" -type f -exec ls -t {} + | head -1)
    
    if [[ -n "$latest_log" ]]; then
        log_success "최근 로그 파일: $(basename $latest_log)"
        
        # 로그에서 에러 패턴 검색
        error_count=$(grep -c "ERROR\|ERRO\|Failed\|failed\|Missing\|missing" "$latest_log" 2>/dev/null || echo "0")
        warning_count=$(grep -c "WARNING\|WARN\|권장" "$latest_log" 2>/dev/null || echo "0")
        
        log_info "  └─ 에러: ${error_count}개, 경고: ${warning_count}개"
        
        if [[ $error_count -gt 0 ]]; then
            log_warning "최근 에러 메시지들:"
            grep "ERROR\|ERRO\|Failed\|failed" "$latest_log" | tail -3 | while read line; do
                echo "    $line"
            done
        fi
    else
        log_info "로그 파일 없음"
    fi
else
    log_info "로그 디렉토리 없음"
fi

echo ""

# ==============================================================================
# 9. 디스크 공간 체크
# ==============================================================================

log_section "Step 9: 디스크 공간 체크"

# 프로젝트 전체 크기
project_size=$(du -sh "$PROJECT_ROOT" 2>/dev/null | cut -f1)
log_info "프로젝트 전체 크기: $project_size"

# AI 모델 디렉토리 크기
if [[ -d "$AI_MODELS_DIR" ]]; then
    models_size=$(du -sh "$AI_MODELS_DIR" 2>/dev/null | cut -f1)
    log_info "AI 모델 크기: $models_size"
fi

# 사용 가능한 디스크 공간
free_space=$(df -h "$PROJECT_ROOT" | tail -1 | awk '{print $4}')
log_info "사용 가능한 공간: $free_space"

echo ""

# ==============================================================================
# 10. 종합 분석 및 권장사항
# ==============================================================================

log_header "종합 분석 결과"
echo "=================================================================="

# 전체 상태 점수 계산
total_issues=$((${#MISSING_PACKAGES[@]} + ${#MISSING_MODELS[@]} + ${#MISSING_FILES[@]} + ${#WARNINGS[@]}))

if [[ $total_issues -eq 0 ]]; then
    log_success "🎉 모든 환경이 완벽하게 설정되어 있습니다!"
elif [[ $total_issues -le 3 ]]; then
    log_success "✅ 환경이 대부분 완성되어 있습니다 (사소한 이슈 $total_issues개)"
elif [[ $total_issues -le 8 ]]; then
    log_warning "⚠️  일부 설정이 필요합니다 (이슈 $total_issues개)"
else
    log_error "❌ 상당한 설정 작업이 필요합니다 (이슈 $total_issues개)"
fi

echo ""

# 세부 분석
log_info "📊 세부 분석:"
echo "  • 누락 패키지: ${#MISSING_PACKAGES[@]}개"
echo "  • 누락 모델: ${#MISSING_MODELS[@]}개"
echo "  • 누락 파일: ${#MISSING_FILES[@]}개"
echo "  • 경고사항: ${#WARNINGS[@]}개"
echo "  • 총 권장사항: ${#RECOMMENDATIONS[@]}개"

echo ""

# 우선순위별 권장사항
if [[ ${#MISSING_PACKAGES[@]} -gt 0 ]]; then
    log_section "🔧 누락 패키지 설치 권장:"
    for package in "${MISSING_PACKAGES[@]}"; do
        echo "  pip install $package"
    done
    echo ""
fi

if [[ ${#MISSING_MODELS[@]} -gt 0 ]]; then
    log_section "🤖 누락 AI 모델 다운로드 권장:"
    for model in "${MISSING_MODELS[@]}"; do
        echo "  • $model"
    done
    echo ""
    echo "  다운로드 명령어:"
    echo "  python3 scripts/download_ai_models.py --all"
    echo ""
fi

if [[ ${#RECOMMENDATIONS[@]} -gt 0 ]]; then
    log_section "💡 추가 권장사항:"
    for rec in "${RECOMMENDATIONS[@]}"; do
        echo "  • $rec"
    done
    echo ""
fi

# 다음 단계 안내
log_section "🚀 다음 단계:"

if [[ ${#MISSING_PACKAGES[@]} -gt 0 ]]; then
    echo "1. 누락 패키지 설치:"
    echo "   cd backend && source mycloset_env/bin/activate"
    echo "   pip install rembg onnxruntime scikit-image mediapipe"
    echo ""
fi

if [[ ${#MISSING_MODELS[@]} -gt 0 ]]; then
    echo "2. AI 모델 다운로드:"
    echo "   python3 scripts/download_ai_models.py --all"
    echo ""
fi

echo "3. 서버 실행 (설정 완료 후):"
echo "   ./scripts/dev.sh"
echo ""

echo "=================================================================="
log_info "체크 완료 시간: $(date)"
log_warning "📝 이 스크립트는 읽기 전용입니다. 아무것도 수정되지 않았습니다."

# 결과를 파일로 저장
REPORT_FILE="$PROJECT_ROOT/status_check_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "MyCloset AI - 상태 체크 리포트 (읽기 전용)"
    echo "생성일시: $(date)"
    echo "프로젝트: $PROJECT_ROOT"
    echo ""
    echo "=== 요약 ==="
    echo "총 이슈 수: $total_issues"
    echo "누락 패키지: ${#MISSING_PACKAGES[@]}"
    echo "누락 모델: ${#MISSING_MODELS[@]}"
    echo "누락 파일: ${#MISSING_FILES[@]}"
    echo "경고사항: ${#WARNINGS[@]}"
    echo ""
    echo "=== 권장사항 ==="
    if [[ ${#MISSING_PACKAGES[@]} -gt 0 ]]; then
        echo "누락 패키지:"
        printf '  %s\n' "${MISSING_PACKAGES[@]}"
        echo ""
    fi
    
    if [[ ${#MISSING_MODELS[@]} -gt 0 ]]; then
        echo "누락 모델:"
        printf '  %s\n' "${MISSING_MODELS[@]}"
        echo ""
    fi
    
    if [[ ${#RECOMMENDATIONS[@]} -gt 0 ]]; then
        echo "기타 권장사항:"
        printf '  %s\n' "${RECOMMENDATIONS[@]}"
    fi
    
} > "$REPORT_FILE"

log_success "📝 상세 리포트 저장: $REPORT_FILE"
echo "=================================================================="