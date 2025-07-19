#!/bin/bash
# quick_find_models.sh
# MyCloset AI - 빠른 모델 파일 검색 스크립트

set -euo pipefail

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 로그 함수들
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_step() {
    echo -e "${PURPLE}🔍 $1${NC}"
}

# 도움말 함수
show_help() {
    cat << EOF
🔍 MyCloset AI - 빠른 모델 파일 검색 도구

사용법:
    $0 [옵션]

옵션:
    -h, --help              이 도움말 표시
    -d, --deep              전체 시스템 딥 스캔
    -p, --path PATH         특정 경로에서만 검색
    -s, --size MIN_SIZE     최소 파일 크기 (MB, 기본: 1)
    -t, --type EXTENSIONS   파일 확장자 (예: "pth,bin,onnx")
    -o, --output FILE       결과를 파일로 저장
    -c, --count             파일 개수만 표시
    -q, --quiet             조용한 모드

예시:
    $0                      # 기본 검색
    $0 --deep              # 전체 시스템 검색
    $0 -p ./ai_models      # 특정 디렉토리 검색
    $0 -t "pth,bin"        # 특정 확장자만 검색
    $0 -s 10               # 10MB 이상 파일만
    $0 -o models.txt       # 결과를 파일로 저장

EOF
}

# 기본 설정
DEEP_SCAN=false
SEARCH_PATH=""
MIN_SIZE_MB=1
FILE_EXTENSIONS="pth pt bin safetensors onnx pb h5 tflite pkl caffemodel"
OUTPUT_FILE=""
COUNT_ONLY=false
QUIET=false

# 명령행 인수 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--deep)
            DEEP_SCAN=true
            shift
            ;;
        -p|--path)
            SEARCH_PATH="$2"
            shift 2
            ;;
        -s|--size)
            MIN_SIZE_MB="$2"
            shift 2
            ;;
        -t|--type)
            FILE_EXTENSIONS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -c|--count)
            COUNT_ONLY=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        *)
            log_error "알 수 없는 옵션: $1"
            show_help
            exit 1
            ;;
    esac
done

# 조용한 모드가 아닐 때만 로그 출력
if [[ "$QUIET" != "true" ]]; then
    cat << "EOF"
🔍 MyCloset AI 모델 검색기
=========================
EOF
fi

# 운영체제 감지
OS_TYPE=$(uname -s)
case $OS_TYPE in
    Darwin*)
        OS_NAME="macOS"
        FIND_CMD="find"
        ;;
    Linux*)
        OS_NAME="Linux"  
        FIND_CMD="find"
        ;;
    CYGWIN*|MINGW*|MSYS*)
        OS_NAME="Windows"
        FIND_CMD="find"
        ;;
    *)
        OS_NAME="Unknown"
        FIND_CMD="find"
        ;;
esac

[[ "$QUIET" != "true" ]] && log_info "운영체제: $OS_NAME"

# 검색 경로 설정
if [[ -n "$SEARCH_PATH" ]]; then
    if [[ ! -d "$SEARCH_PATH" ]]; then
        log_error "경로를 찾을 수 없습니다: $SEARCH_PATH"
        exit 1
    fi
    SEARCH_PATHS=("$SEARCH_PATH")
elif [[ "$DEEP_SCAN" == "true" ]]; then
    [[ "$QUIET" != "true" ]] && log_warning "전체 시스템 스캔은 시간이 오래 걸릴 수 있습니다"
    
    case $OS_NAME in
        "macOS")
            SEARCH_PATHS=("/" "$HOME")
            ;;
        "Linux")
            SEARCH_PATHS=("/home" "/opt" "/usr" "/var")
            ;;
        "Windows")
            SEARCH_PATHS=("/c" "/d" "/e")
            ;;
        *)
            SEARCH_PATHS=("/")
            ;;
    esac
else
    # 일반적인 AI 모델 경로들
    COMMON_PATHS=(
        "."
        "./ai_models"
        "./models"
        "./checkpoints" 
        "./weights"
        "./pretrained"
        "$HOME/.cache"
        "$HOME/models"
        "$HOME/Downloads"
        "$HOME/.cache/huggingface"
        "$HOME/.cache/torch"
        "$HOME/.cache/transformers"
        "$HOME/anaconda3/envs"
        "$HOME/miniconda3/envs"
    )
    
    # macOS 특화 경로
    if [[ "$OS_NAME" == "macOS" ]]; then
        COMMON_PATHS+=(
            "$HOME/Library/Caches"
            "/opt/homebrew"
            "/usr/local"
            "/Applications"
        )
    fi
    
    # 존재하는 경로만 선택
    SEARCH_PATHS=()
    for path in "${COMMON_PATHS[@]}"; do
        if [[ -d "$path" ]]; then
            SEARCH_PATHS+=("$path")
        fi
    done
fi

[[ "$QUIET" != "true" ]] && log_info "검색 경로: ${#SEARCH_PATHS[@]}개"

# 확장자별 find 패턴 생성
create_find_pattern() {
    local extensions="$1"
    local pattern=""
    
    IFS=',' read -ra EXT_ARRAY <<< "$extensions"
    for ext in "${EXT_ARRAY[@]}"; do
        ext=$(echo "$ext" | xargs)  # 공백 제거
        if [[ -n "$pattern" ]]; then
            pattern="$pattern -o"
        fi
        pattern="$pattern -name *.$ext"
    done
    
    echo "\\( $pattern \\)"
}

# 파일 크기를 MB로 변환
bytes_to_mb() {
    local bytes=$1
    echo "scale=2; $bytes / 1024 / 1024" | bc -l 2>/dev/null || echo "0"
}

# 파일 정보 출력
print_file_info() {
    local file_path="$1"
    local file_size_bytes file_size_mb created_time modified_time
    
    if [[ -f "$file_path" ]]; then
        if [[ "$OS_NAME" == "macOS" ]]; then
            file_size_bytes=$(stat -f%z "$file_path" 2>/dev/null || echo "0")
            created_time=$(stat -f%SB -t%Y-%m-%d\ %H:%M:%S "$file_path" 2>/dev/null || echo "Unknown")
            modified_time=$(stat -f%Sm -t%Y-%m-%d\ %H:%M:%S "$file_path" 2>/dev/null || echo "Unknown")
        else
            file_size_bytes=$(stat -c%s "$file_path" 2>/dev/null || echo "0")
            created_time=$(stat -c%w "$file_path" 2>/dev/null | cut -d'.' -f1 || echo "Unknown")
            modified_time=$(stat -c%y "$file_path" 2>/dev/null | cut -d'.' -f1 || echo "Unknown")
        fi
        
        file_size_mb=$(bytes_to_mb "$file_size_bytes")
        
        # MB 크기 확인
        if (( $(echo "$file_size_mb >= $MIN_SIZE_MB" | bc -l) )); then
            if [[ "$COUNT_ONLY" != "true" ]]; then
                echo "📁 $(basename "$file_path")"
                echo "   경로: $file_path"
                echo "   크기: ${file_size_mb}MB"
                echo "   수정: $modified_time"
                echo ""
            fi
            return 0
        fi
    fi
    return 1
}

# Step 분류 함수
classify_step() {
    local filename="$1"
    local filename_lower=$(echo "$filename" | tr '[:upper:]' '[:lower:]')
    
    case "$filename_lower" in
        *human*parsing*|*graphonomy*|*schp*|*atr*)
            echo "step_01_human_parsing"
            ;;
        *pose*estimation*|*openpose*|*mediapipe*|*pose*net*)
            echo "step_02_pose_estimation"
            ;;
        *cloth*seg*|*u2net*|*sam*|*segment*)
            echo "step_03_cloth_segmentation"
            ;;
        *geometric*matching*|*gmm*|*tps*)
            echo "step_04_geometric_matching"
            ;;
        *cloth*warp*|*tom*|*viton*warp*)
            echo "step_05_cloth_warping"
            ;;
        *virtual*fitting*|*ootdiffusion*|*stable*diffusion*|*diffusion*)
            echo "step_06_virtual_fitting"
            ;;
        *post*process*|*enhancement*|*super*resolution*|*srresnet*|*esrgan*)
            echo "step_07_post_processing"
            ;;
        *quality*|*clip*|*aesthetic*|*scoring*)
            echo "step_08_quality_assessment"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# 메인 검색 함수
search_models() {
    local total_found=0
    local total_size_mb=0
    local step_counts=()
    
    # Step별 카운터 초기화
    declare -A step_counter
    step_counter["step_01_human_parsing"]=0
    step_counter["step_02_pose_estimation"]=0
    step_counter["step_03_cloth_segmentation"]=0
    step_counter["step_04_geometric_matching"]=0
    step_counter["step_05_cloth_warping"]=0
    step_counter["step_06_virtual_fitting"]=0
    step_counter["step_07_post_processing"]=0
    step_counter["step_08_quality_assessment"]=0
    step_counter["unknown"]=0
    
    # 임시 결과 파일
    local temp_file
    temp_file=$(mktemp)
    
    [[ "$QUIET" != "true" ]] && log_step "모델 파일 검색 중..."
    
    # 각 경로에서 검색
    for search_path in "${SEARCH_PATHS[@]}"; do
        [[ "$QUIET" != "true" ]] && log_info "검색 중: $search_path"
        
        # find 명령 실행
        local find_pattern
        find_pattern=$(create_find_pattern "$FILE_EXTENSIONS")
        
        # find 명령 실행 (올바른 문법)
        $FIND_CMD "$search_path" -type f $find_pattern 2>/dev/null | while read -r file_path; do
            if print_file_info "$file_path" >> "$temp_file"; then
                echo "$file_path" >> "${temp_file}.paths"
            fi
        done
    done
    
    # 결과 처리
    if [[ -f "${temp_file}.paths" ]]; then
        total_found=$(wc -l < "${temp_file}.paths")
        
        # Step별 분류
        while read -r file_path; do
            filename=$(basename "$file_path")
            step=$(classify_step "$filename")
            ((step_counter["$step"]++))
            
            # 파일 크기 계산
            if [[ -f "$file_path" ]]; then
                local file_size_bytes
                if [[ "$OS_NAME" == "macOS" ]]; then
                    file_size_bytes=$(stat -f%z "$file_path" 2>/dev/null || echo "0")
                else
                    file_size_bytes=$(stat -c%s "$file_path" 2>/dev/null || echo "0")
                fi
                local file_size_mb
                file_size_mb=$(bytes_to_mb "$file_size_bytes")
                total_size_mb=$(echo "$total_size_mb + $file_size_mb" | bc -l)
            fi
        done < "${temp_file}.paths"
    fi
    
    # 결과 출력
    if [[ "$COUNT_ONLY" == "true" ]]; then
        echo "$total_found"
    else
        if [[ -f "$temp_file" ]]; then
            cat "$temp_file"
        fi
        
        # 요약 출력
        if [[ "$QUIET" != "true" ]]; then
            echo "=============================="
            log_success "검색 완료!"
            echo ""
            log_info "총 $total_found개 모델 발견"
            log_info "총 크기: ${total_size_mb}MB"
            echo ""
            log_info "Step별 분류:"
            
            for step in step_01_human_parsing step_02_pose_estimation step_03_cloth_segmentation step_04_geometric_matching step_05_cloth_warping step_06_virtual_fitting step_07_post_processing step_08_quality_assessment unknown; do
                local count=${step_counter["$step"]}
                if [[ $count -gt 0 ]]; then
                    echo "  - $step: $count개"
                fi
            done
        fi
    fi
    
    # 출력 파일 저장
    if [[ -n "$OUTPUT_FILE" ]]; then
        if [[ -f "$temp_file" ]]; then
            cp "$temp_file" "$OUTPUT_FILE"
            [[ "$QUIET" != "true" ]] && log_success "결과 저장: $OUTPUT_FILE"
        fi
    fi
    
    # 임시 파일 정리
    rm -f "$temp_file" "${temp_file}.paths"
    
    return 0
}

# bc 명령 확인
if ! command -v bc &> /dev/null; then
    log_warning "bc 명령을 찾을 수 없습니다. 파일 크기 계산이 부정확할 수 있습니다."
fi

# 메인 실행
main() {
    local start_time
    start_time=$(date +%s)
    
    # 검색 실행
    search_models
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    [[ "$QUIET" != "true" ]] && log_info "검색 시간: ${duration}초"
}

# 스크립트 실행
main "$@"