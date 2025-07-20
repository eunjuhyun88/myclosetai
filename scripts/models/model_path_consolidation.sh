#!/bin/bash

# =============================================================================
# MyCloset AI - Step별 모델 경로 완전 통합 스크립트
# 기존 중복 모델들을 번호를 달아 보존하면서 Step별로 체계적 정리
# =============================================================================

set -e

# 색상 출력 함수들
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
log_header() { echo -e "${PURPLE}\n🚀 $1${NC}"; echo "=" | tr -d '\n' | head -c 80; echo; }
log_step() { echo -e "${CYAN}📋 $1${NC}"; }

# 프로젝트 루트 확인
PROJECT_ROOT=$(pwd)
AI_MODELS_ROOT="$PROJECT_ROOT/ai_models"
BACKEND_AI_MODELS="$PROJECT_ROOT/backend/ai_models"

log_header "MyCloset AI - Step별 모델 경로 완전 통합"
log_info "프로젝트 루트: $PROJECT_ROOT"
log_info "타겟 디렉토리: $AI_MODELS_ROOT"
echo ""

# 1. 현재 상황 분석
log_header "Step 1: 현재 AI 모델 상황 분석"

log_step "현재 AI 모델 디렉토리 구조 분석 중..."

# 기존 ai_models 디렉토리들 찾기
AI_MODEL_DIRS=(
    "$AI_MODELS_ROOT"
    "$BACKEND_AI_MODELS"
    "$PROJECT_ROOT/backend/app/ai_models"
    "$PROJECT_ROOT/backend/app/ai_pipeline/models"
)

echo "📁 발견된 AI 모델 디렉토리들:"
for dir in "${AI_MODEL_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        size=$(du -sh "$dir" 2>/dev/null | cut -f1 || echo "N/A")
        count=$(find "$dir" -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" \) 2>/dev/null | wc -l || echo 0)
        echo "  ✅ $dir (크기: $size, 모델파일: $count개)"
    else
        echo "  ❌ $dir (없음)"
    fi
done

# 2. 타겟 디렉토리 구조 생성
log_header "Step 2: Step별 타겟 디렉토리 구조 생성"

# 표준 Step별 디렉토리 생성
STEP_DIRS=(
    "step_01_human_parsing"
    "step_02_pose_estimation" 
    "step_03_cloth_segmentation"
    "step_04_geometric_matching"
    "step_05_cloth_warping"
    "step_06_virtual_fitting"
    "step_07_post_processing"
    "step_08_quality_assessment"
    "auxiliary_models"
    "huggingface_cache"
    "backup_models"
    "experimental_models"
)

log_step "Step별 표준 디렉토리 생성 중..."

# ai_models 루트 디렉토리 생성
mkdir -p "$AI_MODELS_ROOT"

for step_dir in "${STEP_DIRS[@]}"; do
    target_dir="$AI_MODELS_ROOT/$step_dir"
    mkdir -p "$target_dir"
    
    # 세부 카테고리 디렉토리도 생성
    case $step_dir in
        "step_01_human_parsing")
            mkdir -p "$target_dir/graphonomy" "$target_dir/schp" "$target_dir/densepose" "$target_dir/alternatives"
            ;;
        "step_02_pose_estimation")
            mkdir -p "$target_dir/openpose" "$target_dir/mediapipe" "$target_dir/hrnet" "$target_dir/alternatives"
            ;;
        "step_03_cloth_segmentation") 
            mkdir -p "$target_dir/u2net" "$target_dir/sam" "$target_dir/rembg" "$target_dir/alternatives"
            ;;
        "step_04_geometric_matching")
            mkdir -p "$target_dir/gmm" "$target_dir/tps" "$target_dir/alternatives"
            ;;
        "step_05_cloth_warping")
            mkdir -p "$target_dir/tom" "$target_dir/alternatives"
            ;;
        "step_06_virtual_fitting")
            mkdir -p "$target_dir/ootdiffusion" "$target_dir/stable_diffusion" "$target_dir/hrviton" "$target_dir/alternatives"
            ;;
        "step_07_post_processing")
            mkdir -p "$target_dir/super_resolution" "$target_dir/enhancement" "$target_dir/alternatives"
            ;;
        "step_08_quality_assessment")
            mkdir -p "$target_dir/clip" "$target_dir/alternatives"
            ;;
        "auxiliary_models")
            mkdir -p "$target_dir/clip" "$target_dir/sam" "$target_dir/vae" "$target_dir/text_encoders"
            ;;
        "huggingface_cache")
            mkdir -p "$target_dir/transformers" "$target_dir/diffusers" "$target_dir/models"
            ;;
    esac
    
    # .gitkeep 파일 생성
    touch "$target_dir/.gitkeep"
    
    log_info "생성: $step_dir"
done

log_success "Step별 디렉토리 구조 생성 완료"

# 3. 기존 모델 파일 탐지 및 분류
log_header "Step 3: 기존 모델 파일 탐지 및 Step별 분류"

log_step "모든 AI 모델 파일 탐지 중..."

# 모델 파일 탐지 함수
detect_model_files() {
    local search_dir="$1"
    
    if [ ! -d "$search_dir" ]; then
        return 0
    fi
    
    find "$search_dir" -type f \( \
        -name "*.pth" -o \
        -name "*.pt" -o \
        -name "*.bin" -o \
        -name "*.safetensors" -o \
        -name "*.ckpt" -o \
        -name "*.onnx" -o \
        -name "*.pkl" \
    \) 2>/dev/null
}

# Step별 패턴 매칭 함수
classify_model_by_step() {
    local file_path="$1"
    local file_name=$(basename "$file_path")
    local file_lower=$(echo "$file_name" | tr '[:upper:]' '[:lower:]')
    local path_lower=$(echo "$file_path" | tr '[:upper:]' '[:lower:]')
    
    # Step 01: Human Parsing
    if [[ "$file_lower" =~ (human|parsing|schp|atr|graphonomy|densepose) ]] || \
       [[ "$path_lower" =~ (human.*parsing|step.*01|parsing) ]]; then
        echo "step_01_human_parsing"
        
    # Step 02: Pose Estimation  
    elif [[ "$file_lower" =~ (pose|openpose|body|keypoint|mediapipe|hrnet) ]] || \
         [[ "$path_lower" =~ (pose.*estimation|step.*02|openpose) ]]; then
        echo "step_02_pose_estimation"
        
    # Step 03: Cloth Segmentation
    elif [[ "$file_lower" =~ (u2net|segmentation|cloth.*seg|mask|rembg|sam) ]] || \
         [[ "$path_lower" =~ (cloth.*segmentation|step.*03|u2net) ]]; then
        echo "step_03_cloth_segmentation"
        
    # Step 04: Geometric Matching
    elif [[ "$file_lower" =~ (geometric|matching|gmm|tps) ]] || \
         [[ "$path_lower" =~ (geometric.*matching|step.*04) ]]; then
        echo "step_04_geometric_matching"
        
    # Step 05: Cloth Warping
    elif [[ "$file_lower" =~ (warping|warp|tom|cloth.*warp) ]] || \
         [[ "$path_lower" =~ (cloth.*warping|step.*05) ]]; then
        echo "step_05_cloth_warping"
        
    # Step 06: Virtual Fitting
    elif [[ "$file_lower" =~ (diffusion|ootd|viton|stable|unet|vae) ]] || \
         [[ "$path_lower" =~ (virtual.*fitting|step.*06|ootdiffusion|stable.*diffusion) ]]; then
        echo "step_06_virtual_fitting"
        
    # Step 07: Post Processing
    elif [[ "$file_lower" =~ (super|resolution|esrgan|sr|denoise|enhance|gfpgan|codeformer|swinir) ]] || \
         [[ "$path_lower" =~ (post.*processing|step.*07|super.*resolution) ]]; then
        echo "step_07_post_processing"
        
    # Step 08: Quality Assessment
    elif [[ "$file_lower" =~ (quality|assessment|clip|similarity) ]] || \
         [[ "$path_lower" =~ (quality.*assessment|step.*08) ]]; then
        echo "step_08_quality_assessment"
        
    # Auxiliary Models
    elif [[ "$file_lower" =~ (clip|text.*encoder|feature.*extract) ]] || \
         [[ "$path_lower" =~ (auxiliary|clip) ]]; then
        echo "auxiliary_models"
        
    # HuggingFace Cache
    elif [[ "$path_lower" =~ (huggingface|transformers|diffusers|models--) ]]; then
        echo "huggingface_cache"
        
    else
        echo "experimental_models"
    fi
}

# 모든 소스에서 모델 파일 탐지
declare -A DETECTED_FILES
declare -A STEP_COUNTS

log_step "소스별 모델 파일 탐지 진행 중..."

for source_dir in "${AI_MODEL_DIRS[@]}"; do
    if [ -d "$source_dir" ]; then
        log_info "탐지 중: $source_dir"
        
        while IFS= read -r -d '' file_path; do
            if [ -f "$file_path" ]; then
                step=$(classify_model_by_step "$file_path")
                DETECTED_FILES["$file_path"]="$step"
                STEP_COUNTS["$step"]=$((${STEP_COUNTS["$step"]:-0} + 1))
            fi
        done < <(detect_model_files "$source_dir" | tr '\n' '\0')
    fi
done

# 탐지 결과 출력
echo ""
log_success "모델 파일 탐지 완료"
echo "📊 Step별 탐지된 모델 수:"

for step in "${STEP_DIRS[@]}"; do
    count=${STEP_COUNTS["$step"]:-0}
    if [ $count -gt 0 ]; then
        echo "  ✅ $step: $count개"
    else
        echo "  ⚪ $step: 0개"
    fi
done

# 4. 중복 체크 및 이동 계획 수립
log_header "Step 4: 중복 체크 및 이동 계획 수립"

log_step "중복 파일 체크 및 번호 할당 중..."

declare -A FILE_REGISTRY
declare -A MOVE_PLAN

# 파일 이름 기반 중복 체크
for file_path in "${!DETECTED_FILES[@]}"; do
    step="${DETECTED_FILES[$file_path]}"
    file_name=$(basename "$file_path")
    file_size=$(stat -f%z "$file_path" 2>/dev/null || stat -c%s "$file_path" 2>/dev/null || echo "0")
    
    # 고유 키 생성 (파일명 + 크기)
    unique_key="${file_name}_${file_size}"
    
    target_dir="$AI_MODELS_ROOT/$step"
    
    if [[ -v FILE_REGISTRY["$step:$file_name"] ]]; then
        # 중복 파일 발견 - 번호 할당
        existing_count=${FILE_REGISTRY["$step:$file_name"]}
        new_count=$((existing_count + 1))
        FILE_REGISTRY["$step:$file_name"]=$new_count
        
        # 확장자 분리
        file_base="${file_name%.*}"
        file_ext="${file_name##*.}"
        
        new_file_name="${file_base}_${new_count}.${file_ext}"
        target_path="$target_dir/$new_file_name"
        
        log_warning "중복 발견: $file_name → $new_file_name"
    else
        # 처음 발견된 파일
        FILE_REGISTRY["$step:$file_name"]=1
        target_path="$target_dir/$file_name"
    fi
    
    MOVE_PLAN["$file_path"]="$target_path"
done

# 5. 실제 파일 이동 실행
log_header "Step 5: 실제 파일 이동 실행"

log_warning "실제 파일 이동을 실행하시겠습니까? (y/N)"
read -r confirm

if [[ ! $confirm =~ ^[Yy]$ ]]; then
    log_info "이동 계획만 출력하고 종료합니다."
    
    echo ""
    echo "📋 이동 계획 (미실행):"
    for source_path in "${!MOVE_PLAN[@]}"; do
        target_path="${MOVE_PLAN[$source_path]}"
        file_size=$(du -h "$source_path" 2>/dev/null | cut -f1 || echo "N/A")
        echo "  📁 $source_path → $target_path ($file_size)"
    done
    
    exit 0
fi

log_step "파일 이동 실행 중..."

moved_count=0
error_count=0

for source_path in "${!MOVE_PLAN[@]}"; do
    target_path="${MOVE_PLAN[$source_path]}"
    target_dir=$(dirname "$target_path")
    
    # 타겟 디렉토리 생성 확인
    mkdir -p "$target_dir"
    
    # 파일 이동 (복사 후 원본 유지)
    if cp "$source_path" "$target_path" 2>/dev/null; then
        file_size=$(du -h "$target_path" 2>/dev/null | cut -f1 || echo "N/A")
        log_info "✅ $source_path → $target_path ($file_size)"
        moved_count=$((moved_count + 1))
    else
        log_error "❌ 이동 실패: $source_path → $target_path"
        error_count=$((error_count + 1))
    fi
done

# 6. 결과 요약 및 검증
log_header "Step 6: 결과 요약 및 검증"

log_success "파일 이동 완료!"
echo "📊 이동 결과:"
echo "  ✅ 성공: $moved_count개"
echo "  ❌ 실패: $error_count개"

# 각 Step별 최종 상태 확인
echo ""
echo "📁 Step별 최종 모델 현황:"

for step_dir in "${STEP_DIRS[@]}"; do
    target_dir="$AI_MODELS_ROOT/$step_dir"
    if [ -d "$target_dir" ]; then
        count=$(find "$target_dir" -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" \) 2>/dev/null | wc -l)
        size=$(du -sh "$target_dir" 2>/dev/null | cut -f1 || echo "0B")
        
        if [ $count -gt 0 ]; then
            echo "  ✅ $step_dir: $count개 파일, $size"
        else
            echo "  ⚪ $step_dir: 비어있음"
        fi
    fi
done

# 7. 설정 파일 생성
log_header "Step 7: 최신 경로 설정 파일 생성"

log_step "통합된 경로 설정 파일 생성 중..."

# Python 설정 파일 생성
cat > "$PROJECT_ROOT/backend/app/core/unified_model_paths.py" << 'EOF'
# app/core/unified_model_paths.py
"""
MyCloset AI - 통합된 Step별 모델 경로 설정
자동 생성됨: $(date +'%Y-%m-%d %H:%M:%S')
"""

from pathlib import Path
from typing import Dict, List, Optional

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
AI_MODELS_ROOT = PROJECT_ROOT / "ai_models"

# Step별 표준 경로
STEP_MODEL_PATHS = {
    "step_01_human_parsing": AI_MODELS_ROOT / "step_01_human_parsing",
    "step_02_pose_estimation": AI_MODELS_ROOT / "step_02_pose_estimation", 
    "step_03_cloth_segmentation": AI_MODELS_ROOT / "step_03_cloth_segmentation",
    "step_04_geometric_matching": AI_MODELS_ROOT / "step_04_geometric_matching",
    "step_05_cloth_warping": AI_MODELS_ROOT / "step_05_cloth_warping",
    "step_06_virtual_fitting": AI_MODELS_ROOT / "step_06_virtual_fitting",
    "step_07_post_processing": AI_MODELS_ROOT / "step_07_post_processing",
    "step_08_quality_assessment": AI_MODELS_ROOT / "step_08_quality_assessment",
    "auxiliary_models": AI_MODELS_ROOT / "auxiliary_models",
    "huggingface_cache": AI_MODELS_ROOT / "huggingface_cache",
    "backup_models": AI_MODELS_ROOT / "backup_models",
    "experimental_models": AI_MODELS_ROOT / "experimental_models"
}

def get_step_path(step_name: str) -> Optional[Path]:
    """Step 이름으로 경로 반환"""
    return STEP_MODEL_PATHS.get(step_name)

def get_available_models(step_name: str) -> List[Path]:
    """특정 Step의 사용 가능한 모델 파일들 반환"""
    step_path = get_step_path(step_name)
    if not step_path or not step_path.exists():
        return []
    
    model_files = []
    for pattern in ["*.pth", "*.pt", "*.bin", "*.safetensors"]:
        model_files.extend(step_path.glob(f"**/{pattern}"))
    
    return sorted(model_files)

def get_all_model_files() -> Dict[str, List[Path]]:
    """모든 Step의 모델 파일들 반환"""
    all_models = {}
    for step_name in STEP_MODEL_PATHS.keys():
        all_models[step_name] = get_available_models(step_name)
    
    return all_models

# 통합 정보
INTEGRATION_INFO = {
    "integration_date": "$(date +'%Y-%m-%d %H:%M:%S')",
    "total_steps": len(STEP_MODEL_PATHS),
    "ai_models_root": str(AI_MODELS_ROOT),
    "integration_method": "step_based_classification"
}
EOF

log_success "통합 경로 설정 파일 생성 완료: backend/app/core/unified_model_paths.py"

# 8. 검증 및 최종 확인
log_header "Step 8: 최종 검증"

log_step "통합 결과 검증 중..."

# Python으로 검증 실행
python3 << 'EOF'
import sys
sys.path.append("backend/app")

try:
    from core.unified_model_paths import STEP_MODEL_PATHS, get_all_model_files
    
    print("🔍 통합 결과 검증:")
    all_models = get_all_model_files()
    
    total_files = 0
    for step_name, model_files in all_models.items():
        count = len(model_files)
        total_files += count
        if count > 0:
            print(f"  ✅ {step_name}: {count}개 모델")
        else:
            print(f"  ⚪ {step_name}: 비어있음")
    
    print(f"\n📊 총 모델 파일: {total_files}개")
    print("✅ 통합 경로 설정 검증 완료")
    
except Exception as e:
    print(f"❌ 검증 실패: {e}")
    sys.exit(1)
EOF

echo ""
log_header "🎉 AI 모델 Step별 경로 통합 완료!"

echo ""
log_success "✅ 모든 AI 모델이 Step별로 체계적으로 정리되었습니다"
log_success "✅ 중복 모델들은 번호를 달아 보존되었습니다"
log_success "✅ 통합된 경로 설정 파일이 생성되었습니다"

echo ""
echo "📁 통합된 구조:"
echo "  $AI_MODELS_ROOT/"
echo "  ├── step_01_human_parsing/     (Human Parsing 모델들)"
echo "  ├── step_02_pose_estimation/   (Pose Estimation 모델들)"
echo "  ├── step_03_cloth_segmentation/(Cloth Segmentation 모델들)"
echo "  ├── step_04_geometric_matching/ (Geometric Matching 모델들)"
echo "  ├── step_05_cloth_warping/     (Cloth Warping 모델들)"
echo "  ├── step_06_virtual_fitting/   (Virtual Fitting 모델들)"
echo "  ├── step_07_post_processing/   (Post Processing 모델들)"
echo "  ├── step_08_quality_assessment/(Quality Assessment 모델들)"
echo "  ├── auxiliary_models/          (보조 모델들)"
echo "  ├── huggingface_cache/         (HuggingFace 캐시)"
echo "  └── experimental_models/       (분류되지 않은 모델들)"

echo ""
echo "🚀 다음 단계:"
echo "1. backend/app/core/unified_model_paths.py 경로 설정 확인"
echo "2. 각 Step에서 통합된 경로 사용하도록 코드 업데이트"
echo "3. ModelLoader에서 새 경로 구조 연동"
echo "4. 테스트 실행 및 검증"

echo ""
log_info "스크립트 실행 완료: $(date +'%Y-%m-%d %H:%M:%S')"