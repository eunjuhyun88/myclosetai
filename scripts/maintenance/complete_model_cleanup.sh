#!/bin/bash

# =============================================================================
# MyCloset AI - 완전 모델 정리 스크립트
# 남은 모델 디렉토리들을 Step별로 완전히 정리
# =============================================================================

set -e

# 색상 출력
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

PROJECT_ROOT=$(pwd)
AI_MODELS_ROOT="$PROJECT_ROOT/ai_models"

echo "🚀 MyCloset AI - 완전 모델 정리"
echo "======================================"
log_info "프로젝트 루트: $PROJECT_ROOT"
log_info "AI 모델 루트: $AI_MODELS_ROOT"
echo ""

# 1. 현재 남은 모델 디렉토리 확인
echo "📋 현재 남은 모델 디렉토리들:"
cd "$AI_MODELS_ROOT"

remaining_dirs=(
    "openpose"
    "Self-Correction-Human-Parsing"
    "SAM2"
    "StableVITON"
    "IDM-VTON"
    "OOTDiffusion"
    "clip-vit-base-patch32"
    "cloth_warping"
    "temp_downloads"
)

existing_dirs=()
for dir in "${remaining_dirs[@]}"; do
    if [ -d "$dir" ]; then
        size=$(du -sh "$dir" 2>/dev/null | cut -f1 || echo "N/A")
        echo "  📁 $dir ($size)"
        existing_dirs+=("$dir")
    fi
done

if [ ${#existing_dirs[@]} -eq 0 ]; then
    log_success "모든 모델이 이미 정리되었습니다!"
    exit 0
fi

echo ""
log_warning "총 ${#existing_dirs[@]}개 디렉토리를 이동합니다. 계속하시겠습니까? (y/N)"
read -r confirm

if [[ ! $confirm =~ ^[Yy]$ ]]; then
    log_info "작업이 취소되었습니다."
    exit 0
fi

# 2. Step별 이동 실행
echo ""
echo "🔄 Step별 모델 이동 시작..."

# Step 02: Pose Estimation
if [ -d "openpose" ]; then
    log_info "OpenPose → step_02_pose_estimation/openpose/"
    mv "openpose" "step_02_pose_estimation/openpose/"
    log_success "OpenPose 이동 완료"
fi

# Step 01: Human Parsing
if [ -d "Self-Correction-Human-Parsing" ]; then
    log_info "Self-Correction-Human-Parsing → step_01_human_parsing/schp/"
    mv "Self-Correction-Human-Parsing" "step_01_human_parsing/schp/"
    log_success "Human Parsing 이동 완료"
fi

# Step 03: Cloth Segmentation
if [ -d "SAM2" ]; then
    log_info "SAM2 → step_03_cloth_segmentation/sam/"
    mv "SAM2" "step_03_cloth_segmentation/sam/"
    log_success "SAM2 이동 완료"
fi

# Step 06: Virtual Fitting
if [ -d "StableVITON" ]; then
    log_info "StableVITON → step_06_virtual_fitting/stable_diffusion/"
    mv "StableVITON" "step_06_virtual_fitting/stable_diffusion/"
    log_success "StableVITON 이동 완료"
fi

if [ -d "IDM-VTON" ]; then
    log_info "IDM-VTON → step_06_virtual_fitting/alternatives/"
    mv "IDM-VTON" "step_06_virtual_fitting/alternatives/"
    log_success "IDM-VTON 이동 완료"
fi

if [ -d "OOTDiffusion" ]; then
    log_info "OOTDiffusion → step_06_virtual_fitting/ootdiffusion/"
    mv "OOTDiffusion" "step_06_virtual_fitting/ootdiffusion/"
    log_success "OOTDiffusion 이동 완료"
fi

# Step 05: Cloth Warping
if [ -d "cloth_warping" ]; then
    log_info "cloth_warping → step_05_cloth_warping/tom/"
    mv "cloth_warping" "step_05_cloth_warping/tom/"
    log_success "Cloth Warping 이동 완료"
fi

# Auxiliary Models
if [ -d "clip-vit-base-patch32" ]; then
    log_info "clip-vit-base-patch32 → auxiliary_models/clip/"
    mv "clip-vit-base-patch32" "auxiliary_models/clip/"
    log_success "CLIP 모델 이동 완료"
fi

# 개별 파일들 정리
if [ -f "mps_optimization.py" ]; then
    log_info "mps_optimization.py → auxiliary_models/"
    mv "mps_optimization.py" "auxiliary_models/"
    log_success "MPS 최적화 파일 이동 완료"
fi

# 임시 파일 정리
if [ -d "temp_downloads" ]; then
    log_warning "temp_downloads 디렉토리 정리 중..."
    
    if [ "$(ls -A temp_downloads)" ]; then
        # 비어있지 않으면 backup으로 이동
        mv "temp_downloads" "backup_models/temp_downloads_$(date +%Y%m%d_%H%M%S)"
        log_info "temp_downloads → backup_models/ (백업됨)"
    else
        # 비어있으면 삭제
        rmdir "temp_downloads"
        log_info "빈 temp_downloads 디렉토리 삭제"
    fi
fi

# 3. 심볼릭 링크 정리
echo ""
echo "🔗 심볼릭 링크 정리 중..."

# 끊어진 심볼릭 링크 찾기 및 수정
find . -type l ! -exec test -e {} \; -print 2>/dev/null | while read -r broken_link; do
    log_warning "끊어진 링크 발견: $broken_link"
    rm "$broken_link"
    log_info "끊어진 링크 제거: $broken_link"
done

# 4. checkpoints 디렉토리 내용 분류
echo ""
echo "📦 checkpoints 디렉토리 내용 분류 중..."

if [ -d "checkpoints" ]; then
    cd "checkpoints"
    
    # 각 서브디렉토리를 해당 Step으로 이동
    declare -A checkpoint_mapping=(
        ["human_parsing"]="step_01_human_parsing/alternatives"
        ["pose_estimation"]="step_02_pose_estimation/alternatives" 
        ["cloth_segmentation"]="step_03_cloth_segmentation/alternatives"
        ["geometric_matching"]="step_04_geometric_matching/alternatives"
        ["cloth_warping"]="step_05_cloth_warping/alternatives"
        ["virtual_fitting"]="step_06_virtual_fitting/alternatives"
        ["post_processing"]="step_07_post_processing/alternatives"
        ["quality_assessment"]="step_08_quality_assessment/alternatives"
        ["depth_estimation"]="auxiliary_models"
        ["feature_extraction"]="auxiliary_models"
        ["step_02"]="step_02_pose_estimation/alternatives"
        ["step_05"]="step_05_cloth_warping/alternatives"
        ["step_07"]="step_07_post_processing/alternatives"
    )
    
    for checkpoint_dir in */; do
        checkpoint_dir=${checkpoint_dir%/}  # 뒤의 / 제거
        
        if [[ -v checkpoint_mapping["$checkpoint_dir"] ]]; then
            target_path="../${checkpoint_mapping[$checkpoint_dir]}/$checkpoint_dir"
            
            if [ -d "$checkpoint_dir" ]; then
                log_info "checkpoints/$checkpoint_dir → ${checkpoint_mapping[$checkpoint_dir]}/"
                
                # 타겟 디렉토리 생성
                mkdir -p "../${checkpoint_mapping[$checkpoint_dir]}"
                
                # 이동
                mv "$checkpoint_dir" "$target_path"
                log_success "$checkpoint_dir 이동 완료"
            fi
        else
            log_warning "분류되지 않은 체크포인트: $checkpoint_dir"
        fi
    done
    
    # 개별 파일들 처리
    if [ -f "model_metadata.json" ]; then
        mv "model_metadata.json" "../auxiliary_models/"
        log_info "model_metadata.json → auxiliary_models/"
    fi
    
    cd ..
    
    # checkpoints 디렉토리가 비어있으면 제거
    if [ ! "$(ls -A checkpoints)" ]; then
        rmdir "checkpoints"
        log_info "빈 checkpoints 디렉토리 제거"
    fi
fi

# 5. 기타 정리
echo ""
echo "🧹 기타 파일 정리 중..."

# JSON 파일들 정리
if [ -f "enhanced_model_registry.json" ]; then
    mv "enhanced_model_registry.json" "auxiliary_models/"
    log_info "enhanced_model_registry.json → auxiliary_models/"
fi

if [ -f "environment_enhanced.yml" ]; then
    mv "environment_enhanced.yml" "backup_models/"
    log_info "environment_enhanced.yml → backup_models/"
fi

# 6. 최종 검증
echo ""
echo "🔍 최종 정리 결과 검증..."

cd "$AI_MODELS_ROOT"

# Step별 모델 수 확인
for step_dir in step_*/; do
    if [ -d "$step_dir" ]; then
        step_name=${step_dir%/}
        model_count=$(find "$step_dir" -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.bin" \) 2>/dev/null | wc -l)
        dir_count=$(find "$step_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
        size=$(du -sh "$step_dir" 2>/dev/null | cut -f1 || echo "0B")
        
        if [ "$model_count" -gt 0 ] || [ "$dir_count" -gt 0 ]; then
            echo "  ✅ $step_name: $model_count개 모델, $dir_count개 서브디렉토리 ($size)"
        else
            echo "  ⚪ $step_name: 비어있음"
        fi
    fi
done

# auxiliary_models 확인
if [ -d "auxiliary_models" ]; then
    aux_size=$(du -sh "auxiliary_models" 2>/dev/null | cut -f1 || echo "0B")
    aux_count=$(find "auxiliary_models" -type f | wc -l)
    echo "  📦 auxiliary_models: $aux_count개 파일 ($aux_size)"
fi

# huggingface_cache 확인
if [ -d "huggingface_cache" ]; then
    hf_size=$(du -sh "huggingface_cache" 2>/dev/null | cut -f1 || echo "0B")
    echo "  🤗 huggingface_cache: ($hf_size)"
fi

# backup_models 확인  
if [ -d "backup_models" ]; then
    backup_size=$(du -sh "backup_models" 2>/dev/null | cut -f1 || echo "0B")
    backup_count=$(find "backup_models" -type f | wc -l)
    echo "  💾 backup_models: $backup_count개 파일 ($backup_size)"
fi

# 7. 경로 설정 파일 업데이트
echo ""
echo "⚙️ 경로 설정 파일 업데이트 중..."

# Python 경로 설정 파일 재생성
cat > "$PROJECT_ROOT/backend/app/core/unified_model_paths.py" << 'EOF'
# app/core/unified_model_paths.py
"""
MyCloset AI - 통합된 Step별 모델 경로 설정 (완전 정리 완료)
최종 업데이트: $(date +'%Y-%m-%d %H:%M:%S')
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

# 실제 모델 위치 매핑 (완전 정리 완료 버전)
ACTUAL_MODEL_LOCATIONS = {
    # Step 01: Human Parsing
    "human_parsing": {
        "schp": STEP_MODEL_PATHS["step_01_human_parsing"] / "schp",
        "graphonomy": STEP_MODEL_PATHS["step_01_human_parsing"] / "graphonomy", 
        "densepose": STEP_MODEL_PATHS["step_01_human_parsing"] / "densepose"
    },
    
    # Step 02: Pose Estimation
    "pose_estimation": {
        "openpose": STEP_MODEL_PATHS["step_02_pose_estimation"] / "openpose",
        "mediapipe": STEP_MODEL_PATHS["step_02_pose_estimation"] / "mediapipe",
        "hrnet": STEP_MODEL_PATHS["step_02_pose_estimation"] / "hrnet"
    },
    
    # Step 03: Cloth Segmentation
    "cloth_segmentation": {
        "u2net": STEP_MODEL_PATHS["step_03_cloth_segmentation"] / "u2net",
        "sam": STEP_MODEL_PATHS["step_03_cloth_segmentation"] / "sam",
        "rembg": STEP_MODEL_PATHS["step_03_cloth_segmentation"] / "rembg"
    },
    
    # Step 04: Geometric Matching
    "geometric_matching": {
        "gmm": STEP_MODEL_PATHS["step_04_geometric_matching"] / "gmm",
        "tps": STEP_MODEL_PATHS["step_04_geometric_matching"] / "tps"
    },
    
    # Step 05: Cloth Warping
    "cloth_warping": {
        "tom": STEP_MODEL_PATHS["step_05_cloth_warping"] / "tom"
    },
    
    # Step 06: Virtual Fitting
    "virtual_fitting": {
        "ootdiffusion": STEP_MODEL_PATHS["step_06_virtual_fitting"] / "ootdiffusion",
        "stable_diffusion": STEP_MODEL_PATHS["step_06_virtual_fitting"] / "stable_diffusion",
        "hrviton": STEP_MODEL_PATHS["step_06_virtual_fitting"] / "hrviton",
        "idm_vton": STEP_MODEL_PATHS["step_06_virtual_fitting"] / "alternatives"
    },
    
    # Auxiliary Models  
    "auxiliary": {
        "clip": STEP_MODEL_PATHS["auxiliary_models"] / "clip",
        "sam": STEP_MODEL_PATHS["auxiliary_models"] / "sam",
        "vae": STEP_MODEL_PATHS["auxiliary_models"] / "vae",
        "text_encoders": STEP_MODEL_PATHS["auxiliary_models"] / "text_encoders"
    }
}

def get_step_path(step_name: str) -> Optional[Path]:
    """Step 이름으로 경로 반환"""
    return STEP_MODEL_PATHS.get(step_name)

def get_model_path(category: str, model_type: str) -> Optional[Path]:
    """카테고리와 모델 타입으로 실제 경로 반환"""
    if category in ACTUAL_MODEL_LOCATIONS:
        return ACTUAL_MODEL_LOCATIONS[category].get(model_type)
    return None

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

# 통합 정보 (완전 정리 완료)
INTEGRATION_INFO = {
    "integration_date": "$(date +'%Y-%m-%d %H:%M:%S')",
    "total_steps": len(STEP_MODEL_PATHS),
    "ai_models_root": str(AI_MODELS_ROOT),
    "integration_method": "step_based_classification_complete",
    "cleanup_status": "완전정리완료",
    "actual_model_locations": True
}
EOF

echo ""
echo "🎉 MyCloset AI 모델 완전 정리 완료!"
echo "====================================="
log_success "✅ 모든 모델이 Step별로 체계적으로 정리되었습니다"
log_success "✅ HuggingFace 캐시는 그대로 유지되었습니다"
log_success "✅ 심볼릭 링크가 정리되었습니다"  
log_success "✅ 경로 설정 파일이 업데이트되었습니다"

echo ""
echo "📁 최종 정리된 구조:"
echo "  ai_models/"
echo "  ├── step_01_human_parsing/schp/          (Self-Correction-Human-Parsing)"
echo "  ├── step_02_pose_estimation/openpose/    (OpenPose)" 
echo "  ├── step_03_cloth_segmentation/sam/      (SAM2)"
echo "  ├── step_05_cloth_warping/tom/           (cloth_warping)"
echo "  ├── step_06_virtual_fitting/"
echo "  │   ├── ootdiffusion/                    (OOTDiffusion)"
echo "  │   ├── stable_diffusion/                (StableVITON)"
echo "  │   └── alternatives/                    (IDM-VTON)"
echo "  ├── auxiliary_models/clip/               (clip-vit-base-patch32)"
echo "  ├── huggingface_cache/                   (기존 유지)"
echo "  └── backup_models/                       (백업 파일들)"

echo ""
echo "🚀 다음 단계:"
echo "1. ModelLoader에서 새 경로 구조 사용"
echo "2. 각 Step 클래스 경로 업데이트"
echo "3. AI 파이프라인 테스트 실행"

echo ""
log_info "완전 정리 완료: $(date +'%Y-%m-%d %H:%M:%S')"