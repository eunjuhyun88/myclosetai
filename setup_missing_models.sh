#!/bin/bash
# 🔥 MyCloset AI - 부족한 모델 자동 다운로드 스크립트
# conda 환경 기반으로 필요한 AI 모델들을 체계적으로 다운로드

set -e  # 에러 발생시 스크립트 중단

# ========================================
# 🎯 기본 설정
# ========================================

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 절대 경로로 설정 (사용자 지정)
PROJECT_ROOT="/Users/gimdudeul/MVP/mycloset-ai"
AI_MODELS_DIR="$PROJECT_ROOT/backend/ai_models"

# 경로 존재 확인
if [[ ! -d "$PROJECT_ROOT" ]]; then
    echo -e "${RED}❌ 프로젝트 루트 경로를 찾을 수 없습니다: $PROJECT_ROOT${NC}"
    echo "경로를 확인해주세요."
    exit 1
fi

if [[ ! -d "$AI_MODELS_DIR" ]]; then
    echo -e "${YELLOW}⚠️ AI 모델 디렉토리가 없습니다. 생성 중: $AI_MODELS_DIR${NC}"
    mkdir -p "$AI_MODELS_DIR"
fi

echo -e "${BLUE}🔥 MyCloset AI - 부족한 모델 자동 다운로드${NC}"
echo -e "${BLUE}================================================${NC}"
echo "프로젝트 루트: $PROJECT_ROOT"
echo "AI 모델 디렉토리: $AI_MODELS_DIR"

# ========================================
# 🛠 conda 환경 체크 및 활성화
# ========================================

check_conda_env() {
    echo -e "\n${YELLOW}📋 conda 환경 체크 중...${NC}"
    
    if ! command -v conda &> /dev/null; then
        echo -e "${RED}❌ conda가 설치되지 않았습니다.${NC}"
        echo "Miniconda/Anaconda를 먼저 설치해주세요."
        exit 1
    fi
    
    # mycloset-ai 환경 체크
    if conda info --envs | grep -q "mycloset-ai"; then
        echo -e "${GREEN}✅ mycloset-ai conda 환경 발견${NC}"
        
        # 환경 활성화
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate mycloset-ai
        echo -e "${GREEN}✅ mycloset-ai 환경 활성화 완료${NC}"
    else
        echo -e "${YELLOW}⚠️ mycloset-ai 환경이 없습니다. 생성 중...${NC}"
        conda create -n mycloset-ai python=3.10 -y
        conda activate mycloset-ai
        echo -e "${GREEN}✅ mycloset-ai 환경 생성 및 활성화 완료${NC}"
    fi
}

# ========================================
# 🔍 현재 모델 상태 분석
# ========================================

analyze_current_models() {
    echo -e "\n${YELLOW}🔍 현재 모델 상태 분석 중...${NC}"
    echo "분석 경로: $AI_MODELS_DIR"
    
    # 각 단계별 모델 체크
    declare -A model_status
    
    # Step 01: Human Parsing
    if [[ -f "$AI_MODELS_DIR/checkpoints/step_01_human_parsing/exp-schp-201908301523-atr.pth" ]] || 
       find "$AI_MODELS_DIR" -name "*schp*atr*.pth" 2>/dev/null | grep -q .; then
        model_status["human_parsing"]="✅ 보유"
    else
        model_status["human_parsing"]="❌ 부족"
    fi
    
    # Step 02: Pose Estimation  
    if find "$AI_MODELS_DIR" -name "*body_pose_model*.pth" -o -name "*pose*.pth" 2>/dev/null | grep -q .; then
        model_status["pose_estimation"]="✅ 보유"
    else
        model_status["pose_estimation"]="❌ 부족"
    fi
    
    # Step 03: Cloth Segmentation
    if find "$AI_MODELS_DIR" -name "*sam_vit*.pth" -o -name "*u2net*.pth" 2>/dev/null | grep -q .; then
        model_status["cloth_segmentation"]="✅ 보유"
    else
        model_status["cloth_segmentation"]="❌ 부족"
    fi
    
    # Step 06: Virtual Fitting (가장 중요)
    if find "$AI_MODELS_DIR" -name "*ootd*" -type d 2>/dev/null | grep -q . ||
       find "$AI_MODELS_DIR" -name "*diffusion*.bin" -o -name "*diffusion*.safetensors" 2>/dev/null | grep -q .; then
        model_status["virtual_fitting"]="⚠️ 불완전"
    else
        model_status["virtual_fitting"]="❌ 부족"
    fi
    
    # 현재 존재하는 파일들 나열
    echo -e "\n${BLUE}📁 현재 존재하는 주요 모델 파일들:${NC}"
    find "$AI_MODELS_DIR" -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" 2>/dev/null | head -10 | while read file; do
        size=$(du -h "$file" 2>/dev/null | cut -f1 || echo "?")
        echo "  $(basename "$file"): $size"
    done
    
    # 결과 출력
    echo -e "\n${BLUE}📊 모델 상태 요약:${NC}"
    for step in "${!model_status[@]}"; do
        echo -e "  $step: ${model_status[$step]}"
    done
}

# ========================================
# 📦 필수 Python 패키지 설치
# ========================================

install_dependencies() {
    echo -e "\n${YELLOW}📦 필수 Python 패키지 설치 중...${NC}"
    
    # 기본 패키지
    pip install --upgrade pip
    pip install requests tqdm
    
    # Hugging Face Hub (모델 다운로드용)
    pip install huggingface_hub
    
    # Git LFS (대용량 파일용)
    if ! command -v git-lfs &> /dev/null; then
        echo -e "${YELLOW}⚠️ git-lfs 설치 필요${NC}"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install git-lfs 2>/dev/null || echo "Homebrew로 git-lfs 설치 실패"
        fi
    fi
    
    git lfs install
    
    echo -e "${GREEN}✅ 필수 패키지 설치 완료${NC}"
}

# ========================================
# 🎯 Phase 1: 즉시 필요한 핵심 모델
# ========================================

download_critical_models() {
    echo -e "\n${RED}🚨 Phase 1: 즉시 필요한 핵심 모델 다운로드${NC}"
    echo "다운로드 경로: $AI_MODELS_DIR"
    
    # 기본 디렉토리 구조 생성
    mkdir -p "$AI_MODELS_DIR/checkpoints/step_06_virtual_fitting"
    mkdir -p "$AI_MODELS_DIR/checkpoints/step_01_human_parsing" 
    mkdir -p "$AI_MODELS_DIR/checkpoints/step_02_pose_estimation"
    mkdir -p "$AI_MODELS_DIR/checkpoints/step_03_cloth_segmentation"
    
    # 1. OOTDiffusion 핵심 체크포인트
    echo -e "\n${YELLOW}📥 OOTDiffusion 모델 다운로드 중...${NC}"
    
    cd "$AI_MODELS_DIR/checkpoints/step_06_virtual_fitting"
    
    # Hugging Face CLI 또는 git으로 레포지토리 클론
    if [[ ! -d "OOTDiffusion" ]]; then
        echo "OOTDiffusion 레포지토리 클론 중... (이 작업은 시간이 걸릴 수 있습니다)"
        
        # Git LFS가 있는 경우
        if command -v git-lfs &> /dev/null; then
            git clone https://huggingface.co/levihsu/OOTDiffusion
            
            if [[ -d "OOTDiffusion" ]]; then
                cd OOTDiffusion
                echo "중요한 체크포인트 파일들 다운로드 중..."
                
                # 선별적 LFS pull
                git lfs pull --include="checkpoints/ootd/ootd_dc/checkpoint-36000/*" 2>/dev/null || echo "⚠️ ootd_dc 다운로드 건너뜀"
                git lfs pull --include="checkpoints/ootd/ootd_hd/checkpoint-36000/*" 2>/dev/null || echo "⚠️ ootd_hd 다운로드 건너뜀"
                git lfs pull --include="checkpoints/ootd/vae/*" 2>/dev/null || echo "⚠️ vae 다운로드 건너뜀"
                git lfs pull --include="checkpoints/ootd/text_encoder/*" 2>/dev/null || echo "⚠️ text_encoder 다운로드 건너뜀"
                
                cd ..
                echo -e "${GREEN}✅ OOTDiffusion 기본 구조 다운로드 완료${NC}"
            fi
        else
            echo -e "${YELLOW}⚠️ git-lfs가 없어 핵심 파일만 직접 다운로드합니다${NC}"
            
            # 핵심 파일만 직접 다운로드
            mkdir -p OOTDiffusion/checkpoints/ootd/ootd_dc/checkpoint-36000
            mkdir -p OOTDiffusion/checkpoints/ootd/ootd_hd/checkpoint-36000
            
            # 직접 wget으로 핵심 파일들
            echo "핵심 UNet 모델 다운로드 중..."
            wget -O OOTDiffusion/checkpoints/ootd/ootd_dc/checkpoint-36000/pytorch_model.bin \
                "https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/ootd/ootd_dc/checkpoint-36000/pytorch_model.bin" \
                2>/dev/null || echo "⚠️ ootd_dc 다운로드 실패"
                
            wget -O OOTDiffusion/checkpoints/ootd/ootd_hd/checkpoint-36000/pytorch_model.bin \
                "https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/ootd/ootd_hd/checkpoint-36000/pytorch_model.bin" \
                2>/dev/null || echo "⚠️ ootd_hd 다운로드 실패"
        fi
    else
        echo -e "${GREEN}✅ OOTDiffusion 이미 존재${NC}"
    fi
    
    # 2. Stable Diffusion Inpainting (백업용)
    echo -e "\n${YELLOW}📥 Stable Diffusion Inpainting 다운로드 중...${NC}"
    
    mkdir -p stable_diffusion_inpainting
    cd stable_diffusion_inpainting
    
    # 핵심 UNet 모델만
    if [[ ! -f "diffusion_pytorch_model.bin" ]]; then
        echo "Stable Diffusion UNet 다운로드 중..."
        wget -O diffusion_pytorch_model.bin \
            "https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/unet/diffusion_pytorch_model.bin" \
            2>/dev/null || echo "⚠️ Stable Diffusion 다운로드 실패"
    fi
    
    # 3. 기본 Human Parsing 모델 (즉시 필요)
    echo -e "\n${YELLOW}📥 기본 Human Parsing 모델 확인 중...${NC}"
    
    cd "$AI_MODELS_DIR/checkpoints/step_01_human_parsing"
    
    # 기존 파일 체크
    if ! find "$AI_MODELS_DIR" -name "*schp*atr*.pth" 2>/dev/null | grep -q .; then
        echo "Human Parsing 모델 다운로드 중..."
        wget -O exp-schp-201908301523-atr.pth \
            "https://drive.google.com/uc?export=download&id=1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH" \
            2>/dev/null || echo "⚠️ Human Parsing 다운로드 실패 - 수동 다운로드 필요"
    else
        echo -e "${GREEN}✅ Human Parsing 모델 이미 존재${NC}"
    fi
    
    echo -e "${GREEN}✅ Phase 1 핵심 모델 다운로드 완료${NC}"
}

# ========================================
# 🔥 Phase 2: 중요한 보조 모델
# ========================================

download_important_models() {
    echo -e "\n${YELLOW}🔥 Phase 2: 중요한 보조 모델 다운로드${NC}"
    
    # 1. Human Parsing 모델 업그레이드
    echo -e "\n${YELLOW}📥 Human Parsing 모델 다운로드 중...${NC}"
    
    mkdir -p "$AI_MODELS_DIR/checkpoints/step_01_human_parsing"
    cd "$AI_MODELS_DIR/checkpoints/step_01_human_parsing"
    
    if [[ ! -f "exp-schp-201908301523-atr.pth" ]]; then
        # 직접 다운로드 (GitHub Release)
        wget -O exp-schp-201908301523-atr.pth \
            "https://github.com/PeikeLi/Self-Correction-Human-Parsing/releases/download/v1.0/exp-schp-201908301523-atr.pth" \
            || echo "⚠️ Human Parsing 모델 다운로드 실패 - 수동 다운로드 필요"
    fi
    
    # 2. 개선된 Pose Estimation
    echo -e "\n${YELLOW}📥 OpenPose 모델 다운로드 중...${NC}"
    
    mkdir -p "$AI_MODELS_DIR/checkpoints/step_02_pose_estimation"
    cd "$AI_MODELS_DIR/checkpoints/step_02_pose_estimation"
    
    if [[ ! -f "body_pose_model.pth" ]]; then
        # OpenPose Body Model
        wget -O pose_iter_584000.caffemodel \
            "https://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel" \
            || echo "⚠️ OpenPose 모델 다운로드 실패"
        
        # PyTorch 변환 버전도 시도
        wget -O body_pose_model.pth \
            "https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/releases/download/v0.1/checkpoint_iter_370000.pth" \
            || echo "⚠️ PyTorch Pose 모델 다운로드 실패"
    fi
    
    # 3. SAM2 for Better Segmentation
    echo -e "\n${YELLOW}📥 SAM2 모델 다운로드 중...${NC}"
    
    mkdir -p "$AI_MODELS_DIR/checkpoints/step_03_cloth_segmentation"
    cd "$AI_MODELS_DIR/checkpoints/step_03_cloth_segmentation"
    
    if [[ ! -f "sam2_hiera_large.pt" ]]; then
        wget -O sam2_hiera_large.pt \
            "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt" \
            || echo "⚠️ SAM2 모델 다운로드 실패"
    fi
    
    echo -e "${GREEN}✅ Phase 2 중요 모델 다운로드 완료${NC}"
}

# ========================================
# ⚡ Phase 3: 성능 향상 모델 (선택적)
# ========================================

download_enhancement_models() {
    echo -e "\n${BLUE}⚡ Phase 3: 성능 향상 모델 다운로드 (선택적)${NC}"
    
    read -p "성능 향상 모델도 다운로드하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Phase 3 건너뜀"
        return
    fi
    
    # 1. Geometric Matching
    echo -e "\n${YELLOW}📥 Geometric Matching 모델 다운로드 중...${NC}"
    
    mkdir -p "$AI_MODELS_DIR/checkpoints/step_04_geometric_matching"
    cd "$AI_MODELS_DIR/checkpoints/step_04_geometric_matching"
    
    # CP-VTON GMM 모델
    if [[ ! -f "gmm_final.pth" ]]; then
        wget -O gmm_final.pth \
            "https://github.com/sergeywong/cp-vton/releases/download/v1.0/gmm_final.pth" \
            || echo "⚠️ GMM 모델 다운로드 실패"
    fi
    
    # 2. Cloth Warping  
    echo -e "\n${YELLOW}📥 Cloth Warping 모델 다운로드 중...${NC}"
    
    mkdir -p "$AI_MODELS_DIR/checkpoints/step_05_cloth_warping"
    cd "$AI_MODELS_DIR/checkpoints/step_05_cloth_warping"
    
    if [[ ! -f "tom_final.pth" ]]; then
        wget -O tom_final.pth \
            "https://github.com/sergeywong/cp-vton/releases/download/v1.0/tom_final.pth" \
            || echo "⚠️ TOM 모델 다운로드 실패"
    fi
    
    echo -e "${GREEN}✅ Phase 3 성능 향상 모델 다운로드 완료${NC}"
}

# ========================================
# 🔧 모델 검증 및 정리
# ========================================

verify_and_organize() {
    echo -e "\n${BLUE}🔧 모델 검증 및 정리 중...${NC}"
    
    cd "$AI_MODELS_DIR"
    
    # 다운로드된 모델들 크기 체크
    echo -e "\n${YELLOW}📊 다운로드된 모델 크기:${NC}"
    
    # 각 단계별로 체크
    for step_dir in checkpoints/step_*; do
        if [[ -d "$step_dir" ]]; then
            echo -e "\n  ${BLUE}$(basename "$step_dir"):${NC}"
            find "$step_dir" -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" 2>/dev/null | while read file; do
                size=$(du -h "$file" 2>/dev/null | cut -f1 || echo "?")
                echo "    $(basename "$file"): $size"
            done
        fi
    done
    
    # 전체 용량 체크
    total_size=$(du -sh . 2>/dev/null | cut -f1 || echo "계산 불가")
    echo -e "\n${BLUE}💾 전체 AI 모델 용량: $total_size${NC}"
    
    # 권한 설정
    echo -e "\n${YELLOW}🔑 파일 권한 설정 중...${NC}"
    find . -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" | xargs chmod 644 2>/dev/null || true
    find . -type d | xargs chmod 755 2>/dev/null || true
    
    echo -e "${GREEN}✅ 모델 검증 및 정리 완료${NC}"
}

# ========================================
# 📊 최종 결과 보고
# ========================================

generate_report() {
    echo -e "\n${GREEN}🎉 모델 다운로드 완료 보고서${NC}"
    echo -e "${GREEN}================================${NC}"
    echo "설치 경로: $AI_MODELS_DIR"
    
    # 각 단계별 상태 재체크
    echo -e "\n${BLUE}📋 Step별 모델 상태:${NC}"
    
    steps=(
        "step_01_human_parsing:Human Parsing"
        "step_02_pose_estimation:Pose Estimation"  
        "step_03_cloth_segmentation:Cloth Segmentation"
        "step_04_geometric_matching:Geometric Matching"
        "step_05_cloth_warping:Cloth Warping"
        "step_06_virtual_fitting:Virtual Fitting"
        "step_07_post_processing:Post Processing"
        "step_08_quality_assessment:Quality Assessment"
    )
    
    for step_info in "${steps[@]}"; do
        step_dir=$(echo $step_info | cut -d: -f1)
        step_name=$(echo $step_info | cut -d: -f2)
        
        if [[ -d "$AI_MODELS_DIR/checkpoints/$step_dir" ]] && [[ $(find "$AI_MODELS_DIR/checkpoints/$step_dir" -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" 2>/dev/null | wc -l) -gt 0 ]]; then
            echo -e "  ✅ $step_name"
        else
            echo -e "  ⚠️ $step_name (부분적/없음)"
        fi
    done
    
    # 실제 다운로드된 주요 파일들
    echo -e "\n${BLUE}📁 주요 다운로드된 파일들:${NC}"
    find "$AI_MODELS_DIR" -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" 2>/dev/null | head -15 | while read file; do
        size=$(du -h "$file" 2>/dev/null | cut -f1 || echo "?")
        rel_path=$(echo "$file" | sed "s|$AI_MODELS_DIR/||")
        echo "  $rel_path ($size)"
    done
    
    # 다음 단계 안내
    echo -e "\n${YELLOW}🚀 다음 단계:${NC}"
    echo "1. conda activate mycloset-ai"
    echo "2. cd /Users/gimdudeul/MVP/mycloset-ai"  
    echo "3. python backend/app/ai_pipeline/utils/auto_model_detector.py"
    echo "4. python -m backend.main  # 백엔드 서버 실행"
    
    # 용량 정보
    total_size=$(du -sh "$AI_MODELS_DIR" 2>/dev/null | cut -f1 || echo "계산 불가")
    echo -e "\n${BLUE}💾 총 AI 모델 용량: $total_size${NC}"
    
    # 추가 권장 사항
    echo -e "\n${YELLOW}💡 추가 권장 사항:${NC}"
    echo "- 부족한 모델이 있다면 해당 단계의 공식 문서를 확인하세요"
    echo "- 큰 모델(OOTDiffusion)은 다운로드 시간이 오래 걸릴 수 있습니다"
    echo "- 네트워크 문제로 실패한 모델은 나중에 개별 다운로드하세요"
}

# ========================================
# 🚀 메인 실행 함수
# ========================================

main() {
    echo -e "${BLUE}🚀 모델 다운로드 프로세스 시작${NC}"
    
    # 단계별 실행
    check_conda_env
    analyze_current_models
    install_dependencies
    download_critical_models
    download_important_models
    download_enhancement_models
    verify_and_organize
    generate_report
    
    echo -e "\n${GREEN}🎉 모든 모델 다운로드 프로세스 완료!${NC}"
}

# 스크립트 실행
main "$@"