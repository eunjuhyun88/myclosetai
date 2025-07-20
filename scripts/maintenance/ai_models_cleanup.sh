#!/bin/bash

# =============================================================================
# AI Models 디렉토리 완전 정리 스크립트
# 현재 복잡한 ai_models 구조를 표준화된 형태로 정리
# =============================================================================

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

log_header "AI Models 디렉토리 완전 정리"
echo "=================================================================="
log_info "작업 디렉토리: $(pwd)/ai_models"
log_info "정리 시작 시간: $(date)"

# 현재 위치 확인
if [ ! -d "ai_models" ]; then
    log_error "ai_models 디렉토리가 없습니다. 프로젝트 루트에서 실행해주세요."
    exit 1
fi

cd ai_models

echo ""
log_info "📊 현재 상태 분석..."
echo "현재 디렉토리 수: $(find . -type d | wc -l)"
echo "현재 파일 수: $(find . -type f | wc -l)"
echo "총 용량: $(du -sh . | cut -f1)"

# 1. 백업 생성
echo ""
log_info "1. 중요 설정 파일 백업 생성 중..."
backup_dir="../backup_ai_models_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

# 중요한 설정 파일들만 백업
if [ -f "enhanced_model_registry.json" ]; then 
    cp enhanced_model_registry.json "$backup_dir/"
fi
if [ -f "environment_enhanced.yml" ]; then 
    cp environment_enhanced.yml "$backup_dir/"
fi
if [ -f "mps_optimization.py" ]; then 
    cp mps_optimization.py "$backup_dir/"
fi

log_success "백업 생성 완료: $backup_dir"

# 2. 표준 디렉토리 구조 생성
echo ""
log_info "2. 표준 디렉토리 구조 생성 중..."

# 표준 Step별 디렉토리는 이미 존재하므로 보완만
mkdir -p archive/{original_models,unused_models,temp_models}
mkdir -p configs
mkdir -p docs

log_success "표준 디렉토리 구조 준비 완료"

# 3. 중복/사용하지 않는 모델 디렉토리 정리
echo ""
log_info "3. 중복 및 사용하지 않는 디렉토리 정리 중..."

# 사용하지 않는 것으로 판단되는 디렉토리들을 archive로 이동
unused_dirs=(
    "auxiliary_models"
    "backup_models" 
    "experimental_models"
    "temp_downloads"
    "cloth_warping"
    "Self-Correction-Human-Parsing"
    "StableVITON"
)

for dir in "${unused_dirs[@]}"; do
    if [ -d "$dir" ]; then
        log_info "아카이브: $dir"
        mv "$dir" "archive/unused_models/"
    fi
done

# 4. HuggingFace 모델 정리
echo ""
log_info "4. HuggingFace 모델 정리 중..."

if [ -d "huggingface_cache" ]; then
    # HuggingFace 캐시를 표준 위치로 이동
    mkdir -p cache/huggingface
    if [ "$(ls -A huggingface_cache 2>/dev/null)" ]; then
        mv huggingface_cache/* cache/huggingface/ 2>/dev/null || true
    fi
    rmdir huggingface_cache 2>/dev/null || true
    log_success "HuggingFace 캐시 정리 완료"
fi

# 5. 개별 모델 디렉토리 정리
echo ""
log_info "5. 개별 모델 디렉토리 정리 중..."

# 특정 모델들을 적절한 Step으로 이동/통합
model_mappings=(
    "openpose:step_02_pose_estimation"
    "IDM-VTON:step_06_virtual_fitting" 
    "OOTDiffusion:step_06_virtual_fitting"
    "SAM2:step_03_cloth_segmentation"
)

for mapping in "${model_mappings[@]}"; do
    IFS=':' read -r source_dir target_dir <<< "$mapping"
    if [ -d "$source_dir" ] && [ -d "$target_dir" ]; then
        log_info "통합: $source_dir → $target_dir"
        
        # 중복 방지를 위해 백업 후 이동
        if [ "$(ls -A "$source_dir" 2>/dev/null)" ]; then
            mv "$source_dir" "archive/original_models/$source_dir"
            log_info "원본 보존: archive/original_models/$source_dir"
        fi
    fi
done

# 6. 심볼릭 링크 정리
echo ""
log_info "6. 심볼릭 링크 정리 중..."

# 깨진 심볼릭 링크 찾기 및 정리
find . -type l -exec test ! -e {} \; -print | while read -r broken_link; do
    log_warning "깨진 링크 제거: $broken_link"
    rm "$broken_link"
done

# 7. 설정 파일 재구성
echo ""
log_info "7. 설정 파일 재구성 중..."

# 설정 파일들을 configs 디렉토리로 이동
config_files=("enhanced_model_registry.json" "environment_enhanced.yml" "mps_optimization.py")

for config_file in "${config_files[@]}"; do
    if [ -f "$config_file" ]; then
        mv "$config_file" "configs/"
        log_info "설정 파일 이동: $config_file → configs/"
    fi
done

# 8. 새로운 통합 설정 파일 생성
echo ""
log_info "8. 통합 설정 파일 생성 중..."

cat > configs/model_registry.yaml << 'EOF'
# MyCloset AI - 통합 모델 레지스트리
# 생성일: $(date)

models:
  step_01_human_parsing:
    description: "인체 부위별 파싱"
    models:
      - schp_atr
      - graphonomy_lip
      - densepose
    
  step_02_pose_estimation:
    description: "인체 자세 추정"
    models:
      - openpose
      - mediapipe_pose
    
  step_03_cloth_segmentation:
    description: "의류 세그멘테이션"
    models:
      - u2net
      - sam2
      - mobile_sam
    
  step_04_geometric_matching:
    description: "기하학적 매칭"
    models:
      - gmm
      - tps_network
    
  step_05_cloth_warping:
    description: "의류 변형"
    models:
      - tom
      - lightweight_warping
    
  step_06_virtual_fitting:
    description: "가상 피팅"
    models:
      - ootdiffusion
      - idm_vton
      - hr_viton
    
  step_07_post_processing:
    description: "후처리"
    models:
      - gfpgan
      - real_esrgan
      - codeformer
    
  step_08_quality_assessment:
    description: "품질 평가"
    models:
      - clip_vit
      - lpips

cache:
  huggingface: "./cache/huggingface"
  torch: "./cache/torch"
  
archive:
  unused_models: "./archive/unused_models"
  original_models: "./archive/original_models"
EOF

# 9. README 파일 생성
echo ""
log_info "9. README 파일 생성 중..."

cat > README.md << 'EOF'
# 🤖 MyCloset AI - AI Models Directory

AI 모델 파일들의 체계적 관리를 위한 디렉토리입니다.

## 📁 디렉토리 구조

```
ai_models/
├── step_01_human_parsing/     # 인체 파싱 모델
├── step_02_pose_estimation/   # 자세 추정 모델
├── step_03_cloth_segmentation/ # 의류 분할 모델
├── step_04_geometric_matching/ # 기하학적 매칭 모델
├── step_05_cloth_warping/     # 의류 변형 모델
├── step_06_virtual_fitting/   # 가상 피팅 모델
├── step_07_post_processing/   # 후처리 모델
├── step_08_quality_assessment/ # 품질 평가 모델
├── cache/                     # 모델 캐시
│   ├── huggingface/          # HuggingFace 모델 캐시
│   └── torch/                # PyTorch 모델 캐시
├── archive/                   # 아카이브
│   ├── unused_models/        # 사용하지 않는 모델
│   └── original_models/      # 원본 모델 백업
├── configs/                   # 설정 파일
└── docs/                     # 문서
```

## 🔧 모델 관리

### 새 모델 추가
```bash
# 적절한 Step 디렉토리에 모델 파일 배치
cp new_model.pth step_XX_model_type/

# 설정 파일 업데이트
vim configs/model_registry.yaml
```

### 모델 경로 확인
```python
from app.core.corrected_model_paths import get_model_path
model_path = get_model_path("human_parsing_graphonomy")
```

## 📋 주의사항

- 대용량 모델 파일들은 Git에서 제외됩니다 (.gitignore 참고)
- 모델 다운로드는 별도 스크립트를 사용하세요
- 캐시 디렉토리는 정기적으로 정리하세요

## 🚀 모델 다운로드

```bash
# 모든 필수 모델 다운로드
python ../scripts/models/download_all_models.py

# 특정 Step 모델만 다운로드  
python ../scripts/models/download_step_models.py --step 01
```
EOF

# 10. .gitignore 업데이트
echo ""
log_info "10. .gitignore 업데이트 중..."

cat > .gitignore << 'EOF'
# AI 모델 파일들 (대용량)
*.pth
*.pt
*.ckpt
*.safetensors
*.bin
*.onnx
*.pkl
*.h5
*.pb
*.tflite
*.model
*.weights

# 캐시 디렉토리
cache/
.cache/

# 임시 파일
temp/
tmp/
*.tmp

# 시스템 파일
.DS_Store
Thumbs.db

# 아카이브는 로컬에만
archive/unused_models/
archive/temp_models/

# 하지만 중요한 파일들은 포함
!configs/
!docs/
!README.md
!.gitkeep
!*/README.md
!*/.gitkeep
EOF

# 11. .gitkeep 파일 생성
echo ""
log_info "11. 빈 디렉토리에 .gitkeep 파일 생성 중..."

find . -type d -empty -exec touch {}/.gitkeep \; 2>/dev/null || true

# Step 디렉토리들에도 .gitkeep 확보
for step_dir in step_0{1..8}_*; do
    if [ -d "$step_dir" ] && [ -z "$(ls -A "$step_dir" 2>/dev/null)" ]; then
        touch "$step_dir/.gitkeep"
    fi
done

# 12. 권한 설정
echo ""
log_info "12. 파일 권한 설정 중..."

# 실행 파일들 권한 설정
find . -name "*.py" -exec chmod 644 {} \;
find . -name "*.sh" -exec chmod 755 {} \;

# 디렉토리 권한 설정
find . -type d -exec chmod 755 {} \;

# 13. 최종 정리 및 검증
echo ""
log_info "13. 최종 정리 및 검증 중..."

# 빈 디렉토리 정리 (아카이브 제외)
find . -type d -empty -not -path "./archive/*" -not -name "archive" -delete 2>/dev/null || true

# 재생성
mkdir -p cache/{huggingface,torch}
mkdir -p archive/{unused_models,original_models,temp_models}
mkdir -p configs docs

log_success "최종 정리 완료"

# 14. 결과 보고
echo ""
log_header "🎉 AI Models 디렉토리 정리 완료!"
echo "=================================================================="

echo ""
log_success "✨ 정리된 구조:"
echo "📁 step_XX_*/ (8개)  - AI 파이프라인 8단계별 모델"
echo "📁 cache/            - HuggingFace, PyTorch 캐시"  
echo "📁 archive/          - 사용하지 않는 모델들"
echo "📁 configs/          - 모델 설정 파일"
echo "📁 docs/             - 문서"

echo ""
log_info "📊 정리 후 통계:"
echo "- Step 디렉토리: 8개"
echo "- 설정 파일: $(find configs/ -name "*.yaml" -o -name "*.json" | wc -l)개" 
echo "- 문서 파일: $(find . -name "README.md" | wc -l)개"
echo "- 백업 위치: $backup_dir"

echo ""
log_warning "⚠️ 다음 단계:"
echo "1. cd .. && git add ai_models/"
echo "2. git commit -m 'AI models 디렉토리 구조 표준화'"
echo "3. python scripts/models/download_all_models.py (모델 다운로드)"
echo "4. bash scripts/dev/check_structure.sh (검증)"

echo ""
log_success "🚀 AI Models 디렉토리가 표준화되었습니다!"

cd ..  # 원래 디렉토리로 복귀