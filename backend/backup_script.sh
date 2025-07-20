#!/bin/bash
# MyCloset AI - 핵심 모델 백업 스크립트
# 우선순위 1-6 모델만 선별 백업

set -e

# 색상 정의
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }

echo "💾 MyCloset AI - 핵심 모델 백업"
echo "==============================="
echo "📅 $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 백업 디렉토리 생성
BACKUP_DIR="backup_essential_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

log_info "백업 디렉토리 생성: $BACKUP_DIR"

# 핵심 모델들 (우선순위 1-6) 백업
ESSENTIAL_MODELS=(
    # 우선순위 1: 가상 피팅 핵심
    "ootdiffusion"
    "ootdiffusion_hf"
    
    # 우선순위 2-6: 파이프라인 핵심
    "human_parsing"
    "step_01_human_parsing"
    "openpose" 
    "step_02_pose_estimation"
    "u2net"
    "step_03_cloth_segmentation"
    "step_04_geometric_matching"
    "step_05_cloth_warping"
)

log_info "핵심 모델 백업 시작..."

for model in "${ESSENTIAL_MODELS[@]}"; do
    if [ -d "ai_models/checkpoints/$model" ]; then
        log_info "백업 중: $model"
        cp -r "ai_models/checkpoints/$model" "$BACKUP_DIR/"
        
        # 크기 확인
        size=$(du -sh "$BACKUP_DIR/$model" | cut -f1)
        log_success "백업 완료: $model ($size)"
    else
        log_warning "모델을 찾을 수 없음: $model"
    fi
done

# 백업 요약
echo ""
log_info "백업 요약 생성..."

cat << EOF > "$BACKUP_DIR/backup_info.txt"
MyCloset AI 핵심 모델 백업
========================
생성일시: $(date '+%Y-%m-%d %H:%M:%S')
백업 위치: $BACKUP_DIR

백업된 모델:
$(ls -la "$BACKUP_DIR" | grep "^d" | awk '{print $9}' | grep -v "^\.$\|^\.\.$" | while read dir; do echo "  ✅ $dir"; done)

총 백업 크기: $(du -sh "$BACKUP_DIR" | cut -f1)

복원 방법:
  cp -r $BACKUP_DIR/* ai_models/checkpoints/

주의사항:
  - 이 백업은 우선순위 1-6 핵심 모델만 포함
  - 전체 복원이 아닌 선별적 복원 권장
  - 원본 삭제 전 백업 무결성 확인 필수
EOF

# 백업 크기 확인
BACKUP_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
log_success "백업 완료! 총 크기: $BACKUP_SIZE"

echo ""
echo "📋 백업 정보:"
echo "   위치: $BACKUP_DIR"
echo "   크기: $BACKUP_SIZE"
echo "   모델 수: $(ls -1 "$BACKUP_DIR" | grep -v "backup_info.txt" | wc -l)개"
echo ""
echo "🔐 백업 검증:"
log_info "백업 무결성 확인 중..."

# 각 모델의 주요 파일 확인
for model in "${ESSENTIAL_MODELS[@]}"; do
    if [ -d "$BACKUP_DIR/$model" ]; then
        file_count=$(find "$BACKUP_DIR/$model" -name "*.pth" -o -name "*.bin" -o -name "*.safetensors" | wc -l)
        if [ "$file_count" -gt 0 ]; then
            log_success "$model: $file_count개 모델 파일 확인"
        else
            log_warning "$model: 모델 파일을 찾을 수 없음"
        fi
    fi
done

echo ""
echo "🚀 다음 단계:"
echo "   1. 백업 검증: ls -la $BACKUP_DIR"
echo "   2. 가상 실행: python scripts/smart_model_organizer.py --execute --dry-run"
echo "   3. 실제 실행: python scripts/smart_model_organizer.py --execute"
echo ""
echo "⚠️  백업이 완료되었으므로 안전하게 최적화를 진행할 수 있습니다!"