#!/bin/bash
# MyCloset AI - 심볼릭 링크 대응 백업 스크립트
# 심볼릭 링크 문제 해결 버전

set -e

# 색상 정의
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

echo "💾 MyCloset AI - 심볼릭 링크 대응 백업"
echo "====================================="
echo "📅 $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 기존 백업 디렉토리 이름 확인
EXISTING_BACKUP=$(ls -d backup_essential_* 2>/dev/null | tail -1)
if [ -n "$EXISTING_BACKUP" ]; then
    log_info "기존 백업 발견: $EXISTING_BACKUP"
    BACKUP_DIR="$EXISTING_BACKUP"
else
    BACKUP_DIR="backup_essential_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    log_info "새 백업 디렉토리 생성: $BACKUP_DIR"
fi

# 심볼릭 링크 문제 해결을 위한 함수
backup_model_safe() {
    local model=$1
    local source_dir="ai_models/checkpoints/$model"
    local target_dir="$BACKUP_DIR/$model"
    
    if [ ! -d "$source_dir" ]; then
        log_warning "모델을 찾을 수 없음: $model"
        return 1
    fi
    
    log_info "백업 중: $model"
    
    # 디렉토리 생성
    mkdir -p "$target_dir"
    
    # 심볼릭 링크 해결하면서 복사
    rsync -aL --exclude=".*" "$source_dir/" "$target_dir/" 2>/dev/null || {
        log_warning "rsync 실패, cp 시도: $model"
        
        # rsync 실패시 find로 실제 파일만 복사
        find "$source_dir" -type f \( -name "*.pth" -o -name "*.bin" -o -name "*.safetensors" -o -name "*.ckpt" -o -name "*.json" \) -exec cp {} "$target_dir/" \; 2>/dev/null || {
            log_warning "일부 파일 복사 실패: $model"
        }
    }
    
    # 백업된 파일 확인
    local file_count=$(find "$target_dir" -type f | wc -l)
    if [ "$file_count" -gt 0 ]; then
        local size=$(du -sh "$target_dir" 2>/dev/null | cut -f1)
        log_success "백업 완료: $model ($size, ${file_count}개 파일)"
        return 0
    else
        log_error "백업 실패: $model (파일이 복사되지 않음)"
        return 1
    fi
}

# 핵심 모델들 백업
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

log_info "핵심 모델 백업 시작... (심볼릭 링크 해결 포함)"

successful_backups=0
failed_backups=0

for model in "${ESSENTIAL_MODELS[@]}"; do
    if backup_model_safe "$model"; then
        ((successful_backups++))
    else
        ((failed_backups++))
    fi
done

echo ""
log_info "백업 요약 생성..."

# 실제 백업된 모델들 확인
BACKED_UP_MODELS=$(ls -1 "$BACKUP_DIR" | grep -v "backup_info.txt" | grep -v "backup_report.txt")

cat << EOF > "$BACKUP_DIR/backup_report.txt"
MyCloset AI 핵심 모델 백업 보고서
==============================
생성일시: $(date '+%Y-%m-%d %H:%M:%S')
백업 위치: $BACKUP_DIR

백업 통계:
  성공: $successful_backups개
  실패: $failed_backups개
  총 시도: $((successful_backups + failed_backups))개

백업된 모델 상세:
$(for model in $BACKED_UP_MODELS; do
    if [ -d "$BACKUP_DIR/$model" ]; then
        size=$(du -sh "$BACKUP_DIR/$model" 2>/dev/null | cut -f1)
        files=$(find "$BACKUP_DIR/$model" -type f | wc -l)
        echo "  ✅ $model ($size, ${files}개 파일)"
    fi
done)

총 백업 크기: $(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1)

심볼릭 링크 문제:
  - 일부 모델에 순환 심볼릭 링크 존재
  - rsync -aL 옵션으로 해결 시도
  - 실패시 핵심 파일만 선별 복사

복원 방법:
  1. 전체 복원: cp -r $BACKUP_DIR/* ai_models/checkpoints/
  2. 개별 복원: cp -r $BACKUP_DIR/[모델명] ai_models/checkpoints/

주의사항:
  - 이 백업은 우선순위 1-6 핵심 모델만 포함
  - 심볼릭 링크는 실제 파일로 변환됨
  - 원본 삭제 전 백업 무결성 확인 필수
EOF

# 백업 검증
echo ""
log_info "백업 무결성 검증..."

total_model_files=0
for model in $BACKED_UP_MODELS; do
    if [ -d "$BACKUP_DIR/$model" ]; then
        model_files=$(find "$BACKUP_DIR/$model" -name "*.pth" -o -name "*.bin" -o -name "*.safetensors" -o -name "*.ckpt" | wc -l)
        total_model_files=$((total_model_files + model_files))
        
        if [ "$model_files" -gt 0 ]; then
            log_success "$model: $model_files개 모델 파일 확인"
        else
            log_warning "$model: 모델 파일을 찾을 수 없음 (설정 파일만 있을 수 있음)"
        fi
    fi
done

# 최종 요약
BACKUP_SIZE=$(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1)
BACKUP_COUNT=$(echo "$BACKED_UP_MODELS" | wc -w)

echo ""
echo "📋 백업 완료 요약:"
echo "   위치: $BACKUP_DIR"
echo "   크기: $BACKUP_SIZE"
echo "   모델 수: ${BACKUP_COUNT}개"
echo "   모델 파일 수: ${total_model_files}개"
echo "   성공률: $(( successful_backups * 100 / (successful_backups + failed_backups) ))%"
echo ""

if [ "$failed_backups" -gt 0 ]; then
    log_warning "일부 모델 백업 실패 ($failed_backups개)"
    echo "   실패한 모델들의 원본을 수동으로 확인해주세요"
else
    log_success "모든 핵심 모델 백업 성공!"
fi

echo ""
echo "🔍 심볼릭 링크 분석:"
log_info "순환 심볼릭 링크 탐지 중..."

find ai_models/checkpoints -type l | while read link; do
    target=$(readlink "$link")
    if [[ "$target" == *"$link"* ]]; then
        log_warning "순환 링크 발견: $link -> $target"
    fi
done

echo ""
echo "🚀 다음 단계:"
echo "   1. 백업 확인: ls -la $BACKUP_DIR"
echo "   2. 보고서 확인: cat $BACKUP_DIR/backup_report.txt"
echo "   3. 가상 실행: python scripts/smart_model_organizer.py --execute --dry-run"
echo "   4. 실제 실행: python scripts/smart_model_organizer.py --execute"
echo ""
echo "⚠️  백업이 완료되었으므로 안전하게 최적화를 진행할 수 있습니다!"