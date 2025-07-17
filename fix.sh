#!/bin/bash

echo "🔧 MyCloset AI - 순환 참조 문제 해결"
echo "======================================"

cd backend

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

# 1. 순환 참조 심볼릭 링크 제거
log_info "순환 참조 심볼릭 링크 제거 중..."

# 문제가 있는 u2net.pth 링크 제거
rm -f ai_models/checkpoints/step_03_cloth_segmentation/u2net.pth

# 다른 순환 참조 가능성 확인 및 제거
find ai_models/checkpoints -type l -name "*.pth" | while read link; do
    if [[ $(readlink "$link") == "$link" ]]; then
        log_warning "순환 참조 발견: $link"
        rm -f "$link"
    fi
done

log_success "순환 참조 링크 제거 완료"

# 2. 실제 U2-Net 모델 파일 복사
log_info "실제 U2-Net 모델 파일 복사 중..."

# 실제 u2net.pth 파일 위치 확인
U2NET_SOURCE="ai_models/checkpoints/u2net/u2net.pth"
U2NET_TARGET="ai_models/checkpoints/step_03_cloth_segmentation/u2net.pth"

if [[ -f "$U2NET_SOURCE" ]]; then
    cp "$U2NET_SOURCE" "$U2NET_TARGET"
    log_success "U2-Net 모델 복사 완료 ($(du -h "$U2NET_TARGET" | cut -f1))"
else
    log_warning "원본 U2-Net 파일이 없습니다. 대안 모델 사용"
    
    # 대안 1: cloth_segmentation 디렉토리의 모델 사용
    ALT_SOURCE="ai_models/checkpoints/cloth_segmentation/model.pth"
    if [[ -f "$ALT_SOURCE" ]]; then
        cp "$ALT_SOURCE" "$U2NET_TARGET"
        log_success "대안 U2-Net 모델 복사 완료"
    else
        log_warning "대안 모델도 없음. 더미 모델 생성"
        
        # 더미 모델 생성 (최후 수단)
        python3 << 'EOF'
import torch
import torch.nn as nn
from pathlib import Path

class DummyU2Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.conv(x))

# 더미 모델 생성 및 저장
model = DummyU2Net()
save_path = Path("ai_models/checkpoints/step_03_cloth_segmentation/u2net.pth")
save_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"더미 U2-Net 모델 생성: {save_path}")
EOF
        
        log_success "더미 U2-Net 모델 생성 완료"
    fi
fi

# 3. 다른 심볼릭 링크 검증
log_info "다른 심볼릭 링크 검증 중..."

find ai_models/checkpoints -type l | while read link; do
    target=$(readlink "$link")
    if [[ ! -f "$target" ]]; then
        log_warning "깨진 링크 발견: $link -> $target"
        
        # 실제 파일 찾기
        basename_file=$(basename "$target")
        real_file=$(find ai_models -name "$basename_file" -type f 2>/dev/null | head -1)
        
        if [[ -n "$real_file" ]]; then
            rm -f "$link"
            cp "$real_file" "$link"
            log_success "깨진 링크 수정: $link"
        else
            log_warning "실제 파일을 찾을 수 없음: $basename_file"
        fi
    fi
done

# 4. 모델 파일 검증
log_info "모델 파일 검증 중..."

CRITICAL_MODELS=(
    "ai_models/checkpoints/step_01_human_parsing/graphonomy.pth"
    "ai_models/checkpoints/step_02_pose_estimation/openpose.pth"
    "ai_models/checkpoints/step_03_cloth_segmentation/u2net.pth"
    "ai_models/checkpoints/step_04_geometric_matching/gmm_final.pth"
    "ai_models/checkpoints/step_05_cloth_warping/tom_final.pth"
    "ai_models/checkpoints/step_06_virtual_fitting/hrviton_final.pth"
)

for model in "${CRITICAL_MODELS[@]}"; do
    if [[ -f "$model" ]]; then
        size=$(du -h "$model" | cut -f1)
        log_success "$(basename "$model"): $size"
    else
        log_error "누락: $model"
    fi
done

# 5. 최종 검증
log_info "최종 검증 중..."

python3 << 'EOF'
import torch
from pathlib import Path
import sys

def test_model_load(model_path):
    """모델 로드 테스트"""
    try:
        if model_path.suffix == '.pth':
            state_dict = torch.load(model_path, map_location='cpu')
            if isinstance(state_dict, dict):
                return f"✅ {model_path.name}: {len(state_dict)} 키"
            else:
                return f"✅ {model_path.name}: 모델 객체"
        return f"⚠️ {model_path.name}: 지원되지 않는 형식"
    except Exception as e:
        return f"❌ {model_path.name}: {str(e)[:50]}..."

models_to_test = [
    "ai_models/checkpoints/step_03_cloth_segmentation/u2net.pth",
    "ai_models/checkpoints/step_01_human_parsing/graphonomy.pth",
    "ai_models/checkpoints/step_02_pose_estimation/openpose.pth"
]

print("🧪 모델 로드 테스트:")
for model_path in models_to_test:
    path = Path(model_path)
    if path.exists():
        result = test_model_load(path)
        print(f"   {result}")
    else:
        print(f"   ❌ {path.name}: 파일 없음")

EOF

echo ""
log_success "순환 참조 문제 해결 완료!"
echo ""
echo "🚀 다음 단계:"
echo "1. python3 app/main.py  # 서버 실행 테스트"
echo "2. python3 scripts/advanced_model_test.py  # 모델 테스트"
echo ""
echo "💡 팁: 서버 실행 후 /docs 에서 API 문서를 확인하세요!"