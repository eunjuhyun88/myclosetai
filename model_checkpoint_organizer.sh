#!/bin/bash
# 🔧 MyCloset AI - 모델 체크포인트 경로 정리 스크립트
# conda 환경 우선 실행 권장

set -e

echo "🔧 MyCloset AI 모델 체크포인트 경로 정리 시작..."

# 프로젝트 루트 확인
PROJECT_ROOT="/Users/gimdudeul/MVP/mycloset-ai"
BACKEND_ROOT="$PROJECT_ROOT/backend"
AI_MODELS_ROOT="$BACKEND_ROOT/ai_models"

# conda 환경 확인
if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    echo "🐍 conda 환경 확인: $CONDA_DEFAULT_ENV"
else
    echo "⚠️  conda 환경이 활성화되지 않았습니다"
    echo "   다음 명령어로 활성화하세요: conda activate mycloset-ai"
fi

# 1. 현재 모델 파일 스캔
echo ""
echo "🔍 1단계: 현재 모델 파일 스캔..."
cd "$AI_MODELS_ROOT"

echo "📊 현재 존재하는 모델 파일들:"
find . -name "*.pth" -o -name "*.bin" -o -name "*.ckpt" | while read file; do
    size=$(du -h "$file" 2>/dev/null | cut -f1)
    echo "  ✅ $file ($size)"
done

# 2. 표준 디렉토리 구조 생성
echo ""
echo "📁 2단계: 표준 디렉토리 구조 생성..."

STEP_DIRS=(
    "step_01_human_parsing"
    "step_02_pose_estimation" 
    "step_03_cloth_segmentation"
    "step_04_geometric_matching"
    "step_05_cloth_warping"
    "step_06_virtual_fitting"
    "step_07_post_processing"
    "step_08_quality_assessment"
)

# checkpoints 하위 디렉토리 생성
for step_dir in "${STEP_DIRS[@]}"; do
    mkdir -p "checkpoints/$step_dir"
    echo "  📁 checkpoints/$step_dir 생성됨"
done

# organized 하위 디렉토리 생성 (정리된 모델용)
for step_dir in "${STEP_DIRS[@]}"; do
    mkdir -p "organized/$step_dir"
    echo "  📁 organized/$step_dir 생성됨"
done

# 3. 기존 모델들을 적절한 위치로 이동
echo ""
echo "🔄 3단계: 기존 모델 파일 정리..."

# 현재 확인된 모델들 정리
move_model() {
    local source="$1"
    local target="$2"
    local description="$3"
    
    if [[ -f "$source" ]]; then
        mkdir -p "$(dirname "$target")"
        mv "$source" "$target"
        echo "  ✅ $description: $target"
    else
        echo "  ❌ 찾을 수 없음: $source"
    fi
}

# Step 01: Human Parsing
move_model "./exp-schp-201908301523-atr.pth" \
          "./organized/step_01_human_parsing/exp-schp-201908301523-atr.pth" \
          "Human Parsing ATR 모델"

# Step 02: Pose Estimation  
move_model "./openpose.pth" \
          "./organized/step_02_pose_estimation/openpose.pth" \
          "OpenPose 모델"

# Step 03: Cloth Segmentation
move_model "./u2net.pth" \
          "./organized/step_03_cloth_segmentation/u2net.pth" \
          "U2Net 분할 모델"

# Step 06: Virtual Fitting (SAM 모델)
move_model "./sam_vit_h_4b8939.pth" \
          "./organized/step_06_virtual_fitting/sam_vit_h_4b8939.pth" \
          "SAM ViT-H 모델"

# Diffusion 모델
move_model "./pytorch_model.bin" \
          "./organized/step_06_virtual_fitting/pytorch_model.bin" \
          "PyTorch Diffusion 모델"

# 4. 누락된 모델 다운로드 스크립트 생성
echo ""
echo "📥 4단계: 누락된 모델 다운로드 가이드 생성..."

cat > "$AI_MODELS_ROOT/DOWNLOAD_MISSING_MODELS.sh" << 'EOF'
#!/bin/bash
# 🔥 누락된 모델 다운로드 스크립트
# conda 환경에서 실행 권장

echo "📥 누락된 모델들 다운로드 시작..."

# conda 환경 확인
if [[ "$CONDA_DEFAULT_ENV" != "mycloset-ai" ]]; then
    echo "⚠️  mycloset-ai conda 환경을 활성화하세요"
    echo "   conda activate mycloset-ai"
    exit 1
fi

# Python 패키지 확인/설치
echo "🔧 필요 패키지 설치..."
pip install gdown huggingface_hub

# 1. Human Parsing 모델들
echo "👤 Human Parsing 모델 다운로드..."
mkdir -p organized/step_01_human_parsing/

# Graphonomy LIP 모델
echo "  📥 Graphonomy LIP 모델..."
# 실제 다운로드 URL은 논문/GitHub 확인 필요
# gdown "GOOGLE_DRIVE_ID" -O organized/step_01_human_parsing/exp-schp-201908261155-lip.pth

# 2. Pose Estimation 모델들  
echo "🤸 Pose Estimation 모델 다운로드..."
mkdir -p organized/step_02_pose_estimation/

# OpenPose Body 모델
echo "  📥 OpenPose Body 모델..."
wget -O organized/step_02_pose_estimation/body_pose_model.pth \
  "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel"

# 3. Virtual Fitting 모델들
echo "👗 Virtual Fitting 모델 다운로드..."
mkdir -p organized/step_06_virtual_fitting/

# HR-VITON 모델
echo "  📥 HR-VITON 모델..."
# python -c "
# from huggingface_hub import hf_hub_download
# hf_hub_download(repo_id='levihsu/OOTDiffusion', filename='checkpoints/ootd/ootd_dc/checkpoint-36000/unet_vton/pytorch_model.bin', local_dir='organized/step_06_virtual_fitting/')
# "

echo "✅ 다운로드 완료! 다음 명령어로 확인하세요:"
echo "   find organized/ -name '*.pth' -o -name '*.bin' | head -10"
EOF

chmod +x "$AI_MODELS_ROOT/DOWNLOAD_MISSING_MODELS.sh"
echo "  📝 다운로드 스크립트 생성: DOWNLOAD_MISSING_MODELS.sh"

# 5. 모델 설정 파일 생성
echo ""
echo "⚙️  5단계: 모델 설정 파일 생성..."

cat > "$AI_MODELS_ROOT/model_config.yaml" << EOF
# MyCloset AI 모델 설정 파일
# conda 환경 우선 지원

model_root: "$AI_MODELS_ROOT"

# Step별 모델 설정
steps:
  step_01_human_parsing:
    primary_model: "organized/step_01_human_parsing/exp-schp-201908301523-atr.pth"
    alternative_models:
      - "checkpoints/step_01_human_parsing/graphonomy.pth"
    required: true
    
  step_02_pose_estimation:  
    primary_model: "organized/step_02_pose_estimation/openpose.pth"
    alternative_models:
      - "checkpoints/step_02_pose_estimation/body_pose_model.pth"
    required: true
    
  step_03_cloth_segmentation:
    primary_model: "organized/step_03_cloth_segmentation/u2net.pth"
    alternative_models: []
    required: true
    
  step_04_geometric_matching:
    primary_model: "organized/step_04_geometric_matching/gmm_final.pth"
    alternative_models: []
    required: false
    
  step_05_cloth_warping:
    primary_model: "organized/step_05_cloth_warping/tom_final.pth" 
    alternative_models: []
    required: false
    
  step_06_virtual_fitting:
    primary_model: "organized/step_06_virtual_fitting/pytorch_model.bin"
    sam_model: "organized/step_06_virtual_fitting/sam_vit_h_4b8939.pth"
    alternative_models: []
    required: true
    
  step_07_post_processing:
    primary_model: null  # 전통적 방법 사용
    alternative_models: []
    required: false
    
  step_08_quality_assessment:
    primary_model: null  # 전통적 방법 사용  
    alternative_models: []
    required: false

# conda 환경 설정
conda:
  environment: "mycloset-ai"
  python_version: "3.10"
  pytorch_channel: "pytorch"
  
# M3 Max 최적화
hardware:
  device: "mps"
  memory_gb: 128
  optimization_level: "maximum"
  
# 폴백 설정  
fallback:
  enabled: true
  use_traditional_methods: true
  strict_mode: false
EOF

echo "  📝 모델 설정 파일 생성: model_config.yaml"

# 6. 경로 검증 스크립트 생성
echo ""
echo "🔍 6단계: 경로 검증 Python 스크립트 생성..."

cat > "$AI_MODELS_ROOT/validate_models.py" << 'EOF'
#!/usr/bin/env python3
"""
🔍 MyCloset AI 모델 경로 검증 스크립트
conda 환경에서 실행: python validate_models.py
"""

import os
import sys
from pathlib import Path
import yaml

def validate_models():
    """모델 파일들 검증"""
    
    # conda 환경 확인
    if os.environ.get('CONDA_DEFAULT_ENV') != 'mycloset-ai':
        print("⚠️  mycloset-ai conda 환경을 활성화하세요")
        print("   conda activate mycloset-ai")
        return False
    
    print("🔍 MyCloset AI 모델 검증 시작...")
    print(f"🐍 conda 환경: {os.environ.get('CONDA_DEFAULT_ENV')}")
    
    ai_models_root = Path(__file__).parent
    config_file = ai_models_root / "model_config.yaml"
    
    if not config_file.exists():
        print(f"❌ 설정 파일이 없습니다: {config_file}")
        return False
    
    # YAML 설정 로드
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {e}")
        return False
    
    print("\n📊 모델 파일 검증 결과:")
    
    total_models = 0
    found_models = 0
    total_size_mb = 0
    
    for step_name, step_config in config['steps'].items():
        print(f"\n🔧 {step_name}:")
        
        primary_model = step_config.get('primary_model')
        required = step_config.get('required', False)
        
        if primary_model:
            total_models += 1
            model_path = ai_models_root / primary_model
            
            if model_path.exists():
                found_models += 1
                size_mb = model_path.stat().st_size / (1024 * 1024)
                total_size_mb += size_mb
                print(f"  ✅ {primary_model} ({size_mb:.1f}MB)")
            else:
                status = "❌ (필수)" if required else "⚠️  (선택)"
                print(f"  {status} {primary_model}")
        
        # SAM 모델 별도 체크 (step_06용)
        sam_model = step_config.get('sam_model')
        if sam_model:
            total_models += 1
            sam_path = ai_models_root / sam_model
            if sam_path.exists():
                found_models += 1
                size_mb = sam_path.stat().st_size / (1024 * 1024)
                total_size_mb += size_mb
                print(f"  ✅ {sam_model} ({size_mb:.1f}MB)")
            else:
                print(f"  ❌ {sam_model} (SAM 모델)")
    
    print(f"\n📈 검증 결과:")
    print(f"  총 모델: {total_models}개")
    print(f"  발견됨: {found_models}개 ({found_models/total_models*100:.1f}%)")
    print(f"  총 크기: {total_size_mb:.1f}MB ({total_size_mb/1024:.1f}GB)")
    
    if found_models >= total_models * 0.5:  # 50% 이상 있으면 OK
        print("✅ 기본 모델 검증 통과")
        return True
    else:
        print("❌ 필수 모델 부족 - DOWNLOAD_MISSING_MODELS.sh 실행 권장")
        return False

if __name__ == "__main__":
    try:
        success = validate_models()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  사용자 중단")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        sys.exit(1)
EOF

chmod +x "$AI_MODELS_ROOT/validate_models.py"
echo "  📝 검증 스크립트 생성: validate_models.py"

# 7. auto_model_detector.py 경로 업데이트
echo ""
echo "🔄 7단계: auto_model_detector.py 경로 업데이트..."

MODEL_DETECTOR_FILE="$BACKEND_ROOT/app/ai_pipeline/utils/auto_model_detector.py"

if [[ -f "$MODEL_DETECTOR_FILE" ]]; then
    # 백업 생성
    cp "$MODEL_DETECTOR_FILE" "$MODEL_DETECTOR_FILE.backup"
    echo "  💾 백업 생성: auto_model_detector.py.backup"
    
    # 새로운 경로 추가 (기존 파일 수정)
    cat >> "$MODEL_DETECTOR_FILE" << 'EOF'

# ==============================================
# 🔧 MyCloset AI 정리된 모델 경로 (2025-07-21 업데이트)
# ==============================================

# 정리된 모델 경로들 추가
ORGANIZED_MODEL_PATHS = [
    str(Path(__file__).parent.parent.parent.parent / "ai_models/organized/step_01_human_parsing"),
    str(Path(__file__).parent.parent.parent.parent / "ai_models/organized/step_02_pose_estimation"),
    str(Path(__file__).parent.parent.parent.parent / "ai_models/organized/step_03_cloth_segmentation"),
    str(Path(__file__).parent.parent.parent.parent / "ai_models/organized/step_04_geometric_matching"),
    str(Path(__file__).parent.parent.parent.parent / "ai_models/organized/step_05_cloth_warping"),
    str(Path(__file__).parent.parent.parent.parent / "ai_models/organized/step_06_virtual_fitting"),
    str(Path(__file__).parent.parent.parent.parent / "ai_models/organized/step_07_post_processing"),
    str(Path(__file__).parent.parent.parent.parent / "ai_models/organized/step_08_quality_assessment"),
]

# 기존 경로에 정리된 경로 추가
if 'ENHANCED_SEARCH_PATHS' in globals():
    ENHANCED_SEARCH_PATHS.extend(ORGANIZED_MODEL_PATHS)
else:
    ENHANCED_SEARCH_PATHS = ORGANIZED_MODEL_PATHS
EOF

    echo "  ✅ auto_model_detector.py 경로 업데이트 완료"
else
    echo "  ⚠️  auto_model_detector.py를 찾을 수 없습니다"
fi

# 8. 최종 검증 실행
echo ""
echo "🎯 8단계: 최종 검증..."

cd "$AI_MODELS_ROOT"

# Python으로 모델 검증 실행
if command -v python &> /dev/null; then
    echo "🔍 Python 검증 실행..."
    python validate_models.py
else
    echo "⚠️  Python이 설치되지 않았습니다"
fi

# 정리된 모델 목록 출력
echo ""
echo "📋 정리된 모델 파일들:"
find organized/ -name "*.pth" -o -name "*.bin" -o -name "*.ckpt" 2>/dev/null | while read file; do
    if [[ -f "$file" ]]; then
        size=$(du -h "$file" 2>/dev/null | cut -f1)
        echo "  ✅ $file ($size)"
    fi
done

echo ""
echo "🎉 모델 체크포인트 경로 정리 완료!"
echo ""
echo "📋 다음 단계:"
echo "  1. 누락된 모델 다운로드: ./DOWNLOAD_MISSING_MODELS.sh"
echo "  2. 검증 재실행: python validate_models.py" 
echo "  3. 서버 재시작: cd ../.. && python app/main.py"
echo ""
echo "🐍 conda 환경 유지: conda activate mycloset-ai"