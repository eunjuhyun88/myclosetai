#!/bin/bash

echo "🔧 MyCloset AI 모델 경로 정리 시작"
echo "=================================="

# 프로젝트 루트
PROJECT_ROOT="/Users/gimdudeul/MVP/mycloset-ai"
AI_MODELS_ROOT="$PROJECT_ROOT/backend/ai_models"

# conda 환경 확인
if [ "$CONDA_DEFAULT_ENV" != "mycloset-ai" ]; then
    echo "⚠️ conda 환경을 mycloset-ai로 변경하세요"
    echo "   conda activate mycloset-ai"
    exit 1
fi

echo "🐍 Conda 환경: $CONDA_DEFAULT_ENV ✅"
echo "📁 AI 모델 루트: $AI_MODELS_ROOT"
echo ""

# 1. 표준 Step 디렉토리 생성
echo "📁 1. 표준 Step 디렉토리 생성..."
mkdir -p "$AI_MODELS_ROOT/step_01_human_parsing"
mkdir -p "$AI_MODELS_ROOT/step_02_pose_estimation" 
mkdir -p "$AI_MODELS_ROOT/step_03_cloth_segmentation"
mkdir -p "$AI_MODELS_ROOT/step_04_geometric_matching"
mkdir -p "$AI_MODELS_ROOT/step_05_cloth_warping"
mkdir -p "$AI_MODELS_ROOT/step_06_virtual_fitting"
mkdir -p "$AI_MODELS_ROOT/step_07_post_processing"
mkdir -p "$AI_MODELS_ROOT/step_08_quality_assessment"
echo "   ✅ 8개 Step 디렉토리 생성 완료"

echo ""

# 2. 주요 모델 파일들 심볼릭 링크로 연결
echo "🔗 2. 주요 모델 파일 심볼릭 링크 생성..."

# Step 01: Human Parsing
echo "   🔧 Step 01: Human Parsing"
if [ -d "$AI_MODELS_ROOT/checkpoints/step_01_human_parsing" ]; then
    ln -sf "$AI_MODELS_ROOT/checkpoints/step_01_human_parsing"/* "$AI_MODELS_ROOT/step_01_human_parsing/" 2>/dev/null
    echo "      ✅ checkpoints/step_01_human_parsing 연결"
fi

if [ -d "$AI_MODELS_ROOT/organized/step_01_human_parsing" ]; then
    ln -sf "$AI_MODELS_ROOT/organized/step_01_human_parsing"/* "$AI_MODELS_ROOT/step_01_human_parsing/" 2>/dev/null
    echo "      ✅ organized/step_01_human_parsing 연결"
fi

if [ -d "$AI_MODELS_ROOT/ai_models2/step_01_human_parsing" ]; then
    ln -sf "$AI_MODELS_ROOT/ai_models2/step_01_human_parsing"/* "$AI_MODELS_ROOT/step_01_human_parsing/" 2>/dev/null
    echo "      ✅ ai_models2/step_01_human_parsing 연결"
fi

# Graphonomy 모델 특별 처리
if [ -d "$AI_MODELS_ROOT/Graphonomy" ]; then
    ln -sf "$AI_MODELS_ROOT/Graphonomy"/*.pth "$AI_MODELS_ROOT/step_01_human_parsing/" 2>/dev/null
    echo "      ✅ Graphonomy 모델 연결"
fi

# Step 02: Pose Estimation
echo "   🔧 Step 02: Pose Estimation"
if [ -d "$AI_MODELS_ROOT/checkpoints/step_02_pose_estimation" ]; then
    ln -sf "$AI_MODELS_ROOT/checkpoints/step_02_pose_estimation"/* "$AI_MODELS_ROOT/step_02_pose_estimation/" 2>/dev/null
    echo "      ✅ checkpoints/step_02_pose_estimation 연결"
fi

if [ -d "$AI_MODELS_ROOT/openpose" ]; then
    ln -sf "$AI_MODELS_ROOT/openpose/models"/* "$AI_MODELS_ROOT/step_02_pose_estimation/" 2>/dev/null
    echo "      ✅ OpenPose 모델 연결"
fi

# Step 03: Cloth Segmentation  
echo "   🔧 Step 03: Cloth Segmentation"
if [ -d "$AI_MODELS_ROOT/checkpoints/step_03_cloth_segmentation" ]; then
    ln -sf "$AI_MODELS_ROOT/checkpoints/step_03_cloth_segmentation"/* "$AI_MODELS_ROOT/step_03_cloth_segmentation/" 2>/dev/null
    echo "      ✅ checkpoints/step_03_cloth_segmentation 연결"
fi

if [ -d "$AI_MODELS_ROOT/u2net" ]; then
    ln -sf "$AI_MODELS_ROOT/u2net"/* "$AI_MODELS_ROOT/step_03_cloth_segmentation/" 2>/dev/null
    echo "      ✅ U2Net 모델 연결"
fi

# Step 04: Geometric Matching
echo "   🔧 Step 04: Geometric Matching"
if [ -d "$AI_MODELS_ROOT/checkpoints/step_04_geometric_matching" ]; then
    ln -sf "$AI_MODELS_ROOT/checkpoints/step_04_geometric_matching"/* "$AI_MODELS_ROOT/step_04_geometric_matching/" 2>/dev/null
    echo "      ✅ checkpoints/step_04_geometric_matching 연결"
fi

# Step 05: Cloth Warping
echo "   🔧 Step 05: Cloth Warping"
if [ -d "$AI_MODELS_ROOT/checkpoints/step_05_cloth_warping" ]; then
    ln -sf "$AI_MODELS_ROOT/checkpoints/step_05_cloth_warping"/* "$AI_MODELS_ROOT/step_05_cloth_warping/" 2>/dev/null
    echo "      ✅ checkpoints/step_05_cloth_warping 연결"
fi

# Step 06: Virtual Fitting (가장 중요)
echo "   🔧 Step 06: Virtual Fitting"
if [ -d "$AI_MODELS_ROOT/checkpoints/step_06_virtual_fitting" ]; then
    ln -sf "$AI_MODELS_ROOT/checkpoints/step_06_virtual_fitting"/* "$AI_MODELS_ROOT/step_06_virtual_fitting/" 2>/dev/null
    echo "      ✅ checkpoints/step_06_virtual_fitting 연결"
fi

if [ -d "$AI_MODELS_ROOT/OOTDiffusion" ]; then
    ln -sf "$AI_MODELS_ROOT/OOTDiffusion/checkpoints"/* "$AI_MODELS_ROOT/step_06_virtual_fitting/" 2>/dev/null
    echo "      ✅ OOTDiffusion 모델 연결"
fi

if [ -d "$AI_MODELS_ROOT/HR-VITON" ]; then
    ln -sf "$AI_MODELS_ROOT/HR-VITON"/*.pth "$AI_MODELS_ROOT/step_06_virtual_fitting/" 2>/dev/null
    echo "      ✅ HR-VITON 모델 연결"
fi

# Step 07: Post Processing
echo "   🔧 Step 07: Post Processing"
if [ -d "$AI_MODELS_ROOT/checkpoints/step_07_post_processing" ]; then
    ln -sf "$AI_MODELS_ROOT/checkpoints/step_07_post_processing"/* "$AI_MODELS_ROOT/step_07_post_processing/" 2>/dev/null
    echo "      ✅ checkpoints/step_07_post_processing 연결"
fi

# Step 08: Quality Assessment
echo "   🔧 Step 08: Quality Assessment"
if [ -d "$AI_MODELS_ROOT/checkpoints/step_08_quality_assessment" ]; then
    ln -sf "$AI_MODELS_ROOT/checkpoints/step_08_quality_assessment"/* "$AI_MODELS_ROOT/step_08_quality_assessment/" 2>/dev/null
    echo "      ✅ checkpoints/step_08_quality_assessment 연결"
fi

if [ -d "$AI_MODELS_ROOT/clip_vit_large" ]; then
    ln -sf "$AI_MODELS_ROOT/clip_vit_large"/* "$AI_MODELS_ROOT/step_08_quality_assessment/" 2>/dev/null
    echo "      ✅ CLIP 모델 연결"
fi

echo ""

# 3. 권한 설정
echo "🔐 3. 권한 설정..."
chmod -R 755 "$AI_MODELS_ROOT/step_"*
echo "   ✅ 모든 Step 디렉토리 권한 설정 완료"

echo ""

# 4. 결과 확인
echo "📊 4. 결과 확인..."
for step in {01..08}; do
    step_dir="$AI_MODELS_ROOT/step_${step}_*"
    file_count=$(find $step_dir -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" 2>/dev/null | wc -l)
    echo "   Step ${step}: ${file_count}개 모델 파일"
done

echo ""

# 5. 탐지기 실행을 위한 Python 스크립트 생성
echo "🐍 5. 모델 탐지 확인 스크립트 생성..."

cat > "$PROJECT_ROOT/test_model_detection.py" << 'EOF'
#!/usr/bin/env python3
"""
🔍 모델 탐지 테스트 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent / "backend"
sys.path.insert(0, str(project_root))

def test_model_detection():
    """모델 탐지 테스트"""
    print("🔍 모델 탐지 테스트 시작...")
    
    ai_models_root = Path(__file__).parent / "backend" / "ai_models"
    
    for step in range(1, 9):
        step_dir = ai_models_root / f"step_{step:02d}_*"
        
        # 실제 디렉토리 찾기
        import glob
        step_dirs = glob.glob(str(step_dir))
        
        if step_dirs:
            step_path = Path(step_dirs[0])
            model_files = list(step_path.glob("*.pth")) + list(step_path.glob("*.pt")) + list(step_path.glob("*.bin"))
            print(f"   Step {step:02d}: {len(model_files)}개 모델 파일 발견")
            for f in model_files[:3]:  # 상위 3개만 표시
                print(f"      - {f.name}")
        else:
            print(f"   Step {step:02d}: 디렉토리 없음")

if __name__ == "__main__":
    test_model_detection()
EOF

chmod +x "$PROJECT_ROOT/test_model_detection.py"
echo "   ✅ test_model_detection.py 생성 완료"

echo ""
echo "🎉 모델 경로 정리 완료!"
echo "=================================="
echo ""
echo "📋 다음 단계:"
echo "   1. python3 test_model_detection.py (모델 탐지 확인)"
echo "   2. python3 backend/app/main.py (서버 재시작)"
echo ""
echo "✅ 정리 완료 - 모델 탐지가 이제 작동할 것입니다"