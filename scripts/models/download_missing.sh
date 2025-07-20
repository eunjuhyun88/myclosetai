#!/bin/bash
# 🔽 누락된 모델 자동 다운로드

echo "🔽 MyCloset AI 누락 모델 다운로드 시작..."

# 모델 저장 디렉토리 생성
MODELS_DIR="/Users/gimdudeul/MVP/mycloset-ai/backend/app/ai_pipeline/models"
mkdir -p "$MODELS_DIR/checkpoints/step_04_geometric_matching"
mkdir -p "$MODELS_DIR/checkpoints/step_07_post_processing" 
mkdir -p "$MODELS_DIR/checkpoints/step_08_quality_assessment"

cd "$MODELS_DIR"

echo "📂 현재 디렉토리: $(pwd)"

# 1. Geometric Matching Model (GMM)
echo "🔽 1. Geometric Matching Model 다운로드..."
if [ ! -f "checkpoints/step_04_geometric_matching/gmm_final.pth" ]; then
    echo "   HR-VITON GMM 모델 다운로드 중..."
    # GitHub 릴리즈에서 다운로드 시도
    wget -O "checkpoints/step_04_geometric_matching/gmm_final.pth" \
         "https://github.com/shadow2496/HR-VITON/releases/download/3.0.0/gmm_final.pth" \
         2>/dev/null || echo "   ⚠️ 직접 다운로드 실패 - 수동 다운로드 필요"
    
    # 대안: Google Drive 링크 (수동)
    if [ ! -f "checkpoints/step_04_geometric_matching/gmm_final.pth" ]; then
        echo "   💡 수동 다운로드 링크:"
        echo "      https://drive.google.com/file/d/1M9BVlLR3Pb3NdLfWN6L3Ql8OIJDOqK4L/view"
        echo "      다운로드 후 -> checkpoints/step_04_geometric_matching/gmm_final.pth"
    fi
else
    echo "   ✅ GMM 모델 이미 존재"
fi

# 2. Post Processing Model (Real-ESRGAN)
echo "🔽 2. Post Processing Model 다운로드..."
if [ ! -f "checkpoints/step_07_post_processing/RealESRGAN_x4plus.pth" ]; then
    echo "   Real-ESRGAN 모델 다운로드 중..."
    wget -O "checkpoints/step_07_post_processing/RealESRGAN_x4plus.pth" \
         "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
         2>/dev/null || echo "   ⚠️ 직접 다운로드 실패"
    
    # 대안 링크
    if [ ! -f "checkpoints/step_07_post_processing/RealESRGAN_x4plus.pth" ]; then
        echo "   💡 대안 다운로드:"
        echo "      curl -L -o checkpoints/step_07_post_processing/RealESRGAN_x4plus.pth \\"
        echo "           https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    fi
else
    echo "   ✅ Real-ESRGAN 모델 이미 존재"
fi

# 3. Quality Assessment Model (CLIP)
echo "🔽 3. Quality Assessment Model 다운로드..."
if [ ! -f "checkpoints/step_08_quality_assessment/clip_vit_base_patch32.pt" ]; then
    echo "   CLIP 모델 다운로드 중..."
    
    # Hugging Face에서 다운로드 시도
    wget -O "checkpoints/step_08_quality_assessment/clip_vit_base_patch32.pt" \
         "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin" \
         2>/dev/null || echo "   ⚠️ CLIP 다운로드 실패"
    
    # 대안: 직접 PyTorch Hub 사용 안내
    if [ ! -f "checkpoints/step_08_quality_assessment/clip_vit_base_patch32.pt" ]; then
        echo "   💡 Python으로 CLIP 모델 다운로드:"
        echo "      python -c \"import torch; torch.hub.download_url_to_file('https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-32.pt', 'checkpoints/step_08_quality_assessment/clip_vit_base_patch32.pt')\""
    fi
else
    echo "   ✅ CLIP 모델 이미 존재"
fi

# 4. 다운로드 결과 확인
echo ""
echo "📊 다운로드 결과 확인..."

check_file() {
    if [ -f "$1" ]; then
        size=$(du -h "$1" | cut -f1)
        echo "   ✅ $(basename "$1") - $size"
        return 0
    else
        echo "   ❌ $(basename "$1") - 없음"
        return 1
    fi
}

success_count=0

echo "🔍 모델 파일 검증:"
check_file "checkpoints/step_04_geometric_matching/gmm_final.pth" && ((success_count++))
check_file "checkpoints/step_07_post_processing/RealESRGAN_x4plus.pth" && ((success_count++))
check_file "checkpoints/step_08_quality_assessment/clip_vit_base_patch32.pt" && ((success_count++))

echo ""
echo "📈 다운로드 결과: $success_count/3개 성공"

if [ $success_count -eq 3 ]; then
    echo "🎉 모든 모델 다운로드 완료!"
    echo ""
    echo "🚀 다음 단계:"
    echo "   1. python quick_fix_setup.py"
    echo "   2. python test_immediate.py"
elif [ $success_count -gt 0 ]; then
    echo "⚠️ 일부 모델 다운로드 완료"
    echo ""
    echo "💡 수동 다운로드가 필요한 모델들:"
    [ ! -f "checkpoints/step_04_geometric_matching/gmm_final.pth" ] && echo "   - GMM Final"
    [ ! -f "checkpoints/step_07_post_processing/RealESRGAN_x4plus.pth" ] && echo "   - Real-ESRGAN"
    [ ! -f "checkpoints/step_08_quality_assessment/clip_vit_base_patch32.pt" ] && echo "   - CLIP"
else
    echo "❌ 자동 다운로드 실패"
    echo ""
    echo "💡 수동 다운로드 방법:"
    echo "   1. 브라우저에서 각 모델 링크 접속"
    echo "   2. 해당 디렉토리에 파일 저장"
    echo "   3. 파일명 정확히 맞추기"
fi

echo ""
echo "✅ 누락 모델 다운로드 스크립트 완료!"