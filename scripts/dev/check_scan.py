#!/bin/bash
# 🔍 AI 모델 체크포인트 탐지 및 분석 명령어 모음

echo "🔍 MyCloset AI 체크포인트 탐지 시작..."

# 1. 기본 PyTorch 모델 파일 찾기
echo "📁 PyTorch 모델 파일 (.pth, .pt) 탐지:"
find . -type f \( -name "*.pth" -o -name "*.pt" \) -exec ls -lh {} \; | head -20

# 2. 기타 AI 모델 파일 찾기
echo "📁 기타 모델 파일 (.bin, .safetensors) 탐지:"
find . -type f \( -name "*.bin" -o -name "*.safetensors" \) -exec ls -lh {} \; | head -20

# 3. 크기별 정렬 (큰 파일들이 주로 모델)
echo "📊 크기별 모델 파일 정렬 (상위 15개):"
find . -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" \) -exec ls -lh {} \; | sort -k5 -hr | head -15

# 4. 디렉토리별 그룹핑
echo "📂 디렉토리별 모델 파일 분포:"
find . -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" \) | cut -d'/' -f1-3 | sort | uniq -c | sort -nr

# 5. 특정 패턴 모델 찾기 (Step별)
echo "🎯 Step별 모델 패턴 탐지:"

echo "   Step 01 - Human Parsing:"
find . -type f -name "*.pth" | grep -i -E "(human|parsing|schp|atr|graphonomy)" | head -5

echo "   Step 02 - Pose Estimation:"
find . -type f -name "*.pth" | grep -i -E "(pose|openpose|body|keypoint)" | head -5

echo "   Step 03 - Cloth Segmentation:"
find . -type f -name "*.pth" | grep -i -E "(u2net|segmentation|cloth|mask)" | head -5

echo "   Step 04 - Geometric Matching:"
find . -type f -name "*.pth" | grep -i -E "(geometric|matching|gmm|tps)" | head -5

echo "   Step 05 - Cloth Warping:"
find . -type f -name "*.pth" | grep -i -E "(warping|warp|cloth|viton)" | head -5

echo "   Step 06 - Virtual Fitting:"
find . -type f \( -name "*.pth" -o -name "*.bin" -o -name "*.safetensors" \) | grep -i -E "(diffusion|ootd|viton|stable|unet)" | head -5

echo "   Step 07 - Post Processing:"
find . -type f -name "*.pth" | grep -i -E "(super|resolution|esrgan|sr|denoise)" | head -5

echo "   Step 08 - Quality Assessment:"
find . -type f -name "*.pth" | grep -i -E "(quality|assessment|clip|similarity)" | head -5

# 6. 상세 분석을 위한 Python 스크립트 실행
echo "🔬 상세 체크포인트 분석 실행:"
if [ -f "backend/scripts/analyze_checkpoints.py" ]; then
    python backend/scripts/analyze_checkpoints.py
else
    echo "   ⚠️  analyze_checkpoints.py 파일이 없습니다."
fi

# 7. 모델 크기 통계
echo "📊 모델 파일 크기 통계:"
find . -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" \) -exec stat -f%z {} \; | awk '{
    total += $1; 
    count++; 
    if($1 > max) max = $1; 
    if(min == 0 || $1 < min) min = $1
} END {
    print "총 파일 수: " count
    print "총 크기: " total/1024/1024/1024 " GB"
    print "평균 크기: " total/count/1024/1024 " MB"
    print "최대 크기: " max/1024/1024 " MB"
    print "최소 크기: " min/1024/1024 " MB"
}'

# 8. 실행 권한 및 경로 정보
echo "🔐 실행 환경 정보:"
echo "   현재 디렉토리: $(pwd)"
echo "   Python 경로: $(which python)"
echo "   Conda 환경: $CONDA_DEFAULT_ENV"

echo "✅ 체크포인트 탐지 완료!"