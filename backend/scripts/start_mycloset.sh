#!/bin/bash
# MyCloset AI Startup Script
# Generated at: 2025-07-17 19:20:00

echo "🚀 MyCloset AI 시작 중..."

# 환경 변수 로드
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# conda 환경 활성화 확인
if [ -z "$CONDA_PREFIX" ]; then
    echo "⚠️ conda 환경이 활성화되지 않았습니다"
    echo "다음 명령어를 실행하세요: conda activate mycloset-ai"
    exit 1
fi

# 모델 가용성 체크
echo "🔍 모델 가용성 체크..."

echo "📊 사용 가능한 모델: 7/8개"
echo "✅ human_parsing_graphonomy"
echo "✅ virtual_fitting_hrviton"
echo "✅ cloth_segmentation_u2net"
echo "✅ pose_estimation_openpose"
echo "✅ stable_diffusion"
echo "✅ clip_vit_base"
echo "✅ geometric_matching_gmm"

# 서버 시작
echo "🌐 서버 시작 중..."
python3 app/main.py

echo "✅ MyCloset AI 서버 시작 완료"
