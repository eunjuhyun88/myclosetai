#!/bin/bash
# MyCloset AI - Conda 환경 수정 스크립트 (Python 3.10.11)

echo "🔧 MyCloset AI Conda 환경 수정"
echo "현재 환경: base"
echo "Python: 3.10.11"
echo ""

# 현재 환경 활성화
conda activate base

# NumPy 호환성 해결 (Python 3.12 버전)
echo "🔢 NumPy 호환성 수정 중..."
pip install numpy==1.24.4

# PyTorch M3 Max 최적화 버전 설치
echo "🔥 PyTorch M3 Max 최적화 설치 중..."
pip install torch torchvision torchaudio

# 기타 필수 패키지 업데이트
echo "📚 필수 패키지 업데이트 중..."
pip install --upgrade fastapi uvicorn pydantic

echo "✅ Conda 환경 수정 완료"
echo "🚀 서버 실행: cd backend && python app/main.py"
