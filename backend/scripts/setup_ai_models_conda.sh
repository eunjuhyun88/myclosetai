#!/bin/bash
# scripts/setup_ai_models_conda.sh
# Conda 환경용 MyCloset AI 모델 설정 스크립트

echo "🐍 Conda 환경에서 MyCloset AI 설정 시작..."

# 현재 conda 환경 확인
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "❌ Conda 환경이 활성화되지 않았습니다."
    echo "📝 먼저 conda 환경을 활성화하세요: conda activate mycloset"
    exit 1
fi

echo "✅ 현재 Conda 환경: $CONDA_DEFAULT_ENV"

# 1. Conda로 기본 패키지 설치
echo "📦 Conda로 기본 패키지 설치 중..."

# PyTorch 설치 (Mac의 경우 MPS 지원)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 macOS 감지됨 - PyTorch with MPS 설치..."
    conda install pytorch torchvision torchaudio -c pytorch -y
else
    echo "🐧 Linux 감지됨 - PyTorch with CUDA 설치..."
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
fi

# 2. 추가 AI 라이브러리 설치
echo "🤖 AI 라이브러리 설치 중..."

# Conda로 설치 가능한 패키지들
conda install -c conda-forge opencv pillow numpy scipy pyyaml tqdm -y

# pip로 설치해야 하는 패키지들
pip install transformers diffusers huggingface-hub
pip install mediapipe onnxruntime
pip install rembg gdown
pip install fastapi uvicorn python-multipart python-dotenv

# 3. 기존 ai_models 폴더 활용하여 구조 정리
echo "📁 AI 모델 폴더 구조 정리 중..."

# 필요한 하위 디렉토리 생성
mkdir -p ai_models/{checkpoints,configs,temp}
mkdir -p ai_models/checkpoints/{ootdiffusion,viton_hd,human_parsing,background_removal}

# .gitkeep 파일 생성 (빈 폴더 유지용)
find ai_models -type d -exec touch {}/.gitkeep \;

echo "✅ 폴더 구조 정리 완료"

# 4. 간단한 AI 모델 다운로드 스크립트 생성
cat > scripts/download_models_conda.py << 'EOF'
#!/usr/bin/env python3
"""
Conda 환경용 AI 모델 다운로드 스크립트
"""
import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_model_directories():
    """모델 디렉토리 설정"""
    base_dir = Path("ai_models")
    
    # 체크포인트 디렉토리들
    checkpoint_dirs = [
        "checkpoints/ootdiffusion",
        "checkpoints/viton_hd", 
        "checkpoints/human_parsing",
        "checkpoints/background_removal"
    ]
    
    for dir_name in checkpoint_dirs:
        dir_path = base_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ 디렉토리 생성: {dir_path}")
    
    # 설정 디렉토리
    config_dir = base_dir / "configs"
    config_dir.mkdir(exist_ok=True)
    
    return base_dir

def create_basic_config():
    """기본 설정 파일 생성"""
    config_dir = Path("ai_models/configs")
    
    # 마스터 설정 파일
    master_config = """
# MyCloset AI 모델 설정
models:
  ootdiffusion:
    enabled: true
    path: "ai_models/checkpoints/ootdiffusion"
    device: "auto"  # auto, cuda, mps, cpu
  
  viton_hd:
    enabled: false  # 나중에 활성화
    path: "ai_models/checkpoints/viton_hd"
    device: "auto"

processing:
  default_model: "ootdiffusion"
  image_size: [512, 512]
  batch_size: 1
"""
    
    config_path = config_dir / "models_config.yaml"
    with open(config_path, 'w') as f:
        f.write(master_config)
    
    logger.info(f"✅ 설정 파일 생성: {config_path}")

def download_small_test_model():
    """작은 테스트 모델 다운로드"""
    try:
        from huggingface_hub import snapshot_download
        
        # 작은 테스트용 모델 다운로드
        model_path = "ai_models/checkpoints/ootdiffusion"
        
        logger.info("📥 테스트용 모델 다운로드 중...")
        logger.info("(실제로는 여기서 OOTDiffusion 등의 모델을 다운로드)")
        
        # 실제 구현에서는 실제 모델 다운로드
        # snapshot_download(repo_id="levihsu/OOTDiffusion", local_dir=model_path)
        
        # 지금은 더미 파일 생성
        dummy_file = Path(model_path) / "model_info.txt"
        dummy_file.parent.mkdir(parents=True, exist_ok=True)
        dummy_file.write_text("OOTDiffusion 모델 자리 (실제 다운로드 필요)")
        
        logger.info("✅ 테스트 설정 완료")
        
    except ImportError as e:
        logger.warning(f"⚠️ huggingface_hub 없음: {e}")
        logger.info("💡 설치: pip install huggingface_hub")

def main():
    """메인 함수"""
    print("🤖 Conda 환경용 AI 모델 설정")
    print("=" * 40)
    
    # 1. 디렉토리 설정
    base_dir = setup_model_directories()
    
    # 2. 설정 파일 생성
    create_basic_config()
    
    # 3. 테스트 모델 설정
    download_small_test_model()
    
    print("\n🎉 설정 완료!")
    print(f"📁 모델 저장 위치: {base_dir.absolute()}")
    print("\n📋 다음 단계:")
    print("1. 서버 테스트: python app/main.py")
    print("2. API 확인: http://localhost:8000/docs")

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/download_models_conda.py

# 5. 간단한 서버 실행 스크립트
cat > run_conda_server.sh << 'EOF'
#!/bin/bash
echo "🚀 Conda 환경에서 MyCloset AI 서버 시작..."

# Conda 환경 확인
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "❌ Conda 환경이 활성화되지 않았습니다."
    echo "📝 실행: conda activate mycloset"
    exit 1
fi

echo "✅ Conda 환경: $CONDA_DEFAULT_ENV"

# 모델 설정 실행
echo "🔧 모델 설정 확인 중..."
python scripts/download_models_conda.py

# 서버 시작
echo "🌐 서버 시작 중..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
EOF

chmod +x run_conda_server.sh

echo ""
echo "🎉 Conda 환경 설정 완료!"
echo ""
echo "📋 설치된 구성요소:"
echo "   ✅ PyTorch (Conda)"
echo "   ✅ Computer Vision 라이브러리들"  
echo "   ✅ FastAPI 및 웹 프레임워크"
echo "   ✅ AI 모델 폴더 구조"
echo ""
echo "🚀 다음 단계:"
echo "   1. 모델 설정: python scripts/download_models_conda.py"
echo "   2. 서버 시작: ./run_conda_server.sh"
echo "   3. 웹 확인: http://localhost:8000"
echo ""
echo "💡 팁:"
echo "   - 환경 확인: conda list pytorch"
echo "   - GPU 확인: python -c \"import torch; print(torch.cuda.is_available())\""