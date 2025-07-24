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
