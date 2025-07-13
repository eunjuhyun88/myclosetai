#!/usr/bin/env python3
"""
실제 AI 모델 체크포인트 다운로드 스크립트
"""
import os
import sys
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """필요한 라이브러리 확인"""
    missing = []
    
    try:
        import torch
        logger.info(f"✅ PyTorch {torch.__version__} 설치됨")
    except ImportError:
        missing.append("torch")
    
    try:
        from huggingface_hub import snapshot_download
        logger.info("✅ Hugging Face Hub 사용 가능")
    except ImportError:
        missing.append("huggingface_hub")
    
    try:
        import gdown
        logger.info("✅ gdown 사용 가능")
    except ImportError:
        missing.append("gdown")
    
    if missing:
        logger.error(f"❌ 누락된 패키지: {missing}")
        logger.info("💡 설치 명령어:")
        for pkg in missing:
            logger.info(f"   pip install {pkg}")
        return False
    
    return True

def download_ootdiffusion():
    """OOTDiffusion 모델 다운로드"""
    try:
        from huggingface_hub import snapshot_download
        
        model_dir = Path("ai_models/checkpoints/ootdiffusion")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("📥 OOTDiffusion 다운로드 시작... (약 5-10GB, 시간이 걸립니다)")
        
        # 실제 OOTDiffusion 모델 다운로드
        snapshot_download(
            repo_id="levihsu/OOTDiffusion",
            local_dir=str(model_dir),
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.py"],
            ignore_patterns=["*.git*", "*.md", "*.png", "*.jpg"]
        )
        
        logger.info("✅ OOTDiffusion 다운로드 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ OOTDiffusion 다운로드 실패: {e}")
        logger.info("💡 대안: 수동으로 https://huggingface.co/levihsu/OOTDiffusion 에서 다운로드")
        return False

def download_stable_diffusion_base():
    """Stable Diffusion 기본 모델 다운로드"""
    try:
        from huggingface_hub import snapshot_download
        
        model_dir = Path("ai_models/checkpoints/stable-diffusion-v1-5")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("📥 Stable Diffusion v1.5 다운로드 시작... (약 4GB)")
        
        snapshot_download(
            repo_id="runwayml/stable-diffusion-v1-5",
            local_dir=str(model_dir),
            allow_patterns=["*.safetensors", "*.json", "*.txt"],
            ignore_patterns=["*.git*", "*.ckpt", "*.png"]
        )
        
        logger.info("✅ Stable Diffusion 다운로드 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ Stable Diffusion 다운로드 실패: {e}")
        return False

def download_human_parsing():
    """인체 파싱 모델 다운로드"""
    try:
        import gdown
        
        model_dir = Path("ai_models/checkpoints/human_parsing")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("📥 Human Parsing 모델 다운로드 시작...")
        
        # Self-Correction Human Parsing 모델들
        models = {
            "atr_model.pth": "1ruJg4lqR_jgQPj-9K0PP-L2vJERYOxLP",
            "lip_model.pth": "1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH"
        }
        
        for filename, file_id in models.items():
            output_path = model_dir / filename
            if not output_path.exists():
                url = f"https://drive.google.com/uc?id={file_id}"
                logger.info(f"📥 다운로드 중: {filename}")
                gdown.download(url, str(output_path), quiet=False)
            else:
                logger.info(f"⏭️ 이미 존재: {filename}")
        
        logger.info("✅ Human Parsing 모델 다운로드 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ Human Parsing 다운로드 실패: {e}")
        return False

def download_u2net():
    """U²-Net 배경 제거 모델 다운로드"""
    try:
        import gdown
        
        model_dir = Path("ai_models/checkpoints/background_removal")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("📥 U²-Net 배경 제거 모델 다운로드 시작...")
        
        # U²-Net 모델 다운로드
        url = "https://github.com/xuebinqin/U-2-Net/releases/download/u2net/u2net.pth"
        output_path = model_dir / "u2net.pth"
        
        if not output_path.exists():
            gdown.download(url, str(output_path), quiet=False)
        else:
            logger.info("⏭️ U²-Net 모델이 이미 존재함")
        
        logger.info("✅ U²-Net 모델 다운로드 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ U²-Net 다운로드 실패: {e}")
        return False

def check_disk_space():
    """디스크 공간 확인"""
    try:
        import shutil
        
        total, used, free = shutil.disk_usage(".")
        free_gb = free // (1024**3)
        
        logger.info(f"💾 사용 가능한 디스크 공간: {free_gb} GB")
        
        if free_gb < 20:
            logger.warning("⚠️ 디스크 공간이 부족할 수 있습니다. (권장: 20GB 이상)")
            return False
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ 디스크 공간 확인 실패: {e}")
        return True

def show_download_status():
    """다운로드 상태 확인"""
    checkpoints_dir = Path("ai_models/checkpoints")
    
    logger.info("📊 현재 다운로드 상태:")
    logger.info("=" * 50)
    
    models = {
        "OOTDiffusion": "ootdiffusion",
        "Stable Diffusion": "stable-diffusion-v1-5", 
        "Human Parsing": "human_parsing",
        "Background Removal": "background_removal"
    }
    
    total_size = 0
    
    for name, folder in models.items():
        model_path = checkpoints_dir / folder
        if model_path.exists():
            # 폴더 크기 계산
            size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            size_mb = size / (1024**2)
            total_size += size_mb
            
            if size_mb > 100:  # 100MB 이상이면 다운로드된 것으로 간주
                logger.info(f"✅ {name}: {size_mb:.1f} MB")
            else:
                logger.info(f"❌ {name}: {size_mb:.1f} MB (미완료)")
        else:
            logger.info(f"❌ {name}: 폴더 없음")
    
    logger.info(f"📦 총 다운로드 크기: {total_size:.1f} MB ({total_size/1024:.1f} GB)")

def main():
    """메인 다운로드 함수"""
    print("🤖 실제 AI 모델 체크포인트 다운로드")
    print("=" * 50)
    
    # 1. 의존성 확인
    if not check_dependencies():
        sys.exit(1)
    
    # 2. 디스크 공간 확인
    if not check_disk_space():
        response = input("⚠️ 디스크 공간이 부족할 수 있습니다. 계속하시겠습니까? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # 3. 현재 상태 확인
    show_download_status()
    
    # 4. 다운로드 시작 여부 확인
    print("\n📥 다음 모델들을 다운로드합니다:")
    print("   - OOTDiffusion (~5-10GB)")
    print("   - Stable Diffusion v1.5 (~4GB)")  
    print("   - Human Parsing (~500MB)")
    print("   - U²-Net Background Removal (~200MB)")
    print("\n⏰ 예상 소요 시간: 30분 ~ 2시간 (인터넷 속도에 따라)")
    
    response = input("\n📝 다운로드를 시작하시겠습니까? (y/N): ")
    if response.lower() != 'y':
        print("❌ 다운로드가 취소되었습니다.")
        sys.exit(0)
    
    # 5. 모델 다운로드 실행
    start_time = time.time()
    success_count = 0
    total_models = 4
    
    logger.info("🚀 AI 모델 다운로드 시작...")
    
    # 작은 모델부터 다운로드
    if download_u2net():
        success_count += 1
    
    if download_human_parsing():
        success_count += 1
    
    # 큰 모델들
    if download_stable_diffusion_base():
        success_count += 1
    
    if download_ootdiffusion():
        success_count += 1
    
    # 6. 결과 출력
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "=" * 50)
    print(f"🎉 다운로드 완료! ({success_count}/{total_models} 성공)")
    print(f"⏰ 소요 시간: {elapsed_time/60:.1f}분")
    
    # 최종 상태 확인
    show_download_status()
    
    if success_count == total_models:
        print("\n✅ 모든 AI 모델이 성공적으로 다운로드되었습니다!")
        print("🚀 다음 단계: uvicorn app.main:app --reload")
    else:
        print(f"\n⚠️ 일부 모델 다운로드에 실패했습니다. ({success_count}/{total_models})")
        print("💡 인터넷 연결을 확인하고 다시 실행해보세요.")

if __name__ == "__main__":
    main()