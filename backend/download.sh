#!/usr/bin/env python3
"""
MyCloset AI - 수정된 모델 다운로더 (실제 경로 반영)
올바른 Hugging Face 경로와 대체 다운로드 방법 제공
"""

import os
import sys
import requests
import subprocess
import logging
from pathlib import Path
import zipfile
from tqdm import tqdm
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedModelDownloader:
    def __init__(self, base_path="/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models"):
        self.base_path = Path(base_path)
        self.checkpoints_dir = self.base_path / "checkpoints"
        
        # 디렉토리 생성
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        logger.info(f"🎯 모델 저장 경로: {self.checkpoints_dir}")
    
    def install_requirements(self):
        """필수 패키지 설치"""
        logger.info("📦 필수 패키지 설치 중...")
        
        packages = [
            "huggingface_hub>=0.19.0",
            "git-lfs",
            "transformers>=4.35.0",
            "diffusers>=0.21.0",
            "safetensors",
            "onnxruntime"
        ]
        
        for package in packages:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], check=True, capture_output=True)
                logger.info(f"✅ {package} 설치 완료")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️ {package} 설치 실패: {e}")
    
    def download_ootdiffusion_git_clone(self):
        """Git을 통한 OOTDiffusion 전체 저장소 클론"""
        logger.info("📥 Git을 통한 OOTDiffusion 다운로드...")
        
        ootd_repo_dir = self.base_path / "OOTDiffusion"
        
        if ootd_repo_dir.exists():
            logger.info("⏭️ OOTDiffusion 저장소 이미 존재")
            return True
        
        try:
            # GitHub 저장소 클론
            subprocess.run([
                "git", "clone", "https://github.com/levihsu/OOTDiffusion.git",
                str(ootd_repo_dir)
            ], check=True)
            
            logger.info("✅ OOTDiffusion GitHub 저장소 클론 완료")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Git 클론 실패: {e}")
            return False
    
    def download_ootdiffusion_huggingface(self):
        """Hugging Face를 통한 OOTDiffusion 다운로드"""
        logger.info("🤗 Hugging Face를 통한 OOTDiffusion 다운로드...")
        
        try:
            from huggingface_hub import snapshot_download
            
            ootd_hf_dir = self.checkpoints_dir / "ootdiffusion_hf"
            
            if ootd_hf_dir.exists():
                logger.info("⏭️ OOTDiffusion HF 모델 이미 존재")
                return True
            
            # 전체 저장소 다운로드
            logger.info("📥 OOTDiffusion 전체 저장소 다운로드 중...")
            snapshot_download(
                repo_id="levihsu/OOTDiffusion",
                local_dir=str(ootd_hf_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            logger.info("✅ OOTDiffusion HuggingFace 다운로드 완료")
            return True
            
        except ImportError:
            logger.error("❌ huggingface_hub이 설치되지 않음")
            return False
        except Exception as e:
            logger.error(f"❌ HuggingFace 다운로드 실패: {e}")
            return False
    
    def download_alternative_models(self):
        """대체 모델들 다운로드"""
        logger.info("🔄 대체 모델 다운로드...")
        
        # Stable Diffusion 1.5 (기본 디퓨전 모델)
        try:
            from huggingface_hub import snapshot_download
            
            sd_dir = self.checkpoints_dir / "stable-diffusion-v1-5"
            
            if not sd_dir.exists():
                logger.info("📥 Stable Diffusion v1.5 다운로드 중...")
                snapshot_download(
                    repo_id="runwayml/stable-diffusion-v1-5",
                    local_dir=str(sd_dir),
                    local_dir_use_symlinks=False
                )
                logger.info("✅ Stable Diffusion v1.5 다운로드 완료")
            else:
                logger.info("⏭️ Stable Diffusion v1.5 이미 존재")
                
        except Exception as e:
            logger.error(f"❌ Stable Diffusion 다운로드 실패: {e}")
    
    def download_clip_model(self):
        """CLIP 모델 다운로드 (OOTDiffusion 필수)"""
        logger.info("🎨 CLIP 모델 다운로드...")
        
        try:
            from huggingface_hub import snapshot_download
            
            clip_dir = self.checkpoints_dir / "clip-vit-large-patch14"
            
            if not clip_dir.exists():
                logger.info("📥 CLIP ViT Large 다운로드 중...")
                snapshot_download(
                    repo_id="openai/clip-vit-large-patch14",
                    local_dir=str(clip_dir),
                    local_dir_use_symlinks=False
                )
                logger.info("✅ CLIP 모델 다운로드 완료")
            else:
                logger.info("⏭️ CLIP 모델 이미 존재")
                return True
                
        except Exception as e:
            logger.error(f"❌ CLIP 다운로드 실패: {e}")
            return False
    
    def setup_ootdiffusion_structure(self):
        """OOTDiffusion 디렉토리 구조 설정"""
        logger.info("📁 OOTDiffusion 디렉토리 구조 설정...")
        
        ootd_base = self.checkpoints_dir / "ootdiffusion"
        ootd_base.mkdir(exist_ok=True)
        
        # 필요한 하위 디렉토리들
        subdirs = [
            "checkpoints/ootd",
            "checkpoints/humanparsing", 
            "checkpoints/openpose",
            "checkpoints/clip"
        ]
        
        for subdir in subdirs:
            dir_path = ootd_base / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ 디렉토리 구조 설정 완료")
    
    def create_model_download_guide(self):
        """수동 다운로드 가이드 생성"""
        guide_content = """
# MyCloset AI 모델 수동 다운로드 가이드

## 🎯 OOTDiffusion 모델 다운로드

### 방법 1: 전체 저장소 클론 (권장)
```bash
# 1. Git LFS 설치
brew install git-lfs

# 2. 전체 저장소 클론
cd /Users/gimdudeul/MVP/mycloset-ai/backend/ai_models
git clone https://github.com/levihsu/OOTDiffusion.git
cd OOTDiffusion
git lfs pull
```

### 방법 2: Hugging Face CLI 사용
```bash
# 1. Hugging Face CLI 설치
pip install huggingface_hub[cli]

# 2. 전체 저장소 다운로드
huggingface-cli download levihsu/OOTDiffusion --local-dir ./ootdiffusion_hf
```

### 방법 3: Python 스크립트
```python
from huggingface_hub import snapshot_download

# 전체 저장소 다운로드
snapshot_download(
    repo_id="levihsu/OOTDiffusion",
    local_dir="./ootdiffusion_full",
    local_dir_use_symlinks=False
)
```

## 📦 필수 추가 모델들

### CLIP 모델 (필수)
```bash
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./clip-vit-large-patch14
```

### Stable Diffusion v1.5 (기본 모델)
```bash
huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir ./stable-diffusion-v1-5
```

## 🔧 환경 설정

### 필수 패키지 설치
```bash
pip install torch torchvision torchaudio
pip install transformers diffusers accelerate
pip install opencv-python pillow numpy
pip install onnxruntime
```

## 📁 최종 디렉토리 구조
```
ai_models/
├── OOTDiffusion/                 # GitHub 클론
├── checkpoints/
│   ├── ootdiffusion_hf/         # HuggingFace 다운로드
│   ├── clip-vit-large-patch14/  # CLIP 모델
│   ├── stable-diffusion-v1-5/   # SD 1.5 모델
│   ├── densepose/               # 인체 분석
│   └── sam/                     # 세그멘테이션
└── model_config.yaml
```

## 🆘 문제 해결

### Git LFS 에러
```bash
git lfs install
git lfs pull
```

### 용량 부족
- 최소 15GB 여유 공간 필요
- 외장 드라이브 사용 고려

### 네트워크 에러
- VPN 사용 시 끄고 다운로드
- 여러 번 재시도

## ✅ 다운로드 확인
```bash
# 파일 존재 확인
find ai_models -name "*.safetensors" -o -name "*.bin" -o -name "*.onnx"

# 용량 확인
du -sh ai_models/
```
"""
        
        guide_path = self.base_path / "DOWNLOAD_GUIDE.md"
        with open(guide_path, 'w') as f:
            f.write(guide_content)
        
        logger.info(f"📖 다운로드 가이드 생성: {guide_path}")
    
    def create_simplified_config(self):
        """간소화된 설정 파일 생성"""
        config_content = f"""# MyCloset AI - 간소화된 모델 설정

models:
  # GitHub에서 클론한 OOTDiffusion
  ootdiffusion_github:
    enabled: true
    path: "{self.base_path}/OOTDiffusion"
    type: "github_repo"
    
  # HuggingFace에서 다운로드한 OOTDiffusion  
  ootdiffusion_hf:
    enabled: true
    path: "{self.checkpoints_dir}/ootdiffusion_hf"
    type: "huggingface"
    
  # CLIP 모델 (필수)
  clip:
    enabled: true
    path: "{self.checkpoints_dir}/clip-vit-large-patch14"
    
  # Stable Diffusion 기본 모델
  stable_diffusion:
    enabled: true
    path: "{self.checkpoints_dir}/stable-diffusion-v1-5"

device:
  type: "mps"  # Apple Silicon M3 Max
  fallback: "cpu"
  
processing:
  image_size: 512
  batch_size: 1
  use_safetensors: true
"""
        
        config_path = self.base_path / "simple_config.yaml"
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        logger.info(f"✅ 간소화된 설정 파일 생성: {config_path}")
    
    def verify_downloads_fixed(self):
        """수정된 다운로드 검증"""
        logger.info("🔍 다운로드 검증 중...")
        
        check_paths = [
            ("GitHub OOTDiffusion", self.base_path / "OOTDiffusion"),
            ("HF OOTDiffusion", self.checkpoints_dir / "ootdiffusion_hf"),
            ("CLIP 모델", self.checkpoints_dir / "clip-vit-large-patch14"),
            ("DensePose", self.checkpoints_dir / "densepose"),
            ("SAM", self.checkpoints_dir / "sam")
        ]
        
        success_count = 0
        total_size = 0
        
        for name, path in check_paths:
            if path.exists():
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                total_size += size
                logger.info(f"✅ {name}: {size/1024/1024:.1f}MB")
                success_count += 1
            else:
                logger.warning(f"⚠️ 누락: {name}")
        
        logger.info(f"📊 총 다운로드 크기: {total_size/1024/1024/1024:.2f}GB")
        logger.info(f"🎯 성공률: {success_count}/{len(check_paths)} 모델")
        
        return success_count >= 2  # 최소 2개 모델이 있으면 성공
    
    def download_all_fixed(self):
        """수정된 전체 다운로드"""
        logger.info("🚀 수정된 모델 다운로드 시작...")
        
        # 1. 필수 패키지 설치
        self.install_requirements()
        
        # 2. 디렉토리 구조 설정
        self.setup_ootdiffusion_structure()
        
        print("\n" + "="*60)
        print("🎽 OOTDiffusion 다운로드 (복수 방법 시도)")
        print("="*60)
        
        # 3. OOTDiffusion 다운로드 (여러 방법 시도)
        ootd_success = False
        
        # 방법 1: GitHub 클론
        if self.download_ootdiffusion_git_clone():
            ootd_success = True
            
        # 방법 2: HuggingFace (병행)
        if self.download_ootdiffusion_huggingface():
            ootd_success = True
        
        # 4. CLIP 모델 다운로드 (필수)
        print("\n" + "="*60)
        print("🎨 CLIP 모델 다운로드")
        print("="*60)
        self.download_clip_model()
        
        # 5. 대체 모델들
        print("\n" + "="*60)
        print("🔄 대체 모델들")
        print("="*60)
        self.download_alternative_models()
        
        # 6. 가이드 및 설정 파일 생성
        self.create_model_download_guide()
        self.create_simplified_config()
        
        # 7. 최종 검증
        print("\n" + "="*60)
        print("🔍 최종 검증")
        print("="*60)
        verification_passed = self.verify_downloads_fixed()
        
        # 결과 출력
        print("\n" + "="*60)
        print("📊 다운로드 결과")
        print("="*60)
        
        if verification_passed:
            print("\n🎉 모델 다운로드 완료!")
            print(f"📁 모델 위치: {self.base_path}")
            print(f"📖 가이드: {self.base_path}/DOWNLOAD_GUIDE.md")
            print("\n🔧 다음 단계:")
            print("1. OOTDiffusion이 없다면 수동 다운로드 가이드 참조")
            print("2. 백엔드 서버 실행 테스트")
            print("3. 프론트엔드 연동 테스트")
        else:
            print("\n⚠️ 일부 모델이 다운로드되지 않았습니다.")
            print("📖 수동 다운로드 가이드를 참조하세요:")
            print(f"   {self.base_path}/DOWNLOAD_GUIDE.md")

def main():
    print("🔧 MyCloset AI - 수정된 모델 다운로더")
    print("="*60)
    
    # 경로 확인
    default_path = "/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models"
    print(f"📁 모델 저장 경로: {default_path}")
    
    # 다운로더 실행
    downloader = FixedModelDownloader(default_path)
    
    print("\n🚀 다운로드 시작하시겠습니까?")
    print("- GitHub 저장소 클론")
    print("- HuggingFace 모델 다운로드") 
    print("- 필수 모델들 (CLIP, Stable Diffusion)")
    print("- 수동 다운로드 가이드 생성")
    
    confirm = input("\n계속 진행하시겠습니까? (y/N): ").strip().lower()
    
    if confirm == 'y':
        downloader.download_all_fixed()
    else:
        print("다운로드가 취소되었습니다.")
        print("수동 다운로드 가이드만 생성합니다...")
        downloader.create_model_download_guide()

if __name__ == "__main__":
    main()