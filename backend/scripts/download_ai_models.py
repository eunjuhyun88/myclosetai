# backend/scripts/download_ai_models.py
"""
MyCloset AI - 고품질 AI 모델 다운로드 스크립트
OOTDiffusion, VITON-HD, DensePose 등 최신 모델들 설치
"""

import os
import sys
import requests
import zipfile
import tarfile
import shutil
from pathlib import Path
import subprocess
import logging
from tqdm import tqdm
import torch
import gdown  # Google Drive 다운로드용

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIModelDownloader:
    def __init__(self):
        self.base_dir = Path("ai_models")
        self.base_dir.mkdir(exist_ok=True)
        
        # 모델 디렉토리들
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.configs_dir = self.base_dir / "configs"
        self.temp_dir = self.base_dir / "temp"
        
        for dir_path in [self.checkpoints_dir, self.configs_dir, self.temp_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def download_file(self, url: str, filepath: Path, description: str = ""):
        """진행률 표시와 함께 파일 다운로드"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as file, tqdm(
                desc=description,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"✅ 다운로드 완료: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 다운로드 실패 {url}: {e}")
            return False
    
    def download_ootdiffusion(self):
        """OOTDiffusion 모델 다운로드"""
        logger.info("🤖 OOTDiffusion 모델 다운로드 시작...")
        
        ootd_dir = self.checkpoints_dir / "ootdiffusion"
        ootd_dir.mkdir(exist_ok=True)
        
        # OOTDiffusion 모델 URL들 (실제 URL은 GitHub에서 확인 필요)
        models = {
            "ootd_humanparsing_onnx.zip": "https://huggingface.co/levihsu/OOTDiffusion/resolve/main/ootd_humanparsing_onnx.zip",
            "ootd_diffusion_model.safetensors": "https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/ootd/ootd_diffusion_model.safetensors",
            "vae_ootd.safetensors": "https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/ootd/vae_ootd.safetensors"
        }
        
        for filename, url in models.items():
            filepath = ootd_dir / filename
            if not filepath.exists():
                self.download_file(url, filepath, f"OOTDiffusion - {filename}")
            else:
                logger.info(f"⏭️ 이미 존재: {filepath}")
        
        # 설정 파일 생성
        config_content = """
# OOTDiffusion 설정
model_type: "ootdiffusion"
device: "mps"  # M3 Max
dtype: "float32"
checkpoint_path: "ai_models/checkpoints/ootdiffusion"
human_parsing_path: "ai_models/checkpoints/ootdiffusion/ootd_humanparsing_onnx"
vae_path: "ai_models/checkpoints/ootdiffusion/vae_ootd.safetensors"
"""
        config_path = self.configs_dir / "ootdiffusion.yaml"
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        logger.info("✅ OOTDiffusion 설정 완료")
    
    def download_viton_hd(self):
        """VITON-HD 모델 다운로드"""
        logger.info("🤖 VITON-HD 모델 다운로드 시작...")
        
        viton_dir = self.checkpoints_dir / "viton_hd"
        viton_dir.mkdir(exist_ok=True)
        
        # VITON-HD 모델들
        models = {
            "seg_model.pth": "https://drive.google.com/uc?id=1mhF3_vQSVZZ5QwQlEKhNRrz5dNGSLCU4",
            "gmm_model.pth": "https://drive.google.com/uc?id=1Z7mQzQaHKsQgweLOjNLV-1VoCm_CXKrF",
            "tom_model.pth": "https://drive.google.com/uc?id=1YwovS9d7LwGHBqJYl7Hf9SYKdlXQJHnL"
        }
        
        for filename, file_id in models.items():
            filepath = viton_dir / filename
            if not filepath.exists():
                try:
                    gdown.download(file_id, str(filepath), quiet=False)
                    logger.info(f"✅ VITON-HD 다운로드 완료: {filename}")
                except Exception as e:
                    logger.error(f"❌ VITON-HD 다운로드 실패 {filename}: {e}")
            else:
                logger.info(f"⏭️ 이미 존재: {filepath}")
    
    def download_densepose(self):
        """DensePose 모델 다운로드"""
        logger.info("🤖 DensePose 모델 다운로드 시작...")
        
        densepose_dir = self.checkpoints_dir / "densepose"
        densepose_dir.mkdir(exist_ok=True)
        
        # DensePose 모델
        model_url = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
        model_path = densepose_dir / "model_final_162be9.pkl"
        
        if not model_path.exists():
            self.download_file(model_url, model_path, "DensePose")
        else:
            logger.info(f"⏭️ DensePose 이미 존재")
    
    def download_openpose(self):
        """OpenPose 모델 다운로드"""
        logger.info("🤖 OpenPose 모델 다운로드 시작...")
        
        openpose_dir = self.checkpoints_dir / "openpose"
        openpose_dir.mkdir(exist_ok=True)
        
        # OpenPose body_25 모델
        models = {
            "body_pose_model.pth": "https://www.dropbox.com/s/5v654d2u65fuvyr/body_pose_model.pth?dl=1",
            "hand_pose_model.pth": "https://www.dropbox.com/s/s4uck3lhhzw7hx6/hand_pose_model.pth?dl=1"
        }
        
        for filename, url in models.items():
            filepath = openpose_dir / filename
            if not filepath.exists():
                self.download_file(url, filepath, f"OpenPose - {filename}")
            else:
                logger.info(f"⏭️ 이미 존재: {filepath}")
    
    def install_additional_packages(self):
        """추가 패키지 설치"""
        logger.info("📦 AI 모델용 추가 패키지 설치...")
        
        packages = [
            "diffusers>=0.21.0",
            "transformers>=4.35.0", 
            "accelerate>=0.24.0",
            "xformers",  # M3 Max에서 가능한지 확인 필요
            "onnxruntime",
            "gdown",
            "detectron2 @ git+https://github.com/facebookresearch/detectron2.git",
            "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git"
        ]
        
        for package in packages:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package], 
                             check=True, capture_output=True)
                logger.info(f"✅ 설치 완료: {package}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️ 설치 실패 (선택사항): {package}")
    
    def setup_model_configs(self):
        """모델 설정 파일들 생성"""
        logger.info("⚙️ 모델 설정 파일 생성...")
        
        # 통합 설정 파일
        config = {
            "models": {
                "ootdiffusion": {
                    "enabled": True,
                    "path": "ai_models/checkpoints/ootdiffusion",
                    "device": "mps",
                    "dtype": "float32"
                },
                "viton_hd": {
                    "enabled": True,
                    "path": "ai_models/checkpoints/viton_hd", 
                    "device": "mps",
                    "dtype": "float32"
                },
                "densepose": {
                    "enabled": True,
                    "path": "ai_models/checkpoints/densepose",
                    "device": "mps"
                },
                "openpose": {
                    "enabled": True,
                    "path": "ai_models/checkpoints/openpose",
                    "device": "mps"
                }
            },
            "processing": {
                "image_size": 512,
                "batch_size": 1,
                "num_inference_steps": 20,
                "guidance_scale": 7.5
            }
        }
        
        import yaml
        config_path = self.configs_dir / "models_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"✅ 설정 파일 생성: {config_path}")
    
    def verify_installation(self):
        """설치 검증"""
        logger.info("🔍 설치 검증 중...")
        
        # PyTorch MPS 테스트
        try:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
                x = torch.randn(100, 100).to(device)
                y = torch.randn(100, 100).to(device)
                z = torch.mm(x, y)
                logger.info("✅ M3 Max GPU (MPS) 정상 동작")
            else:
                logger.warning("⚠️ MPS 사용 불가, CPU 모드")
        except Exception as e:
            logger.error(f"❌ GPU 테스트 실패: {e}")
        
        # 모델 파일 존재 확인
        model_paths = [
            self.checkpoints_dir / "ootdiffusion",
            self.checkpoints_dir / "viton_hd", 
            self.checkpoints_dir / "densepose",
            self.checkpoints_dir / "openpose"
        ]
        
        for path in model_paths:
            if path.exists() and any(path.iterdir()):
                logger.info(f"✅ 모델 확인: {path.name}")
            else:
                logger.warning(f"⚠️ 모델 누락: {path.name}")
    
    def download_all(self):
        """전체 모델 다운로드"""
        logger.info("🚀 AI 모델 전체 다운로드 시작...")
        
        try:
            # 1. 추가 패키지 설치
            self.install_additional_packages()
            
            # 2. 모델들 다운로드
            self.download_ootdiffusion()
            self.download_viton_hd()
            self.download_densepose()
            self.download_openpose()
            
            # 3. 설정 파일 생성
            self.setup_model_configs()
            
            # 4. 설치 검증
            self.verify_installation()
            
            logger.info("🎉 모든 AI 모델 설치 완료!")
            
        except Exception as e:
            logger.error(f"❌ 설치 중 오류: {e}")

def main():
    """메인 실행 함수"""
    print("🤖 MyCloset AI - 고품질 AI 모델 다운로더")
    print("=" * 50)
    
    downloader = AIModelDownloader()
    
    # 사용자 선택
    choice = input("""
어떤 모델을 다운로드하시겠습니까?
1. 전체 모델 (권장) - 10GB+ 필요
2. OOTDiffusion만 (최신 고품질)
3. VITON-HD만 (빠른 처리)
4. 설정만 업데이트

선택 (1-4): """).strip()
    
    if choice == "1":
        downloader.download_all()
    elif choice == "2":
        downloader.download_ootdiffusion()
    elif choice == "3":
        downloader.download_viton_hd()
    elif choice == "4":
        downloader.setup_model_configs()
    else:
        print("잘못된 선택입니다.")

if __name__ == "__main__":
    main()