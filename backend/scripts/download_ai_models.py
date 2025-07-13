# scripts/download_ai_models.py
"""
실제 AI 모델 자동 다운로드 및 설정 스크립트
OOTDiffusion, VITON-HD, Graphonomy 등 실제 모델들을 다운로드
"""

import os
import requests
import subprocess
import zipfile
import gdown
from pathlib import Path
from tqdm import tqdm
import yaml
import logging
from huggingface_hub import snapshot_download, hf_hub_download
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIModelDownloader:
    def __init__(self, base_dir="backend"):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "ai_models"
        self.checkpoints_dir = self.models_dir / "checkpoints"
        self.configs_dir = self.models_dir / "configs"
        
        # 디렉토리 생성
        self.models_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.configs_dir.mkdir(exist_ok=True)
        
        # 시스템 정보
        self.device = self._detect_device()
        logger.info(f"🖥️ 감지된 디바이스: {self.device}")
    
    def _detect_device(self):
        """시스템 디바이스 감지"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        else:
            return "cpu"
    
    def download_file(self, url, filepath, desc=None):
        """파일 다운로드 with 진행률 표시"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as file, tqdm(
                desc=desc or f"Downloading {filepath.name}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    pbar.update(size)
            
            logger.info(f"✅ 다운로드 완료: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 다운로드 실패 {filepath}: {e}")
            return False
    
    def download_ootdiffusion(self):
        """OOTDiffusion 모델 다운로드"""
        logger.info("🤖 OOTDiffusion 모델 다운로드 시작...")
        
        ootd_dir = self.checkpoints_dir / "ootdiffusion"
        ootd_dir.mkdir(exist_ok=True)
        
        # OOTDiffusion 모델 URL들 (Hugging Face)
        models = {
            "ootd_humanparsing_onnx.zip": "levihsu/OOTDiffusion",
            "ootd/ootd_diffusion_model.safetensors": "levihsu/OOTDiffusion", 
            "ootd/vae_ootd.safetensors": "levihsu/OOTDiffusion"
        }
        
        try:
            # Hugging Face에서 직접 다운로드
            logger.info("📥 Hugging Face에서 OOTDiffusion 다운로드...")
            snapshot_download(
                repo_id="levihsu/OOTDiffusion",
                local_dir=str(ootd_dir),
                allow_patterns=["*.safetensors", "*.onnx", "*.json", "*.txt"]
            )
            
            # 설정 파일 생성
            config_content = {
                "model_type": "ootdiffusion",
                "device": self.device,
                "dtype": "float32" if self.device == "mps" else "float16",
                "checkpoint_path": str(ootd_dir),
                "human_parsing_path": str(ootd_dir / "ootd_humanparsing_onnx"),
                "vae_path": str(ootd_dir / "ootd" / "vae_ootd.safetensors"),
                "enabled": True
            }
            
            config_path = self.configs_dir / "ootdiffusion.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config_content, f, default_flow_style=False)
            
            logger.info("✅ OOTDiffusion 설정 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ OOTDiffusion 다운로드 실패: {e}")
            return False
    
    def download_viton_hd(self):
        """VITON-HD 모델 다운로드"""
        logger.info("🤖 VITON-HD 모델 다운로드 시작...")
        
        viton_dir = self.checkpoints_dir / "viton_hd"
        viton_dir.mkdir(exist_ok=True)
        
        # VITON-HD GitHub 클론
        try:
            if not (viton_dir / ".git").exists():
                logger.info("📥 VITON-HD GitHub 클론...")
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/shadow2496/VITON-HD.git",
                    str(viton_dir)
                ], check=True)
            
            # 사전 훈련된 가중치 다운로드 (Google Drive)
            weights = {
                "seg_model.pth": "1mhF3_vQSVZZ5QwQlEKhNRrz5dNGSLCU4",
                "gmm_model.pth": "1euphqABryn1xQRMWpXCl7zPYKZDK9O4r", 
                "tom_model.pth": "1S2tbtdLlBR4ZFZHcNtbG9t-xn-1KoHfY"
            }
            
            for filename, file_id in weights.items():
                filepath = viton_dir / "checkpoints" / filename
                filepath.parent.mkdir(exist_ok=True)
                
                if not filepath.exists():
                    logger.info(f"📥 {filename} 다운로드 중...")
                    gdown.download(f"https://drive.google.com/uc?id={file_id}", str(filepath))
            
            # 설정 파일 생성
            config_content = {
                "model_type": "viton_hd",
                "device": self.device,
                "checkpoint_path": str(viton_dir / "checkpoints"),
                "seg_model": str(viton_dir / "checkpoints" / "seg_model.pth"),
                "gmm_model": str(viton_dir / "checkpoints" / "gmm_model.pth"),
                "tom_model": str(viton_dir / "checkpoints" / "tom_model.pth"),
                "enabled": True
            }
            
            config_path = self.configs_dir / "viton_hd.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config_content, f, default_flow_style=False)
            
            logger.info("✅ VITON-HD 설정 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ VITON-HD 다운로드 실패: {e}")
            return False
    
    def download_human_parsing(self):
        """인체 파싱 모델 다운로드 (Graphonomy)"""
        logger.info("🤖 Human Parsing 모델 다운로드 시작...")
        
        parsing_dir = self.checkpoints_dir / "human_parsing"
        parsing_dir.mkdir(exist_ok=True)
        
        try:
            # Self-Correction Human Parsing 다운로드 (더 정확함)
            logger.info("📥 Self-Correction Human Parsing 다운로드...")
            
            # ATR 데이터셋 모델
            atr_model_url = "https://github.com/PeikeLi/Self-Correction-Human-Parsing/releases/download/v1.0/exp-schp-201908261155-atr.pth"
            atr_path = parsing_dir / "atr_model.pth"
            
            if not atr_path.exists():
                self.download_file(atr_model_url, atr_path, "ATR Parsing Model")
            
            # LIP 데이터셋 모델
            lip_model_url = "https://github.com/PeikeLi/Self-Correction-Human-Parsing/releases/download/v1.0/exp-schp-201908301523-lip.pth"
            lip_path = parsing_dir / "lip_model.pth"
            
            if not lip_path.exists():
                self.download_file(lip_model_url, lip_path, "LIP Parsing Model")
            
            # 설정 파일 생성
            config_content = {
                "model_type": "human_parsing",
                "device": self.device,
                "atr_model": str(atr_path),
                "lip_model": str(lip_path),
                "input_size": [473, 473],
                "num_classes": 18,
                "enabled": True
            }
            
            config_path = self.configs_dir / "human_parsing.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config_content, f, default_flow_style=False)
            
            logger.info("✅ Human Parsing 설정 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ Human Parsing 다운로드 실패: {e}")
            return False
    
    def download_background_removal(self):
        """배경 제거 모델 다운로드"""
        logger.info("🤖 배경 제거 모델 다운로드 시작...")
        
        bg_dir = self.checkpoints_dir / "background_removal"
        bg_dir.mkdir(exist_ok=True)
        
        try:
            # U2-Net 모델 다운로드
            u2net_url = "https://github.com/xuebinqin/U-2-Net/releases/download/u2net/u2net.pth"
            u2net_path = bg_dir / "u2net.pth"
            
            if not u2net_path.exists():
                self.download_file(u2net_url, u2net_path, "U2-Net Model")
            
            # 설정 파일 생성
            config_content = {
                "model_type": "background_removal",
                "device": self.device,
                "u2net_model": str(u2net_path),
                "input_size": [320, 320],
                "enabled": True
            }
            
            config_path = self.configs_dir / "background_removal.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config_content, f, default_flow_style=False)
            
            logger.info("✅ 배경 제거 모델 설정 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 배경 제거 모델 다운로드 실패: {e}")
            return False
    
    def create_master_config(self):
        """마스터 설정 파일 생성"""
        master_config = {
            "system": {
                "device": self.device,
                "models_dir": str(self.models_dir),
                "checkpoints_dir": str(self.checkpoints_dir)
            },
            "models": {
                "ootdiffusion": {
                    "enabled": True,
                    "priority": 1,
                    "config_file": "ootdiffusion.yaml"
                },
                "viton_hd": {
                    "enabled": True,
                    "priority": 2,
                    "config_file": "viton_hd.yaml"
                },
                "human_parsing": {
                    "enabled": True,
                    "priority": 1,
                    "config_file": "human_parsing.yaml"
                },
                "background_removal": {
                    "enabled": True,
                    "priority": 1,
                    "config_file": "background_removal.yaml"
                }
            },
            "processing": {
                "default_model": "ootdiffusion",
                "fallback_model": "viton_hd",
                "max_image_size": 1024,
                "batch_size": 1
            }
        }
        
        master_path = self.configs_dir / "models_config.yaml"
        with open(master_path, 'w') as f:
            yaml.dump(master_config, f, default_flow_style=False)
        
        logger.info(f"✅ 마스터 설정 파일 생성: {master_path}")
    
    def download_all(self):
        """모든 모델 다운로드"""
        logger.info("🚀 AI 모델 통합 다운로드 시작...")
        
        success_count = 0
        total_models = 4
        
        # 1. OOTDiffusion
        if self.download_ootdiffusion():
            success_count += 1
        
        # 2. VITON-HD  
        if self.download_viton_hd():
            success_count += 1
        
        # 3. Human Parsing
        if self.download_human_parsing():
            success_count += 1
        
        # 4. Background Removal
        if self.download_background_removal():
            success_count += 1
        
        # 5. 마스터 설정 파일 생성
        self.create_master_config()
        
        logger.info(f"🎉 모델 다운로드 완료: {success_count}/{total_models}")
        
        if success_count == total_models:
            logger.info("✅ 모든 AI 모델이 성공적으로 다운로드되었습니다!")
            logger.info("📁 모델 위치:")
            logger.info(f"   - 체크포인트: {self.checkpoints_dir}")
            logger.info(f"   - 설정 파일: {self.configs_dir}")
        else:
            logger.warning(f"⚠️ 일부 모델 다운로드에 실패했습니다. ({success_count}/{total_models})")
        
        return success_count == total_models

def main():
    """메인 실행 함수"""
    print("🤖 MyCloset AI 모델 다운로드 시스템")
    print("=" * 50)
    
    try:
        downloader = AIModelDownloader()
        success = downloader.download_all()
        
        if success:
            print("\n🎉 설치 완료!")
            print("\n📋 다음 단계:")
            print("1. 의존성 설치: pip install -r requirements-ai.txt")
            print("2. 모델 테스트: python scripts/test_models.py")
            print("3. 서버 실행: uvicorn app.main:app --reload")
        else:
            print("\n❌ 일부 모델 다운로드가 실패했습니다.")
            print("로그를 확인하고 수동으로 다운로드해주세요.")
            
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"❌ 예상치 못한 오류: {e}")

if __name__ == "__main__":
    main()