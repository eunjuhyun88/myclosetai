#!/usr/bin/env python3
# backend/scripts/download_ai_models.py
"""
MyCloset AI - AI 모델 자동 다운로더
M3 Max 최적화 가상 피팅 시스템용 모델들
"""

import os
import sys
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import requests
from tqdm import tqdm
import hashlib

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'model_download.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

class AIModelDownloader:
    """AI 모델 자동 다운로더"""
    
    def __init__(self):
        """다운로더 초기화"""
        self.project_root = PROJECT_ROOT
        self.models_dir = self.project_root / "ai_models"
        self.checkpoints_dir = self.models_dir / "checkpoints"
        self.temp_dir = self.models_dir / "temp"
        
        # 디렉토리 생성
        self.models_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
        # 모델 정보 설정
        self.model_configs = self._setup_model_configs()
        
        logger.info(f"📁 모델 디렉토리: {self.models_dir}")
        logger.info(f"💾 체크포인트 디렉토리: {self.checkpoints_dir}")
    
    def _setup_model_configs(self) -> Dict:
        """모델 설정 정보 반환"""
        
        return {
            "ootdiffusion": {
                "name": "OOTDiffusion",
                "description": "최신 고품질 가상 피팅 모델",
                "size_gb": 4.2,
                "priority": 1,
                "huggingface_repo": "levihsu/OOTDiffusion",
                "local_path": self.checkpoints_dir / "ootdiffusion",
                "required_files": [
                    "model_index.json",
                    "unet/diffusion_pytorch_model.safetensors",
                    "vae/diffusion_pytorch_model.safetensors",
                    "text_encoder/pytorch_model.bin",
                ],
                "download_method": "huggingface"
            },
            
            "viton_hd": {
                "name": "VITON-HD",
                "description": "고해상도 가상 시착 모델",
                "size_gb": 2.8,
                "priority": 2,
                "github_repo": "shadow2496/VITON-HD",
                "local_path": self.checkpoints_dir / "viton_hd",
                "required_files": [
                    "gen.pt",
                    "seg.pt",
                    "pose.pt"
                ],
                "download_method": "github"
            },
            
            "densepose": {
                "name": "DensePose",
                "description": "인체 파싱 및 자세 추정",
                "size_gb": 1.5,
                "priority": 3,
                "model_zoo_url": "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x.pkl",
                "local_path": self.checkpoints_dir / "densepose",
                "required_files": [
                    "densepose_rcnn_R_50_FPN_s1x.pkl",
                    "config.yaml"
                ],
                "download_method": "direct"
            },
            
            "openpose": {
                "name": "OpenPose",
                "description": "실시간 자세 추정",
                "size_gb": 0.8,
                "priority": 4,
                "models": {
                    "pose_coco": "https://storage.googleapis.com/openimages/web/pose_coco.pth",
                    "pose_body_25": "https://storage.googleapis.com/openimages/web/pose_body_25.pth",
                },
                "local_path": self.checkpoints_dir / "openpose",
                "required_files": [
                    "pose_coco.pth",
                    "pose_body_25.pth"
                ],
                "download_method": "direct"
            },
            
            "segment_anything": {
                "name": "Segment Anything (SAM)",
                "description": "범용 이미지 세그멘테이션",
                "size_gb": 2.4,
                "priority": 5,
                "checkpoint_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                "local_path": self.checkpoints_dir / "segment_anything",
                "required_files": [
                    "sam_vit_h_4b8939.pth"
                ],
                "download_method": "direct"
            }
        }
    
    def check_dependencies(self):
        """필요한 패키지 확인 및 설치"""
        
        logger.info("📦 의존성 패키지 확인 중...")
        
        required_packages = [
            "huggingface_hub",
            "transformers",
            "diffusers",
            "gitpython",
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                logger.info(f"✅ {package} 설치됨")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"❌ {package} 누락")
        
        if missing_packages:
            logger.info(f"📥 누락된 패키지 설치 중: {missing_packages}")
            for package in missing_packages:
                subprocess.run([sys.executable, "-m", "pip", "install", package], 
                             check=True)
            logger.info("✅ 의존성 설치 완료")
    
    def download_file_with_progress(self, url: str, filepath: Path, chunk_size: int = 8192) -> bool:
        """진행률 표시와 함께 파일 다운로드"""
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=filepath.name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        size = f.write(chunk)
                        pbar.update(size)
            
            logger.info(f"✅ 다운로드 완료: {filepath.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 다운로드 실패 {filepath.name}: {e}")
            if filepath.exists():
                filepath.unlink()
            return False
    
    def verify_file_integrity(self, filepath: Path, expected_size: Optional[int] = None) -> bool:
        """파일 무결성 검증"""
        
        if not filepath.exists():
            return False
        
        file_size = filepath.stat().st_size
        
        if expected_size and abs(file_size - expected_size) > expected_size * 0.05:  # 5% 오차 허용
            logger.warning(f"⚠️ 파일 크기 불일치: {filepath.name}")
            return False
        
        logger.info(f"✅ 파일 검증 통과: {filepath.name} ({file_size // 1024 // 1024}MB)")
        return True
    
    def download_from_huggingface(self, model_config: Dict) -> bool:
        """Hugging Face에서 모델 다운로드"""
        
        try:
            from huggingface_hub import snapshot_download
            
            repo_id = model_config["huggingface_repo"]
            local_dir = model_config["local_path"]
            
            logger.info(f"📥 Hugging Face에서 다운로드: {repo_id}")
            
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            
            # 필수 파일 존재 확인
            for required_file in model_config["required_files"]:
                file_path = local_dir / required_file
                if not file_path.exists():
                    logger.error(f"❌ 필수 파일 누락: {required_file}")
                    return False
            
            logger.info(f"✅ {model_config['name']} 다운로드 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ Hugging Face 다운로드 실패: {e}")
            return False
    
    def download_from_github(self, model_config: Dict) -> bool:
        """GitHub에서 모델 다운로드"""
        
        try:
            import git
            
            repo_url = f"https://github.com/{model_config['github_repo']}.git"
            local_dir = model_config["local_path"]
            
            logger.info(f"📥 GitHub에서 클론: {repo_url}")
            
            if local_dir.exists():
                shutil.rmtree(local_dir)
            
            git.Repo.clone_from(repo_url, local_dir, depth=1)
            
            logger.info(f"✅ {model_config['name']} 클론 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ GitHub 클론 실패: {e}")
            return False
    
    def download_direct_files(self, model_config: Dict) -> bool:
        """직접 URL에서 파일 다운로드"""
        
        local_dir = model_config["local_path"]
        local_dir.mkdir(parents=True, exist_ok=True)
        
        success = True
        
        # 단일 URL인 경우
        if "checkpoint_url" in model_config:
            url = model_config["checkpoint_url"]
            filename = Path(url).name
            filepath = local_dir / filename
            
            if not self.download_file_with_progress(url, filepath):
                success = False
        
        # 여러 모델 URL인 경우
        elif "models" in model_config:
            for model_name, url in model_config["models"].items():
                filename = Path(url).name
                filepath = local_dir / filename
                
                if not self.download_file_with_progress(url, filepath):
                    success = False
        
        # 단일 모델 URL인 경우
        elif "model_zoo_url" in model_config:
            url = model_config["model_zoo_url"]
            filename = Path(url).name
            filepath = local_dir / filename
            
            if not self.download_file_with_progress(url, filepath):
                success = False
        
        if success:
            logger.info(f"✅ {model_config['name']} 다운로드 완료")
        
        return success
    
    def download_model(self, model_key: str) -> bool:
        """특정 모델 다운로드"""
        
        if model_key not in self.model_configs:
            logger.error(f"❌ 알 수 없는 모델: {model_key}")
            return False
        
        model_config = self.model_configs[model_key]
        
        logger.info(f"🚀 {model_config['name']} 다운로드 시작...")
        logger.info(f"📝 설명: {model_config['description']}")
        logger.info(f"💾 예상 크기: {model_config['size_gb']}GB")
        
        # 이미 다운로드된 경우 확인
        if self.is_model_downloaded(model_key):
            logger.info(f"✅ {model_config['name']} 이미 다운로드됨")
            return True
        
        # 다운로드 방법에 따라 실행
        download_method = model_config["download_method"]
        
        if download_method == "huggingface":
            return self.download_from_huggingface(model_config)
        elif download_method == "github":
            return self.download_from_github(model_config)
        elif download_method == "direct":
            return self.download_direct_files(model_config)
        else:
            logger.error(f"❌ 지원하지 않는 다운로드 방법: {download_method}")
            return False
    
    def is_model_downloaded(self, model_key: str) -> bool:
        """모델 다운로드 여부 확인"""
        
        model_config = self.model_configs[model_key]
        local_path = model_config["local_path"]
        
        if not local_path.exists():
            return False
        
        # 필수 파일 존재 확인
        for required_file in model_config["required_files"]:
            file_path = local_path / required_file
            if not file_path.exists():
                return False
        
        return True
    
    def download_essential_models(self) -> bool:
        """필수 모델들 다운로드"""
        
        essential_models = ["ootdiffusion", "densepose", "openpose"]
        
        logger.info("🎯 필수 모델 다운로드 시작...")
        
        success_count = 0
        
        for model_key in essential_models:
            if self.download_model(model_key):
                success_count += 1
            else:
                logger.error(f"❌ 필수 모델 다운로드 실패: {model_key}")
        
        if success_count == len(essential_models):
            logger.info("✅ 모든 필수 모델 다운로드 완료")
            return True
        else:
            logger.error(f"❌ 필수 모델 다운로드 실패: {success_count}/{len(essential_models)}")
            return False
    
    def download_all_models(self) -> bool:
        """모든 모델 다운로드"""
        
        logger.info("🌟 전체 모델 다운로드 시작...")
        
        # 우선순위 순으로 정렬
        sorted_models = sorted(
            self.model_configs.items(), 
            key=lambda x: x[1]["priority"]
        )
        
        success_count = 0
        total_size = sum(config["size_gb"] for _, config in sorted_models)
        
        logger.info(f"📊 총 다운로드 크기: {total_size:.1f}GB")
        
        for model_key, model_config in sorted_models:
            if self.download_model(model_key):
                success_count += 1
        
        if success_count == len(sorted_models):
            logger.info("🎉 모든 모델 다운로드 완료!")
            return True
        else:
            logger.warning(f"⚠️ 일부 모델 다운로드 실패: {success_count}/{len(sorted_models)}")
            return False
    
    def create_model_config_file(self):
        """모델 설정 파일 생성"""
        
        import yaml
        
        config = {
            "models": {
                model_key: {
                    "name": config["name"],
                    "path": str(config["local_path"]),
                    "device": "mps",  # M3 Max 기본값
                    "enabled": self.is_model_downloaded(model_key)
                }
                for model_key, config in self.model_configs.items()
            },
            "processing": {
                "image_size": 512,
                "batch_size": 1,
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "device": "mps"
            }
        }
        
        config_path = self.models_dir / "models_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"✅ 모델 설정 파일 생성: {config_path}")
    
    def get_download_status(self) -> Dict:
        """다운로드 상태 확인"""
        
        status = {
            "total_models": len(self.model_configs),
            "downloaded": 0,
            "missing": [],
            "total_size_gb": 0,
            "downloaded_size_gb": 0,
        }
        
        for model_key, model_config in self.model_configs.items():
            status["total_size_gb"] += model_config["size_gb"]
            
            if self.is_model_downloaded(model_key):
                status["downloaded"] += 1
                status["downloaded_size_gb"] += model_config["size_gb"]
            else:
                status["missing"].append(model_key)
        
        return status
    
    def cleanup_temp_files(self):
        """임시 파일 정리"""
        
        logger.info("🧹 임시 파일 정리 중...")
        
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir()
        
        logger.info("✅ 임시 파일 정리 완료")

def main():
    """메인 실행 함수"""
    
    print("🤖 MyCloset AI - AI 모델 다운로더")
    print("M3 Max 최적화 가상 피팅 시스템")
    print("=" * 50)
    
    downloader = AIModelDownloader()
    
    # 의존성 확인
    downloader.check_dependencies()
    
    # 현재 상태 확인
    status = downloader.get_download_status()
    print(f"\n📊 현재 상태:")
    print(f"  전체 모델: {status['total_models']}")
    print(f"  다운로드됨: {status['downloaded']}")
    print(f"  누락됨: {len(status['missing'])}")
    print(f"  전체 크기: {status['total_size_gb']:.1f}GB")
    print(f"  다운로드된 크기: {status['downloaded_size_gb']:.1f}GB")
    
    if status['missing']:
        print(f"  누락된 모델: {', '.join(status['missing'])}")
    
    # 사용자 선택
    print(f"\n다운로드 옵션:")
    print("1. 필수 모델만 (OOTDiffusion, DensePose, OpenPose) - 6.5GB")
    print("2. 전체 모델 (권장) - 11.7GB")
    print("3. 개별 모델 선택")
    print("4. 상태 확인만")
    print("0. 종료")
    
    try:
        choice = input("\n선택 (1-4, 0): ").strip()
        
        if choice == "1":
            print("\n🎯 필수 모델 다운로드 시작...")
            success = downloader.download_essential_models()
            
        elif choice == "2":
            print("\n🌟 전체 모델 다운로드 시작...")
            success = downloader.download_all_models()
            
        elif choice == "3":
            print("\n사용 가능한 모델:")
            for i, (key, config) in enumerate(downloader.model_configs.items(), 1):
                downloaded = "✅" if downloader.is_model_downloaded(key) else "❌"
                print(f"  {i}. {config['name']} - {config['size_gb']}GB {downloaded}")
            
            model_num = input("다운로드할 모델 번호: ").strip()
            
            try:
                model_keys = list(downloader.model_configs.keys())
                selected_key = model_keys[int(model_num) - 1]
                success = downloader.download_model(selected_key)
            except (ValueError, IndexError):
                print("❌ 잘못된 번호입니다.")
                return
                
        elif choice == "4":
            print("✅ 상태 확인 완료")
            return
            
        elif choice == "0":
            print("👋 다운로더를 종료합니다.")
            return
            
        else:
            print("❌ 잘못된 선택입니다.")
            return
        
        # 설정 파일 생성
        downloader.create_model_config_file()
        
        # 임시 파일 정리
        downloader.cleanup_temp_files()
        
        # 최종 상태 출력
        final_status = downloader.get_download_status()
        print(f"\n🎉 작업 완료!")
        print(f"📊 최종 상태: {final_status['downloaded']}/{final_status['total_models']} 모델")
        print(f"💾 다운로드된 크기: {final_status['downloaded_size_gb']:.1f}GB")
        
        if final_status['downloaded'] > 0:
            print(f"\n✅ 다음 단계: 개발 서버 실행")
            print(f"cd backend && python run_server.py")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자가 취소했습니다.")
    except Exception as e:
        logger.error(f"❌ 예상치 못한 오류: {e}")

if __name__ == "__main__":
    main()