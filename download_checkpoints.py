#!/usr/bin/env python3
"""
🤖 MyCloset AI - 모델 자동 다운로드 스크립트
M3 Max 128GB 최적화 버전

사용법:
1. python download_models.py --all          # 모든 모델 다운로드
2. python download_models.py --essential    # 필수 모델만
3. python download_models.py --model ootd   # 특정 모델만
"""

import os
import sys
import json
import time
import shutil
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import zipfile
import tarfile
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import argparse

# Hugging Face Hub
try:
    from huggingface_hub import hf_hub_download, snapshot_download, login
    from huggingface_hub.utils import HfHubHTTPError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️ huggingface_hub가 설치되지 않음. pip install huggingface_hub")

# Git LFS
def check_git_lfs():
    """Git LFS 설치 확인"""
    try:
        subprocess.run(["git", "lfs", "version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

class ModelDownloader:
    """AI 모델 다운로더"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.models_dir = base_dir / "ai_models"
        self.models_dir.mkdir(exist_ok=True, parents=True)
        
        # 필수 모델 정의 (우선순위 순)
        self.models_catalog = {
            # 🔥 필수 모델들 (가상 피팅용)
            "ootdiffusion": {
                "name": "OOTDiffusion",
                "priority": 1,
                "size_gb": 8.5,
                "required": True,
                "sources": [
                    {
                        "type": "huggingface",
                        "repo_id": "levihsu/OOTDiffusion",
                        "subfolder": None,
                        "local_dir": "checkpoints/ootdiffusion"
                    },
                    {
                        "type": "direct",
                        "url": "https://github.com/levihsu/OOTDiffusion/releases/download/v1.0/ootd_diffusion.zip",
                        "local_dir": "checkpoints/ootdiffusion"
                    }
                ],
                "description": "최신 고품질 가상 피팅 모델"
            },
            
            "stable_diffusion_inpaint": {
                "name": "Stable Diffusion Inpaint",
                "priority": 2,
                "size_gb": 4.2,
                "required": True,
                "sources": [
                    {
                        "type": "huggingface",
                        "repo_id": "runwayml/stable-diffusion-inpainting",
                        "subfolder": None,
                        "local_dir": "checkpoints/stable_diffusion_inpaint"
                    }
                ],
                "description": "인페인팅 전용 Stable Diffusion"
            },
            
            "human_parsing": {
                "name": "Human Parsing Model",
                "priority": 3,
                "size_gb": 0.3,
                "required": True,
                "sources": [
                    {
                        "type": "huggingface",
                        "repo_id": "mattmdjaga/segformer_b2_clothes",
                        "subfolder": None,
                        "local_dir": "checkpoints/human_parsing"
                    },
                    {
                        "type": "direct",
                        "url": "https://github.com/Engineering-Course/CIHP_PGN/releases/download/v1.0/schp_atr.pth",
                        "filename": "schp_atr.pth",
                        "local_dir": "checkpoints/human_parsing"
                    }
                ],
                "description": "인체 부위 분할 모델"
            },
            
            "pose_estimation": {
                "name": "OpenPose Body Model",
                "priority": 4,
                "size_gb": 0.2,
                "required": True,
                "sources": [
                    {
                        "type": "direct",
                        "url": "https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/models/pose/body_25/pose_iter_584000.caffemodel",
                        "filename": "body_pose_model.pth",
                        "local_dir": "checkpoints/openpose/ckpts"
                    },
                    {
                        "type": "huggingface", 
                        "repo_id": "yolox/yolox",
                        "filename": "yolox_nano.pth",
                        "local_dir": "checkpoints/pose_estimation"
                    }
                ],
                "description": "포즈 추정 모델"
            },
            
            "sam_segmentation": {
                "name": "Segment Anything Model",
                "priority": 5,
                "size_gb": 2.4,
                "required": False,
                "sources": [
                    {
                        "type": "direct",
                        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                        "filename": "sam_vit_h_4b8939.pth",
                        "local_dir": "checkpoints/sam_vit_h"
                    }
                ],
                "description": "고정밀 세그멘테이션 모델"
            },
            
            # 🔥 보조 모델들
            "clip_vit_base": {
                "name": "CLIP ViT-B/32",
                "priority": 6,
                "size_gb": 0.6,
                "required": False,
                "sources": [
                    {
                        "type": "huggingface",
                        "repo_id": "openai/clip-vit-base-patch32",
                        "subfolder": None,
                        "local_dir": "checkpoints/clip-vit-base-patch32"
                    }
                ],
                "description": "텍스트-이미지 이해 모델"
            },
            
            "controlnet_openpose": {
                "name": "ControlNet OpenPose",
                "priority": 7,
                "size_gb": 1.4,
                "required": False,
                "sources": [
                    {
                        "type": "huggingface",
                        "repo_id": "lllyasviel/control_v11p_sd15_openpose",
                        "subfolder": None,
                        "local_dir": "checkpoints/controlnet_openpose"
                    }
                ],
                "description": "포즈 제어 모델"
            },
            
            "cloth_segmentation": {
                "name": "Clothing Segmentation",
                "priority": 8,
                "size_gb": 0.1,
                "required": False,
                "sources": [
                    {
                        "type": "huggingface",
                        "repo_id": "rajistics/u2net_cloth_seg",
                        "subfolder": None,
                        "local_dir": "checkpoints/cloth_segmentation"
                    }
                ],
                "description": "의류 분할 모델"
            }
        }
        
        self.total_essential_size = sum(
            model["size_gb"] for model in self.models_catalog.values() 
            if model["required"]
        )
        
        self.total_all_size = sum(
            model["size_gb"] for model in self.models_catalog.values()
        )
    
    def check_dependencies(self) -> bool:
        """의존성 확인"""
        print("🔍 의존성 확인 중...")
        
        # Git LFS 확인
        if not check_git_lfs():
            print("⚠️ Git LFS가 설치되지 않음")
            print("   설치 방법: brew install git-lfs && git lfs install")
            return False
        
        # Hugging Face Hub 확인
        if not HF_AVAILABLE:
            print("⚠️ huggingface_hub가 설치되지 않음")
            print("   설치 방법: pip install huggingface_hub")
            return False
        
        # 충분한 디스크 공간 확인
        available_space = shutil.disk_usage(self.models_dir).free / (1024**3)
        print(f"💾 사용 가능한 디스크 공간: {available_space:.1f}GB")
        
        if available_space < 15:  # 15GB 최소 필요
            print("❌ 디스크 공간 부족 (최소 15GB 필요)")
            return False
        
        print("✅ 모든 의존성 확인 완료")
        return True
    
    def check_existing_models(self) -> Dict[str, bool]:
        """기존 모델 확인"""
        print("📂 기존 모델 확인 중...")
        
        existing = {}
        for model_key, model_info in self.models_catalog.items():
            model_dir = self.models_dir / model_info["sources"][0]["local_dir"]
            
            # 디렉토리가 존재하고 비어있지 않으면 모델이 있다고 가정
            if model_dir.exists() and any(model_dir.iterdir()):
                existing[model_key] = True
                print(f"   ✅ {model_info['name']}: 이미 존재")
            else:
                existing[model_key] = False
                print(f"   ❌ {model_info['name']}: 없음")
        
        return existing
    
    def download_from_huggingface(self, source: Dict, progress_callback=None) -> bool:
        """Hugging Face에서 모델 다운로드"""
        try:
            repo_id = source["repo_id"]
            local_dir = self.models_dir / source["local_dir"]
            local_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"📥 HuggingFace에서 다운로드: {repo_id}")
            
            if source.get("filename"):
                # 특정 파일만 다운로드
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=source["filename"],
                    cache_dir=str(local_dir),
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False
                )
                print(f"   ✅ 파일 다운로드 완료: {downloaded_path}")
            else:
                # 전체 레포지토리 다운로드
                snapshot_download(
                    repo_id=repo_id,
                    cache_dir=str(local_dir),
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                    ignore_patterns=["*.md", "*.txt", ".gitattributes"]
                )
                print(f"   ✅ 레포지토리 다운로드 완료: {local_dir}")
            
            return True
            
        except HfHubHTTPError as e:
            print(f"   ❌ HuggingFace 다운로드 실패: {e}")
            return False
        except Exception as e:
            print(f"   ❌ 다운로드 실패: {e}")
            return False
    
    def download_from_direct_url(self, source: Dict, progress_callback=None) -> bool:
        """직접 URL에서 다운로드"""
        try:
            url = source["url"]
            local_dir = self.models_dir / source["local_dir"]
            local_dir.mkdir(parents=True, exist_ok=True)
            
            filename = source.get("filename") or Path(urlparse(url).path).name
            local_path = local_dir / filename
            
            print(f"📥 직접 다운로드: {url}")
            print(f"   저장 위치: {local_path}")
            
            # 다운로드
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r   진행률: {progress:.1f}%", end="", flush=True)
            
            print(f"\n   ✅ 다운로드 완료: {local_path}")
            
            # ZIP 파일인 경우 압축 해제
            if local_path.suffix.lower() == '.zip':
                print(f"   📦 압축 해제 중...")
                with zipfile.ZipFile(local_path, 'r') as zip_ref:
                    zip_ref.extractall(local_dir)
                local_path.unlink()  # 압축 파일 삭제
                print(f"   ✅ 압축 해제 완료")
            
            return True
            
        except Exception as e:
            print(f"   ❌ 직접 다운로드 실패: {e}")
            return False
    
    def download_model(self, model_key: str) -> bool:
        """특정 모델 다운로드"""
        if model_key not in self.models_catalog:
            print(f"❌ 알 수 없는 모델: {model_key}")
            return False
        
        model_info = self.models_catalog[model_key]
        print(f"\n🤖 {model_info['name']} 다운로드 시작")
        print(f"   크기: {model_info['size_gb']}GB")
        print(f"   설명: {model_info['description']}")
        
        # 여러 소스 시도
        for i, source in enumerate(model_info["sources"]):
            print(f"\n   📍 소스 {i+1}/{len(model_info['sources'])} 시도...")
            
            success = False
            if source["type"] == "huggingface":
                success = self.download_from_huggingface(source)
            elif source["type"] == "direct":
                success = self.download_from_direct_url(source)
            
            if success:
                print(f"✅ {model_info['name']} 다운로드 성공!")
                return True
            else:
                print(f"⚠️ 소스 {i+1} 실패, 다음 소스 시도...")
        
        print(f"❌ {model_info['name']} 모든 소스에서 다운로드 실패")
        return False
    
    def download_essential_models(self) -> Dict[str, bool]:
        """필수 모델들 다운로드"""
        print(f"\n🔥 필수 모델 다운로드 시작")
        print(f"예상 총 크기: {self.total_essential_size:.1f}GB")
        print("=" * 50)
        
        results = {}
        essential_models = [
            key for key, info in self.models_catalog.items() 
            if info["required"]
        ]
        
        for i, model_key in enumerate(essential_models, 1):
            print(f"\n📋 진행률: {i}/{len(essential_models)}")
            results[model_key] = self.download_model(model_key)
            
            if results[model_key]:
                print(f"✅ {i}/{len(essential_models)} 완료")
            else:
                print(f"❌ {i}/{len(essential_models)} 실패")
        
        return results
    
    def download_all_models(self) -> Dict[str, bool]:
        """모든 모델 다운로드"""
        print(f"\n🚀 전체 모델 다운로드 시작")
        print(f"예상 총 크기: {self.total_all_size:.1f}GB")
        print("=" * 50)
        
        results = {}
        sorted_models = sorted(
            self.models_catalog.items(),
            key=lambda x: x[1]["priority"]
        )
        
        for i, (model_key, model_info) in enumerate(sorted_models, 1):
            print(f"\n📋 진행률: {i}/{len(sorted_models)}")
            results[model_key] = self.download_model(model_key)
            
            if results[model_key]:
                print(f"✅ {i}/{len(sorted_models)} 완료")
            else:
                print(f"❌ {i}/{len(sorted_models)} 실패")
        
        return results
    
    def generate_model_paths(self):
        """다운로드된 모델들의 경로 파일 생성"""
        print("\n📝 모델 경로 파일 생성 중...")
        
        paths_file = self.base_dir / "app" / "core" / "downloaded_model_paths.py"
        paths_file.parent.mkdir(parents=True, exist_ok=True)
        
        content = '''"""
자동 생성된 모델 경로 파일
다운로드된 AI 모델들의 경로 정의
"""

from pathlib import Path

# 기본 경로
MODELS_DIR = Path(__file__).parent.parent.parent / "ai_models"

# 모델 경로들
'''
        
        for model_key, model_info in self.models_catalog.items():
            local_dir = model_info["sources"][0]["local_dir"]
            var_name = model_key.upper() + "_PATH"
            content += f'{var_name} = MODELS_DIR / "{local_dir}"\n'
        
        content += '''
# 모델 경로 딕셔너리
MODEL_PATHS = {
'''
        
        for model_key in self.models_catalog.keys():
            var_name = model_key.upper() + "_PATH"
            content += f'    "{model_key}": {var_name},\n'
        
        content += '''}

def get_model_path(model_name: str) -> Path:
    """모델 경로 가져오기"""
    return MODEL_PATHS.get(model_name)

def is_model_available(model_name: str) -> bool:
    """모델 사용 가능 여부 확인"""
    path = get_model_path(model_name)
    return path and path.exists() and any(path.iterdir())
'''
        
        with open(paths_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 모델 경로 파일 생성: {paths_file}")

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="MyCloset AI 모델 다운로더")
    parser.add_argument("--all", action="store_true", help="모든 모델 다운로드")
    parser.add_argument("--essential", action="store_true", help="필수 모델만 다운로드")
    parser.add_argument("--model", type=str, help="특정 모델만 다운로드")
    parser.add_argument("--check", action="store_true", help="기존 모델 확인만")
    parser.add_argument("--path", type=str, default=".", help="프로젝트 루트 경로")
    
    args = parser.parse_args()
    
    # 프로젝트 경로 설정
    project_root = Path(args.path).resolve()
    backend_dir = project_root / "backend"
    
    if not backend_dir.exists():
        print(f"❌ 백엔드 디렉토리를 찾을 수 없습니다: {backend_dir}")
        sys.exit(1)
    
    print("🤖 MyCloset AI 모델 다운로더")
    print("=" * 50)
    print(f"📁 프로젝트 경로: {project_root}")
    print(f"📁 백엔드 경로: {backend_dir}")
    
    # 다운로더 초기화
    downloader = ModelDownloader(backend_dir)
    
    # 의존성 확인
    if not downloader.check_dependencies():
        print("❌ 의존성 확인 실패")
        sys.exit(1)
    
    # 기존 모델 확인
    existing_models = downloader.check_existing_models()
    
    if args.check:
        print("\n📊 모델 상태 요약:")
        for model_key, exists in existing_models.items():
            status = "✅ 존재" if exists else "❌ 없음"
            model_name = downloader.models_catalog[model_key]["name"]
            print(f"   {status} {model_name}")
        return
    
    # 다운로드 실행
    start_time = time.time()
    results = {}
    
    if args.model:
        # 특정 모델 다운로드
        if args.model not in downloader.models_catalog:
            print(f"❌ 알 수 없는 모델: {args.model}")
            print(f"사용 가능한 모델: {list(downloader.models_catalog.keys())}")
            sys.exit(1)
        
        results[args.model] = downloader.download_model(args.model)
        
    elif args.essential:
        # 필수 모델만 다운로드
        results = downloader.download_essential_models()
        
    elif args.all:
        # 모든 모델 다운로드
        results = downloader.download_all_models()
        
    else:
        # 기본: 필수 모델 다운로드
        print("📋 옵션이 지정되지 않아 필수 모델을 다운로드합니다.")
        print("전체 옵션: --all, --essential, --model <이름>")
        results = downloader.download_essential_models()
    
    # 결과 요약
    duration = time.time() - start_time
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    print(f"\n🎉 다운로드 완료!")
    print("=" * 50)
    print(f"⏱️  소요 시간: {duration:.1f}초")
    print(f"📊 성공률: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count > 0:
        # 경로 파일 생성
        downloader.generate_model_paths()
        
        print(f"\n✅ 성공한 모델:")
        for model_key, success in results.items():
            if success:
                model_name = downloader.models_catalog[model_key]["name"]
                print(f"   ✅ {model_name}")
    
    if success_count < total_count:
        print(f"\n❌ 실패한 모델:")
        for model_key, success in results.items():
            if not success:
                model_name = downloader.models_catalog[model_key]["name"]
                print(f"   ❌ {model_name}")
    
    print(f"\n💡 다음 단계:")
    print(f"   1. 백엔드 서버 재시작")
    print(f"   2. http://localhost:8000/api/models/status 확인")
    print(f"   3. 프론트엔드에서 가상 피팅 테스트")

if __name__ == "__main__":
    main()