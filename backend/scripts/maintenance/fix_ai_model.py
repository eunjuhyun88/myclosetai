#!/usr/bin/env python3
"""
🤖 MyCloset AI 실제 모델 다운로드 스크립트
✅ 시뮬레이션 모드 → 실제 AI 모델 모드 전환
✅ conda 환경 최적화
✅ M3 Max 128GB 메모리 활용
✅ 단계별 검증 및 폴백 지원
"""

import os
import sys
import json
import time
import requests
import hashlib
import subprocess
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, List, Optional, Tuple

class ModelDownloader:
    """AI 모델 다운로드 매니저"""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # 다운로드할 모델 목록 (실제 작동하는 모델들)
        self.models = {
            # Step 01: Human Parsing (중요도: 최고)
            "human_parsing": {
                "name": "인간 파싱 (Human Parsing)",
                "priority": 1,
                "models": [
                    {
                        "name": "Graphonomy",
                        "url": "https://github.com/Gaoyiminggithub/Graphonomy.git",
                        "type": "git",
                        "checkpoint_url": "https://drive.google.com/uc?id=1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH",
                        "checkpoint_name": "inference.pth",
                        "size_mb": 85
                    }
                ]
            },
            
            # Step 02: Pose Estimation (중요도: 높음)
            "pose_estimation": {
                "name": "포즈 추정 (Pose Estimation)", 
                "priority": 2,
                "models": [
                    {
                        "name": "OpenPose",
                        "url": "https://github.com/CMU-Perceptual-Computing-Lab/openpose.git",
                        "type": "git",
                        "checkpoint_url": "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel",
                        "checkpoint_name": "pose_iter_584000.caffemodel",
                        "size_mb": 200
                    }
                ]
            },
            
            # Step 03: Cloth Segmentation (중요도: 높음)
            "cloth_segmentation": {
                "name": "의류 세그멘테이션 (Cloth Segmentation)",
                "priority": 3,
                "models": [
                    {
                        "name": "U2-Net",
                        "url": "https://github.com/xuebinqin/U-2-Net.git",
                        "type": "git", 
                        "checkpoint_url": "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
                        "checkpoint_name": "u2net.pth",
                        "size_mb": 176
                    }
                ]
            },
            
            # Step 06: Virtual Fitting (중요도: 최고)
            "virtual_fitting": {
                "name": "가상 피팅 (Virtual Fitting)",
                "priority": 1,
                "models": [
                    {
                        "name": "OOTDiffusion",
                        "url": "https://github.com/levihsu/OOTDiffusion.git", 
                        "type": "git",
                        "checkpoint_url": "https://huggingface.co/levihsu/OOTDiffusion",
                        "checkpoint_name": "ootd",
                        "size_mb": 2000,
                        "huggingface": True
                    }
                ]
            },
            
            # 기본 모델들 (작은 크기)
            "basic_models": {
                "name": "기본 모델들",
                "priority": 4,
                "models": [
                    {
                        "name": "CLIP",
                        "url": "openai/clip-vit-base-patch32",
                        "type": "huggingface",
                        "size_mb": 600
                    }
                ]
            }
        }
    
    def check_system_requirements(self) -> bool:
        """시스템 요구사항 확인"""
        print("🔍 시스템 요구사항 확인 중...")
        
        # Python 환경 확인
        python_version = sys.version_info
        if python_version < (3, 8):
            print(f"❌ Python 3.8+ 필요 (현재: {python_version.major}.{python_version.minor})")
            return False
        
        # conda 환경 확인
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
        print(f"🐍 Conda 환경: {conda_env}")
        
        # Git 확인
        try:
            result = subprocess.run(['git', '--version'], capture_output=True, text=True)
            print(f"✅ Git: {result.stdout.strip()}")
        except FileNotFoundError:
            print("❌ Git이 설치되어 있지 않습니다")
            return False
        
        # 디스크 공간 확인 (최소 10GB)
        free_space = self.get_free_space_gb()
        if free_space < 10:
            print(f"❌ 디스크 공간 부족: {free_space:.1f}GB (최소 10GB 필요)")
            return False
        
        print(f"✅ 사용 가능한 디스크 공간: {free_space:.1f}GB")
        return True
    
    def get_free_space_gb(self) -> float:
        """사용 가능한 디스크 공간 (GB) 반환"""
        try:
            import shutil
            free_bytes = shutil.disk_usage(self.models_dir).free
            return free_bytes / (1024 ** 3)
        except:
            return 100.0  # 기본값
    
    def download_with_progress(self, url: str, file_path: Path, chunk_size: int = 8192) -> bool:
        """진행률 표시와 함께 파일 다운로드"""
        try:
            print(f"📥 다운로드 중: {url}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r진행률: {percent:.1f}% ({downloaded//1024//1024}MB/{total_size//1024//1024}MB)", end='')
            
            print(f"\n✅ 다운로드 완료: {file_path}")
            return True
            
        except Exception as e:
            print(f"\n❌ 다운로드 실패: {e}")
            if file_path.exists():
                file_path.unlink()  # 실패한 파일 삭제
            return False
    
    def clone_git_repo(self, url: str, target_dir: Path, depth: int = 1) -> bool:
        """Git 저장소 클론"""
        try:
            print(f"📂 Git 클론 중: {url}")
            
            cmd = ['git', 'clone']
            if depth > 0:
                cmd.extend(['--depth', str(depth)])
            cmd.extend([url, str(target_dir)])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Git 클론 완료: {target_dir}")
                return True
            else:
                print(f"❌ Git 클론 실패: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Git 클론 오류: {e}")
            return False
    
    def download_from_huggingface(self, model_name: str, target_dir: Path) -> bool:
        """Hugging Face에서 모델 다운로드"""
        try:
            print(f"🤗 Hugging Face 모델 다운로드: {model_name}")
            
            # transformers 라이브러리 사용
            try:
                from transformers import AutoModel, AutoTokenizer
                
                # 모델과 토크나이저 다운로드
                model = AutoModel.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # 로컬에 저장
                target_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(target_dir)
                tokenizer.save_pretrained(target_dir)
                
                print(f"✅ Hugging Face 모델 저장 완료: {target_dir}")
                return True
                
            except ImportError:
                print("⚠️ transformers 라이브러리가 없습니다. pip install transformers 실행")
                return self.install_and_retry_hf(model_name, target_dir)
            
        except Exception as e:
            print(f"❌ Hugging Face 다운로드 실패: {e}")
            return False
    
    def install_and_retry_hf(self, model_name: str, target_dir: Path) -> bool:
        """transformers 설치 후 재시도"""
        try:
            print("📦 transformers 라이브러리 설치 중...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'transformers'], check=True)
            
            # 재시도
            return self.download_from_huggingface(model_name, target_dir)
            
        except Exception as e:
            print(f"❌ transformers 설치 실패: {e}")
            return False
    
    def download_model_category(self, category_name: str, category_info: Dict) -> bool:
        """모델 카테고리 다운로드"""
        print(f"\n🎯 카테고리 다운로드: {category_info['name']}")
        
        success_count = 0
        total_count = len(category_info['models'])
        
        for model_info in category_info['models']:
            model_name = model_info['name']
            model_type = model_info['type']
            model_url = model_info['url']
            
            # 타겟 디렉토리 설정
            target_dir = self.models_dir / category_name / model_name
            
            # 이미 존재하는지 확인
            if target_dir.exists() and any(target_dir.iterdir()):
                print(f"✅ {model_name} 이미 존재함: {target_dir}")
                success_count += 1
                continue
            
            print(f"\n📦 {model_name} 다운로드 시작...")
            
            # 타입별 다운로드
            success = False
            if model_type == 'git':
                success = self.clone_git_repo(model_url, target_dir)
            elif model_type == 'huggingface':
                success = self.download_from_huggingface(model_url, target_dir)
            
            # 체크포인트 다운로드 (있는 경우)
            if success and 'checkpoint_url' in model_info:
                checkpoint_url = model_info['checkpoint_url']
                checkpoint_name = model_info['checkpoint_name']
                checkpoint_path = target_dir / checkpoint_name
                
                if not checkpoint_path.exists():
                    if checkpoint_url.startswith('https://drive.google.com'):
                        print(f"⚠️ Google Drive 링크 감지: 수동 다운로드 필요")
                        print(f"   URL: {checkpoint_url}")
                        print(f"   저장 위치: {checkpoint_path}")
                    else:
                        success = self.download_with_progress(checkpoint_url, checkpoint_path)
            
            if success:
                success_count += 1
                print(f"✅ {model_name} 다운로드 완료")
            else:
                print(f"❌ {model_name} 다운로드 실패")
        
        print(f"\n📊 {category_info['name']} 결과: {success_count}/{total_count} 성공")
        return success_count > 0
    
    def create_model_configs(self):
        """모델 설정 파일 생성"""
        print("\n📝 모델 설정 파일 생성 중...")
        
        config_file = self.models_dir / "model_configs.json"
        
        configs = {}
        for category_name, category_info in self.models.items():
            for model_info in category_info['models']:
                model_name = model_info['name']
                target_dir = self.models_dir / category_name / model_name
                
                if target_dir.exists():
                    configs[model_name.lower()] = {
                        "name": model_name,
                        "category": category_name,
                        "path": str(target_dir),
                        "type": model_info['type'],
                        "size_mb": model_info.get('size_mb', 0),
                        "available": True
                    }
        
        with open(config_file, 'w') as f:
            json.dump(configs, f, indent=2)
        
        print(f"✅ 설정 파일 생성: {config_file}")
        print(f"📊 사용 가능한 모델: {len(configs)}개")
    
    def run_download(self, priority_only: bool = False) -> bool:
        """전체 다운로드 실행"""
        print("🚀 MyCloset AI 모델 다운로드 시작!")
        print("=" * 50)
        
        # 시스템 요구사항 확인
        if not self.check_system_requirements():
            return False
        
        # 우선순위별 정렬
        sorted_categories = sorted(
            self.models.items(),
            key=lambda x: x[1]['priority']
        )
        
        if priority_only:
            # 우선순위 1,2만 다운로드
            sorted_categories = [(k, v) for k, v in sorted_categories if v['priority'] <= 2]
            print("🎯 우선순위 모델만 다운로드합니다")
        
        success_categories = 0
        total_categories = len(sorted_categories)
        
        for category_name, category_info in sorted_categories:
            if self.download_model_category(category_name, category_info):
                success_categories += 1
        
        # 설정 파일 생성
        self.create_model_configs()
        
        print("\n" + "=" * 50)
        print(f"🎉 다운로드 완료! {success_categories}/{total_categories} 카테고리 성공")
        
        if success_categories > 0:
            print("\n📝 다음 단계:")
            print("1. python app/main.py  # 서버 재시작")
            print("2. 브라우저에서 http://localhost:8000/docs 확인")
            print("3. AI 모델이 실제로 로드되는지 테스트")
            
            print("\n💡 팁:")
            print("- Google Drive 링크는 수동 다운로드가 필요할 수 있습니다")
            print("- 대용량 모델은 시간이 오래 걸릴 수 있습니다")
            print("- 네트워크 오류 시 스크립트를 다시 실행하세요")
            
            return True
        else:
            print("\n❌ 모든 모델 다운로드에 실패했습니다")
            return False

def main():
    """메인 함수"""
    print("🤖 MyCloset AI 실제 모델 다운로드 도구")
    print("=" * 50)
    
    # backend 디렉토리 확인
    if Path.cwd().name != "backend":
        if Path("backend").exists():
            os.chdir("backend")
        else:
            print("❌ backend 디렉토리를 찾을 수 없습니다")
            sys.exit(1)
    
    # 모델 디렉토리 설정
    models_dir = Path("ai_models")
    
    # 다운로더 생성
    downloader = ModelDownloader(models_dir)
    
    # 사용자 선택
    print("\n다운로드 옵션을 선택하세요:")
    print("1. 우선순위 모델만 (빠른 시작) - 약 2GB")
    print("2. 모든 모델 (전체 기능) - 약 10GB")
    print("3. 설정 파일만 생성 (이미 모델 있음)")
    
    choice = input("\n선택 (1-3): ").strip()
    
    if choice == "1":
        success = downloader.run_download(priority_only=True)
    elif choice == "2": 
        success = downloader.run_download(priority_only=False)
    elif choice == "3":
        downloader.create_model_configs()
        success = True
    else:
        print("❌ 잘못된 선택입니다")
        sys.exit(1)
    
    if success:
        print("\n🎉 모델 다운로드가 완료되었습니다!")
        print("이제 실제 AI 모델을 사용할 수 있습니다! 🚀")
    else:
        print("\n❌ 모델 다운로드에 실패했습니다")
        sys.exit(1)

if __name__ == "__main__":
    main()