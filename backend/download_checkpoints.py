#!/usr/bin/env python3
"""
Conda 환경용 실제 AI 모델 체크포인트 다운로더 (최종 버전)
폴백 모델 없음 - 검증된 실제 모델들만 다운로드

사용법:
    cd backend
    conda activate mycloset-ai
    python download_real_models_conda.py
"""

import os
import sys
import logging
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from tqdm import tqdm
import json

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class CondaRealModelDownloader:
    """Conda 환경용 실제 AI 모델 다운로더 (폴백 없음)"""
    
    def __init__(self):
        # 프로젝트 루트 경로 자동 감지
        current_path = Path(__file__).parent
        if current_path.name == "scripts":
            self.project_root = current_path.parent
        else:
            self.project_root = current_path
            
        # AI 모델 저장 경로를 기존 MyCloset AI 구조에 맞춤
        self.models_dir = self.project_root / "ai_models"
        self.checkpoints_dir = self.models_dir / "checkpoints"
        
        # 디렉토리 생성
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # 실제 검증된 모델들만 (100% 동작 보장)
        self.real_models = {
            "ootdiffusion": {
                "name": "OOTDiffusion",
                "description": "최신 실제 가상 피팅 모델",
                "method": "huggingface_git",
                "repo_id": "levihsu/OOTDiffusion",
                "local_dir": "ootdiffusion_hf",
                "size_gb": 8.5,
                "priority": 1,
                "verified": True,
                "essential_files": ["checkpoints", "configs"]
            },
            "human_parsing_atr": {
                "name": "ATR Human Parsing",
                "description": "실제 인체 분할 모델 (ATR 데이터셋)",
                "method": "direct_download",
                "urls": [
                    "https://github.com/PeikeLi/Self-Correction-Human-Parsing/releases/download/checkpoints/exp-schp-201908301523-atr.pth"
                ],
                "backup_urls": [
                    "https://huggingface.co/mattmdjaga/human_parsing/resolve/main/exp-schp-201908301523-atr.pth",
                    "https://drive.google.com/uc?id=1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH&confirm=t"
                ],
                "local_dir": "human_parsing",
                "filename": "exp-schp-201908301523-atr.pth",
                "size_gb": 0.178,
                "priority": 2,
                "verified": True
            },
            "u2net_portrait": {
                "name": "U2Net Portrait Segmentation",
                "description": "실제 배경 제거 모델",
                "method": "direct_download",
                "urls": [
                    "https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2net_portrait.pth"
                ],
                "backup_urls": [
                    "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ&confirm=t"
                ],
                "local_dir": "u2net",
                "filename": "u2net_portrait.pth",
                "size_gb": 0.176,
                "priority": 3,
                "verified": True
            },
            "mediapipe_pose": {
                "name": "MediaPipe Pose Landmarker",
                "description": "Google 공식 포즈 추정 모델",
                "method": "direct_download",
                "urls": [
                    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
                ],
                "local_dir": "mediapipe",
                "filename": "pose_landmarker_heavy.task",
                "size_gb": 0.029,
                "priority": 4,
                "verified": True
            },
            "segment_anything": {
                "name": "Segment Anything Model (SAM)",
                "description": "Meta 공식 세그멘테이션 모델",
                "method": "direct_download",
                "urls": [
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
                ],
                "local_dir": "sam",
                "filename": "sam_vit_h_4b8939.pth",
                "size_gb": 2.56,
                "priority": 5,
                "verified": True
            },
            "stable_diffusion_inpaint": {
                "name": "Stable Diffusion Inpainting",
                "description": "실제 이미지 인페인팅 모델",
                "method": "huggingface_download",
                "repo_id": "runwayml/stable-diffusion-inpainting",
                "local_dir": "stable_diffusion_inpaint",
                "size_gb": 5.21,
                "priority": 6,
                "verified": True,
                "essential_files": ["unet", "vae", "text_encoder", "safety_checker"]
            }
        }
    
    def check_conda_environment(self) -> bool:
        """Conda 환경 확인"""
        logger.info("🐍 Conda 환경 확인 중...")
        
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if not conda_env:
            logger.error("❌ Conda 환경이 활성화되지 않았습니다.")
            logger.info("💡 실행: conda activate mycloset-ai")
            return False
        
        logger.info(f"✅ 현재 Conda 환경: {conda_env}")
        
        # Python 경로 확인
        python_path = sys.executable
        if "conda" in python_path.lower() or "miniforge" in python_path.lower():
            logger.info(f"✅ Python 경로: {python_path}")
        else:
            logger.warning(f"⚠️ Python 경로가 Conda 환경이 아닐 수 있습니다: {python_path}")
        
        return True
    
    def install_conda_dependencies(self) -> bool:
        """Conda 환경에서 필요한 패키지 설치"""
        logger.info("📦 Conda 환경에서 필요한 패키지 설치 중...")
        
        # Conda로 설치할 패키지들
        conda_packages = [
            ("git", "conda-forge"),
            ("git-lfs", "conda-forge"),
            ("curl", "conda-forge"),
            ("wget", "conda-forge")
        ]
        
        # Pip로 설치할 패키지들
        pip_packages = [
            "huggingface_hub",
            "gdown>=4.7.1",
            "requests>=2.28.0",
            "tqdm>=4.64.0"
        ]
        
        # Conda 패키지 설치
        for package, channel in conda_packages:
            try:
                subprocess.run([
                    "conda", "install", "-c", channel, package, "-y", "--quiet"
                ], check=True, capture_output=True, text=True)
                logger.info(f"✅ {package}: Conda로 설치 완료")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️ {package}: Conda 설치 실패, 시스템 버전 사용")
        
        # Pip 패키지 설치
        for package in pip_packages:
            try:
                # 이미 설치된지 확인
                pkg_name = package.split(">=")[0].split("==")[0].replace("-", "_")
                __import__(pkg_name)
                logger.info(f"✅ {package}: 이미 설치됨")
            except ImportError:
                try:
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", package, "--quiet"
                    ], check=True, capture_output=True)
                    logger.info(f"✅ {package}: pip로 설치 완료")
                except subprocess.CalledProcessError as e:
                    logger.error(f"❌ {package}: 설치 실패 - {e}")
                    return False
        
        # Git LFS 초기화
        try:
            subprocess.run(["git", "lfs", "install"], check=True, capture_output=True)
            logger.info("✅ Git LFS 초기화 완료")
        except subprocess.CalledProcessError:
            logger.warning("⚠️ Git LFS 초기화 실패 (선택사항)")
        
        return True
    
    def download_with_progress(self, url: str, filepath: Path, retries: int = 3) -> bool:
        """진행률과 재시도 기능이 있는 다운로드"""
        for attempt in range(retries):
            try:
                # 헤더 설정 (일부 서버에서 User-Agent 필요)
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
                
                response = requests.get(url, stream=True, headers=headers, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                # 임시 파일에 다운로드
                temp_filepath = filepath.with_suffix(filepath.suffix + '.tmp')
                
                with open(temp_filepath, 'wb') as f, tqdm(
                    desc=f"📥 {filepath.name}",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                
                # 다운로드 완료 시 임시 파일을 최종 파일로 이동
                temp_filepath.rename(filepath)
                return True
                
            except Exception as e:
                logger.warning(f"⚠️ 다운로드 시도 {attempt + 1}/{retries} 실패: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # 지수 백오프
                    continue
                else:
                    logger.error(f"❌ 모든 다운로드 시도 실패: {url}")
                    return False
        
        return False
    
    def download_huggingface_model(self, model_info: Dict) -> bool:
        """Hugging Face에서 모델 다운로드"""
        try:
            from huggingface_hub import snapshot_download
            
            local_path = self.checkpoints_dir / model_info["local_dir"]
            repo_id = model_info["repo_id"]
            
            # 이미 존재하고 파일이 있으면 스킵
            if local_path.exists() and any(local_path.iterdir()):
                logger.info(f"✅ {model_info['name']} 이미 존재함")
                return True
            
            logger.info(f"📥 {model_info['name']} 다운로드 시작...")
            logger.info(f"   저장소: {repo_id}")
            logger.info(f"   크기: ~{model_info['size_gb']:.1f}GB")
            
            # 다운로드 시작 시간 기록
            start_time = time.time()
            
            # Hugging Face에서 다운로드
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_path),
                resume_download=True,
                local_dir_use_symlinks=False,
                # Git LFS 파일도 포함
                force_download=False,
                # 진행률 표시
                tqdm_class=tqdm
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"✅ {model_info['name']} 다운로드 완료! ({duration/60:.1f}분)")
            return True
            
        except Exception as e:
            logger.error(f"❌ {model_info['name']} Hugging Face 다운로드 실패: {e}")
            return False
    
    def download_huggingface_git(self, model_info: Dict) -> bool:
        """Git LFS로 Hugging Face 저장소 클론"""
        try:
            local_path = self.checkpoints_dir / model_info["local_dir"]
            repo_id = model_info["repo_id"]
            repo_url = f"https://huggingface.co/{repo_id}"
            
            # 이미 존재하고 파일이 있으면 스킵
            if local_path.exists() and any(local_path.iterdir()):
                logger.info(f"✅ {model_info['name']} 이미 존재함")
                return True
            
            logger.info(f"📥 {model_info['name']} Git 클론 시작...")
            logger.info(f"   저장소: {repo_url}")
            logger.info(f"   크기: ~{model_info['size_gb']:.1f}GB")
            
            start_time = time.time()
            
            # Git 클론 (shallow clone으로 속도 향상)
            subprocess.run([
                "git", "clone", "--depth=1", "--single-branch", repo_url, str(local_path)
            ], check=True, capture_output=True, text=True)
            
            # LFS 파일 다운로드
            subprocess.run([
                "git", "lfs", "pull"
            ], cwd=str(local_path), check=True, capture_output=True, text=True)
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"✅ {model_info['name']} Git 클론 완료! ({duration/60:.1f}분)")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {model_info['name']} Git 클론 실패: {e}")
            return False
    
    def download_direct_model(self, model_info: Dict) -> bool:
        """직접 URL에서 모델 다운로드 (여러 백업 URL 지원)"""
        local_path = self.checkpoints_dir / model_info["local_dir"]
        local_path.mkdir(parents=True, exist_ok=True)
        
        filepath = local_path / model_info["filename"]
        
        # 이미 존재하고 크기가 1MB 이상이면 스킵
        if filepath.exists() and filepath.stat().st_size > 1024 * 1024:
            logger.info(f"✅ {model_info['name']} 이미 존재함")
            return True
        
        logger.info(f"📥 {model_info['name']} 다운로드 시작...")
        logger.info(f"   크기: ~{model_info['size_gb']:.3f}GB")
        
        # 모든 URL 시도 (기본 URL + 백업 URL)
        all_urls = model_info["urls"]
        if "backup_urls" in model_info:
            all_urls.extend(model_info["backup_urls"])
        
        for i, url in enumerate(all_urls, 1):
            logger.info(f"🔗 시도 {i}/{len(all_urls)}: {url[:50]}...")
            
            # Google Drive 특별 처리
            if "google.com" in url or "drive.google.com" in url:
                try:
                    import gdown
                    success = gdown.download(url, str(filepath), quiet=False)
                    if success and filepath.exists() and filepath.stat().st_size > 1024:
                        logger.info(f"✅ {model_info['name']} Google Drive 다운로드 완료!")
                        return True
                except Exception as e:
                    logger.warning(f"⚠️ Google Drive 다운로드 실패: {e}")
                    continue
            else:
                # 일반 HTTP 다운로드
                if self.download_with_progress(url, filepath):
                    if filepath.exists() and filepath.stat().st_size > 1024:
                        logger.info(f"✅ {model_info['name']} 다운로드 완료!")
                        return True
                
                # 실패 시 임시 파일 삭제
                if filepath.exists():
                    filepath.unlink()
        
        logger.error(f"❌ {model_info['name']} 모든 URL 다운로드 실패")
        return False
    
    def verify_model(self, model_key: str, model_info: Dict) -> Tuple[bool, float]:
        """모델 다운로드 검증"""
        local_path = self.checkpoints_dir / model_info["local_dir"]
        
        if not local_path.exists():
            return False, 0.0
        
        # 총 파일 크기 계산
        total_size = 0
        file_count = 0
        
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        size_gb = total_size / (1024**3)
        
        # 검증 기준 (더 엄격하게)
        expected_size = model_info["size_gb"]
        min_size = max(expected_size * 0.8, 0.001)  # 최소 80% 또는 1MB
        
        # 필수 파일 확인 (있는 경우)
        essential_files_found = True
        if "essential_files" in model_info:
            for pattern in model_info["essential_files"]:
                if not list(local_path.glob(f"**/{pattern}*")):
                    essential_files_found = False
                    break
        
        if size_gb >= min_size and file_count > 0 and essential_files_found:
            return True, size_gb
        else:
            return False, size_gb
    
    def show_model_selection(self) -> List[str]:
        """모델 선택 메뉴 표시"""
        print("\n🤖 Conda 환경 실제 AI 모델 다운로드")
        print("=" * 60)
        print("✅ 폴백 없음 - 검증된 실제 모델만 다운로드")
        print("=" * 60)
        
        print("\n📋 검증된 실제 모델들:")
        for i, (key, info) in enumerate(self.real_models.items(), 1):
            local_path = self.checkpoints_dir / info["local_dir"]
            verified, actual_size = self.verify_model(key, info)
            
            if verified:
                status = f"✅ 다운로드됨 ({actual_size:.1f}GB)"
            else:
                status = "❌ 필요"
            
            print(f"{i}. {info['name']} ({info['size_gb']:.1f}GB) - {status}")
            print(f"   {info['description']}")
        
        total_size = sum(info["size_gb"] for info in self.real_models.values())
        print(f"\n📊 전체 크기: {total_size:.1f}GB")
        print(f"📁 저장 위치: {self.checkpoints_dir}")
        
        print("\n🎯 추천 선택:")
        print("  필수 (8.9GB): 1,2,3,4 (OOTDiffusion + Human Parsing + U2Net + MediaPipe)")
        print("  표준 (11.5GB): 1,2,3,4,5 (+ Segment Anything)")
        print("  완전 (16.7GB): all (모든 모델)")
        
        selection = input("\n다운로드할 모델 번호 (쉼표로 구분, 예: 1,2,3,4): ").strip()
        
        if not selection:
            return []
        
        if selection.lower() == 'all':
            return list(self.real_models.keys())
        
        try:
            indices = [int(x.strip()) for x in selection.split(',') if x.strip()]
            model_keys = []
            for i in indices:
                if 1 <= i <= len(self.real_models):
                    model_keys.append(list(self.real_models.keys())[i-1])
            return model_keys
        except (ValueError, IndexError):
            logger.error("❌ 잘못된 선택입니다.")
            return []
    
    def download_selected_models(self, model_keys: List[str]) -> Dict[str, bool]:
        """선택된 모델들 다운로드"""
        if not model_keys:
            logger.error("❌ 선택된 모델이 없습니다.")
            return {}
        
        results = {}
        total_size = sum(self.real_models[k]["size_gb"] for k in model_keys)
        
        print(f"\n📊 다운로드 계획:")
        print(f"   모델 수: {len(model_keys)}개")
        print(f"   총 크기: {total_size:.1f}GB")
        print(f"   예상 시간: {total_size * 1.5:.0f}분 (100Mbps 기준)")
        
        confirm = input("\n실제 모델들을 다운로드하시겠습니까? [y/N]: ").strip().lower()
        if confirm not in ['y', 'yes']:
            logger.info("❌ 다운로드 취소됨")
            return {}
        
        print("\n🚀 실제 AI 모델 다운로드 시작!")
        print("=" * 60)
        
        # 우선순위 순으로 정렬
        sorted_models = sorted(
            [(k, self.real_models[k]) for k in model_keys],
            key=lambda x: x[1]["priority"]
        )
        
        total_start_time = time.time()
        
        for i, (model_key, model_info) in enumerate(sorted_models, 1):
            print(f"\n[{i}/{len(sorted_models)}] {model_info['name']}")
            print(f"📋 {model_info['description']}")
            
            try:
                start_time = time.time()
                
                # 다운로드 방법에 따라 분기
                if model_info["method"] == "huggingface_download":
                    success = self.download_huggingface_model(model_info)
                elif model_info["method"] == "huggingface_git":
                    success = self.download_huggingface_git(model_info)
                elif model_info["method"] == "direct_download":
                    success = self.download_direct_model(model_info)
                else:
                    logger.error(f"❌ 지원하지 않는 다운로드 방법: {model_info['method']}")
                    success = False
                
                end_time = time.time()
                duration = end_time - start_time
                
                if success:
                    # 검증
                    verified, actual_size = self.verify_model(model_key, model_info)
                    if verified:
                        logger.info(f"🎉 {model_info['name']} 검증 완료! ({duration/60:.1f}분, {actual_size:.1f}GB)")
                        results[model_key] = True
                    else:
                        logger.error(f"❌ {model_info['name']} 검증 실패 ({actual_size:.1f}GB)")
                        results[model_key] = False
                else:
                    results[model_key] = False
                    
            except KeyboardInterrupt:
                logger.info("\n⏹ 사용자가 다운로드를 중단했습니다")
                break
            except Exception as e:
                logger.error(f"❌ {model_info['name']} 예상치 못한 오류: {e}")
                results[model_key] = False
        
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        
        print(f"\n⏱️ 총 다운로드 시간: {total_duration/60:.1f}분")
        
        return results
    
    def create_model_registry(self, results: Dict[str, bool]):
        """다운로드된 모델들의 레지스트리 생성"""
        registry = {
            "conda_environment": os.environ.get('CONDA_DEFAULT_ENV', 'unknown'),
            "download_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "project_root": str(self.project_root)
            },
            "models": {}
        }
        
        for model_key, success in results.items():
            model_info = self.real_models[model_key]
            verified, actual_size = self.verify_model(model_key, model_info)
            
            registry["models"][model_key] = {
                "name": model_info["name"],
                "description": model_info["description"],
                "local_path": str(self.checkpoints_dir / model_info["local_dir"]),
                "relative_path": f"ai_models/checkpoints/{model_info['local_dir']}",
                "download_success": success,
                "verified": verified,
                "actual_size_gb": actual_size,
                "expected_size_gb": model_info["size_gb"],
                "method": model_info["method"],
                "priority": model_info["priority"]
            }
        
        registry_path = self.models_dir / "conda_model_registry.json"
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 모델 레지스트리 생성: {registry_path}")
        
        # MyCloset AI 구조에 맞는 설정 파일도 생성
        config_content = f"""# MyCloset AI 실제 모델 설정 (Conda 환경)
# 자동 생성: {time.strftime('%Y-%m-%d %H:%M:%S')}

models:
"""
        
        for model_key, success in results.items():
            if success:
                model_info = self.real_models[model_key]
                config_content += f"""  {model_key}:
    name: "{model_info['name']}"
    path: "ai_models/checkpoints/{model_info['local_dir']}"
    enabled: true
    method: "{model_info['method']}"
    verified: true
    
"""
        
        config_path = self.models_dir / "conda_models_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        logger.info(f"✅ 모델 설정 파일 생성: {config_path}")

def main():
    """메인 함수"""
    print("🚀 Conda 환경 실제 AI 모델 다운로더 시작!")
    print("   ✅ 폴백 모델 없음")
    print("   ✅ 검증된 실제 모델만")
    print("   ✅ MyCloset AI 구조 준수")
    
    downloader = CondaRealModelDownloader()
    
    # 1. Conda 환경 확인
    if not downloader.check_conda_environment():
        return False
    
    # 2. 의존성 설치
    if not downloader.install_conda_dependencies():
        print("\n❌ 의존성 설치에 실패했습니다!")
        return False
    
    # 3. 모델 선택
    model_keys = downloader.show_model_selection()
    if not model_keys:
        print("❌ 선택된 모델이 없습니다.")
        return False
    
    # 4. 모델 다운로드
    results = downloader.download_selected_models(model_keys)
    
    # 5. 결과 요약
    print("\n" + "=" * 60)
    print("📊 실제 AI 모델 다운로드 결과:")
    
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    for model_key, success in results.items():
        model_name = downloader.real_models[model_key]["name"]
        status = "✅ 성공" if success else "❌ 실패"
        
        if success:
            verified, actual_size = downloader.verify_model(model_key, downloader.real_models[model_key])
            status += f" ({actual_size:.1f}GB)"
        
        print(f"  {model_name}: {status}")
    
    print(f"\n성공: {success_count}/{total_count}")
    
    if success_count > 0:
        # 6. 레지스트리 생성
        downloader.create_model_registry(results)
        
        print("\n🎉 실제 AI 모델 다운로드 완료!")
        print("\n📋 다음 단계:")
        print("1. 모델 확인: ls -la ai_models/checkpoints/")
        print("2. 설정 확인: cat ai_models/conda_model_registry.json")
        print("3. Step 테스트: python test_step_01_human_parsing.py")
        print("4. 서버 실행: python app/main.py")  
        print("5. API 테스트: http://localhost:8000/docs")
        
        print(f"\n📁 모델 저장 위치: {downloader.checkpoints_dir}")
        
        return True
    else:
        print("\n❌ 모든 모델 다운로드에 실패했습니다.")
        print("💡 해결 방법:")
        print("   1. 네트워크 연결 확인")
        print("   2. Conda 환경 재확인: conda activate mycloset-ai")
        print("   3. 의존성 재설치: pip install huggingface_hub gdown requests tqdm")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹ 사용자가 프로그램을 중단했습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)