#!/usr/bin/env python3
"""
Step 5 의류 워핑 AI 모델 다운로더
✅ Conda 환경 최적화
✅ M3 Max 128GB 메모리 관리
✅ 자동 체크포인트 검증
✅ 병렬 다운로드 지원
✅ 진행률 표시
"""

import os
import sys
import asyncio
import aiohttp
import aiofiles
import json
import time
import hashlib
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from concurrent.futures import ThreadPoolExecutor
import subprocess
import platform

# 진행률 표시
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("💡 더 나은 진행률 표시를 위해 tqdm을 설치하세요: pip install tqdm")

# AI 라이브러리들
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch가 설치되지 않았습니다")

try:
    from huggingface_hub import snapshot_download, hf_hub_download, login
    from huggingface_hub.utils import HfHubHTTPError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️ Hugging Face Hub가 설치되지 않았습니다: pip install huggingface-hub")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('step_05_download.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """모델 설정"""
    name: str
    repo_id: str
    local_path: str
    size_mb: int
    required_files: List[str]
    optional_files: List[str] = None
    download_method: str = "huggingface"  # huggingface, direct, git
    url: str = None
    checksum: str = None
    
    def __post_init__(self):
        if self.optional_files is None:
            self.optional_files = []

class Step05AIDownloader:
    """Step 5 AI 모델 다운로더"""
    
    def __init__(self, base_dir: Optional[str] = None):
        """초기화"""
        # 기본 경로 설정
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            # 프로젝트 루트 찾기
            current = Path(__file__).resolve()
            for parent in current.parents:
                if (parent / "backend").exists():
                    self.base_dir = parent / "backend" / "ai_models" / "step_05_cloth_warping"
                    break
            else:
                self.base_dir = Path.cwd() / "ai_models" / "step_05_cloth_warping"
        
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 시스템 정보
        self.is_m3_max = self._detect_m3_max()
        self.max_workers = 8 if self.is_m3_max else 4
        self.chunk_size = 1024 * 1024  # 1MB chunks
        
        # 모델 정의
        self.models = self._define_models()
        
        logger.info(f"🚀 Step 5 AI 다운로더 초기화 완료")
        logger.info(f"📁 다운로드 경로: {self.base_dir}")
        logger.info(f"🍎 M3 Max 최적화: {self.is_m3_max}")
        logger.info(f"⚡ 최대 병렬 다운로드: {self.max_workers}")
    
    def _detect_m3_max(self) -> bool:
        """M3 Max 감지"""
        try:
            if platform.system() == "Darwin":  # macOS
                if TORCH_AVAILABLE and torch.backends.mps.is_available():
                    return True
        except Exception:
            pass
        return False
    
    def _define_models(self) -> Dict[str, ModelConfig]:
        """필요한 AI 모델들 정의"""
        return {
            # 1. IDM-VTON (핵심 의류 워핑 모델)
            "idm_vton": ModelConfig(
                name="IDM-VTON",
                repo_id="yisol/IDM-VTON",
                local_path=str(self.base_dir / "idm_vton"),
                size_mb=8500,
                required_files=["model.safetensors", "config.json"],
                optional_files=["tokenizer.json", "scheduler.json"]
            ),
            
            # 2. SAM for Segmentation
            "sam_vit_large": ModelConfig(
                name="SAM-ViT-Large",
                repo_id="facebook/sam-vit-large",
                local_path=str(self.base_dir / "sam"),
                size_mb=2400,
                required_files=["pytorch_model.bin"],
                download_method="direct",
                url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
            ),
            
            # 3. Stable Diffusion Inpainting
            "sd_inpainting": ModelConfig(
                name="Stable Diffusion Inpainting",
                repo_id="runwayml/stable-diffusion-inpainting",
                local_path=str(self.base_dir / "sd_inpainting"),
                size_mb=5100,
                required_files=["unet/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors"]
            ),
            
            # 4. OpenPose for Pose Estimation
            "openpose": ModelConfig(
                name="OpenPose",
                repo_id="lllyasviel/Annotators", 
                local_path=str(self.base_dir / "openpose"),
                size_mb=1200,
                required_files=["body_pose_model.pth"],
                optional_files=["hand_pose_model.pth", "face_pose_model.pth"]
            ),
            
            # 5. CLIP for Feature Extraction
            "clip_vit": ModelConfig(
                name="CLIP-ViT-Large",
                repo_id="openai/clip-vit-large-patch14",
                local_path=str(self.base_dir / "clip"),
                size_mb=1700,
                required_files=["pytorch_model.bin", "config.json"]
            ),
            
            # 6. DensePose (의류 매핑용)
            "densepose": ModelConfig(
                name="DensePose",
                repo_id="facebook/densepose",
                local_path=str(self.base_dir / "densepose"),
                size_mb=800,
                required_files=["model.pkl"],
                download_method="direct",
                url="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x.pkl"
            ),
            
            # 7. Thin-Plate Spline (기하학적 변형용)
            "tps_model": ModelConfig(
                name="TPS Transformation",
                repo_id="microsoft/DiT-XL-2-256",
                local_path=str(self.base_dir / "tps"),
                size_mb=3200,
                required_files=["diffusion_pytorch_model.safetensors"]
            ),
            
            # 8. Texture Synthesis Model
            "texture_synthesis": ModelConfig(
                name="Texture Synthesis",
                repo_id="stabilityai/stable-diffusion-2-inpainting",
                local_path=str(self.base_dir / "texture"),
                size_mb=4600,
                required_files=["unet/diffusion_pytorch_model.safetensors"]
            )
        }
    
    async def download_all_models(self, force_redownload: bool = False) -> Dict[str, bool]:
        """모든 모델 다운로드"""
        print("🚀 Step 5 AI 모델 다운로드 시작")
        print("=" * 60)
        
        # 디스크 공간 확인
        total_size_mb = sum(model.size_mb for model in self.models.values())
        if not self._check_disk_space(total_size_mb):
            logger.error(f"❌ 디스크 공간 부족! 필요: {total_size_mb/1024:.1f}GB")
            return {}
        
        print(f"📊 총 다운로드 크기: {total_size_mb/1024:.1f}GB")
        print(f"📂 다운로드 위치: {self.base_dir}")
        print(f"⚡ 병렬 다운로드 수: {self.max_workers}")
        print()
        
        # Hugging Face 로그인 확인 (선택적)
        await self._check_hf_login()
        
        # 병렬 다운로드 실행
        results = {}
        
        if TQDM_AVAILABLE:
            progress_bar = tqdm(
                total=len(self.models),
                desc="📥 모델 다운로드",
                unit="model",
                ncols=80
            )
        
        # 세마포어로 동시 다운로드 수 제한
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def download_with_semaphore(model_name: str, model_config: ModelConfig):
            async with semaphore:
                success = await self._download_single_model(model_name, model_config, force_redownload)
                if TQDM_AVAILABLE:
                    progress_bar.update(1)
                return model_name, success
        
        # 모든 다운로드 태스크 생성
        tasks = [
            download_with_semaphore(name, config)
            for name, config in self.models.items()
        ]
        
        # 병렬 실행
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        if TQDM_AVAILABLE:
            progress_bar.close()
        
        # 결과 수집
        for result in completed_tasks:
            if isinstance(result, Exception):
                logger.error(f"❌ 다운로드 중 오류: {result}")
            else:
                model_name, success = result
                results[model_name] = success
        
        # 결과 요약
        success_count = sum(results.values())
        total_count = len(results)
        
        print("\n" + "=" * 60)
        print(f"🎉 다운로드 완료: {success_count}/{total_count} 성공")
        
        if success_count == total_count:
            print("✅ 모든 모델이 성공적으로 다운로드되었습니다!")
        else:
            failed_models = [name for name, success in results.items() if not success]
            print(f"⚠️ 실패한 모델들: {', '.join(failed_models)}")
        
        # 검증 실행
        print("\n🔍 모델 검증 시작...")
        verification_results = await self._verify_all_models()
        
        verified_count = sum(verification_results.values())
        print(f"✅ 검증 완료: {verified_count}/{total_count} 통과")
        
        # 요약 보고서 생성
        await self._generate_summary_report(results, verification_results)
        
        return results
    
    async def _download_single_model(
        self,
        model_name: str,
        model_config: ModelConfig,
        force_redownload: bool
    ) -> bool:
        """단일 모델 다운로드"""
        try:
            model_path = Path(model_config.local_path)
            
            # 기존 파일 확인
            if not force_redownload and self._model_exists(model_config):
                logger.info(f"✅ {model_config.name} - 이미 존재함")
                return True
            
            # 디렉토리 생성
            model_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"📥 {model_config.name} 다운로드 시작...")
            
            # 다운로드 방법에 따라 분기
            if model_config.download_method == "huggingface":
                success = await self._download_from_huggingface(model_config)
            elif model_config.download_method == "direct":
                success = await self._download_direct(model_config)
            elif model_config.download_method == "git":
                success = await self._download_from_git(model_config)
            else:
                logger.error(f"❌ 알 수 없는 다운로드 방법: {model_config.download_method}")
                return False
            
            if success:
                # 체크포인트 검증
                if self._verify_model(model_config):
                    logger.info(f"✅ {model_config.name} 다운로드 및 검증 완료")
                    return True
                else:
                    logger.warning(f"⚠️ {model_config.name} 다운로드 완료되었지만 검증 실패")
                    return False
            else:
                logger.error(f"❌ {model_config.name} 다운로드 실패")
                return False
                
        except Exception as e:
            logger.error(f"❌ {model_config.name} 다운로드 중 오류: {e}")
            return False
    
    async def _download_from_huggingface(self, model_config: ModelConfig) -> bool:
        """Hugging Face Hub에서 다운로드"""
        try:
            if not HF_AVAILABLE:
                logger.error("❌ Hugging Face Hub가 설치되지 않음")
                return False
            
            # 특정 파일들만 다운로드 (용량 절약)
            if model_config.required_files:
                for file_pattern in model_config.required_files:
                    try:
                        file_path = hf_hub_download(
                            repo_id=model_config.repo_id,
                            filename=file_pattern,
                            cache_dir=model_config.local_path,
                            local_dir=model_config.local_path,
                            resume_download=True
                        )
                        logger.info(f"  ✅ {file_pattern} 다운로드 완료")
                    except Exception as e:
                        logger.warning(f"  ⚠️ {file_pattern} 다운로드 실패: {e}")
            else:
                # 전체 리포지토리 다운로드
                snapshot_download(
                    repo_id=model_config.repo_id,
                    cache_dir=model_config.local_path,
                    local_dir=model_config.local_path,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
            
            return True
            
        except HfHubHTTPError as e:
            if "401" in str(e):
                logger.error(f"❌ {model_config.name}: 인증 필요 (Hugging Face 로그인)")
            elif "404" in str(e):
                logger.error(f"❌ {model_config.name}: 모델을 찾을 수 없음")
            else:
                logger.error(f"❌ {model_config.name}: HTTP 오류 {e}")
            return False
        except Exception as e:
            logger.error(f"❌ {model_config.name} HF 다운로드 실패: {e}")
            return False
    
    async def _download_direct(self, model_config: ModelConfig) -> bool:
        """직접 URL에서 다운로드"""
        try:
            if not model_config.url:
                logger.error(f"❌ {model_config.name}: 다운로드 URL이 없음")
                return False
            
            filename = Path(model_config.url).name
            file_path = Path(model_config.local_path) / filename
            
            # 이미 존재하면 스킵
            if file_path.exists() and file_path.stat().st_size > 1024:  # 1KB 이상
                logger.info(f"  ✅ {filename} 이미 존재함")
                return True
            
            async with aiohttp.ClientSession() as session:
                logger.info(f"  📥 {filename} 다운로드 중...")
                
                async with session.get(model_config.url) as response:
                    if response.status != 200:
                        logger.error(f"❌ HTTP {response.status}: {model_config.url}")
                        return False
                    
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    
                    async with aiofiles.open(file_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(self.chunk_size):
                            await f.write(chunk)
                            downloaded += len(chunk)
                            
                            # 진행률 로깅 (10MB마다)
                            if downloaded % (10 * 1024 * 1024) == 0:
                                if total_size > 0:
                                    progress = (downloaded / total_size) * 100
                                    logger.info(f"    📊 {filename}: {progress:.1f}% ({downloaded//1024//1024}MB)")
            
            logger.info(f"  ✅ {filename} 다운로드 완료 ({file_path.stat().st_size//1024//1024}MB)")
            return True
            
        except Exception as e:
            logger.error(f"❌ {model_config.name} 직접 다운로드 실패: {e}")
            return False
    
    async def _download_from_git(self, model_config: ModelConfig) -> bool:
        """Git LFS로 다운로드"""
        try:
            model_path = Path(model_config.local_path)
            
            if model_path.exists() and any(model_path.iterdir()):
                logger.info(f"  ✅ {model_config.name} Git 리포지토리 이미 존재")
                return True
            
            # Git clone with LFS
            cmd = [
                "git", "clone",
                f"https://huggingface.co/{model_config.repo_id}",
                str(model_path)
            ]
            
            logger.info(f"  📥 Git clone: {model_config.repo_id}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"  ✅ {model_config.name} Git clone 완료")
                return True
            else:
                logger.error(f"❌ Git clone 실패: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"❌ {model_config.name} Git 다운로드 실패: {e}")
            return False
    
    def _model_exists(self, model_config: ModelConfig) -> bool:
        """모델 존재 확인"""
        model_path = Path(model_config.local_path)
        
        if not model_path.exists():
            return False
        
        # 필수 파일들 확인
        for required_file in model_config.required_files:
            file_patterns = list(model_path.rglob(required_file))
            if not file_patterns:
                return False
        
        return True
    
    def _verify_model(self, model_config: ModelConfig) -> bool:
        """모델 검증"""
        try:
            model_path = Path(model_config.local_path)
            
            # 1. 디렉토리 존재 확인
            if not model_path.exists():
                return False
            
            # 2. 필수 파일 존재 확인
            for required_file in model_config.required_files:
                file_patterns = list(model_path.rglob(required_file))
                if not file_patterns:
                    logger.warning(f"⚠️ 필수 파일 누락: {required_file}")
                    return False
            
            # 3. 파일 크기 확인
            total_size = sum(
                f.stat().st_size for f in model_path.rglob("*") 
                if f.is_file()
            )
            
            expected_size = model_config.size_mb * 1024 * 1024
            size_ratio = total_size / expected_size
            
            if size_ratio < 0.5:  # 50% 미만이면 문제
                logger.warning(f"⚠️ 크기 부족: {total_size//1024//1024}MB < 예상 {model_config.size_mb}MB")
                return False
            
            # 4. PyTorch 모델 파일 검증 (선택적)
            if TORCH_AVAILABLE:
                model_files = list(model_path.rglob("*.bin")) + list(model_path.rglob("*.pth"))
                for model_file in model_files[:2]:  # 처음 2개만 검사
                    try:
                        torch.load(model_file, map_location='cpu')
                    except Exception as e:
                        logger.warning(f"⚠️ 모델 파일 손상 가능성: {model_file.name} - {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 모델 검증 오류: {e}")
            return False
    
    async def _verify_all_models(self) -> Dict[str, bool]:
        """모든 모델 검증"""
        results = {}
        
        for model_name, model_config in self.models.items():
            verified = self._verify_model(model_config)
            results[model_name] = verified
            
            status = "✅ 검증됨" if verified else "❌ 검증 실패"
            logger.info(f"{status} {model_config.name}")
        
        return results
    
    def _check_disk_space(self, required_mb: int) -> bool:
        """디스크 공간 확인"""
        try:
            import shutil
            free_space = shutil.disk_usage(self.base_dir).free
            free_space_mb = free_space // (1024 * 1024)
            
            # 여유 공간 50% 추가 요구
            required_with_buffer = required_mb * 1.5
            
            logger.info(f"💾 디스크 공간: {free_space_mb}MB 여유, {required_with_buffer:.0f}MB 필요")
            
            return free_space_mb >= required_with_buffer
            
        except Exception as e:
            logger.warning(f"⚠️ 디스크 공간 확인 실패: {e}")
            return True
    
    async def _check_hf_login(self):
        """Hugging Face 로그인 확인"""
        try:
            if HF_AVAILABLE:
                from huggingface_hub import whoami
                try:
                    user_info = whoami()
                    logger.info(f"🔐 Hugging Face 로그인: {user_info.get('name', 'Unknown')}")
                except Exception:
                    logger.warning("⚠️ Hugging Face 로그인 안됨 (일부 모델 접근 제한 가능)")
        except Exception:
            pass
    
    async def _generate_summary_report(
        self,
        download_results: Dict[str, bool],
        verification_results: Dict[str, bool]
    ):
        """요약 보고서 생성"""
        report = {
            "step": "05_cloth_warping",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "platform": platform.system(),
                "is_m3_max": self.is_m3_max,
                "torch_available": TORCH_AVAILABLE,
                "hf_available": HF_AVAILABLE
            },
            "download_summary": {
                "total_models": len(self.models),
                "downloaded": sum(download_results.values()),
                "verified": sum(verification_results.values()),
                "failed": len(self.models) - sum(download_results.values())
            },
            "models": {}
        }
        
        # 각 모델 상세 정보
        for model_name, model_config in self.models.items():
            model_path = Path(model_config.local_path)
            actual_size = 0
            
            if model_path.exists():
                actual_size = sum(
                    f.stat().st_size for f in model_path.rglob("*") 
                    if f.is_file()
                ) // (1024 * 1024)  # MB
            
            report["models"][model_name] = {
                "name": model_config.name,
                "repo_id": model_config.repo_id,
                "downloaded": download_results.get(model_name, False),
                "verified": verification_results.get(model_name, False),
                "expected_size_mb": model_config.size_mb,
                "actual_size_mb": actual_size,
                "download_method": model_config.download_method
            }
        
        # 보고서 저장
        report_path = self.base_dir / "download_report.json"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📄 보고서 저장: {report_path}")
            
            # 간단한 요약 출력
            print(f"\n📊 다운로드 요약:")
            print(f"  ✅ 성공: {report['download_summary']['downloaded']}")
            print(f"  🔍 검증됨: {report['download_summary']['verified']}")
            print(f"  ❌ 실패: {report['download_summary']['failed']}")
            print(f"  📁 저장 위치: {self.base_dir}")
            print(f"  📄 상세 보고서: {report_path}")
            
        except Exception as e:
            logger.error(f"❌ 보고서 저장 실패: {e}")

    async def download_specific_models(self, model_names: List[str]) -> Dict[str, bool]:
        """특정 모델들만 다운로드"""
        selected_models = {
            name: config for name, config in self.models.items()
            if name in model_names
        }
        
        if not selected_models:
            logger.error("❌ 선택된 모델이 없습니다")
            return {}
        
        print(f"📥 선택된 모델 다운로드: {', '.join(model_names)}")
        
        # 임시로 모델 목록 교체
        original_models = self.models
        self.models = selected_models
        
        try:
            results = await self.download_all_models()
            return results
        finally:
            self.models = original_models

    def list_available_models(self):
        """사용 가능한 모델 목록 출력"""
        print("📋 사용 가능한 Step 5 AI 모델들:")
        print("=" * 60)
        
        for model_name, config in self.models.items():
            status = "✅ 다운로드됨" if self._model_exists(config) else "📥 다운로드 필요"
            print(f"  {model_name:15} | {config.name:25} | {config.size_mb:>5}MB | {status}")
        
        print("=" * 60)
        total_size = sum(config.size_mb for config in self.models.values())
        print(f"총 크기: {total_size/1024:.1f}GB")

async def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Step 5 AI 모델 다운로더")
    parser.add_argument("--models", nargs="+", help="다운로드할 특정 모델들")
    parser.add_argument("--list", action="store_true", help="사용 가능한 모델 목록 출력")
    parser.add_argument("--force", action="store_true", help="강제 재다운로드")
    parser.add_argument("--base-dir", help="다운로드 기본 디렉토리")
    
    args = parser.parse_args()
    
    # 다운로더 생성
    downloader = Step05AIDownloader(args.base_dir)
    
    if args.list:
        downloader.list_available_models()
        return
    
    try:
        if args.models:
            # 특정 모델들만 다운로드
            results = await downloader.download_specific_models(args.models)
        else:
            # 모든 모델 다운로드
            results = await downloader.download_all_models(args.force)
        
        # 결과 확인
        if results:
            success_count = sum(results.values())
            total_count = len(results)
            
            if success_count == total_count:
                print("\n🎉 모든 모델 다운로드 완료!")
                print("이제 Step 5 의류 워핑을 사용할 수 있습니다.")
            else:
                print(f"\n⚠️ 일부 모델 다운로드 실패: {success_count}/{total_count}")
                failed = [name for name, success in results.items() if not success]
                print(f"실패한 모델들: {', '.join(failed)}")
                
                print("\n💡 해결 방법:")
                print("1. 인터넷 연결 확인")
                print("2. 디스크 공간 확인")
                print("3. Hugging Face 계정 로그인 (일부 모델)")
                print("4. --force 옵션으로 재시도")
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"❌ 실행 중 오류: {e}")
        raise

if __name__ == "__main__":
    # 이벤트 루프 실행
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 다운로드가 중단되었습니다")
    except Exception as e:
        print(f"❌ 치명적 오류: {e}")
        sys.exit(1)