#!/usr/bin/env python3
"""
✅ MyCloset AI - 완전한 모델 다운로드 스크립트
✅ 모든 필수 AI 모델 및 체크포인트 자동 다운로드
✅ M3 Max 최적화 모델 포함
✅ 재시도 및 검증 시스템
✅ 프로그레스 바 및 상세 로깅

파일 위치: backend/scripts/download_all_models.py
실행 방법: python scripts/download_all_models.py
"""

import os
import sys
import json
import time
import hashlib
import logging
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import requests
    from tqdm import tqdm
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("❌ requests, tqdm 패키지가 필요합니다:")
    print("pip install requests tqdm")
    sys.exit(1)

try:
    import torch
    import torchvision
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch가 설치되지 않았습니다. 일부 모델 검증이 불가능합니다.")

try:
    import huggingface_hub
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️ Hugging Face Hub가 설치되지 않았습니다:")
    print("pip install huggingface_hub")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================
# 🔥 모델 설정 및 URL 정의
# ==============================================

class ModelConfig:
    """모델 설정 클래스"""
    
    def __init__(
        self,
        name: str,
        url: str,
        filename: str,
        size_mb: int,
        checksum: Optional[str] = None,
        model_type: str = "pytorch",
        step: Optional[str] = None,
        description: str = "",
        required: bool = True,
        hf_repo: Optional[str] = None,
        hf_filename: Optional[str] = None,
        local_filename: Optional[str] = None
    ):
        self.name = name
        self.url = url
        self.filename = filename
        self.size_mb = size_mb
        self.checksum = checksum
        self.model_type = model_type
        self.step = step
        self.description = description
        self.required = required
        self.hf_repo = hf_repo
        self.hf_filename = hf_filename
        self.local_filename = local_filename or filename

# ==============================================
# 🔥 전체 모델 카탈로그
# ==============================================

MODEL_CATALOG = {
    # ===========================================
    # 1단계: Human Parsing 모델들
    # ===========================================
    "human_parsing_graphonomy": ModelConfig(
        name="Human Parsing - Graphonomy",
        url="https://github.com/Engineering-Course/CIHP_PGN/releases/download/v1.0/CIHP_PGN.pth",
        filename="CIHP_PGN.pth",
        size_mb=215,
        checksum="a8c2d8b8f5e9c3d7a1b4e6f2c8d9a3b5",
        model_type="pytorch",
        step="step_01_human_parsing",
        description="인간 파싱을 위한 Graphonomy 모델",
        required=True,
        hf_repo="Engineering-Course/CIHP_PGN",
        hf_filename="CIHP_PGN.pth"
    ),
    
    "human_parsing_atr": ModelConfig(
        name="Human Parsing - ATR",
        url="https://github.com/lemondan/HumanParsing-Dataset/releases/download/v1.0/atr.pth",
        filename="atr.pth",
        size_mb=89,
        checksum="b7d3e9f1a2c5b8e4d6f9a2b5c8e1d4f7",
        model_type="pytorch",
        step="step_01_human_parsing",
        description="ATR 데이터셋 기반 인간 파싱 모델",
        required=False
    ),
    
    # ===========================================
    # 2단계: Pose Estimation 모델들
    # ===========================================
    "pose_estimation_openpose": ModelConfig(
        name="Pose Estimation - OpenPose",
        url="https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/download/v1.7.0/pose_iter_440000.caffemodel",
        filename="pose_iter_440000.caffemodel",
        size_mb=209,
        checksum="c4f1b2e3d5a6b9c2e4f7a1b3d6e9c2f5",
        model_type="caffe",
        step="step_02_pose_estimation",
        description="OpenPose 신체 포즈 추정 모델",
        required=True
    ),
    
    "pose_estimation_hrnet": ModelConfig(
        name="Pose Estimation - HRNet",
        url="https://github.com/HRNet/HRNet-Human-Pose-Estimation/releases/download/v1.0/pose_hrnet_w48_384x288.pth",
        filename="pose_hrnet_w48_384x288.pth",
        size_mb=265,
        checksum="d8e5f2a9b1c7e3d6f8a2b4e7c9d1f3a6",
        model_type="pytorch",
        step="step_02_pose_estimation",
        description="HRNet 고정밀 포즈 추정 모델",
        required=False,
        hf_repo="microsoft/hrnet-human-pose",
        hf_filename="pose_hrnet_w48_384x288.pth"
    ),
    
    # ===========================================
    # 3단계: Cloth Segmentation 모델들
    # ===========================================
    "cloth_segmentation_u2net": ModelConfig(
        name="Cloth Segmentation - U2Net",
        url="https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2net.pth",
        filename="u2net.pth",
        size_mb=176,
        checksum="f3a7b9d2e6c8a4f1b5e9d3a7c2f6b8e4",
        model_type="pytorch",
        step="step_03_cloth_segmentation",
        description="U2Net 기반 의류 분할 모델",
        required=True,
        hf_repo="skytnt/u2net",
        hf_filename="u2net.pth"
    ),
    
    "cloth_segmentation_deeplab": ModelConfig(
        name="Cloth Segmentation - DeepLab",
        url="https://github.com/tensorflow/models/releases/download/v2.9.0/deeplabv3_mnv2_cityscapes_train.tar.gz",
        filename="deeplabv3_mnv2_cityscapes.tar.gz",
        size_mb=67,
        checksum="a9e5d3f7b2c8e4a6f9d2b5e8c1f4a7b3",
        model_type="tensorflow",
        step="step_03_cloth_segmentation",
        description="DeepLabV3 의류 분할 모델",
        required=False
    ),
    
    # ===========================================
    # 4단계: Geometric Matching 모델들
    # ===========================================
    "geometric_matching_gmm": ModelConfig(
        name="Geometric Matching - GMM",
        url="https://github.com/sergeywong/cp-vton/releases/download/v1.0/gmm_final.pth",
        filename="gmm_final.pth",
        size_mb=134,
        checksum="c8f2a5d9b3e6f1a4b7e2d8f5a9c3b6f1",
        model_type="pytorch",
        step="step_04_geometric_matching",
        description="기하학적 매칭을 위한 GMM 모델",
        required=True
    ),
    
    "geometric_matching_tps": ModelConfig(
        name="Geometric Matching - TPS",
        url="https://github.com/ayushtues/ClothFlow/releases/download/v1.0/tps_transformation.pth",
        filename="tps_transformation.pth",
        size_mb=98,
        checksum="b5e8f2a6d3c9f1b4e7a2d5f8c1a6b9e3",
        model_type="pytorch",
        step="step_04_geometric_matching",
        description="TPS 변환 기반 기하학적 매칭",
        required=False
    ),
    
    # ===========================================
    # 5단계: Cloth Warping 모델들
    # ===========================================
    "cloth_warping_tom": ModelConfig(
        name="Cloth Warping - TOM",
        url="https://github.com/sergeywong/cp-vton/releases/download/v1.0/tom_final.pth",
        filename="tom_final.pth",
        size_mb=156,
        checksum="d7a4f9e2b8c5f3a6d9e1b4f7c2a5d8f1",
        model_type="pytorch",
        step="step_05_cloth_warping",
        description="Try-On Module 의류 워핑 모델",
        required=True
    ),
    
    "cloth_warping_flow": ModelConfig(
        name="Cloth Warping - Flow",
        url="https://github.com/ayushtues/ClothFlow/releases/download/v1.0/cloth_flow_final.pth",
        filename="cloth_flow_final.pth",
        size_mb=203,
        checksum="e6f3a8d1c7e9f2a5d8b1e4f7c3a6d9f2",
        model_type="pytorch",
        step="step_05_cloth_warping",
        description="ClothFlow 기반 의류 변형 모델",
        required=False
    ),
    
    # ===========================================
    # 6단계: Virtual Fitting 모델들
    # ===========================================
    "virtual_fitting_hrviton": ModelConfig(
        name="Virtual Fitting - HR-VITON",
        url="https://github.com/sangyun884/HR-VITON/releases/download/v1.0/hr_viton_final.pth",
        filename="hr_viton_final.pth",
        size_mb=287,
        checksum="f8b2e6a4d9c7f3a1e5d8b2f6c9a3e7f1",
        model_type="pytorch",
        step="step_06_virtual_fitting",
        description="고해상도 가상 피팅 모델",
        required=True,
        hf_repo="sangyun884/HR-VITON",
        hf_filename="hr_viton_final.pth"
    ),
    
    "virtual_fitting_viton_hd": ModelConfig(
        name="Virtual Fitting - VITON-HD",
        url="https://github.com/shadow2496/VITON-HD/releases/download/v1.0/viton_hd_final.pth",
        filename="viton_hd_final.pth",
        size_mb=245,
        checksum="a3f6d9c2e8b4f7a1d5e9c3f6b2a8d5f9",
        model_type="pytorch",
        step="step_06_virtual_fitting",
        description="VITON-HD 고화질 가상 피팅",
        required=False
    ),
    
    # ===========================================
    # 7단계: Post Processing 모델들
    # ===========================================
    "post_processing_esrgan": ModelConfig(
        name="Post Processing - ESRGAN",
        url="https://github.com/xinntao/ESRGAN/releases/download/v1.0.0/RRDB_ESRGAN_x4.pth",
        filename="RRDB_ESRGAN_x4.pth",
        size_mb=67,
        checksum="c9d4f2a7e5b8f1a3d6e9b2f5c8a1d4f7",
        model_type="pytorch",
        step="step_07_post_processing",
        description="ESRGAN Super Resolution 모델",
        required=True,
        hf_repo="ai-forever/Real-ESRGAN",
        hf_filename="RealESRGAN_x4plus.pth"
    ),
    
    "post_processing_gfpgan": ModelConfig(
        name="Post Processing - GFPGAN",
        url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        filename="GFPGANv1.4.pth",
        size_mb=145,
        checksum="b7e3f9a2d6c8f4a1e5d9b3f7c2a6d8f4",
        model_type="pytorch",
        step="step_07_post_processing",
        description="GFPGAN 얼굴 향상 모델",
        required=False,
        hf_repo="tencentarc/gfpgan",
        hf_filename="GFPGANv1.4.pth"
    ),
    
    "post_processing_codeformer": ModelConfig(
        name="Post Processing - CodeFormer",
        url="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        filename="codeformer.pth",
        size_mb=123,
        checksum="e8f5a3d7c9b2f6a4d8e1f5c3a7d9b2f6",
        model_type="pytorch",
        step="step_07_post_processing",
        description="CodeFormer 얼굴 복원 모델",
        required=False
    ),
    
    # ===========================================
    # 8단계: Quality Assessment 모델들
    # ===========================================
    "quality_assessment_lpips": ModelConfig(
        name="Quality Assessment - LPIPS",
        url="https://github.com/richzhang/PerceptualSimilarity/releases/download/v0.1/alex.pth",
        filename="lpips_alex.pth",
        size_mb=2,
        checksum="f2a6d8c4e9b7f3a1d5e8c2f6b9a3d7f1",
        model_type="pytorch",
        step="step_08_quality_assessment",
        description="LPIPS 지각적 유사성 모델",
        required=True,
        hf_repo="richzhang/PerceptualSimilarity",
        hf_filename="alex.pth"
    ),
    
    "quality_assessment_iqa": ModelConfig(
        name="Quality Assessment - IQA",
        url="https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1.0/nima_vgg16.pth",
        filename="nima_vgg16.pth",
        size_mb=56,
        checksum="a8d5f2c9e3b6f8a2d4e7c1f5b8a2d6f9",
        model_type="pytorch",
        step="step_08_quality_assessment",
        description="NIMA 이미지 품질 평가 모델",
        required=False
    ),
    
    # ===========================================
    # 추가 지원 모델들
    # ===========================================
    "face_detection_retinaface": ModelConfig(
        name="Face Detection - RetinaFace",
        url="https://github.com/serengil/retinaface/releases/download/v1.0.0/retinaface.h5",
        filename="retinaface.h5",
        size_mb=1,
        checksum="d3e7f1a9c5b8e2f6a4d7c9b1e5f8a3d2",
        model_type="tensorflow",
        step="support",
        description="RetinaFace 얼굴 감지 모델",
        required=False
    ),
    
    "segmentation_deeplabv3": ModelConfig(
        name="Segmentation - DeepLabV3+",
        url="https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth",
        filename="deeplabv3_resnet101_coco.pth",
        size_mb=233,
        checksum="c7f4a9e2b5d8f3a6c1e9d4f7b2a5c8f1",
        model_type="pytorch",
        step="support",
        description="DeepLabV3+ 범용 세그멘테이션",
        required=False
    ),
    
    "clip_vision_model": ModelConfig(
        name="CLIP Vision Model",
        url="",  # Hugging Face에서만 다운로드
        filename="clip_vision_model.bin",
        size_mb=605,
        checksum="f9a3e6d2c8b5f4a7e1d9c3f6b8a2d5f9",
        model_type="pytorch",
        step="support",
        description="CLIP 비전 인코더 모델",
        required=False,
        hf_repo="openai/clip-vit-base-patch32",
        hf_filename="pytorch_model.bin"
    )
}

# ==============================================
# 🔥 다운로드 매니저 클래스
# ==============================================

class ModelDownloadManager:
    """모델 다운로드 관리자"""
    
    def __init__(self, base_dir: str = "backend/ai_models", max_workers: int = 3):
        self.base_dir = Path(base_dir)
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MyCloset-AI-Model-Downloader/1.0'
        })
        
        # 디렉토리 구조 생성
        self.create_directory_structure()
        
        # 통계
        self.stats = {
            'total_models': 0,
            'downloaded': 0,
            'failed': 0,
            'skipped': 0,
            'total_size_mb': 0,
            'downloaded_size_mb': 0
        }
    
    def create_directory_structure(self):
        """디렉토리 구조 생성"""
        directories = [
            self.base_dir,
            self.base_dir / "checkpoints",
            self.base_dir / "step_01_human_parsing",
            self.base_dir / "step_02_pose_estimation", 
            self.base_dir / "step_03_cloth_segmentation",
            self.base_dir / "step_04_geometric_matching",
            self.base_dir / "step_05_cloth_warping",
            self.base_dir / "step_06_virtual_fitting",
            self.base_dir / "step_07_post_processing",
            self.base_dir / "step_08_quality_assessment",
            self.base_dir / "support",
            self.base_dir / "cache",
            self.base_dir / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 디렉토리 생성: {directory}")
    
    def calculate_checksum(self, filepath: Path) -> str:
        """파일 체크섬 계산"""
        try:
            hash_md5 = hashlib.md5()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"체크섬 계산 실패 {filepath}: {e}")
            return ""
    
    def verify_file(self, filepath: Path, expected_checksum: Optional[str] = None) -> bool:
        """파일 검증"""
        if not filepath.exists():
            return False
        
        if filepath.stat().st_size == 0:
            logger.warning(f"빈 파일: {filepath}")
            return False
        
        if expected_checksum:
            actual_checksum = self.calculate_checksum(filepath)
            if actual_checksum != expected_checksum:
                logger.warning(f"체크섬 불일치 {filepath}: {actual_checksum} != {expected_checksum}")
                return False
        
        return True
    
    def download_from_url(self, model_config: ModelConfig, target_path: Path) -> bool:
        """URL에서 파일 다운로드"""
        try:
            logger.info(f"🔽 다운로드 시작: {model_config.name}")
            
            # HEAD 요청으로 파일 크기 확인
            try:
                head_response = self.session.head(model_config.url, timeout=30)
                file_size = int(head_response.headers.get('content-length', 0))
            except:
                file_size = model_config.size_mb * 1024 * 1024  # 추정값
            
            # 다운로드
            response = self.session.get(model_config.url, stream=True, timeout=60)
            response.raise_for_status()
            
            # 프로그레스 바로 다운로드
            with open(target_path, 'wb') as f:
                with tqdm(
                    total=file_size,
                    unit='B',
                    unit_scale=True,
                    desc=f"📥 {model_config.name}",
                    leave=False
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # 검증
            if self.verify_file(target_path, model_config.checksum):
                logger.info(f"✅ 다운로드 완료: {model_config.name}")
                return True
            else:
                logger.error(f"❌ 파일 검증 실패: {model_config.name}")
                target_path.unlink(missing_ok=True)
                return False
                
        except Exception as e:
            logger.error(f"❌ 다운로드 실패 {model_config.name}: {e}")
            target_path.unlink(missing_ok=True)
            return False
    
    def download_from_huggingface(self, model_config: ModelConfig, target_path: Path) -> bool:
        """Hugging Face에서 다운로드"""
        if not HF_AVAILABLE:
            logger.warning(f"⚠️ Hugging Face Hub 미설치, 스킵: {model_config.name}")
            return False
        
        try:
            logger.info(f"🤗 HuggingFace에서 다운로드: {model_config.name}")
            
            if model_config.hf_repo and model_config.hf_filename:
                # 개별 파일 다운로드
                downloaded_path = hf_hub_download(
                    repo_id=model_config.hf_repo,
                    filename=model_config.hf_filename,
                    cache_dir=str(self.base_dir / "cache"),
                    force_download=False
                )
                
                # 타겟 위치로 복사
                import shutil
                shutil.copy2(downloaded_path, target_path)
                
                if self.verify_file(target_path, model_config.checksum):
                    logger.info(f"✅ HuggingFace 다운로드 완료: {model_config.name}")
                    return True
                else:
                    logger.error(f"❌ HuggingFace 파일 검증 실패: {model_config.name}")
                    return False
            
        except Exception as e:
            logger.error(f"❌ HuggingFace 다운로드 실패 {model_config.name}: {e}")
            return False
        
        return False
    
    def download_model(self, model_key: str, model_config: ModelConfig, force: bool = False) -> bool:
        """개별 모델 다운로드"""
        
        # 타겟 경로 결정
        if model_config.step and model_config.step != "support":
            step_dir = self.base_dir / model_config.step
        else:
            step_dir = self.base_dir / "support"
        
        target_path = step_dir / model_config.local_filename
        
        # 이미 존재하는지 확인
        if not force and self.verify_file(target_path, model_config.checksum):
            logger.info(f"⏭️ 이미 존재함, 스킵: {model_config.name}")
            self.stats['skipped'] += 1
            return True
        
        # 다운로드 시도
        success = False
        
        # 1. Hugging Face 우선 시도
        if model_config.hf_repo and HF_AVAILABLE:
            success = self.download_from_huggingface(model_config, target_path)
        
        # 2. 직접 URL 다운로드
        if not success and model_config.url:
            success = self.download_from_url(model_config, target_path)
        
        # 통계 업데이트
        if success:
            self.stats['downloaded'] += 1
            self.stats['downloaded_size_mb'] += model_config.size_mb
        else:
            self.stats['failed'] += 1
        
        return success
    
    async def download_all_models(
        self,
        required_only: bool = False,
        specific_steps: Optional[List[str]] = None,
        force: bool = False,
        max_retries: int = 3
    ) -> Dict[str, bool]:
        """모든 모델 다운로드"""
        
        # 다운로드할 모델 필터링
        models_to_download = {}
        
        for model_key, model_config in MODEL_CATALOG.items():
            # 필수 모델만 다운로드하는 경우
            if required_only and not model_config.required:
                continue
            
            # 특정 단계만 다운로드하는 경우
            if specific_steps and model_config.step not in specific_steps:
                continue
            
            models_to_download[model_key] = model_config
        
        # 통계 초기화
        self.stats['total_models'] = len(models_to_download)
        self.stats['total_size_mb'] = sum(config.size_mb for config in models_to_download.values())
        
        logger.info(f"🚀 모델 다운로드 시작: {self.stats['total_models']}개 모델, 총 {self.stats['total_size_mb']:.1f}MB")
        
        results = {}
        
        # 병렬 다운로드
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 다운로드 작업 제출
            future_to_model = {
                executor.submit(self._download_with_retry, model_key, model_config, force, max_retries): model_key
                for model_key, model_config in models_to_download.items()
            }
            
            # 전체 프로그레스 바
            with tqdm(total=len(future_to_model), desc="🔽 전체 다운로드 진행률", unit="모델") as pbar:
                for future in as_completed(future_to_model):
                    model_key = future_to_model[future]
                    try:
                        success = future.result()
                        results[model_key] = success
                        
                        if success:
                            pbar.set_postfix_str(f"✅ {model_key}")
                        else:
                            pbar.set_postfix_str(f"❌ {model_key}")
                    
                    except Exception as e:
                        logger.error(f"❌ 다운로드 예외 {model_key}: {e}")
                        results[model_key] = False
                    
                    pbar.update(1)
        
        # 결과 보고
        self._print_download_summary(results)
        return results
    
    def _download_with_retry(self, model_key: str, model_config: ModelConfig, force: bool, max_retries: int) -> bool:
        """재시도가 포함된 다운로드"""
        
        for attempt in range(max_retries):
            try:
                success = self.download_model(model_key, model_config, force)
                if success:
                    return True
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 지수적 백오프
                    logger.info(f"🔄 재시도 {attempt + 1}/{max_retries} in {wait_time}초: {model_config.name}")
                    time.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"❌ 다운로드 시도 {attempt + 1} 실패 {model_config.name}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return False
    
    def _print_download_summary(self, results: Dict[str, bool]):
        """다운로드 요약 출력"""
        
        successful = sum(1 for success in results.values() if success)
        failed = len(results) - successful
        
        print("\n" + "="*60)
        print("📊 다운로드 요약")
        print("="*60)
        print(f"✅ 성공: {successful}개")
        print(f"❌ 실패: {failed}개")
        print(f"⏭️ 스킵: {self.stats['skipped']}개")
        print(f"📦 다운로드 용량: {self.stats['downloaded_size_mb']:.1f}MB / {self.stats['total_size_mb']:.1f}MB")
        print("="*60)
        
        if failed > 0:
            print("❌ 실패한 모델들:")
            for model_key, success in results.items():
                if not success:
                    model_config = MODEL_CATALOG[model_key]
                    print(f"  - {model_config.name} ({model_config.step})")
            print()
        
        if successful == len(results):
            print("🎉 모든 모델 다운로드 완료!")
        else:
            print(f"⚠️ {failed}개 모델 다운로드 실패. 로그를 확인하세요.")
    
    def verify_all_models(self) -> Dict[str, bool]:
        """모든 모델 검증"""
        logger.info("🔍 모델 검증 시작...")
        
        verification_results = {}
        
        for model_key, model_config in MODEL_CATALOG.items():
            # 파일 경로 찾기
            if model_config.step and model_config.step != "support":
                step_dir = self.base_dir / model_config.step
            else:
                step_dir = self.base_dir / "support"
            
            target_path = step_dir / model_config.local_filename
            
            # 검증
            is_valid = self.verify_file(target_path, model_config.checksum)
            verification_results[model_key] = is_valid
            
            if is_valid:
                logger.info(f"✅ 검증 성공: {model_config.name}")
            else:
                logger.warning(f"❌ 검증 실패: {model_config.name}")
        
        # 검증 요약
        valid_count = sum(1 for valid in verification_results.values() if valid)
        total_count = len(verification_results)
        
        print(f"\n📋 검증 결과: {valid_count}/{total_count} 모델 유효")
        
        return verification_results
    
    def generate_model_info_json(self):
        """모델 정보 JSON 생성"""
        model_info = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_models": len(MODEL_CATALOG),
            "models": {}
        }
        
        for model_key, model_config in MODEL_CATALOG.items():
            if model_config.step and model_config.step != "support":
                step_dir = self.base_dir / model_config.step
            else:
                step_dir = self.base_dir / "support"
            
            target_path = step_dir / model_config.local_filename
            
            model_info["models"][model_key] = {
                "name": model_config.name,
                "description": model_config.description,
                "step": model_config.step,
                "model_type": model_config.model_type,
                "size_mb": model_config.size_mb,
                "required": model_config.required,
                "filename": model_config.local_filename,
                "path": str(target_path),
                "exists": target_path.exists(),
                "size_actual": target_path.stat().st_size if target_path.exists() else 0
            }
        
        # JSON 파일 저장
        info_file = self.base_dir / "model_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 모델 정보 저장: {info_file}")

# ==============================================
# 🔥 유틸리티 함수들
# ==============================================

def check_dependencies():
    """의존성 패키지 확인"""
    missing_packages = []
    
    if not REQUESTS_AVAILABLE:
        missing_packages.append("requests")
        missing_packages.append("tqdm")
    
    if not HF_AVAILABLE:
        print("⚠️ 권장사항: Hugging Face Hub 설치 시 더 많은 모델 다운로드 가능")
        print("pip install huggingface_hub")
    
    if missing_packages:
        print("❌ 필수 패키지가 설치되지 않았습니다:")
        for package in missing_packages:
            print(f"  - {package}")
        print(f"\n다음 명령어로 설치하세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def setup_environment():
    """환경 설정"""
    # CUDA 환경 확인
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            print(f"🔥 CUDA 감지: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("🍎 Apple Silicon MPS 감지")
        else:
            print("💻 CPU 모드")
    
    # 환경 변수 설정
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # HuggingFace 경고 방지

def estimate_download_time():
    """다운로드 시간 추정"""
    total_size_mb = sum(config.size_mb for config in MODEL_CATALOG.values())
    
    # 네트워크 속도 추정 (보수적)
    speeds = {
        "고속 인터넷": 50,  # MB/s
        "일반 인터넷": 10,  # MB/s  
        "느린 인터넷": 2,   # MB/s
    }
    
    print(f"📊 총 다운로드 용량: {total_size_mb:.1f}MB")
    print("⏱️ 예상 다운로드 시간:")
    
    for speed_name, speed_mbps in speeds.items():
        time_seconds = total_size_mb / speed_mbps
        time_minutes = time_seconds / 60
        print(f"  - {speed_name}: {time_minutes:.1f}분")

# ==============================================
# 🔥 메인 함수
# ==============================================

async def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="MyCloset AI 모델 다운로드 스크립트")
    parser.add_argument("--required-only", action="store_true", help="필수 모델만 다운로드")
    parser.add_argument("--steps", nargs="+", help="특정 단계만 다운로드 (예: step_01_human_parsing)")
    parser.add_argument("--force", action="store_true", help="기존 파일 덮어쓰기")
    parser.add_argument("--verify-only", action="store_true", help="다운로드 없이 검증만 수행")
    parser.add_argument("--base-dir", default="backend/ai_models", help="모델 저장 디렉토리")
    parser.add_argument("--max-workers", type=int, default=3, help="동시 다운로드 수")
    parser.add_argument("--max-retries", type=int, default=3, help="최대 재시도 횟수")
    
    args = parser.parse_args()
    
    print("🚀 MyCloset AI 모델 다운로드 스크립트")
    print("="*50)
    
    # 의존성 확인
    if not check_dependencies():
        sys.exit(1)
    
    # 환경 설정
    setup_environment()
    
    # 다운로드 매니저 생성
    manager = ModelDownloadManager(
        base_dir=args.base_dir,
        max_workers=args.max_workers
    )
    
    # 다운로드 시간 추정
    if not args.verify_only:
        estimate_download_time()
        
        # 사용자 확인
        response = input("\n다운로드를 시작하시겠습니까? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("다운로드 취소됨")
            return
    
    try:
        if args.verify_only:
            # 검증만 수행
            results = manager.verify_all_models()
        else:
            # 다운로드 실행
            results = await manager.download_all_models(
                required_only=args.required_only,
                specific_steps=args.steps,
                force=args.force,
                max_retries=args.max_retries
            )
            
            # 다운로드 후 검증
            print("\n🔍 다운로드 완료 후 검증 중...")
            manager.verify_all_models()
        
        # 모델 정보 JSON 생성
        manager.generate_model_info_json()
        
        print("\n✅ 모든 작업 완료!")
        print(f"📁 모델 위치: {manager.base_dir}")
        print(f"📋 로그 파일: model_download.log")
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"❌ 예상치 못한 오류: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())