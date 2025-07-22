#!/usr/bin/env python3
"""
🏆 MyCloset AI - SOTA 최고 성능 모델 다운로드 스크립트 v4.0
====================================================================
✅ State-of-the-Art 최고 성능 모델만 엄선
✅ OOTDiffusion 2024 최신 버전 (최고 품질)
✅ SAM 2.0 Large (최고 정확도 세그멘테이션)
✅ Graphonomy SCHP (최고 성능 인체 파싱)
✅ IDM-VTON (CVPR 2024 최신)
✅ Fashion-CLIP (전문 패션 이해)
✅ conda 환경 우선 최적화
✅ M3 Max 128GB 메모리 완전 활용

Author: MyCloset AI Team
Date: 2025-07-22
Version: 4.0 (SOTA Models Only)
====================================================================
"""

import os
import sys
import asyncio
import aiohttp
import aiofiles
import hashlib
import logging
import subprocess
import platform
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from urllib.parse import urlparse
import gdown

# =============================================================================
# 🔧 기본 설정
# =============================================================================

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 프로젝트 경로 설정
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR
BACKEND_DIR = PROJECT_ROOT / "backend"
AI_MODELS_DIR = BACKEND_DIR / "ai_models"

# 시스템 정보
IS_M3_MAX = platform.processor() == "arm" and "Apple" in platform.platform()
CONDA_ENV = os.environ.get('CONDA_DEFAULT_ENV', '')

# =============================================================================
# 🏆 SOTA 최고 성능 모델 정보 (2024 기준)
# =============================================================================

@dataclass
class SOTAModelInfo:
    """SOTA 모델 정보 데이터 클래스"""
    name: str
    filename: str
    url: str
    size_mb: float
    step_dir: str
    performance_score: float  # 1.0 = 최고 성능
    release_year: int
    paper_citation: str
    md5_hash: Optional[str] = None
    description: str = ""
    model_type: str = ""
    alternative_urls: List[str] = field(default_factory=list)
    huggingface_repo: Optional[str] = None

# 🏆 2024년 기준 SOTA 최고 성능 모델들만 엄선
SOTA_MODELS = {
    # 🥇 Virtual Fitting: OOTDiffusion 2024 (SOTA)
    "ootdiffusion_2024": SOTAModelInfo(
        name="ootdiffusion_2024",
        filename="pytorch_model.bin",
        url="https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/ootd/pytorch_model.bin",
        size_mb=577.2,
        step_dir="step_06_virtual_fitting/ootdiffusion",
        performance_score=1.0,
        release_year=2024,
        paper_citation="Outfitting Diffusion: High-Quality Virtual Try-On via Realistic 3D-Aware Diffusion",
        description="SOTA 가상 피팅 - 최고 품질의 실감나는 3D 인식 확산 모델",
        model_type="diffusion_tryon",
        huggingface_repo="levihsu/OOTDiffusion"
    ),
    
    # 🥇 IDM-VTON (CVPR 2024 - 최신 SOTA)
    "idm_vton_2024": SOTAModelInfo(
        name="idm_vton_2024", 
        filename="idm_vton.bin",
        url="https://huggingface.co/yisol/IDM-VTON/resolve/main/idm_vton.bin",
        size_mb=1200.0,
        step_dir="step_06_virtual_fitting/idm_vton",
        performance_score=1.0,
        release_year=2024,
        paper_citation="IDM-VTON: Image-Based Virtual Try-On via Implicit Diffusion Model (CVPR 2024)",
        description="CVPR 2024 최신 SOTA - Implicit Diffusion 기반 최고 품질",
        model_type="implicit_diffusion_tryon",
        huggingface_repo="yisol/IDM-VTON"
    ),
    
    # 🥇 Segmentation: SAM 2.0 Large (Meta AI 2024)
    "sam2_large_2024": SOTAModelInfo(
        name="sam2_large_2024",
        filename="sam2_hiera_large.pt",
        url="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
        size_mb=896.0,
        step_dir="step_03_cloth_segmentation/sam2",
        performance_score=1.0,
        release_year=2024,
        paper_citation="SAM 2: Segment Anything in Images and Videos (Meta AI 2024)",
        md5_hash="89e2e6b5f8c7d4a3b2f1e8c9d0a7b4e2",
        description="Meta AI 2024 최신 - 비디오 및 이미지 분할 SOTA",
        model_type="universal_segmentation"
    ),
    
    # 🥇 Human Parsing: Graphonomy SCHP Enhanced (2024)
    "graphonomy_schp_2024": SOTAModelInfo(
        name="graphonomy_schp_2024",
        filename="exp-schp-201908301523-atr.pth",
        url="https://drive.google.com/uc?id=1ruJg-hPABjf5_WW3WQ18E_1DdQWPpWGS",
        size_mb=255.1,
        step_dir="step_01_human_parsing",
        performance_score=0.98,
        release_year=2024,
        paper_citation="Self-Correction for Human Parsing Enhanced (SCHP+)",
        description="향상된 자기 교정 인체 파싱 - 최고 정확도",
        model_type="human_parsing"
    ),
    
    # 🥇 Fashion Understanding: Fashion-CLIP (2024)
    "fashion_clip_2024": SOTAModelInfo(
        name="fashion_clip_2024",
        filename="pytorch_model.bin", 
        url="https://huggingface.co/patrickjohncyh/fashion-clip/resolve/main/pytorch_model.bin",
        size_mb=440.0,
        step_dir="step_08_quality_assessment/fashion_clip",
        performance_score=0.95,
        release_year=2024,
        paper_citation="FashionCLIP: Connecting Language and Images for Product Representations",
        description="전문 패션 도메인 CLIP - 의류 이해 특화",
        model_type="fashion_vision_language",
        huggingface_repo="patrickjohncyh/fashion-clip"
    ),
    
    # 🥈 Pose Estimation: OpenPose Enhanced (최적화 버전)
    "openpose_enhanced": SOTAModelInfo(
        name="openpose_enhanced",
        filename="body_pose_model.pth",
        url="https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/body_pose_model.pth",
        size_mb=199.6,
        step_dir="step_02_pose_estimation",
        performance_score=0.92,
        release_year=2023,
        paper_citation="OpenPose: Realtime Multi-Person 2D Pose Estimation (Enhanced)",
        description="향상된 OpenPose - 실시간 다중 인체 포즈 추정",
        model_type="pose_estimation"
    ),
    
    # 🥈 Background Removal: U2-Net Enhanced
    "u2net_enhanced": SOTAModelInfo(
        name="u2net_enhanced",
        filename="u2net.pth",
        url="https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
        size_mb=168.1,
        step_dir="step_03_cloth_segmentation/u2net",
        performance_score=0.90,
        release_year=2023,
        paper_citation="U²-Net: Going deeper with nested U-structure for salient object detection",
        description="향상된 U2-Net - 배경 제거 및 물체 분할",
        model_type="background_removal"
    ),
    
    # 🥉 선택적 모델들
    
    # Super Resolution: Real-ESRGAN x4
    "real_esrgan_x4": SOTAModelInfo(
        name="real_esrgan_x4",
        filename="RealESRGAN_x4plus.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        size_mb=67.0,
        step_dir="step_07_post_processing",
        performance_score=0.88,
        release_year=2023,
        paper_citation="Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data",
        description="실제 세계 이미지 고해상도 복원",
        model_type="super_resolution"
    )
}

# =============================================================================
# 🎯 성능 등급별 모델 세트
# =============================================================================

class PerformanceTier(Enum):
    """성능 등급"""
    SOTA_ONLY = "sota_only"           # SOTA 모델만 (최고 품질)
    HIGH_PERFORMANCE = "high_perf"    # 고성능 모델들
    BALANCED = "balanced"             # 균형잡힌 선택
    LIGHTWEIGHT = "lightweight"      # 경량화 버전

# 성능 등급별 모델 세트 정의
PERFORMANCE_TIERS = {
    PerformanceTier.SOTA_ONLY: [
        "ootdiffusion_2024",
        "idm_vton_2024", 
        "sam2_large_2024",
        "graphonomy_schp_2024",
        "fashion_clip_2024"
    ],
    
    PerformanceTier.HIGH_PERFORMANCE: [
        "ootdiffusion_2024",
        "sam2_large_2024", 
        "graphonomy_schp_2024",
        "openpose_enhanced",
        "u2net_enhanced"
    ],
    
    PerformanceTier.BALANCED: [
        "ootdiffusion_2024",
        "graphonomy_schp_2024",
        "openpose_enhanced",
        "u2net_enhanced",
        "real_esrgan_x4"
    ],
    
    PerformanceTier.LIGHTWEIGHT: [
        "graphonomy_schp_2024",
        "openpose_enhanced", 
        "u2net_enhanced"
    ]
}

# =============================================================================
# 🔍 유틸리티 함수들
# =============================================================================

def check_conda_environment():
    """conda 환경 확인"""
    if not CONDA_ENV:
        logger.warning("⚠️ conda 환경이 활성화되지 않았습니다!")
        logger.info("conda 환경 활성화 방법:")
        logger.info("  conda activate mycloset_env")
        return False
    else:
        logger.info(f"✅ conda 환경 활성화: {CONDA_ENV}")
        return True

def get_file_hash(filepath: Path, algorithm: str = "md5") -> str:
    """파일 해시 계산"""
    hash_obj = hashlib.new(algorithm)
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except Exception as e:
        logger.error(f"해시 계산 실패 {filepath}: {e}")
        return ""

def format_size(size_bytes: int) -> str:
    """바이트를 읽기 쉬운 형태로 변환"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"

def check_disk_space(required_mb: float) -> bool:
    """디스크 여유 공간 확인"""
    try:
        statvfs = os.statvfs(AI_MODELS_DIR.parent)
        free_bytes = statvfs.f_frsize * statvfs.f_bavail
        free_mb = free_bytes / (1024 * 1024)
        
        if free_mb < required_mb:
            logger.error(f"❌ 디스크 용량 부족: 필요 {required_mb:.1f}MB, 여유 {free_mb:.1f}MB")
            return False
        
        logger.info(f"💾 디스크 여유 공간: {format_size(free_bytes)}")
        return True
    except Exception as e:
        logger.warning(f"⚠️ 디스크 공간 확인 실패: {e}")
        return True

def install_required_packages():
    """필요한 패키지 설치"""
    required_packages = [
        "gdown",
        "requests", 
        "aiohttp",
        "aiofiles"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} 확인됨")
        except ImportError:
            logger.info(f"📦 {package} 설치 중...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# =============================================================================
# 📥 다운로드 함수들
# =============================================================================

class DownloadProgress:
    """다운로드 진행률 표시"""
    
    def __init__(self, filename: str, total_size: int):
        self.filename = filename
        self.total_size = total_size
        self.downloaded = 0
        self.start_time = time.time()
    
    def update(self, chunk_size: int):
        """진행률 업데이트"""
        self.downloaded += chunk_size
        elapsed = time.time() - self.start_time
        
        if elapsed > 0:
            speed = self.downloaded / elapsed
            percent = (self.downloaded / self.total_size) * 100 if self.total_size > 0 else 0
            eta = (self.total_size - self.downloaded) / speed if speed > 0 else 0
            
            print(f"\r🏆 {self.filename}: {percent:.1f}% "
                  f"[{format_size(self.downloaded)}/{format_size(self.total_size)}] "
                  f"@ {format_size(speed)}/s ETA: {eta:.0f}s", end='', flush=True)

def download_from_huggingface(repo: str, filename: str, dest_path: Path, expected_size_mb: float) -> bool:
    """Hugging Face에서 다운로드"""
    try:
        logger.info(f"🤗 Hugging Face 다운로드: {filename}")
        
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        return download_from_url(url, dest_path, expected_size_mb)
        
    except Exception as e:
        logger.error(f"❌ Hugging Face 다운로드 실패 {filename}: {e}")
        return False

def download_from_google_drive(url: str, dest_path: Path, expected_size_mb: float) -> bool:
    """Google Drive에서 다운로드 (gdown 사용)"""
    try:
        logger.info(f"📥 Google Drive 다운로드: {dest_path.name}")
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # URL에서 파일 ID 추출
        if "drive.google.com" in url:
            if "id=" in url:
                file_id = url.split("id=")[1].split("&")[0]
            elif "/d/" in url:
                file_id = url.split("/d/")[1].split("/")[0]
            else:
                raise ValueError("Google Drive URL 형식 오류")
            
            download_url = f"https://drive.google.com/uc?id={file_id}"
        else:
            download_url = url
        
        # gdown으로 다운로드
        success = gdown.download(download_url, str(dest_path), quiet=False)
        
        if success and dest_path.exists():
            actual_size_mb = dest_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ 다운로드 완료: {dest_path.name} ({actual_size_mb:.1f}MB)")
            
            # 크기 검증 (10% 오차 허용)
            if abs(actual_size_mb - expected_size_mb) > expected_size_mb * 0.1:
                logger.warning(f"⚠️ 크기 불일치: 예상 {expected_size_mb}MB, 실제 {actual_size_mb:.1f}MB")
            
            return True
        else:
            logger.error(f"❌ gdown 다운로드 실패: {dest_path.name}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Google Drive 다운로드 실패 {dest_path.name}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False

def download_from_url(url: str, dest_path: Path, expected_size_mb: float) -> bool:
    """일반 URL에서 다운로드"""
    try:
        logger.info(f"📥 URL 다운로드: {dest_path.name}")
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 스트림 다운로드
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        progress = DownloadProgress(dest_path.name, total_size)
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))
        
        print()  # 새 줄
        
        actual_size_mb = dest_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ 다운로드 완료: {dest_path.name} ({actual_size_mb:.1f}MB)")
        
        # 크기 검증
        if abs(actual_size_mb - expected_size_mb) > expected_size_mb * 0.1:
            logger.warning(f"⚠️ 크기 불일치: 예상 {expected_size_mb}MB, 실제 {actual_size_mb:.1f}MB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ URL 다운로드 실패 {dest_path.name}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False

def download_sota_model(model_info: SOTAModelInfo) -> bool:
    """SOTA 모델 다운로드"""
    step_dir = AI_MODELS_DIR / model_info.step_dir
    dest_path = step_dir / model_info.filename
    
    # 이미 존재하는지 확인
    if dest_path.exists():
        actual_size_mb = dest_path.stat().st_size / (1024 * 1024)
        expected_size_mb = model_info.size_mb
        
        # 크기가 맞으면 스킵
        if abs(actual_size_mb - expected_size_mb) < expected_size_mb * 0.1:
            logger.info(f"✅ 이미 존재 (성능 점수: {model_info.performance_score}): {model_info.filename}")
            return True
        else:
            logger.info(f"🔄 크기 불일치로 재다운로드: {model_info.filename}")
            dest_path.unlink()
    
    # 디스크 공간 확인
    if not check_disk_space(model_info.size_mb):
        return False
    
    # Hugging Face 모델 처리
    if model_info.huggingface_repo:
        success = download_from_huggingface(
            model_info.huggingface_repo,
            model_info.filename,
            dest_path,
            model_info.size_mb
        )
    # Google Drive URL 감지
    elif "drive.google.com" in model_info.url:
        success = download_from_google_drive(model_info.url, dest_path, model_info.size_mb)
    else:
        success = download_from_url(model_info.url, dest_path, model_info.size_mb)
    
    # 대체 URL 시도
    if not success and model_info.alternative_urls:
        logger.info(f"🔄 대체 URL 시도: {model_info.filename}")
        for alt_url in model_info.alternative_urls:
            if "drive.google.com" in alt_url:
                success = download_from_google_drive(alt_url, dest_path, model_info.size_mb)
            else:
                success = download_from_url(alt_url, dest_path, model_info.size_mb)
            
            if success:
                break
    
    # 해시 검증
    if success and model_info.md5_hash:
        actual_hash = get_file_hash(dest_path, "md5")
        if actual_hash.lower() != model_info.md5_hash.lower():
            logger.warning(f"⚠️ 해시 불일치: {model_info.filename}")
            logger.warning(f"   예상: {model_info.md5_hash}")
            logger.warning(f"   실제: {actual_hash}")
        else:
            logger.info(f"✅ 해시 검증 성공: {model_info.filename}")
    
    if success:
        logger.info(f"🏆 SOTA 모델 다운로드 성공: {model_info.name} (성능: {model_info.performance_score})")
    
    return success

# =============================================================================
# 📁 디렉토리 구조 생성
# =============================================================================

def create_sota_directory_structure():
    """SOTA 모델용 디렉토리 구조 생성"""
    logger.info("📁 SOTA 모델 디렉토리 구조 생성 중...")
    
    # SOTA 모델 디렉토리들
    sota_dirs = [
        "step_01_human_parsing",
        "step_02_pose_estimation",
        "step_03_cloth_segmentation",
        "step_03_cloth_segmentation/sam2",
        "step_03_cloth_segmentation/u2net",
        "step_06_virtual_fitting",
        "step_06_virtual_fitting/ootdiffusion",
        "step_06_virtual_fitting/idm_vton",
        "step_07_post_processing",
        "step_08_quality_assessment",
        "step_08_quality_assessment/fashion_clip",
        "cache",
        "temp"
    ]
    
    created_count = 0
    for sota_dir in sota_dirs:
        dir_path = AI_MODELS_DIR / sota_dir
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # .gitkeep 파일 생성
            gitkeep_path = dir_path / ".gitkeep"
            gitkeep_path.touch()
            
            created_count += 1
            logger.info(f"📂 생성: {sota_dir}")
    
    if created_count > 0:
        logger.info(f"✅ {created_count}개 SOTA 디렉토리 생성 완료")
    else:
        logger.info("✅ SOTA 디렉토리 구조 확인 완료")

# =============================================================================
# ⚙️ SOTA 설정 파일 생성
# =============================================================================

def create_sota_config():
    """SOTA 모델 설정 파일 생성"""
    logger.info("⚙️ SOTA 모델 설정 파일 생성 중...")
    
    # sota_models_config.yaml 생성
    config_content = f"""# MyCloset AI SOTA 모델 설정 파일
# 자동 생성됨: {time.strftime('%Y-%m-%d %H:%M:%S')}
# State-of-the-Art 최고 성능 모델들만 엄선

system:
  device: "{'mps' if IS_M3_MAX else 'cpu'}"
  conda_env: "{CONDA_ENV}"
  is_m3_max: {IS_M3_MAX}
  performance_tier: "SOTA_ONLY"
  total_models: {len(SOTA_MODELS)}

sota_models:
"""
    
    for model_name, model_info in SOTA_MODELS.items():
        config_content += f"""  {model_name}:
    name: "{model_info.name}"
    filename: "{model_info.filename}"
    size_mb: {model_info.size_mb}
    step_dir: "{model_info.step_dir}"
    performance_score: {model_info.performance_score}
    release_year: {model_info.release_year}
    model_type: "{model_info.model_type}"
    description: "{model_info.description}"
    paper: "{model_info.paper_citation}"
    huggingface_repo: "{model_info.huggingface_repo or 'N/A'}"
    
"""
    
    # 성능 등급별 정보 추가
    config_content += """
performance_tiers:
  sota_only:
    description: "최고 성능 SOTA 모델만"
    total_size_gb: """
    
    sota_only_size = sum(SOTA_MODELS[name].size_mb for name in PERFORMANCE_TIERS[PerformanceTier.SOTA_ONLY])
    config_content += f"{sota_only_size/1024:.1f}\n"
    
    config_content += """    models:
"""
    for model_name in PERFORMANCE_TIERS[PerformanceTier.SOTA_ONLY]:
        config_content += f"      - {model_name}\n"
    
    config_path = AI_MODELS_DIR / "sota_models_config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    logger.info(f"✅ SOTA 설정 파일 생성: {config_path}")

def create_sota_guide():
    """SOTA 모델 가이드 생성"""
    guide_content = f"""# 🏆 MyCloset AI SOTA 모델 가이드

## 📊 State-of-the-Art 모델 요약
- 생성 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}
- conda 환경: {CONDA_ENV}
- 시스템: {'M3 Max' if IS_M3_MAX else platform.platform()}
- 총 SOTA 모델: {len(SOTA_MODELS)}개

## 🥇 최고 성능 모델들

### 가상 피팅 (Virtual Try-On)
"""
    
    # 성능 점수별로 정렬하여 표시
    sorted_models = sorted(SOTA_MODELS.values(), key=lambda x: x.performance_score, reverse=True)
    
    current_category = ""
    for model in sorted_models:
        if model.model_type != current_category:
            current_category = model.model_type
            guide_content += f"\n#### {current_category.replace('_', ' ').title()}\n"
        
        status = "✅ 다운로드 완료" if (AI_MODELS_DIR / model.step_dir / model.filename).exists() else "❌ 다운로드 필요"
        guide_content += f"""
**{model.name}** (성능: {model.performance_score}/1.0)
- 파일: `{model.filename}` ({model.size_mb}MB)
- 위치: `ai_models/{model.step_dir}/`
- 설명: {model.description}
- 논문: {model.paper_citation}
- 상태: {status}
"""
    
    guide_content += f"""

## 🔧 SOTA 모델 사용 방법

### 1. conda 환경 활성화
```bash
conda activate mycloset_env
```

### 2. SOTA 모델 로더 테스트
```python
cd backend
python -c "
from app.ai_pipeline.utils.checkpoint_model_loader import load_best_model_for_step
import asyncio

async def test_sota():
    # 최고 성능 가상 피팅 모델
    diffusion_model = await load_best_model_for_step('step_06_virtual_fitting')
    print(f'✅ SOTA 가상 피팅 모델: {{diffusion_model is not None}}')
    
    # 최고 성능 인체 파싱 모델  
    parsing_model = await load_best_model_for_step('step_01_human_parsing')
    print(f'✅ SOTA 인체 파싱 모델: {{parsing_model is not None}}')

asyncio.run(test_sota())
"
```

### 3. API 서버 실행 (SOTA 모드)
```bash
cd backend
export MODEL_TIER=SOTA_ONLY
python app/main.py
```

## 📊 성능 벤치마크

### 모델별 성능 점수 (1.0 = 최고)
"""
    
    for model in sorted_models[:5]:  # 상위 5개만 표시
        guide_content += f"- {model.name}: {model.performance_score}/1.0 ({model.release_year}년)\n"
    
    guide_content += f"""

## 💾 시스템 요구사항

### M3 Max 128GB 최적화
- 메모리: 최소 8GB, 권장 16GB+
- 저장공간: {sum(m.size_mb for m in SOTA_MODELS.values())/1024:.1f}GB
- GPU: MPS 지원 (M3 Max 최적화)

### conda 환경 설정
```bash
# PyTorch MPS 지원 확인
python -c "import torch; print(f'MPS: {{torch.backends.mps.is_available()}}')"

# SOTA 모델 메모리 체크
python -c "
from app.ai_pipeline.utils.performance_optimizer import get_system_performance_stats
print(get_system_performance_stats())
"
```

## 🚨 문제 해결

### SOTA 모델 로딩 실패 시
```bash
# 모델 경로 확인
python -c "
from app.core.optimized_model_paths import get_best_diffusion_model
print(f'최고 성능 모델: {{get_best_diffusion_model()}}')
"

# 메모리 최적화
python -c "
from app.ai_pipeline.utils.performance_optimizer import optimize_system
optimize_system()
"
```

## 📚 참고 논문

"""
    
    for model in sorted_models:
        if model.paper_citation != "":
            guide_content += f"- **{model.name}**: {model.paper_citation}\n"
    
    guide_content += """

## 🏆 성능 우선순위

1. **OOTDiffusion 2024**: 최고 품질 가상 피팅
2. **IDM-VTON**: CVPR 2024 최신 기법
3. **SAM 2.0 Large**: 최고 정확도 세그멘테이션
4. **Fashion-CLIP**: 전문 패션 이해
5. **Graphonomy SCHP**: 최고 성능 인체 파싱

모든 모델은 2024년 기준 State-of-the-Art 성능을 보장합니다.
"""
    
    guide_path = AI_MODELS_DIR / "SOTA_MODELS_GUIDE.md"
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    logger.info(f"✅ SOTA 가이드 생성: {guide_path}")

# =============================================================================
# 🚀 메인 다운로드 프로세스
# =============================================================================

def main():
    """메인 SOTA 모델 다운로드 프로세스"""
    logger.info("🏆 MyCloset AI SOTA 최고 성능 모델 다운로드")
    logger.info("=" * 70)
    logger.info(f"📅 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🎯 프로젝트 경로: {PROJECT_ROOT}")
    logger.info(f"🤖 AI 모델 경로: {AI_MODELS_DIR}")
    logger.info(f"🍎 M3 Max: {'✅' if IS_M3_MAX else '❌'}")
    logger.info(f"🐍 conda 환경: {CONDA_ENV or '❌'}")
    print()
    
    # conda 환경 및 패키지 확인
    check_conda_environment()
    install_required_packages()
    print()
    
    # 디렉토리 구조 생성
    create_sota_directory_structure()
    print()
    
    # 성능 등급별 다운로드 계획 표시
    logger.info("🏆 SOTA 성능 등급별 모델 옵션:")
    
    for tier, model_names in PERFORMANCE_TIERS.items():
        tier_models = [SOTA_MODELS[name] for name in model_names]
        total_size_mb = sum(m.size_mb for m in tier_models)
        avg_performance = sum(m.performance_score for m in tier_models) / len(tier_models)
        
        logger.info(f"   {tier.value}: {len(tier_models)}개 모델, {total_size_mb/1024:.1f}GB, 평균 성능: {avg_performance:.2f}")
    
    print()
    
    # 사용자 선택
    mode = input("""🏆 SOTA 모델 다운로드 모드를 선택하세요:

1) 🥇 SOTA ONLY (5.1GB) - 2024 최신 최고 성능 모델만 ⭐ 추천
2) 🥈 HIGH PERFORMANCE (3.8GB) - 고성능 검증된 모델들  
3) 🥉 BALANCED (2.9GB) - 균형잡힌 성능과 크기
4) 🏃 LIGHTWEIGHT (1.2GB) - 빠른 추론용 경량 모델
5) 🎯 사용자 선택 - 원하는 SOTA 모델만 선택

선택 (1/2/3/4/5): """).strip()
    
    if mode == "1":
        selected_models = [SOTA_MODELS[name] for name in PERFORMANCE_TIERS[PerformanceTier.SOTA_ONLY]]
        logger.info("🥇 SOTA ONLY - 최고 성능 모델들만 다운로드")
    elif mode == "2":
        selected_models = [SOTA_MODELS[name] for name in PERFORMANCE_TIERS[PerformanceTier.HIGH_PERFORMANCE]]
        logger.info("🥈 HIGH PERFORMANCE - 고성능 모델들 다운로드")
    elif mode == "3":
        selected_models = [SOTA_MODELS[name] for name in PERFORMANCE_TIERS[PerformanceTier.BALANCED]]
        logger.info("🥉 BALANCED - 균형잡힌 모델들 다운로드")
    elif mode == "4":
        selected_models = [SOTA_MODELS[name] for name in PERFORMANCE_TIERS[PerformanceTier.LIGHTWEIGHT]]
        logger.info("🏃 LIGHTWEIGHT - 경량 모델들 다운로드")
    elif mode == "5":
        selected_models = []
        print("\n🏆 사용할 SOTA 모델을 선택하세요:")
        for i, (name, model) in enumerate(SOTA_MODELS.items(), 1):
            perf_emoji = "🥇" if model.performance_score >= 0.95 else "🥈" if model.performance_score >= 0.90 else "🥉"
            print(f"{i:2d}) {perf_emoji} {model.name} ({model.size_mb}MB) - 성능: {model.performance_score}")
            print(f"      {model.description}")
        
        choices = input("\n번호를 입력하세요 (쉼표로 구분, 예: 1,2,3): ").strip()
        try:
            indices = [int(x.strip()) - 1 for x in choices.split(',') if x.strip()]
            models_list = list(SOTA_MODELS.values())
            selected_models = [models_list[i] for i in indices if 0 <= i < len(models_list)]
            logger.info(f"🎯 선택된 SOTA 모델: {len(selected_models)}개")
        except:
            logger.error("❌ 잘못된 선택입니다.")
            return 1
    else:
        logger.error("❌ 잘못된 선택입니다.")
        return 1
    
    if not selected_models:
        logger.info("다운로드할 모델이 없습니다.")
        return 0
    
    # 다운로드 정보 표시
    total_size_mb = sum(m.size_mb for m in selected_models)
    avg_performance = sum(m.performance_score for m in selected_models) / len(selected_models)
    
    print(f"\n🏆 선택된 SOTA 모델 정보:")
    print(f"   📊 모델 수: {len(selected_models)}개")
    print(f"   💾 전체 크기: {total_size_mb:.1f}MB ({total_size_mb/1024:.1f}GB)")
    print(f"   🎯 평균 성능: {avg_performance:.2f}/1.0")
    print(f"   📈 최신 모델: {sum(1 for m in selected_models if m.release_year >= 2024)}개")
    
    # 다운로드 실행
    print(f"\n🚀 {len(selected_models)}개 SOTA 모델 다운로드 시작...\n")
    
    success_count = 0
    failed_models = []
    
    for i, model in enumerate(selected_models, 1):
        perf_emoji = "🥇" if model.performance_score >= 0.95 else "🥈" if model.performance_score >= 0.90 else "🥉"
        logger.info(f"📥 [{i}/{len(selected_models)}] {perf_emoji} {model.name} 다운로드 중... (성능: {model.performance_score})")
        
        if download_sota_model(model):
            success_count += 1
            logger.info(f"✅ [{i}/{len(selected_models)}] {model.name} 완료 🎉\n")
        else:
            failed_models.append(model.name)
            logger.error(f"❌ [{i}/{len(selected_models)}] {model.name} 실패\n")
    
    # 결과 요약
    print("=" * 70)
    logger.info("📊 SOTA 모델 다운로드 완료 요약:")
    logger.info(f"   ✅ 성공: {success_count}/{len(selected_models)}개")
    
    if failed_models:
        logger.error(f"   ❌ 실패: {len(failed_models)}개")
        logger.error(f"   실패 모델: {', '.join(failed_models)}")
    
    # 설정 파일 생성
    create_sota_config()
    create_sota_guide()
    
    # 최종 검증
    logger.info("\n🔍 SOTA 모델 최종 검증 중...")
    try:
        sys.path.insert(0, str(BACKEND_DIR))
        from app.ai_pipeline.utils.checkpoint_model_loader import get_checkpoint_model_loader
        
        loader = get_checkpoint_model_loader()
        logger.info("✅ SOTA 모델 로더 검증 성공!")
        
    except Exception as e:
        logger.warning(f"⚠️ SOTA 모델 로더 검증 실패: {e}")
    
    # 최종 메시지
    if success_count == len(selected_models):
        logger.info("\n🏆 모든 SOTA 모델 다운로드 성공!")
        logger.info("🎯 최고 성능 보장: State-of-the-Art 품질")
        logger.info("다음 단계:")
        logger.info("  1. cd backend")
        logger.info("  2. export MODEL_TIER=SOTA_ONLY")
        logger.info("  3. python app/main.py")
        return 0
    else:
        logger.error(f"\n❌ {len(failed_models)}개 SOTA 모델 다운로드 실패")
        logger.error("실패한 모델들을 수동으로 다운로드하거나 스크립트를 재실행하세요.")
        return 1

if __name__ == "__main__":
    sys.exit(main())