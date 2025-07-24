#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MyCloset AI - AI 모델 자동 다운로드 스크립트 (수정된 버전)
==========================================================

🤖 기능:
- 실제 Hugging Face 모델 경로 사용
- 체크포인트 파일 검증
- 메모리 최적화 다운로드
- M3 Max 특화 설정
- 진행률 표시

💡 사용법:
python install_models.py --all                # 모든 모델 다운로드
python install_models.py --essential          # 필수 모델만
python install_models.py --model human_parsing # 특정 모델만
"""

import os
import sys
import json
import hashlib
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import urllib.request
from urllib.parse import urlparse

# 진행률 표시
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get('total', 100)
            self.desc = kwargs.get('desc', 'Progress')
            self.current = 0
        
        def update(self, n=1):
            self.current += n
            percent = (self.current / self.total) * 100
            print(f"\r{self.desc}: {percent:.1f}%", end='', flush=True)
        
        def close(self):
            print()
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            self.close()

# Hugging Face Hub
try:
    from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("⚠️ huggingface_hub가 설치되지 않음: pip install huggingface_hub")

# ============================================================================
# 📋 프로젝트 설정
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.absolute()
BACKEND_ROOT = PROJECT_ROOT / "backend"
AI_MODELS_ROOT = BACKEND_ROOT / "ai_models"

# 모델 저장 디렉토리 생성
AI_MODELS_ROOT.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 🤖 AI 모델 정의 (실제 Hugging Face 경로 기반)
# ============================================================================

@dataclass
class ModelInfo:
    """AI 모델 정보"""
    name: str
    description: str
    repo_id: str
    files: List[str]
    size_mb: float
    is_essential: bool = False
    local_path: Optional[str] = None
    checksum: Optional[str] = None
    download_url: Optional[str] = None

# MyCloset AI에서 사용하는 모델들 (수정된 실제 경로)
AI_MODELS = {
    # ========================================================================
    # 🔥 Step 1: Human Parsing (필수) - 실제 존재하는 모델 사용
    # ========================================================================
    "human_parsing_schp": ModelInfo(
        name="Human Parsing (SCHP)",
        description="Self-Correction Human Parsing 모델",
        repo_id="mattmdjaga/segformer_b2_clothes",
        files=["pytorch_model.bin", "config.json"],
        size_mb=255.1,
        is_essential=True,
        local_path="step_01_human_parsing"
    ),
    
    # ========================================================================
    # 🔥 Step 2: Pose Estimation (필수) - ControlNet 사용
    # ========================================================================
    "pose_estimation_controlnet": ModelInfo(
        name="Pose Estimation (ControlNet)",
        description="ControlNet OpenPose 기반 자세 추정 모델",
        repo_id="lllyasviel/control_v11p_sd15_openpose",
        files=["diffusion_pytorch_model.bin", "config.json"],
        size_mb=1400.0,
        is_essential=True,
        local_path="step_02_pose_estimation"
    ),
    
    # ========================================================================
    # 🔥 Step 3: Cloth Segmentation (필수) - U2Net 대체
    # ========================================================================
    "cloth_segmentation_rembg": ModelInfo(
        name="Cloth Segmentation (REMBG)",
        description="REMBG 기반 의류 분할 모델",
        repo_id="skytnt/anime-seg",
        files=["isnetis.onnx"],
        size_mb=168.1,
        is_essential=True,
        local_path="step_03_cloth_segmentation"
    ),
    
    # ========================================================================
    # 🔥 Step 6: Virtual Fitting (핵심!) - 실제 존재하는 OOTDiffusion
    # ========================================================================
    "virtual_fitting_ootd": ModelInfo(
        name="Virtual Fitting (OOTDiffusion)",
        description="OOTDiffusion 가상 피팅 모델",
        repo_id="levihsu/OOTDiffusion",
        files=[
            "ootd_hd/pytorch_model.bin",
            "ootd_hd/config.json"
        ],
        size_mb=577.2,
        is_essential=True,
        local_path="step_06_virtual_fitting"
    ),
    
    # ========================================================================
    # 🔥 보조 모델들 (선택적) - 실제 존재하는 모델들
    # ========================================================================
    "stable_diffusion_base": ModelInfo(
        name="Stable Diffusion 1.5",
        description="Stable Diffusion v1.5 기본 모델",
        repo_id="runwayml/stable-diffusion-v1-5",
        files=[
            "text_encoder/pytorch_model.bin",
            "unet/diffusion_pytorch_model.bin",
            "vae/diffusion_pytorch_model.bin"
        ],
        size_mb=4000.0,
        is_essential=False,
        local_path="stable_diffusion_v15"
    ),
    
    "clip_vit_large": ModelInfo(
        name="CLIP ViT Large",
        description="OpenAI CLIP ViT-Large 모델",
        repo_id="openai/clip-vit-large-patch14",
        files=["pytorch_model.bin", "config.json"],
        size_mb=890.0,
        is_essential=False,
        local_path="clip_vit_large"
    ),
    
    "depth_estimation": ModelInfo(
        name="Depth Estimation",
        description="MiDaS 깊이 추정 모델",
        repo_id="Intel/dpt-large",
        files=["pytorch_model.bin", "config.json"],
        size_mb=1300.0,
        is_essential=False,
        local_path="depth_estimation"
    ),
}

# ============================================================================
# 🔧 다운로드 유틸리티 (개선된 버전)
# ============================================================================

class ModelDownloader:
    """AI 모델 다운로더"""
    
    def __init__(self, base_path: Path = AI_MODELS_ROOT):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.download_stats = {
            'total_models': 0,
            'downloaded': 0,
            'skipped': 0,
            'failed': 0,
            'total_size_mb': 0.0
        }
    
    def check_model_exists(self, model_info: ModelInfo) -> bool:
        """모델 파일 존재 확인"""
        if model_info.local_path:
            local_dir = self.base_path / model_info.local_path
            if local_dir.exists():
                # 모든 필수 파일이 존재하는지 확인
                for file_name in model_info.files:
                    file_path = local_dir / Path(file_name).name
                    if not file_path.exists():
                        return False
                return True
        return False
    
    def list_repo_files_safe(self, repo_id: str) -> List[str]:
        """저장소 파일 목록 안전하게 가져오기"""
        try:
            if HF_HUB_AVAILABLE:
                return list_repo_files(repo_id)
            else:
                return []
        except Exception as e:
            print(f"⚠️ {repo_id} 파일 목록 조회 실패: {e}")
            return []
    
    def find_alternative_files(self, repo_id: str, target_files: List[str]) -> List[str]:
        """대체 파일 찾기"""
        try:
            all_files = self.list_repo_files_safe(repo_id)
            found_files = []
            
            for target_file in target_files:
                # 정확한 파일명 먼저 확인
                if target_file in all_files:
                    found_files.append(target_file)
                    continue
                
                # 패턴 매칭으로 비슷한 파일 찾기
                target_name = Path(target_file).name
                target_ext = Path(target_file).suffix
                
                alternatives = [
                    f for f in all_files 
                    if f.endswith(target_ext) and (
                        target_name.replace('_', '-') in f or
                        target_name.replace('-', '_') in f or
                        Path(f).name == target_name
                    )
                ]
                
                if alternatives:
                    found_files.append(alternatives[0])
                    print(f"   🔄 대체 파일 사용: {target_file} → {alternatives[0]}")
                else:
                    print(f"   ❌ 파일을 찾을 수 없음: {target_file}")
            
            return found_files
            
        except Exception as e:
            print(f"   ⚠️ 대체 파일 탐색 실패: {e}")
            return target_files
    
    def download_from_huggingface(self, model_info: ModelInfo) -> bool:
        """Hugging Face에서 모델 다운로드 (개선된 버전)"""
        if not HF_HUB_AVAILABLE:
            print(f"❌ Hugging Face Hub가 설치되지 않음")
            return False
        
        try:
            local_dir = self.base_path / model_info.local_path
            local_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"📥 {model_info.name} 다운로드 중...")
            print(f"   저장소: {model_info.repo_id}")
            print(f"   예상 크기: {model_info.size_mb:.1f}MB")
            
            # 실제 존재하는 파일 확인 및 대체 파일 찾기
            available_files = self.find_alternative_files(model_info.repo_id, model_info.files)
            
            if not available_files:
                print(f"   ❌ 다운로드할 파일이 없습니다.")
                return False
            
            downloaded_count = 0
            for file_name in available_files:
                try:
                    print(f"   📥 다운로드 중: {file_name}")
                    
                    # 파일 다운로드
                    downloaded_path = hf_hub_download(
                        repo_id=model_info.repo_id,
                        filename=file_name,
                        cache_dir=str(local_dir.parent / "cache"),
                        force_download=False
                    )
                    
                    # 파일을 지정된 위치로 복사
                    target_path = local_dir / Path(file_name).name
                    if Path(downloaded_path) != target_path:
                        shutil.copy2(downloaded_path, target_path)
                    
                    file_size = target_path.stat().st_size / (1024*1024)
                    print(f"   ✅ {file_name} ({file_size:.1f}MB)")
                    downloaded_count += 1
                    
                except Exception as e:
                    print(f"   ❌ {file_name} 다운로드 실패: {e}")
                    continue
            
            if downloaded_count > 0:
                print(f"✅ {model_info.name} 다운로드 완료 ({downloaded_count}/{len(available_files)} 파일)")
                return True
            else:
                print(f"❌ {model_info.name} 다운로드 실패 (모든 파일 실패)")
                return False
            
        except Exception as e:
            print(f"❌ {model_info.name} 다운로드 실패: {e}")
            return False
    
    def download_large_model_in_chunks(self, model_info: ModelInfo) -> bool:
        """대용량 모델 청크 단위 다운로드"""
        try:
            # 대용량 모델의 경우 snapshot_download 사용
            if model_info.size_mb > 1000:
                print(f"📥 대용량 모델 다운로드: {model_info.name}")
                
                local_dir = self.base_path / model_info.local_path
                
                snapshot_download(
                    repo_id=model_info.repo_id,
                    local_dir=str(local_dir),
                    cache_dir=str(self.base_path / "cache"),
                    resume_download=True
                )
                
                print(f"✅ {model_info.name} 대용량 다운로드 완료")
                return True
            else:
                return self.download_from_huggingface(model_info)
                
        except Exception as e:
            print(f"❌ 대용량 다운로드 실패: {e}")
            return self.download_from_huggingface(model_info)
    
    def verify_checksum(self, model_info: ModelInfo) -> bool:
        """체크섬 검증"""
        if not model_info.checksum or not model_info.local_path:
            return True  # 체크섬이 없으면 검증 생략
        
        try:
            local_dir = self.base_path / model_info.local_path
            
            # 모든 파일의 해시 계산
            hasher = hashlib.sha256()
            for file_name in model_info.files:
                file_path = local_dir / Path(file_name).name
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hasher.update(chunk)
            
            calculated_hash = hasher.hexdigest()
            if calculated_hash != model_info.checksum:
                print(f"⚠️ {model_info.name} 체크섬 불일치")
                return False
            
            print(f"✅ {model_info.name} 체크섬 검증 완료")
            return True
            
        except Exception as e:
            print(f"❌ {model_info.name} 체크섬 검증 실패: {e}")
            return False
    
    def download_model(self, model_key: str, model_info: ModelInfo) -> bool:
        """개별 모델 다운로드"""
        self.download_stats['total_models'] += 1
        
        # 이미 존재하는지 확인
        if self.check_model_exists(model_info):
            print(f"⏭️  {model_info.name} 이미 존재함 (건너뛰기)")
            self.download_stats['skipped'] += 1
            return True
        
        # 다운로드 시도
        success = False
        
        # 1. 대용량 모델 청크 다운로드 시도
        if model_info.size_mb > 1000:
            success = self.download_large_model_in_chunks(model_info)
        else:
            # 2. 일반 Hugging Face 다운로드 시도
            success = self.download_from_huggingface(model_info)
        
        if success:
            # 체크섬 검증
            if self.verify_checksum(model_info):
                self.download_stats['downloaded'] += 1
                self.download_stats['total_size_mb'] += model_info.size_mb
                return True
            else:
                success = False
        
        if not success:
            self.download_stats['failed'] += 1
            print(f"❌ {model_info.name} 다운로드 실패")
        
        return success
    
    def download_models(self, model_keys: List[str], 
                       parallel: bool = False, max_workers: int = 2) -> bool:
        """여러 모델 다운로드"""
        print(f"🤖 AI 모델 다운로드 시작 ({len(model_keys)}개 모델)")
        print(f"📁 저장 경로: {self.base_path}")
        print("=" * 60)
        
        if parallel and len(model_keys) > 1:
            # 병렬 다운로드 (대용량 모델은 순차 처리)
            sequential_models = []
            parallel_models = []
            
            for key in model_keys:
                if key in AI_MODELS:
                    if AI_MODELS[key].size_mb > 1000:
                        sequential_models.append(key)
                    else:
                        parallel_models.append(key)
            
            # 대용량 모델 먼저 순차 처리
            for key in sequential_models:
                if key in AI_MODELS:
                    self.download_model(key, AI_MODELS[key])
                    print()
            
            # 소용량 모델 병렬 처리
            if parallel_models:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {}
                    for key in parallel_models:
                        if key in AI_MODELS:
                            future = executor.submit(
                                self.download_model, key, AI_MODELS[key]
                            )
                            futures[future] = key
                    
                    for future in as_completed(futures):
                        key = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            print(f"❌ {key} 다운로드 중 오류: {e}")
        else:
            # 순차 다운로드
            for key in model_keys:
                if key in AI_MODELS:
                    self.download_model(key, AI_MODELS[key])
                    print()  # 빈 줄
        
        # 다운로드 통계 출력
        self.print_download_stats()
        
        return self.download_stats['failed'] == 0
    
    def print_download_stats(self):
        """다운로드 통계 출력"""
        stats = self.download_stats
        print("=" * 60)
        print("📊 다운로드 통계:")
        print(f"   전체 모델: {stats['total_models']}개")
        print(f"   다운로드: {stats['downloaded']}개")
        print(f"   건너뛰기: {stats['skipped']}개")
        print(f"   실패: {stats['failed']}개")
        print(f"   총 크기: {stats['total_size_mb']:.1f}MB")
        
        if stats['failed'] == 0:
            print("✅ 모든 모델 다운로드 완료!")
        else:
            print(f"⚠️ {stats['failed']}개 모델 다운로드 실패")

# ============================================================================
# 🛠️ 추가 유틸리티
# ============================================================================

def create_model_config():
    """모델 설정 파일 생성"""
    config = {
        "models": {},
        "paths": {
            "base_path": str(AI_MODELS_ROOT),
            "cache_path": str(AI_MODELS_ROOT / "cache")
        },
        "settings": {
            "device": "mps",  # M3 Max 기본값
            "precision": "fp16",
            "memory_optimization": True
        }
    }
    
    # 설치된 모델 정보 추가
    for key, model_info in AI_MODELS.items():
        if model_info.local_path:
            local_dir = AI_MODELS_ROOT / model_info.local_path
            if local_dir.exists():
                config["models"][key] = {
                    "name": model_info.name,
                    "path": str(local_dir),
                    "size_mb": model_info.size_mb,
                    "files": model_info.files,
                    "is_essential": model_info.is_essential
                }
    
    # 설정 파일 저장
    config_file = AI_MODELS_ROOT / "model_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 모델 설정 파일 생성: {config_file}")

def check_disk_space(required_mb: float) -> bool:
    """디스크 공간 확인"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(AI_MODELS_ROOT)
        free_mb = free / (1024 * 1024)
        
        print(f"💾 디스크 공간: {free_mb:.1f}MB 사용 가능")
        print(f"📦 필요 공간: {required_mb:.1f}MB")
        
        if free_mb < required_mb:
            print(f"❌ 디스크 공간 부족 ({required_mb - free_mb:.1f}MB 추가 필요)")
            return False
        
        return True
        
    except Exception as e:
        print(f"⚠️ 디스크 공간 확인 실패: {e}")
        return True  # 확인 실패시 계속 진행

def print_model_list():
    """사용 가능한 모델 목록 출력"""
    print("🤖 사용 가능한 AI 모델:")
    print("=" * 80)
    
    essential_models = []
    optional_models = []
    
    for key, model_info in AI_MODELS.items():
        if model_info.is_essential:
            essential_models.append((key, model_info))
        else:
            optional_models.append((key, model_info))
    
    print("\n🔥 필수 모델:")
    for key, model in essential_models:
        downloader = ModelDownloader()
        status = "✅" if downloader.check_model_exists(model) else "⬜"
        print(f"   {status} {key}: {model.name} ({model.size_mb:.1f}MB)")
        print(f"      {model.description}")
    
    print("\n📦 선택적 모델:")
    for key, model in optional_models:
        downloader = ModelDownloader()
        status = "✅" if downloader.check_model_exists(model) else "⬜"
        print(f"   {status} {key}: {model.name} ({model.size_mb:.1f}MB)")
        print(f"      {model.description}")
    
    total_size = sum(model.size_mb for model in AI_MODELS.values())
    essential_size = sum(model.size_mb for model in AI_MODELS.values() if model.is_essential)
    
    print(f"\n📊 크기 요약:")
    print(f"   필수 모델: {essential_size:.1f}MB")
    print(f"   전체 모델: {total_size:.1f}MB")

# ============================================================================
# 🚀 메인 함수
# ============================================================================

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="MyCloset AI 모델 다운로드 도구 (수정된 버전)")
    
    # 다운로드 모드
    parser.add_argument('--all', action='store_true', 
                       help='모든 모델 다운로드')
    parser.add_argument('--essential', action='store_true', 
                       help='필수 모델만 다운로드')
    parser.add_argument('--model', type=str, 
                       help='특정 모델 다운로드')
    parser.add_argument('--models', nargs='+', 
                       help='여러 모델 다운로드')
    
    # 옵션
    parser.add_argument('--list', action='store_true', 
                       help='사용 가능한 모델 목록 출력')
    parser.add_argument('--check', action='store_true', 
                       help='설치된 모델 확인')
    parser.add_argument('--parallel', action='store_true', 
                       help='병렬 다운로드 (빠르지만 불안정할 수 있음)')
    parser.add_argument('--max-workers', type=int, default=2, 
                       help='병렬 다운로드 워커 수')
    parser.add_argument('--force', action='store_true', 
                       help='기존 파일 덮어쓰기')
    
    args = parser.parse_args()
    
    print("🤖 MyCloset AI 모델 다운로드 도구 (실제 Hugging Face 경로)")
    print("=" * 50)
    
    # 모델 목록 출력
    if args.list:
        print_model_list()
        return
    
    # 설치된 모델 확인
    if args.check:
        check_installed_models()
        return
    
    # 다운로드할 모델 결정
    models_to_download = []
    
    if args.all:
        models_to_download = list(AI_MODELS.keys())
        print("📦 모든 모델 다운로드 모드")
    elif args.essential:
        models_to_download = [key for key, model in AI_MODELS.items() if model.is_essential]
        print("🔥 필수 모델만 다운로드 모드")
    elif args.model:
        if args.model in AI_MODELS:
            models_to_download = [args.model]
            print(f"🎯 특정 모델 다운로드: {args.model}")
        else:
            print(f"❌ 모델을 찾을 수 없음: {args.model}")
            print_model_list()
            return
    elif args.models:
        valid_models = [m for m in args.models if m in AI_MODELS]
        invalid_models = [m for m in args.models if m not in AI_MODELS]
        
        if invalid_models:
            print(f"❌ 잘못된 모델: {', '.join(invalid_models)}")
            print_model_list()
            return
        
        models_to_download = valid_models
        print(f"🎯 선택된 모델들: {', '.join(models_to_download)}")
    else:
        # 기본값: 필수 모델 다운로드
        models_to_download = [key for key, model in AI_MODELS.items() if model.is_essential]
        print("🔥 기본 모드: 필수 모델 다운로드")
        print("💡 모든 모델을 다운로드하려면: python install_models.py --all")
    
    if not models_to_download:
        print("❌ 다운로드할 모델이 없습니다.")
        return
    
    # 필요한 디스크 공간 계산
    total_size = sum(AI_MODELS[key].size_mb for key in models_to_download)
    
    # 디스크 공간 확인
    if not check_disk_space(total_size * 1.2):  # 20% 여유분
        if not input("계속 진행하시겠습니까? (y/N): ").lower() == 'y':
            return
    
    # Hugging Face Hub 확인
    if not HF_HUB_AVAILABLE:
        print("❌ huggingface_hub 패키지가 필요합니다:")
        print("   pip install huggingface_hub")
        return
    
    # 다운로드 실행
    downloader = ModelDownloader()
    
    print(f"\n🚀 {len(models_to_download)}개 모델 다운로드 시작")
    print(f"📁 저장 경로: {AI_MODELS_ROOT}")
    print(f"💾 예상 크기: {total_size:.1f}MB")
    
    if not args.force:
        if input("\n계속 진행하시겠습니까? (y/N): ").lower() != 'y':
            return
    
    success = downloader.download_models(
        models_to_download, 
        parallel=args.parallel, 
        max_workers=args.max_workers
    )
    
    # 모델 설정 파일 생성
    create_model_config()
    
    if success:
        print("\n🎉 모든 모델 다운로드 완료!")
        print("\n📝 다음 단계:")
        print("1. conda activate mycloset-ai-clean")
        print("2. cd backend && python app/main.py")
        print("3. 백엔드가 모델을 자동으로 로드합니다")
    else:
        print("\n⚠️ 일부 모델 다운로드에 실패했습니다.")
        print("하지만 기본 기능은 작동할 수 있습니다.")

def check_installed_models():
    """설치된 모델 확인"""
    print("📋 설치된 모델 확인:")
    print("=" * 50)
    
    downloader = ModelDownloader()
    installed_count = 0
    total_size = 0.0
    
    for key, model_info in AI_MODELS.items():
        exists = downloader.check_model_exists(model_info)
        status = "✅ 설치됨" if exists else "❌ 없음"
        essential = "🔥 필수" if model_info.is_essential else "📦 선택적"
        
        print(f"{essential} {key}: {status}")
        print(f"   이름: {model_info.name}")
        print(f"   크기: {model_info.size_mb:.1f}MB")
        
        if exists:
            installed_count += 1
            total_size += model_info.size_mb
            
            # 파일 상세 정보
            if model_info.local_path:
                local_dir = AI_MODELS_ROOT / model_info.local_path
                print(f"   경로: {local_dir}")
                for file_name in model_info.files:
                    file_path = local_dir / Path(file_name).name
                    if file_path.exists():
                        file_size = file_path.stat().st_size / (1024*1024)
                        print(f"     ✅ {file_name} ({file_size:.1f}MB)")
                    else:
                        print(f"     ❌ {file_name}")
        print()
    
    print(f"📊 요약:")
    print(f"   설치된 모델: {installed_count}/{len(AI_MODELS)}개")
    print(f"   총 크기: {total_size:.1f}MB")
    
    # 필수 모델 확인
    essential_models = [key for key, model in AI_MODELS.items() if model.is_essential]
    installed_essential = [key for key in essential_models 
                          if downloader.check_model_exists(AI_MODELS[key])]
    
    if len(installed_essential) == len(essential_models):
        print("✅ 모든 필수 모델이 설치되었습니다!")
    else:
        missing = [key for key in essential_models if key not in installed_essential]
        print(f"⚠️ 누락된 필수 모델: {', '.join(missing)}")
        print("다음 명령어로 설치하세요: python install_models.py --essential")

if __name__ == "__main__":
    main()