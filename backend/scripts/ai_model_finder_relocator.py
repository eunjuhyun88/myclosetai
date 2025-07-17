#!/usr/bin/env python3
"""
🔍 AI 모델 파일 자동 탐지 및 재배치 스크립트
✅ 모든 경로에서 실제 AI 모델 파일들 탐지
✅ 올바른 위치로 자동 이동 및 심볼릭 링크 생성
✅ M3 Max 128GB 최적화
✅ conda 환경 우선 지원
"""

import os
import sys
import shutil
import hashlib
import sqlite3
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_finder.log')
    ]
)
logger = logging.getLogger(__name__)

# ==============================================
# 🔍 모델 파일 정의 및 패턴
# ==============================================

@dataclass
class ModelFileInfo:
    """모델 파일 정보"""
    name: str
    patterns: List[str]
    step: str
    required: bool
    min_size_mb: float
    max_size_mb: float
    target_path: str
    priority: int = 1
    alternative_names: List[str] = field(default_factory=list)
    file_types: List[str] = field(default_factory=lambda: ['.pth', '.pt', '.bin', '.safetensors'])

# 실제 필요한 AI 모델들 정의 (확장된 패턴)
REQUIRED_MODELS = {
    # Step 01: Human Parsing
    "human_parsing_graphonomy": ModelFileInfo(
        name="human_parsing_graphonomy",
        patterns=[
            r".*graphonomy.*\.pth$",
            r".*schp.*atr.*\.pth$", 
            r".*human.*parsing.*\.pth$",
            r".*atr.*model.*\.pth$",
            r".*lip.*parsing.*\.pth$",
            r".*segmentation.*human.*\.pth$"
        ],
        step="step_01_human_parsing",
        required=True,
        min_size_mb=50,
        max_size_mb=500,
        target_path="ai_models/checkpoints/step_01_human_parsing/graphonomy.pth",
        priority=1,
        alternative_names=["schp_atr.pth", "atr_model.pth", "human_parsing.pth"]
    ),
    
    # Step 02: Pose Estimation  
    "pose_estimation_openpose": ModelFileInfo(
        name="pose_estimation_openpose",
        patterns=[
            r".*openpose.*\.pth$",
            r".*pose.*model.*\.pth$",
            r".*body.*pose.*\.pth$",
            r".*coco.*pose.*\.pth$",
            r".*pose.*estimation.*\.pth$"
        ],
        step="step_02_pose_estimation", 
        required=True,
        min_size_mb=10,
        max_size_mb=1000,
        target_path="ai_models/checkpoints/step_02_pose_estimation/openpose.pth",
        priority=1,
        alternative_names=["body_pose_model.pth", "pose_model.pth", "openpose_model.pth"]
    ),
    
    # Step 03: Cloth Segmentation
    "cloth_segmentation_u2net": ModelFileInfo(
        name="cloth_segmentation_u2net", 
        patterns=[
            r".*u2net.*\.pth$",
            r".*cloth.*segmentation.*\.pth$",
            r".*segmentation.*cloth.*\.pth$",
            r".*u2netp.*\.pth$",
            r".*cloth.*mask.*\.pth$"
        ],
        step="step_03_cloth_segmentation",
        required=True, 
        min_size_mb=10,
        max_size_mb=200,
        target_path="ai_models/checkpoints/step_03_cloth_segmentation/u2net.pth",
        priority=1,
        alternative_names=["u2net.pth", "cloth_seg.pth", "segmentation.pth"]
    ),
    
    # Step 04: Geometric Matching
    "geometric_matching_gmm": ModelFileInfo(
        name="geometric_matching_gmm",
        patterns=[
            r".*gmm.*\.pth$",
            r".*geometric.*matching.*\.pth$", 
            r".*tps.*\.pth$",
            r".*matching.*\.pth$",
            r".*alignment.*\.pth$"
        ],
        step="geometric_matching",
        required=True,
        min_size_mb=1,
        max_size_mb=100, 
        target_path="ai_models/checkpoints/gmm_final.pth",
        priority=2,
        alternative_names=["gmm_final.pth", "geometric.pth", "matching.pth"]
    ),
    
    # Step 05: Cloth Warping  
    "cloth_warping_tom": ModelFileInfo(
        name="cloth_warping_tom",
        patterns=[
            r".*tom.*\.pth$",
            r".*cloth.*warping.*\.pth$",
            r".*warping.*\.pth$", 
            r".*try.*on.*\.pth$",
            r".*viton.*\.pth$"
        ],
        step="cloth_warping",
        required=True,
        min_size_mb=10,
        max_size_mb=200,
        target_path="ai_models/checkpoints/tom_final.pth", 
        priority=2,
        alternative_names=["tom_final.pth", "warping.pth", "cloth_warp.pth"]
    ),
    
    # Step 06: Virtual Fitting (Diffusion Models)
    "virtual_fitting_hrviton": ModelFileInfo(
        name="virtual_fitting_hrviton", 
        patterns=[
            r".*hrviton.*\.pth$",
            r".*hr.*viton.*\.pth$",
            r".*viton.*hd.*\.pth$",
            r".*virtual.*fitting.*\.pth$",
            r".*diffusion.*viton.*\.pth$"
        ],
        step="virtual_fitting",
        required=True,
        min_size_mb=100, 
        max_size_mb=2000,
        target_path="ai_models/checkpoints/hrviton_final.pth",
        priority=1,
        alternative_names=["hrviton_final.pth", "hr_viton.pth", "viton_hd.pth"]
    ),
    
    # Diffusion Models (대용량)
    "stable_diffusion": ModelFileInfo(
        name="stable_diffusion",
        patterns=[
            r".*stable.*diffusion.*\.safetensors$",
            r".*sd.*v1.*5.*\.safetensors$",
            r".*diffusion.*pytorch.*model\.bin$",
            r".*unet.*diffusion.*\.bin$",
            r".*v1-5-pruned.*\.safetensors$"
        ],
        step="diffusion_models",
        required=False,
        min_size_mb=2000,
        max_size_mb=8000, 
        target_path="ai_models/diffusion/stable-diffusion-v1-5",
        priority=2,
        alternative_names=["model.safetensors", "pytorch_model.bin"],
        file_types=['.safetensors', '.bin']
    ),
    
    # CLIP Models
    "clip_vit_base": ModelFileInfo(
        name="clip_vit_base",
        patterns=[
            r".*clip.*vit.*base.*\.bin$",
            r".*clip.*base.*patch.*\.bin$", 
            r".*pytorch.*model\.bin$"
        ],
        step="quality_assessment",
        required=False,
        min_size_mb=400,
        max_size_mb=1000,
        target_path="ai_models/clip-vit-base-patch32/pytorch_model.bin",
        priority=3,
        alternative_names=["pytorch_model.bin"],
        file_types=['.bin']
    )
}

# ==============================================
# 🔍 고급 파일 탐지기 클래스  
# ==============================================

class AIModelFinder:
    """AI 모델 파일 자동 탐지 및 분석"""
    
    def __init__(self, project_root: Optional[Path] = None):
        """초기화"""
        if project_root is None:
            # 현재 스크립트 위치에서 프로젝트 루트 찾기
            current_file = Path(__file__).resolve()
            self.project_root = current_file.parent.parent  # scripts -> backend
        else:
            self.project_root = Path(project_root)
        
        self.backend_dir = self.project_root
        self.found_models: Dict[str, List[Dict]] = {}
        self.scan_stats = {
            "total_files_scanned": 0,
            "potential_models_found": 0,
            "confirmed_models": 0,
            "scan_duration": 0.0,
            "errors": 0
        }
        
        # 검색 경로들 (우선순위 순)
        self.search_paths = self._get_comprehensive_search_paths()
        
        logger.info(f"🔍 AI 모델 탐지기 초기화 - 프로젝트 루트: {self.project_root}")
        logger.info(f"📁 검색 경로: {len(self.search_paths)}개")

    def _get_comprehensive_search_paths(self) -> List[Path]:
        """포괄적인 검색 경로 목록 생성"""
        paths = []
        
        # 1. 프로젝트 내부 경로들
        project_paths = [
            self.backend_dir / "ai_models",
            self.backend_dir / "app" / "ai_pipeline" / "models", 
            self.backend_dir / "app" / "models",
            self.backend_dir / "models",
            self.backend_dir / "checkpoints",
            self.backend_dir / "weights",
            self.backend_dir / "pretrained",
            self.backend_dir / ".." / "models",  # mycloset-ai/models
            self.backend_dir / ".." / "ai_models",  # mycloset-ai/ai_models
        ]
        
        # 2. 사용자 홈 디렉토리 경로들  
        home = Path.home()
        home_paths = [
            home / ".cache" / "huggingface" / "hub",
            home / ".cache" / "torch" / "hub", 
            home / ".cache" / "transformers",
            home / "Downloads",
            home / "Desktop",
            home / "Documents" / "AI_Models",
            home / "models",
            home / "ai_models"
        ]
        
        # 3. 시스템 공통 경로들
        system_paths = [
            Path("/opt/ml/models"),
            Path("/usr/local/share/models"),
            Path("/tmp/models"),
            Path("/var/cache/models")
        ]
        
        # 4. conda/pip 설치 경로들 
        conda_paths = self._get_conda_model_paths()
        
        # 5. 외부 저장소 경로들 (macOS 기준)
        if sys.platform == "darwin":
            external_paths = [
                Path("/Volumes") / "외장하드" / "AI_Models",  # 일반적인 외장하드
                Path("/Volumes") / "USB" / "models", 
                Path("/Volumes") / "SSD" / "ai_models"
            ]
            # 실제 마운트된 볼륨들 확인
            if Path("/Volumes").exists():
                for volume in Path("/Volumes").iterdir():
                    if volume.is_dir() and not volume.name.startswith('.'):
                        external_paths.extend([
                            volume / "AI_Models",
                            volume / "models", 
                            volume / "checkpoints",
                            volume / "Downloads"
                        ])
            system_paths.extend(external_paths)
        
        # 모든 경로 결합 (존재하는 것만)
        all_paths = project_paths + home_paths + system_paths + conda_paths
        paths = [p for p in all_paths if p.exists() and p.is_dir()]
        
        # 중복 제거 (실제 경로 기준)
        unique_paths = []
        seen_paths = set()
        for path in paths:
            try:
                real_path = path.resolve()
                if real_path not in seen_paths:
                    unique_paths.append(path)
                    seen_paths.add(real_path)
            except:
                continue
                
        return unique_paths
    
    def _get_conda_model_paths(self) -> List[Path]:
        """conda 환경의 모델 경로들 탐지"""
        conda_paths = []
        
        try:
            # conda 환경 경로 찾기
            conda_env = os.environ.get('CONDA_PREFIX')
            if conda_env:
                conda_env_path = Path(conda_env)
                conda_paths.extend([
                    conda_env_path / "lib" / "python3.11" / "site-packages" / "transformers",
                    conda_env_path / "lib" / "python3.11" / "site-packages" / "diffusers", 
                    conda_env_path / "share" / "models",
                    conda_env_path / "models"
                ])
            
            # conda 설치 루트 경로
            conda_root = os.environ.get('CONDA_ROOT') or Path.home() / "miniforge3"
            if Path(conda_root).exists():
                conda_paths.extend([
                    Path(conda_root) / "pkgs",
                    Path(conda_root) / "envs" / "mycloset-ai" / "lib" / "python3.11" / "site-packages"
                ])
                
        except Exception as e:
            logger.debug(f"conda 경로 탐지 실패: {e}")
        
        return conda_paths

    def scan_all_paths(self, max_workers: int = 4, max_depth: int = 6) -> Dict[str, List[Dict]]:
        """모든 경로에서 AI 모델 파일 스캔"""
        logger.info("🔍 전체 경로 AI 모델 파일 스캔 시작...")
        start_time = time.time()
        
        self.found_models.clear()
        self.scan_stats = {
            "total_files_scanned": 0,
            "potential_models_found": 0, 
            "confirmed_models": 0,
            "scan_duration": 0.0,
            "errors": 0
        }
        
        # 병렬 스캔 실행
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self._scan_directory, path, max_depth): path 
                for path in self.search_paths
            }
            
            for future in as_completed(future_to_path):
                search_path = future_to_path[future]
                try:
                    path_results = future.result()
                    if path_results:
                        self._merge_scan_results(path_results)
                        logger.info(f"✅ 경로 스캔 완료: {search_path} ({len(path_results)} 모델)")
                except Exception as e:
                    logger.error(f"❌ 경로 스캔 실패 {search_path}: {e}")
                    self.scan_stats["errors"] += 1
        
        self.scan_stats["scan_duration"] = time.time() - start_time
        self.scan_stats["confirmed_models"] = len(self.found_models)
        
        logger.info(f"✅ 전체 스캔 완료: {self.scan_stats['confirmed_models']}개 모델 발견")
        self._print_scan_summary()
        
        return self.found_models
    
    def _scan_directory(self, directory: Path, max_depth: int, current_depth: int = 0) -> Dict[str, List[Dict]]:
        """디렉토리 재귀 스캔"""
        results = {}
        
        if current_depth > max_depth:
            return results
            
        try:
            # 권한 확인
            if not os.access(directory, os.R_OK):
                return results
                
            items = list(directory.iterdir())
            files = [item for item in items if item.is_file()]
            subdirs = [item for item in items if item.is_dir() and not item.name.startswith('.')]
            
            # 파일들 검사
            for file_path in files:
                try:
                    self.scan_stats["total_files_scanned"] += 1
                    
                    model_info = self._analyze_file(file_path)
                    if model_info:
                        model_key = model_info['model_key']
                        if model_key not in results:
                            results[model_key] = []
                        results[model_key].append(model_info)
                        self.scan_stats["potential_models_found"] += 1
                        
                except Exception as e:
                    logger.debug(f"파일 분석 오류 {file_path}: {e}")
                    continue
            
            # 하위 디렉토리 재귀 스캔
            for subdir in subdirs:
                # 제외할 디렉토리들
                if subdir.name in ['__pycache__', '.git', 'node_modules', '.vscode', '.idea', '.pytest_cache']:
                    continue
                    
                try:
                    subdir_results = self._scan_directory(subdir, max_depth, current_depth + 1)
                    self._merge_scan_results(subdir_results, results)
                except Exception as e:
                    logger.debug(f"하위 디렉토리 스캔 오류 {subdir}: {e}")
                    continue
        
        except Exception as e:
            logger.debug(f"디렉토리 스캔 오류 {directory}: {e}")
        
        return results
    
    def _analyze_file(self, file_path: Path) -> Optional[Dict]:
        """파일 분석하여 AI 모델인지 판단"""
        try:
            # 기본 필터링
            if not self._is_potential_ai_model(file_path):
                return None
            
            file_stat = file_path.stat()
            file_size_mb = file_stat.st_size / (1024 * 1024)
            
            # 너무 작은 파일 제외
            if file_size_mb < 0.5:
                return None
            
            # 모델 매칭
            matched_models = self._match_against_known_models(file_path, file_size_mb)
            
            if not matched_models:
                return None
            
            # 가장 적합한 모델 선택
            best_match = max(matched_models, key=lambda x: x['confidence'])
            
            if best_match['confidence'] < 0.3:
                return None
            
            # 체크섬 계산 (작은 파일만)
            checksum = None
            if file_size_mb < 100:
                checksum = self._calculate_checksum(file_path)
            
            return {
                'model_key': best_match['model_key'],
                'file_path': str(file_path),
                'file_size_mb': file_size_mb,
                'confidence': best_match['confidence'],
                'checksum': checksum,
                'last_modified': file_stat.st_mtime,
                'target_path': best_match['target_path'],
                'priority': best_match['priority'],
                'step': best_match['step']
            }
            
        except Exception as e:
            logger.debug(f"파일 분석 실패 {file_path}: {e}")
            return None
    
    def _is_potential_ai_model(self, file_path: Path) -> bool:
        """AI 모델 파일 가능성 확인"""
        # 확장자 체크
        ai_extensions = {'.pth', '.pt', '.bin', '.safetensors', '.onnx', '.pkl', '.h5', '.pb'}
        if file_path.suffix.lower() not in ai_extensions:
            return False
        
        # 파일명 패턴 체크  
        file_name = file_path.name.lower()
        ai_keywords = [
            'model', 'checkpoint', 'weight', 'state_dict', 'pytorch_model',
            'diffusion', 'transformer', 'bert', 'clip', 'vit', 'resnet',
            'pose', 'parsing', 'segmentation', 'u2net', 'openpose',
            'viton', 'hrviton', 'stable', 'unet', 'vae'
        ]
        
        return any(keyword in file_name for keyword in ai_keywords)
    
    def _match_against_known_models(self, file_path: Path, file_size_mb: float) -> List[Dict]:
        """알려진 모델 패턴과 매칭"""
        matches = []
        file_path_str = str(file_path).lower()
        file_name = file_path.name.lower()
        
        for model_key, model_info in REQUIRED_MODELS.items():
            confidence = 0.0
            
            # 패턴 매칭
            import re
            for pattern in model_info.patterns:
                if re.search(pattern, file_path_str, re.IGNORECASE):
                    confidence += 20.0
                    break
            
            # 대체 이름 매칭
            for alt_name in model_info.alternative_names:
                if alt_name.lower() in file_name:
                    confidence += 15.0
                    break
            
            # 파일 크기 범위 확인
            if model_info.min_size_mb <= file_size_mb <= model_info.max_size_mb:
                confidence += 10.0
            elif file_size_mb < model_info.min_size_mb:
                confidence -= 15.0
            elif file_size_mb > model_info.max_size_mb:
                confidence -= 5.0
            
            # 파일 확장자 확인
            if file_path.suffix.lower() in model_info.file_types:
                confidence += 5.0
            
            # 경로 기반 점수
            path_parts = file_path.parts
            for part in path_parts:
                if model_info.step in part.lower():
                    confidence += 8.0
                    break
            
            # 우선순위 보너스
            if model_info.priority == 1:
                confidence += 3.0
            
            if confidence > 0:
                matches.append({
                    'model_key': model_key,
                    'confidence': min(confidence / 50.0, 1.0),  # 정규화
                    'target_path': model_info.target_path,
                    'priority': model_info.priority,
                    'step': model_info.step
                })
        
        return matches
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """파일 SHA256 체크섬 계산"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()[:16]  # 처음 16자만
        except Exception:
            return None
    
    def _merge_scan_results(self, new_results: Dict, target: Optional[Dict] = None):
        """스캔 결과 병합"""
        if target is None:
            target = self.found_models
            
        for model_key, model_list in new_results.items():
            if model_key not in target:
                target[model_key] = []
            target[model_key].extend(model_list)
    
    def _print_scan_summary(self):
        """스캔 결과 요약 출력"""
        logger.info("=" * 70)
        logger.info("🎯 AI 모델 파일 스캔 결과 요약")
        logger.info("=" * 70)
        logger.info(f"📊 스캔된 파일: {self.scan_stats['total_files_scanned']:,}개")
        logger.info(f"🔍 잠재적 모델: {self.scan_stats['potential_models_found']}개")
        logger.info(f"✅ 확인된 모델: {self.scan_stats['confirmed_models']}개")
        logger.info(f"⏱️ 스캔 시간: {self.scan_stats['scan_duration']:.2f}초")
        logger.info(f"❌ 에러: {self.scan_stats['errors']}개")
        
        if self.found_models:
            logger.info("\n📁 발견된 모델별 파일 수:")
            for model_key, files in self.found_models.items():
                total_size = sum(f['file_size_mb'] for f in files) / 1024
                logger.info(f"  {model_key}: {len(files)}개 파일 ({total_size:.2f}GB)")

# ==============================================
# 🚀 모델 재배치 및 관리자 클래스
# ==============================================

class AIModelRelocator:
    """AI 모델 파일 재배치 및 관리"""
    
    def __init__(self, project_root: Path, found_models: Dict[str, List[Dict]]):
        self.project_root = Path(project_root)
        self.backend_dir = self.project_root
        self.found_models = found_models
        
        # 타겟 디렉토리 설정
        self.target_base = self.backend_dir / "ai_models"
        self.checkpoints_dir = self.target_base / "checkpoints"
        self.diffusion_dir = self.target_base / "diffusion"
        self.clip_dir = self.target_base / "clip-vit-base-patch32"
        
        # 재배치 통계
        self.relocate_stats = {
            "copied": 0,
            "symlinked": 0,
            "skipped": 0,
            "errors": 0,
            "total_size_gb": 0.0
        }
        
        logger.info(f"🚀 AI 모델 재배치기 초기화 - 타겟: {self.target_base}")
    
    def create_directory_structure(self):
        """필요한 디렉토리 구조 생성"""
        logger.info("📁 디렉토리 구조 생성 중...")
        
        directories = [
            self.target_base,
            self.checkpoints_dir,
            self.checkpoints_dir / "step_01_human_parsing", 
            self.checkpoints_dir / "step_02_pose_estimation",
            self.checkpoints_dir / "step_03_cloth_segmentation",
            self.diffusion_dir,
            self.diffusion_dir / "stable-diffusion-v1-5",
            self.clip_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"✅ 디렉토리 생성: {directory}")
        
        logger.info("✅ 디렉토리 구조 생성 완료")
    
    def relocate_all_models(self, copy_large_files: bool = False, create_symlinks: bool = True) -> Dict[str, Any]:
        """모든 모델을 적절한 위치로 재배치"""
        logger.info("🚀 AI 모델 파일 재배치 시작...")
        
        self.create_directory_structure()
        
        relocate_plan = self._create_relocate_plan()
        
        if not relocate_plan:
            logger.warning("⚠️ 재배치할 모델이 없습니다")
            return {"success": False, "message": "No models to relocate"}
        
        logger.info(f"📋 재배치 계획: {len(relocate_plan)}개 모델")
        
        # 재배치 실행
        results = {}
        for model_key, plan in relocate_plan.items():
            try:
                result = self._relocate_single_model(
                    model_key, plan, copy_large_files, create_symlinks
                )
                results[model_key] = result
                
                if result["success"]:
                    logger.info(f"✅ {model_key} 재배치 완료")
                else:
                    logger.error(f"❌ {model_key} 재배치 실패: {result.get('error', 'Unknown')}")
                    
            except Exception as e:
                logger.error(f"❌ {model_key} 재배치 중 예외: {e}")
                results[model_key] = {"success": False, "error": str(e)}
                self.relocate_stats["errors"] += 1
        
        # 최종 결과
        summary = self._create_relocate_summary(results)
        self._print_relocate_summary(summary)
        
        return summary
    
    def _create_relocate_plan(self) -> Dict[str, Dict]:
        """재배치 계획 생성"""
        plan = {}
        
        for model_key, files in self.found_models.items():
            if not files:
                continue
            
            # 가장 적합한 파일 선택 (신뢰도 + 크기 + 최신성)
            best_file = max(files, key=lambda f: (
                f['confidence'],
                -abs(f['file_size_mb'] - self._get_expected_size(model_key)),
                f['last_modified']
            ))
            
            target_path = self.backend_dir / best_file['target_path']
            
            plan[model_key] = {
                "source_path": Path(best_file['file_path']),
                "target_path": target_path,
                "file_size_mb": best_file['file_size_mb'],
                "confidence": best_file['confidence'],
                "priority": best_file['priority'],
                "checksum": best_file.get('checksum'),
                "alternatives": [Path(f['file_path']) for f in files if f != best_file]
            }
        
        return plan
    
    def _get_expected_size(self, model_key: str) -> float:
        """모델별 예상 크기 반환"""
        if model_key in REQUIRED_MODELS:
            model_info = REQUIRED_MODELS[model_key]
            return (model_info.min_size_mb + model_info.max_size_mb) / 2
        return 100.0  # 기본값
    
    def _relocate_single_model(
        self, 
        model_key: str, 
        plan: Dict, 
        copy_large_files: bool, 
        create_symlinks: bool
    ) -> Dict[str, Any]:
        """단일 모델 재배치"""
        source_path = plan["source_path"]
        target_path = plan["target_path"]
        file_size_mb = plan["file_size_mb"]
        
        try:
            # 소스 파일 존재 확인
            if not source_path.exists():
                return {"success": False, "error": "Source file not found"}
            
            # 타겟 디렉토리 생성
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 이미 타겟이 존재하는 경우
            if target_path.exists():
                # 크기 비교
                existing_size = target_path.stat().st_size / (1024 * 1024)
                if abs(existing_size - file_size_mb) < 1.0:  # 1MB 오차 허용
                    logger.info(f"⏭️ {model_key} 이미 존재 (크기 유사)")
                    self.relocate_stats["skipped"] += 1
                    return {
                        "success": True, 
                        "action": "skipped", 
                        "reason": "Target already exists with similar size"
                    }
                else:
                    # 백업 생성
                    backup_path = target_path.with_suffix(f".backup_{int(time.time())}")
                    shutil.move(target_path, backup_path)
                    logger.info(f"📦 기존 파일 백업: {backup_path.name}")
            
            # 재배치 방법 결정
            action = self._determine_relocate_action(file_size_mb, copy_large_files, create_symlinks)
            
            if action == "copy":
                shutil.copy2(source_path, target_path)
                self.relocate_stats["copied"] += 1
                logger.info(f"📋 복사 완료: {target_path.name} ({file_size_mb:.1f}MB)")
                
            elif action == "symlink":
                # 상대 경로로 심볼릭 링크 생성 시도
                try:
                    relative_source = os.path.relpath(source_path, target_path.parent)
                    target_path.symlink_to(relative_source)
                    self.relocate_stats["symlinked"] += 1
                    logger.info(f"🔗 심볼릭 링크 생성: {target_path.name}")
                except OSError:
                    # 심볼릭 링크 실패시 복사로 폴백
                    shutil.copy2(source_path, target_path)
                    self.relocate_stats["copied"] += 1
                    logger.info(f"📋 심볼릭 링크 실패, 복사로 대체: {target_path.name}")
                    
            elif action == "hardlink":
                try:
                    os.link(source_path, target_path)
                    logger.info(f"🔗 하드 링크 생성: {target_path.name}")
                except OSError:
                    # 하드 링크 실패시 복사로 폴백
                    shutil.copy2(source_path, target_path)
                    self.relocate_stats["copied"] += 1
                    logger.info(f"📋 하드 링크 실패, 복사로 대체: {target_path.name}")
            
            # 권한 설정
            target_path.chmod(0o644)
            
            # 검증
            if target_path.exists():
                actual_size = target_path.stat().st_size / (1024 * 1024)
                self.relocate_stats["total_size_gb"] += actual_size / 1024
                
                return {
                    "success": True,
                    "action": action,
                    "source": str(source_path),
                    "target": str(target_path),
                    "size_mb": actual_size
                }
            else:
                return {"success": False, "error": "Target file not created"}
                
        except Exception as e:
            logger.error(f"❌ {model_key} 재배치 실패: {e}")
            self.relocate_stats["errors"] += 1
            return {"success": False, "error": str(e)}
    
    def _determine_relocate_action(self, file_size_mb: float, copy_large_files: bool, create_symlinks: bool) -> str:
        """재배치 방법 결정"""
        # 대용량 파일 (1GB 이상)
        if file_size_mb > 1000:
            if copy_large_files:
                return "copy"
            elif create_symlinks:
                return "symlink" 
            else:
                return "hardlink"
        
        # 중간 크기 파일 (100MB ~ 1GB)
        elif file_size_mb > 100:
            if create_symlinks:
                return "symlink"
            else:
                return "copy"
        
        # 작은 파일 (100MB 미만)
        else:
            return "copy"
    
    def _create_relocate_summary(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """재배치 결과 요약 생성"""
        successful = [k for k, v in results.items() if v.get("success")]
        failed = [k for k, v in results.items() if not v.get("success")]
        
        return {
            "success": len(failed) == 0,
            "total_models": len(results),
            "successful_count": len(successful),
            "failed_count": len(failed),
            "successful_models": successful,
            "failed_models": failed,
            "relocate_stats": self.relocate_stats,
            "results": results
        }
    
    def _print_relocate_summary(self, summary: Dict[str, Any]):
        """재배치 결과 요약 출력"""
        logger.info("=" * 70)
        logger.info("🎯 AI 모델 재배치 결과 요약")  
        logger.info("=" * 70)
        logger.info(f"📊 총 모델: {summary['total_models']}개")
        logger.info(f"✅ 성공: {summary['successful_count']}개")
        logger.info(f"❌ 실패: {summary['failed_count']}개")
        
        stats = summary['relocate_stats']
        logger.info(f"📋 복사: {stats['copied']}개")
        logger.info(f"🔗 심볼릭 링크: {stats['symlinked']}개") 
        logger.info(f"⏭️ 스킵: {stats['skipped']}개")
        logger.info(f"💾 총 크기: {stats['total_size_gb']:.2f}GB")
        
        if summary['successful_models']:
            logger.info("\n✅ 성공한 모델들:")
            for model in summary['successful_models']:
                logger.info(f"  - {model}")
        
        if summary['failed_models']:
            logger.info("\n❌ 실패한 모델들:")
            for model in summary['failed_models']:
                error = summary['results'][model].get('error', 'Unknown')
                logger.info(f"  - {model}: {error}")

# ==============================================
# 🔧 설정 파일 업데이트 클래스  
# ==============================================

class ConfigUpdater:
    """설정 파일 자동 업데이트"""
    
    def __init__(self, project_root: Path, relocate_summary: Dict[str, Any]):
        self.project_root = Path(project_root)
        self.backend_dir = self.project_root  
        self.summary = relocate_summary
        
    def update_all_configs(self):
        """모든 설정 파일 업데이트"""
        logger.info("🔧 설정 파일 업데이트 시작...")
        
        # 1. ModelLoader 설정 업데이트
        self._update_model_loader_config()
        
        # 2. 환경 변수 파일 생성
        self._create_env_file()
        
        # 3. 모델 경로 설정 파일 생성
        self._create_model_paths_config()
        
        # 4. 시작 스크립트 업데이트
        self._update_startup_script()
        
        logger.info("✅ 설정 파일 업데이트 완료")
    
    def _update_model_loader_config(self):
        """ModelLoader 설정 업데이트"""
        config_content = f'''# Auto-generated model paths configuration
"""
자동 생성된 모델 경로 설정
Generated at: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""

from pathlib import Path

# 베이스 경로
BACKEND_DIR = Path(__file__).parent.parent
AI_MODELS_DIR = BACKEND_DIR / "ai_models"

# 실제 탐지된 모델 경로들
ACTUAL_MODEL_PATHS = {{
'''

        # 성공한 모델들의 경로 추가
        for model_key in self.summary.get('successful_models', []):
            result = self.summary['results'][model_key]
            if result.get('success'):
                target_path = result.get('target')
                if target_path:
                    config_content += f'    "{model_key}": "{target_path}",\n'

        config_content += '''
}

# 모델 가용성 체크
MODEL_AVAILABILITY = {
'''

        for model_key in REQUIRED_MODELS.keys():
            is_available = model_key in self.summary.get('successful_models', [])
            config_content += f'    "{model_key}": {is_available},\n'

        config_content += '''
}

def get_model_path(model_key: str) -> str:
    """모델 경로 반환"""
    return ACTUAL_MODEL_PATHS.get(model_key, "")

def is_model_available(model_key: str) -> bool:
    """모델 사용 가능 여부 확인"""
    return MODEL_AVAILABILITY.get(model_key, False)

def get_available_models() -> list:
    """사용 가능한 모델 목록 반환"""
    return [k for k, v in MODEL_AVAILABILITY.items() if v]
'''

        # 파일 저장
        config_path = self.backend_dir / "app" / "core" / "actual_model_paths.py"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        logger.info(f"✅ ModelLoader 설정 업데이트: {config_path}")
    
    def _create_env_file(self):
        """환경 변수 파일 생성"""
        env_content = f'''# MyCloset AI Environment Configuration
# Generated at: {time.strftime("%Y-%m-%d %H:%M:%S")}

# 모델 모드 설정
MYCLOSET_MODE=development
MYCLOSET_SIMULATION=false

# AI 모델 경로
AI_MODELS_ROOT=./ai_models
CHECKPOINTS_DIR=./ai_models/checkpoints

# 모델 가용성
'''

        for model_key, is_available in zip(REQUIRED_MODELS.keys(), 
                                          [k in self.summary.get('successful_models', []) for k in REQUIRED_MODELS.keys()]):
            env_name = f"MODEL_{model_key.upper()}_AVAILABLE"
            env_content += f'{env_name}={str(is_available).lower()}\n'

        env_content += f'''
# 시스템 설정
PYTORCH_ENABLE_MPS_FALLBACK=1
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8
OMP_NUM_THREADS=16

# 로깅
LOG_LEVEL=INFO
'''

        # .env 파일 저장
        env_path = self.backend_dir / ".env"
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        logger.info(f"✅ 환경 변수 파일 생성: {env_path}")
    
    def _create_model_paths_config(self):
        """모델 경로 설정 JSON 파일 생성"""
        config_data = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "project_root": str(self.project_root),
            "ai_models_dir": str(self.backend_dir / "ai_models"),
            "model_paths": {},
            "model_availability": {},
            "relocate_summary": self.summary
        }
        
        for model_key in REQUIRED_MODELS.keys():
            is_available = model_key in self.summary.get('successful_models', [])
            config_data["model_availability"][model_key] = is_available
            
            if is_available:
                result = self.summary['results'][model_key]
                config_data["model_paths"][model_key] = result.get('target', '')
        
        # JSON 파일 저장
        json_path = self.backend_dir / "app" / "core" / "model_paths.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 모델 경로 JSON 설정: {json_path}")
    
    def _update_startup_script(self):
        """시작 스크립트 업데이트"""
        script_content = f'''#!/bin/bash
# MyCloset AI Startup Script
# Generated at: {time.strftime("%Y-%m-%d %H:%M:%S")}

echo "🚀 MyCloset AI 시작 중..."

# 환경 변수 로드
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# conda 환경 활성화 확인
if [ -z "$CONDA_PREFIX" ]; then
    echo "⚠️ conda 환경이 활성화되지 않았습니다"
    echo "다음 명령어를 실행하세요: conda activate mycloset-ai"
    exit 1
fi

# 모델 가용성 체크
echo "🔍 모델 가용성 체크..."
'''

        available_count = len(self.summary.get('successful_models', []))
        total_count = len(REQUIRED_MODELS)
        
        script_content += f'''
echo "📊 사용 가능한 모델: {available_count}/{total_count}개"
'''

        for model_key in self.summary.get('successful_models', []):
            script_content += f'echo "✅ {model_key}"\n'

        for model_key in self.summary.get('failed_models', []):
            script_content += f'echo "❌ {model_key}"\n'

        script_content += '''
# 서버 시작
echo "🌐 서버 시작 중..."
python3 app/main.py

echo "✅ MyCloset AI 서버 시작 완료"
'''

        # 스크립트 파일 저장
        script_path = self.backend_dir / "start_mycloset.sh"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # 실행 권한 부여
        script_path.chmod(0o755)
        
        logger.info(f"✅ 시작 스크립트 업데이트: {script_path}")

# ==============================================
# 🎯 메인 실행 함수
# ==============================================

def main():
    """메인 실행 함수"""
    logger.info("=" * 70)
    logger.info("🔍 AI 모델 파일 자동 탐지 및 재배치 스크립트 시작")
    logger.info("=" * 70)
    
    # 프로젝트 루트 확인
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # scripts -> backend
    
    logger.info(f"📁 프로젝트 루트: {project_root}")
    
    try:
        # 1단계: 모델 파일 탐지
        logger.info("\n🔍 1단계: AI 모델 파일 전체 탐지")
        finder = AIModelFinder(project_root)
        found_models = finder.scan_all_paths(max_workers=4, max_depth=6)
        
        if not found_models:
            logger.error("❌ 탐지된 AI 모델 파일이 없습니다")
            logger.info("💡 다음을 확인해보세요:")
            logger.info("   - ai_models 폴더가 올바른 위치에 있는지")
            logger.info("   - 모델 파일들이 올바른 확장자(.pth, .bin 등)를 가지는지")
            logger.info("   - 파일 권한이 올바르게 설정되어 있는지")
            return False
        
        # 2단계: 모델 재배치
        logger.info("\n🚀 2단계: AI 모델 파일 재배치")
        relocator = AIModelRelocator(project_root, found_models)
        
        # 사용자 옵션 (기본값으로 설정)
        copy_large_files = False  # 대용량 파일은 심볼릭 링크 사용
        create_symlinks = True    # 심볼릭 링크 생성 활성화
        
        relocate_summary = relocator.relocate_all_models(copy_large_files, create_symlinks)
        
        # 3단계: 설정 파일 업데이트
        logger.info("\n🔧 3단계: 설정 파일 업데이트")
        config_updater = ConfigUpdater(project_root, relocate_summary)
        config_updater.update_all_configs()
        
        # 최종 결과
        logger.info("\n" + "=" * 70)
        logger.info("🎉 AI 모델 자동 탐지 및 재배치 완료!")
        logger.info("=" * 70)
        
        successful_count = relocate_summary.get('successful_count', 0)
        total_count = relocate_summary.get('total_models', 0)
        
        logger.info(f"📊 최종 결과: {successful_count}/{total_count}개 모델 성공")
        
        if successful_count > 0:
            logger.info("✅ 다음 명령어로 서버를 시작할 수 있습니다:")
            logger.info(f"   cd {project_root}")
            logger.info("   ./start_mycloset.sh")
            logger.info("\n또는:")
            logger.info("   python3 app/main.py")
        else:
            logger.error("❌ 재배치된 모델이 없습니다. 모델 파일들을 확인해주세요.")
        
        return successful_count > 0
        
    except Exception as e:
        logger.error(f"❌ 스크립트 실행 중 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)