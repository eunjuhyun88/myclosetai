#!/usr/bin/env python3
"""
🔍 MyCloset AI - 완전 고도화된 AI 모델 체크포인트 검색 스크립트
=================================================================

M3 Max 128GB 최적화, conda 환경 우선, 완전 자동화 지원

특징:
- 🛡️ 권한 안전성 (macOS/Linux/Windows 대응)
- 🚀 병렬 처리 최적화 (16코어 활용)
- 🧠 AI 기반 모델 분류 (8단계 + 프레임워크)
- 📊 실시간 진행률 및 상세 분석
- 🔄 스마트 중복 제거
- 📁 자동 정리 및 이동 기능
- ⚙️ conda 환경 우선 설정
- 🎯 MyCloset AI 특화 최적화

사용법:
    python advanced_scanner.py                    # 표준 스캔
    python advanced_scanner.py --deep            # 딥 스캔
    python advanced_scanner.py --organize        # 스캔 + 자동 정리
    python advanced_scanner.py --conda-first     # conda 환경 우선
    python advanced_scanner.py --repair          # 손상된 모델 복구
"""

import os
import sys
import time
import json
import hashlib
import argparse
import subprocess
import platform
import threading
import asyncio
import sqlite3
import shutil
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
import re
import mimetypes
import pickle
from collections import defaultdict, Counter
import logging
import warnings

# 외부 라이브러리 (안전한 import)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    class tqdm:
        def __init__(self, iterable=None, total=None, desc="", **kwargs):
            self.iterable = iterable or []
            self.total = total or (len(iterable) if hasattr(iterable, '__len__') else 0)
            self.desc = desc
            self.current = 0
            
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.current += 1
                self._update()
            print()
            
        def update(self, n=1):
            self.current += n
            self._update()
            
        def _update(self):
            if self.total > 0:
                percent = (self.current / self.total) * 100
                print(f"\r{self.desc}: {self.current}/{self.total} ({percent:.1f}%)", end='', flush=True)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_scanner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================
# 📊 고급 데이터 모델
# ==============================================

@dataclass
class ModelMetadata:
    """고급 모델 메타데이터"""
    architecture: str = "unknown"
    parameters: Optional[int] = None
    precision: str = "unknown"
    framework_version: str = "unknown"
    training_framework: str = "unknown"
    has_tokenizer: bool = False
    has_config: bool = False
    is_fine_tuned: bool = False
    base_model: Optional[str] = None
    license: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
@dataclass
class ModelInfo:
    """완전한 AI 모델 정보"""
    # 기본 정보
    name: str
    path: str
    absolute_path: str
    size_bytes: int
    size_mb: float
    size_gb: float
    
    # 파일 정보
    extension: str
    mime_type: str
    created_time: datetime
    modified_time: datetime
    access_time: datetime
    checksum_md5: str
    checksum_sha256: str
    
    # AI 모델 분류
    framework: str
    model_type: str
    step_candidate: str
    confidence: float
    architecture: str
    
    # 위치 및 환경
    is_in_project: bool
    is_in_conda: bool
    conda_env_name: Optional[str]
    environment_path: Optional[str]
    parent_directory: str
    
    # 상태 및 검증
    is_valid: bool
    is_complete: bool
    is_corrupted: bool
    validation_errors: List[str]
    
    # 관계성
    companion_files: List[str]
    related_models: List[str]
    duplicate_of: Optional[str]
    
    # 고급 메타데이터
    metadata: ModelMetadata
    
    # 사용량 정보
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    importance_score: float = 0.0
    
@dataclass
class ScanConfig:
    """스캔 설정"""
    include_patterns: List[str] = field(default_factory=lambda: [
        '*.pth', '*.pt', '*.bin', '*.safetensors', '*.ckpt', '*.checkpoint',
        '*.h5', '*.pb', '*.onnx', '*.tflite', '*.pkl', '*.joblib',
        '*.model', '*.weights', '*.npz', '*.npy'
    ])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        'node_modules', '__pycache__', '.git', '.cache/pip',
        'trash', 'recycle', 'temp', 'tmp', '.DS_Store'
    ])
    min_size_mb: float = 0.1
    max_size_gb: float = 100.0
    max_depth: int = 10
    follow_symlinks: bool = False
    conda_priority: bool = True
    deep_scan: bool = False
    verify_integrity: bool = True
    extract_metadata: bool = True

@dataclass
class ScanStatistics:
    """완전한 스캔 통계"""
    # 기본 통계
    total_files_scanned: int = 0
    models_found: int = 0
    total_size_bytes: int = 0
    total_size_gb: float = 0.0
    scan_duration: float = 0.0
    
    # 위치 통계
    locations_scanned: int = 0
    conda_models: int = 0
    project_models: int = 0
    system_models: int = 0
    
    # 품질 통계
    valid_models: int = 0
    corrupted_models: int = 0
    duplicate_groups: int = 0
    unique_models: int = 0
    
    # 프레임워크 분포
    framework_distribution: Dict[str, int] = field(default_factory=dict)
    type_distribution: Dict[str, int] = field(default_factory=dict)
    step_distribution: Dict[str, int] = field(default_factory=dict)
    
    # 성능 통계
    errors_count: int = 0
    warnings_count: int = 0
    processing_speed_files_per_sec: float = 0.0

# ==============================================
# 🔍 완전 고도화된 AI 모델 스캐너
# ==============================================

class AdvancedModelScanner:
    """완전 고도화된 AI 모델 및 체크포인트 스캐너"""
    
    def __init__(self, config: ScanConfig = None):
        self.config = config or ScanConfig()
        self.project_root = Path.cwd()
        self.scan_start_time = time.time()
        
        # 스캔 결과 저장
        self.found_models: List[ModelInfo] = []
        self.scan_locations: Dict[str, List[str]] = {}
        self.duplicates: Dict[str, List[ModelInfo]] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
        # 성능 최적화
        self.cpu_count = os.cpu_count() or 4
        self.max_workers = min(self.cpu_count, 16)  # M3 Max 최적화
        
        # conda 환경 정보
        self.conda_environments = self._detect_conda_environments()
        self.current_conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        
        # 모델 분류 패턴 (고도화)
        self._init_classification_patterns()
        
        # 검증 캐시
        self.validation_cache = {}
        self.metadata_cache = {}
        
        # 데이터베이스 초기화
        self._init_database()
        
        logger.info(f"🚀 AdvancedModelScanner 초기화 완료")
        logger.info(f"💻 시스템: {platform.system()} {platform.machine()}")
        logger.info(f"🐍 Python: {platform.python_version()}")
        logger.info(f"🔧 워커: {self.max_workers}개")
        logger.info(f"🐍 Conda 환경: {len(self.conda_environments)}개 발견")
        
    def _init_classification_patterns(self):
        """AI 모델 분류 패턴 초기화 (고도화)"""
        
        # 프레임워크 패턴 (확장)
        self.framework_patterns = {
            'pytorch': {
                'extensions': ['.pth', '.pt', '.bin'],
                'magic_bytes': [b'PK', b'\x80', b'PYTORCH'],
                'indicators': ['torch', 'pytorch', 'state_dict']
            },
            'safetensors': {
                'extensions': ['.safetensors'],
                'magic_bytes': [b'{"'],
                'indicators': ['safetensors', 'huggingface']
            },
            'tensorflow': {
                'extensions': ['.pb', '.h5', '.tflite'],
                'magic_bytes': [b'\x08', b'\x89HDF'],
                'indicators': ['tensorflow', 'keras', 'saved_model']
            },
            'onnx': {
                'extensions': ['.onnx'],
                'magic_bytes': [b'\x08\x01'],
                'indicators': ['onnx', 'opset']
            },
            'diffusers': {
                'extensions': ['.bin', '.safetensors'],
                'magic_bytes': [],
                'indicators': ['diffusion', 'unet', 'vae', 'scheduler']
            },
            'transformers': {
                'extensions': ['.bin', '.safetensors'],
                'magic_bytes': [],
                'indicators': ['transformer', 'bert', 'gpt', 'clip']
            }
        }
        
        # MyCloset AI 8단계 패턴 (더 정교함)
        self.step_patterns = {
            'step_01_human_parsing': {
                'patterns': [
                    r'human.*pars.*', r'graphonomy', r'schp', r'atr.*', r'lip.*',
                    r'parsing.*human', r'self.*correction.*human',
                    r'human.*segmentation', r'body.*parsing'
                ],
                'models': ['Graphonomy', 'Self-Correction-Human-Parsing', 'ATR', 'LIP'],
                'keywords': ['human', 'parsing', 'segmentation', 'body', 'person']
            },
            'step_02_pose_estimation': {
                'patterns': [
                    r'pose.*estimation', r'openpose', r'mediapipe.*pose', r'dwpose',
                    r'body.*pose', r'keypoint.*detection', r'skeleton.*detection',
                    r'pose.*net', r'human.*pose'
                ],
                'models': ['OpenPose', 'MediaPipe', 'DWPose', 'PoseNet'],
                'keywords': ['pose', 'keypoint', 'skeleton', 'joint', 'landmark']
            },
            'step_03_cloth_segmentation': {
                'patterns': [
                    r'cloth.*seg.*', r'u2net', r'sam.*', r'segment.*anything',
                    r'mask.*rcnn', r'deeplabv3', r'segmentation.*cloth',
                    r'garment.*seg.*', r'clothing.*mask'
                ],
                'models': ['U2Net', 'SAM', 'MaskRCNN', 'DeepLabV3'],
                'keywords': ['cloth', 'garment', 'clothing', 'mask', 'segment']
            },
            'step_04_geometric_matching': {
                'patterns': [
                    r'geometric.*match.*', r'gmm.*', r'tps.*', r'spatial.*transform',
                    r'warping.*grid', r'flow.*estimation', r'optical.*flow',
                    r'matching.*network'
                ],
                'models': ['GMM', 'TPS', 'FlowNet', 'PWCNet'],
                'keywords': ['geometric', 'matching', 'flow', 'transform', 'warp']
            },
            'step_05_cloth_warping': {
                'patterns': [
                    r'cloth.*warp.*', r'tom.*', r'viton.*warp', r'deformation',
                    r'elastic.*transform', r'thin.*plate.*spline', r'warping.*net',
                    r'garment.*warp.*'
                ],
                'models': ['TOM', 'VITON', 'TPS-Warp'],
                'keywords': ['warp', 'deformation', 'elastic', 'spline', 'transform']
            },
            'step_06_virtual_fitting': {
                'patterns': [
                    r'virtual.*fit.*', r'ootdiffusion', r'stable.*diffusion',
                    r'diffusion.*unet', r'try.*on', r'outfit.*diffusion',
                    r'viton.*hd', r'hr.*viton', r'virtual.*tryon'
                ],
                'models': ['OOTDiffusion', 'VITON-HD', 'HR-VITON', 'StableDiffusion'],
                'keywords': ['virtual', 'fitting', 'tryon', 'diffusion', 'generation']
            },
            'step_07_post_processing': {
                'patterns': [
                    r'post.*process.*', r'enhancement', r'super.*resolution',
                    r'sr.*net', r'esrgan', r'real.*esrgan', r'upscal.*',
                    r'denoise.*', r'refine.*', r'enhance.*'
                ],
                'models': ['ESRGAN', 'Real-ESRGAN', 'SRResNet', 'EDSR'],
                'keywords': ['enhancement', 'super', 'resolution', 'upscale', 'denoise']
            },
            'step_08_quality_assessment': {
                'patterns': [
                    r'quality.*assess.*', r'clip.*', r'aesthetic.*', r'scoring',
                    r'evaluation', r'metric.*', r'lpips', r'ssim', r'fid.*',
                    r'perceptual.*loss'
                ],
                'models': ['CLIP', 'LPIPS', 'FID', 'SSIM'],
                'keywords': ['quality', 'assessment', 'metric', 'evaluation', 'score']
            }
        }
        
        # 모델 아키텍처 패턴
        self.architecture_patterns = {
            'transformer': [r'transformer', r'bert', r'gpt', r'clip', r'vit'],
            'cnn': [r'resnet', r'vgg', r'inception', r'mobilenet', r'efficientnet'],
            'unet': [r'unet', r'u.*net', r'segmentation'],
            'gan': [r'gan', r'generator', r'discriminator'],
            'diffusion': [r'diffusion', r'ddpm', r'ddim', r'score'],
            'autoencoder': [r'vae', r'autoencoder', r'encoder.*decoder'],
            'detection': [r'yolo', r'rcnn', r'ssd', r'detection'],
            'pose': [r'pose', r'keypoint', r'landmark']
        }

    def _detect_conda_environments(self) -> Dict[str, Path]:
        """conda 환경 자동 탐지"""
        environments = {}
        
        try:
            # conda info로 환경 목록 가져오기
            result = subprocess.run(
                ['conda', 'env', 'list', '--json'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                env_data = json.loads(result.stdout)
                for env_path in env_data.get('envs', []):
                    env_name = Path(env_path).name
                    environments[env_name] = Path(env_path)
                    
        except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError):
            # conda 명령어 실패 시 수동 탐지
            conda_bases = [
                Path.home() / "miniconda3" / "envs",
                Path.home() / "anaconda3" / "envs", 
                Path("/opt/anaconda3/envs"),
                Path("/opt/miniconda3/envs")
            ]
            
            for base in conda_bases:
                if base.exists():
                    for env_dir in base.iterdir():
                        if env_dir.is_dir() and (env_dir / "bin" / "python").exists():
                            environments[env_dir.name] = env_dir
        
        logger.info(f"🐍 발견된 conda 환경: {list(environments.keys())}")
        return environments

    def _init_database(self):
        """스캔 결과 데이터베이스 초기화"""
        db_path = self.project_root / "model_scanner.db"
        self.db_connection = sqlite3.connect(str(db_path), check_same_thread=False)
        
        # 테이블 생성
        cursor = self.db_connection.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date TEXT,
                model_path TEXT UNIQUE,
                model_name TEXT,
                size_gb REAL,
                framework TEXT,
                step_candidate TEXT,
                confidence REAL,
                checksum TEXT,
                is_valid BOOLEAN,
                metadata TEXT,
                last_seen TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date TEXT,
                total_models INTEGER,
                total_size_gb REAL,
                scan_duration REAL,
                statistics TEXT
            )
        ''')
        
        self.db_connection.commit()

    def scan_comprehensive_system(self, organize: bool = False) -> List[ModelInfo]:
        """완전한 시스템 스캔 실행"""
        logger.info("🚀 완전 고도화된 AI 모델 스캔 시작")
        logger.info("=" * 80)
        
        scan_start = time.time()
        
        # 1. 스캔 경로 최적화 생성
        scan_paths = self._generate_optimized_scan_paths()
        
        # 2. 병렬 스캔 실행
        logger.info(f"🔍 {len(scan_paths)}개 위치에서 병렬 스캔 시작...")
        all_files = self._parallel_file_discovery(scan_paths)
        
        if not all_files:
            logger.warning("❌ AI 모델 파일을 찾을 수 없습니다.")
            return []
        
        # 3. 모델 파일 분석
        logger.info(f"🧠 {len(all_files):,}개 파일 AI 분석 중...")
        self._analyze_models_advanced(all_files)
        
        # 4. 고급 후처리
        self._post_process_results()
        
        # 5. 자동 정리 (옵션)
        if organize:
            self._auto_organize_models()
        
        # 6. 결과 저장 및 출력
        scan_duration = time.time() - scan_start
        self._save_scan_results(scan_duration)
        self._print_comprehensive_report(scan_duration)
        
        return self.found_models

    def _generate_optimized_scan_paths(self) -> List[Path]:
        """최적화된 스캔 경로 생성 (conda 우선)"""
        paths = []
        
        # 1. conda 환경 우선 (가장 높은 우선순위)
        if self.config.conda_priority:
            for env_name, env_path in self.conda_environments.items():
                conda_paths = [
                    env_path / "lib" / "python3.11" / "site-packages",
                    env_path / "lib" / "python3.10" / "site-packages", 
                    env_path / "lib" / "python3.9" / "site-packages",
                    env_path / "share",
                    env_path / "models",
                    env_path / "checkpoints"
                ]
                
                for path in conda_paths:
                    if self._is_accessible_path(path):
                        paths.append(path)
        
        # 2. 프로젝트 경로 (두 번째 우선순위)
        project_paths = [
            self.project_root / "backend" / "ai_models",
            self.project_root / "ai_models",
            self.project_root / "models",
            self.project_root / "checkpoints",
            self.project_root / "weights"
        ]
        
        # 3. 사용자 경로
        home = Path.home()
        user_paths = [
            home / "Downloads",
            home / "Documents" / "AI_Models",
            home / "Desktop",
            home / ".cache" / "huggingface",
            home / ".cache" / "torch", 
            home / ".cache" / "diffusers",
            home / ".cache" / "transformers",
            home / ".local" / "lib",
            home / ".local" / "share"
        ]
        
        # 4. 시스템 경로
        system_paths = []
        system = platform.system().lower()
        
        if system == "darwin":  # macOS
            system_paths = [
                Path("/opt/homebrew/lib"),
                Path("/usr/local/lib"),
                Path("/opt"),
                Path("/Applications") if self.config.deep_scan else None
            ]
        elif system == "linux":
            system_paths = [
                Path("/opt"),
                Path("/usr/local/lib"),
                Path("/usr/share"),
                Path("/var/lib") if self.config.deep_scan else None
            ]
        else:  # Windows
            system_paths = [
                Path("C:/Program Files"),
                home / "AppData"
            ]
        
        # 경로 통합 및 필터링
        all_paths = project_paths + user_paths + [p for p in system_paths if p]
        
        if self.config.conda_priority:
            all_paths = paths + all_paths  # conda 경로를 앞에
        else:
            all_paths = all_paths + paths
        
        # 접근 가능한 경로만 반환
        final_paths = []
        for path in all_paths:
            if self._is_accessible_path(path):
                final_paths.append(path)
        
        # 중복 제거 (부모-자식 관계 확인)
        return self._remove_duplicate_paths(final_paths)

    def _is_accessible_path(self, path: Path) -> bool:
        """경로 접근 가능성 확인 (권한 안전성)"""
        try:
            if not path.exists():
                return False
            
            # 읽기 권한 확인
            if not os.access(path, os.R_OK):
                return False
            
            # 보호된 시스템 경로 제외
            path_str = str(path).lower()
            protected_patterns = [
                '/system/', '/private/var/db', '/dev/', '/proc/',
                'keychain', 'security', 'loginwindow'
            ]
            
            return not any(pattern in path_str for pattern in protected_patterns)
            
        except (PermissionError, OSError):
            return False

    def _remove_duplicate_paths(self, paths: List[Path]) -> List[Path]:
        """중복 경로 제거 (부모-자식 관계 고려)"""
        unique_paths = []
        
        # 경로 길이순 정렬 (짧은 것부터)
        sorted_paths = sorted(set(paths), key=lambda p: len(str(p)))
        
        for path in sorted_paths:
            is_child = False
            for existing in unique_paths:
                try:
                    if path != existing and path.is_relative_to(existing):
                        is_child = True
                        break
                except (ValueError, OSError):
                    continue
            
            if not is_child:
                unique_paths.append(path)
        
        return unique_paths

    def _parallel_file_discovery(self, scan_paths: List[Path]) -> List[Path]:
        """병렬 파일 발견"""
        all_files = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 각 경로별 스캔 작업 제출
            future_to_path = {
                executor.submit(self._scan_path_advanced, path): path
                for path in scan_paths
            }
            
            # 진행률 표시
            if HAS_TQDM:
                progress = tqdm(
                    as_completed(future_to_path), 
                    total=len(future_to_path),
                    desc="경로 스캔"
                )
            else:
                progress = as_completed(future_to_path)
            
            for future in progress:
                path = future_to_path[future]
                try:
                    files = future.result(timeout=120)  # 2분 제한
                    if files:
                        all_files.extend(files)
                        self.scan_locations[str(path)] = [str(f) for f in files]
                        
                        total_size_mb = sum(
                            f.stat().st_size for f in files if f.exists()
                        ) / (1024 * 1024)
                        
                        logger.info(f"✅ {path}: {len(files)}개 파일 ({total_size_mb:.1f}MB)")
                        
                except Exception as e:
                    error_msg = f"스캔 실패 {path}: {e}"
                    self.errors.append(error_msg)
                    logger.warning(f"⚠️ {error_msg}")
        
        return all_files

    def _scan_path_advanced(self, path: Path) -> List[Path]:
        """고급 경로 스캔 (최적화)"""
        found_files = []
        
        try:
            if not self._is_accessible_path(path):
                return found_files
            
            # 시스템별 최적화된 스캔
            if platform.system() != "Windows" and shutil.which('find'):
                found_files = self._unix_find_optimized(path)
            else:
                found_files = self._python_scan_optimized(path)
                
        except Exception as e:
            logger.warning(f"경로 스캔 오류 {path}: {e}")
        
        return found_files

    def _unix_find_optimized(self, path: Path) -> List[Path]:
        """Unix find 명령어 최적화"""
        found_files = []
        
        try:
            # 패턴 기반 find 명령어 구성
            patterns = []
            for pattern in self.config.include_patterns:
                patterns.extend(['-name', f"'{pattern}'"])
            
            if patterns:
                patterns = patterns[:-1] + ['-o'] + patterns[-1:]  # OR 조건
            
            cmd = [
                'find', str(path),
                '-type', 'f',
                '(', *patterns, ')',
                '-size', f'+{int(self.config.min_size_mb)}M',
                '-not', '-path', '*/.*',  # 숨김 폴더 제외
                '-not', '-path', '*/__pycache__/*',
                '-not', '-path', '*/node_modules/*',
                '-maxdepth', str(self.config.max_depth)
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, check=False
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line and not self._should_exclude_file(line):
                        file_path = Path(line)
                        if file_path.exists():
                            found_files.append(file_path)
                            
        except subprocess.SubprocessError:
            # find 실패 시 Python 방식으로 폴백
            found_files = self._python_scan_optimized(path)
            
        return found_files

    def _python_scan_optimized(self, path: Path) -> List[Path]:
        """Python 기반 최적화 스캔"""
        found_files = []
        
        try:
            for pattern in self.config.include_patterns:
                glob_pattern = f"**/{pattern}"
                
                for file_path in path.rglob(pattern):
                    if (file_path.is_file() and 
                        not self._should_exclude_file(str(file_path)) and
                        self._check_file_size(file_path)):
                        found_files.append(file_path)
                        
        except Exception as e:
            logger.warning(f"Python 스캔 실패 {path}: {e}")
        
        return found_files

    def _should_exclude_file(self, file_path: str) -> bool:
        """파일 제외 여부 판단"""
        path_lower = file_path.lower()
        
        for pattern in self.config.exclude_patterns:
            if pattern in path_lower:
                return True
        
        return False

    def _check_file_size(self, file_path: Path) -> bool:
        """파일 크기 검사"""
        try:
            size_bytes = file_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            size_gb = size_mb / 1024
            
            return (self.config.min_size_mb <= size_mb <= 
                   self.config.max_size_gb * 1024)
        except OSError:
            return False

    def _analyze_models_advanced(self, model_files: List[Path]):
        """고급 모델 분석 (병렬 + AI 기반)"""
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 분석 작업 제출
            futures = [
                executor.submit(self._analyze_single_model_advanced, file_path)
                for file_path in model_files
            ]
            
            # 진행률 표시
            if HAS_TQDM:
                progress = tqdm(
                    as_completed(futures),
                    total=len(futures), 
                    desc="AI 모델 분석"
                )
            else:
                progress = as_completed(futures)
            
            for future in progress:
                try:
                    model_info = future.result()
                    if model_info and model_info.is_valid:
                        self.found_models.append(model_info)
                        
                except Exception as e:
                    self.errors.append(f"모델 분석 실패: {e}")

    def _analyze_single_model_advanced(self, file_path: Path) -> Optional[ModelInfo]:
        """단일 모델 고급 분석"""
        try:
            # 기본 파일 정보
            stat_info = file_path.stat()
            size_bytes = stat_info.st_size
            size_mb = size_bytes / (1024 * 1024)
            size_gb = size_mb / 1024
            
            # 체크섬 계산 (최적화)
            checksums = self._calculate_checksums_optimized(file_path, size_mb)
            
            # 프레임워크 분류 (고도화)
            framework = self._classify_framework_advanced(file_path)
            
            # 모델 타입 및 단계 분류
            model_type = self._classify_model_type_advanced(file_path)
            step_candidate, confidence = self._classify_step_advanced(file_path)
            architecture = self._classify_architecture(file_path)
            
            # 환경 정보
            conda_info = self._check_conda_environment(file_path)
            is_in_project = self._is_in_project(file_path)
            
            # 검증 및 메타데이터
            validation_result = self._validate_model_advanced(file_path, framework)
            metadata = self._extract_metadata_advanced(file_path, framework)
            
            # 관련 파일 탐지
            companion_files = self._find_companion_files(file_path)
            
            # 중요도 점수 계산
            importance_score = self._calculate_importance_score(
                file_path, size_gb, framework, step_candidate, confidence
            )
            
            return ModelInfo(
                # 기본 정보
                name=file_path.name,
                path=str(file_path),
                absolute_path=str(file_path.absolute()),
                size_bytes=size_bytes,
                size_mb=size_mb,
                size_gb=size_gb,
                
                # 파일 정보  
                extension=file_path.suffix.lower(),
                mime_type=mimetypes.guess_type(str(file_path))[0] or 'unknown',
                created_time=datetime.fromtimestamp(stat_info.st_ctime),
                modified_time=datetime.fromtimestamp(stat_info.st_mtime),
                access_time=datetime.fromtimestamp(stat_info.st_atime),
                checksum_md5=checksums['md5'],
                checksum_sha256=checksums['sha256'],
                
                # AI 모델 분류
                framework=framework,
                model_type=model_type,
                step_candidate=step_candidate,
                confidence=confidence,
                architecture=architecture,
                
                # 위치 및 환경
                is_in_project=is_in_project,
                is_in_conda=conda_info['is_conda'],
                conda_env_name=conda_info['env_name'],
                environment_path=conda_info['env_path'],
                parent_directory=file_path.parent.name,
                
                # 상태 및 검증
                is_valid=validation_result['is_valid'],
                is_complete=validation_result['is_complete'],
                is_corrupted=validation_result['is_corrupted'],
                validation_errors=validation_result['errors'],
                
                # 관계성
                companion_files=companion_files,
                related_models=[],  # 후처리에서 설정
                duplicate_of=None,  # 후처리에서 설정
                
                # 고급 메타데이터
                metadata=metadata,
                importance_score=importance_score
            )
            
        except Exception as e:
            logger.warning(f"모델 분석 실패 {file_path}: {e}")
            return None

    def _calculate_checksums_optimized(self, file_path: Path, size_mb: float) -> Dict[str, str]:
        """최적화된 체크섬 계산"""
        checksums = {'md5': 'unknown', 'sha256': 'unknown'}
        
        try:
            md5_hasher = hashlib.md5()
            sha256_hasher = hashlib.sha256()
            
            with open(file_path, 'rb') as f:
                if size_mb > 100:  # 100MB 이상은 샘플링
                    # 시작 부분
                    chunk = f.read(1024 * 1024)  # 1MB
                    if chunk:
                        md5_hasher.update(chunk)
                        sha256_hasher.update(chunk)
                    
                    # 중간 부분
                    try:
                        f.seek(int(size_mb * 1024 * 512))  # 중간
                        chunk = f.read(1024 * 1024)
                        if chunk:
                            md5_hasher.update(chunk)
                            sha256_hasher.update(chunk)
                    except:
                        pass
                    
                    # 끝 부분
                    try:
                        f.seek(-1024 * 1024, 2)  # 끝에서 1MB
                        chunk = f.read(1024 * 1024)
                        if chunk:
                            md5_hasher.update(chunk)
                            sha256_hasher.update(chunk)
                    except:
                        pass
                else:
                    # 작은 파일은 전체 해시
                    while True:
                        chunk = f.read(8192)
                        if not chunk:
                            break
                        md5_hasher.update(chunk)
                        sha256_hasher.update(chunk)
            
            checksums['md5'] = md5_hasher.hexdigest()
            checksums['sha256'] = sha256_hasher.hexdigest()[:32]  # 처음 32자만
            
        except Exception as e:
            logger.warning(f"체크섬 계산 실패 {file_path}: {e}")
        
        return checksums

    def _classify_framework_advanced(self, file_path: Path) -> str:
        """고급 프레임워크 분류"""
        extension = file_path.suffix.lower()
        path_str = str(file_path).lower()
        
        # 확장자 기반 1차 분류
        for framework, info in self.framework_patterns.items():
            if extension in info['extensions']:
                # 추가 검증
                if info['indicators']:
                    for indicator in info['indicators']:
                        if indicator in path_str:
                            return framework
                return framework
        
        # 매직 바이트 검증 (소규모 파일만)
        try:
            if file_path.stat().st_size < 100 * 1024 * 1024:  # 100MB 미만
                with open(file_path, 'rb') as f:
                    header = f.read(1024)
                    
                for framework, info in self.framework_patterns.items():
                    for magic in info['magic_bytes']:
                        if magic in header:
                            return framework
        except:
            pass
        
        return 'unknown'

    def _classify_model_type_advanced(self, file_path: Path) -> str:
        """고급 모델 타입 분류"""
        path_str = str(file_path).lower()
        
        # 디렉토리 구조 기반 분류
        path_parts = [part.lower() for part in file_path.parts]
        
        type_indicators = {
            'diffusion_model': ['diffusion', 'stable', 'ootd', 'unet'],
            'clip_model': ['clip', 'vit', 'vision', 'transformer'],
            'pose_model': ['pose', 'openpose', 'dwpose', 'keypoint'],
            'segmentation_model': ['segment', 'u2net', 'mask', 'sam'],
            'parsing_model': ['parsing', 'human', 'atr', 'schp', 'graphonomy'],
            'warping_model': ['warp', 'tom', 'tps', 'flow', 'matching'],
            'checkpoint': ['checkpoint', 'ckpt', 'epoch', 'step'],
            'config_file': ['config', 'tokenizer', 'vocab']
        }
        
        for model_type, indicators in type_indicators.items():
            for indicator in indicators:
                if any(indicator in part for part in path_parts):
                    return model_type
                if indicator in file_path.name.lower():
                    return model_type
        
        return 'unknown'

    def _classify_step_advanced(self, file_path: Path) -> Tuple[str, float]:
        """MyCloset AI 8단계 고급 분류"""
        path_str = str(file_path).lower()
        name_str = file_path.name.lower()
        parent_str = file_path.parent.name.lower()
        
        best_step = "unknown"
        best_confidence = 0.0
        
        for step_name, step_info in self.step_patterns.items():
            confidence = 0.0
            
            # 패턴 매칭
            for pattern in step_info['patterns']:
                try:
                    if re.search(pattern, path_str):
                        confidence = max(confidence, 0.9)
                    elif re.search(pattern, name_str):
                        confidence = max(confidence, 0.8)
                    elif re.search(pattern, parent_str):
                        confidence = max(confidence, 0.7)
                except re.error:
                    # 정규식 오류 시 문자열 검색
                    clean_pattern = pattern.replace(r'\.*', '').replace('.*', '')
                    if clean_pattern in path_str:
                        confidence = max(confidence, 0.6)
            
            # 모델명 매칭
            for model in step_info['models']:
                if model.lower() in path_str:
                    confidence = max(confidence, 0.85)
            
            # 키워드 매칭
            keyword_matches = sum(1 for kw in step_info['keywords'] if kw in path_str)
            if keyword_matches > 0:
                confidence = max(confidence, 0.5 + (keyword_matches * 0.1))
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_step = step_name
        
        return best_step, best_confidence

    def _classify_architecture(self, file_path: Path) -> str:
        """아키텍처 분류"""
        path_str = str(file_path).lower()
        
        for arch_type, patterns in self.architecture_patterns.items():
            for pattern in patterns:
                if re.search(pattern, path_str):
                    return arch_type
        
        return 'unknown'

    def _check_conda_environment(self, file_path: Path) -> Dict[str, Any]:
        """conda 환경 확인"""
        result = {
            'is_conda': False,
            'env_name': None,
            'env_path': None
        }
        
        for env_name, env_path in self.conda_environments.items():
            try:
                if file_path.is_relative_to(env_path):
                    result['is_conda'] = True
                    result['env_name'] = env_name
                    result['env_path'] = str(env_path)
                    break
            except ValueError:
                continue
        
        return result

    def _is_in_project(self, file_path: Path) -> bool:
        """프로젝트 내부 여부 확인"""
        try:
            return file_path.is_relative_to(self.project_root)
        except ValueError:
            return False

    def _validate_model_advanced(self, file_path: Path, framework: str) -> Dict[str, Any]:
        """고급 모델 검증"""
        result = {
            'is_valid': False,
            'is_complete': False,
            'is_corrupted': False,
            'errors': []
        }
        
        try:
            # 기본 파일 검증
            if not file_path.exists():
                result['errors'].append("파일이 존재하지 않음")
                return result
            
            size = file_path.stat().st_size
            if size == 0:
                result['errors'].append("빈 파일")
                return result
            
            # 프레임워크별 검증
            if framework == 'pytorch':
                result.update(self._validate_pytorch_model(file_path))
            elif framework == 'safetensors':
                result.update(self._validate_safetensors_model(file_path))
            elif framework == 'tensorflow':
                result.update(self._validate_tensorflow_model(file_path))
            elif framework == 'onnx':
                result.update(self._validate_onnx_model(file_path))
            else:
                result['is_valid'] = True
                result['is_complete'] = True
            
        except Exception as e:
            result['errors'].append(f"검증 실패: {e}")
        
        return result

    def _validate_pytorch_model(self, file_path: Path) -> Dict[str, Any]:
        """PyTorch 모델 검증"""
        result = {'is_valid': False, 'is_complete': False, 'is_corrupted': False, 'errors': []}
        
        try:
            with open(file_path, 'rb') as f:
                header = f.read(1024)
            
            # PyTorch 매직 바이트 확인
            if b'PK' in header or b'\x80' in header:
                result['is_valid'] = True
                result['is_complete'] = True
            else:
                result['errors'].append("PyTorch 형식이 아님")
                
        except Exception as e:
            result['errors'].append(f"PyTorch 검증 실패: {e}")
            result['is_corrupted'] = True
        
        return result

    def _validate_safetensors_model(self, file_path: Path) -> Dict[str, Any]:
        """Safetensors 모델 검증"""
        result = {'is_valid': False, 'is_complete': False, 'is_corrupted': False, 'errors': []}
        
        try:
            with open(file_path, 'rb') as f:
                header = f.read(1024).decode('utf-8', errors='ignore')
            
            if '{' in header and '"' in header:
                result['is_valid'] = True
                result['is_complete'] = True
            else:
                result['errors'].append("Safetensors 형식이 아님")
                
        except Exception as e:
            result['errors'].append(f"Safetensors 검증 실패: {e}")
            result['is_corrupted'] = True
        
        return result

    def _validate_tensorflow_model(self, file_path: Path) -> Dict[str, Any]:
        """TensorFlow 모델 검증"""
        result = {'is_valid': False, 'is_complete': False, 'is_corrupted': False, 'errors': []}
        
        try:
            extension = file_path.suffix.lower()
            
            with open(file_path, 'rb') as f:
                header = f.read(100)
            
            if extension == '.h5' and header.startswith(b'\x89HDF'):
                result['is_valid'] = True
                result['is_complete'] = True
            elif extension == '.pb' and len(header) > 10:
                result['is_valid'] = True
                result['is_complete'] = True
            else:
                result['errors'].append("TensorFlow 형식 확인 실패")
                
        except Exception as e:
            result['errors'].append(f"TensorFlow 검증 실패: {e}")
            result['is_corrupted'] = True
        
        return result

    def _validate_onnx_model(self, file_path: Path) -> Dict[str, Any]:
        """ONNX 모델 검증"""
        result = {'is_valid': False, 'is_complete': False, 'is_corrupted': False, 'errors': []}
        
        try:
            with open(file_path, 'rb') as f:
                header = f.read(100)
            
            if b'onnx' in header.lower() or len(header) > 50:
                result['is_valid'] = True
                result['is_complete'] = True
            else:
                result['errors'].append("ONNX 형식이 아님")
                
        except Exception as e:
            result['errors'].append(f"ONNX 검증 실패: {e}")
            result['is_corrupted'] = True
        
        return result

    def _extract_metadata_advanced(self, file_path: Path, framework: str) -> ModelMetadata:
        """고급 메타데이터 추출"""
        metadata = ModelMetadata()
        
        try:
            # 동반 파일에서 메타데이터 추출
            parent_dir = file_path.parent
            
            # config.json 확인
            config_path = parent_dir / "config.json"
            if config_path.exists():
                metadata.has_config = True
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    
                    metadata.architecture = config_data.get('model_type', 'unknown')
                    metadata.framework_version = config_data.get('transformers_version', 'unknown')
                    
                    if 'base_model' in config_data:
                        metadata.base_model = config_data['base_model']
                        metadata.is_fine_tuned = True
                    
                except:
                    pass
            
            # tokenizer 확인
            tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'vocab.txt']
            metadata.has_tokenizer = any((parent_dir / tf).exists() for tf in tokenizer_files)
            
            # 파일명에서 정보 추출
            name_lower = file_path.name.lower()
            
            # 정밀도 추출
            if 'fp16' in name_lower:
                metadata.precision = 'fp16'
            elif 'fp32' in name_lower:
                metadata.precision = 'fp32'
            elif 'int8' in name_lower:
                metadata.precision = 'int8'
            
            # 태그 추출
            tags = []
            if 'fine' in name_lower and 'tuned' in name_lower:
                tags.append('fine-tuned')
            if 'checkpoint' in name_lower:
                tags.append('checkpoint')
            if 'epoch' in name_lower:
                tags.append('training')
            
            metadata.tags = tags
            
        except Exception as e:
            logger.warning(f"메타데이터 추출 실패 {file_path}: {e}")
        
        return metadata

    def _find_companion_files(self, file_path: Path) -> List[str]:
        """동반 파일 찾기"""
        companions = []
        parent_dir = file_path.parent
        
        companion_patterns = [
            'config.json', 'config.yaml', 'model_config.json',
            'tokenizer.json', 'tokenizer_config.json', 'vocab.txt',
            'pytorch_model.bin', 'model.safetensors',
            'scheduler_config.json', 'unet_config.json'
        ]
        
        for pattern in companion_patterns:
            companion_path = parent_dir / pattern
            if companion_path.exists() and companion_path != file_path:
                companions.append(str(companion_path))
        
        return companions

    def _calculate_importance_score(
        self, file_path: Path, size_gb: float, framework: str, 
        step_candidate: str, confidence: float
    ) -> float:
        """중요도 점수 계산"""
        score = 0.0
        
        # 기본 점수 (크기 기반)
        score += min(size_gb * 10, 50)  # 최대 50점
        
        # 신뢰도 점수
        score += confidence * 30  # 최대 30점
        
        # 프레임워크 보너스
        framework_bonus = {
            'pytorch': 10, 'safetensors': 8, 'diffusers': 15,
            'transformers': 12, 'onnx': 5
        }
        score += framework_bonus.get(framework, 0)
        
        # Step 보너스 (MyCloset AI 특화)
        if step_candidate != 'unknown':
            score += 20
        
        # 프로젝트 내부 보너스
        if self._is_in_project(file_path):
            score += 15
        
        # conda 환경 보너스
        if any(file_path.is_relative_to(env_path) 
               for env_path in self.conda_environments.values()):
            score += 10
        
        return min(score, 100.0)  # 최대 100점

    def _post_process_results(self):
        """고급 후처리"""
        logger.info("🔄 결과 후처리 중...")
        
        # 1. 중복 탐지
        self._detect_duplicates_advanced()
        
        # 2. 관련 모델 연결
        self._link_related_models()
        
        # 3. 중요도 재계산
        self._recalculate_importance()
        
        # 4. 데이터베이스 업데이트
        self._update_database()

    def _detect_duplicates_advanced(self):
        """고급 중복 탐지"""
        # MD5 기반 그룹화
        md5_groups = defaultdict(list)
        for model in self.found_models:
            if model.checksum_md5 != 'unknown':
                md5_groups[model.checksum_md5].append(model)
        
        # 중복 그룹 처리
        for checksum, models in md5_groups.items():
            if len(models) > 1:
                # 가장 중요한 모델을 원본으로 설정
                primary = max(models, key=lambda m: m.importance_score)
                
                for model in models:
                    if model != primary:
                        model.duplicate_of = primary.path
                
                self.duplicates[checksum] = models

    def _link_related_models(self):
        """관련 모델 연결"""
        # 같은 디렉토리의 모델들 연결
        dir_groups = defaultdict(list)
        for model in self.found_models:
            dir_groups[model.parent_directory].append(model)
        
        for models in dir_groups.values():
            if len(models) > 1:
                for model in models:
                    model.related_models = [
                        m.path for m in models if m != model
                    ]

    def _recalculate_importance(self):
        """중요도 재계산 (관계성 고려)"""
        for model in self.found_models:
            bonus = 0
            
            # 동반 파일 보너스
            if model.companion_files:
                bonus += len(model.companion_files) * 2
            
            # 관련 모델 보너스
            if model.related_models:
                bonus += len(model.related_models)
            
            # 최신성 보너스
            days_old = (datetime.now() - model.modified_time).days
            if days_old < 30:
                bonus += 5
            elif days_old < 90:
                bonus += 3
            
            model.importance_score = min(model.importance_score + bonus, 100.0)

    def _update_database(self):
        """데이터베이스 업데이트"""
        cursor = self.db_connection.cursor()
        
        for model in self.found_models:
            cursor.execute('''
                INSERT OR REPLACE INTO model_scans 
                (scan_date, model_path, model_name, size_gb, framework, 
                 step_candidate, confidence, checksum, is_valid, metadata, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                model.path,
                model.name,
                model.size_gb,
                model.framework,
                model.step_candidate,
                model.confidence,
                model.checksum_md5,
                model.is_valid,
                json.dumps(asdict(model.metadata)),
                datetime.now().isoformat()
            ))
        
        self.db_connection.commit()

    def _auto_organize_models(self):
        """자동 모델 정리"""
        logger.info("📁 자동 모델 정리 시작...")
        
        # 대상 디렉토리 생성
        organized_dir = self.project_root / "backend" / "ai_models" / "organized"
        organized_dir.mkdir(parents=True, exist_ok=True)
        
        # Step별 디렉토리 생성
        for step_name in self.step_patterns.keys():
            step_dir = organized_dir / step_name.replace('step_', '').replace('_', '-')
            step_dir.mkdir(exist_ok=True)
        
        # 모델 이동 (신뢰도 높은 것만)
        moved_count = 0
        for model in self.found_models:
            if (model.confidence > 0.7 and 
                not model.is_in_project and 
                model.step_candidate != 'unknown'):
                
                try:
                    step_name = model.step_candidate.replace('step_', '').replace('_', '-')
                    target_dir = organized_dir / step_name
                    target_path = target_dir / model.name
                    
                    if not target_path.exists():
                        shutil.copy2(model.path, target_path)
                        logger.info(f"✅ 이동: {model.name} → {step_name}")
                        moved_count += 1
                        
                except Exception as e:
                    logger.warning(f"⚠️ 이동 실패 {model.name}: {e}")
        
        logger.info(f"📦 {moved_count}개 모델 정리 완료")

    def _save_scan_results(self, scan_duration: float):
        """스캔 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 통계 계산
        stats = self._calculate_statistics(scan_duration)
        
        # JSON 결과 저장
        result_data = {
            "scan_info": {
                "timestamp": datetime.now().isoformat(),
                "duration": scan_duration,
                "system": platform.system(),
                "machine": platform.machine(),
                "python_version": platform.python_version(),
                "conda_environments": list(self.conda_environments.keys()),
                "current_conda_env": self.current_conda_env
            },
            "statistics": asdict(stats),
            "models": [asdict(model) for model in self.found_models],
            "duplicates": {
                checksum: [asdict(model) for model in models]
                for checksum, models in self.duplicates.items()
            },
            "scan_locations": self.scan_locations,
            "errors": self.errors,
            "warnings": self.warnings
        }
        
        # 파일 저장
        output_file = self.project_root / f"model_scan_complete_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"💾 스캔 결과 저장: {output_file}")
        
        # 데이터베이스에 스캔 이력 저장
        cursor = self.db_connection.cursor()
        cursor.execute('''
            INSERT INTO scan_history 
            (scan_date, total_models, total_size_gb, scan_duration, statistics)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            len(self.found_models),
            stats.total_size_gb,
            scan_duration,
            json.dumps(asdict(stats))
        ))
        self.db_connection.commit()

    def _calculate_statistics(self, scan_duration: float) -> ScanStatistics:
        """상세 통계 계산"""
        stats = ScanStatistics()
        
        # 기본 통계
        stats.total_files_scanned = sum(len(files) for files in self.scan_locations.values())
        stats.models_found = len(self.found_models)
        stats.total_size_bytes = sum(model.size_bytes for model in self.found_models)
        stats.total_size_gb = stats.total_size_bytes / (1024**3)
        stats.scan_duration = scan_duration
        
        # 위치 통계
        stats.locations_scanned = len(self.scan_locations)
        stats.conda_models = sum(1 for model in self.found_models if model.is_in_conda)
        stats.project_models = sum(1 for model in self.found_models if model.is_in_project)
        stats.system_models = stats.models_found - stats.conda_models - stats.project_models
        
        # 품질 통계
        stats.valid_models = sum(1 for model in self.found_models if model.is_valid)
        stats.corrupted_models = sum(1 for model in self.found_models if model.is_corrupted)
        stats.duplicate_groups = len(self.duplicates)
        stats.unique_models = stats.models_found - sum(
            len(models) - 1 for models in self.duplicates.values()
        )
        
        # 분포 통계
        for model in self.found_models:
            # 프레임워크 분포
            fw = model.framework
            stats.framework_distribution[fw] = stats.framework_distribution.get(fw, 0) + 1
            
            # 타입 분포
            mt = model.model_type
            stats.type_distribution[mt] = stats.type_distribution.get(mt, 0) + 1
            
            # Step 분포 (신뢰도 0.5+ 만)
            if model.confidence >= 0.5:
                step = model.step_candidate
                stats.step_distribution[step] = stats.step_distribution.get(step, 0) + 1
        
        # 성능 통계
        stats.errors_count = len(self.errors)
        stats.warnings_count = len(self.warnings)
        if scan_duration > 0:
            stats.processing_speed_files_per_sec = stats.total_files_scanned / scan_duration
        
        return stats

    def _print_comprehensive_report(self, scan_duration: float):
        """완전한 스캔 보고서 출력"""
        stats = self._calculate_statistics(scan_duration)
        
        print("\n" + "=" * 100)
        print("🎯 MyCloset AI - 완전 고도화된 AI 모델 스캔 결과")
        print("=" * 100)
        
        # 기본 정보
        print(f"🕐 스캔 시간: {scan_duration:.1f}초 ({stats.processing_speed_files_per_sec:.1f} 파일/초)")
        print(f"💻 시스템: {platform.system()} {platform.machine()}")
        print(f"🐍 Python: {platform.python_version()}")
        print(f"🔧 워커: {self.max_workers}개 병렬 처리")
        print(f"🐍 conda 환경: {self.current_conda_env or 'None'}")
        
        # 스캔 통계
        print(f"\n📊 스캔 통계:")
        print(f"   📁 스캔 위치: {stats.locations_scanned}곳")
        print(f"   📄 검사 파일: {stats.total_files_scanned:,}개")
        print(f"   🤖 발견 모델: {stats.models_found:,}개")
        print(f"   💾 총 용량: {stats.total_size_gb:.2f}GB")
        print(f"   ✅ 유효 모델: {stats.valid_models}개")
        print(f"   ❌ 손상 모델: {stats.corrupted_models}개")
        
        if not self.found_models:
            print("\n❌ AI 모델을 찾을 수 없습니다.")
            self._print_suggestions()
            return
        
        # 위치별 분포
        print(f"\n📍 위치별 분포:")
        print(f"   🐍 conda 환경: {stats.conda_models}개")
        print(f"   🏠 프로젝트 내부: {stats.project_models}개") 
        print(f"   🌍 시스템 전체: {stats.system_models}개")
        
        # conda 환경별 상세
        if stats.conda_models > 0:
            conda_dist = defaultdict(int)
            for model in self.found_models:
                if model.is_in_conda and model.conda_env_name:
                    conda_dist[model.conda_env_name] += 1
            
            print(f"   conda 환경별:")
            for env_name, count in sorted(conda_dist.items()):
                env_size = sum(m.size_gb for m in self.found_models 
                             if m.conda_env_name == env_name)
                print(f"     - {env_name}: {count}개 ({env_size:.1f}GB)")
        
        # 프레임워크 분포
        print(f"\n🔧 프레임워크별 분포:")
        for fw, count in sorted(stats.framework_distribution.items(), 
                               key=lambda x: x[1], reverse=True):
            fw_size = sum(m.size_gb for m in self.found_models if m.framework == fw)
            percentage = (count / stats.models_found) * 100
            print(f"   - {fw}: {count}개 ({fw_size:.2f}GB, {percentage:.1f}%)")
        
        # MyCloset AI Step 분포
        if stats.step_distribution:
            print(f"\n🎯 MyCloset AI Step별 분포 (신뢰도 50%+):")
            step_names = {
                'step_01_human_parsing': '1️⃣ Human Parsing',
                'step_02_pose_estimation': '2️⃣ Pose Estimation',
                'step_03_cloth_segmentation': '3️⃣ Cloth Segmentation', 
                'step_04_geometric_matching': '4️⃣ Geometric Matching',
                'step_05_cloth_warping': '5️⃣ Cloth Warping',
                'step_06_virtual_fitting': '6️⃣ Virtual Fitting',
                'step_07_post_processing': '7️⃣ Post Processing',
                'step_08_quality_assessment': '8️⃣ Quality Assessment'
            }
            
            total_classified = sum(stats.step_distribution.values())
            for step, count in sorted(stats.step_distribution.items()):
                if count > 0:
                    display_name = step_names.get(step, step)
                    step_size = sum(m.size_gb for m in self.found_models 
                                  if m.step_candidate == step and m.confidence >= 0.5)
                    percentage = (count / total_classified) * 100
                    print(f"   {display_name}: {count}개 ({step_size:.1f}GB, {percentage:.1f}%)")
        
        # 중복 파일 정보
        if self.duplicates:
            duplicate_count = len(self.duplicates)
            total_duplicates = sum(len(models) for models in self.duplicates.values())
            waste_size = sum(
                sum(m.size_gb for m in models[1:])  # 첫 번째 제외한 나머지
                for models in self.duplicates.values()
            )
            print(f"\n🔄 중복 파일 분석:")
            print(f"   중복 그룹: {duplicate_count}개")
            print(f"   중복 파일: {total_duplicates - duplicate_count}개")
            print(f"   절약 가능: {waste_size:.2f}GB")
            
            if duplicate_count <= 5:  # 5개 이하면 상세 표시
                for i, (checksum, models) in enumerate(self.duplicates.items(), 1):
                    print(f"   그룹 {i}: {len(models)}개 파일")
                    for j, model in enumerate(models):
                        marker = "🏆" if j == 0 else "📄"
                        location = "conda" if model.is_in_conda else "project" if model.is_in_project else "system"
                        print(f"     {marker} {model.name} ({location}, {model.size_gb:.1f}GB)")
        
        # 상위 중요 모델들
        print(f"\n🏆 중요도 상위 모델들:")
        top_models = sorted(self.found_models, key=lambda x: x.importance_score, reverse=True)[:10]
        
        for i, model in enumerate(top_models, 1):
            location_icon = "🐍" if model.is_in_conda else "🏠" if model.is_in_project else "🌍"
            step_info = ""
            if model.confidence >= 0.5 and model.step_candidate != 'unknown':
                step_num = model.step_candidate.split('_')[1] if '_' in model.step_candidate else '?'
                step_info = f" | 🎯 Step {step_num}"
            
            confidence_icon = "🟢" if model.confidence >= 0.8 else "🟡" if model.confidence >= 0.5 else "🔴"
            
            print(f"  {i:2d}. {model.name}")
            print(f"      📍 {model.path}")
            print(f"      📊 {model.size_gb:.2f}GB | {model.framework} | {model.model_type}")
            print(f"      {location_icon} {model.conda_env_name or model.parent_directory} | "
                  f"{confidence_icon} {model.confidence:.2f} | ⭐ {model.importance_score:.1f}{step_info}")
        
        # 문제 및 권장사항
        print(f"\n💡 권장사항:")
        recommendations = []
        
        if stats.conda_models > 0 and stats.project_models == 0:
            recommendations.append(f"🔄 conda 환경의 {stats.conda_models}개 모델을 프로젝트로 연결 고려")
        
        if stats.system_models > stats.project_models:
            recommendations.append(f"📦 시스템의 {stats.system_models}개 모델을 프로젝트로 통합 고려")
        
        if stats.duplicate_groups > 0:
            waste_gb = sum(sum(m.size_gb for m in models[1:]) for models in self.duplicates.values())
            recommendations.append(f"🗑️ 중복 파일 {stats.duplicate_groups}그룹 정리로 {waste_gb:.1f}GB 절약 가능")
        
        if stats.corrupted_models > 0:
            recommendations.append(f"🔧 손상된 {stats.corrupted_models}개 모델 복구 또는 제거")
        
        large_models = [m for m in self.found_models if m.size_gb > 2.0]
        if large_models:
            recommendations.append(f"📦 2GB+ 대용량 모델 {len(large_models)}개 최적화 검토")
        
        unclassified = [m for m in self.found_models if m.step_candidate == 'unknown']
        if unclassified:
            recommendations.append(f"🎯 미분류 모델 {len(unclassified)}개 수동 검토 필요")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        if not recommendations:
            print("   ✅ 현재 모델 구성이 최적화되어 있습니다!")
        
        # 오류 및 경고
        if self.errors or self.warnings:
            print(f"\n⚠️ 발견된 문제들:")
            if self.errors:
                print(f"   ❌ 오류 {len(self.errors)}개:")
                for error in self.errors[:3]:
                    print(f"     - {error}")
                if len(self.errors) > 3:
                    print(f"     ... 외 {len(self.errors) - 3}개")
            
            if self.warnings:
                print(f"   ⚠️ 경고 {len(self.warnings)}개:")
                for warning in self.warnings[:3]:
                    print(f"     - {warning}")
                if len(self.warnings) > 3:
                    print(f"     ... 외 {len(self.warnings) - 3}개")
        
        # 다음 단계 안내
        print(f"\n🚀 다음 단계:")
        print("1. python advanced_scanner.py --organize     # 자동 모델 정리")
        print("2. python advanced_scanner.py --repair       # 손상된 모델 복구")
        print("3. 스캔 결과 JSON 파일 확인 및 활용")
        print("4. 중복 파일 정리 및 공간 최적화")

    def _print_suggestions(self):
        """모델을 찾을 수 없을 때 제안사항"""
        print("\n💡 AI 모델을 찾을 수 없습니다. 다음을 확인해보세요:")
        print("\n🔍 검색 확장:")
        print("   python advanced_scanner.py --deep                 # 전체 시스템 딥 스캔")
        print("   python advanced_scanner.py --conda-first          # conda 환경 우선 스캔")
        print("   python advanced_scanner.py --deep --organize      # 딥 스캔 + 자동 정리")
        
        print("\n📁 일반적인 AI 모델 위치:")
        for env_name, env_path in self.conda_environments.items():
            print(f"   🐍 {env_name}: {env_path}/lib/python*/site-packages")
        
        print("   🏠 프로젝트: ./backend/ai_models/")
        print("   📥 다운로드: ~/Downloads/")
        print("   💾 캐시: ~/.cache/huggingface/, ~/.cache/torch/")
        
        print("\n⚙️ 설정 확인:")
        print(f"   최소 크기: {self.config.min_size_mb}MB")
        print(f"   최대 크기: {self.config.max_size_gb}GB")
        print(f"   스캔 깊이: {self.config.max_depth}단계")

    def repair_corrupted_models(self) -> int:
        """손상된 모델 복구 시도"""
        logger.info("🔧 손상된 모델 복구 시작...")
        
        corrupted_models = [m for m in self.found_models if m.is_corrupted]
        if not corrupted_models:
            logger.info("✅ 손상된 모델이 없습니다.")
            return 0
        
        repaired_count = 0
        
        for model in corrupted_models:
            try:
                # 백업 생성
                backup_path = Path(model.path + '.backup')
                if not backup_path.exists():
                    shutil.copy2(model.path, backup_path)
                
                # 복구 시도 (프레임워크별)
                if model.framework == 'pytorch':
                    if self._repair_pytorch_model(Path(model.path)):
                        repaired_count += 1
                        logger.info(f"✅ 복구 성공: {model.name}")
                elif model.framework == 'safetensors':
                    if self._repair_safetensors_model(Path(model.path)):
                        repaired_count += 1
                        logger.info(f"✅ 복구 성공: {model.name}")
                
            except Exception as e:
                logger.warning(f"⚠️ 복구 실패 {model.name}: {e}")
        
        logger.info(f"🎯 총 {repaired_count}/{len(corrupted_models)}개 모델 복구 완료")
        return repaired_count

    def _repair_pytorch_model(self, file_path: Path) -> bool:
        """PyTorch 모델 복구"""
        try:
            # 간단한 헤더 복구 시도
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # 매직 바이트 확인 및 복구
            if not data.startswith(b'PK') and b'PK' in data:
                # PK 헤더를 찾아서 앞으로 이동
                pk_index = data.find(b'PK')
                if pk_index > 0:
                    repaired_data = data[pk_index:]
                    
                    with open(file_path, 'wb') as f:
                        f.write(repaired_data)
                    
                    return True
            
        except Exception:
            pass
        
        return False

    def _repair_safetensors_model(self, file_path: Path) -> bool:
        """Safetensors 모델 복구"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # JSON 헤더 찾기
            if data.startswith(b'{'):
                return True  # 이미 올바름
            
            json_start = data.find(b'{')
            if json_start > 0:
                repaired_data = data[json_start:]
                
                with open(file_path, 'wb') as f:
                    f.write(repaired_data)
                
                return True
            
        except Exception:
            pass
        
        return False

    def generate_conda_config(self, output_file: str = None) -> str:
        """conda 환경 우선 설정 파일 생성"""
        if output_file is None:
            output_file = "conda_model_config.py"
        
        config_content = f'''#!/usr/bin/env python3
"""
🐍 MyCloset AI - Conda 환경 우선 모델 설정
==========================================

자동 생성됨: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
스캔된 모델: {len(self.found_models)}개
conda 환경: {len(self.conda_environments)}개

사용법:
    from conda_model_config import get_model_path, get_conda_models
    
    # 특정 모델 경로 가져오기
    clip_path = get_model_path("clip")
    
    # conda 환경의 모든 모델
    conda_models = get_conda_models()
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

# ==============================================
# 🐍 Conda 환경 정보
# ==============================================

CONDA_ENVIRONMENTS = {{
{self._format_conda_envs_for_config()}
}}

CURRENT_CONDA_ENV = "{self.current_conda_env or 'None'}"

# ==============================================
# 🤖 발견된 모델 경로들
# ==============================================

MODEL_PATHS = {{
{self._format_model_paths_for_config()}
}}

# MyCloset AI 8단계별 모델 매핑
STEP_MODELS = {{
{self._format_step_models_for_config()}
}}

# 프레임워크별 모델 그룹
FRAMEWORK_MODELS = {{
{self._format_framework_models_for_config()}
}}

# ==============================================
# 🔧 유틸리티 함수들
# ==============================================

def get_model_path(model_name: str, prefer_conda: bool = True) -> Optional[str]:
    """모델 경로 가져오기 (conda 우선)"""
    candidates = []
    
    # 모델명으로 직접 검색
    for key, path in MODEL_PATHS.items():
        if model_name.lower() in key.lower():
            candidates.append((key, path))
    
    if not candidates:
        return None
    
    # conda 환경 우선 정렬
    if prefer_conda:
        conda_models = [c for c in candidates if any(env in c[1] for env in CONDA_ENVIRONMENTS.values())]
        if conda_models:
            return conda_models[0][1]
    
    return candidates[0][1]

def get_conda_models() -> Dict[str, List[str]]:
    """conda 환경별 모델 목록"""
    result = {{}}
    
    for env_name, env_path in CONDA_ENVIRONMENTS.items():
        env_models = []
        for model_name, model_path in MODEL_PATHS.items():
            if env_path in model_path:
                env_models.append(model_path)
        result[env_name] = env_models
    
    return result

def get_step_model(step_number: int, prefer_conda: bool = True) -> Optional[str]:
    """Step별 최적 모델 경로"""
    step_key = f"step_{{step_number:02d}}"
    
    for key, models in STEP_MODELS.items():
        if step_key in key and models:
            if prefer_conda:
                # conda 환경의 모델 우선
                conda_models = [m for m in models if any(env in m for env in CONDA_ENVIRONMENTS.values())]
                if conda_models:
                    return conda_models[0]
            return models[0]
    
    return None

def get_framework_models(framework: str) -> List[str]:
    """프레임워크별 모델 목록"""
    return FRAMEWORK_MODELS.get(framework, [])

def validate_model_availability() -> Dict[str, bool]:
    """모델 가용성 검증"""
    result = {{}}
    
    for model_name, model_path in MODEL_PATHS.items():
        result[model_name] = Path(model_path).exists()
    
    return result

def get_model_info(model_path: str) -> Dict[str, any]:
    """모델 상세 정보"""
    path_obj = Path(model_path)
    if not path_obj.exists():
        return {{"error": "파일이 존재하지 않음"}}
    
    stat_info = path_obj.stat()
    return {{
        "name": path_obj.name,
        "size_mb": round(stat_info.st_size / (1024 * 1024), 2),
        "modified": stat_info.st_mtime,
        "is_conda": any(env in model_path for env in CONDA_ENVIRONMENTS.values()),
        "framework": _detect_framework(path_obj)
    }}

def _detect_framework(path: Path) -> str:
    """프레임워크 감지"""
    ext = path.suffix.lower()
    if ext in ['.pth', '.pt']:
        return 'pytorch'
    elif ext == '.safetensors':
        return 'safetensors'
    elif ext in ['.pb', '.h5']:
        return 'tensorflow'
    elif ext == '.onnx':
        return 'onnx'
    return 'unknown'

# ==============================================
# 🚀 Quick Start 예제
# ==============================================

if __name__ == "__main__":
    print("🐍 MyCloset AI Conda 모델 설정")
    print("=" * 50)
    
    print(f"conda 환경: {{len(CONDA_ENVIRONMENTS)}}개")
    print(f"발견된 모델: {{len(MODEL_PATHS)}}개")
    
    # conda 환경별 모델 수
    conda_models = get_conda_models()
    for env_name, models in conda_models.items():
        print(f"  {{env_name}}: {{len(models)}}개 모델")
    
    # Step별 모델 확인
    print("\\nStep별 모델:")
    for i in range(1, 9):
        model_path = get_step_model(i)
        if model_path:
            print(f"  Step {{i:02d}}: {{Path(model_path).name}}")
        else:
            print(f"  Step {{i:02d}}: 없음")
    
    # 가용성 검증
    availability = validate_model_availability()
    available_count = sum(availability.values())
    print(f"\\n가용 모델: {{available_count}}/{{len(MODEL_PATHS)}}개")
'''
        
        config_path = self.project_root / output_file
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        logger.info(f"📝 conda 설정 파일 생성: {config_path}")
        return str(config_path)

    def _format_conda_envs_for_config(self) -> str:
        """conda 환경 설정 형식"""
        lines = []
        for env_name, env_path in self.conda_environments.items():
            lines.append(f'    "{env_name}": "{env_path}",')
        return '\n'.join(lines)

    def _format_model_paths_for_config(self) -> str:
        """모델 경로 설정 형식"""
        lines = []
        for model in sorted(self.found_models, key=lambda x: x.importance_score, reverse=True):
            safe_name = model.name.replace('.', '_').replace('-', '_')
            lines.append(f'    "{safe_name}": "{model.path}",')
        return '\n'.join(lines)

    def _format_step_models_for_config(self) -> str:
        """Step별 모델 설정 형식"""
        lines = []
        step_models = defaultdict(list)
        
        for model in self.found_models:
            if model.confidence >= 0.5 and model.step_candidate != 'unknown':
                step_models[model.step_candidate].append(model.path)
        
        for step_name in sorted(step_models.keys()):
            models = step_models[step_name]
            models_str = ', '.join(f'"{path}"' for path in models)
            lines.append(f'    "{step_name}": [{models_str}],')
        
        return '\n'.join(lines)

    def _format_framework_models_for_config(self) -> str:
        """프레임워크별 모델 설정 형식"""
        lines = []
        framework_models = defaultdict(list)
        
        for model in self.found_models:
            framework_models[model.framework].append(model.path)
        
        for framework in sorted(framework_models.keys()):
            models = framework_models[framework]
            models_str = ', '.join(f'"{path}"' for path in models)
            lines.append(f'    "{framework}": [{models_str}],')
        
        return '\n'.join(lines)

    def cleanup_and_close(self):
        """리소스 정리 및 종료"""
        if hasattr(self, 'db_connection'):
            self.db_connection.close()

# ==============================================
# 🚀 CLI 인터페이스 및 메인 함수
# ==============================================

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="완전 고도화된 AI 모델 체크포인트 검색 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 사용 예시:
  python advanced_scanner.py                           # 표준 스캔
  python advanced_scanner.py --deep                    # 딥 스캔
  python advanced_scanner.py --conda-first             # conda 우선 스캔
  python advanced_scanner.py --organize                # 스캔 + 자동 정리
  python advanced_scanner.py --repair                  # 손상된 모델 복구
  python advanced_scanner.py --deep --organize --repair # 완전 자동화

🐍 conda 환경 최적화:
  python advanced_scanner.py --conda-first --generate-config
        """
    )
    
    # 스캔 옵션
    parser.add_argument('--deep', action='store_true', 
                       help='전체 시스템 딥 스캔 (더 많은 위치 검색)')
    parser.add_argument('--conda-first', action='store_true',
                       help='conda 환경 우선 스캔')
    parser.add_argument('--organize', action='store_true',
                       help='스캔 후 자동 모델 정리')
    parser.add_argument('--repair', action='store_true',
                       help='손상된 모델 복구 시도')
    
    # 설정 옵션
    parser.add_argument('--min-size', type=float, default=0.1,
                       help='최소 파일 크기 (MB, 기본: 0.1)')
    parser.add_argument('--max-size', type=float, default=100.0,
                       help='최대 파일 크기 (GB, 기본: 100.0)')
    parser.add_argument('--max-depth', type=int, default=10,
                       help='최대 스캔 깊이 (기본: 10)')
    parser.add_argument('--workers', type=int, default=None,
                       help='병렬 워커 수 (기본: CPU 코어 수)')
    
    # 출력 옵션
    parser.add_argument('--generate-config', action='store_true',
                       help='conda 설정 파일 생성')
    parser.add_argument('--quiet', action='store_true',
                       help='조용한 모드 (최소 출력)')
    parser.add_argument('--output', type=str,
                       help='결과 저장 파일명')
    
    args = parser.parse_args()
    
    try:
        # 설정 구성
        config = ScanConfig(
            min_size_mb=args.min_size,
            max_size_gb=args.max_size,
            max_depth=args.max_depth,
            conda_priority=args.conda_first,
            deep_scan=args.deep,
            verify_integrity=True,
            extract_metadata=True
        )
        
        # 스캐너 초기화
        scanner = AdvancedModelScanner(config)
        
        if args.workers:
            scanner.max_workers = min(args.workers, scanner.cpu_count)
        
        # 스캔 실행
        models = scanner.scan_comprehensive_system(organize=args.organize)
        
        # 복구 작업 (옵션)
        if args.repair and models:
            repaired = scanner.repair_corrupted_models()
            if repaired > 0:
                logger.info(f"🔧 {repaired}개 모델 복구 완료")
        
        # conda 설정 생성 (옵션)
        if args.generate_config and models:
            config_file = scanner.generate_conda_config()
            print(f"📝 conda 설정 파일 생성: {config_file}")
        
        # 완료 메시지
        if not args.quiet:
            print(f"\n✅ 스캔 완료!")
            print(f"🤖 발견된 모델: {len(models)}개")
            print(f"🐍 conda 모델: {sum(1 for m in models if m.is_in_conda)}개")
            print(f"🏠 프로젝트 모델: {sum(1 for m in models if m.is_in_project)}개")
            
            if models:
                total_size = sum(m.size_gb for m in models)
                avg_importance = sum(m.importance_score for m in models) / len(models)
                print(f"💾 총 용량: {total_size:.2f}GB")
                print(f"⭐ 평균 중요도: {avg_importance:.1f}/100")
        
        # 정리
        scanner.cleanup_and_close()
        
        return 0 if models else 1
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())