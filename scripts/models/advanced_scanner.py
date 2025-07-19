#!/usr/bin/env python3
"""
🔥 완전한 AI 모델 체크포인트 검색 스크립트 - 수정된 버전
=======================================================

MyCloset AI 프로젝트에 특화된 완전한 모델 스캐너
- 실제 프로젝트 구조에 맞춰 정확한 경로 탐지
- conda 환경 우선 검색
- MyCloset AI 8단계 자동 분류
- 완전한 보고서 및 설정 파일 생성

사용법:
    python quick_scanner.py                    # 기본 스캔
    python quick_scanner.py --verbose          # 상세 출력
    python quick_scanner.py --organize         # 스캔 + 설정 생성
    python quick_scanner.py --deep             # 딥 스캔
    python quick_scanner.py --conda-first      # conda 우선
"""

import os
import sys
import json
import shutil
import time
import hashlib
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import re
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """완전한 모델 정보"""
    name: str
    path: str
    absolute_path: str
    size_mb: float
    size_gb: float
    framework: str
    model_type: str
    step_candidate: str
    confidence: float
    is_valid: bool
    is_in_project: bool
    is_in_conda: bool
    conda_env_name: Optional[str]
    parent_directory: str
    created_time: str
    modified_time: str
    checksum: str
    companion_files: List[str]
    importance_score: float
    extension: str

@dataclass 
class ScanStatistics:
    """스캔 통계"""
    total_files_scanned: int = 0
    models_found: int = 0
    total_size_gb: float = 0.0
    scan_duration: float = 0.0
    conda_models: int = 0
    project_models: int = 0
    system_models: int = 0
    valid_models: int = 0
    framework_distribution: Dict[str, int] = None
    step_distribution: Dict[str, int] = None
    
    def __post_init__(self):
        if self.framework_distribution is None:
            self.framework_distribution = {}
        if self.step_distribution is None:
            self.step_distribution = {}

class CompleteModelScanner:
    """완전한 AI 모델 스캐너"""
    
    def __init__(self, verbose: bool = True, conda_first: bool = False, deep_scan: bool = False):
        self.verbose = verbose
        self.conda_first = conda_first
        self.deep_scan = deep_scan
        self.scan_start_time = time.time()
        
        # 현재 위치 및 프로젝트 구조 파악
        self.current_dir = Path.cwd()
        self.project_root = self._find_project_root()
        self.ai_models_dir = self._find_ai_models_dir()
        
        # 결과 저장
        self.found_models: List[ModelInfo] = []
        self.scan_locations: Dict[str, List[str]] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
        # conda 환경 탐지
        self.conda_environments = self._detect_conda_environments()
        self.current_conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        
        # 모델 분류 패턴 초기화
        self._init_classification_patterns()
        
        logger.info(f"🚀 CompleteModelScanner 초기화 완료")
        logger.info(f"📁 프로젝트 루트: {self.project_root}")
        logger.info(f"🤖 AI 모델 디렉토리: {self.ai_models_dir}")
        logger.info(f"🐍 conda 환경: {len(self.conda_environments)}개 발견")
        
    def _find_project_root(self) -> Path:
        """프로젝트 루트 정확히 찾기"""
        current = self.current_dir
        
        # 현재 디렉토리가 mycloset-ai인 경우
        if current.name == 'mycloset-ai':
            return current
        
        # backend 디렉토리에서 실행된 경우
        if current.name == 'backend':
            if (current.parent / 'frontend').exists():
                return current.parent
            return current
        
        # 부모 디렉토리들 검사
        for parent in current.parents:
            if parent.name == 'mycloset-ai':
                return parent
            # backend와 frontend가 같이 있는 디렉토리 찾기
            if (parent / 'backend').exists() and (parent / 'frontend').exists():
                return parent
        
        # 기본값으로 현재 디렉토리
        return current
    
    def _find_ai_models_dir(self) -> Optional[Path]:
        """AI 모델 디렉토리 찾기 (우선순위 기반)"""
        candidates = [
            # 1순위: backend 내부
            self.project_root / "backend" / "ai_models",
            self.current_dir / "ai_models",  # backend에서 실행 시
            
            # 2순위: 프로젝트 루트
            self.project_root / "ai_models",
            
            # 3순위: 현재 위치 기준
            self.current_dir / "backend" / "ai_models",
            self.current_dir.parent / "backend" / "ai_models",
            
            # 4순위: 기타
            self.project_root / "models",
            self.project_root / "checkpoints"
        ]
        
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                # 내용이 있는지 확인
                try:
                    if any(candidate.iterdir()):
                        return candidate
                except PermissionError:
                    continue
        
        return None
    
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
        
        return environments
    
    def _init_classification_patterns(self):
        """AI 모델 분류 패턴 초기화"""
        
        # 프레임워크 패턴
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
            }
        }
        
        # MyCloset AI 8단계 패턴 (정교화)
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
        
        # 모델 타입 패턴
        self.model_type_patterns = {
            'diffusion_model': [r'diffusion', r'stable.*diffusion', r'ootd', r'unet'],
            'clip_model': [r'clip', r'vit.*patch', r'vision.*transformer'],
            'pose_model': [r'pose', r'openpose', r'dwpose', r'keypoint'],
            'segmentation_model': [r'segment', r'u2net', r'mask', r'sam'],
            'parsing_model': [r'parsing', r'human.*parsing', r'atr', r'schp'],
            'warping_model': [r'warp', r'tom', r'tps', r'flow'],
            'checkpoint': [r'checkpoint', r'ckpt', r'epoch', r'step'],
            'config_file': [r'config', r'setup', r'tokenizer']
        }
    
    def scan_complete_system(self) -> List[ModelInfo]:
        """완전한 시스템 스캔"""
        logger.info("🚀 완전한 AI 모델 시스템 스캔 시작")
        logger.info("=" * 80)
        
        scan_start = time.time()
        
        # 1. 스캔 경로 생성
        scan_paths = self._generate_scan_paths()
        
        # 2. 병렬 스캔 실행
        logger.info(f"🔍 {len(scan_paths)}개 위치에서 모델 검색...")
        all_files = self._scan_all_paths(scan_paths)
        
        if not all_files:
            logger.warning("❌ AI 모델 파일을 찾을 수 없습니다.")
            self._debug_scan_paths(scan_paths)
            return []
        
        # 3. 모델 분석
        logger.info(f"🧠 {len(all_files):,}개 파일 AI 분석 중...")
        self._analyze_all_models(all_files)
        
        # 4. 후처리
        self._post_process_results()
        
        # 5. 결과 출력
        scan_duration = time.time() - scan_start
        self._print_complete_results(scan_duration)
        
        return self.found_models
    
    def _generate_scan_paths(self) -> List[Path]:
        """스캔 경로 생성 (우선순위 기반)"""
        paths = []
        
        # 1순위: conda 환경 (conda_first 옵션 시)
        if self.conda_first:
            for env_name, env_path in self.conda_environments.items():
                conda_paths = [
                    env_path / "lib" / "python3.11" / "site-packages",
                    env_path / "lib" / "python3.10" / "site-packages",
                    env_path / "share",
                    env_path / "models"
                ]
                paths.extend([p for p in conda_paths if self._is_accessible(p)])
        
        # 2순위: 프로젝트 내부
        project_paths = [
            self.ai_models_dir,
            self.project_root / "models",
            self.project_root / "checkpoints",
            self.project_root / "weights"
        ]
        paths.extend([p for p in project_paths if p and self._is_accessible(p)])
        
        # 3순위: 사용자 디렉토리
        home = Path.home()
        user_paths = [
            home / "Downloads",
            home / "Documents",
            home / "Desktop",
            home / ".cache" / "huggingface",
            home / ".cache" / "torch",
            home / ".cache" / "diffusers",
            home / ".local" / "lib"
        ]
        paths.extend([p for p in user_paths if self._is_accessible(p)])
        
        # 4순위: 시스템 전체 (deep_scan 옵션 시)
        if self.deep_scan:
            system_paths = []
            system = platform.system().lower()
            
            if system == "darwin":  # macOS
                system_paths = [
                    Path("/opt/homebrew/lib"),
                    Path("/usr/local/lib"),
                    Path("/opt")
                ]
            elif system == "linux":
                system_paths = [
                    Path("/opt"),
                    Path("/usr/local/lib"),
                    Path("/usr/share")
                ]
            
            paths.extend([p for p in system_paths if self._is_accessible(p)])
        
        # conda 환경이 conda_first가 아닐 때 추가
        if not self.conda_first:
            for env_name, env_path in self.conda_environments.items():
                conda_paths = [
                    env_path / "lib" / "python3.11" / "site-packages",
                    env_path / "lib" / "python3.10" / "site-packages"
                ]
                paths.extend([p for p in conda_paths if self._is_accessible(p)])
        
        # 중복 제거
        return self._remove_duplicate_paths(paths)
    
    def _is_accessible(self, path: Path) -> bool:
        """경로 접근 가능성 확인"""
        try:
            if not path.exists():
                return False
            if not os.access(path, os.R_OK):
                return False
            
            # 보호된 경로 제외
            path_str = str(path).lower()
            protected = ['/system/', '/private/', '/dev/', '/proc/', 'keychain', 'security']
            return not any(p in path_str for p in protected)
            
        except (PermissionError, OSError):
            return False
    
    def _remove_duplicate_paths(self, paths: List[Path]) -> List[Path]:
        """중복 경로 제거"""
        unique_paths = []
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
    
    def _scan_all_paths(self, scan_paths: List[Path]) -> List[Path]:
        """모든 경로 스캔"""
        all_files = []
        
        for path in scan_paths:
            if self.verbose:
                logger.info(f"📂 스캔 중: {path}")
            
            files = self._scan_single_path(path)
            if files:
                all_files.extend(files)
                self.scan_locations[str(path)] = [str(f) for f in files]
                
                total_size = sum(f.stat().st_size for f in files if f.exists()) / (1024**2)
                if self.verbose:
                    logger.info(f"  ✅ {len(files)}개 파일 발견 ({total_size:.1f}MB)")
        
        return all_files
    
    def _scan_single_path(self, path: Path) -> List[Path]:
        """단일 경로 스캔"""
        found_files = []
        
        try:
            if not self._is_accessible(path):
                return found_files
            
            # 모델 확장자 검색
            model_extensions = ['.pth', '.pt', '.bin', '.safetensors', '.ckpt', 
                              '.h5', '.pb', '.onnx', '.pkl', '.model', '.weights']
            
            for ext in model_extensions:
                pattern = f"**/*{ext}"
                for file_path in path.rglob(pattern):
                    if (file_path.is_file() and 
                        self._is_model_file(file_path) and
                        not self._should_skip_file(file_path)):
                        found_files.append(file_path)
                        
        except Exception as e:
            if self.verbose:
                logger.warning(f"⚠️ 스캔 오류 {path}: {e}")
            self.errors.append(f"스캔 실패 {path}: {e}")
        
        return found_files
    
    def _is_model_file(self, file_path: Path) -> bool:
        """모델 파일 여부 확인"""
        try:
            size_bytes = file_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            
            # 크기 제한 (0.1MB ~ 50GB)
            return 0.1 <= size_mb <= 50 * 1024
            
        except OSError:
            return False
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """건너뛸 파일 여부"""
        path_str = str(file_path).lower()
        skip_patterns = [
            'node_modules', '__pycache__', '.git', '.cache/pip',
            'trash', 'recycle', 'temp', 'tmp', '.ds_store'
        ]
        return any(pattern in path_str for pattern in skip_patterns)
    
    def _analyze_all_models(self, model_files: List[Path]):
        """모든 모델 분석"""
        
        # 병렬 처리로 분석 속도 향상
        with ThreadPoolExecutor(max_workers=min(len(model_files), 8)) as executor:
            futures = [executor.submit(self._analyze_single_model, f) for f in model_files]
            
            for future in as_completed(futures):
                try:
                    model_info = future.result()
                    if model_info and model_info.is_valid:
                        self.found_models.append(model_info)
                except Exception as e:
                    self.errors.append(f"모델 분석 실패: {e}")
    
    def _analyze_single_model(self, file_path: Path) -> Optional[ModelInfo]:
        """단일 모델 상세 분석"""
        try:
            # 파일 기본 정보
            stat_info = file_path.stat()
            size_bytes = stat_info.st_size
            size_mb = size_bytes / (1024 * 1024)
            size_gb = size_mb / 1024
            
            # 체크섬 계산 (샘플링)
            checksum = self._calculate_checksum(file_path, size_mb)
            
            # 프레임워크 분류
            framework = self._classify_framework(file_path)
            
            # 모델 타입 분류
            model_type = self._classify_model_type(file_path)
            
            # Step 분류
            step_candidate, confidence = self._classify_step(file_path)
            
            # 환경 정보
            is_in_project = self._is_in_project(file_path)
            conda_info = self._check_conda_environment(file_path)
            
            # 검증
            is_valid = self._validate_model(file_path, framework)
            
            # 동반 파일
            companion_files = self._find_companion_files(file_path)
            
            # 중요도 점수
            importance_score = self._calculate_importance(
                file_path, size_gb, framework, step_candidate, confidence, is_in_project
            )
            
            return ModelInfo(
                name=file_path.name,
                path=str(file_path),
                absolute_path=str(file_path.absolute()),
                size_mb=size_mb,
                size_gb=size_gb,
                framework=framework,
                model_type=model_type,
                step_candidate=step_candidate,
                confidence=confidence,
                is_valid=is_valid,
                is_in_project=is_in_project,
                is_in_conda=conda_info['is_conda'],
                conda_env_name=conda_info['env_name'],
                parent_directory=file_path.parent.name,
                created_time=datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
                modified_time=datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                checksum=checksum,
                companion_files=companion_files,
                importance_score=importance_score,
                extension=file_path.suffix.lower()
            )
            
        except Exception as e:
            if self.verbose:
                logger.warning(f"모델 분석 실패 {file_path}: {e}")
            return None
    
    def _calculate_checksum(self, file_path: Path, size_mb: float) -> str:
        """체크섬 계산 (샘플링 방식)"""
        try:
            hasher = hashlib.md5()
            
            with open(file_path, 'rb') as f:
                if size_mb > 100:  # 100MB 이상은 샘플링
                    # 시작, 중간, 끝 부분만 해시
                    chunks = [f.read(1024*1024)]  # 시작 1MB
                    try:
                        f.seek(int(size_mb * 1024 * 512))
                        chunks.append(f.read(1024*1024))  # 중간 1MB
                        f.seek(-1024*1024, 2)
                        chunks.append(f.read(1024*1024))  # 끝 1MB
                    except:
                        pass
                    
                    for chunk in chunks:
                        if chunk:
                            hasher.update(chunk)
                else:
                    # 작은 파일은 전체 해시
                    while True:
                        chunk = f.read(8192)
                        if not chunk:
                            break
                        hasher.update(chunk)
            
            return hasher.hexdigest()[:16]
            
        except Exception:
            return "unknown"
    
    def _classify_framework(self, file_path: Path) -> str:
        """프레임워크 분류"""
        extension = file_path.suffix.lower()
        path_str = str(file_path).lower()
        
        # 확장자 기반
        for framework, info in self.framework_patterns.items():
            if extension in info['extensions']:
                # 추가 지시자 확인
                if info['indicators']:
                    for indicator in info['indicators']:
                        if indicator in path_str:
                            return framework
                return framework
        
        # 바이너리 파일 세부 분류
        if extension == '.bin':
            if any(term in path_str for term in ['pytorch', 'torch', 'transformers']):
                return 'pytorch'
            elif 'tensorflow' in path_str:
                return 'tensorflow'
            else:
                return 'binary'
        
        return 'unknown'
    
    def _classify_model_type(self, file_path: Path) -> str:
        """모델 타입 분류"""
        path_str = str(file_path).lower()
        
        for model_type, patterns in self.model_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, path_str):
                    return model_type
        
        return 'unknown'
    
    def _classify_step(self, file_path: Path) -> Tuple[str, float]:
        """MyCloset AI 8단계 분류"""
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
    
    def _is_in_project(self, file_path: Path) -> bool:
        """프로젝트 내부 여부"""
        try:
            return file_path.is_relative_to(self.project_root)
        except ValueError:
            return False
    
    def _check_conda_environment(self, file_path: Path) -> Dict[str, Any]:
        """conda 환경 확인"""
        result = {'is_conda': False, 'env_name': None}
        
        for env_name, env_path in self.conda_environments.items():
            try:
                if file_path.is_relative_to(env_path):
                    result['is_conda'] = True
                    result['env_name'] = env_name
                    break
            except ValueError:
                continue
        
        return result
    
    def _validate_model(self, file_path: Path, framework: str) -> bool:
        """모델 유효성 검사"""
        try:
            if file_path.stat().st_size < 1024:
                return False
            
            with open(file_path, 'rb') as f:
                header = f.read(100)
            
            if framework == 'pytorch' and (b'PK' in header or b'\x80' in header):
                return True
            elif framework == 'safetensors' and b'{' in header:
                return True
            elif framework == 'tensorflow':
                return len(header) > 10
            elif framework == 'onnx':
                return b'onnx' in header.lower() or len(header) > 50
            else:
                return True
                
        except Exception:
            return False
    
    def _find_companion_files(self, file_path: Path) -> List[str]:
        """동반 파일 찾기"""
        companions = []
        parent_dir = file_path.parent
        
        companion_patterns = [
            'config.json', 'config.yaml', 'tokenizer.json',
            'model_config.json', 'pytorch_model.bin'
        ]
        
        for pattern in companion_patterns:
            companion_path = parent_dir / pattern
            if companion_path.exists() and companion_path != file_path:
                companions.append(str(companion_path))
        
        return companions
    
    def _calculate_importance(
        self, file_path: Path, size_gb: float, framework: str, 
        step: str, confidence: float, is_in_project: bool
    ) -> float:
        """중요도 점수 계산"""
        score = 0.0
        
        # 크기 점수 (최대 30점)
        score += min(size_gb * 10, 30)
        
        # 신뢰도 점수 (최대 25점)
        score += confidence * 25
        
        # 프레임워크 점수 (최대 15점)
        framework_scores = {
            'pytorch': 15, 'safetensors': 12, 'diffusers': 10,
            'transformers': 8, 'tensorflow': 6, 'onnx': 4
        }
        score += framework_scores.get(framework, 0)
        
        # Step 점수 (최대 20점)
        if step != 'unknown':
            score += 20
        
        # 위치 점수 (최대 10점)
        if is_in_project:
            score += 10
        elif any(file_path.is_relative_to(env) for env in self.conda_environments.values()):
            score += 5
        
        return min(score, 100.0)
    
    def _post_process_results(self):
        """결과 후처리"""
        if not self.found_models:
            return
        
        # 중복 탐지 (체크섬 기반)
        checksum_groups = {}
        for model in self.found_models:
            if model.checksum != "unknown":
                if model.checksum not in checksum_groups:
                    checksum_groups[model.checksum] = []
                checksum_groups[model.checksum].append(model)
        
        # 중복된 것들 표시
        for checksum, models in checksum_groups.items():
            if len(models) > 1:
                # 가장 중요한 것을 원본으로 설정
                primary = max(models, key=lambda m: m.importance_score)
                for model in models:
                    if model != primary:
                        model.importance_score *= 0.8  # 중복 패널티
    
    def _debug_scan_paths(self, scan_paths: List[Path]):
        """스캔 경로 디버그"""
        logger.info("🔍 스캔 경로 디버그:")
        
        for i, path in enumerate(scan_paths, 1):
            exists = "✅" if path.exists() else "❌"
            logger.info(f"  {i:2d}. {exists} {path}")
            
            if path.exists():
                try:
                    items = list(path.iterdir())
                    logger.info(f"      📁 {len(items)}개 항목")
                    
                    # 모델 파일 직접 검색
                    model_files = []
                    for item in items[:5]:  # 처음 5개만 체크
                        if item.is_file() and item.suffix.lower() in ['.pth', '.pt', '.bin']:
                            model_files.append(item)
                    
                    if model_files:
                        logger.info(f"      🤖 {len(model_files)}개 모델 파일 발견")
                        for mf in model_files:
                            size_mb = mf.stat().st_size / (1024*1024)
                            logger.info(f"        - {mf.name} ({size_mb:.1f}MB)")
                    
                except Exception as e:
                    logger.info(f"      ❌ 읽기 실패: {e}")
    
    def _print_complete_results(self, scan_duration: float):
        """완전한 결과 출력"""
        stats = self._calculate_statistics(scan_duration)
        
        print("\n" + "=" * 100)
        print("🎯 MyCloset AI - 완전한 AI 모델 스캔 결과")
        print("=" * 100)
        
        # 기본 정보
        print(f"🕐 스캔 시간: {scan_duration:.1f}초")
        print(f"💻 시스템: {platform.system()} {platform.machine()}")
        print(f"🐍 현재 conda 환경: {self.current_conda_env or 'None'}")
        
        # 스캔 통계
        print(f"\n📊 스캔 통계:")
        print(f"   📁 스캔 위치: {len(self.scan_locations)}곳")
        print(f"   📄 검사 파일: {stats.total_files_scanned:,}개")
        print(f"   🤖 발견 모델: {stats.models_found:,}개")
        print(f"   💾 총 용량: {stats.total_size_gb:.2f}GB")
        print(f"   ✅ 유효 모델: {stats.valid_models}개")
        
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
            conda_dist = {}
            for model in self.found_models:
                if model.is_in_conda and model.conda_env_name:
                    conda_dist[model.conda_env_name] = conda_dist.get(model.conda_env_name, 0) + 1
            
            print(f"   conda 환경별:")
            for env_name, count in sorted(conda_dist.items()):
                env_size = sum(m.size_gb for m in self.found_models if m.conda_env_name == env_name)
                print(f"     - {env_name}: {count}개 ({env_size:.1f}GB)")
        
        # 프레임워크 분포
        print(f"\n🔧 프레임워크별 분포:")
        for fw, count in sorted(stats.framework_distribution.items(), key=lambda x: x[1], reverse=True):
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
            
            for step, count in sorted(stats.step_distribution.items()):
                if count > 0:
                    display_name = step_names.get(step, step)
                    step_size = sum(m.size_gb for m in self.found_models 
                                  if m.step_candidate == step and m.confidence >= 0.5)
                    print(f"   {display_name}: {count}개 ({step_size:.1f}GB)")
        
        # 상위 중요 모델들
        print(f"\n🏆 중요도 상위 모델들:")
        top_models = sorted(self.found_models, key=lambda x: x.importance_score, reverse=True)[:15]
        
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
        
        # 권장사항
        print(f"\n💡 권장사항:")
        recommendations = []
        
        if stats.conda_models > 0 and stats.project_models == 0:
            recommendations.append(f"🔄 conda 환경의 {stats.conda_models}개 모델을 프로젝트로 연결 고려")
        
        if stats.system_models > stats.project_models:
            recommendations.append(f"📦 시스템의 {stats.system_models}개 모델을 프로젝트로 통합 고려")
        
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
    
    def _calculate_statistics(self, scan_duration: float) -> ScanStatistics:
        """통계 계산"""
        stats = ScanStatistics()
        
        # 기본 통계
        stats.total_files_scanned = sum(len(files) for files in self.scan_locations.values())
        stats.models_found = len(self.found_models)
        stats.total_size_gb = sum(m.size_gb for m in self.found_models)
        stats.scan_duration = scan_duration
        
        # 위치별 통계
        stats.conda_models = sum(1 for m in self.found_models if m.is_in_conda)
        stats.project_models = sum(1 for m in self.found_models if m.is_in_project)
        stats.system_models = stats.models_found - stats.conda_models - stats.project_models
        stats.valid_models = sum(1 for m in self.found_models if m.is_valid)
        
        # 분포 통계
        for model in self.found_models:
            # 프레임워크 분포
            fw = model.framework
            stats.framework_distribution[fw] = stats.framework_distribution.get(fw, 0) + 1
            
            # Step 분포 (신뢰도 0.5+ 만)
            if model.confidence >= 0.5:
                step = model.step_candidate
                stats.step_distribution[step] = stats.step_distribution.get(step, 0) + 1
        
        return stats
    
    def _print_suggestions(self):
        """제안사항 출력"""
        print("\n💡 AI 모델을 찾을 수 없습니다. 다음을 확인해보세요:")
        
        print("\n🔍 검색 확장:")
        print("   python quick_scanner.py --deep                 # 전체 시스템 딥 스캔")
        print("   python quick_scanner.py --conda-first          # conda 환경 우선 스캔")
        print("   python quick_scanner.py --verbose              # 상세 진행 과정 출력")
        
        print("\n📁 예상 모델 위치:")
        expected_locations = [
            ("🏠 프로젝트", self.ai_models_dir),
            ("📥 다운로드", Path.home() / "Downloads"),
            ("💾 HuggingFace", Path.home() / ".cache" / "huggingface"),
            ("🔥 PyTorch", Path.home() / ".cache" / "torch")
        ]
        
        for desc, location in expected_locations:
            exists = "✅" if location and location.exists() else "❌"
            print(f"   {exists} {desc}: {location}")
        
        print("\n🐍 conda 환경별 확인:")
        for env_name, env_path in self.conda_environments.items():
            site_packages = env_path / "lib" / "python3.11" / "site-packages"
            exists = "✅" if site_packages.exists() else "❌"
            print(f"   {exists} {env_name}: {site_packages}")
        
        print("\n🔧 수동 확인 명령어:")
        print(f"   find {self.project_root} -name '*.pth' -o -name '*.pt' -o -name '*.bin'")
        print(f"   ls -la {self.ai_models_dir}/ 2>/dev/null")
        print(f"   find ~ -name '*.pth' -size +1M 2>/dev/null | head -10")
    
    def generate_config_files(self, output_dir: str = "generated_configs") -> List[str]:
        """설정 파일들 생성"""
        if not self.found_models:
            logger.warning("❌ 설정 파일을 생성할 모델이 없습니다.")
            return []
        
        output_path = self.project_root / output_dir
        output_path.mkdir(exist_ok=True)
        
        generated_files = []
        
        # 1. JSON 결과 파일
        json_file = self._generate_json_config(output_path)
        generated_files.append(json_file)
        
        # 2. Python 설정 파일
        python_file = self._generate_python_config(output_path)
        generated_files.append(python_file)
        
        # 3. conda 환경 설정
        if any(m.is_in_conda for m in self.found_models):
            conda_file = self._generate_conda_config(output_path)
            generated_files.append(conda_file)
        
        logger.info(f"📝 {len(generated_files)}개 설정 파일 생성 완료: {output_path}")
        return generated_files
    
    def _generate_json_config(self, output_path: Path) -> str:
        """JSON 설정 파일 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = output_path / f"model_scan_result_{timestamp}.json"
        
        config_data = {
            "scan_info": {
                "timestamp": datetime.now().isoformat(),
                "project_root": str(self.project_root),
                "ai_models_dir": str(self.ai_models_dir),
                "scan_duration": time.time() - self.scan_start_time,
                "conda_environments": list(self.conda_environments.keys()),
                "current_conda_env": self.current_conda_env
            },
            "statistics": asdict(self._calculate_statistics(time.time() - self.scan_start_time)),
            "models": [asdict(model) for model in self.found_models],
            "scan_locations": self.scan_locations,
            "errors": self.errors,
            "warnings": self.warnings
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False, default=str)
        
        return str(json_file)
    
    def _generate_python_config(self, output_path: Path) -> str:
        """Python 설정 파일 생성"""
        python_file = output_path / "model_paths_config.py"
        
        config_content = f'''#!/usr/bin/env python3
"""
MyCloset AI 모델 경로 설정 - 자동 생성됨
생성 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
발견된 모델: {len(self.found_models)}개
"""

from pathlib import Path
from typing import Dict, List, Optional

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent
AI_MODELS_ROOT = PROJECT_ROOT / "ai_models"

# 발견된 모델 경로들
SCANNED_MODELS = {{
'''
        
        for i, model in enumerate(self.found_models):
            safe_name = model.name.replace('.', '_').replace('-', '_').replace(' ', '_')
            config_content += f'''    "{safe_name}": {{
        "name": "{model.name}",
        "path": Path(r"{model.path}"),
        "framework": "{model.framework}",
        "step": "{model.step_candidate}",
        "confidence": {model.confidence:.3f},
        "size_mb": {model.size_mb:.1f},
        "importance": {model.importance_score:.1f},
        "is_in_project": {model.is_in_project},
        "is_in_conda": {model.is_in_conda},
        "conda_env": "{model.conda_env_name or ''}"
    }},
'''
        
        config_content += '''}}

# Step별 모델 매핑 (신뢰도 50% 이상)
STEP_MODELS = {
'''
        
        # Step별 모델 그룹화
        step_models = {}
        for model in self.found_models:
            if model.confidence >= 0.5:
                step = model.step_candidate
                if step not in step_models:
                    step_models[step] = []
                safe_name = model.name.replace('.', '_').replace('-', '_').replace(' ', '_')
                step_models[step].append(safe_name)
        
        for step, models in step_models.items():
            config_content += f'    "{step}": {models},\n'
        
        config_content += '''}

# 프레임워크별 모델 매핑
FRAMEWORK_MODELS = {
'''
        
        framework_models = {}
        for model in self.found_models:
            fw = model.framework
            if fw not in framework_models:
                framework_models[fw] = []
            safe_name = model.name.replace('.', '_').replace('-', '_').replace(' ', '_')
            framework_models[fw].append(safe_name)
        
        for fw, models in framework_models.items():
            config_content += f'    "{fw}": {models},\n'
        
        config_content += f'''}}

def get_model_path(model_name: str) -> Optional[Path]:
    """모델 경로 반환"""
    for key, info in SCANNED_MODELS.items():
        if model_name.lower() in key.lower() or model_name.lower() in info["name"].lower():
            return info["path"]
    return None

def get_step_models(step: str) -> List[str]:
    """Step별 모델 목록"""
    return STEP_MODELS.get(step, [])

def get_framework_models(framework: str) -> List[str]:
    """프레임워크별 모델 목록"""
    return FRAMEWORK_MODELS.get(framework, [])

def get_best_model_for_step(step: str) -> Optional[str]:
    """Step별 최고 중요도 모델"""
    step_models = get_step_models(step)
    if not step_models:
        return None
    
    best_model = None
    best_score = 0
    
    for model_key in step_models:
        if model_key in SCANNED_MODELS:
            score = SCANNED_MODELS[model_key]["importance"]
            if score > best_score:
                best_score = score
                best_model = model_key
    
    return best_model

def list_available_models() -> Dict[str, dict]:
    """사용 가능한 모델 목록"""
    available = {{}}
    for key, info in SCANNED_MODELS.items():
        if info["path"].exists():
            available[key] = info
    return available

def get_conda_models(env_name: str = None) -> List[str]:
    """conda 환경별 모델 목록"""
    conda_models = []
    for key, info in SCANNED_MODELS.items():
        if info["is_in_conda"]:
            if env_name is None or info["conda_env"] == env_name:
                conda_models.append(key)
    return conda_models

if __name__ == "__main__":
    print("🤖 MyCloset AI 모델 설정")
    print("=" * 50)
    
    available = list_available_models()
    print(f"사용 가능한 모델: {{len(available)}}개")
    
    print("\\nStep별 모델:")
    for step, models in STEP_MODELS.items():
        if models:
            step_name = step.replace('step_', '').replace('_', ' ').title()
            print(f"  {{step_name}}: {{len(models)}}개")
    
    print("\\n프레임워크별 분포:")
    for fw, models in FRAMEWORK_MODELS.items():
        print(f"  {{fw}}: {{len(models)}}개")
'''
        
        with open(python_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        return str(python_file)
    
    def _generate_conda_config(self, output_path: Path) -> str:
        """conda 환경 설정 파일 생성"""
        conda_file = output_path / "conda_model_config.py"
        
        conda_models = [m for m in self.found_models if m.is_in_conda]
        
        config_content = f'''#!/usr/bin/env python3
"""
MyCloset AI - conda 환경별 모델 설정
생성 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
conda 모델: {len(conda_models)}개
"""

from pathlib import Path
from typing import Dict, List

# conda 환경 정보
CONDA_ENVIRONMENTS = {{
'''
        
        for env_name, env_path in self.conda_environments.items():
            config_content += f'    "{env_name}": Path(r"{env_path}"),\n'
        
        config_content += f'''}}

CURRENT_CONDA_ENV = "{self.current_conda_env or 'None'}"

# conda 환경별 모델 매핑
CONDA_MODELS = {{
'''
        
        for env_name in self.conda_environments.keys():
            env_models = [m for m in conda_models if m.conda_env_name == env_name]
            if env_models:
                config_content += f'    "{env_name}": [\n'
                for model in env_models:
                    config_content += f'        "{model.path}",\n'
                config_content += f'    ],\n'
        
        config_content += '''}

def get_conda_model_paths(env_name: str) -> List[str]:
    """conda 환경별 모델 경로 목록"""
    return CONDA_MODELS.get(env_name, [])

def get_current_env_models() -> List[str]:
    """현재 conda 환경의 모델들"""
    if CURRENT_CONDA_ENV != "None":
        return get_conda_model_paths(CURRENT_CONDA_ENV)
    return []

def list_conda_environments() -> List[str]:
    """conda 환경 목록"""
    return list(CONDA_ENVIRONMENTS.keys())

if __name__ == "__main__":
    print("🐍 MyCloset AI conda 환경 모델 설정")
    print("=" * 50)
    
    print(f"현재 환경: {CURRENT_CONDA_ENV}")
    print(f"총 환경: {len(CONDA_ENVIRONMENTS)}개")
    
    for env_name in CONDA_ENVIRONMENTS.keys():
        models = get_conda_model_paths(env_name)
        print(f"  {env_name}: {len(models)}개 모델")
'''
        
        with open(conda_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        return str(conda_file)

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="완전한 AI 모델 체크포인트 검색 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python quick_scanner.py                           # 기본 스캔
  python quick_scanner.py --verbose                 # 상세 출력
  python quick_scanner.py --organize                # 스캔 + 설정 생성
  python quick_scanner.py --deep                    # 딥 스캔
  python quick_scanner.py --conda-first             # conda 우선
  python quick_scanner.py --deep --organize         # 완전 스캔 + 설정
        """
    )
    
    parser.add_argument('--verbose', action='store_true', help='상세 출력')
    parser.add_argument('--organize', action='store_true', help='스캔 후 설정 파일 생성')
    parser.add_argument('--deep', action='store_true', help='전체 시스템 딥 스캔')
    parser.add_argument('--conda-first', action='store_true', help='conda 환경 우선 스캔')
    parser.add_argument('--output-dir', type=str, default='generated_configs', help='설정 파일 출력 디렉토리')
    parser.add_argument('--min-size', type=float, default=0.1, help='최소 파일 크기 (MB)')
    parser.add_argument('--max-size', type=float, default=50.0, help='최대 파일 크기 (GB)')
    
    args = parser.parse_args()
    
    try:
        # 스캐너 초기화
        scanner = CompleteModelScanner(
            verbose=args.verbose,
            conda_first=args.conda_first,
            deep_scan=args.deep
        )
        
        # 완전한 시스템 스캔 실행
        models = scanner.scan_complete_system()
        
        # 설정 파일 생성 (옵션)
        if args.organize and models:
            config_files = scanner.generate_config_files(args.output_dir)
            print(f"\n📝 생성된 설정 파일:")
            for config_file in config_files:
                print(f"   ✅ {config_file}")
        
        # 완료 메시지
        print(f"\n✅ 스캔 완료!")
        print(f"🤖 발견된 모델: {len(models)}개")
        
        if models:
            conda_models = sum(1 for m in models if m.is_in_conda)
            project_models = sum(1 for m in models if m.is_in_project)
            total_size = sum(m.size_gb for m in models)
            avg_importance = sum(m.importance_score for m in models) / len(models)
            
            print(f"🐍 conda 모델: {conda_models}개")
            print(f"🏠 프로젝트 모델: {project_models}개")
            print(f"💾 총 용량: {total_size:.2f}GB")
            print(f"⭐ 평균 중요도: {avg_importance:.1f}/100")
            
            # 다음 단계 안내
            print(f"\n🚀 다음 단계:")
            if args.organize:
                print("1. generated_configs/ 폴더의 설정 파일들 확인")
                print("2. model_paths_config.py를 프로젝트에 import")
                print("3. get_model_path() 함수로 모델 경로 사용")
            else:
                print("1. python quick_scanner.py --organize  # 설정 파일 생성")
                print("2. 중복 모델 정리 및 프로젝트 통합")
                print("3. conda 환경 모델 연결")
        else:
            print("\n🔍 모델을 찾지 못했습니다. 다음을 시도해보세요:")
            print("1. python quick_scanner.py --deep --verbose")
            print("2. python quick_scanner.py --conda-first")
            print("3. 실제 모델 다운로드 여부 확인")
        
        return 0 if models else 1
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())