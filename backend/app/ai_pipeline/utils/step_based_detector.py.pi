# app/ai_pipeline/utils/step_based_detector.py
"""
🔥 MyCloset AI - Step 기반 자동 탐지 시스템 v1.0
✅ Step 클래스들의 체크포인트 파일 자동 탐지
✅ Step = AI 모델 + 처리 로직 통합 구조 완벽 지원
✅ PipelineManager와 완전 연동
✅ M3 Max 128GB 최적화
✅ conda 환경 특화 스캔

🎯 핵심 특징:
- Step 클래스들의 체크포인트 파일 경로 자동 탐지
- Step별 모델 로드 가능성 검증
- 실제 PyTorch 체크포인트 내용 검증
- PipelineManager 호환 설정 자동 생성
- 기존 ModelLoader 호환성 완전 유지
"""

import os
import re
import time
import logging
import hashlib
import json
import threading
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import weakref

# PyTorch 및 AI 라이브러리 (안전한 import)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    from PIL import Image
    IMAGING_AVAILABLE = True
except ImportError:
    IMAGING_AVAILABLE = False

try:
    from transformers import AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

# ==============================================
# 🔥 Step 기반 데이터 구조
# ==============================================

class StepStatus(Enum):
    """Step 상태"""
    AVAILABLE = "available"          # Step 클래스 로드 가능
    CHECKPOINT_MISSING = "checkpoint_missing"  # 체크포인트 없음
    CORRUPTED = "corrupted"         # 체크포인트 손상
    LOADING_FAILED = "loading_failed"  # 로딩 실패
    NOT_FOUND = "not_found"         # Step 클래스 없음

class StepPriority(Enum):
    """Step 우선순위"""
    CRITICAL = 1      # 필수 (Human Parsing, Virtual Fitting)
    HIGH = 2          # 중요 (Pose Estimation, Cloth Segmentation)
    MEDIUM = 3        # 일반 (Cloth Warping, Geometric Matching)
    LOW = 4           # 보조 (Post Processing, Quality Assessment)

@dataclass
class StepCheckpointInfo:
    """Step 체크포인트 정보"""
    step_name: str
    step_class_name: str
    checkpoint_path: Optional[Path] = None
    checkpoint_size_mb: float = 0.0
    pytorch_valid: bool = False
    parameter_count: int = 0
    last_modified: float = 0.0
    step_available: bool = False
    status: StepStatus = StepStatus.NOT_FOUND
    priority: StepPriority = StepPriority.MEDIUM
    alternative_checkpoints: List[Path] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_info: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0

# ==============================================
# 🔥 Step별 체크포인트 패턴 정의
# ==============================================

STEP_CHECKPOINT_PATTERNS = {
    "step_01_human_parsing": {
        "class_name": "HumanParsingStep",
        "priority": StepPriority.CRITICAL,
        "checkpoint_patterns": [
            r".*human.*parsing.*\.pth$",
            r".*schp.*atr.*\.pth$", 
            r".*graphonomy.*\.pth$",
            r".*lip_model.*\.pth$",
            r".*atr_model.*\.pth$"
        ],
        "expected_directories": [
            "checkpoints/human_parsing",
            "checkpoints/step_01",
            "ai_models/human_parsing",
            "models/schp",
            "Self-Correction-Human-Parsing"
        ],
        "file_extensions": [".pth", ".pt"],
        "size_range_mb": (1, 500),
        "expected_parameters": (25000000, 70000000),
        "required_keys": ["state_dict", "model"],
        "expected_layers": ["backbone", "classifier", "conv"]
    },
    
    "step_02_pose_estimation": {
        "class_name": "PoseEstimationStep",
        "priority": StepPriority.HIGH,
        "checkpoint_patterns": [
            r".*pose.*estimation.*\.pth$",
            r".*openpose.*\.pth$",
            r".*body.*25.*\.pth$",
            r".*pose_iter.*\.caffemodel$"
        ],
        "expected_directories": [
            "checkpoints/pose_estimation",
            "checkpoints/step_02",
            "ai_models/openpose",
            "models/openpose",
            "openpose/models"
        ],
        "file_extensions": [".pth", ".pt", ".caffemodel"],
        "size_range_mb": (10, 1000),
        "expected_parameters": (10000000, 200000000),
        "required_keys": ["state_dict", "model"],
        "expected_layers": ["stage", "paf", "heatmap"]
    },
    
    "step_03_cloth_segmentation": {
        "class_name": "ClothSegmentationStep", 
        "priority": StepPriority.HIGH,
        "checkpoint_patterns": [
            r".*cloth.*segmentation.*\.pth$",
            r".*u2net.*\.pth$",
            r".*sam.*vit.*\.pth$",
            r".*mask.*anything.*\.pth$"
        ],
        "expected_directories": [
            "checkpoints/cloth_segmentation",
            "checkpoints/step_03",
            "ai_models/u2net",
            "models/sam",
            "segment-anything"
        ],
        "file_extensions": [".pth", ".pt"],
        "size_range_mb": (10, 3000),
        "expected_parameters": (4000000, 650000000),
        "required_keys": ["state_dict", "model"],
        "expected_layers": ["encoder", "decoder", "outconv"]
    },
    
    "step_04_geometric_matching": {
        "class_name": "GeometricMatchingStep",
        "priority": StepPriority.MEDIUM,
        "checkpoint_patterns": [
            r".*geometric.*matching.*\.pth$",
            r".*gmm.*\.pth$",
            r".*tps.*\.pth$",
            r".*matching.*\.pth$"
        ],
        "expected_directories": [
            "checkpoints/geometric_matching",
            "checkpoints/step_04",
            "ai_models/gmm",
            "models/geometric"
        ],
        "file_extensions": [".pth", ".pt"],
        "size_range_mb": (1, 100),
        "expected_parameters": (500000, 50000000),
        "required_keys": ["state_dict", "model"],
        "expected_layers": ["correlation", "regression", "flow"]
    },
    
    "step_05_cloth_warping": {
        "class_name": "ClothWarpingStep",
        "priority": StepPriority.MEDIUM,
        "checkpoint_patterns": [
            r".*cloth.*warping.*\.pth$",
            r".*tom.*\.pth$",
            r".*warping.*\.pth$",
            r".*viton.*warp.*\.pth$"
        ],
        "expected_directories": [
            "checkpoints/cloth_warping",
            "checkpoints/step_05",
            "ai_models/tom",
            "models/warping"
        ],
        "file_extensions": [".pth", ".pt"],
        "size_range_mb": (10, 4000),
        "expected_parameters": (10000000, 1000000000),
        "required_keys": ["state_dict", "model"],
        "expected_layers": ["generator", "discriminator", "warp"]
    },
    
    "step_06_virtual_fitting": {
        "class_name": "VirtualFittingStep",
        "priority": StepPriority.CRITICAL,
        "checkpoint_patterns": [
            r".*virtual.*fitting.*\.bin$",
            r".*ootd.*unet.*\.bin$",
            r".*stable.*diffusion.*\.bin$",
            r".*diffusion.*\.safetensors$",
            r".*ootdiffusion.*\.bin$"
        ],
        "expected_directories": [
            "checkpoints/virtual_fitting",
            "checkpoints/step_06",
            "ai_models/ootdiffusion",
            "ai_models/OOTDiffusion",
            "models/diffusion",
            "stable-diffusion-v1-5"
        ],
        "file_extensions": [".bin", ".pth", ".pt", ".safetensors"],
        "size_range_mb": (100, 8000),
        "expected_parameters": (100000000, 2000000000),
        "required_keys": ["state_dict", "model", "unet"],
        "expected_layers": ["unet", "vae", "text_encoder"]
    },
    
    "step_07_post_processing": {
        "class_name": "PostProcessingStep",
        "priority": StepPriority.LOW,
        "checkpoint_patterns": [
            r".*post.*processing.*\.pth$",
            r".*enhance.*\.pth$",
            r".*refine.*\.pth$"
        ],
        "expected_directories": [
            "checkpoints/post_processing",
            "checkpoints/step_07",
            "ai_models/enhancement"
        ],
        "file_extensions": [".pth", ".pt"],
        "size_range_mb": (1, 1000),
        "expected_parameters": (1000000, 100000000),
        "required_keys": ["state_dict", "model"],
        "expected_layers": ["enhance", "refine"]
    },
    
    "step_08_quality_assessment": {
        "class_name": "QualityAssessmentStep",
        "priority": StepPriority.LOW,
        "checkpoint_patterns": [
            r".*quality.*assessment.*\.pth$",
            r".*quality.*\.pth$",
            r".*scorer.*\.pth$"
        ],
        "expected_directories": [
            "checkpoints/quality_assessment",
            "checkpoints/step_08",
            "ai_models/quality"
        ],
        "file_extensions": [".pth", ".pt"],
        "size_range_mb": (1, 500),
        "expected_parameters": (1000000, 50000000),
        "required_keys": ["state_dict", "model"],
        "expected_layers": ["classifier", "scorer"]
    }
}

# ==============================================
# 🔥 Step 기반 자동 탐지기 클래스
# ==============================================

class StepBasedDetector:
    """
    🎯 Step 기반 AI 모델 자동 탐지 시스템 v1.0
    - Step 클래스들의 체크포인트 파일 자동 탐지
    - Step별 모델 로드 가능성 검증
    - PipelineManager 호환 설정 생성
    """
    
    def __init__(
        self,
        search_paths: Optional[List[Path]] = None,
        enable_pytorch_validation: bool = True,
        enable_step_loading: bool = True,
        enable_caching: bool = True,
        max_workers: int = 4,
        scan_timeout: int = 300,
        device: Optional[str] = None  # 🔥 선택적 디바이스 설정
    ):
        """Step 기반 탐지기 초기화"""
        
        self.logger = logging.getLogger(f"{__name__}.StepBasedDetector")
        
        # 디바이스 설정 (자동 감지 또는 사용자 지정)
        self.device = self._auto_detect_device(device)
        self.device_info = self._get_device_info()
        
        # 검색 경로 설정
        if search_paths is None:
            self.search_paths = self._get_default_search_paths()
        else:
            self.search_paths = search_paths
        
        # 설정
        self.enable_pytorch_validation = enable_pytorch_validation
        self.enable_step_loading = enable_step_loading
        self.enable_caching = enable_caching
        self.max_workers = max_workers
        self.scan_timeout = scan_timeout
        
        # 탐지 결과 저장
        self.detected_steps: Dict[str, StepCheckpointInfo] = {}
        self.scan_stats = {
            "total_files_scanned": 0,
            "checkpoint_files_found": 0,
            "valid_checkpoints": 0,
            "steps_available": 0,
            "scan_duration": 0.0,
            "last_scan_time": 0,
            "errors_encountered": 0,
            "pytorch_validation_errors": 0
        }
        
        # 캐시 관리
        self.cache_db_path = Path("step_detection_cache.db")
        self.cache_ttl = 86400  # 24시간
        self._cache_lock = threading.RLock()
        
        self.logger.info(f"🎯 Step 기반 탐지기 초기화 완료")
        self.logger.info(f"🔧 디바이스: {self.device} ({self.device_info['type']})")
        self.logger.info(f"📁 검색 경로: {len(self.search_paths)}개")
        
        # 캐시 DB 초기화
        if self.enable_caching:
            self._init_cache_db()

    def _auto_detect_device(self, preferred_device: Optional[str]) -> str:
        """디바이스 자동 감지 (Step 파일들과 동일한 로직)"""
        if preferred_device:
            return preferred_device

        if not TORCH_AVAILABLE:
            return 'cpu'

        try:
            if torch.backends.mps.is_available():
                return 'mps'  # M3 Max 우선
            elif torch.cuda.is_available():
                return 'cuda'  # NVIDIA GPU
            else:
                return 'cpu'  # 폴백
        except:
            return 'cpu'

    def _get_device_info(self) -> Dict[str, Any]:
        """디바이스 정보 수집"""
        device_info = {
            "type": self.device,
            "available": True,
            "memory_gb": 16,  # 기본값
            "is_m3_max": False,
            "supports_fp16": False,
            "max_batch_size": 1
        }
        
        try:
            if self.device == 'mps':
                device_info.update({
                    "is_m3_max": self._detect_m3_max(),
                    "supports_fp16": True,
                    "memory_gb": self._get_available_memory(),
                    "max_batch_size": 8 if self._detect_m3_max() else 4
                })
            elif self.device == 'cuda':
                if TORCH_AVAILABLE:
                    device_info.update({
                        "supports_fp16": torch.cuda.is_available(),
                        "memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 8,
                        "max_batch_size": 4
                    })
            else:  # CPU
                device_info.update({
                    "memory_gb": self._get_system_memory(),
                    "max_batch_size": 1
                })
                
        except Exception as e:
            self.logger.debug(f"디바이스 정보 수집 실패: {e}")
        
        return device_info

    def _detect_m3_max(self) -> bool:
        """M3 Max 감지 (Step 파일들과 동일한 로직)"""
        try:
            import platform
            import subprocess

            if platform.system() == 'Darwin':  # macOS
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True, timeout=5)
                cpu_info = result.stdout.strip()
                return 'M3 Max' in cpu_info or 'M3' in cpu_info
        except:
            pass
        return False

    def _get_available_memory(self) -> float:
        """사용 가능한 메모리 감지"""
        try:
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()
                return memory.total / (1024**3)  # GB로 변환
            else:
                return 16.0  # 기본값
        except Exception:
            return 16.0

    def _get_system_memory(self) -> float:
        """시스템 메모리 감지"""
        return self._get_available_memory()

    def get_device_info(self) -> Dict[str, Any]:
        """디바이스 정보 반환 (공개 메서드)"""
        return self.device_info.copy()

    def get_optimal_config_for_device(self) -> Dict[str, Any]:
        """디바이스에 최적화된 설정 반환"""
        config = {
            "device": self.device,
            "precision": "fp16" if self.device_info["supports_fp16"] else "fp32",
            "batch_size": self.device_info["max_batch_size"],
            "memory_limit_gb": self.device_info["memory_gb"] * 0.8,  # 80% 사용
            "enable_optimization": self.device_info["is_m3_max"]
        }
        
        # M3 Max 특화 설정
        if self.device_info["is_m3_max"]:
            config.update({
                "enable_neural_engine": True,
                "memory_pool_size": min(64, self.device_info["memory_gb"]),
                "concurrent_sessions": 4,
                "quality_priority": True
            })
        
        return config

    def _get_default_search_paths(self) -> List[Path]:
        """기본 검색 경로들 반환"""
        current_file = Path(__file__).resolve()
        backend_dir = current_file.parents[3]  # backend 디렉토리
        
        potential_paths = [
            # 프로젝트 내부 경로들
            backend_dir / "ai_models",
            backend_dir / "app" / "ai_pipeline" / "models",
            backend_dir / "app" / "models",
            backend_dir / "checkpoints",
            backend_dir / "models",
            backend_dir / "weights",
            
            # 상위 디렉토리
            backend_dir.parent / "ai_models",
            backend_dir.parent / "models",
            
            # conda 및 사용자 캐시 경로들
            *self._get_conda_model_paths(),
            Path.home() / ".cache" / "huggingface",
            Path.home() / ".cache" / "torch",
            Path.home() / "Downloads"
        ]
        
        # 실제 존재하고 접근 가능한 경로만 반환
        real_paths = []
        for path in potential_paths:
            try:
                if path.exists() and path.is_dir() and os.access(path, os.R_OK):
                    real_paths.append(path)
                    self.logger.debug(f"✅ 유효한 검색 경로: {path}")
            except Exception as e:
                self.logger.debug(f"경로 확인 실패 {path}: {e}")
                continue
        
        return real_paths

    def _get_conda_model_paths(self) -> List[Path]:
        """conda 환경 모델 경로들 탐지"""
        conda_paths = []
        
        try:
            # 현재 conda 환경
            conda_prefix = os.environ.get('CONDA_PREFIX')
            if conda_prefix and Path(conda_prefix).exists():
                conda_base = Path(conda_prefix)
                conda_paths.extend([
                    conda_base / "lib" / "python3.11" / "site-packages",
                    conda_base / "share" / "models",
                    conda_base / "models",
                    conda_base / "checkpoints"
                ])
            
            # conda 루트 탐지
            conda_root = os.environ.get('CONDA_ROOT')
            if not conda_root:
                possible_roots = [
                    Path.home() / "miniforge3",
                    Path.home() / "miniconda3", 
                    Path.home() / "anaconda3",
                    Path("/opt/conda"),
                    Path("/usr/local/conda")
                ]
                for root in possible_roots:
                    if root.exists():
                        conda_root = str(root)
                        break
            
            if conda_root and Path(conda_root).exists():
                conda_paths.extend([
                    Path(conda_root) / "pkgs",
                    Path(conda_root) / "envs"
                ])
                
        except Exception as e:
            self.logger.debug(f"conda 경로 탐지 실패: {e}")
        
        return conda_paths

    def _init_cache_db(self):
        """캐시 데이터베이스 초기화"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS step_detection_cache (
                        step_name TEXT PRIMARY KEY,
                        checkpoint_path TEXT,
                        checkpoint_size INTEGER,
                        checkpoint_mtime REAL,
                        pytorch_valid INTEGER,
                        parameter_count INTEGER,
                        step_available INTEGER,
                        status TEXT,
                        detection_data TEXT,
                        created_at REAL,
                        accessed_at REAL
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_step_accessed_at ON step_detection_cache(accessed_at)
                """)
                
                conn.commit()
                
            self.logger.debug("✅ Step 탐지 캐시 DB 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"❌ Step 탐지 캐시 DB 초기화 실패: {e}")
            self.enable_caching = False

    def detect_all_steps(
        self, 
        force_rescan: bool = False,
        step_filter: Optional[List[str]] = None,
        min_confidence: float = 0.3
    ) -> Dict[str, StepCheckpointInfo]:
        """
        모든 Step의 체크포인트 자동 탐지
        
        Args:
            force_rescan: 캐시 무시하고 강제 재스캔
            step_filter: 특정 Step만 탐지
            min_confidence: 최소 신뢰도 임계값
            
        Returns:
            Dict[str, StepCheckpointInfo]: 탐지된 Step 정보들
        """
        try:
            self.logger.info("🎯 Step 기반 체크포인트 자동 탐지 시작...")
            start_time = time.time()
            
            # 캐시 확인
            if not force_rescan and self.enable_caching:
                cached_results = self._load_from_cache()
                if cached_results:
                    self.logger.info(f"📦 캐시된 결과 사용: {len(cached_results)}개 Step")
                    self.scan_stats["cache_hits"] = len(cached_results)
                    return cached_results
            
            # 실제 스캔 실행
            self._reset_scan_stats()
            
            # Step 필터링
            if step_filter:
                filtered_patterns = {k: v for k, v in STEP_CHECKPOINT_PATTERNS.items() 
                                   if k in step_filter}
            else:
                filtered_patterns = STEP_CHECKPOINT_PATTERNS
            
            # 병렬 스캔 실행
            if self.max_workers > 1:
                self._parallel_scan_steps(filtered_patterns, min_confidence)
            else:
                self._sequential_scan_steps(filtered_patterns, min_confidence)
            
            # Step 클래스 로드 가능성 검증
            if self.enable_step_loading:
                self._validate_step_loading()
            
            # 스캔 통계 업데이트
            self.scan_stats["steps_available"] = len([s for s in self.detected_steps.values() if s.step_available])
            self.scan_stats["scan_duration"] = time.time() - start_time
            self.scan_stats["last_scan_time"] = time.time()
            
            # 결과 후처리
            self._post_process_step_results(min_confidence)
            
            # 캐시 저장
            if self.enable_caching:
                self._save_to_cache()
            
            self.logger.info(f"✅ Step 탐지 완료: {len(self.detected_steps)}개 Step 발견 ({self.scan_stats['scan_duration']:.2f}초)")
            self._print_step_detection_summary()
            
            return self.detected_steps
            
        except Exception as e:
            self.logger.error(f"❌ Step 탐지 실패: {e}")
            self.scan_stats["errors_encountered"] += 1
            raise

    def _parallel_scan_steps(self, step_patterns: Dict, min_confidence: float):
        """Step들 병렬 스캔"""
        try:
            # 스캔 태스크 생성
            scan_tasks = []
            for step_name, pattern_info in step_patterns.items():
                for search_path in self.search_paths:
                    if search_path.exists():
                        scan_tasks.append((step_name, pattern_info, search_path))
            
            if not scan_tasks:
                self.logger.warning("⚠️ 스캔할 경로가 없습니다")
                return
            
            # ThreadPoolExecutor로 병렬 처리
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {
                    executor.submit(
                        self._scan_path_for_step, 
                        step_name, 
                        pattern_info, 
                        search_path, 
                        min_confidence
                    ): (step_name, search_path)
                    for step_name, pattern_info, search_path in scan_tasks
                }
                
                # 결과 수집
                completed_count = 0
                for future in as_completed(future_to_task, timeout=self.scan_timeout):
                    step_name, search_path = future_to_task[future]
                    try:
                        step_info = future.result()
                        if step_info:
                            # 결과 병합 (스레드 안전)
                            with threading.Lock():
                                self._register_step_safe(step_info)
                        
                        completed_count += 1
                        self.logger.debug(f"✅ {step_name} @ {search_path} 스캔 완료 ({completed_count}/{len(scan_tasks)})")
                        
                    except Exception as e:
                        self.logger.error(f"❌ {step_name} @ {search_path} 스캔 실패: {e}")
                        self.scan_stats["errors_encountered"] += 1
                        
        except Exception as e:
            self.logger.error(f"❌ 병렬 스캔 실패: {e}")
            # 폴백: 순차 스캔
            self._sequential_scan_steps(step_patterns, min_confidence)

    def _sequential_scan_steps(self, step_patterns: Dict, min_confidence: float):
        """Step들 순차 스캔"""
        try:
            for step_name, pattern_info in step_patterns.items():
                self.logger.debug(f"🎯 {step_name} 체크포인트 스캔 중...")
                
                best_checkpoint = None
                best_confidence = 0.0
                
                for search_path in self.search_paths:
                    if search_path.exists():
                        step_info = self._scan_path_for_step(
                            step_name, pattern_info, search_path, min_confidence
                        )
                        if step_info and step_info.confidence_score > best_confidence:
                            best_checkpoint = step_info
                            best_confidence = step_info.confidence_score
                
                if best_checkpoint:
                    self._register_step_safe(best_checkpoint)
                        
        except Exception as e:
            self.logger.error(f"❌ 순차 스캔 실패: {e}")

    def _scan_path_for_step(
        self, 
        step_name: str, 
        pattern_info: Dict, 
        search_path: Path, 
        min_confidence: float,
        max_depth: int = 6,
        current_depth: int = 0
    ) -> Optional[StepCheckpointInfo]:
        """특정 경로에서 Step 체크포인트 스캔"""
        try:
            if current_depth > max_depth:
                return None
            
            # 디렉토리 내용 나열
            try:
                items = list(search_path.iterdir())
            except PermissionError:
                self.logger.debug(f"권한 없음: {search_path}")
                return None
            
            # 파일과 디렉토리 분리
            files = [item for item in items if item.is_file()]
            subdirs = [item for item in items if item.is_dir() and not item.name.startswith('.')]
            
            best_checkpoint = None
            best_confidence = 0.0
            
            # 파일들 분석
            for file_path in files:
                try:
                    self.scan_stats["total_files_scanned"] += 1
                    
                    # Step 체크포인트 파일 확인
                    if self._is_step_checkpoint_file(file_path, pattern_info):
                        self.scan_stats["checkpoint_files_found"] += 1
                        
                        # Step 체크포인트 분석
                        step_info = self._analyze_step_checkpoint(
                            file_path, step_name, pattern_info, min_confidence
                        )
                        
                        if step_info and step_info.confidence_score > best_confidence:
                            best_checkpoint = step_info
                            best_confidence = step_info.confidence_score
                        
                except Exception as e:
                    self.logger.debug(f"파일 분석 오류 {file_path}: {e}")
                    continue
            
            # 하위 디렉토리 재귀 스캔
            if current_depth < max_depth:
                for subdir in subdirs:
                    # 제외할 디렉토리 패턴
                    if subdir.name in ['__pycache__', '.git', 'node_modules', '.vscode', '.idea']:
                        continue
                    
                    try:
                        subdir_result = self._scan_path_for_step(
                            step_name, pattern_info, subdir, min_confidence, 
                            max_depth, current_depth + 1
                        )
                        if subdir_result and subdir_result.confidence_score > best_confidence:
                            best_checkpoint = subdir_result
                            best_confidence = subdir_result.confidence_score
                    except Exception as e:
                        self.logger.debug(f"하위 디렉토리 스캔 오류 {subdir}: {e}")
                        continue
            
            return best_checkpoint
            
        except Exception as e:
            self.logger.debug(f"경로 스캔 오류 {search_path}: {e}")
            return None

    def _is_step_checkpoint_file(self, file_path: Path, pattern_info: Dict) -> bool:
        """Step 체크포인트 파일 확인"""
        try:
            # 확장자 체크
            if file_path.suffix.lower() not in pattern_info.get("file_extensions", [".pth", ".pt"]):
                return False
            
            # 파일 크기 체크
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            size_min, size_max = pattern_info.get("size_range_mb", (1, 10000))
            
            if not (size_min <= file_size_mb <= size_max):
                return False
            
            # 패턴 매칭
            file_path_str = str(file_path).lower()
            for pattern in pattern_info.get("checkpoint_patterns", []):
                if re.search(pattern, file_path_str, re.IGNORECASE):
                    return True
            
            # 디렉토리 기반 매칭
            path_parts = [part.lower() for part in file_path.parts]
            for expected_dir in pattern_info.get("expected_directories", []):
                if any(expected_dir.lower() in part for part in path_parts):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.debug(f"체크포인트 파일 확인 오류: {e}")
            return False

    def _analyze_step_checkpoint(
        self, 
        file_path: Path, 
        step_name: str,
        pattern_info: Dict,
        min_confidence: float
    ) -> Optional[StepCheckpointInfo]:
        """Step 체크포인트 파일 분석"""
        try:
            # 기본 파일 정보
            file_stat = file_path.stat()
            file_size_mb = file_stat.st_size / (1024 * 1024)
            last_modified = file_stat.st_mtime
            
            # 신뢰도 계산
            confidence_score = self._calculate_step_confidence(file_path, pattern_info, file_size_mb)
            
            if confidence_score < min_confidence:
                return None
            
            # PyTorch 체크포인트 검증
            pytorch_valid = False
            parameter_count = 0
            validation_info = {}
            
            if self.enable_pytorch_validation and file_path.suffix.lower() in ['.pth', '.pt']:
                pytorch_valid, parameter_count, validation_info = self._validate_pytorch_checkpoint(
                    file_path, pattern_info
                )
                
                if pytorch_valid:
                    self.scan_stats["valid_checkpoints"] += 1
                    # PyTorch 검증 성공하면 신뢰도 보너스
                    confidence_score = min(confidence_score + 0.2, 1.0)
                else:
                    self.scan_stats["pytorch_validation_errors"] += 1
                    # 검증 실패하면 신뢰도 감소
                    confidence_score = max(confidence_score - 0.3, 0.0)
            
            # Step 정보 생성
            step_info = StepCheckpointInfo(
                step_name=step_name,
                step_class_name=pattern_info.get("class_name", ""),
                checkpoint_path=file_path,
                checkpoint_size_mb=file_size_mb,
                pytorch_valid=pytorch_valid,
                parameter_count=parameter_count,
                last_modified=last_modified,
                step_available=False,  # 나중에 검증
                status=StepStatus.CHECKPOINT_MISSING,
                priority=pattern_info.get("priority", StepPriority.MEDIUM),
                confidence_score=confidence_score,
                metadata={
                    "file_name": file_path.name,
                    "detected_at": time.time(),
                    "auto_detected": True,
                    "pattern_matched": True,
                    **validation_info
                },
                validation_info=validation_info
            )
            
            return step_info
            
        except Exception as e:
            self.logger.debug(f"Step 체크포인트 분석 오류 {file_path}: {e}")
            return None

    def _validate_pytorch_checkpoint(self, file_path: Path, pattern_info: Dict) -> Tuple[bool, int, Dict[str, Any]]:
        """PyTorch 체크포인트 검증"""
        try:
            if not TORCH_AVAILABLE:
                return False, 0, {"error": "PyTorch not available"}
            
            # 안전한 체크포인트 로드
            try:
                checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
            except Exception as e:
                try:
                    checkpoint = torch.load(file_path, map_location='cpu')
                except Exception as e2:
                    return False, 0, {"load_error": str(e2)}
            
            validation_info = {}
            parameter_count = 0
            
            if isinstance(checkpoint, dict):
                # state_dict 확인
                state_dict = None
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    validation_info["has_state_dict"] = True
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                    validation_info["has_model"] = True
                else:
                    # 체크포인트 자체가 state_dict일 수 있음
                    state_dict = checkpoint
                    validation_info["is_direct_state_dict"] = True
                
                if state_dict and isinstance(state_dict, dict):
                    # 레이어 정보 분석
                    layers_info = self._analyze_checkpoint_layers(state_dict, pattern_info)
                    validation_info.update(layers_info)
                    
                    # 파라미터 수 계산
                    parameter_count = self._count_checkpoint_parameters(state_dict)
                    validation_info["parameter_count"] = parameter_count
                    
                    # 파라미터 수 범위 검증
                    expected_params = pattern_info.get("expected_parameters", (0, float('inf')))
                    if expected_params[0] <= parameter_count <= expected_params[1]:
                        validation_info["parameter_range_valid"] = True
                    else:
                        validation_info["parameter_range_valid"] = False
                
                # 메타데이터
                for key in ['epoch', 'version', 'arch', 'model_name']:
                    if key in checkpoint:
                        validation_info[f'checkpoint_{key}'] = str(checkpoint[key])[:100]
                
                return True, parameter_count, validation_info
            
            else:
                # 단순 텐서나 모델 객체
                if hasattr(checkpoint, 'state_dict'):
                    state_dict = checkpoint.state_dict()
                    parameter_count = self._count_checkpoint_parameters(state_dict)
                    return True, parameter_count, {"model_object": True}
                elif torch.is_tensor(checkpoint):
                    return True, checkpoint.numel(), {"single_tensor": True}
                else:
                    return False, 0, {"unknown_format": type(checkpoint).__name__}
            
        except Exception as e:
            return False, 0, {"validation_error": str(e)[:200]}

    def _analyze_checkpoint_layers(self, state_dict: Dict, pattern_info: Dict) -> Dict[str, Any]:
        """체크포인트 레이어 분석"""
        try:
            layers_info = {
                "total_layers": len(state_dict),
                "layer_types": {},
                "layer_names": list(state_dict.keys())[:10]  # 처음 10개만
            }
            
            # 레이어 타입 분석
            layer_type_counts = {}
            for key in state_dict.keys():
                if 'conv' in key.lower():
                    layer_type_counts['conv'] = layer_type_counts.get('conv', 0) + 1
                elif 'bn' in key.lower() or 'batch' in key.lower():
                    layer_type_counts['batch_norm'] = layer_type_counts.get('batch_norm', 0) + 1
                elif 'linear' in key.lower() or 'fc' in key.lower():
                    layer_type_counts['linear'] = layer_type_counts.get('linear', 0) + 1
                elif 'attention' in key.lower() or 'attn' in key.lower():
                    layer_type_counts['attention'] = layer_type_counts.get('attention', 0) + 1
            
            layers_info["layer_types"] = layer_type_counts
            
            # 예상 레이어 확인
            expected_layers = pattern_info.get("expected_layers", [])
            found_layers = 0
            for expected_layer in expected_layers:
                if any(expected_layer in key.lower() for key in state_dict.keys()):
                    found_layers += 1
            
            layers_info["expected_layers_found"] = found_layers
            layers_info["expected_layers_total"] = len(expected_layers)
            layers_info["layer_match_rate"] = found_layers / len(expected_layers) if expected_layers else 1.0
            
            return layers_info
            
        except Exception as e:
            return {"layer_analysis_error": str(e)[:100]}

    def _count_checkpoint_parameters(self, state_dict: Dict) -> int:
        """체크포인트 파라미터 수 계산"""
        try:
            total_params = 0
            for tensor in state_dict.values():
                if torch.is_tensor(tensor):
                    total_params += tensor.numel()
            return total_params
        except Exception:
            return 0

    def _calculate_step_confidence(self, file_path: Path, pattern_info: Dict, file_size_mb: float) -> float:
        """Step 체크포인트 신뢰도 계산"""
        try:
            score = 0.0
            file_name = file_path.name.lower()
            file_path_str = str(file_path).lower()
            
            # 정규식 패턴 매칭 점수
            for pattern in pattern_info.get("checkpoint_patterns", []):
                if re.search(pattern, file_path_str, re.IGNORECASE):
                    score += 30.0
                    break
            
            # 디렉토리 매칭
            path_parts = [part.lower() for part in file_path.parts]
            for expected_dir in pattern_info.get("expected_directories", []):
                if any(expected_dir.lower() in part for part in path_parts):
                    score += 25.0
                    break
            
            # 파일 크기 적정성
            size_min, size_max = pattern_info.get("size_range_mb", (1, 10000))
            size_mid = (size_min + size_max) / 2
            
            if size_min <= file_size_mb <= size_max:
                if abs(file_size_mb - size_mid) / size_mid < 0.5:
                    score += 20.0
                else:
                    score += 15.0
            
            # 파일 확장자
            if file_path.suffix in pattern_info.get("file_extensions", []):
                score += 10.0
            
            # 우선순위 보너스
            priority = pattern_info.get("priority", StepPriority.MEDIUM)
            if priority == StepPriority.CRITICAL:
                score += 10.0
            elif priority == StepPriority.HIGH:
                score += 5.0
            
            # 정규화 (0.0 ~ 1.0)
            confidence = min(score / 100.0, 1.0)
            return max(confidence, 0.0)
            
        except Exception as e:
            self.logger.debug(f"신뢰도 계산 오류: {e}")
            return 0.0

    def _validate_step_loading(self):
        """Step 클래스 로드 가능성 검증"""
        try:
            self.logger.info("🔍 Step 클래스 로드 가능성 검증 중...")
            
            for step_name, step_info in self.detected_steps.items():
                try:
                    # Step 클래스 로드 시도
                    step_available = self._test_step_class_loading(step_name, step_info.step_class_name)
                    
                    step_info.step_available = step_available
                    
                    if step_available:
                        if step_info.pytorch_valid:
                            step_info.status = StepStatus.AVAILABLE
                        else:
                            step_info.status = StepStatus.CHECKPOINT_MISSING
                    else:
                        step_info.status = StepStatus.LOADING_FAILED
                        
                except Exception as e:
                    self.logger.debug(f"Step {step_name} 로드 검증 실패: {e}")
                    step_info.step_available = False
                    step_info.status = StepStatus.NOT_FOUND
            
        except Exception as e:
            self.logger.error(f"❌ Step 로드 검증 실패: {e}")

    def _test_step_class_loading(self, step_name: str, step_class_name: str) -> bool:
        """Step 클래스 로드 테스트"""
        try:
            # steps 모듈에서 Step 클래스 로드 시도
            from ..steps import get_step_class
from app.utils.safe_caller import safe_call, safe_warmup
            
            step_class = get_step_class(step_name)
            
            if step_class is None:
                self.logger.debug(f"❌ {step_name} Step 클래스를 찾을 수 없음")
                return False
            
            # 클래스명 확인
            if step_class.__name__ != step_class_name:
                self.logger.debug(f"⚠️ {step_name} 클래스명 불일치: {step_class.__name__} != {step_class_name}")
            
            self.logger.debug(f"✅ {step_name} Step 클래스 로드 가능")
            return True
            
        except Exception as e:
            self.logger.debug(f"❌ {step_name} Step 클래스 로드 실패: {e}")
            return False

    def _register_step_safe(self, step_info: StepCheckpointInfo):
        """스레드 안전한 Step 등록"""
        with threading.Lock():
            self._register_step(step_info)

    def _register_step(self, step_info: StepCheckpointInfo):
        """Step 정보 등록 (중복 처리)"""
        try:
            step_name = step_info.step_name
            
            if step_name in self.detected_steps:
                existing_step = self.detected_steps[step_name]
                
                # 더 나은 체크포인트로 교체할지 결정
                if self._is_better_checkpoint(step_info, existing_step):
                    step_info.alternative_checkpoints.append(existing_step.checkpoint_path)
                    step_info.alternative_checkpoints.extend(existing_step.alternative_checkpoints)
                    self.detected_steps[step_name] = step_info
                    self.logger.debug(f"🔄 Step 교체: {step_name}")
                else:
                    existing_step.alternative_checkpoints.append(step_info.checkpoint_path)
                    self.logger.debug(f"📎 대체 체크포인트 추가: {step_name}")
            else:
                self.detected_steps[step_name] = step_info
                self.logger.debug(f"✅ 새 Step 등록: {step_name}")
                
        except Exception as e:
            self.logger.error(f"❌ Step 등록 실패: {e}")

    def _is_better_checkpoint(self, new_step: StepCheckpointInfo, existing_step: StepCheckpointInfo) -> bool:
        """새 체크포인트가 기존 것보다 나은지 판단"""
        try:
            # 1. PyTorch 검증 상태 우선
            if new_step.pytorch_valid and not existing_step.pytorch_valid:
                return True
            elif not new_step.pytorch_valid and existing_step.pytorch_valid:
                return False
            
            # 2. 우선순위 비교 (낮은 값이 높은 우선순위)
            if new_step.priority.value < existing_step.priority.value:
                return True
            elif new_step.priority.value > existing_step.priority.value:
                return False
            
            # 3. 신뢰도 비교
            if abs(new_step.confidence_score - existing_step.confidence_score) > 0.1:
                return new_step.confidence_score > existing_step.confidence_score
            
            # 4. 파라미터 수 비교
            if new_step.parameter_count > 0 and existing_step.parameter_count > 0:
                if abs(new_step.parameter_count - existing_step.parameter_count) / max(new_step.parameter_count, existing_step.parameter_count) > 0.2:
                    return new_step.parameter_count > existing_step.parameter_count
            
            # 5. 최신성 비교
            if abs(new_step.last_modified - existing_step.last_modified) > 86400:  # 1일 차이
                return new_step.last_modified > existing_step.last_modified
            
            # 6. 파일 크기 비교
            return new_step.checkpoint_size_mb > existing_step.checkpoint_size_mb
            
        except Exception as e:
            self.logger.debug(f"체크포인트 비교 오류: {e}")
            return new_step.checkpoint_size_mb > existing_step.checkpoint_size_mb

    def _reset_scan_stats(self):
        """스캔 통계 리셋"""
        self.scan_stats.update({
            "total_files_scanned": 0,
            "checkpoint_files_found": 0,
            "valid_checkpoints": 0,
            "steps_available": 0,
            "scan_duration": 0.0,
            "errors_encountered": 0,
            "pytorch_validation_errors": 0
        })

    def _post_process_step_results(self, min_confidence: float):
        """Step 결과 후처리"""
        try:
            # 신뢰도 필터링
            filtered_steps = {
                name: step for name, step in self.detected_steps.items()
                if step.confidence_score >= min_confidence
            }
            self.detected_steps = filtered_steps
            
            # 우선순위에 따른 정렬
            sorted_steps = sorted(
                self.detected_steps.items(),
                key=lambda x: (
                    x[1].priority.value, 
                    -x[1].confidence_score, 
                    -x[1].parameter_count, 
                    -x[1].checkpoint_size_mb
                )
            )
            
            self.detected_steps = {name: step for name, step in sorted_steps}
            
        except Exception as e:
            self.logger.error(f"❌ 후처리 실패: {e}")

    def _print_step_detection_summary(self):
        """Step 탐지 결과 요약 출력"""
        try:
            self.logger.info("=" * 70)
            self.logger.info("🎯 Step 기반 체크포인트 탐지 결과 요약")
            self.logger.info("=" * 70)
            
            total_size_gb = sum(step.checkpoint_size_mb for step in self.detected_steps.values()) / 1024
            total_params = sum(step.parameter_count for step in self.detected_steps.values())
            avg_confidence = sum(step.confidence_score for step in self.detected_steps.values()) / len(self.detected_steps) if self.detected_steps else 0
            available_count = sum(1 for step in self.detected_steps.values() if step.step_available)
            pytorch_valid_count = sum(1 for step in self.detected_steps.values() if step.pytorch_valid)
            
            self.logger.info(f"📊 탐지된 Step: {len(self.detected_steps)}개")
            self.logger.info(f"✅ 로드 가능한 Step: {available_count}개")
            self.logger.info(f"🔍 검증된 체크포인트: {pytorch_valid_count}개")
            self.logger.info(f"💾 총 크기: {total_size_gb:.2f}GB")
            self.logger.info(f"🔍 스캔 파일: {self.scan_stats['total_files_scanned']:,}개")
            self.logger.info(f"📦 체크포인트 파일: {self.scan_stats['checkpoint_files_found']}개")
            self.logger.info(f"⏱️ 소요 시간: {self.scan_stats['scan_duration']:.2f}초")
            self.logger.info(f"🎯 평균 신뢰도: {avg_confidence:.3f}")
            self.logger.info(f"🧮 총 파라미터: {total_params:,}개")
            
            # Step별 상태 분포
            status_distribution = {}
            for step in self.detected_steps.values():
                status = step.status.value
                if status not in status_distribution:
                    status_distribution[status] = 0
                status_distribution[status] += 1
            
            if status_distribution:
                self.logger.info("\n📋 Step 상태별 분포:")
                for status, count in status_distribution.items():
                    self.logger.info(f"  {status}: {count}개")
            
            # 주요 Step들
            if self.detected_steps:
                self.logger.info("\n🏆 탐지된 주요 Step들:")
                for i, (name, step) in enumerate(list(self.detected_steps.items())[:8]):
                    status_icon = "✅" if step.step_available else ("🔍" if step.pytorch_valid else "❓")
                    self.logger.info(f"  {i+1}. {name}")
                    self.logger.info(f"     상태: {status_icon} {step.status.value}")
                    self.logger.info(f"     체크포인트: {step.checkpoint_path.name if step.checkpoint_path else 'None'}")
                    self.logger.info(f"     크기: {step.checkpoint_size_mb:.1f}MB, 신뢰도: {step.confidence_score:.3f}")
                    if step.parameter_count > 0:
                        self.logger.info(f"     파라미터: {step.parameter_count:,}개")
            
            self.logger.info("=" * 70)
                
        except Exception as e:
            self.logger.error(f"❌ 요약 출력 실패: {e}")

    # ==============================================
    # 🔥 캐시 관련 메서드들
    # ==============================================

    def _load_from_cache(self) -> Optional[Dict[str, StepCheckpointInfo]]:
        """캐시에서 로드"""
        try:
            with self._cache_lock:
                with sqlite3.connect(self.cache_db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 만료된 캐시 정리
                    cutoff_time = time.time() - self.cache_ttl
                    cursor.execute("DELETE FROM step_detection_cache WHERE created_at < ?", (cutoff_time,))
                    
                    # 캐시 조회
                    cursor.execute("""
                        SELECT step_name, detection_data
                        FROM step_detection_cache 
                        WHERE created_at > ?
                    """, (cutoff_time,))
                    
                    cached_steps = {}
                    for step_name, detection_data in cursor.fetchall():
                        try:
                            step_data = json.loads(detection_data)
                            step_info = self._deserialize_step_info(step_data)
                            if step_info:
                                # 파일이 여전히 존재하는지 확인
                                if step_info.checkpoint_path and step_info.checkpoint_path.exists():
                                    cached_steps[step_name] = step_info
                        except Exception as e:
                            self.logger.debug(f"캐시 항목 로드 실패 {step_name}: {e}")
                    
                    if cached_steps:
                        # 액세스 시간 업데이트
                        cursor.execute("UPDATE step_detection_cache SET accessed_at = ?", (time.time(),))
                        conn.commit()
                        
                        self.detected_steps = cached_steps
                        return cached_steps
            
            return None
            
        except Exception as e:
            self.logger.debug(f"캐시 로드 실패: {e}")
            return None

    def _save_to_cache(self):
        """캐시에 저장"""
        try:
            with self._cache_lock:
                with sqlite3.connect(self.cache_db_path) as conn:
                    cursor = conn.cursor()
                    current_time = time.time()
                    
                    for step in self.detected_steps.values():
                        try:
                            detection_data = json.dumps(self._serialize_step_info(step))
                            
                            cursor.execute("""
                                INSERT OR REPLACE INTO step_detection_cache 
                                (step_name, checkpoint_path, checkpoint_size, checkpoint_mtime, 
                                 pytorch_valid, parameter_count, step_available, status, 
                                 detection_data, created_at, accessed_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                step.step_name,
                                str(step.checkpoint_path) if step.checkpoint_path else None,
                                int(step.checkpoint_size_mb * 1024 * 1024),
                                step.last_modified,
                                int(step.pytorch_valid),
                                step.parameter_count,
                                int(step.step_available),
                                step.status.value,
                                detection_data,
                                current_time,
                                current_time
                            ))
                        except Exception as e:
                            self.logger.debug(f"Step 캐시 저장 실패 {step.step_name}: {e}")
                    
                    conn.commit()
                    
        except Exception as e:
            self.logger.debug(f"캐시 저장 실패: {e}")

    def _serialize_step_info(self, step_info: StepCheckpointInfo) -> Dict[str, Any]:
        """StepCheckpointInfo를 딕셔너리로 직렬화"""
        return {
            "step_name": step_info.step_name,
            "step_class_name": step_info.step_class_name,
            "checkpoint_path": str(step_info.checkpoint_path) if step_info.checkpoint_path else None,
            "checkpoint_size_mb": step_info.checkpoint_size_mb,
            "pytorch_valid": step_info.pytorch_valid,
            "parameter_count": step_info.parameter_count,
            "last_modified": step_info.last_modified,
            "step_available": step_info.step_available,
            "status": step_info.status.value,
            "priority": step_info.priority.value,
            "alternative_checkpoints": [str(p) for p in step_info.alternative_checkpoints],
            "metadata": step_info.metadata,
            "validation_info": step_info.validation_info,
            "confidence_score": step_info.confidence_score
        }

    def _deserialize_step_info(self, data: Dict[str, Any]) -> Optional[StepCheckpointInfo]:
        """딕셔너리를 StepCheckpointInfo로 역직렬화"""
        try:
            return StepCheckpointInfo(
                step_name=data["step_name"],
                step_class_name=data["step_class_name"],
                checkpoint_path=Path(data["checkpoint_path"]) if data.get("checkpoint_path") else None,
                checkpoint_size_mb=data["checkpoint_size_mb"],
                pytorch_valid=data["pytorch_valid"],
                parameter_count=data["parameter_count"],
                last_modified=data["last_modified"],
                step_available=data["step_available"],
                status=StepStatus(data["status"]),
                priority=StepPriority(data["priority"]),
                alternative_checkpoints=[Path(p) for p in data.get("alternative_checkpoints", [])],
                metadata=data.get("metadata", {}),
                validation_info=data.get("validation_info", {}),
                confidence_score=data["confidence_score"]
            )
        except Exception as e:
            self.logger.debug(f"Step 역직렬화 실패: {e}")
            return None

    # ==============================================
    # 🔥 공개 조회 메서드들
    # ==============================================

    def get_available_steps(self) -> List[StepCheckpointInfo]:
        """로드 가능한 Step들 반환"""
        return [step for step in self.detected_steps.values() if step.step_available]

    def get_steps_by_priority(self, priority: StepPriority) -> List[StepCheckpointInfo]:
        """우선순위별 Step들 반환"""
        return [step for step in self.detected_steps.values() if step.priority == priority]

    def get_step_by_name(self, step_name: str) -> Optional[StepCheckpointInfo]:
        """이름으로 Step 조회"""
        return self.detected_steps.get(step_name)

    def get_critical_steps(self) -> List[StepCheckpointInfo]:
        """필수 Step들 반환"""
        return self.get_steps_by_priority(StepPriority.CRITICAL)

    def get_validated_steps(self) -> Dict[str, StepCheckpointInfo]:
        """PyTorch 검증된 Step들만 반환"""
        return {name: step for name, step in self.detected_steps.items() if step.pytorch_valid}

    def get_all_step_paths(self) -> Dict[str, Path]:
        """모든 Step의 체크포인트 경로 딕셔너리 반환"""
        return {name: step.checkpoint_path for name, step in self.detected_steps.items() 
                if step.checkpoint_path}

    def check_pipeline_readiness(self) -> Dict[str, Any]:
        """파이프라인 준비 상태 확인"""
        try:
            critical_steps = self.get_critical_steps()
            available_critical = [step for step in critical_steps if step.step_available]
            
            total_steps = len(STEP_CHECKPOINT_PATTERNS)
            available_steps = len(self.get_available_steps())
            validated_steps = len(self.get_validated_steps())
            
            readiness = {
                "pipeline_ready": len(available_critical) >= len(critical_steps),
                "critical_steps_ready": len(available_critical),
                "critical_steps_total": len(critical_steps),
                "total_steps_available": available_steps,
                "total_steps_possible": total_steps,
                "validated_steps": validated_steps,
                "readiness_score": available_steps / total_steps if total_steps > 0 else 0,
                "missing_critical_steps": [step.step_name for step in critical_steps if not step.step_available],
                "summary": f"{available_steps}/{total_steps} Steps available, {validated_steps} validated"
            }
            
            return readiness
            
        except Exception as e:
            self.logger.error(f"파이프라인 준비 상태 확인 실패: {e}")
            return {"pipeline_ready": False, "error": str(e)}

# ==============================================
# 🔥 PipelineManager 연동 설정 생성기
# ==============================================

class StepPipelineConfigGenerator:
    """
    🔗 Step 탐지 결과를 PipelineManager 설정으로 변환
    ✅ PipelineManager 완전 호환
    ✅ Step 클래스 기반 설정 생성
    ✅ 순환참조 방지
    """
    
    def __init__(self, detector: StepBasedDetector):
        self.detector = detector
        self.logger = logging.getLogger(f"{__name__}.StepPipelineConfigGenerator")
    
    def generate_pipeline_config(self) -> Dict[str, Any]:
        """PipelineManager용 완전한 설정 생성"""
        try:
            config = {
                "steps": [],
                "step_mappings": {},
                "checkpoint_paths": {},
                "priority_rankings": {},
                "pipeline_metadata": {
                    "total_steps": len(self.detector.detected_steps),
                    "available_steps": len(self.detector.get_available_steps()),
                    "validated_steps": len(self.detector.get_validated_steps()),
                    "generation_time": time.time(),
                    "detector_version": "1.0",
                    "scan_stats": self.detector.scan_stats
                },
                "readiness_check": self.detector.check_pipeline_readiness()
            }
            
            for step_name, step_info in self.detector.detected_steps.items():
                # Step 설정 딕셔너리 생성
                step_config = {
                    "step_name": step_name,
                    "step_class": step_info.step_class_name,
                    "checkpoint_path": str(step_info.checkpoint_path) if step_info.checkpoint_path else None,
                    "device": self.detector.device,  # 🔥 탐지기의 디바이스 사용
                    "precision": "fp16" if self.detector.device_info["supports_fp16"] else "fp32",
                    "step_available": step_info.step_available,
                    "pytorch_validated": step_info.pytorch_valid,
                    "parameter_count": step_info.parameter_count,
                    "priority": step_info.priority.name,
                    "status": step_info.status.value,
                    "confidence_score": step_info.confidence_score,
                    "input_size": self._get_input_size_for_step(step_name),
                    "device_optimized": True,  # 🔥 디바이스 최적화 적용됨
                    "metadata": {
                        **step_info.metadata,
                        "auto_detected": True,
                        "checkpoint_size_mb": step_info.checkpoint_size_mb,
                        "alternative_checkpoints": [str(p) for p in step_info.alternative_checkpoints],
                        "validation_info": step_info.validation_info,
                        "device_config": self.detector.get_optimal_config_for_device()  # 🔥 디바이스 설정 포함
                    }
                }
                config["steps"].append(step_config)
                
                # Step 매핑
                config["step_mappings"][step_name] = {
                    "class_name": step_info.step_class_name,
                    "available": step_info.step_available,
                    "priority": step_info.priority.value
                }
                
                # 체크포인트 경로
                if step_info.checkpoint_path:
                    config["checkpoint_paths"][step_name] = {
                        "primary": str(step_info.checkpoint_path),
                        "alternatives": [str(p) for p in step_info.alternative_checkpoints],
                        "size_mb": step_info.checkpoint_size_mb,
                        "validated": step_info.pytorch_valid
                    }
                
                # 우선순위 랭킹
                config["priority_rankings"][step_name] = {
                    "priority_level": step_info.priority.value,
                    "priority_name": step_info.priority.name,
                    "confidence_score": step_info.confidence_score,
                    "step_available": step_info.step_available,
                    "parameter_count": step_info.parameter_count
                }
            
            return config
            
        except Exception as e:
            self.logger.error(f"❌ PipelineManager 설정 생성 실패: {e}")
            return {"error": str(e)}

    def _get_input_size_for_step(self, step_name: str) -> Tuple[int, int]:
        """Step별 기본 입력 크기"""
        size_mapping = {
            "step_01_human_parsing": (512, 512),
            "step_02_pose_estimation": (368, 368),
            "step_03_cloth_segmentation": (320, 320),
            "step_04_geometric_matching": (512, 384),
            "step_05_cloth_warping": (512, 384),
            "step_06_virtual_fitting": (512, 512),
            "step_07_post_processing": (512, 512),
            "step_08_quality_assessment": (224, 224)
        }
        return size_mapping.get(step_name, (512, 512))

    def generate_model_loader_compatible_config(self) -> Dict[str, Any]:
        """기존 ModelLoader 호환 설정 생성"""
        try:
            model_configs = []
            
            for step_name, step_info in self.detector.detected_steps.items():
                if not step_info.step_available:
                    continue
                
                # ModelLoader 호환 형태로 변환
                model_config = {
                    "name": f"{step_name}_model",
                    "model_type": step_info.step_class_name,
                    "checkpoint_path": str(step_info.checkpoint_path) if step_info.checkpoint_path else None,
                    "device": "auto",
                    "precision": "fp16",
                    "step_name": step_name,
                    "step_class": step_info.step_class_name,
                    "pytorch_validated": step_info.pytorch_valid,
                    "parameter_count": step_info.parameter_count,
                    "priority": step_info.priority.name,
                    "metadata": {
                        **step_info.metadata,
                        "step_based": True,
                        "auto_detected": True
                    }
                }
                model_configs.append(model_config)
            
            return {
                "model_configs": model_configs,
                "total_models": len(model_configs),
                "step_based_detection": True,
                "generation_time": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"❌ ModelLoader 호환 설정 생성 실패: {e}")
            return {"error": str(e)}

# ==============================================
# 🔥 편의 함수들 및 팩토리 함수들
# ==============================================

def create_step_based_detector(
    search_paths: Optional[List[Path]] = None,
    enable_pytorch_validation: bool = True,
    enable_step_loading: bool = True,
    max_workers: int = 4,
    device: Optional[str] = None,  # 🔥 디바이스 선택적 지정
    **kwargs
) -> StepBasedDetector:
    """Step 기반 탐지기 생성 팩토리"""
    return StepBasedDetector(
        search_paths=search_paths,
        enable_pytorch_validation=enable_pytorch_validation,
        enable_step_loading=enable_step_loading,
        max_workers=max_workers,
        device=device,  # 🔥 디바이스 전달
        **kwargs
    )

def quick_step_detection(
    step_filter: Optional[List[str]] = None,
    min_confidence: float = 0.5,
    force_rescan: bool = False,
    validated_only: bool = False
) -> Dict[str, Any]:
    """빠른 Step 탐지 및 결과 반환"""
    try:
        # 탐지기 생성 및 실행
        detector = create_step_based_detector()
        detected_steps = detector.detect_all_steps(
            force_rescan=force_rescan,
            step_filter=step_filter,
            min_confidence=min_confidence
        )
        
        # 검증된 Step만 필터링 (옵션)
        if validated_only:
            detected_steps = detector.get_validated_steps()
        
        # 결과 요약
        summary = {
            "total_steps": len(detected_steps),
            "available_steps": len([s for s in detected_steps.values() if s.step_available]),
            "validated_steps": len([s for s in detected_steps.values() if s.pytorch_valid]),
            "steps_by_priority": {},
            "steps_by_status": {},
            "top_steps": {},
            "pipeline_readiness": detector.check_pipeline_readiness(),
            "scan_stats": detector.scan_stats
        }
        
        # 우선순위별 분류
        for step in detected_steps.values():
            priority = step.priority.name
            if priority not in summary["steps_by_priority"]:
                summary["steps_by_priority"][priority] = []
            summary["steps_by_priority"][priority].append({
                "name": step.step_name,
                "class": step.step_class_name,
                "available": step.step_available,
                "confidence": step.confidence_score,
                "checkpoint_path": str(step.checkpoint_path) if step.checkpoint_path else None
            })
        
        # 상태별 분류
        for step in detected_steps.values():
            status = step.status.value
            if status not in summary["steps_by_status"]:
                summary["steps_by_status"][status] = []
            summary["steps_by_status"][status].append(step.step_name)
        
        # 우선순위별 최고 Step
        priorities = set(step.priority for step in detected_steps.values())
        for priority in priorities:
            priority_steps = [s for s in detected_steps.values() if s.priority == priority]
            if priority_steps:
                best_step = max(priority_steps, key=lambda s: (s.step_available, s.pytorch_valid, s.confidence_score))
                summary["top_steps"][priority.name] = {
                    "name": best_step.step_name,
                    "class": best_step.step_class_name,
                    "available": best_step.step_available,
                    "confidence": best_step.confidence_score,
                    "parameter_count": best_step.parameter_count,
                    "checkpoint_path": str(best_step.checkpoint_path) if best_step.checkpoint_path else None
                }
        
        return summary
        
    except Exception as e:
        logger.error(f"빠른 Step 탐지 실패: {e}")
        return {"error": str(e)}

def generate_pipeline_config_from_steps(
    detector: Optional[StepBasedDetector] = None,
    **detection_kwargs
) -> Dict[str, Any]:
    """
    Step 탐지 결과로부터 PipelineManager 설정 생성
    순환참조 방지하며 딕셔너리 기반 출력
    """
    try:
        logger.info("🎯 Step 기반 PipelineManager 설정 생성 시작...")
        
        # 탐지기가 없으면 새로 생성
        if detector is None:
            detector = create_step_based_detector(**detection_kwargs)
            detected_steps = detector.detect_all_steps()
        else:
            detected_steps = detector.detected_steps
        
        if not detected_steps:
            logger.warning("⚠️ 탐지된 Step이 없습니다")
            return {"success": False, "message": "No steps detected"}
        
        # 설정 생성기 사용
        config_generator = StepPipelineConfigGenerator(detector)
        pipeline_config = config_generator.generate_pipeline_safe_call(config)
        model_loader_config = config_generator.generate_model_loader_compatible_safe_call(config)
        
        # 최종 결과
        available_count = len(detector.get_available_steps())
        validated_count = len(detector.get_validated_steps())
        
        result = {
            "success": True,
            "pipeline_config": pipeline_config,
            "model_loader_config": model_loader_config,
            "detection_summary": {
                "total_steps": len(detected_steps),
                "available_steps": available_count,
                "validated_steps": validated_count,
                "availability_rate": available_count / len(detected_steps) if detected_steps else 0,
                "validation_rate": validated_count / len(detected_steps) if detected_steps else 0,
                "scan_duration": detector.scan_stats["scan_duration"],
                "confidence_avg": sum(s.confidence_score for s in detected_steps.values()) / len(detected_steps),
                "total_parameters": sum(s.parameter_count for s in detected_steps.values())
            },
            "readiness_check": detector.check_pipeline_readiness()
        }
        
        logger.info(f"✅ Step 기반 설정 생성 완료: {len(detected_steps)}개 Step ({available_count}개 사용 가능)")
        return result
        
    except Exception as e:
        logger.error(f"❌ Step 기반 설정 생성 실패: {e}")
        return {"success": False, "error": str(e)}

def validate_step_checkpoints(detected_steps: Dict[str, StepCheckpointInfo]) -> Dict[str, Any]:
    """Step 체크포인트들의 유효성 검증"""
    try:
        validation_result = {
            "valid_steps": [],
            "invalid_steps": [],
            "missing_checkpoints": [],
            "corrupted_checkpoints": [],
            "available_steps": [],
            "unavailable_steps": [],
            "total_size_gb": 0.0,
            "total_parameters": 0
        }
        
        for step_name, step_info in detected_steps.items():
            try:
                # 체크포인트 파일 확인
                if step_info.checkpoint_path and step_info.checkpoint_path.exists():
                    validation_result["valid_steps"].append(step_name)
                    validation_result["total_size_gb"] += step_info.checkpoint_size_mb / 1024
                    validation_result["total_parameters"] += step_info.parameter_count
                    
                    # Step 클래스 사용 가능성 확인
                    if step_info.step_available:
                        validation_result["available_steps"].append(step_name)
                    else:
                        validation_result["unavailable_steps"].append(step_name)
                        
                    # PyTorch 검증 상태 확인
                    if not step_info.pytorch_valid:
                        validation_result["corrupted_checkpoints"].append({
                            "step": step_name,
                            "path": str(step_info.checkpoint_path),
                            "reason": "PyTorch validation failed"
                        })
                else:
                    validation_result["missing_checkpoints"].append({
                        "step": step_name,
                        "expected_path": str(step_info.checkpoint_path) if step_info.checkpoint_path else "Unknown"
                    })
                
            except Exception as e:
                validation_result["invalid_steps"].append({
                    "step": step_name,
                    "error": str(e)
                })
        
        validation_result["summary"] = {
            "total_steps": len(detected_steps),
            "valid_count": len(validation_result["valid_steps"]),
            "invalid_count": len(validation_result["invalid_steps"]),
            "missing_count": len(validation_result["missing_checkpoints"]),
            "corrupted_count": len(validation_result["corrupted_checkpoints"]),
            "available_count": len(validation_result["available_steps"]),
            "validation_rate": len(validation_result["valid_steps"]) / len(detected_steps) if detected_steps else 0,
            "availability_rate": len(validation_result["available_steps"]) / len(detected_steps) if detected_steps else 0
        }
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Step 체크포인트 검증 실패: {e}")
        return {"error": str(e)}

def integrate_with_pipeline_manager(
    pipeline_manager_instance = None,
    auto_configure: bool = True,
    **detection_kwargs
) -> Dict[str, Any]:
    """Step 탐지 및 PipelineManager 통합 (순환참조 방지)"""
    try:
        logger.info("🎯 Step 탐지 및 PipelineManager 통합 시작...")
        
        # 탐지 실행
        detector = create_step_based_detector(**detection_kwargs)
        detected_steps = detector.detect_all_steps()
        
        if not detected_steps:
            logger.warning("⚠️ 탐지된 Step이 없습니다")
            return {"success": False, "message": "No steps detected"}
        
        # 설정 생성기
        config_generator = StepPipelineConfigGenerator(detector)
        
        # PipelineManager 설정 생성
        pipeline_config = config_generator.generate_pipeline_safe_call(config)
        
        # PipelineManager와 통합 (순환참조 방지)
        integration_result = {}
        if auto_configure and pipeline_manager_instance:
            try:
                # PipelineManager 설정 적용
                if hasattr(pipeline_manager_instance, 'configure_from_detection'):
                    pipeline_manager_instance.configure_from_detection(pipeline_config)
                    integration_result["configuration_applied"] = True
                elif hasattr(pipeline_manager_instance, 'update_config'):
                    pipeline_manager_instance.update_config(pipeline_config)
                    integration_result["configuration_updated"] = True
                else:
                    logger.warning("⚠️ PipelineManager에 설정 메서드가 없습니다")
                    integration_result["configuration_method_missing"] = True
                
            except Exception as e:
                logger.warning(f"⚠️ PipelineManager 통합 실패: {e}")
                integration_result["integration_error"] = str(e)
        
        readiness_check = detector.check_pipeline_readiness()
        
        return {
            "success": True,
            "detected_count": len(detected_steps),
            "available_count": len(detector.get_available_steps()),
            "step_names": list(detected_steps.keys()),
            "integration": integration_result,
            "config": pipeline_config,
            "readiness": readiness_check,
            "pipeline_ready": readiness_check.get("pipeline_ready", False)
        }
        
    except Exception as e:
        logger.error(f"❌ 탐지 및 통합 실패: {e}")
        return {"success": False, "error": str(e)}

# ==============================================
# 🔥 하위 호환성을 위한 어댑터 클래스들
# ==============================================

class StepToModelLoaderAdapter:
    """
    기존 ModelLoader와 호환성을 위한 어댑터
    Step 탐지 결과를 ModelLoader 형태로 변환
    """
    
    def __init__(self, detector: StepBasedDetector):
        self.detector = detector
        self.logger = logging.getLogger(f"{__name__}.StepToModelLoaderAdapter")
    
    def get_model_configs(self) -> List[Dict[str, Any]]:
        """ModelLoader가 기대하는 형태로 설정 반환"""
        try:
            config_generator = StepPipelineConfigGenerator(self.detector)
            model_loader_config = config_generator.generate_model_loader_compatible_safe_call(config)
            return model_loader_config.get("model_configs", [])
        except Exception as e:
            self.logger.error(f"모델 설정 생성 실패: {e}")
            return []
    
    def register_steps_to_loader(self, model_loader):
        """탐지된 Step들을 ModelLoader에 등록"""
        try:
            detected_steps = self.detector.detected_steps
            registered_count = 0
            
            for step_name, step_info in detected_steps.items():
                if not step_info.step_available:
                    continue
                
                try:
                    # ModelLoader 호환 설정 생성
                    model_config = {
                        "name": f"{step_name}_model",
                        "model_type": step_info.step_class_name,
                        "checkpoint_path": str(step_info.checkpoint_path) if step_info.checkpoint_path else None,
                        "device": "auto",
                        "precision": "fp16",
                        "step_based": True,
                        "step_name": step_name,
                        "pytorch_validated": step_info.pytorch_valid,
                        "parameter_count": step_info.parameter_count,
                        "confidence_score": step_info.confidence_score
                    }
                    
                    # ModelLoader에 등록
                    if hasattr(model_loader, 'register_model'):
                        model_loader.register_model(f"{step_name}_model", model_config)
                        registered_count += 1
                    elif hasattr(model_loader, '_register_model'):
                        model_loader._register_model(f"{step_name}_model", model_config)
                        registered_count += 1
                        
                except Exception as e:
                    self.logger.warning(f"Step {step_name} 등록 실패: {e}")
            
            self.logger.info(f"✅ {registered_count}개 Step 등록 완료")
            return registered_count
            
        except Exception as e:
            self.logger.error(f"Step 등록 프로세스 실패: {e}")
            return 0

# 모듈 익스포트
__all__ = [
    # 핵심 클래스
    'StepBasedDetector',
    'StepPipelineConfigGenerator', 
    'StepToModelLoaderAdapter',
    
    # 데이터 구조
    'StepCheckpointInfo',
    'StepStatus',
    'StepPriority',
    
    # 설정 패턴
    'STEP_CHECKPOINT_PATTERNS',
    
    # 팩토리 함수들
    'create_step_based_detector',
    'quick_step_detection',
    'generate_pipeline_config_from_steps',
    
    # 유틸리티 함수들
    'validate_step_checkpoints',
    'integrate_with_pipeline_manager',
    
    # 하위 호환성 별칭 (기존 코드와의 호환성)
    'StepDetector',
    'create_step_detector',
    'quick_detection',
    'generate_config_from_steps'
]

# 하위 호환성을 위한 별칭
StepDetector = StepBasedDetector
create_step_detector = create_step_based_detector
quick_detection = quick_step_detection
generate_config_from_steps = generate_pipeline_config_from_steps

logger.info("✅ Step 기반 자동 탐지 시스템 v1.0 로드 완료")
logger.info("🎯 Step 클래스 = AI 모델 + 처리 로직 통합 구조 완벽 지원")
logger.info("🔗 PipelineManager 완전 연동 및 기존 ModelLoader 호환성 유지")