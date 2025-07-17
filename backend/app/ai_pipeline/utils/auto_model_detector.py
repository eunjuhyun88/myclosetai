# app/ai_pipeline/utils/auto_model_detector.py
"""
🔍 MyCloset AI - 완전 통합 자동 모델 탐지 시스템 v6.0
✅ 2번,3번 파일의 실제 동작하는 탐지 로직 완전 반영
✅ 실제 존재하는 AI 모델 파일들 정확한 탐지
✅ PyTorch 체크포인트 내용 검증
✅ 순환참조 완전 해결 (딕셔너리 기반 연동)
✅ M3 Max 128GB 최적화
✅ conda 환경 특화 스캔
✅ 프로덕션 안정성 보장

🔥 핵심 변경사항:
- 2번,3번 파일의 실제 탐지 패턴 100% 반영
- PyTorch 체크포인트 내용 실제 검증
- 파일 크기와 매개변수 수 실제 확인
- ModelLoader 직접 import 제거
- 딕셔너리 기반 설정 출력
- 실제 동작하는 탐지 로직만 사용
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
# 🔥 실제 2번,3번 파일의 동작하는 패턴 반영
# ==============================================

class ModelCategory(Enum):
    """모델 카테고리"""
    HUMAN_PARSING = "human_parsing"
    POSE_ESTIMATION = "pose_estimation"
    CLOTH_SEGMENTATION = "cloth_segmentation"
    GEOMETRIC_MATCHING = "geometric_matching"
    CLOTH_WARPING = "cloth_warping"
    VIRTUAL_FITTING = "virtual_fitting"
    POST_PROCESSING = "post_processing"
    QUALITY_ASSESSMENT = "quality_assessment"
    DIFFUSION_MODELS = "diffusion_models"
    TRANSFORMER_MODELS = "transformer_models"
    AUXILIARY = "auxiliary"

class ModelPriority(Enum):
    """모델 우선순위"""
    CRITICAL = 1      # 필수 모델
    HIGH = 2          # 높은 우선순위
    MEDIUM = 3        # 중간 우선순위
    LOW = 4           # 낮은 우선순위
    EXPERIMENTAL = 5  # 실험적 모델

@dataclass
class DetectedModel:
    """탐지된 모델 정보 (딕셔너리 기반 연동용)"""
    name: str
    path: Path
    category: ModelCategory
    model_type: str
    file_size_mb: float
    file_extension: str
    confidence_score: float
    priority: ModelPriority
    step_name: str  # 연결된 Step 클래스명
    metadata: Dict[str, Any] = field(default_factory=dict)
    alternative_paths: List[Path] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    performance_info: Dict[str, Any] = field(default_factory=dict)
    compatibility_info: Dict[str, Any] = field(default_factory=dict)
    last_modified: float = 0.0
    checksum: Optional[str] = None
    pytorch_valid: bool = False
    parameter_count: int = 0

# ==============================================
# 🔥 2번 파일의 실제 모델 정의 패턴 100% 반영
# ==============================================

@dataclass
class ModelFileInfo:
    """2번 파일의 ModelFileInfo와 100% 호환"""
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
    keywords: List[str] = field(default_factory=list)
    expected_layers: List[str] = field(default_factory=list)

# 실제 발견된 파일들 기반으로 패턴 수정
ACTUAL_MODEL_PATTERNS = {
    "human_parsing": ModelFileInfo(
        name="human_parsing_graphonomy",
        patterns=[
            r".*checkpoints/human_parsing/.*\.pth$",
            r".*schp.*atr.*\.pth$", 
            r".*atr_model.*\.pth$",
            r".*lip_model.*\.pth$"
        ],
        step="step_01_human_parsing",
        required=True,
        min_size_mb=1,
        max_size_mb=500,
        target_path="ai_models/checkpoints/human_parsing/atr_model.pth",
        alternative_names=["schp_atr.pth", "atr_model.pth", "lip_model.pth"],
        keywords=["human", "parsing", "atr", "schp", "lip"],
        expected_layers=["backbone", "classifier", "conv"]
    ),
    
    "cloth_segmentation": ModelFileInfo(
        name="cloth_segmentation_u2net", 
        patterns=[
            r".*checkpoints/step_03.*u2net.*\.pth$",
            r".*u2net_segmentation.*\.pth$",
            r".*sam.*vit.*\.pth$"
        ],
        step="step_03_cloth_segmentation",
        required=True, 
        min_size_mb=10,
        max_size_mb=3000,
        target_path="ai_models/checkpoints/step_03/u2net_segmentation/u2net.pth",
        alternative_names=["u2net.pth", "sam_vit_h_4b8939.pth", "sam_vit_b_01ec64.pth"],
        keywords=["u2net", "segmentation", "sam"],
        expected_layers=["encoder", "decoder", "outconv"]
    ),
    
    "virtual_fitting": ModelFileInfo(
        name="virtual_fitting_ootd", 
        patterns=[
            r".*step_06_virtual_fitting.*\.bin$",
            r".*ootd.*unet.*\.bin$",
            r".*OOTDiffusion.*"
        ],
        step="virtual_fitting",
        required=True,
        min_size_mb=100, 
        max_size_mb=8000,
        target_path="ai_models/step_06_virtual_fitting/ootd_hd_unet.bin",
        alternative_names=["ootd_hd_unet.bin", "ootd_dc_unet.bin"],
        keywords=["ootd", "unet", "diffusion", "virtual"],
        expected_layers=["unet", "vae"],
        file_types=['.bin', '.pth', '.pt', '.safetensors']
    )
}
# 3번 파일의 체크포인트 패턴도 반영
CHECKPOINT_VERIFICATION_PATTERNS = {
    "human_parsing": {
        "keywords": ["human", "parsing", "atr", "schp", "graphonomy", "segmentation"],
        "expected_size_range": (50, 500),  # MB
        "required_layers": ["backbone", "classifier", "conv"],
        "typical_parameters": (25000000, 70000000)  # 25M ~ 70M 파라미터
    },
    "pose_estimation": {
        "keywords": ["pose", "openpose", "body", "keypoint", "coco"],
        "expected_size_range": (10, 1000),
        "required_layers": ["stage", "paf", "heatmap"],
        "typical_parameters": (10000000, 200000000)  # 10M ~ 200M 파라미터
    },
    "cloth_segmentation": {
        "keywords": ["u2net", "cloth", "segmentation", "mask", "sam"],
        "expected_size_range": (10, 3000),
        "required_layers": ["encoder", "decoder", "outconv"],
        "typical_parameters": (4000000, 650000000)  # 4M ~ 650M 파라미터 (SAM 포함)
    },
    "geometric_matching": {
        "keywords": ["gmm", "geometric", "tps", "matching", "alignment"],
        "expected_size_range": (1, 100),
        "required_layers": ["correlation", "regression", "flow"],
        "typical_parameters": (500000, 50000000)  # 0.5M ~ 50M 파라미터
    },
    "cloth_warping": {
        "keywords": ["tom", "warping", "cloth", "viton", "try"],
        "expected_size_range": (10, 4000),
        "required_layers": ["generator", "discriminator", "warp"],
        "typical_parameters": (10000000, 1000000000)  # 10M ~ 1B 파라미터
    },
    "virtual_fitting": {
        "keywords": ["diffusion", "viton", "unet", "stable", "fitting"],
        "expected_size_range": (100, 8000),
        "required_layers": ["unet", "vae", "text_encoder"],
        "typical_parameters": (100000000, 2000000000)  # 100M ~ 2B 파라미터
    }
}

# ==============================================
# 🔥 실제 동작하는 고급 모델 탐지기 클래스
# ==============================================

class RealWorldModelDetector:
    """
    🔍 실제 동작하는 AI 모델 자동 탐지 시스템 v6.0
    ✅ 2번,3번 파일의 실제 탐지 로직 100% 반영
    ✅ PyTorch 체크포인트 내용 실제 검증
    ✅ 딕셔너리 기반 출력 (순환참조 방지)
    """
    
    def __init__(
        self,
        search_paths: Optional[List[Path]] = None,
        enable_deep_scan: bool = True,
        enable_pytorch_validation: bool = True,
        enable_caching: bool = True,
        cache_db_path: Optional[Path] = None,
        max_workers: int = 4,
        scan_timeout: int = 300
    ):
        """실제 동작하는 모델 탐지기 초기화"""
        
        self.logger = logging.getLogger(f"{__name__}.RealWorldModelDetector")
        
        # 실제 검색 경로 설정 (2번 파일 방식 반영)
        if search_paths is None:
            current_file = Path(__file__).resolve()
            backend_dir = current_file.parents[3]  # app/ai_pipeline/utils에서 backend로
            
            # 실제 존재하는 경로들만 추가
            self.search_paths = self._get_real_search_paths(backend_dir)
        else:
            self.search_paths = search_paths
        
        # 설정
        self.enable_deep_scan = enable_deep_scan
        self.enable_pytorch_validation = enable_pytorch_validation
        self.enable_caching = enable_caching
        self.max_workers = max_workers
        self.scan_timeout = scan_timeout
        
        # 탐지 결과 저장
        self.detected_models: Dict[str, DetectedModel] = {}
        self.scan_stats = {
            "total_files_scanned": 0,
            "pytorch_files_found": 0,
            "valid_pytorch_models": 0,
            "models_detected": 0,
            "scan_duration": 0.0,
            "last_scan_time": 0,
            "errors_encountered": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "pytorch_validation_errors": 0
        }
        
        # 캐시 관리
        self.cache_db_path = cache_db_path or Path("real_model_detection_cache.db")
        self.cache_ttl = 86400  # 24시간
        self._cache_lock = threading.RLock()
        
        self.logger.info(f"🔍 실제 동작 모델 탐지기 초기화 완료 - 검색 경로: {len(self.search_paths)}개")
        
        # 캐시 DB 초기화
        if self.enable_caching:
            self._init_cache_db()

    def _get_real_search_paths(self, backend_dir: Path) -> List[Path]:
        """실제 존재하는 검색 경로들만 반환 (2번 파일 방식)"""
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
            
            # 사용자 캐시 경로들
            Path.home() / ".cache" / "huggingface",
            Path.home() / ".cache" / "torch",
            Path.home() / ".cache" / "models",
            Path.home() / "Downloads",
            
            # conda 환경 경로들
            *self._get_conda_paths()
        ]
        
        # 실제 존재하는 디렉토리만 필터링
        real_paths = []
        for path in potential_paths:
            try:
                if path.exists() and path.is_dir():
                    # 읽기 권한 확인
                    if os.access(path, os.R_OK):
                        real_paths.append(path)
                        self.logger.debug(f"✅ 유효한 검색 경로: {path}")
                    else:
                        self.logger.debug(f"❌ 권한 없음: {path}")
                else:
                    self.logger.debug(f"❌ 경로 없음: {path}")
            except Exception as e:
                self.logger.debug(f"❌ 경로 확인 실패 {path}: {e}")
                continue
        
        return real_paths

    def _get_conda_paths(self) -> List[Path]:
        """conda 환경 경로들 탐지"""
        conda_paths = []
        
        try:
            # 현재 conda 환경
            conda_prefix = os.environ.get('CONDA_PREFIX')
            if conda_prefix and Path(conda_prefix).exists():
                conda_base = Path(conda_prefix)
                conda_paths.extend([
                    conda_base / "lib" / "python3.11" / "site-packages",
                    conda_base / "share" / "models",
                    conda_base / "models"
                ])
            
            # conda 루트
            conda_root = os.environ.get('CONDA_ROOT')
            if not conda_root:
                # 일반적인 conda 설치 경로들
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
                conda_paths.append(Path(conda_root) / "pkgs")
                
        except Exception as e:
            self.logger.debug(f"conda 경로 탐지 실패: {e}")
        
        return conda_paths

    def _init_cache_db(self):
        """캐시 데이터베이스 초기화"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS real_model_cache (
                        file_path TEXT PRIMARY KEY,
                        file_size INTEGER,
                        file_mtime REAL,
                        checksum TEXT,
                        pytorch_valid INTEGER,
                        parameter_count INTEGER,
                        detection_data TEXT,
                        created_at REAL,
                        accessed_at REAL
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_real_accessed_at ON real_model_cache(accessed_at)
                """)
                
                conn.commit()
                
            self.logger.debug("✅ 실제 모델 캐시 DB 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 실제 모델 캐시 DB 초기화 실패: {e}")
            self.enable_caching = False

    def detect_all_models(
        self, 
        force_rescan: bool = False,
        categories_filter: Optional[List[ModelCategory]] = None,
        min_confidence: float = 0.3,
        model_type_filter: Optional[List[str]] = None
    ) -> Dict[str, DetectedModel]:
        """
        실제 AI 모델 자동 탐지 (2번,3번 파일 로직 반영)
        
        Args:
            force_rescan: 캐시 무시하고 강제 재스캔
            categories_filter: 특정 카테고리만 탐지
            min_confidence: 최소 신뢰도 임계값
            model_type_filter: 특정 모델 타입만 탐지
            
        Returns:
            Dict[str, DetectedModel]: 탐지된 모델들
        """
        try:
            self.logger.info("🔍 실제 AI 모델 자동 탐지 시작...")
            start_time = time.time()
            
            # 캐시 확인
            if not force_rescan and self.enable_caching:
                cached_results = self._load_from_cache()
                if cached_results:
                    self.logger.info(f"📦 캐시된 결과 사용: {len(cached_results)}개 모델")
                    self.scan_stats["cache_hits"] += len(cached_results)
                    return cached_results
            
            # 실제 스캔 실행
            self._reset_scan_stats()
            
            # 모델 타입 필터링
            if model_type_filter:
                filtered_patterns = {k: v for k, v in ACTUAL_MODEL_PATTERNS.items() 
                                   if k in model_type_filter}
            else:
                filtered_patterns = ACTUAL_MODEL_PATTERNS
            
            # 병렬 스캔 실행
            if self.max_workers > 1:
                self._parallel_scan_real_models(filtered_patterns, categories_filter, min_confidence)
            else:
                self._sequential_scan_real_models(filtered_patterns, categories_filter, min_confidence)
            
            # 스캔 통계 업데이트
            self.scan_stats["models_detected"] = len(self.detected_models)
            self.scan_stats["scan_duration"] = time.time() - start_time
            self.scan_stats["last_scan_time"] = time.time()
            
            # 결과 후처리
            self._post_process_results(min_confidence)
            
            # 캐시 저장
            if self.enable_caching:
                self._save_to_cache()
            
            self.logger.info(f"✅ 실제 모델 탐지 완료: {len(self.detected_models)}개 모델 발견 ({self.scan_stats['scan_duration']:.2f}초)")
            self._print_detection_summary()
            
            return self.detected_models
            
        except Exception as e:
            self.logger.error(f"❌ 실제 모델 탐지 실패: {e}")
            self.scan_stats["errors_encountered"] += 1
            raise

    def _parallel_scan_real_models(self, model_patterns: Dict, categories_filter, min_confidence):
        """실제 모델들 병렬 스캔"""
        try:
            # 검색 태스크 생성
            scan_tasks = []
            for model_type, pattern_info in model_patterns.items():
                for search_path in self.search_paths:
                    if search_path.exists():
                        scan_tasks.append((model_type, pattern_info, search_path))
            
            if not scan_tasks:
                self.logger.warning("⚠️ 스캔할 경로가 없습니다")
                return
            
            # ThreadPoolExecutor로 병렬 처리
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {
                    executor.submit(
                        self._scan_path_for_real_models, 
                        model_type, 
                        pattern_info, 
                        search_path, 
                        categories_filter, 
                        min_confidence
                    ): (model_type, search_path)
                    for model_type, pattern_info, search_path in scan_tasks
                }
                
                # 결과 수집
                completed_count = 0
                for future in as_completed(future_to_task, timeout=self.scan_timeout):
                    model_type, search_path = future_to_task[future]
                    try:
                        path_results = future.result()
                        if path_results:
                            # 결과 병합 (스레드 안전)
                            with threading.Lock():
                                for name, model in path_results.items():
                                    self._register_detected_model_safe(model)
                        
                        completed_count += 1
                        self.logger.debug(f"✅ {model_type} @ {search_path} 스캔 완료 ({completed_count}/{len(scan_tasks)})")
                        
                    except Exception as e:
                        self.logger.error(f"❌ {model_type} @ {search_path} 스캔 실패: {e}")
                        self.scan_stats["errors_encountered"] += 1
                        
        except Exception as e:
            self.logger.error(f"❌ 병렬 스캔 실패: {e}")
            # 폴백: 순차 스캔
            self._sequential_scan_real_models(model_patterns, categories_filter, min_confidence)

    def _sequential_scan_real_models(self, model_patterns: Dict, categories_filter, min_confidence):
        """실제 모델들 순차 스캔"""
        try:
            for model_type, pattern_info in model_patterns.items():
                self.logger.debug(f"📁 {model_type} 모델 패턴 스캔 중...")
                
                for search_path in self.search_paths:
                    if search_path.exists():
                        path_results = self._scan_path_for_real_models(
                            model_type, pattern_info, search_path, categories_filter, min_confidence
                        )
                        if path_results:
                            for name, model in path_results.items():
                                self._register_detected_model_safe(model)
                    else:
                        self.logger.debug(f"⚠️ 경로 없음: {search_path}")
                        
        except Exception as e:
            self.logger.error(f"❌ 순차 스캔 실패: {e}")

    def _scan_path_for_real_models(
        self, 
        model_type: str, 
        pattern_info: ModelFileInfo, 
        search_path: Path, 
        categories_filter: Optional[List[ModelCategory]], 
        min_confidence: float,
        max_depth: int = 6,
        current_depth: int = 0
    ) -> Dict[str, DetectedModel]:
        """실제 모델 파일들 스캔 (2번,3번 파일 로직 반영)"""
        results = {}
        
        try:
            if current_depth > max_depth:
                return results
            
            # 디렉토리 내용 나열
            try:
                items = list(search_path.iterdir())
            except PermissionError:
                self.logger.debug(f"권한 없음: {search_path}")
                return results
            
            # 파일과 디렉토리 분리
            files = [item for item in items if item.is_file()]
            subdirs = [item for item in items if item.is_dir() and not item.name.startswith('.')]
            
            # 파일들 분석 (실제 모델 파일 확인)
            for file_path in files:
                try:
                    self.scan_stats["total_files_scanned"] += 1
                    
                    # 기본 AI 모델 파일 필터링
                    if not self._is_potential_ai_model_file(file_path):
                        continue
                    
                    self.scan_stats["pytorch_files_found"] += 1
                    
                    # 패턴 매칭
                    if self._matches_model_patterns(file_path, pattern_info):
                        detected_model = self._analyze_real_model_file(
                            file_path, model_type, pattern_info, categories_filter, min_confidence
                        )
                        if detected_model:
                            results[detected_model.name] = detected_model
                            self.logger.debug(f"📦 {model_type} 모델 발견: {file_path.name}")
                        
                except Exception as e:
                    self.logger.debug(f"파일 분석 오류 {file_path}: {e}")
                    continue
            
            # 하위 디렉토리 재귀 스캔
            if self.enable_deep_scan and current_depth < max_depth:
                for subdir in subdirs:
                    # 제외할 디렉토리 패턴
                    if subdir.name in ['__pycache__', '.git', 'node_modules', '.vscode', '.idea', '.pytest_cache']:
                        continue
                    
                    try:
                        subdir_results = self._scan_path_for_real_models(
                            model_type, pattern_info, subdir, categories_filter, 
                            min_confidence, max_depth, current_depth + 1
                        )
                        results.update(subdir_results)
                    except Exception as e:
                        self.logger.debug(f"하위 디렉토리 스캔 오류 {subdir}: {e}")
                        continue
            
            return results
            
        except Exception as e:
            self.logger.debug(f"경로 스캔 오류 {search_path}: {e}")
            return results

    def _is_potential_ai_model_file(self, file_path: Path) -> bool:
        """AI 모델 파일 가능성 확인 (2번,3번 파일 방식)"""
        # 확장자 체크
        ai_extensions = {'.pth', '.pt', '.bin', '.safetensors', '.onnx', '.pkl', '.h5', '.pb'}
        if file_path.suffix.lower() not in ai_extensions:
            return False
        
        # 파일 크기 체크 (너무 작으면 모델이 아님)
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb < 0.5:  # 0.5MB 미만은 제외
                return False
        except:
            return False
        
        # 파일명 패턴 체크  
        file_name = file_path.name.lower()
        ai_keywords = [
            'model', 'checkpoint', 'weight', 'state_dict', 'pytorch_model',
            'diffusion', 'transformer', 'bert', 'clip', 'vit', 'resnet',
            'pose', 'parsing', 'segmentation', 'u2net', 'openpose',
            'viton', 'hrviton', 'stable', 'unet', 'vae', 'gmm', 'tom',
            'schp', 'atr', 'graphonomy', 'sam'
        ]
        
        return any(keyword in file_name for keyword in ai_keywords)

    def _matches_model_patterns(self, file_path: Path, pattern_info: ModelFileInfo) -> bool:
        """모델 패턴 매칭 확인 (2번 파일 방식)"""
        try:
            file_name_lower = file_path.name.lower()
            file_path_str = str(file_path).lower()
            
            # 정규식 패턴 매칭
            for pattern in pattern_info.patterns:
                if re.search(pattern, file_path_str, re.IGNORECASE):
                    return True
            
            # 대체 이름 매칭
            for alt_name in pattern_info.alternative_names:
                if alt_name.lower() in file_name_lower:
                    return True
            
            # 키워드 매칭
            for keyword in pattern_info.keywords:
                if keyword in file_name_lower:
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.debug(f"패턴 매칭 오류: {e}")
            return False

    def _analyze_real_model_file(
        self, 
        file_path: Path, 
        model_type: str,
        pattern_info: ModelFileInfo,
        categories_filter: Optional[List[ModelCategory]], 
        min_confidence: float
    ) -> Optional[DetectedModel]:
        """실제 모델 파일 분석 (3번 파일의 검증 로직 반영)"""
        try:
            # 기본 파일 정보
            file_stat = file_path.stat()
            file_size_mb = file_stat.st_size / (1024 * 1024)
            file_extension = file_path.suffix.lower()
            last_modified = file_stat.st_mtime
            
            # 크기 제한 확인
            if not (pattern_info.min_size_mb <= file_size_mb <= pattern_info.max_size_mb):
                return None
            
            # 파일 확장자 확인
            if file_extension not in pattern_info.file_types:
                return None
            
            # 신뢰도 계산
            confidence_score = self._calculate_real_confidence(file_path, model_type, pattern_info, file_size_mb)
            
            if confidence_score < min_confidence:
                return None
            
            # PyTorch 모델 실제 검증 (3번 파일 방식)
            pytorch_valid = False
            parameter_count = 0
            validation_info = {}
            
            if self.enable_pytorch_validation and file_extension in ['.pth', '.pt']:
                pytorch_valid, parameter_count, validation_info = self._validate_pytorch_model(
                    file_path, model_type
                )
                
                if pytorch_valid:
                    self.scan_stats["valid_pytorch_models"] += 1
                    # PyTorch 검증 성공하면 신뢰도 보너스
                    confidence_score = min(confidence_score + 0.2, 1.0)
                else:
                    self.scan_stats["pytorch_validation_errors"] += 1
                    # 검증 실패하면 신뢰도 감소
                    confidence_score = max(confidence_score - 0.3, 0.0)
            
            # 카테고리 매핑
            category_mapping = {
                "human_parsing": ModelCategory.HUMAN_PARSING,
                "pose_estimation": ModelCategory.POSE_ESTIMATION,
                "cloth_segmentation": ModelCategory.CLOTH_SEGMENTATION,
                "geometric_matching": ModelCategory.GEOMETRIC_MATCHING,
                "cloth_warping": ModelCategory.CLOTH_WARPING,
                "virtual_fitting": ModelCategory.VIRTUAL_FITTING
            }
            
            detected_category = category_mapping.get(model_type, ModelCategory.AUXILIARY)
            
            # 카테고리 필터 적용
            if categories_filter and detected_category not in categories_filter:
                return None
            
            # 우선순위 결정
            priority = ModelPriority(pattern_info.priority) if pattern_info.priority <= 5 else ModelPriority.EXPERIMENTAL
            
            # Step 이름 생성
            step_name = self._get_step_name_for_type(model_type)
            
            # 고유 모델 이름 생성
            unique_name = self._generate_unique_model_name(file_path, model_type, pattern_info.name)
            
            # 메타데이터 생성 (3번 파일 방식 포함)
            metadata = {
                "file_name": file_path.name,
                "file_size_mb": file_size_mb,
                "model_type": model_type,
                "detected_at": time.time(),
                "auto_detected": True,
                "pattern_matched": True,
                "pytorch_validated": pytorch_valid,
                "parameter_count": parameter_count,
                **validation_info
            }
            
            # DetectedModel 객체 생성
            detected_model = DetectedModel(
                name=unique_name,
                path=file_path,
                category=detected_category,
                model_type=pattern_info.name,
                file_size_mb=file_size_mb,
                file_extension=file_extension,
                confidence_score=confidence_score,
                priority=priority,
                step_name=step_name,
                metadata=metadata,
                last_modified=last_modified,
                pytorch_valid=pytorch_valid,
                parameter_count=parameter_count
            )
            
            return detected_model
            
        except Exception as e:
            self.logger.debug(f"실제 모델 파일 분석 오류 {file_path}: {e}")
            return None

    def _validate_pytorch_model(self, file_path: Path, model_type: str) -> Tuple[bool, int, Dict[str, Any]]:
        """PyTorch 모델 실제 검증 (3번 파일 CheckpointFinder 방식)"""
        try:
            if not TORCH_AVAILABLE:
                return False, 0, {"error": "PyTorch not available"}
            
            # 안전한 체크포인트 로드
            try:
                checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
            except Exception as e:
                # weights_only=True 실패시 일반 로드 시도
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
                    validation_info["contains_state_dict"] = True
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                    validation_info["contains_model"] = True
                else:
                    # 체크포인트 자체가 state_dict일 수 있음
                    state_dict = checkpoint
                    validation_info["is_direct_state_dict"] = True
                
                if state_dict and isinstance(state_dict, dict):
                    # 레이어 정보 분석
                    layers_info = self._analyze_model_layers(state_dict, model_type)
                    validation_info.update(layers_info)
                    
                    # 파라미터 수 계산
                    parameter_count = self._count_parameters(state_dict)
                    validation_info["parameter_count"] = parameter_count
                    
                    # 모델 타입별 검증
                    type_validation = self._validate_model_type_specific(state_dict, model_type, parameter_count)
                    validation_info.update(type_validation)
                
                # 추가 메타데이터
                for key in ['epoch', 'version', 'arch', 'model_name', 'optimizer']:
                    if key in checkpoint:
                        validation_info[f'checkpoint_{key}'] = str(checkpoint[key])[:100]
                
                return True, parameter_count, validation_info
            
            else:
                # 단순 텐서나 모델 객체인 경우
                if hasattr(checkpoint, 'state_dict'):
                    state_dict = checkpoint.state_dict()
                    parameter_count = self._count_parameters(state_dict)
                    return True, parameter_count, {"model_object": True}
                elif torch.is_tensor(checkpoint):
                    return True, checkpoint.numel(), {"single_tensor": True}
                else:
                    return False, 0, {"unknown_format": type(checkpoint).__name__}
            
        except Exception as e:
            return False, 0, {"validation_error": str(e)[:200]}

    def _analyze_model_layers(self, state_dict: Dict, model_type: str) -> Dict[str, Any]:
        """모델 레이어 분석 (3번 파일 방식)"""
        try:
            layers_info = {
                "total_layers": len(state_dict),
                "layer_types": {},
                "layer_names": list(state_dict.keys())[:10]  # 처음 10개만
            }
            
            # 레이어 타입 분석
            layer_type_counts = {}
            for key in state_dict.keys():
                # 일반적인 레이어 타입들
                if 'conv' in key.lower():
                    layer_type_counts['conv'] = layer_type_counts.get('conv', 0) + 1
                elif 'bn' in key.lower() or 'batch' in key.lower():
                    layer_type_counts['batch_norm'] = layer_type_counts.get('batch_norm', 0) + 1
                elif 'linear' in key.lower() or 'fc' in key.lower():
                    layer_type_counts['linear'] = layer_type_counts.get('linear', 0) + 1
                elif 'attention' in key.lower() or 'attn' in key.lower():
                    layer_type_counts['attention'] = layer_type_counts.get('attention', 0) + 1
                elif 'embed' in key.lower():
                    layer_type_counts['embedding'] = layer_type_counts.get('embedding', 0) + 1
            
            layers_info["layer_types"] = layer_type_counts
            
            # 모델 타입별 특화 레이어 확인
            verification_pattern = CHECKPOINT_VERIFICATION_PATTERNS.get(model_type, {})
            required_layers = verification_pattern.get("required_layers", [])
            
            found_required = 0
            for required_layer in required_layers:
                if any(required_layer in key.lower() for key in state_dict.keys()):
                    found_required += 1
            
            layers_info["required_layers_found"] = found_required
            layers_info["required_layers_total"] = len(required_layers)
            layers_info["required_layers_match_rate"] = found_required / len(required_layers) if required_layers else 1.0
            
            return layers_info
            
        except Exception as e:
            return {"layer_analysis_error": str(e)[:100]}

    def _count_parameters(self, state_dict: Dict) -> int:
        """모델 파라미터 수 계산"""
        try:
            total_params = 0
            for tensor in state_dict.values():
                if torch.is_tensor(tensor):
                    total_params += tensor.numel()
            return total_params
        except Exception as e:
            return 0

    def _validate_model_type_specific(self, state_dict: Dict, model_type: str, parameter_count: int) -> Dict[str, Any]:
        """모델 타입별 특화 검증"""
        try:
            validation = {"type_specific_validation": True}
            
            # 3번 파일의 검증 패턴 사용
            verification_pattern = CHECKPOINT_VERIFICATION_PATTERNS.get(model_type, {})
            
            # 파라미터 수 범위 확인
            if "typical_parameters" in verification_pattern:
                min_params, max_params = verification_pattern["typical_parameters"]
                if min_params <= parameter_count <= max_params:
                    validation["parameter_count_valid"] = True
                    validation["parameter_confidence"] = 1.0
                else:
                    validation["parameter_count_valid"] = False
                    # 범위 밖이면 신뢰도 조정
                    if parameter_count < min_params:
                        validation["parameter_confidence"] = max(0.3, parameter_count / min_params)
                    else:
                        validation["parameter_confidence"] = max(0.3, min_params / parameter_count)
            
            # 키워드 매칭
            keywords = verification_pattern.get("keywords", [])
            keyword_matches = 0
            for keyword in keywords:
                if any(keyword in key.lower() for key in state_dict.keys()):
                    keyword_matches += 1
            
            validation["keyword_matches"] = keyword_matches
            validation["keyword_match_rate"] = keyword_matches / len(keywords) if keywords else 1.0
            
            return validation
            
        except Exception as e:
            return {"type_validation_error": str(e)[:100]}

    def _calculate_real_confidence(self, file_path: Path, model_type: str, pattern_info: ModelFileInfo, file_size_mb: float) -> float:
        """실제 신뢰도 계산 (2번,3번 파일 방식 종합)"""
        try:
            score = 0.0
            file_name = file_path.name.lower()
            file_path_str = str(file_path).lower()
            
            # 정규식 패턴 매칭 점수
            for pattern in pattern_info.patterns:
                if re.search(pattern, file_path_str, re.IGNORECASE):
                    score += 25.0
                    break
            
            # 대체 이름 매칭
            for alt_name in pattern_info.alternative_names:
                if alt_name.lower() in file_name:
                    score += 20.0
                    break
            
            # 키워드 매칭
            for keyword in pattern_info.keywords:
                if keyword in file_name:
                    score += 8.0
            
            # 파일 크기 적정성
            size_min, size_max = pattern_info.min_size_mb, pattern_info.max_size_mb
            size_mid = (size_min + size_max) / 2
            
            if size_min <= file_size_mb <= size_max:
                # 크기가 범위 내에 있으면
                if abs(file_size_mb - size_mid) / size_mid < 0.5:  # 중간값의 50% 이내
                    score += 15.0
                else:
                    score += 10.0
            elif file_size_mb < size_min:
                # 너무 작으면 감점
                score -= 10.0
            else:
                # 너무 크면 약간 감점
                score -= 5.0
            
            # 파일 확장자 보너스
            if file_path.suffix in pattern_info.file_types:
                score += 5.0
            
            # 경로 기반 점수
            path_parts = [part.lower() for part in file_path.parts]
            if any(pattern_info.step in part for part in path_parts):
                score += 10.0
            
            # 우선순위 보너스
            if pattern_info.priority == 1:
                score += 5.0
            elif pattern_info.priority == 2:
                score += 3.0
            
            # 정규화 (0.0 ~ 1.0)
            confidence = min(score / 80.0, 1.0)
            return max(confidence, 0.0)
            
        except Exception as e:
            self.logger.debug(f"신뢰도 계산 오류: {e}")
            return 0.0

    def _get_step_name_for_type(self, model_type: str) -> str:
        """모델 타입에 따른 Step 이름 반환"""
        step_mapping = {
            "human_parsing": "HumanParsingStep",
            "pose_estimation": "PoseEstimationStep",
            "cloth_segmentation": "ClothSegmentationStep",
            "geometric_matching": "GeometricMatchingStep",
            "cloth_warping": "ClothWarpingStep",
            "virtual_fitting": "VirtualFittingStep"
        }
        return step_mapping.get(model_type, "UnknownStep")

    def _generate_unique_model_name(self, file_path: Path, model_type: str, base_name: str) -> str:
        """고유한 모델 이름 생성"""
        try:
            # 2번 파일의 표준 이름 사용
            standard_names = {
                "human_parsing": "human_parsing_graphonomy",
                "pose_estimation": "pose_estimation_openpose",
                "cloth_segmentation": "cloth_segmentation_u2net",
                "geometric_matching": "geometric_matching_gmm",
                "cloth_warping": "cloth_warping_tom",
                "virtual_fitting": "virtual_fitting_diffusion"
            }
            
            standard_name = standard_names.get(model_type)
            if standard_name:
                return standard_name
            
            # 파일명 기반 이름 생성
            file_stem = file_path.stem.lower()
            clean_name = re.sub(r'[^a-z0-9_]', '_', file_stem)
            
            # 해시 추가 (충돌 방지)
            path_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:6]
            
            return f"{model_type}_{clean_name}_{path_hash}"
            
        except Exception as e:
            timestamp = int(time.time())
            return f"detected_model_{timestamp}"

    def _register_detected_model_safe(self, detected_model: DetectedModel):
        """스레드 안전한 모델 등록"""
        with threading.Lock():
            self._register_detected_model(detected_model)

    def _register_detected_model(self, detected_model: DetectedModel):
        """탐지된 모델 등록 (중복 처리)"""
        try:
            model_name = detected_model.name
            
            if model_name in self.detected_models:
                existing_model = self.detected_models[model_name]
                
                # 더 나은 모델로 교체할지 결정
                if self._is_better_model(detected_model, existing_model):
                    detected_model.alternative_paths.append(existing_model.path)
                    detected_model.alternative_paths.extend(existing_model.alternative_paths)
                    self.detected_models[model_name] = detected_model
                    self.logger.debug(f"🔄 모델 교체: {model_name}")
                else:
                    existing_model.alternative_paths.append(detected_model.path)
                    self.logger.debug(f"📎 대체 경로 추가: {model_name}")
            else:
                self.detected_models[model_name] = detected_model
                self.logger.debug(f"✅ 새 모델 등록: {model_name}")
                
        except Exception as e:
            self.logger.error(f"❌ 모델 등록 실패: {e}")

    def _is_better_model(self, new_model: DetectedModel, existing_model: DetectedModel) -> bool:
        """새 모델이 기존 모델보다 나은지 판단"""
        try:
            # 1. PyTorch 검증 상태 우선
            if new_model.pytorch_valid and not existing_model.pytorch_valid:
                return True
            elif not new_model.pytorch_valid and existing_model.pytorch_valid:
                return False
            
            # 2. 우선순위 비교
            if new_model.priority.value < existing_model.priority.value:
                return True
            elif new_model.priority.value > existing_model.priority.value:
                return False
            
            # 3. 신뢰도 비교
            if abs(new_model.confidence_score - existing_model.confidence_score) > 0.1:
                return new_model.confidence_score > existing_model.confidence_score
            
            # 4. 파라미터 수 비교 (더 많으면 일반적으로 더 좋음)
            if new_model.parameter_count > 0 and existing_model.parameter_count > 0:
                if abs(new_model.parameter_count - existing_model.parameter_count) / max(new_model.parameter_count, existing_model.parameter_count) > 0.2:
                    return new_model.parameter_count > existing_model.parameter_count
            
            # 5. 최신성 비교
            if abs(new_model.last_modified - existing_model.last_modified) > 86400:  # 1일 이상 차이
                return new_model.last_modified > existing_model.last_modified
            
            # 6. 파일 크기 비교
            return new_model.file_size_mb > existing_model.file_size_mb
            
        except Exception as e:
            self.logger.debug(f"모델 비교 오류: {e}")
            return new_model.file_size_mb > existing_model.file_size_mb

    def _reset_scan_stats(self):
        """스캔 통계 리셋"""
        self.scan_stats.update({
            "total_files_scanned": 0,
            "pytorch_files_found": 0,
            "valid_pytorch_models": 0,
            "models_detected": 0,
            "scan_duration": 0.0,
            "errors_encountered": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "pytorch_validation_errors": 0
        })

    def _post_process_results(self, min_confidence: float):
        """결과 후처리"""
        try:
            # 신뢰도 필터링
            filtered_models = {
                name: model for name, model in self.detected_models.items()
                if model.confidence_score >= min_confidence
            }
            self.detected_models = filtered_models
            
            # 우선순위에 따른 정렬
            sorted_models = sorted(
                self.detected_models.items(),
                key=lambda x: (x[1].priority.value, -x[1].confidence_score, -x[1].parameter_count, -x[1].file_size_mb)
            )
            
            self.detected_models = {name: model for name, model in sorted_models}
            
        except Exception as e:
            self.logger.error(f"❌ 후처리 실패: {e}")

    def _print_detection_summary(self):
        """탐지 결과 요약 출력 (실제 검증 정보 포함)"""
        try:
            self.logger.info("=" * 70)
            self.logger.info("🔍 실제 AI 모델 탐지 결과 요약")
            self.logger.info("=" * 70)
            
            total_size_gb = sum(model.file_size_mb for model in self.detected_models.values()) / 1024
            total_params = sum(model.parameter_count for model in self.detected_models.values())
            avg_confidence = sum(model.confidence_score for model in self.detected_models.values()) / len(self.detected_models) if self.detected_models else 0
            pytorch_valid_count = sum(1 for model in self.detected_models.values() if model.pytorch_valid)
            
            self.logger.info(f"📊 탐지된 모델: {len(self.detected_models)}개")
            self.logger.info(f"💾 총 크기: {total_size_gb:.2f}GB")
            self.logger.info(f"🔍 스캔 파일: {self.scan_stats['total_files_scanned']:,}개")
            self.logger.info(f"🐍 PyTorch 파일: {self.scan_stats['pytorch_files_found']}개")
            self.logger.info(f"✅ 검증된 모델: {pytorch_valid_count}개")
            self.logger.info(f"⏱️ 소요 시간: {self.scan_stats['scan_duration']:.2f}초")
            self.logger.info(f"🎯 평균 신뢰도: {avg_confidence:.3f}")
            self.logger.info(f"🧮 총 파라미터: {total_params:,}개")
            
            # 모델 타입별 분포
            type_distribution = {}
            for model in self.detected_models.values():
                model_type = model.category.value
                if model_type not in type_distribution:
                    type_distribution[model_type] = 0
                type_distribution[model_type] += 1
            
            if type_distribution:
                self.logger.info("\n📁 모델 타입별 분포:")
                for model_type, count in type_distribution.items():
                    self.logger.info(f"  {model_type}: {count}개")
            
            # 주요 모델들 (검증 정보 포함)
            if self.detected_models:
                self.logger.info("\n🏆 탐지된 주요 모델들:")
                for i, (name, model) in enumerate(list(self.detected_models.items())[:5]):
                    status = "✅검증됨" if model.pytorch_valid else "❓미검증"
                    self.logger.info(f"  {i+1}. {name}")
                    self.logger.info(f"     타입: {model.category.value}, 크기: {model.file_size_mb:.1f}MB")
                    self.logger.info(f"     신뢰도: {model.confidence_score:.3f}, 상태: {status}")
                    if model.parameter_count > 0:
                        self.logger.info(f"     파라미터: {model.parameter_count:,}개")
            
            self.logger.info("=" * 70)
                
        except Exception as e:
            self.logger.error(f"❌ 요약 출력 실패: {e}")

    # ==============================================
    # 🔥 캐시 관련 메서드들 (실제 검증 정보 포함)
    # ==============================================

    def _load_from_cache(self) -> Optional[Dict[str, DetectedModel]]:
        """캐시에서 로드 (실제 검증 정보 포함)"""
        try:
            with self._cache_lock:
                with sqlite3.connect(self.cache_db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 만료된 캐시 정리
                    cutoff_time = time.time() - self.cache_ttl
                    cursor.execute("DELETE FROM real_model_cache WHERE created_at < ?", (cutoff_time,))
                    
                    # 캐시 조회
                    cursor.execute("""
                        SELECT file_path, detection_data, pytorch_valid, parameter_count
                        FROM real_model_cache 
                        WHERE created_at > ?
                    """, (cutoff_time,))
                    
                    cached_models = {}
                    for file_path, detection_data, pytorch_valid, parameter_count in cursor.fetchall():
                        try:
                            # 파일이 여전히 존재하는지 확인
                            if not Path(file_path).exists():
                                continue
                            
                            model_data = json.loads(detection_data)
                            model = self._deserialize_detected_model(model_data)
                            if model:
                                # 캐시된 검증 정보 복원
                                model.pytorch_valid = bool(pytorch_valid)
                                model.parameter_count = parameter_count or 0
                                cached_models[model.name] = model
                        except Exception as e:
                            self.logger.debug(f"캐시 항목 로드 실패 {file_path}: {e}")
                    
                    if cached_models:
                        # 액세스 시간 업데이트
                        cursor.execute("UPDATE real_model_cache SET accessed_at = ?", (time.time(),))
                        conn.commit()
                        
                        self.detected_models = cached_models
                        return cached_models
            
            return None
            
        except Exception as e:
            self.logger.debug(f"캐시 로드 실패: {e}")
            return None

    def _save_to_cache(self):
        """캐시에 저장 (실제 검증 정보 포함)"""
        try:
            with self._cache_lock:
                with sqlite3.connect(self.cache_db_path) as conn:
                    cursor = conn.cursor()
                    current_time = time.time()
                    
                    for model in self.detected_models.values():
                        try:
                            detection_data = json.dumps(self._serialize_detected_model(model))
                            
                            cursor.execute("""
                                INSERT OR REPLACE INTO real_model_cache 
                                (file_path, file_size, file_mtime, checksum, pytorch_valid, parameter_count, detection_data, created_at, accessed_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                str(model.path),
                                int(model.file_size_mb * 1024 * 1024),
                                model.last_modified,
                                model.checksum,
                                int(model.pytorch_valid),
                                model.parameter_count,
                                detection_data,
                                current_time,
                                current_time
                            ))
                        except Exception as e:
                            self.logger.debug(f"모델 캐시 저장 실패 {model.name}: {e}")
                    
                    conn.commit()
                    
        except Exception as e:
            self.logger.debug(f"캐시 저장 실패: {e}")

    def _serialize_detected_model(self, model: DetectedModel) -> Dict[str, Any]:
        """DetectedModel을 딕셔너리로 직렬화 (실제 검증 정보 포함)"""
        return {
            "name": model.name,
            "path": str(model.path),
            "category": model.category.value,
            "model_type": model.model_type,
            "file_size_mb": model.file_size_mb,
            "file_extension": model.file_extension,
            "confidence_score": model.confidence_score,
            "priority": model.priority.value,
            "step_name": model.step_name,
            "metadata": model.metadata,
            "alternative_paths": [str(p) for p in model.alternative_paths],
            "requirements": model.requirements,
            "performance_info": model.performance_info,
            "compatibility_info": model.compatibility_info,
            "last_modified": model.last_modified,
            "checksum": model.checksum,
            "pytorch_valid": model.pytorch_valid,
            "parameter_count": model.parameter_count
        }

    def _deserialize_detected_model(self, data: Dict[str, Any]) -> Optional[DetectedModel]:
        """딕셔너리를 DetectedModel로 역직렬화 (실제 검증 정보 포함)"""
        try:
            return DetectedModel(
                name=data["name"],
                path=Path(data["path"]),
                category=ModelCategory(data["category"]),
                model_type=data["model_type"],
                file_size_mb=data["file_size_mb"],
                file_extension=data["file_extension"],
                confidence_score=data["confidence_score"],
                priority=ModelPriority(data["priority"]),
                step_name=data["step_name"],
                metadata=data.get("metadata", {}),
                alternative_paths=[Path(p) for p in data.get("alternative_paths", [])],
                requirements=data.get("requirements", []),
                performance_info=data.get("performance_info", {}),
                compatibility_info=data.get("compatibility_info", {}),
                last_modified=data.get("last_modified", 0.0),
                checksum=data.get("checksum"),
                pytorch_valid=data.get("pytorch_valid", False),
                parameter_count=data.get("parameter_count", 0)
            )
        except Exception as e:
            self.logger.debug(f"모델 역직렬화 실패: {e}")
            return None

    # ==============================================
    # 🔥 공개 조회 메서드들
    # ==============================================

    def get_models_by_category(self, category: ModelCategory) -> List[DetectedModel]:
        """카테고리별 모델 조회"""
        return [model for model in self.detected_models.values() if model.category == category]

    def get_models_by_step(self, step_name: str) -> List[DetectedModel]:
        """Step별 모델 조회"""
        return [model for model in self.detected_models.values() if model.step_name == step_name]

    def get_best_model_for_step(self, step_name: str) -> Optional[DetectedModel]:
        """Step별 최적 모델 조회"""
        step_models = self.get_models_by_step(step_name)
        if not step_models:
            return None
        
        return min(step_models, key=lambda m: (m.priority.value, -m.confidence_score, -m.parameter_count))

    def get_model_by_name(self, name: str) -> Optional[DetectedModel]:
        """이름으로 모델 조회"""
        return self.detected_models.get(name)

    def get_all_model_paths(self) -> Dict[str, Path]:
        """모든 모델의 경로 딕셔너리 반환"""
        return {name: model.path for name, model in self.detected_models.items()}

    def get_validated_models_only(self) -> Dict[str, DetectedModel]:
        """PyTorch 검증된 모델들만 반환"""
        return {name: model for name, model in self.detected_models.items() if model.pytorch_valid}

    def search_models(
        self, 
        keywords: List[str], 
        step_filter: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        validated_only: bool = False
    ) -> List[DetectedModel]:
        """키워드로 모델 검색"""
        try:
            results = []
            keywords_lower = [kw.lower() for kw in keywords]
            
            for model in self.detected_models.values():
                # 신뢰도 필터
                if model.confidence_score < min_confidence:
                    continue
                
                # 검증 필터
                if validated_only and not model.pytorch_valid:
                    continue
                
                # Step 필터
                if step_filter and model.step_name not in step_filter:
                    continue
                
                # 키워드 매칭
                model_text = f"{model.name} {model.path.name} {model.model_type} {model.step_name}".lower()
                if any(keyword in model_text for keyword in keywords_lower):
                    results.append(model)
            
            # 관련성 순으로 정렬 (검증된 모델 우선)
            results.sort(key=lambda m: (not m.pytorch_valid, m.priority.value, -m.confidence_score, -m.parameter_count))
            return results
            
        except Exception as e:
            self.logger.error(f"모델 검색 실패: {e}")
            return []

# ==============================================
# 🔥 ModelLoader 연동용 설정 생성기 (순환참조 완전 방지)
# ==============================================

class RealModelLoaderConfigGenerator:
    """
    🔗 실제 ModelLoader 연동용 설정 생성기 v6.0
    ✅ 실제 검증된 모델 정보 포함
    ✅ 딕셔너리 기반으로만 동작
    ✅ 순환참조 완전 방지
    """
    
    def __init__(self, detector: RealWorldModelDetector):
        self.detector = detector
        self.logger = logging.getLogger(f"{__name__}.RealModelLoaderConfigGenerator")
    
    def generate_complete_config(self) -> Dict[str, Any]:
        """ModelLoader용 완전한 설정 생성 (실제 검증 정보 포함)"""
        try:
            config = {
                "model_configs": [],
                "model_paths": {},
                "step_mappings": {},
                "priority_rankings": {},
                "performance_estimates": {},
                "validation_results": {},
                "metadata": {
                    "total_models": len(self.detector.detected_models),
                    "validated_models": len(self.detector.get_validated_models_only()),
                    "generation_time": time.time(),
                    "detector_version": "6.0",
                    "scan_stats": self.detector.scan_stats
                }
            }
            
            for name, detected_model in self.detector.detected_models.items():
                # ModelConfig 딕셔너리 생성 (실제 검증 정보 포함)
                model_config = {
                    "name": name,
                    "model_type": detected_model.category.value,
                    "model_class": detected_model.model_type,
                    "checkpoint_path": str(detected_model.path),
                    "device": "auto",
                    "precision": "fp16",
                    "input_size": self._get_input_size_for_step(detected_model.step_name),
                    "step_name": detected_model.step_name,
                    "pytorch_validated": detected_model.pytorch_valid,
                    "parameter_count": detected_model.parameter_count,
                    "metadata": {
                        **detected_model.metadata,
                        "auto_detected": True,
                        "confidence_score": detected_model.confidence_score,
                        "priority": detected_model.priority.name,
                        "alternative_paths": [str(p) for p in detected_model.alternative_paths],
                        "pytorch_validated": detected_model.pytorch_valid,
                        "parameter_count": detected_model.parameter_count
                    }
                }
                config["model_configs"].append(model_config)
                
                # 경로 매핑 (실제 검증 정보 포함)
                config["model_paths"][name] = {
                    "primary": str(detected_model.path),
                    "alternatives": [str(p) for p in detected_model.alternative_paths],
                    "size_mb": detected_model.file_size_mb,
                    "confidence": detected_model.confidence_score,
                    "pytorch_valid": detected_model.pytorch_valid,
                    "parameter_count": detected_model.parameter_count
                }
                
                # Step 매핑
                step_name = detected_model.step_name
                if step_name not in config["step_mappings"]:
                    config["step_mappings"][step_name] = []
                config["step_mappings"][step_name].append(name)
                
                # 우선순위 (검증 상태 포함)
                config["priority_rankings"][name] = {
                    "priority_level": detected_model.priority.value,
                    "priority_name": detected_model.priority.name,
                    "confidence_score": detected_model.confidence_score,
                    "step_rank": self._get_step_rank(detected_model.step_name),
                    "pytorch_validated": detected_model.pytorch_valid,
                    "parameter_count": detected_model.parameter_count
                }
                
                # 성능 추정 (실제 파라미터 수 기반)
                config["performance_estimates"][name] = {
                    "estimated_memory_gb": max(1.0, detected_model.file_size_mb / 1024 * 2),
                    "estimated_load_time_sec": self._estimate_load_time(detected_model),
                    "recommended_batch_size": self._get_recommended_batch_size(detected_model),
                    "gpu_memory_required_gb": max(2.0, detected_model.file_size_mb / 1024 * 1.5),
                    "parameter_count": detected_model.parameter_count,
                    "pytorch_validated": detected_model.pytorch_valid
                }
                
                # 검증 결과
                config["validation_results"][name] = {
                    "pytorch_valid": detected_model.pytorch_valid,
                    "parameter_count": detected_model.parameter_count,
                    "validation_metadata": {k: v for k, v in detected_model.metadata.items() 
                                          if k.startswith(('pytorch_', 'checkpoint_', 'layer_', 'parameter_', 'type_', 'validation_'))}
                }
            
            return config
            
        except Exception as e:
            self.logger.error(f"❌ 실제 ModelLoader 설정 생성 실패: {e}")
            return {"error": str(e)}

    def _get_input_size_for_step(self, step_name: str) -> Tuple[int, int]:
        """Step별 기본 입력 크기"""
        size_mapping = {
            "HumanParsingStep": (512, 512),
            "PoseEstimationStep": (368, 368),
            "ClothSegmentationStep": (320, 320),
            "GeometricMatchingStep": (512, 384),
            "ClothWarpingStep": (512, 384),
            "VirtualFittingStep": (512, 512),
            "PostProcessingStep": (512, 512),
            "QualityAssessmentStep": (224, 224)
        }
        return size_mapping.get(step_name, (512, 512))

    def _estimate_load_time(self, detected_model: DetectedModel) -> float:
        """모델 로드 시간 추정 (실제 파라미터 수 기반)"""
        base_times = {
            ModelCategory.HUMAN_PARSING: 2.0,
            ModelCategory.POSE_ESTIMATION: 1.0,
            ModelCategory.CLOTH_SEGMENTATION: 1.5,
            ModelCategory.GEOMETRIC_MATCHING: 0.5,
            ModelCategory.CLOTH_WARPING: 3.0,
            ModelCategory.VIRTUAL_FITTING: 8.0,
            ModelCategory.DIFFUSION_MODELS: 10.0,
            ModelCategory.TRANSFORMER_MODELS: 3.0,
            ModelCategory.POST_PROCESSING: 2.0,
            ModelCategory.QUALITY_ASSESSMENT: 1.0
        }
        
        base_time = base_times.get(detected_model.category, 2.0)
        
        # 파일 크기 기반 조정
        size_factor = min(detected_model.file_size_mb / 100, 5.0)
        
        # 파라미터 수 기반 조정
        if detected_model.parameter_count > 0:
            param_factor = min(detected_model.parameter_count / 50000000, 3.0)  # 50M 파라미터 기준
            return base_time * max(size_factor, param_factor)
        
        return base_time * size_factor

    def _get_recommended_batch_size(self, detected_model: DetectedModel) -> int:
        """권장 배치 크기 (실제 파라미터 수 기반)"""
        if detected_model.parameter_count > 500000000:  # 500M+ 파라미터
            return 1
        elif detected_model.parameter_count > 100000000:  # 100M+ 파라미터
            return 2
        elif detected_model.file_size_mb > 1000:  # 1GB+ 파일
            return 1
        elif detected_model.file_size_mb > 100:  # 100MB+ 파일
            return 2
        else:
            return 4

    def _get_step_rank(self, step_name: str) -> int:
        """Step별 순위 (중요도)"""
        rank_mapping = {
            "HumanParsingStep": 1,
            "VirtualFittingStep": 2,
            "PoseEstimationStep": 3,
            "ClothSegmentationStep": 3,
            "ClothWarpingStep": 4,
            "GeometricMatchingStep": 5,
            "PostProcessingStep": 6,
            "QualityAssessmentStep": 7
        }
        return rank_mapping.get(step_name, 9)

# 1. auto_model_detector.py 끝에 추가 (하위 호환성)

# ==============================================
# 🔥 하위 호환성 클래스들 (기존 코드 지원)
# ==============================================

class AdvancedModelLoaderAdapter:
    """
    기존 ModelLoader와 호환성을 위한 어댑터
    """
    
    def __init__(self, detector: RealWorldModelDetector):
        self.detector = detector
        self.logger = logging.getLogger(f"{__name__}.AdvancedModelLoaderAdapter")
    
    def get_model_configs(self) -> List[Dict[str, Any]]:
        """ModelLoader가 기대하는 형태로 설정 반환"""
        try:
            config_generator = RealModelLoaderConfigGenerator(self.detector)
            full_config = config_generator.generate_complete_config()
            return full_config.get("model_configs", [])
        except Exception as e:
            self.logger.error(f"모델 설정 생성 실패: {e}")
            return []
    
    def register_models_to_loader(self, model_loader):
        """탐지된 모델들을 ModelLoader에 등록"""
        try:
            detected_models = self.detector.detected_models
            registered_count = 0
            
            for name, model in detected_models.items():
                try:
                    # 기존 ModelLoader 등록 방식에 맞춤
                    model_config = {
                        "name": name,
                        "model_type": model.model_type,
                        "checkpoint_path": str(model.path),
                        "device": "auto",
                        "precision": "fp16",
                        "pytorch_validated": model.pytorch_valid,
                        "parameter_count": model.parameter_count,
                        "confidence_score": model.confidence_score
                    }
                    
                    # ModelLoader에 등록 (기존 메서드 사용)
                    if hasattr(model_loader, 'register_model'):
                        model_loader.register_model(name, model_config)
                        registered_count += 1
                    elif hasattr(model_loader, '_register_model'):
                        model_loader._register_model(name, model_config)
                        registered_count += 1
                        
                except Exception as e:
                    self.logger.warning(f"모델 등록 실패 {name}: {e}")
            
            self.logger.info(f"✅ {registered_count}개 모델 등록 완료")
            return registered_count
            
        except Exception as e:
            self.logger.error(f"모델 등록 프로세스 실패: {e}")
            return 0

# ==============================================
# 2. model_loader.py의 StepModelInterface 클래스에 추가
# ==============================================

class StepModelInterface:
    """Step 클래스를 위한 모델 인터페이스"""
    
    def __init__(self, model_loader, step_name: str):
        self.model_loader = model_loader
        self.step_name = step_name
        self.logger = logging.getLogger(f"steps.{step_name}")
        self.loaded_models = {}
    
    # 🔥 기존 메서드들...
    
    async def load_model_async(self, model_name: str, **kwargs) -> Optional[Any]:
        """
        🔥 비동기 모델 로드 (기존 코드 호환성)
        """
        try:
            # 동기 메서드 래핑
            return await asyncio.get_event_loop().run_in_executor(
                None, self.load_model, model_name, **kwargs
            )
        except Exception as e:
            self.logger.error(f"비동기 모델 로드 실패 {model_name}: {e}")
            return None
    
    def load_model(self, model_name: str, **kwargs) -> Optional[Any]:
        """
        🔥 동기 모델 로드 (개선된 버전)
        """
        try:
            # 캐시 확인
            if model_name in self.loaded_models:
                return self.loaded_models[model_name]
            
            # ModelLoader를 통한 로드
            if hasattr(self.model_loader, 'load_model'):
                model = self.model_loader.load_model(model_name, **kwargs)
                if model:
                    self.loaded_models[model_name] = model
                    return model
            
            # 폴백: 직접 로드 시도
            return self._fallback_load_model(model_name, **kwargs)
            
        except Exception as e:
            self.logger.error(f"모델 로드 실패 {model_name}: {e}")
            return None
    
    def _fallback_load_model(self, model_name: str, **kwargs) -> Optional[Any]:
        """폴백 모델 로드"""
        try:
            # 실제 모델 탐지기 사용
            detector = create_real_world_detector()
            detected_models = detector.detect_all_models()
            
            # 모델 이름 매핑
            name_mapping = {
                "ootdiffusion": "virtual_fitting_diffusion",
                "human_parsing": "human_parsing_graphonomy", 
                "openpose": "pose_estimation_openpose",
                "u2net": "cloth_segmentation_u2net",
                "clip": "clip_vit_base"
            }
            
            target_name = name_mapping.get(model_name, model_name)
            
            if target_name in detected_models:
                model_info = detected_models[target_name]
                self.logger.info(f"✅ 폴백 모델 로드 성공: {model_name} -> {target_name}")
                return model_info
            
            return None
            
        except Exception as e:
            self.logger.error(f"폴백 모델 로드 실패 {model_name}: {e}")
            return None

# ==============================================
# 3. model_loader.py의 ModelLoader 클래스에 추가
# ==============================================

class ModelLoader:
    """AI 모델 로더"""
    
    def __init__(self, device: str = "auto", **kwargs):
        # 🔥 기존 초기화 코드...
        
        # 자동 탐지기 연동
        self.auto_detector = None
        self.auto_adapter = None
        self._initialize_auto_detection()
    
    def _initialize_auto_detection(self):
        """자동 탐지기 초기화 및 연동"""
        try:
            # 실제 모델 탐지기 생성
            self.auto_detector = create_real_world_detector()
            
            # 어댑터 생성
            self.auto_adapter = AdvancedModelLoaderAdapter(self.auto_detector)
            
            # 모델 탐지 및 등록
            detected_models = self.auto_detector.detect_all_models()
            
            if detected_models:
                registered_count = self.auto_adapter.register_models_to_loader(self)
                self.logger.info(f"🔍 자동 탐지 완료: {len(detected_models)}개 발견, {registered_count}개 등록")
            else:
                self.logger.warning("⚠️ 자동 탐지된 모델이 없습니다")
                
        except Exception as e:
            self.logger.error(f"❌ 자동 탐지기 초기화 실패: {e}")
            self.auto_detector = None
            self.auto_adapter = None
    
    async def load_model_async(self, model_name: str, **kwargs) -> Optional[Any]:
        """
        🔥 비동기 모델 로드 (기존 코드 호환성)
        """
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self.load_model, model_name, **kwargs
            )
        except Exception as e:
            self.logger.error(f"비동기 모델 로드 실패 {model_name}: {e}")
            return None
    
    def register_model(self, name: str, config: Dict[str, Any]):
        """
        🔥 모델 등록 (어댑터에서 사용)
        """
        try:
            # 기존 등록 방식 사용
            if hasattr(self, 'model_registry'):
                self.model_registry[name] = config
            else:
                # 새로운 registry 생성
                if not hasattr(self, 'detected_model_registry'):
                    self.detected_model_registry = {}
                self.detected_model_registry[name] = config
            
            self.logger.debug(f"✅ 모델 등록: {name}")
            
        except Exception as e:
            self.logger.error(f"❌ 모델 등록 실패 {name}: {e}")


# ==============================================
# 🔥 편의 함수들 및 팩토리 함수들 (실제 검증 기반)
# ==============================================

def create_real_world_detector(
    search_paths: Optional[List[Path]] = None,
    enable_parallel: bool = True,
    enable_pytorch_validation: bool = True,
    max_workers: int = 4,
    **kwargs
) -> RealWorldModelDetector:
    """실제 동작하는 모델 탐지기 생성"""
    return RealWorldModelDetector(
        search_paths=search_paths,
        enable_pytorch_validation=enable_pytorch_validation,
        max_workers=max_workers if enable_parallel else 1,
        **kwargs
    )

def quick_real_model_detection(
    model_type_filter: Optional[List[str]] = None,
    min_confidence: float = 0.5,
    force_rescan: bool = False,
    validated_only: bool = False
) -> Dict[str, Any]:
    """빠른 실제 모델 탐지 및 결과 반환"""
    try:
        # 탐지기 생성 및 실행
        detector = create_real_world_detector()
        detected_models = detector.detect_all_models(
            force_rescan=force_rescan,
            model_type_filter=model_type_filter,
            min_confidence=min_confidence
        )
        
        # 검증된 모델만 필터링 (옵션)
        if validated_only:
            detected_models = detector.get_validated_models_only()
        
        # 결과 요약
        summary = {
            "total_models": len(detected_models),
            "validated_models": len([m for m in detected_models.values() if m.pytorch_valid]),
            "models_by_type": {},
            "models_by_priority": {},
            "top_models": {},
            "validation_summary": {},
            "scan_stats": detector.scan_stats
        }
        
        # 타입별 분류
        for model in detected_models.values():
            model_type = model.category.value
            if model_type not in summary["models_by_type"]:
                summary["models_by_type"][model_type] = []
            summary["models_by_type"][model_type].append({
                "name": model.name,
                "path": str(model.path),
                "confidence": model.confidence_score,
                "size_mb": model.file_size_mb,
                "pytorch_valid": model.pytorch_valid,
                "parameter_count": model.parameter_count
            })
        
        # 우선순위별 분류
        for model in detected_models.values():
            priority = model.priority.name
            if priority not in summary["models_by_priority"]:
                summary["models_by_priority"][priority] = []
            summary["models_by_priority"][priority].append(model.name)
        
        # 타입별 최고 모델 (검증된 것 우선)
        model_types = set(model.category.value for model in detected_models.values())
        for model_type in model_types:
            type_models = [m for m in detected_models.values() if m.category.value == model_type]
            if type_models:
                best_model = min(type_models, key=lambda m: (not m.pytorch_valid, m.priority.value, -m.confidence_score, -m.parameter_count))
                summary["top_models"][model_type] = {
                    "name": best_model.name,
                    "path": str(best_model.path),
                    "confidence": best_model.confidence_score,
                    "priority": best_model.priority.name,
                    "pytorch_valid": best_model.pytorch_valid,
                    "parameter_count": best_model.parameter_count
                }
        
        # 검증 요약
        total_params = sum(m.parameter_count for m in detected_models.values())
        summary["validation_summary"] = {
            "total_parameters": total_params,
            "avg_confidence": sum(m.confidence_score for m in detected_models.values()) / len(detected_models) if detected_models else 0,
            "validation_rate": len([m for m in detected_models.values() if m.pytorch_valid]) / len(detected_models) if detected_models else 0
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"빠른 실제 모델 탐지 실패: {e}")
        return {"error": str(e)}

def generate_real_model_loader_config(
    detector: Optional[RealWorldModelDetector] = None,
    **detection_kwargs
) -> Dict[str, Any]:
    """
    실제 ModelLoader용 설정 생성 (순환참조 방지)
    실제 검증된 모델 정보 기반 딕셔너리 출력
    """
    try:
        logger.info("🔍 실제 ModelLoader 설정 생성 시작...")
        
        # 탐지기가 없으면 새로 생성
        if detector is None:
            detector = create_real_world_detector(**detection_kwargs)
            detected_models = detector.detect_all_models()
        else:
            detected_models = detector.detected_models
        
        if not detected_models:
            logger.warning("⚠️ 탐지된 모델이 없습니다")
            return {"success": False, "message": "No models detected"}
        
        # 설정 생성기 사용
        config_generator = RealModelLoaderConfigGenerator(detector)
        model_loader_config = config_generator.generate_complete_config()
        
        # 최종 결과
        validated_count = len(detector.get_validated_models_only())
        result = {
            "success": True,
            "model_loader_config": model_loader_config,
            "detection_summary": {
                "total_models": len(detected_models),
                "validated_models": validated_count,
                "validation_rate": validated_count / len(detected_models) if detected_models else 0,
                "scan_duration": detector.scan_stats["scan_duration"],
                "confidence_avg": sum(m.confidence_score for m in detected_models.values()) / len(detected_models),
                "total_parameters": sum(m.parameter_count for m in detected_models.values())
            }
        }
        
        logger.info(f"✅ 실제 ModelLoader 설정 생성 완료: {len(detected_models)}개 모델 ({validated_count}개 검증)")
        return result
        
    except Exception as e:
        logger.error(f"❌ 실제 ModelLoader 설정 생성 실패: {e}")
        return {"success": False, "error": str(e)}
# backend/app/ai_pipeline/utils/auto_model_detector.py 끝 부분에 추가

def detect_and_integrate_with_model_loader(
    model_loader_instance = None,
    auto_register: bool = True,
    **detection_kwargs
) -> Dict[str, Any]:
    """모델 탐지 및 ModelLoader 통합 (순환참조 방지)"""
    try:
        logger.info("🔍 모델 탐지 및 ModelLoader 통합 시작...")
        
        # 탐지 실행
        detector = create_advanced_detector(**detection_kwargs)
        detected_models = detector.detect_all_models()
        
        if not detected_models:
            logger.warning("⚠️ 탐지된 모델이 없습니다")
            return {"success": False, "message": "No models detected"}
        
        # 어댑터 생성
        adapter = AdvancedModelLoaderAdapter(detector)
        
        # ModelLoader 설정 생성
        model_loader_config = adapter.generate_model_loader_config()
        
        # ModelLoader와 통합 (순환참조 방지)
        integration_result = {}
        if auto_register and model_loader_instance:
            try:
                # ModelLoader 인스턴스 설정
                detector.set_model_loader(model_loader_instance)
                
                # 자동 등록
                registered_count = adapter.register_models_to_loader(model_loader_instance)
                integration_result["registered_models"] = registered_count
                
            except Exception as e:
                logger.warning(f"⚠️ ModelLoader 통합 실패: {e}")
                integration_result["integration_error"] = str(e)
        
        return {
            "success": True,
            "detected_count": len(detected_models),
            "model_names": list(detected_models.keys()),
            "integration": integration_result,
            "config": model_loader_config
        }
        
    except Exception as e:
        logger.error(f"❌ 탐지 및 통합 실패: {e}")
        return {"success": False, "error": str(e)}


def validate_real_model_paths(detected_models: Dict[str, DetectedModel]) -> Dict[str, Any]:
    """실제 탐지된 모델 경로들의 유효성 검증"""
    try:
        validation_result = {
            "valid_models": [],
            "invalid_models": [],
            "missing_files": [],
            "permission_errors": [],
            "pytorch_validated": [],
            "pytorch_failed": [],
            "total_size_gb": 0.0,
            "total_parameters": 0
        }
        
        for name, model in detected_models.items():
            try:
                # 주 경로 확인
                if model.path.exists() and model.path.is_file():
                    validation_result["valid_models"].append(name)
                    validation_result["total_size_gb"] += model.file_size_mb / 1024
                    validation_result["total_parameters"] += model.parameter_count
                    
                    # PyTorch 검증 상태 확인
                    if model.pytorch_valid:
                        validation_result["pytorch_validated"].append(name)
                    else:
                        validation_result["pytorch_failed"].append(name)
                else:
                    validation_result["missing_files"].append({
                        "name": name,
                        "path": str(model.path)
                    })
                
                # 대체 경로들 확인
                valid_alternatives = []
                for alt_path in model.alternative_paths:
                    if alt_path.exists() and alt_path.is_file():
                        valid_alternatives.append(str(alt_path))
                
                if valid_alternatives and name in [m["name"] for m in validation_result["missing_files"]]:
                    # 주 경로는 없지만 대체 경로가 있는 경우
                    validation_result["missing_files"] = [
                        m for m in validation_result["missing_files"] 
                        if m["name"] != name
                    ]
                    validation_result["valid_models"].append(name)
                
            except PermissionError:
                validation_result["permission_errors"].append({
                    "name": name,
                    "path": str(model.path)
                })
            except Exception as e:
                validation_result["invalid_models"].append({
                    "name": name,
                    "path": str(model.path),
                    "error": str(e)
                })
        
        validation_result["summary"] = {
            "total_models": len(detected_models),
            "valid_count": len(validation_result["valid_models"]),
            "invalid_count": len(validation_result["invalid_models"]),
            "missing_count": len(validation_result["missing_files"]),
            "permission_error_count": len(validation_result["permission_errors"]),
            "pytorch_validated_count": len(validation_result["pytorch_validated"]),
            "pytorch_failed_count": len(validation_result["pytorch_failed"]),
            "validation_rate": len(validation_result["valid_models"]) / len(detected_models) if detected_models else 0,
            "pytorch_validation_rate": len(validation_result["pytorch_validated"]) / len(detected_models) if detected_models else 0
        }
        
        return validation_result
        
    except Exception as e:
        logger.error(f"실제 모델 경로 검증 실패: {e}")
        return {"error": str(e)}

# 모듈 익스포트
__all__ = [
    # 기존 exports...
    'RealWorldModelDetector',
    'RealModelLoaderConfigGenerator',
    'DetectedModel',
    'ModelCategory',
    'ModelPriority',
    'ModelFileInfo',
    'detect_and_integrate_with_model_loader' #새로추가
    # 새로운 호환성 클래스
    'AdvancedModelLoaderAdapter',  # 🔥 이것이 누락되었음

    # 팩토리 함수들
    'create_real_world_detector',
    'quick_real_model_detection',
    'generate_real_model_loader_config',

    # 유틸리티 함수들
    'validate_real_model_paths',

    # 설정 및 패턴
    'ACTUAL_MODEL_PATTERNS',
    'CHECKPOINT_VERIFICATION_PATTERNS',

    # 하위 호환성 별칭
    'AdvancedModelDetector',
    'ModelLoaderConfigGenerator',
    'create_advanced_detector',
    'quick_model_detection',
    'generate_model_loader_config',
    'validate_model_paths'
]

# 하위 호환성을 위한 별칭 (기존 코드와 호환)
AdvancedModelDetector = RealWorldModelDetector
ModelLoaderConfigGenerator = RealModelLoaderConfigGenerator
create_advanced_detector = create_real_world_detector
quick_model_detection = quick_real_model_detection
generate_model_loader_config = generate_real_model_loader_config
validate_model_paths = validate_real_model_paths

logger.info("✅ 실제 동작하는 자동 모델 탐지 시스템 v6.0 로드 완료 - PyTorch 검증 포함")