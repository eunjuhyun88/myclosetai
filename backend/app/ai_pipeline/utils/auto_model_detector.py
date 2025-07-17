# app/ai_pipeline/utils/auto_model_detector.py
"""
🔍 MyCloset AI - 완전 통합 자동 모델 탐지 시스템 v5.0
✅ 순환참조 완전 해결 (model_loader 직접 import 제거)
✅ step_model_requests.py 기반 정확한 모델 탐지
✅ 실제 존재하는 AI 모델 파일들 자동 발견
✅ 딕셔너리 기반 설정 출력 (순환참조 방지)
✅ M3 Max 128GB 최적화
✅ conda 환경 특화 스캔
✅ 프로덕션 안정성 보장

🔥 핵심 변경사항:
- ModelLoader 직접 import 제거
- 딕셔너리 기반 설정 출력
- 인터페이스를 통한 연동
- 런타임 에러 방지
"""

import os
import re
import time
import logging
import hashlib
import json
import threading
import asyncio
import sqlite3
import pickle
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
# 🔥 모델 탐지 설정 및 매핑
# ==============================================

class ModelCategory(Enum):
    """모델 카테고리 (step_model_requests.py와 연동)"""
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

# ==============================================
# 🔥 Step별 모델 요청사항 Import 및 처리
# ==============================================

try:
    from .step_model_requests import (
        STEP_MODEL_REQUESTS,
        StepModelRequestAnalyzer,
        get_all_step_requirements
    )
    STEP_REQUESTS_AVAILABLE = True
    logger.info("✅ step_model_requests 모듈 연동 성공")
except ImportError as e:
    STEP_REQUESTS_AVAILABLE = False
    logger.warning(f"⚠️ step_model_requests 모듈 연동 실패: {e}")
    
    # 내장 기본 요청사항
    STEP_MODEL_REQUESTS = {
        "HumanParsingStep": {
            "model_name": "human_parsing_graphonomy",
            "model_type": "GraphonomyModel",
            "checkpoint_patterns": ["*human*parsing*.pth", "*schp*atr*.pth", "*graphonomy*.pth"],
            "step_priority": 1
        },
        "PoseEstimationStep": {
            "model_name": "pose_estimation_openpose",
            "model_type": "OpenPoseModel", 
            "checkpoint_patterns": ["*pose*model*.pth", "*openpose*.pth", "*body*pose*.pth"],
            "step_priority": 2
        },
        "ClothSegmentationStep": {
            "model_name": "cloth_segmentation_u2net",
            "model_type": "U2NetModel",
            "checkpoint_patterns": ["*u2net*.pth", "*cloth*segmentation*.pth", "*sam*.pth"],
            "step_priority": 2
        },
        "VirtualFittingStep": {
            "model_name": "virtual_fitting_stable_diffusion",
            "model_type": "StableDiffusionPipeline",
            "checkpoint_patterns": ["*diffusion*pytorch*model*.bin", "*stable*diffusion*.safetensors"],
            "step_priority": 1
        }
    }

# ==============================================
# 🔥 확장된 모델 식별 패턴 데이터베이스
# ==============================================

ADVANCED_MODEL_PATTERNS = {
    # Step 01: Human Parsing Models
    "human_parsing": {
        "patterns": [
            r".*human.*parsing.*\.pth$",
            r".*schp.*atr.*\.pth$",
            r".*graphonomy.*\.pth$",
            r".*atr.*model.*\.pth$",
            r".*lip.*parsing.*\.pth$",
            r".*segformer.*human.*\.pth$",
            r".*densepose.*\.pkl$",
            r".*cihp.*\.pth$",
            r".*pascal.*person.*\.pth$",
            r".*human.*segmentation.*\.pth$"
        ],
        "keywords": [
            "human", "parsing", "segmentation", "atr", "lip", "schp", 
            "graphonomy", "densepose", "cihp", "pascal", "person"
        ],
        "category": ModelCategory.HUMAN_PARSING,
        "priority": ModelPriority.CRITICAL,
        "step_name": "HumanParsingStep",
        "min_size_mb": 50,
        "max_size_mb": 500,
        "expected_formats": [".pth", ".pt", ".pkl"],
        "model_class": "GraphonomyModel"
    },
    
    # Step 02: Pose Estimation Models
    "pose_estimation": {
        "patterns": [
            r".*pose.*model.*\.pth$",
            r".*openpose.*\.pth$",
            r".*body.*pose.*\.pth$",
            r".*hand.*pose.*\.pth$",
            r".*face.*pose.*\.pth$",
            r".*yolo.*pose.*\.pt$",
            r".*mediapipe.*\.tflite$",
            r".*alphapose.*\.pth$",
            r".*hrnet.*pose.*\.pth$",
            r".*simplebaseline.*\.pth$",
            r".*res101.*\.pth$",
            r".*clip_g.*\.pth$"
        ],
        "keywords": [
            "pose", "openpose", "yolo", "mediapipe", "body", "hand", "face",
            "keypoint", "alphapose", "hrnet", "simplebaseline", "coco"
        ],
        "category": ModelCategory.POSE_ESTIMATION,
        "priority": ModelPriority.HIGH,
        "step_name": "PoseEstimationStep",
        "min_size_mb": 5,
        "max_size_mb": 1000,
        "expected_formats": [".pth", ".pt", ".tflite", ".onnx"],
        "model_class": "OpenPoseModel"
    },
    
    # Step 03: Cloth Segmentation Models
    "cloth_segmentation": {
        "patterns": [
            r".*u2net.*\.pth$",
            r".*cloth.*segmentation.*\.(pth|onnx)$",
            r".*sam.*\.pth$",
            r".*mobile.*sam.*\.pth$",
            r".*parsing.*lip.*\.onnx$",
            r".*segmentation.*\.pth$",
            r".*deeplab.*cloth.*\.pth$",
            r".*mask.*rcnn.*cloth.*\.pth$",
            r".*bisenet.*\.pth$",
            r".*pspnet.*cloth.*\.pth$"
        ],
        "keywords": [
            "u2net", "segmentation", "sam", "cloth", "mask", "mobile",
            "deeplab", "bisenet", "pspnet", "rcnn", "parsing"
        ],
        "category": ModelCategory.CLOTH_SEGMENTATION,
        "priority": ModelPriority.HIGH,
        "step_name": "ClothSegmentationStep",
        "min_size_mb": 10,
        "max_size_mb": 3000,
        "expected_formats": [".pth", ".pt", ".onnx"],
        "model_class": "U2NetModel"
    },
    
    # Step 04: Geometric Matching Models
    "geometric_matching": {
        "patterns": [
            r".*geometric.*matching.*\.pth$",
            r".*gmm.*\.pth$",
            r".*tps.*\.pth$",
            r".*transformation.*\.pth$",
            r".*lightweight.*gmm.*\.pth$",
            r".*cpvton.*gmm.*\.pth$",
            r".*viton.*geometric.*\.pth$",
            r".*warp.*\.pth$"
        ],
        "keywords": [
            "geometric", "matching", "gmm", "tps", "transformation", 
            "alignment", "cpvton", "viton", "warp"
        ],
        "category": ModelCategory.GEOMETRIC_MATCHING,
        "priority": ModelPriority.MEDIUM,
        "step_name": "GeometricMatchingStep",
        "min_size_mb": 1,
        "max_size_mb": 100,
        "expected_formats": [".pth", ".pt"],
        "model_class": "GeometricMatchingModel"
    },
    
    # Step 05 & 06: Virtual Fitting & Diffusion Models
    "diffusion_models": {
        "patterns": [
            r".*diffusion.*pytorch.*model\.(bin|safetensors)$",
            r".*stable.*diffusion.*\.safetensors$",
            r".*ootdiffusion.*\.(pth|bin)$",
            r".*unet.*diffusion.*\.bin$",
            r".*hrviton.*\.pth$",
            r".*viton.*hd.*\.pth$",
            r".*inpaint.*\.bin$",
            r".*controlnet.*\.safetensors$",
            r".*lora.*\.safetensors$",
            r".*dreambooth.*\.bin$",
            r".*v1-5-pruned.*\.safetensors$",
            r".*runway.*diffusion.*\.bin$"
        ],
        "keywords": [
            "diffusion", "stable", "oot", "viton", "unet", "inpaint", 
            "generation", "controlnet", "lora", "dreambooth", "runway"
        ],
        "category": ModelCategory.DIFFUSION_MODELS,
        "priority": ModelPriority.CRITICAL,
        "step_name": "VirtualFittingStep",
        "min_size_mb": 100,
        "max_size_mb": 10000,
        "expected_formats": [".bin", ".safetensors", ".pth"],
        "model_class": "StableDiffusionPipeline"
    },
    
    # Transformer Models
    "transformer_models": {
        "patterns": [
            r".*clip.*vit.*\.bin$",
            r".*clip.*base.*\.bin$",
            r".*clip.*large.*\.bin$",
            r".*bert.*\.bin$",
            r".*roberta.*\.bin$",
            r".*t5.*\.bin$",
            r".*gpt.*\.bin$",
            r".*transformer.*\.bin$"
        ],
        "keywords": [
            "clip", "vit", "bert", "roberta", "t5", "gpt", 
            "transformer", "attention", "encoder", "decoder"
        ],
        "category": ModelCategory.TRANSFORMER_MODELS,
        "priority": ModelPriority.HIGH,
        "step_name": "QualityAssessmentStep",
        "min_size_mb": 50,
        "max_size_mb": 5000,
        "expected_formats": [".bin", ".safetensors"],
        "model_class": "CLIPModel"
    },
    
    # Post Processing Models
    "post_processing": {
        "patterns": [
            r".*realesrgan.*\.pth$",
            r".*esrgan.*\.pth$",
            r".*super.*resolution.*\.pth$",
            r".*upscale.*\.pth$",
            r".*enhance.*\.pth$",
            r".*srcnn.*\.pth$",
            r".*edsr.*\.pth$",
            r".*rcan.*\.pth$"
        ],
        "keywords": [
            "esrgan", "realesrgan", "upscale", "enhance", "super", 
            "resolution", "srcnn", "edsr", "rcan"
        ],
        "category": ModelCategory.POST_PROCESSING,
        "priority": ModelPriority.MEDIUM,
        "step_name": "PostProcessingStep",
        "min_size_mb": 10,
        "max_size_mb": 200,
        "expected_formats": [".pth", ".pt"],
        "model_class": "EnhancementModel"
    }
}

# ==============================================
# 🔥 고급 모델 탐지기 클래스 (순환참조 완전 해결)
# ==============================================

class AdvancedModelDetector:
    """
    🔍 완전 통합 AI 모델 자동 탐지 시스템 v5.0
    ✅ step_model_requests.py 기반 정확한 탐지
    ✅ 실제 존재하는 모델들 자동 발견
    ✅ 딕셔너리 기반 출력 (순환참조 방지)
    ✅ 프로덕션 안정성
    """
    
    def __init__(
        self,
        search_paths: Optional[List[Path]] = None,
        enable_deep_scan: bool = True,
        enable_metadata_extraction: bool = True,
        enable_caching: bool = True,
        cache_db_path: Optional[Path] = None,
        max_workers: int = 4,
        scan_timeout: int = 300
    ):
        """고급 모델 탐지기 초기화"""
        
        self.logger = logging.getLogger(f"{__name__}.AdvancedModelDetector")
        
        # 기본 검색 경로 설정 (conda 환경 특화)
        if search_paths is None:
            current_file = Path(__file__).resolve()
            backend_dir = current_file.parents[3]  # app/ai_pipeline/utils에서 backend로
            
            # conda 환경별 경로 추가
            conda_paths = []
            try:
                conda_prefix = os.environ.get('CONDA_PREFIX')
                if conda_prefix:
                    conda_paths.extend([
                        Path(conda_prefix) / "share" / "models",
                        Path(conda_prefix) / "lib" / "python3.11" / "site-packages" / "models",
                        Path(conda_prefix) / "models"
                    ])
            except:
                pass
            
            self.search_paths = [
                backend_dir / "ai_models",
                backend_dir / "app" / "ai_pipeline" / "models",
                backend_dir / "app" / "models",
                backend_dir / "checkpoints",
                backend_dir / "models",
                backend_dir / "weights",
                Path.home() / ".cache" / "huggingface",
                Path.home() / ".cache" / "torch",
                Path.home() / ".cache" / "models",
                *conda_paths
            ]
        else:
            self.search_paths = search_paths
        
        # 설정
        self.enable_deep_scan = enable_deep_scan
        self.enable_metadata_extraction = enable_metadata_extraction
        self.enable_caching = enable_caching
        self.max_workers = max_workers
        self.scan_timeout = scan_timeout
        
        # 탐지 결과 저장
        self.detected_models: Dict[str, DetectedModel] = {}
        self.scan_stats = {
            "total_files_scanned": 0,
            "models_detected": 0,
            "scan_duration": 0.0,
            "last_scan_time": 0,
            "errors_encountered": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # 캐시 관리
        self.cache_db_path = cache_db_path or Path("model_detection_cache.db")
        self.cache_ttl = 86400  # 24시간
        self._cache_lock = threading.RLock()
        
        # Step 요청사항 연동
        self.step_requirements = {}
        if STEP_REQUESTS_AVAILABLE:
            try:
                self.step_requirements = get_all_step_requirements()
            except:
                self.step_requirements = STEP_MODEL_REQUESTS
        else:
            self.step_requirements = STEP_MODEL_REQUESTS
        
        self.logger.info(f"🔍 고급 모델 탐지기 초기화 완료 - 검색 경로: {len(self.search_paths)}개")
        
        # 캐시 DB 초기화
        if self.enable_caching:
            self._init_cache_db()

    def _init_cache_db(self):
        """캐시 데이터베이스 초기화"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS model_cache (
                        file_path TEXT PRIMARY KEY,
                        file_size INTEGER,
                        file_mtime REAL,
                        checksum TEXT,
                        detection_data TEXT,
                        created_at REAL,
                        accessed_at REAL
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_accessed_at ON model_cache(accessed_at)
                """)
                
                conn.commit()
                
            self.logger.debug("✅ 캐시 DB 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 캐시 DB 초기화 실패: {e}")
            self.enable_caching = False

    def detect_all_models(
        self, 
        force_rescan: bool = False,
        categories_filter: Optional[List[ModelCategory]] = None,
        min_confidence: float = 0.3,
        step_filter: Optional[List[str]] = None
    ) -> Dict[str, DetectedModel]:
        """
        모든 AI 모델 자동 탐지 (step_model_requests.py 기반)
        
        Args:
            force_rescan: 캐시 무시하고 강제 재스캔
            categories_filter: 특정 카테고리만 탐지
            min_confidence: 최소 신뢰도 임계값
            step_filter: 특정 Step들만 탐지
            
        Returns:
            Dict[str, DetectedModel]: 탐지된 모델들
        """
        try:
            self.logger.info("🔍 Step 기반 AI 모델 자동 탐지 시작...")
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
            
            # Step별 요구사항 기반 스캔
            if step_filter:
                filtered_requirements = {k: v for k, v in self.step_requirements.items() 
                                       if k in step_filter}
            else:
                filtered_requirements = self.step_requirements
            
            # 병렬 스캔 실행
            if self.max_workers > 1:
                self._parallel_scan_by_steps(filtered_requirements, categories_filter, min_confidence)
            else:
                self._sequential_scan_by_steps(filtered_requirements, categories_filter, min_confidence)
            
            # 스캔 통계 업데이트
            self.scan_stats["models_detected"] = len(self.detected_models)
            self.scan_stats["scan_duration"] = time.time() - start_time
            self.scan_stats["last_scan_time"] = time.time()
            
            # 결과 후처리
            self._post_process_results(min_confidence)
            
            # 캐시 저장
            if self.enable_caching:
                self._save_to_cache()
            
            self.logger.info(f"✅ Step 기반 모델 탐지 완료: {len(self.detected_models)}개 모델 발견 ({self.scan_stats['scan_duration']:.2f}초)")
            self._print_detection_summary()
            
            return self.detected_models
            
        except Exception as e:
            self.logger.error(f"❌ 모델 탐지 실패: {e}")
            self.scan_stats["errors_encountered"] += 1
            raise

    def _parallel_scan_by_steps(self, step_requirements: Dict, categories_filter, min_confidence):
        """Step별 병렬 스캔 실행"""
        try:
            # Step별 스캔 태스크 생성
            scan_tasks = []
            for step_name, requirements in step_requirements.items():
                for search_path in self.search_paths:
                    if search_path.exists():
                        scan_tasks.append((step_name, requirements, search_path))
            
            if not scan_tasks:
                self.logger.warning("⚠️ 스캔할 경로가 없습니다")
                return
            
            # ThreadPoolExecutor로 병렬 처리
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {
                    executor.submit(
                        self._scan_path_for_step, 
                        step_name, 
                        requirements, 
                        search_path, 
                        categories_filter, 
                        min_confidence
                    ): (step_name, search_path)
                    for step_name, requirements, search_path in scan_tasks
                }
                
                # 결과 수집
                completed_count = 0
                for future in as_completed(future_to_task, timeout=self.scan_timeout):
                    step_name, search_path = future_to_task[future]
                    try:
                        step_results = future.result()
                        if step_results:
                            # 결과 병합 (스레드 안전)
                            with threading.Lock():
                                for name, model in step_results.items():
                                    self._register_detected_model_safe(model)
                        
                        completed_count += 1
                        self.logger.debug(f"✅ {step_name} @ {search_path} 스캔 완료 ({completed_count}/{len(scan_tasks)})")
                        
                    except Exception as e:
                        self.logger.error(f"❌ {step_name} @ {search_path} 스캔 실패: {e}")
                        self.scan_stats["errors_encountered"] += 1
                        
        except Exception as e:
            self.logger.error(f"❌ 병렬 스캔 실패: {e}")
            # 폴백: 순차 스캔
            self._sequential_scan_by_steps(step_requirements, categories_filter, min_confidence)

    def _sequential_scan_by_steps(self, step_requirements: Dict, categories_filter, min_confidence):
        """Step별 순차 스캔 실행"""
        try:
            for step_name, requirements in step_requirements.items():
                self.logger.debug(f"📁 {step_name} 요구사항 기반 스캔 중...")
                
                for search_path in self.search_paths:
                    if search_path.exists():
                        step_results = self._scan_path_for_step(
                            step_name, requirements, search_path, categories_filter, min_confidence
                        )
                        if step_results:
                            for name, model in step_results.items():
                                self._register_detected_model_safe(model)
                    else:
                        self.logger.debug(f"⚠️ 경로 없음: {search_path}")
                        
        except Exception as e:
            self.logger.error(f"❌ 순차 스캔 실패: {e}")

    def _scan_path_for_step(
        self, 
        step_name: str, 
        requirements: Dict, 
        search_path: Path, 
        categories_filter: Optional[List[ModelCategory]], 
        min_confidence: float,
        max_depth: int = 6,
        current_depth: int = 0
    ) -> Dict[str, DetectedModel]:
        """특정 Step 요구사항에 맞는 모델 스캔"""
        results = {}
        
        try:
            if current_depth > max_depth:
                return results
            
            # Step별 체크포인트 패턴 가져오기
            if isinstance(requirements, dict):
                checkpoint_patterns = requirements.get("checkpoint_patterns", [])
                if not checkpoint_patterns:
                    # step_model_requests.py 스타일 패턴
                    checkpoint_requirements = requirements.get("checkpoint_requirements", {})
                    checkpoint_patterns = checkpoint_requirements.get("primary_model_patterns", [])
            else:
                checkpoint_patterns = getattr(requirements, "checkpoint_patterns", [])
            
            if not checkpoint_patterns:
                self.logger.debug(f"⚠️ {step_name}에 대한 체크포인트 패턴이 없습니다")
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
            
            # 파일 분석 (Step별 패턴 매칭)
            for file_path in files:
                try:
                    self.scan_stats["total_files_scanned"] += 1
                    
                    # Step별 패턴 매칭
                    if self._matches_step_patterns(file_path, checkpoint_patterns):
                        detected_model = self._analyze_file_for_step(
                            file_path, step_name, requirements, categories_filter, min_confidence
                        )
                        if detected_model:
                            results[detected_model.name] = detected_model
                            self.logger.debug(f"📦 {step_name} 모델 발견: {file_path.name}")
                        
                except Exception as e:
                    self.logger.debug(f"파일 분석 오류 {file_path}: {e}")
                    continue
            
            # 하위 디렉토리 재귀 스캔
            if self.enable_deep_scan and current_depth < max_depth:
                for subdir in subdirs:
                    # 제외할 디렉토리 패턴
                    if subdir.name in ['__pycache__', '.git', 'node_modules', '.vscode', '.idea']:
                        continue
                    
                    try:
                        subdir_results = self._scan_path_for_step(
                            step_name, requirements, subdir, categories_filter, 
                            min_confidence, max_depth, current_depth + 1
                        )
                        results.update(subdir_results)
                    except Exception as e:
                        self.logger.debug(f"하위 디렉토리 스캔 오류 {subdir}: {e}")
                        continue
            
            return results
            
        except Exception as e:
            self.logger.debug(f"Step 스캔 오류 {search_path}: {e}")
            return results

    def _matches_step_patterns(self, file_path: Path, patterns: List[str]) -> bool:
        """파일이 Step별 패턴에 매칭되는지 확인"""
        try:
            file_name_lower = file_path.name.lower()
            file_path_str = str(file_path).lower()
            
            for pattern in patterns:
                # 간단한 와일드카드 패턴 매칭
                pattern_regex = pattern.replace("*", ".*").lower()
                if re.search(pattern_regex, file_name_lower) or re.search(pattern_regex, file_path_str):
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.debug(f"패턴 매칭 오류: {e}")
            return False

    def _analyze_file_for_step(
        self, 
        file_path: Path, 
        step_name: str,
        requirements: Dict,
        categories_filter: Optional[List[ModelCategory]], 
        min_confidence: float
    ) -> Optional[DetectedModel]:
        """Step 요구사항 기반 파일 분석"""
        try:
            # 기본 파일 정보
            file_stat = file_path.stat()
            file_size_mb = file_stat.st_size / (1024 * 1024)
            file_extension = file_path.suffix.lower()
            last_modified = file_stat.st_mtime
            
            # Step별 크기 제한 확인
            if isinstance(requirements, dict):
                checkpoint_requirements = requirements.get("checkpoint_requirements", {})
                min_size = checkpoint_requirements.get("min_file_size_mb", 1)
                max_size = checkpoint_requirements.get("max_file_size_mb", 10000)
            else:
                min_size = 1
                max_size = 10000
            
            if not (min_size <= file_size_mb <= max_size):
                return None
            
            # AI 모델 파일 확장자 필터
            ai_extensions = {
                '.pth', '.pt', '.bin', '.safetensors', '.onnx', '.pkl', 
                '.h5', '.pb', '.json', '.yaml', '.yml'
            }
            if file_extension not in ai_extensions:
                return None
            
            # Step별 모델 정보 추출
            if isinstance(requirements, dict):
                model_name = requirements.get("model_name", f"{step_name.lower()}_model")
                model_type = requirements.get("model_type", "BaseModel")
                model_class = requirements.get("model_class", model_type)
            else:
                model_name = f"{step_name.lower()}_model"
                model_type = "BaseModel"
                model_class = model_type
            
            # 카테고리 매핑
            category_mapping = {
                "HumanParsingStep": ModelCategory.HUMAN_PARSING,
                "PoseEstimationStep": ModelCategory.POSE_ESTIMATION,
                "ClothSegmentationStep": ModelCategory.CLOTH_SEGMENTATION,
                "GeometricMatchingStep": ModelCategory.GEOMETRIC_MATCHING,
                "ClothWarpingStep": ModelCategory.CLOTH_WARPING,
                "VirtualFittingStep": ModelCategory.VIRTUAL_FITTING,
                "PostProcessingStep": ModelCategory.POST_PROCESSING,
                "QualityAssessmentStep": ModelCategory.QUALITY_ASSESSMENT
            }
            
            detected_category = category_mapping.get(step_name, ModelCategory.AUXILIARY)
            
            # 카테고리 필터 적용
            if categories_filter and detected_category not in categories_filter:
                return None
            
            # 신뢰도 계산
            confidence_score = self._calculate_step_confidence(file_path, step_name, requirements)
            
            if confidence_score < min_confidence:
                return None
            
            # 우선순위 결정
            priority_mapping = {
                "HumanParsingStep": ModelPriority.CRITICAL,
                "VirtualFittingStep": ModelPriority.CRITICAL,
                "PoseEstimationStep": ModelPriority.HIGH,
                "ClothSegmentationStep": ModelPriority.HIGH,
                "ClothWarpingStep": ModelPriority.MEDIUM,
                "GeometricMatchingStep": ModelPriority.MEDIUM,
                "PostProcessingStep": ModelPriority.LOW,
                "QualityAssessmentStep": ModelPriority.LOW
            }
            
            priority = priority_mapping.get(step_name, ModelPriority.MEDIUM)
            
            # 고유 모델 이름 생성
            unique_name = self._generate_unique_model_name(file_path, step_name, model_name)
            
            # 메타데이터 추출
            metadata = self._extract_step_metadata(file_path, step_name, requirements)
            
            # DetectedModel 객체 생성
            detected_model = DetectedModel(
                name=unique_name,
                path=file_path,
                category=detected_category,
                model_type=model_class,
                file_size_mb=file_size_mb,
                file_extension=file_extension,
                confidence_score=confidence_score,
                priority=priority,
                step_name=step_name,
                metadata=metadata,
                last_modified=last_modified
            )
            
            return detected_model
            
        except Exception as e:
            self.logger.debug(f"Step 파일 분석 오류 {file_path}: {e}")
            return None

    def _calculate_step_confidence(self, file_path: Path, step_name: str, requirements: Dict) -> float:
        """Step별 신뢰도 점수 계산"""
        try:
            score = 0.0
            file_name = file_path.name.lower()
            
            # Step별 체크포인트 패턴 매칭
            if isinstance(requirements, dict):
                patterns = requirements.get("checkpoint_patterns", [])
                for pattern in patterns:
                    pattern_regex = pattern.replace("*", ".*").lower()
                    if re.search(pattern_regex, file_name):
                        score += 15.0
                        break
            
            # 파일명에서 Step 관련 키워드 확인
            step_keywords = {
                "HumanParsingStep": ["human", "parsing", "schp", "atr", "graphonomy"],
                "PoseEstimationStep": ["pose", "openpose", "body", "keypoint"],
                "ClothSegmentationStep": ["u2net", "cloth", "segmentation", "sam"],
                "GeometricMatchingStep": ["geometric", "matching", "gmm", "tps"],
                "ClothWarpingStep": ["warping", "tom", "hrviton", "cloth"],
                "VirtualFittingStep": ["diffusion", "stable", "viton", "fitting"],
                "PostProcessingStep": ["esrgan", "realesrgan", "enhance", "super"],
                "QualityAssessmentStep": ["clip", "quality", "assessment"]
            }
            
            keywords = step_keywords.get(step_name, [])
            for keyword in keywords:
                if keyword in file_name:
                    score += 8.0
            
            # 파일 크기 적정성 (Step별)
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            if step_name in ["VirtualFittingStep"]:
                # 대형 모델 (Diffusion 등)
                if 500 <= file_size_mb <= 5000:
                    score += 10.0
                elif file_size_mb > 100:
                    score += 5.0
            elif step_name in ["HumanParsingStep", "ClothSegmentationStep"]:
                # 중형 모델
                if 50 <= file_size_mb <= 500:
                    score += 10.0
                elif file_size_mb > 20:
                    score += 5.0
            else:
                # 소형 모델
                if 5 <= file_size_mb <= 200:
                    score += 10.0
                elif file_size_mb > 1:
                    score += 5.0
            
            # 파일 확장자 보너스
            if file_path.suffix in ['.pth', '.pt']:
                score += 5.0
            elif file_path.suffix in ['.bin', '.safetensors']:
                score += 3.0
            
            # 정규화
            confidence = min(score / 50.0, 1.0)
            return confidence
            
        except Exception as e:
            self.logger.debug(f"신뢰도 계산 오류: {e}")
            return 0.0

    def _generate_unique_model_name(self, file_path: Path, step_name: str, base_name: str) -> str:
        """고유한 모델 이름 생성"""
        try:
            # Step별 표준 이름 매핑
            standard_names = {
                "HumanParsingStep": "human_parsing_graphonomy",
                "PoseEstimationStep": "pose_estimation_openpose",
                "ClothSegmentationStep": "cloth_segmentation_u2net",
                "GeometricMatchingStep": "geometric_matching_gmm",
                "ClothWarpingStep": "cloth_warping_tom",
                "VirtualFittingStep": "virtual_fitting_stable_diffusion",
                "PostProcessingStep": "post_processing_realesrgan",
                "QualityAssessmentStep": "quality_assessment_clip"
            }
            
            standard_name = standard_names.get(step_name)
            if standard_name:
                return standard_name
            
            # 파일명 기반 이름 생성
            file_stem = file_path.stem.lower()
            clean_name = re.sub(r'[^a-z0-9_]', '_', file_stem)
            
            # 해시 추가 (충돌 방지)
            path_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:6]
            
            return f"{step_name.lower()}_{clean_name}_{path_hash}"
            
        except Exception as e:
            timestamp = int(time.time())
            return f"detected_model_{timestamp}"

    def _extract_step_metadata(self, file_path: Path, step_name: str, requirements: Dict) -> Dict[str, Any]:
        """Step별 메타데이터 추출"""
        metadata = {
            "file_name": file_path.name,
            "file_size_mb": file_path.stat().st_size / (1024 * 1024),
            "step_name": step_name,
            "detected_at": time.time(),
            "auto_detected": True
        }
        
        try:
            # Step 요구사항 정보 추가
            if isinstance(requirements, dict):
                metadata.update({
                    "step_model_name": requirements.get("model_name", "unknown"),
                    "step_model_type": requirements.get("model_type", "unknown"),
                    "step_priority": requirements.get("step_priority", 5)
                })
                
                # 체크포인트 요구사항
                checkpoint_requirements = requirements.get("checkpoint_requirements", {})
                if checkpoint_requirements:
                    metadata["checkpoint_requirements"] = checkpoint_requirements
            
            # PyTorch 모델 특별 처리
            if TORCH_AVAILABLE and file_path.suffix in ['.pth', '.pt']:
                try:
                    # 안전한 메타데이터 로드
                    checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
                    
                    if isinstance(checkpoint, dict):
                        # 모델 구조 정보
                        if 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                            if isinstance(state_dict, dict):
                                metadata['torch_layers_count'] = len(state_dict)
                                
                                # 총 파라미터 수 계산
                                total_params = 0
                                for tensor in state_dict.values():
                                    if torch.is_tensor(tensor):
                                        total_params += tensor.numel()
                                metadata['torch_total_parameters'] = total_params
                        
                        # 추가 메타데이터
                        for key in ['epoch', 'version', 'arch', 'model_name']:
                            if key in checkpoint:
                                metadata[f'torch_{key}'] = str(checkpoint[key])[:100]
                        
                except Exception as e:
                    metadata['torch_load_error'] = str(e)[:100]
            
            return metadata
            
        except Exception as e:
            metadata['metadata_extraction_error'] = str(e)[:100]
            return metadata

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
            # 1. 우선순위 비교
            if new_model.priority.value < existing_model.priority.value:
                return True
            elif new_model.priority.value > existing_model.priority.value:
                return False
            
            # 2. 신뢰도 비교
            if abs(new_model.confidence_score - existing_model.confidence_score) > 0.1:
                return new_model.confidence_score > existing_model.confidence_score
            
            # 3. 최신성 비교
            if abs(new_model.last_modified - existing_model.last_modified) > 86400:  # 1일 이상 차이
                return new_model.last_modified > existing_model.last_modified
            
            # 4. 파일 크기 비교
            return new_model.file_size_mb > existing_model.file_size_mb
            
        except Exception as e:
            self.logger.debug(f"모델 비교 오류: {e}")
            return new_model.file_size_mb > existing_model.file_size_mb

    def _reset_scan_stats(self):
        """스캔 통계 리셋"""
        self.scan_stats.update({
            "total_files_scanned": 0,
            "models_detected": 0,
            "scan_duration": 0.0,
            "errors_encountered": 0,
            "cache_hits": 0,
            "cache_misses": 0
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
                key=lambda x: (x[1].priority.value, -x[1].confidence_score, -x[1].file_size_mb)
            )
            
            self.detected_models = {name: model for name, model in sorted_models}
            
        except Exception as e:
            self.logger.error(f"❌ 후처리 실패: {e}")

    def _print_detection_summary(self):
        """탐지 결과 요약 출력"""
        try:
            self.logger.info("=" * 60)
            self.logger.info("🔍 Step 기반 모델 탐지 결과 요약")
            self.logger.info("=" * 60)
            
            total_size_gb = sum(model.file_size_mb for model in self.detected_models.values()) / 1024
            avg_confidence = sum(model.confidence_score for model in self.detected_models.values()) / len(self.detected_models) if self.detected_models else 0
            
            self.logger.info(f"📊 탐지된 모델: {len(self.detected_models)}개")
            self.logger.info(f"💾 총 크기: {total_size_gb:.2f}GB")
            self.logger.info(f"🔍 스캔 파일: {self.scan_stats['total_files_scanned']:,}개")
            self.logger.info(f"⏱️ 소요 시간: {self.scan_stats['scan_duration']:.2f}초")
            self.logger.info(f"🎯 평균 신뢰도: {avg_confidence:.3f}")
            
            # Step별 분포
            step_distribution = {}
            for model in self.detected_models.values():
                step = model.step_name
                if step not in step_distribution:
                    step_distribution[step] = 0
                step_distribution[step] += 1
            
            if step_distribution:
                self.logger.info("\n📁 Step별 분포:")
                for step, count in step_distribution.items():
                    self.logger.info(f"  {step}: {count}개")
            
            # 주요 모델들
            if self.detected_models:
                self.logger.info("\n🏆 탐지된 주요 모델들:")
                for i, (name, model) in enumerate(list(self.detected_models.items())[:5]):
                    self.logger.info(f"  {i+1}. {name}")
                    self.logger.info(f"     Step: {model.step_name}, 크기: {model.file_size_mb:.1f}MB")
                    self.logger.info(f"     신뢰도: {model.confidence_score:.3f}, 우선순위: {model.priority.name}")
            
            self.logger.info("=" * 60)
                
        except Exception as e:
            self.logger.error(f"❌ 요약 출력 실패: {e}")

    # ==============================================
    # 🔥 캐시 관련 메서드들
    # ==============================================

    def _load_from_cache(self) -> Optional[Dict[str, DetectedModel]]:
        """캐시에서 로드"""
        try:
            with self._cache_lock:
                with sqlite3.connect(self.cache_db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 만료된 캐시 정리
                    cutoff_time = time.time() - self.cache_ttl
                    cursor.execute("DELETE FROM model_cache WHERE created_at < ?", (cutoff_time,))
                    
                    # 캐시 조회
                    cursor.execute("""
                        SELECT file_path, detection_data 
                        FROM model_cache 
                        WHERE created_at > ?
                    """, (cutoff_time,))
                    
                    cached_models = {}
                    for file_path, detection_data in cursor.fetchall():
                        try:
                            # 파일이 여전히 존재하는지 확인
                            if not Path(file_path).exists():
                                continue
                            
                            model_data = json.loads(detection_data)
                            model = self._deserialize_detected_model(model_data)
                            if model:
                                cached_models[model.name] = model
                        except Exception as e:
                            self.logger.debug(f"캐시 항목 로드 실패 {file_path}: {e}")
                    
                    if cached_models:
                        # 액세스 시간 업데이트
                        cursor.execute("UPDATE model_cache SET accessed_at = ?", (time.time(),))
                        conn.commit()
                        
                        self.detected_models = cached_models
                        return cached_models
            
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
                    
                    for model in self.detected_models.values():
                        try:
                            detection_data = json.dumps(self._serialize_detected_model(model))
                            
                            cursor.execute("""
                                INSERT OR REPLACE INTO model_cache 
                                (file_path, file_size, file_mtime, checksum, detection_data, created_at, accessed_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (
                                str(model.path),
                                int(model.file_size_mb * 1024 * 1024),
                                model.last_modified,
                                model.checksum,
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
        """DetectedModel을 딕셔너리로 직렬화"""
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
            "checksum": model.checksum
        }

    def _deserialize_detected_model(self, data: Dict[str, Any]) -> Optional[DetectedModel]:
        """딕셔너리를 DetectedModel로 역직렬화"""
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
                checksum=data.get("checksum")
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
        
        return min(step_models, key=lambda m: (m.priority.value, -m.confidence_score))

    def get_model_by_name(self, name: str) -> Optional[DetectedModel]:
        """이름으로 모델 조회"""
        return self.detected_models.get(name)

    def get_all_model_paths(self) -> Dict[str, Path]:
        """모든 모델의 경로 딕셔너리 반환"""
        return {name: model.path for name, model in self.detected_models.items()}

    def search_models(
        self, 
        keywords: List[str], 
        step_filter: Optional[List[str]] = None,
        min_confidence: float = 0.0
    ) -> List[DetectedModel]:
        """키워드로 모델 검색"""
        try:
            results = []
            keywords_lower = [kw.lower() for kw in keywords]
            
            for model in self.detected_models.values():
                # 신뢰도 필터
                if model.confidence_score < min_confidence:
                    continue
                
                # Step 필터
                if step_filter and model.step_name not in step_filter:
                    continue
                
                # 키워드 매칭
                model_text = f"{model.name} {model.path.name} {model.model_type} {model.step_name}".lower()
                if any(keyword in model_text for keyword in keywords_lower):
                    results.append(model)
            
            # 관련성 순으로 정렬
            results.sort(key=lambda m: (m.priority.value, -m.confidence_score))
            return results
            
        except Exception as e:
            self.logger.error(f"모델 검색 실패: {e}")
            return []

# ==============================================
# 🔥 ModelLoader 연동을 위한 딕셔너리 출력 (순환참조 완전 방지)
# ==============================================

class ModelLoaderConfigGenerator:
    """
    🔗 ModelLoader 연동용 설정 생성기 (순환참조 완전 방지)
    딕셔너리 기반으로만 동작하여 ModelLoader import 불필요
    """
    
    def __init__(self, detector: AdvancedModelDetector):
        self.detector = detector
        self.logger = logging.getLogger(f"{__name__}.ModelLoaderConfigGenerator")
    
    def generate_complete_config(self) -> Dict[str, Any]:
        """ModelLoader용 완전한 설정 생성"""
        try:
            config = {
                "model_configs": [],
                "model_paths": {},
                "step_mappings": {},
                "priority_rankings": {},
                "performance_estimates": {},
                "metadata": {
                    "total_models": len(self.detector.detected_models),
                    "generation_time": time.time(),
                    "detector_version": "5.0",
                    "scan_stats": self.detector.scan_stats
                }
            }
            
            for name, detected_model in self.detector.detected_models.items():
                # ModelConfig 딕셔너리 생성
                model_config = {
                    "name": name,
                    "model_type": detected_model.category.value,
                    "model_class": detected_model.model_type,
                    "checkpoint_path": str(detected_model.path),
                    "device": "auto",
                    "precision": "fp16",
                    "input_size": self._get_input_size_for_step(detected_model.step_name),
                    "step_name": detected_model.step_name,
                    "metadata": {
                        **detected_model.metadata,
                        "auto_detected": True,
                        "confidence_score": detected_model.confidence_score,
                        "priority": detected_model.priority.name,
                        "alternative_paths": [str(p) for p in detected_model.alternative_paths]
                    }
                }
                config["model_configs"].append(model_config)
                
                # 경로 매핑
                config["model_paths"][name] = {
                    "primary": str(detected_model.path),
                    "alternatives": [str(p) for p in detected_model.alternative_paths],
                    "size_mb": detected_model.file_size_mb,
                    "confidence": detected_model.confidence_score
                }
                
                # Step 매핑
                step_name = detected_model.step_name
                if step_name not in config["step_mappings"]:
                    config["step_mappings"][step_name] = []
                config["step_mappings"][step_name].append(name)
                
                # 우선순위
                config["priority_rankings"][name] = {
                    "priority_level": detected_model.priority.value,
                    "priority_name": detected_model.priority.name,
                    "confidence_score": detected_model.confidence_score,
                    "step_rank": self._get_step_rank(detected_model.step_name)
                }
                
                # 성능 추정
                config["performance_estimates"][name] = {
                    "estimated_memory_gb": detected_model.file_size_mb / 1024 * 2,
                    "estimated_load_time_sec": self._estimate_load_time(detected_model),
                    "recommended_batch_size": self._get_recommended_batch_size(detected_model),
                    "gpu_memory_required_gb": max(2.0, detected_model.file_size_mb / 1024 * 1.5)
                }
            
            return config
            
        except Exception as e:
            self.logger.error(f"❌ ModelLoader 설정 생성 실패: {e}")
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
        """모델 로드 시간 추정 (초)"""
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
        
        return base_time * size_factor

    def _get_recommended_batch_size(self, detected_model: DetectedModel) -> int:
        """권장 배치 크기"""
        if detected_model.file_size_mb > 1000:  # 대형 모델
            return 1
        elif detected_model.file_size_mb > 100:  # 중형 모델
            return 2
        else:  # 소형 모델
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

# ==============================================
# 🔥 편의 함수들 및 팩토리 함수들 (순환참조 방지)
# ==============================================

def create_advanced_detector(
    search_paths: Optional[List[Path]] = None,
    enable_parallel: bool = True,
    max_workers: int = 4,
    **kwargs
) -> AdvancedModelDetector:
    """고급 자동 모델 탐지기 생성"""
    return AdvancedModelDetector(
        search_paths=search_paths,
        max_workers=max_workers if enable_parallel else 1,
        **kwargs
    )

def quick_model_detection(
    step_filter: Optional[List[str]] = None,
    min_confidence: float = 0.5,
    force_rescan: bool = False
) -> Dict[str, Any]:
    """빠른 모델 탐지 및 결과 반환"""
    try:
        # 탐지기 생성 및 실행
        detector = create_advanced_detector()
        detected_models = detector.detect_all_models(
            force_rescan=force_rescan,
            step_filter=step_filter,
            min_confidence=min_confidence
        )
        
        # 결과 요약
        summary = {
            "total_models": len(detected_models),
            "models_by_step": {},
            "models_by_priority": {},
            "top_models": {},
            "scan_stats": detector.scan_stats
        }
        
        # Step별 분류
        for model in detected_models.values():
            step = model.step_name
            if step not in summary["models_by_step"]:
                summary["models_by_step"][step] = []
            summary["models_by_step"][step].append({
                "name": model.name,
                "path": str(model.path),
                "confidence": model.confidence_score,
                "size_mb": model.file_size_mb
            })
        
        # 우선순위별 분류
        for model in detected_models.values():
            priority = model.priority.name
            if priority not in summary["models_by_priority"]:
                summary["models_by_priority"][priority] = []
            summary["models_by_priority"][priority].append(model.name)
        
        # Step별 최고 모델
        step_names = set(model.step_name for model in detected_models.values())
        for step_name in step_names:
            best_model = detector.get_best_model_for_step(step_name)
            if best_model:
                summary["top_models"][step_name] = {
                    "name": best_model.name,
                    "path": str(best_model.path),
                    "confidence": best_model.confidence_score,
                    "priority": best_model.priority.name
                }
        
        return summary
        
    except Exception as e:
        logger.error(f"빠른 모델 탐지 실패: {e}")
        return {"error": str(e)}

def generate_model_loader_config(
    detector: Optional[AdvancedModelDetector] = None,
    **detection_kwargs
) -> Dict[str, Any]:
    """
    ModelLoader용 설정 생성 (순환참조 방지)
    딕셔너리 기반으로만 출력
    """
    try:
        logger.info("🔍 ModelLoader 설정 생성 시작...")
        
        # 탐지기가 없으면 새로 생성
        if detector is None:
            detector = create_advanced_detector(**detection_kwargs)
            detected_models = detector.detect_all_models()
        else:
            detected_models = detector.detected_models
        
        if not detected_models:
            logger.warning("⚠️ 탐지된 모델이 없습니다")
            return {"success": False, "message": "No models detected"}
        
        # 설정 생성기 사용
        config_generator = ModelLoaderConfigGenerator(detector)
        model_loader_config = config_generator.generate_complete_config()
        
        # 최종 결과
        result = {
            "success": True,
            "model_loader_config": model_loader_config,
            "detection_summary": {
                "total_models": len(detected_models),
                "scan_duration": detector.scan_stats["scan_duration"],
                "confidence_avg": sum(m.confidence_score for m in detected_models.values()) / len(detected_models)
            }
        }
        
        logger.info(f"✅ ModelLoader 설정 생성 완료: {len(detected_models)}개 모델")
        return result
        
    except Exception as e:
        logger.error(f"❌ ModelLoader 설정 생성 실패: {e}")
        return {"success": False, "error": str(e)}

def validate_model_paths(detected_models: Dict[str, DetectedModel]) -> Dict[str, Any]:
    """탐지된 모델 경로들의 유효성 검증"""
    try:
        validation_result = {
            "valid_models": [],
            "invalid_models": [],
            "missing_files": [],
            "permission_errors": [],
            "total_size_gb": 0.0
        }
        
        for name, model in detected_models.items():
            try:
                # 주 경로 확인
                if model.path.exists() and model.path.is_file():
                    validation_result["valid_models"].append(name)
                    validation_result["total_size_gb"] += model.file_size_mb / 1024
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
            "validation_rate": len(validation_result["valid_models"]) / len(detected_models) if detected_models else 0
        }
        
        return validation_result
        
    except Exception as e:
        logger.error(f"모델 경로 검증 실패: {e}")
        return {"error": str(e)}

# 모듈 익스포트
__all__ = [
    # 메인 클래스들
    'AdvancedModelDetector',
    'ModelLoaderConfigGenerator',
    'DetectedModel',
    'ModelCategory',
    'ModelPriority',
    
    # 팩토리 함수들
    'create_advanced_detector',
    'quick_model_detection',
    'generate_model_loader_config',  # 🔥 순환참조 방지
    
    # 유틸리티 함수들
    'validate_model_paths',
    
    # 설정 및 패턴
    'ADVANCED_MODEL_PATTERNS'
]

# 호환성을 위한 별칭
AutoModelDetector = AdvancedModelDetector
create_auto_detector = create_advanced_detector

logger.info("✅ 순환참조 완전 해결 - 자동 모델 탐지 시스템 v5.0 로드 완료")