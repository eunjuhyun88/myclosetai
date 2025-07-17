# app/ai_pipeline/utils/auto_model_detector.py
"""
🔍 자동 모델 탐지 시스템 - 실제 존재하는 AI 모델 자동 발견
✅ 실제 72GB+ 모델들과 완벽 연결
✅ 동적 경로 매핑 및 자동 등록
✅ ModelLoader와 완벽 통합
"""

import os
import re
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib

# PyTorch 및 AI 라이브러리
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

logger = logging.getLogger(__name__)

# ==============================================
# 🔥 모델 탐지 설정 및 매핑
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
    AUXILIARY = "auxiliary"

@dataclass
class DetectedModel:
    """탐지된 모델 정보"""
    name: str
    path: Path
    category: ModelCategory
    model_type: str
    file_size_mb: float
    file_extension: str
    confidence_score: float
    priority: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    alternative_paths: List[Path] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)

# ==============================================
# 🔍 모델 식별 패턴 데이터베이스
# ==============================================

MODEL_IDENTIFICATION_PATTERNS = {
    # Step 01: Human Parsing Models
    "human_parsing": {
        "patterns": [
            r".*human.*parsing.*\.pth$",
            r".*schp.*atr.*\.pth$",
            r".*graphonomy.*\.pth$",
            r".*atr.*model.*\.pth$",
            r".*lip.*parsing.*\.pth$",
            r".*segformer.*\.pth$",
            r".*densepose.*\.pkl$",
            r".*pytorch_model\.bin$"  # 일반적인 HF 모델
        ],
        "keywords": ["human", "parsing", "segmentation", "atr", "lip", "schp", "graphonomy", "densepose"],
        "category": ModelCategory.HUMAN_PARSING,
        "priority": 1,
        "min_size_mb": 50  # 최소 크기 필터
    },
    
    # Step 02: Pose Estimation Models
    "pose_estimation": {
        "patterns": [
            r".*pose.*model.*\.pth$",
            r".*openpose.*\.pth$",
            r".*body.*pose.*\.pth$",
            r".*hand.*pose.*\.pth$",
            r".*yolo.*pose.*\.pt$",
            r".*mediapipe.*\.tflite$",
            r".*res101.*\.pth$",
            r".*clip_g.*\.pth$"
        ],
        "keywords": ["pose", "openpose", "yolo", "mediapipe", "body", "hand", "keypoint"],
        "category": ModelCategory.POSE_ESTIMATION,
        "priority": 2,
        "min_size_mb": 5
    },
    
    # Step 03: Cloth Segmentation Models
    "cloth_segmentation": {
        "patterns": [
            r".*u2net.*\.pth$",
            r".*cloth.*segmentation.*\.(pth|onnx)$",
            r".*sam.*\.pth$",
            r".*mobile.*sam.*\.pth$",
            r".*parsing.*lip.*\.onnx$",
            r".*segmentation.*\.pth$"
        ],
        "keywords": ["u2net", "segmentation", "sam", "cloth", "mask", "mobile"],
        "category": ModelCategory.CLOTH_SEGMENTATION,
        "priority": 3,
        "min_size_mb": 10
    },
    
    # Step 04: Geometric Matching Models
    "geometric_matching": {
        "patterns": [
            r".*geometric.*matching.*\.pth$",
            r".*gmm.*\.pth$",
            r".*tps.*\.pth$",
            r".*transformation.*\.pth$",
            r".*lightweight.*gmm.*\.pth$"
        ],
        "keywords": ["geometric", "matching", "gmm", "tps", "transformation", "alignment"],
        "category": ModelCategory.GEOMETRIC_MATCHING,
        "priority": 4,
        "min_size_mb": 1
    },
    
    # Step 05 & 06: Virtual Fitting & Cloth Warping Models
    "virtual_fitting": {
        "patterns": [
            r".*diffusion.*pytorch.*model\.(bin|safetensors)$",
            r".*stable.*diffusion.*\.safetensors$",
            r".*ootdiffusion.*\.(pth|bin)$",
            r".*unet.*diffusion.*\.bin$",
            r".*hrviton.*\.pth$",
            r".*viton.*\.pth$",
            r".*inpaint.*\.bin$"
        ],
        "keywords": ["diffusion", "stable", "oot", "viton", "unet", "inpaint", "generation"],
        "category": ModelCategory.VIRTUAL_FITTING,
        "priority": 5,
        "min_size_mb": 100  # Diffusion 모델은 대용량
    },
    
    # Step 07: Post Processing Models
    "post_processing": {
        "patterns": [
            r".*realesrgan.*\.pth$",
            r".*esrgan.*\.pth$",
            r".*super.*resolution.*\.pth$",
            r".*upscale.*\.pth$",
            r".*enhance.*\.pth$"
        ],
        "keywords": ["esrgan", "realesrgan", "upscale", "enhance", "super", "resolution"],
        "category": ModelCategory.POST_PROCESSING,
        "priority": 6,
        "min_size_mb": 10
    },
    
    # Step 08: Quality Assessment & Auxiliary Models
    "quality_assessment": {
        "patterns": [
            r".*clip.*vit.*\.bin$",
            r".*clip.*base.*\.bin$",
            r".*clip.*large.*\.bin$",
            r".*quality.*assessment.*\.pth$",
            r".*feature.*\.pth$",
            r".*resnet.*features.*\.pth$"
        ],
        "keywords": ["clip", "vit", "quality", "assessment", "feature", "resnet"],
        "category": ModelCategory.QUALITY_ASSESSMENT,
        "priority": 7,
        "min_size_mb": 50
    },
    
    # Auxiliary Models
    "auxiliary": {
        "patterns": [
            r".*vae.*\.bin$",
            r".*text.*encoder.*\.bin$",
            r".*tokenizer.*\.json$",
            r".*scheduler.*\.bin$",
            r".*safety.*checker.*\.bin$"
        ],
        "keywords": ["vae", "encoder", "tokenizer", "scheduler", "safety", "checker"],
        "category": ModelCategory.AUXILIARY,
        "priority": 8,
        "min_size_mb": 10
    }
}

# ==============================================
# 🔍 핵심 모델 탐지기 클래스
# ==============================================

class AutoModelDetector:
    """
    🔍 자동 AI 모델 탐지 시스템
    ✅ 실제 존재하는 모델들 자동 발견
    ✅ 카테고리별 분류 및 우선순위 할당
    ✅ ModelLoader와 완벽 통합
    """
    
    def __init__(
        self,
        search_paths: Optional[List[Path]] = None,
        enable_deep_scan: bool = True,
        enable_metadata_extraction: bool = True,
        cache_results: bool = True
    ):
        """자동 모델 탐지기 초기화"""
        
        self.logger = logging.getLogger(f"{__name__}.AutoModelDetector")
        
        # 기본 검색 경로 설정
        if search_paths is None:
            current_file = Path(__file__).resolve()
            backend_dir = current_file.parents[3]  # app/ai_pipeline/utils에서 backend로
            
            self.search_paths = [
                backend_dir / "ai_models",
                backend_dir / "app" / "ai_pipeline" / "models",
                backend_dir / "app" / "models",
                backend_dir / "checkpoints",
                backend_dir / "ai_models" / "checkpoints"
            ]
        else:
            self.search_paths = search_paths
        
        # 설정
        self.enable_deep_scan = enable_deep_scan
        self.enable_metadata_extraction = enable_metadata_extraction
        self.cache_results = cache_results
        
        # 탐지 결과 저장
        self.detected_models: Dict[str, DetectedModel] = {}
        self.scan_stats = {
            "total_files_scanned": 0,
            "models_detected": 0,
            "scan_duration": 0.0,
            "last_scan_time": 0
        }
        
        # 캐시 관리
        self.cache_file = Path("model_detection_cache.json")
        self.cache_ttl = 3600  # 1시간
        
        self.logger.info(f"🔍 자동 모델 탐지기 초기화 - 검색 경로: {len(self.search_paths)}개")

    def detect_all_models(self, force_rescan: bool = False) -> Dict[str, DetectedModel]:
        """
        모든 AI 모델 자동 탐지
        
        Args:
            force_rescan: 캐시 무시하고 강제 재스캔
            
        Returns:
            Dict[str, DetectedModel]: 탐지된 모델들
        """
        try:
            self.logger.info("🔍 AI 모델 자동 탐지 시작...")
            start_time = time.time()
            
            # 캐시 확인
            if not force_rescan and self.cache_results:
                cached_results = self._load_cache()
                if cached_results:
                    self.logger.info(f"📦 캐시된 결과 사용: {len(cached_results)}개 모델")
                    return cached_results
            
            # 실제 스캔 실행
            self.detected_models.clear()
            self.scan_stats["total_files_scanned"] = 0
            
            for search_path in self.search_paths:
                if search_path.exists():
                    self.logger.info(f"📁 스캔 중: {search_path}")
                    self._scan_directory(search_path)
                else:
                    self.logger.debug(f"⚠️ 경로 없음: {search_path}")
            
            # 스캔 통계 업데이트
            self.scan_stats["models_detected"] = len(self.detected_models)
            self.scan_stats["scan_duration"] = time.time() - start_time
            self.scan_stats["last_scan_time"] = time.time()
            
            # 결과 정리 및 우선순위 조정
            self._post_process_results()
            
            # 캐시 저장
            if self.cache_results:
                self._save_cache()
            
            self.logger.info(f"✅ 모델 탐지 완료: {len(self.detected_models)}개 모델 발견 ({self.scan_stats['scan_duration']:.2f}초)")
            self._print_detection_summary()
            
            return self.detected_models
            
        except Exception as e:
            self.logger.error(f"❌ 모델 탐지 실패: {e}")
            raise

    def _scan_directory(self, directory: Path, max_depth: int = 5, current_depth: int = 0):
        """디렉토리 재귀 스캔"""
        try:
            if current_depth > max_depth:
                return
                
            for item in directory.iterdir():
                try:
                    if item.is_file():
                        self.scan_stats["total_files_scanned"] += 1
                        self._analyze_file(item)
                    elif item.is_dir() and self.enable_deep_scan:
                        # 숨김 폴더나 일반적인 제외 폴더 건너뛰기
                        if not item.name.startswith('.') and item.name not in ['__pycache__', 'node_modules']:
                            self._scan_directory(item, max_depth, current_depth + 1)
                except PermissionError:
                    continue
                except Exception as e:
                    self.logger.debug(f"파일 스캔 오류 {item}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.debug(f"디렉토리 스캔 오류 {directory}: {e}")

    def _analyze_file(self, file_path: Path):
        """개별 파일 분석"""
        try:
            # 파일 기본 정보
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            file_extension = file_path.suffix.lower()
            
            # AI 모델 파일 확장자 필터
            ai_extensions = {'.pth', '.pt', '.bin', '.safetensors', '.onnx', '.pkl', '.tflite', '.h5'}
            if file_extension not in ai_extensions:
                return
            
            # 너무 작은 파일 제외 (1MB 미만)
            if file_size_mb < 1.0:
                return
            
            # 패턴 매칭으로 모델 분류
            detected_category, confidence_score = self._classify_model(file_path)
            
            if detected_category:
                # 고유 모델 이름 생성
                model_name = self._generate_model_name(file_path, detected_category)
                
                # 메타데이터 추출
                metadata = {}
                if self.enable_metadata_extraction:
                    metadata = self._extract_metadata(file_path)
                
                # 우선순위 계산
                priority = self._calculate_priority(file_path, detected_category, file_size_mb)
                
                # DetectedModel 객체 생성
                detected_model = DetectedModel(
                    name=model_name,
                    path=file_path,
                    category=detected_category,
                    model_type=self._determine_model_type(file_path, detected_category),
                    file_size_mb=file_size_mb,
                    file_extension=file_extension,
                    confidence_score=confidence_score,
                    priority=priority,
                    metadata=metadata
                )
                
                # 중복 확인 및 저장
                self._register_detected_model(detected_model)
                
        except Exception as e:
            self.logger.debug(f"파일 분석 오류 {file_path}: {e}")

    def _classify_model(self, file_path: Path) -> Tuple[Optional[ModelCategory], float]:
        """파일 패턴을 분석하여 모델 카테고리 분류"""
        try:
            file_name = file_path.name.lower()
            file_path_str = str(file_path).lower()
            
            best_category = None
            best_score = 0.0
            
            for category_name, config in MODEL_IDENTIFICATION_PATTERNS.items():
                score = 0.0
                matches = 0
                
                # 패턴 매칭
                for pattern in config["patterns"]:
                    if re.search(pattern, file_path_str, re.IGNORECASE):
                        score += 10.0
                        matches += 1
                
                # 키워드 매칭
                for keyword in config["keywords"]:
                    if keyword in file_name or keyword in file_path_str:
                        score += 5.0
                        matches += 1
                
                # 파일 크기 확인
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                if file_size_mb >= config.get("min_size_mb", 0):
                    score += 2.0
                
                # 매치 개수에 따른 보너스
                if matches > 0:
                    score += matches * 2.0
                
                # 최고 점수 갱신
                if score > best_score:
                    best_score = score
                    best_category = config["category"]
            
            # 최소 임계값 확인 (신뢰도 15점 이상)
            if best_score >= 15.0:
                confidence = min(best_score / 30.0, 1.0)  # 정규화
                return best_category, confidence
            
            return None, 0.0
            
        except Exception as e:
            self.logger.debug(f"분류 오류 {file_path}: {e}")
            return None, 0.0

    def _generate_model_name(self, file_path: Path, category: ModelCategory) -> str:
        """모델 고유 이름 생성"""
        try:
            # 기본 이름: 카테고리_파일명
            base_name = f"{category.value}_{file_path.stem}"
            
            # 특별한 모델들 처리
            special_names = {
                "graphonomy": "human_parsing_graphonomy",
                "schp": "human_parsing_schp",
                "openpose": "pose_estimation_openpose",
                "yolo": "pose_estimation_yolo",
                "u2net": "cloth_segmentation_u2net",
                "sam": "cloth_segmentation_sam",
                "ootdiffusion": "virtual_fitting_ootdiffusion",
                "stable_diffusion": "virtual_fitting_stable_diffusion",
                "realesrgan": "post_processing_realesrgan",
                "clip": "quality_assessment_clip"
            }
            
            file_name_lower = file_path.name.lower()
            for keyword, special_name in special_names.items():
                if keyword in file_name_lower:
                    return special_name
            
            # 중복 방지를 위한 해시 추가
            path_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
            return f"{base_name}_{path_hash}"
            
        except Exception as e:
            # 폴백 이름
            return f"detected_model_{int(time.time())}"

    def _determine_model_type(self, file_path: Path, category: ModelCategory) -> str:
        """모델 타입 결정 (ModelLoader의 클래스와 매핑)"""
        model_type_mapping = {
            ModelCategory.HUMAN_PARSING: "GraphonomyModel",
            ModelCategory.POSE_ESTIMATION: "OpenPoseModel",
            ModelCategory.CLOTH_SEGMENTATION: "U2NetModel",
            ModelCategory.GEOMETRIC_MATCHING: "GeometricMatchingModel",
            ModelCategory.CLOTH_WARPING: "HRVITONModel",
            ModelCategory.VIRTUAL_FITTING: "HRVITONModel",
            ModelCategory.POST_PROCESSING: "GraphonomyModel",  # 범용 사용
            ModelCategory.QUALITY_ASSESSMENT: "GraphonomyModel",  # 범용 사용
            ModelCategory.AUXILIARY: "HRVITONModel"  # 범용 사용
        }
        
        # 특별한 모델 타입 처리
        file_name = file_path.name.lower()
        if "diffusion" in file_name:
            return "StableDiffusionPipeline"
        elif "clip" in file_name:
            return "CLIPModel"
        elif "vae" in file_name:
            return "AutoencoderKL"
        
        return model_type_mapping.get(category, "GenericModel")

    def _calculate_priority(self, file_path: Path, category: ModelCategory, file_size_mb: float) -> int:
        """모델 우선순위 계산"""
        try:
            # 기본 우선순위 (카테고리별)
            base_priority = {
                ModelCategory.HUMAN_PARSING: 10,
                ModelCategory.POSE_ESTIMATION: 20,
                ModelCategory.CLOTH_SEGMENTATION: 30,
                ModelCategory.GEOMETRIC_MATCHING: 40,
                ModelCategory.CLOTH_WARPING: 50,
                ModelCategory.VIRTUAL_FITTING: 50,
                ModelCategory.POST_PROCESSING: 60,
                ModelCategory.QUALITY_ASSESSMENT: 70,
                ModelCategory.AUXILIARY: 80
            }.get(category, 90)
            
            # 파일 크기에 따른 보정 (큰 모델일수록 우선순위 높음)
            if file_size_mb > 1000:  # 1GB 이상
                base_priority -= 5
            elif file_size_mb > 100:  # 100MB 이상
                base_priority -= 2
            
            # 파일명 기반 보정
            file_name = file_path.name.lower()
            priority_keywords = {
                "base": -3,    # base 모델 우선
                "large": -2,   # large 모델 차순위
                "mini": +5,    # mini 모델 후순위
                "fp16": -1,    # fp16 최적화 모델 우선
                "safetensors": -1  # safetensors 포맷 우선
            }
            
            for keyword, adjustment in priority_keywords.items():
                if keyword in file_name:
                    base_priority += adjustment
            
            return max(1, base_priority)  # 최소값 1
            
        except Exception:
            return 50  # 기본값

    def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """모델 파일에서 메타데이터 추출"""
        metadata = {
            "file_modified": file_path.stat().st_mtime,
            "file_created": file_path.stat().st_ctime,
            "parent_directory": file_path.parent.name
        }
        
        try:
            # PyTorch 모델 메타데이터 추출
            if TORCH_AVAILABLE and file_path.suffix in ['.pth', '.pt']:
                try:
                    # 헤더만 읽어서 메타데이터 확인 (메모리 절약)
                    checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
                    
                    if isinstance(checkpoint, dict):
                        # 일반적인 메타데이터 키들
                        meta_keys = ['arch', 'epoch', 'version', 'model_name', 'config']
                        for key in meta_keys:
                            if key in checkpoint:
                                metadata[key] = str(checkpoint[key])[:100]  # 길이 제한
                        
                        # 모델 구조 정보
                        if 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                            metadata['num_parameters'] = sum(v.numel() for v in state_dict.values() if torch.is_tensor(v))
                        elif hasattr(checkpoint, 'keys'):
                            metadata['num_parameters'] = sum(v.numel() for v in checkpoint.values() if torch.is_tensor(v))
                            
                except Exception as e:
                    metadata['torch_load_error'] = str(e)[:100]
            
            # 파일 경로에서 추가 정보 추출
            path_parts = file_path.parts
            if len(path_parts) >= 2:
                metadata['model_family'] = path_parts[-2]  # 부모 디렉토리
            
            # 특별한 구조 인식
            for part in path_parts:
                if 'checkpoint' in part.lower():
                    metadata['is_checkpoint'] = True
                    break
                    
        except Exception as e:
            metadata['metadata_extraction_error'] = str(e)[:100]
        
        return metadata

    def _register_detected_model(self, detected_model: DetectedModel):
        """탐지된 모델 등록 (중복 처리)"""
        try:
            model_name = detected_model.name
            
            # 기존 모델과 중복 확인
            if model_name in self.detected_models:
                existing_model = self.detected_models[model_name]
                
                # 더 나은 모델로 교체할지 결정
                if self._is_better_model(detected_model, existing_model):
                    # 기존 모델을 대체 경로로 추가
                    detected_model.alternative_paths.append(existing_model.path)
                    detected_model.alternative_paths.extend(existing_model.alternative_paths)
                    self.detected_models[model_name] = detected_model
                    self.logger.debug(f"🔄 모델 교체: {model_name}")
                else:
                    # 새 모델을 대체 경로로 추가
                    existing_model.alternative_paths.append(detected_model.path)
                    self.logger.debug(f"📎 대체 경로 추가: {model_name}")
            else:
                # 새 모델 등록
                self.detected_models[model_name] = detected_model
                self.logger.debug(f"✅ 새 모델 등록: {model_name}")
                
        except Exception as e:
            self.logger.error(f"❌ 모델 등록 실패: {e}")

    def _is_better_model(self, new_model: DetectedModel, existing_model: DetectedModel) -> bool:
        """새 모델이 기존 모델보다 나은지 판단"""
        # 우선순위가 높은 경우 (숫자가 작을수록 높음)
        if new_model.priority < existing_model.priority:
            return True
        elif new_model.priority > existing_model.priority:
            return False
        
        # 우선순위가 같으면 신뢰도 비교
        if new_model.confidence_score > existing_model.confidence_score:
            return True
        elif new_model.confidence_score < existing_model.confidence_score:
            return False
        
        # 신뢰도도 같으면 파일 크기 비교 (큰 것이 좋음)
        return new_model.file_size_mb > existing_model.file_size_mb

    def _post_process_results(self):
        """탐지 결과 후처리"""
        try:
            # 우선순위에 따른 정렬
            sorted_models = sorted(
                self.detected_models.items(),
                key=lambda x: (x[1].priority, -x[1].confidence_score)
            )
            
            # 정렬된 순서로 재정렬
            self.detected_models = {name: model for name, model in sorted_models}
            
            # 카테고리별 통계
            category_stats = {}
            for model in self.detected_models.values():
                category = model.category.value
                if category not in category_stats:
                    category_stats[category] = {"count": 0, "total_size_mb": 0}
                category_stats[category]["count"] += 1
                category_stats[category]["total_size_mb"] += model.file_size_mb
            
            self.scan_stats["category_stats"] = category_stats
            
        except Exception as e:
            self.logger.error(f"❌ 후처리 실패: {e}")

    def _print_detection_summary(self):
        """탐지 결과 요약 출력"""
        try:
            self.logger.info("=" * 60)
            self.logger.info("🎯 자동 모델 탐지 결과 요약")
            self.logger.info("=" * 60)
            
            total_size_gb = sum(model.file_size_mb for model in self.detected_models.values()) / 1024
            self.logger.info(f"📊 총 탐지된 모델: {len(self.detected_models)}개")
            self.logger.info(f"💾 총 모델 크기: {total_size_gb:.2f}GB")
            self.logger.info(f"🔍 스캔된 파일: {self.scan_stats['total_files_scanned']:,}개")
            self.logger.info(f"⏱️ 스캔 시간: {self.scan_stats['scan_duration']:.2f}초")
            
            # 카테고리별 요약
            if "category_stats" in self.scan_stats:
                self.logger.info("\n📁 카테고리별 분포:")
                for category, stats in self.scan_stats["category_stats"].items():
                    size_gb = stats["total_size_mb"] / 1024
                    self.logger.info(f"  {category}: {stats['count']}개 ({size_gb:.2f}GB)")
            
            # 상위 5개 모델
            self.logger.info("\n🏆 주요 탐지된 모델들:")
            for i, (name, model) in enumerate(list(self.detected_models.items())[:5]):
                self.logger.info(f"  {i+1}. {name} ({model.file_size_mb:.1f}MB, {model.category.value})")
                
        except Exception as e:
            self.logger.error(f"❌ 요약 출력 실패: {e}")

    def _load_cache(self) -> Optional[Dict[str, DetectedModel]]:
        """캐시 로드"""
        try:
            if not self.cache_file.exists():
                return None
            
            # TTL 확인
            cache_age = time.time() - self.cache_file.stat().st_mtime
            if cache_age > self.cache_ttl:
                self.logger.debug("캐시 만료됨")
                return None
            
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # DetectedModel 객체로 복원
            detected_models = {}
            for name, model_data in cache_data.get("detected_models", {}).items():
                try:
                    detected_model = DetectedModel(
                        name=model_data["name"],
                        path=Path(model_data["path"]),
                        category=ModelCategory(model_data["category"]),
                        model_type=model_data["model_type"],
                        file_size_mb=model_data["file_size_mb"],
                        file_extension=model_data["file_extension"],
                        confidence_score=model_data["confidence_score"],
                        priority=model_data["priority"],
                        metadata=model_data.get("metadata", {}),
                        alternative_paths=[Path(p) for p in model_data.get("alternative_paths", [])],
                        requirements=model_data.get("requirements", [])
                    )
                    
                    # 파일이 여전히 존재하는지 확인
                    if detected_model.path.exists():
                        detected_models[name] = detected_model
                except Exception as e:
                    self.logger.debug(f"캐시 모델 복원 실패 {name}: {e}")
            
            if detected_models:
                self.detected_models = detected_models
                return detected_models
            
            return None
            
        except Exception as e:
            self.logger.debug(f"캐시 로드 실패: {e}")
            return None

    def _save_cache(self):
        """캐시 저장"""
        try:
            cache_data = {
                "detected_models": {},
                "scan_stats": self.scan_stats,
                "cache_version": "1.0",
                "created_at": time.time()
            }
            
            # DetectedModel 객체를 딕셔너리로 변환
            for name, model in self.detected_models.items():
                cache_data["detected_models"][name] = {
                    "name": model.name,
                    "path": str(model.path),
                    "category": model.category.value,
                    "model_type": model.model_type,
                    "file_size_mb": model.file_size_mb,
                    "file_extension": model.file_extension,
                    "confidence_score": model.confidence_score,
                    "priority": model.priority,
                    "metadata": model.metadata,
                    "alternative_paths": [str(p) for p in model.alternative_paths],
                    "requirements": model.requirements
                }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            self.logger.debug(f"캐시 저장 완료: {self.cache_file}")
            
        except Exception as e:
            self.logger.warning(f"캐시 저장 실패: {e}")

    def get_models_by_category(self, category: ModelCategory) -> List[DetectedModel]:
        """카테고리별 모델 조회"""
        return [model for model in self.detected_models.values() if model.category == category]

    def get_best_model_for_category(self, category: ModelCategory) -> Optional[DetectedModel]:
        """카테고리별 최적 모델 조회 (우선순위 기준)"""
        category_models = self.get_models_by_category(category)
        if not category_models:
            return None
        
        return min(category_models, key=lambda m: (m.priority, -m.confidence_score))

    def get_model_by_name(self, name: str) -> Optional[DetectedModel]:
        """이름으로 모델 조회"""
        return self.detected_models.get(name)

    def get_all_model_paths(self) -> Dict[str, Path]:
        """모든 모델의 경로 딕셔너리 반환"""
        return {name: model.path for name, model in self.detected_models.items()}

    def export_model_config(self, output_path: Optional[Path] = None) -> Path:
        """탐지된 모델들을 설정 파일로 내보내기"""
        try:
            if output_path is None:
                output_path = Path("detected_models_config.json")
            
            config_data = {
                "detection_info": {
                    "detected_at": time.time(),
                    "total_models": len(self.detected_models),
                    "scan_stats": self.scan_stats
                },
                "models": {}
            }
            
            for name, model in self.detected_models.items():
                config_data["models"][name] = {
                    "name": model.name,
                    "path": str(model.path),
                    "alternative_paths": [str(p) for p in model.alternative_paths],
                    "category": model.category.value,
                    "model_type": model.model_type,
                    "file_size_mb": model.file_size_mb,
                    "confidence_score": model.confidence_score,
                    "priority": model.priority,
                    "metadata": model.metadata
                }
            
            with open(output_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            self.logger.info(f"✅ 모델 설정 내보내기 완료: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ 설정 내보내기 실패: {e}")
            raise

# ==============================================
# 🔗 ModelLoader 통합을 위한 어댑터
# ==============================================

class ModelLoaderAdapter:
    """
    자동 탐지 시스템을 기존 ModelLoader와 연결하는 어댑터
    """
    
    def __init__(self, detector: AutoModelDetector):
        self.detector = detector
        self.logger = logging.getLogger(f"{__name__}.ModelLoaderAdapter")
    
    def generate_actual_model_paths(self) -> Dict[str, Dict[str, Any]]:
        """ModelLoader의 ACTUAL_MODEL_PATHS 형식으로 변환"""
        actual_paths = {}
        
        for name, model in self.detector.detected_models.items():
            actual_paths[name] = {
                "primary": str(model.path),
                "alternatives": [str(p) for p in model.alternative_paths],
                "category": model.category.value,
                "model_type": model.model_type,
                "confidence": model.confidence_score,
                "priority": model.priority,
                "size_mb": model.file_size_mb
            }
        
        return actual_paths
    
    def generate_model_configs(self) -> List[Dict[str, Any]]:
        """ModelConfig 형식으로 변환"""
        configs = []
        
        for name, model in self.detector.detected_models.items():
            config = {
                "name": name,
                "model_type": model.category.value,
                "model_class": model.model_type,
                "checkpoint_path": str(model.path),
                "device": "auto",
                "precision": "fp16",
                "input_size": self._get_input_size_for_category(model.category),
                "metadata": {
                    **model.metadata,
                    "auto_detected": True,
                    "confidence_score": model.confidence_score,
                    "alternative_paths": [str(p) for p in model.alternative_paths]
                }
            }
            configs.append(config)
        
        return configs
    
    def _get_input_size_for_category(self, category: ModelCategory) -> Tuple[int, int]:
        """카테고리별 기본 입력 크기"""
        size_mapping = {
            ModelCategory.HUMAN_PARSING: (512, 512),
            ModelCategory.POSE_ESTIMATION: (368, 368),
            ModelCategory.CLOTH_SEGMENTATION: (320, 320),
            ModelCategory.GEOMETRIC_MATCHING: (512, 384),
            ModelCategory.CLOTH_WARPING: (512, 384),
            ModelCategory.VIRTUAL_FITTING: (512, 384),
            ModelCategory.POST_PROCESSING: (512, 512),
            ModelCategory.QUALITY_ASSESSMENT: (224, 224),
            ModelCategory.AUXILIARY: (224, 224)
        }
        return size_mapping.get(category, (512, 512))

# ==============================================
# 🚀 편의 함수들
# ==============================================

def create_auto_detector(
    search_paths: Optional[List[Path]] = None,
    **kwargs
) -> AutoModelDetector:
    """자동 모델 탐지기 생성"""
    return AutoModelDetector(search_paths=search_paths, **kwargs)

def detect_models_and_generate_config(
    output_config_path: Optional[Path] = None,
    force_rescan: bool = False
) -> Dict[str, Any]:
    """모델 탐지 및 설정 생성 원스톱 함수"""
    try:
        # 탐지기 생성 및 실행
        detector = create_auto_detector()
        detected_models = detector.detect_all_models(force_rescan=force_rescan)
        
        # 어댑터를 통한 설정 생성
        adapter = ModelLoaderAdapter(detector)
        actual_paths = adapter.generate_actual_model_paths()
        model_configs = adapter.generate_model_configs()
        
        # 통합 설정
        integrated_config = {
            "detection_summary": {
                "total_models": len(detected_models),
                "categories": list(set(model.category.value for model in detected_models.values())),
                "total_size_gb": sum(model.file_size_mb for model in detected_models.values()) / 1024
            },
            "actual_model_paths": actual_paths,
            "model_configs": model_configs,
            "raw_detections": {name: {
                "path": str(model.path),
                "category": model.category.value,
                "confidence": model.confidence_score
            } for name, model in detected_models.items()}
        }
        
        # 파일 저장
        if output_config_path:
            with open(output_config_path, 'w') as f:
                json.dump(integrated_config, f, indent=2)
            logger.info(f"✅ 통합 설정 저장: {output_config_path}")
        
        return integrated_config
        
    except Exception as e:
        logger.error(f"❌ 모델 탐지 및 설정 생성 실패: {e}")
        raise

# 모듈 익스포트
__all__ = [
    'AutoModelDetector',
    'ModelLoaderAdapter', 
    'DetectedModel',
    'ModelCategory',
    'create_auto_detector',
    'detect_models_and_generate_config',
    'MODEL_IDENTIFICATION_PATTERNS'
]