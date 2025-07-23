# backend/app/ai_pipeline/utils/auto_model_detector.py
"""
🔥 MyCloset AI - 완전 수정된 자동 모델 탐지기 v3.2 (우선순위 문제 해결)
================================================================================
✅ 기존 2번 파일 구조 최대한 유지
✅ Step 구현체의 기존 load_models() 함수와 완벽 연동
✅ 체크포인트 경로만 정확히 찾아서 Step에게 전달
✅ Step이 실제 AI 모델 생성하는 구조 활용
✅ conda 환경 + M3 Max 최적화 유지
✅ 🔥 크기 기반 우선순위 완전 수정 (50MB 이상 우선)
✅ 🔥 대형 모델 우선 탐지 및 정렬
✅ 🔥 작은 더미 파일 자동 제거
✅ 🔥 기존 함수명/메서드명 100% 유지
================================================================================
"""

import os
import re
import logging
import time
import json
import threading
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

# 안전한 PyTorch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# ==============================================
# 🔥 1. 실제 파일 구조 기반 정확한 매핑 테이블 (크기 우선순위 추가)
# ==============================================

class RealFileMapper:
    """실제 파일 구조 기반 정확한 매핑 시스템 (크기 우선순위 강화)"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.RealFileMapper")
        
        # 🔥 실제 파일 구조 기반 완전 정확한 매핑 (크기 필터 강화)
        self.step_file_mappings = {
            # Step 01: Human Parsing
            "human_parsing_graphonomy": {
                "actual_files": ["exp-schp-201908301523-atr.pth"],
                "search_paths": ["step_01_human_parsing", "checkpoints/step_01_human_parsing"],
                "patterns": [r".*exp-schp.*atr.*\.pth$", r".*graphonomy.*\.pth$"],
                "size_range": (200, 300),  # 🔥 더 엄격한 크기 범위
                "min_size_mb": 200,  # 🔥 최소 크기 추가
                "priority": 1,
                "step_class": "HumanParsingImplementation",
                "model_load_method": "load_models",
            },
            "human_parsing_schp_atr": {
                "actual_files": ["exp-schp-201908301523-atr.pth"],
                "search_paths": ["step_01_human_parsing"],
                "patterns": [r".*exp-schp.*atr.*\.pth$"],
                "size_range": (200, 300),
                "min_size_mb": 200,
                "priority": 1,
                "step_class": "HumanParsingImplementation",
                "model_load_method": "load_models",
            },
            
            # Step 02: Pose Estimation  
            "pose_estimation_openpose": {
                "actual_files": ["openpose.pth", "body_pose_model.pth"],
                "search_paths": ["step_02_pose_estimation", "checkpoints/step_02_pose_estimation"],
                "patterns": [r".*openpose.*\.pth$", r".*body.*pose.*\.pth$"],
                "size_range": (150, 250),
                "min_size_mb": 150,
                "priority": 1,
                "step_class": "PoseEstimationImplementation",
                "model_load_method": "load_models",
            },
            
            # Step 03: Cloth Segmentation
            "cloth_segmentation_u2net": {
                "actual_files": ["u2net.pth"],
                "search_paths": ["step_03_cloth_segmentation", "checkpoints/step_03_cloth_segmentation"],
                "patterns": [r".*u2net.*\.pth$"],
                "size_range": (100, 200),
                "min_size_mb": 100,
                "priority": 1,
                "step_class": "ClothSegmentationImplementation",
                "model_load_method": "load_models",
            },
            "cloth_segmentation_sam": {
                "actual_files": ["sam_vit_h_4b8939.pth"],
                "search_paths": ["step_03_cloth_segmentation"],
                "patterns": [r".*sam_vit_h.*\.pth$"],
                "size_range": (2000, 3000),  # 🔥 SAM은 대용량
                "min_size_mb": 2000,
                "priority": 2,
                "step_class": "ClothSegmentationImplementation",
                "model_load_method": "load_models",
            },
            
            # Step 04: Geometric Matching
            "geometric_matching_gmm": {
                "actual_files": ["gmm.pth", "tps_network.pth"],
                "search_paths": ["step_04_geometric_matching"],
                "patterns": [r".*gmm.*\.pth$", r".*tps.*\.pth$"],
                "size_range": (50, 200),  # 🔥 최소 50MB
                "min_size_mb": 50,
                "priority": 1,
                "step_class": "GeometricMatchingImplementation",
                "model_load_method": "load_models",
            },
            
            # Step 05: Cloth Warping
            "cloth_warping_tom": {
                "actual_files": ["cloth_warping_net.pth", "hrviton_final.pth"],
                "search_paths": ["step_05_cloth_warping"],
                "patterns": [r".*cloth.*warping.*\.pth$", r".*hrviton.*\.pth$"],
                "size_range": (100, 800),
                "min_size_mb": 100,
                "priority": 1,
                "step_class": "ClothWarpingImplementation",
                "model_load_method": "load_models",
            },
            
            # Step 06: Virtual Fitting
            "virtual_fitting_diffusion": {
                "actual_files": ["diffusion_pytorch_model.bin", "pytorch_model.bin"],
                "search_paths": ["step_06_virtual_fitting", "checkpoints/ootdiffusion"],
                "patterns": [r".*diffusion.*pytorch.*model.*\.bin$", r".*pytorch_model\.bin$"],
                "size_range": (500, 800),
                "min_size_mb": 500,
                "priority": 1,
                "step_class": "VirtualFittingImplementation",
                "model_load_method": "load_models",
            },
            "virtual_fitting_ootd": {
                "actual_files": ["diffusion_pytorch_model.safetensors"],
                "search_paths": ["checkpoints/ootdiffusion", "step_06_virtual_fitting/ootdiffusion"],
                "patterns": [r".*diffusion.*safetensors$", r".*ootd.*\.pth$"],
                "size_range": (1000, 8000),
                "min_size_mb": 1000,
                "priority": 2,
                "step_class": "VirtualFittingImplementation",
                "model_load_method": "load_models",
            },
            
            # Step 07: Post Processing
            "post_processing_enhance": {
                "actual_files": ["enhance_model.pth", "ESRGAN_x4.pth"],
                "search_paths": ["step_07_post_processing"],
                "patterns": [r".*enhance.*\.pth$", r".*ESRGAN.*\.pth$"],
                "size_range": (50, 200),
                "min_size_mb": 50,
                "priority": 1,
                "step_class": "PostProcessingImplementation",
                "model_load_method": "load_models",
            },
            
            # Step 08: Quality Assessment
            "quality_assessment_clip": {
                "actual_files": ["pytorch_model.bin"],
                "search_paths": ["step_08_quality_assessment"],
                "patterns": [r".*pytorch_model\.bin$"],
                "size_range": (500, 800),
                "min_size_mb": 500,
                "priority": 1,
                "step_class": "QualityAssessmentImplementation",
                "model_load_method": "load_models",
            }
        }
        
        # 🔥 크기 우선순위 설정
        self.size_priority_threshold = 50  # 50MB 미만은 제외
    
    def find_actual_file(self, request_name: str, ai_models_root: Path) -> Optional[Path]:
        """요청명에 대한 실제 파일 찾기 (크기 우선순위 적용)"""
        if request_name not in self.step_file_mappings:
            self.logger.debug(f"❓ 알 수 없는 요청명: {request_name}")
            return None
        
        mapping = self.step_file_mappings[request_name]
        found_candidates = []
        
        # 1. 정확한 파일명으로 검색
        for filename in mapping["actual_files"]:
            for search_path in mapping["search_paths"]:
                full_path = ai_models_root / search_path / filename
                if full_path.exists() and full_path.is_file():
                    file_size_mb = full_path.stat().st_size / (1024 * 1024)
                    
                    # 🔥 크기 필터 적용
                    if file_size_mb < mapping.get("min_size_mb", self.size_priority_threshold):
                        self.logger.debug(f"🗑️ 크기 부족: {full_path} ({file_size_mb:.1f}MB < {mapping.get('min_size_mb', self.size_priority_threshold)}MB)")
                        continue
                    
                    min_size, max_size = mapping["size_range"]
                    if min_size <= file_size_mb <= max_size:
                        found_candidates.append((full_path, file_size_mb, "exact_match"))
                        self.logger.debug(f"✅ 정확한 매칭: {request_name} → {full_path} ({file_size_mb:.1f}MB)")
                    else:
                        self.logger.debug(f"⚠️ 크기 불일치: {full_path} ({file_size_mb:.1f}MB)")
        
        # 2. 패턴으로 검색
        for search_path in mapping["search_paths"]:
            search_dir = ai_models_root / search_path
            if not search_dir.exists():
                continue
                
            for pattern in mapping["patterns"]:
                try:
                    for file_path in search_dir.rglob("*"):
                        if file_path.is_file() and re.search(pattern, file_path.name, re.IGNORECASE):
                            file_size_mb = file_path.stat().st_size / (1024 * 1024)
                            
                            # 🔥 크기 필터 적용
                            if file_size_mb < mapping.get("min_size_mb", self.size_priority_threshold):
                                continue
                            
                            min_size, max_size = mapping["size_range"]
                            if min_size <= file_size_mb <= max_size:
                                found_candidates.append((file_path, file_size_mb, "pattern_match"))
                                self.logger.debug(f"✅ 패턴 매칭: {request_name} → {file_path} ({file_size_mb:.1f}MB)")
                except Exception as e:
                    self.logger.debug(f"패턴 검색 오류: {e}")
                    continue
        
        # 🔥 크기순 정렬 (큰 것부터)
        if found_candidates:
            found_candidates.sort(key=lambda x: x[1], reverse=True)
            best_match = found_candidates[0]
            self.logger.info(f"🏆 최적 매칭: {request_name} → {best_match[0]} ({best_match[1]:.1f}MB, {best_match[2]})")
            return best_match[0]
        
        self.logger.warning(f"❌ {request_name} 파일을 찾을 수 없습니다 (크기 기준: {mapping.get('min_size_mb', self.size_priority_threshold)}MB 이상)")
        return None
    
    def get_step_info(self, request_name: str) -> Optional[Dict[str, Any]]:
        """🔥 Step 구현체 정보 반환 (기존 함수 유지)"""
        if request_name in self.step_file_mappings:
            mapping = self.step_file_mappings[request_name]
            return {
                "step_class": mapping.get("step_class"),
                "model_load_method": mapping.get("model_load_method"),
                "priority": mapping.get("priority"),
                "patterns": mapping.get("patterns", []),
                "min_size_mb": mapping.get("min_size_mb", self.size_priority_threshold)
            }
        return None

# ==============================================
# 🔥 2. DetectedModel 클래스 (크기 우선순위 추가)
# ==============================================

@dataclass
class DetectedModel:
    """탐지된 모델 정보 + Step 연동 정보 + 크기 우선순위"""
    name: str
    path: Path
    step_name: str
    model_type: str
    file_size_mb: float
    confidence_score: float
    
    # 🔥 Step 구현체 연동 정보
    step_class_name: Optional[str] = None
    model_load_method: Optional[str] = None
    step_can_load: bool = False
    
    # 🔥 크기 우선순위 정보
    priority_score: float = 0.0
    is_large_model: bool = False
    meets_size_requirement: bool = False
    
    # 추가 정보
    checkpoint_path: Optional[str] = None
    device_compatible: bool = True
    recommended_device: str = "cpu"
    
    def __post_init__(self):
        """🔥 우선순위 점수 자동 계산"""
        self.priority_score = self._calculate_priority_score()
        self.is_large_model = self.file_size_mb > 1000  # 1GB 이상
        self.meets_size_requirement = self.file_size_mb >= 50  # 50MB 이상
    
    def _calculate_priority_score(self) -> float:
        """🔥 우선순위 점수 계산"""
        score = 0.0
        
        # 크기 기반 점수 (로그 스케일)
        if self.file_size_mb > 0:
            import math
            score += math.log10(max(self.file_size_mb, 1)) * 100
        
        # 신뢰도 보너스
        score += self.confidence_score * 50
        
        # 대형 모델 보너스
        if self.file_size_mb > 2000:  # 2GB 이상
            score += 200
        elif self.file_size_mb > 1000:  # 1GB 이상
            score += 100
        elif self.file_size_mb > 500:  # 500MB 이상
            score += 50
        elif self.file_size_mb > 200:  # 200MB 이상
            score += 20
        elif self.file_size_mb >= 50:  # 50MB 이상
            score += 10
        else:
            score -= 100  # 50MB 미만은 감점
        
        # Step 로드 가능 보너스
        if self.step_can_load:
            score += 30
        
        return score
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환 (크기 정보 추가)"""
        return {
            "name": self.name,
            "path": str(self.path),
            "checkpoint_path": self.checkpoint_path or str(self.path),
            "step_class": self.step_name,
            "model_type": self.model_type,
            "size_mb": self.file_size_mb,
            "confidence": self.confidence_score,
            "device_config": {
                "recommended_device": self.recommended_device,
                "device_compatible": self.device_compatible
            },
            
            # 🔥 Step 연동 정보
            "step_implementation": {
                "step_class_name": self.step_class_name,
                "model_load_method": self.model_load_method,
                "step_can_load": self.step_can_load,
                "load_ready": self.step_can_load and self.checkpoint_path is not None
            },
            
            # 🔥 크기 우선순위 정보
            "priority_info": {
                "priority_score": self.priority_score,
                "is_large_model": self.is_large_model,
                "meets_size_requirement": self.meets_size_requirement,
                "size_category": self._get_size_category()
            },
            
            "metadata": {
                "detection_time": time.time(),
                "file_extension": self.path.suffix
            }
        }
    
    def _get_size_category(self) -> str:
        """크기 카테고리 분류"""
        if self.file_size_mb >= 2000:
            return "ultra_large"  # 2GB+
        elif self.file_size_mb >= 1000:
            return "large"  # 1GB+
        elif self.file_size_mb >= 500:
            return "medium_large"  # 500MB+
        elif self.file_size_mb >= 200:
            return "medium"  # 200MB+
        elif self.file_size_mb >= 50:
            return "small_valid"  # 50MB+
        else:
            return "too_small"  # 50MB 미만
    
    def can_be_loaded_by_step(self) -> bool:
        """Step 구현체로 로드 가능한지 확인 (크기 요구사항 포함)"""
        return (self.step_can_load and 
                self.step_class_name is not None and 
                self.model_load_method is not None and
                self.checkpoint_path is not None and
                self.meets_size_requirement)

# ==============================================
# 🔥 3. 수정된 모델 탐지기 (크기 우선순위 완전 적용)
# ==============================================

class FixedModelDetector:
    """수정된 모델 탐지기 (크기 우선순위 완전 적용)"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.FixedModelDetector")
        self.file_mapper = RealFileMapper()
        self.ai_models_root = self._find_ai_models_root()
        self.detected_models: Dict[str, DetectedModel] = {}
        
        # 🔥 크기 기반 필터링 설정
        self.min_model_size_mb = 50  # 50MB 미만은 제외
        self.prioritize_large_models = True
        
        # 시스템 정보
        self.is_m3_max = self._detect_m3_max()
        self.conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
        
        # 통계 정보
        self.detection_stats = {
            "total_files_scanned": 0,
            "models_found": 0,
            "large_models_found": 0,
            "small_models_filtered": 0,
            "step_loadable_models": 0,
            "scan_duration": 0.0
        }
        
        self.logger.info(f"🔧 크기 우선순위 모델 탐지기 초기화")
        self.logger.info(f"   AI 모델 루트: {self.ai_models_root}")
        self.logger.info(f"   최소 크기: {self.min_model_size_mb}MB")
        self.logger.info(f"   M3 Max: {self.is_m3_max}, conda: {bool(self.conda_env)}")
    
    def _find_ai_models_root(self) -> Path:
        """AI 모델 루트 디렉토리 찾기 (기존 함수 유지)"""
        current = Path(__file__).resolve()
        
        # 프로젝트 루트 찾기
        for _ in range(10):
            if current.name == 'mycloset-ai' or (current / 'backend').exists():
                return current / 'backend' / 'ai_models'
            if current.name == 'backend':
                return current / 'ai_models'
            if current.parent == current:
                break
            current = current.parent
        
        # 폴백
        fallback_path = Path(__file__).resolve().parent.parent.parent.parent / 'ai_models'
        self.logger.warning(f"⚠️ 폴백 경로 사용: {fallback_path}")
        return fallback_path
    
    def _detect_m3_max(self) -> bool:
        """M3 Max 감지 (기존 함수 유지)"""
        try:
            import platform
            return 'arm64' in platform.machine().lower()
        except:
            return False
    
    def detect_all_models(self) -> Dict[str, DetectedModel]:
        """🔥 모든 모델 탐지 (크기 우선순위 완전 적용)"""
        start_time = time.time()
        self.detected_models.clear()
        self.detection_stats = {
            "total_files_scanned": 0,
            "models_found": 0,
            "large_models_found": 0,
            "small_models_filtered": 0,
            "step_loadable_models": 0,
            "scan_duration": 0.0
        }
        
        if not self.ai_models_root.exists():
            self.logger.error(f"❌ AI 모델 루트가 존재하지 않습니다: {self.ai_models_root}")
            return {}
        
        self.logger.info("🔍 크기 우선순위 기반 모델 탐지 시작...")
        
        # 요청명별로 실제 파일 찾기 + Step 정보 추가 (크기 우선순위 적용)
        for request_name in self.file_mapper.step_file_mappings.keys():
            try:
                # 1. 실제 파일 찾기 (크기 필터 적용)
                actual_file = self.file_mapper.find_actual_file(request_name, self.ai_models_root)
                
                if actual_file:
                    # 2. Step 정보 가져오기
                    step_info = self.file_mapper.get_step_info(request_name)
                    
                    # 3. DetectedModel 생성 (크기 우선순위 포함)
                    model = self._create_detected_model_with_step_info(request_name, actual_file, step_info)
                    if model and model.meets_size_requirement:
                        self.detected_models[model.name] = model
                        self.detection_stats["models_found"] += 1
                        
                        if model.is_large_model:
                            self.detection_stats["large_models_found"] += 1
                        
                        if model.can_be_loaded_by_step():
                            self.detection_stats["step_loadable_models"] += 1
                    elif model:
                        self.detection_stats["small_models_filtered"] += 1
                        self.logger.debug(f"🗑️ 크기 부족으로 제외: {request_name} ({model.file_size_mb:.1f}MB)")
                        
            except Exception as e:
                self.logger.error(f"❌ {request_name} 탐지 실패: {e}")
                continue
        
        # 🔥 추가 파일들 자동 스캔 (크기 우선순위 적용)
        self._scan_additional_files()
        
        # 🔥 크기 우선순위로 정렬
        if self.prioritize_large_models:
            self._sort_models_by_priority()
        
        self.detection_stats["scan_duration"] = time.time() - start_time
        
        self.logger.info(f"🎉 크기 우선순위 모델 탐지 완료: {self.detection_stats['models_found']}개")
        self.logger.info(f"📊 대형 모델: {self.detection_stats['large_models_found']}개")
        self.logger.info(f"🗑️ 작은 모델 제외: {self.detection_stats['small_models_filtered']}개")
        self.logger.info(f"✅ Step 로드 가능: {self.detection_stats['step_loadable_models']}개")
        self.logger.info(f"⏱️ 소요 시간: {self.detection_stats['scan_duration']:.2f}초")
        
        return self.detected_models
    
    def _create_detected_model_with_step_info(self, request_name: str, file_path: Path, step_info: Optional[Dict]) -> Optional[DetectedModel]:
        """DetectedModel 생성 (크기 우선순위 포함)"""
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            # Step 이름 추출
            step_name = self._extract_step_name(request_name)
            
            # 디바이스 설정
            recommended_device = "mps" if self.is_m3_max else "cpu"
            
            # 🔥 Step 연동 정보 설정
            step_class_name = None
            model_load_method = None
            step_can_load = False
            
            if step_info:
                step_class_name = step_info.get("step_class")
                model_load_method = step_info.get("model_load_method", "load_models")
                step_can_load = bool(step_class_name and model_load_method)
            
            # 🔥 신뢰도 계산 (크기 기반)
            confidence_score = self._calculate_size_based_confidence(file_size_mb, step_info)
            
            model = DetectedModel(
                name=request_name,
                path=file_path,
                step_name=step_name,
                model_type=step_name.replace("Step", "").lower(),
                file_size_mb=file_size_mb,
                confidence_score=confidence_score,
                
                # 🔥 Step 연동 정보
                step_class_name=step_class_name,
                model_load_method=model_load_method,
                step_can_load=step_can_load,
                
                checkpoint_path=str(file_path),
                device_compatible=True,
                recommended_device=recommended_device
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"❌ {request_name} 모델 생성 실패: {e}")
            return None
    
    def _calculate_size_based_confidence(self, file_size_mb: float, step_info: Optional[Dict]) -> float:
        """🔥 크기 기반 신뢰도 계산"""
        confidence = 0.5  # 기본값
        
        # 크기 기반 신뢰도
        if file_size_mb >= 2000:  # 2GB+
            confidence = 1.0
        elif file_size_mb >= 1000:  # 1GB+
            confidence = 0.95
        elif file_size_mb >= 500:  # 500MB+
            confidence = 0.9
        elif file_size_mb >= 200:  # 200MB+
            confidence = 0.8
        elif file_size_mb >= 100:  # 100MB+
            confidence = 0.7
        elif file_size_mb >= 50:  # 50MB+
            confidence = 0.6
        else:  # 50MB 미만
            confidence = 0.1
        
        # Step 정보 보너스
        if step_info:
            min_expected_size = step_info.get("min_size_mb", 50)
            if file_size_mb >= min_expected_size:
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _extract_step_name(self, request_name: str) -> str:
        """요청명에서 Step 이름 추출 (기존 함수 유지)"""
        step_mappings = {
            "human_parsing": "HumanParsingStep",
            "pose_estimation": "PoseEstimationStep", 
            "cloth_segmentation": "ClothSegmentationStep",
            "geometric_matching": "GeometricMatchingStep",
            "cloth_warping": "ClothWarpingStep",
            "virtual_fitting": "VirtualFittingStep",
            "post_processing": "PostProcessingStep",
            "quality_assessment": "QualityAssessmentStep"
        }
        
        for key, step_name in step_mappings.items():
            if key in request_name:
                return step_name
        
        return "UnknownStep"
    
    def _scan_additional_files(self):
        """🔥 추가 파일들 자동 스캔 (크기 우선순위 적용)"""
        try:
            # Ultra 모델들 스캔
            ultra_dir = self.ai_models_root / "ultra_models"
            if ultra_dir.exists():
                self._scan_ultra_models(ultra_dir)
            
            # 체크포인트 디렉토리 스캐 
            checkpoints_dir = self.ai_models_root / "checkpoints"
            if checkpoints_dir.exists():
                self._scan_checkpoints(checkpoints_dir)
                
        except Exception as e:
            self.logger.debug(f"추가 스캔 오류: {e}")
    
    def _scan_ultra_models(self, ultra_dir: Path):
        """🔥 Ultra 모델 스캔 (크기 우선순위 적용)"""
        model_extensions = {'.pth', '.bin', '.safetensors', '.ckpt'}
        
        candidates = []
        
        for file_path in ultra_dir.rglob('*'):
            if (file_path.is_file() and 
                file_path.suffix.lower() in model_extensions):
                
                try:
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    
                    # 🔥 크기 필터 적용
                    if file_size_mb < self.min_model_size_mb:
                        self.detection_stats["small_models_filtered"] += 1
                        continue
                    
                    candidates.append((file_path, file_size_mb))
                    
                except Exception as e:
                    self.logger.debug(f"Ultra 모델 처리 오류 {file_path}: {e}")
                    continue
        
        # 🔥 크기순 정렬 (큰 것부터)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        for file_path, file_size_mb in candidates:
            model_name = f"ultra_{file_path.parent.name}_{file_path.stem}"
            
            # 중복 방지
            if model_name in self.detected_models:
                continue
            
            model = DetectedModel(
                name=model_name,
                path=file_path,
                step_name="UltraModel",
                model_type="ultra",
                file_size_mb=file_size_mb,
                confidence_score=self._calculate_size_based_confidence(file_size_mb, None),
                checkpoint_path=str(file_path),
                device_compatible=True,
                recommended_device="mps" if self.is_m3_max else "cpu"
            )
            
            if model.meets_size_requirement:
                self.detected_models[model_name] = model
                self.detection_stats["models_found"] += 1
                
                if model.is_large_model:
                    self.detection_stats["large_models_found"] += 1
                
                self.logger.debug(f"✅ Ultra 모델: {model_name} ({file_size_mb:.1f}MB)")
    
    def _scan_checkpoints(self, checkpoints_dir: Path):
        """🔥 체크포인트 디렉토리 스캔 (크기 우선순위 적용)"""
        candidates = []
        
        for subdir in checkpoints_dir.iterdir():
            if subdir.is_dir():
                for file_path in subdir.rglob('*.pth'):
                    # 중복 방지
                    if file_path.name not in [m.path.name for m in self.detected_models.values()]:
                        try:
                            file_size_mb = file_path.stat().st_size / (1024 * 1024)
                            
                            # 🔥 크기 필터 적용
                            if file_size_mb < self.min_model_size_mb:
                                self.detection_stats["small_models_filtered"] += 1
                                continue
                            
                            candidates.append((file_path, file_size_mb, subdir.name))
                            
                        except Exception as e:
                            self.logger.debug(f"체크포인트 처리 오류 {file_path}: {e}")
                            continue
        
        # 🔥 크기순 정렬 (큰 것부터)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        for file_path, file_size_mb, subdir_name in candidates:
            model_name = f"checkpoint_{subdir_name}_{file_path.stem}"
            
            model = DetectedModel(
                name=model_name,
                path=file_path,
                step_name="CheckpointModel",
                model_type="checkpoint",
                file_size_mb=file_size_mb,
                confidence_score=self._calculate_size_based_confidence(file_size_mb, None),
                checkpoint_path=str(file_path),
                device_compatible=True,
                recommended_device="mps" if self.is_m3_max else "cpu"
            )
            
            if model.meets_size_requirement:
                self.detected_models[model_name] = model
                self.detection_stats["models_found"] += 1
                
                if model.is_large_model:
                    self.detection_stats["large_models_found"] += 1
                
                self.logger.debug(f"✅ 체크포인트: {model_name} ({file_size_mb:.1f}MB)")
    
    def _sort_models_by_priority(self):
        """🔥 모델들을 우선순위로 정렬"""
        try:
            # 우선순위 점수로 정렬
            sorted_items = sorted(
                self.detected_models.items(),
                key=lambda x: x[1].priority_score,
                reverse=True
            )
            
            # 정렬된 순서로 재배치
            self.detected_models = dict(sorted_items)
            
            self.logger.info("🎯 모델 우선순위 정렬 완료")
            
            # 상위 5개 모델 로깅
            for i, (name, model) in enumerate(list(self.detected_models.items())[:5]):
                self.logger.info(f"  {i+1}. {name}: {model.file_size_mb:.1f}MB (점수: {model.priority_score:.1f})")
                
        except Exception as e:
            self.logger.error(f"❌ 모델 정렬 실패: {e}")

# ==============================================
# 🔥 4. ModelLoader 호환 인터페이스 (크기 우선순위 적용)
# ==============================================

def get_step_loadable_models() -> List[Dict[str, Any]]:
    """🔥 Step 구현체로 로드 가능한 모델들만 반환 (크기 우선순위 적용)"""
    detector = get_global_detector()
    models = detector.detect_all_models()
    
    loadable_models = []
    for model in models.values():
        if model.can_be_loaded_by_step():
            model_dict = model.to_dict()
            model_dict["load_instruction"] = {
                "step_class": model.step_class_name,
                "method": model.model_load_method,
                "checkpoint_path": model.checkpoint_path
            }
            loadable_models.append(model_dict)
    
    # 🔥 우선순위 점수로 정렬
    return sorted(loadable_models, key=lambda x: x["priority_info"]["priority_score"], reverse=True)

def create_step_model_loader_config() -> Dict[str, Any]:
    """🔥 Step 구현체 연동용 ModelLoader 설정 생성 (크기 우선순위 적용)"""
    detector = get_global_detector()
    detected_models = detector.detect_all_models()
    
    config = {
        "version": "step_integrated_detector_v3.2_priority_fixed",
        "generated_at": time.time(),
        "device": "mps" if detector.is_m3_max else "cpu",
        "is_m3_max": detector.is_m3_max,
        "conda_env": detector.conda_env,
        "min_model_size_mb": detector.min_model_size_mb,
        "prioritize_large_models": detector.prioritize_large_models,
        "models": {},
        "step_mappings": {},
        "step_loadable_count": 0,
        "detection_stats": detector.detection_stats
    }
    
    # 모델별 설정 (우선순위 순)
    for model_name, model in detected_models.items():
        model_dict = model.to_dict()
        config["models"][model_name] = model_dict
        
        # Step 로드 가능 카운트
        if model.can_be_loaded_by_step():
            config["step_loadable_count"] += 1
        
        # Step 매핑
        step_name = model.step_name
        if step_name not in config["step_mappings"]:
            config["step_mappings"][step_name] = []
        config["step_mappings"][step_name].append(model_name)
    
    # 통계 (크기 기반)
    config["summary"] = {
        "total_models": len(detected_models),
        "large_models": sum(1 for m in detected_models.values() if m.is_large_model),
        "step_loadable_models": config["step_loadable_count"],
        "total_size_gb": sum(m.file_size_mb for m in detected_models.values()) / 1024,
        "average_size_mb": sum(m.file_size_mb for m in detected_models.values()) / len(detected_models) if detected_models else 0,
        "device_optimized": detector.is_m3_max,
        "step_integration_ready": config["step_loadable_count"] > 0,
        "min_size_threshold_mb": detector.min_model_size_mb,
        "priority_sorting_enabled": detector.prioritize_large_models
    }
    
    logger.info(f"✅ 크기 우선순위 설정 생성: {len(detected_models)}개 모델, {config['step_loadable_count']}개 Step 로드 가능")
    logger.info(f"📊 대형 모델: {config['summary']['large_models']}개, 평균 크기: {config['summary']['average_size_mb']:.1f}MB")
    return config

# ==============================================
# 🔥 5. 전역 인스턴스 및 인터페이스 (기존 함수명 유지 + 크기 우선순위)
# ==============================================

_global_detector: Optional[FixedModelDetector] = None
_detector_lock = threading.Lock()

def get_global_detector() -> FixedModelDetector:
    """전역 탐지기 인스턴스 (기존 함수명 유지)"""
    global _global_detector
    if _global_detector is None:
        with _detector_lock:
            if _global_detector is None:
                _global_detector = FixedModelDetector()
    return _global_detector

def quick_model_detection() -> Dict[str, DetectedModel]:
    """빠른 모델 탐지 (기존 함수명 유지, 크기 우선순위 적용)"""
    detector = get_global_detector()
    return detector.detect_all_models()

def list_available_models(step_class: Optional[str] = None) -> List[Dict[str, Any]]:
    """🔥 사용 가능한 모델 목록 (크기 우선순위 정렬, 기존 함수명 유지)"""
    detector = get_global_detector()
    models = detector.detect_all_models()
    
    result = []
    for model in models.values():
        model_dict = model.to_dict()
        
        if step_class and model_dict["step_class"] != step_class:
            continue
        
        result.append(model_dict)
    
    # 🔥 우선순위 점수로 정렬 (큰 것부터)
    return sorted(result, key=lambda x: x["priority_info"]["priority_score"], reverse=True)

def get_models_for_step(step_name: str) -> List[Dict[str, Any]]:
    """🔥 Step별 모델 조회 (크기 우선순위 적용, 기존 함수명 유지)"""
    models = list_available_models(step_class=step_name)
    return models

def validate_model_exists(model_name: str) -> bool:
    """모델 존재 확인 (기존 함수명 유지)"""
    detector = get_global_detector()
    return model_name in detector.detected_models

def generate_advanced_model_loader_config() -> Dict[str, Any]:
    """🔥 고급 ModelLoader 설정 생성 (크기 우선순위 포함, 기존 함수명 유지)"""
    return create_step_model_loader_config()

def create_step_interface(step_name: str, config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """🔥 Step 인터페이스 생성 (크기 우선순위 적용, 기존 함수명 유지)"""
    models = get_models_for_step(step_name)
    if not models:
        return None
    
    # Step 로드 가능한 모델 우선 선택 (크기 우선순위 적용)
    loadable_models = [m for m in models if m.get("step_implementation", {}).get("load_ready", False)]
    primary_model = loadable_models[0] if loadable_models else models[0]
    
    return {
        "step_name": step_name,
        "primary_model": primary_model,
        "config": config or {},
        "load_ready": len(loadable_models) > 0,
        "step_integration": primary_model.get("step_implementation", {}),
        "priority_info": primary_model.get("priority_info", {}),
        "created_at": time.time()
    }

def get_large_models_only() -> List[Dict[str, Any]]:
    """🔥 대형 모델만 반환 (1GB 이상)"""
    detector = get_global_detector()
    models = detector.detect_all_models()
    
    large_models = []
    for model in models.values():
        if model.is_large_model:
            large_models.append(model.to_dict())
    
    return sorted(large_models, key=lambda x: x["size_mb"], reverse=True)

def get_detection_statistics() -> Dict[str, Any]:
    """🔥 탐지 통계 반환"""
    detector = get_global_detector()
    detector.detect_all_models()  # 최신 통계 확보
    
    return {
        "detection_stats": detector.detection_stats,
        "system_info": {
            "ai_models_root": str(detector.ai_models_root),
            "min_model_size_mb": detector.min_model_size_mb,
            "prioritize_large_models": detector.prioritize_large_models,
            "is_m3_max": detector.is_m3_max,
            "conda_env": detector.conda_env
        },
        "model_summary": {
            "total_detected": len(detector.detected_models),
            "large_models": sum(1 for m in detector.detected_models.values() if m.is_large_model),
            "step_loadable": sum(1 for m in detector.detected_models.values() if m.can_be_loaded_by_step()),
            "average_size_mb": sum(m.file_size_mb for m in detector.detected_models.values()) / len(detector.detected_models) if detector.detected_models else 0
        }
    }

# 기존 호환성 별칭 (함수명 유지)
RealWorldModelDetector = FixedModelDetector
create_real_world_detector = lambda **kwargs: FixedModelDetector()
comprehensive_model_detection = quick_model_detection

# ==============================================
# 🔥 6. 익스포트 (기존 함수명 유지)
# ==============================================

__all__ = [
    'FixedModelDetector',
    'DetectedModel', 
    'RealFileMapper',
    'get_global_detector',
    'quick_model_detection',
    'list_available_models',
    'get_models_for_step',
    'get_step_loadable_models',
    'create_step_model_loader_config',
    'generate_advanced_model_loader_config',
    'validate_model_exists',
    'create_step_interface',
    'get_large_models_only',  # 🔥 새로 추가
    'get_detection_statistics',  # 🔥 새로 추가
    
    # 호환성 (기존 함수명 유지)
    'RealWorldModelDetector',
    'create_real_world_detector',
    'comprehensive_model_detection'
]

# ==============================================
# 🔥 7. 초기화 (크기 우선순위 정보 추가)
# ==============================================

logger.info("✅ 완전 수정된 자동 모델 탐지기 v3.2 로드 완료")
logger.info("🎯 체크포인트 경로 → Step 구현체 완벽 연동")
logger.info("🔧 기존 load_models() 함수 활용")
logger.info("✅ Step이 실제 AI 모델 생성하는 구조 지원")
logger.info("🔥 ✅ 크기 기반 우선순위 완전 적용 (50MB 이상)")
logger.info("🔥 ✅ 대형 모델 우선 탐지 및 정렬")
logger.info("🔥 ✅ 작은 더미 파일 자동 제거")
logger.info("✅ 기존 함수명/메서드명 100% 유지")

# 초기화 테스트
try:
    _test_detector = get_global_detector()
    logger.info(f"🚀 크기 우선순위 탐지기 준비 완료!")
    logger.info(f"   AI 모델 루트: {_test_detector.ai_models_root}")
    logger.info(f"   최소 크기: {_test_detector.min_model_size_mb}MB")
    logger.info(f"   M3 Max: {_test_detector.is_m3_max}")
    logger.info(f"   대형 모델 우선: {_test_detector.prioritize_large_models}")
except Exception as e:
    logger.error(f"❌ 초기화 실패: {e}")

if __name__ == "__main__":
    print("🔍 완전 수정된 자동 모델 탐지기 v3.2 테스트")
    print("=" * 60)
    
    # 테스트 실행
    models = quick_model_detection()
    print(f"✅ 탐지된 모델: {len(models)}개")
    
    # 크기별 분류
    large_models = [m for m in models.values() if m.is_large_model]
    valid_models = [m for m in models.values() if m.meets_size_requirement]
    step_loadable = [m for m in models.values() if m.can_be_loaded_by_step()]
    
    print(f"📊 대형 모델 (1GB+): {len(large_models)}개")
    print(f"✅ 유효 모델 (50MB+): {len(valid_models)}개")
    print(f"🔗 Step 로드 가능: {len(step_loadable)}개")
    
    if step_loadable:
        print("\n🏆 상위 Step 로드 가능 모델:")
        for i, model in enumerate(step_loadable[:5]):
            step_info = model.step_implementation if hasattr(model, 'step_implementation') else {}
            print(f"   {i+1}. {model.name}: {model.file_size_mb:.1f}MB (점수: {model.priority_score:.1f})")
    
    # 통계 출력
    stats = get_detection_statistics()
    print(f"\n📈 탐지 통계:")
    print(f"   스캔 시간: {stats['detection_stats']['scan_duration']:.2f}초")
    print(f"   제외된 작은 파일: {stats['detection_stats']['small_models_filtered']}개")
    
    print("🎉 크기 우선순위 테스트 완료!")