# backend/app/ai_pipeline/utils/auto_model_detector.py
"""
🔥 MyCloset AI - 최소 수정 자동 모델 탐지기 v3.1 (Step 구현체 연동)
================================================================================
✅ 기존 2번 파일 구조 최대한 유지
✅ Step 구현체의 기존 load_models() 함수와 완벽 연동
✅ 체크포인트 경로만 정확히 찾아서 Step에게 전달
✅ Step이 실제 AI 모델 생성하는 구조 활용
✅ conda 환경 + M3 Max 최적화 유지
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
# 🔥 1. 실제 파일 구조 기반 정확한 매핑 테이블 (개선)
# ==============================================

class RealFileMapper:
    """실제 파일 구조 기반 정확한 매핑 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.RealFileMapper")
        
        # 🔥 실제 파일 구조 기반 완전 정확한 매핑
        self.step_file_mappings = {
            # Step 01: Human Parsing
            "human_parsing_graphonomy": {
                "actual_files": ["exp-schp-201908301523-atr.pth"],
                "search_paths": ["step_01_human_parsing", "checkpoints/step_01_human_parsing"],
                "patterns": [r".*exp-schp.*atr.*\.pth$", r".*graphonomy.*\.pth$"],
                "size_range": (250, 260),
                "priority": 1,
                "step_class": "HumanParsingImplementation",  # 🔥 Step 구현체 클래스명
                "model_load_method": "load_models",  # 🔥 Step의 모델 로드 함수명
            },
            "human_parsing_schp_atr": {
                "actual_files": ["exp-schp-201908301523-atr.pth"],
                "search_paths": ["step_01_human_parsing"],
                "patterns": [r".*exp-schp.*atr.*\.pth$"],
                "size_range": (250, 260),
                "priority": 1,
                "step_class": "HumanParsingImplementation",
                "model_load_method": "load_models",
            },
            
            # Step 02: Pose Estimation  
            "pose_estimation_openpose": {
                "actual_files": ["openpose.pth", "body_pose_model.pth"],
                "search_paths": ["step_02_pose_estimation", "checkpoints/step_02_pose_estimation"],
                "patterns": [r".*openpose.*\.pth$", r".*body.*pose.*\.pth$"],
                "size_range": (190, 210),
                "priority": 1,
                "step_class": "PoseEstimationImplementation",
                "model_load_method": "load_models",
            },
            
            # Step 03: Cloth Segmentation
            "cloth_segmentation_u2net": {
                "actual_files": ["u2net.pth"],
                "search_paths": ["step_03_cloth_segmentation", "checkpoints/step_03_cloth_segmentation"],
                "patterns": [r".*u2net.*\.pth$"],
                "size_range": (160, 180),
                "priority": 1,
                "step_class": "ClothSegmentationImplementation",
                "model_load_method": "load_models",
            },
            "cloth_segmentation_sam": {
                "actual_files": ["sam_vit_h_4b8939.pth"],
                "search_paths": ["step_03_cloth_segmentation"],
                "patterns": [r".*sam_vit_h.*\.pth$"],
                "size_range": (2400, 2500),
                "priority": 2,
                "step_class": "ClothSegmentationImplementation",
                "model_load_method": "load_models",
            },
            
            # Step 04: Geometric Matching
            "geometric_matching_gmm": {
                "actual_files": ["gmm.pth", "tps_network.pth"],
                "search_paths": ["step_04_geometric_matching"],
                "patterns": [r".*gmm.*\.pth$", r".*tps.*\.pth$"],
                "size_range": (1, 50),
                "priority": 1,
                "step_class": "GeometricMatchingImplementation",
                "model_load_method": "load_models",
            },
            
            # Step 05: Cloth Warping
            "cloth_warping_tom": {
                "actual_files": ["cloth_warping_net.pth", "hrviton_final.pth"],
                "search_paths": ["step_05_cloth_warping"],
                "patterns": [r".*cloth.*warping.*\.pth$", r".*hrviton.*\.pth$"],
                "size_range": (50, 500),
                "priority": 1,
                "step_class": "ClothWarpingImplementation",
                "model_load_method": "load_models",
            },
            
            # Step 06: Virtual Fitting
            "virtual_fitting_diffusion": {
                "actual_files": ["diffusion_pytorch_model.bin", "pytorch_model.bin"],
                "search_paths": ["step_06_virtual_fitting", "checkpoints/ootdiffusion"],
                "patterns": [r".*diffusion.*pytorch.*model.*\.bin$", r".*pytorch_model\.bin$"],
                "size_range": (300, 600),
                "priority": 1,
                "step_class": "VirtualFittingImplementation",
                "model_load_method": "load_models",
            },
            "virtual_fitting_ootd": {
                "actual_files": ["diffusion_pytorch_model.safetensors"],
                "search_paths": ["checkpoints/ootdiffusion", "step_06_virtual_fitting/ootdiffusion"],
                "patterns": [r".*diffusion.*safetensors$", r".*ootd.*\.pth$"],
                "size_range": (1000, 8000),
                "priority": 2,
                "step_class": "VirtualFittingImplementation",
                "model_load_method": "load_models",
            },
            
            # Step 07: Post Processing
            "post_processing_enhance": {
                "actual_files": ["enhance_model.pth", "ESRGAN_x4.pth"],
                "search_paths": ["step_07_post_processing"],
                "patterns": [r".*enhance.*\.pth$", r".*ESRGAN.*\.pth$"],
                "size_range": (10, 200),
                "priority": 1,
                "step_class": "PostProcessingImplementation",
                "model_load_method": "load_models",
            },
            
            # Step 08: Quality Assessment
            "quality_assessment_clip": {
                "actual_files": ["pytorch_model.bin"],
                "search_paths": ["step_08_quality_assessment"],
                "patterns": [r".*pytorch_model\.bin$"],
                "size_range": (500, 600),
                "priority": 1,
                "step_class": "QualityAssessmentImplementation",
                "model_load_method": "load_models",
            }
        }
    
    def find_actual_file(self, request_name: str, ai_models_root: Path) -> Optional[Path]:
        """요청명에 대한 실제 파일 찾기 (기존 함수 유지)"""
        if request_name not in self.step_file_mappings:
            self.logger.debug(f"❓ 알 수 없는 요청명: {request_name}")
            return None
        
        mapping = self.step_file_mappings[request_name]
        
        # 1. 정확한 파일명으로 검색
        for filename in mapping["actual_files"]:
            for search_path in mapping["search_paths"]:
                full_path = ai_models_root / search_path / filename
                if full_path.exists() and full_path.is_file():
                    file_size_mb = full_path.stat().st_size / (1024 * 1024)
                    min_size, max_size = mapping["size_range"]
                    
                    if min_size <= file_size_mb <= max_size:
                        self.logger.info(f"✅ {request_name} → {full_path} ({file_size_mb:.1f}MB)")
                        return full_path
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
                            min_size, max_size = mapping["size_range"]
                            
                            if min_size <= file_size_mb <= max_size:
                                self.logger.info(f"✅ {request_name} → {file_path} (패턴 매칭, {file_size_mb:.1f}MB)")
                                return file_path
                except Exception as e:
                    self.logger.debug(f"패턴 검색 오류: {e}")
                    continue
        
        self.logger.warning(f"❌ {request_name} 파일을 찾을 수 없습니다")
        return None
    
    def get_step_info(self, request_name: str) -> Optional[Dict[str, Any]]:
        """🔥 Step 구현체 정보 반환 (새로 추가)"""
        if request_name in self.step_file_mappings:
            mapping = self.step_file_mappings[request_name]
            return {
                "step_class": mapping.get("step_class"),
                "model_load_method": mapping.get("model_load_method"),
                "priority": mapping.get("priority"),
                "patterns": mapping.get("patterns", []),
            }
        return None

# ==============================================
# 🔥 2. DetectedModel 클래스 (Step 연동 정보 추가)
# ==============================================

@dataclass
class DetectedModel:
    """탐지된 모델 정보 + Step 연동 정보"""
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
    
    # 추가 정보
    checkpoint_path: Optional[str] = None
    device_compatible: bool = True
    recommended_device: str = "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
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
            
            "metadata": {
                "detection_time": time.time(),
                "file_extension": self.path.suffix
            }
        }
    
    def can_be_loaded_by_step(self) -> bool:
        """Step 구현체로 로드 가능한지 확인"""
        return (self.step_can_load and 
                self.step_class_name is not None and 
                self.model_load_method is not None and
                self.checkpoint_path is not None)

# ==============================================
# 🔥 3. 수정된 모델 탐지기 (Step 연동)
# ==============================================

class FixedModelDetector:
    """수정된 모델 탐지기 (Step 구현체 연동)"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.FixedModelDetector")
        self.file_mapper = RealFileMapper()
        self.ai_models_root = self._find_ai_models_root()
        self.detected_models: Dict[str, DetectedModel] = {}
        
        # 시스템 정보
        self.is_m3_max = self._detect_m3_max()
        self.conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
        
        self.logger.info(f"🔧 Step 연동 모델 탐지기 초기화")
        self.logger.info(f"   AI 모델 루트: {self.ai_models_root}")
        self.logger.info(f"   M3 Max: {self.is_m3_max}, conda: {bool(self.conda_env)}")
    
    def _find_ai_models_root(self) -> Path:
        """AI 모델 루트 디렉토리 찾기"""
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
        """M3 Max 감지"""
        try:
            import platform
            return 'arm64' in platform.machine().lower()
        except:
            return False
    
    def detect_all_models(self) -> Dict[str, DetectedModel]:
        """모든 모델 탐지 (Step 연동 정보 포함)"""
        self.detected_models.clear()
        
        if not self.ai_models_root.exists():
            self.logger.error(f"❌ AI 모델 루트가 존재하지 않습니다: {self.ai_models_root}")
            return {}
        
        # 요청명별로 실제 파일 찾기 + Step 정보 추가
        for request_name in self.file_mapper.step_file_mappings.keys():
            try:
                # 1. 실제 파일 찾기
                actual_file = self.file_mapper.find_actual_file(request_name, self.ai_models_root)
                
                if actual_file:
                    # 2. Step 정보 가져오기
                    step_info = self.file_mapper.get_step_info(request_name)
                    
                    # 3. DetectedModel 생성 (Step 연동 정보 포함)
                    model = self._create_detected_model_with_step_info(request_name, actual_file, step_info)
                    if model:
                        self.detected_models[model.name] = model
                        
            except Exception as e:
                self.logger.error(f"❌ {request_name} 탐지 실패: {e}")
                continue
        
        # 추가 파일들 자동 스캔
        self._scan_additional_files()
        
        self.logger.info(f"🎉 Step 연동 모델 탐지 완료: {len(self.detected_models)}개")
        return self.detected_models
    
    def _create_detected_model_with_step_info(self, request_name: str, file_path: Path, step_info: Optional[Dict]) -> Optional[DetectedModel]:
        """DetectedModel 생성 (Step 연동 정보 포함)"""
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
            
            model = DetectedModel(
                name=request_name,
                path=file_path,
                step_name=step_name,
                model_type=step_name.replace("Step", "").lower(),
                file_size_mb=file_size_mb,
                confidence_score=1.0,  # 정확한 매핑이므로 최대 신뢰도
                
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
    
    def _extract_step_name(self, request_name: str) -> str:
        """요청명에서 Step 이름 추출"""
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
        """추가 파일들 자동 스캔 (기존 함수 유지)"""
        try:
            # Ultra 모델들 스캔
            ultra_dir = self.ai_models_root / "ultra_models"
            if ultra_dir.exists():
                self._scan_ultra_models(ultra_dir)
            
            # 체크포인트 디렉토리 스캔
            checkpoints_dir = self.ai_models_root / "checkpoints"
            if checkpoints_dir.exists():
                self._scan_checkpoints(checkpoints_dir)
                
        except Exception as e:
            self.logger.debug(f"추가 스캔 오류: {e}")
    
    def _scan_ultra_models(self, ultra_dir: Path):
        """Ultra 모델 스캔 (기존 함수 유지)"""
        model_extensions = {'.pth', '.bin', '.safetensors', '.ckpt'}
        
        for file_path in ultra_dir.rglob('*'):
            if (file_path.is_file() and 
                file_path.suffix.lower() in model_extensions and
                file_path.stat().st_size > 50 * 1024 * 1024):  # 50MB 이상
                
                try:
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
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
                        confidence_score=0.8,
                        checkpoint_path=str(file_path),
                        device_compatible=True,
                        recommended_device="mps" if self.is_m3_max else "cpu"
                    )
                    
                    self.detected_models[model_name] = model
                    self.logger.debug(f"✅ Ultra 모델: {model_name} ({file_size_mb:.1f}MB)")
                    
                except Exception as e:
                    self.logger.debug(f"Ultra 모델 처리 오류 {file_path}: {e}")
                    continue
    
    def _scan_checkpoints(self, checkpoints_dir: Path):
        """체크포인트 디렉토리 스캔 (기존 함수 유지)"""
        for subdir in checkpoints_dir.iterdir():
            if subdir.is_dir():
                for file_path in subdir.rglob('*.pth'):
                    if file_path.name not in [m.path.name for m in self.detected_models.values()]:
                        try:
                            file_size_mb = file_path.stat().st_size / (1024 * 1024)
                            if file_size_mb > 10:  # 10MB 이상만
                                model_name = f"checkpoint_{subdir.name}_{file_path.stem}"
                                
                                model = DetectedModel(
                                    name=model_name,
                                    path=file_path,
                                    step_name="CheckpointModel",
                                    model_type="checkpoint",
                                    file_size_mb=file_size_mb,
                                    confidence_score=0.6,
                                    checkpoint_path=str(file_path),
                                    device_compatible=True,
                                    recommended_device="mps" if self.is_m3_max else "cpu"
                                )
                                
                                self.detected_models[model_name] = model
                                self.logger.debug(f"✅ 체크포인트: {model_name} ({file_size_mb:.1f}MB)")
                                
                        except Exception as e:
                            self.logger.debug(f"체크포인트 처리 오류 {file_path}: {e}")
                            continue

# ==============================================
# 🔥 4. ModelLoader 호환 인터페이스 (Step 연동)
# ==============================================

def get_step_loadable_models() -> List[Dict[str, Any]]:
    """🔥 Step 구현체로 로드 가능한 모델들만 반환"""
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
    
    return sorted(loadable_models, key=lambda x: x["confidence"], reverse=True)

def create_step_model_loader_config() -> Dict[str, Any]:
    """🔥 Step 구현체 연동용 ModelLoader 설정 생성"""
    detector = get_global_detector()
    detected_models = detector.detect_all_models()
    
    config = {
        "version": "step_integrated_detector_v3.1",
        "generated_at": time.time(),
        "device": "mps" if detector.is_m3_max else "cpu",
        "is_m3_max": detector.is_m3_max,
        "conda_env": detector.conda_env,
        "models": {},
        "step_mappings": {},
        "step_loadable_count": 0
    }
    
    # 모델별 설정
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
    
    # 통계
    config["summary"] = {
        "total_models": len(detected_models),
        "step_loadable_models": config["step_loadable_count"],
        "total_size_gb": sum(m.file_size_mb for m in detected_models.values()) / 1024,
        "device_optimized": detector.is_m3_max,
        "step_integration_ready": config["step_loadable_count"] > 0
    }
    
    logger.info(f"✅ Step 연동 설정 생성: {len(detected_models)}개 모델, {config['step_loadable_count']}개 Step 로드 가능")
    return config

# ==============================================
# 🔥 5. 전역 인스턴스 및 인터페이스 (기존 유지)
# ==============================================

_global_detector: Optional[FixedModelDetector] = None
_detector_lock = threading.Lock()

def get_global_detector() -> FixedModelDetector:
    """전역 탐지기 인스턴스"""
    global _global_detector
    if _global_detector is None:
        with _detector_lock:
            if _global_detector is None:
                _global_detector = FixedModelDetector()
    return _global_detector

def quick_model_detection() -> Dict[str, DetectedModel]:
    """빠른 모델 탐지"""
    detector = get_global_detector()
    return detector.detect_all_models()

def list_available_models(step_class: Optional[str] = None) -> List[Dict[str, Any]]:
    """사용 가능한 모델 목록"""
    detector = get_global_detector()
    models = detector.detect_all_models()
    
    result = []
    for model in models.values():
        model_dict = model.to_dict()
        
        if step_class and model_dict["step_class"] != step_class:
            continue
        
        result.append(model_dict)
    
    return sorted(result, key=lambda x: x["confidence"], reverse=True)

def get_models_for_step(step_name: str) -> List[Dict[str, Any]]:
    """Step별 모델 조회"""
    models = list_available_models(step_class=step_name)
    return models

def validate_model_exists(model_name: str) -> bool:
    """모델 존재 확인"""
    detector = get_global_detector()
    return model_name in detector.detected_models

def generate_advanced_model_loader_config() -> Dict[str, Any]:
    """🔥 고급 ModelLoader 설정 생성 (Step 연동 포함)"""
    return create_step_model_loader_config()

def create_step_interface(step_name: str, config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Step 인터페이스 생성"""
    models = get_models_for_step(step_name)
    if not models:
        return None
    
    # Step 로드 가능한 모델 우선 선택
    loadable_models = [m for m in models if m.get("step_implementation", {}).get("load_ready", False)]
    primary_model = loadable_models[0] if loadable_models else models[0]
    
    return {
        "step_name": step_name,
        "primary_model": primary_model,
        "config": config or {},
        "load_ready": len(loadable_models) > 0,
        "step_integration": primary_model.get("step_implementation", {}),
        "created_at": time.time()
    }

# 기존 호환성 별칭
RealWorldModelDetector = FixedModelDetector
create_real_world_detector = lambda **kwargs: FixedModelDetector()
comprehensive_model_detection = quick_model_detection

# ==============================================
# 🔥 6. 익스포트
# ==============================================

__all__ = [
    'FixedModelDetector',
    'DetectedModel', 
    'RealFileMapper',
    'get_global_detector',
    'quick_model_detection',
    'list_available_models',
    'get_models_for_step',
    'get_step_loadable_models',  # 🔥 새로 추가
    'create_step_model_loader_config',  # 🔥 새로 추가
    'generate_advanced_model_loader_config',
    'validate_model_exists',
    'create_step_interface',
    
    # 호환성
    'RealWorldModelDetector',
    'create_real_world_detector',
    'comprehensive_model_detection'
]

# ==============================================
# 🔥 7. 초기화
# ==============================================

logger.info("✅ Step 연동 자동 모델 탐지기 v3.1 로드 완료")
logger.info("🎯 체크포인트 경로 → Step 구현체 완벽 연동")
logger.info("🔧 기존 load_models() 함수 활용")
logger.info("✅ Step이 실제 AI 모델 생성하는 구조 지원")

# 초기화 테스트
try:
    _test_detector = get_global_detector()
    logger.info(f"🚀 Step 연동 탐지기 준비 완료!")
    logger.info(f"   AI 모델 루트: {_test_detector.ai_models_root}")
    logger.info(f"   M3 Max: {_test_detector.is_m3_max}")
except Exception as e:
    logger.error(f"❌ 초기화 실패: {e}")

if __name__ == "__main__":
    print("🔍 Step 연동 자동 모델 탐지기 v3.1 테스트")
    print("=" * 60)
    
    # 테스트 실행
    models = quick_model_detection()
    print(f"✅ 탐지된 모델: {len(models)}개")
    
    # Step 로드 가능한 모델들 확인
    loadable_models = get_step_loadable_models()
    print(f"✅ Step 로드 가능: {len(loadable_models)}개")
    
    if loadable_models:
        for model in loadable_models[:3]:
            step_info = model["step_implementation"]
            print(f"   - {model['name']}: {step_info['step_class_name']}.{step_info['method']}()")
    
    print("🎉 Step 연동 테스트 완료!")