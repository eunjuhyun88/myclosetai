#!/usr/bin/env python3
"""
🍎 MyCloset AI - Step 모델 요청 관리자
=======================================
Step 클래스들이 필요한 모델을 요청하고 관리하는 시스템

📋 주요 기능:
- Step별 모델 요청 정의
- 모델 의존성 관리  
- 권장 모델 자동 선택
- 메모리 효율적 모델 로딩

🔧 Step 클래스와 ModelLoader 간 브릿지 역할
"""

import logging
from typing import Dict, List, Optional, Any, Type, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
from pathlib import Path

# ==============================================
# 🎯 Step 모델 요청 정의
# ==============================================

class StepModelType(Enum):
    """Step별 모델 타입 정의"""
    # Step 01: Human Parsing
    HUMAN_PARSING_GRAPHONOMY = "human_parsing_graphonomy"
    HUMAN_PARSING_SCHP = "human_parsing_schp"
    HUMAN_PARSING_LIGHTWEIGHT = "human_parsing_lightweight"
    
    # Step 02: Pose Estimation  
    POSE_OPENPOSE = "pose_openpose"
    POSE_MEDIAPIPE = "pose_mediapipe"
    POSE_YOLO = "pose_yolo"
    
    # Step 03: Cloth Segmentation
    CLOTH_U2NET = "cloth_u2net"
    CLOTH_SAM = "cloth_sam"
    CLOTH_REMBG = "cloth_rembg"
    
    # Step 04: Geometric Matching
    GEOMETRIC_GMM = "geometric_gmm"
    GEOMETRIC_TPS = "geometric_tps"
    GEOMETRIC_LIGHTWEIGHT = "geometric_lightweight"
    
    # Step 05: Cloth Warping
    WARPING_TOM = "warping_tom"
    WARPING_HRVITON = "warping_hrviton"
    WARPING_LIGHTWEIGHT = "warping_lightweight"
    
    # Step 06: Virtual Fitting
    FITTING_OOTDIFFUSION = "fitting_ootdiffusion"
    FITTING_HRVITON = "fitting_hrviton"
    FITTING_STABLE_DIFFUSION = "fitting_stable_diffusion"
    
    # Step 07: Post Processing
    POST_ESRGAN = "post_esrgan"
    POST_GFPGAN = "post_gfpgan"
    POST_ENHANCER = "post_enhancer"
    
    # Step 08: Quality Assessment
    QUALITY_CLIP = "quality_clip"
    QUALITY_LPIPS = "quality_lpips"
    QUALITY_COMBINED = "quality_combined"

@dataclass
class ModelRequest:
    """모델 요청 정보"""
    model_type: StepModelType
    required: bool = True
    priority: int = 1  # 1=필수, 2=권장, 3=선택
    memory_mb: Optional[int] = None
    checkpoint_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    fallback_models: Optional[List[StepModelType]] = None

@dataclass 
class StepModelRequirements:
    """Step별 모델 요구사항"""
    step_name: str
    primary_models: List[ModelRequest]
    optional_models: List[ModelRequest] = None
    memory_budget_mb: Optional[int] = None
    concurrent_models: int = 1

# ==============================================
# 🎯 Step별 모델 요구사항 정의
# ==============================================

def get_step_model_requirements() -> Dict[str, StepModelRequirements]:
    """Step별 모델 요구사항 반환"""
    
    requirements = {
        # Step 01: Human Parsing
        "HumanParsingStep": StepModelRequirements(
            step_name="HumanParsingStep",
            primary_models=[
                ModelRequest(
                    model_type=StepModelType.HUMAN_PARSING_GRAPHONOMY,
                    required=True,
                    priority=1,
                    memory_mb=2048,
                    checkpoint_path="ai_models/checkpoints/step_01_human_parsing/graphonomy.pth",
                    fallback_models=[StepModelType.HUMAN_PARSING_SCHP]
                )
            ],
            optional_models=[
                ModelRequest(
                    model_type=StepModelType.HUMAN_PARSING_LIGHTWEIGHT,
                    required=False,
                    priority=3,
                    memory_mb=512
                )
            ],
            memory_budget_mb=4096,
            concurrent_models=1
        ),
        
        # Step 02: Pose Estimation
        "PoseEstimationStep": StepModelRequirements(
            step_name="PoseEstimationStep", 
            primary_models=[
                ModelRequest(
                    model_type=StepModelType.POSE_OPENPOSE,
                    required=True,
                    priority=1,
                    memory_mb=1024,
                    checkpoint_path="ai_models/checkpoints/step_02_pose_estimation/openpose.pth",
                    fallback_models=[StepModelType.POSE_MEDIAPIPE]
                )
            ],
            memory_budget_mb=2048,
            concurrent_models=1
        ),
        
        # Step 03: Cloth Segmentation
        "ClothSegmentationStep": StepModelRequirements(
            step_name="ClothSegmentationStep",
            primary_models=[
                ModelRequest(
                    model_type=StepModelType.CLOTH_U2NET,
                    required=True,
                    priority=1,
                    memory_mb=1024,
                    checkpoint_path="ai_models/checkpoints/step_03_cloth_segmentation/u2net.pth",
                    fallback_models=[StepModelType.CLOTH_SAM]
                )
            ],
            memory_budget_mb=2048,
            concurrent_models=1
        ),
        
        # Step 04: Geometric Matching
        "GeometricMatchingStep": StepModelRequirements(
            step_name="GeometricMatchingStep",
            primary_models=[
                ModelRequest(
                    model_type=StepModelType.GEOMETRIC_GMM,
                    required=True,
                    priority=1,
                    memory_mb=512,
                    checkpoint_path="ai_models/checkpoints/step_04_geometric_matching/gmm_final.pth",
                    fallback_models=[StepModelType.GEOMETRIC_LIGHTWEIGHT]
                )
            ],
            memory_budget_mb=1024,
            concurrent_models=1
        ),
        
        # Step 05: Cloth Warping
        "ClothWarpingStep": StepModelRequirements(
            step_name="ClothWarpingStep",
            primary_models=[
                ModelRequest(
                    model_type=StepModelType.WARPING_TOM,
                    required=True,
                    priority=1,
                    memory_mb=4096,
                    checkpoint_path="ai_models/checkpoints/step_05_cloth_warping/tom_final.pth",
                    fallback_models=[StepModelType.WARPING_LIGHTWEIGHT]
                )
            ],
            memory_budget_mb=8192,
            concurrent_models=1
        ),
        
        # Step 06: Virtual Fitting
        "VirtualFittingStep": StepModelRequirements(
            step_name="VirtualFittingStep",
            primary_models=[
                ModelRequest(
                    model_type=StepModelType.FITTING_OOTDIFFUSION,
                    required=True,
                    priority=1,
                    memory_mb=8192,
                    checkpoint_path="ai_models/checkpoints/step_06_virtual_fitting/ootdiffusion.pth",
                    fallback_models=[StepModelType.FITTING_HRVITON]
                )
            ],
            memory_budget_mb=16384,
            concurrent_models=1
        ),
        
        # Step 07: Post Processing
        "PostProcessingStep": StepModelRequirements(
            step_name="PostProcessingStep",
            primary_models=[
                ModelRequest(
                    model_type=StepModelType.POST_ENHANCER,
                    required=True,
                    priority=1,
                    memory_mb=1024,
                    checkpoint_path="ai_models/checkpoints/step_07_post_processing/RealESRGAN_x4plus.pth"
                )
            ],
            memory_budget_mb=2048,
            concurrent_models=1
        ),
        
        # Step 08: Quality Assessment
        "QualityAssessmentStep": StepModelRequirements(
            step_name="QualityAssessmentStep",
            primary_models=[
                ModelRequest(
                    model_type=StepModelType.QUALITY_COMBINED,
                    required=True,
                    priority=1,
                    memory_mb=512
                )
            ],
            memory_budget_mb=1024,
            concurrent_models=1
        )
    }
    
    return requirements

# ==============================================
# 🔧 모델 요청 관리자
# ==============================================

class StepModelRequestManager:
    """Step 모델 요청 관리자"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.requirements = get_step_model_requirements()
        self.active_requests: Dict[str, List[ModelRequest]] = {}
        
    def get_step_requirements(self, step_name: str) -> Optional[StepModelRequirements]:
        """Step 모델 요구사항 조회"""
        return self.requirements.get(step_name)
        
    def get_required_models(self, step_name: str) -> List[ModelRequest]:
        """Step 필수 모델 목록"""
        requirements = self.get_step_requirements(step_name)
        if not requirements:
            return []
            
        return [model for model in requirements.primary_models if model.required]
        
    def get_recommended_model(self, step_name: str) -> Optional[ModelRequest]:
        """Step 권장 모델 (우선순위 1)"""
        required_models = self.get_required_models(step_name)
        if required_models:
            return min(required_models, key=lambda x: x.priority)
        return None
        
    def get_fallback_models(self, step_name: str, model_type: StepModelType) -> List[StepModelType]:
        """모델 폴백 목록"""
        requirements = self.get_step_requirements(step_name)
        if not requirements:
            return []
            
        for model in requirements.primary_models:
            if model.model_type == model_type and model.fallback_models:
                return model.fallback_models
                
        return []
        
    def estimate_memory_usage(self, step_name: str) -> int:
        """Step 예상 메모리 사용량 (MB)"""
        requirements = self.get_step_requirements(step_name)
        if not requirements:
            return 1024  # 기본값
            
        total_memory = 0
        for model in requirements.primary_models:
            if model.required and model.memory_mb:
                total_memory += model.memory_mb
                
        return max(total_memory, 512)  # 최소 512MB
        
    def validate_checkpoint_paths(self, step_name: str, base_path: Optional[Path] = None) -> Dict[str, bool]:
        """체크포인트 파일 존재 여부 확인"""
        requirements = self.get_step_requirements(step_name)
        if not requirements:
            return {}
            
        base_path = base_path or Path("backend")
        validation_results = {}
        
        for model in requirements.primary_models:
            if model.checkpoint_path:
                full_path = base_path / model.checkpoint_path
                validation_results[model.model_type.value] = full_path.exists()
                
        return validation_results
        
    def get_step_names(self) -> List[str]:
        """등록된 Step 이름 목록"""
        return list(self.requirements.keys())
        
    def get_all_model_types(self) -> List[StepModelType]:
        """모든 모델 타입 목록"""
        model_types = set()
        for requirements in self.requirements.values():
            for model in requirements.primary_models:
                model_types.add(model.model_type)
            if requirements.optional_models:
                for model in requirements.optional_models:
                    model_types.add(model.model_type)
        return list(model_types)

# ==============================================
# 🔥 팩토리 함수 및 전역 인스턴스
# ==============================================

# 전역 모델 요청 관리자
_global_request_manager: Optional[StepModelRequestManager] = None

def get_step_model_request_manager() -> StepModelRequestManager:
    """전역 Step 모델 요청 관리자 반환"""
    global _global_request_manager
    
    if _global_request_manager is None:
        _global_request_manager = StepModelRequestManager()
        
    return _global_request_manager

def create_step_model_request_manager() -> StepModelRequestManager:
    """새로운 Step 모델 요청 관리자 생성"""
    return StepModelRequestManager()

# ==============================================
# 🔧 유틸리티 함수
# ==============================================

def list_step_model_requirements() -> Dict[str, Dict[str, Any]]:
    """Step별 모델 요구사항 요약"""
    manager = get_step_model_request_manager()
    summary = {}
    
    for step_name in manager.get_step_names():
        requirements = manager.get_step_requirements(step_name)
        if requirements:
            summary[step_name] = {
                'primary_models': len(requirements.primary_models),
                'optional_models': len(requirements.optional_models or []),
                'memory_budget_mb': requirements.memory_budget_mb,
                'concurrent_models': requirements.concurrent_models,
                'required_models': [m.model_type.value for m in requirements.primary_models if m.required]
            }
            
    return summary

def validate_all_checkpoint_paths(base_path: Optional[Path] = None) -> Dict[str, Dict[str, bool]]:
    """모든 Step의 체크포인트 경로 검증"""
    manager = get_step_model_request_manager()
    results = {}
    
    for step_name in manager.get_step_names():
        results[step_name] = manager.validate_checkpoint_paths(step_name, base_path)
        
    return results

def get_total_memory_estimate() -> int:
    """전체 시스템 예상 메모리 사용량 (MB)"""
    manager = get_step_model_request_manager()
    total = 0
    
    for step_name in manager.get_step_names():
        total += manager.estimate_memory_usage(step_name)
        
    return total

# ==============================================
# 📋 모듈 Export
# ==============================================

__all__ = [
    # 핵심 클래스
    'StepModelType',
    'ModelRequest', 
    'StepModelRequirements',
    'StepModelRequestManager',
    
    # 팩토리 함수
    'get_step_model_request_manager',
    'create_step_model_request_manager',
    'get_step_model_requirements',
    
    # 유틸리티 함수
    'list_step_model_requirements',
    'validate_all_checkpoint_paths',
    'get_total_memory_estimate'
]

# 로거 설정
logger = logging.getLogger(__name__)
logger.info("✅ Step 모델 요청 관리자 모듈 로드 완료")