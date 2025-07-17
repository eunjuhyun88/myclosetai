# app/ai_pipeline/utils/step_model_requests.py
"""
🔥 Step별 ModelLoader 요청 정보 독립 모듈 v4.0
✅ 순환참조 완전 제거
✅ 다른 모듈 의존성 없음
✅ 순수 데이터 정의만 포함
"""

import os
import re
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# ==============================================
# 🔥 순수 데이터 구조 정의 (의존성 없음)
# ==============================================

class StepPriority(Enum):
    """Step 우선순위 (파이프라인 중요도)"""
    CRITICAL = 1      # 필수 Step
    HIGH = 2          # 중요 Step  
    MEDIUM = 3        # 일반 Step
    LOW = 4           # 보조 Step

@dataclass
class ModelRequestInfo:
    """Step에서 ModelLoader로 요청하는 완전한 정보 (순수 데이터)"""
    # === 기본 모델 정보 ===
    model_name: str
    step_class: str  
    step_priority: StepPriority
    model_type: str
    
    # === 디바이스 및 성능 설정 ===
    device: str = "auto"
    precision: str = "fp16"
    use_neural_engine: bool = True
    enable_metal_shaders: bool = True
    
    # === 입력/출력 스펙 ===
    input_size: Tuple[int, int] = (512, 512)
    num_classes: Optional[int] = None
    output_format: str = "tensor"
    
    # === 체크포인트 요구사항 ===
    checkpoint_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # === 최적화 파라미터 ===
    optimization_params: Dict[str, Any] = field(default_factory=dict)
    
    # === Step별 특화 파라미터 ===
    step_specific_params: Dict[str, Any] = field(default_factory=dict)
    
    # === 대체 모델 및 폴백 ===
    alternative_models: List[str] = field(default_factory=list)
    fallback_config: Dict[str, Any] = field(default_factory=dict)

# ==============================================
# 🔥 Step별 모델 요청 정보 데이터베이스 (순수 데이터)
# ==============================================

STEP_MODEL_REQUESTS = {
    "HumanParsingStep": ModelRequestInfo(
        model_name="human_parsing_graphonomy",
        step_class="HumanParsingStep",
        step_priority=StepPriority.CRITICAL,
        model_type="GraphonomyModel",
        input_size=(512, 512),
        num_classes=20,
        output_format="segmentation_mask",
        checkpoint_requirements={
            "primary_model_patterns": [
                "*human*parsing*.pth",
                "*schp*atr*.pth", 
                "*graphonomy*.pth",
                "*atr*model*.pth"
            ],
            "required_files": ["model.pth"],
            "optional_files": ["config.json", "class_names.txt"],
            "min_file_size_mb": 50,
            "max_file_size_mb": 500,
            "expected_extensions": [".pth", ".pt", ".pkl"],
            "model_architecture": "ResNet + ASPP",
            "expected_layers": ["backbone", "aspp", "classifier"]
        },
        optimization_params={
            "batch_size": 1,
            "max_batch_size": 4,
            "memory_fraction": 0.3,
            "enable_amp": True,
            "cache_model": True,
            "warmup_iterations": 3
        },
        step_specific_params={
            "body_parts": ["head", "torso", "arms", "legs", "accessories"],
            "segmentation_classes": 20,
            "postprocess_enabled": True,
            "confidence_threshold": 0.3
        },
        alternative_models=[
            "human_parsing_atr",
            "human_parsing_lip", 
            "human_parsing_schp"
        ]
    ),
    
    "PoseEstimationStep": ModelRequestInfo(
        model_name="pose_estimation_openpose",
        step_class="PoseEstimationStep",
        step_priority=StepPriority.HIGH,
        model_type="OpenPoseModel",
        input_size=(368, 368),
        num_classes=18,
        output_format="keypoints_heatmap",
        checkpoint_requirements={
            "primary_model_patterns": [
                "*pose*model*.pth",
                "*openpose*.pth",
                "*body*pose*.pth"
            ],
            "required_files": ["body_pose_model.pth"],
            "min_file_size_mb": 10,
            "max_file_size_mb": 200,
            "expected_extensions": [".pth", ".pt", ".onnx"],
            "model_architecture": "VGG + PAF"
        },
        optimization_params={
            "batch_size": 1,
            "max_batch_size": 2,
            "memory_fraction": 0.25,
            "enable_tensorrt": True
        },
        step_specific_params={
            "keypoints_format": "coco",
            "confidence_threshold": 0.1,
            "num_stages": 6
        }
    ),
    
    "ClothSegmentationStep": ModelRequestInfo(
        model_name="cloth_segmentation_u2net",
        step_class="ClothSegmentationStep", 
        step_priority=StepPriority.HIGH,
        model_type="U2NetModel",
        input_size=(320, 320),
        num_classes=1,
        output_format="binary_mask",
        checkpoint_requirements={
            "primary_model_patterns": [
                "*u2net*.pth",
                "*cloth*segmentation*.pth",
                "*sam*vit*.pth"
            ],
            "required_files": ["u2net.pth"],
            "min_file_size_mb": 20,
            "max_file_size_mb": 1000,
            "expected_extensions": [".pth", ".pt", ".onnx"]
        },
        optimization_params={
            "batch_size": 4,
            "max_batch_size": 8,
            "memory_fraction": 0.4,
            "enable_tensorrt": True
        },
        step_specific_params={
            "segmentation_type": "binary",
            "supports_multiple_items": True,
            "background_removal": True
        }
    ),
    
    "GeometricMatchingStep": ModelRequestInfo(
        model_name="geometric_matching_gmm",
        step_class="GeometricMatchingStep",
        step_priority=StepPriority.MEDIUM,
        model_type="GeometricMatchingModel",
        input_size=(512, 384),
        output_format="transformation_matrix",
        checkpoint_requirements={
            "primary_model_patterns": [
                "*geometric*matching*.pth",
                "*gmm*.pth",
                "*tps*.pth"
            ],
            "required_files": ["gmm.pth"],
            "min_file_size_mb": 5,
            "max_file_size_mb": 100,
            "expected_extensions": [".pth", ".pt"]
        },
        optimization_params={
            "batch_size": 2,
            "max_batch_size": 4,
            "memory_fraction": 0.2
        },
        step_specific_params={
            "matching_method": "tps",
            "num_control_points": 25,
            "grid_size": 30
        }
    ),
    
    "ClothWarpingStep": ModelRequestInfo(
        model_name="cloth_warping_tom",
        step_class="ClothWarpingStep",
        step_priority=StepPriority.MEDIUM,
        model_type="HRVITONModel",
        input_size=(512, 384),
        output_format="warped_cloth",
        checkpoint_requirements={
            "primary_model_patterns": [
                "*tom*final*.pth",
                "*cloth*warping*.pth",
                "*hrviton*.pth"
            ],
            "required_files": ["tom_final.pth"],
            "min_file_size_mb": 100,
            "max_file_size_mb": 1000,
            "expected_extensions": [".pth", ".pt"]
        },
        optimization_params={
            "batch_size": 1,
            "max_batch_size": 2,
            "memory_fraction": 0.5,
            "enable_amp": True
        },
        step_specific_params={
            "warping_method": "physics_based",
            "enable_physics": True,
            "deformation_strength": 0.7
        }
    ),
    
    "VirtualFittingStep": ModelRequestInfo(
        model_name="virtual_fitting_stable_diffusion",
        step_class="VirtualFittingStep",
        step_priority=StepPriority.CRITICAL,
        model_type="StableDiffusionPipeline",
        input_size=(512, 512),
        output_format="rgb_image",
        checkpoint_requirements={
            "primary_model_patterns": [
                "*diffusion*pytorch*model*.bin",
                "*stable*diffusion*.safetensors",
                "*ootdiffusion*.pth"
            ],
            "required_files": ["diffusion_pytorch_model.bin"],
            "min_file_size_mb": 500,
            "max_file_size_mb": 5000,
            "expected_extensions": [".bin", ".safetensors", ".pth"]
        },
        optimization_params={
            "batch_size": 1,
            "max_batch_size": 1,
            "memory_fraction": 0.7,
            "enable_attention_slicing": True
        },
        step_specific_params={
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "scheduler_type": "ddim"
        }
    ),
    
    "PostProcessingStep": ModelRequestInfo(
        model_name="post_processing_realesrgan",
        step_class="PostProcessingStep",
        step_priority=StepPriority.LOW,
        model_type="EnhancementModel",
        input_size=(512, 512),
        output_format="enhanced_image",
        checkpoint_requirements={
            "primary_model_patterns": [
                "*realesrgan*.pth",
                "*esrgan*.pth",
                "*enhance*.pth"
            ],
            "required_files": ["realesrgan.pth"],
            "min_file_size_mb": 10,
            "max_file_size_mb": 200,
            "expected_extensions": [".pth", ".pt"]
        },
        optimization_params={
            "batch_size": 2,
            "max_batch_size": 4,
            "memory_fraction": 0.3
        },
        step_specific_params={
            "upscale_factor": 2,
            "enhancement_strength": 0.8
        }
    ),
    
    "QualityAssessmentStep": ModelRequestInfo(
        model_name="quality_assessment_clip",
        step_class="QualityAssessmentStep",
        step_priority=StepPriority.LOW,
        model_type="CLIPModel",
        input_size=(224, 224),
        output_format="quality_scores",
        checkpoint_requirements={
            "primary_model_patterns": [
                "*clip*vit*.bin",
                "*clip*base*.bin",
                "*quality*assessment*.pth"
            ],
            "required_files": ["clip_vit.bin"],
            "min_file_size_mb": 100,
            "max_file_size_mb": 2000,
            "expected_extensions": [".bin", ".pth", ".pt"]
        },
        optimization_params={
            "batch_size": 4,
            "max_batch_size": 8,
            "memory_fraction": 0.25
        },
        step_specific_params={
            "assessment_metrics": ["quality", "realism", "consistency"],
            "quality_threshold": 0.7
        }
    )
}

# ==============================================
# 🔥 순수 함수들 (외부 의존성 없음)
# ==============================================

def get_step_request_info(step_name: str) -> Optional[ModelRequestInfo]:
    """특정 Step의 ModelLoader 요청 정보 반환"""
    return STEP_MODEL_REQUESTS.get(step_name)

def get_all_step_names() -> List[str]:
    """모든 Step 이름 목록 반환"""
    return list(STEP_MODEL_REQUESTS.keys())

def get_checkpoint_requirements(step_name: str) -> Dict[str, Any]:
    """Step별 체크포인트 요구사항 반환"""
    request_info = STEP_MODEL_REQUESTS.get(step_name)
    return request_info.checkpoint_requirements if request_info else {}

def get_optimization_params(step_name: str) -> Dict[str, Any]:
    """Step별 최적화 파라미터 반환"""
    request_info = STEP_MODEL_REQUESTS.get(step_name)
    return request_info.optimization_params if request_info else {}

def validate_model_for_step(step_name: str, model_path: Path, model_size_mb: float) -> Dict[str, Any]:
    """Step 요구사항에 따른 모델 유효성 검증"""
    request_info = STEP_MODEL_REQUESTS.get(step_name)
    if not request_info:
        return {"valid": False, "reason": f"Unknown step: {step_name}"}
    
    requirements = request_info.checkpoint_requirements
    
    # 파일 크기 검증
    min_size = requirements.get("min_file_size_mb", 0)
    max_size = requirements.get("max_file_size_mb", float('inf'))
    
    if not (min_size <= model_size_mb <= max_size):
        return {
            "valid": False,
            "reason": f"File size {model_size_mb}MB not in range [{min_size}, {max_size}]"
        }
    
    # 파일 확장자 검증
    expected_extensions = requirements.get("expected_extensions", [])
    if expected_extensions and model_path.suffix.lower() not in expected_extensions:
        return {
            "valid": False,
            "reason": f"File extension {model_path.suffix} not in {expected_extensions}"
        }
    
    # 패턴 매칭 검증
    patterns = requirements.get("primary_model_patterns", [])
    model_name_lower = model_path.name.lower()
    
    pattern_matched = False
    for pattern in patterns:
        pattern_regex = pattern.replace("*", ".*").lower()
        if re.search(pattern_regex, model_name_lower):
            pattern_matched = True
            break
    
    if patterns and not pattern_matched:
        return {
            "valid": False,
            "reason": f"File name doesn't match any pattern: {patterns}"
        }
    
    return {
        "valid": True,
        "confidence": 0.9,
        "step_priority": request_info.step_priority.name
    }

def get_all_step_requirements() -> Dict[str, Dict[str, Any]]:
    """모든 Step의 요구사항 반환"""
    requirements = {}
    
    for step_name, request_info in STEP_MODEL_REQUESTS.items():
        requirements[step_name] = {
            "search_patterns": request_info.checkpoint_requirements.get("primary_model_patterns", []),
            "file_extensions": request_info.checkpoint_requirements.get("expected_extensions", []),
            "size_range": {
                "min_mb": request_info.checkpoint_requirements.get("min_file_size_mb", 1),
                "max_mb": request_info.checkpoint_requirements.get("max_file_size_mb", 10000)
            },
            "model_config": {
                "model_type": request_info.model_type,
                "input_size": request_info.input_size,
                "num_classes": request_info.num_classes
            },
            "priority": request_info.step_priority.value,
            "priority_name": request_info.step_priority.name
        }
    
    return requirements

# 모듈 익스포트
__all__ = [
    'ModelRequestInfo',
    'StepPriority', 
    'STEP_MODEL_REQUESTS',
    'get_step_request_info',
    'get_all_step_names',
    'get_checkpoint_requirements',
    'get_optimization_params',
    'validate_model_for_step',
    'get_all_step_requirements'
]