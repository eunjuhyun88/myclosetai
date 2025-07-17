# app/ai_pipeline/utils/step_model_requests.py
"""
🔥 Step별 AI 모델 요청 정의 시스템 v4.0
✅ 실제 Step 클래스 요구사항 완벽 반영
✅ ModelLoader와 100% 호환 데이터 구조
✅ auto_model_detector 완벽 연동
✅ M3 Max 128GB 최적화
✅ 프로덕션 안정성 보장
"""

import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# ==============================================
# 🔥 핵심 데이터 구조
# ==============================================

class StepPriority(Enum):
    """Step 우선순위"""
    CRITICAL = 1      # 필수 (Human Parsing, Virtual Fitting)
    HIGH = 2          # 중요 (Pose Estimation, Cloth Segmentation)
    MEDIUM = 3        # 일반 (Cloth Warping, Geometric Matching)
    LOW = 4           # 보조 (Post Processing, Quality Assessment)

@dataclass
class ModelRequest:
    """Step이 ModelLoader에 요청하는 완전한 정보"""
    # 기본 정보
    model_name: str
    step_class: str
    step_priority: StepPriority
    model_class: str
    
    # 입출력 스펙
    input_size: Tuple[int, int] = (512, 512)
    num_classes: Optional[int] = None
    output_format: str = "tensor"
    
    # 디바이스 설정
    device: str = "auto"
    precision: str = "fp16"
    
    # 체크포인트 탐지 정보 (auto_model_detector용)
    checkpoint_patterns: List[str] = field(default_factory=list)
    file_extensions: List[str] = field(default_factory=list)
    size_range_mb: Tuple[float, float] = (1.0, 10000.0)
    
    # 최적화 설정
    optimization_params: Dict[str, Any] = field(default_factory=dict)
    
    # 대체 모델
    alternative_models: List[str] = field(default_factory=list)
    
    # 메타데이터
    metadata: Dict[str, Any] = field(default_factory=dict)

# ==============================================
# 🔥 Step별 실제 모델 요청 정의
# ==============================================

STEP_MODEL_REQUESTS = {
    
    # Step 01: Human Parsing
    "HumanParsingStep": ModelRequest(
        model_name="human_parsing_graphonomy",
        step_class="HumanParsingStep",
        step_priority=StepPriority.CRITICAL,
        model_class="GraphonomyModel",
        input_size=(512, 512),
        num_classes=20,
        output_format="segmentation_mask",
        
        # 자동 탐지 패턴
        checkpoint_patterns=[
            r".*human.*parsing.*\.pth$",
            r".*schp.*atr.*\.pth$",
            r".*graphonomy.*\.pth$",
            r".*inference.*\.pth$"
        ],
        file_extensions=[".pth", ".pt", ".pkl"],
        size_range_mb=(50.0, 500.0),
        
        # 최적화 설정
        optimization_params={
            "batch_size": 1,
            "memory_fraction": 0.3,
            "enable_amp": True,
            "cache_model": True,
            "warmup_iterations": 3
        },
        
        # 대체 모델
        alternative_models=[
            "human_parsing_atr",
            "human_parsing_lip",
            "human_parsing_u2net"
        ],
        
        # 메타데이터
        metadata={
            "description": "20개 부위 인체 파싱",
            "body_parts": ["head", "torso", "arms", "legs", "accessories"],
            "supports_refinement": True,
            "postprocess_enabled": True
        }
    ),
    
    # Step 02: Pose Estimation
    "PoseEstimationStep": ModelRequest(
        model_name="pose_estimation_openpose",
        step_class="PoseEstimationStep",
        step_priority=StepPriority.HIGH,
        model_class="OpenPoseModel",
        input_size=(368, 368),
        num_classes=18,
        output_format="keypoints_heatmap",
        
        checkpoint_patterns=[
            r".*pose.*model.*\.pth$",
            r".*openpose.*\.pth$",
            r".*body.*pose.*\.pth$"
        ],
        file_extensions=[".pth", ".pt", ".tflite"],
        size_range_mb=(10.0, 200.0),
        
        optimization_params={
            "batch_size": 1,
            "memory_fraction": 0.25,
            "inference_threads": 4,
            "enable_tensorrt": True
        },
        
        alternative_models=[
            "pose_estimation_sk",
            "pose_estimation_lightweight"
        ],
        
        metadata={
            "description": "18개 키포인트 포즈 추정",
            "keypoints_format": "coco",
            "supports_hands": True,
            "num_stages": 6
        }
    ),
    
    # Step 03: Cloth Segmentation
    "ClothSegmentationStep": ModelRequest(
        model_name="cloth_segmentation_u2net",
        step_class="ClothSegmentationStep",
        step_priority=StepPriority.HIGH,
        model_class="U2NetModel",
        input_size=(320, 320),
        num_classes=1,
        output_format="binary_mask",
        
        checkpoint_patterns=[
            r".*u2net.*\.pth$",
            r".*cloth.*segmentation.*\.pth$",
            r".*mobile.*sam.*\.pt$"
        ],
        file_extensions=[".pth", ".pt", ".onnx"],
        size_range_mb=(20.0, 1000.0),
        
        optimization_params={
            "batch_size": 4,
            "memory_fraction": 0.4,
            "enable_half_precision": True,
            "tile_processing": True
        },
        
        alternative_models=[
            "cloth_segmentation_sam",
            "cloth_segmentation_deeplabv3"
        ],
        
        metadata={
            "description": "의류 이진 세그멘테이션",
            "supports_multiple_items": True,
            "background_removal": True,
            "edge_refinement": True
        }
    ),
    
    # Step 04: Geometric Matching
    "GeometricMatchingStep": ModelRequest(
        model_name="geometric_matching_gmm",
        step_class="GeometricMatchingStep",
        step_priority=StepPriority.MEDIUM,
        model_class="GeometricMatchingModel",
        input_size=(512, 384),
        output_format="transformation_matrix",
        
        checkpoint_patterns=[
            r".*geometric.*matching.*\.pth$",
            r".*gmm.*\.pth$",
            r".*tps.*\.pth$"
        ],
        file_extensions=[".pth", ".pt"],
        size_range_mb=(5.0, 100.0),
        
        optimization_params={
            "batch_size": 2,
            "memory_fraction": 0.2,
            "enable_jit_compile": True
        },
        
        alternative_models=[
            "geometric_matching_lightweight"
        ],
        
        metadata={
            "description": "TPS 기반 기하학적 매칭",
            "num_control_points": 25,
            "max_iterations": 1000
        }
    ),
    
    # Step 05: Cloth Warping
    "ClothWarpingStep": ModelRequest(
        model_name="cloth_warping_tom",
        step_class="ClothWarpingStep",
        step_priority=StepPriority.MEDIUM,
        model_class="HRVITONModel",
        input_size=(512, 384),
        output_format="warped_cloth",
        
        checkpoint_patterns=[
            r".*tom.*final.*\.pth$",
            r".*cloth.*warping.*\.pth$",
            r".*hrviton.*\.pth$"
        ],
        file_extensions=[".pth", ".pt"],
        size_range_mb=(100.0, 1000.0),
        
        optimization_params={
            "batch_size": 1,
            "memory_fraction": 0.5,
            "enable_amp": True,
            "gradient_accumulation": 2
        },
        
        alternative_models=[
            "cloth_warping_hrviton_v2"
        ],
        
        metadata={
            "description": "물리 기반 의류 워핑",
            "enable_physics": True,
            "cloth_stiffness": 0.3,
            "supports_wrinkles": True
        }
    ),
    
    # Step 06: Virtual Fitting
    "VirtualFittingStep": ModelRequest(
        model_name="virtual_fitting_stable_diffusion",
        step_class="VirtualFittingStep",
        step_priority=StepPriority.CRITICAL,
        model_class="StableDiffusionPipeline",
        input_size=(512, 512),
        output_format="rgb_image",
        
        checkpoint_patterns=[
            r".*diffusion.*pytorch.*model.*\.bin$",
            r".*stable.*diffusion.*\.safetensors$",
            r".*ootdiffusion.*\.pth$",
            r".*unet.*\.bin$"
        ],
        file_extensions=[".bin", ".safetensors", ".pth"],
        size_range_mb=(500.0, 5000.0),
        
        optimization_params={
            "batch_size": 1,
            "memory_fraction": 0.7,
            "enable_attention_slicing": True,
            "enable_cpu_offload": True
        },
        
        alternative_models=[
            "virtual_fitting_oot",
            "virtual_fitting_hrviton"
        ],
        
        metadata={
            "description": "Diffusion 기반 가상 피팅",
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "scheduler_type": "ddim"
        }
    ),
    
    # Step 07: Post Processing
    "PostProcessingStep": ModelRequest(
        model_name="post_processing_realesrgan",
        step_class="PostProcessingStep",
        step_priority=StepPriority.LOW,
        model_class="EnhancementModel",
        input_size=(512, 512),
        output_format="enhanced_image",
        
        checkpoint_patterns=[
            r".*realesrgan.*\.pth$",
            r".*esrgan.*\.pth$",
            r".*enhance.*\.pth$"
        ],
        file_extensions=[".pth", ".pt"],
        size_range_mb=(10.0, 200.0),
        
        optimization_params={
            "batch_size": 2,
            "memory_fraction": 0.3,
            "tile_size": 512,
            "enable_tensorrt": True
        },
        
        alternative_models=[
            "post_processing_esrgan"
        ],
        
        metadata={
            "description": "ESRGAN 기반 이미지 향상",
            "upscale_factor": 2,
            "face_enhancement": True
        }
    ),
    
    # Step 08: Quality Assessment
    "QualityAssessmentStep": ModelRequest(
        model_name="quality_assessment_clip",
        step_class="QualityAssessmentStep",
        step_priority=StepPriority.LOW,
        model_class="CLIPModel",
        input_size=(224, 224),
        output_format="quality_scores",
        
        checkpoint_patterns=[
            r".*clip.*vit.*\.bin$",
            r".*clip.*base.*\.bin$",
            r".*quality.*assessment.*\.pth$"
        ],
        file_extensions=[".bin", ".pth", ".pt"],
        size_range_mb=(100.0, 2000.0),
        
        optimization_params={
            "batch_size": 4,
            "memory_fraction": 0.25,
            "enable_flash_attention": True
        },
        
        alternative_models=[
            "quality_assessment_combined"
        ],
        
        metadata={
            "description": "CLIP 기반 품질 평가",
            "assessment_metrics": ["quality", "realism", "consistency"],
            "quality_threshold": 0.7
        }
    )
}

# ==============================================
# 🔥 요청 분석 함수들
# ==============================================

def get_step_request(step_name: str) -> Optional[ModelRequest]:
    """Step별 모델 요청 정보 반환"""
    return STEP_MODEL_REQUESTS.get(step_name)

def get_all_step_requests() -> Dict[str, ModelRequest]:
    """모든 Step 요청 정보 반환"""
    return STEP_MODEL_REQUESTS.copy()

def get_checkpoint_patterns(step_name: str) -> List[str]:
    """Step별 체크포인트 탐지 패턴 반환"""
    request = get_step_request(step_name)
    return request.checkpoint_patterns if request else []

def get_model_config_for_step(step_name: str, detected_path: Path) -> Dict[str, Any]:
    """Step 요청을 ModelLoader 설정으로 변환"""
    request = get_step_request(step_name)
    if not request:
        return {}
    
    return {
        "name": request.model_name,
        "model_type": request.model_class,
        "model_class": request.model_class,
        "checkpoint_path": str(detected_path),
        "device": request.device,
        "precision": request.precision,
        "input_size": request.input_size,
        "num_classes": request.num_classes,
        "optimization_params": request.optimization_params,
        "metadata": {
            **request.metadata,
            "step_name": step_name,
            "step_priority": request.step_priority.name,
            "auto_detected": True,
            "detection_time": time.time()
        }
    }

def validate_model_for_step(step_name: str, model_path: Path, size_mb: float) -> Dict[str, Any]:
    """Step 요구사항에 따른 모델 검증"""
    request = get_step_request(step_name)
    if not request:
        return {"valid": False, "reason": f"Unknown step: {step_name}"}
    
    # 크기 검증
    min_size, max_size = request.size_range_mb
    if not (min_size <= size_mb <= max_size):
        return {
            "valid": False,
            "reason": f"Size {size_mb}MB not in range [{min_size}, {max_size}]"
        }
    
    # 확장자 검증
    if model_path.suffix.lower() not in request.file_extensions:
        return {
            "valid": False,
            "reason": f"Extension {model_path.suffix} not in {request.file_extensions}"
        }
    
    # 패턴 매칭
    import re
    model_name = model_path.name.lower()
    pattern_matched = False
    
    for pattern in request.checkpoint_patterns:
        if re.search(pattern, model_name):
            pattern_matched = True
            break
    
    if not pattern_matched:
        return {
            "valid": False,
            "reason": f"Name doesn't match patterns: {request.checkpoint_patterns}"
        }
    
    return {
        "valid": True,
        "confidence": 0.9,
        "step_priority": request.step_priority.name,
        "model_class": request.model_class
    }

def get_step_priorities() -> Dict[str, int]:
    """Step별 우선순위 반환"""
    return {
        step_name: request.step_priority.value
        for step_name, request in STEP_MODEL_REQUESTS.items()
    }

def get_steps_by_priority(priority: StepPriority) -> List[str]:
    """우선순위별 Step 목록 반환"""
    return [
        step_name for step_name, request in STEP_MODEL_REQUESTS.items()
        if request.step_priority == priority
    ]
# ===============================================
# 🔥 누락된 클래스들 추가 (step_model_requests.py 맨 끝에 추가)
# ===============================================

class StepModelRequestAnalyzer:
    """Step 모델 요청사항 분석기 - ModelLoader 연동용"""
    
    @staticmethod
    def get_step_request_info(step_name: str) -> Optional[Dict[str, Any]]:
        """Step별 요청 정보 반환 (ModelLoader 호환)"""
        request = STEP_MODEL_REQUESTS.get(step_name)
        if not request:
            return None
        
        return {
            "model_name": request.model_name,
            "model_type": request.model_class,
            "input_size": request.input_size,
            "num_classes": request.num_classes,
            "device": request.device,
            "precision": request.precision,
            "checkpoint_patterns": request.checkpoint_patterns,
            "optimization_params": request.optimization_params,
            "step_priority": request.step_priority.value,
            "alternative_models": request.alternative_models,
            "metadata": request.metadata
        }
    
    @staticmethod
    def get_all_step_requirements() -> Dict[str, Any]:
        """모든 Step 요구사항 반환"""
        return {
            step_name: StepModelRequestAnalyzer.get_step_request_info(step_name)
            for step_name in STEP_MODEL_REQUESTS.keys()
        }
    
    @staticmethod
    def get_critical_steps() -> List[str]:
        """중요한 Step들 반환"""
        return [
            step_name for step_name, request in STEP_MODEL_REQUESTS.items()
            if request.step_priority == StepPriority.CRITICAL
        ]
    
    @staticmethod
    def get_model_for_step(step_name: str) -> Optional[str]:
        """Step에 대한 권장 모델명 반환"""
        request = STEP_MODEL_REQUESTS.get(step_name)
        return request.model_name if request else None

# ModelLoader 호환 함수들 추가
def get_all_step_requirements() -> Dict[str, Any]:
    """전체 Step 요구사항 (ModelLoader 호환)"""
    return StepModelRequestAnalyzer.get_all_step_requirements()

def create_model_loader_config_from_detection(step_name: str, detected_models: List[Path]) -> Dict[str, Any]:
    """탐지된 모델로부터 ModelLoader 설정 생성"""
    request = get_step_request(step_name)
    if not request or not detected_models:
        return {}
    
    # 가장 큰 모델 선택 (일반적으로 메인 모델)
    best_model = max(detected_models, key=lambda p: p.stat().st_size)
    
    return get_model_config_for_step(step_name, best_model)

logger.info(f"✅ Step Model Requests v4.1 로드 완료 - {len(STEP_MODEL_REQUESTS)}개 Step 정의")
logger.info("🔧 StepModelRequestAnalyzer 클래스 추가 - ModelLoader 호환성 완료")
# ==============================================
# 🔥 모듈 익스포트
# ==============================================
__all__ = [
    # 핵심 클래스
    'StepPriority',
    'ModelRequest',
    'StepModelRequestAnalyzer',  # 🔥 추가

    # 데이터
    'STEP_MODEL_REQUESTS',

    # 함수들
    'get_step_request',
    'get_all_step_requests',
    'get_checkpoint_patterns',
    'get_model_config_for_step',
    'validate_model_for_step',
    'get_step_priorities',
    'get_steps_by_priority',
    'get_all_step_requirements',  # 🔥 추가
    'create_model_loader_config_from_detection'  # 🔥 추가
]

logger.info(f"✅ Step Model Requests 로드 완료 - {len(STEP_MODEL_REQUESTS)}개 Step 정의")