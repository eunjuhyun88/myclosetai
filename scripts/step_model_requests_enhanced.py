#!/usr/bin/env python3
"""
📋 MyCloset AI - Step별 모델 요청사항 v6.1 (실제 파일 기반)
✅ 실제 탐지된 파일들로 패턴 업데이트
✅ 신뢰도 기반 매칭 개선
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class StepModelRequest:
    """Step별 모델 요청사항"""
    step_name: str
    model_name: str
    checkpoint_patterns: List[str]
    size_range_mb: tuple = (0.1, 50000.0)  # 최대 50GB까지 허용
    required_files: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.1  # 매우 관대한 임계값
    device: str = "mps"
    precision: str = "fp16"
    priority: int = 1

# 실제 발견된 파일들 기반 패턴 정의
STEP_MODEL_REQUESTS = {
    "HumanParsingStep": StepModelRequest(
        step_name="HumanParsingStep",
        model_name="human_parsing_enhanced",
        checkpoint_patterns=[
            r".*\.pth$",
            r".*\.pt$", 
            r".*\.bin$",
            r".*\.pkl$",
            r".*schp.*\.pth$",
            r".*graphonomy.*\.pth$",
            r".*parsing.*\.pth$",
            r".*human.*\.pth$"
        ],
        confidence_threshold=0.05
    ),
    
    "PoseEstimationStep": StepModelRequest(
        step_name="PoseEstimationStep", 
        model_name="pose_estimation_enhanced",
        checkpoint_patterns=[
            r".*\.pth$",
            r".*\.pt$",
            r".*\.bin$",
            r".*openpose.*\.pth$",
            r".*pose.*\.pth$",
            r".*body.*\.pth$"
        ],
        confidence_threshold=0.05
    ),
    
    "ClothSegmentationStep": StepModelRequest(
        step_name="ClothSegmentationStep",
        model_name="cloth_segmentation_enhanced", 
        checkpoint_patterns=[
            r".*\.pth$",
            r".*\.pt$",
            r".*\.bin$",
            r".*u2net.*\.pth$",
            r".*segment.*\.pth$",
            r".*cloth.*\.pth$"
        ],
        confidence_threshold=0.05
    ),
    
    "GeometricMatchingStep": StepModelRequest(
        step_name="GeometricMatchingStep",
        model_name="geometric_matching_enhanced",
        checkpoint_patterns=[
            r".*\.pth$",
            r".*\.pt$", 
            r".*\.bin$",
            r".*gmm.*\.pth$",
            r".*tps.*\.pth$",
            r".*matching.*\.pth$"
        ],
        confidence_threshold=0.05
    ),
    
    "ClothWarpingStep": StepModelRequest(
        step_name="ClothWarpingStep",
        model_name="cloth_warping_enhanced",
        checkpoint_patterns=[
            r".*\.pth$",
            r".*\.pt$",
            r".*\.bin$", 
            r".*warp.*\.pth$",
            r".*tom.*\.pth$"
        ],
        confidence_threshold=0.05
    ),
    
    "VirtualFittingStep": StepModelRequest(
        step_name="VirtualFittingStep",
        model_name="virtual_fitting_enhanced",
        checkpoint_patterns=[
            r".*\.pth$",
            r".*\.pt$",
            r".*\.bin$",
            r".*\.safetensors$",
            r".*viton.*\.pth$",
            r".*ootd.*\.bin$",
            r".*diffusion.*\.bin$",
            r".*fitting.*\.pth$"
        ],
        size_range_mb=(100.0, 50000.0),  # 대용량 모델
        confidence_threshold=0.02
    ),
    
    "PostProcessingStep": StepModelRequest(
        step_name="PostProcessingStep",
        model_name="post_processing_enhanced",
        checkpoint_patterns=[
            r".*\.pth$",
            r".*\.pt$",
            r".*\.bin$",
            r".*enhance.*\.pth$",
            r".*super.*\.pth$",
            r".*resolution.*\.pth$"
        ],
        confidence_threshold=0.05
    ),
    
    "QualityAssessmentStep": StepModelRequest(
        step_name="QualityAssessmentStep", 
        model_name="quality_assessment_enhanced",
        checkpoint_patterns=[
            r".*\.pth$",
            r".*\.pt$",
            r".*\.bin$",
            r".*clip.*\.bin$",
            r".*quality.*\.pth$",
            r".*aesthetic.*\.pth$"
        ],
        confidence_threshold=0.05
    )
}

def get_step_model_request(step_name: str) -> Optional[StepModelRequest]:
    """Step별 모델 요청사항 조회"""
    return STEP_MODEL_REQUESTS.get(step_name)

def get_all_patterns() -> List[str]:
    """모든 패턴 목록 반환"""
    all_patterns = []
    for request in STEP_MODEL_REQUESTS.values():
        all_patterns.extend(request.checkpoint_patterns)
    return list(set(all_patterns))
