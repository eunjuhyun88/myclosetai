"""
포즈 추정 설정 및 타입 정의
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from PIL import Image


class PoseModel(Enum):
    """포즈 추정 모델 타입"""
    MEDIAPIPE = "mediapipe"
    OPENPOSE = "openpose"
    YOLOV8_POSE = "yolov8_pose"
    HRNET = "hrnet"
    DIFFUSION_POSE = "diffusion_pose"


class PoseQuality(Enum):
    """포즈 품질 등급"""
    EXCELLENT = "excellent"     # 90-100점
    GOOD = "good"              # 75-89점  
    ACCEPTABLE = "acceptable"   # 60-74점
    POOR = "poor"              # 40-59점
    VERY_POOR = "very_poor"    # 0-39점


@dataclass
class EnhancedPoseConfig:
    """강화된 Pose Estimation 설정 (앙상블 시스템 포함)"""
    method: PoseModel = PoseModel.HRNET
    quality_level: PoseQuality = PoseQuality.EXCELLENT
    input_size: Tuple[int, int] = (512, 512)
    
    # 앙상블 시스템 설정
    enable_ensemble: bool = True
    ensemble_models: List[str] = field(default_factory=lambda: ['hrnet', 'yolov8', 'mediapipe', 'openpose'])
    ensemble_method: str = 'weighted_average'  # 'simple_average', 'weighted_average', 'confidence_weighted'
    ensemble_confidence_threshold: float = 0.8
    enable_uncertainty_quantification: bool = True
    enable_confidence_calibration: bool = True
    ensemble_quality_threshold: float = 0.7
    
    # 고급 처리 설정
    enable_subpixel_accuracy: bool = True
    enable_joint_angle_calculation: bool = True
    enable_body_proportion_analysis: bool = True
    enable_pose_quality_assessment: bool = True
    enable_skeleton_structure_analysis: bool = True
    enable_virtual_fitting_optimization: bool = True
    
    # 성능 설정
    use_fp16: bool = True
    confidence_threshold: float = 0.7
    enable_visualization: bool = True
    auto_preprocessing: bool = True
    strict_data_validation: bool = True
    auto_postprocessing: bool = True


@dataclass
class PoseResult:
    """포즈 추정 결과"""
    keypoints: List[List[float]] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    joint_angles: Dict[str, float] = field(default_factory=dict)
    body_proportions: Dict[str, float] = field(default_factory=dict)
    pose_quality: PoseQuality = PoseQuality.POOR
    overall_confidence: float = 0.0
    processing_time: float = 0.0
    model_used: str = ""
    subpixel_accuracy: bool = False
    
    # 고급 분석 결과
    keypoints_with_uncertainty: List[Dict[str, Any]] = field(default_factory=list)
    advanced_body_metrics: Dict[str, Any] = field(default_factory=dict)
    skeleton_structure: Dict[str, Any] = field(default_factory=dict)
    ensemble_info: Dict[str, Any] = field(default_factory=dict)


# COCO 17 키포인트 인덱스 정의
COCO_17_KEYPOINTS = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# OpenPose 18 키포인트 인덱스 정의
OPENPOSE_18_KEYPOINTS = {
    'nose': 0,
    'neck': 1,
    'right_shoulder': 2,
    'right_elbow': 3,
    'right_wrist': 4,
    'left_shoulder': 5,
    'left_elbow': 6,
    'left_wrist': 7,
    'right_hip': 8,
    'right_knee': 9,
    'right_ankle': 10,
    'left_hip': 11,
    'left_knee': 12,
    'left_ankle': 13,
    'right_eye': 14,
    'left_eye': 15,
    'right_ear': 16,
    'left_ear': 17
}
