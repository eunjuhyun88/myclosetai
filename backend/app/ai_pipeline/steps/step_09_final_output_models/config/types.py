#!/usr/bin/env python3
"""
🔥 Final Output 타입 정의 파일
================================================================================

✅ 데이터 구조 정의
✅ 타입 힌트
✅ 결과 타입
✅ 메트릭 타입

Author: MyCloset AI Team
Date: 2025-08-13
Version: 1.0
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

# PyTorch 텐서 타입 (선택적)
try:
    import torch
    TensorType = Union[torch.Tensor, np.ndarray]
except ImportError:
    TensorType = np.ndarray

# ==============================================
# 🔥 기본 타입 정의
# ==============================================

class QualityLevel(Enum):
    """품질 수준"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"

class ConfidenceLevel(Enum):
    """신뢰도 수준"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class OutputStatus(Enum):
    """출력 상태"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    PROCESSING = "processing"

# ==============================================
# 🔥 데이터 구조 정의
# ==============================================

@dataclass
class QualityMetrics:
    """품질 메트릭"""
    psnr: float
    ssim: float
    lpips: float
    fid: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None

@dataclass
class ConfidenceMetrics:
    """신뢰도 메트릭"""
    overall_confidence: float
    model_confidence: float
    data_confidence: float
    quality_confidence: float

@dataclass
class OutputMetadata:
    """출력 메타데이터"""
    timestamp: str
    processing_time: float
    device_used: str
    model_version: str
    quality_score: float
    confidence_score: float
    output_resolution: Tuple[int, int]
    file_size: Optional[int] = None

@dataclass
class StepResult:
    """단계 결과"""
    step_name: str
    status: OutputStatus
    processing_time: float
    quality_score: float
    confidence_score: float
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

# ==============================================
# 🔥 AI 모델 출력 타입
# ==============================================

@dataclass
class TransformerOutput:
    """Transformer 출력"""
    final_image: TensorType
    confidence: float
    quality_score: float
    metadata: TensorType
    attention_weights: Optional[TensorType] = None

@dataclass
class CrossModalOutput:
    """크로스 모달 출력"""
    fused_features: TensorType
    modality_weights: Dict[str, float]
    attention_scores: Optional[TensorType] = None

@dataclass
class GeneratorOutput:
    """생성기 출력"""
    final_output: TensorType
    quality_score: float
    confidence: float
    metadata: TensorType
    cross_modal_features: Optional[TensorType] = None

# ==============================================
# 🔥 통합 출력 타입
# ==============================================

@dataclass
class IntegratedOutput:
    """통합 출력"""
    pipeline_version: str
    total_steps: int
    integration_timestamp: str
    step_results: Dict[str, StepResult]
    final_metrics: Dict[str, Any]
    quality_assessment: Dict[str, Any]
    output_summary: Dict[str, Any]

@dataclass
class FinalOutputResult:
    """최종 출력 결과"""
    step_name: str
    step_version: str
    status: OutputStatus
    integrated_output: IntegratedOutput
    ai_results: Dict[str, Any]
    processing_time: float
    device_used: str
    error: Optional[str] = None

# ==============================================
# 🔥 설정 타입
# ==============================================

@dataclass
class ModelParameters:
    """모델 파라미터"""
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    dropout: float
    output_size: int
    num_channels: int

@dataclass
class QualityParameters:
    """품질 파라미터"""
    min_threshold: float
    assessment_method: str
    auto_refinement: bool
    refinement_iterations: int
    quality_metrics: List[str]

@dataclass
class OutputParameters:
    """출력 파라미터"""
    format: str
    enable_metadata: bool
    enable_confidence: bool
    enable_quality_report: bool
    resolution: Tuple[int, int]
    compression_quality: int

# ==============================================
# 🔥 유틸리티 타입
# ==============================================

@dataclass
class ValidationResult:
    """검증 결과"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]

@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    inference_time: float
    memory_usage: float
    gpu_utilization: Optional[float] = None
    cpu_utilization: Optional[float] = None

# ==============================================
# 🔥 타입 별칭
# ==============================================

# 간단한 타입 별칭
ImageArray = np.ndarray
ImageTensor = TensorType
QualityScore = float
ConfidenceScore = float
ProcessingTime = float
DeviceName = str
ModelVersion = str
Timestamp = str

# 복합 타입 별칭
StepOutputs = Dict[str, StepResult]
QualityMetrics = Dict[str, float]
ModelConfig = Dict[str, Any]
IntegrationConfig = Dict[str, Any]
