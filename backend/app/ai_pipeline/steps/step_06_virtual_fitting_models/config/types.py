"""
Virtual Fitting 타입 정의
가상 피팅에 필요한 모든 데이터 타입과 구조체를 정의합니다.
"""

from typing import (
    Dict, List, Tuple, Optional, Union, Any, Callable,
    Protocol, runtime_checkable
)
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

# 이미지 타입 정의
ImageType = Union[np.ndarray, Image.Image, torch.Tensor]
ImagePath = Union[str, Path]
ImageSize = Tuple[int, int]
ImageChannel = int

# 텐서 타입 정의
TensorType = Union[torch.Tensor, np.ndarray]
TensorShape = Tuple[int, ...]
TensorDevice = Union[str, torch.device]

# 모델 타입 정의
ModelType = Union[nn.Module, Any]
CheckpointPath = Union[str, Path]
ModelConfig = Dict[str, Any]

# 품질 평가 타입 정의
QualityScore = float
ConfidenceScore = float
FittingScore = float

class FittingQuality(Enum):
    """피팅 품질 레벨"""
    LOW = "low"
    BALANCED = "balanced"
    HIGH = "high"
    ULTRA = "ultra"

class FittingModel(Enum):
    """사용 가능한 피팅 모델"""
    HR_VITON = "hr_viton"
    OOTD = "ootd"
    VITON_HD = "viton_hd"
    HYBRID = "hybrid"

class BlendingMode(Enum):
    """블렌딩 모드"""
    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    SOFT_LIGHT = "soft_light"
    HARD_LIGHT = "hard_light"
    COLOR_DODGE = "color_dodge"
    COLOR_BURN = "color_burn"
    DARKEN = "darken"
    LIGHTEN = "lighten"
    DIFFERENCE = "difference"
    EXCLUSION = "exclusion"

class WarpingMethod(Enum):
    """워핑 방법"""
    AFFINE = "affine"
    PERSPECTIVE = "perspective"
    THIN_PLATE_SPLINE = "thin_plate_spline"
    OPTICAL_FLOW = "optical_flow"
    DEFORMABLE_CONVOLUTION = "deformable_convolution"

@dataclass
class ImageInfo:
    """이미지 정보"""
    width: int
    height: int
    channels: int
    format: str
    mode: str
    size_bytes: int
    dpi: Tuple[int, int] = (72, 72)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FittingParameters:
    """피팅 파라미터"""
    quality_level: FittingQuality = FittingQuality.HIGH
    model_type: FittingModel = FittingModel.HYBRID
    blending_mode: BlendingMode = BlendingMode.NORMAL
    warping_method: WarpingMethod = WarpingMethod.THIN_PLATE_SPLINE
    blending_alpha: float = 0.8
    warping_strength: float = 1.0
    confidence_threshold: float = 0.7
    enable_post_processing: bool = True
    enable_quality_assessment: bool = True
    save_intermediate: bool = False
    debug_mode: bool = False

@dataclass
class BodyMeasurements:
    """신체 측정값"""
    height: float  # cm
    weight: float  # kg
    chest: Optional[float] = None  # cm
    waist: Optional[float] = None  # cm
    hips: Optional[float] = None  # cm
    shoulder: Optional[float] = None  # cm
    arm_length: Optional[float] = None  # cm
    inseam: Optional[float] = None  # cm
    neck: Optional[float] = None  # cm
    bicep: Optional[float] = None  # cm
    wrist: Optional[float] = None  # cm
    ankle: Optional[float] = None  # cm

@dataclass
class ClothingInfo:
    """의류 정보"""
    category: str
    subcategory: str
    style: str
    material: str
    color: str
    pattern: str
    size: str
    brand: str
    season: str
    occasion: str
    measurements: Dict[str, float] = field(default_factory=dict)

@dataclass
class PoseKeypoints:
    """포즈 키포인트"""
    keypoints: np.ndarray  # (N, 3) - x, y, confidence
    skeleton: List[Tuple[int, int]]  # 연결된 키포인트 쌍
    confidence_scores: np.ndarray  # (N,) - 각 키포인트의 신뢰도
    num_keypoints: int = 0
    
    def __post_init__(self):
        if self.num_keypoints == 0:
            self.num_keypoints = len(self.keypoints)

@dataclass
class HumanParsing:
    """인간 파싱 결과"""
    segmentation_map: np.ndarray  # 세그멘테이션 맵
    body_parts: Dict[str, np.ndarray]  # 신체 부위별 마스크
    confidence_map: np.ndarray  # 신뢰도 맵
    num_classes: int = 0
    
    def __post_init__(self):
        if self.num_classes == 0:
            self.num_classes = len(np.unique(self.segmentation_map))

@dataclass
class ClothSegmentation:
    """의류 세그멘테이션 결과"""
    segmentation_map: np.ndarray  # 의류 세그멘테이션 맵
    clothing_parts: Dict[str, np.ndarray]  # 의류 부위별 마스크
    confidence_map: np.ndarray  # 신뢰도 맵
    boundaries: np.ndarray  # 경계선
    num_classes: int = 0
    
    def __post_init__(self):
        if self.num_classes == 0:
            self.num_classes = len(np.unique(self.segmentation_map))

@dataclass
class GeometricMatching:
    """기하학적 매칭 결과"""
    transformation_matrix: np.ndarray  # 변환 행렬
    matching_points: List[Tuple[np.ndarray, np.ndarray]]  # 매칭된 점들
    confidence_score: float  # 매칭 신뢰도
    error_metric: float  # 오차 메트릭
    method: str = "ransac"

@dataclass
class ClothWarping:
    """의류 워핑 결과"""
    warped_cloth: np.ndarray  # 워핑된 의류
    warping_field: np.ndarray  # 워핑 필드
    confidence_map: np.ndarray  # 워핑 신뢰도
    distortion_metric: float  # 왜곡 메트릭
    method: WarpingMethod = WarpingMethod.THIN_PLATE_SPLINE

@dataclass
class FittingResult:
    """피팅 결과"""
    result_image: np.ndarray  # 최종 피팅 결과 이미지
    intermediate_results: Dict[str, Any]  # 중간 결과들
    quality_metrics: Dict[str, float]  # 품질 메트릭
    processing_time: float  # 처리 시간
    confidence_score: float  # 전체 신뢰도
    success: bool = True
    error_message: Optional[str] = None

@dataclass
class QualityMetrics:
    """품질 평가 메트릭"""
    ssim_score: float  # SSIM 점수
    psnr_score: float  # PSNR 점수
    lpips_score: float  # LPIPS 점수
    fid_score: float  # FID 점수
    perceptual_quality: float  # 지각적 품질
    overall_quality: float  # 전체 품질
    color_consistency: float  # 색상 일관성
    texture_preservation: float  # 질감 보존
    edge_quality: float  # 엣지 품질
    blending_quality: float  # 블렌딩 품질

@dataclass
class ModelOutput:
    """모델 출력"""
    output_tensor: torch.Tensor
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessingStats:
    """처리 통계"""
    total_time: float
    model_inference_time: float
    preprocessing_time: float
    postprocessing_time: float
    memory_usage: float
    gpu_utilization: float
    cpu_utilization: float

# 프로토콜 정의
@runtime_checkable
class FittingModelProtocol(Protocol):
    """피팅 모델 프로토콜"""
    def forward(self, *args, **kwargs) -> torch.Tensor:
        ...
    
    def predict(self, *args, **kwargs) -> ModelOutput:
        ...
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        ...
    
    def to_device(self, device: torch.device) -> None:
        ...

@runtime_checkable
class PreprocessorProtocol(Protocol):
    """전처리기 프로토콜"""
    def preprocess(self, image: ImageType, **kwargs) -> torch.Tensor:
        ...
    
    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        ...
    
    def resize(self, image: ImageType, target_size: ImageSize) -> ImageType:
        ...

@runtime_checkable
class PostprocessorProtocol(Protocol):
    """후처리기 프로토콜"""
    def postprocess(self, tensor: torch.Tensor, **kwargs) -> np.ndarray:
        ...
    
    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        ...
    
    def enhance_quality(self, image: np.ndarray) -> np.ndarray:
        ...

@runtime_checkable
class QualityAssessorProtocol(Protocol):
    """품질 평가기 프로토콜"""
    def assess_quality(self, result: np.ndarray, reference: np.ndarray) -> QualityMetrics:
        ...
    
    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        ...
    
    def calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        ...
    
    def calculate_lpips(self, img1: np.ndarray, img2: np.ndarray) -> float:
        ...

# 타입 검증 함수들
def validate_image_type(obj: Any) -> bool:
    """이미지 타입 검증"""
    return isinstance(obj, (np.ndarray, Image.Image, torch.Tensor))

def validate_tensor_type(obj: Any) -> bool:
    """텐서 타입 검증"""
    return isinstance(obj, (torch.Tensor, np.ndarray))

def validate_quality_score(score: float) -> bool:
    """품질 점수 검증 (0.0 ~ 1.0)"""
    return 0.0 <= score <= 1.0

def validate_confidence_score(score: float) -> bool:
    """신뢰도 점수 검증 (0.0 ~ 1.0)"""
    return 0.0 <= score <= 1.0

def validate_image_size(size: ImageSize) -> bool:
    """이미지 크기 검증"""
    return len(size) == 2 and all(isinstance(dim, int) and dim > 0 for dim in size)

def validate_measurements(measurements: BodyMeasurements) -> bool:
    """측정값 검증"""
    return (measurements.height > 0 and measurements.weight > 0 and
            (measurements.chest is None or measurements.chest > 0) and
            (measurements.waist is None or measurements.waist > 0) and
            (measurements.hips is None or measurements.hips > 0))

# 타입 변환 함수들
def to_numpy(image: ImageType) -> np.ndarray:
    """이미지를 numpy 배열로 변환"""
    if isinstance(image, np.ndarray):
        return image
    elif isinstance(image, Image.Image):
        return np.array(image)
    elif isinstance(image, torch.Tensor):
        return image.detach().cpu().numpy()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

def to_pil(image: ImageType) -> Image.Image:
    """이미지를 PIL Image로 변환"""
    if isinstance(image, Image.Image):
        return image
    elif isinstance(image, np.ndarray):
        return Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        return Image.fromarray(image.detach().cpu().numpy())
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

def to_tensor(image: ImageType, device: Optional[torch.device] = None) -> torch.Tensor:
    """이미지를 torch tensor로 변환"""
    if isinstance(image, torch.Tensor):
        return image.to(device) if device else image
    elif isinstance(image, np.ndarray):
        tensor = torch.from_numpy(image)
        return tensor.to(device) if device else tensor
    elif isinstance(image, Image.Image):
        tensor = torch.from_numpy(np.array(image))
        return tensor.to(device) if device else tensor
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

# 유틸리티 타입들
CallbackType = Callable[[str, float], None]
ProgressCallback = Callable[[str, float, Dict[str, Any]], None]
ErrorCallback = Callable[[str, Exception], None]
LogCallback = Callable[[str, str, Dict[str, Any]], None]

# 설정 타입들
ConfigDict = Dict[str, Any]
Hyperparameters = Dict[str, Union[int, float, str, bool]]
ModelHyperparameters = Dict[str, Union[int, float, str, bool, List[int], List[float]]]
