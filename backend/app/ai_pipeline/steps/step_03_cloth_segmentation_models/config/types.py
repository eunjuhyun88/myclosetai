"""
Cloth Segmentation 타입 정의
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import numpy as np


class SegmentationMethod(Enum):
    """세그멘테이션 방법"""
    U2NET_CLOTH = "u2net_cloth"         # U2Net 의류 특화 (168.1MB) - 우선순위 1 (M3 Max 안전)
    SAM_HUGE = "sam_huge"               # SAM ViT-Huge (2445.7MB) - 우선순위 2 (메모리 여유시)
    DEEPLABV3_PLUS = "deeplabv3_plus"   # DeepLabV3+ (233.3MB) - 우선순위 3 (나중에)
    MASK_RCNN = "mask_rcnn"             # Mask R-CNN (폴백)
    HYBRID_AI = "hybrid_ai"             # 하이브리드 앙상블


class ClothCategory(Enum):
    """의류 카테고리 (다중 클래스)"""
    BACKGROUND = 0
    SHIRT = 1           # 셔츠/블라우스
    T_SHIRT = 2         # 티셔츠
    SWEATER = 3         # 스웨터/니트
    HOODIE = 4          # 후드티
    JACKET = 5          # 재킷/아우터
    COAT = 6            # 코트
    DRESS = 7           # 원피스
    SKIRT = 8           # 스커트
    PANTS = 9           # 바지
    JEANS = 10          # 청바지
    SHORTS = 11         # 반바지
    SHOES = 12          # 신발
    BOOTS = 13          # 부츠
    SNEAKERS = 14       # 운동화
    BAG = 15            # 가방
    HAT = 16            # 모자
    GLASSES = 17        # 안경
    SCARF = 18          # 스카프
    BELT = 19           # 벨트
    ACCESSORY = 20      # 액세서리


class QualityLevel(Enum):
    """품질 레벨"""
    FAST = "fast"           # 빠른 처리
    BALANCED = "balanced"   # 균형
    HIGH = "high"          # 고품질
    ULTRA = "ultra"        # 최고품질


class SegmentationModel(Enum):
    """의류 분할 모델 타입"""
    SAM = "sam"
    U2NET = "u2net"
    DEEPLABV3 = "deeplabv3"
    MASK_RCNN = "mask_rcnn"


class SegmentationQuality(Enum):
    """분할 품질 등급"""
    EXCELLENT = "excellent"     # 90-100점
    GOOD = "good"              # 75-89점  
    ACCEPTABLE = "acceptable"   # 60-74점
    POOR = "poor"              # 40-59점
    VERY_POOR = "very_poor"    # 0-39점


@dataclass
class ClothSegmentationConfig:
    """의류 세그멘테이션 설정"""
    method: SegmentationMethod = SegmentationMethod.U2NET_CLOTH  # M3 Max 안전 모드
    quality_level: QualityLevel = QualityLevel.HIGH
    input_size: Tuple[int, int] = (512, 512)
    
    # 전처리 설정
    enable_quality_assessment: bool = True
    enable_lighting_normalization: bool = True
    enable_color_correction: bool = True
    
    # 의류 분류 설정
    enable_clothing_classification: bool = True
    classification_confidence_threshold: float = 0.8
    
    # 후처리 설정
    enable_crf_postprocessing: bool = True  # 🔥 CRF 후처리 복원
    enable_edge_refinement: bool = True
    enable_hole_filling: bool = True
    enable_multiscale_processing: bool = True  # 🔥 멀티스케일 처리 복원
    
    # 품질 검증 설정
    enable_quality_validation: bool = True
    quality_threshold: float = 0.7
    enable_auto_retry: bool = True
    max_retry_attempts: int = 3
    
    # 기본 설정
    confidence_threshold: float = 0.5
    enable_visualization: bool = True
    
    # 자동 전처리 설정
    auto_preprocessing: bool = True
    
    # 자동 후처리 설정
    auto_postprocessing: bool = True
    
    # 데이터 검증 설정
    strict_data_validation: bool = True


@dataclass
class SegmentationResult:
    """의류 분할 결과"""
    masks: List[np.ndarray] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    bounding_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    segmentation_quality: SegmentationQuality = SegmentationQuality.POOR
    overall_confidence: float = 0.0
    processing_time: float = 0.0
    model_used: str = ""
    
    # 고급 분석 결과
    edge_quality: float = 0.0
    mask_completeness: float = 0.0
    segmentation_metrics: Dict[str, Any] = field(default_factory=dict)
