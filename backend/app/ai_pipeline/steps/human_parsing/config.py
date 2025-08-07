"""
🔥 Human Parsing 설정 클래스들
==========================

Human Parsing 관련 설정 클래스들을 포함합니다.

주요 클래스:
- HumanParsingModel (Enum): 모델 타입
- QualityLevel (Enum): 품질 레벨
- EnhancedHumanParsingConfig: 메인 설정 클래스

Author: MyCloset AI Team
Date: 2025-08-07
Version: 1.0
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, List


class HumanParsingModel(Enum):
    """인체 파싱 모델 타입 - 상용화 수준 확장"""
    GRAPHONOMY = "graphonomy"
    U2NET = "u2net"
    HRNET = "hrnet"
    DEEPLABV3PLUS = "deeplabv3plus"
    MASK2FORMER = "mask2former"
    SWIN_TRANSFORMER = "swin_transformer"
    ENSEMBLE = "ensemble"  # 다중 모델 앙상블
    MOCK = "mock"


class QualityLevel(Enum):
    """품질 레벨"""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"


@dataclass
class EnhancedHumanParsingConfig:
    """강화된 Human Parsing 설정 (원본 프로젝트 완전 반영)"""
    method: HumanParsingModel = HumanParsingModel.GRAPHONOMY
    quality_level: QualityLevel = QualityLevel.HIGH
    input_size: Tuple[int, int] = (512, 512)
    
    # 전처리 설정
    enable_quality_assessment: bool = True
    enable_lighting_normalization: bool = True
    enable_color_correction: bool = True
    enable_roi_detection: bool = True
    enable_background_analysis: bool = True
    
    # 인체 분류 설정
    enable_body_classification: bool = True
    classification_confidence_threshold: float = 0.8
    
    # Graphonomy 프롬프트 설정
    enable_advanced_prompts: bool = True
    use_box_prompts: bool = True
    use_mask_prompts: bool = True
    enable_iterative_refinement: bool = True
    max_refinement_iterations: int = 3
    
    # 후처리 설정 (고급 알고리즘)
    enable_crf_postprocessing: bool = True
    enable_edge_refinement: bool = True
    enable_hole_filling: bool = True
    enable_multiscale_processing: bool = True
    
    # 품질 검증 설정
    enable_quality_validation: bool = True
    quality_threshold: float = 0.7
    enable_auto_retry: bool = True
    max_retry_attempts: int = 3
    
    # 기본 설정
    enable_visualization: bool = True
    use_fp16: bool = True
    confidence_threshold: float = 0.7
    remove_noise: bool = True
    overlay_opacity: float = 0.6
    
    # 자동 전처리 설정
    auto_preprocessing: bool = True
    
    # 데이터 검증 설정
    strict_data_validation: bool = True
    
    # 자동 후처리 설정
    auto_postprocessing: bool = True
    
    # 🔥 M3 Max 최적화 앙상블 시스템 설정
    enable_ensemble: bool = True
    ensemble_models: List[str] = field(default_factory=lambda: ['graphonomy', 'hrnet', 'deeplabv3plus'])
    ensemble_method: str = 'simple_weighted_average'  # 단순 가중 평균
    ensemble_confidence_threshold: float = 0.8
    enable_uncertainty_quantification: bool = True
    enable_confidence_calibration: bool = True
    ensemble_quality_threshold: float = 0.7
    
    # 🔥 M3 Max 메모리 최적화 설정 (128GB 활용)
    memory_optimization_level: str = 'ultra'  # 'standard', 'high', 'ultra'
    max_memory_usage_gb: int = 100  # 128GB 중 100GB 사용
    enable_memory_pooling: bool = True
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision: bool = True
    enable_dynamic_batching: bool = True
    max_batch_size: int = 4
    enable_memory_monitoring: bool = True
    
    # 🔥 고해상도 처리 시스템 설정 (M3 Max 최적화)
    enable_high_resolution: bool = True
    adaptive_resolution: bool = True
    min_resolution: int = 512
    max_resolution: int = 4096  # M3 Max에서 더 높은 해상도 지원
    target_resolution: int = 2048  # 2K 해상도로 향상
    resolution_quality_threshold: float = 0.85
    enable_super_resolution: bool = True
    enable_noise_reduction: bool = True
    enable_lighting_normalization: bool = True
    enable_color_correction: bool = True
    
    # 🔥 특수 케이스 처리 시스템 설정 (새로 추가)
    enable_special_case_handling: bool = True
    enable_transparent_clothing: bool = True
    enable_layered_clothing: bool = True
    enable_complex_patterns: bool = True
    enable_reflective_materials: bool = True
    enable_oversized_clothing: bool = True
    enable_tight_clothing: bool = True
    special_case_confidence_threshold: float = 0.75
    enable_adaptive_thresholding: bool = True
    enable_context_aware_parsing: bool = True


# 20개 인체 부위 정의 (Graphonomy 표준)
BODY_PARTS = {
    0: 'background',    1: 'hat',          2: 'hair',
    3: 'glove',         4: 'sunglasses',   5: 'upper_clothes',
    6: 'dress',         7: 'coat',         8: 'socks',
    9: 'pants',         10: 'torso_skin',  11: 'scarf',
    12: 'skirt',        13: 'face',        14: 'left_arm',
    15: 'right_arm',    16: 'left_leg',    17: 'right_leg',
    18: 'left_shoe',    19: 'right_shoe'
}

# 시각화 색상 (20개 클래스)
VISUALIZATION_COLORS = {
    0: (0, 0, 0),           # Background
    1: (255, 0, 0),         # Hat
    2: (255, 165, 0),       # Hair
    3: (255, 255, 0),       # Glove
    4: (0, 255, 0),         # Sunglasses
    5: (0, 255, 255),       # Upper-clothes
    6: (0, 0, 255),         # Dress
    7: (255, 0, 255),       # Coat
    8: (128, 0, 128),       # Socks
    9: (255, 192, 203),     # Pants
    10: (255, 218, 185),    # Torso-skin
    11: (210, 180, 140),    # Scarf
    12: (255, 20, 147),     # Skirt
    13: (255, 228, 196),    # Face
    14: (255, 160, 122),    # Left-arm
    15: (255, 182, 193),    # Right-arm
    16: (173, 216, 230),    # Left-leg
    17: (144, 238, 144),    # Right-leg
    18: (139, 69, 19),      # Left-shoe
    19: (160, 82, 45)       # Right-shoe
}
