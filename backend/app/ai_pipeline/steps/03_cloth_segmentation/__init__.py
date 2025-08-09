#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - 통합 패키지
=====================================================================

의류 세그멘테이션을 위한 모든 기능들 (논리적 통합 완료)

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

# 🔥 Base classes
from .base import BaseStepMixin

# 🔥 Configuration
from .config import (
    SegmentationMethod,
    ClothCategory,
    QualityLevel,
    ClothSegmentationConfig
)

# 🔥 Core functionality
from .core import (
    SegmentationCore,
    EnsembleCore
)

# 🔥 Models
from .models import (
    RealDeepLabV3PlusModel,
    RealU2NETModel,
    RealSAMModel,
    MultiHeadSelfAttention,
    PositionalEncoding2D,
    SelfCorrectionModule
)

# 🔥 Ensemble functions
from .ensemble import (
    _run_hybrid_ensemble_sync,
    _combine_ensemble_results,
    _calculate_adaptive_threshold,
    _apply_ensemble_postprocessing
)

# 🔥 Postprocessing functions
from .postprocessing import (
    _fill_holes_and_remove_noise_advanced,
    _evaluate_segmentation_quality,
    _create_segmentation_visualizations,
    _assess_image_quality,
    _normalize_lighting,
    _correct_colors
)

# 🔥 Utils functions
from .utils import (
    _extract_cloth_features,
    _calculate_centroid,
    _calculate_bounding_box,
    _get_cloth_bounding_boxes,
    _get_cloth_centroids,
    _get_cloth_areas,
    _get_cloth_contours_dict,
    _detect_cloth_categories
)

# 🔥 Processors
from .processors import (
    HighResolutionProcessor,
    SpecialCaseProcessor,
    AdvancedPostProcessor,
    QualityEnhancer
)

# 🔥 Services
from .services import (
    ModelLoaderService,
    MemoryService,
    ValidationService,
    TestingService
)

# 🔥 Step classes
try:
    from .step_modularized import (
        ClothSegmentationStepModularized,
        create_cloth_segmentation_step_modularized,
        create_m3_max_segmentation_step_modularized
    )
except ImportError:
    ClothSegmentationStepModularized = None
    create_cloth_segmentation_step_modularized = None
    create_m3_max_segmentation_step_modularized = None

try:
    from .step import (
        ClothSegmentationStep,
        create_cloth_segmentation_step,
        create_m3_max_segmentation_step
    )
except ImportError:
    ClothSegmentationStep = None
    create_cloth_segmentation_step = None
    create_m3_max_segmentation_step = None

__all__ = [
    # 🔥 Base classes
    'BaseStepMixin',
    
    # 🔥 Configuration
    'SegmentationMethod',
    'ClothCategory',
    'QualityLevel',
    'ClothSegmentationConfig',
    
    # 🔥 Core functionality
    'SegmentationCore',
    'EnsembleCore',
    
    # 🔥 Models
    'RealDeepLabV3PlusModel',
    'RealU2NETModel',
    'RealSAMModel',
    'MultiHeadSelfAttention',
    'PositionalEncoding2D',
    'SelfCorrectionModule',
    
    # 🔥 Ensemble functions
    '_run_hybrid_ensemble_sync',
    '_combine_ensemble_results',
    '_calculate_adaptive_threshold',
    '_apply_ensemble_postprocessing',
    
    # 🔥 Postprocessing functions
    '_fill_holes_and_remove_noise_advanced',
    '_evaluate_segmentation_quality',
    '_create_segmentation_visualizations',
    '_assess_image_quality',
    '_normalize_lighting',
    '_correct_colors',
    
    # 🔥 Utils functions
    '_extract_cloth_features',
    '_calculate_centroid',
    '_calculate_bounding_box',
    '_get_cloth_bounding_boxes',
    '_get_cloth_centroids',
    '_get_cloth_areas',
    '_get_cloth_contours_dict',
    '_detect_cloth_categories',
    
    # 🔥 Processors
    'HighResolutionProcessor',
    'SpecialCaseProcessor',
    'AdvancedPostProcessor',
    'QualityEnhancer',
    
    # 🔥 Services
    'ModelLoaderService',
    'MemoryService',
    'ValidationService',
    'TestingService',
    
    # 🔥 Step classes
    'ClothSegmentationStep',
    'ClothSegmentationStepModularized',
    'create_cloth_segmentation_step',
    'create_m3_max_segmentation_step',
    'create_cloth_segmentation_step_modularized',
    'create_m3_max_segmentation_step_modularized'
]
