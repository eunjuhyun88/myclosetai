#!/usr/bin/env python3
"""
🔥 MyCloset AI - Cloth Segmentation Models Package
==================================================

🎯 정확한 구조 구분으로 완벽한 모듈화
=====================================

📁 models/ 폴더: 추론용 신경망 구조 (체크포인트 없이도 동작)
📁 checkpoints/ 폴더: 사전 훈련된 가중치 로딩 및 매핑
📁 model_loader.py: 두 가지를 조합하여 최적의 모델 제공

✅ 8개 Cloth Segmentation 모델 완벽 통합
✅ 앙상블 시스템
✅ 고품질 후처리
✅ 메모리 최적화
✅ 체크포인트 선택적 사용
"""

# 기본 imports
import logging
import os
import sys
from pathlib import Path

# 로거 설정
logger = logging.getLogger(__name__)

# 현재 디렉토리 경로
current_dir = Path(__file__).parent

# config 모듈 import
try:
    from .config import *
    CONFIG_AVAILABLE = True
    logger.info("✅ config 모듈 import 성공")
except ImportError as e:
    CONFIG_AVAILABLE = False
    logger.warning(f"⚠️ config 모듈 import 실패: {e}")

# 지원하는 모델들
SUPPORTED_MODELS = [
    "u2net",
    "sam",
    "deeplabv3plus",
    "hrnet",
    "pspnet",
    "segnet",
    "unetplusplus",
    "attentionunet"
]

# 지원하는 앙상블 방법들
ENSEMBLE_METHODS = [
    "voting",
    "weighted",
    "quality",
    "simple_average"
]

# 사용 가능한 클래스들 (미리 정의)
__all__ = [
    "SUPPORTED_MODELS",
    "ENSEMBLE_METHODS"
]

# ClothSegmentationStep 클래스 import
try:
    from .cloth_segmentation_step import ClothSegmentationStep
    CLOTH_SEGMENTATION_STEP_AVAILABLE = True
    logger.info("✅ ClothSegmentationStep 클래스 import 성공")
    
    # __all__에 추가
    __all__.extend([
        "ClothSegmentationStep",
        "CLOTH_SEGMENTATION_STEP_AVAILABLE"
    ])
    
except ImportError as e:
    CLOTH_SEGMENTATION_STEP_AVAILABLE = False
    logger.error(f"❌ ClothSegmentationStep import 실패: {e}")
    raise ImportError("ClothSegmentationStep을 import할 수 없습니다.")

# 모델 로더 import
try:
    from .cloth_segmentation_model_loader import ClothSegmentationModelLoader
    MODEL_LOADER_AVAILABLE = True
    logger.info("✅ ClothSegmentationModelLoader import 성공")
except ImportError as e:
    MODEL_LOADER_AVAILABLE = False
    logger.error(f"❌ ClothSegmentationModelLoader import 실패: {e}")
    raise ImportError("ClothSegmentationModelLoader을 import할 수 없습니다.")

# 앙상블 시스템 import
try:
    from .ensemble.hybrid_ensemble import ClothSegmentationEnsembleSystem
    ENSEMBLE_SYSTEM_AVAILABLE = True
    logger.info("✅ ClothSegmentationEnsembleSystem import 성공")
except ImportError as e:
    ENSEMBLE_SYSTEM_AVAILABLE = False
    logger.error(f"❌ ClothSegmentationEnsembleSystem import 실패: {e}")
    raise ImportError("ClothSegmentationEnsembleSystem을 import할 수 없습니다.")

# 후처리 시스템 import
try:
    from .postprocessing.postprocessor import ClothSegmentationPostprocessor
    POSTPROCESSOR_AVAILABLE = True
    logger.info("✅ ClothSegmentationPostprocessor import 성공")
except ImportError as e:
    POSTPROCESSOR_AVAILABLE = False
    logger.error(f"❌ ClothSegmentationPostprocessor import 실패: {e}")
    raise ImportError("ClothSegmentationPostprocessor을 import할 수 없습니다.")

# 품질 평가 시스템 import
try:
    from .utils.quality_assessment import ClothSegmentationQualityAssessment
    QUALITY_ASSESSMENT_AVAILABLE = True
    logger.info("✅ ClothSegmentationQualityAssessment import 성공")
except ImportError as e:
    QUALITY_ASSESSMENT_AVAILABLE = False
    logger.error(f"❌ ClothSegmentationQualityAssessment import 실패: {e}")
    raise ImportError("ClothSegmentationQualityAssessment을 import할 수 없습니다.")

# 고급 모듈들 import
try:
    from .models.boundary_refinement import BoundaryRefinementNetwork
    from .models.feature_pyramid_network import FeaturePyramidNetwork
    from .models.iterative_refinement import IterativeRefinementModule
    ADVANCED_MODULES_AVAILABLE = True
    logger.info("✅ 고급 모듈들 import 성공")
except ImportError as e:
    ADVANCED_MODULES_AVAILABLE = False
    logger.warning(f"⚠️ 고급 모듈들 import 실패: {e}")

# 향상된 모델들 import
try:
    from .cloth_segmentation_enhanced_models import (
        EnhancedU2NetModel, EnhancedSAMModel, EnhancedDeepLabV3PlusModel
    )
    ENHANCED_MODELS_AVAILABLE = True
    logger.info("✅ 향상된 모델들 import 성공")
except ImportError as e:
    ENHANCED_MODELS_AVAILABLE = False
    logger.warning(f"⚠️ 향상된 모델들 import 실패: {e}")

# 패키지 정보
__version__ = "1.0.0"
__author__ = "MyCloset AI Team"
__description__ = "Cloth Segmentation Models Package"

# 사용 가능한 클래스들
__all__ = [
    "ClothSegmentationStep",
    "ClothSegmentationModelLoader", 
    "ClothSegmentationEnsembleSystem",
    "ClothSegmentationPostprocessor",
    "ClothSegmentationQualityAssessment",
    "BoundaryRefinementNetwork",
    "FeaturePyramidNetwork",
    "IterativeRefinementModule",
    "EnhancedU2NetModel",
    "EnhancedSAMModel",
    "EnhancedDeepLabV3PlusModel",
    "SUPPORTED_MODELS",
    "ENSEMBLE_METHODS"
]

# config에서 정의된 클래스들도 추가
if CONFIG_AVAILABLE:
    try:
        from .config import EnhancedClothSegmentationConfig, ClothSegmentationModel, QualityLevel, CLOTHING_TYPES, VISUALIZATION_COLORS
        __all__.extend([
            "EnhancedClothSegmentationConfig",
            "ClothSegmentationModel", 
            "QualityLevel",
            "CLOTHING_TYPES",
            "VISUALIZATION_COLORS"
        ])
    except ImportError as e:
        logger.error(f"❌ config 클래스들 import 실패: {e}")

# 패키지 초기화 완료
logger.info(f"✅ Cloth Segmentation Models Package 초기화 완료 (버전: {__version__})")
logger.info(f"✅ 지원하는 모델: {len(SUPPORTED_MODELS)}개")
logger.info(f"✅ 앙상블 방법: {len(ENSEMBLE_METHODS)}개")
logger.info(f"✅ ClothSegmentationStep: {'✅' if CLOTH_SEGMENTATION_STEP_AVAILABLE else '❌'}")
logger.info(f"✅ 모델 로더: {'✅' if MODEL_LOADER_AVAILABLE else '❌'}")
logger.info(f"✅ 앙상블 시스템: {'✅' if ENSEMBLE_SYSTEM_AVAILABLE else '❌'}")
logger.info(f"✅ 후처리 시스템: {'✅' if POSTPROCESSOR_AVAILABLE else '❌'}")
logger.info(f"✅ 품질 평가: {'✅' if QUALITY_ASSESSMENT_AVAILABLE else '❌'}")
logger.info(f"✅ 고급 모듈들: {'✅' if ADVANCED_MODULES_AVAILABLE else '❌'}")
logger.info(f"✅ 향상된 모델들: {'✅' if ENHANCED_MODELS_AVAILABLE else '❌'}")
logger.info(f"✅ config 모듈: {'✅' if CONFIG_AVAILABLE else '❌'}")

# 새로운 구조 정보
logger.info("🎯 새로운 구조 구분:")
logger.info("  📁 models/: 추론용 신경망 구조 (체크포인트 없이도 동작)")
logger.info("  📁 checkpoints/: 사전 훈련된 가중치 로딩 및 매핑")
logger.info("  📁 model_loader.py: 통합 관리 및 최적 모델 제공")
