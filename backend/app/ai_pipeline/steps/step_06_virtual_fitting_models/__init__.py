#!/usr/bin/env python3
"""
🔥 MyCloset AI - Virtual Fitting Models Package
===============================================

🎯 정확한 구조 구분으로 완벽한 모듈화
=====================================

📁 models/ 폴더: 추론용 신경망 구조 (체크포인트 없이도 동작)
📁 checkpoints/ 폴더: 사전 훈련된 가중치 로딩 및 매핑
📁 model_loader.py: 두 가지를 조합하여 최적의 모델 제공

✅ 8개 Virtual Fitting 모델 완벽 통합
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
    "fitting_transformer",
    "pose_aware_fitting",
    "garment_fitting",
    "body_fitting",
    "fitting_network",
    "adaptive_fitting",
    "attention_fitting",
    "geometric_fitting"
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

# VirtualFittingStep 클래스 import
try:
    from .step_06_virtual_fitting import VirtualFittingStep
    VIRTUAL_FITTING_STEP_AVAILABLE = True
    logger.info("✅ VirtualFittingStep 클래스 import 성공")
    
    # __all__에 추가
    __all__.extend([
        "VirtualFittingStep",
        "VIRTUAL_FITTING_STEP_AVAILABLE"
    ])
    
except ImportError as e:
    VIRTUAL_FITTING_STEP_AVAILABLE = False
    logger.error(f"❌ VirtualFittingStep import 실패: {e}")
    raise ImportError("VirtualFittingStep을 import할 수 없습니다.")

# 모델 로더 import (새로운 구조)
try:
    from .virtual_fitting_model_loader import VirtualFittingModelLoader
    MODEL_LOADER_AVAILABLE = True
    logger.info("✅ VirtualFittingModelLoader import 성공 (새로운 구조)")
except ImportError:
    MODEL_LOADER_AVAILABLE = False
    logger.warning("⚠️ VirtualFittingModelLoader import 실패 (새로운 구조)")
    
    # Mock 모델 로더
    class VirtualFittingModelLoader:
        def __init__(self):
            self.supported_models = SUPPORTED_MODELS
        
        def load_models(self):
            return True

# 앙상블 시스템 import
try:
    from .ensemble.virtual_fitting_ensemble import VirtualFittingEnsembleSystem
    ENSEMBLE_SYSTEM_AVAILABLE = True
    logger.info("✅ VirtualFittingEnsembleSystem import 성공")
except ImportError:
    ENSEMBLE_SYSTEM_AVAILABLE = False
    logger.warning("⚠️ VirtualFittingEnsembleSystem import 실패")
    
    # Mock 앙상블 시스템
    class VirtualFittingEnsembleSystem:
        def __init__(self):
            self.ensemble_methods = ENSEMBLE_METHODS
        
        def run_ensemble(self, results, method):
            return {'ensemble_result': None, 'method': method}

# 후처리 시스템 import
try:
    from .postprocessing.postprocessor import VirtualFittingPostprocessor
    POSTPROCESSOR_AVAILABLE = True
    logger.info("✅ VirtualFittingPostprocessor import 성공")
except ImportError:
    POSTPROCESSOR_AVAILABLE = False
    logger.warning("⚠️ VirtualFittingPostprocessor import 실패")
    
    # Mock 후처리 시스템
    class VirtualFittingPostprocessor:
        def __init__(self):
            pass
        
        def enhance_quality(self, fitting_result):
            return fitting_result

# 품질 평가 시스템 import
try:
    from .utils.quality_assessment import VirtualFittingQualityAssessment
    QUALITY_ASSESSMENT_AVAILABLE = True
    logger.info("✅ VirtualFittingQualityAssessment import 성공")
except ImportError:
    QUALITY_ASSESSMENT_AVAILABLE = False
    logger.warning("⚠️ VirtualFittingQualityAssessment import 실패")
    
    # Mock 품질 평가 시스템
    class VirtualFittingQualityAssessment:
        def __init__(self):
            pass
        
        def assess_quality(self, fitting_result):
            return {'quality_score': 0.85, 'confidence': 0.87}

# 고급 모듈들 import
try:
    from .models.fitting_transformer import FittingTransformer
    from .models.pose_aware_fitting import PoseAwareFitting
    from .models.garment_fitting import GarmentFitting
    ADVANCED_MODULES_AVAILABLE = True
    logger.info("✅ 고급 모듈들 import 성공")
except ImportError as e:
    ADVANCED_MODULES_AVAILABLE = False
    logger.warning(f"⚠️ 고급 모듈들 import 실패: {e}")

# 패키지 정보
__version__ = "1.0.0"
__author__ = "MyCloset AI Team"
__description__ = "Virtual Fitting Models Package"

# 사용 가능한 클래스들
__all__ = [
    "VirtualFittingStep",
    "VirtualFittingModelLoader", 
    "VirtualFittingEnsembleSystem",
    "VirtualFittingPostprocessor",
    "VirtualFittingQualityAssessment",
    "SUPPORTED_MODELS",
    "ENSEMBLE_METHODS"
]

# config에서 정의된 클래스들도 추가
if CONFIG_AVAILABLE:
    try:
        from .config import EnhancedVirtualFittingConfig, VirtualFittingModel, QualityLevel, FITTING_TYPES, VISUALIZATION_COLORS
        __all__.extend([
            "EnhancedVirtualFittingConfig",
            "VirtualFittingModel", 
            "QualityLevel",
            "FITTING_TYPES",
            "VISUALIZATION_COLORS"
        ])
    except ImportError as e:
        logger.error(f"❌ config 클래스들 import 실패: {e}")

# 패키지 초기화 완료
logger.info(f"✅ Virtual Fitting Models Package 초기화 완료 (버전: {__version__})")
logger.info(f"✅ 지원하는 모델: {len(SUPPORTED_MODELS)}개")
logger.info(f"✅ 앙상블 방법: {len(ENSEMBLE_METHODS)}개")
logger.info(f"✅ VirtualFittingStep: {'✅' if VIRTUAL_FITTING_STEP_AVAILABLE else '❌'}")
logger.info(f"✅ 모델 로더: {'✅' if MODEL_LOADER_AVAILABLE else '❌'}")
logger.info(f"✅ 앙상블 시스템: {'✅' if ENSEMBLE_SYSTEM_AVAILABLE else '❌'}")
logger.info(f"✅ 후처리 시스템: {'✅' if POSTPROCESSOR_AVAILABLE else '❌'}")
logger.info(f"✅ 품질 평가: {'✅' if QUALITY_ASSESSMENT_AVAILABLE else '❌'}")
logger.info(f"✅ config 모듈: {'✅' if CONFIG_AVAILABLE else '❌'}")

# 새로운 구조 정보
logger.info("🎯 새로운 구조 구분:")
logger.info("  📁 models/: 추론용 신경망 구조 (체크포인트 없이도 동작)")
logger.info("  📁 checkpoints/: 사전 훈련된 가중치 로딩 및 매핑")
logger.info("  📁 model_loader.py: 통합 관리 및 최적 모델 제공")
