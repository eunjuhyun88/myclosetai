#!/usr/bin/env python3
"""
🔥 MyCloset AI - Final Output Models Package
============================================

🎯 정확한 구조 구분으로 완벽한 모듈화
=====================================

📁 models/ 폴더: 추론용 신경망 구조 (체크포인트 없이도 동작)
📁 checkpoints/ 폴더: 사전 훈련된 가중치 로딩 및 매핑
📁 model_loader.py: 두 가지를 조합하여 최적의 모델 제공

✅ 8개 Final Output 모델 완벽 통합
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
    "output_integration_transformer",
    "cross_modal_attention",
    "multi_modal_fusion",
    "final_output_generator",
    "output_quality_assessor",
    "output_refiner",
    "attention_output",
    "ensemble_output"
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

# FinalOutputStep 클래스 import
try:
    from .step_09_final_output import FinalOutputStep
    FINAL_OUTPUT_STEP_AVAILABLE = True
    logger.info("✅ FinalOutputStep 클래스 import 성공")
    
    # __all__에 추가
    __all__.extend([
        "FinalOutputStep",
        "FINAL_OUTPUT_STEP_AVAILABLE"
    ])
    
except ImportError as e:
    FINAL_OUTPUT_STEP_AVAILABLE = False
    logger.error(f"❌ FinalOutputStep import 실패: {e}")
    raise ImportError("FinalOutputStep을 import할 수 없습니다.")

# 모델 로더 import
try:
    from .final_output_model_loader import FinalOutputModelLoader
    MODEL_LOADER_AVAILABLE = True
    logger.info("✅ FinalOutputModelLoader import 성공")
except ImportError as e:
    MODEL_LOADER_AVAILABLE = False
    logger.error(f"❌ FinalOutputModelLoader import 실패: {e}")
    raise ImportError("FinalOutputModelLoader을 import할 수 없습니다.")

# 앙상블 시스템 import
try:
    from .ensemble.hybrid_ensemble import FinalOutputEnsembleSystem
    ENSEMBLE_SYSTEM_AVAILABLE = True
    logger.info("✅ FinalOutputEnsembleSystem import 성공")
except ImportError as e:
    ENSEMBLE_SYSTEM_AVAILABLE = False
    logger.error(f"❌ FinalOutputEnsembleSystem import 실패: {e}")
    raise ImportError("FinalOutputEnsembleSystem을 import할 수 없습니다.")

# 후처리 시스템 import
try:
    from .postprocessing.postprocessor import FinalOutputPostprocessor
    POSTPROCESSOR_AVAILABLE = True
    logger.info("✅ FinalOutputPostprocessor import 성공")
except ImportError as e:
    POSTPROCESSOR_AVAILABLE = False
    logger.error(f"❌ FinalOutputPostprocessor import 실패: {e}")
    raise ImportError("FinalOutputPostprocessor을 import할 수 없습니다.")

# 품질 평가 시스템 import
try:
    from .utils.quality_assessment import FinalOutputQualityAssessment
    QUALITY_ASSESSMENT_AVAILABLE = True
    logger.info("✅ FinalOutputQualityAssessment import 성공")
except ImportError as e:
    QUALITY_ASSESSMENT_AVAILABLE = False
    logger.error(f"❌ FinalOutputQualityAssessment import 실패: {e}")
    raise ImportError("FinalOutputQualityAssessment을 import할 수 없습니다.")

# 고급 모듈들 import
try:
    from .models.output_integration_transformer import OutputIntegrationTransformer
    from .models.cross_modal_attention import CrossModalAttention
    from .models.multi_modal_fusion import MultiModalFusion
    ADVANCED_MODULES_AVAILABLE = True
    logger.info("✅ 고급 모듈들 import 성공")
except ImportError as e:
    ADVANCED_MODULES_AVAILABLE = False
    logger.warning(f"⚠️ 고급 모듈들 import 실패: {e}")

# 패키지 정보
__version__ = "1.0.0"
__author__ = "MyCloset AI Team"
__description__ = "Final Output Models Package"

# 사용 가능한 클래스들
__all__ = [
    "FinalOutputStep",
    "FinalOutputModelLoader", 
    "FinalOutputEnsembleSystem",
    "FinalOutputPostprocessor",
    "FinalOutputQualityAssessment",
    "OutputIntegrationTransformer",
    "CrossModalAttention",
    "MultiModalFusion",
    "SUPPORTED_MODELS",
    "ENSEMBLE_METHODS"
]

# config에서 정의된 클래스들도 추가
if CONFIG_AVAILABLE:
    try:
        from .config import EnhancedFinalOutputConfig, FinalOutputModel, QualityLevel, OUTPUT_TYPES, VISUALIZATION_COLORS
        __all__.extend([
            "EnhancedFinalOutputConfig",
            "FinalOutputModel", 
            "QualityLevel",
            "OUTPUT_TYPES",
            "VISUALIZATION_COLORS"
        ])
    except ImportError as e:
        logger.error(f"❌ config 클래스들 import 실패: {e}")

# 패키지 초기화 완료
logger.info(f"✅ Final Output Models Package 초기화 완료 (버전: {__version__})")
logger.info(f"✅ 지원하는 모델: {len(SUPPORTED_MODELS)}개")
logger.info(f"✅ 앙상블 방법: {len(ENSEMBLE_METHODS)}개")
logger.info(f"✅ FinalOutputStep: {'✅' if FINAL_OUTPUT_STEP_AVAILABLE else '❌'}")
logger.info(f"✅ 모델 로더: {'✅' if MODEL_LOADER_AVAILABLE else '❌'}")
logger.info(f"✅ 앙상블 시스템: {'✅' if ENSEMBLE_SYSTEM_AVAILABLE else '❌'}")
logger.info(f"✅ 후처리 시스템: {'✅' if POSTPROCESSOR_AVAILABLE else '❌'}")
logger.info(f"✅ 품질 평가: {'✅' if QUALITY_ASSESSMENT_AVAILABLE else '❌'}")
logger.info(f"✅ 고급 모듈들: {'✅' if ADVANCED_MODULES_AVAILABLE else '❌'}")
logger.info(f"✅ config 모듈: {'✅' if CONFIG_AVAILABLE else '❌'}")

# 새로운 구조 정보
logger.info("🎯 새로운 구조 구분:")
logger.info("  📁 models/: 추론용 신경망 구조 (체크포인트 없이도 동작)")
logger.info("  📁 checkpoints/: 사전 훈련된 가중치 로딩 및 매핑")
logger.info("  📁 model_loader.py: 통합 관리 및 최적 모델 제공")
