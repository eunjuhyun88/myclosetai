"""
app/services/__init__.py - 서비스 레이어 패키지 초기화

✅ 서비스 레이어 컴포넌트들 export
✅ 의존성 관리
✅ 편리한 import 제공
"""

import logging

# 로깅 설정
logger = logging.getLogger(__name__)

# 핵심 서비스들 import
try:
    # 파이프라인 관련 서비스
    from .pipeline_service import PipelineService, get_pipeline_service
    from .step_service import (
        BaseStepService,
        UploadValidationService,
        MeasurementsValidationService, 
        HumanParsingService,
        VirtualFittingService,
        StepServiceManager,
        get_step_service_manager
    )
    
    # 기존 서비스들
    from .human_analysis import HumanBodyAnalyzer, get_human_analyzer
    from .image_processor import ImageProcessor
    
    # 성공적으로 import된 서비스들
    AVAILABLE_SERVICES = [
        "PipelineService",
        "StepServiceManager", 
        "UploadValidationService",
        "MeasurementsValidationService",
        "HumanParsingService",
        "VirtualFittingService",
        "HumanBodyAnalyzer",  # 기존 서비스
        "ImageProcessor"      # 기존 서비스
    ]
    
    logger.info(f"✅ 서비스 레이어 초기화 완료: {len(AVAILABLE_SERVICES)}개 서비스 로드됨")
    
except ImportError as e:
    logger.error(f"❌ 서비스 레이어 초기화 실패: {e}")
    
    # 폴백: 빈 서비스들
    AVAILABLE_SERVICES = []

# ============================================================================
# 🎯 Export할 항목들
# ============================================================================

__all__ = [
    # 파이프라인 서비스
    "PipelineService",
    "get_pipeline_service",
    
    # 단계별 서비스
    "BaseStepService",
    "UploadValidationService",
    "MeasurementsValidationService",
    "HumanParsingService", 
    "VirtualFittingService",
    "StepServiceManager",
    "get_step_service_manager",
    
    # 기존 서비스들
    "HumanBodyAnalyzer",
    "get_human_analyzer",
    "ImageProcessor",
    
    # 메타 정보
    "AVAILABLE_SERVICES"
]

# ============================================================================
# 🎉 패키지 정보
# ============================================================================

__version__ = "1.0.0"
__author__ = "MyCloset AI Team"
__description__ = "서비스 레이어 - 비즈니스 로직 처리"

logger.info("🎉 MyCloset AI 서비스 레이어 패키지 로드 완료!")