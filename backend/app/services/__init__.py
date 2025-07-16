"""
app/services/__init__.py - 서비스 레이어 패키지 초기화 (수정됨)

✅ PipelineService 클래스 대신 실제 구현체들 import
✅ 기존 구조 최대한 유지
✅ Import 오류 해결
"""

import logging

# 로깅 설정
logger = logging.getLogger(__name__)

# 핵심 서비스들 import (수정됨)
try:
    # 파이프라인 관련 서비스 - 실제 구현체들
    from .pipeline_service import (
        # PipelineService,  # 이 클래스는 존재하지 않음 - 제거
        CompletePipelineService,
        SingleStepPipelineService,
        PipelineStatusService,
        PipelineServiceManager,
        get_pipeline_service_manager,
        get_complete_pipeline_service,
        get_single_step_pipeline_service,
        get_pipeline_status_service
    )
    
    from .step_service import (
        BaseStepService,
        UploadValidationService,
        MeasurementsValidationService, 
        HumanParsingService,
        VirtualFittingService,
        CompletePipelineService as StepCompletePipelineService,  # 별칭 사용
        StepServiceManager,
        get_step_service_manager
    )
    
    # 기존 서비스들
    from .human_analysis import HumanBodyAnalyzer, get_human_analyzer
    from .image_processor import ImageProcessor
    
    # 성공적으로 import된 서비스들
    AVAILABLE_SERVICES = [
        "CompletePipelineService",
        "SingleStepPipelineService", 
        "PipelineStatusService",
        "PipelineServiceManager",
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
# 🎯 Export할 항목들 (수정됨)
# ============================================================================

__all__ = [
    # 파이프라인 서비스 - 실제 구현체들
    "CompletePipelineService",
    "SingleStepPipelineService",
    "PipelineStatusService", 
    "PipelineServiceManager",
    "get_pipeline_service_manager",
    "get_complete_pipeline_service",
    "get_single_step_pipeline_service", 
    "get_pipeline_status_service",
    
    # 기존 호환성을 위한 별칭들
    "get_pipeline_service",  # = get_complete_pipeline_service
    
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

# 기존 호환성을 위한 별칭 함수
async def get_pipeline_service():
    """기존 호환성을 위한 별칭"""
    return await get_complete_pipeline_service()

# ============================================================================
# 🎉 패키지 정보
# ============================================================================

__version__ = "1.0.0"
__author__ = "MyCloset AI Team"
__description__ = "서비스 레이어 - 비즈니스 로직 처리"

logger.info("🎉 MyCloset AI 서비스 레이어 패키지 로드 완료!")