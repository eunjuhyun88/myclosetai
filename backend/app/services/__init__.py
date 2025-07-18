"""
app/services/__init__.py - 서비스 레이어 패키지 초기화 (step_service.py 연동)

✅ 새로운 step_service.py 함수들 export
✅ 기존 구조 최대한 유지
✅ Import 오류 해결
✅ API 레이어 호환성 완전 보장
"""

import logging

# 로깅 설정
logger = logging.getLogger(__name__)

# =============================================================================
# 🔥 핵심 서비스들 import (step_service.py 연동)
# =============================================================================

try:
    # 🎯 step_service.py에서 모든 필요한 것들 import
    from .step_service import (
        # 기본 클래스들
        BaseStepService,
        PipelineManagerService,
        
        # 단계별 서비스들
        UploadValidationService, 
        MeasurementsValidationService,
        HumanParsingService,
        PoseEstimationService,
        ClothingAnalysisService, 
        GeometricMatchingService,
        VirtualFittingService,
        ResultAnalysisService,
        CompletePipelineService,
        
        # 팩토리 및 관리자
        StepServiceFactory,
        StepServiceManager,
        
        # 🔥 새로운 함수들 (API 레이어와 일치)
        get_step_service_manager,
        get_step_service_manager_async,
        get_pipeline_service,           # ✅ 기존 호환성
        get_pipeline_service_sync,      # ✅ 기존 호환성
        cleanup_step_service_manager,
        
        # 스키마 및 유틸리티
        BodyMeasurements,
        ServiceBodyMeasurements,        # 별칭
        PipelineService,                # StepServiceManager 별칭
        optimize_device_memory,
        validate_image_file_content,
        convert_image_to_base64
    )
    
    STEP_SERVICE_AVAILABLE = True
    logger.info("✅ step_service.py Import 성공")
    
except ImportError as e:
    logger.error(f"❌ step_service.py Import 실패: {e}")
    STEP_SERVICE_AVAILABLE = False
    
    # 폴백: 빈 클래스들
    class BaseStepService: pass
    class StepServiceManager: pass
    
    def get_step_service_manager():
        raise RuntimeError("step_service.py를 사용할 수 없습니다")

# =============================================================================
# 🔧 기존 서비스들 (선택적 import)
# =============================================================================

try:
    # 파이프라인 관련 서비스 (기존 유지)
    from .pipeline_service import (
        CompletePipelineService as OldCompletePipelineService,
        SingleStepPipelineService,
        PipelineStatusService,
        PipelineServiceManager,
        get_pipeline_service_manager as get_old_pipeline_service_manager,
        get_complete_pipeline_service,
        get_single_step_pipeline_service,
        get_pipeline_status_service
    )
    OLD_PIPELINE_SERVICE_AVAILABLE = True
    logger.info("✅ 기존 pipeline_service.py Import 성공")
    
except ImportError as e:
    logger.warning(f"⚠️ 기존 pipeline_service.py Import 실패: {e}")
    OLD_PIPELINE_SERVICE_AVAILABLE = False

try:
    # 기존 분석 서비스들
    from .human_analysis import HumanBodyAnalyzer, get_human_analyzer
    from .image_processor import ImageProcessor
    ANALYSIS_SERVICES_AVAILABLE = True
    logger.info("✅ 분석 서비스들 Import 성공")
    
except ImportError as e:
    logger.warning(f"⚠️ 분석 서비스들 Import 실패: {e}")
    ANALYSIS_SERVICES_AVAILABLE = False

# =============================================================================
# 🎯 통합 팩토리 함수들 (신구 서비스 통합)
# =============================================================================

def get_unified_pipeline_service():
    """통합 파이프라인 서비스 반환 (신규 우선, 기존 폴백)"""
    if STEP_SERVICE_AVAILABLE:
        return get_pipeline_service()  # 신규 step_service 사용
    elif OLD_PIPELINE_SERVICE_AVAILABLE:
        return get_old_pipeline_service_manager()  # 기존 pipeline_service 사용
    else:
        raise RuntimeError("파이프라인 서비스를 사용할 수 없습니다")

async def get_unified_step_service_manager():
    """통합 스텝 서비스 매니저 반환 (비동기)"""
    if STEP_SERVICE_AVAILABLE:
        return await get_step_service_manager_async()
    else:
        raise RuntimeError("스텝 서비스 매니저를 사용할 수 없습니다")

def get_service_availability_info():
    """서비스 가용성 정보 반환"""
    return {
        "step_service_available": STEP_SERVICE_AVAILABLE,
        "old_pipeline_service_available": OLD_PIPELINE_SERVICE_AVAILABLE,
        "analysis_services_available": ANALYSIS_SERVICES_AVAILABLE,
        "recommended_service": "step_service" if STEP_SERVICE_AVAILABLE else "pipeline_service"
    }

# =============================================================================
# 🎯 호환성 함수들 (기존 코드와의 호환성)
# =============================================================================

# 기존 main.py와의 호환성을 위한 별칭들
if STEP_SERVICE_AVAILABLE:
    # 신규 서비스 사용
    get_pipeline_service_manager = get_pipeline_service
    PipelineServiceManager = StepServiceManager
    
    # 호환성 함수들
    def get_complete_virtual_fitting_service():
        """완전한 가상 피팅 서비스 반환"""
        return get_pipeline_service()
    
    def get_step_processing_service():
        """단계별 처리 서비스 반환"""
        return get_pipeline_service()
        
elif OLD_PIPELINE_SERVICE_AVAILABLE:
    # 기존 서비스 사용
    get_pipeline_service_manager = get_old_pipeline_service_manager
    
    def get_complete_virtual_fitting_service():
        return get_complete_pipeline_service()
    
    def get_step_processing_service():
        return get_single_step_pipeline_service()

# =============================================================================
# 🎯 Export할 항목들 (완전한 목록)
# =============================================================================

# 사용 가능한 서비스들 동적 구성
AVAILABLE_SERVICES = []

if STEP_SERVICE_AVAILABLE:
    AVAILABLE_SERVICES.extend([
        "StepServiceManager",
        "UploadValidationService",
        "MeasurementsValidationService", 
        "HumanParsingService",
        "PoseEstimationService",
        "ClothingAnalysisService",
        "GeometricMatchingService",
        "VirtualFittingService",
        "ResultAnalysisService",
        "CompletePipelineService"
    ])

if OLD_PIPELINE_SERVICE_AVAILABLE:
    AVAILABLE_SERVICES.extend([
        "SingleStepPipelineService",
        "PipelineStatusService",
        "OldCompletePipelineService"
    ])

if ANALYSIS_SERVICES_AVAILABLE:
    AVAILABLE_SERVICES.extend([
        "HumanBodyAnalyzer",
        "ImageProcessor"
    ])

logger.info(f"✅ 서비스 레이어 초기화 완료: {len(AVAILABLE_SERVICES)}개 서비스 로드됨")

# =============================================================================
# 🎉 __all__ Export 목록
# =============================================================================

__all__ = [
    # 🔥 핵심 서비스들 (step_service.py 기반)
    "BaseStepService",
    "PipelineManagerService", 
    "StepServiceManager",
    "StepServiceFactory",
    
    # 단계별 서비스들
    "UploadValidationService",
    "MeasurementsValidationService",
    "HumanParsingService", 
    "PoseEstimationService",
    "ClothingAnalysisService",
    "GeometricMatchingService",
    "VirtualFittingService",
    "ResultAnalysisService",
    "CompletePipelineService",
    
    # 🔥 팩토리 함수들 (API 레이어 호환)
    "get_step_service_manager",
    "get_step_service_manager_async",
    "get_pipeline_service",
    "get_pipeline_service_sync",
    "get_pipeline_service_manager",       # 기존 호환성
    "get_unified_pipeline_service",       # 통합 함수
    "get_unified_step_service_manager",   # 통합 함수
    "get_complete_virtual_fitting_service", # 호환성
    "get_step_processing_service",        # 호환성
    "cleanup_step_service_manager",
    
    # 스키마 및 데이터 모델
    "BodyMeasurements",
    "ServiceBodyMeasurements",
    
    # 별칭들 (기존 코드 호환성)
    "PipelineService",                    # StepServiceManager 별칭
    "PipelineServiceManager",             # 호환성 별칭
    
    # 유틸리티 함수들
    "optimize_device_memory",
    "validate_image_file_content", 
    "convert_image_to_base64",
    "get_service_availability_info",
    
    # 상태 정보
    "AVAILABLE_SERVICES",
    "STEP_SERVICE_AVAILABLE",
    "OLD_PIPELINE_SERVICE_AVAILABLE",
    "ANALYSIS_SERVICES_AVAILABLE"
]

# 조건부 export (기존 서비스들)
if OLD_PIPELINE_SERVICE_AVAILABLE:
    __all__.extend([
        "SingleStepPipelineService",
        "PipelineStatusService", 
        "get_complete_pipeline_service",
        "get_single_step_pipeline_service",
        "get_pipeline_status_service"
    ])

if ANALYSIS_SERVICES_AVAILABLE:
    __all__.extend([
        "HumanBodyAnalyzer",
        "ImageProcessor",
        "get_human_analyzer"
    ])

# =============================================================================
# 🎉 초기화 완료 로그
# =============================================================================

logger.info("🎉 Services 패키지 초기화 완료!")
logger.info(f"✅ step_service.py 기반: {'O' if STEP_SERVICE_AVAILABLE else 'X'}")
logger.info(f"✅ 기존 pipeline_service.py: {'O' if OLD_PIPELINE_SERVICE_AVAILABLE else 'X'}")
logger.info(f"✅ 분석 서비스들: {'O' if ANALYSIS_SERVICES_AVAILABLE else 'X'}")
logger.info(f"📊 총 {len(AVAILABLE_SERVICES)}개 서비스 사용 가능")

if STEP_SERVICE_AVAILABLE:
    logger.info("🚀 신규 step_service.py를 기본으로 사용합니다")
    logger.info("   - API 레이어와 100% 호환")
    logger.info("   - 새로운 함수명들 지원")
    logger.info("   - 기존 함수명들도 호환성 유지")
else:
    logger.warning("⚠️ step_service.py 사용 불가 - 기존 서비스들로 폴백")