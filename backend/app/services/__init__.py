# backend/app/services/__init__.py
"""
🔥 MyCloset AI - 서비스 레이어 통합 관리
conda 환경 최적화 버전 (2025.07.22)

✅ 단순화된 import 구조
✅ M3 Max conda 환경 호환성
✅ 명확한 서비스 매핑
✅ 안전한 오류 처리
"""

import logging
import os
from typing import Dict, Any, Optional

# 로깅 설정
logger = logging.getLogger(__name__)

# =============================================================================
# 🔥 환경 감지 및 최적화
# =============================================================================

# conda 환경 확인
IN_CONDA = os.environ.get('CONDA_DEFAULT_ENV') is not None
CONDA_ENV_NAME = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')

# M3 Max 환경 확인 (프로젝트 지식 기반)
def is_m3_max_environment():
    """M3 Max 환경 감지"""
    try:
        import platform
        return (
            platform.system() == 'Darwin' and 
            platform.processor() == 'arm' and
            IN_CONDA
        )
    except:
        return False

IS_M3_MAX = is_m3_max_environment()

# =============================================================================
# 🎯 핵심 서비스 Import (우선순위 순서)
# =============================================================================

# 1. step_service.py - 메인 서비스 레이어
try:
    from .step_service import (
        # 핵심 매니저
        UnifiedStepServiceManager,
        UnifiedStepServiceInterface,
        UnifiedStepImplementationManager,
        
        # 기존 호환성
        BaseStepService,
        StepServiceFactory,
        
        # 8단계 개별 서비스들
        UploadValidationService,
        MeasurementsValidationService,
        HumanParsingService,
        PoseEstimationService,
        ClothingAnalysisService,
        GeometricMatchingService,
        VirtualFittingService,
        ResultAnalysisService,
        CompletePipelineService,
        
        # 팩토리 함수들
        get_step_service_manager,
        get_step_service_manager_async,
        cleanup_step_service_manager,
        
        # 상태 관리
        UnifiedServiceStatus,
        ProcessingMode,
        UnifiedServiceMetrics,
        
        # 유틸리티
        get_service_availability_info,
        get_enhanced_system_compatibility_info,
        optimize_device_memory,
        validate_image_file_content,
        convert_image_to_base64,
        
        # 스키마
        BodyMeasurements,
    )
    
    STEP_SERVICE_AVAILABLE = True
    logger.info("✅ step_service.py 로드 성공")
    
except ImportError as e:
    logger.error(f"❌ step_service.py 로드 실패: {e}")
    STEP_SERVICE_AVAILABLE = False
    
    # 폴백 클래스들
    class UnifiedStepServiceManager: pass
    class BaseStepService: pass
    def get_step_service_manager(): 
        raise RuntimeError("step_service.py를 사용할 수 없습니다")

# 2. 기존 서비스들 (선택적)
try:
    from .pipeline_service import (
        CompletePipelineService as LegacyCompletePipelineService,
        SingleStepPipelineService,
        PipelineStatusService,
    )
    LEGACY_PIPELINE_AVAILABLE = True
    logger.info("✅ 기존 pipeline_service.py 로드 성공")
    
except ImportError as e:
    logger.warning(f"⚠️ 기존 pipeline_service.py 로드 실패: {e}")
    LEGACY_PIPELINE_AVAILABLE = False

# 3. 분석 서비스들 (선택적)
try:
    from .human_analysis import HumanBodyAnalyzer, get_human_analyzer
    from .image_processor import ImageProcessor
    ANALYSIS_SERVICES_AVAILABLE = True
    logger.info("✅ 분석 서비스들 로드 성공")
    
except ImportError as e:
    logger.warning(f"⚠️ 분석 서비스들 로드 실패: {e}")
    ANALYSIS_SERVICES_AVAILABLE = False

# =============================================================================
# 🔧 간소화된 팩토리 함수들
# =============================================================================

def get_main_service_manager():
    """메인 서비스 매니저 반환 (단순화)"""
    if STEP_SERVICE_AVAILABLE:
        return get_step_service_manager()
    else:
        raise RuntimeError("서비스 매니저를 사용할 수 없습니다")

async def get_main_service_manager_async():
    """메인 서비스 매니저 반환 (비동기, 단순화)"""
    if STEP_SERVICE_AVAILABLE:
        return await get_step_service_manager_async()
    else:
        raise RuntimeError("서비스 매니저를 사용할 수 없습니다")

def get_pipeline_service():
    """파이프라인 서비스 반환 (호환성)"""
    return get_main_service_manager()

def get_pipeline_service_sync():
    """파이프라인 서비스 반환 (동기, 호환성)"""
    return get_main_service_manager()

def get_pipeline_manager_service():
    """파이프라인 매니저 서비스 반환 (호환성)"""
    return get_main_service_manager()

# =============================================================================
# 🎯 서비스 상태 정보
# =============================================================================

def get_service_status():
    """서비스 상태 정보 반환 (단순화)"""
    base_info = {
        "conda_environment": {
            "active": IN_CONDA,
            "name": CONDA_ENV_NAME,
            "m3_max_optimized": IS_M3_MAX
        },
        "services": {
            "step_service": STEP_SERVICE_AVAILABLE,
            "legacy_pipeline": LEGACY_PIPELINE_AVAILABLE,
            "analysis_services": ANALYSIS_SERVICES_AVAILABLE
        },
        "recommended_usage": "step_service" if STEP_SERVICE_AVAILABLE else "legacy",
        "total_available_services": len(AVAILABLE_SERVICES)
    }
    
    # step_service 상세 정보 추가
    if STEP_SERVICE_AVAILABLE:
        try:
            step_info = get_service_availability_info()
            base_info["step_service_details"] = step_info
        except Exception as e:
            logger.warning(f"step_service 정보 조회 실패: {e}")
    
    return base_info

# =============================================================================
# 🔥 서비스 목록 (단순화)
# =============================================================================

AVAILABLE_SERVICES = []

if STEP_SERVICE_AVAILABLE:
    AVAILABLE_SERVICES.extend([
        "UnifiedStepServiceManager",
        "BaseStepService",
        "StepServiceFactory",
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

if LEGACY_PIPELINE_AVAILABLE:
    AVAILABLE_SERVICES.extend([
        "SingleStepPipelineService",
        "PipelineStatusService"
    ])

if ANALYSIS_SERVICES_AVAILABLE:
    AVAILABLE_SERVICES.extend([
        "HumanBodyAnalyzer",
        "ImageProcessor"
    ])

# =============================================================================
# 🔥 Export 목록 (단순화, 핵심만)
# =============================================================================

__all__ = [
    # 🎯 핵심 클래스들
    "UnifiedStepServiceManager",
    "BaseStepService",
    "StepServiceFactory",
    
    # 🎯 8단계 서비스들
    "UploadValidationService",
    "MeasurementsValidationService",
    "HumanParsingService",
    "PoseEstimationService",
    "ClothingAnalysisService",
    "GeometricMatchingService",
    "VirtualFittingService",
    "ResultAnalysisService",
    "CompletePipelineService",
    
    # 🎯 핵심 팩토리 함수들
    "get_main_service_manager",
    "get_main_service_manager_async",
    "get_step_service_manager",
    "get_step_service_manager_async",
    "cleanup_step_service_manager",
    
    # 🔧 호환성 함수들 (단순화)
    "get_pipeline_service",
    "get_pipeline_service_sync", 
    "get_pipeline_manager_service",
    
    # 🔧 상태 및 유틸리티
    "get_service_status",
    "get_service_availability_info",
    "optimize_device_memory",
    "validate_image_file_content",
    "convert_image_to_base64",
    
    # 🔧 스키마
    "BodyMeasurements",
    
    # 🔧 상태 관리
    "UnifiedServiceStatus",
    "ProcessingMode",
    "UnifiedServiceMetrics",
    
    # 🔧 상수
    "AVAILABLE_SERVICES",
    "STEP_SERVICE_AVAILABLE",
    "LEGACY_PIPELINE_AVAILABLE",
    "ANALYSIS_SERVICES_AVAILABLE"
]

# 조건부 export
if LEGACY_PIPELINE_AVAILABLE:
    __all__.extend([
        "SingleStepPipelineService",
        "PipelineStatusService"
    ])

if ANALYSIS_SERVICES_AVAILABLE:
    __all__.extend([
        "HumanBodyAnalyzer",
        "ImageProcessor",
        "get_human_analyzer"
    ])

# =============================================================================
# 🎉 초기화 완료 및 conda 최적화 확인
# =============================================================================

def _log_initialization_status():
    """초기화 상태 로깅"""
    logger.info("🎉 MyCloset AI 서비스 레이어 초기화 완료!")
    logger.info(f"🐍 conda 환경: {'✅' if IN_CONDA else '❌'} ({CONDA_ENV_NAME})")
    logger.info(f"🍎 M3 Max 최적화: {'✅' if IS_M3_MAX else '❌'}")
    logger.info(f"🎯 step_service: {'✅' if STEP_SERVICE_AVAILABLE else '❌'}")
    logger.info(f"🔧 기존 pipeline: {'✅' if LEGACY_PIPELINE_AVAILABLE else '❌'}")
    logger.info(f"📊 분석 서비스: {'✅' if ANALYSIS_SERVICES_AVAILABLE else '❌'}")
    logger.info(f"📋 총 {len(AVAILABLE_SERVICES)}개 서비스 사용 가능")
    
    if IS_M3_MAX and STEP_SERVICE_AVAILABLE:
        logger.info("🚀 M3 Max + conda + step_service 조합 - 최고 성능 모드!")
        logger.info("   - 128GB Unified Memory 활용")
        logger.info("   - GPU 가속 AI 추론 지원")
        logger.info("   - 고성능 이미지 처리")
    
    if not IN_CONDA:
        logger.warning("⚠️ conda 환경이 아닙니다. 성능 최적화를 위해 conda 사용을 권장합니다.")
        logger.warning("   권장 설정: conda activate mycloset-ai")

# 초기화 로깅 실행
_log_initialization_status()

# 중요한 별칭들 (기존 코드 호환성)
# API 레이어에서 사용하는 이름들
PipelineService = UnifiedStepServiceManager  # 호환성 별칭
StepServiceManager = UnifiedStepServiceManager  # 호환성 별칭
ServiceBodyMeasurements = BodyMeasurements  # 호환성 별칭

logger.info("✅ 모든 호환성 별칭 설정 완료")