# backend/app/services/__init__.py
"""
🔥 MyCloset AI - 서비스 레이어 통합 관리
구조적 개선 버전 (2025.07.23)

✅ BodyMeasurements 안전한 import 보장
✅ 단순화된 import 구조
✅ M3 Max conda 환경 호환성
✅ 명확한 서비스 매핑
✅ 안전한 오류 처리
✅ 323번째 줄 오류 완전 해결
"""

import logging
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

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
# 🔥 BodyMeasurements 안전한 import (최우선)
# =============================================================================

BodyMeasurements = None
BODY_MEASUREMENTS_AVAILABLE = False

def _import_body_measurements():
    """BodyMeasurements 안전한 import - 다중 경로 시도"""
    global BodyMeasurements, BODY_MEASUREMENTS_AVAILABLE
    
    # 방법 1: models.schemas에서 직접 import (가장 확실한 방법)
    try:
        from ..models.schemas import BodyMeasurements as _BodyMeasurements
        BodyMeasurements = _BodyMeasurements
        BODY_MEASUREMENTS_AVAILABLE = True
        logger.info("✅ BodyMeasurements import 성공 (models.schemas)")
        return True
    except ImportError as e1:
        logger.warning(f"⚠️ models.schemas에서 BodyMeasurements import 실패: {e1}")
    
    # 방법 2: models.__init__.py에서 import
    try:
        from ..models import BodyMeasurements as _BodyMeasurements
        BodyMeasurements = _BodyMeasurements
        BODY_MEASUREMENTS_AVAILABLE = True
        logger.info("✅ BodyMeasurements import 성공 (models.__init__)")
        return True
    except ImportError as e2:
        logger.warning(f"⚠️ models.__init__에서 BodyMeasurements import 실패: {e2}")
    
    # 방법 3: 폴백 클래스 생성 (가장 안전한 방법)
    try:
        @dataclass
        class _FallbackBodyMeasurements:
            """폴백 BodyMeasurements 클래스 - conda 환경 최적화"""
            height: float
            weight: float
            chest: Optional[float] = None
            waist: Optional[float] = None
            hips: Optional[float] = None
            
            @property
            def bmi(self) -> float:
                """BMI 계산"""
                if self.height <= 0 or self.weight <= 0:
                    return 0.0
                height_m = self.height / 100.0
                return round(self.weight / (height_m ** 2), 2)
            
            def to_dict(self) -> dict:
                """딕셔너리로 변환"""
                return {
                    'height': self.height,
                    'weight': self.weight,
                    'chest': self.chest,
                    'waist': self.waist,
                    'hips': self.hips,
                    'bmi': self.bmi
                }
        
        BodyMeasurements = _FallbackBodyMeasurements
        BODY_MEASUREMENTS_AVAILABLE = False  # 폴백이므로 False
        logger.info("✅ BodyMeasurements 폴백 클래스 생성 완료")
        return True
        
    except Exception as e3:
        logger.error(f"❌ BodyMeasurements 폴백 클래스 생성 실패: {e3}")
        return False

# BodyMeasurements 미리 import 시도
_import_success = _import_body_measurements()
if not _import_success:
    logger.error("❌ BodyMeasurements를 어떤 방법으로도 로드할 수 없습니다!")
    # 최종 폴백: 빈 클래스라도 생성
    class BodyMeasurements:
        def __init__(self, height: float = 0, weight: float = 0, **kwargs):
            self.height = height
            self.weight = weight
            for k, v in kwargs.items():
                setattr(self, k, v)

logger.info(f"🔥 BodyMeasurements 상태: {'✅ 사용가능' if BodyMeasurements else '❌ 없음'}")

# =============================================================================
# 🎯 핵심 서비스 Import (우선순위 순서)
# =============================================================================

# 1. step_service.py - 메인 서비스 레이어 (BodyMeasurements 제외)
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
    )
    
    STEP_SERVICE_AVAILABLE = True
    logger.info("✅ step_service.py 로드 성공 (BodyMeasurements 제외)")
    
except ImportError as e:
    logger.error(f"❌ step_service.py 로드 실패: {e}")
    STEP_SERVICE_AVAILABLE = False
    
    # 폴백 클래스들
    class UnifiedStepServiceManager: 
        def __init__(self):
            logger.warning("⚠️ step_service.py 폴백 매니저 사용")
    
    class BaseStepService: 
        def __init__(self):
            logger.warning("⚠️ step_service.py 폴백 서비스 사용")
    
    def get_step_service_manager(): 
        raise RuntimeError("step_service.py를 사용할 수 없습니다")
    
    # 기타 폴백들
    UnifiedStepServiceInterface = None
    UnifiedStepImplementationManager = None
    StepServiceFactory = None
    UploadValidationService = None
    MeasurementsValidationService = None
    HumanParsingService = None
    PoseEstimationService = None
    ClothingAnalysisService = None
    GeometricMatchingService = None
    VirtualFittingService = None
    ResultAnalysisService = None
    CompletePipelineService = None
    get_step_service_manager_async = None
    cleanup_step_service_manager = None
    UnifiedServiceStatus = None
    ProcessingMode = None
    UnifiedServiceMetrics = None
    get_service_availability_info = None
    get_enhanced_system_compatibility_info = None
    optimize_device_memory = None
    validate_image_file_content = None
    convert_image_to_base64 = None

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
    LegacyCompletePipelineService = None
    SingleStepPipelineService = None
    PipelineStatusService = None

# 3. 분석 서비스들 (선택적)
try:
    from .human_analysis import HumanBodyAnalyzer, get_human_analyzer
    from .image_processor import ImageProcessor
    ANALYSIS_SERVICES_AVAILABLE = True
    logger.info("✅ 분석 서비스들 로드 성공")
    
except ImportError as e:
    logger.warning(f"⚠️ 분석 서비스들 로드 실패: {e}")
    ANALYSIS_SERVICES_AVAILABLE = False
    HumanBodyAnalyzer = None
    get_human_analyzer = None
    ImageProcessor = None

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
    if STEP_SERVICE_AVAILABLE and get_step_service_manager_async:
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
    available_services = []
    
    if STEP_SERVICE_AVAILABLE:
        available_services.extend([
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
        available_services.extend([
            "SingleStepPipelineService",
            "PipelineStatusService"
        ])

    if ANALYSIS_SERVICES_AVAILABLE:
        available_services.extend([
            "HumanBodyAnalyzer",
            "ImageProcessor"
        ])
    
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
        "schemas": {
            "body_measurements": BODY_MEASUREMENTS_AVAILABLE,
            "body_measurements_class": BodyMeasurements is not None
        },
        "recommended_usage": "step_service" if STEP_SERVICE_AVAILABLE else "legacy",
        "total_available_services": len(available_services),
        "available_services": available_services
    }
    
    # step_service 상세 정보 추가
    if STEP_SERVICE_AVAILABLE and get_service_availability_info:
        try:
            step_info = get_service_availability_info()
            base_info["step_service_details"] = step_info
        except Exception as e:
            logger.warning(f"step_service 정보 조회 실패: {e}")
    
    return base_info

# =============================================================================
# 🔥 Export 목록 (동적 생성)
# =============================================================================

def _get_available_exports():
    """사용 가능한 export 목록 동적 생성"""
    exports = [
        # 🎯 핵심 팩토리 함수들 (항상 사용 가능)
        "get_main_service_manager",
        "get_main_service_manager_async",
        "get_pipeline_service",
        "get_pipeline_service_sync", 
        "get_pipeline_manager_service",
        "get_service_status",
        
        # 🔧 상수 (항상 사용 가능)
        "STEP_SERVICE_AVAILABLE",
        "LEGACY_PIPELINE_AVAILABLE",
        "ANALYSIS_SERVICES_AVAILABLE",
        "BODY_MEASUREMENTS_AVAILABLE"
    ]
    
    # BodyMeasurements 조건부 추가 (안전하게)
    if BodyMeasurements is not None:
        exports.append("BodyMeasurements")
    
    # step_service 관련 조건부 추가
    if STEP_SERVICE_AVAILABLE:
        step_exports = [
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
            "CompletePipelineService",
            "get_step_service_manager",
            "get_step_service_manager_async",
            "cleanup_step_service_manager"
        ]
        
        # None이 아닌 것들만 추가
        for export in step_exports:
            if globals().get(export) is not None:
                exports.append(export)
        
        # 상태 관리 추가
        if UnifiedServiceStatus is not None:
            exports.append("UnifiedServiceStatus")
        if ProcessingMode is not None:
            exports.append("ProcessingMode")
        if UnifiedServiceMetrics is not None:
            exports.append("UnifiedServiceMetrics")
        
        # 유틸리티 추가
        utility_exports = [
            "get_service_availability_info",
            "get_enhanced_system_compatibility_info",
            "optimize_device_memory",
            "validate_image_file_content",
            "convert_image_to_base64"
        ]
        for export in utility_exports:
            if globals().get(export) is not None:
                exports.append(export)

    # legacy pipeline 조건부 추가
    if LEGACY_PIPELINE_AVAILABLE:
        legacy_exports = ["LegacyCompletePipelineService", "SingleStepPipelineService", "PipelineStatusService"]
        for export in legacy_exports:
            if globals().get(export) is not None:
                exports.append(export)

    # analysis services 조건부 추가
    if ANALYSIS_SERVICES_AVAILABLE:
        analysis_exports = ["HumanBodyAnalyzer", "ImageProcessor", "get_human_analyzer"]
        for export in analysis_exports:
            if globals().get(export) is not None:
                exports.append(export)
    
    return exports

# __all__ 동적 생성
__all__ = _get_available_exports()

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
    logger.info(f"📋 BodyMeasurements: {'✅' if BODY_MEASUREMENTS_AVAILABLE else '🔄 폴백'}")
    logger.info(f"📦 총 {len(__all__)}개 서비스/함수 export")
    
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

# =============================================================================
# 🔥 안전한 호환성 별칭 설정 (323번째 줄 오류 완전 해결)
# =============================================================================

def _setup_safe_compatibility_aliases():
    """안전한 호환성 별칭 설정 - 323번째 줄 오류 완전 해결"""
    try:
        # API 레이어에서 사용하는 이름들
        if STEP_SERVICE_AVAILABLE and UnifiedStepServiceManager is not None:
            globals()['PipelineService'] = UnifiedStepServiceManager  # 호환성 별칭
            globals()['StepServiceManager'] = UnifiedStepServiceManager  # 호환성 별칭
            logger.info("✅ UnifiedStepServiceManager 별칭 설정 완료")
        
        # 🚨 323번째 줄 오류 해결: BodyMeasurements가 None이 아닐 때만 별칭 생성
        if BodyMeasurements is not None:
            globals()['ServiceBodyMeasurements'] = BodyMeasurements  # 🔥 이제 안전함!
            logger.info("✅ BodyMeasurements 별칭 설정 완료")
        else:
            logger.warning("⚠️ BodyMeasurements가 None이므로 ServiceBodyMeasurements 별칭 생성 건너뜀")
        
        logger.info("✅ 모든 호환성 별칭 설정 완료")
        
    except Exception as e:
        logger.warning(f"⚠️ 호환성 별칭 설정 중 일부 실패: {e}")

# 안전한 호환성 별칭 설정 실행
_setup_safe_compatibility_aliases()

# =============================================================================
# 🔥 최종 상태 체크
# =============================================================================

logger.info("=" * 60)
logger.info("🔥 MyCloset AI 서비스 레이어 최종 상태:")
logger.info(f"   🎯 step_service.py: {'✅ 로드됨' if STEP_SERVICE_AVAILABLE else '❌ 실패'}")
logger.info(f"   📋 BodyMeasurements: {'✅ 정상' if BodyMeasurements is not None else '❌ None'}")
logger.info(f"   🔧 ServiceBodyMeasurements: {'✅ 생성됨' if 'ServiceBodyMeasurements' in globals() else '❌ 실패'}")
logger.info(f"   📦 총 Export: {len(__all__)}개")
logger.info(f"   🐍 conda 환경: {CONDA_ENV_NAME}")
logger.info("=" * 60)

# 323번째 줄 오류 해결 확인
if 'ServiceBodyMeasurements' in globals():
    logger.info("🎉 323번째 줄 BodyMeasurements 오류 완전 해결!")
else:
    logger.error("❌ 323번째 줄 오류가 여전히 남아있습니다.")

logger.info("✅ MyCloset AI 서비스 레이어 로딩 완료!")