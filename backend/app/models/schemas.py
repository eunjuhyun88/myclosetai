# backend/app/models/schemas.py
"""
🔥 MyCloset AI 스키마 시스템 v7.0 - 올바른 설계 버전
======================================================

✅ conda 환경 완전 호환
✅ M3 Max 최적화 
✅ Pydantic v2 안전한 사용
✅ 실제 프로젝트 구조 기반
✅ 순환참조 완전 방지
✅ 간소화된 validation
✅ 타입 안전성 보장
✅ 프론트엔드 100% 호환

Author: MyCloset AI Team
Date: 2025-07-23
Version: 7.0 (Clean & Production Ready)
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
from enum import Enum

# Pydantic v2 imports (conda 환경 안전)
try:
    from pydantic import BaseModel, Field, field_validator, ConfigDict
    PYDANTIC_AVAILABLE = True
except ImportError:
    # 폴백: 기본 클래스만 제공
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    def Field(default=None, **kwargs):
        return default
    
    def field_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def ConfigDict(**kwargs):
        return {}
    
    PYDANTIC_AVAILABLE = False

logger = logging.getLogger(__name__)

# ==============================================
# 🔥 1. 기본 열거형 (간소화)
# ==============================================

class DeviceType(str, Enum):
    """디바이스 타입"""
    AUTO = "auto"
    CPU = "cpu" 
    CUDA = "cuda"
    MPS = "mps"

class ProcessingStatus(str, Enum):
    """처리 상태"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

class ClothingType(str, Enum):
    """의류 타입"""
    SHIRT = "shirt"
    PANTS = "pants"
    DRESS = "dress"
    JACKET = "jacket"

class QualityLevel(str, Enum):
    """품질 레벨"""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"

# ==============================================
# 🔥 2. 기본 모델 클래스
# ==============================================

class BaseConfigModel(BaseModel):
    """기본 설정 모델 - conda 환경 최적화"""
    if PYDANTIC_AVAILABLE:
        model_config = ConfigDict(
            str_strip_whitespace=True,
            validate_default=True,
            extra="forbid"
        )

# ==============================================
# 🔥 3. 핵심 데이터 모델들
# ==============================================

class BodyMeasurements(BaseConfigModel):
    """신체 측정값 - 간소화된 안전한 버전"""
    height: float = Field(..., ge=100, le=250, description="키 (cm)")
    weight: float = Field(..., ge=30, le=300, description="몸무게 (kg)")
    chest: Optional[float] = Field(None, ge=0, le=150, description="가슴둘레 (cm)")
    waist: Optional[float] = Field(None, ge=0, le=150, description="허리둘레 (cm)")
    hips: Optional[float] = Field(None, ge=0, le=150, description="엉덩이둘레 (cm)")
    
    @field_validator('height', 'weight', 'chest', 'waist', 'hips', mode='before')
    @classmethod
    def validate_numbers(cls, v):
        """안전한 숫자 검증"""
        if v is None:
            return v
        try:
            if isinstance(v, str):
                v = v.strip()
                if not v or v.lower() in ['none', 'null']:
                    return None
                v = float(v)
            return float(v) if isinstance(v, (int, float)) else None
        except (ValueError, TypeError):
            return None
    
    @property
    def bmi(self) -> float:
        """BMI 계산"""
        try:
            if self.height > 0 and self.weight > 0:
                height_m = self.height / 100.0
                return round(self.weight / (height_m ** 2), 2)
            return 0.0
        except:
            return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        try:
            if PYDANTIC_AVAILABLE:
                data = self.model_dump(exclude_none=True)
            else:
                data = self.__dict__.copy()
            data["bmi"] = self.bmi
            return data
        except:
            return {"height": self.height, "weight": self.weight}

class ImageMetadata(BaseConfigModel):
    """이미지 메타데이터"""
    filename: str = Field(..., description="파일명")
    width: int = Field(..., ge=1, description="너비")
    height: int = Field(..., ge=1, description="높이")
    format: str = Field(default="jpeg", description="포맷")
    size_bytes: int = Field(default=0, ge=0, description="파일 크기")

# ==============================================
# 🔥 4. API 응답 모델들
# ==============================================

class APIResponse(BaseConfigModel):
    """표준 API 응답 - 프론트엔드 호환"""
    success: bool = Field(..., description="성공 여부")
    message: str = Field(default="", description="메시지")
    processing_time: float = Field(default=0.0, ge=0, description="처리 시간")
    confidence: float = Field(default=1.0, ge=0, le=1, description="신뢰도")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # 추가 필드들
    session_id: Optional[str] = Field(None, description="세션 ID")
    error: Optional[str] = Field(None, description="에러 메시지")
    fitted_image: Optional[str] = Field(None, description="결과 이미지")

class StepResult(APIResponse):
    """Step 처리 결과 - main.py 호환"""
    step_id: int = Field(..., ge=1, le=8, description="단계 ID")
    step_name: str = Field(..., description="단계명")
    output_data: Dict[str, Any] = Field(default_factory=dict, description="출력 데이터")

class ErrorResponse(BaseConfigModel):
    """에러 응답"""
    success: bool = Field(default=False)
    error_code: str = Field(..., description="에러 코드")
    error_message: str = Field(..., description="에러 메시지")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# ==============================================
# 🔥 5. 가상 피팅 관련 모델들
# ==============================================

class TryOnRequest(BaseConfigModel):
    """가상 피팅 요청 - 프론트엔드 호환"""
    person_image: str = Field(..., description="사용자 이미지")
    clothing_image: str = Field(..., description="의류 이미지")
    clothing_type: ClothingType = Field(default=ClothingType.SHIRT)
    measurements: Optional[BodyMeasurements] = Field(None)
    quality_level: QualityLevel = Field(default=QualityLevel.BALANCED)
    session_id: Optional[str] = Field(None)

class TryOnResult(APIResponse):
    """가상 피팅 결과 - main.py 호환"""
    fitted_image: Optional[str] = Field(None, description="피팅 결과 이미지")
    fit_score: float = Field(default=0.0, ge=0, le=1, description="피팅 점수")
    measurements_analysis: Optional[BodyMeasurements] = Field(None)
    step_results: Optional[List[StepResult]] = Field(None)

# ==============================================
# 🔥 6. 시스템 관련 모델들
# ==============================================

class SystemInfo(BaseConfigModel):
    """시스템 정보 - main.py 호환"""
    status: str = Field(default="healthy")
    device: str = Field(default="mps")
    memory_gb: float = Field(default=128.0, description="메모리 (GB)")
    models_loaded: int = Field(default=0, description="로드된 모델 수")
    is_m3_max: bool = Field(default=True)
    conda_env: Optional[str] = Field(None)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class HealthCheck(BaseConfigModel):
    """헬스체크 응답"""
    status: str = Field(default="ok")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    system_info: Optional[SystemInfo] = Field(None)

# ==============================================
# 🔥 7. AI 모델 관련 (간소화)
# ==============================================

class ModelRequest(BaseConfigModel):
    """AI 모델 요청"""
    model_name: str = Field(..., description="모델명")
    step_class: str = Field(..., description="Step 클래스")
    input_size: Tuple[int, int] = Field(default=(512, 512), description="입력 크기")
    device: str = Field(default="mps")
    
    @field_validator('input_size', mode='before')
    @classmethod
    def validate_input_size(cls, v):
        """input_size 안전 검증"""
        try:
            if isinstance(v, (tuple, list)) and len(v) >= 2:
                return (int(v[0]), int(v[1]))
            elif isinstance(v, str):
                if ',' in v:
                    parts = v.replace('(', '').replace(')', '').split(',')
                    return (int(parts[0].strip()), int(parts[1].strip()))
                elif 'x' in v.lower():
                    parts = v.lower().split('x')
                    return (int(parts[0].strip()), int(parts[1].strip()))
            elif isinstance(v, int):
                return (v, v)
            return (512, 512)
        except:
            return (512, 512)

class ProcessingOptions(BaseConfigModel):
    """처리 옵션"""
    quality_level: QualityLevel = Field(default=QualityLevel.BALANCED)
    device_type: DeviceType = Field(default=DeviceType.MPS)
    enable_optimization: bool = Field(default=True)
    timeout_seconds: int = Field(default=300, ge=30, le=1800)

# ==============================================
# 🔥 8. 유틸리티 함수들
# ==============================================

def create_standard_response(
    success: bool = True,
    message: str = "",
    processing_time: float = 0.0,
    confidence: float = 1.0,
    **kwargs
) -> APIResponse:
    """표준 API 응답 생성"""
    try:
        return APIResponse(
            success=success,
            message=message,
            processing_time=processing_time,
            confidence=confidence,
            **kwargs
        )
    except Exception as e:
        logger.error(f"응답 생성 실패: {e}")
        return APIResponse(
            success=False,
            message=f"응답 생성 실패: {str(e)}"
        )

def create_error_response(
    error_code: str,
    error_message: str,
    **kwargs
) -> ErrorResponse:
    """에러 응답 생성"""
    try:
        return ErrorResponse(
            error_code=error_code,
            error_message=error_message,
            **kwargs
        )
    except Exception as e:
        logger.error(f"에러 응답 생성 실패: {e}")
        return ErrorResponse(
            error_code="INTERNAL_ERROR",
            error_message=f"에러 응답 생성 실패: {str(e)}"
        )

def create_processing_steps() -> List[Dict[str, Any]]:
    """8단계 처리 과정 정의"""
    return [
        {"id": 1, "name": "인체 파싱", "description": "신체 부위 분할"},
        {"id": 2, "name": "포즈 추정", "description": "신체 포즈 감지"},
        {"id": 3, "name": "의류 분할", "description": "의류 영역 분할"},
        {"id": 4, "name": "기하학적 매칭", "description": "의류-신체 매칭"},
        {"id": 5, "name": "의류 변형", "description": "의류 워핑"},
        {"id": 6, "name": "가상 피팅", "description": "최종 합성"},
        {"id": 7, "name": "후처리", "description": "품질 향상"},
        {"id": 8, "name": "품질 평가", "description": "결과 분석"}
    ]

# ==============================================
# 🔥 9. Step별 모델 정의 (간소화)
# ==============================================

STEP_MODEL_MAPPING = {
    "HumanParsingStep": ModelRequest(
        model_name="human_parsing_schp",
        step_class="HumanParsingStep",
        input_size=(512, 512)
    ),
    "PoseEstimationStep": ModelRequest(
        model_name="pose_estimation_openpose", 
        step_class="PoseEstimationStep",
        input_size=(384, 512)
    ),
    "ClothSegmentationStep": ModelRequest(
        model_name="cloth_segmentation_u2net",
        step_class="ClothSegmentationStep", 
        input_size=(320, 320)
    ),
    "GeometricMatchingStep": ModelRequest(
        model_name="geometric_matching_gm",
        step_class="GeometricMatchingStep",
        input_size=(256, 192)
    ),
    "ClothWarpingStep": ModelRequest(
        model_name="cloth_warping_flow",
        step_class="ClothWarpingStep",
        input_size=(256, 192)
    ),
    "VirtualFittingStep": ModelRequest(
        model_name="virtual_fitting_hrviton",
        step_class="VirtualFittingStep",
        input_size=(512, 384)
    ),
    "PostProcessingStep": ModelRequest(
        model_name="post_processing_enhancement",
        step_class="PostProcessingStep",
        input_size=(512, 512)
    ),
    "QualityAssessmentStep": ModelRequest(
        model_name="quality_assessment_metric",
        step_class="QualityAssessmentStep",
        input_size=(256, 256)
    )
}

def get_step_model_request(step_class: str) -> Optional[ModelRequest]:
    """Step별 모델 요청 정보 반환"""
    return STEP_MODEL_MAPPING.get(step_class)

# ==============================================
# 🔥 10. 호환성 별칭들 (기존 코드 지원)
# ==============================================

# main.py 호환
StandardAPIResponse = APIResponse
VirtualTryOnRequest = TryOnRequest
VirtualTryOnResponse = TryOnResult
AISystemStatus = SystemInfo
PipelineProgress = StepResult
QualityMetrics = Dict[str, float]
SystemStats = SystemInfo

# 열거형 별칭
DeviceTypeEnum = DeviceType
ProcessingStatusEnum = ProcessingStatus
QualityLevelEnum = QualityLevel
ClothingTypeEnum = ClothingType

# ==============================================
# 🔥 11. Export 및 검증
# ==============================================

def validate_schemas() -> bool:
    """스키마 검증"""
    try:
        # 기본 테스트
        measurements = BodyMeasurements(height=170, weight=65)
        assert measurements.bmi > 0
        
        response = create_standard_response(success=True, message="테스트")
        assert response.success
        
        logger.info("✅ 스키마 검증 성공")
        return True
    except Exception as e:
        logger.error(f"❌ 스키마 검증 실패: {e}")
        return False

# 모든 클래스 및 함수 Export
__all__ = [
    # 핵심 모델들
    'BaseConfigModel',
    'BodyMeasurements',
    'APIResponse',
    'StepResult',
    'ErrorResponse',
    
    # 가상 피팅 관련
    'TryOnRequest',
    'TryOnResult',
    
    # 시스템 관련
    'SystemInfo',
    'HealthCheck',
    
    # AI 모델 관련
    'ModelRequest',
    'ProcessingOptions',
    'ImageMetadata',
    
    # 열거형들
    'DeviceType',
    'ProcessingStatus',
    'ClothingType',
    'QualityLevel',
    
    # 유틸리티 함수들
    'create_standard_response',
    'create_error_response',
    'create_processing_steps',
    'get_step_model_request',
    'validate_schemas',
    
    # 호환성 별칭들
    'StandardAPIResponse',
    'VirtualTryOnRequest', 
    'VirtualTryOnResponse',
    'AISystemStatus',
    'PipelineProgress',
    'QualityMetrics',
    'SystemStats',
    'DeviceTypeEnum',
    'ProcessingStatusEnum',
    'QualityLevelEnum',
    'ClothingTypeEnum',
    
    # 상수들
    'STEP_MODEL_MAPPING',
    'PYDANTIC_AVAILABLE'
]

# ==============================================
# 🔥 12. 모듈 초기화 및 로깅
# ==============================================

# 환경 감지
IS_CONDA = 'CONDA_DEFAULT_ENV' in os.environ
IS_M3_MAX = 'arm64' in os.uname().machine if hasattr(os, 'uname') else False

logger.info("🔥 MyCloset AI 스키마 v7.0 - 올바른 설계 버전!")
logger.info(f"✅ Pydantic: {'사용 가능' if PYDANTIC_AVAILABLE else '폴백 모드'}")
logger.info(f"🐍 conda 환경: {'활성' if IS_CONDA else '비활성'}")
logger.info(f"🍎 M3 Max: {'지원' if IS_M3_MAX else '일반'}")
logger.info(f"📦 Export 클래스: {len(__all__)}개")

# 자동 검증 실행
if __name__ == "__main__":
    validation_result = validate_schemas()
    print(f"🔍 스키마 검증: {'✅ 성공' if validation_result else '❌ 실패'}")
    print("🚀 MyCloset AI 스키마 시스템 준비 완료!")