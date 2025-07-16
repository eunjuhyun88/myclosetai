"""
MyCloset AI - 완전한 Pydantic V2 스키마 정의 (완전 수정판)
✅ StepResult 클래스 추가 (step_routes.py 호환)
✅ Pydantic V2 완전 호환
✅ 모든 필요한 스키마 클래스 포함
✅ M3 Max 최적화 설정 및 메트릭
✅ 프론트엔드와 완전 호환
✅ pipeline_routes.py 완전 지원
✅ 모든 기능 포함
✅ FastAPI Form import 오류 해결
✅ 실제 구조에 맞춘 설계
"""

import base64
import json
import time
from typing import Dict, Any, Optional, List, Union, Annotated
from datetime import datetime
from enum import Enum

# 🔥 FIXED: FastAPI 필수 import 추가 + Optional 명시적 import
from fastapi import Form, File, UploadFile, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pydantic.functional_validators import AfterValidator

# ========================
# M3 Max 최적화 설정
# ========================

class M3MaxConfig:
    """M3 Max 128GB 환경 최적화 설정"""
    MEMORY_TOTAL = 128 * 1024**3  # 128GB
    MEMORY_AVAILABLE = int(MEMORY_TOTAL * 0.8)  # 80% 사용 가능
    MAX_BATCH_SIZE = 8  # 대용량 메모리 활용
    OPTIMAL_RESOLUTION = (1024, 1024)  # M3 Max 최적 해상도
    ULTRA_RESOLUTION = (2048, 2048)   # M3 Max 울트라 해상도
    MPS_OPTIMIZATION = True
    PARALLEL_PROCESSING = True
    NEURAL_ENGINE = True

# ========================
# 열거형 정의 (Pydantic V2 호환)
# ========================

class ProcessingStatusEnum(str, Enum):
    """처리 상태"""
    INITIALIZED = "initialized"
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL_SUCCESS = "partial_success"
    ERROR = "error"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ClothingTypeEnum(str, Enum):
    """의류 타입 (확장)"""
    SHIRT = "shirt"
    T_SHIRT = "t-shirt" 
    BLOUSE = "blouse"
    PANTS = "pants"
    JEANS = "jeans"
    DRESS = "dress"
    JACKET = "jacket"
    COAT = "coat"
    SKIRT = "skirt"
    SHORTS = "shorts"
    SWEATER = "sweater"
    HOODIE = "hoodie"
    SUIT = "suit"
    VEST = "vest"
    TANK_TOP = "tank_top"
    CARDIGAN = "cardigan"

class FabricTypeEnum(str, Enum):
    """원단 타입 (확장)"""
    COTTON = "cotton"
    DENIM = "denim"
    SILK = "silk"
    POLYESTER = "polyester"
    WOOL = "wool"
    LINEN = "linen"
    LEATHER = "leather"
    KNIT = "knit"
    CHIFFON = "chiffon"
    VELVET = "velvet"
    CASHMERE = "cashmere"
    SPANDEX = "spandex"
    NYLON = "nylon"

class QualityLevelEnum(str, Enum):
    """품질 레벨 (M3 Max 최적화)"""
    FAST = "fast"      # 빠른 처리 (512px, 5-10초)
    BALANCED = "balanced"  # 균형 (768px, 10-20초)
    HIGH = "high"      # 고품질 (1024px, 20-40초)
    ULTRA = "ultra"    # 최고품질 (2048px, 40-80초) - M3 Max 전용
    M3_OPTIMIZED = "m3_optimized"  # M3 Max 특화 모드

class QualityGradeEnum(str, Enum):
    """품질 등급"""
    EXCELLENT_PLUS = "Excellent+"  # M3 Max 울트라
    EXCELLENT = "Excellent"
    GOOD = "Good"
    ACCEPTABLE = "Acceptable"
    POOR = "Poor"
    VERY_POOR = "Very Poor"
    ERROR = "Error"

class StylePreferenceEnum(str, Enum):
    """스타일 선호도"""
    CASUAL = "casual"
    FORMAL = "formal"
    SPORTY = "sporty"
    VINTAGE = "vintage"
    MODERN = "modern"
    TRENDY = "trendy"
    CLASSIC = "classic"
    BOHEMIAN = "bohemian"
    MINIMALIST = "minimalist"
    ROMANTIC = "romantic"

class DeviceTypeEnum(str, Enum):
    """디바이스 타입"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"

class ProcessingModeEnum(str, Enum):
    """처리 모드"""
    SIMULATION = "simulation"
    PRODUCTION = "production"
    HYBRID = "hybrid"
    DEVELOPMENT = "development"
    M3_MAX_OPTIMIZED = "m3_max_optimized"

class ProcessingStage(str, Enum):
    """처리 단계"""
    UPLOAD_VALIDATION = "upload_validation"
    MEASUREMENTS_VALIDATION = "measurements_validation"
    HUMAN_PARSING = "human_parsing"
    POSE_ESTIMATION = "pose_estimation"
    CLOTH_SEGMENTATION = "cloth_segmentation"
    GEOMETRIC_MATCHING = "geometric_matching"
    CLOTH_WARPING = "cloth_warping"
    VIRTUAL_FITTING = "virtual_fitting"
    POST_PROCESSING = "post_processing"
    QUALITY_ASSESSMENT = "quality_assessment"

# ========================
# 유효성 검증 함수들 (Pydantic V2 방식)
# ========================

def validate_positive_number(value: float) -> float:
    """양수 검증"""
    if value <= 0:
        raise ValueError("값은 0보다 커야 합니다")
    return value

def validate_percentage(value: float) -> float:
    """퍼센트 값 검증 (0-1)"""
    if not 0.0 <= value <= 1.0:
        raise ValueError("값은 0.0과 1.0 사이여야 합니다")
    return value

def validate_bmi(value: float) -> float:
    """BMI 검증"""
    if not 10.0 <= value <= 50.0:
        raise ValueError("BMI는 10.0과 50.0 사이여야 합니다")
    return value

def validate_image_data(value: str) -> str:
    """Base64 이미지 데이터 검증"""
    if value.startswith('data:image/'):
        try:
            # data:image/jpeg;base64,... 형식 검증
            header, data = value.split(',', 1)
            base64.b64decode(data)
            return value
        except Exception:
            raise ValueError("잘못된 이미지 데이터 형식입니다")
    else:
        raise ValueError("이미지 데이터는 data:image/ 로 시작해야 합니다")

def validate_rgb_color(value: List[int]) -> List[int]:
    """RGB 색상 값 검증"""
    if len(value) != 3:
        raise ValueError("RGB 값은 정확히 3개여야 합니다")
    
    for color_value in value:
        if not 0 <= color_value <= 255:
            raise ValueError("RGB 값은 0-255 사이여야 합니다")
    
    return value

# 타입 별칭 정의
PositiveFloat = Annotated[float, AfterValidator(validate_positive_number)]
PercentageFloat = Annotated[float, AfterValidator(validate_percentage)]
BMIFloat = Annotated[float, AfterValidator(validate_bmi)]
ImageDataStr = Annotated[str, AfterValidator(validate_image_data)]
RGBColor = Annotated[List[int], AfterValidator(validate_rgb_color)]

# ========================
# 기본 모델들 (Pydantic V2 호환)
# ========================

class BaseConfigModel(BaseModel):
    """기본 설정 모델 (V2 호환)"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid',
        frozen=False,
        # 🔥 FIXED: model_ 네임스페이스 충돌 해결
        protected_namespaces=()
    )

class BodyMeasurements(BaseConfigModel):
    """신체 치수 정보 (M3 Max 최적화)"""
    height: PositiveFloat = Field(..., ge=140, le=220, description="키 (cm)")
    weight: PositiveFloat = Field(..., ge=30, le=150, description="체중 (kg)")
    chest: Optional[PositiveFloat] = Field(None, ge=60, le=150, description="가슴둘레 (cm)")
    waist: Optional[PositiveFloat] = Field(None, ge=50, le=120, description="허리둘레 (cm)")
    hip: Optional[PositiveFloat] = Field(None, ge=70, le=150, description="엉덩이둘레 (cm)")
    shoulder_width: Optional[PositiveFloat] = Field(None, ge=30, le=60, description="어깨너비 (cm)")
    arm_length: Optional[PositiveFloat] = Field(None, ge=50, le=90, description="팔길이 (cm)")
    leg_length: Optional[PositiveFloat] = Field(None, ge=60, le=120, description="다리길이 (cm)")
    neck: Optional[PositiveFloat] = Field(None, ge=25, le=50, description="목둘레 (cm)")
    
    @field_validator('height')
    @classmethod
    def validate_height_range(cls, v: float) -> float:
        """키 범위 검증"""
        if not 140 <= v <= 220:
            raise ValueError('키는 140cm와 220cm 사이여야 합니다')
        return v
    
    @field_validator('weight')
    @classmethod
    def validate_weight_range(cls, v: float) -> float:
        """체중 범위 검증"""
        if not 30 <= v <= 150:
            raise ValueError('체중은 30kg과 150kg 사이여야 합니다')
        return v
    
    @model_validator(mode='after')
    def validate_proportions(self):
        """신체 비율 검증"""
        if self.chest and self.waist:
            if self.chest <= self.waist:
                raise ValueError('가슴둘레는 허리둘레보다 커야 합니다')
        
        if self.hip and self.waist:
            if self.hip <= self.waist:
                raise ValueError('엉덩이둘레는 허리둘레보다 커야 합니다')
        
        return self
    
    @property
    def bmi(self) -> float:
        """BMI 계산"""
        return self.weight / ((self.height / 100) ** 2)
    
    @property
    def body_type(self) -> str:
        """체형 분류"""
        bmi = self.bmi
        if bmi < 18.5:
            return "underweight"
        elif bmi < 25:
            return "normal"
        elif bmi < 30:
            return "overweight"
        else:
            return "obese"

class StylePreferences(BaseConfigModel):
    """스타일 선호도 (확장)"""
    style: StylePreferenceEnum = Field(StylePreferenceEnum.CASUAL, description="전체 스타일")
    fit: str = Field("regular", description="핏 선호도: slim, regular, loose, oversized")
    color_preference: str = Field("original", description="색상 선호도")
    pattern_preference: str = Field("any", description="패턴 선호도")
    formality_level: int = Field(5, ge=1, le=10, description="격식도 (1=매우 캐주얼, 10=매우 포멀)")
    season_preference: Optional[str] = Field(None, description="계절 선호도")
    brand_preference: Optional[str] = Field(None, description="브랜드 선호도")
    
    @field_validator('fit')
    @classmethod
    def validate_fit(cls, v: str) -> str:
        """핏 유효성 검증"""
        valid_fits = ["slim", "regular", "loose", "oversized", "athletic", "relaxed"]
        if v.lower() not in valid_fits:
            raise ValueError(f'핏은 다음 중 하나여야 합니다: {", ".join(valid_fits)}')
        return v.lower()

class ProcessingStep(BaseConfigModel):
    """처리 단계 정보 (프론트엔드 호환)"""
    id: str = Field(..., description="단계 ID")
    name: str = Field(..., description="단계 이름")
    status: str = Field("pending", description="상태: pending, processing, completed, error")
    description: str = Field(..., description="단계 설명")
    progress: int = Field(0, ge=0, le=100, description="진행률 (%)")
    error_message: Optional[str] = Field(None, description="오류 메시지")
    processing_time: Optional[float] = Field(None, description="처리 시간 (초)")
    memory_usage: Optional[float] = Field(None, description="메모리 사용량 (GB)")
    device_info: Optional[str] = Field(None, description="처리 디바이스")
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v: str) -> str:
        """상태 유효성 검증"""
        valid_statuses = ["pending", "processing", "completed", "error", "skipped", "cancelled"]
        if v not in valid_statuses:
            raise ValueError(f'상태는 다음 중 하나여야 합니다: {", ".join(valid_statuses)}')
        return v

# ========================
# 🔥 FIXED: step_routes.py 호환을 위한 StepResult 클래스 추가
# ========================

class StepResult(BaseConfigModel):
    """단계 처리 결과 (step_routes.py 호환)"""
    step_id: str = Field(..., description="단계 ID")
    step_name: str = Field(..., description="단계 이름")
    success: bool = Field(..., description="성공 여부")
    processing_time: float = Field(..., description="처리 시간 (초)")
    memory_used: Optional[float] = Field(None, description="메모리 사용량 (GB)")
    device_used: str = Field("mps", description="사용된 디바이스")
    
    # 결과 데이터
    result_data: Optional[Dict[str, Any]] = Field(None, description="단계 결과 데이터")
    confidence: Optional[float] = Field(None, description="결과 신뢰도")
    quality_score: Optional[float] = Field(None, description="품질 점수")
    
    # 에러 정보
    error_message: Optional[str] = Field(None, description="오류 메시지")
    error_type: Optional[str] = Field(None, description="오류 타입")
    
    # 메타데이터
    metadata: Dict[str, Any] = Field(default_factory=dict, description="메타데이터")
    intermediate_files: List[str] = Field(default_factory=list, description="중간 파일 경로")
    
    @model_validator(mode='after')
    def validate_result(self):
        """결과 유효성 검증"""
        if not self.success and not self.error_message:
            raise ValueError("실패한 단계는 오류 메시지가 필요합니다")
        return self

class StepFormData(BaseConfigModel):
    """Step Routes에서 사용하는 Form 데이터"""
    height: float = Field(..., description="키 (cm)")
    weight: float = Field(..., description="몸무게 (kg)")
    session_id: str = Field(..., description="세션 ID")
    fit_score: float = Field(..., description="핏 점수")
    confidence: float = Field(..., description="신뢰도")
    fitted_image_base64: str = Field(..., description="피팅된 이미지 Base64")

# ========================
# 요청 모델들 (M3 Max 최적화)
# ========================

class VirtualTryOnRequest(BaseConfigModel):
    """가상피팅 요청 (M3 Max 최적화)"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "height": 170.0,
                "weight": 65.0,
                "clothing_type": "shirt",
                "quality_mode": "high",
                "enable_realtime": True
            }
        },
        protected_namespaces=()
    )
    
    # 이미지 데이터
    person_image_data: Optional[ImageDataStr] = Field(None, description="사용자 이미지 (base64)")
    clothing_image_data: Optional[ImageDataStr] = Field(None, description="의류 이미지 (base64)")
    person_image_url: Optional[str] = Field(None, description="사용자 이미지 URL")
    clothing_image_url: Optional[str] = Field(None, description="의류 이미지 URL")
    
    # 기본 정보
    clothing_type: ClothingTypeEnum = Field(..., description="의류 타입")
    fabric_type: FabricTypeEnum = Field(FabricTypeEnum.COTTON, description="원단 타입")
    height: float = Field(170.0, description="키 (cm)")
    weight: float = Field(65.0, description="몸무게 (kg)")
    
    # 처리 옵션
    quality_mode: QualityLevelEnum = Field(QualityLevelEnum.HIGH, description="품질 모드")
    quality_target: PercentageFloat = Field(0.8, description="목표 품질 점수")
    enable_realtime: bool = Field(True, description="실시간 상태 업데이트")
    session_id: Optional[str] = Field(None, description="세션 ID")
    save_intermediate: bool = Field(False, description="중간 결과 저장")
    enable_auto_retry: bool = Field(True, description="자동 재시도")
    
    @model_validator(mode='after')
    def validate_image_input(self):
        """이미지 입력 검증"""
        person_sources = [self.person_image_data, self.person_image_url]
        clothing_sources = [self.clothing_image_data, self.clothing_image_url]
        
        if not any(person_sources):
            raise ValueError('사용자 이미지가 필요합니다 (person_image_data 또는 person_image_url)')
        
        if not any(clothing_sources):
            raise ValueError('의류 이미지가 필요합니다 (clothing_image_data 또는 clothing_image_url)')
        
        return self

# ========================
# 응답 모델들 (확장 및 최적화)
# ========================

class MeasurementResults(BaseConfigModel):
    """측정 결과 (확장)"""
    chest: PositiveFloat = Field(..., description="가슴둘레 (cm)")
    waist: PositiveFloat = Field(..., description="허리둘레 (cm)")
    hip: PositiveFloat = Field(..., description="엉덩이둘레 (cm)")
    bmi: BMIFloat = Field(..., description="BMI")
    body_type: str = Field(..., description="체형 분류")
    shoulder_width: Optional[PositiveFloat] = Field(None, description="어깨너비 (cm)")
    confidence: PercentageFloat = Field(0.8, description="측정 신뢰도")
    measurement_method: str = Field("ai_estimation", description="측정 방법")

class ClothingAnalysis(BaseConfigModel):
    """의류 분석 결과 (확장)"""
    category: str = Field(..., description="의류 카테고리")
    style: str = Field(..., description="스타일")
    dominant_color: RGBColor = Field(..., description="주요 색상 [R, G, B]")
    fabric_type: Optional[str] = Field(None, description="원단 타입")
    pattern: Optional[str] = Field(None, description="패턴")
    season: Optional[str] = Field(None, description="계절감")
    formality: Optional[str] = Field(None, description="격식도")

class QualityMetrics(BaseConfigModel):
    """품질 메트릭 (M3 Max 최적화)"""
    overall_score: PercentageFloat = Field(..., description="전체 품질 점수")
    quality_grade: QualityGradeEnum = Field(..., description="품질 등급")
    confidence: PercentageFloat = Field(..., description="신뢰도")
    breakdown: Dict[str, float] = Field(default_factory=dict, description="세부 품질 분석")
    fit_quality: PercentageFloat = Field(0.8, description="핏 품질")
    processing_quality: PercentageFloat = Field(..., description="처리 품질")
    realism_score: PercentageFloat = Field(..., description="현실감")

class VirtualTryOnResponse(BaseConfigModel):
    """가상피팅 응답 - 프론트엔드 완전 호환"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "session_id": "session_123",
                "fitted_image": "base64_encoded_result",
                "processing_time": 25.5,
                "quality_score": 0.89
            }
        },
        protected_namespaces=()
    )
    
    success: bool = Field(..., description="성공 여부")
    session_id: Optional[str] = Field(None, description="세션 ID")
    status: str = Field(..., description="상태")
    message: str = Field(..., description="메시지")
    device_info: str = Field("M3 Max", description="디바이스 정보")
    
    # 결과 데이터
    fitted_image: Optional[str] = Field(None, description="결과 이미지 (base64)")
    processing_time: Optional[float] = Field(None, description="처리 시간 (초)")
    quality_score: Optional[float] = Field(None, description="품질 점수")
    confidence: Optional[float] = Field(None, description="신뢰도")
    
    # 에러 정보
    error: Optional[str] = Field(None, description="오류 메시지")
    error_type: Optional[str] = Field(None, description="오류 타입")
    
    # 추가 정보
    recommendations: List[str] = Field(default_factory=list, description="추천사항")
    tips: List[str] = Field(default_factory=list, description="사용자 팁")

# ========================
# 시스템 상태 모델들
# ========================

class SystemHealth(BaseConfigModel):
    """시스템 건강 상태"""
    overall_status: str = Field(..., description="전체 상태: healthy, degraded, unhealthy")
    pipeline_initialized: bool = Field(..., description="파이프라인 초기화 상태")
    device_available: bool = Field(..., description="디바이스 사용 가능 여부")
    memory_usage: Dict[str, str] = Field(default_factory=dict, description="메모리 사용량")
    active_sessions: int = Field(0, ge=0, description="활성 세션 수")
    uptime: PositiveFloat = Field(..., description="가동 시간 (초)")
    
    # M3 Max 전용 상태
    mps_available: bool = Field(False, description="MPS 사용 가능 여부")
    neural_engine_available: bool = Field(False, description="Neural Engine 사용 가능 여부")

class HealthCheckResponse(BaseConfigModel):
    """헬스체크 응답"""
    status: str = Field(..., description="서비스 상태")
    timestamp: str = Field(..., description="확인 시간")
    version: str = Field(..., description="버전")
    device: str = Field(..., description="디바이스")
    uptime: float = Field(..., description="가동 시간")
    pipeline_ready: bool = Field(..., description="파이프라인 준비 상태")
    m3_max_optimized: bool = Field(False, description="M3 Max 최적화")

# ========================
# 에러 모델들
# ========================

class ErrorDetail(BaseConfigModel):
    """에러 상세 정보"""
    error_code: str = Field(..., description="오류 코드")
    error_message: str = Field(..., description="오류 메시지")
    error_type: str = Field(..., description="오류 타입")
    step_number: Optional[int] = Field(None, ge=1, le=10, description="오류 발생 단계")
    suggestions: List[str] = Field(default_factory=list, description="해결 제안")

class ErrorResponse(BaseConfigModel):
    """에러 응답"""
    success: bool = Field(False, description="성공 여부")
    error: ErrorDetail = Field(..., description="오류 상세")
    timestamp: str = Field(..., description="오류 시간")
    session_id: Optional[str] = Field(None, description="세션 ID")

# ========================
# WebSocket 관련 스키마들
# ========================

class WebSocketMessage(BaseConfigModel):
    """WebSocket 메시지 기본 구조"""
    message_type: str = Field(..., description="메시지 타입")
    timestamp: float = Field(default_factory=time.time, description="타임스탬프")
    session_id: Optional[str] = Field(None, description="세션 ID")
    data: Optional[Dict[str, Any]] = Field(None, description="메시지 데이터")

class ProgressUpdate(BaseConfigModel):
    """진행 상황 업데이트"""
    stage: str = Field(..., description="현재 단계")
    percentage: float = Field(..., ge=0.0, le=100.0, description="진행률")
    message: Optional[str] = Field(None, description="상태 메시지")
    device: str = Field("M3 Max", description="처리 디바이스")

# ========================
# 유틸리티 함수들
# ========================

def create_processing_steps() -> List[ProcessingStep]:
    """프론트엔드용 처리 단계 생성 (M3 Max 최적화)"""
    return [
        ProcessingStep(
            id="upload_validation",
            name="이미지 업로드 검증",
            status="pending",
            description="이미지를 업로드하고 M3 Max 최적화 검증을 수행합니다"
        ),
        ProcessingStep(
            id="measurements_validation",
            name="신체 측정값 검증",
            status="pending", 
            description="신체 측정값 검증 및 BMI 계산을 수행합니다"
        ),
        ProcessingStep(
            id="human_parsing",
            name="인체 분석 (20개 부위)",
            status="pending",
            description="M3 Max Neural Engine을 활용한 고정밀 인체 분석"
        ),
        ProcessingStep(
            id="pose_estimation",
            name="포즈 추정 (18개 키포인트)",
            status="pending",
            description="MPS 최적화된 실시간 포즈 분석"
        ),
        ProcessingStep(
            id="cloth_segmentation", 
            name="의류 분석 및 세그멘테이션",
            status="pending",
            description="고해상도 의류 세그멘테이션 및 배경 제거"
        ),
        ProcessingStep(
            id="geometric_matching",
            name="기하학적 매칭",
            status="pending",
            description="M3 Max 병렬 처리를 활용한 정밀 매칭"
        ),
        ProcessingStep(
            id="cloth_warping",
            name="의류 변형 및 워핑",
            status="pending",
            description="Metal Performance Shaders를 활용한 물리 시뮬레이션"
        ),
        ProcessingStep(
            id="virtual_fitting",
            name="가상 피팅 생성",
            status="pending",
            description="128GB 메모리를 활용한 고품질 피팅 생성"
        )
    ]

def create_error_response(
    error_code: str, 
    error_message: str, 
    error_type: str = "ProcessingError",
    session_id: Optional[str] = None
) -> ErrorResponse:
    """에러 응답 생성"""
    return ErrorResponse(
        error=ErrorDetail(
            error_code=error_code,
            error_message=error_message,
            error_type=error_type,
            suggestions=["이미지 품질을 확인해 보세요", "다시 시도해 보세요"]
        ),
        timestamp=datetime.now().isoformat(),
        session_id=session_id
    )

# ========================
# Export 리스트 (완전)
# ========================

__all__ = [
    # 설정 클래스
    'M3MaxConfig',
    
    # Enum 클래스들
    'ProcessingStatusEnum',
    'ClothingTypeEnum', 
    'FabricTypeEnum',
    'QualityLevelEnum',
    'QualityGradeEnum',
    'StylePreferenceEnum',
    'DeviceTypeEnum',
    'ProcessingModeEnum',
    'ProcessingStage',
    
    # 검증 함수들
    'validate_positive_number',
    'validate_percentage', 
    'validate_bmi',
    'validate_image_data',
    'validate_rgb_color',
    
    # 타입 별칭들
    'PositiveFloat',
    'PercentageFloat',
    'BMIFloat',
    'ImageDataStr',
    'RGBColor',
    
    # 기본 모델들
    'BaseConfigModel',
    'BodyMeasurements',
    'StylePreferences',
    'ProcessingStep',
    
    # 🔥 FIXED: StepResult 클래스 추가
    'StepResult',
    'StepFormData',
    
    # 요청 모델들
    'VirtualTryOnRequest',
    
    # 응답 모델들
    'MeasurementResults',
    'ClothingAnalysis',
    'QualityMetrics',
    'VirtualTryOnResponse',
    
    # 에러 및 시스템 상태 모델들
    'ErrorDetail',
    'ErrorResponse',
    'SystemHealth',
    'HealthCheckResponse',
    
    # WebSocket 관련 모델들
    'WebSocketMessage',
    'ProgressUpdate',
    
    # 유틸리티 함수들
    'create_processing_steps',
    'create_error_response',
    
    # FastAPI 관련 (step_routes.py 호환)
    'Form',
    'File', 
    'UploadFile',
    'Depends',
    'HTTPException',
    'Request',
    'BackgroundTasks',
    'JSONResponse',
    'Optional'
]

# 모듈 로드 확인
print("🎉 MyCloset AI 완전 수정된 Pydantic V2 스키마 시스템 로드 완료!")
print("✅ StepResult 클래스 추가 - step_routes.py 호환")
print("✅ 모든 필수 스키마 클래스 포함")
print("✅ M3 Max 최적화 기능 완전 지원")
print("✅ 프론트엔드 100% 호환성 보장")
print(f"📊 총 Export 항목: {len(__all__)}개")