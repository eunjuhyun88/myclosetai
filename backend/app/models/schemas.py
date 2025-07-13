"""
MyCloset AI Backend - 통합 Pydantic 스키마 정의
프론트엔드 mycloset-uiux.tsx와 완전 호환되는 데이터 모델
기존 pipeline_manager.py와 완전 통합
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Any, Union
from enum import Enum
import json
from datetime import datetime

# ========================
# 열거형 정의 (통합 및 확장)
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

class FabricTypeEnum(str, Enum):
    """천 재질 (확장)"""
    COTTON = "cotton"
    DENIM = "denim"
    SILK = "silk"
    POLYESTER = "polyester"
    WOOL = "wool"
    LINEN = "linen"
    LEATHER = "leather"
    KNIT = "knit"
    CHIFFON = "chiffon"

class QualityLevelEnum(str, Enum):
    """품질 레벨"""
    FAST = "fast"      # 빠른 처리 (5-10초)
    MEDIUM = "medium"  # 균형잡힌 품질 (15-25초)  
    HIGH = "high"      # 고품질 (30-60초)
    ULTRA = "ultra"    # 최고품질 (60-120초)

class QualityGradeEnum(str, Enum):
    """품질 등급"""
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

# ========================
# 요청 모델 (통합 및 개선)
# ========================

class BodyMeasurements(BaseModel):
    """신체 치수 정보 (확장)"""
    height: float = Field(..., ge=140, le=220, description="키 (cm)")
    weight: float = Field(..., ge=30, le=150, description="체중 (kg)")
    chest: Optional[float] = Field(None, ge=60, le=150, description="가슴둘레 (cm)")
    waist: Optional[float] = Field(None, ge=50, le=120, description="허리둘레 (cm)")
    hip: Optional[float] = Field(None, ge=70, le=150, description="엉덩이둘레 (cm)")
    shoulder_width: Optional[float] = Field(None, ge=30, le=60, description="어깨너비 (cm)")
    arm_length: Optional[float] = Field(None, ge=50, le=90, description="팔길이 (cm)")
    leg_length: Optional[float] = Field(None, ge=60, le=120, description="다리길이 (cm)")
    
    @validator('weight', 'height')
    def validate_measurements(cls, v, field):
        if v <= 0:
            raise ValueError(f'{field.name} must be positive')
        return v
    
    @property
    def bmi(self) -> float:
        """BMI 계산"""
        return self.weight / ((self.height / 100) ** 2)
    
    def get_estimated_measurements(self) -> Dict[str, float]:
        """추정 치수 계산"""
        # 키와 체중 기반 추정
        return {
            "chest": self.chest or self.height * 0.55,
            "waist": self.waist or self.height * 0.45,
            "hip": self.hip or self.height * 0.57,
            "shoulder_width": self.shoulder_width or self.height * 0.25,
            "arm_length": self.arm_length or self.height * 0.38,
            "leg_length": self.leg_length or self.height * 0.50
        }

class StylePreferences(BaseModel):
    """스타일 선호도 (확장)"""
    style: StylePreferenceEnum = Field(StylePreferenceEnum.CASUAL, description="전체 스타일")
    fit: str = Field("regular", description="핏 선호도: slim, regular, loose")
    color_preference: str = Field("original", description="색상 선호도: original, darker, lighter, colorful")
    pattern_preference: str = Field("any", description="패턴 선호도: solid, striped, printed, any")

class VirtualTryOnRequest(BaseModel):
    """가상피팅 요청 (통합)"""
    clothing_type: ClothingTypeEnum = Field(..., description="의류 타입")
    fabric_type: FabricTypeEnum = Field(FabricTypeEnum.COTTON, description="천 재질")
    body_measurements: BodyMeasurements = Field(..., description="신체 치수")
    style_preferences: Optional[StylePreferences] = Field(None, description="스타일 선호도")
    quality_level: QualityLevelEnum = Field(QualityLevelEnum.HIGH, description="처리 품질 레벨")
    quality_target: float = Field(default=0.8, ge=0.1, le=1.0, description="목표 품질 점수")
    save_intermediate: bool = Field(default=False, description="중간 결과 저장 여부")
    save_result: bool = Field(True, description="결과 저장 여부")
    
    @validator('style_preferences', pre=True)
    def parse_style_preferences(cls, v):
        if isinstance(v, str):
            try:
                return StylePreferences.parse_raw(v) if v else StylePreferences()
            except json.JSONDecodeError:
                return StylePreferences()
        return v

    @validator('quality_target')
    def validate_quality_target(cls, v):
        if not 0.1 <= v <= 1.0:
            raise ValueError('품질 목표는 0.1 ~ 1.0 사이여야 합니다')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "clothing_type": "shirt",
                "fabric_type": "cotton",
                "body_measurements": {
                    "height": 170.0,
                    "weight": 65.0,
                    "chest": 95.0,
                    "waist": 80.0,
                    "hip": 95.0
                },
                "style_preferences": {
                    "style": "casual",
                    "fit": "regular",
                    "color_preference": "original"
                },
                "quality_level": "high",
                "quality_target": 0.8
            }
        }

# ========================
# 프론트엔드 호환 모델들
# ========================

class ProcessingStep(BaseModel):
    """프론트엔드 ProcessingStep과 완전 호환"""
    id: str
    name: str
    status: str  # 'pending' | 'processing' | 'completed' | 'error'
    description: str
    progress: int = Field(default=0, ge=0, le=100)
    error_message: Optional[str] = None
    processing_time: Optional[float] = Field(None, description="단계 처리 시간")

# ========================
# 응답 모델 (통합 및 확장)
# ========================

class MeasurementResults(BaseModel):
    """측정 결과 (확장)"""
    chest: float = Field(..., description="가슴둘레 (cm)")
    waist: float = Field(..., description="허리둘레 (cm)")
    hip: float = Field(..., description="엉덩이둘레 (cm)")
    bmi: float = Field(..., description="BMI")
    body_type: Optional[str] = Field(None, description="체형 분류")
    shoulder_width: Optional[float] = Field(None, description="어깨너비 (cm)")
    
    @validator('bmi')
    def validate_bmi(cls, v):
        if not 10 <= v <= 50:
            raise ValueError('BMI must be between 10 and 50')
        return v

class ClothingAnalysis(BaseModel):
    """의류 분석 결과 (확장)"""
    category: str = Field(..., description="의류 카테고리")
    style: str = Field(..., description="스타일")
    dominant_color: List[int] = Field(..., description="주요 색상 [R, G, B]")
    fabric_type: Optional[str] = Field(None, description="원단 타입")
    pattern: Optional[str] = Field(None, description="패턴")
    season: Optional[str] = Field(None, description="계절감")
    formality: Optional[str] = Field(None, description="격식도")
    texture: Optional[str] = Field(None, description="질감")

class FitAnalysis(BaseModel):
    """핏 분석 결과 (확장)"""
    overall_fit_score: float = Field(..., ge=0.0, le=1.0, description="전체 핏 점수")
    body_alignment: float = Field(..., ge=0.0, le=1.0, description="신체 정렬")
    garment_deformation: float = Field(..., ge=0.0, le=1.0, description="의류 변형도")
    size_compatibility: Dict[str, Any] = Field(default_factory=dict, description="사이즈 호환성")
    style_match: Dict[str, Any] = Field(default_factory=dict, description="스타일 매칭")
    comfort_level: Optional[float] = Field(None, ge=0.0, le=1.0, description="착용감")

class QualityMetrics(BaseModel):
    """품질 메트릭 (확장)"""
    overall_score: float = Field(..., ge=0.0, le=1.0, description="전체 품질 점수")
    quality_grade: QualityGradeEnum = Field(..., description="품질 등급")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도")
    breakdown: Dict[str, float] = Field(default_factory=dict, description="세부 품질 분석")
    fit_quality: float = Field(default=0.8, ge=0.0, le=1.0, description="핏 품질")
    processing_quality: float = Field(..., ge=0.0, le=1.0, description="처리 품질")
    realism_score: float = Field(..., ge=0.0, le=1.0, description="현실감")
    detail_preservation: float = Field(..., ge=0.0, le=1.0, description="디테일 보존도")
    technical_quality: Dict[str, float] = Field(default_factory=dict, description="기술적 품질")

class ProcessingStatistics(BaseModel):
    """처리 통계 (확장)"""
    total_time: float = Field(..., description="총 처리 시간 (초)")
    step_times: Dict[str, float] = Field(default_factory=dict, description="단계별 시간")
    steps_completed: int = Field(..., description="완료된 단계 수")
    total_steps: int = Field(default=8, description="전체 단계 수")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="성공률")
    device_used: str = Field(..., description="사용된 디바이스")
    memory_usage: Dict[str, str] = Field(default_factory=dict, description="메모리 사용량")
    efficiency_score: float = Field(default=0.8, ge=0.0, le=1.0, description="효율성 점수")
    optimization: str = Field(..., description="최적화 방식")
    demo_mode: bool = Field(default=False, description="데모 모드 여부")

class ImprovementSuggestions(BaseModel):
    """개선 제안 (확장)"""
    quality_improvements: List[str] = Field(default_factory=list, description="품질 개선")
    performance_optimizations: List[str] = Field(default_factory=list, description="성능 최적화")
    user_experience: List[str] = Field(default_factory=list, description="사용자 경험")
    technical_adjustments: List[str] = Field(default_factory=list, description="기술적 조정")
    style_suggestions: List[str] = Field(default_factory=list, description="스타일 제안")

class ProcessingMetadata(BaseModel):
    """처리 메타데이터 (확장)"""
    timestamp: str
    pipeline_version: str = "2.0.0"
    input_resolution: str
    output_resolution: str
    clothing_type: str
    fabric_type: str
    body_measurements_provided: bool
    style_preferences_provided: bool
    intermediate_results_saved: bool
    device_optimization: str
    memory_optimization_enabled: bool
    parallel_processing_enabled: bool
    api_version: str = "1.0.0"

class ProcessingInfo(BaseModel):
    """처리 정보 (확장)"""
    steps_completed: int = Field(..., description="완료된 단계 수")
    total_steps: int = Field(8, description="전체 단계 수")
    processing_time: float = Field(..., description="총 처리 시간 (초)")
    quality_level: QualityLevelEnum = Field(..., description="처리된 품질 레벨")
    device_used: str = Field(..., description="사용된 디바이스")
    optimization: str = Field(..., description="최적화 방식")
    demo_mode: bool = Field(False, description="데모 모드 여부")
    error_message: Optional[str] = Field(None, description="오류 메시지")
    pipeline_version: str = Field("2.0.0", description="파이프라인 버전")

class ProcessingResult(BaseModel):
    """처리 결과 - 프론트엔드와 완전 호환 (통합)"""
    # 기본 결과
    result_image_url: str = Field(..., description="결과 이미지 URL")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="품질 점수")
    quality_grade: QualityGradeEnum = Field(..., description="품질 등급")
    processing_time: float = Field(..., description="처리 시간 (초)")
    device_used: str = Field(..., description="사용된 디바이스")
    
    # 상세 분석
    fit_analysis: FitAnalysis
    quality_metrics: QualityMetrics
    processing_statistics: ProcessingStatistics
    
    # 개선 제안
    recommendations: List[str] = Field(default_factory=list, description="주요 추천사항")
    improvement_suggestions: ImprovementSuggestions
    next_steps: List[str] = Field(default_factory=list, description="다음 단계")
    
    # 메타데이터
    metadata: ProcessingMetadata
    
    # 추가 정보
    quality_target_achieved: bool = Field(..., description="목표 품질 달성 여부")
    is_fallback: bool = Field(default=False, description="폴백 결과 여부")
    fallback_reason: Optional[str] = Field(None, description="폴백 사유")
    
    # 프론트엔드 호환성을 위한 추가 필드들
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="신뢰도")
    measurements: MeasurementResults
    clothing_analysis: ClothingAnalysis
    fit_score: float = Field(default=0.8, ge=0.0, le=1.0, description="핏 점수")
    
    # 선택적 정보
    alternative_suggestions: Optional[List[str]] = Field(None, description="대안 제안")
    style_compatibility: Optional[float] = Field(None, ge=0, le=1, description="스타일 호환성")
    size_recommendation: Optional[str] = Field(None, description="사이즈 추천")

class ProcessingStatus(BaseModel):
    """처리 상태 - 프론트엔드와 완전 호환 (통합)"""
    session_id: str
    status: ProcessingStatusEnum
    progress: int = Field(..., ge=0, le=100, description="진행률 (%)")
    current_step: str = Field(default="", description="현재 단계")
    
    # 결과 정보
    result: Optional[ProcessingResult] = None
    error: Optional[str] = None
    
    # 시간 정보
    processing_time: float = Field(default=0.0, description="경과 시간 (초)")
    estimated_remaining_time: Optional[float] = Field(None, description="예상 남은 시간 (초)")
    
    # 프론트엔드 호환성을 위한 단계별 상태
    steps: List[ProcessingStep] = Field(default_factory=list, description="단계별 상태")

class VirtualTryOnResponse(BaseModel):
    """가상피팅 응답 - 프론트엔드와 완전 호환 (통합)"""
    success: bool
    session_id: Optional[str] = None
    status: str
    message: str
    
    # 처리 관련
    processing_url: Optional[str] = None
    estimated_time: Optional[int] = Field(None, description="예상 처리 시간 (초)")
    
    # 즉시 결과 (동기식인 경우) - 확장
    fitted_image: Optional[str] = Field(None, description="결과 이미지 (base64)")
    result: Optional[ProcessingResult] = None
    error: Optional[str] = None
    
    # 추가 정보
    tips: List[str] = Field(default_factory=list, description="사용자 팁")
    
    # 기존 paste.txt의 필드들 통합
    processing_time: Optional[float] = Field(None, description="처리 시간 (초)")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="신뢰도")
    measurements: Optional[MeasurementResults] = None
    clothing_analysis: Optional[ClothingAnalysis] = None
    quality_analysis: Optional[QualityMetrics] = None
    fit_score: Optional[float] = Field(None, ge=0, le=1, description="핏 점수")
    recommendations: List[str] = Field(default_factory=list, description="추천사항")
    processing_info: Optional[ProcessingInfo] = None

# ========================
# 에러 응답 모델 (확장)
# ========================

class ErrorDetail(BaseModel):
    """에러 상세 정보 (확장)"""
    error_code: str
    error_message: str
    error_type: str
    step_number: Optional[int] = Field(None, description="오류 발생 단계")
    suggestions: List[str] = Field(default_factory=list, description="해결 제안")
    retry_after: Optional[int] = None  # 재시도 권장 시간 (초)

class ErrorResponse(BaseModel):
    """에러 응답 (확장)"""
    success: bool = False
    error: ErrorDetail
    timestamp: str
    session_id: Optional[str] = None

# ========================
# 시스템 상태 모델 (확장)
# ========================

class ModelStatus(BaseModel):
    """모델 상태 (확장)"""
    model_name: str = Field(..., description="모델명")
    loaded: bool = Field(..., description="로드 상태")
    version: Optional[str] = Field(None, description="모델 버전")
    device: str = Field(..., description="모델이 로드된 디바이스")
    memory_usage: Optional[float] = Field(None, description="메모리 사용량 (GB)")
    initialization_time: Optional[float] = Field(None, description="초기화 시간")
    last_error: Optional[str] = Field(None, description="마지막 오류")

class SystemHealth(BaseModel):
    """시스템 건강 상태 (확장)"""
    overall_status: str  # healthy, degraded, unhealthy
    pipeline_initialized: bool
    device_available: bool
    memory_usage: Dict[str, str]
    active_sessions: int
    error_rate: float
    uptime: float  # 초 단위
    pipeline_ready: bool = Field(..., description="AI 파이프라인 준비 상태")

class PerformanceMetrics(BaseModel):
    """성능 메트릭 (확장)"""
    total_sessions: int = Field(..., description="총 세션 수")
    successful_sessions: int = Field(..., description="성공한 세션 수")
    average_processing_time: float = Field(..., description="평균 처리 시간")
    average_quality_score: float = Field(..., description="평균 품질 점수")
    success_rate: float = Field(..., description="성공률")
    current_load: float = Field(default=0.0, description="현재 부하")
    total_processed: int = Field(..., description="총 처리 건수")

class PipelineStatus(BaseModel):
    """파이프라인 전체 상태 (확장)"""
    initialized: bool
    device: str
    pipeline_config: Dict[str, Any]
    performance_metrics: PerformanceMetrics
    system_health: SystemHealth
    steps_status: List[ModelStatus]
    models_loaded: int = Field(..., description="로드된 모델 수")

class HealthResponse(BaseModel):
    """헬스체크 응답 (통합)"""
    status: str = Field(..., description="시스템 상태")
    timestamp: str = Field(..., description="확인 시간")
    pipeline_ready: bool = Field(..., description="AI 파이프라인 준비 상태")
    memory_status: str = Field(..., description="메모리 상태")
    active_sessions: int = Field(..., description="활성 세션 수")
    version: str = Field(..., description="API 버전")
    device: str = Field(..., description="사용 중인 디바이스")
    pipeline_status: str = Field(..., description="파이프라인 상태")

# ========================
# 세션 관리 스키마들 (확장)
# ========================

class SessionInfo(BaseModel):
    """세션 정보 (확장)"""
    session_id: str = Field(..., description="세션 ID")
    status: ProcessingStatusEnum = Field(..., description="처리 상태")
    start_time: float = Field(..., description="시작 시간 (timestamp)")
    processing_time: Optional[float] = Field(None, description="처리 시간 (초)")
    clothing_type: Optional[ClothingTypeEnum] = Field(None, description="의류 타입")
    quality_level: Optional[QualityLevelEnum] = Field(None, description="품질 레벨")
    has_result: bool = Field(..., description="결과 보유 여부")
    error_message: Optional[str] = Field(None, description="오류 메시지")

class SessionListResponse(BaseModel):
    """세션 목록 응답"""
    active_sessions: int = Field(..., description="활성 세션 수")
    sessions: List[SessionInfo] = Field(..., description="세션 목록")

# ========================
# 설정 및 환경 정보 (확장)
# ========================

class SystemConfiguration(BaseModel):
    """시스템 설정 정보 (확장)"""
    max_upload_size: int = Field(..., description="최대 업로드 크기 (bytes)")
    supported_image_formats: List[str] = Field(..., description="지원하는 이미지 포맷")
    supported_clothing_types: List[str] = Field(..., description="지원하는 의류 타입")
    quality_levels: List[str] = Field(..., description="사용 가능한 품질 레벨")
    default_quality_level: str = Field(..., description="기본 품질 레벨")
    processing_timeout: int = Field(..., description="처리 타임아웃 (초)")
    max_concurrent_sessions: int = Field(..., description="최대 동시 세션")

class EnvironmentInfo(BaseModel):
    """환경 정보 (확장)"""
    python_version: str = Field(..., description="Python 버전")
    pytorch_version: str = Field(..., description="PyTorch 버전")
    cuda_available: bool = Field(..., description="CUDA 사용 가능 여부")
    mps_available: bool = Field(..., description="MPS 사용 가능 여부")
    device_count: int = Field(..., description="사용 가능한 디바이스 수")
    memory_total: Optional[float] = Field(None, description="총 메모리 (GB)")
    api_version: str = Field(..., description="API 버전")

class APIInfo(BaseModel):
    """API 정보 (확장)"""
    name: str = Field(..., description="API 이름")
    version: str = Field(..., description="API 버전")
    description: str = Field(..., description="API 설명")
    documentation_url: str = Field(..., description="문서 URL")
    support_email: Optional[str] = Field(None, description="지원 이메일")
    
    # 기능 정보
    features: List[str] = Field(..., description="지원 기능 목록")
    limitations: List[str] = Field(..., description="제한 사항")
    
    # 환경 정보
    environment: EnvironmentInfo = Field(..., description="환경 정보")
    configuration: SystemConfiguration = Field(..., description="시스템 설정")

# ========================
# 유틸리티 함수들 (확장)
# ========================

def create_processing_steps() -> List[ProcessingStep]:
    """프론트엔드용 처리 단계 생성 (확장)"""
    return [
        ProcessingStep(
            id="upload",
            name="이미지 업로드",
            status="pending",
            description="이미지를 업로드하고 검증합니다"
        ),
        ProcessingStep(
            id="human_parsing",
            name="인체 분석",
            status="pending", 
            description="사용자의 체형을 분석합니다 (Graphonomy)"
        ),
        ProcessingStep(
            id="pose_estimation",
            name="포즈 추정",
            status="pending",
            description="신체 포즈를 분석합니다 (OpenPose/MediaPipe)"
        ),
        ProcessingStep(
            id="cloth_segmentation", 
            name="의류 분석",
            status="pending",
            description="의류를 분석하고 배경을 제거합니다 (U²-Net)"
        ),
        ProcessingStep(
            id="geometric_matching",
            name="기하학적 매칭",
            status="pending",
            description="의류와 신체를 매칭합니다 (TPS 변환)"
        ),
        ProcessingStep(
            id="cloth_warping",
            name="의류 변형",
            status="pending",
            description="의류를 신체에 맞게 변형합니다 (물리 시뮬레이션)"
        ),
        ProcessingStep(
            id="virtual_fitting",
            name="가상 피팅",
            status="pending",
            description="최종 피팅 이미지를 생성합니다 (HR-VITON)"
        ),
        ProcessingStep(
            id="post_processing",
            name="품질 향상",
            status="pending",
            description="이미지 품질을 향상시킵니다"
        ),
        ProcessingStep(
            id="quality_assessment",
            name="품질 평가",
            status="pending",
            description="최종 품질을 평가하고 점수를 산출합니다"
        )
    ]

def update_processing_step_status(
    steps: List[ProcessingStep], 
    step_id: str, 
    status: str, 
    progress: int = 0, 
    error_message: str = None,
    processing_time: float = None
) -> List[ProcessingStep]:
    """처리 단계 상태 업데이트 (확장)"""
    for step in steps:
        if step.id == step_id:
            step.status = status
            step.progress = progress
            if error_message:
                step.error_message = error_message
            if processing_time:
                step.processing_time = processing_time
            break
    return steps

def create_error_response(
    error_code: str, 
    error_message: str, 
    error_type: str = "ProcessingError",
    suggestion: str = None, 
    session_id: str = None,
    step_number: int = None
) -> ErrorResponse:
    """에러 응답 생성 (확장)"""
    suggestions = []
    if suggestion:
        suggestions.append(suggestion)
    
    return ErrorResponse(
        error=ErrorDetail(
            error_code=error_code,
            error_message=error_message,
            error_type=error_type,
            step_number=step_number,
            suggestions=suggestions
        ),
        timestamp=datetime.now().isoformat(),
        session_id=session_id
    )

def convert_pipeline_result_to_frontend(pipeline_result: Dict[str, Any], session_id: str) -> ProcessingResult:
    """pipeline_manager 결과를 프론트엔드 호환 형식으로 변환 (확장)"""
    
    # 기본 메타데이터 생성
    metadata = ProcessingMetadata(
        timestamp=pipeline_result.get('metadata', {}).get('timestamp', datetime.now().isoformat()),
        pipeline_version=pipeline_result.get('metadata', {}).get('pipeline_version', '2.0.0'),
        input_resolution=pipeline_result.get('metadata', {}).get('input_resolution', '512x512'),
        output_resolution=pipeline_result.get('metadata', {}).get('output_resolution', '512x512'),
        clothing_type=pipeline_result.get('metadata', {}).get('clothing_type', 'shirt'),
        fabric_type=pipeline_result.get('metadata', {}).get('fabric_type', 'cotton'),
        body_measurements_provided=pipeline_result.get('metadata', {}).get('body_measurements_provided', False),
        style_preferences_provided=pipeline_result.get('metadata', {}).get('style_preferences_provided', False),
        intermediate_results_saved=pipeline_result.get('metadata', {}).get('intermediate_results_saved', False),
        device_optimization=pipeline_result.get('metadata', {}).get('device_optimization', 'auto'),
        memory_optimization_enabled=pipeline_result.get('metadata', {}).get('memory_optimization_enabled', True),
        parallel_processing_enabled=pipeline_result.get('metadata', {}).get('parallel_processing_enabled', True)
    )
    
    # 측정 결과 생성
    measurements = MeasurementResults(
        chest=95.0,
        waist=80.0, 
        hip=95.0,
        bmi=22.5,
        body_type="normal"
    )
    
    # 의류 분석 생성
    clothing_analysis = ClothingAnalysis(
        category=metadata.clothing_type,
        style="casual",
        dominant_color=[128, 128, 128],
        fabric_type=metadata.fabric_type
    )
    
    # 핏 분석
    fit_analysis = FitAnalysis(
        overall_fit_score=pipeline_result.get('fit_analysis', {}).get('overall_fit_score', 0.8),
        body_alignment=pipeline_result.get('fit_analysis', {}).get('body_alignment', 0.8),
        garment_deformation=pipeline_result.get('fit_analysis', {}).get('garment_deformation', 0.8),
        size_compatibility=pipeline_result.get('fit_analysis', {}).get('size_compatibility', {}),
        style_match=pipeline_result.get('fit_analysis', {}).get('style_match', {})
    )
    
    # 품질 메트릭
    quality_metrics = QualityMetrics(
        overall_score=pipeline_result.get('final_quality_score', 0.8),
        quality_grade=QualityGradeEnum(pipeline_result.get('quality_grade', 'Good')),
        confidence=pipeline_result.get('quality_confidence', 0.8),
        breakdown=pipeline_result.get('quality_breakdown', {}),
        fit_quality=pipeline_result.get('quality_breakdown', {}).get('fit_quality', 0.8),
        processing_quality=0.9,
        realism_score=0.85,
        detail_preservation=0.88,
        technical_quality=pipeline_result.get('quality_breakdown', {}).get('technical_quality', {})
    )
    
    # 처리 통계
    processing_stats = pipeline_result.get('processing_statistics', {})
    processing_statistics = ProcessingStatistics(
        total_time=pipeline_result.get('total_processing_time', 0.0),
        step_times=processing_stats.get('step_times', {}),
        steps_completed=processing_stats.get('steps_completed', 8),
        success_rate=processing_stats.get('success_rate', 1.0),
        device_used=pipeline_result.get('device_used', 'auto'),
        memory_usage=processing_stats.get('memory_usage', {}),
        efficiency_score=processing_stats.get('efficiency_score', 0.8),
        optimization="M3_Max" if pipeline_result.get('device_used') == 'mps' else "Standard"
    )
    
    # 개선 제안
    suggestions = pipeline_result.get('improvement_suggestions', {})
    improvement_suggestions = ImprovementSuggestions(
        quality_improvements=suggestions.get('quality_improvements', []),
        performance_optimizations=suggestions.get('performance_optimizations', []),
        user_experience=suggestions.get('user_experience', []),
        technical_adjustments=suggestions.get('technical_adjustments', [])
    )
    
    return ProcessingResult(
        result_image_url=f"/static/results/{session_id}_result.jpg",
        quality_score=pipeline_result.get('final_quality_score', 0.8),
        quality_grade=QualityGradeEnum(pipeline_result.get('quality_grade', 'Good')),
        processing_time=pipeline_result.get('total_processing_time', 0.0),
        device_used=pipeline_result.get('device_used', 'auto'),
        fit_analysis=fit_analysis,
        quality_metrics=quality_metrics,
        processing_statistics=processing_statistics,
        recommendations=suggestions.get('quality_improvements', [])[:3],
        improvement_suggestions=improvement_suggestions,
        next_steps=pipeline_result.get('next_steps', []),
        metadata=metadata,
        quality_target_achieved=pipeline_result.get('quality_target_achieved', False),
        is_fallback=pipeline_result.get('fallback_used', False),
        fallback_reason=pipeline_result.get('error', None) if pipeline_result.get('fallback_used') else None,
        # 프론트엔드 호환성 필드들
        confidence=pipeline_result.get('quality_confidence', 0.8),
        measurements=measurements,
        clothing_analysis=clothing_analysis,
        fit_score=fit_analysis.overall_fit_score
    )

# 응답 타입 유니온 (확장)
APIResponse = Union[
    VirtualTryOnResponse,
    HealthResponse, 
    PipelineStatus,
    ErrorResponse,
    ProcessingStatus,
    SessionListResponse
]