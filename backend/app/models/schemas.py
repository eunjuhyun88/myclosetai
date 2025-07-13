# backend/app/models/schemas.py
"""
MyCloset AI API 스키마 정의
Pydantic 모델들로 API 요청/응답 검증
"""
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum

# ========================================
# 기본 Enum 클래스들
# ========================================

class ClothingType(str, Enum):
    """의류 타입"""
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

class StylePreference(str, Enum):
    """스타일 선호도"""
    CASUAL = "casual"
    FORMAL = "formal"
    SPORTY = "sporty"
    VINTAGE = "vintage"
    MODERN = "modern"

class QualityLevel(str, Enum):
    """품질 레벨"""
    FAST = "fast"      # 빠른 처리 (5-10초)
    MEDIUM = "medium"  # 균형잡힌 품질 (15-25초)  
    HIGH = "high"      # 고품질 (30-60초)
    ULTRA = "ultra"    # 최고품질 (60-120초)

class ProcessingStatus(str, Enum):
    """처리 상태"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# ========================================
# 요청 스키마들
# ========================================

class BodyMeasurements(BaseModel):
    """신체 치수 정보"""
    height: float = Field(..., ge=140, le=220, description="키 (cm)")
    weight: float = Field(..., ge=30, le=150, description="몸무게 (kg)")
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

class StylePreferences(BaseModel):
    """스타일 선호도"""
    style: StylePreference = Field(StylePreference.CASUAL, description="전체 스타일")
    fit: str = Field("regular", description="핏 선호도: slim, regular, loose")
    color_preference: str = Field("original", description="색상 선호도: original, darker, lighter, colorful")
    pattern_preference: str = Field("any", description="패턴 선호도: solid, striped, printed, any")

class VirtualTryOnRequest(BaseModel):
    """가상 피팅 요청"""
    clothing_type: ClothingType = Field(..., description="의류 타입")
    body_measurements: BodyMeasurements = Field(..., description="신체 치수")
    style_preferences: Optional[StylePreferences] = Field(None, description="스타일 선호도")
    quality_level: QualityLevel = Field(QualityLevel.HIGH, description="처리 품질 레벨")
    save_result: bool = Field(True, description="결과 저장 여부")
    
    class Config:
        schema_extra = {
            "example": {
                "clothing_type": "shirt",
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
                "quality_level": "high"
            }
        }

# ========================================
# 응답 스키마들
# ========================================

class ProcessingInfo(BaseModel):
    """처리 정보"""
    steps_completed: int = Field(..., description="완료된 단계 수")
    total_steps: int = Field(8, description="전체 단계 수")
    processing_time: float = Field(..., description="총 처리 시간 (초)")
    quality_level: QualityLevel = Field(..., description="처리된 품질 레벨")
    device_used: str = Field(..., description="사용된 디바이스")
    optimization: str = Field(..., description="최적화 방식")
    demo_mode: bool = Field(False, description="데모 모드 여부")
    error_message: Optional[str] = Field(None, description="오류 메시지")

class MeasurementResults(BaseModel):
    """측정 결과"""
    chest: float = Field(..., description="가슴둘레 (cm)")
    waist: float = Field(..., description="허리둘레 (cm)")
    hip: float = Field(..., description="엉덩이둘레 (cm)")
    bmi: float = Field(..., description="BMI")
    body_type: Optional[str] = Field(None, description="체형 분류")
    
    @validator('bmi')
    def validate_bmi(cls, v):
        if not 10 <= v <= 50:
            raise ValueError('BMI must be between 10 and 50')
        return v

class ClothingAnalysis(BaseModel):
    """의류 분석 결과"""
    category: str = Field(..., description="의류 카테고리")
    style: str = Field(..., description="스타일")
    dominant_color: List[int] = Field(..., description="주요 색상 [R, G, B]")
    fabric_type: Optional[str] = Field(None, description="원단 타입")
    pattern: Optional[str] = Field(None, description="패턴")
    season: Optional[str] = Field(None, description="계절감")
    formality: Optional[str] = Field(None, description="격식도")

class QualityAnalysis(BaseModel):
    """품질 분석"""
    overall_score: float = Field(..., ge=0, le=1, description="전체 품질 점수")
    fit_quality: float = Field(..., ge=0, le=1, description="핏 품질")
    processing_quality: float = Field(..., ge=0, le=1, description="처리 품질")
    realism_score: float = Field(..., ge=0, le=1, description="현실감 점수")
    detail_preservation: float = Field(..., ge=0, le=1, description="디테일 보존도")

class VirtualTryOnResponse(BaseModel):
    """가상 피팅 응답"""
    success: bool = Field(..., description="처리 성공 여부")
    session_id: str = Field(..., description="세션 ID")
    fitted_image: str = Field(..., description="결과 이미지 (base64)")
    processing_time: float = Field(..., description="처리 시간 (초)")
    confidence: float = Field(..., ge=0, le=1, description="신뢰도")
    
    # 측정 및 분석 결과
    measurements: MeasurementResults = Field(..., description="신체 측정 결과")
    clothing_analysis: ClothingAnalysis = Field(..., description="의류 분석 결과")
    quality_analysis: QualityAnalysis = Field(..., description="품질 분석")
    
    # 피팅 결과
    fit_score: float = Field(..., ge=0, le=1, description="핏 점수")
    recommendations: List[str] = Field(..., description="추천 사항")
    
    # 처리 정보
    processing_info: ProcessingInfo = Field(..., description="처리 정보")
    
    # 선택적 정보
    alternative_suggestions: Optional[List[str]] = Field(None, description="대안 제안")
    style_compatibility: Optional[float] = Field(None, ge=0, le=1, description="스타일 호환성")
    size_recommendation: Optional[str] = Field(None, description="사이즈 추천")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "session_id": "uuid-string",
                "fitted_image": "base64-encoded-image",
                "processing_time": 25.5,
                "confidence": 0.87,
                "measurements": {
                    "chest": 95.0,
                    "waist": 80.0, 
                    "hip": 95.0,
                    "bmi": 22.5
                },
                "clothing_analysis": {
                    "category": "shirt",
                    "style": "casual",
                    "dominant_color": [100, 150, 200]
                },
                "quality_analysis": {
                    "overall_score": 0.87,
                    "fit_quality": 0.85,
                    "processing_quality": 0.92,
                    "realism_score": 0.83,
                    "detail_preservation": 0.89
                },
                "fit_score": 0.85,
                "recommendations": [
                    "Perfect fit for your body type!",
                    "Try this style in different colors"
                ],
                "processing_info": {
                    "steps_completed": 8,
                    "total_steps": 8,
                    "processing_time": 25.5,
                    "quality_level": "high",
                    "device_used": "mps",
                    "optimization": "M3_Max"
                }
            }
        }

# ========================================
# 시스템 상태 스키마들
# ========================================

class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str = Field(..., description="시스템 상태")
    timestamp: str = Field(..., description="확인 시간")
    pipeline_ready: bool = Field(..., description="AI 파이프라인 준비 상태")
    memory_status: str = Field(..., description="메모리 상태")
    active_sessions: int = Field(..., description="활성 세션 수")
    version: str = Field(..., description="API 버전")
    device: str = Field(..., description="사용 중인 디바이스")

class SystemStats(BaseModel):
    """시스템 통계"""
    total_processed: int = Field(..., description="총 처리 건수")
    successful_processes: int = Field(..., description="성공 처리 건수")
    average_processing_time: float = Field(..., description="평균 처리 시간")
    success_rate: float = Field(..., ge=0, le=1, description="성공률")

class ModelStatus(BaseModel):
    """모델 상태"""
    model_name: str = Field(..., description="모델명")
    loaded: bool = Field(..., description="로드 상태")
    version: Optional[str] = Field(None, description="모델 버전")
    device: str = Field(..., description="모델이 로드된 디바이스")
    memory_usage: Optional[float] = Field(None, description="메모리 사용량 (GB)")

class StatusResponse(BaseModel):
    """시스템 상태 응답"""
    backend_status: str = Field(..., description="백엔드 상태")
    timestamp: str = Field(..., description="확인 시간")
    active_sessions: int = Field(..., description="활성 세션 수")
    processing_queue_length: int = Field(..., description="처리 대기열 길이")
    
    # AI 파이프라인 정보
    pipeline_initialized: bool = Field(..., description="파이프라인 초기화 상태")
    device: str = Field(..., description="AI 처리 디바이스")
    models_loaded: int = Field(..., description="로드된 모델 수")
    total_steps: int = Field(..., description="전체 파이프라인 단계")
    
    # 성능 정보
    memory_usage: Dict[str, Any] = Field(..., description="메모리 사용량")
    performance_stats: Dict[str, Any] = Field(..., description="성능 통계")
    
    # 모델별 상태 (선택적)
    model_status: Optional[List[ModelStatus]] = Field(None, description="개별 모델 상태")

# ========================================
# 세션 관리 스키마들  
# ========================================

class SessionInfo(BaseModel):
    """세션 정보"""
    session_id: str = Field(..., description="세션 ID")
    status: ProcessingStatus = Field(..., description="처리 상태")
    start_time: float = Field(..., description="시작 시간 (timestamp)")
    processing_time: Optional[float] = Field(None, description="처리 시간 (초)")
    clothing_type: Optional[ClothingType] = Field(None, description="의류 타입")
    quality_level: Optional[QualityLevel] = Field(None, description="품질 레벨")
    has_result: bool = Field(..., description="결과 보유 여부")
    error_message: Optional[str] = Field(None, description="오류 메시지")

class SessionListResponse(BaseModel):
    """세션 목록 응답"""
    active_sessions: int = Field(..., description="활성 세션 수")
    sessions: List[SessionInfo] = Field(..., description="세션 목록")

# ========================================
# 처리 진행 상태 스키마들
# ========================================

class ProcessingStep(BaseModel):
    """처리 단계 정보"""
    step_number: int = Field(..., ge=1, le=8, description="단계 번호")
    step_name: str = Field(..., description="단계명")
    status: ProcessingStatus = Field(..., description="단계 상태")
    progress_percentage: float = Field(..., ge=0, le=100, description="진행률 (%)")
    processing_time: Optional[float] = Field(None, description="단계 처리 시간")
    error_message: Optional[str] = Field(None, description="오류 메시지")

class ProcessingProgress(BaseModel):
    """처리 진행 상태"""
    session_id: str = Field(..., description="세션 ID")
    overall_progress: float = Field(..., ge=0, le=100, description="전체 진행률 (%)")
    current_step: int = Field(..., ge=1, le=8, description="현재 단계")
    estimated_remaining_time: Optional[float] = Field(None, description="예상 남은 시간 (초)")
    steps: List[ProcessingStep] = Field(..., description="단계별 진행 상태")

class ProcessingResponse(BaseModel):
    """처리 시작 응답"""
    session_id: str = Field(..., description="세션 ID")
    status: ProcessingStatus = Field(..., description="초기 상태")
    message: str = Field(..., description="응답 메시지")
    estimated_time: Optional[float] = Field(None, description="예상 처리 시간 (초)")
    quality_level: QualityLevel = Field(..., description="선택된 품질 레벨")

# ========================================
# 오류 스키마들
# ========================================

class ErrorDetail(BaseModel):
    """오류 상세 정보"""
    error_code: str = Field(..., description="오류 코드")
    error_message: str = Field(..., description="오류 메시지")
    error_type: str = Field(..., description="오류 타입")
    step_number: Optional[int] = Field(None, description="오류 발생 단계")
    suggestions: List[str] = Field(..., description="해결 제안")

class ErrorResponse(BaseModel):
    """오류 응답"""
    success: bool = Field(False, description="성공 여부")
    error: ErrorDetail = Field(..., description="오류 상세")
    session_id: Optional[str] = Field(None, description="세션 ID")
    timestamp: str = Field(..., description="오류 발생 시간")

# ========================================
# 추가 유틸리티 스키마들
# ========================================

class ImageUploadResponse(BaseModel):
    """이미지 업로드 응답"""
    success: bool = Field(..., description="업로드 성공 여부")
    file_id: str = Field(..., description="파일 ID")
    file_url: str = Field(..., description="파일 URL")
    file_size: int = Field(..., description="파일 크기 (bytes)")
    image_dimensions: Dict[str, int] = Field(..., description="이미지 크기 {'width': int, 'height': int}")
    format: str = Field(..., description="이미지 포맷")

class ValidationError(BaseModel):
    """유효성 검사 오류"""
    field: str = Field(..., description="오류 필드")
    message: str = Field(..., description="오류 메시지")
    input_value: Any = Field(..., description="입력값")

class ValidationResponse(BaseModel):
    """유효성 검사 응답"""
    valid: bool = Field(..., description="유효성 여부")
    errors: List[ValidationError] = Field(..., description="오류 목록")

# ========================================
# 설정 및 환경 정보
# ========================================

class SystemConfiguration(BaseModel):
    """시스템 설정 정보"""
    max_upload_size: int = Field(..., description="최대 업로드 크기 (bytes)")
    supported_image_formats: List[str] = Field(..., description="지원하는 이미지 포맷")
    supported_clothing_types: List[str] = Field(..., description="지원하는 의류 타입")
    quality_levels: List[str] = Field(..., description="사용 가능한 품질 레벨")
    default_quality_level: str = Field(..., description="기본 품질 레벨")
    processing_timeout: int = Field(..., description="처리 타임아웃 (초)")

class EnvironmentInfo(BaseModel):
    """환경 정보"""
    python_version: str = Field(..., description="Python 버전")
    pytorch_version: str = Field(..., description="PyTorch 버전")
    cuda_available: bool = Field(..., description="CUDA 사용 가능 여부")
    mps_available: bool = Field(..., description="MPS 사용 가능 여부")
    device_count: int = Field(..., description="사용 가능한 디바이스 수")
    memory_total: Optional[float] = Field(None, description="총 메모리 (GB)")

# ========================================
# API 정보 스키마
# ========================================

class APIInfo(BaseModel):
    """API 정보"""
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

# ========================================
# 유니온 타입 및 제네릭 응답
# ========================================

class SuccessResponse(BaseModel):
    """성공 응답 (제네릭)"""
    success: bool = Field(True, description="성공 여부")
    message: str = Field(..., description="응답 메시지")
    data: Optional[Dict[str, Any]] = Field(None, description="응답 데이터")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="응답 시간")

# 응답 타입 유니온
APIResponse = Union[
    VirtualTryOnResponse,
    HealthResponse, 
    StatusResponse,
    ErrorResponse,
    SuccessResponse
]