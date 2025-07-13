# backend/app/models/schemas.py
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class ModelType(str, Enum):
    """지원하는 AI 모델 타입"""
    DEMO = "demo"
    OOTD = "ootd"
    VITON = "viton"
    ACGPN = "acgpn"

class ClothingCategory(str, Enum):
    """의류 카테고리"""
    UPPER_BODY = "상의"
    LOWER_BODY = "하의"
    DRESS = "원피스"
    OUTERWEAR = "아우터"
    ACCESSORY = "액세서리"

class TryOnRequest(BaseModel):
    """가상 피팅 요청 모델"""
    height: float = Field(..., ge=100, le=250, description="키 (cm)")
    weight: float = Field(..., ge=30, le=200, description="몸무게 (kg)")
    model_type: ModelType = Field(ModelType.DEMO, description="사용할 AI 모델")
    
    @validator('height')
    def validate_height(cls, v):
        if not 100 <= v <= 250:
            raise ValueError('키는 100-250cm 범위여야 합니다')
        return v
    
    @validator('weight')
    def validate_weight(cls, v):
        if not 30 <= v <= 200:
            raise ValueError('몸무게는 30-200kg 범위여야 합니다')
        return v

class ImageInfo(BaseModel):
    """이미지 정보"""
    width: int = Field(..., description="이미지 너비")
    height: int = Field(..., description="이미지 높이")
    format: Optional[str] = Field(None, description="이미지 포맷")
    size_mb: Optional[float] = Field(None, description="파일 크기 (MB)")

class BodyMeasurements(BaseModel):
    """신체 치수"""
    estimated_chest: float = Field(..., description="추정 가슴둘레 (cm)")
    estimated_waist: float = Field(..., description="추정 허리둘레 (cm)")
    estimated_hip: float = Field(..., description="추정 엉덩이둘레 (cm)")
    bmi: float = Field(..., description="BMI")
    body_type: Optional[str] = Field(None, description="체형 타입")

class ClothingAnalysis(BaseModel):
    """의류 분석 결과"""
    category: ClothingCategory = Field(..., description="의류 카테고리")
    style: str = Field(..., description="스타일")
    colors: List[str] = Field(default_factory=list, description="주요 색상들")
    pattern: Optional[str] = Field(None, description="패턴")
    fit_score: int = Field(..., ge=0, le=100, description="적합도 점수")
    material_type: Optional[str] = Field(None, description="소재 타입")

class ProcessingStats(BaseModel):
    """처리 통계"""
    processing_time: float = Field(..., description="처리 시간 (초)")
    confidence_score: float = Field(..., ge=0, le=1, description="신뢰도 점수")
    device_used: str = Field(..., description="사용된 디바이스")
    model_version: Optional[str] = Field(None, description="모델 버전")

class TryOnResponse(BaseModel):
    """가상 피팅 응답 모델"""
    success: bool = Field(..., description="성공 여부")
    session_id: str = Field(..., description="세션 ID")
    result_image_url: str = Field(..., description="결과 이미지 URL")
    processing_stats: ProcessingStats = Field(..., description="처리 통계")
    measurements: BodyMeasurements = Field(..., description="신체 치수")
    clothing_analysis: ClothingAnalysis = Field(..., description="의류 분석")
    recommendations: List[str] = Field(default_factory=list, description="추천 사항")
    timestamp: datetime = Field(default_factory=datetime.now, description="생성 시간")

class ErrorResponse(BaseModel):
    """에러 응답 모델"""
    success: bool = Field(False, description="성공 여부")
    error_code: str = Field(..., description="에러 코드")
    error_message: str = Field(..., description="에러 메시지")
    details: Optional[Dict[str, Any]] = Field(None, description="추가 정보")
    timestamp: datetime = Field(default_factory=datetime.now, description="에러 발생 시간")

class HealthCheckResponse(BaseModel):
    """헬스체크 응답"""
    status: str = Field(..., description="서비스 상태")
    device: str = Field(..., description="사용 중인 디바이스")
    models_loaded: bool = Field(..., description="모델 로드 상태")
    timestamp: datetime = Field(default_factory=datetime.now, description="체크 시간")
    uptime_seconds: Optional[float] = Field(None, description="가동 시간")

class SystemStatus(BaseModel):
    """시스템 상태"""
    backend_status: str = Field(..., description="백엔드 상태")
    gpu_available: bool = Field(..., description="GPU 사용 가능 여부")
    device: str = Field(..., description="현재 디바이스")
    models_ready: bool = Field(..., description="모델 준비 상태")
    upload_limit_mb: int = Field(..., description="업로드 제한 (MB)")
    supported_formats: List[str] = Field(default_factory=lambda: ["jpg", "jpeg", "png"], description="지원 포맷")

class ModelStatus(BaseModel):
    """AI 모델 상태"""
    models_loaded: bool = Field(..., description="모델 로드 상태")
    available_models: List[str] = Field(..., description="사용 가능한 모델들")
    device: str = Field(..., description="모델이 로드된 디바이스")
    memory_usage: Optional[str] = Field(None, description="메모리 사용량")
    model_details: Optional[Dict[str, Any]] = Field(None, description="모델 상세 정보")

class PreprocessRequest(BaseModel):
    """전처리 요청"""
    enhance_quality: bool = Field(True, description="품질 향상 여부")
    remove_background: bool = Field(False, description="배경 제거 여부")
    target_size: Optional[int] = Field(512, description="목표 이미지 크기")

class PreprocessResponse(BaseModel):
    """전처리 응답"""
    success: bool = Field(..., description="성공 여부")
    person_analysis: Dict[str, Any] = Field(..., description="사용자 이미지 분석")
    clothing_analysis: Dict[str, Any] = Field(..., description="의류 이미지 분석")
    processing_time: float = Field(..., description="처리 시간")
    recommendations: List[str] = Field(default_factory=list, description="권장사항")

class UserFeedback(BaseModel):
    """사용자 피드백"""
    session_id: str = Field(..., description="세션 ID")
    rating: int = Field(..., ge=1, le=5, description="평점 (1-5)")
    fit_accuracy: int = Field(..., ge=1, le=5, description="피팅 정확도 (1-5)")
    processing_speed: int = Field(..., ge=1, le=5, description="처리 속도 (1-5)")
    comments: Optional[str] = Field(None, max_length=500, description="추가 의견")
    would_recommend: bool = Field(..., description="추천 의향")

class FeedbackResponse(BaseModel):
    """피드백 응답"""
    success: bool = Field(..., description="피드백 저장 성공 여부")
    message: str = Field(..., description="응답 메시지")
    feedback_id: Optional[str] = Field(None, description="피드백 ID")

# 설정 모델들
class AIModelConfig(BaseModel):
    """AI 모델 설정"""
    model_name: str = Field(..., description="모델 이름")
    device: str = Field(..., description="실행 디바이스")
    batch_size: int = Field(1, description="배치 크기")
    precision: str = Field("float32", description="연산 정밀도")
    memory_limit_mb: Optional[int] = Field(None, description="메모리 제한")

class AppSettings(BaseModel):
    """앱 설정"""
    app_name: str = Field("MyCloset AI", description="앱 이름")
    version: str = Field("1.0.0", description="버전")
    debug: bool = Field(False, description="디버그 모드")
    max_upload_size_mb: int = Field(50, description="최대 업로드 크기")
    supported_image_formats: List[str] = Field(
        default_factory=lambda: ["jpg", "jpeg", "png", "webp"],
        description="지원하는 이미지 포맷"
    )
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"],
        description="CORS 허용 도메인"
    )

# 통계 및 분석 모델들
class UsageStats(BaseModel):
    """사용 통계"""
    total_requests: int = Field(0, description="총 요청 수")
    successful_requests: int = Field(0, description="성공한 요청 수")
    failed_requests: int = Field(0, description="실패한 요청 수")
    average_processing_time: float = Field(0.0, description="평균 처리 시간")
    most_used_model: Optional[str] = Field(None, description="가장 많이 사용된 모델")
    peak_hour: Optional[int] = Field(None, description="피크 시간대")

class PerformanceMetrics(BaseModel):
    """성능 메트릭스"""
    cpu_usage_percent: float = Field(..., description="CPU 사용률")
    memory_usage_percent: float = Field(..., description="메모리 사용률")
    gpu_usage_percent: Optional[float] = Field(None, description="GPU 사용률")
    disk_usage_percent: float = Field(..., description="디스크 사용률")
    active_sessions: int = Field(0, description="활성 세션 수")
    queue_length: int = Field(0, description="대기열 길이")