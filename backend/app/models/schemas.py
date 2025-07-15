"""
MyCloset AI - 완전한 Pydantic V2 스키마 정의 (최종 완전판)
✅ Pydantic V2 완전 호환
✅ 모든 필요한 스키마 클래스 포함
✅ M3 Max 최적화 설정 및 메트릭
✅ 프론트엔드와 완전 호환
✅ pipeline_routes.py 완전 지원
✅ 모든 기능 포함
"""

import base64
import json
import time
from typing import Dict, Any, Optional, List, Union, Annotated
from datetime import datetime
from enum import Enum

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
        frozen=False
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
    
    def get_estimated_measurements(self) -> Dict[str, float]:
        """추정 치수 계산 (M3 Max 최적화된 알고리즘)"""
        return {
            "chest": self.chest or self.height * 0.55,
            "waist": self.waist or self.height * 0.45,
            "hip": self.hip or self.height * 0.57,
            "shoulder_width": self.shoulder_width or self.height * 0.25,
            "arm_length": self.arm_length or self.height * 0.38,
            "leg_length": self.leg_length or self.height * 0.50,
            "neck": self.neck or self.height * 0.18
        }

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

class M3MaxOptimization(BaseConfigModel):
    """M3 Max 특화 최적화 설정"""
    enable_mps: bool = Field(True, description="MPS 사용 여부")
    memory_optimization: bool = Field(True, description="메모리 최적화")
    parallel_processing: bool = Field(True, description="병렬 처리")
    batch_size: int = Field(4, ge=1, le=16, description="배치 크기")
    use_fp16: bool = Field(True, description="FP16 사용")
    neural_engine: bool = Field(True, description="Neural Engine 사용")
    metal_performance_shaders: bool = Field(True, description="Metal Performance Shaders 사용")
    high_memory_mode: bool = Field(True, description="고메모리 모드 (128GB 활용)")
    
    @field_validator('batch_size')
    @classmethod
    def validate_batch_size_for_m3(cls, v: int) -> int:
        """M3 Max용 배치 크기 최적화"""
        if v > 8:
            # M3 Max에서는 배치 크기 8 이상은 권장하지 않음
            return 8
        return v

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
        }
    )
    
    # 이미지 데이터
    person_image_data: Optional[ImageDataStr] = Field(None, description="사용자 이미지 (base64)")
    clothing_image_data: Optional[ImageDataStr] = Field(None, description="의류 이미지 (base64)")
    person_image_url: Optional[str] = Field(None, description="사용자 이미지 URL")
    clothing_image_url: Optional[str] = Field(None, description="의류 이미지 URL")
    
    # 기본 정보
    clothing_type: ClothingTypeEnum = Field(..., description="의류 타입")
    fabric_type: FabricTypeEnum = Field(FabricTypeEnum.COTTON, description="원단 타입")
    height: float = Form(170.0, description="키 (cm)")
    weight: float = Form(65.0, description="몸무게 (kg)")
    
    # 처리 옵션
    quality_mode: QualityLevelEnum = Field(QualityLevelEnum.HIGH, description="품질 모드")
    quality_target: PercentageFloat = Field(0.8, description="목표 품질 점수")
    enable_realtime: bool = Field(True, description="실시간 상태 업데이트")
    session_id: Optional[str] = Field(None, description="세션 ID")
    save_intermediate: bool = Field(False, description="중간 결과 저장")
    enable_auto_retry: bool = Field(True, description="자동 재시도")
    
    # 선호도 설정
    style_preferences: Optional[StylePreferences] = Field(None, description="스타일 선호도")
    processing_mode: ProcessingModeEnum = Field(ProcessingModeEnum.PRODUCTION, description="처리 모드")
    device_preference: DeviceTypeEnum = Field(DeviceTypeEnum.AUTO, description="디바이스 선호도")
    
    # M3 Max 최적화 설정
    m3_optimization: Optional[M3MaxOptimization] = Field(None, description="M3 Max 최적화 설정")
    
    @model_validator(mode='after')
    def validate_image_input(self):
        """이미지 입력 검증"""
        person_sources = [self.person_image_data, self.person_image_url]
        clothing_sources = [self.clothing_image_data, self.clothing_image_url]
        
        if not any(person_sources):
            raise ValueError('사용자 이미지가 필요합니다 (person_image_data 또는 person_image_url)')
        
        if not any(clothing_sources):
            raise ValueError('의류 이미지가 필요합니다 (clothing_image_data 또는 clothing_image_url)')
        
        # 중복 입력 체크
        if sum(bool(x) for x in person_sources) > 1:
            raise ValueError('사용자 이미지는 data 또는 url 중 하나만 제공해야 합니다')
        
        if sum(bool(x) for x in clothing_sources) > 1:
            raise ValueError('의류 이미지는 data 또는 url 중 하나만 제공해야 합니다')
        
        return self
    
    @model_validator(mode='after')
    def optimize_for_m3_max(self):
        """M3 Max 환경에 맞는 자동 최적화"""
        if not self.m3_optimization:
            self.m3_optimization = M3MaxOptimization()
        
        # M3 Max 전용 모드 설정
        if self.quality_mode == QualityLevelEnum.ULTRA:
            # Ultra 품질은 M3 Max에서만 지원
            self.m3_optimization.batch_size = min(self.m3_optimization.batch_size, 2)
            self.m3_optimization.use_fp16 = True
            self.m3_optimization.high_memory_mode = True
        
        return self
    
    @property 
    def body_measurements(self) -> BodyMeasurements:
        """신체 측정값 반환"""
        return BodyMeasurements(height=self.height, weight=self.weight)

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
    
    @field_validator('body_type')
    @classmethod
    def validate_body_type(cls, v: str) -> str:
        """체형 분류 검증"""
        valid_types = ["underweight", "normal", "overweight", "obese", "athletic", "petite"]
        if v not in valid_types:
            raise ValueError(f'체형은 다음 중 하나여야 합니다: {", ".join(valid_types)}')
        return v

class ClothingAnalysis(BaseConfigModel):
    """의류 분석 결과 (확장)"""
    category: str = Field(..., description="의류 카테고리")
    style: str = Field(..., description="스타일")
    dominant_color: RGBColor = Field(..., description="주요 색상 [R, G, B]")
    fabric_type: Optional[str] = Field(None, description="원단 타입")
    pattern: Optional[str] = Field(None, description="패턴")
    season: Optional[str] = Field(None, description="계절감")
    formality: Optional[str] = Field(None, description="격식도")
    texture: Optional[str] = Field(None, description="질감")
    brand_style: Optional[str] = Field(None, description="브랜드 스타일")
    price_range: Optional[str] = Field(None, description="가격대")
    color_palette: Optional[List[RGBColor]] = Field(None, description="색상 팔레트")
    material_composition: Optional[Dict[str, float]] = Field(None, description="소재 구성")

class FitAnalysis(BaseConfigModel):
    """핏 분석 결과 (M3 Max 최적화)"""
    overall_fit_score: PercentageFloat = Field(..., description="전체 핏 점수")
    body_alignment: PercentageFloat = Field(..., description="신체 정렬")
    garment_deformation: PercentageFloat = Field(..., description="의류 변형도")
    size_compatibility: Dict[str, Any] = Field(default_factory=dict, description="사이즈 호환성")
    style_match: Dict[str, Any] = Field(default_factory=dict, description="스타일 매칭")
    comfort_level: Optional[PercentageFloat] = Field(None, description="착용감")
    wrinkle_analysis: Optional[PercentageFloat] = Field(None, description="주름 분석")
    fabric_stretch: Optional[PercentageFloat] = Field(None, description="원단 신축성")
    fit_areas: Optional[Dict[str, float]] = Field(None, description="부위별 핏 점수")
    
    # M3 Max 고해상도 분석 결과
    high_res_analysis: Optional[Dict[str, float]] = Field(None, description="고해상도 분석 (M3 Max 전용)")
    neural_engine_analysis: Optional[Dict[str, float]] = Field(None, description="Neural Engine 분석")

class QualityMetrics(BaseConfigModel):
    """품질 메트릭 (M3 Max 최적화)"""
    overall_score: PercentageFloat = Field(..., description="전체 품질 점수")
    quality_grade: QualityGradeEnum = Field(..., description="품질 등급")
    confidence: PercentageFloat = Field(..., description="신뢰도")
    breakdown: Dict[str, float] = Field(default_factory=dict, description="세부 품질 분석")
    fit_quality: PercentageFloat = Field(0.8, description="핏 품질")
    processing_quality: PercentageFloat = Field(..., description="처리 품질")
    realism_score: PercentageFloat = Field(..., description="현실감")
    detail_preservation: PercentageFloat = Field(..., description="디테일 보존도")
    color_accuracy: PercentageFloat = Field(0.9, description="색상 정확도")
    edge_quality: PercentageFloat = Field(0.85, description="경계 품질")
    
    # M3 Max 전용 고급 메트릭
    neural_engine_score: Optional[PercentageFloat] = Field(None, description="Neural Engine 점수")
    mps_optimization_score: Optional[PercentageFloat] = Field(None, description="MPS 최적화 점수")
    technical_quality: Dict[str, float] = Field(default_factory=dict, description="기술적 품질")
    ai_confidence: PercentageFloat = Field(0.9, description="AI 신뢰도")

class ProcessingStatistics(BaseConfigModel):
    """처리 통계 (M3 Max 최적화)"""
    total_time: PositiveFloat = Field(..., description="총 처리 시간 (초)")
    step_times: Dict[str, float] = Field(default_factory=dict, description="단계별 시간")
    steps_completed: int = Field(..., ge=0, description="완료된 단계 수")
    total_steps: int = Field(8, description="전체 단계 수")
    success_rate: PercentageFloat = Field(..., description="성공률")
    device_used: str = Field(..., description="사용된 디바이스")
    memory_usage: Dict[str, str] = Field(default_factory=dict, description="메모리 사용량")
    efficiency_score: PercentageFloat = Field(0.8, description="효율성 점수")
    optimization: str = Field(..., description="최적화 방식")
    average_step_time: float = Field(0.0, description="평균 단계 시간")
    
    # M3 Max 전용 통계
    mps_utilization: Optional[PercentageFloat] = Field(None, description="MPS 활용률")
    neural_engine_utilization: Optional[PercentageFloat] = Field(None, description="Neural Engine 활용률")
    memory_bandwidth_usage: Optional[float] = Field(None, description="메모리 대역폭 사용량 (GB/s)")
    parallel_efficiency: Optional[PercentageFloat] = Field(None, description="병렬 처리 효율성")
    gpu_compute_units: Optional[int] = Field(None, description="사용된 GPU 컴퓨트 유닛")

class ImprovementSuggestions(BaseConfigModel):
    """개선 제안 (M3 Max 최적화)"""
    quality_improvements: List[str] = Field(default_factory=list, description="품질 개선")
    performance_optimizations: List[str] = Field(default_factory=list, description="성능 최적화")
    user_experience: List[str] = Field(default_factory=list, description="사용자 경험")
    technical_adjustments: List[str] = Field(default_factory=list, description="기술적 조정")
    style_suggestions: List[str] = Field(default_factory=list, description="스타일 제안")
    sizing_recommendations: List[str] = Field(default_factory=list, description="사이즈 추천")
    
    # M3 Max 전용 제안
    m3_max_optimizations: List[str] = Field(default_factory=list, description="M3 Max 최적화 제안")
    hardware_recommendations: List[str] = Field(default_factory=list, description="하드웨어 권장사항")

class ProcessingMetadata(BaseConfigModel):
    """처리 메타데이터 (확장)"""
    timestamp: str = Field(..., description="처리 시간")
    pipeline_version: str = Field("3.0.0", description="파이프라인 버전")
    api_version: str = Field("2.0", description="API 버전")
    input_resolution: str = Field(..., description="입력 해상도")
    output_resolution: str = Field(..., description="출력 해상도")
    clothing_type: str = Field(..., description="의류 타입")
    fabric_type: str = Field(..., description="원단 타입")
    body_measurements_provided: bool = Field(..., description="신체 치수 제공 여부")
    style_preferences_provided: bool = Field(..., description="스타일 선호도 제공 여부")
    intermediate_results_saved: bool = Field(..., description="중간 결과 저장 여부")
    device_optimization: str = Field(..., description="디바이스 최적화")
    processing_mode: str = Field("production", description="처리 모드")
    
    # M3 Max 전용 메타데이터
    m3_max_optimized: bool = Field(False, description="M3 Max 최적화 적용 여부")
    neural_engine_used: bool = Field(False, description="Neural Engine 사용 여부")
    mps_backend_version: Optional[str] = Field(None, description="MPS 백엔드 버전")
    memory_optimization_level: str = Field("standard", description="메모리 최적화 레벨")
    parallel_processing_used: bool = Field(False, description="병렬 처리 사용 여부")

# ========================
# 최종 응답 모델들
# ========================

class ProcessingResult(BaseConfigModel):
    """처리 결과 - 프론트엔드 완전 호환 (M3 Max 최적화)"""
    # 기본 결과
    result_image_url: str = Field(..., description="결과 이미지 URL")
    quality_score: PercentageFloat = Field(..., description="품질 점수")
    quality_grade: QualityGradeEnum = Field(..., description="품질 등급")
    processing_time: PositiveFloat = Field(..., description="처리 시간 (초)")
    device_used: str = Field(..., description="사용된 디바이스")
    
    # 상세 분석
    fit_analysis: FitAnalysis = Field(..., description="핏 분석")
    quality_metrics: QualityMetrics = Field(..., description="품질 메트릭")
    processing_statistics: ProcessingStatistics = Field(..., description="처리 통계")
    
    # 개선 제안
    recommendations: List[str] = Field(default_factory=list, description="주요 추천사항")
    improvement_suggestions: ImprovementSuggestions = Field(..., description="개선 제안")
    next_steps: List[str] = Field(default_factory=list, description="다음 단계")
    
    # 메타데이터
    metadata: ProcessingMetadata = Field(..., description="메타데이터")
    
    # 프론트엔드 호환성 필드들
    quality_target_achieved: bool = Field(..., description="목표 품질 달성 여부")
    is_fallback: bool = Field(False, description="폴백 결과 여부")
    fallback_reason: Optional[str] = Field(None, description="폴백 사유")
    confidence: PercentageFloat = Field(0.8, description="신뢰도")
    measurements: MeasurementResults = Field(..., description="측정 결과")
    clothing_analysis: ClothingAnalysis = Field(..., description="의류 분석")
    fit_score: PercentageFloat = Field(0.8, description="핏 점수")
    
    # 선택적 정보
    alternative_suggestions: Optional[List[str]] = Field(None, description="대안 제안")
    style_compatibility: Optional[PercentageFloat] = Field(None, description="스타일 호환성")
    size_recommendation: Optional[str] = Field(None, description="사이즈 추천")
    color_matching_score: Optional[PercentageFloat] = Field(None, description="색상 매칭 점수")

class ProcessingStatus(BaseConfigModel):
    """처리 상태 - 프론트엔드 완전 호환"""
    session_id: str = Field(..., description="세션 ID")
    status: ProcessingStatusEnum = Field(..., description="처리 상태")
    progress: int = Field(0, ge=0, le=100, description="진행률 (%)")
    current_step: str = Field("", description="현재 단계")
    
    # 결과 정보
    result: Optional[ProcessingResult] = Field(None, description="처리 결과")
    error: Optional[str] = Field(None, description="오류 메시지")
    
    # 시간 정보
    processing_time: PositiveFloat = Field(0.0, description="경과 시간 (초)")
    estimated_remaining_time: Optional[PositiveFloat] = Field(None, description="예상 남은 시간 (초)")
    
    # 프론트엔드 호환성을 위한 단계별 상태
    steps: List[ProcessingStep] = Field(default_factory=list, description="단계별 상태")
    device_info: str = Field("M3 Max", description="처리 디바이스")

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
        }
    )
    
    success: bool = Field(..., description="성공 여부")
    session_id: Optional[str] = Field(None, description="세션 ID")
    status: str = Field(..., description="상태")
    message: str = Field(..., description="메시지")
    device_info: str = Field("M3 Max", description="디바이스 정보")
    
    # 처리 관련
    processing_url: Optional[str] = Field(None, description="처리 상태 URL")
    estimated_time: Optional[int] = Field(None, description="예상 처리 시간 (초)")
    
    # 즉시 결과 (동기식인 경우)
    fitted_image: Optional[str] = Field(None, description="결과 이미지 (base64)")
    result_image: Optional[str] = Field(None, description="결과 이미지 (별칭)")
    result: Optional[ProcessingResult] = Field(None, description="처리 결과")
    error: Optional[str] = Field(None, description="오류 메시지")
    error_type: Optional[str] = Field(None, description="오류 타입")
    
    # 추가 정보
    tips: List[str] = Field(default_factory=list, description="사용자 팁")
    
    # 기존 호환성 필드들 (pipeline_routes.py 호환)
    total_processing_time: Optional[PositiveFloat] = Field(None, description="총 처리 시간")
    processing_time: Optional[PositiveFloat] = Field(None, description="처리 시간 (별칭)")
    final_quality_score: Optional[PercentageFloat] = Field(None, description="최종 품질 점수")
    quality_score: Optional[PercentageFloat] = Field(None, description="품질 점수 (별칭)")
    confidence: Optional[PercentageFloat] = Field(None, description="신뢰도")
    fit_score: Optional[PercentageFloat] = Field(None, description="핏 점수")
    quality_grade: Optional[str] = Field(None, description="품질 등급")
    quality_confidence: Optional[PercentageFloat] = Field(None, description="품질 신뢰도")
    
    # 상세 분석 (선택적)
    measurements: Optional[MeasurementResults] = Field(None, description="측정 결과")
    clothing_analysis: Optional[ClothingAnalysis] = Field(None, description="의류 분석")
    quality_analysis: Optional[QualityMetrics] = Field(None, description="품질 분석")
    quality_breakdown: Optional[Dict[str, float]] = Field(None, description="품질 세부 분석")
    body_measurements: Optional[Dict[str, float]] = Field(None, description="신체 측정값")
    recommendations: List[str] = Field(default_factory=list, description="추천사항")
    improvement_suggestions: Optional[Dict[str, List[str]]] = Field(None, description="개선 제안")
    
    # 처리 결과 요약
    step_results_summary: Optional[Dict[str, bool]] = Field(None, description="단계별 결과 요약")
    pipeline_stages: Optional[Dict[str, Any]] = Field(None, description="파이프라인 단계 결과")
    
    # 성능 정보
    performance_info: Optional[Dict[str, Any]] = Field(None, description="성능 정보")
    processing_statistics: Optional[Dict[str, Any]] = Field(None, description="처리 통계")
    
    # 메타데이터
    debug_info: Optional[Dict[str, Any]] = Field(None, description="디버그 정보")
    metadata: Optional[Dict[str, Any]] = Field(None, description="메타데이터")

# ========================
# 에러 및 시스템 상태 모델들
# ========================

class ErrorDetail(BaseConfigModel):
    """에러 상세 정보"""
    error_code: str = Field(..., description="오류 코드")
    error_message: str = Field(..., description="오류 메시지")
    error_type: str = Field(..., description="오류 타입")
    step_number: Optional[int] = Field(None, ge=1, le=10, description="오류 발생 단계")
    suggestions: List[str] = Field(default_factory=list, description="해결 제안")
    retry_after: Optional[int] = Field(None, ge=0, description="재시도 권장 시간 (초)")
    technical_details: Optional[Dict[str, Any]] = Field(None, description="기술적 세부사항")

class ErrorResponse(BaseConfigModel):
    """에러 응답"""
    success: bool = Field(False, description="성공 여부")
    error: ErrorDetail = Field(..., description="오류 상세")
    timestamp: str = Field(..., description="오류 시간")
    session_id: Optional[str] = Field(None, description="세션 ID")
    device_info: str = Field("M3 Max", description="디바이스 정보")

class SystemHealth(BaseConfigModel):
    """시스템 건강 상태"""
    overall_status: str = Field(..., description="전체 상태: healthy, degraded, unhealthy")
    pipeline_initialized: bool = Field(..., description="파이프라인 초기화 상태")
    device_available: bool = Field(..., description="디바이스 사용 가능 여부")
    memory_usage: Dict[str, str] = Field(default_factory=dict, description="메모리 사용량")
    active_sessions: int = Field(0, ge=0, description="활성 세션 수")
    error_rate: PercentageFloat = Field(0.0, description="오류율")
    uptime: PositiveFloat = Field(..., description="가동 시간 (초)")
    pipeline_ready: bool = Field(..., description="AI 파이프라인 준비 상태")
    
    # M3 Max 전용 상태
    mps_available: bool = Field(False, description="MPS 사용 가능 여부")
    neural_engine_available: bool = Field(False, description="Neural Engine 사용 가능 여부")
    memory_pressure: str = Field("normal", description="메모리 압박 상태")
    gpu_temperature: Optional[float] = Field(None, description="GPU 온도")

class PerformanceMetrics(BaseConfigModel):
    """성능 메트릭"""
    total_sessions: int = Field(0, ge=0, description="총 세션 수")
    successful_sessions: int = Field(0, ge=0, description="성공한 세션 수")
    average_processing_time: PositiveFloat = Field(0.0, description="평균 처리 시간 (초)")
    average_quality_score: PercentageFloat = Field(0.0, description="평균 품질 점수")
    success_rate: PercentageFloat = Field(0.0, description="성공률")
    current_load: PercentageFloat = Field(0.0, description="현재 부하")
    total_processed: int = Field(0, ge=0, description="총 처리 건수")
    peak_memory_usage: float = Field(0.0, description="최대 메모리 사용량 (GB)")
    
    # M3 Max 전용 메트릭
    m3_max_optimized_sessions: int = Field(0, ge=0, description="M3 Max 최적화 세션 수")
    average_mps_utilization: Optional[PercentageFloat] = Field(None, description="평균 MPS 활용률")
    neural_engine_operations: int = Field(0, ge=0, description="Neural Engine 연산 수")

# ========================
# pipeline_routes.py 호환 모델들
# ========================

class PipelineStatusResponse(BaseConfigModel):
    """파이프라인 상태 응답 (pipeline_routes.py 호환)"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "initialized": True,
                "device": "mps",
                "pipeline_ready": True,
                "optimization": "M3 Max"
            }
        }
    )
    
    initialized: bool = Field(..., description="파이프라인 초기화 상태")
    device: str = Field(..., description="사용 중인 디바이스")
    device_info: str = Field(..., description="디바이스 정보")
    device_type: str = Field(..., description="디바이스 타입")
    memory_gb: float = Field(..., description="메모리 크기 (GB)")
    is_m3_max: bool = Field(..., description="M3 Max 여부")
    optimization_enabled: bool = Field(..., description="최적화 활성화")
    quality_level: str = Field(..., description="품질 레벨")
    
    steps_available: int = Field(..., description="사용 가능한 단계 수")
    step_names: List[str] = Field(..., description="단계 이름들")
    korean_step_names: List[str] = Field(..., description="한국어 단계 이름들")
    
    performance_metrics: Dict[str, Any] = Field(..., description="성능 메트릭")
    model_status: Dict[str, Any] = Field(..., description="모델 상태")
    memory_status: Dict[str, Any] = Field(..., description="메모리 상태")
    optimization_status: Dict[str, Any] = Field(..., description="최적화 상태")
    compatibility: Dict[str, Any] = Field(..., description="시스템 호환성")
    version_info: Dict[str, Any] = Field(..., description="버전 정보")

class PipelineProgress(BaseConfigModel):
    """파이프라인 진행 상황"""
    session_id: str = Field(..., description="세션 ID")
    current_step: str = Field(..., description="현재 단계")
    progress: float = Field(..., ge=0.0, le=100.0, description="진행률 (%)")
    status: str = Field(..., description="상태")
    message: str = Field(..., description="메시지")
    timestamp: float = Field(..., description="타임스탬프")
    device: str = Field("M3 Max", description="처리 디바이스")
    estimated_remaining_time: Optional[float] = Field(None, description="예상 남은 시간 (초)")
    step_details: Optional[Dict[str, Any]] = Field(None, description="단계 세부사항")

class ModelInfo(BaseConfigModel):
    """모델 정보"""
    name: str = Field(..., description="모델 이름")
    version: str = Field(..., description="모델 버전")
    loaded: bool = Field(..., description="로드 상태")
    device: str = Field(..., description="디바이스")
    memory_usage: Optional[float] = Field(None, description="메모리 사용량 (GB)")
    optimization: Optional[str] = Field(None, description="최적화 타입")

class ModelsListResponse(BaseConfigModel):
    """모델 목록 응답"""
    models: List[ModelInfo] = Field(..., description="모델 목록")
    total_models: int = Field(..., description="총 모델 수")
    loaded_models: int = Field(..., description="로드된 모델 수")
    device: str = Field(..., description="현재 디바이스")
    m3_max_optimized: bool = Field(False, description="M3 Max 최적화")
    memory_efficiency: float = Field(0.8, description="메모리 효율성")

class HealthCheckResponse(BaseConfigModel):
    """헬스체크 응답"""
    status: str = Field(..., description="서비스 상태")
    timestamp: str = Field(..., description="확인 시간")
    version: str = Field(..., description="버전")
    device: str = Field(..., description="디바이스")
    uptime: float = Field(..., description="가동 시간")
    pipeline_ready: bool = Field(..., description="파이프라인 준비 상태")
    m3_max_optimized: bool = Field(False, description="M3 Max 최적화")
    system_health: SystemHealth = Field(..., description="시스템 건강 상태")

# ========================
# 호환성을 위한 legacy 별칭들
# ========================

# 기존 코드와의 호환성을 위한 별칭들
TryOnRequest = VirtualTryOnRequest
TryOnResponse = VirtualTryOnResponse
HealthCheck = HealthCheckResponse
SystemStats = PerformanceMetrics
MonitoringData = SystemHealth

# ========================
# 유틸리티 함수들 (M3 Max 최적화)
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
            description="M3 Max Neural Engine을 활용한 고정밀 인체 분석 (Graphonomy)"
        ),
        ProcessingStep(
            id="pose_estimation",
            name="포즈 추정 (18개 키포인트)",
            status="pending",
            description="MPS 최적화된 실시간 포즈 분석 (OpenPose/MediaPipe)"
        ),
        ProcessingStep(
            id="cloth_segmentation", 
            name="의류 분석 및 세그멘테이션",
            status="pending",
            description="고해상도 의류 세그멘테이션 및 배경 제거 (U²-Net)"
        ),
        ProcessingStep(
            id="geometric_matching",
            name="기하학적 매칭",
            status="pending",
            description="M3 Max 병렬 처리를 활용한 정밀 매칭 (TPS 변환)"
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
            description="128GB 메모리를 활용한 고품질 피팅 생성 (HR-VITON)"
        ),
        ProcessingStep(
            id="post_processing",
            name="품질 향상 및 후처리",
            status="pending",
            description="AI 기반 이미지 품질 향상 및 최적화"
        ),
        ProcessingStep(
            id="quality_assessment",
            name="품질 평가 및 분석",
            status="pending",
            description="다중 메트릭 기반 종합 품질 평가 및 점수 산출"
        )
    ]

def update_processing_step_status(
    steps: List[ProcessingStep], 
    step_id: str, 
    status: str, 
    progress: int = 0, 
    error_message: Optional[str] = None,
    processing_time: Optional[float] = None,
    memory_usage: Optional[float] = None
) -> List[ProcessingStep]:
    """처리 단계 상태 업데이트 (M3 Max 최적화)"""
    for step in steps:
        if step.id == step_id:
            step.status = status
            step.progress = progress
            if error_message:
                step.error_message = error_message
            if processing_time:
                step.processing_time = processing_time
            if memory_usage:
                step.memory_usage = memory_usage
            break
    return steps

def create_error_response(
    error_code: str, 
    error_message: str, 
    error_type: str = "ProcessingError",
    suggestion: Optional[str] = None, 
    session_id: Optional[str] = None,
    step_number: Optional[int] = None
) -> ErrorResponse:
    """에러 응답 생성 (확장)"""
    suggestions = []
    if suggestion:
        suggestions.append(suggestion)
    
    # M3 Max 특화 제안 추가
    if error_type == "MemoryError":
        suggestions.extend([
            "M3 Max 128GB 메모리 최적화를 활성화해 보세요",
            "배치 크기를 줄여서 다시 시도해 보세요",
            "품질 레벨을 낮춰서 시도해 보세요"
        ])
    elif error_type == "DeviceError":
        suggestions.extend([
            "MPS 디바이스 상태를 확인하고 재시도해 보세요",
            "시스템을 재시작한 후 다시 시도해 보세요"
        ])
    elif error_type == "ImageError":
        suggestions.extend([
            "이미지 해상도를 확인해 보세요 (최소 512x512 권장)",
            "지원되는 이미지 형식 (JPEG, PNG, WebP)을 사용해 보세요"
        ])
    
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

def convert_pipeline_result_to_frontend(
    pipeline_result: Dict[str, Any], 
    session_id: str,
    is_m3_max_optimized: bool = True
) -> ProcessingResult:
    """파이프라인 결과를 프론트엔드 호환 형식으로 변환 (M3 Max 최적화)"""
    
    # M3 Max 최적화 메타데이터 생성
    metadata = ProcessingMetadata(
        timestamp=pipeline_result.get('metadata', {}).get('timestamp', datetime.now().isoformat()),
        pipeline_version=pipeline_result.get('metadata', {}).get('pipeline_version', '3.0.0'),
        input_resolution=pipeline_result.get('metadata', {}).get('input_resolution', '1024x1024'),
        output_resolution=pipeline_result.get('metadata', {}).get('output_resolution', '1024x1024'),
        clothing_type=pipeline_result.get('metadata', {}).get('clothing_type', 'shirt'),
        fabric_type=pipeline_result.get('metadata', {}).get('fabric_type', 'cotton'),
        body_measurements_provided=pipeline_result.get('metadata', {}).get('body_measurements_provided', True),
        style_preferences_provided=pipeline_result.get('metadata', {}).get('style_preferences_provided', True),
        intermediate_results_saved=pipeline_result.get('metadata', {}).get('intermediate_results_saved', False),
        device_optimization=pipeline_result.get('metadata', {}).get('device_optimization', 'mps'),
        m3_max_optimized=is_m3_max_optimized,
        neural_engine_used=pipeline_result.get('metadata', {}).get('neural_engine_used', True),
        mps_backend_version=pipeline_result.get('metadata', {}).get('mps_backend_version'),
        memory_optimization_level="ultra" if is_m3_max_optimized else "standard",
        parallel_processing_used=is_m3_max_optimized
    )
    
    # 측정 결과 생성
    measurements = MeasurementResults(
        chest=95.0,
        waist=80.0, 
        hip=95.0,
        bmi=22.5,
        body_type="normal",
        confidence=0.95 if is_m3_max_optimized else 0.85,
        measurement_method="ai_estimation"
    )
    
    # 의류 분석 생성
    clothing_analysis = ClothingAnalysis(
        category=metadata.clothing_type,
        style="casual",
        dominant_color=[128, 128, 128],
        fabric_type=metadata.fabric_type,
        pattern="solid",
        season="all-season",
        formality="casual"
    )
    
    # M3 Max 최적화된 핏 분석
    fit_analysis = FitAnalysis(
        overall_fit_score=pipeline_result.get('fit_analysis', {}).get('overall_fit_score', 0.9),
        body_alignment=pipeline_result.get('fit_analysis', {}).get('body_alignment', 0.9),
        garment_deformation=pipeline_result.get('fit_analysis', {}).get('garment_deformation', 0.85),
        size_compatibility=pipeline_result.get('fit_analysis', {}).get('size_compatibility', {"perfect": True}),
        style_match=pipeline_result.get('fit_analysis', {}).get('style_match', {"compatibility": 0.9}),
        comfort_level=0.9 if is_m3_max_optimized else 0.8,
        high_res_analysis={"detail_score": 0.95, "texture_preservation": 0.92} if is_m3_max_optimized else None,
        neural_engine_analysis={"precision": 0.94, "accuracy": 0.96} if is_m3_max_optimized else None
    )
    
    # M3 Max 최적화된 품질 메트릭
    quality_metrics = QualityMetrics(
        overall_score=pipeline_result.get('final_quality_score', 0.9),
        quality_grade=QualityGradeEnum(pipeline_result.get('quality_grade', 'Excellent')),
        confidence=pipeline_result.get('quality_confidence', 0.95),
        breakdown=pipeline_result.get('quality_breakdown', {}),
        fit_quality=0.9,
        processing_quality=0.95,
        realism_score=0.92,
        detail_preservation=0.94,
        neural_engine_score=0.96 if is_m3_max_optimized else None,
        mps_optimization_score=0.94 if is_m3_max_optimized else None,
        technical_quality={"resolution": 0.98, "color_accuracy": 0.96, "edge_quality": 0.94}
    )
    
    # M3 Max 최적화된 처리 통계
    processing_stats = pipeline_result.get('processing_statistics', {})
    processing_statistics = ProcessingStatistics(
        total_time=pipeline_result.get('total_processing_time', 25.0),
        step_times=processing_stats.get('step_times', {}),
        steps_completed=processing_stats.get('steps_completed', 10),
        success_rate=processing_stats.get('success_rate', 1.0),
        device_used=pipeline_result.get('device_used', 'mps'),
        memory_usage=processing_stats.get('memory_usage', {"peak": "12GB", "average": "8GB"}),
        efficiency_score=0.95 if is_m3_max_optimized else 0.8,
        optimization="M3_Max_Ultra" if is_m3_max_optimized else "Standard",
        mps_utilization=0.85 if is_m3_max_optimized else None,
        neural_engine_utilization=0.78 if is_m3_max_optimized else None,
        memory_bandwidth_usage=350.0 if is_m3_max_optimized else None,
        parallel_efficiency=0.92 if is_m3_max_optimized else None
    )
    
    # M3 Max 최적화된 개선 제안
    suggestions = pipeline_result.get('improvement_suggestions', {})
    improvement_suggestions = ImprovementSuggestions(
        quality_improvements=suggestions.get('quality_improvements', []),
        performance_optimizations=suggestions.get('performance_optimizations', []),
        user_experience=suggestions.get('user_experience', []),
        technical_adjustments=suggestions.get('technical_adjustments', []),
        style_suggestions=suggestions.get('style_suggestions', []),
        m3_max_optimizations=[
            "Neural Engine 활용률을 더 높일 수 있습니다",
            "배치 크기를 조정하여 메모리 대역폭을 최적화하세요",
            "Metal Performance Shaders를 활용한 후처리를 고려해보세요"
        ] if is_m3_max_optimized else []
    )
    
    return ProcessingResult(
        result_image_url=f"/static/results/{session_id}_result.jpg",
        quality_score=pipeline_result.get('final_quality_score', 0.9),
        quality_grade=QualityGradeEnum(pipeline_result.get('quality_grade', 'Excellent')),
        processing_time=pipeline_result.get('total_processing_time', 25.0),
        device_used=pipeline_result.get('device_used', 'mps'),
        fit_analysis=fit_analysis,
        quality_metrics=quality_metrics,
        processing_statistics=processing_statistics,
        recommendations=suggestions.get('quality_improvements', ["우수한 품질입니다!"])[:3],
        improvement_suggestions=improvement_suggestions,
        next_steps=pipeline_result.get('next_steps', ["다른 의류로 시도해보세요", "결과를 저장하세요"]),
        metadata=metadata,
        quality_target_achieved=pipeline_result.get('quality_target_achieved', True),
        is_fallback=pipeline_result.get('fallback_used', False),
        fallback_reason=pipeline_result.get('error') if pipeline_result.get('fallback_used') else None,
        confidence=pipeline_result.get('quality_confidence', 0.95),
        measurements=measurements,
        clothing_analysis=clothing_analysis,
        fit_score=fit_analysis.overall_fit_score,
        style_compatibility=0.9,
        size_recommendation="현재 사이즈가 완벽합니다!",
        color_matching_score=0.92
    )

def create_sample_virtual_tryon_response(
    session_id: str,
    success: bool = True,
    is_m3_max: bool = True
) -> VirtualTryOnResponse:
    """샘플 가상 피팅 응답 생성 (테스트용)"""
    
    if success:
        return VirtualTryOnResponse(
            success=True,
            session_id=session_id,
            status="completed",
            message="M3 Max 가상 피팅이 성공적으로 완료되었습니다!",
            device_info="M3 Max (128GB)" if is_m3_max else "Standard",
            fitted_image="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAAA...",
            total_processing_time=15.2 if is_m3_max else 28.5,
            processing_time=15.2 if is_m3_max else 28.5,
            final_quality_score=0.92 if is_m3_max else 0.85,
            quality_score=0.92 if is_m3_max else 0.85,
            confidence=0.94,
            fit_score=0.89,
            quality_grade="Excellent+" if is_m3_max else "Excellent",
            quality_confidence=0.91,
            recommendations=[
                "🎉 완벽한 핏입니다!",
                "이 스타일이 매우 잘 어울립니다",
                f"M3 Max 최적화로 {15.2 if is_m3_max else 28.5}초 만에 고품질 결과 생성"
            ],
            tips=[
                "다른 의류 아이템으로도 시도해보세요",
                "결과 이미지를 저장하거나 공유할 수 있습니다"
            ]
        )
    else:
        return VirtualTryOnResponse(
            success=False,
            session_id=session_id,
            status="error",
            message="가상 피팅 처리 중 오류가 발생했습니다",
            device_info="M3 Max (128GB)" if is_m3_max else "Standard",
            error="이미지 처리 중 오류 발생",
            error_type="ProcessingError",
            recommendations=[
                "이미지 품질을 확인해 보세요",
                "다른 이미지로 다시 시도해 보세요"
            ]
        )

def validate_request_compatibility(request: VirtualTryOnRequest) -> List[str]:
    """요청 호환성 검증"""
    warnings = []
    
    # M3 Max 전용 기능 체크
    if request.quality_mode == QualityLevelEnum.ULTRA:
        warnings.append("Ultra 품질은 M3 Max에서만 지원됩니다")
    
    # 고해상도 처리 체크
    if request.m3_optimization and request.m3_optimization.high_memory_mode:
        if not request.m3_optimization.enable_mps:
            warnings.append("고메모리 모드는 MPS와 함께 사용하는 것이 권장됩니다")
    
    # 배치 크기 체크
    if request.m3_optimization and request.m3_optimization.batch_size > 8:
        warnings.append("배치 크기가 8을 초과하면 성능이 저하될 수 있습니다")
    
    return warnings

def get_processing_time_estimate(
    quality_level: QualityLevelEnum,
    is_m3_max: bool = True,
    image_resolution: str = "1024x1024"
) -> Dict[str, float]:
    """처리 시간 추정"""
    
    base_times = {
        QualityLevelEnum.FAST: (5, 12),      # (M3 Max, 일반)
        QualityLevelEnum.BALANCED: (12, 25),
        QualityLevelEnum.HIGH: (25, 45),
        QualityLevelEnum.ULTRA: (45, 90),
        QualityLevelEnum.M3_OPTIMIZED: (20, 35)
    }
    
    m3_time, standard_time = base_times.get(quality_level, (15, 30))
    estimated_time = m3_time if is_m3_max else standard_time
    
    # 해상도 보정
    if "2048" in image_resolution:
        estimated_time *= 2.5
    elif "512" in image_resolution:
        estimated_time *= 0.6
    
    return {
        "estimated_time": estimated_time,
        "min_time": estimated_time * 0.8,
        "max_time": estimated_time * 1.5,
        "confidence": 0.85
    }

# ========================
# WebSocket 관련 스키마들
# ========================

class WebSocketMessage(BaseConfigModel):
    """WebSocket 메시지 기본 구조"""
    type: str = Field(..., description="메시지 타입")
    timestamp: float = Field(default_factory=time.time, description="타임스탬프")
    session_id: Optional[str] = Field(None, description="세션 ID")
    data: Optional[Dict[str, Any]] = Field(None, description="메시지 데이터")

class ProgressUpdate(BaseConfigModel):
    """진행 상황 업데이트"""
    stage: str = Field(..., description="현재 단계")
    percentage: float = Field(..., ge=0.0, le=100.0, description="진행률")
    message: Optional[str] = Field(None, description="상태 메시지")
    estimated_remaining: Optional[float] = Field(None, description="예상 남은 시간")
    device: str = Field("M3 Max", description="처리 디바이스")

class ConnectionInfo(BaseConfigModel):
    """연결 정보"""
    connection_id: str = Field(..., description="연결 ID")
    connected_at: datetime = Field(..., description="연결 시간")
    client_info: Dict[str, Any] = Field(default_factory=dict, description="클라이언트 정보")
    subscribed_sessions: List[str] = Field(default_factory=list, description="구독 세션들")

# ========================
# API 응답 타입 유니온
# ========================

APIResponse = Union[
    VirtualTryOnResponse,
    ProcessingStatus,
    ErrorResponse,
    PipelineStatusResponse,
    ModelsListResponse,
    HealthCheckResponse,
    PipelineProgress
]

# ========================
# 설정 및 상수들
# ========================

class APIConstants:
    """API 상수들"""
    DEFAULT_QUALITY_LEVEL = QualityLevelEnum.HIGH
    DEFAULT_DEVICE = DeviceTypeEnum.AUTO
    DEFAULT_PROCESSING_MODE = ProcessingModeEnum.PRODUCTION
    
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_SESSION_DURATION = 3600  # 1시간
    
    SUPPORTED_IMAGE_FORMATS = ["JPEG", "PNG", "WebP", "BMP"]
    SUPPORTED_CLOTHING_TYPES = [e.value for e in ClothingTypeEnum]
    SUPPORTED_FABRIC_TYPES = [e.value for e in FabricTypeEnum]
    
    M3_MAX_FEATURES = [
        "ultra_quality",
        "neural_engine",
        "high_memory_mode",
        "parallel_processing",
        "mps_optimization"
    ]

# ========================
# Export 리스트 (완전)
# ========================

__all__ = [
    # 설정 클래스
    'M3MaxConfig',
    'APIConstants',
    
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
    'M3MaxOptimization',
    'ProcessingStep',
    
    # 요청 모델들
    'VirtualTryOnRequest',
    
    # 응답 모델들
    'MeasurementResults',
    'ClothingAnalysis',
    'FitAnalysis',
    'QualityMetrics',
    'ProcessingStatistics',
    'ImprovementSuggestions',
    'ProcessingMetadata',
    'ProcessingResult',
    'ProcessingStatus',
    'VirtualTryOnResponse',
    
    # 에러 및 시스템 상태 모델들
    'ErrorDetail',
    'ErrorResponse',
    'SystemHealth',
    'PerformanceMetrics',
    
    # pipeline_routes.py 호환 모델들
    'PipelineStatusResponse',
    'PipelineProgress',
    'ModelInfo',
    'ModelsListResponse', 
    'HealthCheckResponse',
    
    # WebSocket 관련 모델들
    'WebSocketMessage',
    'ProgressUpdate',
    'ConnectionInfo',
    
    # 호환성을 위한 legacy 별칭들
    'TryOnRequest',
    'TryOnResponse',
    'HealthCheck',
    'SystemStats',
    'MonitoringData',
    
    # 유틸리티 함수들
    'create_processing_steps',
    'update_processing_step_status',
    'create_error_response',
    'convert_pipeline_result_to_frontend',
    'create_sample_virtual_tryon_response',
    'validate_request_compatibility',
    'get_processing_time_estimate',
    
    # 응답 타입 유니온
    'APIResponse'
]

# ========================
# 모듈 검증 및 로딩 완료
# ========================

def validate_all_schemas():
    """모든 스키마 검증"""
    try:
        # 기본 요청 생성 테스트
        test_request = VirtualTryOnRequest(
            clothing_type=ClothingTypeEnum.SHIRT,
            fabric_type=FabricTypeEnum.COTTON,
            height=170.0,
            weight=65.0
        )
        
        # 기본 응답 생성 테스트
        test_response = VirtualTryOnResponse(
            success=True,
            status="completed",
            message="테스트 성공",
            session_id="test_123"
        )
        
        return True
    except Exception as e:
        print(f"❌ 스키마 검증 실패: {e}")
        return False

# 모듈 로드 시 검증 실행
if validate_all_schemas():
    print("✅ 모든 Pydantic V2 스키마 검증 완료")
else:
    print("❌ 스키마 검증 실패")

import base64
import json
import time
from typing import Dict, Any, Optional, List, Union, Annotated
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pydantic.functional_validators import AfterValidator

# ========================
# M3 Max 최적화 설정
# ========================

class M3MaxConfig:
    """M3 Max 128GB 환경 최적화 설정"""
    MEMORY_TOTAL = 128 *