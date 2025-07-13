"""
MyCloset AI 데이터 스키마
Pydantic 모델로 API 요청/응답 구조 정의
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

class UserMeasurements(BaseModel):
    """사용자 신체 측정 정보"""
    height: float = Field(..., description="키 (cm)", ge=100, le=250)
    weight: float = Field(..., description="몸무게 (kg)", ge=30, le=200)
    chest: Optional[float] = Field(None, description="가슴둘레 (cm)")
    waist: Optional[float] = Field(None, description="허리둘레 (cm)")
    hip: Optional[float] = Field(None, description="엉덩이둘레 (cm)")

class ClothingAnalysis(BaseModel):
    """의류 분석 정보"""
    category: str = Field(..., description="의류 카테고리")
    style: str = Field(..., description="스타일")
    dominant_color: List[int] = Field(..., description="주요 색상 RGB")
    material: Optional[str] = Field(None, description="소재")
    confidence: Optional[float] = Field(None, description="분류 신뢰도")

class QualityMetrics(BaseModel):
    """품질 평가 메트릭"""
    ssim: float = Field(..., description="구조적 유사성 지수", ge=0, le=1)
    lpips: float = Field(..., description="지각적 유사성", ge=0, le=1)
    fid: Optional[float] = Field(None, description="FID 점수")
    fit_overall: float = Field(..., description="전체 피팅 점수", ge=0, le=1)
    fit_coverage: Optional[float] = Field(None, description="커버리지 점수")
    fit_shape_consistency: Optional[float] = Field(None, description="형태 일치도")
    color_preservation: Optional[float] = Field(None, description="색상 보존도")
    boundary_naturalness: Optional[float] = Field(None, description="경계 자연스러움")

class PipelineProgress(BaseModel):
    """파이프라인 진행 상황"""
    step_id: int = Field(..., description="현재 단계 ID")
    step_name: str = Field(..., description="단계 이름")
    progress: float = Field(..., description="진행률 (0-1)", ge=0, le=1)
    message: str = Field(..., description="진행 메시지")
    timestamp: float = Field(..., description="타임스탬프")
    processing_time: Optional[float] = Field(None, description="소요 시간")

class VirtualTryOnRequest(BaseModel):
    """가상 피팅 요청"""
    person_image_base64: str = Field(..., description="사용자 이미지 (base64)")
    clothing_image_base64: str = Field(..., description="의류 이미지 (base64)")
    measurements: UserMeasurements = Field(..., description="신체 측정 정보")
    quality_mode: str = Field("balanced", description="품질 모드", regex="^(fast|balanced|quality)$")
    connection_id: Optional[str] = Field(None, description="WebSocket 연결 ID")

class VirtualTryOnResponse(BaseModel):
    """가상 피팅 응답"""
    success: bool = Field(..., description="처리 성공 여부")
    fitted_image: Optional[str] = Field(None, description="피팅 결과 이미지 (base64)")
    processing_time: float = Field(..., description="총 처리 시간 (초)")
    confidence: float = Field(..., description="전체 신뢰도", ge=0, le=1)
    
    # 분석 결과
    measurements: Dict[str, float] = Field(..., description="추출된 신체 측정값")
    clothing_analysis: ClothingAnalysis = Field(..., description="의류 분석 결과")
    fit_score: float = Field(..., description="피팅 점수", ge=0, le=1)
    
    # 추천 및 품질
    recommendations: List[str] = Field(..., description="추천사항")
    quality_metrics: QualityMetrics = Field(..., description="품질 메트릭")
    
    # 성능 정보
    memory_usage: Optional[Dict[str, float]] = Field(None, description="메모리 사용량")
    step_times: Optional[Dict[str, float]] = Field(None, description="단계별 처리 시간")
    
    # 오류 정보
    error_message: Optional[str] = Field(None, description="오류 메시지")

class PipelineStatus(BaseModel):
    """파이프라인 상태"""
    status: str = Field(..., description="상태")
    device: str = Field(..., description="사용 디바이스")
    memory_usage: Dict[str, float] = Field(..., description="메모리 사용량")
    models_loaded: List[str] = Field(..., description="로드된 모델 목록")
    active_connections: int = Field(..., description="활성 연결 수")

class HealthCheck(BaseModel):
    """헬스 체크"""
    status: str = Field(..., description="서비스 상태")
    timestamp: datetime = Field(..., description="체크 시간")
    version: str = Field(..., description="버전")
    uptime: float = Field(..., description="업타임 (초)")

class ErrorResponse(BaseModel):
    """오류 응답"""
    success: bool = Field(False, description="성공 여부")
    error_code: str = Field(..., description="오류 코드")
    error_message: str = Field(..., description="오류 메시지")
    timestamp: datetime = Field(..., description="오류 발생 시간")
    request_id: Optional[str] = Field(None, description="요청 ID")

# 품질 등급
class QualityGrade(BaseModel):
    """품질 등급"""
    grade: str = Field(..., description="등급", regex="^(Excellent|Good|Fair|Poor)$")
    score: int = Field(..., description="점수 (0-100)", ge=0, le=100)
    description: str = Field(..., description="등급 설명")

# 사용자 친화적 점수
class UserFriendlyScores(BaseModel):
    """사용자 친화적 점수 (0-100)"""
    overall_quality: int = Field(..., description="전체 품질", ge=0, le=100)
    structural_similarity: int = Field(..., description="구조적 유사성", ge=0, le=100)
    perceptual_quality: int = Field(..., description="지각적 품질", ge=0, le=100)
    color_accuracy: int = Field(..., description="색상 정확도", ge=0, le=100)
    texture_quality: int = Field(..., description="텍스처 품질", ge=0, le=100)
    fitting_accuracy: int = Field(..., description="피팅 정확도", ge=0, le=100)
    geometric_precision: int = Field(..., description="기하학적 정밀도", ge=0, le=100)

# 상세 품질 리포트
class DetailedQualityReport(BaseModel):
    """상세 품질 리포트"""
    summary: Dict[str, str] = Field(..., description="요약 정보")
    user_scores: UserFriendlyScores = Field(..., description="사용자 친화적 점수")
    quality_grade: QualityGrade = Field(..., description="품질 등급")
    technical_metrics: Dict[str, float] = Field(..., description="기술적 메트릭")
    recommendations: List[str] = Field(..., description="개선 추천사항")
    processing_details: Dict[str, Any] = Field(..., description="처리 상세 정보")

# 설정 모델
class PipelineConfig(BaseModel):
    """파이프라인 설정"""
    device: str = Field("mps", description="디바이스")
    batch_size: int = Field(1, description="배치 크기", ge=1, le=4)
    image_size: int = Field(512, description="이미지 크기", ge=256, le=1024)
    use_fp16: bool = Field(True, description="FP16 사용 여부")
    enable_caching: bool = Field(True, description="캐싱 활성화")
    parallel_steps: bool = Field(True, description="병렬 처리")
    memory_limit_gb: float = Field(16.0, description="메모리 한계 (GB)")
    quality_threshold: float = Field(0.8, description="품질 임계치")

# 메트릭 히스토리
class MetricHistory(BaseModel):
    """메트릭 히스토리"""
    timestamp: datetime = Field(..., description="기록 시간")
    metrics: QualityMetrics = Field(..., description="품질 메트릭")
    processing_time: float = Field(..., description="처리 시간")
    memory_usage: float = Field(..., description="메모리 사용량")
    quality_mode: str = Field(..., description="품질 모드")

# 통계 정보
class SystemStats(BaseModel):
    """시스템 통계"""
    total_requests: int = Field(..., description="총 요청 수")
    successful_requests: int = Field(..., description="성공한 요청 수")
    average_processing_time: float = Field(..., description="평균 처리 시간")
    average_quality_score: float = Field(..., description="평균 품질 점수")
    peak_memory_usage: float = Field(..., description="최대 메모리 사용량")
    uptime: float = Field(..., description="가동 시간")
    last_request_time: Optional[datetime] = Field(None, description="마지막 요청 시간")

# WebSocket 메시지
class WebSocketMessage(BaseModel):
    """WebSocket 메시지"""
    type: str = Field(..., description="메시지 타입")
    data: Dict[str, Any] = Field(..., description="메시지 데이터")
    timestamp: float = Field(..., description="타임스탬프")

# 모델 정보
class ModelInfo(BaseModel):
    """모델 정보"""
    name: str = Field(..., description="모델 이름")
    version: str = Field(..., description="모델 버전")
    size_mb: float = Field(..., description="모델 크기 (MB)")
    loaded: bool = Field(..., description="로드 상태")
    load_time: Optional[float] = Field(None, description="로드 시간")
    memory_usage_mb: Optional[float] = Field(None, description="메모리 사용량 (MB)")

# 벤치마크 결과
class BenchmarkResult(BaseModel):
    """벤치마크 결과"""
    quality_mode: str = Field(..., description="품질 모드")
    image_size: int = Field(..., description="이미지 크기")
    processing_time: float = Field(..., description="처리 시간")
    memory_usage: float = Field(..., description="메모리 사용량")
    quality_score: float = Field(..., description="품질 점수")
    step_times: Dict[str, float] = Field(..., description="단계별 시간")
    device_info: Dict[str, str] = Field(..., description="디바이스 정보")

# 모니터링 데이터
class MonitoringData(BaseModel):
    """모니터링 데이터"""
    cpu_usage: float = Field(..., description="CPU 사용률")
    memory_usage: float = Field(..., description="메모리 사용률") 
    gpu_usage: Optional[float] = Field(None, description="GPU 사용률")
    disk_usage: float = Field(..., description="디스크 사용률")
    network_io: Dict[str, float] = Field(..., description="네트워크 I/O")
    active_requests: int = Field(..., description="활성 요청 수")
    queue_size: int = Field(..., description="대기열 크기")
    timestamp: datetime = Field(..., description="측정 시간")