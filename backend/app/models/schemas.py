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
    ssim: float = Field(0.0, description="구조적 유사성 지수", ge=0, le=1)
    lpips: float = Field(0.0, description="지각적 유사성", ge=0, le=1)
    fid: Optional[float] = Field(None, description="FID 점수")
    fit_overall: float = Field(0.0, description="전체 피팅 점수", ge=0, le=1)
    fit_coverage: Optional[float] = Field(None, description="커버리지 점수")
    fit_shape_consistency: Optional[float] = Field(None, description="형태 일치도")
    color_preservation: Optional[float] = Field(None, description="색상 보존도")
    boundary_naturalness: Optional[float] = Field(None, description="경계 자연스러움")

class PipelineProgress(BaseModel):
    """파이프라인 진행 상황"""
    step_id: int = Field(..., description="현재 단계 ID")
    step_name: Optional[str] = Field(None, description="단계 이름")
    progress: float = Field(..., description="진행률 (0-1)", ge=0, le=1)
    message: str = Field(..., description="진행 메시지")
    timestamp: float = Field(..., description="타임스탬프")
    processing_time: Optional[float] = Field(None, description="소요 시간")

# pipeline_routes.py에서 필요한 클래스들
class VirtualTryOnRequest(BaseModel):
    """가상 피팅 요청"""
    height: float = Field(170.0, description="키 (cm)")
    weight: float = Field(65.0, description="몸무게 (kg)")
    quality_mode: str = Field("balanced", description="품질 모드")
    connection_id: Optional[str] = Field(None, description="WebSocket 연결 ID")

class VirtualTryOnResponse(BaseModel):
    """가상 피팅 응답"""
    success: bool = Field(..., description="처리 성공 여부")
    fitted_image: Optional[str] = Field(None, description="피팅 결과 이미지 (base64)")
    processing_time: float = Field(0.0, description="총 처리 시간 (초)")
    confidence: float = Field(0.0, description="전체 신뢰도", ge=0, le=1)
    
    # 분석 결과
    measurements: Dict[str, float] = Field(default_factory=dict, description="추출된 신체 측정값")
    clothing_analysis: Dict[str, Any] = Field(default_factory=dict, description="의류 분석 결과")
    fit_score: float = Field(0.0, description="피팅 점수", ge=0, le=1)
    
    # 추천 및 품질
    recommendations: List[str] = Field(default_factory=list, description="추천사항")
    quality_metrics: Dict[str, float] = Field(default_factory=dict, description="품질 메트릭")
    
    # 성능 정보
    memory_usage: Optional[Dict[str, float]] = Field(None, description="메모리 사용량")
    step_times: Optional[Dict[str, float]] = Field(None, description="단계별 처리 시간")
    
    # 오류 정보
    error_message: Optional[str] = Field(None, description="오류 메시지")

class HealthCheck(BaseModel):
    """헬스 체크"""
    status: str = Field(..., description="서비스 상태")
    timestamp: datetime = Field(..., description="체크 시간")
    version: str = Field(..., description="버전")
    uptime: float = Field(..., description="업타임 (초)")

class SystemStats(BaseModel):
    """시스템 통계"""
    total_requests: int = Field(0, description="총 요청 수")
    successful_requests: int = Field(0, description="성공한 요청 수")
    average_processing_time: float = Field(0.0, description="평균 처리 시간")
    average_quality_score: float = Field(0.0, description="평균 품질 점수")
    peak_memory_usage: float = Field(0.0, description="최대 메모리 사용량")
    uptime: float = Field(0.0, description="가동 시간")
    last_request_time: Optional[datetime] = Field(None, description="마지막 요청 시간")

class MonitoringData(BaseModel):
    """모니터링 데이터"""
    cpu_usage: float = Field(0.0, description="CPU 사용률")
    memory_usage: float = Field(0.0, description="메모리 사용률") 
    disk_usage: float = Field(0.0, description="디스크 사용률")
    network_io: Dict[str, float] = Field(default_factory=dict, description="네트워크 I/O")
    active_requests: int = Field(0, description="활성 요청 수")
    queue_size: int = Field(0, description="대기열 크기")
    timestamp: datetime = Field(default_factory=datetime.now, description="측정 시간")
