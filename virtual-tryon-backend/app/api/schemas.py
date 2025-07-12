# api/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime

class VirtualTryOnRequest(BaseModel):
    """가상 피팅 요청 모델"""
    height: float = Field(..., ge=140, le=220, description="키 (cm)")
    weight: float = Field(..., ge=30, le=150, description="몸무게 (kg)")
    
class BodyMeasurements(BaseModel):
    """신체 측정값"""
    chest: float = Field(..., description="가슴둘레")
    waist: float = Field(..., description="허리둘레")
    hip: float = Field(..., description="엉덩이둘레")
    bmi: float = Field(..., description="BMI")

class ClothingAnalysis(BaseModel):
    """의류 분석 결과"""
    category: str = Field(..., description="의류 카테고리")
    style: str = Field(..., description="스타일")
    dominant_color: List[int] = Field(..., description="주요 색상 (RGB)")

class VirtualTryOnResponse(BaseModel):
    """가상 피팅 응답 모델"""
    success: bool
    fitted_image: str = Field(..., description="Base64 인코딩된 결과 이미지")
    processing_time: float = Field(..., description="처리 시간 (초)")
    confidence: float = Field(..., ge=0, le=1, description="신뢰도")
    measurements: BodyMeasurements
    clothing_analysis: ClothingAnalysis
    fit_score: float = Field(..., ge=0, le=1, description="피팅 점수")
    recommendations: List[str] = Field(..., description="추천사항")

class HealthCheckResponse(BaseModel):
    """헬스체크 응답"""
    status: str
    model_loaded: bool
    device: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ErrorResponse(BaseModel):
    """에러 응답"""
    error: str
    detail: str = None
    timestamp: datetime = Field(default_factory=datetime.now)