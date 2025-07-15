
"""
모델 및 스키마 패키지
"""

# schemas.py에서 필요한 클래스들 import (안전하게)
try:
    from .schemas import *
except ImportError as e:
    print(f"⚠️ 스키마 import 실패: {e}")
    # 폴백 스키마들
    pass

# 하위 호환성을 위한 별칭들
try:
    from .schemas import (
        VirtualTryOnRequest,
        VirtualTryOnResponse, 
        PipelineProgress,
        QualityMetrics,
        HealthCheck,
        SystemStats
    )
    
    # 별칭 추가
    TryOnRequest = VirtualTryOnRequest
    TryOnResponse = VirtualTryOnResponse
    
except ImportError:
    # 폴백 클래스들
    from pydantic import BaseModel
    from typing import Optional, Dict, Any, List
    
    class VirtualTryOnRequest(BaseModel):
        person_image: str
        clothing_image: str
        clothing_type: str = "shirt"
        body_measurements: Optional[Dict[str, float]] = None
    
    class VirtualTryOnResponse(BaseModel):
        success: bool
        result_image: Optional[str] = None
        error: Optional[str] = None
        processing_time: float = 0.0
    
    class PipelineProgress(BaseModel):
        step: str
        progress: int
        total_steps: int = 8
    
    class QualityMetrics(BaseModel):
        overall_score: float
        detail_scores: Dict[str, float] = {}
    
    class HealthCheck(BaseModel):
        status: str = "healthy"
        version: str = "1.0.0"
    
    class SystemStats(BaseModel):
        memory_usage: Dict[str, Any] = {}
        gpu_usage: Dict[str, Any] = {}
    
    # 별칭
    TryOnRequest = VirtualTryOnRequest
    TryOnResponse = VirtualTryOnResponse

__all__ = [
    'VirtualTryOnRequest',
    'VirtualTryOnResponse', 
    'TryOnRequest',
    'TryOnResponse',
    'PipelineProgress',
    'QualityMetrics',
    'HealthCheck',
    'SystemStats'
]
