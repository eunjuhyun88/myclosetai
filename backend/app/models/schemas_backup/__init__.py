"""
스키마 모듈 초기화
"""

# pipeline_schemas.py에서 모든 클래스 import
from .pipeline_schemas import *

# 추가로 필요한 클래스들 (pipeline_routes.py에서 요구하는)
from .pipeline_schemas import (
    VirtualTryOnRequest as TryOnRequest,  # 별칭 추가
    VirtualTryOnResponse,
    PipelineProgress,
    QualityMetrics,
    HealthCheck,
    SystemStats,
    MonitoringData
)
