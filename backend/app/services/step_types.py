# backend/app/services/step_types.py
"""
🔥 Step Types - 데이터 타입 정의
================================================================================

✅ Enum 클래스들
✅ dataclass 구조체들
✅ 타입 힌팅 정의
✅ 직렬화/역직렬화 메서드

Author: MyCloset AI Team
Date: 2025-08-01
Version: 1.0
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# ==============================================
# 🔥 Enum 클래스들
# ==============================================

class ProcessingMode(Enum):
    """처리 모드"""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"
    EXPERIMENTAL = "experimental"
    BATCH = "batch"
    STREAMING = "streaming"

class ServiceStatus(Enum):
    """서비스 상태"""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    BUSY = "busy"
    SUSPENDED = "suspended"

class ProcessingPriority(Enum):
    """처리 우선순위"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

# ==============================================
# 🔥 데이터 구조체들
# ==============================================

@dataclass
class BodyMeasurements:
    """신체 측정 데이터"""
    height: float
    weight: float
    chest: Optional[float] = None
    waist: Optional[float] = None
    hips: Optional[float] = None
    bmi: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "height": self.height,
            "weight": self.weight,
            "chest": self.chest,
            "waist": self.waist,
            "hips": self.hips,
            "bmi": self.bmi
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BodyMeasurements':
        """딕셔너리에서 생성"""
        return cls(**data)

@dataclass
class ProcessingRequest:
    """처리 요청 데이터 구조"""
    request_id: str
    session_id: str
    step_id: int
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    inputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    timeout: float = 300.0
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "step_id": self.step_id,
            "priority": self.priority.value,
            "inputs": self.inputs,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "timeout": self.timeout
        }

@dataclass
class ProcessingResult:
    """처리 결과 데이터 구조"""
    request_id: str
    session_id: str
    step_id: int
    success: bool
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    processing_time: float = 0.0
    completed_at: datetime = field(default_factory=datetime.now)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "step_id": self.step_id,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "processing_time": self.processing_time,
            "completed_at": self.completed_at.isoformat(),
            "confidence": self.confidence
        }

# ==============================================
# 🔥 타입 별칭들
# ==============================================

StepInput = Dict[str, Any]
StepOutput = Dict[str, Any]
SessionData = Dict[str, Any]
MetricsData = Dict[str, Any] 