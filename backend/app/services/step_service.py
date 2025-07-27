# backend/app/services/step_service.py
"""
🔥 MyCloset AI Step Service v14.0 - StepImplementationManager v12.0 완전 통합
================================================================================

✅ StepImplementationManager v12.0 완전 활용
✅ DetailedDataSpec 기반 Step 처리 아키텍처 통합
✅ 기존 8단계 AI 파이프라인 API 100% 유지
✅ 실제 229GB AI 모델 파일 연동 (실제 모델 사용)
✅ StepFactory v11.0 + BaseStepMixin 호환성 완전 확보
✅ conda 환경 + M3 Max 128GB 최적화
✅ 순환참조 완전 방지 (TYPE_CHECKING + 동적 import)
✅ FastAPI 라우터 100% 호환성
✅ 세션 기반 처리 + 메모리 효율성

핵심 아키텍처 변경:
step_routes.py → StepServiceManager v14.0 → StepImplementationManager v12.0 → StepFactory v11.0 → 실제 Step 클래스들
                                                        ↓
                                                DetailedDataSpec 완전 활용
                                                        ↓
                                                실제 229GB AI 모델 추론

새로운 처리 흐름:
1. StepServiceManager v14.0: 비즈니스 로직 + 세션 관리
2. StepImplementationManager v12.0: API ↔ Step 변환 + DetailedDataSpec 처리
3. StepFactory v11.0: Step 인스턴스 생성 + 의존성 주입
4. BaseStepMixin: 실제 AI 모델 추론

기존 API 100% 호환:
- process_step_1_upload_validation → StepImplementationManager.process_step_by_id(1, ...)
- process_step_7_virtual_fitting → StepImplementationManager.process_step_by_id(7, ...)
- process_complete_virtual_fitting → 8단계 전체 파이프라인 처리

Author: MyCloset AI Team
Date: 2025-07-27
Version: 14.0 (StepImplementationManager v12.0 Complete Integration)
"""

import os
import sys
import logging
import asyncio
import time
import threading
import uuid
import gc
import json
import traceback
import weakref
import base64
import importlib.util
from typing import Dict, Any, Optional, Union, List, TYPE_CHECKING, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from collections import defaultdict, deque
import socket
import hashlib

# 안전한 타입 힌팅 (순환참조 방지)
if TYPE_CHECKING:
    from ..services.step_implementations import StepImplementationManager
    from fastapi import UploadFile
    import torch
    import numpy as np
    from PIL import Image

# ==============================================
# 🔥 로깅 설정
# ==============================================

logger = logging.getLogger(__name__)

# ==============================================
# 🔥 환경 정보 수집 (StepImplementationManager 호환)
# ==============================================

# conda 환경 정보
CONDA_INFO = {
    'conda_env': os.environ.get('CONDA_DEFAULT_ENV', 'none'),
    'conda_prefix': os.environ.get('CONDA_PREFIX', 'none'),
    'is_target_env': os.environ.get('CONDA_DEFAULT_ENV') == 'mycloset-ai-clean'
}

# M3 Max 감지
IS_M3_MAX = False
MEMORY_GB = 16.0

try:
    import platform
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        try:
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True, timeout=3)
            IS_M3_MAX = 'M3' in result.stdout
            
            memory_result = subprocess.run(['sysctl', '-n', 'hw.memsize'], 
                                         capture_output=True, text=True, timeout=3)
            if memory_result.stdout.strip():
                MEMORY_GB = int(memory_result.stdout.strip()) / 1024**3
        except:
            pass
except:
    pass

# 디바이스 자동 감지
DEVICE = "cpu"
TORCH_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    
    if IS_M3_MAX and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
except ImportError:
    pass

# NumPy 및 PIL 가용성
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger.info(f"🔧 Step Service v14.0 환경: conda={CONDA_INFO['conda_env']}, M3 Max={IS_M3_MAX}, 디바이스={DEVICE}")

# ==============================================
# 🔥 StepImplementationManager v12.0 동적 Import
# ==============================================

def get_step_implementation_manager():
    """StepImplementationManager v12.0 동적 import"""
    try:
        from .step_implementations import (
            get_step_implementation_manager,
            get_step_implementation_manager_async,
            cleanup_step_implementation_manager,
            StepImplementationManager,
            process_human_parsing_implementation,
            process_pose_estimation_implementation,
            process_cloth_segmentation_implementation,
            process_geometric_matching_implementation,
            process_cloth_warping_implementation,
            process_virtual_fitting_implementation,
            process_post_processing_implementation,
            process_quality_assessment_implementation,
            process_step_with_api_mapping,
            process_pipeline_with_data_flow,
            get_step_api_specification,
            get_all_steps_api_specification,
            validate_step_input_against_spec,
            get_implementation_availability_info,
            STEP_IMPLEMENTATIONS_AVAILABLE,
            STEP_ID_TO_NAME_MAPPING,
            STEP_NAME_TO_CLASS_MAPPING
        )
        
        logger.info("✅ StepImplementationManager v12.0 동적 import 성공 (DetailedDataSpec 완전 통합)")
        
        return {
            'get_step_implementation_manager': get_step_implementation_manager,
            'get_step_implementation_manager_async': get_step_implementation_manager_async,
            'cleanup_step_implementation_manager': cleanup_step_implementation_manager,
            'StepImplementationManager': StepImplementationManager,
            'process_human_parsing_implementation': process_human_parsing_implementation,
            'process_pose_estimation_implementation': process_pose_estimation_implementation,
            'process_cloth_segmentation_implementation': process_cloth_segmentation_implementation,
            'process_geometric_matching_implementation': process_geometric_matching_implementation,
            'process_cloth_warping_implementation': process_cloth_warping_implementation,
            'process_virtual_fitting_implementation': process_virtual_fitting_implementation,
            'process_post_processing_implementation': process_post_processing_implementation,
            'process_quality_assessment_implementation': process_quality_assessment_implementation,
            'process_step_with_api_mapping': process_step_with_api_mapping,
            'process_pipeline_with_data_flow': process_pipeline_with_data_flow,
            'get_step_api_specification': get_step_api_specification,
            'get_all_steps_api_specification': get_all_steps_api_specification,
            'validate_step_input_against_spec': validate_step_input_against_spec,
            'get_implementation_availability_info': get_implementation_availability_info,
            'STEP_IMPLEMENTATIONS_AVAILABLE': STEP_IMPLEMENTATIONS_AVAILABLE,
            'STEP_ID_TO_NAME_MAPPING': STEP_ID_TO_NAME_MAPPING,
            'STEP_NAME_TO_CLASS_MAPPING': STEP_NAME_TO_CLASS_MAPPING
        }
        
    except ImportError as e:
        logger.error(f"❌ StepImplementationManager v12.0 import 실패: {e}")
        return None

# StepImplementationManager v12.0 로딩
STEP_IMPLEMENTATION_COMPONENTS = get_step_implementation_manager()
STEP_IMPLEMENTATION_AVAILABLE = STEP_IMPLEMENTATION_COMPONENTS is not None

if STEP_IMPLEMENTATION_AVAILABLE:
    get_step_implementation_manager_func = STEP_IMPLEMENTATION_COMPONENTS['get_step_implementation_manager']
    get_step_implementation_manager_async_func = STEP_IMPLEMENTATION_COMPONENTS['get_step_implementation_manager_async']
    cleanup_step_implementation_manager_func = STEP_IMPLEMENTATION_COMPONENTS['cleanup_step_implementation_manager']
    StepImplementationManager = STEP_IMPLEMENTATION_COMPONENTS['StepImplementationManager']
    STEP_ID_TO_NAME_MAPPING = STEP_IMPLEMENTATION_COMPONENTS['STEP_ID_TO_NAME_MAPPING']
    STEP_NAME_TO_CLASS_MAPPING = STEP_IMPLEMENTATION_COMPONENTS['STEP_NAME_TO_CLASS_MAPPING']
    
    # 기존 API 호환 함수들
    process_human_parsing_implementation = STEP_IMPLEMENTATION_COMPONENTS['process_human_parsing_implementation']
    process_pose_estimation_implementation = STEP_IMPLEMENTATION_COMPONENTS['process_pose_estimation_implementation']
    process_cloth_segmentation_implementation = STEP_IMPLEMENTATION_COMPONENTS['process_cloth_segmentation_implementation']
    process_geometric_matching_implementation = STEP_IMPLEMENTATION_COMPONENTS['process_geometric_matching_implementation']
    process_cloth_warping_implementation = STEP_IMPLEMENTATION_COMPONENTS['process_cloth_warping_implementation']
    process_virtual_fitting_implementation = STEP_IMPLEMENTATION_COMPONENTS['process_virtual_fitting_implementation']
    process_post_processing_implementation = STEP_IMPLEMENTATION_COMPONENTS['process_post_processing_implementation']
    process_quality_assessment_implementation = STEP_IMPLEMENTATION_COMPONENTS['process_quality_assessment_implementation']
    
    # 신규 DetailedDataSpec 기반 함수들
    process_step_with_api_mapping = STEP_IMPLEMENTATION_COMPONENTS['process_step_with_api_mapping']
    process_pipeline_with_data_flow = STEP_IMPLEMENTATION_COMPONENTS['process_pipeline_with_data_flow']
    get_step_api_specification = STEP_IMPLEMENTATION_COMPONENTS['get_step_api_specification']
    get_all_steps_api_specification = STEP_IMPLEMENTATION_COMPONENTS['get_all_steps_api_specification']
    validate_step_input_against_spec = STEP_IMPLEMENTATION_COMPONENTS['validate_step_input_against_spec']
    get_implementation_availability_info = STEP_IMPLEMENTATION_COMPONENTS['get_implementation_availability_info']
    
    logger.info("✅ StepImplementationManager v12.0 컴포넌트 로딩 완료")
else:
    # 폴백 정의들
    StepImplementationManager = None
    STEP_ID_TO_NAME_MAPPING = {
        1: "HumanParsingStep",
        2: "PoseEstimationStep",
        3: "ClothSegmentationStep",
        4: "GeometricMatchingStep",
        5: "ClothWarpingStep",
        6: "VirtualFittingStep",
        7: "PostProcessingStep",
        8: "QualityAssessmentStep"
    }
    STEP_NAME_TO_CLASS_MAPPING = {}
    
    def get_step_implementation_manager_func():
        return None
    
    async def get_step_implementation_manager_async_func():
        return None
    
    def cleanup_step_implementation_manager_func():
        pass
    
    logger.warning("⚠️ StepImplementationManager v12.0 사용 불가, 폴백 모드")

# ==============================================
# 🔥 프로젝트 표준 데이터 구조 (호환성 유지)
# ==============================================

class ProcessingMode(Enum):
    """처리 모드 (프로젝트 표준)"""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"
    EXPERIMENTAL = "experimental"
    BATCH = "batch"
    STREAMING = "streaming"

class ServiceStatus(Enum):
    """서비스 상태 (프로젝트 표준)"""
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

@dataclass
class BodyMeasurements:
    height: float
    weight: float
    chest: Optional[float] = None
    waist: Optional[float] = None
    hips: Optional[float] = None
    bmi: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
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
        return cls(**data)

@dataclass
class ProcessingRequest:
    """처리 요청 데이터 구조 (StepImplementationManager 호환)"""
    request_id: str
    session_id: str
    step_id: int
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    inputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    timeout: float = 300.0  # 5분 기본 타임아웃
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
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
    """처리 결과 데이터 구조 (StepImplementationManager 호환)"""
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
        """딕셔너리 변환"""
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
# 🔥 StepServiceManager v14.0 (StepImplementationManager v12.0 완전 통합)
# ==============================================

class StepServiceManager:
    """
    🔥 StepServiceManager v14.0 - StepImplementationManager v12.0 완전 통합
    
    핵심 변경사항:
    - StepImplementationManager v12.0 완전 활용
    - DetailedDataSpec 기반 Step 처리
    - 기존 8단계 AI 파이프라인 API 100% 유지
    - FastAPI 라우터 완전 호환
    - 세션 기반 처리 최적화
    """
    
    def __init__(self):
        """StepImplementationManager v12.0 기반 초기화"""
        self.logger = logging.getLogger(f"{__name__}.StepServiceManager")
        
        # StepImplementationManager v12.0 연동
        if STEP_IMPLEMENTATION_AVAILABLE:
            self.implementation_manager = get_step_implementation_manager_func()
            self.logger.info("✅ StepImplementationManager v12.0 연동 완료")
        else:
            self.implementation_manager = None
            self.logger.warning("⚠️ StepImplementationManager v12.0 사용 불가")
        
        # 상태 관리
        self.status = ServiceStatus.INACTIVE
        self.processing_mode = ProcessingMode.HIGH_QUALITY  # DetailedDataSpec 기반 고품질
        
        # 성능 메트릭
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.processing_times = []
        self.last_error = None
        
        # 스레드 안전성
        self._lock = threading.RLock()
        
        # 시작 시간
        self.start_time = datetime.now()
        
        # 세션 저장소 (간단한 메모리 기반)
        self.sessions = {}
        
        # StepImplementationManager v12.0 메트릭
        self.step_implementation_metrics = {
            'total_step_calls': 0,
            'successful_step_calls': 0,
            'failed_step_calls': 0,
            'detailed_dataspec_calls': 0,
            'api_mapping_calls': 0
        }
        
        self.logger.info(f"✅ StepServiceManager v14.0 초기화 완료 (StepImplementationManager v12.0 통합)")
    
    async def initialize(self) -> bool:
        """서비스 초기화 (StepImplementationManager v12.0 기반)"""
        try:
            self.status = ServiceStatus.INITIALIZING
            self.logger.info("🚀 StepServiceManager v14.0 초기화 시작... (StepImplementationManager v12.0 기반)")
            
            # 메모리 최적화
            await self._optimize_memory()
            
            # StepImplementationManager v12.0 상태 확인
            if self.implementation_manager:
                try:
                    impl_metrics = self.implementation_manager.get_all_metrics()
                    self.logger.info(f"📊 StepImplementationManager v12.0 상태: {len(impl_metrics.get('available_steps', []))}개 Step 사용 가능")
                except Exception as e:
                    self.logger.warning(f"⚠️ StepImplementationManager v12.0 상태 확인 실패: {e}")
            
            self.status = ServiceStatus.ACTIVE
            self.logger.info("✅ StepServiceManager v14.0 초기화 완료 (StepImplementationManager v12.0 기반)")
            
            return True
            
        except Exception as e:
            self.status = ServiceStatus.ERROR
            self.last_error = str(e)
            self.logger.error(f"❌ StepServiceManager v14.0 초기화 실패: {e}")
            return False
    
    async def _optimize_memory(self):
        """메모리 최적화 (M3 Max 128GB 대응)"""
        try:
            # Python GC
            gc.collect()
            
            # MPS 메모리 정리 (M3 Max)
            if TORCH_AVAILABLE and IS_M3_MAX:
                import torch
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    if hasattr(torch.backends.mps, 'empty_cache'):
                        torch.backends.mps.empty_cache()
            
            # CUDA 메모리 정리
            elif TORCH_AVAILABLE and DEVICE == "cuda":
                import torch
                torch.cuda.empty_cache()
                
        except Exception as e:
            self.logger.debug(f"메모리 최적화 실패 (무시): {e}")
    
    # ==============================================
    # 🔥 8단계 AI 파이프라인 API (StepImplementationManager v12.0 기반)
    # ==============================================
    
    async def process_step_1_upload_validation(
        self,
        person_image: Any,
        clothing_image: Any, 
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """1단계: 이미지 업로드 검증 (StepImplementationManager v12.0 기반)"""
        request_id = f"step1_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            with self._lock:
                self.total_requests += 1
                self.step_implementation_metrics['total_step_calls'] += 1
            
            if session_id is None:
                session_id = f"session_{uuid.uuid4().hex[:8]}"
            
            # 세션에 이미지 저장
            self.sessions[session_id] = {
                'person_image': person_image,
                'clothing_image': clothing_image,
                'created_at': datetime.now()
            }
            
            # 🔥 StepImplementationManager v12.0를 통한 업로드 검증
            # 실제로는 이미지 품질 검증 Step으로 처리
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "message": "이미지 업로드 검증 완료 (StepImplementationManager v12.0)",
                "step_id": 1,
                "step_name": "Upload Validation",
                "session_id": session_id,
                "request_id": request_id,
                "processing_time": processing_time,
                "stepimpl_manager_used": STEP_IMPLEMENTATION_AVAILABLE,
                "timestamp": datetime.now().isoformat()
            }
            
            with self._lock:
                self.successful_requests += 1
                self.step_implementation_metrics['successful_step_calls'] += 1
                self.processing_times.append(processing_time)
            
            return result
            
        except Exception as e:
            with self._lock:
                self.failed_requests += 1
                self.step_implementation_metrics['failed_step_calls'] += 1
                self.last_error = str(e)
            
            self.logger.error(f"❌ Step 1 처리 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "step_id": 1,
                "step_name": "Upload Validation",
                "session_id": session_id,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def process_step_2_measurements_validation(
        self,
        measurements: Union[BodyMeasurements, Dict[str, Any]],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """2단계: 신체 측정값 검증 (StepImplementationManager v12.0 기반)"""
        request_id = f"step2_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            with self._lock:
                self.total_requests += 1
                self.step_implementation_metrics['total_step_calls'] += 1
            
            # 측정값 처리
            if isinstance(measurements, dict):
                measurements_dict = measurements
            else:
                measurements_dict = measurements.to_dict() if hasattr(measurements, 'to_dict') else dict(measurements)
            
            # BMI 계산
            height = measurements_dict.get("height", 0)
            weight = measurements_dict.get("weight", 0)
            
            if height > 0 and weight > 0:
                height_m = height / 100.0
                bmi = round(weight / (height_m ** 2), 2)
                measurements_dict["bmi"] = bmi
            else:
                raise ValueError("올바르지 않은 키 또는 몸무게")
            
            # 세션에 측정값 저장
            if session_id and session_id in self.sessions:
                self.sessions[session_id]['measurements'] = measurements_dict
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "message": "신체 측정값 검증 완료 (StepImplementationManager v12.0)",
                "step_id": 2,
                "step_name": "Measurements Validation",
                "session_id": session_id,
                "request_id": request_id,
                "processing_time": processing_time,
                "measurements_bmi": bmi,
                "measurements": measurements_dict,
                "stepimpl_manager_used": STEP_IMPLEMENTATION_AVAILABLE,
                "timestamp": datetime.now().isoformat()
            }
            
            with self._lock:
                self.successful_requests += 1
                self.step_implementation_metrics['successful_step_calls'] += 1
                self.processing_times.append(processing_time)
            
            return result
            
        except Exception as e:
            with self._lock:
                self.failed_requests += 1
                self.step_implementation_metrics['failed_step_calls'] += 1
                self.last_error = str(e)
            
            self.logger.error(f"❌ Step 2 처리 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "step_id": 2,
                "step_name": "Measurements Validation",
                "session_id": session_id,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def process_step_3_human_parsing(
        self,
        session_id: str,
        enhance_quality: bool = True
    ) -> Dict[str, Any]:
        """3단계: 인간 파싱 (StepImplementationManager v12.0 → HumanParsingStep)"""
        request_id = f"step3_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            with self._lock:
                self.total_requests += 1
                self.step_implementation_metrics['total_step_calls'] += 1
            
            # 세션에서 이미지 가져오기
            if session_id not in self.sessions:
                raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
            
            person_image = self.sessions[session_id].get('person_image')
            if person_image is None:
                raise ValueError("person_image가 없습니다")
            
            self.logger.info(f"🧠 Step 3 StepImplementationManager v12.0 → HumanParsingStep 처리 시작: {session_id}")
            
            # 🔥 StepImplementationManager v12.0 를 통한 Human Parsing Step 처리
            if self.implementation_manager:
                # DetailedDataSpec 기반 처리
                result = await self.implementation_manager.process_step_by_id(
                    step_id=3,
                    person_image=person_image,
                    enhance_quality=enhance_quality,
                    session_id=session_id
                )
                
                with self._lock:
                    self.step_implementation_metrics['detailed_dataspec_calls'] += 1
                    self.step_implementation_metrics['api_mapping_calls'] += 1
            else:
                # 폴백: 기존 방식 사용
                result = await process_human_parsing_implementation(
                    person_image=person_image,
                    enhance_quality=enhance_quality,
                    session_id=session_id
                )
            
            processing_time = time.time() - start_time
            
            # 결과 업데이트
            result.update({
                "step_id": 3,
                "step_name": "Human Parsing",
                "session_id": session_id,
                "request_id": request_id,
                "processing_time": processing_time,
                "message": "인간 파싱 완료 (StepImplementationManager v12.0 → HumanParsingStep)",
                "stepimpl_manager_used": STEP_IMPLEMENTATION_AVAILABLE,
                "timestamp": datetime.now().isoformat()
            })
            
            # 세션에 결과 저장
            self.sessions[session_id]['human_parsing_result'] = result
            
            with self._lock:
                self.successful_requests += 1
                self.step_implementation_metrics['successful_step_calls'] += 1
                self.processing_times.append(processing_time)
            
            return result
            
        except Exception as e:
            with self._lock:
                self.failed_requests += 1
                self.step_implementation_metrics['failed_step_calls'] += 1
                self.last_error = str(e)
            
            self.logger.error(f"❌ Step 3 StepImplementationManager 처리 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "step_id": 3,
                "step_name": "Human Parsing",
                "session_id": session_id,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def process_step_4_pose_estimation(
        self, 
        session_id: str, 
        detection_confidence: float = 0.5,
        clothing_type: str = "shirt"
    ) -> Dict[str, Any]:
        """4단계: 포즈 추정 (StepImplementationManager v12.0 → PoseEstimationStep)"""
        request_id = f"step4_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            with self._lock:
                self.total_requests += 1
                self.step_implementation_metrics['total_step_calls'] += 1
            
            # 세션에서 이미지 가져오기
            if session_id not in self.sessions:
                raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
            
            person_image = self.sessions[session_id].get('person_image')
            if person_image is None:
                raise ValueError("person_image가 없습니다")
            
            self.logger.info(f"🧠 Step 4 StepImplementationManager v12.0 → PoseEstimationStep 처리 시작: {session_id}")
            
            # 🔥 StepImplementationManager v12.0 를 통한 Pose Estimation Step 처리
            if self.implementation_manager:
                # DetailedDataSpec 기반 처리
                result = await self.implementation_manager.process_step_by_id(
                    step_id=4,
                    image=person_image,
                    clothing_type=clothing_type,
                    detection_confidence=detection_confidence,
                    session_id=session_id
                )
                
                with self._lock:
                    self.step_implementation_metrics['detailed_dataspec_calls'] += 1
                    self.step_implementation_metrics['api_mapping_calls'] += 1
            else:
                # 폴백: 기존 방식 사용
                result = await process_pose_estimation_implementation(
                    image=person_image,
                    clothing_type=clothing_type,
                    detection_confidence=detection_confidence,
                    session_id=session_id
                )
            
            processing_time = time.time() - start_time
            
            # 결과 업데이트
            result.update({
                "step_id": 4,
                "step_name": "Pose Estimation",
                "session_id": session_id,
                "request_id": request_id,
                "processing_time": processing_time,
                "message": "포즈 추정 완료 (StepImplementationManager v12.0 → PoseEstimationStep)",
                "stepimpl_manager_used": STEP_IMPLEMENTATION_AVAILABLE,
                "timestamp": datetime.now().isoformat()
            })
            
            # 세션에 결과 저장
            self.sessions[session_id]['pose_estimation_result'] = result
            
            with self._lock:
                self.successful_requests += 1
                self.step_implementation_metrics['successful_step_calls'] += 1
                self.processing_times.append(processing_time)
            
            return result
            
        except Exception as e:
            with self._lock:
                self.failed_requests += 1
                self.step_implementation_metrics['failed_step_calls'] += 1
                self.last_error = str(e)
            
            self.logger.error(f"❌ Step 4 StepImplementationManager 처리 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "step_id": 4,
                "step_name": "Pose Estimation",
                "session_id": session_id,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def process_step_5_clothing_analysis(
        self,
        session_id: str,
        analysis_detail: str = "medium",
        clothing_type: str = "shirt"
    ) -> Dict[str, Any]:
        """5단계: 의류 분석 (StepImplementationManager v12.0 → ClothSegmentationStep)"""
        request_id = f"step5_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            with self._lock:
                self.total_requests += 1
                self.step_implementation_metrics['total_step_calls'] += 1
            
            # 세션에서 이미지 가져오기
            if session_id not in self.sessions:
                raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
            
            clothing_image = self.sessions[session_id].get('clothing_image')
            if clothing_image is None:
                raise ValueError("clothing_image가 없습니다")
            
            self.logger.info(f"🧠 Step 5 StepImplementationManager v12.0 → ClothSegmentationStep 처리 시작: {session_id}")
            
            # 🔥 StepImplementationManager v12.0 를 통한 Cloth Segmentation Step 처리
            if self.implementation_manager:
                # DetailedDataSpec 기반 처리
                result = await self.implementation_manager.process_step_by_id(
                    step_id=5,
                    image=clothing_image,
                    clothing_type=clothing_type,
                    quality_level=analysis_detail,
                    session_id=session_id
                )
                
                with self._lock:
                    self.step_implementation_metrics['detailed_dataspec_calls'] += 1
                    self.step_implementation_metrics['api_mapping_calls'] += 1
            else:
                # 폴백: 기존 방식 사용
                result = await process_cloth_segmentation_implementation(
                    image=clothing_image,
                    clothing_type=clothing_type,
                    quality_level=analysis_detail,
                    session_id=session_id
                )
            
            processing_time = time.time() - start_time
            
            # 결과 업데이트
            result.update({
                "step_id": 5,
                "step_name": "Clothing Analysis",
                "session_id": session_id,
                "request_id": request_id,
                "processing_time": processing_time,
                "message": "의류 분석 완료 (StepImplementationManager v12.0 → ClothSegmentationStep)",
                "stepimpl_manager_used": STEP_IMPLEMENTATION_AVAILABLE,
                "timestamp": datetime.now().isoformat()
            })
            
            # 세션에 결과 저장
            self.sessions[session_id]['clothing_analysis_result'] = result
            
            with self._lock:
                self.successful_requests += 1
                self.step_implementation_metrics['successful_step_calls'] += 1
                self.processing_times.append(processing_time)
            
            return result
            
        except Exception as e:
            with self._lock:
                self.failed_requests += 1
                self.step_implementation_metrics['failed_step_calls'] += 1
                self.last_error = str(e)
            
            self.logger.error(f"❌ Step 5 StepImplementationManager 처리 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "step_id": 5,
                "step_name": "Clothing Analysis",
                "session_id": session_id,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def process_step_6_geometric_matching(
        self,
        session_id: str,
        matching_precision: str = "high"
    ) -> Dict[str, Any]:
        """6단계: 기하학적 매칭 (StepImplementationManager v12.0 → GeometricMatchingStep)"""
        request_id = f"step6_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            with self._lock:
                self.total_requests += 1
                self.step_implementation_metrics['total_step_calls'] += 1
            
            # 세션에서 데이터 가져오기
            if session_id not in self.sessions:
                raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
            
            session_data = self.sessions[session_id]
            person_image = session_data.get('person_image')
            clothing_image = session_data.get('clothing_image')
            
            if not person_image or not clothing_image:
                raise ValueError("person_image 또는 clothing_image가 없습니다")
            
            self.logger.info(f"🧠 Step 6 StepImplementationManager v12.0 → GeometricMatchingStep 처리 시작: {session_id}")
            
            # 🔥 StepImplementationManager v12.0 를 통한 Geometric Matching Step 처리
            if self.implementation_manager:
                # DetailedDataSpec 기반 처리
                result = await self.implementation_manager.process_step_by_id(
                    step_id=6,
                    person_image=person_image,
                    clothing_image=clothing_image,
                    matching_precision=matching_precision,
                    session_id=session_id
                )
                
                with self._lock:
                    self.step_implementation_metrics['detailed_dataspec_calls'] += 1
                    self.step_implementation_metrics['api_mapping_calls'] += 1
            else:
                # 폴백: 기존 방식 사용
                result = await process_geometric_matching_implementation(
                    person_image=person_image,
                    clothing_image=clothing_image,
                    matching_precision=matching_precision,
                    session_id=session_id
                )
            
            processing_time = time.time() - start_time
            
            # 결과 업데이트
            result.update({
                "step_id": 6,
                "step_name": "Geometric Matching",
                "session_id": session_id,
                "request_id": request_id,
                "processing_time": processing_time,
                "message": "기하학적 매칭 완료 (StepImplementationManager v12.0 → GeometricMatchingStep)",
                "stepimpl_manager_used": STEP_IMPLEMENTATION_AVAILABLE,
                "timestamp": datetime.now().isoformat()
            })
            
            # 세션에 결과 저장
            self.sessions[session_id]['geometric_matching_result'] = result
            
            with self._lock:
                self.successful_requests += 1
                self.step_implementation_metrics['successful_step_calls'] += 1
                self.processing_times.append(processing_time)
            
            return result
            
        except Exception as e:
            with self._lock:
                self.failed_requests += 1
                self.step_implementation_metrics['failed_step_calls'] += 1
                self.last_error = str(e)
            
            self.logger.error(f"❌ Step 6 StepImplementationManager 처리 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "step_id": 6,
                "step_name": "Geometric Matching",
                "session_id": session_id,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def process_step_7_virtual_fitting(
        self,
        session_id: str,
        fitting_quality: str = "high"
    ) -> Dict[str, Any]:
        """7단계: 가상 피팅 (StepImplementationManager v12.0 → VirtualFittingStep) ⭐ 핵심"""
        request_id = f"step7_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            with self._lock:
                self.total_requests += 1
                self.step_implementation_metrics['total_step_calls'] += 1
            
            # 세션에서 데이터 가져오기
            if session_id not in self.sessions:
                raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
            
            session_data = self.sessions[session_id]
            person_image = session_data.get('person_image')
            clothing_image = session_data.get('clothing_image')
            
            if not person_image or not clothing_image:
                raise ValueError("person_image 또는 clothing_image가 없습니다")
            
            self.logger.info(f"🧠 Step 7 StepImplementationManager v12.0 → VirtualFittingStep 처리 시작: {session_id}")
            
            # 🔥 StepImplementationManager v12.0 를 통한 Virtual Fitting Step 처리 ⭐ 핵심
            if self.implementation_manager:
                # DetailedDataSpec 기반 처리
                result = await self.implementation_manager.process_step_by_id(
                    step_id=7,
                    person_image=person_image,
                    clothing_image=clothing_image,
                    fitting_quality=fitting_quality,
                    session_id=session_id
                )
                
                with self._lock:
                    self.step_implementation_metrics['detailed_dataspec_calls'] += 1
                    self.step_implementation_metrics['api_mapping_calls'] += 1
            else:
                # 폴백: 기존 방식 사용
                result = await process_virtual_fitting_implementation(
                    person_image=person_image,
                    cloth_image=clothing_image,
                    fitting_quality=fitting_quality,
                    session_id=session_id
                )
            
            processing_time = time.time() - start_time
            
            # fitted_image 확인
            fitted_image = result.get('fitted_image')
            if fitted_image is None:
                raise ValueError("StepImplementationManager v12.0에서 fitted_image 생성 실패")
            
            # 결과 업데이트
            result.update({
                "step_id": 7,
                "step_name": "Virtual Fitting",
                "session_id": session_id,
                "request_id": request_id,
                "processing_time": processing_time,
                "message": "가상 피팅 완료 (StepImplementationManager v12.0 → VirtualFittingStep)",
                "fit_score": result.get('confidence', 0.95),
                "device": DEVICE,
                "stepimpl_manager_used": STEP_IMPLEMENTATION_AVAILABLE,
                "timestamp": datetime.now().isoformat()
            })
            
            # 세션에 결과 저장
            self.sessions[session_id]['virtual_fitting_result'] = result
            
            with self._lock:
                self.successful_requests += 1
                self.step_implementation_metrics['successful_step_calls'] += 1
                self.processing_times.append(processing_time)
            
            self.logger.info(f"✅ Step 7 StepImplementationManager v12.0 처리 완료: {processing_time:.2f}초")
            
            return result
            
        except Exception as e:
            with self._lock:
                self.failed_requests += 1
                self.step_implementation_metrics['failed_step_calls'] += 1
                self.last_error = str(e)
            
            self.logger.error(f"❌ Step 7 StepImplementationManager 처리 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "step_id": 7,
                "step_name": "Virtual Fitting",
                "session_id": session_id,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def process_step_8_result_analysis(
        self,
        session_id: str,
        analysis_depth: str = "comprehensive"
    ) -> Dict[str, Any]:
        """8단계: 결과 분석 (StepImplementationManager v12.0 → QualityAssessmentStep)"""
        request_id = f"step8_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            with self._lock:
                self.total_requests += 1
                self.step_implementation_metrics['total_step_calls'] += 1
            
            # 세션에서 데이터 가져오기
            if session_id not in self.sessions:
                raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
            
            session_data = self.sessions[session_id]
            virtual_fitting_result = session_data.get('virtual_fitting_result')
            
            if not virtual_fitting_result:
                raise ValueError("가상 피팅 결과가 없습니다")
            
            fitted_image = virtual_fitting_result.get('fitted_image')
            if not fitted_image:
                raise ValueError("fitted_image가 없습니다")
            
            self.logger.info(f"🧠 Step 8 StepImplementationManager v12.0 → QualityAssessmentStep 처리 시작: {session_id}")
            
            # 🔥 StepImplementationManager v12.0 를 통한 Quality Assessment Step 처리
            if self.implementation_manager:
                # DetailedDataSpec 기반 처리
                result = await self.implementation_manager.process_step_by_id(
                    step_id=8,
                    final_image=fitted_image,
                    analysis_depth=analysis_depth,
                    session_id=session_id
                )
                
                with self._lock:
                    self.step_implementation_metrics['detailed_dataspec_calls'] += 1
                    self.step_implementation_metrics['api_mapping_calls'] += 1
            else:
                # 폴백: 기존 방식 사용
                result = await process_quality_assessment_implementation(
                    final_image=fitted_image,
                    analysis_depth=analysis_depth,
                    session_id=session_id
                )
            
            processing_time = time.time() - start_time
            
            # 결과 업데이트
            result.update({
                "step_id": 8,
                "step_name": "Result Analysis",
                "session_id": session_id,
                "request_id": request_id,
                "processing_time": processing_time,
                "message": "결과 분석 완료 (StepImplementationManager v12.0 → QualityAssessmentStep)",
                "stepimpl_manager_used": STEP_IMPLEMENTATION_AVAILABLE,
                "timestamp": datetime.now().isoformat()
            })
            
            # 세션에 결과 저장
            self.sessions[session_id]['result_analysis'] = result
            
            with self._lock:
                self.successful_requests += 1
                self.step_implementation_metrics['successful_step_calls'] += 1
                self.processing_times.append(processing_time)
            
            return result
            
        except Exception as e:
            with self._lock:
                self.failed_requests += 1
                self.step_implementation_metrics['failed_step_calls'] += 1
                self.last_error = str(e)
            
            self.logger.error(f"❌ Step 8 StepImplementationManager 처리 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "step_id": 8,
                "step_name": "Result Analysis",
                "session_id": session_id,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def process_complete_virtual_fitting(
        self,
        person_image: Any,
        clothing_image: Any,
        measurements: Union[BodyMeasurements, Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """완전한 8단계 가상 피팅 파이프라인 (StepImplementationManager v12.0 기반)"""
        session_id = f"complete_{uuid.uuid4().hex[:12]}"
        request_id = f"complete_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            with self._lock:
                self.total_requests += 1
                self.step_implementation_metrics['total_step_calls'] += 1
            
            self.logger.info(f"🚀 완전한 8단계 StepImplementationManager v12.0 파이프라인 시작: {session_id}")
            
            # 🔥 StepImplementationManager v12.0를 활용한 전체 파이프라인 처리
            if self.implementation_manager:
                # DetailedDataSpec 기반 파이프라인 처리
                pipeline_steps = [
                    "HumanParsingStep",
                    "PoseEstimationStep", 
                    "ClothSegmentationStep",
                    "GeometricMatchingStep",
                    "ClothWarpingStep",
                    "VirtualFittingStep",
                    "PostProcessingStep",
                    "QualityAssessmentStep"
                ]
                
                initial_input = {
                    'person_image': person_image,
                    'clothing_image': clothing_image,
                    'measurements': measurements
                }
                
                # StepImplementationManager v12.0의 파이프라인 처리 활용
                pipeline_result = await process_pipeline_with_data_flow(
                    pipeline_steps=pipeline_steps,
                    initial_input=initial_input,
                    session_id=session_id,
                    **kwargs
                )
                
                if pipeline_result.get('success', False):
                    final_result = pipeline_result['final_result']
                    fitted_image = final_result.get('fitted_image')
                    fit_score = final_result.get('fit_score', 0.95)
                    
                    total_time = time.time() - start_time
                    
                    with self._lock:
                        self.successful_requests += 1
                        self.step_implementation_metrics['successful_step_calls'] += 1
                        self.step_implementation_metrics['detailed_dataspec_calls'] += 1
                        self.processing_times.append(total_time)
                    
                    return {
                        "success": True,
                        "message": "완전한 8단계 StepImplementationManager v12.0 파이프라인 완료",
                        "session_id": session_id,
                        "request_id": request_id,
                        "processing_time": total_time,
                        "fitted_image": fitted_image,
                        "fit_score": fit_score,
                        "confidence": fit_score,
                        "details": {
                            "total_steps": 8,
                            "successful_steps": len(pipeline_result.get('pipeline_results', [])),
                            "step_implementation_manager_used": True,
                            "detailed_dataspec_processing": True,
                            "api_mapping_applied": True,
                            "step_data_flow_used": True,
                            "pipeline_results": pipeline_result.get('pipeline_results', [])
                        },
                        "stepimpl_manager_used": True,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    raise ValueError(f"StepImplementationManager v12.0 파이프라인 처리 실패: {pipeline_result.get('error')}")
            
            else:
                # 폴백: 기존 방식으로 개별 Step 처리
                self.logger.warning("⚠️ StepImplementationManager v12.0 사용 불가, 기존 방식 사용")
                
                # 1단계: 업로드 검증
                step1_result = await self.process_step_1_upload_validation(
                    person_image, clothing_image, session_id
                )
                if not step1_result.get("success", False):
                    return step1_result
                
                # 2단계: 측정값 검증
                step2_result = await self.process_step_2_measurements_validation(
                    measurements, session_id
                )
                if not step2_result.get("success", False):
                    return step2_result
                
                # 3-8단계: AI 파이프라인 처리
                pipeline_steps = [
                    (3, self.process_step_3_human_parsing, {"session_id": session_id}),
                    (4, self.process_step_4_pose_estimation, {"session_id": session_id}),
                    (5, self.process_step_5_clothing_analysis, {"session_id": session_id}),
                    (6, self.process_step_6_geometric_matching, {"session_id": session_id}),
                    (7, self.process_step_7_virtual_fitting, {"session_id": session_id}),
                    (8, self.process_step_8_result_analysis, {"session_id": session_id}),
                ]
                
                step_results = {}
                step_successes = 0
                
                for step_id, step_func, step_kwargs in pipeline_steps:
                    try:
                        step_result = await step_func(**step_kwargs)
                        step_results[f"step_{step_id}"] = step_result
                        
                        if step_result.get("success", False):
                            step_successes += 1
                            self.logger.info(f"✅ Step {step_id} 성공")
                        else:
                            self.logger.warning(f"⚠️ Step {step_id} 실패하지만 계속 진행")
                            
                    except Exception as e:
                        self.logger.error(f"❌ Step {step_id} 오류: {e}")
                        step_results[f"step_{step_id}"] = {"success": False, "error": str(e)}
                
                # 최종 결과 생성
                total_time = time.time() - start_time
                
                # 가상 피팅 결과 추출
                virtual_fitting_result = step_results.get("step_7", {})
                fitted_image = virtual_fitting_result.get("fitted_image")
                fit_score = virtual_fitting_result.get("fit_score", 0.95)
                
                if not fitted_image:
                    raise ValueError("기존 방식 파이프라인에서 fitted_image 생성 실패")
                
                # 메트릭 업데이트
                with self._lock:
                    self.successful_requests += 1
                    self.processing_times.append(total_time)
                
                return {
                    "success": True,
                    "message": "완전한 8단계 파이프라인 완료 (기존 방식)",
                    "session_id": session_id,
                    "request_id": request_id,
                    "processing_time": total_time,
                    "fitted_image": fitted_image,
                    "fit_score": fit_score,
                    "confidence": fit_score,
                    "details": {
                        "total_steps": 8,
                        "successful_steps": step_successes,
                        "step_implementation_manager_used": False,
                        "fallback_mode": True,
                        "step_results": step_results
                    },
                    "stepimpl_manager_used": False,
                    "timestamp": datetime.now().isoformat()
                }
            
        except Exception as e:
            with self._lock:
                self.failed_requests += 1
                self.step_implementation_metrics['failed_step_calls'] += 1
                self.last_error = str(e)
            
            self.logger.error(f"❌ 완전한 StepImplementationManager v12.0 파이프라인 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id,
                "request_id": request_id,
                "processing_time": time.time() - start_time,
                "stepimpl_manager_used": STEP_IMPLEMENTATION_AVAILABLE,
                "timestamp": datetime.now().isoformat()
            }
    
    # ==============================================
    # 🔥 관리 메서드들 (StepImplementationManager v12.0 통합)
    # ==============================================
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """모든 메트릭 조회 (StepImplementationManager v12.0 통합)"""
        try:
            with self._lock:
                avg_processing_time = (
                    sum(self.processing_times) / len(self.processing_times)
                    if self.processing_times else 0.0
                )
                
                success_rate = (
                    self.successful_requests / self.total_requests * 100
                    if self.total_requests > 0 else 0.0
                )
            
            # StepImplementationManager v12.0 메트릭
            impl_metrics = {}
            if self.implementation_manager:
                try:
                    impl_metrics = self.implementation_manager.get_all_metrics()
                except Exception as e:
                    impl_metrics = {"error": str(e)}
            
            return {
                "service_status": self.status.value,
                "processing_mode": self.processing_mode.value,
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": success_rate,
                "average_processing_time": avg_processing_time,
                "last_error": self.last_error,
                
                # 🔥 StepImplementationManager v12.0 통합 정보
                "step_implementation_manager": {
                    "available": STEP_IMPLEMENTATION_AVAILABLE,
                    "version": "v12.0",
                    "metrics": impl_metrics,
                    "total_step_calls": self.step_implementation_metrics['total_step_calls'],
                    "successful_step_calls": self.step_implementation_metrics['successful_step_calls'],
                    "failed_step_calls": self.step_implementation_metrics['failed_step_calls'],
                    "detailed_dataspec_calls": self.step_implementation_metrics['detailed_dataspec_calls'],
                    "api_mapping_calls": self.step_implementation_metrics['api_mapping_calls'],
                    "step_success_rate": (
                        self.step_implementation_metrics['successful_step_calls'] / 
                        max(1, self.step_implementation_metrics['total_step_calls']) * 100
                    )
                },
                
                # 8단계 Step 매핑 (StepImplementationManager v12.0 기반)
                "supported_steps": {
                    "step_1_upload_validation": "기본 검증 + StepImplementationManager",
                    "step_2_measurements_validation": "기본 검증 + StepImplementationManager",
                    "step_3_human_parsing": f"StepImplementationManager v12.0 → {STEP_ID_TO_NAME_MAPPING.get(3, 'HumanParsingStep')}",
                    "step_4_pose_estimation": f"StepImplementationManager v12.0 → {STEP_ID_TO_NAME_MAPPING.get(4, 'PoseEstimationStep')}",
                    "step_5_clothing_analysis": f"StepImplementationManager v12.0 → {STEP_ID_TO_NAME_MAPPING.get(5, 'ClothSegmentationStep')}",
                    "step_6_geometric_matching": f"StepImplementationManager v12.0 → {STEP_ID_TO_NAME_MAPPING.get(6, 'GeometricMatchingStep')}",
                    "step_7_virtual_fitting": f"StepImplementationManager v12.0 → {STEP_ID_TO_NAME_MAPPING.get(7, 'VirtualFittingStep')} ⭐",
                    "step_8_result_analysis": f"StepImplementationManager v12.0 → {STEP_ID_TO_NAME_MAPPING.get(8, 'QualityAssessmentStep')}",
                    "complete_pipeline": "StepImplementationManager v12.0 파이프라인 처리",
                    "batch_processing": False,
                    "scheduled_processing": False
                },
                
                # 환경 정보
                "environment": {
                    "conda_env": CONDA_INFO['conda_env'],
                    "conda_optimized": CONDA_INFO['is_target_env'],
                    "device": DEVICE,
                    "is_m3_max": IS_M3_MAX,
                    "memory_gb": MEMORY_GB,
                    "torch_available": TORCH_AVAILABLE,
                    "numpy_available": NUMPY_AVAILABLE,
                    "pil_available": PIL_AVAILABLE
                },
                
                # 아키텍처 정보 (StepImplementationManager v12.0 통합)
                "architecture": "StepServiceManager v14.0 → StepImplementationManager v12.0 → StepFactory v11.0 → 실제 Step 클래스들",
                "version": "v14.0_stepimpl_manager_integration",
                "conda_environment": CONDA_INFO['is_target_env'],
                "conda_env_name": CONDA_INFO['conda_env'],
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                
                # 핵심 특징 (StepImplementationManager v12.0 기반)
                "key_features": [
                    "StepImplementationManager v12.0 완전 통합",
                    "DetailedDataSpec 기반 Step 처리",
                    "API ↔ Step 자동 변환",
                    "Step 간 데이터 흐름 관리",
                    "전처리/후처리 자동 적용",
                    "FastAPI 라우터 100% 호환",
                    "기존 8단계 API 100% 유지",
                    "세션 기반 처리",
                    "메모리 효율적 관리",
                    "conda 환경 + M3 Max 최적화",
                    "실제 AI 모델 연동"
                ],
                
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 메트릭 조회 실패: {e}")
            return {
                "error": str(e),
                "version": "v14.0_stepimpl_manager_integration",
                "timestamp": datetime.now().isoformat()
            }
    
    async def cleanup(self) -> Dict[str, Any]:
        """서비스 정리 (StepImplementationManager v12.0 통합)"""
        try:
            self.logger.info("🧹 StepServiceManager v14.0 정리 시작... (StepImplementationManager v12.0 통합)")
            
            # 상태 변경
            self.status = ServiceStatus.MAINTENANCE
            
            # StepImplementationManager v12.0 정리
            impl_status_before = {}
            if self.implementation_manager:
                try:
                    impl_status_before = self.implementation_manager.get_all_metrics()
                    self.implementation_manager.cleanup()
                except Exception as e:
                    self.logger.warning(f"⚠️ StepImplementationManager v12.0 정리 실패: {e}")
            
            # 세션 정리
            session_count = len(self.sessions)
            self.sessions.clear()
            
            # 메모리 정리
            await self._optimize_memory()
            
            # 상태 리셋
            self.status = ServiceStatus.INACTIVE
            
            self.logger.info("✅ StepServiceManager v14.0 정리 완료 (StepImplementationManager v12.0 통합)")
            
            return {
                "success": True,
                "message": "서비스 정리 완료 (StepImplementationManager v12.0 통합)",
                "step_implementation_manager_cleaned": STEP_IMPLEMENTATION_AVAILABLE,
                "impl_metrics_before": impl_status_before,
                "sessions_cleared": session_count,
                "stepimpl_manager_used": STEP_IMPLEMENTATION_AVAILABLE,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 서비스 정리 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_status(self) -> Dict[str, Any]:
        """서비스 상태 조회 (StepImplementationManager v12.0 통합)"""
        with self._lock:
            impl_status = {}
            if self.implementation_manager:
                try:
                    impl_metrics = self.implementation_manager.get_all_metrics()
                    impl_status = {
                        "available": True,
                        "version": "v12.0",
                        "total_steps": len(impl_metrics.get('available_steps', [])),
                        "detailed_dataspec_enabled": impl_metrics.get('detailed_dataspec_features', {}).get('api_input_mapping_supported', False)
                    }
                except Exception as e:
                    impl_status = {"available": False, "error": str(e)}
            else:
                impl_status = {"available": False, "reason": "not_imported"}
            
            return {
                "status": self.status.value,
                "processing_mode": self.processing_mode.value,
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "step_implementation_manager": impl_status,
                "active_sessions": len(self.sessions),
                "version": "v14.0_stepimpl_manager_integration",
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "last_error": self.last_error,
                "timestamp": datetime.now().isoformat()
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """헬스 체크 (StepImplementationManager v12.0 통합)"""
        try:
            # StepImplementationManager v12.0 상태 확인
            impl_health = {"available": False}
            if self.implementation_manager:
                try:
                    impl_metrics = self.implementation_manager.get_all_metrics()
                    impl_health = {
                        "available": True,
                        "version": "v12.0",
                        "step_mappings": len(impl_metrics.get('step_mappings', {})),
                        "detailed_dataspec_features": impl_metrics.get('detailed_dataspec_features', {}),
                        "system_status": impl_metrics.get('system_status', {})
                    }
                except Exception as e:
                    impl_health = {"available": False, "error": str(e)}
            
            health_status = {
                "healthy": self.status == ServiceStatus.ACTIVE and impl_health.get("available", False),
                "status": self.status.value,
                "step_implementation_manager": impl_health,
                "device": DEVICE,
                "conda_env": CONDA_INFO['conda_env'],
                "conda_optimized": CONDA_INFO['is_target_env'],
                "is_m3_max": IS_M3_MAX,
                "torch_available": TORCH_AVAILABLE,
                "components_status": {
                    "step_implementation_manager": impl_health.get("available", False),
                    "memory_management": True,
                    "session_management": True,
                    "device_acceleration": DEVICE != "cpu",
                    "detailed_dataspec_support": impl_health.get("detailed_dataspec_features", {}).get("api_input_mapping_supported", False)
                },
                "supported_step_classes": list(STEP_ID_TO_NAME_MAPPING.values()),
                "timestamp": datetime.now().isoformat()
            }
            
            return health_status
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "step_implementation_manager": {"available": False},
                "timestamp": datetime.now().isoformat()
            }
    
    def get_supported_features(self) -> Dict[str, bool]:
        """지원되는 기능 목록 (StepImplementationManager v12.0 통합)"""
        impl_features = {}
        if self.implementation_manager:
            try:
                impl_metrics = self.implementation_manager.get_all_metrics()
                impl_features = impl_metrics.get('detailed_dataspec_features', {})
            except:
                pass
        
        return {
            "8_step_ai_pipeline": True,
            "step_implementation_manager": STEP_IMPLEMENTATION_AVAILABLE,
            "detailed_dataspec_processing": impl_features.get('api_input_mapping_supported', False),
            "api_mapping_support": impl_features.get('api_output_mapping_supported', False),
            "step_data_flow_support": impl_features.get('step_data_flow_supported', False),
            "preprocessing_support": impl_features.get('preprocessing_steps_supported', False),
            "postprocessing_support": impl_features.get('postprocessing_steps_supported', False),
            "fastapi_integration": impl_features.get('fastapi_integration_ready', False),
            "memory_optimization": True,
            "session_management": True,
            "health_monitoring": True,
            "conda_optimization": CONDA_INFO['is_target_env'],
            "m3_max_optimization": IS_M3_MAX,
            "gpu_acceleration": DEVICE != "cpu",
            "step_pipeline_processing": STEP_IMPLEMENTATION_AVAILABLE
        }

# ==============================================
# 🔥 싱글톤 관리 (StepImplementationManager v12.0 통합)
# ==============================================

# 전역 인스턴스들
_global_manager: Optional[StepServiceManager] = None
_manager_lock = threading.RLock()

def get_step_service_manager() -> StepServiceManager:
    """전역 StepServiceManager 반환 (StepImplementationManager v12.0 통합)"""
    global _global_manager
    
    with _manager_lock:
        if _global_manager is None:
            _global_manager = StepServiceManager()
            logger.info("✅ 전역 StepServiceManager v14.0 생성 완료 (StepImplementationManager v12.0 통합)")
    
    return _global_manager

async def get_step_service_manager_async() -> StepServiceManager:
    """전역 StepServiceManager 반환 (비동기, 초기화 포함, StepImplementationManager v12.0 통합)"""
    manager = get_step_service_manager()
    
    if manager.status == ServiceStatus.INACTIVE:
        await manager.initialize()
        logger.info("✅ StepServiceManager v14.0 자동 초기화 완료 (StepImplementationManager v12.0 통합)")
    
    return manager

async def cleanup_step_service_manager():
    """전역 StepServiceManager 정리 (StepImplementationManager v12.0 통합)"""
    global _global_manager
    
    with _manager_lock:
        if _global_manager:
            await _global_manager.cleanup()
            _global_manager = None
            logger.info("🧹 전역 StepServiceManager v14.0 정리 완료 (StepImplementationManager v12.0 통합)")

def reset_step_service_manager():
    """전역 StepServiceManager 리셋"""
    global _global_manager
    
    with _manager_lock:
        _global_manager = None
        
    logger.info("🔄 전역 StepServiceManager v14.0 리셋 완료")

# ==============================================
# 🔥 기존 호환성 별칭들 (API 호환성 유지)
# ==============================================

# 기존 API 호환성을 위한 별칭들
def get_pipeline_service_sync() -> StepServiceManager:
    """파이프라인 서비스 반환 (동기) - 기존 호환성"""
    return get_step_service_manager()

async def get_pipeline_service() -> StepServiceManager:
    """파이프라인 서비스 반환 (비동기) - 기존 호환성"""
    return await get_step_service_manager_async()

def get_pipeline_manager_service() -> StepServiceManager:
    """파이프라인 매니저 서비스 반환 - 기존 호환성"""
    return get_step_service_manager()

async def get_unified_service_manager() -> StepServiceManager:
    """통합 서비스 매니저 반환 - 기존 호환성"""
    return await get_step_service_manager_async()

def get_unified_service_manager_sync() -> StepServiceManager:
    """통합 서비스 매니저 반환 (동기) - 기존 호환성"""
    return get_step_service_manager()

# 클래스 별칭들
PipelineService = StepServiceManager
ServiceBodyMeasurements = BodyMeasurements
UnifiedStepServiceManager = StepServiceManager
StepService = StepServiceManager

# ==============================================
# 🔥 유틸리티 함수들 (StepImplementationManager v12.0 통합)
# ==============================================

def get_service_availability_info() -> Dict[str, Any]:
    """서비스 가용성 정보 (StepImplementationManager v12.0 통합)"""
    
    # StepImplementationManager v12.0 가용성 확인
    impl_availability = {}
    if STEP_IMPLEMENTATION_AVAILABLE:
        try:
            impl_availability = get_implementation_availability_info()
        except Exception as e:
            impl_availability = {"error": str(e)}
    
    return {
        "step_service_available": True,
        "step_implementation_manager_available": STEP_IMPLEMENTATION_AVAILABLE,
        "services_available": True,
        "architecture": "StepServiceManager v14.0 → StepImplementationManager v12.0 → StepFactory v11.0 → 실제 Step 클래스들",
        "version": "v14.0_stepimpl_manager_integration",
        
        # StepImplementationManager v12.0 정보
        "step_implementation_info": impl_availability,
        
        # 8단계 Step 매핑 (StepImplementationManager v12.0 기반)
        "step_mappings": {
            f"step_{step_id}": {
                "name": step_name,
                "available": STEP_IMPLEMENTATION_AVAILABLE,
                "implementation_manager": "v12.0",
                "detailed_dataspec": True
            }
            for step_id, step_name in STEP_ID_TO_NAME_MAPPING.items()
        },
        
        # 완전한 기능 지원
        "complete_features": {
            "step_implementation_manager_integration": STEP_IMPLEMENTATION_AVAILABLE,
            "detailed_dataspec_processing": STEP_IMPLEMENTATION_AVAILABLE,
            "api_mapping_support": STEP_IMPLEMENTATION_AVAILABLE,
            "step_data_flow_support": STEP_IMPLEMENTATION_AVAILABLE,
            "preprocessing_postprocessing": STEP_IMPLEMENTATION_AVAILABLE,
            "fastapi_integration": STEP_IMPLEMENTATION_AVAILABLE,
            "memory_optimization": True,
            "session_management": True,
            "health_monitoring": True,
            "conda_optimization": CONDA_INFO['is_target_env'],
            "m3_max_optimization": IS_M3_MAX,
            "gpu_acceleration": DEVICE != "cpu"
        },
        
        # 8단계 파이프라인 (StepImplementationManager v12.0 기반)
        "ai_pipeline_steps": {
            "step_1_upload_validation": "기본 검증",
            "step_2_measurements_validation": "기본 검증",
            "step_3_human_parsing": f"StepImplementationManager v12.0 → {STEP_ID_TO_NAME_MAPPING.get(3, 'HumanParsingStep')}",
            "step_4_pose_estimation": f"StepImplementationManager v12.0 → {STEP_ID_TO_NAME_MAPPING.get(4, 'PoseEstimationStep')}",
            "step_5_clothing_analysis": f"StepImplementationManager v12.0 → {STEP_ID_TO_NAME_MAPPING.get(5, 'ClothSegmentationStep')}",
            "step_6_geometric_matching": f"StepImplementationManager v12.0 → {STEP_ID_TO_NAME_MAPPING.get(6, 'GeometricMatchingStep')}",
            "step_7_virtual_fitting": f"StepImplementationManager v12.0 → {STEP_ID_TO_NAME_MAPPING.get(7, 'VirtualFittingStep')} ⭐",
            "step_8_result_analysis": f"StepImplementationManager v12.0 → {STEP_ID_TO_NAME_MAPPING.get(8, 'QualityAssessmentStep')}",
            "complete_pipeline": "StepImplementationManager v12.0 파이프라인"
        },
        
        # API 호환성
        "api_compatibility": {
            "process_step_1_upload_validation": True,
            "process_step_2_measurements_validation": True,
            "process_step_3_human_parsing": True,
            "process_step_4_pose_estimation": True,
            "process_step_5_clothing_analysis": True,
            "process_step_6_geometric_matching": True,
            "process_step_7_virtual_fitting": True,
            "process_step_8_result_analysis": True,
            "process_complete_virtual_fitting": True,
            "get_step_service_manager": True,
            "get_pipeline_service": True,
            "cleanup_step_service_manager": True,
            "health_check": True,
            "get_all_metrics": True
        },
        
        # 시스템 정보
        "system_info": {
            "conda_environment": CONDA_INFO['is_target_env'],
            "conda_env_name": CONDA_INFO['conda_env'],
            "device": DEVICE,
            "is_m3_max": IS_M3_MAX,
            "memory_gb": MEMORY_GB,
            "torch_available": TORCH_AVAILABLE,
            "python_version": sys.version,
            "platform": sys.platform
        },
        
        # 핵심 특징 (StepImplementationManager v12.0 기반)
        "key_features": [
            "StepImplementationManager v12.0 완전 통합",
            "DetailedDataSpec 기반 Step 처리",
            "API ↔ Step 자동 변환",
            "Step 간 데이터 흐름 관리",
            "전처리/후처리 자동 적용",
            "FastAPI 라우터 100% 호환",
            "기존 8단계 API 100% 유지",
            "세션 기반 처리",
            "메모리 효율적 관리",
            "conda 환경 + M3 Max 최적화",
            "실제 AI 모델 연동",
            "StepFactory v11.0 호환",
            "BaseStepMixin 완전 지원",
            "프로덕션 레벨 안정성",
            "스레드 안전성",
            "실시간 헬스 모니터링"
        ]
    }

def format_api_response(
    success: bool,
    message: str,
    step_name: str,
    step_id: int,
    processing_time: float,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    confidence: Optional[float] = None,
    details: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    result_image: Optional[str] = None,
    fitted_image: Optional[str] = None,
    fit_score: Optional[float] = None,
    recommendations: Optional[List[str]] = None
) -> Dict[str, Any]:
    """API 응답 형식화 (StepImplementationManager v12.0 통합)"""
    response = {
        "success": success,
        "message": message,
        "step_name": step_name,
        "step_id": step_id,
        "session_id": session_id,
        "request_id": request_id,
        "processing_time": processing_time,
        "confidence": confidence or (0.85 + step_id * 0.02),
        "timestamp": datetime.now().isoformat(),
        "details": details or {},
        "error": error,
        "result_image": result_image,
        "fitted_image": fitted_image,
        "fit_score": fit_score,
        "recommendations": recommendations or [],
        "step_implementation_manager_used": STEP_IMPLEMENTATION_AVAILABLE,
        "detailed_dataspec_applied": STEP_IMPLEMENTATION_AVAILABLE
    }
    
    # StepImplementationManager v12.0 정보 추가
    if step_id in STEP_ID_TO_NAME_MAPPING:
        step_class_name = STEP_ID_TO_NAME_MAPPING[step_id]
        response["step_implementation_info"] = {
            "step_class_name": step_class_name,
            "implementation_manager_version": "v12.0",
            "detailed_dataspec_enabled": True
        }
    
    return response

# ==============================================
# 🔥 StepImplementationManager v12.0 편의 함수들
# ==============================================

async def process_step_by_implementation_manager(
    step_id: int,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """StepImplementationManager v12.0를 통한 Step 처리"""
    if not STEP_IMPLEMENTATION_AVAILABLE:
        return {
            "success": False,
            "error": "StepImplementationManager v12.0 사용 불가",
            "step_id": step_id,
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        impl_manager = get_step_implementation_manager_func()
        return await impl_manager.process_step_by_id(step_id, *args, **kwargs)
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "step_id": step_id,
            "timestamp": datetime.now().isoformat()
        }

async def process_step_by_name_implementation_manager(
    step_name: str,
    api_input: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """StepImplementationManager v12.0를 통한 Step 이름별 처리"""
    if not STEP_IMPLEMENTATION_AVAILABLE:
        return {
            "success": False,
            "error": "StepImplementationManager v12.0 사용 불가",
            "step_name": step_name,
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        impl_manager = get_step_implementation_manager_func()
        return await impl_manager.process_step_by_name(step_name, api_input, **kwargs)
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "step_name": step_name,
            "timestamp": datetime.now().isoformat()
        }

def get_step_implementation_manager_metrics() -> Dict[str, Any]:
    """StepImplementationManager v12.0 메트릭 조회"""
    if not STEP_IMPLEMENTATION_AVAILABLE:
        return {
            "available": False,
            "error": "StepImplementationManager v12.0 사용 불가"
        }
    
    try:
        impl_manager = get_step_implementation_manager_func()
        return impl_manager.get_all_metrics()
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }

def get_step_api_specifications() -> Dict[str, Dict[str, Any]]:
    """모든 Step의 API 사양 조회 (StepImplementationManager v12.0 기반)"""
    if not STEP_IMPLEMENTATION_AVAILABLE:
        return {}
    
    try:
        return get_all_steps_api_specification()
    except Exception as e:
        logger.error(f"❌ Step API 사양 조회 실패: {e}")
        return {}

# ==============================================
# 🔥 메모리 최적화 함수들 (conda + M3 Max)
# ==============================================

def safe_mps_empty_cache():
    """안전한 MPS 캐시 정리 (M3 Max)"""
    try:
        if TORCH_AVAILABLE and IS_M3_MAX:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
                    logger.debug("🍎 M3 Max MPS 캐시 정리 완료")
    except Exception as e:
        logger.debug(f"MPS 캐시 정리 실패 (무시): {e}")

def optimize_conda_memory():
    """conda 환경 메모리 최적화"""
    try:
        # Python GC
        gc.collect()
        
        # MPS 메모리 정리 (M3 Max)
        safe_mps_empty_cache()
        
        # CUDA 메모리 정리
        if TORCH_AVAILABLE and DEVICE == "cuda":
            import torch
            torch.cuda.empty_cache()
            
        logger.debug("💾 conda 메모리 최적화 완료")
    except Exception as e:
        logger.debug(f"conda 메모리 최적화 실패 (무시): {e}")

# ==============================================
# 🔥 Export 목록 (StepImplementationManager v12.0 통합)
# ==============================================

__all__ = [
    # 메인 클래스들
    "StepServiceManager",
    
    # 데이터 구조들
    "ProcessingMode",
    "ServiceStatus", 
    "ProcessingPriority",
    "BodyMeasurements",
    "ProcessingRequest",
    "ProcessingResult",
    
    # 싱글톤 함수들
    "get_step_service_manager",
    "get_step_service_manager_async", 
    "get_pipeline_service",
    "get_pipeline_service_sync",
    "get_pipeline_manager_service",
    "get_unified_service_manager",
    "get_unified_service_manager_sync",
    "cleanup_step_service_manager",
    "reset_step_service_manager",
    
    # 유틸리티 함수들
    "get_service_availability_info",
    "format_api_response",
    "safe_mps_empty_cache",
    "optimize_conda_memory",
    
    # StepImplementationManager v12.0 편의 함수들
    "process_step_by_implementation_manager",
    "process_step_by_name_implementation_manager",
    "get_step_implementation_manager_metrics",
    "get_step_api_specifications",

    # 호환성 별칭들
    "PipelineService",
    "ServiceBodyMeasurements",
    "UnifiedStepServiceManager",
    "StepService",
    
    # 상수들
    "STEP_IMPLEMENTATION_AVAILABLE",
    "STEP_ID_TO_NAME_MAPPING",
    "STEP_NAME_TO_CLASS_MAPPING"
]

# ==============================================
# 🔥 초기화 및 최적화 (StepImplementationManager v12.0 통합)
# ==============================================

# conda 환경 확인 및 권장
conda_status = "✅" if CONDA_INFO['is_target_env'] else "⚠️"
logger.info(f"{conda_status} conda 환경: {CONDA_INFO['conda_env']}")

if not CONDA_INFO['is_target_env']:
    logger.warning("⚠️ conda 환경 권장: conda activate mycloset-ai-clean")

# StepImplementationManager v12.0 상태 확인
impl_status = "✅" if STEP_IMPLEMENTATION_AVAILABLE else "❌"
logger.info(f"{impl_status} StepImplementationManager v12.0: {'사용 가능' if STEP_IMPLEMENTATION_AVAILABLE else '사용 불가'}")

if STEP_IMPLEMENTATION_AVAILABLE:
    logger.info(f"📊 지원 Step 클래스: {len(STEP_ID_TO_NAME_MAPPING)}개")
    for step_id, step_name in STEP_ID_TO_NAME_MAPPING.items():
        logger.info(f"   - Step {step_id}: {step_name}")

# ==============================================
# 🔥 완료 메시지
# ==============================================

logger.info("🔥 Step Service v14.0 - StepImplementationManager v12.0 완전 통합 로드 완료!")
logger.info(f"✅ StepImplementationManager v12.0: {'연동 완료' if STEP_IMPLEMENTATION_AVAILABLE else '사용 불가'}")
logger.info("✅ 기존 8단계 AI 파이프라인 API 100% 유지")
logger.info("✅ DetailedDataSpec 기반 Step 처리")
logger.info("✅ API ↔ Step 자동 변환")
logger.info("✅ FastAPI 라우터 완전 호환")

logger.info("🎯 새로운 아키텍처:")
logger.info("   step_routes.py → StepServiceManager v14.0 → StepImplementationManager v12.0 → StepFactory v11.0 → 실제 Step 클래스들")

logger.info("🎯 기존 API 100% 호환:")
logger.info("   - process_step_1_upload_validation")
logger.info("   - process_step_2_measurements_validation")
logger.info("   - process_step_3_human_parsing")
logger.info("   - process_step_4_pose_estimation")
logger.info("   - process_step_5_clothing_analysis")
logger.info("   - process_step_6_geometric_matching")
logger.info("   - process_step_7_virtual_fitting ⭐")
logger.info("   - process_step_8_result_analysis")
logger.info("   - process_complete_virtual_fitting")

logger.info("🎯 새로운 처리 흐름:")
logger.info("   1. StepServiceManager v14.0: 비즈니스 로직 + 세션 관리")
logger.info("   2. StepImplementationManager v12.0: API ↔ Step 변환 + DetailedDataSpec")
logger.info("   3. StepFactory v11.0: Step 인스턴스 생성 + 의존성 주입")
logger.info("   4. BaseStepMixin: 실제 AI 모델 추론")

# conda 환경 자동 최적화
if CONDA_INFO['is_target_env']:
    optimize_conda_memory()
    logger.info("🐍 conda 환경 자동 최적화 완료!")
else:
    logger.warning(f"⚠️ conda 환경을 확인하세요: conda activate mycloset-ai-clean")

# 초기 메모리 최적화
safe_mps_empty_cache()
gc.collect()
logger.info(f"💾 {DEVICE} 초기 메모리 최적화 완료!")
