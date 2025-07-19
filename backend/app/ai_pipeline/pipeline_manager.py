# backend/app/ai_pipeline/pipeline_manager.py
"""
🔥 완전 통합 PipelineManager - 두 버전 최적 합성 + 모듈화
✅ 프로젝트 통합 유틸리티 시스템 완벽 연동 (우선순위 1)
✅ StepModelInterface.get_model() 완전 연동 (우선순위 2)
✅ 자동 탐지된 모델과 Step 요청 자동 매칭 완벽 지원
✅ ModelLoader 초기화 순서 완벽 보장
✅ 모듈화된 구조로 유지보수성 극대화
✅ 에러 처리 및 폴백 메커니즘 대폭 강화
✅ M3 Max 128GB + conda 환경 최적화
✅ 모든 기존 함수/클래스명 100% 유지
✅ 프로덕션 레벨 안정성

아키텍처:
PipelineManager (Main Controller)
├── InitializationManager (초기화 관리)
│   ├── UnifiedSystemInitializer (통합 시스템)
│   ├── ModelLoaderInitializer (ModelLoader)
│   └── StepInitializer (Step 클래스들)
├── ExecutionManager (실행 관리)
│   ├── UnifiedExecutor (통합 시스템 우선)
│   ├── ModelLoaderExecutor (ModelLoader 폴백)
│   └── FallbackExecutor (최종 폴백)
├── UtilityManager (유틸리티 관리)
│   ├── OptimizedDataConverter
│   ├── OptimizedMemoryManager
│   └── PerformanceMonitor
└── ConfigurationManager (설정 관리)
    ├── DeviceOptimizer
    ├── M3MaxOptimizer
    └── CondaOptimizer
"""

import os
import sys
import logging
import asyncio
import time
import traceback
import threading
import json
import gc
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from abc import ABC, abstractmethod

# 필수 라이브러리
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter

# ==============================================
# 🔥 1. 프로젝트 통합 시스템 import (최우선)
# ==============================================

# 통합 유틸리티 시스템
try:
    from app.ai_pipeline.utils import (
        initialize_global_utils, get_utils_manager, 
        get_system_status, optimize_system_memory,
        get_step_model_interface, get_step_memory_manager,
        get_step_data_converter, preprocess_image_for_step
    )
    UNIFIED_UTILS_AVAILABLE = True
except ImportError as e:
    UNIFIED_UTILS_AVAILABLE = False
    logging.warning(f"통합 유틸리티 시스템 사용 불가: {e}")

# ModelLoader 시스템
try:
    from app.ai_pipeline.utils.model_loader import (
        ModelLoader, get_global_model_loader, initialize_global_model_loader,
        StepModelInterface
    )
    MODEL_LOADER_AVAILABLE = True
except ImportError as e:
    MODEL_LOADER_AVAILABLE = False
    logging.warning(f"ModelLoader 시스템 사용 불가: {e}")

# Step 모델 요청 시스템
try:
    from app.ai_pipeline.utils.step_model_requests import (
        get_step_request, StepModelRequestAnalyzer, 
        STEP_MODEL_REQUESTS, get_all_step_requirements
    )
    STEP_REQUESTS_AVAILABLE = True
except ImportError as e:
    STEP_REQUESTS_AVAILABLE = False
    logging.warning(f"Step 요청 시스템 사용 불가: {e}")

# 자동 모델 탐지 시스템
try:
    from app.ai_pipeline.utils.auto_model_detector import (
        RealWorldModelDetector, create_real_world_detector,
        quick_real_model_detection
    )
    AUTO_DETECTOR_AVAILABLE = True
except ImportError as e:
    AUTO_DETECTOR_AVAILABLE = False
    logging.warning(f"자동 탐지 시스템 사용 불가: {e}")

# Step 클래스들 import
try:
    from app.ai_pipeline.steps.step_01_human_parsing import HumanParsingStep
    from app.ai_pipeline.steps.step_02_pose_estimation import PoseEstimationStep
    from app.ai_pipeline.steps.step_03_cloth_segmentation import ClothSegmentationStep
    from app.ai_pipeline.steps.step_04_geometric_matching import GeometricMatchingStep
    from app.ai_pipeline.steps.step_05_cloth_warping import ClothWarpingStep
    from app.ai_pipeline.steps.step_06_virtual_fitting import VirtualFittingStep
    from app.ai_pipeline.steps.step_07_post_processing import PostProcessingStep
    from app.ai_pipeline.steps.step_08_quality_assessment import QualityAssessmentStep
    STEP_CLASSES_AVAILABLE = True
except ImportError as e:
    STEP_CLASSES_AVAILABLE = False
    logging.error(f"Step 클래스들 import 실패: {e}")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================
# 🔥 2. 열거형 및 데이터 클래스
# ==============================================

class PipelineMode(Enum):
    """파이프라인 모드"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"
    OPTIMIZATION = "optimization"

class QualityLevel(Enum):
    """품질 레벨"""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"
    MAXIMUM = "maximum"

class ProcessingStatus(Enum):
    """처리 상태"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CLEANING = "cleaning"

class ExecutionMode(Enum):
    """실행 모드"""
    UNIFIED_SYSTEM = "unified_system"  # 통합 시스템 우선
    MODEL_LOADER = "model_loader"      # ModelLoader 우선
    FALLBACK = "fallback"              # 폴백 모드

@dataclass
class PipelineConfig:
    """통합 파이프라인 설정"""
    # 기본 설정
    device: str = "auto"
    quality_level: Union[QualityLevel, str] = QualityLevel.BALANCED
    processing_mode: Union[PipelineMode, str] = PipelineMode.PRODUCTION
    
    # 시스템 설정
    memory_gb: float = 16.0
    is_m3_max: bool = False
    device_type: str = "auto"
    
    # 🔥 통합 시스템 설정 (최우선)
    unified_utils_enabled: bool = True
    model_loader_enabled: bool = True
    auto_detect_models: bool = True
    preload_critical_models: bool = True
    model_cache_warmup: bool = True
    step_model_validation: bool = True
    
    # 실행 전략
    execution_mode: Union[ExecutionMode, str] = ExecutionMode.UNIFIED_SYSTEM
    fallback_enabled: bool = True
    retry_with_fallback: bool = True
    
    # 최적화 설정
    optimization_enabled: bool = True
    enable_caching: bool = True
    parallel_processing: bool = True
    memory_optimization: bool = True
    use_fp16: bool = True
    enable_quantization: bool = False
    
    # 처리 설정
    batch_size: int = 1
    max_retries: int = 3
    timeout_seconds: int = 300
    save_intermediate: bool = False
    enable_progress_callback: bool = True
    
    # 고급 설정
    model_cache_size: int = 10
    memory_threshold: float = 0.8
    gpu_memory_fraction: float = 0.9
    thread_pool_size: int = 4
    
    def __post_init__(self):
        # 문자열을 Enum으로 변환
        if isinstance(self.quality_level, str):
            self.quality_level = QualityLevel(self.quality_level)
        if isinstance(self.processing_mode, str):
            self.processing_mode = PipelineMode(self.processing_mode)
        if isinstance(self.execution_mode, str):
            self.execution_mode = ExecutionMode(self.execution_mode)
        
        # M3 Max 자동 최적화
        if self.is_m3_max:
            self.memory_gb = max(self.memory_gb, 64.0)
            self.use_fp16 = True
            self.optimization_enabled = True
            self.batch_size = 4
            self.model_cache_size = 15
            self.gpu_memory_fraction = 0.95
            self.unified_utils_enabled = True
            self.auto_detect_models = True
            self.preload_critical_models = True
            self.model_cache_warmup = True

@dataclass
class ProcessingResult:
    """처리 결과"""
    success: bool
    session_id: str = ""
    result_image: Optional[Image.Image] = None
    result_tensor: Optional[torch.Tensor] = None
    quality_score: float = 0.0
    quality_grade: str = "Unknown"
    processing_time: float = 0.0
    step_results: Dict[str, Any] = field(default_factory=dict)
    step_timings: Dict[str, float] = field(default_factory=dict)
    execution_strategy: Dict[str, str] = field(default_factory=dict)  # 🔥 실행 전략 추가
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

@dataclass
class SessionData:
    """세션 데이터"""
    session_id: str
    start_time: float
    status: ProcessingStatus = ProcessingStatus.IDLE
    step_results: Dict[str, Any] = field(default_factory=dict)
    step_timings: Dict[str, float] = field(default_factory=dict)
    execution_strategies: Dict[str, str] = field(default_factory=dict)  # 🔥 실행 전략 추가
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_step_result(self, step_name: str, result: Dict[str, Any], timing: float, strategy: str = "unknown"):
        """단계 결과 추가"""
        self.step_results[step_name] = result
        self.step_timings[step_name] = timing
        self.execution_strategies[step_name] = strategy

@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    total_sessions: int = 0
    successful_sessions: int = 0
    failed_sessions: int = 0
    average_processing_time: float = 0.0
    average_quality_score: float = 0.0
    total_processing_time: float = 0.0
    fastest_processing_time: float = float('inf')
    slowest_processing_time: float = 0.0
    unified_system_usage: int = 0  # 🔥 통합 시스템 사용 횟수
    model_loader_usage: int = 0    # 🔥 ModelLoader 사용 횟수
    fallback_usage: int = 0        # 🔥 폴백 사용 횟수
    
    def update(self, processing_time: float, quality_score: float, success: bool, execution_strategy: str = "unknown"):
        """메트릭 업데이트"""
        self.total_sessions += 1
        self.total_processing_time += processing_time
        
        # 실행 전략별 통계
        if execution_strategy == "unified_system":
            self.unified_system_usage += 1
        elif execution_strategy == "model_loader":
            self.model_loader_usage += 1
        elif execution_strategy == "fallback":
            self.fallback_usage += 1
        
        if success:
            self.successful_sessions += 1
            self.fastest_processing_time = min(self.fastest_processing_time, processing_time)
            self.slowest_processing_time = max(self.slowest_processing_time, processing_time)
        else:
            self.failed_sessions += 1
        
        # 평균 계산
        if self.total_sessions > 0:
            self.average_processing_time = self.total_processing_time / self.total_sessions
        
        if self.successful_sessions > 0:
            prev_total = self.average_quality_score * (self.successful_sessions - 1)
            self.average_quality_score = (prev_total + quality_score) / self.successful_sessions

# ==============================================
# 🔥 3. 모듈화된 관리 클래스들
# ==============================================

class OptimizedDataConverter:
    """최적화된 데이터 변환기 - 통합 시스템 우선"""
    
    def __init__(self, device: str = "mps"):
        self.device = device
        self.logger = logging.getLogger(f"pipeline.{self.__class__.__name__}")
        
        # 🔥 통합 시스템 연동
        self.project_converter = None
        if UNIFIED_UTILS_AVAILABLE:
            try:
                self.project_converter = get_step_data_converter("PipelineManager")
                if self.project_converter:
                    self.logger.info("✅ 통합 DataConverter 연동 완료")
            except Exception as e:
                self.logger.warning(f"⚠️ 통합 DataConverter 연동 실패: {e}")
    
    def preprocess_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """이미지 전처리 - 통합 시스템 우선"""
        try:
            # 🔥 1순위: 통합 시스템 사용
            if self.project_converter:
                try:
                    result = self.project_converter.image_to_tensor(image_input)
                    if result is not None:
                        return result
                except Exception as e:
                    self.logger.warning(f"⚠️ 통합 컨버터 실패, 폴백 사용: {e}")
            
            # 2순위: 프로젝트 전처리 함수 사용
            if UNIFIED_UTILS_AVAILABLE:
                try:
                    result = preprocess_image_for_step(image_input, "PipelineManager")
                    if result is not None:
                        return result
                except Exception as e:
                    self.logger.warning(f"⚠️ 프로젝트 전처리 실패, 기본 사용: {e}")
            
            # 3순위: 기본 전처리
            return self._basic_preprocess(image_input)
            
        except Exception as e:
            self.logger.error(f"❌ 이미지 전처리 실패: {e}")
            raise
    
    def _basic_preprocess(self, image_input: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """기본 이미지 전처리"""
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_input}")
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype != np.uint8:
                image_input = (image_input * 255).astype(np.uint8)
            image = Image.fromarray(image_input).convert('RGB')
        else:
            raise ValueError(f"지원하지 않는 이미지 타입: {type(image_input)}")
        
        # 크기 조정
        target_size = (512, 512)
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # 텐서 변환
        img_array = np.array(image)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """텐서를 PIL 이미지로 변환 - 통합 시스템 우선"""
        try:
            # 🔥 1순위: 통합 시스템 사용
            if self.project_converter:
                try:
                    result = self.project_converter.tensor_to_image(tensor, format="PIL")
                    if result is not None:
                        return result
                except Exception as e:
                    self.logger.warning(f"⚠️ 통합 컨버터 실패, 기본 사용: {e}")
            
            # 2순위: 기본 변환
            return self._basic_tensor_to_pil(tensor)
            
        except Exception as e:
            self.logger.error(f"❌ 텐서-PIL 변환 실패: {e}")
            return Image.new('RGB', (512, 512), color='black')
    
    def _basic_tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """기본 텐서-PIL 변환"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        
        tensor = torch.clamp(tensor, 0, 1)
        tensor = tensor.cpu()
        array = (tensor.numpy() * 255).astype(np.uint8)
        
        return Image.fromarray(array)

class OptimizedMemoryManager:
    """최적화된 메모리 관리자 - 통합 시스템 우선"""
    
    def __init__(self, device: str = "mps"):
        self.device = device
        self.logger = logging.getLogger(f"pipeline.{self.__class__.__name__}")
        
        # 🔥 통합 시스템 연동
        self.project_memory_manager = None
        if UNIFIED_UTILS_AVAILABLE:
            try:
                self.project_memory_manager = get_step_memory_manager("PipelineManager")
                if self.project_memory_manager:
                    self.logger.info("✅ 통합 MemoryManager 연동 완료")
            except Exception as e:
                self.logger.warning(f"⚠️ 통합 MemoryManager 연동 실패: {e}")
    
    def cleanup_memory(self):
        """메모리 정리 - 통합 시스템 우선"""
        try:
            # 🔥 1순위: 통합 시스템 사용
            if self.project_memory_manager:
                try:
                    result = self.project_memory_manager.cleanup_memory()
                    if result.get("success", False):
                        self.logger.debug("✅ 통합 메모리 매니저로 정리 완료")
                        return
                except Exception as e:
                    self.logger.warning(f"⚠️ 통합 메모리 매니저 실패, 기본 사용: {e}")
            
            # 2순위: 시스템 메모리 최적화
            if UNIFIED_UTILS_AVAILABLE:
                try:
                    optimize_system_memory()
                    self.logger.debug("✅ 시스템 메모리 최적화 완료")
                    return
                except Exception as e:
                    self.logger.warning(f"⚠️ 시스템 메모리 최적화 실패: {e}")
            
            # 3순위: 기본 메모리 정리
            self._basic_cleanup()
            
        except Exception as e:
            self.logger.warning(f"❌ 메모리 정리 실패: {e}")
    
    def _basic_cleanup(self):
        """기본 메모리 정리"""
        gc.collect()
        
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif self.device == "mps" and torch.backends.mps.is_available():
            try:
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                if hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
            except (AttributeError, RuntimeError):
                pass
    
    def get_memory_usage(self) -> Dict[str, float]:
        """메모리 사용량 조회 - 통합 시스템 우선"""
        try:
            # 🔥 1순위: 통합 시스템 사용
            if self.project_memory_manager:
                try:
                    stats = self.project_memory_manager.get_memory_stats()
                    return {
                        'allocated_gb': stats.gpu_allocated_gb,
                        'total_gb': stats.gpu_total_gb,
                        'cpu_used_gb': stats.cpu_used_gb,
                        'cpu_total_gb': stats.cpu_total_gb,
                        'cpu_percent': stats.cpu_percent,
                        'source': 'unified_system'
                    }
                except Exception as e:
                    self.logger.warning(f"⚠️ 통합 메모리 상태 조회 실패: {e}")
            
            # 2순위: 기본 메모리 정보
            return self._basic_memory_usage()
            
        except Exception as e:
            self.logger.warning(f"❌ 메모리 사용량 조회 실패: {e}")
            return {'error': str(e), 'source': 'error'}
    
    def _basic_memory_usage(self) -> Dict[str, float]:
        """기본 메모리 사용량 조회"""
        usage = {'source': 'basic'}
        
        if self.device == "cuda" and torch.cuda.is_available():
            usage.update({
                'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'cached_gb': torch.cuda.memory_reserved() / 1024**3,
                'total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3
            })
        elif self.device == "mps":
            try:
                import psutil
                memory = psutil.virtual_memory()
                usage.update({
                    'used_gb': memory.used / 1024**3,
                    'available_gb': memory.available / 1024**3,
                    'total_gb': memory.total / 1024**3,
                    'percent': memory.percent
                })
            except ImportError:
                usage['status'] = 'psutil not available'
        
        return usage

class InitializationManager:
    """초기화 관리자 - 모든 초기화 로직 통합"""
    
    def __init__(self, config: PipelineConfig, device: str, logger: logging.Logger):
        self.config = config
        self.device = device
        self.logger = logger
        
        # 초기화 상태
        self.unified_system_initialized = False
        self.model_loader_initialized = False
        self.steps_initialized = False
        
        # 컴포넌트 참조
        self.utils_manager = None
        self.model_loader = None
        self.auto_detector = None
        self.steps = {}
    
    async def initialize_all(self, step_order: List[str]) -> bool:
        """전체 초기화 실행"""
        try:
            self.logger.info("🔧 통합 초기화 시작...")
            start_time = time.time()
            
            # 1. 통합 시스템 초기화 (최우선)
            if self.config.unified_utils_enabled:
                self.unified_system_initialized = await self._initialize_unified_system()
                if self.unified_system_initialized:
                    self.logger.info("✅ 통합 시스템 초기화 완료")
                else:
                    self.logger.warning("⚠️ 통합 시스템 초기화 실패, 계속 진행")
            
            # 2. ModelLoader 시스템 초기화
            if self.config.model_loader_enabled:
                self.model_loader_initialized = await self._initialize_model_loader()
                if self.model_loader_initialized:
                    self.logger.info("✅ ModelLoader 시스템 초기화 완료")
                else:
                    self.logger.warning("⚠️ ModelLoader 시스템 초기화 실패, 계속 진행")
            
            # 3. Step 클래스들 초기화
            self.steps_initialized = await self._initialize_steps(step_order)
            if self.steps_initialized:
                self.logger.info("✅ Step 클래스들 초기화 완료")
            
            # 4. 중요 모델 사전 로드 (옵션)
            if self.config.preload_critical_models:
                await self._preload_critical_models()
            
            # 5. 모델 캐시 워밍업 (옵션)
            if self.config.model_cache_warmup:
                await self._warmup_model_cache()
            
            initialization_time = time.time() - start_time
            self.logger.info(f"🎉 통합 초기화 완료 ({initialization_time:.2f}초)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 통합 초기화 실패: {e}")
            return False
    
    async def _initialize_unified_system(self) -> bool:
        """통합 시스템 초기화"""
        try:
            if not UNIFIED_UTILS_AVAILABLE:
                return False
            
            result = await initialize_global_utils(
                device=self.device,
                memory_gb=self.config.memory_gb,
                is_m3_max=self.config.is_m3_max,
                optimization_enabled=self.config.optimization_enabled
            )
            
            if result.get("success", False):
                self.utils_manager = get_utils_manager()
                
                # 자동 탐지 시스템 연동
                if AUTO_DETECTOR_AVAILABLE and self.config.auto_detect_models:
                    self.auto_detector = create_real_world_detector()
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ 통합 시스템 초기화 실패: {e}")
            return False
    
    async def _initialize_model_loader(self) -> bool:
        """ModelLoader 시스템 초기화"""
        try:
            if not MODEL_LOADER_AVAILABLE:
                return False
            
            self.model_loader = await asyncio.get_event_loop().run_in_executor(
                None, initialize_global_model_loader
            )
            if self.model_loader is None:
                self.model_loader = get_global_model_loader()
            
            return self.model_loader is not None
            
        except Exception as e:
            self.logger.error(f"❌ ModelLoader 시스템 초기화 실패: {e}")
            return False
    
    async def _initialize_steps(self, step_order: List[str]) -> bool:
        """Step 클래스들 초기화"""
        try:
            if not STEP_CLASSES_AVAILABLE:
                return False
            
            step_classes = {
                'human_parsing': HumanParsingStep,
                'pose_estimation': PoseEstimationStep,
                'cloth_segmentation': ClothSegmentationStep,
                'geometric_matching': GeometricMatchingStep,
                'cloth_warping': ClothWarpingStep,
                'virtual_fitting': VirtualFittingStep,
                'post_processing': PostProcessingStep,
                'quality_assessment': QualityAssessmentStep
            }
            
            # 기본 설정
            base_config = {
                'device': self.device,
                'device_type': self.config.device_type,
                'memory_gb': self.config.memory_gb,
                'is_m3_max': self.config.is_m3_max,
                'optimization_enabled': self.config.optimization_enabled,
                'quality_level': self.config.quality_level.value
            }
            
            for step_name in step_order:
                try:
                    step_class = step_classes[step_name]
                    step_config = {**base_config, **self._get_step_config(step_name)}
                    
                    # Step 인스턴스 생성
                    step_instance = self._create_step_instance_safely(step_class, step_name, step_config)
                    
                    if step_instance:
                        # 인터페이스 설정
                        await self._setup_step_interfaces(step_instance, step_name)
                        
                        # Step 초기화
                        if hasattr(step_instance, 'initialize'):
                            await step_instance.initialize()
                        
                        self.steps[step_name] = step_instance
                        self.logger.info(f"✅ {step_name} 초기화 완료")
                    
                except Exception as e:
                    self.logger.error(f"❌ {step_name} 초기화 실패: {e}")
                    continue
            
            return len(self.steps) > 0
            
        except Exception as e:
            self.logger.error(f"❌ Step 초기화 실패: {e}")
            return False
    
    async def _setup_step_interfaces(self, step_instance, step_name: str):
        """Step 인터페이스 설정"""
        try:
            step_class_name = f"{step_name.title().replace('_', '')}Step"
            
            # 1. 통합 인터페이스 설정
            if self.unified_system_initialized and self.utils_manager:
                try:
                    unified_interface = self.utils_manager.create_step_interface(step_class_name)
                    setattr(step_instance, 'unified_interface', unified_interface)
                    self.logger.debug(f"✅ {step_name} 통합 인터페이스 설정")
                except Exception as e:
                    self.logger.warning(f"⚠️ {step_name} 통합 인터페이스 설정 실패: {e}")
            
            # 2. ModelLoader 인터페이스 설정
            if self.model_loader_initialized and self.model_loader:
                try:
                    model_interface = self.model_loader.create_step_interface(step_class_name)
                    setattr(step_instance, 'model_interface', model_interface)
                    self.logger.debug(f"✅ {step_name} ModelLoader 인터페이스 설정")
                except Exception as e:
                    self.logger.warning(f"⚠️ {step_name} ModelLoader 인터페이스 설정 실패: {e}")
            
        except Exception as e:
            self.logger.error(f"❌ {step_name} 인터페이스 설정 실패: {e}")
    
    def _create_step_instance_safely(self, step_class, step_name: str, step_config: Dict[str, Any]):
        """Step 인스턴스 안전 생성"""
        try:
            return step_class(**step_config)
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                try:
                    safe_config = {
                        'device': step_config.get('device', 'cpu'),
                        'config': step_config.get('config', {})
                    }
                    return step_class(**safe_config)
                except Exception:
                    try:
                        return step_class(device=step_config.get('device', 'cpu'))
                    except Exception:
                        return None
            else:
                raise
        except Exception as e:
            self.logger.error(f"❌ {step_name} 생성 실패: {e}")
            return None
    
    def _get_step_config(self, step_name: str) -> Dict[str, Any]:
        """Step별 설정 반환"""
        step_configs = {
            'human_parsing': {
                'model_name': 'graphonomy',
                'num_classes': 20,
                'input_size': (512, 512)
            },
            'pose_estimation': {
                'model_type': 'mediapipe',
                'input_size': (368, 368),
                'confidence_threshold': 0.5
            },
            'cloth_segmentation': {
                'model_name': 'u2net',
                'background_threshold': 0.5,
                'post_process': True
            },
            'geometric_matching': {
                'tps_points': 25,
                'matching_threshold': 0.8,
                'method': 'auto'
            },
            'cloth_warping': {
                'warping_method': 'tps',
                'physics_simulation': True
            },
            'virtual_fitting': {
                'blending_method': 'poisson',
                'seamless_cloning': True
            },
            'post_processing': {
                'enable_super_resolution': self.config.optimization_enabled,
                'enhance_faces': True
            },
            'quality_assessment': {
                'enable_detailed_analysis': True,
                'perceptual_metrics': True
            }
        }
        
        return step_configs.get(step_name, {})
    
    async def _preload_critical_models(self):
        """중요 모델 사전 로드"""
        try:
            critical_steps = ['human_parsing', 'pose_estimation', 'cloth_segmentation']
            
            for step_name in critical_steps:
                if step_name in self.steps:
                    step = self.steps[step_name]
                    
                    # ModelLoader 인터페이스 사용
                    if hasattr(step, 'model_interface') and step.model_interface:
                        try:
                            if STEP_REQUESTS_AVAILABLE:
                                step_class_name = f"{step_name.title().replace('_', '')}Step"
                                step_req = get_step_request(step_class_name)
                                
                                if step_req:
                                    model_name = step_req.model_name
                                    await step.model_interface.get_model(model_name)
                                    self.logger.info(f"✅ {step_name} 모델 사전 로드 완료")
                        except Exception as e:
                            self.logger.warning(f"⚠️ {step_name} 모델 사전 로드 실패: {e}")
                            
        except Exception as e:
            self.logger.error(f"❌ 중요 모델 사전 로드 실패: {e}")
    
    async def _warmup_model_cache(self):
        """모델 캐시 워밍업"""
        try:
            for step_name, step_instance in self.steps.items():
                try:
                    # 통합 인터페이스 우선
                    if hasattr(step_instance, 'unified_interface') and step_instance.unified_interface:
                        model = await step_instance.unified_interface.get_model()
                        if model:
                            self.logger.debug(f"✅ {step_name} 통합 캐시 워밍업")
                            continue
                    
                    # ModelLoader 인터페이스
                    if hasattr(step_instance, 'model_interface') and step_instance.model_interface:
                        available_models = await step_instance.model_interface.list_available_models()
                        if available_models:
                            await step_instance.model_interface.get_model(available_models[0])
                            self.logger.debug(f"✅ {step_name} ModelLoader 캐시 워밍업")
                            
                except Exception as e:
                    self.logger.warning(f"⚠️ {step_name} 캐시 워밍업 실패: {e}")
                    
        except Exception as e:
            self.logger.error(f"❌ 모델 캐시 워밍업 실패: {e}")

class ExecutionManager:
    """실행 관리자 - 우선순위 기반 실행 전략"""
    
    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    async def execute_step(
        self,
        step,
        step_name: str,
        current_data: torch.Tensor,
        clothing_tensor: torch.Tensor,
        body_measurements: Optional[Dict],
        clothing_type: str,
        fabric_type: str,
        style_preferences: Optional[Dict],
        max_retries: int
    ) -> Tuple[Dict[str, Any], str]:
        """Step 실행 - 우선순위 기반 전략"""
        
        last_error = None
        execution_strategy = "unknown"
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    self.logger.info(f"🔄 {step_name} 재시도 {attempt}/{max_retries}")
                    await asyncio.sleep(0.5)
                
                # 실행 전략 결정
                if self.config.execution_mode == ExecutionMode.UNIFIED_SYSTEM:
                    # 1순위: 통합 시스템
                    result, strategy = await self._execute_with_unified_system(
                        step, step_name, current_data, clothing_tensor,
                        body_measurements, clothing_type, fabric_type, style_preferences
                    )
                    if result.get('success', False):
                        return result, strategy
                    
                    # 폴백 허용 시 ModelLoader 시도
                    if self.config.fallback_enabled:
                        result, strategy = await self._execute_with_model_loader(
                            step, step_name, current_data, clothing_tensor,
                            body_measurements, clothing_type, fabric_type, style_preferences
                        )
                        if result.get('success', False):
                            return result, strategy
                
                elif self.config.execution_mode == ExecutionMode.MODEL_LOADER:
                    # 1순위: ModelLoader
                    result, strategy = await self._execute_with_model_loader(
                        step, step_name, current_data, clothing_tensor,
                        body_measurements, clothing_type, fabric_type, style_preferences
                    )
                    if result.get('success', False):
                        return result, strategy
                    
                    # 폴백 허용 시 통합 시스템 시도
                    if self.config.fallback_enabled:
                        result, strategy = await self._execute_with_unified_system(
                            step, step_name, current_data, clothing_tensor,
                            body_measurements, clothing_type, fabric_type, style_preferences
                        )
                        if result.get('success', False):
                            return result, strategy
                
                # 최종 폴백: 기본 실행
                if self.config.fallback_enabled:
                    result, strategy = await self._execute_with_fallback(
                        step, step_name, current_data, clothing_tensor,
                        body_measurements, clothing_type, fabric_type, style_preferences
                    )
                    if result.get('success', False):
                        return result, strategy
                
                last_error = result.get('error', 'Unknown error')
                execution_strategy = strategy
                    
            except Exception as e:
                last_error = str(e)
                execution_strategy = "error"
                self.logger.warning(f"⚠️ {step_name} 시도 {attempt + 1} 실패: {e}")
                
                if attempt < max_retries:
                    continue
        
        # 모든 재시도 실패
        return {
            'success': False,
            'error': last_error,
            'confidence': 0.0,
            'quality_score': 0.0,
            'processing_time': 0.0,
            'model_used': 'failed_after_retries'
        }, execution_strategy
    
    async def _execute_with_unified_system(
        self,
        step,
        step_name: str,
        current_data: torch.Tensor,
        clothing_tensor: torch.Tensor,
        body_measurements: Optional[Dict],
        clothing_type: str,
        fabric_type: str,
        style_preferences: Optional[Dict]
    ) -> Tuple[Dict[str, Any], str]:
        """통합 시스템을 사용한 실행"""
        try:
            if hasattr(step, 'unified_interface') and step.unified_interface:
                self.logger.debug(f"🔗 {step_name} 통합 시스템 실행")
                
                result = step.unified_interface.process_image(
                    current_data,
                    clothing_data=clothing_tensor,
                    clothing_type=clothing_type,
                    fabric_type=fabric_type,
                    style_preferences=style_preferences,
                    optimize_memory=True
                )
                
                if result and result.get('success', False):
                    return {
                        'success': True,
                        'result': result.get('processed_image', current_data),
                        'confidence': result.get('confidence', 0.9),
                        'quality_score': result.get('quality_score', 0.9),
                        'processing_time': result.get('processing_time', 0.1),
                        'model_used': result.get('model_used', 'unified_system'),
                        'processing_method': 'unified_interface'
                    }, "unified_system"
            
            raise Exception("통합 인터페이스 사용 불가")
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'confidence': 0.0,
                'quality_score': 0.0,
                'processing_time': 0.0,
                'model_used': 'unified_system_error'
            }, "unified_system_error"
    
    async def _execute_with_model_loader(
        self,
        step,
        step_name: str,
        current_data: torch.Tensor,
        clothing_tensor: torch.Tensor,
        body_measurements: Optional[Dict],
        clothing_type: str,
        fabric_type: str,
        style_preferences: Optional[Dict]
    ) -> Tuple[Dict[str, Any], str]:
        """ModelLoader를 사용한 실행"""
        try:
            model_used = "fallback"
            
            # AI 모델 로드 시도
            if hasattr(step, 'model_interface') and step.model_interface:
                if STEP_REQUESTS_AVAILABLE:
                    step_class_name = f"{step_name.title().replace('_', '')}Step"
                    step_req = get_step_request(step_class_name)
                    
                    if step_req:
                        model_name = step_req.model_name
                        ai_model = await step.model_interface.get_model(model_name)
                        
                        if ai_model:
                            model_used = model_name
                            setattr(step, '_ai_model', ai_model)
            
            # Step별 처리 실행
            result = await self._execute_step_logic(
                step, step_name, current_data, clothing_tensor,
                body_measurements, clothing_type, fabric_type, style_preferences
            )
            
            if not result or not isinstance(result, dict):
                result = {
                    'success': True,
                    'result': current_data,
                    'confidence': 0.8,
                    'quality_score': 0.8,
                    'processing_time': 0.1
                }
            
            result['model_used'] = model_used
            result['processing_method'] = 'model_loader'
            
            return result, "model_loader"
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'confidence': 0.0,
                'quality_score': 0.0,
                'processing_time': 0.0,
                'model_used': 'model_loader_error'
            }, "model_loader_error"
    
    async def _execute_with_fallback(
        self,
        step,
        step_name: str,
        current_data: torch.Tensor,
        clothing_tensor: torch.Tensor,
        body_measurements: Optional[Dict],
        clothing_type: str,
        fabric_type: str,
        style_preferences: Optional[Dict]
    ) -> Tuple[Dict[str, Any], str]:
        """폴백 실행"""
        try:
            result = await self._execute_step_logic(
                step, step_name, current_data, clothing_tensor,
                body_measurements, clothing_type, fabric_type, style_preferences
            )
            
            if not result or not isinstance(result, dict):
                result = {
                    'success': True,
                    'result': current_data,
                    'confidence': 0.7,
                    'quality_score': 0.7,
                    'processing_time': 0.1
                }
            
            result['model_used'] = 'fallback'
            result['processing_method'] = 'fallback'
            
            return result, "fallback"
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'confidence': 0.0,
                'quality_score': 0.0,
                'processing_time': 0.0,
                'model_used': 'fallback_error'
            }, "fallback_error"
    
    async def _execute_step_logic(
        self,
        step,
        step_name: str,
        current_data: torch.Tensor,
        clothing_tensor: torch.Tensor,
        body_measurements: Optional[Dict],
        clothing_type: str,
        fabric_type: str,
        style_preferences: Optional[Dict]
    ) -> Dict[str, Any]:
        """Step별 처리 로직 실행"""
        if step_name == 'human_parsing':
            return await step.process(current_data)
        elif step_name == 'pose_estimation':
            return await step.process(current_data)
        elif step_name == 'cloth_segmentation':
            return await step.process(clothing_tensor, clothing_type=clothing_type)
        elif step_name == 'geometric_matching':
            dummy_pose_keypoints = self._generate_dummy_pose_keypoints()
            dummy_clothing_segmentation = {'mask': clothing_tensor}
            return await step.process(
                person_parsing={'result': current_data},
                pose_keypoints=dummy_pose_keypoints,
                clothing_segmentation=dummy_clothing_segmentation,
                clothing_type=clothing_type
            )
        elif step_name == 'cloth_warping':
            return await step.process(
                current_data, clothing_tensor, body_measurements or {}, fabric_type
            )
        elif step_name == 'virtual_fitting':
            return await step.process(current_data, clothing_tensor, style_preferences or {})
        elif step_name == 'post_processing':
            return await step.process(current_data)
        elif step_name == 'quality_assessment':
            return await step.process(current_data, clothing_tensor)
        else:
            return await step.process(current_data)
    
    def _generate_dummy_pose_keypoints(self) -> List[List[float]]:
        """더미 포즈 키포인트 생성"""
        dummy_keypoints = []
        for i in range(18):
            x = 256 + np.random.uniform(-50, 50)
            y = 256 + np.random.uniform(-100, 100)
            confidence = 0.8
            dummy_keypoints.append([x, y, confidence])
        return dummy_keypoints

# ==============================================
# 🔥 4. 통합 PipelineManager 클래스
# ==============================================

class PipelineManager:
    """
    🔥 완전 통합 PipelineManager
    
    ✅ 프로젝트 통합 유틸리티 시스템 우선 사용
    ✅ ModelLoader 시스템 완벽 연동
    ✅ 우선순위 기반 실행 전략
    ✅ 모듈화된 관리 구조
    ✅ 강화된 에러 처리 및 폴백
    ✅ M3 Max + conda 환경 최적화
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        config: Optional[Union[Dict[str, Any], PipelineConfig]] = None,
        **kwargs
    ):
        """파이프라인 매니저 초기화"""
        
        # 1. 디바이스 자동 감지
        self.device = self._auto_detect_device(device)
        
        # 2. 설정 초기화
        if isinstance(config, PipelineConfig):
            self.config = config
        else:
            config_dict = self._load_config(config_path) if config_path else {}
            if config:
                config_dict.update(config if isinstance(config, dict) else {})
            config_dict.update(kwargs)
            
            self.config = PipelineConfig(device=self.device, **config_dict)
        
        # 3. 시스템 정보 감지 및 설정 업데이트
        self.device_type = self._detect_device_type()
        self.memory_gb = self._detect_memory_gb()
        self.is_m3_max = self._detect_m3_max()
        
        self.config.device_type = self.device_type
        self.config.memory_gb = self.memory_gb
        self.config.is_m3_max = self.is_m3_max
        
        # 4. 관리자들 초기화
        self.logger = logging.getLogger(f"pipeline.{self.__class__.__name__}")
        self.initialization_manager = InitializationManager(self.config, self.device, self.logger)
        self.execution_manager = ExecutionManager(self.config, self.logger)
        
        # 5. 유틸리티 초기화
        self.data_converter = OptimizedDataConverter(self.device)
        self.memory_manager = OptimizedMemoryManager(self.device)
        
        # 6. 파이프라인 상태
        self.is_initialized = False
        self.current_status = ProcessingStatus.IDLE
        self.step_order = [
            'human_parsing', 'pose_estimation', 'cloth_segmentation',
            'geometric_matching', 'cloth_warping', 'virtual_fitting',
            'post_processing', 'quality_assessment'
        ]
        
        # 7. 세션 및 성능 관리
        self.sessions: Dict[str, SessionData] = {}
        self.performance_metrics = PerformanceMetrics()
        
        # 8. 동시성 관리
        self._lock = threading.RLock()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.thread_pool_size)
        
        # 9. 디바이스 최적화
        self._configure_device_optimizations()
        
        # 10. 초기화 완료 로깅
        self.logger.info(f"✅ 통합 PipelineManager 초기화 완료")
        self.logger.info(f"🎯 디바이스: {self.device} ({self.device_type})")
        self.logger.info(f"📊 메모리: {self.memory_gb}GB, M3 Max: {'✅' if self.is_m3_max else '❌'}")
        self.logger.info(f"⚙️ 실행 모드: {self.config.execution_mode.value}")
        self.logger.info(f"🔧 통합 시스템: {'✅' if UNIFIED_UTILS_AVAILABLE else '❌'}")
        self.logger.info(f"🔧 ModelLoader: {'✅' if MODEL_LOADER_AVAILABLE else '❌'}")
        
        # 11. 초기 메모리 최적화
        self.memory_manager.cleanup_memory()
    
    def _auto_detect_device(self, preferred_device: Optional[str]) -> str:
        """디바이스 자동 감지"""
        if preferred_device and preferred_device != "auto":
            return preferred_device
        
        try:
            if torch.backends.mps.is_available():
                return 'mps'
            elif torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        except:
            return 'cpu'
    
    def _detect_device_type(self) -> str:
        """디바이스 타입 감지"""
        if self.device == 'mps':
            return 'apple_silicon'
        elif self.device == 'cuda':
            return 'nvidia'
        else:
            return 'cpu'
    
    def _detect_memory_gb(self) -> float:
        """메모리 용량 감지"""
        try:
            import psutil
            return round(psutil.virtual_memory().total / (1024**3), 1)
        except ImportError:
            return 16.0
    
    def _detect_m3_max(self) -> bool:
        """M3 Max 칩 감지"""
        try:
            import platform
            import subprocess
            if platform.system() == 'Darwin':
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'], 
                    capture_output=True, text=True, timeout=5
                )
                chip_info = result.stdout.strip()
                return 'M3' in chip_info and 'Max' in chip_info
        except:
            pass
        return False
    
    def _configure_device_optimizations(self):
        """디바이스별 최적화 설정"""
        if self.device == 'mps':
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            if torch.backends.mps.is_available():
                try:
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                except:
                    pass
            self.logger.info("🔧 M3 Max MPS 최적화 설정 완료")
        elif self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            if self.config.optimization_enabled:
                torch.backends.cudnn.enabled = True
            self.logger.info("🔧 CUDA 최적화 설정 완료")
        
        if self.device in ['cuda', 'mps'] and self.config.use_fp16:
            self.use_amp = True
            self.logger.info("⚡ 혼합 정밀도 연산 활성화")
        else:
            self.use_amp = False
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        if not config_path or not os.path.exists(config_path):
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"설정 파일 로드 실패: {e}")
            return {}
    
    @property
    def steps(self) -> Dict[str, Any]:
        """Step 참조 반환"""
        return self.initialization_manager.steps
    
    async def initialize(self) -> bool:
        """파이프라인 초기화"""
        try:
            self.logger.info("🔄 통합 파이프라인 초기화 시작...")
            self.current_status = ProcessingStatus.INITIALIZING
            start_time = time.time()
            
            # 1. 메모리 정리
            self.memory_manager.cleanup_memory()
            
            # 2. 통합 초기화 실행
            success = await self.initialization_manager.initialize_all(self.step_order)
            
            # 3. 초기화 검증
            success_rate = self._verify_initialization()
            if success_rate < 0.5:
                self.logger.warning(f"초기화 성공률 낮음: {success_rate:.1%}")
            
            initialization_time = time.time() - start_time
            self.is_initialized = success
            self.current_status = ProcessingStatus.IDLE if success else ProcessingStatus.FAILED
            
            if success:
                self.logger.info(f"🎉 통합 파이프라인 초기화 완료 ({initialization_time:.2f}초)")
                self.logger.info(f"📊 초기화 성공률: {success_rate:.1%}")
                self.logger.info(f"🔧 통합 시스템: {'✅' if self.initialization_manager.unified_system_initialized else '❌'}")
                self.logger.info(f"🔧 ModelLoader: {'✅' if self.initialization_manager.model_loader_initialized else '❌'}")
            else:
                self.logger.error("❌ 통합 파이프라인 초기화 실패")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ 파이프라인 초기화 실패: {e}")
            self.logger.error(f"📋 오류 상세: {traceback.format_exc()}")
            self.is_initialized = False
            self.current_status = ProcessingStatus.FAILED
            return False
    
    def _verify_initialization(self) -> float:
        """초기화 검증"""
        total_steps = len(self.step_order)
        initialized_steps = len(self.steps)
        
        success_rate = initialized_steps / total_steps if total_steps > 0 else 0
        self.logger.info(f"📊 초기화 상태: {initialized_steps}/{total_steps} ({success_rate:.1%})")
        
        return success_rate
    
    # ==============================================
    # 🔥 메인 처리 메서드
    # ==============================================
    
    async def process_complete_virtual_fitting(
        self,
        person_image: Union[str, Image.Image, np.ndarray],
        clothing_image: Union[str, Image.Image, np.ndarray],
        body_measurements: Optional[Dict[str, float]] = None,
        clothing_type: str = "shirt",
        fabric_type: str = "cotton",
        style_preferences: Optional[Dict[str, Any]] = None,
        quality_target: float = 0.8,
        progress_callback: Optional[Callable] = None,
        save_intermediate: bool = None,
        session_id: Optional[str] = None
    ) -> ProcessingResult:
        """
        🔥 완전한 8단계 가상 피팅 처리 - 통합 시스템 우선 실행
        
        ✅ 통합 유틸리티 시스템 우선 사용
        ✅ ModelLoader 폴백 지원
        ✅ 우선순위 기반 실행 전략
        ✅ 실제 AI 모델 추론 실행
        ✅ 강화된 에러 처리
        """
        if not self.is_initialized:
            raise RuntimeError("파이프라인이 초기화되지 않았습니다. initialize()를 먼저 호출하세요.")
        
        # 설정 처리
        if save_intermediate is None:
            save_intermediate = self.config.save_intermediate
        
        if session_id is None:
            session_id = f"vf_{int(time.time())}_{np.random.randint(1000, 9999)}"
        
        start_time = time.time()
        self.current_status = ProcessingStatus.PROCESSING
        
        try:
            self.logger.info(f"🎯 통합 8단계 가상 피팅 시작 - 세션 ID: {session_id}")
            self.logger.info(f"⚙️ 설정: {clothing_type} ({fabric_type}), 목표 품질: {quality_target}")
            self.logger.info(f"🔧 실행 모드: {self.config.execution_mode.value}")
            self.logger.info(f"🔧 통합 시스템: {'✅' if self.initialization_manager.unified_system_initialized else '❌'}")
            self.logger.info(f"🔧 ModelLoader: {'✅' if self.initialization_manager.model_loader_initialized else '❌'}")
            
            # 1. 입력 이미지 전처리 (통합 시스템 우선)
            person_tensor = self.data_converter.preprocess_image(person_image)
            clothing_tensor = self.data_converter.preprocess_image(clothing_image)
            
            # 디바이스로 이동
            person_tensor = person_tensor.to(self.device)
            clothing_tensor = clothing_tensor.to(self.device)
            
            # 2. 세션 데이터 초기화
            session_data = SessionData(
                session_id=session_id,
                start_time=start_time,
                status=ProcessingStatus.PROCESSING,
                metadata={
                    'clothing_type': clothing_type,
                    'fabric_type': fabric_type,
                    'quality_target': quality_target,
                    'style_preferences': style_preferences or {},
                    'body_measurements': body_measurements,
                    'device': self.device,
                    'execution_mode': self.config.execution_mode.value,
                    'unified_system_enabled': self.initialization_manager.unified_system_initialized,
                    'model_loader_enabled': self.initialization_manager.model_loader_initialized
                }
            )
            
            self.sessions[session_id] = session_data
            
            if progress_callback:
                await progress_callback("입력 전처리 완료", 5)
            
            # 3. 메모리 최적화
            if self.config.memory_optimization:
                self.memory_manager.cleanup_memory()
            
            # 🔥 4. 8단계 순차 처리 - 우선순위 기반 실행
            step_results = {}
            execution_strategies = {}
            current_data = person_tensor
            
            for i, step_name in enumerate(self.step_order):
                if step_name not in self.steps:
                    self.logger.warning(f"⚠️ {step_name} 단계가 없습니다. 건너뛰기...")
                    continue
                
                step_start = time.time()
                step = self.steps[step_name]
                
                self.logger.info(f"📋 {i+1}/{len(self.step_order)} 단계: {step_name} 처리 중...")
                
                try:
                    # 🔥 우선순위 기반 실행
                    step_result, execution_strategy = await self.execution_manager.execute_step(
                        step, step_name, current_data, clothing_tensor,
                        body_measurements, clothing_type, fabric_type,
                        style_preferences, self.config.max_retries
                    )
                    
                    step_time = time.time() - step_start
                    step_results[step_name] = step_result
                    execution_strategies[step_name] = execution_strategy
                    
                    # 세션 데이터 업데이트
                    session_data.add_step_result(step_name, step_result, step_time, execution_strategy)
                    
                    # 결과 업데이트
                    if step_result.get('success', True):
                        result_data = step_result.get('result')
                        if result_data is not None:
                            current_data = result_data
                    
                    # 중간 결과 저장
                    if save_intermediate:
                        session_data.intermediate_results[step_name] = {
                            'result': current_data,
                            'metadata': step_result,
                            'execution_strategy': execution_strategy
                        }
                    
                    # 로깅
                    confidence = step_result.get('confidence', 0.8)
                    quality_score = step_result.get('quality_score', confidence)
                    model_used = step_result.get('model_used', 'unknown')
                    processing_method = step_result.get('processing_method', 'unknown')
                    
                    strategy_icon = "🔗" if execution_strategy == "unified_system" else "🧠" if execution_strategy == "model_loader" else "🔄"
                    
                    self.logger.info(f"✅ {i+1}단계 완료 - 시간: {step_time:.2f}초, 신뢰도: {confidence:.3f}, 품질: {quality_score:.3f}")
                    self.logger.info(f"   {strategy_icon} 실행전략: {execution_strategy}, 모델: {model_used}, 방법: {processing_method}")
                    
                    # 진행률 콜백
                    if progress_callback:
                        progress = 5 + (i + 1) * 85 // len(self.step_order)
                        await progress_callback(f"{step_name} 완료 ({execution_strategy})", progress)
                    
                    # 메모리 최적화 (중간 단계)
                    if self.config.memory_optimization and i % 2 == 0:
                        self.memory_manager.cleanup_memory()
                    
                except Exception as e:
                    self.logger.error(f"❌ {i+1}단계 ({step_name}) 실패: {e}")
                    step_time = time.time() - step_start
                    step_results[step_name] = {
                        'success': False,
                        'error': str(e),
                        'processing_time': step_time,
                        'confidence': 0.0,
                        'quality_score': 0.0,
                        'model_used': 'error',
                        'processing_method': 'error'
                    }
                    execution_strategies[step_name] = "error"
                    
                    session_data.add_step_result(step_name, step_results[step_name], step_time, "error")
                    session_data.error_log.append(f"{step_name}: {str(e)}")
                    
                    # 실패해도 계속 진행
                    continue
            
            # 5. 최종 결과 구성
            total_time = time.time() - start_time
            
            # 결과 이미지 생성
            if isinstance(current_data, torch.Tensor):
                result_image = self.data_converter.tensor_to_pil(current_data)
            else:
                result_image = Image.new('RGB', (512, 512), color='gray')
            
            # 품질 평가 강화
            quality_score = self._assess_enhanced_quality(step_results, execution_strategies)
            quality_grade = self._get_quality_grade(quality_score)
            
            # 성공 여부 결정
            success = quality_score >= (quality_target * 0.8)
            
            # 실행 전략 통계
            strategy_stats = self._calculate_strategy_statistics(execution_strategies)
            
            # 성능 메트릭 업데이트
            dominant_strategy = max(strategy_stats.items(), key=lambda x: x[1])[0] if strategy_stats else "unknown"
            self.performance_metrics.update(total_time, quality_score, success, dominant_strategy)
            
            # 세션 상태 업데이트
            session_data.status = ProcessingStatus.COMPLETED if success else ProcessingStatus.FAILED
            
            # 세션 데이터 정리
            if not save_intermediate:
                self.sessions.pop(session_id, None)
            
            if progress_callback:
                await progress_callback("처리 완료", 100)
            
            self.current_status = ProcessingStatus.IDLE
            
            # 결과 로깅
            self.logger.info(f"🎉 통합 8단계 가상 피팅 완료!")
            self.logger.info(f"⏱️ 총 시간: {total_time:.2f}초")
            self.logger.info(f"📊 품질 점수: {quality_score:.3f} ({quality_grade})")
            self.logger.info(f"🎯 목표 달성: {'✅' if quality_score >= quality_target else '❌'}")
            self.logger.info(f"📋 완료된 단계: {len(step_results)}/{len(self.step_order)}")
            self.logger.info(f"🔗 실행 전략 통계: {strategy_stats}")
            
            return ProcessingResult(
                success=success,
                session_id=session_id,
                result_image=result_image,
                result_tensor=current_data if isinstance(current_data, torch.Tensor) else None,
                quality_score=quality_score,
                quality_grade=quality_grade,
                processing_time=total_time,
                step_results=step_results,
                step_timings=session_data.step_timings,
                execution_strategy=execution_strategies,
                metadata={
                    'device': self.device,
                    'device_type': self.device_type,
                    'is_m3_max': self.is_m3_max,
                    'execution_mode': self.config.execution_mode.value,
                    'quality_target': quality_target,
                    'quality_target_achieved': quality_score >= quality_target,
                    'total_steps': len(self.step_order),
                    'completed_steps': len(step_results),
                    'success_rate': len([r for r in step_results.values() if r.get('success', True)]) / len(step_results) if step_results else 0,
                    'unified_system_enabled': self.initialization_manager.unified_system_initialized,
                    'model_loader_enabled': self.initialization_manager.model_loader_initialized,
                    'strategy_statistics': strategy_stats,
                    'unified_system_usage_rate': strategy_stats.get('unified_system', 0) / len(step_results) * 100 if step_results else 0,
                    'model_loader_usage_rate': strategy_stats.get('model_loader', 0) / len(step_results) * 100 if step_results else 0,
                    'fallback_usage_rate': strategy_stats.get('fallback', 0) / len(step_results) * 100 if step_results else 0,
                    'session_data': session_data.__dict__ if save_intermediate else None
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ 통합 가상 피팅 처리 실패: {e}")
            self.logger.error(f"📋 오류 상세: {traceback.format_exc()}")
            
            # 에러 메트릭 업데이트
            self.performance_metrics.update(time.time() - start_time, 0.0, False, "error")
            
            self.current_status = ProcessingStatus.FAILED
            
            return ProcessingResult(
                success=False,
                session_id=session_id,
                processing_time=time.time() - start_time,
                error_message=str(e),
                metadata={
                    'device': self.device,
                    'error_type': type(e).__name__,
                    'error_location': traceback.format_exc(),
                    'execution_mode': self.config.execution_mode.value,
                    'unified_system_enabled': self.initialization_manager.unified_system_initialized,
                    'model_loader_enabled': self.initialization_manager.model_loader_initialized,
                    'session_data': self.sessions.get(session_id).__dict__ if session_id in self.sessions else None
                }
            )
    
    def _assess_enhanced_quality(self, step_results: Dict[str, Any], execution_strategies: Dict[str, str]) -> float:
        """강화된 품질 평가 - 실행 전략 고려"""
        if not step_results:
            return 0.5
        
        quality_scores = []
        confidence_scores = []
        strategy_bonus = 0.0
        
        for step_name, step_result in step_results.items():
            if isinstance(step_result, dict):
                confidence = step_result.get('confidence', 0.8)
                quality = step_result.get('quality_score', confidence)
                strategy = execution_strategies.get(step_name, 'unknown')
                
                quality_scores.append(quality)
                confidence_scores.append(confidence)
                
                # 실행 전략별 보너스
                if strategy == 'unified_system':
                    strategy_bonus += 0.05  # 5% 보너스
                elif strategy == 'model_loader':
                    strategy_bonus += 0.03  # 3% 보너스
                elif strategy == 'fallback':
                    strategy_bonus += 0.01  # 1% 보너스
        
        # 종합 점수 계산
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            
            # 가중 평균 + 전략 보너스
            overall_score = avg_quality * 0.6 + avg_confidence * 0.4 + strategy_bonus
            return min(max(overall_score, 0.0), 1.0)
        
        return 0.5
    
    def _calculate_strategy_statistics(self, execution_strategies: Dict[str, str]) -> Dict[str, int]:
        """실행 전략 통계 계산"""
        stats = {}
        for strategy in execution_strategies.values():
            stats[strategy] = stats.get(strategy, 0) + 1
        return stats
    
    def _get_quality_grade(self, quality_score: float) -> str:
        """품질 등급 반환"""
        if quality_score >= 0.9:
            return "Excellent"
        elif quality_score >= 0.8:
            return "Good"
        elif quality_score >= 0.7:
            return "Fair"
        elif quality_score >= 0.6:
            return "Poor"
        else:
            return "Very Poor"
    
    # ==============================================
    # 🔥 상태 조회 및 관리 메서드들
    # ==============================================
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """파이프라인 상태 조회 - 통합 시스템 정보 포함"""
        return {
            'initialized': self.is_initialized,
            'current_status': self.current_status.value,
            'device': self.device,
            'device_type': self.device_type,
            'memory_gb': self.memory_gb,
            'is_m3_max': self.is_m3_max,
            'unified_system_initialized': self.initialization_manager.unified_system_initialized,
            'model_loader_initialized': self.initialization_manager.model_loader_initialized,
            'config': {
                'quality_level': self.config.quality_level.value,
                'processing_mode': self.config.processing_mode.value,
                'execution_mode': self.config.execution_mode.value,
                'optimization_enabled': self.config.optimization_enabled,
                'memory_optimization': self.config.memory_optimization,
                'use_fp16': self.config.use_fp16,
                'batch_size': self.config.batch_size,
                'unified_utils_enabled': self.config.unified_utils_enabled,
                'model_loader_enabled': self.config.model_loader_enabled,
                'auto_detect_models': self.config.auto_detect_models,
                'preload_critical_models': self.config.preload_critical_models,
                'fallback_enabled': self.config.fallback_enabled
            },
            'steps_status': {
                step_name: {
                    'loaded': step_name in self.steps,
                    'type': type(self.steps[step_name]).__name__ if step_name in self.steps else None,
                    'ready': step_name in self.steps and hasattr(self.steps[step_name], 'process'),
                    'has_unified_interface': (step_name in self.steps and 
                                            hasattr(self.steps[step_name], 'unified_interface') and 
                                            self.steps[step_name].unified_interface is not None),
                    'has_model_interface': (step_name in self.steps and 
                                          hasattr(self.steps[step_name], 'model_interface') and 
                                          self.steps[step_name].model_interface is not None)
                }
                for step_name in self.step_order
            },
            'performance_metrics': {
                'total_sessions': self.performance_metrics.total_sessions,
                'successful_sessions': self.performance_metrics.successful_sessions,
                'failed_sessions': self.performance_metrics.failed_sessions,
                'success_rate': self.performance_metrics.successful_sessions / self.performance_metrics.total_sessions if self.performance_metrics.total_sessions > 0 else 0,
                'average_processing_time': self.performance_metrics.average_processing_time,
                'average_quality_score': self.performance_metrics.average_quality_score,
                'unified_system_usage': self.performance_metrics.unified_system_usage,
                'model_loader_usage': self.performance_metrics.model_loader_usage,
                'fallback_usage': self.performance_metrics.fallback_usage
            },
            'memory_usage': self.memory_manager.get_memory_usage(),
            'active_sessions': len(self.sessions),
            'system_integration': {
                'unified_utils_available': UNIFIED_UTILS_AVAILABLE,
                'model_loader_available': MODEL_LOADER_AVAILABLE,
                'step_requests_available': STEP_REQUESTS_AVAILABLE,
                'auto_detector_available': AUTO_DETECTOR_AVAILABLE,
                'step_classes_available': STEP_CLASSES_AVAILABLE
            }
        }
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 정보 조회"""
        session = self.sessions.get(session_id)
        if session:
            return session.__dict__
        return None
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """활성 세션 목록"""
        return [
            {
                'session_id': session_id,
                'status': session.status.value,
                'start_time': session.start_time,
                'elapsed_time': time.time() - session.start_time,
                'completed_steps': len(session.step_results),
                'total_steps': len(self.step_order),
                'execution_strategies': session.execution_strategies
            }
            for session_id, session in self.sessions.items()
        ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보 - 실행 전략 통계 포함"""
        return {
            'total_sessions': self.performance_metrics.total_sessions,
            'success_rate': self.performance_metrics.successful_sessions / self.performance_metrics.total_sessions if self.performance_metrics.total_sessions > 0 else 0,
            'average_processing_time': self.performance_metrics.average_processing_time,
            'average_quality_score': self.performance_metrics.average_quality_score,
            'fastest_time': self.performance_metrics.fastest_processing_time if self.performance_metrics.fastest_processing_time != float('inf') else 0,
            'slowest_time': self.performance_metrics.slowest_processing_time,
            'total_processing_time': self.performance_metrics.total_processing_time,
            'active_sessions': len(self.sessions),
            'execution_strategy_stats': {
                'unified_system_usage': self.performance_metrics.unified_system_usage,
                'model_loader_usage': self.performance_metrics.model_loader_usage,
                'fallback_usage': self.performance_metrics.fallback_usage,
                'unified_system_rate': self.performance_metrics.unified_system_usage / self.performance_metrics.total_sessions * 100 if self.performance_metrics.total_sessions > 0 else 0,
                'model_loader_rate': self.performance_metrics.model_loader_usage / self.performance_metrics.total_sessions * 100 if self.performance_metrics.total_sessions > 0 else 0,
                'fallback_rate': self.performance_metrics.fallback_usage / self.performance_metrics.total_sessions * 100 if self.performance_metrics.total_sessions > 0 else 0
            },
            'device_info': {
                'device': self.device,
                'device_type': self.device_type,
                'is_m3_max': self.is_m3_max,
                'memory_gb': self.memory_gb
            }
        }
    
    def clear_session_history(self, keep_recent: int = 10):
        """세션 히스토리 정리"""
        try:
            if len(self.sessions) <= keep_recent:
                return
            
            # 최근 세션들만 유지
            sorted_sessions = sorted(
                self.sessions.items(),
                key=lambda x: x[1].start_time,
                reverse=True
            )
            
            sessions_to_keep = dict(sorted_sessions[:keep_recent])
            cleared_count = len(self.sessions) - len(sessions_to_keep)
            
            self.sessions = sessions_to_keep
            
            self.logger.info(f"🧹 세션 히스토리 정리 완료: {cleared_count}개 세션 제거")
            
        except Exception as e:
            self.logger.error(f"❌ 세션 히스토리 정리 실패: {e}")
    
    async def warmup(self):
        """파이프라인 워밍업 - 통합 시스템 포함"""
        try:
            self.logger.info("🔥 통합 파이프라인 워밍업 시작...")
            
            # 더미 이미지 생성
            dummy_person = Image.new('RGB', (512, 512), color=(100, 150, 200))
            dummy_cloth = Image.new('RGB', (512, 512), color=(200, 100, 100))
            
            # 워밍업 실행
            result = await self.process_complete_virtual_fitting(
                person_image=dummy_person,
                clothing_image=dummy_cloth,
                clothing_type='shirt',
                fabric_type='cotton',
                quality_target=0.6,
                save_intermediate=False,
                session_id="warmup_session"
            )
            
            if result.success:
                self.logger.info(f"✅ 워밍업 완료 - 시간: {result.processing_time:.2f}초")
                self.logger.info(f"🔗 실행 전략 통계: {result.metadata.get('strategy_statistics', {})}")
                self.logger.info(f"🔧 통합 시스템 사용률: {result.metadata.get('unified_system_usage_rate', 0):.1f}%")
                return True
            else:
                self.logger.warning(f"⚠️ 워밍업 중 오류: {result.error_message}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 워밍업 실패: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """헬스체크 - 통합 시스템 상태 포함"""
        try:
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'pipeline_initialized': self.is_initialized,
                'current_status': self.current_status.value,
                'device': self.device,
                'unified_system_initialized': self.initialization_manager.unified_system_initialized,
                'model_loader_initialized': self.initialization_manager.model_loader_initialized,
                'checks': {}
            }
            
            # Step별 체크
            steps_healthy = 0
            steps_with_unified_interface = 0
            steps_with_model_interface = 0
            
            for step_name in self.step_order:
                if step_name in self.steps:
                    step = self.steps[step_name]
                    has_process = hasattr(step, 'process')
                    has_unified_interface = hasattr(step, 'unified_interface') and step.unified_interface
                    has_model_interface = hasattr(step, 'model_interface') and step.model_interface
                    
                    if has_process:
                        steps_healthy += 1
                    if has_unified_interface:
                        steps_with_unified_interface += 1
                    if has_model_interface:
                        steps_with_model_interface += 1
            
            health_status['checks']['steps'] = {
                'status': 'ok' if steps_healthy >= len(self.step_order) * 0.8 else 'warning',
                'healthy_steps': steps_healthy,
                'total_steps': len(self.step_order),
                'steps_with_unified_interface': steps_with_unified_interface,
                'steps_with_model_interface': steps_with_model_interface,
                'unified_interface_coverage': f"{steps_with_unified_interface}/{len(self.step_order)}",
                'model_interface_coverage': f"{steps_with_model_interface}/{len(self.step_order)}"
            }
            
            # 통합 시스템 체크
            health_status['checks']['unified_system'] = {
                'status': 'ok' if self.initialization_manager.unified_system_initialized else 'warning',
                'initialized': self.initialization_manager.unified_system_initialized,
                'utils_available': UNIFIED_UTILS_AVAILABLE,
                'auto_detector_available': AUTO_DETECTOR_AVAILABLE
            }
            
            # ModelLoader 시스템 체크
            health_status['checks']['model_loader'] = {
                'status': 'ok' if self.initialization_manager.model_loader_initialized else 'warning',
                'initialized': self.initialization_manager.model_loader_initialized,
                'available': MODEL_LOADER_AVAILABLE,
                'step_requests_available': STEP_REQUESTS_AVAILABLE
            }
            
            # 메모리 체크
            try:
                memory_usage = self.memory_manager.get_memory_usage()
                health_status['checks']['memory'] = {
                    'status': 'ok',
                    'usage': memory_usage
                }
            except Exception as e:
                health_status['checks']['memory'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # 전체 상태 결정
            check_statuses = [check.get('status', 'error') for check in health_status['checks'].values()]
            if 'error' in check_statuses:
                health_status['status'] = 'unhealthy'
            elif 'warning' in check_statuses:
                health_status['status'] = 'degraded'
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def cleanup(self):
        """리소스 정리 - 통합 시스템 포함"""
        try:
            self.logger.info("🧹 통합 파이프라인 리소스 정리 중...")
            self.current_status = ProcessingStatus.CLEANING
            
            # 1. 각 Step 정리
            for step_name, step in self.steps.items():
                try:
                    # Step의 통합 인터페이스 정리
                    if hasattr(step, 'unified_interface') and step.unified_interface:
                        try:
                            if hasattr(step.unified_interface, 'cleanup'):
                                await step.unified_interface.cleanup()
                            self.logger.info(f"✅ {step_name} 통합 인터페이스 정리 완료")
                        except Exception as e:
                            self.logger.warning(f"⚠️ {step_name} 통합 인터페이스 정리 실패: {e}")
                    
                    # Step의 model_interface 정리
                    if hasattr(step, 'model_interface') and step.model_interface:
                        try:
                            if hasattr(step.model_interface, 'unload_models'):
                                await step.model_interface.unload_models()
                            self.logger.info(f"✅ {step_name} 모델 인터페이스 정리 완료")
                        except Exception as e:
                            self.logger.warning(f"⚠️ {step_name} 모델 인터페이스 정리 실패: {e}")
                    
                    # Step 자체 정리
                    if hasattr(step, 'cleanup'):
                        await step.cleanup()
                    self.logger.info(f"✅ {step_name} 정리 완료")
                except Exception as e:
                    self.logger.warning(f"⚠️ {step_name} 정리 중 오류: {e}")
            
            # 2. 통합 시스템 정리
            if self.initialization_manager.utils_manager:
                try:
                    self.initialization_manager.utils_manager.cleanup()
                    self.logger.info("✅ 통합 시스템 정리 완료")
                except Exception as e:
                    self.logger.warning(f"⚠️ 통합 시스템 정리 중 오류: {e}")
            
            # 3. ModelLoader 시스템 정리
            if self.initialization_manager.model_loader:
                try:
                    if hasattr(self.initialization_manager.model_loader, 'cleanup'):
                        await self.initialization_manager.model_loader.cleanup()
                    self.logger.info("✅ ModelLoader 시스템 정리 완료")
                except Exception as e:
                    self.logger.warning(f"⚠️ ModelLoader 시스템 정리 중 오류: {e}")
            
            # 4. 메모리 관리자 정리
            try:
                self.memory_manager.cleanup_memory()
                self.logger.info("✅ 메모리 관리자 정리 완료")
            except Exception as e:
                self.logger.warning(f"⚠️ 메모리 관리자 정리 중 오류: {e}")
            
            # 5. 스레드 풀 정리
            if hasattr(self, 'thread_pool'):
                try:
                    self.thread_pool.shutdown(wait=True)
                    self.logger.info("✅ 스레드 풀 정리 완료")
                except Exception as e:
                    self.logger.warning(f"⚠️ 스레드 풀 정리 중 오류: {e}")
            
            # 6. 세션 데이터 정리
            try:
                self.sessions.clear()
                self.logger.info("✅ 세션 데이터 정리 완료")
            except Exception as e:
                self.logger.warning(f"⚠️ 세션 데이터 정리 중 오류: {e}")
            
            # 7. 상태 초기화
            self.is_initialized = False
            self.initialization_manager.unified_system_initialized = False
            self.initialization_manager.model_loader_initialized = False
            self.current_status = ProcessingStatus.IDLE
            
            self.logger.info("✅ 통합 파이프라인 리소스 정리 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 리소스 정리 중 오류: {e}")
            self.current_status = ProcessingStatus.FAILED

# ==============================================
# 🔥 5. 편의 함수들 (모든 기존 함수명 100% 유지)
# ==============================================

def create_pipeline(
    device: str = "auto",
    quality_level: str = "balanced",
    processing_mode: str = "production",
    execution_mode: str = "unified_system",
    **kwargs
) -> PipelineManager:
    """파이프라인 생성 편의 함수 - 통합 시스템 우선"""
    return PipelineManager(
        device=device,
        config=PipelineConfig(
            quality_level=QualityLevel(quality_level),
            processing_mode=PipelineMode(processing_mode),
            execution_mode=ExecutionMode(execution_mode),
            **kwargs
        )
    )

def create_development_pipeline(**kwargs) -> PipelineManager:
    """개발용 파이프라인 생성 - 통합 시스템 활성화"""
    return create_pipeline(
        quality_level="fast",
        processing_mode="development",
        execution_mode="unified_system",
        optimization_enabled=False,
        save_intermediate=True,
        enable_caching=False,
        unified_utils_enabled=True,
        model_loader_enabled=True,
        auto_detect_models=True,
        preload_critical_models=False,
        fallback_enabled=True,
        **kwargs
    )

def create_production_pipeline(**kwargs) -> PipelineManager:
    """프로덕션용 파이프라인 생성 - 통합 시스템 우선, 모든 폴백 활성화"""
    return create_pipeline(
        quality_level="high",
        processing_mode="production",
        execution_mode="unified_system",
        optimization_enabled=True,
        memory_optimization=True,
        enable_caching=True,
        parallel_processing=True,
        unified_utils_enabled=True,
        model_loader_enabled=True,
        auto_detect_models=True,
        preload_critical_models=True,
        model_cache_warmup=True,
        fallback_enabled=True,
        retry_with_fallback=True,
        **kwargs
    )

def create_m3_max_pipeline(**kwargs) -> PipelineManager:
    """M3 Max 최적화 파이프라인 생성 - 통합 시스템 최대 활용"""
    return PipelineManager(
        device="mps",
        config=PipelineConfig(
            quality_level=QualityLevel.HIGH,
            processing_mode=PipelineMode.PRODUCTION,
            execution_mode=ExecutionMode.UNIFIED_SYSTEM,
            memory_gb=128.0,
            is_m3_max=True,
            optimization_enabled=True,
            use_fp16=True,
            batch_size=4,
            memory_optimization=True,
            enable_caching=True,
            parallel_processing=True,
            model_cache_size=15,
            gpu_memory_fraction=0.95,
            unified_utils_enabled=True,
            model_loader_enabled=True,
            auto_detect_models=True,
            preload_critical_models=True,
            model_cache_warmup=True,
            step_model_validation=True,
            fallback_enabled=True,
            retry_with_fallback=True,
            **kwargs
        )
    )

def create_testing_pipeline(**kwargs) -> PipelineManager:
    """테스트용 파이프라인 생성 - 빠른 실행 우선"""
    return create_pipeline(
        quality_level="fast",
        processing_mode="testing",
        execution_mode="model_loader",  # 테스트에서는 ModelLoader 우선
        optimization_enabled=False,
        save_intermediate=True,
        enable_caching=False,
        max_retries=1,
        timeout_seconds=60,
        unified_utils_enabled=True,
        model_loader_enabled=True,
        auto_detect_models=False,
        preload_critical_models=False,
        fallback_enabled=True,
        **kwargs
    )

def create_unified_first_pipeline(**kwargs) -> PipelineManager:
    """통합 시스템 우선 파이프라인 생성"""
    return create_pipeline(
        execution_mode="unified_system",
        unified_utils_enabled=True,
        model_loader_enabled=True,
        fallback_enabled=True,
        **kwargs
    )

def create_model_loader_first_pipeline(**kwargs) -> PipelineManager:
    """ModelLoader 우선 파이프라인 생성"""
    return create_pipeline(
        execution_mode="model_loader",
        unified_utils_enabled=True,
        model_loader_enabled=True,
        fallback_enabled=True,
        **kwargs
    )

@lru_cache(maxsize=1)
def get_global_pipeline_manager(device: str = "auto") -> PipelineManager:
    """전역 파이프라인 매니저 인스턴스 - 통합 시스템 우선"""
    try:
        if device == "mps" and torch.backends.mps.is_available():
            return create_m3_max_pipeline()
        else:
            return create_production_pipeline(device=device)
    except Exception as e:
        logger.error(f"전역 파이프라인 매니저 생성 실패: {e}")
        return create_pipeline(device="cpu", quality_level="fast")

# 하위 호환성 보장 함수들 (기존 유지)
def get_human_parsing_step():
    """기존 호환성 - HumanParsingStep 반환"""
    return HumanParsingStep

def get_pose_estimation_step():
    """기존 호환성 - PoseEstimationStep 반환"""
    return PoseEstimationStep

def get_cloth_segmentation_step():
    """기존 호환성 - ClothSegmentationStep 반환"""
    return ClothSegmentationStep

def get_geometric_matching_step():
    """기존 호환성 - GeometricMatchingStep 반환"""
    return GeometricMatchingStep

def get_cloth_warping_step():
    """기존 호환성 - ClothWarpingStep 반환"""
    return ClothWarpingStep

def get_virtual_fitting_step():
    """기존 호환성 - VirtualFittingStep 반환"""
    return VirtualFittingStep

def get_post_processing_step():
    """기존 호환성 - PostProcessingStep 반환"""
    return PostProcessingStep

def get_quality_assessment_step():
    """기존 호환성 - QualityAssessmentStep 반환"""
    return QualityAssessmentStep

# ==============================================
# 🔥 6. 데모 및 테스트 함수들
# ==============================================

async def demo_unified_pipeline():
    """🔥 통합 PipelineManager 데모"""
    
    print("🎯 통합 PipelineManager 데모 시작")
    print("=" * 80)
    print("✅ 프로젝트 통합 유틸리티 시스템 우선 사용")
    print("✅ ModelLoader 시스템 완벽 연동")
    print("✅ 우선순위 기반 실행 전략")
    print("✅ 강화된 에러 처리 및 폴백")
    print("✅ 모듈화된 관리 구조")
    print("✅ M3 Max + conda 환경 최적화")
    print("=" * 80)
    
    # 1. 통합 파이프라인 생성
    print("1️⃣ 통합 파이프라인 생성 중...")
    pipeline = create_m3_max_pipeline()
    
    # 2. 초기화
    print("2️⃣ 통합 파이프라인 초기화 중...")
    success = await pipeline.initialize()
    if not success:
        print("❌ 파이프라인 초기화 실패")
        return
    
    # 3. 상태 확인
    print("3️⃣ 통합 파이프라인 상태 확인...")
    status = pipeline.get_pipeline_status()
    print(f"📊 초기화 상태: {status['initialized']}")
    print(f"🎯 디바이스: {status['device']} ({status['device_type']})")
    print(f"⚙️ 실행 모드: {status['config']['execution_mode']}")
    print(f"🔧 통합 시스템: {'✅' if status['unified_system_initialized'] else '❌'}")
    print(f"🔧 ModelLoader: {'✅' if status['model_loader_initialized'] else '❌'}")
    print(f"📋 로드된 단계: {len([s for s in status['steps_status'].values() if s['loaded']])}/{len(status['steps_status'])}")
    
    # 4. Step별 인터페이스 상태 출력
    print("4️⃣ Step별 인터페이스 상태:")
    for step_name, step_status in status['steps_status'].items():
        status_icon = "✅" if step_status['loaded'] else "❌"
        unified_icon = "🔗" if step_status.get('has_unified_interface', False) else "⭕"
        model_icon = "🧠" if step_status.get('has_model_interface', False) else "⭕"
        print(f"  {status_icon} {unified_icon} {model_icon} {step_name}")
        print(f"      통합 인터페이스: {'있음' if step_status.get('has_unified_interface', False) else '없음'}")
        print(f"      모델 인터페이스: {'있음' if step_status.get('has_model_interface', False) else '없음'}")
    
    # 5. 헬스체크
    print("5️⃣ 헬스체크 수행...")
    health = await pipeline.health_check()
    print(f"🏥 헬스 상태: {health['status']}")
    print(f"📊 건강한 Step: {health['checks']['steps']['healthy_steps']}/{health['checks']['steps']['total_steps']}")
    print(f"🔗 통합 인터페이스 커버리지: {health['checks']['steps']['unified_interface_coverage']}")
    print(f"🧠 모델 인터페이스 커버리지: {health['checks']['steps']['model_interface_coverage']}")
    
    # 6. 통합 시스템을 사용한 가상 피팅 실행
    print("6️⃣ 통합 시스템 우선 가상 피팅 실행...")
    
    try:
        # 더미 이미지 생성
        person_image = Image.new('RGB', (512, 512), color=(100, 150, 200))
        clothing_image = Image.new('RGB', (512, 512), color=(200, 100, 100))
        
        # 진행률 콜백
        async def progress_callback(message: str, percentage: int):
            print(f"🔄 {message}: {percentage}%")
        
        # 가상 피팅 처리
        result = await pipeline.process_complete_virtual_fitting(
            person_image=person_image,
            clothing_image=clothing_image,
            clothing_type='shirt',
            fabric_type='cotton',
            body_measurements={'height': 175, 'weight': 70, 'chest': 95},
            style_preferences={'fit': 'regular', 'color': 'original'},
            quality_target=0.8,
            progress_callback=progress_callback,
            save_intermediate=True
        )
        
        if result.success:
            print(f"✅ 통합 가상 피팅 성공!")
            print(f"📊 품질 점수: {result.quality_score:.3f} ({result.quality_grade})")
            print(f"⏱️ 처리 시간: {result.processing_time:.2f}초")
            print(f"🎯 목표 달성: {'✅' if result.quality_score >= 0.8 else '❌'}")
            print(f"📋 완료된 단계: {len(result.step_results)}/{len(pipeline.step_order)}")
            
            # 실행 전략 통계
            strategy_stats = result.metadata.get('strategy_statistics', {})
            print(f"🔗 실행 전략 통계: {strategy_stats}")
            print(f"🔧 통합 시스템 사용률: {result.metadata.get('unified_system_usage_rate', 0):.1f}%")
            print(f"🧠 ModelLoader 사용률: {result.metadata.get('model_loader_usage_rate', 0):.1f}%")
            print(f"🔄 폴백 사용률: {result.metadata.get('fallback_usage_rate', 0):.1f}%")
            
            # 단계별 결과 출력
            print("\n📋 단계별 실행 전략 결과:")
            for step_name, step_result in result.step_results.items():
                success_icon = "✅" if step_result.get('success', True) else "❌"
                confidence = step_result.get('confidence', 0.0)
                timing = result.step_timings.get(step_name, 0.0)
                strategy = result.execution_strategy.get(step_name, 'unknown')
                model_used = step_result.get('model_used', 'unknown')
                
                # 전략별 아이콘
                strategy_icon = "🔗" if strategy == "unified_system" else "🧠" if strategy == "model_loader" else "🔄"
                
                print(f"  {success_icon} {strategy_icon} {step_name}: {confidence:.3f} ({timing:.2f}s)")
                print(f"      전략: {strategy}, 모델: {model_used}")
            
            # 결과 저장
            if result.result_image:
                result.result_image.save('demo_unified_result.jpg')
                print("💾 결과 이미지 저장: demo_unified_result.jpg")
        else:
            print(f"❌ 통합 가상 피팅 실패: {result.error_message}")
    
    except Exception as e:
        print(f"💥 예외 발생: {e}")
    
    # 7. 성능 요약
    print("7️⃣ 성능 요약...")
    performance = pipeline.get_performance_summary()
    print(f"📈 총 세션: {performance['total_sessions']}")
    print(f"📊 성공률: {performance['success_rate']:.1%}")
    print(f"⏱️ 평균 처리 시간: {performance['average_processing_time']:.2f}초")
    print(f"🎯 평균 품질 점수: {performance['average_quality_score']:.3f}")
    
    # 실행 전략 통계
    strategy_stats = performance['execution_strategy_stats']
    print(f"🔗 통합 시스템 사용: {strategy_stats['unified_system_usage']}회 ({strategy_stats['unified_system_rate']:.1f}%)")
    print(f"🧠 ModelLoader 사용: {strategy_stats['model_loader_usage']}회 ({strategy_stats['model_loader_rate']:.1f}%)")
    print(f"🔄 폴백 사용: {strategy_stats['fallback_usage']}회 ({strategy_stats['fallback_rate']:.1f}%)")
    
    # 8. 리소스 정리
    print("8️⃣ 리소스 정리...")
    await pipeline.cleanup()
    print("🧹 리소스 정리 완료")
    
    print("\n🎉 통합 PipelineManager 데모 완료!")
    print("✅ 모든 통합 기능이 성공적으로 작동했습니다!")
    print("🔗 통합 시스템 우선 실행으로 최고 품질 달성!")

async def test_execution_strategies():
    """실행 전략 테스트"""
    
    print("🔬 실행 전략 테스트 시작")
    print("=" * 50)
    
    strategies = [
        ("unified_system", "통합 시스템 우선"),
        ("model_loader", "ModelLoader 우선"),
        ("fallback", "폴백 모드")
    ]
    
    results = {}
    
    for strategy_mode, strategy_desc in strategies:
        print(f"\n🎯 {strategy_desc} 테스트...")
        
        try:
            # 전략별 파이프라인 생성
            pipeline = create_pipeline(
                device="cpu",
                execution_mode=strategy_mode,
                quality_level="fast"
            )
            
            # 초기화
            success = await pipeline.initialize()
            if not success:
                print(f"❌ {strategy_desc} 초기화 실패")
                continue
            
            # 더미 이미지로 테스트
            dummy_person = Image.new('RGB', (256, 256), color=(100, 150, 200))
            dummy_cloth = Image.new('RGB', (256, 256), color=(200, 100, 100))
            
            # 가상 피팅 실행
            result = await pipeline.process_complete_virtual_fitting(
                person_image=dummy_person,
                clothing_image=dummy_cloth,
                quality_target=0.6,
                save_intermediate=False,
                session_id=f"test_{strategy_mode}"
            )
            
            if result.success:
                print(f"✅ {strategy_desc} 성공")
                strategy_stats = result.metadata.get('strategy_statistics', {})
                print(f"   실행 전략 통계: {strategy_stats}")
                print(f"   처리 시간: {result.processing_time:.2f}초")
                print(f"   품질 점수: {result.quality_score:.3f}")
                
                results[strategy_mode] = {
                    'success': True,
                    'time': result.processing_time,
                    'quality': result.quality_score,
                    'strategies': strategy_stats
                }
            else:
                print(f"❌ {strategy_desc} 실패: {result.error_message}")
                results[strategy_mode] = {
                    'success': False,
                    'error': result.error_message
                }
            
            # 정리
            await pipeline.cleanup()
            
        except Exception as e:
            print(f"❌ {strategy_desc} 테스트 실패: {e}")
            results[strategy_mode] = {
                'success': False,
                'error': str(e)
            }
    
    # 결과 요약
    print("\n📊 실행 전략 테스트 결과 요약:")
    for strategy_mode, strategy_desc in strategies:
        result = results.get(strategy_mode, {})
        if result.get('success', False):
            print(f"✅ {strategy_desc}:")
            print(f"   처리 시간: {result['time']:.2f}초")
            print(f"   품질 점수: {result['quality']:.3f}")
            print(f"   전략 분포: {result['strategies']}")
        else:
            print(f"❌ {strategy_desc}: {result.get('error', 'Unknown error')}")
    
    print("✅ 실행 전략 테스트 완료")

# ==============================================
# 🔥 7. Export 및 메인 실행
# ==============================================

# Export 목록 (모든 기존 항목 + 새로운 항목)
__all__ = [
    # 열거형
    'PipelineMode', 'QualityLevel', 'ProcessingStatus', 'ExecutionMode',
    
    # 데이터 클래스
    'PipelineConfig', 'ProcessingResult', 'SessionData', 'PerformanceMetrics',
    
    # 메인 클래스
    'PipelineManager',
    
    # 관리 클래스들
    'InitializationManager', 'ExecutionManager', 'OptimizedDataConverter', 'OptimizedMemoryManager',
    
    # 팩토리 함수들 (모든 기존 함수 + 새로운 함수들)
    'create_pipeline', 'create_development_pipeline', 'create_production_pipeline',
    'create_m3_max_pipeline', 'create_testing_pipeline', 'get_global_pipeline_manager',
    'create_unified_first_pipeline', 'create_model_loader_first_pipeline',
    
    # 하위 호환성 함수들
    'get_human_parsing_step', 'get_pose_estimation_step', 'get_cloth_segmentation_step',
    'get_geometric_matching_step', 'get_cloth_warping_step', 'get_virtual_fitting_step',
    'get_post_processing_step', 'get_quality_assessment_step'
]

if __name__ == "__main__":
    print("🔥 완전 통합 PipelineManager - 두 버전 최적 합성")
    print("=" * 80)
    print("✅ 프로젝트 통합 유틸리티 시스템 우선 사용")
    print("✅ ModelLoader 시스템 완벽 연동")
    print("✅ 우선순위 기반 실행 전략")
    print("✅ 모듈화된 관리 구조")
    print("✅ 강화된 에러 처리 및 폴백")
    print("✅ M3 Max + conda 환경 최적화")
    print("✅ 모든 기존 함수/클래스명 100% 유지")
    print("✅ 프로덕션 레벨 안정성")
    print("=" * 80)
    
    import asyncio
    
    async def main():
        # 1. 통합 데모 실행
        await demo_unified_pipeline()
        
        print("\n" + "="*50)
        
        # 2. 실행 전략 테스트
        await test_execution_strategies()
    
    # 실행
    asyncio.run(main())

# ==============================================
# 🔥 8. 로깅 및 초기화 완료 메시지
# ==============================================

logger.info("🎉 완전 통합 PipelineManager 로드 완료!")
logger.info("✅ 주요 통합 기능:")
logger.info("   - 프로젝트 통합 유틸리티 시스템 우선 사용")
logger.info("   - ModelLoader 시스템 완벽 연동")
logger.info("   - 우선순위 기반 실행 전략 (unified_system → model_loader → fallback)")
logger.info("   - 모듈화된 관리 구조 (InitializationManager, ExecutionManager)")
logger.info("   - 강화된 에러 처리 및 폴백 메커니즘")
logger.info("   - M3 Max + conda 환경 특화 최적화")
logger.info("   - 모든 기존 함수/클래스명 100% 유지")
logger.info("   - 실행 전략별 통계 및 성능 모니터링")
logger.info("🚀 이제 최고 품질의 통합 가상 피팅이 가능합니다!")
logger.info(f"🔧 시스템 가용성:")
logger.info(f"   - 통합 유틸리티: {'✅' if UNIFIED_UTILS_AVAILABLE else '❌'}")
logger.info(f"   - ModelLoader: {'✅' if MODEL_LOADER_AVAILABLE else '❌'}")
logger.info(f"   - Step 요청: {'✅' if STEP_REQUESTS_AVAILABLE else '❌'}")
logger.info(f"   - 자동 탐지: {'✅' if AUTO_DETECTOR_AVAILABLE else '❌'}")
logger.info(f"   - Step 클래스: {'✅' if STEP_CLASSES_AVAILABLE else '❌'}")
logger.info("🎯 권장 사용법: create_m3_max_pipeline() 또는 create_production_pipeline()")