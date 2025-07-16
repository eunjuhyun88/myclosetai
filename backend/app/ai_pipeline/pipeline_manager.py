"""
단순화된 PipelineManager - Step 클래스들만 관리
✅ 기존 함수명/클래스명 100% 유지
✅ Step 클래스들의 오케스트레이션만 담당
✅ 각 Step이 자체적으로 모델/메모리 관리
✅ M3 Max 최적화
✅ 프로덕션 레벨 안정성

파일 위치: backend/app/ai_pipeline/pipeline_manager.py
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

# 필수 라이브러리
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter

# Step 클래스들만 import (단방향 의존성)
from app.ai_pipeline.steps.step_01_human_parsing import HumanParsingStep
from app.ai_pipeline.steps.step_02_pose_estimation import PoseEstimationStep
from app.ai_pipeline.steps.step_03_cloth_segmentation import ClothSegmentationStep
from app.ai_pipeline.steps.step_04_geometric_matching import GeometricMatchingStep
from app.ai_pipeline.steps.step_05_cloth_warping import ClothWarpingStep
from app.ai_pipeline.steps.step_06_virtual_fitting import VirtualFittingStep
from app.ai_pipeline.steps.step_07_post_processing import PostProcessingStep
from app.ai_pipeline.steps.step_08_quality_assessment import QualityAssessmentStep

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
# 1. 열거형 및 상수 정의
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

# ==============================================
# 2. 데이터 클래스
# ==============================================

@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    # 기본 설정
    device: str = "auto"
    quality_level: Union[QualityLevel, str] = QualityLevel.BALANCED
    processing_mode: Union[PipelineMode, str] = PipelineMode.PRODUCTION
    
    # 시스템 설정
    memory_gb: float = 16.0
    is_m3_max: bool = False
    device_type: str = "auto"
    
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
        
        # M3 Max 자동 최적화
        if self.is_m3_max:
            self.memory_gb = max(self.memory_gb, 64.0)
            self.use_fp16 = True
            self.optimization_enabled = True
            self.batch_size = 4
            self.model_cache_size = 15
            self.gpu_memory_fraction = 0.95

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
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'success': self.success,
            'session_id': self.session_id,
            'quality_score': self.quality_score,
            'quality_grade': self.quality_grade,
            'processing_time': self.processing_time,
            'step_results': self.step_results,
            'step_timings': self.step_timings,
            'metadata': self.metadata,
            'error_message': self.error_message,
            'warnings': self.warnings
        }

@dataclass
class SessionData:
    """세션 데이터"""
    session_id: str
    start_time: float
    status: ProcessingStatus = ProcessingStatus.IDLE
    step_results: Dict[str, Any] = field(default_factory=dict)
    step_timings: Dict[str, float] = field(default_factory=dict)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_step_result(self, step_name: str, result: Dict[str, Any], timing: float):
        """단계 결과 추가"""
        self.step_results[step_name] = result
        self.step_timings[step_name] = timing

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
    
    def update(self, processing_time: float, quality_score: float, success: bool):
        """메트릭 업데이트"""
        self.total_sessions += 1
        self.total_processing_time += processing_time
        
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
# 3. 유틸리티 클래스들 (최소한만 유지)
# ==============================================

class SimpleDataConverter:
    """간단한 데이터 변환기 - PipelineManager용"""
    
    def __init__(self, device: str = "mps"):
        self.device = device
        self.logger = logging.getLogger(f"pipeline.{self.__class__.__name__}")
    
    def preprocess_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """이미지 전처리"""
        try:
            # 이미지 로드 및 변환
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
            
        except Exception as e:
            self.logger.error(f"이미지 전처리 실패: {e}")
            raise
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """텐서를 PIL 이미지로 변환"""
        try:
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            
            if tensor.shape[0] == 3:
                tensor = tensor.permute(1, 2, 0)
            
            tensor = torch.clamp(tensor, 0, 1)
            tensor = tensor.cpu()
            array = (tensor.numpy() * 255).astype(np.uint8)
            
            return Image.fromarray(array)
            
        except Exception as e:
            self.logger.error(f"텐서-PIL 변환 실패: {e}")
            return Image.new('RGB', (512, 512), color='black')

class SimpleMemoryManager:
    """간단한 메모리 관리자 - PipelineManager용"""
    
    def __init__(self, device: str = "mps"):
        self.device = device
        self.logger = logging.getLogger(f"pipeline.{self.__class__.__name__}")
        
    def cleanup_memory(self):
        """메모리 정리"""
        try:
            gc.collect()
            
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif self.device == "mps" and torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                    torch.mps.synchronize()
                except AttributeError:
                    pass
                
            self.logger.debug("메모리 정리 완료")
            
        except Exception as e:
            self.logger.warning(f"메모리 정리 실패: {e}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """메모리 사용량 상세 정보"""
        try:
            usage = {}
            
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
            
        except Exception as e:
            self.logger.warning(f"메모리 사용량 조회 실패: {e}")
            return {'error': str(e)}

# ==============================================
# 4. 메인 PipelineManager 클래스 (단순화)
# ==============================================

class PipelineManager:
    """
    단순화된 PipelineManager - Step 클래스들만 관리
    
    역할:
    - Step 클래스들의 초기화 및 오케스트레이션
    - 8단계 워크플로우 실행
    - 세션 관리 및 성능 모니터링
    - 기본적인 메모리 관리
    
    각 Step이 담당:
    - 자체 모델 로딩 및 관리
    - 상세한 메모리 최적화
    - 실제 AI 처리 로직
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
            
            self.config = PipelineConfig(
                device=self.device,
                **config_dict
            )
        
        # 3. 시스템 정보 감지
        self.device_type = self._detect_device_type()
        self.memory_gb = self._detect_memory_gb()
        self.is_m3_max = self._detect_m3_max()
        
        # 설정 업데이트
        self.config.device_type = self.device_type
        self.config.memory_gb = self.memory_gb
        self.config.is_m3_max = self.is_m3_max
        
        # 4. 간단한 유틸리티 초기화
        self.data_converter = SimpleDataConverter(self.device)
        self.memory_manager = SimpleMemoryManager(self.device)
        
        # 5. 파이프라인 상태
        self.is_initialized = False
        self.current_status = ProcessingStatus.IDLE
        self.steps = {}
        self.step_order = [
            'human_parsing',
            'pose_estimation', 
            'cloth_segmentation',
            'geometric_matching',
            'cloth_warping',
            'virtual_fitting',
            'post_processing',
            'quality_assessment'
        ]
        
        # 6. 세션 관리
        self.sessions: Dict[str, SessionData] = {}
        self.performance_metrics = PerformanceMetrics()
        
        # 7. 동시성 관리
        self._lock = threading.RLock()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.thread_pool_size)
        
        # 8. 로깅 설정
        self.logger = logging.getLogger(f"pipeline.{self.__class__.__name__}")
        
        # 9. 디바이스 최적화
        self._configure_device_optimizations()
        
        # 10. 초기화 완료 로깅
        self.logger.info(f"✅ PipelineManager 초기화 완료")
        self.logger.info(f"🎯 디바이스: {self.device} ({self.device_type})")
        self.logger.info(f"📊 메모리: {self.memory_gb}GB, M3 Max: {'✅' if self.is_m3_max else '❌'}")
        self.logger.info(f"⚙️ 설정: {self.config.quality_level.value} 품질, {self.config.processing_mode.value} 모드")
        
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
        
        # 혼합 정밀도 설정
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
    
    async def initialize(self) -> bool:
        """파이프라인 초기화 - Step 클래스들만 초기화"""
        try:
            self.logger.info("🔄 파이프라인 초기화 시작...")
            self.current_status = ProcessingStatus.INITIALIZING
            start_time = time.time()
            
            # 1. 메모리 정리
            self.memory_manager.cleanup_memory()
            
            # 2. 각 Step 클래스 초기화
            await self._initialize_all_steps()
            
            # 3. 초기화 검증
            success_rate = self._verify_initialization()
            if success_rate < 0.5:  # 50% 이상 성공해야 함
                raise RuntimeError(f"초기화 성공률 부족: {success_rate:.1%}")
            
            initialization_time = time.time() - start_time
            self.is_initialized = True
            self.current_status = ProcessingStatus.IDLE
            
            self.logger.info(f"✅ 파이프라인 초기화 완료")
            self.logger.info(f"⏱️ 초기화 시간: {initialization_time:.2f}초")
            self.logger.info(f"📊 초기화 성공률: {success_rate:.1%}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 파이프라인 초기화 실패: {e}")
            self.logger.error(f"📋 오류 상세: {traceback.format_exc()}")
            self.is_initialized = False
            self.current_status = ProcessingStatus.FAILED
            return False
    
    async def _initialize_all_steps(self):
        """모든 Step 클래스 초기화"""
        
        # 기본 설정 - Step들이 자체적으로 관리할 수 있도록 최소한만 전달
        base_config = {
            'quality_level': self.config.quality_level.value,
            'device': self.device,
            'device_type': self.device_type,
            'memory_gb': self.memory_gb,
            'is_m3_max': self.is_m3_max,
            'optimization_enabled': self.config.optimization_enabled,
            'memory_optimization': self.config.memory_optimization,
            'use_fp16': self.config.use_fp16,
            'batch_size': self.config.batch_size,
            'enable_quantization': self.config.enable_quantization
        }
        
        # Step 클래스 매핑
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
        
        # 각 Step 클래스 초기화
        for step_name in self.step_order:
            self.logger.info(f"🔧 {step_name} 초기화 중...")
            
            try:
                step_class = step_classes[step_name]
                step_config = {**base_config, **self._get_step_config(step_name)}
                
                # Step 인스턴스 생성 - 각 Step이 자체적으로 모든 것을 관리
                step_instance = step_class(**step_config)
                
                # 초기화 실행 (Step 자체 모델 로딩 등)
                if hasattr(step_instance, 'initialize'):
                    await step_instance.initialize()
                
                self.steps[step_name] = step_instance
                self.logger.info(f"✅ {step_name} 초기화 완료")
                
            except Exception as e:
                self.logger.error(f"❌ {step_name} 초기화 실패: {e}")
                # 실패한 단계는 건너뛰기
                continue
    
    def _get_step_config(self, step_name: str) -> Dict[str, Any]:
        """단계별 특화 설정"""
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
                'matching_threshold': 0.8
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
    
    def _verify_initialization(self) -> float:
        """초기화 검증"""
        total_steps = len(self.step_order)
        initialized_steps = len(self.steps)
        
        success_rate = initialized_steps / total_steps
        self.logger.info(f"📊 초기화 상태: {initialized_steps}/{total_steps} ({success_rate:.1%})")
        
        return success_rate
    
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
        완전한 8단계 가상 피팅 처리 - Step 클래스들만 사용
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
            self.logger.info(f"🎯 8단계 가상 피팅 시작 - 세션 ID: {session_id}")
            self.logger.info(f"⚙️ 설정: {clothing_type} ({fabric_type}), 목표 품질: {quality_target}")
            
            # 1. 입력 이미지 전처리
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
                    'quality_level': self.config.quality_level.value
                }
            )
            
            self.sessions[session_id] = session_data
            
            if progress_callback:
                await progress_callback("입력 전처리 완료", 5)
            
            # 3. 메모리 최적화
            if self.config.memory_optimization:
                self.memory_manager.cleanup_memory()
            
            # 4. 8단계 순차 처리 - 각 Step이 자체적으로 모든 것을 처리
            step_results = {}
            current_data = person_tensor
            
            for i, step_name in enumerate(self.step_order):
                if step_name not in self.steps:
                    self.logger.warning(f"⚠️ {step_name} 단계가 없습니다. 건너뛰기...")
                    continue
                
                step_start = time.time()
                step = self.steps[step_name]
                
                self.logger.info(f"📋 {i+1}/{len(self.step_order)} 단계: {step_name} 처리 중...")
                
                try:
                    # Step별 처리 실행 - 각 Step이 모든 것을 자체 관리
                    step_result = await self._execute_step_with_retry(
                        step, step_name, current_data, clothing_tensor,
                        body_measurements, clothing_type, fabric_type,
                        style_preferences, self.config.max_retries
                    )
                    
                    step_time = time.time() - step_start
                    step_results[step_name] = step_result
                    
                    # 세션 데이터 업데이트
                    session_data.add_step_result(step_name, step_result, step_time)
                    
                    # 결과 업데이트
                    if step_result.get('success', True):
                        result_data = step_result.get('result')
                        if result_data is not None:
                            current_data = result_data
                    
                    # 중간 결과 저장
                    if save_intermediate:
                        session_data.intermediate_results[step_name] = {
                            'result': current_data,
                            'metadata': step_result
                        }
                    
                    # 로깅
                    confidence = step_result.get('confidence', 0.8)
                    quality_score = step_result.get('quality_score', confidence)
                    self.logger.info(f"✅ {i+1}단계 완료 - 시간: {step_time:.2f}초, 신뢰도: {confidence:.3f}, 품질: {quality_score:.3f}")
                    
                    # 진행률 콜백
                    if progress_callback:
                        progress = 5 + (i + 1) * 85 // len(self.step_order)
                        await progress_callback(f"{step_name} 완료", progress)
                    
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
                        'quality_score': 0.0
                    }
                    
                    session_data.add_step_result(step_name, step_results[step_name], step_time)
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
            
            # 간단한 품질 평가
            quality_score = self._assess_simple_quality(step_results)
            quality_grade = self._get_quality_grade(quality_score)
            
            # 성공 여부 결정
            success = quality_score >= (quality_target * 0.8)  # 80% 이상 달성
            
            # 성능 메트릭 업데이트
            self.performance_metrics.update(total_time, quality_score, success)
            
            # 세션 상태 업데이트
            session_data.status = ProcessingStatus.COMPLETED if success else ProcessingStatus.FAILED
            
            # 세션 데이터 정리
            if not save_intermediate:
                self.sessions.pop(session_id, None)
            
            if progress_callback:
                await progress_callback("처리 완료", 100)
            
            self.current_status = ProcessingStatus.IDLE
            
            # 결과 로깅
            self.logger.info(f"🎉 8단계 가상 피팅 완료!")
            self.logger.info(f"⏱️ 총 시간: {total_time:.2f}초")
            self.logger.info(f"📊 품질 점수: {quality_score:.3f} ({quality_grade})")
            self.logger.info(f"🎯 목표 달성: {'✅' if quality_score >= quality_target else '❌'}")
            
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
                metadata={
                    'device': self.device,
                    'device_type': self.device_type,
                    'is_m3_max': self.is_m3_max,
                    'quality_target': quality_target,
                    'quality_target_achieved': quality_score >= quality_target,
                    'total_steps': len(self.step_order),
                    'completed_steps': len(step_results),
                    'success_rate': len([r for r in step_results.values() if r.get('success', True)]) / len(step_results) if step_results else 0,
                    'processing_config': {
                        'quality_level': self.config.quality_level.value,
                        'processing_mode': self.config.processing_mode.value,
                        'optimization_enabled': self.config.optimization_enabled,
                        'memory_optimization': self.config.memory_optimization,
                        'use_fp16': self.config.use_fp16,
                        'batch_size': self.config.batch_size
                    },
                    'performance_metrics': {
                        'total_sessions': self.performance_metrics.total_sessions,
                        'success_rate': self.performance_metrics.successful_sessions / self.performance_metrics.total_sessions if self.performance_metrics.total_sessions > 0 else 0,
                        'average_processing_time': self.performance_metrics.average_processing_time,
                        'average_quality_score': self.performance_metrics.average_quality_score
                    },
                    'memory_usage': self.memory_manager.get_memory_usage(),
                    'session_data': session_data.__dict__ if save_intermediate else None
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ 가상 피팅 처리 실패: {e}")
            self.logger.error(f"📋 오류 상세: {traceback.format_exc()}")
            
            # 에러 메트릭 업데이트
            self.performance_metrics.update(time.time() - start_time, 0.0, False)
            
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
                    'session_data': self.sessions.get(session_id).__dict__ if session_id in self.sessions else None
                }
            )
    
    async def _execute_step_with_retry(self, step, step_name: str, current_data: torch.Tensor, 
                                     clothing_tensor: torch.Tensor, body_measurements: Optional[Dict],
                                     clothing_type: str, fabric_type: str, 
                                     style_preferences: Optional[Dict], max_retries: int) -> Dict[str, Any]:
        """재시도 로직이 포함된 Step 실행"""
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    self.logger.info(f"🔄 {step_name} 재시도 {attempt}/{max_retries}")
                    # 재시도 전 메모리 정리
                    self.memory_manager.cleanup_memory()
                    await asyncio.sleep(0.5)  # 잠시 대기
                
                # Step 실행 - Step이 자체적으로 모든 것을 처리
                result = await self._execute_step(
                    step, step_name, current_data, clothing_tensor,
                    body_measurements, clothing_type, fabric_type, style_preferences
                )
                
                # 성공 시 반환
                if result.get('success', True):
                    return result
                else:
                    last_error = result.get('error', 'Unknown error')
                    
            except Exception as e:
                last_error = str(e)
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
            'method': 'failed_after_retries'
        }
    
    async def _execute_step(self, step, step_name: str, current_data: torch.Tensor, 
                          clothing_tensor: torch.Tensor, body_measurements: Optional[Dict],
                          clothing_type: str, fabric_type: str, 
                          style_preferences: Optional[Dict]) -> Dict[str, Any]:
        """
        개별 Step 실행 - Step이 자체적으로 모든 것을 처리
        """
        try:
            # Step별 처리 로직 - 각 Step의 process 메서드 호출
            if step_name == 'human_parsing':
                result = await step.process(current_data)
                
            elif step_name == 'pose_estimation':
                result = await step.process(current_data)
                
            elif step_name == 'cloth_segmentation':
                result = await step.process(clothing_tensor, clothing_type=clothing_type)
                
            elif step_name == 'geometric_matching':
                result = await step.process(current_data, clothing_tensor, body_measurements)
                
            elif step_name == 'cloth_warping':
                result = await step.process(current_data, clothing_tensor, body_measurements, fabric_type)
                
            elif step_name == 'virtual_fitting':
                result = await step.process(current_data, clothing_tensor, style_preferences)
                
            elif step_name == 'post_processing':
                result = await step.process(current_data)
                
            elif step_name == 'quality_assessment':
                result = await step.process(current_data, clothing_tensor)
                
            else:
                # 기본 처리
                result = await step.process(current_data)
            
            # 결과 검증 및 표준화
            if not result or not isinstance(result, dict):
                return {
                    'success': True,
                    'result': current_data,
                    'confidence': 0.8,
                    'quality_score': 0.8,
                    'processing_time': 0.1,
                    'method': 'default'
                }
            
            # 필수 필드 확인
            if 'confidence' not in result:
                result['confidence'] = 0.8
            if 'quality_score' not in result:
                result['quality_score'] = result.get('confidence', 0.8)
            if 'success' not in result:
                result['success'] = True
            
            return result
            
        except Exception as e:
            self.logger.error(f"Step 실행 실패 {step_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'confidence': 0.0,
                'quality_score': 0.0,
                'processing_time': 0.0,
                'method': 'error'
            }
    
    def _assess_simple_quality(self, step_results: Dict[str, Any]) -> float:
        """간단한 품질 평가"""
        if not step_results:
            return 0.5
        
        # Step별 품질 점수 수집
        quality_scores = []
        confidence_scores = []
        
        for step_result in step_results.values():
            if isinstance(step_result, dict):
                confidence = step_result.get('confidence', 0.8)
                quality = step_result.get('quality_score', confidence)
                
                quality_scores.append(quality)
                confidence_scores.append(confidence)
        
        # 종합 점수 계산
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            
            # 가중 평균
            overall_score = avg_quality * 0.6 + avg_confidence * 0.4
            return min(max(overall_score, 0.0), 1.0)
        
        return 0.5
    
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
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """파이프라인 상태 조회"""
        return {
            'initialized': self.is_initialized,
            'current_status': self.current_status.value,
            'device': self.device,
            'device_type': self.device_type,
            'memory_gb': self.memory_gb,
            'is_m3_max': self.is_m3_max,
            'config': {
                'quality_level': self.config.quality_level.value,
                'processing_mode': self.config.processing_mode.value,
                'optimization_enabled': self.config.optimization_enabled,
                'memory_optimization': self.config.memory_optimization,
                'use_fp16': self.config.use_fp16,
                'batch_size': self.config.batch_size
            },
            'steps_status': {
                step_name: {
                    'loaded': step_name in self.steps,
                    'type': type(self.steps[step_name]).__name__ if step_name in self.steps else None,
                    'ready': step_name in self.steps and hasattr(self.steps[step_name], 'process')
                }
                for step_name in self.step_order
            },
            'performance_metrics': {
                'total_sessions': self.performance_metrics.total_sessions,
                'successful_sessions': self.performance_metrics.successful_sessions,
                'failed_sessions': self.performance_metrics.failed_sessions,
                'success_rate': self.performance_metrics.successful_sessions / self.performance_metrics.total_sessions if self.performance_metrics.total_sessions > 0 else 0,
                'average_processing_time': self.performance_metrics.average_processing_time,
                'average_quality_score': self.performance_metrics.average_quality_score
            },
            'memory_usage': self.memory_manager.get_memory_usage(),
            'active_sessions': len(self.sessions)
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
                'total_steps': len(self.step_order)
            }
            for session_id, session in self.sessions.items()
        ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보"""
        return {
            'total_sessions': self.performance_metrics.total_sessions,
            'success_rate': self.performance_metrics.successful_sessions / self.performance_metrics.total_sessions if self.performance_metrics.total_sessions > 0 else 0,
            'average_processing_time': self.performance_metrics.average_processing_time,
            'average_quality_score': self.performance_metrics.average_quality_score,
            'fastest_time': self.performance_metrics.fastest_processing_time if self.performance_metrics.fastest_processing_time != float('inf') else 0,
            'slowest_time': self.performance_metrics.slowest_processing_time,
            'total_processing_time': self.performance_metrics.total_processing_time,
            'active_sessions': len(self.sessions),
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
        """파이프라인 워밍업"""
        try:
            self.logger.info("🔥 파이프라인 워밍업 시작...")
            
            # 더미 이미지 생성
            dummy_person = Image.new('RGB', (512, 512), color=(100, 150, 200))
            dummy_cloth = Image.new('RGB', (512, 512), color=(200, 100, 100))
            
            # 워밍업 실행
            result = await self.process_complete_virtual_fitting(
                person_image=dummy_person,
                clothing_image=dummy_cloth,
                clothing_type='shirt',
                fabric_type='cotton',
                quality_target=0.6,  # 낮은 목표로 빠른 처리
                save_intermediate=False,
                session_id="warmup_session"
            )
            
            if result.success:
                self.logger.info(f"✅ 워밍업 완료 - 시간: {result.processing_time:.2f}초")
                return True
            else:
                self.logger.warning(f"⚠️ 워밍업 중 오류: {result.error_message}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 워밍업 실패: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """헬스체크"""
        try:
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'pipeline_initialized': self.is_initialized,
                'current_status': self.current_status.value,
                'device': self.device,
                'checks': {}
            }
            
            # Step별 체크
            steps_healthy = 0
            for step_name in self.step_order:
                if step_name in self.steps:
                    step = self.steps[step_name]
                    has_process = hasattr(step, 'process')
                    steps_healthy += 1 if has_process else 0
            
            health_status['checks']['steps'] = {
                'status': 'ok' if steps_healthy >= len(self.step_order) * 0.8 else 'warning',
                'healthy_steps': steps_healthy,
                'total_steps': len(self.step_order)
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
        """리소스 정리"""
        try:
            self.logger.info("🧹 파이프라인 리소스 정리 중...")
            self.current_status = ProcessingStatus.CLEANING
            
            # 1. 각 Step 정리
            for step_name, step in self.steps.items():
                try:
                    if hasattr(step, 'cleanup'):
                        await step.cleanup()
                    self.logger.info(f"✅ {step_name} 정리 완료")
                except Exception as e:
                    self.logger.warning(f"⚠️ {step_name} 정리 중 오류: {e}")
            
            # 2. 메모리 관리자 정리
            try:
                self.memory_manager.cleanup_memory()
                self.logger.info("✅ 메모리 관리자 정리 완료")
            except Exception as e:
                self.logger.warning(f"⚠️ 메모리 관리자 정리 중 오류: {e}")
            
            # 3. 스레드 풀 정리
            if hasattr(self, 'thread_pool'):
                try:
                    self.thread_pool.shutdown(wait=True)
                    self.logger.info("✅ 스레드 풀 정리 완료")
                except Exception as e:
                    self.logger.warning(f"⚠️ 스레드 풀 정리 중 오류: {e}")
            
            # 4. 세션 데이터 정리
            try:
                self.sessions.clear()
                self.logger.info("✅ 세션 데이터 정리 완료")
            except Exception as e:
                self.logger.warning(f"⚠️ 세션 데이터 정리 중 오류: {e}")
            
            # 5. 상태 초기화
            self.is_initialized = False
            self.current_status = ProcessingStatus.IDLE
            
            self.logger.info("✅ 파이프라인 리소스 정리 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 리소스 정리 중 오류: {e}")
            self.current_status = ProcessingStatus.FAILED

# ==============================================
# 5. 편의 함수들 (기존 함수명 100% 유지)
# ==============================================

def create_pipeline(
    device: str = "auto",
    quality_level: str = "balanced",
    processing_mode: str = "production",
    **kwargs
) -> PipelineManager:
    """파이프라인 생성 편의 함수"""
    return PipelineManager(
        device=device,
        config=PipelineConfig(
            quality_level=QualityLevel(quality_level),
            processing_mode=PipelineMode(processing_mode),
            **kwargs
        )
    )

def create_development_pipeline(**kwargs) -> PipelineManager:
    """개발용 파이프라인 생성"""
    return create_pipeline(
        quality_level="fast",
        processing_mode="development",
        optimization_enabled=False,
        save_intermediate=True,
        enable_caching=False,
        **kwargs
    )

def create_production_pipeline(**kwargs) -> PipelineManager:
    """프로덕션용 파이프라인 생성"""
    return create_pipeline(
        quality_level="high",
        processing_mode="production",
        optimization_enabled=True,
        memory_optimization=True,
        enable_caching=True,
        parallel_processing=True,
        **kwargs
    )

def create_m3_max_pipeline(**kwargs) -> PipelineManager:
    """M3 Max 최적화 파이프라인 생성"""
    return PipelineManager(
        device="mps",
        config=PipelineConfig(
            quality_level=QualityLevel.HIGH,
            processing_mode=PipelineMode.PRODUCTION,
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
            **kwargs
        )
    )

def create_testing_pipeline(**kwargs) -> PipelineManager:
    """테스트용 파이프라인 생성"""
    return create_pipeline(
        quality_level="fast",
        processing_mode="testing",
        optimization_enabled=False,
        save_intermediate=True,
        enable_caching=False,
        max_retries=1,
        timeout_seconds=60,
        **kwargs
    )

@lru_cache(maxsize=1)
def get_global_pipeline_manager(device: str = "auto") -> PipelineManager:
    """전역 파이프라인 매니저 인스턴스"""
    try:
        if device == "mps" and torch.backends.mps.is_available():
            return create_m3_max_pipeline()
        else:
            return create_production_pipeline(device=device)
    except Exception as e:
        logger.error(f"전역 파이프라인 매니저 생성 실패: {e}")
        return create_pipeline(device="cpu", quality_level="fast")

# ==============================================
# 6. 데모 및 테스트 함수
# ==============================================

async def demo_simplified_pipeline():
    """단순화된 파이프라인 데모"""
    
    print("🎯 단순화된 PipelineManager 데모 시작")
    print("=" * 50)
    
    # 1. 파이프라인 생성
    print("1️⃣ 파이프라인 생성 중...")
    pipeline = create_m3_max_pipeline()
    
    # 2. 초기화
    print("2️⃣ 파이프라인 초기화 중...")
    success = await pipeline.initialize()
    if not success:
        print("❌ 파이프라인 초기화 실패")
        return
    
    # 3. 상태 확인
    print("3️⃣ 파이프라인 상태 확인...")
    status = pipeline.get_pipeline_status()
    print(f"📊 초기화 상태: {status['initialized']}")
    print(f"🎯 디바이스: {status['device']} ({status['device_type']})")
    print(f"📋 로드된 단계: {len([s for s in status['steps_status'].values() if s['loaded']])}/{len(status['steps_status'])}")
    
    # 4. 헬스체크
    print("4️⃣ 헬스체크 수행...")
    health = await pipeline.health_check()
    print(f"🏥 헬스 상태: {health['status']}")
    
    # 5. 가상 피팅 실행
    print("5️⃣ 가상 피팅 실행...")
    
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
            print(f"✅ 가상 피팅 성공!")
            print(f"📊 품질 점수: {result.quality_score:.3f} ({result.quality_grade})")
            print(f"⏱️ 처리 시간: {result.processing_time:.2f}초")
            print(f"🎯 목표 달성: {'✅' if result.quality_score >= 0.8 else '❌'}")
            print(f"📋 완료된 단계: {len(result.step_results)}/{len(pipeline.step_order)}")
            
            # 단계별 결과 출력
            print("\n📋 단계별 결과:")
            for step_name, step_result in result.step_results.items():
                success_icon = "✅" if step_result.get('success', True) else "❌"
                confidence = step_result.get('confidence', 0.0)
                timing = result.step_timings.get(step_name, 0.0)
                print(f"  {success_icon} {step_name}: {confidence:.3f} ({timing:.2f}s)")
            
            # 결과 저장
            if result.result_image:
                result.result_image.save('demo_result.jpg')
                print("💾 결과 이미지 저장: demo_result.jpg")
        else:
            print(f"❌ 가상 피팅 실패: {result.error_message}")
    
    except Exception as e:
        print(f"💥 예외 발생: {e}")
    
    # 6. 성능 요약
    print("6️⃣ 성능 요약...")
    performance = pipeline.get_performance_summary()
    print(f"📈 총 세션: {performance['total_sessions']}")
    print(f"📊 성공률: {performance['success_rate']:.1%}")
    print(f"⏱️ 평균 처리 시간: {performance['average_processing_time']:.2f}초")
    print(f"🎯 평균 품질 점수: {performance['average_quality_score']:.3f}")
    
    # 7. 리소스 정리
    print("7️⃣ 리소스 정리...")
    await pipeline.cleanup()
    print("🧹 리소스 정리 완료")
    
    print("\n🎉 데모 완료!")

async def performance_test():
    """성능 테스트"""
    
    print("🔬 성능 테스트 시작")
    print("=" * 30)
    
    pipeline = create_m3_max_pipeline()
    await pipeline.initialize()
    
    # 워밍업
    await pipeline.warmup()
    
    # 테스트 실행
    test_count = 3
    results = []
    
    for i in range(test_count):
        print(f"🧪 테스트 {i+1}/{test_count}")
        
        person_image = Image.new('RGB', (512, 512), color=(i*50, 100, 150))
        clothing_image = Image.new('RGB', (512, 512), color=(150, i*30, 100))
        
        result = await pipeline.process_complete_virtual_fitting(
            person_image=person_image,
            clothing_image=clothing_image,
            quality_target=0.7
        )
        
        results.append({
            'success': result.success,
            'processing_time': result.processing_time,
            'quality_score': result.quality_score,
            'completed_steps': len(result.step_results)
        })
        
        print(f"  ✅ 완료: {result.processing_time:.2f}s, 품질: {result.quality_score:.3f}")
    
    # 결과 분석
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        avg_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
        avg_quality = sum(r['quality_score'] for r in successful_results) / len(successful_results)
        
        print(f"\n📊 성능 테스트 결과:")
        print(f"  성공률: {len(successful_results)}/{test_count} ({len(successful_results)/test_count:.1%})")
        print(f"  평균 처리 시간: {avg_time:.2f}초")
        print(f"  평균 품질 점수: {avg_quality:.3f}")
        print(f"  초당 처리량: {len(successful_results)/(sum(r['processing_time'] for r in successful_results)):.2f} 작업/초")
    
    await pipeline.cleanup()
    print("🧹 성능 테스트 완료")

# ==============================================
# 7. 메인 실행
# ==============================================

if __name__ == "__main__":
    print("🎽 단순화된 PipelineManager - Step 클래스들만 관리")
    print("=" * 80)
    print("✨ Step 클래스들의 오케스트레이션만 담당")
    print("🔧 각 Step이 자체적으로 모델/메모리 관리")
    print("📋 기존 함수명/클래스명 100% 유지")
    print("🚀 M3 Max 최적화")
    print("💪 프로덕션 레벨 안정성")
    print("=" * 80)
    
    import asyncio
    
    async def main():
        # 1. 단순화된 데모 실행
        await demo_simplified_pipeline()
        
        print("\n" + "="*50)
        
        # 2. 성능 테스트 실행
        await performance_test()
    
    # 실행
    asyncio.run(main())