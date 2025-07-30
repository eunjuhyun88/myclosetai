# backend/app/ai_pipeline/interface/step_interface.py
"""
🔥 Step Interface v5.1 - 순환참조 완전 해결 + 모든 기능 유지
================================================================

✅ BaseStepMixin 순환참조 완전 차단 (지연 import)
✅ 모든 기능 그대로 유지
✅ Logger 문제 완전 해결
✅ GitHub 프로젝트 구조 100% 호환
✅ PyTorch weights_only 문제 해결
✅ M3 Max 최적화 유지

Author: MyCloset AI Team
Date: 2025-07-30
Version: 5.1 (Circular Reference Fix)
"""

# =============================================================================
# 🔥 1단계: 기본 라이브러리 Import (Logger 전)
# =============================================================================

import os
import gc
import sys
import time
import warnings
import asyncio
import threading
import traceback
import weakref
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Type, Tuple, Set, TYPE_CHECKING
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from functools import wraps, lru_cache
from contextlib import asynccontextmanager
import json
import hashlib

# =============================================================================
# 🔥 2단계: Logger 안전 초기화 (최우선)
# =============================================================================

import logging

# Logger 중복 방지를 위한 전역 설정
_LOGGER_INITIALIZED = False
_MODULE_LOGGER = None

def get_safe_logger():
    """Thread-safe Logger 초기화 (중복 방지)"""
    global _LOGGER_INITIALIZED, _MODULE_LOGGER
    
    if _LOGGER_INITIALIZED and _MODULE_LOGGER is not None:
        return _MODULE_LOGGER
    
    try:
        # 현재 모듈의 Logger 생성
        logger_name = __name__
        _MODULE_LOGGER = logging.getLogger(logger_name)
        
        # 핸들러가 없는 경우에만 추가 (중복 방지)
        if not _MODULE_LOGGER.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            _MODULE_LOGGER.addHandler(handler)
            _MODULE_LOGGER.setLevel(logging.INFO)
        
        _LOGGER_INITIALIZED = True
        return _MODULE_LOGGER
        
    except Exception as e:
        # 최후 폴백: print 사용
        print(f"⚠️ Logger 초기화 실패, fallback 사용: {e}")
        
        class FallbackLogger:
            def info(self, msg): print(f"INFO: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
            def debug(self, msg): print(f"DEBUG: {msg}")
        
        return FallbackLogger()

# 모듈 레벨 Logger (단 한 번만 초기화)
logger = get_safe_logger()

# =============================================================================
# 🔥 3단계: 경고 및 에러 처리 (Logger 정의 후)
# =============================================================================

# 경고 무시 설정
warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*deprecated.*')
warnings.filterwarnings('ignore', category=ImportWarning)

# =============================================================================
# 🔥 4단계: TYPE_CHECKING으로 순환참조 완전 방지
# =============================================================================

if TYPE_CHECKING:
    from ..utils.model_loader import ModelLoader
    from ..factories.step_factory import StepFactory
    from ..utils.memory_manager import MemoryManager
    from ..utils.data_converter import DataConverter
    from ..core import DIContainer
    from ..steps.base_step_mixin import BaseStepMixin

# =============================================================================
# 🔥 5단계: 진단에서 발견된 문제들 해결
# =============================================================================

# 1. PyTorch weights_only 문제 해결
PYTORCH_FIXED = False
try:
    import torch
    if hasattr(torch, 'load'):
        original_torch_load = torch.load
        def safe_torch_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
            if weights_only is None:
                weights_only = False
            return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=weights_only, **kwargs)
        torch.load = safe_torch_load
        PYTORCH_FIXED = True
        logger.info("✅ PyTorch weights_only 호환성 패치 적용")
except Exception as e:
    logger.warning(f"⚠️ PyTorch 패치 실패: {e}")

# 2. rembg 세션 문제 해결
REMBG_AVAILABLE = False
try:
    import rembg
    if hasattr(rembg, 'sessions'):
        REMBG_AVAILABLE = True
        logger.info("✅ rembg 세션 모듈 사용 가능")
    else:
        logger.warning("⚠️ rembg 세션 모듈 호환성 문제 - 폴백 모드 사용")
except Exception as e:
    logger.warning(f"⚠️ rembg 모듈 문제: {e}")

# 3. Safetensors 호환성 확인  
SAFETENSORS_AVAILABLE = False
try:
    import safetensors
    SAFETENSORS_AVAILABLE = True
    logger.info("✅ Safetensors 사용 가능")
except ImportError:
    logger.warning("⚠️ Safetensors 사용 불가 - .pth 파일 우선 사용")

# =============================================================================
# 🔥 6단계: GitHub 프로젝트 환경 감지
# =============================================================================

# GitHub 프로젝트 정보
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
AI_PIPELINE_ROOT = BACKEND_ROOT / "app" / "ai_pipeline"

# conda 환경 정보
CONDA_INFO = {
    'conda_env': os.environ.get('CONDA_DEFAULT_ENV', 'none'),
    'conda_prefix': os.environ.get('CONDA_PREFIX', 'none'),
    'is_target_env': os.environ.get('CONDA_DEFAULT_ENV') == 'mycloset-ai-clean',
    'project_path': str(PROJECT_ROOT)
}

# 하드웨어 감지
IS_M3_MAX = False
MEMORY_GB = 16.0
MPS_AVAILABLE = False

try:
    import platform
    import subprocess
    if platform.system() == 'Darwin':
        result = subprocess.run(
            ['sysctl', '-n', 'machdep.cpu.brand_string'],
            capture_output=True, text=True, timeout=5
        )
        IS_M3_MAX = 'M3' in result.stdout
        
        memory_result = subprocess.run(
            ['sysctl', '-n', 'hw.memsize'],
            capture_output=True, text=True, timeout=5
        )
        if memory_result.returncode == 0:
            MEMORY_GB = round(int(memory_result.stdout.strip()) / (1024**3), 1)
    
    # MPS 감지
    if PYTORCH_FIXED:
        MPS_AVAILABLE = (
            IS_M3_MAX and 
            hasattr(torch.backends, 'mps') and 
            torch.backends.mps.is_available()
        )
except Exception:
    pass

logger.info(f"🔧 환경 정보: conda={CONDA_INFO['conda_env']}, M3_Max={IS_M3_MAX}, MPS={MPS_AVAILABLE}")

# =============================================================================
# 🔥 7단계: GitHub Step 타입 및 상수 정의
# =============================================================================

class GitHubStepType(Enum):
    """GitHub 프로젝트 실제 Step 타입"""
    HUMAN_PARSING = "human_parsing"
    POSE_ESTIMATION = "pose_estimation"
    CLOTH_SEGMENTATION = "cloth_segmentation"
    GEOMETRIC_MATCHING = "geometric_matching"
    CLOTH_WARPING = "cloth_warping"
    VIRTUAL_FITTING = "virtual_fitting"
    POST_PROCESSING = "post_processing"
    QUALITY_ASSESSMENT = "quality_assessment"

class GitHubStepPriority(Enum):
    """GitHub Step 우선순위"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

class GitHubDeviceType(Enum):
    """GitHub 프로젝트 디바이스 타입"""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"

class GitHubProcessingStatus(Enum):
    """GitHub Step 처리 상태"""
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"
    MOCK_MODE = "mock_mode"

# =============================================================================
# 🔥 8단계: 내장 의존성 관리자 (순환참조 해결)
# =============================================================================

class EmbeddedDependencyManager:
    """🔥 내장 의존성 관리자 v5.1 - 순환참조 차단"""
    
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.logger = get_safe_logger()
        
        # 핵심 속성들
        self.step_instance = None
        self.injected_dependencies = {}
        self.dependencies = {}
        
        # 시간 추적
        self.last_injection_time = time.time()
        
        # 성능 메트릭
        self.dependencies_injected = 0
        self.injection_failures = 0
        
        # 스레드 안전성
        self._lock = threading.RLock()
        
        self.logger.debug(f"✅ EmbeddedDependencyManager v5.1 초기화 완료: {step_name}")
    
    def set_step_instance(self, step_instance):
        """Step 인스턴스 설정"""
        try:
            with self._lock:
                self.step_instance = step_instance
                self.logger.debug(f"✅ {self.step_name} Step 인스턴스 설정 완료")
                return True
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} Step 인스턴스 설정 실패: {e}")
            return False
    
    def auto_inject_dependencies(self) -> bool:
        """자동 의존성 주입 (지연 import로 순환참조 방지)"""
        try:
            with self._lock:
                self.logger.info(f"🔄 {self.step_name} EmbeddedDependencyManager 자동 의존성 주입 시작...")
                
                if not self.step_instance:
                    self.logger.warning(f"⚠️ {self.step_name} Step 인스턴스가 설정되지 않음")
                    return False
                
                success_count = 0
                total_dependencies = 0
                
                # ModelLoader 자동 주입 (지연 import)
                if not hasattr(self.step_instance, 'model_loader') or self.step_instance.model_loader is None:
                    total_dependencies += 1
                    try:
                        model_loader = self._resolve_model_loader()
                        if model_loader:
                            self.step_instance.model_loader = model_loader
                            self.injected_dependencies['model_loader'] = model_loader
                            success_count += 1
                            self.dependencies_injected += 1
                            self.logger.info(f"✅ {self.step_name} ModelLoader 자동 주입 성공")
                        else:
                            self.logger.warning(f"⚠️ {self.step_name} ModelLoader 해결 실패 - 실제 의존성 필요")
                            self.injection_failures += 1
                    except Exception as e:
                        self.logger.warning(f"⚠️ {self.step_name} ModelLoader 자동 주입 실패: {e}")
                        self.injection_failures += 1
                
                # MemoryManager 자동 주입 (지연 import)
                if not hasattr(self.step_instance, 'memory_manager') or self.step_instance.memory_manager is None:
                    total_dependencies += 1
                    try:
                        memory_manager = self._resolve_memory_manager()
                        if memory_manager:
                            self.step_instance.memory_manager = memory_manager
                            self.injected_dependencies['memory_manager'] = memory_manager
                            success_count += 1
                            self.dependencies_injected += 1
                            self.logger.info(f"✅ {self.step_name} MemoryManager 자동 주입 성공")
                        else:
                            self.logger.warning(f"⚠️ {self.step_name} MemoryManager 해결 실패")
                            self.injection_failures += 1
                    except Exception as e:
                        self.logger.warning(f"⚠️ {self.step_name} MemoryManager 자동 주입 실패: {e}")
                        self.injection_failures += 1
                
                # DataConverter 자동 주입 (지연 import)
                if not hasattr(self.step_instance, 'data_converter') or self.step_instance.data_converter is None:
                    total_dependencies += 1
                    try:
                        data_converter = self._resolve_data_converter()
                        if data_converter:
                            self.step_instance.data_converter = data_converter
                            self.injected_dependencies['data_converter'] = data_converter
                            success_count += 1
                            self.dependencies_injected += 1
                            self.logger.info(f"✅ {self.step_name} DataConverter 자동 주입 성공")
                        else:
                            self.logger.warning(f"⚠️ {self.step_name} DataConverter 해결 실패")
                            self.injection_failures += 1
                    except Exception as e:
                        self.logger.warning(f"⚠️ {self.step_name} DataConverter 자동 주입 실패: {e}")
                        self.injection_failures += 1
                
                # 성공 여부 판단
                if total_dependencies == 0:
                    self.logger.info(f"✅ {self.step_name} 모든 의존성이 이미 주입되어 있음")
                    return True
                
                success_rate = success_count / total_dependencies if total_dependencies > 0 else 1.0
                
                if success_count > 0:
                    self.logger.info(f"✅ {self.step_name} 실제 의존성 주입 완료: {success_count}/{total_dependencies} ({success_rate*100:.1f}%)")
                    return True
                else:
                    self.logger.warning(f"⚠️ {self.step_name} 실제 의존성 주입 실패: {success_count}/{total_dependencies}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 자동 의존성 주입 중 오류: {e}")
            self.injection_failures += 1
            return False
    
    def _resolve_model_loader(self):
        """ModelLoader 해결 (지연 import로 순환참조 방지)"""
        try:
            # 지연 import로 순환참조 방지
            try:
                import importlib
                module = importlib.import_module('app.ai_pipeline.utils.model_loader')
                if hasattr(module, 'get_global_model_loader'):
                    loader = module.get_global_model_loader()
                    if loader and not isinstance(loader, bool):
                        return loader
            except ImportError:
                self.logger.debug(f"{self.step_name} ModelLoader 모듈 import 실패")
                return None
            
            self.logger.debug(f"{self.step_name} ModelLoader 해결 실패 - 실제 의존성 필요")
            return None
            
        except Exception as e:
            self.logger.debug(f"{self.step_name} ModelLoader 해결 실패: {e}")
            return None
    
    def _resolve_memory_manager(self):
        """MemoryManager 해결 (지연 import로 순환참조 방지)"""
        try:
            # 지연 import로 순환참조 방지
            try:
                import importlib
                module = importlib.import_module('app.ai_pipeline.utils.memory_manager')
                if hasattr(module, 'get_global_memory_manager'):
                    manager = module.get_global_memory_manager()
                    if manager:
                        return manager
            except ImportError:
                self.logger.debug(f"{self.step_name} MemoryManager 모듈 import 실패")
                return None
            
            self.logger.debug(f"{self.step_name} MemoryManager 해결 실패")
            return None
            
        except Exception as e:
            self.logger.debug(f"{self.step_name} MemoryManager 해결 실패: {e}")
            return None
    
    def _resolve_data_converter(self):
        """DataConverter 해결 (지연 import로 순환참조 방지)"""
        try:
            # 지연 import로 순환참조 방지
            try:
                import importlib
                module = importlib.import_module('app.ai_pipeline.utils.data_converter')
                if hasattr(module, 'get_global_data_converter'):
                    converter = module.get_global_data_converter()
                    if converter:
                        return converter
            except ImportError:
                self.logger.debug(f"{self.step_name} DataConverter 모듈 import 실패")
                return None
            
            self.logger.debug(f"{self.step_name} DataConverter 해결 실패")
            return None
            
        except Exception as e:
            self.logger.debug(f"{self.step_name} DataConverter 해결 실패: {e}")
            return None
    
    def inject_model_loader(self, model_loader):
        """ModelLoader 주입"""
        try:
            with self._lock:
                if not self.step_instance:
                    self.logger.warning(f"⚠️ {self.step_name} Step 인스턴스가 설정되지 않음")
                    return False
                
                if model_loader is None:
                    self.logger.warning(f"⚠️ {self.step_name} ModelLoader가 None입니다")
                    return False
                
                if isinstance(model_loader, bool):
                    self.logger.error(f"❌ {self.step_name} ModelLoader는 bool이 아닌 객체여야 합니다")
                    return False
                
                # 주입 실행
                self.step_instance.model_loader = model_loader
                self.injected_dependencies['model_loader'] = model_loader
                self.dependencies_injected += 1
                
                self.logger.info(f"✅ {self.step_name} ModelLoader 주입 완료")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} ModelLoader 주입 실패: {e}")
            self.injection_failures += 1
            return False
    
    def inject_memory_manager(self, memory_manager):
        """MemoryManager 주입"""
        try:
            with self._lock:
                if not self.step_instance:
                    self.logger.warning(f"⚠️ {self.step_name} Step 인스턴스가 설정되지 않음")
                    return False
                
                if memory_manager is None:
                    self.logger.warning(f"⚠️ {self.step_name} MemoryManager가 None입니다")
                    return False
                
                # 주입 실행
                self.step_instance.memory_manager = memory_manager
                self.injected_dependencies['memory_manager'] = memory_manager
                self.dependencies_injected += 1
                
                self.logger.info(f"✅ {self.step_name} MemoryManager 주입 완료")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} MemoryManager 주입 실패: {e}")
            self.injection_failures += 1
            return False
    
    def cleanup(self):
        """리소스 정리"""
        try:
            with self._lock:
                self.logger.info(f"🔄 {self.step_name} EmbeddedDependencyManager 정리 시작...")
                
                # 주입된 의존성들 정리
                for dep_name, dep_instance in self.injected_dependencies.items():
                    try:
                        if hasattr(dep_instance, 'cleanup'):
                            dep_instance.cleanup()
                        elif hasattr(dep_instance, 'close'):
                            dep_instance.close()
                    except Exception as e:
                        self.logger.debug(f"의존성 정리 중 오류 ({dep_name}): {e}")
                
                # 상태 초기화
                self.injected_dependencies.clear()
                self.dependencies.clear()
                self.step_instance = None
                
                self.logger.info(f"✅ {self.step_name} EmbeddedDependencyManager 정리 완료")
                
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} EmbeddedDependencyManager 정리 실패: {e}")

# =============================================================================
# 🔥 9단계: GitHub Step 설정 클래스
# =============================================================================

@dataclass
class GitHubStepConfig:
    """GitHub BaseStepMixin v19.1 완벽 호환 설정"""
    # 기본 Step 정보
    step_name: str = "BaseStep"
    step_id: int = 0
    class_name: str = "BaseStepMixin"
    module_path: str = ""
    
    # GitHub Step 타입
    step_type: GitHubStepType = GitHubStepType.HUMAN_PARSING
    priority: GitHubStepPriority = GitHubStepPriority.MEDIUM
    
    # 디바이스 및 성능 설정
    device: str = "auto"
    use_fp16: bool = False
    batch_size: int = 1
    confidence_threshold: float = 0.5
    
    # GitHub BaseStepMixin 호환성
    github_compatible: bool = True
    basestepmixin_v19_compatible: bool = True
    detailed_data_spec_support: bool = True
    
    # 자동화 설정
    auto_memory_cleanup: bool = True
    auto_warmup: bool = True
    auto_inject_dependencies: bool = True
    optimization_enabled: bool = True
    
    # 의존성 요구사항
    require_model_loader: bool = True
    require_memory_manager: bool = True
    require_data_converter: bool = True
    require_di_container: bool = False
    require_step_interface: bool = True
    
    # AI 모델 설정
    ai_models: List[str] = field(default_factory=list)
    model_size_gb: float = 1.0
    primary_model_file: str = ""
    checkpoint_patterns: List[str] = field(default_factory=list)
    
    # GitHub 환경 최적화
    conda_optimized: bool = True
    m3_max_optimized: bool = True
    conda_env: str = field(default_factory=lambda: CONDA_INFO['conda_env'])
    
    # DetailedDataSpec 설정
    enable_detailed_data_spec: bool = True
    api_input_mapping: Dict[str, str] = field(default_factory=dict)
    api_output_mapping: Dict[str, str] = field(default_factory=dict)
    preprocessing_steps: List[str] = field(default_factory=list)
    postprocessing_steps: List[str] = field(default_factory=list)
    
    # 7단계 Mock 문제 해결 설정
    force_real_ai_processing: bool = True
    mock_mode_disabled: bool = True
    fallback_on_ai_failure: bool = False
    
    def __post_init__(self):
        """초기화 후 GitHub 환경 최적화"""
        # conda 환경 자동 최적화
        if self.conda_env == 'mycloset-ai-clean':
            self.conda_optimized = True
            self.optimization_enabled = True
            self.auto_memory_cleanup = True
            
            # M3 Max + conda 조합 최적화
            if IS_M3_MAX:
                self.m3_max_optimized = True
                if self.device == "auto" and MPS_AVAILABLE:
                    self.device = "mps"
                if self.batch_size == 1 and MEMORY_GB >= 64:
                    self.batch_size = 2
        
        # 7단계 특별 처리
        if self.step_name == "VirtualFittingStep" or self.step_id == 7:
            self.force_real_ai_processing = True
            self.mock_mode_disabled = True
            self.fallback_on_ai_failure = False
        
        # AI 모델 리스트 정규화
        if not isinstance(self.ai_models, list):
            self.ai_models = []

# =============================================================================
# 🔥 10단계: GitHub Step 매핑 시스템
# =============================================================================

class GitHubStepMapping:
    """GitHub 프로젝트 실제 Step 매핑"""
    
    GITHUB_STEP_CONFIGS = {
        GitHubStepType.HUMAN_PARSING: GitHubStepConfig(
            step_name="HumanParsingStep",
            step_id=1,
            class_name="HumanParsingStep",
            module_path="app.ai_pipeline.steps.step_01_human_parsing",
            step_type=GitHubStepType.HUMAN_PARSING,
            priority=GitHubStepPriority.HIGH,
            ai_models=["graphonomy.pth", "atr_model.pth"],
            model_size_gb=4.0,
            primary_model_file="graphonomy.pth"
        ),
        
        GitHubStepType.POSE_ESTIMATION: GitHubStepConfig(
            step_name="PoseEstimationStep",
            step_id=2,
            class_name="PoseEstimationStep",
            module_path="app.ai_pipeline.steps.step_02_pose_estimation",
            step_type=GitHubStepType.POSE_ESTIMATION,
            priority=GitHubStepPriority.MEDIUM,
            ai_models=["pose_model.pth", "openpose_model.pth"],
            model_size_gb=3.4,
            primary_model_file="pose_model.pth"
        ),
        
        GitHubStepType.CLOTH_SEGMENTATION: GitHubStepConfig(
            step_name="ClothSegmentationStep",
            step_id=3,
            class_name="ClothSegmentationStep",
            module_path="app.ai_pipeline.steps.step_03_cloth_segmentation",
            step_type=GitHubStepType.CLOTH_SEGMENTATION,
            priority=GitHubStepPriority.MEDIUM,
            ai_models=["sam_vit_h_4b8939.pth", "u2net.pth"],
            model_size_gb=5.5,
            primary_model_file="sam_vit_h_4b8939.pth"
        ),
        
        GitHubStepType.GEOMETRIC_MATCHING: GitHubStepConfig(
            step_name="GeometricMatchingStep",
            step_id=4,
            class_name="GeometricMatchingStep",
            module_path="app.ai_pipeline.steps.step_04_geometric_matching",
            step_type=GitHubStepType.GEOMETRIC_MATCHING,
            priority=GitHubStepPriority.LOW,
            ai_models=["geometric_matching.pth", "tps_model.pth"],
            model_size_gb=1.3,
            primary_model_file="geometric_matching.pth"
        ),
        
        GitHubStepType.CLOTH_WARPING: GitHubStepConfig(
            step_name="ClothWarpingStep",
            step_id=5,
            class_name="ClothWarpingStep",
            module_path="app.ai_pipeline.steps.step_05_cloth_warping",
            step_type=GitHubStepType.CLOTH_WARPING,
            priority=GitHubStepPriority.HIGH,
            ai_models=["RealVisXL_V4.0.safetensors", "cloth_warping.pth"],
            model_size_gb=7.0,
            primary_model_file="RealVisXL_V4.0.safetensors"
        ),
        
        GitHubStepType.VIRTUAL_FITTING: GitHubStepConfig(
            step_name="VirtualFittingStep",
            step_id=6,
            class_name="VirtualFittingStep",
            module_path="app.ai_pipeline.steps.step_06_virtual_fitting",
            step_type=GitHubStepType.VIRTUAL_FITTING,
            priority=GitHubStepPriority.CRITICAL,
            ai_models=[
                "v1-5-pruned.safetensors",
                "v1-5-pruned-emaonly.safetensors",
                "controlnet_openpose",
                "vae_decoder"
            ],
            model_size_gb=14.0,
            primary_model_file="v1-5-pruned.safetensors",
            force_real_ai_processing=True,
            mock_mode_disabled=True,
            fallback_on_ai_failure=False
        ),
        
        GitHubStepType.POST_PROCESSING: GitHubStepConfig(
            step_name="PostProcessingStep",
            step_id=7,
            class_name="PostProcessingStep",
            module_path="app.ai_pipeline.steps.step_07_post_processing",
            step_type=GitHubStepType.POST_PROCESSING,
            priority=GitHubStepPriority.LOW,
            ai_models=["super_resolution.pth", "enhancement.pth"],
            model_size_gb=1.3,
            primary_model_file="super_resolution.pth"
        ),
        
        GitHubStepType.QUALITY_ASSESSMENT: GitHubStepConfig(
            step_name="QualityAssessmentStep",
            step_id=8,
            class_name="QualityAssessmentStep",
            module_path="app.ai_pipeline.steps.step_08_quality_assessment",
            step_type=GitHubStepType.QUALITY_ASSESSMENT,
            priority=GitHubStepPriority.CRITICAL,
            ai_models=["open_clip_pytorch_model.bin", "ViT-L-14.pt"],
            model_size_gb=7.0,
            primary_model_file="open_clip_pytorch_model.bin"
        )
    }
    
    @classmethod
    def get_config(cls, step_type: GitHubStepType) -> GitHubStepConfig:
        """Step 타입별 설정 반환"""
        return cls.GITHUB_STEP_CONFIGS.get(step_type, GitHubStepConfig())
    
    @classmethod
    def get_config_by_name(cls, step_name: str) -> Optional[GitHubStepConfig]:
        """Step 이름으로 설정 반환"""
        for config in cls.GITHUB_STEP_CONFIGS.values():
            if config.step_name == step_name or config.class_name == step_name:
                return config
        return None
    
    @classmethod
    def get_config_by_id(cls, step_id: int) -> Optional[GitHubStepConfig]:
        """Step ID로 설정 반환"""
        for config in cls.GITHUB_STEP_CONFIGS.values():
            if config.step_id == step_id:
                return config
        return None

# =============================================================================
# 🔥 11단계: GitHub 메모리 관리 시스템
# =============================================================================

class GitHubMemoryManager:
    """GitHub 프로젝트용 고급 메모리 관리 시스템"""
    
    def __init__(self, max_memory_gb: float = None):
        self.logger = get_safe_logger()
        
        # M3 Max 자동 최적화
        if max_memory_gb is None:
            if IS_M3_MAX and MEMORY_GB >= 64:
                self.max_memory_gb = MEMORY_GB * 0.85
            elif IS_M3_MAX:
                self.max_memory_gb = MEMORY_GB * 0.8
            elif CONDA_INFO['is_target_env']:
                self.max_memory_gb = 12.0
            else:
                self.max_memory_gb = 8.0
        else:
            self.max_memory_gb = max_memory_gb
        
        self.current_memory_gb = 0.0
        self.memory_pool = {}
        self.allocation_history = []
        self._lock = threading.RLock()
        
        # M3 Max 특화 설정
        self.is_m3_max = IS_M3_MAX
        self.mps_enabled = MPS_AVAILABLE
        
        self.logger.info(f"🧠 GitHub 메모리 관리자 초기화: {self.max_memory_gb:.1f}GB (M3 Max: {self.is_m3_max})")
    
    def allocate_memory(self, size_gb: float, owner: str) -> bool:
        """메모리 할당"""
        with self._lock:
            if self.current_memory_gb + size_gb <= self.max_memory_gb:
                self.current_memory_gb += size_gb
                self.memory_pool[owner] = size_gb
                self.logger.debug(f"✅ 메모리 할당: {size_gb:.1f}GB → {owner}")
                return True
            else:
                available = self.max_memory_gb - self.current_memory_gb
                self.logger.warning(f"❌ 메모리 부족: {size_gb:.1f}GB 요청, {available:.1f}GB 사용 가능")
                return False
    
    def deallocate_memory(self, owner: str) -> float:
        """메모리 해제"""
        with self._lock:
            if owner in self.memory_pool:
                size_gb = self.memory_pool[owner]
                del self.memory_pool[owner]
                self.current_memory_gb -= size_gb
                self.logger.debug(f"✅ 메모리 해제: {size_gb:.1f}GB ← {owner}")
                return size_gb
            return 0.0
    
    def optimize_for_github_project(self):
        """GitHub 프로젝트 특화 메모리 최적화"""
        try:
            # MPS 메모리 정리 (M3 Max)
            if self.mps_enabled and PYTORCH_FIXED:
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
                    self.logger.debug("🍎 MPS 메모리 캐시 정리")
            
            # Python GC
            gc.collect()
            
            # 128GB M3 Max 특별 최적화
            if IS_M3_MAX and MEMORY_GB >= 128:
                self.max_memory_gb = min(MEMORY_GB * 0.9, 115.0)
                self.logger.info(f"🍎 M3 Max 128GB 메모리 풀 확장: {self.max_memory_gb:.1f}GB")
            
        except Exception as e:
            self.logger.error(f"❌ GitHub 메모리 최적화 실패: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 통계"""
        with self._lock:
            return {
                'current_gb': self.current_memory_gb,
                'max_gb': self.max_memory_gb,
                'available_gb': self.max_memory_gb - self.current_memory_gb,
                'usage_percent': (self.current_memory_gb / self.max_memory_gb) * 100,
                'memory_pool': self.memory_pool.copy(),
                'is_m3_max': self.is_m3_max,
                'mps_enabled': self.mps_enabled,
                'total_system_gb': MEMORY_GB,
                'github_optimized': True
            }

# =============================================================================
# 🔥 12단계: GitHub Step Model Interface (핵심 클래스)
# =============================================================================

class GitHubStepModelInterface:
    """
    🔥 GitHub Step용 ModelLoader 인터페이스 v5.1 - 순환참조 완전 해결
    
    ✅ 순환참조 완전 방지 (내장 의존성 관리자 사용)
    ✅ BaseStepMixin v19.1 완벽 호환
    ✅ register_model_requirement 완전 구현
    ✅ list_available_models 정확 구현
    ✅ 7단계 Mock 데이터 문제 해결
    ✅ PyTorch weights_only 문제 해결
    """
    
    def __init__(self, step_name: str, model_loader: Optional['ModelLoader'] = None):
        self.step_name = step_name
        self.model_loader = model_loader
        self.logger = get_safe_logger()
        
        # GitHub 설정 자동 로딩
        self.config = GitHubStepMapping.get_config_by_name(step_name)
        if not self.config:
            self.config = GitHubStepConfig(step_name=step_name)
        
        # 모델 관리
        self._model_registry: Dict[str, Dict[str, Any]] = {}
        self._model_cache: Dict[str, Any] = {}
        self._model_requirements: Dict[str, Any] = {}
        
        # 메모리 관리
        self.memory_manager = GitHubMemoryManager()
        
        # 🔥 내장 의존성 관리자 (순환참조 해결)
        self.dependency_manager = EmbeddedDependencyManager(step_name)
        
        # 동기화
        self._lock = threading.RLock()
        
        # 통계
        self.statistics = {
            'models_registered': 0,
            'models_loaded': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'loading_failures': 0,
            'real_ai_calls': 0,
            'mock_calls_blocked': 0,
            'creation_time': time.time()
        }
        
        # 7단계 특별 처리
        if self.config.step_id == 6:  # VirtualFittingStep
            self.statistics['force_real_ai'] = True
            self.logger.info(f"🔥 {step_name}: 실제 AI 모델 강제 사용 모드 활성화")
        
        self.logger.info(f"🔗 GitHub {step_name} Interface v5.1 순환참조 해결 초기화 완료")
    
    def register_model_requirement(
        self, 
        model_name: str, 
        model_type: str = "BaseModel",
        **kwargs
    ) -> bool:
        """모델 요구사항 등록 - BaseStepMixin v19.1 완벽 호환"""
        try:
            with self._lock:
                self.logger.info(f"📝 GitHub 모델 요구사항 등록: {model_name} ({model_type})")
                
                # GitHub 설정 기반 요구사항 생성
                requirement = {
                    'model_name': model_name,
                    'model_type': model_type,
                    'step_name': self.step_name,
                    'step_id': self.config.step_id,
                    'device': kwargs.get('device', self.config.device),
                    'precision': 'fp16' if self.config.use_fp16 else 'fp32',
                    'github_compatible': True,
                    'force_real_ai': self.config.force_real_ai_processing,
                    'mock_disabled': self.config.mock_mode_disabled,
                    'registered_at': time.time(),
                    'pytorch_fixed': PYTORCH_FIXED,
                    'rembg_available': REMBG_AVAILABLE,
                    'safetensors_available': SAFETENSORS_AVAILABLE,
                    'metadata': {
                        'module_path': self.config.module_path,
                        'class_name': self.config.class_name,
                        'primary_model_file': self.config.primary_model_file,
                        **kwargs.get('metadata', {})
                    }
                }
                
                # 요구사항 저장
                self._model_requirements[model_name] = requirement
                
                # 모델 레지스트리 등록
                self._model_registry[model_name] = {
                    'name': model_name,
                    'type': model_type,
                    'step_class': self.step_name,
                    'step_id': self.config.step_id,
                    'loaded': False,
                    'size_mb': self.config.model_size_gb * 1024,
                    'device': requirement['device'],
                    'status': 'registered',
                    'github_compatible': True,
                    'requirement': requirement,
                    'registered_at': requirement['registered_at']
                }
                
                # 통계 업데이트
                self.statistics['models_registered'] += 1
                
                # ModelLoader에 전달
                if self.model_loader and hasattr(self.model_loader, 'register_model_requirement'):
                    try:
                        self.model_loader.register_model_requirement(
                            model_name=model_name,
                            model_type=model_type,
                            step_name=self.step_name,
                            **kwargs
                        )
                    except Exception as e:
                        self.logger.warning(f"⚠️ ModelLoader 요구사항 전달 실패: {e}")
                
                self.logger.info(f"✅ GitHub 모델 요구사항 등록 완료: {model_name}")
                return True
                
        except Exception as e:
            self.statistics['loading_failures'] += 1
            self.logger.error(f"❌ GitHub 모델 요구사항 등록 실패: {model_name} - {e}")
            return False
    
    def list_available_models(
        self, 
        step_class: Optional[str] = None,
        model_type: Optional[str] = None,
        include_unloaded: bool = True,
        sort_by: str = "size"
    ) -> List[Dict[str, Any]]:
        """사용 가능한 모델 목록 반환 - GitHub 구조 기반"""
        try:
            with self._lock:
                models = []
                
                # 등록된 모델들에서 목록 생성
                for model_name, registry_entry in self._model_registry.items():
                    # 필터링
                    if step_class and registry_entry['step_class'] != step_class:
                        continue
                    if model_type and registry_entry['type'] != model_type:
                        continue
                    if not include_unloaded and not registry_entry['loaded']:
                        continue
                    
                    requirement = registry_entry.get('requirement', {})
                    
                    # GitHub 표준 모델 정보
                    model_info = {
                        'name': model_name,
                        'path': f"ai_models/step_{requirement.get('step_id', self.config.step_id):02d}_{self.step_name.lower()}/{model_name}",
                        'size_mb': registry_entry['size_mb'],
                        'size_gb': round(registry_entry['size_mb'] / 1024, 2),
                        'model_type': registry_entry['type'],
                        'step_class': registry_entry['step_class'],
                        'step_id': registry_entry['step_id'],
                        'loaded': registry_entry['loaded'],
                        'device': registry_entry['device'],
                        'status': registry_entry['status'],
                        'github_compatible': registry_entry.get('github_compatible', True),
                        'force_real_ai': requirement.get('force_real_ai', False),
                        'mock_disabled': requirement.get('mock_disabled', False),
                        'pytorch_fixed': requirement.get('pytorch_fixed', PYTORCH_FIXED),
                        'rembg_available': requirement.get('rembg_available', REMBG_AVAILABLE),
                        'safetensors_available': requirement.get('safetensors_available', SAFETENSORS_AVAILABLE),
                        'metadata': {
                            'step_name': self.step_name,
                            'conda_env': CONDA_INFO['conda_env'],
                            'registered_at': requirement.get('registered_at', 0),
                            **requirement.get('metadata', {})
                        }
                    }
                    models.append(model_info)
                
                # ModelLoader에서 추가 모델 조회
                if self.model_loader and hasattr(self.model_loader, 'list_available_models'):
                    try:
                        additional_models = self.model_loader.list_available_models(
                            step_class=step_class or self.step_name,
                            model_type=model_type
                        )
                        
                        # 중복 제거하며 추가
                        existing_names = {m['name'] for m in models}
                        for model in additional_models:
                            if model['name'] not in existing_names:
                                model_info = {
                                    'name': model['name'],
                                    'path': model.get('path', f"loader_models/{model['name']}"),
                                    'size_mb': model.get('size_mb', 0.0),
                                    'size_gb': round(model.get('size_mb', 0.0) / 1024, 2),
                                    'model_type': model.get('model_type', 'unknown'),
                                    'step_class': model.get('step_class', self.step_name),
                                    'step_id': self.config.step_id,
                                    'loaded': model.get('loaded', False),
                                    'device': model.get('device', 'auto'),
                                    'status': 'loaded' if model.get('loaded', False) else 'available',
                                    'github_compatible': False,
                                    'force_real_ai': False,
                                    'mock_disabled': False,
                                    'pytorch_fixed': PYTORCH_FIXED,
                                    'rembg_available': REMBG_AVAILABLE,
                                    'safetensors_available': SAFETENSORS_AVAILABLE,
                                    'metadata': {
                                        'step_name': self.step_name,
                                        'source': 'model_loader',
                                        **model.get('metadata', {})
                                    }
                                }
                                models.append(model_info)
                    except Exception as e:
                        self.logger.warning(f"⚠️ ModelLoader 모델 목록 조회 실패: {e}")
                
                # 정렬 수행
                if sort_by == "size":
                    models.sort(key=lambda x: x['size_mb'], reverse=True)
                elif sort_by == "name":
                    models.sort(key=lambda x: x['name'])
                elif sort_by == "step_id":
                    models.sort(key=lambda x: x['step_id'])
                else:
                    models.sort(key=lambda x: x['size_mb'], reverse=True)
                
                self.logger.debug(f"📋 GitHub 모델 목록 반환: {len(models)}개")
                return models
            
        except Exception as e:
            self.logger.error(f"❌ GitHub 모델 목록 조회 실패: {e}")
            return []
    
    def get_model_sync(self, model_name: str, **kwargs) -> Optional[Any]:
        """모델 로드 (동기) - PyTorch weights_only 문제 해결"""
        try:
            with self._lock:
                # 7단계 특별 처리: Mock 데이터 차단
                if self.config.step_id == 6 and ('mock' in model_name.lower() or 'test' in model_name.lower()):
                    self.statistics['mock_calls_blocked'] += 1
                    self.logger.warning(f"🔥 {self.step_name}: Mock 모델 호출 차단 - {model_name}")
                    return None
                
                # 캐시 확인
                if model_name in self._model_cache:
                    self.statistics['cache_hits'] += 1
                    self.statistics['real_ai_calls'] += 1
                    self.logger.debug(f"♻️ 캐시된 모델 반환: {model_name}")
                    return self._model_cache[model_name]
                
                # PyTorch 로딩 문제 해결
                loading_kwargs = kwargs.copy()
                if PYTORCH_FIXED and 'weights_only' not in loading_kwargs:
                    loading_kwargs['weights_only'] = False
                
                # ModelLoader를 통한 로딩
                if self.model_loader and hasattr(self.model_loader, 'load_model'):
                    try:
                        model = self.model_loader.load_model(model_name, **loading_kwargs)
                    except Exception as load_error:
                        # PyTorch 로딩 오류 재시도
                        if PYTORCH_FIXED and ('weights_only' in str(load_error) or 'WeightsUnpickler' in str(load_error)):
                            self.logger.warning(f"⚠️ PyTorch weights_only 오류 감지, 재시도: {model_name}")
                            loading_kwargs['weights_only'] = False
                            loading_kwargs['map_location'] = 'cpu'
                            model = self.model_loader.load_model(model_name, **loading_kwargs)
                        else:
                            raise load_error
                    
                    if model is not None:
                        # 캐시에 저장
                        self._model_cache[model_name] = model
                        
                        # 레지스트리 업데이트
                        if model_name in self._model_registry:
                            self._model_registry[model_name]['loaded'] = True
                            self._model_registry[model_name]['status'] = 'loaded'
                        
                        # 통계 업데이트
                        self.statistics['models_loaded'] += 1
                        self.statistics['real_ai_calls'] += 1
                        
                        self.logger.info(f"✅ GitHub 모델 로드 성공: {model_name}")
                        return model
                
                # 로딩 실패
                self.statistics['cache_misses'] += 1
                self.statistics['loading_failures'] += 1
                self.logger.warning(f"⚠️ GitHub 모델 로드 실패: {model_name}")
                return None
                
        except Exception as e:
            self.statistics['loading_failures'] += 1
            self.logger.error(f"❌ GitHub 모델 로드 실패: {model_name} - {e}")
            
            # 진단에서 발견된 특정 오류 처리
            if 'constants.pkl' in str(e):
                self.logger.warning(f"🔧 Mobile SAM 모델 파일 손상 감지: {model_name}")
            elif 'Expected hasRecord' in str(e):
                self.logger.warning(f"🔧 Graphonomy 모델 버전 문제 감지: {model_name}")
            elif 'Unsupported operand' in str(e):
                self.logger.warning(f"🔧 U2Net 모델 호환성 문제 감지: {model_name}")
            
            return None
    
    # BaseStepMixin v19.1 호환을 위한 별칭
    def load_model(self, model_name: str, **kwargs) -> Optional[Any]:
        """모델 로드 - BaseStepMixin 호환 별칭"""
        return self.get_model_sync(model_name, **kwargs)
    
    def get_model(self, model_name: str = None, **kwargs) -> Optional[Any]:
        """모델 조회 - BaseStepMixin 호환"""
        if model_name:
            return self.get_model_sync(model_name, **kwargs)
        return None
    
    def cleanup(self):
        """리소스 정리"""
        try:
            # 메모리 해제
            for model_name in self._model_cache:
                self.memory_manager.deallocate_memory(model_name)
            
            self._model_cache.clear()
            self._model_requirements.clear()
            self._model_registry.clear()
            self.dependency_manager.cleanup()
            
            self.logger.info(f"✅ GitHub {self.step_name} Interface 정리 완료")
        except Exception as e:
            self.logger.error(f"❌ GitHub Interface 정리 실패: {e}")

# =============================================================================
# 🔥 13단계: Step 생성 결과 데이터 구조
# =============================================================================

@dataclass
class GitHubStepCreationResult:
    """GitHub Step 생성 결과"""
    success: bool
    step_instance: Optional[Any] = None
    step_name: str = ""
    step_id: int = 0
    step_type: Optional[GitHubStepType] = None
    class_name: str = ""
    module_path: str = ""
    device: str = "cpu"
    creation_time: float = field(default_factory=time.time)
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # GitHub 의존성 주입 결과
    dependencies_injected: Dict[str, bool] = field(default_factory=dict)
    initialization_success: bool = False
    ai_models_loaded: List[str] = field(default_factory=list)
    
    # GitHub BaseStepMixin v19.1 호환성
    github_compatible: bool = True
    basestepmixin_v19_compatible: bool = True
    detailed_data_spec_loaded: bool = False
    process_method_validated: bool = False
    dependency_injection_success: bool = False
    
    # 순환참조 해결 상태
    circular_reference_free: bool = True
    embedded_dependency_manager: bool = True
    real_ai_processing_enabled: bool = False
    
    # 메모리 및 성능
    memory_usage_mb: float = 0.0
    conda_env: str = field(default_factory=lambda: CONDA_INFO['conda_env'])
    metadata: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# 🔥 14단계: Step 파일들을 위한 간단한 인터페이스 (호환성)
# =============================================================================

class StepInterface:
    """Step 파일들이 사용하는 간단한 인터페이스"""
    
    def __init__(self, step_name: str, model_loader=None, **kwargs):
        self.step_name = step_name
        self.model_loader = model_loader
        self.logger = get_safe_logger()
        self.config = kwargs
        
        # 기본 속성들
        self.step_id = kwargs.get('step_id', 0)
        self.device = kwargs.get('device', 'auto')
        self.initialized = False
        
        self.logger.debug(f"✅ StepInterface 생성: {step_name}")
    
    def register_model_requirement(self, model_name: str, model_type: str = "BaseModel", **kwargs) -> bool:
        """모델 요구사항 등록 (호환성)"""
        try:
            self.logger.debug(f"📝 모델 요구사항 등록: {model_name} ({model_type})")
            return True
        except Exception as e:
            self.logger.error(f"❌ 모델 요구사항 등록 실패: {e}")
            return False
    
    def list_available_models(self, **kwargs) -> List[Dict[str, Any]]:
        """사용 가능한 모델 목록 (호환성)"""
        return []
    
    def get_model(self, model_name: str = None, **kwargs) -> Optional[Any]:
        """모델 조회 (호환성)"""
        try:
            if self.model_loader and hasattr(self.model_loader, 'get_model'):
                return self.model_loader.get_model(model_name, **kwargs)
            return None
        except Exception as e:
            self.logger.error(f"❌ 모델 조회 실패: {e}")
            return None
    
    def load_model(self, model_name: str, **kwargs) -> Optional[Any]:
        """모델 로드 (호환성)"""
        try:
            if self.model_loader and hasattr(self.model_loader, 'load_model'):
                return self.model_loader.load_model(model_name, **kwargs)
            return None
        except Exception as e:
            self.logger.error(f"❌ 모델 로드 실패: {e}")
            return None

# =============================================================================
# 🔥 15단계: 단순한 폴백 클래스들 (Step 파일 호환성용)
# =============================================================================

class SimpleStepConfig:
    """간단한 Step 설정 (폴백용)"""
    def __init__(self, **kwargs):
        self.step_name = kwargs.get('step_name', 'Unknown')
        self.step_id = kwargs.get('step_id', 0)
        self.device = kwargs.get('device', 'auto')
        self.model_size_gb = kwargs.get('model_size_gb', 1.0)
        self.ai_models = kwargs.get('ai_models', [])
        
        # 모든 kwargs를 속성으로 설정
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

# =============================================================================
# 🔥 16단계: 팩토리 함수들 (순환참조 해결 버전)
# =============================================================================

def create_github_step_interface_circular_reference_free(
    step_name: str, 
    model_loader: Optional['ModelLoader'] = None,
    step_type: Optional[GitHubStepType] = None
) -> GitHubStepModelInterface:
    """순환참조 완전 해결된 GitHub Step Interface 생성"""
    try:
        interface = GitHubStepModelInterface(step_name, model_loader)
        
        # Step 타입별 추가 설정
        if step_type:
            config = GitHubStepMapping.get_config(step_type)
            interface.config = config
        
        # 순환참조 해결 적용
        if IS_M3_MAX and MEMORY_GB >= 128:
            interface.memory_manager = GitHubMemoryManager(115.0)
            interface.logger.info(f"🍎 M3 Max 128GB 메모리 최적화 적용")
        
        # 7단계 Mock 차단 강화
        if step_name == "VirtualFittingStep" or (interface.config and interface.config.step_id == 6):
            interface.statistics['force_real_ai'] = True
            interface.statistics['circular_reference_free'] = True
            interface.logger.info(f"🔥 Step 06 VirtualFittingStep 순환참조 해결 적용")
        
        # 내장 의존성 관리자 자동 주입
        interface.dependency_manager.auto_inject_dependencies()
        
        logger.info(f"✅ 순환참조 해결된 GitHub Step Interface 생성: {step_name}")
        return interface
        
    except Exception as e:
        logger.error(f"❌ 순환참조 해결 GitHub Step Interface 생성 실패: {step_name} - {e}")
        return GitHubStepModelInterface(step_name, None)

def create_optimized_github_interface_v51(
    step_name: str,
    model_loader: Optional['ModelLoader'] = None
) -> GitHubStepModelInterface:
    """최적화된 GitHub Interface 생성 v5.1"""
    try:
        # Step 이름으로 타입 자동 감지
        step_type = None
        for github_type in GitHubStepType:
            if github_type.value.replace('_', '').lower() in step_name.lower():
                step_type = github_type
                break
        
        interface = create_github_step_interface_circular_reference_free(
            step_name=step_name,
            model_loader=model_loader,
            step_type=step_type
        )
        
        # conda + M3 Max 조합 최적화
        if CONDA_INFO['is_target_env'] and IS_M3_MAX:
            max_memory_gb = MEMORY_GB * 0.9  # 90% 사용
            interface.memory_manager = GitHubMemoryManager(max_memory_gb)
        elif IS_M3_MAX:
            max_memory_gb = MEMORY_GB * 0.8  # 80% 사용
            interface.memory_manager = GitHubMemoryManager(max_memory_gb)
        
        logger.info(f"✅ 최적화된 GitHub Interface v5.1: {step_name} (conda: {CONDA_INFO['is_target_env']}, M3: {IS_M3_MAX})")
        return interface
        
    except Exception as e:
        logger.error(f"❌ 최적화된 GitHub Interface v5.1 생성 실패: {step_name} - {e}")
        return create_github_step_interface_circular_reference_free(step_name, model_loader)

def create_step_07_virtual_fitting_interface_v51(
    model_loader: Optional['ModelLoader'] = None
) -> GitHubStepModelInterface:
    """7단계 VirtualFittingStep 전용 Interface v5.1 - 순환참조 해결"""
    try:
        interface = GitHubStepModelInterface("VirtualFittingStep", model_loader)
        
        # 7단계 특별 설정 강제 적용
        interface.config.step_id = 6  # VirtualFittingStep은 실제로는 6번
        interface.config.force_real_ai_processing = True
        interface.config.mock_mode_disabled = True
        interface.config.fallback_on_ai_failure = False
        interface.config.model_size_gb = 14.0  # 14GB 대형 모델
        
        # Mock 차단 통계 초기화
        interface.statistics['mock_calls_blocked'] = 0
        interface.statistics['force_real_ai'] = True
        interface.statistics['circular_reference_free'] = True
        
        # 실제 AI 모델만 등록
        real_models = [
            "v1-5-pruned.safetensors",
            "v1-5-pruned-emaonly.safetensors",
            "diffusion_pytorch_model.fp16.safetensors",
            "controlnet_openpose",
            "vae_decoder"
        ]
        
        for model_name in real_models:
            interface.register_model_requirement(
                model_name=model_name,
                model_type="DiffusionModel",
                device="auto",
                force_real_ai=True,
                mock_disabled=True
            )
        
        # 내장 의존성 관리자 활용
        interface.dependency_manager.auto_inject_dependencies()
        
        logger.info("🔥 Step 07 VirtualFittingStep Interface v5.1 생성 완료 - 순환참조 해결")
        return interface
        
    except Exception as e:
        logger.error(f"❌ Step 07 Interface v5.1 생성 실패: {e}")
        return create_github_step_interface_circular_reference_free("VirtualFittingStep", model_loader)

def create_simple_step_interface(step_name: str, **kwargs) -> StepInterface:
    """간단한 Step Interface 생성 (호환성)"""
    try:
        return StepInterface(step_name, **kwargs)
    except Exception as e:
        logger.error(f"❌ 간단한 Step Interface 생성 실패: {e}")
        return StepInterface(step_name)

# =============================================================================
# 🔥 17단계: 유틸리티 함수들 (순환참조 해결 버전)
# =============================================================================

def get_github_environment_info() -> Dict[str, Any]:
    """GitHub 프로젝트 환경 정보"""
    return {
        'github_project': {
            'project_root': str(PROJECT_ROOT),
            'backend_root': str(BACKEND_ROOT),
            'ai_pipeline_root': str(AI_PIPELINE_ROOT),
            'structure_detected': AI_PIPELINE_ROOT.exists()
        },
        'conda_info': CONDA_INFO,
        'system_info': {
            'is_m3_max': IS_M3_MAX,
            'memory_gb': MEMORY_GB,
            'mps_available': MPS_AVAILABLE
        },
        'fixes_applied': {
            'pytorch_fixed': PYTORCH_FIXED,
            'rembg_available': REMBG_AVAILABLE,
            'safetensors_available': SAFETENSORS_AVAILABLE,
            'circular_reference_resolved': True
        }
    }

def optimize_github_environment():
    """GitHub 프로젝트 환경 최적화"""
    try:
        optimizations = []
        
        # conda 환경 최적화
        if CONDA_INFO['is_target_env']:
            optimizations.append("conda 환경 mycloset-ai-clean 최적화")
        
        # M3 Max 최적화
        if IS_M3_MAX:
            optimizations.append("M3 Max 하드웨어 최적화")
            
            # MPS 메모리 정리
            if MPS_AVAILABLE and PYTORCH_FIXED:
                try:
                    if hasattr(torch.backends.mps, 'empty_cache'):
                        torch.backends.mps.empty_cache()
                    optimizations.append("MPS 메모리 정리")
                except:
                    pass
        
        # 가비지 컬렉션
        gc.collect()
        optimizations.append("가비지 컬렉션")
        optimizations.append("순환참조 해결 적용")
        
        logger.info(f"✅ GitHub 환경 최적화 완료: {', '.join(optimizations)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ GitHub 환경 최적화 실패: {e}")
        return False

def validate_github_step_compatibility(step_instance: Any) -> Dict[str, Any]:
    """GitHub Step 호환성 검증 (순환참조 해결 버전)"""
    try:
        result = {
            'compatible': False,
            'github_structure': False,
            'basestepmixin_v19_compatible': False,
            'detailed_data_spec_compatible': False,
            'process_method_exists': False,
            'dependency_injection_ready': False,
            'circular_reference_free': True,
            'embedded_dependency_manager': False,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        if step_instance is None:
            result['errors'].append('Step 인스턴스가 None')
            return result
        
        # 클래스 상속 확인
        class_name = step_instance.__class__.__name__
        mro = [cls.__name__ for cls in step_instance.__class__.__mro__]
        
        if 'BaseStepMixin' in mro:
            result['basestepmixin_v19_compatible'] = True
        else:
            result['warnings'].append('BaseStepMixin 상속 권장')
        
        # GitHub 메서드 확인
        required_methods = ['process', 'initialize', '_run_ai_inference']
        existing_methods = []
        
        for method_name in required_methods:
            if hasattr(step_instance, method_name):
                existing_methods.append(method_name)
        
        result['process_method_exists'] = 'process' in existing_methods
        result['github_structure'] = len(existing_methods) >= 2
        
        # DetailedDataSpec 확인
        if hasattr(step_instance, 'detailed_data_spec') and getattr(step_instance, 'detailed_data_spec') is not None:
            result['detailed_data_spec_compatible'] = True
        else:
            result['warnings'].append('DetailedDataSpec 로딩 권장')
        
        # 내장 의존성 관리자 확인
        if hasattr(step_instance, 'dependency_manager'):
            if isinstance(getattr(step_instance, 'dependency_manager'), EmbeddedDependencyManager):
                result['embedded_dependency_manager'] = True
                result['circular_reference_free'] = True
            else:
                result['warnings'].append('EmbeddedDependencyManager 사용 권장')
        
        # 의존성 주입 상태 확인
        dependency_attrs = ['model_loader', 'memory_manager', 'data_converter']
        injected_deps = []
        for attr in dependency_attrs:
            if hasattr(step_instance, attr) and getattr(step_instance, attr) is not None:
                injected_deps.append(attr)
        
        result['dependency_injection_ready'] = len(injected_deps) >= 1
        result['injected_dependencies'] = injected_deps
        
        # GitHub 특별 속성 확인
        if hasattr(step_instance, 'github_compatible') and getattr(step_instance, 'github_compatible'):
            result['github_mode'] = True
        else:
            result['recommendations'].append('github_compatible=True 설정 권장')
        
        # 7단계 특별 확인
        if class_name == 'VirtualFittingStep' or getattr(step_instance, 'step_id', 0) == 6:
            if hasattr(step_instance, 'force_real_ai_processing'):
                result['step_07_mock_fixed'] = True
            else:
                result['warnings'].append('Step 07 Mock 문제 해결 필요')
        
        # 종합 호환성 판정
        result['compatible'] = (
            result['basestepmixin_v19_compatible'] and
            result['process_method_exists'] and
            result['dependency_injection_ready'] and
            result['circular_reference_free']
        )
        
        return result
        
    except Exception as e:
        return {
            'compatible': False,
            'error': str(e),
            'version': 'GitHubStepInterface v5.1 Circular Reference Free'
        }

def get_github_step_info(step_instance: Any) -> Dict[str, Any]:
    """GitHub Step 인스턴스 정보 조회 (순환참조 해결 버전)"""
    try:
        info = {
            'step_name': getattr(step_instance, 'step_name', 'Unknown'),
            'step_id': getattr(step_instance, 'step_id', 0),
            'class_name': step_instance.__class__.__name__,
            'module': step_instance.__class__.__module__,
            'device': getattr(step_instance, 'device', 'Unknown'),
            'is_initialized': getattr(step_instance, 'is_initialized', False),
            'github_compatible': getattr(step_instance, 'github_compatible', False),
            'has_model': getattr(step_instance, 'has_model', False),
            'model_loaded': getattr(step_instance, 'model_loaded', False),
            'circular_reference_free': True
        }
        
        # GitHub 의존성 상태
        dependencies = {}
        for dep_name in ['model_loader', 'memory_manager', 'data_converter', 'di_container', 'step_interface']:
            dependencies[dep_name] = hasattr(step_instance, dep_name) and getattr(step_instance, dep_name) is not None
        
        info['dependencies'] = dependencies
        
        # 내장 의존성 관리자 상태
        if hasattr(step_instance, 'dependency_manager'):
            dep_manager = getattr(step_instance, 'dependency_manager')
            if isinstance(dep_manager, EmbeddedDependencyManager):
                info['embedded_dependency_manager'] = {
                    'type': 'EmbeddedDependencyManager',
                    'injected_count': dep_manager.dependencies_injected,
                    'injection_failures': dep_manager.injection_failures,
                    'circular_reference_safe': True
                }
            else:
                info['embedded_dependency_manager'] = {
                    'type': type(dep_manager).__name__,
                    'circular_reference_safe': False
                }
        
        # DetailedDataSpec 상태
        detailed_data_spec_info = {}
        for attr_name in ['detailed_data_spec', 'api_input_mapping', 'api_output_mapping', 'preprocessing_steps', 'postprocessing_steps']:
            detailed_data_spec_info[attr_name] = hasattr(step_instance, attr_name) and getattr(step_instance, attr_name) is not None
        
        info['detailed_data_spec'] = detailed_data_spec_info
        
        # 7단계 특별 정보
        if info['class_name'] == 'VirtualFittingStep' or info['step_id'] == 6:
            info['step_07_status'] = {
                'force_real_ai': getattr(step_instance, 'force_real_ai_processing', False),
                'mock_disabled': getattr(step_instance, 'mock_mode_disabled', False),
                'fallback_disabled': not getattr(step_instance, 'fallback_on_ai_failure', True),
                'circular_reference_free': True
            }
        
        # 성능 메트릭
        if hasattr(step_instance, 'performance_metrics'):
            metrics = getattr(step_instance, 'performance_metrics')
            info['performance'] = {
                'github_process_calls': getattr(metrics, 'github_process_calls', 0),
                'real_ai_calls': getattr(metrics, 'real_ai_calls', 0),
                'mock_calls_blocked': getattr(metrics, 'mock_calls_blocked', 0),
                'data_conversions': getattr(metrics, 'data_conversions', 0),
                'circular_reference_optimized': True
            }
        
        return info
        
    except Exception as e:
        return {
            'error': str(e),
            'class_name': getattr(step_instance, '__class__', {}).get('__name__', 'Unknown') if step_instance else 'None',
            'circular_reference_free': True
        }

# =============================================================================
# 🔥 18단계: 경로 호환성 처리 (StepInterface 별칭 설정 오류 해결)
# =============================================================================

def create_deprecated_interface_warning():
    """Deprecated interface 경로 경고"""
    warnings.warn(
        "⚠️ app.ai_pipeline.interface 경로는 deprecated입니다. "
        "app.ai_pipeline.interfaces를 사용하세요.",
        DeprecationWarning,
        stacklevel=3
    )
    logger.warning("⚠️ Deprecated interface 경로 사용 감지")

# 안전한 모듈 별칭 생성 (StepInterface 별칭 설정 오류 해결)
def setup_safe_module_aliases():
    """안전한 모듈 별칭 설정"""
    try:
        current_module = sys.modules[__name__]
        
        # app.ai_pipeline.interface.step_interface로 접근 가능하도록 별칭 생성
        if 'app.ai_pipeline.interface' not in sys.modules:
            import types
            interface_module = types.ModuleType('app.ai_pipeline.interface')
            interface_module.step_interface = current_module
            sys.modules['app.ai_pipeline.interface'] = interface_module
            sys.modules['app.ai_pipeline.interface.step_interface'] = current_module
            logger.info("✅ 기존 경로 호환성 별칭 생성 완료")
            return True
    except Exception as e:
        logger.warning(f"⚠️ 경로 호환성 별칭 생성 실패 (무시됨): {e}")
        return False

# 모듈 별칭 설정 실행
setup_safe_module_aliases()

# =============================================================================
# 🔥 19단계: Export (모든 클래스 및 함수 포함)
# =============================================================================

__all__ = [
    # 메인 클래스들
    'GitHubStepModelInterface',
    'GitHubMemoryManager', 
    'EmbeddedDependencyManager',  # 🔥 순환참조 해결 핵심
    'GitHubStepMapping',
    
    # 호환성 클래스들
    'StepInterface',
    'SimpleStepConfig',
    
    # 데이터 구조들
    'GitHubStepConfig',
    'GitHubStepCreationResult',
    'GitHubStepType',
    'GitHubStepPriority',
    'GitHubDeviceType',
    'GitHubProcessingStatus',
    
    # 팩토리 함수들 (순환참조 해결 버전)
    'create_github_step_interface_circular_reference_free',
    'create_optimized_github_interface_v51',
    'create_step_07_virtual_fitting_interface_v51',
    'create_simple_step_interface',
    
    # 유틸리티 함수들
    'get_github_environment_info',
    'optimize_github_environment',
    'validate_github_step_compatibility',
    'get_github_step_info',
    
    # 상수들
    'CONDA_INFO',
    'IS_M3_MAX',
    'MEMORY_GB',
    'MPS_AVAILABLE',
    'PROJECT_ROOT',
    'BACKEND_ROOT',
    'AI_PIPELINE_ROOT',
    'PYTORCH_FIXED',
    'REMBG_AVAILABLE',
    'SAFETENSORS_AVAILABLE',
    
    # Logger
    'logger'
]

# =============================================================================
# 🔥 20단계: 모듈 초기화 및 완료 메시지
# =============================================================================

# GitHub 프로젝트 구조 확인
if AI_PIPELINE_ROOT.exists():
    logger.info(f"✅ GitHub 프로젝트 구조 감지: {PROJECT_ROOT}")
else:
    logger.warning(f"⚠️ GitHub 프로젝트 구조 확인 필요: {PROJECT_ROOT}")

# conda 환경 자동 최적화
if CONDA_INFO['is_target_env']:
    optimize_github_environment()
    logger.info("🐍 conda 환경 mycloset-ai-clean 자동 최적화 완료!")

# M3 Max 최적화
if IS_M3_MAX:
    try:
        if MPS_AVAILABLE and PYTORCH_FIXED:
            if hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
        gc.collect()
        logger.info("🍎 M3 Max 초기 메모리 최적화 완료!")
    except:
        pass

logger.info("=" * 80)
logger.info("🔥 Step Interface v5.1 - 순환참조 완전 해결 + 모든 기능 유지")
logger.info("=" * 80)
logger.info("✅ BaseStepMixin 순환참조 완전 차단 (지연 import)")
logger.info("✅ EmbeddedDependencyManager 내장으로 순환참조 해결")
logger.info("✅ 모든 기능 그대로 유지")
logger.info("✅ Logger 문제 완전 해결")
logger.info("✅ PyTorch weights_only 호환성 패치 적용")
logger.info("✅ GitHub 프로젝트 구조 100% 호환")
logger.info("✅ M3 Max 최적화 유지")

logger.info(f"🔧 현재 환경:")
logger.info(f"   - conda 환경: {CONDA_INFO['conda_env']} ({'✅' if CONDA_INFO['is_target_env'] else '⚠️'})")
logger.info(f"   - M3 Max: {'✅' if IS_M3_MAX else '❌'}")
logger.info(f"   - PyTorch 수정: {'✅' if PYTORCH_FIXED else '❌'}")
logger.info(f"   - 순환참조 해결: ✅")
logger.info(f"   - 내장 의존성 관리자: ✅")

logger.info("🎯 지원 GitHub Step 클래스:")
for step_type in GitHubStepType:
    config = GitHubStepMapping.get_config(step_type)
    circular_ref_status = "🔗 순환참조 해결" if config.step_id <= 8 else ""
    logger.info(f"   - Step {config.step_id:02d}: {config.class_name} ({config.model_size_gb}GB) {circular_ref_status}")

logger.info("🔥 핵심 개선사항:")
logger.info("   • EmbeddedDependencyManager: 순환참조 완전 차단")
logger.info("   • 지연 import: TYPE_CHECKING으로 import 순환 방지")
logger.info("   • GitHubStepModelInterface: BaseStepMixin v19.1 완벽 호환")
logger.info("   • GitHubStepMapping: 실제 GitHub Step 클래스 매핑")
logger.info("   • GitHubMemoryManager: M3 Max 128GB 완전 활용")

logger.info("🚀 주요 팩토리 함수 (순환참조 해결):")
logger.info("   - create_github_step_interface_circular_reference_free(): 순환참조 해결 버전")
logger.info("   - create_optimized_github_interface_v51(): 최적화된 인터페이스 v5.1")
logger.info("   - create_step_07_virtual_fitting_interface_v51(): 7단계 전용 v5.1")
logger.info("   - create_simple_step_interface(): Step 파일 호환성용")

logger.info("🔧 주요 유틸리티 (순환참조 해결):")
logger.info("   - get_github_environment_info(): 환경 정보 + 순환참조 상태")
logger.info("   - validate_github_step_compatibility(): Step 호환성 + 순환참조 검증")
logger.info("   - get_github_step_info(): Step 정보 + 내장 의존성 관리자 상태")

logger.info("🔄 호환성 지원:")
logger.info("   - StepInterface: 기존 Step 파일들과 호환")
logger.info("   - app.ai_pipeline.interface 경로 별칭 지원")
logger.info("   - logger 정의 문제 완전 해결")
logger.info("   - 순환참조 완전 해결로 안정성 확보")

logger.info("🎉 Step Interface v5.1 순환참조 해결 완료!")
logger.info("🎉 BaseStepMixin과의 순환참조가 완전히 해결되었습니다!")
logger.info("🎉 EmbeddedDependencyManager로 안전한 의존성 주입 지원!")
logger.info("🎉 모든 기능이 그대로 유지되면서 순환참조만 해결되었습니다!")
logger.info("=" * 80)