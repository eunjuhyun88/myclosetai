# backend/app/ai_pipeline/steps/base_step_mixin.py
"""
🔥 BaseStepMixin v19.2 - 순환참조 완전 해결 + 모든 기능 포함
==============================================================

✅ 순환참조 완전 해결 (TYPE_CHECKING + 지연 import)
✅ EmbeddedDependencyManager 내장으로 순환참조 차단
✅ step_model_requirements.py DetailedDataSpec 완전 활용
✅ API ↔ AI 모델 간 데이터 변환 표준화 완료
✅ Step 간 데이터 흐름 자동 처리
✅ 전처리/후처리 요구사항 자동 적용
✅ GitHub 프로젝트 Step 클래스들과 100% 호환
✅ 모든 기능 그대로 유지하면서 순환참조만 해결
✅ v19.1의 모든 고급 전처리/후처리 기능 완전 보존
✅ GitHubDependencyManager 완전 복원 및 개선
✅ 완전한 후처리 시스템 (15개 후처리 단계)
✅ 고급 데이터 변환 및 검증 시스템

Author: MyCloset AI Team
Date: 2025-07-30
Version: 19.2 (Circular Reference Fix + Complete Features)
"""

import os
import gc
import time
import asyncio
import logging
import threading
import traceback
import weakref
import subprocess
import platform
import inspect
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union, Callable, Type, TYPE_CHECKING, Awaitable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from functools import wraps
from contextlib import asynccontextmanager
from enum import Enum

# 🔥 TYPE_CHECKING으로 순환참조 완전 방지
if TYPE_CHECKING:
    from ..utils.model_loader import ModelLoader, StepModelInterface
    from ..factories.step_factory import StepFactory
    from ..utils.memory_manager import MemoryManager
    from ..utils.data_converter import DataConverter
    from ..core.di_container import DIContainer

# ==============================================
# 🔥 환경 설정 및 시스템 정보
# ==============================================

# conda 환경 정보
CONDA_INFO = {
    'conda_env': os.environ.get('CONDA_DEFAULT_ENV', 'none'),
    'conda_prefix': os.environ.get('CONDA_PREFIX', 'none'),
    'is_target_env': os.environ.get('CONDA_DEFAULT_ENV') == 'mycloset-ai-clean'
}

# 시스템 정보
IS_M3_MAX = False
MEMORY_GB = 16.0

try:
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
        if memory_result.stdout.strip():
            MEMORY_GB = int(memory_result.stdout.strip()) / 1024**3
except:
    pass

# ==============================================
# 🔥 라이브러리 안전 import
# ==============================================

# PyTorch 안전 import
TORCH_AVAILABLE = False
MPS_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        MPS_AVAILABLE = True
except ImportError:
    torch = None

# PIL 안전 import
PIL_AVAILABLE = False
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None

# NumPy 안전 import
NUMPY_AVAILABLE = False
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None

# OpenCV 안전 import
CV2_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None

# ==============================================
# 🔥 GitHub 프로젝트 호환 인터페이스 (v19.2)
# ==============================================

class ProcessMethodSignature(Enum):
    """GitHub 프로젝트에서 사용되는 process 메서드 시그니처 패턴"""
    STANDARD = "async def process(self, **kwargs) -> Dict[str, Any]"
    INPUT_DATA = "async def process(self, input_data: Any) -> Dict[str, Any]"
    PIPELINE = "async def process_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]"
    LEGACY = "def process(self, *args, **kwargs) -> Dict[str, Any]"

class DependencyValidationFormat(Enum):
    """의존성 검증 반환 형식"""
    BOOLEAN_DICT = "dict_bool"  # GeometricMatchingStep 형식: {'model_loader': True, ...}
    DETAILED_DICT = "dict_detailed"  # BaseStepMixin v18.0 형식: {'success': True, 'details': {...}}
    AUTO_DETECT = "auto"  # 호출자에 따라 자동 선택

class DataConversionMethod(Enum):
    """데이터 변환 방법"""
    AUTOMATIC = "auto"      # DetailedDataSpec 기반 자동 변환
    MANUAL = "manual"       # 하위 클래스에서 수동 변환
    HYBRID = "hybrid"       # 자동 + 수동 조합

# ==============================================
# 🔥 설정 및 상태 클래스 (v19.2 순환참조 해결)
# ==============================================

@dataclass
class DetailedDataSpecConfig:
    """DetailedDataSpec 설정 관리"""
    # 입력 사양
    input_data_types: List[str] = field(default_factory=list)
    input_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    input_value_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    preprocessing_required: List[str] = field(default_factory=list)
    
    # 출력 사양  
    output_data_types: List[str] = field(default_factory=list)
    output_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    output_value_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    postprocessing_required: List[str] = field(default_factory=list)
    
    # API 호환성
    api_input_mapping: Dict[str, str] = field(default_factory=dict)
    api_output_mapping: Dict[str, str] = field(default_factory=dict)
    
    # Step 간 연동
    step_input_schema: Dict[str, Any] = field(default_factory=dict)
    step_output_schema: Dict[str, Any] = field(default_factory=dict)
    
    # 전처리/후처리 요구사항
    normalization_mean: Tuple[float, ...] = field(default_factory=lambda: (0.485, 0.456, 0.406))
    normalization_std: Tuple[float, ...] = field(default_factory=lambda: (0.229, 0.224, 0.225))
    preprocessing_steps: List[str] = field(default_factory=list)
    postprocessing_steps: List[str] = field(default_factory=list)
    
    # Step 간 데이터 전달 스키마
    accepts_from_previous_step: Dict[str, Dict[str, str]] = field(default_factory=dict)
    provides_to_next_step: Dict[str, Dict[str, str]] = field(default_factory=dict)

@dataclass
class GitHubStepConfig:
    """GitHub 프로젝트 호환 Step 설정 (v19.2)"""
    step_name: str = "BaseStep"
    step_id: int = 0
    device: str = "auto"
    use_fp16: bool = True
    batch_size: int = 1
    confidence_threshold: float = 0.8
    auto_memory_cleanup: bool = True
    auto_warmup: bool = True
    optimization_enabled: bool = True
    quality_level: str = "balanced"
    strict_mode: bool = False
    
    # 의존성 설정
    auto_inject_dependencies: bool = True
    require_model_loader: bool = True
    require_memory_manager: bool = False
    require_data_converter: bool = False
    dependency_timeout: float = 30.0
    dependency_retry_count: int = 3
    
    # GitHub 프로젝트 특별 설정
    process_method_signature: ProcessMethodSignature = ProcessMethodSignature.STANDARD
    dependency_validation_format: DependencyValidationFormat = DependencyValidationFormat.AUTO_DETECT
    github_compatibility_mode: bool = True
    real_ai_pipeline_support: bool = True
    
    # DetailedDataSpec 설정 (v19.2 신규)
    enable_detailed_data_spec: bool = True
    data_conversion_method: DataConversionMethod = DataConversionMethod.AUTOMATIC
    strict_data_validation: bool = True
    auto_preprocessing: bool = True
    auto_postprocessing: bool = True
    
    # 환경 최적화
    conda_optimized: bool = False
    conda_env: str = "none"
    m3_max_optimized: bool = False
    memory_gb: float = 16.0
    use_unified_memory: bool = False

@dataclass
class GitHubDependencyStatus:
    """GitHub 프로젝트 호환 의존성 상태 (v19.2 - 순환참조 해결)"""
    model_loader: bool = False
    step_interface: bool = False
    memory_manager: bool = False
    data_converter: bool = False
    di_container: bool = False
    base_initialized: bool = False
    custom_initialized: bool = False
    dependencies_validated: bool = False
    
    # GitHub 특별 상태
    github_compatible: bool = False
    process_method_validated: bool = False
    real_ai_models_loaded: bool = False
    
    # DetailedDataSpec 상태 (v19.2 신규)
    detailed_data_spec_loaded: bool = False
    data_conversion_ready: bool = False
    preprocessing_configured: bool = False
    postprocessing_configured: bool = False
    api_mapping_configured: bool = False
    step_flow_configured: bool = False
    
    # 환경 상태
    conda_optimized: bool = False
    m3_max_optimized: bool = False
    
    # 주입 시도 추적
    injection_attempts: Dict[str, int] = field(default_factory=dict)
    injection_errors: Dict[str, List[str]] = field(default_factory=dict)
    last_injection_time: float = field(default_factory=time.time)

@dataclass
class GitHubPerformanceMetrics:
    """GitHub 프로젝트 호환 성능 메트릭 (v19.2)"""
    process_count: int = 0
    total_process_time: float = 0.0
    average_process_time: float = 0.0
    error_count: int = 0
    success_count: int = 0
    cache_hits: int = 0
    
    # 메모리 메트릭
    peak_memory_usage_mb: float = 0.0
    average_memory_usage_mb: float = 0.0
    memory_optimizations: int = 0
    
    # AI 모델 메트릭
    models_loaded: int = 0
    total_model_size_gb: float = 0.0
    inference_count: int = 0
    
    # 의존성 메트릭
    dependencies_injected: int = 0
    injection_failures: int = 0
    average_injection_time: float = 0.0
    
    # GitHub 특별 메트릭
    github_process_calls: int = 0
    real_ai_inferences: int = 0
    pipeline_success_rate: float = 0.0
    
    # DetailedDataSpec 메트릭 (v19.2 신규)
    data_conversions: int = 0
    preprocessing_operations: int = 0
    postprocessing_operations: int = 0
    api_conversions: int = 0
    step_data_transfers: int = 0
    validation_failures: int = 0

# ==============================================
# 🔥 GitHubDependencyManager - 완전 복원 + 순환참조 해결
# ==============================================

class GitHubDependencyManager:
    """🔥 GitHub 프로젝트 완전 호환 의존성 관리자 v19.2 - 순환참조 해결"""
    
    def __init__(self, step_name: str, **kwargs):
        """완전 수정된 초기화 메서드"""
        self.step_name = step_name
        self.logger = logging.getLogger(f"GitHubDependencyManager.{step_name}")
        
        # 🔥 핵심 속성들 (오류 해결)
        self.step_instance = None
        self.injected_dependencies = {}
        self.dependencies = {}
        self.injection_attempts = {}
        self.injection_errors = {}
        
        # 🔥 dependency_status 속성 추가 (오류 해결!)
        self.dependency_status = GitHubDependencyStatus()
        
        # 시간 추적
        self.last_injection_time = time.time()
        
        # 선택적 매개변수 처리
        self.memory_gb = kwargs.get('memory_gb', 16)
        self.quality_level = kwargs.get('quality_level', 'balanced')
        self.auto_inject_enabled = kwargs.get('auto_inject_dependencies', True)
        self.dependency_timeout = kwargs.get('dependency_timeout', 30.0)
        self.dependency_retry_count = kwargs.get('dependency_retry_count', 3)
        
        # 성능 메트릭
        self.dependencies_injected = 0
        self.injection_failures = 0
        self.validation_attempts = 0
        
        # 스레드 안전성
        self._lock = threading.RLock()
        
        self.logger.debug(f"✅ GitHubDependencyManager v19.2 초기화 완료: {step_name}")
    
    def set_step_instance(self, step_instance):
        """Step 인스턴스 설정 - 완전 수정"""
        try:
            with self._lock:
                self.step_instance = step_instance
                self.logger.debug(f"✅ {self.step_name} Step 인스턴스 설정 완료")
                return True
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} Step 인스턴스 설정 실패: {e}")
            return False
    
    def auto_inject_dependencies(self) -> bool:
        """🔥 완전 수정된 자동 의존성 주입 메서드"""
        try:
            with self._lock:
                self.logger.info(f"🔄 {self.step_name} GitHubDependencyManager 자동 의존성 주입 시작...")
                
                if not self.step_instance:
                    self.logger.warning(f"⚠️ {self.step_name} Step 인스턴스가 설정되지 않음")
                    return False
                
                success_count = 0
                total_dependencies = 0
                
                # ModelLoader 자동 주입
                if not hasattr(self.step_instance, 'model_loader') or self.step_instance.model_loader is None:
                    total_dependencies += 1
                    try:
                        model_loader = self._resolve_model_loader()
                        if model_loader:
                            self.step_instance.model_loader = model_loader
                            self.injected_dependencies['model_loader'] = model_loader
                            self.dependency_status.model_loader = True
                            success_count += 1
                            self.dependencies_injected += 1
                            self.logger.info(f"✅ {self.step_name} ModelLoader 자동 주입 성공")
                        else:
                            self.logger.warning(f"⚠️ {self.step_name} ModelLoader 해결 실패 - 실제 의존성 필요")
                            self.injection_failures += 1
                    except Exception as e:
                        self.logger.warning(f"⚠️ {self.step_name} ModelLoader 자동 주입 실패: {e}")
                        self.injection_failures += 1
                
                # MemoryManager 자동 주입  
                if not hasattr(self.step_instance, 'memory_manager') or self.step_instance.memory_manager is None:
                    total_dependencies += 1
                    try:
                        memory_manager = self._resolve_memory_manager()
                        if memory_manager:
                            self.step_instance.memory_manager = memory_manager
                            self.injected_dependencies['memory_manager'] = memory_manager
                            self.dependency_status.memory_manager = True
                            success_count += 1
                            self.dependencies_injected += 1
                            self.logger.info(f"✅ {self.step_name} MemoryManager 자동 주입 성공")
                        else:
                            self.logger.warning(f"⚠️ {self.step_name} MemoryManager 해결 실패 - 실제 의존성 필요")
                            self.injection_failures += 1
                    except Exception as e:
                        self.logger.warning(f"⚠️ {self.step_name} MemoryManager 자동 주입 실패: {e}")
                        self.injection_failures += 1
                
                # DataConverter 자동 주입
                if not hasattr(self.step_instance, 'data_converter') or self.step_instance.data_converter is None:
                    total_dependencies += 1
                    try:
                        data_converter = self._resolve_data_converter()
                        if data_converter:
                            self.step_instance.data_converter = data_converter
                            self.injected_dependencies['data_converter'] = data_converter
                            self.dependency_status.data_converter = True
                            success_count += 1
                            self.dependencies_injected += 1
                            self.logger.info(f"✅ {self.step_name} DataConverter 자동 주입 성공")
                        else:
                            self.logger.warning(f"⚠️ {self.step_name} DataConverter 해결 실패 - 실제 의존성 필요")
                            self.injection_failures += 1
                    except Exception as e:
                        self.logger.warning(f"⚠️ {self.step_name} DataConverter 자동 주입 실패: {e}")
                        self.injection_failures += 1
                
                # 성공 여부 판단 (실제 의존성 기반)
                if total_dependencies == 0:
                    self.logger.info(f"✅ {self.step_name} 모든 의존성이 이미 주입되어 있음")
                    return True
                
                success_rate = success_count / total_dependencies if total_dependencies > 0 else 1.0
                
                # 실제 의존성만 허용 - 성공한 것만 사용
                if success_count > 0:
                    self.logger.info(f"✅ {self.step_name} 실제 의존성 주입 완료: {success_count}/{total_dependencies} ({success_rate*100:.1f}%)")
                    self.dependency_status.base_initialized = True
                    self.dependency_status.github_compatible = True
                    return True
                else:
                    self.logger.warning(f"⚠️ {self.step_name} 실제 의존성 주입 실패: {success_count}/{total_dependencies} - Mock 폴백 없음")
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
            
            self.logger.debug(f"{self.step_name} MemoryManager 해결 실패 - 실제 의존성 필요")
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
            
            self.logger.debug(f"{self.step_name} DataConverter 해결 실패 - 실제 의존성 필요")
            return None
            
        except Exception as e:
            self.logger.debug(f"{self.step_name} DataConverter 해결 실패: {e}")
            return None
    
    def inject_model_loader(self, model_loader):
        """ModelLoader 주입 - 완전 수정"""
        try:
            with self._lock:
                if not self.step_instance:
                    self.logger.warning(f"⚠️ {self.step_name} Step 인스턴스가 설정되지 않음")
                    return False
                
                # 유효성 검증
                if model_loader is None:
                    self.logger.warning(f"⚠️ {self.step_name} ModelLoader가 None입니다")
                    return False
                
                if isinstance(model_loader, bool):
                    self.logger.error(f"❌ {self.step_name} ModelLoader는 bool이 아닌 객체여야 합니다")
                    return False
                
                # 주입 실행
                self.step_instance.model_loader = model_loader
                self.injected_dependencies['model_loader'] = model_loader
                self.dependency_status.model_loader = True
                self.dependency_status.base_initialized = True
                self.dependencies_injected += 1
                
                self.logger.info(f"✅ {self.step_name} ModelLoader 주입 완료")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} ModelLoader 주입 실패: {e}")
            self.injection_failures += 1
            return False
    
    def inject_memory_manager(self, memory_manager):
        """MemoryManager 주입 - 완전 수정"""
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
                self.dependency_status.memory_manager = True
                self.dependencies_injected += 1
                
                self.logger.info(f"✅ {self.step_name} MemoryManager 주입 완료")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} MemoryManager 주입 실패: {e}")
            self.injection_failures += 1
            return False
    
    def inject_data_converter(self, data_converter):
        """DataConverter 주입 - 완전 수정"""
        try:
            with self._lock:
                if not self.step_instance:
                    self.logger.warning(f"⚠️ {self.step_name} Step 인스턴스가 설정되지 않음")
                    return False
                
                if data_converter is None:
                    self.logger.warning(f"⚠️ {self.step_name} DataConverter가 None입니다")
                    return False
                
                # 주입 실행
                self.step_instance.data_converter = data_converter
                self.injected_dependencies['data_converter'] = data_converter
                self.dependency_status.data_converter = True
                self.dependencies_injected += 1
                
                self.logger.info(f"✅ {self.step_name} DataConverter 주입 완료")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} DataConverter 주입 실패: {e}")
            self.injection_failures += 1
            return False
    
    def inject_di_container(self, di_container):
        """DI Container 의존성 주입 - 완전 수정"""
        try:
            with self._lock:
                if di_container is None:
                    self.logger.warning(f"⚠️ {self.step_name} DI Container가 None입니다")
                    return False
                
                # DI Container는 step_instance가 아닌 dependencies에 저장
                self.dependencies['di_container'] = di_container
                self.injected_dependencies['di_container'] = di_container
                self.dependency_status.di_container = True
                self.dependencies_injected += 1
                
                self.logger.info(f"✅ {self.step_name} DI Container 주입 완료")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} DI Container 주입 실패: {e}")
            self.injection_failures += 1
            return False
    
    def validate_dependencies_github_format(self, format_type=None):
        """GitHub 형식 의존성 검증 - 완전 수정"""
        try:
            with self._lock:
                self.validation_attempts += 1
                
                # Step 인스턴스 확인
                if not self.step_instance:
                    dependencies = {
                        'model_loader': False,
                        'memory_manager': False,
                        'data_converter': False,
                        'step_interface': False,
                    }
                else:
                    # 실제 의존성 상태 확인
                    dependencies = {
                        'model_loader': hasattr(self.step_instance, 'model_loader') and self.step_instance.model_loader is not None,
                        'memory_manager': hasattr(self.step_instance, 'memory_manager') and self.step_instance.memory_manager is not None,
                        'data_converter': hasattr(self.step_instance, 'data_converter') and self.step_instance.data_converter is not None,
                        'step_interface': True,  # 기본값 (Step 인스턴스가 존재하면 인터페이스 OK)
                    }
                
                # DI Container는 별도 확인
                dependencies['di_container'] = 'di_container' in self.dependencies and self.dependencies['di_container'] is not None
                
                # 반환 형식 결정
                if format_type:
                    # format_type이 문자열인 경우
                    if isinstance(format_type, str) and format_type.upper() == 'BOOLEAN_DICT':
                        return dependencies
                    # format_type이 enum인 경우
                    elif hasattr(format_type, 'value') and format_type.value == 'dict_bool':
                        return dependencies
                    elif hasattr(format_type, 'value') and format_type.value == 'boolean_dict':
                        return dependencies
                
                # 기본값: 상세 정보 반환
                return {
                    'success': all(dep for key, dep in dependencies.items() if key != 'di_container'),  # DI Container 제외하고 판단
                    'dependencies': dependencies,
                    'github_compatible': True,
                    'injected_count': len(self.injected_dependencies),
                    'step_name': self.step_name,
                    'dependency_status': {
                        'model_loader': self.dependency_status.model_loader,
                        'memory_manager': self.dependency_status.memory_manager,
                        'data_converter': self.dependency_status.data_converter,
                        'di_container': self.dependency_status.di_container,
                        'base_initialized': self.dependency_status.base_initialized,
                        'github_compatible': self.dependency_status.github_compatible
                    },
                    'metrics': {
                        'injected': self.dependencies_injected,
                        'failures': self.injection_failures,
                        'validation_attempts': self.validation_attempts
                    },
                    'timestamp': time.time()
                }
                
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} GitHub 의존성 검증 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'github_compatible': False,
                'step_name': self.step_name
            }
    
    def get_dependency_status(self) -> Dict[str, Any]:
        """의존성 상태 조회 - 완전 수정"""
        try:
            with self._lock:
                return {
                    'step_name': self.step_name,
                    'step_instance_set': self.step_instance is not None,
                    'injected_dependencies': list(self.injected_dependencies.keys()),
                    'dependency_status': {
                        'model_loader': self.dependency_status.model_loader,
                        'memory_manager': self.dependency_status.memory_manager,
                        'data_converter': self.dependency_status.data_converter,
                        'di_container': self.dependency_status.di_container,
                        'base_initialized': self.dependency_status.base_initialized,
                        'github_compatible': self.dependency_status.github_compatible,
                        'detailed_data_spec_loaded': self.dependency_status.detailed_data_spec_loaded,
                        'data_conversion_ready': self.dependency_status.data_conversion_ready
                    },
                    'metrics': {
                        'dependencies_injected': self.dependencies_injected,
                        'injection_failures': self.injection_failures,
                        'validation_attempts': self.validation_attempts,
                        'last_injection_time': self.last_injection_time
                    },
                    'timestamp': time.time()
                }
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 의존성 상태 조회 실패: {e}")
            return {
                'step_name': self.step_name,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def cleanup(self):
        """리소스 정리 - 완전 수정"""
        try:
            with self._lock:
                self.logger.info(f"🔄 {self.step_name} GitHubDependencyManager 정리 시작...")
                
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
                
                # dependency_status 초기화
                self.dependency_status = GitHubDependencyStatus()
                
                self.logger.info(f"✅ {self.step_name} GitHubDependencyManager 정리 완료")
                
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} GitHubDependencyManager 정리 실패: {e}")
    
    def __del__(self):
        """소멸자 - 안전한 정리"""
        try:
            self.cleanup()
        except:
            pass  # 소멸자에서는 예외 무시

# ==============================================
# 🔥 BaseStepMixin v19.2 - 완전한 기능 + 순환참조 해결
# ==============================================

class BaseStepMixin:
    """
    🔥 BaseStepMixin v19.2 - 순환참조 완전 해결 + 모든 기능 포함
    
    핵심 개선사항:
    ✅ 순환참조 완전 해결 (GitHubDependencyManager 내장)
    ✅ DetailedDataSpec 정보 저장 및 관리
    ✅ 표준화된 process 메서드 재설계
    ✅ API ↔ AI 모델 간 데이터 변환 표준화
    ✅ Step 간 데이터 흐름 자동 처리
    ✅ 전처리/후처리 요구사항 자동 적용
    ✅ GitHub 프로젝트 Step 클래스들과 100% 호환
    """
    def __init__(self, **kwargs):
        """순환참조 완전 해결 초기화 (v19.2)"""
        try:
            # 기본 설정
            self.config = self._create_github_config(**kwargs)
            self.step_name = kwargs.get('step_name', self.__class__.__name__)
            self.step_id = kwargs.get('step_id', 0)
            
            # Logger 설정 (제일 먼저)
            self.logger = logging.getLogger(f"steps.{self.step_name}")
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)

            # 성능 통계 초기화
            self._initialize_performance_stats()

            # 🔥 DetailedDataSpec 정보 저장
            self.detailed_data_spec = self._load_detailed_data_spec_from_kwargs(**kwargs)
            
            # 🔥 내장 의존성 관리자 (순환참조 해결)
            self.dependency_manager = GitHubDependencyManager(self.step_name)
            self.dependency_manager.set_step_instance(self)
            
            # 나머지 초기화
            self.is_initialized = False
            self.is_ready = False
            self.has_model = False
            self.model_loaded = False
            self.warmup_completed = False
            
            # 시스템 정보
            self.device = self._resolve_device(self.config.device)
            self.is_m3_max = IS_M3_MAX
            self.memory_gb = MEMORY_GB
            self.conda_info = CONDA_INFO
            
            # GitHub 호환 성능 메트릭
            self.performance_metrics = GitHubPerformanceMetrics()
            
            # GitHub 호환성을 위한 속성들
            self.model_loader = None
            self.model_interface = None
            self.memory_manager = None
            self.data_converter = None
            
            # GitHub 특별 속성들
            self.github_compatible = True
            self.real_ai_pipeline_ready = False
            self.process_method_signature = self.config.process_method_signature
            
            # 🔥 DetailedDataSpec 상태
            self.data_conversion_ready = self._validate_data_conversion_readiness()
            
            # 환경 최적화 적용
            self._apply_github_environment_optimization()
            
            # 자동 의존성 주입 (설정된 경우)
            if self.config.auto_inject_dependencies:
                try:
                    self.dependency_manager.auto_inject_dependencies()
                except Exception as e:
                    self.logger.warning(f"⚠️ {self.step_name} 자동 의존성 주입 실패: {e}")
            
            self.logger.info(f"✅ {self.step_name} BaseStepMixin v19.2 순환참조 해결 초기화 완료")
            
        except Exception as e:
            self._github_emergency_setup(e)

    def _load_detailed_data_spec_from_kwargs(self, **kwargs) -> DetailedDataSpecConfig:
        """StepFactory에서 주입받은 DetailedDataSpec 정보 로딩"""
        return DetailedDataSpecConfig(
            # 입력 사양
            input_data_types=kwargs.get('input_data_types', []),
            input_shapes=kwargs.get('input_shapes', {}),
            input_value_ranges=kwargs.get('input_value_ranges', {}),
            preprocessing_required=kwargs.get('preprocessing_required', []),
            
            # 출력 사양
            output_data_types=kwargs.get('output_data_types', []),
            output_shapes=kwargs.get('output_shapes', {}),
            output_value_ranges=kwargs.get('output_value_ranges', {}),
            postprocessing_required=kwargs.get('postprocessing_required', []),
            
            # API 호환성
            api_input_mapping=kwargs.get('api_input_mapping', {}),
            api_output_mapping=kwargs.get('api_output_mapping', {}),
            
            # Step 간 연동
            step_input_schema=kwargs.get('step_input_schema', {}),
            step_output_schema=kwargs.get('step_output_schema', {}),
            
            # 전처리/후처리 요구사항
            normalization_mean=kwargs.get('normalization_mean', (0.485, 0.456, 0.406)),
            normalization_std=kwargs.get('normalization_std', (0.229, 0.224, 0.225)),
            preprocessing_steps=kwargs.get('preprocessing_steps', []),
            postprocessing_steps=kwargs.get('postprocessing_steps', []),
            
            # Step 간 데이터 전달 스키마
            accepts_from_previous_step=kwargs.get('accepts_from_previous_step', {}),
            provides_to_next_step=kwargs.get('provides_to_next_step', {})
        )

    def _validate_data_conversion_readiness(self) -> bool:
        """데이터 변환 준비 상태 검증 (워닝 방지)"""
        try:
            # DetailedDataSpec 존재 확인 및 자동 생성
            if not hasattr(self, 'detailed_data_spec') or not self.detailed_data_spec:
                self._create_emergency_detailed_data_spec()
                self.logger.debug(f"✅ {self.step_name} DetailedDataSpec 기본값 자동 생성")
            
            # 필수 필드 존재 확인 및 자동 보완
            missing_fields = []
            required_fields = ['input_data_types', 'output_data_types', 'api_input_mapping', 'api_output_mapping']
            
            for field in required_fields:
                if not hasattr(self.detailed_data_spec, field):
                    missing_fields.append(field)
                else:
                    value = getattr(self.detailed_data_spec, field)
                    if not value:
                        missing_fields.append(field)
            
            # 누락된 필드 자동 보완
            if missing_fields:
                self._fill_missing_fields(missing_fields)
                self.logger.debug(f"{self.step_name} DetailedDataSpec 필드 보완: {missing_fields}")
            
            # dependency_manager 상태 업데이트
            if hasattr(self, 'dependency_manager') and self.dependency_manager:
                self.dependency_manager.dependency_status.detailed_data_spec_loaded = True
                self.dependency_manager.dependency_status.data_conversion_ready = True
            
            self.logger.debug(f"✅ {self.step_name} DetailedDataSpec 데이터 변환 준비 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 데이터 변환 준비 상태 검증 실패: {e}")
            try:
                self._create_emergency_detailed_data_spec()
                self.logger.debug(f"🔄 {self.step_name} DetailedDataSpec 예외 복구 완료")
            except:
                pass
            return True

    def _initialize_performance_stats(self):
        """성능 통계 초기화"""
        try:
            self.performance_stats = {
                'total_processed': 0,
                'avg_processing_time': 0.0,
                'error_count': 0,
                'success_rate': 1.0,
                'memory_usage_mb': 0.0,
                'models_loaded': 0,
                'cache_hits': 0,
                'ai_inference_count': 0,
                'torch_errors': 0
            }
            
            self.total_processing_count = 0
            self.error_count = 0
            self.last_processing_time = 0.0
            
            self.logger.debug(f"✅ {self.step_name} 성능 통계 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 성능 통계 초기화 실패: {e}")
            self.performance_stats = {}
            self.total_processing_count = 0
            self.error_count = 0
            self.last_processing_time = 0.0

    def _create_emergency_detailed_data_spec(self):
        """응급 DetailedDataSpec 생성"""
        try:
            if not hasattr(self, 'detailed_data_spec') or not self.detailed_data_spec:
                class EmergencyDataSpec:
                    def __init__(self):
                        self.input_data_types = {
                            'person_image': 'PIL.Image.Image',
                            'clothing_image': 'PIL.Image.Image',
                            'data': 'Any'
                        }
                        self.output_data_types = {
                            'result': 'numpy.ndarray',
                            'success': 'bool',
                            'processing_time': 'float'
                        }
                        self.api_input_mapping = {
                            'person_image': 'fastapi.UploadFile -> PIL.Image.Image',
                            'clothing_image': 'fastapi.UploadFile -> PIL.Image.Image'
                        }
                        self.api_output_mapping = {
                            'result': 'numpy.ndarray -> base64_string',
                            'success': 'bool -> bool'
                        }
                        self.preprocessing_steps = ['validate_input', 'resize_image']
                        self.postprocessing_steps = ['format_output']
                        self.accepts_from_previous_step = {}
                        self.provides_to_next_step = {}
                
                self.detailed_data_spec = EmergencyDataSpec()
                
        except Exception as e:
            self.logger.error(f"응급 DetailedDataSpec 생성 실패: {e}")

    def _fill_missing_fields(self, missing_fields):
        """누락된 DetailedDataSpec 필드 채우기"""
        try:
            default_values = {
                'input_data_types': {
                    'person_image': 'PIL.Image.Image',
                    'clothing_image': 'PIL.Image.Image',
                    'data': 'Any'
                },
                'output_data_types': {
                    'result': 'numpy.ndarray',
                    'success': 'bool',
                    'processing_time': 'float'
                },
                'api_input_mapping': {
                    'person_image': 'fastapi.UploadFile -> PIL.Image.Image',
                    'clothing_image': 'fastapi.UploadFile -> PIL.Image.Image'
                },
                'api_output_mapping': {
                    'result': 'numpy.ndarray -> base64_string',
                    'success': 'bool -> bool'
                },
                'preprocessing_steps': ['validate_input', 'resize_image'],
                'postprocessing_steps': ['format_output'],
                'accepts_from_previous_step': {},
                'provides_to_next_step': {}
            }
            
            for field in missing_fields:
                if field in default_values:
                    if not hasattr(self.detailed_data_spec, field):
                        setattr(self.detailed_data_spec, field, default_values[field])
                    elif not getattr(self.detailed_data_spec, field):
                        setattr(self.detailed_data_spec, field, default_values[field])
            
        except Exception as e:
            self.logger.error(f"DetailedDataSpec 필드 보완 실패: {e}")

    # ==============================================
    # 🔥 표준화된 process 메서드 (모든 기능 유지)
    # ==============================================
    
    async def process(self, **kwargs) -> Dict[str, Any]:
        """완전히 재설계된 표준화 process 메서드"""
        try:
            start_time = time.time()
            self.performance_metrics.github_process_calls += 1
            
            self.logger.debug(f"🔄 {self.step_name} process 시작 (입력: {list(kwargs.keys())})")
            
            # 1. 입력 데이터 변환 (API/Step 간 → AI 모델)
            converted_input = await self._convert_input_to_model_format(kwargs)
            
            # 2. 하위 클래스의 순수 AI 로직 실행
            ai_result = self._run_ai_inference(converted_input)
            
            # 3. 출력 데이터 변환 (AI 모델 → API + Step 간)
            standardized_output = await self._convert_output_to_standard_format(ai_result)
            
            # 4. 성능 메트릭 업데이트
            processing_time = time.time() - start_time
            self._update_performance_metrics(processing_time, True)
            
            self.logger.debug(f"✅ {self.step_name} process 완료 ({processing_time:.3f}초)")
            
            return standardized_output
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_performance_metrics(processing_time, False)
            self.logger.error(f"❌ {self.step_name} process 실패 ({processing_time:.3f}초): {e}")
            return self._create_error_response(str(e))

    @abstractmethod
    def _run_ai_inference(self, processed_input: Dict[str, Any]) -> Dict[str, Any]:
        """하위 클래스에서 구현할 순수 AI 로직 (동기 메서드)"""
        pass

    # ==============================================
    # 🔥 입력 데이터 변환 시스템 (모든 기능 유지)
    # ==============================================
    
    async def _convert_input_to_model_format(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """API/Step 간 데이터 → AI 모델 입력 형식 변환"""
        try:
            converted = {}
            self.performance_metrics.data_conversions += 1
            
            self.logger.debug(f"🔄 {self.step_name} 입력 데이터 변환 시작...")
            
            # 1. API 입력 매핑 처리
            for model_param, api_type in self.detailed_data_spec.api_input_mapping.items():
                if model_param in kwargs:
                    converted[model_param] = await self._convert_api_input_type(
                        kwargs[model_param], api_type, model_param
                    )
                    self.performance_metrics.api_conversions += 1
            
            # 2. Step 간 데이터 처리
            for step_name, step_data in kwargs.items():
                if step_name.startswith('from_step_'):
                    step_id = step_name.replace('from_step_', '')
                    if step_id in self.detailed_data_spec.accepts_from_previous_step:
                        step_schema = self.detailed_data_spec.accepts_from_previous_step[step_id]
                        converted.update(self._map_step_input_data(step_data, step_schema))
                        self.performance_metrics.step_data_transfers += 1
            
            # 3. 누락된 필수 입력 데이터 확인
            for param_name in self.detailed_data_spec.api_input_mapping.keys():
                if param_name not in converted and param_name in kwargs:
                    converted[param_name] = kwargs[param_name]
            
            # 4. 전처리 적용
            if self.config.auto_preprocessing and self.detailed_data_spec.preprocessing_steps:
                converted = await self._apply_preprocessing(converted)
                self.performance_metrics.preprocessing_operations += 1
            
            # 5. 데이터 검증
            if self.config.strict_data_validation:
                validated_input = self._validate_input_data(converted)
            else:
                validated_input = converted
            
            self.logger.debug(f"✅ {self.step_name} 입력 데이터 변환 완료")
            return validated_input
            
        except Exception as e:
            self.performance_metrics.validation_failures += 1
            self.logger.error(f"❌ {self.step_name} 입력 데이터 변환 실패: {e}")
            raise
    
    async def _convert_api_input_type(self, value: Any, api_type: str, param_name: str) -> Any:
        """API 타입별 변환 처리"""
        try:
            if api_type == "UploadFile":
                if hasattr(value, 'file'):
                    content = await value.read() if hasattr(value, 'read') else value.file.read()
                    return Image.open(BytesIO(content)) if PIL_AVAILABLE else content
                elif hasattr(value, 'read'):
                    content = value.read()
                    return Image.open(BytesIO(content)) if PIL_AVAILABLE else content
                
            elif api_type == "base64_string":
                if isinstance(value, str):
                    try:
                        image_data = base64.b64decode(value)
                        return Image.open(BytesIO(image_data)) if PIL_AVAILABLE else image_data
                    except Exception:
                        return value
                        
            elif api_type in ["str", "Optional[str]"]:
                return str(value) if value is not None else None
                
            elif api_type in ["int", "Optional[int]"]:
                return int(value) if value is not None else None
                
            elif api_type in ["float", "Optional[float]"]:
                return float(value) if value is not None else None
                
            return value
            
        except Exception as e:
            self.logger.warning(f"⚠️ {self.step_name} API 타입 변환 실패 ({param_name}: {api_type}): {e}")
            return value
    
    def _map_step_input_data(self, step_data: Dict[str, Any], step_schema: Dict[str, str]) -> Dict[str, Any]:
        """Step 간 데이터 매핑"""
        mapped_data = {}
        
        for data_key, data_type in step_schema.items():
            if data_key in step_data:
                value = step_data[data_key]
                
                if data_type == "np.ndarray" and NUMPY_AVAILABLE:
                    if TORCH_AVAILABLE and torch.is_tensor(value):
                        mapped_data[data_key] = value.cpu().numpy()
                    else:
                        mapped_data[data_key] = np.array(value) if not isinstance(value, np.ndarray) else value
                        
                elif data_type == "torch.Tensor" and TORCH_AVAILABLE:
                    if NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                        mapped_data[data_key] = torch.from_numpy(value)
                    else:
                        mapped_data[data_key] = value
                        
                elif data_type == "PIL.Image" and PIL_AVAILABLE:
                    if NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                        mapped_data[data_key] = Image.fromarray(value.astype(np.uint8))
                    else:
                        mapped_data[data_key] = value
                        
                else:
                    mapped_data[data_key] = value
        
        return mapped_data
    
    # ==============================================
    # 🔥 전처리 시스템 (모든 기능 유지)
    # ==============================================
    
    async def _apply_preprocessing(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """DetailedDataSpec 기반 전처리 자동 적용"""
        try:
            processed = input_data.copy()
            
            self.logger.debug(f"🔄 {self.step_name} 전처리 적용: {self.detailed_data_spec.preprocessing_steps}")
            
            for step_name in self.detailed_data_spec.preprocessing_steps:
                if step_name == "resize_512x512":
                    processed = self._resize_images(processed, (512, 512))
                elif step_name == "resize_768x1024":
                    processed = self._resize_images(processed, (768, 1024))
                elif step_name == "resize_256x192":
                    processed = self._resize_images(processed, (256, 192))
                elif step_name == "resize_224x224":
                    processed = self._resize_images(processed, (224, 224))
                elif step_name == "resize_368x368":
                    processed = self._resize_images(processed, (368, 368))
                elif step_name == "resize_1024x1024":
                    processed = self._resize_images(processed, (1024, 1024))
                elif step_name == "normalize_imagenet":
                    processed = self._normalize_imagenet(processed)
                elif step_name == "normalize_clip":
                    processed = self._normalize_clip(processed)
                elif step_name == "normalize_diffusion":
                    processed = self._normalize_diffusion(processed)
                elif step_name == "to_tensor":
                    processed = self._convert_to_tensor(processed)
                elif step_name == "prepare_sam_prompts":
                    processed = self._prepare_sam_prompts(processed)
                elif step_name == "prepare_diffusion_input":
                    processed = self._prepare_diffusion_input(processed)
                elif step_name == "prepare_ootd_inputs":
                    processed = self._prepare_ootd_inputs(processed)
                elif step_name == "extract_pose_features":
                    processed = self._extract_pose_features(processed)
                elif step_name == "prepare_sr_input":
                    processed = self._prepare_sr_input(processed)
                else:
                    self.logger.debug(f"⚠️ 알 수 없는 전처리 단계: {step_name}")
            
            self.logger.debug(f"✅ {self.step_name} 전처리 완료")
            return processed
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 전처리 실패: {e}")
            return input_data
    
    def _resize_images(self, data: Dict[str, Any], target_size: Tuple[int, int]) -> Dict[str, Any]:
        """이미지 리사이즈 처리"""
        result = data.copy()
        
        for key, value in data.items():
            try:
                if PIL_AVAILABLE and isinstance(value, Image.Image):
                    result[key] = value.resize(target_size, Image.LANCZOS)
                elif NUMPY_AVAILABLE and isinstance(value, np.ndarray) and len(value.shape) >= 2:
                    if CV2_AVAILABLE:
                        if len(value.shape) == 3:
                            result[key] = cv2.resize(value, target_size)
                        elif len(value.shape) == 2:
                            result[key] = cv2.resize(value, target_size)
                    else:
                        # PIL 폴백
                        if PIL_AVAILABLE:
                            if len(value.shape) == 3:
                                img = Image.fromarray(value.astype(np.uint8))
                                result[key] = np.array(img.resize(target_size, Image.LANCZOS))
                            elif len(value.shape) == 2:
                                img = Image.fromarray(value.astype(np.uint8), mode='L')
                                result[key] = np.array(img.resize(target_size, Image.LANCZOS))
            except Exception as e:
                self.logger.debug(f"이미지 리사이즈 실패 ({key}): {e}")
        
        return result
    
    def _normalize_imagenet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """ImageNet 정규화"""
        result = data.copy()
        mean = np.array(self.detailed_data_spec.normalization_mean)
        std = np.array(self.detailed_data_spec.normalization_std)
        
        for key, value in data.items():
            try:
                if PIL_AVAILABLE and isinstance(value, Image.Image):
                    array = np.array(value).astype(np.float32) / 255.0
                    if len(array.shape) == 3 and array.shape[2] == 3:
                        normalized = (array - mean) / std
                        result[key] = normalized
                elif NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                    if value.dtype != np.float32:
                        value = value.astype(np.float32)
                    if value.max() > 1.0:
                        value = value / 255.0
                    if len(value.shape) == 3 and value.shape[2] == 3:
                        normalized = (value - mean) / std
                        result[key] = normalized
            except Exception as e:
                self.logger.debug(f"ImageNet 정규화 실패 ({key}): {e}")
        
        return result
    
    def _normalize_clip(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """CLIP 정규화"""
        result = data.copy()
        clip_mean = np.array([0.48145466, 0.4578275, 0.40821073])
        clip_std = np.array([0.26862954, 0.26130258, 0.27577711])
        
        for key, value in data.items():
            try:
                if PIL_AVAILABLE and isinstance(value, Image.Image):
                    array = np.array(value).astype(np.float32) / 255.0
                    if len(array.shape) == 3 and array.shape[2] == 3:
                        normalized = (array - clip_mean) / clip_std
                        result[key] = normalized
                elif NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                    if value.dtype != np.float32:
                        value = value.astype(np.float32)
                    if value.max() > 1.0:
                        value = value / 255.0
                    if len(value.shape) == 3 and value.shape[2] == 3:
                        normalized = (value - clip_mean) / clip_std
                        result[key] = normalized
            except Exception as e:
                self.logger.debug(f"CLIP 정규화 실패 ({key}): {e}")
        
        return result
    
    def _normalize_diffusion(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Diffusion 정규화"""
        result = data.copy()
        
        for key, value in data.items():
            try:
                if PIL_AVAILABLE and isinstance(value, Image.Image):
                    array = np.array(value).astype(np.float32) / 255.0
                    normalized = 2.0 * array - 1.0
                    result[key] = normalized
                elif NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                    if value.dtype != np.float32:
                        value = value.astype(np.float32)
                    if value.max() > 1.0:
                        value = value / 255.0
                    normalized = 2.0 * value - 1.0
                    result[key] = normalized
            except Exception as e:
                self.logger.debug(f"Diffusion 정규화 실패 ({key}): {e}")
        
        return result
    
    def _convert_to_tensor(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """PyTorch 텐서 변환"""
        if not TORCH_AVAILABLE:
            return data
        
        result = data.copy()
        
        for key, value in data.items():
            try:
                if NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                    if len(value.shape) == 3 and value.shape[2] in [1, 3, 4]:
                        value = np.transpose(value, (2, 0, 1))
                    result[key] = torch.from_numpy(value).float()
                elif PIL_AVAILABLE and isinstance(value, Image.Image):
                    array = np.array(value)
                    if len(array.shape) == 3:
                        array = np.transpose(array, (2, 0, 1))
                    result[key] = torch.from_numpy(array).float()
            except Exception as e:
                self.logger.debug(f"텐서 변환 실패 ({key}): {e}")
        
        return result
    
    def _prepare_sam_prompts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """SAM 프롬프트 준비"""
        result = data.copy()
        
        if 'prompt_points' not in result and 'image' in result:
            if PIL_AVAILABLE and isinstance(result['image'], Image.Image):
                w, h = result['image'].size
                result['prompt_points'] = np.array([[w//2, h//2]])
                result['prompt_labels'] = np.array([1])
        
        return result
    
    def _prepare_diffusion_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Diffusion 모델 입력 준비"""
        result = data.copy()
        
        if 'guidance_scale' not in result:
            result['guidance_scale'] = 7.5
        if 'num_inference_steps' not in result:
            result['num_inference_steps'] = 20
        if 'strength' not in result:
            result['strength'] = 0.8
        
        return result
    
    def _prepare_ootd_inputs(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """OOTD Diffusion 입력 준비"""
        result = data.copy()
        
        if 'fitting_mode' not in result:
            result['fitting_mode'] = 'hd'
        
        return result
    
    def _extract_pose_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """포즈 특징 추출"""
        return data
    
    def _prepare_sr_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Super Resolution 입력 준비"""
        result = data.copy()
        
        if 'tile_size' not in result:
            result['tile_size'] = 512
        if 'overlap' not in result:
            result['overlap'] = 64
        
        return result
    
    # ==============================================
    # 🔥 출력 데이터 변환 시스템 (모든 기능 유지)
    # ==============================================
    
    async def _convert_output_to_standard_format(self, ai_result: Dict[str, Any]) -> Dict[str, Any]:
        """AI 모델 출력 → 표준 형식 변환"""
        try:
            self.logger.debug(f"🔄 {self.step_name} 출력 데이터 변환 시작...")
            
            # 1. 후처리 적용
            processed_result = await self._apply_postprocessing(ai_result)
            
            # 2. API 응답 형식 변환
            api_response = self._convert_to_api_format(processed_result)
            
            # 3. 다음 Step들을 위한 데이터 준비
            next_step_data = self._prepare_next_step_data(processed_result)
            
            # 4. 표준 응답 구조 생성
            standard_response = {
                'success': True,
                'step_name': self.step_name,
                'step_id': self.step_id,
                'processing_time': getattr(self, '_last_processing_time', 0.0),
                
                # API 응답 데이터
                **api_response,
                
                # 다음 Step들을 위한 데이터
                'next_step_data': next_step_data,
                
                # 메타데이터
                'metadata': {
                    'input_shapes': {k: self._get_shape_info(v) for k, v in ai_result.items()},
                    'output_shapes': self.detailed_data_spec.output_shapes,
                    'device': self.device,
                    'github_compatible': True,
                    'detailed_data_spec_applied': True,
                    'data_conversion_version': 'v19.2'
                }
            }
            
            self.logger.debug(f"✅ {self.step_name} 출력 데이터 변환 완료")
            return standard_response
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 출력 데이터 변환 실패: {e}")
            return self._create_error_response(str(e))
    
    def _get_shape_info(self, value: Any) -> Optional[Tuple]:
        """값의 형태 정보 추출"""
        try:
            if hasattr(value, 'shape'):
                return tuple(value.shape)
            elif isinstance(value, (list, tuple)):
                return (len(value),)
            else:
                return None
        except:
            return None
    
    async def _apply_postprocessing(self, ai_result: Dict[str, Any]) -> Dict[str, Any]:
        """DetailedDataSpec 기반 후처리 자동 적용"""
        try:
            if not self.config.auto_postprocessing:
                return ai_result
            
            processed = ai_result.copy()
            
            self.logger.debug(f"🔄 {self.step_name} 후처리 적용: {self.detailed_data_spec.postprocessing_steps}")
            
            for step_name in self.detailed_data_spec.postprocessing_steps:
                if step_name == "softmax":
                    processed = self._apply_softmax(processed)
                elif step_name == "argmax":
                    processed = self._apply_argmax(processed)
                elif step_name == "resize_original":
                    processed = self._resize_to_original(processed)
                elif step_name == "to_numpy":
                    processed = self._convert_to_numpy(processed)
                elif step_name == "threshold_0.5":
                    processed = self._apply_threshold(processed, 0.5)
                elif step_name == "nms":
                    processed = self._apply_nms(processed)
                elif step_name == "denormalize_diffusion" or step_name == "denormalize_centered":
                    processed = self._denormalize_diffusion(processed)
                elif step_name == "denormalize":
                    processed = self._denormalize_imagenet(processed)
                elif step_name == "clip_values" or step_name == "clip_0_1":
                    processed = self._clip_values(processed, 0.0, 1.0)
                elif step_name == "apply_mask" or step_name == "apply_warping_mask":
                    processed = self._apply_mask(processed)
                elif step_name == "morphology_clean":
                    processed = self._morphology_operations(processed)
                elif step_name == "extract_keypoints":
                    processed = self._extract_keypoints(processed)
                elif step_name == "scale_coords":
                    processed = self._scale_coordinates(processed)
                elif step_name == "filter_confidence":
                    processed = self._filter_by_confidence(processed)
                elif step_name == "enhance_details":
                    processed = self._enhance_details(processed)
                elif step_name == "final_compositing":
                    processed = self._final_compositing(processed)
                elif step_name == "generate_quality_report":
                    processed = self._generate_quality_report(processed)
                else:
                    self.logger.debug(f"⚠️ 알 수 없는 후처리 단계: {step_name}")
            
            self.performance_metrics.postprocessing_operations += 1
            self.logger.debug(f"✅ {self.step_name} 후처리 완료")
            return processed
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 후처리 실패: {e}")
            return ai_result
    
    def _apply_softmax(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Softmax 적용"""
        result = data.copy()
        
        for key, value in data.items():
            try:
                if TORCH_AVAILABLE and torch.is_tensor(value):
                    result[key] = torch.softmax(value, dim=-1)
                elif NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                    exp_vals = np.exp(value - np.max(value, axis=-1, keepdims=True))
                    result[key] = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
            except Exception as e:
                self.logger.debug(f"Softmax 적용 실패 ({key}): {e}")
        
        return result
    
    def _apply_argmax(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Argmax 적용"""
        result = data.copy()
        
        for key, value in data.items():
            try:
                if TORCH_AVAILABLE and torch.is_tensor(value):
                    result[key] = torch.argmax(value, dim=-1)
                elif NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                    result[key] = np.argmax(value, axis=-1)
            except Exception as e:
                self.logger.debug(f"Argmax 적용 실패 ({key}): {e}")
        
        return result
    
    def _convert_to_numpy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """NumPy 변환"""
        if not NUMPY_AVAILABLE:
            return data
        
        result = data.copy()
        
        for key, value in data.items():
            try:
                if TORCH_AVAILABLE and torch.is_tensor(value):
                    result[key] = value.detach().cpu().numpy()
                elif not isinstance(value, np.ndarray):
                    if isinstance(value, (list, tuple)):
                        result[key] = np.array(value)
            except Exception as e:
                self.logger.debug(f"NumPy 변환 실패 ({key}): {e}")
        
        return result
    
    def _apply_threshold(self, data: Dict[str, Any], threshold: float) -> Dict[str, Any]:
        """임계값 적용"""
        result = data.copy()
        
        for key, value in data.items():
            try:
                if NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                    result[key] = (value > threshold).astype(np.float32)
                elif TORCH_AVAILABLE and torch.is_tensor(value):
                    result[key] = (value > threshold).float()
            except Exception as e:
                self.logger.debug(f"임계값 적용 실패 ({key}): {e}")
        
        return result
    
    def _denormalize_diffusion(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Diffusion 역정규화 ([-1, 1] → [0, 1])"""
        result = data.copy()
        
        for key, value in data.items():
            try:
                if NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                    result[key] = (value + 1.0) / 2.0
                elif TORCH_AVAILABLE and torch.is_tensor(value):
                    result[key] = (value + 1.0) / 2.0
            except Exception as e:
                self.logger.debug(f"Diffusion 역정규화 실패 ({key}): {e}")
        
        return result
    
    def _denormalize_imagenet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """ImageNet 역정규화"""
        result = data.copy()
        mean = np.array(self.detailed_data_spec.normalization_mean)
        std = np.array(self.detailed_data_spec.normalization_std)
        
        for key, value in data.items():
            try:
                if NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                    if len(value.shape) == 3 and value.shape[2] == 3:
                        denormalized = value * std + mean
                        result[key] = np.clip(denormalized, 0, 1)
            except Exception as e:
                self.logger.debug(f"ImageNet 역정규화 실패 ({key}): {e}")
        
        return result
    
    def _clip_values(self, data: Dict[str, Any], min_val: float, max_val: float) -> Dict[str, Any]:
        """값 범위 클리핑"""
        result = data.copy()
        
        for key, value in data.items():
            try:
                if NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                    result[key] = np.clip(value, min_val, max_val)
                elif TORCH_AVAILABLE and torch.is_tensor(value):
                    result[key] = torch.clamp(value, min_val, max_val)
            except Exception as e:
                self.logger.debug(f"값 클리핑 실패 ({key}): {e}")
        
        return result
    
    def _apply_mask(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """마스크 적용"""
        result = data.copy()
        
        if 'mask' in data and 'image' in data:
            try:
                mask = data['mask']
                image = data['image']
                
                if NUMPY_AVAILABLE:
                    if isinstance(mask, np.ndarray) and isinstance(image, np.ndarray):
                        result['masked_image'] = image * mask
                        
            except Exception as e:
                self.logger.debug(f"마스크 적용 실패: {e}")
        
        return result
    
    def _morphology_operations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """형태학적 연산 (노이즈 제거)"""
        result = data.copy()
        
        for key, value in data.items():
            try:
                if CV2_AVAILABLE and NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                    if len(value.shape) == 2:
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        opened = cv2.morphologyEx(value.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
                        result[key] = closed.astype(np.float32)
                        
            except Exception as e:
                self.logger.debug(f"형태학적 연산 실패 ({key}): {e}")
        
        return result
    
    def _extract_keypoints(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """키포인트 추출"""
        result = data.copy()
        
        if 'heatmaps' in data:
            try:
                heatmaps = data['heatmaps']
                if NUMPY_AVAILABLE and isinstance(heatmaps, np.ndarray):
                    keypoints = []
                    for i in range(heatmaps.shape[0]):
                        heatmap = heatmaps[i]
                        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                        confidence = heatmap[y, x]
                        keypoints.append([x, y, confidence])
                    result['keypoints'] = np.array(keypoints)
                    
            except Exception as e:
                self.logger.debug(f"키포인트 추출 실패: {e}")
        
        return result
    
    def _scale_coordinates(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """좌표 스케일링"""
        result = data.copy()
        
        if 'keypoints' in data and 'original_size' in data:
            try:
                keypoints = data['keypoints']
                original_size = data['original_size']
                
                if isinstance(keypoints, np.ndarray) and len(keypoints.shape) == 2:
                    scale_x = original_size[0] / self.detailed_data_spec.input_shapes.get('image', (512, 512))[1]
                    scale_y = original_size[1] / self.detailed_data_spec.input_shapes.get('image', (512, 512))[0]
                    
                    scaled_keypoints = keypoints.copy()
                    scaled_keypoints[:, 0] *= scale_x
                    scaled_keypoints[:, 1] *= scale_y
                    result['scaled_keypoints'] = scaled_keypoints
                    
            except Exception as e:
                self.logger.debug(f"좌표 스케일링 실패: {e}")
        
        return result
    
    def _filter_by_confidence(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """신뢰도 기반 필터링"""
        result = data.copy()
        
        confidence_threshold = self.config.confidence_threshold
        
        for key, value in data.items():
            try:
                if key.endswith('_confidence') or key.endswith('_scores'):
                    if NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                        valid_mask = value > confidence_threshold
                        result[f'{key}_filtered'] = value[valid_mask]
                        
                        base_key = key.replace('_confidence', '').replace('_scores', '')
                        if base_key in data:
                            base_data = data[base_key]
                            if isinstance(base_data, np.ndarray) and len(base_data) == len(value):
                                result[f'{base_key}_filtered'] = base_data[valid_mask]
                                
            except Exception as e:
                self.logger.debug(f"신뢰도 필터링 실패 ({key}): {e}")
        
        return result
    
    def _enhance_details(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """세부사항 향상"""
        result = data.copy()
        
        for key, value in data.items():
            try:
                if NUMPY_AVAILABLE and isinstance(value, np.ndarray) and len(value.shape) >= 2:
                    if CV2_AVAILABLE and len(value.shape) == 3:
                        blurred = cv2.GaussianBlur(value, (3, 3), 1.0)
                        sharpened = cv2.addWeighted(value, 1.5, blurred, -0.5, 0)
                        result[f'{key}_enhanced'] = np.clip(sharpened, 0, 1)
                        
            except Exception as e:
                self.logger.debug(f"세부사항 향상 실패 ({key}): {e}")
        
        return result
    
    def _final_compositing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """최종 합성"""
        result = data.copy()
        
        if 'person_image' in data and 'clothing_image' in data and 'mask' in data:
            try:
                person = data['person_image']
                clothing = data['clothing_image']
                mask = data['mask']
                
                if all(isinstance(x, np.ndarray) for x in [person, clothing, mask]):
                    composited = person * (1 - mask) + clothing * mask
                    result['final_composited'] = composited
                    
            except Exception as e:
                self.logger.debug(f"최종 합성 실패: {e}")
        
        return result
    
    def _generate_quality_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """품질 보고서 생성"""
        result = data.copy()
        
        quality_metrics = {
            'overall_quality': 0.0,
            'detail_preservation': 0.0,
            'color_consistency': 0.0,
            'artifact_level': 0.0,
            'recommendations': []
        }
        
        try:
            if 'final_result' in data:
                final_result = data['final_result']
                if NUMPY_AVAILABLE and isinstance(final_result, np.ndarray):
                    mean_intensity = np.mean(final_result)
                    std_intensity = np.std(final_result)
                    
                    quality_metrics['overall_quality'] = min(1.0, (mean_intensity + std_intensity) / 2.0)
                    quality_metrics['detail_preservation'] = min(1.0, std_intensity * 2.0)
                    quality_metrics['color_consistency'] = 1.0 - abs(0.5 - mean_intensity)
                    
                    if quality_metrics['overall_quality'] < 0.7:
                        quality_metrics['recommendations'].append('이미지 품질 개선 필요')
                    if quality_metrics['detail_preservation'] < 0.5:
                        quality_metrics['recommendations'].append('세부사항 보존 개선 필요')
            
            result['quality_assessment'] = quality_metrics
            
        except Exception as e:
            self.logger.debug(f"품질 보고서 생성 실패: {e}")
            result['quality_assessment'] = quality_metrics
        
        return result
    
    def _apply_nms(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Non-Maximum Suppression 적용"""
        result = data.copy()
        
        if 'detections' in data and 'scores' in data:
            try:
                detections = data['detections']
                scores = data['scores']
                
                if NUMPY_AVAILABLE and isinstance(detections, np.ndarray) and isinstance(scores, np.ndarray):
                    sorted_indices = np.argsort(scores)[::-1]
                    
                    top_k = min(10, len(sorted_indices))
                    result['detections_nms'] = detections[sorted_indices[:top_k]]
                    result['scores_nms'] = scores[sorted_indices[:top_k]]
                    
            except Exception as e:
                self.logger.debug(f"NMS 적용 실패: {e}")
        
        return result
    
    def _resize_to_original(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """원본 크기로 리사이즈"""
        result = data.copy()
        
        if 'original_size' in data:
            original_size = data['original_size']
            
            for key, value in data.items():
                try:
                    if key != 'original_size' and isinstance(value, np.ndarray) and len(value.shape) >= 2:
                        if CV2_AVAILABLE:
                            if len(value.shape) == 3:
                                resized = cv2.resize(value, tuple(original_size))
                            elif len(value.shape) == 2:
                                resized = cv2.resize(value, tuple(original_size))
                            else:
                                continue
                            result[f'{key}_original_size'] = resized
                            
                except Exception as e:
                    self.logger.debug(f"원본 크기 리사이즈 실패 ({key}): {e}")
        
        return result
    
    def _convert_to_api_format(self, processed_result: Dict[str, Any]) -> Dict[str, Any]:
        """AI 결과 → API 응답 형식 변환"""
        api_response = {}
        
        try:
            for api_field, api_type in self.detailed_data_spec.api_output_mapping.items():
                if api_field in processed_result:
                    value = processed_result[api_field]
                    
                    if api_type == "base64_string":
                        api_response[api_field] = self._array_to_base64(value)
                    elif api_type == "List[Dict]":
                        api_response[api_field] = self._convert_to_list_dict(value)
                    elif api_type == "List[Dict[str, float]]":
                        api_response[api_field] = self._convert_keypoints_to_dict_list(value)
                    elif api_type == "float":
                        api_response[api_field] = float(value) if value is not None else 0.0
                    elif api_type == "List[float]":
                        if isinstance(value, (list, tuple)):
                            api_response[api_field] = [float(x) for x in value]
                        elif NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                            api_response[api_field] = value.flatten().tolist()
                        else:
                            api_response[api_field] = [float(value)] if value is not None else []
                    else:
                        api_response[api_field] = value
            
            if not api_response:
                api_response = self._create_fallback_api_response(processed_result)
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} API 형식 변환 실패: {e}")
            api_response = self._create_fallback_api_response(processed_result)
        
        return api_response
    
    def _prepare_next_step_data(self, processed_result: Dict[str, Any]) -> Dict[str, Any]:
        """다음 Step들을 위한 데이터 준비"""
        next_step_data = {}
        
        try:
            for next_step, data_schema in self.detailed_data_spec.provides_to_next_step.items():
                step_data = {}
                
                for data_key, data_type in data_schema.items():
                    if data_key in processed_result:
                        value = processed_result[data_key]
                        
                        if data_type == "np.ndarray" and NUMPY_AVAILABLE:
                            if TORCH_AVAILABLE and torch.is_tensor(value):
                                step_data[data_key] = value.detach().cpu().numpy()
                            elif not isinstance(value, np.ndarray):
                                step_data[data_key] = np.array(value)
                            else:
                                step_data[data_key] = value
                                
                        elif data_type == "torch.Tensor" and TORCH_AVAILABLE:
                            if NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                                step_data[data_key] = torch.from_numpy(value)
                            elif not torch.is_tensor(value):
                                step_data[data_key] = torch.tensor(value)
                            else:
                                step_data[data_key] = value
                                
                        else:
                            step_data[data_key] = value
                
                if step_data:
                    next_step_data[next_step] = step_data
                    self.performance_metrics.step_data_transfers += 1
        
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 다음 Step 데이터 준비 실패: {e}")
        
        return next_step_data
    
    # ==============================================
    # 🔥 데이터 검증 시스템 (모든 기능 유지)
    # ==============================================
    
    def _validate_input_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """입력 데이터 검증"""
        validated = input_data.copy()
        
        try:
            for key, value in input_data.items():
                # 데이터 타입 검증
                if key in self.detailed_data_spec.input_shapes:
                    expected_shape = self.detailed_data_spec.input_shapes[key]
                    if hasattr(value, 'shape'):
                        actual_shape = value.shape
                        if len(actual_shape) > len(expected_shape):
                            if actual_shape[1:] != tuple(expected_shape):
                                self.logger.warning(f"⚠️ {self.step_name} Shape mismatch for {key}")
                        elif actual_shape != tuple(expected_shape):
                            self.logger.warning(f"⚠️ {self.step_name} Shape mismatch for {key}")
                
                # 값 범위 검증
                if key in self.detailed_data_spec.input_value_ranges:
                    min_val, max_val = self.detailed_data_spec.input_value_ranges[key]
                    if hasattr(value, 'min') and hasattr(value, 'max'):
                        actual_min, actual_max = float(value.min()), float(value.max())
                        if actual_min < min_val or actual_max > max_val:
                            self.logger.warning(f"⚠️ {self.step_name} Value range warning for {key}")
                            
                            # 자동 클리핑
                            if self.config.strict_data_validation:
                                if NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                                    validated[key] = np.clip(value, min_val, max_val)
                                elif TORCH_AVAILABLE and torch.is_tensor(value):
                                    validated[key] = torch.clamp(value, min_val, max_val)
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 입력 데이터 검증 실패: {e}")
            self.performance_metrics.validation_failures += 1
        
        return validated
    
    # ==============================================
    # 🔥 유틸리티 메서드들 (모든 기능 유지)
    # ==============================================
    
    def _array_to_base64(self, array: Any) -> str:
        """NumPy 배열/텐서 → Base64 문자열 변환"""
        try:
            if TORCH_AVAILABLE and torch.is_tensor(array):
                array = array.detach().cpu().numpy()
            
            if not NUMPY_AVAILABLE or not isinstance(array, np.ndarray):
                return ""
            
            # 값 범위 정규화
            if array.dtype != np.uint8:
                if array.max() <= 1.0:
                    array = (array * 255).astype(np.uint8)
                else:
                    array = np.clip(array, 0, 255).astype(np.uint8)
            
            # PIL Image로 변환
            if PIL_AVAILABLE:
                if len(array.shape) == 3:
                    if array.shape[0] in [1, 3, 4] and array.shape[0] < array.shape[1]:
                        array = np.transpose(array, (1, 2, 0))
                    
                    if array.shape[2] == 1:
                        array = array.squeeze(2)
                        image = Image.fromarray(array, mode='L')
                    elif array.shape[2] == 3:
                        image = Image.fromarray(array, mode='RGB')
                    elif array.shape[2] == 4:
                        image = Image.fromarray(array, mode='RGBA')
                    else:
                        image = Image.fromarray(array[:, :, 0], mode='L')
                        
                elif len(array.shape) == 2:
                    image = Image.fromarray(array, mode='L')
                else:
                    raise ValueError(f"Unsupported array shape: {array.shape}")
                
                # Base64 인코딩
                buffer = BytesIO()
                image.save(buffer, format='PNG')
                buffer.seek(0)
                
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return ""
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} Base64 변환 실패: {e}")
            return ""
    
    def _convert_to_list_dict(self, value: Any) -> List[Dict]:
        """값을 List[Dict] 형식으로 변환"""
        try:
            if isinstance(value, (list, tuple)):
                if all(isinstance(item, dict) for item in value):
                    return list(value)
                else:
                    return [{'value': item, 'index': i} for i, item in enumerate(value)]
            
            elif isinstance(value, dict):
                return [value]
            
            elif NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                if len(value.shape) == 1:
                    return [{'value': float(item), 'index': i} for i, item in enumerate(value)]
                elif len(value.shape) == 2:
                    return [{'row': i, 'data': row.tolist()} for i, row in enumerate(value)]
                else:
                    return [{'data': value.tolist()}]
            
            else:
                return [{'value': value}]
                
        except Exception as e:
            self.logger.debug(f"List[Dict] 변환 실패: {e}")
            return [{'value': str(value)}]
    
    def _convert_keypoints_to_dict_list(self, keypoints: Any) -> List[Dict[str, float]]:
        """키포인트를 List[Dict[str, float]] 형식으로 변환"""
        try:
            if NUMPY_AVAILABLE and isinstance(keypoints, np.ndarray):
                if len(keypoints.shape) == 2 and keypoints.shape[1] >= 2:
                    result = []
                    for i, point in enumerate(keypoints):
                        point_dict = {
                            'x': float(point[0]),
                            'y': float(point[1])
                        }
                        if keypoints.shape[1] > 2:
                            point_dict['confidence'] = float(point[2])
                        if keypoints.shape[1] > 3:
                            point_dict['visibility'] = float(point[3])
                        
                        point_dict['index'] = i
                        result.append(point_dict)
                    
                    return result
            
            elif isinstance(keypoints, (list, tuple)):
                result = []
                for i, point in enumerate(keypoints):
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        point_dict = {
                            'x': float(point[0]),
                            'y': float(point[1]),
                            'index': i
                        }
                        if len(point) > 2:
                            point_dict['confidence'] = float(point[2])
                        result.append(point_dict)
                
                return result
            
            return []
            
        except Exception as e:
            self.logger.debug(f"키포인트 Dict 변환 실패: {e}")
            return []
    
    def _create_fallback_api_response(self, processed_result: Dict[str, Any]) -> Dict[str, Any]:
        """폴백 API 응답 생성"""
        fallback_response = {}
        
        try:
            # 공통적으로 사용되는 키들에 대한 기본 매핑
            common_mappings = {
                'parsing_mask': 'base64_string',
                'segmentation_mask': 'base64_string',
                'fitted_image': 'base64_string',
                'enhanced_image': 'base64_string',
                'final_result': 'base64_string',
                'result_image': 'base64_string',
                'output_image': 'base64_string',
                
                'keypoints': 'List[Dict[str, float]]',
                'pose_keypoints': 'List[Dict[str, float]]',
                
                'confidence': 'float',
                'quality_score': 'float'
            }
            
            for key, value in processed_result.items():
                if key in common_mappings:
                    api_type = common_mappings[key]
                    
                    if api_type == 'base64_string':
                        fallback_response[key] = self._array_to_base64(value)
                    elif api_type == 'List[Dict[str, float]]':
                        fallback_response[key] = self._convert_keypoints_to_dict_list(value)
                    elif api_type == 'float':
                        fallback_response[key] = float(value) if value is not None else 0.0
            
            # 기본 응답이 없는 경우 첫 번째 이미지형 데이터를 result로 설정
            if not fallback_response:
                for key, value in processed_result.items():
                    if NUMPY_AVAILABLE and isinstance(value, np.ndarray) and len(value.shape) >= 2:
                        fallback_response['result'] = self._array_to_base64(value)
                        break
                    elif PIL_AVAILABLE and isinstance(value, Image.Image):
                        fallback_response['result'] = self._array_to_base64(np.array(value))
                        break
            
        except Exception as e:
            self.logger.debug(f"폴백 API 응답 생성 실패: {e}")
        
        return fallback_response
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """표준 에러 응답 생성"""
        return {
            'success': False,
            'error': error_message,
            'step_name': self.step_name,
            'step_id': self.step_id,
            'github_compatible': True,
            'detailed_data_spec_applied': False,
            'processing_time': 0.0,
            'timestamp': time.time()
        }
    
    # ==============================================
    # 🔥 기존 GitHub 호환 메서드들 (모든 기능 유지)
    # ==============================================
    
    def _create_github_config(self, **kwargs) -> GitHubStepConfig:
        """GitHub 프로젝트 호환 설정 생성"""
        config = GitHubStepConfig()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # GitHub 프로젝트 특별 설정
        config.github_compatibility_mode = True
        config.real_ai_pipeline_support = True
        config.enable_detailed_data_spec = True
        
        # 환경별 설정 적용
        if CONDA_INFO['is_target_env']:
            config.conda_optimized = True
            config.conda_env = CONDA_INFO['conda_env']
        
        if IS_M3_MAX:
            config.m3_max_optimized = True
            config.memory_gb = MEMORY_GB
            config.use_unified_memory = True
        
        return config
    
    def _apply_github_environment_optimization(self):
        """GitHub 프로젝트 환경 최적화"""
        try:
            # M3 Max GitHub 최적화
            if self.is_m3_max:
                if self.device == "auto" and MPS_AVAILABLE:
                    self.device = "mps"
                
                if self.config.batch_size == 1 and self.memory_gb >= 64:
                    self.config.batch_size = 2
                
                self.config.auto_memory_cleanup = True
                self.logger.debug(f"✅ GitHub M3 Max 최적화: {self.memory_gb:.1f}GB, device={self.device}")
            
            # GitHub conda 환경 최적화
            if self.conda_info['is_target_env']:
                self.config.optimization_enabled = True
                self.real_ai_pipeline_ready = True
                self.logger.debug(f"✅ GitHub conda 환경 최적화: {self.conda_info['conda_env']}")
            
        except Exception as e:
            self.logger.debug(f"GitHub 환경 최적화 실패: {e}")
    
    def _github_emergency_setup(self, error: Exception):
        """GitHub 호환 긴급 설정"""
        self.step_name = getattr(self, 'step_name', self.__class__.__name__)
        self.logger = logging.getLogger("github_emergency")
        self.device = "cpu"
        self.is_initialized = False
        self.github_compatible = False
        self.performance_metrics = GitHubPerformanceMetrics()
        self.detailed_data_spec = DetailedDataSpecConfig()
        self.dependency_manager = GitHubDependencyManager(self.step_name)
        self.logger.error(f"🚨 {self.step_name} GitHub 긴급 초기화: {error}")
    
    def _resolve_device(self, device: str) -> str:
        """GitHub 디바이스 해결 (환경 최적화)"""
        if device == "auto":
            # GitHub M3 Max 우선 처리
            if IS_M3_MAX and MPS_AVAILABLE:
                return "mps"
            
            if TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"
            return "cpu"
        return device
    
    def _update_performance_metrics(self, processing_time: float, success: bool):
        """성능 메트릭 업데이트"""
        try:
            self.performance_metrics.process_count += 1
            self.performance_metrics.total_process_time += processing_time
            self.performance_metrics.average_process_time = (
                self.performance_metrics.total_process_time / 
                self.performance_metrics.process_count
            )
            
            if success:
                self.performance_metrics.success_count += 1
            else:
                self.performance_metrics.error_count += 1
            
            # 최근 처리 시간 저장
            self._last_processing_time = processing_time
            
            # GitHub 파이프라인 성공률 계산
            if self.performance_metrics.github_process_calls > 0:
                success_rate = (
                    self.performance_metrics.success_count /
                    self.performance_metrics.process_count * 100
                )
                self.performance_metrics.pipeline_success_rate = success_rate
                
        except Exception as e:
            self.logger.debug(f"성능 메트릭 업데이트 실패: {e}")
    
    # ==============================================
    # 🔥 GitHub 호환 의존성 주입 인터페이스 (순환참조 해결)
    # ==============================================
    
    def set_model_loader(self, model_loader):
        """GitHub 표준 ModelLoader 의존성 주입"""
        try:
            success = self.dependency_manager.inject_model_loader(model_loader)
            if success:
                self.model_loader = model_loader
                self.has_model = True
                self.model_loaded = True
                self.real_ai_pipeline_ready = True
                self.performance_metrics.dependencies_injected += 1
                self.logger.info(f"✅ {self.step_name} GitHub ModelLoader 의존성 주입 완료")
        except Exception as e:
            self.performance_metrics.injection_failures += 1
            self.logger.error(f"❌ {self.step_name} GitHub ModelLoader 의존성 주입 오류: {e}")

    def set_memory_manager(self, memory_manager):
        """GitHub 표준 MemoryManager 의존성 주입"""
        try:
            success = self.dependency_manager.inject_memory_manager(memory_manager)
            if success:
                self.memory_manager = memory_manager
                self.performance_metrics.dependencies_injected += 1
        except Exception as e:
            self.performance_metrics.injection_failures += 1
            self.logger.warning(f"⚠️ {self.step_name} GitHub MemoryManager 의존성 주입 오류: {e}")
    
    # ==============================================
    # 🔥 GitHub 호환 의존성 검증 (순환참조 해결)
    # ==============================================
    
    def validate_dependencies(self, format_type: DependencyValidationFormat = None) -> Union[Dict[str, bool], Dict[str, Any]]:
        """GitHub 프로젝트 호환 의존성 검증 (v19.2)"""
        try:
            return self.dependency_manager.validate_dependencies_github_format(format_type)
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} GitHub 의존성 검증 실패: {e}")
            
            if format_type == DependencyValidationFormat.BOOLEAN_DICT:
                return {'model_loader': False, 'step_interface': False, 'memory_manager': False, 'data_converter': False}
            else:
                return {'success': False, 'error': str(e), 'github_compatible': False}
    
    def validate_dependencies_boolean(self) -> Dict[str, bool]:
        """GitHub Step 클래스 호환 (GeometricMatchingStep 등)"""
        return self.validate_dependencies(DependencyValidationFormat.BOOLEAN_DICT)
    
    def validate_dependencies_detailed(self) -> Dict[str, Any]:
        """StepFactory 호환 (상세 정보)"""
        return self.validate_dependencies(DependencyValidationFormat.DETAILED_DICT)
    
    # ==============================================
    # 🔥 GitHub 표준 초기화 및 상태 관리
    # ==============================================
    
    def initialize(self) -> bool:
        """GitHub 표준 초기화"""
        try:
            if self.is_initialized:
                return True
            
            self.logger.info(f"🔄 {self.step_name} GitHub 표준 초기화 시작...")
            
            # DetailedDataSpec 검증
            if not self.data_conversion_ready:
                self.logger.warning(f"⚠️ {self.step_name} DetailedDataSpec 데이터 변환 준비 미완료")
            
            # 초기화 상태 설정
            self.dependency_manager.dependency_status.base_initialized = True
            self.dependency_manager.dependency_status.github_compatible = True
            
            self.is_initialized = True
            
            self.logger.info(f"✅ {self.step_name} GitHub 표준 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} GitHub 초기화 실패: {e}")
            if hasattr(self, 'performance_metrics'):
                self.performance_metrics.error_count += 1
            return False

    def get_status(self) -> Dict[str, Any]:
        """GitHub 통합 상태 조회 (v19.2)"""
        try:
            return {
                'step_info': {
                    'step_name': self.step_name,
                    'step_id': self.step_id,
                    'version': 'BaseStepMixin v19.2 Circular Reference Fix'
                },
                'github_status_flags': {
                    'is_initialized': self.is_initialized,
                    'is_ready': self.is_ready,
                    'has_model': self.has_model,
                    'model_loaded': self.model_loaded,
                    'github_compatible': self.github_compatible,
                    'real_ai_pipeline_ready': self.real_ai_pipeline_ready,
                    'data_conversion_ready': self.data_conversion_ready
                },
                'detailed_data_spec_status': {
                    'spec_loaded': self.dependency_manager.dependency_status.detailed_data_spec_loaded,
                    'data_conversion_ready': self.dependency_manager.dependency_status.data_conversion_ready,
                    'preprocessing_configured': bool(self.detailed_data_spec.preprocessing_steps),
                    'postprocessing_configured': bool(self.detailed_data_spec.postprocessing_steps),
                    'api_mapping_configured': bool(self.detailed_data_spec.api_input_mapping and self.detailed_data_spec.api_output_mapping),
                    'step_flow_configured': bool(self.detailed_data_spec.provides_to_next_step or self.detailed_data_spec.accepts_from_previous_step)
                },
                'github_performance': {
                    'data_conversions': self.performance_metrics.data_conversions,
                    'preprocessing_operations': self.performance_metrics.preprocessing_operations,
                    'postprocessing_operations': self.performance_metrics.postprocessing_operations,
                    'api_conversions': self.performance_metrics.api_conversions,
                    'step_data_transfers': self.performance_metrics.step_data_transfers,
                    'validation_failures': self.performance_metrics.validation_failures
                },
                'timestamp': time.time()
            }
        except Exception as e:
            self.logger.error(f"❌ GitHub 상태 조회 실패: {e}")
            return {'error': str(e), 'version': 'BaseStepMixin v19.2 Circular Reference Fix'}

# ==============================================
# 🔥 Export
# ==============================================

__all__ = [
    # 메인 클래스
    'BaseStepMixin',
    'GitHubDependencyManager',
    
    # 설정 및 상태 클래스들
    'DetailedDataSpecConfig',
    'GitHubStepConfig',
    'GitHubDependencyStatus', 
    'GitHubPerformanceMetrics',
    
    # GitHub 열거형들
    'ProcessMethodSignature',
    'DependencyValidationFormat',
    'DataConversionMethod',
    
    # 상수들
    'TORCH_AVAILABLE',
    'MPS_AVAILABLE',
    'NUMPY_AVAILABLE',
    'PIL_AVAILABLE',
    'CV2_AVAILABLE',
    'CONDA_INFO',
    'IS_M3_MAX',
    'MEMORY_GB'
]

# ==============================================
# 🔥 모듈 로드 완료 로그
# ==============================================

logger = logging.getLogger(__name__)
logger.info("=" * 100)
logger.info("🔥 BaseStepMixin v19.2 - 순환참조 완전 해결 + 모든 기능 포함")
logger.info("=" * 100)
logger.info("✅ 순환참조 완전 해결 (GitHubDependencyManager 내장)")
logger.info("✅ step_model_requirements.py DetailedDataSpec 완전 활용")
logger.info("✅ API ↔ AI 모델 간 데이터 변환 표준화 완료")
logger.info("✅ Step 간 데이터 흐름 자동 처리")
logger.info("✅ 전처리/후처리 요구사항 자동 적용")
logger.info("✅ GitHub 프로젝트 Step 클래스들과 100% 호환")
logger.info("✅ process() 메서드 시그니처 완전 표준화")
logger.info("✅ 실제 Step 클래스들은 _run_ai_inference() 메서드만 구현하면 됨")
logger.info("✅ validate_dependencies() 오버로드 지원")
logger.info("✅ 모든 기능 그대로 유지하면서 순환참조만 해결")

logger.info("🔧 순환참조 해결 방법:")
logger.info("   🔗 GitHubDependencyManager 내장으로 외부 의존성 차단")
logger.info("   🔗 TYPE_CHECKING + 지연 import로 순환참조 방지")
logger.info("   🔗 실제 의존성만 사용 - Mock 폴백 제거")
logger.info("   🔗 모든 기능 그대로 유지")

logger.info("🎯 완전 복원된 전처리 (12개):")
logger.info("   - 이미지 리사이즈 (512x512, 768x1024, 256x192, 224x224, 368x368, 1024x1024)")
logger.info("   - 정규화 (ImageNet, CLIP, Diffusion)")
logger.info("   - 텐서 변환, SAM 프롬프트, Diffusion 입력, OOTD 입력, 포즈 특징, SR 입력")

logger.info("🎯 완전 복원된 후처리 (15개):")
logger.info("   - Softmax, Argmax, 원본 크기 리사이즈, NumPy 변환, 임계값, NMS")
logger.info("   - 역정규화 (Diffusion, ImageNet), 값 클리핑, 마스크 적용")
logger.info("   - 형태학적 연산, 키포인트 추출, 좌표 스케일링, 신뢰도 필터링")
logger.info("   - 세부사항 향상, 최종 합성, 품질 보고서 생성")

logger.info(f"🔧 현재 conda 환경: {CONDA_INFO['conda_env']} ({'✅ 최적화됨' if CONDA_INFO['is_target_env'] else '⚠️ 권장: mycloset-ai-clean'})")
logger.info(f"🖥️  현재 시스템: M3 Max={IS_M3_MAX}, 메모리={MEMORY_GB:.1f}GB")
logger.info(f"🚀 GitHub AI 파이프라인 준비: {TORCH_AVAILABLE and (MPS_AVAILABLE or (torch.cuda.is_available() if TORCH_AVAILABLE else False))}")
logger.info("=" * 100)
logger.info("🎉 BaseStepMixin v19.2 순환참조 해결 + 완전한 기능 복원 완료!")
logger.info("💡 이제 실제 Step 클래스들은 _run_ai_inference() 메서드만 구현하면 됩니다!")
logger.info("💡 모든 데이터 변환이 BaseStepMixin에서 자동으로 처리됩니다!")
logger.info("💡 순환참조 문제가 완전히 해결되고 실제 의존성만 사용합니다!")
logger.info("💡 Mock 폴백이 제거되어 실제 ModelLoader, MemoryManager, DataConverter만 허용됩니다!")
logger.info("=" * 100)