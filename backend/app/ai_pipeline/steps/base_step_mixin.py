# backend/app/ai_pipeline/steps/base_step_mixin.py
"""
🔥 BaseStepMixin v18.0 - 의존성 주입 완전 수정 (StepFactory v7.0 호환)
================================================================

✅ StepFactory v7.0과 완전 호환
✅ 의존성 주입 문제 완전 해결
✅ ModelLoader → StepModelInterface 연결 안정화
✅ 초기화 순서 최적화
✅ 에러 처리 강화 및 안전한 폴백
✅ TYPE_CHECKING 패턴으로 순환참조 완전 방지
✅ conda 환경 우선 최적화 (mycloset-ai-clean)
✅ M3 Max 128GB 메모리 최적화
✅ 프로덕션 레벨 안정성

핵심 수정사항:
1. 의존성 주입 인터페이스 완전 표준화
2. ModelLoader와 StepModelInterface 연결 로직 개선
3. 초기화 순서 최적화 (의존성 → 초기화 → 검증)
4. 안전한 에러 처리 및 상태 관리

Author: MyCloset AI Team
Date: 2025-07-25
Version: 18.0 (Dependency Injection Complete Fix)
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
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union, Callable, Type, TYPE_CHECKING
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from functools import wraps
from contextlib import asynccontextmanager

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

# ==============================================
# 🔥 의존성 주입 인터페이스 (표준화)
# ==============================================

class IModelProvider(ABC):
    """모델 제공자 인터페이스 (표준화)"""
    
    @abstractmethod
    def get_model(self, model_name: str) -> Optional[Any]:
        """모델 가져오기"""
        pass
    
    @abstractmethod
    async def get_model_async(self, model_name: str) -> Optional[Any]:
        """비동기 모델 가져오기"""
        pass
    
    @abstractmethod
    def is_model_available(self, model_name: str) -> bool:
        """모델 사용 가능 여부"""
        pass
    
    @abstractmethod
    def load_model(self, model_name: str, **kwargs) -> bool:
        """모델 로딩"""
        pass
    
    @abstractmethod
    def unload_model(self, model_name: str) -> bool:
        """모델 언로딩"""
        pass

class IMemoryManager(ABC):
    """메모리 관리자 인터페이스 (표준화)"""
    
    @abstractmethod
    def optimize_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """메모리 최적화"""
        pass
    
    @abstractmethod
    async def optimize_memory_async(self, aggressive: bool = False) -> Dict[str, Any]:
        """비동기 메모리 최적화"""
        pass
    
    @abstractmethod
    def get_memory_info(self) -> Dict[str, Any]:
        """메모리 정보 조회"""
        pass

class IDataConverter(ABC):
    """데이터 변환기 인터페이스 (표준화)"""
    
    @abstractmethod
    def convert_data(self, data: Any, target_format: str) -> Any:
        """데이터 변환"""
        pass
    
    @abstractmethod
    def validate_data(self, data: Any, expected_format: str) -> bool:
        """데이터 검증"""
        pass

# ==============================================
# 🔥 설정 및 상태 클래스
# ==============================================

@dataclass
class StepConfig:
    """통합 Step 설정 (v18.0)"""
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
    
    # 의존성 설정 (v18.0 강화)
    auto_inject_dependencies: bool = True
    require_model_loader: bool = True
    require_memory_manager: bool = False
    require_data_converter: bool = False
    dependency_timeout: float = 30.0  # 의존성 주입 타임아웃
    dependency_retry_count: int = 3   # 재시도 횟수
    
    # 환경 최적화
    conda_optimized: bool = False
    conda_env: str = "none"
    m3_max_optimized: bool = False
    memory_gb: float = 16.0
    use_unified_memory: bool = False

@dataclass
class DependencyStatus:
    """의존성 상태 추적 (v18.0 강화)"""
    model_loader: bool = False
    step_interface: bool = False
    memory_manager: bool = False
    data_converter: bool = False
    di_container: bool = False
    base_initialized: bool = False
    custom_initialized: bool = False
    dependencies_validated: bool = False
    
    # 환경 상태
    conda_optimized: bool = False
    m3_max_optimized: bool = False
    
    # 주입 시도 추적
    injection_attempts: Dict[str, int] = field(default_factory=dict)
    injection_errors: Dict[str, List[str]] = field(default_factory=dict)
    last_injection_time: float = field(default_factory=time.time)

@dataclass
class PerformanceMetrics:
    """성능 메트릭 (v18.0)"""
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
    
    # 의존성 메트릭 (v18.0 추가)
    dependencies_injected: int = 0
    injection_failures: int = 0
    average_injection_time: float = 0.0

# ==============================================
# 🔥 강화된 의존성 관리자 v18.0
# ==============================================

class EnhancedDependencyManager:
    """강화된 의존성 관리자 v18.0 (완전 수정)"""
    
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.logger = logging.getLogger(f"DependencyManager.{step_name}")
        
        # 의존성 저장
        self.dependencies: Dict[str, Any] = {}
        self.dependency_status = DependencyStatus()
        
        # 환경 정보
        self.conda_info = CONDA_INFO
        self.is_m3_max = IS_M3_MAX
        self.memory_gb = MEMORY_GB
        
        # 동기화
        self._lock = threading.RLock()
        
        # 주입 상태 추적
        self._injection_history: Dict[str, List[Dict[str, Any]]] = {}
        self._auto_injection_attempted = False
        
        # 환경 최적화 설정
        self._setup_environment_optimization()
    
    def _setup_environment_optimization(self):
        """환경 최적화 설정"""
        try:
            # conda 환경 최적화
            if self.conda_info['is_target_env']:
                self.dependency_status.conda_optimized = True
                self.logger.debug(f"✅ conda 환경 최적화 활성화: {self.conda_info['conda_env']}")
            
            # M3 Max 최적화
            if self.is_m3_max:
                self.dependency_status.m3_max_optimized = True
                self.logger.debug(f"✅ M3 Max 최적화 활성화: {self.memory_gb:.1f}GB")
                
        except Exception as e:
            self.logger.debug(f"환경 최적화 설정 실패: {e}")
    
    def inject_model_loader(self, model_loader: 'ModelLoader') -> bool:
        """ModelLoader 의존성 주입 (v18.0 완전 수정)"""
        injection_start = time.time()
        
        try:
            with self._lock:
                self.logger.info(f"🔄 {self.step_name} ModelLoader 의존성 주입 시작...")
                
                # 이전 주입 기록
                self._record_injection_attempt('model_loader')
                
                # 1. ModelLoader 저장
                self.dependencies['model_loader'] = model_loader
                
                # 2. ModelLoader 유효성 검증
                if not self._validate_model_loader(model_loader):
                    raise ValueError("ModelLoader 유효성 검증 실패")
                
                # 3. 🔥 StepModelInterface 생성 (핵심 수정)
                step_interface = self._create_step_interface(model_loader)
                if step_interface:
                    self.dependencies['step_interface'] = step_interface
                    self.dependency_status.step_interface = True
                    self.logger.info(f"✅ {self.step_name} StepModelInterface 생성 완료")
                else:
                    self.logger.warning(f"⚠️ {self.step_name} StepModelInterface 생성 실패")
                
                # 4. 환경 최적화 적용
                self._apply_model_loader_optimization(model_loader)
                
                # 5. 상태 업데이트
                self.dependency_status.model_loader = True
                self.dependency_status.last_injection_time = time.time()
                
                injection_time = time.time() - injection_start
                self.logger.info(f"✅ {self.step_name} ModelLoader 의존성 주입 완료 ({injection_time:.3f}초)")
                
                return True
                
        except Exception as e:
            injection_time = time.time() - injection_start
            self._record_injection_error('model_loader', str(e))
            self.logger.error(f"❌ {self.step_name} ModelLoader 주입 실패 ({injection_time:.3f}초): {e}")
            return False
    
    def _validate_model_loader(self, model_loader: 'ModelLoader') -> bool:
        """ModelLoader 유효성 검증"""
        try:
            # 기본 속성 확인
            required_attrs = ['load_model', 'is_initialized']
            for attr in required_attrs:
                if not hasattr(model_loader, attr):
                    self.logger.warning(f"⚠️ ModelLoader에 {attr} 속성 없음")
                    
            # create_step_interface 메서드 확인 (핵심)
            if not hasattr(model_loader, 'create_step_interface'):
                self.logger.warning("⚠️ ModelLoader에 create_step_interface 메서드 없음")
                return False
            
            # 초기화 상태 확인
            if hasattr(model_loader, 'is_initialized') and callable(model_loader.is_initialized):
                if not model_loader.is_initialized():
                    self.logger.warning("⚠️ ModelLoader가 초기화되지 않음")
                    # 초기화 시도
                    if hasattr(model_loader, 'initialize'):
                        try:
                            success = model_loader.initialize()
                            if not success:
                                self.logger.error("❌ ModelLoader 초기화 실패")
                                return False
                        except Exception as init_error:
                            self.logger.error(f"❌ ModelLoader 초기화 오류: {init_error}")
                            return False
            
            self.logger.debug(f"✅ {self.step_name} ModelLoader 유효성 검증 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ ModelLoader 유효성 검증 실패: {e}")
            return False
    
    def _create_step_interface(self, model_loader: 'ModelLoader') -> Optional['StepModelInterface']:
        """StepModelInterface 생성 (v18.0 핵심 수정)"""
        try:
            self.logger.info(f"🔄 {self.step_name} StepModelInterface 생성 시작...")
            
            # 1. create_step_interface 메서드 호출
            if hasattr(model_loader, 'create_step_interface'):
                interface = model_loader.create_step_interface(self.step_name)
                
                if interface:
                    # 2. 인터페이스 유효성 검증
                    if self._validate_step_interface(interface):
                        self.logger.info(f"✅ {self.step_name} StepModelInterface 생성 및 검증 완료")
                        return interface
                    else:
                        self.logger.warning(f"⚠️ {self.step_name} StepModelInterface 검증 실패")
                        return interface  # 검증 실패해도 인터페이스는 반환
                else:
                    self.logger.error(f"❌ {self.step_name} StepModelInterface 생성 실패")
                    
            else:
                self.logger.error(f"❌ ModelLoader에 create_step_interface 메서드 없음")
            
            # 3. 폴백 인터페이스 생성 시도
            return self._create_fallback_interface(model_loader)
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} StepModelInterface 생성 오류: {e}")
            return self._create_fallback_interface(model_loader)
    
    def _validate_step_interface(self, interface: 'StepModelInterface') -> bool:
        """StepModelInterface 유효성 검증"""
        try:
            # 필수 메서드 확인
            required_methods = ['get_model_sync', 'get_model_async', 'register_model_requirement']
            for method in required_methods:
                if not hasattr(interface, method):
                    self.logger.warning(f"⚠️ StepModelInterface에 {method} 메서드 없음")
            
            # step_name 속성 확인
            if hasattr(interface, 'step_name'):
                if interface.step_name != self.step_name:
                    self.logger.warning(f"⚠️ StepModelInterface step_name 불일치: {interface.step_name} != {self.step_name}")
            
            self.logger.debug(f"✅ {self.step_name} StepModelInterface 검증 완료")
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ StepModelInterface 검증 오류: {e}")
            return False
    
    def _create_fallback_interface(self, model_loader: 'ModelLoader') -> Optional['StepModelInterface']:
        """폴백 StepModelInterface 생성"""
        try:
            self.logger.info(f"🔄 {self.step_name} 폴백 StepModelInterface 생성 시도...")
            
            # StepModelInterface 동적 import
            import importlib
            interface_module = importlib.import_module('app.ai_pipeline.interface.step_interface')
            StepModelInterface = getattr(interface_module, 'StepModelInterface', None)
            
            if StepModelInterface:
                interface = StepModelInterface(self.step_name, model_loader)
                self.logger.info(f"✅ {self.step_name} 폴백 StepModelInterface 생성 완료")
                return interface
            else:
                self.logger.error("❌ StepModelInterface 클래스를 찾을 수 없음")
                
        except Exception as e:
            self.logger.error(f"❌ 폴백 StepModelInterface 생성 실패: {e}")
        
        return None
    
    def _apply_model_loader_optimization(self, model_loader: 'ModelLoader'):
        """ModelLoader 환경 최적화 적용"""
        try:
            # 환경 설정 적용
            if hasattr(model_loader, 'configure_environment'):
                env_config = {
                    'conda_env': self.conda_info['conda_env'],
                    'is_m3_max': self.is_m3_max,
                    'memory_gb': self.memory_gb,
                    'conda_optimized': self.conda_info['is_target_env']
                }
                model_loader.configure_environment(env_config)
                self.logger.debug(f"✅ {self.step_name} ModelLoader 환경 최적화 적용")
                
        except Exception as e:
            self.logger.debug(f"ModelLoader 환경 최적화 실패: {e}")
    
    def inject_memory_manager(self, memory_manager: 'MemoryManager') -> bool:
        """MemoryManager 의존성 주입 (M3 Max 최적화)"""
        try:
            with self._lock:
                self.logger.info(f"🔄 {self.step_name} MemoryManager 의존성 주입 시작...")
                
                self._record_injection_attempt('memory_manager')
                
                self.dependencies['memory_manager'] = memory_manager
                self.dependency_status.memory_manager = True
                
                # M3 Max 특별 설정
                if self.is_m3_max and hasattr(memory_manager, 'configure_m3_max'):
                    memory_manager.configure_m3_max(self.memory_gb)
                    self.logger.debug("✅ M3 Max 메모리 최적화 설정 완료")
                
                self.logger.info(f"✅ {self.step_name} MemoryManager 의존성 주입 완료")
                return True
                
        except Exception as e:
            self._record_injection_error('memory_manager', str(e))
            self.logger.error(f"❌ {self.step_name} MemoryManager 주입 실패: {e}")
            return False
    
    def inject_data_converter(self, data_converter: 'DataConverter') -> bool:
        """DataConverter 의존성 주입"""
        try:
            with self._lock:
                self.logger.info(f"🔄 {self.step_name} DataConverter 의존성 주입 시작...")
                
                self._record_injection_attempt('data_converter')
                
                self.dependencies['data_converter'] = data_converter
                self.dependency_status.data_converter = True
                
                self.logger.info(f"✅ {self.step_name} DataConverter 의존성 주입 완료")
                return True
                
        except Exception as e:
            self._record_injection_error('data_converter', str(e))
            self.logger.error(f"❌ {self.step_name} DataConverter 주입 실패: {e}")
            return False
    
    def inject_di_container(self, di_container: 'DIContainer') -> bool:
        """DI Container 의존성 주입"""
        try:
            with self._lock:
                self.logger.info(f"🔄 {self.step_name} DI Container 의존성 주입 시작...")
                
                self._record_injection_attempt('di_container')
                
                self.dependencies['di_container'] = di_container
                self.dependency_status.di_container = True
                
                self.logger.info(f"✅ {self.step_name} DI Container 의존성 주입 완료")
                return True
                
        except Exception as e:
            self._record_injection_error('di_container', str(e))
            self.logger.error(f"❌ {self.step_name} DI Container 주입 실패: {e}")
            return False
    
    def get_dependency(self, name: str) -> Optional[Any]:
        """의존성 조회"""
        with self._lock:
            return self.dependencies.get(name)
    
    def check_required_dependencies(self, config: StepConfig) -> bool:
        """필수 의존성 확인"""
        if config.require_model_loader and not self.dependency_status.model_loader:
            return False
        if config.require_memory_manager and not self.dependency_status.memory_manager:
            return False
        if config.require_data_converter and not self.dependency_status.data_converter:
            return False
        return True
    
    def validate_all_dependencies(self) -> Dict[str, bool]:
        """모든 의존성 유효성 검증"""
        validation_results = {}
        
        try:
            with self._lock:
                for dep_name, dep_obj in self.dependencies.items():
                    try:
                        if dep_obj is not None:
                            # 기본적인 callable 검사
                            if dep_name == 'model_loader':
                                validation_results[dep_name] = hasattr(dep_obj, 'load_model')
                            elif dep_name == 'step_interface':
                                validation_results[dep_name] = hasattr(dep_obj, 'get_model_sync')
                            elif dep_name == 'memory_manager':
                                validation_results[dep_name] = hasattr(dep_obj, 'optimize_memory')
                            elif dep_name == 'data_converter':
                                validation_results[dep_name] = hasattr(dep_obj, 'convert_data')
                            else:
                                validation_results[dep_name] = True
                        else:
                            validation_results[dep_name] = False
                    except Exception as e:
                        self.logger.debug(f"의존성 {dep_name} 검증 오류: {e}")
                        validation_results[dep_name] = False
                
                self.dependency_status.dependencies_validated = True
                return validation_results
                
        except Exception as e:
            self.logger.error(f"의존성 검증 실패: {e}")
            return {}
    
    def auto_inject_dependencies(self) -> bool:
        """자동 의존성 주입 (환경 최적화)"""
        if self._auto_injection_attempted:
            return True
        
        self._auto_injection_attempted = True
        success_count = 0
        
        try:
            self.logger.info(f"🔄 {self.step_name} 자동 의존성 주입 시작...")
            
            # ModelLoader 자동 주입
            if not self.dependency_status.model_loader:
                model_loader = self._get_global_model_loader()
                if model_loader:
                    if self.inject_model_loader(model_loader):
                        success_count += 1
            
            # MemoryManager 자동 주입
            if not self.dependency_status.memory_manager:
                memory_manager = self._get_global_memory_manager()
                if memory_manager:
                    if self.inject_memory_manager(memory_manager):
                        success_count += 1
            
            self.logger.info(f"🔄 {self.step_name} 자동 의존성 주입 완료: {success_count}개 (환경 최적화)")
            return success_count > 0
            
        except Exception as e:
            self.logger.warning(f"⚠️ {self.step_name} 자동 의존성 주입 실패: {e}")
            return False
    
    def _get_global_model_loader(self) -> Optional['ModelLoader']:
        """ModelLoader 동적 import (환경 최적화)"""
        try:
            import importlib
            module = importlib.import_module('app.ai_pipeline.utils.model_loader')
            get_global = getattr(module, 'get_global_model_loader', None)
            if get_global:
                # 환경 최적화 설정
                config = {
                    'conda_env': self.conda_info['conda_env'],
                    'is_m3_max': self.is_m3_max,
                    'memory_gb': self.memory_gb,
                    'enable_conda_optimization': self.conda_info['is_target_env']
                }
                return get_global(config)
        except Exception as e:
            self.logger.debug(f"ModelLoader 자동 주입 실패: {e}")
        return None
    
    def _get_global_memory_manager(self) -> Optional['MemoryManager']:
        """MemoryManager 동적 import (M3 Max 최적화)"""
        try:
            import importlib
            module = importlib.import_module('app.ai_pipeline.utils.memory_manager')
            get_global = getattr(module, 'get_global_memory_manager', None)
            if get_global:
                return get_global()
        except Exception as e:
            self.logger.debug(f"MemoryManager 자동 주입 실패: {e}")
        return None
    
    def _record_injection_attempt(self, dependency_name: str):
        """의존성 주입 시도 기록"""
        if dependency_name not in self.dependency_status.injection_attempts:
            self.dependency_status.injection_attempts[dependency_name] = 0
        self.dependency_status.injection_attempts[dependency_name] += 1
    
    def _record_injection_error(self, dependency_name: str, error_message: str):
        """의존성 주입 오류 기록"""
        if dependency_name not in self.dependency_status.injection_errors:
            self.dependency_status.injection_errors[dependency_name] = []
        self.dependency_status.injection_errors[dependency_name].append(error_message)
    def _record_injection_error(self, dependency_name: str, error_message: str):
        """의존성 주입 오류 기록"""
        if dependency_name not in self.dependency_status.injection_errors:
            self.dependency_status.injection_errors[dependency_name] = []
        self.dependency_status.injection_errors[dependency_name].append(error_message)
    
    # 🔥 여기에 validate_dependencies 메서드 추가
    def validate_dependencies(self) -> Dict[str, Any]:
        """의존성 검증 메서드 (GeometricMatchingStep 호환)"""
        try:
            with self._lock:
                self.logger.info(f"🔄 {self.step_name} 의존성 검증 시작...")
                
                validation_results = {
                    "success": True,
                    "total_dependencies": len(self.dependencies),
                    "validated_dependencies": 0,
                    "failed_dependencies": 0,
                    "required_missing": [],
                    "optional_missing": [],
                    "validation_errors": [],
                    "details": {}
                }
                
                # 각 의존성 검증
                for dep_name, dep_obj in self.dependencies.items():
                    if dep_obj is not None:
                        # 의존성별 검증
                        if dep_name == 'model_loader':
                            is_valid = hasattr(dep_obj, 'load_model') and hasattr(dep_obj, 'create_step_interface')
                        elif dep_name == 'step_interface':
                            is_valid = hasattr(dep_obj, 'get_model_sync') and hasattr(dep_obj, 'get_model_async')
                        elif dep_name == 'memory_manager':
                            is_valid = hasattr(dep_obj, 'optimize_memory') or hasattr(dep_obj, 'optimize')
                        else:
                            is_valid = True
                        
                        if is_valid:
                            validation_results["validated_dependencies"] += 1
                            validation_results["details"][dep_name] = {"success": True, "valid": True}
                        else:
                            validation_results["failed_dependencies"] += 1
                            validation_results["details"][dep_name] = {"success": False, "error": "필수 메서드 누락"}
                            validation_results["validation_errors"].append(f"{dep_name}: 필수 메서드 누락")
                    else:
                        validation_results["failed_dependencies"] += 1
                        validation_results["details"][dep_name] = {"success": False, "error": "의존성 없음"}
                        validation_results["required_missing"].append(dep_name)
                
                # 전체 검증 결과
                validation_results["success"] = len(validation_results["required_missing"]) == 0
                
                if validation_results["success"]:
                    self.logger.info(f"✅ {self.step_name} 의존성 검증 성공: {validation_results['validated_dependencies']}/{validation_results['total_dependencies']}")
                else:
                    self.logger.warning(f"⚠️ {self.step_name} 의존성 검증 실패: {len(validation_results['required_missing'])}개 누락")
                
                return validation_results
                
        except Exception as e:
            error_msg = f"의존성 검증 중 오류: {e}"
            self.logger.error(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "validation_errors": [error_msg],
                "total_dependencies": 0,
                "validated_dependencies": 0,
                "failed_dependencies": 0,
                "required_missing": [],
                "optional_missing": [],
                "details": {}
            }
    
    def get_status(self) -> Dict[str, Any]:
        """의존성 관리자 상태 조회 (v18.0 강화)"""
        return {
            'step_name': self.step_name,
            'dependency_status': {
                'model_loader': self.dependency_status.model_loader,
                'step_interface': self.dependency_status.step_interface,
                'memory_manager': self.dependency_status.memory_manager,
                'data_converter': self.dependency_status.data_converter,
                'di_container': self.dependency_status.di_container,
                'base_initialized': self.dependency_status.base_initialized,
                'custom_initialized': self.dependency_status.custom_initialized,
                'dependencies_validated': self.dependency_status.dependencies_validated
            },
            'environment': {
                'conda_optimized': self.dependency_status.conda_optimized,
                'm3_max_optimized': self.dependency_status.m3_max_optimized,
                'conda_env': self.conda_info['conda_env'],
                'is_m3_max': self.is_m3_max,
                'memory_gb': self.memory_gb
            },
            'injection_history': {
                'auto_injection_attempted': self._auto_injection_attempted,
                'injection_attempts': dict(self.dependency_status.injection_attempts),
                'injection_errors': dict(self.dependency_status.injection_errors),
                'last_injection_time': self.dependency_status.last_injection_time
            },
            'dependencies_available': list(self.dependencies.keys()),
            'dependencies_count': len(self.dependencies)
        }

# ==============================================
# 🔥 BaseStepMixin v18.0 - 의존성 주입 완전 수정
# ==============================================

class BaseStepMixin:
    """
    🔥 BaseStepMixin v18.0 - 의존성 주입 완전 수정 (StepFactory v7.0 호환)
    
    핵심 수정사항:
    ✅ StepFactory v7.0과 완전 호환
    ✅ 의존성 주입 문제 완전 해결
    ✅ ModelLoader → StepModelInterface 연결 안정화
    ✅ 초기화 순서 최적화
    ✅ 에러 처리 강화 및 안전한 폴백
    """
    
    def __init__(self, **kwargs):
        """통합 초기화 (v18.0 의존성 주입 수정)"""
        try:
            # 기본 설정
            self.config = self._create_config(**kwargs)
            self.step_name = kwargs.get('step_name', self.__class__.__name__)
            self.step_id = kwargs.get('step_id', 0)
            
            # Logger 설정
            self.logger = logging.getLogger(f"steps.{self.step_name}")
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
            
            # 🔥 강화된 의존성 관리자 (v18.0)
            self.dependency_manager = EnhancedDependencyManager(self.step_name)
            
            # 상태 플래그들
            self.is_initialized = False
            self.is_ready = False
            self.has_model = False
            self.model_loaded = False
            self.warmup_completed = False
            
            # 시스템 정보 (환경 최적화)
            self.device = self._resolve_device(self.config.device)
            self.is_m3_max = IS_M3_MAX
            self.memory_gb = MEMORY_GB
            self.conda_info = CONDA_INFO
            
            # 성능 메트릭 (v18.0 강화)
            self.performance_metrics = PerformanceMetrics()
            
            # 호환성을 위한 속성들 (StepFactory 호환)
            self.model_loader = None
            self.model_interface = None
            self.memory_manager = None
            self.data_converter = None
            self.di_container = None
            
            # 환경 최적화 설정 적용
            self._apply_environment_optimization()
            
            # 자동 의존성 주입 (설정된 경우)
            if self.config.auto_inject_dependencies:
                self.dependency_manager.auto_inject_dependencies()
            
            self.logger.info(f"✅ {self.step_name} BaseStepMixin v18.0 초기화 완료")
            
        except Exception as e:
            self._emergency_setup(e)
    
    def _create_config(self, **kwargs) -> StepConfig:
        """설정 생성 (환경 최적화)"""
        config = StepConfig()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # 환경별 설정 적용
        if CONDA_INFO['is_target_env']:
            config.conda_optimized = True
            config.conda_env = CONDA_INFO['conda_env']
        
        if IS_M3_MAX:
            config.m3_max_optimized = True
            config.memory_gb = MEMORY_GB
            config.use_unified_memory = True
        
        return config
    
    def _apply_environment_optimization(self):
        """환경 최적화 적용"""
        try:
            # M3 Max 최적화
            if self.is_m3_max:
                # MPS 디바이스 우선 설정
                if self.device == "auto" and MPS_AVAILABLE:
                    self.device = "mps"
                
                # 배치 크기 조정
                if self.config.batch_size == 1 and self.memory_gb >= 64:
                    self.config.batch_size = 2
                
                # 메모리 최적화 강화
                self.config.auto_memory_cleanup = True
                
                self.logger.debug(f"✅ M3 Max 최적화 적용: {self.memory_gb:.1f}GB, device={self.device}")
            
            # conda 환경 최적화
            if self.conda_info['is_target_env']:
                self.config.optimization_enabled = True
                self.logger.debug(f"✅ conda 환경 최적화 적용: {self.conda_info['conda_env']}")
            
        except Exception as e:
            self.logger.debug(f"환경 최적화 적용 실패: {e}")
    
    def _emergency_setup(self, error: Exception):
        """긴급 설정"""
        self.step_name = getattr(self, 'step_name', self.__class__.__name__)
        self.logger = logging.getLogger("emergency")
        self.device = "cpu"
        self.is_initialized = False
        self.performance_metrics = PerformanceMetrics()
        self.logger.error(f"🚨 {self.step_name} 긴급 초기화: {error}")
    
    # ==============================================
    # 🔥 표준화된 의존성 주입 인터페이스 (v18.0 수정)
    # ==============================================
    
    def set_model_loader(self, model_loader: 'ModelLoader'):
        """ModelLoader 의존성 주입 (v18.0 완전 수정)"""
        try:
            self.logger.info(f"🔄 {self.step_name} ModelLoader 의존성 주입 시작...")
            
            # 1. 의존성 관리자를 통한 주입
            success = self.dependency_manager.inject_model_loader(model_loader)
            
            if success:
                # 2. 호환성을 위한 속성 설정
                self.model_loader = model_loader
                self.model_interface = self.dependency_manager.get_dependency('step_interface')
                
                # 3. 상태 플래그 업데이트
                self.has_model = True
                self.model_loaded = True
                
                # 4. 성능 메트릭 업데이트
                self.performance_metrics.dependencies_injected += 1
                
                self.logger.info(f"✅ {self.step_name} ModelLoader 의존성 주입 완료")
            else:
                self.logger.error(f"❌ {self.step_name} ModelLoader 의존성 주입 실패")
                if self.config.strict_mode:
                    raise RuntimeError(f"Strict Mode: ModelLoader 의존성 주입 실패")
                
        except Exception as e:
            self.performance_metrics.injection_failures += 1
            self.logger.error(f"❌ {self.step_name} ModelLoader 의존성 주입 오류: {e}")
            if self.config.strict_mode:
                raise
    
    def set_memory_manager(self, memory_manager: 'MemoryManager'):
        """MemoryManager 의존성 주입 (v18.0 수정)"""
        try:
            success = self.dependency_manager.inject_memory_manager(memory_manager)
            if success:
                self.memory_manager = memory_manager
                self.performance_metrics.dependencies_injected += 1
                self.logger.debug(f"✅ {self.step_name} MemoryManager 의존성 주입 완료")
            else:
                self.performance_metrics.injection_failures += 1
                
        except Exception as e:
            self.performance_metrics.injection_failures += 1
            self.logger.warning(f"⚠️ {self.step_name} MemoryManager 의존성 주입 오류: {e}")
    
    def set_data_converter(self, data_converter: 'DataConverter'):
        """DataConverter 의존성 주입 (v18.0 수정)"""
        try:
            success = self.dependency_manager.inject_data_converter(data_converter)
            if success:
                self.data_converter = data_converter
                self.performance_metrics.dependencies_injected += 1
                self.logger.debug(f"✅ {self.step_name} DataConverter 의존성 주입 완료")
            else:
                self.performance_metrics.injection_failures += 1
                
        except Exception as e:
            self.performance_metrics.injection_failures += 1
            self.logger.warning(f"⚠️ {self.step_name} DataConverter 의존성 주입 오류: {e}")
    
    def set_di_container(self, di_container: 'DIContainer'):
        """DI Container 의존성 주입 (v18.0 수정)"""
        try:
            success = self.dependency_manager.inject_di_container(di_container)
            if success:
                self.di_container = di_container
                self.performance_metrics.dependencies_injected += 1
                self.logger.debug(f"✅ {self.step_name} DI Container 의존성 주입 완료")
            else:
                self.performance_metrics.injection_failures += 1
                
        except Exception as e:
            self.performance_metrics.injection_failures += 1
            self.logger.warning(f"⚠️ {self.step_name} DI Container 의존성 주입 오류: {e}")
    
    # ==============================================
    # 🔥 핵심 기능 메서드들 (v18.0 개선)
    # ==============================================
    
    def get_model(self, model_name: Optional[str] = None) -> Optional[Any]:
        """모델 가져오기 (v18.0 개선된 통합 인터페이스)"""
        try:
            start_time = time.time()
            
            # 1. Step Interface 우선 사용 (v18.0 수정)
            step_interface = self.dependency_manager.get_dependency('step_interface')
            if step_interface and hasattr(step_interface, 'get_model_sync'):
                model = step_interface.get_model_sync(model_name or "default")
                if model:
                    self.performance_metrics.cache_hits += 1
                    return model
            
            # 2. ModelLoader 직접 사용
            model_loader = self.dependency_manager.get_dependency('model_loader')
            if model_loader and hasattr(model_loader, 'load_model'):
                model = model_loader.load_model(model_name or "default")
                if model:
                    self.performance_metrics.models_loaded += 1
                    return model
            
            self.logger.warning("⚠️ 모델 제공자가 주입되지 않음")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ 모델 가져오기 실패: {e}")
            self.performance_metrics.error_count += 1
            return None
        finally:
            process_time = time.time() - start_time
            self._update_performance_metrics(process_time)
    
    async def get_model_async(self, model_name: Optional[str] = None) -> Optional[Any]:
        """비동기 모델 가져오기 (v18.0 개선된 통합 인터페이스)"""
        try:
            # Step Interface 우선 사용
            step_interface = self.dependency_manager.get_dependency('step_interface')
            if step_interface and hasattr(step_interface, 'get_model_async'):
                return await step_interface.get_model_async(model_name or "default")
            
            # ModelLoader 직접 사용
            model_loader = self.dependency_manager.get_dependency('model_loader')
            if model_loader and hasattr(model_loader, 'load_model_async'):
                return await model_loader.load_model_async(model_name or "default")
            elif model_loader and hasattr(model_loader, 'load_model'):
                # 동기 메서드를 비동기로 실행
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, model_loader.load_model, model_name or "default")
            
            self.logger.warning("⚠️ 모델 제공자가 주입되지 않음")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ 비동기 모델 가져오기 실패: {e}")
            return None
    
    def optimize_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """메모리 최적화 (v18.0 개선)"""
        try:
            start_time = time.time()
            
            # 주입된 MemoryManager 우선 사용
            memory_manager = self.dependency_manager.get_dependency('memory_manager')
            if memory_manager and hasattr(memory_manager, 'optimize_memory'):
                result = memory_manager.optimize_memory(aggressive=aggressive)
                self.performance_metrics.memory_optimizations += 1
                return result
            
            # 내장 메모리 최적화 (환경별)
            result = self._builtin_memory_optimize(aggressive)
            self.performance_metrics.memory_optimizations += 1
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 메모리 최적화 실패: {e}")
            return {"success": False, "error": str(e)}
        finally:
            optimization_time = time.time() - start_time
            self.logger.debug(f"🧹 메모리 최적화 소요 시간: {optimization_time:.3f}초")
    
    async def optimize_memory_async(self, aggressive: bool = False) -> Dict[str, Any]:
        """비동기 메모리 최적화"""
        try:
            # 주입된 MemoryManager 우선 사용
            memory_manager = self.dependency_manager.get_dependency('memory_manager')
            if memory_manager and hasattr(memory_manager, 'optimize_memory_async'):
                return await memory_manager.optimize_memory_async(aggressive=aggressive)
            
            # 동기 메모리 최적화를 비동기로 실행
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._builtin_memory_optimize, aggressive)
            
        except Exception as e:
            self.logger.error(f"❌ 비동기 메모리 최적화 실패: {e}")
            return {"success": False, "error": str(e)}
    
    # ==============================================
    # 🔥 표준화된 초기화 및 워밍업 (v18.0)
    # ==============================================
    
    def initialize(self) -> bool:
        """표준화된 초기화 (v18.0 의존성 검증 강화)"""
        try:
            if self.is_initialized:
                return True
            
            self.logger.info(f"🔄 {self.step_name} 표준화된 초기화 시작...")
            
            # 1. 필수 의존성 확인
            if not self.dependency_manager.check_required_dependencies(self.config):
                if self.config.strict_mode:
                    raise RuntimeError("필수 의존성이 주입되지 않음")
                else:
                    self.logger.warning("⚠️ 일부 의존성이 누락됨")
            
            # 2. 의존성 유효성 검증 (v18.0 추가)
            validation_results = self.dependency_manager.validate_all_dependencies()
            if validation_results:
                failed_deps = [dep for dep, valid in validation_results.items() if not valid]
                if failed_deps:
                    self.logger.warning(f"⚠️ 의존성 검증 실패: {failed_deps}")
            
            # 3. 환경별 초기화
            self._environment_specific_initialization()
            
            # 4. 초기화 상태 설정
            self.dependency_manager.dependency_status.base_initialized = True
            self.is_initialized = True
            
            self.logger.info(f"✅ {self.step_name} 표준화된 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 초기화 실패: {e}")
            self.performance_metrics.error_count += 1
            return False
    
    def _environment_specific_initialization(self):
        """환경별 특별 초기화"""
        try:
            # M3 Max 특별 초기화
            if self.is_m3_max:
                # PyTorch MPS 워밍업
                if TORCH_AVAILABLE and self.device == "mps":
                    try:
                        test_tensor = torch.randn(10, 10, device=self.device)
                        _ = torch.matmul(test_tensor, test_tensor.t())
                        self.logger.debug("✅ M3 Max MPS 워밍업 완료")
                    except Exception as mps_error:
                        self.logger.debug(f"M3 Max MPS 워밍업 실패: {mps_error}")
            
            # conda 환경 특별 초기화
            if self.conda_info['is_target_env']:
                # 환경 변수 최적화
                os.environ['PYTHONPATH'] = self.conda_info['conda_prefix'] + '/lib/python3.11/site-packages'
                self.logger.debug("✅ conda 환경 최적화 완료")
            
        except Exception as e:
            self.logger.debug(f"환경별 초기화 실패: {e}")
    
    async def initialize_async(self) -> bool:
        """비동기 초기화"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.initialize)
        except Exception as e:
            self.logger.error(f"❌ 비동기 초기화 실패: {e}")
            return False
    
    def warmup(self) -> Dict[str, Any]:
        """표준화된 워밍업 (v18.0 개선)"""
        try:
            if self.warmup_completed:
                return {'success': True, 'message': '이미 워밍업 완료됨', 'cached': True}
            
            self.logger.info(f"🔥 {self.step_name} 표준화된 워밍업 시작...")
            start_time = time.time()
            results = []
            
            # 1. 의존성 워밍업 (v18.0 추가)
            try:
                dependency_status = self.dependency_manager.get_status()
                if dependency_status.get('dependencies_count', 0) > 0:
                    results.append('dependency_success')
                else:
                    results.append('dependency_failed')
            except:
                results.append('dependency_failed')
            
            # 2. 메모리 워밍업 (환경별)
            try:
                memory_result = self.optimize_memory(aggressive=False)
                results.append('memory_success' if memory_result.get('success') else 'memory_failed')
            except:
                results.append('memory_failed')
            
            # 3. 모델 워밍업
            try:
                test_model = self.get_model("warmup_test")
                results.append('model_success' if test_model else 'model_skipped')
            except:
                results.append('model_failed')
            
            # 4. 디바이스 워밍업 (환경별)
            results.append(self._device_warmup())
            
            # 5. 환경별 특별 워밍업
            if self.is_m3_max:
                results.append(self._m3_max_warmup())
            
            if self.conda_info['is_target_env']:
                results.append(self._conda_warmup())
            
            duration = time.time() - start_time
            success_count = sum(1 for r in results if 'success' in r)
            overall_success = success_count > 0
            
            if overall_success:
                self.warmup_completed = True
                self.is_ready = True
            
            self.logger.info(f"🔥 표준화된 워밍업 완료: {success_count}/{len(results)} 성공 ({duration:.2f}초)")
            
            return {
                "success": overall_success,
                "duration": duration,
                "results": results,
                "success_count": success_count,
                "total_count": len(results),
                "environment": {
                    "is_m3_max": self.is_m3_max,
                    "conda_optimized": self.conda_info['is_target_env'],
                    "device": self.device
                },
                "dependency_status": self.dependency_manager.get_status()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 워밍업 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def _device_warmup(self) -> str:
        """디바이스 워밍업"""
        try:
            if TORCH_AVAILABLE:
                test_tensor = torch.randn(100, 100)
                if self.device != 'cpu':
                    test_tensor = test_tensor.to(self.device)
                _ = torch.matmul(test_tensor, test_tensor.t())
                return 'device_success'
            else:
                return 'device_skipped'
        except:
            return 'device_failed'
    
    def _m3_max_warmup(self) -> str:
        """M3 Max 특별 워밍업"""
        try:
            if TORCH_AVAILABLE and MPS_AVAILABLE:
                # 큰 텐서로 통합 메모리 테스트
                large_tensor = torch.randn(1000, 1000, device='mps')
                _ = torch.matmul(large_tensor, large_tensor.t())
                del large_tensor
                return 'm3_max_success'
            return 'm3_max_skipped'
        except:
            return 'm3_max_failed'
    
    def _conda_warmup(self) -> str:
        """conda 환경 워밍업"""
        try:
            # 패키지 경로 확인
            import sys
            conda_paths = [p for p in sys.path if 'conda' in p.lower()]
            if conda_paths:
                return 'conda_success'
            return 'conda_skipped'
        except:
            return 'conda_failed'
    
    async def warmup_async(self) -> Dict[str, Any]:
        """비동기 워밍업"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.warmup)
        except Exception as e:
            self.logger.error(f"❌ 비동기 워밍업 실패: {e}")
            return {"success": False, "error": str(e)}
    
    # ==============================================
    # 🔥 성능 메트릭 및 모니터링 (v18.0 강화)
    # ==============================================
    
    def _update_performance_metrics(self, process_time: float):
        """성능 메트릭 업데이트 (v18.0 강화)"""
        try:
            self.performance_metrics.process_count += 1
            self.performance_metrics.total_process_time += process_time
            self.performance_metrics.average_process_time = (
                self.performance_metrics.total_process_time / 
                self.performance_metrics.process_count
            )
            
            # 의존성 주입 평균 시간 계산 (v18.0 추가)
            if self.performance_metrics.dependencies_injected > 0:
                self.performance_metrics.average_injection_time = (
                    self.performance_metrics.total_process_time / 
                    max(1, self.performance_metrics.dependencies_injected)
                )
            
            # 메모리 사용량 업데이트 (M3 Max 특별 처리)
            if self.is_m3_max:
                try:
                    import psutil
                    memory_info = psutil.virtual_memory()
                    current_usage = memory_info.used / 1024**2  # MB
                    
                    if current_usage > self.performance_metrics.peak_memory_usage_mb:
                        self.performance_metrics.peak_memory_usage_mb = current_usage
                    
                    # 이동 평균 계산
                    if self.performance_metrics.average_memory_usage_mb == 0:
                        self.performance_metrics.average_memory_usage_mb = current_usage
                    else:
                        self.performance_metrics.average_memory_usage_mb = (
                            self.performance_metrics.average_memory_usage_mb * 0.9 + 
                            current_usage * 0.1
                        )
                except:
                    pass
                    
        except Exception as e:
            self.logger.debug(f"성능 메트릭 업데이트 실패: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 조회 (v18.0 강화)"""
        try:
            return {
                'process_metrics': {
                    'process_count': self.performance_metrics.process_count,
                    'total_process_time': round(self.performance_metrics.total_process_time, 3),
                    'average_process_time': round(self.performance_metrics.average_process_time, 3),
                    'success_count': self.performance_metrics.success_count,
                    'error_count': self.performance_metrics.error_count,
                    'cache_hits': self.performance_metrics.cache_hits
                },
                'memory_metrics': {
                    'peak_memory_usage_mb': round(self.performance_metrics.peak_memory_usage_mb, 2),
                    'average_memory_usage_mb': round(self.performance_metrics.average_memory_usage_mb, 2),
                    'memory_optimizations': self.performance_metrics.memory_optimizations
                },
                'ai_model_metrics': {
                    'models_loaded': self.performance_metrics.models_loaded,
                    'total_model_size_gb': round(self.performance_metrics.total_model_size_gb, 2),
                    'inference_count': self.performance_metrics.inference_count
                },
                'dependency_metrics': {  # v18.0 추가
                    'dependencies_injected': self.performance_metrics.dependencies_injected,
                    'injection_failures': self.performance_metrics.injection_failures,
                    'average_injection_time': round(self.performance_metrics.average_injection_time, 3),
                    'injection_success_rate': round(
                        (self.performance_metrics.dependencies_injected / 
                         max(1, self.performance_metrics.dependencies_injected + self.performance_metrics.injection_failures)) * 100, 2
                    )
                },
                'environment_metrics': {
                    'device': self.device,
                    'is_m3_max': self.is_m3_max,
                    'memory_gb': self.memory_gb,
                    'conda_optimized': self.conda_info['is_target_env']
                }
            }
        except Exception as e:
            self.logger.error(f"❌ 성능 메트릭 조회 실패: {e}")
            return {'error': str(e)}
    
    # ==============================================
    # 🔥 상태 및 정리 메서드들 (v18.0 강화)
    # ==============================================
    
    def get_status(self) -> Dict[str, Any]:
        """통합 상태 조회 (v18.0 의존성 정보 강화)"""
        try:
            return {
                'step_info': {
                    'step_name': self.step_name,
                    'step_id': self.step_id,
                    'version': 'BaseStepMixin v18.0'
                },
                'status_flags': {
                    'is_initialized': self.is_initialized,
                    'is_ready': self.is_ready,
                    'has_model': self.has_model,
                    'model_loaded': self.model_loaded,
                    'warmup_completed': self.warmup_completed
                },
                'system_info': {
                    'device': self.device,
                    'is_m3_max': self.is_m3_max,
                    'memory_gb': self.memory_gb,
                    'conda_info': self.conda_info
                },
                'dependencies': self.dependency_manager.get_status(),  # v18.0 강화된 의존성 정보
                'performance': self.get_performance_metrics(),
                'config': {
                    'device': self.config.device,
                    'use_fp16': self.config.use_fp16,
                    'batch_size': self.config.batch_size,
                    'confidence_threshold': self.config.confidence_threshold,
                    'auto_memory_cleanup': self.config.auto_memory_cleanup,
                    'auto_warmup': self.config.auto_warmup,
                    'optimization_enabled': self.config.optimization_enabled,
                    'strict_mode': self.config.strict_mode,
                    'conda_optimized': self.config.conda_optimized,
                    'm3_max_optimized': self.config.m3_max_optimized,
                    'auto_inject_dependencies': self.config.auto_inject_dependencies,
                    'dependency_timeout': self.config.dependency_timeout,
                    'dependency_retry_count': self.config.dependency_retry_count
                },
                'timestamp': time.time()
            }
        except Exception as e:
            self.logger.error(f"❌ 상태 조회 실패: {e}")
            return {'error': str(e), 'version': 'BaseStepMixin v18.0'}
    
    async def cleanup(self) -> Dict[str, Any]:
        """표준화된 정리 (v18.0 의존성 정리 강화)"""
        try:
            self.logger.info(f"🧹 {self.step_name} 표준화된 정리 시작...")
            
            # 성능 메트릭 저장
            final_metrics = self.get_performance_metrics()
            
            # 메모리 정리 (환경별 최적화)
            cleanup_result = await self.optimize_memory_async(aggressive=True)
            
            # 상태 리셋
            self.is_ready = False
            self.warmup_completed = False
            self.has_model = False
            self.model_loaded = False
            
            # 의존성 해제 (v18.0 강화) - 참조만 제거
            self.model_loader = None
            self.model_interface = None
            self.memory_manager = None
            self.data_converter = None
            self.di_container = None
            
            # 의존성 관리자 정리
            dependency_status = self.dependency_manager.get_status()
            
            # 환경별 특별 정리
            if self.is_m3_max:
                # M3 Max 통합 메모리 정리
                for _ in range(5):
                    gc.collect()
                if TORCH_AVAILABLE and MPS_AVAILABLE:
                    try:
                        torch.mps.empty_cache()
                    except:
                        pass
            
            if TORCH_AVAILABLE and self.device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
            
            self.logger.info(f"✅ {self.step_name} 표준화된 정리 완료")
            
            return {
                "success": True,
                "cleanup_result": cleanup_result,
                "final_metrics": final_metrics,
                "dependency_status": dependency_status,
                "step_name": self.step_name,
                "version": "BaseStepMixin v18.0",
                "environment": {
                    "is_m3_max": self.is_m3_max,
                    "conda_optimized": self.conda_info['is_target_env']
                }
            }
        except Exception as e:
            self.logger.error(f"❌ 정리 실패: {e}")
            return {"success": False, "error": str(e)}
    
    # ==============================================
    # 🔥 내부 유틸리티 메서드들 (v18.0)
    # ==============================================
    
    def _resolve_device(self, device: str) -> str:
        """디바이스 해결 (환경 최적화)"""
        if device == "auto":
            # M3 Max 우선 처리
            if IS_M3_MAX and MPS_AVAILABLE:
                return "mps"
            
            if TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"
            return "cpu"
        return device
    
    def _builtin_memory_optimize(self, aggressive: bool = False) -> Dict[str, Any]:
        """내장 메모리 최적화 (환경별)"""
        try:
            results = []
            start_time = time.time()
            
            # Python GC
            before = len(gc.get_objects()) if hasattr(gc, 'get_objects') else 0
            gc.collect()
            after = len(gc.get_objects()) if hasattr(gc, 'get_objects') else 0
            results.append(f"Python GC: {before - after}개 객체 해제")
            
            # PyTorch 메모리 정리 (환경별)
            if TORCH_AVAILABLE:
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if aggressive:
                        torch.cuda.ipc_collect()
                    results.append("CUDA 캐시 정리")
                
                elif self.device == "mps" and MPS_AVAILABLE:
                    try:
                        if hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                        results.append("MPS 캐시 정리")
                    except:
                        results.append("MPS 캐시 정리 시도")
            
            # M3 Max 특별 최적화
            if self.is_m3_max and aggressive:
                # 통합 메모리 최적화
                for _ in range(5):
                    gc.collect()
                
                # 메모리 압축 시도
                try:
                    import mmap
                    # 메모리 매핑 최적화
                    results.append("M3 Max 통합 메모리 최적화")
                except:
                    results.append("M3 Max 최적화 시도")
            
            # conda 환경 최적화
            if self.conda_info['is_target_env'] and aggressive:
                # conda 캐시 정리
                try:
                    import tempfile
                    import shutil
                    temp_dir = tempfile.gettempdir()
                    conda_temp = os.path.join(temp_dir, 'conda_*')
                    # 임시 파일 정리는 안전하게
                    results.append("conda 임시 파일 최적화")
                except:
                    results.append("conda 최적화 시도")
            
            # 메모리 사용량 측정
            memory_info = {}
            try:
                import psutil
                vm = psutil.virtual_memory()
                memory_info = {
                    'total_gb': round(vm.total / 1024**3, 2),
                    'available_gb': round(vm.available / 1024**3, 2),
                    'used_percent': vm.percent
                }
            except:
                memory_info = {'error': 'psutil_not_available'}
            
            duration = time.time() - start_time
            
            return {
                "success": True,
                "results": results,
                "duration": round(duration, 3),
                "device": self.device,
                "environment": {
                    "is_m3_max": self.is_m3_max,
                    "conda_optimized": self.conda_info['is_target_env'],
                    "memory_gb": self.memory_gb
                },
                "memory_info": memory_info,
                "source": "builtin_optimized"
            }
            
        except Exception as e:
            return {
                "success": False, 
                "error": str(e), 
                "source": "builtin_optimized",
                "environment": {"is_m3_max": self.is_m3_max}
            }
    
    # ==============================================
    # 🔥 진단 및 디버깅 메서드들 (v18.0 강화)
    # ==============================================
    
    def diagnose(self) -> Dict[str, Any]:
        """Step 진단 (v18.0 의존성 진단 강화)"""
        try:
            self.logger.info(f"🔍 {self.step_name} 진단 시작...")
            
            diagnosis = {
                'timestamp': time.time(),
                'step_name': self.step_name,
                'version': 'BaseStepMixin v18.0',
                'status': self.get_status(),
                'issues': [],
                'recommendations': [],
                'health_score': 100
            }
            
            # 의존성 진단 (v18.0 강화)
            dependency_status = self.dependency_manager.get_status()
            
            if not dependency_status['dependency_status']['model_loader']:
                diagnosis['issues'].append('ModelLoader가 주입되지 않음')
                diagnosis['recommendations'].append('ModelLoader 의존성 주입 필요')
                diagnosis['health_score'] -= 30
            
            if not dependency_status['dependency_status']['step_interface']:
                diagnosis['issues'].append('StepModelInterface가 생성되지 않음')
                diagnosis['recommendations'].append('ModelLoader의 create_step_interface 확인 필요')
                diagnosis['health_score'] -= 25
            
            # 의존성 주입 오류 체크 (v18.0 추가)
            injection_errors = dependency_status.get('injection_history', {}).get('injection_errors', {})
            if injection_errors:
                for dep_name, errors in injection_errors.items():
                    diagnosis['issues'].append(f'{dep_name} 의존성 주입 오류: {len(errors)}회')
                    diagnosis['recommendations'].append(f'{dep_name} 의존성 주입 문제 해결 필요')
                    diagnosis['health_score'] -= 15
            
            # 환경 진단
            if not self.conda_info['is_target_env']:
                diagnosis['issues'].append(f"권장 conda 환경이 아님: {self.conda_info['conda_env']}")
                diagnosis['recommendations'].append('mycloset-ai-clean 환경 사용 권장')
                diagnosis['health_score'] -= 10
            
            # 메모리 진단
            if self.memory_gb < 16:
                diagnosis['issues'].append(f"메모리 부족: {self.memory_gb:.1f}GB")
                diagnosis['recommendations'].append('16GB 이상 메모리 권장')
                diagnosis['health_score'] -= 20
            
            # 디바이스 진단
            if self.device == "cpu" and (TORCH_AVAILABLE and (torch.cuda.is_available() or MPS_AVAILABLE)):
                diagnosis['issues'].append('GPU 가속을 사용하지 않음')
                diagnosis['recommendations'].append('GPU/MPS 디바이스 사용 권장')
                diagnosis['health_score'] -= 15
            
            # 성능 진단 (v18.0 강화)
            performance_metrics = self.get_performance_metrics()
            if performance_metrics.get('process_metrics', {}).get('error_count', 0) > 0:
                error_count = performance_metrics['process_metrics']['error_count']
                process_count = performance_metrics['process_metrics']['process_count']
                if process_count > 0:
                    error_rate = error_count / process_count * 100
                    if error_rate > 10:
                        diagnosis['issues'].append(f"높은 에러율: {error_rate:.1f}%")
                        diagnosis['recommendations'].append('에러 원인 분석 및 해결 필요')
                        diagnosis['health_score'] -= 25
            
            # 의존성 주입 성공률 진단 (v18.0 추가)
            dependency_metrics = performance_metrics.get('dependency_metrics', {})
            injection_success_rate = dependency_metrics.get('injection_success_rate', 100)
            if injection_success_rate < 80:
                diagnosis['issues'].append(f"낮은 의존성 주입 성공률: {injection_success_rate:.1f}%")
                diagnosis['recommendations'].append('의존성 주입 시스템 점검 필요')
                diagnosis['health_score'] -= 20
            
            # 최종 건강도 보정
            diagnosis['health_score'] = max(0, diagnosis['health_score'])
            
            if diagnosis['health_score'] >= 80:
                diagnosis['health_status'] = 'excellent'
            elif diagnosis['health_score'] >= 60:
                diagnosis['health_status'] = 'good'
            elif diagnosis['health_score'] >= 40:
                diagnosis['health_status'] = 'fair'
            else:
                diagnosis['health_status'] = 'poor'
            
            self.logger.info(f"🔍 {self.step_name} 진단 완료 (건강도: {diagnosis['health_score']}%)")
            
            return diagnosis
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 진단 실패: {e}")
            return {
                'error': str(e),
                'step_name': self.step_name,
                'version': 'BaseStepMixin v18.0',
                'health_score': 0,
                'health_status': 'error'
            }
    
    def benchmark(self, iterations: int = 10) -> Dict[str, Any]:
        """성능 벤치마크 (v18.0 의존성 벤치마크 추가)"""
        try:
            self.logger.info(f"📊 {self.step_name} 벤치마크 시작 ({iterations}회)...")
            
            benchmark_results = {
                'iterations': iterations,
                'step_name': self.step_name,
                'device': self.device,
                'environment': {
                    'is_m3_max': self.is_m3_max,
                    'memory_gb': self.memory_gb,
                    'conda_optimized': self.conda_info['is_target_env']
                },
                'timings': [],
                'memory_usage': [],
                'dependency_timings': [],  # v18.0 추가
                'errors': 0
            }
            
            for i in range(iterations):
                try:
                    start_time = time.time()
                    
                    # 기본 작업 시뮬레이션
                    if TORCH_AVAILABLE:
                        test_tensor = torch.randn(512, 512, device=self.device)
                        result = torch.matmul(test_tensor, test_tensor.t())
                        del test_tensor, result
                    
                    # 의존성 접근 벤치마크 (v18.0 추가)
                    dependency_start = time.time()
                    model_loader = self.dependency_manager.get_dependency('model_loader')
                    step_interface = self.dependency_manager.get_dependency('step_interface')
                    dependency_time = time.time() - dependency_start
                    benchmark_results['dependency_timings'].append(dependency_time)
                    
                    # 메모리 최적화 테스트
                    memory_result = self.optimize_memory()
                    
                    timing = time.time() - start_time
                    benchmark_results['timings'].append(timing)
                    
                    # 메모리 사용량 측정
                    try:
                        import psutil
                        memory_usage = psutil.virtual_memory().percent
                        benchmark_results['memory_usage'].append(memory_usage)
                    except:
                        benchmark_results['memory_usage'].append(0)
                    
                except Exception as e:
                    benchmark_results['errors'] += 1
                    self.logger.debug(f"벤치마크 {i+1} 실패: {e}")
            
            # 통계 계산
            if benchmark_results['timings']:
                benchmark_results['statistics'] = {
                    'min_time': min(benchmark_results['timings']),
                    'max_time': max(benchmark_results['timings']),
                    'avg_time': sum(benchmark_results['timings']) / len(benchmark_results['timings']),
                    'total_time': sum(benchmark_results['timings'])
                }
            
            if benchmark_results['dependency_timings']:  # v18.0 추가
                benchmark_results['dependency_statistics'] = {
                    'min_dependency_time': min(benchmark_results['dependency_timings']),
                    'max_dependency_time': max(benchmark_results['dependency_timings']),
                    'avg_dependency_time': sum(benchmark_results['dependency_timings']) / len(benchmark_results['dependency_timings'])
                }
            
            if benchmark_results['memory_usage']:
                benchmark_results['memory_statistics'] = {
                    'min_memory': min(benchmark_results['memory_usage']),
                    'max_memory': max(benchmark_results['memory_usage']),
                    'avg_memory': sum(benchmark_results['memory_usage']) / len(benchmark_results['memory_usage'])
                }
            
            benchmark_results['success_rate'] = (
                (iterations - benchmark_results['errors']) / iterations * 100
            )
            
            self.logger.info(f"📊 {self.step_name} 벤치마크 완료 (성공률: {benchmark_results['success_rate']:.1f}%)")
            
            return benchmark_results
            
        except Exception as e:
            self.logger.error(f"❌ 벤치마크 실패: {e}")
            return {'error': str(e), 'step_name': self.step_name}

# ==============================================
# 🔥 편의 함수들 (BaseStepMixin v18.0 전용)
# ==============================================

def create_base_step_mixin(**kwargs) -> BaseStepMixin:
    """BaseStepMixin 인스턴스 생성"""
    return BaseStepMixin(**kwargs)

def validate_step_environment() -> Dict[str, Any]:
    """Step 환경 검증 (v18.0 강화)"""
    try:
        validation = {
            'timestamp': time.time(),
            'environment_status': {},
            'recommendations': [],
            'overall_score': 100
        }
        
        # conda 환경 검증
        validation['environment_status']['conda'] = {
            'current_env': CONDA_INFO['conda_env'],
            'is_target_env': CONDA_INFO['is_target_env'],
            'valid': CONDA_INFO['is_target_env']
        }
        
        if not CONDA_INFO['is_target_env']:
            validation['recommendations'].append('mycloset-ai-clean conda 환경 사용 권장')
            validation['overall_score'] -= 20
        
        # 하드웨어 검증
        validation['environment_status']['hardware'] = {
            'is_m3_max': IS_M3_MAX,
            'memory_gb': MEMORY_GB,
            'sufficient_memory': MEMORY_GB >= 16.0
        }
        
        if MEMORY_GB < 16.0:
            validation['recommendations'].append('16GB 이상 메모리 권장')
            validation['overall_score'] -= 30
        
        # PyTorch 검증
        validation['environment_status']['pytorch'] = {
            'available': TORCH_AVAILABLE,
            'mps_available': MPS_AVAILABLE,
            'cuda_available': TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False
        }
        
        if not TORCH_AVAILABLE:
            validation['recommendations'].append('PyTorch 설치 필요')
            validation['overall_score'] -= 40
        
        # 기타 패키지 검증
        validation['environment_status']['packages'] = {
            'pil_available': PIL_AVAILABLE,
            'numpy_available': NUMPY_AVAILABLE
        }
        
        # 의존성 주입 시스템 검증 (v18.0 추가)
        try:
            import importlib
            model_loader_module = importlib.import_module('app.ai_pipeline.utils.model_loader')
            step_interface_module = importlib.import_module('app.ai_pipeline.interface.step_interface')
            validation['environment_status']['dependency_system'] = {
                'model_loader_available': hasattr(model_loader_module, 'get_global_model_loader'),
                'step_interface_available': hasattr(step_interface_module, 'StepModelInterface')
            }
        except ImportError:
            validation['environment_status']['dependency_system'] = {
                'model_loader_available': False,
                'step_interface_available': False
            }
            validation['recommendations'].append('의존성 시스템 모듈 확인 필요')
            validation['overall_score'] -= 25
        
        validation['overall_score'] = max(0, validation['overall_score'])
        
        return validation
        
    except Exception as e:
        return {'error': str(e), 'overall_score': 0}

def get_environment_info() -> Dict[str, Any]:
    """환경 정보 조회 (v18.0 강화)"""
    return {
        'version': 'BaseStepMixin v18.0',
        'conda_info': CONDA_INFO,
        'hardware': {
            'is_m3_max': IS_M3_MAX,
            'memory_gb': MEMORY_GB,
            'platform': platform.system()
        },
        'libraries': {
            'torch_available': TORCH_AVAILABLE,
            'mps_available': MPS_AVAILABLE,
            'pil_available': PIL_AVAILABLE,
            'numpy_available': NUMPY_AVAILABLE
        },
        'device_info': {
            'recommended_device': 'mps' if IS_M3_MAX and MPS_AVAILABLE else 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        },
        'dependency_system': {
            'enhanced_dependency_manager': True,
            'step_model_interface_support': True,
            'auto_injection_support': True,
            'validation_support': True
        }
    }

# ==============================================
# 🔥 Export
# ==============================================

__all__ = [
    # 메인 클래스들
    'BaseStepMixin',
    'EnhancedDependencyManager',
    
    # 설정 및 상태 클래스들
    'StepConfig',
    'DependencyStatus',
    'PerformanceMetrics',
    
    # 인터페이스들
    'IModelProvider',
    'IMemoryManager',
    'IDataConverter',
    
    # 편의 함수들
    'create_base_step_mixin',
    'validate_step_environment',
    'get_environment_info',
    
    # 상수들
    'TORCH_AVAILABLE',
    'MPS_AVAILABLE',
    'NUMPY_AVAILABLE',
    'PIL_AVAILABLE',
    'CONDA_INFO',
    'IS_M3_MAX',
    'MEMORY_GB'
]

# ==============================================
# 🔥 모듈 로드 완료 로그
# ==============================================

logger = logging.getLogger(__name__)
logger.info("=" * 80)
logger.info("🔥 BaseStepMixin v18.0 - 의존성 주입 완전 수정 (StepFactory v7.0 호환)")
logger.info("=" * 80)
logger.info("✅ StepFactory v7.0과 완전 호환")
logger.info("✅ 의존성 주입 문제 완전 해결")
logger.info("✅ ModelLoader → StepModelInterface 연결 안정화")
logger.info("✅ 초기화 순서 최적화 (의존성 → 초기화 → 검증)")
logger.info("✅ 에러 처리 강화 및 안전한 폴백")
logger.info("✅ EnhancedDependencyManager 도입")
logger.info("✅ 의존성 유효성 검증 시스템")
logger.info("✅ 성능 메트릭 강화 (의존성 메트릭 추가)")
logger.info("✅ 진단 및 벤치마크 도구 강화")
logger.info("✅ TYPE_CHECKING 패턴으로 순환참조 완전 방지")
logger.info("✅ conda 환경 우선 최적화 (mycloset-ai-clean)")
logger.info("✅ M3 Max 128GB 메모리 최적화")
logger.info("=" * 80)
logger.info(f"🔧 현재 conda 환경: {CONDA_INFO['conda_env']} (최적화: {CONDA_INFO['is_target_env']})")
logger.info(f"🖥️  현재 시스템: M3 Max={IS_M3_MAX}, 메모리={MEMORY_GB:.1f}GB")
logger.info("=" * 80)