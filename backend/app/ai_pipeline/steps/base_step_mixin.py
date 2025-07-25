# backend/app/ai_pipeline/steps/base_step_mixin.py
"""
🔥 BaseStepMixin v16.0 - 순환참조 완전 해결 + 통합 의존성 주입
================================================================

✅ TYPE_CHECKING 패턴으로 순환참조 완전 방지
✅ 통합된 의존성 주입 인터페이스
✅ 모든 Step 클래스 호환성 보장
✅ 초기화 로직 표준화
✅ 프로덕션 레벨 안정성

Author: MyCloset AI Team
Date: 2025-07-24
Version: 16.0 (Circular Reference Complete Solution)
"""

import os
import gc
import time
import asyncio
import logging
import threading
import traceback
import weakref
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
# 🔥 의존성 주입 인터페이스 (추상화)
# ==============================================

class IModelProvider(ABC):
    """모델 제공자 인터페이스"""
    
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

class IMemoryManager(ABC):
    """메모리 관리자 인터페이스"""
    
    @abstractmethod
    def optimize_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """메모리 최적화"""
        pass
    
    @abstractmethod
    async def optimize_memory_async(self, aggressive: bool = False) -> Dict[str, Any]:
        """비동기 메모리 최적화"""
        pass

class IDataConverter(ABC):
    """데이터 변환기 인터페이스"""
    
    @abstractmethod
    def convert_data(self, data: Any, target_format: str) -> Any:
        """데이터 변환"""
        pass

# ==============================================
# 🔥 설정 클래스
# ==============================================

@dataclass
class StepConfig:
    """통합 Step 설정"""
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

@dataclass
class DependencyStatus:
    """의존성 상태 추적"""
    model_loader: bool = False
    step_interface: bool = False
    memory_manager: bool = False
    data_converter: bool = False
    di_container: bool = False
    base_initialized: bool = False
    custom_initialized: bool = False

# ==============================================
# 🔥 통합 의존성 관리자
# ==============================================

class UnifiedDependencyManager:
    """통합 의존성 관리자 (순환참조 방지)"""
    
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.logger = logging.getLogger(f"DependencyManager.{step_name}")
        
        # 의존성 저장
        self.dependencies: Dict[str, Any] = {}
        self.dependency_status = DependencyStatus()
        
        # 동기화
        self._lock = threading.RLock()
        
        # 자동 주입 플래그
        self._auto_injection_attempted = False
    
    def inject_model_loader(self, model_loader: 'ModelLoader') -> bool:
        """ModelLoader 의존성 주입"""
        try:
            with self._lock:
                self.dependencies['model_loader'] = model_loader
                self.dependency_status.model_loader = True
                
                # Step 인터페이스 생성
                if hasattr(model_loader, 'create_step_interface'):
                    interface = model_loader.create_step_interface(self.step_name)
                    self.dependencies['step_interface'] = interface
                    self.dependency_status.step_interface = True
                
                self.logger.info("✅ ModelLoader 의존성 주입 완료")
                return True
        except Exception as e:
            self.logger.error(f"❌ ModelLoader 주입 실패: {e}")
            return False
    
    def inject_memory_manager(self, memory_manager: 'MemoryManager') -> bool:
        """MemoryManager 의존성 주입"""
        try:
            with self._lock:
                self.dependencies['memory_manager'] = memory_manager
                self.dependency_status.memory_manager = True
                self.logger.info("✅ MemoryManager 의존성 주입 완료")
                return True
        except Exception as e:
            self.logger.error(f"❌ MemoryManager 주입 실패: {e}")
            return False
    
    def inject_data_converter(self, data_converter: 'DataConverter') -> bool:
        """DataConverter 의존성 주입"""
        try:
            with self._lock:
                self.dependencies['data_converter'] = data_converter
                self.dependency_status.data_converter = True
                self.logger.info("✅ DataConverter 의존성 주입 완료")
                return True
        except Exception as e:
            self.logger.error(f"❌ DataConverter 주입 실패: {e}")
            return False
    
    def inject_di_container(self, di_container: 'DIContainer') -> bool:
        """DI Container 의존성 주입"""
        try:
            with self._lock:
                self.dependencies['di_container'] = di_container
                self.dependency_status.di_container = True
                self.logger.info("✅ DI Container 의존성 주입 완료")
                return True
        except Exception as e:
            self.logger.error(f"❌ DI Container 주입 실패: {e}")
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
    
    def auto_inject_dependencies(self) -> bool:
        """자동 의존성 주입 (동적 import 사용)"""
        if self._auto_injection_attempted:
            return True
        
        self._auto_injection_attempted = True
        success_count = 0
        
        try:
            # ModelLoader 자동 주입
            if not self.dependency_status.model_loader:
                model_loader = self._get_global_model_loader()
                if model_loader:
                    self.inject_model_loader(model_loader)
                    success_count += 1
            
            # MemoryManager 자동 주입
            if not self.dependency_status.memory_manager:
                memory_manager = self._get_global_memory_manager()
                if memory_manager:
                    self.inject_memory_manager(memory_manager)
                    success_count += 1
            
            self.logger.info(f"🔄 자동 의존성 주입 완료: {success_count}개")
            return success_count > 0
            
        except Exception as e:
            self.logger.warning(f"⚠️ 자동 의존성 주입 실패: {e}")
            return False
    
    def _get_global_model_loader(self) -> Optional['ModelLoader']:
        """ModelLoader 동적 import (순환참조 방지)"""
        try:
            import importlib
            module = importlib.import_module('app.ai_pipeline.utils.model_loader')
            get_global = getattr(module, 'get_global_model_loader', None)
            if get_global:
                return get_global()
        except Exception as e:
            self.logger.debug(f"ModelLoader 자동 주입 실패: {e}")
        return None
    
    def _get_global_memory_manager(self) -> Optional['MemoryManager']:
        """MemoryManager 동적 import (순환참조 방지)"""
        try:
            import importlib
            module = importlib.import_module('app.ai_pipeline.utils.memory_manager')
            get_global = getattr(module, 'get_global_memory_manager', None)
            if get_global:
                return get_global()
        except Exception as e:
            self.logger.debug(f"MemoryManager 자동 주입 실패: {e}")
        return None

# ==============================================
# 🔥 BaseStepMixin v16.0 - 완전 통합 버전
# ==============================================

class BaseStepMixin:
    """
    🔥 BaseStepMixin v16.0 - 순환참조 완전 해결 + 통합 의존성 주입
    
    ✅ TYPE_CHECKING 패턴으로 순환참조 방지
    ✅ 통합된 의존성 주입 인터페이스
    ✅ 모든 Step 클래스 호환성 보장
    ✅ 초기화 로직 표준화
    """
    
    def __init__(self, **kwargs):
        """통합 초기화"""
        try:
            # 기본 설정
            self.config = self._create_config(**kwargs)
            self.step_name = kwargs.get('step_name', self.__class__.__name__)
            self.step_id = kwargs.get('step_id', 0)
            
            # Logger 설정
            self.logger = logging.getLogger(f"pipeline.steps.{self.step_name}")
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
            
            # 통합 의존성 관리자
            self.dependency_manager = UnifiedDependencyManager(self.step_name)
            
            # 상태 플래그들
            self.is_initialized = False
            self.is_ready = False
            self.has_model = False
            self.model_loaded = False
            self.warmup_completed = False
            
            # 시스템 정보
            self.device = self._resolve_device(self.config.device)
            self.is_m3_max = self._detect_m3_max()
            self.memory_gb = self._get_memory_info()
            
            # 성능 메트릭
            self.performance_metrics = {
                'process_count': 0,
                'total_process_time': 0.0,
                'average_process_time': 0.0,
                'error_count': 0,
                'success_count': 0,
                'cache_hits': 0
            }
            
            # 호환성을 위한 속성들
            self.model_loader = None
            self.model_interface = None
            self.memory_manager = None
            self.data_converter = None
            self.di_container = None
            
            # 자동 의존성 주입 (설정된 경우)
            if self.config.auto_inject_dependencies:
                self.dependency_manager.auto_inject_dependencies()
            
            self.logger.info(f"✅ {self.step_name} BaseStepMixin v16.0 초기화 완료")
            
        except Exception as e:
            self._emergency_setup(e)
    
    def _create_config(self, **kwargs) -> StepConfig:
        """설정 생성"""
        config = StepConfig()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def _emergency_setup(self, error: Exception):
        """긴급 설정"""
        self.step_name = getattr(self, 'step_name', self.__class__.__name__)
        self.logger = logging.getLogger("emergency")
        self.device = "cpu"
        self.is_initialized = False
        self.logger.error(f"🚨 {self.step_name} 긴급 초기화: {error}")
    
    # ==============================================
    # 🔥 통합 의존성 주입 인터페이스
    # ==============================================
    
    def set_model_loader(self, model_loader: 'ModelLoader'):
        """ModelLoader 의존성 주입 (표준 인터페이스)"""
        success = self.dependency_manager.inject_model_loader(model_loader)
        if success:
            self.model_loader = model_loader
            self.model_interface = self.dependency_manager.get_dependency('step_interface')
            self.has_model = True
            self.model_loaded = True
    
    def set_memory_manager(self, memory_manager: 'MemoryManager'):
        """MemoryManager 의존성 주입 (표준 인터페이스)"""
        success = self.dependency_manager.inject_memory_manager(memory_manager)
        if success:
            self.memory_manager = memory_manager
    
    def set_data_converter(self, data_converter: 'DataConverter'):
        """DataConverter 의존성 주입 (표준 인터페이스)"""
        success = self.dependency_manager.inject_data_converter(data_converter)
        if success:
            self.data_converter = data_converter
    
    def set_di_container(self, di_container: 'DIContainer'):
        """DI Container 의존성 주입 (표준 인터페이스)"""
        success = self.dependency_manager.inject_di_container(di_container)
        if success:
            self.di_container = di_container
    
    # ==============================================
    # 🔥 핵심 기능 메서드들
    # ==============================================
    
    def get_model(self, model_name: Optional[str] = None) -> Optional[Any]:
        """모델 가져오기 (통합 인터페이스)"""
        try:
            # Step Interface 우선 사용
            step_interface = self.dependency_manager.get_dependency('step_interface')
            if step_interface and hasattr(step_interface, 'get_model_sync'):
                return step_interface.get_model_sync(model_name or "default")
            
            # ModelLoader 직접 사용
            model_loader = self.dependency_manager.get_dependency('model_loader')
            if model_loader and hasattr(model_loader, 'load_model'):
                return model_loader.load_model(model_name or "default")
            
            self.logger.warning("⚠️ 모델 제공자가 주입되지 않음")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ 모델 가져오기 실패: {e}")
            return None
    
    async def get_model_async(self, model_name: Optional[str] = None) -> Optional[Any]:
        """비동기 모델 가져오기 (통합 인터페이스)"""
        try:
            # Step Interface 우선 사용
            step_interface = self.dependency_manager.get_dependency('step_interface')
            if step_interface and hasattr(step_interface, 'get_model_async'):
                return await step_interface.get_model_async(model_name or "default")
            
            # ModelLoader 직접 사용
            model_loader = self.dependency_manager.get_dependency('model_loader')
            if model_loader and hasattr(model_loader, 'load_model_async'):
                return await model_loader.load_model_async(model_name or "default")
            
            self.logger.warning("⚠️ 모델 제공자가 주입되지 않음")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ 비동기 모델 가져오기 실패: {e}")
            return None
    
    def optimize_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """메모리 최적화 (통합 인터페이스)"""
        try:
            # 주입된 MemoryManager 우선 사용
            memory_manager = self.dependency_manager.get_dependency('memory_manager')
            if memory_manager and hasattr(memory_manager, 'optimize_memory'):
                return memory_manager.optimize_memory(aggressive=aggressive)
            
            # 내장 메모리 최적화
            return self._builtin_memory_optimize(aggressive)
            
        except Exception as e:
            self.logger.error(f"❌ 메모리 최적화 실패: {e}")
            return {"success": False, "error": str(e)}
    
    async def optimize_memory_async(self, aggressive: bool = False) -> Dict[str, Any]:
        """비동기 메모리 최적화 (통합 인터페이스)"""
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
    # 🔥 표준화된 초기화 및 워밍업
    # ==============================================
    
    def initialize(self) -> bool:
        """표준화된 초기화"""
        try:
            if self.is_initialized:
                return True
            
            # 필수 의존성 확인
            if not self.dependency_manager.check_required_dependencies(self.config):
                if self.config.strict_mode:
                    raise RuntimeError("필수 의존성이 주입되지 않음")
                else:
                    self.logger.warning("⚠️ 일부 의존성이 누락됨")
            
            # 초기화 상태 설정
            self.dependency_manager.dependency_status.base_initialized = True
            self.is_initialized = True
            
            self.logger.info(f"✅ {self.step_name} 표준화된 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 초기화 실패: {e}")
            return False
    
    async def initialize_async(self) -> bool:
        """비동기 초기화"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.initialize)
        except Exception as e:
            self.logger.error(f"❌ 비동기 초기화 실패: {e}")
            return False
    
    def warmup(self) -> Dict[str, Any]:
        """표준화된 워밍업"""
        try:
            if self.warmup_completed:
                return {'success': True, 'message': '이미 워밍업 완료됨', 'cached': True}
            
            self.logger.info(f"🔥 {self.step_name} 표준화된 워밍업 시작...")
            start_time = time.time()
            results = []
            
            # 1. 메모리 워밍업
            try:
                memory_result = self.optimize_memory()
                results.append('memory_success' if memory_result.get('success') else 'memory_failed')
            except:
                results.append('memory_failed')
            
            # 2. 모델 워밍업
            try:
                test_model = self.get_model("warmup_test")
                results.append('model_success' if test_model else 'model_skipped')
            except:
                results.append('model_failed')
            
            # 3. 디바이스 워밍업
            try:
                if TORCH_AVAILABLE:
                    test_tensor = torch.randn(10, 10)
                    if self.device != 'cpu':
                        test_tensor = test_tensor.to(self.device)
                    _ = torch.matmul(test_tensor, test_tensor.t())
                    results.append('device_success')
                else:
                    results.append('device_skipped')
            except:
                results.append('device_failed')
            
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
                "total_count": len(results)
            }
            
        except Exception as e:
            self.logger.error(f"❌ 워밍업 실패: {e}")
            return {"success": False, "error": str(e)}
    
    async def warmup_async(self) -> Dict[str, Any]:
        """비동기 워밍업"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.warmup)
        except Exception as e:
            self.logger.error(f"❌ 비동기 워밍업 실패: {e}")
            return {"success": False, "error": str(e)}
    
    # ==============================================
    # 🔥 상태 및 정리 메서드들
    # ==============================================
    
    def get_status(self) -> Dict[str, Any]:
        """통합 상태 조회"""
        try:
            return {
                'step_name': self.step_name,
                'step_id': self.step_id,
                'is_initialized': self.is_initialized,
                'is_ready': self.is_ready,
                'has_model': self.has_model,
                'model_loaded': self.model_loaded,
                'warmup_completed': self.warmup_completed,
                'device': self.device,
                'is_m3_max': self.is_m3_max,
                'memory_gb': self.memory_gb,
                'dependencies': {
                    'model_loader': self.dependency_manager.dependency_status.model_loader,
                    'step_interface': self.dependency_manager.dependency_status.step_interface,
                    'memory_manager': self.dependency_manager.dependency_status.memory_manager,
                    'data_converter': self.dependency_manager.dependency_status.data_converter,
                    'di_container': self.dependency_manager.dependency_status.di_container,
                },
                'performance_metrics': self.performance_metrics,
                'config': self.config.__dict__,
                'version': '16.0-unified',
                'timestamp': time.time()
            }
        except Exception as e:
            self.logger.error(f"❌ 상태 조회 실패: {e}")
            return {'error': str(e), 'version': '16.0-unified'}
    
    async def cleanup(self) -> Dict[str, Any]:
        """표준화된 정리"""
        try:
            self.logger.info(f"🧹 {self.step_name} 표준화된 정리 시작...")
            
            # 메모리 정리
            cleanup_result = await self.optimize_memory_async(aggressive=True)
            
            # 상태 리셋
            self.is_ready = False
            self.warmup_completed = False
            self.has_model = False
            self.model_loaded = False
            
            # 의존성 해제 (참조만 제거)
            self.model_loader = None
            self.model_interface = None
            self.memory_manager = None
            self.data_converter = None
            self.di_container = None
            
            self.logger.info(f"✅ {self.step_name} 표준화된 정리 완료")
            
            return {
                "success": True,
                "cleanup_result": cleanup_result,
                "step_name": self.step_name,
                "version": "16.0-unified"
            }
        except Exception as e:
            self.logger.error(f"❌ 정리 실패: {e}")
            return {"success": False, "error": str(e)}
    
    # ==============================================
    # 🔥 내부 유틸리티 메서드들
    # ==============================================
    
    def _resolve_device(self, device: str) -> str:
        """디바이스 해결"""
        if device == "auto":
            if TORCH_AVAILABLE:
                if MPS_AVAILABLE:
                    return "mps"
                elif torch.cuda.is_available():
                    return "cuda"
            return "cpu"
        return device
    
    def _detect_m3_max(self) -> bool:
        """M3 Max 감지"""
        try:
            import platform
            import subprocess
            if platform.system() == 'Darwin':
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True, text=True, timeout=5
                )
                return 'M3' in result.stdout
        except:
            pass
        return False
    
    def _get_memory_info(self) -> float:
        """메모리 정보 조회"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.total / 1024**3
        except:
            return 16.0
    
    def _builtin_memory_optimize(self, aggressive: bool = False) -> Dict[str, Any]:
        """내장 메모리 최적화"""
        try:
            results = []
            
            # Python GC
            before = len(gc.get_objects()) if hasattr(gc, 'get_objects') else 0
            gc.collect()
            after = len(gc.get_objects()) if hasattr(gc, 'get_objects') else 0
            results.append(f"Python GC: {before - after}개 객체 해제")
            
            # PyTorch 메모리 정리
            if TORCH_AVAILABLE:
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
                for _ in range(3):
                    gc.collect()
                results.append("M3 Max 통합 메모리 최적화")
            
            return {
                "success": True,
                "results": results,
                "device": self.device,
                "is_m3_max": self.is_m3_max,
                "source": "builtin"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "source": "builtin"}

# ==============================================
# 🔥 Step별 특화 Mixin들
# ==============================================

class HumanParsingMixin(BaseStepMixin):
    """Step 1: Human Parsing 특화 Mixin"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('step_name', 'HumanParsingStep')
        kwargs.setdefault('step_id', 1)
        super().__init__(**kwargs)
        
        self.num_classes = 20
        self.parsing_categories = [
            'background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes',
            'dress', 'coat', 'socks', 'pants', 'jumpsuits', 'scarf', 'skirt',
            'face', 'left_arm', 'right_arm', 'left_leg', 'right_leg', 'left_shoe', 'right_shoe'
        ]

class PoseEstimationMixin(BaseStepMixin):
    """Step 2: Pose Estimation 특화 Mixin"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('step_name', 'PoseEstimationStep')
        kwargs.setdefault('step_id', 2)
        super().__init__(**kwargs)
        
        self.num_keypoints = 18
        self.keypoint_names = [
            'nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
            'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip', 'right_knee',
            'right_ankle', 'left_hip', 'left_knee', 'left_ankle', 'right_eye',
            'left_eye', 'right_ear', 'left_ear'
        ]

class ClothSegmentationMixin(BaseStepMixin):
    """Step 3: Cloth Segmentation 특화 Mixin"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('step_name', 'ClothSegmentationStep')
        kwargs.setdefault('step_id', 3)
        super().__init__(**kwargs)
        
        self.segmentation_methods = ['traditional', 'u2net', 'deeplab', 'auto', 'hybrid']

class GeometricMatchingMixin(BaseStepMixin):
    """Step 4: Geometric Matching 특화 Mixin"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('step_name', 'GeometricMatchingStep')
        kwargs.setdefault('step_id', 4)
        super().__init__(**kwargs)
        
        self.matching_methods = ['thin_plate_spline', 'affine', 'perspective', 'flow_based']

class ClothWarpingMixin(BaseStepMixin):
    """Step 5: Cloth Warping 특화 Mixin"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('step_name', 'ClothWarpingStep')
        kwargs.setdefault('step_id', 5)
        super().__init__(**kwargs)

class VirtualFittingMixin(BaseStepMixin):
    """Step 6: Virtual Fitting 특화 Mixin"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('step_name', 'VirtualFittingStep')
        kwargs.setdefault('step_id', 6)
        super().__init__(**kwargs)
        
        self.fitting_modes = ['standard', 'high_quality', 'fast', 'experimental']

class PostProcessingMixin(BaseStepMixin):
    """Step 7: Post Processing 특화 Mixin"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('step_name', 'PostProcessingStep')
        kwargs.setdefault('step_id', 7)
        super().__init__(**kwargs)
        
        self.processing_methods = ['super_resolution', 'denoising', 'color_correction', 'sharpening']

class QualityAssessmentMixin(BaseStepMixin):
    """Step 8: Quality Assessment 특화 Mixin"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('step_name', 'QualityAssessmentStep')
        kwargs.setdefault('step_id', 8)
        super().__init__(**kwargs)
        
        self.assessment_criteria = ['perceptual_quality', 'technical_quality', 'aesthetic_quality']

# ==============================================
# 🔥 편의 함수들
# ==============================================

def create_step_mixin(step_name: str, step_id: int, **kwargs) -> BaseStepMixin:
    """통합 BaseStepMixin 인스턴스 생성"""
    kwargs.update({'step_name': step_name, 'step_id': step_id})
    return BaseStepMixin(**kwargs)

def create_human_parsing_step(**kwargs) -> HumanParsingMixin:
    """Human Parsing Step 생성"""
    return HumanParsingMixin(**kwargs)

def create_pose_estimation_step(**kwargs) -> PoseEstimationMixin:
    """Pose Estimation Step 생성"""
    return PoseEstimationMixin(**kwargs)

def create_cloth_segmentation_step(**kwargs) -> ClothSegmentationMixin:
    """Cloth Segmentation Step 생성"""
    return ClothSegmentationMixin(**kwargs)

def create_geometric_matching_step(**kwargs) -> GeometricMatchingMixin:
    """Geometric Matching Step 생성"""
    return GeometricMatchingMixin(**kwargs)

def create_cloth_warping_step(**kwargs) -> ClothWarpingMixin:
    """Cloth Warping Step 생성"""
    return ClothWarpingMixin(**kwargs)

def create_virtual_fitting_step(**kwargs) -> VirtualFittingMixin:
    """Virtual Fitting Step 생성"""
    return VirtualFittingMixin(**kwargs)

def create_post_processing_step(**kwargs) -> PostProcessingMixin:
    """Post Processing Step 생성"""
    return PostProcessingMixin(**kwargs)

def create_quality_assessment_step(**kwargs) -> QualityAssessmentMixin:
    """Quality Assessment Step 생성"""
    return QualityAssessmentMixin(**kwargs)

# ==============================================
# 🔥 Export
# ==============================================

__all__ = [
    # 메인 클래스들
    'BaseStepMixin',
    'UnifiedDependencyManager',
    'StepConfig',
    'DependencyStatus',
    
    # 인터페이스들
    'IModelProvider',
    'IMemoryManager',
    'IDataConverter',
    
    # Step별 특화 Mixin들
    'HumanParsingMixin',
    'PoseEstimationMixin',
    'ClothSegmentationMixin',
    'GeometricMatchingMixin',
    'ClothWarpingMixin',
    'VirtualFittingMixin',
    'PostProcessingMixin',
    'QualityAssessmentMixin',
    
    # 편의 함수들
    'create_step_mixin',
    'create_human_parsing_step',
    'create_pose_estimation_step',
    'create_cloth_segmentation_step',
    'create_geometric_matching_step',
    'create_cloth_warping_step',
    'create_virtual_fitting_step',
    'create_post_processing_step',
    'create_quality_assessment_step',
    
    # 상수들
    'TORCH_AVAILABLE',
    'MPS_AVAILABLE',
    'NUMPY_AVAILABLE',
    'PIL_AVAILABLE'
]

# 모듈 로드 완료
logger = logging.getLogger(__name__)
logger.info("=" * 80)
logger.info("🔥 BaseStepMixin v16.0 - 순환참조 완전 해결 + 통합 의존성 주입")
logger.info("=" * 80)
logger.info("✅ TYPE_CHECKING 패턴으로 순환참조 완전 방지")
logger.info("✅ 통합된 의존성 주입 인터페이스")
logger.info("✅ 모든 Step 클래스 호환성 보장")
logger.info("✅ 초기화 로직 표준화")
logger.info("✅ UnifiedDependencyManager 도입")
logger.info("✅ 자동 의존성 주입 시스템")
logger.info("✅ 프로덕션 레벨 안정성")
logger.info("=" * 80)