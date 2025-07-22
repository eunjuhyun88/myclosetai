# backend/app/ai_pipeline/steps/base_step_mixin.py
"""
🔥 BaseStepMixin v15.0 - 완전 격리 + 순수 의존성 주입 버전 (완전체)
====================================================================

✅ 동적 import 완전 제거 - 순환참조 100% 차단
✅ 순수 의존성 주입만 사용
✅ 모든 외부 모듈 참조 제거
✅ 완전 격리된 아키텍처
✅ M3 Max 128GB 최적화
✅ conda 환경 우선
✅ 3단계 간단한 초기화
✅ 프로덕션 레벨 안정성
✅ 모든 Step에서 참조하는 기능 완전 포함
✅ 빠진 함수들 모두 복원

🔥 핵심 원칙:
- BaseStepMixin은 어떤 모듈도 직접 import 하지 않음
- 모든 기능은 의존성 주입으로만 제공
- 외부 모듈과 완전히 독립적
- 순환참조 불가능한 구조
- Step 파일들이 요구하는 모든 메서드 제공

Author: MyCloset AI Team
Date: 2025-07-22
Version: 15.0 (Pure DI Isolated - Complete)
"""

# ==============================================
# 🔥 1. 최소한의 기본 import만 (외부 모듈 없음)
# ==============================================
import os
import gc
import time
import asyncio
import logging
import threading
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Type, Callable, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
import platform
import subprocess
from datetime import datetime

# 시스템 모니터링 (선택적)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ==============================================
# 🔥 2. conda 환경 및 시스템 체크
# ==============================================
CONDA_INFO = {
    'conda_env': os.environ.get('CONDA_DEFAULT_ENV', 'none'),
    'conda_prefix': os.environ.get('CONDA_PREFIX', 'none'),
    'python_path': os.path.dirname(os.__file__)
}

if CONDA_INFO['conda_env'] != 'none':
    print(f"✅ conda 환경 감지: {CONDA_INFO['conda_env']}")
else:
    print("⚠️ conda 환경 권장: conda activate mycloset-ai")

# ==============================================
# 🔥 3. 안전한 라이브러리 Import (외부 의존성 없음)
# ==============================================

# PyTorch 안전 Import (순수 라이브러리만)
TORCH_AVAILABLE = False
MPS_AVAILABLE = False
try:
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    TORCH_AVAILABLE = True
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        MPS_AVAILABLE = True
        print("🍎 M3 Max MPS 사용 가능")
    
except ImportError:
    print("⚠️ PyTorch 없음 - conda install pytorch torchvision torchaudio -c pytorch")

# NumPy 안전 Import
NUMPY_AVAILABLE = False
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("⚠️ NumPy 없음 - conda install numpy")

# PIL 안전 Import  
PIL_AVAILABLE = False
try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    print("⚠️ PIL 없음 - conda install pillow")

# ==============================================
# 🔥 4. 간단한 설정 클래스
# ==============================================
@dataclass
class StepConfig:
    """Step 설정"""
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
    device_type: str = "auto"
    strict_mode: bool = True

# ==============================================
# 🔥 5. 의존성 주입용 인터페이스 (추상화)
# ==============================================
class IModelProvider(ABC):
    """모델 제공자 인터페이스"""
    
    @abstractmethod
    def get_model(self, model_name: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    async def get_model_async(self, model_name: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    def is_model_available(self, model_name: str) -> bool:
        pass

class IMemoryManager(ABC):
    """메모리 관리자 인터페이스"""
    
    @abstractmethod
    def optimize_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def optimize_memory_async(self, aggressive: bool = False) -> Dict[str, Any]:
        pass

class IDataConverter(ABC):
    """데이터 변환기 인터페이스"""
    
    @abstractmethod
    def convert_data(self, data: Any, target_format: str) -> Any:
        pass
    
    @abstractmethod
    async def convert_data_async(self, data: Any, target_format: str) -> Any:
        pass

# ==============================================
# 🔥 6. 내장 메모리 최적화 (외부 의존성 없음)
# ==============================================
class InternalMemoryOptimizer:
    """내장 메모리 최적화 (완전 독립적)"""
    
    def __init__(self, device: str = "mps"):
        self.device = device
        self.is_m3_max = self._detect_m3_max()
    
    def _detect_m3_max(self) -> bool:
        """M3 Max 감지"""
        try:
            if platform.system() == 'Darwin':
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True, text=True, timeout=5
                )
                return 'M3' in result.stdout
        except:
            pass
        return False
    
    def optimize(self, aggressive: bool = False) -> Dict[str, Any]:
        """메모리 최적화 실행"""
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
                    except Exception:
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
                "is_m3_max": self.is_m3_max
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def optimize_async(self, aggressive: bool = False) -> Dict[str, Any]:
        """비동기 메모리 최적화"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.optimize(aggressive))
        except Exception as e:
            return {"success": False, "error": str(e)}

# SimpleMemoryOptimizer 별칭 (기존 호환성)
SimpleMemoryOptimizer = InternalMemoryOptimizer

# ==============================================
# 🔥 7. BaseStepMixin v15.0 - 완전 격리 버전 (완전체)
# ==============================================
class BaseStepMixin:
    """
    🔥 BaseStepMixin v15.0 - 완전 격리 + 순수 의존성 주입 (완전체)
    
    ✅ 동적 import 완전 제거
    ✅ 순수 의존성 주입만 사용  
    ✅ 모든 외부 모듈 참조 제거
    ✅ 완전 격리된 아키텍처
    ✅ 순환참조 불가능
    ✅ Step 파일들이 요구하는 모든 메서드 제공
    ✅ 빠진 함수들 모두 복원
    """
    
    def __init__(self, **kwargs):
        """3단계 격리된 초기화"""
        try:
            # STEP 1: 기본 설정 (격리)
            self._setup_basic_isolated(**kwargs)
            
            # STEP 2: 시스템 설정 (격리)
            self._setup_system_isolated()
            
            # STEP 3: 완료 (격리)
            self._finalize_isolated()
            
            self.logger.info(f"✅ {self.step_name} BaseStepMixin v15.0 완전 격리 초기화 완료")
            
        except Exception as e:
            self._emergency_setup_isolated(e)
    
    def _setup_basic_isolated(self, **kwargs):
        """STEP 1: 기본 설정 (완전 격리)"""
        # 설정
        self.config = StepConfig()
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # 기본 속성
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
        
        # 상태 플래그들
        self.is_initialized = False
        self.is_ready = False
        self.has_model = False
        self.model_loaded = False
        self.warmup_completed = False
        
        # 🔥 의존성 주입을 위한 인터페이스 속성들 (완전 격리)
        self.model_provider: Optional[IModelProvider] = None
        self.memory_manager: Optional[IMemoryManager] = None 
        self.data_converter: Optional[IDataConverter] = None
        
        # 🔥 폐기 예정이지만 호환성을 위한 속성들
        self.model_loader = None  # ModelLoader 호환성
        
        # 내부 캐시 (완전 독립적)
        self.model_cache = {}
        self.loaded_models = {}
        
        # 성능 메트릭
        self.performance_metrics = {
            'process_count': 0,
            'total_process_time': 0.0,
            'average_process_time': 0.0,
            'error_history': [],
            'success_count': 0,
            'cache_hits': 0
        }
        
        # 에러 추적
        self.error_count = 0
        self.last_error = None
        self.total_processing_count = 0
        self.last_processing_time = None
    
    def _setup_system_isolated(self):
        """STEP 2: 시스템 설정 (완전 격리)"""
        # 디바이스 감지
        if self.config.device == "auto":
            self.device = self._detect_optimal_device()
        else:
            self.device = self.config.device
        
        # M3 Max 감지
        self.is_m3_max = self._detect_m3_max()
        
        # 메모리 정보
        self.memory_gb = self._get_memory_info()
        
        # 🔥 완전 독립적인 메모리 최적화 시스템
        self.internal_memory_optimizer = InternalMemoryOptimizer(self.device)
        
        # 현재 모델 (격리된 상태)
        self._current_model = None
        self._current_model_name = None
    
    def _finalize_isolated(self):
        """STEP 3: 완료 (완전 격리)"""
        self.is_initialized = True
        
        # 자동 워밍업 (설정된 경우)
        if self.config.auto_warmup:
            try:
                warmup_result = self.warmup_isolated()
                if warmup_result.get('success', False):
                    self.warmup_completed = True
                    self.is_ready = True
            except Exception as e:
                self.logger.warning(f"⚠️ 자동 워밍업 실패: {e}")
    
    def _emergency_setup_isolated(self, error: Exception):
        """긴급 설정 (완전 격리)"""
        self.step_name = getattr(self, 'step_name', self.__class__.__name__)
        self.logger = logging.getLogger("emergency")
        self.device = "cpu"
        self.is_initialized = False
        self.error_count = 1
        self.last_error = str(error)
        print(f"🚨 {self.step_name} 긴급 격리 초기화: {error}")
    
    # ==============================================
    # 🔥 8. 의존성 주입 메서드들 (순수 인터페이스)
    # ==============================================
    
    def set_model_provider(self, model_provider: IModelProvider):
        """모델 제공자 의존성 주입"""
        self.model_provider = model_provider
        self.logger.info("✅ ModelProvider 주입 완료")
        if model_provider:
            self.has_model = True
            self.model_loaded = True
    
    def set_memory_manager(self, memory_manager: IMemoryManager):
        """메모리 관리자 의존성 주입"""
        self.memory_manager = memory_manager
        self.logger.info("✅ MemoryManager 주입 완료")
    
    def set_data_converter(self, data_converter: IDataConverter):
        """데이터 변환기 의존성 주입"""
        self.data_converter = data_converter
        self.logger.info("✅ DataConverter 주입 완료")
    
    def set_model_loader(self, model_loader):
        """ModelLoader 의존성 주입 (폐기 예정, 호환성만 유지)"""
        self.model_loader = model_loader
        # 어댑터 패턴으로 변환 시도
        if hasattr(model_loader, 'get_model'):
            self.model_provider = ModelLoaderAdapter(model_loader)
        if model_loader:
            self.has_model = True
            self.model_loaded = True
        self.logger.info("✅ ModelLoader 주입 완료 (폐기 예정 - ModelProvider 사용 권장)")
    
    # ==============================================
    # 🔥 9. 시스템 감지 메서드들 (완전 독립)
    # ==============================================
    
    def _detect_optimal_device(self) -> str:
        """최적 디바이스 감지"""
        try:
            if TORCH_AVAILABLE:
                if MPS_AVAILABLE:
                    return "mps"
                elif hasattr(torch, 'cuda') and torch.cuda.is_available():
                    return "cuda"
            return "cpu"
        except:
            return "cpu"
    
    def _detect_m3_max(self) -> bool:
        """M3 Max 감지"""
        try:
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
            if PSUTIL_AVAILABLE:
                import psutil
                memory = psutil.virtual_memory()
                return memory.total / 1024**3
        except:
            pass
        return 16.0
    
    # ==============================================
    # 🔥 10. 핵심 기능 메서드들 (순수 DI 기반)
    # ==============================================
    
    def get_model(self, model_name: Optional[str] = None) -> Optional[Any]:
        """모델 가져오기 (순수 DI 기반)"""
        try:
            # 캐시 확인
            cache_key = model_name or "default"
            if cache_key in self.model_cache:
                self.performance_metrics['cache_hits'] += 1
                return self.model_cache[cache_key]
            
            # 🔥 순수 DI: ModelProvider 우선 사용
            if self.model_provider:
                try:
                    model = self.model_provider.get_model(model_name or "default")
                    if model:
                        self.model_cache[cache_key] = model
                        self.has_model = True
                        self.model_loaded = True
                        self._current_model = model
                        self._current_model_name = model_name
                        return model
                except Exception as e:
                    self.logger.debug(f"ModelProvider를 통한 모델 로드 실패: {e}")
            
            # 🔥 폴백: 기존 ModelLoader 호환성 (폐기 예정)
            if self.model_loader and hasattr(self.model_loader, 'get_model'):
                try:
                    model = self.model_loader.get_model(model_name or "default")
                    if model:
                        self.model_cache[cache_key] = model
                        self.has_model = True
                        self.model_loaded = True
                        self._current_model = model
                        self._current_model_name = model_name
                        return model
                except Exception as e:
                    self.logger.debug(f"ModelLoader 폴백 실패: {e}")
            
            self.logger.warning("⚠️ 모델 제공자가 주입되지 않음 - 의존성 주입 필요")
            return None
                
        except Exception as e:
            self.logger.error(f"❌ 모델 가져오기 실패: {e}")
            return None
    
    async def get_model_async(self, model_name: Optional[str] = None) -> Optional[Any]:
        """모델 가져오기 (비동기, 순수 DI 기반)"""
        try:
            # 캐시 확인
            cache_key = model_name or "default"
            if cache_key in self.model_cache:
                self.performance_metrics['cache_hits'] += 1
                return self.model_cache[cache_key]
            
            # 순수 DI: ModelProvider 비동기 사용
            if self.model_provider:
                try:
                    model = await self.model_provider.get_model_async(model_name or "default")
                    if model:
                        self.model_cache[cache_key] = model
                        self.has_model = True
                        self.model_loaded = True
                        return model
                except Exception as e:
                    self.logger.debug(f"비동기 ModelProvider 실패: {e}")
            
            # 폴백: 기존 ModelLoader 비동기 호환성
            if self.model_loader:
                try:
                    if hasattr(self.model_loader, 'get_model_async'):
                        model = await self.model_loader.get_model_async(model_name or "default")
                    elif hasattr(self.model_loader, 'get_model'):
                        loop = asyncio.get_event_loop()
                        model = await loop.run_in_executor(None, 
                            lambda: self.model_loader.get_model(model_name or "default"))
                    else:
                        model = None
                    
                    if model:
                        self.model_cache[cache_key] = model
                        self.has_model = True
                        self.model_loaded = True
                        return model
                except Exception as e:
                    self.logger.debug(f"비동기 ModelLoader 폴백 실패: {e}")
            
            self.logger.warning("⚠️ 모델 제공자가 주입되지 않음 - 의존성 주입 필요")
            return None
                
        except Exception as e:
            self.logger.error(f"❌ 비동기 모델 가져오기 실패: {e}")
            return None
    
    def optimize_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """메모리 최적화 (순수 DI 기반)"""
        try:
            # 순수 DI: MemoryManager 우선 사용
            if self.memory_manager:
                try:
                    result = self.memory_manager.optimize_memory(aggressive=aggressive)
                    result["source"] = "injected_memory_manager"
                    return result
                except Exception as e:
                    self.logger.debug(f"주입된 MemoryManager 실패: {e}")
            
            # 폴백: 내장 독립적 메모리 최적화 사용
            result = self.internal_memory_optimizer.optimize(aggressive=aggressive)
            result["source"] = "internal_optimizer"
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 메모리 최적화 실패: {e}")
            return {"success": False, "error": str(e)}
    
    async def optimize_memory_async(self, aggressive: bool = False) -> Dict[str, Any]:
        """메모리 최적화 (비동기, 순수 DI 기반)"""
        try:
            # 순수 DI: MemoryManager 비동기 사용
            if self.memory_manager:
                try:
                    result = await self.memory_manager.optimize_memory_async(aggressive=aggressive)
                    result["source"] = "injected_memory_manager"
                    return result
                except Exception as e:
                    self.logger.debug(f"비동기 MemoryManager 실패: {e}")
            
            # 폴백: 내장 비동기 메모리 최적화
            result = await self.internal_memory_optimizer.optimize_async(aggressive=aggressive)
            result["source"] = "internal_optimizer"
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 비동기 메모리 최적화 실패: {e}")
            return {"success": False, "error": str(e)}
    
    # ==============================================
    # 🔥 11. 워밍업 메서드들 (완전 복원)
    # ==============================================
    
    def warmup_isolated(self) -> Dict[str, Any]:
        """격리된 워밍업 실행"""
        try:
            if self.warmup_completed:
                return {'success': True, 'message': '이미 워밍업 완료됨', 'cached': True}
            
            self.logger.info(f"🔥 {self.step_name} 격리된 워밍업 시작...")
            start_time = time.time()
            results = []
            
            # 1. 메모리 워밍업
            try:
                memory_result = self.optimize_memory()
                results.append('memory_success' if memory_result.get('success') else 'memory_failed')
            except:
                results.append('memory_failed')
            
            # 2. 모델 워밍업 (의존성 주입 기반)
            try:
                if self.model_provider or self.model_loader:
                    test_model = self.get_model("warmup_test")
                    results.append('model_success' if test_model else 'model_skipped')
                else:
                    results.append('model_skipped_no_provider')
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
            
            self.logger.info(f"🔥 격리된 워밍업 완료: {success_count}/{len(results)} 성공 ({duration:.2f}초)")
            
            return {
                "success": overall_success,
                "duration": duration,
                "results": results,
                "success_count": success_count,
                "total_count": len(results),
                "isolated": True
            }
            
        except Exception as e:
            self.logger.error(f"❌ 격리된 워밍업 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def warmup(self) -> Dict[str, Any]:
        """워밍업 (동기 버전)"""
        return self.warmup_isolated()
    
    async def warmup_async(self) -> Dict[str, Any]:
        """비동기 워밍업"""
        try:
            if self.warmup_completed:
                return {'success': True, 'message': '이미 워밍업 완료됨', 'cached': True}
            
            self.logger.info(f"🔥 {self.step_name} 비동기 워밍업 시작...")
            start_time = time.time()
            results = []
            
            # 1. 비동기 메모리 워밍업
            try:
                memory_result = await self.optimize_memory_async()
                results.append('memory_async_success' if memory_result.get('success') else 'memory_async_failed')
            except:
                results.append('memory_async_failed')
            
            # 2. 비동기 모델 워밍업
            try:
                if self.model_provider or self.model_loader:
                    test_model = await self.get_model_async("warmup_test")
                    results.append('model_async_success' if test_model else 'model_async_skipped')
                else:
                    results.append('model_async_skipped')
            except:
                results.append('model_async_failed')
            
            # 3. 비동기 디바이스 워밍업
            try:
                if TORCH_AVAILABLE:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self._device_warmup_sync)
                    results.append('device_async_success')
                else:
                    results.append('device_async_skipped')
            except:
                results.append('device_async_failed')
            
            duration = time.time() - start_time
            success_count = sum(1 for r in results if 'success' in r)
            overall_success = success_count > 0
            
            if overall_success:
                self.warmup_completed = True
                self.is_ready = True
            
            self.logger.info(f"🔥 비동기 워밍업 완료: {success_count}/{len(results)} 성공 ({duration:.2f}초)")
            
            return {
                "success": overall_success,
                "duration": duration,
                "results": results,
                "success_count": success_count,
                "total_count": len(results),
                "async": True
            }
            
        except Exception as e:
            self.logger.error(f"❌ 비동기 워밍업 실패: {e}")
            return {"success": False, "error": str(e), "async": True}
    
    def _device_warmup_sync(self):
        """동기 디바이스 워밍업"""
        try:
            if TORCH_AVAILABLE:
                test_tensor = torch.randn(10, 10)
                if self.device != 'cpu':
                    test_tensor = test_tensor.to(self.device)
                _ = torch.matmul(test_tensor, test_tensor.t())
                return True
        except:
            pass
        return False
    
    async def warmup_step(self) -> Dict[str, Any]:
        """Step 워밍업 (BaseStepMixin 호환용)"""
        return await self.warmup_async()
    
    # ==============================================
    # 🔥 12. 초기화 및 정리 메서드들 (완전 복원)
    # ==============================================
    
    def initialize(self) -> bool:
        """초기화 메서드 - Step들이 사용"""
        try:
            if self.is_initialized:
                return True
            
            self.is_initialized = True
            self.logger.info(f"✅ {self.step_name} 격리된 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 초기화 실패: {e}")
            return False
    
    async def initialize_async(self) -> bool:
        """비동기 초기화 메서드"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.initialize)
        except Exception as e:
            self.logger.error(f"❌ 비동기 초기화 실패: {e}")
            return False
    
    async def cleanup(self) -> Dict[str, Any]:
        """정리 (완전 격리)"""
        try:
            self.logger.info(f"🧹 {self.step_name} 격리된 정리 시작...")
            
            # 모델 캐시 정리
            self.model_cache.clear()
            self.loaded_models.clear()
            
            # 메모리 정리
            cleanup_result = await self.optimize_memory_async(aggressive=True)
            
            # 상태 리셋
            self.is_ready = False
            self.warmup_completed = False
            
            # 의존성 해제 (참조만 제거)
            self.model_provider = None
            self.memory_manager = None
            self.data_converter = None
            self.model_loader = None  # 호환성
            
            self.logger.info(f"✅ {self.step_name} 격리된 정리 완료")
            
            return {
                "success": True,
                "cleanup_result": cleanup_result,
                "step_name": self.step_name,
                "isolated": True
            }
        
        except Exception as e:
            self.logger.warning(f"⚠️ 정리 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def cleanup_models(self):
        """모델 정리 - Step들이 사용"""
        try:
            # 모델 캐시 정리
            self.model_cache.clear()
            self.loaded_models.clear()
            
            # 현재 모델 초기화
            self._current_model = None
            self._current_model_name = None
            
            # PyTorch 메모리 정리
            if TORCH_AVAILABLE:
                if self.device == "mps" and MPS_AVAILABLE:
                    try:
                        if hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                    except:
                        pass
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
                
                gc.collect()
            
            self.has_model = False
            self.model_loaded = False
            self.logger.info(f"🧹 {self.step_name} 격리된 모델 정리 완료")
                
        except Exception as e:
            self.logger.warning(f"⚠️ 모델 정리 중 오류: {e}")
    
    # ==============================================
    # 🔥 13. 상태 및 성능 메서드들 (완전 복원)
    # ==============================================
    
    def get_status(self) -> Dict[str, Any]:
        """Step 상태 조회 (완전 격리)"""
        try:
            return {
                'step_name': self.step_name,
                'step_id': getattr(self, 'step_id', 0),
                'is_initialized': self.is_initialized,
                'is_ready': self.is_ready,
                'has_model': self.has_model,
                'model_loaded': self.model_loaded,
                'warmup_completed': self.warmup_completed,
                'device': self.device,
                'is_m3_max': self.is_m3_max,
                'memory_gb': self.memory_gb,
                'error_count': self.error_count,
                'last_error': self.last_error,
                'total_processing_count': self.total_processing_count,
                # 의존성 정보 (격리된)
                'dependencies': {
                    'model_provider': self.model_provider is not None,
                    'memory_manager': self.memory_manager is not None,
                    'data_converter': self.data_converter is not None,
                    'model_loader_deprecated': self.model_loader is not None,
                },
                'performance_metrics': self.performance_metrics,
                'conda_info': CONDA_INFO,
                'timestamp': time.time(),
                'version': '15.0-pure-di-isolated',
                'isolated': True
            }
            
        except Exception as e:
            self.logger.error(f"❌ 상태 조회 실패: {e}")
            return {
                'step_name': getattr(self, 'step_name', 'unknown'),
                'error': str(e),
                'version': '15.0-pure-di-isolated',
                'timestamp': time.time(),
                'isolated': True
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 조회 (완전 격리)"""
        try:
            return {
                'total_processing_count': self.total_processing_count,
                'last_processing_time': self.last_processing_time,
                'error_count': self.error_count,
                'success_rate': self._calculate_success_rate(),
                'average_process_time': self.performance_metrics.get('average_process_time', 0.0),
                'total_process_time': self.performance_metrics.get('total_process_time', 0.0),
                'cache_hits': self.performance_metrics.get('cache_hits', 0),
                'cache_hit_rate': self._calculate_cache_hit_rate(),
                'version': '15.0-pure-di-isolated',
                'isolated': True
            }
            
        except Exception as e:
            self.logger.error(f"❌ 성능 요약 조회 실패: {e}")
            return {'version': '15.0-pure-di-isolated', 'error': str(e), 'isolated': True}
    
    def _calculate_success_rate(self) -> float:
        """성공률 계산 (완전 격리)"""
        try:
            total = self.total_processing_count
            errors = self.error_count
            if total > 0:
                return (total - errors) / total * 100.0
            return 0.0
        except:
            return 0.0
    
    def _calculate_cache_hit_rate(self) -> float:
        """캐시 히트율 계산"""
        try:
            hits = self.performance_metrics.get('cache_hits', 0)
            total = self.total_processing_count
            if total > 0:
                return (hits / total) * 100.0
            return 0.0
        except:
            return 0.0
    
    def record_processing(self, duration: float, success: bool = True):
        """처리 기록 - Step들이 사용"""
        try:
            self.total_processing_count += 1
            self.last_processing_time = time.time()
            
            if success:
                self.performance_metrics['success_count'] += 1
            else:
                self.error_count += 1
            
            # 성능 메트릭 업데이트
            self.performance_metrics['process_count'] = self.total_processing_count
            self.performance_metrics['total_process_time'] += duration
            
            if self.total_processing_count > 0:
                self.performance_metrics['average_process_time'] = (
                    self.performance_metrics['total_process_time'] / self.total_processing_count
                )
            
        except Exception as e:
            self.logger.warning(f"⚠️ 처리 기록 실패: {e}")
    
    def __del__(self):
        """소멸자 (안전한 정리)"""
        try:
            if hasattr(self, 'model_cache'):
                self.model_cache.clear()
        except:
            pass

# ==============================================
# 🔥 14. 호환성 어댑터 (기존 ModelLoader 지원)
# ==============================================
class ModelLoaderAdapter(IModelProvider):
    """기존 ModelLoader를 IModelProvider로 변환하는 어댑터"""
    
    def __init__(self, model_loader):
        self.model_loader = model_loader
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """모델 가져오기"""
        try:
            if hasattr(self.model_loader, 'get_model'):
                return self.model_loader.get_model(model_name)
            elif hasattr(self.model_loader, 'load_model'):
                return self.model_loader.load_model(model_name)
            return None
        except Exception:
            return None
    
    async def get_model_async(self, model_name: str) -> Optional[Any]:
        """비동기 모델 가져오기"""
        try:
            if hasattr(self.model_loader, 'get_model_async'):
                return await self.model_loader.get_model_async(model_name)
            elif hasattr(self.model_loader, 'load_model_async'):
                return await self.model_loader.load_model_async(model_name)
            else:
                # 동기 메서드를 비동기로 실행
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: self.get_model(model_name))
        except Exception:
            return None
    
    def is_model_available(self, model_name: str) -> bool:
        """모델 사용 가능 여부"""
        try:
            if hasattr(self.model_loader, 'is_model_available'):
                return self.model_loader.is_model_available(model_name)
            return self.get_model(model_name) is not None
        except Exception:
            return False

# ==============================================
# 🔥 15. Step별 특화 Mixin들 (완전 격리)
# ==============================================

class HumanParsingMixin(BaseStepMixin):
    """Step 1: Human Parsing 특화 Mixin (완전 격리)"""
    
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
    """Step 2: Pose Estimation 특화 Mixin (완전 격리)"""
    
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
    """Step 3: Cloth Segmentation 특화 Mixin (완전 격리)"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('step_name', 'ClothSegmentationStep')
        kwargs.setdefault('step_id', 3)
        super().__init__(**kwargs)
        
        self.segmentation_methods = ['traditional', 'u2net', 'deeplab', 'auto', 'hybrid']
        self.segmentation_method = kwargs.get('segmentation_method', 'u2net')

class GeometricMatchingMixin(BaseStepMixin):
    """Step 4: Geometric Matching 특화 Mixin (완전 격리)"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('step_name', 'GeometricMatchingStep')
        kwargs.setdefault('step_id', 4)
        super().__init__(**kwargs)
        
        self.matching_methods = ['thin_plate_spline', 'affine', 'perspective', 'flow_based']
        self.matching_method = kwargs.get('matching_method', 'thin_plate_spline')
        self.grid_size = kwargs.get('grid_size', (5, 5))

class ClothWarpingMixin(BaseStepMixin):
    """Step 5: Cloth Warping 특화 Mixin (완전 격리)"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('step_name', 'ClothWarpingStep')
        kwargs.setdefault('step_id', 5)
        super().__init__(**kwargs)
        
        self.warping_stages = ['preprocessing', 'geometric_transformation', 'texture_mapping', 'postprocessing']
        self.warping_quality = kwargs.get('warping_quality', 'high')
        self.preserve_texture = kwargs.get('preserve_texture', True)

class VirtualFittingMixin(BaseStepMixin):
    """Step 6: Virtual Fitting 특화 Mixin (완전 격리) - 핵심 단계"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('step_name', 'VirtualFittingStep')
        kwargs.setdefault('step_id', 6)
        super().__init__(**kwargs)
        
        self.fitting_modes = ['standard', 'high_quality', 'fast', 'experimental']
        self.fitting_mode = kwargs.get('fitting_mode', 'high_quality')
        self.diffusion_steps = kwargs.get('diffusion_steps', 50)
        self.guidance_scale = kwargs.get('guidance_scale', 7.5)
        self.use_ootd = kwargs.get('use_ootd', True)

class PostProcessingMixin(BaseStepMixin):
    """Step 7: Post Processing 특화 Mixin (완전 격리)"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('step_name', 'PostProcessingStep')
        kwargs.setdefault('step_id', 7)
        super().__init__(**kwargs)
        
        self.processing_methods = ['super_resolution', 'denoising', 'color_correction', 'sharpening']
        self.enhancement_level = kwargs.get('enhancement_level', 'medium')
        self.super_resolution_factor = kwargs.get('super_resolution_factor', 2.0)

class QualityAssessmentMixin(BaseStepMixin):
    """Step 8: Quality Assessment 특화 Mixin (완전 격리)"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('step_name', 'QualityAssessmentStep')
        kwargs.setdefault('step_id', 8)
        super().__init__(**kwargs)
        
        self.assessment_criteria = ['perceptual_quality', 'technical_quality', 'aesthetic_quality', 'overall_quality']
        self.quality_threshold = kwargs.get('quality_threshold', 0.7)
        self.use_clip_score = kwargs.get('use_clip_score', True)

# ==============================================
# 🔥 16. 편의 함수들 (완전 격리) + 기존 호환성 (🔥 문제 해결!)
# ==============================================

def create_isolated_step_mixin(step_name: str, step_id: int, **kwargs) -> BaseStepMixin:
    """격리된 BaseStepMixin 인스턴스 생성 - 🔥 빠진 함수 복원!"""
    kwargs.update({'step_name': step_name, 'step_id': step_id})
    return BaseStepMixin(**kwargs)

def create_step_mixin(step_name: str, step_id: int, **kwargs) -> BaseStepMixin:
    """BaseStepMixin 인스턴스 생성 (기존 호환성)"""
    return create_isolated_step_mixin(step_name, step_id, **kwargs)

def create_human_parsing_step(**kwargs) -> HumanParsingMixin:
    """Human Parsing Step 생성 (격리)"""
    return HumanParsingMixin(**kwargs)

def create_pose_estimation_step(**kwargs) -> PoseEstimationMixin:
    """Pose Estimation Step 생성 (격리)"""
    return PoseEstimationMixin(**kwargs)

def create_cloth_segmentation_step(**kwargs) -> ClothSegmentationMixin:
    """Cloth Segmentation Step 생성 (격리)"""
    return ClothSegmentationMixin(**kwargs)

def create_geometric_matching_step(**kwargs) -> GeometricMatchingMixin:
    """Geometric Matching Step 생성 (격리)"""
    return GeometricMatchingMixin(**kwargs)

def create_cloth_warping_step(**kwargs) -> ClothWarpingMixin:
    """Cloth Warping Step 생성 (격리)"""
    return ClothWarpingMixin(**kwargs)

def create_virtual_fitting_step(**kwargs) -> VirtualFittingMixin:
    """Virtual Fitting Step 생성 (격리) - 핵심"""
    return VirtualFittingMixin(**kwargs)

def create_post_processing_step(**kwargs) -> PostProcessingMixin:
    """Post Processing Step 생성 (격리)"""
    return PostProcessingMixin(**kwargs)

def create_quality_assessment_step(**kwargs) -> QualityAssessmentMixin:
    """Quality Assessment Step 생성 (격리)"""
    return QualityAssessmentMixin(**kwargs)

def create_m3_max_optimized_step(step_type: str, **kwargs) -> BaseStepMixin:
    """M3 Max 최적화된 Step 생성 (격리)"""
    kwargs.update({
        'device': 'mps',
        'auto_memory_cleanup': True,
        'use_fp16': True
    })
    
    step_creators = {
        'human_parsing': create_human_parsing_step,
        'pose_estimation': create_pose_estimation_step,
        'cloth_segmentation': create_cloth_segmentation_step,
        'geometric_matching': create_geometric_matching_step,
        'cloth_warping': create_cloth_warping_step,
        'virtual_fitting': create_virtual_fitting_step,
        'post_processing': create_post_processing_step,
        'quality_assessment': create_quality_assessment_step,
    }
    
    creator = step_creators.get(step_type, create_isolated_step_mixin)
    return creator(**kwargs)

# ==============================================
# 🔥 17. 모듈 내보내기 (완전 격리) + 빠진 기능 추가
# ==============================================

__all__ = [
    # 🔥 메인 클래스들 (완전 격리)
    'BaseStepMixin',
    'StepConfig',
    'InternalMemoryOptimizer',
    'SimpleMemoryOptimizer',  # 🔥 기존 호환성 별칭
    
    # 🔥 인터페이스들
    'IModelProvider',
    'IMemoryManager', 
    'IDataConverter',
    'ModelLoaderAdapter',
    
    # 🔥 Step별 특화 Mixin들 (완전 격리)
    'HumanParsingMixin',
    'PoseEstimationMixin', 
    'ClothSegmentationMixin',
    'GeometricMatchingMixin',
    'ClothWarpingMixin',
    'VirtualFittingMixin',
    'PostProcessingMixin',
    'QualityAssessmentMixin',
    
    # 편의 함수들 (완전 격리) + 🔥 기존 호환성 함수명들 추가
    'create_isolated_step_mixin',  # 🔥 빠진 함수 복원!
    'create_step_mixin',  # 🔥 기존 호환성
    'create_human_parsing_step',
    'create_pose_estimation_step',
    'create_cloth_segmentation_step',
    'create_geometric_matching_step',
    'create_cloth_warping_step',
    'create_virtual_fitting_step',
    'create_post_processing_step',
    'create_quality_assessment_step',
    'create_m3_max_optimized_step',
    
    # 상수들
    'TORCH_AVAILABLE',
    'MPS_AVAILABLE',
    'NUMPY_AVAILABLE',
    'PIL_AVAILABLE',
    'CONDA_INFO'
]

# ==============================================
# 🔥 18. 모듈 로드 완료 메시지
# ==============================================

print("=" * 80)
print("🎉 BaseStepMixin v15.0 - 완전 격리 + 순수 의존성 주입 (완전체)!")
print("=" * 80)
print("🔥 완전 격리 달성:")
print("   ✅ 동적 import 완전 제거 - 순환참조 100% 차단")
print("   ✅ 순수 의존성 주입만 사용")
print("   ✅ 모든 외부 모듈 참조 제거")
print("   ✅ 완전 독립적인 아키텍처")
print("   ✅ 순환참조 불가능한 구조")
print("   ✅ create_isolated_step_mixin 함수 복원!")
print("")
print("🔥 인터페이스 기반 설계:")
print("   🔌 IModelProvider - 모델 제공자 인터페이스")
print("   🧠 IMemoryManager - 메모리 관리자 인터페이스") 
print("   📊 IDataConverter - 데이터 변환기 인터페이스")
print("   🔧 ModelLoaderAdapter - 기존 호환성 어댑터")
print("")
print("🔥 순수 의존성 주입 메서드:")
print("   💉 set_model_provider() - 모델 제공자 주입")
print("   💉 set_memory_manager() - 메모리 관리자 주입")
print("   💉 set_data_converter() - 데이터 변환기 주입")
print("   ⚠️  set_model_loader() - 폐기 예정 (호환성만)")
print("")
print("🚀 Step들이 사용하는 핵심 메서드들 (완전 복원):")
print("   🤖 get_model(), get_model_async() - 순수 DI")
print("   🧹 optimize_memory(), optimize_memory_async() - 순수 DI")
print("   🔥 warmup(), warmup_async(), warmup_step() - 격리된 워밍업")
print("   📊 get_status(), get_performance_summary() - 격리된 상태")
print("   🔧 initialize(), initialize_async() - 초기화")
print("   🧹 cleanup(), cleanup_models() - 격리된 정리")
print("   📝 record_processing() - 처리 기록")
print("")
print("🎯 8단계 AI 파이프라인 Step별 Mixin (완전 격리):")
print("   1️⃣ HumanParsingMixin - 신체 영역 분할")
print("   2️⃣ PoseEstimationMixin - 포즈 감지")
print("   3️⃣ ClothSegmentationMixin - 의류 분할")
print("   4️⃣ GeometricMatchingMixin - 기하학적 매칭")
print("   5️⃣ ClothWarpingMixin - 의류 변형")
print("   6️⃣ VirtualFittingMixin - 가상 피팅 (핵심)")
print("   7️⃣ PostProcessingMixin - 후처리")
print("   8️⃣ QualityAssessmentMixin - 품질 평가")
print("")
print("🔥 완전 복원된 편의 함수들:")
print("   ✅ create_isolated_step_mixin() - 빠진 함수 복원!")
print("   ✅ create_step_mixin() - 기존 호환성")
print("   ✅ create_*_step() - 모든 Step 생성자")
print("   ✅ create_m3_max_optimized_step() - M3 Max 최적화")
print("")
print(f"🔧 시스템 정보:")
print(f"   conda 환경: {CONDA_INFO['conda_env']}")
print(f"   PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}")
print(f"   MPS (M3 Max): {'✅' if MPS_AVAILABLE else '❌'}")
print(f"   NumPy: {'✅' if NUMPY_AVAILABLE else '❌'}")
print(f"   PIL: {'✅' if PIL_AVAILABLE else '❌'}")
print("")
print("🎉 완전 격리 성공! - 순환참조 원천 차단!")
print("🎉 이제 BaseStepMixin은 어떤 모듈도 직접 참조하지 않습니다!")
print("🎉 모든 기능은 순수 의존성 주입으로만 제공됩니다!")
print("🔥 NameError: create_isolated_step_mixin 완전 해결!")
print("=" * 80)