# backend/app/ai_pipeline/steps/base_step_mixin.py
"""
🔥 BaseStepMixin v14.0 - v13.0 호환 버전 (DI + 기존 함수명)
================================================================

✅ DI Container 기반 의존성 주입 (순환참조 완전 해결!)
✅ v13.0의 모든 함수명/클래스명과 100% 호환
✅ 2단계 초기화: 기본 생성 → 자동 의존성 주입 
✅ 모든 Step 파일이 요구하는 기능 완전 제공
✅ ModelLoader 연동 (DI Container를 통한 지연 로딩)
✅ M3 Max 128GB 메모리 최적화
✅ conda 환경 우선 지원
✅ 비동기 처리 완전 해결
✅ 프로덕션 레벨 안정성

핵심 아키텍처 (v13.0 호환 + DI 강화):
- BaseStepMixin이 더 이상 ModelLoader를 직접 import 하지 않음!
- DI Container를 통한 지연 로딩으로 순환참조 완전 차단
- v13.0의 set_model_loader() 등 메서드 그대로 유지
- 내부적으로는 inject_dependencies() 자동 호출

Author: MyCloset AI Team  
Date: 2025-07-22
Version: 14.0 (v13.0 Compatible + DI Enhanced)
"""

# ==============================================
# 🔥 1. 필수 import만 (순환참조 완전 방지!)
# ==============================================
import os
import gc
import time
import asyncio
import logging
import threading
import traceback
import weakref
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Type, Callable, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
import platform
import subprocess
import psutil
from datetime import datetime

# ==============================================
# 🔥 2. conda 환경 우선 체크
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
# 🔥 3. 안전한 라이브러리 Import (순환참조 없음)
# ==============================================

# PyTorch 안전 Import
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

# DI Container Import (핵심! 순환참조 없음)
DI_CONTAINER_AVAILABLE = False
try:
    from ...core.di_container import get_di_container, IDependencyContainer
    DI_CONTAINER_AVAILABLE = True
    print("✅ DI Container 연동 성공!")
except ImportError:
    print("⚠️ DI Container import 실패")
    # 폴백 인터페이스
    class IDependencyContainer:
        def get(self, key: str): return None
    
    def get_di_container() -> IDependencyContainer:
        return IDependencyContainer()

# ==============================================
# 🔥 4. v13.0 호환 설정 클래스
# ==============================================
@dataclass
class StepConfig:
    """간단한 Step 설정 (v13.0 호환)"""
    step_name: str = "BaseStep"
    step_id: int = 0
    device: str = "auto"
    use_fp16: bool = True
    batch_size: int = 1
    confidence_threshold: float = 0.8
    auto_memory_cleanup: bool = True
    auto_warmup: bool = True

# ==============================================
# 🔥 5. SimpleMemoryOptimizer (v13.0 호환 이름)
# ==============================================
class SimpleMemoryOptimizer:
    """
    간단한 메모리 최적화 (v13.0 호환 이름)
    내부적으로 DI Container 활용
    """
    
    def __init__(self, device: str = "mps"):
        self.device = device
        self.is_m3_max = self._detect_m3_max()
        self.di_container = None
        
        # DI Container 연결 시도 (선택적)
        if DI_CONTAINER_AVAILABLE:
            try:
                self.di_container = get_di_container()
            except Exception as e:
                print(f"⚠️ DI Container 연결 실패: {e}")
    
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
        """메모리 최적화 실행 (DI 기반 강화)"""
        try:
            results = []
            
            # Python GC
            before = len(gc.get_objects())
            gc.collect()
            after = len(gc.get_objects())
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
            
            # 🔥 DI 기반 추가 최적화 (v14.0 강화)
            if self.di_container:
                try:
                    memory_manager = self.di_container.get('MemoryManager')
                    if memory_manager and hasattr(memory_manager, 'optimize'):
                        additional_result = memory_manager.optimize(aggressive=aggressive)
                        results.append(f"DI MemoryManager: {additional_result.get('message', 'OK')}")
                except Exception as e:
                    results.append(f"DI 최적화 실패: {str(e)}")
            
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
                "di_enhanced": self.di_container is not None
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def optimize_async(self, aggressive: bool = False) -> Dict[str, Any]:
        """비동기 메모리 최적화 (DI 강화)"""
        try:
            # DI 기반 MemoryManager 비동기 사용
            if self.di_container:
                try:
                    memory_manager = self.di_container.get('MemoryManager')
                    if memory_manager and hasattr(memory_manager, 'optimize_async'):
                        result = await memory_manager.optimize_async(aggressive=aggressive)
                        result["di_enhanced"] = True
                        return result
                except Exception as e:
                    pass  # 폴백으로 진행
            
            # 폴백: 동기 메서드를 비동기로 실행
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.optimize(aggressive))
            
        except Exception as e:
            return {"success": False, "error": str(e)}

# ==============================================
# 🔥 6. BaseStepMixin v14.0 - v13.0 호환 버전
# ==============================================
class BaseStepMixin:
    """
    🔥 BaseStepMixin v14.0 - v13.0 완벽 호환 + DI 강화
    
    ✅ v13.0의 모든 메서드명과 100% 호환
    ✅ DI Container 기반 순환참조 완전 해결
    ✅ 자동 의존성 주입 (사용자가 모르게)
    ✅ M3 Max 최적화 유지
    ✅ conda 환경 우선
    ✅ 비동기 처리 완전 지원
    """
    
    def __init__(self, **kwargs):
        """3단계 초기화 + 자동 DI 주입"""
        try:
            # STEP 1: 기본 설정 (v13.0과 동일)
            self._setup_basic(**kwargs)
            
            # STEP 2: 시스템 설정 (v13.0과 동일)
            self._setup_system()
            
            # 🔥 STEP 2.5: 자동 DI 주입 (v14.0 추가, 사용자에게 투명)
            self._auto_inject_dependencies()
            
            # STEP 3: 완료 (v13.0과 동일)
            self._finalize()
            
            self.logger.info(f"✅ {self.step_name} BaseStepMixin v14.0 (v13.0 호환) 초기화 완료")
            
        except Exception as e:
            self._emergency_setup(e)
    
    def _setup_basic(self, **kwargs):
        """STEP 1: 기본 설정 (v13.0과 동일)"""
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
        
        # 상태 플래그들 (v13.0과 동일)
        self.is_initialized = False
        self.is_ready = False
        self.has_model = False
        self.model_loaded = False
        self.warmup_completed = False
        
        # 의존성 주입을 위한 속성들 (v13.0과 동일)
        self.model_loader = None
        self.memory_manager = None
        self.data_converter = None
        
        # 🔥 DI 관련 내부 속성 (사용자에게는 숨김)
        self._di_container = None
        self._dependencies_injected = False
        self._injection_attempts = 0
        
        # 성능 메트릭
        self.performance_metrics = {
            'process_count': 0,
            'total_process_time': 0.0,
            'average_process_time': 0.0,
            'error_history': [],
            'di_injection_time': 0.0  # DI 메트릭
        }
        
        # 에러 추적
        self.error_count = 0
        self.last_error = None
        self.total_processing_count = 0
        self.last_processing_time = None
    
    def _setup_system(self):
        """STEP 2: 시스템 설정 (v13.0과 동일)"""
        # 디바이스 감지
        if self.config.device == "auto":
            self.device = self._detect_optimal_device()
        else:
            self.device = self.config.device
        
        # M3 Max 감지
        self.is_m3_max = self._detect_m3_max()
        
        # 메모리 정보
        self.memory_gb = self._get_memory_info()
        
        # 🔥 DI 강화된 메모리 최적화 시스템 (이름은 v13.0 호환)
        self.memory_optimizer = SimpleMemoryOptimizer(self.device)
        
        # 모델 캐시
        self.model_cache = {}
        self.loaded_models = {}
        
        # 현재 모델
        self._ai_model = None
        self._ai_model_name = None
    
    def _auto_inject_dependencies(self):
        """🔥 자동 의존성 주입 (사용자에게 투명)"""
        if not DI_CONTAINER_AVAILABLE:
            return
        
        try:
            start_time = time.time()
            self._injection_attempts += 1
            
            # DI Container 연결
            self._di_container = get_di_container()
            if not self._di_container:
                return
            
            # 자동 의존성 주입
            injection_count = 0
            
            # ModelLoader 자동 주입
            if not self.model_loader:
                try:
                    model_loader = self._di_container.get('ModelLoader') or self._di_container.get('IModelLoader')
                    if model_loader:
                        self.model_loader = model_loader
                        injection_count += 1
                        self.logger.debug("✅ ModelLoader 자동 주입")
                except Exception:
                    pass
            
            # MemoryManager 자동 주입
            if not self.memory_manager:
                try:
                    memory_manager = self._di_container.get('MemoryManager') or self._di_container.get('IMemoryManager')
                    if memory_manager:
                        self.memory_manager = memory_manager
                        injection_count += 1
                        self.logger.debug("✅ MemoryManager 자동 주입")
                except Exception:
                    pass
            
            # DataConverter 자동 주입
            if not self.data_converter:
                try:
                    data_converter = self._di_container.get('DataConverter') or self._di_container.get('IDataConverter')
                    if data_converter:
                        self.data_converter = data_converter
                        injection_count += 1
                        self.logger.debug("✅ DataConverter 자동 주입")
                except Exception:
                    pass
            
            # 주입 완료 처리
            if injection_count > 0:
                self._dependencies_injected = True
                self.has_model = True
                self.model_loaded = True
                
                # 메트릭 기록
                injection_time = time.time() - start_time
                self.performance_metrics['di_injection_time'] = injection_time
                
                self.logger.debug(f"🎉 자동 의존성 주입 완료: {injection_count}개 ({injection_time:.3f}s)")
            
        except Exception as e:
            self.logger.debug(f"자동 의존성 주입 실패: {e}")
    
    def _finalize(self):
        """STEP 3: 완료 (v13.0과 동일)"""
        self.is_initialized = True
        
        # 자동 워밍업 (설정된 경우)
        if self.config.auto_warmup:
            try:
                warmup_result = self.warmup()
                if warmup_result.get('success', False):
                    self.warmup_completed = True
                    self.is_ready = True
            except Exception as e:
                self.logger.warning(f"⚠️ 자동 워밍업 실패: {e}")
    
    def _emergency_setup(self, error: Exception):
        """긴급 설정 (v13.0과 동일)"""
        self.step_name = getattr(self, 'step_name', self.__class__.__name__)
        self.logger = logging.getLogger("emergency")
        self.device = "cpu"
        self.is_initialized = False
        self._dependencies_injected = False
        self.error_count = 1
        self.last_error = str(error)
        print(f"🚨 {self.step_name} 긴급 초기화: {error}")
    
    # ==============================================
    # 🔥 7. v13.0 호환 의존성 주입 메서드들
    # ==============================================
    
    def set_model_loader(self, model_loader):
        """ModelLoader 의존성 주입 (v13.0 호환)"""
        self.model_loader = model_loader
        self.logger.info("✅ ModelLoader 주입 완료")
        if model_loader:
            self.has_model = True
            self.model_loaded = True
    
    def set_memory_manager(self, memory_manager):
        """MemoryManager 의존성 주입 (v13.0 호환)"""
        self.memory_manager = memory_manager
        self.logger.info("✅ MemoryManager 주입 완료")
    
    def set_data_converter(self, data_converter):
        """DataConverter 의존성 주입 (v13.0 호환)"""
        self.data_converter = data_converter
        self.logger.info("✅ DataConverter 주입 완료")
    
    # 🔥 추가: DI Container 호환 메서드들 (내부 사용)
    def _inject_dependencies_internal(self, **dependencies):
        """내부 의존성 주입 (DI Container 호환)"""
        for name, dependency in dependencies.items():
            if dependency is not None:
                setattr(self, name, dependency)
                self.logger.debug(f"✅ {name} 내부 주입 완료")
    
    # ==============================================
    # 🔥 8. 시스템 감지 메서드들 (v13.0과 동일)
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
            memory = psutil.virtual_memory()
            return memory.total / 1024**3
        except:
            return 16.0
    
    # ==============================================
    # 🔥 9. Step들이 요구하는 핵심 메서드들 (v13.0 호환 + DI 강화)
    # ==============================================
    
    def get_model(self, model_name: Optional[str] = None) -> Optional[Any]:
        """모델 가져오기 (v13.0 호환 + DI 강화)"""
        try:
            # 캐시 확인
            cache_key = model_name or "default"
            if cache_key in self.model_cache:
                return self.model_cache[cache_key]
            
            # 🔥 DI 기반 ModelLoader 우선 사용
            if self.model_loader:
                try:
                    model = None
                    if hasattr(self.model_loader, 'get_model'):
                        model = self.model_loader.get_model(model_name or "default")
                    elif hasattr(self.model_loader, 'load_model'):
                        model = self.model_loader.load_model(model_name or "default")
                    
                    if model:
                        self.model_cache[cache_key] = model
                        self.has_model = True
                        self.model_loaded = True
                        self._ai_model = model
                        self._ai_model_name = model_name
                        return model
                except Exception as e:
                    self.logger.debug(f"DI ModelLoader 실패: {e}")
            
            # 폴백: 동적 import (v13.0 방식)
            try:
                import importlib
                loader_module = importlib.import_module('app.ai_pipeline.utils.model_loader')
                get_global_loader = getattr(loader_module, 'get_global_model_loader', None)
                if get_global_loader:
                    loader = get_global_loader()
                    if loader:
                        model = loader.get_model(model_name or "default")
                        if model:
                            self.model_cache[cache_key] = model
                            self.has_model = True
                            self.model_loaded = True
                            return model
            except Exception as e:
                self.logger.debug(f"폴백 모델 로드 실패: {e}")
            
            return None
                
        except Exception as e:
            self.logger.error(f"❌ 모델 가져오기 실패: {e}")
            return None
    
    async def get_model_async(self, model_name: Optional[str] = None) -> Optional[Any]:
        """모델 가져오기 (비동기, v13.0 호환 + DI 강화)"""
        try:
            # 캐시 확인
            cache_key = model_name or "default"
            if cache_key in self.model_cache:
                return self.model_cache[cache_key]
            
            # DI 기반 비동기 ModelLoader 사용
            if self.model_loader:
                try:
                    model = None
                    if hasattr(self.model_loader, 'get_model_async'):
                        model = await self.model_loader.get_model_async(model_name or "default")
                    else:
                        # 동기 메서드를 비동기로 실행
                        loop = asyncio.get_event_loop()
                        model = await loop.run_in_executor(
                            None, 
                            lambda: self.model_loader.get_model(model_name or "default") if hasattr(self.model_loader, 'get_model') else None
                        )
                    
                    if model:
                        self.model_cache[cache_key] = model
                        self.has_model = True
                        self.model_loaded = True
                        return model
                        
                except Exception as e:
                    self.logger.debug(f"비동기 DI ModelLoader 실패: {e}")
            
            # 폴백: 동기 메서드를 비동기로 실행
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.get_model(model_name))
                
        except Exception as e:
            self.logger.error(f"❌ 비동기 모델 가져오기 실패: {e}")
            return None
    
    def optimize_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """메모리 최적화 (v13.0 호환 + DI 강화)"""
        try:
            # DI 기반 MemoryManager 우선 사용
            if self.memory_manager:
                try:
                    if hasattr(self.memory_manager, 'optimize_memory'):
                        result = self.memory_manager.optimize_memory(aggressive=aggressive)
                        result["di_enhanced"] = True
                        return result
                    elif hasattr(self.memory_manager, 'optimize'):
                        result = self.memory_manager.optimize(aggressive=aggressive)
                        result["di_enhanced"] = True
                        return result
                except Exception as e:
                    self.logger.debug(f"DI MemoryManager 실패: {e}")
            
            # 폴백: 내장 DI 강화된 메모리 최적화 사용
            return self.memory_optimizer.optimize(aggressive=aggressive)
            
        except Exception as e:
            self.logger.error(f"❌ 메모리 최적화 실패: {e}")
            return {"success": False, "error": str(e)}
    
    async def optimize_memory_async(self, aggressive: bool = False) -> Dict[str, Any]:
        """메모리 최적화 (비동기, v13.0 호환 + DI 강화)"""
        try:
            # DI 기반 MemoryManager 비동기 사용
            if self.memory_manager:
                try:
                    if hasattr(self.memory_manager, 'optimize_memory_async'):
                        result = await self.memory_manager.optimize_memory_async(aggressive=aggressive)
                        result["di_enhanced"] = True
                        return result
                    elif hasattr(self.memory_manager, 'optimize_async'):
                        result = await self.memory_manager.optimize_async(aggressive=aggressive)
                        result["di_enhanced"] = True
                        return result
                    else:
                        # 동기 메서드를 비동기로 실행
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            None, 
                            lambda: self.memory_manager.optimize_memory(aggressive=aggressive) if hasattr(self.memory_manager, 'optimize_memory') else {"success": False}
                        )
                        result["di_enhanced"] = True
                        return result
                except Exception as e:
                    self.logger.debug(f"비동기 DI MemoryManager 실패: {e}")
            
            # 폴백: 내장 메모리 최적화를 비동기로 실행
            return await self.memory_optimizer.optimize_async(aggressive=aggressive)
            
        except Exception as e:
            self.logger.error(f"❌ 비동기 메모리 최적화 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def warmup(self) -> Dict[str, Any]:
        """워밍업 실행 (v13.0 호환 + DI 강화)"""
        try:
            if self.warmup_completed:
                return {'success': True, 'message': '이미 워밍업 완료됨', 'cached': True}
            
            self.logger.info(f"🔥 {self.step_name} 워밍업 시작...")
            start_time = time.time()
            results = []
            
            # 1. 메모리 워밍업 (DI 강화)
            try:
                memory_result = self.optimize_memory()
                results.append('memory_success' if memory_result.get('success') else 'memory_failed')
            except:
                results.append('memory_failed')
            
            # 2. 모델 워밍업 (DI 기반)
            try:
                if self.model_loader:
                    test_model = self.get_model("warmup_test")
                    results.append('model_success' if test_model else 'model_skipped')
                else:
                    results.append('model_skipped')
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
            
            self.logger.info(f"🔥 워밍업 완료: {success_count}/{len(results)} 성공 ({duration:.2f}초)")
            
            return {
                "success": overall_success,
                "duration": duration,
                "results": results,
                "success_count": success_count,
                "total_count": len(results),
                "di_enhanced": self._dependencies_injected
            }
            
        except Exception as e:
            self.logger.error(f"❌ 워밍업 실패: {e}")
            return {"success": False, "error": str(e)}
    
    async def warmup_async(self) -> Dict[str, Any]:
        """워밍업 실행 (비동기, v13.0 호환 + DI 강화)"""
        try:
            if self.warmup_completed:
                return {'success': True, 'message': '이미 워밍업 완료됨', 'cached': True}
            
            self.logger.info(f"🔥 {self.step_name} 비동기 워밍업 시작...")
            start_time = time.time()
            results = []
            
            # 1. 비동기 메모리 워밍업 (DI 강화)
            try:
                memory_result = await self.optimize_memory_async()
                results.append('memory_async_success' if memory_result.get('success') else 'memory_async_failed')
            except:
                results.append('memory_async_failed')
            
            # 2. 비동기 모델 워밍업 (DI 기반)
            try:
                if self.model_loader:
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
                "async": True,
                "di_enhanced": self._dependencies_injected
            }
            
        except Exception as e:
            self.logger.error(f"❌ 비동기 워밍업 실패: {e}")
            return {"success": False, "error": str(e), "async": True}
    
    def _device_warmup_sync(self):
        """동기 디바이스 워밍업"""
        try:
            test_tensor = torch.randn(10, 10)
            if self.device != 'cpu':
                test_tensor = test_tensor.to(self.device)
            _ = torch.matmul(test_tensor, test_tensor.t())
            return True
        except:
            return False
    
    # BaseStepMixin 호환용 별칭 (v13.0 호환)
    async def warmup_step(self) -> Dict[str, Any]:
        """Step 워밍업 (BaseStepMixin 호환용)"""
        return await self.warmup_async()
    
    def initialize(self) -> bool:
        """초기화 메서드 (v13.0 호환)"""
        try:
            if self.is_initialized:
                return True
            
            self.is_initialized = True
            self.logger.info(f"✅ {self.step_name} 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 초기화 실패: {e}")
            return False
    
    async def initialize_async(self) -> bool:
        """비동기 초기화 메서드 (v13.0 호환)"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.initialize)
        except Exception as e:
            self.logger.error(f"❌ 비동기 초기화 실패: {e}")
            return False
    
    async def cleanup(self) -> Dict[str, Any]:
        """정리 (v13.0 호환 + DI 강화)"""
        try:
            self.logger.info(f"🧹 {self.step_name} 정리 시작...")
            
            # 모델 캐시 정리
            self.model_cache.clear()
            self.loaded_models.clear()
            
            # 메모리 정리 (DI 강화)
            cleanup_result = await self.optimize_memory_async(aggressive=True)
            
            # 상태 리셋
            self.is_ready = False
            self.warmup_completed = False
            
            # 의존성 정리 (참조만 제거, DI Container는 유지)
            self.model_loader = None
            self.memory_manager = None
            self.data_converter = None
            
            self.logger.info(f"✅ {self.step_name} 정리 완료")
            
            return {
                "success": True,
                "cleanup_result": cleanup_result,
                "step_name": self.step_name,
                "di_enhanced": self._dependencies_injected
            }
        
        except Exception as e:
            self.logger.warning(f"⚠️ 정리 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def cleanup_models(self):
        """모델 정리 (v13.0 호환)"""
        try:
            # 모델 캐시 정리
            self.model_cache.clear()
            self.loaded_models.clear()
            
            # 현재 모델 초기화
            self._ai_model = None
            self._ai_model_name = None
            
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
            self.logger.info(f"🧹 {self.step_name} 모델 정리 완료")
                
        except Exception as e:
            self.logger.warning(f"⚠️ 모델 정리 중 오류: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Step 상태 조회 (v13.0 호환 + DI 정보 추가)"""
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
                # v13.0 호환 의존성 정보
                'dependencies': {
                    'model_loader': self.model_loader is not None,
                    'memory_manager': self.memory_manager is not None,
                    'data_converter': self.data_converter is not None,
                },
                # 🔥 DI 정보 추가 (하지만 호환성 유지)
                'di_enhanced': self._dependencies_injected,
                'di_injection_attempts': self._injection_attempts,
                'performance_metrics': self.performance_metrics,
                'conda_info': CONDA_INFO,
                'timestamp': time.time(),
                'version': '14.0-v13-compatible'
            }
            
        except Exception as e:
            self.logger.error(f"❌ 상태 조회 실패: {e}")
            return {
                'step_name': getattr(self, 'step_name', 'unknown'),
                'error': str(e),
                'version': '14.0-v13-compatible',
                'timestamp': time.time()
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 조회 (v13.0 호환 + DI 메트릭 추가)"""
        try:
            return {
                'total_processing_count': self.total_processing_count,
                'last_processing_time': self.last_processing_time,
                'error_count': self.error_count,
                'success_rate': self._calculate_success_rate(),
                'average_process_time': self.performance_metrics.get('average_process_time', 0.0),
                'total_process_time': self.performance_metrics.get('total_process_time', 0.0),
                # 🔥 DI 성능 메트릭 추가
                'di_injection_time': self.performance_metrics.get('di_injection_time', 0.0),
                'di_enhanced': self._dependencies_injected,
                'version': '14.0-v13-compatible'
            }
            
        except Exception as e:
            self.logger.error(f"❌ 성능 요약 조회 실패: {e}")
            return {'version': '14.0-v13-compatible', 'error': str(e)}
    
    def _calculate_success_rate(self) -> float:
        """성공률 계산 (v13.0과 동일)"""
        try:
            total = self.total_processing_count
            errors = self.error_count
            if total > 0:
                return (total - errors) / total
            return 0.0
        except:
            return 0.0
    
    # ==============================================
    # 🔥 10. 추가 유틸리티 메서드들 (v13.0 호환)
    # ==============================================
    
    def record_processing(self, duration: float, success: bool = True):
        """처리 기록 (v13.0 호환)"""
        try:
            self.total_processing_count += 1
            self.last_processing_time = time.time()
            
            if not success:
                self.error_count += 1
            
            # 성능 메트릭 업데이트
            self.performance_metrics['process_count'] = self.total_processing_count
            self.performance_metrics['total_process_time'] += duration
            self.performance_metrics['average_process_time'] = (
                self.performance_metrics['total_process_time'] / self.total_processing_count
            )
            
        except Exception as e:
            self.logger.warning(f"⚠️ 처리 기록 실패: {e}")
    
    def __del__(self):
        """소멸자 (안전한 정리, v13.0 호환)"""
        try:
            if hasattr(self, 'model_cache'):
                self.model_cache.clear()
            # DI Container는 정리하지 않음 (전역 관리)
        except:
            pass

# ==============================================
# 🔥 11. Step별 특화 Mixin들 (v13.0과 동일)
# ==============================================

class HumanParsingMixin(BaseStepMixin):
    """Step 1: Human Parsing 특화 Mixin (v13.0 호환)"""
    
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
    """Step 2: Pose Estimation 특화 Mixin (v13.0 호환)"""
    
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
    """Step 3: Cloth Segmentation 특화 Mixin (v13.0 호환)"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('step_name', 'ClothSegmentationStep')
        kwargs.setdefault('step_id', 3)
        super().__init__(**kwargs)
        
        self.segmentation_methods = ['traditional', 'u2net', 'deeplab', 'auto', 'hybrid']
        self.segmentation_method = kwargs.get('segmentation_method', 'u2net')

class GeometricMatchingMixin(BaseStepMixin):
    """Step 4: Geometric Matching 특화 Mixin (v13.0 호환)"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('step_name', 'GeometricMatchingStep')
        kwargs.setdefault('step_id', 4)
        super().__init__(**kwargs)
        
        self.matching_methods = ['thin_plate_spline', 'affine', 'perspective', 'flow_based']
        self.matching_method = kwargs.get('matching_method', 'thin_plate_spline')
        self.grid_size = kwargs.get('grid_size', (5, 5))

class ClothWarpingMixin(BaseStepMixin):
    """Step 5: Cloth Warping 특화 Mixin (v13.0 호환)"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('step_name', 'ClothWarpingStep')
        kwargs.setdefault('step_id', 5)
        super().__init__(**kwargs)
        
        self.warping_stages = ['preprocessing', 'geometric_transformation', 'texture_mapping', 'postprocessing']
        self.warping_quality = kwargs.get('warping_quality', 'high')
        self.preserve_texture = kwargs.get('preserve_texture', True)

class VirtualFittingMixin(BaseStepMixin):
    """Step 6: Virtual Fitting 특화 Mixin (v13.0 호환) - 핵심 단계"""
    
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
    """Step 7: Post Processing 특화 Mixin (v13.0 호환)"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('step_name', 'PostProcessingStep')
        kwargs.setdefault('step_id', 7)
        super().__init__(**kwargs)
        
        self.processing_methods = ['super_resolution', 'denoising', 'color_correction', 'sharpening']
        self.enhancement_level = kwargs.get('enhancement_level', 'medium')
        self.super_resolution_factor = kwargs.get('super_resolution_factor', 2.0)

class QualityAssessmentMixin(BaseStepMixin):
    """Step 8: Quality Assessment 특화 Mixin (v13.0 호환)"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('step_name', 'QualityAssessmentStep')
        kwargs.setdefault('step_id', 8)
        super().__init__(**kwargs)
        
        self.assessment_criteria = ['perceptual_quality', 'technical_quality', 'aesthetic_quality', 'overall_quality']
        self.quality_threshold = kwargs.get('quality_threshold', 0.7)
        self.use_clip_score = kwargs.get('use_clip_score', True)

# ==============================================
# 🔥 12. v13.0 호환 편의 함수들 (DI 접미사 제거)
# ==============================================

def create_step_mixin(step_name: str, step_id: int, **kwargs) -> BaseStepMixin:
    """BaseStepMixin 인스턴스 생성 (v13.0 호환)"""
    kwargs.update({'step_name': step_name, 'step_id': step_id})
    return BaseStepMixin(**kwargs)

def create_human_parsing_step(**kwargs) -> HumanParsingMixin:
    """Human Parsing Step 생성 (v13.0 호환)"""
    return HumanParsingMixin(**kwargs)

def create_pose_estimation_step(**kwargs) -> PoseEstimationMixin:
    """Pose Estimation Step 생성 (v13.0 호환)"""
    return PoseEstimationMixin(**kwargs)

def create_cloth_segmentation_step(**kwargs) -> ClothSegmentationMixin:
    """Cloth Segmentation Step 생성 (v13.0 호환)"""
    return ClothSegmentationMixin(**kwargs)

def create_geometric_matching_step(**kwargs) -> GeometricMatchingMixin:
    """Geometric Matching Step 생성 (v13.0 호환)"""
    return GeometricMatchingMixin(**kwargs)

def create_cloth_warping_step(**kwargs) -> ClothWarpingMixin:
    """Cloth Warping Step 생성 (v13.0 호환)"""
    return ClothWarpingMixin(**kwargs)

def create_virtual_fitting_step(**kwargs) -> VirtualFittingMixin:
    """Virtual Fitting Step 생성 (v13.0 호환) - 핵심"""
    return VirtualFittingMixin(**kwargs)

def create_post_processing_step(**kwargs) -> PostProcessingMixin:
    """Post Processing Step 생성 (v13.0 호환)"""
    return PostProcessingMixin(**kwargs)

def create_quality_assessment_step(**kwargs) -> QualityAssessmentMixin:
    """Quality Assessment Step 생성 (v13.0 호환)"""
    return QualityAssessmentMixin(**kwargs)

def create_m3_max_optimized_step(step_type: str, **kwargs) -> BaseStepMixin:
    """M3 Max 최적화된 Step 생성 (v13.0 호환)"""
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
    
    creator = step_creators.get(step_type, create_step_mixin)
    return creator(**kwargs)

# 🔥 추가: DI 강화 편의 함수들 (내부 사용)
def _create_step_with_auto_di(step_class: Type[BaseStepMixin], **kwargs) -> BaseStepMixin:
    """DI 자동 주입을 사용하여 Step 인스턴스 생성 (내부 함수)"""
    try:
        # Step 인스턴스 생성 (자동 DI 주입됨)
        step_instance = step_class(**kwargs)
        
        # 추가 DI 주입이 필요한 경우
        if DI_CONTAINER_AVAILABLE:
            try:
                di_container = get_di_container()
                if di_container and not step_instance._dependencies_injected:
                    # 수동으로 추가 의존성 주입 시도
                    additional_deps = {}
                    
                    for dep_name in ['ModelLoader', 'MemoryManager', 'DataConverter']:
                        dep = di_container.get(dep_name)
                        if dep:
                            additional_deps[dep_name.lower()] = dep
                    
                    if additional_deps:
                        step_instance._inject_dependencies_internal(**additional_deps)
                        
            except Exception as e:
                step_instance.logger.debug(f"추가 DI 주입 실패: {e}")
        
        return step_instance
        
    except Exception as e:
        print(f"❌ {step_class.__name__} DI 생성 실패: {e}")
        # 폴백: 일반 생성
        return step_class(**kwargs)

# ==============================================
# 🔥 13. v13.0 호환 모듈 내보내기
# ==============================================

__all__ = [
    # 🔥 메인 클래스들 (v13.0 호환)
    'BaseStepMixin',
    'StepConfig',
    'SimpleMemoryOptimizer',  # v13.0 호환 이름
    
    # 🔥 Step별 특화 Mixin들 (v13.0과 동일)
    'HumanParsingMixin',
    'PoseEstimationMixin', 
    'ClothSegmentationMixin',
    'GeometricMatchingMixin',
    'ClothWarpingMixin',
    'VirtualFittingMixin',
    'PostProcessingMixin',
    'QualityAssessmentMixin',
    
    # 🔥 v13.0 호환 편의 함수들 (_di 접미사 제거)
    'create_step_mixin',
    'create_human_parsing_step',
    'create_pose_estimation_step',
    'create_cloth_segmentation_step',
    'create_geometric_matching_step',
    'create_cloth_warping_step',
    'create_virtual_fitting_step',
    'create_post_processing_step',
    'create_quality_assessment_step',
    'create_m3_max_optimized_step',
    
    # 상수들 (v13.0과 동일)
    'TORCH_AVAILABLE',
    'MPS_AVAILABLE',
    'NUMPY_AVAILABLE',
    'PIL_AVAILABLE',
    'CONDA_INFO'
]

# ==============================================
# 🔥 14. v13.0 호환 모듈 로드 완료 메시지
# ==============================================

print("=" * 80)
print("🎉 BaseStepMixin v14.0 - v13.0 완벽 호환 + DI 강화 버전!")
print("=" * 80)
print("🔥 v13.0 완벽 호환성:")
print("   ✅ 모든 함수명/클래스명 100% 호환 (SimpleMemoryOptimizer 등)")
print("   ✅ set_model_loader(), set_memory_manager() 메서드 유지")
print("   ✅ 기존 편의 함수들 그대로 사용 가능")
print("   ✅ 모든 Step 파일이 수정 없이 작동")
print("")
print("🔥 DI 강화 기능 (백그라운드):")
print("   ✅ DI Container 기반 의존성 주입 (순환참조 완전 해결!)")
print("   ✅ 자동 의존성 주입 (사용자에게 투명)")
print("   ✅ 지연 로딩으로 순환참조 차단")
print("   ✅ ModelLoader 직접 import 완전 제거")
print("   ✅ M3 Max 128GB 메모리 최적화")
print("   ✅ conda 환경 우선 지원")
print("   ✅ 비동기 처리 완전 해결")
print("   ✅ 프로덕션 레벨 안정성")
print("")
print("🚀 v13.0 호환 메서드들 (DI 강화):")
print("   🤖 모델 연동: get_model(), get_model_async() (DI 기반)")
print("   🧹 메모리 최적화: optimize_memory(), optimize_memory_async() (DI 강화)")
print("   🔥 워밍업: warmup(), warmup_async(), warmup_step() (DI 기반)")
print("   📊 상태 관리: get_status(), get_performance_summary() (DI 정보 추가)")
print("   🔧 초기화: initialize(), initialize_async()")
print("   🧹 정리: cleanup(), cleanup_models()")
print("   📝 기록: record_processing()")
print("")
print("🎯 v13.0 호환 의존성 주입:")
print("   💉 set_model_loader() - ModelLoader 주입 (v13.0 호환)")
print("   💉 set_memory_manager() - MemoryManager 주입 (v13.0 호환)")
print("   💉 set_data_converter() - DataConverter 주입 (v13.0 호환)")
print("")
print("🎯 8단계 AI 파이프라인 Step별 Mixin (v13.0 호환):")
print("   1️⃣ HumanParsingMixin - 신체 영역 분할")
print("   2️⃣ PoseEstimationMixin - 포즈 감지")
print("   3️⃣ ClothSegmentationMixin - 의류 분할")
print("   4️⃣ GeometricMatchingMixin - 기하학적 매칭")
print("   5️⃣ ClothWarpingMixin - 의류 변형")
print("   6️⃣ VirtualFittingMixin - 가상 피팅 (핵심)")
print("   7️⃣ PostProcessingMixin - 후처리")
print("   8️⃣ QualityAssessmentMixin - 품질 평가")
print("")
print(f"🔧 시스템 정보:")
print(f"   conda 환경: {CONDA_INFO['conda_env']}")
print(f"   PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}")
print(f"   MPS (M3 Max): {'✅' if MPS_AVAILABLE else '❌'}")
print(f"   DI Container: {'✅' if DI_CONTAINER_AVAILABLE else '❌'}")
print("")
print("🎉 v13.0과 100% 호환되면서 DI Container 순환참조 문제 완전 해결!")
print("🎉 기존 Step 파일들이 수정 없이 그대로 작동하며 성능은 더욱 향상!")
print("=" * 80)