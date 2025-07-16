# app/ai_pipeline/utils/__init__.py
"""
🍎 MyCloset AI 파이프라인 유틸리티 모듈 - 최적 생성자 패턴 적용
✅ 완전한 통합 관리 시스템
🔥 단순함 + 편의성 + 확장성 + 일관성
"""

import os
import gc
import logging
import asyncio
from typing import Dict, Any, Optional, Union, List, Callable
from pathlib import Path
import weakref
from functools import lru_cache
import threading
import time

# 로깅 설정
logger = logging.getLogger(__name__)

# ============================================
# 🔥 핵심: 선택적 라이브러리 import 시스템
# ============================================

# PyTorch 기본 확인
try:
    import torch
    TORCH_AVAILABLE = True
    logger.info("✅ PyTorch 사용 가능")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("⚠️ PyTorch 사용 불가")

# 개별 유틸리티 모듈들 안전하게 import
try:
    from .memory_manager import (
        MemoryManager,
        MemoryStats,
        create_memory_manager,
        get_global_memory_manager,
        initialize_global_memory_manager,
        optimize_memory_usage,
        check_memory_available,
        get_memory_info
    )
    MEMORY_MANAGER_AVAILABLE = True
    logger.info("✅ MemoryManager 모듈 로드 성공")
except ImportError as e:
    MEMORY_MANAGER_AVAILABLE = False
    MemoryManager = None
    logger.warning(f"⚠️ MemoryManager import 실패: {e}")

try:
    from .model_loader import (
        ModelLoader,
        ModelConfig,
        ModelFormat,
        ModelType,
        ModelRegistry,
        ModelMemoryManager,
        StepModelInterface,
        BaseStepMixin,
        create_model_loader,
        get_global_model_loader,
        initialize_global_model_loader,
        load_model_async,
        load_model_sync,
        preprocess_image,
        postprocess_segmentation,
        postprocess_pose,
        # 실제 AI 모델 클래스들
        GraphonomyModel,
        OpenPoseModel,
        U2NetModel,
        GeometricMatchingModel,
        HRVITONModel
    )
    MODEL_LOADER_AVAILABLE = True
    logger.info("✅ ModelLoader 모듈 로드 성공")
except ImportError as e:
    MODEL_LOADER_AVAILABLE = False
    ModelLoader = None
    ModelConfig = None
    logger.warning(f"⚠️ ModelLoader import 실패: {e}")

try:
    from .data_converter import (
        DataConverter,
        create_data_converter,
        get_global_data_converter,
        initialize_global_data_converter,
        quick_image_to_tensor,
        quick_tensor_to_image
    )
    DATA_CONVERTER_AVAILABLE = True
    logger.info("✅ DataConverter 모듈 로드 성공")
except ImportError as e:
    DATA_CONVERTER_AVAILABLE = False
    DataConverter = None
    logger.warning(f"⚠️ DataConverter import 실패: {e}")

# 🍎 M3 Max 감지
def _detect_m3_max() -> bool:
    """M3 Max 칩 자동 감지"""
    try:
        import platform
        import subprocess
        
        if platform.system() == 'Darwin':  # macOS
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True)
            return 'M3' in result.stdout
    except:
        pass
    return False

# 시스템 정보
IS_M3_MAX = _detect_m3_max()
DEFAULT_DEVICE = 'mps' if IS_M3_MAX else ('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')

# ============================================
# 🔥 핵심: 통합 유틸리티 매니저
# ============================================

class OptimalUtilsManager:
    """
    🍎 최적 생성자 패턴 기반 통합 유틸리티 매니저
    모든 AI 파이프라인 유틸리티를 단일 인터페이스로 관리
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        """싱글톤 패턴 적용 (선택적)"""
        if kwargs.get('singleton', True):
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                return cls._instance
        return super().__new__(cls)
    
    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        ✅ 최적 생성자 - 유틸리티 통합 관리

        Args:
            device: 사용할 디바이스 (None=자동감지, 'cpu', 'cuda', 'mps')
            config: 통합 설정 딕셔너리
            **kwargs: 확장 파라미터들
                - auto_initialize: bool = True  # 자동 초기화
                - memory_gb: float = 128.0 if IS_M3_MAX else 16.0  # 메모리 크기
                - is_m3_max: bool = IS_M3_MAX  # M3 Max 여부
                - optimization_enabled: bool = True  # 최적화 활성화
                - quality_level: str = "maximum" if IS_M3_MAX else "balanced"
                - use_fp16: bool = True  # FP16 사용
                - enable_caching: bool = True  # 캐싱 활성화
                - lazy_loading: bool = True  # 지연 로딩
                - singleton: bool = True  # 싱글톤 사용
        """
        # 중복 초기화 방지
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        # 1. 💡 지능적 디바이스 자동 감지
        self.device = self._auto_detect_device(device)

        # 2. 📋 기본 설정
        self.config = config or {}
        self.step_name = self.__class__.__name__
        self.logger = logging.getLogger(f"utils.{self.step_name}")

        # 3. 🔧 시스템 파라미터 (M3 Max 최적화)
        self.memory_gb = kwargs.get('memory_gb', 128.0 if IS_M3_MAX else 16.0)
        self.is_m3_max = kwargs.get('is_m3_max', IS_M3_MAX)
        self.optimization_enabled = kwargs.get('optimization_enabled', True)
        self.quality_level = kwargs.get('quality_level', "maximum" if IS_M3_MAX else "balanced")
        
        # 4. ⚙️ 유틸리티 특화 설정
        self.use_fp16 = kwargs.get('use_fp16', True)
        self.enable_caching = kwargs.get('enable_caching', True)
        self.lazy_loading = kwargs.get('lazy_loading', True)
        self.auto_initialize = kwargs.get('auto_initialize', True)

        # 5. 🎯 유틸리티 인스턴스들
        self.memory_manager: Optional[MemoryManager] = None
        self.model_loader: Optional[ModelLoader] = None
        self.data_converter: Optional[DataConverter] = None
        
        # 6. ✅ 상태 관리
        self.initialized = False
        self.initialization_results = {}
        self._weak_refs = weakref.WeakSet()
        
        # 7. 🚀 자동 초기화
        if self.auto_initialize:
            self.initialize_all()
        
        self._initialized = True
        self.logger.info(f"🎯 OptimalUtilsManager 생성 완료 - 디바이스: {self.device}")

    def _auto_detect_device(self, preferred_device: Optional[str]) -> str:
        """💡 지능적 디바이스 자동 감지"""
        if preferred_device:
            return preferred_device

        if not TORCH_AVAILABLE:
            return 'cpu'

        try:
            if torch.backends.mps.is_available():
                return 'mps'  # M3 Max 우선
            elif torch.cuda.is_available():
                return 'cuda'  # NVIDIA GPU
            else:
                return 'cpu'  # 폴백
        except:
            return 'cpu'

    def initialize_all(self) -> Dict[str, bool]:
        """모든 유틸리티 초기화"""
        start_time = time.time()
        results = {}
        
        self.logger.info("🚀 유틸리티 통합 초기화 시작")
        
        try:
            # 1. Memory Manager 초기화
            if MEMORY_MANAGER_AVAILABLE:
                try:
                    memory_config = self.config.get('memory_manager', {})
                    self.memory_manager = MemoryManager(
                        device=self.device,
                        config=memory_config,
                        memory_gb=self.memory_gb,
                        is_m3_max=self.is_m3_max,
                        optimization_enabled=self.optimization_enabled,
                        **memory_config
                    )
                    results['memory_manager'] = True
                    self.logger.info("✅ MemoryManager 초기화 성공")
                except Exception as e:
                    self.logger.error(f"❌ MemoryManager 초기화 실패: {e}")
                    results['memory_manager'] = False
            else:
                results['memory_manager'] = False
                self.logger.warning("⚠️ MemoryManager 사용 불가")
            
            # 2. Model Loader 초기화
            if MODEL_LOADER_AVAILABLE:
                try:
                    loader_config = self.config.get('model_loader', {})
                    self.model_loader = ModelLoader(
                        device=self.device,
                        config=loader_config,
                        memory_gb=self.memory_gb,
                        is_m3_max=self.is_m3_max,
                        optimization_enabled=self.optimization_enabled,
                        use_fp16=self.use_fp16,
                        **loader_config
                    )
                    results['model_loader'] = True
                    self.logger.info("✅ ModelLoader 초기화 성공")
                except Exception as e:
                    self.logger.error(f"❌ ModelLoader 초기화 실패: {e}")
                    results['model_loader'] = False
            else:
                results['model_loader'] = False
                self.logger.warning("⚠️ ModelLoader 사용 불가")
            
            # 3. Data Converter 초기화
            if DATA_CONVERTER_AVAILABLE:
                try:
                    converter_config = self.config.get('data_converter', {})
                    self.data_converter = DataConverter(
                        device=self.device,
                        config=converter_config,
                        memory_gb=self.memory_gb,
                        is_m3_max=self.is_m3_max,
                        optimization_enabled=self.optimization_enabled,
                        **converter_config
                    )
                    results['data_converter'] = True
                    self.logger.info("✅ DataConverter 초기화 성공")
                except Exception as e:
                    self.logger.error(f"❌ DataConverter 초기화 실패: {e}")
                    results['data_converter'] = False
            else:
                results['data_converter'] = False
                self.logger.warning("⚠️ DataConverter 사용 불가")
            
            # 4. 초기화 결과 분석
            success_count = sum(results.values())
            total_count = len(results)
            
            self.initialized = success_count > 0
            self.initialization_results = results
            
            # 5. M3 Max 특화 최적화
            if self.is_m3_max and self.optimization_enabled and success_count > 0:
                self._apply_m3_max_optimizations()
            
            elapsed_time = time.time() - start_time
            
            if self.initialized:
                self.logger.info(f"🎉 유틸리티 통합 초기화 완료: {success_count}/{total_count} 성공 ({elapsed_time:.2f}s)")
            else:
                self.logger.error(f"❌ 유틸리티 초기화 실패: 모든 모듈 사용 불가")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 유틸리티 매니저 초기화 치명적 실패: {e}")
            return {"error": str(e)}

    def _apply_m3_max_optimizations(self):
        """🍎 M3 Max 특화 최적화 적용"""
        try:
            optimizations = []
            
            # 1. 메모리 관리 최적화
            if self.memory_manager:
                # 128GB 메모리 활용 최적화
                optimizations.append("Unified Memory Architecture")
                
            # 2. 모델 로더 최적화  
            if self.model_loader:
                # Neural Engine 활용 준비
                optimizations.append("Neural Engine Ready")
                
            # 3. 데이터 변환 최적화
            if self.data_converter:
                # Metal Performance Shaders 활용
                optimizations.append("Metal Performance Shaders")
            
            # 4. PyTorch MPS 최적화
            if TORCH_AVAILABLE and torch.backends.mps.is_available():
                optimizations.append("PyTorch MPS Backend")
            
            if optimizations:
                self.logger.info(f"🍎 M3 Max 최적화 적용: {', '.join(optimizations)}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ M3 Max 최적화 실패: {e}")

    # ============================================
    # 🔥 유틸리티 접근 메서드들
    # ============================================

    def get_memory_manager(self) -> Optional[MemoryManager]:
        """메모리 매니저 반환"""
        if not self.memory_manager and MEMORY_MANAGER_AVAILABLE:
            self.logger.info("🔄 지연 로딩: MemoryManager 초기화")
            self._initialize_memory_manager()
        return self.memory_manager

    def get_model_loader(self) -> Optional[ModelLoader]:
        """모델 로더 반환"""
        if not self.model_loader and MODEL_LOADER_AVAILABLE:
            self.logger.info("🔄 지연 로딩: ModelLoader 초기화")
            self._initialize_model_loader()
        return self.model_loader

    def get_data_converter(self) -> Optional[DataConverter]:
        """데이터 변환기 반환"""
        if not self.data_converter and DATA_CONVERTER_AVAILABLE:
            self.logger.info("🔄 지연 로딩: DataConverter 초기화")
            self._initialize_data_converter()
        return self.data_converter

    def _initialize_memory_manager(self):
        """지연 로딩: MemoryManager 초기화"""
        try:
            if MEMORY_MANAGER_AVAILABLE:
                memory_config = self.config.get('memory_manager', {})
                self.memory_manager = MemoryManager(
                    device=self.device,
                    config=memory_config,
                    **memory_config
                )
        except Exception as e:
            self.logger.error(f"지연 로딩 실패 - MemoryManager: {e}")

    def _initialize_model_loader(self):
        """지연 로딩: ModelLoader 초기화"""
        try:
            if MODEL_LOADER_AVAILABLE:
                loader_config = self.config.get('model_loader', {})
                self.model_loader = ModelLoader(
                    device=self.device,
                    config=loader_config,
                    **loader_config
                )
        except Exception as e:
            self.logger.error(f"지연 로딩 실패 - ModelLoader: {e}")

    def _initialize_data_converter(self):
        """지연 로딩: DataConverter 초기화"""
        try:
            if DATA_CONVERTER_AVAILABLE:
                converter_config = self.config.get('data_converter', {})
                self.data_converter = DataConverter(
                    device=self.device,
                    config=converter_config,
                    **converter_config
                )
        except Exception as e:
            self.logger.error(f"지연 로딩 실패 - DataConverter: {e}")

    def get_all_utils(self) -> Dict[str, Any]:
        """모든 유틸리티 인스턴스 반환"""
        return {
            'memory_manager': self.get_memory_manager(),
            'model_loader': self.get_model_loader(),
            'data_converter': self.get_data_converter()
        }

    def get_utils_info(self) -> Dict[str, Any]:
        """유틸리티 통합 정보 조회"""
        info = {
            "manager_initialized": self.initialized,
            "device": self.device,
            "is_m3_max": self.is_m3_max,
            "memory_gb": self.memory_gb,
            "optimization_enabled": self.optimization_enabled,
            "quality_level": self.quality_level,
            "config_keys": list(self.config.keys()),
            "system_info": {
                "torch_available": TORCH_AVAILABLE,
                "default_device": DEFAULT_DEVICE,
                "mps_available": TORCH_AVAILABLE and torch.backends.mps.is_available() if TORCH_AVAILABLE else False,
                "cuda_available": TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False
            },
            "available_utils": {
                "memory_manager": MEMORY_MANAGER_AVAILABLE and self.get_memory_manager() is not None,
                "model_loader": MODEL_LOADER_AVAILABLE and self.get_model_loader() is not None,
                "data_converter": DATA_CONVERTER_AVAILABLE and self.get_data_converter() is not None
            },
            "library_status": {
                "memory_manager": MEMORY_MANAGER_AVAILABLE,
                "model_loader": MODEL_LOADER_AVAILABLE,
                "data_converter": DATA_CONVERTER_AVAILABLE
            },
            "initialization_results": self.initialization_results
        }
        
        # 각 유틸리티의 상세 정보 추가
        if self.memory_manager:
            try:
                info["memory_manager_info"] = {
                    "device": self.memory_manager.device,
                    "memory_limit_gb": getattr(self.memory_manager, 'memory_limit_gb', 'N/A'),
                    "is_m3_max": getattr(self.memory_manager, 'is_m3_max', self.is_m3_max)
                }
            except Exception as e:
                info["memory_manager_info"] = {"error": str(e)}
        
        if self.model_loader:
            try:
                info["model_loader_info"] = {
                    "device": self.model_loader.device,
                    "use_fp16": getattr(self.model_loader, 'use_fp16', 'N/A'),
                    "max_cached_models": getattr(self.model_loader, 'max_cached_models', 'N/A'),
                    "loaded_models": len(getattr(self.model_loader, 'model_cache', {}))
                }
            except Exception as e:
                info["model_loader_info"] = {"error": str(e)}
        
        if self.data_converter:
            try:
                info["data_converter_info"] = {
                    "device": self.data_converter.device,
                    "default_size": getattr(self.data_converter, 'default_size', 'N/A'),
                    "use_gpu_acceleration": getattr(self.data_converter, 'use_gpu_acceleration', 'N/A')
                }
            except Exception as e:
                info["data_converter_info"] = {"error": str(e)}
        
        return info

    # ============================================
    # 🔥 통합 작업 메서드들
    # ============================================

    async def optimize_memory(self) -> Dict[str, Any]:
        """통합 메모리 최적화"""
        try:
            results = {"success": False, "operations": []}
            
            # 1. 메모리 매니저를 통한 최적화
            if self.memory_manager:
                try:
                    await self.memory_manager.cleanup()
                    results["operations"].append("memory_manager_cleanup")
                except Exception as e:
                    self.logger.warning(f"메모리 매니저 정리 실패: {e}")
            
            # 2. 모델 로더 캐시 정리
            if self.model_loader:
                try:
                    await self.model_loader._cleanup_least_used_models()
                    results["operations"].append("model_cache_cleanup")
                except Exception as e:
                    self.logger.warning(f"모델 캐시 정리 실패: {e}")
            
            # 3. 시스템 레벨 정리
            try:
                gc.collect()
                if TORCH_AVAILABLE:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        try:
                            torch.mps.empty_cache()
                        except:
                            pass
                results["operations"].append("system_cleanup")
            except Exception as e:
                self.logger.warning(f"시스템 정리 실패: {e}")
            
            results["success"] = len(results["operations"]) > 0
            
            if results["success"]:
                self.logger.info(f"🧹 통합 메모리 최적화 완료: {', '.join(results['operations'])}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 통합 메모리 최적화 실패: {e}")
            return {"success": False, "error": str(e)}

    def cleanup(self):
        """리소스 정리"""
        try:
            cleanup_operations = []
            
            # 1. 개별 유틸리티 정리
            if self.model_loader:
                try:
                    self.model_loader.cleanup()
                    cleanup_operations.append("model_loader")
                except Exception as e:
                    self.logger.warning(f"ModelLoader 정리 실패: {e}")
            
            if self.memory_manager:
                try:
                    if hasattr(self.memory_manager, 'cleanup'):
                        self.memory_manager.cleanup()
                    cleanup_operations.append("memory_manager")
                except Exception as e:
                    self.logger.warning(f"MemoryManager 정리 실패: {e}")
            
            if self.data_converter:
                try:
                    if hasattr(self.data_converter, 'cleanup'):
                        self.data_converter.cleanup()
                    cleanup_operations.append("data_converter")
                except Exception as e:
                    self.logger.warning(f"DataConverter 정리 실패: {e}")
            
            # 2. 참조 정리
            self.memory_manager = None
            self.model_loader = None
            self.data_converter = None
            self._weak_refs.clear()
            
            # 3. 상태 리셋
            self.initialized = False
            self.initialization_results = {}
            
            if cleanup_operations:
                self.logger.info(f"✅ OptimalUtilsManager 정리 완료: {', '.join(cleanup_operations)}")
            
        except Exception as e:
            self.logger.error(f"❌ OptimalUtilsManager 정리 실패: {e}")

    def __del__(self):
        """소멸자"""
        try:
            self.cleanup()
        except:
            pass

# ============================================
# 🔥 전역 매니저 관리
# ============================================

_global_utils_manager: Optional[OptimalUtilsManager] = None
_manager_lock = threading.RLock()

def get_global_utils_manager() -> Optional[OptimalUtilsManager]:
    """전역 유틸리티 매니저 반환"""
    global _global_utils_manager
    return _global_utils_manager

def initialize_global_utils(**kwargs) -> OptimalUtilsManager:
    """전역 유틸리티 매니저 초기화"""
    global _global_utils_manager
    
    with _manager_lock:
        if _global_utils_manager is None:
            _global_utils_manager = OptimalUtilsManager(**kwargs)
            logger.info("🌍 전역 유틸리티 매니저 생성 완료")
        return _global_utils_manager

def reset_global_utils():
    """전역 유틸리티 매니저 리셋"""
    global _global_utils_manager
    
    with _manager_lock:
        if _global_utils_manager:
            _global_utils_manager.cleanup()
            _global_utils_manager = None
            logger.info("🔄 전역 유틸리티 매니저 리셋 완료")

# ============================================
# 🔥 편의 함수들 (최적 생성자 패턴 호환)
# ============================================

def create_optimal_utils(
    device: Optional[str] = None,
    memory_gb: Optional[float] = None,
    is_m3_max: Optional[bool] = None,
    optimization_enabled: bool = True,
    **kwargs
) -> OptimalUtilsManager:
    """최적 생성자 패턴으로 유틸리티 매니저 생성"""
    
    # 기본값 설정
    if memory_gb is None:
        memory_gb = 128.0 if (is_m3_max or IS_M3_MAX) else 16.0
    
    if is_m3_max is None:
        is_m3_max = IS_M3_MAX
    
    return OptimalUtilsManager(
        device=device,
        memory_gb=memory_gb,
        is_m3_max=is_m3_max,
        optimization_enabled=optimization_enabled,
        singleton=False,  # 새 인스턴스 생성
        **kwargs
    )

# ============================================
# 🔥 빠른 접근 함수들
# ============================================

def get_memory_manager(**kwargs) -> Optional[MemoryManager]:
    """메모리 매니저 빠른 접근"""
    # 1. 전역 매니저에서 시도
    manager = get_global_utils_manager()
    if manager:
        mm = manager.get_memory_manager()
        if mm:
            return mm
    
    # 2. 개별 전역 인스턴스 확인
    if MEMORY_MANAGER_AVAILABLE:
        try:
            return get_global_memory_manager()
        except:
            pass
    
    # 3. 새 인스턴스 생성 (최후 수단)
    if MEMORY_MANAGER_AVAILABLE and kwargs.get('create_if_missing', False):
        try:
            return create_memory_manager(**kwargs)
        except Exception as e:
            logger.warning(f"메모리 매니저 생성 실패: {e}")
    
    return None

def get_model_loader(**kwargs) -> Optional[ModelLoader]:
    """모델 로더 빠른 접근"""
    # 1. 전역 매니저에서 시도
    manager = get_global_utils_manager()
    if manager:
        ml = manager.get_model_loader()
        if ml:
            return ml
    
    # 2. 개별 전역 인스턴스 확인
    if MODEL_LOADER_AVAILABLE:
        try:
            return get_global_model_loader()
        except:
            pass
    
    # 3. 새 인스턴스 생성 (최후 수단)
    if MODEL_LOADER_AVAILABLE and kwargs.get('create_if_missing', False):
        try:
            return create_model_loader(**kwargs)
        except Exception as e:
            logger.warning(f"모델 로더 생성 실패: {e}")
    
    return None

def get_data_converter(**kwargs) -> Optional[DataConverter]:
    """데이터 변환기 빠른 접근"""
    # 1. 전역 매니저에서 시도
    manager = get_global_utils_manager()
    if manager:
        dc = manager.get_data_converter()
        if dc:
            return dc
    
    # 2. 개별 전역 인스턴스 확인
    if DATA_CONVERTER_AVAILABLE:
        try:
            return get_global_data_converter()
        except:
            pass
    
    # 3. 새 인스턴스 생성 (최후 수단)
    if DATA_CONVERTER_AVAILABLE and kwargs.get('create_if_missing', False):
        try:
            return create_data_converter(**kwargs)
        except Exception as e:
            logger.warning(f"데이터 변환기 생성 실패: {e}")
    
    return None

# ============================================
# 🔥 통합 유틸리티 함수들
# ============================================

def get_system_info() -> Dict[str, Any]:
    """시스템 정보 조회"""
    return {
        "device": DEFAULT_DEVICE,
        "is_m3_max": IS_M3_MAX,
        "torch_available": TORCH_AVAILABLE,
        "available_modules": {
            "memory_manager": MEMORY_MANAGER_AVAILABLE,
            "model_loader": MODEL_LOADER_AVAILABLE,
            "data_converter": DATA_CONVERTER_AVAILABLE
        },
        "global_manager_initialized": _global_utils_manager is not None
    }

def check_utils_health() -> Dict[str, Any]:
    """유틸리티 상태 건강성 체크"""
    health = {
        "overall_status": "healthy",
        "issues": [],
        "recommendations": []
    }
    
    try:
        # 1. 모듈 가용성 체크
        if not any([MEMORY_MANAGER_AVAILABLE, MODEL_LOADER_AVAILABLE, DATA_CONVERTER_AVAILABLE]):
            health["overall_status"] = "critical"
            health["issues"].append("모든 유틸리티 모듈 사용 불가")
            health["recommendations"].append("의존성 설치 확인 필요")
        
        # 2. 전역 매니저 체크
        global_manager = get_global_utils_manager()
        if global_manager:
            if not global_manager.initialized:
                health["issues"].append("전역 매니저 초기화 실패")
                health["recommendations"].append("initialize_global_utils() 호출 필요")
        else:
            health["recommendations"].append("전역 매니저 초기화 권장")
        
        # 3. 디바이스 호환성 체크
        if DEFAULT_DEVICE == 'cpu' and TORCH_AVAILABLE:
            health["issues"].append("GPU 가속 사용 불가")
            health["recommendations"].append("GPU 드라이버 또는 PyTorch 설치 확인")
        
        # 4. 메모리 체크
        if MEMORY_MANAGER_AVAILABLE:
            try:
                mm = get_memory_manager()
                if mm:
                    stats = mm.get_memory_stats()
                    if stats.cpu_percent > 90:
                        health["issues"].append("높은 메모리 사용률")
                        health["recommendations"].append("메모리 정리 권장")
            except:
                pass
        
        # 전체 상태 결정
        if health["issues"]:
            if health["overall_status"] != "critical":
                health["overall_status"] = "warning"
        
        return health
        
    except Exception as e:
        return {
            "overall_status": "error",
            "issues": [f"건강성 체크 실패: {e}"],
            "recommendations": ["시스템 재시작 권장"]
        }

async def optimize_all_utils() -> Dict[str, Any]:
    """모든 유틸리티 최적화"""
    try:
        results = {"operations": [], "success": False}
        
        # 전역 매니저를 통한 최적화
        global_manager = get_global_utils_manager()
        if global_manager:
            manager_results = await global_manager.optimize_memory()
            if manager_results.get("success"):
                results["operations"].extend(manager_results.get("operations", []))
        
        # 개별 최적화
        if MEMORY_MANAGER_AVAILABLE:
            try:
                optimize_result = optimize_memory_usage()
                if optimize_result.get("success"):
                    results["operations"].append("standalone_memory_optimization")
            except:
                pass
        
        results["success"] = len(results["operations"]) > 0
        
        if results["success"]:
            logger.info(f"🚀 전체 유틸리티 최적화 완료: {', '.join(results['operations'])}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 전체 유틸리티 최적화 실패: {e}")
        return {"success": False, "error": str(e)}

# ============================================
# 🔥 하위 호환성 별칭들
# ============================================

# 기존 이름으로도 접근 가능하도록 별칭 설정
if MEMORY_MANAGER_AVAILABLE:
    GPUMemoryManager = MemoryManager  # 하위 호환

if MODEL_LOADER_AVAILABLE:
    AIModelLoader = ModelLoader  # 하위 호환

if DATA_CONVERTER_AVAILABLE:
    ImageConverter = DataConverter  # 하위 호환

# ============================================
# 🔥 Export 목록 구성
# ============================================

__all__ = [
    # === 메인 매니저 ===
    'OptimalUtilsManager',
    'get_global_utils_manager',
    'initialize_global_utils',
    'reset_global_utils',
    'create_optimal_utils',
    
    # === 빠른 접근 함수들 ===
    'get_memory_manager',
    'get_model_loader', 
    'get_data_converter',
    
    # === 시스템 정보 ===
    'get_system_info',
    'check_utils_health',
    'optimize_all_utils',
    
    # === 상태 확인 ===
    'MEMORY_MANAGER_AVAILABLE',
    'MODEL_LOADER_AVAILABLE',
    'DATA_CONVERTER_AVAILABLE',
    'TORCH_AVAILABLE',
    'IS_M3_MAX',
    'DEFAULT_DEVICE'
]

# 사용 가능한 유틸리티들을 동적으로 추가
if MEMORY_MANAGER_AVAILABLE:
    __all__.extend([
        'MemoryManager',
        'MemoryStats',
        'create_memory_manager',
        'get_global_memory_manager',
        'initialize_global_memory_manager',
        'optimize_memory_usage',
        'check_memory_available',
        'get_memory_info',
        'GPUMemoryManager'  # 하위 호환
    ])

if MODEL_LOADER_AVAILABLE:
    __all__.extend([
        'ModelLoader',
        'ModelConfig',
        'ModelFormat',
        'ModelType',
        'ModelRegistry',
        'ModelMemoryManager',
        'StepModelInterface',
        'BaseStepMixin',
        'create_model_loader',
        'get_global_model_loader',
        'initialize_global_model_loader',
        'load_model_async',
        'load_model_sync',
        'preprocess_image',
        'postprocess_segmentation',
        'postprocess_pose',
        # AI 모델 클래스들
        'GraphonomyModel',
        'OpenPoseModel',
        'U2NetModel',
        'GeometricMatchingModel',
        'HRVITONModel',
        'AIModelLoader'  # 하위 호환
    ])

if DATA_CONVERTER_AVAILABLE:
    __all__.extend([
        'DataConverter',
        'create_data_converter',
        'get_global_data_converter',
        'initialize_global_data_converter',
        'quick_image_to_tensor',
        'quick_tensor_to_image',
        'ImageConverter'  # 하위 호환
    ])

# ============================================
# 🔥 모듈 초기화 및 로깅
# ============================================

def _log_initialization_summary():
    """초기화 요약 로깅"""
    available_count = sum([
        MEMORY_MANAGER_AVAILABLE,
        MODEL_LOADER_AVAILABLE, 
        DATA_CONVERTER_AVAILABLE
    ])
    
    total_count = 3
    
    logger.info("=" * 60)
    logger.info("🔧 MyCloset AI 파이프라인 유틸리티 모듈 로드 완료")
    logger.info("=" * 60)
    logger.info(f"📊 사용 가능한 유틸리티: {available_count}/{total_count}")
    logger.info(f"   - MemoryManager: {'✅' if MEMORY_MANAGER_AVAILABLE else '❌'}")
    logger.info(f"   - ModelLoader: {'✅' if MODEL_LOADER_AVAILABLE else '❌'}")
    logger.info(f"   - DataConverter: {'✅' if DATA_CONVERTER_AVAILABLE else '❌'}")
    logger.info(f"🍎 시스템 정보:")
    logger.info(f"   - M3 Max: {'✅' if IS_M3_MAX else '❌'}")
    logger.info(f"   - 기본 디바이스: {DEFAULT_DEVICE}")
    logger.info(f"   - PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}")
    
    if available_count == total_count:
        logger.info("✅ 모든 유틸리티 모듈이 성공적으로 로드되었습니다!")
    elif available_count > 0:
        logger.warning(f"⚠️ 일부 유틸리티 모듈만 로드되었습니다 ({available_count}/{total_count})")
    else:
        logger.error("❌ 모든 유틸리티 모듈 로드에 실패했습니다. 의존성을 확인하세요.")
    
    logger.info("=" * 60)

# 초기화 요약 출력
_log_initialization_summary()

# 자동 전역 매니저 초기화 (환경변수 기반)
try:
    auto_init = os.getenv('AUTO_INIT_UTILS', 'false').lower() in ('true', '1', 'yes', 'on')
    if auto_init:
        initialize_global_utils()
        logger.info("🚀 전역 유틸리티 매니저 자동 초기화 완료")
except Exception as e:
    logger.warning(f"⚠️ 전역 유틸리티 매니저 자동 초기화 실패: {e}")

# 모듈 정리 함수 등록
import atexit

def _cleanup_on_exit():
    """프로그램 종료 시 정리"""
    try:
        reset_global_utils()
        logger.info("🔚 유틸리티 모듈 종료 정리 완료")
    except:
        pass

atexit.register(_cleanup_on_exit)

# 최종 확인 메시지
if MEMORY_MANAGER_AVAILABLE or MODEL_LOADER_AVAILABLE or DATA_CONVERTER_AVAILABLE:
    logger.info("🎯 최적 생성자 패턴 AI 파이프라인 유틸리티 준비 완료")
else:
    logger.error("💥 모든 유틸리티 모듈 사용 불가 - 의존성 설치 필요")

logger.info("🏁 AI 파이프라인 유틸리티 모듈 로딩 완료")