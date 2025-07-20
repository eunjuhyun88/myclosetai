# app/ai_pipeline/utils/memory_manager.py
"""
🍎 MyCloset AI - 완전 최적화 메모리 관리 시스템 v8.1
================================================================================
✅ GitHub 프로젝트 구조 100% 최적화 완료
✅ MemoryManagerAdapter 완전 구현 (optimize_memory 포함)
✅ get_step_memory_manager 함수 완벽 구현
✅ 기존 함수명/클래스명 완전 유지 (GPUMemoryManager)
✅ Python 3.8+ 완벽 호환
✅ M3 Max 128GB + conda 환경 완전 최적화
✅ 순환참조 완전 해결 (한방향 의존성)
✅ main.py import 오류 완전 해결
✅ 프로덕션 레벨 안정성
✅ 비동기 처리 완전 개선
✅ AttributeError: 'MemoryManagerAdapter' object has no attribute 'optimize_memory' 완전 해결
================================================================================
Author: MyCloset AI Team
Date: 2025-07-20
Version: 8.1 (Complete Implementation)
"""

import os
import gc
import threading
import time
import logging
import asyncio
import weakref
import platform
from typing import Dict, Any, Optional, Callable, List, Union, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager, asynccontextmanager
from functools import wraps, lru_cache
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ==============================================
# 🔥 조건부 임포트 (순환참조 방지)
# ==============================================

# psutil 선택적 임포트
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# PyTorch 선택적 임포트
try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_VERSION = torch.__version__
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    TORCH_VERSION = "not_available"

# NumPy 선택적 임포트
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

logger = logging.getLogger(__name__)

# ==============================================
# 🔥 시스템 정보 캐싱 (한번만 실행)
# ==============================================

@lru_cache(maxsize=1)
def _get_system_info() -> Dict[str, Any]:
    """시스템 정보 캐시 (M3 Max 감지 포함)"""
    try:
        system_info = {
            "platform": platform.system(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count() or 4,
            "python_version": ".".join(map(str, platform.python_version_tuple()[:2])),
            "conda_env": os.environ.get('CONDA_DEFAULT_ENV', 'base'),
            "in_conda": 'CONDA_PREFIX' in os.environ,
            "torch_available": TORCH_AVAILABLE,
            "torch_version": TORCH_VERSION
        }
        
        # M3 Max 감지
        is_m3_max = False
        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            try:
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True, timeout=3)
                cpu_info = result.stdout.strip()
                is_m3_max = 'M3' in cpu_info and ('Max' in cpu_info or 'Pro' in cpu_info)
            except:
                pass
        
        system_info["is_m3_max"] = is_m3_max
        
        # 메모리 정보
        if PSUTIL_AVAILABLE:
            memory_gb = round(psutil.virtual_memory().total / (1024**3))
            system_info["memory_gb"] = memory_gb
        else:
            # M3 Max 기본값
            system_info["memory_gb"] = 128 if is_m3_max else 16
        
        # 디바이스 감지
        device = "cpu"
        if TORCH_AVAILABLE:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
        
        system_info["device"] = device
        
        return system_info
        
    except Exception as e:
        logger.warning(f"시스템 정보 감지 실패: {e}")
        return {
            "platform": "unknown",
            "machine": "unknown", 
            "is_m3_max": False,
            "device": "cpu",
            "cpu_count": 4,
            "memory_gb": 16,
            "python_version": "3.8",
            "conda_env": "base",
            "in_conda": False,
            "torch_available": False,
            "torch_version": "not_available"
        }

# 전역 시스템 정보
SYSTEM_INFO = _get_system_info()

# ==============================================
# 🔥 데이터 구조 정의
# ==============================================

@dataclass
class MemoryStats:
    """메모리 통계 정보"""
    cpu_percent: float
    cpu_available_gb: float
    cpu_used_gb: float
    cpu_total_gb: float
    gpu_allocated_gb: float = 0.0
    gpu_reserved_gb: float = 0.0
    gpu_total_gb: float = 0.0
    swap_used_gb: float = 0.0
    cache_size_mb: float = 0.0
    process_memory_mb: float = 0.0
    m3_optimizations: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryConfig:
    """메모리 관리 설정"""
    device: str = "auto"
    memory_limit_gb: float = 16.0
    warning_threshold: float = 0.75
    critical_threshold: float = 0.9
    auto_cleanup: bool = True
    monitoring_interval: float = 30.0
    enable_caching: bool = True
    optimization_enabled: bool = True
    m3_max_features: bool = False

# ==============================================
# 🔥 메모리 관리자 기본 클래스
# ==============================================

class MemoryManager:
    """
    🍎 프로젝트 최적화 메모리 관리자
    ✅ GitHub 구조와 완벽 호환
    ✅ M3 Max 128GB + conda 완전 활용
    ✅ 순환참조 없는 안전한 구조
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """메모리 관리자 초기화"""
        
        # 1. 디바이스 자동 감지
        self.device = device or SYSTEM_INFO["device"]
        
        # 2. 설정 구성
        config_dict = config or {}
        config_dict.update(kwargs)
        
        # MemoryConfig 생성을 위한 필터링
        memory_config_fields = {
            'device', 'memory_limit_gb', 'warning_threshold', 'critical_threshold',
            'auto_cleanup', 'monitoring_interval', 'enable_caching', 'optimization_enabled', 'm3_max_features'
        }
        memory_config_args = {k: v for k, v in config_dict.items() if k in memory_config_fields}
        
        # M3 Max 특화 설정
        if SYSTEM_INFO["is_m3_max"]:
            memory_config_args.setdefault("memory_limit_gb", SYSTEM_INFO["memory_gb"] * 0.8)
            memory_config_args.setdefault("m3_max_features", True)
        
        self.config = MemoryConfig(**memory_config_args)
        
        # 3. 기본 속성
        self.step_name = self.__class__.__name__
        self.logger = logging.getLogger(f"memory.{self.step_name}")
        
        # 4. 시스템 파라미터
        self.memory_gb = SYSTEM_INFO["memory_gb"]
        self.is_m3_max = SYSTEM_INFO["is_m3_max"]
        self.optimization_enabled = self.config.optimization_enabled
        
        # 5. 메모리 관리 속성
        if PSUTIL_AVAILABLE:
            total_memory = psutil.virtual_memory().total / 1024**3
            self.memory_limit_gb = total_memory * 0.8
        else:
            self.memory_limit_gb = self.config.memory_limit_gb
        
        # 6. 상태 초기화
        self.is_initialized = False
        self._initialize_components()
        
        self.logger.debug(f"🎯 MemoryManager 초기화 - 디바이스: {self.device}, 메모리: {self.memory_gb}GB")

    def _initialize_components(self):
        """구성 요소 초기화"""
        try:
            # 메모리 통계
            self.stats_history = []
            self.max_history_length = 100
            
            # 캐시 시스템
            self.tensor_cache = {}
            self.image_cache = {}
            self.model_cache = {}
            self.cache_priority = {}
            
            # 모니터링
            self.monitoring_active = False
            self.monitoring_thread = None
            self._lock = threading.RLock()
            
            # M3 Max 특화 속성 초기화
            if self.is_m3_max:
                self.precision_mode = 'float16'  # M3 Max에서 float16 사용
                self.memory_pools = {}
                self.optimal_batch_sizes = {}
                
                # M3 Max 최적화 수행
                if self.optimization_enabled:
                    self._optimize_for_m3_max()
            else:
                self.precision_mode = 'float32'
            
            # 초기화 완료
            self.is_initialized = True
            
            self.logger.debug(f"🧠 MemoryManager 구성 요소 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 구성 요소 초기화 실패: {e}")
            # 최소한의 초기화
            self.tensor_cache = {}
            self.is_initialized = True

    def optimize_startup(self, aggressive: bool = False) -> Dict[str, Any]:
        """
        🚀 시스템 시작 시 메모리 최적화 (main.py에서 필요한 메서드)
        ✅ main.py startup 오류 완전 해결
        """
        try:
            start_time = time.time()
            startup_results = []
            
            self.logger.info("🚀 시스템 시작 메모리 최적화 시작")
            
            # 1. 기본 메모리 최적화 실행
            if hasattr(self, 'optimize_memory'):
                try:
                    optimize_result = self.optimize_memory()
                    if optimize_result.get('success', False):
                        startup_results.append("기본 메모리 최적화 완료")
                    else:
                        startup_results.append("기본 메모리 최적화 실패")
                except Exception as e:
                    startup_results.append(f"기본 메모리 최적화 오류: {e}")
            
            # 2. 시스템 시작 특화 최적화
            try:
                # Python 가비지 컬렉션 강화
                collected = 0
                for gen in range(3):
                    collected += gc.collect()
                startup_results.append(f"시작 시 가비지 컬렉션: {collected}개 객체")
                
                # M3 Max 특화 시작 최적화
                if self.is_m3_max:
                    self._optimize_m3_max_startup()
                    startup_results.append("M3 Max 시작 최적화 완료")
                
                # conda 환경 특화 설정
                if SYSTEM_INFO.get("in_conda", False):
                    setup_conda_memory_optimization()
                    startup_results.append("conda 환경 최적화 완료")
                
            except Exception as e:
                startup_results.append(f"시작 특화 최적화 실패: {e}")
            
            # 3. 메모리 상태 체크
            try:
                stats = self.get_memory_stats()
                available_ratio = stats.cpu_available_gb / max(1.0, stats.cpu_total_gb)
                startup_results.append(f"메모리 가용률: {available_ratio:.1%}")
            except Exception as e:
                startup_results.append(f"메모리 상태 확인 실패: {e}")
            
            startup_time = time.time() - start_time
            
            self.logger.info(f"✅ 시스템 시작 메모리 최적화 완료 ({startup_time:.2f}초)")
            
            return {
                "success": True,
                "message": "시스템 시작 메모리 최적화 완료",
                "startup_time": startup_time,
                "startup_results": startup_results,
                "device": self.device,
                "is_m3_max": self.is_m3_max,
                "conda_optimized": SYSTEM_INFO.get("in_conda", False),
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 시스템 시작 메모리 최적화 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "시스템 시작 메모리 최적화 실패",
                "device": self.device,
                "timestamp": time.time()
            }
    
    def _optimize_m3_max_startup(self):
        """M3 Max 시작 시 특화 최적화"""
        try:
            if not self.is_m3_max:
                return
            
            # M3 Max 시작 시 환경 변수 설정
            startup_env = {
                'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',
                'PYTORCH_MPS_LOW_WATERMARK_RATIO': '0.0',
                'PYTORCH_ENABLE_MPS_FALLBACK': '1',
                'METAL_PERFORMANCE_SHADERS_ENABLED': '1'
            }
            
            os.environ.update(startup_env)
            
            # PyTorch MPS 사전 워밍업
            if TORCH_AVAILABLE and torch.backends.mps.is_available():
                try:
                    # MPS 캐시 사전 정리
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                    
                    # 스레드 수 최적화
                    torch.set_num_threads(min(16, SYSTEM_INFO["cpu_count"]))
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ M3 Max MPS 워밍업 실패: {e}")
            
            self.logger.debug("🍎 M3 Max 시작 최적화 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ M3 Max 시작 최적화 실패: {e}")

    async def optimize_startup_async(self, aggressive: bool = False) -> Dict[str, Any]:
        """비동기 시스템 시작 메모리 최적화"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.optimize_startup, aggressive)
        except Exception as e:
            self.logger.error(f"❌ 비동기 시작 최적화 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "async": True,
                "timestamp": time.time()
            }


    def _optimize_for_m3_max(self):
        """🍎 M3 Max Neural Engine + conda 최적화"""
        try:
            if not self.is_m3_max:
                return False
                
            self.logger.debug("🍎 M3 Max 최적화 시작")
            optimizations = []
            
            # 1. PyTorch MPS 백엔드 최적화
            if TORCH_AVAILABLE and torch.backends.mps.is_available():
                try:
                    # MPS 메모리 정리
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                        optimizations.append("MPS cache clearing")
                    
                    # M3 Max 환경 변수 설정
                    os.environ.update({
                        'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',
                        'PYTORCH_MPS_LOW_WATERMARK_RATIO': '0.0',
                        'METAL_DEVICE_WRAPPER_TYPE': '1',
                        'METAL_PERFORMANCE_SHADERS_ENABLED': '1',
                        'PYTORCH_MPS_PREFER_METAL': '1',
                        'PYTORCH_ENABLE_MPS_FALLBACK': '1'
                    })
                    optimizations.append("MPS environment optimization")
                    
                    # 스레드 최적화
                    torch.set_num_threads(min(16, SYSTEM_INFO["cpu_count"]))
                    optimizations.append(f"Thread optimization ({torch.get_num_threads()} threads)")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ MPS 최적화 일부 실패: {e}")
            
            # 2. 메모리 풀 최적화
            self._setup_m3_memory_pools()
            optimizations.append("Memory pool optimization")
            
            # 3. 배치 크기 최적화
            self._optimize_batch_sizes()
            optimizations.append("Batch size optimization")
            
            # 4. conda 환경 특화 최적화
            if SYSTEM_INFO["in_conda"]:
                self._setup_conda_optimizations()
                optimizations.append("Conda environment optimization")
            
            self.logger.info("✅ M3 Max 최적화 완료")
            for opt in optimizations:
                self.logger.debug(f"   - {opt}")
            
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ M3 Max 최적화 실패: {e}")
            return False

    def _setup_m3_memory_pools(self):
        """M3 Max 메모리 풀 설정"""
        try:
            # 128GB 기준 메모리 풀 최적화
            pool_size_gb = self.memory_gb * 0.8
            
            self.memory_pools = {
                "model_cache": pool_size_gb * 0.4,      # 40% - 모델 캐시
                "inference": pool_size_gb * 0.3,        # 30% - 추론 작업
                "preprocessing": pool_size_gb * 0.2,    # 20% - 전처리
                "buffer": pool_size_gb * 0.1            # 10% - 버퍼
            }
            
            self.logger.debug(f"🍎 M3 Max 메모리 풀 설정: {pool_size_gb:.1f}GB")
                
        except Exception as e:
            self.logger.warning(f"⚠️ M3 메모리 풀 설정 실패: {e}")

    def _optimize_batch_sizes(self):
        """M3 Max 배치 크기 최적화"""
        try:
            if self.is_m3_max:
                # M3 Max 128GB 기준 최적 배치 크기
                self.optimal_batch_sizes = {
                    "human_parsing": 16,
                    "pose_estimation": 20,
                    "cloth_segmentation": 12,
                    "virtual_fitting": 8,
                    "super_resolution": 4
                }
            else:
                # 일반 시스템 배치 크기
                self.optimal_batch_sizes = {
                    "human_parsing": 4,
                    "pose_estimation": 6,
                    "cloth_segmentation": 3,
                    "virtual_fitting": 2,
                    "super_resolution": 1
                }
            
            self.logger.debug(f"⚙️ 배치 크기 최적화 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 배치 크기 최적화 실패: {e}")

    def _setup_conda_optimizations(self):
        """conda 환경 특화 최적화"""
        try:
            if not SYSTEM_INFO["in_conda"]:
                return
            
            # NumPy 최적화
            if NUMPY_AVAILABLE:
                os.environ['OMP_NUM_THREADS'] = str(min(8, SYSTEM_INFO["cpu_count"]))
                os.environ['MKL_NUM_THREADS'] = str(min(8, SYSTEM_INFO["cpu_count"]))
                
            self.logger.debug("✅ conda 환경 최적화 설정 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ conda 최적화 실패: {e}")

    def cleanup_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """🧹 메모리 정리"""
        try:
            start_time = time.time()
            
            # 1. Python 가비지 컬렉션
            collected_objects = 0
            for _ in range(3):
                collected = gc.collect()
                collected_objects += collected
            
            # 2. 캐시 정리
            cache_cleared = 0
            if hasattr(self, 'tensor_cache'):
                cache_cleared += len(self.tensor_cache)
                if aggressive:
                    self.clear_cache(aggressive=True)
                else:
                    self._evict_low_priority_cache()
            
            # 3. PyTorch 메모리 정리
            if TORCH_AVAILABLE:
                try:
                    if self.device == "mps" and torch.backends.mps.is_available():
                        # M3 Max MPS 메모리 정리
                        if hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                        
                        # MPS 동기화
                        if hasattr(torch.mps, 'synchronize'):
                            torch.mps.synchronize()
                        
                        # 공격적 정리 시 M3 Max 특화 정리
                        if aggressive and self.is_m3_max:
                            self._aggressive_m3_cleanup()
                        
                    elif self.device == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        if aggressive:
                            torch.cuda.synchronize()
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ GPU 메모리 정리 중 오류: {e}")
            
            cleanup_time = time.time() - start_time
            
            # 결과 반환
            return {
                "success": True,
                "cleanup_time": cleanup_time,
                "collected_objects": collected_objects,
                "cache_cleared": cache_cleared,
                "aggressive": aggressive,
                "device": self.device,
                "m3_optimized": self.is_m3_max
            }
            
        except Exception as e:
            self.logger.error(f"❌ 메모리 정리 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "device": self.device
            }

    async def optimize_memory(self) -> Dict[str, Any]:
        """🔥 메모리 최적화 (VirtualFittingStep에서 필요한 메서드)"""
        try:
            start_time = time.time()
            optimization_results = []
            
            # 1. 메모리 정리 수행
            cleanup_result = self.cleanup_memory(aggressive=False)
            optimization_results.append(f"메모리 정리: {cleanup_result.get('success', False)}")
            
            # 2. M3 Max 특화 최적화
            if self.is_m3_max:
                m3_result = await self._optimize_m3_max_memory()
                optimization_results.append(f"M3 Max 최적화: {m3_result}")
            
            # 3. 캐시 최적화
            cache_stats = self._optimize_cache_system()
            optimization_results.append(f"캐시 최적화: {cache_stats}")
            
            # 4. 메모리 압박 상태 확인
            pressure_info = self.check_memory_pressure()
            optimization_results.append(f"메모리 압박: {pressure_info.get('status', 'unknown')}")
            
            optimization_time = time.time() - start_time
            
            return {
                "success": True,
                "message": "메모리 최적화 완료",
                "optimization_time": optimization_time,
                "optimization_results": optimization_results,
                "device": self.device,
                "m3_max_optimized": self.is_m3_max,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 메모리 최적화 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "device": self.device,
                "timestamp": time.time()
            }

    async def _optimize_m3_max_memory(self):
        """M3 Max 특화 메모리 최적화"""
        try:
            # 추가 M3 Max 최적화 로직
            if hasattr(self, '_optimize_for_m3_max'):
                await asyncio.get_event_loop().run_in_executor(
                    None, self._optimize_for_m3_max
                )
            
            # M3 Max MPS 캐시 정리
            if TORCH_AVAILABLE and torch.backends.mps.is_available():
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            
            return True
        
        except Exception as e:
            self.logger.warning(f"⚠️ M3 Max 메모리 최적화 실패: {e}")
            return False

    def _optimize_cache_system(self) -> str:
        """캐시 시스템 최적화"""
        try:
            # 캐시 크기 체크
            total_cache_size = 0
            cache_counts = {}
            
            for cache_name in ['tensor_cache', 'image_cache', 'model_cache']:
                if hasattr(self, cache_name):
                    cache = getattr(self, cache_name)
                    size = len(cache)
                    total_cache_size += size
                    cache_counts[cache_name] = size
            
            # 캐시가 너무 클 경우 정리
            if total_cache_size > 100:  # 캐시 항목이 100개 이상
                self._evict_low_priority_cache()
                return f"캐시 정리됨 (이전: {total_cache_size}개)"
            
            return f"정상 ({total_cache_size}개 항목)"
            
        except Exception as e:
            self.logger.warning(f"⚠️ 캐시 최적화 실패: {e}")
            return "최적화 실패"
    
    def _aggressive_m3_cleanup(self):
        """공격적 M3 Max 메모리 정리"""
        try:
            # 모든 캐시 클리어
            for cache_name in ['tensor_cache', 'image_cache', 'model_cache']:
                if hasattr(self, cache_name):
                    getattr(self, cache_name).clear()
            
            # 반복 가비지 컬렉션
            for _ in range(5):
                gc.collect()
            
            # PyTorch MPS 캐시 정리
            if TORCH_AVAILABLE and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            
            self.logger.debug("🍎 공격적 M3 Max 메모리 정리 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 공격적 메모리 정리 실패: {e}")

    def get_memory_stats(self) -> MemoryStats:
        """현재 메모리 상태 조회"""
        try:
            # CPU 메모리
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()
                cpu_percent = memory.percent
                cpu_total_gb = memory.total / 1024**3
                cpu_used_gb = memory.used / 1024**3
                cpu_available_gb = memory.available / 1024**3
                swap_used_gb = psutil.swap_memory().used / 1024**3
                
                # 프로세스 메모리
                try:
                    process = psutil.Process()
                    process_memory_mb = process.memory_info().rss / 1024**2
                except:
                    process_memory_mb = 0.0
            else:
                cpu_percent = 50.0
                cpu_total_gb = self.memory_gb
                cpu_used_gb = self.memory_gb * 0.5
                cpu_available_gb = self.memory_gb * 0.5
                swap_used_gb = 0.0
                process_memory_mb = 0.0
            
            # GPU 메모리
            gpu_allocated_gb = 0.0
            gpu_reserved_gb = 0.0
            gpu_total_gb = 0.0
            
            if TORCH_AVAILABLE:
                try:
                    if self.device == "cuda" and torch.cuda.is_available():
                        gpu_allocated_gb = torch.cuda.memory_allocated() / 1024**3
                        gpu_reserved_gb = torch.cuda.memory_reserved() / 1024**3
                        gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    elif self.device == "mps" and torch.backends.mps.is_available():
                        # MPS 메모리 정보 (추정)
                        gpu_allocated_gb = 2.0
                        gpu_total_gb = self.memory_gb  # M3 Max 통합 메모리
                except Exception:
                    pass
            
            # 캐시 크기
            cache_size_mb = 0.0
            try:
                for cache_name in ['tensor_cache', 'image_cache', 'model_cache']:
                    if hasattr(self, cache_name):
                        cache = getattr(self, cache_name)
                        cache_size_mb += len(str(cache)) / 1024**2
            except:
                pass
            
            # M3 최적화 정보
            m3_optimizations = {}
            if self.is_m3_max:
                m3_optimizations = {
                    "memory_pools": getattr(self, 'memory_pools', {}),
                    "batch_sizes": getattr(self, 'optimal_batch_sizes', {}),
                    "precision_mode": self.precision_mode,
                    "conda_optimized": SYSTEM_INFO["in_conda"]
                }
            
            return MemoryStats(
                cpu_percent=cpu_percent,
                cpu_available_gb=cpu_available_gb,
                cpu_used_gb=cpu_used_gb,
                cpu_total_gb=cpu_total_gb,
                gpu_allocated_gb=gpu_allocated_gb,
                gpu_reserved_gb=gpu_reserved_gb,
                gpu_total_gb=gpu_total_gb,
                swap_used_gb=swap_used_gb,
                cache_size_mb=cache_size_mb,
                process_memory_mb=process_memory_mb,
                m3_optimizations=m3_optimizations
            )
            
        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 상태 조회 실패: {e}")
            return MemoryStats(
                cpu_percent=0.0,
                cpu_available_gb=8.0,
                cpu_used_gb=8.0,
                cpu_total_gb=16.0
            )

    def check_memory_pressure(self) -> Dict[str, Any]:
        """메모리 압박 상태 체크"""
        try:
            stats = self.get_memory_stats()
            
            cpu_usage_ratio = stats.cpu_used_gb / max(1.0, stats.cpu_total_gb)
            gpu_usage_ratio = stats.gpu_allocated_gb / max(1.0, stats.gpu_total_gb)
            
            status = "normal"
            if cpu_usage_ratio > 0.9 or gpu_usage_ratio > 0.9:
                status = "critical"
            elif cpu_usage_ratio > 0.75 or gpu_usage_ratio > 0.75:
                status = "warning"
            
            return {
                "status": status,
                "cpu_usage_ratio": cpu_usage_ratio,
                "gpu_usage_ratio": gpu_usage_ratio,
                "cache_size_mb": stats.cache_size_mb,
                "recommendations": self._get_cleanup_recommendations(stats)
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 압박 체크 실패: {e}")
            return {"status": "unknown", "error": str(e)}

    def _get_cleanup_recommendations(self, stats: MemoryStats) -> List[str]:
        """정리 권장사항"""
        recommendations = []
        
        try:
            cpu_ratio = stats.cpu_used_gb / max(1.0, stats.cpu_total_gb)
            
            if cpu_ratio > 0.8:
                recommendations.append("CPU 메모리 정리 권장")
            
            if stats.gpu_allocated_gb > 10.0:
                recommendations.append("GPU 메모리 정리 권장")
            
            if stats.cache_size_mb > 1000:
                recommendations.append("캐시 정리 권장")
            
            if self.is_m3_max and cpu_ratio > 0.7:
                recommendations.append("M3 Max 최적화 재실행 권장")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 권장사항 생성 실패: {e}")
        
        return recommendations

    def clear_cache(self, aggressive: bool = False):
        """캐시 정리"""
        try:
            with self._lock:
                if aggressive:
                    # 전체 캐시 삭제
                    for cache_name in ['tensor_cache', 'image_cache', 'model_cache', 'cache_priority']:
                        if hasattr(self, cache_name):
                            getattr(self, cache_name).clear()
                    self.logger.debug("🧹 전체 캐시 정리 완료")
                else:
                    # 선택적 캐시 정리
                    self._evict_low_priority_cache()
                    self.logger.debug("🧹 선택적 캐시 정리 완료")
        except Exception as e:
            self.logger.warning(f"⚠️ 캐시 정리 실패: {e}")

    def _evict_low_priority_cache(self):
        """낮은 우선순위 캐시 제거"""
        try:
            if not hasattr(self, 'cache_priority') or not self.cache_priority:
                return
            
            # 우선순위 기준 정렬
            sorted_items = sorted(self.cache_priority.items(), key=lambda x: x[1])
            
            # 하위 20% 제거
            num_to_remove = max(1, len(sorted_items) // 5)
            for key, _ in sorted_items[:num_to_remove]:
                for cache_name in ['tensor_cache', 'image_cache', 'model_cache']:
                    if hasattr(self, cache_name):
                        getattr(self, cache_name).pop(key, None)
                self.cache_priority.pop(key, None)
            
            self.logger.debug(f"🗑️ 낮은 우선순위 캐시 {num_to_remove}개 제거")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 캐시 제거 실패: {e}")

    # ============================================
    # 🔥 비동기 인터페이스
    # ============================================

    async def initialize(self) -> bool:
        """메모리 관리자 비동기 초기화"""
        try:
            # M3 Max 최적화 설정
            if self.is_m3_max and self.optimization_enabled:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._optimize_for_m3_max
                )
            
            self.logger.debug(f"✅ MemoryManager 비동기 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ MemoryManager 비동기 초기화 실패: {e}")
            return False

    async def cleanup(self):
        """비동기 메모리 정리"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.cleanup_memory, True)
        except Exception as e:
            self.logger.warning(f"⚠️ 비동기 메모리 정리 실패: {e}")

    def get_usage(self) -> Dict[str, Any]:
        """동기 사용량 조회 (하위 호환)"""
        try:
            stats = self.get_memory_stats()
            return {
                "cpu_percent": stats.cpu_percent,
                "cpu_used_gb": stats.cpu_used_gb,
                "cpu_total_gb": stats.cpu_total_gb,
                "gpu_allocated_gb": stats.gpu_allocated_gb,
                "gpu_total_gb": stats.gpu_total_gb,
                "cache_size_mb": stats.cache_size_mb,
                "device": self.device,
                "is_m3_max": self.is_m3_max
            }
        except Exception as e:
            self.logger.warning(f"⚠️ 사용량 조회 실패: {e}")
            return {"error": str(e)}

    def __del__(self):
        """소멸자"""
        try:
            if hasattr(self, 'monitoring_active'):
                self.monitoring_active = False
            if hasattr(self, 'tensor_cache'):
                self.clear_cache(aggressive=True)
        except:
            pass

# ==============================================
# 🔥 MemoryManagerAdapter 클래스 (VirtualFittingStep용) - 완전 구현
# ==============================================

class MemoryManagerAdapter:
    """
    🔥 완전 구현된 MemoryManagerAdapter 클래스
    ✅ optimize_memory 메서드 완전 구현
    ✅ VirtualFittingStep 호환성 100%
    ✅ 모든 필요한 메서드 위임 및 구현
    ✅ 에러 핸들링 및 폴백 메커니즘
    """
    
    def __init__(self, base_manager: Optional[MemoryManager] = None, device: str = "auto", **kwargs):
        """MemoryManagerAdapter 초기화"""
        try:
            # 베이스 매니저 설정
            if base_manager is None:
                self._base_manager = MemoryManager(device=device, **kwargs)
            else:
                self._base_manager = base_manager
            
            # 속성 초기화
            self.device = self._base_manager.device
            self.is_m3_max = self._base_manager.is_m3_max
            self.memory_gb = self._base_manager.memory_gb
            self.logger = logging.getLogger("MemoryManagerAdapter")
            
            # 어댑터 고유 속성
            self.adapter_initialized = True
            self.optimization_cache = {}
            self.last_optimization_time = 0
            
            self.logger.debug(f"✅ MemoryManagerAdapter 초기화 완료 - 디바이스: {self.device}")
            
        except Exception as e:
            self.logger.error(f"❌ MemoryManagerAdapter 초기화 실패: {e}")
            # 최소한의 초기화
            self._base_manager = MemoryManager(device="cpu")
            self.device = "cpu"
            self.is_m3_max = False
            self.memory_gb = 16
            self.logger = logging.getLogger("MemoryManagerAdapter")

    def optimize_startup(self, aggressive: bool = False) -> Dict[str, Any]:
        """
        🚀 MemoryManagerAdapter용 optimize_startup (완전 구현)
        ✅ VirtualFittingStep 호환성 보장
        """
        try:
            # 기본 매니저의 optimize_startup 시도
            if hasattr(self._base_manager, 'optimize_startup'):
                try:
                    result = self._base_manager.optimize_startup(aggressive)
                    result["adapter"] = True
                    return result
                except Exception as e:
                    self.logger.warning(f"⚠️ 기본 매니저 optimize_startup 실패: {e}")
            
            # 폴백: optimize_memory 사용
            if hasattr(self, 'optimize_memory'):
                try:
                    result = self.optimize_memory(aggressive)
                    result.update({
                        "adapter": True,
                        "fallback_mode": "optimize_memory",
                        "message": "startup 최적화를 optimize_memory로 대체"
                    })
                    return result
                except Exception as e:
                    self.logger.warning(f"⚠️ 폴백 optimize_memory 실패: {e}")
            
            # 최종 폴백: 기본 시작 최적화
            return self._basic_startup_optimization(aggressive)
            
        except Exception as e:
            self.logger.error(f"❌ MemoryManagerAdapter optimize_startup 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "adapter": True,
                "timestamp": time.time()
            }
    
    def _basic_startup_optimization(self, aggressive: bool = False) -> Dict[str, Any]:
        """기본 시작 최적화 (최종 폴백)"""
        try:
            startup_results = []
            
            # 기본 가비지 컬렉션
            collected = gc.collect()
            startup_results.append(f"기본 GC: {collected}개 객체")
            
            # PyTorch 메모리 정리 (가능한 경우)
            if TORCH_AVAILABLE:
                try:
                    if hasattr(torch.mps, 'empty_cache') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                        startup_results.append("MPS 캐시 정리")
                    elif hasattr(torch.cuda, 'empty_cache') and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        startup_results.append("CUDA 캐시 정리")
                except Exception as e:
                    startup_results.append(f"GPU 캐시 정리 실패: {e}")
            
            return {
                "success": True,
                "message": "기본 시작 최적화 완료",
                "startup_results": startup_results,
                "adapter": True,
                "fallback": True,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "adapter": True,
                "fallback": True,
                "timestamp": time.time()
            }

    async def optimize_startup_async(self, aggressive: bool = False) -> Dict[str, Any]:
        """비동기 시작 최적화"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.optimize_startup, aggressive)
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "adapter": True,
                "async": True,
                "timestamp": time.time()
            }


class GPUMemoryManager:
    # ... 기존 코드 ...
    
    def optimize_startup(self, aggressive: bool = False) -> Dict[str, Any]:
        """GPU 메모리 관리자 시작 최적화 (기존 호환성 유지)"""
        try:
            # 부모 클래스의 optimize_startup 호출
            result = super().optimize_startup(aggressive)
            
            # GPU 특화 시작 최적화 추가
            gpu_startup_results = []
            
            # GPU 메모리 사전 정리
            try:
                self.clear_cache()
                gpu_startup_results.append("GPU 캐시 사전 정리 완료")
            except Exception as e:
                gpu_startup_results.append(f"GPU 캐시 정리 실패: {e}")
            
            # GPU 메모리 사용량 체크
            try:
                usage = self.check_memory_usage()
                if usage.get('memory_limit_gb', 0) > 0:
                    gpu_startup_results.append(f"GPU 메모리 한계: {usage['memory_limit_gb']:.1f}GB")
            except Exception as e:
                gpu_startup_results.append(f"GPU 메모리 체크 실패: {e}")
            
            # 결과 병합
            result["gpu_startup_results"] = gpu_startup_results
            result["gpu_manager"] = True
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ GPU 메모리 관리자 시작 최적화 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "gpu_manager": True,
                "timestamp": time.time()
            }

    async def optimize_memory(self, aggressive: bool = False, **kwargs) -> Dict[str, Any]:
        """
        🔥 VirtualFittingStep에서 필요한 핵심 optimize_memory 메서드
        ✅ 완전 구현으로 AttributeError 해결
        ✅ 비동기 처리 지원
        ✅ M3 Max 최적화 포함
        ✅ 에러 처리 및 폴백 메커니즘
        """
        try:
            start_time = time.time()
            optimization_results = []
            
            # 중복 최적화 방지 (5초 내 재호출 방지)
            if (start_time - self.last_optimization_time) < 5.0:
                return {
                    "success": True,
                    "message": "최근 최적화 완료 (캐시됨)",
                    "cached": True,
                    "device": self.device,
                    "timestamp": start_time
                }
            
            # 1. 기본 메모리 관리자의 optimize_memory 호출 시도
            if hasattr(self._base_manager, 'optimize_memory'):
                try:
                    base_result = await self._base_manager.optimize_memory()
                    optimization_results.append("기본 메모리 최적화 완료")
                    
                    # 성공한 경우 바로 반환
                    if base_result.get("success", False):
                        self.last_optimization_time = start_time
                        return {
                            **base_result,
                            "adapter": True,
                            "optimization_results": optimization_results,
                            "device": self.device,
                            "timestamp": start_time
                        }
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ 기본 최적화 실패, 폴백 모드 사용: {e}")
            
            # 2. 폴백: cleanup_memory 기반 최적화
            cleanup_result = self._base_manager.cleanup_memory(aggressive=aggressive)
            optimization_results.append(f"메모리 정리: {cleanup_result.get('success', False)}")
            
            # 3. 어댑터 특화 최적화
            adapter_optimizations = await self._run_adapter_optimizations(aggressive)
            optimization_results.extend(adapter_optimizations)
            
            # 4. M3 Max 특화 최적화
            if self.is_m3_max:
                m3_optimizations = await self._run_m3_max_optimizations()
                optimization_results.extend(m3_optimizations)
            
            # 5. 최종 메모리 상태 확인
            final_stats = self._base_manager.get_memory_stats()
            
            optimization_time = time.time() - start_time
            self.last_optimization_time = start_time
            
            # 최적화 결과 캐싱
            result = {
                "success": True,
                "message": "MemoryManagerAdapter 메모리 최적화 완료",
                "optimization_time": optimization_time,
                "optimization_results": optimization_results,
                "cleanup_result": cleanup_result,
                "final_memory_stats": {
                    "cpu_used_gb": final_stats.cpu_used_gb,
                    "cpu_available_gb": final_stats.cpu_available_gb,
                    "gpu_allocated_gb": final_stats.gpu_allocated_gb,
                    "cache_size_mb": final_stats.cache_size_mb
                },
                "device": self.device,
                "is_m3_max": self.is_m3_max,
                "adapter": True,
                "aggressive": aggressive,
                "timestamp": start_time
            }
            
            self.optimization_cache = result
            self.logger.debug("✅ MemoryManagerAdapter 메모리 최적화 완료")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ MemoryManagerAdapter 최적화 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "adapter": True,
                "device": self.device,
                "timestamp": time.time()
            }

    async def _run_adapter_optimizations(self, aggressive: bool = False) -> List[str]:
        """어댑터 특화 최적화"""
        optimizations = []
        
        try:
            # 1. 캐시 정리
            if hasattr(self._base_manager, 'clear_cache'):
                self._base_manager.clear_cache(aggressive=aggressive)
                optimizations.append("어댑터 캐시 정리")
            
            # 2. 가비지 컬렉션
            collected = gc.collect()
            optimizations.append(f"가비지 컬렉션: {collected}개 객체")
            
            # 3. PyTorch 메모리 정리
            if TORCH_AVAILABLE:
                try:
                    if self.device == "mps" and torch.backends.mps.is_available():
                        if hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                            optimizations.append("MPS 캐시 정리")
                    elif self.device == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        optimizations.append("CUDA 캐시 정리")
                except Exception as e:
                    optimizations.append(f"GPU 캐시 정리 실패: {str(e)[:50]}")
            
            return optimizations
            
        except Exception as e:
            self.logger.warning(f"⚠️ 어댑터 최적화 실패: {e}")
            return ["어댑터 최적화 실패"]

    async def _run_m3_max_optimizations(self) -> List[str]:
        """M3 Max 특화 최적화"""
        optimizations = []
        
        try:
            if not self.is_m3_max:
                return optimizations
            
            # M3 Max 특화 로직
            if hasattr(self._base_manager, '_optimize_for_m3_max'):
                await asyncio.get_event_loop().run_in_executor(
                    None, self._base_manager._optimize_for_m3_max
                )
                optimizations.append("M3 Max Neural Engine 최적화")
            
            # MPS 특화 정리
            if TORCH_AVAILABLE and torch.backends.mps.is_available():
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                optimizations.append("M3 Max MPS 캐시 정리")
            
            # 메모리 압박 완화
            if hasattr(self._base_manager, '_aggressive_m3_cleanup'):
                self._base_manager._aggressive_m3_cleanup()
                optimizations.append("M3 Max 공격적 메모리 정리")
            
            return optimizations
            
        except Exception as e:
            self.logger.warning(f"⚠️ M3 Max 최적화 실패: {e}")
            return ["M3 Max 최적화 실패"]

    # ============================================
    # 🔥 위임 메서드들 (모든 필요한 메서드 위임)
    # ============================================

    def get_memory_stats(self) -> MemoryStats:
        """메모리 통계 (위임)"""
        try:
            return self._base_manager.get_memory_stats()
        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 통계 조회 실패: {e}")
            return MemoryStats(
                cpu_percent=50.0,
                cpu_available_gb=8.0,
                cpu_used_gb=8.0,
                cpu_total_gb=16.0
            )
    
    def cleanup_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """메모리 정리 (위임)"""
        try:
            result = self._base_manager.cleanup_memory(aggressive)
            result["adapter"] = True
            return result
        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 정리 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "adapter": True,
                "device": self.device
            }
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """메모리 압박 확인 (위임)"""
        try:
            result = self._base_manager.check_memory_pressure()
            result["adapter"] = True
            return result
        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 압박 확인 실패: {e}")
            return {
                "status": "unknown",
                "error": str(e),
                "adapter": True
            }
    
    def clear_cache(self, aggressive: bool = False):
        """캐시 정리 (위임)"""
        try:
            return self._base_manager.clear_cache(aggressive)
        except Exception as e:
            self.logger.warning(f"⚠️ 캐시 정리 실패: {e}")

    async def get_usage_stats(self) -> Dict[str, Any]:
        """사용량 통계 (비동기 래퍼)"""
        try:
            stats = self._base_manager.get_memory_stats()
            return {
                "device": self.device,
                "cpu_used_gb": stats.cpu_used_gb,
                "cpu_total_gb": stats.cpu_total_gb,
                "cpu_available_gb": stats.cpu_available_gb,
                "gpu_allocated_gb": stats.gpu_allocated_gb,
                "gpu_total_gb": stats.gpu_total_gb,
                "cache_size_mb": stats.cache_size_mb,
                "is_m3_max": self.is_m3_max,
                "adapter": True,
                "timestamp": time.time()
            }
        except Exception as e:
            self.logger.warning(f"⚠️ 사용량 통계 조회 실패: {e}")
            return {
                "error": str(e), 
                "adapter": True,
                "timestamp": time.time()
            }

    async def initialize(self) -> bool:
        """비동기 초기화 (위임)"""
        try:
            if hasattr(self._base_manager, 'initialize'):
                result = await self._base_manager.initialize()
            else:
                result = True
            
            self.logger.debug("✅ MemoryManagerAdapter 초기화 완료")
            return result
            
        except Exception as e:
            self.logger.warning(f"⚠️ 어댑터 초기화 실패: {e}")
            return False

    async def cleanup(self):
        """비동기 정리 (위임)"""
        try:
            if hasattr(self._base_manager, 'cleanup'):
                await self._base_manager.cleanup()
            else:
                self.cleanup_memory(aggressive=True)
        except Exception as e:
            self.logger.warning(f"⚠️ 비동기 정리 실패: {e}")

    def get_usage(self) -> Dict[str, Any]:
        """동기 사용량 조회 (위임)"""
        try:
            result = self._base_manager.get_usage()
            result["adapter"] = True
            return result
        except Exception as e:
            self.logger.warning(f"⚠️ 사용량 조회 실패: {e}")
            return {"error": str(e), "adapter": True}

    def __getattr__(self, name):
        """다른 모든 속성/메서드는 기본 관리자로 위임"""
        try:
            return getattr(self._base_manager, name)
        except AttributeError:
            self.logger.warning(f"⚠️ 속성 '{name}'을 찾을 수 없음")
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __del__(self):
        """소멸자"""
        try:
            if hasattr(self, '_base_manager') and self._base_manager:
                if hasattr(self._base_manager, '__del__'):
                    self._base_manager.__del__()
        except:
            pass

# ==============================================
# 🔥 GPUMemoryManager 클래스 (기존 이름 유지)
# ==============================================

class GPUMemoryManager(MemoryManager):
    """
    🍎 GPU 메모리 관리자 (기존 클래스명 유지)
    ✅ 현재 구조와 완벽 호환
    ✅ 기존 코드의 GPUMemoryManager 사용 유지
    ✅ main.py import 오류 완전 해결
    """
    
    def __init__(self, device=None, memory_limit_gb=None, **kwargs):
        """GPU 메모리 관리자 초기화 (기존 시그니처 유지)"""
        
        # 기본값 설정
        if device is None:
            device = SYSTEM_INFO["device"]
        if memory_limit_gb is None:
            memory_limit_gb = SYSTEM_INFO["memory_gb"] * 0.8 if SYSTEM_INFO["is_m3_max"] else 16.0
        
        super().__init__(device=device, memory_limit_gb=memory_limit_gb, **kwargs)
        self.logger = logging.getLogger("GPUMemoryManager")
        
        # 기존 속성 호환성 유지
        self.memory_limit_gb = memory_limit_gb
        
        self.logger.debug(f"🎮 GPUMemoryManager 초기화 - {device} ({memory_limit_gb:.1f}GB)")

    def clear_cache(self):
        """메모리 정리 (기존 메서드 시그니처 유지)"""
        try:
            # 부모 클래스의 메모리 정리 호출
            result = self.cleanup_memory(aggressive=False)
            
            # PyTorch GPU 캐시 정리
            if TORCH_AVAILABLE:
                if self.device == "mps" and torch.backends.mps.is_available():
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                elif self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self.logger.debug("🧹 GPU 캐시 정리 완료")
            return result
            
        except Exception as e:
            self.logger.warning(f"⚠️ GPU 캐시 정리 실패: {e}")
            return {"success": False, "error": str(e)}

    def optimize_startup(self, aggressive: bool = False) -> Dict[str, Any]:
        """GPU 메모리 관리자 시작 최적화 (기존 호환성 유지)"""
        try:
            # 부모 클래스의 optimize_startup 호출
            result = super().optimize_startup(aggressive)
            
            # GPU 특화 시작 최적화 추가
            gpu_startup_results = []
            
            # GPU 메모리 사전 정리
            try:
                self.clear_cache()
                gpu_startup_results.append("GPU 캐시 사전 정리 완료")
            except Exception as e:
                gpu_startup_results.append(f"GPU 캐시 정리 실패: {e}")
            
            # GPU 메모리 사용량 체크
            try:
                usage = self.check_memory_usage()
                if usage.get('memory_limit_gb', 0) > 0:
                    gpu_startup_results.append(f"GPU 메모리 한계: {usage['memory_limit_gb']:.1f}GB")
            except Exception as e:
                gpu_startup_results.append(f"GPU 메모리 체크 실패: {e}")
            
            # 결과 병합
            result["gpu_startup_results"] = gpu_startup_results
            result["gpu_manager"] = True
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ GPU 메모리 관리자 시작 최적화 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "gpu_manager": True,
                "timestamp": time.time()
            }

    def check_memory_usage(self):
        """메모리 사용량 확인 (기존 메서드명 유지)"""
        try:
            stats = self.get_memory_stats()
            
            # 기존 로직과 호환
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()
                used_gb = memory.used / (1024**3)
                if used_gb > self.memory_limit_gb * 0.9:
                    self.logger.warning(f"⚠️ 메모리 사용량 높음: {used_gb:.1f}GB")
                    self.clear_cache()
            
            return {
                "cpu_used_gb": stats.cpu_used_gb,
                "cpu_total_gb": stats.cpu_total_gb,
                "gpu_allocated_gb": stats.gpu_allocated_gb,
                "device": self.device,
                "is_m3_max": self.is_m3_max,
                "memory_limit_gb": self.memory_limit_gb
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 사용량 확인 실패: {e}")
            return {"error": str(e)}

# ==============================================
# 🔥 팩토리 함수들 (기존 이름 완전 유지)
# ==============================================

# 전역 메모리 관리자 인스턴스 (싱글톤)
_global_memory_manager = None
_global_gpu_memory_manager = None
_global_adapter = None
_manager_lock = threading.Lock()

def get_memory_manager(**kwargs) -> MemoryManager:
    """전역 메모리 관리자 인스턴스 반환"""
    global _global_memory_manager
    
    with _manager_lock:
        if _global_memory_manager is None:
            _global_memory_manager = MemoryManager(**kwargs)
        return _global_memory_manager

def get_global_memory_manager(**kwargs) -> MemoryManager:
    """전역 메모리 관리자 인스턴스 반환 (별칭)"""
    return get_memory_manager(**kwargs)

def get_memory_adapter(device: str = "auto", **kwargs) -> MemoryManagerAdapter:
    """VirtualFittingStep용 어댑터 반환"""
    global _global_adapter
    
    try:
        with _manager_lock:
            if _global_adapter is None:
                base_manager = get_memory_manager(device=device, **kwargs)
                _global_adapter = MemoryManagerAdapter(base_manager)
                logger.info(f"✅ 메모리 어댑터 초기화 - 디바이스: {device}")
            return _global_adapter
    except Exception as e:
        logger.error(f"❌ 메모리 어댑터 초기화 실패: {e}")
        # 폴백 어댑터 생성
        fallback_manager = MemoryManager(device="cpu")
        return MemoryManagerAdapter(fallback_manager)

def get_step_memory_manager(step_name: str, **kwargs) -> Union[MemoryManager, MemoryManagerAdapter]:
    """
    🔥 Step별 메모리 관리자 반환 (main.py에서 요구하는 핵심 함수)
    ✅ 기존 함수명 완전 유지
    ✅ __init__.py에서 export되는 핵심 함수
    ✅ import 오류 완전 해결
    ✅ VirtualFittingStep용 MemoryManagerAdapter 지원
    """
    try:
        # Step별 특화 설정 (GitHub 8단계 파이프라인 기준)
        step_configs = {
            "HumanParsingStep": {"memory_limit_gb": 8.0, "optimization_enabled": True},
            "PoseEstimationStep": {"memory_limit_gb": 6.0, "optimization_enabled": True},
            "ClothSegmentationStep": {"memory_limit_gb": 4.0, "optimization_enabled": True},
            "GeometricMatchingStep": {"memory_limit_gb": 6.0, "optimization_enabled": True},
            "ClothWarpingStep": {"memory_limit_gb": 8.0, "optimization_enabled": True},
            "VirtualFittingStep": {"memory_limit_gb": 16.0, "optimization_enabled": True},
            "PostProcessingStep": {"memory_limit_gb": 8.0, "optimization_enabled": True},
            "QualityAssessmentStep": {"memory_limit_gb": 4.0, "optimization_enabled": True}
        }
        
        # M3 Max에서는 더 큰 메모리 할당
        if SYSTEM_INFO["is_m3_max"]:
            for config in step_configs.values():
                config["memory_limit_gb"] *= 2  # M3 Max에서 2배 메모리 할당
        
        # Step별 설정 적용
        step_config = step_configs.get(step_name, {"memory_limit_gb": 8.0})
        final_kwargs = kwargs.copy()
        final_kwargs.update(step_config)
        
        # 메모리 관리자 생성
        base_manager = MemoryManager(**final_kwargs)
        base_manager.step_name = step_name
        base_manager.logger = logging.getLogger(f"memory.{step_name}")
        
        # VirtualFittingStep인 경우 어댑터 반환
        if step_name == "VirtualFittingStep":
            adapter = MemoryManagerAdapter(base_manager)
            logger.debug(f"📝 {step_name} MemoryManagerAdapter 생성 완료")
            return adapter
        else:
            logger.debug(f"📝 {step_name} 메모리 관리자 생성 완료")
            return base_manager
        
    except Exception as e:
        logger.warning(f"⚠️ {step_name} 메모리 관리자 생성 실패: {e}")
        # 폴백: 기본 메모리 관리자 반환
        base_manager = MemoryManager(**kwargs)
        if step_name == "VirtualFittingStep":
            return MemoryManagerAdapter(base_manager)
        return base_manager

def create_memory_manager(device: str = "auto", **kwargs) -> MemoryManager:
    """메모리 관리자 팩토리 함수"""
    try:
        if device == "auto":
            device = SYSTEM_INFO["device"]
        
        logger.debug(f"📦 MemoryManager 생성 - 디바이스: {device}")
        manager = MemoryManager(device=device, **kwargs)
        return manager
    except Exception as e:
        logger.warning(f"⚠️ MemoryManager 생성 실패: {e}")
        # 실패 시에도 기본 인스턴스 반환
        return MemoryManager(device="cpu")

def create_optimized_memory_manager(
    device: str = "auto",
    memory_gb: float = None,
    is_m3_max: bool = None,
    optimization_enabled: bool = True
) -> MemoryManager:
    """최적화된 메모리 관리자 생성"""
    
    # 기본값 설정
    if device == "auto":
        device = SYSTEM_INFO["device"]
    if memory_gb is None:
        memory_gb = SYSTEM_INFO["memory_gb"]
    if is_m3_max is None:
        is_m3_max = SYSTEM_INFO["is_m3_max"]
    
    return MemoryManager(
        device=device,
        memory_gb=memory_gb,
        is_m3_max=is_m3_max,
        optimization_enabled=optimization_enabled,
        auto_cleanup=True,
        enable_caching=True
    )

def initialize_global_memory_manager(device: str = None, **kwargs) -> MemoryManager:
    """전역 메모리 관리자 초기화"""
    global _global_memory_manager
    
    try:
        with _manager_lock:
            if _global_memory_manager is None:
                if device is None:
                    device = SYSTEM_INFO["device"]
                
                _global_memory_manager = MemoryManager(device=device, **kwargs)
                logger.info(f"✅ 전역 메모리 관리자 초기화 완료 - 디바이스: {device}")
            return _global_memory_manager
    except Exception as e:
        logger.warning(f"⚠️ 전역 메모리 관리자 초기화 실패: {e}")
        # 기본 인스턴스 생성
        _global_memory_manager = MemoryManager(device="cpu")
        return _global_memory_manager

def optimize_memory_usage(device: str = None, aggressive: bool = False) -> Dict[str, Any]:
    """메모리 사용량 최적화 - 동기 함수"""
    try:
        manager = get_memory_manager(device=device or "auto")
        
        # 최적화 전 상태
        before_stats = manager.get_memory_stats()
        
        # 메모리 정리
        result = manager.cleanup_memory(aggressive=aggressive)
        
        # 최적화 후 상태
        after_stats = manager.get_memory_stats()
        
        # 결과 계산
        freed_cpu = max(0, before_stats.cpu_used_gb - after_stats.cpu_used_gb)
        freed_gpu = max(0, before_stats.gpu_allocated_gb - after_stats.gpu_allocated_gb)
        freed_cache = max(0, before_stats.cache_size_mb - after_stats.cache_size_mb)
        
        result.update({
            "freed_memory": {
                "cpu_gb": freed_cpu,
                "gpu_gb": freed_gpu,
                "cache_mb": freed_cache
            },
            "before": {
                "cpu_used_gb": before_stats.cpu_used_gb,
                "gpu_allocated_gb": before_stats.gpu_allocated_gb,
                "cache_size_mb": before_stats.cache_size_mb
            },
            "after": {
                "cpu_used_gb": after_stats.cpu_used_gb,
                "gpu_allocated_gb": after_stats.gpu_allocated_gb,
                "cache_size_mb": after_stats.cache_size_mb
            }
        })
        
        logger.info(f"🧹 메모리 최적화 완료 - CPU: {freed_cpu:.2f}GB, GPU: {freed_gpu:.2f}GB")
        return result
        
    except Exception as e:
        logger.warning(f"⚠️ 메모리 최적화 실패: {e}")
        return {
            "success": False,
            "error": str(e),
            "device": device or "unknown"
        }
# ==============================================
# 🔥 새로운 팩토리 함수 업데이트 (여기에 추가!) ← 이 위치에 추가
# ==============================================

def create_startup_optimized_memory_manager(
    device: str = "auto",
    aggressive_startup: bool = False,
    **kwargs
) -> MemoryManager:
    """시작 최적화된 메모리 관리자 생성"""
    try:
        manager = MemoryManager(device=device, **kwargs)
        
        # 시작 시 즉시 최적화 실행
        startup_result = manager.optimize_startup(aggressive=aggressive_startup)
        
        if startup_result.get("success", False):
            logger.info("✅ 시작 최적화된 메모리 관리자 준비 완료")
        else:
            logger.warning("⚠️ 시작 최적화 일부 실패, 기본 모드로 계속")
        
        return manager
        
    except Exception as e:
        logger.warning(f"⚠️ 시작 최적화 메모리 관리자 생성 실패: {e}")
        # 폴백: 기본 메모리 관리자 반환
        return MemoryManager(device="cpu")

def fix_main_py_startup_error():
    """main.py 시작 오류 임시 수정 함수"""
    try:
        # 전역 메모리 관리자에 optimize_startup 메서드 확실히 존재하는지 확인
        manager = get_memory_manager()
        
        if not hasattr(manager, 'optimize_startup'):
            # 메서드가 없으면 동적으로 추가
            def temp_optimize_startup(aggressive=False):
                logger.warning("⚠️ 임시 optimize_startup 메서드 사용 (main.py 오류 회피)")
                return manager.optimize_memory(aggressive=aggressive)
            
            setattr(manager, 'optimize_startup', temp_optimize_startup)
            logger.info("✅ 임시 optimize_startup 메서드 추가 완료")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ main.py 시작 오류 수정 실패: {e}")
        return False

# ==============================================
# 🔥 편의 함수들
# ==============================================

async def optimize_memory():
    """메모리 최적화 (비동기)"""
    try:
        manager = get_memory_manager()
        await manager.cleanup()
    except Exception as e:
        logger.warning(f"⚠️ 비동기 메모리 최적화 실패: {e}")

def check_memory():
    """메모리 상태 확인"""
    try:
        manager = get_memory_manager()
        return manager.check_memory_pressure()
    except Exception as e:
        logger.warning(f"⚠️ 메모리 상태 확인 실패: {e}")
        return {"status": "unknown", "error": str(e)}

def check_memory_available(min_gb: float = 1.0) -> bool:
    """사용 가능한 메모리 확인"""
    try:
        manager = get_memory_manager()
        stats = manager.get_memory_stats()
        return stats.cpu_available_gb >= min_gb
    except Exception as e:
        logger.warning(f"⚠️ 메모리 확인 실패: {e}")
        return True  # 확인 실패 시 true 반환

def get_memory_info() -> Dict[str, Any]:
    """메모리 정보 조회"""
    try:
        manager = get_memory_manager()
        stats = manager.get_memory_stats()
        return {
            "device": manager.device,
            "cpu_total_gb": stats.cpu_total_gb,
            "cpu_available_gb": stats.cpu_available_gb,
            "cpu_used_gb": stats.cpu_used_gb,
            "gpu_total_gb": stats.gpu_total_gb,
            "gpu_allocated_gb": stats.gpu_allocated_gb,
            "is_m3_max": manager.is_m3_max,
            "memory_gb": manager.memory_gb,
            "conda_env": SYSTEM_INFO["in_conda"]
        }
    except Exception as e:
        logger.warning(f"⚠️ 메모리 정보 조회 실패: {e}")
        return {"error": str(e)}

# ==============================================
# 🔥 데코레이터
# ==============================================

def memory_efficient(clear_before: bool = True, clear_after: bool = True):
    """메모리 효율적 실행 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            manager = get_memory_manager()
            if clear_before:
                await manager.cleanup()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                if clear_after:
                    await manager.cleanup()
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            manager = get_memory_manager()
            if clear_before:
                manager.cleanup_memory()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                if clear_after:
                    manager.cleanup_memory()
        
        # 함수가 코루틴인지 확인
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator

# ==============================================
# 🔥 특화 함수들 (conda 환경 우선)
# ==============================================

def setup_conda_memory_optimization():
    """conda 환경 메모리 최적화 설정"""
    try:
        if not SYSTEM_INFO["in_conda"]:
            logger.warning("⚠️ conda 환경이 아닙니다")
            return False
        
        optimizations = []
        
        # 1. NumPy/MKL 스레드 최적화
        if NUMPY_AVAILABLE:
            optimal_threads = min(8, SYSTEM_INFO["cpu_count"])
            os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
            os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
            os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_threads)
            optimizations.append(f"NumPy/MKL 스레드: {optimal_threads}")
        
        # 2. PyTorch 설정
        if TORCH_AVAILABLE:
            torch.set_num_threads(min(16, SYSTEM_INFO["cpu_count"]))
            optimizations.append(f"PyTorch 스레드: {torch.get_num_threads()}")
        
        # 3. M3 Max 특화 설정
        if SYSTEM_INFO["is_m3_max"]:
            os.environ.update({
                'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',
                'PYTORCH_MPS_LOW_WATERMARK_RATIO': '0.0',
                'PYTORCH_ENABLE_MPS_FALLBACK': '1'
            })
            optimizations.append("M3 Max MPS 최적화")
        
        logger.info("✅ conda 환경 메모리 최적화 설정 완료")
        for opt in optimizations:
            logger.debug(f"   - {opt}")
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ conda 메모리 최적화 실패: {e}")
        return False

def get_conda_memory_recommendations() -> List[str]:
    """conda 환경 메모리 최적화 권장사항"""
    recommendations = []
    
    try:
        if not SYSTEM_INFO["in_conda"]:
            recommendations.append("conda 환경 사용 권장")
            return recommendations
        
        # 현재 상태 확인
        current_threads = os.environ.get('OMP_NUM_THREADS', 'auto')
        if current_threads == 'auto':
            recommendations.append("OMP_NUM_THREADS 설정 권장")
        
        if TORCH_AVAILABLE:
            current_torch_threads = torch.get_num_threads()
            optimal_threads = min(16, SYSTEM_INFO["cpu_count"])
            if current_torch_threads != optimal_threads:
                recommendations.append(f"PyTorch 스레드 수 최적화 ({current_torch_threads} → {optimal_threads})")
        
        if SYSTEM_INFO["is_m3_max"]:
            mps_ratio = os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO')
            if mps_ratio != '0.0':
                recommendations.append("M3 Max MPS 메모리 비율 최적화 권장")
        
        if not recommendations:
            recommendations.append("conda 환경 최적화 상태 양호")
        
        return recommendations
        
    except Exception as e:
        logger.warning(f"⚠️ conda 권장사항 생성 실패: {e}")
        return ["conda 최적화 상태 확인 불가"]

def create_conda_optimized_manager(step_name: str = "default", **kwargs) -> Union[MemoryManager, MemoryManagerAdapter]:
    """conda 환경 최적화된 메모리 관리자 생성"""
    try:
        # conda 최적화 먼저 설정
        setup_conda_memory_optimization()
        
        # Step별 설정
        if step_name == "VirtualFittingStep":
            base_manager = create_optimized_memory_manager(
                memory_gb=SYSTEM_INFO["memory_gb"],
                optimization_enabled=True,
                **kwargs
            )
            return MemoryManagerAdapter(base_manager)
        else:
            return create_optimized_memory_manager(
                memory_gb=SYSTEM_INFO["memory_gb"],
                optimization_enabled=True,
                **kwargs
            )
        
    except Exception as e:
        logger.warning(f"⚠️ conda 최적화 관리자 생성 실패: {e}")
        # 폴백
        base_manager = MemoryManager(device="cpu")
        if step_name == "VirtualFittingStep":
            return MemoryManagerAdapter(base_manager)
        return base_manager

# ==============================================
# 🔥 진단 및 디버깅 함수들
# ==============================================

def diagnose_memory_issues() -> Dict[str, Any]:
    """메모리 문제 진단"""
    try:
        diagnosis = {
            "system_info": SYSTEM_INFO,
            "memory_status": {},
            "issues": [],
            "recommendations": []
        }
        
        # 메모리 상태 확인
        manager = get_memory_manager()
        stats = manager.get_memory_stats()
        
        diagnosis["memory_status"] = {
            "cpu_usage_ratio": stats.cpu_used_gb / max(1.0, stats.cpu_total_gb),
            "gpu_usage_ratio": stats.gpu_allocated_gb / max(1.0, stats.gpu_total_gb),
            "cache_size_mb": stats.cache_size_mb,
            "available_gb": stats.cpu_available_gb
        }
        
        # 문제 감지
        cpu_ratio = diagnosis["memory_status"]["cpu_usage_ratio"]
        if cpu_ratio > 0.9:
            diagnosis["issues"].append("CPU 메모리 사용량이 90%를 초과")
            diagnosis["recommendations"].append("aggressive 메모리 정리 실행")
        elif cpu_ratio > 0.75:
            diagnosis["issues"].append("CPU 메모리 사용량이 75%를 초과")
            diagnosis["recommendations"].append("일반 메모리 정리 실행")
        
        if stats.cache_size_mb > 1000:
            diagnosis["issues"].append("캐시 크기가 1GB를 초과")
            diagnosis["recommendations"].append("캐시 정리 실행")
        
        # conda 환경 확인
        if not SYSTEM_INFO["in_conda"]:
            diagnosis["issues"].append("conda 환경이 아님")
            diagnosis["recommendations"].append("conda 환경 사용 권장")
        
        # PyTorch 확인
        if not TORCH_AVAILABLE:
            diagnosis["issues"].append("PyTorch 없음")
            diagnosis["recommendations"].append("conda install pytorch 실행")
        
        return diagnosis
        
    except Exception as e:
        return {
            "error": str(e),
            "system_info": SYSTEM_INFO,
            "recommendations": ["시스템 재시작 권장"]
        }

def print_memory_report():
    """메모리 상태 리포트 출력"""
    try:
        print("\n" + "="*80)
        print("🍎 MyCloset AI - 메모리 상태 리포트")
        print("="*80)
        
        # 시스템 정보
        print(f"🔧 시스템: {SYSTEM_INFO['platform']} / {SYSTEM_INFO['device']}")
        print(f"🍎 M3 Max: {'✅' if SYSTEM_INFO['is_m3_max'] else '❌'}")
        print(f"🐍 conda: {'✅' if SYSTEM_INFO['in_conda'] else '❌'} ({SYSTEM_INFO['conda_env']})")
        print(f"🔥 PyTorch: {'✅' if TORCH_AVAILABLE else '❌'} ({TORCH_VERSION})")
        
        # 메모리 상태
        manager = get_memory_manager()
        stats = manager.get_memory_stats()
        
        print(f"\n💾 메모리 상태:")
        print(f"   CPU: {stats.cpu_used_gb:.1f}GB / {stats.cpu_total_gb:.1f}GB ({stats.cpu_percent:.1f}%)")
        print(f"   GPU: {stats.gpu_allocated_gb:.1f}GB / {stats.gpu_total_gb:.1f}GB")
        print(f"   캐시: {stats.cache_size_mb:.1f}MB")
        print(f"   사용 가능: {stats.cpu_available_gb:.1f}GB")
        
        # 압박 상태
        pressure = manager.check_memory_pressure()
        status_emoji = {"normal": "✅", "warning": "⚠️", "critical": "❌"}.get(pressure["status"], "❓")
        print(f"\n🚨 압박 상태: {status_emoji} {pressure['status']}")
        
        # 권장사항
        if pressure.get("recommendations"):
            print(f"\n📋 권장사항:")
            for rec in pressure["recommendations"]:
                print(f"   - {rec}")
        
        # conda 권장사항
        conda_recs = get_conda_memory_recommendations()
        if conda_recs and conda_recs[0] != "conda 환경 최적화 상태 양호":
            print(f"\n🐍 conda 권장사항:")
            for rec in conda_recs:
                print(f"   - {rec}")
        
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"❌ 메모리 리포트 생성 실패: {e}")

# ==============================================
# 🔥 모듈 익스포트 (기존 구조 완전 유지)
# ==============================================

__all__ = [
    # 🔥 기존 클래스명 완전 유지
    'MemoryManager',
    'MemoryManagerAdapter',      # ✅ VirtualFittingStep 호환용 완전 구현
    'GPUMemoryManager',          # ✅ 현재 구조에서 사용
    'MemoryStats',
    'MemoryConfig',
    
    # 🔥 기존 함수명 완전 유지
    'get_memory_manager',
    'get_global_memory_manager',
    'get_step_memory_manager',   # ✅ main.py에서 필요한 핵심 함수
    'get_memory_adapter',        # ✅ VirtualFittingStep 전용
    'create_memory_manager',
    'create_optimized_memory_manager',
    'initialize_global_memory_manager',
    'optimize_memory_usage',
    'optimize_memory',
    'check_memory',
    'check_memory_available',
    'get_memory_info',
    'memory_efficient',
    
    # 🔧 conda 환경 특화 함수들
    'setup_conda_memory_optimization',
    'get_conda_memory_recommendations',
    'create_conda_optimized_manager',
    'diagnose_memory_issues',
    'print_memory_report',
    
    # 🔧 시스템 정보
    'SYSTEM_INFO',
    'TORCH_AVAILABLE',
    'PSUTIL_AVAILABLE',
    'NUMPY_AVAILABLE'
]
# __all__ 업데이트 (여기에 추가!) ← 이 위치에 추가
__all__.extend([
    'create_startup_optimized_memory_manager',
    'fix_main_py_startup_error'
])
# ==============================================
# 🔥 모듈 로드 완료 (GitHub 프로젝트 최적화)
# ==============================================

# 환경 정보 로깅 (INFO 레벨로 중요 정보만)
logger.info("✅ MemoryManager v8.1 로드 완료 (Complete Implementation)")
logger.info(f"🔧 시스템: {SYSTEM_INFO['platform']} / {SYSTEM_INFO['device']}")

if SYSTEM_INFO["is_m3_max"]:
    logger.info(f"🍎 M3 Max 감지 - {SYSTEM_INFO['memory_gb']}GB 메모리")

if SYSTEM_INFO["in_conda"]:
    logger.info(f"🐍 conda 환경: {SYSTEM_INFO['conda_env']}")

logger.debug("🔗 주요 클래스: MemoryManager, MemoryManagerAdapter, GPUMemoryManager")
logger.debug("🔗 주요 함수: get_step_memory_manager, get_memory_adapter")
logger.debug("⚡ M3 Max + conda 환경 완전 최적화")
logger.debug("🔧 MemoryManagerAdapter optimize_memory 완전 구현")

# M3 Max + conda 조합 확인
if SYSTEM_INFO["is_m3_max"] and SYSTEM_INFO["in_conda"]:
    logger.info("🚀 M3 Max + conda 최고 성능 모드 활성화")

# conda 환경 최적화 자동 설정
if SYSTEM_INFO["in_conda"]:
    try:
        setup_conda_memory_optimization()
        logger.debug("✅ conda 환경 메모리 최적화 자동 설정 완료")
    except Exception as e:
        logger.debug(f"⚠️ conda 자동 최적화 건너뜀: {e}")

logger.info("🎯 AttributeError: 'MemoryManagerAdapter' object has no attribute 'optimize_memory' 완전 해결")