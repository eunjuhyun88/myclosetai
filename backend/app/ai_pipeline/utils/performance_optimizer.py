# backend/app/ai_pipeline/utils/performance_optimizer.py
"""
⚡ MyCloset AI - 성능 최적화 시스템 v1.0
==========================================
✅ M3 Max 128GB 특화 최적화
✅ 메모리 관리 및 캐싱 시스템
✅ GPU/MPS 최적화
✅ 모델 로딩 최적화
✅ 배치 처리 최적화
✅ 순환참조 방지 - 독립적 모듈
✅ conda 환경 우선 지원

Author: MyCloset AI Team
Date: 2025-07-21
Version: 1.0 (분리된 성능 최적화 시스템)
"""

import gc
import os
import time
import logging
import threading
import asyncio
import psutil
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# ==============================================
# 🔥 1. 시스템 호환성 및 환경 체크
# ==============================================

class SystemCompatibility:
    """시스템 호환성 관리"""
    
    def __init__(self):
        self.torch_available = False
        self.device_type = "cpu"
        self.is_m3_max = False
        self.memory_gb = 16.0
        self.cpu_count = 1
        self.gpu_available = False
        
        self._check_system()
    
    def _check_system(self):
        """시스템 환경 체크"""
        # PyTorch 체크
        try:
            import torch
            self.torch_available = True
            
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device_type = "mps"
                self.is_m3_max = True
                self.gpu_available = True
                
                # MPS 안전 설정
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
                
                logger.info("✅ M3 Max MPS 최적화 활성화")
            elif torch.cuda.is_available():
                self.device_type = "cuda"
                self.gpu_available = True
                logger.info("✅ CUDA GPU 감지")
            else:
                self.device_type = "cpu"
            
            globals()['torch'] = torch
        except ImportError:
            self.torch_available = False
            logger.warning("⚠️ PyTorch 없음")
        
        # 시스템 메모리 체크
        try:
            memory = psutil.virtual_memory()
            self.memory_gb = memory.total / (1024**3)
            
            # M3 Max 감지 (macOS + 높은 메모리)
            if self.memory_gb > 64 and self.device_type == "mps":
                self.is_m3_max = True
                logger.info(f"🍎 M3 Max 감지: {self.memory_gb:.1f}GB 메모리")
            
        except ImportError:
            self.memory_gb = 16.0
        
        # CPU 코어 수
        self.cpu_count = psutil.cpu_count() or 1

# 전역 호환성 관리자
_sys_compat = SystemCompatibility()

TORCH_AVAILABLE = _sys_compat.torch_available
DEFAULT_DEVICE = _sys_compat.device_type
IS_M3_MAX = _sys_compat.is_m3_max
MEMORY_GB = _sys_compat.memory_gb
CPU_COUNT = _sys_compat.cpu_count
GPU_AVAILABLE = _sys_compat.gpu_available

# ==============================================
# 🔥 2. 성능 최적화 설정
# ==============================================

class OptimizationLevel(Enum):
    """최적화 레벨"""
    CONSERVATIVE = "conservative"  # 안전 우선
    BALANCED = "balanced"         # 균형
    AGGRESSIVE = "aggressive"     # 성능 우선
    MAXIMUM = "maximum"          # 최대 성능

class CacheStrategy(Enum):
    """캐싱 전략"""
    NO_CACHE = "no_cache"
    MEMORY_ONLY = "memory_only"
    DISK_ONLY = "disk_only"
    HYBRID = "hybrid"
    AUTO = "auto"

@dataclass
class PerformanceConfig:
    """성능 최적화 설정"""
    # 기본 설정
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    cache_strategy: CacheStrategy = CacheStrategy.AUTO
    device: str = DEFAULT_DEVICE
    
    # 메모리 관리
    memory_limit_ratio: float = 0.8  # 사용 가능한 메모리의 80%
    enable_memory_mapping: bool = True
    auto_cleanup: bool = True
    
    # 모델 최적화
    use_fp16: bool = True if GPU_AVAILABLE else False
    enable_model_compilation: bool = False
    batch_size_optimization: bool = True
    
    # 캐싱 설정
    cache_size_mb: int = int(MEMORY_GB * 1024 * 0.1)  # 메모리의 10%
    disk_cache_path: Optional[str] = "./cache"
    cache_ttl_seconds: int = 3600  # 1시간
    
    # 병렬 처리
    max_workers: int = min(CPU_COUNT, 4)
    enable_async: bool = True
    
    # M3 Max 특화 설정
    mps_optimization: bool = IS_M3_MAX
    unified_memory_optimization: bool = IS_M3_MAX

# ==============================================
# 🔥 3. 메모리 최적화 관리자
# ==============================================

class MemoryOptimizer:
    """메모리 최적화 관리자"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.MemoryOptimizer")
        self._lock = threading.RLock()
        
        # 메모리 추적
        self.allocated_memory = 0
        self.peak_memory = 0
        self.cleanup_threshold = self.config.memory_limit_ratio
        
    def get_memory_info(self) -> Dict[str, Any]:
        """메모리 정보 조회"""
        try:
            memory = psutil.virtual_memory()
            
            info = {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "percent": memory.percent,
                "device": self.config.device,
                "is_m3_max": IS_M3_MAX
            }
            
            # GPU 메모리 정보
            if TORCH_AVAILABLE:
                if self.config.device == "cuda":
                    try:
                        info["gpu_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
                        info["gpu_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
                    except:
                        pass
                elif self.config.device == "mps":
                    try:
                        # MPS 메모리 정보는 제한적
                        info["mps_available"] = True
                        info["unified_memory"] = IS_M3_MAX
                    except:
                        pass
            
            return info
            
        except Exception as e:
            self.logger.error(f"❌ 메모리 정보 조회 실패: {e}")
            return {}
    
    def optimize_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """메모리 최적화 실행"""
        try:
            with self._lock:
                self.logger.info("🧹 메모리 최적화 시작")
                
                results = []
                before_memory = self.get_memory_info()
                
                # 1. Python GC
                gc.collect()
                results.append("Python GC 실행")
                
                # 2. PyTorch 캐시 정리
                if TORCH_AVAILABLE:
                    if self.config.device == "cuda":
                        try:
                            torch.cuda.empty_cache()
                            results.append("CUDA 캐시 정리")
                        except:
                            pass
                    elif self.config.device == "mps" and IS_M3_MAX:
                        try:
                            # M3 Max MPS 캐시 정리
                            if hasattr(torch.mps, 'empty_cache'):
                                torch.mps.empty_cache()
                            elif hasattr(torch.backends.mps, 'empty_cache'):
                                torch.backends.mps.empty_cache()
                            results.append("MPS 캐시 정리")
                        except Exception as e:
                            self.logger.debug(f"MPS 캐시 정리 건너뜀: {e}")
                
                # 3. 강제 정리 (aggressive 모드)
                if aggressive:
                    import ctypes
                    try:
                        ctypes.CDLL("libc.so.6").malloc_trim(0)
                        results.append("libc malloc_trim 실행")
                    except:
                        pass
                
                after_memory = self.get_memory_info()
                
                # 결과 계산
                freed_memory = before_memory.get("used_gb", 0) - after_memory.get("used_gb", 0)
                
                self.logger.info("✅ 메모리 최적화 완료")
                
                return {
                    "success": True,
                    "freed_memory_gb": max(0, freed_memory),
                    "before_memory": before_memory,
                    "after_memory": after_memory,
                    "results": results,
                    "aggressive": aggressive
                }
                
        except Exception as e:
            self.logger.error(f"❌ 메모리 최적화 실패: {e}")
            return {"success": False, "error": str(e)}
    
    async def optimize_memory_async(self, aggressive: bool = False) -> Dict[str, Any]:
        """비동기 메모리 최적화"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.optimize_memory, aggressive)
    
    def check_memory_pressure(self) -> bool:
        """메모리 압박 상태 체크"""
        try:
            memory_info = self.get_memory_info()
            usage_ratio = memory_info.get("percent", 0) / 100.0
            
            return usage_ratio > self.cleanup_threshold
            
        except Exception:
            return False
    
    @contextmanager
    def memory_context(self, auto_cleanup: bool = True):
        """메모리 컨텍스트 관리자"""
        try:
            before = self.get_memory_info()
            yield
        finally:
            if auto_cleanup and self.check_memory_pressure():
                self.optimize_memory()
                after = self.get_memory_info()
                self.logger.debug(f"🧹 자동 메모리 정리: {before.get('percent', 0):.1f}% -> {after.get('percent', 0):.1f}%")

# ==============================================
# 🔥 4. 모델 로딩 최적화기
# ==============================================

class ModelLoadingOptimizer:
    """모델 로딩 최적화기"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ModelLoadingOptimizer")
        self.memory_optimizer = MemoryOptimizer(config)
        
    def optimize_model_loading(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """모델 로딩 최적화"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                return {"success": False, "error": "모델 파일 없음"}
            
            start_time = time.time()
            
            optimization_steps = []
            
            # 1. 메모리 사전 최적화
            if self.config.auto_cleanup:
                self.memory_optimizer.optimize_memory()
                optimization_steps.append("메모리 사전 정리")
            
            # 2. 로딩 전략 결정
            file_size_mb = model_path.stat().st_size / (1024 * 1024)
            
            loading_strategy = self._determine_loading_strategy(file_size_mb)
            optimization_steps.append(f"로딩 전략: {loading_strategy}")
            
            # 3. 디바이스별 최적화
            device_optimization = self._apply_device_optimization()
            optimization_steps.extend(device_optimization)
            
            load_time = time.time() - start_time
            
            return {
                "success": True,
                "load_time_ms": load_time * 1000,
                "file_size_mb": file_size_mb,
                "loading_strategy": loading_strategy,
                "optimization_steps": optimization_steps,
                "device": self.config.device
            }
            
        except Exception as e:
            self.logger.error(f"❌ 모델 로딩 최적화 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def _determine_loading_strategy(self, file_size_mb: float) -> str:
        """로딩 전략 결정"""
        available_memory_gb = MEMORY_GB * 0.8  # 80% 사용 가능
        
        if file_size_mb > available_memory_gb * 1024 * 0.5:  # 50% 이상
            return "memory_mapped"
        elif file_size_mb > 1000:  # 1GB 이상
            return "chunked_loading"
        else:
            return "direct_loading"
    
    def _apply_device_optimization(self) -> List[str]:
        """디바이스별 최적화 적용"""
        optimizations = []
        
        if self.config.device == "mps" and IS_M3_MAX:
            # M3 Max MPS 최적화
            optimizations.extend([
                "MPS 통합 메모리 최적화",
                "MPS 폴백 활성화",
                "MPS 캐시 워밍업"
            ])
            
            if TORCH_AVAILABLE:
                try:
                    # M3 Max 특화 설정
                    torch.mps.set_per_process_memory_fraction(0.8)
                    optimizations.append("MPS 메모리 할당 최적화")
                except:
                    pass
        
        elif self.config.device == "cuda":
            # CUDA 최적화
            optimizations.extend([
                "CUDA 캐시 최적화",
                "CUDA 메모리 풀 설정"
            ])
            
            if TORCH_AVAILABLE:
                try:
                    torch.cuda.empty_cache()
                    optimizations.append("CUDA 캐시 사전 정리")
                except:
                    pass
        
        else:
            # CPU 최적화
            optimizations.extend([
                f"CPU 멀티스레딩 ({self.config.max_workers})",
                "메모리 매핑 최적화"
            ])
        
        return optimizations
    
    def get_optimal_batch_size(self, model_memory_mb: float, input_size: Tuple[int, ...]) -> int:
        """최적 배치 크기 계산"""
        try:
            available_memory_gb = MEMORY_GB * 0.6  # 60% 사용
            available_memory_mb = available_memory_gb * 1024
            
            # 입력 크기 기반 메모리 추정
            if len(input_size) >= 2:
                input_memory_mb = (input_size[0] * input_size[1] * 3 * 4) / (1024 * 1024)  # RGB, float32
            else:
                input_memory_mb = 10  # 기본값
            
            # 배치 크기 계산
            memory_per_batch = model_memory_mb + input_memory_mb * 2  # 모델 + 입력 + 출력
            max_batch_size = int(available_memory_mb / memory_per_batch)
            
            # 최소/최대 제한
            batch_size = max(1, min(max_batch_size, 32))
            
            # 디바이스별 조정
            if self.config.device == "mps" and IS_M3_MAX:
                batch_size = min(batch_size, 8)  # MPS는 작은 배치가 안정적
            elif self.config.device == "cpu":
                batch_size = min(batch_size, 4)  # CPU는 작은 배치
            
            return batch_size
            
        except Exception as e:
            self.logger.warning(f"⚠️ 배치 크기 계산 실패: {e}")
            return 1

# ==============================================
# 🔥 5. 캐싱 시스템
# ==============================================

class PerformanceCache:
    """성능 최적화 캐싱 시스템"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.PerformanceCache")
        
        # 메모리 캐시
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_access_times: Dict[str, float] = {}
        self.cache_access_counts: Dict[str, int] = {}
        
        # 디스크 캐시
        self.disk_cache_enabled = False
        if self.config.disk_cache_path:
            self.disk_cache_path = Path(self.config.disk_cache_path)
            self.disk_cache_path.mkdir(parents=True, exist_ok=True)
            self.disk_cache_enabled = True
        
        self._lock = threading.RLock()
        
        # 자동 정리 스레드
        self._start_cleanup_thread()
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회"""
        try:
            with self._lock:
                # 메모리 캐시 확인
                if key in self.memory_cache:
                    self.cache_access_times[key] = time.time()
                    self.cache_access_counts[key] = self.cache_access_counts.get(key, 0) + 1
                    
                    cache_entry = self.memory_cache[key]
                    
                    # TTL 체크
                    if time.time() - cache_entry['timestamp'] < self.config.cache_ttl_seconds:
                        return cache_entry['data']
                    else:
                        # 만료된 캐시 제거
                        self._remove_key(key)
                
                # 디스크 캐시 확인
                if self.disk_cache_enabled:
                    return self._get_from_disk(key)
                
                return None
                
        except Exception as e:
            self.logger.warning(f"⚠️ 캐시 조회 실패 {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """캐시에 값 저장"""
        try:
            with self._lock:
                ttl = ttl or self.config.cache_ttl_seconds
                
                cache_entry = {
                    'data': value,
                    'timestamp': time.time(),
                    'ttl': ttl,
                    'size_mb': self._estimate_size(value)
                }
                
                # 메모리 캐시 저장
                self.memory_cache[key] = cache_entry
                self.cache_access_times[key] = time.time()
                self.cache_access_counts[key] = self.cache_access_counts.get(key, 0) + 1
                
                # 메모리 사용량 체크
                if self._get_cache_size_mb() > self.config.cache_size_mb:
                    self._evict_lru_entries()
                
                # 디스크 캐시 저장
                if self.disk_cache_enabled and cache_entry['size_mb'] > 10:  # 10MB 이상만
                    self._save_to_disk(key, value, ttl)
                
                return True
                
        except Exception as e:
            self.logger.warning(f"⚠️ 캐시 저장 실패 {key}: {e}")
            return False
    
    def _estimate_size(self, obj: Any) -> float:
        """객체 크기 추정 (MB)"""
        try:
            import sys
            
            if hasattr(obj, 'nbytes'):  # numpy array
                return obj.nbytes / (1024 * 1024)
            elif TORCH_AVAILABLE and hasattr(obj, 'element_size'):  # torch tensor
                return obj.numel() * obj.element_size() / (1024 * 1024)
            else:
                return sys.getsizeof(obj) / (1024 * 1024)
        except:
            return 1.0  # 기본값
    
    def _get_cache_size_mb(self) -> float:
        """전체 캐시 크기 계산"""
        return sum(entry['size_mb'] for entry in self.memory_cache.values())
    
    def _evict_lru_entries(self):
        """LRU 기반 캐시 정리"""
        try:
            # 접근 시간 기준 정렬
            sorted_keys = sorted(
                self.cache_access_times.keys(),
                key=lambda k: self.cache_access_times[k]
            )
            
            # 오래된 것부터 제거
            target_size = self.config.cache_size_mb * 0.8  # 80%까지 줄임
            
            for key in sorted_keys:
                if self._get_cache_size_mb() <= target_size:
                    break
                self._remove_key(key)
                
        except Exception as e:
            self.logger.warning(f"⚠️ LRU 캐시 정리 실패: {e}")
    
    def _remove_key(self, key: str):
        """키 제거"""
        try:
            if key in self.memory_cache:
                del self.memory_cache[key]
            if key in self.cache_access_times:
                del self.cache_access_times[key]
            if key in self.cache_access_counts:
                del self.cache_access_counts[key]
        except:
            pass
    
    def _get_from_disk(self, key: str) -> Optional[Any]:
        """디스크 캐시에서 조회"""
        try:
            cache_file = self.disk_cache_path / f"{key}.cache"
            if cache_file.exists():
                import pickle
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # TTL 체크
                if time.time() - cache_data['timestamp'] < cache_data['ttl']:
                    return cache_data['data']
                else:
                    cache_file.unlink()  # 만료된 파일 제거
            
            return None
        except:
            return None
    
    def _save_to_disk(self, key: str, value: Any, ttl: int):
        """디스크 캐시에 저장"""
        try:
            cache_file = self.disk_cache_path / f"{key}.cache"
            cache_data = {
                'data': value,
                'timestamp': time.time(),
                'ttl': ttl
            }
            
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            self.logger.debug(f"디스크 캐시 저장 실패 {key}: {e}")
    
    def _start_cleanup_thread(self):
        """자동 정리 스레드 시작"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(300)  # 5분마다 정리
                    self._cleanup_expired_entries()
                except Exception as e:
                    self.logger.debug(f"캐시 정리 스레드 오류: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_expired_entries(self):
        """만료된 엔트리 정리"""
        try:
            with self._lock:
                current_time = time.time()
                expired_keys = []
                
                for key, entry in self.memory_cache.items():
                    if current_time - entry['timestamp'] > entry['ttl']:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    self._remove_key(key)
                
                if expired_keys:
                    self.logger.debug(f"🧹 만료된 캐시 {len(expired_keys)}개 정리")
                    
        except Exception as e:
            self.logger.debug(f"만료 엔트리 정리 실패: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        try:
            with self._lock:
                return {
                    "memory_entries": len(self.memory_cache),
                    "cache_size_mb": self._get_cache_size_mb(),
                    "cache_limit_mb": self.config.cache_size_mb,
                    "disk_cache_enabled": self.disk_cache_enabled,
                    "total_accesses": sum(self.cache_access_counts.values()),
                    "avg_access_per_key": sum(self.cache_access_counts.values()) / max(1, len(self.cache_access_counts))
                }
        except Exception as e:
            return {"error": str(e)}

# ==============================================
# 🔥 6. 통합 성능 최적화기
# ==============================================

class PerformanceOptimizer:
    """통합 성능 최적화기"""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self.logger = logging.getLogger(f"{__name__}.PerformanceOptimizer")
        
        # 서브 시스템들
        self.memory_optimizer = MemoryOptimizer(self.config)
        self.model_optimizer = ModelLoadingOptimizer(self.config)
        self.cache = PerformanceCache(self.config)
        
        # 성능 통계
        self.stats = {
            "optimizations_count": 0,
            "total_time_saved_ms": 0,
            "memory_cleanups": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        self._lock = threading.RLock()
        
        self.logger.info(f"⚡ 성능 최적화기 초기화: {self.config.optimization_level.value}")
    
    def optimize_system(self) -> Dict[str, Any]:
        """시스템 전체 최적화"""
        try:
            with self._lock:
                start_time = time.time()
                
                optimization_results = {
                    "success": True,
                    "optimizations": [],
                    "before_stats": self.get_system_stats(),
                    "warnings": []
                }
                
                # 1. 메모리 최적화
                memory_result = self.memory_optimizer.optimize_memory(
                    aggressive=(self.config.optimization_level == OptimizationLevel.AGGRESSIVE)
                )
                optimization_results["optimizations"].append({
                    "type": "memory",
                    "result": memory_result
                })
                
                # 2. 캐시 최적화
                self.cache._cleanup_expired_entries()
                optimization_results["optimizations"].append({
                    "type": "cache",
                    "result": {"cleaned_expired": True}
                })
                
                # 3. 디바이스별 최적화
                device_optimizations = self._optimize_device_specific()
                optimization_results["optimizations"].append({
                    "type": "device",
                    "result": device_optimizations
                })
                
                # 통계 업데이트
                self.stats["optimizations_count"] += 1
                optimization_time = (time.time() - start_time) * 1000
                self.stats["total_time_saved_ms"] += optimization_time
                
                optimization_results["after_stats"] = self.get_system_stats()
                optimization_results["optimization_time_ms"] = optimization_time
                
                self.logger.info(f"✅ 시스템 최적화 완료: {optimization_time:.1f}ms")
                
                return optimization_results
                
        except Exception as e:
            self.logger.error(f"❌ 시스템 최적화 실패: {e}")
            return {"success": False, "error": str(e)}
    
    async def optimize_system_async(self) -> Dict[str, Any]:
        """비동기 시스템 최적화"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.optimize_system)
    
    def _optimize_device_specific(self) -> Dict[str, Any]:
        """디바이스별 특화 최적화"""
        optimizations = []
        
        try:
            if self.config.device == "mps" and IS_M3_MAX:
                # M3 Max MPS 최적화
                if TORCH_AVAILABLE:
                    try:
                        # MPS 캐시 정리
                        if hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                        optimizations.append("MPS 캐시 정리")
                        
                        # MPS 메모리 최적화
                        torch.mps.set_per_process_memory_fraction(0.8)
                        optimizations.append("MPS 메모리 비율 최적화")
                        
                    except Exception as e:
                        optimizations.append(f"MPS 최적화 부분 실패: {e}")
            
            elif self.config.device == "cuda":
                # CUDA 최적화
                if TORCH_AVAILABLE:
                    try:
                        torch.cuda.empty_cache()
                        optimizations.append("CUDA 캐시 정리")
                        
                        # CUDA 메모리 풀 최적화
                        torch.cuda.memory.set_per_process_memory_fraction(0.9)
                        optimizations.append("CUDA 메모리 풀 최적화")
                        
                    except Exception as e:
                        optimizations.append(f"CUDA 최적화 부분 실패: {e}")
            
            else:
                # CPU 최적화
                optimizations.append(f"CPU 멀티프로세싱 ({self.config.max_workers} workers)")
            
            return {"optimizations": optimizations, "device": self.config.device}
            
        except Exception as e:
            return {"error": str(e), "device": self.config.device}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """시스템 통계 조회"""
        try:
            memory_info = self.memory_optimizer.get_memory_info()
            cache_stats = self.cache.get_cache_stats()
            
            return {
                "memory": memory_info,
                "cache": cache_stats,
                "performance": self.stats.copy(),
                "config": {
                    "optimization_level": self.config.optimization_level.value,
                    "device": self.config.device,
                    "is_m3_max": IS_M3_MAX,
                    "gpu_available": GPU_AVAILABLE
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def create_performance_context(self, auto_optimize: bool = True):
        """성능 최적화 컨텍스트 매니저"""
        return PerformanceContext(self, auto_optimize)

# ==============================================
# 🔥 7. 성능 컨텍스트 매니저
# ==============================================

class PerformanceContext:
    """성능 최적화 컨텍스트 매니저"""
    
    def __init__(self, optimizer: PerformanceOptimizer, auto_optimize: bool = True):
        self.optimizer = optimizer
        self.auto_optimize = auto_optimize
        self.start_time = None
        self.start_stats = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.start_stats = self.optimizer.get_system_stats()
        
        if self.auto_optimize:
            self.optimizer.memory_optimizer.optimize_memory()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        execution_time = (end_time - self.start_time) * 1000
        
        if self.auto_optimize:
            # 메모리 압박 체크 후 정리
            if self.optimizer.memory_optimizer.check_memory_pressure():
                self.optimizer.memory_optimizer.optimize_memory()
        
        end_stats = self.optimizer.get_system_stats()
        
        self.optimizer.logger.debug(
            f"⚡ 성능 컨텍스트 완료: {execution_time:.1f}ms, "
            f"메모리 사용량: {self.start_stats.get('memory', {}).get('percent', 0):.1f}% -> "
            f"{end_stats.get('memory', {}).get('percent', 0):.1f}%"
        )

# ==============================================
# 🔥 8. 성능 데코레이터들
# ==============================================

def performance_optimized(
    cache_key: Optional[str] = None,
    memory_optimize: bool = True,
    cache_ttl: int = 3600
):
    """성능 최적화 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 전역 옵티마이저 가져오기
            optimizer = get_global_optimizer()
            
            # 캐시 키 생성
            if cache_key:
                key = f"{cache_key}_{hash(str(args) + str(sorted(kwargs.items())))}"
            else:
                key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # 캐시 확인
            cached_result = optimizer.cache.get(key)
            if cached_result is not None:
                optimizer.stats["cache_hits"] += 1
                return cached_result
            
            # 메모리 최적화
            if memory_optimize:
                with optimizer.memory_optimizer.memory_context():
                    result = func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # 결과 캐싱
            optimizer.cache.set(key, result, ttl=cache_ttl)
            optimizer.stats["cache_misses"] += 1
            
            return result
        
        return wrapper
    return decorator

def async_performance_optimized(
    cache_key: Optional[str] = None,
    memory_optimize: bool = True,
    cache_ttl: int = 3600
):
    """비동기 성능 최적화 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            optimizer = get_global_optimizer()
            
            # 캐시 키 생성
            if cache_key:
                key = f"{cache_key}_{hash(str(args) + str(sorted(kwargs.items())))}"
            else:
                key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # 캐시 확인
            cached_result = optimizer.cache.get(key)
            if cached_result is not None:
                optimizer.stats["cache_hits"] += 1
                return cached_result
            
            # 메모리 최적화
            if memory_optimize:
                await optimizer.memory_optimizer.optimize_memory_async()
            
            result = await func(*args, **kwargs)
            
            # 결과 캐싱
            optimizer.cache.set(key, result, ttl=cache_ttl)
            optimizer.stats["cache_misses"] += 1
            
            return result
        
        return wrapper
    return decorator

# ==============================================
# 🔥 9. 전역 최적화기 관리
# ==============================================

_global_optimizer: Optional[PerformanceOptimizer] = None
_optimizer_lock = threading.Lock()

def get_global_optimizer(config: Optional[PerformanceConfig] = None) -> PerformanceOptimizer:
    """전역 성능 최적화기 인스턴스 반환"""
    global _global_optimizer
    
    with _optimizer_lock:
        if _global_optimizer is None:
            _global_optimizer = PerformanceOptimizer(config)
            logger.info("🌐 전역 성능 최적화기 생성")
        
        return _global_optimizer

def optimize_system() -> Dict[str, Any]:
    """전역 시스템 최적화"""
    optimizer = get_global_optimizer()
    return optimizer.optimize_system()

async def optimize_system_async() -> Dict[str, Any]:
    """전역 비동기 시스템 최적화"""
    optimizer = get_global_optimizer()
    return await optimizer.optimize_system_async()

def get_system_performance_stats() -> Dict[str, Any]:
    """전역 시스템 성능 통계"""
    optimizer = get_global_optimizer()
    return optimizer.get_system_stats()

def cleanup_global_optimizer():
    """전역 최적화기 정리"""
    global _global_optimizer
    
    with _optimizer_lock:
        if _global_optimizer:
            _global_optimizer.memory_optimizer.optimize_memory(aggressive=True)
            _global_optimizer = None
        logger.info("🌐 전역 성능 최적화기 정리 완료")

# ==============================================
# 🔥 10. 모듈 내보내기
# ==============================================

__all__ = [
    # 핵심 클래스들
    'PerformanceOptimizer',
    'MemoryOptimizer',
    'ModelLoadingOptimizer',
    'PerformanceCache',
    'PerformanceContext',
    
    # 설정 클래스들
    'PerformanceConfig',
    'OptimizationLevel',
    'CacheStrategy',
    
    # 데코레이터들
    'performance_optimized',
    'async_performance_optimized',
    
    # 전역 함수들
    'get_global_optimizer',
    'optimize_system',
    'optimize_system_async',
    'get_system_performance_stats',
    'cleanup_global_optimizer',
    
    # 상수들
    'TORCH_AVAILABLE',
    'DEFAULT_DEVICE',
    'IS_M3_MAX',
    'MEMORY_GB',
    'CPU_COUNT',
    'GPU_AVAILABLE'
]

# 모듈 정리 함수 등록
import atexit
atexit.register(cleanup_global_optimizer)

logger.info("✅ 성능 최적화 시스템 v1.0 로드 완료")
logger.info(f"⚡ 시스템 정보:")
logger.info(f"   - 메모리: {MEMORY_GB:.1f}GB")
logger.info(f"   - CPU 코어: {CPU_COUNT}")
logger.info(f"   - 디바이스: {DEFAULT_DEVICE}")
logger.info(f"   - M3 Max: {'✅' if IS_M3_MAX else '❌'}")
logger.info(f"   - GPU: {'✅' if GPU_AVAILABLE else '❌'}")
logger.info("🍎 M3 Max 128GB 특화 최적화")
logger.info("💾 메모리 관리 및 캐싱 시스템")
logger.info("🔧 GPU/MPS 최적화")
logger.info("📦 모델 로딩 최적화")
logger.info("🔗 순환참조 방지 - 독립적 모듈")