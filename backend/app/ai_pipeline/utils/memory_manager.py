# app/ai_pipeline/utils/memory_manager.py
"""
MyCloset AI - 지능형 메모리 관리 시스템 (M3 Max 최적화)
- 동적 메모리 할당
- 캐시 최적화  
- GPU 메모리 모니터링 (MPS/CUDA)
- 자동 가비지 컬렉션
- OOM 방지
- Apple Silicon 최적화
"""
import os
import gc
import threading
import time
import logging
import asyncio
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass
from contextlib import contextmanager, asynccontextmanager
import weakref
from functools import wraps
import numpy as np

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
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)

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

class MemoryManager:
    """지능형 GPU/CPU 메모리 관리자 - Apple Silicon M3 Max 최적화"""
    
    def __init__(self, 
                 device: str = "auto", 
                 memory_limit_gb: float = None,
                 warning_threshold: float = 0.75,
                 critical_threshold: float = 0.9,
                 auto_cleanup: bool = True,
                 monitoring_interval: float = 30.0,
                 enable_caching: bool = True):
        
        # 디바이스 자동 감지
        self.device = self._detect_optimal_device(device)
        
        # 메모리 제한 자동 설정
        if memory_limit_gb is None:
            if PSUTIL_AVAILABLE:
                total_memory = psutil.virtual_memory().total / 1024**3
                self.memory_limit_gb = total_memory * 0.8  # 80% 사용
            else:
                self.memory_limit_gb = 16.0  # 기본값
        else:
            self.memory_limit_gb = memory_limit_gb
            
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.auto_cleanup = auto_cleanup
        self.monitoring_interval = monitoring_interval
        self.enable_caching = enable_caching
        
        # 메모리 통계
        self.stats_history: List[MemoryStats] = []
        self.max_history_length = 100
        
        # 캐시 관리 (약한 참조 사용)
        self.model_cache = weakref.WeakValueDictionary() if enable_caching else {}
        self.tensor_cache = {} if enable_caching else {}
        self.cache_priority = {} if enable_caching else {}
        self.image_cache = {} if enable_caching else {}
        
        # 모니터링 스레드
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # 콜백 함수들
        self.warning_callbacks: List[Callable] = []
        self.critical_callbacks: List[Callable] = []
        
        # GPU 정보 감지
        self.gpu_info = self._detect_gpu_info()
        
        # 초기화
        if auto_cleanup:
            self.start_monitoring()
        
        logger.info(f"MemoryManager 초기화 완료")
        logger.info(f"- Device: {self.device}")
        logger.info(f"- Memory Limit: {self.memory_limit_gb:.1f}GB")
        logger.info(f"- GPU Info: {self.gpu_info}")
    
    def _detect_optimal_device(self, preferred: str) -> str:
        """최적 디바이스 자동 감지"""
        if preferred != "auto":
            return preferred
            
        if not TORCH_AVAILABLE:
            return "cpu"
            
        try:
            # M3 Max MPS 우선
            if torch.backends.mps.is_available():
                return "mps"
            # CUDA 다음
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except Exception as e:
            logger.warning(f"디바이스 감지 실패: {e}")
            return "cpu"
    
    def _detect_gpu_info(self) -> Dict[str, Any]:
        """GPU 정보 감지"""
        info = {
            "available": False,
            "type": "none",
            "memory_gb": 0.0,
            "name": "CPU Only"
        }
        
        if not TORCH_AVAILABLE:
            return info
            
        try:
            if self.device == 'cuda' and torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                info.update({
                    "available": True,
                    "type": "cuda",
                    "memory_gb": props.total_memory / 1024**3,
                    "name": props.name
                })
            elif self.device == 'mps' and torch.backends.mps.is_available():
                # MPS는 시스템 메모리 공유 (M3 Max 최적화)
                if PSUTIL_AVAILABLE:
                    system_memory = psutil.virtual_memory().total / 1024**3
                else:
                    system_memory = 128.0  # M3 Max 기본값
                info.update({
                    "available": True,
                    "type": "mps",
                    "memory_gb": system_memory * 0.7,  # 70% 할당
                    "name": "Apple Silicon MPS"
                })
        except Exception as e:
            logger.error(f"GPU 정보 감지 실패: {e}")
        
        return info
    
    async def get_memory_status(self) -> Dict[str, Any]:
        """현재 메모리 상태 조회 (비동기)"""
        return self.get_memory_stats().__dict__
    
    def get_memory_stats(self) -> MemoryStats:
        """현재 메모리 상태 조회"""
        try:
            if PSUTIL_AVAILABLE:
                # CPU 메모리
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()
                process = psutil.Process()
                
                stats = MemoryStats(
                    cpu_percent=memory.percent,
                    cpu_available_gb=memory.available / 1024**3,
                    cpu_used_gb=memory.used / 1024**3,
                    cpu_total_gb=memory.total / 1024**3,
                    swap_used_gb=swap.used / 1024**3,
                    cache_size_mb=self._get_cache_size_mb(),
                    process_memory_mb=process.memory_info().rss / 1024**2
                )
            else:
                # psutil 없이 기본값
                stats = MemoryStats(
                    cpu_percent=50.0,
                    cpu_available_gb=64.0,
                    cpu_used_gb=64.0,
                    cpu_total_gb=128.0,
                    cache_size_mb=self._get_cache_size_mb(),
                    process_memory_mb=1024.0
                )
            
            # GPU 메모리
            if TORCH_AVAILABLE and self.gpu_info["available"]:
                try:
                    if self.device == 'cuda':
                        stats.gpu_allocated_gb = torch.cuda.memory_allocated() / 1024**3
                        stats.gpu_reserved_gb = torch.cuda.memory_reserved() / 1024**3
                        stats.gpu_total_gb = self.gpu_info["memory_gb"]
                    elif self.device == 'mps':
                        # MPS 메모리 상태 (M3 Max)
                        stats.gpu_allocated_gb = torch.mps.current_allocated_memory() / 1024**3
                        stats.gpu_total_gb = self.gpu_info["memory_gb"]
                except Exception as e:
                    logger.debug(f"GPU 메모리 상태 조회 실패: {e}")
            
            return stats
            
        except Exception as e:
            logger.error(f"메모리 상태 조회 실패: {e}")
            # 기본값 반환
            return MemoryStats(
                cpu_percent=50.0,
                cpu_available_gb=64.0,
                cpu_used_gb=64.0,
                cpu_total_gb=128.0
            )
    
    def _get_cache_size_mb(self) -> float:
        """캐시 크기 계산 (MB)"""
        if not self.enable_caching:
            return 0.0
            
        total_size = 0
        
        try:
            # 텐서 캐시
            for tensor in self.tensor_cache.values():
                if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
                    total_size += tensor.numel() * tensor.element_size()
                elif isinstance(tensor, np.ndarray):
                    total_size += tensor.nbytes
            
            # 이미지 캐시
            for img_data in self.image_cache.values():
                if isinstance(img_data, (bytes, bytearray)):
                    total_size += len(img_data)
                elif isinstance(img_data, np.ndarray):
                    total_size += img_data.nbytes
                    
        except Exception as e:
            logger.debug(f"캐시 크기 계산 실패: {e}")
        
        return total_size / 1024**2
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """메모리 압박 상태 확인"""
        stats = self.get_memory_stats()
        
        pressure_info = {
            'cpu_pressure': 'none',
            'gpu_pressure': 'none',
            'process_pressure': 'none',
            'overall_pressure': 'none',
            'recommendations': [],
            'stats': stats.__dict__
        }
        
        # CPU 메모리 압박 확인
        if stats.cpu_percent > self.critical_threshold * 100:
            pressure_info['cpu_pressure'] = 'critical'
            pressure_info['recommendations'].append('💥 CPU 메모리 임계: 즉시 정리 필요')
        elif stats.cpu_percent > self.warning_threshold * 100:
            pressure_info['cpu_pressure'] = 'warning'
            pressure_info['recommendations'].append('⚠️ CPU 메모리 사용량 높음')
        
        # GPU 메모리 압박 확인
        if stats.gpu_total_gb > 0:
            gpu_usage_ratio = stats.gpu_allocated_gb / stats.gpu_total_gb
            if gpu_usage_ratio > self.critical_threshold:
                pressure_info['gpu_pressure'] = 'critical'
                pressure_info['recommendations'].append('💥 GPU 메모리 임계: 모델 언로드 필요')
            elif gpu_usage_ratio > self.warning_threshold:
                pressure_info['gpu_pressure'] = 'warning'
                pressure_info['recommendations'].append('⚠️ GPU 메모리 사용량 높음')
        
        # 프로세스 메모리 확인
        if stats.process_memory_mb > self.memory_limit_gb * 1024 * 0.9:
            pressure_info['process_pressure'] = 'critical'
            pressure_info['recommendations'].append('💥 프로세스 메모리 한계 근접')
        
        # 전체 압박 수준 결정
        pressures = [pressure_info['cpu_pressure'], pressure_info['gpu_pressure'], pressure_info['process_pressure']]
        if 'critical' in pressures:
            pressure_info['overall_pressure'] = 'critical'
        elif 'warning' in pressures:
            pressure_info['overall_pressure'] = 'warning'
        
        return pressure_info
    
    async def cleanup(self):
        """비동기 메모리 정리"""
        await asyncio.get_event_loop().run_in_executor(None, self.clear_cache, True)
    
    def clear_cache(self, aggressive: bool = False):
        """캐시 정리"""
        if not self.enable_caching:
            return
            
        try:
            with self._lock:
                cleared_items = 0
                
                if aggressive:
                    # 모든 캐시 정리
                    cleared_items += len(self.tensor_cache)
                    self.tensor_cache.clear()
                    self.cache_priority.clear()
                    self.image_cache.clear()
                    logger.info(f"🧹 전체 캐시 정리: {cleared_items}개 항목")
                else:
                    # 우선순위가 낮은 캐시만 정리
                    to_remove = [k for k, v in self.cache_priority.items() if v < 0.5]
                    for key in to_remove:
                        self.tensor_cache.pop(key, None)
                        self.cache_priority.pop(key, None)
                        cleared_items += 1
                    
                    # 이미지 캐시 부분 정리 (LRU 방식)
                    if len(self.image_cache) > 50:
                        items_to_remove = len(self.image_cache) - 30
                        for _ in range(items_to_remove):
                            if self.image_cache:
                                self.image_cache.popitem()
                                cleared_items += 1
                    
                    logger.info(f"🧹 선택적 캐시 정리: {cleared_items}개 항목")
                
                # GPU 캐시 정리
                if TORCH_AVAILABLE:
                    if self.device == 'cuda' and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    elif self.device == 'mps' and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                
                # Python 가비지 컬렉션
                collected = gc.collect()
                logger.debug(f"🗑️ 가비지 컬렉션: {collected}개 객체 정리")
                
        except Exception as e:
            logger.error(f"캐시 정리 실패: {e}")
    
    def smart_cleanup(self):
        """지능형 메모리 정리"""
        pressure = self.check_memory_pressure()
        
        if pressure['overall_pressure'] == 'critical':
            logger.warning("💥 메모리 임계 상태 - 적극적 정리 실행")
            self.clear_cache(aggressive=True)
            
            # 추가 정리 작업
            self._emergency_cleanup()
            
            # 콜백 실행
            for callback in self.critical_callbacks:
                try:
                    callback(pressure)
                except Exception as e:
                    logger.error(f"Critical callback 실행 실패: {e}")
                    
        elif pressure['overall_pressure'] == 'warning':
            logger.info("⚠️ 메모리 사용량 높음 - 부분 정리 실행")
            self.clear_cache(aggressive=False)
            
            # 콜백 실행
            for callback in self.warning_callbacks:
                try:
                    callback(pressure)
                except Exception as e:
                    logger.error(f"Warning callback 실행 실패: {e}")
    
    def _emergency_cleanup(self):
        """비상 메모리 정리"""
        try:
            # 모든 약한 참조 정리
            if self.enable_caching:
                self.model_cache.clear()
            
            # 시스템 레벨 정리
            if hasattr(gc, 'set_threshold'):
                gc.set_threshold(100, 10, 10)
                gc.collect()
                gc.set_threshold(700, 10, 10)  # 기본값 복원
            
            logger.info("🚨 비상 메모리 정리 완료")
            
        except Exception as e:
            logger.error(f"비상 정리 실패: {e}")
    
    # 캐시 관리 메서드들
    def add_to_cache(self, key: str, data: Any, priority: float = 0.5, cache_type: str = "tensor") -> bool:
        """캐시에 데이터 추가"""
        if not self.enable_caching:
            return False
            
        try:
            # 메모리 압박 시 캐시 추가 거부
            pressure = self.check_memory_pressure()
            if pressure['overall_pressure'] == 'critical':
                logger.warning("메모리 부족으로 캐시 추가 거부")
                return False
            
            with self._lock:
                if cache_type == "image":
                    self.image_cache[key] = data
                else:
                    self.tensor_cache[key] = data
                    self.cache_priority[key] = priority
                
                # 캐시 크기 제한
                if len(self.tensor_cache) > 100:
                    self._evict_low_priority_cache()
                
                if len(self.image_cache) > 50:
                    excess = len(self.image_cache) - 30
                    for _ in range(excess):
                        if self.image_cache:
                            self.image_cache.popitem()
            
            return True
            
        except Exception as e:
            logger.error(f"캐시 추가 실패: {e}")
            return False
    
    def get_from_cache(self, key: str, cache_type: str = "tensor") -> Optional[Any]:
        """캐시에서 데이터 조회"""
        if not self.enable_caching:
            return None
            
        try:
            with self._lock:
                if cache_type == "image":
                    return self.image_cache.get(key)
                else:
                    data = self.tensor_cache.get(key)
                    if data is not None:
                        # 사용 시 우선순위 증가
                        self.cache_priority[key] = min(1.0, self.cache_priority.get(key, 0.5) + 0.1)
                    return data
        except Exception as e:
            logger.error(f"캐시 조회 실패: {e}")
            return None
    
    def _evict_low_priority_cache(self):
        """낮은 우선순위 캐시 제거"""
        if not self.cache_priority:
            return
        
        try:
            # 우선순위 기준 정렬
            sorted_items = sorted(self.cache_priority.items(), key=lambda x: x[1])
            
            # 하위 20% 제거
            num_to_remove = max(1, len(sorted_items) // 5)
            for key, _ in sorted_items[:num_to_remove]:
                self.tensor_cache.pop(key, None)
                self.cache_priority.pop(key, None)
            
            logger.debug(f"낮은 우선순위 캐시 {num_to_remove}개 제거")
            
        except Exception as e:
            logger.error(f"캐시 제거 실패: {e}")
    
    # 컨텍스트 매니저
    @asynccontextmanager
    async def memory_efficient_context(self, clear_before: bool = True, clear_after: bool = True):
        """비동기 메모리 효율적 컨텍스트 매니저"""
        if clear_before:
            await self.cleanup()
        
        initial_stats = self.get_memory_stats()
        
        try:
            yield
        finally:
            if clear_after:
                await self.cleanup()
            
            final_stats = self.get_memory_stats()
            memory_diff = final_stats.gpu_allocated_gb - initial_stats.gpu_allocated_gb
            
            if memory_diff > 0.1:  # 100MB 이상 증가
                logger.info(f"📊 컨텍스트 메모리 사용량: +{memory_diff:.2f}GB")
    
    @contextmanager
    def memory_efficient_sync_context(self, clear_before: bool = True, clear_after: bool = True):
        """동기 메모리 효율적 컨텍스트 매니저"""
        if clear_before:
            self.clear_cache()
        
        initial_stats = self.get_memory_stats()
        
        try:
            yield
        finally:
            if clear_after:
                self.clear_cache()
            
            final_stats = self.get_memory_stats()
            memory_diff = final_stats.gpu_allocated_gb - initial_stats.gpu_allocated_gb
            
            if memory_diff > 0.1:
                logger.info(f"📊 컨텍스트 메모리 사용량: +{memory_diff:.2f}GB")
    
    # 모니터링
    def start_monitoring(self):
        """메모리 모니터링 시작"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("📊 메모리 모니터링 시작")
    
    def stop_monitoring(self):
        """메모리 모니터링 중지"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("📊 메모리 모니터링 중지")
    
    def _monitor_loop(self):
        """모니터링 루프"""
        while self.monitoring_active:
            try:
                stats = self.get_memory_stats()
                
                with self._lock:
                    self.stats_history.append(stats)
                    
                    # 히스토리 크기 제한
                    if len(self.stats_history) > self.max_history_length:
                        self.stats_history.pop(0)
                
                # 자동 정리 실행
                if self.auto_cleanup:
                    self.smart_cleanup()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"메모리 모니터링 오류: {e}")
                time.sleep(10)
    
    # 유틸리티 메서드들
    async def get_usage_stats(self) -> Dict[str, Any]:
        """사용 통계 (기존 호환성)"""
        stats = self.get_memory_stats()
        return {
            "memory_usage_mb": stats.cpu_used_gb * 1024,
            "memory_free_mb": stats.cpu_available_gb * 1024,
            "memory_percentage": stats.cpu_percent,
            "process_memory_mb": stats.process_memory_mb,
            "gpu_memory_gb": stats.gpu_allocated_gb,
            "cache_size_mb": stats.cache_size_mb
        }
    
    def add_warning_callback(self, callback: Callable):
        """경고 콜백 추가"""
        self.warning_callbacks.append(callback)
    
    def add_critical_callback(self, callback: Callable):
        """위험 콜백 추가"""
        self.critical_callbacks.append(callback)
    
    def get_memory_report(self) -> Dict[str, Any]:
        """상세 메모리 보고서"""
        current_stats = self.get_memory_stats()
        pressure = self.check_memory_pressure()
        
        report = {
            'timestamp': time.time(),
            'device_info': self.gpu_info,
            'current_stats': current_stats.__dict__,
            'pressure_analysis': pressure,
            'cache_info': {
                'enabled': self.enable_caching,
                'tensor_cache_size': len(self.tensor_cache) if self.enable_caching else 0,
                'image_cache_size': len(self.image_cache) if self.enable_caching else 0,
                'cache_size_mb': current_stats.cache_size_mb,
                'model_cache_size': len(self.model_cache) if self.enable_caching else 0
            },
            'monitoring': {
                'active': self.monitoring_active,
                'history_length': len(self.stats_history),
                'interval_seconds': self.monitoring_interval
            },
            'configuration': {
                'memory_limit_gb': self.memory_limit_gb,
                'warning_threshold': self.warning_threshold,
                'critical_threshold': self.critical_threshold,
                'auto_cleanup': self.auto_cleanup
            }
        }
        
        # 최근 추세 분석
        if len(self.stats_history) >= 5:
            recent_stats = self.stats_history[-5:]
            report['trends'] = {
                'cpu_trend_percent': recent_stats[-1].cpu_percent - recent_stats[0].cpu_percent,
                'gpu_trend_gb': recent_stats[-1].gpu_allocated_gb - recent_stats[0].gpu_allocated_gb,
                'process_trend_mb': recent_stats[-1].process_memory_mb - recent_stats[0].process_memory_mb
            }
        
        return report
    
    def optimize_for_inference(self):
        """추론 최적화 설정"""
        if not TORCH_AVAILABLE:
            return
            
        try:
            # 추론 모드 설정
            torch.set_grad_enabled(False)
            
            # 백엔드 최적화
            if self.device == 'cuda':
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            elif self.device == 'mps':
                # MPS 최적화 (M3 Max)
                torch.backends.mps.is_available()  # MPS 활성화 확인
            
            # 캐시 정리
            self.clear_cache(aggressive=True)
            
            logger.info(f"🚀 {self.device.upper()} 추론 최적화 완료")
            
        except Exception as e:
            logger.error(f"추론 최적화 실패: {e}")
    
    def __del__(self):
        """소멸자"""
        try:
            self.stop_monitoring()
            if self.enable_caching:
                self.clear_cache(aggressive=True)
        except:
            pass

# 전역 메모리 관리자 인스턴스 (싱글톤)
_global_memory_manager: Optional[MemoryManager] = None

def get_memory_manager(**kwargs) -> MemoryManager:
    """전역 메모리 관리자 인스턴스 반환"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager(**kwargs)
    return _global_memory_manager

def get_global_memory_manager(**kwargs) -> MemoryManager:
    """전역 메모리 관리자 인스턴스 반환 (별칭)"""
    return get_memory_manager(**kwargs)

# 편의 함수들
async def optimize_memory():
    """메모리 최적화 (비동기)"""
    manager = get_memory_manager()
    await manager.cleanup()

def check_memory():
    """메모리 상태 확인"""
    manager = get_memory_manager()
    return manager.check_memory_pressure()

# 데코레이터
def memory_efficient(clear_before: bool = True, clear_after: bool = True):
    """메모리 효율적 실행 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            manager = get_memory_manager()
            async with manager.memory_efficient_context(clear_before, clear_after):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            manager = get_memory_manager()
            with manager.memory_efficient_sync_context(clear_before, clear_after):
                return func(*args, **kwargs)
        
        # 함수가 코루틴인지 확인
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator