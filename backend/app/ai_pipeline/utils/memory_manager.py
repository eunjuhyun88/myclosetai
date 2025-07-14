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
        self.model_cache = weakref.WeakValueDictionary()
        self.tensor_cache = {}
        self.image_cache = {}
        self.cache_priority = {}
        
        # 스레드 안전성
        self._lock = threading.RLock()
        
        # 모니터링
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # 최적화 플래그
        self.cleanup_triggered = False
        self.last_cleanup = time.time()
        self.cleanup_cooldown = 5.0  # 5초 쿨다운
        
        logger.info(f"🧠 MemoryManager 초기화 - 디바이스: {self.device}, 메모리 제한: {self.memory_limit_gb:.1f}GB")
    
    def _detect_optimal_device(self, device: str) -> str:
        """최적 디바이스 자동 감지"""
        if device != "auto":
            return device
            
        if TORCH_AVAILABLE:
            if torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            elif torch.cuda.is_available():
                return "cuda"
        
        return "cpu"
    
    # 메모리 상태 조회
    def get_memory_stats(self) -> MemoryStats:
        """현재 메모리 상태 조회"""
        stats = MemoryStats(
            cpu_percent=0.0,
            cpu_available_gb=0.0,
            cpu_used_gb=0.0,
            cpu_total_gb=0.0
        )
        
        try:
            # CPU 메모리
            if PSUTIL_AVAILABLE:
                vm = psutil.virtual_memory()
                stats.cpu_percent = vm.percent
                stats.cpu_total_gb = vm.total / 1024**3
                stats.cpu_used_gb = vm.used / 1024**3
                stats.cpu_available_gb = vm.available / 1024**3
                stats.swap_used_gb = psutil.swap_memory().used / 1024**3
                
                # 프로세스 메모리
                process = psutil.Process()
                stats.process_memory_mb = process.memory_info().rss / 1024**2
            
            # GPU 메모리 (PyTorch)
            if TORCH_AVAILABLE:
                if self.device == "cuda" and torch.cuda.is_available():
                    stats.gpu_allocated_gb = torch.cuda.memory_allocated() / 1024**3
                    stats.gpu_reserved_gb = torch.cuda.memory_reserved() / 1024**3
                    stats.gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                elif self.device == "mps" and torch.backends.mps.is_available():
                    # MPS는 시스템 RAM 공유
                    stats.gpu_allocated_gb = torch.mps.current_allocated_memory() / 1024**3 if hasattr(torch.mps, 'current_allocated_memory') else 0.0
            
            # 캐시 크기
            stats.cache_size_mb = self._calculate_cache_size()
            
        except Exception as e:
            logger.warning(f"메모리 상태 조회 실패: {e}")
        
        return stats
    
    def _calculate_cache_size(self) -> float:
        """캐시 크기 계산 (MB)"""
        try:
            size_mb = 0.0
            
            # 텐서 캐시
            for tensor in self.tensor_cache.values():
                if hasattr(tensor, 'numel') and hasattr(tensor, 'element_size'):
                    size_mb += tensor.numel() * tensor.element_size() / 1024**2
            
            # 이미지 캐시 (대략적 계산)
            size_mb += len(self.image_cache) * 2.0  # 평균 2MB per image
            
            return size_mb
        except:
            return 0.0
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """메모리 압박 상태 확인"""
        stats = self.get_memory_stats()
        
        cpu_pressure = stats.cpu_percent / 100.0
        gpu_pressure = 0.0
        
        if stats.gpu_total_gb > 0:
            gpu_pressure = stats.gpu_allocated_gb / stats.gpu_total_gb
        
        overall_pressure = max(cpu_pressure, gpu_pressure)
        
        status = "normal"
        if overall_pressure > self.critical_threshold:
            status = "critical"
        elif overall_pressure > self.warning_threshold:
            status = "warning"
        
        return {
            "status": status,
            "cpu_pressure": cpu_pressure,
            "gpu_pressure": gpu_pressure,
            "overall_pressure": overall_pressure,
            "stats": stats,
            "recommendations": self._get_memory_recommendations(overall_pressure)
        }
    
    def _get_memory_recommendations(self, pressure: float) -> List[str]:
        """메모리 압박에 따른 권장사항"""
        recommendations = []
        
        if pressure > self.critical_threshold:
            recommendations.extend([
                "즉시 캐시 정리 실행",
                "불필요한 모델 언로드",
                "배치 크기 축소",
                "이미지 해상도 감소"
            ])
        elif pressure > self.warning_threshold:
            recommendations.extend([
                "주기적 캐시 정리",
                "메모리 모니터링 강화",
                "필요시 배치 크기 조정"
            ])
        else:
            recommendations.append("현재 메모리 상태 양호")
        
        return recommendations
    
    # 메모리 최적화
    async def cleanup(self, aggressive: bool = False):
        """비동기 메모리 정리"""
        current_time = time.time()
        
        # 쿨다운 체크
        if current_time - self.last_cleanup < self.cleanup_cooldown and not aggressive:
            return
        
        try:
            with self._lock:
                self.cleanup_triggered = True
                
                # 캐시 정리
                if aggressive:
                    self.clear_cache(aggressive=True)
                else:
                    await self.smart_cleanup()
                
                # 가비지 컬렉션
                gc.collect()
                
                # GPU 메모리 정리
                if TORCH_AVAILABLE:
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    elif self.device == "mps":
                        torch.mps.empty_cache()
                
                self.last_cleanup = current_time
                self.cleanup_triggered = False
                
                logger.debug("🧹 메모리 정리 완료")
                
        except Exception as e:
            logger.error(f"메모리 정리 실패: {e}")
    
    def clear_cache(self, cache_type: str = "all", aggressive: bool = False):
        """캐시 정리"""
        try:
            with self._lock:
                cleared_count = 0
                
                if cache_type in ["all", "tensor"]:
                    if aggressive:
                        cleared_count += len(self.tensor_cache)
                        self.tensor_cache.clear()
                        self.cache_priority.clear()
                    else:
                        # 우선순위 낮은 것만 정리
                        to_remove = [k for k, v in self.cache_priority.items() if v < 0.3]
                        for key in to_remove:
                            self.tensor_cache.pop(key, None)
                            self.cache_priority.pop(key, None)
                        cleared_count += len(to_remove)
                
                if cache_type in ["all", "image"]:
                    if aggressive:
                        cleared_count += len(self.image_cache)
                        self.image_cache.clear()
                    else:
                        # 오래된 이미지부터 절반 정리
                        items = list(self.image_cache.items())
                        remove_count = len(items) // 2
                        for i in range(remove_count):
                            key, _ = items[i]
                            self.image_cache.pop(key, None)
                        cleared_count += remove_count
                
                if cleared_count > 0:
                    logger.debug(f"🗑️ 캐시 정리 완료: {cleared_count}개 항목")
                
        except Exception as e:
            logger.error(f"캐시 정리 실패: {e}")
    
    async def smart_cleanup(self):
        """지능형 메모리 정리"""
        pressure_info = self.check_memory_pressure()
        pressure = pressure_info["overall_pressure"]
        
        if pressure > self.critical_threshold:
            # 긴급 정리
            self.clear_cache(aggressive=True)
            await asyncio.sleep(0.1)  # 정리 시간 확보
            
        elif pressure > self.warning_threshold:
            # 선택적 정리
            self.clear_cache(aggressive=False)
            self._evict_low_priority_cache()
    
    # 캐시 관리
    def add_to_cache(self, key: str, data: Any, cache_type: str = "tensor", priority: float = 0.5) -> bool:
        """캐시에 데이터 추가"""
        if not self.enable_caching:
            return False
        
        try:
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
    
    # 성능 최적화
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
    
    # 유틸리티 메서드들
    async def get_usage_stats(self) -> Dict[str, Any]:
        """사용 통계 (기존 호환성)"""
        stats = self.get_memory_stats()
        pressure_info = self.check_memory_pressure()
        
        return {
            "memory_usage": {
                "cpu_percent": stats.cpu_percent,
                "cpu_used_gb": stats.cpu_used_gb,
                "cpu_total_gb": stats.cpu_total_gb,
                "gpu_allocated_gb": stats.gpu_allocated_gb,
                "gpu_total_gb": stats.gpu_total_gb,
                "cache_size_mb": stats.cache_size_mb
            },
            "pressure": pressure_info,
            "cache_info": {
                "tensor_cache_size": len(self.tensor_cache),
                "image_cache_size": len(self.image_cache),
                "model_cache_size": len(self.model_cache)
            }
        }
    
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

# ============================================
# 🔥 핵심: 누락된 optimize_memory_usage 함수
# ============================================

def optimize_memory_usage(device: str = None, aggressive: bool = False) -> Dict[str, Any]:
    """
    🔥 메모리 사용량 최적화 (누락된 함수)
    
    Args:
        device: 대상 디바이스 ('mps', 'cuda', 'cpu')
        aggressive: 공격적 정리 여부
    
    Returns:
        최적화 결과 정보
    """
    try:
        manager = get_memory_manager(device=device or "auto")
        
        # 최적화 전 상태
        before_stats = manager.get_memory_stats()
        
        # 메모리 정리
        manager.clear_cache(aggressive=aggressive)
        
        # PyTorch 메모리 정리
        if TORCH_AVAILABLE:
            gc.collect()
            if manager.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif manager.device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        # 최적화 후 상태
        after_stats = manager.get_memory_stats()
        
        # 결과 계산
        freed_cpu = before_stats.cpu_used_gb - after_stats.cpu_used_gb
        freed_gpu = before_stats.gpu_allocated_gb - after_stats.gpu_allocated_gb
        freed_cache = before_stats.cache_size_mb - after_stats.cache_size_mb
        
        result = {
            "success": True,
            "device": manager.device,
            "freed_memory": {
                "cpu_gb": max(0, freed_cpu),
                "gpu_gb": max(0, freed_gpu),
                "cache_mb": max(0, freed_cache)
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
        }
        
        logger.info(f"🧹 메모리 최적화 완료 - CPU: {freed_cpu:.2f}GB, GPU: {freed_gpu:.2f}GB, 캐시: {freed_cache:.1f}MB")
        return result
        
    except Exception as e:
        logger.error(f"메모리 최적화 실패: {e}")
        return {
            "success": False,
            "error": str(e),
            "device": device or "unknown"
        }

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