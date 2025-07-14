"""
지능형 메모리 관리 시스템
- 동적 메모리 할당
- 캐시 최적화
- GPU 메모리 모니터링
- 자동 가비지 컬렉션
- OOM 방지
"""
import psutil
import torch
import gc
import threading
import time
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from contextlib import contextmanager
import weakref
from functools import wraps
import numpy as np

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

class MemoryManager:
    """GPU 메모리 관리자"""
    
    def __init__(self, 
                 device: str = "mps", 
                 memory_limit_gb: float = 16.0,
                 warning_threshold: float = 0.8,
                 critical_threshold: float = 0.95,
                 auto_cleanup: bool = True,
                 monitoring_interval: float = 30.0):
        
        self.device = torch.device(device)
        self.memory_limit_gb = memory_limit_gb
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.auto_cleanup = auto_cleanup
        self.monitoring_interval = monitoring_interval
        
        # 메모리 통계
        self.stats_history: List[MemoryStats] = []
        self.max_history_length = 100
        
        # 캐시 관리
        self.model_cache = weakref.WeakValueDictionary()
        self.tensor_cache = {}
        self.cache_priority = {}  # 캐시 우선순위
        
        # 모니터링 스레드
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # 콜백 함수들
        self.warning_callbacks: List[Callable] = []
        self.critical_callbacks: List[Callable] = []
        
        # 초기화
        self._detect_gpu_info()
        if auto_cleanup:
            self.start_monitoring()
        
        logger.info(f"MemoryManager 초기화 - Device: {device}, Limit: {memory_limit_gb}GB")
    
    def _detect_gpu_info(self):
        """GPU 정보 감지"""
        try:
            if self.device.type == 'cuda' and torch.cuda.is_available():
                self.gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"CUDA GPU 감지: {self.gpu_total_gb:.1f}GB")
            elif self.device.type == 'mps' and torch.backends.mps.is_available():
                # MPS는 시스템 메모리 공유
                self.gpu_total_gb = psutil.virtual_memory().total / 1024**3 * 0.7  # 70% 할당
                logger.info(f"MPS GPU 감지: {self.gpu_total_gb:.1f}GB (시스템 메모리 공유)")
            else:
                logger.warning("GPU를 사용할 수 없음, CPU 모드로 동작")
        except Exception as e:
            logger.error(f"GPU 정보 감지 실패: {e}")
    
    def get_memory_stats(self) -> MemoryStats:
        """현재 메모리 상태 조회"""
        # CPU 메모리
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        stats = MemoryStats(
            cpu_percent=memory.percent,
            cpu_available_gb=memory.available / 1024**3,
            cpu_used_gb=memory.used / 1024**3,
            cpu_total_gb=memory.total / 1024**3,
            swap_used_gb=swap.used / 1024**3,
            cache_size_mb=self._get_cache_size_mb()
        )
        
        # GPU 메모리
        try:
            if self.device.type == 'cuda' and torch.cuda.is_available():
                stats.gpu_allocated_gb = torch.cuda.memory_allocated() / 1024**3
                stats.gpu_reserved_gb = torch.cuda.memory_reserved() / 1024**3
                stats.gpu_total_gb = self.gpu_total_gb
            elif self.device.type == 'mps' and torch.backends.mps.is_available():
                stats.gpu_allocated_gb = torch.mps.current_allocated_memory() / 1024**3
                stats.gpu_total_gb = self.gpu_total_gb
        except Exception as e:
            logger.debug(f"GPU 메모리 상태 조회 실패: {e}")
        
        return stats
    
    def _get_cache_size_mb(self) -> float:
        """캐시 크기 계산 (MB)"""
        total_size = 0
        for tensor in self.tensor_cache.values():
            if isinstance(tensor, torch.Tensor):
                total_size += tensor.numel() * tensor.element_size()
        return total_size / 1024**2
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """메모리 압박 상태 확인"""
        stats = self.get_memory_stats()
        
        pressure_info = {
            'cpu_pressure': 'none',
            'gpu_pressure': 'none',
            'recommendations': []
        }
        
        # CPU 메모리 압박 확인
        if stats.cpu_percent > self.critical_threshold * 100:
            pressure_info['cpu_pressure'] = 'critical'
            pressure_info['recommendations'].append('CPU 메모리 부족: 즉시 정리 필요')
        elif stats.cpu_percent > self.warning_threshold * 100:
            pressure_info['cpu_pressure'] = 'warning'
            pressure_info['recommendations'].append('CPU 메모리 사용량 높음')
        
        # GPU 메모리 압박 확인
        if stats.gpu_total_gb > 0:
            gpu_usage_ratio = stats.gpu_allocated_gb / stats.gpu_total_gb
            if gpu_usage_ratio > self.critical_threshold:
                pressure_info['gpu_pressure'] = 'critical'
                pressure_info['recommendations'].append('GPU 메모리 부족: 모델 언로드 필요')
            elif gpu_usage_ratio > self.warning_threshold:
                pressure_info['gpu_pressure'] = 'warning'
                pressure_info['recommendations'].append('GPU 메모리 사용량 높음')
        
        return pressure_info
    
    def clear_cache(self, aggressive: bool = False):
        """캐시 정리"""
        try:
            # 텐서 캐시 정리
            if aggressive:
                self.tensor_cache.clear()
                self.cache_priority.clear()
                logger.info("모든 캐시 정리 완료")
            else:
                # 우선순위가 낮은 캐시만 정리
                to_remove = [k for k, v in self.cache_priority.items() if v < 0.5]
                for key in to_remove:
                    self.tensor_cache.pop(key, None)
                    self.cache_priority.pop(key, None)
                logger.info(f"낮은 우선순위 캐시 {len(to_remove)}개 정리")
            
            # GPU 캐시 정리
            if self.device.type == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif self.device.type == 'mps' and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            # Python 가비지 컬렉션
            gc.collect()
            
        except Exception as e:
            logger.error(f"캐시 정리 실패: {e}")
    
    def smart_cleanup(self):
        """지능형 메모리 정리"""
        pressure = self.check_memory_pressure()
        
        if pressure['cpu_pressure'] == 'critical' or pressure['gpu_pressure'] == 'critical':
            logger.warning("메모리 압박 상태 - 적극적 정리 실행")
            self.clear_cache(aggressive=True)
            
            # 콜백 실행
            for callback in self.critical_callbacks:
                try:
                    callback(pressure)
                except Exception as e:
                    logger.error(f"Critical callback 실행 실패: {e}")
                    
        elif pressure['cpu_pressure'] == 'warning' or pressure['gpu_pressure'] == 'warning':
            logger.info("메모리 사용량 높음 - 부분 정리 실행")
            self.clear_cache(aggressive=False)
            
            # 콜백 실행
            for callback in self.warning_callbacks:
                try:
                    callback(pressure)
                except Exception as e:
                    logger.error(f"Warning callback 실행 실패: {e}")
    
    def add_to_cache(self, key: str, tensor: torch.Tensor, priority: float = 0.5):
        """캐시에 텐서 추가"""
        try:
            # 메모리 압박 시 캐시 추가 거부
            pressure = self.check_memory_pressure()
            if pressure['gpu_pressure'] == 'critical':
                logger.warning("메모리 부족으로 캐시 추가 거부")
                return False
            
            self.tensor_cache[key] = tensor
            self.cache_priority[key] = priority
            
            # 캐시 크기 제한
            if len(self.tensor_cache) > 100:  # 최대 100개
                self._evict_low_priority_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"캐시 추가 실패: {e}")
            return False
    
    def get_from_cache(self, key: str) -> Optional[torch.Tensor]:
        """캐시에서 텐서 조회"""
        tensor = self.tensor_cache.get(key)
        if tensor is not None:
            # 사용 시 우선순위 증가
            self.cache_priority[key] = min(1.0, self.cache_priority.get(key, 0.5) + 0.1)
        return tensor
    
    def _evict_low_priority_cache(self):
        """낮은 우선순위 캐시 제거"""
        if not self.cache_priority:
            return
        
        # 우선순위 기준 정렬
        sorted_items = sorted(self.cache_priority.items(), key=lambda x: x[1])
        
        # 하위 20% 제거
        num_to_remove = max(1, len(sorted_items) // 5)
        for key, _ in sorted_items[:num_to_remove]:
            self.tensor_cache.pop(key, None)
            self.cache_priority.pop(key, None)
        
        logger.debug(f"낮은 우선순위 캐시 {num_to_remove}개 제거")
    
    @contextmanager
    def memory_efficient_context(self, clear_cache_before: bool = True, clear_cache_after: bool = True):
        """메모리 효율적 컨텍스트 매니저"""
        if clear_cache_before:
            self.clear_cache()
        
        initial_stats = self.get_memory_stats()
        
        try:
            yield
        finally:
            if clear_cache_after:
                self.clear_cache()
            
            final_stats = self.get_memory_stats()
            memory_used = final_stats.gpu_allocated_gb - initial_stats.gpu_allocated_gb
            
            if memory_used > 0.1:  # 100MB 이상 사용
                logger.info(f"컨텍스트에서 {memory_used:.2f}GB GPU 메모리 사용")
    
    def memory_monitor_decorator(self, clear_before: bool = True, clear_after: bool = True):
        """메모리 모니터링 데코레이터"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.memory_efficient_context(clear_before, clear_after):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def start_monitoring(self):
        """메모리 모니터링 시작"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("메모리 모니터링 시작")
    
    def stop_monitoring(self):
        """메모리 모니터링 중지"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("메모리 모니터링 중지")
    
    def _monitor_loop(self):
        """모니터링 루프"""
        while self.monitoring_active:
            try:
                stats = self.get_memory_stats()
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
                time.sleep(10)  # 오류 시 대기 시간 증가
    
    def add_warning_callback(self, callback: Callable):
        """경고 콜백 추가"""
        self.warning_callbacks.append(callback)
    
    def add_critical_callback(self, callback: Callable):
        """위험 콜백 추가"""
        self.critical_callbacks.append(callback)
    
    def get_memory_report(self) -> Dict[str, Any]:
        """메모리 사용 보고서"""
        current_stats = self.get_memory_stats()
        pressure = self.check_memory_pressure()
        
        report = {
            'current_stats': current_stats.__dict__,
            'pressure_analysis': pressure,
            'cache_info': {
                'tensor_cache_size': len(self.tensor_cache),
                'cache_size_mb': current_stats.cache_size_mb,
                'model_cache_size': len(self.model_cache)
            },
            'monitoring': {
                'active': self.monitoring_active,
                'history_length': len(self.stats_history)
            }
        }
        
        # 최근 추세 분석
        if len(self.stats_history) >= 5:
            recent_stats = self.stats_history[-5:]
            report['trends'] = {
                'cpu_trend': recent_stats[-1].cpu_percent - recent_stats[0].cpu_percent,
                'gpu_trend': recent_stats[-1].gpu_allocated_gb - recent_stats[0].gpu_allocated_gb
            }
        
        return report
    
    def optimize_for_inference(self):
        """추론 최적화 설정"""
        # 추론 모드 설정
        torch.set_grad_enabled(False)
        
        # 백엔드 최적화
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # 캐시 정리
        self.clear_cache(aggressive=True)
        
        logger.info("추론 최적화 설정 완료")
    
    def __del__(self):
        """소멸자"""
        try:
            self.stop_monitoring()
            self.clear_cache(aggressive=True)
        except:
            pass