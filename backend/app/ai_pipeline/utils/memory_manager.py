# app/ai_pipeline/utils/memory_manager.py
"""
MyCloset AI - 지능형 메모리 관리 시스템 (M3 Max 최적화)
✅ 최적 생성자 패턴 적용 + create_memory_manager 함수 추가
🔥 핵심: 누락된 팩토리 함수들 모두 추가
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
    """
    지능형 GPU/CPU 메모리 관리자 - Apple Silicon M3 Max 최적화
    ✅ 최적 생성자 패턴 적용
    """
    
    def __init__(
        self,
        device: Optional[str] = None,  # 🔥 최적 패턴: None으로 자동 감지
        config: Optional[Dict[str, Any]] = None,
        **kwargs  # 🚀 확장성: 무제한 추가 파라미터
    ):
        """
        ✅ 최적 생성자 - 메모리 관리 특화

        Args:
            device: 사용할 디바이스 (None=자동감지, 'cpu', 'cuda', 'mps')
            config: 메모리 관리 설정 딕셔너리
            **kwargs: 확장 파라미터들
                - device_type: str = "auto"
                - memory_gb: float = 16.0  
                - is_m3_max: bool = False
                - optimization_enabled: bool = True
                - quality_level: str = "balanced"
                - memory_limit_gb: float = None  # 메모리 제한
                - warning_threshold: float = 0.75  # 경고 임계치
                - critical_threshold: float = 0.9  # 위험 임계치
                - auto_cleanup: bool = True  # 자동 정리
                - monitoring_interval: float = 30.0  # 모니터링 주기
                - enable_caching: bool = True  # 캐싱 활성화
        """
        # 1. 💡 지능적 디바이스 자동 감지
        self.device = self._auto_detect_device(device)

        # 2. 📋 기본 설정
        self.config = config or {}
        self.step_name = self.__class__.__name__
        self.logger = logging.getLogger(f"utils.{self.step_name}")

        # 3. 🔧 표준 시스템 파라미터 추출 (일관성)
        self.device_type = kwargs.get('device_type', 'auto')
        self.memory_gb = kwargs.get('memory_gb', 16.0)
        self.is_m3_max = kwargs.get('is_m3_max', self._detect_m3_max())
        self.optimization_enabled = kwargs.get('optimization_enabled', True)
        self.quality_level = kwargs.get('quality_level', 'balanced')

        # 4. ⚙️ 메모리 관리 특화 파라미터
        memory_limit_gb = kwargs.get('memory_limit_gb', None)
        if memory_limit_gb is None:
            if PSUTIL_AVAILABLE:
                total_memory = psutil.virtual_memory().total / 1024**3
                self.memory_limit_gb = total_memory * 0.8  # 80% 사용
            else:
                self.memory_limit_gb = 16.0  # 기본값
        else:
            self.memory_limit_gb = memory_limit_gb
            
        self.warning_threshold = kwargs.get('warning_threshold', 0.75)
        self.critical_threshold = kwargs.get('critical_threshold', 0.9)
        self.auto_cleanup = kwargs.get('auto_cleanup', True)
        self.monitoring_interval = kwargs.get('monitoring_interval', 30.0)
        self.enable_caching = kwargs.get('enable_caching', True)

        # 5. ⚙️ 스텝별 특화 파라미터를 config에 병합
        self._merge_step_specific_config(kwargs)

        # 6. ✅ 상태 초기화
        self.is_initialized = False

        # 7. 🎯 기존 클래스별 고유 초기화 로직 실행
        self._initialize_step_specific()

        self.logger.info(f"🎯 {self.step_name} 초기화 - 디바이스: {self.device}")

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

    def _detect_m3_max(self) -> bool:
        """🍎 M3 Max 칩 자동 감지"""
        try:
            import platform
            import subprocess

            if platform.system() == 'Darwin':  # macOS
                # M3 Max 감지 로직
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                return 'M3' in result.stdout
        except:
            pass
        return False

    def _merge_step_specific_config(self, kwargs: Dict[str, Any]):
        """⚙️ 스텝별 특화 설정 병합"""
        # 시스템 파라미터 제외하고 모든 kwargs를 config에 병합
        system_params = {
            'device_type', 'memory_gb', 'is_m3_max', 
            'optimization_enabled', 'quality_level',
            'memory_limit_gb', 'warning_threshold', 'critical_threshold',
            'auto_cleanup', 'monitoring_interval', 'enable_caching'
        }

        for key, value in kwargs.items():
            if key not in system_params:
                self.config[key] = value

    def _initialize_step_specific(self):
        """🎯 기존 초기화 로직 완전 유지"""
        # 메모리 통계
        self.stats_history: List[MemoryStats] = []
        self.max_history_length = 100
        
        # 캐시 시스템
        self.tensor_cache: Dict[str, Any] = {}
        self.image_cache: Dict[str, Any] = {}
        self.model_cache: Dict[str, Any] = {}
        self.cache_priority: Dict[str, float] = {}
        
        # 모니터링
        self.monitoring_active = False
        self.monitoring_thread = None
        self._lock = threading.Lock()
        
        logger.info(f"🧠 MemoryManager 초기화 - 디바이스: {self.device}, 메모리 제한: {self.memory_limit_gb:.1f}GB")
        
        # M3 Max 최적화 설정
        if self.device == "mps":
            logger.info("🍎 M3 Max 최적화 모드 활성화")
        
        # 초기화 완료
        self.is_initialized = True
    
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
                process = psutil.Process()
                process_memory_mb = process.memory_info().rss / 1024**2
            else:
                cpu_percent = 0.0
                cpu_total_gb = 16.0
                cpu_used_gb = 8.0
                cpu_available_gb = 8.0
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
                        gpu_allocated_gb = 2.0  # 임시값
                        gpu_total_gb = 128.0  # M3 Max 128GB
                except Exception:
                    pass
            
            # 캐시 크기
            cache_size_mb = 0.0
            if self.enable_caching:
                cache_size_mb = sum(
                    len(str(v)) / 1024**2 for v in 
                    [*self.tensor_cache.values(), *self.image_cache.values(), *self.model_cache.values()]
                )
            
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
                process_memory_mb=process_memory_mb
            )
            
        except Exception as e:
            logger.error(f"메모리 상태 조회 실패: {e}")
            return MemoryStats(
                cpu_percent=0.0,
                cpu_available_gb=8.0,
                cpu_used_gb=8.0,
                cpu_total_gb=16.0
            )
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """메모리 압박 상태 체크"""
        stats = self.get_memory_stats()
        
        cpu_usage_ratio = stats.cpu_used_gb / stats.cpu_total_gb
        gpu_usage_ratio = stats.gpu_allocated_gb / max(1.0, stats.gpu_total_gb)
        
        status = "normal"
        if cpu_usage_ratio > self.critical_threshold or gpu_usage_ratio > self.critical_threshold:
            status = "critical"
        elif cpu_usage_ratio > self.warning_threshold or gpu_usage_ratio > self.warning_threshold:
            status = "warning"
        
        return {
            "status": status,
            "cpu_usage_ratio": cpu_usage_ratio,
            "gpu_usage_ratio": gpu_usage_ratio,
            "cache_size_mb": stats.cache_size_mb,
            "recommendations": self._get_cleanup_recommendations(stats)
        }
    
    def _get_cleanup_recommendations(self, stats: MemoryStats) -> List[str]:
        """정리 권장사항"""
        recommendations = []
        
        cpu_ratio = stats.cpu_used_gb / stats.cpu_total_gb
        if cpu_ratio > 0.8:
            recommendations.append("CPU 메모리 정리 권장")
        
        if stats.gpu_allocated_gb > 10.0:
            recommendations.append("GPU 메모리 정리 권장")
        
        if stats.cache_size_mb > 1000:
            recommendations.append("캐시 정리 권장")
        
        return recommendations
    
    def clear_cache(self, aggressive: bool = False):
        """캐시 정리"""
        try:
            if not self.enable_caching:
                return
                
            with self._lock:
                if aggressive:
                    # 전체 캐시 삭제
                    self.tensor_cache.clear()
                    self.image_cache.clear()
                    self.model_cache.clear()
                    self.cache_priority.clear()
                    logger.info("🧹 전체 캐시 정리 완료")
                else:
                    # 선택적 캐시 정리
                    self._evict_low_priority_cache()
                    logger.debug("🧹 선택적 캐시 정리 완료")
        except Exception as e:
            logger.error(f"캐시 정리 실패: {e}")
    
    def smart_cleanup(self):
        """지능형 메모리 정리"""
        try:
            pressure = self.check_memory_pressure()
            
            if pressure["status"] == "critical":
                self.clear_cache(aggressive=True)
                if TORCH_AVAILABLE:
                    gc.collect()
                    if self.device == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif self.device == "mps" and torch.backends.mps.is_available():
                        # MPS는 empty_cache가 없으므로 대체 방법
                        pass
                logger.info("🚨 긴급 메모리 정리 실행")
            elif pressure["status"] == "warning":
                self.clear_cache(aggressive=False)
                logger.debug("⚠️ 예방적 메모리 정리 실행")
        except Exception as e:
            logger.error(f"지능형 정리 실패: {e}")
    
    async def cleanup(self):
        """비동기 메모리 정리"""
        try:
            # 동기 정리 작업을 비동기로 실행
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.smart_cleanup)
        except Exception as e:
            logger.error(f"비동기 메모리 정리 실패: {e}")
    
    def cache_tensor(self, key: str, tensor: Any, priority: float = 0.5):
        """텐서 캐싱"""
        if not self.enable_caching:
            return
            
        try:
            with self._lock:
                self.tensor_cache[key] = tensor
                self.cache_priority[key] = priority
        except Exception as e:
            logger.error(f"텐서 캐싱 실패: {e}")
    
    def get_cached_tensor(self, key: str, cache_type: str = "tensor") -> Optional[Any]:
        """캐시된 데이터 조회"""
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

    async def initialize(self) -> bool:
        """메모리 관리자 초기화"""
        try:
            # M3 Max 최적화 설정
            if self.is_m3_max and self.optimization_enabled:
                self.optimize_for_inference()
            
            # 모니터링 시작 (옵션)
            if self.config.get('start_monitoring', False):
                self.start_monitoring()
            
            self.logger.info(f"✅ 메모리 관리자 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 메모리 관리자 초기화 실패: {e}")
            return False

    async def get_step_info(self) -> Dict[str, Any]:
        """메모리 관리자 정보 반환"""
        return {
            "step_name": self.step_name,
            "device": self.device,
            "device_type": self.device_type,
            "memory_gb": self.memory_gb,
            "is_m3_max": self.is_m3_max,
            "optimization_enabled": self.optimization_enabled,
            "quality_level": self.quality_level,
            "initialized": self.is_initialized,
            "config_keys": list(self.config.keys()),
            "specialized_features": {
                "memory_limit_gb": self.memory_limit_gb,
                "warning_threshold": self.warning_threshold,
                "critical_threshold": self.critical_threshold,
                "auto_cleanup": self.auto_cleanup,
                "monitoring_interval": self.monitoring_interval,
                "enable_caching": self.enable_caching
            },
            "current_stats": self.get_memory_stats().__dict__,
            "pressure_info": self.check_memory_pressure()
        }
    
    def __del__(self):
        """소멸자"""
        try:
            self.stop_monitoring()
            if self.enable_caching:
                self.clear_cache(aggressive=True)
        except:
            pass

# ============================================
# 🔥 핵심: 누락된 팩토리 함수들 모두 추가
# ============================================

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

# 🔥 핵심: main.py에서 찾는 함수 추가
def create_memory_manager(device: str = "auto", **kwargs) -> MemoryManager:
    """
    🔥 메모리 관리자 팩토리 함수 - main.py에서 사용
    
    Args:
        device: 사용할 디바이스
        **kwargs: 추가 설정
    
    Returns:
        MemoryManager 인스턴스
    """
    try:
        logger.info(f"📦 MemoryManager 생성 - 디바이스: {device}")
        manager = MemoryManager(device=device, **kwargs)
        return manager
    except Exception as e:
        logger.error(f"❌ MemoryManager 생성 실패: {e}")
        # 실패 시에도 기본 인스턴스 반환
        return MemoryManager(device="cpu")

# 추가 팩토리 함수들
def create_optimized_memory_manager(
    device: str = "auto",
    memory_gb: float = 16.0,
    is_m3_max: bool = None,
    optimization_enabled: bool = True
) -> MemoryManager:
    """최적화된 메모리 관리자 생성"""
    if is_m3_max is None:
        is_m3_max = _detect_m3_max()
    
    return MemoryManager(
        device=device,
        memory_gb=memory_gb,
        is_m3_max=is_m3_max,
        optimization_enabled=optimization_enabled,
        auto_cleanup=True,
        enable_caching=True
    )

def _detect_m3_max() -> bool:
    """M3 Max 감지 헬퍼"""
    try:
        import platform
        import subprocess
        if platform.system() == 'Darwin':
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True)
            return 'M3' in result.stdout
    except:
        pass
    return False

# ============================================
# 🔥 핵심: optimize_memory_usage 함수 - 동기로 수정
# ============================================

def optimize_memory_usage(device: str = None, aggressive: bool = False) -> Dict[str, Any]:
    """
    🔥 메모리 사용량 최적화 - 동기 함수로 수정
    
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
                # MPS는 empty_cache 없으므로 대체 방법
                torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None
        
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

def check_memory_available(min_gb: float = 1.0) -> bool:
    """사용 가능한 메모리 확인"""
    try:
        manager = get_memory_manager()
        stats = manager.get_memory_stats()
        return stats.cpu_available_gb >= min_gb
    except Exception:
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
            "gpu_total_gb": stats.gpu_total_gb,
            "gpu_allocated_gb": stats.gpu_allocated_gb
        }
    except Exception as e:
        return {"error": str(e)}

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

# 모듈 익스포트
__all__ = [
    'MemoryManager',
    'MemoryStats',
    'get_memory_manager',
    'get_global_memory_manager',
    'create_memory_manager',  # 🔥 핵심 추가
    'create_optimized_memory_manager',
    'optimize_memory_usage',
    'optimize_memory',
    'check_memory',
    'check_memory_available',  # 🔥 핵심 추가
    'get_memory_info',
    'memory_efficient'
]

# 모듈 로드 확인
logger.info("✅ MemoryManager 모듈 로드 완료 - 모든 팩토리 함수 포함")