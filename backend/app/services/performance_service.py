"""
성능 모니터링 서비스
"""

import time
import logging
import gc
import psutil
from typing import Dict, Any, Optional
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)


def create_performance_monitor(operation_name: str):
    """성능 모니터링 컨텍스트 매니저 생성"""
    
    class PerformanceMetric:
        def __init__(self, name):
            self.name = name
            self.start_time = None
            self.end_time = None
            self.duration = None
            self.memory_before = None
            self.memory_after = None
            self.memory_used = None
        
        def __enter__(self):
            self.start_time = time.time()
            self.memory_before = self._get_memory_usage()
            logger.info(f"🔄 성능 모니터링 시작: {self.name}")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end_time = time.time()
            self.duration = self.end_time - self.start_time
            self.memory_after = self._get_memory_usage()
            self.memory_used = self.memory_after - self.memory_before
            
            logger.info(f"✅ 성능 모니터링 완료: {self.name}")
            logger.info(f"   - 소요 시간: {self.duration:.3f}초")
            logger.info(f"   - 메모리 사용: {self.memory_used:.1f}MB")
            
            if exc_type:
                logger.error(f"❌ 성능 모니터링 중 오류: {exc_type.__name__}: {exc_val}")
        
        def _get_memory_usage(self) -> float:
            """현재 메모리 사용량 조회 (MB)"""
            try:
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024
            except Exception:
                return 0.0
    
    return PerformanceMetric(operation_name)


def get_system_performance_metrics() -> Dict[str, Any]:
    """시스템 성능 메트릭 조회"""
    try:
        # CPU 사용률
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 메모리 정보
        memory = psutil.virtual_memory()
        
        # 디스크 정보
        disk = psutil.disk_usage('/')
        
        # 네트워크 정보
        network = psutil.net_io_counters()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "percent": cpu_percent,
                "count": psutil.cpu_count(),
                "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            "memory": {
                "total": memory.total / 1024**3,  # GB
                "available": memory.available / 1024**3,  # GB
                "used": memory.used / 1024**3,  # GB
                "percent": memory.percent
            },
            "disk": {
                "total": disk.total / 1024**3,  # GB
                "used": disk.used / 1024**3,  # GB
                "free": disk.free / 1024**3,  # GB
                "percent": (disk.used / disk.total) * 100
            },
            "network": {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
        }
    except Exception as e:
        logger.error(f"❌ 시스템 성능 메트릭 조회 실패: {e}")
        return {"error": str(e)}


def optimize_memory_usage() -> Dict[str, Any]:
    """메모리 사용량 최적화"""
    try:
        # 가비지 컬렉션
        collected = gc.collect()
        
        # 메모리 사용량 확인
        memory_before = psutil.virtual_memory().percent
        
        # 강제 가비지 컬렉션
        for _ in range(3):
            gc.collect()
        
        # 메모리 사용량 재확인
        memory_after = psutil.virtual_memory().percent
        
        return {
            "success": True,
            "collected_objects": collected,
            "memory_before": memory_before,
            "memory_after": memory_after,
            "memory_saved": memory_before - memory_after,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ 메모리 최적화 실패: {e}")
        return {"success": False, "error": str(e)}


@contextmanager
def performance_timer(operation_name: str):
    """성능 타이머 컨텍스트 매니저"""
    start_time = time.time()
    start_memory = get_system_performance_metrics()
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = get_system_performance_metrics()
        
        duration = end_time - start_time
        memory_diff = end_memory.get('memory', {}).get('used', 0) - start_memory.get('memory', {}).get('used', 0)
        
        logger.info(f"⏱️ 성능 타이머 - {operation_name}:")
        logger.info(f"   - 소요 시간: {duration:.3f}초")
        logger.info(f"   - 메모리 변화: {memory_diff:.1f}MB") 