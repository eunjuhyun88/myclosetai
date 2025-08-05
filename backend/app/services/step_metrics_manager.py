# backend/app/services/step_metrics_manager.py
"""
🔥 Step Metrics Manager - 메트릭 및 성능 관리
================================================================================

✅ 성능 메트릭 수집
✅ 메모리 사용량 모니터링
✅ 처리 시간 통계
✅ 시스템 상태 모니터링

Author: MyCloset AI Team
Date: 2025-08-01
Version: 1.0
"""

import logging
import time
import gc
import psutil
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ProcessingMetrics:
    """처리 메트릭 데이터"""
    step_name: str
    step_id: int
    processing_time: float
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    error_message: Optional[str] = None

class StepMetricsManager:
    """Step 메트릭 관리자"""
    
    def __init__(self, max_history: int = 1000):
        self.processing_history: deque = deque(maxlen=max_history)
        self.step_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.system_metrics: Dict[str, Any] = {}
        self.start_time = datetime.now()
        
        # 통계 데이터
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_processing_time = 0.0
        
    def record_processing(self, step_name: str, step_id: int, processing_time: float, 
                         success: bool, memory_usage_mb: Optional[float] = None,
                         cpu_usage_percent: Optional[float] = None, 
                         error_message: Optional[str] = None) -> None:
        """처리 메트릭 기록"""
        metric = ProcessingMetrics(
            step_name=step_name,
            step_id=step_id,
            processing_time=processing_time,
            success=success,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=cpu_usage_percent,
            error_message=error_message
        )
        
        self.processing_history.append(metric)
        
        # 통계 업데이트
        self.total_requests += 1
        self.total_processing_time += processing_time
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Step별 통계 업데이트
        step_key = f"{step_name}_{step_id}"
        if step_key not in self.step_metrics:
            self.step_metrics[step_key] = {
                'step_name': step_name,
                'step_id': step_id,
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'total_processing_time': 0.0,
                'avg_processing_time': 0.0,
                'min_processing_time': float('inf'),
                'max_processing_time': 0.0,
                'last_processed': None
            }
        
        step_stats = self.step_metrics[step_key]
        step_stats['total_requests'] += 1
        step_stats['total_processing_time'] += processing_time
        
        if success:
            step_stats['successful_requests'] += 1
        else:
            step_stats['failed_requests'] += 1
        
        step_stats['avg_processing_time'] = step_stats['total_processing_time'] / step_stats['total_requests']
        step_stats['min_processing_time'] = min(step_stats['min_processing_time'], processing_time)
        step_stats['max_processing_time'] = max(step_stats['max_processing_time'], processing_time)
        step_stats['last_processed'] = datetime.now().isoformat()
        
        logger.debug(f"✅ 메트릭 기록: {step_name} (Step {step_id}) - {processing_time:.2f}s")
    
    def get_metrics(self) -> Dict[str, Any]:
        """전체 메트릭 조회"""
        current_time = datetime.now()
        uptime_seconds = (current_time - self.start_time).total_seconds()
        
        return {
            "uptime_seconds": uptime_seconds,
            "uptime_formatted": str(timedelta(seconds=int(uptime_seconds))),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0,
            "total_processing_time": self.total_processing_time,
            "avg_processing_time": (self.total_processing_time / self.total_requests) if self.total_requests > 0 else 0,
            "requests_per_second": (self.total_requests / uptime_seconds) if uptime_seconds > 0 else 0,
            "step_metrics": dict(self.step_metrics),
            "system_metrics": self.get_system_metrics()
        }
    
    def get_step_metrics(self, step_name: str, step_id: int) -> Dict[str, Any]:
        """특정 Step 메트릭 조회"""
        step_key = f"{step_name}_{step_id}"
        return self.step_metrics.get(step_key, {
            'step_name': step_name,
            'step_id': step_id,
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0,
            'min_processing_time': 0.0,
            'max_processing_time': 0.0,
            'last_processed': None
        })
    
    def get_recent_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """최근 메트릭 조회"""
        recent_metrics = []
        for metric in list(self.processing_history)[-limit:]:
            recent_metrics.append({
                'step_name': metric.step_name,
                'step_id': metric.step_id,
                'processing_time': metric.processing_time,
                'success': metric.success,
                'timestamp': metric.timestamp.isoformat(),
                'memory_usage_mb': metric.memory_usage_mb,
                'cpu_usage_percent': metric.cpu_usage_percent,
                'error_message': metric.error_message
            })
        return recent_metrics
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """시스템 메트릭 조회"""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # 메모리 사용률
            memory = psutil.virtual_memory()
            
            # 디스크 사용률
            disk = psutil.disk_usage('/')
            
            # 네트워크 I/O
            network = psutil.net_io_counters()
            
            self.system_metrics = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_total_gb': disk.total / (1024**3),
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"⚠️ 시스템 메트릭 조회 실패: {e}")
            self.system_metrics = {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
        return self.system_metrics
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 조회"""
        current_time = datetime.now()
        uptime_seconds = (current_time - self.start_time).total_seconds()
        
        # 시간대별 요청 수 계산
        hourly_requests = defaultdict(int)
        for metric in self.processing_history:
            hour_key = metric.timestamp.strftime('%Y-%m-%d %H:00')
            hourly_requests[hour_key] += 1
        
        # 처리 시간 분포 계산
        processing_times = [m.processing_time for m in self.processing_history]
        time_distribution = {
            '0-1s': len([t for t in processing_times if t < 1]),
            '1-5s': len([t for t in processing_times if 1 <= t < 5]),
            '5-10s': len([t for t in processing_times if 5 <= t < 10]),
            '10s+': len([t for t in processing_times if t >= 10])
        }
        
        return {
            'uptime_seconds': uptime_seconds,
            'total_requests': self.total_requests,
            'success_rate': (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0,
            'avg_processing_time': (self.total_processing_time / self.total_requests) if self.total_requests > 0 else 0,
            'requests_per_second': (self.total_requests / uptime_seconds) if uptime_seconds > 0 else 0,
            'hourly_requests': dict(hourly_requests),
            'time_distribution': time_distribution,
            'system_metrics': self.get_system_metrics(),
            'step_performance': {
                step_key: {
                    'avg_time': stats['avg_processing_time'],
                    'success_rate': (stats['successful_requests'] / stats['total_requests'] * 100) if stats['total_requests'] > 0 else 0,
                    'total_requests': stats['total_requests']
                }
                for step_key, stats in self.step_metrics.items()
            }
        }
    
    def export_metrics_csv(self) -> str:
        """메트릭을 CSV 형식으로 내보내기"""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # 헤더 작성
        writer.writerow([
            'timestamp', 'step_name', 'step_id', 'processing_time', 
            'success', 'memory_usage_mb', 'cpu_usage_percent', 'error_message'
        ])
        
        # 데이터 작성
        for metric in self.processing_history:
            writer.writerow([
                metric.timestamp.isoformat(),
                metric.step_name,
                metric.step_id,
                metric.processing_time,
                metric.success,
                metric.memory_usage_mb or '',
                metric.cpu_usage_percent or '',
                metric.error_message or ''
            ])
        
        return output.getvalue()
    
    def reset_metrics(self) -> Dict[str, Any]:
        """메트릭 초기화"""
        self.processing_history.clear()
        self.step_metrics.clear()
        self.system_metrics.clear()
        self.start_time = datetime.now()
        
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_processing_time = 0.0
        
        logger.info("✅ 메트릭 초기화 완료")
        return {
            "success": True,
            "message": "Metrics reset successfully"
        }
    
    def get_memory_usage(self) -> float:
        """메모리 사용량 조회 (MB)"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # MB로 변환
        except Exception as e:
            logger.warning(f"⚠️ 메모리 사용량 조회 실패: {e}")
            return 0.0
    
    def optimize_memory(self) -> Dict[str, Any]:
        """메모리 최적화"""
        try:
            # 가비지 컬렉션 강제 실행
            gc.collect()
            
            # 메모리 사용량 측정
            before_memory = self.get_memory_usage()
            
            # 오래된 메트릭 정리 (최근 100개만 유지)
            if len(self.processing_history) > 100:
                recent_metrics = list(self.processing_history)[-100:]
                self.processing_history.clear()
                self.processing_history.extend(recent_metrics)
            
            after_memory = self.get_memory_usage()
            memory_saved = before_memory - after_memory
            
            logger.info(f"✅ 메모리 최적화 완료: {memory_saved:.2f}MB 절약")
            
            return {
                "success": True,
                "before_memory_mb": before_memory,
                "after_memory_mb": after_memory,
                "memory_saved_mb": memory_saved,
                "message": f"Memory optimization completed: {memory_saved:.2f}MB saved"
            }
            
        except Exception as e:
            logger.error(f"❌ 메모리 최적화 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Memory optimization failed"
            } 