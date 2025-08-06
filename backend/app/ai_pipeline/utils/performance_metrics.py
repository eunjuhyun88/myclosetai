#!/usr/bin/env python3
"""
🔥 MyCloset AI - 성능 메트릭 및 시간 측정 유틸리티
===================================================

✅ 각 Step별 상세 시간 측정
✅ 추론 단계별 세분화된 시간 분석
✅ 실시간 성능 모니터링
✅ 메모리 사용량 추적
✅ 디바이스별 성능 비교
✅ 성능 최적화 권장사항

Author: MyCloset AI Team
Date: 2025-08-01
Version: 1.0
"""

import time
import logging
import psutil
import gc
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps
import torch

logger = logging.getLogger(__name__)

@dataclass
class StepTimingMetrics:
    """Step별 시간 측정 메트릭"""
    step_name: str
    step_id: int
    total_time: float = 0.0
    preprocessing_time: float = 0.0
    model_loading_time: float = 0.0
    inference_time: float = 0.0
    postprocessing_time: float = 0.0
    memory_peak_mb: float = 0.0
    device_used: str = "cpu"
    model_used: str = "unknown"
    success: bool = True
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'step_name': self.step_name,
            'step_id': self.step_id,
            'total_time': self.total_time,
            'preprocessing_time': self.preprocessing_time,
            'model_loading_time': self.model_loading_time,
            'inference_time': self.inference_time,
            'postprocessing_time': self.postprocessing_time,
            'memory_peak_mb': self.memory_peak_mb,
            'device_used': self.device_used,
            'model_used': self.model_used,
            'success': self.success,
            'error_message': self.error_message
        }

class StepPerformanceMonitor:
    """Step 성능 모니터링 클래스"""
    
    def __init__(self):
        self.step_metrics: List[StepTimingMetrics] = []
        self.current_step: Optional[StepTimingMetrics] = None
        self.start_memory = 0.0
        
    def start_step_monitoring(self, step_name: str, step_id: int) -> None:
        """Step 모니터링 시작"""
        self.current_step = StepTimingMetrics(
            step_name=step_name,
            step_id=step_id
        )
        self.start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        logger.info(f"🔥 [{step_name}] Step {step_id} 모니터링 시작")
        
    def end_step_monitoring(self, success: bool = True, error_message: str = "") -> Optional[StepTimingMetrics]:
        """Step 모니터링 종료"""
        if self.current_step is None:
            return None
            
        # 메모리 사용량 계산
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        self.current_step.memory_peak_mb = end_memory - self.start_memory
        self.current_step.success = success
        self.current_step.error_message = error_message
        
        # 메트릭 저장
        self.step_metrics.append(self.current_step)
        
        # 상세 로그 출력
        self._log_step_performance(self.current_step)
        
        result = self.current_step
        self.current_step = None
        return result
        
    def _log_step_performance(self, metrics: StepTimingMetrics) -> None:
        """Step 성능 로그 출력"""
        logger.info(f"🎯 [{metrics.step_name}] Step {metrics.step_id} 완료!")
        logger.info(f"   📊 총 처리 시간: {metrics.total_time:.3f}초")
        logger.info(f"   📊 전처리 시간: {metrics.preprocessing_time:.3f}초")
        logger.info(f"   📊 모델 로딩 시간: {metrics.model_loading_time:.3f}초")
        logger.info(f"   📊 추론 시간: {metrics.inference_time:.3f}초")
        logger.info(f"   📊 후처리 시간: {metrics.postprocessing_time:.3f}초")
        logger.info(f"   📊 메모리 사용량: {metrics.memory_peak_mb:.1f}MB")
        logger.info(f"   🖥️ 사용 디바이스: {metrics.device_used}")
        logger.info(f"   🧠 사용 모델: {metrics.model_used}")
        logger.info(f"   ✅ 성공 여부: {metrics.success}")
        
        if not metrics.success:
            logger.error(f"   ❌ 오류 메시지: {metrics.error_message}")
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """전체 성능 요약"""
        if not self.step_metrics:
            return {"message": "측정된 Step이 없습니다"}
            
        total_time = sum(m.total_time for m in self.step_metrics)
        total_memory = sum(m.memory_peak_mb for m in self.step_metrics)
        success_count = sum(1 for m in self.step_metrics if m.success)
        
        return {
            "total_steps": len(self.step_metrics),
            "successful_steps": success_count,
            "failed_steps": len(self.step_metrics) - success_count,
            "total_processing_time": total_time,
            "average_step_time": total_time / len(self.step_metrics),
            "total_memory_usage_mb": total_memory,
            "average_memory_per_step_mb": total_memory / len(self.step_metrics),
            "step_details": [m.to_dict() for m in self.step_metrics]
        }

@contextmanager
def step_timing_context(step_name: str, step_id: int, monitor: StepPerformanceMonitor):
    """Step 시간 측정 컨텍스트 매니저"""
    try:
        monitor.start_step_monitoring(step_name, step_id)
        yield monitor
    except Exception as e:
        monitor.end_step_monitoring(success=False, error_message=str(e))
        raise
    else:
        monitor.end_step_monitoring(success=True)

def timing_decorator(step_name: str, step_id: int):
    """Step 시간 측정 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = StepPerformanceMonitor()
            with step_timing_context(step_name, step_id, monitor):
                return func(*args, **kwargs)
        return wrapper
    return decorator

class DetailedStepTimer:
    """상세 Step 타이머"""
    
    def __init__(self, step_name: str, step_id: int):
        self.step_name = step_name
        self.step_id = step_id
        self.start_time = time.time()
        self.stage_times = {}
        self.current_stage = None
        self.stage_start_time = None
        
    def start_stage(self, stage_name: str) -> None:
        """단계 시작"""
        if self.current_stage:
            self.end_stage()
            
        self.current_stage = stage_name
        self.stage_start_time = time.time()
        logger.info(f"🔥 [{self.step_name}] {stage_name} 시작")
        
    def end_stage(self) -> None:
        """단계 종료"""
        if self.current_stage and self.stage_start_time:
            stage_time = time.time() - self.stage_start_time
            self.stage_times[self.current_stage] = stage_time
            logger.info(f"✅ [{self.step_name}] {self.current_stage} 완료: {stage_time:.3f}초")
            
            self.current_stage = None
            self.stage_start_time = None
            
    def end_timing(self, success: bool = True, device: str = "cpu", model: str = "unknown") -> Dict[str, Any]:
        """전체 타이밍 종료"""
        self.end_stage()  # 마지막 단계 종료
        
        total_time = time.time() - self.start_time
        
        # 상세 로그 출력
        logger.info(f"🎯 [{self.step_name}] Step {self.step_id} 전체 완료!")
        logger.info(f"   📊 총 처리 시간: {total_time:.3f}초")
        
        for stage_name, stage_time in self.stage_times.items():
            logger.info(f"   📊 {stage_name}: {stage_time:.3f}초")
            
        logger.info(f"   🖥️ 사용 디바이스: {device}")
        logger.info(f"   🧠 사용 모델: {model}")
        logger.info(f"   ✅ 성공 여부: {success}")
        
        return {
            'step_name': self.step_name,
            'step_id': self.step_id,
            'total_time': total_time,
            'stage_times': self.stage_times,
            'device_used': device,
            'model_used': model,
            'success': success
        }

def create_step_timer(step_name: str, step_id: int) -> DetailedStepTimer:
    """Step 타이머 생성"""
    return DetailedStepTimer(step_name, step_id)

def log_step_performance(step_name: str, step_id: int, timing_data: Dict[str, Any]) -> None:
    """Step 성능 로그 출력"""
    logger.info(f"🎯 [{step_name}] Step {step_id} 성능 요약:")
    logger.info(f"   📊 총 시간: {timing_data.get('total_time', 0):.3f}초")
    
    stage_times = timing_data.get('stage_times', {})
    for stage, time_taken in stage_times.items():
        logger.info(f"   📊 {stage}: {time_taken:.3f}초")
        
    logger.info(f"   🖥️ 디바이스: {timing_data.get('device_used', 'unknown')}")
    logger.info(f"   🧠 모델: {timing_data.get('model_used', 'unknown')}")

# 전역 성능 모니터 인스턴스
global_performance_monitor = StepPerformanceMonitor()

def get_global_performance_monitor() -> StepPerformanceMonitor:
    """전역 성능 모니터 반환"""
    return global_performance_monitor 