#!/usr/bin/env python3
"""
🔥 MyCloset AI - Performance Tracking Mixin
===========================================

성능 추적 및 메트릭 관련 기능을 담당하는 Mixin 클래스
처리 시간, 메모리 사용량, 성공률 등을 추적

Author: MyCloset AI Team
Date: 2025-08-14
Version: 2.0
"""

import time
import logging
from typing import Dict, Any, Optional

class PerformanceTrackingMixin:
    """성능 추적 관련 기능을 제공하는 Mixin"""
    
    def _initialize_performance_stats(self):
        """성능 통계 초기화"""
        self.performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'peak_memory_usage': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'last_request_time': None,
            'startup_time': time.time()
        }
        
        # 성능 메트릭 초기화
        if hasattr(self, 'performance_metrics'):
            self.performance_metrics.process_count = 0
            self.performance_metrics.total_process_time = 0.0
            self.performance_metrics.average_process_time = 0.0
            self.performance_metrics.error_count = 0
            self.performance_metrics.success_count = 0
            self.performance_metrics.cache_hits = 0

    def log_performance(self, processing_time: float, success: bool = True):
        """성능 로깅"""
        try:
            # 기본 성능 통계 업데이트
            self.performance_stats['total_requests'] += 1
            self.performance_stats['total_processing_time'] += processing_time
            
            if success:
                self.performance_stats['successful_requests'] += 1
            else:
                self.performance_stats['failed_requests'] += 1
            
            # 평균 처리 시간 계산
            total_requests = self.performance_stats['total_requests']
            if total_requests > 0:
                self.performance_stats['average_processing_time'] = (
                    self.performance_stats['total_processing_time'] / total_requests
                )
            
            self.performance_stats['last_request_time'] = time.time()
            
            # 성능 메트릭 업데이트
            if hasattr(self, 'performance_metrics'):
                self.performance_metrics.process_count += 1
                self.performance_metrics.total_process_time += processing_time
                self.performance_metrics.average_process_time = (
                    self.performance_metrics.total_process_time / self.performance_metrics.process_count
                )
                
                if success:
                    self.performance_metrics.success_count += 1
                else:
                    self.performance_metrics.error_count += 1
            
            # 로깅
            if hasattr(self, 'logger'):
                self.logger.debug(f"📊 성능 업데이트: {processing_time:.3f}s, 성공: {success}")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"성능 로깅 실패: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        try:
            # 현재 메모리 사용량 추가
            current_memory = self._get_current_memory_usage()
            if current_memory > self.performance_stats['peak_memory_usage']:
                self.performance_stats['peak_memory_usage'] = current_memory
            
            # 가동 시간 계산
            uptime = time.time() - self.performance_stats['startup_time']
            
            # 성공률 계산
            total_requests = self.performance_stats['total_requests']
            success_rate = 0.0
            if total_requests > 0:
                success_rate = (self.performance_stats['successful_requests'] / total_requests) * 100
            
            stats = self.performance_stats.copy()
            stats.update({
                'uptime_seconds': uptime,
                'success_rate_percent': success_rate,
                'current_memory_usage': current_memory
            })
            
            return stats
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"성능 통계 반환 실패: {e}")
            return {}

    def reset_performance_stats(self):
        """성능 통계 초기화"""
        try:
            self._initialize_performance_stats()
            if hasattr(self, 'logger'):
                self.logger.info("📊 성능 통계 초기화 완료")
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"성능 통계 초기화 실패: {e}")

    def _get_current_memory_usage(self) -> float:
        """현재 메모리 사용량 반환 (MB)"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # MB로 변환
        except ImportError:
            # psutil이 없는 경우 기본값 반환
            return 0.0
        except Exception:
            return 0.0

    def _update_performance_metrics(self, processing_time: float, success: bool):
        """성능 메트릭 업데이트"""
        try:
            if hasattr(self, 'performance_metrics'):
                self.performance_metrics.process_count += 1
                self.performance_metrics.total_process_time += processing_time
                
                if success:
                    self.performance_metrics.success_count += 1
                else:
                    self.performance_metrics.error_count += 1
                
                # 평균 처리 시간 업데이트
                if self.performance_metrics.process_count > 0:
                    self.performance_metrics.average_process_time = (
                        self.performance_metrics.total_process_time / self.performance_metrics.process_count
                    )
                    
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"성능 메트릭 업데이트 실패: {e}")

    def get_central_hub_stats(self) -> Dict[str, Any]:
        """Central Hub 통계 반환"""
        try:
            if hasattr(self, 'performance_metrics'):
                return {
                    'process_count': self.performance_metrics.process_count,
                    'success_count': self.performance_metrics.success_count,
                    'error_count': self.performance_metrics.error_count,
                    'average_process_time': self.performance_metrics.average_process_time,
                    'dependencies_injected': self.performance_metrics.dependencies_injected,
                    'injection_failures': self.performance_metrics.injection_failures,
                    'central_hub_requests': self.performance_metrics.central_hub_requests,
                    'service_resolutions': self.performance_metrics.service_resolutions
                }
            return {}
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Central Hub 통계 반환 실패: {e}")
            return {}
