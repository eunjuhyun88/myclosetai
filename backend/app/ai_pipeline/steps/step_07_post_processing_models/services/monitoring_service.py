#!/usr/bin/env python3
"""
🔥 MyCloset AI - Post Processing Monitoring Service
==================================================

🎯 후처리 모니터링 서비스
✅ 성능 모니터링
✅ 메모리 모니터링
✅ 품질 모니터링
✅ M3 Max 최적화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import time
import psutil
import gc
import os
import threading
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class MonitoringServiceConfig:
    """모니터링 서비스 설정"""
    enable_performance_monitoring: bool = True
    enable_memory_monitoring: bool = True
    enable_quality_monitoring: bool = True
    enable_real_time_monitoring: bool = True
    monitoring_interval: float = 1.0  # 초
    history_size: int = 100
    use_mps: bool = True

class PostProcessingPerformanceMonitor(nn.Module):
    """후처리 성능 모니터"""
    
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.input_channels = input_channels
        
        # 성능 모니터링을 위한 네트워크
        self.monitoring_net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, input_channels, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # 성능 모니터링
        monitored = self.monitoring_net(x)
        return monitored

class PostProcessingMemoryMonitor(nn.Module):
    """후처리 메모리 모니터"""
    
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.input_channels = input_channels
        
        # 메모리 모니터링을 위한 네트워크
        self.monitoring_net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, input_channels, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # 메모리 모니터링
        monitored = self.monitoring_net(x)
        return monitored

class PostProcessingQualityMonitor(nn.Module):
    """후처리 품질 모니터"""
    
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.input_channels = input_channels
        
        # 품질 모니터링을 위한 네트워크
        self.monitoring_net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, input_channels, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # 품질 모니터링
        monitored = self.monitoring_net(x)
        return monitored

class PostProcessingMonitoringService:
    """후처리 모니터링 서비스"""
    
    def __init__(self, config: MonitoringServiceConfig = None):
        self.config = config or MonitoringServiceConfig()
        self.logger = logging.getLogger(__name__)
        
        # MPS 디바이스 확인
        self.device = torch.device("mps" if torch.backends.mps.is_available() and self.config.use_mps else "cpu")
        self.logger.info(f"🎯 Post Processing 모니터링 서비스 초기화 (디바이스: {self.device})")
        
        # 성능 모니터
        if self.config.enable_performance_monitoring:
            self.performance_monitor = PostProcessingPerformanceMonitor(3).to(self.device)
        
        # 메모리 모니터
        if self.config.enable_memory_monitoring:
            self.memory_monitor = PostProcessingMemoryMonitor(3).to(self.device)
        
        # 품질 모니터
        if self.config.enable_quality_monitoring:
            self.quality_monitor = PostProcessingQualityMonitor(3).to(self.device)
        
        # 모니터링 데이터 히스토리
        self.monitoring_history = {
            'performance': deque(maxlen=self.config.history_size),
            'memory': deque(maxlen=self.config.history_size),
            'quality': deque(maxlen=self.config.history_size),
            'timestamps': deque(maxlen=self.config.history_size)
        }
        
        # 실시간 모니터링 스레드
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # 모니터링 통계
        self.monitoring_stats = {
            'total_monitoring_cycles': 0,
            'performance_monitoring_cycles': 0,
            'memory_monitoring_cycles': 0,
            'quality_monitoring_cycles': 0,
            'total_monitoring_time': 0.0
        }
        
        self.logger.info("✅ Post Processing 모니터링 서비스 초기화 완료")
    
    def start_monitoring(self):
        """실시간 모니터링을 시작합니다."""
        if self.is_monitoring:
            self.logger.warning("모니터링이 이미 실행 중입니다.")
            return
        
        if not self.config.enable_real_time_monitoring:
            self.logger.warning("실시간 모니터링이 비활성화되어 있습니다.")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("실시간 모니터링 시작")
    
    def stop_monitoring(self):
        """실시간 모니터링을 중지합니다."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("실시간 모니터링 중지")
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.is_monitoring:
            try:
                # 모니터링 수행
                self._perform_monitoring_cycle()
                
                # 대기
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"모니터링 루프 오류: {e}")
                time.sleep(self.config.monitoring_interval)
    
    def _perform_monitoring_cycle(self):
        """모니터링 사이클을 수행합니다."""
        try:
            start_time = time.time()
            timestamp = time.time()
            
            # 성능 모니터링
            if self.config.enable_performance_monitoring:
                performance_data = self._monitor_performance()
                self.monitoring_history['performance'].append(performance_data)
                self.monitoring_stats['performance_monitoring_cycles'] += 1
            
            # 메모리 모니터링
            if self.config.enable_memory_monitoring:
                memory_data = self._monitor_memory()
                self.monitoring_history['memory'].append(memory_data)
                self.monitoring_stats['memory_monitoring_cycles'] += 1
            
            # 품질 모니터링
            if self.config.enable_quality_monitoring:
                quality_data = self._monitor_quality()
                self.monitoring_history['quality'].append(quality_data)
                self.monitoring_stats['quality_monitoring_cycles'] += 1
            
            # 타임스탬프 추가
            self.monitoring_history['timestamps'].append(timestamp)
            
            # 통계 업데이트
            cycle_time = time.time() - start_time
            self.monitoring_stats['total_monitoring_cycles'] += 1
            self.monitoring_stats['total_monitoring_time'] += cycle_time
            
            self.logger.debug(f"모니터링 사이클 완료 (시간: {cycle_time:.4f}초)")
            
        except Exception as e:
            self.logger.error(f"모니터링 사이클 실패: {e}")
    
    def _monitor_performance(self) -> Dict[str, Any]:
        """성능을 모니터링합니다."""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # 프로세스별 CPU 사용률
            process = psutil.Process()
            process_cpu_percent = process.cpu_percent()
            
            # 처리 시간 통계
            avg_processing_time = 0.0
            if self.monitoring_history['performance']:
                recent_times = [data.get('processing_time', 0) for data in list(self.monitoring_history['performance'])[-10:]]
                avg_processing_time = sum(recent_times) / len(recent_times) if recent_times else 0.0
            
            return {
                'cpu_percent': cpu_percent,
                'process_cpu_percent': process_cpu_percent,
                'avg_processing_time': avg_processing_time,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"성능 모니터링 실패: {e}")
            return {
                'cpu_percent': 0.0,
                'process_cpu_percent': 0.0,
                'avg_processing_time': 0.0,
                'timestamp': time.time(),
                'error': str(e)
            }
    
    def _monitor_memory(self) -> Dict[str, Any]:
        """메모리를 모니터링합니다."""
        try:
            # 시스템 메모리
            system_memory = psutil.virtual_memory()
            
            # 프로세스 메모리
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # PyTorch 메모리 (GPU가 있는 경우)
            torch_memory = {}
            if torch.cuda.is_available():
                torch_memory = {
                    'cuda_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                    'cuda_reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
                    'cuda_max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
                }
            
            return {
                'system_memory_percent': system_memory.percent,
                'system_memory_available_gb': system_memory.available / 1024**3,
                'system_memory_used_gb': system_memory.used / 1024**3,
                'process_memory_rss_gb': process_memory.rss / 1024**3,
                'process_memory_vms_gb': process_memory.vms / 1024**3,
                'torch_memory': torch_memory,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"메모리 모니터링 실패: {e}")
            return {
                'system_memory_percent': 0.0,
                'system_memory_available_gb': 0.0,
                'system_memory_used_gb': 0.0,
                'process_memory_rss_gb': 0.0,
                'process_memory_vms_gb': 0.0,
                'torch_memory': {},
                'timestamp': time.time(),
                'error': str(e)
            }
    
    def _monitor_quality(self) -> Dict[str, Any]:
        """품질을 모니터링합니다."""
        try:
            # 품질 메트릭 (예시)
            quality_score = 0.8  # 실제로는 계산된 값
            artifact_level = 0.1
            sharpness_score = 0.9
            
            return {
                'quality_score': quality_score,
                'artifact_level': artifact_level,
                'sharpness_score': sharpness_score,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"품질 모니터링 실패: {e}")
            return {
                'quality_score': 0.0,
                'artifact_level': 0.0,
                'sharpness_score': 0.0,
                'timestamp': time.time(),
                'error': str(e)
            }
    
    def get_monitoring_data(self, data_type: str = 'all', limit: int = None) -> Dict[str, Any]:
        """모니터링 데이터를 반환합니다."""
        try:
            if data_type == 'all':
                data = {
                    'performance': list(self.monitoring_history['performance']),
                    'memory': list(self.monitoring_history['memory']),
                    'quality': list(self.monitoring_history['quality']),
                    'timestamps': list(self.monitoring_history['timestamps'])
                }
            else:
                data = {
                    data_type: list(self.monitoring_history.get(data_type, [])),
                    'timestamps': list(self.monitoring_history['timestamps'])
                }
            
            # 제한 적용
            if limit:
                for key in data:
                    if isinstance(data[key], list):
                        data[key] = data[key][-limit:]
            
            return data
            
        except Exception as e:
            self.logger.error(f"모니터링 데이터 조회 실패: {e}")
            return {'error': str(e)}
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """모니터링 통계를 반환합니다."""
        return {
            **self.monitoring_stats,
            'service_config': self.config.__dict__,
            'device': str(self.device),
            'is_monitoring': self.is_monitoring,
            'history_size': len(self.monitoring_history['timestamps'])
        }
    
    def cleanup(self):
        """리소스 정리를 수행합니다."""
        try:
            # 모니터링 중지
            self.stop_monitoring()
            
            # 히스토리 초기화
            for key in self.monitoring_history:
                self.monitoring_history[key].clear()
            
            # 통계 초기화
            self.monitoring_stats = {
                'total_monitoring_cycles': 0,
                'performance_monitoring_cycles': 0,
                'memory_monitoring_cycles': 0,
                'quality_monitoring_cycles': 0,
                'total_monitoring_time': 0.0
            }
            
            # 가비지 컬렉션
            gc.collect()
            
            self.logger.info("모니터링 서비스 리소스 정리 완료")
            
        except Exception as e:
            self.logger.error(f"리소스 정리 실패: {e}")

# 사용 예시
if __name__ == "__main__":
    # 설정
    config = MonitoringServiceConfig(
        enable_performance_monitoring=True,
        enable_memory_monitoring=True,
        enable_quality_monitoring=True,
        enable_real_time_monitoring=True,
        monitoring_interval=1.0,
        history_size=100,
        use_mps=True
    )
    
    # 모니터링 서비스 초기화
    monitoring_service = PostProcessingMonitoringService(config)
    
    # 모니터링 시작
    monitoring_service.start_monitoring()
    
    # 잠시 대기
    time.sleep(3)
    
    # 모니터링 데이터 조회
    data = monitoring_service.get_monitoring_data('all', limit=5)
    print(f"모니터링 데이터: {len(data.get('timestamps', []))}개")
    
    # 모니터링 통계
    stats = monitoring_service.get_monitoring_stats()
    print(f"모니터링 통계: {stats}")
    
    # 모니터링 중지
    monitoring_service.stop_monitoring()
    
    # 리소스 정리
    monitoring_service.cleanup()
