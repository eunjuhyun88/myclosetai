"""
Performance Utilities
성능 모니터링과 최적화를 위한 유틸리티 함수들을 제공하는 클래스
"""

import torch
import torch.nn as nn
import time
import psutil
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from contextlib import contextmanager
import functools
import gc

# 프로젝트 로깅 설정 import
from backend.app.ai_pipeline.utils.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class PerformanceMetrics:
    """성능 메트릭을 저장하는 데이터 클래스"""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float]
    throughput: float
    latency: float

class PerformanceUtils:
    """
    성능 모니터링과 최적화를 위한 유틸리티 함수들을 제공하는 클래스
    """
    
    def __init__(self):
        """성능 유틸리티 초기화"""
        self.performance_history = []
        self.monitoring_enabled = True
        
        logger.info("PerformanceUtils initialized")
    
    @contextmanager
    def performance_monitor(self, operation_name: str = "operation"):
        """
        성능 모니터링 컨텍스트 매니저
        
        Args:
            operation_name: 모니터링할 작업 이름
        """
        if not self.monitoring_enabled:
            yield
            return
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = psutil.cpu_percent()
        start_gpu = self._get_gpu_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu = psutil.cpu_percent()
            end_gpu = self._get_gpu_usage()
            
            # 성능 메트릭 계산
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            cpu_usage = (start_cpu + end_cpu) / 2
            gpu_usage = (start_gpu + end_gpu) / 2 if start_gpu is not None and end_gpu is not None else None
            
            # 메트릭 저장
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage_mb=memory_delta,
                cpu_usage_percent=cpu_usage,
                gpu_usage_percent=gpu_usage,
                throughput=1.0 / execution_time if execution_time > 0 else 0.0,
                latency=execution_time * 1000  # ms 단위
            )
            
            self.performance_history.append({
                'operation': operation_name,
                'timestamp': time.time(),
                'metrics': metrics
            })
            
            # 로그 출력
            logger.info(f"성능 모니터링 - {operation_name}: "
                       f"실행시간: {execution_time:.3f}s, "
                       f"메모리 변화: {memory_delta:.2f}MB, "
                       f"CPU: {cpu_usage:.1f}%")
    
    def monitor_function(self, operation_name: str = None):
        """
        함수 성능 모니터링 데코레이터
        
        Args:
            operation_name: 모니터링할 작업 이름
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                name = operation_name or func.__name__
                with self.performance_monitor(name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def benchmark_function(self, func: Callable, 
                         num_runs: int = 100,
                         warmup_runs: int = 10,
                         *args, **kwargs) -> Dict[str, Any]:
        """
        함수 성능 벤치마크
        
        Args:
            func: 벤치마크할 함수
            num_runs: 실행 횟수
            warmup_runs: 워밍업 실행 횟수
            *args, **kwargs: 함수 인자들
            
        Returns:
            벤치마크 결과
        """
        try:
            logger.info(f"함수 벤치마크 시작: {func.__name__}")
            
            # 워밍업
            for _ in range(warmup_runs):
                func(*args, **kwargs)
            
            # CUDA 동기화
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # 벤치마크 실행
            execution_times = []
            memory_usages = []
            
            for i in range(num_runs):
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                result = func(*args, **kwargs)
                
                end_time = time.time()
                end_memory = self._get_memory_usage()
                
                execution_time = end_time - start_time
                memory_usage = end_memory - start_memory
                
                execution_times.append(execution_time)
                memory_usages.append(memory_usage)
                
                if (i + 1) % 10 == 0:
                    logger.debug(f"벤치마크 진행률: {i + 1}/{num_runs}")
            
            # CUDA 동기화
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # 통계 계산
            avg_execution_time = sum(execution_times) / len(execution_times)
            min_execution_time = min(execution_times)
            max_execution_time = max(execution_times)
            std_execution_time = self._calculate_std(execution_times)
            
            avg_memory_usage = sum(memory_usages) / len(memory_usages)
            max_memory_usage = max(memory_usages)
            
            throughput = 1.0 / avg_execution_time if avg_execution_time > 0 else 0.0
            
            benchmark_results = {
                'function_name': func.__name__,
                'num_runs': num_runs,
                'warmup_runs': warmup_runs,
                'execution_time': {
                    'average': avg_execution_time,
                    'min': min_execution_time,
                    'max': max_execution_time,
                    'std': std_execution_time,
                    'all_times': execution_times
                },
                'memory_usage': {
                    'average_mb': avg_memory_usage,
                    'max_mb': max_memory_usage,
                    'all_usages': memory_usages
                },
                'throughput': throughput,
                'latency_ms': avg_execution_time * 1000
            }
            
            logger.info(f"함수 벤치마크 완료: {func.__name__} - "
                       f"평균 실행시간: {avg_execution_time*1000:.2f}ms, "
                       f"처리량: {throughput:.2f} ops/s")
            
            return benchmark_results
            
        except Exception as e:
            logger.error(f"함수 벤치마크 중 오류 발생: {e}")
            return {'error': str(e)}
    
    def profile_memory_usage(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        함수의 메모리 사용량 프로파일링
        
        Args:
            func: 프로파일링할 함수
            *args, **kwargs: 함수 인자들
            
        Returns:
            메모리 프로파일링 결과
        """
        try:
            logger.info(f"메모리 프로파일링 시작: {func.__name__}")
            
            # 초기 메모리 상태
            initial_memory = self._get_memory_usage()
            initial_gpu_memory = self._get_gpu_memory_usage()
            
            # 가비지 컬렉션
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 함수 실행
            start_memory = self._get_memory_usage()
            start_gpu_memory = self._get_gpu_memory_usage()
            
            result = func(*args, **kwargs)
            
            end_memory = self._get_memory_usage()
            end_gpu_memory = self._get_gpu_memory_usage()
            
            # 가비지 컬렉션
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            final_memory = self._get_memory_usage()
            final_gpu_memory = self._get_gpu_memory_usage()
            
            # 메모리 분석
            peak_memory = max(start_memory, end_memory)
            memory_increase = end_memory - start_memory
            memory_leak = final_memory - initial_memory
            
            gpu_memory_increase = 0
            gpu_memory_leak = 0
            if start_gpu_memory is not None and end_gpu_memory is not None:
                gpu_memory_increase = end_gpu_memory - start_gpu_memory
                if final_gpu_memory is not None:
                    gpu_memory_leak = final_gpu_memory - initial_gpu_memory
            
            profile_results = {
                'function_name': func.__name__,
                'cpu_memory': {
                    'initial_mb': initial_memory,
                    'start_mb': start_memory,
                    'peak_mb': peak_memory,
                    'end_mb': end_memory,
                    'final_mb': final_memory,
                    'increase_mb': memory_increase,
                    'leak_mb': memory_leak
                },
                'gpu_memory': {
                    'initial_mb': initial_gpu_memory,
                    'start_mb': start_gpu_memory,
                    'end_mb': end_gpu_memory,
                    'final_mb': final_gpu_memory,
                    'increase_mb': gpu_memory_increase,
                    'leak_mb': gpu_memory_leak
                }
            }
            
            logger.info(f"메모리 프로파일링 완료: {func.__name__} - "
                       f"CPU 메모리 증가: {memory_increase:.2f}MB, "
                       f"GPU 메모리 증가: {gpu_memory_increase:.2f}MB")
            
            return profile_results
            
        except Exception as e:
            logger.error(f"메모리 프로파일링 중 오류 발생: {e}")
            return {'error': str(e)}
    
    def optimize_memory(self):
        """메모리 최적화 실행"""
        try:
            logger.info("메모리 최적화 시작")
            
            # 가비지 컬렉션
            collected = gc.collect()
            logger.debug(f"가비지 컬렉션 완료: {collected} 객체 수집")
            
            # CUDA 캐시 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA 캐시 정리 완료")
            
            # 메모리 상태 확인
            current_memory = self._get_memory_usage()
            current_gpu_memory = self._get_gpu_memory_usage()
            
            optimization_results = {
                'cpu_memory_mb': current_memory,
                'gpu_memory_mb': current_gpu_memory,
                'garbage_collected': collected
            }
            
            logger.info(f"메모리 최적화 완료 - "
                       f"CPU 메모리: {current_memory:.2f}MB, "
                       f"GPU 메모리: {current_gpu_memory:.2f}MB")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"메모리 최적화 중 오류 발생: {e}")
            return {'error': str(e)}
    
    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 반환"""
        try:
            system_info = {
                'cpu': {
                    'count': psutil.cpu_count(),
                    'count_logical': psutil.cpu_count(logical=True),
                    'usage_percent': psutil.cpu_percent(interval=1),
                    'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
                },
                'memory': {
                    'total_gb': psutil.virtual_memory().total / (1024**3),
                    'available_gb': psutil.virtual_memory().available / (1024**3),
                    'used_gb': psutil.virtual_memory().used / (1024**3),
                    'percent': psutil.virtual_memory().percent
                },
                'disk': {
                    'total_gb': psutil.disk_usage('/').total / (1024**3),
                    'used_gb': psutil.disk_usage('/').used / (1024**3),
                    'free_gb': psutil.disk_usage('/').free / (1024**3),
                    'percent': psutil.disk_usage('/').percent
                }
            }
            
            # GPU 정보 추가
            if torch.cuda.is_available():
                gpu_info = {
                    'device_count': torch.cuda.device_count(),
                    'current_device': torch.cuda.current_device(),
                    'device_name': torch.cuda.get_device_name(),
                    'memory_allocated_mb': torch.cuda.memory_allocated() / (1024**2),
                    'memory_reserved_mb': torch.cuda.memory_reserved() / (1024**2),
                    'memory_total_mb': torch.cuda.get_device_properties(0).total_memory / (1024**2)
                }
                system_info['gpu'] = gpu_info
            
            return system_info
            
        except Exception as e:
            logger.error(f"시스템 정보 조회 중 오류 발생: {e}")
            return {'error': str(e)}
    
    def get_performance_summary(self, operation_filter: Optional[str] = None) -> Dict[str, Any]:
        """성능 요약 정보 반환"""
        try:
            if not self.performance_history:
                return {'message': '성능 기록이 없습니다'}
            
            # 필터링
            filtered_history = self.performance_history
            if operation_filter:
                filtered_history = [h for h in self.performance_history 
                                  if operation_filter in h['operation']]
            
            if not filtered_history:
                return {'message': f'필터 "{operation_filter}"에 맞는 성능 기록이 없습니다'}
            
            # 통계 계산
            total_operations = len(filtered_history)
            total_execution_time = sum(h['metrics'].execution_time for h in filtered_history)
            total_memory_usage = sum(h['metrics'].memory_usage_mb for h in filtered_history)
            
            avg_execution_time = total_execution_time / total_operations
            avg_memory_usage = total_memory_usage / total_operations
            
            # 작업별 통계
            operation_stats = {}
            for record in filtered_history:
                op_name = record['operation']
                if op_name not in operation_stats:
                    operation_stats[op_name] = {
                        'count': 0,
                        'total_time': 0.0,
                        'total_memory': 0.0,
                        'times': [],
                        'memory_usages': []
                    }
                
                stats = operation_stats[op_name]
                stats['count'] += 1
                stats['total_time'] += record['metrics'].execution_time
                stats['total_memory'] += record['metrics'].memory_usage_mb
                stats['times'].append(record['metrics'].execution_time)
                stats['memory_usages'].append(record['metrics'].memory_usage_mb)
            
            # 각 작업의 평균과 표준편차 계산
            for op_name, stats in operation_stats.items():
                stats['avg_time'] = stats['total_time'] / stats['count']
                stats['avg_memory'] = stats['total_memory'] / stats['count']
                stats['std_time'] = self._calculate_std(stats['times'])
                stats['std_memory'] = self._calculate_std(stats['memory_usages'])
                stats['min_time'] = min(stats['times'])
                stats['max_time'] = max(stats['times'])
            
            summary = {
                'total_operations': total_operations,
                'total_execution_time': total_execution_time,
                'total_memory_usage_mb': total_memory_usage,
                'overall_averages': {
                    'execution_time': avg_execution_time,
                    'memory_usage_mb': avg_memory_usage
                },
                'operation_statistics': operation_stats
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"성능 요약 생성 중 오류 발생: {e}")
            return {'error': str(e)}
    
    def clear_performance_history(self):
        """성능 기록 초기화"""
        try:
            self.performance_history.clear()
            logger.info("성능 기록 초기화 완료")
        except Exception as e:
            logger.error(f"성능 기록 초기화 중 오류 발생: {e}")
    
    def enable_monitoring(self, enabled: bool = True):
        """모니터링 활성화/비활성화"""
        self.monitoring_enabled = enabled
        status = "활성화" if enabled else "비활성화"
        logger.info(f"성능 모니터링 {status}")
    
    def _get_memory_usage(self) -> float:
        """현재 메모리 사용량 반환 (MB)"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # bytes to MB
        except Exception as e:
            logger.error(f"메모리 사용량 조회 중 오류 발생: {e}")
            return 0.0
    
    def _get_gpu_usage(self) -> Optional[float]:
        """현재 GPU 사용량 반환 (MB)"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)  # bytes to MB
            return None
        except Exception as e:
            logger.error(f"GPU 사용량 조회 중 오류 발생: {e}")
            return None
    
    def _get_gpu_memory_usage(self) -> Optional[float]:
        """현재 GPU 메모리 사용량 반환 (MB)"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_reserved() / (1024 * 1024)  # bytes to MB
            return None
        except Exception as e:
            logger.error(f"GPU 메모리 사용량 조회 중 오류 발생: {e}")
            return None
    
    def _calculate_std(self, values: List[float]) -> float:
        """표준편차 계산"""
        try:
            if len(values) < 2:
                return 0.0
            
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
            return variance ** 0.5
            
        except Exception as e:
            logger.error(f"표준편차 계산 중 오류 발생: {e}")
            return 0.0
    
    def export_performance_report(self, filepath: str, format_type: str = 'json') -> bool:
        """
        성능 보고서 내보내기
        
        Args:
            filepath: 저장 경로
            format_type: 형식 ('json', 'csv')
            
        Returns:
            내보내기 성공 여부
        """
        try:
            import json
            import csv
            
            summary = self.get_performance_summary()
            
            if format_type == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                    
            elif format_type == 'csv':
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # 헤더 작성
                    writer.writerow(['Operation', 'Count', 'Avg Time (s)', 'Avg Memory (MB)', 'Std Time', 'Std Memory'])
                    
                    # 데이터 작성
                    for op_name, stats in summary.get('operation_statistics', {}).items():
                        writer.writerow([
                            op_name,
                            stats['count'],
                            f"{stats['avg_time']:.6f}",
                            f"{stats['avg_memory']:.2f}",
                            f"{stats['std_time']:.6f}",
                            f"{stats['std_memory']:.2f}"
                        ])
            else:
                raise ValueError(f"지원하지 않는 형식: {format_type}")
            
            logger.info(f"성능 보고서 내보내기 완료: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"성능 보고서 내보내기 중 오류 발생: {e}")
            return False
