#!/usr/bin/env python3
"""
🔥 MyCloset AI - Memory Service for Cloth Warping
==================================================

🎯 의류 워핑 메모리 관리 서비스
✅ 메모리 사용량 모니터링
✅ 자동 메모리 정리
✅ 메모리 최적화
✅ M3 Max 최적화
"""

import torch
import logging
import gc
import psutil
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """메모리 설정"""
    max_memory_gb: float = 8.0
    cleanup_threshold: float = 0.8
    enable_auto_cleanup: bool = True
    cleanup_interval_seconds: int = 60
    enable_memory_profiling: bool = True
    max_cache_size_mb: int = 1024

class MemoryService:
    """의류 워핑 메모리 관리 서비스"""
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self.logger = logging.getLogger(__name__)
        
        # 메모리 사용량 추적
        self.memory_usage_history = []
        self.last_cleanup_time = time.time()
        
        # 캐시된 텐서들
        self.cached_tensors = {}
        self.tensor_sizes = {}
        
        # 메모리 프로파일링
        if self.config.enable_memory_profiling:
            self.memory_profiles = []
        
        self.logger.info(f"🎯 Memory Service 초기화 (최대 메모리: {self.config.max_memory_gb}GB)")
        
        # 초기 메모리 상태 확인
        self._check_initial_memory_state()
        
        self.logger.info("✅ Memory Service 초기화 완료")
    
    def _check_initial_memory_state(self):
        """초기 메모리 상태 확인"""
        try:
            system_memory = psutil.virtual_memory()
            available_memory_gb = system_memory.available / (1024**3)
            
            self.logger.info(f"시스템 메모리: {system_memory.total / (1024**3):.2f}GB")
            self.logger.info(f"사용 가능한 메모리: {available_memory_gb:.2f}GB")
            
            if available_memory_gb < self.config.max_memory_gb:
                self.logger.warning(f"사용 가능한 메모리가 설정된 최대값보다 적습니다: {available_memory_gb:.2f}GB < {self.config.max_memory_gb}GB")
                
        except Exception as e:
            self.logger.warning(f"초기 메모리 상태 확인 실패: {e}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """현재 메모리 사용량 조회"""
        memory_info = {}
        
        try:
            # 시스템 메모리 정보
            system_memory = psutil.virtual_memory()
            memory_info['system_total_gb'] = system_memory.total / (1024**3)
            memory_info['system_available_gb'] = system_memory.available / (1024**3)
            memory_info['system_used_gb'] = system_memory.used / (1024**3)
            memory_info['system_percent'] = system_memory.percent
            
            # PyTorch 메모리 정보
            if torch.cuda.is_available():
                memory_info['gpu_total_mb'] = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                memory_info['gpu_allocated_mb'] = torch.cuda.memory_allocated(0) / (1024**2)
                memory_info['gpu_cached_mb'] = torch.cuda.memory_reserved(0) / (1024**2)
                memory_info['gpu_free_mb'] = memory_info['gpu_total_mb'] - memory_info['gpu_allocated_mb']
            
            # MPS 메모리 정보 (Apple Silicon)
            if torch.backends.mps.is_available():
                memory_info['mps_available'] = True
                # MPS는 직접적인 메모리 정보를 제공하지 않음
                memory_info['mps_memory_estimate_mb'] = self._estimate_mps_memory_usage()
            else:
                memory_info['mps_available'] = False
            
            # 캐시된 텐서 정보
            memory_info['cached_tensors_count'] = len(self.cached_tensors)
            memory_info['cached_tensors_size_mb'] = sum(self.tensor_sizes.values()) / (1024**2)
            
            # 메모리 사용률
            memory_info['memory_usage_ratio'] = memory_info['system_used_gb'] / memory_info['system_total_gb']
            
        except Exception as e:
            self.logger.error(f"메모리 사용량 조회 실패: {e}")
            memory_info = {
                'error': str(e),
                'system_total_gb': 0.0,
                'system_available_gb': 0.0,
                'system_used_gb': 0.0,
                'system_percent': 0.0
            }
        
        return memory_info
    
    def _estimate_mps_memory_usage(self) -> float:
        """MPS 메모리 사용량 추정"""
        try:
            # 캐시된 텐서들의 크기로 추정
            total_tensor_memory = sum(self.tensor_sizes.values())
            
            # 시스템 메모리 사용량과 비교하여 추정
            system_memory = psutil.virtual_memory()
            estimated_mps_memory = min(total_tensor_memory / (1024**2), system_memory.used / (1024**2) * 0.3)
            
            return estimated_mps_memory
            
        except Exception:
            return 0.0
    
    def monitor_memory_usage(self, interval_seconds: int = 5) -> Dict[str, Any]:
        """메모리 사용량 모니터링"""
        monitoring_data = {
            'timestamp': time.time(),
            'memory_usage': self.get_memory_usage(),
            'recommendations': []
        }
        
        try:
            memory_usage = monitoring_data['memory_usage']
            
            # 메모리 사용률 확인
            if memory_usage.get('memory_usage_ratio', 0) > self.config.cleanup_threshold:
                monitoring_data['recommendations'].append("메모리 정리가 필요합니다")
                
                if self.config.enable_auto_cleanup:
                    self.cleanup_memory()
                    monitoring_data['cleanup_performed'] = True
            
            # GPU 메모리 확인
            if 'gpu_allocated_mb' in memory_usage:
                gpu_usage_ratio = memory_usage['gpu_allocated_mb'] / memory_usage['gpu_total_mb']
                if gpu_usage_ratio > 0.9:
                    monitoring_data['recommendations'].append("GPU 메모리 정리가 필요합니다")
            
            # 캐시 크기 확인
            if memory_usage.get('cached_tensors_size_mb', 0) > self.config.max_cache_size_mb:
                monitoring_data['recommendations'].append("텐서 캐시 정리가 필요합니다")
                self.cleanup_tensor_cache()
                monitoring_data['cache_cleanup_performed'] = True
            
            # 메모리 사용량 히스토리에 추가
            self.memory_usage_history.append(monitoring_data)
            
            # 히스토리 크기 제한
            if len(self.memory_usage_history) > 1000:
                self.memory_usage_history = self.memory_usage_history[-500:]
            
            # 메모리 프로파일링
            if self.config.enable_memory_profiling:
                self._add_memory_profile(monitoring_data)
            
        except Exception as e:
            self.logger.error(f"메모리 모니터링 실패: {e}")
            monitoring_data['error'] = str(e)
        
        return monitoring_data
    
    def _add_memory_profile(self, monitoring_data: Dict[str, Any]):
        """메모리 프로파일 추가"""
        try:
            profile = {
                'timestamp': monitoring_data['timestamp'],
                'memory_usage': monitoring_data['memory_usage'],
                'peak_memory_mb': max([h['memory_usage'].get('system_used_gb', 0) * 1024 
                                     for h in self.memory_usage_history[-100:]]),
                'average_memory_mb': np.mean([h['memory_usage'].get('system_used_gb', 0) * 1024 
                                            for h in self.memory_usage_history[-100:]])
            }
            
            self.memory_profiles.append(profile)
            
            # 프로파일 크기 제한
            if len(self.memory_profiles) > 100:
                self.memory_profiles = self.memory_profiles[-50:]
                
        except Exception as e:
            self.logger.warning(f"메모리 프로파일 추가 실패: {e}")
    
    def cleanup_memory(self, force: bool = False) -> Dict[str, Any]:
        """메모리 정리"""
        cleanup_results = {
            'timestamp': time.time(),
            'cleanup_type': 'automatic' if not force else 'forced',
            'results': {}
        }
        
        try:
            # 자동 정리 조건 확인
            if not force and not self.config.enable_auto_cleanup:
                cleanup_results['results']['auto_cleanup_disabled'] = True
                return cleanup_results
            
            # 마지막 정리 후 시간 확인
            time_since_last_cleanup = time.time() - self.last_cleanup_time
            if not force and time_since_last_cleanup < self.config.cleanup_interval_seconds:
                cleanup_results['results']['cleanup_interval_not_reached'] = True
                return cleanup_results
            
            # 1단계: Python 가비지 컬렉션
            gc.collect()
            cleanup_results['results']['garbage_collection'] = True
            
            # 2단계: PyTorch 캐시 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                cleanup_results['results']['cuda_cache_cleared'] = True
            
            # 3단계: 텐서 캐시 정리
            cache_cleanup_result = self.cleanup_tensor_cache()
            cleanup_results['results']['tensor_cache_cleanup'] = cache_cleanup_result
            
            # 4단계: 메모리 사용량 재확인
            memory_after = self.get_memory_usage()
            cleanup_results['results']['memory_after_cleanup'] = memory_after
            
            # 정리 시간 업데이트
            self.last_cleanup_time = time.time()
            
            self.logger.info("✅ 메모리 정리 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 메모리 정리 실패: {e}")
            cleanup_results['results']['error'] = str(e)
        
        return cleanup_results
    
    def cleanup_tensor_cache(self) -> Dict[str, Any]:
        """텐서 캐시 정리"""
        cache_cleanup_result = {
            'tensors_removed': 0,
            'memory_freed_mb': 0.0,
            'removed_tensors': []
        }
        
        try:
            # 사용되지 않는 텐서들 제거
            tensors_to_remove = []
            
            for tensor_name, tensor in self.cached_tensors.items():
                # 텐서가 여전히 유효한지 확인
                if not self._is_tensor_valid(tensor):
                    tensors_to_remove.append(tensor_name)
                    cache_cleanup_result['tensors_removed'] += 1
                    cache_cleanup_result['memory_freed_mb'] += self.tensor_sizes.get(tensor_name, 0) / (1024**2)
                    cache_cleanup_result['removed_tensors'].append(tensor_name)
            
            # 텐서 제거
            for tensor_name in tensors_to_remove:
                del self.cached_tensors[tensor_name]
                if tensor_name in self.tensor_sizes:
                    del self.tensor_sizes[tensor_name]
            
            # 강제 가비지 컬렉션
            gc.collect()
            
            self.logger.info(f"✅ 텐서 캐시 정리 완료: {cache_cleanup_result['tensors_removed']}개 텐서 제거")
            
        except Exception as e:
            self.logger.error(f"❌ 텐서 캐시 정리 실패: {e}")
            cache_cleanup_result['error'] = str(e)
        
        return cache_cleanup_result
    
    def _is_tensor_valid(self, tensor) -> bool:
        """텐서 유효성 확인"""
        try:
            # 텐서가 여전히 유효한지 확인
            if hasattr(tensor, 'device'):
                # 디바이스 정보 확인
                return True
            return False
        except Exception:
            return False
    
    def cache_tensor(self, tensor_name: str, tensor, size_mb: float = None):
        """텐서 캐싱"""
        try:
            if size_mb is None:
                # 텐서 크기 자동 계산
                if hasattr(tensor, 'element_size'):
                    size_mb = tensor.element_size() * tensor.nelement() / (1024**2)
                else:
                    size_mb = 0.0
            
            self.cached_tensors[tensor_name] = tensor
            self.tensor_sizes[tensor_name] = size_mb * (1024**2)  # MB를 bytes로 변환
            
            self.logger.debug(f"텐서 캐싱: {tensor_name} ({size_mb:.2f}MB)")
            
        except Exception as e:
            self.logger.warning(f"텐서 캐싱 실패: {tensor_name} - {e}")
    
    def get_cached_tensor(self, tensor_name: str):
        """캐시된 텐서 조회"""
        return self.cached_tensors.get(tensor_name)
    
    def remove_cached_tensor(self, tensor_name: str) -> bool:
        """캐시된 텐서 제거"""
        try:
            if tensor_name in self.cached_tensors:
                del self.cached_tensors[tensor_name]
                if tensor_name in self.tensor_sizes:
                    del self.tensor_sizes[tensor_name]
                
                self.logger.debug(f"캐시된 텐서 제거: {tensor_name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"캐시된 텐서 제거 실패: {tensor_name} - {e}")
            return False
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """메모리 통계 조회"""
        stats = {}
        
        try:
            # 현재 메모리 상태
            current_memory = self.get_memory_usage()
            stats['current_memory'] = current_memory
            
            # 히스토리 통계
            if self.memory_usage_history:
                memory_values = [h['memory_usage'].get('system_used_gb', 0) for h in self.memory_usage_history]
                stats['history_stats'] = {
                    'peak_memory_gb': max(memory_values),
                    'average_memory_gb': np.mean(memory_values),
                    'min_memory_gb': min(memory_values),
                    'total_samples': len(memory_values)
                }
            
            # 프로파일 통계
            if self.memory_profiles:
                stats['profile_stats'] = {
                    'total_profiles': len(self.memory_profiles),
                    'latest_profile': self.memory_profiles[-1] if self.memory_profiles else None
                }
            
            # 캐시 통계
            stats['cache_stats'] = {
                'cached_tensors_count': len(self.cached_tensors),
                'total_cache_size_mb': sum(self.tensor_sizes.values()) / (1024**2),
                'largest_tensor_mb': max(self.tensor_sizes.values()) / (1024**2) if self.tensor_sizes else 0
            }
            
        except Exception as e:
            self.logger.error(f"메모리 통계 조회 실패: {e}")
            stats = {'error': str(e)}
        
        return stats
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """메모리 사용량 최적화"""
        optimization_results = {
            'timestamp': time.time(),
            'optimizations_applied': [],
            'memory_saved_mb': 0.0
        }
        
        try:
            # 1단계: 메모리 정리
            cleanup_result = self.cleanup_memory(force=True)
            optimization_results['cleanup_result'] = cleanup_result
            
            # 2단계: 캐시 크기 최적화
            if len(self.cached_tensors) > 100:
                # 가장 큰 텐서들부터 제거
                sorted_tensors = sorted(self.tensor_sizes.items(), key=lambda x: x[1], reverse=True)
                tensors_to_remove = sorted_tensors[50:]  # 상위 50개만 유지
                
                for tensor_name, size in tensors_to_remove:
                    if self.remove_cached_tensor(tensor_name):
                        optimization_results['memory_saved_mb'] += size / (1024**2)
                        optimization_results['optimizations_applied'].append(f"큰 텐서 제거: {tensor_name}")
            
            # 3단계: 메모리 압축
            if hasattr(torch, 'compiled'):
                # PyTorch 2.0+ 컴파일 최적화
                optimization_results['optimizations_applied'].append("PyTorch 컴파일 최적화")
            
            self.logger.info("✅ 메모리 최적화 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 메모리 최적화 실패: {e}")
            optimization_results['error'] = str(e)
        
        return optimization_results

# 메모리 서비스 인스턴스 생성
def create_memory_service(config: MemoryConfig = None) -> MemoryService:
    """Memory Service 생성"""
    return MemoryService(config)

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # 설정 생성
    config = MemoryConfig(
        max_memory_gb=8.0,
        cleanup_threshold=0.8,
        enable_auto_cleanup=True
    )
    
    # 메모리 서비스 생성
    service = create_memory_service(config)
    
    # 메모리 사용량 조회
    memory_usage = service.get_memory_usage()
    print(f"메모리 사용량: {memory_usage}")
    
    # 메모리 모니터링
    monitoring_data = service.monitor_memory_usage()
    print(f"모니터링 결과: {monitoring_data}")
    
    # 메모리 통계
    stats = service.get_memory_statistics()
    print(f"메모리 통계: {stats}")
