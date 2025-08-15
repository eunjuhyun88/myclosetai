#!/usr/bin/env python3
"""
🔥 MyCloset AI - Cloth Warping Memory Service
=============================================

🎯 의류 워핑 메모리 서비스
✅ 메모리 사용량 모니터링
✅ 메모리 최적화
✅ 캐시 관리
✅ M3 Max 최적화
"""

import logging
import psutil
import torch
import gc
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class MemoryServiceConfig:
    """메모리 서비스 설정"""
    enable_memory_monitoring: bool = True
    enable_memory_optimization: bool = True
    enable_cache_management: bool = True
    memory_threshold: float = 0.8  # 80% 사용 시 경고
    cache_size_limit: int = 1024 * 1024 * 1024  # 1GB
    use_mps: bool = True

class ClothWarpingMemoryService:
    """의류 워핑 메모리 서비스"""
    
    def __init__(self, config: MemoryServiceConfig = None):
        self.config = config or MemoryServiceConfig()
        self.logger = logging.getLogger(__name__)
        self.logger.info("🎯 Cloth Warping 메모리 서비스 초기화")
        
        # 메모리 사용량 추적
        self.memory_usage_history = []
        self.cache_objects = {}
        self.last_cleanup_time = time.time()
        
        # MPS 디바이스 확인
        self.device = torch.device("mps" if torch.backends.mps.is_available() and self.config.use_mps else "cpu")
        
        self.logger.info("✅ Cloth Warping 메모리 서비스 초기화 완료")
    
    def get_system_memory_info(self) -> Dict[str, Any]:
        """시스템 메모리 정보를 반환합니다."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent': memory.percent,
                'free': memory.free
            }
        except Exception as e:
            self.logger.error(f"시스템 메모리 정보 조회 실패: {e}")
            return {}
    
    def get_torch_memory_info(self) -> Dict[str, Any]:
        """PyTorch 메모리 정보를 반환합니다."""
        try:
            if self.device.type == 'mps':
                # MPS 디바이스의 경우
                return {
                    'device': str(self.device),
                    'memory_allocated': torch.mps.current_allocated_memory(),
                    'memory_reserved': torch.mps.driver_allocated_memory(),
                    'memory_cached': 0  # MPS는 캐시 메모리 정보를 제공하지 않음
                }
            elif self.device.type == 'cuda':
                # CUDA 디바이스의 경우
                return {
                    'device': str(self.device),
                    'memory_allocated': torch.cuda.memory_allocated(self.device),
                    'memory_reserved': torch.cuda.memory_reserved(self.device),
                    'memory_cached': torch.cuda.memory_cached(self.device)
                }
            else:
                # CPU의 경우
                return {
                    'device': str(self.device),
                    'memory_allocated': 0,
                    'memory_reserved': 0,
                    'memory_cached': 0
                }
        except Exception as e:
            self.logger.error(f"PyTorch 메모리 정보 조회 실패: {e}")
            return {}
    
    def get_memory_usage_summary(self) -> Dict[str, Any]:
        """메모리 사용량 요약을 반환합니다."""
        system_memory = self.get_system_memory_info()
        torch_memory = self.get_torch_memory_info()
        
        # 메모리 사용량 기록
        memory_info = {
            'timestamp': time.time(),
            'system_memory': system_memory,
            'torch_memory': torch_memory,
            'cache_size': len(self.cache_objects)
        }
        
        self.memory_usage_history.append(memory_info)
        
        # 최근 100개만 유지
        if len(self.memory_usage_history) > 100:
            self.memory_usage_history = self.memory_usage_history[-100:]
        
        return memory_info
    
    def check_memory_threshold(self) -> bool:
        """메모리 임계값을 확인합니다."""
        memory_info = self.get_memory_usage_summary()
        system_memory = memory_info.get('system_memory', {})
        
        if 'percent' in system_memory:
            return system_memory['percent'] > (self.config.memory_threshold * 100)
        
        return False
    
    def optimize_memory(self) -> Dict[str, Any]:
        """메모리를 최적화합니다."""
        if not self.config.enable_memory_optimization:
            return {'status': 'disabled'}
        
        optimization_results = {}
        
        try:
            # PyTorch 캐시 정리
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                optimization_results['torch_cache_cleared'] = True
            
            # Python 가비지 컬렉션
            gc.collect()
            optimization_results['garbage_collected'] = True
            
            # 캐시 정리
            if self.config.enable_cache_management:
                cache_cleared = self.clear_cache()
                optimization_results['cache_cleared'] = cache_cleared
            
            # 메모리 정보 업데이트
            memory_info = self.get_memory_usage_summary()
            optimization_results['memory_info'] = memory_info
            
            self.logger.info("메모리 최적화 완료")
            optimization_results['status'] = 'success'
            
        except Exception as e:
            self.logger.error(f"메모리 최적화 실패: {e}")
            optimization_results['status'] = 'error'
            optimization_results['error'] = str(e)
        
        return optimization_results
    
    def add_to_cache(self, key: str, value: Any, size: int = 0) -> bool:
        """캐시에 객체를 추가합니다."""
        if not self.config.enable_cache_management:
            return False
        
        try:
            # 캐시 크기 확인
            current_cache_size = sum(obj.get('size', 0) for obj in self.cache_objects.values())
            
            if current_cache_size + size > self.config.cache_size_limit:
                # 캐시가 가득 찬 경우 오래된 항목 제거
                self._cleanup_cache()
            
            # 캐시에 추가
            self.cache_objects[key] = {
                'value': value,
                'size': size,
                'timestamp': time.time()
            }
            
            self.logger.debug(f"캐시에 추가됨: {key} (크기: {size})")
            return True
            
        except Exception as e:
            self.logger.error(f"캐시 추가 실패: {e}")
            return False
    
    def get_from_cache(self, key: str) -> Optional[Any]:
        """캐시에서 객체를 가져옵니다."""
        if not self.config.enable_cache_management:
            return None
        
        try:
            if key in self.cache_objects:
                cache_item = self.cache_objects[key]
                cache_item['last_accessed'] = time.time()
                return cache_item['value']
            return None
            
        except Exception as e:
            self.logger.error(f"캐시 조회 실패: {e}")
            return None
    
    def clear_cache(self) -> bool:
        """캐시를 정리합니다."""
        try:
            cleared_count = len(self.cache_objects)
            self.cache_objects.clear()
            self.last_cleanup_time = time.time()
            
            self.logger.info(f"캐시 정리 완료: {cleared_count}개 항목 제거")
            return True
            
        except Exception as e:
            self.logger.error(f"캐시 정리 실패: {e}")
            return False
    
    def _cleanup_cache(self):
        """캐시를 정리합니다 (내부 메서드)."""
        try:
            # 오래된 항목들을 제거
            current_time = time.time()
            keys_to_remove = []
            
            for key, item in self.cache_objects.items():
                # 1시간 이상 된 항목 제거
                if current_time - item['timestamp'] > 3600:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cache_objects[key]
            
            if keys_to_remove:
                self.logger.debug(f"캐시 자동 정리: {len(keys_to_remove)}개 항목 제거")
                
        except Exception as e:
            self.logger.error(f"캐시 자동 정리 실패: {e}")
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """메모리 통계를 반환합니다."""
        current_memory = self.get_memory_usage_summary()
        
        # 히스토리 분석
        if self.memory_usage_history:
            memory_percentages = [item['system_memory'].get('percent', 0) for item in self.memory_usage_history]
            avg_memory_usage = sum(memory_percentages) / len(memory_percentages)
            max_memory_usage = max(memory_percentages)
            min_memory_usage = min(memory_percentages)
        else:
            avg_memory_usage = max_memory_usage = min_memory_usage = 0
        
        return {
            'current_memory': current_memory,
            'memory_history': {
                'total_records': len(self.memory_usage_history),
                'average_usage_percent': avg_memory_usage,
                'max_usage_percent': max_memory_usage,
                'min_usage_percent': min_memory_usage
            },
            'cache_info': {
                'total_objects': len(self.cache_objects),
                'cache_size_limit': self.config.cache_size_limit,
                'last_cleanup': self.last_cleanup_time
            },
            'config': self.config.__dict__
        }
    
    def monitor_memory_continuously(self, interval: float = 5.0, duration: float = 60.0):
        """지속적으로 메모리를 모니터링합니다."""
        if not self.config.enable_memory_monitoring:
            self.logger.warning("메모리 모니터링이 비활성화되어 있습니다.")
            return
        
        start_time = time.time()
        self.logger.info(f"메모리 모니터링 시작 (간격: {interval}초, 지속시간: {duration}초)")
        
        try:
            while time.time() - start_time < duration:
                memory_info = self.get_memory_usage_summary()
                system_memory = memory_info.get('system_memory', {})
                
                if 'percent' in system_memory:
                    usage_percent = system_memory['percent']
                    self.logger.info(f"메모리 사용량: {usage_percent:.1f}%")
                    
                    # 임계값 초과 시 경고
                    if usage_percent > (self.config.memory_threshold * 100):
                        self.logger.warning(f"메모리 사용량이 임계값을 초과했습니다: {usage_percent:.1f}%")
                        
                        # 자동 최적화
                        if self.config.enable_memory_optimization:
                            self.optimize_memory()
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("메모리 모니터링이 중단되었습니다.")
        except Exception as e:
            self.logger.error(f"메모리 모니터링 중 오류 발생: {e}")
        
        self.logger.info("메모리 모니터링 완료")

# 사용 예시
if __name__ == "__main__":
    # 설정
    config = MemoryServiceConfig(
        enable_memory_monitoring=True,
        enable_memory_optimization=True,
        enable_cache_management=True,
        memory_threshold=0.8,
        cache_size_limit=1024 * 1024 * 1024,  # 1GB
        use_mps=True
    )
    
    # 메모리 서비스 초기화
    memory_service = ClothWarpingMemoryService(config)
    
    # 메모리 정보 조회
    memory_info = memory_service.get_memory_usage_summary()
    print(f"현재 메모리 정보: {memory_info}")
    
    # 캐시에 테스트 데이터 추가
    test_data = torch.randn(100, 100)
    memory_service.add_to_cache('test_tensor', test_data, size=100 * 100 * 4)  # 4 bytes per float32
    
    # 캐시에서 데이터 조회
    retrieved_data = memory_service.get_from_cache('test_tensor')
    print(f"캐시에서 조회된 데이터: {retrieved_data is not None}")
    
    # 메모리 통계
    stats = memory_service.get_memory_statistics()
    print(f"메모리 통계: {stats}")
    
    # 메모리 최적화
    optimization_result = memory_service.optimize_memory()
    print(f"메모리 최적화 결과: {optimization_result}")
