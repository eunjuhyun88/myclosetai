# app/ai_pipeline/interfaces/memory_interface.py
"""
🔥 메모리 관리 인터페이스 v2.0 - M3 Max 최적화
===============================================

✅ BaseStepMixin v10.0 완벽 호환
✅ DI Container 인터페이스 패턴 적용
✅ M3 Max 128GB 메모리 최적화
✅ MPS (Metal Performance Shaders) 지원
✅ 통합 메모리 활용 최적화
✅ 비동기 메모리 관리 지원
✅ conda 환경 완벽 지원
✅ 프로덕션 안정성 보장

Author: MyCloset AI Team
Date: 2025-07-20
Version: 2.0 (M3 Max Optimized)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
from enum import Enum
import time

# ==============================================
# 🔥 메모리 관리 관련 데이터 클래스
# ==============================================

class MemoryOptimizationLevel(Enum):
    """메모리 최적화 레벨"""
    MINIMAL = "minimal"      # 최소한의 정리
    STANDARD = "standard"    # 표준 정리
    AGGRESSIVE = "aggressive"  # 공격적 정리
    MAXIMUM = "maximum"      # 최대 정리

class MemoryDeviceType(Enum):
    """메모리 디바이스 타입"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"              # M3 Max Metal
    UNIFIED = "unified"      # M3 Max 통합 메모리

# ==============================================
# 🔥 메모리 관리자 인터페이스
# ==============================================

class IMemoryManager(ABC):
    """
    메모리 관리자 인터페이스
    
    BaseStepMixin v10.0의 memory_manager 속성으로 주입됨
    M3 Max 128GB 통합 메모리 최적화 특화
    """
    
    @abstractmethod
    def optimize_memory(self, level: MemoryOptimizationLevel = MemoryOptimizationLevel.STANDARD) -> Dict[str, Any]:
        """
        동기 메모리 최적화
        
        Args:
            level: 최적화 레벨
            
        Returns:
            {
                'success': bool,
                'before_usage_gb': float,
                'after_usage_gb': float,
                'freed_gb': float,
                'duration': float,
                'optimizations_applied': List[str],
                'device_type': str,
                'is_m3_max': bool
            }
        """
        pass
    
    @abstractmethod
    async def optimize_memory_async(self, level: MemoryOptimizationLevel = MemoryOptimizationLevel.STANDARD) -> Dict[str, Any]:
        """
        비동기 메모리 최적화
        
        Args:
            level: 최적화 레벨
            
        Returns:
            메모리 최적화 결과 (optimize_memory와 동일)
        """
        pass
    
    @abstractmethod
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        메모리 상태 조회
        
        Returns:
            {
                'total_gb': float,
                'available_gb': float,
                'used_gb': float,
                'usage_percent': float,
                'device_type': str,
                'is_m3_max': bool,
                'unified_memory': bool,
                'mps_available': bool,
                'torch_memory': Dict[str, float],  # PyTorch 메모리 (있는 경우)
                'timestamp': float
            }
        """
        pass
    
    @abstractmethod
    def check_memory_threshold(self, threshold: float = 0.85) -> Tuple[bool, Dict[str, Any]]:
        """
        메모리 임계값 확인
        
        Args:
            threshold: 임계값 (0.0-1.0)
            
        Returns:
            (임계값_초과_여부, 메모리_상태)
        """
        pass
    
    @abstractmethod
    def cleanup_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """
        메모리 정리 (호환성 메서드)
        
        Args:
            aggressive: 공격적 정리 여부
            
        Returns:
            정리 결과
        """
        pass
    
    @abstractmethod
    def setup_m3_max_optimization(self) -> Dict[str, Any]:
        """
        M3 Max 특화 최적화 설정
        
        Returns:
            {
                'success': bool,
                'unified_memory_enabled': bool,
                'mps_enabled': bool,
                'neural_engine_ready': bool,
                'memory_pooling_enabled': bool,
                'optimizations': List[str]
            }
        """
        pass
    
    @abstractmethod
    def monitor_memory_usage(self, duration_seconds: float = 60.0) -> Dict[str, Any]:
        """
        메모리 사용량 모니터링
        
        Args:
            duration_seconds: 모니터링 시간
            
        Returns:
            {
                'monitoring_duration': float,
                'samples_count': int,
                'avg_usage_gb': float,
                'peak_usage_gb': float,
                'min_usage_gb': float,
                'usage_trend': str,  # 'increasing', 'decreasing', 'stable'
                'samples': List[Dict[str, Any]]
            }
        """
        pass

# ==============================================
# 🔥 M3 Max 특화 메모리 관리자 인터페이스
# ==============================================

class IM3MaxMemoryManager(IMemoryManager):
    """
    M3 Max 특화 메모리 관리자 인터페이스
    
    통합 메모리 아키텍처 최적화
    """
    
    @abstractmethod
    def optimize_unified_memory(self) -> Dict[str, Any]:
        """
        통합 메모리 최적화
        
        Returns:
            최적화 결과
        """
        pass
    
    @abstractmethod
    def setup_mps_memory_management(self) -> Dict[str, Any]:
        """
        MPS 메모리 관리 설정
        
        Returns:
            설정 결과
        """
        pass
    
    @abstractmethod
    def enable_neural_engine_memory_sharing(self) -> Dict[str, Any]:
        """
        Neural Engine 메모리 공유 활성화
        
        Returns:
            활성화 결과
        """
        pass
    
    @abstractmethod
    def get_m3_max_memory_topology(self) -> Dict[str, Any]:
        """
        M3 Max 메모리 토폴로지 정보
        
        Returns:
            {
                'total_unified_memory_gb': float,
                'gpu_memory_gb': float,
                'neural_engine_memory_gb': float,
                'shared_memory_gb': float,
                'memory_bandwidth_gbps': float,
                'memory_channels': int
            }
        """
        pass
    
    @abstractmethod
    def optimize_memory_for_large_models(self, model_size_gb: float) -> Dict[str, Any]:
        """
        대용량 모델용 메모리 최적화
        
        Args:
            model_size_gb: 모델 크기 (GB)
            
        Returns:
            최적화 결과
        """
        pass

# ==============================================
# 🔥 메모리 감시자 인터페이스
# ==============================================

class IMemoryWatcher(ABC):
    """
    메모리 감시자 인터페이스
    
    실시간 메모리 모니터링 및 자동 최적화
    """
    
    @abstractmethod
    def start_monitoring(self, interval_seconds: float = 5.0) -> bool:
        """
        메모리 모니터링 시작
        
        Args:
            interval_seconds: 모니터링 간격
            
        Returns:
            시작 성공 여부
        """
        pass
    
    @abstractmethod
    def stop_monitoring(self) -> Dict[str, Any]:
        """
        메모리 모니터링 중지
        
        Returns:
            모니터링 결과 요약
        """
        pass
    
    @abstractmethod
    def set_auto_cleanup_threshold(self, threshold: float) -> None:
        """
        자동 정리 임계값 설정
        
        Args:
            threshold: 임계값 (0.0-1.0)
        """
        pass
    
    @abstractmethod
    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        모니터링 상태 조회
        
        Returns:
            모니터링 상태 정보
        """
        pass
    
    @abstractmethod
    def register_memory_callback(self, callback: callable, threshold: float) -> str:
        """
        메모리 임계값 콜백 등록
        
        Args:
            callback: 콜백 함수
            threshold: 트리거 임계값
            
        Returns:
            콜백 ID
        """
        pass
    
    @abstractmethod
    def unregister_memory_callback(self, callback_id: str) -> bool:
        """
        메모리 콜백 해제
        
        Args:
            callback_id: 콜백 ID
            
        Returns:
            해제 성공 여부
        """
        pass

# ==============================================
# 🔥 캐시 관리자 인터페이스
# ==============================================

class ICacheManager(ABC):
    """
    캐시 관리자 인터페이스
    
    모델 및 데이터 캐시 관리
    """
    
    @abstractmethod
    def clear_model_cache(self) -> Dict[str, Any]:
        """
        모델 캐시 정리
        
        Returns:
            정리 결과
        """
        pass
    
    @abstractmethod
    def clear_data_cache(self) -> Dict[str, Any]:
        """
        데이터 캐시 정리
        
        Returns:
            정리 결과
        """
        pass
    
    @abstractmethod
    def clear_all_cache(self) -> Dict[str, Any]:
        """
        모든 캐시 정리
        
        Returns:
            정리 결과
        """
        pass
    
    @abstractmethod
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        캐시 상태 조회
        
        Returns:
            캐시 사용량 정보
        """
        pass
    
    @abstractmethod
    def optimize_cache_usage(self) -> Dict[str, Any]:
        """
        캐시 사용량 최적화
        
        Returns:
            최적화 결과
        """
        pass
    
    @abstractmethod
    def set_cache_limits(self, model_cache_gb: float, data_cache_gb: float) -> bool:
        """
        캐시 제한 설정
        
        Args:
            model_cache_gb: 모델 캐시 제한 (GB)
            data_cache_gb: 데이터 캐시 제한 (GB)
            
        Returns:
            설정 성공 여부
        """
        pass

# ==============================================
# 🔥 GPU 메모리 관리자 인터페이스
# ==============================================

class IGPUMemoryManager(ABC):
    """
    GPU 메모리 관리자 인터페이스
    
    CUDA, MPS 등 GPU 메모리 특화 관리
    """
    
    @abstractmethod
    def clear_gpu_cache(self, device: str = "auto") -> Dict[str, Any]:
        """
        GPU 캐시 정리
        
        Args:
            device: 대상 디바이스 ("cuda", "mps", "auto")
            
        Returns:
            정리 결과
        """
        pass
    
    @abstractmethod
    def get_gpu_memory_info(self, device: str = "auto") -> Dict[str, Any]:
        """
        GPU 메모리 정보 조회
        
        Args:
            device: 대상 디바이스
            
        Returns:
            GPU 메모리 정보
        """
        pass
    
    @abstractmethod
    def optimize_gpu_memory(self, device: str = "auto") -> Dict[str, Any]:
        """
        GPU 메모리 최적화
        
        Args:
            device: 대상 디바이스
            
        Returns:
            최적화 결과
        """
        pass
    
    @abstractmethod
    def set_gpu_memory_fraction(self, fraction: float, device: str = "auto") -> bool:
        """
        GPU 메모리 사용 비율 설정
        
        Args:
            fraction: 사용 비율 (0.0-1.0)
            device: 대상 디바이스
            
        Returns:
            설정 성공 여부
        """
        pass
    
    @abstractmethod
    def enable_gpu_memory_growth(self, device: str = "auto") -> bool:
        """
        GPU 메모리 점진적 할당 활성화
        
        Args:
            device: 대상 디바이스
            
        Returns:
            활성화 성공 여부
        """
        pass

# ==============================================
# 🔥 인터페이스 타입 유니온 및 내보내기
# ==============================================

# 편의성 타입 별칭
MemoryManagerInterface = IMemoryManager
M3MaxMemoryManagerInterface = IM3MaxMemoryManager
MemoryWatcherInterface = IMemoryWatcher
CacheManagerInterface = ICacheManager
GPUMemoryManagerInterface = IGPUMemoryManager

# 메모리 관련 인터페이스 목록
MEMORY_INTERFACES = [
    'IMemoryManager',
    'IM3MaxMemoryManager',
    'IMemoryWatcher', 
    'ICacheManager',
    'IGPUMemoryManager'
]

# 모듈 내보내기
__all__ = [
    # 인터페이스들
    'IMemoryManager',
    'IM3MaxMemoryManager',
    'IMemoryWatcher',
    'ICacheManager', 
    'IGPUMemoryManager',
    
    # 데이터 클래스들
    'MemoryOptimizationLevel',
    'MemoryDeviceType',
    
    # 편의성 타입 별칭
    'MemoryManagerInterface',
    'M3MaxMemoryManagerInterface', 
    'MemoryWatcherInterface',
    'CacheManagerInterface',
    'GPUMemoryManagerInterface',
    
    # 유틸리티
    'MEMORY_INTERFACES'
]

# 모듈 로드 완료 메시지
print("✅ Memory Interface v2.0 로드 완료 - M3 Max 128GB 최적화")
print("🍎 M3 Max 통합 메모리 아키텍처 특화")
print("⚡ MPS (Metal Performance Shaders) 지원")
print("🔗 BaseStepMixin v10.0과 100% 호환")
print("🚀 메모리 관리 인터페이스 5종 정의 완료!")