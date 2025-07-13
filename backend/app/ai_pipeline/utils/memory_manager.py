"""
MyCloset AI GPU 메모리 매니저
M3 Max 통합 메모리 최적화 및 관리
"""

import gc
import logging
import psutil
import torch
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import time
from collections import defaultdict

@dataclass
class MemoryStats:
    """메모리 통계 정보"""
    total_system_gb: float
    used_system_gb: float
    available_system_gb: float
    mps_allocated_gb: float
    mps_cached_gb: float
    process_memory_gb: float
    timestamp: float

class GPUMemoryManager:
    """M3 Max 통합 메모리 관리자"""
    
    def __init__(self, device: torch.device, memory_limit_gb: float = 16.0):
        self.device = device
        self.memory_limit_gb = memory_limit_gb
        self.logger = logging.getLogger(__name__)
        
        # 메모리 사용량 추적
        self.memory_history: List[MemoryStats] = []
        self.peak_memory_gb = 0.0
        
        # 모델 캐시 관리
        self.model_cache: Dict[str, torch.nn.Module] = {}
        self.cache_order: List[str] = []  # LRU 캐시를 위한 순서
        self.max_cached_models = 3
        
        # 메모리 임계치 설정
        self.warning_threshold = 0.8  # 80% 사용시 경고
        self.critical_threshold = 0.9  # 90% 사용시 긴급 정리
        
        # 스레드 안전성
        self.lock = threading.Lock()
        
        self.logger.info(f"메모리 매니저 초기화 - 한계: {memory_limit_gb}GB")

    def get_memory_stats(self) -> MemoryStats:
        """현재 메모리 상태 조회"""
        # 시스템 메모리
        system_memory = psutil.virtual_memory()
        total_system_gb = system_memory.total / (1024**3)
        used_system_gb = system_memory.used / (1024**3)
        available_system_gb = system_memory.available / (1024**3)
        
        # 프로세스 메모리
        process = psutil.Process()
        process_memory_gb = process.memory_info().rss / (1024**3)
        
        # MPS 메모리 (M3 Max)
        mps_allocated_gb = 0.0
        mps_cached_gb = 0.0
        
        if self.device.type == "mps":
            try:
                mps_allocated_gb = torch.mps.current_allocated_memory() / (1024**3)
                # MPS는 cached memory API가 제한적이므로 추정값 사용
                mps_cached_gb = max(0, process_memory_gb - mps_allocated_gb)
            except Exception as e:
                self.logger.warning(f"MPS 메모리 정보 조회 실패: {e}")
        
        stats = MemoryStats(
            total_system_gb=total_system_gb,
            used_system_gb=used_system_gb,
            available_system_gb=available_system_gb,
            mps_allocated_gb=mps_allocated_gb,
            mps_cached_gb=mps_cached_gb,
            process_memory_gb=process_memory_gb,
            timestamp=time.time()
        )
        
        # 피크 메모리 업데이트
        if mps_allocated_gb > self.peak_memory_gb:
            self.peak_memory_gb = mps_allocated_gb
        
        return stats

    def check_memory_usage(self) -> bool:
        """메모리 사용량 체크 및 필요시 정리"""
        with self.lock:
            stats = self.get_memory_stats()
            self.memory_history.append(stats)
            
            # 히스토리 크기 제한 (최근 100개)
            if len(self.memory_history) > 100:
                self.memory_history = self.memory_history[-100:]
            
            # 메모리 사용률 계산
            usage_ratio = stats.process_memory_gb / self.memory_limit_gb
            
            if usage_ratio > self.critical_threshold:
                self.logger.warning(f"임계 메모리 사용량: {usage_ratio:.2%}")
                self._emergency_cleanup()
                return False
            elif usage_ratio > self.warning_threshold:
                self.logger.info(f"높은 메모리 사용량: {usage_ratio:.2%}")
                self._smart_cleanup()
                return True
            
            return True

    def _emergency_cleanup(self):
        """긴급 메모리 정리"""
        self.logger.warning("긴급 메모리 정리 시작")
        
        # 모든 모델 캐시 제거
        self.clear_model_cache()
        
        # Python 가비지 컬렉션
        gc.collect()
        
        # MPS 캐시 정리
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        self.logger.info("긴급 메모리 정리 완료")

    def _smart_cleanup(self):
        """스마트 메모리 정리 (필요한 것만)"""
        # LRU 기반 모델 캐시 정리
        if len(self.model_cache) > 1:
            oldest_model = self.cache_order[0]
            self.remove_from_cache(oldest_model)
        
        # 메모리 정리
        gc.collect()
        
        if self.device.type == "mps":
            torch.mps.empty_cache()

    @contextmanager
    def memory_context(self, operation_name: str):
        """메모리 사용량 모니터링 컨텍스트"""
        start_stats = self.get_memory_stats()
        start_time = time.time()
        
        try:
            yield
        finally:
            end_stats = self.get_memory_stats()
            end_time = time.time()
            
            memory_delta = end_stats.mps_allocated_gb - start_stats.mps_allocated_gb
            time_delta = end_time - start_time
            
            self.logger.info(
                f"{operation_name} - 메모리 변화: {memory_delta:+.2f}GB, "
                f"소요시간: {time_delta:.2f}초"
            )

    def cache_model(self, model_name: str, model: torch.nn.Module):
        """모델 캐싱 (LRU 방식)"""
        with self.lock:
            # 기존 모델이 있다면 순서 업데이트
            if model_name in self.model_cache:
                self.cache_order.remove(model_name)
            
            # 캐시 크기 제한 체크
            while len(self.model_cache) >= self.max_cached_models:
                oldest_model = self.cache_order.pop(0)
                del self.model_cache[oldest_model]
                self.logger.info(f"모델 캐시에서 제거: {oldest_model}")
            
            # 새 모델 캐싱
            self.model_cache[model_name] = model
            self.cache_order.append(model_name)
            
            self.logger.info(f"모델 캐시에 추가: {model_name}")

    def get_cached_model(self, model_name: str) -> Optional[torch.nn.Module]:
        """캐시된 모델 조회 (LRU 업데이트)"""
        with self.lock:
            if model_name in self.model_cache:
                # LRU 순서 업데이트
                self.cache_order.remove(model_name)
                self.cache_order.append(model_name)
                return self.model_cache[model_name]
            return None

    def remove_from_cache(self, model_name: str):
        """특정 모델을 캐시에서 제거"""
        with self.lock:
            if model_name in self.model_cache:
                del self.model_cache[model_name]
                self.cache_order.remove(model_name)
                self.logger.info(f"모델 캐시에서 제거: {model_name}")

    def clear_model_cache(self):
        """모델 캐시 전체 정리"""
        with self.lock:
            self.model_cache.clear()
            self.cache_order.clear()
            self.logger.info("모델 캐시 전체 정리 완료")

    def clear_cache(self):
        """전체 캐시 정리"""
        self.clear_model_cache()
        gc.collect()
        
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

    def get_memory_report(self) -> Dict[str, Any]:
        """상세 메모리 리포트"""
        stats = self.get_memory_stats()
        
        # 메모리 사용률 계산
        usage_ratio = stats.process_memory_gb / self.memory_limit_gb
        
        # 메모리 효율성 분석
        efficiency_score = 1.0 - (stats.mps_cached_gb / max(stats.mps_allocated_gb, 0.1))
        
        return {
            "current_stats": {
                "total_system_gb": stats.total_system_gb,
                "used_system_gb": stats.used_system_gb,
                "available_system_gb": stats.available_system_gb,
                "mps_allocated_gb": stats.mps_allocated_gb,
                "mps_cached_gb": stats.mps_cached_gb,
                "process_memory_gb": stats.process_memory_gb,
                "usage_ratio": usage_ratio,
                "efficiency_score": efficiency_score
            },
            "limits": {
                "memory_limit_gb": self.memory_limit_gb,
                "warning_threshold": self.warning_threshold,
                "critical_threshold": self.critical_threshold
            },
            "peak_memory_gb": self.peak_memory_gb,
            "cached_models": list(self.model_cache.keys()),
            "cache_utilization": len(self.model_cache) / self.max_cached_models,
            "recommendations": self._get_recommendations(stats, usage_ratio)
        }

    def _get_recommendations(self, stats: MemoryStats, usage_ratio: float) -> List[str]:
        """메모리 최적화 추천사항"""
        recommendations = []
        
        if usage_ratio > 0.9:
            recommendations.append("긴급: 메모리 사용량이 90%를 초과했습니다. 배치 크기를 줄이거나 이미지 해상도를 낮추세요.")
        elif usage_ratio > 0.8:
            recommendations.append("경고: 메모리 사용량이 80%를 초과했습니다. 불필요한 모델 캐시를 정리하세요.")
        
        if stats.mps_cached_gb > 2.0:
            recommendations.append("MPS 캐시가 2GB를 초과했습니다. torch.mps.empty_cache()를 실행하세요.")
        
        if len(self.model_cache) == self.max_cached_models:
            recommendations.append("모델 캐시가 가득 찼습니다. 사용하지 않는 모델을 제거하세요.")
        
        if stats.available_system_gb < 2.0:
            recommendations.append("시스템 메모리가 부족합니다. 다른 애플리케이션을 종료하세요.")
        
        return recommendations

    def optimize_for_batch_size(self, base_batch_size: int = 1) -> int:
        """현재 메모리 상황에 맞는 최적 배치 크기 계산"""
        stats = self.get_memory_stats()
        usage_ratio = stats.process_memory_gb / self.memory_limit_gb
        
        if usage_ratio > 0.8:
            return max(1, base_batch_size // 2)
        elif usage_ratio < 0.5:
            return min(4, base_batch_size * 2)
        else:
            return base_batch_size

    def profile_memory_usage(self, duration_seconds: int = 60):
        """메모리 사용량 프로파일링"""
        self.logger.info(f"메모리 프로파일링 시작 - {duration_seconds}초")
        
        start_time = time.time()
        profile_data = []
        
        while time.time() - start_time < duration_seconds:
            stats = self.get_memory_stats()
            profile_data.append({
                "timestamp": stats.timestamp,
                "mps_allocated_gb": stats.mps_allocated_gb,
                "process_memory_gb": stats.process_memory_gb
            })
            time.sleep(1)
        
        # 프로파일 결과 분석
        avg_mps = sum(d["mps_allocated_gb"] for d in profile_data) / len(profile_data)
        max_mps = max(d["mps_allocated_gb"] for d in profile_data)
        
        self.logger.info(f"프로파일링 완료 - 평균: {avg_mps:.2f}GB, 최대: {max_mps:.2f}GB")
        
        return {
            "duration_seconds": duration_seconds,
            "average_mps_gb": avg_mps,
            "peak_mps_gb": max_mps,
            "data_points": len(profile_data),
            "raw_data": profile_data
        }

# M3 Max 특화 최적화 도구
class M3MaxOptimizer:
    """M3 Max 특화 메모리 최적화"""
    
    @staticmethod
    def optimize_tensor_operations():
        """텐서 연산 최적화 설정"""
        # MPS 최적화 설정
        if torch.backends.mps.is_available():
            # MPS 메모리 할당 전략 최적화
            torch.mps.set_per_process_memory_fraction(0.8)
        
        # 메모리 효율적 attention 사용
        torch.backends.cuda.enable_flash_sdp(False)  # MPS에서는 비활성화
        
        # 자동 mixed precision 설정
        torch.backends.cudnn.benchmark = False  # MPS에서는 불필요
        
        return True

    @staticmethod
    def get_optimal_image_size(available_memory_gb: float) -> int:
        """사용 가능한 메모리에 따른 최적 이미지 크기"""
        if available_memory_gb > 12:
            return 1024
        elif available_memory_gb > 8:
            return 768
        elif available_memory_gb > 4:
            return 512
        else:
            return 256

    @staticmethod
    def estimate_memory_usage(
        batch_size: int,
        image_size: int,
        model_complexity: str = "medium"
    ) -> float:
        """예상 메모리 사용량 계산 (GB)"""
        
        # 기본 이미지 메모리 (RGB, float32)
        image_memory = (batch_size * 3 * image_size * image_size * 4) / (1024**3)
        
        # 모델 복잡도별 가중치
        complexity_multiplier = {
            "simple": 2.0,
            "medium": 4.0,
            "complex": 8.0
        }.get(model_complexity, 4.0)
        
        # 총 예상 메모리 (이미지 + 모델 + 중간 결과)
        total_memory = image_memory * complexity_multiplier
        
        return total_memory

# 사용 예시
def example_usage():
    """메모리 매니저 사용 예시"""
    
    # 디바이스 설정
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 메모리 매니저 초기화
    memory_manager = GPUMemoryManager(device, memory_limit_gb=16.0)
    
    # M3 Max 최적화 적용
    M3MaxOptimizer.optimize_tensor_operations()
    
    # 메모리 사용량 체크
    memory_manager.check_memory_usage()
    
    # 메모리 컨텍스트 사용
    with memory_manager.memory_context("모델 추론"):
        # 실제 모델 추론 코드
        dummy_tensor = torch.randn(1, 3, 512, 512).to(device)
        result = dummy_tensor * 2
    
    # 메모리 리포트
    report = memory_manager.get_memory_report()
    print(f"메모리 사용률: {report['current_stats']['usage_ratio']:.2%}")
    
    # 추천사항 출력
    for rec in report['recommendations']:
        print(f"💡 {rec}")

if __name__ == "__main__":
    example_usage()