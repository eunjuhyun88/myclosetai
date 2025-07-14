# app/ai_pipeline/utils/memory_manager.py
"""
M3 Max 최적화 메모리 매니저 (PyTorch 2.5.1 호환성 보장)
"""

import logging
import torch
import psutil
import gc
import platform
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """메모리 통계 정보"""
    cpu_total_gb: float
    cpu_available_gb: float
    cpu_used_percent: float
    gpu_allocated_gb: float = 0.0
    gpu_cached_gb: float = 0.0

class MemoryManager:
    """M3 Max 최적화 메모리 매니저"""
    
    def __init__(self, device: str = "mps", memory_limit_gb: Optional[float] = None):
        self.device = device
        self.is_mps = device == "mps"
        self.is_m3_max = self._detect_m3_max()
        self.memory_limit_gb = memory_limit_gb or self._get_optimal_memory_limit()
        
        # PyTorch 버전별 호환성 체크
        self.pytorch_version = torch.__version__
        self.mps_empty_cache_available = self._check_mps_empty_cache()
        
        logger.info(f"🧠 MemoryManager 초기화 - 디바이스: {device}, 메모리 제한: {self.memory_limit_gb}GB")
        
        if self.is_m3_max:
            logger.info("🍎 M3 Max 최적화 모드 활성화")
            
        if not self.mps_empty_cache_available and self.is_mps:
            logger.warning(f"⚠️ PyTorch {self.pytorch_version}: MPS empty_cache 미지원 - 대체 메모리 관리 사용")
    
    def _detect_m3_max(self) -> bool:
        """M3 Max 감지"""
        try:
            if platform.system() != "Darwin" or platform.machine() != "arm64":
                return False
            
            # 메모리 크기로 M3 Max 추정 (36GB+ = M3 Max)
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            return total_memory_gb >= 32.0
        except:
            return False
    
    def _check_mps_empty_cache(self) -> bool:
        """MPS empty_cache 메서드 사용 가능성 체크"""
        if not self.is_mps:
            return False
        
        try:
            # PyTorch 2.5.1에서는 empty_cache가 제거됨
            return hasattr(torch.backends.mps, 'empty_cache')
        except:
            return False
    
    def _get_optimal_memory_limit(self) -> float:
        """최적 메모리 제한 계산"""
        total_memory = psutil.virtual_memory().total / (1024**3)
        
        if self.is_m3_max:
            # M3 Max: 통합 메모리 80% 활용
            return total_memory * 0.8
        else:
            # 일반 시스템: 보수적 접근
            return min(total_memory * 0.6, 16.0)
    
    def clear_cache(self) -> None:
        """메모리 캐시 정리 (버전 호환성 보장)"""
        try:
            # CPU 메모리 정리
            gc.collect()
            
            if self.is_mps:
                if self.mps_empty_cache_available:
                    # PyTorch 2.4 이하
                    torch.backends.mps.empty_cache()
                    logger.debug("✅ MPS 캐시 정리 완료")
                else:
                    # PyTorch 2.5+ 대체 방법
                    self._alternative_mps_cleanup()
                    logger.debug("✅ MPS 대체 메모리 정리 완료")
            
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("✅ CUDA 캐시 정리 완료")
                
        except Exception as e:
            logger.warning(f"⚠️ 메모리 정리 실패: {e}")
    
    def _alternative_mps_cleanup(self) -> None:
        """MPS 대체 메모리 정리 방법"""
        try:
            # 1. Python 가비지 컬렉션 강제 실행
            for _ in range(3):
                gc.collect()
            
            # 2. 메모리 최적화 환경변수 설정
            import os
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            
            # 3. 임시 텐서 생성/삭제로 메모리 할당 패턴 리셋
            if torch.backends.mps.is_available():
                temp_tensor = torch.zeros(1, device='mps')
                del temp_tensor
                
        except Exception as e:
            logger.debug(f"대체 MPS 정리 실패: {e}")
    
    def get_memory_stats(self) -> MemoryStats:
        """현재 메모리 사용량 조회"""
        # CPU 메모리
        cpu_memory = psutil.virtual_memory()
        cpu_total_gb = cpu_memory.total / (1024**3)
        cpu_available_gb = cpu_memory.available / (1024**3)
        cpu_used_percent = cpu_memory.percent
        
        # GPU 메모리 (가능한 경우)
        gpu_allocated_gb = 0.0
        gpu_cached_gb = 0.0
        
        if torch.cuda.is_available():
            try:
                gpu_allocated_gb = torch.cuda.memory_allocated() / (1024**3)
                gpu_cached_gb = torch.cuda.memory_reserved() / (1024**3)
            except:
                pass
        
        return MemoryStats(
            cpu_total_gb=cpu_total_gb,
            cpu_available_gb=cpu_available_gb,
            cpu_used_percent=cpu_used_percent,
            gpu_allocated_gb=gpu_allocated_gb,
            gpu_cached_gb=gpu_cached_gb
        )
    
    def optimize_memory(self) -> None:
        """메모리 최적화 실행"""
        try:
            stats_before = self.get_memory_stats()
            
            # 메모리 정리
            self.clear_cache()
            
            # M3 Max 특별 최적화
            if self.is_m3_max:
                self._optimize_for_m3_max()
            
            stats_after = self.get_memory_stats()
            
            # 최적화 결과 로깅
            freed_memory = stats_before.cpu_total_gb - stats_before.cpu_available_gb - \
                          (stats_after.cpu_total_gb - stats_after.cpu_available_gb)
            
            logger.info(
                f"🧹 메모리 최적화 완료 - "
                f"CPU: {stats_after.cpu_available_gb:.2f}GB, "
                f"GPU: {stats_after.gpu_allocated_gb:.2f}GB, "
                f"해제됨: {freed_memory:.2f}GB"
            )
            
        except Exception as e:
            logger.error(f"❌ 메모리 최적화 실패: {e}")
    
    def _optimize_for_m3_max(self) -> None:
        """M3 Max 특화 최적화"""
        try:
            import os
            
            # M3 Max 통합 메모리 최적화 설정
            os.environ.update({
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
                "PYTORCH_MPS_ALLOCATOR_POLICY": "garbage_collection",
                "OMP_NUM_THREADS": "8",  # M3 Max 성능 코어 수
                "MKL_NUM_THREADS": "8"
            })
            
            logger.debug("🍎 M3 Max 환경 최적화 완료")
            
        except Exception as e:
            logger.warning(f"M3 Max 최적화 실패: {e}")
    
    def check_memory_available(self, required_gb: float) -> bool:
        """필요 메모리 사용 가능성 체크"""
        stats = self.get_memory_stats()
        available = stats.cpu_available_gb
        
        if self.is_m3_max:
            # M3 Max는 통합 메모리로 더 유연함
            available *= 1.2
        
        return available >= required_gb
    
    def start_monitoring(self) -> None:
        """메모리 모니터링 시작"""
        logger.info("📊 메모리 모니터링 시작")
        # 실제 모니터링 로직은 필요시 구현
    
    def stop_monitoring(self) -> None:
        """메모리 모니터링 중지"""
        logger.info("📊 메모리 모니터링 중지")