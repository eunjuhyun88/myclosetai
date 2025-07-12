# backend/app/core/gpu_config.py
"""
M3 Max GPU 최적화 설정
Metal Performance Shaders를 활용한 AI 가속
"""
import torch
import os
import subprocess
import platform
import logging

logger = logging.getLogger(__name__)

class M3GPUConfig:
    def __init__(self):
        self.device = self._get_optimal_device()
        self.memory_fraction = 0.8  # 메모리 사용량 제한
        self.setup_optimizations()
        
    def _get_optimal_device(self):
        """최적의 디바이스 선택"""
        # Apple Silicon 체크
        if platform.machine() == "arm64" and platform.system() == "Darwin":
            if torch.backends.mps.is_available():
                logger.info("✅ Apple M3 Max GPU (Metal) 사용 가능")
                return "mps"
            else:
                logger.warning("⚠️ MPS 사용 불가, CPU 모드로 실행")
                return "cpu"
        else:
            # NVIDIA GPU 체크
            if torch.cuda.is_available():
                logger.info("✅ NVIDIA GPU (CUDA) 사용 가능")
                return "cuda"
            else:
                logger.info("ℹ️ CPU 모드로 실행")
                return "cpu"
    
    def setup_optimizations(self):
        """시스템 최적화 설정"""
        if self.device == "mps":
            # Metal 최적화 환경변수
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            
        elif self.device == "cuda":
            # CUDA 최적화
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            
        # 공통 최적화
        os.environ['OMP_NUM_THREADS'] = '8'
        os.environ['MKL_NUM_THREADS'] = '8'
        
    def optimize_memory(self):
        """메모리 최적화"""
        try:
            if self.device == "mps":
                # Metal 메모리 정리
                torch.mps.empty_cache()
                logger.info("🧹 MPS 메모리 캐시 정리 완료")
                
            elif self.device == "cuda":
                # CUDA 메모리 정리
                torch.cuda.empty_cache()
                logger.info("🧹 CUDA 메모리 캐시 정리 완료")
                
        except Exception as e:
            logger.warning(f"메모리 최적화 실패: {e}")
        
    def get_model_config(self):
        """모델 설정 반환"""
        config = {
            "device": self.device,
            "memory_efficient": True,
            "batch_size": 1,  # M3 Max/CUDA 안전한 배치 크기
        }
        
        if self.device == "mps":
            config.update({
                "dtype": torch.float32,  # MPS는 float16 불안정할 수 있음
                "max_memory_mb": 24000,  # M3 Max 메모리 한계
            })
        elif self.device == "cuda":
            config.update({
                "dtype": torch.float16,  # CUDA는 float16 지원
                "mixed_precision": True,
            })
        else:
            config.update({
                "dtype": torch.float32,
                "max_memory_mb": 8000,   # CPU 메모리 제한
            })
            
        return config
    
    def get_device_info(self):
        """디바이스 정보 반환"""
        info = {
            "device": self.device,
            "platform": platform.system(),
            "machine": platform.machine(),
        }
        
        if self.device == "mps":
            info.update({
                "gpu_name": "Apple M3 Max",
                "memory_available": "128GB (통합 메모리)",
                "cores": "30-40 GPU 코어",
            })
        elif self.device == "cuda":
            if torch.cuda.is_available():
                info.update({
                    "gpu_name": torch.cuda.get_device_name(0),
                    "memory_available": f"{torch.cuda.get_device_properties(0).total_memory // 1024**3}GB",
                    "cuda_version": torch.version.cuda,
                })
        
        return info
    
    def benchmark_device(self):
        """디바이스 벤치마크"""
        try:
            logger.info("🔧 디바이스 벤치마크 시작...")
            
            device = torch.device(self.device)
            
            # 간단한 연산 테스트
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            
            import time
            start = time.time()
            for _ in range(100):
                z = torch.mm(x, y)
            end = time.time()
            
            avg_time = (end - start) / 100
            logger.info(f"✅ 벤치마크 완료: {avg_time:.4f}초/연산")
            
            return {
                "success": True,
                "avg_operation_time": avg_time,
                "device": self.device
            }
            
        except Exception as e:
            logger.error(f"❌ 벤치마크 실패: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# 전역 GPU 설정 인스턴스
gpu_config = M3GPUConfig()
DEVICE = gpu_config.device
MODEL_CONFIG = gpu_config.get_model_config()
DEVICE_INFO = gpu_config.get_device_info()

# 초기화 로그
logger.info(f"🔧 GPU 설정 완료: {DEVICE}")
logger.info(f"📊 디바이스 정보: {DEVICE_INFO}")

# 벤치마크 실행 (선택적)
def run_benchmark():
    """벤치마크 실행"""
    return gpu_config.benchmark_device()