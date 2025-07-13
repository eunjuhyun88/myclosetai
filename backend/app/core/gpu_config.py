# backend/app/core/gpu_config.py
"""
MyCloset AI - M3 Max GPU 최적화 설정
Apple M3 Max (128GB RAM, 30-40 GPU Core) 전용 Metal Performance Shaders 활용
"""

import torch
import os
import platform
import logging
import gc
import psutil
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class M3MaxGPUConfig:
    """M3 Max GPU 최적화 설정 클래스"""
    
    def __init__(self):
        """GPU 설정 초기화"""
        self.device = self._detect_optimal_device()
        self.memory_fraction = 0.8  # 128GB 중 80% 활용
        self.is_m3_max = self._check_m3_max()
        self.setup_optimizations()
        
        # 초기화 로그
        logger.info(f"🔧 GPU 설정 초기화 완료")
        logger.info(f"📱 디바이스: {self.device}")
        logger.info(f"🧠 메모리 할당: {self.memory_fraction * 100}%")
        logger.info(f"⚡ M3 Max 모드: {self.is_m3_max}")
        
    def _detect_optimal_device(self) -> str:
        """최적의 디바이스 자동 감지"""
        # Apple Silicon 체크 (M1/M2/M3)
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            if torch.backends.mps.is_available():
                logger.info("✅ Apple Silicon GPU (Metal) 감지됨")
                return "mps"
            else:
                logger.warning("⚠️ Metal 사용 불가, CPU 모드로 실행")
                return "cpu"
        
        # NVIDIA GPU 체크 (서버 환경용)
        elif torch.cuda.is_available():
            logger.info("✅ NVIDIA GPU (CUDA) 감지됨")
            return "cuda"
        
        # CPU 폴백
        else:
            logger.info("ℹ️ GPU 없음, CPU 모드로 실행")
            return "cpu"
    
    def _check_m3_max(self) -> bool:
        """M3 Max 칩 확인"""
        try:
            # macOS에서 시스템 정보 확인
            if platform.system() == "Darwin":
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                cpu_info = result.stdout.strip()
                
                # M3 Max 확인 (실제로는 더 정교한 감지 필요)
                total_memory = psutil.virtual_memory().total // (1024**3)  # GB
                
                # 128GB RAM이면 M3 Max로 추정
                if total_memory >= 120:  # 실제 사용 가능한 메모리는 조금 적음
                    logger.info(f"✅ M3 Max 감지됨 (RAM: {total_memory}GB)")
                    return True
                else:
                    logger.info(f"ℹ️ M3 Pro/일반 감지됨 (RAM: {total_memory}GB)")
                    return False
            return False
            
        except Exception as e:
            logger.warning(f"M3 Max 감지 실패: {e}")
            return False
    
    def setup_optimizations(self):
        """시스템 최적화 설정"""
        
        if self.device == "mps":
            # Metal Performance Shaders 최적화
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            
            # M3 Max 전용 최적화
            if self.is_m3_max:
                os.environ['MPS_FORCE_HEAPS_OVERRIDE'] = '1'
                os.environ['MPS_MEMORY_ALLOCATION_POLICY'] = 'optimal'
                
        elif self.device == "cuda":
            # CUDA 최적화 (서버 환경용)
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
        # 공통 CPU 최적화
        if self.is_m3_max:
            # M3 Max는 14코어 (10 성능 + 4 효율)
            os.environ['OMP_NUM_THREADS'] = '10'  # 성능 코어만 사용
            os.environ['MKL_NUM_THREADS'] = '10'
        else:
            os.environ['OMP_NUM_THREADS'] = '8'
            os.environ['MKL_NUM_THREADS'] = '8'
            
        # PyTorch 최적화
        torch.set_num_threads(10 if self.is_m3_max else 8)
        
        logger.info("⚙️ 시스템 최적화 설정 완료")
    
    def get_model_config(self) -> Dict[str, Any]:
        """AI 모델별 최적화 설정 반환"""
        
        base_config = {
            "device": self.device,
            "memory_efficient": True,
            "batch_size": 1,  # 안전한 배치 크기
            "gradient_checkpointing": True,  # 메모리 절약
        }
        
        if self.device == "mps":
            # M3 Max Metal 설정
            mps_config = {
                "dtype": torch.float32,  # MPS는 float16이 불안정할 수 있음
                "attention_slicing": True,
                "cpu_offload": False,  # 128GB RAM이므로 CPU 오프로드 비활성화
            }
            
            if self.is_m3_max:
                # M3 Max 전용 고성능 설정
                mps_config.update({
                    "max_memory_mb": 24000,  # 24GB GPU 메모리 할당
                    "enable_flash_attention": True,
                    "mixed_precision": False,  # MPS에서는 비활성화
                })
            else:
                # M3 Pro/일반 보수적 설정
                mps_config.update({
                    "max_memory_mb": 12000,
                    "enable_flash_attention": False,
                })
                
            base_config.update(mps_config)
            
        elif self.device == "cuda":
            # NVIDIA GPU 설정
            cuda_config = {
                "dtype": torch.float16,
                "mixed_precision": True,
                "attention_slicing": False,
                "cpu_offload": False,
            }
            base_config.update(cuda_config)
            
        else:
            # CPU 설정
            cpu_config = {
                "dtype": torch.float32,
                "max_memory_mb": 8000,
                "cpu_offload": True,
                "low_cpu_mem_usage": True,
            }
            base_config.update(cpu_config)
            
        return base_config
    
    def optimize_memory(self):
        """메모리 최적화 실행"""
        try:
            # Python 가비지 컬렉션
            gc.collect()
            
            # PyTorch 메모리 정리
            if self.device == "mps":
                torch.mps.empty_cache()
                logger.debug("🧹 MPS 메모리 캐시 정리 완료")
                
            elif self.device == "cuda":
                torch.cuda.empty_cache()
                logger.debug("🧹 CUDA 메모리 캐시 정리 완료")
                
        except Exception as e:
            logger.warning(f"메모리 최적화 실패: {e}")
    
    def get_device_info(self) -> Dict[str, Any]:
        """디바이스 상세 정보 반환"""
        
        # 기본 정보
        info = {
            "device": self.device,
            "platform": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
        }
        
        # 메모리 정보
        memory = psutil.virtual_memory()
        info["system_memory"] = {
            "total_gb": round(memory.total / (1024**3), 1),
            "available_gb": round(memory.available / (1024**3), 1),
            "used_percent": memory.percent,
        }
        
        # GPU별 상세 정보
        if self.device == "mps":
            info["gpu_info"] = {
                "name": "Apple M3 Max" if self.is_m3_max else "Apple Silicon",
                "memory_pool": "통합 메모리 (Unified Memory)",
                "cores": "30-40 GPU 코어" if self.is_m3_max else "알 수 없음",
                "neural_engine": "16코어" if self.is_m3_max else "알 수 없음",
                "memory_bandwidth": "400GB/s" if self.is_m3_max else "알 수 없음",
            }
            
        elif self.device == "cuda":
            if torch.cuda.is_available():
                gpu_props = torch.cuda.get_device_properties(0)
                info["gpu_info"] = {
                    "name": gpu_props.name,
                    "memory_gb": round(gpu_props.total_memory / (1024**3), 1),
                    "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                    "multiprocessor_count": gpu_props.multiprocessor_count,
                }
        
        return info
    
    def benchmark_device(self, iterations: int = 100) -> Dict[str, Any]:
        """디바이스 성능 벤치마크"""
        
        logger.info(f"🔧 디바이스 벤치마크 시작 ({iterations}회 반복)...")
        
        try:
            device = torch.device(self.device)
            
            # 테스트 텐서 생성
            size = 1000 if self.is_m3_max else 500
            x = torch.randn(size, size, device=device)
            y = torch.randn(size, size, device=device)
            
            # 워밍업
            for _ in range(10):
                _ = torch.mm(x, y)
            
            # 실제 벤치마크
            import time
            start = time.time()
            
            for _ in range(iterations):
                z = torch.mm(x, y)
                
            end = time.time()
            
            avg_time = (end - start) / iterations
            operations_per_sec = 1.0 / avg_time
            
            result = {
                "success": True,
                "device": self.device,
                "tensor_size": f"{size}x{size}",
                "iterations": iterations,
                "total_time_sec": round(end - start, 4),
                "avg_time_per_operation_ms": round(avg_time * 1000, 4),
                "operations_per_second": round(operations_per_sec, 2),
                "result_shape": list(z.shape),
            }
            
            logger.info(f"✅ 벤치마크 완료: {avg_time*1000:.2f}ms/연산")
            return result
            
        except Exception as e:
            logger.error(f"❌ 벤치마크 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "device": self.device,
            }
    
    def test_ai_pipeline(self) -> bool:
        """AI 파이프라인 테스트"""
        
        logger.info("🧪 AI 파이프라인 테스트 시작...")
        
        try:
            device = torch.device(self.device)
            
            # 간단한 신경망 테스트
            import torch.nn as nn
            
            model = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            ).to(device)
            
            # 테스트 입력
            batch_size = 1 if self.device == "mps" else 4
            x = torch.randn(batch_size, 512, device=device)
            
            # 순전파 테스트
            with torch.no_grad():
                output = model(x)
            
            logger.info(f"✅ AI 파이프라인 테스트 성공: {output.shape}")
            return True
            
        except Exception as e:
            logger.error(f"❌ AI 파이프라인 테스트 실패: {e}")
            return False
    
    def get_optimization_summary(self) -> str:
        """최적화 설정 요약 반환"""
        
        summary = []
        summary.append("🎯 M3 Max GPU 최적화 설정 요약")
        summary.append("=" * 40)
        summary.append(f"디바이스: {self.device}")
        summary.append(f"M3 Max 모드: {'활성화' if self.is_m3_max else '비활성화'}")
        summary.append(f"메모리 할당: {self.memory_fraction * 100}%")
        
        if self.device == "mps":
            summary.append("Metal Performance Shaders 활성화")
            summary.append("통합 메모리 (Unified Memory) 활용")
            
        config = self.get_model_config()
        summary.append(f"배치 크기: {config['batch_size']}")
        summary.append(f"데이터 타입: {config['dtype']}")
        summary.append(f"메모리 효율 모드: {config['memory_efficient']}")
        
        return "\n".join(summary)

# 전역 GPU 설정 인스턴스
gpu_config = M3MaxGPUConfig()

# 자주 사용되는 설정들
DEVICE = gpu_config.device
MODEL_CONFIG = gpu_config.get_model_config()
DEVICE_INFO = gpu_config.get_device_info()

# 초기화 시 정보 출력
logger.info("\n" + gpu_config.get_optimization_summary())

# 유틸리티 함수들
def get_torch_device() -> torch.device:
    """PyTorch 디바이스 객체 반환"""
    return torch.device(DEVICE)

def optimize_model_for_device(model: torch.nn.Module) -> torch.nn.Module:
    """모델을 디바이스에 최적화"""
    model = model.to(DEVICE)
    
    if DEVICE == "mps":
        # MPS 최적화
        model.eval()  # MPS에서는 eval 모드가 더 안정적
        
    elif DEVICE == "cuda":
        # CUDA 최적화
        if MODEL_CONFIG.get("mixed_precision", False):
            model.half()
            
    return model

def create_optimized_tensor(data, dtype=None) -> torch.Tensor:
    """디바이스 최적화된 텐서 생성"""
    if dtype is None:
        dtype = MODEL_CONFIG["dtype"]
        
    if isinstance(data, torch.Tensor):
        return data.to(device=DEVICE, dtype=dtype)
    else:
        return torch.tensor(data, device=DEVICE, dtype=dtype)

# 스타트업 시 자동 실행
def startup_gpu_check():
    """애플리케이션 시작 시 GPU 체크"""
    logger.info("🚀 GPU 설정 시작 검사...")
    
    # 기본 정보 출력
    for key, value in DEVICE_INFO.items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"  {sub_key}: {sub_value}")
        else:
            logger.info(f"{key}: {value}")
    
    # AI 파이프라인 테스트
    gpu_config.test_ai_pipeline()
    
    logger.info("✅ GPU 설정 검사 완료")

if __name__ == "__main__":
    # 직접 실행 시 테스트
    startup_gpu_check()
    
    # 벤치마크 실행
    benchmark_result = gpu_config.benchmark_device()
    print("\n🔧 벤치마크 결과:")
    for key, value in benchmark_result.items():
        print(f"  {key}: {value}")