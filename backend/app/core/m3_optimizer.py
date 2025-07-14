# app/core/m3_optimizer.py
"""
M3 Max 128GB 특화 최적화 시스템
- 40코어 GPU + 16코어 Neural Engine 활용
- 128GB 통합 메모리 최적화
- 400GB/s 메모리 대역폭 극대화
"""

import os
import logging
import torch
import psutil
import platform
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class M3MaxSpecs:
    """M3 Max 하드웨어 스펙"""
    gpu_cores: int = 40
    neural_engine_cores: int = 16
    cpu_performance_cores: int = 12
    cpu_efficiency_cores: int = 4
    total_memory_gb: int = 128
    memory_bandwidth_gbps: int = 400
    max_memory_allocation_gb: int = 100  # 80% 사용 권장

class M3MaxOptimizer:
    """M3 Max 128GB 특화 최적화"""
    
    def __init__(self):
        self.specs = M3MaxSpecs()
        self.device = "mps"
        self.is_m3_max = self._verify_m3_max()
        self.optimization_config = self._create_optimization_config()
        
        logger.info("🍎 M3 Max 128GB 최적화 시스템 초기화")
        
        if self.is_m3_max:
            self._apply_m3_max_optimizations()
        else:
            logger.warning("⚠️ M3 Max가 아닌 환경에서 실행 중")
    
    def _verify_m3_max(self) -> bool:
        """M3 Max 환경 확인"""
        try:
            system_info = platform.uname()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # M3 Max 확인 조건
            is_apple_silicon = (
                system_info.system == "Darwin" and 
                system_info.machine == "arm64"
            )
            
            has_sufficient_memory = memory_gb >= 100  # 128GB 중 100GB+ 인식
            has_mps = torch.backends.mps.is_available()
            
            if is_apple_silicon and has_sufficient_memory and has_mps:
                logger.info(f"✅ M3 Max 환경 확인: {memory_gb:.0f}GB 메모리")
                return True
            else:
                logger.info(f"❌ M3 Max 환경 아님: {memory_gb:.0f}GB 메모리")
                return False
                
        except Exception as e:
            logger.warning(f"M3 Max 확인 실패: {e}")
            return False
    
    def _create_optimization_config(self) -> Dict[str, Any]:
        """M3 Max 최적화 설정 생성"""
        config = {
            # GPU 설정
            "device": "mps",
            "dtype": torch.float32,  # MPS 최적화
            "memory_fraction": 0.8,  # 128GB 중 80% 활용
            
            # 배치 처리 최적화
            "max_batch_size": 8,  # 대용량 메모리 활용
            "prefetch_factor": 4,
            "num_workers": 8,  # CPU 코어 수 맞춤
            
            # 메모리 관리
            "memory_pool_size_gb": 80,  # 80GB 메모리 풀
            "cache_size_gb": 20,  # 20GB 캐시
            "swap_threshold": 0.9,
            
            # Neural Engine 활용
            "neural_engine_enabled": True,
            "coreml_optimization": True,
            
            # Metal Performance Shaders 최적화
            "mps_optimization": {
                "enable_fusion": True,
                "enable_memory_efficient_attention": False,  # MPS 호환성
                "enable_gradient_checkpointing": True,
                "max_split_size_mb": 256
            },
            
            # 모델별 최적화
            "model_optimizations": {
                "diffusion_models": {
                    "attention_slicing": True,
                    "cpu_offload": False,  # 충분한 메모리로 GPU 유지
                    "sequential_cpu_offload": False,
                    "enable_vae_slicing": False
                },
                "segmentation_models": {
                    "tile_size": 1024,  # 고해상도 처리
                    "overlap": 64,
                    "batch_inference": True
                },
                "pose_estimation": {
                    "multi_scale": True,
                    "high_precision": True
                }
            }
        }
        
        return config
    
    def _apply_m3_max_optimizations(self) -> None:
        """M3 Max 시스템 최적화 적용"""
        try:
            # 환경 변수 설정
            env_vars = {
                # PyTorch MPS 최적화
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
                "PYTORCH_MPS_ALLOCATOR_POLICY": "garbage_collection",
                "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                
                # M3 Max CPU 최적화
                "OMP_NUM_THREADS": str(self.specs.cpu_performance_cores),
                "MKL_NUM_THREADS": str(self.specs.cpu_performance_cores),
                "NUMBA_NUM_THREADS": str(self.specs.cpu_performance_cores),
                
                # 메모리 최적화
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256",
                "MALLOC_ARENA_MAX": "4",
                
                # Metal 가속
                "METAL_DEVICE_WRAPPER_TYPE": "1",
                "METAL_PERFORMANCE_SHADERS_ENABLED": "1"
            }
            
            os.environ.update(env_vars)
            
            # PyTorch 최적화 설정
            if torch.backends.mps.is_available():
                # MPS 메모리 관리 (PyTorch 버전 호환성 체크)
                try:
                    # PyTorch 2.4 이하
                    if hasattr(torch.backends.mps, 'empty_cache'):
                        torch.backends.mps.empty_cache()
                        logger.info("✅ MPS 캐시 정리 완료")
                except AttributeError:
                    # PyTorch 2.5+ 대체 방법
                    logger.info("ℹ️ PyTorch 2.5+ 환경 - 대체 메모리 관리 사용")
            
            # CPU 최적화
            torch.set_num_threads(self.specs.cpu_performance_cores)
            
            logger.info("🚀 M3 Max 최적화 설정 적용 완료")
            
        except Exception as e:
            logger.error(f"❌ M3 Max 최적화 적용 실패: {e}")
    
    def get_optimal_batch_size(self, model_type: str = "diffusion") -> int:
        """모델 타입별 최적 배치 크기 계산"""
        base_sizes = {
            "diffusion": 4,      # Stable Diffusion 계열
            "segmentation": 8,   # U-Net 계열
            "pose": 16,          # 포즈 추정
            "classification": 32 # 분류 모델
        }
        
        base_size = base_sizes.get(model_type, 4)
        
        # M3 Max 128GB 메모리 활용도에 따른 배치 크기 조정
        if self.is_m3_max:
            memory_multiplier = min(self.specs.total_memory_gb / 32, 4.0)  # 최대 4배
            optimal_size = int(base_size * memory_multiplier)
            return min(optimal_size, 16)  # 안정성을 위해 최대 16으로 제한
        
        return base_size
    
    def get_memory_allocation(self) -> Dict[str, int]:
        """메모리 할당 계획"""
        total_gb = self.specs.total_memory_gb
        
        allocation = {
            "ai_models": int(total_gb * 0.6),      # 76.8GB - AI 모델
            "image_cache": int(total_gb * 0.15),   # 19.2GB - 이미지 캐시
            "system_buffer": int(total_gb * 0.1),  # 12.8GB - 시스템 버퍼
            "temp_processing": int(total_gb * 0.1), # 12.8GB - 임시 처리
            "os_reserved": int(total_gb * 0.05)    # 6.4GB - OS 예약
        }
        
        return allocation
    
    def optimize_for_model(self, model_name: str) -> Dict[str, Any]:
        """특정 모델에 대한 최적화 설정"""
        base_config = self.optimization_config.copy()
        
        if "diffusion" in model_name.lower():
            # Stable Diffusion 최적화
            base_config.update({
                "batch_size": self.get_optimal_batch_size("diffusion"),
                "attention_slicing": True,
                "memory_efficient_attention": False,  # MPS 호환성
                "enable_xformers": False,  # MPS에서 미지원
                "gradient_checkpointing": True
            })
            
        elif "segmentation" in model_name.lower() or "u2net" in model_name.lower():
            # 세그멘테이션 모델 최적화
            base_config.update({
                "batch_size": self.get_optimal_batch_size("segmentation"),
                "tile_processing": True,
                "tile_size": 1024,
                "enable_amp": False  # MPS AMP 호환성 이슈
            })
            
        elif "pose" in model_name.lower() or "openpose" in model_name.lower():
            # 포즈 추정 최적화
            base_config.update({
                "batch_size": self.get_optimal_batch_size("pose"),
                "multi_scale": True,
                "nms_threshold": 0.5
            })
        
        return base_config
    
    def monitor_performance(self) -> Dict[str, Any]:
        """실시간 성능 모니터링"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            "memory_used_gb": (memory.total - memory.available) / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "memory_usage_percent": memory.percent,
            "cpu_usage_percent": cpu_percent,
            "estimated_ai_memory_gb": self.get_memory_allocation()["ai_models"],
            "optimization_active": self.is_m3_max,
            "device": self.device,
            "batch_size_recommendation": self.get_optimal_batch_size()
        }
    
    def create_model_config(self, model_type: str) -> Dict[str, Any]:
        """모델별 최적화 설정 생성"""
        return {
            "device": self.device,
            "torch_dtype": torch.float32,
            "low_cpu_mem_usage": False,  # 충분한 메모리 있음
            "device_map": None,  # 단일 GPU 사용
            "max_memory": {0: f"{self.specs.max_memory_allocation_gb}GB"},
            "offload_folder": None,  # 메모리 충분으로 offload 불필요
            "offload_state_dict": False,
            "use_safetensors": True,
            "variant": "fp32",  # MPS 최적화
            
            # M3 Max 특화 설정
            "enable_attention_slicing": True,
            "enable_cpu_offload": False,
            "enable_model_cpu_offload": False,
            "enable_sequential_cpu_offload": False,
            
            # 메모리 최적화
            "memory_efficient_attention": False,  # MPS 호환성
            "use_memory_efficient_attention_xformers": False,
            "attention_slice_size": "auto",
            
            # 성능 설정
            "num_images_per_prompt": 1,
            "batch_size": self.get_optimal_batch_size(model_type),
            "max_batch_size": 8,
            
            # 품질 설정
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
            "scheduler": "DPMSolverMultistepScheduler"
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 반환"""
        return {
            "hardware": {
                "chip": "Apple M3 Max",
                "gpu_cores": self.specs.gpu_cores,
                "neural_engine": f"{self.specs.neural_engine_cores} cores",
                "cpu_cores": f"{self.specs.cpu_performance_cores} performance + {self.specs.cpu_efficiency_cores} efficiency",
                "memory": f"{self.specs.total_memory_gb}GB Unified Memory",
                "memory_bandwidth": f"{self.specs.memory_bandwidth_gbps}GB/s"
            },
            "optimization": {
                "enabled": self.is_m3_max,
                "device": self.device,
                "memory_allocation": self.get_memory_allocation(),
                "batch_size_diffusion": self.get_optimal_batch_size("diffusion"),
                "batch_size_segmentation": self.get_optimal_batch_size("segmentation")
            },
            "pytorch": {
                "version": torch.__version__,
                "mps_available": torch.backends.mps.is_available(),
                "cuda_available": torch.cuda.is_available()
            }
        }

# 전역 최적화 인스턴스
_m3_optimizer: Optional[M3MaxOptimizer] = None

def get_m3_optimizer() -> M3MaxOptimizer:
    """M3 Max 최적화 인스턴스 반환"""
    global _m3_optimizer
    if _m3_optimizer is None:
        _m3_optimizer = M3MaxOptimizer()
    return _m3_optimizer

def is_m3_max_optimized() -> bool:
    """M3 Max 최적화 활성 여부"""
    optimizer = get_m3_optimizer()
    return optimizer.is_m3_max

def get_optimal_config(model_type: str = "diffusion") -> Dict[str, Any]:
    """최적화된 모델 설정 반환"""
    optimizer = get_m3_optimizer()
    return optimizer.create_model_config(model_type)

# 추가 호환성 함수들
def create_m3_optimizer() -> M3MaxOptimizer:
    """M3 최적화 인스턴스 생성"""
    return M3MaxOptimizer()

def get_m3_config() -> Dict[str, Any]:
    """M3 Max 설정 반환"""
    optimizer = get_m3_optimizer()
    return optimizer.get_system_info()

def optimize_for_m3_max() -> bool:
    """M3 Max 최적화 적용"""
    optimizer = get_m3_optimizer()
    return optimizer.is_m3_max