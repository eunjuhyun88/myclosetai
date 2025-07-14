# app/core/gpu_config.py
"""
M3 Max GPU 설정 및 최적화 (PyTorch 2.5.1 호환)
- 누락된 export 함수들 추가
- main.py에서 요구하는 모든 함수 구현
"""

import os
import logging
import platform
import psutil
import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GPUInfo:
    """GPU 정보"""
    device: str
    name: str
    memory_gb: float
    is_m3_max: bool
    pytorch_version: str
    mps_available: bool

class GPUConfig:
    """M3 Max GPU 설정 매니저"""
    
    def __init__(self):
        self.device = self._detect_optimal_device()
        self.gpu_info = self._get_gpu_info()
        self.config = self._create_config()
        
        # 필수 export
        self.DEVICE = self.device
        self.MODEL_CONFIG = self.config
        
        logger.info(f"🚀 GPU 설정 초기화 완료: {self.device}")
        self._apply_optimizations()
    
    def _detect_optimal_device(self) -> str:
        """최적 디바이스 감지"""
        # CUDA 우선 확인
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"🎮 CUDA GPU 감지: {device_name}")
            return "cuda"
        
        # MPS (Apple Silicon) 확인
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("🍎 Apple Silicon MPS 감지")
            return "mps"
        
        # CPU 폴백
        logger.info("💻 CPU 모드 사용")
        return "cpu"
    
    def _get_gpu_info(self) -> GPUInfo:
        """GPU 정보 수집"""
        system_info = platform.uname()
        total_memory = psutil.virtual_memory().total / (1024**3)
        
        # M3 Max 감지 (메모리 크기 + ARM64 아키텍처)
        is_m3_max = (
            system_info.system == "Darwin" and 
            system_info.machine == "arm64" and 
            total_memory >= 32.0  # M3 Max는 36GB+ 통합 메모리
        )
        
        if self.device == "cuda":
            name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        elif self.device == "mps":
            name = "Apple Silicon MPS"
            if is_m3_max:
                name = "Apple M3 Max (MPS)"
            memory_gb = total_memory  # MPS는 통합 메모리 사용
        else:
            name = "CPU"
            memory_gb = total_memory
        
        return GPUInfo(
            device=self.device,
            name=name,
            memory_gb=memory_gb,
            is_m3_max=is_m3_max,
            pytorch_version=torch.__version__,
            mps_available=hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        )
    
    def _create_config(self) -> Dict[str, Any]:
        """디바이스별 최적 설정 생성"""
        base_config = {
            "device": self.device,
            "memory_fraction": 0.8,
            "enable_attention_slicing": True,
            "mixed_precision": True,
        }
        
        if self.device == "cuda":
            base_config.update({
                "dtype": torch.float16,
                "enable_memory_efficient_attention": True,
                "enable_xformers": True,
                "enable_cpu_offload": False,
            })
        
        elif self.device == "mps":
            # M3 Max 특화 설정
            if self.gpu_info.is_m3_max:
                base_config.update({
                    "dtype": torch.float32,  # MPS는 float32 권장
                    "memory_fraction": 0.85,  # M3 Max는 통합 메모리로 더 높게
                    "enable_memory_efficient_attention": False,  # MPS 호환성
                    "enable_cpu_offload": False,
                    "batch_size_multiplier": 1.5,  # M3 Max 성능 활용
                })
            else:
                base_config.update({
                    "dtype": torch.float32,
                    "memory_fraction": 0.7,
                    "enable_memory_efficient_attention": False,
                    "enable_cpu_offload": True,
                })
        
        else:  # CPU
            base_config.update({
                "dtype": torch.float32,
                "enable_memory_efficient_attention": False,
                "enable_cpu_offload": True,
                "num_threads": min(psutil.cpu_count(logical=False), 8),
            })
        
        return base_config
    
    def _apply_optimizations(self) -> None:
        """디바이스별 최적화 적용"""
        try:
            if self.device == "cuda":
                self._optimize_cuda()
            elif self.device == "mps":
                self._optimize_mps()
            else:
                self._optimize_cpu()
            
            logger.info(f"✅ {self.device.upper()} 최적화 적용 완료")
            
        except Exception as e:
            logger.error(f"❌ 최적화 적용 실패: {e}")
    
    def _optimize_cuda(self) -> None:
        """CUDA 최적화"""
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # 메모리 관리
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        logger.info("🎮 CUDA 최적화 완료")
    
    def _optimize_mps(self) -> None:
        """MPS (Apple Silicon) 최적화"""
        try:
            # 환경변수 설정
            optimization_env = {
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
            }
            
            if self.gpu_info.is_m3_max:
                # M3 Max 특화 최적화
                optimization_env.update({
                    "PYTORCH_MPS_ALLOCATOR_POLICY": "garbage_collection",
                    "OMP_NUM_THREADS": "8",  # M3 Max 성능 코어 수
                    "MKL_NUM_THREADS": "8",
                })
                logger.info("🍎 M3 Max 특화 최적화 적용")
            
            os.environ.update(optimization_env)
            
            # PyTorch 2.5.1 호환성 체크
            if hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
                logger.info("✅ MPS 캐시 정리 완료")
            else:
                logger.info("ℹ️ MPS empty_cache 미지원 - 대체 메모리 관리 사용")
            
            logger.info("🍎 MPS 최적화 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ MPS 최적화 실패: {e}")
    
    def _optimize_cpu(self) -> None:
        """CPU 최적화"""
        try:
            # 스레드 수 최적화
            num_threads = min(psutil.cpu_count(logical=False), 8)
            torch.set_num_threads(num_threads)
            
            # Intel MKL 최적화 (Intel CPU)
            os.environ.update({
                "OMP_NUM_THREADS": str(num_threads),
                "MKL_NUM_THREADS": str(num_threads),
                "NUMBA_NUM_THREADS": str(num_threads),
            })
            
            logger.info(f"💻 CPU 최적화 완료 (스레드: {num_threads})")
            
        except Exception as e:
            logger.warning(f"⚠️ CPU 최적화 실패: {e}")
    
    def get_device_info(self) -> Dict[str, Any]:
        """디바이스 정보 반환"""
        return {
            "device": self.gpu_info.device,
            "name": self.gpu_info.name,
            "memory_gb": round(self.gpu_info.memory_gb, 1),
            "is_m3_max": self.gpu_info.is_m3_max,
            "pytorch_version": self.gpu_info.pytorch_version,
            "mps_available": self.gpu_info.mps_available,
            "config": self.config
        }
    
    def test_device(self) -> bool:
        """디바이스 테스트"""
        try:
            # 간단한 텐서 연산 테스트
            test_tensor = torch.randn(1, 100, device=self.device)
            result = torch.nn.functional.relu(test_tensor)
            
            logger.info(f"✅ {self.device} 디바이스 테스트 성공")
            return True
            
        except Exception as e:
            logger.error(f"❌ {self.device} 디바이스 테스트 실패: {e}")
            return False
    
    def get_optimal_settings(self, model_type: str = "diffusion") -> Dict[str, Any]:
        """최적 설정 반환 (누락된 함수 추가)"""
        base_settings = self.config.copy()
        
        # 모델 타입별 특화 설정
        if model_type == "diffusion":
            if self.device == "mps" and self.gpu_info.is_m3_max:
                base_settings.update({
                    "batch_size": 2,
                    "num_inference_steps": 20,
                    "guidance_scale": 7.5,
                    "enable_attention_slicing": True
                })
            elif self.device == "cuda":
                base_settings.update({
                    "batch_size": 4,
                    "num_inference_steps": 50,
                    "guidance_scale": 7.5,
                    "enable_xformers": True
                })
            else:
                base_settings.update({
                    "batch_size": 1,
                    "num_inference_steps": 20,
                    "guidance_scale": 7.5
                })
        
        elif model_type == "segmentation":
            base_settings.update({
                "input_size": (512, 512),
                "confidence_threshold": 0.5,
                "use_tta": self.device != "cpu"
            })
        
        elif model_type == "pose_estimation":
            base_settings.update({
                "input_size": (368, 368),
                "confidence_threshold": 0.1,
                "use_tensorrt": self.device == "cuda"
            })
        
        return base_settings

# 전역 설정 초기화
def initialize_gpu_config():
    """GPU 설정 초기화 및 전역 변수 설정"""
    global gpu_config, DEVICE, MODEL_CONFIG
    
    try:
        gpu_config = GPUConfig()
        DEVICE = gpu_config.DEVICE
        MODEL_CONFIG = gpu_config.MODEL_CONFIG
        
        logger.info(f"🚀 GPU 설정 초기화 완료: {DEVICE}")
        
        # 디바이스 테스트
        if gpu_config.test_device():
            logger.info("✅ GPU 설정 검증 완료")
        else:
            logger.error("❌ GPU 설정 검증 실패")
            
        return gpu_config
        
    except Exception as e:
        logger.error(f"❌ GPU 설정 초기화 실패: {e}")
        
        # 안전한 폴백 설정
        DEVICE = "cpu"
        MODEL_CONFIG = {
            "device": "cpu",
            "dtype": torch.float32,
            "enable_memory_efficient_attention": False,
        }
        return None

# 초기화 실행
gpu_config = initialize_gpu_config()

# 하위 호환성을 위한 추가 exports
if gpu_config:
    DEVICE = gpu_config.DEVICE
    MODEL_CONFIG = gpu_config.MODEL_CONFIG
    DEVICE_INFO = gpu_config.get_device_info()
else:
    DEVICE = "cpu"
    MODEL_CONFIG = {"device": "cpu", "dtype": torch.float32}
    DEVICE_INFO = {
        "device": "cpu",
        "name": "CPU",
        "memory_gb": 0,
        "is_m3_max": False,
        "pytorch_version": torch.__version__,
        "mps_available": False
    }

# ==========================================
# MAIN.PY에서 요구하는 EXPORT 함수들 추가
# ==========================================

def get_device_config():
    """디바이스 설정 반환 (호환성)"""
    return {
        "device": DEVICE,
        "model_config": MODEL_CONFIG,
        "device_info": DEVICE_INFO
    }

def get_device():
    """디바이스 정보 반환 (호환성)"""
    return DEVICE

def get_model_config():
    """모델 설정 반환 (호환성)"""
    return MODEL_CONFIG

def get_device_info():
    """디바이스 정보 반환 (호환성)"""
    return DEVICE_INFO

def get_optimal_settings(model_type: str = "diffusion") -> Dict[str, Any]:
    """최적 설정 반환 (main.py에서 누락된 함수)"""
    if gpu_config and hasattr(gpu_config, 'get_optimal_settings'):
        return gpu_config.get_optimal_settings(model_type)
    else:
        # 폴백 설정
        return {
            "device": DEVICE,
            "dtype": torch.float32,
            "batch_size": 1,
            "memory_fraction": 0.8,
            "enable_optimization": True
        }

def set_device_optimization(enable: bool = True) -> bool:
    """디바이스 최적화 설정/해제"""
    try:
        if gpu_config:
            if enable:
                gpu_config._apply_optimizations()
            logger.info(f"🔧 디바이스 최적화 {'활성화' if enable else '비활성화'}")
            return True
        else:
            logger.warning("GPU 설정이 초기화되지 않음")
            return False
    except Exception as e:
        logger.error(f"디바이스 최적화 설정 실패: {e}")
        return False

def get_performance_info() -> Dict[str, Any]:
    """성능 정보 반환"""
    if gpu_config:
        info = gpu_config.get_device_info()
        info.update({
            "optimization_applied": True,
            "pytorch_version": torch.__version__,
            "performance_tips": _get_performance_tips()
        })
        return info
    else:
        return {
            "device": "cpu",
            "optimization_applied": False,
            "performance_tips": ["GPU 설정을 초기화하세요"]
        }

def _get_performance_tips() -> list:
    """성능 향상 팁 반환"""
    tips = []
    
    if DEVICE == "mps":
        tips.extend([
            "MPS 사용 중 - M3 Max에 최적화됨",
            "통합 메모리의 장점을 활용하세요",
            "배치 크기를 늘려 성능을 향상시킬 수 있습니다"
        ])
    elif DEVICE == "cuda":
        tips.extend([
            "CUDA 가속 활성화됨",
            "메모리 사용량을 모니터링하세요",
            "혼합 정밀도 연산을 활용하세요"
        ])
    else:
        tips.extend([
            "CPU 모드 사용 중",
            "GPU가 사용 가능한지 확인하세요",
            "스레드 수 최적화가 적용됨"
        ])
    
    return tips