# app/core/gpu_config.py
"""
MyCloset AI - GPU 설정 및 M3 Max 최적화
수정된 버전: import 오류 해결 및 완전한 M3 Max 지원
"""
import os
import logging
import platform
import subprocess
from typing import Dict, Any, Optional, Tuple

# PyTorch 선택적 import (안전 처리)
try:
    import torch
    import torch.backends.mps
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# psutil 선택적 import
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

logger = logging.getLogger(__name__)

class GPUConfig:
    """M3 Max 최적화 GPU 설정 관리자"""
    
    def __init__(self):
        self.device = "cpu"
        self.device_name = "CPU"
        self.memory_gb = 0
        self.is_apple_silicon = False
        self.is_m3_max = False
        self.optimization_level = "basic"
        
        self._detect_hardware()
        self._configure_device()
        self._apply_optimizations()
    
    def _detect_hardware(self):
        """하드웨어 감지 및 M3 Max 특별 처리"""
        try:
            # macOS 및 Apple Silicon 감지
            is_macos = platform.system() == "Darwin"
            
            if is_macos:
                try:
                    result = subprocess.run(
                        ["sysctl", "-n", "machdep.cpu.brand_string"], 
                        capture_output=True, 
                        text=True, 
                        timeout=5
                    )
                    cpu_brand = result.stdout.strip()
                    
                    if "Apple" in cpu_brand:
                        self.is_apple_silicon = True
                        
                        # M3 Max 특별 감지
                        if "M3 Max" in cpu_brand:
                            self.is_m3_max = True
                            self.optimization_level = "m3_max"
                            logger.info(f"🍎 M3 Max 감지: {cpu_brand}")
                        elif "M3" in cpu_brand:
                            self.optimization_level = "m3"
                        elif "M2" in cpu_brand:
                            self.optimization_level = "m2"
                        elif "M1" in cpu_brand:
                            self.optimization_level = "m1"
                            
                        self.device_name = cpu_brand
                        
                except subprocess.TimeoutExpired:
                    logger.warning("CPU 정보 감지 타임아웃")
                except Exception as e:
                    logger.warning(f"CPU 정보 감지 실패: {e}")
            
            # 메모리 정보 수집
            if PSUTIL_AVAILABLE:
                self.memory_gb = psutil.virtual_memory().total / (1024**3)
            else:
                # 폴백: sysctl로 메모리 정보 획득 (macOS)
                if is_macos:
                    try:
                        result = subprocess.run(
                            ["sysctl", "-n", "hw.memsize"], 
                            capture_output=True, 
                            text=True, 
                            timeout=5
                        )
                        memory_bytes = int(result.stdout.strip())
                        self.memory_gb = memory_bytes / (1024**3)
                    except:
                        self.memory_gb = 16.0  # 기본값
                        
        except Exception as e:
            logger.error(f"하드웨어 감지 실패: {e}")
            self.memory_gb = 8.0  # 안전한 기본값
    
    def _configure_device(self):
        """최적 디바이스 선택"""
        if not TORCH_AVAILABLE:
            self.device = "cpu"
            logger.warning("PyTorch 미설치 - CPU 모드")
            return
            
        # MPS (Apple Silicon) 우선
        if self.is_apple_silicon and torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("✅ MPS (Metal Performance Shaders) 활성화")
            
        # CUDA 두 번째 우선순위
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.device_name = torch.cuda.get_device_name(0)
            logger.info(f"✅ CUDA 활성화: {self.device_name}")
            
        # CPU 폴백
        else:
            self.device = "cpu"
            logger.info("CPU 모드로 실행")
    
    def _apply_optimizations(self):
        """M3 Max 특화 최적화 적용"""
        if not TORCH_AVAILABLE:
            return
            
        try:
            if self.device == "mps":
                self._optimize_mps()
            elif self.device == "cuda":
                self._optimize_cuda()
            else:
                self._optimize_cpu()
                
            logger.info(f"✅ {self.device.upper()} 최적화 완료")
            
        except Exception as e:
            logger.error(f"최적화 적용 실패: {e}")
    
    def _optimize_mps(self):
        """MPS (Apple Silicon) 최적화"""
        try:
            # MPS 환경 변수 설정
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            
            if self.is_m3_max:
                # M3 Max 특별 최적화 (128GB 통합 메모리)
                os.environ["PYTORCH_MPS_ALLOCATOR_POLICY"] = "garbage_collection"
                
                # 대용량 메모리 활용
                if self.memory_gb >= 64:
                    torch.mps.set_per_process_memory_fraction(0.8)
                    logger.info("🚀 M3 Max 대용량 메모리 최적화 활성화")
            
            # MPS 초기화
            if torch.backends.mps.is_available():
                test_tensor = torch.randn(1).to('mps')
                del test_tensor
                torch.mps.empty_cache()
                
            logger.info("🍎 MPS 최적화 완료")
            
        except Exception as e:
            logger.warning(f"MPS 최적화 실패: {e}")
    
    def _optimize_cuda(self):
        """CUDA 최적화"""
        try:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            logger.info("🎮 CUDA 최적화 완료")
        except Exception as e:
            logger.warning(f"CUDA 최적화 실패: {e}")
    
    def _optimize_cpu(self):
        """CPU 최적화"""
        try:
            if PSUTIL_AVAILABLE:
                cpu_cores = psutil.cpu_count(logical=False)
                torch.set_num_threads(min(cpu_cores, 8))
            logger.info("🖥️ CPU 최적화 완료")
        except Exception as e:
            logger.warning(f"CPU 최적화 실패: {e}")
    
    def get_recommended_batch_size(self) -> int:
        """권장 배치 크기 반환"""
        if self.device == "mps":
            if self.is_m3_max and self.memory_gb >= 64:
                return 2  # M3 Max에서 안전한 배치 크기
            return 1
        elif self.device == "cuda":
            return 4
        else:
            return 1
    
    def get_recommended_precision(self) -> str:
        """권장 정밀도 반환"""
        if self.device in ["mps", "cuda"]:
            return "float16"  # Mixed precision 지원
        return "float32"
    
    def get_memory_fraction(self) -> float:
        """메모리 할당 비율 반환"""
        if self.is_m3_max and self.memory_gb >= 64:
            return 0.8  # 128GB에서 80% 사용
        elif self.device == "mps":
            return 0.7
        elif self.device == "cuda":
            return 0.8
        return 1.0
    
    def get_config_dict(self) -> Dict[str, Any]:
        """완전한 설정 딕셔너리 반환"""
        return {
            "device": self.device,
            "device_name": self.device_name,
            "memory_gb": round(self.memory_gb, 1),
            "is_apple_silicon": self.is_apple_silicon,
            "is_m3_max": self.is_m3_max,
            "optimization_level": self.optimization_level,
            "recommended_batch_size": self.get_recommended_batch_size(),
            "recommended_precision": self.get_recommended_precision(),
            "memory_fraction": self.get_memory_fraction(),
            "torch_available": TORCH_AVAILABLE,
            "mps_available": TORCH_AVAILABLE and torch.backends.mps.is_available() if TORCH_AVAILABLE else False,
            "cuda_available": TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False
        }

# 전역 인스턴스 생성 및 export (지연 로딩)
_gpu_config_instance: Optional[GPUConfig] = None

def get_gpu_config() -> GPUConfig:
    """GPU 설정 인스턴스 반환 (싱글톤 패턴)"""
    global _gpu_config_instance
    if _gpu_config_instance is None:
        _gpu_config_instance = GPUConfig()
    return _gpu_config_instance

def configure_gpu() -> Dict[str, Any]:
    """GPU 설정 및 정보 반환"""
    config = get_gpu_config()
    return config.get_config_dict()

def get_optimal_device() -> str:
    """최적 디바이스 반환"""
    config = get_gpu_config()
    return config.device

def is_m3_max() -> bool:
    """M3 Max 여부 확인"""
    config = get_gpu_config()
    return config.is_m3_max

def get_device_info() -> Tuple[str, str, float]:
    """디바이스 정보 반환 (device, name, memory)"""
    config = get_gpu_config()
    return config.device, config.device_name, config.memory_gb

# 환경 변수 자동 설정
def _set_environment_variables():
    """최적화 환경 변수 설정"""
    # PyTorch 환경 변수
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    
    # 멀티프로세싱 최적화
    if PSUTIL_AVAILABLE:
        cpu_count = psutil.cpu_count(logical=False)
        os.environ.setdefault("OMP_NUM_THREADS", str(min(cpu_count, 8)))
        os.environ.setdefault("MKL_NUM_THREADS", str(min(cpu_count, 8)))

# 모듈 로드 시 자동 실행
_set_environment_variables()

# 지연 로딩을 위한 편의 변수들 (실제 사용 시에만 초기화)
def get_gpu_device():
    """현재 GPU 디바이스 반환"""
    return get_gpu_config().device

def get_gpu_info():
    """GPU 정보 딕셔너리 반환"""
    return get_gpu_config().get_config_dict()

# Export할 것들 명시
__all__ = [
    'GPUConfig',
    'get_gpu_config', 
    'configure_gpu',
    'get_optimal_device',
    'is_m3_max',
    'get_device_info',
    'get_gpu_device',
    'get_gpu_info'
]