# app/core/gpu_config.py
"""
MyCloset AI - GPU 설정 및 최적화 (M3 Max 특화)
- Apple Silicon MPS 최적화
- CUDA 호환성 
- 메모리 관리
- 성능 튜닝
"""
import os
import logging
import platform
from typing import Dict, Any, Optional, Tuple
import subprocess

# PyTorch 선택적 import
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
    """GPU 설정 및 최적화 관리자"""
    
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
        """하드웨어 감지"""
        try:
            # macOS 여부 확인
            is_macos = platform.system() == "Darwin"
            
            if is_macos:
                # Apple Silicon 감지
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
                            logger.info(f"🍎 M3 감지: {cpu_brand}")
                        elif any(m in cpu_brand for m in ["M1", "M2"]):
                            self.optimization_level = "apple_silicon"
                            logger.info(f"🍎 Apple Silicon 감지: {cpu_brand}")
                
                except Exception as e:
                    logger.warning(f"CPU 정보 감지 실패: {e}")
            
            # 메모리 정보 
            if PSUTIL_AVAILABLE:
                total_memory = psutil.virtual_memory().total
                self.memory_gb = total_memory / (1024 ** 3)
                logger.info(f"💾 시스템 메모리: {self.memory_gb:.1f}GB")
            
        except Exception as e:
            logger.error(f"하드웨어 감지 실패: {e}")
    
    def _configure_device(self):
        """최적 디바이스 설정"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch가 설치되지 않음, CPU 모드로 실행")
            self.device = "cpu"
            self.device_name = "CPU"
            return
        
        # 디바이스 우선순위: MPS > CUDA > CPU
        if torch.backends.mps.is_available() and self.is_apple_silicon:
            self.device = "mps"
            self.device_name = "Apple Silicon MPS"
            logger.info("🚀 MPS (Metal Performance Shaders) 활성화")
            
        elif torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            self.device_name = f"CUDA - {gpu_name}"
            logger.info(f"🚀 CUDA 활성화: {gpu_name}")
            
        else:
            self.device = "cpu"
            self.device_name = "CPU"
            logger.info("⚡ CPU 모드로 실행")
    
    def _apply_optimizations(self):
        """디바이스별 최적화 적용"""
        if not TORCH_AVAILABLE:
            return
        
        try:
            # 공통 최적화
            torch.set_grad_enabled(False)  # 추론 모드
            
            if self.device == "mps":
                self._optimize_mps()
            elif self.device == "cuda":
                self._optimize_cuda()
            elif self.device == "cpu":
                self._optimize_cpu()
            
            logger.info(f"✅ {self.device.upper()} 최적화 완료")
            
        except Exception as e:
            logger.error(f"최적화 적용 실패: {e}")
    
    def _optimize_mps(self):
        """MPS (Apple Silicon) 최적화"""
        if not torch.backends.mps.is_available():
            return
        
        try:
            # MPS 최적화 설정
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # 메모리 관리
            
            if self.is_m3_max:
                # M3 Max 특별 최적화
                os.environ["PYTORCH_MPS_ALLOCATOR_POLICY"] = "garbage_collection"
                
                # M3 Max는 128GB 통합 메모리 활용 가능
                if self.memory_gb >= 64:
                    # 대용량 메모리 최적화
                    torch.backends.mps.empty_cache = lambda: None  # 캐시 관리 커스터마이징
            
            # MPS 백엔드 최적화
            torch.backends.mps.is_available()  # MPS 초기화 트리거
            
            logger.info("🍎 MPS 최적화 설정 완료")
            
        except Exception as e:
            logger.warning(f"MPS 최적화 실패: {e}")
    
    def _optimize_cuda(self):
        """CUDA 최적화"""
        try:
            # CUDA 최적화 설정
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # 메모리 관리
            torch.cuda.empty_cache()
            
            # 멀티 GPU 지원 (필요시)
            if torch.cuda.device_count() > 1:
                logger.info(f"🚀 {torch.cuda.device_count()}개 GPU 감지")
            
            logger.info("🎮 CUDA 최적화 설정 완료")
            
        except Exception as e:
            logger.warning(f"CUDA 최적화 실패: {e}")
    
    def _optimize_cpu(self):
        """CPU 최적화"""
        try:
            # CPU 스레드 최적화
            if hasattr(torch, 'set_num_threads'):
                # 물리 코어 수 감지
                physical_cores = psutil.cpu_count(logical=False) if PSUTIL_AVAILABLE else 4
                torch.set_num_threads(min(physical_cores, 8))  # 최대 8스레드
            
            # Intel MKL 최적화 (Intel CPU)
            if "intel" in platform.processor().lower():
                os.environ["MKL_NUM_THREADS"] = str(min(4, psutil.cpu_count(logical=False) if PSUTIL_AVAILABLE else 4))
            
            logger.info("⚡ CPU 최적화 설정 완료")
            
        except Exception as e:
            logger.warning(f"CPU 최적화 실패: {e}")
    
    def get_recommended_batch_size(self, model_size: str = "medium") -> int:
        """모델 크기별 권장 배치 크기"""
        
        # 기본 배치 크기 매핑
        base_batch_sizes = {
            "small": {"mps": 8, "cuda": 16, "cpu": 4},
            "medium": {"mps": 4, "cuda": 8, "cpu": 2}, 
            "large": {"mps": 2, "cuda": 4, "cpu": 1},
            "xlarge": {"mps": 1, "cuda": 2, "cpu": 1}
        }
        
        base_size = base_batch_sizes.get(model_size, base_batch_sizes["medium"])[self.device]
        
        # M3 Max 특별 조정
        if self.is_m3_max and self.memory_gb >= 64:
            base_size = min(base_size * 2, 16)  # 최대 2배, 상한 16
        
        return base_size
    
    def get_recommended_precision(self) -> str:
        """권장 정밀도"""
        if self.device == "mps":
            # MPS는 float16 지원이 제한적
            return "float32"
        elif self.device == "cuda":
            return "float16"  # GPU에서는 mixed precision 활용
        else:
            return "float32"
    
    def get_memory_fraction(self) -> float:
        """사용할 메모리 비율"""
        if self.device == "mps":
            # MPS는 통합 메모리, 보수적 접근
            if self.is_m3_max and self.memory_gb >= 64:
                return 0.6  # 60% 사용
            else:
                return 0.4  # 40% 사용
        elif self.device == "cuda":
            return 0.8  # GPU 메모리의 80%
        else:
            return 0.5  # CPU 메모리의 50%
    
    def setup_multiprocessing(self):
        """멀티프로세싱 설정"""
        if self.device == "mps":
            # MPS는 멀티프로세싱 제한
            torch.multiprocessing.set_start_method('spawn', force=True)
            return 1  # 단일 프로세스
        else:
            # CPU/CUDA는 멀티프로세싱 가능
            max_workers = min(4, psutil.cpu_count(logical=False) if PSUTIL_AVAILABLE else 2)
            return max_workers
    
    def get_config_dict(self) -> Dict[str, Any]:
        """설정 딕셔너리 반환"""
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
            "max_workers": self.setup_multiprocessing()
        }

# 전역 GPU 설정 인스턴스
_gpu_config: Optional[GPUConfig] = None

def get_gpu_config() -> GPUConfig:
    """GPU 설정 인스턴스 반환 (싱글톤)"""
    global _gpu_config
    if _gpu_config is None:
        _gpu_config = GPUConfig()
    return _gpu_config

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

# 환경 변수 설정 (import 시 자동 실행)
def _set_environment_variables():
    """환경 변수 자동 설정"""
    
    # PyTorch 환경 변수
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    
    # 멀티프로세싱 설정
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    
    # CUDA 설정 (있는 경우)
    os.environ.setdefault("CUDA_CACHE_DISABLE", "0")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")

# 모듈 로드 시 자동 실행
_set_environment_variables()