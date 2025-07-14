# backend/app/core/gpu_config.py
"""
GPU 설정 및 디바이스 관리
Apple Silicon MPS, CUDA, CPU 지원
"""

import torch
import platform
import psutil
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class GPUConfig:
    """GPU 설정 관리 클래스"""
    
    def __init__(self):
        self._device = self._detect_device()
        self._device_info = self._collect_device_info()
        self._initialize_settings()
        
    def _detect_device(self) -> str:
        """최적 디바이스 자동 감지"""
        try:
            # 1. MPS (Apple Silicon) 확인
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
                logger.info("🍎 Apple Silicon MPS 감지됨")
                return "mps"
                
            # 2. CUDA (NVIDIA GPU) 확인
            elif torch.cuda.is_available():
                logger.info(f"🔥 CUDA GPU 감지됨: {torch.cuda.get_device_name(0)}")
                return "cuda"
                
            # 3. CPU 폴백
            else:
                logger.info("💻 CPU 모드로 실행")
                return "cpu"
                
        except Exception as e:
            logger.error(f"디바이스 감지 실패: {e}")
            return "cpu"
    
    def _collect_device_info(self) -> Dict[str, Any]:
        """디바이스 정보 수집"""
        try:
            # 시스템 정보
            info = {
                'device': self._device,
                'platform': platform.platform(),
                'processor': platform.processor(),
                'architecture': platform.architecture()[0],
                'pytorch_version': torch.__version__,
            }
            
            # 메모리 정보
            memory = psutil.virtual_memory()
            info['system_memory_gb'] = round(memory.total / 1024**3, 1)
            info['available_memory_gb'] = round(memory.available / 1024**3, 1)
            
            # 디바이스별 세부 정보
            if self._device == "mps":
                info.update({
                    'mps_available': torch.backends.mps.is_available(),
                    'mps_built': torch.backends.mps.is_built(),
                    'apple_silicon': 'arm' in platform.processor().lower() or 'arm' in platform.machine().lower()
                })
                
            elif self._device == "cuda":
                info.update({
                    'cuda_version': torch.version.cuda,
                    'gpu_count': torch.cuda.device_count(),
                    'gpu_name': torch.cuda.get_device_name(0),
                    'gpu_memory_gb': round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
                })
            
            return info
            
        except Exception as e:
            logger.error(f"디바이스 정보 수집 실패: {e}")
            return {'device': self._device, 'error': str(e)}
    
    def _initialize_settings(self):
        """PyTorch 설정 초기화"""
        try:
            import os
            
            # 스레드 수 설정
            if self._device == "cpu":
                torch.set_num_threads(min(4, psutil.cpu_count()))
            
            # MPS 최적화 설정
            if self._device == "mps":
                os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
            
            # CUDA 최적화 설정
            if self._device == "cuda":
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            
            self._log_device_info()
            logger.info(f"✅ PyTorch 설정 초기화 완료 - Device: {self._device}")
            
        except Exception as e:
            logger.warning(f"PyTorch 설정 초기화 실패: {e}")
    
    def _log_device_info(self):
        """디바이스 정보 로깅"""
        logger.info("=" * 50)
        logger.info("🖥️  시스템 정보")
        logger.info(f"플랫폼: {self._device_info.get('platform', 'Unknown')}")
        logger.info(f"프로세서: {self._device_info.get('processor', 'Unknown')}")
        logger.info(f"아키텍처: {self._device_info.get('architecture', 'Unknown')}")
        
        if self._device_info.get('apple_silicon'):
            logger.info("🍎 Apple Silicon 감지됨")
            
        logger.info(f"PyTorch 버전: {self._device_info.get('pytorch_version', 'Unknown')}")
        logger.info(f"디바이스: {self._device}")
        
        if self._device == "mps":
            logger.info(f"MPS 사용 가능: {self._device_info.get('mps_available', False)}")
            logger.info(f"MPS 빌드됨: {self._device_info.get('mps_built', False)}")
            
        elif self._device == "cuda":
            logger.info(f"CUDA 버전: {self._device_info.get('cuda_version', 'Unknown')}")
            logger.info(f"GPU 개수: {self._device_info.get('gpu_count', 0)}")
            logger.info(f"GPU 이름: {self._device_info.get('gpu_name', 'Unknown')}")
            logger.info(f"GPU 메모리: {self._device_info.get('gpu_memory_gb', 0)}GB")
            
        logger.info(f"시스템 메모리: {self._device_info.get('system_memory_gb', 0)}GB")
        logger.info("=" * 50)
    
    @property
    def device(self) -> str:
        """현재 디바이스 반환"""
        return self._device
    
    @property 
    def device_info(self) -> Dict[str, Any]:
        """디바이스 정보 반환"""
        return self._device_info.copy()
    
    def get_model_config(self) -> Dict[str, Any]:
        """모델 설정 반환"""
        config = {
            "device": self._device,
            "memory_efficient": True,
            "batch_size": 1,
        }
        
        if self._device == "mps":
            config.update({
                "dtype": torch.float32,  # MPS는 float16 불안정할 수 있음
                "max_memory_mb": 16000,  # 안전한 메모리 제한
            })
        elif self._device == "cuda":
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
    
    def optimize_memory(self):
        """메모리 최적화"""
        try:
            if self._device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif self._device == "mps":
                torch.mps.empty_cache()
                torch.mps.synchronize()
                
            # Python 가비지 컬렉션
            import gc
            gc.collect()
            
            logger.debug("메모리 정리 완료")
            
        except Exception as e:
            logger.warning(f"메모리 정리 실패: {e}")
    
    def check_memory_available(self, required_gb: float = 4.0) -> bool:
        """메모리 사용 가능 여부 확인"""
        try:
            if self._device == "cuda" and torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated(0)
                available = (total_memory - allocated_memory) / 1024**3
                return available >= required_gb
                
            elif self._device == "mps":
                # MPS는 시스템 메모리 공유
                memory_info = psutil.virtual_memory()
                available_gb = memory_info.available / 1024**3
                return available_gb * 0.6 >= required_gb  # MPS는 60% 정도 사용 가능
                
            else:
                # CPU 메모리
                memory_info = psutil.virtual_memory()
                available_gb = memory_info.available / 1024**3
                return available_gb >= required_gb
                
        except Exception as e:
            logger.warning(f"메모리 확인 실패: {e}")
            return True  # 확인 실패 시 계속 진행


# 전역 인스턴스 생성
_gpu_config_instance = GPUConfig()

# 전역 변수 export (이전 코드와의 호환성 유지)
DEVICE = _gpu_config_instance.device
DEVICE_INFO = _gpu_config_instance.device_info
MODEL_CONFIG = _gpu_config_instance.get_model_config()

# 함수들 (이전 코드와의 호환성 유지)
def get_device() -> str:
    """현재 디바이스 반환"""
    return _gpu_config_instance.device

def get_device_info() -> Dict[str, Any]:
    """디바이스 정보 반환"""
    return _gpu_config_instance.device_info

def get_optimal_settings() -> Dict[str, Any]:
    """디바이스별 최적 설정 반환"""
    return _gpu_config_instance.get_model_config()

def optimize_memory():
    """메모리 최적화"""
    _gpu_config_instance.optimize_memory()

def check_memory_available(required_gb: float = 4.0) -> bool:
    """메모리 사용 가능 여부 확인"""
    return _gpu_config_instance.check_memory_available(required_gb)

# 클래스 인스턴스도 export (새로운 코드에서 사용)
gpu_config = _gpu_config_instance

# 초기화 로그
logger.info(f"🔧 GPU 설정 완료: {DEVICE}")
logger.info(f"📊 모델 설정: {MODEL_CONFIG}")