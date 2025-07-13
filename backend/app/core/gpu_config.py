"""
GPU 설정 및 디바이스 관리 (호환성 개선 버전)
"""

import torch
import logging

logger = logging.getLogger(__name__)

def setup_device():
    """디바이스 설정"""
    try:
        if torch.backends.mps.is_available():
            device = "mps"
            print("🍎 Apple Silicon MPS 사용")
            
            # MPS 캐시 정리 (버전 호환성 체크)
            try:
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
            except AttributeError:
                logger.info("MPS empty_cache 미지원 (PyTorch 버전 문제)")
                
        elif torch.cuda.is_available():
            device = "cuda"
            print("🚀 NVIDIA CUDA 사용")
            torch.cuda.empty_cache()
        else:
            device = "cpu"
            print("💻 CPU 사용")
            
    except Exception as e:
        logger.warning(f"디바이스 설정 중 오류: {e}")
        device = "cpu"
        print("💻 CPU 사용 (fallback)")
    
    return device

# 글로벌 설정
DEVICE = setup_device()

MODEL_CONFIG = {
    "device": DEVICE,
    "dtype": torch.float16 if DEVICE in ["cuda", "mps"] else torch.float32,
    "max_batch_size": 4 if DEVICE != "cpu" else 1
}

gpu_config = {
    "device": DEVICE,
    "available": DEVICE != "cpu",
    "config": MODEL_CONFIG
}

DEVICE_INFO = {
    "device": DEVICE,
    "torch_version": torch.__version__,
    "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
    "cuda_available": torch.cuda.is_available()
}

logger.info(f"디바이스 설정 완료: {DEVICE}")
