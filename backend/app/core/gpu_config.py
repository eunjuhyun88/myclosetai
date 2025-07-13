"""
GPU 및 디바이스 설정
"""
import torch

# 디바이스 자동 감지
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"🚀 CUDA GPU 사용: {torch.cuda.get_device_name()}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps" 
    print("🍎 Apple Silicon MPS 사용")
else:
    DEVICE = "cpu"
    print("💻 CPU 사용")

# 모델 설정
MODEL_CONFIG = {
    "device": DEVICE,
    "dtype": torch.float32 if DEVICE == "mps" else torch.float16,
    "memory_fraction": 0.8,
    "enable_attention_slicing": True,
    "enable_memory_efficient_attention": DEVICE != "mps"
}

# GPU 메모리 최적화
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
elif DEVICE == "mps":
    # MPS 최적화 설정
    torch.backends.mps.empty_cache()
