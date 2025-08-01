
import torch
import warnings

# PyTorch 2.7+ 호환성 패치
_original_load = torch.load

def patched_load(f, map_location=None, pickle_module=None, **kwargs):
    """3단계 안전 로딩 + MPS float64 문제 해결"""
    try:
        # 1단계: weights_only=True
        checkpoint = _original_load(f, map_location=map_location, weights_only=True, **kwargs)
    except Exception:
        try:
            # 2단계: weights_only=False  
            warnings.warn("Using weights_only=False for legacy model compatibility")
            checkpoint = _original_load(f, map_location=map_location, weights_only=False, **kwargs)
        except Exception:
            # 3단계: 기본 로딩
            checkpoint = _original_load(f, map_location=map_location, **kwargs)
    
    # 🔥 MPS 디바이스에서 float64 → float32 변환
    if map_location == 'mps' or (hasattr(map_location, 'type') and map_location.type == 'mps'):
        checkpoint = _convert_mps_tensors_to_float32(checkpoint)
    
    return checkpoint

def _convert_mps_tensors_to_float32(checkpoint):
    """MPS용 체크포인트 텐서 변환"""
    def convert_tensor(tensor):
        if hasattr(tensor, 'dtype') and tensor.dtype == torch.float64:
            return tensor.to(torch.float32)
        return tensor
    
    def recursive_convert(obj):
        if torch.is_tensor(obj):
            return convert_tensor(obj)
        elif isinstance(obj, dict):
            return {key: recursive_convert(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(recursive_convert(item) for item in obj)
        else:
            return obj
    
    try:
        return recursive_convert(checkpoint)
    except Exception:
        return checkpoint


# 패치 적용
torch.load = patched_load
