
import torch
import warnings

# PyTorch 2.7+ 호환성 패치
_original_load = torch.load

def patched_load(f, map_location=None, pickle_module=None, **kwargs):
    """3단계 안전 로딩"""
    try:
        # 1단계: weights_only=True
        return _original_load(f, map_location=map_location, weights_only=True, **kwargs)
    except Exception:
        try:
            # 2단계: weights_only=False  
            warnings.warn("Using weights_only=False for legacy model compatibility")
            return _original_load(f, map_location=map_location, weights_only=False, **kwargs)
        except Exception:
            # 3단계: 기본 로딩
            return _original_load(f, map_location=map_location, **kwargs)

# 패치 적용
torch.load = patched_load
