#!/usr/bin/env python3
"""
PyTorch 2.7+ weights_only 호환성 패치
=====================================
"""

import torch
import warnings
from typing import Any, Optional
from pathlib import Path

def safe_torch_load(file_path: Path, map_location: str = 'cpu') -> Optional[Any]:
    """PyTorch 2.7+ 안전 로딩 함수"""
    try:
        # 1단계: 안전 모드 (weights_only=True)
        try:
            return torch.load(file_path, map_location=map_location, weights_only=True)
        except RuntimeError as e:
            if "legacy .tar format" in str(e) or "TorchScript" in str(e):
                # 2단계: 호환 모드 (weights_only=False)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return torch.load(file_path, map_location=map_location, weights_only=False)
            raise
        
    except Exception as e:
        print(f"⚠️ {file_path} 로딩 실패: {e}")
        return None

# 전역 패치 적용
def apply_pytorch_patch():
    """PyTorch 로딩 함수 패치 적용"""
    original_load = torch.load
    
    def patched_load(f, map_location=None, pickle_module=None, **kwargs):
        # weights_only 기본값 설정
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = True
            
        try:
            return original_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)
        except RuntimeError as e:
            if "legacy .tar format" in str(e) or "TorchScript" in str(e):
                kwargs['weights_only'] = False
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return original_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)
            raise
    
    torch.load = patched_load
    print("✅ PyTorch 2.7 weights_only 호환성 패치 적용 완료")

# 자동 적용
apply_pytorch_patch()
