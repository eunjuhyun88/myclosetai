#!/usr/bin/env python3
"""
디바이스 안전 설정 (자동 생성)
"""
import torch

def get_safe_device():
    """안전한 디바이스 반환"""
    if torch.backends.mps.is_available():
        try:
            # MPS 간단 테스트
            test_tensor = torch.randn(1, 1).to('mps')
            return 'mps'
        except Exception:
            return 'cpu'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def safe_model_load(model_path, target_device=None):
    """디바이스 안전 모델 로딩"""
    if target_device is None:
        target_device = get_safe_device()
    
    # 1단계: CPU로 로딩
    model_data = torch.load(model_path, map_location='cpu')
    
    # 2단계: 타겟 디바이스로 이동 (torch.jit 아닌 경우만)
    if target_device != 'cpu' and not hasattr(model_data, '_c'):
        if isinstance(model_data, dict) and 'state_dict' in model_data:
            new_state_dict = {}
            for key, tensor in model_data['state_dict'].items():
                new_state_dict[key] = tensor.to(target_device)
            model_data['state_dict'] = new_state_dict
        elif hasattr(model_data, 'to'):
            model_data = model_data.to(target_device)
    
    return model_data, target_device

if __name__ == "__main__":
    print(f"🎯 권장 디바이스: {get_safe_device()}")
