#!/usr/bin/env python3
"""
안전한 PyTorch 테스트 스크립트
Segmentation fault 없이 PyTorch 테스트
"""

import sys
import os

# 안전 환경변수 설정
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def test_imports():
    """안전한 import 테스트"""
    print("🔍 Python 라이브러리 테스트 중...")
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except Exception as e:
        print(f"❌ NumPy 실패: {e}")
        return False
    
    try:
        # PyTorch를 조심스럽게 import
        print("🔥 PyTorch import 시도 중...")
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        # 기본 텐서 생성 테스트
        x = torch.tensor([1.0, 2.0, 3.0])
        print(f"✅ 텐서 생성 성공: {x}")
        
        # 디바이스 정보
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ Apple MPS 사용 가능 (하지만 CPU 사용 권장)")
            device = "cpu"  # 안정성을 위해 CPU 사용
        elif torch.cuda.is_available():
            print(f"✅ CUDA 사용 가능: {torch.cuda.get_device_name()}")
            device = "cpu"  # 안정성을 위해 CPU 사용
        else:
            print("✅ CPU 모드")
            device = "cpu"
            
        print(f"🎯 사용할 디바이스: {device}")
        
        # 간단한 연산 테스트
        y = x * 2
        print(f"✅ 연산 테스트 성공: {y}")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch 테스트 실패: {e}")
        return False

def test_basic_model():
    """기본 모델 테스트"""
    try:
        import torch
        import torch.nn as nn
        
        print("🧠 기본 모델 테스트 중...")
        
        # 간단한 모델 생성
        model = nn.Linear(3, 1)
        
        # 테스트 입력
        x = torch.tensor([[1.0, 2.0, 3.0]])
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
            
        print(f"✅ 모델 테스트 성공: {output}")
        return True
        
    except Exception as e:
        print(f"❌ 모델 테스트 실패: {e}")
        return False

def main():
    print("🐍 PyTorch 안정성 테스트 시작")
    print("=" * 40)
    
    # 시스템 정보
    print(f"🔧 Python: {sys.version}")
    print(f"💻 플랫폼: {sys.platform}")
    
    # Import 테스트
    if not test_imports():
        print("\n❌ Import 테스트 실패")
        sys.exit(1)
    
    # 모델 테스트
    if not test_basic_model():
        print("\n❌ 모델 테스트 실패")
        sys.exit(1)
    
    print("\n🎉 모든 테스트 성공!")
    print("✅ PyTorch가 안전하게 작동합니다.")
    print("✅ 이제 MyCloset AI 백엔드를 실행할 수 있습니다.")

if __name__ == "__main__":
    main()
