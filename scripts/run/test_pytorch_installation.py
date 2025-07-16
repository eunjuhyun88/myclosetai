#!/usr/bin/env python3
"""
PyTorch 설치 확인 및 기능 테스트
"""

import sys
import os

def test_pytorch_installation():
    """PyTorch 설치 및 기능 테스트"""
    print("🔥 PyTorch 설치 테스트 시작...")
    print("=" * 50)
    
    try:
        import torch
        print(f"✅ PyTorch 버전: {torch.__version__}")
        
        # 기본 텐서 생성 테스트
        x = torch.tensor([1.0, 2.0, 3.0])
        print(f"✅ 텐서 생성 성공: {x}")
        
        # 기본 연산 테스트
        y = x * 2 + 1
        print(f"✅ 기본 연산 성공: {y}")
        
        # 디바이스 확인
        print("\n🖥️ 사용 가능한 디바이스:")
        
        # CPU 항상 사용 가능
        print("  ✅ CPU: 사용 가능")
        cpu_tensor = torch.randn(3, 3, device='cpu')
        print(f"     CPU 텐서 테스트: {cpu_tensor.shape}")
        
        # CUDA 확인
        if torch.cuda.is_available():
            print(f"  ✅ CUDA: 사용 가능 ({torch.cuda.get_device_name()})")
            cuda_tensor = torch.randn(3, 3, device='cuda')
            print(f"     CUDA 텐서 테스트: {cuda_tensor.shape}")
            recommended_device = "cuda"
        else:
            print("  ℹ️ CUDA: 사용 불가")
            
        # MPS (Apple Silicon) 확인
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("  ✅ MPS (Apple Silicon): 사용 가능")
            try:
                mps_tensor = torch.randn(3, 3, device='mps')
                print(f"     MPS 텐서 테스트: {mps_tensor.shape}")
                if 'recommended_device' not in locals():
                    recommended_device = "mps"
            except Exception as e:
                print(f"  ⚠️ MPS 테스트 실패: {e}")
                print("     CPU 사용을 권장합니다")
                recommended_device = "cpu"
        else:
            print("  ℹ️ MPS: 사용 불가")
            
        if 'recommended_device' not in locals():
            recommended_device = "cpu"
            
        print(f"\n🎯 권장 디바이스: {recommended_device}")
        
        # 간단한 신경망 테스트
        print("\n🧠 신경망 테스트...")
        model = torch.nn.Linear(3, 2)
        test_input = torch.randn(1, 3)
        
        with torch.no_grad():
            output = model(test_input)
            
        print(f"✅ 신경망 테스트 성공: 입력 {test_input.shape} → 출력 {output.shape}")
        
        # 메모리 사용량 확인
        if recommended_device == "cuda":
            print(f"\n💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
        elif recommended_device == "mps":
            print("\n💾 통합 메모리 사용 (Apple Silicon)")
        
        print("\n🎉 모든 테스트 성공!")
        print(f"✅ PyTorch가 정상적으로 설치되었습니다.")
        print(f"🎯 MyCloset AI에서 사용할 디바이스: {recommended_device}")
        
        return True, recommended_device
        
    except ImportError as e:
        print(f"❌ PyTorch 임포트 실패: {e}")
        return False, "none"
    except Exception as e:
        print(f"❌ PyTorch 테스트 실패: {e}")
        return False, "cpu"

def test_ai_dependencies():
    """AI 관련 의존성 테스트"""
    print("\n📦 AI 의존성 테스트...")
    
    dependencies = [
        ("numpy", "넘파이"),
        ("PIL", "Pillow (이미지 처리)"),
        ("cv2", "OpenCV (컴퓨터 비전)"),
        ("scipy", "SciPy (과학 계산)"),
        ("skimage", "scikit-image (이미지 처리)")
    ]
    
    for package, description in dependencies:
        try:
            if package == "PIL":
                import PIL
                print(f"  ✅ {description}: {PIL.__version__}")
            elif package == "cv2":
                import cv2
                print(f"  ✅ {description}: {cv2.__version__}")
            elif package == "skimage":
                import skimage
                print(f"  ✅ {description}: {skimage.__version__}")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"  ✅ {description}: {version}")
        except ImportError:
            print(f"  ❌ {description}: 설치되지 않음")

if __name__ == "__main__":
    print(f"🐍 Python: {sys.version}")
    print(f"💻 플랫폼: {sys.platform}")
    
    success, device = test_pytorch_installation()
    test_ai_dependencies()
    
    if success:
        print("\n" + "="*50)
        print("🎉 설치 완료! MyCloset AI Backend를 실행할 수 있습니다.")
        print(f"🎯 사용할 디바이스: {device}")
        print("🚀 실행: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    else:
        print("\n" + "="*50)
        print("❌ PyTorch 설치에 문제가 있습니다.")
        print("🔧 해결 방법:")
        print("   conda install pytorch torchvision -c pytorch -y")
        sys.exit(1)
