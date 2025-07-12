#!/usr/bin/env python3
"""
MyCloset AI MVP - 설치 확인 테스트
모든 패키지가 정상적으로 설치되었는지 확인
"""

import sys
import subprocess

def test_imports():
    """필수 패키지 import 테스트"""
    
    packages = [
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('cv2', 'OpenCV'),
        ('mediapipe', 'MediaPipe'),
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('sklearn', 'Scikit-learn')
    ]
    
    print("🧪 패키지 설치 확인 테스트 시작...\n")
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name:15} - 설치 완료")
        except ImportError as e:
            print(f"❌ {name:15} - 설치 실패: {e}")
            return False
    
    return True

def test_versions():
    """주요 패키지 버전 확인"""
    
    print("\n📋 주요 패키지 버전 정보:")
    
    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA 사용 가능: {torch.cuda.is_available()}")
        print(f"   MPS 사용 가능: {torch.backends.mps.is_available()}")
    except:
        pass
    
    try:
        import cv2
        print(f"   OpenCV: {cv2.__version__}")
    except:
        pass
    
    try:
        import mediapipe as mp
        print(f"   MediaPipe: {mp.__version__}")
    except:
        pass

def test_basic_functionality():
    """기본 기능 테스트"""
    
    print("\n🔧 기본 기능 테스트:")
    
    try:
        # NumPy 배열 생성 테스트
        import numpy as np
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        print("✅ NumPy 배열 생성 - 정상")
        
        # OpenCV 이미지 처리 테스트
        import cv2
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        print("✅ OpenCV 이미지 변환 - 정상")
        
        # MediaPipe 초기화 테스트
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True)
        print("✅ MediaPipe Pose 초기화 - 정상")
        
        # PyTorch 텐서 생성 테스트
        import torch
        tensor = torch.zeros(1, 3, 224, 224)
        print("✅ PyTorch 텐서 생성 - 정상")
        
        return True
        
    except Exception as e:
        print(f"❌ 기본 기능 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 실행"""
    
    print("=" * 50)
    print("MyCloset AI MVP - 환경 설정 확인")
    print("=" * 50)
    
    print(f"Python 버전: {sys.version}")
    print(f"Python 경로: {sys.executable}\n")
    
    # 1. 패키지 import 테스트
    if not test_imports():
        print("\n❌ 일부 패키지가 설치되지 않았습니다.")
        print("다음 명령어로 재설치해보세요:")
        print("pip install -r requirements.txt")
        return False
    
    # 2. 버전 정보 확인
    test_versions()
    
    # 3. 기본 기능 테스트
    if not test_basic_functionality():
        print("\n❌ 기본 기능 테스트에 실패했습니다.")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 모든 테스트 통과! 개발 환경 준비 완료!")
    print("=" * 50)
    print("\n다음 단계:")
    print("1. python main.py 실행")
    print("2. http://localhost:8000/docs 접속")
    print("3. API 테스트 시작")
    
    return True

if __name__ == "__main__":
    main()