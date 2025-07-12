#!/usr/bin/env python3
"""설치 확인 및 기본 테스트"""

import sys
print(f"Python 버전: {sys.version}")

# 패키지 임포트 테스트
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    print(f"   MPS 사용 가능: {torch.backends.mps.is_available()}")
    
    import cv2
    print(f"✅ OpenCV {cv2.__version__}")
    
    import mediapipe as mp
    print(f"✅ MediaPipe 설치 완료")
    
    import numpy as np
    print(f"✅ NumPy {np.__version__}")
    
    from fastapi import FastAPI
    print(f"✅ FastAPI 설치 완료")
    
    print("\n🎉 모든 패키지가 정상적으로 설치되었습니다!")
    
except ImportError as e:
    print(f"❌ 오류: {e}")
    print("필요한 패키지를 설치해주세요.")

# 간단한 MediaPipe 테스트
def test_mediapipe():
    print("\n🧪 MediaPipe 기능 테스트...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    
    # 더미 이미지로 테스트
    import numpy as np
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    results = pose.process(dummy_image)
    
    print("✅ MediaPipe 정상 작동!")
    
if __name__ == "__main__":
    test_mediapipe()
