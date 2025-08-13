#!/usr/bin/env python3
"""
🧪 2단계 Pose Estimation 간단 테스트
"""

import sys
import os
from pathlib import Path

# 현재 디렉토리를 sys.path에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    # 숫자로 시작하는 디렉토리명을 import할 때는 __import__ 사용
    step_module = __import__('steps.02_pose_estimation.step_modularized', fromlist=['PoseEstimationStep'])
    PoseEstimationStep = step_module.PoseEstimationStep
    
    print("✅ 2단계 import 성공")
    
    # 스텝 생성
    step = PoseEstimationStep()
    print("✅ 2단계 초기화 성공")
    
    # 모델 상태 확인
    print(f"📊 모델 로딩 상태: {step.models_loading_status}")
    print(f"📊 로드된 모델들: {list(step.models.keys())}")
    
    # Mock 이미지로 테스트
    import numpy as np
    mock_image = np.random.rand(512, 512, 3).astype(np.float32)
    
    print("\n🧪 Mock 이미지로 테스트 시작...")
    result = step.process(image=mock_image)
    
    print(f"✅ 테스트 결과: {result.get('success', False)}")
    if result.get('success'):
        print(f"📊 결과 키: {list(result.keys())}")
    else:
        print(f"❌ 에러: {result.get('error', '알 수 없는 에러')}")
    
except Exception as e:
    print(f"❌ 테스트 실패: {e}")
    import traceback
    traceback.print_exc()
