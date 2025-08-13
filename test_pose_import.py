#!/usr/bin/env python3
"""
PoseEstimationStep 임포트 테스트 (프로젝트 루트에서)
"""

import sys
import os

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print(f"현재 디렉토리: {current_dir}")
print(f"Python 경로에 추가됨: {current_dir in sys.path}")

try:
    from backend.app.ai_pipeline.steps.02_pose_estimation.core.PoseEstimationStep import PoseEstimationStep
    print("✅ PoseEstimationStep 임포트 성공!")
    print(f"클래스 이름: {PoseEstimationStep.__name__}")
    
    # 간단한 인스턴스 생성 테스트
    step = PoseEstimationStep()
    print(f"✅ 인스턴스 생성 성공: {type(step).__name__}")
    
except Exception as e:
    print(f"❌ 임포트 실패: {e}")
    print(f"에러 타입: {type(e).__name__}")
    import traceback
    traceback.print_exc()
