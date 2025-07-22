#!/usr/bin/env python3
"""
🔍 모델 탐지 테스트 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent / "backend"
sys.path.insert(0, str(project_root))

def test_model_detection():
    """모델 탐지 테스트"""
    print("🔍 모델 탐지 테스트 시작...")
    
    ai_models_root = Path(__file__).parent / "backend" / "ai_models"
    
    for step in range(1, 9):
        step_dir = ai_models_root / f"step_{step:02d}_*"
        
        # 실제 디렉토리 찾기
        import glob
        step_dirs = glob.glob(str(step_dir))
        
        if step_dirs:
            step_path = Path(step_dirs[0])
            model_files = list(step_path.glob("*.pth")) + list(step_path.glob("*.pt")) + list(step_path.glob("*.bin"))
            print(f"   Step {step:02d}: {len(model_files)}개 모델 파일 발견")
            for f in model_files[:3]:  # 상위 3개만 표시
                print(f"      - {f.name}")
        else:
            print(f"   Step {step:02d}: 디렉토리 없음")

if __name__ == "__main__":
    test_model_detection()
