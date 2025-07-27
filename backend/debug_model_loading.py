#!/usr/bin/env python3
"""
AI 모델 로딩 문제 해결 테스트
backend/debug_model_loading.py
"""

import sys
import os
from pathlib import Path
sys.path.append('.')

def test_model_loading_fixes():
    """모델 로딩 수정사항 테스트"""
    
    print("🔧 AI 모델 로딩 수정사항 테스트 시작...")
    
    # 1. MPS 환경 설정 테스트
    try:
        from app.ai_pipeline.utils.device_manager import DeviceManager
        DeviceManager.setup_mps_compatibility()
        print("✅ MPS 호환성 설정 완료")
    except Exception as e:
        print(f"❌ MPS 설정 실패: {e}")
    
    # 2. ModelLoader._find_checkpoint_file 테스트
    try:
        from app.ai_pipeline.utils.model_loader import get_global_model_loader
        
        model_loader = get_global_model_loader()
        if hasattr(model_loader, '_find_checkpoint_file'):
            print("✅ _find_checkpoint_file 메서드 존재")
            
            # 테스트 검색
            test_models = [
                "cloth_segmentation_u2net",
                "geometric_matching_model", 
                "pose_estimation_openpose"
            ]
            
            for model_name in test_models:
                result = model_loader._find_checkpoint_file(model_name)
                print(f"   {model_name}: {'✅' if result else '❌'}")
                if result:
                    print(f"     경로: {result}")
        else:
            print("❌ _find_checkpoint_file 메서드 없음")
            
    except Exception as e:
        print(f"❌ ModelLoader 테스트 실패: {e}")
    
    # 3. 체크포인트 파일 탐지 테스트
    try:
        ai_models_path = Path("ai_models")
        if ai_models_path.exists():
            checkpoint_files = []
            checkpoint_files.extend(list(ai_models_path.rglob("*.pth")))
            checkpoint_files.extend(list(ai_models_path.rglob("*.safetensors")))
            checkpoint_files.extend(list(ai_models_path.rglob("*.bin")))
            checkpoint_files.extend(list(ai_models_path.rglob("*.pt")))
            
            print(f"✅ 체크포인트 파일 탐지: {len(checkpoint_files)}개")
            
            # 큰 파일들 (1GB 이상) 표시
            large_files = [f for f in checkpoint_files if f.stat().st_size > 1024*1024*1024]
            if large_files:
                print(f"🔥 대형 모델 ({len(large_files)}개):")
                for file in large_files[:5]:  # 상위 5개만
                    size_gb = file.stat().st_size / (1024*1024*1024)
                    print(f"   {file.name}: {size_gb:.1f}GB")
        else:
            print("❌ ai_models 디렉토리 없음")
            
    except Exception as e:
        print(f"❌ 체크포인트 탐지 실패: {e}")
    
    # 4. Step별 AI 모델 로딩 테스트
    test_steps = [
        ("PoseEstimationStep", 2),
        ("GeometricMatchingStep", 4),
        ("VirtualFittingStep", 6)
    ]
    
    for step_name, step_id in test_steps:
        try:
            if step_name == "PoseEstimationStep":
                from app.ai_pipeline.steps.step_02_pose_estimation import PoseEstimationStep
                step = PoseEstimationStep(device='mps')
            elif step_name == "GeometricMatchingStep":
                from app.ai_pipeline.steps.step_04_geometric_matching import GeometricMatchingStep  
                step = GeometricMatchingStep(device='mps')
            elif step_name == "VirtualFittingStep":
                from app.ai_pipeline.steps.step_06_virtual_fitting import VirtualFittingStep
                step = VirtualFittingStep(device='mps')
            
            # 초기화 테스트
            init_result = step.initialize()
            print(f"✅ {step_name} 초기화: {'성공' if init_result else '실패'}")
            
        except Exception as e:
            print(f"❌ {step_name} 테스트 실패: {e}")
    
    print("\n🎉 테스트 완료!")

if __name__ == "__main__":
    test_model_loading_fixes()