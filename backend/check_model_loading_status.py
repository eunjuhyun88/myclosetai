#!/usr/bin/env python3
"""
각 스텝별 모델 로딩 상태 확인 스크립트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

import torch
from pathlib import Path
import logging

# 로깅 설정 (간단하게)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_model_files():
    """모델 파일 존재 여부 확인"""
    print("🔍 모델 파일 존재 여부 확인:")
    print("=" * 50)
    
    model_paths = [
        "ai_models/Self-Correction-Human-Parsing/exp-schp-201908261155-atr.pth",
        "ai_models/U2Net/u2net.pth",
        "ai_models/DeepLabV3+/deeplabv3plus.pth",
        "ai_models/SAM/sam_vit_h_4b8939.pth",
        "ai_models/OpenPose/pose_iter_584000.caffemodel",
        "ai_models/YOLOv8/yolov8n-pose.pt",
        "ai_models/GMM/gmm_final.pth",
        "ai_models/RealVisXL/realvisxl_v4.0.safetensors"
    ]
    
    for path in model_paths:
        full_path = Path(path)
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print(f"✅ {path} ({size_mb:.1f}MB)")
        else:
            print(f"❌ {path} (없음)")

def check_step_models():
    """각 스텝별 모델 로딩 상태 확인"""
    print("\n🧠 각 스텝별 모델 로딩 상태:")
    print("=" * 50)
    
    try:
        from app.ai_pipeline.steps.step_01_human_parsing import HumanParsingStep
        from app.ai_pipeline.steps.step_02_pose_estimation import PoseEstimationStep
        from app.ai_pipeline.steps.step_03_cloth_segmentation import ClothSegmentationStep
        from app.ai_pipeline.steps.step_04_geometric_matching import GeometricMatchingStep
        from app.ai_pipeline.steps.step_05_cloth_warping import ClothWarpingStep
        from app.ai_pipeline.steps.step_06_virtual_fitting import VirtualFittingStep
        from app.ai_pipeline.steps.step_07_post_processing import PostProcessingStep
        from app.ai_pipeline.steps.step_08_quality_assessment import QualityAssessmentStep
        
        steps = [
            ("Step 1: Human Parsing", HumanParsingStep),
            ("Step 2: Pose Estimation", PoseEstimationStep),
            ("Step 3: Cloth Segmentation", ClothSegmentationStep),
            ("Step 4: Geometric Matching", GeometricMatchingStep),
            ("Step 5: Cloth Warping", ClothWarpingStep),
            ("Step 6: Virtual Fitting", VirtualFittingStep),
            ("Step 7: Post Processing", PostProcessingStep),
            ("Step 8: Quality Assessment", QualityAssessmentStep)
        ]
        
        for step_name, step_class in steps:
            try:
                print(f"\n🔍 {step_name} 확인 중...")
                step = step_class()
                
                # 모델 로딩 시도
                if hasattr(step, '_load_ai_models_via_central_hub'):
                    result = step._load_ai_models_via_central_hub()
                    if result:
                        print(f"✅ {step_name}: 모델 로딩 성공")
                    else:
                        print(f"❌ {step_name}: 모델 로딩 실패")
                else:
                    print(f"⚠️ {step_name}: _load_ai_models_via_central_hub 메서드 없음")
                    
            except Exception as e:
                print(f"❌ {step_name}: 오류 발생 - {str(e)[:100]}...")
                
    except Exception as e:
        print(f"❌ 스텝 임포트 실패: {e}")

def check_checkpoint_loading():
    """체크포인트 로딩 상태 확인"""
    print("\n📦 체크포인트 로딩 상태:")
    print("=" * 50)
    
    try:
        checkpoint_path = "ai_models/Self-Correction-Human-Parsing/exp-schp-201908261155-atr.pth"
        if Path(checkpoint_path).exists():
            print(f"🔄 {checkpoint_path} 로딩 중...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if isinstance(checkpoint, dict):
                print(f"✅ 체크포인트 로딩 성공: {len(checkpoint)}개 키")
                
                # 키 구조 확인
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    print(f"📊 state_dict 키 수: {len(state_dict)}")
                    
                    # 키 샘플 출력
                    sample_keys = list(state_dict.keys())[:5]
                    print(f"🔍 키 샘플: {sample_keys}")
                    
                    # 모델 구조 확인
                    total_params = sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))
                    print(f"📊 총 파라미터 수: {total_params:,}")
                else:
                    print(f"📊 직접 키들: {list(checkpoint.keys())[:5]}")
            else:
                print(f"⚠️ 체크포인트가 딕셔너리가 아님: {type(checkpoint)}")
        else:
            print(f"❌ 체크포인트 파일 없음: {checkpoint_path}")
            
    except Exception as e:
        print(f"❌ 체크포인트 로딩 실패: {e}")

if __name__ == "__main__":
    print("🍎 MyCloset AI 모델 로딩 상태 확인")
    print("=" * 60)
    
    check_model_files()
    check_checkpoint_loading()
    check_step_models()
    
    print("\n✅ 모델 로딩 상태 확인 완료!") 