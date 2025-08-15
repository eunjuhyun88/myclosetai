#!/usr/bin/env python3
"""
🔥 MyCloset AI - 전체 파이프라인 간단한 통합 테스트
==================================================

각 Step의 기본 기능을 테스트하고 전체 워크플로우를 검증합니다.
"""

import os
import sys
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_step_01_human_parsing():
    """Step 01: Human Parsing 테스트"""
    print("\n🔍 Step 01: Human Parsing 테스트")
    print("=" * 50)
    
    try:
        # U2Net 모델 테스트
        sys.path.append('step_01_human_parsing_models/models')
        from human_parsing_u2net import U2Net
        
        model = U2Net()
        print("✅ U2Net 모델 초기화 성공")
        
        # 간단한 테스트 이미지 생성
        import numpy as np
        import torch
        test_image = np.zeros((512, 512, 3), dtype=np.uint8)
        input_tensor = torch.from_numpy(test_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        with torch.no_grad():
            output = model(input_tensor)
            print(f"✅ U2Net 추론 성공: {output[0].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Step 01 테스트 실패: {e}")
        return False

def test_step_02_pose_estimation():
    """Step 02: Pose Estimation 테스트"""
    print("\n🔍 Step 02: Pose Estimation 테스트")
    print("=" * 50)
    
    try:
        # HRNet Pose 모델 테스트
        sys.path.append('step_02_pose_estimation_models/models')
        from pose_estimation_models import HRNetPoseModel
        
        model = HRNetPoseModel()
        print("✅ HRNet Pose 모델 초기화 성공")
        
        # 간단한 테스트 이미지 생성
        import numpy as np
        import torch
        test_image = np.zeros((256, 256, 3), dtype=np.uint8)
        input_tensor = torch.from_numpy(test_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        with torch.no_grad():
            output = model(input_tensor)
            print(f"✅ HRNet Pose 추론 성공: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Step 02 테스트 실패: {e}")
        return False

def test_step_03_cloth_segmentation():
    """Step 03: Cloth Segmentation 테스트"""
    print("\n🔍 Step 03: Cloth Segmentation 테스트")
    print("=" * 50)
    
    try:
        # SAM 모델 테스트
        sys.path.append('step_03_cloth_segmentation_models/models')
        from cloth_segmentation_sam import SAMModel
        
        model = SAMModel()
        print("✅ SAM 모델 초기화 성공")
        
        # 간단한 테스트 이미지 생성 (더 작은 크기로)
        import numpy as np
        import torch
        test_image = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # numpy를 tensor로 변환
        input_tensor = torch.from_numpy(test_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # SAM 추론 (segment_clothing 메서드 사용) - 타임아웃 설정
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("SAM 추론이 너무 오래 걸립니다")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30초 타임아웃
        
        try:
            with torch.no_grad():
                masks = model.segment_clothing(input_tensor)
                print(f"✅ SAM 옷감 세그멘테이션 성공: {masks.shape}")
        finally:
            signal.alarm(0)  # 타임아웃 해제
        
        return True
        
    except Exception as e:
        print(f"❌ Step 03 테스트 실패: {e}")
        return False

def test_step_04_geometric_matching():
    """Step 04: Geometric Matching 테스트"""
    print("\n🔍 Step 04: Geometric Matching 테스트")
    print("=" * 50)
    
    try:
        # GMM 모델 테스트
        sys.path.append('step_04_geometric_matching_models/models')
        from geometric_models import GMMModel
        
        model = GMMModel()
        print("✅ GMM 모델 초기화 성공")
        
        # 간단한 테스트 데이터 생성
        import numpy as np
        import torch
        person_image = np.zeros((512, 384, 3), dtype=np.uint8)
        cloth_image = np.zeros((512, 384, 3), dtype=np.uint8)
        
        # numpy를 tensor로 변환
        person_tensor = torch.from_numpy(person_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        cloth_tensor = torch.from_numpy(cloth_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        with torch.no_grad():
            result = model.match_geometrically(person_tensor, cloth_tensor)
            print(f"✅ GMM 기하학적 매칭 성공: {type(result)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Step 04 테스트 실패: {e}")
        return False

def test_step_05_cloth_warping():
    """Step 05: Cloth Warping 테스트"""
    print("\n🔍 Step 05: Cloth Warping 테스트")
    print("=" * 50)
    
    try:
        # RAFT Warping 모델 테스트 (상대 import 문제 없는 모델)
        sys.path.append('step_05_cloth_warping_models/models')
        from raft_warping_model import RAFTWarpingModel
        
        model = RAFTWarpingModel()
        print("✅ RAFT Warping 모델 초기화 성공")
        
        # 간단한 테스트 데이터 생성
        import numpy as np
        import torch
        cloth_image = np.zeros((512, 384, 3), dtype=np.uint8)
        person_image = np.zeros((512, 384, 3), dtype=np.uint8)
        
        # numpy를 tensor로 변환
        cloth_tensor = torch.from_numpy(cloth_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        person_tensor = torch.from_numpy(person_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        with torch.no_grad():
            result = model.warp_clothing(person_tensor, cloth_tensor)
            print(f"✅ RAFT Warping 성공: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Step 05 테스트 실패: {e}")
        return False

def test_step_06_virtual_fitting():
    """Step 06: Virtual Fitting 테스트"""
    print("\n🔍 Step 06: Virtual Fitting 테스트")
    print("=" * 50)
    
    try:
        # OOTD 모델 테스트
        sys.path.append('step_06_virtual_fitting_models/models')
        from ootd_model import OOTDModel
        
        model = OOTDModel()
        print("✅ OOTD 모델 초기화 성공")
        
        # 간단한 테스트 데이터 생성
        import numpy as np
        import torch
        person_image = np.zeros((512, 384, 3), dtype=np.uint8)
        cloth_image = np.zeros((512, 384, 3), dtype=np.uint8)
        
        # numpy를 tensor로 변환
        person_tensor = torch.from_numpy(person_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        cloth_tensor = torch.from_numpy(cloth_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        with torch.no_grad():
            result = model.virtual_try_on(person_tensor, cloth_tensor)
            print(f"✅ OOTD 가상 피팅 성공: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Step 06 테스트 실패: {e}")
        return False

def test_step_07_post_processing():
    """Step 07: Post Processing 테스트"""
    print("\n🔍 Step 07: Post Processing 테스트")
    print("=" * 50)
    
    try:
        # RealESRGAN 모델 테스트
        sys.path.append('step_07_post_processing_models/models')
        from realesrgan_model import RRDBNet
        
        model = RRDBNet()
        print("✅ RealESRGAN 모델 초기화 성공")
        
        # 간단한 테스트 데이터 생성
        import numpy as np
        import torch
        test_image = np.zeros((512, 512, 3), dtype=np.uint8)
        input_tensor = torch.from_numpy(test_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        with torch.no_grad():
            enhanced = model(input_tensor)
            print(f"✅ RealESRGAN 향상 성공: {enhanced.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Step 07 테스트 실패: {e}")
        return False

def test_step_08_quality_assessment():
    """Step 08: Quality Assessment 테스트"""
    print("\n🔍 Step 08: Quality Assessment 테스트")
    print("=" * 50)
    
    try:
        # Quality Assessment 모델 테스트
        sys.path.append('step_08_quality_assessment_models/models')
        from quality_assessor import QualityAssessorWrapper
        
        model = QualityAssessorWrapper()
        print("✅ Quality Assessment 모델 초기화 성공")
        
        # 간단한 테스트 데이터 생성
        import numpy as np
        import torch
        test_image = np.zeros((512, 512, 3), dtype=np.uint8)
        input_tensor = torch.from_numpy(test_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        with torch.no_grad():
            quality_score = model.assess_quality(input_tensor)
            print(f"✅ Quality Assessment 성공: {type(quality_score)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Step 08 테스트 실패: {e}")
        return False

def test_step_09_final_output():
    """Step 09: Final Output 테스트"""
    print("\n🔍 Step 09: Final Output 테스트")
    print("=" * 50)
    
    try:
        # Final Output Generator 모델 테스트
        sys.path.append('step_09_final_output_models/models')
        from final_output_models import FinalOutputModel
        
        model = FinalOutputModel()
        print("✅ Final Output Generator 모델 초기화 성공")
        
        # 간단한 테스트 데이터 생성
        import numpy as np
        import torch
        test_image = np.zeros((512, 512, 3), dtype=np.uint8)
        input_tensor = torch.from_numpy(test_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # 더미 텍스트 입력 생성
        dummy_text = torch.zeros(1, 768)  # 간단한 텍스트 임베딩
        
        with torch.no_grad():
            output = model.generate_output(input_tensor, dummy_text)
            print(f"✅ Final Output Generator 성공: {type(output)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Step 09 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🔥 MyCloset AI - 전체 파이프라인 간단한 통합 테스트")
    print("=" * 70)
    
    # PyTorch import
    try:
        import torch
        print(f"✅ PyTorch 버전: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch import 실패")
        return
    
    # 각 Step 테스트 실행
    test_results = {}
    
    test_results['step_01'] = test_step_01_human_parsing()
    test_results['step_02'] = test_step_02_pose_estimation()
    test_results['step_03'] = test_step_03_cloth_segmentation()
    test_results['step_04'] = test_step_04_geometric_matching()
    test_results['step_05'] = test_step_05_cloth_warping()
    test_results['step_06'] = test_step_06_virtual_fitting()
    test_results['step_07'] = test_step_07_post_processing()
    test_results['step_08'] = test_step_08_quality_assessment()
    test_results['step_09'] = test_step_09_final_output()
    
    # 결과 요약
    print("\n" + "=" * 70)
    print("🎯 전체 파이프라인 통합 테스트 결과")
    print("=" * 70)
    
    success_count = 0
    for step, result in test_results.items():
        status = "✅ 성공" if result else "❌ 실패"
        print(f"{step:15}: {status}")
        if result:
            success_count += 1
    
    print(f"\n📊 최종 결과: {success_count}/9 성공")
    
    if success_count == 9:
        print("🎉 모든 Steps가 정상적으로 작동합니다!")
    elif success_count >= 6:
        print("👍 대부분의 Steps가 정상적으로 작동합니다.")
    else:
        print("⚠️ 일부 Steps에 문제가 있습니다.")
    
    print(f"\n🚀 MyCloset AI 파이프라인 준비 상태: {success_count/9*100:.1f}%")

if __name__ == "__main__":
    main()
