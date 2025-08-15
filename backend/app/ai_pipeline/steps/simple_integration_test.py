#!/usr/bin/env python3
"""
🔥 MyCloset AI - 간단한 통합 테스트
====================================

각 Step의 기본 기능을 개별적으로 테스트합니다.
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
        # Step 01 디렉토리로 이동
        step01_dir = "step_01_human_parsing_models"
        if os.path.exists(step01_dir):
            print(f"✅ Step 01 디렉토리 존재: {step01_dir}")
            
            # 주요 파일들 확인
            files_to_check = [
                "step_01_human_parsing.py",
                "models/",
                "checkpoints/",
                "ensemble/",
                "preprocessing/",
                "postprocessing/"
            ]
            
            for file_path in files_to_check:
                full_path = os.path.join(step01_dir, file_path)
                if os.path.exists(full_path):
                    print(f"  ✅ {file_path}")
                else:
                    print(f"  ❌ {file_path}")
        else:
            print(f"❌ Step 01 디렉토리 없음: {step01_dir}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Step 01 테스트 실패: {e}")
        return False

def test_step_02_pose_estimation():
    """Step 02: Pose Estimation 테스트"""
    print("\n🔍 Step 02: Pose Estimation 테스트")
    print("=" * 50)
    
    try:
        # Step 02 디렉토리로 이동
        step02_dir = "step_02_pose_estimation_models"
        if os.path.exists(step02_dir):
            print(f"✅ Step 02 디렉토리 존재: {step02_dir}")
            
            # 주요 파일들 확인
            files_to_check = [
                "step_02_pose_estimation.py",
                "models/",
                "checkpoints/",
                "test_pose_inference.py"
            ]
            
            for file_path in files_to_check:
                full_path = os.path.join(step02_dir, file_path)
                if os.path.exists(full_path):
                    print(f"  ✅ {file_path}")
                else:
                    print(f"  ❌ {file_path}")
        else:
            print(f"❌ Step 02 디렉토리 없음: {step02_dir}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Step 02 테스트 실패: {e}")
        return False

def test_step_03_cloth_segmentation():
    """Step 03: Cloth Segmentation 테스트"""
    print("\n🔍 Step 03: Cloth Segmentation 테스트")
    print("=" * 50)
    
    try:
        # Step 03 디렉토리로 이동
        step03_dir = "step_03_cloth_segmentation_models"
        if os.path.exists(step03_dir):
            print(f"✅ Step 03 디렉토리 존재: {step03_dir}")
            
            # 주요 파일들 확인
            files_to_check = [
                "step_03_cloth_segmentation.py",
                "models/",
                "checkpoints/",
                "ensemble/"
            ]
            
            for file_path in files_to_check:
                full_path = os.path.join(step03_dir, file_path)
                if os.path.exists(full_path):
                    print(f"  ✅ {file_path}")
                else:
                    print(f"  ❌ {file_path}")
        else:
            print(f"❌ Step 03 디렉토리 없음: {step03_dir}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Step 03 테스트 실패: {e}")
        return False

def test_step_04_geometric_matching():
    """Step 04: Geometric Matching 테스트"""
    print("\n🔍 Step 04: Geometric Matching 테스트")
    print("=" * 50)
    
    try:
        # Step 04 디렉토리로 이동
        step04_dir = "step_04_geometric_matching_models"
        if os.path.exists(step04_dir):
            print(f"✅ Step 04 디렉토리 존재: {step04_dir}")
            
            # 주요 파일들 확인
            files_to_check = [
                "step_04_geometric_matching.py",
                "models/",
                "checkpoints/"
            ]
            
            for file_path in files_to_check:
                full_path = os.path.join(step04_dir, file_path)
                if os.path.exists(full_path):
                    print(f"  ✅ {file_path}")
                else:
                    print(f"  ❌ {file_path}")
        else:
            print(f"❌ Step 04 디렉토리 없음: {step04_dir}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Step 04 테스트 실패: {e}")
        return False

def test_step_05_cloth_warping():
    """Step 05: Cloth Warping 테스트"""
    print("\n🔍 Step 05: Cloth Warping 테스트")
    print("=" * 50)
    
    try:
        # Step 05 디렉토리로 이동
        step05_dir = "step_05_cloth_warping_models"
        if os.path.exists(step05_dir):
            print(f"✅ Step 05 디렉토리 존재: {step05_dir}")
            
            # 주요 파일들 확인
            files_to_check = [
                "step_05_cloth_warping.py",
                "models/",
                "checkpoints/"
            ]
            
            for file_path in files_to_check:
                full_path = os.path.join(step05_dir, file_path)
                if os.path.exists(full_path):
                    print(f"  ✅ {file_path}")
                else:
                    print(f"  ❌ {file_path}")
        else:
            print(f"❌ Step 05 디렉토리 없음: {step05_dir}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Step 05 테스트 실패: {e}")
        return False

def test_step_06_virtual_fitting():
    """Step 06: Virtual Fitting 테스트"""
    print("\n🔍 Step 06: Virtual Fitting 테스트")
    print("=" * 50)
    
    try:
        # Step 06 디렉토리로 이동
        step06_dir = "step_06_virtual_fitting_models"
        if os.path.exists(step06_dir):
            print(f"✅ Step 06 디렉토리 존재: {step06_dir}")
            
            # 주요 파일들 확인
            files_to_check = [
                "step_06_virtual_fitting.py",
                "models/",
                "checkpoints/"
            ]
            
            for file_path in files_to_check:
                full_path = os.path.join(step06_dir, file_path)
                if os.path.exists(full_path):
                    print(f"  ✅ {file_path}")
                else:
                    print(f"  ❌ {file_path}")
        else:
            print(f"❌ Step 06 디렉토리 없음: {step06_dir}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Step 06 테스트 실패: {e}")
        return False

def test_step_07_post_processing():
    """Step 07: Post Processing 테스트"""
    print("\n🔍 Step 07: Post Processing 테스트")
    print("=" * 50)
    
    try:
        # Step 07 디렉토리로 이동
        step07_dir = "step_07_post_processing_models"
        if os.path.exists(step07_dir):
            print(f"✅ Step 07 디렉토리 존재: {step07_dir}")
            
            # 주요 파일들 확인
            files_to_check = [
                "step_07_post_processing.py",
                "models/",
                "checkpoints/"
            ]
            
            for file_path in files_to_check:
                full_path = os.path.join(step07_dir, file_path)
                if os.path.exists(full_path):
                    print(f"  ✅ {file_path}")
                else:
                    print(f"  ❌ {file_path}")
        else:
            print(f"❌ Step 07 디렉토리 없음: {step07_dir}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Step 07 테스트 실패: {e}")
        return False

def test_step_08_quality_assessment():
    """Step 08: Quality Assessment 테스트"""
    print("\n🔍 Step 08: Quality Assessment 테스트")
    print("=" * 50)
    
    try:
        # Step 08 디렉토리로 이동
        step08_dir = "step_08_quality_assessment_models"
        if os.path.exists(step08_dir):
            print(f"✅ Step 08 디렉토리 존재: {step08_dir}")
            
            # 주요 파일들 확인
            files_to_check = [
                "step_08_quality_assessment.py",
                "models/",
                "checkpoints/"
            ]
            
            for file_path in files_to_check:
                full_path = os.path.join(step08_dir, file_path)
                if os.path.exists(full_path):
                    print(f"  ✅ {file_path}")
                else:
                    print(f"  ❌ {file_path}")
        else:
            print(f"❌ Step 08 디렉토리 없음: {step08_dir}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Step 08 테스트 실패: {e}")
        return False

def test_step_09_final_output():
    """Step 09: Final Output 테스트"""
    print("\n🔍 Step 09: Final Output 테스트")
    print("=" * 50)
    
    try:
        # Step 09 디렉토리로 이동
        step09_dir = "step_09_final_output_models"
        if os.path.exists(step09_dir):
            print(f"✅ Step 09 디렉토리 존재: {step09_dir}")
            
            # 주요 파일들 확인
            files_to_check = [
                "step_09_final_output.py",
                "models/",
                "checkpoints/"
            ]
            
            for file_path in files_to_check:
                full_path = os.path.join(step09_dir, file_path)
                if os.path.exists(full_path):
                    print(f"  ✅ {file_path}")
                else:
                    print(f"  ❌ {file_path}")
        else:
            print(f"❌ Step 09 디렉토리 없음: {step09_dir}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Step 09 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🔥 MyCloset AI - 간단한 통합 테스트 시작")
    print("=" * 60)
    
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
    print("\n" + "=" * 60)
    print("🎯 통합 테스트 결과 요약")
    print("=" * 60)
    
    success_count = 0
    for step, result in test_results.items():
        status = "✅ 성공" if result else "❌ 실패"
        print(f"{step:15}: {status}")
        if result:
            success_count += 1
    
    print(f"\n📊 최종 결과: {success_count}/9 성공")
    
    if success_count == 9:
        print("🎉 모든 Steps가 정상적으로 구성되어 있습니다!")
    elif success_count >= 6:
        print("👍 대부분의 Steps가 정상적으로 구성되어 있습니다.")
    else:
        print("⚠️ 일부 Steps에 문제가 있습니다.")

if __name__ == "__main__":
    main()
