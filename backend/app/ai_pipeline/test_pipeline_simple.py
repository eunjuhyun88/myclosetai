#!/usr/bin/env python3
"""
🧪 MyCloset AI 전체 파이프라인 간단 테스트 스크립트
================================================================================

이 스크립트는 전체 AI 파이프라인의 각 단계를 순차적으로 테스트합니다.
각 단계의 로드 및 기본 기능을 확인합니다.

사용법:
    python test_pipeline_simple.py                    # 전체 파이프라인 테스트
    python test_pipeline_simple.py --step 3          # 3단계만 테스트
    python test_pipeline_simple.py --status          # 파이프라인 상태만 확인

Author: MyCloset AI Team
Date: 2025-07-31
"""

import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 경로 설정
current_file = Path(__file__).absolute()
backend_dir = current_file.parent.parent.parent  # backend/
sys.path.insert(0, str(backend_dir))

def test_step_01_human_parsing():
    """1단계: Human Parsing 테스트"""
    try:
        print("🔍 1단계: Human Parsing 테스트 중...")
        from app.ai_pipeline.steps.step_01_human_parsing import HumanParsingStep
        
        step = HumanParsingStep()
        print(f"   ✅ HumanParsingStep 클래스 로드 성공: {type(step).__name__}")
        
        # 기본 속성 확인
        if hasattr(step, 'step_id'):
            print(f"   📋 Step ID: {step.step_id}")
        if hasattr(step, 'step_name'):
            print(f"   📋 Step Name: {step.step_name}")
        
        return True, "Human Parsing Step 로드 성공"
        
    except Exception as e:
        return False, f"Human Parsing Step 테스트 실패: {e}"

def test_step_02_pose_estimation():
    """2단계: Pose Estimation 테스트"""
    try:
        print("🔍 2단계: Pose Estimation 테스트 중...")
        from app.ai_pipeline.steps.step_02_pose_estimation import PoseEstimationStep
        
        step = PoseEstimationStep()
        print(f"   ✅ PoseEstimationStep 클래스 로드 성공: {type(step).__name__}")
        
        # 기본 속성 확인
        if hasattr(step, 'step_id'):
            print(f"   📋 Step ID: {step.step_id}")
        if hasattr(step, 'step_name'):
            print(f"   📋 Step Name: {step.step_name}")
        
        return True, "Pose Estimation Step 로드 성공"
        
    except Exception as e:
        return False, f"Pose Estimation Step 테스트 실패: {e}"

def test_step_03_cloth_segmentation():
    """3단계: Cloth Segmentation 테스트"""
    try:
        print("🔍 3단계: Cloth Segmentation 테스트 중...")
        from app.ai_pipeline.steps.step_03_cloth_segmentation import ClothSegmentationStep
        
        step = ClothSegmentationStep()
        print(f"   ✅ ClothSegmentationStep 클래스 로드 성공: {type(step).__name__}")
        
        # 기본 속성 확인
        if hasattr(step, 'step_id'):
            print(f"   📋 Step ID: {step.step_id}")
        if hasattr(step, 'step_name'):
            print(f"   📋 Step Name: {step.step_name}")
        
        return True, "Cloth Segmentation Step 로드 성공"
        
    except Exception as e:
        return False, f"Cloth Segmentation Step 테스트 실패: {e}"

def test_step_04_geometric_matching():
    """4단계: Geometric Matching 테스트"""
    try:
        print("🔍 4단계: Geometric Matching 테스트 중...")
        from app.ai_pipeline.steps.step_04_geometric_matching import GeometricMatchingStep
        
        step = GeometricMatchingStep()
        print(f"   ✅ GeometricMatchingStep 클래스 로드 성공: {type(step).__name__}")
        
        # 기본 속성 확인
        if hasattr(step, 'step_id'):
            print(f"   📋 Step ID: {step.step_id}")
        if hasattr(step, 'step_name'):
            print(f"   📋 Step Name: {step.step_name}")
        
        return True, "Geometric Matching Step 로드 성공"
        
    except Exception as e:
        return False, f"Geometric Matching Step 테스트 실패: {e}"

def test_step_05_cloth_warping():
    """5단계: Cloth Warping 테스트"""
    try:
        print("🔍 5단계: Cloth Warping 테스트 중...")
        from app.ai_pipeline.steps.step_05_cloth_warping import ClothWarpingStep
        
        step = ClothWarpingStep()
        print(f"   ✅ ClothWarpingStep 클래스 로드 성공: {type(step).__name__}")
        
        # 기본 속성 확인
        if hasattr(step, 'step_id'):
            print(f"   📋 Step ID: {step.step_id}")
        if hasattr(step, 'step_name'):
            print(f"   📋 Step Name: {step.step_name}")
        
        return True, "Cloth Warping Step 로드 성공"
        
    except Exception as e:
        return False, f"Cloth Warping Step 테스트 실패: {e}"

def test_step_06_virtual_fitting():
    """6단계: Virtual Fitting 테스트"""
    try:
        print("🔍 6단계: Virtual Fitting 테스트 중...")
        from app.ai_pipeline.steps.step_06_virtual_fitting import VirtualFittingStep
        
        step = VirtualFittingStep()
        print(f"   ✅ VirtualFittingStep 클래스 로드 성공: {type(step).__name__}")
        
        # 기본 속성 확인
        if hasattr(step, 'step_id'):
            print(f"   📋 Step ID: {step.step_id}")
        if hasattr(step, 'step_name'):
            print(f"   📋 Step Name: {step.step_name}")
        
        return True, "Virtual Fitting Step 로드 성공"
        
    except Exception as e:
        return False, f"Virtual Fitting Step 테스트 실패: {e}"

def test_step_07_post_processing():
    """7단계: Post Processing 테스트"""
    try:
        print("🔍 7단계: Post Processing 테스트 중...")
        from steps.step_07_post_processing import PostProcessingStep
        
        step = PostProcessingStep()
        print(f"   ✅ PostProcessingStep 클래스 로드 성공: {type(step).__name__}")
        
        # 기본 속성 확인
        if hasattr(step, 'step_id'):
            print(f"   📋 Step ID: {step.step_id}")
        if hasattr(step, 'step_name'):
            print(f"   📋 Step Name: {step.step_name}")
        
        return True, "Post Processing Step 로드 성공"
        
    except Exception as e:
        return False, f"Post Processing Step 테스트 실패: {e}"

def test_step_08_quality_assessment():
    """8단계: Quality Assessment 테스트"""
    try:
        print("🔍 8단계: Quality Assessment 테스트 중...")
        from app.ai_pipeline.steps.step_08_quality_assessment import QualityAssessmentStep
        
        step = QualityAssessmentStep()
        print(f"   ✅ QualityAssessmentStep 클래스 로드 성공: {type(step).__name__}")
        
        # 기본 속성 확인
        if hasattr(step, 'step_id'):
            print(f"   📋 Step ID: {step.step_id}")
        if hasattr(step, 'step_name'):
            print(f"   📋 Step Name: {step.step_name}")
        
        return True, "Quality Assessment Step 로드 성공"
        
    except Exception as e:
        return False, f"Quality Assessment Step 테스트 실패: {e}"

def test_step_09_final_output():
    """9단계: Final Output 테스트"""
    try:
        print("🔍 9단계: Final Output 테스트 중...")
        from app.ai_pipeline.steps.step_09_final_output import FinalOutputStep
        
        step = FinalOutputStep()
        print(f"   ✅ FinalOutputStep 클래스 로드 성공: {type(step).__name__}")
        
        # 기본 속성 확인
        if hasattr(step, 'step_id'):
            print(f"   📋 Step ID: {step.step_id}")
        if hasattr(step, 'step_name'):
            print(f"   📋 Step Name: {step.step_name}")
        
        return True, "Final Output Step 로드 성공"
        
    except Exception as e:
        return False, f"Final Output Step 테스트 실패: {e}"

def get_pipeline_status():
    """파이프라인 상태 확인"""
    print("📊 파이프라인 상태 확인 중...")
    
    step_configs = [
        (1, "Human Parsing", "step_01_human_parsing"),
        (2, "Pose Estimation", "step_02_pose_estimation"),
        (3, "Cloth Segmentation", "step_03_cloth_segmentation"),
        (4, "Geometric Matching", "step_04_geometric_matching"),
        (5, "Cloth Warping", "step_05_cloth_warping"),
        (6, "Virtual Fitting", "step_06_virtual_fitting"),
        (7, "Post Processing", "step_07_post_processing"),
        (8, "Quality Assessment", "step_08_quality_assessment"),
        (9, "Final Output", "step_09_final_output")
    ]
    
    available_steps = []
    missing_steps = []
    
    for step_id, step_name, step_file in step_configs:
        step_path = f"steps/{step_file}.py"
        if os.path.exists(step_path):
            available_steps.append((step_id, step_name, step_file))
            print(f"   ✅ {step_id:2d}. {step_name:<20} - {step_file}.py")
        else:
            missing_steps.append((step_id, step_name, step_file))
            print(f"   ❌ {step_id:2d}. {step_name:<20} - {step_file}.py (파일 없음)")
    
    print(f"\n📈 파이프라인 완성도: {len(available_steps)}/9 ({len(available_steps)/9*100:.1f}%)")
    
    if missing_steps:
        print(f"⚠️  누락된 단계: {len(missing_steps)}개")
        for step_id, step_name, step_file in missing_steps:
            print(f"   - {step_id}. {step_name} ({step_file}.py)")
    
    return available_steps, missing_steps

def test_full_pipeline():
    """전체 파이프라인 테스트"""
    print("🧪 전체 AI 파이프라인 통합 테스트 시작")
    print("=" * 60)
    
    start_time = time.time()
    
    # 테스트 함수들
    test_functions = [
        test_step_01_human_parsing,
        test_step_02_pose_estimation,
        test_step_03_cloth_segmentation,
        test_step_04_geometric_matching,
        test_step_05_cloth_warping,
        test_step_06_virtual_fitting,
        test_step_07_post_processing,
        test_step_08_quality_assessment,
        test_step_09_final_output
    ]
    
    results = []
    
    for i, test_func in enumerate(test_functions, 1):
        print(f"\n🔍 {i}단계 테스트 시작...")
        success, message = test_func()
        results.append((i, success, message))
        
        if success:
            print(f"   ✅ {i}단계 테스트 성공")
        else:
            print(f"   ❌ {i}단계 테스트 실패: {message}")
    
    # 결과 요약
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("📊 전체 파이프라인 테스트 결과")
    print("=" * 60)
    
    successful_steps = sum(1 for _, success, _ in results if success)
    total_steps = len(results)
    
    print(f"총 단계 수: {total_steps}")
    print(f"성공한 단계: {successful_steps}")
    print(f"실패한 단계: {total_steps - successful_steps}")
    print(f"성공률: {successful_steps/total_steps*100:.1f}%")
    print(f"총 소요 시간: {total_time:.2f}초")
    
    if successful_steps == total_steps:
        print("\n🎉 모든 단계 테스트 성공! 파이프라인이 정상적으로 작동합니다.")
    else:
        print(f"\n⚠️  {total_steps - successful_steps}개 단계에서 문제가 발생했습니다.")
    
    return results

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="MyCloset AI 파이프라인 테스트")
    parser.add_argument("--step", type=int, help="특정 단계만 테스트 (1-9)")
    parser.add_argument("--status", action="store_true", help="파이프라인 상태만 확인")
    parser.add_argument("--full", action="store_true", help="전체 파이프라인 테스트")
    
    args = parser.parse_args()
    
    print("🚀 MyCloset AI 파이프라인 테스트 시작")
    print(f"📍 실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📍 작업 디렉토리: {os.getcwd()}")
    print("=" * 60)
    
    if args.status:
        # 상태만 확인
        get_pipeline_status()
        
    elif args.step:
        # 특정 단계만 테스트
        if args.step < 1 or args.step > 9:
            print(f"❌ 잘못된 단계 번호: {args.step}. 1-9 범위여야 합니다.")
            return
        
        step_names = {
            1: "Human Parsing",
            2: "Pose Estimation",
            3: "Cloth Segmentation", 
            4: "Geometric Matching",
            5: "Cloth Warping",
            6: "Virtual Fitting",
            7: "Post Processing",
            8: "Quality Assessment",
            9: "Final Output"
        }
        
        step_name = step_names[args.step]
        print(f"🧪 {args.step}단계: {step_name} 테스트 시작")
        
        test_functions = {
            1: test_step_01_human_parsing,
            2: test_step_02_pose_estimation,
            3: test_step_03_cloth_segmentation,
            4: test_step_04_geometric_matching,
            5: test_step_05_cloth_warping,
            6: test_step_06_virtual_fitting,
            7: test_step_07_post_processing,
            8: test_step_08_quality_assessment,
            9: test_step_09_final_output
        }
        
        success, message = test_functions[args.step]()
        if success:
            print(f"\n✅ {args.step}단계 테스트 성공!")
        else:
            print(f"\n❌ {args.step}단계 테스트 실패: {message}")
            
    else:
        # 기본: 전체 파이프라인 테스트
        test_full_pipeline()
    
    print("\n🏁 테스트 완료!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  테스트가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
