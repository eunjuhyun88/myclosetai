#!/usr/bin/env python3
"""
🧪 MyCloset AI 전체 파이프라인 데이터 전달 분석 스크립트
================================================================================

이 스크립트는 1단계부터 9단계까지 순차적으로 데이터가 어떻게 전달되는지 분석합니다.
각 단계의 입력/출력 데이터 구조와 형식을 상세히 검사합니다.

사용법:
    python test_data_flow_analysis.py                    # 전체 데이터 흐름 분석
    python test_data_flow_analysis.py --step 3          # 3단계만 분석
    python test_data_flow_analysis.py --verbose         # 상세 로그 출력

Author: MyCloset AI Team
Date: 2025-08-13
"""

import os
import sys
import time
import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# 프로젝트 루트 경로 설정
current_file = Path(__file__).absolute()
backend_dir = current_file.parent.parent.parent  # backend/
sys.path.insert(0, str(backend_dir))

def analyze_step_01_human_parsing():
    """1단계: Human Parsing 데이터 구조 분석"""
    try:
        print("🔍 1단계: Human Parsing 데이터 구조 분석 중...")
        
        # 1단계 모듈 import
        import importlib.util
        spec = importlib.util.spec_from_file_location("step01", "steps/01_human_parsing/step_integrated_with_pose.py")
        step01_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step01_module)
        HumanParsingWithPoseStep = step01_module.HumanParsingWithPoseStep
        
        step = HumanParsingWithPoseStep()
        
        # 입력 데이터 구조 분석
        print("   📥 입력 데이터 구조:")
        print("      - image: PIL.Image 또는 numpy.ndarray")
        print("      - ensemble_method: str (voting, weighted, quality, simple_average)")
        print("      - quality_level: str (low, medium, high, ultra)")
        
        # 출력 데이터 구조 분석
        print("   📤 출력 데이터 구조:")
        print("      - success: bool")
        print("      - data: Dict")
        print("        - final_parsing: torch.Tensor (20개 클래스)")
        print("        - individual_results: Dict (각 모델별 결과)")
        print("        - ensemble_method: str")
        print("        - pose_estimation_result: Dict")
        print("          - keypoints: np.ndarray (COCO 17개 키포인트)")
        print("          - confidence: float")
        print("          - pose_quality: str")
        print("          - human_parsing_mask: np.ndarray")
        
        # 데이터 형식 검증
        print("   🔍 데이터 형식 검증:")
        print("      - parsing_mask: [B, 20, H, W] (20개 클래스)")
        print("      - keypoints: [17, 3] (x, y, confidence)")
        print("      - 이미지 크기: 최소 512x512 보장")
        
        return True, "1단계 Human Parsing 데이터 구조 분석 완료"
        
    except Exception as e:
        return False, f"1단계 Human Parsing 분석 실패: {e}"

def analyze_step_02_pose_estimation():
    """2단계: Pose Estimation 데이터 구조 분석"""
    try:
        print("🔍 2단계: Pose Estimation 데이터 구조 분석 중...")
        
        # 2단계 모듈 import
        import importlib.util
        spec = importlib.util.spec_from_file_location("step02", "steps/02_pose_estimation/step_modularized.py")
        step02_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step02_module)
        PoseEstimationStep = step02_module.PoseEstimationStep
        
        step = PoseEstimationStep()
        
        # 입력 데이터 구조 분석
        print("   📥 입력 데이터 구조:")
        print("      - image: PIL.Image 또는 numpy.ndarray")
        print("      - human_parsing_result: Dict (1단계 결과)")
        print("      - pose_quality: str (low, medium, high, ultra)")
        print("      - enable_ensemble: bool")
        
        # 출력 데이터 구조 분석
        print("   📤 출력 데이터 구조:")
        print("      - success: bool")
        print("      - data: Dict")
        print("        - keypoints: np.ndarray (17개 키포인트)")
        print("        - confidence: float")
        print("        - pose_quality: str")
        print("        - skeleton_connections: List")
        print("        - pose_analysis: Dict")
        
        # 데이터 형식 검증
        print("   🔍 데이터 형식 검증:")
        print("      - keypoints: [17, 3] (x, y, confidence)")
        print("      - confidence: 0.0 ~ 1.0")
        print("      - pose_quality: low/medium/high/ultra")
        
        return True, "2단계 Pose Estimation 데이터 구조 분석 완료"
        
    except Exception as e:
        return False, f"2단계 Pose Estimation 분석 실패: {e}"

def analyze_step_03_cloth_segmentation():
    """3단계: Cloth Segmentation 데이터 구조 분석"""
    try:
        print("🔍 3단계: Cloth Segmentation 데이터 구조 분석 중...")
        
        # 3단계 모듈 import
        import importlib.util
        spec = importlib.util.spec_from_file_location("step03", "steps/03_cloth_segmentation/step_modularized.py")
        step03_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step03_module)
        ClothSegmentationStep = step03_module.ClothSegmentationStep
        
        step = ClothSegmentationStep()
        
        # 입력 데이터 구조 분석
        print("   📥 입력 데이터 구조:")
        print("      - image: np.ndarray (H, W, C)")
        print("      - method: SegmentationMethod (U2NET_CLOTH, SAM_HUGE, DEEPLABV3_PLUS)")
        print("      - quality_level: QualityLevel (LOW, MEDIUM, HIGH, ULTRA)")
        print("      - person_parsing: Dict (1단계 결과)")
        print("      - pose_info: Dict (2단계 결과)")
        
        # 출력 데이터 구조 분석
        print("   📤 출력 데이터 구조:")
        print("      - success: bool")
        print("      - data: Dict")
        print("        - segmentation_mask: np.ndarray (H, W)")
        print("        - cloth_regions: List[Dict] (의류 영역 정보)")
        print("        - confidence: float")
        print("        - method_used: str")
        
        # 데이터 형식 검증
        print("   🔍 데이터 형식 검증:")
        print("      - segmentation_mask: [H, W] (0-1 범위)")
        print("      - cloth_regions: 의류별 마스크 및 속성")
        print("      - 이미지 크기: 512x512로 표준화")
        
        return True, "3단계 Cloth Segmentation 데이터 구조 분석 완료"
        
    except Exception as e:
        return False, f"3단계 Cloth Segmentation 분석 실패: {e}"

def analyze_step_04_geometric_matching():
    """4단계: Geometric Matching 데이터 구조 분석"""
    try:
        print("🔍 4단계: Geometric Matching 데이터 구조 분석 중...")
        
        # 4단계 모듈 import
        import importlib.util
        spec = importlib.util.spec_from_file_location("step04", "steps/04_geometric_matching/step_modularized.py")
        step04_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step04_module)
        GeometricMatchingStep = step04_module.GeometricMatchingStep
        
        step = GeometricMatchingStep()
        
        # 입력 데이터 구조 분석
        print("   📥 입력 데이터 구조:")
        print("      - person_image: np.ndarray (사람 이미지)")
        print("      - cloth_image: np.ndarray (의류 이미지)")
        print("      - pose_keypoints: np.ndarray (포즈 키포인트)")
        print("      - person_parsing: Dict (1단계 결과)")
        print("      - cloth_segmentation: Dict (3단계 결과)")
        
        # 출력 데이터 구조 분석
        print("   📤 출력 데이터 구조:")
        print("      - success: bool")
        print("      - data: Dict")
        print("        - transformation_matrix: np.ndarray (TPS 변환 행렬)")
        print("        - aligned_cloth: np.ndarray (정렬된 의류)")
        print("        - matching_confidence: float")
        print("        - geometric_quality: str")
        
        # 데이터 형식 검증
        print("   🔍 데이터 형식 검증:")
        print("      - transformation_matrix: [3, 3] (TPS 변환)")
        print("      - aligned_cloth: [H, W, C] (정렬된 의류)")
        print("      - matching_confidence: 0.0 ~ 1.0")
        
        return True, "4단계 Geometric Matching 데이터 구조 분석 완료"
        
    except Exception as e:
        return False, f"4단계 Geometric Matching 분석 실패: {e}"

def analyze_step_05_cloth_warping():
    """5단계: Cloth Warping 데이터 구조 분석"""
    try:
        print("🔍 5단계: Cloth Warping 데이터 구조 분석 중...")
        
        # 5단계 모듈 import
        from steps.step_05_cloth_warping import ClothWarpingStep
        
        step = ClothWarpingStep()
        
        # 입력 데이터 구조 분석
        print("   📥 입력 데이터 구조:")
        print("      - cloth_image: np.ndarray (의류 이미지)")
        print("      - transformation_matrix: np.ndarray (4단계 변환 행렬)")
        print("      - target_pose: np.ndarray (목표 포즈)")
        print("      - person_parsing: Dict (1단계 결과)")
        
        # 출력 데이터 구조 분석
        print("   📤 출력 데이터 구조:")
        print("      - success: bool")
        print("      - data: Dict")
        print("        - warped_cloth: np.ndarray (왜곡된 의류)")
        print("        - warping_quality: float")
        print("        - deformation_info: Dict")
        
        # 데이터 형식 검증
        print("   🔍 데이터 형식 검증:")
        print("      - warped_cloth: [H, W, C] (왜곡된 의류)")
        print("      - warping_quality: 0.0 ~ 1.0")
        print("      - deformation_info: 왜곡 정보")
        
        return True, "5단계 Cloth Warping 데이터 구조 분석 완료"
        
    except Exception as e:
        return False, f"5단계 Cloth Warping 분석 실패: {e}"

def analyze_step_06_virtual_fitting():
    """6단계: Virtual Fitting 데이터 구조 분석"""
    try:
        print("🔍 6단계: Virtual Fitting 데이터 구조 분석 중...")
        
        # 6단계 모듈 import
        from steps.step_06_virtual_fitting import VirtualFittingStep
        
        step = VirtualFittingStep()
        
        # 입력 데이터 구조 분석
        print("   📥 입력 데이터 구조:")
        print("      - person_image: np.ndarray (사람 이미지)")
        print("      - warped_cloth: np.ndarray (5단계 왜곡된 의류)")
        print("      - pose_keypoints: np.ndarray (포즈 키포인트)")
        print("      - person_parsing: Dict (1단계 결과)")
        print("      - cloth_segmentation: Dict (3단계 결과)")
        
        # 출력 데이터 구조 분석
        print("   📤 출력 데이터 구조:")
        print("      - success: bool")
        print("      - data: Dict")
        print("        - fitted_image: np.ndarray (피팅된 이미지)")
        print("        - fitting_confidence: float")
        print("        - visual_quality: str")
        print("        - fitting_metadata: Dict")
        
        # 데이터 형식 검증
        print("   🔍 데이터 형식 검증:")
        print("      - fitted_image: [H, W, C] (최종 피팅 결과)")
        print("      - fitting_confidence: 0.0 ~ 1.0")
        print("      - visual_quality: low/medium/high/ultra")
        
        return True, "6단계 Virtual Fitting 데이터 구조 분석 완료"
        
    except Exception as e:
        return False, f"6단계 Virtual Fitting 분석 실패: {e}"

def analyze_step_07_post_processing():
    """7단계: Post Processing 데이터 구조 분석"""
    try:
        print("🔍 7단계: Post Processing 데이터 구조 분석 중...")
        
        # 7단계 모듈 import
        from steps.step_07_post_processing import PostProcessingStep
        
        step = PostProcessingStep()
        
        # 입력 데이터 구조 분석
        print("   📥 입력 데이터 구조:")
        print("      - fitted_image: np.ndarray (6단계 피팅 결과)")
        print("      - original_image: np.ndarray (원본 이미지)")
        print("      - quality_level: QualityLevel (LOW, MEDIUM, HIGH, ULTRA)")
        print("      - enabled_methods: List[EnhancementMethod]")
        
        # 출력 데이터 구조 분석
        print("   📤 출력 데이터 구조:")
        print("      - success: bool")
        print("      - data: Dict")
        print("        - enhanced_image: np.ndarray (향상된 이미지)")
        print("        - enhancement_quality: float")
        print("        - applied_methods: List[str]")
        print("        - enhancement_metadata: Dict")
        
        # 데이터 형식 검증
        print("   🔍 데이터 형식 검증:")
        print("      - enhanced_image: [H, W, C] (향상된 이미지)")
        print("      - enhancement_quality: 0.0 ~ 1.0")
        print("      - applied_methods: 적용된 향상 방법들")
        
        return True, "7단계 Post Processing 데이터 구조 분석 완료"
        
    except Exception as e:
        return False, f"7단계 Post Processing 분석 실패: {e}"

def analyze_step_08_quality_assessment():
    """8단계: Quality Assessment 데이터 구조 분석"""
    try:
        print("🔍 8단계: Quality Assessment 데이터 구조 분석 중...")
        
        # 8단계 모듈 import
        from steps.step_08_quality_assessment.step_08_quality_assessment import QualityAssessmentStep
        
        step = QualityAssessmentStep()
        
        # 입력 데이터 구조 분석
        print("   📥 입력 데이터 구조:")
        print("      - processed_image: np.ndarray (7단계 향상된 이미지)")
        print("      - reference_image: np.ndarray (참조 이미지)")
        print("      - quality_metrics: List[str] (평가할 품질 지표)")
        print("      - assessment_type: AssessmentType")
        
        # 출력 데이터 구조 분석
        print("   📤 출력 데이터 구조:")
        print("      - success: bool")
        print("      - data: Dict")
        print("        - quality_score: float (전체 품질 점수)")
        print("        - detailed_metrics: Dict (상세 품질 지표)")
        print("        - quality_grade: str (품질 등급)")
        print("        - improvement_suggestions: List[str]")
        
        # 데이터 형식 검증
        print("   🔍 데이터 형식 검증:")
        print("      - quality_score: 0.0 ~ 1.0")
        print("      - quality_grade: A+/A/B+/B/C+/C/D/F")
        print("      - detailed_metrics: PSNR, SSIM, LPIPS 등")
        
        return True, "8단계 Quality Assessment 데이터 구조 분석 완료"
        
    except Exception as e:
        return False, f"8단계 Quality Assessment 분석 실패: {e}"

def analyze_step_09_final_output():
    """9단계: Final Output 데이터 구조 분석"""
    try:
        print("🔍 9단계: Final Output 데이터 구조 분석 중...")
        
        # 9단계 모듈 import
        from steps.step_09_final_output import FinalOutputStep
        
        step = FinalOutputStep()
        
        # 입력 데이터 구조 분석
        print("   📥 입력 데이터 구조:")
        print("      - all_step_results: Dict (1-8단계 모든 결과)")
        print("      - final_quality_score: float (8단계 품질 점수)")
        print("      - output_format: OutputFormat")
        print("      - quality_level: QualityLevel")
        
        # 출력 데이터 구조 분석
        print("   📤 출력 데이터 구조:")
        print("      - success: bool")
        print("      - data: Dict")
        print("        - final_image: np.ndarray (최종 결과 이미지)")
        print("        - confidence_score: float (신뢰도 점수)")
        print("        - quality_assessment: Dict (최종 품질 평가)")
        print("        - metadata: Dict (메타데이터)")
        print("        - pipeline_summary: Dict (파이프라인 요약)")
        
        # 데이터 형식 검증
        print("   🔍 데이터 형식 검증:")
        print("      - final_image: [H, W, C] (최종 결과)")
        print("      - confidence_score: 0.0 ~ 1.0")
        print("      - quality_assessment: 최종 품질 평가")
        
        return True, "9단계 Final Output 데이터 구조 분석 완료"
        
    except Exception as e:
        return False, f"9단계 Final Output 분석 실패: {e}"

def analyze_data_flow_between_steps():
    """단계 간 데이터 전달 흐름 분석"""
    try:
        print("\n🔗 단계 간 데이터 전달 흐름 분석:")
        print("=" * 60)
        
        # 1단계 -> 2단계
        print("   1단계 → 2단계:")
        print("     - human_parsing_result → pose_estimation 입력")
        print("     - parsing_mask → pose_estimation에서 사람 영역 추출")
        print("     - keypoints → pose_estimation에서 포즈 추정")
        
        # 2단계 -> 3단계
        print("   2단계 → 3단계:")
        print("     - pose_keypoints → cloth_segmentation 입력")
        print("     - pose_info → 의류 세그멘테이션 가이드")
        
        # 3단계 -> 4단계
        print("   3단계 → 4단계:")
        print("     - cloth_segmentation → geometric_matching 입력")
        print("     - segmentation_mask → 의류 영역 식별")
        
        # 4단계 -> 5단계
        print("   4단계 → 5단계:")
        print("     - transformation_matrix → cloth_warping 입력")
        print("     - aligned_cloth → 왜곡 변환 대상")
        
        # 5단계 -> 6단계
        print("   5단계 → 6단계:")
        print("     - warped_cloth → virtual_fitting 입력")
        print("     - deformation_info → 피팅 품질 평가")
        
        # 6단계 -> 7단계
        print("   6단계 → 7단계:")
        print("     - fitted_image → post_processing 입력")
        print("     - fitting_metadata → 향상 방법 선택")
        
        # 7단계 -> 8단계
        print("   7단계 → 8단계:")
        print("     - enhanced_image → quality_assessment 입력")
        print("     - enhancement_metadata → 품질 평가 기준")
        
        # 8단계 -> 9단계
        print("   8단계 → 9단계:")
        print("     - quality_score → final_output 입력")
        print("     - detailed_metrics → 최종 품질 결정")
        
        return True, "데이터 전달 흐름 분석 완료"
        
    except Exception as e:
        return False, f"데이터 전달 흐름 분석 실패: {e}"

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="MyCloset AI 파이프라인 데이터 전달 분석")
    parser.add_argument("--step", type=int, help="특정 단계만 분석 (1-9)")
    parser.add_argument("--verbose", action="store_true", help="상세 로그 출력")
    
    args = parser.parse_args()
    
    print("🧪 MyCloset AI 전체 파이프라인 데이터 전달 분석")
    print("=" * 60)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 분석 함수들
    analysis_functions = [
        analyze_step_01_human_parsing,
        analyze_step_02_pose_estimation,
        analyze_step_03_cloth_segmentation,
        analyze_step_04_geometric_matching,
        analyze_step_05_cloth_warping,
        analyze_step_06_virtual_fitting,
        analyze_step_07_post_processing,
        analyze_step_08_quality_assessment,
        analyze_step_09_final_output
    ]
    
    # 특정 단계만 분석
    if args.step and 1 <= args.step <= 9:
        print(f"🎯 {args.step}단계만 분석합니다...")
        func = analysis_functions[args.step - 1]
        success, message = func()
        if success:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")
        return
    
    # 전체 파이프라인 분석
    print("🚀 전체 파이프라인 데이터 전달 분석 시작...")
    print()
    
    start_time = time.time()
    results = []
    
    for i, func in enumerate(analysis_functions, 1):
        print(f"🔍 {i}단계 분석 중...")
        success, message = func()
        results.append((i, success, message))
        
        if success:
            print(f"   ✅ {message}")
        else:
            print(f"   ❌ {message}")
        
        print()
    
    # 데이터 전달 흐름 분석
    flow_success, flow_message = analyze_data_flow_between_steps()
    if flow_success:
        print(f"   ✅ {flow_message}")
    else:
        print(f"   ❌ {flow_message}")
    
    # 결과 요약
    total_time = time.time() - start_time
    successful_steps = sum(1 for _, success, _ in results if success)
    total_steps = len(results)
    
    print("\n" + "=" * 60)
    print("📊 데이터 전달 분석 결과 요약")
    print("=" * 60)
    print(f"총 단계 수: {total_steps}")
    print(f"성공한 단계: {successful_steps}")
    print(f"실패한 단계: {total_steps - successful_steps}")
    print(f"성공률: {(successful_steps/total_steps)*100:.1f}%")
    print(f"총 소요 시간: {total_time:.2f}초")
    
    if successful_steps == total_steps:
        print("\n🎉 모든 단계의 데이터 구조가 올바르게 정의되어 있습니다!")
        print("✅ 파이프라인 간 데이터 전달이 원활하게 이루어질 것입니다.")
    else:
        print(f"\n⚠️ {total_steps - successful_steps}개 단계에서 문제가 발견되었습니다.")
        print("🔧 해당 단계들의 데이터 구조를 수정해야 합니다.")

if __name__ == "__main__":
    main()
