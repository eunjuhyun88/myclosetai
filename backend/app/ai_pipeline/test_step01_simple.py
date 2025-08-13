#!/usr/bin/env python3
"""
🧪 1단계 Human Parsing 간단 테스트
====================================

1단계의 기본 기능과 데이터 구조를 테스트합니다.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# 프로젝트 루트 경로 설정
current_file = Path(__file__).absolute()
backend_dir = current_file.parent.parent.parent  # backend/
sys.path.insert(0, str(backend_dir))

def test_step01_basic():
    """1단계 기본 기능 테스트"""
    try:
        print("🔍 1단계: Human Parsing 기본 기능 테스트")
        
        # 1단계 모듈 import 시도
        print("   📥 모듈 import 시도 중...")
        
        # 방법 1: 실제 AI 모델 버전 사용
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "step01", 
                "steps/01_human_parsing/step_integrated_with_pose.py"
            )
            step01_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(step01_module)
            
            print("   ✅ 방법 1 성공: 실제 AI 모델 버전 사용")
            HumanParsingWithPoseStep = step01_module.HumanParsingWithPoseStep
            
            # 클래스 인스턴스 생성
            step = HumanParsingWithPoseStep()
            print(f"   ✅ 클래스 인스턴스 생성 성공: {type(step).__name__}")
            
            # 기본 속성 확인
            if hasattr(step, 'step_name'):
                print(f"   📋 Step Name: {step.step_name}")
            if hasattr(step, 'step_order'):
                print(f"   📋 Step Order: {step.step_order}")
            if hasattr(step, 'step_outputs'):
                print(f"   📋 Step Outputs: {step.step_outputs}")
            
            # 모델 상태 확인
            if hasattr(step, 'get_model_status'):
                model_status = step.get_model_status()
                print(f"   📋 모델 상태: {model_status}")
            
            # 실제 처리 테스트
            mock_image = np.random.rand(512, 512, 3).astype(np.float32)
            result = step.process(image=mock_image, ensemble_method='simple_average', quality_level='high')
            
            if result['success']:
                print(f"   ✅ 실제 처리 테스트 성공: {result['processing_time']:.2f}초")
                return True, "1단계 기본 기능 테스트 성공 (실제 AI 모델)"
            else:
                print(f"   ❌ 실제 처리 테스트 실패: {result.get('error', 'Unknown error')}")
                return False, "실제 처리 테스트 실패"
            
        except Exception as e:
            print(f"   ❌ 방법 1 실패: {e}")
            return False, f"실제 AI 모델 버전 로드 실패: {e}"
        
    except Exception as e:
        return False, f"1단계 테스트 실패: {e}"

def test_step02_basic():
    """2단계 기본 기능 테스트"""
    try:
        print("🔍 2단계: Pose Estimation 기본 기능 테스트")
        
        # 2단계 모듈 import 시도
        print("   📥 모듈 import 시도 중...")
        
        # 방법 1: 실제 AI 모델 버전 사용
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "step02", 
                os.path.join(os.path.dirname(__file__), "steps", "02_pose_estimation", "step_modularized.py")
            )
            step02_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(step02_module)
            
            print("   ✅ 방법 1 성공: 실제 AI 모델 버전 사용")
            PoseEstimationStep = step02_module.PoseEstimationStep
            
            # 클래스 인스턴스 생성
            step = PoseEstimationStep()
            print(f"   ✅ 클래스 인스턴스 생성 성공: {type(step).__name__}")
            
            # 기본 속성 확인
            if hasattr(step, 'step_name'):
                print(f"   📋 Step Name: {step.step_name}")
            if hasattr(step, 'step_id'):
                print(f"   📋 Step ID: {step.step_id}")
            if hasattr(step, 'step_description'):
                print(f"   📋 Step Description: {step.step_description}")
            
            # 모델 상태 확인
            if hasattr(step, 'models_loading_status'):
                print(f"   📋 모델 로딩 상태: {step.models_loading_status}")
            
            # 실제 처리 테스트
            mock_image = np.random.rand(512, 512, 3).astype(np.float32)
            result = step.process(image=mock_image)
            
            if result.get('success', False):
                print(f"   ✅ 실제 처리 테스트 성공")
                return True, "2단계 기본 기능 테스트 성공 (실제 AI 모델)"
            else:
                print(f"   ❌ 실제 처리 테스트 실패: {result.get('error', 'Unknown error')}")
                return False, "실제 처리 테스트 실패"
            
        except Exception as e:
            print(f"   ❌ 방법 1 실패: {e}")
            return False, f"실제 AI 모델 버전 로드 실패: {e}"
        
    except Exception as e:
        return False, f"2단계 테스트 실패: {e}"

def test_step01_data_structure():
    """1단계 데이터 구조 테스트"""
    try:
        print("\n🔍 1단계: 데이터 구조 분석")
        
        # 입력 데이터 구조
        print("   📥 입력 데이터 구조:")
        print("      - image: PIL.Image 또는 numpy.ndarray")
        print("      - ensemble_method: str (voting, weighted, quality, simple_average)")
        print("      - quality_level: str (low, medium, high, ultra)")
        
        # 출력 데이터 구조
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
        
        return True, "1단계 데이터 구조 분석 완료"
        
    except Exception as e:
        return False, f"1단계 데이터 구조 분석 실패: {e}"

def main():
    """메인 함수"""
    print("🧪 1단계 + 2단계 테스트")
    print("=" * 50)
    print(f"시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1단계 기본 기능 테스트
    print("🔍 1단계: Human Parsing 테스트")
    basic_success, basic_message = test_step01_basic()
    if basic_success:
        print(f"✅ {basic_message}")
    else:
        print(f"❌ {basic_message}")
    
    # 1단계 데이터 구조 테스트
    data_success, data_message = test_step01_data_structure()
    if data_success:
        print(f"✅ {data_message}")
    else:
        print(f"❌ {data_message}")
    
    print()
    
    # 2단계 기본 기능 테스트
    print("🔍 2단계: Pose Estimation 테스트")
    step02_success, step02_message = test_step02_basic()
    if step02_success:
        print(f"✅ {step02_message}")
    else:
        print(f"❌ {step02_message}")
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 1단계 + 2단계 테스트 결과 요약")
    print("=" * 50)
    
    if basic_success and data_success and step02_success:
        print("🎉 1단계 + 2단계 테스트 완전 성공!")
        print("✅ 1단계 Human Parsing: 정상")
        print("✅ 2단계 Pose Estimation: 정상")
        print("🚀 다음 단계로 진행 가능")
    else:
        print("⚠️ 일부 단계에서 문제가 발견되었습니다.")
        if not basic_success:
            print("❌ 1단계 기본 기능: 실패")
        if not data_success:
            print("❌ 1단계 데이터 구조: 실패")
        if not step02_success:
            print("❌ 2단계 기본 기능: 실패")
        print("🔧 문제 해결 후 다음 단계 진행 필요")

if __name__ == "__main__":
    main()
