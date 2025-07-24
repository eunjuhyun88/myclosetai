#!/usr/bin/env python3
"""
Step 01 Human Parsing 실제 작동 테스트
실행 방법: python test_step01.py
"""

import asyncio
import sys
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_step01_actual_operation():
    """Step 01 실제 작동 테스트"""
    
    print("🧪 Step 01 Human Parsing 실제 작동 테스트 시작")
    print("=" * 60)
    
    test_results = {}
    
    # 1. 모듈 임포트 테스트
    print("🔍 1. 모듈 임포트 테스트...")
    try:
        from backend.app.ai_pipeline.steps.step_01_human_parsing import (
            HumanParsingStep,
            create_human_parsing_step,
            test_all_features,
            test_strict_mode,
            test_real_model_loading
        )
        test_results["import"] = True
        print("   ✅ 모듈 임포트 성공")
    except ImportError as e:
        test_results["import"] = False
        print(f"   ❌ 모듈 임포트 실패: {e}")
        return test_results
    
    # 2. ModelLoader 연동 테스트
    print("\n🔍 2. ModelLoader 연동 테스트...")
    try:
        from backend.app.ai_pipeline.utils.model_loader import get_global_model_loader
        model_loader = get_global_model_loader()
        test_results["model_loader"] = True
        print("   ✅ ModelLoader 연동 성공")
    except Exception as e:
        test_results["model_loader"] = False
        print(f"   ❌ ModelLoader 연동 실패: {e}")
    
    # 3. BaseStepMixin 연동 테스트
    print("\n🔍 3. BaseStepMixin 연동 테스트...")
    try:
        from backend.app.ai_pipeline.steps.base_step_mixin import BaseStepMixin
        test_results["base_step_mixin"] = True
        print("   ✅ BaseStepMixin 연동 성공")
    except Exception as e:
        test_results["base_step_mixin"] = False
        print(f"   ❌ BaseStepMixin 연동 실패: {e}")
    
    # 4. Step 생성 테스트 (strict_mode=False로 안전하게)
    print("\n🔍 4. Step 생성 테스트...")
    try:
        step = await create_human_parsing_step(
            device="cpu",  # 안전한 CPU 모드
            config={
                "strict_mode": False,  # 실패 허용 모드
                "enable_visualization": False,  # 시각화 비활성화
                "warmup_enabled": False  # 워밍업 비활성화
            }
        )
        test_results["step_creation"] = True
        print("   ✅ Step 생성 성공")
        
        # 5. Step 정보 확인
        step_info = await step.get_step_info()
        print(f"   📊 Step 정보: {step_info.get('step_name', 'unknown')}")
        print(f"   🔧 디바이스: {step_info.get('device', 'unknown')}")
        print(f"   🚨 Strict Mode: {step_info.get('config', {}).get('strict_mode', False)}")
        
    except Exception as e:
        test_results["step_creation"] = False
        print(f"   ❌ Step 생성 실패: {e}")
    
    # 6. 더미 이미지 처리 테스트
    print("\n🔍 5. 더미 이미지 처리 테스트...")
    try:
        import torch
        dummy_image = torch.randn(1, 3, 512, 512)
        
        result = await step.process(dummy_image)
        
        if result.get("success"):
            test_results["processing"] = True
            print("   ✅ 이미지 처리 성공")
            print(f"   📊 신뢰도: {result.get('confidence', 0):.3f}")
            print(f"   ⏱️ 처리 시간: {result.get('processing_time', 0):.3f}초")
        else:
            test_results["processing"] = False
            print(f"   ❌ 이미지 처리 실패: {result.get('message', 'Unknown')}")
            
    except Exception as e:
        test_results["processing"] = False
        print(f"   ❌ 이미지 처리 테스트 실패: {e}")
    
    # 7. 리소스 정리
    print("\n🔍 6. 리소스 정리 테스트...")
    try:
        if 'step' in locals():
            await step.cleanup()
        test_results["cleanup"] = True
        print("   ✅ 리소스 정리 성공")
    except Exception as e:
        test_results["cleanup"] = False
        print(f"   ❌ 리소스 정리 실패: {e}")
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📋 테스트 결과 요약")
    print("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    success_rate = (passed_tests / total_tests) * 100
    
    for test_name, result in test_results.items():
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name:20} : {status}")
    
    print("-" * 60)
    print(f"전체 테스트: {total_tests}개")
    print(f"통과: {passed_tests}개 | 실패: {total_tests - passed_tests}개")
    print(f"성공률: {success_rate:.1f}%")
    
    # 최종 판정
    if success_rate >= 90:
        print("\n🎉 Step 01이 완벽하게 작동합니다!")
        verdict = "완전 작동"
    elif success_rate >= 70:
        print("\n✅ Step 01이 대부분 작동합니다!")
        verdict = "대부분 작동"
    elif success_rate >= 50:
        print("\n⚠️ Step 01이 부분적으로 작동합니다.")
        verdict = "부분 작동"
    else:
        print("\n❌ Step 01에 문제가 있습니다.")
        verdict = "작동 안함"
    
    print("=" * 60)
    
    return {
        "verdict": verdict,
        "success_rate": success_rate,
        "test_results": test_results
    }

async def test_step01_strict_mode():
    """Step 01 Strict Mode 테스트"""
    print("\n🚨 Step 01 Strict Mode 테스트")
    print("=" * 40)
    
    try:
        from backend.app.ai_pipeline.steps.step_01_human_parsing import create_human_parsing_step
        
        # Strict Mode로 Step 생성 시도
        step = await create_human_parsing_step(
            device="auto",
            config={
                "strict_mode": True,  # 🔥 엄격 모드
                "enable_visualization": True,
                "model_name": "human_parsing_graphonomy"
            }
        )
        
        print("✅ Strict Mode Step 생성 성공")
        print("🔥 실제 AI 모델이 정상 작동 중!")
        
        await step.cleanup()
        return True
        
    except RuntimeError as e:
        print(f"🚨 예상된 Strict Mode 에러: {e}")
        print("💡 실제 AI 모델 체크포인트가 필요합니다")
        return False
    except Exception as e:
        print(f"❌ 예상치 못한 에러: {e}")
        return False

def run_comprehensive_test():
    """종합 테스트 실행"""
    try:
        # 기본 작동 테스트
        result = asyncio.run(test_step01_actual_operation())
        
        # Strict Mode 테스트
        strict_result = asyncio.run(test_step01_strict_mode())
        
        return result, strict_result
        
    except Exception as e:
        print(f"❌ 테스트 실행 실패: {e}")
        return None, None

if __name__ == "__main__":
    print("🎯 Step 01 Human Parsing 종합 작동 테스트")
    print("🔧 이 테스트는 Step 01이 실제로 작동하는지 확인합니다")
    print()
    
    basic_result, strict_result = run_comprehensive_test()
    
    print("\n" + "🎯 최종 결론" + "=" * 50)
    
    if basic_result and basic_result["success_rate"] >= 70:
        if strict_result:
            print("🎉 Step 01이 실제 AI 모델과 함께 완벽 작동!")
            print("✅ 실제 프로덕션 환경에서 사용 가능")
        else:
            print("✅ Step 01 구조는 정상, 실제 AI 모델 필요")
            print("💡 ModelLoader + 체크포인트 파일 설정 후 완전 작동")
    else:
        print("❌ Step 01에 구조적 문제가 있습니다")
        print("🔧 의존성 모듈 확인 및 수정 필요")
    
    print("=" * 70)