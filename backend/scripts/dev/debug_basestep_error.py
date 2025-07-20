# debug_basestep_error.py
"""
🔍 BaseStepMixin 초기화 실패 에러 추적 스크립트
어떤 Step에서 에러가 발생하는지 정확히 찾아보기
"""

import logging
import traceback
import sys
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_step_initialization():
    """모든 Step 클래스 초기화 테스트"""
    print("🔍 모든 Step 클래스 초기화 테스트")
    print("=" * 60)
    
    step_classes = [
        ("HumanParsingStep", "app.ai_pipeline.steps.step_01_human_parsing"),
        ("PoseEstimationStep", "app.ai_pipeline.steps.step_02_pose_estimation"),
        ("ClothSegmentationStep", "app.ai_pipeline.steps.step_03_cloth_segmentation"),
        ("GeometricMatchingStep", "app.ai_pipeline.steps.step_04_geometric_matching"),
        ("ClothWarpingStep", "app.ai_pipeline.steps.step_05_cloth_warping"),
        ("VirtualFittingStep", "app.ai_pipeline.steps.step_06_virtual_fitting"),
        ("PostProcessingStep", "app.ai_pipeline.steps.step_07_post_processing"),
        ("QualityAssessmentStep", "app.ai_pipeline.steps.step_08_quality_assessment"),
    ]
    
    results = {}
    
    for step_name, module_path in step_classes:
        print(f"\n🔧 {step_name} 테스트 중...")
        
        try:
            # 모듈 import
            module = __import__(module_path, fromlist=[step_name])
            step_class = getattr(module, step_name)
            
            print(f"   ✅ Import 성공: {step_name}")
            
            # 클래스 MRO 확인
            mro = [cls.__name__ for cls in step_class.__mro__]
            print(f"   📋 MRO: {' -> '.join(mro)}")
            
            # 인스턴스 생성 시도
            try:
                # 기본 파라미터로 인스턴스 생성
                step_instance = step_class(device="cpu")
                print(f"   ✅ 인스턴스 생성 성공")
                
                # 기본 속성 확인
                attrs = ['logger', 'step_name', 'device', 'is_initialized']
                for attr in attrs:
                    if hasattr(step_instance, attr):
                        value = getattr(step_instance, attr)
                        print(f"   📍 {attr}: {value}")
                    else:
                        print(f"   ⚠️ {attr}: 누락")
                
                results[step_name] = {
                    'import': True,
                    'instance': True,
                    'error': None
                }
                
            except Exception as init_error:
                print(f"   ❌ 인스턴스 생성 실패: {init_error}")
                print(f"   📋 에러 타입: {type(init_error).__name__}")
                
                # 상세 에러 분석
                if "object.__init__()" in str(init_error):
                    print(f"   🎯 BaseStepMixin super() 호출 문제 발견!")
                    print(f"   📋 MRO 체인에서 super() 호출 시 object에 파라미터 전달")
                
                # 스택 트레이스 출력
                print(f"   📋 스택 트레이스:")
                for line in traceback.format_exc().split('\n')[-10:]:
                    if line.strip():
                        print(f"      {line}")
                
                results[step_name] = {
                    'import': True,
                    'instance': False,
                    'error': str(init_error),
                    'error_type': type(init_error).__name__
                }
        
        except Exception as import_error:
            print(f"   ❌ Import 실패: {import_error}")
            results[step_name] = {
                'import': False,
                'instance': False,
                'error': str(import_error),
                'error_type': type(import_error).__name__
            }
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    problem_steps = []
    
    for step_name, result in results.items():
        if result['import'] and result['instance']:
            print(f"✅ {step_name}: 성공")
            success_count += 1
        else:
            print(f"❌ {step_name}: 실패 - {result.get('error', 'Unknown')}")
            fail_count += 1
            problem_steps.append(step_name)
    
    print(f"\n📊 전체 결과: 성공 {success_count}개, 실패 {fail_count}개")
    
    if problem_steps:
        print(f"\n🎯 문제가 있는 Step들:")
        for step in problem_steps:
            error_info = results[step]
            print(f"   ❌ {step}")
            print(f"      에러: {error_info.get('error', 'Unknown')}")
            print(f"      타입: {error_info.get('error_type', 'Unknown')}")
    
    return results

def test_base_step_mixin_directly():
    """BaseStepMixin 직접 테스트"""
    print("\n🔧 BaseStepMixin 직접 테스트")
    print("=" * 60)
    
    try:
        from app.ai_pipeline.steps.base_step_mixin import BaseStepMixin
        
        print("✅ BaseStepMixin import 성공")
        
        # 직접 인스턴스 생성
        try:
            mixin = BaseStepMixin(device="cpu")
            print("✅ BaseStepMixin 직접 인스턴스 생성 성공")
            print(f"   logger: {getattr(mixin, 'logger', 'None')}")
            print(f"   device: {getattr(mixin, 'device', 'None')}")
            print(f"   step_name: {getattr(mixin, 'step_name', 'None')}")
            
        except Exception as e:
            print(f"❌ BaseStepMixin 직접 인스턴스 생성 실패: {e}")
            print(f"   📋 에러 타입: {type(e).__name__}")
            traceback.print_exc()
    
    except Exception as e:
        print(f"❌ BaseStepMixin import 실패: {e}")

def analyze_inheritance_chain():
    """상속 체인 분석"""
    print("\n🔍 상속 체인 분석")
    print("=" * 60)
    
    try:
        # VirtualFittingStep 특별히 확인 (로그에서 자주 보이는 문제)
        from app.ai_pipeline.steps.step_06_virtual_fitting import VirtualFittingStep
        
        print("📋 VirtualFittingStep 상속 체인:")
        for i, cls in enumerate(VirtualFittingStep.__mro__):
            print(f"   {i+1}. {cls.__name__} ({cls.__module__})")
            
            # __init__ 메서드 확인
            if hasattr(cls, '__init__'):
                init_method = cls.__init__
                print(f"      __init__: {init_method}")
                
                # 함수 시그니처 확인
                try:
                    import inspect
                    sig = inspect.signature(init_method)
                    print(f"      시그니처: {sig}")
                except:
                    print(f"      시그니처: 확인 불가")
        
        # 실제 인스턴스 생성 과정 추적
        print(f"\n🔧 VirtualFittingStep 인스턴스 생성 과정 추적:")
        
        class DebugVirtualFittingStep(VirtualFittingStep):
            def __init__(self, *args, **kwargs):
                print(f"   1. DebugVirtualFittingStep.__init__ 시작")
                print(f"      args: {args}")
                print(f"      kwargs: {kwargs}")
                
                try:
                    super().__init__(*args, **kwargs)
                    print(f"   2. super().__init__ 성공")
                except Exception as e:
                    print(f"   2. super().__init__ 실패: {e}")
                    raise
        
        # 디버그 인스턴스 생성
        debug_instance = DebugVirtualFittingStep(device="cpu")
        print(f"   3. 디버그 인스턴스 생성 성공")
        
    except Exception as e:
        print(f"❌ 상속 체인 분석 실패: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("🔍 BaseStepMixin 에러 추적 시작")
    print("=" * 80)
    
    # 1. BaseStepMixin 직접 테스트
    test_base_step_mixin_directly()
    
    # 2. 모든 Step 클래스 테스트
    results = test_step_initialization()
    
    # 3. 상속 체인 분석
    analyze_inheritance_chain()
    
    print("\n" + "=" * 80)
    print("🎯 에러 추적 완료")
    print("=" * 80)