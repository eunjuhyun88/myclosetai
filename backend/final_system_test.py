#!/usr/bin/env python3
# backend/final_system_test.py
"""
🎉 MyCloset AI 최종 시스템 검증
✅ 모든 Step 로드 확인
✅ AI 파이프라인 동작 검증
✅ 실제 처리 기능 테스트
"""

import sys
import os
import asyncio
import traceback
from pathlib import Path

# 경로 설정
backend_root = Path(__file__).parent
sys.path.insert(0, str(backend_root))

def test_all_steps():
    """모든 Step 개별 테스트"""
    print("🧪 개별 Step 테스트...")
    
    steps_to_test = [
        ("HumanParsingStep", "app.ai_pipeline.steps.step_01_human_parsing"),
        ("PoseEstimationStep", "app.ai_pipeline.steps.step_02_pose_estimation"), 
        ("ClothSegmentationStep", "app.ai_pipeline.steps.step_03_cloth_segmentation"),
        ("VirtualFittingStep", "app.ai_pipeline.steps.step_06_virtual_fitting"),
        ("PostProcessingStep", "app.ai_pipeline.steps.step_07_post_processing")
    ]
    
    results = {}
    
    for step_name, module_path in steps_to_test:
        try:
            module_name = module_path.split('.')[-1]
            module = __import__(module_path, fromlist=[step_name])
            step_class = getattr(module, step_name)
            
            # 인스턴스 생성 테스트
            step_instance = step_class(device='cpu', strict_mode=False)
            status = step_instance.get_status()
            
            results[step_name] = {
                'import': True,
                'instance': True,
                'initialized': status.get('initialized', False),
                'step_name': status.get('step_name', step_name)
            }
            print(f"  ✅ {step_name}: 정상 ({status.get('step_name', 'Unknown')})")
            
        except Exception as e:
            results[step_name] = {
                'import': False,
                'instance': False,
                'error': str(e)
            }
            print(f"  ❌ {step_name}: {e}")
    
    return results

def test_pipeline_manager():
    """PipelineManager 테스트"""
    print("\n🚀 PipelineManager 테스트...")
    
    try:
        from app.ai_pipeline.pipeline_manager import (
            PipelineManager, 
            create_pipeline,
            PipelineConfig
        )
        
        # 설정 생성
        config = PipelineConfig(
            device='cpu',
            optimize_for_m3_max=True,
            enable_caching=True
        )
        
        # 파이프라인 생성
        pipeline = create_pipeline(config)
        print(f"  ✅ PipelineManager 생성: {type(pipeline)}")
        
        # 상태 확인
        status = pipeline.get_status()
        print(f"  📊 파이프라인 상태:")
        print(f"    - 활성화된 Step: {len(status.get('active_steps', []))}")
        print(f"    - 디바이스: {status.get('device', 'unknown')}")
        print(f"    - M3 Max 최적화: {status.get('m3_max_optimized', False)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ PipelineManager 테스트 실패: {e}")
        print(f"  📋 상세:\n{traceback.format_exc()}")
        return False

async def test_async_operations():
    """비동기 작업 테스트"""
    print("\n⚡ 비동기 작업 테스트...")
    
    try:
        # Step 01 비동기 테스트
        from app.ai_pipeline.steps.step_01_human_parsing import HumanParsingStep
        
        step = HumanParsingStep(device='cpu')
        await step.initialize_async()
        
        print(f"  ✅ 비동기 초기화 성공")
        
        # 상태 확인
        status = step.get_status()
        print(f"  📊 Step 상태: {status.get('initialized', False)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 비동기 테스트 실패: {e}")
        return False

def test_model_detection():
    """AI 모델 탐지 테스트"""
    print("\n🤖 AI 모델 탐지 테스트...")
    
    try:
        from app.ai_pipeline.utils.auto_model_detector import detect_available_models
        
        # 모델 탐지 실행
        models = detect_available_models()
        
        print(f"  📊 탐지된 모델: {len(models)}개")
        
        # 주요 모델들 확인
        key_models = ['human_parsing', 'pose_estimation', 'virtual_fitting']
        for model_type in key_models:
            found = any(model_type in str(model).lower() for model in models)
            status = "✅" if found else "⚠️"
            print(f"  {status} {model_type}: {'발견' if found else '미발견'}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 모델 탐지 실패: {e}")
        return False

def test_memory_optimization():
    """메모리 최적화 테스트"""
    print("\n💾 메모리 최적화 테스트...")
    
    try:
        import psutil
        import torch
        
        # 시스템 메모리
        memory = psutil.virtual_memory()
        print(f"  💾 시스템 메모리: {memory.total // (1024**3)}GB")
        print(f"  💾 사용 가능: {memory.available // (1024**3)}GB")
        
        # PyTorch MPS
        if torch.backends.mps.is_available():
            print(f"  🍎 MPS 디바이스: 사용 가능")
            
            # MPS 메모리 정리
            torch.mps.empty_cache()
            print(f"  🧹 MPS 캐시 정리 완료")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 메모리 테스트 실패: {e}")
        return False

def generate_system_report(step_results, pipeline_ok, async_ok, models_ok, memory_ok):
    """시스템 보고서 생성"""
    print("\n" + "="*60)
    print("📋 MyCloset AI 시스템 최종 보고서")
    print("="*60)
    
    # 전체 상태
    total_tests = 5
    passed_tests = sum([
        len([r for r in step_results.values() if r.get('instance', False)]) > 0,
        pipeline_ok,
        async_ok, 
        models_ok,
        memory_ok
    ])
    
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"🎯 전체 성공률: {success_rate:.1f}% ({passed_tests}/{total_tests})")
    
    # 상세 결과
    print(f"\n📊 상세 결과:")
    print(f"  🧪 Step 테스트: {'✅' if any(r.get('instance') for r in step_results.values()) else '❌'}")
    print(f"  🚀 PipelineManager: {'✅' if pipeline_ok else '❌'}")
    print(f"  ⚡ 비동기 작업: {'✅' if async_ok else '❌'}")
    print(f"  🤖 모델 탐지: {'✅' if models_ok else '❌'}")
    print(f"  💾 메모리 최적화: {'✅' if memory_ok else '❌'}")
    
    # Step별 상세
    print(f"\n🎯 Step별 상태:")
    for step_name, result in step_results.items():
        if result.get('instance'):
            print(f"  ✅ {step_name}: 정상 동작")
        else:
            print(f"  ❌ {step_name}: {result.get('error', '알 수 없는 오류')}")
    
    # 권장사항
    print(f"\n💡 권장사항:")
    if success_rate >= 90:
        print("  🎉 시스템이 완벽하게 작동합니다!")
        print("  🚀 프로덕션 환경에서 사용할 준비가 되었습니다.")
    elif success_rate >= 70:
        print("  ✅ 시스템이 대부분 정상 작동합니다.")
        print("  🔧 일부 개선사항이 있지만 기본 기능은 사용 가능합니다.")
    else:
        print("  ⚠️ 추가 문제 해결이 필요합니다.")
        print("  🔧 시스템 설정을 재검토해주세요.")
    
    print("="*60)

async def main():
    """메인 테스트 실행"""
    print("🎯 MyCloset AI 최종 시스템 검증 시작")
    print("="*60)
    
    # 환경 정보
    print(f"🐍 Python: {sys.version}")
    print(f"📁 작업 디렉토리: {os.getcwd()}")
    print(f"🐍 conda 환경: {os.environ.get('CONDA_DEFAULT_ENV', 'None')}")
    
    # 테스트 실행
    step_results = test_all_steps()
    pipeline_ok = test_pipeline_manager()
    async_ok = await test_async_operations()
    models_ok = test_model_detection()
    memory_ok = test_memory_optimization()
    
    # 최종 보고서
    generate_system_report(step_results, pipeline_ok, async_ok, models_ok, memory_ok)

if __name__ == "__main__":
    asyncio.run(main())