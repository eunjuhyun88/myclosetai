#!/usr/bin/env python3
"""
🔍 Auto Detector가 경로를 정확히 찾아서 ModelLoader에 제대로 연동하는지 확인
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 설정
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

print("🔍 Auto Detector 경로 매핑 정확성 테스트...")
print(f"📁 프로젝트 루트: {project_root}")
print("=" * 60)

def test_step1_auto_detector_finds_models():
    """1단계: Auto Detector가 모델들을 찾는지 확인"""
    
    print("🔍 1단계: Auto Detector 모델 탐지 테스트...")
    
    try:
        from backend.app.ai_pipeline.utils.auto_model_detector import (
            create_real_world_detector
        )
        
        detector = create_real_world_detector(
            enable_pytorch_validation=True,
            max_workers=1  # 안전한 단일 워커
        )
        
        detected_models = detector.detect_all_models(
            force_rescan=True,  # 강제 재스캔
            min_confidence=0.3
        )
        
        if detected_models:
            print(f"✅ Auto Detector 성공: {len(detected_models)}개 모델 탐지")
            
            # 경로 정확성 확인
            print(f"\n📍 탐지된 모델 경로 확인:")
            
            for i, (name, model) in enumerate(detected_models.items(), 1):
                path_exists = model.path.exists()
                path_readable = os.access(model.path, os.R_OK) if path_exists else False
                
                print(f"   {i}. {name}")
                print(f"      📁 경로: {model.path}")
                print(f"      ✅ 존재: {'YES' if path_exists else 'NO'}")
                print(f"      ✅ 읽기: {'YES' if path_readable else 'NO'}")
                print(f"      📊 크기: {model.file_size_mb:.1f}MB")
                print(f"      🎯 Step: {model.step_name}")
                print(f"      🔍 신뢰도: {model.confidence_score:.2f}")
                print(f"      ✅ 검증: {'YES' if model.pytorch_valid else 'NO'}")
                print("")
            
            return detected_models
        else:
            print("❌ Auto Detector 실패: 모델을 찾지 못함")
            return None
            
    except Exception as e:
        print(f"❌ Auto Detector 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_step2_modelloader_integration(detected_models):
    """2단계: 탐지된 모델이 ModelLoader에 제대로 연동되는지 확인"""
    
    print("🔧 2단계: ModelLoader 연동 테스트...")
    
    if not detected_models:
        print("❌ 탐지된 모델이 없어 연동 테스트 불가")
        return False
    
    try:
        from backend.app.ai_pipeline.utils.model_loader import ModelLoader
        
        # ModelLoader 초기화
        loader = ModelLoader()
        print("✅ ModelLoader 초기화 성공")
        
        # 탐지된 모델들을 ModelLoader에 등록 시도
        print(f"\n📝 탐지된 모델들 등록 시도...")
        
        registration_results = {}
        
        for model_name, detected_model in detected_models.items():
            try:
                # ModelLoader 형식으로 변환
                model_config = {
                    'name': model_name,
                    'type': str(detected_model.category.value),
                    'checkpoint_path': str(detected_model.path),
                    'device': 'auto',
                    'pytorch_validated': detected_model.pytorch_valid,
                    'file_size_mb': detected_model.file_size_mb,
                    'confidence_score': detected_model.confidence_score,
                    'step_name': detected_model.step_name,
                    'auto_detected': True
                }
                
                # 등록 시도
                success = loader.register_model(model_name, model_config)
                registration_results[model_name] = {
                    'success': success,
                    'config': model_config,
                    'detected_model': detected_model
                }
                
                status = "✅" if success else "❌"
                print(f"   {status} {model_name}: {'성공' if success else '실패'}")
                
            except Exception as e:
                registration_results[model_name] = {
                    'success': False,
                    'error': str(e),
                    'detected_model': detected_model
                }
                print(f"   ❌ {model_name}: 오류 - {e}")
        
        # 등록 결과 확인
        successful_registrations = sum(1 for r in registration_results.values() if r.get('success', False))
        print(f"\n📊 등록 결과: {successful_registrations}/{len(detected_models)}개 성공")
        
        return registration_results
        
    except Exception as e:
        print(f"❌ ModelLoader 연동 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_step3_step_mapping(registration_results):
    """3단계: Step별 모델 매핑이 제대로 되는지 확인"""
    
    print("🎯 3단계: Step별 모델 매핑 테스트...")
    
    if not registration_results:
        print("❌ 등록 결과가 없어 매핑 테스트 불가")
        return False
    
    try:
        from backend.app.ai_pipeline.utils.model_loader import ModelLoader
        
        loader = ModelLoader()
        
        # Step별 모델 확인
        step_mapping_results = {}
        
        required_steps = [
            'HumanParsingStep',
            'PoseEstimationStep', 
            'ClothSegmentationStep',
            'GeometricMatchingStep',
            'ClothWarpingStep',
            'VirtualFittingStep',
            'PostProcessingStep',
            'QualityAssessmentStep'
        ]
        
        print(f"\n🎯 Step별 모델 매핑 확인:")
        
        for step_name in required_steps:
            try:
                # Step에 해당하는 모델들 찾기
                step_models = []
                
                for model_name, reg_result in registration_results.items():
                    if reg_result.get('success', False):
                        detected_model = reg_result['detected_model']
                        if detected_model.step_name == step_name:
                            step_models.append({
                                'name': model_name,
                                'path': str(detected_model.path),
                                'confidence': detected_model.confidence_score,
                                'validated': detected_model.pytorch_valid
                            })
                
                step_mapping_results[step_name] = step_models
                
                if step_models:
                    print(f"   ✅ {step_name}: {len(step_models)}개 모델")
                    for model in step_models:
                        status = "✅" if model['validated'] else "❓"
                        print(f"      {status} {model['name']} (신뢰도: {model['confidence']:.2f})")
                else:
                    print(f"   ❌ {step_name}: 모델 없음")
                    
            except Exception as e:
                print(f"   ❌ {step_name}: 오류 - {e}")
        
        # 매핑 결과 요약
        covered_steps = len([s for s, models in step_mapping_results.items() if models])
        coverage_percentage = (covered_steps / len(required_steps)) * 100
        
        print(f"\n📊 Step 매핑 결과:")
        print(f"   🎯 커버된 Step: {covered_steps}/{len(required_steps)}개 ({coverage_percentage:.1f}%)")
        
        return step_mapping_results
        
    except Exception as e:
        print(f"❌ Step 매핑 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_step4_actual_model_loading(step_mapping_results):
    """4단계: 실제로 모델 로딩이 가능한지 확인"""
    
    print("🔥 4단계: 실제 모델 로딩 테스트...")
    
    if not step_mapping_results:
        print("❌ Step 매핑 결과가 없어 로딩 테스트 불가")
        return False
    
    try:
        from backend.app.ai_pipeline.utils.model_loader import ModelLoader
        
        loader = ModelLoader()
        
        # 실제 모델 로딩 시도
        loading_results = {}
        
        # 각 Step별로 모델 로딩 시도
        for step_name, models in step_mapping_results.items():
            if models:  # 모델이 있는 Step만
                try:
                    print(f"\n🔄 {step_name} 모델 로딩 시도...")
                    
                    # 첫 번째 모델로 로딩 시도
                    first_model = models[0]
                    model_name = first_model['name']
                    
                    # 실제 로딩 시도 (안전하게)
                    try:
                        loaded_model = loader.get_model(model_name)
                        if loaded_model:
                            loading_results[step_name] = {
                                'success': True,
                                'model_name': model_name,
                                'model_type': type(loaded_model).__name__
                            }
                            print(f"   ✅ {step_name}: {model_name} 로딩 성공")
                        else:
                            loading_results[step_name] = {
                                'success': False,
                                'model_name': model_name,
                                'error': 'None 반환됨'
                            }
                            print(f"   ❌ {step_name}: {model_name} 로딩 실패 (None)")
                            
                    except Exception as load_error:
                        loading_results[step_name] = {
                            'success': False,
                            'model_name': model_name,
                            'error': str(load_error)
                        }
                        print(f"   ❌ {step_name}: {model_name} 로딩 실패 - {load_error}")
                        
                except Exception as e:
                    loading_results[step_name] = {
                        'success': False,
                        'error': f"전체 실패: {e}"
                    }
                    print(f"   ❌ {step_name}: 전체 실패 - {e}")
        
        # 로딩 결과 요약
        successful_loads = sum(1 for r in loading_results.values() if r.get('success', False))
        total_attempts = len(loading_results)
        
        print(f"\n📊 모델 로딩 결과:")
        print(f"   🔥 성공한 로딩: {successful_loads}/{total_attempts}개")
        
        if successful_loads > 0:
            print(f"   ✅ 성공한 Step들:")
            for step_name, result in loading_results.items():
                if result.get('success', False):
                    print(f"      • {step_name}: {result['model_name']}")
        
        if successful_loads < total_attempts:
            print(f"   ❌ 실패한 Step들:")
            for step_name, result in loading_results.items():
                if not result.get('success', False):
                    print(f"      • {step_name}: {result.get('error', '알 수 없는 오류')}")
        
        return loading_results
        
    except Exception as e:
        print(f"❌ 모델 로딩 테스트 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """메인 테스트 함수"""
    
    print("🔍 Auto Detector → ModelLoader 완전 연동 테스트 시작...")
    print("=" * 60)
    
    # 1단계: Auto Detector 모델 탐지
    detected_models = test_step1_auto_detector_finds_models()
    
    if not detected_models:
        print("\n❌ 1단계 실패: Auto Detector가 모델을 찾지 못했습니다")
        return False
    
    # 2단계: ModelLoader 연동
    registration_results = test_step2_modelloader_integration(detected_models)
    
    if not registration_results:
        print("\n❌ 2단계 실패: ModelLoader 연동이 안됩니다")
        return False
    
    # 3단계: Step 매핑
    step_mapping_results = test_step3_step_mapping(registration_results)
    
    if not step_mapping_results:
        print("\n❌ 3단계 실패: Step 매핑이 안됩니다")
        return False
    
    # 4단계: 실제 모델 로딩
    loading_results = test_step4_actual_model_loading(step_mapping_results)
    
    # 최종 결과
    print("\n" + "=" * 60)
    print("🎯 최종 테스트 결과:")
    
    if loading_results:
        successful_loads = sum(1 for r in loading_results.values() if r.get('success', False))
        
        if successful_loads > 0:
            print(f"✅ Auto Detector → ModelLoader 연동 성공!")
            print(f"   📦 탐지된 모델: {len(detected_models)}개")
            print(f"   🔧 등록된 모델: {sum(1 for r in registration_results.values() if r.get('success', False))}개")
            print(f"   🎯 커버된 Step: {len([s for s, models in step_mapping_results.items() if models])}개")
            print(f"   🔥 로딩 가능한 모델: {successful_loads}개")
            
            print(f"\n🚀 결론: Auto Detector가 경로를 정확히 찾아서 ModelLoader에 제대로 연동됩니다!")
            return True
        else:
            print(f"⚠️ Auto Detector는 모델을 찾지만 ModelLoader 로딩에 문제가 있습니다")
            return False
    else:
        print(f"❌ Auto Detector → ModelLoader 연동에 문제가 있습니다")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🎉 테스트 완료: Auto Detector가 제대로 작동합니다!")
    else:
        print(f"\n💡 문제 해결이 필요합니다.")