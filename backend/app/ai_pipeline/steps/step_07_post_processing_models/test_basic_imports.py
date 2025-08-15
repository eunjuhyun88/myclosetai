#!/usr/bin/env python3
"""
Basic Import Test for Post Processing Models

이 스크립트는 후처리 모델들의 기본 import가 정상적으로 작동하는지 테스트합니다.
"""

def test_basic_imports():
    """기본 import 테스트"""
    try:
        print("Testing basic imports...")
        
        # 메인 패키지 import 테스트
        from backend.app.ai_pipeline.steps.step_07_post_processing_models import PostProcessingStep, PostProcessingModelLoader, PostProcessingInferenceEngine
        print("✓ Main package imports successful")
        
        # 설정 import 테스트
        from backend.app.ai_pipeline.steps.step_07_post_processing_models.config import PostProcessingModelConfig, PostProcessingInferenceConfig
        print("✓ Config imports successful")
        
        # 모델 import 테스트
        from backend.app.ai_pipeline.steps.step_07_post_processing_models.models import SwinIRModel, RealESRGANModel, GFPGANModel, CodeFormerModel
        print("✓ Model imports successful")
        
        print("\n🎉 All basic imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_config_creation():
    """설정 생성 테스트"""
    try:
        print("\nTesting configuration creation...")
        
        from backend.app.ai_pipeline.steps.step_07_post_processing_models.config import PostProcessingModelConfig, PostProcessingInferenceConfig
        
        # 모델 설정 생성
        model_config = PostProcessingModelConfig()
        print("✓ Model config created successfully")
        
        # 추론 설정 생성
        inference_config = PostProcessingInferenceConfig()
        print("✓ Inference config created successfully")
        
        # 설정 검증
        if model_config.validate():
            print("✓ Model config validation passed")
        else:
            print("❌ Model config validation failed")
            
        if inference_config.validate():
            print("✓ Inference config validation passed")
        else:
            print("❌ Inference config validation failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Config creation error: {e}")
        return False

def test_model_loader():
    """모델 로더 테스트"""
    try:
        print("\nTesting model loader...")
        
        from backend.app.ai_pipeline.steps.step_07_post_processing_models.post_processing_model_loader import PostProcessingModelLoader
        
        # 모델 로더 생성
        loader = PostProcessingModelLoader()
        print("✓ Model loader created successfully")
        
        # 지원 모델 확인
        supported_models = loader.get_loaded_model_types()
        print(f"✓ Supported models: {loader.supported_models}")
        
        # 메모리 사용량 확인
        memory_info = loader.get_memory_usage()
        print(f"✓ Memory info: {memory_info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loader error: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🚀 Starting Post Processing Models Basic Tests\n")
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Configuration Creation", test_config_creation),
        ("Model Loader", test_model_loader)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        if test_func():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The basic structure is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()
