# ModelLoader 자동 통합 패치 - main.py에 추가
def auto_integrate_detected_models():
    """main.py 시작 시 자동 실행될 모델 통합 함수"""
    try:
        from app.ai_pipeline.utils.model_loader import get_global_model_loader
        model_loader = get_global_model_loader()
        
        if not model_loader:
            return False
        
        # 하드코딩된 모델 정보 (실제 탐지 결과 기반)
        detected_models = {
            "realvis_xl": {
                "name": "realvis_xl",
                "path": "/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/step_05_cloth_warping/RealVisXL_V4.0.safetensors",
                "checkpoint_path": "/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/step_05_cloth_warping/RealVisXL_V4.0.safetensors",
                "size_mb": 6616.631227493286,
                "ai_model_info": {"ai_class": "RealVisXLModel"},
                "step_class": "ClothWarpingStep",
                "model_type": "processing",
                "loaded": False,
                "device": "mps"
            },
            "vgg19_warping": {
                "name": "vgg19_warping",
                "path": "/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/step_05_cloth_warping/ultra_models/vgg19_warping.pth",
                "checkpoint_path": "/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/step_05_cloth_warping/ultra_models/vgg19_warping.pth",
                "size_mb": 548.0597839355469,
                "ai_model_info": {"ai_class": "RealVGG19Model"},
                "step_class": "ClothWarpingStep",
                "model_type": "warping",
                "loaded": False,
                "device": "mps"
            },
            "vgg16_warping": {
                "name": "vgg16_warping",
                "path": "/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/step_05_cloth_warping/ultra_models/vgg16_warping_ultra.pth",
                "checkpoint_path": "/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/step_05_cloth_warping/ultra_models/vgg16_warping_ultra.pth",
                "size_mb": 527.8031978607178,
                "ai_model_info": {"ai_class": "RealVGG16Model"},
                "step_class": "ClothWarpingStep",
                "model_type": "warping",
                "loaded": False,
                "device": "mps"
            },
        }
        
        # available_models에 추가
        current_models = getattr(model_loader, '_available_models_cache', {})
        current_models.update(detected_models)
        
        # 캐시 업데이트
        if hasattr(model_loader, '_available_models_cache'):
            model_loader._available_models_cache = current_models
        
        # 통합 성공 플래그 설정
        if hasattr(model_loader, '_integration_successful'):
            model_loader._integration_successful = True
        
        print(f"✅ 자동 모델 통합 완료: {len(detected_models)}개")
        return True
        
    except Exception as e:
        print(f"❌ 자동 모델 통합 실패: {e}")
        return False
