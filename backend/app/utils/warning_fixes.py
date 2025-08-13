# 워닝 해결 패치 - main.py 시작 시 자동 실행
def apply_warning_fixes():
    """main.py에서 자동 실행될 워닝 해결 패치"""
    try:
        # SmartMapper 강제 연동
        from app.ai_pipeline.utils.smart_model_mapper import get_global_smart_mapper
        # from app.ai_pipeline.utils.model_loader import get_global_model_loader  # 이 파일은 비어있음
        from app.ai_pipeline.models.model_loader import CentralModelLoader
        
        smart_mapper = get_global_smart_mapper()
        # model_loader = get_global_model_loader()  # 이 함수는 존재하지 않음
        model_loader = CentralModelLoader() if CentralModelLoader else None
        
        if smart_mapper and model_loader:
            # 누락된 모델들 자동 추가
            missing_models = ["vgg16_warping", "vgg19_warping", "densenet121"]
            
            for model_name in missing_models:
                mapping_result = smart_mapper.get_model_path(model_name)
                if mapping_result and mapping_result.found:
                    if hasattr(model_loader, '_available_models_cache'):
                        model_loader._available_models_cache[model_name] = {
                            "name": model_name,
                            "path": str(mapping_result.actual_path),
                            "checkpoint_path": str(mapping_result.actual_path),
                            "ai_model_info": {"ai_class": mapping_result.ai_class},
                            "size_mb": mapping_result.size_mb,
                            "loaded": False,
                            "device": "mps"
                        }
            
            return True
            
    except Exception as e:
        print(f"⚠️ 자동 워닝 해결 실패: {e}")
        
    return False
