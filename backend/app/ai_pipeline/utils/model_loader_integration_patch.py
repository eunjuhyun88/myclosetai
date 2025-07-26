# backend/app/ai_pipeline/utils/model_loader_integration_patch.py
"""
🔥 ModelLoader 통합 패치 - 워닝 완전 제거 v1.0
================================================================================
✅ SmartModelPathMapper 완전 연동
✅ 누락된 모델 워닝 해결 
✅ BaseStepMixin v18.0 완전 호환
✅ 기존 model_loader.py 코드 최소 수정
================================================================================
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from .smart_model_mapper import get_global_smart_mapper, resolve_model_path

logger = logging.getLogger(__name__)

class ModelLoaderIntegrationPatch:
    """ModelLoader 통합 패치 클래스"""
    
    def __init__(self, original_model_loader):
        self.original_loader = original_model_loader
        self.smart_mapper = get_global_smart_mapper()
        self.logger = logging.getLogger(f"{__name__}.IntegrationPatch")
        
        # 워닝 해결 카운터
        self.resolved_warnings = {
            "missing_models": 0,
            "path_corrections": 0,
            "successful_mappings": 0
        }
        
        self.logger.info("🔧 ModelLoader 통합 패치 활성화")
    
    def resolve_missing_model_path(self, model_name: str, **kwargs) -> Optional[str]:
        """🔥 누락된 모델 경로 해결"""
        try:
            # SmartMapper로 경로 해결
            mapping_info = self.smart_mapper.get_model_path(model_name)
            
            if mapping_info and mapping_info.actual_path:
                self.resolved_warnings["missing_models"] += 1
                self.logger.info(f"✅ 누락 모델 해결: {model_name} → {mapping_info.actual_path}")
                return str(mapping_info.actual_path)
            
            # 폴백: 원본 ModelLoader의 available_models 확인
            if hasattr(self.original_loader, 'available_models'):
                available_dict = self.original_loader.available_models
                if model_name in available_dict:
                    model_info = available_dict[model_name]
                    path = model_info.get("checkpoint_path") or model_info.get("path")
                    if path and Path(path).exists():
                        self.resolved_warnings["path_corrections"] += 1
                        return str(path)
            
            self.logger.warning(f"⚠️ 모델 경로 해결 실패: {model_name}")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ 모델 경로 해결 실패 {model_name}: {e}")
            return None
    
    def patch_load_model_method(self):
        """load_model 메서드 패치"""
        original_load_model = self.original_loader.load_model
        
        def patched_load_model(model_name: str, **kwargs):
            try:
                # 원본 메서드 먼저 시도
                result = original_load_model(model_name, **kwargs)
                if result:
                    self.resolved_warnings["successful_mappings"] += 1
                    return result
                
                # 실패 시 SmartMapper로 경로 해결 후 재시도
                resolved_path = self.resolve_missing_model_path(model_name)
                if resolved_path:
                    self.logger.info(f"🔄 패치된 경로로 재시도: {model_name}")
                    
                    # available_models에 임시 추가
                    if hasattr(self.original_loader, '_available_models_cache'):
                        self.original_loader._available_models_cache[model_name] = {
                            "name": model_name,
                            "path": resolved_path,
                            "checkpoint_path": resolved_path,
                            "ai_model_info": {"ai_class": "BaseRealAIModel"},
                            "size_mb": Path(resolved_path).stat().st_size / (1024 * 1024)
                        }
                    
                    return original_load_model(model_name, **kwargs)
                
                return None
                
            except Exception as e:
                self.logger.error(f"❌ 패치된 로드 실패 {model_name}: {e}")
                return None
        
        # 메서드 교체
        self.original_loader.load_model = patched_load_model
        self.logger.info("✅ load_model 메서드 패치 완료")
    
    def patch_load_model_with_fallback_method(self):
        """load_model_with_fallback 메서드 추가"""
        def load_model_with_fallback(model_name: str, **kwargs):
            """누락된 모델에 대한 폴백 처리"""
            try:
                # 기본 로딩 시도
                result = self.original_loader.load_model(model_name, **kwargs)
                if result:
                    return result
                
                self.logger.warning(f"⚠️ 기본 모델 로딩 실패: {model_name}")
                
                # SmartMapper로 경로 해결
                resolved_path = self.resolve_missing_model_path(model_name)
                
                if resolved_path:
                    self.logger.info(f"🔄 폴백 경로로 재시도: {model_name}")
                    
                    # 직접 경로 지정해서 로딩
                    if hasattr(self.original_loader, 'load_model_from_path'):
                        return self.original_loader.load_model_from_path(resolved_path, **kwargs)
                    else:
                        # 대안: available_models 업데이트 후 재시도
                        self._update_available_models_with_path(model_name, resolved_path)
                        return self.original_loader.load_model(model_name, **kwargs)
                else:
                    self.logger.error(f"❌ 모델 로딩 완전 실패: {model_name}")
                    return None
                    
            except Exception as e:
                self.logger.error(f"❌ 폴백 로딩 실패 {model_name}: {e}")
                return None
        
        # 메서드 추가
        self.original_loader.load_model_with_fallback = load_model_with_fallback
        self.logger.info("✅ load_model_with_fallback 메서드 추가 완료")
    
    def _update_available_models_with_path(self, model_name: str, resolved_path: str):
        """available_models에 해결된 경로 업데이트"""
        try:
            if hasattr(self.original_loader, '_available_models_cache'):
                mapping_info = self.smart_mapper.get_model_path(model_name)
                
                model_info = {
                    "name": model_name,
                    "path": resolved_path,
                    "checkpoint_path": resolved_path,
                    "size_mb": Path(resolved_path).stat().st_size / (1024 * 1024),
                    "loaded": False,
                    "device": getattr(self.original_loader, 'device', 'cpu'),
                    "ai_model_info": {
                        "ai_class": mapping_info.ai_class if mapping_info else "BaseRealAIModel",
                        "can_create_ai_model": True,
                        "device_compatible": True
                    },
                    "metadata": {
                        "resolution_source": "smart_mapper",
                        "original_missing": True
                    }
                }
                
                self.original_loader._available_models_cache[model_name] = model_info
                self.logger.info(f"✅ available_models 업데이트: {model_name}")
                
        except Exception as e:
            self.logger.error(f"❌ available_models 업데이트 실패: {e}")
    
    def patch_available_models_property(self):
        """available_models 속성 패치"""
        original_available_models = self.original_loader.available_models
        
        def patched_available_models():
            try:
                # 원본 모델들 가져오기
                available_dict = original_available_models
                if not isinstance(available_dict, dict):
                    available_dict = {}
                
                # SmartMapper에서 추가 모델들 찾기
                for model_name in self.smart_mapper.unified_model_mappings.keys():
                    if model_name not in available_dict:
                        mapping_info = self.smart_mapper.get_model_path(model_name)
                        if mapping_info and mapping_info.actual_path:
                            available_dict[model_name] = {
                                "name": model_name,
                                "path": str(mapping_info.actual_path),
                                "checkpoint_path": str(mapping_info.actual_path),
                                "size_mb": mapping_info.size_mb,
                                "step_class": mapping_info.step_class or "UnknownStep",
                                "ai_model_info": {
                                    "ai_class": mapping_info.ai_class or "BaseRealAIModel"
                                },
                                "metadata": {
                                    "source": "smart_mapper_discovery"
                                }
                            }
                
                return available_dict
                
            except Exception as e:
                self.logger.error(f"❌ available_models 패치 실패: {e}")
                return original_available_models
        
        # 속성을 property로 교체
        if hasattr(self.original_loader.__class__, 'available_models'):
            self.original_loader.__class__.available_models = property(lambda self: patched_available_models())
        else:
            self.original_loader.available_models = patched_available_models()
        
        self.logger.info("✅ available_models 속성 패치 완료")
    
    def apply_all_patches(self):
        """모든 패치 적용"""
        try:
            self.patch_load_model_method()
            self.patch_load_model_with_fallback_method()
            self.patch_available_models_property()
            
            self.logger.info("🎉 모든 ModelLoader 패치 적용 완료")
            self.logger.info(f"📊 해결된 워닝들: {self.resolved_warnings}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 패치 적용 실패: {e}")
            return False
    
    def get_patch_status(self) -> Dict[str, Any]:
        """패치 상태 조회"""
        return {
            "patches_applied": True,
            "resolved_warnings": self.resolved_warnings.copy(),
            "smart_mapper_stats": self.smart_mapper.get_mapping_statistics(),
            "original_loader_type": type(self.original_loader).__name__
        }

# ==============================================
# 🔥 간편 적용 함수들
# ==============================================

def apply_model_loader_patches(model_loader_instance) -> bool:
    """ModelLoader에 패치 적용"""
    try:
        patch_system = ModelLoaderIntegrationPatch(model_loader_instance)
        return patch_system.apply_all_patches()
    except Exception as e:
        logger.error(f"❌ ModelLoader 패치 적용 실패: {e}")
        return False

def resolve_missing_models_globally() -> Dict[str, Any]:
    """전역적으로 누락된 모델들 해결"""
    try:
        smart_mapper = get_global_smart_mapper()
        
        # 공통 누락 모델들
        missing_models = [
            "realvis_xl", "vgg16_warping", "vgg19_warping", "densenet121",
            "post_processing_model", "super_resolution", 
            "clip_vit_large", "quality_assessment"
        ]
        
        resolved_models = {}
        
        for model_name in missing_models:
            mapping_info = smart_mapper.get_model_path(model_name)
            if mapping_info and mapping_info.actual_path:
                resolved_models[model_name] = {
                    "path": str(mapping_info.actual_path),
                    "size_mb": mapping_info.size_mb,
                    "ai_class": mapping_info.ai_class
                }
        
        logger.info(f"✅ 전역 누락 모델 해결: {len(resolved_models)}개")
        return {
            "resolved_count": len(resolved_models),
            "resolved_models": resolved_models,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"❌ 전역 모델 해결 실패: {e}")
        return {"success": False, "error": str(e)}

def create_missing_model_mapping() -> Dict[str, str]:
    """누락된 모델들의 매핑 테이블 생성"""
    try:
        smart_mapper = get_global_smart_mapper()
        mapping_table = {}
        
        for model_name in smart_mapper.unified_model_mappings.keys():
            mapping_info = smart_mapper.get_model_path(model_name)
            if mapping_info and mapping_info.actual_path:
                mapping_table[model_name] = str(mapping_info.actual_path)
        
        return mapping_table
        
    except Exception as e:
        logger.error(f"❌ 매핑 테이블 생성 실패: {e}")
        return {}

# Export
__all__ = [
    'ModelLoaderIntegrationPatch',
    'apply_model_loader_patches', 
    'resolve_missing_models_globally',
    'create_missing_model_mapping'
]

logger.info("🔧 ModelLoader 통합 패치 시스템 로드 완료")
logger.info("🎯 워닝 제거 및 누락 모델 해결 준비 완료")