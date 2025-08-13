"""
포즈 앙상블 매니저
"""
import logging
from typing import Dict, Any, List, Optional

# 상대 임포트 수정
try:
    from ..config.types import EnhancedPoseConfig
except ImportError:
    from app.ai_pipeline.steps.step_02_pose_estimation.config.types import EnhancedPoseConfig

try:
    from .ensemble_system import PoseEnsembleSystem
except ImportError:
    from app.ai_pipeline.steps.step_02_pose_estimation.ensemble.ensemble_system import PoseEnsembleSystem


class PoseEnsembleManager:
    """포즈 앙상블 매니저"""
    
    def __init__(self, config: EnhancedPoseConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.PoseEnsembleManager")
        self.ensemble_system = PoseEnsembleSystem(
            num_keypoints=17,
            ensemble_models=config.ensemble_models
        )
    
    def load_ensemble_models(self, model_loader) -> Dict[str, Any]:
        """앙상블 모델 로딩"""
        try:
            models = {}
            for model_name in self.config.ensemble_models:
                model = model_loader.get_model(model_name)
                if model:
                    models[model_name] = model
                    self.logger.info(f"✅ {model_name} 모델 로딩 완료")
                else:
                    self.logger.warning(f"⚠️ {model_name} 모델 로딩 실패")
            
            return models
            
        except Exception as e:
            self.logger.error(f"❌ 앙상블 모델 로딩 실패: {e}")
            return {}
    
    def run_ensemble_inference(self, image, device='cuda') -> Dict[str, Any]:
        """앙상블 추론 실행"""
        try:
            # 개별 모델 추론
            model_results = {}
            
            for model_name, model in self.ensemble_system.ensemble_models.items():
                try:
                    result = model.detect_poses(image)
                    model_results[model_name] = result
                except Exception as e:
                    self.logger.error(f"❌ {model_name} 추론 실패: {e}")
                    model_results[model_name] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # 앙상블 방법에 따른 결과 통합
            if self.config.ensemble_method == 'weighted_average':
                return self.ensemble_system.ensemble_keypoints(model_results)
            elif self.config.ensemble_method == 'confidence_weighted':
                return self.ensemble_system.confidence_weighted_ensemble(model_results)
            else:
                return self.ensemble_system.ensemble_keypoints(model_results)
                
        except Exception as e:
            self.logger.error(f"❌ 앙상블 추론 실패: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """앙상블 정보 반환"""
        return {
            'ensemble_method': self.config.ensemble_method,
            'ensemble_models': self.config.ensemble_models,
            'enable_ensemble': self.config.enable_ensemble,
            'confidence_threshold': self.config.ensemble_confidence_threshold
        }
