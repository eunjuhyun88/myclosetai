"""
포즈 추정 설정 관리
"""
from typing import Dict, Any, Optional
from .types import EnhancedPoseConfig, PoseModel, PoseQuality
from .constants import *


class PoseEstimationConfigManager:
    """포즈 추정 설정 관리자"""
    
    def __init__(self):
        self.default_config = EnhancedPoseConfig()
        self.custom_configs = {}
    
    def get_default_config(self) -> EnhancedPoseConfig:
        """기본 설정 반환"""
        return self.default_config
    
    def create_config(self, **kwargs) -> EnhancedPoseConfig:
        """새로운 설정 생성"""
        return EnhancedPoseConfig(**kwargs)
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """모델별 설정 반환"""
        model_configs = {
            'hrnet': {
                'input_size': (512, 512),
                'confidence_threshold': 0.7,
                'enable_subpixel_accuracy': True,
                'use_fp16': True
            },
            'openpose': {
                'input_size': (368, 368),
                'confidence_threshold': 0.6,
                'enable_subpixel_accuracy': True,
                'use_fp16': False
            },
            'yolo_pose': {
                'input_size': (640, 640),
                'confidence_threshold': 0.5,
                'enable_subpixel_accuracy': False,
                'use_fp16': True
            },
            'mediapipe': {
                'input_size': (640, 640),
                'confidence_threshold': 0.5,
                'enable_subpixel_accuracy': False,
                'use_fp16': False
            }
        }
        return model_configs.get(model_name, {})
    
    def get_ensemble_config(self) -> Dict[str, Any]:
        """앙상블 설정 반환"""
        return {
            'enable_ensemble': True,
            'ensemble_models': ['hrnet', 'openpose', 'yolo_pose', 'mediapipe'],
            'ensemble_method': 'weighted_average',
            'ensemble_confidence_threshold': 0.8,
            'model_weights': MODEL_WEIGHTS
        }
    
    def validate_config(self, config: EnhancedPoseConfig) -> bool:
        """설정 유효성 검증"""
        try:
            # 기본 검증
            if not isinstance(config.input_size, tuple) or len(config.input_size) != 2:
                return False
            
            if not (0.0 <= config.confidence_threshold <= 1.0):
                return False
            
            if not (0.0 <= config.ensemble_confidence_threshold <= 1.0):
                return False
            
            # 앙상블 설정 검증
            if config.enable_ensemble:
                if not config.ensemble_models:
                    return False
                
                valid_methods = ['simple_average', 'weighted_average', 'confidence_weighted']
                if config.ensemble_method not in valid_methods:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def merge_configs(self, base_config: EnhancedPoseConfig, **updates) -> EnhancedPoseConfig:
        """설정 병합"""
        config_dict = {
            'method': base_config.method,
            'quality_level': base_config.quality_level,
            'input_size': base_config.input_size,
            'enable_ensemble': base_config.enable_ensemble,
            'ensemble_models': base_config.ensemble_models,
            'ensemble_method': base_config.ensemble_method,
            'ensemble_confidence_threshold': base_config.ensemble_confidence_threshold,
            'enable_uncertainty_quantification': base_config.enable_uncertainty_quantification,
            'enable_confidence_calibration': base_config.enable_confidence_calibration,
            'ensemble_quality_threshold': base_config.ensemble_quality_threshold,
            'enable_subpixel_accuracy': base_config.enable_subpixel_accuracy,
            'enable_joint_angle_calculation': base_config.enable_joint_angle_calculation,
            'enable_body_proportion_analysis': base_config.enable_body_proportion_analysis,
            'enable_pose_quality_assessment': base_config.enable_pose_quality_assessment,
            'enable_skeleton_structure_analysis': base_config.enable_skeleton_structure_analysis,
            'enable_virtual_fitting_optimization': base_config.enable_virtual_fitting_optimization,
            'use_fp16': base_config.use_fp16,
            'confidence_threshold': base_config.confidence_threshold,
            'enable_visualization': base_config.enable_visualization,
            'auto_preprocessing': base_config.auto_preprocessing,
            'strict_data_validation': base_config.strict_data_validation,
            'auto_postprocessing': base_config.auto_postprocessing
        }
        
        # 업데이트 적용
        config_dict.update(updates)
        
        return EnhancedPoseConfig(**config_dict)


# 전역 설정 관리자 인스턴스
config_manager = PoseEstimationConfigManager()


def get_default_pose_config() -> EnhancedPoseConfig:
    """기본 포즈 설정 반환"""
    return config_manager.get_default_config()


def create_pose_config(**kwargs) -> EnhancedPoseConfig:
    """포즈 설정 생성"""
    return config_manager.create_config(**kwargs)


def get_model_specific_config(model_name: str) -> Dict[str, Any]:
    """모델별 설정 반환"""
    return config_manager.get_model_config(model_name)
