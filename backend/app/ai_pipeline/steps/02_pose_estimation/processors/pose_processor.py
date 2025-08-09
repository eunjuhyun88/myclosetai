#!/usr/bin/env python3
"""
🔥 MyCloset AI - Pose Estimation Processor
=========================================

✅ 전처리/후처리 기능 분리
✅ 기존 step.py 기능 보존
✅ 모듈화된 구조 적용
"""

import time
import logging
from app.ai_pipeline.utils.common_imports import (
    np, Image, cv2, torch, TORCH_AVAILABLE, PIL_AVAILABLE, CV2_AVAILABLE,
    Dict, Any, Optional, Tuple, List, Union
)

from ..config import EnhancedPoseConfig

logger = logging.getLogger(__name__)

class PoseProcessor:
    """포즈 추정 전처리/후처리 프로세서"""
    
    def __init__(self, config: EnhancedPoseConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.PoseProcessor")
    
    def preprocess_input(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """입력 데이터 전처리"""
        try:
            # 이미지 추출
            image = input_data.get('image') or input_data.get('person_image')
            
            # 세션에서 이미지 로드 (이미지가 없는 경우)
            if image is None and 'session_id' in input_data:
                session_manager = self._get_service_from_central_hub('session_manager')
                if session_manager and hasattr(session_manager, 'get_session_images_sync'):
                    person_image, _ = session_manager.get_session_images_sync(input_data['session_id'])
                    image = person_image
            
            if image is None:
                return None
            
            # 이미지 전처리
            if hasattr(image, 'convert'):  # PIL Image
                image_np = np.array(image.convert('RGB'))
            elif hasattr(image, 'shape'):  # NumPy array
                image_np = image
            else:
                return None
            
            return {
                'image': image_np,
                'input_size': self.config.input_size,
                'confidence_threshold': self.config.confidence_threshold,
                'device': getattr(self, 'device', 'cpu')
            }
            
        except Exception as e:
            self.logger.error(f"❌ 입력 데이터 전처리 실패: {e}")
            return None
    
    def postprocess_results(self, inference_result: Dict[str, Any], analysis_result: Dict[str, Any], processed_input: Dict[str, Any]) -> Dict[str, Any]:
        """결과 후처리"""
        try:
            # 기본 결과 구조
            result = {
                'success': True,
                'step_name': 'pose_estimation',
                'step_id': 2,
                'processing_time': time.time(),
                'model_used': inference_result.get('model_used', 'unknown'),
                'confidence': inference_result.get('confidence', 0.0),
                'keypoints': inference_result.get('keypoints', []),
                'keypoints_count': len(inference_result.get('keypoints', [])),
                'pose_quality': inference_result.get('pose_quality', 'unknown'),
                'body_proportions': inference_result.get('body_proportions', {}),
                'joint_angles': inference_result.get('joint_angles', {}),
                'skeleton_structure': inference_result.get('skeleton_structure', {}),
                'visualization': inference_result.get('visualization', {}),
                'metadata': {
                    'input_size': processed_input.get('input_size', self.config.input_size),
                    'confidence_threshold': self.config.confidence_threshold,
                    'device_used': processed_input.get('device', 'cpu'),
                    'ensemble_mode': self.config.enable_ensemble,
                    'models_loaded': getattr(self, 'models_loading_status', {})
                }
            }
            
            # 분석 결과 추가
            if analysis_result:
                result['analysis'] = analysis_result
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 결과 후처리 실패: {e}")
            return {'error': f'결과 후처리 실패: {e}'}
    
    def _get_service_from_central_hub(self, service_key: str):
        """Central Hub에서 서비스 조회"""
        try:
            from app.ai_pipeline.utils.common_imports import _get_central_hub_container
            container = _get_central_hub_container()
            if container:
                return container.get_service(service_key)
            return None
        except Exception as e:
            self.logger.debug(f"Central Hub 서비스 조회 실패: {e}")
            return None
