"""
Model loading utilities for geometric matching step.
"""

import torch
import torch.nn as nn
import logging
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

# 순환 import 방지를 위해 지연 import 사용
# from ..models import (
#     CompleteAdvancedGeometricMatchingAI,
#     GeometricMatchingModule,
#     OpticalFlowNetwork,
#     KeypointMatchingNetwork
# )

logger = logging.getLogger(__name__)


class GeometricMatchingModelLoader:
    """기하학적 매칭 모델 로더"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_geometric_matching_models(self, step_instance) -> bool:
        """기하학적 매칭 모델들 로딩"""
        try:
            start_time = time.time()
            
            self.logger.info("🔄 기하학적 매칭 모델들 로딩 시작...")
            
            # 고급 AI 모델 로딩
            self._load_advanced_geometric_ai(step_instance)
            
            # GMM 모델 로딩
            self._load_gmm_model(step_instance)
            
            # 광학 흐름 모델 로딩
            self._load_optical_flow_model(step_instance)
            
            # 키포인트 매처 로딩
            self._load_keypoint_matcher(step_instance)
            
            loading_time = time.time() - start_time
            if hasattr(step_instance, 'performance_stats'):
                step_instance.performance_stats['model_loading_time'] = loading_time
            
            if hasattr(step_instance, 'status'):
                step_instance.status.update_status(
                    models_loaded=True,
                    initialization_complete=True
                )
            
            self.logger.info(f"✅ 모델 로딩 완료 (소요시간: {loading_time:.2f}초)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 모델 로딩 실패: {e}")
            if hasattr(step_instance, 'status'):
                step_instance.status.update_status(models_loaded=False)
            return False
    
    def _load_advanced_geometric_ai(self, step_instance) -> Optional[nn.Module]:
        """고급 기하학적 AI 모델 로딩"""
        try:
            self.logger.info("🔄 고급 기하학적 AI 모델 로딩 시작...")
            
            # 순환 import 방지를 위해 지연 import
            from ..models import CompleteAdvancedGeometricMatchingAI
            
            # 모델 생성
            step_instance.advanced_ai_models = {}
            step_instance.advanced_ai_models['complete_advanced'] = CompleteAdvancedGeometricMatchingAI(
                input_nc=6,
                initialize_weights=True
            )
            
            # 디바이스로 이동
            step_instance.advanced_ai_models['complete_advanced'] = step_instance.advanced_ai_models['complete_advanced'].to(step_instance.device)
            step_instance.advanced_ai_models['complete_advanced'].eval()
            
            if hasattr(step_instance, 'status'):
                step_instance.status.update_status(advanced_ai_loaded=True)
            
            self.logger.info("✅ 고급 기하학적 AI 모델 로딩 완료")
            return step_instance.advanced_ai_models['complete_advanced']
            
        except Exception as e:
            self.logger.error(f"❌ 고급 기하학적 AI 모델 로딩 실패: {e}")
            if hasattr(step_instance, 'status'):
                step_instance.status.update_status(advanced_ai_loaded=False)
            return None
    
    def _load_gmm_model(self, step_instance) -> Optional[nn.Module]:
        """GMM 모델 로딩"""
        try:
            self.logger.info("🔄 GMM 모델 로딩 시작...")
            
            # 순환 import 방지를 위해 지연 import
            from ..models import GeometricMatchingModule
            
            # 모델 생성
            step_instance.geometric_matching_models = {}
            step_instance.geometric_matching_models['gmm'] = GeometricMatchingModule(
                input_nc=6,
                output_nc=2,
                num_control_points=20,
                initialize_weights=True
            )
            
            # 디바이스로 이동
            step_instance.geometric_matching_models['gmm'] = step_instance.geometric_matching_models['gmm'].to(step_instance.device)
            step_instance.geometric_matching_models['gmm'].eval()
            
            self.logger.info("✅ GMM 모델 로딩 완료")
            return step_instance.geometric_matching_models['gmm']
            
        except Exception as e:
            self.logger.error(f"❌ GMM 모델 로딩 실패: {e}")
            return None
    
    def _load_optical_flow_model(self, step_instance) -> Optional[nn.Module]:
        """광학 흐름 모델 로딩"""
        try:
            self.logger.info("🔄 광학 흐름 모델 로딩 시작...")
            
            # 순환 import 방지를 위해 지연 import
            from ..models import OpticalFlowNetwork
            
            # 모델 생성
            step_instance.geometric_matching_models['optical_flow'] = OpticalFlowNetwork(
                feature_dim=256,
                hidden_dim=128,
                num_iters=12
            )
            
            # 디바이스로 이동
            step_instance.geometric_matching_models['optical_flow'] = step_instance.geometric_matching_models['optical_flow'].to(step_instance.device)
            step_instance.geometric_matching_models['optical_flow'].eval()
            
            self.logger.info("✅ 광학 흐름 모델 로딩 완료")
            return step_instance.geometric_matching_models['optical_flow']
            
        except Exception as e:
            self.logger.error(f"❌ 광학 흐름 모델 로딩 실패: {e}")
            return None
    
    def _load_keypoint_matcher(self, step_instance) -> Optional[nn.Module]:
        """키포인트 매처 로딩"""
        try:
            self.logger.info("🔄 키포인트 매처 로딩 시작...")
            
            # 순환 import 방지를 위해 지연 import
            from ..models import KeypointMatchingNetwork
            
            # 모델 생성
            step_instance.geometric_matching_models['keypoint_matcher'] = KeypointMatchingNetwork(
                num_keypoints=20,
                feature_dim=256
            )
            
            # 디바이스로 이동
            step_instance.geometric_matching_models['keypoint_matcher'] = step_instance.geometric_matching_models['keypoint_matcher'].to(step_instance.device)
            step_instance.geometric_matching_models['keypoint_matcher'].eval()
            
            self.logger.info("✅ 키포인트 매처 로딩 완료")
            return step_instance.geometric_matching_models['keypoint_matcher']
            
        except Exception as e:
            self.logger.error(f"❌ 키포인트 매처 로딩 실패: {e}")
            return None
    
    def load_pretrained_weights(self, step_instance, model_loader, checkpoint_name: str):
        """사전 훈련된 가중치 로딩"""
        try:
            self.logger.info(f"🔄 사전 훈련된 가중치 로딩: {checkpoint_name}")
            
            # 체크포인트 파일 찾기
            checkpoint_path = step_instance.model_path_mapper.find_model_file(checkpoint_name)
            if not checkpoint_path:
                self.logger.warning(f"⚠️ 체크포인트 파일을 찾을 수 없음: {checkpoint_name}")
                return False
            
            # 가중치 로딩
            checkpoint = torch.load(checkpoint_path, map_location=step_instance.device)
            
            # 모델에 가중치 적용
            for model_name, model in step_instance.geometric_matching_models.items():
                if model_name in checkpoint:
                    model.load_state_dict(checkpoint[model_name])
                    self.logger.info(f"✅ {model_name} 가중치 로딩 완료")
            
            for model_name, model in step_instance.advanced_ai_models.items():
                if model_name in checkpoint:
                    model.load_state_dict(checkpoint[model_name])
                    self.logger.info(f"✅ {model_name} 가중치 로딩 완료")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 사전 훈련된 가중치 로딩 실패: {e}")
            return False
