"""
🔥 Model Ensemble Manager
========================

모델 앙상블 관리 시스템

Author: MyCloset AI Team
Date: 2025-08-07
Version: 1.0
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import logging


class ModelEnsembleManager:
    """모델 앙상블 관리자"""
    
    def __init__(self, config):
        self.config = config
        self.ensemble_models = {}
        self.loaded_models = {}  # loaded_models 속성 추가
        self.model_confidences = {}
        self.ensemble_weights = {}
        self.logger = logging.getLogger(__name__)
        
        # MPS 디바이스 감지 및 타입 설정
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.dtype = torch.float32  # MPS에서는 float32 사용
        
        # 앙상블 설정
        self.ensemble_config = {
            'use_weighted_average': True,
            'use_uncertainty_weighting': True,
            'use_quality_weighting': True,
            'min_confidence_threshold': 0.3,
            'max_models_in_ensemble': 5
        }
    
    def load_ensemble_models(self, model_loader) -> bool:
        """앙상블 모델 로딩 (실제 사용 가능한 모델만)"""
        try:
            # 실제 사용 가능한 모델들만 로딩
            available_models = []
            
            # Graphonomy 모델 로딩
            graphonomy_model = self._load_graphonomy_model(model_loader)
            if graphonomy_model:
                self.ensemble_models['graphonomy'] = graphonomy_model
                self.loaded_models['graphonomy'] = graphonomy_model
                self.model_confidences['graphonomy'] = 0.9
                available_models.append('graphonomy')
            
            # U2Net 모델 로딩 (실제로 사용 가능한 모델)
            u2net_model = self._load_u2net_model(model_loader)
            if u2net_model:
                self.ensemble_models['u2net'] = u2net_model
                self.loaded_models['u2net'] = u2net_model
                self.model_confidences['u2net'] = 0.85
                available_models.append('u2net')
            
            # HRNet 모델 로딩 (선택적)
            hrnet_model = self._load_hrnet_model(model_loader)
            if hrnet_model:
                self.ensemble_models['hrnet'] = hrnet_model
                self.loaded_models['hrnet'] = hrnet_model
                self.model_confidences['hrnet'] = 0.85
                available_models.append('hrnet')
            
            # DeepLabV3+ 모델 로딩 (선택적)
            deeplabv3plus_model = self._load_deeplabv3plus_model(model_loader)
            if deeplabv3plus_model:
                self.ensemble_models['deeplabv3plus'] = deeplabv3plus_model
                self.loaded_models['deeplabv3plus'] = deeplabv3plus_model
                self.model_confidences['deeplabv3plus'] = 0.8
                available_models.append('deeplabv3plus')
            
            # Mask2Former 모델 로딩 (선택적)
            mask2former_model = self._load_mask2former_model(model_loader)
            if mask2former_model:
                self.ensemble_models['mask2former'] = mask2former_model
                self.loaded_models['mask2former'] = mask2former_model
                self.model_confidences['mask2former'] = 0.75
                available_models.append('mask2former')
            
            if available_models:
                self.logger.info(f"✅ 앙상블 모델 로딩 완료: {available_models}")
                return True
            else:
                self.logger.warning("⚠️ 사용 가능한 앙상블 모델이 없음")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ 앙상블 모델 로딩 실패: {e}")
            return False
    
    def _load_graphonomy_model(self, model_loader):
        """Graphonomy 모델 로딩"""
        try:
            # ModelLoader를 통해 실제 감지된 모델들 로딩
            available_models = [
                'human_parsing_schp',  # 1173MB 메인 모델
                'graphonomy.pth',      # 기본 Graphonomy
                'exp-schp-201908301523-atr.pth'  # SCHP 모델
            ]
            
            for model_name in available_models:
                try:
                    # ModelLoader의 load_model 메서드 사용
                    if hasattr(model_loader, 'load_model') and callable(model_loader.load_model):
                        model = model_loader.load_model(model_name)
                        if model and hasattr(model, 'get_model_instance'):
                            self.logger.info(f"✅ Graphonomy 모델 로딩 성공: {model_name}")
                            return model.get_model_instance()
                    
                    # 대안: get_model 메서드 사용
                    if hasattr(model_loader, 'get_model') and callable(model_loader.get_model):
                        model = model_loader.get_model(model_name)
                        if model:
                            self.logger.info(f"✅ Graphonomy 모델 로딩 성공: {model_name}")
                            return model
                            
                except Exception as e:
                    self.logger.warning(f"⚠️ Graphonomy 모델 로딩 실패 ({model_name}): {e}")
                    continue
            
            self.logger.warning("⚠️ 사용 가능한 Graphonomy 모델이 없음")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Graphonomy 모델 로딩 실패: {e}")
            return None
    
    def _load_u2net_model(self, model_loader):
        """U2Net 모델 로딩"""
        try:
            # ModelLoader를 통해 U2Net 모델 로딩
            u2net_models = [
                'u2net.pth',
                'u2net_official.pth',
                'cloth_segmentation_sam'  # SAM 모델도 대안으로 사용
            ]
            
            for model_name in u2net_models:
                try:
                    # ModelLoader의 load_model 메서드 사용
                    if hasattr(model_loader, 'load_model') and callable(model_loader.load_model):
                        model = model_loader.load_model(model_name)
                        if model and hasattr(model, 'get_model_instance'):
                            self.logger.info(f"✅ U2Net 모델 로딩 성공: {model_name}")
                            return model.get_model_instance()
                    
                    # 대안: get_model 메서드 사용
                    if hasattr(model_loader, 'get_model') and callable(model_loader.get_model):
                        model = model_loader.get_model(model_name)
                        if model:
                            self.logger.info(f"✅ U2Net 모델 로딩 성공: {model_name}")
                            return model
                            
                except Exception as e:
                    self.logger.warning(f"⚠️ U2Net 모델 로딩 실패 ({model_name}): {e}")
                    continue
            
            self.logger.warning("⚠️ 사용 가능한 U2Net 모델이 없음")
            return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ U2Net 모델 로딩 실패: {e}")
            return None
    
    def _load_hrnet_model(self, model_loader):
        """HRNet 모델 로딩"""
        try:
            if hasattr(model_loader, 'get_model') and callable(model_loader.get_model):
                return model_loader.get_model('hrnet')
            return None
        except Exception as e:
            self.logger.warning(f"⚠️ HRNet 모델 로딩 실패: {e}")
            return None
    
    def _load_deeplabv3plus_model(self, model_loader):
        """DeepLabV3+ 모델 로딩"""
        try:
            # ModelLoader를 통해 DeepLabV3+ 모델 로딩
            deeplab_models = [
                'deeplabv3plus.pth',
                'deeplab_resnet101.pth',
                'fcn_resnet101_ultra.pth'
            ]
            
            for model_name in deeplab_models:
                try:
                    # ModelLoader의 load_model 메서드 사용
                    if hasattr(model_loader, 'load_model') and callable(model_loader.load_model):
                        model = model_loader.load_model(model_name)
                        if model and hasattr(model, 'get_model_instance'):
                            self.logger.info(f"✅ DeepLabV3+ 모델 로딩 성공: {model_name}")
                            return model.get_model_instance()
                    
                    # 대안: get_model 메서드 사용
                    if hasattr(model_loader, 'get_model') and callable(model_loader.get_model):
                        model = model_loader.get_model(model_name)
                        if model:
                            self.logger.info(f"✅ DeepLabV3+ 모델 로딩 성공: {model_name}")
                            return model
                            
                except Exception as e:
                    self.logger.warning(f"⚠️ DeepLabV3+ 모델 로딩 실패 ({model_name}): {e}")
                    continue
            
            self.logger.warning("⚠️ 사용 가능한 DeepLabV3+ 모델이 없음")
            return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ DeepLabV3+ 모델 로딩 실패: {e}")
            return None
    
    def _load_mask2former_model(self, model_loader):
        """Mask2Former 모델 로딩"""
        try:
            if hasattr(model_loader, 'get_model') and callable(model_loader.get_model):
                return model_loader.get_model('mask2former')
            return None
        except Exception as e:
            self.logger.warning(f"⚠️ Mask2Former 모델 로딩 실패: {e}")
            return None
    
    def run_ensemble_inference(self, input_tensor, device='cuda') -> Dict[str, Any]:
        """앙상블 추론 실행 (MPS 타입 일관성 유지)"""
        try:
            if not self.ensemble_models:
                return {'error': 'No ensemble models loaded'}
            
            # 입력 텐서를 MPS 타입으로 통일
            input_tensor = input_tensor.to(device=self.device, dtype=self.dtype)
            
            # 각 모델별 추론 실행
            model_outputs = {}
            model_uncertainties = {}
            
            for model_name, model in self.ensemble_models.items():
                try:
                    # 모델을 MPS 타입으로 통일
                    model = model.to(device=self.device, dtype=self.dtype)
                    model.eval()
                    with torch.no_grad():
                        output = model(input_tensor)
                        
                        # 출력 정규화
                        if isinstance(output, dict):
                            if 'parsing_pred' in output:
                                model_outputs[model_name] = output['parsing_pred']
                            elif 'parsing' in output:
                                model_outputs[model_name] = output['parsing']
                            else:
                                # 첫 번째 텐서 사용
                                for key, value in output.items():
                                    if isinstance(value, torch.Tensor):
                                        model_outputs[model_name] = value
                                        break
                        elif isinstance(output, torch.Tensor):
                            model_outputs[model_name] = output
                        
                        # 불확실성 계산
                        if model_outputs[model_name] is not None:
                            uncertainty = self._calculate_model_uncertainty(model_outputs[model_name])
                            model_uncertainties[model_name] = uncertainty
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ {model_name} 모델 추론 실패: {e}")
                    continue
            
            if not model_outputs:
                return {'error': 'All ensemble models failed'}
            
            # 앙상블 결과 계산
            ensemble_result = self._calculate_ensemble_result(
                model_outputs, model_uncertainties
            )
            
            return ensemble_result
            
        except Exception as e:
            self.logger.error(f"❌ 앙상블 추론 실패: {e}")
            return {'error': str(e)}
    
    def _calculate_model_uncertainty(self, model_output):
        """모델 불확실성 계산"""
        try:
            # 엔트로피 기반 불확실성
            probs = torch.softmax(model_output, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            uncertainty = torch.mean(entropy).item()
            return uncertainty
        except:
            return 0.5  # 기본값
    
    def _calculate_ensemble_result(self, model_outputs, model_uncertainties):
        """앙상블 결과 계산"""
        try:
            # 가중 평균 계산
            weighted_sum = None
            total_weight = 0
            
            for model_name, output in model_outputs.items():
                if output is None:
                    continue
                
                # 가중치 계산
                confidence = self.model_confidences.get(model_name, 0.5)
                uncertainty = model_uncertainties.get(model_name, 0.5)
                
                # 품질 기반 가중치
                quality_weight = self._calculate_ensemble_quality(uncertainty, confidence)
                
                if weighted_sum is None:
                    weighted_sum = output * quality_weight
                else:
                    weighted_sum += output * quality_weight
                
                total_weight += quality_weight
            
            if weighted_sum is not None and total_weight > 0:
                ensemble_output = weighted_sum / total_weight
            else:
                # 폴백: 단순 평균
                outputs_list = [output for output in model_outputs.values() if output is not None]
                if outputs_list:
                    ensemble_output = torch.stack(outputs_list).mean(dim=0)
                else:
                    return {'error': 'No valid model outputs'}
            
            return {
                'ensemble_output': ensemble_output,
                'model_outputs': model_outputs,
                'model_uncertainties': model_uncertainties,
                'model_confidences': self.model_confidences,
                'ensemble_quality': total_weight / len(model_outputs) if model_outputs else 0
            }
            
        except Exception as e:
            self.logger.error(f"❌ 앙상블 결과 계산 실패: {e}")
            return {'error': str(e)}
    
    def _calculate_ensemble_quality(self, uncertainty, confidence) -> float:
        """앙상블 품질 계산"""
        # 불확실성이 낮고 신뢰도가 높을수록 높은 품질
        quality = confidence * (1 - uncertainty)
        return max(0.1, quality)  # 최소 가중치 보장
