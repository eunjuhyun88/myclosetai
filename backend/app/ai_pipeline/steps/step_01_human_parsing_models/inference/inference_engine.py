"""
🔥 추론 관련 메서드들 - 기존 step.py의 모든 기능 복원 + inference_engines.py 통합
"""
import logging
import time
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

class InferenceEngine:
    """추론 관련 메서드들을 담당하는 클래스 - 기존 step.py의 모든 기능 복원"""
    
    def __init__(self, step_instance):
        self.step = step_instance
        self.logger = logging.getLogger(f"{__name__}.InferenceEngine")
    
    def run_ai_inference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """🔥 M3 Max 최적화 고도화된 AI 앙상블 인체 파싱 추론 시스템"""
        self.logger.info("🚀 M3 Max 최적화 AI 앙상블 인체 파싱 시작")
        
        # 디바이스 설정
        device = 'mps:0' if torch.backends.mps.is_available() else 'cpu'
        device_str = str(device)
        self.step.device = device
        self.step.device_str = device_str
        
        try:
            start_time = time.time()
            
            # 1. 입력 데이터 검증 및 이미지 추출
            image = self.extract_input_image(input_data)
            if image is None:
                raise ValueError("입력 이미지를 찾을 수 없습니다")
            
            # 2. 앙상블 시스템 초기화
            ensemble_results = {}
            model_confidences = {}
            
            # 3. 각 모델별 추론 실행
            for model_name, model in self.step.loaded_models.items():
                try:
                    self.logger.info(f"🔥 {model_name} 모델 추론 시작")
                    
                    # 이미지 전처리
                    processed_input = self.preprocess_image_for_model(image, model_name)
                    
                    # 모델별 안전 추론 실행
                    if model_name == 'graphonomy':
                        result = self.run_graphonomy_safe_inference(processed_input, model, device_str)
                    elif model_name == 'hrnet':
                        result = self.run_hrnet_safe_inference(processed_input, model, device_str)
                    elif model_name == 'deeplabv3plus':
                        result = self.run_deeplabv3plus_safe_inference(processed_input, model, device_str)
                    elif model_name == 'u2net':
                        result = self.run_u2net_safe_inference(processed_input, model, device_str)
                    else:
                        result = self.run_generic_safe_inference(processed_input, model, device_str)
                    
                    # 결과 유효성 검증
                    if result and 'parsing_output' in result and result['parsing_output'] is not None:
                        ensemble_results[model_name] = result['parsing_output']
                        
                        # 신뢰도 계산
                        confidence = result.get('confidence', 0.8)
                        if isinstance(confidence, torch.Tensor):
                            confidence = self.step.utils.safe_tensor_to_scalar(confidence)
                        elif isinstance(confidence, (list, tuple)):
                            confidence = float(confidence[0]) if confidence else 0.8
                        else:
                            confidence = float(confidence)
                        
                        # NaN 값 방지
                        if not (confidence > 0 and confidence <= 1):
                            confidence = 0.8
                        
                        model_confidences[model_name] = confidence
                        self.logger.info(f"✅ {model_name} 모델 추론 완료 (신뢰도: {confidence:.3f})")
                    else:
                        self.logger.warning(f"⚠️ {model_name} 모델 결과가 유효하지 않습니다")
                        continue
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ {model_name} 모델 추론 실패: {e}")
                    continue
            
            # 4. 앙상블 융합 실행
            if len(ensemble_results) >= 2:
                self.logger.info("🔥 고급 앙상블 융합 시스템 실행")
                
                try:
                    # 모델 출력들을 텐서로 변환
                    model_outputs_list = []
                    for model_name, output in ensemble_results.items():
                        if isinstance(output, dict):
                            if 'parsing_output' in output:
                                model_outputs_list.append(output['parsing_output'])
                            else:
                                # 첫 번째 텐서 값 찾기
                                for key, value in output.items():
                                    if isinstance(value, torch.Tensor):
                                        model_outputs_list.append(value)
                                        break
                        else:
                            model_outputs_list.append(output)
                    
                    # 각 모델 출력의 채널 수를 20개로 통일
                    standardized_outputs = []
                    for output in model_outputs_list:
                        if hasattr(output, 'device') and str(output.device).startswith('mps'):
                            if output.dtype != torch.float32:
                                output = output.to(torch.float32)
                        else:
                            output = output.to(torch.float32)
                        
                        if output.shape[1] != 20:
                            if output.shape[1] > 20:
                                output = output[:, :20, :, :]
                            else:
                                # 채널 수가 부족한 경우 패딩
                                padding = torch.zeros(output.shape[0], 20 - output.shape[1], output.shape[2], output.shape[3], device=output.device)
                                output = torch.cat([output, padding], dim=1)
                        
                        standardized_outputs.append(output)
                    
                    # 앙상블 융합
                    if standardized_outputs:
                        ensemble_output = torch.stack(standardized_outputs, dim=0)
                        ensemble_output = torch.mean(ensemble_output, dim=0, keepdim=True)
                        
                        # 신뢰도 가중 평균
                        if model_confidences:
                            weights = torch.tensor([model_confidences.get(name, 0.5) for name in ensemble_results.keys()], device=ensemble_output.device)
                            weights = weights / weights.sum()
                            
                            weighted_output = torch.zeros_like(ensemble_output)
                            for i, (name, output) in enumerate(ensemble_results.items()):
                                if i < len(weights):
                                    weighted_output += weights[i] * standardized_outputs[i]
                            
                            final_output = weighted_output
                        else:
                            final_output = ensemble_output
                        
                        # 결과 생성
                        result = {
                            'parsing_output': final_output,
                            'confidence': sum(model_confidences.values()) / len(model_confidences) if model_confidences else 0.8,
                            'model_used': 'ensemble',
                            'ensemble_results': ensemble_results,
                            'model_confidences': model_confidences
                        }
                    else:
                        raise ValueError("앙상블 출력이 생성되지 않았습니다")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ 앙상블 융합 실패: {e}")
                    # 단일 모델 결과 사용
                    if ensemble_results:
                        model_name = list(ensemble_results.keys())[0]
                        result = {
                            'parsing_output': ensemble_results[model_name],
                            'confidence': model_confidences.get(model_name, 0.8),
                            'model_used': model_name,
                            'ensemble_results': ensemble_results,
                            'model_confidences': model_confidences
                        }
                    else:
                        raise ValueError("유효한 모델 결과가 없습니다")
            
            elif len(ensemble_results) == 1:
                # 단일 모델 결과
                model_name = list(ensemble_results.keys())[0]
                result = {
                    'parsing_output': ensemble_results[model_name],
                    'confidence': model_confidences.get(model_name, 0.8),
                    'model_used': model_name,
                    'ensemble_results': ensemble_results,
                    'model_confidences': model_confidences
                }
            else:
                raise ValueError("유효한 모델 결과가 없습니다")
            
            # 5. 후처리
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            result['success'] = True
            
            self.logger.info(f"✅ AI 추론 완료 (처리시간: {processing_time:.2f}초)")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ AI 추론 실패: {e}")
            return self.create_error_response(str(e))
    
    def extract_input_image(self, input_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """입력 데이터에서 이미지 추출"""
        return self.step.utils.extract_input_image(input_data)
    
    def preprocess_image_for_model(self, image: np.ndarray, model_name: str) -> torch.Tensor:
        """모델용 이미지 전처리"""
        return self.step.utils.preprocess_image_for_model(image, model_name)
    
    def run_graphonomy_safe_inference(self, input_tensor: torch.Tensor, model: nn.Module, device: str) -> Dict[str, Any]:
        """🔥 Enhanced Graphonomy 모델 안전 추론 - 새로 구현한 고급 모듈들과 통합"""
        try:
            # 모델을 디바이스로 이동
            model = self.prepare_model_for_inference(model, device)
            model.eval()
            
            with torch.no_grad():
                # 🔥 새로 구현한 고급 모듈들과 통합된 출력 처리
                output = model(input_tensor)
                
                # 새로운 출력 구조 처리
                if isinstance(output, dict):
                    # 🔥 완전한 논문 기반 신경망 구조의 출력 처리
                    if 'parsing' in output:
                        parsing_output = output['parsing']
                    elif 'parsing_pred' in output:
                        parsing_output = output['parsing_pred']
                    else:
                        parsing_output = output.get('final_predictions', None)
                    
                    # 경계 맵 처리
                    boundary_maps = output.get('boundary_maps', None)
                    
                    # 정제 히스토리 처리
                    refinement_history = output.get('refinement_history', None)
                    
                    # 어텐션 가중치 처리
                    attention_weights = output.get('attention_weights', None)
                    
                    # FPN 특징 처리
                    fpn_features = output.get('fpn_features', None)
                    
                    # 융합 특징 처리
                    fused_features = output.get('fused_features', None)
                    
                    # 신뢰도 계산 (새로운 구조 기반)
                    confidence = self._calculate_enhanced_confidence(
                        parsing_output, boundary_maps, refinement_history
                    )
                    
                    return {
                        'parsing_output': parsing_output,
                        'confidence': confidence,
                        'boundary_maps': boundary_maps,
                        'refinement_history': refinement_history,
                        'attention_weights': attention_weights,
                        'fpn_features': fpn_features,
                        'fused_features': fused_features,
                        'model_type': 'enhanced_graphonomy'
                    }
                else:
                    # 기존 출력 구조 처리 (하위 호환성)
                    return self._extract_parsing_from_output(output, device)
                    
        except Exception as e:
            self.logger.error(f"Enhanced Graphonomy 추론 실패: {e}")
            return self.create_error_response(f"Enhanced Graphonomy 추론 실패: {e}")
    
    def run_hrnet_safe_inference(self, input_tensor: torch.Tensor, model: nn.Module, device: str) -> Dict[str, Any]:
        """HRNet 안전 추론"""
        try:
            model = self.prepare_model_for_inference(model, device)
            model.eval()
            
            with torch.no_grad():
                output = model(input_tensor)
                
                # 출력 처리
                if isinstance(output, dict):
                    parsing_output = output.get('parsing_pred', output.get('parsing_output', output.get('output')))
                    confidence = output.get('confidence', 0.8)
                else:
                    parsing_output = output
                    confidence = 0.8
                
                # 텐서 정규화
                if isinstance(parsing_output, torch.Tensor):
                    if parsing_output.dim() == 4:
                        parsing_output = parsing_output.squeeze(0)
                    
                    # 소프트맥스 적용
                    if parsing_output.shape[0] > 1:
                        parsing_output = F.softmax(parsing_output, dim=0)
                
                return {
                    'parsing_output': parsing_output,
                    'confidence': confidence,
                    'model_used': 'hrnet'
                }
                
        except Exception as e:
            self.logger.warning(f"⚠️ HRNet 추론 실패: {e}")
            return self.create_error_response(f"HRNet 추론 실패: {e}")
    
    def run_deeplabv3plus_safe_inference(self, input_tensor: torch.Tensor, model: nn.Module, device: str) -> Dict[str, Any]:
        """🔥 Enhanced DeepLabV3+ 모델 안전 추론 - 새로 구현한 고급 모듈들과 통합"""
        try:
            # 모델을 디바이스로 이동
            model = self.prepare_model_for_inference(model, device)
            model.eval()
            
            with torch.no_grad():
                # 🔥 새로 구현한 고급 모듈들과 통합된 출력 처리
                output = model(input_tensor)
                
                # 새로운 출력 구조 처리
                if isinstance(output, dict):
                    # 🔥 완전한 논문 기반 신경망 구조의 출력 처리
                    if 'parsing' in output:
                        parsing_output = output['parsing']
                    elif 'parsing_pred' in output:
                        parsing_output = output['parsing_pred']
                    else:
                        parsing_output = output.get('final_predictions', None)
                    
                    # 경계 맵 처리
                    boundary_maps = output.get('boundary_maps', None)
                    
                    # 정제 히스토리 처리
                    refinement_history = output.get('refinement_history', None)
                    
                    # FPN 특징 처리
                    fpn_features = output.get('fpn_features', None)
                    
                    # 백본 특징 처리
                    backbone_features = output.get('backbone_features', None)
                    
                    # ASPP 특징 처리
                    aspp_features = output.get('aspp_features', None)
                    
                    # 신뢰도 계산 (새로운 구조 기반)
                    confidence = self._calculate_enhanced_confidence(
                        parsing_output, boundary_maps, refinement_history
                    )
                    
                    return {
                        'parsing_output': parsing_output,
                        'confidence': confidence,
                        'boundary_maps': boundary_maps,
                        'refinement_history': refinement_history,
                        'fpn_features': fpn_features,
                        'backbone_features': backbone_features,
                        'aspp_features': aspp_features,
                        'model_type': 'enhanced_deeplabv3plus'
                    }
                else:
                    # 기존 출력 구조 처리 (하위 호환성)
                    return self._extract_parsing_from_output(output, device)
                    
        except Exception as e:
            self.logger.error(f"Enhanced DeepLabV3+ 추론 실패: {e}")
            return self.create_error_response(f"Enhanced DeepLabV3+ 추론 실패: {e}")
    
    def run_u2net_safe_inference(self, input_tensor: torch.Tensor, model: nn.Module, device: str) -> Dict[str, Any]:
        """🔥 Enhanced U2Net 모델 안전 추론 - 새로 구현한 고급 모듈들과 통합"""
        try:
            # 모델을 디바이스로 이동
            model = self.prepare_model_for_inference(model, device)
            model.eval()
            
            with torch.no_grad():
                # 🔥 새로 구현한 고급 모듈들과 통합된 출력 처리
                output = model(input_tensor)
                
                # 새로운 출력 구조 처리
                if isinstance(output, dict):
                    # 🔥 완전한 논문 기반 신경망 구조의 출력 처리
                    if 'parsing' in output:
                        parsing_output = output['parsing']
                    elif 'parsing_pred' in output:
                        parsing_output = output['parsing_pred']
                    else:
                        parsing_output = output.get('final_predictions', None)
                    
                    # 경계 맵 처리
                    boundary_maps = output.get('boundary_maps', None)
                    
                    # 정제 히스토리 처리
                    refinement_history = output.get('refinement_history', None)
                    
                    # FPN 특징 처리
                    fpn_features = output.get('fpn_features', None)
                    
                    # 인코더/디코더 특징 처리
                    encoder_features = output.get('encoder_features', None)
                    decoder_features = output.get('decoder_features', None)
                    
                    # 신뢰도 계산 (새로운 구조 기반)
                    confidence = self._calculate_enhanced_confidence(
                        parsing_output, boundary_maps, refinement_history
                    )
                    
                    return {
                        'parsing_output': parsing_output,
                        'confidence': confidence,
                        'boundary_maps': boundary_maps,
                        'refinement_history': refinement_history,
                        'fpn_features': fpn_features,
                        'encoder_features': encoder_features,
                        'decoder_features': decoder_features,
                        'model_type': 'enhanced_u2net'
                    }
                else:
                    # 기존 출력 구조 처리 (하위 호환성)
                    return self._extract_parsing_from_output(output, device)
                    
        except Exception as e:
            self.logger.error(f"Enhanced U2Net 추론 실패: {e}")
            return self.create_error_response(f"Enhanced U2Net 추론 실패: {e}")
    
    def run_generic_safe_inference(self, input_tensor: torch.Tensor, model: nn.Module, device: str) -> Dict[str, Any]:
        """일반 모델 안전 추론"""
        try:
            model = self.prepare_model_for_inference(model, device)
            model.eval()
            
            with torch.no_grad():
                output = model(input_tensor)
                
                # 출력 처리
                if isinstance(output, dict):
                    parsing_output = output.get('parsing_pred', output.get('parsing_output', output.get('output')))
                    confidence = output.get('confidence', 0.8)
                else:
                    parsing_output = output
                    confidence = 0.8
                
                # 텐서 정규화
                if isinstance(parsing_output, torch.Tensor):
                    if parsing_output.dim() == 4:
                        parsing_output = parsing_output.squeeze(0)
                    
                    # 소프트맥스 적용
                    if parsing_output.shape[0] > 1:
                        parsing_output = F.softmax(parsing_output, dim=0)
                
                return {
                    'parsing_output': parsing_output,
                    'confidence': confidence,
                    'model_used': 'generic'
                }
                
        except Exception as e:
            self.logger.warning(f"⚠️ 일반 모델 추론 실패: {e}")
            return self.create_error_response(f"일반 모델 추론 실패: {e}")
    
    def prepare_model_for_inference(self, model: nn.Module, device_str: str) -> nn.Module:
        """추론을 위한 모델 준비"""
        try:
            if not isinstance(model, nn.Module):
                self.logger.warning("⚠️ 모델이 nn.Module이 아닙니다")
                return model
            
            # 디바이스 이동
            if device_str.startswith('mps') and torch.backends.mps.is_available():
                model = model.to('mps')
            elif device_str.startswith('cuda') and torch.cuda.is_available():
                model = model.to('cuda')
            else:
                model = model.to('cpu')
            
            return model
        except Exception as e:
            self.logger.warning(f"⚠️ 모델 준비 실패: {e}")
            return model
    
    def create_error_response(self, error_message: str) -> Dict[str, Any]:
        """에러 응답 생성"""
        return {
            'success': False,
            'error': error_message,
            'parsing_output': torch.zeros((20, 512, 512)),
            'confidence': 0.0,
            'model_used': 'none',
            'processing_time': 0.0
        }
    
    # 🔥 inference_engines.py에서 추가된 메서드들
    
    def _safe_tensor_to_scalar(self, tensor_value):
        """텐서를 스칼라로 안전하게 변환"""
        if isinstance(tensor_value, torch.Tensor):
            if tensor_value.numel() == 1:
                return tensor_value.item()
            else:
                return tensor_value.mean().item()
        elif isinstance(tensor_value, (int, float)):
            return float(tensor_value)
        else:
            return 0.8  # 기본값
    
    def _extract_actual_model(self, model) -> Optional[nn.Module]:
        """실제 모델 인스턴스 추출 (표준화)"""
        try:
            if hasattr(model, 'model_instance') and model.model_instance is not None:
                return model.model_instance
            elif hasattr(model, 'get_model_instance'):
                return model.get_model_instance()
            elif callable(model):
                return model
            else:
                return None
        except Exception as e:
            self.logger.warning(f"⚠️ 모델 인스턴스 추출 실패: {e}")
            return None
    
    def _create_standard_output(self, device) -> Dict[str, Any]:
        """표준 출력 생성"""
        return {
            'parsing_pred': torch.zeros((1, 20, 512, 512), device=device),
            'parsing_output': torch.zeros((1, 20, 512, 512), device=device),
            'confidence': 0.5,
            'edge_output': None
        }
    
    def _extract_parsing_from_output(self, output, device) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """모델 출력에서 파싱 결과 추출"""
        try:
            if output is None:
                self.logger.warning("⚠️ AI 모델 출력이 None입니다.")
                return torch.zeros((1, 20, 512, 512), device=device), None
            
            if isinstance(output, dict):
                parsing_keys = ['parsing', 'parsing_pred', 'output', 'parsing_output', 'logits', 'pred', 'prediction']
                parsing_tensor = None
                confidence_tensor = None
                
                for key in parsing_keys:
                    if key in output and output[key] is not None:
                        if isinstance(output[key], torch.Tensor):
                            parsing_tensor = output[key]
                            break
                        elif isinstance(output[key], (list, tuple)) and len(output[key]) > 0:
                            if isinstance(output[key][0], torch.Tensor):
                                parsing_tensor = output[key][0]
                                break
                
                return parsing_tensor, confidence_tensor
            else:
                return output, None
                
        except Exception as e:
            self.logger.warning(f"⚠️ 파싱 결과 추출 실패: {e}")
            return torch.zeros((1, 20, 512, 512), device=device), None
    
    def _standardize_channels(self, tensor: torch.Tensor, target_channels: int = 20) -> torch.Tensor:
        """텐서 채널 수 표준화"""
        try:
            if tensor.dim() == 3:
                current_channels = tensor.shape[0]
            elif tensor.dim() == 4:
                current_channels = tensor.shape[1]
            else:
                return tensor
            
            if current_channels == target_channels:
                return tensor
            elif current_channels > target_channels:
                if tensor.dim() == 3:
                    return tensor[:target_channels]
                else:
                    return tensor[:, :target_channels]
            else:
                # 채널 수가 부족한 경우 패딩
                if tensor.dim() == 3:
                    padding = torch.zeros(target_channels - current_channels, tensor.shape[1], tensor.shape[2], device=tensor.device)
                    return torch.cat([tensor, padding], dim=0)
                else:
                    padding = torch.zeros(tensor.shape[0], target_channels - current_channels, tensor.shape[2], tensor.shape[3], device=tensor.device)
                    return torch.cat([tensor, padding], dim=1)
                    
        except Exception as e:
            self.logger.warning(f"⚠️ 채널 수 표준화 실패: {e}")
            return tensor

    def _calculate_enhanced_confidence(self, parsing_output, boundary_maps, refinement_history):
        """🔥 새로 구현한 고급 모듈들의 출력을 기반으로 신뢰도를 계산"""
        try:
            if parsing_output is None:
                return 0.8
            
            # 기본 신뢰도 계산
            if isinstance(parsing_output, torch.Tensor):
                # 텐서의 통계를 기반으로 신뢰도 계산
                with torch.no_grad():
                    # 소프트맥스 적용
                    if parsing_output.dim() > 1:
                        probs = F.softmax(parsing_output, dim=1)
                        # 최대 확률값을 신뢰도로 사용
                        confidence = torch.max(probs).item()
                    else:
                        confidence = torch.sigmoid(parsing_output).mean().item()
            else:
                confidence = 0.8
            
            # 경계 맵이 있으면 신뢰도 향상
            if boundary_maps is not None:
                confidence = min(confidence * 1.1, 0.95)
            
            # 정제 히스토리가 있으면 신뢰도 향상
            if refinement_history is not None:
                confidence = min(confidence * 1.05, 0.95)
            
            # NaN 값 방지
            if not (confidence > 0 and confidence <= 1):
                confidence = 0.8
            
            return confidence
            
        except Exception as e:
            self.logger.warning(f"신뢰도 계산 실패: {e}")
            return 0.8
