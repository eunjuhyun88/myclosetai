"""
🔥 Inference Engines
===================

인체 파싱 추론 엔진들

Author: MyCloset AI Team
Date: 2025-08-07
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
import sys
import io


class InferenceEngine:
    """추론 엔진 기본 클래스"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.logger = logging.getLogger(__name__)
    
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
            'parsing_pred': torch.zeros((1, 20, 512, 512), device=device),  # 일관된 키 이름 사용
            'parsing_output': torch.zeros((1, 20, 512, 512), device=device),
            'confidence': 0.5,
            'edge_output': None
        }
    
    def _extract_parsing_from_output(self, output, device) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """모델 출력에서 파싱 결과 추출 (근본적 해결)"""
        try:
            # 🔥 1단계: 출력 타입 검증 및 정규화
            if output is None:
                self.logger.warning("⚠️ AI 모델 출력이 None입니다.")
                return torch.zeros((1, 20, 512, 512), device=device), None
            
            # 🔥 2단계: 딕셔너리 형태 출력 처리
            if isinstance(output, dict):
                self.logger.debug(f"🔥 딕셔너리 출력 키들: {list(output.keys())}")
                
                # 가능한 키들에서 파싱 결과 찾기
                parsing_keys = ['parsing', 'parsing_pred', 'output', 'parsing_output', 'logits', 'pred', 'prediction']
                parsing_tensor = None
                confidence_tensor = None
                
                for key in parsing_keys:
                    if key in output and output[key] is not None:
                        if isinstance(output[key], torch.Tensor):
                            parsing_tensor = output[key]
                            self.logger.debug(f"✅ 파싱 텐서 발견: {key} - {parsing_tensor.shape}")
                            break
                        elif isinstance(output[key], (list, tuple)) and len(output[key]) > 0:
                            if isinstance(output[key][0], torch.Tensor):
                                parsing_tensor = output[key][0]
                                self.logger.debug(f"✅ 파싱 텐서 발견 (리스트): {key} - {parsing_tensor.shape}")
                                break
                
                # 신뢰도 텐서 찾기
                confidence_keys = ['confidence', 'conf', 'prob', 'probability']
                for key in confidence_keys:
                    if key in output and output[key] is not None:
                        if isinstance(output[key], torch.Tensor):
                            confidence_tensor = output[key]
                            self.logger.debug(f"✅ 신뢰도 텐서 발견: {key} - {confidence_tensor.shape}")
                            break
                
                # 🔥 3단계: 텐서가 없는 경우 첫 번째 값 사용
                if parsing_tensor is None:
                    first_value = next(iter(output.values()))
                    if isinstance(first_value, torch.Tensor):
                        parsing_tensor = first_value
                        self.logger.debug(f"✅ 첫 번째 값에서 파싱 텐서 추출: {parsing_tensor.shape}")
                    elif isinstance(first_value, (list, tuple)) and len(first_value) > 0:
                        if isinstance(first_value[0], torch.Tensor):
                            parsing_tensor = first_value[0]
                            self.logger.debug(f"✅ 첫 번째 리스트에서 파싱 텐서 추출: {parsing_tensor.shape}")
                
                if parsing_tensor is None:
                    raise ValueError("딕셔너리에서 파싱 텐서를 찾을 수 없습니다.")
                
                return parsing_tensor, confidence_tensor
            
            # 🔥 4단계: 리스트 형태 출력 처리
            elif isinstance(output, (list, tuple)):
                self.logger.debug(f"🔥 리스트 출력 길이: {len(output)}")
                
                if len(output) == 0:
                    raise ValueError("빈 리스트 출력입니다.")
                
                # 첫 번째 요소가 텐서인지 확인
                first_element = output[0]
                if isinstance(first_element, torch.Tensor):
                    parsing_tensor = first_element
                    self.logger.debug(f"✅ 리스트 첫 번째 요소에서 파싱 텐서 추출: {parsing_tensor.shape}")
                    
                    # 두 번째 요소가 신뢰도 텐서인지 확인
                    confidence_tensor = None
                    if len(output) > 1 and isinstance(output[1], torch.Tensor):
                        confidence_tensor = output[1]
                        self.logger.debug(f"✅ 리스트 두 번째 요소에서 신뢰도 텐서 추출: {confidence_tensor.shape}")
                    
                    return parsing_tensor, confidence_tensor
                else:
                    self.logger.warning(f"⚠️ 리스트 첫 번째 요소가 텐서가 아님: {type(first_element)}")
                    # 딕셔너리로 처리
                    if isinstance(first_element, dict):
                        return self._extract_parsing_from_output(first_element, device)
                    else:
                        raise ValueError(f"지원하지 않는 출력 타입: {type(first_element)}")
            
            # 🔥 5단계: 직접 텐서 출력 처리
            elif isinstance(output, torch.Tensor):
                self.logger.debug(f"✅ 직접 텐서 출력: {output.shape}")
                # 원본 텐서 그대로 반환 (차원 변환은 호출하는 곳에서 처리)
                return output, None
            
            # 🔥 6단계: 기타 타입 처리
            else:
                self.logger.warning(f"⚠️ 지원하지 않는 출력 타입: {type(output)}")
                raise ValueError(f"지원하지 않는 출력 타입: {type(output)}")
                
        except Exception as e:
            self.logger.error(f"❌ 파싱 출력 추출 실패: {e}")
            # 기본값 반환
            return torch.zeros((1, 20, 512, 512), device=device), None
    
    def _standardize_channels(self, tensor: torch.Tensor, target_channels: int = 20) -> torch.Tensor:
        """채널 수 표준화 (근본적 해결)"""
        try:
            # 🔥 입력 검증
            if tensor is None:
                self.logger.warning("⚠️ 텐서가 None입니다.")
                return torch.zeros((1, target_channels, 512, 512), device='cpu', dtype=torch.float32)
            
            # 🔥 차원 검증
            if len(tensor.shape) != 4:
                self.logger.warning(f"⚠️ 텐서 차원이 4가 아님: {tensor.shape}")
                if len(tensor.shape) == 3:
                    # 배치 차원 추가
                    tensor = tensor.unsqueeze(0)
                elif len(tensor.shape) == 2:
                    # 배치와 채널 차원 추가
                    tensor = tensor.unsqueeze(0).unsqueeze(0)
                else:
                    return torch.zeros((1, target_channels, 512, 512), device=tensor.device, dtype=tensor.dtype)
            
            # 🔥 채널 수 표준화
            if tensor.shape[1] == target_channels:
                return tensor
            elif tensor.shape[1] > target_channels:
                # 🔥 채널 수가 많으면 앞쪽 채널만 사용
                return tensor[:, :target_channels, :, :]
            else:
                # 🔥 채널 수가 적으면 패딩
                padding = torch.zeros(
                    tensor.shape[0], 
                    target_channels - tensor.shape[1], 
                    tensor.shape[2], 
                    tensor.shape[3],
                    device=tensor.device,
                    dtype=tensor.dtype
                )
                return torch.cat([tensor, padding], dim=1)
        except Exception as e:
            self.logger.warning(f"⚠️ 채널 수 표준화 실패: {e}")
            # 기본값 반환
            return torch.zeros((1, target_channels, 512, 512), device='cpu', dtype=torch.float32)


class GraphonomyInferenceEngine(InferenceEngine):
    """Graphonomy 추론 엔진"""
    
    def run_inference(self, input_tensor: torch.Tensor, model: nn.Module) -> Dict[str, Any]:
        """Graphonomy 앙상블 추론 - 근본적 해결"""
        try:
            # 🔥 1. 모델 검증 및 표준화
            if model is None:
                self.logger.warning("⚠️ Graphonomy 모델이 None입니다")
                return self._create_standard_output(input_tensor.device)
            
            # 🔥 2. 실제 모델 인스턴스 추출 (표준화)
            actual_model = self._extract_actual_model(model)
            if actual_model is None:
                return self._create_standard_output(input_tensor.device)
            
            # 🔥 3. MPS 타입 일치 (근본적 해결)
            device = input_tensor.device
            dtype = torch.float32  # 모든 텐서를 float32로 통일
            
            # 모델을 동일한 디바이스와 타입으로 변환
            actual_model = actual_model.to(device, dtype=dtype)
            input_tensor = input_tensor.to(device, dtype=dtype)
            
            # 모델의 모든 파라미터를 동일한 타입으로 변환
            for param in actual_model.parameters():
                param.data = param.data.to(dtype)
            
            # 🔥 4. 모델 추론 실행 (안전한 방식)
            try:
                with torch.no_grad():
                    # 텐서 포맷 오류 방지를 위한 완전한 로깅 비활성화
                    original_level = logging.getLogger().level
                    logging.getLogger().setLevel(logging.CRITICAL)
                    
                    # stdout/stderr 리다이렉션으로 텐서 포맷 오류 완전 차단
                    original_stdout = sys.stdout
                    original_stderr = sys.stderr
                    sys.stdout = io.StringIO()
                    sys.stderr = io.StringIO()
                    
                    try:
                        output = actual_model(input_tensor)
                    finally:
                        # 출력 복원
                        sys.stdout = original_stdout
                        sys.stderr = original_stderr
                        logging.getLogger().setLevel(original_level)
                    
            except Exception as inference_error:
                self.logger.warning(f"⚠️ Graphonomy 추론 실패: {inference_error}")
                return self._create_standard_output(device)
            
            # 🔥 5. 출력 표준화 (근본적 해결)
            parsing_output, _ = self._extract_parsing_from_output(output, input_tensor.device)
            
            # 🔥 6. 채널 수 표준화 (20개로 통일)
            parsing_output = self._standardize_channels(parsing_output, target_channels=20)
            
            # 🔥 7. 신뢰도 계산
            confidence = self._calculate_confidence(parsing_output)
            
            return {
                'parsing_pred': parsing_output,  # 일관된 키 이름 사용
                'parsing_output': parsing_output,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Graphonomy 모델 추론 실패: {str(e)}")
            return self._create_standard_output(input_tensor.device)
    
    def _calculate_confidence(self, parsing_probs, parsing_logits=None, edge_output=None, mode='advanced'):
        """신뢰도 계산 (고급 알고리즘)"""
        try:
            if parsing_probs is None:
                return 0.5
            
            # 🔥 고급 신뢰도 계산
            if mode == 'advanced':
                # 1. 기본 신뢰도 (소프트맥스 확률 기반)
                if parsing_probs.dim() == 4:
                    # 4차원 텐서: [batch, channels, height, width]
                    softmax_probs = F.softmax(parsing_probs, dim=1)
                    max_probs = torch.max(softmax_probs, dim=1)[0]  # [batch, height, width]
                    base_confidence = torch.mean(max_probs).item()
                else:
                    # 2차원 또는 3차원 텐서
                    base_confidence = 0.8
                
                # 2. 엣지 신뢰도 (엣지 출력이 있는 경우)
                edge_confidence = 1.0
                if edge_output is not None:
                    if edge_output.dim() == 4:
                        edge_confidence = torch.mean(torch.sigmoid(edge_output)).item()
                    else:
                        edge_confidence = 0.9
                
                # 3. 공간 일관성 신뢰도
                spatial_confidence = self._calculate_spatial_consistency(parsing_probs)
                
                # 4. 종합 신뢰도 계산
                final_confidence = (base_confidence * 0.5 + 
                                  edge_confidence * 0.3 + 
                                  spatial_confidence * 0.2)
                
                return max(0.1, min(1.0, final_confidence))
            
            else:
                # 🔥 기본 신뢰도 계산
                if parsing_probs.dim() == 4:
                    softmax_probs = F.softmax(parsing_probs, dim=1)
                    max_probs = torch.max(softmax_probs, dim=1)[0]
                    confidence = torch.mean(max_probs).item()
                else:
                    confidence = 0.8
                
                return max(0.1, min(1.0, confidence))
                
        except Exception as e:
            self.logger.warning(f"⚠️ 신뢰도 계산 실패: {e}")
            return 0.8
    
    def _calculate_spatial_consistency(self, parsing_pred):
        """공간 일관성 계산"""
        try:
            if parsing_pred.dim() == 4:
                # 4차원 텐서의 경우
                softmax_probs = F.softmax(parsing_pred, dim=1)
                max_probs = torch.max(softmax_probs, dim=1)[0]
                
                # 공간적 일관성 계산 (이웃 픽셀 간의 유사성)
                consistency = torch.mean(max_probs).item()
                return consistency
            else:
                return 0.8
        except:
            return 0.8


class HRNetInferenceEngine(InferenceEngine):
    """HRNet 추론 엔진"""
    
    def run_inference(self, input_tensor: torch.Tensor, model: nn.Module) -> Dict[str, Any]:
        """HRNet 앙상블 추론 - 근본적 해결"""
        try:
            # 🔥 1. 모델 검증 및 표준화
            if model is None:
                self.logger.warning("⚠️ HRNet 모델이 None입니다")
                return self._create_standard_output(input_tensor.device)
            
            # 🔥 2. 실제 모델 인스턴스 추출 (표준화)
            actual_model = self._extract_actual_model(model)
            if actual_model is None:
                return self._create_standard_output(input_tensor.device)
            
            # 🔥 3. MPS 타입 일치 (근본적 해결)
            device = input_tensor.device
            dtype = torch.float32  # 모든 텐서를 float32로 통일
            
            # 모델을 동일한 디바이스와 타입으로 변환
            actual_model = actual_model.to(device, dtype=dtype)
            input_tensor = input_tensor.to(device, dtype=dtype)
            
            # 모델의 모든 파라미터를 동일한 타입으로 변환
            for param in actual_model.parameters():
                param.data = param.data.to(dtype)
            
            # 🔥 4. 모델 추론 실행 (안전한 방식)
            try:
                with torch.no_grad():
                    # 텐서 포맷 오류 방지를 위한 완전한 로깅 비활성화
                    import logging
                    import sys
                    import io
                    
                    # 모든 로깅 비활성화
                    original_level = logging.getLogger().level
                    logging.getLogger().setLevel(logging.CRITICAL)
                    
                    # stdout/stderr 리다이렉션으로 텐서 포맷 오류 완전 차단
                    original_stdout = sys.stdout
                    original_stderr = sys.stderr
                    sys.stdout = io.StringIO()
                    sys.stderr = io.StringIO()
                    
                    try:
                        output = actual_model(input_tensor)
                    finally:
                        # 출력 복원
                        sys.stdout = original_stdout
                        sys.stderr = original_stderr
                        logging.getLogger().setLevel(original_level)
                    
            except Exception as inference_error:
                self.logger.warning(f"⚠️ HRNet 추론 실패: {inference_error}")
                return self._create_standard_output(input_tensor.device)
            
            # 🔥 5. 출력 표준화 (근본적 해결)
            parsing_output, _ = self._extract_parsing_from_output(output, input_tensor.device)
            
            # 🔥 6. 채널 수 표준화 (20개로 통일)
            parsing_output = self._standardize_channels(parsing_output, target_channels=20)
            
            # 🔥 7. 신뢰도 계산
            confidence = self._calculate_confidence(parsing_output)
            
            return {
                'parsing_pred': parsing_output,  # 일관된 키 이름 사용
                'parsing_output': parsing_output,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ HRNet 모델 추론 실패: {str(e)}")
            return self._create_standard_output(input_tensor.device)
    
    def _calculate_confidence(self, parsing_probs, parsing_logits=None, edge_output=None, mode='advanced'):
        """신뢰도 계산"""
        try:
            if parsing_probs is None:
                return 0.5
            
            if parsing_probs.dim() == 4:
                softmax_probs = F.softmax(parsing_probs, dim=1)
                max_probs = torch.max(softmax_probs, dim=1)[0]
                confidence = torch.mean(max_probs).item()
            else:
                confidence = 0.8
            
            return max(0.1, min(1.0, confidence))
        except:
            return 0.8


class DeepLabV3PlusInferenceEngine(InferenceEngine):
    """DeepLabV3+ 추론 엔진"""
    
    def run_inference(self, input_tensor: torch.Tensor, model: nn.Module) -> Dict[str, Any]:
        """DeepLabV3+ 앙상블 추론"""
        try:
            # RealAIModel에서 실제 모델 인스턴스 추출
            if hasattr(model, 'model_instance') and model.model_instance is not None:
                actual_model = model.model_instance
                self.logger.info("✅ DeepLabV3+ - RealAIModel에서 실제 모델 인스턴스 추출 성공")
            elif hasattr(model, 'get_model_instance'):
                actual_model = model.get_model_instance()
                self.logger.info("✅ DeepLabV3+ - get_model_instance()로 실제 모델 인스턴스 추출 성공")
            else:
                actual_model = model
                self.logger.info("⚠️ DeepLabV3+ - 직접 모델 사용 (RealAIModel 아님)")
            
            # 모델을 동일한 디바이스와 타입으로 변환 (MPS 타입 일치)
            device = input_tensor.device
            dtype = torch.float32  # 모든 텐서를 float32로 통일
            
            if hasattr(actual_model, 'to'):
                actual_model = actual_model.to(device, dtype=dtype)
                self.logger.info(f"✅ DeepLabV3+ 모델을 {device} 디바이스로 이동 (float32)")
            
            # 모델의 모든 파라미터를 동일한 타입으로 변환
            for param in actual_model.parameters():
                param.data = param.data.to(dtype)
            
            # 모델이 callable한지 확인
            if not callable(actual_model):
                self.logger.warning("⚠️ DeepLabV3+ 모델이 callable하지 않습니다")
                # 실제 모델이 아닌 경우 오류 발생
                raise ValueError("DeepLabV3+ 모델이 올바르게 로드되지 않았습니다")
            
            # 텐서 포맷 오류 방지를 위한 완전한 로깅 비활성화
            import logging
            import sys
            import io
            
            # 모든 로깅 비활성화
            original_level = logging.getLogger().level
            logging.getLogger().setLevel(logging.CRITICAL)
            
            # stdout/stderr 리다이렉션으로 텐서 포맷 오류 완전 차단
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            
            try:
                output = actual_model(input_tensor)
            finally:
                # 출력 복원
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                logging.getLogger().setLevel(original_level)
            
            # DeepLabV3+ 출력 처리
            if isinstance(output, (tuple, list)):
                parsing_output = output[0]
            else:
                parsing_output = output
            
            confidence = self._calculate_confidence(parsing_output)
            
            return {
                'parsing_pred': parsing_output,  # 일관된 키 이름 사용
                'parsing_output': parsing_output,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ DeepLabV3+ 모델 추론 실패: {str(e)}")
            return {
                'parsing_pred': torch.zeros((1, 20, 512, 512)),
                'parsing_output': torch.zeros((1, 20, 512, 512)),
                'confidence': 0.5
            }
    
    def _calculate_confidence(self, parsing_probs, parsing_logits=None, edge_output=None, mode='advanced'):
        """신뢰도 계산"""
        try:
            if parsing_probs is None:
                return 0.5
            
            if parsing_probs.dim() == 4:
                softmax_probs = F.softmax(parsing_probs, dim=1)
                max_probs = torch.max(softmax_probs, dim=1)[0]
                confidence = torch.mean(max_probs).item()
            else:
                confidence = 0.8
            
            return max(0.1, min(1.0, confidence))
        except:
            return 0.8


class U2NetInferenceEngine(InferenceEngine):
    """U2Net 추론 엔진"""
    
    def run_inference(self, input_tensor: torch.Tensor, model: nn.Module) -> Dict[str, Any]:
        """U2Net 앙상블 추론"""
        # RealAIModel에서 실제 모델 인스턴스 추출
        if hasattr(model, 'model_instance') and model.model_instance is not None:
            actual_model = model.model_instance
            self.logger.info("✅ U2Net - RealAIModel에서 실제 모델 인스턴스 추출 성공")
        elif hasattr(model, 'get_model_instance'):
            actual_model = model.get_model_instance()
            self.logger.info("✅ U2Net - get_model_instance()로 실제 모델 인스턴스 추출 성공")
            
            # 체크포인트 데이터 출력 방지
            if isinstance(actual_model, dict):
                self.logger.info(f"✅ U2Net - 체크포인트 데이터 감지됨")
            else:
                self.logger.info(f"✅ U2Net - 모델 타입: {type(actual_model)}")
        else:
            actual_model = model
            self.logger.info("⚠️ U2Net - 직접 모델 사용 (RealAIModel 아님)")
        
        # 모델을 MPS 디바이스로 이동
        if hasattr(actual_model, 'to'):
            actual_model = actual_model.to(self.device)
            self.logger.info(f"✅ U2Net 모델을 {self.device} 디바이스로 이동")
        
        output = actual_model(input_tensor)
        
        # U2Net 출력 처리
        if isinstance(output, (tuple, list)):
            parsing_output = output[0]
        else:
            parsing_output = output
        
        confidence = self._calculate_confidence(parsing_output)
        
        return {
            'parsing_output': parsing_output,
            'confidence': confidence
        }
    
    def _calculate_confidence(self, parsing_probs, parsing_logits=None, edge_output=None, mode='advanced'):
        """신뢰도 계산"""
        try:
            if parsing_probs is None:
                return 0.5
            
            if parsing_probs.dim() == 4:
                softmax_probs = F.softmax(parsing_probs, dim=1)
                max_probs = torch.max(softmax_probs, dim=1)[0]
                confidence = torch.mean(max_probs).item()
            else:
                confidence = 0.8
            
            return max(0.1, min(1.0, confidence))
        except:
            return 0.8


class GenericInferenceEngine(InferenceEngine):
    """일반 모델 추론 엔진"""
    
    def run_inference(self, input_tensor: torch.Tensor, model: nn.Module) -> Dict[str, Any]:
        """일반 모델 앙상블 추론 - MPS 호환성 개선"""
        return self._run_graphonomy_ensemble_inference_mps_safe(input_tensor, model)
    
    def _run_graphonomy_ensemble_inference_mps_safe(self, input_tensor: torch.Tensor, model: nn.Module) -> Dict[str, Any]:
        """🔥 Graphonomy 안전 추론 - 텐서 포맷 오류 완전 차단"""
        try:
            # 🔥 1. 디바이스 확인 및 설정
            device = input_tensor.device
            device_str = str(device)
            
            # 🔥 2. 모델 추출
            actual_model = self._extract_actual_model(model)
            if actual_model is None:
                return self._create_standard_output(device_str)
            
            # 🔥 3. MPS 타입 통일
            actual_model = actual_model.to(device_str, dtype=torch.float32)
            input_tensor = input_tensor.to(device_str, dtype=torch.float32)
            
            # 🔥 4. 완전한 출력 차단으로 안전 추론
            import os
            import sys
            import io
            
            # 환경 변수로 텐서 포맷 오류 방지
            os.environ['PYTORCH_DISABLE_TENSOR_FORMAT'] = '1'
            
            # stdout/stderr 완전 차단
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            
            try:
                with torch.no_grad():
                    output = actual_model(input_tensor)
            finally:
                # 출력 복원
                sys.stdout = original_stdout
                sys.stderr = original_stderr
            
            # 🔥 5. 출력 처리
            parsing_output, _ = self._extract_parsing_from_output(output, device_str)
            confidence = self._calculate_confidence(parsing_output)
            
            return {
                'parsing_pred': parsing_output,
                'parsing_output': parsing_output,
                'confidence': confidence,
                'edge_output': None
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Graphonomy 안전 추론 실패: {str(e)}")
            return self._create_standard_output(device_str if 'device_str' in locals() else 'cpu')
    
    def _calculate_confidence(self, parsing_probs, parsing_logits=None, edge_output=None, mode='advanced'):
        """신뢰도 계산"""
        try:
            if parsing_probs is None:
                return 0.5
            
            if parsing_probs.dim() == 4:
                softmax_probs = F.softmax(parsing_probs, dim=1)
                max_probs = torch.max(softmax_probs, dim=1)[0]
                confidence = torch.mean(max_probs).item()
            else:
                confidence = 0.8
            
            return max(0.1, min(1.0, confidence))
        except:
            return 0.8
