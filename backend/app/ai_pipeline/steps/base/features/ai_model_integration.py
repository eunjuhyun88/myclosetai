#!/usr/bin/env python3
"""
🔥 MyCloset AI - AI Model Integration Mixin
==========================================

AI 모델 통합 및 추론 실행을 담당하는 Mixin 클래스
- 모델 로딩 및 관리
- AI 추론 실행
- 입력/출력 데이터 변환
- 체크포인트 관리

Author: MyCloset AI Team
Date: 2025-08-14
Version: 2.0
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path

# 선택적 import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

class AIModelIntegrationMixin:
    """AI 모델 통합 및 추론 실행을 담당하는 Mixin"""
    
    def _run_ai_inference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """AI 추론 실행 - 기본 구현"""
        try:
            self.logger.info(f"🔥 {self.step_name} AI 추론 시작")
            start_time = time.time()
            
            # 기본 추론 로직
            result = self._run_step_specific_inference(input_data)
            
            # 성능 측정
            processing_time = time.time() - start_time
            self._update_performance_metrics('ai_inference', processing_time, True)
            
            self.logger.info(f"✅ {self.step_name} AI 추론 완료 ({processing_time:.3f}초)")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} AI 추론 실패: {e}")
            self._update_performance_metrics('ai_inference', 0.0, False, str(e))
            return self._create_error_response(str(e))
    
    def _run_step_specific_inference(self, input_data: Dict[str, Any], checkpoint_data: Any = None, device: str = None) -> Dict[str, Any]:
        """Step별 특화 추론 실행 - 하위 클래스에서 구현"""
        try:
            # 기본 구현: 입력 데이터를 그대로 반환
            self.logger.debug(f"🔄 {self.step_name} 기본 추론 실행")
            
            # 체크포인트가 있는 경우 로드
            if checkpoint_data and hasattr(self, '_load_checkpoint'):
                self._load_checkpoint(checkpoint_data)
            
            # 디바이스 설정
            target_device = device or self.device
            
            # 모델이 있는 경우 추론 실행
            if hasattr(self, 'model') and self.model is not None:
                return self._execute_model_inference(input_data, target_device)
            
            # 모델이 없는 경우 Mock 결과 반환
            return self._create_mock_inference_result(input_data)
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 특화 추론 실패: {e}")
            return self._create_error_response(str(e))
    
    def _execute_model_inference(self, input_data: Dict[str, Any], device: str) -> Dict[str, Any]:
        """모델 추론 실행"""
        try:
            if not TORCH_AVAILABLE:
                return self._create_mock_inference_result(input_data)
            
            # 입력 데이터를 텐서로 변환
            tensor_input = self._convert_input_to_tensor(input_data)
            
            # 모델을 지정된 디바이스로 이동
            if hasattr(self.model, 'to'):
                self.model.to(device)
            
            # 추론 실행
            with torch.no_grad():
                if hasattr(self.model, 'forward'):
                    output = self.model(tensor_input)
                elif hasattr(self.model, 'predict'):
                    output = self.model.predict(tensor_input)
                elif hasattr(self.model, 'detect'):
                    output = self.model.detect(tensor_input)
                else:
                    return self._create_mock_inference_result(input_data)
            
            # 출력을 표준 형식으로 변환
            return self._convert_model_output_to_standard(output)
            
        except Exception as e:
            self.logger.error(f"❌ 모델 추론 실행 실패: {e}")
            return self._create_mock_inference_result(input_data)
    
    def _convert_input_to_tensor(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """입력 데이터를 텐서로 변환"""
        result = {}
        
        for key, value in input_data.items():
            try:
                if NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                    if len(value.shape) == 3 and value.shape[2] in [1, 3, 4]:
                        value = np.transpose(value, (2, 0, 1))
                    tensor = torch.from_numpy(value).float()
                    
                    # MPS 디바이스에서 float64 → float32 변환
                    if self.device == 'mps' and tensor.dtype == torch.float64:
                        tensor = tensor.to(torch.float32)
                    
                    result[key] = tensor
                    
                elif PIL_AVAILABLE and isinstance(value, Image.Image):
                    array = np.array(value).astype(np.float32)
                    if len(array.shape) == 3 and array.shape[2] in [1, 3, 4]:
                        array = np.transpose(array, (2, 0, 1))
                    tensor = torch.from_numpy(array)
                    
                    # MPS 디바이스에서 float64 → float32 변환
                    if self.device == 'mps' and tensor.dtype == torch.float64:
                        tensor = tensor.to(torch.float32)
                    
                    result[key] = tensor
                    
                else:
                    result[key] = value
                    
            except Exception as e:
                self.logger.debug(f"텐서 변환 실패 ({key}): {e}")
                result[key] = value
        
        return result
    
    def _convert_model_output_to_standard(self, model_output: Any) -> Dict[str, Any]:
        """모델 출력을 표준 형식으로 변환"""
        try:
            if isinstance(model_output, torch.Tensor):
                # 텐서를 numpy로 변환
                if NUMPY_AVAILABLE:
                    output_array = model_output.detach().cpu().numpy()
                else:
                    output_array = model_output.detach().cpu().tolist()
                
                return {
                    'output': output_array,
                    'output_type': 'tensor',
                    'shape': list(model_output.shape),
                    'dtype': str(model_output.dtype)
                }
            
            elif isinstance(model_output, dict):
                # 딕셔너리 형태의 출력
                converted = {}
                for key, value in model_output.items():
                    if isinstance(value, torch.Tensor):
                        if NUMPY_AVAILABLE:
                            converted[key] = value.detach().cpu().numpy()
                        else:
                            converted[key] = value.detach().cpu().tolist()
                    else:
                        converted[key] = value
                
                return converted
            
            else:
                # 기타 형태의 출력
                return {
                    'output': model_output,
                    'output_type': type(model_output).__name__
                }
                
        except Exception as e:
            self.logger.error(f"❌ 모델 출력 변환 실패: {e}")
            return {
                'output': model_output,
                'error': str(e)
            }
    
    def _create_mock_inference_result(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock 추론 결과 생성"""
        return {
            'status': 'mock',
            'step_name': self.step_name,
            'input_keys': list(input_data.keys()),
            'message': 'Mock 추론 결과 - 실제 모델이 로드되지 않음'
        }
    
    def _load_primary_model(self):
        """주요 모델 로드"""
        try:
            if hasattr(self, 'model_loader') and self.model_loader:
                self.model = self.model_loader.load_primary_model()
                self.has_model = True
                self.model_loaded = True
                self.logger.info(f"✅ {self.step_name} 주요 모델 로드 완료")
                return True
            else:
                self.logger.warning(f"⚠️ {self.step_name} model_loader가 없음")
                return False
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 주요 모델 로드 실패: {e}")
            return False
    
    def _update_performance_metrics(self, operation: str, duration: float, success: bool, error: str = None):
        """성능 메트릭 업데이트"""
        try:
            if hasattr(self, 'performance_metrics'):
                self.performance_metrics.record_operation(operation, duration, success, error)
        except Exception as e:
            self.logger.debug(f"성능 메트릭 업데이트 실패: {e}")
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """에러 응답 생성"""
        return {
            'success': False,
            'error': error_message,
            'step_name': self.step_name,
            'step_id': getattr(self, 'step_id', 0)
        }
