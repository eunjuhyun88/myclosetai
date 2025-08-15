"""
Validation Utilities
입력 검증과 오류 처리를 위한 유틸리티 함수들을 제공하는 클래스
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass
import re
import os
from pathlib import Path

# 프로젝트 로깅 설정 import
from backend.app.ai_pipeline.utils.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class ValidationResult:
    """검증 결과를 저장하는 데이터 클래스"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    validation_details: Dict[str, Any]

class ValidationUtils:
    """
    입력 검증과 오류 처리를 위한 유틸리티 함수들을 제공하는 클래스
    """
    
    def __init__(self):
        """검증 유틸리티 초기화"""
        self.validation_rules = {}
        self.custom_validators = {}
        
        logger.info("ValidationUtils initialized")
    
    def validate_image_input(self, image: Union[torch.Tensor, np.ndarray, Image.Image, str], 
                           expected_format: str = 'tensor',
                           expected_shape: Optional[Tuple[int, ...]] = None,
                           expected_dtype: Optional[torch.dtype] = None) -> ValidationResult:
        """
        이미지 입력 검증
        
        Args:
            image: 검증할 이미지
            expected_format: 예상 형식 ('tensor', 'numpy', 'pil', 'file')
            expected_shape: 예상 형태
            expected_dtype: 예상 데이터 타입
            
        Returns:
            검증 결과
        """
        try:
            errors = []
            warnings = []
            validation_details = {}
            
            # 기본 타입 검증
            if not self._is_valid_image_type(image):
                errors.append(f"지원하지 않는 이미지 타입: {type(image)}")
                return ValidationResult(False, errors, warnings, validation_details)
            
            # 형식별 검증
            if expected_format == 'tensor':
                validation_result = self._validate_tensor_image(image, expected_shape, expected_dtype)
            elif expected_format == 'numpy':
                validation_result = self._validate_numpy_image(image, expected_shape)
            elif expected_format == 'pil':
                validation_result = self._validate_pil_image(image)
            elif expected_format == 'file':
                validation_result = self._validate_image_file(image)
            else:
                errors.append(f"알 수 없는 예상 형식: {expected_format}")
                return ValidationResult(False, errors, warnings, validation_details)
            
            # 결과 병합
            errors.extend(validation_result.errors)
            warnings.extend(validation_result.warnings)
            validation_details.update(validation_result.validation_details)
            
            # 최종 검증 결과
            is_valid = len(errors) == 0
            
            return ValidationResult(is_valid, errors, warnings, validation_details)
            
        except Exception as e:
            logger.error(f"이미지 입력 검증 중 오류 발생: {e}")
            return ValidationResult(False, [f"검증 중 오류 발생: {str(e)}"], [], {})
    
    def validate_model_input(self, model: nn.Module, 
                           input_tensor: torch.Tensor,
                           expected_output_shape: Optional[Tuple[int, ...]] = None) -> ValidationResult:
        """
        모델 입력 검증
        
        Args:
            model: 검증할 모델
            input_tensor: 입력 텐서
            expected_output_shape: 예상 출력 형태
            
        Returns:
            검증 결과
        """
        try:
            errors = []
            warnings = []
            validation_details = {}
            
            # 모델 검증
            model_validation = self._validate_model(model)
            errors.extend(model_validation.errors)
            warnings.extend(model_validation.warnings)
            validation_details.update(model_validation.validation_details)
            
            # 입력 텐서 검증
            input_validation = self._validate_input_tensor(input_tensor)
            errors.extend(input_validation.errors)
            warnings.extend(input_validation.warnings)
            validation_details.update(input_validation.validation_details)
            
            # 모델 호환성 검증
            if len(errors) == 0:
                compatibility_validation = self._validate_model_compatibility(model, input_tensor)
                errors.extend(compatibility_validation.errors)
                warnings.extend(compatibility_validation.warnings)
                validation_details.update(compatibility_validation.validation_details)
            
            # 출력 형태 검증
            if expected_output_shape and len(errors) == 0:
                output_validation = self._validate_expected_output(model, input_tensor, expected_output_shape)
                errors.extend(output_validation.errors)
                warnings.extend(output_validation.warnings)
                validation_details.update(output_validation.validation_details)
            
            # 최종 검증 결과
            is_valid = len(errors) == 0
            
            return ValidationResult(is_valid, errors, warnings, validation_details)
            
        except Exception as e:
            logger.error(f"모델 입력 검증 중 오류 발생: {e}")
            return ValidationResult(False, [f"검증 중 오류 발생: {str(e)}"], [], {})
    
    def validate_config(self, config: Dict[str, Any], 
                       required_keys: List[str],
                       optional_keys: List[str] = None,
                       value_validators: Optional[Dict[str, Callable]] = None) -> ValidationResult:
        """
        설정 딕셔너리 검증
        
        Args:
            config: 검증할 설정
            required_keys: 필수 키 목록
            optional_keys: 선택적 키 목록
            value_validators: 값 검증 함수들
            
        Returns:
            검증 결과
        """
        try:
            errors = []
            warnings = []
            validation_details = {}
            
            if not isinstance(config, dict):
                errors.append("설정이 딕셔너리가 아닙니다")
                return ValidationResult(False, errors, warnings, validation_details)
            
            # 필수 키 검증
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                errors.append(f"필수 키가 누락되었습니다: {missing_keys}")
            
            # 선택적 키 검증
            if optional_keys:
                unknown_keys = [key for key in config.keys() 
                              if key not in required_keys and key not in optional_keys]
                if unknown_keys:
                    warnings.append(f"알 수 없는 키가 발견되었습니다: {unknown_keys}")
            
            # 값 검증
            if value_validators:
                for key, validator in value_validators.items():
                    if key in config:
                        try:
                            validation_result = validator(config[key])
                            if not validation_result:
                                errors.append(f"키 '{key}'의 값이 유효하지 않습니다")
                        except Exception as e:
                            errors.append(f"키 '{key}'의 값 검증 중 오류 발생: {e}")
            
            # 검증 세부사항
            validation_details = {
                'total_keys': len(config),
                'required_keys_found': len([k for k in required_keys if k in config]),
                'optional_keys_found': len([k for k in (optional_keys or []) if k in config]),
                'unknown_keys': [k for k in config.keys() 
                               if k not in required_keys and k not in (optional_keys or [])]
            }
            
            # 최종 검증 결과
            is_valid = len(errors) == 0
            
            return ValidationResult(is_valid, errors, warnings, validation_details)
            
        except Exception as e:
            logger.error(f"설정 검증 중 오류 발생: {e}")
            return ValidationResult(False, [f"검증 중 오류 발생: {str(e)}"], [], {})
    
    def validate_file_path(self, file_path: Union[str, Path], 
                          expected_extensions: Optional[List[str]] = None,
                          must_exist: bool = True,
                          must_be_file: bool = True) -> ValidationResult:
        """
        파일 경로 검증
        
        Args:
            file_path: 검증할 파일 경로
            expected_extensions: 예상 파일 확장자 목록
            must_exist: 파일이 존재해야 하는지 여부
            must_be_file: 파일이 실제 파일이어야 하는지 여부
            
        Returns:
            검증 결과
        """
        try:
            errors = []
            warnings = []
            validation_details = {}
            
            # 경로 객체 변환
            if isinstance(file_path, str):
                path_obj = Path(file_path)
            else:
                path_obj = file_path
            
            # 존재 여부 검증
            if must_exist and not path_obj.exists():
                errors.append(f"파일이 존재하지 않습니다: {file_path}")
            
            # 파일 타입 검증
            if must_exist and must_be_file and path_obj.exists() and not path_obj.is_file():
                errors.append(f"경로가 파일이 아닙니다: {file_path}")
            
            # 확장자 검증
            if expected_extensions and path_obj.exists():
                file_extension = path_obj.suffix.lower()
                if file_extension not in expected_extensions:
                    warnings.append(f"예상하지 않은 파일 확장자: {file_extension}")
            
            # 검증 세부사항
            validation_details = {
                'path_exists': path_obj.exists(),
                'is_file': path_obj.is_file() if path_obj.exists() else False,
                'file_extension': path_obj.suffix.lower() if path_obj.exists() else None,
                'file_size': path_obj.stat().st_size if path_obj.exists() and path_obj.is_file() else None
            }
            
            # 최종 검증 결과
            is_valid = len(errors) == 0
            
            return ValidationResult(is_valid, errors, warnings, validation_details)
            
        except Exception as e:
            logger.error(f"파일 경로 검증 중 오류 발생: {e}")
            return ValidationResult(False, [f"검증 중 오류 발생: {str(e)}"], [], {})
    
    def add_custom_validator(self, name: str, validator_func: Callable):
        """
        사용자 정의 검증 함수 추가
        
        Args:
            name: 검증 함수 이름
            validator_func: 검증 함수
        """
        try:
            self.custom_validators[name] = validator_func
            logger.info(f"사용자 정의 검증 함수 추가: {name}")
        except Exception as e:
            logger.error(f"사용자 정의 검증 함수 추가 중 오류 발생: {e}")
    
    def get_validation_rules(self) -> Dict[str, Any]:
        """검증 규칙 반환"""
        return self.validation_rules.copy()
    
    def set_validation_rules(self, rules: Dict[str, Any]):
        """검증 규칙 설정"""
        try:
            self.validation_rules.update(rules)
            logger.info("검증 규칙 업데이트 완료")
        except Exception as e:
            logger.error(f"검증 규칙 설정 중 오류 발생: {e}")
    
    def _is_valid_image_type(self, image: Any) -> bool:
        """이미지 타입 유효성 검사"""
        valid_types = (torch.Tensor, np.ndarray, Image.Image, str, Path)
        return isinstance(image, valid_types)
    
    def _validate_tensor_image(self, image: torch.Tensor, 
                              expected_shape: Optional[Tuple[int, ...]], 
                              expected_dtype: Optional[torch.dtype]) -> ValidationResult:
        """텐서 이미지 검증"""
        errors = []
        warnings = []
        validation_details = {}
        
        try:
            # 텐서 타입 검증
            if not isinstance(image, torch.Tensor):
                errors.append("이미지가 PyTorch 텐서가 아닙니다")
                return ValidationResult(False, errors, warnings, validation_details)
            
            # 차원 검증
            if image.dim() != 3:
                errors.append(f"이미지 차원이 3이 아닙니다: {image.dim()}")
            
            # 채널 수 검증
            if image.size(0) not in [1, 3, 4]:
                errors.append(f"지원하지 않는 채널 수: {image.size(0)}")
            
            # 형태 검증
            if expected_shape:
                if image.shape != expected_shape:
                    errors.append(f"예상 형태와 일치하지 않습니다: 예상 {expected_shape}, 실제 {image.shape}")
            
            # 데이터 타입 검증
            if expected_dtype and image.dtype != expected_dtype:
                warnings.append(f"예상 데이터 타입과 일치하지 않습니다: 예상 {expected_dtype}, 실제 {image.dtype}")
            
            # 값 범위 검증
            if image.numel() > 0:
                min_val = float(image.min())
                max_val = float(image.max())
                
                if min_val < -1e6 or max_val > 1e6:
                    warnings.append(f"이미지 값 범위가 비정상적입니다: [{min_val}, {max_val}]")
                
                if torch.isnan(image).any():
                    errors.append("이미지에 NaN 값이 포함되어 있습니다")
                
                if torch.isinf(image).any():
                    errors.append("이미지에 Inf 값이 포함되어 있습니다")
            
            # 검증 세부사항
            validation_details = {
                'shape': list(image.shape),
                'dtype': str(image.dtype),
                'device': str(image.device),
                'numel': image.numel(),
                'min_value': float(image.min()) if image.numel() > 0 else None,
                'max_value': float(image.max()) if image.numel() > 0 else None
            }
            
        except Exception as e:
            errors.append(f"텐서 이미지 검증 중 오류 발생: {e}")
        
        return ValidationResult(len(errors) == 0, errors, warnings, validation_details)
    
    def _validate_numpy_image(self, image: np.ndarray, 
                             expected_shape: Optional[Tuple[int, ...]]) -> ValidationResult:
        """NumPy 이미지 검증"""
        errors = []
        warnings = []
        validation_details = {}
        
        try:
            # NumPy 배열 타입 검증
            if not isinstance(image, np.ndarray):
                errors.append("이미지가 NumPy 배열이 아닙니다")
                return ValidationResult(False, errors, warnings, validation_details)
            
            # 차원 검증
            if image.ndim != 3:
                errors.append(f"이미지 차원이 3이 아닙니다: {image.ndim}")
            
            # 채널 수 검증
            if image.shape[-1] not in [1, 3, 4]:
                errors.append(f"지원하지 않는 채널 수: {image.shape[-1]}")
            
            # 형태 검증
            if expected_shape:
                if image.shape != expected_shape:
                    errors.append(f"예상 형태와 일치하지 않습니다: 예상 {expected_shape}, 실제 {image.shape}")
            
            # 데이터 타입 검증
            if not np.issubdtype(image.dtype, np.number):
                errors.append(f"지원하지 않는 데이터 타입: {image.dtype}")
            
            # 값 범위 검증
            if image.size > 0:
                min_val = float(image.min())
                max_val = float(image.max())
                
                if min_val < -1e6 or max_val > 1e6:
                    warnings.append(f"이미지 값 범위가 비정상적입니다: [{min_val}, {max_val}]")
                
                if np.isnan(image).any():
                    errors.append("이미지에 NaN 값이 포함되어 있습니다")
                
                if np.isinf(image).any():
                    errors.append("이미지에 Inf 값이 포함되어 있습니다")
            
            # 검증 세부사항
            validation_details = {
                'shape': list(image.shape),
                'dtype': str(image.dtype),
                'size': image.size,
                'min_value': float(image.min()) if image.size > 0 else None,
                'max_value': float(image.max()) if image.size > 0 else None
            }
            
        except Exception as e:
            errors.append(f"NumPy 이미지 검증 중 오류 발생: {e}")
        
        return ValidationResult(len(errors) == 0, errors, warnings, validation_details)
    
    def _validate_pil_image(self, image: Image.Image) -> ValidationResult:
        """PIL 이미지 검증"""
        errors = []
        warnings = []
        validation_details = {}
        
        try:
            # PIL 이미지 타입 검증
            if not isinstance(image, Image.Image):
                errors.append("이미지가 PIL 이미지가 아닙니다")
                return ValidationResult(False, errors, warnings, validation_details)
            
            # 모드 검증
            valid_modes = ['L', 'RGB', 'RGBA', 'LA', 'P']
            if image.mode not in valid_modes:
                warnings.append(f"지원하지 않는 이미지 모드: {image.mode}")
            
            # 크기 검증
            width, height = image.size
            if width <= 0 or height <= 0:
                errors.append(f"유효하지 않은 이미지 크기: {width}x{height}")
            
            if width > 10000 or height > 10000:
                warnings.append(f"이미지 크기가 매우 큽니다: {width}x{height}")
            
            # 검증 세부사항
            validation_details = {
                'mode': image.mode,
                'size': image.size,
                'format': image.format,
                'width': width,
                'height': height
            }
            
        except Exception as e:
            errors.append(f"PIL 이미지 검증 중 오류 발생: {e}")
        
        return ValidationResult(len(errors) == 0, errors, warnings, validation_details)
    
    def _validate_image_file(self, file_path: Union[str, Path]) -> ValidationResult:
        """이미지 파일 검증"""
        errors = []
        warnings = []
        validation_details = {}
        
        try:
            # 파일 경로 검증
            path_validation = self.validate_file_path(file_path, 
                                                    expected_extensions=['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
                                                    must_exist=True,
                                                    must_be_file=True)
            
            if not path_validation.is_valid:
                errors.extend(path_validation.errors)
                warnings.extend(path_validation.warnings)
                validation_details.update(path_validation.validation_details)
                return ValidationResult(False, errors, warnings, validation_details)
            
            # 파일 크기 검증
            file_size = Path(file_path).stat().st_size
            if file_size == 0:
                errors.append("파일이 비어있습니다")
            elif file_size > 100 * 1024 * 1024:  # 100MB
                warnings.append(f"파일 크기가 매우 큽니다: {file_size / (1024*1024):.1f}MB")
            
            # 검증 세부사항
            validation_details = {
                'file_path': str(file_path),
                'file_size_bytes': file_size,
                'file_size_mb': file_size / (1024*1024),
                'file_extension': Path(file_path).suffix.lower()
            }
            
        except Exception as e:
            errors.append(f"이미지 파일 검증 중 오류 발생: {e}")
        
        return ValidationResult(len(errors) == 0, errors, warnings, validation_details)
    
    def _validate_model(self, model: nn.Module) -> ValidationResult:
        """모델 검증"""
        errors = []
        warnings = []
        validation_details = {}
        
        try:
            # 모델 타입 검증
            if not isinstance(model, nn.Module):
                errors.append("모델이 nn.Module의 인스턴스가 아닙니다")
                return ValidationResult(False, errors, warnings, validation_details)
            
            # 모델 상태 검증
            if model.training:
                warnings.append("모델이 훈련 모드에 있습니다")
            
            # 파라미터 검증
            total_params = sum(p.numel() for p in model.parameters())
            if total_params == 0:
                warnings.append("모델에 파라미터가 없습니다")
            
            # 검증 세부사항
            validation_details = {
                'model_type': type(model).__name__,
                'total_parameters': total_params,
                'training_mode': model.training,
                'device': next(model.parameters()).device if total_params > 0 else None
            }
            
        except Exception as e:
            errors.append(f"모델 검증 중 오류 발생: {e}")
        
        return ValidationResult(len(errors) == 0, errors, warnings, validation_details)
    
    def _validate_input_tensor(self, input_tensor: torch.Tensor) -> ValidationResult:
        """입력 텐서 검증"""
        errors = []
        warnings = []
        validation_details = {}
        
        try:
            # 텐서 타입 검증
            if not isinstance(input_tensor, torch.Tensor):
                errors.append("입력이 PyTorch 텐서가 아닙니다")
                return ValidationResult(False, errors, warnings, validation_details)
            
            # 차원 검증
            if input_tensor.dim() < 2:
                errors.append(f"입력 텐서 차원이 너무 작습니다: {input_tensor.dim()}")
            
            # 값 검증
            if input_tensor.numel() > 0:
                if torch.isnan(input_tensor).any():
                    errors.append("입력 텐서에 NaN 값이 포함되어 있습니다")
                
                if torch.isinf(input_tensor).any():
                    errors.append("입력 텐서에 Inf 값이 포함되어 있습니다")
            
            # 검증 세부사항
            validation_details = {
                'shape': list(input_tensor.shape),
                'dtype': str(input_tensor.dtype),
                'device': str(input_tensor.device),
                'numel': input_tensor.numel(),
                'dim': input_tensor.dim()
            }
            
        except Exception as e:
            errors.append(f"입력 텐서 검증 중 오류 발생: {e}")
        
        return ValidationResult(len(errors) == 0, errors, warnings, validation_details)
    
    def _validate_model_compatibility(self, model: nn.Module, 
                                    input_tensor: torch.Tensor) -> ValidationResult:
        """모델 호환성 검증"""
        errors = []
        warnings = []
        validation_details = {}
        
        try:
            # 디바이스 호환성 검증
            model_device = next(model.parameters()).device
            input_device = input_tensor.device
            
            if model_device != input_device:
                warnings.append(f"모델과 입력 텐서의 디바이스가 다릅니다: 모델 {model_device}, 입력 {input_device}")
            
            # 데이터 타입 호환성 검증
            model_dtype = next(model.parameters()).dtype
            input_dtype = input_tensor.dtype
            
            if model_dtype != input_dtype:
                warnings.append(f"모델과 입력 텐서의 데이터 타입이 다릅니다: 모델 {model_dtype}, 입력 {input_dtype}")
            
            # 검증 세부사항
            validation_details = {
                'model_device': str(model_device),
                'input_device': str(input_device),
                'model_dtype': str(model_dtype),
                'input_dtype': str(input_dtype),
                'devices_match': model_device == input_device,
                'dtypes_match': model_dtype == input_dtype
            }
            
        except Exception as e:
            errors.append(f"모델 호환성 검증 중 오류 발생: {e}")
        
        return ValidationResult(len(errors) == 0, errors, warnings, validation_details)
    
    def _validate_expected_output(self, model: nn.Module, 
                                 input_tensor: torch.Tensor, 
                                 expected_output_shape: Tuple[int, ...]) -> ValidationResult:
        """예상 출력 형태 검증"""
        errors = []
        warnings = []
        validation_details = {}
        
        try:
            # 모델을 평가 모드로 설정
            model.eval()
            
            # 추론 실행
            with torch.no_grad():
                try:
                    output = model(input_tensor)
                    
                    # 출력 형태 검증
                    if output.shape != expected_output_shape:
                        errors.append(f"출력 형태가 예상과 일치하지 않습니다: 예상 {expected_output_shape}, 실제 {output.shape}")
                    
                    # 검증 세부사항
                    validation_details = {
                        'expected_shape': expected_output_shape,
                        'actual_shape': list(output.shape),
                        'shapes_match': output.shape == expected_output_shape,
                        'output_dtype': str(output.dtype),
                        'output_device': str(output.device)
                    }
                    
                except Exception as e:
                    errors.append(f"모델 추론 중 오류 발생: {e}")
                    validation_details = {'error': str(e)}
            
        except Exception as e:
            errors.append(f"예상 출력 검증 중 오류 발생: {e}")
        
        return ValidationResult(len(errors) == 0, errors, warnings, validation_details)
