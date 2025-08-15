#!/usr/bin/env python3
"""
🔥 MyCloset AI - Validation Mixin
==================================

검증 관련 기능을 담당하는 Mixin 클래스
입력 데이터 검증, 의존성 검증 등을 담당

Author: MyCloset AI Team
Date: 2025-08-14
Version: 2.0
"""

import logging
from typing import Dict, Any, Optional, Union
from enum import Enum

class DependencyValidationFormat(Enum):
    """의존성 검증 반환 형식"""
    BOOLEAN_DICT = "dict_bool"  # GeometricMatchingStep 형식: {'model_loader': True, ...}
    DETAILED_DICT = "dict_detailed"  # BaseStepMixin v18.0 형식: {'success': True, 'details': {...}}
    AUTO_DETECT = "auto"  # 호출자에 따라 자동 선택

class ValidationMixin:
    """검증 관련 기능을 제공하는 Mixin"""
    
    def validate_dependencies(self, format_type: DependencyValidationFormat = None) -> Union[Dict[str, bool], Dict[str, Any]]:
        """의존성 검증"""
        try:
            if format_type is None:
                format_type = DependencyValidationFormat.AUTO_DETECT
            
            # 자동 감지
            if format_type == DependencyValidationFormat.AUTO_DETECT:
                # GitHub 프로젝트 호환성을 위해 boolean 형식 우선
                format_type = DependencyValidationFormat.BOOLEAN_DICT
            
            if format_type == DependencyValidationFormat.BOOLEAN_DICT:
                return self.validate_dependencies_boolean()
            elif format_type == DependencyValidationFormat.DETAILED_DICT:
                return self.validate_dependencies_detailed()
            else:
                return self.validate_dependencies_boolean()
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"의존성 검증 실패: {e}")
            return {'error': str(e)}

    def validate_dependencies_boolean(self) -> Dict[str, bool]:
        """의존성 검증 (boolean 형식) - GitHub 프로젝트 호환"""
        try:
            validation_result = {}
            
            # ModelLoader 검증
            if hasattr(self, 'model_loader') and self.model_loader:
                validation_result['model_loader'] = True
                if hasattr(self, 'logger'):
                    self.logger.debug("✅ ModelLoader 의존성 검증 통과")
            else:
                validation_result['model_loader'] = False
                if hasattr(self, 'logger'):
                    self.logger.warning("❌ ModelLoader 의존성 누락")
            
            # MemoryManager 검증
            if hasattr(self, 'memory_manager') and self.memory_manager:
                validation_result['memory_manager'] = True
                if hasattr(self, 'logger'):
                    self.logger.debug("✅ MemoryManager 의존성 검증 통과")
            else:
                validation_result['memory_manager'] = False
                if hasattr(self, 'logger'):
                    self.logger.debug("ℹ️ MemoryManager 의존성 선택사항")
            
            # DataConverter 검증
            if hasattr(self, 'data_converter') and self.data_converter:
                validation_result['data_converter'] = True
                if hasattr(self, 'logger'):
                    self.logger.debug("✅ DataConverter 의존성 검증 통과")
            else:
                validation_result['data_converter'] = False
                if hasattr(self, 'logger'):
                    self.logger.debug("ℹ️ DataConverter 의존성 선택사항")
            
            # Central Hub Container 검증
            if hasattr(self, 'central_hub_container') and self.central_hub_container:
                validation_result['central_hub_container'] = True
                if hasattr(self, 'logger'):
                    self.logger.debug("✅ Central Hub Container 의존성 검증 통과")
            else:
                validation_result['central_hub_container'] = False
                if hasattr(self, 'logger'):
                    self.logger.warning("❌ Central Hub Container 의존성 누락")
            
            # 전체 검증 결과
            all_valid = all(validation_result.values())
            validation_result['all_dependencies_valid'] = all_valid
            
            if hasattr(self, 'logger'):
                if all_valid:
                    self.logger.info("✅ 모든 필수 의존성 검증 통과")
                else:
                    self.logger.warning("⚠️ 일부 의존성 검증 실패")
            
            return validation_result
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"의존성 검증 (boolean) 실패: {e}")
            return {'error': str(e), 'all_dependencies_valid': False}

    def validate_dependencies_detailed(self) -> Dict[str, Any]:
        """의존성 검증 (상세 형식) - BaseStepMixin v18.0 호환"""
        try:
            validation_result = {
                'success': True,
                'details': {},
                'summary': {
                    'total_dependencies': 0,
                    'valid_dependencies': 0,
                    'missing_dependencies': 0,
                    'optional_dependencies': 0
                },
                'recommendations': []
            }
            
            # 필수 의존성들
            required_dependencies = ['model_loader', 'central_hub_container']
            optional_dependencies = ['memory_manager', 'data_converter']
            
            all_dependencies = required_dependencies + optional_dependencies
            validation_result['summary']['total_dependencies'] = len(all_dependencies)
            
            # 각 의존성 검증
            for dep_name in all_dependencies:
                is_required = dep_name in required_dependencies
                is_valid = hasattr(self, dep_name) and getattr(self, dep_name) is not None
                
                validation_result['details'][dep_name] = {
                    'valid': is_valid,
                    'required': is_required,
                    'type': type(getattr(self, dep_name, None)).__name__ if hasattr(self, dep_name) else 'None'
                }
                
                if is_valid:
                    validation_result['summary']['valid_dependencies'] += 1
                    if hasattr(self, 'logger'):
                        self.logger.debug(f"✅ {dep_name} 의존성 검증 통과")
                else:
                    if is_required:
                        validation_result['summary']['missing_dependencies'] += 1
                        validation_result['success'] = False
                        validation_result['recommendations'].append(f"{dep_name} 의존성 주입 필요")
                        if hasattr(self, 'logger'):
                            self.logger.warning(f"❌ 필수 의존성 {dep_name} 누락")
                    else:
                        validation_result['summary']['optional_dependencies'] += 1
                        if hasattr(self, 'logger'):
                            self.logger.debug(f"ℹ️ 선택사항 의존성 {dep_name} 누락")
            
            # 검증 결과 요약
            if validation_result['summary']['missing_dependencies'] > 0:
                validation_result['success'] = False
                validation_result['recommendations'].append("필수 의존성 주입 후 재검증 필요")
            
            if hasattr(self, 'logger'):
                if validation_result['success']:
                    self.logger.info("✅ 모든 필수 의존성 검증 통과")
                else:
                    self.logger.warning(f"⚠️ {validation_result['summary']['missing_dependencies']}개 필수 의존성 누락")
            
            return validation_result
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"의존성 검증 (상세) 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'details': {},
                'summary': {'total_dependencies': 0, 'valid_dependencies': 0, 'missing_dependencies': 0, 'optional_dependencies': 0},
                'recommendations': ['의존성 검증 중 오류 발생']
            }

    def _validate_data_conversion_readiness(self) -> bool:
        """데이터 변환 준비 상태 검증"""
        try:
            # DetailedDataSpec 로드 확인
            if not hasattr(self, 'detailed_data_spec'):
                return False
            
            # API 매핑 설정 확인
            if not hasattr(self, 'api_input_mapping') or not hasattr(self, 'api_output_mapping'):
                return False
            
            # 기본 변환 메서드 확인
            required_methods = [
                'convert_api_input_to_step_input',
                'convert_step_output_to_api_response'
            ]
            
            for method_name in required_methods:
                if not hasattr(self, method_name):
                    return False
            
            return True
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"데이터 변환 준비 상태 검증 실패: {e}")
            return False

    def _validate_input_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """입력 데이터 검증"""
        try:
            validation_result = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'validated_data': input_data.copy()
            }
            
            # 기본 데이터 타입 검증
            if not isinstance(input_data, dict):
                validation_result['valid'] = False
                validation_result['errors'].append("입력 데이터는 딕셔너리여야 합니다")
                return validation_result
            
            # 필수 필드 검증
            required_fields = getattr(self, 'preprocessing_required', [])
            for field in required_fields:
                if field not in input_data:
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"필수 필드 '{field}'가 누락되었습니다")
                elif input_data[field] is None:
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"필수 필드 '{field}'가 None입니다")
            
            # 데이터 타입 검증
            if hasattr(self, 'input_data_types'):
                for field, expected_type in self.input_data_types.items():
                    if field in input_data:
                        actual_type = type(input_data[field]).__name__
                        if not self._is_compatible_type(input_data[field], expected_type):
                            validation_result['warnings'].append(
                                f"필드 '{field}'의 타입이 예상과 다릅니다 (예상: {expected_type}, 실제: {actual_type})"
                            )
            
            # 검증 결과 로깅
            if hasattr(self, 'logger'):
                if validation_result['valid']:
                    if validation_result['warnings']:
                        self.logger.warning(f"입력 데이터 검증 완료 (경고: {len(validation_result['warnings'])}개)")
                    else:
                        self.logger.debug("입력 데이터 검증 완료")
                else:
                    self.logger.error(f"입력 데이터 검증 실패: {len(validation_result['errors'])}개 오류")
            
            return validation_result
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"입력 데이터 검증 실패: {e}")
            return {
                'valid': False,
                'errors': [f"검증 중 오류 발생: {str(e)}"],
                'warnings': [],
                'validated_data': input_data
            }

    def _is_compatible_type(self, value: Any, expected_type: str) -> bool:
        """타입 호환성 검사"""
        try:
            if expected_type == "image":
                # 이미지 타입 검사 (numpy array, PIL Image, base64 string 등)
                import numpy as np
                from PIL import Image
                
                if isinstance(value, np.ndarray):
                    return True
                elif isinstance(value, Image.Image):
                    return True
                elif isinstance(value, str) and value.startswith('data:image'):
                    return True
                elif isinstance(value, str) and len(value) > 100:  # base64 문자열 추정
                    return True
                return False
                
            elif expected_type == "float":
                try:
                    float(value)
                    return True
                except (ValueError, TypeError):
                    return False
                    
            elif expected_type == "int":
                try:
                    int(value)
                    return True
                except (ValueError, TypeError):
                    return False
                    
            elif expected_type == "list":
                return isinstance(value, (list, tuple))
                
            elif expected_type == "dict":
                return isinstance(value, dict)
                
            else:
                return True
                
        except Exception:
            return False

    def validate_step_environment(self) -> Dict[str, Any]:
        """Step 환경 검증"""
        try:
            validation_result = {
                'success': True,
                'environment': {},
                'issues': []
            }
            
            # Python 버전 확인
            import sys
            validation_result['environment']['python_version'] = sys.version
            
            # PyTorch 사용 가능 여부
            try:
                import torch
                validation_result['environment']['torch_available'] = True
                validation_result['environment']['torch_version'] = torch.__version__
                validation_result['environment']['cuda_available'] = torch.cuda.is_available()
                validation_result['environment']['mps_available'] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            except ImportError:
                validation_result['environment']['torch_available'] = False
                validation_result['issues'].append("PyTorch가 설치되지 않았습니다")
            
            # NumPy 사용 가능 여부
            try:
                import numpy as np
                validation_result['environment']['numpy_available'] = True
                validation_result['environment']['numpy_version'] = np.__version__
            except ImportError:
                validation_result['environment']['numpy_available'] = False
                validation_result['issues'].append("NumPy가 설치되지 않았습니다")
            
            # PIL 사용 가능 여부
            try:
                from PIL import Image
                validation_result['environment']['pil_available'] = True
                validation_result['environment']['pil_version'] = Image.__version__
            except ImportError:
                validation_result['environment']['pil_available'] = False
                validation_result['issues'].append("PIL이 설치되지 않았습니다")
            
            # 디바이스 설정 확인
            device = getattr(self, 'device', 'cpu')
            validation_result['environment']['device'] = device
            
            if device == 'mps' and not validation_result['environment'].get('mps_available', False):
                validation_result['issues'].append("MPS 디바이스가 사용 불가능합니다")
                validation_result['success'] = False
            
            # 메모리 사용량 확인
            try:
                import psutil
                memory_info = psutil.virtual_memory()
                validation_result['environment']['total_memory_gb'] = memory_info.total / (1024**3)
                validation_result['environment']['available_memory_gb'] = memory_info.available / (1024**3)
                
                if memory_info.available < 1024**3:  # 1GB 미만
                    validation_result['issues'].append("사용 가능한 메모리가 부족합니다 (1GB 미만)")
                    validation_result['success'] = False
                    
            except ImportError:
                validation_result['environment']['memory_info'] = "psutil 없음"
            
            # 검증 결과 로깅
            if hasattr(self, 'logger'):
                if validation_result['success']:
                    self.logger.info("✅ Step 환경 검증 완료")
                else:
                    self.logger.warning(f"⚠️ Step 환경 검증 실패: {len(validation_result['issues'])}개 이슈")
            
            return validation_result
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Step 환경 검증 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'environment': {},
                'issues': ['환경 검증 중 오류 발생']
            }
