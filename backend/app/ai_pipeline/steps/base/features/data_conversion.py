#!/usr/bin/env python3
"""
🔥 MyCloset AI - Data Conversion Mixin
======================================

데이터 변환 관련 기능을 담당하는 Mixin 클래스
API ↔ AI 모델 간 데이터 변환을 담당

Author: MyCloset AI Team
Date: 2025-08-14
Version: 2.0
"""

import logging
import base64
from typing import Dict, Any, Optional, List, Tuple, Union
from io import BytesIO

# NumPy 선택적 import
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

class DataConversionMixin:
    """데이터 변환 관련 기능을 제공하는 Mixin"""
    
    def _inject_detailed_data_spec_attributes(self, kwargs: Dict[str, Any]):
        """DetailedDataSpec 속성 자동 주입"""
        # ✅ API 매핑 속성 주입
        self.api_input_mapping = kwargs.get('api_input_mapping', {})
        self.api_output_mapping = kwargs.get('api_output_mapping', {})
        
        # ✅ Step 간 데이터 흐름 속성 주입  
        self.accepts_from_previous_step = kwargs.get('accepts_from_previous_step', {})
        self.provides_to_next_step = kwargs.get('provides_to_next_step', {})
        
        # ✅ 전처리/후처리 속성 주입
        self.preprocessing_steps = kwargs.get('preprocessing_steps', [])
        self.postprocessing_steps = kwargs.get('postprocessing_steps', [])
        self.preprocessing_required = kwargs.get('preprocessing_required', [])
        self.postprocessing_required = kwargs.get('postprocessing_required', [])
        
        # ✅ 데이터 타입 및 스키마 속성 주입
        self.input_data_types = kwargs.get('input_data_types', [])
        self.output_data_types = kwargs.get('output_data_types', [])
        self.step_input_schema = kwargs.get('step_input_schema', {})
        self.step_output_schema = kwargs.get('step_output_schema', {})
        
        # ✅ 정규화 파라미터 주입
        self.normalization_mean = kwargs.get('normalization_mean', (0.485, 0.456, 0.406))
        self.normalization_std = kwargs.get('normalization_std', (0.229, 0.224, 0.225))
        
        # ✅ 메타정보 주입
        self.detailed_data_spec_loaded = kwargs.get('detailed_data_spec_loaded', True)
        self.detailed_data_spec_version = kwargs.get('detailed_data_spec_version', 'v11.2')
        self.step_model_requirements_integrated = kwargs.get('step_model_requirements_integrated', True)
        self.central_hub_integrated = kwargs.get('central_hub_integrated', True)
        
        # ✅ FastAPI 호환성 플래그
        self.fastapi_compatible = len(self.api_input_mapping) > 0
        
        if hasattr(self, 'logger'):
            self.logger.debug(f"✅ {getattr(self, 'step_name', 'Unknown')} DetailedDataSpec 속성 주입 완료")

    async def convert_api_input_to_step_input(self, api_input: Dict[str, Any]) -> Dict[str, Any]:
        """API 입력을 Step 입력으로 변환 - 비동기 버전"""
        if not hasattr(self, 'api_input_mapping') or not self.api_input_mapping:
            # 매핑이 없으면 그대로 반환
            if hasattr(self, 'logger'):
                self.logger.debug(f"{getattr(self, 'step_name', 'Unknown')} API 매핑 없음, 원본 반환")
            return api_input
        
        converted = {}
        
        # ✅ API 매핑 기반 변환
        for api_param, api_type in self.api_input_mapping.items():
            if api_param in api_input:
                converted_value = await self._convert_api_input_type(
                    api_input[api_param], api_type, api_param
                )
                converted[api_param] = converted_value
        
        # ✅ 누락된 필수 입력 데이터 확인
        for param_name in self.api_input_mapping.keys():
            if param_name not in converted and param_name in api_input:
                converted[param_name] = api_input[param_name]
        
        if hasattr(self, 'logger'):
            self.logger.debug(f"✅ {getattr(self, 'step_name', 'Unknown')} API → Step 변환 완료")
        return converted

    def convert_api_input_to_step_input(self, api_input: Dict[str, Any]) -> Dict[str, Any]:
        """API 입력을 Step 입력으로 변환 (kwargs 방식)"""
        try:
            step_input = api_input.copy()
            
            # 🔥 kwargs에서 데이터 직접 가져오기 (세션 의존성 제거)
            if not hasattr(self, 'api_input_mapping') or not self.api_input_mapping:
                # 매핑이 없으면 kwargs 그대로 반환
                if hasattr(self, 'logger'):
                    self.logger.debug(f"{getattr(self, 'step_name', 'Unknown')} API 매핑 없음, kwargs 그대로 반환")
                return step_input
            
            converted = {}
            
            # ✅ API 매핑 기반 변환 (kwargs 방식)
            for api_param, api_type in self.api_input_mapping.items():
                if api_param in step_input:
                    converted_value = self._convert_api_input_type_sync(
                        step_input[api_param], api_type, api_param
                    )
                    converted[api_param] = converted_value
                else:
                    # kwargs에 없으면 기본값 사용
                    if hasattr(self, 'logger'):
                        self.logger.debug(f"ℹ️ {api_param}가 kwargs에 없음 - 기본값 사용")
            
            # ✅ 누락된 필수 입력 데이터 확인
            for param_name in self.api_input_mapping.keys():
                if param_name not in converted and param_name in step_input:
                    converted[param_name] = step_input[param_name]
            
            if hasattr(self, 'logger'):
                self.logger.debug(f"✅ {getattr(self, 'step_name', 'Unknown')} kwargs → Step 변환 완료")
            return converted
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"❌ {getattr(self, 'step_name', 'Unknown')} API 입력 변환 실패: {e}")
            return api_input

    def convert_step_output_to_api_response(self, step_output: Dict[str, Any]) -> Dict[str, Any]:
        """Step 출력을 API 응답으로 변환 - 활성화"""
        if not hasattr(self, 'api_output_mapping') or not self.api_output_mapping:
            # 매핑이 없으면 그대로 반환
            return step_output
        
        api_response = {}
        
        # ✅ API 출력 매핑 기반 변환
        for step_key, api_type in self.api_output_mapping.items():
            if step_key in step_output:
                converted_value = self._convert_step_output_type_sync(
                    step_output[step_key], api_type, step_key
                )
                api_response[step_key] = converted_value
        
        # ✅ 메타데이터 추가
        api_response.update({
            'step_name': getattr(self, 'step_name', 'Unknown'),
            'processing_time': step_output.get('processing_time', 0),
            'confidence': step_output.get('confidence', 0.95),
            'success': step_output.get('success', True)
        })
        
        if hasattr(self, 'logger'):
            self.logger.debug(f"✅ {getattr(self, 'step_name', 'Unknown')} Step → API 변환 완료")
        return api_response

    def _convert_step_output_type_sync(self, value: Any, api_type: str, param_name: str) -> Any:
        """Step 출력 타입을 API 타입으로 변환 (동기 버전)"""
        if api_type == "base64_string":
            return self._array_to_base64(value)
        elif api_type == "List[Dict]":
            return self._convert_to_list_dict(value)
        elif api_type == "List[Dict[str, float]]":
            return self._convert_keypoints_to_dict_list(value)
        elif api_type == "float":
            return float(value) if value is not None else 0.0
        elif api_type == "List[float]":
            if isinstance(value, (list, tuple)):
                return [float(x) for x in value]
            elif hasattr(np, 'ndarray') and isinstance(value, np.ndarray):
                return value.flatten().tolist()
            else:
                return [float(value)] if value is not None else []
        else:
            return value

    def _array_to_base64(self, array: Any) -> str:
        """배열을 base64 문자열로 변환"""
        try:
            if NUMPY_AVAILABLE and hasattr(np, 'ndarray') and isinstance(array, np.ndarray):
                # numpy 배열을 PIL Image로 변환
                if array.dtype == np.uint8:
                    from PIL import Image
                    image = Image.fromarray(array)
                else:
                    # float 배열을 uint8로 변환
                    array = (array * 255).astype(np.uint8)
                    from PIL import Image
                    image = Image.fromarray(array)
                
                # PIL Image를 base64로 변환
                buffer = BytesIO()
                image.save(buffer, format='PNG')
                buffer.seek(0)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
            else:
                return str(array)
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"배열을 base64로 변환 실패: {e}")
            return str(array)

    def _convert_to_list_dict(self, value: Any) -> List[Dict]:
        """값을 List[Dict] 형태로 변환"""
        try:
            if isinstance(value, list):
                if all(isinstance(item, dict) for item in value):
                    return value
                else:
                    return [{'value': item} for item in value]
            elif isinstance(value, dict):
                return [value]
            else:
                return [{'value': value}]
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"List[Dict] 변환 실패: {e}")
            return [{'value': value}]

    def _convert_keypoints_to_dict_list(self, keypoints: Any) -> List[Dict[str, float]]:
        """키포인트를 List[Dict[str, float]] 형태로 변환"""
        try:
            if isinstance(keypoints, list):
                if all(isinstance(item, dict) for item in keypoints):
                    return keypoints
                else:
                    return [{'x': float(item[0]), 'y': float(item[1])} if len(item) >= 2 else {'value': float(item[0])} for item in keypoints]
            elif hasattr(np, 'ndarray') and isinstance(keypoints, np.ndarray):
                if keypoints.ndim == 2:
                    return [{'x': float(kp[0]), 'y': float(kp[1])} for kp in keypoints]
                else:
                    return [{'value': float(kp)} for kp in keypoints]
            else:
                return [{'value': float(keypoints)}]
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"키포인트 변환 실패: {e}")
            return [{'value': 0.0}]

    async def _convert_api_input_type(self, value: Any, api_type: str, param_name: str) -> Any:
        """API 입력 타입을 Step 타입으로 변환 (비동기 버전)"""
        # 기본적으로 동기 버전 사용
        return self._convert_api_input_type_sync(value, api_type, param_name)

    def _convert_api_input_type_sync(self, value: Any, api_type: str, param_name: str) -> Any:
        """API 입력 타입을 Step 타입으로 변환 (동기 버전)"""
        try:
            if api_type == "base64_string":
                # base64 문자열을 numpy 배열로 변환
                if isinstance(value, str):
                    image_data = base64.b64decode(value)
                    from PIL import Image
                    image = Image.open(BytesIO(image_data))
                    return np.array(image)
                return value
            elif api_type == "List[Dict]":
                return self._convert_to_list_dict(value)
            elif api_type == "List[Dict[str, float]]":
                return self._convert_keypoints_to_dict_list(value)
            elif api_type == "float":
                return float(value) if value is not None else 0.0
            elif api_type == "List[float]":
                if isinstance(value, (list, tuple)):
                    return [float(x) for x in value]
                else:
                    return [float(value)] if value is not None else []
            else:
                return value
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"API 입력 타입 변환 실패: {e}")
            return value
