"""
🔥 입력 검증기
==============

후처리를 위한 입력 검증 시스템:
1. 입력 데이터 검증
2. 이미지 품질 검사
3. 데이터 형식 변환
4. 전처리 파이프라인

Author: MyCloset AI Team
Date: 2025-08-14
Version: 1.0
"""

import logging
import numpy as np
import cv2
from typing import Dict, Any, Optional, List, Tuple, Union
import os

logger = logging.getLogger(__name__)

class InputValidator:
    """입력 검증기"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.InputValidator")
        
        # 검증 통계
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'validation_errors': []
        }
        
        # 검증 규칙
        self.validation_rules = {
            'min_image_size': (64, 64),
            'max_image_size': (4096, 4096),
            'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
            'min_channels': 1,
            'max_channels': 4,
            'data_type': np.uint8
        }
    
    def validate_input(self, input_data: Union[np.ndarray, str, Dict[str, Any]]) -> Dict[str, Any]:
        """입력 데이터 검증"""
        try:
            self.logger.info("🔍 입력 데이터 검증 시작")
            
            validation_result = {
                'is_valid': False,
                'errors': [],
                'warnings': [],
                'validated_data': None,
                'validation_info': {}
            }
            
            # 입력 타입에 따른 검증
            if isinstance(input_data, np.ndarray):
                result = self._validate_numpy_array(input_data)
            elif isinstance(input_data, str):
                result = self._validate_file_path(input_data)
            elif isinstance(input_data, dict):
                result = self._validate_dictionary(input_data)
            else:
                validation_result['errors'].append(f"지원하지 않는 입력 타입: {type(input_data)}")
                return validation_result
            
            # 검증 결과 병합
            validation_result.update(result)
            
            # 검증 통계 업데이트
            self._update_validation_stats(validation_result['is_valid'])
            
            if validation_result['is_valid']:
                self.logger.info("✅ 입력 데이터 검증 완료")
            else:
                self.logger.warning(f"⚠️ 입력 데이터 검증 실패: {len(validation_result['errors'])}개 오류")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ 입력 데이터 검증 실패: {e}")
            return {
                'is_valid': False,
                'errors': [f'검증 중 오류 발생: {e}'],
                'warnings': [],
                'validated_data': None,
                'validation_info': {}
            }
    
    def _validate_numpy_array(self, image: np.ndarray) -> Dict[str, Any]:
        """NumPy 배열 검증"""
        try:
            validation_result = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'validated_data': image,
                'validation_info': {}
            }
            
            # 차원 검증
            if image.ndim < 2 or image.ndim > 3:
                validation_result['errors'].append(f"이미지 차원이 잘못되었습니다: {image.ndim}D (2D 또는 3D 필요)")
                validation_result['is_valid'] = False
                return validation_result
            
            # 크기 검증
            height, width = image.shape[:2]
            min_h, min_w = self.validation_rules['min_image_size']
            max_h, max_w = self.validation_rules['max_image_size']
            
            if height < min_h or width < min_w:
                validation_result['errors'].append(f"이미지 크기가 너무 작습니다: {width}x{height} (최소 {min_w}x{min_h} 필요)")
                validation_result['is_valid'] = False
            
            if height > max_h or width > max_w:
                validation_result['warnings'].append(f"이미지 크기가 매우 큽니다: {width}x{height} (최대 {max_w}x{max_h} 권장)")
            
            # 채널 수 검증
            if image.ndim == 3:
                channels = image.shape[2]
                min_channels = self.validation_rules['min_channels']
                max_channels = self.validation_rules['max_channels']
                
                if channels < min_channels or channels > max_channels:
                    validation_result['errors'].append(f"채널 수가 잘못되었습니다: {channels} (1~4 필요)")
                    validation_result['is_valid'] = False
            
            # 데이터 타입 검증
            if image.dtype != self.validation_rules['data_type']:
                validation_result['warnings'].append(f"데이터 타입이 예상과 다릅니다: {image.dtype} (권장: {self.validation_rules['data_type']})")
                
                # 데이터 타입 변환
                try:
                    converted_image = image.astype(self.validation_rules['data_type'])
                    validation_result['validated_data'] = converted_image
                    validation_result['warnings'].append("데이터 타입을 자동으로 변환했습니다")
                except Exception as e:
                    validation_result['errors'].append(f"데이터 타입 변환 실패: {e}")
                    validation_result['is_valid'] = False
            
            # 이미지 품질 검사
            quality_info = self._check_image_quality(image)
            validation_result['validation_info']['quality'] = quality_info
            
            # 품질 경고 추가
            if quality_info.get('is_dark', False):
                validation_result['warnings'].append("이미지가 너무 어둡습니다")
            
            if quality_info.get('is_bright', False):
                validation_result['warnings'].append("이미지가 너무 밝습니다")
            
            if quality_info.get('is_blurry', False):
                validation_result['warnings'].append("이미지가 흐릿합니다")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ NumPy 배열 검증 실패: {e}")
            return {
                'is_valid': False,
                'errors': [f'NumPy 배열 검증 실패: {e}'],
                'warnings': [],
                'validated_data': None,
                'validation_info': {}
            }
    
    def _validate_file_path(self, file_path: str) -> Dict[str, Any]:
        """파일 경로 검증"""
        try:
            validation_result = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'validated_data': None,
                'validation_info': {}
            }
            
            # 파일 존재 여부 확인
            if not os.path.exists(file_path):
                validation_result['errors'].append(f"파일이 존재하지 않습니다: {file_path}")
                validation_result['is_valid'] = False
                return validation_result
            
            # 파일 확장자 검증
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in self.validation_rules['supported_formats']:
                validation_result['errors'].append(f"지원하지 않는 파일 형식: {file_ext}")
                validation_result['is_valid'] = False
                return validation_result
            
            # 파일 크기 확인
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                validation_result['errors'].append("파일이 비어있습니다")
                validation_result['is_valid'] = False
                return validation_result
            
            # 이미지 로드 시도
            try:
                image = cv2.imread(file_path)
                if image is None:
                    validation_result['errors'].append("이미지 파일을 로드할 수 없습니다")
                    validation_result['is_valid'] = False
                    return validation_result
                
                # 로드된 이미지 검증
                image_validation = self._validate_numpy_array(image)
                validation_result.update(image_validation)
                
                # 파일 정보 추가
                validation_result['validation_info']['file_info'] = {
                    'path': file_path,
                    'size': file_size,
                    'extension': file_ext
                }
                
            except Exception as e:
                validation_result['errors'].append(f"이미지 로드 실패: {e}")
                validation_result['is_valid'] = False
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ 파일 경로 검증 실패: {e}")
            return {
                'is_valid': False,
                'errors': [f'파일 경로 검증 실패: {e}'],
                'warnings': [],
                'validated_data': None,
                'validation_info': {}
            }
    
    def _validate_dictionary(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """딕셔너리 검증"""
        try:
            validation_result = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'validated_data': None,
                'validation_info': {}
            }
            
            # 필수 키 확인
            required_keys = ['image']
            for key in required_keys:
                if key not in data_dict:
                    validation_result['errors'].append(f"필수 키가 누락되었습니다: {key}")
                    validation_result['is_valid'] = False
            
            if not validation_result['is_valid']:
                return validation_result
            
            # 이미지 데이터 검증
            image_data = data_dict['image']
            if isinstance(image_data, np.ndarray):
                image_validation = self._validate_numpy_array(image_data)
                validation_result.update(image_validation)
            elif isinstance(image_data, str):
                image_validation = self._validate_file_path(image_data)
                validation_result.update(image_validation)
            else:
                validation_result['errors'].append(f"이미지 데이터 타입이 잘못되었습니다: {type(image_data)}")
                validation_result['is_valid'] = False
            
            # 추가 메타데이터 검증
            if 'metadata' in data_dict:
                metadata = data_dict['metadata']
                if isinstance(metadata, dict):
                    validation_result['validation_info']['metadata'] = metadata
                else:
                    validation_result['warnings'].append("메타데이터가 딕셔너리가 아닙니다")
            
            # 설정 검증
            if 'config' in data_dict:
                config = data_dict['config']
                if isinstance(config, dict):
                    config_validation = self._validate_config(config)
                    validation_result['validation_info']['config'] = config_validation
                else:
                    validation_result['warnings'].append("설정이 딕셔너리가 아닙니다")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ 딕셔너리 검증 실패: {e}")
            return {
                'is_valid': False,
                'errors': [f'딕셔너리 검증 실패: {e}'],
                'warnings': [],
                'validated_data': None,
                'validation_info': {}
            }
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """설정 검증"""
        try:
            config_validation = {
                'is_valid': True,
                'errors': [],
                'warnings': []
            }
            
            # 품질 향상 설정 검증
            if 'enhancement' in config:
                enhancement = config['enhancement']
                if isinstance(enhancement, dict):
                    # 향상 타입 검증
                    if 'type' in enhancement:
                        valid_types = ['comprehensive', 'noise_reduction', 'sharpness', 'contrast']
                        if enhancement['type'] not in valid_types:
                            config_validation['warnings'].append(f"알 수 없는 향상 타입: {enhancement['type']}")
                    
                    # 파라미터 검증
                    for param, value in enhancement.items():
                        if param != 'type':
                            if not isinstance(value, (int, float)):
                                config_validation['warnings'].append(f"향상 파라미터 {param}이 숫자가 아닙니다: {value}")
                else:
                    config_validation['warnings'].append("향상 설정이 딕셔너리가 아닙니다")
            
            # 출력 설정 검증
            if 'output' in config:
                output = config['output']
                if isinstance(output, dict):
                    # 크기 설정 검증
                    if 'resize' in output:
                        resize = output['resize']
                        if isinstance(resize, dict):
                            if 'width' in resize and 'height' in resize:
                                width = resize['width']
                                height = resize['height']
                                if not (isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0):
                                    config_validation['errors'].append("크기 설정이 잘못되었습니다")
                                    config_validation['is_valid'] = False
                    
                    # 품질 설정 검증
                    if 'quality' in output:
                        quality = output['quality']
                        if not (isinstance(quality, (int, float)) and 0 <= quality <= 100):
                            config_validation['warnings'].append("품질 설정이 0~100 범위를 벗어납니다")
                else:
                    config_validation['warnings'].append("출력 설정이 딕셔너리가 아닙니다")
            
            return config_validation
            
        except Exception as e:
            self.logger.error(f"❌ 설정 검증 실패: {e}")
            return {
                'is_valid': False,
                'errors': [f'설정 검증 실패: {e}'],
                'warnings': []
            }
    
    def _check_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """이미지 품질 검사"""
        try:
            quality_info = {
                'is_dark': False,
                'is_bright': False,
                'is_blurry': False,
                'contrast_level': 'normal',
                'noise_level': 'low'
            }
            
            # 그레이스케일 변환
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 밝기 검사
            mean_brightness = np.mean(gray)
            if mean_brightness < 50:
                quality_info['is_dark'] = True
            elif mean_brightness > 200:
                quality_info['is_bright'] = True
            
            # 대비 검사
            contrast = np.std(gray)
            if contrast < 30:
                quality_info['contrast_level'] = 'low'
            elif contrast > 80:
                quality_info['contrast_level'] = 'high'
            
            # 선명도 검사 (흐림 정도)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = np.var(laplacian)
            if sharpness < 100:
                quality_info['is_blurry'] = True
            
            # 노이즈 검사 (간단한 버전)
            # 작은 커널로 블러링한 후 원본과의 차이로 노이즈 추정
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            noise = np.mean(np.abs(gray.astype(float) - blurred.astype(float)))
            if noise > 15:
                quality_info['noise_level'] = 'high'
            elif noise > 8:
                quality_info['noise_level'] = 'medium'
            
            return quality_info
            
        except Exception as e:
            self.logger.error(f"❌ 이미지 품질 검사 실패: {e}")
            return {
                'is_dark': False,
                'is_bright': False,
                'is_blurry': False,
                'contrast_level': 'unknown',
                'noise_level': 'unknown'
            }
    
    def preprocess_input(self, input_data: Union[np.ndarray, str, Dict[str, Any]]) -> Optional[np.ndarray]:
        """입력 데이터 전처리"""
        try:
            self.logger.info("🚀 입력 데이터 전처리 시작")
            
            # 입력 검증
            validation_result = self.validate_input(input_data)
            
            if not validation_result['is_valid']:
                self.logger.error(f"❌ 입력 검증 실패: {validation_result['errors']}")
                return None
            
            # 검증된 데이터 추출
            validated_data = validation_result['validated_data']
            
            if isinstance(validated_data, np.ndarray):
                # 이미지 전처리
                preprocessed = self._preprocess_image(validated_data)
                self.logger.info("✅ 입력 데이터 전처리 완료")
                return preprocessed
            else:
                self.logger.error("❌ 검증된 데이터가 이미지가 아닙니다")
                return None
            
        except Exception as e:
            self.logger.error(f"❌ 입력 데이터 전처리 실패: {e}")
            return None
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리"""
        try:
            preprocessed = image.copy()
            
            # 크기 정규화 (너무 큰 이미지 축소)
            height, width = preprocessed.shape[:2]
            max_size = 2048
            
            if height > max_size or width > max_size:
                scale = max_size / max(height, width)
                new_height = int(height * scale)
                new_width = int(width * scale)
                preprocessed = cv2.resize(preprocessed, (new_width, new_height))
                self.logger.info(f"📏 이미지 크기 정규화: {width}x{height} -> {new_width}x{new_height}")
            
            # 데이터 타입 정규화
            if preprocessed.dtype != self.validation_rules['data_type']:
                preprocessed = preprocessed.astype(self.validation_rules['data_type'])
            
            return preprocessed
            
        except Exception as e:
            self.logger.error(f"❌ 이미지 전처리 실패: {e}")
            return image
    
    def set_validation_rules(self, rules: Dict[str, Any]):
        """검증 규칙 설정"""
        try:
            for key, value in rules.items():
                if key in self.validation_rules:
                    self.validation_rules[key] = value
                    self.logger.info(f"✅ {key} 규칙 설정: {value}")
                else:
                    self.logger.warning(f"⚠️ 알 수 없는 규칙: {key}")
                    
        except Exception as e:
            self.logger.error(f"❌ 검증 규칙 설정 실패: {e}")
    
    def get_validation_rules(self) -> Dict[str, Any]:
        """검증 규칙 반환"""
        return self.validation_rules.copy()
    
    def _update_validation_stats(self, success: bool):
        """검증 통계 업데이트"""
        try:
            self.validation_stats['total_validations'] += 1
            
            if success:
                self.validation_stats['successful_validations'] += 1
            else:
                self.validation_stats['failed_validations'] += 1
                
        except Exception as e:
            self.logger.error(f"❌ 검증 통계 업데이트 실패: {e}")
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """검증 통계 반환"""
        return self.validation_stats.copy()
    
    def reset_validation_stats(self):
        """검증 통계 초기화"""
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'validation_errors': []
        }
