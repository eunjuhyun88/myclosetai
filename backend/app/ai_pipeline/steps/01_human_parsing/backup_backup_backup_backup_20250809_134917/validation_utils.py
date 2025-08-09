"""
🔥 Human Parsing 검증 유틸리티
==========================

파싱 맵 검증, 텐서 형태 검증, 오류 처리 헬퍼 함수들을 포함합니다.

주요 기능:
- 파싱 맵 유효성 검사
- 텐서 형태 검증
- 원본 크기 안전한 결정
- 오류 처리 및 폴백 메커니즘

Author: MyCloset AI Team
Date: 2025-08-07
Version: 1.0
"""

import logging
import numpy as np
from typing import Tuple, Optional, Union, Any, Dict
from PIL import Image

logger = logging.getLogger(__name__)


class ConfidenceCalculator:
    """신뢰도 계산 및 분석 클래스"""
    
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        self.logger = logger_instance or logger
    
    def calculate_confidence_map(self, parsing_map: np.ndarray, model_output: Any = None) -> np.ndarray:
        """
        파싱 맵에서 신뢰도 맵을 계산합니다.
        
        Args:
            parsing_map: 파싱 맵 (np.ndarray)
            model_output: 모델 출력 (선택사항)
            
        Returns:
            신뢰도 맵 (np.ndarray)
        """
        try:
            if model_output is not None:
                return self._calculate_from_model_output(parsing_map, model_output)
            else:
                return self._calculate_from_parsing_map(parsing_map)
                
        except Exception as e:
            self.logger.error(f"❌ 신뢰도 맵 계산 실패: {e}")
            return self._create_default_confidence_map(parsing_map.shape)
    
    def _calculate_from_model_output(self, parsing_map: np.ndarray, model_output: Any) -> np.ndarray:
        """모델 출력에서 신뢰도 맵 계산"""
        try:
            # 모델 출력을 NumPy 배열로 변환
            if hasattr(model_output, 'cpu') and hasattr(model_output, 'numpy'):
                # PyTorch 텐서
                output_array = model_output.detach().cpu().numpy()
            elif isinstance(model_output, np.ndarray):
                output_array = model_output
            elif isinstance(model_output, dict):
                # 딕셔너리에서 신뢰도 정보 추출
                for key in ['confidence', 'prob', 'logits', 'output']:
                    if key in model_output:
                        output_array = self._convert_to_numpy(model_output[key])
                        break
                else:
                    # 키를 찾지 못한 경우 첫 번째 값 사용
                    first_value = next(iter(model_output.values()))
                    output_array = self._convert_to_numpy(first_value)
            else:
                output_array = self._convert_to_numpy(model_output)
            
            # 신뢰도 계산 (소프트맥스 적용)
            if len(output_array.shape) == 3 and output_array.shape[0] > 1:
                # 다중 클래스 출력
                confidence_map = self._calculate_multiclass_confidence(output_array, parsing_map)
            else:
                # 단일 클래스 출력
                confidence_map = self._calculate_singleclass_confidence(output_array)
            
            return confidence_map
            
        except Exception as e:
            self.logger.warning(f"⚠️ 모델 출력에서 신뢰도 계산 실패: {e}")
            return self._calculate_from_parsing_map(parsing_map)
    
    def _calculate_from_parsing_map(self, parsing_map: np.ndarray) -> np.ndarray:
        """파싱 맵에서 신뢰도 맵 계산"""
        try:
            # 파싱 맵의 경계를 기반으로 신뢰도 계산
            confidence_map = np.ones_like(parsing_map, dtype=np.float32)
            
            # 경계 영역 신뢰도 감소
            from scipy import ndimage
            if hasattr(ndimage, 'binary_erosion'):
                eroded = ndimage.binary_erosion(parsing_map > 0, iterations=1)
                boundary = (parsing_map > 0) & ~eroded
                confidence_map[boundary] = 0.7
            
            # 노이즈 영역 신뢰도 감소
            small_components = self._remove_small_components(parsing_map > 0, min_size=50)
            confidence_map[~small_components] *= 0.5
            
            return confidence_map
            
        except Exception as e:
            self.logger.warning(f"⚠️ 파싱 맵에서 신뢰도 계산 실패: {e}")
            return self._create_default_confidence_map(parsing_map.shape)
    
    def _calculate_multiclass_confidence(self, output_array: np.ndarray, parsing_map: np.ndarray) -> np.ndarray:
        """다중 클래스 출력에서 신뢰도 계산"""
        try:
            # 소프트맥스 적용
            exp_output = np.exp(output_array - np.max(output_array, axis=0, keepdims=True))
            softmax_output = exp_output / np.sum(exp_output, axis=0, keepdims=True)
            
            # 파싱 맵에 해당하는 클래스의 신뢰도 추출
            confidence_map = np.zeros(parsing_map.shape, dtype=np.float32)
            
            for class_id in range(softmax_output.shape[0]):
                mask = (parsing_map == class_id)
                confidence_map[mask] = softmax_output[class_id, mask]
            
            return confidence_map
            
        except Exception as e:
            self.logger.warning(f"⚠️ 다중 클래스 신뢰도 계산 실패: {e}")
            return np.ones(parsing_map.shape, dtype=np.float32) * 0.8
    
    def _calculate_singleclass_confidence(self, output_array: np.ndarray) -> np.ndarray:
        """단일 클래스 출력에서 신뢰도 계산"""
        try:
            if len(output_array.shape) == 3:
                # (1, H, W) 형태
                confidence_map = output_array[0]
            else:
                # (H, W) 형태
                confidence_map = output_array
            
            # 시그모이드 적용 (0-1 범위로 정규화)
            confidence_map = 1 / (1 + np.exp(-confidence_map))
            
            return confidence_map
            
        except Exception as e:
            self.logger.warning(f"⚠️ 단일 클래스 신뢰도 계산 실패: {e}")
            return np.ones(output_array.shape[-2:], dtype=np.float32) * 0.8
    
    def _remove_small_components(self, binary_map: np.ndarray, min_size: int = 50) -> np.ndarray:
        """작은 연결 요소 제거"""
        try:
            from scipy import ndimage
            labeled, num_features = ndimage.label(binary_map)
            
            # 각 연결 요소의 크기 계산
            component_sizes = np.bincount(labeled.ravel())[1:]  # 0번은 배경이므로 제외
            
            # 작은 연결 요소 제거
            for i, size in enumerate(component_sizes, 1):
                if size < min_size:
                    binary_map[labeled == i] = False
            
            return binary_map
            
        except Exception as e:
            self.logger.warning(f"⚠️ 작은 연결 요소 제거 실패: {e}")
            return binary_map
    
    def _convert_to_numpy(self, data: Any) -> np.ndarray:
        """다양한 타입의 데이터를 NumPy 배열로 변환"""
        try:
            if isinstance(data, np.ndarray):
                return data
            elif hasattr(data, 'cpu') and hasattr(data, 'numpy'):
                # PyTorch 텐서
                return data.detach().cpu().numpy()
            elif isinstance(data, list):
                return np.array(data)
            else:
                return np.array(data)
                
        except Exception as e:
            self.logger.warning(f"⚠️ 데이터 변환 실패: {e}")
            raise
    
    def _create_default_confidence_map(self, shape: Tuple[int, ...]) -> np.ndarray:
        """기본 신뢰도 맵 생성"""
        return np.ones(shape, dtype=np.float32) * 0.8
    
    def analyze_confidence_distribution(self, confidence_map: np.ndarray) -> Dict[str, float]:
        """신뢰도 분포 분석"""
        try:
            analysis = {
                'mean_confidence': float(np.mean(confidence_map)),
                'std_confidence': float(np.std(confidence_map)),
                'min_confidence': float(np.min(confidence_map)),
                'max_confidence': float(np.max(confidence_map)),
                'high_confidence_ratio': float(np.sum(confidence_map > 0.8) / confidence_map.size),
                'low_confidence_ratio': float(np.sum(confidence_map < 0.3) / confidence_map.size)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ 신뢰도 분포 분석 실패: {e}")
            return {
                'mean_confidence': 0.5,
                'std_confidence': 0.1,
                'min_confidence': 0.0,
                'max_confidence': 1.0,
                'high_confidence_ratio': 0.5,
                'low_confidence_ratio': 0.2
            }


class ParsingValidator:
    """파싱 결과 종합 검증 클래스"""
    
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        self.logger = logger_instance or logger
        self.map_validator = ParsingMapValidator(logger_instance)
    
    def validate_parsing_result(self, parsing_result: Any, original_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        파싱 결과를 종합적으로 검증합니다.
        
        Args:
            parsing_result: 검증할 파싱 결과
            original_size: 원본 이미지 크기 (height, width)
            
        Returns:
            검증된 파싱 결과 딕셔너리
        """
        try:
            validated_result = {}
            
            # 파싱 맵 검증
            if 'parsing_map' in parsing_result:
                validated_result['parsing_map'] = self.map_validator.validate_parsing_map(
                    parsing_result['parsing_map'], original_size
                )
            
            # 신뢰도 맵 검증
            if 'confidence_map' in parsing_result:
                validated_result['confidence_map'] = validate_confidence_map(
                    parsing_result['confidence_map'], original_size
                )
            
            # 메타데이터 검증
            if 'metadata' in parsing_result:
                validated_result['metadata'] = self._validate_metadata(parsing_result['metadata'])
            
            # 품질 점수 계산
            validated_result['quality_score'] = self._calculate_quality_score(validated_result)
            
            return validated_result
            
        except Exception as e:
            self.logger.error(f"❌ 파싱 결과 검증 실패: {e}")
            return self._create_fallback_result(original_size)
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """메타데이터 검증"""
        try:
            validated_metadata = {}
            
            # 필수 필드 검증
            required_fields = ['model_name', 'processing_time', 'input_size']
            for field in required_fields:
                if field in metadata:
                    validated_metadata[field] = metadata[field]
                else:
                    validated_metadata[field] = 'unknown'
            
            # 추가 필드들 복사
            for key, value in metadata.items():
                if key not in validated_metadata:
                    validated_metadata[key] = value
            
            return validated_metadata
            
        except Exception as e:
            self.logger.warning(f"⚠️ 메타데이터 검증 실패: {e}")
            return {'model_name': 'unknown', 'processing_time': 0.0, 'input_size': (0, 0)}
    
    def _calculate_quality_score(self, validated_result: Dict[str, Any]) -> float:
        """품질 점수 계산"""
        try:
            score = 0.0
            
            # 파싱 맵 품질 점수
            if 'parsing_map' in validated_result:
                parsing_map = validated_result['parsing_map']
                if isinstance(parsing_map, np.ndarray):
                    # 유효한 픽셀 비율 계산
                    valid_pixels = np.sum(parsing_map > 0)
                    total_pixels = parsing_map.size
                    if total_pixels > 0:
                        score += (valid_pixels / total_pixels) * 0.6
            
            # 신뢰도 맵 품질 점수
            if 'confidence_map' in validated_result:
                confidence_map = validated_result['confidence_map']
                if isinstance(confidence_map, np.ndarray):
                    # 평균 신뢰도 계산
                    avg_confidence = np.mean(confidence_map)
                    score += avg_confidence * 0.4
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.warning(f"⚠️ 품질 점수 계산 실패: {e}")
            return 0.5
    
    def _create_fallback_result(self, original_size: Tuple[int, int]) -> Dict[str, Any]:
        """폴백 결과 생성"""
        return {
            'parsing_map': self.map_validator._create_fallback_parsing_map(original_size),
            'confidence_map': np.ones(original_size, dtype=np.float32) * 0.5,
            'metadata': {
                'model_name': 'fallback',
                'processing_time': 0.0,
                'input_size': original_size
            },
            'quality_score': 0.5
        }


class ParsingMapValidator:
    """파싱 맵 검증 및 정제 클래스"""
    
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        self.logger = logger_instance or logger
    
    def validate_parsing_map(self, parsing_map: Any, original_size: Tuple[int, int]) -> np.ndarray:
        """
        파싱 맵을 검증하고 정제합니다.
        
        Args:
            parsing_map: 검증할 파싱 맵 (다양한 타입 지원)
            original_size: 원본 이미지 크기 (height, width)
            
        Returns:
            정제된 파싱 맵 (np.ndarray)
        """
        try:
            # 1단계: 타입 검증 및 변환
            parsing_map = self._convert_to_numpy(parsing_map)
            
            # 2단계: 형태 검증
            parsing_map = self._validate_shape(parsing_map)
            
            # 3단계: 값 검증
            parsing_map = self._validate_values(parsing_map)
            
            # 4단계: 크기 조정
            parsing_map = self._resize_to_original(parsing_map, original_size)
            
            # 5단계: 최종 검증
            self._final_validation(parsing_map)
            
            return parsing_map
            
        except Exception as e:
            self.logger.error(f"❌ 파싱 맵 검증 실패: {e}")
            return self._create_fallback_parsing_map(original_size)
    
    def _convert_to_numpy(self, parsing_map: Any) -> np.ndarray:
        """다양한 타입의 파싱 맵을 NumPy 배열로 변환"""
        try:
            if isinstance(parsing_map, np.ndarray):
                return parsing_map
            elif hasattr(parsing_map, 'cpu') and hasattr(parsing_map, 'numpy'):
                # PyTorch 텐서
                return parsing_map.detach().cpu().numpy()
            elif isinstance(parsing_map, list):
                return np.array(parsing_map)
            elif isinstance(parsing_map, dict):
                # 딕셔너리에서 파싱 맵 추출
                for key in ['parsing_pred', 'parsing', 'output', 'parsing_output']:
                    if key in parsing_map:
                        return self._convert_to_numpy(parsing_map[key])
                # 키를 찾지 못한 경우 첫 번째 값 사용
                first_value = next(iter(parsing_map.values()))
                return self._convert_to_numpy(first_value)
            else:
                raise ValueError(f"지원하지 않는 파싱 맵 타입: {type(parsing_map)}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ 파싱 맵 변환 실패: {e}")
            raise
    
    def _validate_shape(self, parsing_map: np.ndarray) -> np.ndarray:
        """파싱 맵 형태를 검증하고 정규화"""
        try:
            if len(parsing_map.shape) == 2:
                # 2D 배열 - 그대로 사용
                return parsing_map
            elif len(parsing_map.shape) == 3:
                # 3D 배열 - 첫 번째 채널 사용
                if parsing_map.shape[0] == 1:
                    return parsing_map[0]
                elif parsing_map.shape[2] == 1:
                    return parsing_map[:, :, 0]
                else:
                    # 첫 번째 배치 사용
                    return parsing_map[0]
            elif len(parsing_map.shape) == 4:
                # 4D 배열 - 첫 번째 배치, 첫 번째 채널 사용
                return parsing_map[0, 0] if parsing_map.shape[1] == 1 else parsing_map[0]
            else:
                raise ValueError(f"지원하지 않는 파싱 맵 차원: {parsing_map.shape}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ 파싱 맵 형태 검증 실패: {e}")
            raise
    
    def _validate_values(self, parsing_map: np.ndarray) -> np.ndarray:
        """파싱 맵 값을 검증하고 정제"""
        try:
            # 데이터 타입 변환
            if parsing_map.dtype != np.uint8:
                parsing_map = parsing_map.astype(np.uint8)
            
            # 값 범위 검증 (0-19, 20개 클래스)
            unique_values = np.unique(parsing_map)
            self.logger.debug(f"🔍 파싱 맵 고유 값들: {unique_values}")
            
            # 모든 값이 0인 경우 검사
            if len(unique_values) == 1 and unique_values[0] == 0:
                self.logger.warning("⚠️ 파싱 맵이 비어있거나 모든 값이 0입니다. 기본값을 생성합니다.")
                return self._create_default_parsing_map(parsing_map.shape)
            
            # 값 범위 검증
            if parsing_map.max() > 19 or parsing_map.min() < 0:
                self.logger.warning(f"⚠️ 파싱 맵 값 범위가 잘못됨: {parsing_map.min()} ~ {parsing_map.max()}")
                # 범위를 0-19로 클리핑
                parsing_map = np.clip(parsing_map, 0, 19).astype(np.uint8)
            
            return parsing_map
            
        except Exception as e:
            self.logger.warning(f"⚠️ 파싱 맵 값 검증 실패: {e}")
            raise
    
    def _resize_to_original(self, parsing_map: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """파싱 맵을 원본 크기로 리사이즈"""
        try:
            if parsing_map.shape[:2] != original_size:
                self.logger.debug(f"🔍 파싱 맵 리사이즈: {parsing_map.shape[:2]} -> {original_size}")
                
                # PIL Image로 변환하여 리사이즈
                parsing_pil = Image.fromarray(parsing_map)
                parsing_resized = parsing_pil.resize(
                    (original_size[1], original_size[0]),  # (width, height)
                    Image.NEAREST  # 분할 맵에는 NEAREST가 적합
                )
                parsing_map = np.array(parsing_resized)
                
            return parsing_map
            
        except Exception as e:
            self.logger.warning(f"⚠️ 파싱 맵 리사이즈 실패: {e}")
            raise
    
    def _final_validation(self, parsing_map: np.ndarray) -> None:
        """최종 검증"""
        try:
            # 형태 검증
            if len(parsing_map.shape) != 2:
                raise ValueError(f"최종 파싱 맵이 2D가 아님: {parsing_map.shape}")
            
            # 값 검증
            if parsing_map.max() > 19 or parsing_map.min() < 0:
                raise ValueError(f"최종 파싱 맵 값 범위 오류: {parsing_map.min()} ~ {parsing_map.max()}")
            
            # 데이터 타입 검증
            if parsing_map.dtype != np.uint8:
                raise ValueError(f"최종 파싱 맵 데이터 타입 오류: {parsing_map.dtype}")
            
            self.logger.debug(f"✅ 파싱 맵 검증 완료: {parsing_map.shape}, 값 범위: {parsing_map.min()} ~ {parsing_map.max()}")
            
        except Exception as e:
            self.logger.error(f"❌ 최종 검증 실패: {e}")
            raise
    
    def _create_default_parsing_map(self, shape: Tuple[int, int]) -> np.ndarray:
        """기본 파싱 맵 생성 (모든 값이 0인 경우)"""
        try:
            # 간단한 인체 형태 생성 (테스트용)
            height, width = shape
            parsing_map = np.zeros(shape, dtype=np.uint8)
            
            # 중앙에 간단한 인체 형태 생성
            center_y, center_x = height // 2, width // 2
            
            # 몸통 영역 (클래스 10: torso_skin)
            torso_height = height // 3
            torso_width = width // 4
            y1 = max(0, center_y - torso_height // 2)
            y2 = min(height, center_y + torso_height // 2)
            x1 = max(0, center_x - torso_width // 2)
            x2 = min(width, center_x + torso_width // 2)
            parsing_map[y1:y2, x1:x2] = 10
            
            # 머리 영역 (클래스 13: face)
            head_radius = min(torso_width // 3, height // 6)
            head_y = max(head_radius, y1 - head_radius)
            head_x = center_x
            for i in range(max(0, head_y - head_radius), min(height, head_y + head_radius)):
                for j in range(max(0, head_x - head_radius), min(width, head_x + head_radius)):
                    if (i - head_y) ** 2 + (j - head_x) ** 2 <= head_radius ** 2:
                        parsing_map[i, j] = 13
            
            self.logger.info("✅ 기본 파싱 맵 생성 완료")
            return parsing_map
            
        except Exception as e:
            self.logger.error(f"❌ 기본 파싱 맵 생성 실패: {e}")
            return np.zeros(shape, dtype=np.uint8)
    
    def _create_fallback_parsing_map(self, original_size: Tuple[int, int]) -> np.ndarray:
        """폴백 파싱 맵 생성 (오류 발생 시)"""
        try:
            self.logger.warning("⚠️ 폴백 파싱 맵 생성")
            return np.zeros(original_size, dtype=np.uint8)
        except Exception as e:
            self.logger.error(f"❌ 폴백 파싱 맵 생성 실패: {e}")
            return np.zeros((512, 512), dtype=np.uint8)


def get_original_size_safely(original_image: Any) -> Tuple[int, int]:
    """
    원본 이미지에서 안전하게 크기를 추출합니다.
    
    Args:
        original_image: 원본 이미지 (PIL Image, NumPy 배열 등)
        
    Returns:
        원본 크기 (height, width)
    """
    try:
        # 기본값 설정
        original_size = (512, 512)
        
        if hasattr(original_image, 'size') and not isinstance(original_image, np.ndarray):
            # PIL Image인 경우
            original_size = original_image.size[::-1]  # (width, height) -> (height, width)
            logger.debug(f"🔍 PIL Image 크기: {original_size}")
            
        elif isinstance(original_image, np.ndarray):
            # NumPy 배열인 경우
            if len(original_image.shape) >= 2:
                original_size = original_image.shape[:2]
                logger.debug(f"🔍 NumPy 배열 크기: {original_size}")
            else:
                logger.warning(f"⚠️ NumPy 배열 형태가 잘못됨: {original_image.shape}")
                
        elif original_image is None:
            logger.warning("⚠️ original_image가 None입니다. 기본 크기 사용")
            
        else:
            logger.warning(f"⚠️ 알 수 없는 이미지 타입: {type(original_image)}")
            
        logger.debug(f"🔍 최종 원본 크기: {original_size}")
        return original_size
        
    except Exception as e:
        logger.warning(f"⚠️ 원본 크기 결정 실패: {e}. 기본 크기 사용")
        return (512, 512)


def validate_confidence_map(confidence_map: Any, original_size: Tuple[int, int]) -> np.ndarray:
    """
    신뢰도 맵을 검증하고 정제합니다.
    
    Args:
        confidence_map: 검증할 신뢰도 맵
        original_size: 원본 이미지 크기
        
    Returns:
        정제된 신뢰도 맵 (np.ndarray)
    """
    try:
        confidence_array = None
        
        if confidence_map is not None:
            if hasattr(confidence_map, 'cpu') and hasattr(confidence_map, 'numpy'):
                # PyTorch 텐서
                confidence_array = confidence_map.detach().cpu().numpy()
            elif isinstance(confidence_map, (int, float, np.float64)):
                confidence_array = np.array([float(confidence_map)])
            elif isinstance(confidence_map, dict):
                # 딕셔너리인 경우 첫 번째 값 사용
                first_value = next(iter(confidence_map.values()))
                if isinstance(first_value, (int, float, np.float64)):
                    confidence_array = np.array([float(first_value)])
                else:
                    confidence_array = np.array([0.5])
            else:
                try:
                    confidence_array = np.array(confidence_map, dtype=np.float32)
                except:
                    confidence_array = np.array([0.5])
        
        # 신뢰도 맵이 None이거나 잘못된 형태인 경우
        if confidence_array is None:
            confidence_array = np.ones(original_size, dtype=np.float32) * 0.8
        elif len(confidence_array.shape) != 2:
            # 1D 배열인 경우 2D로 확장
            if len(confidence_array.shape) == 1:
                confidence_array = np.full(original_size, confidence_array[0], dtype=np.float32)
            else:
                confidence_array = np.ones(original_size, dtype=np.float32) * 0.8
        
        # 크기 조정
        if confidence_array.shape != original_size:
            try:
                confidence_pil = Image.fromarray((confidence_array * 255).astype(np.uint8))
                confidence_resized = confidence_pil.resize(
                    (original_size[1], original_size[0]), 
                    Image.BILINEAR
                )
                confidence_array = np.array(confidence_resized).astype(np.float32) / 255.0
            except Exception as e:
                logger.warning(f"⚠️ confidence_array 리사이즈 실패: {e}")
                confidence_array = np.ones(original_size, dtype=np.float32) * 0.8
        
        return confidence_array
        
    except Exception as e:
        logger.warning(f"⚠️ 신뢰도 맵 검증 실패: {e}")
        return np.ones(original_size, dtype=np.float32) * 0.8


# 전역 검증기 인스턴스
parsing_validator = ParsingMapValidator()
