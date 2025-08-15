"""
Image Preprocessor
후처리 전 이미지 전처리를 담당하는 클래스
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from PIL import Image
import cv2
import logging

# 프로젝트 로깅 설정 import
from backend.app.ai_pipeline.utils.logging_config import get_logger

logger = get_logger(__name__)

class ImagePreprocessor:
    """
    후처리 전 이미지 전처리를 담당하는 클래스
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Args:
            device: 사용할 디바이스
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 전처리 설정
        self.preprocessing_config = {
            'resize_method': 'bilinear',  # 'bilinear', 'bicubic', 'nearest'
            'normalization': 'standard',  # 'standard', 'minmax', 'none'
            'color_space': 'RGB',  # 'RGB', 'BGR', 'GRAY'
            'padding': 'reflect',  # 'reflect', 'constant', 'replicate'
            'interpolation_order': 1,
            'enable_augmentation': False,
            'target_size': None,  # (width, height)
            'maintain_aspect_ratio': True
        }
        
        logger.info(f"ImagePreprocessor initialized on device: {self.device}")
    
    def preprocess_image(self, image: Union[torch.Tensor, np.ndarray, Image.Image], 
                        config: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        이미지 전처리 실행
        
        Args:
            image: 입력 이미지
            config: 전처리 설정
            
        Returns:
            전처리된 이미지 텐서
        """
        try:
            # 설정 병합
            if config is None:
                config = {}
            
            preprocessing_config = {**self.preprocessing_config, **config}
            
            # 이미지를 텐서로 변환
            if isinstance(image, np.ndarray):
                tensor_image = self._numpy_to_tensor(image)
            elif isinstance(image, Image.Image):
                tensor_image = self._pil_to_tensor(image)
            elif isinstance(image, torch.Tensor):
                tensor_image = image.clone()
            else:
                raise ValueError(f"지원하지 않는 이미지 타입: {type(image)}")
            
            # 이미지를 디바이스로 이동
            tensor_image = tensor_image.to(self.device)
            
            # 1. 색상 공간 변환
            if preprocessing_config['color_space'] != 'RGB':
                tensor_image = self._convert_color_space(tensor_image, 'RGB', preprocessing_config['color_space'])
            
            # 2. 크기 조정
            if preprocessing_config['target_size'] is not None:
                tensor_image = self._resize_image(tensor_image, preprocessing_config['target_size'], 
                                                preprocessing_config['resize_method'],
                                                preprocessing_config['maintain_aspect_ratio'])
            
            # 3. 패딩
            if preprocessing_config['padding'] != 'none':
                tensor_image = self._apply_padding(tensor_image, preprocessing_config['padding'])
            
            # 4. 정규화
            if preprocessing_config['normalization'] != 'none':
                tensor_image = self._normalize_image(tensor_image, preprocessing_config['normalization'])
            
            # 5. 데이터 증강 (선택적)
            if preprocessing_config['enable_augmentation']:
                tensor_image = self._apply_augmentation(tensor_image)
            
            return tensor_image
            
        except Exception as e:
            logger.error(f"이미지 전처리 중 오류 발생: {e}")
            raise
    
    def _numpy_to_tensor(self, numpy_image: np.ndarray) -> torch.Tensor:
        """NumPy 배열을 텐서로 변환"""
        try:
            # 텐서로 변환
            tensor = torch.from_numpy(numpy_image).float()
            
            # 채널 순서 조정 (HWC -> CHW)
            if tensor.dim() == 3 and tensor.size(-1) in [1, 3, 4]:
                tensor = tensor.permute(2, 0, 1)
            
            # 값 범위 정규화 (0-255 -> 0-1)
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            
            return tensor
            
        except Exception as e:
            logger.error(f"NumPy to Tensor 변환 중 오류 발생: {e}")
            raise
    
    def _pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """PIL 이미지를 텐서로 변환"""
        try:
            # PIL 이미지를 NumPy로 변환
            numpy_image = np.array(pil_image)
            
            # NumPy를 텐서로 변환
            return self._numpy_to_tensor(numpy_image)
            
        except Exception as e:
            logger.error(f"PIL to Tensor 변환 중 오류 발생: {e}")
            raise
    
    def _convert_color_space(self, image: torch.Tensor, 
                            from_space: str, to_space: str) -> torch.Tensor:
        """색상 공간 변환"""
        try:
            if from_space == to_space:
                return image
            
            # 텐서를 NumPy로 변환
            numpy_image = image.detach().cpu().numpy()
            
            # 채널 순서 조정 (CHW -> HWC)
            if numpy_image.ndim == 3 and numpy_image.shape[0] in [1, 3, 4]:
                numpy_image = numpy_image.transpose(1, 2, 0)
            
            # 색상 공간 변환
            if from_space == 'RGB' and to_space == 'BGR':
                converted = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            elif from_space == 'BGR' and to_space == 'RGB':
                converted = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
            elif from_space in ['RGB', 'BGR'] and to_space == 'GRAY':
                converted = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY if from_space == 'RGB' else cv2.COLOR_BGR2GRAY)
                converted = converted[..., np.newaxis]  # 채널 차원 추가
            elif from_space == 'GRAY' and to_space in ['RGB', 'BGR']:
                converted = cv2.cvtColor(numpy_image, cv2.COLOR_GRAY2RGB if to_space == 'RGB' else cv2.COLOR_GRAY2BGR)
            else:
                logger.warning(f"지원하지 않는 색상 공간 변환: {from_space} -> {to_space}")
                return image
            
            # NumPy를 텐서로 변환
            tensor = torch.from_numpy(converted).float()
            
            # 채널 순서 조정 (HWC -> CHW)
            if tensor.dim() == 3 and tensor.size(-1) in [1, 3, 4]:
                tensor = tensor.permute(2, 0, 1)
            
            return tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"색상 공간 변환 중 오류 발생: {e}")
            return image
    
    def _resize_image(self, image: torch.Tensor, target_size: Tuple[int, int], 
                      method: str, maintain_aspect_ratio: bool) -> torch.Tensor:
        """이미지 크기 조정"""
        try:
            current_height, current_width = image.size(-2), image.size(-1)
            target_width, target_height = target_size
            
            if maintain_aspect_ratio:
                # 종횡비 유지
                aspect_ratio = current_width / current_height
                target_aspect_ratio = target_width / target_height
                
                if aspect_ratio > target_aspect_ratio:
                    # 너비에 맞춤
                    new_width = target_width
                    new_height = int(target_width / aspect_ratio)
                else:
                    # 높이에 맞춤
                    new_height = target_height
                    new_width = int(target_height * aspect_ratio)
                
                target_size = (new_height, new_width)
            
            # 보간 방법 선택
            if method == 'bilinear':
                mode = 'bilinear'
            elif method == 'bicubic':
                mode = 'bicubic'
            elif method == 'nearest':
                mode = 'nearest'
            else:
                mode = 'bilinear'
            
            # 크기 조정
            resized = F.interpolate(
                image.unsqueeze(0), 
                size=target_size, 
                mode=mode, 
                align_corners=False
            )
            
            return resized.squeeze(0)
            
        except Exception as e:
            logger.error(f"이미지 크기 조정 중 오류 발생: {e}")
            return image
    
    def _apply_padding(self, image: torch.Tensor, padding_type: str) -> torch.Tensor:
        """패딩 적용"""
        try:
            if padding_type == 'none':
                return image
            
            # 패딩 크기 계산 (모델 입력 크기에 맞춤)
            current_height, current_width = image.size(-2), image.size(-1)
            
            # 2의 거듭제곱으로 맞춤
            target_height = 2 ** int(np.ceil(np.log2(current_height)))
            target_width = 2 ** int(np.ceil(np.log2(current_width)))
            
            # 패딩 크기 계산
            pad_height = max(0, target_height - current_height)
            pad_width = max(0, target_width - current_width)
            
            if pad_height == 0 and pad_width == 0:
                return image
            
            # 패딩 적용
            if padding_type == 'reflect':
                padding = (pad_width//2, pad_width - pad_width//2, 
                          pad_height//2, pad_height - pad_height//2)
                padded = F.pad(image.unsqueeze(0), padding, mode='reflect')
            elif padding_type == 'constant':
                padding = (pad_width//2, pad_width - pad_width//2, 
                          pad_height//2, pad_height - pad_height//2)
                padded = F.pad(image.unsqueeze(0), padding, mode='constant', value=0)
            elif padding_type == 'replicate':
                padding = (pad_width//2, pad_width - pad_width//2, 
                          pad_height//2, pad_height - pad_height//2)
                padded = F.pad(image.unsqueeze(0), padding, mode='replicate')
            else:
                return image
            
            return padded.squeeze(0)
            
        except Exception as e:
            logger.error(f"패딩 적용 중 오류 발생: {e}")
            return image
    
    def _normalize_image(self, image: torch.Tensor, normalization_type: str) -> torch.Tensor:
        """이미지 정규화"""
        try:
            if normalization_type == 'none':
                return image
            
            if normalization_type == 'standard':
                # 표준 정규화 (평균=0, 표준편차=1)
                mean = torch.mean(image, dim=[1, 2], keepdim=True)
                std = torch.std(image, dim=[1, 2], keepdim=True)
                
                # 0으로 나누기 방지
                std = torch.clamp(std, min=1e-8)
                
                normalized = (image - mean) / std
                
            elif normalization_type == 'minmax':
                # 최소-최대 정규화 (0-1 범위)
                min_val = torch.min(image)
                max_val = torch.max(image)
                
                if max_val > min_val:
                    normalized = (image - min_val) / (max_val - min_val)
                else:
                    normalized = image
                    
            else:
                logger.warning(f"알 수 없는 정규화 타입: {normalization_type}")
                return image
            
            return normalized
            
        except Exception as e:
            logger.error(f"이미지 정규화 중 오류 발생: {e}")
            return image
    
    def _apply_augmentation(self, image: torch.Tensor) -> torch.Tensor:
        """데이터 증강 적용"""
        try:
            augmented = image.clone()
            
            # 랜덤 수평 뒤집기
            if torch.rand(1).item() > 0.5:
                augmented = torch.flip(augmented, dims=[-1])
            
            # 랜덤 회전 (90도 단위)
            if torch.rand(1).item() > 0.5:
                k = torch.randint(1, 4, (1,)).item()
                augmented = torch.rot90(augmented, k, dims=[-2, -1])
            
            # 랜덤 밝기 조정
            if torch.rand(1).item() > 0.5:
                brightness_factor = 0.8 + torch.rand(1).item() * 0.4  # 0.8-1.2
                augmented = torch.clamp(augmented * brightness_factor, 0, 1)
            
            # 랜덤 대비 조정
            if torch.rand(1).item() > 0.5:
                contrast_factor = 0.8 + torch.rand(1).item() * 0.4  # 0.8-1.2
                mean = torch.mean(augmented)
                augmented = torch.clamp((augmented - mean) * contrast_factor + mean, 0, 1)
            
            return augmented
            
        except Exception as e:
            logger.error(f"데이터 증강 적용 중 오류 발생: {e}")
            return image
    
    def batch_preprocess(self, images: List[Union[torch.Tensor, np.ndarray, Image.Image]], 
                        config: Optional[Dict[str, Any]] = None) -> List[torch.Tensor]:
        """
        배치 이미지 전처리
        
        Args:
            images: 입력 이미지 리스트
            config: 전처리 설정
            
        Returns:
            전처리된 이미지 텐서 리스트
        """
        try:
            logger.info(f"배치 전처리 시작 - {len(images)}개 이미지")
            
            preprocessed_images = []
            for i, image in enumerate(images):
                logger.debug(f"이미지 {i+1}/{len(images)} 전처리 중...")
                
                preprocessed = self.preprocess_image(image, config)
                preprocessed_images.append(preprocessed)
            
            logger.info("배치 전처리 완료")
            return preprocessed_images
            
        except Exception as e:
            logger.error(f"배치 전처리 중 오류 발생: {e}")
            raise
    
    def get_preprocessing_info(self, image: Union[torch.Tensor, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """전처리 정보 반환"""
        try:
            info = {}
            
            if isinstance(image, torch.Tensor):
                info['type'] = 'torch.Tensor'
                info['shape'] = list(image.shape)
                info['dtype'] = str(image.dtype)
                info['device'] = str(image.device)
                info['min_value'] = float(image.min())
                info['max_value'] = float(image.max())
                info['mean_value'] = float(image.mean())
            elif isinstance(image, np.ndarray):
                info['type'] = 'numpy.ndarray'
                info['shape'] = list(image.shape)
                info['dtype'] = str(image.dtype)
                info['min_value'] = float(image.min())
                info['max_value'] = float(image.max())
                info['mean_value'] = float(image.mean())
            elif isinstance(image, Image.Image):
                info['type'] = 'PIL.Image'
                info['size'] = image.size
                info['mode'] = image.mode
                info['format'] = image.format
            else:
                info['type'] = str(type(image))
            
            return info
            
        except Exception as e:
            logger.error(f"전처리 정보 추출 중 오류 발생: {e}")
            return {'error': str(e)}
    
    def set_preprocessing_config(self, **kwargs):
        """전처리 설정 업데이트"""
        self.preprocessing_config.update(kwargs)
        logger.info("전처리 설정 업데이트 완료")
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """전처리 설정 반환"""
        return self.preprocessing_config.copy()
    
    def get_available_methods(self) -> Dict[str, List[str]]:
        """사용 가능한 전처리 방법들 반환"""
        return {
            'resize_method': ['bilinear', 'bicubic', 'nearest'],
            'normalization': ['standard', 'minmax', 'none'],
            'color_space': ['RGB', 'BGR', 'GRAY'],
            'padding': ['reflect', 'constant', 'replicate', 'none']
        }
