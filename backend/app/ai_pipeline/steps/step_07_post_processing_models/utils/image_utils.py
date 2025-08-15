"""
Image Utilities
이미지 처리와 관련된 유틸리티 함수들
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Union, Tuple, List, Optional, Dict, Any
import numpy as np
from PIL import Image
import cv2
import logging

# 프로젝트 로깅 설정 import
from backend.app.ai_pipeline.utils.logging_config import get_logger

logger = get_logger(__name__)

class ImageUtils:
    """
    이미지 처리와 관련된 유틸리티 함수들을 제공하는 클래스
    """
    
    @staticmethod
    def pil_to_tensor(pil_image: Image.Image, normalize: bool = True) -> torch.Tensor:
        """
        PIL 이미지를 PyTorch 텐서로 변환
        
        Args:
            pil_image: PIL 이미지
            normalize: 정규화 여부 (0-1 범위)
            
        Returns:
            PyTorch 텐서
        """
        try:
            # PIL 이미지를 텐서로 변환
            to_tensor = transforms.ToTensor()
            tensor = to_tensor(pil_image)
            
            if not normalize:
                # 0-255 범위로 변환
                tensor = tensor * 255.0
            
            return tensor
        except Exception as e:
            logger.error(f"PIL 이미지를 텐서로 변환 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor, denormalize: bool = True) -> Image.Image:
        """
        PyTorch 텐서를 PIL 이미지로 변환
        
        Args:
            tensor: PyTorch 텐서
            denormalize: 역정규화 여부 (0-255 범위)
            
        Returns:
            PIL 이미지
        """
        try:
            if denormalize:
                # 0-1 범위를 0-255 범위로 변환
                tensor = tensor.clamp(0, 1) * 255.0
            
            # 텐서를 PIL 이미지로 변환
            to_pil = transforms.ToPILImage()
            pil_image = to_pil(tensor)
            
            return pil_image
        except Exception as e:
            logger.error(f"텐서를 PIL 이미지로 변환 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def numpy_to_tensor(numpy_array: np.ndarray, normalize: bool = True) -> torch.Tensor:
        """
        NumPy 배열을 PyTorch 텐서로 변환
        
        Args:
            numpy_array: NumPy 배열
            normalize: 정규화 여부 (0-1 범위)
            
        Returns:
            PyTorch 텐서
        """
        try:
            # NumPy 배열을 텐서로 변환
            tensor = torch.from_numpy(numpy_array).float()
            
            # 채널 순서 조정 (HWC -> CHW)
            if tensor.dim() == 3 and tensor.size(-1) in [1, 3, 4]:
                tensor = tensor.permute(2, 0, 1)
            
            if normalize and tensor.max() > 1.0:
                # 0-255 범위를 0-1 범위로 정규화
                tensor = tensor / 255.0
            
            return tensor
        except Exception as e:
            logger.error(f"NumPy 배열을 텐서로 변환 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def tensor_to_numpy(tensor: torch.Tensor, denormalize: bool = True) -> np.ndarray:
        """
        PyTorch 텐서를 NumPy 배열로 변환
        
        Args:
            tensor: PyTorch 텐서
            denormalize: 역정규화 여부 (0-255 범위)
            
        Returns:
            NumPy 배열
        """
        try:
            # 텐서를 NumPy 배열로 변환
            numpy_array = tensor.detach().cpu().numpy()
            
            # 채널 순서 조정 (CHW -> HWC)
            if numpy_array.ndim == 3 and numpy_array.shape[0] in [1, 3, 4]:
                numpy_array = numpy_array.transpose(1, 2, 0)
            
            if denormalize:
                # 0-1 범위를 0-255 범위로 변환
                numpy_array = (numpy_array * 255.0).astype(np.uint8)
            
            return numpy_array
        except Exception as e:
            logger.error(f"텐서를 NumPy 배열로 변환 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def resize_image(image: Union[torch.Tensor, np.ndarray, Image.Image], 
                    target_size: Tuple[int, int], 
                    interpolation: str = 'bilinear') -> Union[torch.Tensor, np.ndarray, Image.Image]:
        """
        이미지 크기 조정
        
        Args:
            image: 입력 이미지 (텐서, NumPy 배열, 또는 PIL 이미지)
            target_size: 목표 크기 (width, height)
            interpolation: 보간 방법 ('bilinear', 'bicubic', 'nearest')
            
        Returns:
            크기가 조정된 이미지
        """
        try:
            if isinstance(image, torch.Tensor):
                return ImageUtils._resize_tensor(image, target_size, interpolation)
            elif isinstance(image, np.ndarray):
                return ImageUtils._resize_numpy(image, target_size, interpolation)
            elif isinstance(image, Image.Image):
                return ImageUtils._resize_pil(image, target_size, interpolation)
            else:
                raise ValueError(f"지원하지 않는 이미지 타입: {type(image)}")
        except Exception as e:
            logger.error(f"이미지 크기 조정 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def _resize_tensor(tensor: torch.Tensor, target_size: Tuple[int, int], 
                       interpolation: str) -> torch.Tensor:
        """텐서 이미지 크기 조정"""
        try:
            # 보간 방법 선택
            mode = 'bilinear' if interpolation == 'bilinear' else 'bicubic' if interpolation == 'bicubic' else 'nearest'
            
            # 크기 조정
            resized = F.interpolate(
                tensor.unsqueeze(0), 
                size=target_size, 
                mode=mode, 
                align_corners=False
            )
            
            return resized.squeeze(0)
        except Exception as e:
            logger.error(f"텐서 이미지 크기 조정 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def _resize_numpy(numpy_array: np.ndarray, target_size: Tuple[int, int], 
                      interpolation: str) -> np.ndarray:
        """NumPy 배열 이미지 크기 조정"""
        try:
            # OpenCV 보간 방법 선택
            if interpolation == 'bilinear':
                cv_interpolation = cv2.INTER_LINEAR
            elif interpolation == 'bicubic':
                cv_interpolation = cv2.INTER_CUBIC
            else:
                cv_interpolation = cv2.INTER_NEAREST
            
            # 크기 조정
            resized = cv2.resize(numpy_array, target_size, interpolation=cv_interpolation)
            
            return resized
        except Exception as e:
            logger.error(f"NumPy 배열 이미지 크기 조정 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def _resize_pil(pil_image: Image.Image, target_size: Tuple[int, int], 
                    interpolation: str) -> Image.Image:
        """PIL 이미지 크기 조정"""
        try:
            # PIL 보간 방법 선택
            if interpolation == 'bilinear':
                pil_interpolation = Image.BILINEAR
            elif interpolation == 'bicubic':
                pil_interpolation = Image.BICUBIC
            else:
                pil_interpolation = Image.NEAREST
            
            # 크기 조정
            resized = pil_image.resize(target_size, pil_interpolation)
            
            return resized
        except Exception as e:
            logger.error(f"PIL 이미지 크기 조정 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def crop_image(image: Union[torch.Tensor, np.ndarray, Image.Image], 
                   crop_box: Tuple[int, int, int, int]) -> Union[torch.Tensor, np.ndarray, Image.Image]:
        """
        이미지 크롭
        
        Args:
            image: 입력 이미지
            crop_box: 크롭 영역 (left, top, right, bottom)
            
        Returns:
            크롭된 이미지
        """
        try:
            if isinstance(image, torch.Tensor):
                return ImageUtils._crop_tensor(image, crop_box)
            elif isinstance(image, np.ndarray):
                return ImageUtils._crop_numpy(image, crop_box)
            elif isinstance(image, Image.Image):
                return ImageUtils._crop_pil(image, crop_box)
            else:
                raise ValueError(f"지원하지 않는 이미지 타입: {type(image)}")
        except Exception as e:
            logger.error(f"이미지 크롭 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def _crop_tensor(tensor: torch.Tensor, crop_box: Tuple[int, int, int, int]) -> torch.Tensor:
        """텐서 이미지 크롭"""
        try:
            left, top, right, bottom = crop_box
            return tensor[:, top:bottom, left:right]
        except Exception as e:
            logger.error(f"텐서 이미지 크롭 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def _crop_numpy(numpy_array: np.ndarray, crop_box: Tuple[int, int, int, int]) -> np.ndarray:
        """NumPy 배열 이미지 크롭"""
        try:
            left, top, right, bottom = crop_box
            return numpy_array[top:bottom, left:right]
        except Exception as e:
            logger.error(f"NumPy 배열 이미지 크롭 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def _crop_pil(pil_image: Image.Image, crop_box: Tuple[int, int, int, int]) -> Image.Image:
        """PIL 이미지 크롭"""
        try:
            return pil_image.crop(crop_box)
        except Exception as e:
            logger.error(f"PIL 이미지 크롭 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def normalize_image(image: Union[torch.Tensor, np.ndarray], 
                       mean: Optional[List[float]] = None, 
                       std: Optional[List[float]] = None) -> Union[torch.Tensor, np.ndarray]:
        """
        이미지 정규화
        
        Args:
            image: 입력 이미지
            mean: 평균값 (None이면 0.5 사용)
            std: 표준편차 (None이면 0.5 사용)
            
        Returns:
            정규화된 이미지
        """
        try:
            if mean is None:
                mean = [0.5, 0.5, 0.5]
            if std is None:
                std = [0.5, 0.5, 0.5]
            
            if isinstance(image, torch.Tensor):
                return ImageUtils._normalize_tensor(image, mean, std)
            elif isinstance(image, np.ndarray):
                return ImageUtils._normalize_numpy(image, mean, std)
            else:
                raise ValueError(f"지원하지 않는 이미지 타입: {type(image)}")
        except Exception as e:
            logger.error(f"이미지 정규화 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def _normalize_tensor(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
        """텐서 이미지 정규화"""
        try:
            # 정규화 변환 생성
            normalize_transform = transforms.Normalize(mean=mean, std=std)
            return normalize_transform(tensor)
        except Exception as e:
            logger.error(f"텐서 이미지 정규화 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def _normalize_numpy(numpy_array: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
        """NumPy 배열 이미지 정규화"""
        try:
            # 0-1 범위로 정규화
            normalized = numpy_array.astype(np.float32) / 255.0
            
            # 평균과 표준편차로 정규화
            for i in range(min(len(mean), normalized.shape[-1])):
                normalized[..., i] = (normalized[..., i] - mean[i]) / std[i]
            
            return normalized
        except Exception as e:
            logger.error(f"NumPy 배열 이미지 정규화 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def denormalize_image(image: Union[torch.Tensor, np.ndarray], 
                          mean: Optional[List[float]] = None, 
                          std: Optional[List[float]] = None) -> Union[torch.Tensor, np.ndarray]:
        """
        이미지 역정규화
        
        Args:
            image: 입력 이미지
            mean: 평균값 (None이면 0.5 사용)
            std: 표준편차 (None이면 0.5 사용)
            
        Returns:
            역정규화된 이미지
        """
        try:
            if mean is None:
                mean = [0.5, 0.5, 0.5]
            if std is None:
                std = [0.5, 0.5, 0.5]
            
            if isinstance(image, torch.Tensor):
                return ImageUtils._denormalize_tensor(image, mean, std)
            elif isinstance(image, np.ndarray):
                return ImageUtils._denormalize_numpy(image, mean, std)
            else:
                raise ValueError(f"지원하지 않는 이미지 타입: {type(image)}")
        except Exception as e:
            logger.error(f"이미지 역정규화 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def _denormalize_tensor(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
        """텐서 이미지 역정규화"""
        try:
            # 역정규화 변환 생성
            denormalize_transform = transforms.Compose([
                transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1/s for s in std]),
                transforms.Normalize(mean=[-m for m in mean], std=[1.0, 1.0, 1.0])
            ])
            return denormalize_transform(tensor)
        except Exception as e:
            logger.error(f"텐서 이미지 역정규화 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def _denormalize_numpy(numpy_array: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
        """NumPy 배열 이미지 역정규화"""
        try:
            denormalized = numpy_array.copy()
            
            # 평균과 표준편차로 역정규화
            for i in range(min(len(mean), denormalized.shape[-1])):
                denormalized[..., i] = denormalized[..., i] * std[i] + mean[i]
            
            # 0-255 범위로 변환
            denormalized = np.clip(denormalized * 255.0, 0, 255).astype(np.uint8)
            
            return denormalized
        except Exception as e:
            logger.error(f"NumPy 배열 이미지 역정규화 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def convert_color_space(image: Union[torch.Tensor, np.ndarray], 
                           from_space: str, to_space: str) -> Union[torch.Tensor, np.ndarray]:
        """
        색상 공간 변환
        
        Args:
            image: 입력 이미지
            from_space: 원본 색상 공간 ('RGB', 'BGR', 'GRAY', 'HSV')
            to_space: 목표 색상 공간 ('RGB', 'BGR', 'GRAY', 'HSV')
            
        Returns:
            변환된 이미지
        """
        try:
            if isinstance(image, torch.Tensor):
                return ImageUtils._convert_color_space_tensor(image, from_space, to_space)
            elif isinstance(image, np.ndarray):
                return ImageUtils._convert_color_space_numpy(image, from_space, to_space)
            else:
                raise ValueError(f"지원하지 않는 이미지 타입: {type(image)}")
        except Exception as e:
            logger.error(f"색상 공간 변환 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def _convert_color_space_tensor(tensor: torch.Tensor, from_space: str, to_space: str) -> torch.Tensor:
        """텐서 이미지 색상 공간 변환"""
        try:
            # 텐서를 NumPy로 변환
            numpy_array = ImageUtils.tensor_to_numpy(tensor, denormalize=False)
            
            # 색상 공간 변환
            converted = ImageUtils._convert_color_space_numpy(numpy_array, from_space, to_space)
            
            # NumPy를 텐서로 변환
            return ImageUtils.numpy_to_tensor(converted, normalize=True)
        except Exception as e:
            logger.error(f"텐서 이미지 색상 공간 변환 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def _convert_color_space_numpy(numpy_array: np.ndarray, from_space: str, to_space: str) -> np.ndarray:
        """NumPy 배열 이미지 색상 공간 변환"""
        try:
            if from_space == to_space:
                return numpy_array
            
            # OpenCV 색상 공간 변환
            if from_space == 'RGB' and to_space == 'BGR':
                return cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
            elif from_space == 'BGR' and to_space == 'RGB':
                return cv2.cvtColor(numpy_array, cv2.COLOR_BGR2RGB)
            elif from_space in ['RGB', 'BGR'] and to_space == 'GRAY':
                return cv2.cvtColor(numpy_array, cv2.COLOR_RGB2GRAY if from_space == 'RGB' else cv2.COLOR_BGR2GRAY)
            elif from_space == 'GRAY' and to_space in ['RGB', 'BGR']:
                return cv2.cvtColor(numpy_array, cv2.COLOR_GRAY2RGB if to_space == 'RGB' else cv2.COLOR_GRAY2BGR)
            elif from_space in ['RGB', 'BGR'] and to_space == 'HSV':
                return cv2.cvtColor(numpy_array, cv2.COLOR_RGB2HSV if from_space == 'RGB' else cv2.COLOR_BGR2HSV)
            elif from_space == 'HSV' and to_space in ['RGB', 'BGR']:
                return cv2.cvtColor(numpy_array, cv2.COLOR_HSV2RGB if to_space == 'RGB' else cv2.COLOR_HSV2BGR)
            else:
                raise ValueError(f"지원하지 않는 색상 공간 변환: {from_space} -> {to_space}")
        except Exception as e:
            logger.error(f"NumPy 배열 이미지 색상 공간 변환 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def get_image_info(image: Union[torch.Tensor, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        이미지 정보 반환
        
        Args:
            image: 입력 이미지
            
        Returns:
            이미지 정보 딕셔너리
        """
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
            logger.error(f"이미지 정보 추출 중 오류 발생: {e}")
            return {'error': str(e)}
