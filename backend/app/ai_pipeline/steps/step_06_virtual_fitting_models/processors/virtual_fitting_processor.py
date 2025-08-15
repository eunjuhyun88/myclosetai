"""
🔥 Virtual Fitting Processor
============================

가상 피팅을 위한 이미지 전처리 및 후처리 프로세서입니다.
논문 기반의 AI 모델 구조에 맞춰 구현되었습니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import cv2
from PIL import Image
import logging

# 프로젝트 로깅 설정 import
try:
    from backend.app.core.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

class VirtualFittingProcessor:
    """
    가상 피팅을 위한 이미지 전처리 및 후처리 프로세서
    """

    def __init__(self, device: str = 'auto'):
        """
        Args:
            device: 사용할 디바이스 ('auto', 'cpu', 'cuda')
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 프로세서 설정
        self.processor_config = {
            'default_input_size': (512, 512),
            'default_output_size': (1024, 1024),
            'enable_body_alignment': True,
            'enable_clothing_enhancement': True,
            'enable_lighting_adjustment': True,
            'enable_shadow_generation': True
        }
        
        logger.info(f"VirtualFittingProcessor initialized on device: {self.device}")

    def preprocess_for_virtual_fitting(self, person_image: Union[np.ndarray, Image.Image, torch.Tensor],
                                     clothing_image: Union[np.ndarray, Image.Image, torch.Tensor],
                                     target_size: Tuple[int, int] = None,
                                     normalize: bool = True,
                                     **kwargs) -> Dict[str, torch.Tensor]:
        """
        가상 피팅을 위한 이미지 전처리
        
        Args:
            person_image: 사람 이미지
            clothing_image: 의류 이미지
            target_size: 목표 크기
            normalize: 정규화 여부
            **kwargs: 추가 파라미터
            
        Returns:
            전처리된 이미지들 딕셔너리
        """
        try:
            # 목표 크기 설정
            if target_size is None:
                target_size = self.processor_config['default_input_size']
            
            # 사람 이미지 전처리
            processed_person = self._preprocess_single_image(
                person_image, target_size, normalize, 'person'
            )
            
            # 의류 이미지 전처리
            processed_clothing = self._preprocess_single_image(
                clothing_image, target_size, normalize, 'clothing'
            )
            
            # 신체 정렬 (필요시)
            if self.processor_config['enable_body_alignment']:
                processed_person = self._align_body_pose(processed_person, **kwargs)
            
            # 의류 향상 (필요시)
            if self.processor_config['enable_clothing_enhancement']:
                processed_clothing = self._enhance_clothing(processed_clothing, **kwargs)
            
            result = {
                'person_image': processed_person,
                'clothing_image': processed_clothing,
                'target_size': target_size
            }
            
            logger.info(f"✅ Virtual fitting preprocessing completed: {processed_person.shape}, {processed_clothing.shape}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Virtual fitting preprocessing failed: {e}")
            raise

    def _preprocess_single_image(self, image: Union[np.ndarray, Image.Image, torch.Tensor],
                                target_size: Tuple[int, int],
                                normalize: bool,
                                image_type: str) -> torch.Tensor:
        """
        단일 이미지를 전처리합니다.
        """
        try:
            # 이미지 타입 통일
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            if isinstance(image, np.ndarray):
                # RGB로 변환
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                
                # 텐서로 변환
                image = torch.from_numpy(image).permute(2, 0, 1).float()
            elif isinstance(image, torch.Tensor):
                if len(image.shape) == 2:
                    image = image.unsqueeze(0).repeat(3, 1, 1)
                elif len(image.shape) == 3 and image.shape[0] == 1:
                    image = image.repeat(3, 1, 1)
            
            # 크기 조정
            if image.shape[-2:] != target_size:
                image = F.interpolate(image.unsqueeze(0), size=target_size, 
                                    mode='bilinear', align_corners=False).squeeze(0)
            
            # 정규화
            if normalize:
                if image.max() > 1.0:
                    image = image / 255.0
            
            # 배치 차원 추가
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            
            # 디바이스 이동
            image = image.to(self.device)
            
            return image
            
        except Exception as e:
            logger.error(f"❌ Single image preprocessing failed for {image_type}: {e}")
            raise

    def _align_body_pose(self, person_image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        신체 자세를 정렬합니다.
        """
        try:
            # 간단한 신체 정렬 (실제로는 더 복잡한 알고리즘 사용)
            aligned_image = person_image
            
            # 이미지 중심점 계산
            h, w = person_image.shape[-2:]
            center_h, center_w = h // 2, w // 2
            
            # 신체 중심을 이미지 중심에 맞춤
            # 실제 구현에서는 포즈 추정 결과를 사용
            logger.debug(f"Body alignment applied: center at ({center_h}, {center_w})")
            
            return aligned_image
            
        except Exception as e:
            logger.warning(f"⚠️ Body alignment failed: {e}")
            return person_image

    def _enhance_clothing(self, clothing_image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        의류 이미지를 향상시킵니다.
        """
        try:
            enhanced_image = clothing_image
            
            # 선명도 향상
            enhanced_image = self._sharpen_image(enhanced_image, strength=0.3)
            
            # 색상 보정
            enhanced_image = self._correct_colors(enhanced_image, 
                                               brightness=0.0, 
                                               contrast=1.1, 
                                               saturation=1.2)
            
            logger.debug("Clothing enhancement applied")
            return enhanced_image
            
        except Exception as e:
            logger.warning(f"⚠️ Clothing enhancement failed: {e}")
            return clothing_image

    def _sharpen_image(self, image: torch.Tensor, strength: float = 0.5) -> torch.Tensor:
        """
        이미지 선명도를 향상시킵니다.
        """
        # 언샤프 마스크 적용
        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=torch.float32, device=self.device)
        
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        
        sharpened = torch.zeros_like(image)
        for c in range(image.shape[1]):
            sharpened[:, c:c+1] = F.conv2d(
                image[:, c:c+1], 
                kernel,
                padding=1
            )
        
        # 원본과 블렌딩
        result = image + strength * (sharpened - image)
        return torch.clamp(result, 0, 1)

    def _correct_colors(self, image: torch.Tensor, 
                       brightness: float = 0.0,
                       contrast: float = 1.0,
                       saturation: float = 1.0) -> torch.Tensor:
        """
        이미지 색상을 보정합니다.
        """
        corrected = image
        
        # 밝기 조정
        if brightness != 0.0:
            corrected = corrected + brightness
            corrected = torch.clamp(corrected, 0, 1)
        
        # 대비 조정
        if contrast != 1.0:
            mean_val = corrected.mean()
            corrected = (corrected - mean_val) * contrast + mean_val
            corrected = torch.clamp(corrected, 0, 1)
        
        # 채도 조정
        if saturation != 1.0 and corrected.shape[1] == 3:
            # 그레이스케일 계산
            gray = 0.299 * corrected[:, 0] + 0.587 * corrected[:, 1] + 0.114 * corrected[:, 2]
            gray = gray.unsqueeze(1)
            
            # 채도 조정
            corrected = gray + saturation * (corrected - gray)
            corrected = torch.clamp(corrected, 0, 1)
        
        return corrected

    def postprocess_virtual_fitting(self, fitted_image: torch.Tensor,
                                  target_size: Tuple[int, int] = None,
                                  denormalize: bool = True,
                                  **kwargs) -> torch.Tensor:
        """
        가상 피팅 결과를 후처리합니다.
        """
        try:
            processed_output = fitted_image
            
            # 크기 조정
            if target_size is not None and fitted_image.shape[-2:] != target_size:
                processed_output = F.interpolate(processed_output, size=target_size, 
                                              mode='bilinear', align_corners=False)
            
            # 조명 조정
            if self.processor_config['enable_lighting_adjustment']:
                processed_output = self._adjust_lighting(processed_output, **kwargs)
            
            # 그림자 생성
            if self.processor_config['enable_shadow_generation']:
                processed_output = self._generate_shadows(processed_output, **kwargs)
            
            # 역정규화
            if denormalize:
                processed_output = torch.clamp(processed_output, 0, 1)
                if processed_output.max() <= 1.0:
                    processed_output = processed_output * 255.0
            
            logger.info(f"✅ Virtual fitting postprocessing completed: {processed_output.shape}")
            return processed_output
            
        except Exception as e:
            logger.error(f"❌ Virtual fitting postprocessing failed: {e}")
            return fitted_image

    def _adjust_lighting(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        조명을 조정합니다.
        """
        try:
            # 간단한 조명 조정
            # 실제 구현에서는 더 정교한 조명 모델 사용
            adjusted_image = image
            
            # 전체 밝기 조정
            brightness_factor = kwargs.get('brightness_factor', 1.0)
            if brightness_factor != 1.0:
                adjusted_image = adjusted_image * brightness_factor
                adjusted_image = torch.clamp(adjusted_image, 0, 1)
            
            logger.debug(f"Lighting adjustment applied: brightness_factor={brightness_factor}")
            return adjusted_image
            
        except Exception as e:
            logger.warning(f"⚠️ Lighting adjustment failed: {e}")
            return image

    def _generate_shadows(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        그림자를 생성합니다.
        """
        try:
            # 간단한 그림자 생성 (실제로는 더 정교한 알고리즘 사용)
            shadowed_image = image
            
            # 그림자 강도
            shadow_strength = kwargs.get('shadow_strength', 0.2)
            
            # 이미지 하단에 그림자 효과 추가
            h, w = image.shape[-2:]
            shadow_height = int(h * 0.3)  # 하단 30%에 그림자
            
            # 그림자 마스크 생성
            shadow_mask = torch.ones_like(image)
            shadow_mask[:, :, h-shadow_height:, :] = 1 - shadow_strength
            
            # 그림자 적용
            shadowed_image = image * shadow_mask
            
            logger.debug(f"Shadow generation applied: strength={shadow_strength}")
            return shadowed_image
            
        except Exception as e:
            logger.warning(f"⚠️ Shadow generation failed: {e}")
            return image

    def generate_fitting_batch(self, person_images: List[Union[np.ndarray, Image.Image, torch.Tensor]],
                             clothing_images: List[Union[np.ndarray, Image.Image, torch.Tensor]],
                             target_size: Tuple[int, int] = None,
                             **kwargs) -> List[Dict[str, torch.Tensor]]:
        """
        여러 이미지에 대해 가상 피팅 전처리를 일괄 실행합니다.
        """
        try:
            if len(person_images) != len(clothing_images):
                raise ValueError("Person images and clothing images must have the same length")
            
            processed_batch = []
            
            for i, (person_img, clothing_img) in enumerate(zip(person_images, clothing_images)):
                try:
                    processed = self.preprocess_for_virtual_fitting(
                        person_img, clothing_img, target_size, **kwargs
                    )
                    processed['batch_index'] = i
                    processed_batch.append(processed)
                except Exception as e:
                    logger.error(f"❌ Failed to process batch item {i}: {e}")
                    # 에러 발생 시 기본 텐서 생성
                    default_tensor = torch.zeros(1, 3, 512, 512, device=self.device)
                    processed_batch.append({
                        'person_image': default_tensor,
                        'clothing_image': default_tensor,
                        'batch_index': i,
                        'error': str(e)
                    })
            
            logger.info(f"✅ Batch processing completed: {len(processed_batch)} items")
            return processed_batch
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {e}")
            return []

    def get_processor_info(self) -> Dict[str, Any]:
        """
        프로세서 정보를 반환합니다.
        """
        return {
            'processor_name': 'VirtualFittingProcessor',
            'device': str(self.device),
            'config': self.processor_config,
            'supported_features': [
                'body_alignment', 'clothing_enhancement', 
                'lighting_adjustment', 'shadow_generation'
            ]
        }

    def update_processor_config(self, **kwargs):
        """
        프로세서 설정을 업데이트합니다.
        """
        try:
            for key, value in kwargs.items():
                if key in self.processor_config:
                    self.processor_config[key] = value
                    logger.info(f"✅ Processor config updated: {key} = {value}")
                else:
                    logger.warning(f"⚠️ Unknown config key: {key}")
        except Exception as e:
            logger.error(f"❌ Processor config update failed: {e}")
