"""
🔥 Quality Assessment Processor
==============================

품질 평가를 위한 이미지 전처리 및 후처리 프로세서입니다.
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

class QualityProcessor:
    """
    품질 평가를 위한 이미지 전처리 및 후처리 프로세서
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
        
        logger.info(f"QualityProcessor initialized on device: {self.device}")

    def preprocess_for_quality_assessment(self, image: Union[np.ndarray, Image.Image, torch.Tensor],
                                       target_size: Tuple[int, int] = (224, 224),
                                       normalize: bool = True) -> torch.Tensor:
        """
        품질 평가를 위한 이미지 전처리
        
        Args:
            image: 입력 이미지
            target_size: 목표 크기
            normalize: 정규화 여부
            
        Returns:
            전처리된 이미지 텐서
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
                image = image / 255.0
            
            # 배치 차원 추가
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            
            # 디바이스 이동
            image = image.to(self.device)
            
            logger.info(f"✅ Image preprocessing completed: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"❌ Image preprocessing failed: {e}")
            raise

    def enhance_image_quality(self, image: torch.Tensor, 
                            enhancement_type: str = 'denoise',
                            **kwargs) -> torch.Tensor:
        """
        이미지 품질 향상
        
        Args:
            image: 입력 이미지 텐서
            enhancement_type: 향상 타입
            **kwargs: 추가 파라미터
            
        Returns:
            향상된 이미지 텐서
        """
        try:
            if enhancement_type == 'denoise':
                return self._denoise_image(image, **kwargs)
            elif enhancement_type == 'sharpen':
                return self._sharpen_image(image, **kwargs)
            elif enhancement_type == 'contrast':
                return self._enhance_contrast(image, **kwargs)
            elif enhancement_type == 'brightness':
                return self._enhance_brightness(image, **kwargs)
            else:
                logger.warning(f"⚠️ Unknown enhancement type: {enhancement_type}")
                return image
                
        except Exception as e:
            logger.error(f"❌ Image enhancement failed: {e}")
            return image

    def _denoise_image(self, image: torch.Tensor, 
                      kernel_size: int = 3,
                      sigma: float = 1.0) -> torch.Tensor:
        """
        이미지 노이즈 제거
        """
        # 가우시안 필터 적용
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # 2D 가우시안 커널 생성
        kernel = self._create_gaussian_kernel2d(kernel_size, sigma)
        kernel = kernel.to(self.device)
        
        # 각 채널에 대해 컨볼루션 적용
        denoised = torch.zeros_like(image)
        for c in range(image.shape[1]):
            denoised[:, c:c+1] = F.conv2d(
                image[:, c:c+1], 
                kernel.unsqueeze(0).unsqueeze(0),
                padding=kernel_size // 2
            )
        
        return denoised

    def _sharpen_image(self, image: torch.Tensor, 
                      strength: float = 1.0) -> torch.Tensor:
        """
        이미지 선명도 향상
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

    def _enhance_contrast(self, image: torch.Tensor, 
                         alpha: float = 1.2,
                         beta: float = 0.0) -> torch.Tensor:
        """
        이미지 대비 향상
        """
        # 히스토그램 평활화와 유사한 효과
        mean_val = image.mean()
        enhanced = alpha * (image - mean_val) + mean_val + beta
        return torch.clamp(enhanced, 0, 1)

    def _enhance_brightness(self, image: torch.Tensor, 
                          beta: float = 0.1) -> torch.Tensor:
        """
        이미지 밝기 향상
        """
        enhanced = image + beta
        return torch.clamp(enhanced, 0, 1)

    def _create_gaussian_kernel2d(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """
        2D 가우시안 커널 생성
        """
        kernel = torch.zeros(kernel_size, kernel_size)
        center = kernel_size // 2
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[i, j] = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        # 정규화
        kernel = kernel / kernel.sum()
        return kernel

    def extract_quality_features(self, image: torch.Tensor) -> Dict[str, float]:
        """
        이미지에서 품질 관련 특징 추출
        
        Args:
            image: 입력 이미지 텐서
            
        Returns:
            품질 특징 딕셔너리
        """
        try:
            features = {}
            
            # 밝기
            features['brightness'] = image.mean().item()
            
            # 대비 (표준편차)
            features['contrast'] = image.std().item()
            
            # 선명도 (라플라시안 분산)
            features['sharpness'] = self._calculate_sharpness(image)
            
            # 노이즈 레벨
            features['noise_level'] = self._estimate_noise_level(image)
            
            # 색상 분포
            features['color_variance'] = self._calculate_color_variance(image)
            
            logger.info(f"✅ Quality features extracted: {len(features)} features")
            return features
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            return {}

    def _calculate_sharpness(self, image: torch.Tensor) -> float:
        """
        이미지 선명도 계산 (라플라시안 분산)
        """
        try:
            # 그레이스케일로 변환
            if image.shape[1] == 3:
                gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
            else:
                gray = image[:, 0]
            
            # 라플라시안 필터
            laplacian_kernel = torch.tensor([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            
            laplacian = F.conv2d(gray.unsqueeze(1), laplacian_kernel, padding=1)
            
            # 분산 계산
            sharpness = laplacian.var().item()
            return sharpness
            
        except Exception as e:
            logger.warning(f"⚠️ Sharpness calculation failed: {e}")
            return 0.0

    def _estimate_noise_level(self, image: torch.Tensor) -> float:
        """
        이미지 노이즈 레벨 추정
        """
        try:
            # 고주파 성분 추출
            high_pass_kernel = torch.tensor([
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]
            ], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            
            high_pass = F.conv2d(image[:, 0:1], high_pass_kernel, padding=1)
            
            # 노이즈 레벨 추정
            noise_level = high_pass.std().item()
            return noise_level
            
        except Exception as e:
            logger.warning(f"⚠️ Noise level estimation failed: {e}")
            return 0.0

    def _calculate_color_variance(self, image: torch.Tensor) -> float:
        """
        색상 분산 계산
        """
        try:
            if image.shape[1] == 3:
                # RGB 채널 간 분산
                color_variance = image.var(dim=1).mean().item()
            else:
                color_variance = 0.0
            
            return color_variance
            
        except Exception as e:
            logger.warning(f"⚠️ Color variance calculation failed: {e}")
            return 0.0

    def batch_process(self, images: List[torch.Tensor],
                     preprocessing: bool = True,
                     enhancement: bool = False,
                     feature_extraction: bool = True) -> Dict[str, Any]:
        """
        여러 이미지를 일괄 처리
        
        Args:
            images: 입력 이미지 리스트
            preprocessing: 전처리 여부
            enhancement: 품질 향상 여부
            feature_extraction: 특징 추출 여부
            
        Returns:
            처리 결과 딕셔너리
        """
        results = {
            'processed_images': [],
            'features': [],
            'processing_time': 0.0
        }
        
        start_time = torch.cuda.Event() if torch.cuda.is_available() else None
        
        for i, image in enumerate(images):
            try:
                processed_image = image
                
                # 전처리
                if preprocessing:
                    processed_image = self.preprocess_for_quality_assessment(
                        image, target_size=(224, 224), normalize=True
                    )
                
                # 품질 향상
                if enhancement:
                    processed_image = self.enhance_image_quality(
                        processed_image, enhancement_type='denoise'
                    )
                
                # 특징 추출
                features = {}
                if feature_extraction:
                    features = self.extract_quality_features(processed_image)
                    features['image_index'] = i
                
                results['processed_images'].append(processed_image)
                results['features'].append(features)
                
            except Exception as e:
                logger.error(f"❌ Failed to process image {i}: {e}")
                results['processed_images'].append(image)
                results['features'].append({'image_index': i, 'error': str(e)})
        
        # 처리 시간 측정
        if start_time:
            end_time = torch.cuda.Event()
            end_time.record()
            torch.cuda.synchronize()
            results['processing_time'] = start_time.elapsed_time(end_time) / 1000.0
        
        logger.info(f"✅ Batch processing completed: {len(images)} images")
        return results
