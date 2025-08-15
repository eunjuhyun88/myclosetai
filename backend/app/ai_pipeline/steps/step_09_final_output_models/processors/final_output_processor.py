"""
🔥 Final Output Processor
=========================

최종 출력 생성을 위한 전처리 및 후처리 프로세서입니다.
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

class FinalOutputProcessor:
    """
    최종 출력 생성을 위한 전처리 및 후처리 프로세서
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
            'enable_quality_enhancement': True,
            'enable_noise_reduction': True,
            'enable_edge_enhancement': True,
            'enable_color_correction': True
        }
        
        logger.info(f"FinalOutputProcessor initialized on device: {self.device}")

    def preprocess_for_final_output(self, input_data: Union[np.ndarray, Image.Image, torch.Tensor],
                                 target_size: Tuple[int, int] = None,
                                 normalize: bool = True,
                                 **kwargs) -> torch.Tensor:
        """
        최종 출력 생성을 위한 입력 데이터 전처리
        
        Args:
            input_data: 입력 데이터
            target_size: 목표 크기
            normalize: 정규화 여부
            **kwargs: 추가 파라미터
            
        Returns:
            전처리된 데이터 텐서
        """
        try:
            # 목표 크기 설정
            if target_size is None:
                target_size = self.processor_config['default_input_size']
            
            # 이미지 타입 통일
            if isinstance(input_data, Image.Image):
                input_data = np.array(input_data)
            
            if isinstance(input_data, np.ndarray):
                # RGB로 변환
                if len(input_data.shape) == 2:
                    input_data = cv2.cvtColor(input_data, cv2.COLOR_GRAY2RGB)
                elif input_data.shape[2] == 4:
                    input_data = cv2.cvtColor(input_data, cv2.COLOR_RGBA2RGB)
                
                # 텐서로 변환
                input_data = torch.from_numpy(input_data).permute(2, 0, 1).float()
            elif isinstance(input_data, torch.Tensor):
                if len(input_data.shape) == 2:
                    input_data = input_data.unsqueeze(0).repeat(3, 1, 1)
                elif len(input_data.shape) == 3 and input_data.shape[0] == 1:
                    input_data = input_data.repeat(3, 1, 1)
            
            # 크기 조정
            if input_data.shape[-2:] != target_size:
                input_data = F.interpolate(input_data.unsqueeze(0), size=target_size, 
                                        mode='bilinear', align_corners=False).squeeze(0)
            
            # 정규화
            if normalize:
                if input_data.max() > 1.0:
                    input_data = input_data / 255.0
            
            # 배치 차원 추가
            if len(input_data.shape) == 3:
                input_data = input_data.unsqueeze(0)
            
            # 디바이스 이동
            input_data = input_data.to(self.device)
            
            logger.info(f"✅ Input preprocessing completed: {input_data.shape}")
            return input_data
            
        except Exception as e:
            logger.error(f"❌ Input preprocessing failed: {e}")
            raise

    def postprocess_final_output(self, output: torch.Tensor,
                               target_size: Tuple[int, int] = None,
                               denormalize: bool = True,
                               **kwargs) -> torch.Tensor:
        """
        최종 출력 후처리
        
        Args:
            output: 모델 출력
            target_size: 목표 크기
            denormalize: 역정규화 여부
            **kwargs: 추가 파라미터
            
        Returns:
            후처리된 출력 텐서
        """
        try:
            processed_output = output
            
            # 크기 조정
            if target_size is not None and output.shape[-2:] != target_size:
                processed_output = F.interpolate(processed_output, size=target_size, 
                                              mode='bilinear', align_corners=False)
            
            # 품질 향상 적용
            if self.processor_config['enable_quality_enhancement']:
                processed_output = self._apply_quality_enhancements(processed_output, **kwargs)
            
            # 역정규화
            if denormalize:
                processed_output = torch.clamp(processed_output, 0, 1)
                if processed_output.max() <= 1.0:
                    processed_output = processed_output * 255.0
            
            logger.info(f"✅ Output postprocessing completed: {processed_output.shape}")
            return processed_output
            
        except Exception as e:
            logger.error(f"❌ Output postprocessing failed: {e}")
            return output

    def _apply_quality_enhancements(self, output: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        품질 향상을 적용합니다.
        """
        enhanced_output = output
        
        # 노이즈 감소
        if self.processor_config['enable_noise_reduction']:
            enhanced_output = self._reduce_noise(enhanced_output, **kwargs)
        
        # 에지 향상
        if self.processor_config['enable_edge_enhancement']:
            enhanced_output = self._enhance_edges(enhanced_output, **kwargs)
        
        # 색상 보정
        if self.processor_config['enable_color_correction']:
            enhanced_output = self._correct_colors(enhanced_output, **kwargs)
        
        return enhanced_output

    def _reduce_noise(self, image: torch.Tensor, 
                     kernel_size: int = 3,
                     sigma: float = 1.0) -> torch.Tensor:
        """
        이미지 노이즈를 감소시킵니다.
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

    def _enhance_edges(self, image: torch.Tensor, 
                      strength: float = 0.5) -> torch.Tensor:
        """
        이미지 에지를 향상시킵니다.
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
        
        # 채도 조정 (간단한 방법)
        if saturation != 1.0 and corrected.shape[1] == 3:
            # 그레이스케일 계산
            gray = 0.299 * corrected[:, 0] + 0.587 * corrected[:, 1] + 0.114 * corrected[:, 2]
            gray = gray.unsqueeze(1)
            
            # 채도 조정
            corrected = gray + saturation * (corrected - gray)
            corrected = torch.clamp(corrected, 0, 1)
        
        return corrected

    def _create_gaussian_kernel2d(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """
        2D 가우시안 커널을 생성합니다.
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

    def generate_final_output_batch(self, input_data_list: List[Union[np.ndarray, Image.Image, torch.Tensor]],
                                  target_size: Tuple[int, int] = None,
                                  **kwargs) -> List[torch.Tensor]:
        """
        여러 입력에 대해 최종 출력을 일괄 생성합니다.
        
        Args:
            input_data_list: 입력 데이터 리스트
            target_size: 목표 크기
            **kwargs: 추가 파라미터
            
        Returns:
            전처리된 데이터 리스트
        """
        try:
            processed_data = []
            
            for i, input_data in enumerate(input_data_list):
                try:
                    processed = self.preprocess_for_final_output(
                        input_data, target_size=target_size, **kwargs
                    )
                    processed_data.append(processed)
                except Exception as e:
                    logger.error(f"❌ Failed to process input {i}: {e}")
                    # 에러 발생 시 원본 데이터 반환
                    if isinstance(input_data, torch.Tensor):
                        processed_data.append(input_data)
                    else:
                        # 기본 텐서 생성
                        default_tensor = torch.zeros(1, 3, 512, 512, device=self.device)
                        processed_data.append(default_tensor)
            
            logger.info(f"✅ Batch processing completed: {len(processed_data)} inputs")
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {e}")
            return []

    def apply_output_optimizations(self, output: torch.Tensor,
                                 optimizations: List[str] = None,
                                 **kwargs) -> torch.Tensor:
        """
        출력에 최적화를 적용합니다.
        
        Args:
            output: 입력 출력
            optimizations: 적용할 최적화 목록
            **kwargs: 최적화 파라미터
            
        Returns:
            최적화된 출력
        """
        try:
            if optimizations is None:
                optimizations = ['noise_reduction', 'edge_enhancement', 'color_correction']
            
            optimized_output = output
            
            for optimization in optimizations:
                if optimization == 'noise_reduction':
                    optimized_output = self._reduce_noise(optimized_output, **kwargs)
                elif optimization == 'edge_enhancement':
                    optimized_output = self._enhance_edges(optimized_output, **kwargs)
                elif optimization == 'color_correction':
                    optimized_output = self._correct_colors(optimized_output, **kwargs)
                elif optimization == 'resolution_enhancement':
                    optimized_output = self._enhance_resolution(optimized_output, **kwargs)
            
            logger.info(f"✅ Output optimizations applied: {optimizations}")
            return optimized_output
            
        except Exception as e:
            logger.error(f"❌ Output optimization failed: {e}")
            return output

    def _enhance_resolution(self, image: torch.Tensor, 
                          scale_factor: float = 2.0,
                          method: str = 'bicubic') -> torch.Tensor:
        """
        이미지 해상도를 향상시킵니다.
        """
        try:
            # 목표 크기 계산
            current_size = image.shape[-2:]
            target_size = (int(current_size[0] * scale_factor), int(current_size[1] * scale_factor))
            
            # 보간 방법 선택
            if method == 'bicubic':
                mode = 'bicubic'
            elif method == 'bilinear':
                mode = 'bilinear'
            elif method == 'nearest':
                mode = 'nearest'
            else:
                mode = 'bicubic'
            
            # 해상도 향상
            enhanced = F.interpolate(image, size=target_size, 
                                   mode=mode, align_corners=False)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"⚠️ Resolution enhancement failed: {e}")
            return image

    def get_processor_info(self) -> Dict[str, Any]:
        """
        프로세서 정보를 반환합니다.
        """
        return {
            'processor_name': 'FinalOutputProcessor',
            'device': str(self.device),
            'config': self.processor_config,
            'supported_optimizations': [
                'noise_reduction', 'edge_enhancement', 
                'color_correction', 'resolution_enhancement'
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
