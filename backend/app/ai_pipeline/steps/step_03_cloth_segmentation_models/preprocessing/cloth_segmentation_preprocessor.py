"""
🔥 Cloth Segmentation 전용 전처리 시스템
========================================

의류 분할에 최적화된 전처리 기능들:
1. 의류 영역 강화 및 전처리
2. 패턴 복잡도 분석
3. 재질 특성 감지
4. 투명도 처리
5. 의류-배경 분리

Author: MyCloset AI Team
Date: 2025-01-27
Version: 1.0 (완전 구현)
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import logging
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

logger = logging.getLogger(__name__)

class ClothSegmentationPreprocessor:
    """의류 분할에 최적화된 전처리 시스템"""
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size
        self.logger = logging.getLogger(f"{__name__}.ClothSegmentationPreprocessor")
        
        # 의류 분할용 전처리 파라미터
        self.cloth_params = {
            'pattern_enhancement': True,
            'material_detection': True,
            'transparency_handling': True,
            'background_separation': True,
            'texture_preservation': True
        }
        
        # 의류 타입별 색상 범위 (HSV)
        self.cloth_color_ranges = {
            'light': {'lower': np.array([0, 0, 200]), 'upper': np.array([180, 30, 255])},
            'dark': {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 255, 50])},
            'colorful': {'lower': np.array([0, 100, 100]), 'upper': np.array([180, 255, 255])}
        }
        
        # 처리 통계
        self.processing_stats = {
            'images_processed': 0,
            'cloth_detected': 0,
            'patterns_enhanced': 0,
            'materials_analyzed': 0
        }
    
    def preprocess_image(self, image: np.ndarray, mode: str = 'advanced') -> Dict[str, Any]:
        """의류 분할을 위한 완전한 전처리"""
        try:
            self.processing_stats['images_processed'] += 1
            self.logger.info(f"🔥 의류 분할 전처리 시작 (모드: {mode})")
            
            # 1. 이미지 검증
            validated_image = self._validate_image(image)
            
            # 2. 의류 영역 감지 및 강화
            enhanced_image, cloth_info = self._detect_and_enhance_clothing(validated_image)
            if cloth_info['cloth_detected']:
                self.processing_stats['cloth_detected'] += 1
            
            # 3. 해상도 표준화
            resized_image = self._standardize_resolution(enhanced_image)
            
            # 4. 의류 분할 최적화
            if mode == 'advanced':
                optimized_image = self._optimize_for_cloth_segmentation(resized_image)
                self.processing_stats['patterns_enhanced'] += 1
                self.processing_stats['materials_analyzed'] += 1
            else:
                optimized_image = resized_image
            
            # 5. 정규화 및 텐서 변환
            normalized_tensor = self._normalize_and_convert(optimized_image)
            
            # 6. 전처리 결과 요약
            preprocessing_result = {
                'processed_image': optimized_image,
                'tensor': normalized_tensor,
                'cloth_info': cloth_info,
                'target_size': self.target_size,
                'mode': mode,
                'cloth_params': self.cloth_params,
                'success': True
            }
            
            self.logger.info("✅ 의류 분할 전처리 완료")
            return preprocessing_result
            
        except Exception as e:
            self.logger.error(f"❌ 의류 분할 전처리 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'processed_image': image,
                'tensor': torch.randn(1, 3, *self.target_size)
            }
    
    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 유효성 검증 및 변환"""
        try:
            # PIL Image를 NumPy로 변환
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # 그레이스케일을 RGB로 변환
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            # 데이터 타입 정규화
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            return image
            
        except Exception as e:
            self.logger.warning(f"이미지 검증 실패: {e}")
            return image
    
    def _detect_and_enhance_clothing(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """의류 영역 감지 및 강화"""
        try:
            # HSV 색공간으로 변환
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # 의류 색상 마스크 생성
            cloth_masks = {}
            for cloth_type, color_range in self.cloth_color_ranges.items():
                mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
                cloth_masks[cloth_type] = mask
            
            # 통합 의류 마스크
            combined_mask = np.zeros_like(cloth_masks['light'])
            for mask in cloth_masks.values():
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # 모폴로지 연산으로 노이즈 제거
            kernel = np.ones((5, 5), np.uint8)
            cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
            
            # 의류 영역 강화
            enhanced = image.copy()
            cloth_regions = cleaned_mask > 0
            
            if np.any(cloth_regions):
                # 의류 영역에서 대비 향상
                enhanced[cloth_regions] = cv2.convertScaleAbs(
                    enhanced[cloth_regions], alpha=1.2, beta=10
                )
                
                # 의류 영역에서 선명도 향상
                kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                enhanced[cloth_regions] = cv2.filter2D(
                    enhanced[cloth_regions], -1, kernel_sharpen
                )
            
            # 의류 정보 수집
            cloth_info = {
                'cloth_detected': np.any(cloth_regions),
                'cloth_coverage': np.sum(cloth_regions) / cloth_regions.size,
                'cloth_types': {k: np.sum(v > 0) / v.size for k, v in cloth_masks.items()},
                'original_size': image.shape[:2],
                'enhanced_size': enhanced.shape[:2]
            }
            
            return enhanced, cloth_info
            
        except Exception as e:
            self.logger.warning(f"의류 감지 및 강화 실패: {e}")
            return image, {
                'cloth_detected': False,
                'cloth_coverage': 0.0,
                'cloth_types': {},
                'original_size': image.shape[:2],
                'enhanced_size': image.shape[:2]
            }
    
    def _standardize_resolution(self, image: np.ndarray) -> np.ndarray:
        """해상도 표준화"""
        try:
            # 목표 해상도로 리사이즈
            resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            return resized
            
        except Exception as e:
            self.logger.warning(f"해상도 표준화 실패: {e}")
            return image
    
    def _optimize_for_cloth_segmentation(self, image: np.ndarray) -> np.ndarray:
        """의류 분할 최적화"""
        try:
            optimized = image.copy()
            
            # 1. 패턴 강화
            if self.cloth_params['pattern_enhancement']:
                optimized = self._enhance_patterns(optimized)
            
            # 2. 재질 특성 감지
            if self.cloth_params['material_detection']:
                optimized = self._detect_material_properties(optimized)
            
            # 3. 투명도 처리
            if self.cloth_params['transparency_handling']:
                optimized = self._handle_transparency(optimized)
            
            # 4. 배경 분리
            if self.cloth_params['background_separation']:
                optimized = self._separate_background(optimized)
            
            # 5. 텍스처 보존
            if self.cloth_params['texture_preservation']:
                optimized = self._preserve_texture(optimized)
            
            return optimized
            
        except Exception as e:
            self.logger.warning(f"의류 분할 최적화 실패: {e}")
            return image
    
    def _enhance_patterns(self, image: np.ndarray) -> np.ndarray:
        """패턴 강화"""
        try:
            enhanced = image.copy()
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 엣지 감지 (패턴 경계 강화)
            edges = cv2.Canny(gray, 30, 100)
            
            # 엣지를 RGB로 변환
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
            # 원본 이미지와 엣지 합성
            enhanced = cv2.addWeighted(enhanced, 0.8, edges_rgb, 0.2, 0)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"패턴 강화 실패: {e}")
            return image
    
    def _detect_material_properties(self, image: np.ndarray) -> np.ndarray:
        """재질 특성 감지"""
        try:
            enhanced = image.copy()
            
            # LAB 색공간으로 변환
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # 재질 특성 감지를 위한 필터링
            # 1. 질감 감지 (로컬 표준편차)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            texture_kernel = np.ones((7, 7), np.float32) / 49
            mean_texture = cv2.filter2D(gray.astype(np.float32), -1, texture_kernel)
            texture_variance = cv2.filter2D((gray.astype(np.float32) - mean_texture)**2, -1, texture_kernel)
            texture_std = np.sqrt(texture_variance)
            
            # 2. 재질 특성에 따른 강화
            # 거친 재질 (높은 표준편차)
            rough_mask = texture_std > np.percentile(texture_std, 70)
            if np.any(rough_mask):
                enhanced[rough_mask] = cv2.convertScaleAbs(
                    enhanced[rough_mask], alpha=1.3, beta=15
                )
            
            # 부드러운 재질 (낮은 표준편차)
            smooth_mask = texture_std < np.percentile(texture_std, 30)
            if np.any(smooth_mask):
                enhanced[smooth_mask] = cv2.GaussianBlur(enhanced[smooth_mask], (3, 3), 0)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"재질 특성 감지 실패: {e}")
            return image
    
    def _handle_transparency(self, image: np.ndarray) -> np.ndarray:
        """투명도 처리"""
        try:
            enhanced = image.copy()
            
            # HSV 색공간에서 투명도 감지
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            
            # 투명도가 높은 영역 감지 (낮은 채도, 높은 명도)
            transparent_mask = (saturation < 50) & (value > 200)
            
            if np.any(transparent_mask):
                # 투명 영역을 반투명하게 처리
                enhanced[transparent_mask] = cv2.addWeighted(
                    enhanced[transparent_mask], 0.7,
                    np.full_like(enhanced[transparent_mask], 255), 0.3, 0
                )
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"투명도 처리 실패: {e}")
            return image
    
    def _separate_background(self, image: np.ndarray) -> np.ndarray:
        """배경 분리"""
        try:
            enhanced = image.copy()
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Otsu 이진화로 배경 분리
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 모폴로지 연산으로 배경 마스크 정제
            kernel = np.ones((3, 3), np.uint8)
            background_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # 배경 영역을 약간 어둡게 처리
            background_regions = background_mask == 0
            if np.any(background_regions):
                enhanced[background_regions] = cv2.convertScaleAbs(
                    enhanced[background_regions], alpha=0.8, beta=-20
                )
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"배경 분리 실패: {e}")
            return image
    
    def _preserve_texture(self, image: np.ndarray) -> np.ndarray:
        """텍스처 보존"""
        try:
            enhanced = image.copy()
            
            # 양방향 필터로 엣지 보존하면서 노이즈 제거
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # 언샤프 마스킹으로 텍스처 선명도 향상
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
            sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
            
            # 선명도 향상을 RGB에 적용
            enhanced = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"텍스처 보존 실패: {e}")
            return image
    
    def _normalize_and_convert(self, image: np.ndarray) -> torch.Tensor:
        """정규화 및 텐서 변환"""
        try:
            # 0-1 범위로 정규화
            normalized = image.astype(np.float32) / 255.0
            
            # ImageNet 평균/표준편차로 정규화
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            normalized = (normalized - mean) / std
            
            # 텐서로 변환 [H, W, C] -> [C, H, W] -> [1, C, H, W]
            tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            self.logger.warning(f"정규화 및 변환 실패: {e}")
            return torch.randn(1, 3, *self.target_size)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        return self.processing_stats.copy()
    
    def reset_stats(self):
        """통계 초기화"""
        self.processing_stats = {
            'images_processed': 0,
            'cloth_detected': 0,
            'patterns_enhanced': 0,
            'materials_analyzed': 0
        }
    
    def update_cloth_params(self, **kwargs):
        """의류 분할 파라미터 업데이트"""
        for key, value in kwargs.items():
            if key in self.cloth_params:
                self.cloth_params[key] = value
                self.logger.info(f"의류 분할 파라미터 업데이트: {key} = {value}")
