"""
Texture Utilities for Cloth Warping
의류 워핑을 위한 텍스처 유틸리티
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class TextureUtils:
    """텍스처 유틸리티 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_texture_features(self, image: np.ndarray) -> dict:
        """텍스처 특징 추출"""
        try:
            # 그레이스케일 변환
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            features = {}
            
            # GLCM 특징 (간단한 버전)
            features['contrast'] = self._calculate_contrast(gray)
            features['homogeneity'] = self._calculate_homogeneity(gray)
            features['energy'] = self._calculate_energy(gray)
            
            # 엣지 밀도
            features['edge_density'] = self._calculate_edge_density(gray)
            
            # 텍스처 방향성
            features['orientation'] = self._calculate_texture_orientation(gray)
            
            return features
        except Exception as e:
            self.logger.error(f"텍스처 특징 추출 실패: {e}")
            return {}
    
    def apply_texture_transfer(
        self, 
        source_image: np.ndarray, 
        target_image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """텍스처 전송 적용"""
        try:
            # 마스크가 없으면 전체 이미지에 적용
            if mask is None:
                mask = np.ones_like(target_image[:, :, 0])
            
            # 소스 이미지의 텍스처 특징 추출
            source_features = self.extract_texture_features(source_image)
            
            # 타겟 이미지에 텍스처 적용
            result = target_image.copy()
            
            # 간단한 텍스처 전송 (실제로는 더 복잡한 알고리즘 사용)
            if source_features.get('contrast', 0) > 0.5:
                # 대비가 높은 경우 엣지 강화
                result = self._enhance_edges(result, mask)
            
            if source_features.get('homogeneity', 0) > 0.7:
                # 균질성이 높은 경우 평활화
                result = self._smooth_texture(result, mask)
            
            return result
        except Exception as e:
            self.logger.error(f"텍스처 전송 실패: {e}")
            return target_image
    
    def preserve_texture_quality(
        self, 
        image: np.ndarray, 
        quality_threshold: float = 0.8
    ) -> np.ndarray:
        """텍스처 품질 보존"""
        try:
            # 텍스처 품질 평가
            quality = self._assess_texture_quality(image)
            
            if quality < quality_threshold:
                # 품질이 낮은 경우 향상
                image = self._enhance_texture_quality(image)
            
            return image
        except Exception as e:
            self.logger.error(f"텍스처 품질 보존 실패: {e}")
            return image
    
    def _calculate_contrast(self, gray_image: np.ndarray) -> float:
        """대비 계산"""
        try:
            # 표준편차를 이용한 대비 계산
            contrast = np.std(gray_image.astype(np.float64))
            return min(1.0, contrast / 255.0)
        except Exception:
            return 0.0
    
    def _calculate_homogeneity(self, gray_image: np.ndarray) -> float:
        """균질성 계산"""
        try:
            # 분산의 역수를 이용한 균질성 계산
            variance = np.var(gray_image.astype(np.float64))
            homogeneity = 1.0 / (1.0 + variance / 10000.0)
            return min(1.0, homogeneity)
        except Exception:
            return 0.0
    
    def _calculate_energy(self, gray_image: np.ndarray) -> float:
        """에너지 계산"""
        try:
            # 픽셀 값의 제곱합을 이용한 에너지 계산
            energy = np.sum(gray_image.astype(np.float64) ** 2)
            normalized_energy = energy / (gray_image.shape[0] * gray_image.shape[1] * 255 ** 2)
            return min(1.0, normalized_energy)
        except Exception:
            return 0.0
    
    def _calculate_edge_density(self, gray_image: np.ndarray) -> float:
        """엣지 밀도 계산"""
        try:
            # Sobel 엣지 검출
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            
            # 엣지 강도
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # 엣지 밀도 (임계값 이상인 픽셀 비율)
            threshold = np.mean(edge_magnitude)
            edge_density = np.sum(edge_magnitude > threshold) / edge_magnitude.size
            
            return min(1.0, edge_density)
        except Exception:
            return 0.0
    
    def _calculate_texture_orientation(self, gray_image: np.ndarray) -> float:
        """텍스처 방향성 계산"""
        try:
            # Sobel 엣지의 방향
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            
            # 방향각 계산
            angles = np.arctan2(sobel_y, sobel_x)
            
            # 방향성 히스토그램
            hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
            
            # 방향성 점수 (가장 우세한 방향의 비율)
            orientation_score = np.max(hist) / np.sum(hist)
            
            return min(1.0, orientation_score)
        except Exception:
            return 0.0
    
    def _enhance_edges(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """엣지 강화"""
        try:
            # Unsharp masking을 이용한 엣지 강화
            blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
            enhanced = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
            
            # 마스크 적용
            result = image.copy()
            result[mask > 0] = enhanced[mask > 0]
            
            return result
        except Exception:
            return image
    
    def _smooth_texture(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """텍스처 평활화"""
        try:
            # 가우시안 필터를 이용한 평활화
            smoothed = cv2.GaussianBlur(image, (5, 5), 1.0)
            
            # 마스크 적용
            result = image.copy()
            result[mask > 0] = smoothed[mask > 0]
            
            return result
        except Exception:
            return image
    
    def _assess_texture_quality(self, image: np.ndarray) -> float:
        """텍스처 품질 평가"""
        try:
            # 여러 품질 지표의 평균
            features = self.extract_texture_features(image)
            
            quality_indicators = [
                features.get('contrast', 0),
                features.get('homogeneity', 0),
                features.get('energy', 0),
                features.get('edge_density', 0)
            ]
            
            # 품질 점수 계산
            quality_score = np.mean(quality_indicators)
            
            return quality_score
        except Exception:
            return 0.0
    
    def _enhance_texture_quality(self, image: np.ndarray) -> np.ndarray:
        """텍스처 품질 향상"""
        try:
            # CLAHE를 이용한 대비 향상
            if len(image.shape) == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(image)
            
            return enhanced
        except Exception:
            return image
