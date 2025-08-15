"""
Deformation Utilities for Cloth Warping
의류 워핑을 위한 변형 유틸리티
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class DeformationUtils:
    """의류 변형 유틸리티 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def apply_thin_plate_spline(
        self, 
        image: np.ndarray, 
        source_points: np.ndarray, 
        target_points: np.ndarray
    ) -> np.ndarray:
        """Thin Plate Spline 변형 적용"""
        try:
            # TPS 변형 행렬 계산
            tps_matrix = cv2.estimateAffinePartial2D(source_points, target_points)[0]
            
            # 이미지에 변형 적용
            height, width = image.shape[:2]
            warped_image = cv2.warpAffine(image, tps_matrix, (width, height))
            
            return warped_image
        except Exception as e:
            self.logger.error(f"TPS 변형 실패: {e}")
            return image
    
    def apply_perspective_transform(
        self, 
        image: np.ndarray, 
        source_corners: np.ndarray, 
        target_corners: np.ndarray
    ) -> np.ndarray:
        """원근 변형 적용"""
        try:
            # 원근 변형 행렬 계산
            perspective_matrix = cv2.getPerspectiveTransform(source_corners, target_corners)
            
            # 이미지에 변형 적용
            height, width = image.shape[:2]
            warped_image = cv2.warpPerspective(image, perspective_matrix, (width, height))
            
            return warped_image
        except Exception as e:
            self.logger.error(f"원근 변형 실패: {e}")
            return image
    
    def apply_affine_transform(
        self, 
        image: np.ndarray, 
        rotation: float = 0, 
        scale: float = 1.0, 
        translation: Tuple[int, int] = (0, 0)
    ) -> np.ndarray:
        """아핀 변형 적용"""
        try:
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            
            # 변형 행렬 계산
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation, scale)
            rotation_matrix[0, 2] += translation[0]
            rotation_matrix[1, 2] += translation[1]
            
            # 이미지에 변형 적용
            warped_image = cv2.warpAffine(image, rotation_matrix, (width, height))
            
            return warped_image
        except Exception as e:
            self.logger.error(f"아핀 변형 실패: {e}")
            return image
    
    def calculate_deformation_quality(
        self, 
        original: np.ndarray, 
        warped: np.ndarray
    ) -> float:
        """변형 품질 계산"""
        try:
            # 구조적 유사성 계산
            ssim = self._calculate_ssim(original, warped)
            
            # 엣지 보존도 계산
            edge_preservation = self._calculate_edge_preservation(original, warped)
            
            # 종합 품질 점수
            quality_score = (ssim + edge_preservation) / 2.0
            
            return quality_score
        except Exception as e:
            self.logger.error(f"품질 계산 실패: {e}")
            return 0.0
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """SSIM 계산 (간단한 버전)"""
        try:
            # 그레이스케일 변환
            if len(img1.shape) == 3:
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            if len(img2.shape) == 3:
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # 정규화
            img1 = img1.astype(np.float64) / 255.0
            img2 = img2.astype(np.float64) / 255.0
            
            # 간단한 유사도 계산
            diff = np.abs(img1 - img2)
            similarity = 1.0 - np.mean(diff)
            
            return max(0.0, similarity)
        except Exception as e:
            self.logger.error(f"SSIM 계산 실패: {e}")
            return 0.0
    
    def _calculate_edge_preservation(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """엣지 보존도 계산"""
        try:
            # 그레이스케일 변환
            if len(img1.shape) == 3:
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            if len(img2.shape) == 3:
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # Sobel 엣지 검출
            edges1 = cv2.Sobel(img1, cv2.CV_64F, 1, 1)
            edges2 = cv2.Sobel(img2, cv2.CV_64F, 1, 1)
            
            # 엣지 강도 정규화
            edges1 = np.abs(edges1) / np.max(np.abs(edges1))
            edges2 = np.abs(edges2) / np.max(np.abs(edges2))
            
            # 엣지 보존도 계산
            edge_similarity = 1.0 - np.mean(np.abs(edges1 - edges2))
            
            return max(0.0, edge_similarity)
        except Exception as e:
            self.logger.error(f"엣지 보존도 계산 실패: {e}")
            return 0.0
    
    def optimize_deformation_parameters(
        self, 
        image: np.ndarray, 
        target_shape: Tuple[int, int]
    ) -> dict:
        """변형 파라미터 최적화"""
        try:
            # 기본 파라미터
            params = {
                'rotation': 0.0,
                'scale': 1.0,
                'translation_x': 0,
                'translation_y': 0
            }
            
            # 이미지 크기에 따른 스케일 조정
            current_height, current_width = image.shape[:2]
            target_height, target_width = target_shape
            
            scale_x = target_width / current_width
            scale_y = target_height / current_height
            params['scale'] = min(scale_x, scale_y)
            
            # 중앙 정렬을 위한 이동량 계산
            params['translation_x'] = (target_width - current_width * params['scale']) // 2
            params['translation_y'] = (target_height - current_height * params['scale']) // 2
            
            return params
        except Exception as e:
            self.logger.error(f"파라미터 최적화 실패: {e}")
            return {'rotation': 0.0, 'scale': 1.0, 'translation_x': 0, 'translation_y': 0}
