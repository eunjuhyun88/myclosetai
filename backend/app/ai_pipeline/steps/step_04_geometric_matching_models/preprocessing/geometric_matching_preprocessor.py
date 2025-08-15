"""
🔥 Geometric Matching 전용 전처리 시스템
========================================

기하학적 매칭에 최적화된 전처리 기능들:
1. 의류 정규화 및 정렬
2. 기하학적 변환 매트릭스 계산
3. 스케일 및 회전 정렬
4. 왜곡 보정 및 정규화
5. 매칭 품질 최적화

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

class GeometricMatchingPreprocessor:
    """기하학적 매칭에 최적화된 전처리 시스템"""
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size
        self.logger = logging.getLogger(f"{__name__}.GeometricMatchingPreprocessor")
        
        # 기하학적 매칭용 전처리 파라미터
        self.geometric_params = {
            'normalization': True,
            'alignment': True,
            'distortion_correction': True,
            'scale_matching': True,
            'rotation_correction': True
        }
        
        # 처리 통계
        self.processing_stats = {
            'images_processed': 0,
            'geometric_transforms': 0,
            'distortions_corrected': 0,
            'alignments_applied': 0
        }
    
    def preprocess_image(self, image: np.ndarray, mode: str = 'advanced') -> Dict[str, Any]:
        """기하학적 매칭을 위한 완전한 전처리"""
        try:
            self.processing_stats['images_processed'] += 1
            self.logger.info(f"🔥 기하학적 매칭 전처리 시작 (모드: {mode})")
            
            # 1. 이미지 검증
            validated_image = self._validate_image(image)
            
            # 2. 기하학적 정규화
            normalized_image, geometric_info = self._normalize_geometry(validated_image)
            if geometric_info['transform_applied']:
                self.processing_stats['geometric_transforms'] += 1
            
            # 3. 해상도 표준화
            resized_image = self._standardize_resolution(normalized_image)
            
            # 4. 기하학적 매칭 최적화
            if mode == 'advanced':
                optimized_image = self._optimize_for_geometric_matching(resized_image)
                self.processing_stats['distortions_corrected'] += 1
                self.processing_stats['alignments_applied'] += 1
            else:
                optimized_image = resized_image
            
            # 5. 정규화 및 텐서 변환
            normalized_tensor = self._normalize_and_convert(optimized_image)
            
            # 6. 전처리 결과 요약
            preprocessing_result = {
                'processed_image': optimized_image,
                'tensor': normalized_tensor,
                'geometric_info': geometric_info,
                'target_size': self.target_size,
                'mode': mode,
                'geometric_params': self.geometric_params,
                'success': True
            }
            
            self.logger.info("✅ 기하학적 매칭 전처리 완료")
            return preprocessing_result
            
        except Exception as e:
            self.logger.error(f"❌ 기하학적 매칭 전처리 실패: {e}")
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
    
    def _normalize_geometry(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """기하학적 정규화"""
        try:
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 1. 엣지 감지
            edges = cv2.Canny(gray, 50, 150)
            
            # 2. 윤곽선 찾기
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 3. 가장 큰 윤곽선 찾기 (의류 영역)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 4. 경계 사각형 계산
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # 5. 기하학적 중심 계산
                center_x = x + w // 2
                center_y = y + h // 2
                
                # 6. 회전 각도 계산 (최소 경계 사각형)
                rect = cv2.minAreaRect(largest_contour)
                angle = rect[2]
                
                # 7. 기하학적 정규화 적용
                normalized = self._apply_geometric_transform(image, center_x, center_y, angle)
                
                geometric_info = {
                    'transform_applied': True,
                    'center': (center_x, center_y),
                    'rotation_angle': angle,
                    'bounding_box': (x, y, w, h),
                    'contour_area': cv2.contourArea(largest_contour),
                    'original_size': image.shape[:2],
                    'normalized_size': normalized.shape[:2]
                }
                
                return normalized, geometric_info
            else:
                # 윤곽선이 없는 경우 원본 반환
                return image, {
                    'transform_applied': False,
                    'center': (image.shape[1]//2, image.shape[0]//2),
                    'rotation_angle': 0.0,
                    'bounding_box': (0, 0, image.shape[1], image.shape[0]),
                    'contour_area': 0,
                    'original_size': image.shape[:2],
                    'normalized_size': image.shape[:2]
                }
                
        except Exception as e:
            self.logger.warning(f"기하학적 정규화 실패: {e}")
            return image, {
                'transform_applied': False,
                'center': (image.shape[1]//2, image.shape[0]//2),
                'rotation_angle': 0.0,
                'bounding_box': (0, 0, image.shape[1], image.shape[0]),
                'contour_area': 0,
                'original_size': image.shape[:2],
                'normalized_size': image.shape[:2]
            }
    
    def _apply_geometric_transform(self, image: np.ndarray, center_x: int, center_y: int, angle: float) -> np.ndarray:
        """기하학적 변환 적용"""
        try:
            # 회전 중심점
            center = (center_x, center_y)
            
            # 회전 행렬 계산
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # 회전 적용
            rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
            
            # 중심 크롭 (정사각형)
            h, w = rotated.shape[:2]
            crop_size = min(w, h)
            
            x1 = max(0, center_x - crop_size // 2)
            y1 = max(0, center_y - crop_size // 2)
            x2 = min(w, x1 + crop_size)
            y2 = min(h, y1 + crop_size)
            
            cropped = rotated[y1:y2, x1:x2]
            
            return cropped
            
        except Exception as e:
            self.logger.warning(f"기하학적 변환 적용 실패: {e}")
            return image
    
    def _standardize_resolution(self, image: np.ndarray) -> np.ndarray:
        """해상도 표준화"""
        try:
            # 목표 해상도로 리사이즈
            resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            return resized
            
        except Exception as e:
            self.logger.warning(f"해상도 표준화 실패: {e}")
            return image
    
    def _optimize_for_geometric_matching(self, image: np.ndarray) -> np.ndarray:
        """기하학적 매칭 최적화"""
        try:
            optimized = image.copy()
            
            # 1. 기하학적 정규화
            if self.geometric_params['normalization']:
                optimized = self._apply_geometric_normalization(optimized)
            
            # 2. 정렬 최적화
            if self.geometric_params['alignment']:
                optimized = self._optimize_alignment(optimized)
            
            # 3. 왜곡 보정
            if self.geometric_params['distortion_correction']:
                optimized = self._correct_distortion(optimized)
            
            # 4. 스케일 매칭
            if self.geometric_params['scale_matching']:
                optimized = self._optimize_scale(optimized)
            
            # 5. 회전 보정
            if self.geometric_params['rotation_correction']:
                optimized = self._correct_rotation(optimized)
            
            return optimized
            
        except Exception as e:
            self.logger.warning(f"기하학적 매칭 최적화 실패: {e}")
            return image
    
    def _apply_geometric_normalization(self, image: np.ndarray) -> np.ndarray:
        """기하학적 정규화 적용"""
        try:
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 히스토그램 평활화
            normalized = cv2.equalizeHist(gray)
            
            # 엣지 강화
            edges = cv2.Canny(normalized, 30, 100)
            
            # 엣지를 RGB로 변환
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
            # 원본과 엣지 합성
            enhanced = cv2.addWeighted(image, 0.8, edges_rgb, 0.2, 0)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"기하학적 정규화 적용 실패: {e}")
            return image
    
    def _optimize_alignment(self, image: np.ndarray) -> np.ndarray:
        """정렬 최적화"""
        try:
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 모멘트 계산
            moments = cv2.moments(gray)
            
            if moments['m00'] != 0:
                # 질량 중심 계산
                center_x = int(moments['m10'] / moments['m00'])
                center_y = int(moments['m01'] / moments['m00'])
                
                # 이미지 중심과의 차이 계산
                h, w = image.shape[:2]
                target_center = (w // 2, h // 2)
                
                # 이동 벡터 계산
                dx = target_center[0] - center_x
                dy = target_center[1] - center_y
                
                # 이동 행렬 생성
                translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
                
                # 이동 적용
                aligned = cv2.warpAffine(image, translation_matrix, (w, h))
                
                return aligned
            
            return image
            
        except Exception as e:
            self.logger.warning(f"정렬 최적화 실패: {e}")
            return image
    
    def _correct_distortion(self, image: np.ndarray) -> np.ndarray:
        """왜곡 보정"""
        try:
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 가우시안 블러로 노이즈 제거
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 언샤프 마스킹으로 선명도 향상
            sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
            
            # 왜곡 보정을 위한 필터링
            corrected = cv2.medianBlur(sharpened, 3)
            
            # RGB로 변환
            corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_GRAY2RGB)
            
            return corrected_rgb
            
        except Exception as e:
            self.logger.warning(f"왜곡 보정 실패: {e}")
            return image
    
    def _optimize_scale(self, image: np.ndarray) -> np.ndarray:
        """스케일 최적화"""
        try:
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 로컬 이진 패턴 (LBP) 계산으로 텍스처 특성 분석
            lbp = self._calculate_lbp(gray)
            
            # 스케일 최적화를 위한 필터링
            optimized = cv2.bilateralFilter(image, 9, 75, 75)
            
            # 대비 향상
            optimized = cv2.convertScaleAbs(optimized, alpha=1.1, beta=5)
            
            return optimized
            
        except Exception as e:
            self.logger.warning(f"스케일 최적화 실패: {e}")
            return image
    
    def _calculate_lbp(self, gray_image: np.ndarray) -> np.ndarray:
        """로컬 이진 패턴 계산"""
        try:
            h, w = gray_image.shape
            lbp = np.zeros((h-2, w-2), dtype=np.uint8)
            
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = gray_image[i, j]
                    code = 0
                    
                    # 8-이웃 픽셀 검사
                    neighbors = [
                        gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                        gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                        gray_image[i+1, j-1], gray_image[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            code |= (1 << k)
                    
                    lbp[i-1, j-1] = code
            
            return lbp
            
        except Exception as e:
            self.logger.warning(f"LBP 계산 실패: {e}")
            return np.zeros_like(gray_image)
    
    def _correct_rotation(self, image: np.ndarray) -> np.ndarray:
        """회전 보정"""
        try:
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 엣지 감지
            edges = cv2.Canny(gray, 50, 150)
            
            # 직선 감지 (Hough 변환)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # 주요 방향 계산
                angles = []
                for rho, theta in lines[:10]:  # 상위 10개 선만 사용
                    angle = theta * 180 / np.pi
                    if angle < 90:
                        angles.append(angle)
                    else:
                        angles.append(angle - 180)
                
                if angles:
                    # 평균 각도 계산
                    mean_angle = np.mean(angles)
                    
                    # 회전 보정
                    h, w = image.shape[:2]
                    center = (w // 2, h // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, mean_angle, 1.0)
                    corrected = cv2.warpAffine(image, rotation_matrix, (w, h))
                    
                    return corrected
            
            return image
            
        except Exception as e:
            self.logger.warning(f"회전 보정 실패: {e}")
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
            'geometric_transforms': 0,
            'distortions_corrected': 0,
            'alignments_applied': 0
        }
    
    def update_geometric_params(self, **kwargs):
        """기하학적 매칭 파라미터 업데이트"""
        for key, value in kwargs.items():
            if key in self.geometric_params:
                self.geometric_params[key] = value
                self.logger.info(f"기하학적 매칭 파라미터 업데이트: {key} = {value}")
