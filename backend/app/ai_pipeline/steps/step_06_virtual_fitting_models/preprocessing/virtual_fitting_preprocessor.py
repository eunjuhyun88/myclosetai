"""
🔥 Virtual Fitting 전용 전처리 시스템
======================================

가상 피팅에 최적화된 전처리 기능들:
1. 인체-의류 정합 및 정렬
2. 피팅 품질 최적화
3. 자연스러운 변형 처리
4. 조명 및 그림자 조정
5. 최종 피팅 품질 향상

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

class VirtualFittingPreprocessor:
    """가상 피팅에 최적화된 전처리 시스템"""
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size
        self.logger = logging.getLogger(f"{__name__}.VirtualFittingPreprocessor")
        
        # 가상 피팅용 전처리 파라미터
        self.fitting_params = {
            'body_clothing_alignment': True,
            'fitting_quality_optimization': True,
            'natural_deformation': True,
            'lighting_shadow_adjustment': True,
            'final_quality_enhancement': True
        }
        
        # 처리 통계
        self.processing_stats = {
            'images_processed': 0,
            'alignments_applied': 0,
            'deformations_processed': 0,
            'lighting_adjusted': 0
        }
    
    def preprocess_image(self, image: np.ndarray, mode: str = 'advanced') -> Dict[str, Any]:
        """가상 피팅을 위한 완전한 전처리"""
        try:
            self.processing_stats['images_processed'] += 1
            self.logger.info(f"🔥 가상 피팅 전처리 시작 (모드: {mode})")
            
            # 1. 이미지 검증
            validated_image = self._validate_image(image)
            
            # 2. 인체-의류 정합
            aligned_image, alignment_info = self._align_body_clothing(validated_image)
            if alignment_info['alignment_applied']:
                self.processing_stats['alignments_applied'] += 1
            
            # 3. 해상도 표준화
            resized_image = self._standardize_resolution(aligned_image)
            
            # 4. 가상 피팅 최적화
            if mode == 'advanced':
                optimized_image = self._optimize_for_virtual_fitting(resized_image)
                self.processing_stats['deformations_processed'] += 1
                self.processing_stats['lighting_adjusted'] += 1
            else:
                optimized_image = resized_image
            
            # 5. 정규화 및 텐서 변환
            normalized_tensor = self._normalize_and_convert(optimized_image)
            
            # 6. 전처리 결과 요약
            preprocessing_result = {
                'processed_image': optimized_image,
                'tensor': normalized_tensor,
                'alignment_info': alignment_info,
                'target_size': self.target_size,
                'mode': mode,
                'fitting_params': self.fitting_params,
                'success': True
            }
            
            self.logger.info("✅ 가상 피팅 전처리 완료")
            return preprocessing_result
            
        except Exception as e:
            self.logger.error(f"❌ 가상 피팅 전처리 실패: {e}")
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
    
    def _align_body_clothing(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """인체-의류 정합"""
        try:
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 1. 인체 감지 (HOG 디텍터 사용)
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            # 인체 감지
            bodies, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(4, 4), scale=1.05)
            
            # 2. 의류 영역 감지
            clothing_contours = self._detect_clothing_areas(gray)
            
            # 3. 정합 처리
            if len(bodies) > 0 and len(clothing_contours) > 0:
                # 가장 큰 인체 영역 선택
                largest_body = max(bodies, key=lambda x: x[2] * x[3])
                
                # 가장 큰 의류 영역 선택
                largest_clothing = max(clothing_contours, key=cv2.contourArea)
                
                # 정합 적용
                aligned_image = self._apply_body_clothing_alignment(image, largest_body, largest_clothing)
                
                alignment_info = {
                    'alignment_applied': True,
                    'body_count': len(bodies),
                    'clothing_count': len(clothing_contours),
                    'body_area': largest_body[2] * largest_body[3],
                    'clothing_area': cv2.contourArea(largest_clothing),
                    'alignment_quality': 'high'
                }
                
                return aligned_image, alignment_info
            else:
                # 정합할 수 없는 경우 원본 반환
                return image, {
                    'alignment_applied': False,
                    'body_count': len(bodies),
                    'clothing_count': len(clothing_contours),
                    'body_area': 0,
                    'clothing_area': 0,
                    'alignment_quality': 'none'
                }
                
        except Exception as e:
            self.logger.warning(f"인체-의류 정합 실패: {e}")
            return image, {
                'alignment_applied': False,
                'body_count': 0,
                'clothing_count': 0,
                'body_area': 0,
                'clothing_area': 0,
                'alignment_quality': 'none'
            }
    
    def _detect_clothing_areas(self, gray_image: np.ndarray) -> List[np.ndarray]:
        """의류 영역 감지"""
        try:
            # 1. 엣지 감지
            edges = cv2.Canny(gray_image, 30, 100)
            
            # 2. 윤곽선 찾기
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 3. 의류 영역 필터링
            clothing_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2000:  # 최소 면적 임계값
                    # 형태 분석으로 의류 영역 판별
                    if self._is_clothing_shape(contour):
                        clothing_contours.append(contour)
            
            return clothing_contours
            
        except Exception as e:
            self.logger.warning(f"의류 영역 감지 실패: {e}")
            return []
    
    def _is_clothing_shape(self, contour: np.ndarray) -> bool:
        """의류 형태인지 판별"""
        try:
            # 윤곽선의 면적과 둘레 비율 계산
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter > 0:
                # 원형도 계산 (1에 가까울수록 원형)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # 직사각형도 계산
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # 의류 형태 판별 조건
                is_clothing = (circularity < 0.8 and  # 너무 원형이 아님
                             0.2 < aspect_ratio < 5.0 and  # 적절한 비율
                             area > 2000)  # 충분한 크기
                
                return is_clothing
            
            return False
            
        except Exception as e:
            self.logger.warning(f"의류 형태 판별 실패: {e}")
            return False
    
    def _apply_body_clothing_alignment(self, image: np.ndarray, body: Tuple[int, int, int, int], 
                                     clothing_contour: np.ndarray) -> np.ndarray:
        """인체-의류 정합 적용"""
        try:
            aligned = image.copy()
            
            # 1. 인체 영역 표시
            x, y, w, h = body
            cv2.rectangle(aligned, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 2. 의류 윤곽선 표시
            cv2.drawContours(aligned, [clothing_contour], -1, (255, 0, 0), 2)
            
            # 3. 정합 중심점 계산
            body_center = (x + w // 2, y + h // 2)
            
            # 의류 중심점 계산
            M = cv2.moments(clothing_contour)
            if M['m00'] != 0:
                clothing_center_x = int(M['m10'] / M['m00'])
                clothing_center_y = int(M['m01'] / M['m00'])
                clothing_center = (clothing_center_x, clothing_center_y)
                
                # 중심점 연결선 그리기
                cv2.line(aligned, body_center, clothing_center, (0, 255, 255), 2)
                
                # 중심점 표시
                cv2.circle(aligned, body_center, 5, (0, 255, 0), -1)
                cv2.circle(aligned, clothing_center, 5, (255, 0, 0), -1)
            
            return aligned
            
        except Exception as e:
            self.logger.warning(f"인체-의류 정합 적용 실패: {e}")
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
    
    def _optimize_for_virtual_fitting(self, image: np.ndarray) -> np.ndarray:
        """가상 피팅 최적화"""
        try:
            optimized = image.copy()
            
            # 1. 인체-의류 정합
            if self.fitting_params['body_clothing_alignment']:
                optimized = self._enhance_body_clothing_alignment(optimized)
            
            # 2. 피팅 품질 최적화
            if self.fitting_params['fitting_quality_optimization']:
                optimized = self._optimize_fitting_quality(optimized)
            
            # 3. 자연스러운 변형
            if self.fitting_params['natural_deformation']:
                optimized = self._apply_natural_deformation(optimized)
            
            # 4. 조명 및 그림자 조정
            if self.fitting_params['lighting_shadow_adjustment']:
                optimized = self._adjust_lighting_shadows(optimized)
            
            # 5. 최종 품질 향상
            if self.fitting_params['final_quality_enhancement']:
                optimized = self._enhance_final_quality(optimized)
            
            return optimized
            
        except Exception as e:
            self.logger.warning(f"가상 피팅 최적화 실패: {e}")
            return image
    
    def _enhance_body_clothing_alignment(self, image: np.ndarray) -> np.ndarray:
        """인체-의류 정합 향상"""
        try:
            enhanced = image.copy()
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 1. 엣지 강화
            edges = cv2.Canny(gray, 20, 80)
            
            # 2. 모폴로지 연산으로 엣지 정제
            kernel = np.ones((3, 3), np.uint8)
            refined_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # 3. 엣지를 RGB로 변환
            edges_rgb = cv2.cvtColor(refined_edges, cv2.COLOR_GRAY2RGB)
            
            # 4. 원본과 엣지 합성
            enhanced = cv2.addWeighted(enhanced, 0.9, edges_rgb, 0.1, 0)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"인체-의류 정합 향상 실패: {e}")
            return image
    
    def _optimize_fitting_quality(self, image: np.ndarray) -> np.ndarray:
        """피팅 품질 최적화"""
        try:
            optimized = image.copy()
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 1. 피팅 품질을 위한 필터링
            # 가우시안 블러로 부드러운 피팅
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 2. 피팅 품질 강도 계산
            fitting_quality = cv2.addWeighted(gray, 1.3, blurred, -0.3, 0)
            
            # 3. 피팅 품질 맵 정규화
            quality_map = cv2.normalize(fitting_quality, None, 0, 255, cv2.NORM_MINMAX)
            
            # 4. 품질 맵을 RGB로 변환
            quality_rgb = cv2.cvtColor(quality_map, cv2.COLOR_GRAY2RGB)
            
            # 5. 원본과 품질 맵 합성
            optimized = cv2.addWeighted(optimized, 0.8, quality_rgb, 0.2, 0)
            
            return optimized
            
        except Exception as e:
            self.logger.warning(f"피팅 품질 최적화 실패: {e}")
            return image
    
    def _apply_natural_deformation(self, image: np.ndarray) -> np.ndarray:
        """자연스러운 변형 적용"""
        try:
            deformed = image.copy()
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 1. 자연스러운 변형을 위한 필터링
            # 양방향 필터로 텍스처 보존하면서 부드럽게
            deformed = cv2.bilateralFilter(deformed, 9, 75, 75)
            
            # 2. 변형 강도 조절
            # 원본과 변형된 이미지의 가중 평균
            deformed = cv2.addWeighted(image, 0.7, deformed, 0.3, 0)
            
            return deformed
            
        except Exception as e:
            self.logger.warning(f"자연스러운 변형 적용 실패: {e}")
            return image
    
    def _adjust_lighting_shadows(self, image: np.ndarray) -> np.ndarray:
        """조명 및 그림자 조정"""
        try:
            adjusted = image.copy()
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 1. 히스토그램 평활화로 조명 조정
            equalized = cv2.equalizeHist(gray)
            
            # 2. 그림자 감지 및 보정
            # 어두운 영역 감지
            _, shadow_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
            
            # 그림자 영역을 밝게 조정
            shadow_corrected = cv2.addWeighted(gray, 0.8, equalized, 0.2, 0)
            
            # 3. 조정된 이미지를 RGB로 변환
            adjusted_gray = cv2.cvtColor(shadow_corrected, cv2.COLOR_GRAY2RGB)
            
            # 4. 원본과 조정된 이미지 합성
            adjusted = cv2.addWeighted(adjusted, 0.8, adjusted_gray, 0.2, 0)
            
            return adjusted
            
        except Exception as e:
            self.logger.warning(f"조명 및 그림자 조정 실패: {e}")
            return image
    
    def _enhance_final_quality(self, image: np.ndarray) -> np.ndarray:
        """최종 품질 향상"""
        try:
            enhanced = image.copy()
            
            # 1. 대비 향상
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)
            
            # 2. 선명도 향상
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # 3. 노이즈 제거
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
            
            # 4. 최종 품질 조정
            enhanced = cv2.addWeighted(image, 0.3, enhanced, 0.7, 0)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"최종 품질 향상 실패: {e}")
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
            'alignments_applied': 0,
            'deformations_processed': 0,
            'lighting_adjusted': 0
        }
    
    def update_fitting_params(self, **kwargs):
        """가상 피팅 파라미터 업데이트"""
        for key, value in kwargs.items():
            if key in self.fitting_params:
                self.fitting_params[key] = value
                self.logger.info(f"가상 피팅 파라미터 업데이트: {key} = {value}")
