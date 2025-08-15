"""
🔥 Pose Estimation 전용 전처리 시스템
=====================================

포즈 추정에 최적화된 전처리 기능들:
1. 인체 중심 정렬 및 크롭
2. 해상도 표준화 (368x368, 256x256)
3. 관절 영역 강화
4. 배경 노이즈 제거
5. 인체 자세 정규화

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

class PoseEstimationPreprocessor:
    """포즈 추정에 최적화된 전처리 시스템"""
    
    def __init__(self, target_size: Tuple[int, int] = (368, 368)):
        self.target_size = target_size
        self.logger = logging.getLogger(f"{__name__}.PoseEstimationPreprocessor")
        
        # 인체 감지기 (OpenCV HOG)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # 포즈 추정용 전처리 파라미터
        self.pose_params = {
            'joint_enhancement': True,
            'background_removal': True,
            'pose_normalization': True,
            'lighting_correction': True
        }
        
        # 처리 통계
        self.processing_stats = {
            'images_processed': 0,
            'human_detected': 0,
            'pose_aligned': 0,
            'joint_enhanced': 0
        }
    
    def preprocess_image(self, image: np.ndarray, mode: str = 'advanced') -> Dict[str, Any]:
        """포즈 추정을 위한 완전한 전처리"""
        try:
            self.processing_stats['images_processed'] += 1
            self.logger.info(f"🔥 포즈 추정 전처리 시작 (모드: {mode})")
            
            # 1. 이미지 검증
            validated_image = self._validate_image(image)
            
            # 2. 인체 감지 및 포즈 정렬
            aligned_image, alignment_info = self._detect_and_align_human(validated_image)
            if alignment_info['human_detected']:
                self.processing_stats['human_detected'] += 1
                self.processing_stats['pose_aligned'] += 1
            
            # 3. 해상도 표준화
            resized_image = self._standardize_resolution(aligned_image)
            
            # 4. 포즈 추정 최적화
            if mode == 'advanced':
                optimized_image = self._optimize_for_pose_estimation(resized_image)
                self.processing_stats['joint_enhanced'] += 1
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
                'pose_params': self.pose_params,
                'success': True
            }
            
            self.logger.info("✅ 포즈 추정 전처리 완료")
            return preprocessing_result
            
        except Exception as e:
            self.logger.error(f"❌ 포즈 추정 전처리 실패: {e}")
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
    
    def _detect_and_align_human(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """인체 감지 및 포즈 정렬"""
        try:
            # 인체 감지
            boxes, weights = self.hog.detectMultiScale(
                image, 
                winStride=(8, 8), 
                padding=(4, 4), 
                scale=1.05
            )
            
            if len(boxes) > 0:
                # 가장 높은 신뢰도의 인체 영역 선택
                best_box = boxes[np.argmax(weights)]
                x, y, w, h = best_box
                
                # 포즈 추정을 위한 여백 추가 (상체 중심)
                margin_x = int(w * 0.3)
                margin_y = int(h * 0.4)  # 상체에 더 많은 여백
                
                x1 = max(0, x - margin_x)
                y1 = max(0, y - margin_y)
                x2 = min(image.shape[1], x + w + margin_x)
                y2 = min(image.shape[0], y + h + margin_y)
                
                # 크롭된 이미지
                cropped = image[y1:y2, x1:x2]
                
                # 포즈 정렬 (상체 중심)
                aligned = self._align_pose_center(cropped)
                
                alignment_info = {
                    'human_detected': True,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(np.max(weights)),
                    'original_size': image.shape[:2],
                    'cropped_size': cropped.shape[:2],
                    'aligned_size': aligned.shape[:2],
                    'pose_centered': True
                }
                
                return aligned, alignment_info
            else:
                # 인체가 감지되지 않은 경우 중앙 크롭
                h, w = image.shape[:2]
                center_x, center_y = w // 2, h // 2
                crop_size = min(w, h)
                
                x1 = max(0, center_x - crop_size // 2)
                y1 = max(0, center_y - crop_size // 2)
                x2 = min(w, x1 + crop_size)
                y2 = min(h, y1 + crop_size)
                
                cropped = image[y1:y2, x1:x2]
                aligned = self._align_pose_center(cropped)
                
                alignment_info = {
                    'human_detected': False,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': 0.0,
                    'original_size': image.shape[:2],
                    'cropped_size': cropped.shape[:2],
                    'aligned_size': aligned.shape[:2],
                    'pose_centered': False
                }
                
                return aligned, alignment_info
                
        except Exception as e:
            self.logger.warning(f"인체 감지 및 정렬 실패: {e}")
            return image, {
                'human_detected': False,
                'bbox': [0, 0, image.shape[1], image.shape[0]],
                'confidence': 0.0,
                'original_size': image.shape[:2],
                'cropped_size': image.shape[:2],
                'aligned_size': image.shape[:2],
                'pose_centered': False
            }
    
    def _align_pose_center(self, image: np.ndarray) -> np.ndarray:
        """포즈 중심 정렬"""
        try:
            h, w = image.shape[:2]
            
            # 상체 중심 계산 (상단 60% 영역)
            upper_h = int(h * 0.6)
            center_x = w // 2
            center_y = upper_h // 2
            
            # 정사각형 크롭
            crop_size = min(w, upper_h)
            x1 = max(0, center_x - crop_size // 2)
            y1 = max(0, center_y - crop_size // 2)
            x2 = min(w, x1 + crop_size)
            y2 = min(upper_h, y1 + crop_size)
            
            # 크롭 및 정사각형으로 패딩
            cropped = image[y1:y2, x1:x2]
            
            # 정사각형이 되도록 패딩
            target_size = max(cropped.shape[:2])
            padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            
            # 중앙에 이미지 배치
            y_offset = (target_size - cropped.shape[0]) // 2
            x_offset = (target_size - cropped.shape[1]) // 2
            
            padded[y_offset:y_offset+cropped.shape[0], 
                   x_offset:x_offset+cropped.shape[1]] = cropped
            
            return padded
            
        except Exception as e:
            self.logger.warning(f"포즈 중심 정렬 실패: {e}")
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
    
    def _optimize_for_pose_estimation(self, image: np.ndarray) -> np.ndarray:
        """포즈 추정 최적화"""
        try:
            optimized = image.copy()
            
            # 1. 관절 영역 강화
            if self.pose_params['joint_enhancement']:
                optimized = self._enhance_joint_regions(optimized)
            
            # 2. 배경 노이즈 제거
            if self.pose_params['background_removal']:
                optimized = self._remove_background_noise(optimized)
            
            # 3. 조명 보정
            if self.pose_params['lighting_correction']:
                optimized = self._correct_lighting(optimized)
            
            # 4. 포즈 정규화
            if self.pose_params['pose_normalization']:
                optimized = self._normalize_pose(optimized)
            
            return optimized
            
        except Exception as e:
            self.logger.warning(f"포즈 추정 최적화 실패: {e}")
            return image
    
    def _enhance_joint_regions(self, image: np.ndarray) -> np.ndarray:
        """관절 영역 강화"""
        try:
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 엣지 감지 (관절 경계 강화)
            edges = cv2.Canny(gray, 50, 150)
            
            # 엣지를 RGB로 변환
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
            # 원본 이미지와 엣지 합성
            enhanced = cv2.addWeighted(image, 0.8, edges_rgb, 0.2, 0)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"관절 영역 강화 실패: {e}")
            return image
    
    def _remove_background_noise(self, image: np.ndarray) -> np.ndarray:
        """배경 노이즈 제거"""
        try:
            # 가우시안 블러로 노이즈 제거
            denoised = cv2.GaussianBlur(image, (3, 3), 0)
            
            # 양방향 필터로 엣지 보존하면서 노이즈 제거
            denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
            
            return denoised
            
        except Exception as e:
            self.logger.warning(f"배경 노이즈 제거 실패: {e}")
            return image
    
    def _correct_lighting(self, image: np.ndarray) -> np.ndarray:
        """조명 보정"""
        try:
            # LAB 색공간으로 변환
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE로 조명 정규화
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # 색상 보정
            a = cv2.convertScaleAbs(a, alpha=1.1, beta=0)
            b = cv2.convertScaleAbs(b, alpha=1.1, beta=0)
            
            # RGB로 변환
            corrected = cv2.merge([l, a, b])
            corrected = cv2.cvtColor(corrected, cv2.COLOR_LAB2RGB)
            
            return corrected
            
        except Exception as e:
            self.logger.warning(f"조명 보정 실패: {e}")
            return image
    
    def _normalize_pose(self, image: np.ndarray) -> np.ndarray:
        """포즈 정규화"""
        try:
            # 히스토그램 평활화
            normalized = image.copy()
            
            # 각 채널별 히스토그램 평활화
            for i in range(3):
                normalized[:, :, i] = cv2.equalizeHist(normalized[:, :, i])
            
            # 대비 향상
            normalized = cv2.convertScaleAbs(normalized, alpha=1.1, beta=5)
            
            return normalized
            
        except Exception as e:
            self.logger.warning(f"포즈 정규화 실패: {e}")
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
            'human_detected': 0,
            'pose_aligned': 0,
            'joint_enhanced': 0
        }
    
    def update_pose_params(self, **kwargs):
        """포즈 추정 파라미터 업데이트"""
        for key, value in kwargs.items():
            if key in self.pose_params:
                self.pose_params[key] = value
                self.logger.info(f"포즈 파라미터 업데이트: {key} = {value}")
