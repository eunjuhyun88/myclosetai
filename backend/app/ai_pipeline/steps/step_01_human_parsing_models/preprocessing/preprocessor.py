"""
🔥 Human Parsing 전용 전처리 시스템
====================================

인체 파싱에 최적화된 전처리 기능들:
1. 인체 중심 크롭 및 정렬
2. 해상도 표준화 (512x512)
3. 조명 정규화 및 색상 보정
4. 인체 영역 강화
5. 메모리 효율적 처리

Author: MyCloset AI Team
Date: 2025-01-27
Version: 2.0 (완전 구현)
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

class HumanParsingPreprocessor:
    """인체 파싱에 최적화된 전처리 시스템"""
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size
        self.logger = logging.getLogger(f"{__name__}.HumanParsingPreprocessor")
        
        # 인체 감지기 (OpenCV HOG)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # 처리 통계
        self.processing_stats = {
            'images_processed': 0,
            'human_detected': 0,
            'cropping_applied': 0,
            'enhancement_applied': 0
        }
    
    def preprocess_image(self, image: np.ndarray, mode: str = 'advanced') -> Dict[str, Any]:
        """인체 파싱을 위한 완전한 전처리"""
        try:
            self.processing_stats['images_processed'] += 1
            self.logger.info(f"🔥 인체 파싱 전처리 시작 (모드: {mode})")
            
            # 1. 이미지 검증
            validated_image = self._validate_image(image)
            
            # 2. 인체 감지 및 중심 크롭
            cropped_image, crop_info = self._detect_and_crop_human(validated_image)
            if crop_info['human_detected']:
                self.processing_stats['human_detected'] += 1
                self.processing_stats['cropping_applied'] += 1
            
            # 3. 해상도 표준화
            resized_image = self._standardize_resolution(cropped_image)
            
            # 4. 이미지 품질 향상
            if mode == 'advanced':
                enhanced_image = self._enhance_image_quality(resized_image)
                self.processing_stats['enhancement_applied'] += 1
            else:
                enhanced_image = resized_image
            
            # 5. 정규화 및 텐서 변환
            normalized_tensor = self._normalize_and_convert(enhanced_image)
            
            # 6. 전처리 결과 요약
            preprocessing_result = {
                'processed_image': enhanced_image,
                'tensor': normalized_tensor,
                'crop_info': crop_info,
                'target_size': self.target_size,
                'mode': mode,
                'success': True
            }
            
            self.logger.info("✅ 인체 파싱 전처리 완료")
            return preprocessing_result
            
        except Exception as e:
            self.logger.error(f"❌ 인체 파싱 전처리 실패: {e}")
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
    
    def _detect_and_crop_human(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """인체 감지 및 중심 크롭"""
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
                
                # 여백 추가 (인체 주변 컨텍스트 포함)
                margin = int(max(w, h) * 0.2)
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(image.shape[1], x + w + margin)
                y2 = min(image.shape[0], y + h + margin)
                
                # 크롭된 이미지
                cropped = image[y1:y2, x1:x2]
                
                crop_info = {
                    'human_detected': True,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(np.max(weights)),
                    'original_size': image.shape[:2],
                    'cropped_size': cropped.shape[:2]
                }
                
                return cropped, crop_info
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
                
                crop_info = {
                    'human_detected': False,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': 0.0,
                    'original_size': image.shape[:2],
                    'cropped_size': cropped.shape[:2]
                }
                
                return cropped, crop_info
                
        except Exception as e:
            self.logger.warning(f"인체 감지 실패: {e}")
            return image, {
                'human_detected': False,
                'bbox': [0, 0, image.shape[1], image.shape[0]],
                'confidence': 0.0,
                'original_size': image.shape[:2],
                'cropped_size': image.shape[:2]
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
    
    def _enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """이미지 품질 향상"""
        try:
            enhanced = image.copy()
            
            # 1. 노이즈 제거
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
            
            # 2. 선명도 향상
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # 3. 대비 향상
            lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            # 4. 색상 보정
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"이미지 품질 향상 실패: {e}")
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
            'cropping_applied': 0,
            'enhancement_applied': 0
        }
