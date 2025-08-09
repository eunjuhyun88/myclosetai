#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Feature Extraction
=====================================================================

특성 추출 관련 기능들을 분리한 모듈

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

import logging
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

def _extract_cloth_features(self, masks: Dict[str, np.ndarray], image: np.ndarray) -> Dict[str, Any]:
    """의류 특성 추출"""
    try:
        features = {}
        
        for mask_type, mask in masks.items():
            if mask is None or not np.any(mask):
                continue
            
            mask_features = {
                'area': np.sum(mask),
                'centroid': _calculate_centroid(mask),
                'bbox': _calculate_bounding_box(mask),
                'contours': _extract_cloth_contours(mask),
                'aspect_ratio': _calculate_aspect_ratio(mask),
                'compactness': _calculate_compactness(mask)
            }
            
            features[mask_type] = mask_features
        
        return features
        
    except Exception as e:
        logger.warning(f"의류 특성 추출 실패: {e}")
        return {}

def _calculate_centroid(self, mask: np.ndarray) -> Tuple[float, float]:
    """중심점 계산"""
    try:
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            return (0.0, 0.0)
        
        centroid_y = np.mean(y_coords)
        centroid_x = np.mean(x_coords)
        
        return (centroid_x, centroid_y)
        
    except Exception as e:
        logger.warning(f"중심점 계산 실패: {e}")
        return (0.0, 0.0)

def _calculate_bounding_box(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
    """바운딩 박스 계산"""
    try:
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            return (0, 0, 0, 0)
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        return (x_min, y_min, x_max, y_max)
        
    except Exception as e:
        logger.warning(f"바운딩 박스 계산 실패: {e}")
        return (0, 0, 0, 0)

def _extract_cloth_contours(self, mask: np.ndarray) -> List[np.ndarray]:
    """의류 컨투어 추출"""
    try:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
        
    except Exception as e:
        logger.warning(f"컨투어 추출 실패: {e}")
        return []

def _calculate_aspect_ratio(self, mask: np.ndarray) -> float:
    """종횡비 계산"""
    try:
        bbox = _calculate_bounding_box(self, mask)
        if bbox[2] == bbox[0] or bbox[3] == bbox[1]:
            return 1.0
        
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        return width / height
        
    except Exception as e:
        logger.warning(f"종횡비 계산 실패: {e}")
        return 1.0

def _calculate_compactness(self, mask: np.ndarray) -> float:
    """조밀도 계산"""
    try:
        area = np.sum(mask)
        if area == 0:
            return 0.0
        
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        
        # 가장 큰 컨투어의 둘레 계산
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter == 0:
            return 0.0
        
        # 조밀도 = 4π * 면적 / 둘레^2
        compactness = 4 * np.pi * area / (perimeter ** 2)
        
        return min(compactness, 1.0)
        
    except Exception as e:
        logger.warning(f"조밀도 계산 실패: {e}")
        return 0.0

def _get_cloth_bounding_boxes(self, masks: Dict[str, np.ndarray]) -> Dict[str, Dict[str, int]]:
    """의류 바운딩 박스들 조회"""
    try:
        bounding_boxes = {}
        
        for mask_type, mask in masks.items():
            if mask is not None and np.any(mask):
                bbox = _calculate_bounding_box(self, mask)
                bounding_boxes[mask_type] = {
                    'x_min': bbox[0],
                    'y_min': bbox[1],
                    'x_max': bbox[2],
                    'y_max': bbox[3],
                    'width': bbox[2] - bbox[0],
                    'height': bbox[3] - bbox[1]
                }
        
        return bounding_boxes
        
    except Exception as e:
        logger.warning(f"바운딩 박스들 조회 실패: {e}")
        return {}

def _get_cloth_centroids(self, masks: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, float]]:
    """의류 중심점들 조회"""
    try:
        centroids = {}
        
        for mask_type, mask in masks.items():
            if mask is not None and np.any(mask):
                centroid = _calculate_centroid(self, mask)
                centroids[mask_type] = centroid
        
        return centroids
        
    except Exception as e:
        logger.warning(f"중심점들 조회 실패: {e}")
        return {}

def _get_cloth_areas(self, masks: Dict[str, np.ndarray]) -> Dict[str, int]:
    """의류 면적들 조회"""
    try:
        areas = {}
        
        for mask_type, mask in masks.items():
            if mask is not None:
                area = np.sum(mask)
                areas[mask_type] = int(area)
        
        return areas
        
    except Exception as e:
        logger.warning(f"면적들 조회 실패: {e}")
        return {}

def _get_cloth_contours_dict(self, masks: Dict[str, np.ndarray]) -> Dict[str, List[np.ndarray]]:
    """의류 컨투어들 조회"""
    try:
        contours_dict = {}
        
        for mask_type, mask in masks.items():
            if mask is not None and np.any(mask):
                contours = _extract_cloth_contours(self, mask)
                contours_dict[mask_type] = contours
        
        return contours_dict
        
    except Exception as e:
        logger.warning(f"컨투어들 조회 실패: {e}")
        return {}

def _detect_cloth_categories(self, masks: Dict[str, np.ndarray]) -> List[str]:
    """의류 카테고리 감지"""
    try:
        categories = []
        
        for mask_type, mask in masks.items():
            if mask is None or not np.any(mask):
                continue
            
            # 마스크 타입에서 카테고리 추출
            if 'shirt' in mask_type.lower():
                categories.append('shirt')
            elif 'pants' in mask_type.lower():
                categories.append('pants')
            elif 'dress' in mask_type.lower():
                categories.append('dress')
            elif 'jacket' in mask_type.lower():
                categories.append('jacket')
            elif 'skirt' in mask_type.lower():
                categories.append('skirt')
            elif 'coat' in mask_type.lower():
                categories.append('coat')
            elif 'sweater' in mask_type.lower():
                categories.append('sweater')
            elif 'hoodie' in mask_type.lower():
                categories.append('hoodie')
            else:
                categories.append('unknown')
        
        return list(set(categories))  # 중복 제거
        
    except Exception as e:
        logger.warning(f"의류 카테고리 감지 실패: {e}")
        return []
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Feature Extraction
=====================================================================

특성 추출 관련 기능들을 분리한 모듈

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

import logging
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

def _extract_cloth_features(self, masks: Dict[str, np.ndarray], image: np.ndarray) -> Dict[str, Any]:
    """의류 특성 추출"""
    try:
        features = {}
        
        for mask_type, mask in masks.items():
            if mask is None or not np.any(mask):
                continue
            
            mask_features = {
                'area': np.sum(mask),
                'centroid': _calculate_centroid(mask),
                'bbox': _calculate_bounding_box(mask),
                'contours': _extract_cloth_contours(mask),
                'aspect_ratio': _calculate_aspect_ratio(mask),
                'compactness': _calculate_compactness(mask)
            }
            
            features[mask_type] = mask_features
        
        return features
        
    except Exception as e:
        logger.warning(f"의류 특성 추출 실패: {e}")
        return {}

def _calculate_centroid(self, mask: np.ndarray) -> Tuple[float, float]:
    """중심점 계산"""
    try:
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            return (0.0, 0.0)
        
        centroid_y = np.mean(y_coords)
        centroid_x = np.mean(x_coords)
        
        return (centroid_x, centroid_y)
        
    except Exception as e:
        logger.warning(f"중심점 계산 실패: {e}")
        return (0.0, 0.0)

def _calculate_bounding_box(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
    """바운딩 박스 계산"""
    try:
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            return (0, 0, 0, 0)
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        return (x_min, y_min, x_max, y_max)
        
    except Exception as e:
        logger.warning(f"바운딩 박스 계산 실패: {e}")
        return (0, 0, 0, 0)

def _extract_cloth_contours(self, mask: np.ndarray) -> List[np.ndarray]:
    """의류 컨투어 추출"""
    try:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
        
    except Exception as e:
        logger.warning(f"컨투어 추출 실패: {e}")
        return []

def _calculate_aspect_ratio(self, mask: np.ndarray) -> float:
    """종횡비 계산"""
    try:
        bbox = _calculate_bounding_box(self, mask)
        if bbox[2] == bbox[0] or bbox[3] == bbox[1]:
            return 1.0
        
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        return width / height
        
    except Exception as e:
        logger.warning(f"종횡비 계산 실패: {e}")
        return 1.0

def _calculate_compactness(self, mask: np.ndarray) -> float:
    """조밀도 계산"""
    try:
        area = np.sum(mask)
        if area == 0:
            return 0.0
        
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        
        # 가장 큰 컨투어의 둘레 계산
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter == 0:
            return 0.0
        
        # 조밀도 = 4π * 면적 / 둘레^2
        compactness = 4 * np.pi * area / (perimeter ** 2)
        
        return min(compactness, 1.0)
        
    except Exception as e:
        logger.warning(f"조밀도 계산 실패: {e}")
        return 0.0

def _get_cloth_bounding_boxes(self, masks: Dict[str, np.ndarray]) -> Dict[str, Dict[str, int]]:
    """의류 바운딩 박스들 조회"""
    try:
        bounding_boxes = {}
        
        for mask_type, mask in masks.items():
            if mask is not None and np.any(mask):
                bbox = _calculate_bounding_box(self, mask)
                bounding_boxes[mask_type] = {
                    'x_min': bbox[0],
                    'y_min': bbox[1],
                    'x_max': bbox[2],
                    'y_max': bbox[3],
                    'width': bbox[2] - bbox[0],
                    'height': bbox[3] - bbox[1]
                }
        
        return bounding_boxes
        
    except Exception as e:
        logger.warning(f"바운딩 박스들 조회 실패: {e}")
        return {}

def _get_cloth_centroids(self, masks: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, float]]:
    """의류 중심점들 조회"""
    try:
        centroids = {}
        
        for mask_type, mask in masks.items():
            if mask is not None and np.any(mask):
                centroid = _calculate_centroid(self, mask)
                centroids[mask_type] = centroid
        
        return centroids
        
    except Exception as e:
        logger.warning(f"중심점들 조회 실패: {e}")
        return {}

def _get_cloth_areas(self, masks: Dict[str, np.ndarray]) -> Dict[str, int]:
    """의류 면적들 조회"""
    try:
        areas = {}
        
        for mask_type, mask in masks.items():
            if mask is not None:
                area = np.sum(mask)
                areas[mask_type] = int(area)
        
        return areas
        
    except Exception as e:
        logger.warning(f"면적들 조회 실패: {e}")
        return {}

def _get_cloth_contours_dict(self, masks: Dict[str, np.ndarray]) -> Dict[str, List[np.ndarray]]:
    """의류 컨투어들 조회"""
    try:
        contours_dict = {}
        
        for mask_type, mask in masks.items():
            if mask is not None and np.any(mask):
                contours = _extract_cloth_contours(self, mask)
                contours_dict[mask_type] = contours
        
        return contours_dict
        
    except Exception as e:
        logger.warning(f"컨투어들 조회 실패: {e}")
        return {}

def _detect_cloth_categories(self, masks: Dict[str, np.ndarray]) -> List[str]:
    """의류 카테고리 감지"""
    try:
        categories = []
        
        for mask_type, mask in masks.items():
            if mask is None or not np.any(mask):
                continue
            
            # 마스크 타입에서 카테고리 추출
            if 'shirt' in mask_type.lower():
                categories.append('shirt')
            elif 'pants' in mask_type.lower():
                categories.append('pants')
            elif 'dress' in mask_type.lower():
                categories.append('dress')
            elif 'jacket' in mask_type.lower():
                categories.append('jacket')
            elif 'skirt' in mask_type.lower():
                categories.append('skirt')
            elif 'coat' in mask_type.lower():
                categories.append('coat')
            elif 'sweater' in mask_type.lower():
                categories.append('sweater')
            elif 'hoodie' in mask_type.lower():
                categories.append('hoodie')
            else:
                categories.append('unknown')
        
        return list(set(categories))  # 중복 제거
        
    except Exception as e:
        logger.warning(f"의류 카테고리 감지 실패: {e}")
        return []
