"""
🔥 Human Parsing 후처리기
======================

Human Parsing 결과에 대한 후처리 기능을 제공합니다.

주요 기능:
- CRF 후처리
- 엣지 정제
- 홀 채우기 및 노이즈 제거
- 품질 향상
- 멀티스케일 처리

Author: MyCloset AI Team
Date: 2025-08-07
Version: 1.0
"""

import numpy as np
import cv2
from typing import Optional
from scipy import ndimage
from skimage import morphology, filters


class AdvancedPostProcessor:
    """고급 후처리 프로세서"""
    
    @staticmethod
    def apply_crf_postprocessing(parsing_map: np.ndarray, image: np.ndarray, num_iterations: int = 10) -> np.ndarray:
        """
        CRF 후처리를 적용합니다.
        
        Args:
            parsing_map: 파싱 맵
            image: 원본 이미지
            num_iterations: 반복 횟수
            
        Returns:
            후처리된 파싱 맵
        """
        try:
            # 간단한 형태학적 후처리로 대체 (DenseCRF 없을 때)
            # 1. 노이즈 제거
            kernel = np.ones((3, 3), np.uint8)
            parsing_map = cv2.morphologyEx(parsing_map, cv2.MORPH_CLOSE, kernel)
            parsing_map = cv2.morphologyEx(parsing_map, cv2.MORPH_OPEN, kernel)
            
            # 2. 경계 스무딩
            parsing_map = cv2.medianBlur(parsing_map, 3)
            
            return parsing_map
            
        except Exception as e:
            print(f"⚠️ CRF 후처리 실패: {e}")
            return parsing_map
    
    @staticmethod
    def apply_multiscale_processing(image: np.ndarray, initial_parsing: np.ndarray) -> np.ndarray:
        """
        멀티스케일 처리를 적용합니다.
        
        Args:
            image: 원본 이미지
            initial_parsing: 초기 파싱 맵
            
        Returns:
            멀티스케일 처리된 파싱 맵
        """
        try:
            # 간단한 멀티스케일 처리
            # 1. 원본 크기
            result = initial_parsing.copy()
            
            # 2. 1/2 크기에서 처리
            h, w = image.shape[:2]
            half_h, half_w = h // 2, w // 2
            
            if half_h > 0 and half_w > 0:
                half_image = cv2.resize(image, (half_w, half_h))
                half_parsing = cv2.resize(initial_parsing, (half_w, half_h), interpolation=cv2.INTER_NEAREST)
                
                # 간단한 엣지 정제
                half_parsing = AdvancedPostProcessor.apply_edge_refinement(half_parsing, half_image)
                
                # 원본 크기로 복원
                half_parsing_resized = cv2.resize(half_parsing, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # 결과 융합
                result = np.where(result == 0, half_parsing_resized, result)
            
            return result
            
        except Exception as e:
            print(f"⚠️ 멀티스케일 처리 실패: {e}")
            return initial_parsing
    
    @staticmethod
    def apply_edge_refinement(parsing_map: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        엣지 정제를 적용합니다.
        
        Args:
            parsing_map: 파싱 맵
            image: 원본 이미지
            
        Returns:
            엣지 정제된 파싱 맵
        """
        try:
            # 1. 엣지 검출
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            edges = cv2.Canny(gray, 50, 150)
            
            # 2. 파싱 맵 경계 검출
            parsing_edges = cv2.Canny(parsing_map.astype(np.uint8), 1, 2)
            
            # 3. 엣지 정보를 활용한 정제
            kernel = np.ones((2, 2), np.uint8)
            refined_edges = cv2.dilate(parsing_edges, kernel, iterations=1)
            
            # 4. 엣지 근처의 파싱 맵 정제
            edge_mask = refined_edges > 0
            if np.any(edge_mask):
                # 엣지 근처에서 가장 빈번한 클래스로 채우기
                for i in range(parsing_map.shape[0]):
                    for j in range(parsing_map.shape[1]):
                        if edge_mask[i, j]:
                            # 주변 영역에서 가장 빈번한 클래스 찾기
                            y1, y2 = max(0, i-2), min(parsing_map.shape[0], i+3)
                            x1, x2 = max(0, j-2), min(parsing_map.shape[1], j+3)
                            neighborhood = parsing_map[y1:y2, x1:x2]
                            if neighborhood.size > 0:
                                unique, counts = np.unique(neighborhood, return_counts=True)
                                most_common = unique[np.argmax(counts)]
                                parsing_map[i, j] = most_common
            
            return parsing_map
            
        except Exception as e:
            print(f"⚠️ 엣지 정제 실패: {e}")
            return parsing_map
    
    @staticmethod
    def apply_hole_filling_and_noise_removal(parsing_map: np.ndarray) -> np.ndarray:
        """
        홀 채우기 및 노이즈 제거를 적용합니다.
        
        Args:
            parsing_map: 파싱 맵
            
        Returns:
            홀 채우기 및 노이즈 제거된 파싱 맵
        """
        try:
            result = parsing_map.copy()
            
            # 1. 작은 노이즈 제거
            for class_id in range(20):  # 20개 클래스
                mask = (result == class_id).astype(np.uint8)
                if np.any(mask):
                    # 연결 요소 분석
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
                    
                    # 작은 연결 요소 제거
                    for i in range(1, num_labels):  # 0은 배경
                        area = stats[i, cv2.CC_STAT_AREA]
                        if area < 50:  # 50 픽셀 미만은 노이즈로 간주
                            result[labels == i] = 0
            
            # 2. 홀 채우기
            for class_id in range(1, 20):  # 배경(0) 제외
                mask = (result == class_id).astype(np.uint8)
                if np.any(mask):
                    # 모폴로지 연산으로 홀 채우기
                    kernel = np.ones((3, 3), np.uint8)
                    filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    result = np.where(filled_mask == 1, class_id, result)
            
            # 3. 최종 스무딩
            kernel = np.ones((2, 2), np.uint8)
            result = cv2.morphologyEx(result.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            
            return result
            
        except Exception as e:
            print(f"⚠️ 홀 채우기 및 노이즈 제거 실패: {e}")
            return parsing_map
    
    @staticmethod
    def apply_quality_enhancement(parsing_map: np.ndarray, image: np.ndarray, confidence_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        품질 향상을 적용합니다.
        
        Args:
            parsing_map: 파싱 맵
            image: 원본 이미지
            confidence_map: 신뢰도 맵 (선택사항)
            
        Returns:
            품질 향상된 파싱 맵
        """
        try:
            result = parsing_map.copy()
            
            # 1. 신뢰도 기반 필터링
            if confidence_map is not None:
                low_confidence_mask = confidence_map < 0.5
                if np.any(low_confidence_mask):
                    # 낮은 신뢰도 영역을 주변 영역으로 채우기
                    for i in range(parsing_map.shape[0]):
                        for j in range(parsing_map.shape[1]):
                            if low_confidence_mask[i, j]:
                                # 주변 영역에서 가장 빈번한 클래스 찾기
                                y1, y2 = max(0, i-3), min(parsing_map.shape[0], i+4)
                                x1, x2 = max(0, j-3), min(parsing_map.shape[1], j+4)
                                neighborhood = parsing_map[y1:y2, x1:x2]
                                if neighborhood.size > 0:
                                    unique, counts = np.unique(neighborhood, return_counts=True)
                                    most_common = unique[np.argmax(counts)]
                                    result[i, j] = most_common
            
            # 2. 경계 정제
            result = AdvancedPostProcessor.apply_edge_refinement(result, image)
            
            # 3. 홀 채우기 및 노이즈 제거
            result = AdvancedPostProcessor.apply_hole_filling_and_noise_removal(result)
            
            return result
            
        except Exception as e:
            print(f"⚠️ 품질 향상 실패: {e}")
            return parsing_map
