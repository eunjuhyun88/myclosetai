"""
🔥 Quality Assessment
====================

품질 평가 및 분석 메서드들

Author: MyCloset AI Team
Date: 2025-08-07
Version: 1.0
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple
import logging


class QualityAssessment:
    """품질 평가 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_quality_metrics(self, parsing_map: np.ndarray, confidence_map: np.ndarray = None) -> Dict[str, float]:
        """품질 메트릭 계산"""
        try:
            metrics = {}
            
            # 기본 품질 지표
            metrics['unique_labels'] = len(np.unique(parsing_map))
            metrics['coverage_ratio'] = np.sum(parsing_map > 0) / parsing_map.size
            
            # 경계 품질
            edge_quality = self._calculate_edge_quality(parsing_map)
            metrics['edge_quality'] = edge_quality
            
            # 일관성 품질
            consistency_quality = self._calculate_consistency_quality(parsing_map)
            metrics['consistency_quality'] = consistency_quality
            
            # 전체 품질 점수
            overall_quality = self._calculate_overall_quality(metrics)
            metrics['overall_quality'] = overall_quality
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ 품질 메트릭 계산 실패: {e}")
            return {'overall_quality': 0.5}
    
    def _calculate_edge_quality(self, parsing_map: np.ndarray) -> float:
        """경계 품질 계산"""
        try:
            # 경계 검출
            edges = cv2.Canny(parsing_map.astype(np.uint8), 50, 150)
            
            # 경계 밀도
            edge_density = np.sum(edges > 0) / edges.size
            
            # 경계 연속성
            kernel = np.ones((3, 3), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            continuity_score = np.sum(dilated_edges > 0) / np.sum(edges > 0) if np.sum(edges > 0) > 0 else 0
            
            # 경계 품질 점수
            edge_quality = (edge_density * 0.4 + continuity_score * 0.6)
            
            return min(1.0, edge_quality)
            
        except Exception as e:
            self.logger.warning(f"⚠️ 경계 품질 계산 실패: {e}")
            return 0.5
    
    def _calculate_consistency_quality(self, parsing_map: np.ndarray) -> float:
        """일관성 품질 계산"""
        try:
            # 지역적 일관성
            kernel = np.ones((5, 5), np.uint8)
            eroded = cv2.erode(parsing_map.astype(np.uint8), kernel, iterations=1)
            dilated = cv2.dilate(parsing_map.astype(np.uint8), kernel, iterations=1)
            
            # 일관성 점수
            consistency_score = np.sum(eroded == parsing_map) / parsing_map.size
            
            return min(1.0, consistency_score)
            
        except Exception as e:
            self.logger.warning(f"⚠️ 일관성 품질 계산 실패: {e}")
            return 0.5
    
    def _calculate_overall_quality(self, metrics: Dict[str, float]) -> float:
        """전체 품질 점수 계산"""
        try:
            # 가중 평균
            weights = {
                'unique_labels': 0.2,
                'coverage_ratio': 0.3,
                'edge_quality': 0.3,
                'consistency_quality': 0.2
            }
            
            overall_score = 0.0
            total_weight = 0.0
            
            for metric_name, weight in weights.items():
                if metric_name in metrics:
                    if metric_name == 'unique_labels':
                        # 라벨 수를 0-1 범위로 정규화
                        normalized_value = min(1.0, metrics[metric_name] / 20.0)
                        overall_score += normalized_value * weight
                    else:
                        overall_score += metrics[metric_name] * weight
                    total_weight += weight
            
            if total_weight > 0:
                return overall_score / total_weight
            else:
                return 0.5
                
        except Exception as e:
            self.logger.warning(f"⚠️ 전체 품질 계산 실패: {e}")
            return 0.5
    
    def analyze_detected_parts(self, parsing_map: np.ndarray) -> Dict[str, Any]:
        """감지된 신체 부위 분석"""
        try:
            analysis = {}
            
            # 고유 라벨 분석
            unique_labels = np.unique(parsing_map)
            analysis['unique_labels'] = unique_labels.tolist()
            analysis['num_parts'] = len(unique_labels)
            
            # 각 부위별 분석
            part_analysis = {}
            for label in unique_labels:
                if label == 0:  # 배경
                    continue
                
                mask = (parsing_map == label)
                area = np.sum(mask)
                area_ratio = area / parsing_map.size
                
                # 경계 상자 계산
                coords = np.where(mask)
                if len(coords[0]) > 0:
                    y_min, y_max = np.min(coords[0]), np.max(coords[0])
                    x_min, x_max = np.min(coords[1]), np.max(coords[1])
                    bbox = {
                        'x_min': int(x_min),
                        'y_min': int(y_min),
                        'x_max': int(x_max),
                        'y_max': int(y_max),
                        'width': int(x_max - x_min),
                        'height': int(y_max - y_min)
                    }
                else:
                    bbox = None
                
                part_analysis[int(label)] = {
                    'area': int(area),
                    'area_ratio': float(area_ratio),
                    'bbox': bbox,
                    'quality': self._evaluate_region_quality(mask)
                }
            
            analysis['part_analysis'] = part_analysis
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ 부위 분석 실패: {e}")
            return {'unique_labels': [], 'num_parts': 0, 'part_analysis': {}}
    
    def _evaluate_region_quality(self, mask: np.ndarray) -> float:
        """영역 품질 평가"""
        try:
            if np.sum(mask) == 0:
                return 0.0
            
            # 경계 품질
            edges = cv2.Canny(mask.astype(np.uint8), 50, 150)
            edge_density = np.sum(edges > 0) / np.sum(mask)
            
            # 형태 품질 (원형도)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            else:
                circularity = 0
            
            # 품질 점수
            quality = (edge_density * 0.6 + circularity * 0.4)
            
            return min(1.0, quality)
            
        except Exception as e:
            self.logger.warning(f"⚠️ 영역 품질 평가 실패: {e}")
            return 0.5
    
    def assess_image_quality(self, image) -> Dict[str, float]:
        """이미지 품질 평가"""
        try:
            if image is None:
                return {'overall_quality': 0.0}
            
            # NumPy 배열로 변환
            if hasattr(image, 'convert'):
                image_np = np.array(image.convert('RGB'))
            elif hasattr(image, 'shape'):
                image_np = image
            else:
                return {'overall_quality': 0.5}
            
            quality_metrics = {}
            
            # 밝기 평가
            if len(image_np.shape) == 3:
                brightness = np.mean(image_np)
                quality_metrics['brightness'] = min(1.0, brightness / 255.0)
                
                # 대비 평가
                contrast = np.std(image_np)
                quality_metrics['contrast'] = min(1.0, contrast / 100.0)
                
                # 색상 품질
                color_quality = self._assess_color_quality(image_np)
                quality_metrics['color_quality'] = color_quality
            else:
                # 그레이스케일
                brightness = np.mean(image_np)
                quality_metrics['brightness'] = min(1.0, brightness / 255.0)
                quality_metrics['contrast'] = min(1.0, np.std(image_np) / 100.0)
                quality_metrics['color_quality'] = 0.5
            
            # 해상도 품질
            h, w = image_np.shape[:2]
            resolution_quality = min(1.0, (h * w) / (1920 * 1080))  # 1080p 기준
            quality_metrics['resolution_quality'] = resolution_quality
            
            # 전체 품질
            overall_quality = np.mean(list(quality_metrics.values()))
            quality_metrics['overall_quality'] = overall_quality
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"❌ 이미지 품질 평가 실패: {e}")
            return {'overall_quality': 0.5}
    
    def _assess_color_quality(self, image_np: np.ndarray) -> float:
        """색상 품질 평가"""
        try:
            # 색상 다양성
            unique_colors = len(np.unique(image_np.reshape(-1, 3), axis=0))
            color_diversity = min(1.0, unique_colors / 10000)
            
            # 색상 균형
            hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
            saturation = np.mean(hsv[:, :, 1])
            color_balance = min(1.0, saturation / 255.0)
            
            # 색상 품질 점수
            color_quality = (color_diversity * 0.6 + color_balance * 0.4)
            
            return color_quality
            
        except Exception as e:
            self.logger.warning(f"⚠️ 색상 품질 평가 실패: {e}")
            return 0.5
    
    def create_visualization(self, parsing_map: np.ndarray, original_image) -> Dict[str, Any]:
        """시각화 생성"""
        try:
            visualization = {}
            
            # 컬러 맵 생성
            colored_parsing = self._create_colored_parsing_map(parsing_map)
            visualization['colored_parsing'] = colored_parsing
            
            # 오버레이 이미지 생성
            if original_image is not None:
                overlay_image = self._create_overlay_image(original_image, colored_parsing)
                visualization['overlay_image'] = overlay_image
            
            # 경계 상자 정보
            bbox_info = self._get_bounding_box(parsing_map)
            visualization['bounding_box'] = bbox_info
            
            return visualization
            
        except Exception as e:
            self.logger.error(f"❌ 시각화 생성 실패: {e}")
            return {}
    
    def _create_colored_parsing_map(self, parsing_map: np.ndarray) -> np.ndarray:
        """컬러 파싱 맵 생성"""
        try:
            # 20개 클래스에 대한 컬러 팔레트
            colors = [
                [0, 0, 0],      # 0: 배경
                [128, 0, 0],    # 1: 모자
                [255, 0, 0],    # 2: 머리카락
                [0, 85, 0],     # 3: 글로브
                [170, 0, 51],   # 4: 선글라스
                [255, 85, 0],   # 5: 상의
                [0, 0, 85],     # 6: 드레스
                [0, 119, 221],  # 7: 코트
                [85, 85, 0],    # 8: 양말
                [0, 85, 85],    # 9: 바지
                [85, 51, 0],    # 10: 점퍼
                [52, 86, 128],  # 11: 스카프
                [0, 128, 0],    # 12: 스커트
                [0, 0, 255],    # 13: 얼굴
                [51, 169, 220], # 14: 왼팔
                [0, 255, 255],  # 15: 오른팔
                [255, 255, 0],  # 16: 왼다리
                [255, 0, 255],  # 17: 오른다리
                [169, 169, 169],# 18: 왼발
                [169, 0, 169]   # 19: 오른발
            ]
            
            colored_map = np.zeros((*parsing_map.shape, 3), dtype=np.uint8)
            
            for label in range(len(colors)):
                mask = (parsing_map == label)
                colored_map[mask] = colors[label]
            
            return colored_map
            
        except Exception as e:
            self.logger.error(f"❌ 컬러 맵 생성 실패: {e}")
            return np.zeros((*parsing_map.shape, 3), dtype=np.uint8)
    
    def _create_overlay_image(self, original_image: np.ndarray, colored_parsing: np.ndarray) -> np.ndarray:
        """오버레이 이미지 생성"""
        try:
            # 원본 이미지 준비
            if hasattr(original_image, 'convert'):
                original_np = np.array(original_image.convert('RGB'))
            else:
                original_np = original_image
            
            # 크기 맞추기
            if original_np.shape[:2] != colored_parsing.shape[:2]:
                colored_parsing = cv2.resize(colored_parsing, (original_np.shape[1], original_np.shape[0]))
            
            # 알파 블렌딩
            alpha = 0.6
            overlay = cv2.addWeighted(original_np, 1-alpha, colored_parsing, alpha, 0)
            
            return overlay
            
        except Exception as e:
            self.logger.error(f"❌ 오버레이 생성 실패: {e}")
            return original_image
    
    def _get_bounding_box(self, mask: np.ndarray) -> Dict[str, int]:
        """경계 상자 계산"""
        try:
            coords = np.where(mask > 0)
            if len(coords[0]) > 0:
                y_min, y_max = np.min(coords[0]), np.max(coords[0])
                x_min, x_max = np.min(coords[1]), np.max(coords[1])
                
                return {
                    'x_min': int(x_min),
                    'y_min': int(y_min),
                    'x_max': int(x_max),
                    'y_max': int(y_max),
                    'width': int(x_max - x_min),
                    'height': int(y_max - y_min)
                }
            else:
                return {'x_min': 0, 'y_min': 0, 'x_max': 0, 'y_max': 0, 'width': 0, 'height': 0}
                
        except Exception as e:
            self.logger.error(f"❌ 경계 상자 계산 실패: {e}")
            return {'x_min': 0, 'y_min': 0, 'x_max': 0, 'y_max': 0, 'width': 0, 'height': 0}
