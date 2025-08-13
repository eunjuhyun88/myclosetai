#!/usr/bin/env python3
"""
🔥 MyCloset AI - Pose Estimation Keypoint Processor
=================================================

✅ 키포인트 처리 기능 분리
✅ 기존 step.py 기능 보존
✅ 모듈화된 구조 적용
"""

import logging
from app.ai_pipeline.utils.common_imports import (
    np, math, Dict, Any, Optional, Tuple, List, Union, torch
)

logger = logging.getLogger(__name__)

class KeypointProcessor:
    """키포인트 처리기"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.KeypointProcessor")
    
    def extract_keypoints_from_heatmaps(self, heatmaps: torch.Tensor) -> List[List[float]]:
        """히트맵에서 키포인트 추출"""
        try:
            if not isinstance(heatmaps, torch.Tensor):
                heatmaps = torch.tensor(heatmaps)
            
            keypoints = []
            num_keypoints = heatmaps.shape[0]
            
            for i in range(num_keypoints):
                heatmap = heatmaps[i].cpu().numpy()
                
                # 최대값 위치 찾기
                y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                confidence = heatmap[y, x]
                
                # 서브픽셀 정확도 계산
                if confidence > 0.1:
                    refined_x, refined_y = self._calculate_subpixel_accuracy(heatmap, x, y)
                    keypoints.append([refined_x, refined_y, confidence])
                else:
                    keypoints.append([0.0, 0.0, 0.0])
            
            return keypoints
            
        except Exception as e:
            self.logger.error(f"❌ 히트맵에서 키포인트 추출 실패: {e}")
            return []
    
    def extract_keypoints_with_subpixel_accuracy(self, heatmaps: torch.Tensor, scale_factor: float = 1.0) -> List[List[float]]:
        """서브픽셀 정확도로 키포인트 추출"""
        try:
            if not isinstance(heatmaps, torch.Tensor):
                heatmaps = torch.tensor(heatmaps)
            
            keypoints = []
            num_keypoints = heatmaps.shape[0]
            
            for i in range(num_keypoints):
                heatmap = heatmaps[i].cpu().numpy()
                
                # 최대값 위치 찾기
                y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                confidence = heatmap[y, x]
                
                if confidence > 0.1:
                    # 서브픽셀 정확도 계산
                    refined_x, refined_y = self._calculate_subpixel_accuracy(heatmap, x, y)
                    
                    # 스케일 팩터 적용
                    scaled_x = refined_x * scale_factor
                    scaled_y = refined_y * scale_factor
                    
                    keypoints.append([scaled_x, scaled_y, confidence])
                else:
                    keypoints.append([0.0, 0.0, 0.0])
            
            return keypoints
            
        except Exception as e:
            self.logger.error(f"❌ 서브픽셀 정확도 키포인트 추출 실패: {e}")
            return []
    
    def convert_tensor_to_keypoints(self, output_tensor: torch.Tensor) -> List[List[float]]:
        """텐서를 키포인트로 변환"""
        try:
            if not isinstance(output_tensor, torch.Tensor):
                output_tensor = torch.tensor(output_tensor)
            
            # 텐서 형태 확인 및 변환
            if len(output_tensor.shape) == 4:  # (batch, channels, height, width)
                output_tensor = output_tensor.squeeze(0)  # 배치 차원 제거
            
            if len(output_tensor.shape) == 3:  # (channels, height, width)
                return self.extract_keypoints_from_heatmaps(output_tensor)
            else:
                self.logger.error(f"❌ 지원하지 않는 텐서 형태: {output_tensor.shape}")
                return []
                
        except Exception as e:
            self.logger.error(f"❌ 텐서를 키포인트로 변환 실패: {e}")
            return []
    
    def validate_keypoints(self, keypoints: List[List[float]]) -> bool:
        """키포인트 검증"""
        try:
            if not keypoints or len(keypoints) == 0:
                return False
            
            # 기본 검증: 17개 키포인트 확인
            if len(keypoints) != 17:
                return False
            
            # 각 키포인트가 3개 값(x, y, confidence)을 가지는지 확인
            for kp in keypoints:
                if len(kp) != 3:
                    return False
                if not all(isinstance(val, (int, float)) for val in kp):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 키포인트 검증 실패: {e}")
            return False
    
    def filter_keypoints_by_confidence(self, keypoints: List[List[float]], confidence_threshold: float = 0.5) -> List[List[float]]:
        """신뢰도 기준으로 키포인트 필터링"""
        try:
            filtered_keypoints = []
            
            for kp in keypoints:
                if len(kp) >= 3 and kp[2] >= confidence_threshold:
                    filtered_keypoints.append(kp)
                else:
                    # 신뢰도가 낮은 키포인트는 0으로 설정
                    filtered_keypoints.append([0.0, 0.0, 0.0])
            
            return filtered_keypoints
            
        except Exception as e:
            self.logger.error(f"❌ 키포인트 필터링 실패: {e}")
            return keypoints
    
    def normalize_keypoints(self, keypoints: List[List[float]], image_size: Tuple[int, int]) -> List[List[float]]:
        """키포인트 정규화"""
        try:
            if not keypoints:
                return []
            
            width, height = image_size
            normalized_keypoints = []
            
            for kp in keypoints:
                if len(kp) >= 2:
                    normalized_x = kp[0] / width
                    normalized_y = kp[1] / height
                    confidence = kp[2] if len(kp) >= 3 else 0.0
                    normalized_keypoints.append([normalized_x, normalized_y, confidence])
                else:
                    normalized_keypoints.append([0.0, 0.0, 0.0])
            
            return normalized_keypoints
            
        except Exception as e:
            self.logger.error(f"❌ 키포인트 정규화 실패: {e}")
            return keypoints
    
    def denormalize_keypoints(self, keypoints: List[List[float]], image_size: Tuple[int, int]) -> List[List[float]]:
        """키포인트 역정규화"""
        try:
            if not keypoints:
                return []
            
            width, height = image_size
            denormalized_keypoints = []
            
            for kp in keypoints:
                if len(kp) >= 2:
                    denormalized_x = kp[0] * width
                    denormalized_y = kp[1] * height
                    confidence = kp[2] if len(kp) >= 3 else 0.0
                    denormalized_keypoints.append([denormalized_x, denormalized_y, confidence])
                else:
                    denormalized_keypoints.append([0.0, 0.0, 0.0])
            
            return denormalized_keypoints
            
        except Exception as e:
            self.logger.error(f"❌ 키포인트 역정규화 실패: {e}")
            return keypoints
    
    def _calculate_subpixel_accuracy(self, heatmap: np.ndarray, x: int, y: int) -> Tuple[float, float]:
        """서브픽셀 정확도 계산"""
        try:
            height, width = heatmap.shape
            
            # 주변 픽셀들의 가중 평균 계산
            refined_x = x
            refined_y = y
            
            # 3x3 윈도우에서 가중 평균 계산
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        weight = heatmap[ny, nx]
                        refined_x += nx * weight
                        refined_y += ny * weight
            
            # 정규화
            total_weight = np.sum(heatmap[max(0, y-1):min(height, y+2), max(0, x-1):min(width, x+2)])
            if total_weight > 0:
                refined_x /= total_weight
                refined_y /= total_weight
            
            return float(refined_x), float(refined_y)
            
        except Exception as e:
            self.logger.error(f"❌ 서브픽셀 정확도 계산 실패: {e}")
            return float(x), float(y)
