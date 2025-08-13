#!/usr/bin/env python3
"""
🔥 MyCloset AI - Pose Estimation Visualizer
==========================================

✅ 시각화 기능 분리
✅ 기존 step.py 기능 보존
✅ 모듈화된 구조 적용
"""

import logging
from app.ai_pipeline.utils.common_imports import (
    np, Image, cv2, PIL_AVAILABLE, CV2_AVAILABLE,
    Dict, Any, Optional, Tuple, List, Union
)

logger = logging.getLogger(__name__)

class PoseVisualizer:
    """포즈 추정 시각화기"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PoseVisualizer")
        
        # 키포인트 색상 정의
        self.keypoint_colors = [
            (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
            (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
            (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
            (255, 0, 255), (255, 0, 170)
        ]
        
        # 스켈레톤 연결 정의
        self.skeleton_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8),
            (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14), (0, 15),
            (15, 17), (0, 16), (16, 18)
        ]
    
    def create_visualization(self, image: np.ndarray, result: Dict[str, Any]) -> Dict[str, Any]:
        """포즈 시각화 생성"""
        try:
            if image is None:
                return {}
            
            keypoints = result.get('keypoints', [])
            if not keypoints:
                return {}
            
            # 이미지 복사
            if len(image.shape) == 3:
                vis_image = image.copy()
            else:
                vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # 키포인트 그리기
            vis_image = self._draw_keypoints(vis_image, keypoints)
            
            # 스켈레톤 그리기
            vis_image = self._draw_skeleton(vis_image, keypoints)
            
            # 바운딩 박스 그리기
            vis_image = self._draw_bounding_box(vis_image, keypoints)
            
            return {
                'visualization_image': vis_image,
                'keypoints_drawn': len(keypoints),
                'skeleton_drawn': True
            }
            
        except Exception as e:
            self.logger.error(f"❌ 포즈 시각화 생성 실패: {e}")
            return {}
    
    def _draw_keypoints(self, image: np.ndarray, keypoints: List[List[float]]) -> np.ndarray:
        """키포인트 그리기"""
        try:
            for i, kp in enumerate(keypoints):
                if len(kp) >= 2 and kp[2] > 0.5:  # confidence > 0.5
                    x, y = int(kp[0]), int(kp[1])
                    color = self.keypoint_colors[i % len(self.keypoint_colors)]
                    
                    # 키포인트 원 그리기
                    cv2.circle(image, (x, y), 4, color, -1)
                    
                    # 키포인트 번호 그리기 (선택적)
                    if i < 10:  # 처음 10개만 번호 표시
                        cv2.putText(image, str(i), (x+5, y-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            return image
            
        except Exception as e:
            self.logger.error(f"❌ 키포인트 그리기 실패: {e}")
            return image
    
    def _draw_skeleton(self, image: np.ndarray, keypoints: List[List[float]]) -> np.ndarray:
        """스켈레톤 그리기"""
        try:
            for connection in self.skeleton_connections:
                if len(connection) == 2:
                    start_idx, end_idx = connection
                    
                    if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                        len(keypoints[start_idx]) >= 3 and len(keypoints[end_idx]) >= 3 and
                        keypoints[start_idx][2] > 0.5 and keypoints[end_idx][2] > 0.5):
                        
                        start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                        end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                        
                        # 선 그리기
                        cv2.line(image, start_point, end_point, (0, 255, 0), 2)
            
            return image
            
        except Exception as e:
            self.logger.error(f"❌ 스켈레톤 그리기 실패: {e}")
            return image
    
    def _draw_bounding_box(self, image: np.ndarray, keypoints: List[List[float]]) -> np.ndarray:
        """바운딩 박스 그리기"""
        try:
            if not keypoints:
                return image
            
            # 유효한 키포인트만 필터링
            valid_keypoints = [kp for kp in keypoints if len(kp) >= 2 and kp[2] > 0.5]
            
            if not valid_keypoints:
                return image
            
            # 바운딩 박스 계산
            x_coords = [kp[0] for kp in valid_keypoints]
            y_coords = [kp[1] for kp in valid_keypoints]
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            # 바운딩 박스 그리기
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            
            return image
            
        except Exception as e:
            self.logger.error(f"❌ 바운딩 박스 그리기 실패: {e}")
            return image
