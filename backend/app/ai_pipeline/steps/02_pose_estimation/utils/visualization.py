"""
포즈 시각화 유틸리티
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Union, List, Tuple, Dict, Any

from ..config.constants import SKELETON_CONNECTIONS, KEYPOINT_COLORS


def draw_pose_on_image(
    image: Union[np.ndarray, Image.Image],
    keypoints: List[List[float]],
    confidence_threshold: float = 0.5,
    keypoint_size: int = 4,
    line_width: int = 3
) -> Image.Image:
    """이미지에 포즈 그리기"""
    try:
        # 이미지 변환
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image.copy()
        
        draw = ImageDraw.Draw(pil_image)
        width, height = pil_image.size
        
        # 키포인트 그리기
        for i, keypoint in enumerate(keypoints):
            if len(keypoint) >= 2 and keypoint[2] >= confidence_threshold:
                x = int(keypoint[0] * width)
                y = int(keypoint[1] * height)
                
                # 키포인트 색상
                color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]
                
                # 키포인트 원 그리기
                draw.ellipse(
                    [x - keypoint_size, y - keypoint_size, 
                     x + keypoint_size, y + keypoint_size],
                    fill=color,
                    outline=(255, 255, 255)
                )
        
        # 스켈레톤 연결선 그리기
        for connection in SKELETON_CONNECTIONS:
            if (connection[0] < len(keypoints) and connection[1] < len(keypoints) and
                len(keypoints[connection[0]]) >= 2 and len(keypoints[connection[1]]) >= 2 and
                keypoints[connection[0]][2] >= confidence_threshold and 
                keypoints[connection[1]][2] >= confidence_threshold):
                
                x1 = int(keypoints[connection[0]][0] * width)
                y1 = int(keypoints[connection[0]][1] * height)
                x2 = int(keypoints[connection[1]][0] * width)
                y2 = int(keypoints[connection[1]][1] * height)
                
                draw.line([x1, y1, x2, y2], fill=(255, 255, 255), width=line_width)
        
        return pil_image
        
    except Exception as e:
        print(f"포즈 시각화 실패: {e}")
        return image if isinstance(image, Image.Image) else Image.fromarray(image)


def create_pose_visualization_data(
    keypoints: List[List[float]],
    image_size: Tuple[int, int]
) -> Dict[str, Any]:
    """포즈 시각화 데이터 생성"""
    try:
        width, height = image_size
        
        # 키포인트 좌표 변환
        keypoint_coords = []
        for keypoint in keypoints:
            if len(keypoint) >= 2:
                x = keypoint[0] * width
                y = keypoint[1] * height
                confidence = keypoint[2] if len(keypoint) > 2 else 1.0
                keypoint_coords.append([x, y, confidence])
            else:
                keypoint_coords.append([0, 0, 0])
        
        # 스켈레톤 연결 정보
        skeleton_connections = []
        for connection in SKELETON_CONNECTIONS:
            if (connection[0] < len(keypoint_coords) and 
                connection[1] < len(keypoint_coords)):
                skeleton_connections.append({
                    'from': connection[0],
                    'to': connection[1],
                    'from_coord': keypoint_coords[connection[0]][:2],
                    'to_coord': keypoint_coords[connection[1]][:2]
                })
        
        return {
            'keypoints': keypoint_coords,
            'skeleton_connections': skeleton_connections,
            'image_size': image_size,
            'num_keypoints': len(keypoint_coords)
        }
        
    except Exception as e:
        print(f"포즈 시각화 데이터 생성 실패: {e}")
        return {}
