"""
포즈 변환 유틸리티
"""
from typing import List, Dict, Any
from ..config.types import COCO_17_KEYPOINTS, OPENPOSE_18_KEYPOINTS


def convert_coco17_to_openpose18(coco_keypoints: List[List[float]]) -> List[List[float]]:
    """COCO 17 키포인트를 OpenPose 18 키포인트로 변환"""
    try:
        if len(coco_keypoints) < 17:
            # 부족한 키포인트는 0으로 채움
            coco_keypoints.extend([[0.0, 0.0, 0.0]] * (17 - len(coco_keypoints)))
        
        # OpenPose 18 키포인트 초기화
        openpose_keypoints = [[0.0, 0.0, 0.0] for _ in range(18)]
        
        # COCO → OpenPose 매핑
        coco_to_openpose = {
            0: 0,   # nose
            1: 15,  # left_eye
            2: 14,  # right_eye
            3: 17,  # left_ear
            4: 16,  # right_ear
            5: 5,   # left_shoulder
            6: 2,   # right_shoulder
            7: 6,   # left_elbow
            8: 3,   # right_elbow
            9: 7,   # left_wrist
            10: 4,  # right_wrist
            11: 11, # left_hip
            12: 8,  # right_hip
            13: 12, # left_knee
            14: 9,  # right_knee
            15: 13, # left_ankle
            16: 10  # right_ankle
        }
        
        # 키포인트 변환
        for coco_idx, openpose_idx in coco_to_openpose.items():
            if coco_idx < len(coco_keypoints):
                openpose_keypoints[openpose_idx] = coco_keypoints[coco_idx]
        
        # neck 키포인트 계산 (어깨 중심점)
        if (len(coco_keypoints) > 5 and len(coco_keypoints) > 6 and
            coco_keypoints[5][2] > 0 and coco_keypoints[6][2] > 0):
            left_shoulder = coco_keypoints[5]
            right_shoulder = coco_keypoints[6]
            neck_x = (left_shoulder[0] + right_shoulder[0]) / 2
            neck_y = (left_shoulder[1] + right_shoulder[1]) / 2
            neck_conf = (left_shoulder[2] + right_shoulder[2]) / 2
            openpose_keypoints[1] = [neck_x, neck_y, neck_conf]
        
        return openpose_keypoints
        
    except Exception as e:
        # 변환 실패 시 빈 키포인트 반환
        return [[0.0, 0.0, 0.0] for _ in range(18)]


def convert_openpose18_to_coco17(openpose_keypoints: List[List[float]]) -> List[List[float]]:
    """OpenPose 18 키포인트를 COCO 17 키포인트로 변환"""
    try:
        if len(openpose_keypoints) < 18:
            # 부족한 키포인트는 0으로 채움
            openpose_keypoints.extend([[0.0, 0.0, 0.0]] * (18 - len(openpose_keypoints)))
        
        # COCO 17 키포인트 초기화
        coco_keypoints = [[0.0, 0.0, 0.0] for _ in range(17)]
        
        # OpenPose → COCO 매핑
        openpose_to_coco = {
            0: 0,   # nose
            1: None,  # neck (COCO에는 없음)
            2: 6,   # right_shoulder
            3: 8,   # right_elbow
            4: 10,  # right_wrist
            5: 5,   # left_shoulder
            6: 7,   # left_elbow
            7: 9,   # left_wrist
            8: 12,  # right_hip
            9: 14,  # right_knee
            10: 16, # right_ankle
            11: 11, # left_hip
            12: 13, # left_knee
            13: 15, # left_ankle
            14: 2,  # right_eye
            15: 1,  # left_eye
            16: 4,  # right_ear
            17: 3   # left_ear
        }
        
        # 키포인트 변환
        for openpose_idx, coco_idx in openpose_to_coco.items():
            if coco_idx is not None and openpose_idx < len(openpose_keypoints):
                coco_keypoints[coco_idx] = openpose_keypoints[openpose_idx]
        
        return coco_keypoints
        
    except Exception as e:
        # 변환 실패 시 빈 키포인트 반환
        return [[0.0, 0.0, 0.0] for _ in range(17)]


def convert_mediapipe_to_coco17(mediapipe_keypoints: List[List[float]]) -> List[List[float]]:
    """MediaPipe 키포인트를 COCO 17 키포인트로 변환"""
    try:
        if len(mediapipe_keypoints) < 33:
            # MediaPipe는 33개 키포인트
            return [[0.0, 0.0, 0.0] for _ in range(17)]
        
        # COCO 17 키포인트 초기화
        coco_keypoints = [[0.0, 0.0, 0.0] for _ in range(17)]
        
        # MediaPipe → COCO 매핑
        mediapipe_to_coco = {
            0: 0,   # nose
            2: 1,   # left_eye
            5: 2,   # right_eye
            7: 3,   # left_ear
            8: 4,   # right_ear
            11: 5,  # left_shoulder
            12: 6,  # right_shoulder
            13: 7,  # left_elbow
            14: 8,  # right_elbow
            15: 9,  # left_wrist
            16: 10, # right_wrist
            23: 11, # left_hip
            24: 12, # right_hip
            25: 13, # left_knee
            26: 14, # right_knee
            27: 15, # left_ankle
            28: 16  # right_ankle
        }
        
        # 키포인트 변환
        for mediapipe_idx, coco_idx in mediapipe_to_coco.items():
            if mediapipe_idx < len(mediapipe_keypoints):
                coco_keypoints[coco_idx] = mediapipe_keypoints[mediapipe_idx]
        
        return coco_keypoints
        
    except Exception as e:
        # 변환 실패 시 빈 키포인트 반환
        return [[0.0, 0.0, 0.0] for _ in range(17)]


def normalize_keypoints(keypoints: List[List[float]], image_size: tuple) -> List[List[float]]:
    """키포인트를 이미지 크기에 맞게 정규화"""
    try:
        if not keypoints:
            return []
        
        width, height = image_size
        normalized_keypoints = []
        
        for keypoint in keypoints:
            if len(keypoint) >= 2:
                x = keypoint[0] / width
                y = keypoint[1] / height
                conf = keypoint[2] if len(keypoint) > 2 else 1.0
                normalized_keypoints.append([x, y, conf])
            else:
                normalized_keypoints.append([0.0, 0.0, 0.0])
        
        return normalized_keypoints
        
    except Exception as e:
        return keypoints


def denormalize_keypoints(keypoints: List[List[float]], image_size: tuple) -> List[List[float]]:
    """정규화된 키포인트를 원본 크기로 변환"""
    try:
        if not keypoints:
            return []
        
        width, height = image_size
        denormalized_keypoints = []
        
        for keypoint in keypoints:
            if len(keypoint) >= 2:
                x = keypoint[0] * width
                y = keypoint[1] * height
                conf = keypoint[2] if len(keypoint) > 2 else 1.0
                denormalized_keypoints.append([x, y, conf])
            else:
                denormalized_keypoints.append([0.0, 0.0, 0.0])
        
        return denormalized_keypoints
        
    except Exception as e:
        return keypoints


def validate_keypoints(keypoints: List[List[float]]) -> bool:
    """키포인트 유효성 검증"""
    try:
        if not keypoints:
            return False
        
        # 기본 구조 검증
        for keypoint in keypoints:
            if not isinstance(keypoint, list):
                return False
            
            if len(keypoint) < 2:
                return False
            
            # 좌표값 검증
            for i in range(2):
                if not isinstance(keypoint[i], (int, float)):
                    return False
                
                # 정규화된 좌표인 경우 (0-1 범위)
                if 0 <= keypoint[i] <= 1:
                    continue
                
                # 픽셀 좌표인 경우 (음수 허용, 너무 큰 값은 제한)
                if keypoint[i] < -1000 or keypoint[i] > 10000:
                    return False
        
        return True
        
    except Exception as e:
        return False


def filter_keypoints_by_confidence(keypoints: List[List[float]], threshold: float = 0.5) -> List[List[float]]:
    """신뢰도 기준으로 키포인트 필터링"""
    try:
        filtered_keypoints = []
        
        for keypoint in keypoints:
            if len(keypoint) >= 3 and keypoint[2] >= threshold:
                filtered_keypoints.append(keypoint)
            elif len(keypoint) >= 2:
                # 신뢰도가 없는 경우 기본값 사용
                filtered_keypoints.append([keypoint[0], keypoint[1], 1.0])
        
        return filtered_keypoints
        
    except Exception as e:
        return keypoints


def interpolate_missing_keypoints(keypoints: List[List[float]]) -> List[List[float]]:
    """누락된 키포인트 보간"""
    try:
        if not keypoints:
            return []
        
        interpolated_keypoints = []
        
        for i, keypoint in enumerate(keypoints):
            if len(keypoint) >= 2 and keypoint[0] != 0 and keypoint[1] != 0:
                # 유효한 키포인트
                interpolated_keypoints.append(keypoint)
            else:
                # 누락된 키포인트 - 주변 키포인트로 보간
                interpolated_keypoint = interpolate_from_neighbors(keypoints, i)
                interpolated_keypoints.append(interpolated_keypoint)
        
        return interpolated_keypoints
        
    except Exception as e:
        return keypoints


def interpolate_from_neighbors(keypoints: List[List[float]], index: int) -> List[float]:
    """주변 키포인트로 보간"""
    try:
        # 주변 유효한 키포인트 찾기
        valid_neighbors = []
        
        for i in range(max(0, index - 3), min(len(keypoints), index + 4)):
            if i != index and i < len(keypoints):
                keypoint = keypoints[i]
                if len(keypoint) >= 2 and keypoint[0] != 0 and keypoint[1] != 0:
                    valid_neighbors.append(keypoint)
        
        if valid_neighbors:
            # 평균값 계산
            avg_x = sum(kp[0] for kp in valid_neighbors) / len(valid_neighbors)
            avg_y = sum(kp[1] for kp in valid_neighbors) / len(valid_neighbors)
            avg_conf = sum(kp[2] for kp in valid_neighbors if len(kp) > 2) / len(valid_neighbors)
            return [avg_x, avg_y, avg_conf]
        else:
            # 보간할 수 없는 경우 기본값
            return [0.0, 0.0, 0.0]
            
    except Exception as e:
        return [0.0, 0.0, 0.0]
