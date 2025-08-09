"""
포즈 추정 관련 상수 정의
"""

# 키포인트 연결 구조 (스켈레톤)
SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8),
    (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14), (0, 15),
    (15, 17), (0, 16), (16, 18)
]

# 키포인트 색상 매핑
KEYPOINT_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85), (255, 0, 0)
]

# COCO 17 키포인트 정의 (MediaPipe, YOLOv8 표준)
COCO_17_KEYPOINTS_LIST = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# OpenPose 18 키포인트 정의 
OPENPOSE_18_KEYPOINTS_LIST = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "middle_hip", "right_hip", 
    "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear"
]

# 기본 설정값들
DEFAULT_INPUT_SIZE = (512, 512)
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_QUALITY_THRESHOLD = 0.8

# 모델별 가중치 (앙상블용)
MODEL_WEIGHTS = {
    'hrnet': 0.4,
    'openpose': 0.3,
    'yolo_pose': 0.2,
    'mediapipe': 0.1
}

# 의류 피팅에 중요한 관절들
CLOTHING_IMPORTANT_JOINTS = [5, 6, 7, 8, 9, 10, 12, 13]  # 어깨, 팔꿈치, 손목, 엉덩이, 무릎

# 좌우 대칭 관절 쌍
SYMMETRIC_JOINT_PAIRS = [
    (5, 6),   # left_shoulder, right_shoulder
    (7, 8),   # left_elbow, right_elbow
    (9, 10),  # left_wrist, right_wrist
    (12, 13)  # left_hip, right_hip
]
