#!/usr/bin/env python3
"""
🔥 MyCloset AI - Pose Estimation Geometry Analyzer
=================================================

✅ 포즈 기하학적 분석 기능 분리
✅ 기존 step.py 기능 보존
✅ 모듈화된 구조 적용
"""

import logging
from app.ai_pipeline.utils.common_imports import (
    np, math, Dict, Any, Optional, Tuple, List, Union
)

logger = logging.getLogger(__name__)

class PoseGeometryAnalyzer:
    """포즈 기하학적 분석기"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PoseGeometryAnalyzer")
    
    def calculate_joint_angles(self, keypoints: List[List[float]]) -> Dict[str, float]:
        """관절 각도 계산"""
        try:
            angles = {}
            
            if len(keypoints) < 17:
                return angles
            
            # 어깨 각도 계산
            if self._validate_keypoints_for_angle(keypoints, [5, 7, 9]):  # 왼쪽 어깨-팔꿈치-손목
                angles['left_shoulder'] = self._calculate_angle_3points(
                    keypoints[5], keypoints[7], keypoints[9]
                )
            
            if self._validate_keypoints_for_angle(keypoints, [6, 8, 10]):  # 오른쪽 어깨-팔꿈치-손목
                angles['right_shoulder'] = self._calculate_angle_3points(
                    keypoints[6], keypoints[8], keypoints[10]
                )
            
            # 팔꿈치 각도 계산
            if self._validate_keypoints_for_angle(keypoints, [5, 7, 9]):  # 왼쪽 어깨-팔꿈치-손목
                angles['left_elbow'] = self._calculate_angle_3points(
                    keypoints[5], keypoints[7], keypoints[9]
                )
            
            if self._validate_keypoints_for_angle(keypoints, [6, 8, 10]):  # 오른쪽 어깨-팔꿈치-손목
                angles['right_elbow'] = self._calculate_angle_3points(
                    keypoints[6], keypoints[8], keypoints[10]
                )
            
            # 무릎 각도 계산
            if self._validate_keypoints_for_angle(keypoints, [11, 13, 15]):  # 왼쪽 엉덩이-무릎-발목
                angles['left_knee'] = self._calculate_angle_3points(
                    keypoints[11], keypoints[13], keypoints[15]
                )
            
            if self._validate_keypoints_for_angle(keypoints, [12, 14, 16]):  # 오른쪽 엉덩이-무릎-발목
                angles['right_knee'] = self._calculate_angle_3points(
                    keypoints[12], keypoints[14], keypoints[16]
                )
            
            # 엉덩이 각도 계산
            if self._validate_keypoints_for_angle(keypoints, [5, 11, 13]):  # 왼쪽 어깨-엉덩이-무릎
                angles['left_hip'] = self._calculate_angle_3points(
                    keypoints[5], keypoints[11], keypoints[13]
                )
            
            if self._validate_keypoints_for_angle(keypoints, [6, 12, 14]):  # 오른쪽 어깨-엉덩이-무릎
                angles['right_hip'] = self._calculate_angle_3points(
                    keypoints[6], keypoints[12], keypoints[14]
                )
            
            return angles
            
        except Exception as e:
            self.logger.error(f"❌ 관절 각도 계산 실패: {e}")
            return {}
    
    def calculate_body_proportions(self, keypoints: List[List[float]]) -> Dict[str, float]:
        """신체 비율 계산"""
        try:
            proportions = {}
            
            if len(keypoints) < 17:
                return proportions
            
            # 어깨 너비
            if self._validate_keypoints_for_distance(keypoints, [5, 6]):
                proportions['shoulder_width'] = self._calculate_distance(keypoints[5], keypoints[6])
            
            # 엉덩이 너비
            if self._validate_keypoints_for_distance(keypoints, [11, 12]):
                proportions['hip_width'] = self._calculate_distance(keypoints[11], keypoints[12])
            
            # 왼쪽 팔 길이
            if self._validate_keypoints_for_distance(keypoints, [5, 7]):
                proportions['left_arm_upper'] = self._calculate_distance(keypoints[5], keypoints[7])
            
            if self._validate_keypoints_for_distance(keypoints, [7, 9]):
                proportions['left_arm_lower'] = self._calculate_distance(keypoints[7], keypoints[9])
            
            # 오른쪽 팔 길이
            if self._validate_keypoints_for_distance(keypoints, [6, 8]):
                proportions['right_arm_upper'] = self._calculate_distance(keypoints[6], keypoints[8])
            
            if self._validate_keypoints_for_distance(keypoints, [8, 10]):
                proportions['right_arm_lower'] = self._calculate_distance(keypoints[8], keypoints[10])
            
            # 왼쪽 다리 길이
            if self._validate_keypoints_for_distance(keypoints, [11, 13]):
                proportions['left_leg_upper'] = self._calculate_distance(keypoints[11], keypoints[13])
            
            if self._validate_keypoints_for_distance(keypoints, [13, 15]):
                proportions['left_leg_lower'] = self._calculate_distance(keypoints[13], keypoints[15])
            
            # 오른쪽 다리 길이
            if self._validate_keypoints_for_distance(keypoints, [12, 14]):
                proportions['right_leg_upper'] = self._calculate_distance(keypoints[12], keypoints[14])
            
            if self._validate_keypoints_for_distance(keypoints, [14, 16]):
                proportions['right_leg_lower'] = self._calculate_distance(keypoints[14], keypoints[16])
            
            # 몸통 길이
            if self._validate_keypoints_for_distance(keypoints, [5, 11]):
                proportions['left_torso'] = self._calculate_distance(keypoints[5], keypoints[11])
            
            if self._validate_keypoints_for_distance(keypoints, [6, 12]):
                proportions['right_torso'] = self._calculate_distance(keypoints[6], keypoints[12])
            
            return proportions
            
        except Exception as e:
            self.logger.error(f"❌ 신체 비율 계산 실패: {e}")
            return {}
    
    def calculate_pose_direction(self, keypoints: List[List[float]]) -> str:
        """포즈 방향 계산"""
        try:
            if len(keypoints) < 17:
                return "unknown"
            
            # 어깨 중심점 계산
            if self._validate_keypoints_for_center(keypoints, [5, 6]):
                shoulder_center = self._calculate_center_point(keypoints[5], keypoints[6])
            else:
                return "unknown"
            
            # 엉덩이 중심점 계산
            if self._validate_keypoints_for_center(keypoints, [11, 12]):
                hip_center = self._calculate_center_point(keypoints[11], keypoints[12])
            else:
                return "unknown"
            
            # 어깨와 엉덩이의 상대적 위치로 방향 결정
            if shoulder_center[0] > hip_center[0] + 10:
                return "left"
            elif shoulder_center[0] < hip_center[0] - 10:
                return "right"
            else:
                return "front"
                
        except Exception as e:
            self.logger.error(f"❌ 포즈 방향 계산 실패: {e}")
            return "unknown"
    
    def calculate_pose_stability(self, keypoints: List[List[float]]) -> float:
        """포즈 안정성 계산"""
        try:
            if len(keypoints) < 17:
                return 0.0
            
            # 유효한 키포인트만 필터링
            valid_keypoints = [kp for kp in keypoints if len(kp) >= 2 and kp[2] > 0.5]
            
            if not valid_keypoints:
                return 0.0
            
            # 중심점 계산
            center_x = np.mean([kp[0] for kp in valid_keypoints])
            center_y = np.mean([kp[1] for kp in valid_keypoints])
            
            # 각 키포인트의 중심점으로부터의 거리 계산
            distances = []
            for kp in valid_keypoints:
                distance = np.sqrt((kp[0] - center_x)**2 + (kp[1] - center_y)**2)
                distances.append(distance)
            
            # 안정성 점수 (거리의 표준편차가 작을수록 안정적)
            if distances:
                stability_score = 1.0 / (1.0 + np.std(distances))
                return min(stability_score, 1.0)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"❌ 포즈 안정성 계산 실패: {e}")
            return 0.0
    
    def calculate_body_orientation(self, keypoints: List[List[float]]) -> Dict[str, float]:
        """신체 방향 계산"""
        try:
            orientation = {}
            
            if len(keypoints) < 17:
                return orientation
            
            # 어깨 각도로 신체 회전 계산
            if self._validate_keypoints_for_angle(keypoints, [5, 6]):
                shoulder_angle = self._calculate_angle_between_points(keypoints[5], keypoints[6])
                orientation['shoulder_rotation'] = shoulder_angle
            
            # 엉덩이 각도로 신체 회전 계산
            if self._validate_keypoints_for_angle(keypoints, [11, 12]):
                hip_angle = self._calculate_angle_between_points(keypoints[11], keypoints[12])
                orientation['hip_rotation'] = hip_angle
            
            # 전체 신체 회전 (어깨와 엉덩이의 평균)
            if 'shoulder_rotation' in orientation and 'hip_rotation' in orientation:
                orientation['body_rotation'] = (orientation['shoulder_rotation'] + orientation['hip_rotation']) / 2
            
            return orientation
            
        except Exception as e:
            self.logger.error(f"❌ 신체 방향 계산 실패: {e}")
            return {}
    
    def build_skeleton_structure(self, keypoints: List[List[float]]) -> Dict[str, Any]:
        """스켈레톤 구조 생성"""
        try:
            skeleton = {
                'joints': {},
                'bones': {},
                'connections': []
            }
            
            if len(keypoints) < 17:
                return skeleton
            
            # 관절 정의
            joint_names = [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ]
            
            # 관절 위치 저장
            for i, name in enumerate(joint_names):
                if i < len(keypoints):
                    skeleton['joints'][name] = {
                        'position': keypoints[i][:2],
                        'confidence': keypoints[i][2] if len(keypoints[i]) > 2 else 0.0
                    }
            
            # 뼈대 연결 정의
            bone_connections = [
                ('left_shoulder', 'right_shoulder'),
                ('left_shoulder', 'left_elbow'),
                ('left_elbow', 'left_wrist'),
                ('right_shoulder', 'right_elbow'),
                ('right_elbow', 'right_wrist'),
                ('left_shoulder', 'left_hip'),
                ('right_shoulder', 'right_hip'),
                ('left_hip', 'right_hip'),
                ('left_hip', 'left_knee'),
                ('left_knee', 'left_ankle'),
                ('right_hip', 'right_knee'),
                ('right_knee', 'right_ankle')
            ]
            
            # 뼈대 연결 생성
            for start_joint, end_joint in bone_connections:
                if start_joint in skeleton['joints'] and end_joint in skeleton['joints']:
                    start_pos = skeleton['joints'][start_joint]['position']
                    end_pos = skeleton['joints'][end_joint]['position']
                    
                    bone_length = self._calculate_distance(start_pos, end_pos)
                    skeleton['bones'][f"{start_joint}_to_{end_joint}"] = {
                        'start': start_pos,
                        'end': end_pos,
                        'length': bone_length
                    }
                    
                    skeleton['connections'].append({
                        'start': start_joint,
                        'end': end_joint,
                        'length': bone_length
                    })
            
            return skeleton
            
        except Exception as e:
            self.logger.error(f"❌ 스켈레톤 구조 생성 실패: {e}")
            return {'joints': {}, 'bones': {}, 'connections': []}
    
    def _validate_keypoints_for_angle(self, keypoints: List[List[float]], indices: List[int]) -> bool:
        """각도 계산을 위한 키포인트 검증"""
        try:
            for idx in indices:
                if idx >= len(keypoints) or len(keypoints[idx]) < 2:
                    return False
                if keypoints[idx][2] < 0.5:  # 신뢰도가 낮은 경우
                    return False
            return True
        except Exception:
            return False
    
    def _validate_keypoints_for_distance(self, keypoints: List[List[float]], indices: List[int]) -> bool:
        """거리 계산을 위한 키포인트 검증"""
        try:
            for idx in indices:
                if idx >= len(keypoints) or len(keypoints[idx]) < 2:
                    return False
                if keypoints[idx][2] < 0.5:  # 신뢰도가 낮은 경우
                    return False
            return True
        except Exception:
            return False
    
    def _validate_keypoints_for_center(self, keypoints: List[List[float]], indices: List[int]) -> bool:
        """중심점 계산을 위한 키포인트 검증"""
        try:
            for idx in indices:
                if idx >= len(keypoints) or len(keypoints[idx]) < 2:
                    return False
                if keypoints[idx][2] < 0.5:  # 신뢰도가 낮은 경우
                    return False
            return True
        except Exception:
            return False
    
    def _calculate_angle_3points(self, p1: List[float], p2: List[float], p3: List[float]) -> float:
        """3점으로 각도 계산"""
        try:
            if not all(p1) or not all(p2) or not all(p3):
                return 0.0
            
            # 벡터 계산
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            # 각도 계산
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            return np.degrees(angle)
            
        except Exception as e:
            self.logger.error(f"❌ 3점 각도 계산 실패: {e}")
            return 0.0
    
    def _calculate_distance(self, p1: List[float], p2: List[float]) -> float:
        """두 점 간의 거리 계산"""
        try:
            if len(p1) < 2 or len(p2) < 2:
                return 0.0
            
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            
        except Exception as e:
            self.logger.error(f"❌ 거리 계산 실패: {e}")
            return 0.0
    
    def _calculate_center_point(self, p1: List[float], p2: List[float]) -> List[float]:
        """두 점의 중심점 계산"""
        try:
            if len(p1) < 2 or len(p2) < 2:
                return [0.0, 0.0]
            
            center_x = (p1[0] + p2[0]) / 2
            center_y = (p1[1] + p2[1]) / 2
            
            return [center_x, center_y]
            
        except Exception as e:
            self.logger.error(f"❌ 중심점 계산 실패: {e}")
            return [0.0, 0.0]
    
    def _calculate_angle_between_points(self, p1: List[float], p2: List[float]) -> float:
        """두 점 사이의 각도 계산"""
        try:
            if len(p1) < 2 or len(p2) < 2:
                return 0.0
            
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            angle = np.arctan2(dy, dx)
            return np.degrees(angle)
            
        except Exception as e:
            self.logger.error(f"❌ 두 점 간 각도 계산 실패: {e}")
            return 0.0
