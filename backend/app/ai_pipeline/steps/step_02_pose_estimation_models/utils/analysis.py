"""
포즈 분석 유틸리티
"""
import math
from typing import Dict, Any, List, Tuple
from ..config.constants import CLOTHING_IMPORTANT_JOINTS, SYMMETRIC_JOINT_PAIRS


def analyze_pose_for_clothing(
    keypoints: List[List[float]],
    clothing_type: str = "default",
    confidence_threshold: float = 0.5
) -> Dict[str, Any]:
    """의류 피팅을 위한 포즈 분석"""
    try:
        if not keypoints or len(keypoints) == 0:
            return {
                'success': False,
                'error': '키포인트가 없습니다'
            }
        
        # 신뢰도 필터링
        valid_keypoints = []
        for i, kp in enumerate(keypoints):
            if len(kp) >= 3 and kp[2] >= confidence_threshold:
                valid_keypoints.append((i, kp))
        
        if len(valid_keypoints) < 5:
            return {
                'success': False,
                'error': '유효한 키포인트가 부족합니다'
            }
        
        # 의류 피팅에 중요한 관절 분석
        clothing_analysis = analyze_clothing_joints(valid_keypoints)
        
        # 포즈 안정성 분석
        stability_score = analyze_posture_stability(valid_keypoints)
        
        # 의류별 특화 분석
        clothing_specific = analyze_clothing_specific_requirements(
            valid_keypoints, clothing_type
        )
        
        return {
            'success': True,
            'clothing_analysis': clothing_analysis,
            'stability_score': stability_score,
            'clothing_specific': clothing_specific,
            'valid_keypoints_count': len(valid_keypoints),
            'total_keypoints_count': len(keypoints)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'포즈 분석 실패: {e}'
        }


def analyze_clothing_joints(valid_keypoints: List[Tuple[int, List[float]]]) -> Dict[str, Any]:
    """의류 피팅에 중요한 관절 분석"""
    try:
        analysis = {
            'shoulder_alignment': 0.0,
            'hip_alignment': 0.0,
            'arm_angles': {},
            'leg_angles': {},
            'torso_straightness': 0.0
        }
        
        # 어깨 정렬 분석
        left_shoulder = None
        right_shoulder = None
        
        for idx, kp in valid_keypoints:
            if idx == 5:  # left_shoulder
                left_shoulder = kp
            elif idx == 6:  # right_shoulder
                right_shoulder = kp
        
        if left_shoulder and right_shoulder:
            # 어깨 높이 차이 계산
            height_diff = abs(left_shoulder[1] - right_shoulder[1])
            analysis['shoulder_alignment'] = max(0.0, 1.0 - height_diff * 10)
        
        # 엉덩이 정렬 분석
        left_hip = None
        right_hip = None
        
        for idx, kp in valid_keypoints:
            if idx == 11:  # left_hip
                left_hip = kp
            elif idx == 12:  # right_hip
                right_hip = kp
        
        if left_hip and right_hip:
            # 엉덩이 높이 차이 계산
            height_diff = abs(left_hip[1] - right_hip[1])
            analysis['hip_alignment'] = max(0.0, 1.0 - height_diff * 10)
        
        # 팔 각도 분석
        arm_angles = calculate_arm_angles(valid_keypoints)
        analysis['arm_angles'] = arm_angles
        
        # 다리 각도 분석
        leg_angles = calculate_leg_angles(valid_keypoints)
        analysis['leg_angles'] = leg_angles
        
        # 몸통 직선성 분석
        torso_straightness = calculate_torso_straightness(valid_keypoints)
        analysis['torso_straightness'] = torso_straightness
        
        return analysis
        
    except Exception as e:
        return {
            'error': f'관절 분석 실패: {e}'
        }


def analyze_posture_stability(valid_keypoints: List[Tuple[int, List[float]]]) -> float:
    """포즈 안정성 분석"""
    try:
        if len(valid_keypoints) < 3:
            return 0.0
        
        # 좌우 대칭성 분석
        symmetry_score = 0.0
        symmetry_count = 0
        
        for left_idx, right_idx in SYMMETRIC_JOINT_PAIRS:
            left_kp = None
            right_kp = None
            
            for idx, kp in valid_keypoints:
                if idx == left_idx:
                    left_kp = kp
                elif idx == right_idx:
                    right_kp = kp
            
            if left_kp and right_kp:
                # 좌우 거리 계산
                distance = math.sqrt(
                    (left_kp[0] - right_kp[0])**2 + 
                    (left_kp[1] - right_kp[1])**2
                )
                symmetry_score += distance
                symmetry_count += 1
        
        if symmetry_count > 0:
            avg_symmetry = symmetry_score / symmetry_count
            # 거리가 적당하면 안정적 (너무 가깝거나 멀면 불안정)
            stability = max(0.0, 1.0 - abs(avg_symmetry - 0.3) * 2)
        else:
            stability = 0.5
        
        return stability
        
    except Exception as e:
        return 0.5


def analyze_clothing_specific_requirements(
    valid_keypoints: List[Tuple[int, List[float]]], 
    clothing_type: str
) -> Dict[str, Any]:
    """의류별 특화 요구사항 분석"""
    try:
        analysis = {
            'fit_score': 0.0,
            'recommendations': [],
            'issues': []
        }
        
        if clothing_type == "shirt" or clothing_type == "top":
            # 상의 피팅 분석
            shoulder_score = analyze_shoulder_fit(valid_keypoints)
            arm_score = analyze_arm_fit(valid_keypoints)
            torso_score = analyze_torso_fit(valid_keypoints)
            
            analysis['fit_score'] = (shoulder_score + arm_score + torso_score) / 3
            
            if shoulder_score < 0.7:
                analysis['issues'].append("어깨 정렬이 불안정합니다")
            if arm_score < 0.7:
                analysis['issues'].append("팔 각도가 부자연스럽습니다")
            if torso_score < 0.7:
                analysis['issues'].append("몸통이 기울어져 있습니다")
                
        elif clothing_type == "pants" or clothing_type == "bottom":
            # 하의 피팅 분석
            hip_score = analyze_hip_fit(valid_keypoints)
            leg_score = analyze_leg_fit(valid_keypoints)
            
            analysis['fit_score'] = (hip_score + leg_score) / 2
            
            if hip_score < 0.7:
                analysis['issues'].append("엉덩이 정렬이 불안정합니다")
            if leg_score < 0.7:
                analysis['issues'].append("다리 각도가 부자연스럽습니다")
        
        # 추천사항 생성
        if analysis['fit_score'] > 0.8:
            analysis['recommendations'].append("포즈가 안정적입니다")
        elif analysis['fit_score'] > 0.6:
            analysis['recommendations'].append("포즈를 약간 조정하면 더 좋습니다")
        else:
            analysis['recommendations'].append("포즈를 다시 잡아주세요")
        
        return analysis
        
    except Exception as e:
        return {
            'fit_score': 0.0,
            'recommendations': [],
            'issues': [f'분석 실패: {e}']
        }


def calculate_arm_angles(valid_keypoints: List[Tuple[int, List[float]]]) -> Dict[str, float]:
    """팔 각도 계산"""
    angles = {}
    
    try:
        # 왼팔 각도
        left_shoulder = get_keypoint_by_index(valid_keypoints, 5)
        left_elbow = get_keypoint_by_index(valid_keypoints, 7)
        left_wrist = get_keypoint_by_index(valid_keypoints, 9)
        
        if left_shoulder and left_elbow and left_wrist:
            angles['left_arm'] = calculate_angle_3points(left_shoulder, left_elbow, left_wrist)
        
        # 오른팔 각도
        right_shoulder = get_keypoint_by_index(valid_keypoints, 6)
        right_elbow = get_keypoint_by_index(valid_keypoints, 8)
        right_wrist = get_keypoint_by_index(valid_keypoints, 10)
        
        if right_shoulder and right_elbow and right_wrist:
            angles['right_arm'] = calculate_angle_3points(right_shoulder, right_elbow, right_wrist)
        
    except Exception as e:
        pass
    
    return angles


def calculate_leg_angles(valid_keypoints: List[Tuple[int, List[float]]]) -> Dict[str, float]:
    """다리 각도 계산"""
    angles = {}
    
    try:
        # 왼다리 각도
        left_hip = get_keypoint_by_index(valid_keypoints, 11)
        left_knee = get_keypoint_by_index(valid_keypoints, 13)
        left_ankle = get_keypoint_by_index(valid_keypoints, 15)
        
        if left_hip and left_knee and left_ankle:
            angles['left_leg'] = calculate_angle_3points(left_hip, left_knee, left_ankle)
        
        # 오른다리 각도
        right_hip = get_keypoint_by_index(valid_keypoints, 12)
        right_knee = get_keypoint_by_index(valid_keypoints, 14)
        right_ankle = get_keypoint_by_index(valid_keypoints, 16)
        
        if right_hip and right_knee and right_ankle:
            angles['right_leg'] = calculate_angle_3points(right_hip, right_knee, right_ankle)
        
    except Exception as e:
        pass
    
    return angles


def calculate_torso_straightness(valid_keypoints: List[Tuple[int, List[float]]]) -> float:
    """몸통 직선성 계산"""
    try:
        # 어깨와 엉덩이를 연결한 선의 기울기 계산
        left_shoulder = get_keypoint_by_index(valid_keypoints, 5)
        right_shoulder = get_keypoint_by_index(valid_keypoints, 6)
        left_hip = get_keypoint_by_index(valid_keypoints, 11)
        right_hip = get_keypoint_by_index(valid_keypoints, 12)
        
        if left_shoulder and right_shoulder and left_hip and right_hip:
            # 어깨 중심점
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            
            # 엉덩이 중심점
            hip_center_x = (left_hip[0] + right_hip[0]) / 2
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            
            # 기울기 계산
            if abs(hip_center_x - shoulder_center_x) > 0.01:
                slope = abs((hip_center_y - shoulder_center_y) / (hip_center_x - shoulder_center_x))
                # 기울기가 작을수록 직선적
                straightness = max(0.0, 1.0 - slope)
            else:
                straightness = 1.0
            
            return straightness
        
        return 0.5
        
    except Exception as e:
        return 0.5


def get_keypoint_by_index(valid_keypoints: List[Tuple[int, List[float]]], index: int) -> List[float]:
    """인덱스로 키포인트 찾기"""
    for idx, kp in valid_keypoints:
        if idx == index:
            return kp
    return None


def calculate_angle_3points(p1: List[float], p2: List[float], p3: List[float]) -> float:
    """3점으로 각도 계산"""
    try:
        # 벡터 계산
        v1 = [p1[0] - p2[0], p1[1] - p2[1]]
        v2 = [p3[0] - p2[0], p3[1] - p2[1]]
        
        # 내적 계산
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        
        # 벡터 크기 계산
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 > 0 and mag2 > 0:
            cos_angle = dot_product / (mag1 * mag2)
            cos_angle = max(-1.0, min(1.0, cos_angle))  # 범위 제한
            angle = math.acos(cos_angle)
            return math.degrees(angle)
        
        return 0.0
        
    except Exception as e:
        return 0.0


def analyze_shoulder_fit(valid_keypoints: List[Tuple[int, List[float]]]) -> float:
    """어깨 피팅 분석"""
    return analyze_clothing_joints(valid_keypoints).get('shoulder_alignment', 0.0)


def analyze_arm_fit(valid_keypoints: List[Tuple[int, List[float]]]) -> float:
    """팔 피팅 분석"""
    arm_angles = calculate_arm_angles(valid_keypoints)
    if arm_angles:
        # 팔 각도가 자연스러운 범위에 있는지 확인
        natural_angles = []
        for angle in arm_angles.values():
            if 30 <= angle <= 150:  # 자연스러운 팔 각도 범위
                natural_angles.append(1.0)
            else:
                natural_angles.append(0.5)
        return sum(natural_angles) / len(natural_angles) if natural_angles else 0.5
    return 0.5


def analyze_torso_fit(valid_keypoints: List[Tuple[int, List[float]]]) -> float:
    """몸통 피팅 분석"""
    return calculate_torso_straightness(valid_keypoints)


def analyze_hip_fit(valid_keypoints: List[Tuple[int, List[float]]]) -> float:
    """엉덩이 피팅 분석"""
    return analyze_clothing_joints(valid_keypoints).get('hip_alignment', 0.0)


def analyze_leg_fit(valid_keypoints: List[Tuple[int, List[float]]]) -> float:
    """다리 피팅 분석"""
    leg_angles = calculate_leg_angles(valid_keypoints)
    if leg_angles:
        # 다리 각도가 자연스러운 범위에 있는지 확인
        natural_angles = []
        for angle in leg_angles.values():
            if 60 <= angle <= 180:  # 자연스러운 다리 각도 범위
                natural_angles.append(1.0)
            else:
                natural_angles.append(0.5)
        return sum(natural_angles) / len(natural_angles) if natural_angles else 0.5
    return 0.5
