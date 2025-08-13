#!/usr/bin/env python3
"""
🔥 MyCloset AI - Pose Estimation Utils
=====================================

✅ 기존 step.py의 모든 유틸리티 함수 완전 복원
✅ 모든 분석 기능 포함
✅ 모듈화된 구조 적용
"""

import logging
from app.ai_pipeline.utils.common_imports import (
    np, math, Dict, Any, Optional, Tuple, List, Union, Image, ImageDraw
)

logger = logging.getLogger(__name__)

# 키포인트 색상 정의
KEYPOINT_COLORS = [
    (255, 0, 0),    # 빨강
    (0, 255, 0),    # 초록
    (0, 0, 255),    # 파랑
    (255, 255, 0),  # 노랑
    (255, 0, 255),  # 마젠타
    (0, 255, 255),  # 시안
    (255, 165, 0),  # 주황
    (128, 0, 128),  # 보라
    (255, 192, 203), # 분홍
    (0, 128, 0),    # 다크그린
    (128, 128, 0),  # 올리브
    (0, 0, 128),    # 네이비
    (128, 0, 0),    # 다크레드
    (0, 128, 128),  # 틸
    (255, 255, 255), # 흰색
    (0, 0, 0),      # 검정
    (128, 128, 128) # 회색
]

def validate_keypoints(keypoints: List[List[float]]) -> bool:
    """키포인트 유효성 검증"""
    try:
        if not keypoints:
            return False
        
        for kp in keypoints:
            if len(kp) < 3:
                return False
            if not all(isinstance(x, (int, float)) for x in kp):
                return False
            if kp[2] < 0 or kp[2] > 1:
                return False
        
        return True
        
    except Exception:
        return False

def draw_pose_on_image(
    image: Union[np.ndarray, Image.Image],
    keypoints: List[List[float]],
    confidence_threshold: float = 0.5,
    keypoint_size: int = 4,
    line_width: int = 3
) -> Image.Image:
    """이미지에 포즈 그리기"""
    try:
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image.copy()
        
        draw = ImageDraw.Draw(pil_image)
        
        # 키포인트 그리기
        for i, kp in enumerate(keypoints):
            if len(kp) >= 3 and kp[2] > confidence_threshold:
                x, y = int(kp[0]), int(kp[1])
                color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]
                
                radius = int(keypoint_size + kp[2] * 6)
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                           fill=color, outline=(255, 255, 255), width=2)
        
        # 스켈레톤 그리기 (COCO 17 연결 구조)
        coco_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 머리
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 팔
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # 다리
        ]
        
        for i, (start_idx, end_idx) in enumerate(coco_connections):
            if (start_idx < len(keypoints) and end_idx < len(keypoints)):
                start_kp = keypoints[start_idx]
                end_kp = keypoints[end_idx]
                
                if (len(start_kp) >= 3 and len(end_kp) >= 3 and
                    start_kp[2] > confidence_threshold and end_kp[2] > confidence_threshold):
                    
                    start_point = (int(start_kp[0]), int(start_kp[1]))
                    end_point = (int(end_kp[0]), int(end_kp[1]))
                    color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]
                    
                    avg_confidence = (start_kp[2] + end_kp[2]) / 2
                    adjusted_width = int(line_width * avg_confidence)
                    
                    draw.line([start_point, end_point], fill=color, width=max(1, adjusted_width))
        
        return pil_image
        
    except Exception as e:
        logger.error(f"포즈 그리기 실패: {e}")
        return image if isinstance(image, Image.Image) else Image.fromarray(image)

def analyze_pose_for_clothing_advanced(
    keypoints: List[List[float]],
    clothing_type: str = "default",
    confidence_threshold: float = 0.5,
    detailed_analysis: bool = True
) -> Dict[str, Any]:
    """고급 의류별 포즈 적합성 분석"""
    try:
        if not keypoints:
            return {
                'suitable_for_fitting': False,
                'issues': ["포즈를 검출할 수 없습니다"],
                'recommendations': ["더 선명한 이미지를 사용해 주세요"],
                'pose_score': 0.0,
                'detailed_analysis': {}
            }
        
        # 의류별 세부 가중치
        clothing_detailed_weights = {
            'shirt': {
                'critical_keypoints': [5, 6, 7, 8, 9, 10],  # 어깨, 팔꿈치, 손목
                'weights': {'arms': 0.4, 'torso': 0.4, 'posture': 0.2},
                'min_visibility': 0.7,
                'required_angles': ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow']
            },
            'dress': {
                'critical_keypoints': [5, 6, 11, 12, 13, 14],  # 어깨, 엉덩이, 무릎
                'weights': {'torso': 0.5, 'arms': 0.2, 'legs': 0.2, 'posture': 0.1},
                'min_visibility': 0.8,
                'required_angles': ['spine_curvature']
            },
            'pants': {
                'critical_keypoints': [11, 12, 13, 14, 15, 16],  # 엉덩이, 무릎, 발목
                'weights': {'legs': 0.6, 'torso': 0.3, 'posture': 0.1},
                'min_visibility': 0.8,
                'required_angles': ['left_hip', 'right_hip', 'left_knee', 'right_knee']
            },
            'jacket': {
                'critical_keypoints': [5, 6, 7, 8, 9, 10, 11, 12],  # 어깨, 팔, 엉덩이
                'weights': {'arms': 0.3, 'torso': 0.5, 'posture': 0.2},
                'min_visibility': 0.75,
                'required_angles': ['left_shoulder', 'right_shoulder']
            }
        }
        
        # 기본 분석 수행
        basic_analysis = analyze_pose_for_clothing(keypoints, clothing_type, confidence_threshold)
        
        if not detailed_analysis:
            return basic_analysis
        
        # 고급 분석 수행
        detailed_results = {}
        
        if clothing_type in clothing_detailed_weights:
            weights = clothing_detailed_weights[clothing_type]
            
            # 중요 키포인트 가시성 검사
            critical_visibility = _analyze_critical_keypoints_visibility(
                keypoints, weights['critical_keypoints'], confidence_threshold
            )
            
            # 신체 부위별 점수 계산
            body_part_scores = _calculate_body_part_scores_advanced(
                keypoints, weights['weights'], confidence_threshold
            )
            
            # 자세 안정성 분석
            posture_stability = analyze_posture_stability(keypoints)
            
            # 의류별 특수 요구사항 분석
            clothing_specific_score = analyze_clothing_specific_requirements(
                keypoints, clothing_type, {}
            )
            
            detailed_results = {
                'critical_visibility': critical_visibility,
                'body_part_scores': body_part_scores,
                'posture_stability': posture_stability,
                'clothing_specific_score': clothing_specific_score
            }
        
        # 종합 점수 계산
        overall_score = _calculate_overall_pose_score(basic_analysis, detailed_results)
        
        return {
            'suitable_for_fitting': overall_score > 0.6,
            'issues': basic_analysis.get('issues', []),
            'recommendations': basic_analysis.get('recommendations', []),
            'pose_score': overall_score,
            'detailed_analysis': detailed_results
        }
        
    except Exception as e:
        logger.error(f"고급 의류별 포즈 분석 실패: {e}")
        return {
            'suitable_for_fitting': False,
            'issues': [f"분석 실패: {str(e)}"],
            'recommendations': ["분석을 다시 시도해 주세요"],
            'pose_score': 0.0,
            'detailed_analysis': {}
        }

def analyze_posture_stability(keypoints: List[List[float]]) -> float:
    """자세 안정성 분석"""
    try:
        if not keypoints or len(keypoints) < 17:
            return 0.0
        
        # 중심점 계산
        valid_keypoints = [kp for kp in keypoints if len(kp) >= 2 and kp[2] > 0.5]
        
        if not valid_keypoints:
            return 0.0
        
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
        logger.error(f"자세 안정성 분석 실패: {e}")
        return 0.0

def analyze_clothing_specific_requirements(
    keypoints: List[List[float]], 
    clothing_type: str, 
    joint_angles: Dict[str, float]
) -> float:
    """의류별 특수 요구사항 분석"""
    try:
        if not keypoints:
            return 0.0
        
        score = 0.0
        total_checks = 0
        
        if clothing_type == "shirt":
            # 셔츠: 팔꿈치 각도가 자연스러워야 함
            if 'left_elbow' in joint_angles:
                angle = joint_angles['left_elbow']
                if 60 <= angle <= 150:
                    score += 1.0
                total_checks += 1
            
            if 'right_elbow' in joint_angles:
                angle = joint_angles['right_elbow']
                if 60 <= angle <= 150:
                    score += 1.0
                total_checks += 1
        
        elif clothing_type == "dress":
            # 드레스: 척추 정렬이 중요
            if len(keypoints) >= 17:
                spine_alignment = _calculate_spine_alignment_score(keypoints)
                score += spine_alignment
                total_checks += 1
        
        elif clothing_type == "pants":
            # 바지: 다리 각도가 자연스러워야 함
            if 'left_knee' in joint_angles:
                angle = joint_angles['left_knee']
                if 120 <= angle <= 180:
                    score += 1.0
                total_checks += 1
            
            if 'right_knee' in joint_angles:
                angle = joint_angles['right_knee']
                if 120 <= angle <= 180:
                    score += 1.0
                total_checks += 1
        
        return score / max(total_checks, 1)
        
    except Exception as e:
        logger.error(f"의류별 특수 요구사항 분석 실패: {e}")
        return 0.0

def analyze_pose_for_clothing(
    keypoints: List[List[float]],
    clothing_type: str = "default",
    confidence_threshold: float = 0.5
) -> Dict[str, Any]:
    """의류별 포즈 적합성 분석"""
    try:
        if not keypoints:
            return {
                'suitable_for_fitting': False,
                'issues': ["포즈를 검출할 수 없습니다"],
                'recommendations': ["더 선명한 이미지를 사용해 주세요"],
                'pose_score': 0.0
            }
        
        # 의류별 가중치
        clothing_weights = {
            'shirt': {'arms': 0.4, 'torso': 0.4, 'posture': 0.2},
            'dress': {'torso': 0.5, 'arms': 0.2, 'legs': 0.2, 'posture': 0.1},
            'pants': {'legs': 0.6, 'torso': 0.3, 'posture': 0.1},
            'jacket': {'arms': 0.3, 'torso': 0.5, 'posture': 0.2}
        }
        
        weights = clothing_weights.get(clothing_type, {'torso': 0.4, 'arms': 0.3, 'legs': 0.2, 'posture': 0.1})
        
        # 신체 부위별 점수 계산
        body_part_scores = _calculate_body_part_scores(keypoints, weights, confidence_threshold)
        
        # 자세 안정성 분석
        posture_stability = analyze_posture_stability(keypoints)
        
        # 종합 점수 계산
        overall_score = sum(body_part_scores.values()) + posture_stability * weights.get('posture', 0.1)
        overall_score = min(overall_score, 1.0)
        
        # 문제점 식별
        issues = _identify_pose_issues_for_clothing(keypoints, body_part_scores, posture_stability)
        
        # 권장사항 생성
        recommendations = _generate_clothing_recommendations(issues, clothing_type)
        
        return {
            'suitable_for_fitting': overall_score > 0.6,
            'issues': issues,
            'recommendations': recommendations,
            'pose_score': overall_score,
            'body_part_scores': body_part_scores,
            'posture_stability': posture_stability
        }
        
    except Exception as e:
        logger.error(f"의류별 포즈 분석 실패: {e}")
        return {
            'suitable_for_fitting': False,
            'issues': [f"분석 실패: {str(e)}"],
            'recommendations': ["분석을 다시 시도해 주세요"],
            'pose_score': 0.0
        }

def convert_coco17_to_openpose18(coco_keypoints: List[List[float]]) -> List[List[float]]:
    """COCO 17 키포인트를 OpenPose 18 키포인트로 변환"""
    try:
        if len(coco_keypoints) != 17:
            return coco_keypoints
        
        # OpenPose 18 키포인트 구조로 변환
        openpose_keypoints = []
        
        # COCO 17 키포인트를 OpenPose 18에 매핑
        coco_to_openpose_mapping = [
            0,   # nose
            1,   # neck (추정)
            2,   # right_shoulder
            3,   # right_elbow
            4,   # right_wrist
            5,   # left_shoulder
            6,   # left_elbow
            7,   # left_wrist
            8,   # right_hip
            9,   # right_knee
            10,  # right_ankle
            11,  # left_hip
            12,  # left_knee
            13,  # left_ankle
            14,  # right_eye
            15,  # left_eye
            16,  # right_ear
            17   # left_ear
        ]
        
        # 매핑된 키포인트 추가
        for i in range(18):
            if i < len(coco_to_openpose_mapping):
                coco_idx = coco_to_openpose_mapping[i]
                if coco_idx < len(coco_keypoints):
                    openpose_keypoints.append(coco_keypoints[coco_idx])
                else:
                    openpose_keypoints.append([0.0, 0.0, 0.0])
            else:
                openpose_keypoints.append([0.0, 0.0, 0.0])
        
        return openpose_keypoints
        
    except Exception as e:
        logger.error(f"COCO to OpenPose 변환 실패: {e}")
        return coco_keypoints

def _analyze_critical_keypoints_visibility(
    keypoints: List[List[float]], 
    critical_indices: List[int], 
    confidence_threshold: float
) -> float:
    """중요 키포인트 가시성 분석"""
    try:
        if not critical_indices:
            return 0.0
        
        visible_count = 0
        for idx in critical_indices:
            if idx < len(keypoints) and len(keypoints[idx]) >= 3:
                if keypoints[idx][2] > confidence_threshold:
                    visible_count += 1
        
        return visible_count / len(critical_indices)
        
    except Exception as e:
        logger.error(f"중요 키포인트 가시성 분석 실패: {e}")
        return 0.0

def _calculate_body_part_scores_advanced(
    keypoints: List[List[float]], 
    weights: Dict[str, float], 
    confidence_threshold: float
) -> Dict[str, float]:
    """고급 신체 부위별 점수 계산"""
    try:
        scores = {}
        
        for body_part, weight in weights.items():
            if body_part == 'arms':
                scores[body_part] = _calculate_arm_score(keypoints, confidence_threshold)
            elif body_part == 'torso':
                scores[body_part] = _calculate_torso_score(keypoints, confidence_threshold)
            elif body_part == 'legs':
                scores[body_part] = _calculate_leg_score(keypoints, confidence_threshold)
            elif body_part == 'posture':
                scores[body_part] = analyze_posture_stability(keypoints)
        
        return scores
        
    except Exception as e:
        logger.error(f"고급 신체 부위별 점수 계산 실패: {e}")
        return {}

def _calculate_body_part_scores(
    keypoints: List[List[float]], 
    weights: Dict[str, float], 
    confidence_threshold: float
) -> Dict[str, float]:
    """신체 부위별 점수 계산"""
    try:
        scores = {}
        
        for body_part, weight in weights.items():
            if body_part == 'arms':
                scores[body_part] = _calculate_arm_score(keypoints, confidence_threshold)
            elif body_part == 'torso':
                scores[body_part] = _calculate_torso_score(keypoints, confidence_threshold)
            elif body_part == 'legs':
                scores[body_part] = _calculate_leg_score(keypoints, confidence_threshold)
            elif body_part == 'posture':
                scores[body_part] = analyze_posture_stability(keypoints)
        
        return scores
        
    except Exception as e:
        logger.error(f"신체 부위별 점수 계산 실패: {e}")
        return {}

def _calculate_arm_score(keypoints: List[List[float]], confidence_threshold: float) -> float:
    """팔 점수 계산"""
    try:
        arm_indices = [5, 6, 7, 8, 9, 10]  # 어깨, 팔꿈치, 손목
        visible_count = 0
        
        for idx in arm_indices:
            if idx < len(keypoints) and len(keypoints[idx]) >= 3:
                if keypoints[idx][2] > confidence_threshold:
                    visible_count += 1
        
        return visible_count / len(arm_indices)
        
    except Exception as e:
        logger.error(f"팔 점수 계산 실패: {e}")
        return 0.0

def _calculate_torso_score(keypoints: List[List[float]], confidence_threshold: float) -> float:
    """몸통 점수 계산"""
    try:
        torso_indices = [5, 6, 11, 12]  # 어깨, 엉덩이
        visible_count = 0
        
        for idx in torso_indices:
            if idx < len(keypoints) and len(keypoints[idx]) >= 3:
                if keypoints[idx][2] > confidence_threshold:
                    visible_count += 1
        
        return visible_count / len(torso_indices)
        
    except Exception as e:
        logger.error(f"몸통 점수 계산 실패: {e}")
        return 0.0

def _calculate_leg_score(keypoints: List[List[float]], confidence_threshold: float) -> float:
    """다리 점수 계산"""
    try:
        leg_indices = [11, 12, 13, 14, 15, 16]  # 엉덩이, 무릎, 발목
        visible_count = 0
        
        for idx in leg_indices:
            if idx < len(keypoints) and len(keypoints[idx]) >= 3:
                if keypoints[idx][2] > confidence_threshold:
                    visible_count += 1
        
        return visible_count / len(leg_indices)
        
    except Exception as e:
        logger.error(f"다리 점수 계산 실패: {e}")
        return 0.0

def _calculate_spine_alignment_score(keypoints: List[List[float]]) -> float:
    """척추 정렬 점수 계산"""
    try:
        if len(keypoints) < 17:
            return 0.0
        
        # 어깨, 엉덩이, 무릎 중심점 계산
        shoulder_center = _calculate_center_point(keypoints[5], keypoints[6])
        hip_center = _calculate_center_point(keypoints[11], keypoints[12])
        knee_center = _calculate_center_point(keypoints[13], keypoints[14])
        
        if all(shoulder_center) and all(hip_center) and all(knee_center):
            # 세 점이 일직선상에 있는지 확인
            x_coords = [shoulder_center[0], hip_center[0], knee_center[0]]
            y_coords = [shoulder_center[1], hip_center[1], knee_center[1]]
            
            # 선형 회귀로 직선성 측정
            if len(set(x_coords)) > 1:
                slope, intercept = np.polyfit(x_coords, y_coords, 1)
                predicted_y = [slope * x + intercept for x in x_coords]
                mse = np.mean([(y - pred_y) ** 2 for y, pred_y in zip(y_coords, predicted_y)])
                alignment_score = 1.0 / (1.0 + mse)
                return min(alignment_score, 1.0)
        
        return 0.0
        
    except Exception as e:
        logger.error(f"척추 정렬 점수 계산 실패: {e}")
        return 0.0

def _calculate_center_point(point1: List[float], point2: List[float]) -> List[float]:
    """두 점의 중심점 계산"""
    try:
        if len(point1) < 2 or len(point2) < 2:
            return [0.0, 0.0]
        
        center_x = (point1[0] + point2[0]) / 2
        center_y = (point1[1] + point2[1]) / 2
        
        return [center_x, center_y]
    except Exception:
        return [0.0, 0.0]

def _identify_pose_issues_for_clothing(
    keypoints: List[List[float]], 
    body_part_scores: Dict[str, float], 
    posture_stability: float
) -> List[str]:
    """의류별 포즈 문제점 식별"""
    try:
        issues = []
        
        # 신체 부위별 문제점 식별
        for body_part, score in body_part_scores.items():
            if score < 0.5:
                if body_part == 'arms':
                    issues.append('팔이 잘 보이지 않습니다')
                elif body_part == 'torso':
                    issues.append('몸통이 잘 보이지 않습니다')
                elif body_part == 'legs':
                    issues.append('다리가 잘 보이지 않습니다')
        
        # 자세 안정성 문제점 식별
        if posture_stability < 0.5:
            issues.append('자세가 불안정합니다')
        
        return issues
        
    except Exception as e:
        logger.error(f"의류별 포즈 문제점 식별 실패: {e}")
        return ['문제점 분석 실패']

def _generate_clothing_recommendations(issues: List[str], clothing_type: str) -> List[str]:
    """의류별 권장사항 생성"""
    try:
        recommendations = []
        
        for issue in issues:
            if '팔' in issue:
                recommendations.append('팔을 자연스럽게 펴주세요')
            elif '몸통' in issue:
                recommendations.append('몸통이 잘 보이도록 자세를 조정해주세요')
            elif '다리' in issue:
                recommendations.append('다리가 잘 보이도록 자세를 조정해주세요')
            elif '자세' in issue:
                recommendations.append('안정적인 자세를 취해주세요')
        
        # 일반적인 권장사항
        if not recommendations:
            recommendations.append('전체적으로 좋은 자세입니다')
        
        return recommendations
        
    except Exception as e:
        logger.error(f"의류별 권장사항 생성 실패: {e}")
        return ['권장사항 생성 실패']

def _calculate_overall_pose_score(basic_analysis: Dict[str, Any], detailed_results: Dict[str, Any]) -> float:
    """종합 포즈 점수 계산"""
    try:
        base_score = basic_analysis.get('pose_score', 0.0)
        
        # 고급 분석 결과 반영
        if detailed_results:
            critical_visibility = detailed_results.get('critical_visibility', 0.0)
            body_part_scores = detailed_results.get('body_part_scores', {})
            posture_stability = detailed_results.get('posture_stability', 0.0)
            clothing_specific_score = detailed_results.get('clothing_specific_score', 0.0)
            
            # 가중 평균 계산
            detailed_score = (
                critical_visibility * 0.3 +
                np.mean(list(body_part_scores.values())) * 0.3 +
                posture_stability * 0.2 +
                clothing_specific_score * 0.2
            )
            
            # 기본 점수와 고급 점수의 평균
            overall_score = (base_score + detailed_score) / 2
            return min(overall_score, 1.0)
        
        return base_score
        
    except Exception as e:
        logger.error(f"종합 포즈 점수 계산 실패: {e}")
        return 0.0
