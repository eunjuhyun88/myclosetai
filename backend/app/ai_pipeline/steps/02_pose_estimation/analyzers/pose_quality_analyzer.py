#!/usr/bin/env python3
"""
🔥 MyCloset AI - Pose Estimation Quality Analyzer
================================================

✅ 포즈 품질 분석 기능 분리
✅ 기존 step.py 기능 보존
✅ 모듈화된 구조 적용
"""

import logging
from app.ai_pipeline.utils.common_imports import (
    np, math, Dict, Any, Optional, Tuple, List, Union
)

logger = logging.getLogger(__name__)

class PoseQualityAnalyzer:
    """포즈 품질 분석기"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PoseQualityAnalyzer")
    
    def assess_pose_quality(self, 
                          keypoints: List[List[float]], 
                          joint_angles: Dict[str, float], 
                          body_proportions: Dict[str, float]) -> Dict[str, Any]:
        """포즈 품질 평가"""
        try:
            if not keypoints:
                return {'quality_score': 0.0, 'quality_level': 'very_poor', 'issues': ['키포인트가 없습니다']}
            
            # 각 품질 지표 계산
            anatomical_score = self._assess_anatomical_plausibility(keypoints, joint_angles)
            symmetry_score = self._assess_body_symmetry(keypoints, body_proportions)
            balance_score = self._analyze_left_right_balance(keypoints)
            alignment_score = self._analyze_posture_alignment(keypoints)
            
            # 종합 품질 점수 계산
            quality_score = (anatomical_score + symmetry_score + balance_score + alignment_score) / 4
            
            # 품질 레벨 결정
            quality_level = self._determine_quality_level(quality_score)
            
            # 문제점 식별
            issues = self._identify_pose_issues(keypoints, joint_angles, body_proportions, {
                'anatomical': anatomical_score,
                'symmetry': symmetry_score,
                'balance': balance_score,
                'alignment': alignment_score
            })
            
            # 권장사항 생성
            recommendations = self._generate_pose_recommendations(issues, {
                'anatomical': anatomical_score,
                'symmetry': symmetry_score,
                'balance': balance_score,
                'alignment': alignment_score
            })
            
            return {
                'quality_score': quality_score,
                'quality_level': quality_level,
                'anatomical_score': anatomical_score,
                'symmetry_score': symmetry_score,
                'balance_score': balance_score,
                'alignment_score': alignment_score,
                'issues': issues,
                'recommendations': recommendations
            }
            
        except Exception as e:
            self.logger.error(f"❌ 포즈 품질 평가 실패: {e}")
            return {'quality_score': 0.0, 'quality_level': 'unknown', 'issues': [str(e)]}
    
    def _assess_anatomical_plausibility(self, keypoints: List[List[float]], joint_angles: Dict[str, float]) -> float:
        """해부학적 타당성 평가"""
        try:
            score = 0.0
            total_checks = 0
            
            # 관절 각도 검증
            if joint_angles:
                for angle_name, angle_value in joint_angles.items():
                    if self._validate_joint_angle(angle_name, angle_value):
                        score += 1.0
                    total_checks += 1
            
            # 키포인트 간 거리 검증
            if len(keypoints) >= 17:
                # 어깨-팔꿈치-손목 비율 검증
                if self._validate_arm_proportions(keypoints):
                    score += 1.0
                total_checks += 1
                
                # 엉덩이-무릎-발목 비율 검증
                if self._validate_leg_proportions(keypoints):
                    score += 1.0
                total_checks += 1
            
            return score / max(total_checks, 1)
            
        except Exception as e:
            self.logger.error(f"❌ 해부학적 타당성 평가 실패: {e}")
            return 0.0
    
    def _assess_body_symmetry(self, keypoints: List[List[float]], body_proportions: Dict[str, float]) -> float:
        """신체 대칭성 평가"""
        try:
            if len(keypoints) < 17:
                return 0.0
            
            symmetry_scores = []
            
            # 좌우 어깨 대칭성
            if len(keypoints) > 6:
                left_shoulder = keypoints[5]
                right_shoulder = keypoints[6]
                if all(left_shoulder) and all(right_shoulder):
                    shoulder_symmetry = self._calculate_symmetry_score(left_shoulder, right_shoulder)
                    symmetry_scores.append(shoulder_symmetry)
            
            # 좌우 엉덩이 대칭성
            if len(keypoints) > 12:
                left_hip = keypoints[11]
                right_hip = keypoints[12]
                if all(left_hip) and all(right_hip):
                    hip_symmetry = self._calculate_symmetry_score(left_hip, right_hip)
                    symmetry_scores.append(hip_symmetry)
            
            # 좌우 무릎 대칭성
            if len(keypoints) > 14:
                left_knee = keypoints[13]
                right_knee = keypoints[14]
                if all(left_knee) and all(right_knee):
                    knee_symmetry = self._calculate_symmetry_score(left_knee, right_knee)
                    symmetry_scores.append(knee_symmetry)
            
            return np.mean(symmetry_scores) if symmetry_scores else 0.0
            
        except Exception as e:
            self.logger.error(f"❌ 신체 대칭성 평가 실패: {e}")
            return 0.0
    
    def _analyze_left_right_balance(self, keypoints: List[List[float]]) -> float:
        """좌우 균형 분석"""
        try:
            if len(keypoints) < 17:
                return 0.0
            
            # 중심선 계산
            center_x = np.mean([kp[0] for kp in keypoints if len(kp) >= 2 and kp[2] > 0.5])
            
            # 좌우 키포인트들의 중심선으로부터의 거리 계산
            left_distances = []
            right_distances = []
            
            # 좌측 키포인트들
            left_indices = [5, 7, 9, 11, 13, 15]  # 왼쪽 어깨, 팔꿈치, 손목, 엉덩이, 무릎, 발목
            for idx in left_indices:
                if idx < len(keypoints) and len(keypoints[idx]) >= 2 and keypoints[idx][2] > 0.5:
                    distance = abs(keypoints[idx][0] - center_x)
                    left_distances.append(distance)
            
            # 우측 키포인트들
            right_indices = [6, 8, 10, 12, 14, 16]  # 오른쪽 어깨, 팔꿈치, 손목, 엉덩이, 무릎, 발목
            for idx in right_indices:
                if idx < len(keypoints) and len(keypoints[idx]) >= 2 and keypoints[idx][2] > 0.5:
                    distance = abs(keypoints[idx][0] - center_x)
                    right_distances.append(distance)
            
            # 좌우 균형 점수 계산
            if left_distances and right_distances:
                left_avg = np.mean(left_distances)
                right_avg = np.mean(right_distances)
                balance_score = 1.0 - abs(left_avg - right_avg) / max(left_avg + right_avg, 1e-6)
                return max(balance_score, 0.0)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"❌ 좌우 균형 분석 실패: {e}")
            return 0.0
    
    def _analyze_posture_alignment(self, keypoints: List[List[float]]) -> float:
        """자세 정렬 분석"""
        try:
            if len(keypoints) < 17:
                return 0.0
            
            alignment_scores = []
            
            # 척추 정렬 (어깨-엉덩이-무릎)
            if len(keypoints) > 14:
                shoulder_center = self._calculate_center_point(keypoints[5], keypoints[6])
                hip_center = self._calculate_center_point(keypoints[11], keypoints[12])
                knee_center = self._calculate_center_point(keypoints[13], keypoints[14])
                
                if all(shoulder_center) and all(hip_center) and all(knee_center):
                    spine_alignment = self._calculate_spine_alignment(shoulder_center, hip_center, knee_center)
                    alignment_scores.append(spine_alignment)
            
            # 머리-어깨 정렬
            if len(keypoints) > 6:
                nose = keypoints[0]
                shoulder_center = self._calculate_center_point(keypoints[5], keypoints[6])
                
                if all(nose) and all(shoulder_center):
                    head_shoulder_alignment = self._calculate_head_shoulder_alignment(nose, shoulder_center)
                    alignment_scores.append(head_shoulder_alignment)
            
            return np.mean(alignment_scores) if alignment_scores else 0.0
            
        except Exception as e:
            self.logger.error(f"❌ 자세 정렬 분석 실패: {e}")
            return 0.0
    
    def _determine_quality_level(self, quality_score: float) -> str:
        """품질 레벨 결정"""
        if quality_score >= 0.9:
            return 'excellent'
        elif quality_score >= 0.75:
            return 'good'
        elif quality_score >= 0.6:
            return 'acceptable'
        elif quality_score >= 0.4:
            return 'poor'
        else:
            return 'very_poor'
    
    def _identify_pose_issues(self, 
                            keypoints: List[List[float]], 
                            joint_angles: Dict[str, float], 
                            body_proportions: Dict[str, float],
                            scores: Dict[str, float]) -> List[str]:
        """포즈 문제점 식별"""
        try:
            issues = []
            
            # 품질 점수 기반 문제점 식별
            for metric, score in scores.items():
                if score < 0.5:
                    if metric == 'anatomical':
                        issues.append('해부학적으로 부자연스러운 자세')
                    elif metric == 'symmetry':
                        issues.append('신체 대칭성이 부족함')
                    elif metric == 'balance':
                        issues.append('좌우 균형이 맞지 않음')
                    elif metric == 'alignment':
                        issues.append('자세 정렬이 부족함')
            
            # 관절 각도 기반 문제점 식별
            for angle_name, angle_value in joint_angles.items():
                if not self._validate_joint_angle(angle_name, angle_value):
                    issues.append(f'{angle_name} 각도가 부자연스러움')
            
            return issues
            
        except Exception as e:
            self.logger.error(f"❌ 포즈 문제점 식별 실패: {e}")
            return ['문제점 분석 실패']
    
    def _generate_pose_recommendations(self, issues: List[str], scores: Dict[str, float]) -> List[str]:
        """포즈 권장사항 생성"""
        try:
            recommendations = []
            
            for issue in issues:
                if '해부학적' in issue:
                    recommendations.append('자연스러운 자세를 취해주세요')
                elif '대칭성' in issue:
                    recommendations.append('좌우 균형을 맞춰주세요')
                elif '균형' in issue:
                    recommendations.append('중앙을 기준으로 균형을 맞춰주세요')
                elif '정렬' in issue:
                    recommendations.append('척추를 곧게 펴고 어깨를 펴주세요')
                elif '각도' in issue:
                    recommendations.append('관절 각도를 자연스럽게 조정해주세요')
            
            # 일반적인 권장사항
            if not recommendations:
                recommendations.append('전체적으로 좋은 자세입니다')
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ 포즈 권장사항 생성 실패: {e}")
            return ['권장사항 생성 실패']
    
    def _validate_joint_angle(self, angle_name: str, angle_value: float) -> bool:
        """관절 각도 검증"""
        try:
            # 관절별 정상 각도 범위 정의
            angle_ranges = {
                'left_shoulder': (0, 180),
                'right_shoulder': (0, 180),
                'left_knee': (0, 180),
                'right_knee': (0, 180),
                'left_elbow': (0, 180),
                'right_elbow': (0, 180)
            }
            
            if angle_name in angle_ranges:
                min_angle, max_angle = angle_ranges[angle_name]
                return min_angle <= angle_value <= max_angle
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 관절 각도 검증 실패: {e}")
            return False
    
    def _validate_arm_proportions(self, keypoints: List[List[float]]) -> bool:
        """팔 비율 검증"""
        try:
            if len(keypoints) < 10:
                return False
            
            # 어깨-팔꿈치-손목 거리 계산
            shoulder_elbow = self._calculate_distance(keypoints[5], keypoints[7])
            elbow_wrist = self._calculate_distance(keypoints[7], keypoints[9])
            
            if shoulder_elbow > 0 and elbow_wrist > 0:
                ratio = elbow_wrist / shoulder_elbow
                return 0.5 <= ratio <= 2.0
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ 팔 비율 검증 실패: {e}")
            return False
    
    def _validate_leg_proportions(self, keypoints: List[List[float]]) -> bool:
        """다리 비율 검증"""
        try:
            if len(keypoints) < 16:
                return False
            
            # 엉덩이-무릎-발목 거리 계산
            hip_knee = self._calculate_distance(keypoints[11], keypoints[13])
            knee_ankle = self._calculate_distance(keypoints[13], keypoints[15])
            
            if hip_knee > 0 and knee_ankle > 0:
                ratio = knee_ankle / hip_knee
                return 0.5 <= ratio <= 2.0
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ 다리 비율 검증 실패: {e}")
            return False
    
    def _calculate_symmetry_score(self, point1: List[float], point2: List[float]) -> float:
        """대칭성 점수 계산"""
        try:
            if len(point1) < 2 or len(point2) < 2:
                return 0.0
            
            # Y축 기준 대칭성 계산
            y_diff = abs(point1[1] - point2[1])
            max_y = max(point1[1], point2[1])
            
            if max_y > 0:
                symmetry_score = 1.0 - (y_diff / max_y)
                return max(symmetry_score, 0.0)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"❌ 대칭성 점수 계산 실패: {e}")
            return 0.0
    
    def _calculate_center_point(self, point1: List[float], point2: List[float]) -> List[float]:
        """두 점의 중심점 계산"""
        try:
            if len(point1) < 2 or len(point2) < 2:
                return [0.0, 0.0]
            
            center_x = (point1[0] + point2[0]) / 2
            center_y = (point1[1] + point2[1]) / 2
            
            return [center_x, center_y]
            
        except Exception as e:
            self.logger.error(f"❌ 중심점 계산 실패: {e}")
            return [0.0, 0.0]
    
    def _calculate_spine_alignment(self, shoulder_center: List[float], hip_center: List[float], knee_center: List[float]) -> float:
        """척추 정렬 계산"""
        try:
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
            self.logger.error(f"❌ 척추 정렬 계산 실패: {e}")
            return 0.0
    
    def _calculate_head_shoulder_alignment(self, nose: List[float], shoulder_center: List[float]) -> float:
        """머리-어깨 정렬 계산"""
        try:
            if len(nose) < 2 or len(shoulder_center) < 2:
                return 0.0
            
            # 머리가 어깨 중앙 위에 있는지 확인
            x_diff = abs(nose[0] - shoulder_center[0])
            max_x = max(abs(nose[0]), abs(shoulder_center[0]))
            
            if max_x > 0:
                alignment_score = 1.0 - (x_diff / max_x)
                return max(alignment_score, 0.0)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"❌ 머리-어깨 정렬 계산 실패: {e}")
            return 0.0
    
    def _calculate_distance(self, point1: List[float], point2: List[float]) -> float:
        """두 점 간의 거리 계산"""
        try:
            if len(point1) < 2 or len(point2) < 2:
                return 0.0
            
            return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
            
        except Exception as e:
            self.logger.error(f"❌ 거리 계산 실패: {e}")
            return 0.0
