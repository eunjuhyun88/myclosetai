#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥 MyCloset AI - Pose Analyzer
==============================

✅ 포즈 품질 분석 및 평가
✅ 관절 각도 계산
✅ 신체 비율 분석
✅ 가상 피팅 최적화

파일 위치: backend/app/ai_pipeline/steps/pose_estimation/utils/pose_analyzer.py
작성자: MyCloset AI Team
날짜: 2025-08-01
버전: v1.0
"""

import os
import sys
import time
import logging
import warnings
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

# 공통 imports
from app.ai_pipeline.utils.common_imports import (
    np, DEVICE, TORCH_AVAILABLE,
    Path, Dict, Any, Optional, Tuple, List, Union,
    dataclass, field, Enum
)

# 경고 무시
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=ImportWarning)

logger = logging.getLogger(__name__)

class PoseQuality(Enum):
    """포즈 품질 등급"""
    EXCELLENT = "excellent"     # 90-100점
    GOOD = "good"              # 75-89점  
    ACCEPTABLE = "acceptable"   # 60-74점
    POOR = "poor"              # 40-59점
    VERY_POOR = "very_poor"    # 0-39점

class PoseAnalyzer:
    """포즈 분석기 - 품질 평가 및 최적화"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PoseAnalyzer")
        
        # COCO 17 키포인트 정의
        self.coco_keypoints = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        # 관절 연결 정의
        self.joint_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 머리
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 상체
            (5, 11), (6, 12), (11, 12),  # 몸통
            (11, 13), (13, 15), (12, 14), (14, 16)  # 하체
        ]
        
        # 신체 부위 정의
        self.body_parts = {
            "head": [0, 1, 2, 3, 4],
            "torso": [5, 6, 11, 12],
            "left_arm": [5, 7, 9],
            "right_arm": [6, 8, 10],
            "left_leg": [11, 13, 15],
            "right_leg": [12, 14, 16]
        }
    
    @staticmethod
    def calculate_joint_angles(keypoints: List[List[float]]) -> Dict[str, float]:
        """관절 각도 계산"""
        try:
            angles = {}
            
            def calculate_angle_3points(p1, p2, p3):
                """3점으로 각도 계산"""
                if len(p1) < 2 or len(p2) < 2 or len(p3) < 2:
                    return 0.0
                
                # 벡터 계산
                v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
                v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
                
                # 벡터 정규화
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                
                if v1_norm == 0 or v2_norm == 0:
                    return 0.0
                
                v1 = v1 / v1_norm
                v2 = v2 / v2_norm
                
                # 각도 계산
                dot_product = np.dot(v1, v2)
                dot_product = np.clip(dot_product, -1.0, 1.0)
                angle = np.arccos(dot_product)
                
                return np.degrees(angle)
            
            def calculate_directional_angle(p1, p2, p3):
                """방향성을 고려한 각도 계산"""
                if len(p1) < 2 or len(p2) < 2 or len(p3) < 2:
                    return 0.0
                
                # 벡터 계산
                v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
                v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
                
                # 외적을 이용한 방향성 계산
                cross_product = np.cross(v1, v2)
                dot_product = np.dot(v1, v2)
                
                angle = np.arctan2(np.linalg.norm(cross_product), dot_product)
                angle = np.degrees(angle)
                
                # 방향성에 따른 부호 결정
                if cross_product < 0:
                    angle = -angle
                
                return angle
            
            # 주요 관절 각도 계산
            if len(keypoints) >= 17:
                # 어깨 각도
                if all(len(keypoints[i]) >= 2 for i in [5, 6, 11]):
                    angles["left_shoulder"] = calculate_angle_3points(
                        keypoints[5], keypoints[6], keypoints[11]
                    )
                
                if all(len(keypoints[i]) >= 2 for i in [6, 5, 12]):
                    angles["right_shoulder"] = calculate_angle_3points(
                        keypoints[6], keypoints[5], keypoints[12]
                    )
                
                # 팔꿈치 각도
                if all(len(keypoints[i]) >= 2 for i in [5, 7, 9]):
                    angles["left_elbow"] = calculate_angle_3points(
                        keypoints[5], keypoints[7], keypoints[9]
                    )
                
                if all(len(keypoints[i]) >= 2 for i in [6, 8, 10]):
                    angles["right_elbow"] = calculate_angle_3points(
                        keypoints[6], keypoints[8], keypoints[10]
                    )
                
                # 무릎 각도
                if all(len(keypoints[i]) >= 2 for i in [11, 13, 15]):
                    angles["left_knee"] = calculate_angle_3points(
                        keypoints[11], keypoints[13], keypoints[15]
                    )
                
                if all(len(keypoints[i]) >= 2 for i in [12, 14, 16]):
                    angles["right_knee"] = calculate_angle_3points(
                        keypoints[12], keypoints[14], keypoints[16]
                    )
                
                # 엉덩이 각도
                if all(len(keypoints[i]) >= 2 for i in [5, 11, 13]):
                    angles["left_hip"] = calculate_angle_3points(
                        keypoints[5], keypoints[11], keypoints[13]
                    )
                
                if all(len(keypoints[i]) >= 2 for i in [6, 12, 14]):
                    angles["right_hip"] = calculate_angle_3points(
                        keypoints[6], keypoints[12], keypoints[14]
                    )
                
                # 목 각도 (어깨와 코 기준)
                if all(len(keypoints[i]) >= 2 for i in [0, 5, 6]):
                    angles["neck"] = calculate_angle_3points(
                        keypoints[0], keypoints[5], keypoints[6]
                    )
            
            return angles
            
        except Exception as e:
            logger.error(f"❌ 관절 각도 계산 실패: {e}")
            return {}
    
    @staticmethod
    def calculate_body_proportions(keypoints: List[List[float]]) -> Dict[str, float]:
        """신체 비율 계산"""
        try:
            proportions = {}
            
            def calculate_distance(p1, p2):
                """두 점 간의 거리 계산"""
                if len(p1) < 2 or len(p2) < 2:
                    return 0.0
                return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            
            def calculate_body_part_length(keypoint_indices):
                """신체 부위 길이 계산"""
                if len(keypoint_indices) < 2:
                    return 0.0
                
                total_length = 0.0
                for i in range(len(keypoint_indices) - 1):
                    idx1, idx2 = keypoint_indices[i], keypoint_indices[i + 1]
                    if idx1 < len(keypoints) and idx2 < len(keypoints):
                        distance = calculate_distance(keypoints[idx1], keypoints[idx2])
                        total_length += distance
                
                return total_length
            
            if len(keypoints) >= 17:
                # 신체 부위 길이 계산
                proportions["head_length"] = calculate_body_part_length([0, 1, 2, 3, 4])
                proportions["torso_length"] = calculate_body_part_length([5, 6, 11, 12])
                proportions["left_arm_length"] = calculate_body_part_length([5, 7, 9])
                proportions["right_arm_length"] = calculate_body_part_length([6, 8, 10])
                proportions["left_leg_length"] = calculate_body_part_length([11, 13, 15])
                proportions["right_leg_length"] = calculate_body_part_length([12, 14, 16])
                
                # 신체 비율 계산
                if proportions["torso_length"] > 0:
                    proportions["head_to_torso_ratio"] = proportions["head_length"] / proportions["torso_length"]
                    proportions["arm_to_torso_ratio"] = (proportions["left_arm_length"] + proportions["right_arm_length"]) / (2 * proportions["torso_length"])
                    proportions["leg_to_torso_ratio"] = (proportions["left_leg_length"] + proportions["right_leg_length"]) / (2 * proportions["torso_length"])
                
                # 좌우 대칭성 계산
                if proportions["left_arm_length"] > 0 and proportions["right_arm_length"] > 0:
                    proportions["arm_symmetry"] = min(proportions["left_arm_length"], proportions["right_arm_length"]) / max(proportions["left_arm_length"], proportions["right_arm_length"])
                
                if proportions["left_leg_length"] > 0 and proportions["right_leg_length"] > 0:
                    proportions["leg_symmetry"] = min(proportions["left_leg_length"], proportions["right_leg_length"]) / max(proportions["left_leg_length"], proportions["right_leg_length"])
                
                # 전체 신체 길이
                proportions["total_height"] = proportions["head_length"] + proportions["torso_length"] + (proportions["left_leg_length"] + proportions["right_leg_length"]) / 2
                
                # 신체 비율 표준화
                if proportions["total_height"] > 0:
                    for key in proportions:
                        if key != "total_height" and "ratio" not in key and "symmetry" not in key:
                            proportions[f"{key}_normalized"] = proportions[key] / proportions["total_height"]
            
            return proportions
            
        except Exception as e:
            logger.error(f"❌ 신체 비율 계산 실패: {e}")
            return {}
    
    def assess_pose_quality(self, 
                          keypoints: List[List[float]], 
                          joint_angles: Dict[str, float], 
                          body_proportions: Dict[str, float]) -> Dict[str, Any]:
        """포즈 품질 평가"""
        try:
            quality_scores = {}
            
            # 1. 해부학적 타당성 평가
            anatomical_score = self._assess_anatomical_plausibility(keypoints, joint_angles)
            quality_scores["anatomical_plausibility"] = anatomical_score
            
            # 2. 신체 대칭성 평가
            symmetry_score = self._assess_body_symmetry(keypoints, body_proportions)
            quality_scores["body_symmetry"] = symmetry_score
            
            # 3. 관절 각도 정상성 평가
            joint_score = self._validate_joint_angles(joint_angles)
            quality_scores["joint_angles"] = joint_score
            
            # 4. 신체 비율 정상성 평가
            proportion_score = self._validate_body_proportions(body_proportions)
            quality_scores["body_proportions"] = proportion_score
            
            # 5. 좌우 균형 평가
            balance_score = self._analyze_left_right_balance(keypoints)
            quality_scores["left_right_balance"] = balance_score
            
            # 6. 자세 정렬 평가
            alignment_score = self._analyze_posture_alignment(keypoints)
            quality_scores["posture_alignment"] = alignment_score
            
            # 7. 가상 피팅 적합성 평가
            fitting_score = self._assess_virtual_fitting_suitability(keypoints, joint_angles, body_proportions)
            quality_scores["virtual_fitting_suitability"] = fitting_score
            
            # 종합 품질 점수 계산
            overall_score = sum(quality_scores.values()) / len(quality_scores)
            quality_scores["overall_quality"] = overall_score
            
            # 품질 등급 결정
            quality_grade = self._determine_quality_grade(overall_score)
            quality_scores["quality_grade"] = quality_grade
            
            # 문제점 식별
            issues = self._identify_pose_issues(keypoints, joint_angles, body_proportions, quality_scores)
            quality_scores["issues"] = issues
            
            # 개선 권장사항
            recommendations = self._generate_pose_recommendations(issues, quality_scores)
            quality_scores["recommendations"] = recommendations
            
            return quality_scores
            
        except Exception as e:
            logger.error(f"❌ 포즈 품질 평가 실패: {e}")
            return {"error": str(e)}
    
    def _assess_anatomical_plausibility(self, keypoints: List[List[float]], joint_angles: Dict[str, float]) -> float:
        """해부학적 타당성 평가"""
        try:
            score = 0.0
            total_checks = 0
            
            # 관절 각도 범위 검사
            angle_ranges = {
                "left_elbow": (0, 180),
                "right_elbow": (0, 180),
                "left_knee": (0, 180),
                "right_knee": (0, 180),
                "left_hip": (0, 180),
                "right_hip": (0, 180)
            }
            
            for joint, (min_angle, max_angle) in angle_ranges.items():
                if joint in joint_angles:
                    angle = joint_angles[joint]
                    if min_angle <= angle <= max_angle:
                        score += 1.0
                    else:
                        # 범위를 벗어나면 페널티
                        penalty = min(abs(angle - min_angle), abs(angle - max_angle)) / 90.0
                        score += max(0, 1.0 - penalty)
                    total_checks += 1
            
            # 키포인트 간 거리 검사
            if len(keypoints) >= 17:
                # 어깨 너비 검사
                shoulder_width = np.sqrt((keypoints[5][0] - keypoints[6][0])**2 + (keypoints[5][1] - keypoints[6][1])**2)
                if 0.1 < shoulder_width < 0.5:  # 정규화된 좌표 기준
                    score += 1.0
                total_checks += 1
                
                # 팔 길이 검사
                left_arm_length = np.sqrt((keypoints[5][0] - keypoints[9][0])**2 + (keypoints[5][1] - keypoints[9][1])**2)
                right_arm_length = np.sqrt((keypoints[6][0] - keypoints[10][0])**2 + (keypoints[6][1] - keypoints[10][1])**2)
                
                if 0.2 < left_arm_length < 0.8 and 0.2 < right_arm_length < 0.8:
                    score += 1.0
                total_checks += 1
            
            return score / total_checks if total_checks > 0 else 0.0
            
        except Exception as e:
            logger.error(f"❌ 해부학적 타당성 평가 실패: {e}")
            return 0.0
    
    def _assess_body_symmetry(self, keypoints: List[List[float]], body_proportions: Dict[str, float]) -> float:
        """신체 대칭성 평가"""
        try:
            score = 0.0
            total_checks = 0
            
            # 좌우 팔 길이 대칭성
            if "arm_symmetry" in body_proportions:
                score += body_proportions["arm_symmetry"]
                total_checks += 1
            
            # 좌우 다리 길이 대칭성
            if "leg_symmetry" in body_proportions:
                score += body_proportions["leg_symmetry"]
                total_checks += 1
            
            # 키포인트 좌우 대칭성
            if len(keypoints) >= 17:
                # 어깨 높이 대칭성
                shoulder_height_diff = abs(keypoints[5][1] - keypoints[6][1])
                max_height = max(keypoints[5][1], keypoints[6][1])
                if max_height > 0:
                    shoulder_symmetry = 1.0 - (shoulder_height_diff / max_height)
                    score += shoulder_symmetry
                    total_checks += 1
                
                # 엉덩이 높이 대칭성
                hip_height_diff = abs(keypoints[11][1] - keypoints[12][1])
                max_hip_height = max(keypoints[11][1], keypoints[12][1])
                if max_hip_height > 0:
                    hip_symmetry = 1.0 - (hip_height_diff / max_hip_height)
                    score += hip_symmetry
                    total_checks += 1
            
            return score / total_checks if total_checks > 0 else 0.0
            
        except Exception as e:
            logger.error(f"❌ 신체 대칭성 평가 실패: {e}")
            return 0.0
    
    def _point_to_line_distance(self, point, line_start, line_end):
        """점에서 선까지의 거리 계산"""
        try:
            if len(point) < 2 or len(line_start) < 2 or len(line_end) < 2:
                return 0.0
            
            # 선의 방향 벡터
            line_vec = np.array([line_end[0] - line_start[0], line_end[1] - line_start[1]])
            point_vec = np.array([point[0] - line_start[0], point[1] - line_start[1]])
            
            # 선의 길이
            line_length = np.linalg.norm(line_vec)
            if line_length == 0:
                return np.linalg.norm(point_vec)
            
            # 점에서 선까지의 거리
            t = np.dot(point_vec, line_vec) / (line_length ** 2)
            t = np.clip(t, 0, 1)
            
            projection = line_start + t * line_vec
            distance = np.linalg.norm(np.array(point) - projection)
            
            return distance
            
        except Exception as e:
            logger.error(f"❌ 점-선 거리 계산 실패: {e}")
            return 0.0
    
    def _validate_joint_angles(self, joint_angles: Dict[str, float]) -> Dict[str, bool]:
        """관절 각도 검증"""
        try:
            valid_angles = {}
            
            # 정상 각도 범위 정의
            normal_ranges = {
                "left_elbow": (60, 150),
                "right_elbow": (60, 150),
                "left_knee": (120, 180),
                "right_knee": (120, 180),
                "left_hip": (0, 120),
                "right_hip": (0, 120),
                "left_shoulder": (0, 180),
                "right_shoulder": (0, 180)
            }
            
            for joint, angle in joint_angles.items():
                if joint in normal_ranges:
                    min_angle, max_angle = normal_ranges[joint]
                    valid_angles[joint] = min_angle <= angle <= max_angle
                else:
                    valid_angles[joint] = True  # 알 수 없는 관절은 유효로 간주
            
            return valid_angles
            
        except Exception as e:
            logger.error(f"❌ 관절 각도 검증 실패: {e}")
            return {}
    
    def _validate_body_proportions(self, body_proportions: Dict[str, float]) -> Dict[str, Any]:
        """신체 비율 검증"""
        try:
            validation_results = {}
            
            # 정상 비율 범위 정의
            normal_ranges = {
                "head_to_torso_ratio": (0.2, 0.4),
                "arm_to_torso_ratio": (0.6, 1.2),
                "leg_to_torso_ratio": (1.0, 1.8),
                "arm_symmetry": (0.8, 1.0),
                "leg_symmetry": (0.8, 1.0)
            }
            
            for proportion, value in body_proportions.items():
                if proportion in normal_ranges:
                    min_val, max_val = normal_ranges[proportion]
                    validation_results[proportion] = {
                        "value": value,
                        "is_normal": min_val <= value <= max_val,
                        "normal_range": (min_val, max_val)
                    }
                else:
                    validation_results[proportion] = {
                        "value": value,
                        "is_normal": True,  # 알 수 없는 비율은 정상으로 간주
                        "normal_range": None
                    }
            
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ 신체 비율 검증 실패: {e}")
            return {}
    
    def _analyze_left_right_balance(self, keypoints: List[List[float]]) -> Dict[str, Any]:
        """좌우 균형 분석"""
        try:
            balance_analysis = {}
            
            if len(keypoints) >= 17:
                # 어깨 균형
                left_shoulder = keypoints[5]
                right_shoulder = keypoints[6]
                shoulder_balance = abs(left_shoulder[1] - right_shoulder[1])
                balance_analysis["shoulder_balance"] = 1.0 - min(shoulder_balance, 0.1) / 0.1
                
                # 엉덩이 균형
                left_hip = keypoints[11]
                right_hip = keypoints[12]
                hip_balance = abs(left_hip[1] - right_hip[1])
                balance_analysis["hip_balance"] = 1.0 - min(hip_balance, 0.1) / 0.1
                
                # 무릎 균형
                left_knee = keypoints[13]
                right_knee = keypoints[14]
                knee_balance = abs(left_knee[1] - right_knee[1])
                balance_analysis["knee_balance"] = 1.0 - min(knee_balance, 0.1) / 0.1
                
                # 전체 균형 점수
                balance_scores = list(balance_analysis.values())
                balance_analysis["overall_balance"] = sum(balance_scores) / len(balance_scores)
            
            return balance_analysis
            
        except Exception as e:
            logger.error(f"❌ 좌우 균형 분석 실패: {e}")
            return {}
    
    def _analyze_posture_alignment(self, keypoints: List[List[float]]) -> Dict[str, Any]:
        """자세 정렬 분석"""
        try:
            alignment_analysis = {}
            
            if len(keypoints) >= 17:
                # 척추 정렬 (어깨-엉덩이-무릎)
                shoulder_center = [(keypoints[5][0] + keypoints[6][0]) / 2, (keypoints[5][1] + keypoints[6][1]) / 2]
                hip_center = [(keypoints[11][0] + keypoints[12][0]) / 2, (keypoints[11][1] + keypoints[12][1]) / 2]
                knee_center = [(keypoints[13][0] + keypoints[14][0]) / 2, (keypoints[13][1] + keypoints[14][1]) / 2]
                
                # 척추 직선성 계산
                spine_deviation = self._point_to_line_distance(hip_center, shoulder_center, knee_center)
                alignment_analysis["spine_alignment"] = 1.0 - min(spine_deviation, 0.1) / 0.1
                
                # 머리 정렬 (코-어깨 중심)
                head_alignment = self._point_to_line_distance(keypoints[0], shoulder_center, hip_center)
                alignment_analysis["head_alignment"] = 1.0 - min(head_alignment, 0.1) / 0.1
                
                # 전체 정렬 점수
                alignment_scores = list(alignment_analysis.values())
                alignment_analysis["overall_alignment"] = sum(alignment_scores) / len(alignment_scores)
            
            return alignment_analysis
            
        except Exception as e:
            logger.error(f"❌ 자세 정렬 분석 실패: {e}")
            return {}
    
    def _assess_virtual_fitting_suitability(self, keypoints: List[List[float]], joint_angles: Dict[str, float], body_proportions: Dict[str, float]) -> float:
        """가상 피팅 적합성 평가"""
        try:
            suitability_score = 0.0
            total_factors = 0
            
            # 1. 신체 비율 적합성
            if "arm_to_torso_ratio" in body_proportions:
                ratio = body_proportions["arm_to_torso_ratio"]
                if 0.7 <= ratio <= 1.1:  # 의류 피팅에 적합한 범위
                    suitability_score += 1.0
                else:
                    suitability_score += max(0, 1.0 - abs(ratio - 0.9) / 0.4)
                total_factors += 1
            
            # 2. 자세 안정성
            if "overall_balance" in self._analyze_left_right_balance(keypoints):
                balance = self._analyze_left_right_balance(keypoints)["overall_balance"]
                suitability_score += balance
                total_factors += 1
            
            # 3. 관절 각도 적합성
            suitable_angles = 0
            total_angles = 0
            for joint, angle in joint_angles.items():
                if "elbow" in joint and 60 <= angle <= 150:
                    suitable_angles += 1
                elif "knee" in joint and 120 <= angle <= 180:
                    suitable_angles += 1
                elif "hip" in joint and 0 <= angle <= 120:
                    suitable_angles += 1
                total_angles += 1
            
            if total_angles > 0:
                angle_suitability = suitable_angles / total_angles
                suitability_score += angle_suitability
                total_factors += 1
            
            return suitability_score / total_factors if total_factors > 0 else 0.0
            
        except Exception as e:
            logger.error(f"❌ 가상 피팅 적합성 평가 실패: {e}")
            return 0.0
    
    def _identify_pose_issues(self, 
                            keypoints: List[List[float]], 
                            joint_angles: Dict[str, float], 
                            body_proportions: Dict[str, float],
                            scores: Dict[str, float]) -> List[str]:
        """포즈 문제점 식별"""
        try:
            issues = []
            
            # 해부학적 타당성 문제
            if scores.get("anatomical_plausibility", 1.0) < 0.7:
                issues.append("해부학적으로 부자연스러운 자세")
            
            # 대칭성 문제
            if scores.get("body_symmetry", 1.0) < 0.8:
                issues.append("신체 좌우 불균형")
            
            # 관절 각도 문제
            for joint, angle in joint_angles.items():
                if "elbow" in joint and not (60 <= angle <= 150):
                    issues.append(f"{joint} 각도 비정상")
                elif "knee" in joint and not (120 <= angle <= 180):
                    issues.append(f"{joint} 각도 비정상")
                elif "hip" in joint and not (0 <= angle <= 120):
                    issues.append(f"{joint} 각도 비정상")
            
            # 신체 비율 문제
            if "arm_to_torso_ratio" in body_proportions:
                ratio = body_proportions["arm_to_torso_ratio"]
                if ratio < 0.6:
                    issues.append("팔이 너무 짧음")
                elif ratio > 1.2:
                    issues.append("팔이 너무 김")
            
            # 균형 문제
            if scores.get("left_right_balance", 1.0) < 0.8:
                issues.append("좌우 균형 불량")
            
            # 정렬 문제
            if scores.get("posture_alignment", 1.0) < 0.8:
                issues.append("자세 정렬 불량")
            
            return issues
            
        except Exception as e:
            logger.error(f"❌ 포즈 문제점 식별 실패: {e}")
            return ["분석 중 오류 발생"]
    
    def _generate_pose_recommendations(self, issues: List[str], scores: Dict[str, float]) -> List[str]:
        """포즈 개선 권장사항 생성"""
        try:
            recommendations = []
            
            # 일반적인 개선 권장사항
            if scores.get("overall_quality", 1.0) < 0.7:
                recommendations.append("전반적인 자세 개선이 필요합니다")
            
            # 구체적인 문제별 권장사항
            for issue in issues:
                if "해부학적" in issue:
                    recommendations.append("자연스러운 자세로 조정하세요")
                elif "불균형" in issue:
                    recommendations.append("좌우 균형을 맞추세요")
                elif "각도" in issue:
                    recommendations.append("관절 각도를 정상 범위로 조정하세요")
                elif "팔이 너무 짧음" in issue:
                    recommendations.append("팔을 더 펴서 촬영하세요")
                elif "팔이 너무 김" in issue:
                    recommendations.append("팔을 몸에 더 가깝게 위치시키세요")
                elif "균형" in issue:
                    recommendations.append("몸의 중심을 맞추세요")
                elif "정렬" in issue:
                    recommendations.append("척추를 곧게 펴고 머리를 바르게 세우세요")
            
            # 가상 피팅 최적화 권장사항
            if scores.get("virtual_fitting_suitability", 1.0) < 0.8:
                recommendations.append("의류 피팅을 위해 정면을 향한 자세로 촬영하세요")
            
            # 기본 권장사항
            if not recommendations:
                recommendations.append("현재 자세가 양호합니다")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ 포즈 권장사항 생성 실패: {e}")
            return ["분석 중 오류가 발생했습니다"]
    
    def _determine_quality_grade(self, overall_score: float) -> PoseQuality:
        """품질 등급 결정"""
        try:
            if overall_score >= 0.9:
                return PoseQuality.EXCELLENT
            elif overall_score >= 0.75:
                return PoseQuality.GOOD
            elif overall_score >= 0.6:
                return PoseQuality.ACCEPTABLE
            elif overall_score >= 0.4:
                return PoseQuality.POOR
            else:
                return PoseQuality.VERY_POOR
                
        except Exception as e:
            logger.error(f"❌ 품질 등급 결정 실패: {e}")
            return PoseQuality.POOR

# 모듈 내보내기
__all__ = ["PoseAnalyzer", "PoseQuality"]
