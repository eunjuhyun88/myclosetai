# backend/app/services/human_analysis.py
"""
인체 분석 및 분할 시스템
- 인체 부위별 세그멘테이션
- 3D 포즈 추정
- 체형 분석
- 의류 착용 영역 계산
"""

import cv2
import numpy as np
import torch
import mediapipe as mp
from PIL import Image, ImageDraw
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class HumanBodyAnalyzer:
    """인체 분석 및 분할"""
    
    def __init__(self):
        # MediaPipe 초기화
        self.mp_pose = mp.solutions.pose
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 모델 초기화
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        
        self.segmentation_model = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )
        
        # 인체 부위 정의
        self.body_parts = {
            'head': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 얼굴, 머리
            'torso': [11, 12, 23, 24],  # 어깨, 엉덩이
            'left_arm': [11, 13, 15, 17, 19, 21],  # 왼팔
            'right_arm': [12, 14, 16, 18, 20, 22],  # 오른팔
            'left_leg': [23, 25, 27, 29, 31],  # 왼다리
            'right_leg': [24, 26, 28, 30, 32]  # 오른다리
        }
        
    async def analyze_human_body(self, image: Image.Image) -> Dict[str, Any]:
        """종합적인 인체 분석"""
        
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 1. 포즈 검출
        pose_results = self.detect_pose(cv_image)
        
        # 2. 인체 세그멘테이션
        segmentation_mask = self.segment_human(cv_image)
        
        # 3. 인체 부위별 분할
        body_parts_map = self.parse_body_parts(cv_image, pose_results)
        
        # 4. 3D 포즈 추정
        pose_3d = self.estimate_3d_pose(pose_results)
        
        # 5. 체형 분석
        body_measurements = self.analyze_body_measurements(pose_results, image.size)
        
        # 6. 의류 착용 영역 계산
        clothing_regions = self.calculate_clothing_regions(body_parts_map, pose_results)
        
        return {
            'pose_landmarks': pose_results,
            'segmentation_mask': segmentation_mask,
            'body_parts_map': body_parts_map,
            'pose_3d': pose_3d,
            'body_measurements': body_measurements,
            'clothing_regions': clothing_regions,
            'confidence': self._calculate_confidence(pose_results)
        }
    
    def detect_pose(self, cv_image: np.ndarray) -> Dict[str, Any]:
        """포즈 검출"""
        try:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(rgb_image)
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                
                return {
                    'landmarks': landmarks,
                    'segmentation_mask': results.segmentation_mask,
                    'detected': True
                }
            else:
                logger.warning("포즈 검출 실패")
                return {'detected': False}
                
        except Exception as e:
            logger.error(f"포즈 검출 오류: {e}")
            return {'detected': False}
    
    def segment_human(self, cv_image: np.ndarray) -> np.ndarray:
        """인체 세그멘테이션"""
        try:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            results = self.segmentation_model.process(rgb_image)
            
            # 세그멘테이션 마스크를 0-255 범위로 변환
            mask = (results.segmentation_mask * 255).astype(np.uint8)
            
            return mask
            
        except Exception as e:
            logger.error(f"인체 세그멘테이션 오류: {e}")
            return np.zeros((cv_image.shape[0], cv_image.shape[1]), dtype=np.uint8)
    
    def parse_body_parts(self, cv_image: np.ndarray, pose_results: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """인체 부위별 분할"""
        height, width = cv_image.shape[:2]
        body_parts_map = {}
        
        if not pose_results['detected']:
            return body_parts_map
        
        landmarks = pose_results['landmarks']
        
        for part_name, landmark_indices in self.body_parts.items():
            # 각 부위별 마스크 생성
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # 랜드마크 좌표 추출
            points = []
            for idx in landmark_indices:
                if idx < len(landmarks) and landmarks[idx]['visibility'] > 0.5:
                    x = int(landmarks[idx]['x'] * width)
                    y = int(landmarks[idx]['y'] * height)
                    points.append([x, y])
            
            if len(points) >= 3:
                # 볼록 껍질 생성
                hull = cv2.convexHull(np.array(points))
                cv2.fillPoly(mask, [hull], 255)
                
                body_parts_map[part_name] = mask
        
        return body_parts_map
    
    def estimate_3d_pose(self, pose_results: Dict[str, Any]) -> Dict[str, Any]:
        """3D 포즈 추정"""
        if not pose_results['detected']:
            return {}
        
        landmarks = pose_results['landmarks']
        
        # 3D 좌표 추출
        pose_3d = {
            'world_landmarks': [],
            'angles': {},
            'skeleton_structure': []
        }
        
        # 주요 관절 각도 계산
        pose_3d['angles'] = self._calculate_joint_angles(landmarks)
        
        # 골격 구조 정의
        pose_3d['skeleton_structure'] = self._build_skeleton_structure(landmarks)
        
        return pose_3d
    
    def analyze_body_measurements(self, pose_results: Dict[str, Any], image_size: Tuple[int, int]) -> Dict[str, float]:
        """체형 분석 및 치수 계산"""
        if not pose_results['detected']:
            return {}
        
        landmarks = pose_results['landmarks']
        width, height = image_size
        
        measurements = {}
        
        try:
            # 어깨 너비
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            shoulder_width = abs(left_shoulder['x'] - right_shoulder['x']) * width
            measurements['shoulder_width'] = shoulder_width
            
            # 몸통 길이
            shoulder_center_y = (left_shoulder['y'] + right_shoulder['y']) / 2
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            hip_center_y = (left_hip['y'] + right_hip['y']) / 2
            torso_length = abs(hip_center_y - shoulder_center_y) * height
            measurements['torso_length'] = torso_length
            
            # 엉덩이 너비
            hip_width = abs(left_hip['x'] - right_hip['x']) * width
            measurements['hip_width'] = hip_width
            
            # 팔 길이
            left_elbow = landmarks[13]
            left_wrist = landmarks[15]
            arm_length = np.sqrt(
                ((left_shoulder['x'] - left_elbow['x']) * width) ** 2 +
                ((left_shoulder['y'] - left_elbow['y']) * height) ** 2
            ) + np.sqrt(
                ((left_elbow['x'] - left_wrist['x']) * width) ** 2 +
                ((left_elbow['y'] - left_wrist['y']) * height) ** 2
            )
            measurements['arm_length'] = arm_length
            
            # 다리 길이
            left_knee = landmarks[25]
            left_ankle = landmarks[27]
            leg_length = np.sqrt(
                ((left_hip['x'] - left_knee['x']) * width) ** 2 +
                ((left_hip['y'] - left_knee['y']) * height) ** 2
            ) + np.sqrt(
                ((left_knee['x'] - left_ankle['x']) * width) ** 2 +
                ((left_knee['y'] - left_ankle['y']) * height) ** 2
            )
            measurements['leg_length'] = leg_length
            
        except Exception as e:
            logger.error(f"체형 분석 오류: {e}")
        
        return measurements
    
    def calculate_clothing_regions(self, body_parts_map: Dict[str, np.ndarray], pose_results: Dict[str, Any]) -> Dict[str, Dict]:
        """의류 착용 영역 계산"""
        clothing_regions = {}
        
        if not pose_results['detected']:
            return clothing_regions
        
        # 상의 영역 (torso + 팔 일부)
        if 'torso' in body_parts_map:
            torso_mask = body_parts_map['torso']
            
            # 상의 영역 확장 (어깨, 가슴 부분)
            upper_region = self._expand_upper_clothing_region(torso_mask, pose_results)
            
            clothing_regions['upper_body'] = {
                'mask': upper_region,
                'bounds': self._get_mask_bounds(upper_region),
                'area': np.sum(upper_region > 0)
            }
        
        # 하의 영역 (엉덩이 + 다리 상부)
        if 'left_leg' in body_parts_map and 'right_leg' in body_parts_map:
            leg_mask = cv2.bitwise_or(body_parts_map['left_leg'], body_parts_map['right_leg'])
            
            # 하의 영역 조정
            lower_region = self._adjust_lower_clothing_region(leg_mask, pose_results)
            
            clothing_regions['lower_body'] = {
                'mask': lower_region,
                'bounds': self._get_mask_bounds(lower_region),
                'area': np.sum(lower_region > 0)
            }
        
        return clothing_regions
    
    def _expand_upper_clothing_region(self, torso_mask: np.ndarray, pose_results: Dict[str, Any]) -> np.ndarray:
        """상의 영역 확장"""
        # 팽창 연산으로 영역 확장
        kernel = np.ones((15, 15), np.uint8)
        expanded = cv2.dilate(torso_mask, kernel, iterations=2)
        
        return expanded
    
    def _adjust_lower_clothing_region(self, leg_mask: np.ndarray, pose_results: Dict[str, Any]) -> np.ndarray:
        """하의 영역 조정"""
        # 상부만 선택 (바지 영역)
        height = leg_mask.shape[0]
        
        # 하의는 다리 상부 70%만 사용
        lower_region = leg_mask.copy()
        cut_line = int(height * 0.7)
        lower_region[cut_line:, :] = 0
        
        return lower_region
    
    def _get_mask_bounds(self, mask: np.ndarray) -> Dict[str, int]:
        """마스크 경계 박스 계산"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            return {'x': x, 'y': y, 'width': w, 'height': h}
        
        return {'x': 0, 'y': 0, 'width': 0, 'height': 0}
    
    def _calculate_joint_angles(self, landmarks: List[Dict]) -> Dict[str, float]:
        """관절 각도 계산"""
        angles = {}
        
        try:
            # 팔꿈치 각도 (왼쪽)
            shoulder = np.array([landmarks[11]['x'], landmarks[11]['y']])
            elbow = np.array([landmarks[13]['x'], landmarks[13]['y']])
            wrist = np.array([landmarks[15]['x'], landmarks[15]['y']])
            
            angles['left_elbow'] = self._calculate_angle(shoulder, elbow, wrist)
            
            # 무릎 각도 (왼쪽)
            hip = np.array([landmarks[23]['x'], landmarks[23]['y']])
            knee = np.array([landmarks[25]['x'], landmarks[25]['y']])
            ankle = np.array([landmarks[27]['x'], landmarks[27]['y']])
            
            angles['left_knee'] = self._calculate_angle(hip, knee, ankle)
            
        except Exception as e:
            logger.error(f"관절 각도 계산 오류: {e}")
        
        return angles
    
    def _calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """세 점으로 각도 계산"""
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _build_skeleton_structure(self, landmarks: List[Dict]) -> List[Tuple[int, int]]:
        """골격 구조 정의"""
        # MediaPipe 포즈 연결선 정의
        connections = [
            (11, 12),  # 어깨
            (11, 13), (13, 15),  # 왼팔
            (12, 14), (14, 16),  # 오른팔
            (11, 23), (12, 24),  # 몸통
            (23, 24),  # 엉덩이
            (23, 25), (25, 27),  # 왼다리
            (24, 26), (26, 28),  # 오른다리
        ]
        
        return connections
    
    def _calculate_confidence(self, pose_results: Dict[str, Any]) -> float:
        """전체 신뢰도 계산"""
        if not pose_results['detected']:
            return 0.0
        
        landmarks = pose_results['landmarks']
        visibilities = [lm['visibility'] for lm in landmarks if 'visibility' in lm]
        
        if visibilities:
            return np.mean(visibilities)
        
        return 0.0
    
    def visualize_analysis(self, image: Image.Image, analysis_result: Dict[str, Any]) -> Image.Image:
        """분석 결과 시각화"""
        
        # 원본 이미지 복사
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        # 포즈 랜드마크 그리기
        if analysis_result['pose_landmarks']['detected']:
            landmarks = analysis_result['pose_landmarks']['landmarks']
            width, height = image.size
            
            # 랜드마크 점 그리기
            for landmark in landmarks:
                x = int(landmark['x'] * width)
                y = int(landmark['y'] * height)
                if landmark['visibility'] > 0.5:
                    draw.ellipse([x-3, y-3, x+3, y+3], fill='red')
            
            # 골격 연결선 그리기
            connections = analysis_result['pose_3d']['skeleton_structure']
            for connection in connections:
                if (connection[0] < len(landmarks) and 
                    connection[1] < len(landmarks) and
                    landmarks[connection[0]]['visibility'] > 0.5 and
                    landmarks[connection[1]]['visibility'] > 0.5):
                    
                    x1 = int(landmarks[connection[0]]['x'] * width)
                    y1 = int(landmarks[connection[0]]['y'] * height)
                    x2 = int(landmarks[connection[1]]['x'] * width)
                    y2 = int(landmarks[connection[1]]['y'] * height)
                    
                    draw.line([x1, y1, x2, y2], fill='blue', width=2)
        
        return vis_image

# 전역 인체 분석기
human_analyzer = HumanBodyAnalyzer()