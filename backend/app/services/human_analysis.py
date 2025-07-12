# backend/app/services/human_analysis.py
"""
MyCloset AI 인체 분석 및 분할 시스템
- 인체 부위별 세그멘테이션
- 3D 포즈 추정
- 체형 분석
- 의류 착용 영역 계산
"""

import cv2
import numpy as np
import torch
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Any, Optional
import logging
import os
import json
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BodyMeasurement:
    """체형 측정 데이터"""
    shoulder_width: float
    torso_length: float
    hip_width: float
    arm_length: float
    leg_length: float
    confidence: float

@dataclass
class ClothingRegion:
    """의류 착용 영역"""
    mask: np.ndarray
    bounds: Dict[str, int]
    area: int
    center: Tuple[int, int]

class HumanBodyAnalyzer:
    """인체 분석 및 분할 전문 클래스"""
    
    def __init__(self, use_gpu: bool = True):
        """
        초기화
        Args:
            use_gpu: GPU 사용 여부
        """
        self.device = self._setup_device(use_gpu)
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # MediaPipe 모델 초기화
        try:
            self._init_mediapipe_models()
            logger.info("MediaPipe 모델 초기화 완료")
        except Exception as e:
            logger.error(f"MediaPipe 초기화 실패: {e}")
            raise
        
        # 인체 부위 매핑
        self.body_parts_mapping = self._init_body_parts()
        
        # 의류 카테고리 정의
        self.clothing_categories = {
            'upper_body': ['상의', '아우터', '원피스'],
            'lower_body': ['하의', '스커트', '반바지'],
            'full_body': ['원피스', '점프수트']
        }
    
    def _setup_device(self, use_gpu: bool) -> str:
        """디바이스 설정"""
        if use_gpu and torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA GPU 사용: {torch.cuda.get_device_name()}")
        elif use_gpu and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Apple Silicon GPU (MPS) 사용")
        else:
            device = "cpu"
            logger.info("CPU 사용")
        
        return device
    
    def _init_mediapipe_models(self):
        """MediaPipe 모델 초기화"""
        # Pose 검출 모델
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # 셀피 세그멘테이션 모델
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmentation_model = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1  # 고정밀 모델
        )
        
        # 손 검출 모델 (필요시)
        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        
        # 그리기 유틸리티
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def _init_body_parts(self) -> Dict[str, List[int]]:
        """인체 부위별 랜드마크 매핑"""
        return {
            'head': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 얼굴과 머리
            'neck': [11, 12],  # 목 (어깨 포인트 사용)
            'torso': [11, 12, 23, 24],  # 몸통 (어깨-엉덩이)
            'left_arm': [11, 13, 15, 17, 19, 21],  # 왼팔 전체
            'right_arm': [12, 14, 16, 18, 20, 22],  # 오른팔 전체
            'left_leg': [23, 25, 27, 29, 31],  # 왼다리 전체
            'right_leg': [24, 26, 28, 30, 32],  # 오른다리 전체
            'core_points': [11, 12, 23, 24]  # 핵심 포인트
        }
    
    async def analyze_human_body(self, image: Image.Image, 
                                analysis_options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        종합적인 인체 분석 수행
        
        Args:
            image: 분석할 이미지
            analysis_options: 분석 옵션
            
        Returns:
            분석 결과 딕셔너리
        """
        start_time = time.time()
        
        # 기본 옵션 설정
        options = analysis_options or {
            'enable_pose': True,
            'enable_segmentation': True,
            'enable_measurements': True,
            'enable_clothing_regions': True,
            'visualization': False
        }
        
        try:
            # 이미지 전처리
            processed_image = self._preprocess_image(image)
            cv_image = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
            
            # 병렬 분석 실행
            analysis_tasks = []
            
            if options.get('enable_pose', True):
                analysis_tasks.append(self._analyze_pose_async(cv_image))
            
            if options.get('enable_segmentation', True):
                analysis_tasks.append(self._analyze_segmentation_async(cv_image))
            
            # 분석 결과 수집
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # 결과 처리
            pose_result = results[0] if len(results) > 0 else {}
            segmentation_result = results[1] if len(results) > 1 else {}
            
            # 추가 분석 수행
            analysis_result = {
                'timestamp': time.time(),
                'processing_time': 0,
                'image_info': {
                    'width': image.width,
                    'height': image.height,
                    'format': image.format
                },
                'pose_analysis': pose_result,
                'segmentation': segmentation_result,
                'body_measurements': {},
                'clothing_regions': {},
                'confidence_score': 0.0,
                'status': 'success'
            }
            
            # 포즈가 성공적으로 분석된 경우 추가 처리
            if pose_result.get('detected', False):
                
                # 체형 측정
                if options.get('enable_measurements', True):
                    measurements = self._calculate_body_measurements(
                        pose_result['landmarks'], 
                        (image.width, image.height)
                    )
                    analysis_result['body_measurements'] = measurements.__dict__
                
                # 의류 착용 영역 계산
                if options.get('enable_clothing_regions', True):
                    clothing_regions = self._calculate_clothing_regions(
                        pose_result, 
                        cv_image.shape[:2]
                    )
                    analysis_result['clothing_regions'] = clothing_regions
                
                # 신뢰도 계산
                analysis_result['confidence_score'] = self._calculate_overall_confidence(
                    pose_result, segmentation_result
                )
            
            analysis_result['processing_time'] = time.time() - start_time
            
            # 시각화 (옵션)
            if options.get('visualization', False):
                visualization = self.create_visualization(image, analysis_result)
                analysis_result['visualization'] = visualization
            
            logger.info(f"인체 분석 완료: {analysis_result['processing_time']:.2f}초")
            return analysis_result
            
        except Exception as e:
            logger.error(f"인체 분석 오류: {e}")
            return {
                'status': 'error',
                'error_message': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """이미지 전처리"""
        # 크기 정규화 (최대 1024x1024)
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # RGB 변환
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    async def _analyze_pose_async(self, cv_image: np.ndarray) -> Dict[str, Any]:
        """비동기 포즈 분석"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self._detect_pose, 
            cv_image
        )
    
    async def _analyze_segmentation_async(self, cv_image: np.ndarray) -> Dict[str, Any]:
        """비동기 세그멘테이션 분석"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self._segment_human, 
            cv_image
        )
    
    def _detect_pose(self, cv_image: np.ndarray) -> Dict[str, Any]:
        """포즈 검출 및 분석"""
        try:
            # RGB 변환
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # 포즈 검출
            results = self.pose_detector.process(rgb_image)
            
            if results.pose_landmarks:
                # 랜드마크 데이터 추출
                landmarks = []
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    landmarks.append({
                        'id': idx,
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                
                # 3D 포즈 정보
                pose_3d = self._estimate_3d_pose(landmarks)
                
                # 인체 부위별 분할
                body_parts = self._parse_body_parts(landmarks, cv_image.shape[:2])
                
                return {
                    'detected': True,
                    'landmarks': landmarks,
                    'pose_3d': pose_3d,
                    'body_parts': body_parts,
                    'world_landmarks': self._extract_world_landmarks(results),
                    'segmentation_mask': self._process_pose_segmentation(results.segmentation_mask)
                }
            else:
                logger.warning("포즈 검출 실패 - 인체를 찾을 수 없음")
                return {'detected': False, 'error': 'No pose detected'}
                
        except Exception as e:
            logger.error(f"포즈 검출 오류: {e}")
            return {'detected': False, 'error': str(e)}
    
    def _segment_human(self, cv_image: np.ndarray) -> Dict[str, Any]:
        """인체 세그멘테이션"""
        try:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            results = self.segmentation_model.process(rgb_image)
            
            if results.segmentation_mask is not None:
                # 마스크 후처리
                mask = (results.segmentation_mask * 255).astype(np.uint8)
                
                # 노이즈 제거
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # 마스크 통계
                total_pixels = mask.shape[0] * mask.shape[1]
                human_pixels = np.sum(mask > 128)
                coverage = human_pixels / total_pixels
                
                return {
                    'success': True,
                    'mask': mask,
                    'coverage': coverage,
                    'human_pixels': int(human_pixels),
                    'total_pixels': int(total_pixels)
                }
            else:
                return {'success': False, 'error': 'Segmentation failed'}
                
        except Exception as e:
            logger.error(f"세그멘테이션 오류: {e}")
            return {'success': False, 'error': str(e)}
    
    def _estimate_3d_pose(self, landmarks: List[Dict]) -> Dict[str, Any]:
        """3D 포즈 추정"""
        pose_3d = {
            'angles': {},
            'skeleton_connections': [],
            'pose_classification': 'unknown'
        }
        
        try:
            # 주요 관절 각도 계산
            angles = {}
            
            # 팔꿈치 각도 (왼쪽)
            if all(idx < len(landmarks) for idx in [11, 13, 15]):
                shoulder = np.array([landmarks[11]['x'], landmarks[11]['y']])
                elbow = np.array([landmarks[13]['x'], landmarks[13]['y']])
                wrist = np.array([landmarks[15]['x'], landmarks[15]['y']])
                angles['left_elbow'] = self._calculate_angle(shoulder, elbow, wrist)
            
            # 팔꿈치 각도 (오른쪽)
            if all(idx < len(landmarks) for idx in [12, 14, 16]):
                shoulder = np.array([landmarks[12]['x'], landmarks[12]['y']])
                elbow = np.array([landmarks[14]['x'], landmarks[14]['y']])
                wrist = np.array([landmarks[16]['x'], landmarks[16]['y']])
                angles['right_elbow'] = self._calculate_angle(shoulder, elbow, wrist)
            
            # 무릎 각도 (왼쪽)
            if all(idx < len(landmarks) for idx in [23, 25, 27]):
                hip = np.array([landmarks[23]['x'], landmarks[23]['y']])
                knee = np.array([landmarks[25]['x'], landmarks[25]['y']])
                ankle = np.array([landmarks[27]['x'], landmarks[27]['y']])
                angles['left_knee'] = self._calculate_angle(hip, knee, ankle)
            
            # 무릎 각도 (오른쪽)
            if all(idx < len(landmarks) for idx in [24, 26, 28]):
                hip = np.array([landmarks[24]['x'], landmarks[24]['y']])
                knee = np.array([landmarks[26]['x'], landmarks[26]['y']])
                ankle = np.array([landmarks[28]['x'], landmarks[28]['y']])
                angles['right_knee'] = self._calculate_angle(hip, knee, ankle)
            
            pose_3d['angles'] = angles
            
            # 골격 연결선
            connections = [
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # 팔
                (11, 23), (12, 24), (23, 24),  # 몸통
                (23, 25), (25, 27), (24, 26), (26, 28)  # 다리
            ]
            pose_3d['skeleton_connections'] = connections
            
            # 자세 분류 (기본적인 분류)
            pose_3d['pose_classification'] = self._classify_pose(landmarks, angles)
            
        except Exception as e:
            logger.error(f"3D 포즈 추정 오류: {e}")
        
        return pose_3d
    
    def _parse_body_parts(self, landmarks: List[Dict], image_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """인체 부위별 분할"""
        height, width = image_shape
        body_parts = {}
        
        for part_name, landmark_indices in self.body_parts_mapping.items():
            try:
                # 마스크 생성
                mask = np.zeros((height, width), dtype=np.uint8)
                
                # 유효한 랜드마크 포인트 수집
                points = []
                for idx in landmark_indices:
                    if idx < len(landmarks) and landmarks[idx]['visibility'] > 0.5:
                        x = int(landmarks[idx]['x'] * width)
                        y = int(landmarks[idx]['y'] * height)
                        # 경계 확인
                        if 0 <= x < width and 0 <= y < height:
                            points.append([x, y])
                
                if len(points) >= 3:
                    # 볼록 껍질 생성 및 채우기
                    points_array = np.array(points, dtype=np.int32)
                    hull = cv2.convexHull(points_array)
                    cv2.fillPoly(mask, [hull], 255)
                    
                    # 부위별 확장 (의류 피팅을 위해)
                    if part_name in ['torso', 'left_arm', 'right_arm']:
                        kernel = np.ones((20, 20), np.uint8)
                        mask = cv2.dilate(mask, kernel, iterations=1)
                    
                    body_parts[part_name] = mask
                
            except Exception as e:
                logger.error(f"부위 분할 오류 ({part_name}): {e}")
        
        return body_parts
    
    def _calculate_body_measurements(self, landmarks: List[Dict], 
                                   image_size: Tuple[int, int]) -> BodyMeasurement:
        """체형 측정"""
        width, height = image_size
        
        try:
            # 어깨 너비
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            shoulder_width = abs(left_shoulder['x'] - right_shoulder['x']) * width
            
            # 몸통 길이
            shoulder_center_y = (left_shoulder['y'] + right_shoulder['y']) / 2
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            hip_center_y = (left_hip['y'] + right_hip['y']) / 2
            torso_length = abs(hip_center_y - shoulder_center_y) * height
            
            # 엉덩이 너비
            hip_width = abs(left_hip['x'] - right_hip['x']) * width
            
            # 팔 길이 (왼팔 기준)
            left_elbow = landmarks[13]
            left_wrist = landmarks[15]
            upper_arm = np.sqrt(
                ((left_shoulder['x'] - left_elbow['x']) * width) ** 2 +
                ((left_shoulder['y'] - left_elbow['y']) * height) ** 2
            )
            lower_arm = np.sqrt(
                ((left_elbow['x'] - left_wrist['x']) * width) ** 2 +
                ((left_elbow['y'] - left_wrist['y']) * height) ** 2
            )
            arm_length = upper_arm + lower_arm
            
            # 다리 길이 (왼다리 기준)
            left_knee = landmarks[25]
            left_ankle = landmarks[27]
            upper_leg = np.sqrt(
                ((left_hip['x'] - left_knee['x']) * width) ** 2 +
                ((left_hip['y'] - left_knee['y']) * height) ** 2
            )
            lower_leg = np.sqrt(
                ((left_knee['x'] - left_ankle['x']) * width) ** 2 +
                ((left_knee['y'] - left_ankle['y']) * height) ** 2
            )
            leg_length = upper_leg + lower_leg
            
            # 측정 신뢰도 계산
            visibilities = [lm['visibility'] for lm in [
                left_shoulder, right_shoulder, left_hip, right_hip,
                left_elbow, left_wrist, left_knee, left_ankle
            ]]
            confidence = np.mean(visibilities)
            
            return BodyMeasurement(
                shoulder_width=shoulder_width,
                torso_length=torso_length,
                hip_width=hip_width,
                arm_length=arm_length,
                leg_length=leg_length,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"체형 측정 오류: {e}")
            return BodyMeasurement(0, 0, 0, 0, 0, 0)
    
    def _calculate_clothing_regions(self, pose_result: Dict[str, Any], 
                                  image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """의류 착용 영역 계산"""
        if not pose_result.get('detected', False):
            return {}
        
        height, width = image_shape
        landmarks = pose_result['landmarks']
        clothing_regions = {}
        
        try:
            # 상의 영역 (어깨, 가슴, 배)
            upper_points = []
            for idx in [11, 12, 23, 24]:  # 어깨와 엉덩이 포인트
                if idx < len(landmarks) and landmarks[idx]['visibility'] > 0.5:
                    x = int(landmarks[idx]['x'] * width)
                    y = int(landmarks[idx]['y'] * height)
                    upper_points.append([x, y])
            
            if len(upper_points) >= 3:
                upper_mask = np.zeros((height, width), dtype=np.uint8)
                hull = cv2.convexHull(np.array(upper_points))
                cv2.fillPoly(upper_mask, [hull], 255)
                
                # 상의 영역 확장
                kernel = np.ones((30, 30), np.uint8)
                upper_mask = cv2.dilate(upper_mask, kernel, iterations=1)
                
                clothing_regions['upper_body'] = {
                    'mask': upper_mask.tolist(),
                    'bounds': self._get_mask_bounds(upper_mask),
                    'area': int(np.sum(upper_mask > 0)),
                    'center': self._get_mask_center(upper_mask)
                }
            
            # 하의 영역 (엉덩이, 허벅지)
            lower_points = []
            for idx in [23, 24, 25, 26]:  # 엉덩이와 무릎 포인트
                if idx < len(landmarks) and landmarks[idx]['visibility'] > 0.5:
                    x = int(landmarks[idx]['x'] * width)
                    y = int(landmarks[idx]['y'] * height)
                    lower_points.append([x, y])
            
            if len(lower_points) >= 3:
                lower_mask = np.zeros((height, width), dtype=np.uint8)
                hull = cv2.convexHull(np.array(lower_points))
                cv2.fillPoly(lower_mask, [hull], 255)
                
                # 하의 영역 조정 (다리 전체가 아닌 상부만)
                hip_y = int((landmarks[23]['y'] + landmarks[24]['y']) / 2 * height)
                knee_y = int((landmarks[25]['y'] + landmarks[26]['y']) / 2 * height) if landmarks[25]['visibility'] > 0.5 and landmarks[26]['visibility'] > 0.5 else hip_y + 100
                
                # 하의 길이 조정
                lower_height = min(int((knee_y - hip_y) * 1.2), height - hip_y)
                lower_mask[hip_y + lower_height:, :] = 0
                
                clothing_regions['lower_body'] = {
                    'mask': lower_mask.tolist(),
                    'bounds': self._get_mask_bounds(lower_mask),
                    'area': int(np.sum(lower_mask > 0)),
                    'center': self._get_mask_center(lower_mask)
                }
            
        except Exception as e:
            logger.error(f"의류 영역 계산 오류: {e}")
        
        return clothing_regions
    
    def _calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """세 점으로 각도 계산"""
        try:
            ba = a - b
            bc = c - b
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            # 수치적 안정성을 위해 클리핑
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            angle = np.arccos(cosine_angle)
            
            return float(np.degrees(angle))
        except:
            return 0.0
    
    def _classify_pose(self, landmarks: List[Dict], angles: Dict[str, float]) -> str:
        """자세 분류"""
        try:
            # 팔 위치 확인
            left_wrist_y = landmarks[15]['y'] if landmarks[15]['visibility'] > 0.5 else 1.0
            right_wrist_y = landmarks[16]['y'] if landmarks[16]['visibility'] > 0.5 else 1.0
            shoulder_y = (landmarks[11]['y'] + landmarks[12]['y']) / 2
            
            # 팔이 올라가 있는지 확인
            arms_raised = left_wrist_y < shoulder_y or right_wrist_y < shoulder_y
            
            # 다리 위치 확인
            leg_spread = abs(landmarks[27]['x'] - landmarks[28]['x']) if landmarks[27]['visibility'] > 0.5 and landmarks[28]['visibility'] > 0.5 else 0
            
            if arms_raised and leg_spread > 0.1:
                return 'T-pose'
            elif arms_raised:
                return 'arms_raised'
            elif leg_spread > 0.15:
                return 'wide_stance'
            else:
                return 'standing'
                
        except:
            return 'unknown'
    
    def _extract_world_landmarks(self, results) -> List[Dict]:
        """월드 좌표 랜드마크 추출"""
        world_landmarks = []
        if hasattr(results, 'pose_world_landmarks') and results.pose_world_landmarks:
            for landmark in results.pose_world_landmarks.landmark:
                world_landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
        return world_landmarks
    
    def _process_pose_segmentation(self, segmentation_mask) -> Optional[np.ndarray]:
        """포즈 세그멘테이션 마스크 처리"""
        if segmentation_mask is not None:
            mask = (segmentation_mask * 255).astype(np.uint8)
            return mask.tolist()
        return None
    
    def _get_mask_bounds(self, mask: np.ndarray) -> Dict[str, int]:
        """마스크 경계 상자 계산"""
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                return {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
        except:
            pass
        return {'x': 0, 'y': 0, 'width': 0, 'height': 0}
    
    def _get_mask_center(self, mask: np.ndarray) -> Tuple[int, int]:
        """마스크 중심점 계산"""
        try:
            M = cv2.moments(mask)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
        except:
            pass
        return (0, 0)
    
    def _calculate_overall_confidence(self, pose_result: Dict, 
                                    segmentation_result: Dict) -> float:
        """전체 신뢰도 계산"""
        confidences = []
        
        # 포즈 신뢰도
        if pose_result.get('detected', False):
            landmarks = pose_result['landmarks']
            pose_confidence = np.mean([lm['visibility'] for lm in landmarks])
            confidences.append(pose_confidence)
        
        # 세그멘테이션 신뢰도
        if segmentation_result.get('success', False):
            coverage = segmentation_result.get('coverage', 0)
            # 커버리지를 신뢰도로 변환 (0.1-0.8 범위를 0-1로 매핑)
            seg_confidence = min(max((coverage - 0.1) / 0.7, 0), 1)
            confidences.append(seg_confidence)
        
        return float(np.mean(confidences)) if confidences else 0.0
    
    def create_visualization(self, original_image: Image.Image, 
                           analysis_result: Dict[str, Any]) -> Image.Image:
        """분석 결과 시각화"""
        
        # 원본 이미지 복사
        vis_image = original_image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        width, height = vis_image.size
        
        # 포즈 랜드마크 시각화
        pose_analysis = analysis_result.get('pose_analysis', {})
        if pose_analysis.get('detected', False):
            landmarks = pose_analysis['landmarks']
            
            # 랜드마크 포인트 그리기
            for landmark in landmarks:
                if landmark['visibility'] > 0.5:
                    x = int(landmark['x'] * width)
                    y = int(landmark['y'] * height)
                    
                    # 점 그리기
                    draw.ellipse([x-4, y-4, x+4, y+4], fill='red', outline='white')
            
            # 골격 연결선 그리기
            pose_3d = pose_analysis.get('pose_3d', {})
            connections = pose_3d.get('skeleton_connections', [])
            
            for connection in connections:
                try:
                    start_idx, end_idx = connection
                    if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                        landmarks[start_idx]['visibility'] > 0.5 and 
                        landmarks[end_idx]['visibility'] > 0.5):
                        
                        x1 = int(landmarks[start_idx]['x'] * width)
                        y1 = int(landmarks[start_idx]['y'] * height)
                        x2 = int(landmarks[end_idx]['x'] * width)
                        y2 = int(landmarks[end_idx]['y'] * height)
                        
                        draw.line([x1, y1, x2, y2], fill='blue', width=3)
                except:
                    continue
        
        # 신뢰도 및 정보 표시
        confidence = analysis_result.get('confidence_score', 0)
        processing_time = analysis_result.get('processing_time', 0)
        
        # 텍스트 정보 추가
        info_text = f"신뢰도: {confidence:.2f} | 처리시간: {processing_time:.2f}s"
        
        try:
            # 폰트 설정 (시스템 기본 폰트 사용)
            draw.text((10, 10), info_text, fill='white', stroke_width=2, stroke_fill='black')
        except:
            # 폰트 로드 실패시 기본 텍스트
            draw.text((10, 10), info_text, fill='white')
        
        return vis_image
    
    def get_fitting_compatibility(self, analysis_result: Dict[str, Any], 
                                clothing_type: str) -> Dict[str, Any]:
        """의류 피팅 호환성 분석"""
        
        compatibility = {
            'score': 0.0,
            'recommended': False,
            'issues': [],
            'suggestions': []
        }
        
        try:
            if not analysis_result.get('pose_analysis', {}).get('detected', False):
                compatibility['issues'].append('포즈 검출 실패')
                return compatibility
            
            confidence = analysis_result.get('confidence_score', 0)
            pose_classification = analysis_result.get('pose_analysis', {}).get('pose_3d', {}).get('pose_classification', 'unknown')
            clothing_regions = analysis_result.get('clothing_regions', {})
            
            # 기본 점수
            score = confidence * 100
            
            # 포즈 적합성 검사
            if pose_classification in ['T-pose', 'standing']:
                score += 20
            elif pose_classification == 'arms_raised':
                score += 10
                compatibility['suggestions'].append('팔을 자연스럽게 내리면 더 좋은 결과를 얻을 수 있습니다')
            else:
                score -= 10
                compatibility['issues'].append('부자연스러운 포즈 감지됨')
            
            # 의류 타입별 검사
            if clothing_type in self.clothing_categories['upper_body']:
                if 'upper_body' in clothing_regions:
                    region_area = clothing_regions['upper_body']['area']
                    if region_area > 1000:  # 충분한 영역
                        score += 15
                    else:
                        compatibility['issues'].append('상체 영역이 너무 작음')
                        score -= 15
                else:
                    compatibility['issues'].append('상체 영역 검출 실패')
                    score -= 20
            
            elif clothing_type in self.clothing_categories['lower_body']:
                if 'lower_body' in clothing_regions:
                    region_area = clothing_regions['lower_body']['area']
                    if region_area > 800:  # 충분한 영역
                        score += 15
                    else:
                        compatibility['issues'].append('하체 영역이 너무 작음')
                        score -= 15
                else:
                    compatibility['issues'].append('하체 영역 검출 실패')
                    score -= 20
            
            # 최종 점수 정규화
            score = max(0, min(100, score))
            compatibility['score'] = score
            compatibility['recommended'] = score >= 70
            
            if score >= 90:
                compatibility['suggestions'].append('완벽한 피팅 조건입니다!')
            elif score >= 70:
                compatibility['suggestions'].append('좋은 피팅 결과를 기대할 수 있습니다')
            else:
                compatibility['suggestions'].append('더 나은 피팅을 위해 포즈를 조정해 보세요')
            
        except Exception as e:
            logger.error(f"피팅 호환성 분석 오류: {e}")
            compatibility['issues'].append(f'분석 오류: {str(e)}')
        
        return compatibility
    
    def __del__(self):
        """정리 작업"""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except:
            pass

# 전역 인체 분석기 인스턴스
human_analyzer = None

def get_human_analyzer(use_gpu: bool = True) -> HumanBodyAnalyzer:
    """전역 인체 분석기 인스턴스 반환 (싱글톤 패턴)"""
    global human_analyzer
    if human_analyzer is None:
        human_analyzer = HumanBodyAnalyzer(use_gpu=use_gpu)
    return human_analyzer

# 편의 함수들
async def quick_pose_analysis(image: Image.Image) -> Dict[str, Any]:
    """빠른 포즈 분석"""
    analyzer = get_human_analyzer()
    return await analyzer.analyze_human_body(
        image, 
        {
            'enable_pose': True,
            'enable_segmentation': False,
            'enable_measurements': False,
            'enable_clothing_regions': False
        }
    )

async def full_body_analysis(image: Image.Image, visualization: bool = False) -> Dict[str, Any]:
    """전체 인체 분석"""
    analyzer = get_human_analyzer()
    return await analyzer.analyze_human_body(
        image,
        {
            'enable_pose': True,
            'enable_segmentation': True,
            'enable_measurements': True,
            'enable_clothing_regions': True,
            'visualization': visualization
        }
    )