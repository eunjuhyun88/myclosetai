# app/ai_pipeline/steps/step_02_pose_estimation.py
"""
2단계: 포즈 추정 (Pose Estimation) - 수정된 버전
Pipeline Manager와 완전 호환되는 실제 포즈 추정 시스템
MediaPipe + OpenPose 호환 + M3 Max 최적화
"""
import os
import logging
import time
import asyncio
import json
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import cv2
from PIL import Image
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# MediaPipe (실제 포즈 추정용)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe가 설치되지 않았습니다. 대안 방법을 사용합니다.")

logger = logging.getLogger(__name__)

class PoseEstimationStep:
    """
    포즈 추정 스텝 - Pipeline Manager 완전 호환
    - M3 Max MPS 최적화
    - MediaPipe 기반 실제 포즈 추정
    - OpenPose 18 키포인트 호환
    - 실시간 품질 분석
    """
    
    # OpenPose 18 키포인트 정의
    OPENPOSE_18_KEYPOINTS = [
        "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
        "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
        "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
        "left_eye", "right_ear", "left_ear"
    ]
    
    def __init__(self, device: str, config: Optional[Dict[str, Any]] = None):
        """
        초기화 - Pipeline Manager 완전 호환
        
        Args:
            device: 사용할 디바이스 (mps, cuda, cpu)
            config: 설정 딕셔너리 (선택적)
        """
        # model_loader는 내부에서 전역 함수로 가져옴
        from app.ai_pipeline.utils.model_loader import get_global_model_loader
        self.model_loader = get_global_model_loader()
        
        self.device = device
        self.config = config or {}
        self.is_initialized = False
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        # 포즈 추정 설정
        self.pose_config = self.config.get('pose', {
            'model_complexity': 2,           # MediaPipe 모델 복잡도 (0, 1, 2)
            'min_detection_confidence': 0.7, # 최소 검출 신뢰도
            'min_tracking_confidence': 0.5,  # 최소 추적 신뢰도
            'enable_segmentation': False,     # 세그멘테이션 활성화
            'max_image_size': 1024,          # 최대 이미지 크기
            'use_face': True,                # 얼굴 키포인트 사용
            'use_hands': False               # 손 키포인트 사용 (성능상 비활성화)
        })
        
        # MediaPipe 모델 변수들
        self.mp_pose = None
        self.mp_drawing = None
        self.mp_drawing_styles = None
        self.pose_detector = None
        
        # 통계 및 캐시
        self.processing_stats = {
            'total_processed': 0,
            'successful_detections': 0,
            'average_processing_time': 0.0,
            'last_quality_score': 0.0
        }
        
        self.logger.info(f"🏃 포즈 추정 스텝 초기화 - 디바이스: {device}")
    
    async def initialize(self) -> bool:
        """초기화 메서드"""
        try:
            if MEDIAPIPE_AVAILABLE:
                # MediaPipe 초기화
                self.mp_pose = mp.solutions.pose
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_drawing_styles = mp.solutions.drawing_styles
                
                # 포즈 검출기 초기화
                self.pose_detector = self.mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=self.pose_config['model_complexity'],
                    enable_segmentation=self.pose_config['enable_segmentation'],
                    min_detection_confidence=self.pose_config['min_detection_confidence'],
                    min_tracking_confidence=self.pose_config['min_tracking_confidence']
                )
                
                self.logger.info("✅ MediaPipe 포즈 추정 모델 초기화 완료")
            else:
                # 폴백: 더미 검출기
                self.pose_detector = self._create_dummy_detector()
                self.logger.warning("⚠️ MediaPipe 없음 - 더미 포즈 검출기 사용")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 포즈 추정 초기화 실패: {e}")
            # 에러 시 더미 검출기로 폴백
            self.pose_detector = self._create_dummy_detector()
            self.is_initialized = True
            return True
    
    async def process(self, person_image: Union[str, np.ndarray, Image.Image], **kwargs) -> Dict[str, Any]:
        """
        포즈 추정 처리
        
        Args:
            person_image: 입력 이미지 (경로, numpy 배열, PIL 이미지)
            **kwargs: 추가 매개변수
            
        Returns:
            Dict: 처리 결과
        """
        start_time = time.time()
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # 이미지 로드 및 전처리
            image_array = await self._load_and_preprocess_image(person_image)
            if image_array is None:
                return self._create_empty_result("이미지 로드 실패")
            
            # 포즈 추정 실행
            if MEDIAPIPE_AVAILABLE and self.pose_detector:
                pose_result = await self._detect_pose_mediapipe(image_array)
            else:
                pose_result = await self._detect_pose_dummy(image_array)
            
            # OpenPose 18 형식으로 변환
            keypoints_18 = self._convert_to_openpose_18(pose_result, image_array.shape)
            
            # 포즈 분석
            pose_analysis = self._analyze_pose(keypoints_18, image_array.shape)
            
            # 품질 평가
            quality_metrics = self._evaluate_pose_quality(keypoints_18, pose_result)
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            
            # 통계 업데이트
            self._update_statistics(processing_time, quality_metrics['overall_confidence'])
            
            # 결과 구성
            result = {
                'success': True,
                'keypoints_18': keypoints_18,
                'keypoints_mediapipe': pose_result,
                'pose_confidence': quality_metrics['overall_confidence'],
                'body_orientation': pose_analysis['orientation'],
                'pose_analysis': pose_analysis,
                'quality_metrics': quality_metrics,
                'processing_info': {
                    'processing_time': processing_time,
                    'keypoints_detected': sum(1 for kp in keypoints_18 if kp[2] > 0.5),
                    'total_keypoints': 18,
                    'detection_method': 'mediapipe' if MEDIAPIPE_AVAILABLE else 'dummy'
                }
            }
            
            self.logger.info(f"✅ 포즈 추정 완료 - 신뢰도: {quality_metrics['overall_confidence']:.3f}, 시간: {processing_time:.3f}초")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 포즈 추정 실패: {e}")
            return self._create_empty_result(f"처리 오류: {str(e)}")
    
    async def _load_and_preprocess_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> Optional[np.ndarray]:
        """이미지 로드 및 전처리"""
        try:
            # 입력 타입에 따른 로드
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    self.logger.error(f"이미지 파일이 존재하지 않음: {image_input}")
                    return None
                image_array = cv2.imread(image_input)
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            elif isinstance(image_input, np.ndarray):
                image_array = image_input.copy()
                if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                    # BGR to RGB 변환 (OpenCV 이미지인 경우)
                    if image_array.max() > 1.0:  # 0-255 범위인 경우
                        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            elif isinstance(image_input, Image.Image):
                image_array = np.array(image_input.convert('RGB'))
            else:
                self.logger.error(f"지원하지 않는 이미지 형식: {type(image_input)}")
                return None
            
            # 크기 조정 (필요한 경우)
            height, width = image_array.shape[:2]
            max_size = self.pose_config['max_image_size']
            
            if max(height, width) > max_size:
                if height > width:
                    new_height = max_size
                    new_width = int(width * max_size / height)
                else:
                    new_width = max_size
                    new_height = int(height * max_size / width)
                
                image_array = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
                self.logger.info(f"🔄 이미지 크기 조정: ({width}, {height}) -> ({new_width}, {new_height})")
            
            return image_array
            
        except Exception as e:
            self.logger.error(f"이미지 전처리 실패: {e}")
            return None
    
    async def _detect_pose_mediapipe(self, image_array: np.ndarray) -> List[Dict]:
        """MediaPipe를 사용한 포즈 추정"""
        try:
            # MediaPipe 처리
            results = self.pose_detector.process(image_array)
            
            if results.pose_landmarks:
                # 키포인트 추출
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                
                self.logger.debug(f"MediaPipe 키포인트 검출: {len(landmarks)}개")
                return landmarks
            else:
                self.logger.warning("MediaPipe에서 포즈를 감지하지 못했습니다.")
                return []
                
        except Exception as e:
            self.logger.error(f"MediaPipe 포즈 추정 실패: {e}")
            return []
    
    async def _detect_pose_dummy(self, image_array: np.ndarray) -> List[Dict]:
        """더미 포즈 추정 (폴백용)"""
        height, width = image_array.shape[:2]
        
        # 가상의 포즈 키포인트 생성
        dummy_landmarks = []
        
        # 33개 MediaPipe 키포인트 생성
        for i in range(33):
            # 이미지 중앙 근처에 랜덤하게 배치
            x = 0.4 + np.random.random() * 0.2  # 40-60% 지점
            y = 0.3 + np.random.random() * 0.4  # 30-70% 지점
            z = np.random.random() * 0.1
            visibility = 0.7 + np.random.random() * 0.3  # 0.7-1.0
            
            dummy_landmarks.append({
                'x': x,
                'y': y,
                'z': z,
                'visibility': visibility
            })
        
        self.logger.debug("더미 포즈 키포인트 생성 완료")
        return dummy_landmarks
    
    def _convert_to_openpose_18(self, mediapipe_landmarks: List[Dict], image_shape: Tuple) -> List[List[float]]:
        """MediaPipe 키포인트를 OpenPose 18 형식으로 변환"""
        
        height, width = image_shape[:2]
        keypoints_18 = [[0.0, 0.0, 0.0] for _ in range(18)]
        
        if not mediapipe_landmarks:
            return keypoints_18
        
        try:
            # MediaPipe -> OpenPose 18 매핑
            mp_to_op18 = {
                0: 0,   # nose
                12: 1,  # neck (어깨 중점으로 계산)
                12: 2,  # right_shoulder
                14: 3,  # right_elbow
                16: 4,  # right_wrist
                11: 5,  # left_shoulder
                13: 6,  # left_elbow
                15: 7,  # left_wrist
                24: 8,  # right_hip
                26: 9,  # right_knee
                28: 10, # right_ankle
                23: 11, # left_hip
                25: 12, # left_knee
                27: 13, # left_ankle
                2: 14,  # right_eye
                5: 15,  # left_eye
                8: 16,  # right_ear
                7: 17   # left_ear
            }
            
            # 기본 매핑
            for op_idx, mp_idx in mp_to_op18.items():
                if mp_idx < len(mediapipe_landmarks):
                    landmark = mediapipe_landmarks[mp_idx]
                    keypoints_18[op_idx] = [
                        landmark['x'] * width,
                        landmark['y'] * height,
                        landmark['visibility']
                    ]
            
            # 목 (neck) 계산 - 양쪽 어깨의 중점
            if len(mediapipe_landmarks) > 12:
                left_shoulder = mediapipe_landmarks[11]
                right_shoulder = mediapipe_landmarks[12]
                
                neck_x = (left_shoulder['x'] + right_shoulder['x']) / 2 * width
                neck_y = (left_shoulder['y'] + right_shoulder['y']) / 2 * height
                neck_conf = min(left_shoulder['visibility'], right_shoulder['visibility'])
                
                keypoints_18[1] = [neck_x, neck_y, neck_conf]
            
            return keypoints_18
            
        except Exception as e:
            self.logger.error(f"키포인트 변환 실패: {e}")
            return keypoints_18
    
    def _analyze_pose(self, keypoints_18: List[List[float]], image_shape: Tuple) -> Dict[str, Any]:
        """포즈 분석"""
        
        analysis = {}
        
        # 1. 신체 방향 추정
        analysis["orientation"] = self._estimate_body_orientation(keypoints_18)
        
        # 2. 관절 각도 계산
        analysis["angles"] = self._calculate_joint_angles(keypoints_18)
        
        # 3. 신체 비율 분석
        analysis["proportions"] = self._analyze_body_proportions(keypoints_18)
        
        # 4. 바운딩 박스 계산
        analysis["bbox"] = self._calculate_pose_bbox(keypoints_18, image_shape)
        
        # 5. 포즈 타입 분류
        analysis["pose_type"] = self._classify_pose_type(keypoints_18, analysis["angles"])
        
        return analysis
    
    def _estimate_body_orientation(self, keypoints_18: List[List[float]]) -> str:
        """신체 방향 추정 (정면/측면/뒷면)"""
        
        try:
            # 어깨와 엉덩이 키포인트
            left_shoulder = keypoints_18[5]
            right_shoulder = keypoints_18[2]
            left_hip = keypoints_18[11]
            right_hip = keypoints_18[8]
            
            # 모든 키포인트가 검출된 경우만
            if all(kp[2] > 0.5 for kp in [left_shoulder, right_shoulder, left_hip, right_hip]):
                
                # 어깨 너비와 엉덩이 너비
                shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
                hip_width = abs(right_hip[0] - left_hip[0])
                
                # 평균 너비
                avg_width = (shoulder_width + hip_width) / 2
                
                # 임계값 기반 분류 (이미지 크기에 비례)
                if avg_width < 80:
                    return "side"      # 측면
                elif avg_width < 150:
                    return "diagonal"  # 대각선
                else:
                    return "front"     # 정면
            
            return "unknown"
            
        except Exception as e:
            logger.warning(f"신체 방향 추정 실패: {e}")
            return "unknown"
    
    def _calculate_joint_angles(self, keypoints_18: List[List[float]]) -> Dict[str, float]:
        """주요 관절 각도 계산"""
        
        def angle_between_points(p1, p2, p3):
            """세 점으로 각도 계산 (p2가 꼭지점)"""
            if any(p[2] < 0.5 for p in [p1, p2, p3]):
                return 0.0
            
            # 벡터 계산
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            # 각도 계산
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            
            return float(angle)
        
        angles = {}
        
        try:
            # 팔 각도
            angles["left_arm_angle"] = angle_between_points(
                keypoints_18[5], keypoints_18[6], keypoints_18[7]  # shoulder-elbow-wrist
            )
            angles["right_arm_angle"] = angle_between_points(
                keypoints_18[2], keypoints_18[3], keypoints_18[4]
            )
            
            # 다리 각도
            angles["left_leg_angle"] = angle_between_points(
                keypoints_18[11], keypoints_18[12], keypoints_18[13]  # hip-knee-ankle
            )
            angles["right_leg_angle"] = angle_between_points(
                keypoints_18[8], keypoints_18[9], keypoints_18[10]
            )
            
            # 몸통 각도 (어깨-목-엉덩이)
            if all(keypoints_18[i][2] > 0.5 for i in [1, 5, 11]):
                angles["torso_angle"] = angle_between_points(
                    keypoints_18[5], keypoints_18[1], keypoints_18[11]
                )
            
        except Exception as e:
            logger.warning(f"관절 각도 계산 실패: {e}")
        
        return angles
    
    def _analyze_body_proportions(self, keypoints_18: List[List[float]]) -> Dict[str, float]:
        """신체 비율 분석"""
        
        proportions = {}
        
        try:
            # 머리 길이 (코-목)
            if keypoints_18[0][2] > 0.5 and keypoints_18[1][2] > 0.5:
                head_length = abs(keypoints_18[1][1] - keypoints_18[0][1])
                proportions["head_length"] = head_length
            
            # 몸통 길이 (목-엉덩이)
            if keypoints_18[1][2] > 0.5 and keypoints_18[8][2] > 0.5:
                torso_length = abs(keypoints_18[8][1] - keypoints_18[1][1])
                proportions["torso_length"] = torso_length
            
            # 다리 길이 (엉덩이-발목)
            if keypoints_18[8][2] > 0.5 and keypoints_18[10][2] > 0.5:
                leg_length = abs(keypoints_18[10][1] - keypoints_18[8][1])
                proportions["leg_length"] = leg_length
            
            # 어깨 너비
            if keypoints_18[2][2] > 0.5 and keypoints_18[5][2] > 0.5:
                shoulder_width = abs(keypoints_18[5][0] - keypoints_18[2][0])
                proportions["shoulder_width"] = shoulder_width
            
            # 비율 계산
            if "head_length" in proportions and "torso_length" in proportions:
                proportions["head_to_torso_ratio"] = proportions["head_length"] / (proportions["torso_length"] + 1e-8)
            
            if "torso_length" in proportions and "leg_length" in proportions:
                proportions["torso_to_leg_ratio"] = proportions["torso_length"] / (proportions["leg_length"] + 1e-8)
                
        except Exception as e:
            logger.warning(f"신체 비율 분석 실패: {e}")
        
        return proportions
    
    def _calculate_pose_bbox(self, keypoints_18: List[List[float]], image_shape: Tuple) -> Dict[str, int]:
        """포즈 바운딩 박스 계산"""
        
        # 유효한 키포인트만 선택
        valid_points = [(x, y) for x, y, conf in keypoints_18 if conf > 0.5]
        
        if not valid_points:
            return {"x": 0, "y": 0, "width": 0, "height": 0}
        
        xs, ys = zip(*valid_points)
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # 여백 추가 (15%)
        margin_x = int((x_max - x_min) * 0.15)
        margin_y = int((y_max - y_min) * 0.15)
        
        height, width = image_shape
        
        bbox = {
            "x": max(0, int(x_min - margin_x)),
            "y": max(0, int(y_min - margin_y)),
            "width": min(width, int(x_max - x_min + 2 * margin_x)),
            "height": min(height, int(y_max - y_min + 2 * margin_y))
        }
        
        return bbox
    
    def _classify_pose_type(self, keypoints_18: List[List[float]], angles: Dict[str, float]) -> str:
        """포즈 타입 분류"""
        
        try:
            # 팔 각도 기반 분류
            left_arm = angles.get("left_arm_angle", 180)
            right_arm = angles.get("right_arm_angle", 180)
            
            # 다리 각도
            left_leg = angles.get("left_leg_angle", 180)
            right_leg = angles.get("right_leg_angle", 180)
            
            # T-포즈 (팔이 수평)
            if abs(left_arm - 180) < 20 and abs(right_arm - 180) < 20:
                return "t_pose"
            
            # A-포즈 (팔이 약간 아래)
            elif 140 < left_arm < 170 and 140 < right_arm < 170:
                return "a_pose"
            
            # 걷기 포즈
            elif abs(left_leg - right_leg) > 30:
                return "walking"
            
            # 앉기 포즈
            elif left_leg < 140 or right_leg < 140:
                return "sitting"
            
            # 서있는 포즈 (기본)
            else:
                return "standing"
                
        except Exception as e:
            logger.warning(f"포즈 타입 분류 실패: {e}")
            return "unknown"
    
    def _evaluate_pose_quality(self, keypoints_18: List[List[float]], mediapipe_keypoints: List[Dict]) -> Dict[str, float]:
        """포즈 품질 평가"""
        
        try:
            # 1. 검출 비율
            detected_18 = sum(1 for kp in keypoints_18 if kp[2] > 0.5)
            detection_rate = detected_18 / 18
            
            # 2. 주요 키포인트 검출 비율
            major_indices = [0, 1, 2, 5, 8, 11]  # nose, neck, shoulders, hips
            major_detected = sum(1 for idx in major_indices if keypoints_18[idx][2] > 0.5)
            major_detection_rate = major_detected / len(major_indices)
            
            # 3. 평균 신뢰도
            confidences = [kp[2] for kp in keypoints_18 if kp[2] > 0]
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # 4. 대칭성 점수
            symmetry_score = self._calculate_symmetry_score(keypoints_18)
            
            # 5. 전체 신뢰도 계산
            overall_confidence = (
                detection_rate * 0.3 +
                major_detection_rate * 0.3 +
                avg_confidence * 0.3 +
                symmetry_score * 0.1
            )
            
            return {
                'overall_confidence': overall_confidence,
                'detection_rate': detection_rate,
                'major_detection_rate': major_detection_rate,
                'average_confidence': avg_confidence,
                'symmetry_score': symmetry_score,
                'quality_grade': self._get_quality_grade(overall_confidence)
            }
            
        except Exception as e:
            logger.error(f"품질 평가 실패: {e}")
            return {
                'overall_confidence': 0.0,
                'detection_rate': 0.0,
                'major_detection_rate': 0.0,
                'average_confidence': 0.0,
                'symmetry_score': 0.0,
                'quality_grade': 'poor'
            }
    
    def _calculate_symmetry_score(self, keypoints_18: List[List[float]]) -> float:
        """좌우 대칭성 점수 계산"""
        
        # 대칭 키포인트 쌍
        symmetric_pairs = [
            (2, 5),   # shoulders
            (3, 6),   # elbows
            (4, 7),   # wrists
            (8, 11),  # hips
            (9, 12),  # knees
            (10, 13), # ankles
            (14, 15), # eyes
            (16, 17)  # ears
        ]
        
        symmetry_scores = []
        
        for right_idx, left_idx in symmetric_pairs:
            right_kp = keypoints_18[right_idx]
            left_kp = keypoints_18[left_idx]
            
            if right_kp[2] > 0.5 and left_kp[2] > 0.5:
                # 신뢰도 차이
                conf_diff = abs(right_kp[2] - left_kp[2])
                conf_similarity = 1.0 - conf_diff
                
                # 위치 대칭성 (Y 좌표 차이)
                y_diff = abs(right_kp[1] - left_kp[1])
                max_y = max(right_kp[1], left_kp[1])
                if max_y > 0:
                    y_similarity = 1.0 - min(y_diff / max_y, 1.0)
                else:
                    y_similarity = 1.0
                
                # 종합 대칭성
                pair_symmetry = (conf_similarity + y_similarity) / 2
                symmetry_scores.append(pair_symmetry)
        
        return np.mean(symmetry_scores) if symmetry_scores else 0.0
    
    def _get_quality_grade(self, overall_confidence: float) -> str:
        """품질 등급 반환"""
        if overall_confidence >= 0.9:
            return "excellent"
        elif overall_confidence >= 0.8:
            return "good"
        elif overall_confidence >= 0.6:
            return "fair"
        elif overall_confidence >= 0.4:
            return "poor"
        else:
            return "very_poor"
    
    def _create_dummy_detector(self):
        """더미 검출기 생성"""
        class DummyDetector:
            def process(self, image):
                return None
        return DummyDetector()
    
    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """빈 결과 생성"""
        return {
            'success': False,
            'error': reason,
            'keypoints_18': [[0.0, 0.0, 0.0] for _ in range(18)],
            'keypoints_mediapipe': [],
            'pose_confidence': 0.0,
            'body_orientation': 'unknown',
            'pose_analysis': {},
            'quality_metrics': {
                'overall_confidence': 0.0,
                'quality_grade': 'failed'
            },
            'processing_info': {
                'processing_time': 0.0,
                'keypoints_detected': 0,
                'total_keypoints': 18,
                'detection_method': 'none'
            }
        }
    
    def _update_statistics(self, processing_time: float, quality_score: float):
        """통계 업데이트"""
        self.processing_stats['total_processed'] += 1
        
        if quality_score > 0.5:
            self.processing_stats['successful_detections'] += 1
        
        # 이동 평균으로 처리 시간 업데이트
        alpha = 0.1
        self.processing_stats['average_processing_time'] = (
            alpha * processing_time + 
            (1 - alpha) * self.processing_stats['average_processing_time']
        )
        
        self.processing_stats['last_quality_score'] = quality_score
    
    def visualize_pose(self, image: np.ndarray, keypoints_18: List[List[float]], save_path: Optional[str] = None) -> np.ndarray:
        """포즈 시각화"""
        
        vis_image = image.copy()
        
        # 키포인트 그리기
        for i, (x, y, conf) in enumerate(keypoints_18):
            if conf > 0.5:
                cv2.circle(vis_image, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.putText(vis_image, str(i), (int(x), int(y-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # 연결선 그리기
        connections = self._get_openpose_connections()
        for pt1_idx, pt2_idx in connections:
            pt1 = keypoints_18[pt1_idx]
            pt2 = keypoints_18[pt2_idx]
            
            if pt1[2] > 0.5 and pt2[2] > 0.5:
                cv2.line(vis_image, 
                        (int(pt1[0]), int(pt1[1])), 
                        (int(pt2[0]), int(pt2[1])), 
                        (255, 0, 0), 3)
        
        # 저장
        if save_path:
            cv2.imwrite(save_path, vis_image)
            logger.info(f"💾 포즈 시각화 저장: {save_path}")
        
        return vis_image
    
    def export_keypoints(self, keypoints_18: List[List[float]], format: str = "json") -> str:
        """키포인트 내보내기"""
        
        if format.lower() == "json":
            export_data = {
                "format": "openpose_18",
                "keypoints": keypoints_18,
                "keypoint_names": self.OPENPOSE_18_KEYPOINTS,
                "connections": self._get_openpose_connections()
            }
            return json.dumps(export_data, indent=2)
        
        elif format.lower() == "csv":
            lines = ["id,name,x,y,confidence"]
            for i, (x, y, conf) in enumerate(keypoints_18):
                lines.append(f"{i},{self.OPENPOSE_18_KEYPOINTS[i]},{x},{y},{conf}")
            return "\n".join(lines)
        
        else:
            raise ValueError(f"지원하지 않는 형식: {format}")
    
    def _get_openpose_connections(self) -> List[List[int]]:
        """OpenPose 연결선 정보"""
        return [
            # 몸통
            [1, 2], [1, 5], [2, 8], [5, 11], [8, 11],
            
            # 오른팔
            [2, 3], [3, 4],
            
            # 왼팔  
            [5, 6], [6, 7],
            
            # 오른다리
            [8, 9], [9, 10],
            
            # 왼다리
            [11, 12], [12, 13],
            
            # 머리
            [1, 0], [0, 14], [0, 15], [14, 16], [15, 17]
        ]
    
    async def cleanup(self):
        """리소스 정리"""
        if self.pose_detector:
            if hasattr(self.pose_detector, 'close'):
                self.pose_detector.close()
            self.pose_detector = None
        
        self.mp_pose = None
        self.mp_drawing = None
        self.is_initialized = False
        
        logger.info("🧹 포즈 추정 스텝 정리 완료")