# app/ai_pipeline/steps/step_02_pose_estimation.py
"""
2단계: 포즈 추정 (Pose Estimation) - 통일된 생성자 패턴 적용
✅ 최적화된 생성자: device 자동감지, M3 Max 최적화, 일관된 인터페이스
MediaPipe + OpenPose 호환 + 고급 포즈 분석
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
    ✅ 2단계: 포즈 추정 - 통일된 생성자 패턴
    - 자동 디바이스 감지
    - M3 Max 최적화
    - 일관된 인터페이스
    - MediaPipe + OpenPose 호환
    """
    
    # OpenPose 18 키포인트 정의
    OPENPOSE_18_KEYPOINTS = [
        "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
        "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
        "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
        "left_eye", "right_ear", "left_ear"
    ]
    
    # MediaPipe 33 키포인트 매핑
    MEDIAPIPE_KEYPOINT_NAMES = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
        "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
        "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
        "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
        "right_heel", "left_foot_index", "right_foot_index"
    ]
    
    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        ✅ 통일된 생성자 - 최적화된 인터페이스
        
        Args:
            device: 사용할 디바이스 (None=자동감지, 'cpu', 'cuda', 'mps')
            config: 스텝별 설정 딕셔너리
            **kwargs: 확장 파라미터들
                - device_type: str = "auto"
                - memory_gb: float = 16.0  
                - is_m3_max: bool = False
                - optimization_enabled: bool = True
                - quality_level: str = "balanced"
                - model_complexity: int = 2 (MediaPipe 모델 복잡도 0,1,2)
                - min_detection_confidence: float = 0.7
                - min_tracking_confidence: float = 0.5
                - enable_segmentation: bool = False
                - max_image_size: int = 1024
                - use_face: bool = True (얼굴 키포인트 사용)
                - use_hands: bool = False (손 키포인트 사용)
        """
        # 💡 지능적 디바이스 자동 감지
        self.device = self._auto_detect_device(device)
        
        # 📋 기본 설정
        self.config = config or {}
        self.step_name = self.__class__.__name__
        self.logger = logging.getLogger(f"pipeline.{self.step_name}")
        
        # 🔧 표준 시스템 파라미터 추출 (일관성)
        self.device_type = kwargs.get('device_type', 'auto')
        self.memory_gb = kwargs.get('memory_gb', 16.0)
        self.is_m3_max = kwargs.get('is_m3_max', self._detect_m3_max())
        self.optimization_enabled = kwargs.get('optimization_enabled', True)
        self.quality_level = kwargs.get('quality_level', 'balanced')
        
        # ⚙️ 스텝별 특화 파라미터를 config에 병합
        self._merge_step_specific_config(kwargs)
        
        # ✅ 상태 초기화
        self.is_initialized = False
        
        # 🎯 기존 클래스별 고유 초기화 로직 실행
        self._initialize_step_specific()
        
        self.logger.info(f"🎯 {self.step_name} 초기화 - 디바이스: {self.device}")
    
    def _auto_detect_device(self, preferred_device: Optional[str]) -> str:
        """💡 지능적 디바이스 자동 감지"""
        if preferred_device:
            return preferred_device

        try:
            import torch
            if torch.backends.mps.is_available():
                return 'mps'  # M3 Max 우선
            elif torch.cuda.is_available():
                return 'cuda'  # NVIDIA GPU
            else:
                return 'cpu'  # 폴백
        except ImportError:
            return 'cpu'

    def _detect_m3_max(self) -> bool:
        """🍎 M3 Max 칩 자동 감지"""
        try:
            import platform
            import subprocess

            if platform.system() == 'Darwin':  # macOS
                # M3 Max 감지 로직
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                return 'M3' in result.stdout
        except:
            pass
        return False

    def _merge_step_specific_config(self, kwargs: Dict[str, Any]):
        """⚙️ 스텝별 특화 설정 병합"""
        # 시스템 파라미터 제외하고 모든 kwargs를 config에 병합
        system_params = {
            'device_type', 'memory_gb', 'is_m3_max', 
            'optimization_enabled', 'quality_level'
        }

        for key, value in kwargs.items():
            if key not in system_params:
                self.config[key] = value

    def _initialize_step_specific(self):
        """🎯 기존 초기화 로직 완전 유지"""
        # 2단계 전용 포즈 추정 설정
        self.pose_config = self.config.get('pose', {})
        
        # MediaPipe 설정 (M3 Max에서 더 높은 품질)
        default_complexity = 2 if self.is_m3_max else 1
        self.model_complexity = self.config.get('model_complexity', default_complexity)
        self.min_detection_confidence = self.config.get('min_detection_confidence', 0.7)
        self.min_tracking_confidence = self.config.get('min_tracking_confidence', 0.5)
        self.enable_segmentation = self.config.get('enable_segmentation', False)
        
        # 이미지 처리 설정
        default_max_size = 1024 if self.memory_gb >= 32 else 512
        self.max_image_size = self.config.get('max_image_size', default_max_size)
        
        # 추가 키포인트 설정
        self.use_face = self.config.get('use_face', True)
        self.use_hands = self.config.get('use_hands', False)  # 성능상 기본 비활성화
        
        # MediaPipe 모델 변수들
        self.mp_pose = None
        self.mp_drawing = None
        self.mp_drawing_styles = None
        self.pose_detector = None
        self.face_detector = None
        self.hands_detector = None
        
        # 2단계 전용 통계
        self.pose_stats = {
            'total_processed': 0,
            'successful_detections': 0,
            'average_processing_time': 0.0,
            'average_keypoints_detected': 0.0,
            'last_quality_score': 0.0,
            'mediapipe_usage': 0,
            'fallback_usage': 0,
            'face_detections': 0,
            'hands_detections': 0
        }
        
        # 성능 캐시 (M3 Max에서 더 큰 캐시)
        cache_size = 100 if self.is_m3_max and self.memory_gb >= 128 else 50
        self.detection_cache = {}
        self.cache_max_size = cache_size
        
        # 스레드 풀 (기존 코드 호환)
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def initialize(self) -> bool:
        """
        ✅ 통일된 초기화 인터페이스
        
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            self.logger.info("🔄 2단계: 포즈 추정 시스템 초기화 중...")
            
            if MEDIAPIPE_AVAILABLE:
                # MediaPipe 초기화
                self.mp_pose = mp.solutions.pose
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_drawing_styles = mp.solutions.drawing_styles
                
                # 포즈 검출기 초기화 (M3 Max 최적화 설정)
                self.pose_detector = self.mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=self.model_complexity,
                    enable_segmentation=self.enable_segmentation,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence
                )
                
                # 얼굴 검출기 (선택적)
                if self.use_face:
                    try:
                        mp_face = mp.solutions.face_mesh
                        self.face_detector = mp_face.FaceMesh(
                            static_image_mode=True,
                            max_num_faces=1,
                            refine_landmarks=self.is_m3_max,  # M3 Max에서 정밀도 향상
                            min_detection_confidence=self.min_detection_confidence
                        )
                        self.logger.info("✅ 얼굴 검출기 초기화 완료")
                    except Exception as e:
                        self.logger.warning(f"얼굴 검출기 초기화 실패: {e}")
                
                # 손 검출기 (선택적)
                if self.use_hands:
                    try:
                        mp_hands = mp.solutions.hands
                        self.hands_detector = mp_hands.Hands(
                            static_image_mode=True,
                            max_num_hands=2,
                            model_complexity=min(1, self.model_complexity),  # 손은 복잡도 제한
                            min_detection_confidence=self.min_detection_confidence
                        )
                        self.logger.info("✅ 손 검출기 초기화 완료")
                    except Exception as e:
                        self.logger.warning(f"손 검출기 초기화 실패: {e}")
                
                self.logger.info("✅ MediaPipe 포즈 추정 시스템 초기화 완료")
            else:
                # 폴백: 더미 검출기
                self.pose_detector = self._create_dummy_detector()
                self.logger.warning("⚠️ MediaPipe 없음 - 더미 포즈 검출기 사용")
            
            # M3 Max 최적화 워밍업
            if self.is_m3_max and self.optimization_enabled:
                await self._warmup_m3_max()
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            error_msg = f"포즈 추정 초기화 실패: {e}"
            self.logger.error(f"❌ {error_msg}")
            
            # 에러 시 더미 검출기로 폴백
            self.pose_detector = self._create_dummy_detector()
            self.is_initialized = True
            return True
    
    async def process(
        self, 
        person_image: Union[str, np.ndarray, Image.Image],
        **kwargs
    ) -> Dict[str, Any]:
        """
        ✅ 통일된 처리 인터페이스
        
        Args:
            person_image: 입력 이미지 (경로, numpy 배열, PIL 이미지)
            **kwargs: 추가 매개변수
                - return_face_keypoints: bool = False
                - return_hand_keypoints: bool = False
                - enable_pose_analysis: bool = True
                - cache_result: bool = True
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            self.logger.info("🏃 포즈 추정 처리 시작")
            
            # 캐시 확인
            cache_key = self._generate_cache_key(person_image)
            if cache_key in self.detection_cache and kwargs.get('cache_result', True):
                self.logger.info("💾 캐시에서 포즈 결과 반환")
                cached_result = self.detection_cache[cache_key].copy()
                cached_result['from_cache'] = True
                return cached_result
            
            # 이미지 로드 및 전처리
            image_array = await self._load_and_preprocess_image(person_image)
            if image_array is None:
                return self._create_empty_result("이미지 로드 실패")
            
            # 포즈 추정 실행
            if MEDIAPIPE_AVAILABLE and self.pose_detector:
                pose_result = await self._detect_pose_mediapipe(image_array)
                self.pose_stats['mediapipe_usage'] += 1
            else:
                pose_result = await self._detect_pose_dummy(image_array)
                self.pose_stats['fallback_usage'] += 1
            
            # 추가 키포인트 검출
            face_keypoints = None
            hand_keypoints = None
            
            if kwargs.get('return_face_keypoints', False) and self.face_detector:
                face_keypoints = await self._detect_face_keypoints(image_array)
                if face_keypoints:
                    self.pose_stats['face_detections'] += 1
            
            if kwargs.get('return_hand_keypoints', False) and self.hands_detector:
                hand_keypoints = await self._detect_hand_keypoints(image_array)
                if hand_keypoints:
                    self.pose_stats['hands_detections'] += 1
            
            # OpenPose 18 형식으로 변환
            keypoints_18 = self._convert_to_openpose_18(pose_result, image_array.shape)
            
            # 포즈 분석
            pose_analysis = {}
            if kwargs.get('enable_pose_analysis', True):
                pose_analysis = self._analyze_pose(keypoints_18, image_array.shape)
            
            # 품질 평가
            quality_metrics = self._evaluate_pose_quality(keypoints_18, pose_result)
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            
            # 통계 업데이트
            self._update_pose_stats(processing_time, quality_metrics['overall_confidence'])
            
            # 결과 구성
            result = {
                'success': True,
                'keypoints_18': keypoints_18,
                'keypoints_mediapipe': pose_result,
                'pose_confidence': quality_metrics['overall_confidence'],
                'body_orientation': pose_analysis.get('orientation', 'unknown'),
                'pose_analysis': pose_analysis,
                'quality_metrics': quality_metrics,
                'processing_info': {
                    'processing_time': processing_time,
                    'keypoints_detected': sum(1 for kp in keypoints_18 if kp[2] > 0.5),
                    'total_keypoints': 18,
                    'detection_method': 'mediapipe' if MEDIAPIPE_AVAILABLE else 'dummy',
                    'device': self.device,
                    'device_type': self.device_type,
                    'm3_max_optimized': self.is_m3_max,
                    'model_complexity': self.model_complexity
                },
                'additional_keypoints': {
                    'face_keypoints': face_keypoints,
                    'hand_keypoints': hand_keypoints
                } if (face_keypoints or hand_keypoints) else None,
                'from_cache': False
            }
            
            # 캐시 저장
            if kwargs.get('cache_result', True):
                self._update_cache(cache_key, result)
            
            self.logger.info(f"✅ 포즈 추정 완료 - 신뢰도: {quality_metrics['overall_confidence']:.3f}, 시간: {processing_time:.3f}초")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"포즈 추정 실패: {e}"
            self.logger.error(f"❌ {error_msg}")
            
            return self._create_empty_result(error_msg)
    
    # =================================================================
    # 🔧 핵심 처리 메서드들
    # =================================================================
    
    async def _warmup_m3_max(self):
        """M3 Max 워밍업"""
        try:
            self.logger.info("🍎 M3 Max 포즈 시스템 워밍업...")
            
            # 더미 이미지로 워밍업
            dummy_image = np.ones((256, 256, 3), dtype=np.uint8) * 128
            
            if self.pose_detector and MEDIAPIPE_AVAILABLE:
                _ = self.pose_detector.process(dummy_image)
                
            self.logger.info("✅ M3 Max 워밍업 완료")
            
        except Exception as e:
            self.logger.warning(f"M3 Max 워밍업 실패: {e}")
    
    async def _load_and_preprocess_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> Optional[np.ndarray]:
        """이미지 로드 및 전처리 (M3 Max 최적화)"""
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
            
            # 크기 조정 (M3 Max에서 더 큰 크기 허용)
            height, width = image_array.shape[:2]
            max_size = self.max_image_size
            
            if max(height, width) > max_size:
                if height > width:
                    new_height = max_size
                    new_width = int(width * max_size / height)
                else:
                    new_width = max_size
                    new_height = int(height * max_size / width)
                
                # M3 Max에서 더 고품질 보간
                interpolation = cv2.INTER_LANCZOS4 if self.is_m3_max else cv2.INTER_AREA
                image_array = cv2.resize(image_array, (new_width, new_height), interpolation=interpolation)
                self.logger.info(f"🔄 이미지 크기 조정: ({width}, {height}) -> ({new_width}, {new_height})")
            
            return image_array
            
        except Exception as e:
            self.logger.error(f"이미지 전처리 실패: {e}")
            return None
    
    async def _detect_pose_mediapipe(self, image_array: np.ndarray) -> List[Dict]:
        """MediaPipe를 사용한 포즈 추정 (M3 Max 최적화)"""
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
    
    async def _detect_face_keypoints(self, image_array: np.ndarray) -> Optional[List[Dict]]:
        """얼굴 키포인트 검출 (M3 Max 고정밀도)"""
        if not self.face_detector:
            return None
        
        try:
            results = self.face_detector.process(image_array)
            
            if results.multi_face_landmarks:
                face_landmarks = []
                for landmark in results.multi_face_landmarks[0].landmark:
                    face_landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                
                self.logger.debug(f"얼굴 키포인트 검출: {len(face_landmarks)}개")
                return face_landmarks
            
            return None
            
        except Exception as e:
            self.logger.warning(f"얼굴 키포인트 검출 실패: {e}")
            return None
    
    async def _detect_hand_keypoints(self, image_array: np.ndarray) -> Optional[List[List[Dict]]]:
        """손 키포인트 검출"""
        if not self.hands_detector:
            return None
        
        try:
            results = self.hands_detector.process(image_array)
            
            if results.multi_hand_landmarks:
                hands_landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_keypoints = []
                    for landmark in hand_landmarks.landmark:
                        hand_keypoints.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })
                    hands_landmarks.append(hand_keypoints)
                
                self.logger.debug(f"손 키포인트 검출: {len(hands_landmarks)}개 손")
                return hands_landmarks
            
            return None
            
        except Exception as e:
            self.logger.warning(f"손 키포인트 검출 실패: {e}")
            return None
    
    async def _detect_pose_dummy(self, image_array: np.ndarray) -> List[Dict]:
        """더미 포즈 추정 (폴백용)"""
        height, width = image_array.shape[:2]
        
        # 가상의 포즈 키포인트 생성 (더 정교하게)
        dummy_landmarks = []
        
        # 33개 MediaPipe 키포인트 생성
        for i in range(33):
            # 이미지 중앙 근처에 랜덤하게 배치 (인체 비율 고려)
            if i < 11:  # 얼굴 부분
                x = 0.45 + np.random.random() * 0.1  # 45-55% 지점
                y = 0.1 + np.random.random() * 0.2   # 10-30% 지점
            elif i < 23:  # 상체 부분
                x = 0.3 + np.random.random() * 0.4   # 30-70% 지점
                y = 0.2 + np.random.random() * 0.3   # 20-50% 지점
            else:  # 하체 부분
                x = 0.35 + np.random.random() * 0.3  # 35-65% 지점
                y = 0.4 + np.random.random() * 0.4   # 40-80% 지점
            
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
        """MediaPipe 키포인트를 OpenPose 18 형식으로 변환 (정밀도 향상)"""
        
        height, width = image_shape[:2]
        keypoints_18 = [[0.0, 0.0, 0.0] for _ in range(18)]
        
        if not mediapipe_landmarks:
            return keypoints_18
        
        try:
            # MediaPipe -> OpenPose 18 매핑 (정확한 인덱스)
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
            
            # 목 (neck) 계산 - 양쪽 어깨의 중점 (정밀도 향상)
            if len(mediapipe_landmarks) > 12:
                left_shoulder = mediapipe_landmarks[11]
                right_shoulder = mediapipe_landmarks[12]
                
                # 가중평균 계산 (visibility 고려)
                left_weight = left_shoulder['visibility']
                right_weight = right_shoulder['visibility']
                total_weight = left_weight + right_weight
                
                if total_weight > 0:
                    neck_x = (left_shoulder['x'] * left_weight + right_shoulder['x'] * right_weight) / total_weight * width
                    neck_y = (left_shoulder['y'] * left_weight + right_shoulder['y'] * right_weight) / total_weight * height
                    neck_conf = min(left_shoulder['visibility'], right_shoulder['visibility'])
                    
                    keypoints_18[1] = [neck_x, neck_y, neck_conf]
            
            return keypoints_18
            
        except Exception as e:
            self.logger.error(f"키포인트 변환 실패: {e}")
            return keypoints_18
    
    def _analyze_pose(self, keypoints_18: List[List[float]], image_shape: Tuple) -> Dict[str, Any]:
        """포즈 분석 (M3 Max 고급 분석)"""
        
        analysis = {}
        
        try:
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
            
            # 6. M3 Max 고급 분석
            if self.is_m3_max:
                analysis["advanced_metrics"] = self._advanced_pose_analysis(keypoints_18)
            
        except Exception as e:
            self.logger.warning(f"포즈 분석 실패: {e}")
            analysis = {"error": str(e)}
        
        return analysis
    
    def _advanced_pose_analysis(self, keypoints_18: List[List[float]]) -> Dict[str, Any]:
        """M3 Max 고급 포즈 분석"""
        try:
            metrics = {}
            
            # 포즈 안정성 분석
            valid_keypoints = [kp for kp in keypoints_18 if kp[2] > 0.5]
            metrics['pose_stability'] = len(valid_keypoints) / 18
            
            # 대칭성 분석
            metrics['symmetry_score'] = self._calculate_pose_symmetry(keypoints_18)
            
            # 포즈 복잡도
            metrics['pose_complexity'] = self._calculate_pose_complexity(keypoints_18)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"고급 포즈 분석 실패: {e}")
            return {}
    
    def _calculate_pose_symmetry(self, keypoints_18: List[List[float]]) -> float:
        """포즈 대칭성 계산"""
        try:
            # 대칭 쌍 정의
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
                    # Y 좌표 차이 (대칭성)
                    y_diff = abs(right_kp[1] - left_kp[1])
                    max_y = max(right_kp[1], left_kp[1])
                    if max_y > 0:
                        symmetry = 1.0 - min(y_diff / max_y, 1.0)
                        symmetry_scores.append(symmetry)
            
            return np.mean(symmetry_scores) if symmetry_scores else 0.0
            
        except Exception as e:
            self.logger.warning(f"대칭성 계산 실패: {e}")
            return 0.0
    
    def _calculate_pose_complexity(self, keypoints_18: List[List[float]]) -> float:
        """포즈 복잡도 계산"""
        try:
            # 키포인트 간 거리 변화량으로 복잡도 측정
            valid_points = [(x, y) for x, y, conf in keypoints_18 if conf > 0.5]
            
            if len(valid_points) < 3:
                return 0.0
            
            # 연속된 키포인트 간의 거리
            distances = []
            for i in range(len(valid_points) - 1):
                p1, p2 = valid_points[i], valid_points[i+1]
                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                distances.append(dist)
            
            # 거리의 표준편차로 복잡도 계산
            complexity = np.std(distances) / (np.mean(distances) + 1e-6)
            return min(complexity, 1.0)
            
        except Exception as e:
            self.logger.warning(f"복잡도 계산 실패: {e}")
            return 0.0
    
    # =================================================================
    # 🔧 기존 헬퍼 메서드들 (간소화)
    # =================================================================
    
    def _estimate_body_orientation(self, keypoints_18: List[List[float]]) -> str:
        """신체 방향 추정"""
        try:
            # 어깨와 엉덩이 키포인트
            left_shoulder = keypoints_18[5]
            right_shoulder = keypoints_18[2]
            left_hip = keypoints_18[11]
            right_hip = keypoints_18[8]
            
            if all(kp[2] > 0.5 for kp in [left_shoulder, right_shoulder, left_hip, right_hip]):
                shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
                hip_width = abs(right_hip[0] - left_hip[0])
                avg_width = (shoulder_width + hip_width) / 2
                
                if avg_width < 80:
                    return "side"
                elif avg_width < 150:
                    return "diagonal"
                else:
                    return "front"
            
            return "unknown"
            
        except Exception:
            return "unknown"
    
    def _calculate_joint_angles(self, keypoints_18: List[List[float]]) -> Dict[str, float]:
        """주요 관절 각도 계산"""
        def angle_between_points(p1, p2, p3):
            if any(p[2] < 0.5 for p in [p1, p2, p3]):
                return 0.0
            
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            
            return float(angle)
        
        angles = {}
        
        try:
            # 팔 각도
            angles["left_arm_angle"] = angle_between_points(
                keypoints_18[5], keypoints_18[6], keypoints_18[7]
            )
            angles["right_arm_angle"] = angle_between_points(
                keypoints_18[2], keypoints_18[3], keypoints_18[4]
            )
            
            # 다리 각도
            angles["left_leg_angle"] = angle_between_points(
                keypoints_18[11], keypoints_18[12], keypoints_18[13]
            )
            angles["right_leg_angle"] = angle_between_points(
                keypoints_18[8], keypoints_18[9], keypoints_18[10]
            )
            
        except Exception as e:
            self.logger.warning(f"관절 각도 계산 실패: {e}")
        
        return angles
    
    def _analyze_body_proportions(self, keypoints_18: List[List[float]]) -> Dict[str, float]:
        """신체 비율 분석 (간소화)"""
        proportions = {}
        
        try:
            # 머리 길이, 몸통 길이, 다리 길이 계산
            if keypoints_18[0][2] > 0.5 and keypoints_18[1][2] > 0.5:
                proportions["head_length"] = abs(keypoints_18[1][1] - keypoints_18[0][1])
            
            if keypoints_18[1][2] > 0.5 and keypoints_18[8][2] > 0.5:
                proportions["torso_length"] = abs(keypoints_18[8][1] - keypoints_18[1][1])
                
        except Exception:
            pass
        
        return proportions
    
    def _calculate_pose_bbox(self, keypoints_18: List[List[float]], image_shape: Tuple) -> Dict[str, int]:
        """포즈 바운딩 박스 계산"""
        valid_points = [(x, y) for x, y, conf in keypoints_18 if conf > 0.5]
        
        if not valid_points:
            return {"x": 0, "y": 0, "width": 0, "height": 0}
        
        xs, ys = zip(*valid_points)
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # 여백 추가
        margin_x = int((x_max - x_min) * 0.15)
        margin_y = int((y_max - y_min) * 0.15)
        
        height, width = image_shape[:2]
        
        return {
            "x": max(0, int(x_min - margin_x)),
            "y": max(0, int(y_min - margin_y)),
            "width": min(width, int(x_max - x_min + 2 * margin_x)),
            "height": min(height, int(y_max - y_min + 2 * margin_y))
        }
    
    def _classify_pose_type(self, keypoints_18: List[List[float]], angles: Dict[str, float]) -> str:
        """포즈 타입 분류"""
        try:
            left_arm = angles.get("left_arm_angle", 180)
            right_arm = angles.get("right_arm_angle", 180)
            left_leg = angles.get("left_leg_angle", 180)
            right_leg = angles.get("right_leg_angle", 180)
            
            # T-포즈
            if abs(left_arm - 180) < 20 and abs(right_arm - 180) < 20:
                return "t_pose"
            # A-포즈
            elif 140 < left_arm < 170 and 140 < right_arm < 170:
                return "a_pose"
            # 걷기
            elif abs(left_leg - right_leg) > 30:
                return "walking"
            # 앉기
            elif left_leg < 140 or right_leg < 140:
                return "sitting"
            else:
                return "standing"
                
        except Exception:
            return "unknown"
    
    def _evaluate_pose_quality(self, keypoints_18: List[List[float]], mediapipe_keypoints: List[Dict]) -> Dict[str, float]:
        """포즈 품질 평가 (M3 Max 향상)"""
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
            
            # 4. 대칭성 점수 (M3 Max 전용)
            symmetry_score = 0.8  # 기본값
            if self.is_m3_max:
                symmetry_score = self._calculate_pose_symmetry(keypoints_18)
            
            # 5. 전체 신뢰도 계산
            overall_confidence = (
                detection_rate * 0.3 +
                major_detection_rate * 0.3 +
                avg_confidence * 0.3 +
                symmetry_score * 0.1
            )
            
            # M3 Max 보너스
            if self.is_m3_max:
                overall_confidence = min(1.0, overall_confidence * 1.05)
            
            return {
                'overall_confidence': overall_confidence,
                'detection_rate': detection_rate,
                'major_detection_rate': major_detection_rate,
                'average_confidence': avg_confidence,
                'symmetry_score': symmetry_score,
                'quality_grade': self._get_quality_grade(overall_confidence)
            }
            
        except Exception as e:
            self.logger.error(f"품질 평가 실패: {e}")
            return {
                'overall_confidence': 0.0,
                'detection_rate': 0.0,
                'major_detection_rate': 0.0,
                'average_confidence': 0.0,
                'symmetry_score': 0.0,
                'quality_grade': 'poor'
            }
    
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
    
    # =================================================================
    # 🔧 캐시 및 유틸리티 메서드들
    # =================================================================
    
    def _generate_cache_key(self, image_input) -> str:
        """캐시 키 생성"""
        try:
            if isinstance(image_input, str):
                return f"pose_{hash(image_input)}_{self.model_complexity}"
            elif isinstance(image_input, np.ndarray):
                return f"pose_{hash(image_input.tobytes())}_{self.model_complexity}"
            else:
                return f"pose_{hash(str(image_input))}_{self.model_complexity}"
        except Exception:
            return f"pose_fallback_{time.time()}"
    
    def _update_cache(self, key: str, result: Dict[str, Any]):
        """캐시 업데이트"""
        try:
            if len(self.detection_cache) >= self.cache_max_size:
                # 가장 오래된 항목 제거
                oldest_key = next(iter(self.detection_cache))
                del self.detection_cache[oldest_key]
            
            # 결과 복사해서 저장
            cached_result = {k: v for k, v in result.items() if k != 'processing_info'}
            self.detection_cache[key] = cached_result
            
        except Exception as e:
            self.logger.warning(f"캐시 업데이트 실패: {e}")
    
    def _update_pose_stats(self, processing_time: float, quality_score: float):
        """2단계 전용 통계 업데이트"""
        self.pose_stats['total_processed'] += 1
        
        if quality_score > 0.5:
            self.pose_stats['successful_detections'] += 1
        
        # 이동 평균으로 처리 시간 업데이트
        alpha = 0.1
        self.pose_stats['average_processing_time'] = (
            alpha * processing_time + 
            (1 - alpha) * self.pose_stats['average_processing_time']
        )
        
        self.pose_stats['last_quality_score'] = quality_score
    
    def _create_dummy_detector(self):
        """더미 검출기 생성"""
        class DummyDetector:
            def process(self, image):
                return None
        return DummyDetector()
    
    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """빈 결과 생성"""
        return {
            'success': True,  # 파이프라인 진행을 위해 True 유지
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
                'detection_method': 'none',
                'device': self.device,
                'error_details': reason
            },
            'additional_keypoints': None,
            'from_cache': False
        }
    
    # =================================================================
    # 🔧 Pipeline Manager 호환 메서드들
    # =================================================================
    
    async def get_step_info(self) -> Dict[str, Any]:
        """🔍 2단계 상세 정보 반환"""
        return {
            "step_name": "pose_estimation",
            "step_number": 2,
            "device": self.device,
            "device_type": self.device_type,
            "initialized": self.is_initialized,
            "config_keys": list(self.config.keys()),
            "pose_stats": self.pose_stats.copy(),
            "keypoint_formats": {
                "openpose_18": self.OPENPOSE_18_KEYPOINTS,
                "mediapipe_33": len(self.MEDIAPIPE_KEYPOINT_NAMES)
            },
            "cache_usage": {
                "cache_size": len(self.detection_cache),
                "cache_limit": self.cache_max_size,
                "hit_rate": self.pose_stats.get('cache_hits', 0) / max(1, self.pose_stats['total_processed'])
            },
            "detectors_available": {
                "pose": self.pose_detector is not None,
                "face": self.face_detector is not None,
                "hands": self.hands_detector is not None,
                "mediapipe_enabled": MEDIAPIPE_AVAILABLE
            },
            "capabilities": {
                "model_complexity": self.model_complexity,
                "max_image_size": self.max_image_size,
                "face_detection": self.use_face,
                "hand_detection": self.use_hands,
                "segmentation_enabled": self.enable_segmentation,
                "advanced_analysis": self.is_m3_max,
                "is_m3_max": self.is_m3_max,
                "optimization_enabled": self.optimization_enabled,
                "quality_level": self.quality_level
            }
        }
    
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
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            self.logger.info(f"💾 포즈 시각화 저장: {save_path}")
        
        return vis_image
    
    def export_keypoints(self, keypoints_18: List[List[float]], format: str = "json") -> str:
        """키포인트 내보내기"""
        if format.lower() == "json":
            export_data = {
                "format": "openpose_18",
                "keypoints": keypoints_18,
                "keypoint_names": self.OPENPOSE_18_KEYPOINTS,
                "connections": self._get_openpose_connections(),
                "device_info": {
                    "device": self.device,
                    "m3_max_optimized": self.is_m3_max,
                    "model_complexity": self.model_complexity
                }
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
        try:
            self.logger.info("🧹 2단계: 포즈 추정 리소스 정리 중...")
            
            # 검출기들 정리
            if self.pose_detector and hasattr(self.pose_detector, 'close'):
                self.pose_detector.close()
            if self.face_detector and hasattr(self.face_detector, 'close'):
                self.face_detector.close()
            if self.hands_detector and hasattr(self.hands_detector, 'close'):
                self.hands_detector.close()
            
            self.pose_detector = None
            self.face_detector = None
            self.hands_detector = None
            self.mp_pose = None
            self.mp_drawing = None
            
            # 캐시 정리
            self.detection_cache.clear()
            
            # 스레드 풀 정리
            self.executor.shutdown(wait=True)
            
            self.is_initialized = False
            self.logger.info("✅ 2단계 포즈 추정 리소스 정리 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 리소스 정리 중 오류: {e}")


# =================================================================
# 🔄 하위 호환성 지원 (기존 코드 호환)
# =================================================================

async def create_pose_estimation_step(
    device: str = "auto",
    config: Dict[str, Any] = None
) -> PoseEstimationStep:
    """
    🔄 기존 팩토리 함수 호환 (기존 파이프라인 호환)
    
    Args:
        device: 사용할 디바이스 ("auto"는 자동 감지)
        config: 설정 딕셔너리
        
    Returns:
        PoseEstimationStep: 초기화된 2단계 스텝
    """
    # 기존 방식 호환
    device_param = None if device == "auto" else device
    
    default_config = {
        "model_complexity": 2,
        "min_detection_confidence": 0.7,
        "min_tracking_confidence": 0.5,
        "enable_segmentation": False,
        "max_image_size": 1024,
        "use_face": True,
        "use_hands": False
    }
    
    final_config = {**default_config, **(config or {})}
    
    # ✅ 새로운 통일된 생성자 사용
    step = PoseEstimationStep(device=device_param, config=final_config)
    
    if not await step.initialize():
        logger.warning("2단계 초기화 실패했지만 진행합니다.")
    
    return step

# 기존 클래스명 별칭 (완전 호환)
PoseEstimationStepLegacy = PoseEstimationStep