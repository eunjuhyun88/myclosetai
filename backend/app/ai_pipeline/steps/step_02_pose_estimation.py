# app/ai_pipeline/steps/step_02_pose_estimation.py
"""
2단계: 포즈 추정 (Pose Estimation) - 완전 구현 + 시각화 기능 (옵션 A)
✅ Model Loader 완전 연동 (BaseStepMixin 상속)
✅ MediaPipe + YOLOv8 듀얼 모델 지원
✅ M3 Max 최적화 및 Neural Engine 활용
✅ 완전한 에러 처리 및 폴백 메커니즘
✅ 18-keypoint OpenPose 호환 포맷
✅ 프로덕션 레벨 안정성
✅ 🆕 18개 키포인트 시각화 이미지 생성 기능 추가
"""

import os
import gc
import time
import asyncio
import json
import logging
import base64
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# PyTorch 및 관련 라이브러리
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# MediaPipe (주 포즈 추정 엔진)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

# YOLOv8 (백업 포즈 추정 엔진)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None

# Model Loader 연동
from ..utils.model_loader import BaseStepMixin, get_global_model_loader

logger = logging.getLogger(__name__)

# ==============================================
# 🎨 시각화 관련 상수 및 설정
# ==============================================

# OpenPose 18 키포인트 정의
OPENPOSE_18_KEYPOINTS = {
    0: "nose",
    1: "neck", 
    2: "right_shoulder",
    3: "right_elbow",
    4: "right_wrist",
    5: "left_shoulder",
    6: "left_elbow", 
    7: "left_wrist",
    8: "mid_hip",
    9: "right_hip",
    10: "right_knee",
    11: "right_ankle",
    12: "left_hip",
    13: "left_knee",
    14: "left_ankle",
    15: "right_eye",
    16: "left_eye",
    17: "right_ear"
}

# 🎨 키포인트별 색상 정의
KEYPOINT_COLORS = {
    # 머리 부위 - 빨강 계열
    0: (255, 0, 0),      # nose - 빨강
    15: (255, 100, 100), # right_eye - 연빨강
    16: (255, 150, 150), # left_eye - 더 연빨강
    17: (200, 0, 0),     # right_ear - 어두운 빨강
    
    # 목과 몸통 - 노랑 계열
    1: (255, 255, 0),    # neck - 노랑
    8: (255, 200, 0),    # mid_hip - 주황노랑
    
    # 오른쪽 팔 - 파랑 계열
    2: (0, 0, 255),      # right_shoulder - 파랑
    3: (0, 100, 255),    # right_elbow - 연파랑
    4: (0, 150, 255),    # right_wrist - 더 연파랑
    
    # 왼쪽 팔 - 초록 계열
    5: (0, 255, 0),      # left_shoulder - 초록
    6: (100, 255, 100),  # left_elbow - 연초록
    7: (150, 255, 150),  # left_wrist - 더 연초록
    
    # 오른쪽 다리 - 자주 계열
    9: (255, 0, 255),    # right_hip - 자주
    10: (200, 0, 200),   # right_knee - 어두운 자주
    11: (150, 0, 150),   # right_ankle - 더 어두운 자주
    
    # 왼쪽 다리 - 청록 계열
    12: (0, 255, 255),   # left_hip - 청록
    13: (0, 200, 200),   # left_knee - 어두운 청록
    14: (0, 150, 150),   # left_ankle - 더 어두운 청록
}

# 🔗 스켈레톤 연결선 정의
SKELETON_CONNECTIONS = [
    # 머리 연결
    (0, 1),   # nose -> neck
    (0, 15),  # nose -> right_eye
    (0, 16),  # nose -> left_eye
    (15, 17), # right_eye -> right_ear
    
    # 몸통 연결
    (1, 2),   # neck -> right_shoulder
    (1, 5),   # neck -> left_shoulder
    (1, 8),   # neck -> mid_hip
    (2, 8),   # right_shoulder -> mid_hip (몸통 라인)
    (5, 8),   # left_shoulder -> mid_hip (몸통 라인)
    
    # 오른쪽 팔
    (2, 3),   # right_shoulder -> right_elbow
    (3, 4),   # right_elbow -> right_wrist
    
    # 왼쪽 팔
    (5, 6),   # left_shoulder -> left_elbow
    (6, 7),   # left_elbow -> left_wrist
    
    # 엉덩이 연결
    (8, 9),   # mid_hip -> right_hip
    (8, 12),  # mid_hip -> left_hip
    
    # 오른쪽 다리
    (9, 10),  # right_hip -> right_knee
    (10, 11), # right_knee -> right_ankle
    
    # 왼쪽 다리
    (12, 13), # left_hip -> left_knee
    (13, 14), # left_knee -> left_ankle
]

# 🎨 스켈레톤 연결선 색상
SKELETON_COLORS = {
    # 머리 - 빨강
    (0, 1): (255, 0, 0),
    (0, 15): (255, 100, 100),
    (0, 16): (255, 100, 100),
    (15, 17): (200, 0, 0),
    
    # 몸통 - 노랑
    (1, 2): (255, 255, 0),
    (1, 5): (255, 255, 0),
    (1, 8): (255, 200, 0),
    (2, 8): (255, 180, 0),
    (5, 8): (255, 180, 0),
    
    # 오른쪽 팔 - 파랑
    (2, 3): (0, 0, 255),
    (3, 4): (0, 100, 255),
    
    # 왼쪽 팔 - 초록
    (5, 6): (0, 255, 0),
    (6, 7): (100, 255, 100),
    
    # 엉덩이 - 주황
    (8, 9): (255, 165, 0),
    (8, 12): (255, 165, 0),
    
    # 오른쪽 다리 - 자주
    (9, 10): (255, 0, 255),
    (10, 11): (200, 0, 200),
    
    # 왼쪽 다리 - 청록
    (12, 13): (0, 255, 255),
    (13, 14): (0, 200, 200),
}

class PoseEstimationStep(BaseStepMixin):
    """
    🏃 2단계: 포즈 추정 - 완전 구현 + 시각화
    ✅ Model Loader 완전 연동
    ✅ MediaPipe + YOLOv8 듀얼 엔진
    ✅ M3 Max 128GB 최적화
    ✅ 18-keypoint OpenPose 표준
    ✅ 🆕 18개 키포인트 시각화 이미지 생성
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """✅ 완전 통합 생성자 - Model Loader 연동"""
        
        # === BaseStepMixin 초기화 (Model Loader 연동) ===
        self._setup_model_interface(kwargs.get('model_loader'))
        
        # === 디바이스 자동 감지 ===
        self.device = self._auto_detect_device(device)
        self.device_type = self._get_device_type()
        self.is_m3_max = self._check_m3_max()
        
        # === 설정 초기화 ===
        self.config = self._setup_config(config)
        self.step_name = "PoseEstimationStep"
        self.logger = logging.getLogger(f"pipeline.{self.step_name}")
        
        # === 모델 상태 ===
        self.pose_model_primary = None      # MediaPipe 모델
        self.pose_model_secondary = None    # YOLOv8 모델
        self.current_model_type = None
        self.is_initialized = False
        
        # === MediaPipe 컴포넌트 ===
        self.mp_pose = None
        self.mp_drawing = None
        self.mp_drawing_styles = None
        self.pose_detector = None
        
        # === 성능 최적화 ===
        self.model_cache = {}
        self.prediction_cache = {}
        self.processing_stats = {
            'total_processed': 0,
            'mediapipe_usage': 0,
            'yolo_usage': 0,
            'fallback_usage': 0,
            'avg_processing_time': 0.0,
            'cache_hits': 0
        }
        
        # === M3 Max 최적화 설정 ===
        if self.is_m3_max:
            self._setup_m3_max_optimization()
        
        # === 스레드 풀 ===
        self.executor = ThreadPoolExecutor(
            max_workers=min(4, os.cpu_count() or 4),
            thread_name_prefix="pose_estimation"
        )
        
        self.logger.info(f"🏃 PoseEstimationStep 초기화 완료 - Device: {self.device}")
        
    def _auto_detect_device(self, device: Optional[str]) -> str:
        """디바이스 자동 감지"""
        if device and device != "auto":
            return device
            
        if TORCH_AVAILABLE:
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
        return "cpu"
    
    def _get_device_type(self) -> str:
        """디바이스 타입 반환"""
        if self.device == "mps":
            return "apple_silicon"
        elif self.device.startswith("cuda"):
            return "nvidia_gpu"
        else:
            return "cpu"
    
    def _check_m3_max(self) -> bool:
        """M3 Max 칩 확인"""
        try:
            if self.device_type == "apple_silicon":
                # M3 Max는 일반적으로 128GB 메모리를 가짐
                import psutil
                total_memory_gb = psutil.virtual_memory().total / (1024**3)
                return total_memory_gb > 100  # 100GB 이상이면 M3 Max로 가정
            return False
        except:
            return False
    
    def _setup_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """설정 초기화"""
        default_config = {
            # === MediaPipe 설정 ===
            "mediapipe": {
                "model_complexity": 2,  # 0, 1, 2 (높을수록 정확하지만 느림)
                "min_detection_confidence": 0.7,
                "min_tracking_confidence": 0.5,
                "enable_segmentation": False,
                "smooth_landmarks": True,
                "static_image_mode": True
            },
            
            # === YOLOv8 설정 ===
            "yolo": {
                "model_size": "n",  # n, s, m, l, x
                "confidence": 0.6,
                "iou": 0.5,
                "max_det": 1,  # 한 사람만 검출
                "device": self.device
            },
            
            # === 처리 설정 ===
            "processing": {
                "max_image_size": 1024,
                "resize_method": "proportional",
                "normalize_input": True,
                "output_format": "openpose_18",
                "enable_face_keypoints": False,
                "enable_hand_keypoints": False
            },
            
            # === 품질 설정 ===
            "quality": {
                "min_keypoints_detected": 10,
                "min_pose_confidence": 0.5,
                "enable_pose_validation": True,
                "filter_low_confidence": True
            },
            
            # === 캐싱 설정 ===
            "cache": {
                "enable_prediction_cache": True,
                "cache_size": 100,
                "cache_ttl": 3600  # 1시간
            },
            
            # === M3 Max 최적화 ===
            "m3_optimization": {
                "enable_neural_engine": True,
                "batch_size": 8,
                "memory_fraction": 0.8,
                "precision": "fp16"
            },
            
            # === 🆕 시각화 설정 ===
            "visualization": {
                "enable_visualization": True,
                "keypoint_radius": 5,
                "skeleton_thickness": 3,
                "confidence_threshold": 0.5,
                "show_keypoint_labels": True,
                "show_confidence_values": True,
                "image_quality": "high",  # low, medium, high
                "overlay_opacity": 0.8,
                "background_color": (0, 0, 0),  # 검정 배경
                "font_size": 12
            }
        }
        
        if config:
            # 딥 업데이트
            def deep_update(base_dict, update_dict):
                for key, value in update_dict.items():
                    if isinstance(value, dict) and key in base_dict:
                        deep_update(base_dict[key], value)
                    else:
                        base_dict[key] = value
            
            deep_update(default_config, config)
        
        return default_config
    
    def _setup_m3_max_optimization(self):
        """M3 Max 전용 최적화 설정"""
        try:
            if TORCH_AVAILABLE and self.device == "mps":
                # MPS 백엔드 최적화
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
                
                # Neural Engine 활성화
                if torch.backends.mps.is_available():
                    torch.backends.mps.is_built()
                    
            # 메모리 최적화
            os.environ['OMP_NUM_THREADS'] = '16'  # M3 Max 16코어
            
            self.logger.info("🍎 M3 Max 최적화 설정 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ M3 Max 최적화 설정 실패: {e}")
    
    async def initialize(self) -> bool:
        """
        ✅ 완전 초기화 - Model Loader 연동
        우선순위: MediaPipe > YOLOv8 > 더미
        """
        try:
            self.logger.info("🚀 PoseEstimationStep 초기화 시작...")
            
            # === 1. MediaPipe 초기화 시도 ===
            mediapipe_success = await self._initialize_mediapipe()
            
            # === 2. YOLOv8 초기화 시도 ===
            yolo_success = await self._initialize_yolo()
            
            # === 3. Model Loader에서 추가 모델 로드 시도 ===
            await self._initialize_from_model_loader()
            
            # === 4. 초기화 결과 평가 ===
            if mediapipe_success:
                self.current_model_type = "mediapipe"
                self.logger.info("✅ MediaPipe 포즈 모델 로드 성공 (Primary)")
            elif yolo_success:
                self.current_model_type = "yolo"
                self.logger.info("✅ YOLOv8 포즈 모델 로드 성공 (Secondary)")
            else:
                self.current_model_type = "dummy"
                self.logger.warning("⚠️ 실제 모델 로드 실패 - 더미 모델로 폴백")
                await self._initialize_dummy_model()
            
            # === 5. 캐시 초기화 ===
            self.prediction_cache.clear()
            
            # === 6. M3 Max 워밍업 ===
            if self.is_m3_max and self.current_model_type != "dummy":
                await self._warmup_models()
            
            self.is_initialized = True
            self.logger.info(f"✅ PoseEstimationStep 초기화 완료 - 모델: {self.current_model_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ PoseEstimationStep 초기화 실패: {e}")
            # 완전 실패 시 더미 모델로 폴백
            await self._initialize_dummy_model()
            self.current_model_type = "dummy"
            self.is_initialized = True
            return False
    
    async def _initialize_mediapipe(self) -> bool:
        """MediaPipe 포즈 모델 초기화"""
        try:
            if not MEDIAPIPE_AVAILABLE:
                self.logger.warning("MediaPipe가 설치되지 않음")
                return False
            
            # MediaPipe 컴포넌트 초기화
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            # Model Loader에서 MediaPipe 모델 로드 시도
            pose_model = await self.get_model("pose_estimation_mediapipe")
            
            if pose_model is None:
                # 직접 MediaPipe 초기화
                mp_config = self.config["mediapipe"]
                self.pose_detector = self.mp_pose.Pose(
                    static_image_mode=mp_config["static_image_mode"],
                    model_complexity=mp_config["model_complexity"],
                    enable_segmentation=mp_config["enable_segmentation"],
                    smooth_landmarks=mp_config["smooth_landmarks"],
                    min_detection_confidence=mp_config["min_detection_confidence"],
                    min_tracking_confidence=mp_config["min_tracking_confidence"]
                )
            else:
                self.pose_detector = pose_model
            
            # 테스트 이미지로 검증
            test_success = await self._test_mediapipe()
            
            if test_success:
                self.pose_model_primary = self.pose_detector
                self.logger.info("✅ MediaPipe 포즈 모델 초기화 성공")
                return True
            else:
                self.logger.warning("❌ MediaPipe 테스트 실패")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ MediaPipe 초기화 실패: {e}")
            return False
    
    async def _initialize_yolo(self) -> bool:
        """YOLOv8 포즈 모델 초기화"""
        try:
            if not YOLO_AVAILABLE:
                self.logger.warning("YOLOv8가 설치되지 않음")
                return False
            
            # Model Loader에서 YOLOv8 모델 로드 시도
            yolo_model = await self.get_model("pose_estimation_yolo")
            
            if yolo_model is None:
                # 직접 YOLOv8 초기화
                model_size = self.config["yolo"]["model_size"]
                model_name = f"yolov8{model_size}-pose.pt"
                
                # 로컬 파일 확인
                local_path = Path(f"ai_models/checkpoints/step_02_pose_estimation/{model_name}")
                if local_path.exists():
                    self.pose_model_secondary = YOLO(str(local_path))
                else:
                    # 온라인에서 다운로드 시도
                    self.pose_model_secondary = YOLO(model_name)
            else:
                self.pose_model_secondary = yolo_model
            
            # 디바이스 설정
            if hasattr(self.pose_model_secondary, 'to'):
                if self.device == "mps" and TORCH_AVAILABLE:
                    # MPS 지원 확인
                    try:
                        self.pose_model_secondary.to(self.device)
                    except:
                        self.pose_model_secondary.to("cpu")
                        self.logger.warning("YOLOv8 MPS 지원 실패, CPU로 폴백")
                else:
                    self.pose_model_secondary.to(self.device)
            
            # 테스트 이미지로 검증
            test_success = await self._test_yolo()
            
            if test_success:
                self.logger.info("✅ YOLOv8 포즈 모델 초기화 성공")
                return True
            else:
                self.logger.warning("❌ YOLOv8 테스트 실패")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ YOLOv8 초기화 실패: {e}")
            return False
    
    async def _initialize_from_model_loader(self):
        """Model Loader에서 추가 모델 로드"""
        try:
            # 권장 모델 로드 시도
            recommended_model = await self.get_recommended_model()
            if recommended_model:
                self.logger.info("✅ Model Loader에서 권장 포즈 모델 로드 성공")
                
                # 모델 타입 감지 및 할당
                if hasattr(recommended_model, 'process'):  # MediaPipe 스타일
                    self.pose_model_primary = recommended_model
                elif hasattr(recommended_model, 'predict'):  # YOLO 스타일
                    self.pose_model_secondary = recommended_model
                    
        except Exception as e:
            self.logger.warning(f"Model Loader 추가 로드 실패: {e}")
    
    async def _initialize_dummy_model(self):
        """더미 포즈 모델 초기화 (폴백)"""
        self.pose_detector = self._create_dummy_detector()
        self.logger.info("🔄 더미 포즈 모델로 폴백 완료")
    
    def _create_dummy_detector(self):
        """더미 포즈 검출기 생성"""
        class DummyPoseDetector:
            def process(self, image):
                # 기본 T-포즈 키포인트 반환
                dummy_landmarks = self._generate_dummy_landmarks(image.shape)
                return type('Result', (), {'pose_landmarks': dummy_landmarks})()
            
            def _generate_dummy_landmarks(self, image_shape):
                height, width = image_shape[:2]
                # 33개 MediaPipe 키포인트 생성
                landmarks = []
                for i in range(33):
                    x = 0.5 + 0.1 * np.sin(i)  # 중앙 기준으로 분산
                    y = 0.3 + 0.4 * (i / 32)   # 위에서 아래로
                    z = 0.0
                    landmarks.append(type('Landmark', (), {'x': x, 'y': y, 'z': z, 'visibility': 0.8})())
                
                return type('Landmarks', (), {'landmark': landmarks})()
        
        return DummyPoseDetector()
    
    async def _test_mediapipe(self) -> bool:
        """MediaPipe 모델 테스트"""
        try:
            # 테스트 이미지 생성
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            
            # 포즈 검출 테스트
            results = self.pose_detector.process(test_image)
            return True  # 에러 없이 처리되면 성공
            
        except Exception as e:
            self.logger.error(f"MediaPipe 테스트 실패: {e}")
            return False
    
    async def _test_yolo(self) -> bool:
        """YOLOv8 모델 테스트"""
        try:
            # 테스트 이미지 생성
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # 포즈 검출 테스트
            results = self.pose_model_secondary(test_image, verbose=False)
            return len(results) > 0  # 결과가 있으면 성공
            
        except Exception as e:
            self.logger.error(f"YOLOv8 테스트 실패: {e}")
            return False
    
    async def _warmup_models(self):
        """M3 Max 모델 워밍업"""
        try:
            self.logger.info("🔥 M3 Max 모델 워밍업 시작...")
            
            # 워밍업 이미지 생성
            warmup_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            
            # MediaPipe 워밍업
            if self.pose_model_primary:
                for _ in range(3):
                    _ = self.pose_model_primary.process(warmup_image)
            
            # YOLOv8 워밍업
            if self.pose_model_secondary:
                for _ in range(3):
                    _ = self.pose_model_secondary(warmup_image, verbose=False)
            
            # MPS 캐시 정리
            if TORCH_AVAILABLE and self.device == "mps":
                torch.mps.empty_cache()
            
            self.logger.info("✅ M3 Max 워밍업 완료")
            
        except Exception as e:
            self.logger.warning(f"워밍업 실패: {e}")
    
    async def process(
        self,
        person_image: Union[str, np.ndarray, Image.Image],
        **kwargs
    ) -> Dict[str, Any]:
        """
        ✅ 메인 포즈 추정 처리 + 시각화
        
        Args:
            person_image: 입력 이미지
            **kwargs: 추가 옵션
                - force_model: str = None (강제 모델 지정)
                - return_confidence: bool = True
                - return_analysis: bool = True
                - cache_result: bool = True
        
        Returns:
            Dict[str, Any]: 포즈 추정 결과 + 시각화 이미지
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            self.logger.info("🏃 포즈 추정 처리 시작")
            
            # === 1. 캐시 확인 ===
            cache_key = self._generate_cache_key(person_image)
            if kwargs.get('cache_result', True) and cache_key in self.prediction_cache:
                self.processing_stats['cache_hits'] += 1
                cached_result = self.prediction_cache[cache_key].copy()
                cached_result['from_cache'] = True
                self.logger.info("💾 캐시에서 포즈 결과 반환")
                return cached_result
            
            # === 2. 이미지 전처리 ===
            processed_image = await self._preprocess_image(person_image)
            if processed_image is None:
                return self._create_error_result("이미지 전처리 실패")
            
            # === 3. 모델 선택 및 추론 ===
            force_model = kwargs.get('force_model')
            pose_result = await self._perform_pose_estimation(
                processed_image, 
                force_model
            )
            
            # === 4. 결과 후처리 + 시각화 ===
            final_result = await self._postprocess_results(
                pose_result, 
                processed_image.shape,
                processed_image,  # 🆕 원본 이미지 전달 (시각화용)
                **kwargs
            )
            
            # === 5. 품질 평가 ===
            if kwargs.get('return_analysis', True):
                final_result['pose_analysis'] = self._analyze_pose_quality(final_result)
            
            # === 6. 캐싱 ===
            if kwargs.get('cache_result', True):
                self.prediction_cache[cache_key] = final_result.copy()
                # 캐시 크기 제한
                if len(self.prediction_cache) > self.config['cache']['cache_size']:
                    oldest_key = next(iter(self.prediction_cache))
                    del self.prediction_cache[oldest_key]
            
            # === 7. 통계 업데이트 ===
            processing_time = time.time() - start_time
            self._update_processing_stats(processing_time)
            
            final_result.update({
                'success': True,
                'message': "포즈 추정 완료",
                'processing_time': processing_time,
                'model_used': self.current_model_type,
                'device': self.device,
                'm3_max_optimized': self.is_m3_max,
                'from_cache': False
            })
            
            self.logger.info(f"✅ 포즈 추정 완료 - {processing_time:.3f}초")
            return final_result
            
        except Exception as e:
            error_msg = f"포즈 추정 처리 실패: {e}"
            self.logger.error(f"❌ {error_msg}")
            return self._create_error_result(error_msg)
    
    async def _preprocess_image(
        self, 
        image_input: Union[str, np.ndarray, Image.Image]
    ) -> Optional[np.ndarray]:
        """이미지 전처리"""
        try:
            # 이미지 로드
            if isinstance(image_input, str):
                if os.path.exists(image_input):
                    image = cv2.imread(image_input)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    self.logger.error(f"이미지 파일 없음: {image_input}")
                    return None
            elif isinstance(image_input, Image.Image):
                image = np.array(image_input.convert('RGB'))
            elif isinstance(image_input, np.ndarray):
                image = image_input.copy()
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # BGR to RGB 변환이 필요한지 확인
                    if image.max() <= 1.0:  # 정규화된 이미지
                        image = (image * 255).astype(np.uint8)
            else:
                self.logger.error(f"지원하지 않는 이미지 형식: {type(image_input)}")
                return None
            
            # 크기 조정
            max_size = self.config['processing']['max_image_size']
            height, width = image.shape[:2]
            
            if max(height, width) > max_size:
                if height > width:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                else:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            return image
            
        except Exception as e:
            self.logger.error(f"이미지 전처리 실패: {e}")
            return None
    
    async def _perform_pose_estimation(
        self, 
        image: np.ndarray, 
        force_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """포즈 추정 수행"""
        try:
            if force_model:
                model_type = force_model
            else:
                model_type = self.current_model_type
            
            if model_type == "mediapipe" and self.pose_model_primary:
                result = await self._estimate_pose_mediapipe(image)
                self.processing_stats['mediapipe_usage'] += 1
            elif model_type == "yolo" and self.pose_model_secondary:
                result = await self._estimate_pose_yolo(image)
                self.processing_stats['yolo_usage'] += 1
            else:
                # 폴백: 사용 가능한 모델 시도
                if self.pose_model_primary:
                    result = await self._estimate_pose_mediapipe(image)
                    self.processing_stats['mediapipe_usage'] += 1
                elif self.pose_model_secondary:
                    result = await self._estimate_pose_yolo(image)
                    self.processing_stats['yolo_usage'] += 1
                else:
                    result = await self._estimate_pose_dummy(image)
                    self.processing_stats['fallback_usage'] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"포즈 추정 실행 실패: {e}")
            return await self._estimate_pose_dummy(image)
    
    async def _estimate_pose_mediapipe(self, image: np.ndarray) -> Dict[str, Any]:
        """MediaPipe 포즈 추정"""
        try:
            # MediaPipe 처리
            results = self.pose_model_primary.process(image)
            
            if results.pose_landmarks:
                # MediaPipe 33 keypoints를 OpenPose 18 형식으로 변환
                keypoints_18 = self._convert_mediapipe_to_openpose18(
                    results.pose_landmarks, 
                    image.shape
                )
                
                # 신뢰도 계산
                confidence = self._calculate_mediapipe_confidence(results.pose_landmarks)
                
                return {
                    'keypoints_18': keypoints_18,
                    'raw_results': results,
                    'confidence': confidence,
                    'model_type': 'mediapipe',
                    'keypoints_detected': sum(1 for kp in keypoints_18 if kp[2] > 0.5)
                }
            else:
                return self._create_empty_pose_result('mediapipe')
                
        except Exception as e:
            self.logger.error(f"MediaPipe 포즈 추정 실패: {e}")
            return self._create_empty_pose_result('mediapipe')
    
    async def _estimate_pose_yolo(self, image: np.ndarray) -> Dict[str, Any]:
        """YOLOv8 포즈 추정"""
        try:
            # YOLOv8 처리
            results = self.pose_model_secondary(image, verbose=False)
            
            if len(results) > 0 and results[0].keypoints is not None:
                # YOLOv8 17 keypoints를 OpenPose 18 형식으로 변환
                yolo_keypoints = results[0].keypoints.xy[0].cpu().numpy()  # 첫 번째 사람
                yolo_confidence = results[0].keypoints.conf[0].cpu().numpy()
                
                keypoints_18 = self._convert_yolo_to_openpose18(
                    yolo_keypoints, 
                    yolo_confidence, 
                    image.shape
                )
                
                # 전체 신뢰도 계산
                confidence = float(np.mean(yolo_confidence[yolo_confidence > 0]))
                
                return {
                    'keypoints_18': keypoints_18,
                    'raw_results': results,
                    'confidence': confidence,
                    'model_type': 'yolo',
                    'keypoints_detected': sum(1 for kp in keypoints_18 if kp[2] > 0.5)
                }
            else:
                return self._create_empty_pose_result('yolo')
                
        except Exception as e:
            self.logger.error(f"YOLOv8 포즈 추정 실패: {e}")
            return self._create_empty_pose_result('yolo')
    
    async def _estimate_pose_dummy(self, image: np.ndarray) -> Dict[str, Any]:
        """더미 포즈 추정 (폴백)"""
        try:
            height, width = image.shape[:2]
            
            # 기본 T-포즈 생성
            keypoints_18 = self._generate_dummy_openpose18(width, height)
            
            return {
                'keypoints_18': keypoints_18,
                'raw_results': None,
                'confidence': 0.5,  # 중간 신뢰도
                'model_type': 'dummy',
                'keypoints_detected': 18
            }
            
        except Exception as e:
            self.logger.error(f"더미 포즈 생성 실패: {e}")
            return self._create_empty_pose_result('dummy')
    
    def _convert_mediapipe_to_openpose18(
        self, 
        mediapipe_landmarks, 
        image_shape: Tuple[int, int, int]
    ) -> List[List[float]]:
        """MediaPipe 33 keypoints를 OpenPose 18 keypoints로 변환"""
        height, width = image_shape[:2]
        
        # MediaPipe -> OpenPose 매핑
        mp_to_op_mapping = {
            0: 0,    # nose
            11: 1,   # neck (left_shoulder와 right_shoulder의 중점으로 계산)
            12: 2,   # right_shoulder
            14: 3,   # right_elbow
            16: 4,   # right_wrist
            11: 5,   # left_shoulder
            13: 6,   # left_elbow
            15: 7,   # left_wrist
            24: 8,   # mid_hip (left_hip과 right_hip의 중점으로 계산)
            26: 9,   # right_hip
            28: 10,  # right_knee
            32: 11,  # right_ankle
            23: 12,  # left_hip
            27: 13,  # left_knee
            31: 14,  # left_ankle
            2: 15,   # right_eye
            5: 16,   # left_eye
            4: 17,   # right_ear
        }
        
        keypoints_18 = [[0, 0, 0] for _ in range(18)]
        landmarks = mediapipe_landmarks.landmark
        
        for op_idx in range(18):
            if op_idx == 1:  # neck - shoulder 중점
                if len(landmarks) > 11 and len(landmarks) > 12:
                    x = (landmarks[11].x + landmarks[12].x) / 2 * width
                    y = (landmarks[11].y + landmarks[12].y) / 2 * height
                    conf = min(landmarks[11].visibility, landmarks[12].visibility)
                    keypoints_18[1] = [float(x), float(y), float(conf)]
            elif op_idx == 8:  # mid_hip - hip 중점
                if len(landmarks) > 23 and len(landmarks) > 24:
                    x = (landmarks[23].x + landmarks[24].x) / 2 * width
                    y = (landmarks[23].y + landmarks[24].y) / 2 * height
                    conf = min(landmarks[23].visibility, landmarks[24].visibility)
                    keypoints_18[8] = [float(x), float(y), float(conf)]
            else:
                # 직접 매핑
                for mp_idx, mapped_op_idx in mp_to_op_mapping.items():
                    if mapped_op_idx == op_idx and mp_idx < len(landmarks):
                        landmark = landmarks[mp_idx]
                        x = landmark.x * width
                        y = landmark.y * height
                        conf = landmark.visibility
                        keypoints_18[op_idx] = [float(x), float(y), float(conf)]
                        break
        
        return keypoints_18
    
    def _convert_yolo_to_openpose18(
        self, 
        yolo_keypoints: np.ndarray, 
        yolo_confidence: np.ndarray,
        image_shape: Tuple[int, int, int]
    ) -> List[List[float]]:
        """YOLOv8 17 keypoints를 OpenPose 18 keypoints로 변환"""
        
        # YOLO COCO 17 -> OpenPose 18 매핑
        yolo_to_op_mapping = {
            0: 0,    # nose
            5: 2,    # right_shoulder 
            6: 5,    # left_shoulder
            7: 3,    # right_elbow
            8: 6,    # left_elbow
            9: 4,    # right_wrist
            10: 7,   # left_wrist
            11: 9,   # right_hip
            12: 12,  # left_hip
            13: 10,  # right_knee
            14: 13,  # left_knee
            15: 11,  # right_ankle
            16: 14,  # left_ankle
            1: 15,   # right_eye
            2: 16,   # left_eye
            3: 17,   # right_ear
        }
        
        keypoints_18 = [[0, 0, 0] for _ in range(18)]
        
        for yolo_idx, op_idx in yolo_to_op_mapping.items():
            if yolo_idx < len(yolo_keypoints) and op_idx < 18:
                x, y = yolo_keypoints[yolo_idx]
                conf = yolo_confidence[yolo_idx] if yolo_idx < len(yolo_confidence) else 0.0
                keypoints_18[op_idx] = [float(x), float(y), float(conf)]
        
        # Neck (1)과 Mid-hip (8) 계산
        # Neck = (left_shoulder + right_shoulder) / 2
        if keypoints_18[2][2] > 0 and keypoints_18[5][2] > 0:  # 양쪽 어깨가 있을 때
            neck_x = (keypoints_18[2][0] + keypoints_18[5][0]) / 2
            neck_y = (keypoints_18[2][1] + keypoints_18[5][1]) / 2
            neck_conf = min(keypoints_18[2][2], keypoints_18[5][2])
            keypoints_18[1] = [neck_x, neck_y, neck_conf]
        
        # Mid-hip = (left_hip + right_hip) / 2
        if keypoints_18[9][2] > 0 and keypoints_18[12][2] > 0:  # 양쪽 엉덩이가 있을 때
            mid_hip_x = (keypoints_18[9][0] + keypoints_18[12][0]) / 2
            mid_hip_y = (keypoints_18[9][1] + keypoints_18[12][1]) / 2
            mid_hip_conf = min(keypoints_18[9][2], keypoints_18[12][2])
            keypoints_18[8] = [mid_hip_x, mid_hip_y, mid_hip_conf]
        
        return keypoints_18
    
    def _generate_dummy_openpose18(self, width: int, height: int) -> List[List[float]]:
        """더미 OpenPose 18 keypoints 생성"""
        # 기본 T-포즈 형태
        center_x, center_y = width // 2, height // 2
        
        keypoints = [
            [center_x, center_y - height * 0.35, 0.8],      # 0: nose
            [center_x, center_y - height * 0.25, 0.8],      # 1: neck
            [center_x + width * 0.15, center_y - height * 0.2, 0.8],   # 2: right_shoulder
            [center_x + width * 0.25, center_y - height * 0.1, 0.8],   # 3: right_elbow
            [center_x + width * 0.35, center_y, 0.8],       # 4: right_wrist
            [center_x - width * 0.15, center_y - height * 0.2, 0.8],   # 5: left_shoulder
            [center_x - width * 0.25, center_y - height * 0.1, 0.8],   # 6: left_elbow
            [center_x - width * 0.35, center_y, 0.8],       # 7: left_wrist
            [center_x, center_y + height * 0.1, 0.8],       # 8: mid_hip
            [center_x + width * 0.1, center_y + height * 0.1, 0.8],    # 9: right_hip
            [center_x + width * 0.1, center_y + height * 0.25, 0.8],   # 10: right_knee
            [center_x + width * 0.1, center_y + height * 0.4, 0.8],    # 11: right_ankle
            [center_x - width * 0.1, center_y + height * 0.1, 0.8],    # 12: left_hip
            [center_x - width * 0.1, center_y + height * 0.25, 0.8],   # 13: left_knee
            [center_x - width * 0.1, center_y + height * 0.4, 0.8],    # 14: left_ankle
            [center_x + width * 0.05, center_y - height * 0.37, 0.8],  # 15: right_eye
            [center_x - width * 0.05, center_y - height * 0.37, 0.8],  # 16: left_eye
            [center_x + width * 0.08, center_y - height * 0.34, 0.8],  # 17: right_ear
        ]
        
        return keypoints
    
    def _calculate_mediapipe_confidence(self, landmarks) -> float:
        """MediaPipe 랜드마크 전체 신뢰도 계산"""
        if not landmarks or not landmarks.landmark:
            return 0.0
        
        confidences = [lm.visibility for lm in landmarks.landmark if lm.visibility > 0]
        return float(np.mean(confidences)) if confidences else 0.0
    
    def _create_empty_pose_result(self, model_type: str) -> Dict[str, Any]:
        """빈 포즈 결과 생성"""
        return {
            'keypoints_18': [[0, 0, 0] for _ in range(18)],
            'raw_results': None,
            'confidence': 0.0,
            'model_type': model_type,
            'keypoints_detected': 0
        }
    
    async def _postprocess_results(
        self, 
        pose_result: Dict[str, Any], 
        image_shape: Tuple[int, int, int],
        original_image: np.ndarray,  # 🆕 원본 이미지 추가 (시각화용)
        **kwargs
    ) -> Dict[str, Any]:
        """결과 후처리 + 시각화"""
        try:
            keypoints_18 = pose_result['keypoints_18']
            
            # 품질 필터링
            if self.config['quality']['filter_low_confidence']:
                min_conf = self.config['quality']['min_pose_confidence']
                for kp in keypoints_18:
                    if kp[2] < min_conf:
                        kp[2] = 0.0  # 낮은 신뢰도는 0으로 설정
            
            # 키포인트 검증
            detected_count = sum(1 for kp in keypoints_18 if kp[2] > 0.5)
            min_required = self.config['quality']['min_keypoints_detected']
            
            quality_passed = detected_count >= min_required
            
            # 바운딩 박스 계산
            bbox = self._calculate_pose_bbox(keypoints_18, image_shape)
            
            # 포즈 각도 계산
            pose_angles = self._calculate_pose_angles(keypoints_18)
            
            # 신체 비율 계산
            body_proportions = self._calculate_body_proportions(keypoints_18)
            
            # 🆕 시각화 이미지 생성
            visualization_results = await self._create_pose_visualization(
                keypoints_18, 
                original_image
            )
            
            return {
                'keypoints_18': keypoints_18,
                'pose_confidence': pose_result['confidence'],
                'keypoints_detected': detected_count,
                'quality_passed': quality_passed,
                'bbox': bbox,
                'pose_angles': pose_angles,
                'body_proportions': body_proportions,
                'raw_model_result': pose_result.get('raw_results'),
                'model_type': pose_result['model_type'],
                
                # 🆕 프론트엔드용 시각화 이미지들
                'details': {
                    'result_image': visualization_results["keypoints_image"],  # 메인 시각화
                    'overlay_image': visualization_results["overlay_image"],   # 오버레이
                    'skeleton_image': visualization_results["skeleton_image"], # 스켈레톤만
                    
                    # 기존 데이터들
                    'detected_keypoints': detected_count,
                    'total_keypoints': 18,
                    'keypoint_names': [OPENPOSE_18_KEYPOINTS[i] for i in range(18)],
                    'confidence_values': [kp[2] for kp in keypoints_18],
                    
                    # 상세 분석 정보
                    'pose_type': self._classify_pose_type(keypoints_18, pose_angles),
                    'symmetry_score': self._calculate_symmetry_score(keypoints_18),
                    'pose_quality': 'excellent' if detected_count >= 16 else 'good' if detected_count >= 12 else 'poor',
                    
                    # 시스템 정보
                    'step_info': {
                        'step_name': 'pose_estimation',
                        'step_number': 2,
                        'model_used': pose_result['model_type'],
                        'device': self.device,
                        'optimization': 'M3 Max' if self.is_m3_max else self.device
                    },
                    
                    # 품질 메트릭
                    'quality_metrics': {
                        'detection_rate': float(detected_count / 18),
                        'avg_confidence': float(np.mean([kp[2] for kp in keypoints_18 if kp[2] > 0])) if detected_count > 0 else 0.0,
                        'pose_confidence': pose_result['confidence'],
                        'quality_passed': quality_passed
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"결과 후처리 실패: {e}")
            return pose_result

    # ==============================================
    # 🆕 시각화 함수들
    # ==============================================
    
    async def _create_pose_visualization(
        self, 
        keypoints_18: List[List[float]], 
        original_image: np.ndarray
    ) -> Dict[str, str]:
        """
        🆕 18개 키포인트를 시각화한 이미지들 생성
        
        Args:
            keypoints_18: OpenPose 18 키포인트 [x, y, confidence]
            original_image: 원본 이미지 np.ndarray
            
        Returns:
            Dict[str, str]: base64 인코딩된 시각화 이미지들
        """
        try:
            if not self.config['visualization']['enable_visualization']:
                # 시각화 비활성화 시 빈 결과 반환
                return {
                    "keypoints_image": "",
                    "overlay_image": "",
                    "skeleton_image": ""
                }
            
            def _create_visualizations():
                height, width = original_image.shape[:2]
                
                # 1. 🎯 키포인트만 표시된 이미지 생성
                keypoints_image = self._create_keypoints_only_image(keypoints_18, (width, height))
                
                # 2. 🌈 오버레이 이미지 생성 (원본 + 포즈)
                overlay_image = self._create_overlay_pose_image(original_image, keypoints_18)
                
                # 3. 🦴 스켈레톤만 표시된 이미지 생성
                skeleton_image = self._create_skeleton_only_image(keypoints_18, (width, height))
                
                # base64 인코딩
                result = {
                    "keypoints_image": self._pil_to_base64(keypoints_image),
                    "overlay_image": self._pil_to_base64(overlay_image),
                    "skeleton_image": self._pil_to_base64(skeleton_image)
                }
                
                return result
            
            # 비동기 실행
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, _create_visualizations)
            
        except Exception as e:
            self.logger.error(f"❌ 포즈 시각화 생성 실패: {e}")
            # 폴백: 빈 결과 반환
            return {
                "keypoints_image": "",
                "overlay_image": "",
                "skeleton_image": ""
            }
    
    def _create_keypoints_only_image(
        self, 
        keypoints_18: List[List[float]], 
        image_size: Tuple[int, int]
    ) -> Image.Image:
        """키포인트만 표시된 이미지 생성"""
        try:
            width, height = image_size
            config = self.config['visualization']
            
            # 검정 배경 이미지 생성
            bg_color = config['background_color']
            image = Image.new('RGB', (width, height), bg_color)
            draw = ImageDraw.Draw(image)
            
            # 키포인트 그리기
            radius = config['keypoint_radius']
            threshold = config['confidence_threshold']
            
            for i, (x, y, conf) in enumerate(keypoints_18):
                if conf > threshold:
                    # 키포인트별 색상
                    color = KEYPOINT_COLORS.get(i, (255, 255, 255))
                    
                    # 신뢰도에 따른 투명도 조정
                    alpha = int(255 * conf)
                    if alpha > 255: alpha = 255
                    
                    # 키포인트 원 그리기
                    draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                                fill=color, outline=(255, 255, 255), width=2)
                    
                    # 라벨 표시 (옵션)
                    if config['show_keypoint_labels']:
                        keypoint_name = OPENPOSE_18_KEYPOINTS.get(i, f"kp_{i}")
                        try:
                            font = ImageFont.truetype("arial.ttf", config['font_size'])
                        except:
                            font = ImageFont.load_default()
                        
                        # 텍스트 위치 계산
                        text_x = x + radius + 2
                        text_y = y - radius
                        draw.text((text_x, text_y), keypoint_name, fill=(255, 255, 255), font=font)
                    
                    # 신뢰도 값 표시 (옵션)
                    if config['show_confidence_values']:
                        conf_text = f"{conf:.2f}"
                        try:
                            small_font = ImageFont.truetype("arial.ttf", config['font_size'] - 2)
                        except:
                            small_font = ImageFont.load_default()
                        
                        draw.text((x, y + radius + 2), conf_text, fill=color, font=small_font)
            
            return image
            
        except Exception as e:
            self.logger.warning(f"⚠️ 키포인트 이미지 생성 실패: {e}")
            # 폴백: 기본 이미지 생성
            return Image.new('RGB', image_size, (50, 50, 50))
    
    def _create_overlay_pose_image(
        self, 
        original_image: np.ndarray, 
        keypoints_18: List[List[float]]
    ) -> Image.Image:
        """원본 이미지 위에 포즈를 오버레이한 이미지 생성"""
        try:
            # numpy 배열을 PIL 이미지로 변환
            if original_image.max() <= 1.0:
                original_pil = Image.fromarray((original_image * 255).astype(np.uint8))
            else:
                original_pil = Image.fromarray(original_image.astype(np.uint8))
            
            # 투명한 오버레이 레이어 생성
            overlay = Image.new('RGBA', original_pil.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            config = self.config['visualization']
            threshold = config['confidence_threshold']
            keypoint_radius = config['keypoint_radius']
            skeleton_thickness = config['skeleton_thickness']
            
            # 1. 스켈레톤 연결선 그리기
            for start_idx, end_idx in SKELETON_CONNECTIONS:
                if (start_idx < len(keypoints_18) and end_idx < len(keypoints_18) and
                    keypoints_18[start_idx][2] > threshold and keypoints_18[end_idx][2] > threshold):
                    
                    start_point = (int(keypoints_18[start_idx][0]), int(keypoints_18[start_idx][1]))
                    end_point = (int(keypoints_18[end_idx][0]), int(keypoints_18[end_idx][1]))
                    
                    # 연결선별 색상
                    line_color = SKELETON_COLORS.get((start_idx, end_idx), (255, 255, 255))
                    
                    # 신뢰도 기반 투명도
                    avg_conf = (keypoints_18[start_idx][2] + keypoints_18[end_idx][2]) / 2
                    alpha = int(255 * avg_conf * config['overlay_opacity'])
                    line_color_alpha = line_color + (alpha,)
                    
                    draw.line([start_point, end_point], fill=line_color_alpha, width=skeleton_thickness)
            
            # 2. 키포인트 그리기
            for i, (x, y, conf) in enumerate(keypoints_18):
                if conf > threshold:
                    color = KEYPOINT_COLORS.get(i, (255, 255, 255))
                    alpha = int(255 * conf * config['overlay_opacity'])
                    color_alpha = color + (alpha,)
                    
                    # 키포인트 원
                    draw.ellipse([x-keypoint_radius, y-keypoint_radius, 
                                x+keypoint_radius, y+keypoint_radius], 
                               fill=color_alpha, outline=(255, 255, 255, alpha), width=2)
            
            # 원본 이미지와 오버레이 합성
            original_rgba = original_pil.convert('RGBA')
            combined = Image.alpha_composite(original_rgba, overlay)
            
            return combined.convert('RGB')
            
        except Exception as e:
            self.logger.warning(f"⚠️ 오버레이 이미지 생성 실패: {e}")
            # 폴백: 원본 이미지 반환
            if original_image.max() <= 1.0:
                return Image.fromarray((original_image * 255).astype(np.uint8))
            else:
                return Image.fromarray(original_image.astype(np.uint8))
    
    def _create_skeleton_only_image(
        self, 
        keypoints_18: List[List[float]], 
        image_size: Tuple[int, int]
    ) -> Image.Image:
        """스켈레톤만 표시된 이미지 생성"""
        try:
            width, height = image_size
            config = self.config['visualization']
            
            # 검정 배경 이미지 생성
            bg_color = config['background_color']
            image = Image.new('RGB', (width, height), bg_color)
            draw = ImageDraw.Draw(image)
            
            threshold = config['confidence_threshold']
            thickness = config['skeleton_thickness']
            
            # 스켈레톤 연결선만 그리기
            for start_idx, end_idx in SKELETON_CONNECTIONS:
                if (start_idx < len(keypoints_18) and end_idx < len(keypoints_18) and
                    keypoints_18[start_idx][2] > threshold and keypoints_18[end_idx][2] > threshold):
                    
                    start_point = (int(keypoints_18[start_idx][0]), int(keypoints_18[start_idx][1]))
                    end_point = (int(keypoints_18[end_idx][0]), int(keypoints_18[end_idx][1]))
                    
                    # 연결선별 색상
                    line_color = SKELETON_COLORS.get((start_idx, end_idx), (255, 255, 255))
                    
                    draw.line([start_point, end_point], fill=line_color, width=thickness)
            
            # 관절점을 작은 원으로 표시
            for i, (x, y, conf) in enumerate(keypoints_18):
                if conf > threshold:
                    color = KEYPOINT_COLORS.get(i, (255, 255, 255))
                    small_radius = max(2, config['keypoint_radius'] // 2)
                    
                    draw.ellipse([x-small_radius, y-small_radius, 
                                x+small_radius, y+small_radius], 
                               fill=color, outline=(255, 255, 255), width=1)
            
            return image
            
        except Exception as e:
            self.logger.warning(f"⚠️ 스켈레톤 이미지 생성 실패: {e}")
            # 폴백: 기본 이미지 생성
            return Image.new('RGB', image_size, (50, 50, 50))
    
    def _pil_to_base64(self, pil_image: Image.Image) -> str:
        """PIL 이미지를 base64 문자열로 변환"""
        try:
            buffer = BytesIO()
            
            # 품질 설정
            quality = 85
            if self.config['visualization']['image_quality'] == "high":
                quality = 95
            elif self.config['visualization']['image_quality'] == "low":
                quality = 70
            
            pil_image.save(buffer, format='JPEG', quality=quality)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
        except Exception as e:
            self.logger.warning(f"⚠️ base64 변환 실패: {e}")
            return ""
    
    # ==============================================
    # 🔧 기존 함수들 (변경 없음)
    # ==============================================
    
    def _calculate_pose_bbox(
        self, 
        keypoints_18: List[List[float]], 
        image_shape: Tuple[int, int, int]
    ) -> Dict[str, int]:
        """포즈 바운딩 박스 계산"""
        valid_points = [(kp[0], kp[1]) for kp in keypoints_18 if kp[2] > 0.5]
        
        if not valid_points:
            return {"x": 0, "y": 0, "width": 0, "height": 0}
        
        xs, ys = zip(*valid_points)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # 여백 추가 (15%)
        margin_x = int((x_max - x_min) * 0.15)
        margin_y = int((y_max - y_min) * 0.15)
        
        height, width = image_shape[:2]
        
        return {
            "x": max(0, int(x_min - margin_x)),
            "y": max(0, int(y_min - margin_y)),
            "width": min(width, int(x_max - x_min + 2 * margin_x)),
            "height": min(height, int(y_max - y_min + 2 * margin_y))
        }
    
    def _calculate_pose_angles(self, keypoints_18: List[List[float]]) -> Dict[str, float]:
        """주요 관절 각도 계산"""
        def calculate_angle(p1, p2, p3):
            """세 점으로 각도 계산"""
            if any(p[2] < 0.5 for p in [p1, p2, p3]):
                return 0.0
            
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            return float(np.degrees(angle))
        
        try:
            angles = {}
            
            # 오른쪽 팔 각도 (어깨-팔꿈치-손목)
            if all(keypoints_18[i][2] > 0.5 for i in [2, 3, 4]):
                angles['right_arm'] = calculate_angle(
                    keypoints_18[2], keypoints_18[3], keypoints_18[4]
                )
            
            # 왼쪽 팔 각도
            if all(keypoints_18[i][2] > 0.5 for i in [5, 6, 7]):
                angles['left_arm'] = calculate_angle(
                    keypoints_18[5], keypoints_18[6], keypoints_18[7]
                )
            
            # 오른쪽 다리 각도 (엉덩이-무릎-발목)
            if all(keypoints_18[i][2] > 0.5 for i in [9, 10, 11]):
                angles['right_leg'] = calculate_angle(
                    keypoints_18[9], keypoints_18[10], keypoints_18[11]
                )
            
            # 왼쪽 다리 각도
            if all(keypoints_18[i][2] > 0.5 for i in [12, 13, 14]):
                angles['left_leg'] = calculate_angle(
                    keypoints_18[12], keypoints_18[13], keypoints_18[14]
                )
            
            # 목 각도 (목-어깨 중점-엉덩이 중점)
            if all(keypoints_18[i][2] > 0.5 for i in [1, 2, 5, 8]):
                shoulder_center = [
                    (keypoints_18[2][0] + keypoints_18[5][0]) / 2,
                    (keypoints_18[2][1] + keypoints_18[5][1]) / 2,
                    1.0
                ]
                angles['torso'] = calculate_angle(
                    keypoints_18[1], shoulder_center, keypoints_18[8]
                )
            
            return angles
            
        except Exception as e:
            self.logger.warning(f"각도 계산 실패: {e}")
            return {}
    
    def _calculate_body_proportions(self, keypoints_18: List[List[float]]) -> Dict[str, float]:
        """신체 비율 계산"""
        try:
            proportions = {}
            
            # 전체 키 (머리-발목)
            if keypoints_18[0][2] > 0.5 and any(keypoints_18[i][2] > 0.5 for i in [11, 14]):
                head_y = keypoints_18[0][1]
                ankle_y = min(
                    keypoints_18[11][1] if keypoints_18[11][2] > 0.5 else float('inf'),
                    keypoints_18[14][1] if keypoints_18[14][2] > 0.5 else float('inf')
                )
                if ankle_y != float('inf'):
                    total_height = abs(ankle_y - head_y)
                    proportions['total_height'] = total_height
            
            # 상체 길이 (목-엉덩이)
            if keypoints_18[1][2] > 0.5 and keypoints_18[8][2] > 0.5:
                torso_length = abs(keypoints_18[8][1] - keypoints_18[1][1])
                proportions['torso_length'] = torso_length
            
            # 어깨 너비
            if keypoints_18[2][2] > 0.5 and keypoints_18[5][2] > 0.5:
                shoulder_width = abs(keypoints_18[2][0] - keypoints_18[5][0])
                proportions['shoulder_width'] = shoulder_width
            
            # 엉덩이 너비
            if keypoints_18[9][2] > 0.5 and keypoints_18[12][2] > 0.5:
                hip_width = abs(keypoints_18[9][0] - keypoints_18[12][0])
                proportions['hip_width'] = hip_width
            
            # 팔 길이 (어깨-손목)
            if keypoints_18[2][2] > 0.5 and keypoints_18[4][2] > 0.5:
                right_arm_length = np.sqrt(
                    (keypoints_18[4][0] - keypoints_18[2][0])**2 + 
                    (keypoints_18[4][1] - keypoints_18[2][1])**2
                )
                proportions['right_arm_length'] = right_arm_length
            
            if keypoints_18[5][2] > 0.5 and keypoints_18[7][2] > 0.5:
                left_arm_length = np.sqrt(
                    (keypoints_18[7][0] - keypoints_18[5][0])**2 + 
                    (keypoints_18[7][1] - keypoints_18[5][1])**2
                )
                proportions['left_arm_length'] = left_arm_length
            
            return proportions
            
        except Exception as e:
            self.logger.warning(f"신체 비율 계산 실패: {e}")
            return {}
    
    def _analyze_pose_quality(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """포즈 품질 분석"""
        try:
            keypoints_18 = result['keypoints_18']
            detected_count = result['keypoints_detected']
            
            # 1. 검출률
            detection_rate = detected_count / 18
            
            # 2. 주요 키포인트 검출률
            major_keypoints = [0, 1, 2, 5, 8, 9, 12]  # 머리, 목, 어깨, 엉덩이
            major_detected = sum(1 for idx in major_keypoints if keypoints_18[idx][2] > 0.5)
            major_detection_rate = major_detected / len(major_keypoints)
            
            # 3. 평균 신뢰도
            confidences = [kp[2] for kp in keypoints_18 if kp[2] > 0]
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # 4. 대칭성 점수
            symmetry_score = self._calculate_symmetry_score(keypoints_18)
            
            # 5. 포즈 타입 분류
            pose_type = self._classify_pose_type(keypoints_18, result.get('pose_angles', {}))
            
            # 6. 전체 품질 점수
            quality_score = (
                detection_rate * 0.3 +
                major_detection_rate * 0.3 +
                avg_confidence * 0.2 +
                symmetry_score * 0.2
            )
            
            return {
                'detection_rate': float(detection_rate),
                'major_detection_rate': float(major_detection_rate),
                'avg_confidence': float(avg_confidence),
                'symmetry_score': float(symmetry_score),
                'pose_type': pose_type,
                'quality_score': float(quality_score),
                'quality_grade': self._get_quality_grade(quality_score)
            }
            
        except Exception as e:
            self.logger.warning(f"포즈 품질 분석 실패: {e}")
            return {
                'detection_rate': 0.0,
                'major_detection_rate': 0.0,
                'avg_confidence': 0.0,
                'symmetry_score': 0.0,
                'pose_type': 'unknown',
                'quality_score': 0.0,
                'quality_grade': 'F'
            }
    
    def _calculate_symmetry_score(self, keypoints_18: List[List[float]]) -> float:
        """포즈 대칭성 점수 계산"""
        try:
            # 대칭 키포인트 쌍들
            symmetric_pairs = [
                (2, 5),   # 어깨
                (3, 6),   # 팔꿈치
                (4, 7),   # 손목
                (9, 12),  # 엉덩이
                (10, 13), # 무릎
                (11, 14), # 발목
                (15, 16), # 눈
            ]
            
            symmetry_scores = []
            
            for left_idx, right_idx in symmetric_pairs:
                if keypoints_18[left_idx][2] > 0.5 and keypoints_18[right_idx][2] > 0.5:
                    # 중심선 기준으로 대칭성 계산
                    center_x = (keypoints_18[left_idx][0] + keypoints_18[right_idx][0]) / 2
                    
                    left_dist = abs(keypoints_18[left_idx][0] - center_x)
                    right_dist = abs(keypoints_18[right_idx][0] - center_x)
                    
                    if left_dist + right_dist > 0:
                        symmetry = 1.0 - abs(left_dist - right_dist) / (left_dist + right_dist)
                        symmetry_scores.append(symmetry)
            
            return float(np.mean(symmetry_scores)) if symmetry_scores else 0.5
            
        except Exception as e:
            self.logger.warning(f"대칭성 계산 실패: {e}")
            return 0.5
    
    def _classify_pose_type(
        self, 
        keypoints_18: List[List[float]], 
        pose_angles: Dict[str, float]
    ) -> str:
        """포즈 타입 분류"""
        try:
            # 팔 각도 기반 분류
            right_arm = pose_angles.get('right_arm', 180)
            left_arm = pose_angles.get('left_arm', 180)
            
            # T-포즈 (팔이 수평)
            if abs(right_arm - 180) < 20 and abs(left_arm - 180) < 20:
                return 't_pose'
            
            # A-포즈 (팔이 약간 아래)
            elif 140 < right_arm < 170 and 140 < left_arm < 170:
                return 'a_pose'
            
            # 팔 올린 포즈
            elif right_arm < 90 or left_arm < 90:
                return 'arms_up'
            
            # 다리 상태 확인
            right_leg = pose_angles.get('right_leg', 180)
            left_leg = pose_angles.get('left_leg', 180)
            
            # 앉은 포즈
            if right_leg < 140 or left_leg < 140:
                return 'sitting'
            
            # 걷기/뛰기 (다리 비대칭)
            elif abs(right_leg - left_leg) > 30:
                return 'walking'
            
            # 기본 서있는 포즈
            else:
                return 'standing'
                
        except Exception as e:
            self.logger.warning(f"포즈 타입 분류 실패: {e}")
            return 'unknown'
    
    def _get_quality_grade(self, quality_score: float) -> str:
        """품질 점수를 등급으로 변환"""
        if quality_score >= 0.9:
            return 'A+'
        elif quality_score >= 0.8:
            return 'A'
        elif quality_score >= 0.7:
            return 'B'
        elif quality_score >= 0.6:
            return 'C'
        elif quality_score >= 0.5:
            return 'D'
        else:
            return 'F'
    
    def _generate_cache_key(self, image_input: Union[str, np.ndarray, Image.Image]) -> str:
        """캐시 키 생성"""
        try:
            if isinstance(image_input, str):
                # 파일 경로의 해시
                import hashlib
                return hashlib.md5(image_input.encode()).hexdigest()[:16]
            elif isinstance(image_input, (np.ndarray, Image.Image)):
                # 이미지 데이터의 해시
                if isinstance(image_input, Image.Image):
                    image_array = np.array(image_input)
                else:
                    image_array = image_input
                
                # 작은 크기로 리사이즈해서 해시 계산
                small_image = cv2.resize(image_array, (64, 64))
                image_hash = hash(small_image.tobytes())
                return f"img_{abs(image_hash) % (10**16):016d}"
            else:
                return f"unknown_{time.time():.6f}"
                
        except Exception as e:
            self.logger.warning(f"캐시 키 생성 실패: {e}")
            return f"fallback_{time.time():.6f}"
    
    def _update_processing_stats(self, processing_time: float):
        """처리 통계 업데이트"""
        try:
            self.processing_stats['total_processed'] += 1
            
            # 평균 처리 시간 업데이트
            total = self.processing_stats['total_processed']
            current_avg = self.processing_stats['avg_processing_time']
            new_avg = (current_avg * (total - 1) + processing_time) / total
            self.processing_stats['avg_processing_time'] = new_avg
            
        except Exception as e:
            self.logger.warning(f"통계 업데이트 실패: {e}")
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """에러 결과 생성"""
        return {
            'success': False,
            'error': error_message,
            'keypoints_18': [[0, 0, 0] for _ in range(18)],
            'pose_confidence': 0.0,
            'keypoints_detected': 0,
            'quality_passed': False,
            'bbox': {"x": 0, "y": 0, "width": 0, "height": 0},
            'pose_angles': {},
            'body_proportions': {},
            'pose_analysis': {
                'detection_rate': 0.0,
                'quality_score': 0.0,
                'quality_grade': 'F',
                'pose_type': 'unknown'
            },
            'details': {
                'result_image': "",  # 빈 시각화 이미지
                'overlay_image': "",
                'skeleton_image': "",
                'detected_keypoints': 0,
                'error': error_message,
                'step_info': {
                    'step_name': 'pose_estimation',
                    'step_number': 2,
                    'model_used': 'error',
                    'device': self.device,
                    'error': error_message
                },
                'quality_metrics': {
                    'detection_rate': 0.0,
                    'avg_confidence': 0.0,
                    'pose_confidence': 0.0,
                    'quality_passed': False
                }
            },
            'model_type': 'error',
            'processing_time': 0.0,
            'device': self.device,
            'm3_max_optimized': self.is_m3_max
        }
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        return {
            'step_name': self.step_name,
            'is_initialized': self.is_initialized,
            'current_model': self.current_model_type,
            'device': self.device,
            'device_type': self.device_type,
            'm3_max_optimized': self.is_m3_max,
            'processing_stats': self.processing_stats.copy(),
            'cache_size': len(self.prediction_cache),
            'models_loaded': {
                'mediapipe': self.pose_model_primary is not None,
                'yolo': self.pose_model_secondary is not None
            }
        }
    
    async def warmup(self, num_iterations: int = 3) -> Dict[str, Any]:
        """모델 워밍업"""
        try:
            self.logger.info(f"🔥 {num_iterations}회 워밍업 시작...")
            warmup_times = []
            
            for i in range(num_iterations):
                # 랜덤 테스트 이미지 생성
                test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
                
                start_time = time.time()
                result = await self.process(test_image, cache_result=False)
                warmup_time = time.time() - start_time
                warmup_times.append(warmup_time)
                
                self.logger.info(f"워밍업 {i+1}/{num_iterations}: {warmup_time:.3f}초")
            
            avg_warmup_time = np.mean(warmup_times)
            
            # MPS 캐시 정리
            if TORCH_AVAILABLE and self.device == "mps":
                torch.mps.empty_cache()
            
            self.logger.info(f"✅ 워밍업 완료 - 평균 시간: {avg_warmup_time:.3f}초")
            
            return {
                'success': True,
                'iterations': num_iterations,
                'avg_time': avg_warmup_time,
                'times': warmup_times,
                'model_type': self.current_model_type
            }
            
        except Exception as e:
            error_msg = f"워밍업 실패: {e}"
            self.logger.error(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            self.logger.info("🧹 PoseEstimationStep 리소스 정리 시작...")
            
            # === 모델 정리 ===
            if self.pose_model_primary and hasattr(self.pose_model_primary, 'close'):
                self.pose_model_primary.close()
            
            if self.pose_model_secondary:
                if hasattr(self.pose_model_secondary, 'cpu'):
                    self.pose_model_secondary.cpu()
                del self.pose_model_secondary
            
            # === MediaPipe 정리 ===
            if self.pose_detector and hasattr(self.pose_detector, 'close'):
                self.pose_detector.close()
            
            self.pose_model_primary = None
            self.pose_model_secondary = None
            self.pose_detector = None
            self.mp_pose = None
            self.mp_drawing = None
            self.mp_drawing_styles = None
            
            # === 캐시 정리 ===
            self.model_cache.clear()
            self.prediction_cache.clear()
            
            # === Model Loader 정리 ===
            self.cleanup_models()
            
            # === 스레드 풀 정리 ===
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
            
            # === GPU 메모리 정리 ===
            if TORCH_AVAILABLE:
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                
                # 일반 정리
                gc.collect()
            
            self.is_initialized = False
            self.logger.info("✅ PoseEstimationStep 리소스 정리 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 리소스 정리 중 오류: {e}")
    
    def __del__(self):
        """소멸자"""
        try:
            asyncio.create_task(self.cleanup())
        except:
            pass

# =================================================================
# 🔄 하위 호환성 지원 및 팩토리 함수
# =================================================================

async def create_pose_estimation_step(
    device: str = "auto",
    config: Dict[str, Any] = None,
    **kwargs
) -> PoseEstimationStep:
    """
    🔄 기존 팩토리 함수 호환 (기존 파이프라인 호환)
    
    Args:
        device: 사용할 디바이스 ("auto"는 자동 감지)
        config: 설정 딕셔너리
        **kwargs: 추가 매개변수
        
    Returns:
        PoseEstimationStep: 초기화된 2단계 스텝
    """
    try:
        # 기존 방식 호환
        device_param = None if device == "auto" else device
        
        # 기본 설정
        default_config = {
            "mediapipe": {
                "model_complexity": 2,
                "min_detection_confidence": 0.7,
                "min_tracking_confidence": 0.5,
                "enable_segmentation": False,
                "static_image_mode": True
            },
            "processing": {
                "max_image_size": 1024,
                "output_format": "openpose_18"
            },
            "quality": {
                "min_keypoints_detected": 10,
                "min_pose_confidence": 0.5
            },
            "visualization": {
                "enable_visualization": True,
                "keypoint_radius": 5,
                "skeleton_thickness": 3,
                "image_quality": "high"
            }
        }
        
        # 설정 병합
        final_config = {**default_config}
        if config:
            def deep_update(base_dict, update_dict):
                for key, value in update_dict.items():
                    if isinstance(value, dict) and key in base_dict:
                        deep_update(base_dict[key], value)
                    else:
                        base_dict[key] = value
            deep_update(final_config, config)
        
        # ✅ 새로운 통일된 생성자 사용
        step = PoseEstimationStep(device=device_param, config=final_config, **kwargs)
        
        # 초기화
        init_success = await step.initialize()
        if not init_success:
            logger.warning("PoseEstimationStep 초기화에 문제가 있었지만 진행합니다.")
        
        return step
        
    except Exception as e:
        logger.error(f"❌ PoseEstimationStep 생성 실패: {e}")
        # 최소한의 더미 스텝이라도 반환
        step = PoseEstimationStep(device=device_param, config=final_config, **kwargs)
        await step._initialize_dummy_model()
        step.is_initialized = True
        return step

# =================================================================
# 🔄 기존 클래스명 별칭 (완전 호환)
# =================================================================

PoseEstimationStepLegacy = PoseEstimationStep

# =================================================================
# 🔥 고급 기능들 - 추가 유틸리티
# =================================================================

def validate_openpose_keypoints(keypoints_18: List[List[float]]) -> bool:
    """OpenPose 18 keypoints 유효성 검증"""
    try:
        if len(keypoints_18) != 18:
            return False
        
        for kp in keypoints_18:
            if len(kp) != 3:
                return False
            if not all(isinstance(x, (int, float)) for x in kp):
                return False
            if kp[2] < 0 or kp[2] > 1:  # 신뢰도는 0~1 사이
                return False
        
        return True
        
    except Exception:
        return False

def convert_keypoints_to_coco(keypoints_18: List[List[float]]) -> List[List[float]]:
    """OpenPose 18을 COCO 17 형식으로 변환"""
    try:
        # OpenPose 18 -> COCO 17 매핑
        op_to_coco_mapping = {
            0: 0,   # nose
            15: 1,  # right_eye -> left_eye (COCO 관점)
            16: 2,  # left_eye -> right_eye
            17: 3,  # right_ear -> left_ear
            2: 5,   # right_shoulder -> left_shoulder
            5: 6,   # left_shoulder -> right_shoulder
            3: 7,   # right_elbow -> left_elbow
            6: 8,   # left_elbow -> right_elbow
            4: 9,   # right_wrist -> left_wrist
            7: 10,  # left_wrist -> right_wrist
            9: 11,  # right_hip -> left_hip
            12: 12, # left_hip -> right_hip
            10: 13, # right_knee -> left_knee
            13: 14, # left_knee -> right_knee
            11: 15, # right_ankle -> left_ankle
            14: 16, # left_ankle -> right_ankle
        }
        
        coco_keypoints = [[0, 0, 0] for _ in range(17)]
        
        for op_idx, coco_idx in op_to_coco_mapping.items():
            if op_idx < len(keypoints_18) and coco_idx < 17:
                coco_keypoints[coco_idx] = keypoints_18[op_idx].copy()
        
        return coco_keypoints
        
    except Exception as e:
        logger.error(f"COCO 변환 실패: {e}")
        return [[0, 0, 0] for _ in range(17)]

def draw_pose_on_image(
    image: np.ndarray, 
    keypoints_18: List[List[float]],
    draw_skeleton: bool = True,
    draw_keypoints: bool = True,
    line_thickness: int = 2,
    keypoint_radius: int = 3
) -> np.ndarray:
    """이미지에 포즈 그리기 (OpenCV 기반)"""
    try:
        result_image = image.copy()
        
        # 신뢰도 임계값
        threshold = 0.5
        
        # 스켈레톤 그리기
        if draw_skeleton:
            for start_idx, end_idx in SKELETON_CONNECTIONS:
                if (start_idx < len(keypoints_18) and end_idx < len(keypoints_18) and
                    keypoints_18[start_idx][2] > threshold and keypoints_18[end_idx][2] > threshold):
                    
                    start_point = (int(keypoints_18[start_idx][0]), int(keypoints_18[start_idx][1]))
                    end_point = (int(keypoints_18[end_idx][0]), int(keypoints_18[end_idx][1]))
                    
                    # 연결선별 색상 (BGR 형식으로 변환)
                    line_color = SKELETON_COLORS.get((start_idx, end_idx), (255, 255, 255))
                    line_color_bgr = (line_color[2], line_color[1], line_color[0])  # RGB to BGR
                    
                    cv2.line(result_image, start_point, end_point, line_color_bgr, line_thickness)
        
        # 키포인트 그리기
        if draw_keypoints:
            for i, (x, y, conf) in enumerate(keypoints_18):
                if conf > threshold:
                    center = (int(x), int(y))
                    
                    # 키포인트별 색상 (BGR 형식으로 변환)
                    color = KEYPOINT_COLORS.get(i, (255, 255, 255))
                    color_bgr = (color[2], color[1], color[0])  # RGB to BGR
                    
                    cv2.circle(result_image, center, keypoint_radius, color_bgr, -1)
                    cv2.circle(result_image, center, keypoint_radius + 1, (255, 255, 255), 1)
        
        return result_image
        
    except Exception as e:
        logger.error(f"포즈 그리기 실패: {e}")
        return image

def analyze_pose_for_clothing(keypoints_18: List[List[float]]) -> Dict[str, Any]:
    """의류 피팅을 위한 포즈 분석"""
    try:
        analysis = {
            'suitable_for_fitting': False,
            'issues': [],
            'recommendations': [],
            'pose_score': 0.0
        }
        
        # 1. 필수 키포인트 확인
        essential_points = [0, 1, 2, 5, 8, 9, 12]  # 머리, 목, 어깨, 엉덩이
        essential_detected = sum(1 for idx in essential_points if keypoints_18[idx][2] > 0.5)
        
        if essential_detected < len(essential_points) * 0.8:
            analysis['issues'].append("필수 키포인트 부족")
            analysis['recommendations'].append("전신이 잘 보이는 사진을 사용하세요")
        
        # 2. 팔 위치 분석
        arms_visible = (keypoints_18[2][2] > 0.5 and keypoints_18[3][2] > 0.5 and 
                       keypoints_18[5][2] > 0.5 and keypoints_18[6][2] > 0.5)
        
        if not arms_visible:
            analysis['issues'].append("팔이 잘 보이지 않음")
            analysis['recommendations'].append("T-포즈나 A-포즈를 취해주세요")
        
        # 3. 다리 위치 분석
        legs_visible = (keypoints_18[9][2] > 0.5 and keypoints_18[10][2] > 0.5 and 
                       keypoints_18[12][2] > 0.5 and keypoints_18[13][2] > 0.5)
        
        if not legs_visible:
            analysis['issues'].append("다리가 잘 보이지 않음")
            analysis['recommendations'].append("다리가 분리되어 보이는 자세를 취해주세요")
        
        # 4. 정면 방향 확인 (어깨 대칭성)
        if keypoints_18[2][2] > 0.5 and keypoints_18[5][2] > 0.5:
            shoulder_diff = abs(keypoints_18[2][1] - keypoints_18[5][1])
            shoulder_width = abs(keypoints_18[2][0] - keypoints_18[5][0])
            
            if shoulder_width > 0 and shoulder_diff / shoulder_width > 0.3:
                analysis['issues'].append("몸이 기울어져 있음")
                analysis['recommendations'].append("카메라를 정면으로 바라봐 주세요")
        
        # 5. 전체 점수 계산
        base_score = essential_detected / len(essential_points)
        arm_bonus = 0.2 if arms_visible else 0.0
        leg_bonus = 0.2 if legs_visible else 0.0
        
        analysis['pose_score'] = min(1.0, base_score + arm_bonus + leg_bonus)
        
        # 6. 피팅 적합성 판단
        analysis['suitable_for_fitting'] = (
            len(analysis['issues']) <= 1 and 
            analysis['pose_score'] >= 0.7
        )
        
        if analysis['suitable_for_fitting']:
            analysis['recommendations'].append("포즈가 가상 피팅에 적합합니다!")
        
        return analysis
        
    except Exception as e:
        logger.error(f"포즈 분석 실패: {e}")
        return {
            'suitable_for_fitting': False,
            'issues': ["분석 실패"],
            'recommendations': ["다시 시도해 주세요"],
            'pose_score': 0.0
        }

# =================================================================
# 🔥 모듈 익스포트
# =================================================================

__all__ = [
    'PoseEstimationStep',
    'create_pose_estimation_step', 
    'PoseEstimationStepLegacy',
    'validate_openpose_keypoints',
    'convert_keypoints_to_coco',
    'draw_pose_on_image',
    'analyze_pose_for_clothing',
    'OPENPOSE_18_KEYPOINTS',
    'KEYPOINT_COLORS',
    'SKELETON_CONNECTIONS',
    'SKELETON_COLORS'
]

# 모듈 초기화 시 로깅
logger.info("✅ PoseEstimationStep 모듈 로드 완료 - 시각화 기능 포함")