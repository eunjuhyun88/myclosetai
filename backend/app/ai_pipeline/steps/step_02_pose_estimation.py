"""
2단계: 포즈 추정 (Pose Estimation) - 18개 키포인트 
M3 Max 최적화 버전 (MediaPipe + MPS 백엔드)
"""
import os
import logging
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import cv2
from PIL import Image
import json

# MediaPipe 관련 임포트
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    logging.warning("MediaPipe 설치 필요: pip install mediapipe")
    MP_AVAILABLE = False

# CoreML 지원 (M3 Max 전용)
try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

logger = logging.getLogger(__name__)

class PoseEstimationStep:
    """포즈 추정 스텝 - 18개 키포인트 검출 (M3 Max 최적화)"""
    
    # OpenPose 호환 18개 키포인트 정의
    OPENPOSE_18_KEYPOINTS = {
        0: "Nose",
        1: "Neck", 
        2: "R-Shoulder",
        3: "R-Elbow", 
        4: "R-Wrist",
        5: "L-Shoulder",
        6: "L-Elbow",
        7: "L-Wrist",
        8: "R-Hip",
        9: "R-Knee",
        10: "R-Ankle",
        11: "L-Hip", 
        12: "L-Knee",
        13: "L-Ankle",
        14: "R-Eye",
        15: "L-Eye",
        16: "R-Ear",
        17: "L-Ear"
    }
    
    # MediaPipe → OpenPose 매핑
    MP_TO_OPENPOSE_18 = {
        0: 0,   # nose
        # neck은 어깨 중점으로 계산
        11: 5,  # left_shoulder
        12: 2,  # right_shoulder  
        13: 6,  # left_elbow
        14: 3,  # right_elbow
        15: 7,  # left_wrist
        16: 4,  # right_wrist
        23: 11, # left_hip
        24: 8,  # right_hip
        25: 12, # left_knee  
        26: 9,  # right_knee
        27: 13, # left_ankle
        28: 10, # right_ankle
        2: 15,  # left_eye_inner → left_eye
        5: 14,  # right_eye_inner → right_eye  
        7: 17,  # left_ear
        8: 16   # right_ear
    }
    
    def __init__(self, model_loader, device: str, config: Dict[str, Any] = None):
        """
        Args:
            model_loader: 모델 로더 인스턴스
            device: 사용할 디바이스 ('cpu', 'cuda', 'mps')
            config: 설정 딕셔너리
        """
        self.model_loader = model_loader
        self.device = device
        self.config = config or {}
        
        # 기본 설정
        self.model_complexity = self.config.get('model_complexity', 2)  # 0, 1, 2
        self.min_detection_confidence = self.config.get('min_detection_confidence', 0.7)
        self.min_tracking_confidence = self.config.get('min_tracking_confidence', 0.5)
        self.static_image_mode = self.config.get('static_image_mode', True)
        
        # 성능 최적화 설정 (M3 Max)
        self.use_mps = device == 'mps' and torch.backends.mps.is_available()
        self.use_coreml = COREML_AVAILABLE and self.config.get('use_coreml', True)
        
        # 모델 관련
        self.mp_pose = None
        self.pose_model = None
        self.coreml_model = None
        self.is_initialized = False
        
        logger.info(f"🎯 포즈 추정 스텝 초기화 - 디바이스: {device}, MPS: {self.use_mps}, CoreML: {self.use_coreml}")
    
    async def initialize(self) -> bool:
        """모델 초기화"""
        try:
            logger.info("🔄 포즈 추정 모델 로드 중...")
            
            # MediaPipe 초기화
            if MP_AVAILABLE:
                self.mp_pose = mp.solutions.pose
                self.pose_model = self.mp_pose.Pose(
                    static_image_mode=self.static_image_mode,
                    model_complexity=self.model_complexity,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                    enable_segmentation=False  # 성능 최적화
                )
                logger.info("✅ MediaPipe Pose 모델 로드 완료")
            else:
                logger.warning("⚠️ MediaPipe 사용 불가, 데모 모드로 실행")
                self._create_demo_model()
            
            # CoreML 모델 로드 시도 (M3 Max 최적화)
            if self.use_coreml:
                await self._load_coreml_model()
            
            self.is_initialized = True
            logger.info("✅ 포즈 추정 모델 초기화 완료")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 포즈 추정 모델 초기화 실패: {e}")
            self.is_initialized = False
            return False
    
    async def _load_coreml_model(self):
        """CoreML 모델 로드 (M3 Max 전용)"""
        try:
            model_path = self.config.get('coreml_model_path', 'app/models/ai_models/pose_estimation.mlmodel')
            
            if os.path.exists(model_path):
                self.coreml_model = ct.models.MLModel(model_path)
                logger.info("✅ CoreML 포즈 모델 로드 완료")
            else:
                logger.info("ℹ️ CoreML 모델 파일 없음, MediaPipe 사용")
                
        except Exception as e:
            logger.warning(f"⚠️ CoreML 모델 로드 실패: {e}")
    
    def _create_demo_model(self):
        """데모용 포즈 추정 모델"""
        logger.info("🔧 데모 포즈 모델 생성 중...")
        # 실제 배포 시에는 제거
        pass
    
    def process(self, person_image_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        포즈 추정 처리
        
        Args:
            person_image_tensor: 사용자 이미지 텐서 [1, 3, H, W]
            
        Returns:
            처리 결과 딕셔너리
        """
        if not self.is_initialized:
            raise RuntimeError("포즈 추정 모델이 초기화되지 않았습니다.")
        
        start_time = time.time()
        
        try:
            # 텐서를 OpenCV 이미지로 변환
            cv_image = self._tensor_to_cv_image(person_image_tensor)
            
            # CoreML 우선 시도 (M3 Max 최적화)
            if self.coreml_model is not None:
                result = self._process_with_coreml(cv_image)
            else:
                # MediaPipe로 처리
                result = self._process_with_mediapipe(cv_image)
            
            processing_time = time.time() - start_time
            
            # 결과 구성
            final_result = {
                "success": result["success"],
                "keypoints_18": result["keypoints_18"],
                "keypoints_raw": result.get("keypoints_raw", []),
                "pose_confidence": result["confidence"],
                "body_orientation": result["orientation"],
                "pose_angles": result["angles"],
                "bounding_box": result["bbox"],
                "pose_connections": self._get_pose_connections(),
                "processing_time": processing_time,
                "model_used": result["model_used"],
                "quality_metrics": result["quality_metrics"]
            }
            
            logger.info(f"✅ 포즈 추정 완료 - 처리시간: {processing_time:.3f}초, 신뢰도: {result['confidence']:.3f}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"❌ 포즈 추정 처리 실패: {e}")
            raise
    
    def _tensor_to_cv_image(self, tensor: torch.Tensor) -> np.ndarray:
        """텐서를 OpenCV 이미지로 변환"""
        # [1, 3, H, W] → [H, W, 3]
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # [3, H, W] → [H, W, 3]
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        
        # 0-1 범위로 정규화
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        
        # numpy 변환 및 BGR로 변환
        image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image_bgr
    
    def _process_with_coreml(self, cv_image: np.ndarray) -> Dict[str, Any]:
        """CoreML 모델로 포즈 추정 (M3 Max 최적화)"""
        try:
            # CoreML 입력 준비
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # 모델 예측
            prediction = self.coreml_model.predict({'image': pil_image})
            
            # 결과 파싱 (모델에 따라 다름)
            keypoints = self._parse_coreml_output(prediction, cv_image.shape[:2])
            
            return {
                "success": True,
                "keypoints_18": keypoints["openpose_18"],
                "keypoints_raw": keypoints["raw"],
                "confidence": keypoints["confidence"],
                "orientation": self._estimate_orientation(keypoints["openpose_18"]),
                "angles": self._calculate_pose_angles(keypoints["openpose_18"]),
                "bbox": self._calculate_bbox(keypoints["openpose_18"]),
                "model_used": "CoreML",
                "quality_metrics": self._evaluate_pose_quality(keypoints["openpose_18"])
            }
            
        except Exception as e:
            logger.warning(f"⚠️ CoreML 처리 실패, MediaPipe로 fallback: {e}")
            return self._process_with_mediapipe(cv_image)
    
    def _process_with_mediapipe(self, cv_image: np.ndarray) -> Dict[str, Any]:
        """MediaPipe로 포즈 추정"""
        try:
            # RGB 변환
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # 포즈 검출
            results = self.pose_model.process(image_rgb)
            
            if not results.pose_landmarks:
                return self._create_empty_result("MediaPipe - 포즈 검출 실패")
            
            # 키포인트 추출 및 변환
            keypoints = self._extract_mediapipe_keypoints(results.pose_landmarks, cv_image.shape[:2])
            
            # OpenPose 18 형식으로 변환
            openpose_18 = self._convert_to_openpose_18(keypoints)
            
            # 신뢰도 계산
            confidence = self._calculate_pose_confidence(keypoints)
            
            return {
                "success": True,
                "keypoints_18": openpose_18,
                "keypoints_raw": keypoints,
                "confidence": confidence,
                "orientation": self._estimate_orientation(openpose_18),
                "angles": self._calculate_pose_angles(openpose_18),
                "bbox": self._calculate_bbox(openpose_18),
                "model_used": "MediaPipe",
                "quality_metrics": self._evaluate_pose_quality(openpose_18)
            }
            
        except Exception as e:
            logger.error(f"MediaPipe 처리 실패: {e}")
            return self._create_empty_result(f"MediaPipe 에러: {str(e)}")
    
    def _extract_mediapipe_keypoints(self, landmarks, image_shape: Tuple[int, int]) -> List[Dict]:
        """MediaPipe 랜드마크에서 키포인트 추출"""
        height, width = image_shape
        keypoints = []
        
        for idx, landmark in enumerate(landmarks.landmark):
            keypoint = {
                "id": idx,
                "x": landmark.x,           # 정규화된 좌표 (0-1)
                "y": landmark.y,
                "z": landmark.z,           # 상대적 깊이
                "visibility": landmark.visibility,
                "x_px": int(landmark.x * width),   # 픽셀 좌표
                "y_px": int(landmark.y * height),
                "confidence": landmark.visibility
            }
            keypoints.append(keypoint)
        
        return keypoints
    
    def _convert_to_openpose_18(self, mediapipe_keypoints: List[Dict]) -> List[List[float]]:
        """MediaPipe 키포인트를 OpenPose 18 형식으로 변환"""
        openpose_18 = [[0, 0, 0] for _ in range(18)]  # [x, y, confidence]
        
        # 매핑 적용
        for mp_idx, op_idx in self.MP_TO_OPENPOSE_18.items():
            if mp_idx < len(mediapipe_keypoints):
                mp_kp = mediapipe_keypoints[mp_idx]
                openpose_18[op_idx] = [
                    mp_kp["x_px"],
                    mp_kp["y_px"], 
                    mp_kp["confidence"]
                ]
        
        # Neck (1번) 계산: 양 어깨의 중점
        if openpose_18[2][2] > 0 and openpose_18[5][2] > 0:  # 양 어깨가 모두 검출된 경우
            neck_x = (openpose_18[2][0] + openpose_18[5][0]) / 2
            neck_y = (openpose_18[2][1] + openpose_18[5][1]) / 2
            neck_conf = min(openpose_18[2][2], openpose_18[5][2])
            openpose_18[1] = [neck_x, neck_y, neck_conf]
        
        return openpose_18
    
    def _calculate_pose_confidence(self, keypoints: List[Dict]) -> float:
        """전체 포즈의 신뢰도 계산"""
        if not keypoints:
            return 0.0
        
        # 주요 키포인트들의 가중평균
        major_keypoints = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # MediaPipe 인덱스
        
        total_weight = 0
        weighted_confidence = 0
        
        for idx in major_keypoints:
            if idx < len(keypoints):
                weight = 1.5 if idx in [11, 12, 23, 24] else 1.0  # 어깨와 엉덩이 가중치 증가
                confidence = keypoints[idx]["confidence"]
                
                weighted_confidence += confidence * weight
                total_weight += weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _estimate_orientation(self, keypoints_18: List[List[float]]) -> str:
        """신체 방향 추정"""
        try:
            # 어깨 키포인트 확인
            left_shoulder = keypoints_18[5]   # L-Shoulder
            right_shoulder = keypoints_18[2]  # R-Shoulder
            
            if left_shoulder[2] == 0 or right_shoulder[2] == 0:
                return "unknown"
            
            # 어깨 너비
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            
            # 이미지 너비 대비 어깨 너비 비율로 방향 추정
            if shoulder_width < 50:  # 픽셀 기준, 조정 필요
                return "side"
            elif shoulder_width < 100:
                return "diagonal"
            else:
                return "front"
                
        except Exception:
            return "unknown"
    
    def _calculate_pose_angles(self, keypoints_18: List[List[float]]) -> Dict[str, float]:
        """주요 관절 각도 계산"""
        def angle_between_points(p1, p2, p3):
            """세 점 사이의 각도 계산 (p2가 꼭지점)"""
            if any(p[2] == 0 for p in [p1, p2, p3]):  # 신뢰도 0인 경우
                return 0.0
            
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            
            return float(angle)
        
        angles = {}
        
        try:
            # 왼팔 각도 (어깨-팔꿈치-손목)
            angles["left_arm"] = angle_between_points(
                keypoints_18[5],   # L-Shoulder
                keypoints_18[6],   # L-Elbow  
                keypoints_18[7]    # L-Wrist
            )
            
            # 오른팔 각도
            angles["right_arm"] = angle_between_points(
                keypoints_18[2],   # R-Shoulder
                keypoints_18[3],   # R-Elbow
                keypoints_18[4]    # R-Wrist
            )
            
            # 왼다리 각도 (엉덩이-무릎-발목)
            angles["left_leg"] = angle_between_points(
                keypoints_18[11],  # L-Hip
                keypoints_18[12],  # L-Knee
                keypoints_18[13]   # L-Ankle
            )
            
            # 오른다리 각도
            angles["right_leg"] = angle_between_points(
                keypoints_18[8],   # R-Hip
                keypoints_18[9],   # R-Knee  
                keypoints_18[10]   # R-Ankle
            )
            
        except Exception as e:
            logger.warning(f"관절 각도 계산 실패: {e}")
        
        return angles
    
    def _calculate_bbox(self, keypoints_18: List[List[float]]) -> Dict[str, int]:
        """포즈 바운딩 박스 계산"""
        valid_points = [(x, y) for x, y, c in keypoints_18 if c > 0]
        
        if not valid_points:
            return {"x": 0, "y": 0, "width": 0, "height": 0}
        
        xs, ys = zip(*valid_points)
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # 여백 추가 (10%)
        margin_x = int((x_max - x_min) * 0.1)
        margin_y = int((y_max - y_min) * 0.1)
        
        return {
            "x": max(0, int(x_min - margin_x)),
            "y": max(0, int(y_min - margin_y)),
            "width": int(x_max - x_min + 2 * margin_x),
            "height": int(y_max - y_min + 2 * margin_y)
        }
    
    def _evaluate_pose_quality(self, keypoints_18: List[List[float]]) -> Dict[str, float]:
        """포즈 품질 평가"""
        # 검출된 키포인트 수
        detected_count = sum(1 for kp in keypoints_18 if kp[2] > 0.5)
        detection_rate = detected_count / 18
        
        # 주요 키포인트 검출 여부
        major_keypoints = [0, 1, 2, 5, 8, 11]  # nose, neck, shoulders, hips
        major_detected = sum(1 for idx in major_keypoints if keypoints_18[idx][2] > 0.5)
        major_rate = major_detected / len(major_keypoints)
        
        # 대칭성 평가
        symmetry_score = self._calculate_symmetry_score(keypoints_18)
        
        # 전체 품질 점수
        overall_quality = (detection_rate * 0.4 + major_rate * 0.4 + symmetry_score * 0.2)
        
        return {
            "overall": overall_quality,
            "detection_rate": detection_rate,
            "major_keypoints_rate": major_rate,
            "symmetry_score": symmetry_score,
            "detected_keypoints": detected_count
        }
    
    def _calculate_symmetry_score(self, keypoints_18: List[List[float]]) -> float:
        """좌우 대칭성 점수 계산"""
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
        
        for left_idx, right_idx in symmetric_pairs:
            left_kp = keypoints_18[left_idx]
            right_kp = keypoints_18[right_idx]
            
            if left_kp[2] > 0.5 and right_kp[2] > 0.5:
                # 신뢰도 차이
                conf_diff = abs(left_kp[2] - right_kp[2])
                symmetry_scores.append(1.0 - conf_diff)
        
        return np.mean(symmetry_scores) if symmetry_scores else 0.0
    
    def _get_pose_connections(self) -> List[List[int]]:
        """포즈 연결선 정보 (OpenPose 18 기준)"""
        return [
            [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],     # 상체
            [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], # 하체  
            [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],        # 머리
            [2, 16], [5, 17]                                      # 어깨-귀 연결
        ]
    
    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """빈 결과 생성"""
        return {
            "success": False,
            "keypoints_18": [[0, 0, 0] for _ in range(18)],
            "keypoints_raw": [],
            "confidence": 0.0,
            "orientation": "unknown",
            "angles": {},
            "bbox": {"x": 0, "y": 0, "width": 0, "height": 0},
            "model_used": "None",
            "quality_metrics": {
                "overall": 0.0,
                "detection_rate": 0.0,
                "major_keypoints_rate": 0.0,
                "symmetry_score": 0.0,
                "detected_keypoints": 0
            },
            "error": reason
        }
    
    def _parse_coreml_output(self, prediction: Dict, image_shape: Tuple[int, int]) -> Dict:
        """CoreML 출력 파싱 (모델별 구현 필요)"""
        # 실제 CoreML 모델 출력에 따라 구현
        # 예시 구조
        return {
            "raw": [],
            "openpose_18": [[0, 0, 0] for _ in range(18)],
            "confidence": 0.0
        }
    
    def visualize_pose(self, image: np.ndarray, keypoints_18: List[List[float]]) -> np.ndarray:
        """포즈 시각화"""
        vis_image = image.copy()
        
        # 키포인트 그리기
        for i, (x, y, conf) in enumerate(keypoints_18):
            if conf > 0.5:
                color = (0, 255, 0) if conf > 0.8 else (0, 255, 255)
                cv2.circle(vis_image, (int(x), int(y)), 5, color, -1)
                # 키포인트 번호 표시
                cv2.putText(vis_image, str(i), (int(x+5), int(y-5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # 연결선 그리기
        connections = self._get_pose_connections()
        for connection in connections:
            pt1_idx, pt2_idx = connection
            pt1 = keypoints_18[pt1_idx]
            pt2 = keypoints_18[pt2_idx]
            
            if pt1[2] > 0.5 and pt2[2] > 0.5:
                cv2.line(vis_image, 
                        (int(pt1[0]), int(pt1[1])), 
                        (int(pt2[0]), int(pt2[1])), 
                        (255, 0, 0), 2)
        
        return vis_image
    
    async def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_name": "PoseEstimation",
            "keypoint_format": "OpenPose_18",
            "device": self.device,
            "use_mps": self.use_mps,
            "use_coreml": self.use_coreml,
            "mediapipe_available": MP_AVAILABLE,
            "coreml_available": COREML_AVAILABLE,
            "initialized": self.is_initialized,
            "model_complexity": self.model_complexity,
            "min_detection_confidence": self.min_detection_confidence,
            "keypoints": list(self.OPENPOSE_18_KEYPOINTS.values())
        }
    
    async def cleanup(self):
        """리소스 정리"""
        if self.pose_model:
            self.pose_model.close()
            self.pose_model = None
        
        if self.coreml_model:
            del self.coreml_model
            self.coreml_model = None
        
        self.is_initialized = False
        logger.info("🧹 포즈 추정 스텝 리소스 정리 완료")