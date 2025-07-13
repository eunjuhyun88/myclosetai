"""
MyCloset AI 2단계: 포즈 추정 (Pose Estimation)
MediaPipe + OpenPose 기반 18개 키포인트 검출 시스템
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import logging
from typing import Dict, List, Optional, Tuple, Any
import asyncio
from pathlib import Path
import math

class OpenPoseNet(nn.Module):
    """OpenPose 기반 포즈 추정 네트워크"""
    
    def __init__(self, num_keypoints: int = 18):
        super().__init__()
        self.num_keypoints = num_keypoints
        
        # VGG-19 백본 (경량화)
        self.backbone = self._create_vgg_backbone()
        
        # PAF (Part Affinity Fields) 브랜치
        self.paf_stages = self._create_paf_stages()
        
        # 키포인트 히트맵 브랜치
        self.keypoint_stages = self._create_keypoint_stages()

    def _create_vgg_backbone(self):
        """VGG-19 기반 특징 추출기"""
        layers = []
        
        # Block 1
        layers.extend([
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        ])
        
        # Block 2
        layers.extend([
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        ])
        
        # Block 3
        layers.extend([
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        ])
        
        # Block 4
        layers.extend([
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        ])
        
        return nn.Sequential(*layers)

    def _create_paf_stages(self):
        """PAF (Part Affinity Fields) 스테이지 생성"""
        stages = nn.ModuleList()
        
        # Stage 1
        stage1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 38, 1)  # 19 limbs * 2 (x, y vectors)
        )
        stages.append(stage1)
        
        # Stages 2-6
        for i in range(5):
            stage = nn.Sequential(
                nn.Conv2d(128 + 38 + self.num_keypoints, 128, 7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 38, 1)
            )
            stages.append(stage)
        
        return stages

    def _create_keypoint_stages(self):
        """키포인트 히트맵 스테이지 생성"""
        stages = nn.ModuleList()
        
        # Stage 1
        stage1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.num_keypoints, 1)
        )
        stages.append(stage1)
        
        # Stages 2-6
        for i in range(5):
            stage = nn.Sequential(
                nn.Conv2d(128 + 38 + self.num_keypoints, 128, 7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, self.num_keypoints, 1)
            )
            stages.append(stage)
        
        return stages

    def forward(self, x):
        # 백본 특징 추출
        features = self.backbone(x)
        
        # 초기 PAF와 키포인트 예측
        paf_out = self.paf_stages[0](features)
        keypoint_out = self.keypoint_stages[0](features)
        
        paf_outputs = [paf_out]
        keypoint_outputs = [keypoint_out]
        
        # 반복적 정제
        for i in range(1, len(self.paf_stages)):
            # 이전 예측과 특징을 결합
            combined = torch.cat([features, paf_out, keypoint_out], dim=1)
            
            # PAF 정제
            paf_out = self.paf_stages[i](combined)
            paf_outputs.append(paf_out)
            
            # 키포인트 정제
            keypoint_out = self.keypoint_stages[i](combined)
            keypoint_outputs.append(keypoint_out)
        
        return paf_outputs, keypoint_outputs

class MediaPipeWrapper:
    """MediaPipe 포즈 추정 래퍼"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect_pose(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """MediaPipe로 포즈 검출"""
        try:
            # BGR을 RGB로 변환
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 포즈 검출
            results = self.pose.process(rgb_image)
            
            if results.pose_landmarks:
                # 키포인트 추출
                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    keypoints.append([
                        landmark.x * image.shape[1],  # x 좌표
                        landmark.y * image.shape[0],  # y 좌표
                        landmark.visibility  # 가시성 점수
                    ])
                
                return {
                    "keypoints": keypoints,
                    "pose_landmarks": results.pose_landmarks,
                    "pose_world_landmarks": results.pose_world_landmarks
                }
            
            return None
            
        except Exception as e:
            logging.error(f"MediaPipe 포즈 검출 오류: {e}")
            return None

class PoseEstimationStep:
    """2단계: 포즈 추정 실행 클래스"""
    
    # 18개 키포인트 정의 (OpenPose 형식)
    KEYPOINT_NAMES = [
        "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
        "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
        "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
        "left_eye", "right_ear", "left_ear"
    ]
    
    # 연결 관계 (limbs)
    POSE_PAIRS = [
        [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
        [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
        [2, 16], [5, 17]
    ]
    
    def __init__(self, config, device, model_loader):
        self.config = config
        self.device = device
        self.model_loader = model_loader
        self.logger = logging.getLogger(__name__)
        
        # 모델 초기화
        self.openpose_model = None
        self.mediapipe_model = None
        self.model_loaded = False
        
        # MediaPipe 래퍼 초기화
        self.mediapipe_wrapper = MediaPipeWrapper()
        
        # 후처리 파라미터
        self.confidence_threshold = 0.1
        self.nms_threshold = 0.05

    async def process(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """포즈 추정 메인 처리"""
        try:
            # 모델 로드 (필요시)
            if not self.model_loaded:
                await self._load_models()
            
            # OpenPose 추론
            openpose_result = await self._openpose_inference(input_tensor)
            
            # MediaPipe 추론 (보조)
            mediapipe_result = await self._mediapipe_inference(input_tensor)
            
            # 결과 융합
            final_result = await self._fuse_results(openpose_result, mediapipe_result)
            
            # 후처리
            processed_result = await self._postprocess(final_result, input_tensor.shape[2:])
            
            return {
                "keypoints": processed_result["keypoints"],
                "pose_heatmaps": processed_result["heatmaps"],
                "paf_fields": processed_result["paf_fields"],
                "confidence_scores": processed_result["confidence_scores"],
                "pose_skeleton": processed_result["skeleton"],
                "metadata": {
                    "num_keypoints": len(self.KEYPOINT_NAMES),
                    "detection_method": "OpenPose + MediaPipe Fusion",
                    "processing_time": processed_result["processing_time"]
                }
            }
            
        except Exception as e:
            self.logger.error(f"포즈 추정 처리 중 오류: {str(e)}")
            raise

    async def _load_models(self):
        """모델 로드 및 초기화"""
        try:
            # OpenPose 모델 로드
            cached_model = self.model_loader.memory_manager.get_cached_model("pose_estimation")
            
            if cached_model is not None:
                self.openpose_model = cached_model
                self.logger.info("캐시된 포즈 추정 모델 로드")
            else:
                # 새 OpenPose 모델 생성
                self.openpose_model = OpenPoseNet(num_keypoints=18)
                
                # 사전 훈련된 가중치 로드
                checkpoint_path = Path("models/checkpoints/pose_estimation_openpose.pth")
                if checkpoint_path.exists():
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    self.openpose_model.load_state_dict(checkpoint['model_state_dict'])
                    self.logger.info("OpenPose 가중치 로드 완료")
                else:
                    self.logger.warning("OpenPose 가중치를 찾을 수 없습니다. 랜덤 초기화 사용")
                
                # 모델을 디바이스로 이동
                self.openpose_model = self.openpose_model.to(self.device)
                
                # FP16 최적화
                if self.config.use_fp16 and self.device.type == "mps":
                    self.openpose_model = self.openpose_model.half()
                
                # 평가 모드
                self.openpose_model.eval()
                
                # 모델 캐싱
                self.model_loader.memory_manager.cache_model("pose_estimation", self.openpose_model)
            
            self.model_loaded = True
            self.logger.info("포즈 추정 모델 로드 완료")
            
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {str(e)}")
            raise

    async def _openpose_inference(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """OpenPose 모델 추론"""
        with torch.no_grad():
            if self.config.use_fp16 and self.device.type == "mps":
                input_tensor = input_tensor.half()
            
            # 추론 실행
            paf_outputs, keypoint_outputs = self.openpose_model(input_tensor)
            
            # 마지막 스테이지 결과 사용
            final_paf = paf_outputs[-1]
            final_keypoints = keypoint_outputs[-1]
            
            return {
                "paf_fields": final_paf,
                "keypoint_heatmaps": final_keypoints,
                "all_stages": {
                    "paf": paf_outputs,
                    "keypoints": keypoint_outputs
                }
            }

    async def _mediapipe_inference(self, input_tensor: torch.Tensor) -> Optional[Dict[str, Any]]:
        """MediaPipe 모델 추론"""
        try:
            # 텐서를 numpy 배열로 변환
            if input_tensor.dim() == 4:
                image_np = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
            else:
                image_np = input_tensor.cpu().numpy().transpose(1, 2, 0)
            
            # [0, 1] → [0, 255] 변환
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            
            # MediaPipe 포즈 검출
            result = self.mediapipe_wrapper.detect_pose(image_np)
            
            return result
            
        except Exception as e:
            self.logger.warning(f"MediaPipe 추론 실패: {e}")
            return None

    async def _fuse_results(self, openpose_result: Dict[str, Any], mediapipe_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """OpenPose와 MediaPipe 결과 융합"""
        import time
        start_time = time.time()
        
        # OpenPose 키포인트 추출
        keypoint_heatmaps = openpose_result["keypoint_heatmaps"].cpu().numpy()
        paf_fields = openpose_result["paf_fields"].cpu().numpy()
        
        # 키포인트 검출
        openpose_keypoints = self._extract_keypoints_from_heatmaps(keypoint_heatmaps[0])
        
        # MediaPipe 결과가 있으면 융합
        if mediapipe_result and mediapipe_result["keypoints"]:
            mediapipe_keypoints = self._convert_mediapipe_to_openpose(mediapipe_result["keypoints"])
            
            # 신뢰도 기반 융합
            fused_keypoints = self._confidence_based_fusion(openpose_keypoints, mediapipe_keypoints)
        else:
            fused_keypoints = openpose_keypoints
        
        processing_time = time.time() - start_time
        
        return {
            "keypoints": fused_keypoints,
            "heatmaps": keypoint_heatmaps[0],
            "paf_fields": paf_fields[0],
            "processing_time": processing_time
        }

    def _extract_keypoints_from_heatmaps(self, heatmaps: np.ndarray) -> List[Tuple[float, float, float]]:
        """히트맵에서 키포인트 추출"""
        keypoints = []
        
        for i in range(heatmaps.shape[0]):
            heatmap = heatmaps[i]
            
            # 최대값 위치 찾기
            max_val = np.max(heatmap)
            
            if max_val > self.confidence_threshold:
                max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                y, x = max_idx
                
                # 서브픽셀 정확도로 정제
                if 1 <= x < heatmap.shape[1] - 1 and 1 <= y < heatmap.shape[0] - 1:
                    dx = 0.5 * (heatmap[y, x + 1] - heatmap[y, x - 1])
                    dy = 0.5 * (heatmap[y + 1, x] - heatmap[y - 1, x])
                    x += dx
                    y += dy
                
                keypoints.append((float(x), float(y), float(max_val)))
            else:
                keypoints.append((0.0, 0.0, 0.0))
        
        return keypoints

    def _convert_mediapipe_to_openpose(self, mediapipe_keypoints: List[List[float]]) -> List[Tuple[float, float, float]]:
        """MediaPipe 키포인트를 OpenPose 형식으로 변환"""
        # MediaPipe → OpenPose 매핑 (33개 → 18개)
        mapping = {
            0: 0,   # nose
            11: 1,  # neck (추정)
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
        
        openpose_keypoints = []
        
        for i in range(18):
            if i in mapping and mapping[i] < len(mediapipe_keypoints):
                mp_kp = mediapipe_keypoints[mapping[i]]
                openpose_keypoints.append((mp_kp[0], mp_kp[1], mp_kp[2]))
            else:
                openpose_keypoints.append((0.0, 0.0, 0.0))
        
        # neck 추정 (양쪽 어깨의 중점)
        if len(mediapipe_keypoints) > 12:
            left_shoulder = mediapipe_keypoints[11]
            right_shoulder = mediapipe_keypoints[12]
            if left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5:
                neck_x = (left_shoulder[0] + right_shoulder[0]) / 2
                neck_y = (left_shoulder[1] + right_shoulder[1]) / 2
                neck_conf = min(left_shoulder[2], right_shoulder[2])
                openpose_keypoints[1] = (neck_x, neck_y, neck_conf)
        
        return openpose_keypoints

    def _confidence_based_fusion(self, openpose_kp: List[Tuple[float, float, float]], mediapipe_kp: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """신뢰도 기반 키포인트 융합"""
        fused_keypoints = []
        
        for i in range(len(openpose_kp)):
            op_conf = openpose_kp[i][2] if i < len(openpose_kp) else 0.0
            mp_conf = mediapipe_kp[i][2] if i < len(mediapipe_kp) else 0.0
            
            if op_conf > mp_conf:
                fused_keypoints.append(openpose_kp[i])
            elif mp_conf > 0.1:  # MediaPipe 최소 신뢰도
                fused_keypoints.append(mediapipe_kp[i])
            else:
                fused_keypoints.append(openpose_kp[i])
        
        return fused_keypoints

    async def _postprocess(self, fused_result: Dict[str, Any], image_size: Tuple[int, int]) -> Dict[str, Any]:
        """결과 후처리"""
        keypoints = fused_result["keypoints"]
        
        # 이미지 크기에 맞게 스케일링
        height, width = image_size
        heatmap_height, heatmap_width = fused_result["heatmaps"].shape[1:3]
        
        scale_x = width / heatmap_width
        scale_y = height / heatmap_height
        
        scaled_keypoints = []
        confidence_scores = []
        
        for x, y, conf in keypoints:
            if conf > 0:
                scaled_x = x * scale_x
                scaled_y = y * scale_y
                scaled_keypoints.append((scaled_x, scaled_y, conf))
            else:
                scaled_keypoints.append((0.0, 0.0, 0.0))
            
            confidence_scores.append(conf)
        
        # 스켈레톤 연결 생성
        skeleton = self._create_skeleton(scaled_keypoints)
        
        return {
            "keypoints": scaled_keypoints,
            "heatmaps": fused_result["heatmaps"],
            "paf_fields": fused_result["paf_fields"],
            "confidence_scores": confidence_scores,
            "skeleton": skeleton,
            "processing_time": fused_result["processing_time"]
        }

    def _create_skeleton(self, keypoints: List[Tuple[float, float, float]]) -> List[Dict[str, Any]]:
        """키포인트 연결로 스켈레톤 생성"""
        skeleton = []
        
        for pair in self.POSE_PAIRS:
            point_a_idx, point_b_idx = pair
            
            if (point_a_idx < len(keypoints) and point_b_idx < len(keypoints)):
                point_a = keypoints[point_a_idx]
                point_b = keypoints[point_b_idx]
                
                # 두 점 모두 신뢰도가 있는 경우만
                if point_a[2] > self.confidence_threshold and point_b[2] > self.confidence_threshold:
                    skeleton.append({
                        "from": {
                            "x": point_a[0],
                            "y": point_a[1],
                            "name": self.KEYPOINT_NAMES[point_a_idx],
                            "confidence": point_a[2]
                        },
                        "to": {
                            "x": point_b[0],
                            "y": point_b[1],
                            "name": self.KEYPOINT_NAMES[point_b_idx],
                            "confidence": point_b[2]
                        },
                        "limb_confidence": min(point_a[2], point_b[2])
                    })
        
        return skeleton

    def visualize_pose(self, image: np.ndarray, pose_result: Dict[str, Any], save_path: Optional[str] = None) -> np.ndarray:
        """포즈 시각화"""
        vis_image = image.copy()
        keypoints = pose_result["keypoints"]
        skeleton = pose_result["skeleton"]
        
        # 키포인트 그리기
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > self.confidence_threshold:
                color = (0, 255, 0) if conf > 0.5 else (0, 255, 255)
                cv2.circle(vis_image, (int(x), int(y)), 3, color, -1)
                cv2.putText(vis_image, str(i), (int(x), int(y-5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # 스켈레톤 그리기
        for limb in skeleton:
            point_a = limb["from"]
            point_b = limb["to"]
            
            color = (255, 0, 0) if limb["limb_confidence"] > 0.5 else (128, 128, 128)
            thickness = 2 if limb["limb_confidence"] > 0.5 else 1
            
            cv2.line(vis_image, 
                    (int(point_a["x"]), int(point_a["y"])),
                    (int(point_b["x"]), int(point_b["y"])),
                    color, thickness)
        
        # 저장 (옵션)
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image

    def get_pose_statistics(self, pose_result: Dict[str, Any]) -> Dict[str, Any]:
        """포즈 통계 정보"""
        keypoints = pose_result["keypoints"]
        confidence_scores = pose_result["confidence_scores"]
        
        detected_keypoints = sum(1 for conf in confidence_scores if conf > self.confidence_threshold)
        avg_confidence = np.mean([conf for conf in confidence_scores if conf > 0])
        
        # 신체 각도 계산 (예: 어깨 기울기)
        angles = self._calculate_body_angles(keypoints)
        
        return {
            "total_keypoints": len(keypoints),
            "detected_keypoints": detected_keypoints,
            "detection_rate": detected_keypoints / len(keypoints),
            "average_confidence": float(avg_confidence) if avg_confidence else 0.0,
            "max_confidence": max(confidence_scores),
            "min_confidence": min([conf for conf in confidence_scores if conf > 0], default=0),
            "body_angles": angles,
            "pose_completeness": detected_keypoints / len(keypoints)
        }

    def _calculate_body_angles(self, keypoints: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """신체 각도 계산"""
        angles = {}
        
        try:
            # 어깨 기울기 (2: right_shoulder, 5: left_shoulder)
            if (len(keypoints) > 5 and keypoints[2][2] > 0 and keypoints[5][2] > 0):
                right_shoulder = keypoints[2]
                left_shoulder = keypoints[5]
                
                dx = right_shoulder[0] - left_shoulder[0]
                dy = right_shoulder[1] - left_shoulder[1]
                shoulder_angle = math.degrees(math.atan2(dy, dx))
                angles["shoulder_tilt"] = shoulder_angle
            
            # 상체 기울기 (1: neck, 8: right_hip, 11: left_hip)
            if (len(keypoints) > 11 and keypoints[1][2] > 0 and 
                keypoints[8][2] > 0 and keypoints[11][2] > 0):
                neck = keypoints[1]
                right_hip = keypoints[8]
                left_hip = keypoints[11]
                
                hip_center_x = (right_hip[0] + left_hip[0]) / 2
                hip_center_y = (right_hip[1] + left_hip[1]) / 2
                
                dx = neck[0] - hip_center_x
                dy = neck[1] - hip_center_y
                torso_angle = math.degrees(math.atan2(dx, dy))
                angles["torso_lean"] = torso_angle
                
        except Exception as e:
            self.logger.warning(f"각도 계산 실패: {e}")
        
        return angles

    async def warmup(self, dummy_input: torch.Tensor):
        """모델 워밍업"""
        if not self.model_loaded:
            await self._load_models()
        
        with torch.no_grad():
            _ = await self._openpose_inference(dummy_input)
        
        self.logger.info("포즈 추정 모델 워밍업 완료")

    def cleanup(self):
        """리소스 정리"""
        if self.openpose_model is not None:
            del self.openpose_model
            self.openpose_model = None
        
        if self.mediapipe_wrapper:
            self.mediapipe_wrapper.pose.close()
            self.mediapipe_wrapper = None
            
        self.model_loaded = False
        self.logger.info("포즈 추정 모델 리소스 정리 완료")

# 사용 예시
async def example_usage():
    """포즈 추정 사용 예시"""
    from ..utils.memory_manager import GPUMemoryManager
    from ..utils.model_loader import ModelLoader
    
    # 설정
    class Config:
        image_size = 512
        use_fp16 = True
    
    config = Config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 메모리 매니저 및 모델 로더
    memory_manager = GPUMemoryManager(device, 16.0)
    model_loader = ModelLoader(device, True)
    model_loader.memory_manager = memory_manager
    
    # 포즈 추정 단계 초기화
    pose_estimation = PoseEstimationStep(config, device, model_loader)
    
    # 더미 입력 생성
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    
    # 처리
    result = await pose_estimation.process(dummy_input)
    
    print(f"포즈 추정 완료 - {len(result['keypoints'])}개 키포인트 검출")
    print(f"처리 시간: {result['metadata']['processing_time']:.2f}초")
    
    # 통계 정보
    stats = pose_estimation.get_pose_statistics(result)
    print(f"검출률: {stats['detection_rate']:.2%}")
    print(f"평균 신뢰도: {stats['average_confidence']:.3f}")
    
    # 각도 정보
    if stats['body_angles']:
        print("신체 각도:")
        for angle_name, angle_value in stats['body_angles'].items():
            print(f"  {angle_name}: {angle_value:.1f}°")

if __name__ == "__main__":
    asyncio.run(example_usage())