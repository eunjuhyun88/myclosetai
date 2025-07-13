"""
4단계: 기하학적 매칭 (Geometric Matching) - 통합 버전
두 파일의 장점을 모두 포함한 완전한 TPS 변환 + 메쉬 워핑 시스템
M3 Max 최적화 포함
"""
import os
import logging
import time
import math
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import json

logger = logging.getLogger(__name__)

class GeometricMatchingStep:
    """기하학적 매칭 스텝 - TPS 변환 + 메쉬 워핑 통합 버전"""
    
    # 의류별 주요 매칭 포인트 정의 (더 세분화)
    CLOTHING_KEYPOINTS = {
        'shirt': ['left_shoulder', 'right_shoulder', 'left_sleeve', 'right_sleeve', 
                 'collar', 'hem', 'left_armpit', 'right_armpit'],
        'pants': ['waist_left', 'waist_right', 'left_leg', 'right_leg', 
                 'left_ankle', 'right_ankle', 'left_thigh', 'right_thigh'],
        'dress': ['left_shoulder', 'right_shoulder', 'waist_left', 'waist_right', 
                 'hem_left', 'hem_right', 'left_hip', 'right_hip'],
        'skirt': ['waist_left', 'waist_right', 'hem_left', 'hem_right']
    }
    
    # OpenPose 18 키포인트와 의류 키포인트 매핑
    POSE_TO_CLOTHING = {
        'shirt': {
            5: 'left_shoulder',   # left_shoulder
            2: 'right_shoulder',  # right_shoulder  
            7: 'left_sleeve',     # left_elbow
            4: 'right_sleeve',    # right_elbow
            1: 'collar',          # neck
            11: 'hem',            # left_hip (하단)
        },
        'pants': {
            11: 'waist_left',     # left_hip
            8: 'waist_right',     # right_hip
            12: 'left_leg',       # left_knee
            9: 'right_leg',       # right_knee
            13: 'left_ankle',     # left_ankle
            10: 'right_ankle',    # right_ankle
        },
        'dress': {
            5: 'left_shoulder',   # left_shoulder
            2: 'right_shoulder',  # right_shoulder
            11: 'waist_left',     # left_hip
            8: 'waist_right',     # right_hip
            13: 'hem_left',       # left_ankle
            10: 'hem_right',      # right_ankle
        }
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
        
        # TPS 변환 설정
        self.tps_config = self.config.get('tps_transform', {
            'regularization': 0.001,
            'smoothing': 0.01,
            'kernel': 'thin_plate_spline',
            'mesh_density': 15
        })
        
        # 매칭 알고리즘 설정
        self.matching_config = self.config.get('matching', {
            'feature_method': 'sift',
            'keypoint_threshold': 0.02,
            'outlier_threshold': 2.0,
            'max_keypoints': 50,
            'matching_threshold': 50.0,
            'min_matching_points': 4
        })
        
        # 성능 최적화 설정 (M3 Max)
        self.use_mps = device == 'mps' and torch.backends.mps.is_available()
        
        # 변환 객체들
        self.tps_solver = None
        self.tps_transformer = None
        self.mesh_warper = None
        
        self.is_initialized = False
        
        logger.info(f"🎯 기하학적 매칭 스텝 초기화 - 디바이스: {device}, MPS: {self.use_mps}")
    
    async def initialize(self) -> bool:
        """초기화"""
        try:
            logger.info("🔄 기하학적 매칭 초기화 중...")
            
            # TPS 솔버 초기화 (수학적으로 정확한 버전)
            self.tps_solver = TPSSolver(
                device=self.device, 
                reg_factor=self.tps_config['regularization']
            )
            
            # TPS 변환기 초기화 (RBF 기반)
            self.tps_transformer = ThinPlateSplineTransform(
                regularization=self.tps_config['regularization'],
                smoothing=self.tps_config['smoothing']
            )
            
            # 메쉬 워핑 초기화
            self.mesh_warper = MeshBasedWarping(
                mesh_size=self.tps_config['mesh_density']
            )
            
            self.is_initialized = True
            logger.info("✅ 기하학적 매칭 초기화 완료")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 기하학적 매칭 초기화 실패: {e}")
            self.is_initialized = False
            return False
    
    def process(
        self,
        person_image_tensor: torch.Tensor,
        clothing_image_tensor: torch.Tensor,
        clothing_mask: torch.Tensor,
        pose_keypoints: List[List[float]],
        parsing_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        기하학적 매칭 처리 (통합 버전)
        
        Args:
            person_image_tensor: 사용자 이미지 텐서 [1, 3, H, W]
            clothing_image_tensor: 의류 이미지 텐서 [1, 3, H, W]  
            clothing_mask: 의류 마스크 텐서 [1, 1, H, W]
            pose_keypoints: 18개 포즈 키포인트
            parsing_result: 인체 파싱 결과
            
        Returns:
            처리 결과 딕셔너리
        """
        if not self.is_initialized:
            raise RuntimeError("기하학적 매칭이 초기화되지 않았습니다.")
        
        start_time = time.time()
        
        try:
            # 1. 의류 타입 결정
            clothing_type = self._determine_clothing_type(parsing_result, pose_keypoints)
            
            # 2. 텐서를 numpy 배열로 변환
            person_img = self._tensor_to_numpy(person_image_tensor)
            cloth_img = self._tensor_to_numpy(clothing_image_tensor)
            cloth_mask = self._tensor_to_numpy(clothing_mask, is_mask=True)
            
            # 3. 신체 키포인트 추출 (포즈 기반)
            body_keypoints = self._extract_body_keypoints(pose_keypoints, clothing_type)
            
            # 4. 의류 키포인트 추출 (윤곽선 기반)
            clothing_keypoints = self._extract_clothing_keypoints_from_contour(
                cloth_img, cloth_mask, clothing_type
            )
            
            # 5. 키포인트 매칭 (Hungarian 알고리즘 + 직접 매칭)
            matched_pairs = self._match_keypoints_advanced(body_keypoints, clothing_keypoints)
            
            # 6. TPS 변환 매트릭스 계산
            tps_matrix = self._calculate_tps_transform(matched_pairs)
            
            # 7. TPS 변환 적용
            warped_cloth, warped_mask = self._apply_tps_transform(
                cloth_img, cloth_mask, matched_pairs
            )
            
            # 8. 메쉬 기반 세밀 조정
            refined_cloth, refined_mask = self._apply_mesh_refinement(
                warped_cloth, warped_mask, matched_pairs
            )
            
            # 9. 결과 품질 평가
            quality_metrics = self._evaluate_matching_quality_comprehensive(
                cloth_img, refined_cloth, matched_pairs, body_keypoints, clothing_keypoints
            )
            
            # 10. 변형 영역 계산
            deformation_regions = self._calculate_deformation_regions(matched_pairs, clothing_type)
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "warped_clothing": self._numpy_to_tensor(refined_cloth),
                "warped_mask": self._numpy_to_tensor(refined_mask, is_mask=True),
                "tps_matrix": tps_matrix,
                "matched_pairs": matched_pairs,
                "body_keypoints": body_keypoints,
                "clothing_keypoints": clothing_keypoints,
                "clothing_type": clothing_type,
                "transform_quality": quality_metrics,
                "deformation_regions": deformation_regions,
                "confidence": float(quality_metrics.get('overall_score', 0.7)),
                "processing_time": processing_time,
                "num_matched_points": len(matched_pairs),
                "transform_method": "TPS + Mesh Hybrid"
            }
            
            logger.info(f"✅ 기하학적 매칭 완료 - 처리시간: {processing_time:.3f}초, 매칭 포인트: {len(matched_pairs)}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 기하학적 매칭 처리 실패: {e}")
            raise
    
    def _determine_clothing_type(self, parsing_result: Dict[str, Any], pose_keypoints: List[List[float]]) -> str:
        """의류 타입 결정 (향상된 로직)"""
        try:
            # 파싱 결과에서 감지된 신체 부위 분석
            detected_parts = parsing_result.get('body_parts_detected', {})
            
            # 의류 관련 부위 확인
            has_upper_clothes = any(part in detected_parts for part in ['upper_clothes', 'dress', 'coat', 'top'])
            has_lower_clothes = any(part in detected_parts for part in ['pants', 'skirt', 'bottom'])
            has_dress = 'dress' in detected_parts
            
            # 포즈에서 신체 영역 분석
            has_upper_body = len([kp for kp in pose_keypoints[:11] if len(kp) > 2 and kp[2] > 0.3]) >= 3
            has_lower_body = len([kp for kp in pose_keypoints[11:] if len(kp) > 2 and kp[2] > 0.3]) >= 2
            
            # 의류 타입 결정 로직 (우선순위 기반)
            if has_dress and has_upper_body and has_lower_body:
                return 'dress'
            elif has_upper_clothes and has_upper_body:
                return 'shirt'
            elif has_lower_clothes and has_lower_body:
                return 'pants'
            elif 'skirt' in detected_parts:
                return 'skirt'
            else:
                # 기본값: 포즈 기반 추정
                return 'shirt' if has_upper_body else 'pants'
            
        except Exception as e:
            logger.warning(f"의류 타입 결정 실패: {e}")
            return 'shirt'  # 기본값
    
    def _tensor_to_numpy(self, tensor: torch.Tensor, is_mask: bool = False) -> np.ndarray:
        """텐서를 numpy 배열로 변환"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        if is_mask:
            # 마스크의 경우 2D로 변환
            if tensor.dim() == 3:
                tensor = tensor.squeeze(0)
            return (tensor.cpu().numpy() * 255).astype(np.uint8)
        else:
            # 이미지의 경우 [3, H, W] → [H, W, 3]으로 변환
            if tensor.shape[0] == 3:
                tensor = tensor.permute(1, 2, 0)
            
            # 0-1 범위를 0-255로 변환
            if tensor.max() <= 1.0:
                tensor = tensor * 255
            
            return tensor.cpu().numpy().astype(np.uint8)
    
    def _numpy_to_tensor(self, array: np.ndarray, is_mask: bool = False) -> torch.Tensor:
        """numpy 배열을 텐서로 변환"""
        if is_mask:
            # 마스크: [H, W] → [1, 1, H, W]
            if array.ndim == 2:
                tensor = torch.from_numpy(array / 255.0).float()
                return tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        else:
            # 이미지: [H, W, 3] → [1, 3, H, W]
            if array.ndim == 3:
                tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
                return tensor.unsqueeze(0).to(self.device)
        
        return torch.from_numpy(array).to(self.device)
    
    def _extract_body_keypoints(self, pose_keypoints: List[List[float]], clothing_type: str) -> Dict[str, Tuple[float, float]]:
        """신체 키포인트 추출 (포즈 기반)"""
        body_keypoints = {}
        
        try:
            # 의류 타입에 따른 관련 키포인트 추출
            if clothing_type in self.POSE_TO_CLOTHING:
                for pose_idx, clothing_name in self.POSE_TO_CLOTHING[clothing_type].items():
                    if pose_idx < len(pose_keypoints):
                        kp = pose_keypoints[pose_idx]
                        if len(kp) > 2 and kp[2] > 0.3:  # 신뢰도 체크
                            body_keypoints[clothing_name] = (kp[0], kp[1])
            
            # 추가 계산된 포인트들
            if clothing_type == 'shirt':
                # 겨드랑이 계산
                if 5 < len(pose_keypoints) and 7 < len(pose_keypoints):
                    left_shoulder = pose_keypoints[5]
                    left_elbow = pose_keypoints[7]
                    if len(left_shoulder) > 2 and len(left_elbow) > 2:
                        if left_shoulder[2] > 0.3 and left_elbow[2] > 0.3:
                            armpit_x = left_shoulder[0] + (left_elbow[0] - left_shoulder[0]) * 0.3
                            armpit_y = left_shoulder[1] + (left_elbow[1] - left_shoulder[1]) * 0.3
                            body_keypoints['left_armpit'] = (armpit_x, armpit_y)
                
                if 2 < len(pose_keypoints) and 4 < len(pose_keypoints):
                    right_shoulder = pose_keypoints[2]
                    right_elbow = pose_keypoints[4]
                    if len(right_shoulder) > 2 and len(right_elbow) > 2:
                        if right_shoulder[2] > 0.3 and right_elbow[2] > 0.3:
                            armpit_x = right_shoulder[0] + (right_elbow[0] - right_shoulder[0]) * 0.3
                            armpit_y = right_shoulder[1] + (right_elbow[1] - right_shoulder[1]) * 0.3
                            body_keypoints['right_armpit'] = (armpit_x, armpit_y)
            
            elif clothing_type == 'pants':
                # 허벅지 중간점 계산
                if 11 < len(pose_keypoints) and 12 < len(pose_keypoints):
                    left_hip = pose_keypoints[11]
                    left_knee = pose_keypoints[12]
                    if len(left_hip) > 2 and len(left_knee) > 2:
                        if left_hip[2] > 0.3 and left_knee[2] > 0.3:
                            thigh_x = (left_hip[0] + left_knee[0]) / 2
                            thigh_y = (left_hip[1] + left_knee[1]) / 2
                            body_keypoints['left_thigh'] = (thigh_x, thigh_y)
                
                if 8 < len(pose_keypoints) and 9 < len(pose_keypoints):
                    right_hip = pose_keypoints[8]
                    right_knee = pose_keypoints[9]
                    if len(right_hip) > 2 and len(right_knee) > 2:
                        if right_hip[2] > 0.3 and right_knee[2] > 0.3:
                            thigh_x = (right_hip[0] + right_knee[0]) / 2
                            thigh_y = (right_hip[1] + right_knee[1]) / 2
                            body_keypoints['right_thigh'] = (thigh_x, thigh_y)
            
        except Exception as e:
            logger.warning(f"신체 키포인트 추출 실패: {e}")
        
        return body_keypoints
    
    def _extract_clothing_keypoints_from_contour(
        self, 
        cloth_img: np.ndarray, 
        cloth_mask: np.ndarray, 
        clothing_type: str
    ) -> Dict[str, Tuple[float, float]]:
        """윤곽선 기반 의류 키포인트 추출 (향상된 버전)"""
        
        clothing_keypoints = {}
        
        try:
            # 윤곽선 찾기
            contours, _ = cv2.findContours(cloth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return self._generate_default_keypoints(cloth_img.shape[1], cloth_img.shape[0], clothing_type)
            
            # 가장 큰 윤곽선 선택
            main_contour = max(contours, key=cv2.contourArea)
            
            # 바운딩 박스
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # 의류 타입별 키포인트 추출
            if clothing_type == 'shirt':
                clothing_keypoints = self._extract_shirt_keypoints_detailed(main_contour, x, y, w, h)
            elif clothing_type == 'pants':
                clothing_keypoints = self._extract_pants_keypoints_detailed(main_contour, x, y, w, h)
            elif clothing_type == 'dress':
                clothing_keypoints = self._extract_dress_keypoints_detailed(main_contour, x, y, w, h)
            elif clothing_type == 'skirt':
                clothing_keypoints = self._extract_skirt_keypoints_detailed(main_contour, x, y, w, h)
            else:
                clothing_keypoints = self._generate_default_keypoints(w, h, clothing_type)
            
        except Exception as e:
            logger.warning(f"의류 키포인트 추출 실패: {e}")
            # 실패 시 기본 키포인트 생성
            h, w = cloth_img.shape[:2]
            clothing_keypoints = self._generate_default_keypoints(w, h, clothing_type)
        
        return clothing_keypoints
    
    def _extract_shirt_keypoints_detailed(
        self, 
        contour: np.ndarray, 
        x: int, y: int, w: int, h: int
    ) -> Dict[str, Tuple[float, float]]:
        """상의 키포인트 상세 추출"""
        keypoints = {}
        
        try:
            # 어깨 라인 (상단 15% 지점)
            shoulder_y = y + int(h * 0.15)
            left_shoulder = self._find_contour_point_at_height(contour, shoulder_y, 'left')
            right_shoulder = self._find_contour_point_at_height(contour, shoulder_y, 'right')
            
            keypoints['left_shoulder'] = (left_shoulder[0], left_shoulder[1])
            keypoints['right_shoulder'] = (right_shoulder[0], right_shoulder[1])
            
            # 겨드랑이 (어깨에서 25% 아래)
            armpit_y = y + int(h * 0.25)
            left_armpit = self._find_contour_point_at_height(contour, armpit_y, 'left')
            right_armpit = self._find_contour_point_at_height(contour, armpit_y, 'right')
            
            keypoints['left_armpit'] = (left_armpit[0], left_armpit[1])
            keypoints['right_armpit'] = (right_armpit[0], right_armpit[1])
            
            # 소매 끝 (좌우 극단점)
            leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
            rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
            
            keypoints['left_sleeve'] = leftmost
            keypoints['right_sleeve'] = rightmost
            
            # 목 라인 (최상단 중앙)
            top_points = contour[contour[:, :, 1] < y + h * 0.1]
            if len(top_points) > 0:
                neck_center = np.mean(top_points, axis=0)
                keypoints['collar'] = (int(neck_center[0][0]), int(neck_center[0][1]))
            else:
                keypoints['collar'] = (x + w // 2, y)
            
            # 하단 (hem)
            bottom_y = y + int(h * 0.9)
            hem_points = contour[np.abs(contour[:, :, 1] - bottom_y) < h * 0.1]
            if len(hem_points) > 0:
                hem_center = np.mean(hem_points, axis=0)
                keypoints['hem'] = (int(hem_center[0][0]), int(hem_center[0][1]))
            else:
                keypoints['hem'] = (x + w // 2, y + h)
            
        except Exception as e:
            logger.warning(f"상의 키포인트 추출 실패: {e}")
        
        return keypoints
    
    def _extract_pants_keypoints_detailed(
        self, 
        contour: np.ndarray, 
        x: int, y: int, w: int, h: int
    ) -> Dict[str, Tuple[float, float]]:
        """바지 키포인트 상세 추출"""
        keypoints = {}
        
        try:
            # 허리 라인 (상단 10%)
            waist_y = y + int(h * 0.1)
            left_waist = self._find_contour_point_at_height(contour, waist_y, 'left')
            right_waist = self._find_contour_point_at_height(contour, waist_y, 'right')
            
            keypoints['waist_left'] = (left_waist[0], left_waist[1])
            keypoints['waist_right'] = (right_waist[0], right_waist[1])
            
            # 허벅지 (상단 40%)
            thigh_y = y + int(h * 0.4)
            left_thigh = self._find_contour_point_at_height(contour, thigh_y, 'left')
            right_thigh = self._find_contour_point_at_height(contour, thigh_y, 'right')
            
            keypoints['left_thigh'] = (left_thigh[0], left_thigh[1])
            keypoints['right_thigh'] = (right_thigh[0], right_thigh[1])
            
            # 무릎 (중간 60%)
            knee_y = y + int(h * 0.6)
            left_knee = self._find_contour_point_at_height(contour, knee_y, 'left')
            right_knee = self._find_contour_point_at_height(contour, knee_y, 'right')
            
            keypoints['left_leg'] = (left_knee[0], left_knee[1])
            keypoints['right_leg'] = (right_knee[0], right_knee[1])
            
            # 발목 (하단 90%)
            ankle_y = y + int(h * 0.9)
            left_ankle = self._find_contour_point_at_height(contour, ankle_y, 'left')
            right_ankle = self._find_contour_point_at_height(contour, ankle_y, 'right')
            
            keypoints['left_ankle'] = (left_ankle[0], left_ankle[1])
            keypoints['right_ankle'] = (right_ankle[0], right_ankle[1])
            
        except Exception as e:
            logger.warning(f"바지 키포인트 추출 실패: {e}")
        
        return keypoints
    
    def _extract_dress_keypoints_detailed(
        self, 
        contour: np.ndarray, 
        x: int, y: int, w: int, h: int
    ) -> Dict[str, Tuple[float, float]]:
        """원피스 키포인트 상세 추출"""
        keypoints = {}
        
        try:
            # 상의 부분 (상단 50%)
            shirt_keypoints = self._extract_shirt_keypoints_detailed(contour, x, y, w, int(h * 0.5))
            keypoints.update(shirt_keypoints)
            
            # 허리 라인 (중간 40%)
            waist_y = y + int(h * 0.4)
            left_waist = self._find_contour_point_at_height(contour, waist_y, 'left')
            right_waist = self._find_contour_point_at_height(contour, waist_y, 'right')
            
            keypoints['waist_left'] = (left_waist[0], left_waist[1])
            keypoints['waist_right'] = (right_waist[0], right_waist[1])
            
            # 엉덩이 라인 (중간 60%)
            hip_y = y + int(h * 0.6)
            left_hip = self._find_contour_point_at_height(contour, hip_y, 'left')
            right_hip = self._find_contour_point_at_height(contour, hip_y, 'right')
            
            keypoints['left_hip'] = (left_hip[0], left_hip[1])
            keypoints['right_hip'] = (right_hip[0], right_hip[1])
            
            # 밑단 (하단 95%)
            hem_y = y + int(h * 0.95)
            left_hem = self._find_contour_point_at_height(contour, hem_y, 'left')
            right_hem = self._find_contour_point_at_height(contour, hem_y, 'right')
            
            keypoints['hem_left'] = (left_hem[0], left_hem[1])
            keypoints['hem_right'] = (right_hem[0], right_hem[1])
            
        except Exception as e:
            logger.warning(f"원피스 키포인트 추출 실패: {e}")
        
        return keypoints
    
    def _extract_skirt_keypoints_detailed(
        self, 
        contour: np.ndarray, 
        x: int, y: int, w: int, h: int
    ) -> Dict[str, Tuple[float, float]]:
        """스커트 키포인트 상세 추출"""
        keypoints = {}
        
        try:
            # 허리 라인 (상단 10%)
            waist_y = y + int(h * 0.1)
            left_waist = self._find_contour_point_at_height(contour, waist_y, 'left')
            right_waist = self._find_contour_point_at_height(contour, waist_y, 'right')
            
            keypoints['waist_left'] = (left_waist[0], left_waist[1])
            keypoints['waist_right'] = (right_waist[0], right_waist[1])
            
            # 밑단 (하단 95%)
            hem_y = y + int(h * 0.95)
            left_hem = self._find_contour_point_at_height(contour, hem_y, 'left')
            right_hem = self._find_contour_point_at_height(contour, hem_y, 'right')
            
            keypoints['hem_left'] = (left_hem[0], left_hem[1])
            keypoints['hem_right'] = (right_hem[0], right_hem[1])
            
        except Exception as e:
            logger.warning(f"스커트 키포인트 추출 실패: {e}")
        
        return keypoints
    
    def _find_contour_point_at_height(
        self, 
        contour: np.ndarray, 
        y: int, 
        side: str
    ) -> List[int]:
        """특정 높이에서 윤곽선의 좌/우 끝점 찾기"""
        tolerance = 15
        points_at_height = []
        
        for point in contour:
            if abs(point[0][1] - y) < tolerance:
                points_at_height.append(point[0])
        
        if not points_at_height:
            # 가장 가까운 포인트 찾기
            distances = [abs(point[0][1] - y) for point in contour]
            nearest_idx = np.argmin(distances)
            return contour[nearest_idx][0].tolist()
        
        # 좌측 또는 우측 극단점 선택
        if side == 'left':
            return min(points_at_height, key=lambda p: p[0]).tolist()
        else:
            return max(points_at_height, key=lambda p: p[0]).tolist()
    
    def _generate_default_keypoints(self, w: int, h: int, clothing_type: str) -> Dict[str, Tuple[float, float]]:
        """기본 키포인트 생성"""
        keypoints = {}
        
        if clothing_type == 'shirt':
            keypoints = {
                'left_shoulder': (w * 0.2, h * 0.15),
                'right_shoulder': (w * 0.8, h * 0.15),
                'left_sleeve': (w * 0.05, h * 0.4),
                'right_sleeve': (w * 0.95, h * 0.4),
                'left_armpit': (w * 0.25, h * 0.25),
                'right_armpit': (w * 0.75, h * 0.25),
                'collar': (w * 0.5, h * 0.05),
                'hem': (w * 0.5, h * 0.9)
            }
        elif clothing_type == 'pants':
            keypoints = {
                'waist_left': (w * 0.3, h * 0.1),
                'waist_right': (w * 0.7, h * 0.1),
                'left_thigh': (w * 0.35, h * 0.4),
                'right_thigh': (w * 0.65, h * 0.4),
                'left_leg': (w * 0.35, h * 0.6),
                'right_leg': (w * 0.65, h * 0.6),
                'left_ankle': (w * 0.35, h * 0.9),
                'right_ankle': (w * 0.65, h * 0.9)
            }
        elif clothing_type == 'dress':
            keypoints = {
                'left_shoulder': (w * 0.2, h * 0.1),
                'right_shoulder': (w * 0.8, h * 0.1),
                'waist_left': (w * 0.25, h * 0.4),
                'waist_right': (w * 0.75, h * 0.4),
                'left_hip': (w * 0.3, h * 0.6),
                'right_hip': (w * 0.7, h * 0.6),
                'hem_left': (w * 0.3, h * 0.95),
                'hem_right': (w * 0.7, h * 0.95)
            }
        elif clothing_type == 'skirt':
            keypoints = {
                'waist_left': (w * 0.25, h * 0.1),
                'waist_right': (w * 0.75, h * 0.1),
                'hem_left': (w * 0.2, h * 0.95),
                'hem_right': (w * 0.8, h * 0.95)
            }
        
        return keypoints
    
    def _match_keypoints_advanced(
        self,
        body_keypoints: Dict[str, Tuple[float, float]],
        clothing_keypoints: Dict[str, Tuple[float, float]]
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float], str]]:
        """고급 키포인트 매칭 (직접 매칭 + Hungarian 알고리즘)"""
        
        matched_pairs = []
        
        try:
            # 1. 동일한 이름의 키포인트 직접 매칭
            common_names = set(body_keypoints.keys()) & set(clothing_keypoints.keys())
            used_body_names = set()
            used_clothing_names = set()
            
            for name in common_names:
                body_point = body_keypoints[name]
                clothing_point = clothing_keypoints[name]
                
                # 거리 검사
                distance = math.sqrt((body_point[0] - clothing_point[0])**2 + (body_point[1] - clothing_point[1])**2)
                
                # 매칭 허용 (거리 기반)
                if distance < self.matching_config['matching_threshold'] * 2:  # 관대한 임계값
                    matched_pairs.append((body_point, clothing_point, name))
                    used_body_names.add(name)
                    used_clothing_names.add(name)
            
            # 2. 남은 키포인트들에 대해 Hungarian 알고리즘 적용
            available_body = {k: v for k, v in body_keypoints.items() if k not in used_body_names}
            available_clothing = {k: v for k, v in clothing_keypoints.items() if k not in used_clothing_names}
            
            if available_body and available_clothing:
                additional_pairs = self._find_additional_matches_hungarian(
                    available_body, available_clothing
                )
                matched_pairs.extend(additional_pairs)
            
            # 3. 최소 매칭 개수 확보
            if len(matched_pairs) < self.matching_config['min_matching_points']:
                extra_pairs = self._generate_additional_correspondences(
                    body_keypoints, clothing_keypoints, matched_pairs
                )
                matched_pairs.extend(extra_pairs)
            
        except Exception as e:
            logger.warning(f"키포인트 매칭 실패: {e}")
        
        logger.info(f"매칭된 키포인트 쌍: {len(matched_pairs)}개")
        return matched_pairs
    
    def _find_additional_matches_hungarian(
        self,
        available_body: Dict[str, Tuple[float, float]],
        available_clothing: Dict[str, Tuple[float, float]]
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float], str]]:
        """Hungarian 알고리즘을 사용한 추가 매칭"""
        
        additional_pairs = []
        
        try:
            if not available_body or not available_clothing:
                return additional_pairs
            
            # 거리 기반 매칭
            body_points = list(available_body.values())
            clothing_points = list(available_clothing.values())
            body_names = list(available_body.keys())
            clothing_names = list(available_clothing.keys())
            
            # 거리 행렬 계산
            distances = cdist(body_points, clothing_points)
            
            # Hungarian 알고리즘으로 최적 할당
            row_indices, col_indices = linear_sum_assignment(distances)
            
            for i, j in zip(row_indices, col_indices):
                if distances[i, j] < self.matching_config['matching_threshold'] * 2:
                    body_point = body_points[i]
                    clothing_point = clothing_points[j]
                    match_name = f"{body_names[i]}_to_{clothing_names[j]}"
                    additional_pairs.append((body_point, clothing_point, match_name))
            
        except Exception as e:
            logger.warning(f"Hungarian 매칭 실패: {e}")
        
        return additional_pairs
    
    def _generate_additional_correspondences(
        self,
        body_keypoints: Dict[str, Tuple[float, float]],
        clothing_keypoints: Dict[str, Tuple[float, float]],
        existing_pairs: List
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float], str]]:
        """추가 대응점 생성"""
        
        additional_pairs = []
        
        try:
            # 의류 바운딩 박스의 모서리점들 추가
            if clothing_keypoints:
                cloth_xs = [kp[0] for kp in clothing_keypoints.values()]
                cloth_ys = [kp[1] for kp in clothing_keypoints.values()]
                
                cloth_min_x, cloth_max_x = min(cloth_xs), max(cloth_xs)
                cloth_min_y, cloth_max_y = min(cloth_ys), max(cloth_ys)
                
                # 의류 중심점
                cloth_center = (
                    (cloth_min_x + cloth_max_x) / 2,
                    (cloth_min_y + cloth_max_y) / 2
                )
                
                # 신체 중심점 계산
                if body_keypoints:
                    body_xs = [kp[0] for kp in body_keypoints.values()]
                    body_ys = [kp[1] for kp in body_keypoints.values()]
                    
                    body_center = (
                        sum(body_xs) / len(body_xs),
                        sum(body_ys) / len(body_ys)
                    )
                    
                    additional_pairs.append((body_center, cloth_center, "center_correspondence"))
            
        except Exception as e:
            logger.warning(f"추가 대응점 생성 실패: {e}")
        
        return additional_pairs
    
    def _calculate_tps_transform(
        self, 
        matched_pairs: List[Tuple[Tuple[float, float], Tuple[float, float], str]]
    ) -> Optional[np.ndarray]:
        """TPS 변환 매트릭스 계산"""
        if len(matched_pairs) < self.matching_config['min_matching_points']:
            logger.warning(f"매칭 포인트 부족: {len(matched_pairs)} < {self.matching_config['min_matching_points']}")
            return None
        
        try:
            # 소스 포인트 (의류) 및 타겟 포인트 (신체) 추출
            source_points = np.array([pair[1] for pair in matched_pairs], dtype=np.float32)  # 의류
            target_points = np.array([pair[0] for pair in matched_pairs], dtype=np.float32)  # 신체
            
            # TPS 솔버로 변환 계산
            tps_matrix = self.tps_solver.solve(source_points, target_points)
            
            return tps_matrix
            
        except Exception as e:
            logger.error(f"TPS 변환 계산 실패: {e}")
            return None
    
    def _apply_tps_transform(
        self,
        cloth_img: np.ndarray,
        cloth_mask: np.ndarray,
        matched_pairs: List[Tuple[Tuple[float, float], Tuple[float, float], str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """TPS 변환 적용"""
        
        if len(matched_pairs) < 3:
            logger.warning("매칭 포인트가 부족합니다. 원본을 반환합니다.")
            return cloth_img, cloth_mask
        
        try:
            # 소스와 타겟 포인트 분리
            source_points = np.array([pair[1] for pair in matched_pairs], dtype=np.float32)
            target_points = np.array([pair[0] for pair in matched_pairs], dtype=np.float32)
            
            # TPS 변환기에 학습
            self.tps_transformer.fit(source_points, target_points)
            
            # 이미지 워핑
            warped_cloth = self.tps_transformer.transform_image(cloth_img)
            warped_mask = self.tps_transformer.transform_image(cloth_mask)
            
            # 마스크 이진화
            if len(warped_mask.shape) == 3:
                warped_mask = cv2.cvtColor(warped_mask, cv2.COLOR_BGR2GRAY)
            warped_mask = (warped_mask > 128).astype(np.uint8) * 255
            
            return warped_cloth, warped_mask
            
        except Exception as e:
            logger.error(f"TPS 변환 적용 실패: {e}")
            return cloth_img, cloth_mask
    
    def _apply_mesh_refinement(
        self,
        warped_cloth: np.ndarray,
        warped_mask: np.ndarray,
        matched_pairs: List[Tuple[Tuple[float, float], Tuple[float, float], str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """메쉬 기반 세밀 조정"""
        
        try:
            # 메쉬 생성 및 세밀 조정
            refined_cloth = self.mesh_warper.refine_warping(
                warped_cloth, warped_mask, matched_pairs
            )
            
            # 마스크도 동일하게 정제
            refined_mask = self.mesh_warper.refine_warping(
                warped_mask, warped_mask, matched_pairs
            )
            
            return refined_cloth, refined_mask
            
        except Exception as e:
            logger.warning(f"메쉬 정제 실패: {e}")
            return warped_cloth, warped_mask
    
    def _evaluate_matching_quality_comprehensive(
        self,
        original_cloth: np.ndarray,
        warped_cloth: np.ndarray,
        matched_pairs: List,
        body_keypoints: Dict,
        clothing_keypoints: Dict
    ) -> Dict[str, float]:
        """종합적인 매칭 품질 평가"""
        
        metrics = {}
        
        try:
            # 1. 키포인트 매칭 정확도
            num_matched = len(matched_pairs)
            total_possible = min(len(body_keypoints), len(clothing_keypoints))
            
            metrics['keypoint_count'] = num_matched
            metrics['matching_ratio'] = num_matched / max(1, total_possible)
            metrics['keypoint_density'] = num_matched / max(original_cloth.shape[:2])
            
            # 2. 평균 매칭 거리
            if matched_pairs:
                distances = []
                for body_point, clothing_point, _ in matched_pairs:
                    dist = math.sqrt((body_point[0] - clothing_point[0])**2 + (body_point[1] - clothing_point[1])**2)
                    distances.append(dist)
                
                metrics['average_distance'] = np.mean(distances)
                metrics['max_distance'] = np.max(distances)
                metrics['distance_std'] = np.std(distances)
                
                # 거리 기반 품질 점수
                normalized_distance = min(1.0, metrics['average_distance'] / self.matching_config['matching_threshold'])
                metrics['distance_score'] = 1.0 - normalized_distance
            else:
                metrics['average_distance'] = float('inf')
                metrics['distance_score'] = 0.0
            
            # 3. 변형 일관성 (인접 키포인트 간 거리 비율)
            if len(matched_pairs) >= 2:
                source_distances = []
                target_distances = []
                
                for i in range(len(matched_pairs)):
                    for j in range(i+1, len(matched_pairs)):
                        src_dist = np.linalg.norm(np.array(matched_pairs[i][1]) - np.array(matched_pairs[j][1]))
                        tgt_dist = np.linalg.norm(np.array(matched_pairs[i][0]) - np.array(matched_pairs[j][0]))
                        
                        if src_dist > 0:
                            source_distances.append(src_dist)
                            target_distances.append(tgt_dist)
                
                if source_distances:
                    distance_ratios = np.array(target_distances) / np.array(source_distances)
                    metrics['deformation_consistency'] = 1.0 - min(1.0, np.std(distance_ratios))
                else:
                    metrics['deformation_consistency'] = 0.0
            else:
                metrics['deformation_consistency'] = 0.0
            
            # 4. 이미지 품질 (히스토그램 유사도)
            metrics['transform_quality'] = self._calculate_image_similarity(original_cloth, warped_cloth)
            
            # 5. 전체 품질 점수 (가중평균)
            overall_score = (
                metrics['matching_ratio'] * 0.3 +
                metrics.get('distance_score', 0) * 0.3 +
                metrics['deformation_consistency'] * 0.2 +
                metrics['transform_quality'] * 0.2
            )
            metrics['overall_score'] = min(1.0, max(0.0, overall_score))
            
            # 6. 신뢰도 레벨
            if metrics['overall_score'] > 0.8:
                metrics['confidence_level'] = 'high'
            elif metrics['overall_score'] > 0.6:
                metrics['confidence_level'] = 'medium'
            else:
                metrics['confidence_level'] = 'low'
            
        except Exception as e:
            logger.warning(f"매칭 품질 평가 실패: {e}")
            metrics = {'overall_score': 0.5, 'confidence_level': 'medium'}
        
        return metrics
    
    def _calculate_image_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """이미지 유사도 계산"""
        try:
            # 크기 맞추기
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # 그레이스케일 변환
            if len(img1.shape) == 3:
                img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            else:
                img1_gray = img1
                
            if len(img2.shape) == 3:
                img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else:
                img2_gray = img2
            
            # 히스토그램 비교
            hist1 = cv2.calcHist([img1_gray], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2_gray], [0], None, [256], [0, 256])
            
            # 코사인 유사도
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            return max(0.0, correlation)
            
        except Exception:
            return 0.5  # 기본값
    
    def _calculate_deformation_regions(
        self, 
        matched_pairs: List, 
        clothing_type: str
    ) -> Dict[str, Any]:
        """변형 영역 계산 (향상된 버전)"""
        regions = {}
        
        try:
            if not matched_pairs:
                return regions
            
            # 변형 벡터 계산
            deformation_vectors = []
            for body_point, clothing_point, name in matched_pairs:
                vector = (body_point[0] - clothing_point[0], body_point[1] - clothing_point[1])
                deformation_vectors.append({
                    'name': name,
                    'vector': vector,
                    'magnitude': math.sqrt(vector[0]**2 + vector[1]**2),
                    'angle': math.atan2(vector[1], vector[0])
                })
            
            regions['deformation_vectors'] = deformation_vectors
            
            # 통계 정보
            magnitudes = [v['magnitude'] for v in deformation_vectors]
            regions['average_deformation'] = np.mean(magnitudes)
            regions['max_deformation'] = np.max(magnitudes)
            regions['min_deformation'] = np.min(magnitudes)
            regions['deformation_variance'] = np.var(magnitudes)
            
            # 주요 변형 방향
            angles = [v['angle'] for v in deformation_vectors]
            regions['primary_direction'] = np.mean(angles)
            regions['direction_consistency'] = 1.0 - (np.std(angles) / np.pi)
            
            # 의류별 특화 분석
            if clothing_type == 'shirt':
                regions['sleeve_analysis'] = self._analyze_sleeve_deformation(deformation_vectors)
                regions['torso_analysis'] = self._analyze_torso_deformation(deformation_vectors)
            elif clothing_type == 'pants':
                regions['waist_analysis'] = self._analyze_waist_deformation(deformation_vectors)
                regions['leg_analysis'] = self._analyze_leg_deformation(deformation_vectors)
            elif clothing_type == 'dress':
                regions['upper_analysis'] = self._analyze_torso_deformation(deformation_vectors)
                regions['lower_analysis'] = self._analyze_leg_deformation(deformation_vectors)
            
        except Exception as e:
            logger.warning(f"변형 영역 계산 실패: {e}")
        
        return regions
    
    def _analyze_sleeve_deformation(self, vectors: List[Dict]) -> Dict[str, float]:
        """소매 변형 분석"""
        sleeve_vectors = [v for v in vectors if 'sleeve' in v['name'].lower() or 'armpit' in v['name'].lower()]
        if not sleeve_vectors:
            return {'magnitude': 0.0, 'asymmetry': 0.0, 'consistency': 1.0}
        
        magnitudes = [v['magnitude'] for v in sleeve_vectors]
        angles = [v['angle'] for v in sleeve_vectors]
        
        return {
            'magnitude': np.mean(magnitudes),
            'asymmetry': np.std(magnitudes) if len(magnitudes) > 1 else 0.0,
            'consistency': 1.0 - (np.std(angles) / np.pi) if len(angles) > 1 else 1.0
        }
    
    def _analyze_torso_deformation(self, vectors: List[Dict]) -> Dict[str, float]:
        """몸통 변형 분석"""
        torso_keywords = ['shoulder', 'collar', 'hem', 'chest']
        torso_vectors = [v for v in vectors if any(keyword in v['name'].lower() for keyword in torso_keywords)]
        
        if not torso_vectors:
            return {'magnitude': 0.0, 'uniformity': 1.0, 'stability': 1.0}
        
        magnitudes = [v['magnitude'] for v in torso_vectors]
        angles = [v['angle'] for v in torso_vectors]
        
        return {
            'magnitude': np.mean(magnitudes),
            'uniformity': 1.0 - (np.std(magnitudes) / np.mean(magnitudes)) if np.mean(magnitudes) > 0 else 1.0,
            'stability': 1.0 - (np.std(angles) / np.pi) if len(angles) > 1 else 1.0
        }
    
    def _analyze_waist_deformation(self, vectors: List[Dict]) -> Dict[str, float]:
        """허리 변형 분석"""
        waist_vectors = [v for v in vectors if 'waist' in v['name'].lower()]
        if not waist_vectors:
            return {'magnitude': 0.0, 'symmetry': 1.0, 'fit_quality': 1.0}
        
        magnitudes = [v['magnitude'] for v in waist_vectors]
        
        return {
            'magnitude': np.mean(magnitudes),
            'symmetry': 1.0 - (np.std(magnitudes) / np.mean(magnitudes)) if np.mean(magnitudes) > 0 else 1.0,
            'fit_quality': 1.0 / (1.0 + np.mean(magnitudes) / 50.0)  # 변형이 적을수록 좋은 핏
        }
    
    def _analyze_leg_deformation(self, vectors: List[Dict]) -> Dict[str, float]:
        """다리 변형 분석"""
        leg_keywords = ['leg', 'ankle', 'thigh', 'knee']
        leg_vectors = [v for v in vectors if any(keyword in v['name'].lower() for keyword in leg_keywords)]
        
        if not leg_vectors:
            return {'magnitude': 0.0, 'proportion': 1.0, 'length_consistency': 1.0}
        
        magnitudes = [v['magnitude'] for v in leg_vectors]
        
        return {
            'magnitude': np.mean(magnitudes),
            'proportion': min(magnitudes) / max(magnitudes) if max(magnitudes) > 0 else 1.0,
            'length_consistency': 1.0 - (np.std(magnitudes) / np.mean(magnitudes)) if np.mean(magnitudes) > 0 else 1.0
        }
    
    async def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "step_name": "GeometricMatching",
            "version": "unified_v1.0",
            "transform_method": "TPS + Mesh Hybrid",
            "device": self.device,
            "use_mps": self.use_mps,
            "initialized": self.is_initialized,
            "tps_config": self.tps_config,
            "matching_config": self.matching_config,
            "supported_clothing_types": list(self.CLOTHING_KEYPOINTS.keys()),
            "pose_mapping": self.POSE_TO_CLOTHING,
            "min_keypoints": self.matching_config["min_matching_points"],
            "max_keypoints": self.matching_config["max_keypoints"],
            "features": [
                "Hungarian algorithm matching",
                "Contour-based keypoint extraction", 
                "TPS + Mesh hybrid warping",
                "Comprehensive quality evaluation",
                "M3 Max optimization"
            ]
        }
    
    async def cleanup(self):
        """리소스 정리"""
        if self.tps_solver:
            del self.tps_solver
            self.tps_solver = None
            
        if self.tps_transformer:
            del self.tps_transformer
            self.tps_transformer = None
        
        if self.mesh_warper:
            del self.mesh_warper
            self.mesh_warper = None
        
        self.is_initialized = False
        logger.info("🧹 기하학적 매칭 스텝 리소스 정리 완료")


class TPSSolver:
    """Thin Plate Spline 변환 솔버 (수학적으로 정확한 버전)"""
    
    def __init__(self, device: str, reg_factor: float = 0.1):
        self.device = device
        self.reg_factor = reg_factor
    
    def solve(self, source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
        """TPS 변환 매트릭스 계산"""
        try:
            n_points = source_points.shape[0]
            
            # 거리 행렬 계산
            distances = cdist(source_points, source_points)
            
            # TPS 기저 함수: r^2 * log(r)
            with np.errstate(divide='ignore', invalid='ignore'):
                K = np.where(distances == 0, 0, distances**2 * np.log(distances))
            
            # 시스템 행렬 구성
            P = np.hstack([np.ones((n_points, 1)), source_points])
            
            # 상단 블록
            top_block = np.hstack([K + self.reg_factor * np.eye(n_points), P])
            
            # 하단 블록
            bottom_block = np.hstack([P.T, np.zeros((3, 3))])
            
            # 전체 시스템 행렬
            A = np.vstack([top_block, bottom_block])
            
            # 우변 벡터
            b_x = np.hstack([target_points[:, 0], np.zeros(3)])
            b_y = np.hstack([target_points[:, 1], np.zeros(3)])
            
            # 선형 시스템 해결
            weights_x = np.linalg.solve(A, b_x)
            weights_y = np.linalg.solve(A, b_y)
            
            # 변환 매트릭스 반환
            tps_matrix = np.column_stack([weights_x, weights_y])
            
            return tps_matrix
            
        except Exception as e:
            logger.error(f"TPS 해결 실패: {e}")
            # 항등 변환 반환
            return np.eye(source_points.shape[0] + 3, 2)
    
    def transform(self, points: np.ndarray, tps_matrix: np.ndarray, source_points: np.ndarray) -> np.ndarray:
        """TPS 변환 적용"""
        try:
            n_source = source_points.shape[0]
            n_points = points.shape[0]
            
            # 변환할 점들과 소스 점들 간의 거리
            distances = cdist(points, source_points)
            
            # TPS 기저 함수 계산
            with np.errstate(divide='ignore', invalid='ignore'):
                U = np.where(distances == 0, 0, distances**2 * np.log(distances))
            
            # 어파인 부분
            affine_part = np.hstack([np.ones((n_points, 1)), points])
            
            # 전체 기저 함수
            basis = np.hstack([U, affine_part])
            
            # 변환 적용
            transformed = basis @ tps_matrix
            
            return transformed
            
        except Exception as e:
            logger.error(f"TPS 변환 적용 실패: {e}")
            return points  # 원본 점들 반환


class ThinPlateSplineTransform:
    """Thin Plate Spline 변환 구현 (RBF 기반)"""
    
    def __init__(self, regularization: float = 0.001, smoothing: float = 0.01):
        self.regularization = regularization
        self.smoothing = smoothing
        self.rbf_x = None
        self.rbf_y = None
        self.control_points = None
        self.target_points = None
    
    def fit(self, control_points: np.ndarray, target_points: np.ndarray):
        """TPS 변환 학습"""
        self.control_points = control_points.astype(np.float32)
        self.target_points = target_points.astype(np.float32)
        
        # RBF 보간기 생성
        self.rbf_x = RBFInterpolator(
            self.control_points,
            self.target_points[:, 0],
            kernel='thin_plate_spline',
            smoothing=self.smoothing
        )
        
        self.rbf_y = RBFInterpolator(
            self.control_points,
            self.target_points[:, 1],
            kernel='thin_plate_spline',
            smoothing=self.smoothing
        )
    
    def transform_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 변환"""
        h, w = image.shape[:2]
        
        # 그리드 생성
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        grid_points = np.stack([x.ravel(), y.ravel()], axis=-1).astype(np.float32)
        
        # 변환 적용
        transformed_x = self.rbf_x(grid_points).reshape(h, w)
        transformed_y = self.rbf_y(grid_points).reshape(h, w)
        
        # 리매핑
        map_x = transformed_x.astype(np.float32)
        map_y = transformed_y.astype(np.float32)
        
        # 이미지 워핑
        warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        return warped


class MeshBasedWarping:
    """메쉬 기반 워핑 (세밀 조정용)"""
    
    def __init__(self, mesh_size: int = 15):
        self.mesh_size = mesh_size
    
    def refine_warping(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        matched_pairs: List[Tuple[Tuple[float, float], Tuple[float, float], str]]
    ) -> np.ndarray:
        """메쉬 기반 세밀 조정"""
        
        try:
            # 1. 기본 가우시안 스무딩
            refined = cv2.GaussianBlur(image, (3, 3), 0.5)
            
            # 2. 엣지 보존
            if len(image.shape) == 3:
                edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 50, 150)
            else:
                edges = cv2.Canny(image, 50, 150)
            
            # 엣지 영역은 원본 유지
            edge_mask = edges > 0
            if len(image.shape) == 3:
                edge_mask = np.stack([edge_mask] * 3, axis=2)
            
            refined = np.where(edge_mask, image, refined)
            
            # 3. 키포인트 주변 국소 조정
            if matched_pairs:
                refined = self._apply_local_adjustments(refined, matched_pairs)
            
            # 4. 마스크 기반 블렌딩
            if mask is not None and mask.max() > 0:
                mask_normalized = mask.astype(np.float32) / 255.0
                if len(image.shape) == 3 and len(mask_normalized.shape) == 2:
                    mask_normalized = np.stack([mask_normalized] * 3, axis=2)
                
                # 마스크 영역만 적용
                refined = refined * mask_normalized + image * (1 - mask_normalized)
            
            return refined.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"메쉬 기반 워핑 실패: {e}")
            return image
    
    def _apply_local_adjustments(
        self, 
        image: np.ndarray, 
        matched_pairs: List[Tuple[Tuple[float, float], Tuple[float, float], str]]
    ) -> np.ndarray:
        """키포인트 주변 국소 조정"""
        
        try:
            h, w = image.shape[:2]
            adjusted = image.copy()
            
            for body_point, cloth_point, name in matched_pairs:
                # 변형 벡터
                dx = body_point[0] - cloth_point[0]
                dy = body_point[1] - cloth_point[1]
                
                # 영향 반경 (적응적)
                influence_radius = min(50, max(20, math.sqrt(dx*dx + dy*dy)))
                
                # 중심점 주변 영역
                center_x, center_y = int(cloth_point[0]), int(cloth_point[1])
                
                # 영역 범위 계산
                x_min = max(0, center_x - int(influence_radius))
                x_max = min(w, center_x + int(influence_radius))
                y_min = max(0, center_y - int(influence_radius))
                y_max = min(h, center_y + int(influence_radius))
                
                # 국소 영역에 부드러운 변형 적용
                for y in range(y_min, y_max):
                    for x in range(x_min, x_max):
                        dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                        
                        if dist < influence_radius:
                            # 가우시안 가중치
                            weight = math.exp(-(dist**2) / (2 * (influence_radius/3)**2))
                            
                            # 새로운 위치 계산
                            new_x = x + dx * weight
                            new_y = y + dy * weight
                            
                            # 경계 확인 및 값 보간
                            if 0 <= new_x < w-1 and 0 <= new_y < h-1:
                                # 바이리니어 보간
                                x1, y1 = int(new_x), int(new_y)
                                x2, y2 = x1 + 1, y1 + 1
                                
                                dx_frac = new_x - x1
                                dy_frac = new_y - y1
                                
                                if len(image.shape) == 3:
                                    interpolated = (
                                        image[y1, x1] * (1-dx_frac) * (1-dy_frac) +
                                        image[y1, x2] * dx_frac * (1-dy_frac) +
                                        image[y2, x1] * (1-dx_frac) * dy_frac +
                                        image[y2, x2] * dx_frac * dy_frac
                                    )
                                else:
                                    interpolated = (
                                        image[y1, x1] * (1-dx_frac) * (1-dy_frac) +
                                        image[y1, x2] * dx_frac * (1-dy_frac) +
                                        image[y2, x1] * (1-dx_frac) * dy_frac +
                                        image[y2, x2] * dx_frac * dy_frac
                                    )
                                
                                # 가중 평균으로 블렌딩
                                adjusted[y, x] = (
                                    adjusted[y, x] * (1 - weight) + 
                                    interpolated * weight
                                ).astype(np.uint8)
            
            return adjusted
            
        except Exception as e:
            logger.warning(f"국소 조정 실패: {e}")
            return image


# 추가 유틸리티 함수들
def visualize_keypoints(
    image: np.ndarray, 
    keypoints: Dict[str, Tuple[float, float]], 
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 5,
    thickness: int = 2
) -> np.ndarray:
    """키포인트 시각화"""
    vis_image = image.copy()
    
    for name, (x, y) in keypoints.items():
        cv2.circle(vis_image, (int(x), int(y)), radius, color, thickness)
        cv2.putText(vis_image, name, (int(x)+10, int(y)-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return vis_image


def visualize_matches(
    image1: np.ndarray,
    image2: np.ndarray, 
    matched_pairs: List[Tuple[Tuple[float, float], Tuple[float, float], str]]
) -> np.ndarray:
    """매칭 결과 시각화"""
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    h = max(h1, h2)
    w = w1 + w2
    
    # 이미지 합치기
    vis_image = np.zeros((h, w, 3), dtype=np.uint8)
    vis_image[:h1, :w1] = image1 if len(image1.shape) == 3 else cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    vis_image[:h2, w1:w1+w2] = image2 if len(image2.shape) == 3 else cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    
    # 매칭 라인 그리기
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i, (body_point, cloth_point, name) in enumerate(matched_pairs):
        color = colors[i % len(colors)]
        
        # 포인트 그리기
        cv2.circle(vis_image, (int(body_point[0]), int(body_point[1])), 8, color, -1)
        cv2.circle(vis_image, (int(cloth_point[0] + w1), int(cloth_point[1])), 8, color, -1)
        
        # 연결 라인 그리기
        cv2.line(vis_image, 
                (int(body_point[0]), int(body_point[1])),
                (int(cloth_point[0] + w1), int(cloth_point[1])),
                color, 2)
        
        # 이름 표시
        cv2.putText(vis_image, f"{i+1}", 
                   (int(body_point[0])-20, int(body_point[1])-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return vis_image


def save_intermediate_results(
    cloth_img: np.ndarray,
    warped_cloth: np.ndarray,
    matched_pairs: List,
    output_dir: str,
    step_name: str = "geometric_matching"
):
    """중간 결과 저장"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 원본 의류 이미지 저장
        cv2.imwrite(os.path.join(output_dir, f"{step_name}_original_cloth.jpg"), cloth_img)
        
        # 변형된 의류 이미지 저장
        cv2.imwrite(os.path.join(output_dir, f"{step_name}_warped_cloth.jpg"), warped_cloth)
        
        # 키포인트 정보 저장
        keypoint_data = {
            "num_matches": len(matched_pairs),
            "matches": [
                {
                    "body_point": [float(pair[0][0]), float(pair[0][1])],
                    "cloth_point": [float(pair[1][0]), float(pair[1][1])], 
                    "name": pair[2]
                }
                for pair in matched_pairs
            ]
        }
        
        with open(os.path.join(output_dir, f"{step_name}_keypoints.json"), 'w') as f:
            json.dump(keypoint_data, f, indent=2)
        
        logger.info(f"중간 결과 저장 완료: {output_dir}")
        
    except Exception as e:
        logger.warning(f"중간 결과 저장 실패: {e}")


# 성능 최적화를 위한 M3 Max 전용 함수들
def optimize_for_m3_max():
    """M3 Max 칩 최적화 설정"""
    if torch.backends.mps.is_available():
        # MPS 메모리 관리 최적화
        torch.mps.empty_cache()
        
        # 수치 안정성 설정
        torch.backends.mps.enable_fusion = True
        
        logger.info("M3 Max MPS 최적화 설정 완료")
        return True
    
    return False


def batch_process_keypoints(
    keypoints_list: List[Dict[str, Tuple[float, float]]],
    batch_size: int = 8
) -> List[Dict[str, Tuple[float, float]]]:
    """키포인트 배치 처리 (메모리 최적화)"""
    processed = []
    
    for i in range(0, len(keypoints_list), batch_size):
        batch = keypoints_list[i:i+batch_size]
        
        # 배치 처리 로직
        batch_processed = []
        for kp_dict in batch:
            # 키포인트 정규화 및 필터링
            filtered_kp = {
                name: point for name, point in kp_dict.items()
                if 0 <= point[0] <= 2048 and 0 <= point[1] <= 2048  # 합리적인 이미지 크기 범위
            }
            batch_processed.append(filtered_kp)
        
        processed.extend(batch_processed)
        
        # 메모리 정리
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    return processed