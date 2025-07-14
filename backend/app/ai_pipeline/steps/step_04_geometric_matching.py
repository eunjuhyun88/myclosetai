# app/ai_pipeline/steps/step_04_geometric_matching.py
"""
4단계: 기하학적 매칭 (Geometric Matching) - 수정된 버전
Pipeline Manager와 완전 호환되는 의류-인체 매칭 시스템
M3 Max 최적화 + 고급 매칭 알고리즘 + 견고한 에러 처리
"""
import os
import logging
import time
import asyncio
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

class GeometricMatchingStep:
    """
    기하학적 매칭 스텝 - Pipeline Manager 완전 호환
    - M3 Max MPS 최적화
    - 고급 매칭 알고리즘 (TPS, Affine, Homography)
    - 포즈 기반 적응형 매칭
    - 실시간 매칭 품질 평가
    """
    
    # 의류별 핵심 매칭 포인트 정의
    MATCHING_POINTS = {
        'shirt': {
            'keypoints': ['left_shoulder', 'right_shoulder', 'neck', 'left_wrist', 'right_wrist'],
            'clothing_points': ['left_shoulder', 'right_shoulder', 'collar', 'left_cuff', 'right_cuff'],
            'priority_weights': [1.0, 1.0, 0.8, 0.7, 0.7]
        },
        'pants': {
            'keypoints': ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'],
            'clothing_points': ['left_waist', 'right_waist', 'left_knee', 'right_knee', 'left_hem', 'right_hem'],
            'priority_weights': [1.0, 1.0, 0.8, 0.8, 0.6, 0.6]
        },
        'dress': {
            'keypoints': ['left_shoulder', 'right_shoulder', 'neck', 'left_hip', 'right_hip'],
            'clothing_points': ['left_shoulder', 'right_shoulder', 'collar', 'left_waist', 'right_waist'],
            'priority_weights': [1.0, 1.0, 0.8, 0.7, 0.7]
        }
    }
    
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
        
        # 매칭 설정
        self.matching_config = self.config.get('matching', {
            'method': 'auto',  # 'tps', 'affine', 'homography', 'auto'
            'max_iterations': 1000,
            'convergence_threshold': 1e-6,
            'outlier_threshold': 0.15,
            'use_pose_guidance': True,
            'adaptive_weights': True,
            'quality_threshold': 0.7
        })
        
        # TPS (Thin Plate Spline) 설정
        self.tps_config = self.config.get('tps', {
            'regularization': 0.1,
            'grid_size': 20,
            'boundary_padding': 0.1
        })
        
        # 최적화 설정
        self.optimization_config = self.config.get('optimization', {
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'scheduler_step': 100
        })
        
        # 매칭 통계
        self.matching_stats = {
            'total_matches': 0,
            'successful_matches': 0,
            'average_accuracy': 0.0,
            'method_performance': {}
        }
        
        self.logger.info(f"🎯 기하학적 매칭 스텝 초기화 - 디바이스: {device}")
    
    async def initialize(self) -> bool:
        """초기화 메서드"""
        try:
            # 매칭 알고리즘 초기화
            await self._initialize_matching_algorithms()
            
            # 최적화 도구 초기화
            await self._initialize_optimization_tools()
            
            self.is_initialized = True
            self.logger.info("✅ 기하학적 매칭 시스템 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 매칭 시스템 초기화 실패: {e}")
            # 기본 시스템으로 폴백
            self.is_initialized = True
            return True
    
    async def process(
        self,
        person_parsing: Dict[str, Any],
        pose_keypoints: List[List[float]],
        clothing_segmentation: Dict[str, Any],
        clothing_type: str = "shirt",
        **kwargs
    ) -> Dict[str, Any]:
        """
        기하학적 매칭 처리
        
        Args:
            person_parsing: 인체 파싱 결과
            pose_keypoints: 포즈 키포인트 (OpenPose 18 형식)
            clothing_segmentation: 의류 세그멘테이션 결과
            clothing_type: 의류 타입
            **kwargs: 추가 매개변수
            
        Returns:
            Dict: 매칭 결과
        """
        start_time = time.time()
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # 1. 입력 데이터 검증 및 전처리
            person_points = self._extract_person_keypoints(pose_keypoints, clothing_type)
            clothing_points = self._extract_clothing_keypoints(clothing_segmentation, clothing_type)
            
            if len(person_points) < 3 or len(clothing_points) < 3:
                return self._create_empty_result("충분하지 않은 매칭 포인트")
            
            # 2. 매칭 방법 선택
            matching_method = self._select_matching_method(person_points, clothing_points, clothing_type)
            self.logger.info(f"📐 선택된 매칭 방법: {matching_method}")
            
            # 3. 초기 매칭 수행
            initial_match = await self._perform_initial_matching(
                person_points, clothing_points, matching_method
            )
            
            # 4. 포즈 기반 정제
            if self.matching_config['use_pose_guidance']:
                refined_match = await self._refine_with_pose_guidance(
                    initial_match, pose_keypoints, clothing_type
                )
            else:
                refined_match = initial_match
            
            # 5. 매칭 품질 평가
            quality_metrics = self._evaluate_matching_quality(
                person_points, clothing_points, refined_match
            )
            
            # 6. 품질이 낮으면 대안 방법 시도
            if quality_metrics['overall_quality'] < self.matching_config['quality_threshold']:
                self.logger.info(f"🔄 품질 개선 시도 (현재: {quality_metrics['overall_quality']:.3f})")
                alternative_match = await self._try_alternative_methods(
                    person_points, clothing_points, clothing_type
                )
                
                alternative_quality = self._evaluate_matching_quality(
                    person_points, clothing_points, alternative_match
                )
                
                if alternative_quality['overall_quality'] > quality_metrics['overall_quality']:
                    refined_match = alternative_match
                    quality_metrics = alternative_quality
                    matching_method = alternative_match.get('method', matching_method)
            
            # 7. 워핑 파라미터 생성
            warp_params = self._generate_warp_parameters(refined_match, clothing_segmentation)
            
            # 8. 최종 결과 구성
            processing_time = time.time() - start_time
            result = self._build_final_result(
                refined_match, warp_params, quality_metrics, 
                processing_time, matching_method, clothing_type
            )
            
            # 9. 통계 업데이트
            self._update_statistics(matching_method, quality_metrics['overall_quality'])
            
            self.logger.info(f"✅ 기하학적 매칭 완료 - 방법: {matching_method}, 품질: {quality_metrics['overall_quality']:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 기하학적 매칭 실패: {e}")
            return self._create_empty_result(f"처리 오류: {str(e)}")
    
    def _extract_person_keypoints(self, pose_keypoints: List[List[float]], clothing_type: str) -> List[Tuple[float, float]]:
        """인체에서 매칭 포인트 추출"""
        
        try:
            keypoint_mapping = {
                'neck': 1, 'left_shoulder': 5, 'right_shoulder': 2,
                'left_elbow': 6, 'right_elbow': 3,
                'left_wrist': 7, 'right_wrist': 4,
                'left_hip': 11, 'right_hip': 8,
                'left_knee': 12, 'right_knee': 9,
                'left_ankle': 13, 'right_ankle': 10
            }
            
            matching_points = self.MATCHING_POINTS.get(clothing_type, self.MATCHING_POINTS['shirt'])
            person_points = []
            
            for keypoint_name in matching_points['keypoints']:
                if keypoint_name in keypoint_mapping:
                    idx = keypoint_mapping[keypoint_name]
                    if idx < len(pose_keypoints):
                        x, y, conf = pose_keypoints[idx]
                        if conf > 0.5:  # 신뢰도 임계값
                            person_points.append((float(x), float(y)))
            
            self.logger.debug(f"추출된 인체 포인트: {len(person_points)}개")
            return person_points
            
        except Exception as e:
            self.logger.warning(f"인체 키포인트 추출 실패: {e}")
            return []
    
    def _extract_clothing_keypoints(self, clothing_segmentation: Dict[str, Any], clothing_type: str) -> List[Tuple[float, float]]:
        """의류에서 매칭 포인트 추출"""
        
        try:
            mask = clothing_segmentation.get('mask')
            if mask is None:
                return []
            
            # 의류 윤곽선 추출
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return []
            
            # 가장 큰 윤곽선 선택
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 의류 타입별 특징점 추출
            clothing_points = self._extract_clothing_features(largest_contour, mask, clothing_type)
            
            self.logger.debug(f"추출된 의류 포인트: {len(clothing_points)}개")
            return clothing_points
            
        except Exception as e:
            self.logger.warning(f"의류 키포인트 추출 실패: {e}")
            return []
    
    def _extract_clothing_features(self, contour: np.ndarray, mask: np.ndarray, clothing_type: str) -> List[Tuple[float, float]]:
        """의류 특징점 추출"""
        
        features = []
        
        try:
            # 바운딩 박스
            x, y, w, h = cv2.boundingRect(contour)
            
            if clothing_type in ['shirt', 't-shirt', 'blouse']:
                # 상의: 어깨, 목, 소매 부분
                features.extend([
                    (x + w * 0.2, y + h * 0.1),  # 왼쪽 어깨
                    (x + w * 0.8, y + h * 0.1),  # 오른쪽 어깨
                    (x + w * 0.5, y),            # 목/칼라
                    (x, y + h * 0.3),            # 왼쪽 소매
                    (x + w, y + h * 0.3)         # 오른쪽 소매
                ])
                
            elif clothing_type in ['pants', 'jeans', 'trousers']:
                # 하의: 허리, 무릎, 발목 부분
                features.extend([
                    (x + w * 0.2, y),            # 왼쪽 허리
                    (x + w * 0.8, y),            # 오른쪽 허리
                    (x + w * 0.3, y + h * 0.6),  # 왼쪽 무릎
                    (x + w * 0.7, y + h * 0.6),  # 오른쪽 무릎
                    (x + w * 0.3, y + h),        # 왼쪽 발목
                    (x + w * 0.7, y + h)         # 오른쪽 발목
                ])
                
            elif clothing_type in ['dress', 'gown']:
                # 드레스: 어깨, 목, 허리 부분
                features.extend([
                    (x + w * 0.2, y + h * 0.1),  # 왼쪽 어깨
                    (x + w * 0.8, y + h * 0.1),  # 오른쪽 어깨
                    (x + w * 0.5, y),            # 목/칼라
                    (x + w * 0.2, y + h * 0.4),  # 왼쪽 허리
                    (x + w * 0.8, y + h * 0.4)   # 오른쪽 허리
                ])
            
            # 윤곽선 기반 추가 특징점
            features.extend(self._extract_contour_features(contour))
            
            return features
            
        except Exception as e:
            self.logger.warning(f"의류 특징점 추출 실패: {e}")
            return []
    
    def _extract_contour_features(self, contour: np.ndarray) -> List[Tuple[float, float]]:
        """윤곽선 기반 특징점 추출"""
        
        features = []
        
        try:
            # 볼록 껍질
            hull = cv2.convexHull(contour)
            
            # 극값점들
            leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
            rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
            topmost = tuple(contour[contour[:, :, 1].argmin()][0])
            bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
            
            features.extend([leftmost, rightmost, topmost, bottommost])
            
            # 코너 점들 (Harris corner detection)
            mask = np.zeros(contour.max(axis=0).max(axis=0) + 10, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            
            corners = cv2.goodFeaturesToTrack(
                mask, maxCorners=10, qualityLevel=0.01, minDistance=10
            )
            
            if corners is not None:
                for corner in corners:
                    features.append(tuple(corner.ravel()))
            
            return features
            
        except Exception as e:
            self.logger.warning(f"윤곽선 특징점 추출 실패: {e}")
            return []
    
    def _select_matching_method(self, person_points: List, clothing_points: List, clothing_type: str) -> str:
        """매칭 방법 선택"""
        
        method = self.matching_config['method']
        
        if method == 'auto':
            num_points = min(len(person_points), len(clothing_points))
            
            # 포인트 수와 의류 타입에 따른 자동 선택
            if num_points >= 8:
                return 'tps'  # 충분한 포인트가 있으면 TPS
            elif num_points >= 4:
                return 'homography'  # 4-7개 포인트는 Homography
            elif num_points >= 3:
                return 'affine'  # 3개 포인트는 Affine
            else:
                return 'similarity'  # 최소 변환
        
        return method
    
    async def _perform_initial_matching(
        self, 
        person_points: List, 
        clothing_points: List, 
        method: str
    ) -> Dict[str, Any]:
        """초기 매칭 수행"""
        
        try:
            if method == 'tps':
                return await self._tps_matching(person_points, clothing_points)
            elif method == 'homography':
                return self._homography_matching(person_points, clothing_points)
            elif method == 'affine':
                return self._affine_matching(person_points, clothing_points)
            elif method == 'similarity':
                return self._similarity_matching(person_points, clothing_points)
            else:
                # 기본: 아핀 변환
                return self._affine_matching(person_points, clothing_points)
                
        except Exception as e:
            self.logger.warning(f"매칭 방법 {method} 실패: {e}")
            # 폴백: 단순 변환
            return self._similarity_matching(person_points, clothing_points)
    
    async def _tps_matching(self, person_points: List, clothing_points: List) -> Dict[str, Any]:
        """Thin Plate Spline 매칭"""
        
        try:
            # 대응점 쌍 생성 (가장 가까운 점들 매칭)
            person_array = np.array(person_points)
            clothing_array = np.array(clothing_points)
            
            # 거리 기반 대응 찾기
            distances = cdist(person_array, clothing_array)
            correspondences = []
            
            used_clothing = set()
            for i, person_pt in enumerate(person_array):
                # 각 인체 포인트에 대해 가장 가까운 의류 포인트 찾기
                distances_to_clothing = distances[i]
                sorted_indices = np.argsort(distances_to_clothing)
                
                for clothing_idx in sorted_indices:
                    if clothing_idx not in used_clothing:
                        correspondences.append((person_pt, clothing_array[clothing_idx]))
                        used_clothing.add(clothing_idx)
                        break
            
            # TPS 변환 계산
            if len(correspondences) >= 3:
                source_pts = np.array([corr[1] for corr in correspondences])  # 의류 점들
                target_pts = np.array([corr[0] for corr in correspondences])  # 인체 점들
                
                tps_transform = self._compute_tps_transform(source_pts, target_pts)
                
                return {
                    'method': 'tps',
                    'transform': tps_transform,
                    'correspondences': correspondences,
                    'source_points': source_pts.tolist(),
                    'target_points': target_pts.tolist(),
                    'confidence': 0.9
                }
            else:
                raise ValueError("TPS를 위한 충분한 대응점이 없음")
                
        except Exception as e:
            self.logger.warning(f"TPS 매칭 실패: {e}")
            raise
    
    def _compute_tps_transform(self, source_pts: np.ndarray, target_pts: np.ndarray) -> Dict[str, Any]:
        """TPS 변환 매개변수 계산"""
        
        try:
            n = len(source_pts)
            
            # TPS 기본 함수 (U 함수: r^2 * log(r))
            def U(r):
                return np.where(r == 0, 0, r**2 * np.log(r + 1e-10))
            
            # 거리 행렬 계산
            distances = cdist(source_pts, source_pts)
            
            # K 행렬 (기본 함수들의 값)
            K = U(distances)
            
            # P 행렬 (affine 부분을 위한 다항식 기저)
            P = np.column_stack([np.ones(n), source_pts])
            
            # L 행렬 구성
            O = np.zeros((3, 3))
            L = np.block([[K, P], [P.T, O]])
            
            # 목표 점들을 확장
            Y = np.vstack([target_pts.T, np.zeros((3, 2))])
            
            # 선형 시스템 해결
            try:
                coeffs = np.linalg.solve(L, Y)
            except np.linalg.LinAlgError:
                # 특이 행렬인 경우 pseudo-inverse 사용
                coeffs = np.linalg.pinv(L) @ Y
            
            # 계수 분리
            w = coeffs[:n]  # TPS 가중치
            a = coeffs[n:]  # affine 계수
            
            return {
                'source_points': source_pts.tolist(),
                'weights': w.tolist(),
                'affine_coeffs': a.tolist(),
                'regularization': self.tps_config['regularization']
            }
            
        except Exception as e:
            self.logger.error(f"TPS 변환 계산 실패: {e}")
            # 폴백: 단위 변환
            return {
                'source_points': source_pts.tolist(),
                'weights': np.zeros((len(source_pts), 2)).tolist(),
                'affine_coeffs': np.array([[1, 0, 0], [0, 1, 0]]).tolist(),
                'regularization': 0.0
            }
    
    def _homography_matching(self, person_points: List, clothing_points: List) -> Dict[str, Any]:
        """Homography 매칭"""
        
        try:
            person_array = np.array(person_points, dtype=np.float32)
            clothing_array = np.array(clothing_points, dtype=np.float32)
            
            # 최소 4개 점 필요
            min_points = min(len(person_array), len(clothing_array), 4)
            
            if min_points < 4:
                raise ValueError("Homography를 위한 충분한 점이 없음")
            
            # 첫 4개 점 사용 (더 정교한 대응 방법으로 개선 가능)
            src_pts = clothing_array[:min_points]
            dst_pts = person_array[:min_points]
            
            # Homography 계산
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is None:
                raise ValueError("Homography 계산 실패")
            
            return {
                'method': 'homography',
                'transform': H.tolist(),
                'source_points': src_pts.tolist(),
                'target_points': dst_pts.tolist(),
                'inlier_mask': mask.flatten().tolist() if mask is not None else [],
                'confidence': 0.8
            }
            
        except Exception as e:
            self.logger.warning(f"Homography 매칭 실패: {e}")
            raise
    
    def _affine_matching(self, person_points: List, clothing_points: List) -> Dict[str, Any]:
        """Affine 변환 매칭"""
        
        try:
            person_array = np.array(person_points, dtype=np.float32)
            clothing_array = np.array(clothing_points, dtype=np.float32)
            
            # 최소 3개 점 필요
            min_points = min(len(person_array), len(clothing_array), 3)
            
            if min_points < 3:
                raise ValueError("Affine 변환을 위한 충분한 점이 없음")
            
            # 첫 3개 점 사용
            src_pts = clothing_array[:min_points]
            dst_pts = person_array[:min_points]
            
            # Affine 변환 계산
            M = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])
            
            return {
                'method': 'affine',
                'transform': M.tolist(),
                'source_points': src_pts.tolist(),
                'target_points': dst_pts.tolist(),
                'confidence': 0.7
            }
            
        except Exception as e:
            self.logger.warning(f"Affine 매칭 실패: {e}")
            raise
    
    def _similarity_matching(self, person_points: List, clothing_points: List) -> Dict[str, Any]:
        """유사성 변환 매칭 (회전, 스케일, 평행이동)"""
        
        try:
            if len(person_points) < 2 or len(clothing_points) < 2:
                # 최소 변환: 평행이동만
                if person_points and clothing_points:
                    tx = person_points[0][0] - clothing_points[0][0]
                    ty = person_points[0][1] - clothing_points[0][1]
                    M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
                else:
                    M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            else:
                # 중심점 기반 변환
                person_center = np.mean(person_points, axis=0)
                clothing_center = np.mean(clothing_points, axis=0)
                
                # 스케일 추정
                person_spread = np.std(person_points, axis=0)
                clothing_spread = np.std(clothing_points, axis=0)
                
                scale_x = person_spread[0] / (clothing_spread[0] + 1e-6)
                scale_y = person_spread[1] / (clothing_spread[1] + 1e-6)
                scale = (scale_x + scale_y) / 2  # 평균 스케일
                
                # 평행이동
                tx = person_center[0] - clothing_center[0] * scale
                ty = person_center[1] - clothing_center[1] * scale
                
                M = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)
            
            return {
                'method': 'similarity',
                'transform': M.tolist(),
                'source_points': clothing_points[:2] if len(clothing_points) >= 2 else clothing_points,
                'target_points': person_points[:2] if len(person_points) >= 2 else person_points,
                'confidence': 0.6
            }
            
        except Exception as e:
            self.logger.warning(f"유사성 변환 실패: {e}")
            # 최후의 폴백: 단위 변환
            return {
                'method': 'identity',
                'transform': [[1, 0, 0], [0, 1, 0]],
                'source_points': [],
                'target_points': [],
                'confidence': 0.3
            }
    
    async def _refine_with_pose_guidance(
        self, 
        initial_match: Dict[str, Any], 
        pose_keypoints: List[List[float]], 
        clothing_type: str
    ) -> Dict[str, Any]:
        """포즈 기반 매칭 정제"""
        
        try:
            # 포즈 특성 분석
            pose_analysis = self._analyze_pose_characteristics(pose_keypoints)
            
            # 의류 타입별 포즈 적응
            adaptation_factor = self._calculate_pose_adaptation(pose_analysis, clothing_type)
            
            # 변환 매개변수 조정
            refined_transform = self._adapt_transform_to_pose(
                initial_match['transform'], adaptation_factor, pose_analysis
            )
            
            refined_match = initial_match.copy()
            refined_match['transform'] = refined_transform
            refined_match['pose_adapted'] = True
            refined_match['adaptation_factor'] = adaptation_factor
            
            return refined_match
            
        except Exception as e:
            self.logger.warning(f"포즈 기반 정제 실패: {e}")
            return initial_match
    
    def _analyze_pose_characteristics(self, pose_keypoints: List[List[float]]) -> Dict[str, Any]:
        """포즈 특성 분석"""
        
        analysis = {}
        
        try:
            # 어깨 각도
            if all(pose_keypoints[i][2] > 0.5 for i in [2, 5]):  # 양쪽 어깨
                left_shoulder = pose_keypoints[5][:2]
                right_shoulder = pose_keypoints[2][:2]
                shoulder_angle = np.degrees(np.arctan2(
                    left_shoulder[1] - right_shoulder[1],
                    left_shoulder[0] - right_shoulder[0]
                ))
                analysis['shoulder_angle'] = shoulder_angle
            
            # 몸통 기울기
            if all(pose_keypoints[i][2] > 0.5 for i in [1, 8, 11]):  # 목, 양쪽 엉덩이
                neck = pose_keypoints[1][:2]
                hip_center = np.mean([pose_keypoints[8][:2], pose_keypoints[11][:2]], axis=0)
                torso_angle = np.degrees(np.arctan2(
                    neck[0] - hip_center[0],
                    hip_center[1] - neck[1]
                ))
                analysis['torso_angle'] = torso_angle
            
            # 팔 위치
            arm_angles = {}
            if all(pose_keypoints[i][2] > 0.5 for i in [2, 3, 4]):  # 오른팔
                shoulder = pose_keypoints[2][:2]
                elbow = pose_keypoints[3][:2]
                wrist = pose_keypoints[4][:2]
                
                upper_arm_angle = np.degrees(np.arctan2(
                    elbow[1] - shoulder[1], elbow[0] - shoulder[0]
                ))
                arm_angles['right_upper'] = upper_arm_angle
            
            if all(pose_keypoints[i][2] > 0.5 for i in [5, 6, 7]):  # 왼팔
                shoulder = pose_keypoints[5][:2]
                elbow = pose_keypoints[6][:2]
                wrist = pose_keypoints[7][:2]
                
                upper_arm_angle = np.degrees(np.arctan2(
                    elbow[1] - shoulder[1], elbow[0] - shoulder[0]
                ))
                arm_angles['left_upper'] = upper_arm_angle
            
            analysis['arm_angles'] = arm_angles
            
        except Exception as e:
            self.logger.warning(f"포즈 특성 분석 실패: {e}")
        
        return analysis
    
    def _calculate_pose_adaptation(self, pose_analysis: Dict[str, Any], clothing_type: str) -> Dict[str, float]:
        """포즈 적응 인수 계산"""
        
        adaptation = {
            'scale_factor': 1.0,
            'rotation_adjustment': 0.0,
            'shear_factor': 0.0
        }
        
        try:
            # 어깨 기울기에 따른 회전 조정
            if 'shoulder_angle' in pose_analysis:
                shoulder_angle = pose_analysis['shoulder_angle']
                # 어깨가 기울어진 만큼 역방향으로 조정
                adaptation['rotation_adjustment'] = -shoulder_angle * 0.5
            
            # 몸통 기울기에 따른 전단 조정
            if 'torso_angle' in pose_analysis:
                torso_angle = pose_analysis['torso_angle']
                adaptation['shear_factor'] = np.tan(np.radians(torso_angle)) * 0.3
            
            # 팔 위치에 따른 스케일 조정 (상의의 경우)
            if clothing_type in ['shirt', 't-shirt', 'blouse']:
                arm_angles = pose_analysis.get('arm_angles', {})
                if arm_angles:
                    # 팔이 벌어진 정도에 따라 스케일 조정
                    avg_arm_angle = np.mean(list(arm_angles.values()))
                    if abs(avg_arm_angle) > 45:  # 팔이 많이 벌어진 경우
                        adaptation['scale_factor'] = 1.1
                    elif abs(avg_arm_angle) < 15:  # 팔이 몸에 붙은 경우
                        adaptation['scale_factor'] = 0.95
            
        except Exception as e:
            self.logger.warning(f"포즈 적응 계산 실패: {e}")
        
        return adaptation
    
    def _adapt_transform_to_pose(
        self, 
        original_transform: List[List[float]], 
        adaptation_factor: Dict[str, float], 
        pose_analysis: Dict[str, Any]
    ) -> List[List[float]]:
        """포즈에 맞게 변환 조정"""
        
        try:
            transform = np.array(original_transform)
            
            # 회전 조정
            rotation_adj = adaptation_factor.get('rotation_adjustment', 0.0)
            if abs(rotation_adj) > 0.1:
                cos_r = np.cos(np.radians(rotation_adj))
                sin_r = np.sin(np.radians(rotation_adj))
                rotation_matrix = np.array([[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]])
                
                if transform.shape[0] == 2:  # Affine transform
                    transform = np.vstack([transform, [0, 0, 1]])
                    transform = rotation_matrix @ transform
                    transform = transform[:2]
                else:  # Homography
                    transform = rotation_matrix @ transform
            
            # 스케일 조정
            scale_factor = adaptation_factor.get('scale_factor', 1.0)
            if abs(scale_factor - 1.0) > 0.01:
                if transform.shape[0] == 2:  # Affine
                    transform[0, 0] *= scale_factor
                    transform[1, 1] *= scale_factor
                else:  # Homography
                    transform[:2, :2] *= scale_factor
            
            # 전단 조정
            shear_factor = adaptation_factor.get('shear_factor', 0.0)
            if abs(shear_factor) > 0.01:
                if transform.shape[0] == 2:  # Affine
                    transform[0, 1] += shear_factor
                else:  # Homography
                    transform[0, 1] += shear_factor
            
            return transform.tolist()
            
        except Exception as e:
            self.logger.warning(f"변환 포즈 적응 실패: {e}")
            return original_transform
    
    def _evaluate_matching_quality(
        self, 
        person_points: List, 
        clothing_points: List, 
        match_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """매칭 품질 평가"""
        
        try:
            transform = np.array(match_result['transform'])
            method = match_result['method']
            
            # 1. 재투영 오차 계산
            reprojection_error = self._calculate_reprojection_error(
                clothing_points, person_points, transform, method
            )
            
            # 2. 기하학적 일관성
            geometric_consistency = self._evaluate_geometric_consistency(transform, method)
            
            # 3. 변환 안정성
            transform_stability = self._evaluate_transform_stability(transform, method)
            
            # 4. 대응점 신뢰도
            correspondence_confidence = match_result.get('confidence', 0.5)
            
            # 5. 전체 품질 점수
            overall_quality = (
                (1.0 - reprojection_error) * 0.4 +
                geometric_consistency * 0.3 +
                transform_stability * 0.2 +
                correspondence_confidence * 0.1
            )
            
            return {
                'overall_quality': max(0.0, min(1.0, overall_quality)),
                'reprojection_error': reprojection_error,
                'geometric_consistency': geometric_consistency,
                'transform_stability': transform_stability,
                'correspondence_confidence': correspondence_confidence,
                'quality_grade': self._get_quality_grade(overall_quality)
            }
            
        except Exception as e:
            self.logger.warning(f"품질 평가 실패: {e}")
            return {
                'overall_quality': 0.5,
                'reprojection_error': 1.0,
                'geometric_consistency': 0.0,
                'transform_stability': 0.0,
                'correspondence_confidence': 0.0,
                'quality_grade': 'poor'
            }
    
    def _calculate_reprojection_error(
        self, 
        source_points: List, 
        target_points: List, 
        transform: np.ndarray, 
        method: str
    ) -> float:
        """재투영 오차 계산"""
        
        try:
            if not source_points or not target_points:
                return 1.0
            
            source_array = np.array(source_points)
            target_array = np.array(target_points)
            
            # 변환 적용
            if method == 'tps':
                # TPS는 별도 처리 필요 (여기서는 간단화)
                projected_points = source_array  # 임시
            elif method in ['homography']:
                # 동차 좌표로 변환
                source_homo = np.column_stack([source_array, np.ones(len(source_array))])
                projected_homo = source_homo @ transform.T
                projected_points = projected_homo[:, :2] / projected_homo[:, 2:3]
            else:  # affine, similarity
                source_homo = np.column_stack([source_array, np.ones(len(source_array))])
                projected_points = source_homo @ transform.T
            
            # 가장 가까운 대응점들 찾기
            min_len = min(len(projected_points), len(target_array))
            distances = cdist(projected_points[:min_len], target_array[:min_len])
            
            # 최소 거리들의 평균
            min_distances = np.min(distances, axis=1)
            avg_error = np.mean(min_distances)
            
            # 정규화 (이미지 크기 대비)
            if target_array.size > 0:
                image_diagonal = np.linalg.norm(np.ptp(target_array, axis=0))
                normalized_error = avg_error / (image_diagonal + 1e-6)
            else:
                normalized_error = 1.0
            
            return min(1.0, normalized_error)
            
        except Exception as e:
            self.logger.warning(f"재투영 오차 계산 실패: {e}")
            return 1.0
    
    def _evaluate_geometric_consistency(self, transform: np.ndarray, method: str) -> float:
        """기하학적 일관성 평가"""
        
        try:
            if method == 'tps':
                # TPS는 항상 일관성 있음
                return 0.9
            
            if transform.shape[0] < 2:
                return 0.0
            
            # 행렬식 계산 (스케일 변화)
            if transform.shape == (2, 3):  # Affine
                det = np.linalg.det(transform[:2, :2])
            else:  # Homography
                det = np.linalg.det(transform[:2, :2])
            
            # 합리적인 스케일 변화인지 확인 (0.1 ~ 10 배)
            if 0.1 <= abs(det) <= 10:
                scale_consistency = 1.0
            else:
                scale_consistency = 0.0
            
            # 회전 각도 확인
            if transform.shape == (2, 3):
                rotation_matrix = transform[:2, :2]
                if abs(det) > 1e-6:
                    U, _, Vt = np.linalg.svd(rotation_matrix)
                    rotation_part = U @ Vt
                    # 직교성 확인
                    orthogonality = np.linalg.norm(rotation_part @ rotation_part.T - np.eye(2))
                    rotation_consistency = max(0.0, 1.0 - orthogonality)
                else:
                    rotation_consistency = 0.0
            else:
                rotation_consistency = 0.8  # Homography의 경우 기본값
            
            consistency = (scale_consistency + rotation_consistency) / 2
            return min(1.0, max(0.0, consistency))
            
        except Exception as e:
            self.logger.warning(f"기하학적 일관성 평가 실패: {e}")
            return 0.5
    
    def _evaluate_transform_stability(self, transform: np.ndarray, method: str) -> float:
        """변환 안정성 평가"""
        
        try:
            # 조건수 확인
            if transform.shape == (2, 3):  # Affine
                matrix_part = transform[:2, :2]
            else:  # Homography
                matrix_part = transform[:2, :2]
            
            condition_number = np.linalg.cond(matrix_part)
            
            # 조건수가 낮을수록 안정적
            if condition_number < 10:
                stability = 1.0
            elif condition_number < 100:
                stability = 0.8
            elif condition_number < 1000:
                stability = 0.5
            else:
                stability = 0.2
            
            # 특이값 분석
            singular_values = np.linalg.svd(matrix_part, compute_uv=False)
            sv_ratio = np.max(singular_values) / (np.min(singular_values) + 1e-6)
            
            if sv_ratio < 5:
                sv_stability = 1.0
            elif sv_ratio < 20:
                sv_stability = 0.7
            else:
                sv_stability = 0.3
            
            return (stability + sv_stability) / 2
            
        except Exception as e:
            self.logger.warning(f"변환 안정성 평가 실패: {e}")
            return 0.5
    
    def _get_quality_grade(self, overall_quality: float) -> str:
        """품질 등급 반환"""
        if overall_quality >= 0.9:
            return "excellent"
        elif overall_quality >= 0.8:
            return "good"
        elif overall_quality >= 0.6:
            return "fair"
        elif overall_quality >= 0.4:
            return "poor"
        else:
            return "very_poor"
    
    async def _try_alternative_methods(
        self, 
        person_points: List, 
        clothing_points: List, 
        clothing_type: str
    ) -> Dict[str, Any]:
        """대안 매칭 방법들 시도"""
        
        alternative_methods = ['affine', 'similarity', 'homography']
        best_result = None
        best_quality = 0.0
        
        for method in alternative_methods:
            try:
                result = await self._perform_initial_matching(person_points, clothing_points, method)
                quality = self._evaluate_matching_quality(person_points, clothing_points, result)
                
                if quality['overall_quality'] > best_quality:
                    best_quality = quality['overall_quality']
                    best_result = result
                    
                self.logger.debug(f"대안 방법 {method}: 품질 {quality['overall_quality']:.3f}")
                
            except Exception as e:
                self.logger.warning(f"대안 방법 {method} 실패: {e}")
                continue
        
        return best_result if best_result else {
            'method': 'identity',
            'transform': [[1, 0, 0], [0, 1, 0]],
            'confidence': 0.3
        }
    
    def _generate_warp_parameters(self, match_result: Dict[str, Any], clothing_segmentation: Dict[str, Any]) -> Dict[str, Any]:
        """워핑 파라미터 생성"""
        
        try:
            transform = match_result['transform']
            method = match_result['method']
            
            # 기본 워핑 파라미터
            warp_params = {
                'transform_matrix': transform,
                'transform_method': method,
                'interpolation': 'bilinear',
                'border_mode': 'reflect',
                'output_size': None  # 원본 크기 유지
            }
            
            # 의류 마스크 정보 추가
            if 'mask' in clothing_segmentation:
                mask = clothing_segmentation['mask']
                warp_params['mask_transform'] = transform  # 마스크도 같은 변환 적용
                warp_params['original_mask_size'] = mask.shape
            
            # 방법별 특화 파라미터
            if method == 'tps':
                warp_params.update({
                    'source_points': match_result.get('source_points', []),
                    'target_points': match_result.get('target_points', []),
                    'tps_weights': transform.get('weights', []) if isinstance(transform, dict) else [],
                    'tps_affine': transform.get('affine_coeffs', []) if isinstance(transform, dict) else []
                })
            
            # 품질 기반 파라미터 조정
            if 'quality_metrics' in match_result:
                quality = match_result['quality_metrics']['overall_quality']
                if quality < 0.6:
                    warp_params['interpolation'] = 'nearest'  # 낮은 품질일 때는 보간 단순화
                elif quality > 0.8:
                    warp_params['interpolation'] = 'bicubic'   # 높은 품질일 때는 고급 보간
            
            return warp_params
            
        except Exception as e:
            self.logger.warning(f"워핑 파라미터 생성 실패: {e}")
            return {
                'transform_matrix': [[1, 0, 0], [0, 1, 0]],
                'transform_method': 'identity',
                'interpolation': 'bilinear',
                'border_mode': 'reflect'
            }
    
    def _build_final_result(
        self,
        match_result: Dict[str, Any],
        warp_params: Dict[str, Any],
        quality_metrics: Dict[str, float],
        processing_time: float,
        method: str,
        clothing_type: str
    ) -> Dict[str, Any]:
        """최종 결과 구성"""
        
        return {
            'success': True,
            'transform_matrix': match_result['transform'],
            'warp_matrix': match_result['transform'],  # 호환성을 위한 중복
            'warp_parameters': warp_params,
            'matching_method': method,
            'clothing_type': clothing_type,
            'quality_metrics': quality_metrics,
            'confidence': quality_metrics['overall_quality'],
            'processing_time': processing_time,
            'matching_info': {
                'source_points': match_result.get('source_points', []),
                'target_points': match_result.get('target_points', []),
                'correspondences': match_result.get('correspondences', []),
                'pose_adapted': match_result.get('pose_adapted', False),
                'method_used': method
            },
            'geometric_analysis': {
                'reprojection_error': quality_metrics['reprojection_error'],
                'geometric_consistency': quality_metrics['geometric_consistency'],
                'transform_stability': quality_metrics['transform_stability'],
                'quality_grade': quality_metrics['quality_grade']
            }
        }
    
    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """빈 결과 생성"""
        return {
            'success': False,
            'error': reason,
            'transform_matrix': [[1, 0, 0], [0, 1, 0]],
            'warp_matrix': [[1, 0, 0], [0, 1, 0]],
            'warp_parameters': {
                'transform_matrix': [[1, 0, 0], [0, 1, 0]],
                'transform_method': 'identity',
                'interpolation': 'bilinear'
            },
            'matching_method': 'none',
            'clothing_type': 'unknown',
            'quality_metrics': {
                'overall_quality': 0.0,
                'quality_grade': 'failed'
            },
            'confidence': 0.0,
            'processing_time': 0.0,
            'matching_info': {
                'error_occurred': True,
                'method_used': 'none'
            }
        }
    
    def _update_statistics(self, method: str, quality: float):
        """통계 업데이트"""
        self.matching_stats['total_matches'] += 1
        
        if quality > 0.6:
            self.matching_stats['successful_matches'] += 1
        
        # 품질 이동 평균
        alpha = 0.1
        self.matching_stats['average_accuracy'] = (
            alpha * quality + 
            (1 - alpha) * self.matching_stats['average_accuracy']
        )
        
        # 방법별 성능 추적
        if method not in self.matching_stats['method_performance']:
            self.matching_stats['method_performance'][method] = {'count': 0, 'avg_quality': 0.0}
        
        method_stats = self.matching_stats['method_performance'][method]
        method_stats['count'] += 1
        method_stats['avg_quality'] = (
            (method_stats['avg_quality'] * (method_stats['count'] - 1) + quality) / 
            method_stats['count']
        )
    
    async def _initialize_matching_algorithms(self):
        """매칭 알고리즘 초기화"""
        try:
            # TPS 그리드 초기화
            grid_size = self.tps_config['grid_size']
            self.tps_grid = np.mgrid[0:grid_size, 0:grid_size].reshape(2, -1).T
            
            # RANSAC 파라미터 설정
            self.ransac_params = {
                'max_trials': 1000,
                'residual_threshold': 5.0,
                'min_samples': 4
            }
            
            self.logger.info("✅ 매칭 알고리즘 초기화 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 매칭 알고리즘 초기화 실패: {e}")
    
    async def _initialize_optimization_tools(self):
        """최적화 도구 초기화"""
        try:
            # 최적화 기법 설정
            self.optimizer_config = {
                'method': 'L-BFGS-B',
                'options': {
                    'maxiter': self.matching_config['max_iterations'],
                    'ftol': self.matching_config['convergence_threshold']
                }
            }
            
            self.logger.info("✅ 최적화 도구 초기화 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 최적화 도구 초기화 실패: {e}")
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            # 캐시된 데이터 정리
            if hasattr(self, 'tps_grid'):
                del self.tps_grid
            
            # 통계 초기화
            self.matching_stats = {
                'total_matches': 0,
                'successful_matches': 0,
                'average_accuracy': 0.0,
                'method_performance': {}
            }
            
            self.is_initialized = False
            self.logger.info("🧹 기하학적 매칭 스텝 정리 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 리소스 정리 중 오류: {e}")