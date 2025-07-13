"""
MyCloset AI 4단계: 기하학적 매칭 (Geometric Matching)
TPS (Thin Plate Spline) 변환 기반 신체-의류 간 대응점 매칭 시스템
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import logging
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import math

class TPSTransform:
    """Thin Plate Spline 변환 클래스"""
    
    def __init__(self):
        self.source_points = None
        self.target_points = None
        self.weights = None
        self.affine_params = None

    def fit(self, source_points: np.ndarray, target_points: np.ndarray):
        """TPS 변환 파라미터 계산"""
        n = source_points.shape[0]
        
        # TPS 커널 행렬 K 계산
        K = self._compute_kernel_matrix(source_points)
        
        # P 행렬 (아핀 변환용)
        P = np.hstack([np.ones((n, 1)), source_points])
        
        # L 행렬 구성
        L = np.zeros((n + 3, n + 3))
        L[:n, :n] = K
        L[:n, n:] = P
        L[n:, :n] = P.T
        
        # Y 벡터 (목표점 + 제약조건)
        Y = np.zeros((n + 3, 2))
        Y[:n] = target_points
        
        # 선형 시스템 해결
        try:
            solution = np.linalg.solve(L, Y)
            self.weights = solution[:n]
            self.affine_params = solution[n:]
            self.source_points = source_points.copy()
            self.target_points = target_points.copy()
            return True
        except np.linalg.LinAlgError:
            logging.warning("TPS 변환 계산 실패")
            return False

    def _compute_kernel_matrix(self, points: np.ndarray) -> np.ndarray:
        """TPS 커널 행렬 계산"""
        n = points.shape[0]
        K = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    r_sq = np.sum((points[i] - points[j]) ** 2)
                    if r_sq > 0:
                        K[i, j] = r_sq * np.log(r_sq)
        
        return K

    def transform(self, points: np.ndarray) -> np.ndarray:
        """점들에 TPS 변환 적용"""
        if self.weights is None or self.affine_params is None:
            raise ValueError("TPS 변환이 학습되지 않았습니다")
        
        n_source = self.source_points.shape[0]
        n_points = points.shape[0]
        
        # 비선형 부분 계산
        nonlinear_part = np.zeros((n_points, 2))
        for i in range(n_source):
            r_sq = np.sum((points - self.source_points[i]) ** 2, axis=1)
            mask = r_sq > 0
            if np.any(mask):
                phi = np.zeros_like(r_sq)
                phi[mask] = r_sq[mask] * np.log(r_sq[mask])
                nonlinear_part += self.weights[i] * phi.reshape(-1, 1)
        
        # 아핀 부분 계산
        P = np.hstack([np.ones((n_points, 1)), points])
        affine_part = P @ self.affine_params
        
        return nonlinear_part + affine_part

    def transform_image(self, image: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
        """이미지에 TPS 변환 적용"""
        h, w = output_shape
        
        # 출력 이미지의 모든 픽셀 좌표 생성
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        coords = np.column_stack([x_coords.ravel(), y_coords.ravel()])
        
        # 역변환으로 소스 좌표 계산
        source_coords = self.inverse_transform(coords)
        
        # 이미지 보간
        transformed_image = self._interpolate_image(image, source_coords, (h, w))
        
        return transformed_image

    def inverse_transform(self, points: np.ndarray) -> np.ndarray:
        """TPS 역변환 (근사적)"""
        # 반복적 방법으로 역변환 근사
        transformed_points = points.copy()
        
        for _ in range(5):  # 5회 반복
            forward_transform = self.transform(transformed_points)
            error = points - forward_transform
            transformed_points += 0.5 * error  # 수렴 속도 조절
        
        return transformed_points

    def _interpolate_image(self, image: np.ndarray, coords: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
        """양선형 보간으로 이미지 변환"""
        h, w = output_shape
        
        # 좌표 분리
        x_coords = coords[:, 0].reshape(h, w)
        y_coords = coords[:, 1].reshape(h, w)
        
        # OpenCV의 remap 함수 사용
        if len(image.shape) == 3:
            transformed = cv2.remap(
                image, 
                x_coords.astype(np.float32), 
                y_coords.astype(np.float32),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
        else:
            transformed = cv2.remap(
                image, 
                x_coords.astype(np.float32), 
                y_coords.astype(np.float32),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
        
        return transformed

class FeaturePointMatcher:
    """특징점 매칭 클래스"""
    
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()

    def extract_keypoints(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[List, np.ndarray]:
        """SIFT 특징점 추출"""
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 특징점 검출
        keypoints, descriptors = self.sift.detectAndCompute(gray, mask)
        
        # 키포인트를 좌표 배열로 변환
        if keypoints:
            points = np.array([kp.pt for kp in keypoints])
        else:
            points = np.empty((0, 2))
        
        return keypoints, points, descriptors

    def match_features(self, desc1: np.ndarray, desc2: np.ndarray, ratio_threshold: float = 0.75) -> List[Tuple[int, int]]:
        """특징점 매칭 (Lowe's ratio test)"""
        if desc1 is None or desc2 is None:
            return []
        
        # kNN 매칭 (k=2)
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Lowe's ratio test로 좋은 매칭만 선별
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append((m.queryIdx, m.trainIdx))
        
        return good_matches

class ContourMatcher:
    """윤곽선 기반 매칭 클래스"""
    
    def __init__(self):
        pass

    def extract_contour_points(self, mask: np.ndarray, num_points: int = 50) -> np.ndarray:
        """마스크에서 윤곽선 점들 추출"""
        # 윤곽선 찾기
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.empty((0, 2))
        
        # 가장 큰 윤곽선 선택
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 윤곽선 점들을 균등하게 샘플링
        perimeter = cv2.arcLength(largest_contour, True)
        epsilon = perimeter / num_points
        
        # 윤곽선 근사화
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 점들을 균등 간격으로 재샘플링
        contour_points = self._resample_contour(largest_contour.reshape(-1, 2), num_points)
        
        return contour_points

    def _resample_contour(self, contour: np.ndarray, num_points: int) -> np.ndarray:
        """윤곽선을 균등 간격으로 재샘플링"""
        # 윤곽선 길이 계산
        distances = np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1))
        cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
        total_length = cumulative_distances[-1]
        
        # 균등 간격 점들 계산
        target_distances = np.linspace(0, total_length, num_points)
        resampled_points = np.zeros((num_points, 2))
        
        for i, target_dist in enumerate(target_distances):
            # 가장 가까운 세그먼트 찾기
            idx = np.searchsorted(cumulative_distances, target_dist) - 1
            idx = max(0, min(idx, len(contour) - 2))
            
            # 선형 보간
            segment_start = cumulative_distances[idx]
            segment_end = cumulative_distances[idx + 1]
            
            if segment_end > segment_start:
                t = (target_dist - segment_start) / (segment_end - segment_start)
                resampled_points[i] = (1 - t) * contour[idx] + t * contour[idx + 1]
            else:
                resampled_points[i] = contour[idx]
        
        return resampled_points

    def match_contours(self, contour1: np.ndarray, contour2: np.ndarray) -> List[Tuple[int, int]]:
        """두 윤곽선 간 대응점 매칭"""
        if len(contour1) == 0 or len(contour2) == 0:
            return []
        
        # 거리 행렬 계산
        distance_matrix = cdist(contour1, contour2)
        
        # Hungarian 알고리즘으로 최적 매칭
        row_indices, col_indices = linear_sum_assignment(distance_matrix)
        
        # 매칭 결과 반환
        matches = list(zip(row_indices, col_indices))
        
        return matches

class GeometricMatchingStep:
    """4단계: 기하학적 매칭 실행 클래스"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # 매칭 컴포넌트 초기화
        self.tps_transform = TPSTransform()
        self.feature_matcher = FeaturePointMatcher()
        self.contour_matcher = ContourMatcher()
        
        # 매칭 파라미터
        self.min_matches = 4  # TPS 변환을 위한 최소 대응점 수
        self.feature_ratio_threshold = 0.75
        self.contour_points = 30
        self.max_distance_threshold = 50.0

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """기하학적 매칭 메인 처리"""
        try:
            # 입력 데이터 추출
            person_image = input_data["person_image"]
            cloth_image = input_data["cloth_image"]
            human_parsing = input_data["human_parsing"]
            pose_keypoints = input_data["pose_keypoints"]
            cloth_mask = input_data["cloth_mask"]
            
            # 텐서를 numpy로 변환
            person_np = self._tensor_to_numpy(person_image)
            cloth_np = self._tensor_to_numpy(cloth_image)
            cloth_mask_np = self._tensor_to_numpy(cloth_mask).squeeze()
            
            # 신체 영역 마스크 생성
            body_mask = self._create_body_mask(human_parsing, person_np.shape[:2])
            
            # 다중 방법으로 대응점 찾기
            matching_results = await self._find_correspondences(
                person_np, cloth_np, body_mask, cloth_mask_np, pose_keypoints
            )
            
            # TPS 변환 계산
            tps_result = await self._compute_tps_transform(matching_results)
            
            # 변환 품질 평가
            quality_metrics = await self._evaluate_transform_quality(tps_result, matching_results)
            
            return {
                "tps_transform": tps_result["transform"],
                "correspondence_points": tps_result["correspondences"],
                "transform_matrix": tps_result["matrix"],
                "quality_metrics": quality_metrics,
                "matching_methods": {
                    "feature_matches": matching_results["feature_matches"],
                    "contour_matches": matching_results["contour_matches"],
                    "pose_guided_matches": matching_results["pose_matches"]
                },
                "metadata": {
                    "total_correspondences": len(tps_result["correspondences"]),
                    "transform_type": "TPS",
                    "processing_time": tps_result["processing_time"]
                }
            }
            
        except Exception as e:
            self.logger.error(f"기하학적 매칭 처리 중 오류: {str(e)}")
            raise

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """텐서를 numpy 배열로 변환"""
        if tensor.dim() == 4:  # [B, C, H, W]
            numpy_array = tensor[0].cpu().numpy()
            if numpy_array.shape[0] == 3:  # RGB
                numpy_array = numpy_array.transpose(1, 2, 0)
        elif tensor.dim() == 3:  # [C, H, W]
            numpy_array = tensor.cpu().numpy()
            if numpy_array.shape[0] == 3:  # RGB
                numpy_array = numpy_array.transpose(1, 2, 0)
            else:
                numpy_array = numpy_array.squeeze(0)
        else:
            numpy_array = tensor.cpu().numpy()
        
        # [0, 1] → [0, 255] 변환
        if numpy_array.max() <= 1.0:
            numpy_array = (numpy_array * 255).astype(np.uint8)
        
        return numpy_array

    def _create_body_mask(self, human_parsing: Dict[str, Any], image_shape: Tuple[int, int]) -> np.ndarray:
        """인체 파싱에서 신체 마스크 생성"""
        if isinstance(human_parsing["parsing_map"], np.ndarray):
            parsing_map = human_parsing["parsing_map"]
        else:
            parsing_map = human_parsing["parsing_map"].cpu().numpy()
        
        # 의류 관련 부위 ID (상의, 하의, 원피스 등)
        clothing_ids = [5, 6, 7, 9, 10, 12]  # upper_clothes, dress, coat, pants, jumpsuits, skirt
        
        # 의류 영역 마스크 생성
        body_mask = np.zeros_like(parsing_map, dtype=np.uint8)
        for cloth_id in clothing_ids:
            body_mask[parsing_map == cloth_id] = 1
        
        # 이미지 크기에 맞게 리사이즈
        if body_mask.shape != image_shape:
            body_mask = cv2.resize(body_mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        
        return body_mask

    async def _find_correspondences(
        self, 
        person_image: np.ndarray, 
        cloth_image: np.ndarray,
        body_mask: np.ndarray,
        cloth_mask: np.ndarray,
        pose_keypoints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """다중 방법으로 대응점 찾기"""
        import time
        start_time = time.time()
        
        # 1. 특징점 기반 매칭
        feature_matches = await self._feature_based_matching(
            person_image, cloth_image, body_mask, cloth_mask
        )
        
        # 2. 윤곽선 기반 매칭
        contour_matches = await self._contour_based_matching(
            body_mask, cloth_mask
        )
        
        # 3. 포즈 가이드 매칭
        pose_matches = await self._pose_guided_matching(
            pose_keypoints, body_mask, cloth_mask
        )
        
        processing_time = time.time() - start_time
        
        return {
            "feature_matches": feature_matches,
            "contour_matches": contour_matches,
            "pose_matches": pose_matches,
            "processing_time": processing_time
        }

    async def _feature_based_matching(
        self, 
        person_image: np.ndarray, 
        cloth_image: np.ndarray,
        body_mask: np.ndarray,
        cloth_mask: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """SIFT 특징점 기반 매칭"""
        try:
            # 특징점 추출
            person_kp, person_pts, person_desc = self.feature_matcher.extract_keypoints(
                person_image, body_mask
            )
            cloth_kp, cloth_pts, cloth_desc = self.feature_matcher.extract_keypoints(
                cloth_image, cloth_mask.astype(np.uint8)
            )
            
            if len(person_pts) == 0 or len(cloth_pts) == 0:
                return []
            
            # 특징점 매칭
            matches = self.feature_matcher.match_features(
                person_desc, cloth_desc, self.feature_ratio_threshold
            )
            
            # 대응점 추출
            correspondences = []
            for person_idx, cloth_idx in matches:
                person_pt = person_pts[person_idx]
                cloth_pt = cloth_pts[cloth_idx]
                correspondences.append((person_pt, cloth_pt))
            
            return correspondences
            
        except Exception as e:
            self.logger.warning(f"특징점 매칭 실패: {e}")
            return []

    async def _contour_based_matching(
        self, 
        body_mask: np.ndarray, 
        cloth_mask: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """윤곽선 기반 매칭"""
        try:
            # 윤곽선 점 추출
            body_contour = self.contour_matcher.extract_contour_points(
                body_mask, self.contour_points
            )
            cloth_contour = self.contour_matcher.extract_contour_points(
                cloth_mask, self.contour_points
            )
            
            if len(body_contour) == 0 or len(cloth_contour) == 0:
                return []
            
            # 윤곽선 매칭
            matches = self.contour_matcher.match_contours(body_contour, cloth_contour)
            
            # 대응점 추출
            correspondences = []
            for body_idx, cloth_idx in matches:
                body_pt = body_contour[body_idx]
                cloth_pt = cloth_contour[cloth_idx]
                
                # 거리 임계치 확인
                distance = np.linalg.norm(body_pt - cloth_pt)
                if distance < self.max_distance_threshold:
                    correspondences.append((body_pt, cloth_pt))
            
            return correspondences
            
        except Exception as e:
            self.logger.warning(f"윤곽선 매칭 실패: {e}")
            return []

    async def _pose_guided_matching(
        self, 
        pose_keypoints: Dict[str, Any],
        body_mask: np.ndarray,
        cloth_mask: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """포즈 가이드 매칭"""
        try:
            if "keypoints" not in pose_keypoints:
                return []
            
            keypoints = pose_keypoints["keypoints"]
            correspondences = []
            
            # 의류와 관련된 키포인트들 (어깨, 팔꿈치, 손목, 엉덩이 등)
            relevant_keypoint_indices = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # OpenPose 기준
            
            for kp_idx in relevant_keypoint_indices:
                if kp_idx < len(keypoints):
                    x, y, confidence = keypoints[kp_idx]
                    
                    if confidence > 0.5 and 0 <= x < body_mask.shape[1] and 0 <= y < body_mask.shape[0]:
                        # 신체 마스크에서 해당 점 확인
                        if body_mask[int(y), int(x)] > 0:
                            # 의류 마스크에서 대응점 찾기
                            cloth_pt = self._find_corresponding_cloth_point(
                                np.array([x, y]), cloth_mask, keypoint_type=kp_idx
                            )
                            
                            if cloth_pt is not None:
                                correspondences.append((np.array([x, y]), cloth_pt))
            
            return correspondences
            
        except Exception as e:
            self.logger.warning(f"포즈 가이드 매칭 실패: {e}")
            return []

    def _find_corresponding_cloth_point(
        self, 
        body_point: np.ndarray, 
        cloth_mask: np.ndarray,
        keypoint_type: int
    ) -> Optional[np.ndarray]:
        """신체 키포인트에 대응하는 의류 점 찾기"""
        # 키포인트 타입에 따른 검색 전략
        if keypoint_type in [2, 5]:  # 어깨
            search_radius = 30
        elif keypoint_type in [3, 6]:  # 팔꿈치
            search_radius = 25
        elif keypoint_type in [4, 7]:  # 손목
            search_radius = 20
        else:  # 기타
            search_radius = 35
        
        # 검색 영역 설정
        x, y = int(body_point[0]), int(body_point[1])
        x_min = max(0, x - search_radius)
        x_max = min(cloth_mask.shape[1], x + search_radius)
        y_min = max(0, y - search_radius)
        y_max = min(cloth_mask.shape[0], y + search_radius)
        
        # 검색 영역에서 의류 픽셀 찾기
        search_region = cloth_mask[y_min:y_max, x_min:x_max]
        cloth_pixels = np.where(search_region > 0)
        
        if len(cloth_pixels[0]) == 0:
            return None
        
        # 가장 가까운 의류 픽셀 찾기
        cloth_coords = np.column_stack([
            cloth_pixels[1] + x_min,  # x 좌표
            cloth_pixels[0] + y_min   # y 좌표
        ])
        
        distances = np.linalg.norm(cloth_coords - body_point, axis=1)
        closest_idx = np.argmin(distances)
        
        return cloth_coords[closest_idx]

    async def _compute_tps_transform(self, matching_results: Dict[str, Any]) -> Dict[str, Any]:
        """TPS 변환 계산"""
        import time
        start_time = time.time()
        
        # 모든 대응점 수집
        all_correspondences = []
        
        # 특징점 매칭 결과
        all_correspondences.extend(matching_results["feature_matches"])
        
        # 윤곽선 매칭 결과 (가중치 조정)
        contour_matches = matching_results["contour_matches"]
        if len(contour_matches) > 10:  # 너무 많으면 샘플링
            step = len(contour_matches) // 10
            contour_matches = contour_matches[::step]
        all_correspondences.extend(contour_matches)
        
        # 포즈 가이드 매칭 결과 (높은 가중치)
        pose_matches = matching_results["pose_matches"]
        # 포즈 매칭은 3번 추가하여 가중치 증가
        all_correspondences.extend(pose_matches * 3)
        
        if len(all_correspondences) < self.min_matches:
            raise ValueError(f"충분한 대응점을 찾을 수 없습니다: {len(all_correspondences)} < {self.min_matches}")
        
        # 대응점을 배열로 변환
        source_points = np.array([pt[1] for pt in all_correspondences])  # 의류 점들
        target_points = np.array([pt[0] for pt in all_correspondences])  # 신체 점들
        
        # TPS 변환 학습
        success = self.tps_transform.fit(source_points, target_points)
        
        if not success:
            raise ValueError("TPS 변환 계산에 실패했습니다")
        
        processing_time = time.time() - start_time
        
        return {
            "transform": self.tps_transform,
            "correspondences": all_correspondences,
            "matrix": {
                "weights": self.tps_transform.weights.tolist() if self.tps_transform.weights is not None else None,
                "affine_params": self.tps_transform.affine_params.tolist() if self.tps_transform.affine_params is not None else None,
                "source_points": source_points.tolist(),
                "target_points": target_points.tolist()
            },
            "processing_time": processing_time
        }

    async def _evaluate_transform_quality(
        self, 
        tps_result: Dict[str, Any], 
        matching_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """변환 품질 평가"""
        try:
            tps_transform = tps_result["transform"]
            correspondences = tps_result["correspondences"]
            
            if len(correspondences) == 0:
                return {"error": "대응점이 없습니다"}
            
            # 변환 정확도 평가
            source_points = np.array([pt[1] for pt in correspondences])
            target_points = np.array([pt[0] for pt in correspondences])
            
            # 변환 적용
            transformed_points = tps_transform.transform(source_points)
            
            # 평균 제곱근 오차 (RMSE)
            errors = np.linalg.norm(transformed_points - target_points, axis=1)
            rmse = np.sqrt(np.mean(errors ** 2))
            
            # 최대 오차
            max_error = np.max(errors)
            
            # 중앙값 오차
            median_error = np.median(errors)
            
            # 변환 안정성 (조건수 기반)
            try:
                condition_number = np.linalg.cond(tps_transform.weights) if tps_transform.weights is not None else float('inf')
            except:
                condition_number = float('inf')
            
            # 커버리지 평가 (대응점 분포)
            coverage_score = self._evaluate_point_coverage(source_points, target_points)
            
            # 전체 품질 점수 계산
            quality_score = self._calculate_overall_quality(rmse, max_error, condition_number, coverage_score)
            
            return {
                "rmse": float(rmse),
                "max_error": float(max_error),
                "median_error": float(median_error),
                "condition_number": float(condition_number),
                "coverage_score": float(coverage_score),
                "quality_score": float(quality_score),
                "num_correspondences": len(correspondences)
            }
            
        except Exception as e:
            self.logger.error(f"변환 품질 평가 실패: {e}")
            return {"error": str(e)}

    def _evaluate_point_coverage(self, source_points: np.ndarray, target_points: np.ndarray) -> float:
        """대응점 분포 커버리지 평가"""
        if len(source_points) < 4:
            return 0.0
        
        # 각 점집합의 볼록 껍질 계산
        try:
            from scipy.spatial import ConvexHull
            
            source_hull = ConvexHull(source_points)
            target_hull = ConvexHull(target_points)
            
            # 볼록 껍질 면적 비율
            source_area = source_hull.volume  # 2D에서는 면적
            target_area = target_hull.volume
            
            if target_area > 0:
                area_ratio = min(source_area / target_area, target_area / source_area)
            else:
                area_ratio = 0.0
            
            # 점들의 분포 균등성 평가
            source_center = np.mean(source_points, axis=0)
            target_center = np.mean(target_points, axis=0)
            
            source_distances = np.linalg.norm(source_points - source_center, axis=1)
            target_distances = np.linalg.norm(target_points - target_center, axis=1)
            
            # 거리 분포의 유사성
            distance_similarity = 1.0 - abs(np.std(source_distances) - np.std(target_distances)) / max(np.std(source_distances), np.std(target_distances), 1.0)
            
            coverage_score = (area_ratio + distance_similarity) / 2.0
            
            return min(1.0, max(0.0, coverage_score))
            
        except Exception as e:
            self.logger.warning(f"커버리지 평가 실패: {e}")
            return 0.5  # 기본값

    def _calculate_overall_quality(
        self, 
        rmse: float, 
        max_error: float, 
        condition_number: float, 
        coverage_score: float
    ) -> float:
        """전체 품질 점수 계산"""
        # RMSE 점수 (낮을수록 좋음)
        rmse_score = max(0, 1.0 - rmse / 50.0)  # 50픽셀을 최대 오차로 가정
        
        # 최대 오차 점수
        max_error_score = max(0, 1.0 - max_error / 100.0)  # 100픽셀을 최대 오차로 가정
        
        # 조건수 점수 (안정성)
        stability_score = 1.0 / (1.0 + condition_number / 1000.0) if condition_number != float('inf') else 0.0
        
        # 가중 평균
        overall_quality = (
            rmse_score * 0.4 +
            max_error_score * 0.3 +
            stability_score * 0.1 +
            coverage_score * 0.2
        )
        
        return min(1.0, max(0.0, overall_quality))

    def visualize_matching(
        self, 
        person_image: np.ndarray, 
        cloth_image: np.ndarray,
        matching_result: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """매칭 결과 시각화"""
        
        # 이미지 크기 맞추기
        h1, w1 = person_image.shape[:2]
        h2, w2 = cloth_image.shape[:2]
        
        # 높이를 맞춰서 수평으로 연결
        target_height = max(h1, h2)
        
        person_resized = cv2.resize(person_image, (int(w1 * target_height / h1), target_height))
        cloth_resized = cv2.resize(cloth_image, (int(w2 * target_height / h2), target_height))
        
        # 이미지 연결
        combined_image = np.hstack([person_resized, cloth_resized])
        
        # 대응점 그리기
        correspondences = matching_result["correspondence_points"]
        
        for i, (person_pt, cloth_pt) in enumerate(correspondences):
            # 좌표 스케일링
            person_pt_scaled = person_pt * target_height / h1
            cloth_pt_scaled = cloth_pt * target_height / h2
            cloth_pt_scaled[0] += person_resized.shape[1]  # x 오프셋
            
            # 점 그리기
            cv2.circle(combined_image, tuple(person_pt_scaled.astype(int)), 3, (0, 255, 0), -1)
            cv2.circle(combined_image, tuple(cloth_pt_scaled.astype(int)), 3, (255, 0, 0), -1)
            
            # 연결선 그리기
            cv2.line(combined_image, 
                    tuple(person_pt_scaled.astype(int)), 
                    tuple(cloth_pt_scaled.astype(int)), 
                    (255, 255, 0), 1)
            
            # 번호 표시
            cv2.putText(combined_image, str(i), 
                       tuple((person_pt_scaled + 5).astype(int)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # 품질 정보 표시
        quality_metrics = matching_result["quality_metrics"]
        info_text = [
            f"Correspondences: {len(correspondences)}",
            f"RMSE: {quality_metrics.get('rmse', 0):.2f}",
            f"Quality: {quality_metrics.get('quality_score', 0):.3f}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(combined_image, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 저장 (옵션)
        if save_path:
            cv2.imwrite(save_path, combined_image)
        
        return combined_image

# 사용 예시
async def example_usage():
    """기하학적 매칭 사용 예시"""
    
    # 설정
    class Config:
        image_size = 512
        use_fp16 = True
    
    config = Config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 기하학적 매칭 단계 초기화
    geometric_matching = GeometricMatchingStep(config, device)
    
    # 더미 입력 데이터 생성
    dummy_person = torch.randn(1, 3, 512, 512).to(device)
    dummy_cloth = torch.randn(1, 3, 512, 512).to(device)
    
    input_data = {
        "person_image": dummy_person,
        "cloth_image": dummy_cloth,
        "human_parsing": {
            "parsing_map": np.random.randint(0, 20, (512, 512))
        },
        "pose_keypoints": {
            "keypoints": [(100, 100, 0.9), (150, 120, 0.8), (200, 140, 0.7)]
        },
        "cloth_mask": torch.ones(1, 1, 512, 512).to(device)
    }
    
    # 처리
    result = await geometric_matching.process(input_data)
    
    print(f"기하학적 매칭 완료")
    print(f"대응점 수: {result['metadata']['total_correspondences']}")
    print(f"처리 시간: {result['metadata']['processing_time']:.2f}초")
    
    # 품질 평가
    quality = result["quality_metrics"]
    print("변환 품질:")
    for metric, value in quality.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")

if __name__ == "__main__":
    asyncio.run(example_usage())