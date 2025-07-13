"""
MyCloset AI 5단계: 옷 워핑 (Cloth Warping)
TPS 변환 기반 의류 신체 형태 적응 시스템
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata
import logging
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import math

class AdvancedTPSWarper:
    """고급 TPS 워핑 클래스"""
    
    def __init__(self):
        self.grid_resolution = 50
        self.smoothing_factor = 0.1
        
    def create_dense_flow_field(self, tps_transform, image_shape: Tuple[int, int]) -> np.ndarray:
        """조밀한 플로우 필드 생성"""
        h, w = image_shape
        
        # 그리드 생성
        y_coords, x_coords = np.mgrid[0:h:self.grid_resolution, 0:w:self.grid_resolution]
        grid_points = np.column_stack([x_coords.ravel(), y_coords.ravel()])
        
        # TPS 변환 적용
        transformed_points = tps_transform.transform(grid_points)
        
        # 플로우 벡터 계산
        flow_vectors = transformed_points - grid_points
        
        # 전체 이미지로 보간
        y_full, x_full = np.mgrid[0:h, 0:w]
        
        flow_x = griddata(
            grid_points, flow_vectors[:, 0], 
            (x_full, y_full), method='cubic', fill_value=0
        )
        flow_y = griddata(
            grid_points, flow_vectors[:, 1], 
            (x_full, y_full), method='cubic', fill_value=0
        )
        
        # 플로우 필드 결합
        flow_field = np.stack([flow_x, flow_y], axis=2)
        
        return flow_field
    
    def apply_edge_preserving_smoothing(self, flow_field: np.ndarray, image: np.ndarray) -> np.ndarray:
        """엣지 보존 플로우 스무딩"""
        # 이미지 그래디언트 계산
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 엣지 강도에 따른 스무딩 가중치
        edge_weight = 1.0 / (1.0 + gradient_magnitude / 100.0)
        
        # 각 채널별로 스무딩 적용
        smoothed_flow = np.zeros_like(flow_field)
        for c in range(flow_field.shape[2]):
            # 가중치 적용된 가우시안 스무딩
            smoothed = cv2.GaussianBlur(flow_field[:, :, c], (5, 5), 1.0)
            smoothed_flow[:, :, c] = flow_field[:, :, c] * edge_weight + smoothed * (1 - edge_weight)
        
        return smoothed_flow

class SeamCarver:
    """시임 카빙 기반 적응적 리사이징"""
    
    def __init__(self):
        pass
    
    def calculate_energy_map(self, image: np.ndarray) -> np.ndarray:
        """에너지 맵 계산"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 그래디언트 기반 에너지
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        energy = np.abs(grad_x) + np.abs(grad_y)
        
        return energy
    
    def find_vertical_seam(self, energy_map: np.ndarray) -> np.ndarray:
        """수직 시임 찾기"""
        h, w = energy_map.shape
        dp = energy_map.copy()
        
        # 동적 프로그래밍으로 최소 에너지 경로 찾기
        for i in range(1, h):
            for j in range(w):
                # 이전 행에서 올 수 있는 위치들
                candidates = []
                if j > 0:
                    candidates.append(dp[i-1, j-1])
                candidates.append(dp[i-1, j])
                if j < w - 1:
                    candidates.append(dp[i-1, j+1])
                
                dp[i, j] += min(candidates)
        
        # 역추적으로 시임 경로 찾기
        seam = np.zeros(h, dtype=int)
        seam[-1] = np.argmin(dp[-1])
        
        for i in range(h-2, -1, -1):
            prev_j = seam[i+1]
            candidates = []
            indices = []
            
            if prev_j > 0:
                candidates.append(dp[i, prev_j-1])
                indices.append(prev_j-1)
            candidates.append(dp[i, prev_j])
            indices.append(prev_j)
            if prev_j < w - 1:
                candidates.append(dp[i, prev_j+1])
                indices.append(prev_j+1)
            
            seam[i] = indices[np.argmin(candidates)]
        
        return seam
    
    def remove_vertical_seam(self, image: np.ndarray, seam: np.ndarray) -> np.ndarray:
        """수직 시임 제거"""
        h, w = image.shape[:2]
        if len(image.shape) == 3:
            new_image = np.zeros((h, w-1, image.shape[2]), dtype=image.dtype)
            for i in range(h):
                j = seam[i]
                new_image[i, :j] = image[i, :j]
                new_image[i, j:] = image[i, j+1:]
        else:
            new_image = np.zeros((h, w-1), dtype=image.dtype)
            for i in range(h):
                j = seam[i]
                new_image[i, :j] = image[i, :j]
                new_image[i, j:] = image[i, j+1:]
        
        return new_image
    
    def adaptive_resize(self, image: np.ndarray, target_width: int) -> np.ndarray:
        """적응적 리사이징"""
        current_width = image.shape[1]
        
        if current_width == target_width:
            return image
        
        result_image = image.copy()
        
        if current_width > target_width:
            # 시임 제거
            num_seams = current_width - target_width
            for _ in range(num_seams):
                energy_map = self.calculate_energy_map(result_image)
                seam = self.find_vertical_seam(energy_map)
                result_image = self.remove_vertical_seam(result_image, seam)
        else:
            # 단순 리사이징 (확대는 복잡한 시임 삽입 대신)
            result_image = cv2.resize(result_image, (target_width, image.shape[0]))
        
        return result_image

class ClothWarpingStep:
    """5단계: 옷 워핑 실행 클래스"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # 워핑 컴포넌트
        self.tps_warper = AdvancedTPSWarper()
        self.seam_carver = SeamCarver()
        
        # 워핑 파라미터
        self.interpolation_method = cv2.INTER_LINEAR
        self.boundary_mode = cv2.BORDER_REFLECT
        self.smoothing_sigma = 1.0
        self.edge_preservation_weight = 0.3
        self.quality_enhancement = True

    async def process(self, input_data: Dict[str, Any]) -> torch.Tensor:
        """옷 워핑 메인 처리"""
        try:
            # 입력 데이터 추출
            cloth_image = input_data["cloth_image"]
            cloth_mask = input_data["cloth_mask"]
            tps_transform = input_data["tps_transform"]
            target_shape = input_data["target_shape"]
            
            # 텐서를 numpy로 변환
            cloth_np = self._tensor_to_numpy(cloth_image)
            cloth_mask_np = self._tensor_to_numpy(cloth_mask).squeeze()
            
            # 목표 형태 처리
            if isinstance(target_shape, dict):
                target_mask = self._process_target_shape(target_shape, cloth_np.shape[:2])
            else:
                target_mask = self._tensor_to_numpy(target_shape).squeeze()
            
            # 멀티 스테이지 워핑 실행
            warping_result = await self._multi_stage_warping(
                cloth_np, cloth_mask_np, tps_transform, target_mask
            )
            
            # 품질 향상
            if self.quality_enhancement:
                enhanced_result = await self._enhance_warping_quality(
                    warping_result, cloth_np, cloth_mask_np
                )
            else:
                enhanced_result = warping_result
            
            # 텐서로 변환하여 반환
            result_tensor = self._numpy_to_tensor(enhanced_result["warped_cloth"])
            
            return result_tensor
            
        except Exception as e:
            self.logger.error(f"옷 워핑 처리 중 오류: {str(e)}")
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

    def _numpy_to_tensor(self, numpy_array: np.ndarray) -> torch.Tensor:
        """numpy 배열을 텐서로 변환"""
        # [0, 255] → [0, 1] 변환
        if numpy_array.max() > 1.0:
            numpy_array = numpy_array.astype(np.float32) / 255.0
        
        # 차원 조정
        if len(numpy_array.shape) == 3:  # [H, W, C]
            tensor = torch.from_numpy(numpy_array.transpose(2, 0, 1))  # [C, H, W]
        else:  # [H, W]
            tensor = torch.from_numpy(numpy_array).unsqueeze(0)  # [1, H, W]
        
        return tensor.unsqueeze(0).to(self.device)  # [1, C, H, W]

    def _process_target_shape(self, target_shape_data: Dict[str, Any], image_shape: Tuple[int, int]) -> np.ndarray:
        """목표 형태 데이터 처리"""
        if "parsing_map" in target_shape_data:
            parsing_map = target_shape_data["parsing_map"]
            if isinstance(parsing_map, np.ndarray):
                # 의류 관련 영역만 추출
                clothing_ids = [5, 6, 7, 9, 10, 12]  # 상의, 원피스, 외투, 하의, 점프수트, 스커트
                target_mask = np.zeros_like(parsing_map, dtype=np.uint8)
                for cloth_id in clothing_ids:
                    target_mask[parsing_map == cloth_id] = 1
                
                # 이미지 크기에 맞게 리사이즈
                if target_mask.shape != image_shape:
                    target_mask = cv2.resize(target_mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
                
                return target_mask.astype(np.float32)
        
        # 기본값: 전체 이미지
        return np.ones(image_shape, dtype=np.float32)

    async def _multi_stage_warping(
        self, 
        cloth_image: np.ndarray,
        cloth_mask: np.ndarray,
        tps_transform,
        target_mask: np.ndarray
    ) -> Dict[str, Any]:
        """멀티 스테이지 워핑"""
        import time
        start_time = time.time()
        
        # 1단계: 기본 TPS 워핑
        basic_warped = await self._basic_tps_warping(cloth_image, cloth_mask, tps_transform)
        
        # 2단계: 플로우 필드 기반 정제
        flow_refined = await self._flow_field_refinement(
            basic_warped["warped_cloth"], 
            basic_warped["warped_mask"],
            tps_transform,
            cloth_image.shape[:2]
        )
        
        # 3단계: 적응적 크기 조정
        size_adapted = await self._adaptive_size_adjustment(
            flow_refined["refined_cloth"],
            flow_refined["refined_mask"],
            target_mask
        )
        
        # 4단계: 경계 조화
        boundary_harmonized = await self._boundary_harmonization(
            size_adapted["adapted_cloth"],
            size_adapted["adapted_mask"],
            target_mask
        )
        
        processing_time = time.time() - start_time
        
        return {
            "warped_cloth": boundary_harmonized["harmonized_cloth"],
            "warped_mask": boundary_harmonized["harmonized_mask"],
            "warping_flow": flow_refined["flow_field"],
            "quality_metrics": {
                "warping_accuracy": boundary_harmonized["accuracy"],
                "boundary_smoothness": boundary_harmonized["smoothness"],
                "size_adaptation": size_adapted["adaptation_score"]
            },
            "processing_time": processing_time
        }

    async def _basic_tps_warping(
        self, 
        cloth_image: np.ndarray,
        cloth_mask: np.ndarray,
        tps_transform
    ) -> Dict[str, Any]:
        """기본 TPS 워핑"""
        try:
            # TPS 변환으로 이미지 워핑
            warped_cloth = tps_transform.transform_image(cloth_image, cloth_image.shape[:2])
            
            # 마스크도 동일하게 워핑
            if len(cloth_mask.shape) == 2:
                mask_3ch = np.stack([cloth_mask, cloth_mask, cloth_mask], axis=2)
            else:
                mask_3ch = cloth_mask
            
            warped_mask_3ch = tps_transform.transform_image(mask_3ch, cloth_image.shape[:2])
            warped_mask = warped_mask_3ch[:, :, 0] if len(warped_mask_3ch.shape) == 3 else warped_mask_3ch
            
            # 이진화
            warped_mask = (warped_mask > 0.5).astype(np.float32)
            
            return {
                "warped_cloth": warped_cloth,
                "warped_mask": warped_mask
            }
            
        except Exception as e:
            self.logger.warning(f"기본 TPS 워핑 실패: {e}")
            # 원본 반환
            return {
                "warped_cloth": cloth_image,
                "warped_mask": cloth_mask
            }

    async def _flow_field_refinement(
        self, 
        warped_cloth: np.ndarray,
        warped_mask: np.ndarray,
        tps_transform,
        original_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """플로우 필드 기반 정제"""
        try:
            # 조밀한 플로우 필드 생성
            flow_field = self.tps_warper.create_dense_flow_field(tps_transform, original_shape)
            
            # 엣지 보존 스무딩
            smoothed_flow = self.tps_warper.apply_edge_preserving_smoothing(flow_field, warped_cloth)
            
            # 스무딩된 플로우로 재워핑
            h, w = original_shape
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            
            # 새로운 좌표 계산
            new_x = x_coords + smoothed_flow[:, :, 0]
            new_y = y_coords + smoothed_flow[:, :, 1]
            
            # 경계 처리
            new_x = np.clip(new_x, 0, w - 1)
            new_y = np.clip(new_y, 0, h - 1)
            
            # 재매핑
            refined_cloth = cv2.remap(
                warped_cloth, 
                new_x.astype(np.float32), 
                new_y.astype(np.float32),
                self.interpolation_method,
                borderMode=self.boundary_mode
            )
            
            refined_mask = cv2.remap(
                warped_mask, 
                new_x.astype(np.float32), 
                new_y.astype(np.float32),
                self.interpolation_method,
                borderMode=self.boundary_mode
            )
            
            return {
                "refined_cloth": refined_cloth,
                "refined_mask": refined_mask,
                "flow_field": smoothed_flow
            }
            
        except Exception as e:
            self.logger.warning(f"플로우 필드 정제 실패: {e}")
            return {
                "refined_cloth": warped_cloth,
                "refined_mask": warped_mask,
                "flow_field": np.zeros((original_shape[0], original_shape[1], 2))
            }

    async def _adaptive_size_adjustment(
        self, 
        cloth_image: np.ndarray,
        cloth_mask: np.ndarray,
        target_mask: np.ndarray
    ) -> Dict[str, Any]:
        """적응적 크기 조정"""
        try:
            # 목표 영역의 바운딩 박스 계산
            target_coords = np.where(target_mask > 0.5)
            
            if len(target_coords[0]) == 0:
                return {
                    "adapted_cloth": cloth_image,
                    "adapted_mask": cloth_mask,
                    "adaptation_score": 0.0
                }
            
            target_y_min, target_y_max = np.min(target_coords[0]), np.max(target_coords[0])
            target_x_min, target_x_max = np.min(target_coords[1]), np.max(target_coords[1])
            target_width = target_x_max - target_x_min + 1
            target_height = target_y_max - target_y_min + 1
            
            # 현재 의류 영역의 바운딩 박스
            cloth_coords = np.where(cloth_mask > 0.5)
            
            if len(cloth_coords[0]) == 0:
                return {
                    "adapted_cloth": cloth_image,
                    "adapted_mask": cloth_mask,
                    "adaptation_score": 0.0
                }
            
            cloth_y_min, cloth_y_max = np.min(cloth_coords[0]), np.max(cloth_coords[0])
            cloth_x_min, cloth_x_max = np.min(cloth_coords[1]), np.max(cloth_coords[1])
            cloth_width = cloth_x_max - cloth_x_min + 1
            cloth_height = cloth_y_max - cloth_y_min + 1
            
            # 크기 비율 계산
            scale_x = target_width / cloth_width
            scale_y = target_height / cloth_height
            
            # 종횡비 보존을 위한 스케일 조정
            scale = min(scale_x, scale_y)
            
            # 스케일링이 필요한 경우
            if abs(scale - 1.0) > 0.1:  # 10% 이상 차이날 때만
                new_width = int(cloth_image.shape[1] * scale)
                new_height = int(cloth_image.shape[0] * scale)
                
                # 시임 카빙 기반 적응적 리사이징 (가로 방향)
                if abs(scale_x - 1.0) > 0.2:  # 20% 이상 차이
                    adapted_cloth = self.seam_carver.adaptive_resize(cloth_image, new_width)
                    adapted_mask = self.seam_carver.adaptive_resize(
                        (cloth_mask * 255).astype(np.uint8), new_width
                    ).astype(np.float32) / 255.0
                else:
                    adapted_cloth = cv2.resize(cloth_image, (new_width, new_height))
                    adapted_mask = cv2.resize(cloth_mask, (new_width, new_height))
                
                # 원본 크기로 패딩 또는 크롭
                adapted_cloth = self._resize_with_padding(adapted_cloth, cloth_image.shape[:2])
                adapted_mask = self._resize_with_padding(adapted_mask, cloth_mask.shape)
                
                adaptation_score = 1.0 - abs(scale - 1.0)
            else:
                adapted_cloth = cloth_image
                adapted_mask = cloth_mask
                adaptation_score = 1.0
            
            return {
                "adapted_cloth": adapted_cloth,
                "adapted_mask": adapted_mask,
                "adaptation_score": adaptation_score
            }
            
        except Exception as e:
            self.logger.warning(f"크기 조정 실패: {e}")
            return {
                "adapted_cloth": cloth_image,
                "adapted_mask": cloth_mask,
                "adaptation_score": 0.5
            }

    def _resize_with_padding(self, image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """패딩을 사용한 리사이징"""
        target_h, target_w = target_shape
        current_h, current_w = image.shape[:2]
        
        if current_h == target_h and current_w == target_w:
            return image
        
        # 패딩 또는 크롭 계산
        if current_h < target_h or current_w < target_w:
            # 패딩 필요
            pad_h = max(0, target_h - current_h)
            pad_w = max(0, target_w - current_w)
            
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            if len(image.shape) == 3:
                padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='reflect')
            else:
                padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
            
            return padded
        else:
            # 크롭 필요
            crop_h = (current_h - target_h) // 2
            crop_w = (current_w - target_w) // 2
            
            return image[crop_h:crop_h + target_h, crop_w:crop_w + target_w]

    async def _boundary_harmonization(
        self, 
        cloth_image: np.ndarray,
        cloth_mask: np.ndarray,
        target_mask: np.ndarray
    ) -> Dict[str, Any]:
        """경계 조화"""
        try:
            # 마스크 경계 스무딩
            smoothed_mask = cv2.GaussianBlur(cloth_mask, (5, 5), self.smoothing_sigma)
            
            # 목표 마스크와의 교집합
            combined_mask = np.minimum(smoothed_mask, target_mask)
            
            # 경계 조화된 이미지 생성
            harmonized_cloth = cloth_image.copy()
            
            # 마스크 밖 영역을 부드럽게 페이드 아웃
            fade_mask = cv2.GaussianBlur(combined_mask, (11, 11), 2.0)
            
            if len(harmonized_cloth.shape) == 3:
                fade_mask_3ch = np.stack([fade_mask, fade_mask, fade_mask], axis=2)
                harmonized_cloth = harmonized_cloth * fade_mask_3ch
            else:
                harmonized_cloth = harmonized_cloth * fade_mask
            
            # 품질 메트릭 계산
            accuracy = np.sum(combined_mask) / max(np.sum(cloth_mask), 1)
            
            # 경계 매끄러움 평가
            boundary_gradient = np.gradient(combined_mask)
            smoothness = 1.0 / (1.0 + np.mean(np.abs(boundary_gradient)))
            
            return {
                "harmonized_cloth": harmonized_cloth,
                "harmonized_mask": combined_mask,
                "accuracy": float(accuracy),
                "smoothness": float(smoothness)
            }
            
        except Exception as e:
            self.logger.warning(f"경계 조화 실패: {e}")
            return {
                "harmonized_cloth": cloth_image,
                "harmonized_mask": cloth_mask,
                "accuracy": 0.5,
                "smoothness": 0.5
            }

    async def _enhance_warping_quality(
        self, 
        warping_result: Dict[str, Any],
        original_cloth: np.ndarray,
        original_mask: np.ndarray
    ) -> Dict[str, Any]:
        """워핑 품질 향상"""
        try:
            warped_cloth = warping_result["warped_cloth"]
            warped_mask = warping_result["warped_mask"]
            
            # 1. 색상 보정
            color_corrected = self._apply_color_correction(warped_cloth, original_cloth, warped_mask)
            
            # 2. 디테일 보존
            detail_enhanced = self._preserve_clothing_details(
                color_corrected, warped_cloth, warped_mask
            )
            
            # 3. 아티팩트 제거
            artifact_removed = self._remove_warping_artifacts(detail_enhanced, warped_mask)
            
            # 4. 최종 샤프닝
            final_enhanced = self._apply_adaptive_sharpening(artifact_removed, warped_mask)
            
            # 향상된 결과로 업데이트
            enhanced_result = warping_result.copy()
            enhanced_result["warped_cloth"] = final_enhanced
            
            return enhanced_result
            
        except Exception as e:
            self.logger.warning(f"품질 향상 실패: {e}")
            return warping_result

    def _apply_color_correction(
        self, 
        warped_cloth: np.ndarray, 
        original_cloth: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """색상 보정"""
        try:
            if len(warped_cloth.shape) != 3:
                return warped_cloth
            
            # 마스크 영역에서 색상 히스토그램 매칭
            corrected = warped_cloth.copy()
            
            for c in range(3):  # RGB 채널별로
                original_channel = original_cloth[:, :, c][mask > 0.5]
                warped_channel = warped_cloth[:, :, c][mask > 0.5]
                
                if len(original_channel) > 0 and len(warped_channel) > 0:
                    # 히스토그램 매칭
                    corrected_channel = self._match_histogram(warped_channel, original_channel)
                    corrected[:, :, c][mask > 0.5] = corrected_channel
            
            return corrected
            
        except Exception as e:
            self.logger.warning(f"색상 보정 실패: {e}")
            return warped_cloth

    def _match_histogram(self, source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """히스토그램 매칭"""
        # 히스토그램 계산
        source_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
        ref_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])
        
        # 누적 분포 함수 계산
        source_cdf = source_hist.cumsum()
        ref_cdf = ref_hist.cumsum()
        
        # 정규화
        source_cdf = source_cdf / source_cdf[-1]
        ref_cdf = ref_cdf / ref_cdf[-1]
        
        # 매핑 테이블 생성
        mapping = np.zeros(256)
        for i in range(256):
            diff = np.abs(source_cdf[i] - ref_cdf)
            mapping[i] = np.argmin(diff)
        
        # 매핑 적용
        matched = mapping[source.astype(int)]
        
        return matched.astype(source.dtype)

    def _preserve_clothing_details(
        self, 
        warped_cloth: np.ndarray, 
        original_warped: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """의류 디테일 보존"""
        try:
            # 고주파 디테일 추출
            blurred = cv2.GaussianBlur(original_warped, (3, 3), 1.0)
            high_freq = original_warped.astype(np.float32) - blurred.astype(np.float32)
            
            # 디테일 강화
            detail_weight = 0.3
            enhanced = warped_cloth.astype(np.float32) + high_freq * detail_weight
            
            # 마스크 영역에만 적용
            if len(enhanced.shape) == 3:
                mask_3ch = np.stack([mask, mask, mask], axis=2)
                result = warped_cloth * (1 - mask_3ch) + enhanced * mask_3ch
            else:
                result = warped_cloth * (1 - mask) + enhanced * mask
            
            return np.clip(result, 0, 255).astype(np.uint8)
            
        except Exception as e:
            self.logger.warning(f"디테일 보존 실패: {e}")
            return warped_cloth

    def _remove_warping_artifacts(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """워핑 아티팩트 제거"""
        try:
            # 미디언 필터로 점 노이즈 제거
            denoised = cv2.medianBlur(image, 3)
            
            # 마스크 영역에만 적용
            if len(image.shape) == 3:
                mask_3ch = np.stack([mask, mask, mask], axis=2)
                result = image * (1 - mask_3ch) + denoised * mask_3ch
            else:
                result = image * (1 - mask) + denoised * mask
            
            return result.astype(np.uint8)
            
        except Exception as e:
            self.logger.warning(f"아티팩트 제거 실패: {e}")
            return image

    def _apply_adaptive_sharpening(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """적응적 샤프닝"""
        try:
            # 언샤프 마스킹
            blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
            sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
            
            # 마스크 영역에만 적용
            if len(image.shape) == 3:
                mask_3ch = np.stack([mask, mask, mask], axis=2)
                result = image * (1 - mask_3ch * 0.3) + sharpened * (mask_3ch * 0.3)
            else:
                result = image * (1 - mask * 0.3) + sharpened * (mask * 0.3)
            
            return np.clip(result, 0, 255).astype(np.uint8)
            
        except Exception as e:
            self.logger.warning(f"샤프닝 실패: {e}")
            return image

    def visualize_warping_result(
        self, 
        original_cloth: np.ndarray, 
        warped_cloth: np.ndarray,
        flow_field: np.ndarray,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """워핑 결과 시각화"""
        
        # 이미지 크기 맞추기
        h, w = original_cloth.shape[:2]
        
        # 플로우 필드 시각화
        flow_magnitude = np.sqrt(flow_field[:, :, 0]**2 + flow_field[:, :, 1]**2)
        flow_normalized = (flow_magnitude / np.max(flow_magnitude) * 255).astype(np.uint8)
        flow_colored = cv2.applyColorMap(flow_normalized, cv2.COLORMAP_JET)
        
        # 결과 조합
        combined = np.hstack([
            cv2.resize(original_cloth, (w//2, h//2)),
            cv2.resize(warped_cloth, (w//2, h//2))
        ])
        
        flow_resized = cv2.resize(flow_colored, (w, h//2))
        
        final_result = np.vstack([combined, flow_resized])
        
        # 라벨 추가
        cv2.putText(final_result, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(final_result, "Warped", (w//2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(final_result, "Flow Field", (10, h//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 저장 (옵션)
        if save_path:
            cv2.imwrite(save_path, final_result)
        
        return final_result

# 사용 예시
async def example_usage():
    """옷 워핑 사용 예시"""
    
    # 설정
    class Config:
        image_size = 512
        use_fp16 = True
    
    config = Config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 옷 워핑 단계 초기화
    cloth_warping = ClothWarpingStep(config, device)
    
    # 더미 TPS 변환 객체 (실제로는 4단계에서 받음)
    class DummyTPS:
        def transform_image(self, image, shape):
            # 간단한 더미 변환 (실제로는 복잡한 TPS 변환)
            return cv2.resize(image, (shape[1], shape[0]))
    
    dummy_tps = DummyTPS()
    
    # 더미 입력 데이터
    dummy_cloth = torch.randn(1, 3, 512, 512).to(device)
    dummy_mask = torch.ones(1, 1, 512, 512).to(device)
    
    input_data = {
        "cloth_image": dummy_cloth,
        "cloth_mask": dummy_mask,
        "tps_transform": dummy_tps,
        "target_shape": {
            "parsing_map": np.random.randint(0, 20, (512, 512))
        }
    }
    
    # 처리
    result = await cloth_warping.process(input_data)
    
    print(f"옷 워핑 완료 - 출력 크기: {result.shape}")
    print("워핑이 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    asyncio.run(example_usage())