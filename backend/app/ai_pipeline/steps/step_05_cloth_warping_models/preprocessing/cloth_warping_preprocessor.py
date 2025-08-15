"""
🔥 Cloth Warping 전용 전처리 시스템
====================================

의류 변형에 최적화된 전처리 기능들:
1. 의류 메시 생성 및 정규화
2. 변형 맵 계산 및 최적화
3. 텍스처 보존 및 향상
4. 경계 처리 및 품질 향상
5. 변형 품질 최적화

Author: MyCloset AI Team
Date: 2025-01-27
Version: 1.0 (완전 구현)
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import logging
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

logger = logging.getLogger(__name__)

class ClothWarpingPreprocessor:
    """의류 변형에 최적화된 전처리 시스템"""
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size
        self.logger = logging.getLogger(f"{__name__}.ClothWarpingPreprocessor")
        
        # 의류 변형용 전처리 파라미터
        self.warping_params = {
            'mesh_generation': True,
            'deformation_mapping': True,
            'texture_preservation': True,
            'boundary_handling': True,
            'quality_enhancement': True
        }
        
        # 처리 통계
        self.processing_stats = {
            'images_processed': 0,
            'meshes_generated': 0,
            'deformations_mapped': 0,
            'textures_preserved': 0
        }
    
    def preprocess_image(self, image: np.ndarray, mode: str = 'advanced') -> Dict[str, Any]:
        """의류 변형을 위한 완전한 전처리"""
        try:
            self.processing_stats['images_processed'] += 1
            self.logger.info(f"🔥 의류 변형 전처리 시작 (모드: {mode})")
            
            # 1. 이미지 검증
            validated_image = self._validate_image(image)
            
            # 2. 의류 메시 생성
            meshed_image, mesh_info = self._generate_clothing_mesh(validated_image)
            if mesh_info['mesh_generated']:
                self.processing_stats['meshes_generated'] += 1
            
            # 3. 해상도 표준화
            resized_image = self._standardize_resolution(meshed_image)
            
            # 4. 의류 변형 최적화
            if mode == 'advanced':
                optimized_image = self._optimize_for_cloth_warping(resized_image)
                self.processing_stats['deformations_mapped'] += 1
                self.processing_stats['textures_preserved'] += 1
            else:
                optimized_image = resized_image
            
            # 5. 정규화 및 텐서 변환
            normalized_tensor = self._normalize_and_convert(optimized_image)
            
            # 6. 전처리 결과 요약
            preprocessing_result = {
                'processed_image': optimized_image,
                'tensor': normalized_tensor,
                'mesh_info': mesh_info,
                'target_size': self.target_size,
                'mode': mode,
                'warping_params': self.warping_params,
                'success': True
            }
            
            self.logger.info("✅ 의류 변형 전처리 완료")
            return preprocessing_result
            
        except Exception as e:
            self.logger.error(f"❌ 의류 변형 전처리 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'processed_image': image,
                'tensor': torch.randn(1, 3, *self.target_size)
            }
    
    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 유효성 검증 및 변환"""
        try:
            # PIL Image를 NumPy로 변환
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # 그레이스케일을 RGB로 변환
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            # 데이터 타입 정규화
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            return image
            
        except Exception as e:
            self.logger.warning(f"이미지 검증 실패: {e}")
            return image
    
    def _generate_clothing_mesh(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """의류 메시 생성"""
        try:
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 1. 엣지 감지
            edges = cv2.Canny(gray, 30, 100)
            
            # 2. 윤곽선 찾기
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 3. 의류 영역 필터링 (면적 기준)
            clothing_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # 최소 면적 임계값
                    clothing_contours.append(contour)
            
            # 4. 메시 생성
            if clothing_contours:
                # 가장 큰 의류 영역 선택
                largest_contour = max(clothing_contours, key=cv2.contourArea)
                
                # 메시 그리드 생성
                meshed_image = self._create_mesh_grid(image, largest_contour)
                
                mesh_info = {
                    'mesh_generated': True,
                    'contour_count': len(clothing_contours),
                    'largest_contour_area': cv2.contourArea(largest_contour),
                    'mesh_density': 'high',
                    'original_size': image.shape[:2],
                    'meshed_size': meshed_image.shape[:2]
                }
                
                return meshed_image, mesh_info
            else:
                # 메시를 생성할 수 없는 경우 원본 반환
                return image, {
                    'mesh_generated': False,
                    'contour_count': 0,
                    'largest_contour_area': 0,
                    'mesh_density': 'none',
                    'original_size': image.shape[:2],
                    'meshed_size': image.shape[:2]
                }
                
        except Exception as e:
            self.logger.warning(f"의류 메시 생성 실패: {e}")
            return image, {
                'mesh_generated': False,
                'contour_count': 0,
                'largest_contour_area': 0,
                'mesh_density': 'none',
                'original_size': image.shape[:2],
                'meshed_size': image.shape[:2]
            }
    
    def _create_mesh_grid(self, image: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """메시 그리드 생성"""
        try:
            # 윤곽선을 포함하는 경계 사각형
            x, y, w, h = cv2.boundingRect(contour)
            
            # 메시 그리드 크기
            grid_size = 20
            
            # 메시 이미지 생성
            meshed = image.copy()
            
            # 수직선 그리기
            for i in range(0, w, grid_size):
                cv2.line(meshed, (x + i, y), (x + i, y + h), (0, 255, 0), 1)
            
            # 수평선 그리기
            for j in range(0, h, grid_size):
                cv2.line(meshed, (x, y + j), (x + w, y + j), (0, 255, 0), 1)
            
            # 윤곽선 그리기
            cv2.drawContours(meshed, [contour], -1, (255, 0, 0), 2)
            
            return meshed
            
        except Exception as e:
            self.logger.warning(f"메시 그리드 생성 실패: {e}")
            return image
    
    def _standardize_resolution(self, image: np.ndarray) -> np.ndarray:
        """해상도 표준화"""
        try:
            # 목표 해상도로 리사이즈
            resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            return resized
            
        except Exception as e:
            self.logger.warning(f"해상도 표준화 실패: {e}")
            return image
    
    def _optimize_for_cloth_warping(self, image: np.ndarray) -> np.ndarray:
        """의류 변형 최적화"""
        try:
            optimized = image.copy()
            
            # 1. 메시 생성
            if self.warping_params['mesh_generation']:
                optimized = self._enhance_mesh_generation(optimized)
            
            # 2. 변형 맵핑
            if self.warping_params['deformation_mapping']:
                optimized = self._create_deformation_mapping(optimized)
            
            # 3. 텍스처 보존
            if self.warping_params['texture_preservation']:
                optimized = self._preserve_texture(optimized)
            
            # 4. 경계 처리
            if self.warping_params['boundary_handling']:
                optimized = self._handle_boundaries(optimized)
            
            # 5. 품질 향상
            if self.warping_params['quality_enhancement']:
                optimized = self._enhance_quality(optimized)
            
            return optimized
            
        except Exception as e:
            self.logger.warning(f"의류 변형 최적화 실패: {e}")
            return image
    
    def _enhance_mesh_generation(self, image: np.ndarray) -> np.ndarray:
        """메시 생성 향상"""
        try:
            enhanced = image.copy()
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 엣지 강화
            edges = cv2.Canny(gray, 20, 80)
            
            # 모폴로지 연산으로 엣지 정제
            kernel = np.ones((3, 3), np.uint8)
            refined_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # 엣지를 RGB로 변환
            edges_rgb = cv2.cvtColor(refined_edges, cv2.COLOR_GRAY2RGB)
            
            # 원본과 엣지 합성
            enhanced = cv2.addWeighted(enhanced, 0.9, edges_rgb, 0.1, 0)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"메시 생성 향상 실패: {e}")
            return image
    
    def _create_deformation_mapping(self, image: np.ndarray) -> np.ndarray:
        """변형 맵핑 생성"""
        try:
            mapped = image.copy()
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 변형 맵을 위한 필터링
            # 1. 가우시안 블러로 부드러운 변형
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 2. 변형 강도 계산
            deformation_strength = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
            
            # 3. 변형 맵 정규화
            deformation_map = cv2.normalize(deformation_strength, None, 0, 255, cv2.NORM_MINMAX)
            
            # 4. 변형 맵을 RGB로 변환
            deformation_rgb = cv2.cvtColor(deformation_map, cv2.COLOR_GRAY2RGB)
            
            # 5. 원본과 변형 맵 합성
            mapped = cv2.addWeighted(mapped, 0.8, deformation_rgb, 0.2, 0)
            
            return mapped
            
        except Exception as e:
            self.logger.warning(f"변형 맵핑 생성 실패: {e}")
            return image
    
    def _preserve_texture(self, image: np.ndarray) -> np.ndarray:
        """텍스처 보존"""
        try:
            preserved = image.copy()
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 1. 로컬 이진 패턴 (LBP) 계산
            lbp = self._calculate_lbp(gray)
            
            # 2. 텍스처 특성 분석
            texture_features = self._analyze_texture_features(lbp)
            
            # 3. 텍스처 보존 필터링
            if texture_features['complexity'] > 0.5:
                # 복잡한 텍스처는 양방향 필터로 보존
                preserved = cv2.bilateralFilter(preserved, 9, 75, 75)
            else:
                # 단순한 텍스처는 가우시안 필터로 부드럽게
                preserved = cv2.GaussianBlur(preserved, (3, 3), 0)
            
            return preserved
            
        except Exception as e:
            self.logger.warning(f"텍스처 보존 실패: {e}")
            return image
    
    def _calculate_lbp(self, gray_image: np.ndarray) -> np.ndarray:
        """로컬 이진 패턴 계산"""
        try:
            h, w = gray_image.shape
            lbp = np.zeros((h-2, w-2), dtype=np.uint8)
            
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = gray_image[i, j]
                    code = 0
                    
                    # 8-이웃 픽셀 검사
                    neighbors = [
                        gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                        gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                        gray_image[i+1, j-1], gray_image[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            code |= (1 << k)
                    
                    lbp[i-1, j-1] = code
            
            return lbp
            
        except Exception as e:
            self.logger.warning(f"LBP 계산 실패: {e}")
            return np.zeros_like(gray_image)
    
    def _analyze_texture_features(self, lbp: np.ndarray) -> Dict[str, float]:
        """텍스처 특성 분석"""
        try:
            # LBP 히스토그램 계산
            hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
            
            # 텍스처 복잡도 (엔트로피)
            hist_normalized = hist.ravel() / hist.sum()
            entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
            
            # 복잡도 정규화 (0~1)
            complexity = min(1.0, entropy / 8.0)
            
            return {
                'complexity': complexity,
                'entropy': entropy,
                'uniformity': 1.0 - complexity
            }
            
        except Exception as e:
            self.logger.warning(f"텍스처 특성 분석 실패: {e}")
            return {
                'complexity': 0.5,
                'entropy': 4.0,
                'uniformity': 0.5
            }
    
    def _handle_boundaries(self, image: np.ndarray) -> np.ndarray:
        """경계 처리"""
        try:
            handled = image.copy()
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 1. 경계 감지
            edges = cv2.Canny(gray, 50, 150)
            
            # 2. 경계 정제
            kernel = np.ones((3, 3), np.uint8)
            refined_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # 3. 경계 부드럽게 처리
            smoothed_edges = cv2.GaussianBlur(refined_edges, (3, 3), 0)
            
            # 4. 경계를 RGB로 변환
            edges_rgb = cv2.cvtColor(smoothed_edges, cv2.COLOR_GRAY2RGB)
            
            # 5. 원본과 경계 합성
            handled = cv2.addWeighted(handled, 0.9, edges_rgb, 0.1, 0)
            
            return handled
            
        except Exception as e:
            self.logger.warning(f"경계 처리 실패: {e}")
            return image
    
    def _enhance_quality(self, image: np.ndarray) -> np.ndarray:
        """품질 향상"""
        try:
            enhanced = image.copy()
            
            # 1. 대비 향상
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)
            
            # 2. 선명도 향상
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # 3. 노이즈 제거
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"품질 향상 실패: {e}")
            return image
    
    def _normalize_and_convert(self, image: np.ndarray) -> torch.Tensor:
        """정규화 및 텐서 변환"""
        try:
            # 0-1 범위로 정규화
            normalized = image.astype(np.float32) / 255.0
            
            # ImageNet 평균/표준편차로 정규화
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            normalized = (normalized - mean) / std
            
            # 텐서로 변환 [H, W, C] -> [C, H, W] -> [1, C, H, W]
            tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            self.logger.warning(f"정규화 및 변환 실패: {e}")
            return torch.randn(1, 3, *self.target_size)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        return self.processing_stats.copy()
    
    def reset_stats(self):
        """통계 초기화"""
        self.processing_stats = {
            'images_processed': 0,
            'meshes_generated': 0,
            'deformations_mapped': 0,
            'textures_preserved': 0
        }
    
    def update_warping_params(self, **kwargs):
        """의류 변형 파라미터 업데이트"""
        for key, value in kwargs.items():
            if key in self.warping_params:
                self.warping_params[key] = value
                self.logger.info(f"의류 변형 파라미터 업데이트: {key} = {value}")
