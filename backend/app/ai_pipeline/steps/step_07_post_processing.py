"""
MyCloset AI 7단계: 후처리 (Post-processing)
색상 보정, 노이즈 제거, 엣지 스무딩, 조명 일치 기반 품질 향상 시스템
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage
from skimage import exposure, filters, restoration
import logging
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import math

class ColorCorrector:
    """색상 보정 클래스"""
    
    def __init__(self):
        pass
    
    def histogram_matching(self, source: np.ndarray, reference: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """히스토그램 매칭 기반 색상 보정"""
        if len(source.shape) != 3 or len(reference.shape) != 3:
            return source
        
        corrected = source.copy()
        
        for c in range(3):  # RGB 채널별 처리
            if mask is not None:
                source_channel = source[:, :, c][mask > 0.5]
                ref_channel = reference[:, :, c][mask > 0.5]
            else:
                source_channel = source[:, :, c].flatten()
                ref_channel = reference[:, :, c].flatten()
            
            if len(source_channel) > 0 and len(ref_channel) > 0:
                matched_channel = exposure.match_histograms(
                    source[:, :, c], reference[:, :, c]
                )
                corrected[:, :, c] = matched_channel
        
        return corrected
    
    def white_balance_correction(self, image: np.ndarray) -> np.ndarray:
        """화이트 밸런스 보정"""
        # Gray World 알고리즘
        mean_rgb = np.mean(image.reshape(-1, 3), axis=0)
        gray_mean = np.mean(mean_rgb)
        
        correction_factors = gray_mean / mean_rgb
        correction_factors = np.clip(correction_factors, 0.5, 2.0)  # 극단적 보정 방지
        
        corrected = image.astype(np.float32)
        for c in range(3):
            corrected[:, :, c] *= correction_factors[c]
        
        return np.clip(corrected, 0, 255).astype(np.uint8)
    
    def gamma_correction(self, image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        """감마 보정"""
        normalized = image.astype(np.float32) / 255.0
        corrected = np.power(normalized, 1.0 / gamma)
        return (corrected * 255).astype(np.uint8)
    
    def adaptive_contrast_enhancement(self, image: np.ndarray, clip_limit: float = 0.03) -> np.ndarray:
        """적응적 대비 향상 (CLAHE)"""
        if len(image.shape) == 3:
            # LAB 색공간으로 변환
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])  # L 채널에만 적용
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced

class NoiseReducer:
    """노이즈 제거 클래스"""
    
    def __init__(self):
        pass
    
    def bilateral_filtering(self, image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """양방향 필터링"""
        if len(image.shape) == 3:
            filtered = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        else:
            filtered = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        
        return filtered
    
    def non_local_means_denoising(self, image: np.ndarray, h: float = 10, template_window_size: int = 7, search_window_size: int = 21) -> np.ndarray:
        """Non-local Means 디노이징"""
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(
                image, None, h, h, template_window_size, search_window_size
            )
        else:
            denoised = cv2.fastNlMeansDenoising(
                image, None, h, template_window_size, search_window_size
            )
        
        return denoised
    
    def wiener_filtering(self, image: np.ndarray, noise_variance: float = 0.1) -> np.ndarray:
        """위너 필터링"""
        try:
            if len(image.shape) == 3:
                denoised = np.zeros_like(image)
                for c in range(3):
                    denoised[:, :, c] = restoration.wiener(
                        image[:, :, c], 
                        noise=noise_variance
                    )
            else:
                denoised = restoration.wiener(image, noise=noise_variance)
            
            return (np.clip(denoised, 0, 255)).astype(np.uint8)
        except Exception as e:
            logging.warning(f"위너 필터링 실패: {e}")
            return image
    
    def selective_gaussian_blur(self, image: np.ndarray, sigma: float = 1.0, threshold: float = 20) -> np.ndarray:
        """선택적 가우시안 블러 (엣지 보존)"""
        if len(image.shape) == 3:
            # 그레이스케일로 엣지 검출
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 엣지 검출
        edges = cv2.Canny(gray, threshold, threshold * 2)
        edge_mask = edges > 0
        
        # 가우시안 블러 적용
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        
        # 엣지가 아닌 영역에만 블러 적용
        result = image.copy()
        if len(image.shape) == 3:
            edge_mask_3ch = np.stack([edge_mask, edge_mask, edge_mask], axis=2)
            result[~edge_mask_3ch] = blurred[~edge_mask_3ch]
        else:
            result[~edge_mask] = blurred[~edge_mask]
        
        return result

class EdgeSmoother:
    """엣지 스무딩 클래스"""
    
    def __init__(self):
        pass
    
    def anisotropic_diffusion(self, image: np.ndarray, num_iter: int = 10, delta_t: float = 0.1, kappa: float = 50) -> np.ndarray:
        """이방성 확산 필터"""
        if len(image.shape) == 3:
            # 각 채널별로 처리
            result = np.zeros_like(image, dtype=np.float64)
            for c in range(3):
                result[:, :, c] = self._anisotropic_diffusion_2d(
                    image[:, :, c].astype(np.float64), num_iter, delta_t, kappa
                )
        else:
            result = self._anisotropic_diffusion_2d(
                image.astype(np.float64), num_iter, delta_t, kappa
            )
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _anisotropic_diffusion_2d(self, image: np.ndarray, num_iter: int, delta_t: float, kappa: float) -> np.ndarray:
        """2D 이방성 확산"""
        img = image.copy()
        
        for _ in range(num_iter):
            # 그래디언트 계산
            grad_n = np.roll(img, -1, axis=0) - img  # 북쪽
            grad_s = np.roll(img, 1, axis=0) - img   # 남쪽
            grad_e = np.roll(img, -1, axis=1) - img  # 동쪽
            grad_w = np.roll(img, 1, axis=1) - img   # 서쪽
            
            # 확산 계수 계산
            c_n = np.exp(-(grad_n / kappa) ** 2)
            c_s = np.exp(-(grad_s / kappa) ** 2)
            c_e = np.exp(-(grad_e / kappa) ** 2)
            c_w = np.exp(-(grad_w / kappa) ** 2)
            
            # 이미지 업데이트
            img += delta_t * (c_n * grad_n + c_s * grad_s + c_e * grad_e + c_w * grad_w)
        
        return img
    
    def guided_filter(self, image: np.ndarray, guide: np.ndarray, radius: int = 8, epsilon: float = 0.01) -> np.ndarray:
        """가이드 필터"""
        try:
            # 가이드 이미지가 컬러인 경우 그레이스케일로 변환
            if len(guide.shape) == 3:
                guide_gray = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY)
            else:
                guide_gray = guide
            
            # 가이드 필터 적용
            mean_I = cv2.boxFilter(guide_gray.astype(np.float32), cv2.CV_32F, (radius, radius))
            mean_p = cv2.boxFilter(image.astype(np.float32), cv2.CV_32F, (radius, radius))
            mean_Ip = cv2.boxFilter((guide_gray * image).astype(np.float32), cv2.CV_32F, (radius, radius))
            
            cov_Ip = mean_Ip - mean_I * mean_p
            mean_II = cv2.boxFilter((guide_gray * guide_gray).astype(np.float32), cv2.CV_32F, (radius, radius))
            var_I = mean_II - mean_I * mean_I
            
            a = cov_Ip / (var_I + epsilon)
            b = mean_p - a * mean_I
            
            mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
            mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))
            
            filtered = mean_a * guide_gray + mean_b
            
            return np.clip(filtered, 0, 255).astype(np.uint8)
            
        except Exception as e:
            logging.warning(f"가이드 필터 실패: {e}")
            return image

class LightingHarmonizer:
    """조명 일치 클래스"""
    
    def __init__(self):
        pass
    
    def poisson_blending(self, source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """푸아송 블렌딩"""
        try:
            # 마스크 영역의 중심점 계산
            coords = np.where(mask > 0.5)
            if len(coords[0]) == 0:
                return target
            
            center_y = int(np.mean(coords[0]))
            center_x = int(np.mean(coords[1]))
            center = (center_x, center_y)
            
            # 푸아송 블렌딩 실행
            blended = cv2.seamlessClone(
                source, target, (mask * 255).astype(np.uint8), 
                center, cv2.NORMAL_CLONE
            )
            
            return blended
            
        except Exception as e:
            logging.warning(f"푸아송 블렌딩 실패: {e}")
            return target
    
    def multi_band_blending(self, source: np.ndarray, target: np.ndarray, mask: np.ndarray, levels: int = 5) -> np.ndarray:
        """멀티밴드 블렌딩"""
        try:
            # 라플라시안 피라미드 생성
            source_pyramid = self._build_laplacian_pyramid(source, levels)
            target_pyramid = self._build_laplacian_pyramid(target, levels)
            mask_pyramid = self._build_gaussian_pyramid(mask, levels)
            
            # 블렌딩된 피라미드 생성
            blended_pyramid = []
            for i in range(levels):
                mask_level = mask_pyramid[i]
                if len(mask_level.shape) == 2:
                    mask_level = np.stack([mask_level, mask_level, mask_level], axis=2)
                
                blended_level = source_pyramid[i] * mask_level + target_pyramid[i] * (1 - mask_level)
                blended_pyramid.append(blended_level)
            
            # 피라미드에서 이미지 복원
            blended = self._reconstruct_from_pyramid(blended_pyramid)
            
            return np.clip(blended, 0, 255).astype(np.uint8)
            
        except Exception as e:
            logging.warning(f"멀티밴드 블렌딩 실패: {e}")
            return target
    
    def _build_gaussian_pyramid(self, image: np.ndarray, levels: int) -> List[np.ndarray]:
        """가우시안 피라미드 생성"""
        pyramid = [image.astype(np.float32)]
        
        for i in range(levels - 1):
            pyramid.append(cv2.pyrDown(pyramid[i]))
        
        return pyramid
    
    def _build_laplacian_pyramid(self, image: np.ndarray, levels: int) -> List[np.ndarray]:
        """라플라시안 피라미드 생성"""
        gaussian_pyramid = self._build_gaussian_pyramid(image, levels)
        laplacian_pyramid = []
        
        for i in range(levels - 1):
            expanded = cv2.pyrUp(gaussian_pyramid[i + 1])
            # 크기 맞추기
            if expanded.shape[:2] != gaussian_pyramid[i].shape[:2]:
                expanded = cv2.resize(expanded, (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
            
            laplacian = gaussian_pyramid[i] - expanded
            laplacian_pyramid.append(laplacian)
        
        laplacian_pyramid.append(gaussian_pyramid[-1])
        
        return laplacian_pyramid
    
    def _reconstruct_from_pyramid(self, laplacian_pyramid: List[np.ndarray]) -> np.ndarray:
        """라플라시안 피라미드에서 이미지 복원"""
        image = laplacian_pyramid[-1]
        
        for i in range(len(laplacian_pyramid) - 2, -1, -1):
            expanded = cv2.pyrUp(image)
            # 크기 맞추기
            if expanded.shape[:2] != laplacian_pyramid[i].shape[:2]:
                expanded = cv2.resize(expanded, (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0]))
            
            image = expanded + laplacian_pyramid[i]
        
        return image
    
    def match_lighting_conditions(self, source: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """조명 조건 매칭"""
        if len(source.shape) != 3 or len(target.shape) != 3:
            return source
        
        # LAB 색공간으로 변환
        source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # L 채널(명도)에서 통계 매칭
        if mask is not None:
            source_l = source_lab[:, :, 0][mask > 0.5]
            target_l = target_lab[:, :, 0][mask > 0.5]
        else:
            source_l = source_lab[:, :, 0].flatten()
            target_l = target_lab[:, :, 0].flatten()
        
        if len(source_l) > 0 and len(target_l) > 0:
            source_mean, source_std = np.mean(source_l), np.std(source_l)
            target_mean, target_std = np.mean(target_l), np.std(target_l)
            
            # 명도 조정
            adjusted_l = (source_lab[:, :, 0] - source_mean) * (target_std / max(source_std, 1e-6)) + target_mean
            source_lab[:, :, 0] = np.clip(adjusted_l, 0, 100)
        
        # RGB로 다시 변환
        adjusted = cv2.cvtColor(source_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        return adjusted

class PostProcessingStep:
    """7단계: 후처리 실행 클래스"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 후처리 컴포넌트 초기화
        self.color_corrector = ColorCorrector()
        self.noise_reducer = NoiseReducer()
        self.edge_smoother = EdgeSmoother()
        self.lighting_harmonizer = LightingHarmonizer()
        
        # 후처리 파라미터
        self.enhancement_strength = 0.7
        self.noise_reduction_level = 0.5
        self.edge_smoothing_level = 0.3
        self.color_correction_weight = 0.8

    async def process(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """후처리 메인 처리"""
        try:
            # 텐서를 numpy로 변환
            if input_tensor.dim() == 4:  # [B, C, H, W]
                image_np = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
            else:
                image_np = input_tensor.cpu().numpy().transpose(1, 2, 0)
            
            # [0, 1] → [0, 255] 변환
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            
            # 다단계 후처리 실행
            enhanced_result = await self._multi_stage_enhancement(image_np)
            
            # 텐서로 변환하여 반환
            result_tensor = self._numpy_to_tensor(enhanced_result["enhanced_image"])
            
            return result_tensor
            
        except Exception as e:
            self.logger.error(f"후처리 중 오류: {str(e)}")
            raise

    async def _multi_stage_enhancement(self, image: np.ndarray) -> Dict[str, Any]:
        """다단계 후처리"""
        import time
        start_time = time.time()
        
        # 1단계: 노이즈 제거
        denoised = await self._noise_reduction_stage(image)
        
        # 2단계: 색상 보정
        color_corrected = await self._color_correction_stage(denoised)
        
        # 3단계: 엣지 스무딩
        edge_smoothed = await self._edge_smoothing_stage(color_corrected)
        
        # 4단계: 조명 일치
        lighting_matched = await self._lighting_harmonization_stage(edge_smoothed, image)
        
        # 5단계: 최종 향상
        final_enhanced = await self._final_enhancement_stage(lighting_matched)
        
        processing_time = time.time() - start_time
        
        return {
            "enhanced_image": final_enhanced,
            "intermediate_results": {
                "denoised": denoised,
                "color_corrected": color_corrected,
                "edge_smoothed": edge_smoothed,
                "lighting_matched": lighting_matched
            },
            "processing_time": processing_time
        }

    async def _noise_reduction_stage(self, image: np.ndarray) -> np.ndarray:
        """노이즈 제거 단계"""
        try:
            # 여러 노이즈 제거 기법 조합
            
            # 1. 양방향 필터링 (엣지 보존하면서 노이즈 제거)
            bilateral_filtered = self.noise_reducer.bilateral_filtering(image)
            
            # 2. 선택적 가우시안 블러 (엣지 보존)
            selective_blurred = self.noise_reducer.selective_gaussian_blur(
                bilateral_filtered, sigma=0.8, threshold=15
            )
            
            # 3. 약한 Non-local Means (디테일 보존)
            if self.noise_reduction_level > 0.5:
                nlm_denoised = self.noise_reducer.non_local_means_denoising(
                    selective_blurred, h=8
                )
                # 원본과 블렌딩
                alpha = 0.6
                result = cv2.addWeighted(selective_blurred, 1-alpha, nlm_denoised, alpha, 0)
            else:
                result = selective_blurred
            
            return result
            
        except Exception as e:
            self.logger.warning(f"노이즈 제거 실패: {e}")
            return image

    async def _color_correction_stage(self, image: np.ndarray) -> np.ndarray:
        """색상 보정 단계"""
        try:
            # 1. 화이트 밸런스 보정
            white_balanced = self.color_corrector.white_balance_correction(image)
            
            # 2. 적응적 대비 향상 (CLAHE)
            contrast_enhanced = self.color_corrector.adaptive_contrast_enhancement(
                white_balanced, clip_limit=0.02
            )
            
            # 3. 감마 보정
            gamma_corrected = self.color_corrector.gamma_correction(
                contrast_enhanced, gamma=1.1
            )
            
            # 4. 원본과 블렌딩
            alpha = self.color_correction_weight
            result = cv2.addWeighted(image, 1-alpha, gamma_corrected, alpha, 0)
            
            return result
            
        except Exception as e:
            self.logger.warning(f"색상 보정 실패: {e}")
            return image

    async def _edge_smoothing_stage(self, image: np.ndarray) -> np.ndarray:
        """엣지 스무딩 단계"""
        try:
            # 1. 가이드 필터 (엣지 보존 스무딩)
            guided_filtered = self.edge_smoother.guided_filter(
                image, image, radius=6, epsilon=0.01
            )
            
            # 2. 약한 이방성 확산
            if self.edge_smoothing_level > 0.3:
                anisotropic_smoothed = self.edge_smoother.anisotropic_diffusion(
                    guided_filtered, num_iter=5, delta_t=0.1, kappa=30
                )
                # 원본과 블렌딩
                alpha = 0.4
                result = cv2.addWeighted(guided_filtered, 1-alpha, anisotropic_smoothed, alpha, 0)
            else:
                result = guided_filtered
            
            return result
            
        except Exception as e:
            self.logger.warning(f"엣지 스무딩 실패: {e}")
            return image

    async def _lighting_harmonization_stage(self, image: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """조명 일치 단계"""
        try:
            # 조명 조건 매칭
            lighting_matched = self.lighting_harmonizer.match_lighting_conditions(
                image, reference
            )
            
            # 원본과 블렌딩 (자연스러운 조명 조정)
            alpha = 0.6
            result = cv2.addWeighted(image, 1-alpha, lighting_matched, alpha, 0)
            
            return result
            
        except Exception as e:
            self.logger.warning(f"조명 일치 실패: {e}")
            return image

    async def _final_enhancement_stage(self, image: np.ndarray) -> np.ndarray:
        """최종 향상 단계"""
        try:
            # 1. 미세한 샤프닝
            kernel = np.array([[-0.1, -0.1, -0.1],
                              [-0.1,  1.8, -0.1],
                              [-0.1, -0.1, -0.1]])
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # 2. 색상 채도 약간 향상
            if len(image.shape) == 3:
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] *= 1.1  # 채도 10% 증가
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                saturation_enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            else:
                saturation_enhanced = image
            
            # 3. 최종 블렌딩
            alpha = self.enhancement_strength
            result = cv2.addWeighted(image, 1-alpha, sharpened, alpha*0.3, 0)
            result = cv2.addWeighted(result, 0.9, saturation_enhanced, 0.1, 0)
            
            return np.clip(result, 0, 255).astype(np.uint8)
            
        except Exception as e:
            self.logger.warning(f"최종 향상 실패: {e}")
            return image

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
        
        return tensor.unsqueeze(0).to(self.config.device if hasattr(self.config, 'device') else 'cpu')

    def compare_before_after(self, original: np.ndarray, enhanced: np.ndarray, save_path: Optional[str] = None) -> np.ndarray:
        """전후 비교 이미지 생성"""
        
        # 이미지 크기 맞추기
        h, w = original.shape[:2]
        enhanced_resized = cv2.resize(enhanced, (w, h))
        
        # 수평으로 연결
        comparison = np.hstack([original, enhanced_resized])
        
        # 구분선 그리기
        cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 2)
        
        # 라벨 추가
        cv2.putText(comparison, "Before", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(comparison, "After", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 저장 (옵션)
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        
        return comparison

    def get_enhancement_metrics(self, original: np.ndarray, enhanced: np.ndarray) -> Dict[str, float]:
        """향상 품질 메트릭"""
        try:
            # PSNR 계산
            mse = np.mean((original.astype(np.float32) - enhanced.astype(np.float32)) ** 2)
            if mse == 0:
                psnr = float('inf')
            else:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            
            # SSIM 계산 (간단한 버전)
            ssim = self._calculate_ssim(original, enhanced)
            
            # 엣지 보존도
            edge_preservation = self._calculate_edge_preservation(original, enhanced)
            
            # 색상 일치도
            color_consistency = self._calculate_color_consistency(original, enhanced)
            
            return {
                "psnr": float(psnr),
                "ssim": float(ssim),
                "edge_preservation": float(edge_preservation),
                "color_consistency": float(color_consistency)
            }
            
        except Exception as e:
            self.logger.error(f"메트릭 계산 실패: {e}")
            return {}

    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """SSIM 계산"""
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            img1_gray = img1
            img2_gray = img2
        
        mu1 = cv2.GaussianBlur(img1_gray.astype(np.float32), (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2_gray.astype(np.float32), (11, 11), 1.5)
        
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(img1_gray.astype(np.float32) * img1_gray.astype(np.float32), (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2_gray.astype(np.float32) * img2_gray.astype(np.float32), (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1_gray.astype(np.float32) * img2_gray.astype(np.float32), (11, 11), 1.5) - mu1_mu2
        
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return np.mean(ssim_map)

    def _calculate_edge_preservation(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """엣지 보존도 계산"""
        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        else:
            orig_gray = original
            enh_gray = enhanced
        
        # 소벨 엣지 검출
        orig_edges = cv2.Sobel(orig_gray, cv2.CV_64F, 1, 1, ksize=3)
        enh_edges = cv2.Sobel(enh_gray, cv2.CV_64F, 1, 1, ksize=3)
        
        # 정규화된 상관계수
        correlation = np.corrcoef(orig_edges.flatten(), enh_edges.flatten())[0, 1]
        
        return max(0, correlation)

    def _calculate_color_consistency(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """색상 일치도 계산"""
        if len(original.shape) != 3 or len(enhanced.shape) != 3:
            return 1.0
        
        # 각 채널별 히스토그램 비교
        consistency_scores = []
        
        for c in range(3):
            hist1 = cv2.calcHist([original], [c], None, [256], [0, 256])
            hist2 = cv2.calcHist([enhanced], [c], None, [256], [0, 256])
            
            # 히스토그램 정규화
            hist1 = hist1 / np.sum(hist1)
            hist2 = hist2 / np.sum(hist2)
            
            # 상관계수 계산
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            consistency_scores.append(correlation)
        
        return np.mean(consistency_scores)

# 사용 예시
async def example_usage():
    """후처리 사용 예시"""
    
    # 설정
    class Config:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    config = Config()
    
    # 후처리 단계 초기화
    post_processing = PostProcessingStep(config)
    
    # 더미 입력 생성 (가상 피팅 결과)
    dummy_input = torch.randn(1, 3, 512, 512).to(config.device)
    
    # 처리
    result = await post_processing.process(dummy_input)
    
    print(f"후처리 완료 - 출력 크기: {result.shape}")
    print("이미지 품질이 향상되었습니다!")

if __name__ == "__main__":
    asyncio.run(example_usage())