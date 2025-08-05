#!/usr/bin/env python3
"""
🔥 MyCloset AI - Metrics Utility
================================

메트릭 계산 관련 함수들
- OOTD, VITON-HD, Diffusion 메트릭 계산
- SSIM, LPIPS, 품질 평가 함수들

Author: MyCloset AI Team
Date: 2025-07-31
Version: 1.0
"""

# Common imports
from app.ai_pipeline.utils.common_imports import (
    torch, nn, F, TORCH_AVAILABLE,
    logging, Dict, Any, Optional, Tuple
)

if not TORCH_AVAILABLE:
    raise ImportError("PyTorch is required for metrics calculation")

def calculate_ootd_metrics(fitted_tensor: torch.Tensor, person_tensor: torch.Tensor, 
                          cloth_tensor: torch.Tensor) -> Dict[str, float]:
    """OOTD 모델 메트릭 계산"""
    try:
        metrics = {}
        
        # SSIM 계산
        metrics['ssim'] = calculate_ssim(fitted_tensor, person_tensor)
        
        # LPIPS 계산 (간단한 버전)
        metrics['lpips'] = calculate_lpips_simple(fitted_tensor, person_tensor)
        
        # 피팅 품질 평가
        metrics['fitting_quality'] = assess_fitting_quality_tensor(fitted_tensor, person_tensor, cloth_tensor)
        
        # 텍스처 보존도
        metrics['texture_preservation'] = calculate_texture_preservation(fitted_tensor, cloth_tensor)
        
        # 이미지 품질
        metrics['image_quality'] = assess_image_quality(fitted_tensor)
        
        # 의류 유사도
        metrics['cloth_similarity'] = calculate_cloth_similarity(fitted_tensor, cloth_tensor)
        
        # 전체 품질 점수
        metrics['overall_quality'] = (
            metrics['ssim'] * 0.3 +
            metrics['fitting_quality'] * 0.3 +
            metrics['texture_preservation'] * 0.2 +
            metrics['image_quality'] * 0.2
        )
        
        return metrics
        
    except Exception as e:
        logging.error(f"OOTD 메트릭 계산 실패: {e}")
        return {
            'ssim': 0.5, 'lpips': 0.5, 'fitting_quality': 0.5,
            'texture_preservation': 0.5, 'image_quality': 0.5,
            'cloth_similarity': 0.5, 'overall_quality': 0.5
        }

def calculate_viton_metrics(fitted_tensor: torch.Tensor, person_tensor: torch.Tensor,
                           cloth_tensor: torch.Tensor, result: Dict[str, Any]) -> Dict[str, float]:
    """VITON-HD 모델 메트릭 계산"""
    try:
        metrics = {}
        
        # 기본 메트릭들
        metrics['ssim'] = calculate_ssim(fitted_tensor, person_tensor)
        metrics['lpips'] = calculate_lpips_simple(fitted_tensor, person_tensor)
        metrics['fitting_quality'] = assess_fitting_quality_tensor(fitted_tensor, person_tensor, cloth_tensor)
        
        # VITON-HD 특화 메트릭
        if 'flow' in result:
            metrics['flow_consistency'] = calculate_flow_consistency(result['flow'])
            metrics['warping_quality'] = assess_warping_quality(fitted_tensor, cloth_tensor, result['flow'])
        
        # 텍스처 및 품질 메트릭
        metrics['texture_preservation'] = calculate_texture_preservation(fitted_tensor, cloth_tensor)
        metrics['image_quality'] = assess_image_quality(fitted_tensor)
        metrics['cloth_similarity'] = calculate_cloth_similarity(fitted_tensor, cloth_tensor)
        metrics['realism'] = assess_realism(fitted_tensor)
        
        # 전체 품질 점수
        metrics['overall_quality'] = (
            metrics['ssim'] * 0.25 +
            metrics['fitting_quality'] * 0.25 +
            metrics.get('flow_consistency', 0.5) * 0.15 +
            metrics['texture_preservation'] * 0.15 +
            metrics['image_quality'] * 0.1 +
            metrics['realism'] * 0.1
        )
        
        return metrics
        
    except Exception as e:
        logging.error(f"VITON-HD 메트릭 계산 실패: {e}")
        return {
            'ssim': 0.5, 'lpips': 0.5, 'fitting_quality': 0.5,
            'flow_consistency': 0.5, 'warping_quality': 0.5,
            'texture_preservation': 0.5, 'image_quality': 0.5,
            'cloth_similarity': 0.5, 'realism': 0.5, 'overall_quality': 0.5
        }

def calculate_diffusion_metrics(fitted_tensor: torch.Tensor, person_tensor: torch.Tensor,
                               cloth_tensor: torch.Tensor) -> Dict[str, float]:
    """Diffusion 모델 메트릭 계산"""
    try:
        metrics = {}
        
        # 기본 메트릭들
        metrics['ssim'] = calculate_ssim(fitted_tensor, person_tensor)
        metrics['lpips'] = calculate_lpips_simple(fitted_tensor, person_tensor)
        metrics['fitting_quality'] = assess_fitting_quality_tensor(fitted_tensor, person_tensor, cloth_tensor)
        
        # Diffusion 특화 메트릭
        metrics['realism'] = assess_realism(fitted_tensor)
        metrics['texture_preservation'] = calculate_texture_preservation(fitted_tensor, cloth_tensor)
        metrics['image_quality'] = assess_image_quality(fitted_tensor)
        metrics['cloth_similarity'] = calculate_cloth_similarity(fitted_tensor, cloth_tensor)
        
        # 전체 품질 점수
        metrics['overall_quality'] = (
            metrics['ssim'] * 0.2 +
            metrics['fitting_quality'] * 0.2 +
            metrics['realism'] * 0.25 +
            metrics['texture_preservation'] * 0.15 +
            metrics['image_quality'] * 0.2
        )
        
        return metrics
        
    except Exception as e:
        logging.error(f"Diffusion 메트릭 계산 실패: {e}")
        return {
            'ssim': 0.5, 'lpips': 0.5, 'fitting_quality': 0.5,
            'realism': 0.5, 'texture_preservation': 0.5,
            'image_quality': 0.5, 'cloth_similarity': 0.5, 'overall_quality': 0.5
        }

def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """SSIM (Structural Similarity Index) 계산"""
    try:
        if img1.shape != img2.shape:
            return 0.5
        
        # 정규화
        img1 = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8)
        img2 = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-8)
        
        # 그레이스케일 변환
        if img1.shape[1] == 3:
            img1 = 0.299 * img1[:, 0] + 0.587 * img1[:, 1] + 0.114 * img1[:, 2]
        if img2.shape[1] == 3:
            img2 = 0.299 * img2[:, 0] + 0.587 * img2[:, 1] + 0.114 * img2[:, 2]
        
        # 간단한 SSIM 계산
        mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean().item()
        
    except Exception as e:
        logging.error(f"SSIM 계산 실패: {e}")
        return 0.5

def calculate_lpips_simple(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """간단한 LPIPS 계산 (실제 LPIPS 대신 MSE 기반)"""
    try:
        if img1.shape != img2.shape:
            return 0.5
        
        # 정규화
        img1 = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8)
        img2 = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-8)
        
        # MSE 계산
        mse = F.mse_loss(img1, img2)
        
        # LPIPS 스타일 점수로 변환 (낮을수록 좋음)
        lpips_score = 1.0 / (1.0 + mse.item())
        
        return lpips_score
        
    except Exception as e:
        logging.error(f"LPIPS 계산 실패: {e}")
        return 0.5

def assess_fitting_quality_tensor(fitted_tensor: torch.Tensor, person_tensor: torch.Tensor,
                                 cloth_tensor: torch.Tensor) -> float:
    """텐서 기반 피팅 품질 평가"""
    try:
        # 구조적 유사성
        structural_similarity = calculate_ssim(fitted_tensor, person_tensor)
        
        # 의류 보존도
        cloth_preservation = calculate_ssim(fitted_tensor, cloth_tensor)
        
        # 색상 일관성
        color_consistency = calculate_color_consistency(fitted_tensor, person_tensor)
        
        # 전체 품질 점수
        quality_score = (
            structural_similarity * 0.4 +
            cloth_preservation * 0.3 +
            color_consistency * 0.3
        )
        
        return quality_score
        
    except Exception as e:
        logging.error(f"피팅 품질 평가 실패: {e}")
        return 0.5

def calculate_flow_consistency(flow: torch.Tensor) -> float:
    """Flow 일관성 계산"""
    try:
        # Flow의 부드러움 계산
        flow_grad_x = torch.gradient(flow[:, 0], dim=1)[0]
        flow_grad_y = torch.gradient(flow[:, 1], dim=2)[0]
        
        # Flow 변화량
        flow_magnitude = torch.sqrt(flow_grad_x**2 + flow_grad_y**2)
        
        # 일관성 점수 (변화량이 적을수록 높은 점수)
        consistency = 1.0 / (1.0 + flow_magnitude.mean().item())
        
        return consistency
        
    except Exception as e:
        logging.error(f"Flow 일관성 계산 실패: {e}")
        return 0.5

def assess_warping_quality(fitted_tensor: torch.Tensor, cloth_tensor: torch.Tensor,
                          flow: torch.Tensor) -> float:
    """워핑 품질 평가"""
    try:
        # 의류 보존도
        cloth_preservation = calculate_ssim(fitted_tensor, cloth_tensor)
        
        # Flow 일관성
        flow_consistency = calculate_flow_consistency(flow)
        
        # 구조적 무결성
        structural_integrity = assess_structural_integrity(fitted_tensor)
        
        # 전체 워핑 품질
        warping_quality = (
            cloth_preservation * 0.4 +
            flow_consistency * 0.3 +
            structural_integrity * 0.3
        )
        
        return warping_quality
        
    except Exception as e:
        logging.error(f"워핑 품질 평가 실패: {e}")
        return 0.5

def calculate_texture_preservation(fitted_tensor: torch.Tensor, cloth_tensor: torch.Tensor) -> float:
    """텍스처 보존도 계산"""
    try:
        # 고주파 성분 추출 (텍스처)
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        fitted_texture = F.conv2d(fitted_tensor.mean(dim=1, keepdim=True), kernel, padding=1)
        cloth_texture = F.conv2d(cloth_tensor.mean(dim=1, keepdim=True), kernel, padding=1)
        
        # 텍스처 유사도
        texture_similarity = calculate_ssim(fitted_texture, cloth_texture)
        
        return texture_similarity
        
    except Exception as e:
        logging.error(f"텍스처 보존도 계산 실패: {e}")
        return 0.5

def assess_image_quality(fitted_tensor: torch.Tensor) -> float:
    """이미지 품질 평가"""
    try:
        # 선명도 계산 (Laplacian 분산)
        gray = fitted_tensor.mean(dim=1, keepdim=True)
        laplacian = F.conv2d(gray, torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0), padding=1)
        sharpness = laplacian.var().item()
        
        # 노이즈 레벨 추정
        noise_level = estimate_noise_level(fitted_tensor)
        
        # 품질 점수
        quality_score = min(1.0, sharpness / 100.0) * (1.0 - noise_level)
        
        return quality_score
        
    except Exception as e:
        logging.error(f"이미지 품질 평가 실패: {e}")
        return 0.5

def calculate_cloth_similarity(fitted_tensor: torch.Tensor, cloth_tensor: torch.Tensor) -> float:
    """의류 유사도 계산"""
    try:
        # 색상 유사도
        color_similarity = calculate_color_consistency(fitted_tensor, cloth_tensor)
        
        # 텍스처 유사도
        texture_similarity = calculate_texture_preservation(fitted_tensor, cloth_tensor)
        
        # 전체 유사도
        similarity = (color_similarity + texture_similarity) / 2.0
        
        return similarity
        
    except Exception as e:
        logging.error(f"의류 유사도 계산 실패: {e}")
        return 0.5

def assess_realism(fitted_tensor: torch.Tensor) -> float:
    """현실감 평가"""
    try:
        # 자연스러운 색상 분포
        color_naturalness = assess_color_naturalness(fitted_tensor)
        
        # 구조적 일관성
        structural_consistency = assess_structural_integrity(fitted_tensor)
        
        # 전체 현실감
        realism = (color_naturalness + structural_consistency) / 2.0
        
        return realism
        
    except Exception as e:
        logging.error(f"현실감 평가 실패: {e}")
        return 0.5

def calculate_color_consistency(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """색상 일관성 계산"""
    try:
        # 색상 히스토그램 비교
        hist1 = torch.histc(img1, bins=256, min=0, max=1)
        hist2 = torch.histc(img2, bins=256, min=0, max=1)
        
        # 정규화
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()
        
        # 히스토그램 유사도
        similarity = 1.0 - torch.abs(hist1 - hist2).sum() / 2.0
        
        return similarity.item()
        
    except Exception as e:
        logging.error(f"색상 일관성 계산 실패: {e}")
        return 0.5

def assess_structural_integrity(fitted_tensor: torch.Tensor) -> float:
    """구조적 무결성 평가"""
    try:
        # 엣지 일관성
        gray = fitted_tensor.mean(dim=1, keepdim=True)
        edges = F.conv2d(gray, torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0), padding=1)
        
        # 엣지 강도의 일관성
        edge_consistency = 1.0 / (1.0 + edges.std().item())
        
        return edge_consistency
        
    except Exception as e:
        logging.error(f"구조적 무결성 평가 실패: {e}")
        return 0.5

def estimate_noise_level(fitted_tensor: torch.Tensor) -> float:
    """노이즈 레벨 추정"""
    try:
        # 고주파 성분으로 노이즈 추정
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        high_freq = F.conv2d(fitted_tensor.mean(dim=1, keepdim=True), kernel, padding=1)
        
        # 노이즈 레벨
        noise_level = high_freq.abs().mean().item()
        
        return min(1.0, noise_level)
        
    except Exception as e:
        logging.error(f"노이즈 레벨 추정 실패: {e}")
        return 0.5

def assess_color_naturalness(fitted_tensor: torch.Tensor) -> float:
    """색상 자연스러움 평가"""
    try:
        # 색상 분포의 자연스러움 (간단한 버전)
        # 실제로는 더 복잡한 색상 모델 사용
        
        # 색상 범위 체크
        color_range = fitted_tensor.max() - fitted_tensor.min()
        naturalness = min(1.0, color_range.item())
        
        return naturalness
        
    except Exception as e:
        logging.error(f"색상 자연스러움 평가 실패: {e}")
        return 0.5 