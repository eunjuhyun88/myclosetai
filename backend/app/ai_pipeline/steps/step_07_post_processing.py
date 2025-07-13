"""
7단계: 후처리 (Post Processing) - 품질 향상
Super Resolution, 노이즈 제거, 색상 보정을 통한 최종 품질 향상
"""
import os
import logging
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

class PostProcessingStep:
    """후처리 스텝 - 품질 향상 및 최종 보정"""
    
    def __init__(self, model_loader, device: str, config: Dict[str, Any] = None):
        """
        Args:
            model_loader: 모델 로더 인스턴스
            device: 사용할 디바이스
            config: 설정 딕셔너리
        """
        self.model_loader = model_loader
        self.device = device
        self.config = config or {}
        
        # 기본 설정
        self.enable_super_resolution = self.config.get('enable_super_resolution', True)
        self.enable_denoising = self.config.get('enable_denoising', True)
        self.enable_color_correction = self.config.get('enable_color_correction', True)
        self.enable_sharpening = self.config.get('enable_sharpening', True)
        self.scale_factor = self.config.get('scale_factor', 2)  # SR 스케일
        
        # 모델 관련
        self.sr_model = None
        self.denoising_model = None
        self.is_initialized = False
        
        logger.info(f"🎯 후처리 스텝 초기화 - 디바이스: {device}")
    
    async def initialize(self) -> bool:
        """후처리 모델 초기화"""
        try:
            logger.info("🔄 후처리 모델 로드 중...")
            
            # Super Resolution 모델 초기화
            if self.enable_super_resolution:
                await self._initialize_sr_model()
            
            # 노이즈 제거 모델 초기화
            if self.enable_denoising:
                await self._initialize_denoising_model()
            
            self.is_initialized = True
            logger.info("✅ 후처리 모델 로드 완료")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 후처리 모델 초기화 실패: {e}")
            self.is_initialized = False
            return False
    
    async def _initialize_sr_model(self):
        """Super Resolution 모델 초기화"""
        try:
            # ESRGAN 또는 Real-ESRGAN 스타일 모델
            self.sr_model = self._create_sr_model()
            
            # 사전 훈련된 가중치 로드
            model_path = self._get_sr_model_path()
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.sr_model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"✅ Super Resolution 가중치 로드: {model_path}")
            else:
                logger.warning(f"⚠️ SR 가중치 파일 없음: {model_path} - 데모 모드로 실행")
            
            # 모델 최적화
            self.sr_model = self.model_loader.optimize_model(self.sr_model, 'post_processing')
            self.sr_model.eval()
            
        except Exception as e:
            logger.warning(f"SR 모델 초기화 실패: {e}")
            self.sr_model = None
    
    async def _initialize_denoising_model(self):
        """노이즈 제거 모델 초기화"""
        try:
            # DnCNN 스타일 노이즈 제거 모델
            self.denoising_model = self._create_denoising_model()
            
            # 가중치 로드
            model_path = self._get_denoising_model_path()
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.denoising_model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"✅ 노이즈 제거 가중치 로드: {model_path}")
            else:
                logger.warning(f"⚠️ 노이즈 제거 가중치 파일 없음: {model_path}")
            
            # 모델 최적화
            self.denoising_model = self.model_loader.optimize_model(self.denoising_model, 'post_processing')
            self.denoising_model.eval()
            
        except Exception as e:
            logger.warning(f"노이즈 제거 모델 초기화 실패: {e}")
            self.denoising_model = None
    
    def _create_sr_model(self):
        """Super Resolution 모델 생성 (ESRGAN 스타일)"""
        class RRDB(nn.Module):
            """Residual in Residual Dense Block"""
            
            def __init__(self, nf, gc=32):
                super(RRDB, self).__init__()
                self.RDB1 = ResidualDenseBlock(nf, gc)
                self.RDB2 = ResidualDenseBlock(nf, gc)
                self.RDB3 = ResidualDenseBlock(nf, gc)
                
            def forward(self, x):
                out = self.RDB1(x)
                out = self.RDB2(out)
                out = self.RDB3(out)
                return out * 0.2 + x
        
        class ResidualDenseBlock(nn.Module):
            def __init__(self, nf=64, gc=32):
                super(ResidualDenseBlock, self).__init__()
                # Dense layers
                self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
                self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
                self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
                self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
                self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
                self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
                
            def forward(self, x):
                x1 = self.lrelu(self.conv1(x))
                x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
                x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
                x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
                x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
                return x5 * 0.2 + x
        
        class RRDBNet(nn.Module):
            def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, scale=2):
                super(RRDBNet, self).__init__()
                
                self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
                self.RRDB_trunk = nn.Sequential(*[RRDB(nf) for _ in range(nb)])
                self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
                
                # Upsampling
                if scale == 4:
                    self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
                    self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
                elif scale == 2:
                    self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
                
                self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
                self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
                
                self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
                self.scale = scale
                
            def forward(self, x):
                fea = self.conv_first(x)
                trunk = self.trunk_conv(self.RRDB_trunk(fea))
                fea = fea + trunk
                
                # Upsampling
                fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
                if self.scale == 4:
                    fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
                
                out = self.conv_last(self.lrelu(self.HRconv(fea)))
                return out
        
        return RRDBNet(scale=self.scale_factor).to(self.device)
    
    def _create_denoising_model(self):
        """노이즈 제거 모델 생성 (DnCNN 스타일)"""
        class DnCNN(nn.Module):
            def __init__(self, channels=3, num_of_layers=17):
                super(DnCNN, self).__init__()
                kernel_size = 3
                padding = 1
                features = 64
                
                layers = []
                layers.append(nn.Conv2d(in_channels=channels, out_channels=features, 
                                      kernel_size=kernel_size, padding=padding, bias=False))
                layers.append(nn.ReLU(inplace=True))
                
                for _ in range(num_of_layers - 2):
                    layers.append(nn.Conv2d(in_channels=features, out_channels=features, 
                                          kernel_size=kernel_size, padding=padding, bias=False))
                    layers.append(nn.BatchNorm2d(features))
                    layers.append(nn.ReLU(inplace=True))
                
                layers.append(nn.Conv2d(in_channels=features, out_channels=channels, 
                                      kernel_size=kernel_size, padding=padding, bias=False))
                
                self.dncnn = nn.Sequential(*layers)
                
            def forward(self, x):
                noise = self.dncnn(x)
                return x - noise  # 잔차 학습
        
        return DnCNN().to(self.device)
    
    def _get_sr_model_path(self) -> str:
        """SR 모델 파일 경로"""
        model_dir = self.config.get('model_dir', 'app/models/ai_models/sr')
        model_file = self.config.get('sr_model_file', 'esrgan_x2.pth')
        return os.path.join(model_dir, model_file)
    
    def _get_denoising_model_path(self) -> str:
        """노이즈 제거 모델 파일 경로"""
        model_dir = self.config.get('model_dir', 'app/models/ai_models/denoising')
        model_file = self.config.get('denoising_model_file', 'dncnn.pth')
        return os.path.join(model_dir, model_file)
    
    def process(self, fitted_image_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        후처리 실행
        
        Args:
            fitted_image_tensor: 6단계 가상 피팅 결과 이미지 텐서
            
        Returns:
            처리 결과 딕셔너리
        """
        if not self.is_initialized:
            raise RuntimeError("후처리 모듈이 초기화되지 않았습니다.")
        
        start_time = time.time()
        
        try:
            current_image = fitted_image_tensor.clone()
            processing_steps = []
            
            # 1. 노이즈 제거
            if self.enable_denoising:
                denoised_image, denoise_info = self._apply_denoising(current_image)
                current_image = denoised_image
                processing_steps.append(denoise_info)
            
            # 2. Super Resolution
            if self.enable_super_resolution:
                sr_image, sr_info = self._apply_super_resolution(current_image)
                current_image = sr_image
                processing_steps.append(sr_info)
            
            # 3. 색상 보정
            if self.enable_color_correction:
                color_corrected, color_info = self._apply_color_correction(current_image)
                current_image = color_corrected
                processing_steps.append(color_info)
            
            # 4. 샤프닝
            if self.enable_sharpening:
                sharpened_image, sharp_info = self._apply_sharpening(current_image)
                current_image = sharpened_image
                processing_steps.append(sharp_info)
            
            # 5. 최종 품질 검증
            quality_metrics = self._evaluate_enhancement_quality(fitted_image_tensor, current_image)
            
            # 6. 후처리 통계
            enhancement_stats = self._calculate_enhancement_stats(fitted_image_tensor, current_image)
            
            processing_time = time.time() - start_time
            
            result = {
                "enhanced_image": current_image,
                "enhancement_score": float(quality_metrics.get('overall_score', 0.8)),
                "quality_metrics": quality_metrics,
                "enhancement_stats": enhancement_stats,
                "processing_steps": processing_steps,
                "processing_time": processing_time,
                "improvements_applied": len(processing_steps)
            }
            
            logger.info(f"✅ 후처리 완료 - 처리시간: {processing_time:.3f}초, 개선점수: {quality_metrics.get('overall_score', 0):.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 후처리 실패: {e}")
            raise
    
    def _apply_denoising(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """노이즈 제거 적용"""
        step_start = time.time()
        
        try:
            if self.denoising_model is not None:
                # 딥러닝 기반 노이즈 제거
                with torch.no_grad():
                    denoised = self.denoising_model(image_tensor)
                    denoised = torch.clamp(denoised, 0, 1)
                
                # 노이즈 제거 효과 측정
                noise_reduction = self._measure_noise_reduction(image_tensor, denoised)
                method = "DnCNN"
                
            else:
                # 전통적인 방법 (가우시안 블러)
                denoised = self._apply_gaussian_denoising(image_tensor)
                noise_reduction = 0.3  # 추정값
                method = "Gaussian"
            
            processing_time = time.time() - step_start
            
            info = {
                "step": "denoising",
                "method": method,
                "noise_reduction": noise_reduction,
                "processing_time": processing_time,
                "applied": True
            }
            
            return denoised, info
            
        except Exception as e:
            logger.warning(f"노이즈 제거 실패: {e}")
            info = {"step": "denoising", "applied": False, "error": str(e)}
            return image_tensor, info
    
    def _apply_gaussian_denoising(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """가우시안 블러를 이용한 노이즈 제거"""
        try:
            kernel_size = 3
            sigma = 0.5
            
            # 가우시안 커널 생성
            kernel_1d = torch.tensor([
                np.exp(-(x - kernel_size//2)**2 / (2 * sigma**2)) 
                for x in range(kernel_size)
            ]).float().to(self.device)
            kernel_1d = kernel_1d / kernel_1d.sum()
            
            # 2D 커널
            kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
            kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
            
            # 채널별 컨볼루션
            denoised_channels = []
            for i in range(image_tensor.shape[1]):
                channel = image_tensor[:, i:i+1, :, :]
                denoised_channel = F.conv2d(channel, kernel_2d, padding=kernel_size//2)
                denoised_channels.append(denoised_channel)
            
            denoised = torch.cat(denoised_channels, dim=1)
            
            return denoised
            
        except Exception as e:
            logger.warning(f"가우시안 노이즈 제거 실패: {e}")
            return image_tensor
    
    def _apply_super_resolution(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Super Resolution 적용"""
        step_start = time.time()
        
        try:
            if self.sr_model is not None and self.scale_factor > 1:
                # 딥러닝 기반 Super Resolution
                with torch.no_grad():
                    # 입력 크기 제한 (메모리 절약)
                    h, w = image_tensor.shape[2], image_tensor.shape[3]
                    max_size = 512
                    
                    if max(h, w) > max_size:
                        # 타일 기반 처리
                        sr_image = self._apply_tiled_sr(image_tensor, max_size)
                    else:
                        sr_image = self.sr_model(image_tensor)
                    
                    sr_image = torch.clamp(sr_image, 0, 1)
                
                # 품질 향상 측정
                quality_improvement = self._measure_sr_quality(image_tensor, sr_image)
                method = "ESRGAN"
                
            else:
                # 바이큐빅 업스케일링
                scale = self.scale_factor if self.scale_factor > 1 else 1
                sr_image = F.interpolate(
                    image_tensor, 
                    scale_factor=scale, 
                    mode='bicubic', 
                    align_corners=False
                )
                quality_improvement = 0.2  # 추정값
                method = "Bicubic"
            
            processing_time = time.time() - step_start
            
            info = {
                "step": "super_resolution",
                "method": method,
                "scale_factor": self.scale_factor,
                "quality_improvement": quality_improvement,
                "processing_time": processing_time,
                "applied": True
            }
            
            return sr_image, info
            
        except Exception as e:
            logger.warning(f"Super Resolution 실패: {e}")
            info = {"step": "super_resolution", "applied": False, "error": str(e)}
            return image_tensor, info
    
    def _apply_tiled_sr(self, image_tensor: torch.Tensor, tile_size: int) -> torch.Tensor:
        """타일 기반 Super Resolution"""
        try:
            b, c, h, w = image_tensor.shape
            scale = self.scale_factor
            
            # 출력 크기
            output_h, output_w = h * scale, w * scale
            sr_image = torch.zeros(b, c, output_h, output_w, device=self.device)
            
            # 타일 단위로 처리
            for y in range(0, h, tile_size):
                for x in range(0, w, tile_size):
                    # 타일 영역
                    y_end = min(y + tile_size, h)
                    x_end = min(x + tile_size, w)
                    
                    tile = image_tensor[:, :, y:y_end, x:x_end]
                    
                    # SR 적용
                    sr_tile = self.sr_model(tile)
                    
                    # 출력 위치
                    sr_y = y * scale
                    sr_x = x * scale
                    sr_y_end = sr_y + sr_tile.shape[2]
                    sr_x_end = sr_x + sr_tile.shape[3]
                    
                    sr_image[:, :, sr_y:sr_y_end, sr_x:sr_x_end] = sr_tile
            
            return sr_image
            
        except Exception as e:
            logger.warning(f"타일 기반 SR 실패: {e}")
            return F.interpolate(image_tensor, scale_factor=self.scale_factor, mode='bicubic')
    
    def _apply_color_correction(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """색상 보정 적용"""
        step_start = time.time()
        
        try:
            # PIL 이미지로 변환
            pil_image = self._tensor_to_pil(image_tensor)
            
            # 색상 보정 적용
            enhanced_image = pil_image
            corrections = []
            
            # 1. 대비 개선
            enhancer = ImageEnhance.Contrast(enhanced_image)
            enhanced_image = enhancer.enhance(1.1)  # 10% 대비 증가
            corrections.append("contrast")
            
            # 2. 채도 조정
            enhancer = ImageEnhance.Color(enhanced_image)
            enhanced_image = enhancer.enhance(1.05)  # 5% 채도 증가
            corrections.append("saturation")
            
            # 3. 밝기 조정 (필요한 경우)
            brightness_factor = self._calculate_brightness_factor(pil_image)
            if abs(brightness_factor - 1.0) > 0.05:
                enhancer = ImageEnhance.Brightness(enhanced_image)
                enhanced_image = enhancer.enhance(brightness_factor)
                corrections.append("brightness")
            
            # 4. 감마 보정
            gamma_corrected = self._apply_gamma_correction(enhanced_image, 1.1)
            enhanced_image = gamma_corrected
            corrections.append("gamma")
            
            # 텐서로 다시 변환
            corrected_tensor = self._pil_to_tensor(enhanced_image)
            
            processing_time = time.time() - step_start
            
            info = {
                "step": "color_correction",
                "corrections_applied": corrections,
                "brightness_factor": brightness_factor,
                "processing_time": processing_time,
                "applied": True
            }
            
            return corrected_tensor, info
            
        except Exception as e:
            logger.warning(f"색상 보정 실패: {e}")
            info = {"step": "color_correction", "applied": False, "error": str(e)}
            return image_tensor, info
    
    def _apply_sharpening(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """샤프닝 적용"""
        step_start = time.time()
        
        try:
            # PIL 이미지로 변환
            pil_image = self._tensor_to_pil(image_tensor)
            
            # 언샵 마스크 필터 적용
            sharpened = pil_image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
            
            # 텐서로 변환
            sharpened_tensor = self._pil_to_tensor(sharpened)
            
            # 샤프닝 강도 측정
            sharpness_improvement = self._measure_sharpness(image_tensor, sharpened_tensor)
            
            processing_time = time.time() - step_start
            
            info = {
                "step": "sharpening",
                "method": "UnsharpMask",
                "sharpness_improvement": sharpness_improvement,
                "processing_time": processing_time,
                "applied": True
            }
            
            return sharpened_tensor, info
            
        except Exception as e:
            logger.warning(f"샤프닝 실패: {e}")
            info = {"step": "sharpening", "applied": False, "error": str(e)}
            return image_tensor, info
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """텐서를 PIL 이미지로 변환"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        
        # 0-1 범위로 클램핑
        tensor = torch.clamp(tensor, 0, 1)
        
        # NumPy 배열로 변환
        array = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        return Image.fromarray(array)
    
    def _pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """PIL 이미지를 텐서로 변환"""
        array = np.array(pil_image) / 255.0
        tensor = torch.from_numpy(array).float().permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)
    
    def _calculate_brightness_factor(self, image: Image.Image) -> float:
        """적절한 밝기 팩터 계산"""
        try:
            # 그레이스케일로 변환하여 평균 밝기 계산
            gray = image.convert('L')
            histogram = gray.histogram()
            
            # 평균 밝기 계산
            total_pixels = sum(histogram)
            brightness_sum = sum(i * count for i, count in enumerate(histogram))
            average_brightness = brightness_sum / total_pixels
            
            # 적정 밝기 (128)를 기준으로 조정 팩터 계산
            target_brightness = 128
            factor = target_brightness / average_brightness
            
            # 극단적인 조정 방지
            return max(0.8, min(1.3, factor))
            
        except Exception as e:
            logger.warning(f"밝기 팩터 계산 실패: {e}")
            return 1.0
    
    def _apply_gamma_correction(self, image: Image.Image, gamma: float) -> Image.Image:
        """감마 보정 적용"""
        try:
            # 감마 테이블 생성
            gamma_table = [int(((i / 255.0) ** (1.0 / gamma)) * 255) for i in range(256)]
            
            # 각 채널에 감마 보정 적용
            if image.mode == 'RGB':
                r, g, b = image.split()
                r = r.point(gamma_table)
                g = g.point(gamma_table)
                b = b.point(gamma_table)
                return Image.merge('RGB', (r, g, b))
            else:
                return image.point(gamma_table)
                
        except Exception as e:
            logger.warning(f"감마 보정 실패: {e}")
            return image
    
    def _measure_noise_reduction(self, original: torch.Tensor, denoised: torch.Tensor) -> float:
        """노이즈 제거 효과 측정"""
        try:
            # 노이즈 추정 (라플라시안 분산)
            laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            
            original_gray = 0.299 * original[:, 0] + 0.587 * original[:, 1] + 0.114 * original[:, 2]
            denoised_gray = 0.299 * denoised[:, 0] + 0.587 * denoised[:, 1] + 0.114 * denoised[:, 2]
            
            original_noise = F.conv2d(original_gray.unsqueeze(1), laplacian, padding=1)
            denoised_noise = F.conv2d(denoised_gray.unsqueeze(1), laplacian, padding=1)
            
            original_variance = torch.var(original_noise)
            denoised_variance = torch.var(denoised_noise)
            
            if original_variance > 0:
                reduction = 1.0 - (denoised_variance / original_variance).item()
                return max(0.0, min(1.0, reduction))
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"노이즈 제거 측정 실패: {e}")
            return 0.3
    
    def _measure_sr_quality(self, original: torch.Tensor, sr_image: torch.Tensor) -> float:
        """Super Resolution 품질 측정"""
        try:
            # 원본을 SR 크기로 업스케일
            upscaled_original = F.interpolate(original, size=sr_image.shape[2:], mode='bicubic', align_corners=False)
            
            # PSNR 계산
            mse = torch.mean((sr_image - upscaled_original) ** 2)
            if mse > 0:
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                # PSNR을 0-1 범위로 정규화 (30dB 이상을 1.0으로)
                quality = min(1.0, max(0.0, (psnr.item() - 20) / 10))
            else:
                quality = 1.0
            
            return quality
            
        except Exception as e:
            logger.warning(f"SR 품질 측정 실패: {e}")
            return 0.5
    
    def _measure_sharpness(self, original: torch.Tensor, sharpened: torch.Tensor) -> float:
        """샤프닝 효과 측정"""
        try:
            # 라플라시안 분산으로 샤프니스 측정
            laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            
            original_gray = 0.299 * original[:, 0] + 0.587 * original[:, 1] + 0.114 * original[:, 2]
            sharpened_gray = 0.299 * sharpened[:, 0] + 0.587 * sharpened[:, 1] + 0.114 * sharpened[:, 2]
            
            original_edges = F.conv2d(original_gray.unsqueeze(1), laplacian, padding=1)
            sharpened_edges = F.conv2d(sharpened_gray.unsqueeze(1), laplacian, padding=1)
            
            original_sharpness = torch.var(original_edges)
            sharpened_sharpness = torch.var(sharpened_edges)
            
            if original_sharpness > 0:
                improvement = (sharpened_sharpness / original_sharpness).item() - 1.0
                return max(0.0, min(1.0, improvement))
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"샤프니스 측정 실패: {e}")
            return 0.2
    
    def _evaluate_enhancement_quality(self, original: torch.Tensor, enhanced: torch.Tensor) -> Dict[str, float]:
        """품질 향상 평가"""
        metrics = {}
        
        try:
            # 구조적 유사도
            metrics['structural_similarity'] = self._calculate_ssim(original, enhanced)
            
            # 색상 다양성 (색상 히스토그램의 엔트로피)
            metrics['color_diversity'] = self._calculate_color_diversity(enhanced)
            
            # 디테일 보존도
            metrics['detail_preservation'] = self._calculate_detail_preservation(original, enhanced)
            
            # 아티팩트 레벨
            metrics['artifact_level'] = 1.0 - self._detect_artifacts(enhanced)
            
            # 전체 품질 점수
            overall_score = (
                metrics['structural_similarity'] * 0.3 +
                metrics['color_diversity'] * 0.2 +
                metrics['detail_preservation'] * 0.3 +
                metrics['artifact_level'] * 0.2
            )
            metrics['overall_score'] = max(0.0, min(1.0, overall_score))
            
        except Exception as e:
            logger.warning(f"품질 평가 실패: {e}")
            metrics = {'overall_score': 0.8}
        
        return metrics
    
    def _calculate_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """SSIM 계산"""
        try:
            # 그레이스케일 변환
            if img1.shape[1] == 3:
                gray1 = 0.299 * img1[:, 0] + 0.587 * img1[:, 1] + 0.114 * img1[:, 2]
                gray2 = 0.299 * img2[:, 0] + 0.587 * img2[:, 1] + 0.114 * img2[:, 2]
            else:
                gray1 = img1.squeeze(1)
                gray2 = img2.squeeze(1)
            
            # 크기 조정 (필요한 경우)
            if gray1.shape != gray2.shape:
                gray2 = F.interpolate(gray2.unsqueeze(1), size=gray1.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
            
            # SSIM 계산
            mu1 = torch.mean(gray1)
            mu2 = torch.mean(gray2)
            
            sigma1_sq = torch.var(gray1)
            sigma2_sq = torch.var(gray2)
            sigma12 = torch.mean((gray1 - mu1) * (gray2 - mu2))
            
            C1, C2 = 0.01**2, 0.03**2
            ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
            
            return float(ssim.item())
            
        except Exception as e:
            logger.warning(f"SSIM 계산 실패: {e}")
            return 0.8
    
    def _calculate_color_diversity(self, image: torch.Tensor) -> float:
        """색상 다양성 계산"""
        try:
            diversity_scores = []
            
            for c in range(3):  # RGB 채널
                channel = image[:, c, :, :].flatten()
                histogram = torch.histc(channel, bins=256, min=0, max=1)
                
                # 정규화
                histogram = histogram / torch.sum(histogram)
                
                # 엔트로피 계산
                entropy = -torch.sum(histogram * torch.log(histogram + 1e-10))
                diversity_scores.append(entropy.item())
            
            # 평균 엔트로피를 0-1 범위로 정규화
            avg_entropy = np.mean(diversity_scores)
            normalized_diversity = min(1.0, avg_entropy / 8.0)  # log(256) ≈ 8
            
            return normalized_diversity
            
        except Exception as e:
            logger.warning(f"색상 다양성 계산 실패: {e}")
            return 0.7
    
    def _calculate_detail_preservation(self, original: torch.Tensor, enhanced: torch.Tensor) -> float:
        """디테일 보존도 계산"""
        try:
            # 고주파 성분 비교
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # 그레이스케일 변환
            orig_gray = 0.299 * original[:, 0] + 0.587 * original[:, 1] + 0.114 * original[:, 2]
            enh_gray = 0.299 * enhanced[:, 0] + 0.587 * enhanced[:, 1] + 0.114 * enhanced[:, 2]
            
            # 크기 조정
            if orig_gray.shape != enh_gray.shape:
                enh_gray = F.interpolate(enh_gray.unsqueeze(1), size=orig_gray.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
            
            # 그라디언트 계산
            orig_grad_x = F.conv2d(orig_gray.unsqueeze(1), sobel_x, padding=1)
            orig_grad_y = F.conv2d(orig_gray.unsqueeze(1), sobel_y, padding=1)
            enh_grad_x = F.conv2d(enh_gray.unsqueeze(1), sobel_x, padding=1)
            enh_grad_y = F.conv2d(enh_gray.unsqueeze(1), sobel_y, padding=1)
            
            orig_grad_mag = torch.sqrt(orig_grad_x**2 + orig_grad_y**2)
            enh_grad_mag = torch.sqrt(enh_grad_x**2 + enh_grad_y**2)
            
            # 상관계수 계산
            correlation = torch.corrcoef(torch.stack([orig_grad_mag.flatten(), enh_grad_mag.flatten()]))[0, 1]
            
            return float(correlation.item()) if not torch.isnan(correlation) else 0.8
            
        except Exception as e:
            logger.warning(f"디테일 보존도 계산 실패: {e}")
            return 0.8
    
    def _detect_artifacts(self, image: torch.Tensor) -> float:
        """아티팩트 검출"""
        try:
            # 라플라시안으로 급격한 변화 검출
            laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            
            gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
            edges = F.conv2d(gray.unsqueeze(1), laplacian, padding=1)
            
            # 극값 검출
            edge_variance = torch.var(edges)
            artifact_score = min(1.0, edge_variance.item() / 0.1)  # 정규화
            
            return artifact_score
            
        except Exception as e:
            logger.warning(f"아티팩트 검출 실패: {e}")
            return 0.1
    
    def _calculate_enhancement_stats(self, original: torch.Tensor, enhanced: torch.Tensor) -> Dict[str, Any]:
        """품질 향상 통계"""
        stats = {}
        
        try:
            # 크기 비교
            orig_size = original.shape[2:]
            enh_size = enhanced.shape[2:]
            stats['size_increase'] = {
                'original': orig_size,
                'enhanced': enh_size,
                'factor': (enh_size[0] * enh_size[1]) / (orig_size[0] * orig_size[1])
            }
            
            # 밝기 변화
            orig_brightness = torch.mean(original).item()
            enh_brightness = torch.mean(enhanced).item()
            stats['brightness_change'] = enh_brightness - orig_brightness
            
            # 대비 변화
            orig_contrast = torch.std(original).item()
            enh_contrast = torch.std(enhanced).item()
            stats['contrast_change'] = enh_contrast - orig_contrast
            
            # 색상 채도 변화
            orig_saturation = self._calculate_saturation(original)
            enh_saturation = self._calculate_saturation(enhanced)
            stats['saturation_change'] = enh_saturation - orig_saturation
            
        except Exception as e:
            logger.warning(f"통계 계산 실패: {e}")
            stats = {}
        
        return stats
    
    def _calculate_saturation(self, image: torch.Tensor) -> float:
        """색상 채도 계산"""
        try:
            # RGB를 HSV로 변환 후 채도 계산 (근사)
            r, g, b = image[:, 0], image[:, 1], image[:, 2]
            
            max_val = torch.max(torch.max(r, g), b)
            min_val = torch.min(torch.min(r, g), b)
            
            saturation = (max_val - min_val) / (max_val + 1e-8)
            
            return torch.mean(saturation).item()
            
        except Exception as e:
            logger.warning(f"채도 계산 실패: {e}")
            return 0.5
    
    async def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "enable_super_resolution": self.enable_super_resolution,
            "enable_denoising": self.enable_denoising,
            "enable_color_correction": self.enable_color_correction,
            "enable_sharpening": self.enable_sharpening,
            "scale_factor": self.scale_factor,
            "device": self.device,
            "initialized": self.is_initialized,
            "sr_model_loaded": self.sr_model is not None,
            "denoising_model_loaded": self.denoising_model is not None
        }
    
    async def cleanup(self):
        """리소스 정리"""
        if self.sr_model:
            del self.sr_model
            self.sr_model = None
        
        if self.denoising_model:
            del self.denoising_model
            self.denoising_model = None
        
        self.is_initialized = False
        logger.info("🧹 후처리 스텝 리소스 정리 완료")