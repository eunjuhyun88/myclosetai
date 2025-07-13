"""
MyCloset AI 6단계: 가상 피팅 생성 (Virtual Try-on Generation)
HR-VITON + ACGPN + PF-AFN 기반 고해상도 가상 피팅 시스템
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import logging
from typing import Dict, List, Optional, Tuple, Any
import asyncio
from pathlib import Path
import math

class MultiScaleDiscriminator(nn.Module):
    """다중 스케일 판별자"""
    def __init__(self, input_nc=3, ndf=64, n_layers=3, num_D=3):
        super().__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        
        # 각 스케일별 판별자 생성
        for i in range(num_D):
            subnet = NLayerDiscriminator(input_nc, ndf, n_layers)
            setattr(self, f'scale_{i}', subnet)

    def forward(self, input):
        results = []
        get_intermediate_features = True
        
        for i in range(self.num_D):
            D = getattr(self, f'scale_{i}')
            result = D(input)
            results.append(result)
            
            if i != (self.num_D - 1):
                input = F.avg_pool2d(input, kernel_size=3, stride=2, padding=1)
        
        return results

class NLayerDiscriminator(nn.Module):
    """N-Layer 판별자"""
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super().__init__()
        
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                         kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                     kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class AttentionModule(nn.Module):
    """어텐션 메커니즘"""
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        
        # Query, Key, Value 계산
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, H * W)
        proj_value = self.value_conv(x).view(B, -1, H * W)
        
        # 어텐션 맵 계산
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        # 어텐션 적용
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        # 잔차 연결
        out = self.gamma * out + x
        return out

class FeatureFusionModule(nn.Module):
    """특징 융합 모듈"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.attention = AttentionModule(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.attention(x)
        return x

class UNetGenerator(nn.Module):
    """U-Net 기반 생성자"""
    def __init__(self, input_nc=6, output_nc=3, num_downs=8, ngf=64):
        super().__init__()
        
        # 인코더 구축
        self.encoder = self._build_encoder(input_nc, ngf, num_downs)
        
        # 디코더 구축  
        self.decoder = self._build_decoder(ngf, num_downs, output_nc)
        
        # 특징 융합 모듈들
        self.fusion_modules = nn.ModuleList([
            FeatureFusionModule(ngf * (2**i), ngf * (2**i))
            for i in range(num_downs)
        ])

    def _build_encoder(self, input_nc, ngf, num_downs):
        """인코더 구축"""
        encoder = nn.ModuleList()
        
        # 첫 번째 레이어
        encoder.append(
            nn.Sequential(
                nn.Conv2d(input_nc, ngf, 4, 2, 1),
                nn.LeakyReLU(0.2, True)
            )
        )
        
        # 중간 레이어들
        for i in range(1, num_downs):
            mult = min(2**i, 8)
            encoder.append(
                nn.Sequential(
                    nn.Conv2d(ngf * mult//2, ngf * mult, 4, 2, 1),
                    nn.BatchNorm2d(ngf * mult),
                    nn.LeakyReLU(0.2, True)
                )
            )
        
        return encoder

    def _build_decoder(self, ngf, num_downs, output_nc):
        """디코더 구축"""
        decoder = nn.ModuleList()
        
        # 중간 레이어들
        for i in range(num_downs-1, 0, -1):
            mult = min(2**i, 8)
            decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(ngf * mult * 2, ngf * mult//2, 4, 2, 1),
                    nn.BatchNorm2d(ngf * mult//2),
                    nn.ReLU(True),
                    nn.Dropout(0.5)
                )
            )
        
        # 마지막 레이어
        decoder.append(
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 2, output_nc, 4, 2, 1),
                nn.Tanh()
            )
        )
        
        return decoder

    def forward(self, x):
        # 인코딩
        encoder_features = []
        current = x
        
        for i, encoder_layer in enumerate(self.encoder):
            current = encoder_layer(current)
            encoder_features.append(current)
        
        # 디코딩 (skip connection 포함)
        current = encoder_features[-1]
        
        for i, decoder_layer in enumerate(self.decoder[:-1]):
            current = decoder_layer(current)
            
            # Skip connection
            skip_idx = len(encoder_features) - 2 - i
            if skip_idx >= 0:
                skip_feature = encoder_features[skip_idx]
                
                # 특징 융합
                if i < len(self.fusion_modules):
                    fused = torch.cat([current, skip_feature], dim=1)
                    current = self.fusion_modules[i](fused)
                else:
                    current = torch.cat([current, skip_feature], dim=1)
        
        # 최종 출력
        output = self.decoder[-1](current)
        return output

class FlowEstimationNetwork(nn.Module):
    """플로우 추정 네트워크"""
    def __init__(self, input_channels=6):
        super().__init__()
        
        # 특징 추출기
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 플로우 추정 헤드
        self.flow_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 3, padding=1)  # x, y 플로우
        )

    def forward(self, person_img, cloth_img):
        # 입력 결합
        combined = torch.cat([person_img, cloth_img], dim=1)
        
        # 특징 추출
        features = self.feature_extractor(combined)
        
        # 플로우 추정
        flow = self.flow_head(features)
        
        # 원본 해상도로 업샘플링
        flow = F.interpolate(flow, size=person_img.shape[2:], 
                           mode='bilinear', align_corners=True)
        
        return flow

class VirtualTryOnGenerator(nn.Module):
    """메인 가상 피팅 생성자"""
    def __init__(self, input_nc=9, output_nc=3):
        super().__init__()
        
        # 의류 변형 네트워크
        self.cloth_warp_net = UNetGenerator(input_nc=6, output_nc=2)  # 워핑 필드
        
        # 플로우 추정 네트워크
        self.flow_net = FlowEstimationNetwork(input_channels=6)
        
        # 최종 합성 네트워크
        self.synthesis_net = UNetGenerator(input_nc=input_nc, output_nc=output_nc)
        
        # 마스크 예측 네트워크
        self.mask_net = UNetGenerator(input_nc=6, output_nc=1)

    def warp_cloth(self, cloth, flow):
        """플로우를 사용하여 의류 워핑"""
        B, C, H, W = cloth.shape
        
        # 그리드 생성
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).to(cloth.device)
        grid = grid.repeat(B, 1, 1, 1)
        
        # 플로우 적용
        warped_grid = grid + flow
        
        # 정규화 [-1, 1]
        warped_grid[:, 0] = 2.0 * warped_grid[:, 0] / (W - 1) - 1.0
        warped_grid[:, 1] = 2.0 * warped_grid[:, 1] / (H - 1) - 1.0
        
        # 차원 재정렬 [B, H, W, 2]
        warped_grid = warped_grid.permute(0, 2, 3, 1)
        
        # 그리드 샘플링
        warped_cloth = F.grid_sample(cloth, warped_grid, align_corners=True)
        
        return warped_cloth

    def forward(self, person_img, cloth_img, person_parse, cloth_parse, pose_map):
        """전체 파이프라인 실행"""
        B, C, H, W = person_img.shape
        
        # 1. 플로우 추정
        flow = self.flow_net(person_img, cloth_img)
        
        # 2. 의류 워핑
        warped_cloth = self.warp_cloth(cloth_img, flow)
        
        # 3. 워핑 마스크 생성
        cloth_mask_input = torch.cat([person_img, cloth_img], dim=1)
        warp_mask = torch.sigmoid(self.mask_net(cloth_mask_input))
        
        # 4. 최종 합성
        synthesis_input = torch.cat([
            person_img,      # 원본 사람 이미지
            warped_cloth,    # 워핑된 의류
            person_parse,    # 사람 파싱 맵
            cloth_parse,     # 의류 파싱 맵  
            pose_map,        # 포즈 맵
            warp_mask        # 워핑 마스크
        ], dim=1)
        
        try_on_result = self.synthesis_net(synthesis_input)
        
        return {
            'try_on_image': try_on_result,
            'warped_cloth': warped_cloth,
            'warp_mask': warp_mask,
            'flow': flow
        }

class VirtualFittingStep:
    """6단계: 가상 피팅 생성 실행 클래스"""
    
    def __init__(self, config, device, model_loader):
        self.config = config
        self.device = device
        self.model_loader = model_loader
        self.logger = logging.getLogger(__name__)
        
        # 모델 초기화
        self.generator = None
        self.discriminator = None
        self.model_loaded = False
        
        # 손실 함수 가중치
        self.lambda_l1 = 10.0
        self.lambda_perceptual = 5.0
        self.lambda_adversarial = 1.0
        self.lambda_flow = 1.0

    async def process(self, input_data: Dict[str, Any]) -> torch.Tensor:
        """가상 피팅 메인 처리"""
        try:
            # 모델 로드 (필요시)
            if not self.model_loaded:
                await self._load_models()
            
            # 입력 데이터 준비
            processed_input = await self._prepare_input(input_data)
            
            # 가상 피팅 생성
            with torch.no_grad():
                result = await self._generate_try_on(processed_input)
            
            # 후처리
            final_result = await self._postprocess(result)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"가상 피팅 처리 중 오류: {str(e)}")
            raise

    async def _load_models(self):
        """모델 로드 및 초기화"""
        try:
            # 캐시된 모델 확인
            cached_generator = self.model_loader.memory_manager.get_cached_model("virtual_fitting_gen")
            
            if cached_generator is not None:
                self.generator = cached_generator
                self.logger.info("캐시된 가상 피팅 생성자 로드")
            else:
                # 새 생성자 모델 생성
                self.generator = VirtualTryOnGenerator(input_nc=15, output_nc=3)
                
                # 사전 훈련된 가중치 로드
                checkpoint_path = Path("models/checkpoints/virtual_fitting_generator.pth")
                if checkpoint_path.exists():
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    self.generator.load_state_dict(checkpoint['generator_state_dict'])
                    self.logger.info("생성자 가중치 로드 완료")
                else:
                    self.logger.warning("생성자 가중치를 찾을 수 없습니다. 랜덤 초기화 사용")
                
                # 모델을 디바이스로 이동
                self.generator = self.generator.to(self.device)
                
                # FP16 최적화
                if self.config.use_fp16 and self.device.type == "mps":
                    self.generator = self.generator.half()
                
                # 평가 모드
                self.generator.eval()
                
                # 모델 캐싱
                self.model_loader.memory_manager.cache_model("virtual_fitting_gen", self.generator)
            
            self.model_loaded = True
            self.logger.info("가상 피팅 모델 로드 완료")
            
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {str(e)}")
            raise

    async def _prepare_input(self, input_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """입력 데이터 준비 및 전처리"""
        person_image = input_data["person_image"]      # [B, 3, H, W]
        human_parsing = input_data["human_parsing"]    # Dict
        pose_keypoints = input_data["pose_keypoints"]  # Dict
        warped_cloth = input_data["warped_cloth"]      # [B, 3, H, W]
        cloth_mask = input_data["cloth_mask"]          # [B, 1, H, W]
        
        # 파싱 맵을 텐서로 변환
        if isinstance(human_parsing["parsing_map"], np.ndarray):
            parsing_tensor = torch.from_numpy(human_parsing["parsing_map"]).unsqueeze(0).unsqueeze(0).float()
            parsing_tensor = parsing_tensor.to(self.device)
        else:
            parsing_tensor = human_parsing["parsing_map"]
        
        # 포즈 맵 생성
        pose_map = self._create_pose_map(pose_keypoints, person_image.shape[2:])
        
        # 의류 파싱 맵 (간단한 버전)
        cloth_parse = cloth_mask  # 의류 마스크를 파싱 맵으로 사용
        
        # 정규화 확인
        if person_image.max() > 1.0:
            person_image = person_image / 255.0
        if warped_cloth.max() > 1.0:
            warped_cloth = warped_cloth / 255.0
        
        # FP16 변환 (필요시)
        if self.config.use_fp16 and self.device.type == "mps":
            person_image = person_image.half()
            warped_cloth = warped_cloth.half()
            parsing_tensor = parsing_tensor.half()
            pose_map = pose_map.half()
            cloth_parse = cloth_parse.half()
        
        return {
            "person_img": person_image,
            "cloth_img": warped_cloth,
            "person_parse": parsing_tensor,
            "cloth_parse": cloth_parse,
            "pose_map": pose_map
        }

    def _create_pose_map(self, pose_keypoints: Dict[str, Any], image_size: Tuple[int, int]) -> torch.Tensor:
        """포즈 키포인트를 히트맵으로 변환"""
        H, W = image_size
        pose_map = torch.zeros(1, 18, H, W).to(self.device)
        
        if "keypoints" in pose_keypoints:
            keypoints = pose_keypoints["keypoints"]
            
            # 각 키포인트를 가우시안 히트맵으로 변환
            for i, (x, y, confidence) in enumerate(keypoints):
                if confidence > 0.5 and 0 <= x < W and 0 <= y < H:
                    # 가우시안 커널 생성
                    sigma = 4
                    kernel_size = 4 * sigma + 1
                    
                    # 키포인트 주변 영역에 가우시안 적용
                    x_min = max(0, int(x) - kernel_size // 2)
                    x_max = min(W, int(x) + kernel_size // 2 + 1)
                    y_min = max(0, int(y) - kernel_size // 2)
                    y_max = min(H, int(y) + kernel_size // 2 + 1)
                    
                    # 가우시안 값 계산
                    for py in range(y_min, y_max):
                        for px in range(x_min, x_max):
                            dist_sq = (px - x) ** 2 + (py - y) ** 2
                            value = confidence * math.exp(-dist_sq / (2 * sigma ** 2))
                            pose_map[0, i, py, px] = max(pose_map[0, i, py, px], value)
        
        return pose_map

    async def _generate_try_on(self, input_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """가상 피팅 이미지 생성"""
        # 메모리 효율적 추론
        if self.config.use_fp16 and self.device.type == "mps":
            with torch.autocast(device_type='cpu'):  # MPS autocast 제한
                result = self.generator(
                    input_data["person_img"],
                    input_data["cloth_img"],
                    input_data["person_parse"],
                    input_data["cloth_parse"],
                    input_data["pose_map"]
                )
        else:
            result = self.generator(
                input_data["person_img"],
                input_data["cloth_img"],
                input_data["person_parse"],
                input_data["cloth_parse"],
                input_data["pose_map"]
            )
        
        return result

    async def _postprocess(self, generation_result: Dict[str, torch.Tensor]) -> torch.Tensor:
        """생성 결과 후처리"""
        try_on_image = generation_result["try_on_image"]
        
        # 값 범위 정규화 [-1, 1] → [0, 1]
        try_on_image = (try_on_image + 1.0) / 2.0
        try_on_image = torch.clamp(try_on_image, 0.0, 1.0)
        
        # 추가 품질 향상 (옵션)
        if hasattr(self, 'enhance_quality') and self.enhance_quality:
            try_on_image = self._enhance_image_quality(try_on_image)
        
        return try_on_image

    def _enhance_image_quality(self, image: torch.Tensor) -> torch.Tensor:
        """이미지 품질 향상"""
        # 간단한 샤프닝 필터 적용
        kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(image.device)
        
        # 각 채널에 적용
        enhanced = torch.zeros_like(image)
        for c in range(image.shape[1]):
            enhanced[:, c:c+1] = F.conv2d(
                image[:, c:c+1], kernel, padding=1
            )
        
        # 원본과 블렌딩
        enhanced = 0.8 * image + 0.2 * enhanced
        enhanced = torch.clamp(enhanced, 0.0, 1.0)
        
        return enhanced

    def calculate_fitting_quality(
        self, 
        try_on_result: torch.Tensor,
        original_person: torch.Tensor,
        warped_cloth: torch.Tensor
    ) -> Dict[str, float]:
        """피팅 품질 평가"""
        
        # SSIM 계산 (구조적 유사성)
        ssim_score = self._calculate_ssim(try_on_result, original_person)
        
        # 의류 보존 정도 평가
        cloth_preservation = self._evaluate_cloth_preservation(try_on_result, warped_cloth)
        
        # 자연스러움 평가 (gradient magnitude 기반)
        naturalness = self._evaluate_naturalness(try_on_result)
        
        # 전체 피팅 점수 계산
        overall_score = (ssim_score * 0.4 + cloth_preservation * 0.4 + naturalness * 0.2)
        
        return {
            "ssim_score": float(ssim_score),
            "cloth_preservation": float(cloth_preservation),
            "naturalness": float(naturalness),
            "overall_score": float(overall_score)
        }

    def _calculate_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """SSIM 계산 (간단한 버전)"""
        mu1 = F.avg_pool2d(img1, 3, 1, 1)
        mu2 = F.avg_pool2d(img2, 3, 1, 1)
        
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()

    def _evaluate_cloth_preservation(self, try_on_result: torch.Tensor, warped_cloth: torch.Tensor) -> torch.Tensor:
        """의류 보존 정도 평가"""
        # 의류 영역에서의 색상 일치도 계산
        color_diff = torch.abs(try_on_result - warped_cloth)
        preservation_score = 1.0 - color_diff.mean()
        return preservation_score

    def _evaluate_naturalness(self, image: torch.Tensor) -> torch.Tensor:
        """자연스러움 평가 (그래디언트 기반)"""
        # 소벨 필터로 그래디언트 계산
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(image.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(image.device)
        
        # 각 채널의 그래디언트 계산
        grad_x = F.conv2d(image.mean(dim=1, keepdim=True), sobel_x, padding=1)
        grad_y = F.conv2d(image.mean(dim=1, keepdim=True), sobel_y, padding=1)
        
        # 그래디언트 크기
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # 자연스러운 이미지는 적당한 그래디언트를 가져야 함
        optimal_gradient = 0.1
        naturalness = 1.0 - torch.abs(grad_magnitude.mean() - optimal_gradient)
        
        return torch.clamp(naturalness, 0.0, 1.0)

    async def warmup(self, dummy_input: torch.Tensor):
        """모델 워밍업"""
        if not self.model_loaded:
            await self._load_models()
        
        # 더미 입력 생성
        dummy_data = {
            "person_img": dummy_input,
            "cloth_img": dummy_input,
            "person_parse": torch.zeros(1, 1, dummy_input.shape[2], dummy_input.shape[3]).to(self.device),
            "cloth_parse": torch.zeros(1, 1, dummy_input.shape[2], dummy_input.shape[3]).to(self.device),
            "pose_map": torch.zeros(1, 18, dummy_input.shape[2], dummy_input.shape[3]).to(self.device)
        }
        
        with torch.no_grad():
            _ = await self._generate_try_on(dummy_data)
        
        self.logger.info("가상 피팅 모델 워밍업 완료")

    def cleanup(self):
        """리소스 정리"""
        if self.generator is not None:
            del self.generator
            self.generator = None
        
        if self.discriminator is not None:
            del self.discriminator
            self.discriminator = None
            
        self.model_loaded = False
        self.logger.info("가상 피팅 모델 리소스 정리 완료")

# 사용 예시
async def example_usage():
    """가상 피팅 사용 예시"""
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
    
    # 가상 피팅 단계 초기화
    virtual_fitting = VirtualFittingStep(config, device, model_loader)
    
    # 더미 입력 생성
    dummy_person = torch.randn(1, 3, 512, 512).to(device)
    dummy_cloth = torch.randn(1, 3, 512, 512).to(device)
    
    input_data = {
        "person_image": dummy_person,
        "human_parsing": {
            "parsing_map": np.random.randint(0, 20, (512, 512)),
        },
        "pose_keypoints": {
            "keypoints": [(100, 100, 0.9), (150, 120, 0.8)]  # 예시 키포인트
        },
        "warped_cloth": dummy_cloth,
        "cloth_mask": torch.ones(1, 1, 512, 512).to(device)
    }
    
    # 처리
    result = await virtual_fitting.process(input_data)
    
    print(f"가상 피팅 완료 - 출력 크기: {result.shape}")
    
    # 품질 평가
    quality_scores = virtual_fitting.calculate_fitting_quality(
        result, dummy_person, dummy_cloth
    )
    
    print("품질 점수:")
    for metric, score in quality_scores.items():
        print(f"  {metric}: {score:.3f}")

if __name__ == "__main__":
    asyncio.run(example_usage())