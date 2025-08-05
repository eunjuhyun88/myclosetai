#!/usr/bin/env python3
"""
🔥 MyCloset AI - Geometric Matching Utility
==========================================

기하학적 매칭 관련 모듈들
- GeometricMatchingModule: 기하학적 매칭 모듈
- FlowPredictor: Flow field 예측
- WarpingModule: 이미지 워핑

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
    raise ImportError("PyTorch is required for geometric matching")

class GeometricMatchingModule(nn.Module):
    """기하학적 매칭 모듈"""
    
    def __init__(self):
        super().__init__()
        
        # Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(6, 64, 4, stride=2, padding=1),  # person + cloth
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        
        # Flow Predictor
        self.flow_predictor = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 2, 3, padding=1),  # 2D flow
            nn.Tanh()
        )
        
        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(2, 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(2, 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(2, 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(2, 2, 3, padding=1)
        )
    
    def forward(self, person_img, cloth_img, person_parse=None):
        # Concatenate inputs
        x = torch.cat([person_img, cloth_img], dim=1)
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Predict flow
        flow = self.flow_predictor(features)
        flow = self.upsample(flow)
        
        # Warp cloth using flow
        warped_cloth = self.warp_cloth(cloth_img, flow)
        
        return warped_cloth, flow
    
    def warp_cloth(self, cloth, flow):
        """Flow를 사용한 의류 워핑"""
        b, c, h, w = cloth.shape
        
        # Create grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=cloth.device),
            torch.linspace(-1, 1, w, device=cloth.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(b, 1, 1, 1)
        
        # Apply flow
        flow_norm = flow / torch.tensor([w, h], device=flow.device).view(1, 2, 1, 1) * 2
        warped_grid = grid + flow_norm
        warped_grid = warped_grid.permute(0, 2, 3, 1)
        
        # Sample warped cloth
        warped_cloth = F.grid_sample(cloth, warped_grid, align_corners=True)
        
        return warped_cloth

class FlowPredictor(nn.Module):
    """Flow field 예측 모듈"""
    
    def __init__(self, in_channels=512, out_channels=2):
        super().__init__()
        
        self.flow_predictor = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, features):
        return self.flow_predictor(features)

class WarpingModule(nn.Module):
    """이미지 워핑 모듈"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, image, flow_field):
        """Flow field를 사용한 이미지 워핑"""
        b, c, h, w = image.shape
        
        # Create normalized grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=image.device),
            torch.linspace(-1, 1, w, device=image.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(b, 1, 1, 1)
        
        # Apply flow
        flow_norm = flow_field / torch.tensor([w, h], device=flow_field.device).view(1, 2, 1, 1) * 2
        warped_grid = grid + flow_norm
        warped_grid = warped_grid.permute(0, 2, 3, 1)
        
        # Sample warped image
        warped_image = F.grid_sample(image, warped_grid, align_corners=True, mode='bilinear')
        
        return warped_image

class DeformableConv2d(nn.Module):
    """Deformable Convolution 2D"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Regular convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
        # Offset prediction
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, 3, padding=1)
        
    def forward(self, x):
        # Predict offsets
        offset = self.offset_conv(x)
        
        # Apply deformable convolution
        # Note: This is a simplified version. Full implementation would require
        # custom CUDA kernels or using existing libraries like torchvision.ops
        return self.conv(x)

class SpatialTransformer(nn.Module):
    """공간 변환 모듈"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )
        
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        
        return x 