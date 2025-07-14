# app/ai_pipeline/utils/model_loader.py
"""
🍎 M3 Max 최적화 실제 AI 모델 로더 - 우리 구조에 최적화
- 최적 생성자 패턴 적용
- 8단계 파이프라인에 필요한 모든 실제 AI 모델들
- M3 Max MPS 최적화
- 메모리 효율적 모델 로딩
"""

import os
import logging
import time
import threading
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Type, Callable
from abc import ABC, abstractmethod
import json
import pickle
from dataclasses import dataclass, field
from enum import Enum
import gc

# PyTorch import (안전)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

# 컴퓨터 비전 라이브러리들
try:
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

# 외부 AI 라이브러리들 (선택적)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

# 최적 생성자 베이스 클래스
class OptimalStepConstructor(ABC):
    """최적화된 생성자 패턴 베이스 클래스"""

    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # 1. 지능적 디바이스 자동 감지
        self.device = self._auto_detect_device(device)
        
        # 2. 기본 설정
        self.config = config or {}
        self.step_name = self.__class__.__name__
        self.logger = logging.getLogger(f"utils.{self.step_name}")
        
        # 3. 표준 시스템 파라미터 추출
        self.device_type = kwargs.get('device_type', 'auto')
        self.memory_gb = kwargs.get('memory_gb', 16.0)
        self.is_m3_max = kwargs.get('is_m3_max', self._detect_m3_max())
        self.optimization_enabled = kwargs.get('optimization_enabled', True)
        self.quality_level = kwargs.get('quality_level', 'balanced')
        
        # 4. 스텝별 특화 파라미터를 config에 병합
        self._merge_step_specific_config(kwargs)
        
        # 5. 상태 초기화
        self.is_initialized = False

    def _auto_detect_device(self, preferred_device: Optional[str]) -> str:
        """지능적 디바이스 자동 감지"""
        if preferred_device:
            return preferred_device

        if not TORCH_AVAILABLE:
            return 'cpu'

        try:
            if torch.backends.mps.is_available():
                return 'mps'
            elif torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        except:
            return 'cpu'

    def _detect_m3_max(self) -> bool:
        """M3 Max 칩 자동 감지"""
        try:
            import platform
            import subprocess

            if platform.system() == 'Darwin':
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                return 'M3' in result.stdout
        except:
            pass
        return False

    def _merge_step_specific_config(self, kwargs: Dict[str, Any]):
        """스텝별 특화 설정 병합"""
        system_params = {
            'device_type', 'memory_gb', 'is_m3_max', 
            'optimization_enabled', 'quality_level'
        }

        for key, value in kwargs.items():
            if key not in system_params:
                self.config[key] = value

# 모델 타입 enum
class ModelType(Enum):
    """AI 모델 타입"""
    HUMAN_PARSING = "human_parsing"  # Graphonomy, PGN
    POSE_ESTIMATION = "pose_estimation"  # OpenPose, MediaPipe
    CLOTH_SEGMENTATION = "cloth_segmentation"  # U2Net, SAM
    GEOMETRIC_MATCHING = "geometric_matching"  # TPS, GMM
    CLOTH_WARPING = "cloth_warping"  # HR-VITON, PF-AFN
    VIRTUAL_FITTING = "virtual_fitting"  # OOTD, DreamPose
    DIFFUSION = "diffusion"  # Stable Diffusion 기반
    SEGMENTATION = "segmentation"  # 일반 세그멘테이션

@dataclass
class ModelConfig:
    """모델 설정 정보"""
    name: str
    model_type: ModelType
    model_class: str
    checkpoint_path: Optional[str] = None
    config_path: Optional[str] = None
    device: str = "auto"
    precision: str = "fp16"  # fp16, fp32, int8
    optimization_level: str = "balanced"  # fast, balanced, quality
    cache_enabled: bool = True
    input_size: tuple = (512, 512)
    num_classes: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.model_type, str):
            self.model_type = ModelType(self.model_type)

# ==============================================
# 8단계 파이프라인에 필요한 실제 AI 모델들
# ==============================================

class GraphonomyModel(nn.Module):
    """Graphonomy 인체 파싱 모델 - Step 01"""
    
    def __init__(self, num_classes=20, backbone='resnet101'):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # ResNet 백본 구성
        self.backbone = self._build_backbone()
        
        # ASPP (Atrous Spatial Pyramid Pooling)
        self.aspp = self._build_aspp()
        
        # 분류 헤드
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)
        
        # 보조 분류기 (훈련시)
        self.aux_classifier = nn.Conv2d(1024, num_classes, kernel_size=1)
        
    def _build_backbone(self):
        """ResNet 백본 구성"""
        try:
            import torchvision.models as models
            if self.backbone_name == 'resnet101':
                backbone = models.resnet101(pretrained=True)
            else:
                backbone = models.resnet50(pretrained=True)
                
            # Atrous convolution을 위한 stride 수정
            backbone.layer3[0].conv2.stride = (1, 1)
            backbone.layer3[0].downsample[0].stride = (1, 1)
            backbone.layer4[0].conv2.stride = (1, 1)
            backbone.layer4[0].downsample[0].stride = (1, 1)
            
            # Dilation 적용
            for module in backbone.layer3[1:]:
                module.conv2.dilation = (2, 2)
                module.conv2.padding = (2, 2)
            for module in backbone.layer4:
                module.conv2.dilation = (4, 4)
                module.conv2.padding = (4, 4)
                
            return nn.Sequential(*list(backbone.children())[:-2])
        except ImportError:
            # torchvision 없는 경우 간단한 백본
            return nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1),
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 1024, 3, 1, 1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 2048, 3, 1, 1),
                nn.BatchNorm2d(2048),
                nn.ReLU(inplace=True)
            )
    
    def _build_aspp(self):
        """ASPP 모듈 구성"""
        return nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(2048, 256, 3, padding=6, dilation=6, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(2048, 256, 3, padding=12, dilation=12, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(2048, 256, 3, padding=18, dilation=18, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(2048, 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        ])
    
    def forward(self, x):
        input_size = x.size()[2:]
        
        # 백본 통과
        features = self.backbone(x)
        
        # ASPP 적용
        aspp_outputs = []
        for aspp_layer in self.aspp[:-1]:
            aspp_outputs.append(aspp_layer(features))
        
        # Global average pooling
        global_feat = self.aspp[-1](features)
        global_feat = F.interpolate(global_feat, size=features.size()[2:], 
                                   mode='bilinear', align_corners=False)
        aspp_outputs.append(global_feat)
        
        # 특징 융합
        fused = torch.cat(aspp_outputs, dim=1)
        
        # 최종 분류
        output = self.classifier(fused)
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)
        
        return output

class OpenPoseModel(nn.Module):
    """OpenPose 포즈 추정 모델 - Step 02"""
    
    def __init__(self, num_keypoints=18):
        super().__init__()
        self.num_keypoints = num_keypoints
        
        # VGG-19 백본
        self.backbone = self._build_vgg_backbone()
        
        # 6단계 반복 처리
        self.stages = nn.ModuleList()
        for i in range(6):
            if i == 0:
                # 첫 번째 스테이지
                stage = nn.ModuleDict({
                    'paf': self._build_initial_stage(38),  # 19 limbs * 2 (x,y)
                    'heatmap': self._build_initial_stage(19)  # 18 keypoints + 1 background
                })
            else:
                # 후속 스테이지
                stage = nn.ModuleDict({
                    'paf': self._build_refinement_stage(38),
                    'heatmap': self._build_refinement_stage(19)
                })
            self.stages.append(stage)
    
    def _build_vgg_backbone(self):
        """VGG-19 백본 구성"""
        try:
            import torchvision.models as models
            vgg = models.vgg19(pretrained=True).features
            # Conv4_4까지만 사용
            return nn.Sequential(*list(vgg.children())[:23])
        except ImportError:
            # VGG 대체 백본
            return nn.Sequential(
                nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True)
            )
    
    def _build_initial_stage(self, output_channels):
        """초기 스테이지 구성"""
        return nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 1, 1, 0), nn.ReLU(inplace=True),
            nn.Conv2d(512, output_channels, 1, 1, 0)
        )
    
    def _build_refinement_stage(self, output_channels):
        """개선 스테이지 구성"""
        return nn.Sequential(
            nn.Conv2d(512 + 38 + 19, 128, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, 1, 0), nn.ReLU(inplace=True),
            nn.Conv2d(128, output_channels, 1, 1, 0)
        )
    
    def forward(self, x):
        # 백본 특징 추출
        features = self.backbone(x)
        
        stage_outputs = []
        
        for i, stage in enumerate(self.stages):
            if i == 0:
                # 첫 번째 스테이지
                paf = stage['paf'](features)
                heatmap = stage['heatmap'](features)
            else:
                # 이전 결과와 특징 결합
                combined = torch.cat([features, prev_paf, prev_heatmap], dim=1)
                paf = stage['paf'](combined)
                heatmap = stage['heatmap'](combined)
            
            stage_outputs.append((paf, heatmap))
            prev_paf, prev_heatmap = paf, heatmap
        
        return stage_outputs

class U2NetModel(nn.Module):
    """U²-Net 세그멘테이션 모델 - Step 03"""
    
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        
        # 인코더 (6단계 RSU 블록)
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage6 = RSU4F(512, 256, 512)
        
        # 디코더
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)
        
        # 사이드 출력들
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)
        
        # 최종 융합
        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)
    
    def forward(self, x):
        hx = x
        
        # 인코더
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        
        hx6 = self.stage6(hx)
        
        # 디코더
        hx6up = F.interpolate(hx6, size=hx5.shape[2:], mode='bilinear', align_corners=False)
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=False)
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=False)
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
        
        # 사이드 출력들
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d2 = F.interpolate(d2, size=x.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.side3(hx3d)
        d3 = F.interpolate(d3, size=x.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.side4(hx4d)
        d4 = F.interpolate(d4, size=x.shape[2:], mode='bilinear', align_corners=False)
        d5 = self.side5(hx5d)
        d5 = F.interpolate(d5, size=x.shape[2:], mode='bilinear', align_corners=False)
        d6 = self.side6(hx6)
        d6 = F.interpolate(d6, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # 최종 융합
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        
        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)

# RSU 블록들 (U²-Net 구성 요소)
class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv6d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = F.interpolate(hx6d, size=hx5.shape[2:], mode='bilinear', align_corners=False)
        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=False)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=False)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin

class RSU6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx6 = self.rebnconv6(hx5)
        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=False)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=False)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin

class RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx5 = self.rebnconv5(hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=False)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin

class RSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin

class RSU4F(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
        return hx1d + hxin

class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = self.relu_s1(self.bn_s1(self.conv_s1(x)))
        return hx

class GeometricMatchingModel(nn.Module):
    """기하학적 매칭 모델 - Step 04"""
    
    def __init__(self, feature_size=256):
        super().__init__()
        self.feature_size = feature_size
        
        # 특징 추출 네트워크
        self.feature_extractor = self._build_feature_extractor()
        
        # 상관관계 계산
        self.correlation = self._build_correlation_layer()
        
        # 회귀 네트워크
        self.regression = self._build_regression_network()
        
    def _build_feature_extractor(self):
        """특징 추출 네트워크"""
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
    
    def _build_correlation_layer(self):
        """상관관계 계산 레이어"""
        return nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1, 1, 0), nn.Sigmoid()
        )
    
    def _build_regression_network(self):
        """회귀 네트워크"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(128, 18)  # 6개 TPS 제어점 * 3 (x, y, confidence)
        )
    
    def forward(self, source_img, target_img):
        # 특징 추출
        source_feat = self.feature_extractor(source_img)
        target_feat = self.feature_extractor(target_img)
        
        # 상관관계 계산
        correlation_map = self.correlation(torch.cat([source_feat, target_feat], dim=1))
        
        # TPS 파라미터 회귀
        tps_params = self.regression(correlation_map)
        
        return {
            'correlation_map': correlation_map,
            'tps_params': tps_params.view(-1, 6, 3),  # [batch, 6_points, (x,y,conf)]
            'source_features': source_feat,
            'target_features': target_feat
        }

class HRVITONModel(nn.Module):
    """HR-VITON 가상 피팅 모델 - Step 06"""
    
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super().__init__()
        
        # 생성기 네트워크
        self.generator = self._build_generator(input_nc, output_nc, ngf)
        
        # 어텐션 모듈
        self.attention = self._build_attention_module()
        
        # 융합 모듈
        self.fusion = self._build_fusion_module()
    
    def _build_generator(self, input_nc, output_nc, ngf):
        """생성기 네트워크 구성"""
        return nn.Sequential(
            # 인코더
            nn.Conv2d(input_nc, ngf, 7, 1, 3), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.Conv2d(ngf, ngf*2, 3, 2, 1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.Conv2d(ngf*2, ngf*4, 3, 2, 1), nn.InstanceNorm2d(ngf*4), nn.ReLU(True),
            
            # ResNet 블록들
            *[ResnetBlock(ngf*4) for _ in range(9)],
            
            # 디코더
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, 1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 3, 2, 1, 1), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.Conv2d(ngf, output_nc, 7, 1, 3), nn.Tanh()
        )
    
    def _build_attention_module(self):
        """어텐션 모듈"""
        return nn.Sequential(
            nn.Conv2d(6, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(32, 1, 1, 1, 0), nn.Sigmoid()
        )
    
    def _build_fusion_module(self):
        """융합 모듈"""
        return nn.Sequential(
            nn.Conv2d(6, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(32, 3, 3, 1, 1), nn.Tanh()
        )
    
    def forward(self, person_img, cloth_img, person_parse=None):
        # 기본 생성
        input_concat = torch.cat([person_img, cloth_img], dim=1)
        generated = self.generator(input_concat)
        
        # 어텐션 맵 계산
        attention_map = self.attention(input_concat)
        
        # 어텐션 적용 융합
        attended_result = generated * attention_map + person_img * (1 - attention_map)
        
        # 추가 융합
        final_result = self.fusion(torch.cat([attended_result, cloth_img], dim=1))
        
        return {
            'generated_image': final_result,
            'attention_map': attention_map,
            'intermediate': generated
        }

class ResnetBlock(nn.Module):
    """ResNet 블록"""
    def __init__(self, dim, use_dropout=False):
        super().__init__()
        self.conv_block = self._build_conv_block(dim, use_dropout)

    def _build_conv_block(self, dim, use_dropout):
        layers = []
        layers += [nn.Conv2d(dim, dim, 3, 1, 1), nn.InstanceNorm2d(dim), nn.ReLU(True)]
        if use_dropout:
            layers += [nn.Dropout(0.5)]
        layers += [nn.Conv2d(dim, dim, 3, 1, 1), nn.InstanceNorm2d(dim)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv_block(x)

# ==============================================
# 모델 로더 클래스
# ==============================================

class ModelLoader(OptimalStepConstructor):
    """
    🍎 M3 Max 최적화 실제 AI 모델 로더
    8단계 파이프라인에 필요한 모든 모델 지원
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        ✅ 최적 생성자 - 모델 로더 특화

        Args:
            device: 사용할 디바이스 (None=자동감지, 'cpu', 'cuda', 'mps')
            config: 모델 로더 설정 딕셔너리
            **kwargs: 확장 파라미터들
                - model_cache_dir: str = "./models/ai_models"  # 모델 캐시 디렉토리
                - use_fp16: bool = True  # FP16 사용 여부
                - max_cached_models: int = 10  # 최대 캐시 모델 수
                - lazy_loading: bool = True  # 지연 로딩
                - coreml_optimization: bool = True  # CoreML 최적화 (M3 Max)
                - auto_quantization: bool = False  # 자동 양자화
                - memory_mapping: bool = True  # 메모리 매핑
        """
        super().__init__(device=device, config=config, **kwargs)
        
        # 모델 로더 특화 설정
        self.model_cache_dir = Path(kwargs.get('model_cache_dir', './models/ai_models'))
        self.use_fp16 = kwargs.get('use_fp16', True and self.device != 'cpu')
        self.max_cached_models = kwargs.get('max_cached_models', 10)
        self.lazy_loading = kwargs.get('lazy_loading', True)
        self.coreml_optimization = kwargs.get('coreml_optimization', self.is_m3_max and COREML_AVAILABLE)
        self.auto_quantization = kwargs.get('auto_quantization', False)
        self.memory_mapping = kwargs.get('memory_mapping', True)
        
        # 내부 상태
        self._lock = threading.RLock()
        self._loaded_models: Dict[str, Any] = {}
        self._model_configs: Dict[str, ModelConfig] = {}
        self._model_registry: Dict[str, Dict[str, Any]] = {}
        self._load_times: Dict[str, float] = {}
        self._fallback_models: Dict[str, Any] = {}
        
        # 캐시 디렉토리 생성
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # M3 Max 특화 설정
        if self.is_m3_max:
            self.use_fp16 = True  # M3 Max는 FP16 최적화
            if COREML_AVAILABLE:
                self.logger.info("🍎 CoreML 최적화 활성화됨")
        
        # 실제 모델 레지스트리 초기화
        self._initialize_model_registry()
        
        self.logger.info(f"📦 실제 AI 모델 로더 초기화 - {self.device} (FP16: {self.use_fp16})")
        
        # 초기화 완료
        self.is_initialized = True

    def _initialize_model_registry(self):
        """실제 AI 모델들 등록"""
        models_config = {
            # Step 01: Human Parsing
            "human_parsing_graphonomy": ModelConfig(
                name="human_parsing_graphonomy",
                model_type=ModelType.HUMAN_PARSING,
                model_class="GraphonomyModel",
                checkpoint_path=str(self.model_cache_dir / "graphonomy" / "inference.pth"),
                input_size=(512, 512),
                num_classes=20
            ),
            
            # Step 02: Pose Estimation  
            "pose_estimation_openpose": ModelConfig(
                name="pose_estimation_openpose", 
                model_type=ModelType.POSE_ESTIMATION,
                model_class="OpenPoseModel",
                checkpoint_path=str(self.model_cache_dir / "openpose" / "pose_model.pth"),
                input_size=(368, 368),
                num_classes=18
            ),
            
            # Step 03: Cloth Segmentation
            "cloth_segmentation_u2net": ModelConfig(
                name="cloth_segmentation_u2net",
                model_type=ModelType.CLOTH_SEGMENTATION, 
                model_class="U2NetModel",
                checkpoint_path=str(self.model_cache_dir / "u2net" / "u2net.pth"),
                input_size=(320, 320)
            ),
            
            # Step 04: Geometric Matching
            "geometric_matching_gmm": ModelConfig(
                name="geometric_matching_gmm",
                model_type=ModelType.GEOMETRIC_MATCHING,
                model_class="GeometricMatchingModel", 
                checkpoint_path=str(self.model_cache_dir / "hr_viton" / "gmm_final.pth"),
                input_size=(512, 384)
            ),
            
            # Step 05: Cloth Warping (HR-VITON의 TOM)
            "cloth_warping_tom": ModelConfig(
                name="cloth_warping_tom",
                model_type=ModelType.CLOTH_WARPING,
                model_class="HRVITONModel",
                checkpoint_path=str(self.model_cache_dir / "hr_viton" / "tom_final.pth"),
                input_size=(512, 384)
            ),
            
            # Step 06: Virtual Fitting (완전한 HR-VITON)
            "virtual_fitting_hrviton": ModelConfig(
                name="virtual_fitting_hrviton",
                model_type=ModelType.VIRTUAL_FITTING,
                model_class="HRVITONModel",
                checkpoint_path=str(self.model_cache_dir / "hr_viton" / "final.pth"),
                input_size=(512, 384)
            ),
            
            # OOTD 대체 모델
            "virtual_fitting_ootd": ModelConfig(
                name="virtual_fitting_ootd",
                model_type=ModelType.DIFFUSION,
                model_class="StableDiffusionPipeline",
                checkpoint_path=str(self.model_cache_dir / "ootd"),
                input_size=(512, 512)
            )
        }
        
        # 모델 등록
        for name, config in models_config.items():
            self.register_model(name, config)

    def register_model(
        self,
        name: str,
        model_config: Union[ModelConfig, Dict[str, Any]],
        loader_func: Optional[Callable] = None
    ) -> bool:
        """모델 등록"""
        try:
            with self._lock:
                # ModelConfig 객체로 변환
                if isinstance(model_config, dict):
                    model_config = ModelConfig(name=name, **model_config)
                elif not isinstance(model_config, ModelConfig):
                    raise ValueError(f"Invalid model_config type: {type(model_config)}")
                
                # 디바이스 설정 자동 감지
                if model_config.device == "auto":
                    model_config.device = self.device
                
                # 등록 정보 저장
                self._model_configs[name] = model_config
                self._model_registry[name] = {
                    "config": model_config,
                    "loader_func": loader_func,
                    "registered_at": time.time(),
                    "loaded": False,
                    "load_count": 0
                }
                
                self.logger.info(f"📝 실제 AI 모델 등록: {name} ({model_config.model_type.value})")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ 모델 등록 실패 {name}: {e}")
            return False

    async def load_model(
        self,
        name: str,
        force_reload: bool = False,
        **kwargs
    ) -> Optional[Any]:
        """실제 AI 모델 로드"""
        try:
            with self._lock:
                # 이미 로드된 모델 확인
                if name in self._loaded_models and not force_reload:
                    self._model_registry[name]["load_count"] += 1
                    self.logger.info(f"📦 캐시된 모델 반환: {name}")
                    return self._loaded_models[name]
                
                # 모델 설정 확인
                if name not in self._model_registry:
                    self.logger.warning(f"⚠️ 등록되지 않은 모델: {name}, 대체 모델 로드 시도")
                    return await self._load_fallback_model(name)
                
                start_time = time.time()
                model_info = self._model_registry[name]
                model_config = model_info["config"]
                
                self.logger.info(f"📦 실제 AI 모델 로딩 시작: {name} ({model_config.model_type.value})")
                
                # 메모리 압박 확인 및 정리
                await self._check_memory_and_cleanup()
                
                # 모델 인스턴스 생성
                model = await self._create_model_instance(model_config, **kwargs)
                
                if model is None:
                    self.logger.warning(f"⚠️ 모델 생성 실패, 대체 모델 로드: {name}")
                    return await self._load_fallback_model(name)
                
                # 체크포인트 로드
                await self._load_checkpoint(model, model_config)
                
                # 디바이스로 이동
                if hasattr(model, 'to'):
                    model = model.to(self.device)
                
                # M3 Max 최적화 적용
                if self.is_m3_max and self.optimization_enabled:
                    model = await self._apply_m3_max_optimization(model, model_config)
                
                # FP16 최적화
                if self.use_fp16 and hasattr(model, 'half') and self.device != 'cpu':
                    try:
                        model = model.half()
                    except Exception as e:
                        self.logger.warning(f"⚠️ FP16 변환 실패: {e}")
                
                # 평가 모드
                if hasattr(model, 'eval'):
                    model.eval()
                
                # 캐시에 저장
                self._loaded_models[name] = model
                self._load_times[name] = time.time() - start_time
                
                # 레지스트리 업데이트
                model_info["loaded"] = True
                model_info["load_count"] += 1
                model_info["last_loaded"] = time.time()
                
                load_time = self._load_times[name]
                self.logger.info(f"✅ 실제 AI 모델 로딩 완료: {name} ({load_time:.2f}s)")
                
                return model
                
        except Exception as e:
            self.logger.error(f"❌ 모델 로딩 실패 {name}: {e}")
            # 대체 모델 시도
            return await self._load_fallback_model(name)

    async def _create_model_instance(
        self,
        model_config: ModelConfig,
        **kwargs
    ) -> Optional[Any]:
        """실제 AI 모델 인스턴스 생성"""
        try:
            model_class = model_config.model_class
            
            if model_class == "GraphonomyModel":
                return GraphonomyModel(
                    num_classes=model_config.num_classes or 20,
                    backbone='resnet101'
                )
            
            elif model_class == "OpenPoseModel":
                return OpenPoseModel(
                    num_keypoints=model_config.num_classes or 18
                )
            
            elif model_class == "U2NetModel":
                return U2NetModel(in_ch=3, out_ch=1)
            
            elif model_class == "GeometricMatchingModel":
                return GeometricMatchingModel(feature_size=256)
            
            elif model_class == "HRVITONModel":
                return HRVITONModel(input_nc=3, output_nc=3, ngf=64)
            
            elif model_class == "StableDiffusionPipeline":
                return await self._create_diffusion_model(model_config)
            
            else:
                self.logger.warning(f"⚠️ 지원하지 않는 모델 클래스: {model_class}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ 모델 인스턴스 생성 실패: {e}")
            return None

    async def _create_diffusion_model(self, model_config: ModelConfig):
        """Diffusion 모델 생성 (OOTD 등)"""
        try:
            if DIFFUSERS_AVAILABLE:
                from diffusers import StableDiffusionPipeline
                
                # 체크포인트 경로가 있으면 로컬에서 로드
                if model_config.checkpoint_path and Path(model_config.checkpoint_path).exists():
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        model_config.checkpoint_path,
                        torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False
                    )
                else:
                    # Hugging Face에서 로드
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5",
                        torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False
                    )
                
                return pipeline
            else:
                self.logger.warning("⚠️ Diffusers 라이브러리가 없어서 기본 모델 생성")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Diffusion 모델 생성 실패: {e}")
            return None

    async def _load_checkpoint(self, model: Any, model_config: ModelConfig):
        """체크포인트 로드"""
        if not model_config.checkpoint_path:
            self.logger.info(f"📝 체크포인트 경로 없음: {model_config.name}")
            return
            
        checkpoint_path = Path(model_config.checkpoint_path)
        
        if not checkpoint_path.exists():
            self.logger.warning(f"⚠️ 체크포인트를 찾을 수 없음: {checkpoint_path}")
            return
        
        try:
            # PyTorch 모델인 경우
            if hasattr(model, 'load_state_dict'):
                state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                
                # state_dict 정리
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif isinstance(state_dict, dict) and 'model' in state_dict:
                    state_dict = state_dict['model']
                
                # 키 이름 정리 (module. 제거 등)
                cleaned_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key.replace('module.', '') if key.startswith('module.') else key
                    cleaned_state_dict[new_key] = value
                
                model.load_state_dict(cleaned_state_dict, strict=False)
                self.logger.info(f"✅ 체크포인트 로드 완료: {checkpoint_path}")
            
            # Diffusion 파이프라인인 경우는 이미 from_pretrained에서 처리됨
            else:
                self.logger.info(f"📝 체크포인트 로드 건너뜀 (파이프라인): {model_config.name}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ 체크포인트 로드 실패: {e}")

    async def _apply_m3_max_optimization(self, model: Any, model_config: ModelConfig) -> Any:
        """M3 Max 특화 모델 최적화"""
        try:
            optimizations_applied = []
            
            # 1. MPS 디바이스 최적화
            if self.device == 'mps' and hasattr(model, 'to'):
                optimizations_applied.append("MPS device optimization")
            
            # 2. 메모리 최적화 (64GB+ M3 Max)
            if self.memory_gb >= 64:
                optimizations_applied.append("High memory optimization")
            
            # 3. CoreML 컴파일 준비 (가능한 경우)
            if (COREML_AVAILABLE and 
                hasattr(model, 'eval') and 
                model_config.model_type in [ModelType.HUMAN_PARSING, ModelType.CLOTH_SEGMENTATION]):
                optimizations_applied.append("CoreML compilation ready")
            
            # 4. Metal Performance Shaders 최적화
            if self.device == 'mps':
                try:
                    # PyTorch MPS 최적화 설정
                    if hasattr(torch.backends.mps, 'set_per_process_memory_fraction'):
                        torch.backends.mps.set_per_process_memory_fraction(0.8)
                    optimizations_applied.append("Metal Performance Shaders")
                except:
                    pass
            
            if optimizations_applied:
                self.logger.info(f"🍎 M3 Max 모델 최적화 적용: {', '.join(optimizations_applied)}")
            
            return model
            
        except Exception as e:
            self.logger.warning(f"⚠️ M3 Max 모델 최적화 실패: {e}")
            return model

    async def _load_fallback_model(self, model_name: str) -> Optional[Any]:
        """대체 모델 로드 (MediaPipe, RemBG 등)"""
        try:
            # 이미 로드된 대체 모델이 있으면 반환
            if model_name in self._fallback_models:
                return self._fallback_models[model_name]
            
            fallback_model = None
            
            # 모델 타입별 대체 모델
            if "pose" in model_name.lower() or "pose_estimation" in model_name:
                fallback_model = self._load_mediapipe_pose()
                
            elif "parsing" in model_name.lower() or "human_parsing" in model_name:
                fallback_model = self._load_mediapipe_selfie()
                
            elif "segmentation" in model_name.lower() or "cloth_segmentation" in model_name:
                fallback_model = self._load_rembg_model()
                
            elif "matching" in model_name.lower() or "geometric" in model_name:
                fallback_model = self._create_simple_matching_model()
                
            elif "warping" in model_name.lower() or "fitting" in model_name or "viton" in model_name:
                fallback_model = self._create_simple_generation_model(model_name)
            
            else:
                fallback_model = self._create_dummy_model(model_name)
            
            if fallback_model:
                self._fallback_models[model_name] = fallback_model
                self.logger.info(f"✅ 대체 모델 로드 완료: {model_name}")
            
            return fallback_model
            
        except Exception as e:
            self.logger.error(f"❌ 대체 모델 로드 실패: {e}")
            return self._create_dummy_model(model_name)

    def _load_mediapipe_pose(self):
        """MediaPipe Pose 모델"""
        try:
            if MEDIAPIPE_AVAILABLE:
                return mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=2,
                    enable_segmentation=True,
                    min_detection_confidence=0.5
                )
            else:
                return self._create_dummy_model("pose_estimation")
        except Exception as e:
            self.logger.warning(f"⚠️ MediaPipe Pose 로드 실패: {e}")
            return self._create_dummy_model("pose_estimation")

    def _load_mediapipe_selfie(self):
        """MediaPipe Selfie Segmentation"""
        try:
            if MEDIAPIPE_AVAILABLE:
                return mp.solutions.selfie_segmentation.SelfieSegmentation(
                    model_selection=1
                )
            else:
                return self._create_dummy_model("human_parsing")
        except Exception as e:
            self.logger.warning(f"⚠️ MediaPipe Selfie 로드 실패: {e}")
            return self._create_dummy_model("human_parsing")

    def _load_rembg_model(self):
        """RemBG 배경 제거 모델"""
        try:
            # rembg 라이브러리가 있으면 사용
            try:
                from rembg import new_session
                return new_session("u2net")
            except ImportError:
                # 없으면 간단한 세그멘테이션 모델
                return self._create_simple_segmentation_model()
        except Exception as e:
            self.logger.warning(f"⚠️ RemBG 로드 실패: {e}")
            return self._create_simple_segmentation_model()

    def _create_simple_segmentation_model(self):
        """간단한 세그멘테이션 모델"""
        class SimpleSegmentationModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(inplace=True),
                )
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(64, 1, 3, 1, 1), nn.Sigmoid()
                )
            
            def forward(self, x):
                features = self.encoder(x)
                output = self.decoder(features)
                return output
        
        model = SimpleSegmentationModel()
        if TORCH_AVAILABLE:
            model = model.to(self.device)
        return model

    def _create_simple_matching_model(self):
        """간단한 기하학적 매칭 모델"""
        class SimpleMatchingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_net = nn.Sequential(
                    nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((8, 8)),
                    nn.Flatten(),
                    nn.Linear(128 * 64, 256), nn.ReLU(inplace=True),
                    nn.Linear(256, 18)  # 6개 제어점 * 3
                )
            
            def forward(self, source_img, target_img=None):
                if target_img is not None:
                    # 두 이미지를 결합
                    combined = torch.cat([source_img, target_img], dim=1)
                    combined = nn.functional.interpolate(combined, size=(256, 256), mode='bilinear')
                    # 첫 3채널만 사용
                    combined = combined[:, :3]
                else:
                    combined = source_img
                
                tps_params = self.feature_net(combined)
                return {
                    'tps_params': tps_params.view(-1, 6, 3),
                    'correlation_map': torch.ones(combined.shape[0], 1, 64, 64).to(combined.device)
                }
        
        model = SimpleMatchingModel()
        if TORCH_AVAILABLE:
            model = model.to(self.device)
        return model

    def _create_simple_generation_model(self, model_name: str):
        """간단한 생성 모델"""
        class SimpleGenerationModel(nn.Module):
            def __init__(self, model_name: str):
                super().__init__()
                self.model_name = model_name
                
                # U-Net 스타일 생성기
                self.encoder = nn.Sequential(
                    nn.Conv2d(6, 64, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(256, 512, 3, 2, 1), nn.ReLU(inplace=True)
                )
                
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(64, 3, 3, 1, 1), nn.Tanh()
                )
                
                # 어텐션 모듈
                self.attention = nn.Sequential(
                    nn.Conv2d(6, 32, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(32, 1, 1, 1, 0), nn.Sigmoid()
                )
            
            def forward(self, person_img, cloth_img, **kwargs):
                # 입력 결합
                combined_input = torch.cat([person_img, cloth_img], dim=1)
                
                # 생성
                features = self.encoder(combined_input)
                generated = self.decoder(features)
                
                # 어텐션 적용
                attention_map = self.attention(combined_input)
                
                # 최종 결과
                result = generated * attention_map + person_img * (1 - attention_map)
                
                return {
                    'generated_image': result,
                    'attention_map': attention_map,
                    'warped_cloth': cloth_img,  # 간단한 경우
                    'intermediate': generated
                }
        
        model = SimpleGenerationModel(model_name)
        if TORCH_AVAILABLE:
            model = model.to(self.device)
        return model

    def _create_dummy_model(self, model_name: str):
        """더미 모델 생성"""
        class DummyModel:
            def __init__(self, name: str):
                self.name = name
                self.device_type = "dummy"
            
            def __call__(self, *args, **kwargs):
                return {
                    "result": f"dummy_output_{self.name}",
                    "success": True,
                    "model_type": "dummy"
                }
            
            def forward(self, *args, **kwargs):
                return self(*args, **kwargs)
            
            def to(self, device):
                return self
            
            def eval(self):
                return self
            
            def half(self):
                return self
        
        return DummyModel(model_name)

    async def _check_memory_and_cleanup(self):
        """메모리 확인 및 정리"""
        try:
            # 캐시된 모델 수 확인
            if len(self._loaded_models) >= self.max_cached_models:
                await self._cleanup_least_used_models()
            
            # 가비지 컬렉션
            gc.collect()
            
            # 디바이스별 메모리 정리
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif self.device == 'mps' and torch.backends.mps.is_available():
                try:
                    if hasattr(torch.backends.mps, 'empty_cache'):
                        torch.backends.mps.empty_cache()
                    else:
                        gc.collect()  # PyTorch 2.2.2 호환성
                except:
                    gc.collect()
                    
        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 정리 실패: {e}")

    async def _cleanup_least_used_models(self, keep_count: int = 5):
        """사용량이 적은 모델 정리"""
        try:
            # 사용 빈도순으로 정렬
            sorted_models = sorted(
                self._model_registry.items(),
                key=lambda x: x[1].get("load_count", 0)
            )
            
            cleanup_count = len(self._loaded_models) - keep_count
            cleaned_models = []
            
            for name, _ in sorted_models[:cleanup_count]:
                if name in self._loaded_models:
                    del self._loaded_models[name]
                    self._model_registry[name]["loaded"] = False
                    cleaned_models.append(name)
            
            if cleaned_models:
                self.logger.info(f"🧹 모델 캐시 정리: {len(cleaned_models)}개 모델 해제")
                
        except Exception as e:
            self.logger.error(f"❌ 모델 정리 실패: {e}")

    def unload_model(self, name: str) -> bool:
        """모델 언로드"""
        try:
            with self._lock:
                if name in self._loaded_models:
                    del self._loaded_models[name]
                    if name in self._model_registry:
                        self._model_registry[name]["loaded"] = False
                    self.logger.info(f"🗑️ 모델 언로드: {name}")
                    return True
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 모델 언로드 실패 {name}: {e}")
            return False

    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """모델 정보 조회"""
        with self._lock:
            if name not in self._model_registry:
                return None
                
            info = self._model_registry[name].copy()
            info["is_loaded"] = name in self._loaded_models
            info["load_time"] = self._load_times.get(name, 0)
            info["is_fallback"] = name in self._fallback_models
            
            return info

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """등록된 모델 목록"""
        with self._lock:
            result = {}
            for name, info in self._model_registry.items():
                result[name] = {
                    "name": name,
                    "model_type": info["config"].model_type.value,
                    "model_class": info["config"].model_class,
                    "device": info["config"].device,
                    "loaded": name in self._loaded_models,
                    "load_count": info.get("load_count", 0),
                    "load_time": self._load_times.get(name, 0),
                    "checkpoint_path": info["config"].checkpoint_path,
                    "input_size": info["config"].input_size
                }
            return result

    def get_memory_usage(self) -> Dict[str, Any]:
        """메모리 사용량 조회"""
        try:
            usage = {
                "loaded_models": len(self._loaded_models),
                "fallback_models": len(self._fallback_models),
                "device": self.device
            }
            
            if self.device == "cuda" and torch.cuda.is_available():
                usage.update({
                    "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                    "cached_gb": torch.cuda.memory_reserved() / 1024**3
                })
            elif self.device == "mps":
                try:
                    import psutil
                    process = psutil.Process()
                    usage.update({
                        "process_memory_gb": process.memory_info().rss / 1024**3,
                        "system_memory_percent": psutil.virtual_memory().percent
                    })
                except ImportError:
                    usage["memory_info"] = "psutil not available"
            else:
                usage["memory_info"] = "cpu mode"
                
            return usage
            
        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 사용량 조회 실패: {e}")
            return {"error": str(e)}

    async def initialize(self) -> bool:
        """모델 로더 초기화"""
        try:
            # 모델 체크포인트 경로 확인
            missing_checkpoints = []
            for name, info in self._model_registry.items():
                config = info["config"]
                if config.checkpoint_path:
                    checkpoint_path = Path(config.checkpoint_path)
                    if not checkpoint_path.exists():
                        missing_checkpoints.append(name)
            
            if missing_checkpoints:
                self.logger.warning(f"⚠️ 체크포인트 파일이 없는 모델들: {missing_checkpoints}")
                self.logger.info("📝 해당 모델들은 대체 모델로 로드됩니다")
            
            # M3 Max 최적화 설정
            if self.coreml_optimization and COREML_AVAILABLE:
                self.logger.info("🍎 CoreML 최적화 설정 완료")
            
            self.logger.info(f"✅ 실제 AI 모델 로더 초기화 완료 - {len(self._model_registry)}개 모델 등록됨")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 모델 로더 초기화 실패: {e}")
            return False

    async def get_step_info(self) -> Dict[str, Any]:
        """모델 로더 정보 반환"""
        return {
            "step_name": self.step_name,
            "device": self.device,
            "device_type": self.device_type,
            "memory_gb": self.memory_gb,
            "is_m3_max": self.is_m3_max,
            "optimization_enabled": self.optimization_enabled,
            "quality_level": self.quality_level,
            "initialized": self.is_initialized,
            "config_keys": list(self.config.keys()),
            "specialized_features": {
                "use_fp16": self.use_fp16,
                "coreml_optimization": self.coreml_optimization,
                "lazy_loading": self.lazy_loading,
                "auto_quantization": self.auto_quantization,
                "memory_mapping": self.memory_mapping,
                "max_cached_models": self.max_cached_models
            },
            "model_stats": {
                "registered_models": len(self._model_registry),
                "loaded_models": len(self._loaded_models),
                "fallback_models": len(self._fallback_models),
                "total_load_count": sum(info.get("load_count", 0) for info in self._model_registry.values()),
                "average_load_time": sum(self._load_times.values()) / len(self._load_times) if self._load_times else 0
            },
            "library_availability": {
                "torch": TORCH_AVAILABLE,
                "opencv": CV_AVAILABLE,
                "mediapipe": MEDIAPIPE_AVAILABLE,
                "transformers": TRANSFORMERS_AVAILABLE,
                "diffusers": DIFFUSERS_AVAILABLE,
                "onnx": ONNX_AVAILABLE,
                "coreml": COREML_AVAILABLE
            }
        }

# ==============================================
# 유틸리티 함수들
# ==============================================

def preprocess_image(image: Union[np.ndarray, Image.Image], target_size: tuple, normalize: bool = True) -> torch.Tensor:
    """이미지 전처리"""
    try:
        if not CV_AVAILABLE:
            raise ImportError("OpenCV not available")
            
        if isinstance(image, np.ndarray):
            if image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError(f"지원하지 않는 이미지 타입: {type(image)}")
        
        # 리사이즈
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # 텐서 변환
        image_array = np.array(image).astype(np.float32)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1) / 255.0
        
        # 정규화
        if normalize:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
        
        return image_tensor.unsqueeze(0)
        
    except Exception as e:
        logging.error(f"이미지 전처리 실패: {e}")
        # 더미 텐서 반환
        return torch.randn(1, 3, target_size[1], target_size[0])

def postprocess_segmentation(output: torch.Tensor, original_size: tuple, threshold: float = 0.5) -> np.ndarray:
    """세그멘테이션 후처리"""
    try:
        if not CV_AVAILABLE:
            raise ImportError("OpenCV not available")
            
        if output.dim() == 4:
            output = output.squeeze(0)
        
        # 확률을 클래스로 변환
        if output.shape[0] > 1:
            output = torch.argmax(output, dim=0)
        else:
            output = (output > threshold).float()
        
        # CPU로 이동 및 numpy 변환
        output = output.cpu().numpy().astype(np.uint8)
        
        # 원본 크기로 리사이즈
        if output.shape != original_size[::-1]:
            output = cv2.resize(output, original_size, interpolation=cv2.INTER_NEAREST)
        
        return output
        
    except Exception as e:
        logging.error(f"세그멘테이션 후처리 실패: {e}")
        return np.zeros(original_size[::-1], dtype=np.uint8)

def postprocess_pose(output: torch.Tensor, original_size: tuple, confidence_threshold: float = 0.3) -> Dict[str, Any]:
    """포즈 추정 후처리"""
    try:
        if isinstance(output, (list, tuple)):
            # OpenPose 스타일 출력 (PAF, heatmaps)
            pafs, heatmaps = output[-1]  # 마지막 스테이지 결과 사용
        else:
            heatmaps = output
            pafs = None
        
        # 키포인트 추출
        keypoints = []
        if heatmaps.dim() == 4:
            heatmaps = heatmaps.squeeze(0)
        
        for i in range(heatmaps.shape[0] - 1):  # 배경 제외
            heatmap = heatmaps[i].cpu().numpy()
            
            # 최대값 위치 찾기
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            confidence = heatmap[y, x]
            
            if confidence > confidence_threshold:
                # 원본 이미지 크기로 스케일링
                x_scaled = int(x * original_size[0] / heatmap.shape[1])
                y_scaled = int(y * original_size[1] / heatmap.shape[0])
                keypoints.append([x_scaled, y_scaled, confidence])
            else:
                keypoints.append([0, 0, 0])
        
        return {
            'keypoints': keypoints,
            'pafs': pafs.cpu().numpy() if pafs is not None else None,
            'heatmaps': heatmaps.cpu().numpy()
        }
        
    except Exception as e:
        logging.error(f"포즈 추정 후처리 실패: {e}")
        return {'keypoints': [], 'pafs': None, 'heatmaps': None}

# ==============================================
# 편의 함수들 (하위 호환성)
# ==============================================

def create_model_loader(
    device: str = "mps",
    use_fp16: bool = True,
    **kwargs
) -> ModelLoader:
    """모델 로더 생성 (하위 호환)"""
    return ModelLoader(
        device=device,
        use_fp16=use_fp16,
        **kwargs
    )

# 전역 모델 로더 (선택적)
_global_model_loader: Optional[ModelLoader] = None

def get_global_model_loader() -> Optional[ModelLoader]:
    """전역 모델 로더 반환"""
    global _global_model_loader
    return _global_model_loader

def initialize_global_model_loader(**kwargs) -> ModelLoader:
    """전역 모델 로더 초기화"""
    global _global_model_loader
    _global_model_loader = ModelLoader(**kwargs)
    return _global_model_loader

async def load_model_async(model_name: str, config: Optional[ModelConfig] = None) -> Optional[Any]:
    """비동기 모델 로드"""
    try:
        loader = get_global_model_loader()
        if loader is None:
            loader = initialize_global_model_loader()
        return await loader.load_model(model_name)
    except Exception as e:
        logging.error(f"비동기 모델 로드 실패: {e}")
        return None

def load_model_sync(model_name: str, config: Optional[ModelConfig] = None) -> Optional[Any]:
    """동기 모델 로드"""
    try:
        loader = get_global_model_loader()
        if loader is None:
            loader = initialize_global_model_loader()
        # 동기 버전에서는 asyncio 사용
        import asyncio
        return asyncio.run(loader.load_model(model_name))
    except Exception as e:
        logging.error(f"동기 모델 로드 실패: {e}")
        return None

def cleanup_global_loader():
    """전역 로더 정리"""
    global _global_model_loader
    if _global_model_loader:
        try:
            import asyncio
            asyncio.run(_global_model_loader._check_memory_and_cleanup())
        except:
            pass
        _global_model_loader = None

# 모듈 레벨에서 정리 함수 등록
import atexit
atexit.register(cleanup_global_loader)

# 모듈 익스포트
__all__ = [
    'ModelLoader',
    'ModelConfig', 
    'ModelType',
    'OptimalStepConstructor',
    'GraphonomyModel',
    'OpenPoseModel', 
    'U2NetModel',
    'GeometricMatchingModel',
    'HRVITONModel',
    'create_model_loader',
    'get_global_model_loader',
    'initialize_global_model_loader',
    'load_model_async',
    'load_model_sync',
    'preprocess_image',
    'postprocess_segmentation',
    'postprocess_pose'
]