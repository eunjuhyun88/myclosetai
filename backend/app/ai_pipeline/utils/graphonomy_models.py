"""
🔥 Graphonomy 모델 정의
모든 Graphonomy 관련 모델들을 체계적으로 관리
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling 모듈"""
    
    def __init__(self, in_channels=2048, out_channels=256, atrous_rates=[6, 12, 18]):
        super().__init__()
        
        # 1x1 convolution
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Atrous convolutions
        self.atrous_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for rate in atrous_rates
        ])
        
        # Global average pooling
        self.conv_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Output convolution
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    
    def forward(self, x):
        # 1x1 convolution
        conv1x1 = self.conv1x1(x)
        
        # Atrous convolutions
        atrous_convs = [conv(x) for conv in self.atrous_convs]
        
        # Global average pooling
        global_feat = self.conv_global(x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate all features
        concat_feat = torch.cat([conv1x1] + atrous_convs + [global_feat], dim=1)
        
        # Output convolution
        output = self.conv_out(concat_feat)
        
        return output


class SelfAttentionBlock(nn.Module):
    """Self-Attention 블록"""
    
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        channel_weights = self.channel_attention(x)
        x = x * channel_weights
        
        # Spatial attention
        spatial_weights = self.spatial_attention(x)
        x = x * spatial_weights
        
        return x


class ResNetBottleneck(nn.Module):
    """ResNet Bottleneck 블록"""
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet101Backbone(nn.Module):
    """ResNet-101 백본"""
    
    def __init__(self):
        super().__init__()
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(ResNetBottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(ResNetBottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(ResNetBottleneck, 256, 23, stride=2)
        self.layer4 = self._make_layer(ResNetBottleneck, 512, 3, stride=2)
        
        self._init_weights()
    
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        
        return {
            'layer1': layer1,
            'layer2': layer2,
            'layer3': layer3,
            'layer4': layer4
        }


class ProgressiveParsingModule(nn.Module):
    """Progressive Parsing 모듈"""
    
    def __init__(self, num_classes=20, num_stages=3, hidden_dim=256):
        super().__init__()
        self.num_stages = num_stages
        self.hidden_dim = hidden_dim
        
        # Stage별 특성 추출기
        self.stage_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_classes + (hidden_dim if i > 0 else 0), hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ) for i in range(num_stages)
        ])
        
        # Stage별 attention 모듈
        self.stage_attention = nn.ModuleList([
            SelfAttentionBlock(hidden_dim) for _ in range(num_stages)
        ])
        
        # Stage별 예측기
        self.stage_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
                nn.BatchNorm2d(hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 2, num_classes, 1)
            ) for _ in range(num_stages)
        ])
        
        # Stage별 confidence 예측기
        self.confidence_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 1),
                nn.Sigmoid()
            ) for _ in range(num_stages)
        ])
        
        # Cross-stage fusion
        self.cross_stage_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * num_stages, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Final refinement
        self.final_refiner = nn.Sequential(
            nn.Conv2d(hidden_dim + num_classes, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, 1)
        )
    
    def forward(self, initial_parsing, base_features):
        stage_results = []
        stage_features = []
        current_input = initial_parsing
        
        for i in range(self.num_stages):
            # Feature extraction
            if i == 0:
                feat_input = current_input
            else:
                feat_input = torch.cat([current_input, stage_features[-1]], dim=1)
            
            stage_feat = self.stage_extractors[i](feat_input)
            
            # Apply attention
            attended_feat = self.stage_attention[i](stage_feat)
            stage_features.append(attended_feat)
            
            # Prediction
            parsing_pred = self.stage_predictors[i](attended_feat)
            confidence = self.confidence_predictors[i](attended_feat)
            
            # Progressive refinement with residual connection
            if i == 0:
                refined_parsing = parsing_pred
            else:
                # Weighted combination with previous stage
                weight = confidence
                refined_parsing = (1 - weight) * current_input + weight * parsing_pred
            
            stage_results.append({
                'parsing': refined_parsing,
                'confidence': confidence,
                'features': attended_feat,
                'stage': i
            })
            
            current_input = refined_parsing
        
        # Cross-stage feature fusion
        if len(stage_features) > 1:
            fused_features = self.cross_stage_fusion(torch.cat(stage_features, dim=1))
            
            # Final refinement
            final_input = torch.cat([current_input, fused_features], dim=1)
            final_refinement = self.final_refiner(final_input)
            
            # Add refined result as final stage
            stage_results.append({
                'parsing': current_input + final_refinement * 0.2,
                'confidence': torch.ones_like(confidence) * 0.9,
                'features': fused_features,
                'stage': 'final'
            })
        
        return stage_results


class SelfCorrectionModule(nn.Module):
    """Self-Correction 모듈"""
    
    def __init__(self, num_classes=20, hidden_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Error detection network
        self.error_detector = nn.Sequential(
            nn.Conv2d(num_classes + hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Correction network
        self.correction_net = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, num_classes, 1)
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention for correction
        self.spatial_attention = SelfAttentionBlock(hidden_dim)
    
    def forward(self, initial_parsing, features):
        # Concatenate parsing and features
        combined_input = torch.cat([initial_parsing, features], dim=1)
        
        # Error detection
        error_features = self.error_detector(combined_input)
        
        # Apply spatial attention
        attended_features = self.spatial_attention(error_features)
        
        # Generate correction
        correction = self.correction_net(attended_features)
        
        # Estimate confidence
        confidence = self.confidence_estimator(attended_features)
        
        # Apply correction with confidence weighting
        corrected_parsing = initial_parsing + correction * confidence
        
        return corrected_parsing, {
            'correction': correction,
            'confidence': confidence,
            'spatial_confidence': confidence
        }


class IterativeRefinementModule(nn.Module):
    """Iterative Refinement 모듈"""
    
    def __init__(self, num_classes=20, hidden_dim=256, max_iterations=3):
        super().__init__()
        self.num_classes = num_classes
        self.max_iterations = max_iterations
        
        # Refinement network
        self.refinement_net = nn.Sequential(
            nn.Conv2d(num_classes, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, 1)
        )
        
        # Convergence checker
        self.convergence_checker = nn.Sequential(
            nn.Conv2d(num_classes, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, initial_parsing):
        refinement_results = []
        current_parsing = initial_parsing
        
        for i in range(self.max_iterations):
            # Generate refinement
            refinement = self.refinement_net(current_parsing)
            
            # Apply refinement
            refined_parsing = current_parsing + refinement * 0.1
            
            # Check convergence
            convergence_score = self.convergence_checker(refined_parsing)
            
            refinement_results.append({
                'parsing': refined_parsing,
                'refinement': refinement,
                'convergence': convergence_score,
                'iteration': i
            })
            
            # Update current parsing
            current_parsing = refined_parsing
            
            # Early stopping if converged
            if convergence_score > 0.95:
                break
        
        return refinement_results


class AdvancedGraphonomyResNetASPP(nn.Module):
    """고급 Graphonomy ResNet-101 + ASPP + Self-Attention + Progressive Parsing 완전 구현"""
    
    def __init__(self, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        
        # ResNet-101 백본
        self.backbone = ResNet101Backbone()
        
        # ASPP 모듈 (2048 -> 256)
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        
        # Self-Attention 모듈
        self.self_attention = SelfAttentionBlock(in_channels=256)
        
        # Progressive Parsing 모듈
        self.progressive_parsing = ProgressiveParsingModule(
            num_classes=num_classes, 
            num_stages=3,
            hidden_dim=256
        )
        
        # Self-Correction 모듈
        self.self_correction = SelfCorrectionModule(
            num_classes=num_classes,
            hidden_dim=256
        )
        
        # Iterative Refinement 모듈
        self.iterative_refine = IterativeRefinementModule(
            num_classes=num_classes,
            hidden_dim=256,
            max_iterations=3
        )
        
        # 기본 분류기
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        # Edge detection branch
        self.edge_classifier = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """고급 순전파 (모든 알고리즘 완전 적용)"""
        input_size = x.shape[2:]
        
        # 1. Backbone features (ResNet-101)
        backbone_features = self.backbone(x)
        high_level_features = backbone_features['layer4']  # 2048 channels
        
        # 2. ASPP (Multi-scale context aggregation)
        aspp_features = self.aspp(high_level_features)  # 256 channels
        
        # 3. Self-Attention (Spatial attention mechanism)
        attended_features = self.self_attention(aspp_features)
        
        # 4. 기본 분류 (초기 파싱)
        initial_parsing = self.classifier(attended_features)
        
        # 5. Progressive Parsing (3단계 정제)
        progressive_results = self.progressive_parsing(initial_parsing, attended_features)
        final_progressive = progressive_results[-1]['parsing']
        
        # 6. Self-Correction Learning (SCHP 알고리즘)
        corrected_parsing, correction_info = self.self_correction(
            final_progressive, attended_features
        )
        
        # 7. Iterative Refinement (수렴 기반 정제)
        refinement_results = self.iterative_refine(corrected_parsing)
        final_refined = refinement_results[-1]['parsing']
        
        # 8. Edge detection
        edge_output = self.edge_classifier(attended_features)
        
        # 9. 입력 크기로 업샘플링
        final_parsing = F.interpolate(
            final_refined, size=input_size,
            mode='bilinear', align_corners=False
        )
        
        edge_output_resized = F.interpolate(
            edge_output, size=input_size,
            mode='bilinear', align_corners=False
        )
        
        return {
            'parsing': final_parsing,
            'edge': edge_output_resized,
            'progressive_results': progressive_results,
            'correction_info': correction_info,
            'refinement_results': refinement_results,
            'intermediate_features': {
                'backbone': backbone_features,
                'aspp': aspp_features,
                'attention': attended_features
            }
        } 