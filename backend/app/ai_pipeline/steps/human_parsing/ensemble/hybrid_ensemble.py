"""
🔥 Hybrid Ensemble 모듈
======================

하이브리드 앙상블 시스템 구현

Author: MyCloset AI Team
Date: 2025-08-07
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List


class HybridEnsembleModule(nn.Module):
    """하이브리드 앙상블 모듈 (다중 모델 결합)"""
    
    def __init__(self, num_classes=20, num_models=3, hidden_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.num_models = num_models
        self.hidden_dim = hidden_dim
        
        # MPS 디바이스 감지 및 타입 설정
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.dtype = torch.float32  # MPS에서는 float32 사용
        
        # 앙상블 가중치 학습 네트워크 (MPS 타입 일관성)
        self.weight_learner = nn.Sequential(
            nn.Conv2d(num_classes * num_models, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, num_models, 1),
            nn.Softmax(dim=1)
        ).to(device=self.device, dtype=self.dtype)
        
        # Confidence 기반 가중치 조정 (MPS 타입 일관성)
        self.confidence_adapter = nn.Sequential(
            nn.Conv2d(num_models, hidden_dim // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, num_models, 1),
            nn.Sigmoid()
        ).to(device=self.device, dtype=self.dtype)
        
        # 공간적 일관성 검사 (MPS 타입 일관성)
        self.spatial_consistency = nn.Sequential(
            nn.Conv2d(num_classes * num_models, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, num_models, 1),
            nn.Sigmoid()
        ).to(device=self.device, dtype=self.dtype)
        
        # 최종 융합 네트워크 (MPS 타입 일관성)
        self.final_fusion = nn.Sequential(
            nn.Conv2d(num_classes * num_models, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, num_classes, 1)
        ).to(device=self.device, dtype=self.dtype)
        
        # 불확실성 정량화 (MPS 타입 일관성)
        self.uncertainty_estimator = nn.Sequential(
            nn.Conv2d(num_classes * num_models, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid()
        ).to(device=self.device, dtype=self.dtype)
    
    def forward(self, model_outputs, confidences):
        """
        하이브리드 앙상블 순전파 (MPS 타입 일관성 유지)
        
        Args:
            model_outputs: 모델 출력 리스트 (각각 B, num_classes, H, W)
            confidences: 신뢰도 맵 리스트 (각각 B, 1, H, W)
        
        Returns:
            ensemble_result: 앙상블 결과
        """
        # MPS 타입 일관성 유지
        unified_outputs = []
        for output in model_outputs:
            if hasattr(output, 'to'):
                output = output.to(device=self.device, dtype=self.dtype)
            unified_outputs.append(output)
        
        unified_confidences = []
        for conf in confidences:
            if hasattr(conf, 'to'):
                conf = conf.to(device=self.device, dtype=self.dtype)
            unified_confidences.append(conf)
        
        batch_size, _, height, width = unified_outputs[0].shape
        
        # 1. 모든 모델 출력을 채널 차원으로 결합
        concatenated_outputs = torch.cat(unified_outputs, dim=1)
        
        # 2. 학습된 가중치 계산
        learned_weights = self.weight_learner(concatenated_outputs)
        
        # 3. 신뢰도 기반 가중치 조정
        confidence_tensor = torch.cat(unified_confidences, dim=1)
        confidence_adjusted_weights = self.confidence_adapter(confidence_tensor)
        
        # 4. 공간적 일관성 검사
        spatial_weights = self.spatial_consistency(concatenated_outputs)
        
        # 5. 최종 가중치 계산 (세 가지 가중치의 조합)
        final_weights = (learned_weights + confidence_adjusted_weights + spatial_weights) / 3.0
        
        # 6. 가중 평균 앙상블
        weighted_outputs = []
        for i, output in enumerate(model_outputs):
            weight = final_weights[:, i:i+1, :, :]
            weighted_outputs.append(output * weight)
        
        ensemble_output = sum(weighted_outputs)
        
        # 7. 최종 융합 네트워크 적용
        final_ensemble = self.final_fusion(concatenated_outputs)
        
        # 8. 불확실성 정량화
        uncertainty = self.uncertainty_estimator(concatenated_outputs)
        
        # 9. 결과 조합 (가중 평균 + 융합 네트워크)
        final_output = ensemble_output + final_ensemble
        
        return {
            'ensemble_output': final_output,
            'weighted_ensemble': ensemble_output,
            'fused_ensemble': final_ensemble,
            'ensemble_weights': final_weights,
            'uncertainty': uncertainty,
            'model_confidences': confidences,
            'spatial_consistency': spatial_weights
        }
