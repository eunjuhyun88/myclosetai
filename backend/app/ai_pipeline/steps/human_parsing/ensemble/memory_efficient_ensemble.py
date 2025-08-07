"""
🔥 Memory Efficient Ensemble System
==================================

메모리 효율적 앙상블 시스템

Author: MyCloset AI Team
Date: 2025-08-07
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional


class MemoryEfficientEnsembleSystem(nn.Module):
    """메모리 효율적 앙상블 시스템"""
    
    def __init__(self, num_classes=20, ensemble_models=None, hidden_dim=None, config=None):
        super().__init__()
        self.num_classes = num_classes
        self.ensemble_models = ensemble_models or []
        self.hidden_dim = hidden_dim or 256
        self.config = config
        
        # MPS 디바이스 감지 및 타입 설정
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.dtype = torch.float32  # MPS에서는 float32 사용
        
        # 메모리 효율적 앙상블 네트워크 (MPS 타입 일관성)
        self.ensemble_net = nn.Sequential(
            nn.Conv2d(num_classes * len(self.ensemble_models) if self.ensemble_models else num_classes, 
                     self.hidden_dim, 3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim // 2, num_classes, 1)
        ).to(device=self.device, dtype=self.dtype)
        
        # 가중치 학습 네트워크 (MPS 타입 일관성)
        self.weight_learner = nn.Sequential(
            nn.Conv2d(num_classes * len(self.ensemble_models) if self.ensemble_models else num_classes, 
                     self.hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim // 2, len(self.ensemble_models) if self.ensemble_models else 1, 1),
            nn.Softmax(dim=1)
        ).to(device=self.device, dtype=self.dtype)
        
        # 불확실성 정량화 (MPS 타입 일관성)
        self.uncertainty_estimator = nn.Sequential(
            nn.Conv2d(num_classes * len(self.ensemble_models) if self.ensemble_models else num_classes, 
                     self.hidden_dim // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim // 4, 1, 1),
            nn.Sigmoid()
        ).to(device=self.device, dtype=self.dtype)
        
        # 메모리 최적화 설정
        self.memory_optimization = {
            'gradient_checkpointing': True,
            'mixed_precision': True,
            'tensor_cores': True,
            'memory_efficient_attention': True
        }
        
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
    
    def _standardize_tensor_sizes(self, tensors, target_size=None):
        """텐서 크기 표준화 (MPS 타입 일관성 유지)"""
        if not tensors:
            return tensors
        
        if target_size is None:
            # 첫 번째 텐서의 크기를 기준으로 설정
            target_size = tensors[0].shape[2:]
        
        standardized_tensors = []
        for tensor in tensors:
            # MPS 타입 일관성 확인
            if hasattr(tensor, 'device') and str(tensor.device).startswith('mps'):
                # MPS 디바이스의 경우 float32로 통일
                if tensor.dtype != torch.float32:
                    tensor = tensor.to(torch.float32)
            
            if tensor.shape[2:] != target_size:
                # 메모리 효율적인 리사이징
                tensor = F.interpolate(
                    tensor, size=target_size, 
                    mode='bilinear', align_corners=False
                )
            standardized_tensors.append(tensor)
        
        return standardized_tensors
    
    def _standardize_channels(self, tensor, target_channels=20):
        """채널 수 표준화 (MPS 타입 일관성 유지)"""
        current_channels = tensor.shape[1]
        
        # MPS 타입 일관성 확인
        if hasattr(tensor, 'device') and str(tensor.device).startswith('mps'):
            # MPS 디바이스의 경우 float32로 통일
            if tensor.dtype != torch.float32:
                tensor = tensor.to(torch.float32)
            target_dtype = torch.float32
        else:
            target_dtype = tensor.dtype
        
        if current_channels == target_channels:
            return tensor
        elif current_channels > target_channels:
            # 채널 수 줄이기
            return tensor[:, :target_channels, :, :]
        else:
            # 채널 수 늘리기 (패딩)
            padding = torch.zeros(
                tensor.shape[0], target_channels - current_channels,
                tensor.shape[2], tensor.shape[3],
                device=tensor.device, dtype=target_dtype  # 명시적으로 타입 지정
            )
            return torch.cat([tensor, padding], dim=1)
    
    def forward(self, model_outputs, model_confidences=None):
        """
        메모리 효율적 앙상블 순전파
        
        Args:
            model_outputs: 모델 출력 리스트
            model_confidences: 모델 신뢰도 리스트 (선택사항)
        
        Returns:
            ensemble_result: 앙상블 결과
        """
        if not model_outputs:
            return None
        
        # 0. MPS 타입 일관성 유지 (강화된 버전)
        # 모든 텐서를 동일한 디바이스와 타입으로 통일
        unified_outputs = []
        for output in model_outputs:
            if hasattr(output, 'to'):
                # 모든 텐서를 MPS 디바이스의 float32로 통일
                output = output.to(device=self.device, dtype=self.dtype)
            unified_outputs.append(output)
        
        # 1. 텐서 크기 표준화
        standardized_outputs = self._standardize_tensor_sizes(unified_outputs)
        
        # 2. 채널 수 표준화
        standardized_outputs = [self._standardize_channels(output, self.num_classes) 
                              for output in standardized_outputs]
        
        # 3. 모든 출력을 채널 차원으로 결합
        concatenated_outputs = torch.cat(standardized_outputs, dim=1)
        
        # 4. 메모리 효율적 앙상블 적용
        if self.memory_optimization['gradient_checkpointing']:
            # 그래디언트 체크포인팅으로 메모리 절약
            ensemble_output = torch.utils.checkpoint.checkpoint(
                self.ensemble_net, concatenated_outputs
            )
        else:
            ensemble_output = self.ensemble_net(concatenated_outputs)
        
        # 5. 가중치 학습
        learned_weights = self.weight_learner(concatenated_outputs)
        
        # 6. 불확실성 정량화
        uncertainty = self.uncertainty_estimator(concatenated_outputs)
        
        # 7. 가중 평균 앙상블 (메모리 효율적)
        weighted_sum = torch.zeros_like(standardized_outputs[0])
        for i, output in enumerate(standardized_outputs):
            weight = learned_weights[:, i:i+1, :, :]
            weighted_sum += output * weight
        
        # 8. 최종 결과 조합
        final_output = ensemble_output + weighted_sum
        
        return {
            'ensemble_output': final_output,
            'weighted_ensemble': weighted_sum,
            'learned_ensemble': ensemble_output,
            'ensemble_weights': learned_weights,
            'uncertainty': uncertainty,
            'num_models': len(standardized_outputs),
            'memory_optimization': self.memory_optimization
        }
