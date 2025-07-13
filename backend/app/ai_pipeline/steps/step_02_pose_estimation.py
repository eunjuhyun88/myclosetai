"""
2단계: 포즈 추정 (Pose Estimation)
18개 키포인트 검출
"""

import asyncio
import torch
import numpy as np

class PoseEstimationStep:
    """포즈 추정 단계"""
    
    def __init__(self, config, device, model_loader):
        self.config = config
        self.device = device
        self.model_loader = model_loader
    
    async def process(self, person_tensor: torch.Tensor) -> torch.Tensor:
        """포즈 추정 처리"""
        await asyncio.sleep(0.4)
        
        # 더미 키포인트 생성 (18개 키포인트)
        batch_size = person_tensor.shape[0]
        dummy_keypoints = torch.randn(batch_size, 18, 3)  # [x, y, confidence]
        
        return dummy_keypoints.to(self.device)
    
    async def warmup(self, dummy_input: torch.Tensor):
        await self.process(dummy_input)
