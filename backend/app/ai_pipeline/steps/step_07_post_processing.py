"""
7단계: post_processing
"""

import asyncio
import torch

class postprocessingStep:
    """post_processing 단계"""
    
    def __init__(self, config, device, model_loader=None):
        self.config = config
        self.device = device
        self.model_loader = model_loader
    
    async def process(self, input_data) -> torch.Tensor:
        """post_processing 처리"""
        await asyncio.sleep(0.3)
        
        if isinstance(input_data, dict):
            # 복합 입력의 경우 첫 번째 텐서 사용
            first_key = list(input_data.keys())[0]
            sample_tensor = input_data[first_key]
            if hasattr(sample_tensor, 'shape'):
                return torch.randn_like(sample_tensor).to(self.device)
            else:
                return torch.randn(1, 3, 512, 512).to(self.device)
        else:
            return torch.randn_like(input_data).to(self.device)
    
    async def warmup(self, dummy_input):
        await self.process(dummy_input)
