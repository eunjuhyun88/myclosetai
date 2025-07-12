# backend/app/core/m3_optimizer.py
class M3MaxOptimizer:
    def __init__(self):
        self.device = "mps"  # Metal Performance Shaders
        self.max_memory = 20000  # MB (128GB 중 20GB 할당)
        
    def optimize_model_loading(self):
        """모델을 순차적으로 로드하여 메모리 효율성 극대화"""
        torch.mps.set_per_process_memory_fraction(0.8)