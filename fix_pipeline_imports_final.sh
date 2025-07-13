#!/bin/bash

echo "🔨 누락된 파이프라인 클래스들 생성 중..."

cd backend

# 1. 먼저 필요한 유틸리티 클래스들 생성
echo "📦 기본 유틸리티 클래스 생성 중..."

# 간단한 메모리 매니저
cat > app/ai_pipeline/utils/memory_manager.py << 'EOF'
"""메모리 관리 유틸리티"""
import psutil
import torch
import logging

logger = logging.getLogger(__name__)

class GPUMemoryManager:
    def __init__(self, device="mps", memory_limit_gb=16.0):
        self.device = device
        self.memory_limit_gb = memory_limit_gb
    
    def clear_cache(self):
        """메모리 정리"""
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def check_memory_usage(self):
        """메모리 사용량 확인"""
        memory = psutil.virtual_memory()
        used_gb = memory.used / (1024**3)
        if used_gb > self.memory_limit_gb * 0.9:
            logger.warning(f"메모리 사용량 높음: {used_gb:.1f}GB")
            self.clear_cache()
EOF

# 모델 로더
cat > app/ai_pipeline/utils/model_loader.py << 'EOF'
"""모델 로딩 유틸리티"""
import torch
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, device="mps", use_fp16=True):
        self.device = torch.device(device)
        self.use_fp16 = use_fp16
        self.loaded_models = {}
    
    def load_model(self, model_name, model_path=None):
        """더미 모델 로드"""
        logger.info(f"더미 모델 로드: {model_name}")
        
        class DummyModel:
            def __init__(self, name):
                self.name = name
            
            def __call__(self, *args, **kwargs):
                return {"result": f"dummy_{self.name}", "success": True}
        
        model = DummyModel(model_name)
        self.loaded_models[model_name] = model
        return model
EOF

# 데이터 변환기
cat > app/ai_pipeline/utils/data_converter.py << 'EOF'
"""데이터 변환 유틸리티"""
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class DataConverter:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
    
    def image_to_tensor(self, image, size=512):
        """PIL 이미지를 텐서로 변환"""
        if isinstance(image, Image.Image):
            image = image.resize((size, size))
            tensor = self.transform(image)
            return tensor.unsqueeze(0)  # 배치 차원 추가
        return image
    
    def tensor_to_numpy(self, tensor):
        """텐서를 numpy 배열로 변환"""
        if torch.is_tensor(tensor):
            # 배치 차원 제거하고 (C, H, W) -> (H, W, C)로 변환
            tensor = tensor.squeeze(0) if tensor.dim() == 4 else tensor
            if tensor.dim() == 3:
                tensor = tensor.permute(1, 2, 0)
            
            # [0, 1] 범위를 [0, 255]로 변환
            array = tensor.cpu().numpy()
            if array.max() <= 1.0:
                array = (array * 255).astype(np.uint8)
            return array
        return tensor
EOF

# 2. 각 단계별 더미 클래스들 생성
echo "🔧 8단계 더미 클래스들 생성 중..."

for i in {1..8}; do
    step_names=(
        "human_parsing"
        "pose_estimation" 
        "cloth_segmentation"
        "geometric_matching"
        "cloth_warping"
        "virtual_fitting"
        "post_processing"
        "quality_assessment"
    )
    
    step_name=${step_names[$((i-1))]}
    file_name="step_0${i}_${step_name}.py"
    
    cat > "app/ai_pipeline/steps/$file_name" << EOF
"""Step $i: ${step_name^} 단계"""

import asyncio
import torch
import numpy as np
from typing import Any, Dict

class ${step_name^}Step:
    def __init__(self, config=None, device="mps", model_loader=None):
        self.config = config
        self.device = device
        self.model_loader = model_loader
        self.name = "${step_name}"
    
    async def process(self, input_data: Any) -> Dict[str, Any]:
        """${step_name} 처리 (더미)"""
        # 처리 시뮬레이션
        await asyncio.sleep(0.5)
        
        result = {
            "step": "${step_name}",
            "success": True,
            "data": f"processed_${step_name}",
            "confidence": 0.85 + (hash("${step_name}") % 100) / 1000.0
        }
        
        # 특별한 반환값들
        if "${step_name}" == "human_parsing":
            result["body_measurements"] = {
                "chest": 88.0, "waist": 70.0, "hip": 92.0, "bmi": 22.5
            }
        elif "${step_name}" == "cloth_segmentation":
            result["cloth_type"] = "상의"
            result["cloth_confidence"] = 0.9
        elif "${step_name}" == "quality_assessment":
            result = {
                "overall_score": 0.88,
                "fit_coverage": 0.85,
                "color_preservation": 0.92,
                "fit_overall": 0.87,
                "ssim": 0.89,
                "lpips": 0.85
            }
        
        return result
    
    async def warmup(self, dummy_input):
        """워밍업"""
        await asyncio.sleep(0.1)
    
    def cleanup(self):
        """정리"""
        pass
EOF

done

# 3. 메인 pipeline_manager.py 교체 
echo "🔄 pipeline_manager.py 재생성 중..."

cat > app/ai_pipeline/pipeline_manager.py << 'EOF'
"""
MyCloset AI 가상 피팅 파이프라인 메인 클래스
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass

# 유틸리티 import
from .utils.memory_manager import GPUMemoryManager
from .utils.model_loader import ModelLoader
from .utils.data_converter import DataConverter

# 각 단계 import
from .steps.step_01_human_parsing import HumanParsingStep
from .steps.step_02_pose_estimation import PoseEstimationStep  
from .steps.step_03_cloth_segmentation import ClothSegmentationStep
from .steps.step_04_geometric_matching import GeometricMatchingStep
from .steps.step_05_cloth_warping import ClothWarpingStep
from .steps.step_06_virtual_fitting import VirtualFittingStep
from .steps.step_07_post_processing import PostProcessingStep
from .steps.step_08_quality_assessment import QualityAssessmentStep

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    device: str = "mps"
    batch_size: int = 1
    image_size: int = 512
    use_fp16: bool = True
    enable_caching: bool = True
    parallel_steps: bool = True
    memory_limit_gb: float = 16.0
    quality_threshold: float = 0.8
    quality_mode: str = "balanced"

@dataclass 
class PipelineResult:
    """파이프라인 실행 결과"""
    success: bool
    fitted_image: Optional[np.ndarray] = None
    processing_time: float = 0.0
    step_times: Dict[str, float] = None
    quality_scores: Dict[str, float] = None
    intermediate_results: Dict[str, Any] = None
    memory_usage: Dict[str, float] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.step_times is None:
            self.step_times = {}
        if self.quality_scores is None:
            self.quality_scores = {}
        if self.intermediate_results is None:
            self.intermediate_results = {}
        if self.memory_usage is None:
            self.memory_usage = {}

class VirtualTryOnPipeline:
    """메인 가상 피팅 파이프라인 클래스"""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.logger = logging.getLogger(__name__)
        
        # 디바이스 설정
        self.device = self._setup_device()
        
        # 유틸리티 초기화
        self.memory_manager = GPUMemoryManager(
            device=str(self.device), 
            memory_limit_gb=self.config.memory_limit_gb
        )
        self.model_loader = ModelLoader(device=str(self.device), use_fp16=self.config.use_fp16)
        self.data_converter = DataConverter()
        
        # 각 단계 초기화
        self._initialize_steps()
        
        # 성능 추적
        self.step_times = {}
        self.progress_callback: Optional[Callable] = None
        
        self.logger.info(f"✅ VirtualTryOnPipeline 초기화 완료 (device: {self.device})")

    def _setup_device(self):
        """디바이스 설정"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _initialize_steps(self):
        """8단계 초기화"""
        try:
            self.steps = {
                "human_parsing": HumanParsingStep(self.config, str(self.device), self.model_loader),
                "pose_estimation": PoseEstimationStep(self.config, str(self.device), self.model_loader),
                "cloth_segmentation": ClothSegmentationStep(self.config, str(self.device), self.model_loader),
                "geometric_matching": GeometricMatchingStep(self.config, str(self.device)),
                "cloth_warping": ClothWarpingStep(self.config, str(self.device)),
                "virtual_fitting": VirtualFittingStep(self.config, str(self.device), self.model_loader),
                "post_processing": PostProcessingStep(self.config, str(self.device)),
                "quality_assessment": QualityAssessmentStep(self.config, str(self.device))
            }
            self.logger.info("✅ 8단계 파이프라인 초기화 완료")
        except Exception as e:
            self.logger.error(f"❌ 단계 초기화 실패: {e}")
            self.steps = {}

    async def process(
        self, 
        person_image: Image.Image,
        cloth_image: Image.Image, 
        user_measurements: Dict[str, float] = None
    ) -> PipelineResult:
        """메인 파이프라인 실행"""
        start_time = time.time()
        
        try:
            self.logger.info("🚀 8단계 가상 피팅 파이프라인 시작")
            
            # 메모리 정리
            self.memory_manager.clear_cache()
            
            # 이미지 전처리
            person_tensor = self.data_converter.image_to_tensor(person_image, self.config.image_size)
            cloth_tensor = self.data_converter.image_to_tensor(cloth_image, self.config.image_size)
            
            # 각 단계 실행
            intermediate_results = {}
            step_names = list(self.steps.keys())
            
            current_data = {
                "person_image": person_tensor,
                "cloth_image": cloth_tensor,
                "user_measurements": user_measurements or {}
            }
            
            for i, (step_name, step) in enumerate(self.steps.items()):
                step_start = time.time()
                
                # 진행률 콜백
                if self.progress_callback:
                    await self.progress_callback(i, 0.0)
                
                self.logger.info(f"📋 단계 {i+1}/8: {step_name} 처리 중...")
                
                # 단계 실행
                step_result = await step.process(current_data)
                intermediate_results[step_name] = step_result
                
                # 다음 단계를 위해 데이터 업데이트
                if isinstance(step_result, dict):
                    current_data.update(step_result)
                
                # 시간 기록
                step_time = time.time() - step_start
                self.step_times[step_name] = step_time
                
                # 진행률 완료
                if self.progress_callback:
                    await self.progress_callback(i, 1.0)
                
                self.logger.info(f"✅ {step_name} 완료 ({step_time:.2f}s)")
            
            # 최종 결과 이미지 생성 (더미)
            final_image = self._generate_dummy_result_image()
            
            # 품질 점수 (더미)
            quality_scores = intermediate_results.get('quality_assessment', {
                "overall_score": 0.88,
                "fit_coverage": 0.85, 
                "color_preservation": 0.92,
                "fit_overall": 0.87
            })
            
            processing_time = time.time() - start_time
            
            result = PipelineResult(
                success=True,
                fitted_image=final_image,
                processing_time=processing_time,
                step_times=self.step_times.copy(),
                quality_scores=quality_scores,
                intermediate_results=intermediate_results,
                memory_usage=self.memory_manager.get_usage() if hasattr(self.memory_manager, 'get_usage') else {}
            )
            
            self.logger.info(f"🎉 파이프라인 완료! 총 시간: {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 파이프라인 오류: {e}")
            return PipelineResult(
                success=False,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _generate_dummy_result_image(self) -> np.ndarray:
        """더미 결과 이미지 생성"""
        # 512x512 컬러풀한 더미 이미지
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # 그라데이션 패턴 생성
        for i in range(512):
            for j in range(512):
                image[i, j] = [
                    int(128 + 127 * np.sin(i * 0.02)),
                    int(128 + 127 * np.cos(j * 0.02)), 
                    int(128 + 127 * np.sin((i + j) * 0.01))
                ]
        
        return image

    def get_pipeline_status(self) -> Dict[str, Any]:
        """파이프라인 상태 반환"""
        return {
            "device": str(self.device),
            "config": self.config.__dict__ if hasattr(self.config, '__dict__') else {},
            "steps_loaded": len(self.steps),
            "step_names": list(self.steps.keys()),
            "memory_usage": getattr(self.memory_manager, 'get_usage', lambda: {})()
        }
    
    async def warmup(self):
        """파이프라인 워밍업"""
        self.logger.info("🔥 파이프라인 워밍업 시작...")
        
        # 더미 이미지로 워밍업
        dummy_person = Image.new('RGB', (512, 512), color=(255, 0, 0))
        dummy_cloth = Image.new('RGB', (512, 512), color=(0, 0, 255))
        
        try:
            result = await self.process(dummy_person, dummy_cloth)
            if result.success:
                self.logger.info("✅ 워밍업 완료")
            else:
                self.logger.warning(f"⚠️ 워밍업 중 오류: {result.error_message}")
        except Exception as e:
            self.logger.warning(f"⚠️ 워밍업 실패: {e}")
    
    def cleanup(self):
        """리소스 정리"""
        self.logger.info("🧹 파이프라인 리소스 정리...")
        
        for step in self.steps.values():
            if hasattr(step, 'cleanup'):
                step.cleanup()
        
        self.memory_manager.clear_cache()
        self.steps.clear()
        
        self.logger.info("✅ 리소스 정리 완료")

class PipelineFactory:
    """파이프라인 팩토리 클래스"""
    
    @staticmethod
    def create_optimized_pipeline(
        memory_gb: float = 16.0,
        quality_mode: str = "balanced"
    ) -> VirtualTryOnPipeline:
        """최적화된 파이프라인 생성"""
        
        # 품질 모드별 설정
        if quality_mode == "fast":
            config = PipelineConfig(
                device="mps",
                image_size=256,
                use_fp16=True,
                parallel_steps=True,
                memory_limit_gb=memory_gb,
                quality_threshold=0.7,
                quality_mode="fast"
            )
        elif quality_mode == "quality":
            config = PipelineConfig(
                device="mps", 
                image_size=1024,
                use_fp16=False,
                parallel_steps=False,
                memory_limit_gb=memory_gb,
                quality_threshold=0.9,
                quality_mode="quality"
            )
        else:  # balanced
            config = PipelineConfig(
                device="mps",
                image_size=512,
                use_fp16=True,
                parallel_steps=True,
                memory_limit_gb=memory_gb,
                quality_threshold=0.8,
                quality_mode="balanced"
            )
        
        pipeline = VirtualTryOnPipeline(config)
        logger.info(f"✅ {quality_mode} 모드 파이프라인 생성 완료")
        
        return pipeline

# 하위 호환성을 위한 별칭
VirtualFittingPipeline = VirtualTryOnPipeline
EOF

echo "✅ 모든 파이프라인 클래스 생성 완료!"

# 4. Python import 테스트
echo "🧪 import 테스트 중..."

python -c "
import sys
sys.path.insert(0, '.')

try:
    from app.ai_pipeline.pipeline_manager import VirtualTryOnPipeline, PipelineFactory
    from app.core.pipeline_config import PipelineConfig
    print('✅ 모든 클래스 import 성공!')
    
    # 간단한 기능 테스트
    pipeline = PipelineFactory.create_optimized_pipeline()
    print(f'✅ PipelineFactory 테스트 성공: {type(pipeline).__name__}')
    
    status = pipeline.get_pipeline_status()
    print(f'✅ Pipeline 상태 조회 성공: {len(status)} 항목')
    
except Exception as e:
    print(f'❌ 테스트 실패: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 파이프라인 클래스 생성 및 테스트 완료!"
    echo "================================================"
    echo "✅ VirtualTryOnPipeline: 8단계 AI 파이프라인 실행"
    echo "✅ PipelineFactory: 품질 모드별 파이프라인 생성"
    echo "✅ 8개 처리 단계: 더미 구현으로 테스트 가능"
    echo "✅ 메모리 관리: M3 Max 최적화"
    echo ""
    echo "🚀 이제 서버를 재시작하세요:"
    echo "   python run_server.py"
else
    echo "❌ 테스트 실패. 수동 확인이 필요합니다."
fi