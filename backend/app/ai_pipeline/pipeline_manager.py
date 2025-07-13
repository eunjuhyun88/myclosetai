"""
MyCloset AI 8단계 가상 피팅 파이프라인 매니저
M3 Max 최적화된 고성능 파이프라인 오케스트레이션
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import psutil

from .utils.memory_manager import GPUMemoryManager
from .utils.model_loader import ModelLoader
from .utils.data_converter import DataConverter

# 각 단계 모듈 import
from .steps.step_01_human_parsing import HumanParsingStep
from .steps.step_02_pose_estimation import PoseEstimationStep
from .steps.step_03_cloth_segmentation import ClothSegmentationStep
from .steps.step_04_geometric_matching import GeometricMatchingStep
from .steps.step_05_cloth_warping import ClothWarpingStep
from .steps.step_06_virtual_fitting import VirtualFittingStep
from .steps.step_07_post_processing import PostProcessingStep
from .steps.step_08_quality_assessment import QualityAssessmentStep

@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    device: str = "mps"  # M3 Max MPS 백엔드
    batch_size: int = 1
    image_size: int = 512
    use_fp16: bool = True
    enable_caching: bool = True
    parallel_steps: bool = True
    memory_limit_gb: float = 16.0  # M3 Max 통합 메모리 한계
    quality_threshold: float = 0.8

@dataclass
class PipelineResult:
    """파이프라인 실행 결과"""
    success: bool
    fitted_image: Optional[np.ndarray]
    processing_time: float
    step_times: Dict[str, float]
    quality_scores: Dict[str, float]
    intermediate_results: Dict[str, Any]
    memory_usage: Dict[str, float]
    error_message: Optional[str] = None

class VirtualTryOnPipeline:
    """메인 가상 피팅 파이프라인"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 디바이스 설정 (M3 Max 최적화)
        self.device = self._setup_device()
        
        # 메모리 매니저 초기화
        self.memory_manager = GPUMemoryManager(
            device=self.device,
            memory_limit_gb=config.memory_limit_gb
        )
        
        # 모델 로더 초기화
        self.model_loader = ModelLoader(
            device=self.device,
            use_fp16=config.use_fp16
        )
        
        # 데이터 변환기 초기화
        self.data_converter = DataConverter()
        
        # 각 단계 초기화
        self._initialize_steps()
        
        # 성능 모니터링
        self.step_times = {}
        self.memory_usage = {}
        
        self.logger.info(f"파이프라인 초기화 완료 - Device: {self.device}")

    def _setup_device(self) -> torch.device:
        """M3 Max MPS 디바이스 설정"""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            self.logger.info("M3 Max MPS 백엔드 사용")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            self.logger.info("CUDA 백엔드 사용")
        else:
            device = torch.device("cpu")
            self.logger.warning("CPU 백엔드 사용 - 성능이 제한될 수 있습니다")
        return device

    def _initialize_steps(self):
        """각 파이프라인 단계 초기화"""
        self.steps = {
            "human_parsing": HumanParsingStep(
                config=self.config,
                device=self.device,
                model_loader=self.model_loader
            ),
            "pose_estimation": PoseEstimationStep(
                config=self.config,
                device=self.device,
                model_loader=self.model_loader
            ),
            "cloth_segmentation": ClothSegmentationStep(
                config=self.config,
                device=self.device,
                model_loader=self.model_loader
            ),
            "geometric_matching": GeometricMatchingStep(
                config=self.config,
                device=self.device
            ),
            "cloth_warping": ClothWarpingStep(
                config=self.config,
                device=self.device
            ),
            "virtual_fitting": VirtualFittingStep(
                config=self.config,
                device=self.device,
                model_loader=self.model_loader
            ),
            "post_processing": PostProcessingStep(
                config=self.config
            ),
            "quality_assessment": QualityAssessmentStep(
                config=self.config,
                device=self.device
            )
        }

    async def process(
        self,
        person_image: Image.Image,
        cloth_image: Image.Image,
        user_measurements: Optional[Dict[str, float]] = None
    ) -> PipelineResult:
        """메인 파이프라인 실행"""
        start_time = time.time()
        intermediate_results = {}
        step_times = {}
        
        try:
            self.logger.info("파이프라인 실행 시작")
            
            # 메모리 초기화
            self.memory_manager.clear_cache()
            
            # 입력 이미지 전처리
            person_tensor = self.data_converter.image_to_tensor(
                person_image, self.config.image_size
            ).to(self.device)
            cloth_tensor = self.data_converter.image_to_tensor(
                cloth_image, self.config.image_size
            ).to(self.device)
            
            # 단계 1-2: 병렬 실행 (인체 파싱 + 포즈 추정)
            if self.config.parallel_steps:
                step1_result, step2_result = await self._execute_parallel_steps_1_2(
                    person_tensor
                )
            else:
                step1_result = await self._execute_step("human_parsing", person_tensor)
                step2_result = await self._execute_step("pose_estimation", person_tensor)
            
            intermediate_results["human_parsing"] = step1_result
            intermediate_results["pose_estimation"] = step2_result
            
            # 단계 3: 의류 세그멘테이션
            step3_result = await self._execute_step("cloth_segmentation", cloth_tensor)
            intermediate_results["cloth_segmentation"] = step3_result
            
            # 단계 4: 기하학적 매칭
            step4_result = await self._execute_step(
                "geometric_matching",
                {
                    "human_parsing": step1_result,
                    "pose_keypoints": step2_result,
                    "cloth_mask": step3_result,
                    "person_image": person_tensor,
                    "cloth_image": cloth_tensor
                }
            )
            intermediate_results["geometric_matching"] = step4_result
            
            # 단계 5: 옷 워핑
            step5_result = await self._execute_step(
                "cloth_warping",
                {
                    "cloth_image": cloth_tensor,
                    "cloth_mask": step3_result,
                    "tps_transform": step4_result,
                    "target_shape": step1_result
                }
            )
            intermediate_results["cloth_warping"] = step5_result
            
            # 단계 6: 가상 피팅 생성
            step6_result = await self._execute_step(
                "virtual_fitting",
                {
                    "person_image": person_tensor,
                    "human_parsing": step1_result,
                    "pose_keypoints": step2_result,
                    "warped_cloth": step5_result,
                    "cloth_mask": step3_result
                }
            )
            intermediate_results["virtual_fitting"] = step6_result
            
            # 단계 7-8: 병렬 실행 (후처리 + 품질 평가)
            if self.config.parallel_steps:
                final_image, quality_scores = await self._execute_parallel_steps_7_8(
                    step6_result, person_image, cloth_image
                )
            else:
                final_image = await self._execute_step("post_processing", step6_result)
                quality_scores = await self._execute_step(
                    "quality_assessment",
                    {
                        "result_image": final_image,
                        "original_person": person_image,
                        "original_cloth": cloth_image
                    }
                )
            
            intermediate_results["post_processing"] = final_image
            intermediate_results["quality_assessment"] = quality_scores
            
            # 최종 결과 생성
            processing_time = time.time() - start_time
            
            # 메모리 사용량 기록
            memory_usage = self._get_memory_usage()
            
            return PipelineResult(
                success=True,
                fitted_image=self.data_converter.tensor_to_numpy(final_image),
                processing_time=processing_time,
                step_times=self.step_times,
                quality_scores=quality_scores,
                intermediate_results=intermediate_results,
                memory_usage=memory_usage
            )
            
        except Exception as e:
            self.logger.error(f"파이프라인 실행 중 오류 발생: {str(e)}")
            return PipelineResult(
                success=False,
                fitted_image=None,
                processing_time=time.time() - start_time,
                step_times=self.step_times,
                quality_scores={},
                intermediate_results=intermediate_results,
                memory_usage=self._get_memory_usage(),
                error_message=str(e)
            )

    async def _execute_step(self, step_name: str, input_data: Any) -> Any:
        """개별 단계 실행"""
        step_start_time = time.time()
        
        try:
            # 메모리 체크
            self.memory_manager.check_memory_usage()
            
            # 단계 실행
            result = await self.steps[step_name].process(input_data)
            
            # 실행 시간 기록
            execution_time = time.time() - step_start_time
            self.step_times[step_name] = execution_time
            
            self.logger.info(f"{step_name} 완료 - {execution_time:.2f}초")
            return result
            
        except Exception as e:
            self.logger.error(f"{step_name} 실행 중 오류: {str(e)}")
            raise

    async def _execute_parallel_steps_1_2(self, person_tensor: torch.Tensor) -> Tuple[Any, Any]:
        """단계 1-2 병렬 실행 (인체 파싱 + 포즈 추정)"""
        tasks = [
            self._execute_step("human_parsing", person_tensor),
            self._execute_step("pose_estimation", person_tensor)
        ]
        return await asyncio.gather(*tasks)

    async def _execute_parallel_steps_7_8(
        self, 
        fitting_result: torch.Tensor,
        original_person: Image.Image,
        original_cloth: Image.Image
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """단계 7-8 병렬 실행 (후처리 + 품질 평가)"""
        
        # 후처리는 동기적으로 먼저 실행
        final_image = await self._execute_step("post_processing", fitting_result)
        
        # 품질 평가는 비동기로 실행
        quality_task = self._execute_step(
            "quality_assessment",
            {
                "result_image": final_image,
                "original_person": original_person,
                "original_cloth": original_cloth
            }
        )
        
        quality_scores = await quality_task
        return final_image, quality_scores

    def _get_memory_usage(self) -> Dict[str, float]:
        """메모리 사용량 조회"""
        memory_info = {}
        
        # 시스템 메모리
        memory_info["system_memory_gb"] = psutil.virtual_memory().used / (1024**3)
        
        # MPS 메모리 (가능한 경우)
        if self.device.type == "mps":
            try:
                memory_info["mps_memory_gb"] = torch.mps.current_allocated_memory() / (1024**3)
            except:
                memory_info["mps_memory_gb"] = 0.0
        
        return memory_info

    def get_pipeline_status(self) -> Dict[str, Any]:
        """파이프라인 상태 조회"""
        return {
            "device": str(self.device),
            "config": self.config.__dict__,
            "memory_usage": self._get_memory_usage(),
            "step_times": self.step_times,
            "models_loaded": [name for name, step in self.steps.items() if hasattr(step, 'model_loaded') and step.model_loaded]
        }

    async def warmup(self):
        """파이프라인 워밍업 (첫 실행 시 모델 로딩)"""
        self.logger.info("파이프라인 워밍업 시작")
        
        # 더미 데이터로 각 단계 초기화
        dummy_image = torch.randn(1, 3, self.config.image_size, self.config.image_size).to(self.device)
        
        for step_name, step in self.steps.items():
            try:
                if hasattr(step, 'warmup'):
                    await step.warmup(dummy_image)
                self.logger.info(f"{step_name} 워밍업 완료")
            except Exception as e:
                self.logger.warning(f"{step_name} 워밍업 실패: {str(e)}")
        
        self.logger.info("파이프라인 워밍업 완료")

    def cleanup(self):
        """리소스 정리"""
        self.logger.info("파이프라인 리소스 정리")
        
        # 각 단계 정리
        for step in self.steps.values():
            if hasattr(step, 'cleanup'):
                step.cleanup()
        
        # 메모리 정리
        self.memory_manager.clear_cache()
        
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

# 파이프라인 팩토리
class PipelineFactory:
    """파이프라인 생성 팩토리"""
    
    @staticmethod
    def create_optimized_pipeline(
        memory_gb: float = 16.0,
        quality_mode: str = "balanced"  # "fast", "balanced", "quality"
    ) -> VirtualTryOnPipeline:
        """최적화된 파이프라인 생성"""
        
        if quality_mode == "fast":
            config = PipelineConfig(
                device="mps",
                batch_size=1,
                image_size=256,
                use_fp16=True,
                enable_caching=True,
                parallel_steps=True,
                memory_limit_gb=memory_gb,
                quality_threshold=0.7
            )
        elif quality_mode == "quality":
            config = PipelineConfig(
                device="mps",
                batch_size=1,
                image_size=1024,
                use_fp16=False,
                enable_caching=True,
                parallel_steps=False,
                memory_limit_gb=memory_gb,
                quality_threshold=0.9
            )
        else:  # balanced
            config = PipelineConfig(
                device="mps",
                batch_size=1,
                image_size=512,
                use_fp16=True,
                enable_caching=True,
                parallel_steps=True,
                memory_limit_gb=memory_gb,
                quality_threshold=0.8
            )
        
        return VirtualTryOnPipeline(config)

# 사용 예시
async def main():
    """파이프라인 사용 예시"""
    
    # 파이프라인 생성
    pipeline = PipelineFactory.create_optimized_pipeline(
        memory_gb=16.0,
        quality_mode="balanced"
    )
    
    # 워밍업
    await pipeline.warmup()
    
    # 이미지 로드
    person_image = Image.open("person.jpg")
    cloth_image = Image.open("cloth.jpg")
    
    # 파이프라인 실행
    result = await pipeline.process(person_image, cloth_image)
    
    if result.success:
        print(f"처리 완료! 소요시간: {result.processing_time:.2f}초")
        print(f"품질 점수: {result.quality_scores}")
        
        # 결과 이미지 저장
        Image.fromarray(result.fitted_image).save("result.jpg")
    else:
        print(f"처리 실패: {result.error_message}")
    
    # 정리
    pipeline.cleanup()

if __name__ == "__main__":
    asyncio.run(main())