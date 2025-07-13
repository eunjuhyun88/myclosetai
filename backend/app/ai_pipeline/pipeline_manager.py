"""
8단계 AI 파이프라인 매니저 - 실제 모델 연결
MyCloset AI의 핵심 엔진
"""
import os
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import cv2
from concurrent.futures import ThreadPoolExecutor
import gc

# 8단계 파이프라인 스텝들 임포트
from .steps.step_01_human_parsing import HumanParsingStep
from .steps.step_02_pose_estimation import PoseEstimationStep
from .steps.step_03_cloth_segmentation import ClothSegmentationStep
from .steps.step_04_geometric_matching import GeometricMatchingStep
from .steps.step_05_cloth_warping import ClothWarpingStep
from .steps.step_06_virtual_fitting import VirtualFittingStep
from .steps.step_07_post_processing import PostProcessingStep
from .steps.step_08_quality_assessment import QualityAssessmentStep

# 유틸리티 임포트
from .utils.model_loader import ModelLoader
from .utils.memory_manager import MemoryManager
from .utils.data_converter import DataConverter

from ..core.gpu_config import GPUConfig
from ..core.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)

class PipelineManager:
    """8단계 AI 파이프라인 통합 매니저"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Args:
            config: 파이프라인 설정
        """
        self.config = config or PipelineConfig()
        self.gpu_config = GPUConfig()
        
        # 디바이스 설정
        self.device = self.gpu_config.get_optimal_device()
        logger.info(f"🔧 파이프라인 디바이스: {self.device}")
        
        # 컴포넌트 초기화
        self.model_loader = ModelLoader(device=self.device)
        self.memory_manager = MemoryManager(device=self.device)
        self.data_converter = DataConverter()
        
        # 8단계 파이프라인 스텝들
        self.steps = {}
        self.is_initialized = False
        
        # 처리 통계
        self.stats = {
            "total_processed": 0,
            "successful_processes": 0,
            "average_time_per_step": {},
            "memory_usage": [],
            "error_counts": {}
        }
        
        # 스레드 풀
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        logger.info("🤖 PipelineManager 초기화 완료")
    
    async def initialize(self) -> bool:
        """파이프라인 초기화 - 모든 모델 로드"""
        try:
            logger.info("🚀 8단계 AI 파이프라인 초기화 시작...")
            
            # GPU 메모리 최적화
            await self.memory_manager.optimize_memory()
            
            # 1단계: 인체 파싱 (Graphonomy)
            logger.info("1️⃣ 인체 파싱 모델 로드 중...")
            self.steps[1] = HumanParsingStep(
                model_loader=self.model_loader,
                device=self.device,
                config=self.config.step_configs.get('human_parsing', {})
            )
            await self.steps[1].initialize()
            
            # 2단계: 포즈 추정 (OpenPose/MediaPipe)
            logger.info("2️⃣ 포즈 추정 모델 로드 중...")
            self.steps[2] = PoseEstimationStep(
                model_loader=self.model_loader,
                device=self.device,
                config=self.config.step_configs.get('pose_estimation', {})
            )
            await self.steps[2].initialize()
            
            # 3단계: 의류 세그멘테이션 (U²-Net)
            logger.info("3️⃣ 의류 세그멘테이션 모델 로드 중...")
            self.steps[3] = ClothSegmentationStep(
                model_loader=self.model_loader,
                device=self.device,
                config=self.config.step_configs.get('cloth_segmentation', {})
            )
            await self.steps[3].initialize()
            
            # 4단계: 기하학적 매칭 (TPS)
            logger.info("4️⃣ 기하학적 매칭 모델 로드 중...")
            self.steps[4] = GeometricMatchingStep(
                model_loader=self.model_loader,
                device=self.device,
                config=self.config.step_configs.get('geometric_matching', {})
            )
            await self.steps[4].initialize()
            
            # 5단계: 옷 워핑
            logger.info("5️⃣ 옷 워핑 모델 로드 중...")
            self.steps[5] = ClothWarpingStep(
                model_loader=self.model_loader,
                device=self.device,
                config=self.config.step_configs.get('cloth_warping', {})
            )
            await self.steps[5].initialize()
            
            # 6단계: 가상 피팅 (HR-VITON/ACGPN)
            logger.info("6️⃣ 가상 피팅 모델 로드 중...")
            self.steps[6] = VirtualFittingStep(
                model_loader=self.model_loader,
                device=self.device,
                config=self.config.step_configs.get('virtual_fitting', {})
            )
            await self.steps[6].initialize()
            
            # 7단계: 후처리 (Super Resolution)
            logger.info("7️⃣ 후처리 모델 로드 중...")
            self.steps[7] = PostProcessingStep(
                model_loader=self.model_loader,
                device=self.device,
                config=self.config.step_configs.get('post_processing', {})
            )
            await self.steps[7].initialize()
            
            # 8단계: 품질 평가
            logger.info("8️⃣ 품질 평가 모델 로드 중...")
            self.steps[8] = QualityAssessmentStep(
                model_loader=self.model_loader,
                device=self.device,
                config=self.config.step_configs.get('quality_assessment', {})
            )
            await self.steps[8].initialize()
            
            self.is_initialized = True
            logger.info("✅ 8단계 AI 파이프라인 초기화 완료!")
            
            # 메모리 상태 로그
            await self.memory_manager.log_memory_status()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 파이프라인 초기화 실패: {e}")
            self.is_initialized = False
            return False
    
    async def process_virtual_tryon(
        self,
        person_image: Image.Image,
        clothing_image: Image.Image,
        height: float = 170.0,
        weight: float = 65.0,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        8단계 가상 피팅 파이프라인 실행
        
        Args:
            person_image: 사용자 이미지
            clothing_image: 의류 이미지  
            height: 키 (cm)
            weight: 몸무게 (kg)
            progress_callback: 진행 상태 콜백 함수
            
        Returns:
            처리 결과
        """
        if not self.is_initialized:
            raise RuntimeError("파이프라인이 초기화되지 않았습니다.")
        
        start_time = time.time()
        process_id = f"process_{int(time.time() * 1000)}"
        
        result = {
            "success": False,
            "process_id": process_id,
            "fitted_image": None,
            "processing_time": 0.0,
            "confidence": 0.0,
            "fit_score": 0.0,
            "quality_score": 0.0,
            "measurements": {},
            "recommendations": [],
            "pipeline_stages": {},
            "debug_info": {}
        }
        
        try:
            logger.info(f"🎯 가상 피팅 시작 - Process ID: {process_id}")
            
            # 입력 이미지 전처리
            person_tensor = self.data_converter.pil_to_tensor(person_image, self.device)
            clothing_tensor = self.data_converter.pil_to_tensor(clothing_image, self.device)
            
            # 8단계 파이프라인 실행
            step_results = {}
            
            # 1단계: 인체 파싱 (20개 부위 분할)
            if progress_callback:
                await progress_callback(1, "인체 파싱 실행 중...", 0)
            
            step_1_result = await self._execute_step(
                1, "인체 파싱", 
                self.steps[1].process,
                person_tensor,
                progress_callback
            )
            step_results[1] = step_1_result
            result["pipeline_stages"]["step_1_human_parsing"] = step_1_result
            
            # 2단계: 포즈 추정 (18개 키포인트)
            if progress_callback:
                await progress_callback(2, "포즈 추정 실행 중...", 12.5)
            
            step_2_result = await self._execute_step(
                2, "포즈 추정",
                self.steps[2].process,
                person_tensor, height, weight,
                progress_callback
            )
            step_results[2] = step_2_result
            result["pipeline_stages"]["step_2_pose_estimation"] = step_2_result
            
            # 3단계: 의류 세그멘테이션
            if progress_callback:
                await progress_callback(3, "의류 세그멘테이션 실행 중...", 25)
            
            step_3_result = await self._execute_step(
                3, "의류 세그멘테이션",
                self.steps[3].process,
                clothing_tensor,
                progress_callback
            )
            step_results[3] = step_3_result
            result["pipeline_stages"]["step_3_cloth_segmentation"] = step_3_result
            
            # 4단계: 기하학적 매칭
            if progress_callback:
                await progress_callback(4, "기하학적 매칭 실행 중...", 37.5)
            
            step_4_result = await self._execute_step(
                4, "기하학적 매칭",
                self.steps[4].process,
                person_tensor, clothing_tensor, 
                step_1_result, step_2_result,
                progress_callback
            )
            step_results[4] = step_4_result
            result["pipeline_stages"]["step_4_geometric_matching"] = step_4_result
            
            # 5단계: 옷 워핑
            if progress_callback:
                await progress_callback(5, "옷 워핑 실행 중...", 50)
            
            step_5_result = await self._execute_step(
                5, "옷 워핑",
                self.steps[5].process,
                clothing_tensor, step_4_result, height, weight,
                progress_callback
            )
            step_results[5] = step_5_result
            result["pipeline_stages"]["step_5_cloth_warping"] = step_5_result
            
            # 6단계: 가상 피팅 생성
            if progress_callback:
                await progress_callback(6, "가상 피팅 생성 중...", 62.5)
            
            step_6_result = await self._execute_step(
                6, "가상 피팅 생성",
                self.steps[6].process,
                person_tensor, step_5_result,
                step_1_result, step_2_result,
                progress_callback
            )
            step_results[6] = step_6_result
            result["pipeline_stages"]["step_6_virtual_fitting"] = step_6_result
            
            # 7단계: 후처리
            if progress_callback:
                await progress_callback(7, "후처리 실행 중...", 75)
            
            step_7_result = await self._execute_step(
                7, "후처리",
                self.steps[7].process,
                step_6_result["fitted_image"],
                progress_callback
            )
            step_results[7] = step_7_result
            result["pipeline_stages"]["step_7_post_processing"] = step_7_result
            
            # 8단계: 품질 평가
            if progress_callback:
                await progress_callback(8, "품질 평가 실행 중...", 87.5)
            
            step_8_result = await self._execute_step(
                8, "품질 평가",
                self.steps[8].process,
                step_7_result["enhanced_image"],
                person_tensor, clothing_tensor,
                progress_callback
            )
            step_results[8] = step_8_result
            result["pipeline_stages"]["step_8_quality_assessment"] = step_8_result
            
            # 최종 결과 설정
            final_image = step_7_result["enhanced_image"]
            result.update({
                "success": True,
                "fitted_image": self.data_converter.tensor_to_base64(final_image),
                "processing_time": time.time() - start_time,
                "confidence": step_6_result.get("confidence", 0.85),
                "fit_score": step_8_result.get("fit_score", 0.80),
                "quality_score": step_8_result.get("quality_score", 0.82),
                "measurements": step_2_result.get("measurements", {}),
                "recommendations": step_8_result.get("recommendations", [])
            })
            
            if progress_callback:
                await progress_callback(8, "완료!", 100)
            
            # 통계 업데이트
            self._update_stats(result["processing_time"], True, step_results)
            
            logger.info(f"✅ 가상 피팅 완료 - Process ID: {process_id}, 시간: {result['processing_time']:.2f}초")
            
        except Exception as e:
            result.update({
                "processing_time": time.time() - start_time,
                "error": str(e),
                "error_step": getattr(e, 'step', 'unknown')
            })
            self._update_stats(result["processing_time"], False, {})
            logger.error(f"❌ 가상 피팅 실패 - Process ID: {process_id}: {e}")
            
        finally:
            # 메모리 정리
            await self.memory_manager.cleanup_step()
        
        return result
    
    async def _execute_step(
        self, 
        step_num: int, 
        step_name: str, 
        step_func, 
        *args, 
        progress_callback=None
    ) -> Dict[str, Any]:
        """개별 스텝 실행"""
        step_start = time.time()
        
        try:
            logger.info(f"{step_num}️⃣ {step_name} 시작...")
            
            # 비동기 실행
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor, step_func, *args
            )
            
            processing_time = time.time() - step_start
            result["processing_time"] = processing_time
            result["step_number"] = step_num
            result["step_name"] = step_name
            
            # 통계 업데이트
            if step_name not in self.stats["average_time_per_step"]:
                self.stats["average_time_per_step"][step_name] = []
            self.stats["average_time_per_step"][step_name].append(processing_time)
            
            logger.info(f"✅ {step_name} 완료 - {processing_time:.2f}초")
            return result
            
        except Exception as e:
            processing_time = time.time() - step_start
            logger.error(f"❌ {step_name} 실패: {e}")
            
            # 에러 통계 업데이트
            if step_name not in self.stats["error_counts"]:
                self.stats["error_counts"][step_name] = 0
            self.stats["error_counts"][step_name] += 1
            
            # 에러를 step 정보와 함께 re-raise
            error = Exception(f"{step_name} 실패: {str(e)}")
            error.step = step_num
            raise error
    
    def _update_stats(
        self, 
        processing_time: float, 
        success: bool, 
        step_results: Dict[int, Any]
    ):
        """처리 통계 업데이트"""
        self.stats["total_processed"] += 1
        
        if success:
            self.stats["successful_processes"] += 1
        
        # 메모리 사용량 기록
        if self.device == "cuda":
            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
        elif self.device == "mps":
            memory_used = torch.mps.current_allocated_memory() / 1024**3  # GB
        else:
            import psutil
            memory_used = psutil.virtual_memory().used / 1024**3  # GB
        
        self.stats["memory_usage"].append(memory_used)
        
        # 메모리 사용량 로그 제한
        if len(self.stats["memory_usage"]) > 100:
            self.stats["memory_usage"] = self.stats["memory_usage"][-50:]
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """파이프라인 상태 조회"""
        return {
            "initialized": self.is_initialized,
            "device": self.device,
            "steps_loaded": len(self.steps),
            "total_steps": 8,
            "memory_status": await self.memory_manager.get_memory_status(),
            "stats": {
                "total_processed": self.stats["total_processed"],
                "success_rate": (
                    self.stats["successful_processes"] / self.stats["total_processed"] 
                    if self.stats["total_processed"] > 0 else 0
                ),
                "average_step_times": {
                    step: sum(times) / len(times) 
                    for step, times in self.stats["average_time_per_step"].items()
                },
                "error_counts": self.stats["error_counts"],
                "current_memory_gb": (
                    self.stats["memory_usage"][-1] 
                    if self.stats["memory_usage"] else 0
                )
            }
        }
    
    async def warmup(self) -> bool:
        """파이프라인 웜업 - 더미 데이터로 테스트"""
        try:
            logger.info("🔥 파이프라인 웜업 시작...")
            
            # 더미 이미지 생성 (512x512)
            dummy_person = Image.fromarray(
                np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            )
            dummy_clothing = Image.fromarray(
                np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            )
            
            # 웜업 실행
            result = await self.process_virtual_tryon(
                dummy_person, dummy_clothing, 
                height=170, weight=65
            )
            
            if result["success"]:
                logger.info("✅ 파이프라인 웜업 완료")
                return True
            else:
                logger.warning("⚠️ 파이프라인 웜업 부분 실패")
                return False
                
        except Exception as e:
            logger.error(f"❌ 파이프라인 웜업 실패: {e}")
            return False
    
    async def cleanup(self):
        """리소스 정리"""
        logger.info("🧹 파이프라인 매니저 정리 중...")
        
        # 각 스텝 정리
        for step_num, step in self.steps.items():
            try:
                if hasattr(step, 'cleanup'):
                    await step.cleanup()
            except Exception as e:
                logger.warning(f"스텝 {step_num} 정리 중 오류: {e}")
        
        # 메모리 매니저 정리
        await self.memory_manager.cleanup()
        
        # 스레드 풀 종료
        self.executor.shutdown(wait=True)
        
        # 스텝 딕셔너리 정리
        self.steps.clear()
        self.is_initialized = False
        
        logger.info("✅ 파이프라인 매니저 정리 완료")

# 전역 파이프라인 매니저 인스턴스
_pipeline_manager: Optional[PipelineManager] = None

def get_pipeline_manager() -> PipelineManager:
    """파이프라인 매니저 싱글톤 인스턴스 반환"""
    global _pipeline_manager
    if _pipeline_manager is None:
        _pipeline_manager = PipelineManager()
    return _pipeline_manager