"""
app/services/pipeline_service.py - 서비스 레이어 (리팩토링)

✅ RealAIPipelineProcessor를 서비스 레이어로 분리
✅ 비즈니스 로직 중심화
✅ API 레이어와 AI 처리 레이어 분리
✅ 명확한 책임 분리
✅ 프론트엔드 호환성 100% 유지
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, List
import numpy as np
import torch
from PIL import Image
from fastapi import UploadFile

# AI 파이프라인 컴포넌트 import
from app.ai_pipeline.pipeline_manager import PipelineManager
from app.ai_pipeline.steps.step_01_human_parsing import HumanParsingStep
from app.ai_pipeline.steps.step_02_pose_estimation import PoseEstimationStep
from app.ai_pipeline.steps.step_03_cloth_segmentation import ClothSegmentationStep
from app.ai_pipeline.steps.step_04_geometric_matching import GeometricMatchingStep
from app.ai_pipeline.steps.step_05_cloth_warping import ClothWarpingStep
from app.ai_pipeline.steps.step_06_virtual_fitting import VirtualFittingStep
from app.ai_pipeline.steps.step_07_post_processing import PostProcessingStep
from app.ai_pipeline.steps.step_08_quality_assessment import QualityAssessmentStep

# 유틸리티들
try:
    from app.ai_pipeline.utils.model_loader import ModelLoader
    from app.ai_pipeline.utils.memory_manager import MemoryManager
    from app.ai_pipeline.utils.data_converter import DataConverter
    UTILS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"AI Pipeline Utils import 실패: {e}")
    UTILS_AVAILABLE = False

# 스키마
try:
    from app.models.schemas import BodyMeasurements, ClothingType, ProcessingStatus
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False
    # 폴백 스키마
    class BodyMeasurements:
        def __init__(self, height: float, weight: float, **kwargs):
            self.height = height
            self.weight = weight
            for k, v in kwargs.items():
                setattr(self, k, v)

# 로깅 설정
logger = logging.getLogger(__name__)

def get_optimal_device() -> str:
    """최적 디바이스 선택"""
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    except Exception as e:
        logger.warning(f"디바이스 감지 실패: {e}")
        return "cpu"

def optimize_device_memory(device: str):
    """디바이스별 메모리 최적화"""
    try:
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
        else:
            import gc
            gc.collect()
    except Exception as e:
        logger.warning(f"메모리 최적화 실패: {e}")

# ============================================================================
# 🔧 서비스 레이어 - 핵심 비즈니스 로직
# ============================================================================

class PipelineService:
    """
    파이프라인 서비스 레이어
    - 비즈니스 로직 중심화
    - API 레이어와 AI 처리 레이어 분리
    - 에러 처리 및 상태 관리
    - 로깅 및 모니터링
    """
    
    def __init__(self, device: Optional[str] = None):
        """서비스 초기화"""
        self.device = device or get_optimal_device()
        self.logger = logging.getLogger(f"services.{self.__class__.__name__}")
        
        # 파이프라인 매니저
        self.pipeline_manager: Optional[PipelineManager] = None
        
        # AI 단계들
        self.ai_steps: Dict[str, Any] = {}
        
        # 유틸리티들
        self.utils: Dict[str, Any] = {}
        
        # 상태 관리
        self.initialized = False
        self.processing_sessions = {}
        self.model_load_status = {}
        
        self.logger.info(f"🔧 PipelineService 초기화 - Device: {self.device}")
    
    async def initialize(self) -> bool:
        """서비스 초기화"""
        try:
            if self.initialized:
                return True
            
            self.logger.info("🚀 PipelineService 초기화 시작...")
            
            # 1. 메모리 최적화
            optimize_device_memory(self.device)
            
            # 2. 파이프라인 매니저 초기화
            await self._initialize_pipeline_manager()
            
            # 3. AI 단계들 초기화
            await self._initialize_ai_steps()
            
            # 4. 유틸리티 초기화
            await self._initialize_utilities()
            
            # 5. 상태 확인
            await self._check_initialization_status()
            
            self.initialized = True
            self.logger.info("✅ PipelineService 초기화 완료!")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ PipelineService 초기화 실패: {e}")
            raise
    
    async def _initialize_pipeline_manager(self):
        """파이프라인 매니저 초기화"""
        try:
            self.pipeline_manager = PipelineManager(device=self.device)
            
            if hasattr(self.pipeline_manager, 'initialize'):
                await self.pipeline_manager.initialize()
            
            self.logger.info("✅ PipelineManager 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"❌ PipelineManager 초기화 실패: {e}")
            raise
    
    async def _initialize_ai_steps(self):
        """AI 단계들 초기화"""
        try:
            step_classes = {
                "step_01": HumanParsingStep,
                "step_02": PoseEstimationStep,
                "step_03": ClothSegmentationStep,
                "step_04": GeometricMatchingStep,
                "step_05": ClothWarpingStep,
                "step_06": VirtualFittingStep,
                "step_07": PostProcessingStep,
                "step_08": QualityAssessmentStep
            }
            
            for step_name, step_class in step_classes.items():
                try:
                    step_instance = step_class(device=self.device)
                    
                    if hasattr(step_instance, 'initialize'):
                        await step_instance.initialize()
                    
                    self.ai_steps[step_name] = step_instance
                    self.logger.info(f"✅ {step_name} 초기화 완료")
                    
                except Exception as e:
                    self.logger.error(f"❌ {step_name} 초기화 실패: {e}")
                    # 개별 Step 실패는 전체 초기화를 중단시키지 않음
                    continue
            
            self.logger.info(f"✅ AI Steps 초기화 완료: {len(self.ai_steps)}/8")
            
        except Exception as e:
            self.logger.error(f"❌ AI Steps 초기화 실패: {e}")
            raise
    
    async def _initialize_utilities(self):
        """유틸리티 초기화"""
        try:
            if UTILS_AVAILABLE:
                self.utils = {
                    'model_loader': ModelLoader(device=self.device),
                    'memory_manager': MemoryManager(device=self.device),
                    'data_converter': DataConverter()
                }
                self.logger.info("✅ AI Pipeline Utils 초기화 완료")
            else:
                self.logger.warning("⚠️ AI Pipeline Utils 불가용")
                
        except Exception as e:
            self.logger.error(f"❌ 유틸리티 초기화 실패: {e}")
            # 유틸리티 실패는 전체 초기화를 중단시키지 않음
            self.utils = {}
    
    async def _check_initialization_status(self):
        """초기화 상태 확인"""
        try:
            for step_name, step_instance in self.ai_steps.items():
                if hasattr(step_instance, 'is_model_loaded'):
                    self.model_load_status[step_name] = step_instance.is_model_loaded()
                else:
                    self.model_load_status[step_name] = True
            
            self.logger.info(f"📊 모델 로드 상태: {self.model_load_status}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 상태 확인 실패: {e}")
    
    # ========================================================================
    # 🎯 핵심 비즈니스 로직 메서드들
    # ========================================================================
    
    async def process_step(self, step_id: int, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        개별 단계 처리 (통합 인터페이스)
        
        Args:
            step_id: 단계 번호 (1-8)
            inputs: 입력 데이터 딕셔너리
            
        Returns:
            Dict: 처리 결과
        """
        start_time = time.time()
        
        try:
            # 서비스 초기화 확인
            if not self.initialized:
                await self.initialize()
            
            # 메모리 최적화
            optimize_device_memory(self.device)
            
            # 단계별 처리
            if step_id == 1:
                return await self._process_step_1(inputs)
            elif step_id == 2:
                return await self._process_step_2(inputs)
            elif step_id == 3:
                return await self._process_step_3(inputs)
            elif step_id == 4:
                return await self._process_step_4(inputs)
            elif step_id == 5:
                return await self._process_step_5(inputs)
            elif step_id == 6:
                return await self._process_step_6(inputs)
            elif step_id == 7:
                return await self._process_step_7(inputs)
            elif step_id == 8:
                return await self._process_step_8(inputs)
            else:
                raise ValueError(f"지원되지 않는 단계 ID: {step_id}")
                
        except Exception as e:
            self.logger.error(f"❌ Step {step_id} 처리 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "step_id": step_id,
                "processing_time": time.time() - start_time,
                "device": self.device
            }
    
    async def _process_step_1(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """1단계: 이미지 업로드 검증"""
        start_time = time.time()
        
        try:
            person_image = inputs.get("person_image")
            clothing_image = inputs.get("clothing_image")
            
            if not person_image or not clothing_image:
                raise ValueError("person_image와 clothing_image가 필요합니다")
            
            # 파일 검증
            person_validation = await self._validate_image_file(person_image, "person")
            clothing_validation = await self._validate_image_file(clothing_image, "clothing")
            
            if not person_validation["valid"] or not clothing_validation["valid"]:
                return {
                    "success": False,
                    "error": "파일 검증 실패",
                    "details": {
                        "person_error": person_validation.get("error"),
                        "clothing_error": clothing_validation.get("error")
                    },
                    "step_id": 1,
                    "processing_time": time.time() - start_time,
                    "device": self.device
                }
            
            # 이미지 품질 분석
            person_img = await self._load_and_preprocess_image(person_image)
            clothing_img = await self._load_and_preprocess_image(clothing_image)
            
            person_quality = await self._analyze_image_quality(person_img, "person")
            clothing_quality = await self._analyze_image_quality(clothing_img, "clothing")
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "message": "이미지 업로드 검증 완료",
                "step_id": 1,
                "processing_time": processing_time,
                "confidence": min(person_quality["confidence"], clothing_quality["confidence"]),
                "device": self.device,
                "details": {
                    "person_analysis": person_quality,
                    "clothing_analysis": clothing_quality,
                    "ready_for_next_step": True
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Step 1 처리 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "step_id": 1,
                "processing_time": time.time() - start_time,
                "device": self.device
            }
    
    async def _process_step_3(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """3단계: 인간 파싱"""
        start_time = time.time()
        
        try:
            person_image = inputs.get("person_image")
            if not person_image:
                raise ValueError("person_image가 필요합니다")
            
            # 이미지 로드
            person_img = await self._load_and_preprocess_image(person_image)
            person_array = np.array(person_img)
            
            # AI 인간 파싱 처리
            if "step_01" in self.ai_steps:
                parsing_result = await self.ai_steps["step_01"].process(person_array)
                
                return {
                    "success": True,
                    "message": "인간 파싱 완료",
                    "step_id": 3,
                    "processing_time": time.time() - start_time,
                    "device": self.device,
                    "details": {
                        "detected_segments": parsing_result.get("detected_segments", []),
                        "confidence": parsing_result.get("confidence", 0.0),
                        "processing_method": "HumanParsingStep"
                    }
                }
            else:
                raise RuntimeError("HumanParsingStep이 초기화되지 않았습니다")
                
        except Exception as e:
            self.logger.error(f"❌ Step 3 처리 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "step_id": 3,
                "processing_time": time.time() - start_time,
                "device": self.device
            }
    
    async def _process_step_7(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """7단계: 가상 피팅"""
        start_time = time.time()
        
        try:
            person_image = inputs.get("person_image")
            clothing_image = inputs.get("clothing_image")
            clothing_type = inputs.get("clothing_type", "auto_detect")
            
            if not person_image or not clothing_image:
                raise ValueError("person_image와 clothing_image가 필요합니다")
            
            # 이미지 로드
            person_img = await self._load_and_preprocess_image(person_image)
            clothing_img = await self._load_and_preprocess_image(clothing_image)
            person_array = np.array(person_img)
            clothing_array = np.array(clothing_img)
            
            # AI 가상 피팅 처리
            if "step_06" in self.ai_steps:
                fitting_result = await self.ai_steps["step_06"].process(
                    person_array, clothing_array, clothing_type=clothing_type
                )
                
                return {
                    "success": True,
                    "message": "가상 피팅 완료",
                    "step_id": 7,
                    "processing_time": time.time() - start_time,
                    "device": self.device,
                    "details": {
                        "clothing_type": clothing_type,
                        "fitting_quality": fitting_result.get("quality", 0.0),
                        "confidence": fitting_result.get("confidence", 0.0),
                        "processing_method": "VirtualFittingStep"
                    }
                }
            else:
                raise RuntimeError("VirtualFittingStep이 초기화되지 않았습니다")
                
        except Exception as e:
            self.logger.error(f"❌ Step 7 처리 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "step_id": 7,
                "processing_time": time.time() - start_time,
                "device": self.device
            }
    
    async def process_full_pipeline(
        self, 
        person_image: UploadFile, 
        clothing_image: UploadFile, 
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """전체 파이프라인 처리"""
        start_time = time.time()
        
        try:
            # 서비스 초기화 확인
            if not self.initialized:
                await self.initialize()
            
            # 파이프라인 매니저를 통한 전체 처리
            if self.pipeline_manager:
                # 이미지 로드
                person_img = await self._load_and_preprocess_image(person_image)
                clothing_img = await self._load_and_preprocess_image(clothing_image)
                
                # 파이프라인 매니저 호출
                result = await self.pipeline_manager.process_complete_virtual_fitting(
                    person_img, clothing_img, options or {}
                )
                
                return {
                    "success": True,
                    "message": "전체 파이프라인 처리 완료",
                    "processing_time": time.time() - start_time,
                    "device": self.device,
                    "result": result
                }
            else:
                raise RuntimeError("PipelineManager가 초기화되지 않았습니다")
                
        except Exception as e:
            self.logger.error(f"❌ 전체 파이프라인 처리 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "device": self.device
            }
    
    # ========================================================================
    # 🔧 헬퍼 메서드들
    # ========================================================================
    
    async def _validate_image_file(self, file: UploadFile, file_type: str) -> Dict[str, Any]:
        """이미지 파일 검증"""
        try:
            max_size = 50 * 1024 * 1024  # 50MB
            if hasattr(file, 'size') and file.size and file.size > max_size:
                return {
                    "valid": False,
                    "error": f"{file_type} 이미지가 50MB를 초과합니다"
                }
            
            allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
            if file.content_type not in allowed_types:
                return {
                    "valid": False,
                    "error": f"{file_type} 이미지: 지원되지 않는 파일 형식"
                }
            
            return {"valid": True}
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"파일 검증 중 오류: {str(e)}"
            }
    
    async def _load_and_preprocess_image(self, file: UploadFile) -> Image.Image:
        """이미지 로드 및 전처리"""
        from io import BytesIO
        
        content = await file.read()
        await file.seek(0)
        image = Image.open(BytesIO(content)).convert('RGB')
        return image.resize((512, 512), Image.Resampling.LANCZOS)
    
    async def _analyze_image_quality(self, image: Image.Image, image_type: str) -> Dict[str, Any]:
        """이미지 품질 분석"""
        try:
            import cv2
            
            # 기본 품질 분석
            width, height = image.size
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(cv_image)
            
            # 품질 점수 계산
            quality_score = min(1.0, sharpness / 1000.0 + brightness / 255.0) / 2
            
            return {
                "confidence": quality_score,
                "quality_metrics": {
                    "sharpness": min(1.0, sharpness / 1000.0),
                    "brightness": brightness / 255.0,
                    "resolution": f"{width}x{height}"
                },
                "recommendations": [
                    f"이미지 품질: {'우수' if quality_score > 0.8 else '양호' if quality_score > 0.6 else '개선 필요'}",
                    f"해상도: {width}x{height}"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"이미지 품질 분석 실패: {e}")
            return {
                "confidence": 0.7,
                "quality_metrics": {"error": str(e)},
                "recommendations": ["기본 품질 분석 적용됨"]
            }
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            self.logger.info("🧹 PipelineService 정리 시작...")
            
            # AI 단계들 정리
            for step_name, step in self.ai_steps.items():
                try:
                    if hasattr(step, 'cleanup'):
                        await step.cleanup()
                except Exception as e:
                    self.logger.warning(f"⚠️ {step_name} 정리 실패: {e}")
            
            # 파이프라인 매니저 정리
            if self.pipeline_manager and hasattr(self.pipeline_manager, 'cleanup'):
                try:
                    await self.pipeline_manager.cleanup()
                except Exception as e:
                    self.logger.warning(f"⚠️ 파이프라인 매니저 정리 실패: {e}")
            
            # 메모리 정리
            optimize_device_memory(self.device)
            
            self.initialized = False
            self.logger.info("✅ PipelineService 정리 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 정리 실패: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """서비스 상태 반환"""
        return {
            "initialized": self.initialized,
            "device": self.device,
            "pipeline_manager": self.pipeline_manager is not None,
            "ai_steps_loaded": len(self.ai_steps),
            "ai_steps": list(self.ai_steps.keys()),
            "model_load_status": self.model_load_status,
            "utils_available": len(self.utils) > 0,
            "processing_sessions": len(self.processing_sessions),
            "service_type": "PipelineService"
        }


# ============================================================================
# 🎯 싱글톤 서비스 인스턴스
# ============================================================================

_pipeline_service_instance: Optional[PipelineService] = None

async def get_pipeline_service() -> PipelineService:
    """PipelineService 싱글톤 인스턴스 반환"""
    global _pipeline_service_instance
    
    if _pipeline_service_instance is None:
        _pipeline_service_instance = PipelineService()
        await _pipeline_service_instance.initialize()
        logger.info("✅ PipelineService 싱글톤 인스턴스 초기화 완료")
    
    return _pipeline_service_instance


# ============================================================================
# 🎉 EXPORT
# ============================================================================

__all__ = ["PipelineService", "get_pipeline_service"]