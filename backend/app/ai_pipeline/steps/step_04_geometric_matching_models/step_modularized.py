#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 04: Geometric Matching - Modularized Version
==================================================================

✅ 기존 step.py 기능 그대로 보존
✅ 분리된 모듈들 사용 (core/, models/, utils/)
✅ 모듈화된 구조 적용
✅ 중복 코드 제거
✅ 유지보수성 향상

파일 위치: backend/app/ai_pipeline/steps/04_geometric_matching/step_modularized.py
작성자: MyCloset AI Team  
날짜: 2025-08-09
버전: v1.0 (Modularized)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# 🔥 분리된 모듈들 import
from .core import (
    BaseOpticalFlowModel, BaseGeometricMatcher,
    CommonBottleneckBlock, CommonConvBlock, CommonInitialConv,
    CommonFeatureExtractor, CommonAttentionBlock, CommonGRUConvBlock,
    GeometricMatchingConfig, ProcessingStatus,
    GeometricMatchingInitializer, GeometricMatchingModelLoader, GeometricMatchingProcessor
)

from .models import (
    DeepLabV3PlusBackbone, ASPPModule,
    SelfAttentionKeypointMatcher, EdgeAwareTransformationModule, ProgressiveGeometricRefinement,
    GeometricMatchingModule, SimpleTPS, TPSGridGenerator, BottleneckBlock,
    OpticalFlowNetwork, KeypointMatchingNetwork,
    CompleteAdvancedGeometricMatchingAI, AdvancedGeometricMatcher
)

from .utils import EnhancedModelPathMapper

# BaseStepMixin import
from app.ai_pipeline.steps.base.base_step_mixin import BaseStepMixin

logger = logging.getLogger(__name__)

# ==============================================
# 🔥 GeometricMatchingStep - 모듈화된 버전
# ==============================================

class GeometricMatchingStep(BaseStepMixin):
    """
    🔥 Geometric Matching Step - 모듈화된 버전
    
    ✅ 기존 step.py 기능 그대로 보존
    ✅ 분리된 모듈들 사용
    ✅ 중복 코드 제거
    ✅ 유지보수성 향상
    """
    
    def __init__(self, **kwargs):
        """초기화"""
        super().__init__(**kwargs)
        
        # 초기화 유틸리티 사용
        self.initializer = GeometricMatchingInitializer()
        self.model_loader = GeometricMatchingModelLoader()
        self.processor = GeometricMatchingProcessor()
        
        # 기본 속성 초기화
        self.initializer.initialize_step_attributes(self)
        
        # 기하학적 매칭 특화 속성 초기화
        self.initializer.initialize_geometric_matching_specifics(self, **kwargs)
        
        # 모델 로딩 상태
        self.processing_status = ProcessingStatus()
        
        # 설정
        self.config = GeometricMatchingConfig()
        
        # 모델들
        self.geometric_matching_models = {}
        self.advanced_ai_models = {}
        
        logger.info("🔥 GeometricMatchingStep 초기화 완료")

    def process(self, **kwargs) -> Dict[str, Any]:
        """
        🔥 기하학적 매칭 처리 - 메인 프로세스
        
        Args:
            **kwargs: 입력 데이터 (person_image, clothing_image 등)
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        start_time = time.time()
        
        try:
            # 입력 검증 및 추출
            person_image, clothing_image, person_parsing_data, pose_data, clothing_segmentation_data = \
                self.processor.validate_and_extract_inputs(kwargs)
            
            # 이미지 텐서 준비
            person_tensor = self.processor.prepare_image_tensor(person_image, self.device)
            clothing_tensor = self.processor.prepare_image_tensor(clothing_image, self.device)
            
            # AI 모델 실행
            inference_results = self.processor.execute_all_ai_models(
                self, person_tensor, clothing_tensor,
                person_parsing_data, pose_data, clothing_segmentation_data
            )
            
            # 결과 후처리
            final_result = self.processor.postprocess_geometric_matching_result(
                inference_results, person_tensor, clothing_tensor
            )
            
            # 성능 통계 업데이트
            processing_time = time.time() - start_time
            self.processor.update_performance_stats(self, processing_time, True)
            
            logger.info(f"✅ 기하학적 매칭 처리 완료 (소요시간: {processing_time:.2f}초)")
            
            return final_result
            
        except Exception as e:
            logger.error(f"❌ 기하학적 매칭 처리 실패: {e}")
            self.processor.update_performance_stats(self, 0.0, False)
            return self.processor.create_error_response(str(e))

    def _run_ai_inference(self, processed_input: Dict[str, Any]) -> Dict[str, Any]:
        """AI 추론 실행"""
        try:
            # 모델 로딩 확인
            if not self.geometric_matching_models:
                self.model_loader.load_geometric_matching_models(self)
            
            # AI 모델 실행
            results = self.processor.execute_all_ai_models(
                self,
                processed_input['person_tensor'],
                processed_input['clothing_tensor'],
                processed_input.get('person_parsing_data'),
                processed_input.get('pose_data'),
                processed_input.get('clothing_segmentation_data')
            )
            
            return results
            
        except Exception as e:
            logger.error(f"❌ AI 추론 실행 실패: {e}")
            return {'error': str(e)}

    def get_model_status(self) -> Dict[str, Any]:
        """모델 상태 반환"""
        return {
            'geometric_matching_models_loaded': len(self.geometric_matching_models) > 0,
            'advanced_ai_models_loaded': len(self.advanced_ai_models) > 0,
            'device': self.device,
            'processing_status': self.processing_status.get_status_summary()
        }

    async def initialize(self):
        """비동기 초기화"""
        try:
            logger.info("🔄 GeometricMatchingStep 비동기 초기화 시작...")
            
            # 모델 로딩
            self.model_loader.load_geometric_matching_models(self)
            
            # 초기화 완료
            self.processing_status.update_status(initialization_complete=True)
            
            logger.info("✅ GeometricMatchingStep 비동기 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ 비동기 초기화 실패: {e}")

    async def cleanup(self):
        """리소스 정리"""
        try:
            logger.info("🔄 GeometricMatchingStep 리소스 정리 시작...")
            
            # 모델들 정리
            self.geometric_matching_models.clear()
            self.advanced_ai_models.clear()
            
            # 캐시 정리
            self.cache.clear()
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.backends.mps.empty_cache()
            
            logger.info("✅ GeometricMatchingStep 리소스 정리 완료")
            
        except Exception as e:
            logger.error(f"❌ 리소스 정리 실패: {e}")

# ==============================================
# 🔥 Factory Functions
# ==============================================

async def create_geometric_matching_step(
    device: str = "auto",
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> GeometricMatchingStep:
    """
    🔥 GeometricMatchingStep 비동기 생성 함수
    
    Args:
        device: 디바이스 설정
        config: 설정 딕셔너리
        **kwargs: 추가 인자
        
    Returns:
        GeometricMatchingStep: 생성된 스텝 인스턴스
    """
    step = GeometricMatchingStep(device=device, **kwargs)
    await step.initialize()
    return step

def create_geometric_matching_step_sync(
    device: str = "auto",
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> GeometricMatchingStep:
    """
    🔥 GeometricMatchingStep 동기 생성 함수
    
    Args:
        device: 디바이스 설정
        config: 설정 딕셔너리
        **kwargs: 추가 인자
        
    Returns:
        GeometricMatchingStep: 생성된 스텝 인스턴스
    """
    return GeometricMatchingStep(device=device, **kwargs)
