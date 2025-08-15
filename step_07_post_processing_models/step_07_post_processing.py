"""
Step 07: Post Processing Models

후처리 모델들을 실행하는 메인 스텝입니다.
논문 기반의 AI 모델 구조로 구현되었습니다.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List
import numpy as np
from PIL import Image
import logging

# 프로젝트 로깅 설정 import
from backend.app.ai_pipeline.utils.logging_config import get_logger

# 모델 로더와 추론 엔진 import
from .post_processing_model_loader import PostProcessingModelLoader
from .inference.inference_engine import PostProcessingInferenceEngine

logger = get_logger(__name__)


class PostProcessingStep:
    """
    후처리 모델들을 실행하는 메인 스텝 클래스
    
    지원 모델:
    - SwinIR (Super-Resolution)
    - Real-ESRGAN (Enhancement)
    - GFPGAN (Face Restoration)
    - CodeFormer (Face Restoration)
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Args:
            checkpoint_dir: 체크포인트 파일들이 저장된 디렉토리 경로
        """
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 로더와 추론 엔진 초기화
        self.model_loader = PostProcessingModelLoader(checkpoint_dir)
        self.inference_engine = PostProcessingInferenceEngine(self.model_loader)
        
        logger.info(f"PostProcessingStep initialized on device: {self.device}")
    
    def process_image(self, image: Union[np.ndarray, Image.Image, torch.Tensor], 
                     model_type: str = 'swinir', **kwargs) -> np.ndarray:
        """
        이미지를 후처리 모델로 처리합니다.
        
        Args:
            image: 입력 이미지
            model_type: 사용할 모델 타입 ('swinir', 'realesrgan', 'gfpgan', 'codeformer')
            **kwargs: 추가 처리 파라미터
            
        Returns:
            처리된 이미지
        """
        try:
            logger.info(f"Processing image with {model_type} model")
            
            # 추론 엔진을 통해 이미지 처리
            result = self.inference_engine.process_image(image, model_type, **kwargs)
            
            logger.info(f"Successfully processed image with {model_type}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing image with {model_type}: {str(e)}")
            raise RuntimeError(f"Failed to process image with {model_type}: {str(e)}")
    
    def batch_process(self, images: List[Union[np.ndarray, Image.Image, torch.Tensor]], 
                     model_type: str = 'swinir', **kwargs) -> List[np.ndarray]:
        """
        여러 이미지를 배치로 처리합니다.
        
        Args:
            images: 입력 이미지 리스트
            model_type: 사용할 모델 타입
            **kwargs: 추가 처리 파라미터
            
        Returns:
            처리된 이미지 리스트
        """
        try:
            logger.info(f"Batch processing {len(images)} images with {model_type} model")
            
            # 추론 엔진을 통해 배치 처리
            results = self.inference_engine.batch_process(images, model_type, **kwargs)
            
            logger.info(f"Successfully batch processed {len(images)} images with {model_type}")
            return results
            
        except Exception as e:
            logger.error(f"Error batch processing images with {model_type}: {str(e)}")
            raise RuntimeError(f"Failed to batch process images with {model_type}: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """
        사용 가능한 모델 타입들을 반환합니다.
        
        Returns:
            사용 가능한 모델 타입 리스트
        """
        return list(self.model_loader.supported_models.keys())
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """
        모델의 정보를 반환합니다.
        
        Args:
            model_type: 모델 타입
            
        Returns:
            모델 정보 딕셔너리
        """
        return self.model_loader.get_model_info(model_type)
    
    def validate_model(self, model_type: str) -> Dict[str, Any]:
        """
        모델의 유효성을 검증합니다.
        
        Args:
            model_type: 모델 타입
            
        Returns:
            검증 결과 딕셔너리
        """
        return self.model_loader.validate_model(model_type)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        현재 메모리 사용량을 반환합니다.
        
        Returns:
            메모리 사용량 정보
        """
        return self.model_loader.get_memory_usage()
    
    def get_inference_config(self, model_type: str) -> Dict[str, Any]:
        """
        모델의 추론 설정을 반환합니다.
        
        Args:
            model_type: 모델 타입
            
        Returns:
            추론 설정 딕셔너리
        """
        return self.inference_engine.get_inference_config(model_type)
    
    def update_inference_config(self, model_type: str, config: Dict[str, Any]) -> bool:
        """
        모델의 추론 설정을 업데이트합니다.
        
        Args:
            model_type: 모델 타입
            config: 새로운 설정
            
        Returns:
            업데이트 성공 여부
        """
        return self.inference_engine.update_inference_config(model_type, config)
    
    def unload_model(self, model_type: str) -> bool:
        """
        지정된 모델을 메모리에서 해제합니다.
        
        Args:
            model_type: 모델 타입
            
        Returns:
            해제 성공 여부
        """
        return self.model_loader.unload_model(model_type)
    
    def unload_all_models(self) -> int:
        """
        모든 로드된 모델을 메모리에서 해제합니다.
        
        Returns:
            해제된 모델 수
        """
        return self.model_loader.unload_all_models()
    
    def get_loaded_model_types(self) -> List[str]:
        """
        현재 로드된 모델 타입들을 반환합니다.
        
        Returns:
            로드된 모델 타입 리스트
        """
        return self.model_loader.get_loaded_model_types()


def main():
    """메인 실행 함수"""
    try:
        logger.info("Starting Post Processing Step...")
        
        # PostProcessingStep 인스턴스 생성
        post_processing = PostProcessingStep()
        
        # 사용 가능한 모델 확인
        available_models = post_processing.get_available_models()
        logger.info(f"Available models: {available_models}")
        
        # 모델 정보 출력
        for model_type in available_models:
            model_info = post_processing.get_model_info(model_type)
            if model_info:
                logger.info(f"Model {model_type}: {model_info}")
        
        # 메모리 사용량 확인
        memory_info = post_processing.get_memory_usage()
        logger.info(f"Memory usage: {memory_info}")
        
        logger.info("Post Processing Step completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in Post Processing Step: {str(e)}")
        raise


if __name__ == "__main__":
    main()
