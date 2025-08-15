"""
🔥 Quality Assessment Inference Engine
=====================================

품질 평가 모델들의 추론을 담당하는 엔진 클래스입니다.
논문 기반의 AI 모델 구조에 맞춰 구현되었습니다.

지원 모델:
- QualityNet (Image Quality Assessment)
- BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
- NIQE (Natural Image Quality Evaluator)
- PIQE (Perception-based Image Quality Evaluator)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import cv2
from PIL import Image
import logging

# 프로젝트 로깅 설정 import
try:
    from backend.app.core.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

class QualityAssessmentInferenceEngine:
    """
    품질 평가 모델들의 추론을 담당하는 엔진 클래스

    지원 모델:
    - QualityNet (Image Quality Assessment)
    - BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
    - NIQE (Natural Image Quality Evaluator)
    - PIQE (Perception-based Image Quality Evaluator)
    """

    def __init__(self, model_loader=None):
        """
        Args:
            model_loader: 모델 로더 인스턴스
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loader = model_loader
        self.loaded_models = {}

        # 지원하는 모델 타입들
        self.supported_models = ['qualitynet', 'brisque', 'niqe', 'piqe']

        # 추론 설정
        self.inference_config = {
            'qualitynet': {
                'input_size': (224, 224),
                'batch_size': 32,
                'normalize': True,
                'use_tta': True
            },
            'brisque': {
                'patch_size': 96,
                'stride': 48,
                'normalize': True
            },
            'niqe': {
                'patch_size': 96,
                'stride': 48,
                'normalize': True
            },
            'piqe': {
                'patch_size': 96,
                'stride': 48,
                'normalize': True
            }
        }

        logger.info(f"QualityAssessmentInferenceEngine initialized on device: {self.device}")

    def load_model(self, model_type: str, **kwargs) -> nn.Module:
        """
        지정된 타입의 모델을 로드합니다.

        Args:
            model_type: 모델 타입
            **kwargs: 모델 로드에 필요한 추가 파라미터

        Returns:
            로드된 모델
        """
        if self.model_loader:
            return self.model_loader.load_model(model_type, **kwargs)
        else:
            raise RuntimeError("Model loader not initialized")

    def assess_image_quality(self, image: Union[np.ndarray, Image.Image, torch.Tensor],
                           model_type: str = 'qualitynet', **kwargs) -> Dict[str, Any]:
        """
        이미지 품질을 평가합니다.

        Args:
            image: 입력 이미지
            model_type: 사용할 모델 타입
            **kwargs: 추가 파라미터

        Returns:
            품질 평가 결과 딕셔너리
        """
        try:
            # 이미지 전처리
            processed_image = self._preprocess_image(image, model_type)
            
            # 모델 로드 (필요시)
            if model_type not in self.loaded_models:
                self.loaded_models[model_type] = self.load_model(model_type, **kwargs)
            
            model = self.loaded_models[model_type]
            model.eval()
            
            # 추론 실행
            with torch.no_grad():
                if model_type == 'qualitynet':
                    result = self._inference_qualitynet(model, processed_image, **kwargs)
                elif model_type == 'brisque':
                    result = self._inference_brisque(model, processed_image, **kwargs)
                elif model_type == 'niqe':
                    result = self._inference_niqe(model, processed_image, **kwargs)
                elif model_type == 'piqe':
                    result = self._inference_piqe(model, processed_image, **kwargs)
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
            
            # 결과 후처리
            result = self._postprocess_result(result, model_type)
            
            logger.info(f"✅ Quality assessment completed for {model_type}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Quality assessment failed: {e}")
            raise

    def _preprocess_image(self, image: Union[np.ndarray, Image.Image, torch.Tensor],
                         model_type: str) -> torch.Tensor:
        """
        이미지를 전처리합니다.
        """
        # PIL Image를 numpy array로 변환
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # numpy array를 torch tensor로 변환
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                image = torch.from_numpy(image).permute(2, 0, 1).float()
            else:
                image = torch.from_numpy(image).unsqueeze(0).float()
        
        # 정규화
        if self.inference_config[model_type].get('normalize', False):
            image = image / 255.0
        
        # 배치 차원 추가
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # 디바이스 이동
        image = image.to(self.device)
        
        return image

    def _inference_qualitynet(self, model: nn.Module, image: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """
        QualityNet 모델로 추론을 실행합니다.
        """
        # 입력 크기 조정
        input_size = self.inference_config['qualitynet']['input_size']
        if image.shape[-2:] != input_size:
            image = F.interpolate(image, size=input_size, mode='bilinear', align_corners=False)
        
        # 추론 실행
        output = model(image)
        
        # 결과 해석
        if isinstance(output, torch.Tensor):
            quality_score = output.item() if output.numel() == 1 else output.mean().item()
        else:
            quality_score = float(output)
        
        return {
            'quality_score': quality_score,
            'confidence': 0.95,
            'model_type': 'qualitynet'
        }

    def _inference_brisque(self, model: nn.Module, image: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """
        BRISQUE 모델로 추론을 실행합니다.
        """
        # BRISQUE는 패치 기반 분석
        patch_size = self.inference_config['brisque']['patch_size']
        stride = self.inference_config['brisque']['stride']
        
        # 이미지를 패치로 분할
        patches = self._extract_patches(image, patch_size, stride)
        
        # 각 패치에 대해 품질 평가
        patch_scores = []
        for patch in patches:
            with torch.no_grad():
                score = model(patch.unsqueeze(0))
                patch_scores.append(score.item())
        
        # 전체 품질 점수 계산
        quality_score = np.mean(patch_scores)
        
        return {
            'quality_score': quality_score,
            'confidence': 0.90,
            'model_type': 'brisque',
            'patch_scores': patch_scores
        }

    def _inference_niqe(self, model: nn.Module, image: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """
        NIQE 모델로 추론을 실행합니다.
        """
        # NIQE는 자연스러운 이미지 품질 평가
        # 낮은 점수가 더 좋은 품질을 의미
        quality_score = model(image)
        
        return {
            'quality_score': quality_score.item(),
            'confidence': 0.88,
            'model_type': 'niqe'
        }

    def _inference_piqe(self, model: nn.Module, image: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """
        PIQE 모델로 추론을 실행합니다.
        """
        # PIQE는 지각 기반 품질 평가
        quality_score = model(image)
        
        return {
            'quality_score': quality_score.item(),
            'confidence': 0.92,
            'model_type': 'piqe'
        }

    def _extract_patches(self, image: torch.Tensor, patch_size: int, stride: int) -> List[torch.Tensor]:
        """
        이미지에서 패치를 추출합니다.
        """
        patches = []
        h, w = image.shape[-2:]
        
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = image[..., i:i + patch_size, j:j + patch_size]
                patches.append(patch)
        
        return patches

    def _postprocess_result(self, result: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """
        추론 결과를 후처리합니다.
        """
        # 품질 등급 분류
        quality_score = result['quality_score']
        
        if model_type == 'qualitynet':
            # QualityNet: 높은 점수가 좋은 품질
            if quality_score >= 0.8:
                quality_grade = 'Excellent'
            elif quality_score >= 0.6:
                quality_grade = 'Good'
            elif quality_score >= 0.4:
                quality_grade = 'Fair'
            else:
                quality_grade = 'Poor'
        elif model_type == 'brisque':
            # BRISQUE: 낮은 점수가 좋은 품질
            if quality_score <= 20:
                quality_grade = 'Excellent'
            elif quality_score <= 40:
                quality_grade = 'Good'
            elif quality_score <= 60:
                quality_grade = 'Fair'
            else:
                quality_grade = 'Poor'
        elif model_type == 'niqe':
            # NIQE: 낮은 점수가 좋은 품질
            if quality_score <= 3:
                quality_grade = 'Excellent'
            elif quality_score <= 5:
                quality_grade = 'Good'
            elif quality_score <= 7:
                quality_grade = 'Fair'
            else:
                quality_grade = 'Poor'
        elif model_type == 'piqe':
            # PIQE: 낮은 점수가 좋은 품질
            if quality_score <= 30:
                quality_grade = 'Excellent'
            elif quality_score <= 50:
                quality_grade = 'Good'
            elif quality_score <= 70:
                quality_grade = 'Fair'
            else:
                quality_grade = 'Poor'
        else:
            quality_grade = 'Unknown'
        
        # 결과에 품질 등급 추가
        result['quality_grade'] = quality_grade
        result['timestamp'] = torch.cuda.Event() if torch.cuda.is_available() else None
        
        return result

    def batch_assess_quality(self, images: List[Union[np.ndarray, Image.Image, torch.Tensor]],
                           model_type: str = 'qualitynet', **kwargs) -> List[Dict[str, Any]]:
        """
        여러 이미지의 품질을 일괄 평가합니다.
        """
        results = []
        for i, image in enumerate(images):
            try:
                result = self.assess_image_quality(image, model_type, **kwargs)
                result['image_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"❌ Failed to assess image {i}: {e}")
                results.append({
                    'image_index': i,
                    'error': str(e),
                    'quality_score': 0.0,
                    'quality_grade': 'Error'
                })
        
        return results

    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """
        모델 정보를 반환합니다.
        """
        if model_type in self.inference_config:
            return {
                'model_type': model_type,
                'supported': True,
                'config': self.inference_config[model_type],
                'device': str(self.device)
            }
        else:
            return {
                'model_type': model_type,
                'supported': False,
                'error': 'Unsupported model type'
            }
