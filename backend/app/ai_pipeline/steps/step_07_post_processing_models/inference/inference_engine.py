"""
Post Processing Inference Engine

후처리 모델들의 추론을 담당하는 엔진 클래스입니다.
논문 기반의 AI 모델 구조에 맞춰 구현되었습니다.
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
from backend.app.core.logging_config import get_logger

logger = get_logger(__name__)


class PostProcessingInferenceEngine:
    """
    후처리 모델들의 추론을 담당하는 엔진 클래스

    지원 모델:
    - SwinIR (Super-Resolution)
    - Real-ESRGAN (Enhancement)
    - GFPGAN (Face Restoration)
    - CodeFormer (Face Restoration)
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
        self.supported_models = ['swinir', 'realesrgan', 'gfpgan', 'codeformer']

        # 추론 설정
        self.inference_config = {
            'swinir': {
                'tile_size': 400,
                'tile_pad': 10,
                'pre_pad': 0,
                'half': True
            },
            'realesrgan': {
                'tile_size': 400,
                'tile_pad': 10,
                'pre_pad': 0,
                'half': True
            },
            'gfpgan': {
                'bg_upsampler': True,
                'bg_tile': 400,
                'suffix': None,
                'only_center_face': False,
                'aligned': False
            },
            'codeformer': {
                'background_enhance': True,
                'face_upsample': True,
                'upscale': 2,
                'codeformer_fidelity': 0.7
            }
        }

        logger.info(f"PostProcessingInferenceEngine initialized on device: {self.device}")

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

    def process_image(self, image: Union[np.ndarray, Image.Image, torch.Tensor],
                      model_type: str, **kwargs) -> np.ndarray:
        """
        이미지를 후처리 모델로 처리합니다.

        Args:
            image: 입력 이미지
            model_type: 사용할 모델 타입
            **kwargs: 추가 처리 파라미터

        Returns:
            처리된 이미지
        """
        try:
            logger.info(f"Processing image with {model_type} model")

            # 이미지 전처리
            processed_image = self._preprocess_image(image, model_type)

            # 모델 추론
            if model_type == 'swinir':
                result = self._inference_swinir(processed_image, **kwargs)
            elif model_type == 'realesrgan':
                result = self._inference_realesrgan(processed_image, **kwargs)
            elif model_type == 'gfpgan':
                result = self._inference_gfpgan(processed_image, **kwargs)
            elif model_type == 'codeformer':
                result = self._inference_codeformer(processed_image, **kwargs)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # 이미지 후처리
            final_result = self._postprocess_image(result, model_type)

            logger.info(f"Successfully processed image with {model_type}")
            return final_result

        except Exception as e:
            logger.error(f"Error processing image with {model_type}: {str(e)}")
            raise RuntimeError(f"Failed to process image with {model_type}: {str(e)}")

    def _preprocess_image(self, image: Union[np.ndarray, Image.Image, torch.Tensor],
                          model_type: str) -> torch.Tensor:
        """
        이미지를 모델 입력에 맞게 전처리합니다.

        Args:
            image: 입력 이미지
            model_type: 모델 타입

        Returns:
            전처리된 이미지 텐서
        """
        # 이미지 타입 통일
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            if image.dtype != torch.uint8:
                image = (image * 255).to(torch.uint8)
            image = image.cpu().numpy()
            image = Image.fromarray(image)

        # PIL 이미지를 텐서로 변환
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 텐서 변환
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (B, C, H, W)

        # 디바이스 이동
        image_tensor = image_tensor.to(self.device)

        # 모델별 특화 전처리
        if model_type in ['swinir', 'realesrgan']:
            # 정규화
            image_tensor = image_tensor * 2 - 1  # [-1, 1] 범위로 변환

        return image_tensor

    def _postprocess_image(self, image_tensor: torch.Tensor, model_type: str) -> np.ndarray:
        """
        모델 출력을 이미지로 후처리합니다.

        Args:
            image_tensor: 모델 출력 텐서
            model_type: 모델 타입

        Returns:
            후처리된 이미지 배열
        """
        # 디바이스에서 CPU로 이동
        image_tensor = image_tensor.detach().cpu()

        # 모델별 특화 후처리
        if model_type in ['swinir', 'realesrgan']:
            # [-1, 1] 범위를 [0, 1]로 변환
            image_tensor = (image_tensor + 1) / 2

        # 클램핑
        image_tensor = torch.clamp(image_tensor, 0, 1)

        # 차원 변환 및 numpy 변환
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)  # (C, H, W)

        image_tensor = image_tensor.permute(1, 2, 0)  # (H, W, C)
        image_array = (image_tensor.numpy() * 255).astype(np.uint8)

        return image_array

    def _inference_swinir(self, image_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        SwinIR 모델로 추론을 수행합니다.

        Args:
            image_tensor: 입력 이미지 텐서
            **kwargs: 추가 파라미터

        Returns:
            추론 결과 텐서
        """
        # 모델 로드
        model = self.load_model('swinir')

        # 설정 가져오기
        config = self.inference_config['swinir'].copy()
        config.update(kwargs)

        # 타일링 처리 (메모리 효율성)
        if image_tensor.shape[-1] > config['tile_size'] or image_tensor.shape[-2] > config['tile_size']:
            result = self._tile_inference_swinir(model, image_tensor, config)
        else:
            with torch.no_grad():
                result = model(image_tensor)

        return result

    def _inference_realesrgan(self, image_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Real-ESRGAN 모델로 추론을 수행합니다.

        Args:
            image_tensor: 입력 이미지 텐서
            **kwargs: 추가 파라미터

        Returns:
            추론 결과 텐서
        """
        # 모델 로드
        model = self.load_model('realesrgan')

        # 설정 가져오기
        config = self.inference_config['realesrgan'].copy()
        config.update(kwargs)

        # 타일링 처리
        if image_tensor.shape[-1] > config['tile_size'] or image_tensor.shape[-2] > config['tile_size']:
            result = self._tile_inference_realesrgan(model, image_tensor, config)
        else:
            with torch.no_grad():
                result = model(image_tensor)

        return result

    def _inference_gfpgan(self, image_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        GFPGAN 모델로 추론을 수행합니다.

        Args:
            image_tensor: 입력 이미지 텐서
            **kwargs: 추가 파라미터

        Returns:
            추론 결과 텐서
        """
        # 모델 로드
        model = self.load_model('gfpgan')

        # 설정 가져오기
        config = self.inference_config['gfpgan'].copy()
        config.update(kwargs)

        with torch.no_grad():
            result = model(image_tensor)

        return result

    def _inference_codeformer(self, image_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        CodeFormer 모델로 추론을 수행합니다.

        Args:
            image_tensor: 입력 이미지 텐서
            **kwargs: 추가 처리 파라미터

        Returns:
            추론 결과 텐서
        """
        # 모델 로드
        model = self.load_model('codeformer')

        # 설정 가져오기
        config = self.inference_config['codeformer'].copy()
        config.update(kwargs)

        with torch.no_grad():
            result = model(image_tensor, **config)

        return result

    def _tile_inference_swinir(self, model: nn.Module, image_tensor: torch.Tensor,
                                config: Dict[str, Any]) -> torch.Tensor:
        """
        SwinIR 모델의 타일링 추론을 수행합니다.

        Args:
            model: SwinIR 모델
            image_tensor: 입력 이미지 텐서
            config: 추론 설정

        Returns:
            타일링 처리된 결과 텐서
        """
        b, c, h, w = image_tensor.shape
        tile_size = config['tile_size']
        tile_pad = config['tile_pad']

        # 타일 크기 계산
        tile_h = min(tile_size, h)
        tile_w = min(tile_size, w)

        # 결과 텐서 초기화
        result = torch.zeros_like(image_tensor)

        # 타일별 처리
        for i in range(0, h, tile_h):
            for j in range(0, w, tile_w):
                # 타일 추출
                tile = image_tensor[:, :, i:i+tile_h, j:j+tile_w]

                # 패딩 추가
                if tile_pad > 0:
                    tile = F.pad(tile, (tile_pad, tile_pad, tile_pad, tile_pad), 'reflect')

                # 추론
                with torch.no_grad():
                    tile_result = model(tile)

                # 패딩 제거
                if tile_pad > 0:
                    tile_result = tile_result[:, :, tile_pad:-tile_pad, tile_pad:-tile_pad]

                # 결과에 복사
                result[:, :, i:i+tile_h, j:j+tile_w] = tile_result

        return result

    def _tile_inference_realesrgan(self, model: nn.Module, image_tensor: torch.Tensor,
                                   config: Dict[str, Any]) -> torch.Tensor:
        """
        Real-ESRGAN 모델의 타일링 추론을 수행합니다.

        Args:
            model: Real-ESRGAN 모델
            image_tensor: 입력 이미지 텐서
            config: 추론 설정

        Returns:
            타일링 처리된 결과 텐서
        """
        b, c, h, w = image_tensor.shape
        tile_size = config['tile_size']
        tile_pad = config['tile_pad']

        # 타일 크기 계산
        tile_h = min(tile_size, h)
        tile_w = min(tile_size, w)

        # 결과 텐서 초기화
        result = torch.zeros_like(image_tensor)

        # 타일별 처리
        for i in range(0, h, tile_h):
            for j in range(0, w, tile_w):
                # 타일 추출
                tile = image_tensor[:, :, i:i+tile_h, j:j+tile_w]

                # 패딩 추가
                if tile_pad > 0:
                    tile = F.pad(tile, (tile_pad, tile_pad, tile_pad, tile_pad), 'reflect')

                # 추론
                with torch.no_grad():
                    tile_result = model(tile)

                # 패딩 제거
                if tile_pad > 0:
                    tile_result = tile_result[:, :, tile_pad:-tile_pad, tile_pad:-tile_pad]

                # 결과에 복사
                result[:, :, i:i+tile_h, j:j+tile_w] = tile_result

        return result

    def batch_process(self, images: List[Union[np.ndarray, Image.Image, torch.Tensor]],
                      model_type: str, **kwargs) -> List[np.ndarray]:
        """
        여러 이미지를 배치로 처리합니다.

        Args:
            images: 입력 이미지 리스트
            model_type: 사용할 모델 타입
            **kwargs: 추가 처리 파라미터

        Returns:
            처리된 이미지 리스트
        """
        results = []

        for i, image in enumerate(images):
            try:
                logger.info(f"Processing image {i+1}/{len(images)} with {model_type}")
                result = self.process_image(image, model_type, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing image {i+1}: {str(e)}")
                # 에러 발생 시 원본 이미지 반환
                if isinstance(image, np.ndarray):
                    results.append(image)
                elif isinstance(image, Image.Image):
                    results.append(np.array(image))
                else:
                    results.append(image.cpu().numpy())

        return results

    def get_inference_config(self, model_type: str) -> Dict[str, Any]:
        """
        모델의 추론 설정을 반환합니다.

        Args:
            model_type: 모델 타입

        Returns:
            추론 설정 딕셔너리
        """
        return self.inference_config.get(model_type, {}).copy()

    def update_inference_config(self, model_type: str, config: Dict[str, Any]) -> bool:
        """
        모델의 추론 설정을 업데이트합니다.

        Args:
            model_type: 모델 타입
            config: 새로운 설정

        Returns:
            업데이트 성공 여부
        """
        if model_type in self.inference_config:
            self.inference_config[model_type].update(config)
            logger.info(f"Updated inference config for {model_type}")
            return True
        else:
            logger.warning(f"Unknown model type: {model_type}")
            return False
