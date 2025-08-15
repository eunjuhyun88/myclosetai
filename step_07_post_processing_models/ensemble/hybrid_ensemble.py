"""
Hybrid Ensemble for Post Processing Models

후처리 모델들의 하이브리드 앙상블을 구현합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from PIL import Image
import logging

# 프로젝트 로깅 설정 import
from backend.app.ai_pipeline.utils.logging_config import get_logger

logger = get_logger(__name__)


class HybridEnsemble(nn.Module):
    """
    후처리 모델들의 하이브리드 앙상블 클래스
    
    여러 모델의 결과를 조합하여 최적의 결과를 생성합니다.
    """
    
    def __init__(self, model_loader, ensemble_method='weighted_average'):
        """
        Args:
            model_loader: 모델 로더 인스턴스
            ensemble_method: 앙상블 방법 ('weighted_average', 'voting', 'stacking')
        """
        super().__init__()
        
        self.model_loader = model_loader
        self.ensemble_method = ensemble_method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 앙상블에 사용할 모델 타입들
        self.ensemble_models = ['swinir', 'realesrgan', 'gfpgan', 'codeformer']
        
        # 모델별 가중치 (기본값)
        self.model_weights = {
            'swinir': 0.3,
            'realesrgan': 0.3,
            'gfpgan': 0.2,
            'codeformer': 0.2
        }
        
        # 앙상블 방법별 함수 매핑
        self.ensemble_methods = {
            'weighted_average': self._weighted_average_ensemble,
            'voting': self._voting_ensemble,
            'stacking': self._stacking_ensemble,
            'adaptive': self._adaptive_ensemble
        }
        
        # 메타 모델 (스태킹용)
        if ensemble_method == 'stacking':
            self.meta_model = self._create_meta_model()
        
        logger.info(f"HybridEnsemble initialized with method: {ensemble_method}")
    
    def _create_meta_model(self) -> nn.Module:
        """메타 모델을 생성합니다 (스태킹 앙상블용)"""
        return nn.Sequential(
            nn.Linear(len(self.ensemble_models), 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: Union[np.ndarray, Image.Image, torch.Tensor], 
                model_types: Optional[List[str]] = None, **kwargs) -> np.ndarray:
        """
        입력 이미지를 앙상블 모델들로 처리합니다.
        
        Args:
            x: 입력 이미지
            model_types: 사용할 모델 타입들 (None이면 모든 모델 사용)
            **kwargs: 추가 파라미터
            
        Returns:
            앙상블 처리된 이미지
        """
        if model_types is None:
            model_types = self.ensemble_models
        
        try:
            logger.info(f"Processing image with hybrid ensemble: {model_types}")
            
            # 각 모델로 처리
            model_outputs = {}
            for model_type in model_types:
                if model_type in self.ensemble_models:
                    try:
                        model = self.model_loader.load_model(model_type)
                        with torch.no_grad():
                            output = model(x)
                        model_outputs[model_type] = output
                        logger.info(f"Successfully processed with {model_type}")
                    except Exception as e:
                        logger.warning(f"Failed to process with {model_type}: {str(e)}")
                        continue
            
            if not model_outputs:
                raise RuntimeError("No models successfully processed the image")
            
            # 앙상블 적용
            ensemble_output = self.ensemble_methods[self.ensemble_method](
                model_outputs, x, **kwargs
            )
            
            logger.info(f"Successfully processed image with hybrid ensemble")
            return ensemble_output
            
        except Exception as e:
            logger.error(f"Error in hybrid ensemble processing: {str(e)}")
            raise RuntimeError(f"Failed to process image with hybrid ensemble: {str(e)}")
    
    def _weighted_average_ensemble(self, model_outputs: Dict[str, torch.Tensor], 
                                  original_input: Union[np.ndarray, Image.Image, torch.Tensor], 
                                  **kwargs) -> np.ndarray:
        """
        가중 평균 앙상블을 적용합니다.
        
        Args:
            model_outputs: 모델별 출력
            original_input: 원본 입력
            **kwargs: 추가 파라미터
            
        Returns:
            앙상블된 출력
        """
        # 가중치 정규화
        total_weight = sum(self.model_weights[model_type] for model_type in model_outputs.keys())
        normalized_weights = {k: v / total_weight for k, v in self.model_weights.items()}
        
        # 가중 평균 계산
        ensemble_output = None
        for model_type, output in model_outputs.items():
            weight = normalized_weights[model_type]
            if ensemble_output is None:
                ensemble_output = output * weight
            else:
                ensemble_output += output * weight
        
        # numpy 배열로 변환
        if isinstance(ensemble_output, torch.Tensor):
            ensemble_output = ensemble_output.detach().cpu().numpy()
        
        return ensemble_output
    
    def _voting_ensemble(self, model_outputs: Dict[str, torch.Tensor], 
                         original_input: Union[np.ndarray, Image.Image, torch.Tensor], 
                         **kwargs) -> np.ndarray:
        """
        투표 기반 앙상블을 적용합니다.
        
        Args:
            model_outputs: 모델별 출력
            original_input: 원본 입력
            **kwargs: 추가 파라미터
            
        Returns:
            앙상블된 출력
        """
        # 각 모델의 출력을 이진화 (임계값 기반)
        threshold = kwargs.get('threshold', 0.5)
        binary_outputs = {}
        
        for model_type, output in model_outputs.items():
            if isinstance(output, torch.Tensor):
                binary_output = (output > threshold).float()
                binary_outputs[model_type] = binary_output
        
        # 투표 결과 계산
        vote_sum = sum(binary_outputs.values())
        final_vote = (vote_sum > len(binary_outputs) / 2).float()
        
        # numpy 배열로 변환
        if isinstance(final_vote, torch.Tensor):
            final_vote = final_vote.detach().cpu().numpy()
        
        return final_vote
    
    def _stacking_ensemble(self, model_outputs: Dict[str, torch.Tensor], 
                           original_input: Union[np.ndarray, Image.Image, torch.Tensor], 
                           **kwargs) -> np.ndarray:
        """
        스태킹 앙상블을 적용합니다.
        
        Args:
            model_outputs: 모델별 출력
            original_input: 원본 입력
            **kwargs: 추가 파라미터
            
        Returns:
            앙상블된 출력
        """
        # 메타 특성 생성
        meta_features = []
        for model_type in self.ensemble_models:
            if model_type in model_outputs:
                output = model_outputs[model_type]
                # 특성 추출 (간단한 통계)
                if isinstance(output, torch.Tensor):
                    mean_val = torch.mean(output).item()
                    std_val = torch.std(output).item()
                    meta_features.extend([mean_val, std_val])
                else:
                    meta_features.extend([0.0, 0.0])
            else:
                meta_features.extend([0.0, 0.0])
        
        # 메타 모델로 최종 예측
        meta_input = torch.tensor(meta_features, dtype=torch.float32).unsqueeze(0)
        meta_output = self.meta_model(meta_input)
        
        # 가중 평균으로 최종 결과 생성
        final_output = self._weighted_average_ensemble(model_outputs, original_input, **kwargs)
        
        # 메타 모델의 가중치를 적용
        meta_weight = meta_output.item()
        final_output = final_output * meta_weight + final_output * (1 - meta_weight)
        
        return final_output
    
    def _adaptive_ensemble(self, model_outputs: Dict[str, torch.Tensor], 
                          original_input: Union[np.ndarray, Image.Image, torch.Tensor], 
                          **kwargs) -> np.ndarray:
        """
        적응형 앙상블을 적용합니다.
        
        Args:
            model_outputs: 모델별 출력
            original_input: 원본 입력
            **kwargs: 추가 파라미터
            
        Returns:
            앙상블된 출력
        """
        # 입력 이미지의 특성에 따라 가중치 조정
        if isinstance(original_input, torch.Tensor):
            input_tensor = original_input
        elif isinstance(original_input, np.ndarray):
            input_tensor = torch.from_numpy(original_input)
        else:
            input_tensor = torch.from_numpy(np.array(original_input))
        
        # 이미지 품질 평가 (간단한 방법)
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.squeeze(0)
        
        # 밝기와 대비 계산
        mean_brightness = torch.mean(input_tensor).item()
        std_contrast = torch.std(input_tensor).item()
        
        # 품질에 따른 가중치 조정
        quality_weights = {}
        for model_type in model_outputs.keys():
            base_weight = self.model_weights[model_type]
            
            if model_type == 'swinir':
                # SwinIR은 저해상도 이미지에 효과적
                if mean_brightness < 0.5:
                    quality_weights[model_type] = base_weight * 1.2
                else:
                    quality_weights[model_type] = base_weight
                    
            elif model_type == 'realesrgan':
                # Real-ESRGAN은 노이즈가 있는 이미지에 효과적
                if std_contrast > 0.3:
                    quality_weights[model_type] = base_weight * 1.3
                else:
                    quality_weights[model_type] = base_weight
                    
            elif model_type == 'gfpgan':
                # GFPGAN은 얼굴 이미지에 효과적
                quality_weights[model_type] = base_weight * 1.1
                
            elif model_type == 'codeformer':
                # CodeFormer은 복원이 필요한 이미지에 효과적
                if mean_brightness < 0.4 or std_contrast < 0.2:
                    quality_weights[model_type] = base_weight * 1.4
                else:
                    quality_weights[model_type] = base_weight
            else:
                quality_weights[model_type] = base_weight
        
        # 가중 평균 적용
        return self._weighted_average_ensemble(model_outputs, original_input, 
                                             custom_weights=quality_weights, **kwargs)
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        모델 가중치를 업데이트합니다.
        
        Args:
            new_weights: 새로운 가중치 딕셔너리
        """
        for model_type, weight in new_weights.items():
            if model_type in self.model_weights:
                self.model_weights[model_type] = weight
        
        logger.info(f"Updated model weights: {self.model_weights}")
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """
        앙상블 정보를 반환합니다.
        
        Returns:
            앙상블 정보 딕셔너리
        """
        return {
            'ensemble_method': self.ensemble_method,
            'ensemble_models': self.ensemble_models,
            'model_weights': self.model_weights,
            'device': str(self.device)
        }
    
    def evaluate_ensemble_performance(self, test_images: List[Union[np.ndarray, Image.Image, torch.Tensor]], 
                                    ground_truth: List[Union[np.ndarray, Image.Image, torch.Tensor]]) -> Dict[str, float]:
        """
        앙상블 성능을 평가합니다.
        
        Args:
            test_images: 테스트 이미지 리스트
            ground_truth: 정답 이미지 리스트
            
        Returns:
            성능 메트릭 딕셔너리
        """
        if len(test_images) != len(ground_truth):
            raise ValueError("Number of test images and ground truth must match")
        
        # 간단한 성능 메트릭 계산
        psnr_scores = []
        ssim_scores = []
        
        for test_img, gt_img in zip(test_images, ground_truth):
            # PSNR 계산
            psnr = self._calculate_psnr(test_img, gt_img)
            psnr_scores.append(psnr)
            
            # SSIM 계산 (간단한 버전)
            ssim = self._calculate_ssim(test_img, gt_img)
            ssim_scores.append(ssim)
        
        return {
            'psnr_mean': np.mean(psnr_scores),
            'psnr_std': np.std(psnr_scores),
            'ssim_mean': np.mean(ssim_scores),
            'ssim_std': np.std(ssim_scores)
        }
    
    def _calculate_psnr(self, img1: Union[np.ndarray, Image.Image, torch.Tensor], 
                        img2: Union[np.ndarray, Image.Image, torch.Tensor]) -> float:
        """PSNR을 계산합니다."""
        # 이미지를 numpy 배열로 변환
        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
        elif isinstance(img1, Image.Image):
            img1 = np.array(img1)
        
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()
        elif isinstance(img2, Image.Image):
            img2 = np.array(img2)
        
        # 정규화
        img1 = img1.astype(np.float64) / 255.0
        img2 = img2.astype(np.float64) / 255.0
        
        # MSE 계산
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        
        # PSNR 계산
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        return psnr
    
    def _calculate_ssim(self, img1: Union[np.ndarray, Image.Image, torch.Tensor], 
                        img2: Union[np.ndarray, Image.Image, torch.Tensor]) -> float:
        """SSIM을 계산합니다 (간단한 버전)"""
        # 이미지를 numpy 배열로 변환
        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
        elif isinstance(img1, Image.Image):
            img1 = np.array(img1)
        
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()
        elif isinstance(img2, Image.Image):
            img2 = np.array(img2)
        
        # 정규화
        img1 = img1.astype(np.float64) / 255.0
        img2 = img2.astype(np.float64) / 255.0
        
        # 간단한 SSIM 계산 (전체 이미지에 대해)
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.std(img1)
        sigma2 = np.std(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 ** 2 + sigma2 ** 2 + c2))
        
        return ssim
