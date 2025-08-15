"""
Post Processing Ensemble

후처리 모델들의 전용 앙상블 클래스입니다.
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


class PostProcessingEnsemble(nn.Module):
    """
    후처리 모델들의 전용 앙상블 클래스
    
    후처리 작업에 특화된 앙상블 방법들을 제공합니다.
    """
    
    def __init__(self, model_loader, ensemble_strategy='quality_aware'):
        """
        Args:
            model_loader: 모델 로더 인스턴스
            ensemble_strategy: 앙상블 전략 ('quality_aware', 'task_specific', 'adaptive')
        """
        super().__init__()
        
        self.model_loader = model_loader
        self.ensemble_strategy = ensemble_strategy
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 후처리 모델들의 특성
        self.model_characteristics = {
            'swinir': {
                'strength': 'super_resolution',
                'best_for': ['low_resolution', 'blurry_images'],
                'weakness': ['noise_amplification', 'artifacts']
            },
            'realesrgan': {
                'strength': 'real_world_enhancement',
                'best_for': ['noisy_images', 'compressed_images'],
                'weakness': ['over_smoothing', 'texture_loss']
            },
            'gfpgan': {
                'strength': 'face_restoration',
                'best_for': ['face_images', 'portrait_photos'],
                'weakness': ['background_artifacts', 'non_face_content']
            },
            'codeformer': {
                'strength': 'robust_restoration',
                'best_for': ['degraded_images', 'mixed_content'],
                'weakness': ['computational_cost', 'memory_usage']
            }
        }
        
        # 앙상블 전략별 함수 매핑
        self.ensemble_strategies = {
            'quality_aware': self._quality_aware_ensemble,
            'task_specific': self._task_specific_ensemble,
            'adaptive': self._adaptive_ensemble,
            'progressive': self._progressive_ensemble
        }
        
        # 품질 평가 가중치
        self.quality_weights = {
            'sharpness': 0.3,
            'noise_level': 0.2,
            'color_fidelity': 0.2,
            'structural_integrity': 0.3
        }
        
        logger.info(f"PostProcessingEnsemble initialized with strategy: {ensemble_strategy}")
    
    def forward(self, x: Union[np.ndarray, Image.Image, torch.Tensor], 
                task_type: str = 'general', **kwargs) -> np.ndarray:
        """
        입력 이미지를 후처리 앙상블로 처리합니다.
        
        Args:
            x: 입력 이미지
            task_type: 작업 타입 ('general', 'face', 'landscape', 'document')
            **kwargs: 추가 파라미터
            
        Returns:
            앙상블 처리된 이미지
        """
        try:
            logger.info(f"Processing image with post processing ensemble: {task_type}")
            
            # 앙상블 전략 적용
            ensemble_output = self.ensemble_strategies[self.ensemble_strategy](
                x, task_type, **kwargs
            )
            
            logger.info(f"Successfully processed image with post processing ensemble")
            return ensemble_output
            
        except Exception as e:
            logger.error(f"Error in post processing ensemble: {str(e)}")
            raise RuntimeError(f"Failed to process image with post processing ensemble: {str(e)}")
    
    def _quality_aware_ensemble(self, x: Union[np.ndarray, Image.Image, torch.Tensor], 
                               task_type: str, **kwargs) -> np.ndarray:
        """
        품질 인식 앙상블을 적용합니다.
        
        Args:
            x: 입력 이미지
            task_type: 작업 타입
            **kwargs: 추가 파라미터
            
        Returns:
            앙상블된 출력
        """
        # 입력 이미지 품질 평가
        quality_scores = self._assess_image_quality(x)
        
        # 품질에 따른 모델 선택 및 가중치 조정
        model_weights = self._calculate_quality_based_weights(quality_scores, task_type)
        
        # 각 모델로 처리
        model_outputs = {}
        for model_type, weight in model_weights.items():
            if weight > 0:
                try:
                    model = self.model_loader.load_model(model_type)
                    with torch.no_grad():
                        output = model(x)
                    model_outputs[model_type] = output
                except Exception as e:
                    logger.warning(f"Failed to process with {model_type}: {str(e)}")
                    continue
        
        # 가중 평균으로 앙상블
        return self._apply_weighted_ensemble(model_outputs, model_weights)
    
    def _task_specific_ensemble(self, x: Union[np.ndarray, Image.Image, torch.Tensor], 
                               task_type: str, **kwargs) -> np.ndarray:
        """
        작업별 특화 앙상블을 적용합니다.
        
        Args:
            x: 입력 이미지
            task_type: 작업 타입
            **kwargs: 추가 파라미터
            
        Returns:
            앙상블된 출력
        """
        # 작업별 모델 우선순위 설정
        task_priorities = {
            'face': ['gfpgan', 'codeformer', 'swinir', 'realesrgan'],
            'landscape': ['swinir', 'realesrgan', 'codeformer', 'gfpgan'],
            'document': ['realesrgan', 'swinir', 'codeformer', 'gfpgan'],
            'general': ['codeformer', 'swinir', 'realesrgan', 'gfpgan']
        }
        
        priority_models = task_priorities.get(task_type, task_priorities['general'])
        
        # 우선순위에 따른 순차적 처리
        final_output = None
        for model_type in priority_models:
            try:
                model = self.model_loader.load_model(model_type)
                with torch.no_grad():
                    output = model(x)
                
                if final_output is None:
                    final_output = output
                else:
                    # 점진적 개선
                    final_output = self._progressive_improvement(final_output, output, model_type)
                
                logger.info(f"Applied {model_type} for {task_type} task")
                
            except Exception as e:
                logger.warning(f"Failed to apply {model_type}: {str(e)}")
                continue
        
        if final_output is None:
            raise RuntimeError("No models successfully processed the image")
        
        # numpy 배열로 변환
        if isinstance(final_output, torch.Tensor):
            final_output = final_output.detach().cpu().numpy()
        
        return final_output
    
    def _adaptive_ensemble(self, x: Union[np.ndarray, Image.Image, torch.Tensor], 
                          task_type: str, **kwargs) -> np.ndarray:
        """
        적응형 앙상블을 적용합니다.
        
        Args:
            x: 입력 이미지
            task_type: 작업 타입
            **kwargs: 추가 파라미터
            
        Returns:
            앙상블된 출력
        """
        # 이미지 특성 분석
        image_features = self._extract_image_features(x)
        
        # 특성에 따른 모델 선택
        selected_models = self._select_models_by_features(image_features, task_type)
        
        # 선택된 모델들로 처리
        model_outputs = {}
        for model_type in selected_models:
            try:
                model = self.model_loader.load_model(model_type)
                with torch.no_grad():
                    output = model(x)
                model_outputs[model_type] = output
            except Exception as e:
                logger.warning(f"Failed to process with {model_type}: {str(e)}")
                continue
        
        # 적응형 가중치 계산
        adaptive_weights = self._calculate_adaptive_weights(image_features, selected_models)
        
        # 가중 평균으로 앙상블
        return self._apply_weighted_ensemble(model_outputs, adaptive_weights)
    
    def _progressive_ensemble(self, x: Union[np.ndarray, Image.Image, torch.Tensor], 
                             task_type: str, **kwargs) -> np.ndarray:
        """
        점진적 앙상블을 적용합니다.
        
        Args:
            x: 입력 이미지
            task_type: 작업 타입
            **kwargs: 추가 파라미터
            
        Returns:
            앙상블된 출력
        """
        # 점진적 처리 순서
        progressive_order = ['swinir', 'realesrgan', 'gfpgan', 'codeformer']
        
        current_output = x
        for model_type in progressive_order:
            try:
                model = self.model_loader.load_model(model_type)
                with torch.no_grad():
                    output = model(current_output)
                
                # 점진적 개선 적용
                current_output = self._progressive_improvement(current_output, output, model_type)
                logger.info(f"Applied progressive improvement with {model_type}")
                
            except Exception as e:
                logger.warning(f"Failed to apply {model_type} in progressive ensemble: {str(e)}")
                continue
        
        # numpy 배열로 변환
        if isinstance(current_output, torch.Tensor):
            current_output = current_output.detach().cpu().numpy()
        
        return current_output
    
    def _assess_image_quality(self, x: Union[np.ndarray, Image.Image, torch.Tensor]) -> Dict[str, float]:
        """이미지 품질을 평가합니다."""
        if isinstance(x, torch.Tensor):
            img_tensor = x
        elif isinstance(x, np.ndarray):
            img_tensor = torch.from_numpy(x)
        else:
            img_tensor = torch.from_numpy(np.array(x))
        
        if img_tensor.dim() == 4:
            img_tensor = img_tensor.squeeze(0)
        
        # 밝기 평가
        brightness = torch.mean(img_tensor).item()
        
        # 대비 평가
        contrast = torch.std(img_tensor).item()
        
        # 선명도 평가 (간단한 방법)
        if img_tensor.dim() == 3:
            gray = torch.mean(img_tensor, dim=0)
        else:
            gray = img_tensor
        
        # Laplacian 기반 선명도
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        laplacian = F.conv2d(gray.unsqueeze(0).unsqueeze(0), laplacian_kernel.unsqueeze(0).unsqueeze(0))
        sharpness = torch.var(laplacian).item()
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'sharpness': sharpness
        }
    
    def _calculate_quality_based_weights(self, quality_scores: Dict[str, float], 
                                       task_type: str) -> Dict[str, float]:
        """품질 기반 가중치를 계산합니다."""
        weights = {}
        
        for model_type in self.model_characteristics.keys():
            weight = 0.0
            
            if model_type == 'swinir':
                # SwinIR은 저해상도, 블러 이미지에 효과적
                if quality_scores['sharpness'] < 0.1:
                    weight += 0.4
                if quality_scores['contrast'] < 0.2:
                    weight += 0.3
                    
            elif model_type == 'realesrgan':
                # Real-ESRGAN은 노이즈가 있는 이미지에 효과적
                if quality_scores['contrast'] > 0.3:
                    weight += 0.4
                if quality_scores['brightness'] < 0.5:
                    weight += 0.3
                    
            elif model_type == 'gfpgan':
                # GFPGAN은 얼굴 이미지에 효과적
                if task_type == 'face':
                    weight += 0.5
                weight += 0.2  # 기본 가중치
                
            elif model_type == 'codeformer':
                # CodeFormer은 전반적으로 효과적
                weight += 0.3
                if quality_scores['brightness'] < 0.4:
                    weight += 0.2
            
            weights[model_type] = max(0.0, min(1.0, weight))
        
        # 가중치 정규화
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _extract_image_features(self, x: Union[np.ndarray, Image.Image, torch.Tensor]) -> Dict[str, float]:
        """이미지 특성을 추출합니다."""
        quality_scores = self._assess_image_quality(x)
        
        # 추가 특성
        if isinstance(x, torch.Tensor):
            img_tensor = x
        elif isinstance(x, np.ndarray):
            img_tensor = torch.from_numpy(x)
        else:
            img_tensor = torch.from_numpy(np.array(x))
        
        if img_tensor.dim() == 4:
            img_tensor = img_tensor.squeeze(0)
        
        # 색상 분포
        if img_tensor.dim() == 3:
            color_std = torch.std(img_tensor, dim=(1, 2)).mean().item()
        else:
            color_std = 0.0
        
        # 엣지 밀도
        edge_kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32)
        if img_tensor.dim() == 3:
            gray = torch.mean(img_tensor, dim=0)
        else:
            gray = img_tensor
        
        edges = F.conv2d(gray.unsqueeze(0).unsqueeze(0), edge_kernel.unsqueeze(0).unsqueeze(0))
        edge_density = torch.mean(torch.abs(edges)).item()
        
        features = quality_scores.copy()
        features.update({
            'color_variation': color_std,
            'edge_density': edge_density
        })
        
        return features
    
    def _select_models_by_features(self, features: Dict[str, float], 
                                 task_type: str) -> List[str]:
        """특성에 따라 모델을 선택합니다."""
        selected_models = []
        
        # 작업별 기본 모델
        if task_type == 'face':
            selected_models.extend(['gfpgan', 'codeformer'])
        elif task_type == 'landscape':
            selected_models.extend(['swinir', 'realesrgan'])
        elif task_type == 'document':
            selected_models.extend(['realesrgan', 'swinir'])
        else:
            selected_models.extend(['codeformer', 'swinir'])
        
        # 특성 기반 추가 모델
        if features['sharpness'] < 0.1:
            if 'swinir' not in selected_models:
                selected_models.append('swinir')
        
        if features['contrast'] > 0.3:
            if 'realesrgan' not in selected_models:
                selected_models.append('realesrgan')
        
        if features['edge_density'] > 0.1:
            if 'codeformer' not in selected_models:
                selected_models.append('codeformer')
        
        return selected_models[:3]  # 최대 3개 모델 선택
    
    def _calculate_adaptive_weights(self, features: Dict[str, float], 
                                  selected_models: List[str]) -> Dict[str, float]:
        """적응형 가중치를 계산합니다."""
        weights = {}
        
        for model_type in selected_models:
            weight = 0.0
            
            if model_type == 'swinir':
                weight += 0.3
                if features['sharpness'] < 0.1:
                    weight += 0.2
                    
            elif model_type == 'realesrgan':
                weight += 0.3
                if features['contrast'] > 0.3:
                    weight += 0.2
                    
            elif model_type == 'gfpgan':
                weight += 0.3
                if features['color_variation'] > 0.2:
                    weight += 0.2
                    
            elif model_type == 'codeformer':
                weight += 0.4  # 기본적으로 높은 가중치
                if features['edge_density'] > 0.1:
                    weight += 0.1
            
            weights[model_type] = max(0.0, min(1.0, weight))
        
        # 가중치 정규화
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _progressive_improvement(self, current: torch.Tensor, 
                               improvement: torch.Tensor, 
                               model_type: str) -> torch.Tensor:
        """점진적 개선을 적용합니다."""
        # 모델별 개선 강도
        improvement_strength = {
            'swinir': 0.7,
            'realesrgan': 0.6,
            'gfpgan': 0.5,
            'codeformer': 0.8
        }
        
        strength = improvement_strength.get(model_type, 0.5)
        
        # 점진적 개선 적용
        improved = current * (1 - strength) + improvement * strength
        
        return improved
    
    def _apply_weighted_ensemble(self, model_outputs: Dict[str, torch.Tensor], 
                                weights: Dict[str, float]) -> np.ndarray:
        """가중 평균 앙상블을 적용합니다."""
        if not model_outputs:
            raise RuntimeError("No model outputs available for ensemble")
        
        # 가중 평균 계산
        ensemble_output = None
        for model_type, output in model_outputs.items():
            weight = weights.get(model_type, 0.0)
            if weight > 0:
                if ensemble_output is None:
                    ensemble_output = output * weight
                else:
                    ensemble_output += output * weight
        
        # numpy 배열로 변환
        if isinstance(ensemble_output, torch.Tensor):
            ensemble_output = ensemble_output.detach().cpu().numpy()
        
        return ensemble_output
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """앙상블 정보를 반환합니다."""
        return {
            'ensemble_strategy': self.ensemble_strategy,
            'model_characteristics': self.model_characteristics,
            'quality_weights': self.quality_weights,
            'device': str(self.device)
        }
    
    def update_quality_weights(self, new_weights: Dict[str, float]):
        """품질 평가 가중치를 업데이트합니다."""
        for metric, weight in new_weights.items():
            if metric in self.quality_weights:
                self.quality_weights[metric] = weight
        
        logger.info(f"Updated quality weights: {self.quality_weights}")
