"""
Adaptive Ensemble
이미지 특성에 따라 동적으로 모델을 선택하고 결합하는 앙상블 클래스
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from PIL import Image
import logging
import time
from dataclasses import dataclass

# 프로젝트 로깅 설정 import
from backend.app.ai_pipeline.utils.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class EnsembleResult:
    """앙상블 결과를 저장하는 데이터 클래스"""
    final_output: torch.Tensor
    selected_models: List[str]
    model_weights: List[float]
    confidence_scores: List[float]
    processing_time: float
    ensemble_method: str

class AdaptiveEnsemble:
    """
    이미지 특성에 따라 동적으로 모델을 선택하고 결합하는 앙상블 클래스
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Args:
            device: 사용할 디바이스
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 앙상블 설정
        self.ensemble_config = {
            'selection_strategy': 'quality_based',  # 'quality_based', 'content_based', 'hybrid'
            'weight_calculation': 'adaptive',  # 'adaptive', 'fixed', 'performance_based'
            'confidence_threshold': 0.7,
            'max_models': 3,
            'enable_dynamic_selection': True,
            'fallback_strategy': 'best_single'
        }
        
        # 모델 성능 히스토리
        self.model_performance_history = {}
        
        # 이미지 특성 분석기
        self.feature_analyzers = {
            'noise_level': self._analyze_noise_level,
            'blur_level': self._analyze_blur_level,
            'texture_complexity': self._analyze_texture_complexity,
            'color_distribution': self._analyze_color_distribution,
            'edge_density': self._analyze_edge_density
        }
        
        logger.info(f"AdaptiveEnsemble initialized on device: {self.device}")
    
    def ensemble_models(self, input_image: torch.Tensor, 
                       model_outputs: Dict[str, torch.Tensor],
                       model_configs: Dict[str, Dict[str, Any]]) -> EnsembleResult:
        """
        모델 앙상블 실행
        
        Args:
            input_image: 입력 이미지
            model_outputs: 모델별 출력 결과
            model_configs: 모델별 설정
            
        Returns:
            앙상블 결과
        """
        start_time = time.time()
        
        try:
            logger.info("적응형 앙상블 시작")
            
            # 이미지 특성 분석
            image_features = self._analyze_image_features(input_image)
            
            # 모델 선택
            selected_models = self._select_models(image_features, model_configs)
            
            # 가중치 계산
            model_weights = self._calculate_weights(selected_models, image_features, model_configs)
            
            # 앙상블 실행
            final_output = self._execute_ensemble(model_outputs, selected_models, model_weights)
            
            # 신뢰도 점수 계산
            confidence_scores = self._calculate_confidence_scores(selected_models, model_weights)
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            
            # 결과 생성
            result = EnsembleResult(
                final_output=final_output,
                selected_models=selected_models,
                model_weights=model_weights,
                confidence_scores=confidence_scores,
                processing_time=processing_time,
                ensemble_method='adaptive'
            )
            
            logger.info(f"적응형 앙상블 완료 (선택된 모델: {selected_models})")
            return result
            
        except Exception as e:
            logger.error(f"적응형 앙상블 중 오류 발생: {e}")
            
            # 오류 시 단일 모델 결과 반환
            fallback_model = list(model_outputs.keys())[0] if model_outputs else None
            
            return EnsembleResult(
                final_output=model_outputs.get(fallback_model, input_image),
                selected_models=[fallback_model] if fallback_model else [],
                model_weights=[1.0] if fallback_model else [],
                confidence_scores=[0.0] if fallback_model else [],
                processing_time=time.time() - start_time,
                ensemble_method='fallback'
            )
    
    def _analyze_image_features(self, image: torch.Tensor) -> Dict[str, float]:
        """이미지 특성 분석"""
        try:
            features = {}
            
            for feature_name, analyzer_func in self.feature_analyzers.items():
                try:
                    features[feature_name] = analyzer_func(image)
                except Exception as e:
                    logger.warning(f"특성 {feature_name} 분석 중 오류 발생: {e}")
                    features[feature_name] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"이미지 특성 분석 중 오류 발생: {e}")
            return {}
    
    def _analyze_noise_level(self, image: torch.Tensor) -> float:
        """노이즈 레벨 분석"""
        try:
            # 가우시안 블러를 사용한 노이즈 추정
            blurred = F.avg_pool2d(image.unsqueeze(0), kernel_size=3, stride=1, padding=1)
            
            # 원본과 블러된 이미지의 차이로 노이즈 추정
            noise_map = torch.abs(image.unsqueeze(0) - blurred)
            noise_level = torch.mean(noise_map).item()
            
            # 0-1 범위로 정규화
            normalized_noise = min(1.0, noise_level * 10.0)
            
            return normalized_noise
            
        except Exception as e:
            logger.error(f"노이즈 레벨 분석 중 오류 발생: {e}")
            return 0.0
    
    def _analyze_blur_level(self, image: torch.Tensor) -> float:
        """블러 레벨 분석"""
        try:
            # 라플라시안 필터를 사용한 선명도 분석
            laplacian_kernel = torch.tensor([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            
            # 그레이스케일 변환
            if image.size(0) == 3:
                gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            else:
                gray = image[0]
            
            # 엣지 검출
            edges = F.conv2d(gray.unsqueeze(0).unsqueeze(0), laplacian_kernel, padding=1)
            edge_strength = torch.mean(torch.abs(edges)).item()
            
            # 블러 레벨 (엣지 강도가 낮을수록 블러)
            blur_level = max(0, 1.0 - edge_strength / 2.0)
            
            return blur_level
            
        except Exception as e:
            logger.error(f"블러 레벨 분석 중 오류 발생: {e}")
            return 0.0
    
    def _analyze_texture_complexity(self, image: torch.Tensor) -> float:
        """텍스처 복잡도 분석"""
        try:
            # 그레이스케일 변환
            if image.size(0) == 3:
                gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            else:
                gray = image[0]
            
            # 로컬 표준편차를 사용한 텍스처 복잡도 계산
            kernel_size = 5
            padding = kernel_size // 2
            
            # 평균 계산
            mean_kernel = torch.ones(1, 1, kernel_size, kernel_size, device=self.device) / (kernel_size * kernel_size)
            local_mean = F.conv2d(gray.unsqueeze(0).unsqueeze(0), mean_kernel, padding=padding)
            
            # 분산 계산
            local_var = F.conv2d((gray.unsqueeze(0).unsqueeze(0) - local_mean) ** 2, mean_kernel, padding=padding)
            local_std = torch.sqrt(local_var)
            
            # 텍스처 복잡도 (표준편차의 평균)
            texture_complexity = torch.mean(local_std).item()
            
            # 0-1 범위로 정규화
            normalized_complexity = min(1.0, texture_complexity * 5.0)
            
            return normalized_complexity
            
        except Exception as e:
            logger.error(f"텍스처 복잡도 분석 중 오류 발생: {e}")
            return 0.0
    
    def _analyze_color_distribution(self, image: torch.Tensor) -> float:
        """색상 분포 분석"""
        try:
            if image.size(0) != 3:  # RGB가 아닌 경우
                return 0.5
            
            # 각 채널의 표준편차 계산
            channel_stds = []
            for c in range(3):
                channel_std = torch.std(image[c]).item()
                channel_stds.append(channel_std)
            
            # 색상 분포 다양성 (표준편차의 평균)
            color_diversity = np.mean(channel_stds)
            
            # 0-1 범위로 정규화
            normalized_diversity = min(1.0, color_diversity * 2.0)
            
            return normalized_diversity
            
        except Exception as e:
            logger.error(f"색상 분포 분석 중 오류 발생: {e}")
            return 0.5
    
    def _analyze_edge_density(self, image: torch.Tensor) -> float:
        """엣지 밀도 분석"""
        try:
            # Sobel 필터를 사용한 엣지 검출
            sobel_x = torch.tensor([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            
            sobel_y = torch.tensor([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            
            # 그레이스케일 변환
            if image.size(0) == 3:
                gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            else:
                gray = image[0]
            
            # 엣지 검출
            grad_x = F.conv2d(gray.unsqueeze(0).unsqueeze(0), sobel_x, padding=1)
            grad_y = F.conv2d(gray.unsqueeze(0).unsqueeze(0), sobel_y, padding=1)
            
            # 엣지 강도
            edge_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
            
            # 엣지 밀도 (엣지 강도의 평균)
            edge_density = torch.mean(edge_magnitude).item()
            
            # 0-1 범위로 정규화
            normalized_density = min(1.0, edge_density / 2.0)
            
            return normalized_density
            
        except Exception as e:
            logger.error(f"엣지 밀도 분석 중 오류 발생: {e}")
            return 0.0
    
    def _select_models(self, image_features: Dict[str, float], 
                       model_configs: Dict[str, Dict[str, Any]]) -> List[str]:
        """모델 선택"""
        try:
            if not self.ensemble_config['enable_dynamic_selection']:
                # 동적 선택이 비활성화된 경우 모든 모델 선택
                return list(model_configs.keys())
            
            strategy = self.ensemble_config['selection_strategy']
            
            if strategy == 'quality_based':
                return self._quality_based_selection(image_features, model_configs)
            elif strategy == 'content_based':
                return self._content_based_selection(image_features, model_configs)
            elif strategy == 'hybrid':
                return self._hybrid_selection(image_features, model_configs)
            else:
                logger.warning(f"알 수 없는 선택 전략: {strategy}")
                return list(model_configs.keys())
                
        except Exception as e:
            logger.error(f"모델 선택 중 오류 발생: {e}")
            return list(model_configs.keys())
    
    def _quality_based_selection(self, image_features: Dict[str, float], 
                                model_configs: Dict[str, Dict[str, Any]]) -> List[str]:
        """품질 기반 모델 선택"""
        try:
            # 이미지 품질 점수 계산
            quality_score = self._calculate_quality_score(image_features)
            
            # 품질에 따른 모델 선택
            if quality_score > 0.8:
                # 고품질 이미지: 정밀한 모델들 선택
                selected = [name for name, config in model_configs.items() 
                           if config.get('model_type') in ['swinir', 'realesrgan']]
            elif quality_score > 0.5:
                # 중간 품질 이미지: 균형잡힌 모델들 선택
                selected = [name for name, config in model_configs.items() 
                           if config.get('model_type') in ['swinir', 'gfpgan']]
            else:
                # 저품질 이미지: 강력한 복원 모델들 선택
                selected = [name for name, config in model_configs.items() 
                           if config.get('model_type') in ['gfpgan', 'codeformer']]
            
            # 최대 모델 수 제한
            max_models = self.ensemble_config['max_models']
            if len(selected) > max_models:
                selected = selected[:max_models]
            
            return selected
            
        except Exception as e:
            logger.error(f"품질 기반 선택 중 오류 발생: {e}")
            return list(model_configs.keys())
    
    def _content_based_selection(self, image_features: Dict[str, float], 
                                model_configs: Dict[str, Dict[str, Any]]) -> List[str]:
        """내용 기반 모델 선택"""
        try:
            selected = []
            
            # 노이즈가 높은 경우: 노이즈 제거 모델 선택
            if image_features.get('noise_level', 0) > 0.6:
                noise_models = [name for name, config in model_configs.items() 
                               if config.get('model_type') in ['swinir', 'realesrgan']]
                selected.extend(noise_models)
            
            # 블러가 높은 경우: 선명도 향상 모델 선택
            if image_features.get('blur_level', 0) > 0.6:
                sharpening_models = [name for name, config in model_configs.items() 
                                   if config.get('model_type') in ['swinir', 'realesrgan']]
                selected.extend(sharpening_models)
            
            # 텍스처가 복잡한 경우: 고해상도 모델 선택
            if image_features.get('texture_complexity', 0) > 0.7:
                high_res_models = [name for name, config in model_configs.items() 
                                 if config.get('model_type') in ['swinir']]
                selected.extend(high_res_models)
            
            # 엣지가 많은 경우: 엣지 보존 모델 선택
            if image_features.get('edge_density', 0) > 0.7:
                edge_models = [name for name, config in model_configs.items() 
                             if config.get('model_type') in ['swinir', 'gfpgan']]
                selected.extend(edge_models)
            
            # 중복 제거
            selected = list(set(selected))
            
            # 최대 모델 수 제한
            max_models = self.ensemble_config['max_models']
            if len(selected) > max_models:
                selected = selected[:max_models]
            
            # 선택된 모델이 없는 경우 기본 모델들 선택
            if not selected:
                selected = list(model_configs.keys())[:max_models]
            
            return selected
            
        except Exception as e:
            logger.error(f"내용 기반 선택 중 오류 발생: {e}")
            return list(model_configs.keys())
    
    def _hybrid_selection(self, image_features: Dict[str, float], 
                          model_configs: Dict[str, Dict[str, Any]]) -> List[str]:
        """하이브리드 모델 선택"""
        try:
            # 품질 기반 선택
            quality_models = self._quality_based_selection(image_features, model_configs)
            
            # 내용 기반 선택
            content_models = self._content_based_selection(image_features, model_configs)
            
            # 두 방법의 결과를 결합
            hybrid_models = list(set(quality_models + content_models))
            
            # 최대 모델 수 제한
            max_models = self.ensemble_config['max_models']
            if len(hybrid_models) > max_models:
                # 품질 점수에 따라 정렬하여 상위 모델들 선택
                model_scores = []
                for model_name in hybrid_models:
                    score = self._calculate_model_score(model_name, image_features, model_configs)
                    model_scores.append((model_name, score))
                
                model_scores.sort(key=lambda x: x[1], reverse=True)
                hybrid_models = [name for name, _ in model_scores[:max_models]]
            
            return hybrid_models
            
        except Exception as e:
            logger.error(f"하이브리드 선택 중 오류 발생: {e}")
            return list(model_configs.keys())
    
    def _calculate_weights(self, selected_models: List[str], 
                          image_features: Dict[str, float],
                          model_configs: Dict[str, Dict[str, Any]]) -> List[float]:
        """모델 가중치 계산"""
        try:
            weight_method = self.ensemble_config['weight_calculation']
            
            if weight_method == 'fixed':
                # 균등 가중치
                weights = [1.0 / len(selected_models)] * len(selected_models)
            elif weight_method == 'performance_based':
                # 성능 기반 가중치
                weights = self._performance_based_weights(selected_models)
            elif weight_method == 'adaptive':
                # 적응형 가중치
                weights = self._adaptive_weights(selected_models, image_features, model_configs)
            else:
                logger.warning(f"알 수 없는 가중치 계산 방법: {weight_method}")
                weights = [1.0 / len(selected_models)] * len(selected_models)
            
            # 가중치 정규화
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(selected_models)] * len(selected_models)
            
            return weights
            
        except Exception as e:
            logger.error(f"가중치 계산 중 오류 발생: {e}")
            return [1.0 / len(selected_models)] * len(selected_models)
    
    def _performance_based_weights(self, selected_models: List[str]) -> List[float]:
        """성능 기반 가중치"""
        try:
            weights = []
            
            for model_name in selected_models:
                if model_name in self.model_performance_history:
                    # 평균 성능 점수 사용
                    avg_performance = np.mean(self.model_performance_history[model_name])
                    weights.append(avg_performance)
                else:
                    # 성능 기록이 없는 경우 기본 가중치
                    weights.append(1.0)
            
            return weights
            
        except Exception as e:
            logger.error(f"성능 기반 가중치 계산 중 오류 발생: {e}")
            return [1.0] * len(selected_models)
    
    def _adaptive_weights(self, selected_models: List[str], 
                         image_features: Dict[str, float],
                         model_configs: Dict[str, Dict[str, Any]]) -> List[float]:
        """적응형 가중치"""
        try:
            weights = []
            
            for model_name in selected_models:
                # 모델 점수 계산
                model_score = self._calculate_model_score(model_name, image_features, model_configs)
                weights.append(model_score)
            
            return weights
            
        except Exception as e:
            logger.error(f"적응형 가중치 계산 중 오류 발생: {e}")
            return [1.0] * len(selected_models)
    
    def _calculate_model_score(self, model_name: str, 
                              image_features: Dict[str, float],
                              model_configs: Dict[str, Dict[str, Any]]) -> float:
        """모델 점수 계산"""
        try:
            model_config = model_configs.get(model_name, {})
            model_type = model_config.get('model_type', 'unknown')
            
            # 모델 타입별 적합성 점수
            type_scores = {
                'swinir': self._calculate_swinir_score(image_features),
                'realesrgan': self._calculate_realesrgan_score(image_features),
                'gfpgan': self._calculate_gfpgan_score(image_features),
                'codeformer': self._calculate_codeformer_score(image_features)
            }
            
            base_score = type_scores.get(model_type, 0.5)
            
            # 성능 히스토리 반영
            if model_name in self.model_performance_history:
                performance_score = np.mean(self.model_performance_history[model_name])
                # 기본 점수와 성능 점수를 결합
                final_score = 0.7 * base_score + 0.3 * performance_score
            else:
                final_score = base_score
            
            return final_score
            
        except Exception as e:
            logger.error(f"모델 점수 계산 중 오류 발생: {e}")
            return 0.5
    
    def _calculate_swinir_score(self, image_features: Dict[str, float]) -> float:
        """SwinIR 모델 점수 계산"""
        try:
            # SwinIR은 고해상도, 텍스처 복잡한 이미지에 적합
            score = 0.5  # 기본 점수
            
            # 텍스처 복잡도가 높을수록 높은 점수
            if 'texture_complexity' in image_features:
                score += image_features['texture_complexity'] * 0.3
            
            # 엣지 밀도가 높을수록 높은 점수
            if 'edge_density' in image_features:
                score += image_features['edge_density'] * 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"SwinIR 점수 계산 중 오류 발생: {e}")
            return 0.5
    
    def _calculate_realesrgan_score(self, image_features: Dict[str, float]) -> float:
        """Real-ESRGAN 모델 점수 계산"""
        try:
            # Real-ESRGAN은 노이즈가 많고 블러가 있는 이미지에 적합
            score = 0.5  # 기본 점수
            
            # 노이즈 레벨이 높을수록 높은 점수
            if 'noise_level' in image_features:
                score += image_features['noise_level'] * 0.4
            
            # 블러 레벨이 높을수록 높은 점수
            if 'blur_level' in image_features:
                score += image_features['blur_level'] * 0.3
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Real-ESRGAN 점수 계산 중 오류 발생: {e}")
            return 0.5
    
    def _calculate_gfpgan_score(self, image_features: Dict[str, float]) -> float:
        """GFPGAN 모델 점수 계산"""
        try:
            # GFPGAN은 얼굴 복원에 특화
            score = 0.5  # 기본 점수
            
            # 블러 레벨이 높을수록 높은 점수
            if 'blur_level' in image_features:
                score += image_features['blur_level'] * 0.3
            
            # 노이즈 레벨이 높을수록 높은 점수
            if 'noise_level' in image_features:
                score += image_features['noise_level'] * 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"GFPGAN 점수 계산 중 오류 발생: {e}")
            return 0.5
    
    def _calculate_codeformer_score(self, image_features: Dict[str, float]) -> float:
        """CodeFormer 모델 점수 계산"""
        try:
            # CodeFormer은 강력한 복원 능력을 가짐
            score = 0.5  # 기본 점수
            
            # 전반적인 이미지 품질이 낮을수록 높은 점수
            quality_score = self._calculate_quality_score(image_features)
            score += (1.0 - quality_score) * 0.4
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"CodeFormer 점수 계산 중 오류 발생: {e}")
            return 0.5
    
    def _calculate_quality_score(self, image_features: Dict[str, float]) -> float:
        """이미지 품질 점수 계산"""
        try:
            # 각 특성의 가중 평균으로 품질 점수 계산
            feature_weights = {
                'noise_level': -0.3,  # 노이즈는 품질을 낮춤
                'blur_level': -0.3,   # 블러는 품질을 낮춤
                'texture_complexity': 0.2,  # 텍스처는 품질을 높임
                'edge_density': 0.2,  # 엣지는 품질을 높임
                'color_distribution': 0.2   # 색상 다양성은 품질을 높임
            }
            
            quality_score = 0.5  # 기본 점수
            
            for feature_name, weight in feature_weights.items():
                if feature_name in image_features:
                    quality_score += image_features[feature_name] * weight
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"품질 점수 계산 중 오류 발생: {e}")
            return 0.5
    
    def _execute_ensemble(self, model_outputs: Dict[str, torch.Tensor], 
                         selected_models: List[str],
                         model_weights: List[float]) -> torch.Tensor:
        """앙상블 실행"""
        try:
            if not selected_models:
                raise ValueError("선택된 모델이 없습니다")
            
            # 첫 번째 모델의 출력을 기준으로 초기화
            first_model = selected_models[0]
            ensemble_output = model_outputs[first_model].clone() * model_weights[0]
            
            # 나머지 모델들의 출력을 가중 평균으로 결합
            for i, model_name in enumerate(selected_models[1:], 1):
                if model_name in model_outputs:
                    ensemble_output += model_outputs[model_name] * model_weights[i]
            
            return ensemble_output
            
        except Exception as e:
            logger.error(f"앙상블 실행 중 오류 발생: {e}")
            # 오류 시 첫 번째 모델 출력 반환
            first_model = selected_models[0] if selected_models else list(model_outputs.keys())[0]
            return model_outputs.get(first_model, torch.zeros_like(list(model_outputs.values())[0]))
    
    def _calculate_confidence_scores(self, selected_models: List[str], 
                                   model_weights: List[float]) -> List[float]:
        """신뢰도 점수 계산"""
        try:
            confidence_scores = []
            
            for weight in model_weights:
                # 가중치가 높을수록 높은 신뢰도
                confidence = min(1.0, weight * 2.0)
                confidence_scores.append(confidence)
            
            return confidence_scores
            
        except Exception as e:
            logger.error(f"신뢰도 점수 계산 중 오류 발생: {e}")
            return [0.5] * len(selected_models)
    
    def update_model_performance(self, model_name: str, performance_score: float):
        """모델 성능 업데이트"""
        try:
            if model_name not in self.model_performance_history:
                self.model_performance_history[model_name] = []
            
            self.model_performance_history[model_name].append(performance_score)
            
            # 최근 100개 성능 기록만 유지
            if len(self.model_performance_history[model_name]) > 100:
                self.model_performance_history[model_name] = self.model_performance_history[model_name][-100:]
                
        except Exception as e:
            logger.error(f"모델 성능 업데이트 중 오류 발생: {e}")
    
    def get_ensemble_config(self) -> Dict[str, Any]:
        """앙상블 설정 반환"""
        return self.ensemble_config.copy()
    
    def set_ensemble_config(self, **kwargs):
        """앙상블 설정 업데이트"""
        self.ensemble_config.update(kwargs)
        logger.info("앙상블 설정 업데이트 완료")
    
    def get_model_performance_history(self) -> Dict[str, List[float]]:
        """모델 성능 히스토리 반환"""
        return self.model_performance_history.copy()
    
    def reset_performance_history(self):
        """성능 히스토리 초기화"""
        self.model_performance_history.clear()
        logger.info("성능 히스토리 초기화 완료")
