#!/usr/bin/env python3
"""
🔥 MyCloset AI - Cloth Segmentation Hybrid Ensemble System
==========================================================

🎯 다중 모델 앙상블을 통한 정확도 향상
✅ 8개 Cloth Segmentation 모델 통합
✅ M3 Max 최적화
✅ 메모리 효율적 처리
✅ 품질 기반 가중치 조정
"""

# PyTorch import 시도
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    # torch가 없을 때는 기본 타입 사용
    class MockNNModule:
        """Mock nn.Module (torch 없음)"""
        pass
    # nn.Module을 MockNNModule으로 대체
    class nn:
        Module = MockNNModule
    F = None

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

# 공통 imports 시스템 사용
try:
    from app.ai_pipeline.utils.common_imports import (
        np, cv2, PIL_AVAILABLE, CV2_AVAILABLE, NUMPY_AVAILABLE
    )
except ImportError:
    import numpy as np
    import cv2

# 로거 설정
logger = logging.getLogger(__name__)

@dataclass
class EnsembleConfig:
    """앙상블 설정"""
    method: str = "quality_weighted"  # voting, weighted, quality, simple_average
    quality_threshold: float = 0.7
    confidence_threshold: float = 0.5
    max_models: int = 8
    use_mps: bool = True
    memory_efficient: bool = True

class ClothSegmentationEnsembleSystem(nn.Module):
    """
    🔥 Cloth Segmentation 앙상블 시스템
    
    다중 모델의 출력을 통합하여 최종 의류 분할 결과를 생성합니다.
    """
    
    def __init__(self, config: EnsembleConfig = None):
        super().__init__()
        self.config = config or EnsembleConfig()
        self.logger = logging.getLogger(__name__)
        
        # MPS 디바이스 확인
        self.device = torch.device("mps" if torch.backends.mps.is_available() and self.config.use_mps else "cpu")
        self.logger.info(f"🎯 Cloth Segmentation 앙상블 시스템 초기화 (디바이스: {self.device})")
        
        # 앙상블 가중치 초기화
        self.ensemble_weights = nn.Parameter(torch.ones(8) / 8)  # 8개 모델
        
        # 품질 평가 메트릭
        self.quality_metrics = {
            "confidence": 0.0,
            "spatial_consistency": 0.0,
            "segmentation_quality": 0.0,
            "boundary_accuracy": 0.0
        }
        
        self.logger.info("✅ Cloth Segmentation 앙상블 시스템 초기화 완료")
    
    def forward(self, model_outputs: List[torch.Tensor], 
                confidences: List[float] = None,
                quality_scores: List[float] = None) -> torch.Tensor:
        """
        앙상블 추론 수행
    
    Args:
            model_outputs: 각 모델의 출력 (List[torch.Tensor])
            confidences: 각 모델의 신뢰도 (List[float])
            quality_scores: 각 모델의 품질 점수 (List[float])
        
    Returns:
            앙상블된 의류 분할 결과
        """
        if not model_outputs:
            raise ValueError("모델 출력이 비어있습니다.")
        
        # 입력 검증
        num_models = len(model_outputs)
        if num_models > self.config.max_models:
            self.logger.warning(f"모델 수가 최대 허용치를 초과합니다: {num_models} > {self.config.max_models}")
            model_outputs = model_outputs[:self.config.max_models]
        
        # 디바이스 이동
        model_outputs = [output.to(self.device) if isinstance(output, torch.Tensor) else torch.tensor(output, device=self.device) 
                        for output in model_outputs]
        
        # 앙상블 방법에 따른 통합
        if self.config.method == "voting":
            ensemble_output = self._voting_ensemble(model_outputs, confidences)
        elif self.config.method == "weighted":
            ensemble_output = self._weighted_ensemble(model_outputs, confidences)
        elif self.config.method == "quality":
            ensemble_output = self._quality_weighted_ensemble(model_outputs, quality_scores)
        elif self.config.method == "simple_average":
            ensemble_output = self._simple_average_ensemble(model_outputs)
        else:
            self.logger.warning(f"알 수 없는 앙상블 방법: {self.config.method}, 기본값 사용")
            ensemble_output = self._quality_weighted_ensemble(model_outputs, quality_scores)
        
        # 품질 메트릭 업데이트
        self._update_quality_metrics(ensemble_output, model_outputs)
        
        return ensemble_output
    
    def _voting_ensemble(self, model_outputs: List[torch.Tensor], 
                         confidences: List[float] = None) -> torch.Tensor:
        """투표 기반 앙상블"""
        self.logger.debug("🎯 투표 기반 앙상블 수행")
        
        # 신뢰도 기반 가중치 계산
        if confidences:
            weights = torch.tensor(confidences, device=self.device)
            weights = F.softmax(weights, dim=0)
        else:
            weights = torch.ones(len(model_outputs), device=self.device) / len(model_outputs)
        
        # 가중 평균 계산
        ensemble_output = torch.zeros_like(model_outputs[0])
        for i, output in enumerate(model_outputs):
            ensemble_output += weights[i] * output
        
        return ensemble_output
    
    def _weighted_ensemble(self, model_outputs: List[torch.Tensor], 
                          confidences: List[float] = None) -> torch.Tensor:
        """가중치 기반 앙상블"""
        self.logger.debug("🎯 가중치 기반 앙상블 수행")
        
        # 신뢰도 기반 가중치
        if confidences:
            weights = torch.tensor(confidences, device=self.device)
            weights = F.softmax(weights, dim=0)
        else:
            weights = self.ensemble_weights[:len(model_outputs)]
            weights = F.softmax(weights, dim=0)
        
        # 가중 평균 계산
        ensemble_output = torch.zeros_like(model_outputs[0])
        for i, output in enumerate(model_outputs):
            ensemble_output += weights[i] * output
        
        return ensemble_output
    
    def _quality_weighted_ensemble(self, model_outputs: List[torch.Tensor], 
                                 quality_scores: List[float] = None) -> torch.Tensor:
        """품질 기반 가중치 앙상블"""
        self.logger.debug("🎯 품질 기반 가중치 앙상블 수행")
        
        # 품질 점수 기반 가중치
        if quality_scores:
            weights = torch.tensor(quality_scores, device=self.device)
            weights = F.softmax(weights, dim=0)
        else:
            # 기본 품질 점수 (신뢰도 기반)
            weights = torch.ones(len(model_outputs), device=self.device) / len(model_outputs)
        
        # 품질 임계값 적용
        quality_mask = weights > self.config.quality_threshold
        if quality_mask.sum() > 0:
            weights = weights * quality_mask.float()
            weights = F.softmax(weights, dim=0)
        
        # 가중 평균 계산
        ensemble_output = torch.zeros_like(model_outputs[0])
        for i, output in enumerate(model_outputs):
            ensemble_output += weights[i] * output
        
        return ensemble_output
    
    def _simple_average_ensemble(self, model_outputs: List[torch.Tensor]) -> torch.Tensor:
        """단순 평균 앙상블"""
        self.logger.debug("🎯 단순 평균 앙상블 수행")
        
        # 모든 모델 출력의 평균
        ensemble_output = torch.stack(model_outputs).mean(dim=0)
        return ensemble_output
    
    def _update_quality_metrics(self, ensemble_output: torch.Tensor, 
                              model_outputs: List[torch.Tensor]):
        """품질 메트릭 업데이트"""
        # 신뢰도 계산
        if ensemble_output.dim() > 0:
            self.quality_metrics["confidence"] = float(ensemble_output.mean().item())
        
        # 공간 일관성 계산
        if len(model_outputs) > 1:
            spatial_consistency = self._calculate_spatial_consistency(model_outputs)
            self.quality_metrics["spatial_consistency"] = spatial_consistency
        
        # 분할 품질 계산
        segmentation_quality = self._calculate_segmentation_quality(ensemble_output)
        self.quality_metrics["segmentation_quality"] = segmentation_quality
        
        # 경계 정확도 계산
        boundary_accuracy = self._calculate_boundary_accuracy(ensemble_output)
        self.quality_metrics["boundary_accuracy"] = boundary_accuracy
    
    def _calculate_spatial_consistency(self, model_outputs: List[torch.Tensor]) -> float:
        """공간 일관성 계산"""
        if len(model_outputs) < 2:
            return 0.0
        
        # 각 모델 출력 간의 유사도 계산
        similarities = []
        for i in range(len(model_outputs)):
            for j in range(i + 1, len(model_outputs)):
                sim = F.cosine_similarity(
                    model_outputs[i].flatten(), 
                    model_outputs[j].flatten(), 
                    dim=0
                )
                similarities.append(sim.item())
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _calculate_segmentation_quality(self, output: torch.Tensor) -> float:
        """분할 품질 계산"""
        if output.dim() == 0:
            return 0.0
        
        # 분할 마스크의 품질 평가
        if output.dim() > 1:
            # 2D 이상인 경우 평균값 사용
            output_flat = output.mean(dim=0) if output.dim() > 1 else output
        else:
            output_flat = output
        
        # 분할 품질 (entropy 기반)
        if output_flat.numel() > 1:
            probs = F.softmax(output_flat, dim=0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            max_entropy = torch.log(torch.tensor(float(output_flat.numel())))
            quality_score = float(entropy / max_entropy)
        else:
            quality_score = 0.0
        
        return quality_score
    
    def _calculate_boundary_accuracy(self, output: torch.Tensor) -> float:
        """경계 정확도 계산"""
        if output.dim() == 0:
            return 0.0
        
        # 경계의 선명도 계산
        if output.dim() > 2:
            # 2D 이미지의 경우 경계 검출
            batch_size = output.size(0)
            boundary_scores = []
            
            for b in range(batch_size):
                # Sobel 필터를 사용한 경계 검출
                if output.size(1) > 1:  # 채널이 있는 경우
                    img = output[b].mean(dim=0)  # 채널 평균
                else:
                    img = output[b]
                
                # 경계 강도 계산
                if img.dim() == 2:
                    # 2D 텐서를 numpy로 변환
                    img_np = img.detach().cpu().numpy()
                    
                    # Sobel 필터 적용
                    sobel_x = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=3)
                    sobel_y = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=3)
                    
                    # 경계 강도
                    boundary_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
                    boundary_score = float(np.mean(boundary_magnitude))
                    boundary_scores.append(boundary_score)
            
            return float(np.mean(boundary_scores)) if boundary_scores else 0.0
        else:
            return 0.0
    
    def get_quality_metrics(self) -> Dict[str, float]:
        """품질 메트릭 반환"""
        return self.quality_metrics.copy()
    
    def update_ensemble_weights(self, new_weights: torch.Tensor):
        """앙상블 가중치 업데이트"""
        if new_weights.shape == self.ensemble_weights.shape:
            with torch.no_grad():
                self.ensemble_weights.copy_(new_weights)
            self.logger.info("✅ 앙상블 가중치 업데이트 완료")
        else:
            self.logger.warning(f"가중치 차원 불일치: {new_weights.shape} vs {self.ensemble_weights.shape}")
    
    def get_ensemble_info(self) -> Dict[str, Union[str, int, float]]:
        """앙상블 시스템 정보 반환"""
        return {
            "method": self.config.method,
            "num_models": self.config.max_models,
            "device": str(self.device),
            "quality_threshold": self.config.quality_threshold,
            "confidence_threshold": self.config.confidence_threshold,
            "memory_efficient": self.config.memory_efficient,
            "current_weights": self.ensemble_weights.detach().cpu().numpy().tolist()
        }

# 앙상블 시스템 인스턴스 생성 함수
def create_cloth_segmentation_ensemble(config: EnsembleConfig = None) -> ClothSegmentationEnsembleSystem:
    """Cloth Segmentation 앙상블 시스템 생성"""
    return ClothSegmentationEnsembleSystem(config)

# 기본 설정으로 앙상블 시스템 생성
def create_default_ensemble() -> ClothSegmentationEnsembleSystem:
    """기본 설정으로 앙상블 시스템 생성"""
    config = EnsembleConfig(
        method="quality_weighted",
        quality_threshold=0.7,
        confidence_threshold=0.5,
        max_models=8,
        use_mps=True,
        memory_efficient=True
    )
    return ClothSegmentationEnsembleSystem(config)

# 동기 앙상블 실행 함수 추가 (import 호환성을 위해)
def _run_hybrid_ensemble_sync(model_outputs: List[torch.Tensor], 
                              confidences: List[float] = None,
                              quality_scores: List[float] = None,
                              config: EnsembleConfig = None) -> torch.Tensor:
    """
    동기 앙상블 실행 함수 (import 호환성)
    
    Args:
        model_outputs: 각 모델의 출력 (List[torch.Tensor])
        confidences: 각 모델의 신뢰도 (List[float])
        quality_scores: 각 모델의 품질 점수 (List[float])
        config: 앙상블 설정
    
    Returns:
        앙상블된 의류 분할 결과
    """
    try:
        # 기본 설정 사용
        if config is None:
            config = EnsembleConfig()
        
        # 앙상블 시스템 생성
        ensemble = ClothSegmentationEnsembleSystem(config)
        
        # 앙상블 실행
        result = ensemble(model_outputs, confidences, quality_scores)
        
        return result
        
    except Exception as e:
        logger.error(f"동기 앙상블 실행 실패: {e}")
        # 오류 시 첫 번째 출력 반환
        if model_outputs:
            return model_outputs[0]
        else:
            # 빈 출력 생성
            if TORCH_AVAILABLE:
                return torch.zeros(1, 1, 256, 256)
            else:
                return None

# _combine_ensemble_results 함수 추가 (import 호환성을 위해)
def _combine_ensemble_results(model_outputs: List[torch.Tensor], 
                             confidences: List[float] = None,
                             quality_scores: List[float] = None,
                             method: str = "quality_weighted") -> torch.Tensor:
    """
    앙상블 결과 결합 함수 (import 호환성)
    
    Args:
        model_outputs: 각 모델의 출력 (List[torch.Tensor])
        confidences: 각 모델의 신뢰도 (List[float])
        quality_scores: 각 모델의 품질 점수 (List[float])
        method: 앙상블 방법
    
    Returns:
        결합된 결과
    """
    try:
        if not model_outputs:
            raise ValueError("모델 출력이 비어있습니다")
        
        # 기본값 설정
        if confidences is None:
            confidences = [1.0] * len(model_outputs)
        if quality_scores is None:
            quality_scores = [1.0] * len(model_outputs)
        
        # 모든 출력을 동일한 형태로 변환
        outputs = []
        for output in model_outputs:
            if TORCH_AVAILABLE and isinstance(output, torch.Tensor):
                outputs.append(output.detach().cpu().numpy())
            else:
                outputs.append(np.array(output))
        
        # 앙상블 방법에 따른 결합
        if method == "simple_average":
            # 단순 평균
            combined = np.mean(outputs, axis=0)
        elif method == "weighted_average":
            # 가중 평균 (신뢰도 기반)
            weights = np.array(confidences)
            weights = weights / np.sum(weights)  # 정규화
            combined = np.average(outputs, axis=0, weights=weights)
        elif method == "quality_weighted":
            # 품질 기반 가중 평균
            weights = np.array(quality_scores)
            weights = weights / np.sum(weights)  # 정규화
            combined = np.average(outputs, axis=0, weights=weights)
        else:
            # 기본값: 단순 평균
            combined = np.mean(outputs, axis=0)
        
        # torch 텐서로 변환 (가능한 경우)
        if TORCH_AVAILABLE:
            return torch.from_numpy(combined).float()
        else:
            return combined
            
    except Exception as e:
        logger.error(f"앙상블 결과 결합 실패: {e}")
        # 오류 시 첫 번째 출력 반환
        if model_outputs:
            if TORCH_AVAILABLE and isinstance(model_outputs[0], torch.Tensor):
                return model_outputs[0]
            else:
                return np.array(model_outputs[0])
        else:
            # 빈 출력 생성
            if TORCH_AVAILABLE:
                return torch.zeros(1, 1, 256, 256)
            else:
                return np.zeros((1, 1, 256, 256))

# _calculate_adaptive_threshold 함수 추가 (import 호환성을 위해)
def _calculate_adaptive_threshold(confidences: List[float], 
                                quality_scores: List[float] = None,
                                base_threshold: float = 0.5) -> float:
    """
    적응형 임계값 계산 함수 (import 호환성)
    
    Args:
        confidences: 각 모델의 신뢰도 리스트
        quality_scores: 각 모델의 품질 점수 리스트
        base_threshold: 기본 임계값
    
    Returns:
        계산된 적응형 임계값
    """
    try:
        if not confidences:
            return base_threshold
        
        # 신뢰도 기반 임계값
        confidence_threshold = np.mean(confidences) * 0.8
        
        # 품질 점수 기반 임계값 (있는 경우)
        if quality_scores and len(quality_scores) == len(confidences):
            quality_threshold = np.mean(quality_scores) * 0.7
            # 신뢰도와 품질 점수의 가중 평균
            adaptive_threshold = 0.6 * confidence_threshold + 0.4 * quality_threshold
        else:
            adaptive_threshold = confidence_threshold
        
        # 기본 임계값과 비교하여 적절한 범위 내로 제한
        final_threshold = np.clip(adaptive_threshold, base_threshold * 0.5, base_threshold * 1.5)
        
        return float(final_threshold)
        
    except Exception as e:
        logger.warning(f"적응형 임계값 계산 실패: {e}, 기본값 사용")
        return base_threshold

# _apply_ensemble_postprocessing 함수 추가 (import 호환성을 위해)
def _apply_ensemble_postprocessing(ensemble_output: torch.Tensor,
                                 individual_outputs: List[torch.Tensor],
                                 confidences: List[float] = None,
                                 method: str = "quality_weighted") -> torch.Tensor:
    """
    앙상블 후처리 적용 함수 (import 호환성)
    
    Args:
        ensemble_output: 앙상블 결과
        individual_outputs: 개별 모델 출력
        confidences: 각 모델의 신뢰도
        method: 후처리 방법
    
    Returns:
        후처리된 결과
    """
    try:
        if not individual_outputs:
            return ensemble_output
        
        # 기본값 설정
        if confidences is None:
            confidences = [1.0] * len(individual_outputs)
        
        # 후처리 방법에 따른 처리
        if method == "confidence_weighted":
            # 신뢰도 기반 가중 평균
            weights = np.array(confidences)
            weights = weights / np.sum(weights)  # 정규화
            
            # 모든 출력을 동일한 형태로 변환
            outputs = []
            for output in individual_outputs:
                if TORCH_AVAILABLE and isinstance(output, torch.Tensor):
                    outputs.append(output.detach().cpu().numpy())
                else:
                    outputs.append(np.array(output))
            
            # 가중 평균 계산
            weighted_output = np.average(outputs, axis=0, weights=weights)
            
            # torch 텐서로 변환 (가능한 경우)
            if TORCH_AVAILABLE:
                return torch.from_numpy(weighted_output).float()
            else:
                return weighted_output
                
        elif method == "quality_enhancement":
            # 품질 향상 후처리
            if TORCH_AVAILABLE and isinstance(ensemble_output, torch.Tensor):
                # 간단한 품질 향상: 노이즈 제거
                enhanced = ensemble_output.clone()
                # 임계값 기반 필터링
                threshold = 0.1
                enhanced[enhanced < threshold] = 0
                return enhanced
            else:
                # numpy 배열인 경우
                enhanced = np.array(ensemble_output).copy()
                threshold = 0.1
                enhanced[enhanced < threshold] = 0
                return enhanced
                
        else:
            # 기본값: 원본 앙상블 출력 반환
            return ensemble_output
            
    except Exception as e:
        logger.error(f"앙상블 후처리 적용 실패: {e}")
        # 오류 시 원본 출력 반환
        return ensemble_output

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # 앙상블 시스템 생성
    ensemble = create_default_ensemble()
    
    # 테스트 데이터 생성
    batch_size, channels, height, width = 2, 1, 256, 256
    test_outputs = [
        torch.randn(batch_size, channels, height, width) for _ in range(4)
    ]
    
    # 앙상블 수행
    result = ensemble(test_outputs)
    print(f"앙상블 결과 형태: {result.shape}")
    print(f"품질 메트릭: {ensemble.get_quality_metrics()}")
    print(f"앙상블 정보: {ensemble.get_ensemble_info()}")
