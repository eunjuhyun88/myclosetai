#!/usr/bin/env python3
"""
🔥 MyCloset AI - Quality Assessment Hybrid Ensemble System
==========================================================

🎯 다중 모델 앙상블을 통한 정확도 향상
✅ 8개 Quality Assessment 모델 통합
✅ M3 Max 최적화
✅ 메모리 효율적 처리
✅ 품질 기반 가중치 조정
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

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

class QualityAssessmentEnsembleSystem(nn.Module):
    """
    🔥 Quality Assessment 앙상블 시스템
    
    다중 모델의 출력을 통합하여 최종 품질 평가 결과를 생성합니다.
    """
    
    def __init__(self, config: EnsembleConfig = None):
        super().__init__()
        self.config = config or EnsembleConfig()
        self.logger = logging.getLogger(__name__)
        
        # MPS 디바이스 확인
        self.device = torch.device("mps" if torch.backends.mps.is_available() and self.config.use_mps else "cpu")
        self.logger.info(f"🎯 Quality Assessment 앙상블 시스템 초기화 (디바이스: {self.device})")
        
        # 앙상블 가중치 초기화
        self.ensemble_weights = nn.Parameter(torch.ones(8) / 8)  # 8개 모델
        
        # 품질 평가 메트릭
        self.quality_metrics = {
            "confidence": 0.0,
            "spatial_consistency": 0.0,
            "assessment_accuracy": 0.0,
            "evaluation_consistency": 0.0
        }
        
        self.logger.info("✅ Quality Assessment 앙상블 시스템 초기화 완료")
    
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
            앙상블된 품질 평가 결과
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
        
        # 평가 정확도 계산
        assessment_accuracy = self._calculate_assessment_accuracy(ensemble_output)
        self.quality_metrics["assessment_accuracy"] = assessment_accuracy
        
        # 평가 일관성 계산
        evaluation_consistency = self._calculate_evaluation_consistency(ensemble_output)
        self.quality_metrics["evaluation_consistency"] = evaluation_consistency
    
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
    
    def _calculate_assessment_accuracy(self, output: torch.Tensor) -> float:
        """평가 정확도 계산"""
        if output.dim() == 0:
            return 0.0
        
        # 품질 평가 결과의 정확도 평가
        if output.dim() > 1:
            # 2D 이상인 경우 평균값 사용
            output_flat = output.mean(dim=0) if output.dim() > 1 else output
        else:
            output_flat = output
        
        # 평가 정확도 (entropy 기반)
        if output_flat.numel() > 1:
            probs = F.softmax(output_flat, dim=0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            max_entropy = torch.log(torch.tensor(float(output_flat.numel())))
            accuracy_score = float(entropy / max_entropy)
        else:
            accuracy_score = 0.0
        
        return accuracy_score
    
    def _calculate_evaluation_consistency(self, output: torch.Tensor) -> float:
        """평가 일관성 계산"""
        if output.dim() == 0:
            return 0.0
        
        # 품질 평가 결과의 일관성 평가
        if output.dim() > 2:
            # 2D 이상인 경우 평가 일관성 계산
            batch_size = output.size(0)
            consistency_scores = []
            
            for b in range(batch_size):
                # 평가 일관성 계산
                if output.size(1) > 1:  # 채널이 있는 경우
                    assessment_result = output[b].mean(dim=0)  # 채널 평균
                else:
                    assessment_result = output[b]
                
                # 평가 일관성 점수 계산
                if assessment_result.dim() == 2:
                    # 2D 텐서를 numpy로 변환
                    result_np = assessment_result.detach().cpu().numpy()
                    
                    # 평가 일관성 계산 (예: 평가 점수의 분포 균등성)
                    # 1. 평가 점수의 분포 균등성
                    score_hist = np.histogram(result_np, bins=20)[0]
                    score_entropy = -np.sum(score_hist * np.log(score_hist + 1e-8))
                    max_score_entropy = np.log(20)
                    distribution_score = score_entropy / max_score_entropy
                    
                    # 2. 평가 점수의 표준편차 (적당한 분산이 좋음)
                    score_std = np.std(result_np)
                    variance_score = 1.0 / (1.0 + score_std)
                    
                    # 종합 일관성 점수
                    consistency_score = (distribution_score * 0.6 + variance_score * 0.4)
                    consistency_scores.append(consistency_score)
            
            return float(np.mean(consistency_scores)) if consistency_scores else 0.0
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
def create_quality_assessment_ensemble(config: EnsembleConfig = None) -> QualityAssessmentEnsembleSystem:
    """Quality Assessment 앙상블 시스템 생성"""
    return QualityAssessmentEnsembleSystem(config)

# 기본 설정으로 앙상블 시스템 생성
def create_default_ensemble() -> QualityAssessmentEnsembleSystem:
    """기본 설정으로 앙상블 시스템 생성"""
    config = EnsembleConfig(
        method="quality_weighted",
        quality_threshold=0.7,
        confidence_threshold=0.5,
        max_models=8,
        use_mps=True,
        memory_efficient=True
    )
    return QualityAssessmentEnsembleSystem(config)

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # 앙상블 시스템 생성
    ensemble = create_default_ensemble()
    
    # 테스트 데이터 생성
    batch_size, num_metrics = 2, 10  # 10개 품질 메트릭
    test_outputs = [
        torch.randn(batch_size, num_metrics) for _ in range(4)
    ]
    
    # 앙상블 수행
    result = ensemble(test_outputs)
    print(f"앙상블 결과 형태: {result.shape}")
    print(f"품질 메트릭: {ensemble.get_quality_metrics()}")
    print(f"앙상블 정보: {ensemble.get_ensemble_info()}")
