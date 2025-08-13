#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥 MyCloset AI - Pose Ensemble System
=====================================

✅ 다중 모델 앙상블 시스템
✅ 가중 평균 및 신뢰도 기반 융합
✅ 불확실성 정량화
✅ 가상 피팅 최적화

파일 위치: backend/app/ai_pipeline/steps/pose_estimation/ensemble/pose_ensemble_system.py
작성자: MyCloset AI Team
날짜: 2025-08-01
버전: v1.0
"""

import os
import sys
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

# 공통 imports
from app.ai_pipeline.utils.common_imports import (
    torch, nn, F, np, DEVICE, TORCH_AVAILABLE,
    Path, Dict, Any, Optional, Tuple, List, Union,
    dataclass, field
)

# 경고 무시
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=ImportWarning)

logger = logging.getLogger(__name__)

@dataclass
class EnsembleConfig:
    """앙상블 설정"""
    num_keypoints: int = 17
    ensemble_method: str = 'weighted_average'  # 'simple_average', 'weighted_average', 'confidence_weighted'
    confidence_threshold: float = 0.8
    enable_uncertainty_quantification: bool = True
    enable_confidence_calibration: bool = True
    ensemble_quality_threshold: float = 0.7

class PoseEnsembleSystem(nn.Module):
    """포즈 추정 앙상블 시스템"""
    
    def __init__(self, num_keypoints=17, ensemble_models=None, hidden_dim=256):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.ensemble_models = ensemble_models or []
        self.hidden_dim = hidden_dim
        
        # 앙상블 융합 네트워크
        self.fusion_network = nn.Sequential(
            nn.Linear(num_keypoints * 3 * len(ensemble_models), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_keypoints * 3)  # x, y, confidence
        )
        
        # 신뢰도 보정 네트워크
        self.confidence_calibration = nn.Sequential(
            nn.Linear(num_keypoints * len(ensemble_models), hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_keypoints),
            nn.Sigmoid()
        )
        
        # 불확실성 추정 네트워크
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(num_keypoints * len(ensemble_models), hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_keypoints),
            nn.Softplus()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, model_outputs, model_confidences=None):
        """
        앙상블 추론
        
        Args:
            model_outputs: List[Dict] - 각 모델의 출력
            model_confidences: List[float] - 각 모델의 신뢰도
            
        Returns:
            Dict - 앙상블 결과
        """
        if not model_outputs:
            return {"error": "모델 출력이 없습니다"}
        
        try:
            # 입력 데이터 준비
            keypoints_list = []
            confidences_list = []
            
            for output in model_outputs:
                if "keypoints" in output and output["keypoints"]:
                    keypoints = output["keypoints"]
                    # 키포인트를 평면화
                    flat_keypoints = []
                    for kp in keypoints:
                        if len(kp) >= 3:
                            flat_keypoints.extend([kp[0], kp[1], kp[2]])  # x, y, confidence
                        else:
                            flat_keypoints.extend([0.0, 0.0, 0.0])
                    keypoints_list.append(flat_keypoints)
                    
                    # 개별 키포인트 신뢰도
                    kp_confidences = [kp[2] if len(kp) > 2 else 0.0 for kp in keypoints]
                    confidences_list.append(kp_confidences)
            
            if not keypoints_list:
                return {"error": "유효한 키포인트가 없습니다"}
            
            # 텐서 변환
            keypoints_tensor = torch.tensor(keypoints_list, dtype=torch.float32)
            confidences_tensor = torch.tensor(confidences_list, dtype=torch.float32)
            
            # 앙상블 융합
            fused_keypoints = self._ensemble_fusion(keypoints_tensor, confidences_tensor)
            
            # 신뢰도 보정
            calibrated_confidences = self.confidence_calibration(confidences_tensor.view(-1))
            
            # 불확실성 추정
            uncertainties = self.uncertainty_estimator(confidences_tensor.view(-1))
            
            # 결과 구성
            result = {
                "keypoints": fused_keypoints,
                "confidence_scores": calibrated_confidences.tolist(),
                "uncertainties": uncertainties.tolist(),
                "ensemble_info": {
                    "num_models": len(model_outputs),
                    "fusion_method": "neural_ensemble",
                    "calibration_applied": True,
                    "uncertainty_quantified": True
                },
                "success": True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 앙상블 추론 실패: {e}")
            return {"error": str(e), "success": False}
    
    def _ensemble_fusion(self, keypoints_tensor, confidences_tensor):
        """앙상블 융합"""
        # 신경망 기반 융합
        batch_size = keypoints_tensor.size(0)
        flattened_input = keypoints_tensor.view(batch_size, -1)
        
        fused_output = self.fusion_network(flattened_input)
        fused_keypoints = fused_output.view(self.num_keypoints, 3)  # x, y, confidence
        
        return fused_keypoints.tolist()
    
    def _simple_average_fusion(self, keypoints_list):
        """단순 평균 융합"""
        if not keypoints_list:
            return []
        
        num_models = len(keypoints_list)
        num_keypoints = len(keypoints_list[0]) // 3
        
        averaged_keypoints = []
        for i in range(num_keypoints):
            x_sum = sum(keypoints_list[j][i*3] for j in range(num_models))
            y_sum = sum(keypoints_list[j][i*3+1] for j in range(num_models))
            conf_sum = sum(keypoints_list[j][i*3+2] for j in range(num_models))
            
            averaged_keypoints.append([
                x_sum / num_models,
                y_sum / num_models,
                conf_sum / num_models
            ])
        
        return averaged_keypoints
    
    def _weighted_average_fusion(self, keypoints_list, confidences_list):
        """가중 평균 융합"""
        if not keypoints_list or not confidences_list:
            return []
        
        num_models = len(keypoints_list)
        num_keypoints = len(keypoints_list[0]) // 3
        
        # 모델별 가중치 계산
        model_weights = torch.softmax(torch.tensor(confidences_list), dim=0)
        
        weighted_keypoints = []
        for i in range(num_keypoints):
            x_weighted = sum(keypoints_list[j][i*3] * model_weights[j] for j in range(num_models))
            y_weighted = sum(keypoints_list[j][i*3+1] * model_weights[j] for j in range(num_models))
            conf_weighted = sum(keypoints_list[j][i*3+2] * model_weights[j] for j in range(num_models))
            
            weighted_keypoints.append([x_weighted, y_weighted, conf_weighted])
        
        return weighted_keypoints
    
    def _confidence_weighted_fusion(self, keypoints_list, confidences_list):
        """신뢰도 기반 가중 융합"""
        if not keypoints_list or not confidences_list:
            return []
        
        num_models = len(keypoints_list)
        num_keypoints = len(keypoints_list[0]) // 3
        
        fused_keypoints = []
        for i in range(num_keypoints):
            # 각 키포인트별로 신뢰도 기반 융합
            kp_confidences = [confidences_list[j][i] for j in range(num_models)]
            total_confidence = sum(kp_confidences)
            
            if total_confidence > 0:
                weights = [conf / total_confidence for conf in kp_confidences]
                
                x_fused = sum(keypoints_list[j][i*3] * weights[j] for j in range(num_models))
                y_fused = sum(keypoints_list[j][i*3+1] * weights[j] for j in range(num_models))
                conf_fused = sum(keypoints_list[j][i*3+2] * weights[j] for j in range(num_models))
            else:
                # 신뢰도가 모두 0인 경우 단순 평균
                x_fused = sum(keypoints_list[j][i*3] for j in range(num_models)) / num_models
                y_fused = sum(keypoints_list[j][i*3+1] for j in range(num_models)) / num_models
                conf_fused = 0.0
            
            fused_keypoints.append([x_fused, y_fused, conf_fused])
        
        return fused_keypoints

class PoseEnsembleManager:
    """포즈 앙상블 관리자"""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.ensemble_system = PoseEnsembleSystem(
            num_keypoints=config.num_keypoints,
            ensemble_models=['hrnet', 'openpose', 'yolo_pose', 'mediapipe'],
            hidden_dim=256
        )
        self.logger = logging.getLogger(f"{__name__}.PoseEnsembleManager")
    
    def load_ensemble_models(self, model_loader) -> Dict[str, Any]:
        """앙상블 모델 로드"""
        try:
            models = {}
            for model_name in ['hrnet', 'openpose', 'yolo_pose', 'mediapipe']:
                model = model_loader.get_model(model_name)
                if model:
                    models[model_name] = model
                    self.logger.info(f"✅ {model_name} 앙상블 모델 로드 완료")
                else:
                    self.logger.warning(f"⚠️ {model_name} 앙상블 모델 로드 실패")
            
            return models
            
        except Exception as e:
            self.logger.error(f"❌ 앙상블 모델 로드 실패: {e}")
            return {}
    
    def run_ensemble_inference(self, image, device='cuda') -> Dict[str, Any]:
        """앙상블 추론 실행"""
        try:
            start_time = time.time()
            
            # 각 모델로 추론 실행
            model_outputs = []
            model_confidences = []
            
            for model_name, model in self.ensemble_system.ensemble_models.items():
                try:
                    result = model.detect_poses(image)
                    if result.get("success", False):
                        model_outputs.append(result)
                        # 전체 신뢰도 계산
                        confidences = result.get("confidence_scores", [])
                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                        model_confidences.append(avg_confidence)
                        self.logger.info(f"✅ {model_name} 추론 성공 (신뢰도: {avg_confidence:.3f})")
                    else:
                        self.logger.warning(f"⚠️ {model_name} 추론 실패: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    self.logger.error(f"❌ {model_name} 추론 중 오류: {e}")
            
            if not model_outputs:
                return {"error": "모든 모델 추론 실패", "success": False}
            
            # 앙상블 융합
            ensemble_result = self.ensemble_system(model_outputs, model_confidences)
            
            if ensemble_result.get("success", False):
                ensemble_result["processing_time"] = time.time() - start_time
                ensemble_result["models_used"] = list(self.ensemble_system.ensemble_models.keys())
                ensemble_result["ensemble_method"] = self.config.ensemble_method
                
                self.logger.info(f"✅ 앙상블 추론 완료 (모델 수: {len(model_outputs)})")
            else:
                self.logger.error(f"❌ 앙상블 융합 실패: {ensemble_result.get('error', 'Unknown error')}")
            
            return ensemble_result
            
        except Exception as e:
            self.logger.error(f"❌ 앙상블 추론 실패: {e}")
            return {"error": str(e), "success": False}
    
    def calibrate_confidences(self, confidences: List[float]) -> List[float]:
        """신뢰도 보정"""
        if not self.config.enable_confidence_calibration:
            return confidences
        
        try:
            # Platt scaling 또는 temperature scaling 적용
            calibrated = []
            for conf in confidences:
                # 간단한 보정: sigmoid 함수 적용
                calibrated_conf = 1.0 / (1.0 + np.exp(-2.0 * (conf - 0.5)))
                calibrated.append(calibrated_conf)
            
            return calibrated
            
        except Exception as e:
            self.logger.warning(f"⚠️ 신뢰도 보정 실패: {e}")
            return confidences
    
    def estimate_uncertainty(self, model_outputs: List[Dict]) -> List[float]:
        """불확실성 추정"""
        if not self.config.enable_uncertainty_quantification:
            return [0.0] * self.config.num_keypoints
        
        try:
            uncertainties = []
            num_models = len(model_outputs)
            
            if num_models < 2:
                return [0.0] * self.config.num_keypoints
            
            # 각 키포인트별 불확실성 계산
            for i in range(self.config.num_keypoints):
                positions = []
                for output in model_outputs:
                    keypoints = output.get("keypoints", [])
                    if i < len(keypoints) and len(keypoints[i]) >= 2:
                        positions.append([keypoints[i][0], keypoints[i][1]])
                
                if len(positions) >= 2:
                    # 위치 분산 계산
                    positions = np.array(positions)
                    variance = np.var(positions, axis=0)
                    uncertainty = np.sqrt(np.sum(variance))
                else:
                    uncertainty = 0.0
                
                uncertainties.append(uncertainty)
            
            return uncertainties
            
        except Exception as e:
            self.logger.warning(f"⚠️ 불확실성 추정 실패: {e}")
            return [0.0] * self.config.num_keypoints
    
    def assess_ensemble_quality(self, ensemble_result: Dict[str, Any]) -> float:
        """앙상블 품질 평가"""
        try:
            quality_score = 0.0
            
            # 신뢰도 기반 품질
            confidences = ensemble_result.get("confidence_scores", [])
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                quality_score += avg_confidence * 0.4
            
            # 불확실성 기반 품질
            uncertainties = ensemble_result.get("uncertainties", [])
            if uncertainties:
                avg_uncertainty = sum(uncertainties) / len(uncertainties)
                uncertainty_quality = max(0, 1.0 - avg_uncertainty)
                quality_score += uncertainty_quality * 0.3
            
            # 모델 수 기반 품질
            models_used = ensemble_result.get("models_used", [])
            model_quality = min(1.0, len(models_used) / 4.0)  # 최대 4개 모델
            quality_score += model_quality * 0.3
            
            return min(1.0, quality_score)
            
        except Exception as e:
            self.logger.warning(f"⚠️ 앙상블 품질 평가 실패: {e}")
            return 0.5

# 모듈 내보내기
__all__ = ["PoseEnsembleSystem", "PoseEnsembleManager", "EnsembleConfig"]
