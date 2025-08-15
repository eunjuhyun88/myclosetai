#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 02: Pose Estimation Quality Enhancement
=============================================================

🎯 포즈 추정 결과 품질 향상
✅ 키포인트 정제 및 보간
✅ 자세 일관성 개선
✅ 노이즈 제거 및 스무딩
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import numpy as np

# PyTorch import 시도
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)

@dataclass
class EnhancementConfig:
    """품질 향상 설정"""
    smoothing_factor: float = 0.8
    interpolation_threshold: float = 0.3
    consistency_check: bool = True
    noise_reduction: bool = True
    temporal_window: int = 5

class QualityEnhancement:
    """
    🔥 포즈 추정 품질 향상 시스템
    
    포즈 추정 결과의 품질을 향상시키고 일관성을 개선합니다.
    """
    
    def __init__(self, config: EnhancementConfig = None):
        self.config = config or EnhancementConfig()
        self.logger = logging.getLogger(__name__)
        
        # 시간적 버퍼 (품질 향상용)
        self.temporal_buffer = []
        self.max_buffer_size = self.config.temporal_window
        
        self.logger.info("🎯 포즈 추정 품질 향상 시스템 초기화 완료")
    
    def enhance_pose_quality(self, keypoints: Union[torch.Tensor, np.ndarray],
                           confidences: Optional[Union[torch.Tensor, np.ndarray]] = None) -> Dict[str, Any]:
        """
        포즈 품질 향상
        
        Args:
            keypoints: 키포인트 좌표 (B, N, 3) - (x, y, confidence)
            confidences: 신뢰도 점수 (B, N)
        
        Returns:
            향상된 결과
        """
        try:
            # numpy로 변환
            if TORCH_AVAILABLE and isinstance(keypoints, torch.Tensor):
                keypoints_np = keypoints.detach().cpu().numpy()
            else:
                keypoints_np = np.array(keypoints)
            
            if confidences is not None:
                if TORCH_AVAILABLE and isinstance(confidences, torch.Tensor):
                    confidences_np = confidences.detach().cpu().numpy()
                else:
                    confidences_np = np.array(confidences)
            else:
                # 키포인트에서 신뢰도 추출
                confidences_np = keypoints_np[:, :, 2] if keypoints_np.shape[-1] == 3 else np.ones(keypoints_np.shape[:2])
            
            # 1단계: 노이즈 제거
            if self.config.noise_reduction:
                cleaned_keypoints = self._reduce_noise(keypoints_np, confidences_np)
            else:
                cleaned_keypoints = keypoints_np
            
            # 2단계: 일관성 검사 및 개선
            if self.config.consistency_check:
                consistent_keypoints = self._improve_consistency(cleaned_keypoints, confidences_np)
            else:
                consistent_keypoints = cleaned_keypoints
            
            # 3단계: 시간적 스무딩
            smoothed_keypoints = self._temporal_smoothing(consistent_keypoints)
            
            # 4단계: 낮은 신뢰도 키포인트 보간
            interpolated_keypoints = self._interpolate_low_confidence(smoothed_keypoints, confidences_np)
            
            # 시간적 버퍼 업데이트
            self._update_temporal_buffer(interpolated_keypoints)
            
            # 최종 품질 점수 계산
            final_quality = self._calculate_final_quality(interpolated_keypoints, confidences_np)
            
            result = {
                'enhanced_keypoints': interpolated_keypoints,
                'original_keypoints': keypoints_np,
                'confidences': confidences_np,
                'quality_score': final_quality,
                'enhancement_applied': {
                    'noise_reduction': self.config.noise_reduction,
                    'consistency_improvement': self.config.consistency_check,
                    'temporal_smoothing': True,
                    'interpolation': True
                }
            }
            
            self.logger.info(f"✅ 포즈 품질 향상 완료 (품질 점수: {final_quality:.3f})")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 포즈 품질 향상 실패: {e}")
            return {
                'enhanced_keypoints': keypoints_np if 'keypoints_np' in locals() else np.zeros((1, 17, 3)),
                'original_keypoints': keypoints_np if 'keypoints_np' in locals() else np.zeros((1, 17, 3)),
                'confidences': confidences_np if 'confidences_np' in locals() else np.zeros((1, 17)),
                'quality_score': 0.0,
                'error': str(e)
            }
    
    def _reduce_noise(self, keypoints: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        """노이즈 제거"""
        cleaned = keypoints.copy()
        
        # 낮은 신뢰도 키포인트에 대해 이전 프레임 값 사용
        low_confidence_mask = confidences < self.config.interpolation_threshold
        
        if self.temporal_buffer:
            prev_keypoints = self.temporal_buffer[-1]
            for batch_idx in range(keypoints.shape[0]):
                for kp_idx in range(keypoints.shape[1]):
                    if low_confidence_mask[batch_idx, kp_idx]:
                        if prev_keypoints.shape[0] > batch_idx and prev_keypoints.shape[1] > kp_idx:
                            cleaned[batch_idx, kp_idx] = prev_keypoints[batch_idx, kp_idx]
        
        return cleaned
    
    def _improve_consistency(self, keypoints: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        """일관성 개선"""
        consistent = keypoints.copy()
        
        # 키포인트 간의 거리 일관성 검사
        for batch_idx in range(keypoints.shape[0]):
            batch_keypoints = keypoints[batch_idx]
            batch_confidences = confidences[batch_idx]
            
            # 유효한 키포인트만 선택
            valid_mask = batch_confidences > self.config.interpolation_threshold
            if np.sum(valid_mask) < 2:
                continue
            
            valid_keypoints = batch_keypoints[valid_mask]
            
            # 키포인트 간의 거리 계산
            distances = []
            for i in range(len(valid_keypoints)):
                for j in range(i + 1, len(valid_keypoints)):
                    dist = np.linalg.norm(valid_keypoints[i][:2] - valid_keypoints[j][:2])
                    distances.append(dist)
            
            if distances:
                mean_distance = np.mean(distances)
                std_distance = np.std(distances)
                
                # 비정상적으로 먼 거리인 키포인트 보정
                for i, dist in enumerate(distances):
                    if abs(dist - mean_distance) > 2 * std_distance:
                        # 해당 키포인트를 평균 위치로 보정
                        if self.temporal_buffer:
                            prev_keypoints = self.temporal_buffer[-1]
                            if prev_keypoints.shape[0] > batch_idx:
                                consistent[batch_idx, i] = prev_keypoints[batch_idx, i]
        
        return consistent
    
    def _temporal_smoothing(self, keypoints: np.ndarray) -> np.ndarray:
        """시간적 스무딩"""
        if not self.temporal_buffer:
            return keypoints
        
        smoothed = keypoints.copy()
        smoothing_factor = self.config.smoothing_factor
        
        for batch_idx in range(keypoints.shape[0]):
            for kp_idx in range(keypoints.shape[1]):
                # 이전 프레임들의 가중 평균
                weighted_sum = np.zeros_like(keypoints[batch_idx, kp_idx])
                total_weight = 0.0
                
                for frame_idx, prev_keypoints in enumerate(self.temporal_buffer):
                    if prev_keypoints.shape[0] > batch_idx and prev_keypoints.shape[1] > kp_idx:
                        weight = smoothing_factor ** (len(self.temporal_buffer) - frame_idx)
                        weighted_sum += weight * prev_keypoints[batch_idx, kp_idx]
                        total_weight += weight
                
                if total_weight > 0:
                    smoothed[batch_idx, kp_idx] = weighted_sum / total_weight
        
        return smoothed
    
    def _interpolate_low_confidence(self, keypoints: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        """낮은 신뢰도 키포인트 보간"""
        interpolated = keypoints.copy()
        
        for batch_idx in range(keypoints.shape[0]):
            for kp_idx in range(keypoints.shape[1]):
                if confidences[batch_idx, kp_idx] < self.config.interpolation_threshold:
                    # 이전 프레임 값으로 보간
                    if self.temporal_buffer:
                        prev_keypoints = self.temporal_buffer[-1]
                        if prev_keypoints.shape[0] > batch_idx and prev_keypoints.shape[1] > kp_idx:
                            interpolated[batch_idx, kp_idx] = prev_keypoints[batch_idx, kp_idx]
        
        return interpolated
    
    def _update_temporal_buffer(self, keypoints: np.ndarray):
        """시간적 버퍼 업데이트"""
        self.temporal_buffer.append(keypoints.copy())
        
        # 버퍼 크기 제한
        if len(self.temporal_buffer) > self.max_buffer_size:
            self.temporal_buffer.pop(0)
    
    def _calculate_final_quality(self, keypoints: np.ndarray, confidences: np.ndarray) -> float:
        """최종 품질 점수 계산"""
        try:
            # 신뢰도 점수
            confidence_score = np.mean(confidences)
            
            # 키포인트 분포 점수
            distribution_score = self._calculate_distribution_score(keypoints)
            
            # 일관성 점수
            consistency_score = self._calculate_consistency_score(keypoints)
            
            # 종합 품질 점수
            final_score = (
                0.4 * confidence_score +
                0.3 * distribution_score +
                0.3 * consistency_score
            )
            
            return float(np.clip(final_score, 0.0, 1.0))
            
        except Exception as e:
            self.logger.warning(f"품질 점수 계산 실패: {e}")
            return 0.0
    
    def _calculate_distribution_score(self, keypoints: np.ndarray) -> float:
        """분포 점수 계산"""
        try:
            # 키포인트들이 너무 집중되지 않았는지 확인
            batch_scores = []
            
            for batch_idx in range(keypoints.shape[0]):
                batch_keypoints = keypoints[batch_idx]
                
                # 키포인트 간의 거리 계산
                distances = []
                for i in range(batch_keypoints.shape[0]):
                    for j in range(i + 1, batch_keypoints.shape[0]):
                        dist = np.linalg.norm(batch_keypoints[i][:2] - batch_keypoints[j][:2])
                        distances.append(dist)
                
                if distances:
                    distances = np.array(distances)
                    # 적절한 거리 범위 내에 있는 비율
                    good_distances = np.sum((distances > 10) & (distances < 200))
                    distribution_score = good_distances / len(distances) if len(distances) > 0 else 0.0
                    batch_scores.append(distribution_score)
            
            return float(np.mean(batch_scores)) if batch_scores else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_consistency_score(self, keypoints: np.ndarray) -> float:
        """일관성 점수 계산"""
        try:
            if not self.temporal_buffer:
                return 0.5  # 기본값
            
            # 이전 프레임과의 일관성
            prev_keypoints = self.temporal_buffer[-1]
            consistency_scores = []
            
            for batch_idx in range(min(keypoints.shape[0], prev_keypoints.shape[0])):
                batch_consistency = []
                for kp_idx in range(min(keypoints.shape[1], prev_keypoints.shape[1])):
                    current_kp = keypoints[batch_idx, kp_idx][:2]
                    prev_kp = prev_keypoints[batch_idx, kp_idx][:2]
                    
                    # 키포인트 이동 거리
                    movement = np.linalg.norm(current_kp - prev_kp)
                    
                    # 적절한 이동 거리인지 확인 (너무 크거나 작지 않음)
                    if 1.0 <= movement <= 50.0:
                        batch_consistency.append(1.0)
                    else:
                        batch_consistency.append(max(0.0, 1.0 - movement / 100.0))
                
                if batch_consistency:
                    consistency_scores.append(np.mean(batch_consistency))
            
            return float(np.mean(consistency_scores)) if consistency_scores else 0.5
            
        except Exception:
            return 0.5
    
    def reset_temporal_state(self):
        """시간적 상태 초기화"""
        self.temporal_buffer.clear()
        self.logger.info("✅ 시간적 상태 초기화 완료")
    
    def get_config(self) -> EnhancementConfig:
        """현재 설정 반환"""
        return self.config
    
    def update_config(self, new_config: EnhancementConfig):
        """설정 업데이트"""
        self.config = new_config
        self.logger.info("✅ 품질 향상 설정 업데이트 완료")

# 기본 품질 향상 시스템 생성 함수
def create_pose_estimation_quality_enhancement(config: EnhancementConfig = None) -> QualityEnhancement:
    """포즈 추정 품질 향상 시스템 생성"""
    return QualityEnhancement(config)

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # 품질 향상 시스템 생성
    enhancer = create_pose_estimation_quality_enhancement()
    
    # 테스트 데이터
    test_keypoints = np.random.rand(1, 17, 3)  # 1개 배치, 17개 키포인트, (x, y, conf)
    test_confidences = np.random.rand(1, 17)
    
    # 품질 향상 수행
    result = enhancer.enhance_pose_quality(test_keypoints, test_confidences)
    
    print(f"품질 향상 결과: {result['quality_score']:.3f}")
    print(f"적용된 향상 기법: {result['enhancement_applied']}")
