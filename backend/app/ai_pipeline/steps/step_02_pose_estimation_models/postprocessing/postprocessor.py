#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 02: Pose Estimation Postprocessor
=======================================================

🎯 포즈 추정 결과 후처리
✅ 키포인트 정제 및 필터링
✅ 자세 품질 평가
✅ 결과 최적화
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
class PostprocessingConfig:
    """후처리 설정"""
    confidence_threshold: float = 0.5
    smoothing_factor: float = 0.8
    max_keypoints: int = 17
    use_temporal_smoothing: bool = True
    quality_check: bool = True

class Postprocessor:
    """
    🔥 포즈 추정 후처리기
    
    포즈 추정 결과를 정제하고 최적화합니다.
    """
    
    def __init__(self, config: PostprocessingConfig = None):
        self.config = config or PostprocessingConfig()
        self.logger = logging.getLogger(__name__)
        
        # 이전 프레임 키포인트 (시간적 스무딩용)
        self.previous_keypoints = None
        
        self.logger.info("🎯 포즈 추정 후처리기 초기화 완료")
    
    def process_keypoints(self, keypoints: Union[torch.Tensor, np.ndarray], 
                         confidences: Optional[Union[torch.Tensor, np.ndarray]] = None) -> Dict[str, Any]:
        """
        키포인트 후처리
        
        Args:
            keypoints: 키포인트 좌표 (B, N, 3) - (x, y, confidence)
            confidences: 신뢰도 점수 (B, N)
        
        Returns:
            후처리된 결과
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
            
            # 신뢰도 기반 필터링
            filtered_keypoints = self._filter_by_confidence(keypoints_np, confidences_np)
            
            # 시간적 스무딩
            if self.config.use_temporal_smoothing:
                smoothed_keypoints = self._temporal_smoothing(filtered_keypoints)
            else:
                smoothed_keypoints = filtered_keypoints
            
            # 품질 평가
            quality_score = self._assess_pose_quality(smoothed_keypoints, confidences_np)
            
            # 결과 정리
            result = {
                'keypoints': smoothed_keypoints,
                'confidences': confidences_np,
                'quality_score': quality_score,
                'num_valid_keypoints': np.sum(confidences_np > self.config.confidence_threshold),
                'postprocessing_config': self.config
            }
            
            self.logger.info(f"✅ 키포인트 후처리 완료 (품질 점수: {quality_score:.3f})")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 키포인트 후처리 실패: {e}")
            return {
                'keypoints': keypoints_np if 'keypoints_np' in locals() else np.zeros((1, self.config.max_keypoints, 3)),
                'confidences': confidences_np if 'confidences_np' in locals() else np.zeros((1, self.config.max_keypoints)),
                'quality_score': 0.0,
                'num_valid_keypoints': 0,
                'error': str(e)
            }
    
    def _filter_by_confidence(self, keypoints: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        """신뢰도 기반 필터링"""
        # 신뢰도 임계값 미만 키포인트 제거
        mask = confidences > self.config.confidence_threshold
        
        # 필터링된 키포인트 반환
        filtered = keypoints.copy()
        filtered[~mask] = 0  # 낮은 신뢰도 키포인트는 0으로 설정
        
        return filtered
    
    def _temporal_smoothing(self, keypoints: np.ndarray) -> np.ndarray:
        """시간적 스무딩"""
        if self.previous_keypoints is None:
            self.previous_keypoints = keypoints.copy()
            return keypoints
        
        # 이전 프레임과 현재 프레임을 가중 평균
        smoothed = (self.config.smoothing_factor * self.previous_keypoints + 
                   (1 - self.config.smoothing_factor) * keypoints)
        
        # 이전 프레임 업데이트
        self.previous_keypoints = smoothed.copy()
        
        return smoothed
    
    def _assess_pose_quality(self, keypoints: np.ndarray, confidences: np.ndarray) -> float:
        """자세 품질 평가"""
        try:
            # 유효한 키포인트 수
            valid_count = np.sum(confidences > self.config.confidence_threshold)
            
            # 평균 신뢰도
            mean_confidence = np.mean(confidences[confidences > self.config.confidence_threshold])
            
            # 키포인트 분산 (너무 집중되지 않았는지)
            if valid_count > 0:
                valid_keypoints = keypoints[confidences > self.config.confidence_threshold]
                variance = np.var(valid_keypoints[:, :2])  # x, y 좌표만
            else:
                variance = 0.0
            
            # 품질 점수 계산 (0-1 범위)
            quality_score = (
                0.4 * (valid_count / self.config.max_keypoints) +  # 유효 키포인트 비율
                0.4 * mean_confidence +  # 평균 신뢰도
                0.2 * min(1.0, variance / 1000.0)  # 분산 (정규화)
            )
            
            return float(np.clip(quality_score, 0.0, 1.0))
            
        except Exception as e:
            self.logger.warning(f"품질 평가 실패: {e}")
            return 0.0
    
    def reset_temporal_state(self):
        """시간적 상태 초기화"""
        self.previous_keypoints = None
        self.logger.info("✅ 시간적 상태 초기화 완료")
    
    def get_config(self) -> PostprocessingConfig:
        """현재 설정 반환"""
        return self.config
    
    def update_config(self, new_config: PostprocessingConfig):
        """설정 업데이트"""
        self.config = new_config
        self.logger.info("✅ 후처리 설정 업데이트 완료")

# PoseEstimationPostprocessor 클래스 추가 (import 호환성을 위해)
class PoseEstimationPostprocessor(Postprocessor):
    """
    🔥 포즈 추정 후처리기 (Postprocessor 상속)
    
    포즈 추정 결과를 정제하고 최적화합니다.
    """
    
    def __init__(self, config: PostprocessingConfig = None):
        super().__init__(config)
        self.logger.info("🎯 PoseEstimationPostprocessor 초기화 완료")
    
    def get_processor_info(self) -> Dict[str, Any]:
        """후처리기 정보 반환"""
        return {
            'processor_type': 'PoseEstimationPostprocessor',
            'inherits_from': 'Postprocessor',
            'config': self.config,
            'capabilities': [
                'confidence_filtering',
                'temporal_smoothing',
                'quality_assessment',
                'keypoint_optimization'
            ]
        }

# 기본 후처리기 생성 함수
def create_pose_estimation_postprocessor(config: PostprocessingConfig = None) -> PoseEstimationPostprocessor:
    """포즈 추정 후처리기 생성"""
    return PoseEstimationPostprocessor(config)

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # 후처리기 생성
    postprocessor = create_pose_estimation_postprocessor()
    
    # 테스트 데이터
    test_keypoints = np.random.rand(1, 17, 3)  # 1개 배치, 17개 키포인트, (x, y, conf)
    test_confidences = np.random.rand(1, 17)
    
    # 후처리 수행
    result = postprocessor.process_keypoints(test_keypoints, test_confidences)
    
    print(f"후처리 결과: {result['quality_score']:.3f}")
    print(f"유효 키포인트 수: {result['num_valid_keypoints']}")
