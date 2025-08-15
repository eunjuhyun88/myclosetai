#!/usr/bin/env python3
"""
🔥 MyCloset AI - Cloth Warping Inference Engine
===============================================

🎯 의류 워핑 추론 엔진
✅ 다중 모델 추론 관리
✅ M3 Max 최적화
✅ 메모리 효율적 처리
✅ 실시간 워핑 지원
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import time
import cv2

# 로거 설정
logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    """추론 설정"""
    batch_size: int = 1
    use_mps: bool = True
    memory_efficient: bool = True
    enable_ensemble: bool = True
    confidence_threshold: float = 0.5
    max_models: int = 8
    warp_resolution: Tuple[int, int] = (256, 256)

class ClothWarpingInferenceEngine(nn.Module):
    """
    🔥 Cloth Warping 추론 엔진
    
    의류 워핑을 위한 고성능 추론 시스템입니다.
    """
    
    def __init__(self, config: InferenceConfig = None):
        super().__init__()
        self.config = config or InferenceConfig()
        self.logger = logging.getLogger(__name__)
        
        # MPS 디바이스 확인
        self.device = torch.device("mps" if torch.backends.mps.is_available() and self.config.use_mps else "cpu")
        self.logger.info(f"🎯 Cloth Warping 추론 엔진 초기화 (디바이스: {self.device})")
        
        # 모델 초기화
        self.models = {}
        self.model_weights = {}
        self.ensemble_system = None
        
        # 추론 통계
        self.inference_stats = {
            "total_inferences": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "success_rate": 1.0
        }
        
        self.logger.info("✅ Cloth Warping 추론 엔진 초기화 완료")
    
    def load_model(self, model_name: str, model_path: str, weight: float = 1.0):
        """모델 로드"""
        try:
            # 실제 모델 로드 로직 (여기서는 더미 모델 생성)
            model = self._create_dummy_model(model_name)
            model.to(self.device)
            model.eval()
            
            self.models[model_name] = model
            self.model_weights[model_name] = weight
            
            self.logger.info(f"✅ 모델 로드 완료: {model_name} (가중치: {weight})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 모델 로드 실패: {model_name} - {str(e)}")
            return False
    
    def _create_dummy_model(self, model_name: str) -> nn.Module:
        """더미 모델 생성 (실제 구현에서는 실제 모델 로드)"""
        class DummyClothWarpingModel(nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name
                # 간단한 CNN 기반 워핑 모델
                self.feature_extractor = nn.Sequential(
                    nn.Conv2d(6, 64, 3, padding=1),  # 3채널 의류 + 3채널 타겟
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU()
                )
                
                self.warping_head = nn.Sequential(
                    nn.Conv2d(256, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 2, 1),  # 2채널 (x, y 변위)
                    nn.Tanh()  # -1 ~ 1 범위
                )
            
            def forward(self, cloth_image, target_image):
                # 의류와 타겟 이미지 결합
                combined_input = torch.cat([cloth_image, target_image], dim=1)
                features = self.feature_extractor(combined_input)
                warping_field = self.warping_head(features)
                return warping_field
        
        return DummyClothWarpingModel(model_name)
    
    def set_ensemble_system(self, ensemble_system):
        """앙상블 시스템 설정"""
        self.ensemble_system = ensemble_system
        self.logger.info("✅ 앙상블 시스템 설정 완료")
    
    def forward(self, cloth_image: torch.Tensor, target_image: torch.Tensor, 
                pose_keypoints: Optional[torch.Tensor] = None,
                body_shape: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        추론 수행
        
        Args:
            cloth_image: 의류 이미지 (B, C, H, W)
            target_image: 타겟 이미지 (B, C, H, W)
            pose_keypoints: 포즈 키포인트 (B, N, 2)
            body_shape: 신체 형태 정보 (B, M)
        
        Returns:
            워핑 결과
        """
        start_time = time.time()
        
        try:
            # 입력 검증
            if not self._validate_inputs(cloth_image, target_image):
                raise ValueError("입력 검증 실패")
            
            # 디바이스 이동
            cloth_image = cloth_image.to(self.device)
            target_image = target_image.to(self.device)
            if pose_keypoints is not None:
                pose_keypoints = pose_keypoints.to(self.device)
            if body_shape is not None:
                body_shape = body_shape.to(self.device)
            
            # 개별 모델 추론
            model_outputs = []
            model_confidences = []
            
            for model_name, model in self.models.items():
                try:
                    with torch.no_grad():
                        # 모델별 추론
                        output = self._inference_single_model(model, cloth_image, target_image, pose_keypoints, body_shape)
                        confidence = self._calculate_confidence(output)
                        
                        model_outputs.append(output)
                        model_confidences.append(confidence)
                        
                except Exception as e:
                    self.logger.warning(f"모델 {model_name} 추론 실패: {str(e)}")
                    continue
            
            if not model_outputs:
                raise RuntimeError("모든 모델 추론 실패")
            
            # 앙상블 추론
            if self.ensemble_system and len(model_outputs) > 1:
                ensemble_output = self.ensemble_system(model_outputs, model_confidences)
            else:
                ensemble_output = model_outputs[0] if model_outputs else torch.zeros_like(cloth_image[:, :2, :, :])
            
            # 후처리
            final_output = self._postprocess_output(ensemble_output, cloth_image, target_image)
            
            # 워핑된 의류 생성
            warped_cloth = self._apply_warping(cloth_image, final_output)
            
            # 추론 통계 업데이트
            inference_time = time.time() - start_time
            self._update_inference_stats(inference_time, True)
            
            # 결과 반환
            result = {
                "warping_field": final_output,
                "warped_cloth": warped_cloth,
                "model_outputs": model_outputs,
                "model_confidences": model_confidences,
                "ensemble_output": ensemble_output,
                "inference_time": inference_time,
                "success": True
            }
            
            self.logger.debug(f"✅ 워핑 추론 완료 - 시간: {inference_time:.3f}초")
            return result
            
        except Exception as e:
            # 추론 실패 처리
            inference_time = time.time() - start_time
            self._update_inference_stats(inference_time, False)
            
            self.logger.error(f"❌ 워핑 추론 실패: {str(e)}")
            
            # 실패 시 기본값 반환
            return {
                "warping_field": torch.zeros_like(cloth_image[:, :2, :, :]),
                "warped_cloth": cloth_image,
                "model_outputs": [],
                "model_confidences": [],
                "ensemble_output": torch.zeros_like(cloth_image[:, :2, :, :]),
                "inference_time": inference_time,
                "success": False,
                "error": str(e)
            }
    
    def _validate_inputs(self, cloth_image: torch.Tensor, target_image: torch.Tensor) -> bool:
        """입력 검증"""
        if cloth_image.dim() != 4 or target_image.dim() != 4:
            return False
        
        if cloth_image.size(0) != target_image.size(0):
            return False
        
        if cloth_image.size(2) != target_image.size(2) or cloth_image.size(3) != target_image.size(3):
            return False
        
        return True
    
    def _inference_single_model(self, model: nn.Module, cloth_image: torch.Tensor, 
                               target_image: torch.Tensor, pose_keypoints: Optional[torch.Tensor],
                               body_shape: Optional[torch.Tensor]) -> torch.Tensor:
        """단일 모델 추론"""
        # 이미지 전처리
        processed_cloth = self._preprocess_image(cloth_image)
        processed_target = self._preprocess_image(target_image)
        
        # 모델 추론
        if pose_keypoints is not None and body_shape is not None:
            # 포즈와 신체 형태 정보가 있는 경우
            output = model(processed_cloth, processed_target, pose_keypoints, body_shape)
        else:
            # 기본 워핑
            output = model(processed_cloth, processed_target)
        
        return output
    
    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """이미지 전처리"""
        # 정규화 (0-1 범위)
        if image.max() > 1.0:
            image = image / 255.0
        
        # 크기 조정 (필요한 경우)
        target_size = self.config.warp_resolution
        if image.size(2) != target_size[0] or image.size(3) != target_size[1]:
            image = F.interpolate(image, size=target_size, mode='bilinear', align_corners=False)
        
        return image
    
    def _calculate_confidence(self, output: torch.Tensor) -> float:
        """출력 신뢰도 계산"""
        if output.numel() == 0:
            return 0.0
        
        # 워핑 필드의 품질 기반 신뢰도 계산
        # 변위의 크기가 적절한지 확인
        displacement_magnitude = torch.sqrt(output[:, 0:1]**2 + output[:, 1:2]**2)
        avg_displacement = float(displacement_magnitude.mean().item())
        
        # 적절한 변위 범위 (0.1 ~ 0.5)에서 높은 신뢰도
        if 0.1 <= avg_displacement <= 0.5:
            confidence = 0.9
        elif 0.05 <= avg_displacement <= 0.8:
            confidence = 0.7
        else:
            confidence = 0.3
        
        return confidence
    
    def _postprocess_output(self, output: torch.Tensor, cloth_image: torch.Tensor, 
                           target_image: torch.Tensor) -> torch.Tensor:
        """출력 후처리"""
        # 출력 크기를 원본 이미지 크기로 조정
        if output.size(2) != cloth_image.size(2) or output.size(3) != cloth_image.size(3):
            output = F.interpolate(output, size=(cloth_image.size(2), cloth_image.size(3)), 
                                 mode='bilinear', align_corners=False)
        
        # 워핑 필드 스무딩
        output = self._smooth_warping_field(output)
        
        # 신뢰도 임계값 적용
        if self.config.confidence_threshold > 0:
            confidence_mask = self._calculate_warping_confidence(output) > self.config.confidence_threshold
            output = output * confidence_mask.float()
        
        return output
    
    def _smooth_warping_field(self, warping_field: torch.Tensor) -> torch.Tensor:
        """워핑 필드 스무딩"""
        # 가우시안 스무딩 적용
        smoothed_field = warping_field.clone()
        
        for b in range(warping_field.size(0)):
            for c in range(warping_field.size(1)):
                channel = warping_field[b, c]
                if channel.numel() > 0:
                    smoothed_field[b, c] = self._gaussian_smooth_2d(channel)
        
        return smoothed_field
    
    def _gaussian_smooth_2d(self, channel: torch.Tensor) -> torch.Tensor:
        """2D 가우시안 스무딩"""
        if channel.dim() != 2:
            return channel
        
        # 가우시안 커널 생성
        kernel_size = 3
        sigma = 0.5
        
        # 1D 가우시안 커널
        x = torch.arange(-kernel_size // 2, kernel_size // 2 + 1, device=channel.device)
        gaussian_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        
        # 2D 가우시안 커널
        gaussian_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
        
        # 패딩 추가
        padded_channel = F.pad(channel.unsqueeze(0).unsqueeze(0), 
                              (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), 
                              mode='reflect')
        
        # 컨볼루션 적용
        smoothed_channel = F.conv2d(padded_channel, gaussian_2d.unsqueeze(0).unsqueeze(0))
        
        return smoothed_channel.squeeze()
    
    def _calculate_warping_confidence(self, warping_field: torch.Tensor) -> torch.Tensor:
        """워핑 신뢰도 계산"""
        # 변위의 크기와 방향 일관성 기반 신뢰도
        displacement_magnitude = torch.sqrt(warping_field[:, 0:1]**2 + warping_field[:, 1:2]**2)
        
        # 적절한 변위 범위에서 높은 신뢰도
        confidence = torch.ones_like(displacement_magnitude)
        
        # 너무 큰 변위는 낮은 신뢰도
        confidence[displacement_magnitude > 0.8] = 0.3
        confidence[displacement_magnitude > 0.5] = 0.6
        
        # 너무 작은 변위도 낮은 신뢰도
        confidence[displacement_magnitude < 0.05] = 0.4
        
        return confidence
    
    def _apply_warping(self, cloth_image: torch.Tensor, warping_field: torch.Tensor) -> torch.Tensor:
        """워핑 적용"""
        batch_size, channels, height, width = cloth_image.shape
        
        # 그리드 생성
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=self.device),
            torch.linspace(-1, 1, width, device=self.device),
            indexing='ij'
        )
        
        # 워핑 필드 적용
        warped_grid_x = grid_x + warping_field[:, 0] * 0.5  # -1 ~ 1 범위로 제한
        warped_grid_y = grid_y + warping_field[:, 1] * 0.5
        
        # 그리드 정규화
        warped_grid_x = torch.clamp(warped_grid_x, -1, 1)
        warped_grid_y = torch.clamp(warped_grid_y, -1, 1)
        
        # 그리드 결합
        warped_grid = torch.stack([warped_grid_x, warped_grid_y], dim=-1)
        
        # 워핑 적용
        warped_cloth = F.grid_sample(cloth_image, warped_grid, mode='bilinear', 
                                    padding_mode='border', align_corners=False)
        
        return warped_cloth
    
    def _update_inference_stats(self, inference_time: float, success: bool):
        """추론 통계 업데이트"""
        self.inference_stats["total_inferences"] += 1
        self.inference_stats["total_time"] += inference_time
        self.inference_stats["avg_time"] = self.inference_stats["total_time"] / self.inference_stats["total_inferences"]
        
        if not success:
            failed_count = int((1 - self.inference_stats["success_rate"]) * self.inference_stats["total_inferences"])
            self.inference_stats["success_rate"] = (self.inference_stats["total_inferences"] - failed_count - 1) / self.inference_stats["total_inferences"]
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """추론 통계 반환"""
        return self.inference_stats.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        model_info = {}
        for model_name, model in self.models.items():
            model_info[model_name] = {
                "weight": self.model_weights.get(model_name, 1.0),
                "parameters": sum(p.numel() for p in model.parameters()),
                "device": str(next(model.parameters()).device)
            }
        return model_info
    
    def clear_cache(self):
        """캐시 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("✅ 캐시 정리 완료")

# 추론 엔진 인스턴스 생성 함수
def create_cloth_warping_inference_engine(config: InferenceConfig = None) -> ClothWarpingInferenceEngine:
    """Cloth Warping 추론 엔진 생성"""
    return ClothWarpingInferenceEngine(config)

# 기본 설정으로 추론 엔진 생성
def create_default_inference_engine() -> ClothWarpingInferenceEngine:
    """기본 설정으로 추론 엔진 생성"""
    config = InferenceConfig(
        batch_size=1,
        use_mps=True,
        memory_efficient=True,
        enable_ensemble=True,
        confidence_threshold=0.5,
        max_models=8,
        warp_resolution=(256, 256)
    )
    return ClothWarpingInferenceEngine(config)

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # 추론 엔진 생성
    engine = create_default_inference_engine()
    
    # 더미 모델 로드
    engine.load_model("warping_model_1", "dummy_path_1", 1.0)
    engine.load_model("warping_model_2", "dummy_path_2", 0.8)
    
    # 테스트 데이터 생성
    batch_size, channels, height, width = 2, 3, 256, 256
    test_cloth = torch.randn(batch_size, channels, height, width)
    test_target = torch.randn(batch_size, channels, height, width)
    
    # 추론 수행
    result = engine(test_cloth, test_target)
    print(f"워핑 추론 결과: {result['success']}")
    print(f"추론 시간: {result['inference_time']:.3f}초")
    print(f"워핑된 의류 형태: {result['warped_cloth'].shape}")
    print(f"모델 정보: {engine.get_model_info()}")
    print(f"추론 통계: {engine.get_inference_stats()}")
