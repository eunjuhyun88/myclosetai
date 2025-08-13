"""
🔥 Processing Utils
==================

이미지 처리 및 유틸리티 메서드들

Author: MyCloset AI Team
Date: 2025-08-07
Version: 1.0
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple, List
import logging


class ProcessingUtils:
    """이미지 처리 유틸리티 클래스"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.logger = logging.getLogger(__name__)
    
    def preprocess_image(self, image, device: str = None, mode: str = 'advanced'):
        """이미지 전처리"""
        try:
            if device is None:
                device = self.device
            
            # NumPy 배열로 변환
            if hasattr(image, 'convert'):  # PIL Image
                image_np = np.array(image.convert('RGB'))
            elif hasattr(image, 'shape'):  # NumPy 배열
                image_np = image
            else:
                raise ValueError("지원하지 않는 이미지 타입")
            
            # 이미지 정규화
            if image_np.dtype != np.float32:
                image_np = image_np.astype(np.float32) / 255.0
            
            # 텐서로 변환
            if len(image_np.shape) == 3:
                image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
            else:
                image_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0)
            
            # 디바이스로 이동
            image_tensor = image_tensor.to(device)
            
            # 고급 전처리
            if mode == 'advanced':
                image_tensor = self._apply_advanced_preprocessing(image_tensor)
            
            return image_tensor
            
        except Exception as e:
            self.logger.error(f"❌ 이미지 전처리 실패: {e}")
            # 기본 전처리
            return self._create_default_tensor(device)
    
    def _apply_advanced_preprocessing(self, image_tensor):
        """고급 전처리 적용"""
        try:
            # 정규화
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image_tensor.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(image_tensor.device)
            
            image_tensor = (image_tensor - mean) / std
            
            return image_tensor
            
        except Exception as e:
            self.logger.warning(f"⚠️ 고급 전처리 실패: {e}")
            return image_tensor
    
    def _create_default_tensor(self, device):
        """기본 텐서 생성"""
        return torch.randn(1, 3, 512, 512).to(device)
    
    def postprocess_result(self, inference_result: Dict[str, Any], original_image, model_type: str = 'graphonomy') -> Dict[str, Any]:
        """추론 결과 후처리"""
        try:
            # 파싱 맵 추출
            parsing_pred = inference_result.get('parsing_pred')
            if parsing_pred is None:
                self.logger.warning("⚠️ 파싱 예측 결과가 없음")
                return self._create_fallback_result()
            
            # 텐서를 NumPy로 변환
            if isinstance(parsing_pred, torch.Tensor):
                parsing_map = parsing_pred.detach().cpu().numpy()
            else:
                parsing_map = np.array(parsing_pred)
            
            # 차원 정리
            if len(parsing_map.shape) == 4:  # [B, C, H, W] -> [H, W]
                parsing_map = parsing_map[0]  # 배치 차원 제거
                if parsing_map.shape[0] > 1:  # 채널이 여러 개인 경우
                    parsing_map = np.argmax(parsing_map, axis=0)  # 최대값 인덱스
            elif len(parsing_map.shape) == 3:  # [B, H, W] -> [H, W]
                parsing_map = parsing_map[0]
            
            # 신뢰도 계산
            confidence = inference_result.get('confidence', 0.8)
            
            # 품질 메트릭 계산
            quality_metrics = self._calculate_quality_metrics(parsing_map)
            
            return {
                'parsing_map': parsing_map,
                'confidence': confidence,
                'quality_metrics': quality_metrics,
                'model_type': model_type
            }
            
        except Exception as e:
            self.logger.error(f"❌ 후처리 실패: {e}")
            return self._create_fallback_result()
    
    def _create_fallback_result(self):
        """폴백 결과 생성"""
        return {
            'parsing_map': np.zeros((512, 512), dtype=np.uint8),
            'confidence': 0.5,
            'quality_metrics': {'overall_quality': 0.5},
            'model_type': 'fallback'
        }
    
    def _calculate_quality_metrics(self, parsing_map: np.ndarray) -> Dict[str, float]:
        """품질 메트릭 계산"""
        try:
            # 기본 품질 메트릭
            unique_labels = len(np.unique(parsing_map))
            mean_intensity = np.mean(parsing_map)
            std_intensity = np.std(parsing_map)
            
            # 전체 품질 점수
            overall_quality = min(1.0, (unique_labels / 20.0) * 0.5 + (mean_intensity / 255.0) * 0.3 + (std_intensity / 50.0) * 0.2)
            
            return {
                'overall_quality': overall_quality,
                'unique_labels': unique_labels,
                'mean_intensity': mean_intensity,
                'std_intensity': std_intensity
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ 품질 메트릭 계산 실패: {e}")
            return {'overall_quality': 0.5}
    
    def standardize_channels(self, tensor: torch.Tensor, target_channels: int = 20) -> torch.Tensor:
        """채널 수 표준화 (근본적 해결)"""
        try:
            # 🔥 입력 검증
            if tensor is None:
                self.logger.warning("⚠️ 텐서가 None입니다.")
                return torch.zeros((1, target_channels, 512, 512), device='cpu', dtype=torch.float32)
            
            # 🔥 차원 검증
            if len(tensor.shape) != 4:
                self.logger.warning(f"⚠️ 텐서 차원이 4가 아님: {tensor.shape}")
                if len(tensor.shape) == 3:
                    # 배치 차원 추가
                    tensor = tensor.unsqueeze(0)
                elif len(tensor.shape) == 2:
                    # 배치와 채널 차원 추가
                    tensor = tensor.unsqueeze(0).unsqueeze(0)
                else:
                    return torch.zeros((1, target_channels, 512, 512), device=tensor.device, dtype=tensor.dtype)
            
            # 🔥 채널 수 표준화
            if tensor.shape[1] == target_channels:
                return tensor
            elif tensor.shape[1] > target_channels:
                # 🔥 채널 수가 많으면 앞쪽 채널만 사용
                return tensor[:, :target_channels, :, :]
            else:
                # 🔥 채널 수가 적으면 패딩
                padding = torch.zeros(
                    tensor.shape[0], 
                    target_channels - tensor.shape[1], 
                    tensor.shape[2], 
                    tensor.shape[3],
                    device=tensor.device,
                    dtype=tensor.dtype
                )
                return torch.cat([tensor, padding], dim=1)
        except Exception as e:
            self.logger.warning(f"⚠️ 채널 수 표준화 실패: {e}")
            # 기본값 반환
            return torch.zeros((1, target_channels, 512, 512), device='cpu', dtype=torch.float32)
    
    def calculate_ensemble_uncertainty(self, ensemble_results: Dict[str, torch.Tensor]) -> float:
        """앙상블 불확실성 계산"""
        try:
            if not ensemble_results:
                return 0.5
            
            # 🔥 1. 각 모델의 예측을 텐서로 변환
            predictions = []
            for model_name, result in ensemble_results.items():
                try:
                    if isinstance(result, torch.Tensor):
                        pred = result
                    elif isinstance(result, dict):
                        pred = result.get('parsing_pred', result.get('output', None))
                    else:
                        continue
                    
                    if pred is not None and pred.dim() == 4:
                        # 소프트맥스 적용
                        pred_probs = F.softmax(pred, dim=1)
                        predictions.append(pred_probs)
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ {model_name} 예측 처리 실패: {e}")
                    continue
            
            if len(predictions) < 2:
                return 0.5
            
            # 🔥 2. 예측 평균 계산
            mean_prediction = torch.stack(predictions).mean(dim=0)
            
            # 🔥 3. 분산 계산 (불확실성 측정)
            variance = torch.stack(predictions).var(dim=0)
            
            # 🔥 4. 평균 분산을 불확실성으로 사용
            uncertainty = torch.mean(variance).item()
            
            # 🔥 5. 정규화 (0~1 범위)
            uncertainty = min(1.0, uncertainty * 10.0)  # 스케일링
            
            return uncertainty
            
        except Exception as e:
            self.logger.warning(f"⚠️ 앙상블 불확실성 계산 실패: {e}")
            return 0.5
    
    def calibrate_ensemble_confidence(self, model_confidences: Dict[str, float], uncertainty: float) -> float:
        """앙상블 신뢰도 보정"""
        if not model_confidences:
            return 0.0
        
        # 기본 신뢰도 (가중 평균) - 시퀀스 오류 방지
        try:
            # 값들이 숫자인지 확인하고 변환
            confidence_values = []
            for key, value in model_confidences.items():
                try:
                    if isinstance(value, (list, tuple)):
                        # 시퀀스인 경우 첫 번째 값 사용
                        if value:
                            confidence_values.append(float(value[0]))
                        else:
                            confidence_values.append(0.5)
                    elif isinstance(value, (int, float)):
                        confidence_values.append(float(value))
                    elif isinstance(value, np.ndarray):
                        # numpy 배열인 경우 첫 번째 값 사용
                        confidence_values.append(float(value.flatten()[0]))
                    else:
                        # 기타 타입은 0.5로 설정
                        confidence_values.append(0.5)
                except Exception as e:
                    self.logger.warning(f"⚠️ 신뢰도 값 변환 실패 ({key}): {e}")
                    confidence_values.append(0.5)
            
            if not confidence_values:
                return 0.5
            
            weights = np.array(confidence_values)
            base_confidence = np.average(weights, weights=weights)
            
        except Exception as e:
            self.logger.warning(f"⚠️ 신뢰도 보정 실패: {e}")
            # 폴백: 단순 평균
            base_confidence = 0.8
        
        # 불확실성에 따른 보정
        uncertainty_penalty = uncertainty * 0.5  # 불확실성 페널티
        calibrated_confidence = max(0.0, min(1.0, base_confidence - uncertainty_penalty))
        
        return calibrated_confidence
    
    def memory_efficient_resize(self, image, target_size):
        """메모리 효율적인 리사이즈"""
        try:
            if isinstance(image, torch.Tensor):
                # 텐서 리사이즈
                return F.interpolate(image.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
            elif isinstance(image, np.ndarray):
                # NumPy 배열 리사이즈
                return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
            else:
                return image
        except Exception as e:
            self.logger.warning(f"⚠️ 리사이즈 실패: {e}")
            return image
    
    def normalize_lighting(self, image):
        """조명 정규화"""
        try:
            if isinstance(image, np.ndarray):
                # 히스토그램 평활화
                if len(image.shape) == 3:
                    # 컬러 이미지
                    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
                    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                else:
                    # 그레이스케일 이미지
                    return cv2.equalizeHist(image)
            else:
                return image
        except Exception as e:
            self.logger.warning(f"⚠️ 조명 정규화 실패: {e}")
            return image
    
    def correct_colors(self, image):
        """색상 보정"""
        try:
            if isinstance(image, np.ndarray) and len(image.shape) == 3:
                # 색상 보정 (간단한 감마 보정)
                gamma = 1.1
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                return cv2.LUT(image, table)
            else:
                return image
        except Exception as e:
            self.logger.warning(f"⚠️ 색상 보정 실패: {e}")
            return image
