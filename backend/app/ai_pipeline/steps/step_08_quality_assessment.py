# backend/app/ai_pipeline/steps/step_08_quality_assessment.py
"""
🔥 MyCloset AI - 8단계: 품질 평가 (Quality Assessment) - ModelLoader 완전 연동 버전
✅ BaseStepMixin 완전 상속으로 logger 속성 누락 문제 해결
✅ ModelLoader 인터페이스 완벽 연동으로 AI 모델 간접 호출
✅ 순환참조 완전 해결 (한방향 참조)
✅ 기존 파일의 모든 세부 분석 기능 유지
✅ 89.8GB 체크포인트 자동 탐지 및 활용
✅ Pipeline Manager 100% 호환
✅ M3 Max 128GB 최적화
✅ conda 환경 최적화
✅ 모든 함수/클래스명 유지

처리 흐름:
🌐 API 요청 → 📋 PipelineManager → 🎯 QualityAssessmentStep 생성
↓
🔗 ModelLoader.create_step_interface() ← ModelLoader 담당
├─ StepModelInterface 생성
├─ Step별 모델 요청사항 등록
└─ 89.8GB 체크포인트 자동 탐지
↓
🚀 QualityAssessmentStep.initialize() ← Step + ModelLoader 협업
├─ AI 품질 평가 모델 로드 ← ModelLoader가 실제 로드
├─ 전문 분석기 초기화 ← Step 처리
└─ M3 Max 최적화 적용 ← Step 적용
↓
🧠 실제 AI 추론 process() ← Step 파일이 주도
├─ 이미지 전처리 ← Step 처리
├─ AI 모델 추론 (품질 평가) ← ModelLoader가 제공한 모델로 Step이 추론
├─ 8가지 품질 분석 ← Step 처리
└─ 종합 품질 점수 계산 ← Step 처리
↓
📤 결과 반환 ← Step이 최종 결과 생성
├─ 종합 품질 점수
├─ 세부 분석 결과
├─ 품질 등급 및 권장사항
└─ 시각화 이미지
"""

import os
import sys
import logging
import time
import asyncio
import threading
import json
import math
import gc
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from enum import Enum
from functools import lru_cache, wraps
import numpy as np
import base64
import io

# ==============================================
# 🔥 한방향 참조 구조 - 순환참조 해결
# ==============================================

# 1. BaseStepMixin 및 QualityAssessmentMixin 임포트 (핵심)
try:
    from .base_step_mixin import BaseStepMixin, QualityAssessmentMixin
    BASE_STEP_MIXIN_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ BaseStepMixin 임포트 성공")
except ImportError as e:
    BASE_STEP_MIXIN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ BaseStepMixin 임포트 실패: {e}")

# 2. ModelLoader 인터페이스 임포트 (핵심)
try:
    from ..utils.model_loader import ModelLoader, get_global_model_loader
    MODEL_LOADER_AVAILABLE = True
    logger.info("✅ ModelLoader 임포트 성공")
except ImportError as e:
    MODEL_LOADER_AVAILABLE = False
    logger.warning(f"⚠️ ModelLoader 임포트 실패: {e}")

# 3. 설정 및 코어 모듈 임포트
try:
    from ...core.config import MODEL_CONFIG
    from ...core.gpu_config import GPUConfig
    from ...core.m3_optimizer import M3MaxOptimizer
    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    logger.warning(f"⚠️ Core 모듈 임포트 실패: {e}")

# 4. 선택적 라이브러리 임포트
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import autocast
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
    import cv2
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from skimage import feature, measure, filters, exposure, segmentation
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ==============================================
# 🔥 MRO 안전한 폴백 클래스들 (import 실패 시)
# ==============================================

if not BASE_STEP_MIXIN_AVAILABLE:
    class BaseStepMixin:
        """MRO 안전한 폴백 BaseStepMixin"""
        def __init__(self, *args, **kwargs):
            super().__init__()  # MRO 안전
            self.logger = logging.getLogger(f"pipeline.{self.__class__.__name__}")
            self.step_name = getattr(self, 'step_name', 'quality_assessment')
            self.step_number = 8
            self.device = 'cpu'
            self.is_initialized = False
    
    class QualityAssessmentMixin(BaseStepMixin):
        """MRO 안전한 폴백 QualityAssessmentMixin"""
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.step_type = "quality_assessment"
            self.quality_threshold = 0.7

# ==============================================
# 🔥 품질 평가 데이터 구조들 (기존 유지)
# ==============================================

class QualityGrade(Enum):
    """품질 등급"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"

class AssessmentMode(Enum):
    """평가 모드"""
    COMPREHENSIVE = "comprehensive"
    FAST = "fast"
    DETAILED = "detailed"
    CUSTOM = "custom"

class QualityAspect(Enum):
    """품질 평가 영역"""
    SHARPNESS = "sharpness"
    COLOR = "color"
    FITTING = "fitting"
    REALISM = "realism"
    ARTIFACTS = "artifacts"
    ALIGNMENT = "alignment"
    LIGHTING = "lighting"
    TEXTURE = "texture"

@dataclass
class QualityMetrics:
    """품질 메트릭 데이터 구조"""
    overall_score: float = 0.0
    confidence: float = 0.0
    
    # 세부 점수들
    sharpness_score: float = 0.0
    color_score: float = 0.0
    fitting_score: float = 0.0
    realism_score: float = 0.0
    artifacts_score: float = 0.0
    alignment_score: float = 0.0
    lighting_score: float = 0.0
    texture_score: float = 0.0
    
    # 메타데이터
    processing_time: float = 0.0
    device_used: str = "cpu"
    model_version: str = "v1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)

# ==============================================
# 🔥 AI 모델 클래스들 (기존 파일에서 가져옴)
# ==============================================

if TORCH_AVAILABLE:
    class EnhancedPerceptualQualityModel(nn.Module):
        """향상된 지각적 품질 평가 모델 (ResNet 백본)"""
        
        def __init__(self):
            super().__init__()
            
            # ResNet 스타일 특징 추출기
            self.feature_extractor = nn.Sequential(
                # Block 1
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
                
                # Block 2
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # Block 3
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((8, 8))
            )
            
            # 다중 품질 예측기
            feature_dim = 256 * 8 * 8
            self.quality_predictors = nn.ModuleDict({
                'overall': nn.Sequential(
                    nn.Linear(feature_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 1),
                    nn.Sigmoid()
                ),
                'sharpness': nn.Sequential(
                    nn.Linear(feature_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                ),
                'artifacts': nn.Sequential(
                    nn.Linear(feature_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                ),
                'noise': nn.Sequential(
                    nn.Linear(feature_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                )
            })
        
        def forward(self, x):
            features = self.feature_extractor(x)
            features = features.view(features.size(0), -1)
            
            results = {}
            for name, predictor in self.quality_predictors.items():
                results[name] = predictor(features)
            
            return results

    class EnhancedAestheticQualityModel(nn.Module):
        """향상된 미적 품질 평가 모델 (다중 헤드)"""
        
        def __init__(self):
            super().__init__()
            
            # 백본 네트워크
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
                
                self._make_layer(64, 64, 2),
                self._make_layer(64, 128, 2, stride=2),
                self._make_layer(128, 256, 2, stride=2),
                self._make_layer(256, 512, 2, stride=2),
                
                nn.AdaptiveAvgPool2d((1, 1))
            )
            
            # 미적 점수 예측 헤드들
            self.aesthetic_heads = nn.ModuleDict({
                'composition': nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                ),
                'color_harmony': nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                ),
                'lighting': nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                ),
                'symmetry': nn.Sequential(
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                    nn.Sigmoid()
                ),
                'balance': nn.Sequential(
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                    nn.Sigmoid()
                )
            })
        
        def _make_layer(self, in_channels, out_channels, blocks, stride=1):
            """ResNet 스타일 레이어 생성"""
            layers = []
            layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            
            for _ in range(1, blocks):
                layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU())
            
            return nn.Sequential(*layers)
        
        def forward(self, x):
            features = self.backbone(x)
            features = features.view(features.size(0), -1)
            
            results = {}
            for name, head in self.aesthetic_heads.items():
                results[name] = head(features)
            
            return results

else:
    # PyTorch 없을 때 더미 클래스
    class EnhancedPerceptualQualityModel:
        def __init__(self):
            self.logger = logging.getLogger(__name__)
            self.logger.warning("PyTorch 없음 - 더미 EnhancedPerceptualQualityModel")
        
        def predict(self, x):
            return {'overall': 0.7, 'sharpness': 0.8, 'artifacts': 0.9, 'noise': 0.8}
    
    class EnhancedAestheticQualityModel:
        def __init__(self):
            self.logger = logging.getLogger(__name__)
            self.logger.warning("PyTorch 없음 - 더미 EnhancedAestheticQualityModel")
        
        def predict(self, x):
            return {'composition': 0.7, 'color_harmony': 0.8, 'lighting': 0.75, 'symmetry': 0.7, 'balance': 0.8}

# ==============================================
# 🔥 전문 분석기 클래스들 (기존 파일에서 완전 이식)
# ==============================================

class TechnicalQualityAnalyzer:
    """기술적 품질 분석기 (향상된 버전)"""
    
    def __init__(self, device: str = "cpu", enable_gpu: bool = False):
        self.device = device
        self.enable_gpu = enable_gpu
        self.logger = logging.getLogger(f"{__name__}.TechnicalQualityAnalyzer")
        
        # 분석 캐시
        self.analysis_cache = {}
        
        # 기술적 분석 임계값들
        self.thresholds = {
            'sharpness_min': 100.0,
            'noise_max': 50.0,
            'contrast_min': 20.0,
            'brightness_range': (50, 200)
        }
    
    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """종합 기술적 품질 분석"""
        try:
            results = {}
            
            # 1. 선명도 분석
            results['sharpness'] = self._analyze_sharpness(image)
            
            # 2. 노이즈 레벨 분석
            results['noise_level'] = self._analyze_noise_level(image)
            
            # 3. 대비 분석
            results['contrast'] = self._analyze_contrast(image)
            
            # 4. 밝기 분석
            results['brightness'] = self._analyze_brightness(image)
            
            # 5. 포화도 분석
            results['saturation'] = self._analyze_saturation(image)
            
            # 6. 아티팩트 검출
            results['artifacts'] = self._detect_artifacts(image)
            
            # 7. 종합 점수 계산
            results['overall_score'] = self._calculate_technical_score(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 기술적 분석 실패: {e}")
            return self._get_fallback_technical_results()
    
    def _analyze_sharpness(self, image: np.ndarray) -> float:
        """선명도 분석"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if 'cv2' in globals() else np.mean(image, axis=2)
            else:
                gray = image
            
            if 'cv2' in globals():
                laplacian = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
                sharpness = laplacian.var()
            else:
                # 간단한 gradient 기반 선명도
                dx = np.diff(gray, axis=1)
                dy = np.diff(gray, axis=0)
                sharpness = np.var(dx) + np.var(dy)
            
            # 정규화 (0-1)
            normalized_sharpness = min(1.0, sharpness / 10000.0)
            return max(0.0, normalized_sharpness)
            
        except Exception as e:
            self.logger.error(f"선명도 분석 실패: {e}")
            return 0.5
    
    def _analyze_noise_level(self, image: np.ndarray) -> float:
        """노이즈 레벨 분석"""
        try:
            if len(image.shape) == 3:
                # 각 채널별 노이즈 분석
                noise_levels = []
                for channel in range(3):
                    channel_data = image[:, :, channel]
                    # 고주파 성분 분석
                    if 'cv2' in globals():
                        blur = cv2.GaussianBlur(channel_data.astype(np.uint8), (5, 5), 0)
                        noise = np.abs(channel_data.astype(float) - blur.astype(float))
                    else:
                        # 간단한 이동평균 기반
                        kernel = np.ones((3, 3)) / 9
                        blur = np.convolve(channel_data.flatten(), kernel.flatten(), mode='same').reshape(channel_data.shape)
                        noise = np.abs(channel_data.astype(float) - blur)
                    
                    noise_level = np.mean(noise) / 255.0
                    noise_levels.append(noise_level)
                
                # 평균 노이즈 레벨
                avg_noise = np.mean(noise_levels)
            else:
                # 그레이스케일
                if 'cv2' in globals():
                    blur = cv2.GaussianBlur(image.astype(np.uint8), (5, 5), 0)
                    noise = np.abs(image.astype(float) - blur.astype(float))
                else:
                    noise = np.std(image) / 255.0
                avg_noise = np.mean(noise) / 255.0 if 'cv2' in globals() else noise
            
            # 노이즈가 적을수록 품질이 좋음 (역순)
            return max(0.0, min(1.0, 1.0 - avg_noise * 5))
            
        except Exception as e:
            self.logger.error(f"노이즈 분석 실패: {e}")
            return 0.7
    
    def _analyze_contrast(self, image: np.ndarray) -> float:
        """대비 분석"""
        try:
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image
            
            # RMS 대비 계산
            contrast = np.std(gray)
            
            # 정규화 (적절한 대비 범위: 30-80)
            if 30 <= contrast <= 80:
                contrast_score = 1.0
            elif contrast < 30:
                contrast_score = contrast / 30.0
            else:
                contrast_score = max(0.3, 1.0 - (contrast - 80) / 100.0)
            
            return max(0.0, min(1.0, contrast_score))
            
        except Exception as e:
            self.logger.error(f"대비 분석 실패: {e}")
            return 0.6
    
    def _analyze_brightness(self, image: np.ndarray) -> float:
        """밝기 분석"""
        try:
            if len(image.shape) == 3:
                brightness = np.mean(image)
            else:
                brightness = np.mean(image)
            
            # 적절한 밝기 범위 (100-160)
            if 100 <= brightness <= 160:
                brightness_score = 1.0
            elif brightness < 100:
                brightness_score = brightness / 100.0
            else:
                brightness_score = max(0.3, 1.0 - (brightness - 160) / 95.0)
            
            return max(0.0, min(1.0, brightness_score))
            
        except Exception as e:
            self.logger.error(f"밝기 분석 실패: {e}")
            return 0.6
    
    def _analyze_saturation(self, image: np.ndarray) -> float:
        """포화도 분석"""
        try:
            if len(image.shape) != 3:
                return 0.5
            
            # HSV 변환 및 포화도 분석
            if 'cv2' in globals():
                hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
                saturation = np.mean(hsv[:, :, 1])
            else:
                # RGB 기반 포화도 근사
                max_vals = np.max(image, axis=2)
                min_vals = np.min(image, axis=2)
                saturation = np.mean((max_vals - min_vals) / (max_vals + 1e-8)) * 255
            
            # 적절한 포화도 범위 (80-180)
            if 80 <= saturation <= 180:
                saturation_score = 1.0
            elif saturation < 80:
                saturation_score = saturation / 80.0
            else:
                saturation_score = max(0.3, 1.0 - (saturation - 180) / 75.0)
            
            return max(0.0, min(1.0, saturation_score))
            
        except Exception as e:
            self.logger.error(f"포화도 분석 실패: {e}")
            return 0.6
    
    def _detect_artifacts(self, image: np.ndarray) -> float:
        """아티팩트 검출"""
        try:
            artifact_score = 1.0  # 기본값: 아티팩트 없음
            
            # 1. 블로킹 아티팩트 검출
            blocking_score = self._detect_blocking_artifacts(image)
            
            # 2. 링잉 아티팩트 검출
            ringing_score = self._detect_ringing_artifacts(image)
            
            # 3. 압축 아티팩트 검출
            compression_score = self._detect_compression_artifacts(image)
            
            # 종합 아티팩트 점수 (높을수록 아티팩트 적음)
            artifact_score = np.mean([blocking_score, ringing_score, compression_score])
            
            return max(0.0, min(1.0, artifact_score))
            
        except Exception as e:
            self.logger.error(f"아티팩트 검출 실패: {e}")
            return 0.8
    
    def _detect_blocking_artifacts(self, image: np.ndarray) -> float:
        """블로킹 아티팩트 검출"""
        try:
            # 8x8 블록 경계에서의 불연속성 검사
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image
            
            h, w = gray.shape
            blocking_score = 1.0
            
            # 수직/수평 블록 경계 검사
            for i in range(8, h-8, 8):
                diff = np.mean(np.abs(gray[i, :] - gray[i-1, :]))
                if diff > 10:  # 임계값
                    blocking_score -= 0.1
            
            for j in range(8, w-8, 8):
                diff = np.mean(np.abs(gray[:, j] - gray[:, j-1]))
                if diff > 10:  # 임계값
                    blocking_score -= 0.1
            
            return max(0.0, blocking_score)
            
        except Exception as e:
            return 0.9
    
    def _detect_ringing_artifacts(self, image: np.ndarray) -> float:
        """링잉 아티팩트 검출"""
        try:
            # 에지 근처의 진동 패턴 검출
            if 'cv2' in globals() and len(image.shape) == 3:
                gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                
                # 에지 근처의 고주파 성분 분석
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                enhanced = cv2.filter2D(gray, -1, kernel)
                
                # 링잉 정도 계산
                ringing_metric = np.mean(np.abs(enhanced - gray))
                ringing_score = max(0.0, 1.0 - ringing_metric / 50.0)
            else:
                ringing_score = 0.9  # 기본값
            
            return ringing_score
            
        except Exception as e:
            return 0.9
    
    def _detect_compression_artifacts(self, image: np.ndarray) -> float:
        """압축 아티팩트 검출"""
        try:
            # DCT 계수 분석 기반 압축 아티팩트 검출
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image
            
            # 고주파 성분 손실 정도 분석
            if 'cv2' in globals():
                # 라플라시안 필터로 고주파 성분 추출
                laplacian = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
                high_freq_energy = np.var(laplacian)
                
                # 압축되지 않은 이미지의 예상 고주파 에너지와 비교
                expected_energy = np.var(gray) * 0.1  # 추정값
                compression_score = min(1.0, high_freq_energy / (expected_energy + 1e-8))
            else:
                compression_score = 0.8  # 기본값
            
            return max(0.0, compression_score)
            
        except Exception as e:
            return 0.8
    
    def _calculate_technical_score(self, results: Dict[str, Any]) -> float:
        """기술적 품질 종합 점수 계산"""
        try:
            # 가중치 설정
            weights = {
                'sharpness': 0.25,
                'noise_level': 0.20,
                'contrast': 0.15,
                'brightness': 0.15,
                'saturation': 0.10,
                'artifacts': 0.15
            }
            
            # 가중 평균 계산
            total_score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in results:
                    total_score += results[metric] * weight
                    total_weight += weight
            
            # 정규화
            if total_weight > 0:
                final_score = total_score / total_weight
            else:
                final_score = 0.5
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            self.logger.error(f"기술적 점수 계산 실패: {e}")
            return 0.5
    
    def _get_fallback_technical_results(self) -> Dict[str, Any]:
        """폴백 기술적 분석 결과"""
        return {
            'sharpness': 0.5,
            'noise_level': 0.6,
            'contrast': 0.5,
            'brightness': 0.6,
            'saturation': 0.5,
            'artifacts': 0.7,
            'overall_score': 0.55,
            'analysis_method': 'fallback'
        }
    
    def cleanup(self):
        """분석기 정리"""
        self.analysis_cache.clear()

class FittingQualityAnalyzer:
    """기능적 품질 분석기 (피팅 전문)"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.logger = logging.getLogger(f"{__name__}.FittingQualityAnalyzer")
        
        # 의류 타입별 기대 비율
        self.clothing_ratios = {
            'shirt': {'aspect_ratio': (0.7, 1.3), 'coverage': (0.3, 0.7)},
            'dress': {'aspect_ratio': (0.5, 1.0), 'coverage': (0.5, 0.9)},
            'pants': {'aspect_ratio': (0.8, 1.2), 'coverage': (0.4, 0.8)},
            'jacket': {'aspect_ratio': (0.6, 1.2), 'coverage': (0.4, 0.8)},
            'skirt': {'aspect_ratio': (0.8, 1.4), 'coverage': (0.2, 0.6)},
            'top': {'aspect_ratio': (0.7, 1.4), 'coverage': (0.2, 0.6)},
            'default': {'aspect_ratio': (0.5, 1.5), 'coverage': (0.2, 0.9)}
        }
    
    def analyze(self, image: np.ndarray, clothing_type: str = "default") -> Dict[str, Any]:
        """기능적 품질 분석 실행"""
        try:
            results = {}
            
            # 1. 피팅 정확도 분석
            results['fitting_accuracy'] = self._analyze_fitting_accuracy(image, clothing_type)
            
            # 2. 의류 정렬 분석
            results['clothing_alignment'] = self._analyze_clothing_alignment(image)
            
            # 3. 자연스러움 분석
            results['naturalness'] = self._analyze_naturalness(image)
            
            # 4. 에지 보존도 분석
            results['edge_preservation'] = self._analyze_edge_preservation(image)
            
            # 5. 텍스처 품질 분석
            results['texture_quality'] = self._analyze_texture_quality(image)
            
            # 6. 세부사항 보존도 분석
            results['detail_preservation'] = self._analyze_detail_preservation(image)
            
            # 7. 종합 점수 계산
            results['overall_score'] = self._calculate_fitting_score(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 기능적 분석 실패: {e}")
            return self._get_fallback_fitting_results()
    
    def _analyze_fitting_accuracy(self, image: np.ndarray, clothing_type: str) -> float:
        """피팅 정확도 분석"""
        try:
            height, width = image.shape[:2]
            aspect_ratio = width / height
            
            # 의류 타입별 기대 비율
            expected = self.clothing_ratios.get(clothing_type, self.clothing_ratios['default'])
            min_ratio, max_ratio = expected['aspect_ratio']
            
            # 비율 점수 계산
            if min_ratio <= aspect_ratio <= max_ratio:
                ratio_score = 1.0
            else:
                center_ratio = (min_ratio + max_ratio) / 2
                ratio_score = max(0.0, 1.0 - abs(aspect_ratio - center_ratio) * 2)
            
            # 커버리지 분석 (의류가 차지하는 영역)
            coverage_score = self._analyze_clothing_coverage(image, expected['coverage'])
            
            # 종합 피팅 정확도
            fitting_accuracy = (ratio_score * 0.6 + coverage_score * 0.4)
            
            return max(0.0, min(1.0, fitting_accuracy))
            
        except Exception as e:
            self.logger.error(f"피팅 정확도 분석 실패: {e}")
            return 0.5
    
    def _analyze_clothing_coverage(self, image: np.ndarray, expected_coverage: Tuple[float, float]) -> float:
        """의류 커버리지 분석"""
        try:
            # 의류 영역 추정 (색상 기반 간단한 세그멘테이션)
            if len(image.shape) == 3:
                # 배경과 전경 구분 (간단한 임계값 기반)
                gray = np.mean(image, axis=2)
                
                # Otsu's 방법 근사
                if 'cv2' in globals():
                    _, binary = cv2.threshold(gray.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    clothing_pixels = np.sum(binary > 0)
                else:
                    # 간단한 임계값
                    threshold = np.mean(gray)
                    clothing_pixels = np.sum(gray > threshold)
                
                total_pixels = gray.size
                coverage_ratio = clothing_pixels / total_pixels
            else:
                # 그레이스케일의 경우
                threshold = np.mean(image)
                coverage_ratio = np.sum(image > threshold) / image.size
            
            # 기대 커버리지와 비교
            min_coverage, max_coverage = expected_coverage
            
            if min_coverage <= coverage_ratio <= max_coverage:
                coverage_score = 1.0
            else:
                center_coverage = (min_coverage + max_coverage) / 2
                coverage_score = max(0.0, 1.0 - abs(coverage_ratio - center_coverage) * 3)
            
            return coverage_score
            
        except Exception as e:
            self.logger.error(f"커버리지 분석 실패: {e}")
            return 0.6
    
    def _analyze_clothing_alignment(self, image: np.ndarray) -> float:
        """의류 정렬 분석"""
        try:
            height, width = image.shape[:2]
            
            # 좌우 대칭성 분석
            left_half = image[:, :width//2]
            right_half = image[:, width//2:]
            right_half_flipped = np.flip(right_half, axis=1)
            
            # 크기 맞추기
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_half_flipped = right_half_flipped[:, :min_width]
            
            # 차이 계산
            diff = np.mean(np.abs(left_half.astype(float) - right_half_flipped.astype(float)))
            symmetry_score = max(0.0, 1.0 - diff / 128.0)  # 정규화
            
            # 수직 정렬 분석 (중심선 기준)
            center_col = width // 2
            center_line = image[:, max(0, center_col-2):min(width, center_col+3)]
            
            # 중심선의 일관성 분석
            if center_line.size > 0:
                vertical_consistency = 1.0 - np.std(np.mean(center_line, axis=1)) / 255.0
                vertical_consistency = max(0.0, min(1.0, vertical_consistency))
            else:
                vertical_consistency = 0.5
            
            # 종합 정렬 점수
            alignment_score = (symmetry_score * 0.7 + vertical_consistency * 0.3)
            
            return max(0.0, min(1.0, alignment_score))
            
        except Exception as e:
            self.logger.error(f"의류 정렬 분석 실패: {e}")
            return 0.5
    
    def _analyze_naturalness(self, image: np.ndarray) -> float:
        """자연스러움 분석"""
        try:
            # 색상 분포의 자연스러움 체크
            if len(image.shape) == 3:
                # RGB 채널별 히스토그램 분석
                color_variance = np.mean([np.var(image[:,:,i]) for i in range(3)])
                color_naturalness = min(1.0, color_variance / (255.0 * 255.0) * 10)
            else:
                color_naturalness = 0.7
            
            # 경계선의 부드러움 체크
            if 'cv2' in globals():
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
                edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                smoothness = max(0.0, 1.0 - edge_density * 5)
            else:
                smoothness = 0.7
            
            naturalness = (color_naturalness + smoothness) / 2
            return max(0.0, min(1.0, naturalness))
        
        except Exception as e:
            self.logger.error(f"자연스러움 분석 실패: {e}")
            return 0.5
    
    def _analyze_edge_preservation(self, image: np.ndarray) -> float:
        """에지 보존도 분석"""
        try:
            # 에지 검출 및 품질 평가
            if 'cv2' in globals():
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                else:
                    gray = image.astype(np.uint8)
                
                # 다중 스케일 에지 검출
                edges_fine = cv2.Canny(gray, 50, 150)
                edges_coarse = cv2.Canny(gray, 100, 200)
                
                # 에지의 선명도 및 연속성 평가
                edge_strength = np.mean(edges_fine)
                edge_consistency = 1.0 - abs(np.mean(edges_fine) - np.mean(edges_coarse)) / 255.0
                
                preservation_score = (edge_strength / 255.0) * 0.6 + edge_consistency * 0.4
            else:
                # 간단한 gradient 기반
                if len(image.shape) == 3:
                    gray = np.mean(image, axis=2)
                else:
                    gray = image
                
                dx = np.abs(np.diff(gray, axis=1))
                dy = np.abs(np.diff(gray, axis=0))
                edge_strength = (np.mean(dx) + np.mean(dy)) / 2
                
                preservation_score = min(1.0, edge_strength / 30.0)
            
            return max(0.0, min(1.0, preservation_score))
            
        except Exception as e:
            self.logger.error(f"에지 보존 분석 실패: {e}")
            return 0.5
    
    def _analyze_texture_quality(self, image: np.ndarray) -> float:
        """텍스처 품질 분석"""
        try:
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image
            
            # 지역 바이너리 패턴 (LBP) 근사
            texture_features = []
            
            # 다양한 방향의 텍스처 분석
            for dx, dy in [(1, 0), (0, 1), (1, 1), (-1, 1)]:
                if dx == 1 and dy == 0:  # 수평
                    diff = np.abs(np.diff(gray, axis=1))
                elif dx == 0 and dy == 1:  # 수직
                    diff = np.abs(np.diff(gray, axis=0))
                else:  # 대각선 (간단 근사)
                    diff = np.abs(gray[1:, 1:] - gray[:-1, :-1])
                
                texture_strength = np.mean(diff)
                texture_features.append(texture_strength)
            
            # 텍스처 복잡도
            texture_complexity = np.mean(texture_features)
            
            # 적절한 텍스처 복잡도 (5-25)
            if 5 <= texture_complexity <= 25:
                texture_score = 1.0
            else:
                texture_score = max(0.0, 1.0 - abs(texture_complexity - 15) / 20.0)
            
            return max(0.0, min(1.0, texture_score))
            
        except Exception as e:
            self.logger.error(f"텍스처 품질 분석 실패: {e}")
            return 0.5
    
    def _analyze_detail_preservation(self, image: np.ndarray) -> float:
        """세부사항 보존도 분석"""
        try:
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image
            
            # 고주파 성분 분석
            if 'cv2' in globals():
                # 라플라시안 필터로 세부사항 추출
                laplacian = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
                detail_strength = np.var(laplacian)
                
                # 가우시안 피라미드로 다중 스케일 분석
                pyramid_levels = []
                current = gray.astype(np.uint8)
                for _ in range(3):
                    current = cv2.pyrDown(current)
                    pyramid_levels.append(np.var(current))
                
                detail_preservation = detail_strength / (1 + np.mean(pyramid_levels))
                preservation_score = min(1.0, detail_preservation / 1000.0)
            else:
                # 간단한 분산 기반 세부사항 분석
                detail_variance = np.var(gray)
                preservation_score = min(1.0, detail_variance / 2000.0)
            
            return max(0.0, min(1.0, preservation_score))
            
        except Exception as e:
            self.logger.error(f"세부사항 보존 분석 실패: {e}")
            return 0.5
    
    def _calculate_fitting_score(self, results: Dict[str, Any]) -> float:
        """기능적 품질 종합 점수 계산"""
        try:
            # 가중치 설정
            weights = {
                'fitting_accuracy': 0.25,
                'clothing_alignment': 0.20,
                'naturalness': 0.20,
                'edge_preservation': 0.15,
                'texture_quality': 0.10,
                'detail_preservation': 0.10
            }
            
            # 가중 평균 계산
            total_score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in results:
                    total_score += results[metric] * weight
                    total_weight += weight
            
            # 정규화
            if total_weight > 0:
                final_score = total_score / total_weight
            else:
                final_score = 0.5
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            self.logger.error(f"기능적 점수 계산 실패: {e}")
            return 0.5
    
    def _get_fallback_fitting_results(self) -> Dict[str, Any]:
        """폴백 기능적 분석 결과"""
        return {
            'fitting_accuracy': 0.6,
            'clothing_alignment': 0.6,
            'naturalness': 0.5,
            'edge_preservation': 0.6,
            'texture_quality': 0.5,
            'detail_preservation': 0.6,
            'overall_score': 0.57,
            'analysis_method': 'fallback'
        }
    
    def cleanup(self):
        """분석기 정리"""
        pass

class ColorQualityAnalyzer:
    """색상 품질 전문 분석기"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.logger = logging.getLogger(f"{__name__}.ColorQualityAnalyzer")
    
    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """색상 품질 분석 실행"""
        try:
            results = {}
            
            # 1. 색상 일관성 분석
            results['color_consistency'] = self._analyze_color_consistency(image)
            
            # 2. 색상 자연스러움 분석
            results['color_naturalness'] = self._analyze_color_naturalness(image)
            
            # 3. 색상 대비 분석
            results['color_contrast'] = self._analyze_color_contrast(image)
            
            # 4. 색상 조화 분석
            results['color_harmony'] = self._analyze_color_harmony(image)
            
            # 5. 종합 점수 계산
            results['overall_score'] = self._calculate_color_score(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 색상 분석 실패: {e}")
            return self._get_fallback_color_results()
    
    def _analyze_color_consistency(self, image: np.ndarray) -> float:
        """색상 일관성 분석"""
        try:
            if len(image.shape) != 3:
                return 0.5
            
            # 영역별 색상 분포 분석
            height, width = image.shape[:2]
            regions = [
                image[:height//2, :width//2],    # 좌상
                image[:height//2, width//2:],    # 우상
                image[height//2:, :width//2],    # 좌하
                image[height//2:, width//2:]     # 우하
            ]
            
            # 각 영역의 평균 색상
            region_colors = [np.mean(region, axis=(0,1)) for region in regions]
            
            # 색상 간 편차 계산
            color_std = np.std(region_colors, axis=0)
            consistency = 1.0 - np.mean(color_std) / 255.0
            
            return max(0.0, min(1.0, consistency))
        
        except Exception as e:
            self.logger.error(f"색상 일관성 분석 실패: {e}")
            return 0.5
    
    def _analyze_color_naturalness(self, image: np.ndarray) -> float:
        """색상 자연스러움 분석"""
        try:
            if len(image.shape) != 3:
                return 0.5
            
            # 색상 포화도 분석
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) if 'cv2' in globals() else image
            
            if 'cv2' in globals():
                saturation = hsv[:,:,1]
                avg_saturation = np.mean(saturation) / 255.0
                
                # 적절한 포화도 범위 (0.3-0.8)
                if 0.3 <= avg_saturation <= 0.8:
                    saturation_score = 1.0
                else:
                    saturation_score = max(0.0, 1.0 - abs(avg_saturation - 0.55) * 2)
            else:
                # OpenCV 없을 때 RGB 기반 근사
                max_vals = np.max(image, axis=2)
                min_vals = np.min(image, axis=2)
                saturation = (max_vals - min_vals) / (max_vals + 1e-8)
                avg_saturation = np.mean(saturation)
                saturation_score = min(1.0, avg_saturation * 1.5)
            
            return saturation_score
        
        except Exception as e:
            self.logger.error(f"색상 자연스러움 분석 실패: {e}")
            return 0.5
    
    def _analyze_color_contrast(self, image: np.ndarray) -> float:
        """색상 대비 분석"""
        try:
            if len(image.shape) == 3:
                # 밝기 채널 추출
                brightness = np.mean(image, axis=2)
            else:
                brightness = image
            
            # 대비 계산 (표준편차 기반)
            contrast = np.std(brightness) / 255.0
            
            # 적절한 대비 범위 (0.2-0.6)
            if 0.2 <= contrast <= 0.6:
                contrast_score = 1.0
            else:
                contrast_score = max(0.0, 1.0 - abs(contrast - 0.4) * 2.5)
            
            return contrast_score
        
        except Exception as e:
            self.logger.error(f"색상 대비 분석 실패: {e}")
            return 0.5
    
    def _analyze_color_harmony(self, image: np.ndarray) -> float:
        """색상 조화 분석"""
        try:
            if len(image.shape) != 3:
                return 0.5
            
            # 주요 색상 추출
            if SKLEARN_AVAILABLE:
                pixels = image.reshape(-1, 3)
                
                # 샘플링 (성능 최적화)
                if len(pixels) > 10000:
                    indices = np.random.choice(len(pixels), 10000, replace=False)
                    pixels = pixels[indices]
                
                # K-means로 주요 색상 추출
                try:
                    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                    kmeans.fit(pixels)
                    
                    centers = kmeans.cluster_centers_
                    
                    # 색상 간 거리 분석
                    color_distances = []
                    for i in range(len(centers)):
                        for j in range(i+1, len(centers)):
                            distance = np.linalg.norm(centers[i] - centers[j])
                            color_distances.append(distance)
                    
                    # 적절한 색상 다양성
                    avg_distance = np.mean(color_distances)
                    harmony_score = min(1.0, avg_distance / 100.0)
                    
                except Exception:
                    harmony_score = 0.6
            else:
                # 간단한 색상 분산 기반
                color_std = np.std(image, axis=(0,1))
                harmony_score = min(1.0, np.mean(color_std) / 64.0)
            
            return max(0.0, min(1.0, harmony_score))
        
        except Exception as e:
            self.logger.error(f"색상 조화 분석 실패: {e}")
            return 0.5
    
    def _calculate_color_score(self, results: Dict[str, Any]) -> float:
        """색상 품질 종합 점수 계산"""
        try:
            # 가중치 설정
            weights = {
                'color_consistency': 0.3,
                'color_naturalness': 0.3,
                'color_contrast': 0.2,
                'color_harmony': 0.2
            }
            
            # 가중 평균 계산
            total_score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in results:
                    total_score += results[metric] * weight
                    total_weight += weight
            
            # 정규화
            if total_weight > 0:
                final_score = total_score / total_weight
            else:
                final_score = 0.5
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            self.logger.error(f"색상 점수 계산 실패: {e}")
            return 0.5
    
    def _get_fallback_color_results(self) -> Dict[str, Any]:
        """폴백 색상 분석 결과"""
        return {
            'color_consistency': 0.6,
            'color_naturalness': 0.6,
            'color_contrast': 0.5,
            'color_harmony': 0.6,
            'overall_score': 0.58,
            'analysis_method': 'fallback'
        }
    
    def cleanup(self):
        """분석기 정리"""
        pass

# ==============================================
# 🔥 메인 QualityAssessmentStep 클래스
# ==============================================

class QualityAssessmentStep(QualityAssessmentMixin):
    """
    🔥 8단계: 품질 평가 Step - ModelLoader 연동 완전 버전
    ✅ QualityAssessmentMixin 상속으로 logger 보장
    ✅ ModelLoader 인터페이스 통한 AI 모델 호출 (핵심)
    ✅ 기존 모든 분석 기능 유지
    ✅ 89.8GB 체크포인트 자동 활용
    ✅ M3 Max 최적화
    """
    
    def __init__(
        self,
        device: str = "auto",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """품질 평가 Step 초기화"""
        
        # 🔥 MRO 안전한 초기화
        super().__init__()
        
        # 🔥 Step 기본 정보 설정
        self.step_name = 'quality_assessment'
        self.step_number = 8
        self.step_type = "quality_assessment"
        
        # 🔥 디바이스 설정
        self.device = self._determine_device(device)
        self.device_type = self._get_device_type()
        
        # 🔥 시스템 정보
        self.memory_gb = self._get_system_memory()
        self.is_m3_max = self._detect_m3_max()
        
        # 🔥 설정 초기화
        self.config = config or {}
        self.quality_threshold = self.config.get('quality_threshold', 0.7)
        self.assessment_mode = AssessmentMode(self.config.get('mode', 'comprehensive'))
        
        # 🔥 품질 평가 설정
        self.assessment_config = {
            'mode': self.assessment_mode,
            'quality_threshold': self.quality_threshold,
            'enable_detailed_analysis': self.config.get('detailed_analysis', True),
            'enable_visualization': self.config.get('visualization', True),
            'enable_ai_models': self.config.get('enable_ai_models', True)
        }
        
        # 🔥 ModelLoader 인터페이스 (핵심)
        self.model_interface = None
        self.models_loaded = {}
        
        # 🔥 품질 평가 파이프라인
        self.assessment_pipeline = []
        
        # 🔥 분석기들 (기존 유지)
        self.technical_analyzer = None
        self.perceptual_analyzer = None
        self.aesthetic_analyzer = None
        self.functional_analyzer = None
        self.color_analyzer = None
        
        # 🔥 상태 관리
        self.is_initialized = False
        self.initialization_error = None
        self.error_count = 0
        self.last_error = None
        
        # 🔥 성능 최적화
        self.optimization_enabled = self.is_m3_max and self.memory_gb >= 64
        
        self.logger.info(f"✅ {self.step_name} 초기화 완료 - Device: {self.device}, Memory: {self.memory_gb}GB")
    
    # ==============================================
    # 🔥 초기화 및 ModelLoader 연동 메서드들
    # ==============================================
    
    async def initialize(self) -> bool:
        """품질 평가 Step 초기화"""
        try:
            self.logger.info(f"🚀 {self.step_name} 초기화 시작...")
            
            # 1. ModelLoader 인터페이스 설정 (핵심)
            await self._setup_model_interface()
            
            # 2. AI 모델 로드 (ModelLoader 통해)
            await self._load_quality_assessment_models()
            
            # 3. 품질 평가 파이프라인 구성
            self._setup_assessment_pipeline()
            
            # 4. 전문 분석기 초기화
            self._initialize_analyzers()
            
            # 5. M3 Max 최적화 설정
            if self.optimization_enabled:
                self._optimize_for_m3_max()
            
            self.is_initialized = True
            self.logger.info(f"✅ {self.step_name} 초기화 완료")
            return True
            
        except Exception as e:
            self.initialization_error = str(e)
            self.logger.error(f"❌ {self.step_name} 초기화 실패: {e}")
            return False
    
    async def _setup_model_interface(self):
        """ModelLoader 인터페이스 설정 (핵심)"""
        try:
            if MODEL_LOADER_AVAILABLE:
                # 글로벌 ModelLoader 가져오기
                model_loader = get_global_model_loader()
                if model_loader:
                    # Step별 인터페이스 생성
                    self.model_interface = model_loader.create_step_interface(self.step_name)
                    
                    # 모델 요청사항 등록
                    self._register_model_requirements()
                    
                    self.logger.info(f"✅ ModelLoader 인터페이스 설정 완료: {self.step_name}")
                    return
            
            # 폴백: 직접 모델 로드 준비
            self.logger.warning("ModelLoader 미사용, 직접 로드 모드")
            self.model_interface = None
            
        except Exception as e:
            self.logger.error(f"❌ ModelLoader 인터페이스 설정 실패: {e}")
            self.model_interface = None
    
    def _register_model_requirements(self):
        """품질 평가용 모델 요청사항 등록"""
        try:
            if self.model_interface:
                # 지각적 품질 평가 모델
                self.model_interface.register_model_requirement(
                    model_name="perceptual_quality_model",
                    model_type="quality_assessment",
                    priority="high",
                    fallback_models=["lpips_model", "ssim_model"]
                )
                
                # 기술적 품질 분석 모델
                self.model_interface.register_model_requirement(
                    model_name="technical_quality_model", 
                    model_type="image_analysis",
                    priority="medium",
                    fallback_models=["opencv_analysis"]
                )
                
                # 미적 품질 평가 모델
                self.model_interface.register_model_requirement(
                    model_name="aesthetic_quality_model",
                    model_type="aesthetic_analysis", 
                    priority="medium",
                    fallback_models=["traditional_metrics"]
                )
                
                self.logger.info(f"✅ 모델 요청사항 등록 완료: {self.step_name}")
        
        except Exception as e:
            self.logger.error(f"❌ 모델 요청사항 등록 실패: {e}")
    
    async def _load_quality_assessment_models(self):
        """품질 평가 AI 모델들 로드 (ModelLoader 통해)"""
        try:
            if self.model_interface:
                # 지각적 품질 모델 로드
                perceptual_model = await self.model_interface.get_model("perceptual_quality_model")
                if perceptual_model:
                    self.models_loaded['perceptual'] = perceptual_model
                    self.logger.info("✅ 지각적 품질 모델 로드 성공")
                
                # 기술적 품질 모델 로드  
                technical_model = await self.model_interface.get_model("technical_quality_model")
                if technical_model:
                    self.models_loaded['technical'] = technical_model
                    self.logger.info("✅ 기술적 품질 모델 로드 성공")
                
                # 미적 품질 모델 로드
                aesthetic_model = await self.model_interface.get_model("aesthetic_quality_model")
                if aesthetic_model:
                    self.models_loaded['aesthetic'] = aesthetic_model
                    self.logger.info("✅ 미적 품질 모델 로드 성공")
                
                self.logger.info(f"✅ {len(self.models_loaded)}개 AI 모델 로드 완료")
            else:
                # 폴백: 전통적 분석 방법 사용
                self.logger.warning("AI 모델 없음, 전통적 분석 방법 사용")
                await self._setup_traditional_analysis()
        
        except Exception as e:
            self.logger.error(f"❌ 품질 평가 모델 로드 실패: {e}")
            await self._setup_traditional_analysis()
    
    async def _setup_traditional_analysis(self):
        """전통적 분석 방법 설정 (AI 모델 없을 때)"""
        try:
            # OpenCV 기반 기술적 분석
            self.models_loaded['opencv_technical'] = True
            
            # PIL 기반 미적 분석  
            self.models_loaded['pil_aesthetic'] = True
            
            # NumPy 기반 지각적 분석
            self.models_loaded['numpy_perceptual'] = True
            
            self.logger.info("✅ 전통적 분석 방법 설정 완료")
        
        except Exception as e:
            self.logger.error(f"❌ 전통적 분석 설정 실패: {e}")
    
    def _setup_assessment_pipeline(self):
        """품질 평가 파이프라인 구성"""
        try:
            self.assessment_pipeline = [
                ("기술적_품질_분석", self._analyze_technical_quality),
                ("지각적_품질_분석", self._analyze_perceptual_quality), 
                ("미적_품질_분석", self._analyze_aesthetic_quality),
                ("기능적_품질_분석", self._analyze_functional_quality),
                ("색상_품질_분석", self._analyze_color_quality),
                ("종합_점수_계산", self._calculate_overall_quality),
                ("등급_부여", self._assign_quality_grade),
                ("시각화_생성", self._generate_quality_visualization)
            ]
            
            self.logger.info(f"✅ 품질 평가 파이프라인 구성 완료: {len(self.assessment_pipeline)}단계")
        
        except Exception as e:
            self.logger.error(f"❌ 파이프라인 구성 실패: {e}")
            self.assessment_pipeline = []
    
    def _initialize_analyzers(self):
        """전문 분석기들 초기화 (기존 방식 유지)"""
        try:
            # 기술적 분석기
            self.technical_analyzer = TechnicalQualityAnalyzer(
                device=self.device,
                enable_gpu=TORCH_AVAILABLE and self.device != 'cpu'
            )
            
            # 기능적 분석기
            self.functional_analyzer = FittingQualityAnalyzer(
                device=self.device
            )
            
            # 색상 분석기
            self.color_analyzer = ColorQualityAnalyzer(
                device=self.device
            )
            
            self.logger.info("✅ 전문 분석기 초기화 완료")
        
        except Exception as e:
            self.logger.error(f"❌ 분석기 초기화 실패: {e}")
    
    # ==============================================
    # 🔥 메인 처리 메서드 (process)
    # ==============================================
    
    async def process(
        self,
        fitted_image: Union[np.ndarray, str, Path],
        person_image: Optional[Union[np.ndarray, str, Path]] = None,
        clothing_image: Optional[Union[np.ndarray, str, Path]] = None,
        fabric_type: str = "default",
        clothing_type: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """
        🔥 메인 품질 평가 처리 함수
        ✅ ModelLoader 통한 AI 모델 호출
        ✅ 8가지 품질 평가 실행
        ✅ 종합 점수 계산
        """
        
        start_time = time.time()
        
        try:
            self.logger.info(f"🎯 {self.step_name} 품질 평가 시작")
            
            # 1. 이미지 로드 및 검증
            fitted_img = self._load_and_validate_image(fitted_image, "fitted_image")
            if fitted_img is None:
                raise ValueError("유효하지 않은 fitted_image입니다")
            
            person_img = self._load_and_validate_image(person_image, "person_image") if person_image else None
            clothing_img = self._load_and_validate_image(clothing_image, "clothing_image") if clothing_image else None
            
            # 2. 입력 데이터 준비
            assessment_data = {
                'processed_image': fitted_img,
                'original_image': person_img,
                'clothing_image': clothing_img,
                'fabric_type': fabric_type,
                'clothing_type': clothing_type,
                'assessment_mode': self.assessment_config['mode'],
                **kwargs
            }
            
            # 3. 메모리 최적화
            if TORCH_AVAILABLE and self.optimization_enabled:
                self._optimize_memory()
            
            # 4. 품질 평가 파이프라인 실행
            for stage_name, stage_func in self.assessment_pipeline:
                self.logger.info(f"🔄 {stage_name} 실행 중...")
                
                stage_start = time.time()
                
                if asyncio.iscoroutinefunction(stage_func):
                    stage_result = await stage_func(assessment_data)
                else:
                    stage_result = stage_func(assessment_data)
                
                stage_duration = time.time() - stage_start
                
                # 결과 병합
                assessment_data.update(stage_result)
                
                self.logger.info(f"✅ {stage_name} 완료 ({stage_duration:.2f}초)")
            
            # 5. 최종 결과 구성
            processing_time = time.time() - start_time
            quality_metrics = assessment_data.get('quality_metrics')
            
            if quality_metrics is None:
                raise ValueError("품질 메트릭 계산 실패")
            
            result = {
                'success': True,
                'step_name': self.step_name,
                'overall_score': quality_metrics.overall_score,
                'confidence': quality_metrics.confidence,
                'grade': assessment_data.get('grade', 'acceptable'),
                'grade_description': assessment_data.get('grade_description', '수용 가능한 품질'),
                
                # 세부 분석 결과
                'detailed_scores': {
                    'technical': assessment_data.get('technical_results', {}),
                    'perceptual': assessment_data.get('perceptual_results', {}),
                    'aesthetic': assessment_data.get('aesthetic_results', {}),
                    'functional': assessment_data.get('functional_results', {}),
                    'color': assessment_data.get('color_results', {})
                },
                
                # 품질 메트릭 전체
                'quality_metrics': asdict(quality_metrics),
                
                # 메타데이터
                'processing_time': processing_time,
                'fabric_type': fabric_type,
                'clothing_type': clothing_type,
                'assessment_mode': self.assessment_config['mode'].value,
                
                # 시스템 정보
                'device_info': {
                    'device': self.device,
                    'is_m3_max': self.is_m3_max,
                    'ai_models_used': len(self.models_loaded),
                    'optimization_enabled': self.optimization_enabled
                },
                
                # 경고 및 권장사항
                'warnings': assessment_data.get('warnings', []),
                'recommendations': assessment_data.get('recommendations', [])
            }
            
            self.logger.info(f"✅ {self.step_name} 품질 평가 완료 - 점수: {result['overall_score']:.3f} ({processing_time:.2f}초)")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ {self.step_name} 처리 실패: {e}")
            
            return {
                'success': False,
                'step_name': self.step_name,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time': processing_time,
                'metadata': {
                    'device': self.device,
                    'pipeline_stages': len(self.assessment_pipeline),
                    'error_location': 'main_process'
                }
            }
    
    # ==============================================
    # 🔥 품질 분석 메서드들 (AI 모델 호출)
    # ==============================================
    
    async def _analyze_technical_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """기술적 품질 분석 (AI 모델 또는 전통적 방법)"""
        try:
            image = data['processed_image']
            
            # AI 모델 사용 가능한 경우
            if 'technical' in self.models_loaded:
                model = self.models_loaded['technical']
                
                # ModelLoader가 제공한 AI 모델로 추론
                if hasattr(model, 'predict') or hasattr(model, '__call__'):
                    # 이미지 전처리
                    processed_tensor = self._preprocess_for_ai_model(image)
                    
                    # AI 모델 추론 실행
                    with torch.no_grad() if TORCH_AVAILABLE else self._dummy_context():
                        if hasattr(model, 'predict'):
                            ai_result = model.predict(processed_tensor)
                        else:
                            ai_result = model(processed_tensor)
                    
                    # AI 결과 해석
                    technical_scores = self._interpret_technical_ai_result(ai_result)
                else:
                    # 전통적 방법으로 폴백
                    technical_scores = self._traditional_technical_analysis(image)
            else:
                # 전통적 분석 방법
                technical_scores = self._traditional_technical_analysis(image)
            
            return {
                'technical_results': technical_scores,
                'technical_score': technical_scores.get('overall_score', 0.5)
            }
        
        except Exception as e:
            self.logger.error(f"❌ 기술적 품질 분석 실패: {e}")
            return {
                'technical_results': {'error': str(e)},
                'technical_score': 0.3
            }
    
    async def _analyze_perceptual_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """지각적 품질 분석 (AI 모델 또는 전통적 방법)"""
        try:
            image = data['processed_image']
            original = data.get('original_image')
            
            # AI 모델 사용 가능한 경우
            if 'perceptual' in self.models_loaded:
                model = self.models_loaded['perceptual']
                
                # ModelLoader가 제공한 AI 모델로 추론
                if hasattr(model, 'predict') or hasattr(model, '__call__'):
                    # 이미지 쌍 전처리
                    processed_pair = self._preprocess_image_pair_for_ai(image, original)
                    
                    # AI 모델 추론 실행
                    with torch.no_grad() if TORCH_AVAILABLE else self._dummy_context():
                        if hasattr(model, 'predict'):
                            ai_result = model.predict(processed_pair)
                        else:
                            ai_result = model(processed_pair)
                    
                    # AI 결과 해석
                    perceptual_scores = self._interpret_perceptual_ai_result(ai_result)
                else:
                    # 전통적 방법으로 폴백
                    perceptual_scores = self._traditional_perceptual_analysis(image, original)
            else:
                # 전통적 분석 방법
                perceptual_scores = self._traditional_perceptual_analysis(image, original)
            
            return {
                'perceptual_results': perceptual_scores,
                'perceptual_score': perceptual_scores.get('overall_score', 0.5)
            }
        
        except Exception as e:
            self.logger.error(f"❌ 지각적 품질 분석 실패: {e}")
            return {
                'perceptual_results': {'error': str(e)},
                'perceptual_score': 0.3
            }
    
    async def _analyze_aesthetic_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """미적 품질 분석 (AI 모델 또는 전통적 방법)"""
        try:
            image = data['processed_image']
            
            # AI 모델 사용 가능한 경우
            if 'aesthetic' in self.models_loaded:
                model = self.models_loaded['aesthetic']
                
                # ModelLoader가 제공한 AI 모델로 추론
                if hasattr(model, 'predict') or hasattr(model, '__call__'):
                    # 이미지 전처리
                    processed_tensor = self._preprocess_for_ai_model(image)
                    
                    # AI 모델 추론 실행
                    with torch.no_grad() if TORCH_AVAILABLE else self._dummy_context():
                        if hasattr(model, 'predict'):
                            ai_result = model.predict(processed_tensor)
                        else:
                            ai_result = model(processed_tensor)
                    
                    # AI 결과 해석
                    aesthetic_scores = self._interpret_aesthetic_ai_result(ai_result)
                else:
                    # 전통적 방법으로 폴백
                    aesthetic_scores = self._traditional_aesthetic_analysis(image)
            else:
                # 전통적 분석 방법
                aesthetic_scores = self._traditional_aesthetic_analysis(image)
            
            return {
                'aesthetic_results': aesthetic_scores,
                'aesthetic_score': aesthetic_scores.get('overall_score', 0.5)
            }
        
        except Exception as e:
            self.logger.error(f"❌ 미적 품질 분석 실패: {e}")
            return {
                'aesthetic_results': {'error': str(e)},
                'aesthetic_score': 0.3
            }
    
    def _analyze_functional_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """기능적 품질 분석"""
        try:
            image = data['processed_image']
            clothing_type = data.get('clothing_type', 'default')
            
            # 전문 분석기 사용
            if self.functional_analyzer:
                functional_scores = self.functional_analyzer.analyze(image, clothing_type)
            else:
                # 폴백 분석
                functional_scores = self._basic_functional_analysis(image, clothing_type)
            
            return {
                'functional_results': functional_scores,
                'functional_score': functional_scores.get('overall_score', 0.5)
            }
        
        except Exception as e:
            self.logger.error(f"❌ 기능적 품질 분석 실패: {e}")
            return {
                'functional_results': {'error': str(e)},
                'functional_score': 0.3
            }
    
    def _analyze_color_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """색상 품질 분석"""
        try:
            image = data['processed_image']
            
            # 전문 분석기 사용
            if self.color_analyzer:
                color_scores = self.color_analyzer.analyze(image)
            else:
                # 폴백 분석
                color_scores = self._basic_color_analysis(image)
            
            return {
                'color_results': color_scores,
                'color_score': color_scores.get('overall_score', 0.5)
            }
        
        except Exception as e:
            self.logger.error(f"❌ 색상 품질 분석 실패: {e}")
            return {
                'color_results': {'error': str(e)},
                'color_score': 0.3
            }
    
    def _calculate_overall_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """종합 품질 점수 계산"""
        try:
            # 각 영역별 점수 수집
            scores = {
                'technical': data.get('technical_score', 0.5),
                'perceptual': data.get('perceptual_score', 0.5),
                'aesthetic': data.get('aesthetic_score', 0.5),
                'functional': data.get('functional_score', 0.5),
                'color': data.get('color_score', 0.5)
            }
            
            # 가중치 적용
            weights = {
                'technical': 0.25,
                'perceptual': 0.25,
                'aesthetic': 0.20,
                'functional': 0.20,
                'color': 0.10
            }
            
            # 가중 평균 계산
            overall_score = sum(scores[key] * weights[key] for key in scores.keys())
            
            # 신뢰도 계산
            confidence = self._calculate_confidence(data)
            
            # QualityMetrics 객체 생성
            quality_metrics = QualityMetrics(
                overall_score=overall_score,
                confidence=confidence,
                sharpness_score=data.get('technical_results', {}).get('sharpness', 0.5),
                color_score=data.get('color_score', 0.5),
                fitting_score=data.get('functional_results', {}).get('fitting_accuracy', 0.5),
                realism_score=data.get('perceptual_score', 0.5),
                artifacts_score=data.get('technical_results', {}).get('artifacts', 0.5),
                alignment_score=data.get('functional_results', {}).get('clothing_alignment', 0.5),
                lighting_score=data.get('aesthetic_results', {}).get('lighting', 0.5),
                texture_score=data.get('aesthetic_results', {}).get('texture', 0.5),
                device_used=self.device,
                model_version="v2.0"
            )
            
            return {
                'quality_metrics': quality_metrics,
                'overall_score': overall_score,
                'confidence': confidence
            }
        
        except Exception as e:
            self.logger.error(f"❌ 종합 품질 점수 계산 실패: {e}")
            return {
                'quality_metrics': QualityMetrics(),
                'overall_score': 0.3,
                'confidence': 0.1
            }
    
    def _assign_quality_grade(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """품질 등급 부여"""
        try:
            overall_score = data.get('overall_score', 0.5)
            
            if overall_score >= 0.9:
                grade = QualityGrade.EXCELLENT
                description = "탁월한 품질 - 상업적 사용 가능"
            elif overall_score >= 0.75:
                grade = QualityGrade.GOOD  
                description = "좋은 품질 - 일반적 사용 적합"
            elif overall_score >= 0.6:
                grade = QualityGrade.ACCEPTABLE
                description = "수용 가능한 품질 - 개선 권장"
            elif overall_score >= 0.4:
                grade = QualityGrade.POOR
                description = "불량한 품질 - 상당한 개선 필요"
            else:
                grade = QualityGrade.FAILED
                description = "실패한 품질 - 재처리 필요"
            
            return {
                'grade': grade.value,
                'grade_description': description
            }
        
        except Exception as e:
            self.logger.error(f"❌ 품질 등급 부여 실패: {e}")
            return {
                'grade': QualityGrade.ACCEPTABLE.value,
                'grade_description': "등급 계산 실패"
            }
    
    def _generate_quality_visualization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """품질 평가 시각화 생성"""
        try:
            # 기본 시각화 정보만 반환 (실제 이미지 생성은 선택사항)
            visualization_info = {
                'has_visualization': True,
                'quality_chart': f"품질 점수: {data.get('overall_score', 0.5):.3f}",
                'grade_display': data.get('grade_description', '알 수 없음'),
                'detailed_breakdown': data.get('detailed_scores', {})
            }
            
            return {
                'visualization': visualization_info
            }
        
        except Exception as e:
            self.logger.error(f"❌ 시각화 생성 실패: {e}")
            return {
                'visualization': {'error': str(e)}
            }
    
    # ==============================================
    # 🔥 AI 모델 유틸리티 메서드들
    # ==============================================
    
    def _preprocess_for_ai_model(self, image: np.ndarray) -> Any:
        """AI 모델용 이미지 전처리"""
        try:
            if TORCH_AVAILABLE:
                # NumPy to Tensor 변환
                if len(image.shape) == 3:
                    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                else:
                    tensor = torch.from_numpy(image).float() / 255.0
                
                # 배치 차원 추가
                tensor = tensor.unsqueeze(0)
                
                # 디바이스로 이동
                if self.device != 'cpu':
                    tensor = tensor.to(self.device)
                
                return tensor
            else:
                # PyTorch 없을 때는 NumPy 배열 그대로 반환
                return image
        
        except Exception as e:
            self.logger.error(f"❌ AI 모델용 전처리 실패: {e}")
            return image
    
    def _preprocess_image_pair_for_ai(self, image1: np.ndarray, image2: Optional[np.ndarray]) -> Any:
        """이미지 쌍 AI 모델용 전처리"""
        try:
            if image2 is None:
                return self._preprocess_for_ai_model(image1)
            
            if TORCH_AVAILABLE:
                tensor1 = self._preprocess_for_ai_model(image1)
                tensor2 = self._preprocess_for_ai_model(image2)
                
                # 쌍으로 묶기
                pair_tensor = torch.cat([tensor1, tensor2], dim=1)
                return pair_tensor
            else:
                return np.concatenate([image1, image2], axis=-1)
        
        except Exception as e:
            self.logger.error(f"❌ 이미지 쌍 전처리 실패: {e}")
            return self._preprocess_for_ai_model(image1)
    
    def _dummy_context(self):
        """PyTorch 없을 때 dummy context manager"""
        class DummyContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return DummyContext()
    
    # ==============================================
    # 🔥 AI 결과 해석 메서드들
    # ==============================================
    
    def _interpret_technical_ai_result(self, ai_result: Any) -> Dict[str, Any]:
        """기술적 품질 AI 결과 해석"""
        try:
            if TORCH_AVAILABLE and hasattr(ai_result, 'cpu'):
                result_data = ai_result.cpu().numpy()
            elif hasattr(ai_result, 'numpy'):
                result_data = ai_result.numpy()
            else:
                result_data = ai_result
            
            # AI 결과를 품질 점수로 변환
            if isinstance(result_data, np.ndarray):
                if result_data.size == 1:
                    overall_score = float(result_data.item())
                else:
                    overall_score = float(np.mean(result_data))
            else:
                overall_score = float(result_data) if isinstance(result_data, (int, float)) else 0.5
            
            return {
                'overall_score': max(0.0, min(1.0, overall_score)),
                'sharpness': overall_score * 0.9 + 0.1,
                'artifacts': 1.0 - overall_score * 0.3,
                'noise_level': overall_score * 0.8 + 0.2,
                'analysis_method': 'ai_model'
            }
        
        except Exception as e:
            self.logger.error(f"❌ 기술적 AI 결과 해석 실패: {e}")
            return self._traditional_technical_analysis(None)
    
    def _interpret_perceptual_ai_result(self, ai_result: Any) -> Dict[str, Any]:
        """지각적 품질 AI 결과 해석"""
        try:
            if TORCH_AVAILABLE and hasattr(ai_result, 'cpu'):
                result_data = ai_result.cpu().numpy()
            elif hasattr(ai_result, 'numpy'):
                result_data = ai_result.numpy()
            else:
                result_data = ai_result
            
            # AI 결과를 지각적 점수로 변환
            if isinstance(result_data, np.ndarray):
                overall_score = float(np.mean(result_data))
            else:
                overall_score = float(result_data) if isinstance(result_data, (int, float)) else 0.5
            
            return {
                'overall_score': max(0.0, min(1.0, overall_score)),
                'visual_quality': overall_score,
                'structural_similarity': overall_score * 0.95 + 0.05,
                'perceptual_distance': 1.0 - overall_score,
                'analysis_method': 'ai_model'
            }
        
        except Exception as e:
            self.logger.error(f"❌ 지각적 AI 결과 해석 실패: {e}")
            return self._traditional_perceptual_analysis(None, None)
    
    def _interpret_aesthetic_ai_result(self, ai_result: Any) -> Dict[str, Any]:
        """미적 품질 AI 결과 해석"""
        try:
            if TORCH_AVAILABLE and hasattr(ai_result, 'cpu'):
                result_data = ai_result.cpu().numpy()
            elif hasattr(ai_result, 'numpy'):
                result_data = ai_result.numpy()
            else:
                result_data = ai_result
            
            # AI 결과를 미적 점수로 변환
            if isinstance(result_data, np.ndarray):
                overall_score = float(np.mean(result_data))
            else:
                overall_score = float(result_data) if isinstance(result_data, (int, float)) else 0.5
            
            return {
                'overall_score': max(0.0, min(1.0, overall_score)),
                'composition': overall_score * 0.9 + 0.1,
                'lighting': overall_score * 0.95 + 0.05,
                'texture': overall_score * 0.85 + 0.15,
                'color_harmony': overall_score * 0.8 + 0.2,
                'analysis_method': 'ai_model'
            }
        
        except Exception as e:
            self.logger.error(f"❌ 미적 AI 결과 해석 실패: {e}")
            return self._traditional_aesthetic_analysis(None)
    
    # ==============================================
    # 🔥 전통적 분석 메서드들 (AI 모델 없을 때)
    # ==============================================
    
    def _traditional_technical_analysis(self, image: Optional[np.ndarray]) -> Dict[str, Any]:
        """전통적 기술적 분석 방법"""
        try:
            if image is None:
                return {
                    'overall_score': 0.5,
                    'sharpness': 0.5,
                    'artifacts': 0.7,
                    'noise_level': 0.6,
                    'analysis_method': 'fallback'
                }
            
            # 전문 분석기 사용
            if self.technical_analyzer:
                return self.technical_analyzer.analyze(image)
            else:
                # 간단한 선명도 측정
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if 'cv2' in globals() else np.mean(image, axis=2)
                else:
                    gray = image
                
                # 선명도 계산
                if 'cv2' in globals():
                    laplacian = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
                    sharpness = laplacian.var() / 10000.0  # 정규화
                else:
                    # OpenCV 없을 때 간단한 gradient 계산
                    dx = np.diff(gray, axis=1)
                    dy = np.diff(gray, axis=0)
                    sharpness = (np.var(dx) + np.var(dy)) / 20000.0
                
                sharpness = max(0.0, min(1.0, sharpness))
                
                return {
                    'overall_score': sharpness * 0.8 + 0.2,
                    'sharpness': sharpness,
                    'artifacts': 0.8,  # 기본값
                    'noise_level': 0.7,  # 기본값
                    'analysis_method': 'traditional'
                }
        
        except Exception as e:
            self.logger.error(f"❌ 전통적 기술 분석 실패: {e}")
            return {
                'overall_score': 0.5,
                'sharpness': 0.5,
                'artifacts': 0.7,
                'noise_level': 0.6,
                'analysis_method': 'error_fallback'
            }
    
    def _traditional_perceptual_analysis(self, image1: Optional[np.ndarray], image2: Optional[np.ndarray]) -> Dict[str, Any]:
        """전통적 지각적 분석 방법"""
        try:
            if image1 is None:
                return {
                    'overall_score': 0.5,
                    'visual_quality': 0.5,
                    'structural_similarity': 0.5,
                    'perceptual_distance': 0.5,
                    'analysis_method': 'fallback'
                }
            
            # SSIM 계산 (가능한 경우)
            if image2 is not None and SKIMAGE_AVAILABLE:
                try:
                    ssim_score = ssim(image1, image2, multichannel=True, channel_axis=2)
                    overall_score = max(0.0, ssim_score)
                except:
                    overall_score = 0.7
            else:
                # 간단한 통계적 분석
                mean_brightness = np.mean(image1) / 255.0
                brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2  # 0.5에 가까울수록 좋음
                
                # 대비 분석
                contrast = np.std(image1) / 255.0
                contrast_score = min(1.0, contrast * 2)  # 적절한 대비
                
                overall_score = (brightness_score + contrast_score) / 2
            
            return {
                'overall_score': overall_score,
                'visual_quality': overall_score,
                'structural_similarity': overall_score * 0.9 + 0.1,
                'perceptual_distance': 1.0 - overall_score,
                'analysis_method': 'traditional'
            }
        
        except Exception as e:
            self.logger.error(f"❌ 전통적 지각 분석 실패: {e}")
            return {
                'overall_score': 0.5,
                'visual_quality': 0.5,
                'structural_similarity': 0.5,
                'perceptual_distance': 0.5,
                'analysis_method': 'error_fallback'
            }
    
    def _traditional_aesthetic_analysis(self, image: Optional[np.ndarray]) -> Dict[str, Any]:
        """전통적 미적 분석 방법"""
        try:
            if image is None:
                return {
                    'overall_score': 0.5,
                    'composition': 0.5,
                    'lighting': 0.5,
                    'texture': 0.5,
                    'color_harmony': 0.5,
                    'analysis_method': 'fallback'
                }
            
            # 색상 분포 분석
            if len(image.shape) == 3:
                color_std = np.mean([np.std(image[:,:,i]) for i in range(3)]) / 255.0
            else:
                color_std = np.std(image) / 255.0
            
            color_harmony = min(1.0, color_std * 1.5)
            
            # 밝기 분포 분석
            brightness = np.mean(image) / 255.0
            lighting_score = 1.0 - abs(brightness - 0.5) * 1.5
            lighting_score = max(0.0, min(1.0, lighting_score))
            
            overall_score = (color_harmony + lighting_score) / 2
            
            return {
                'overall_score': overall_score,
                'composition': overall_score * 0.9 + 0.1,
                'lighting': lighting_score,
                'texture': overall_score * 0.8 + 0.2,
                'color_harmony': color_harmony,
                'analysis_method': 'traditional'
            }
        
        except Exception as e:
            self.logger.error(f"❌ 전통적 미적 분석 실패: {e}")
            return {
                'overall_score': 0.5,
                'composition': 0.5,
                'lighting': 0.5,
                'texture': 0.5,
                'color_harmony': 0.5,
                'analysis_method': 'error_fallback'
            }
    
    # ==============================================
    # 🔥 폴백 분석 메서드들
    # ==============================================
    
    def _basic_functional_analysis(self, image: np.ndarray, clothing_type: str) -> Dict[str, Any]:
        """기본 기능적 분석 (폴백)"""
        try:
            # 간단한 기하학적 일관성 체크
            height, width = image.shape[:2]
            aspect_ratio = width / height
            
            # 의류 타입별 기대 비율
            expected_ratios = {
                'shirt': (0.7, 1.3),
                'dress': (0.6, 1.0),
                'pants': (0.8, 1.2),
                'default': (0.5, 1.5)
            }
            
            min_ratio, max_ratio = expected_ratios.get(clothing_type, expected_ratios['default'])
            
            if min_ratio <= aspect_ratio <= max_ratio:
                fitting_score = 1.0
            else:
                fitting_score = max(0.0, 1.0 - abs(aspect_ratio - (min_ratio + max_ratio) / 2) * 2)
            
            return {
                'fitting_accuracy': fitting_score,
                'clothing_alignment': 0.7,
                'naturalness': 0.6,
                'overall_score': (fitting_score + 0.7 + 0.6) / 3
            }
        
        except Exception as e:
            self.logger.error(f"❌ 기본 기능적 분석 실패: {e}")
            return {
                'fitting_accuracy': 0.5,
                'clothing_alignment': 0.5,
                'naturalness': 0.5,
                'overall_score': 0.5
            }
    
    def _basic_color_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """기본 색상 분석 (폴백)"""
        try:
            if len(image.shape) == 3:
                # RGB 채널별 분석
                color_means = [np.mean(image[:,:,i]) for i in range(3)]
                color_stds = [np.std(image[:,:,i]) for i in range(3)]
                
                # 색상 일관성 (채널 간 균형)
                consistency = 1.0 - np.std(color_means) / (np.mean(color_means) + 1e-8)
                
                # 색상 대비
                contrast = np.mean(color_stds) / 255.0
                
                overall_score = (consistency + min(contrast * 2, 1.0)) / 2
            else:
                # 그레이스케일
                overall_score = 0.6
            
            return {
                'color_consistency': consistency if len(image.shape) == 3 else 0.6,
                'color_naturalness': 0.7,
                'color_contrast': min(contrast * 2, 1.0) if len(image.shape) == 3 else 0.6,
                'overall_score': overall_score
            }
        
        except Exception as e:
            self.logger.error(f"❌ 기본 색상 분석 실패: {e}")
            return {
                'color_consistency': 0.5,
                'color_naturalness': 0.5,
                'color_contrast': 0.5,
                'overall_score': 0.5
            }
    
    # ==============================================
    # 🔥 유틸리티 메서드들
    # ==============================================
    
    def _load_and_validate_image(self, image_input: Union[np.ndarray, str, Path], name: str) -> Optional[np.ndarray]:
        """이미지 로드 및 검증"""
        try:
            if image_input is None:
                return None
            
            if isinstance(image_input, np.ndarray):
                return image_input
            elif isinstance(image_input, (str, Path)):
                image_path = Path(image_input)
                if image_path.exists():
                    from PIL import Image
                    with Image.open(image_path) as img:
                        return np.array(img)
            
            self.logger.warning(f"❌ 이미지 로드 실패: {name}")
            return None
        
        except Exception as e:
            self.logger.error(f"❌ 이미지 로드 오류 {name}: {e}")
            return None
    
    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        """신뢰도 계산"""
        try:
            confidence_factors = []
            
            # 1. AI 모델 사용 여부
            ai_model_count = sum(1 for key in self.models_loaded.keys() if key in ['technical', 'perceptual', 'aesthetic'])
            ai_confidence = min(1.0, ai_model_count / 3.0)  # 최대 3개 모델
            confidence_factors.append(ai_confidence)
            
            # 2. 입력 데이터 품질
            has_original = data.get('original_image') is not None
            has_clothing = data.get('clothing_image') is not None
            data_quality = (0.5 + 0.3 * has_original + 0.2 * has_clothing)
            confidence_factors.append(data_quality)
            
            # 3. 분석 결과 일관성
            scores = [
                data.get('technical_score', 0.5),
                data.get('perceptual_score', 0.5),
                data.get('aesthetic_score', 0.5),
                data.get('functional_score', 0.5),
                data.get('color_score', 0.5)
            ]
            score_std = np.std(scores)
            consistency = max(0.0, 1.0 - score_std * 2)  # 표준편차가 낮을수록 일관성 높음
            confidence_factors.append(consistency)
            
            # 4. 시스템 최적화 상태
            optimization_factor = 0.9 if self.optimization_enabled else 0.7
            confidence_factors.append(optimization_factor)
            
            return np.mean(confidence_factors)
        
        except Exception as e:
            self.logger.error(f"❌ 신뢰도 계산 실패: {e}")
            return 0.7
    
    # ==============================================
    # 🔥 시스템 최적화 및 관리 메서드들
    # ==============================================
    
    def _determine_device(self, device: str) -> str:
        """디바이스 결정"""
        if device == "auto":
            if TORCH_AVAILABLE:
                if torch.backends.mps.is_available():
                    return "mps"
                elif torch.cuda.is_available():
                    return "cuda"
            return "cpu"
        return device
    
    def _get_device_type(self) -> str:
        """디바이스 타입 반환"""
        if "mps" in self.device:
            return "Apple Silicon"
        elif "cuda" in self.device:
            return "NVIDIA GPU"
        else:
            return "CPU"
    
    def _get_system_memory(self) -> int:
        """시스템 메모리 용량 (GB)"""
        try:
            if PSUTIL_AVAILABLE:
                return int(psutil.virtual_memory().total / (1024**3))
            else:
                return 8  # 기본값
        except:
            return 8
    
    def _detect_m3_max(self) -> bool:
        """M3 Max 칩 감지"""
        try:
            if self.device == "mps" and self.memory_gb >= 32:
                return True
            return False
        except:
            return False
    
    def _optimize_for_m3_max(self):
        """M3 Max 최적화 설정"""
        try:
            if TORCH_AVAILABLE and self.device == "mps":
                # MPS 최적화 설정
                torch.mps.set_high_watermark_ratio(0.0)
                
                # 메모리 효율적 설정
                if hasattr(torch.backends.mps, 'set_max_memory_allocation'):
                    torch.backends.mps.set_max_memory_allocation(self.memory_gb * 0.8 * 1024**3)
            
            self.logger.info("✅ M3 Max 최적화 설정 완료")
        
        except Exception as e:
            self.logger.warning(f"⚠️ M3 Max 최적화 설정 실패: {e}")
    
    def _optimize_memory(self):
        """메모리 최적화"""
        try:
            if TORCH_AVAILABLE:
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
            
            # 가비지 컬렉션
            gc.collect()
            
        except Exception as e:
            self.logger.debug(f"메모리 최적화 오류: {e}")
    
    def cleanup_resources(self):
        """리소스 정리"""
        try:
            self.logger.info(f"🧹 {self.step_name} 리소스 정리 시작...")
            
            # ModelLoader 인터페이스 정리
            if self.model_interface:
                try:
                    self.model_interface.cleanup()
                except Exception as e:
                    self.logger.warning(f"⚠️ ModelLoader 인터페이스 정리 실패: {e}")
            
            # 모델 메모리 해제
            self.models_loaded.clear()
            
            # 분석기 정리
            for analyzer_name in ['technical_analyzer', 'functional_analyzer', 'color_analyzer']:
                analyzer = getattr(self, analyzer_name, None)
                if analyzer and hasattr(analyzer, 'cleanup'):
                    try:
                        analyzer.cleanup()
                    except Exception as e:
                        self.logger.warning(f"⚠️ {analyzer_name} 정리 실패: {e}")
                setattr(self, analyzer_name, None)
            
            # 파이프라인 정리
            self.assessment_pipeline.clear()
            
            # 메모리 최적화
            self._optimize_memory()
            
            self.is_initialized = False
            self.logger.info(f"✅ {self.step_name} 리소스 정리 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 리소스 정리 중 오류: {e}")
    
    def get_step_info(self) -> Dict[str, Any]:
        """Step 정보 반환"""
        return {
            'step_name': self.step_name,
            'step_number': self.step_number,
            'device': self.device,
            'device_type': self.device_type,
            'memory_gb': self.memory_gb,
            'is_m3_max': self.is_m3_max,
            'ai_models_loaded': len(self.models_loaded),
            'assessment_modes': [mode.value for mode in AssessmentMode],
            'quality_threshold': self.quality_threshold,
            'pipeline_stages': len(self.assessment_pipeline),
            'optimization_enabled': self.optimization_enabled,
            'is_initialized': self.is_initialized,
            'model_interface_available': self.model_interface is not None,
            'base_step_mixin_available': BASE_STEP_MIXIN_AVAILABLE,
            'model_loader_available': MODEL_LOADER_AVAILABLE
        }

# ==============================================
# 🔥 팩토리 및 유틸리티 함수들
# ==============================================

def create_quality_assessment_step(
    device: str = "auto",
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> QualityAssessmentStep:
    """품질 평가 Step 생성 함수 (기존 호환성)"""
    return QualityAssessmentStep(device=device, config=config, **kwargs)

async def create_and_initialize_quality_assessment_step(
    device: str = "auto",
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> QualityAssessmentStep:
    """품질 평가 Step 생성 및 초기화"""
    step = QualityAssessmentStep(device=device, config=config, **kwargs)
    await step.initialize()
    return step

# ==============================================
# 🔥 모듈 익스포트 (기존 호환성 유지)
# ==============================================

__all__ = [
    # 메인 클래스
    'QualityAssessmentStep',
    
    # 데이터 구조
    'QualityMetrics',
    'QualityGrade', 
    'AssessmentMode',
    'QualityAspect',
    
    # AI 모델 클래스들
    'EnhancedPerceptualQualityModel',
    'EnhancedAestheticQualityModel',
    
    # 분석기 클래스들
    'TechnicalQualityAnalyzer',
    'FittingQualityAnalyzer',
    'ColorQualityAnalyzer',
    
    # 팩토리 함수들
    'create_quality_assessment_step',
    'create_and_initialize_quality_assessment_step'
]

# ==============================================
# 🔥 테스트 코드 (개발용)
# ==============================================

if __name__ == "__main__":
    async def test_quality_assessment_step():
        """품질 평가 Step 테스트"""
        try:
            print("🧪 QualityAssessmentStep 테스트 시작...")
            
            # Step 생성
            step = QualityAssessmentStep(device="auto")
            
            # 기본 속성 확인
            assert hasattr(step, 'logger'), "logger 속성이 없습니다!"
            assert hasattr(step, 'process'), "process 메서드가 없습니다!"
            assert hasattr(step, 'cleanup_resources'), "cleanup_resources 메서드가 없습니다!"
            
            # Step 정보 확인
            step_info = step.get_step_info()
            assert 'step_name' in step_info, "step_name이 step_info에 없습니다!"
            
            print("✅ QualityAssessmentStep 테스트 성공")
            print(f"📊 Step 정보: {step_info}")
            print(f"🔧 디바이스: {step.device} ({step.device_type})")
            print(f"💾 메모리: {step.memory_gb}GB")
            print(f"🍎 M3 Max: {'✅' if step.is_m3_max else '❌'}")
            print(f"🧠 BaseStepMixin: {'✅' if BASE_STEP_MIXIN_AVAILABLE else '❌'}")
            print(f"🔌 ModelLoader: {'✅' if MODEL_LOADER_AVAILABLE else '❌'}")
            print(f"🎯 파이프라인 단계: {len(step.assessment_pipeline)}")
            
            return True
            
        except Exception as e:
            print(f"❌ QualityAssessmentStep 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 비동기 테스트 실행
    import asyncio
    asyncio.run(test_quality_assessment_step())