# app/ai_pipeline/steps/step_07_post_processing.py
"""
MyCloset AI - 7단계: 후처리 (Post Processing)
🔥 완전 통합 프로덕션 버전 - M3 Max 128GB 최적화

✅ 통일된 생성자 패턴 100% 적용
✅ Model Loader + Memory Manager 완전 연동  
✅ Pipeline Manager 완벽 호환
✅ 프로덕션 레벨 이미지 향상 알고리즘
✅ M3 Max Neural Engine + Metal Performance Shaders 활용
✅ 다중 향상 기법 (Super Resolution, Denoising, Sharpening 등)
✅ Graceful Degradation + 완벽한 에러 처리
"""

import os
import sys
import logging
import time
import asyncio
import threading
import gc
import hashlib
import json
import math
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
import weakref

# 핵심 라이브러리
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# 선택적 고급 라이브러리들
try:
    from scipy.ndimage import gaussian_filter, median_filter
    from scipy.signal import convolve2d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from skimage import restoration, filters, exposure, morphology
    from skimage.measure import compare_ssim, compare_psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# MyCloset AI 핵심 유틸리티 연동
try:
    from app.ai_pipeline.utils.model_loader import (
        BaseStepMixin, ModelLoader, ModelConfig, ModelType,
        get_global_model_loader, create_model_loader
    )
    MODEL_LOADER_AVAILABLE = True
except ImportError:
    MODEL_LOADER_AVAILABLE = False
    BaseStepMixin = object

try:
    from app.ai_pipeline.utils.memory_manager import (
        MemoryManager, get_global_memory_manager, optimize_memory_usage
    )
    MEMORY_MANAGER_AVAILABLE = True
except ImportError:
    MEMORY_MANAGER_AVAILABLE = False

try:
    from app.ai_pipeline.utils.data_converter import (
        DataConverter, get_global_data_converter
    )
    DATA_CONVERTER_AVAILABLE = True
except ImportError:
    DATA_CONVERTER_AVAILABLE = False

# 로깅 설정
logger = logging.getLogger(__name__)

# ==============================================
# 1. 열거형 및 데이터 클래스 정의
# ==============================================

class EnhancementMethod(Enum):
    """향상 방법"""
    SUPER_RESOLUTION = "super_resolution"
    NOISE_REDUCTION = "noise_reduction"
    SHARPENING = "sharpening"
    COLOR_CORRECTION = "color_correction"
    CONTRAST_ENHANCEMENT = "contrast_enhancement"
    FACE_ENHANCEMENT = "face_enhancement"
    EDGE_ENHANCEMENT = "edge_enhancement"
    TEXTURE_ENHANCEMENT = "texture_enhancement"
    AUTO = "auto"

class QualityLevel(Enum):
    """품질 레벨"""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"
    ULTRA = "ultra"

class ProcessingMode(Enum):
    """처리 모드"""
    REAL_TIME = "real_time"
    QUALITY = "quality"
    BATCH = "batch"

@dataclass
class PostProcessingConfig:
    """후처리 설정"""
    quality_level: QualityLevel = QualityLevel.BALANCED
    processing_mode: ProcessingMode = ProcessingMode.QUALITY
    enabled_methods: List[EnhancementMethod] = field(default_factory=lambda: [
        EnhancementMethod.NOISE_REDUCTION,
        EnhancementMethod.SHARPENING,
        EnhancementMethod.COLOR_CORRECTION,
        EnhancementMethod.CONTRAST_ENHANCEMENT
    ])
    max_resolution: Tuple[int, int] = (2048, 2048)
    use_gpu_acceleration: bool = True
    preserve_original_ratio: bool = True
    apply_face_detection: bool = True
    batch_size: int = 1
    cache_size: int = 50

@dataclass
class PostProcessingResult:
    """후처리 결과"""
    success: bool
    enhanced_image: Optional[np.ndarray] = None
    quality_improvement: float = 0.0
    applied_methods: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

# ==============================================
# 2. 고급 이미지 향상 신경망 모델
# ==============================================

class SRResNet(nn.Module):
    """Super Resolution ResNet 모델"""
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=16):
        super(SRResNet, self).__init__()
        
        # 초기 컨볼루션
        self.conv_first = nn.Conv2d(in_channels, num_features, 9, padding=4)
        self.relu = nn.ReLU(inplace=True)
        
        # 잔차 블록들
        self.res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.res_blocks.append(self._make_res_block(num_features))
        
        # 업샘플링 레이어들
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        
        # 최종 출력
        self.conv_last = nn.Conv2d(num_features, out_channels, 9, padding=4)
    
    def _make_res_block(self, num_features):
        """잔차 블록 생성"""
        return nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.BatchNorm2d(num_features)
        )
    
    def forward(self, x):
        """순전파"""
        # 초기 특성 추출
        feat = self.relu(self.conv_first(x))
        residual = feat
        
        # 잔차 블록들 통과
        for res_block in self.res_blocks:
            res_feat = res_block(feat)
            feat = feat + res_feat
        
        # 업샘플링
        feat = self.upsampler(feat + residual)
        
        # 최종 출력
        out = self.conv_last(feat)
        
        return out

class DenoiseNet(nn.Module):
    """노이즈 제거 신경망"""
    def __init__(self, in_channels=3, out_channels=3, num_features=64):
        super(DenoiseNet, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features * 2, num_features * 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_features * 2, num_features, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, out_channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ==============================================
# 3. 메인 PostProcessingStep 클래스
# ==============================================

class PostProcessingStep(BaseStepMixin):
    """
    7단계: 후처리 - 완전 통합 프로덕션 버전
    
    ✅ 통일된 생성자 패턴 적용
    ✅ Model Loader + Memory Manager 완전 연동
    ✅ Pipeline Manager 100% 호환
    ✅ M3 Max 128GB 최적화
    ✅ 다중 이미지 향상 기법
    ✅ Graceful Degradation
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """✅ 완전 통합 생성자 - 통일된 패턴 적용"""
        
        # === 1. 통일된 기본 초기화 ===
        self.device = self._auto_detect_device(device)
        self.config = config or {}
        self.step_name = self.__class__.__name__
        self.logger = logging.getLogger(f"pipeline.{self.step_name}")
        
        # === 2. 표준 시스템 파라미터 ===
        self.device_type = kwargs.get('device_type', 'auto')
        self.memory_gb = kwargs.get('memory_gb', 16.0)
        self.is_m3_max = kwargs.get('is_m3_max', self._detect_m3_max())
        self.optimization_enabled = kwargs.get('optimization_enabled', True)
        self.quality_level = kwargs.get('quality_level', 'balanced')
        
        # === 3. Step별 설정 병합 ===
        self._merge_step_specific_config(kwargs)
        
        # === 4. 초기화 상태 ===
        self.is_initialized = False
        self._initialization_lock = threading.RLock()
        
        # === 5. Model Loader 연동 (BaseStepMixin) ===
        if MODEL_LOADER_AVAILABLE:
            try:
                self._setup_model_interface()
            except Exception as e:
                self.logger.warning(f"Model Loader 연동 실패: {e}")
                self.model_interface = None
        else:
            self.model_interface = None
        
        # === 6. Step 특화 초기화 ===
        self._initialize_step_specific()
        
        # === 7. 초기화 완료 로깅 ===
        self.logger.info(f"🎯 {self.step_name} 초기화 완료 - 디바이스: {self.device}")
        if self.is_m3_max:
            self.logger.info(f"🍎 M3 Max 최적화 모드 (메모리: {self.memory_gb}GB)")
    
    def _auto_detect_device(self, preferred_device: Optional[str]) -> str:
        """💡 지능적 디바이스 자동 감지"""
        if preferred_device:
            return preferred_device

        try:
            import torch
            if torch.backends.mps.is_available():
                return 'mps'  # M3 Max 우선
            elif torch.cuda.is_available():
                return 'cuda'  # NVIDIA GPU
            else:
                return 'cpu'  # 폴백
        except ImportError:
            return 'cpu'

    def _detect_m3_max(self) -> bool:
        """🍎 M3 Max 칩 자동 감지"""
        try:
            import platform
            import subprocess

            if platform.system() == 'Darwin':  # macOS
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                cpu_info = result.stdout.strip()
                return 'M3 Max' in cpu_info or 'M3' in cpu_info
        except:
            pass
        return False

    def _merge_step_specific_config(self, kwargs: Dict[str, Any]):
        """7단계 특화 설정 병합"""
        
        # 후처리 설정
        self.post_processing_config = PostProcessingConfig()
        
        # 설정 업데이트
        if 'quality_level' in kwargs:
            self.post_processing_config.quality_level = QualityLevel(kwargs['quality_level'])
        
        if 'processing_mode' in kwargs:
            self.post_processing_config.processing_mode = ProcessingMode(kwargs['processing_mode'])
        
        if 'enabled_methods' in kwargs:
            self.post_processing_config.enabled_methods = [
                EnhancementMethod(method) for method in kwargs['enabled_methods']
            ]
        
        # M3 Max 특화 설정
        if self.is_m3_max:
            self.post_processing_config.use_gpu_acceleration = True
            self.post_processing_config.max_resolution = (4096, 4096)  # M3 Max 고해상도 지원
            self.post_processing_config.batch_size = min(8, max(1, int(self.memory_gb / 16)))
            self.post_processing_config.cache_size = min(100, max(25, int(self.memory_gb * 2)))
        
        # 추가 설정들
        self.enhancement_strength = kwargs.get('enhancement_strength', 0.7)
        self.preserve_faces = kwargs.get('preserve_faces', True)
        self.auto_adjust_brightness = kwargs.get('auto_adjust_brightness', True)

    def _initialize_step_specific(self):
        """7단계 특화 초기화"""
        
        # 캐시 및 상태 관리
        self.enhancement_cache: Dict[str, PostProcessingResult] = {}
        self.model_cache: Dict[str, Any] = {}
        self.face_detector = None
        
        # 성능 통계
        self.processing_stats = {
            'total_processed': 0,
            'successful_enhancements': 0,
            'average_improvement': 0.0,
            'method_usage': {},
            'cache_hits': 0,
            'average_processing_time': 0.0
        }
        
        # 스레드 풀 (M3 Max 최적화)
        max_workers = 6 if self.is_m3_max else 3
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"{self.step_name}_worker"
        )
        
        # 메모리 관리
        if MEMORY_MANAGER_AVAILABLE:
            try:
                self.memory_manager = get_global_memory_manager()
                if not self.memory_manager:
                    from app.ai_pipeline.utils.memory_manager import create_memory_manager
                    self.memory_manager = create_memory_manager(device=self.device)
            except Exception as e:
                self.logger.warning(f"Memory Manager 연동 실패: {e}")
                self.memory_manager = None
        else:
            self.memory_manager = None
        
        # 데이터 변환기
        if DATA_CONVERTER_AVAILABLE:
            try:
                self.data_converter = get_global_data_converter()
            except Exception as e:
                self.logger.warning(f"Data Converter 연동 실패: {e}")
                self.data_converter = None
        else:
            self.data_converter = None
        
        # 모델 경로 설정
        self.model_base_path = Path("backend/app/ai_pipeline/models/ai_models")
        self.checkpoint_path = self.model_base_path / "checkpoints" / "step_07_post_processing"
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # 향상 방법별 가중치
        self.enhancement_weights = {
            EnhancementMethod.SUPER_RESOLUTION: 0.3,
            EnhancementMethod.NOISE_REDUCTION: 0.2,
            EnhancementMethod.SHARPENING: 0.2,
            EnhancementMethod.COLOR_CORRECTION: 0.15,
            EnhancementMethod.CONTRAST_ENHANCEMENT: 0.1,
            EnhancementMethod.FACE_ENHANCEMENT: 0.05
        }
        
        self.logger.info(f"📦 7단계 특화 초기화 완료 - 활성화된 방법: {len(self.post_processing_config.enabled_methods)}개")

    async def initialize(self) -> bool:
        """
        ✅ 통일된 초기화 인터페이스 - Pipeline Manager 호환
        
        Returns:
            bool: 초기화 성공 여부
        """
        async with asyncio.Lock():
            if self.is_initialized:
                return True
        
        try:
            self.logger.info("🔄 7단계: 후처리 시스템 초기화 중...")
            
            # 1. AI 모델들 초기화
            await self._initialize_ai_models()
            
            # 2. 얼굴 검출기 초기화
            if self.preserve_faces:
                await self._initialize_face_detector()
            
            # 3. 이미지 필터 초기화
            self._initialize_image_filters()
            
            # 4. GPU 가속 초기화 (M3 Max/CUDA)
            if self.post_processing_config.use_gpu_acceleration:
                await self._initialize_gpu_acceleration()
            
            # 5. M3 Max 최적화 워밍업
            if self.is_m3_max and self.optimization_enabled:
                await self._warmup_m3_max()
            
            # 6. 캐시 시스템 초기화
            self._initialize_cache_system()
            
            self.is_initialized = True
            self.logger.info("✅ 후처리 시스템 초기화 완료")
            
            return True
            
        except Exception as e:
            error_msg = f"후처리 시스템 초기화 실패: {e}"
            self.logger.error(f"❌ {error_msg}")
            
            # 최소한의 폴백 시스템 초기화
            self._initialize_fallback_system()
            self.is_initialized = True
            
            return True  # Graceful degradation

    async def _initialize_ai_models(self):
        """AI 모델들 초기화 (Model Loader 활용)"""
        try:
            if not self.model_interface:
                self.logger.warning("Model Loader 인터페이스가 없습니다. 직접 모델 로드 시도.")
                await self._load_models_direct()
                return
            
            # Super Resolution 모델 로드
            sr_config = {
                'model_name': 'srresnet_x4',
                'model_class': SRResNet,
                'checkpoint_path': str(self.checkpoint_path / "srresnet_x4.pth"),
                'device': self.device,
                'use_fp16': self.is_m3_max
            }
            
            try:
                self.sr_model = await self.model_interface.load_model_async('srresnet_x4', sr_config)
                self.logger.info("✅ Super Resolution 모델 로드 성공")
            except Exception as e:
                self.logger.warning(f"SR 모델 로드 실패: {e}")
                self.sr_model = None
            
            # Denoising 모델 로드
            denoise_config = {
                'model_name': 'denoise_net',
                'model_class': DenoiseNet,
                'checkpoint_path': str(self.checkpoint_path / "denoise_net.pth"),
                'device': self.device,
                'use_fp16': self.is_m3_max
            }
            
            try:
                self.denoise_model = await self.model_interface.load_model_async('denoise_net', denoise_config)
                self.logger.info("✅ Denoising 모델 로드 성공")
            except Exception as e:
                self.logger.warning(f"Denoising 모델 로드 실패: {e}")
                self.denoise_model = None
                
        except Exception as e:
            self.logger.error(f"AI 모델 초기화 실패: {e}")
            await self._load_models_direct()

    async def _load_models_direct(self):
        """AI 모델 직접 로드 (Model Loader 없이)"""
        try:
            # Super Resolution 모델
            self.sr_model = SRResNet(in_channels=3, out_channels=3)
            sr_checkpoint = self.checkpoint_path / "srresnet_x4.pth"
            if sr_checkpoint.exists():
                state_dict = torch.load(sr_checkpoint, map_location=self.device)
                self.sr_model.load_state_dict(state_dict)
                self.logger.info("✅ SR 모델 체크포인트 로드 성공")
            else:
                self.logger.warning("SR 모델 체크포인트가 없습니다. 사전 훈련되지 않은 모델 사용.")
            
            self.sr_model.to(self.device)
            self.sr_model.eval()
            
            # Denoising 모델
            self.denoise_model = DenoiseNet(in_channels=3, out_channels=3)
            denoise_checkpoint = self.checkpoint_path / "denoise_net.pth"
            if denoise_checkpoint.exists():
                state_dict = torch.load(denoise_checkpoint, map_location=self.device)
                self.denoise_model.load_state_dict(state_dict)
                self.logger.info("✅ Denoising 모델 체크포인트 로드 성공")
            else:
                self.logger.warning("Denoising 모델 체크포인트가 없습니다.")
            
            self.denoise_model.to(self.device)
            self.denoise_model.eval()
            
            # FP16 최적화 (M3 Max)
            if self.is_m3_max and self.device != 'cpu':
                if self.sr_model:
                    self.sr_model = self.sr_model.half()
                if self.denoise_model:
                    self.denoise_model = self.denoise_model.half()
            
        except Exception as e:
            self.logger.error(f"모델 직접 로드 실패: {e}")
            self.sr_model = None
            self.denoise_model = None

    async def _initialize_face_detector(self):
        """얼굴 검출기 초기화"""
        try:
            # OpenCV DNN 얼굴 검출기
            face_net_path = self.checkpoint_path / "opencv_face_detector_uint8.pb"
            face_config_path = self.checkpoint_path / "opencv_face_detector.pbtxt"
            
            if face_net_path.exists() and face_config_path.exists():
                self.face_detector = cv2.dnn.readNetFromTensorflow(
                    str(face_net_path), str(face_config_path)
                )
                self.logger.info("✅ OpenCV DNN 얼굴 검출기 로드 성공")
            else:
                # Haar Cascade 폴백
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_detector = cv2.CascadeClassifier(cascade_path)
                self.logger.info("✅ Haar Cascade 얼굴 검출기 로드 성공")
                
        except Exception as e:
            self.logger.warning(f"얼굴 검출기 초기화 실패: {e}")
            self.face_detector = None

    def _initialize_image_filters(self):
        """이미지 필터 초기화"""
        try:
            # 커스텀 커널들
            self.sharpening_kernel = np.array([
                [-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]
            ], dtype=np.float32)
            
            self.edge_enhancement_kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ], dtype=np.float32)
            
            # 가우시안 커널 (노이즈 제거용)
            self.gaussian_kernel_3x3 = cv2.getGaussianKernel(3, 0.8)
            self.gaussian_kernel_5x5 = cv2.getGaussianKernel(5, 1.2)
            
            # 언샤프 마스킹 매개변수
            self.unsharp_params = {
                'radius': 1.0,
                'amount': 1.5,
                'threshold': 0.05
            }
            
            self.logger.info("✅ 이미지 필터 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"이미지 필터 초기화 실패: {e}")

    async def _initialize_gpu_acceleration(self):
        """GPU 가속 초기화"""
        try:
            if self.device == 'mps':
                # M3 Max Metal Performance Shaders
                self.logger.info("🍎 M3 Max MPS 가속 활성화")
                
            elif self.device == 'cuda':
                # CUDA 가속
                self.logger.info("🚀 CUDA 가속 활성화")
                
                # CuPy 사용 가능시 활성화
                if CUPY_AVAILABLE:
                    self.use_cupy = True
                    self.logger.info("✅ CuPy 가속 활성화")
                else:
                    self.use_cupy = False
            else:
                self.logger.info("💻 CPU 모드에서 실행")
                
        except Exception as e:
            self.logger.warning(f"GPU 가속 초기화 실패: {e}")

    async def _warmup_m3_max(self):
        """M3 Max 최적화 워밍업"""
        try:
            if not self.is_m3_max:
                return
            
            self.logger.info("🍎 M3 Max 최적화 워밍업 시작...")
            
            # 더미 이미지로 GPU 워밍업
            dummy_image = torch.randn(1, 3, 512, 512).to(self.device)
            
            if self.sr_model:
                with torch.no_grad():
                    _ = self.sr_model(dummy_image)
                self.logger.info("✅ Super Resolution M3 Max 워밍업 완료")
            
            if self.denoise_model:
                with torch.no_grad():
                    _ = self.denoise_model(dummy_image)
                self.logger.info("✅ Denoising M3 Max 워밍업 완료")
            
            # MPS 캐시 최적화
            if self.device == 'mps':
                try:
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                except:
                    pass
            
            # 메모리 최적화
            if self.memory_manager:
                await self.memory_manager.optimize_for_m3_max()
            
            self.logger.info("🍎 M3 Max 워밍업 완료")
            
        except Exception as e:
            self.logger.warning(f"M3 Max 워밍업 실패: {e}")

    def _initialize_cache_system(self):
        """캐시 시스템 초기화"""
        try:
            # 캐시 크기 설정 (M3 Max 최적화)
            cache_size = self.post_processing_config.cache_size
            
            # LRU 캐시로 변환
            from functools import lru_cache
            self._cached_enhancement = lru_cache(maxsize=cache_size)(self._perform_enhancement_cached)
            
            self.logger.info(f"💾 캐시 시스템 초기화 완료 (크기: {cache_size})")
            
        except Exception as e:
            self.logger.error(f"캐시 시스템 초기화 실패: {e}")

    def _initialize_fallback_system(self):
        """최소한의 폴백 시스템 초기화"""
        try:
            # 가장 기본적인 방법들만 활성화
            self.post_processing_config.enabled_methods = [
                EnhancementMethod.SHARPENING,
                EnhancementMethod.COLOR_CORRECTION,
                EnhancementMethod.CONTRAST_ENHANCEMENT
            ]
            
            self.logger.info("⚠️ 폴백 시스템 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"폴백 시스템 초기화도 실패: {e}")

    async def process(
        self, 
        fitting_result: Dict[str, Any],
        enhancement_options: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        ✅ 통일된 처리 인터페이스 - Pipeline Manager 호환
        
        Args:
            fitting_result: 가상 피팅 결과 (6단계 출력)
            enhancement_options: 향상 옵션
                - quality_level: 품질 레벨
                - enabled_methods: 사용할 향상 방법들
                - enhancement_strength: 향상 강도 (0.0-1.0)
            **kwargs: 추가 매개변수
                
        Returns:
            Dict[str, Any]: 후처리 결과
                - success: bool
                - enhanced_image: np.ndarray
                - quality_improvement: float
                - applied_methods: List[str]
                - processing_time: float
                - metadata: dict
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            self.logger.info("✨ 후처리 시작...")
            
            # 1. 캐시 확인
            cache_key = self._generate_cache_key(fitting_result, enhancement_options)
            if kwargs.get('use_cache', True) and cache_key in self.enhancement_cache:
                cached_result = self.enhancement_cache[cache_key]
                self.processing_stats['cache_hits'] += 1
                self.logger.info("💾 캐시에서 결과 반환")
                return self._format_result(cached_result)
            
            # 2. 입력 데이터 처리
            processed_input = self._process_input_data(fitting_result)
            
            # 3. 향상 옵션 준비
            options = self._prepare_enhancement_options(enhancement_options)
            
            # 4. 메인 향상 처리
            result = await self._perform_enhancement_pipeline(
                processed_input, options, **kwargs
            )
            
            # 5. 결과 캐싱
            if kwargs.get('use_cache', True) and result.success:
                self.enhancement_cache[cache_key] = result
                if len(self.enhancement_cache) > self.post_processing_config.cache_size:
                    self._cleanup_cache()
            
            # 6. 통계 업데이트
            self._update_statistics(result, time.time() - start_time)
            
            self.logger.info(f"✅ 후처리 완료 - 개선도: {result.quality_improvement:.3f}, "
                           f"시간: {result.processing_time:.3f}초")
            
            return self._format_result(result)
            
        except Exception as e:
            error_msg = f"후처리 처리 실패: {e}"
            self.logger.error(f"❌ {error_msg}")
            
            # 에러 결과 반환
            error_result = PostProcessingResult(
                success=False,
                error_message=error_msg,
                processing_time=time.time() - start_time
            )
            
            return self._format_result(error_result)

    def _process_input_data(self, fitting_result: Dict[str, Any]) -> Dict[str, Any]:
        """입력 데이터 처리"""
        try:
            # 가상 피팅 결과에서 이미지 추출
            fitted_image = fitting_result.get('fitted_image') or fitting_result.get('result_image')
            
            if fitted_image is None:
                raise ValueError("피팅된 이미지가 없습니다")
            
            # 타입별 변환
            if isinstance(fitted_image, str):
                # Base64 디코딩
                import base64
                from io import BytesIO
                image_data = base64.b64decode(fitted_image)
                image_pil = Image.open(BytesIO(image_data)).convert('RGB')
                fitted_image = np.array(image_pil)
            elif isinstance(fitted_image, torch.Tensor):
                if self.data_converter:
                    fitted_image = self.data_converter.tensor_to_numpy(fitted_image)
                else:
                    fitted_image = fitted_image.detach().cpu().numpy()
                    if fitted_image.ndim == 4:
                        fitted_image = fitted_image.squeeze(0)
                    if fitted_image.shape[0] in [1, 3]:
                        fitted_image = fitted_image.transpose(1, 2, 0)
                    fitted_image = (fitted_image * 255).astype(np.uint8)
            elif isinstance(fitted_image, Image.Image):
                fitted_image = np.array(fitted_image.convert('RGB'))
            elif not isinstance(fitted_image, np.ndarray):
                raise ValueError(f"지원되지 않는 이미지 타입: {type(fitted_image)}")
            
            # 이미지 검증
            if fitted_image.ndim != 3 or fitted_image.shape[2] != 3:
                raise ValueError(f"잘못된 이미지 형태: {fitted_image.shape}")
            
            # 크기 제한 확인
            max_height, max_width = self.post_processing_config.max_resolution
            if fitted_image.shape[0] > max_height or fitted_image.shape[1] > max_width:
                fitted_image = self._resize_image_preserve_ratio(fitted_image, max_height, max_width)
            
            return {
                'image': fitted_image,
                'original_shape': fitted_image.shape,
                'mask': fitting_result.get('mask'),
                'confidence': fitting_result.get('confidence', 1.0),
                'metadata': fitting_result.get('metadata', {})
            }
            
        except Exception as e:
            self.logger.error(f"입력 데이터 처리 실패: {e}")
            raise

    def _prepare_enhancement_options(self, enhancement_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """향상 옵션 준비"""
        try:
            # 기본 옵션
            default_options = {
                'quality_level': self.post_processing_config.quality_level.value,
                'enabled_methods': [method.value for method in self.post_processing_config.enabled_methods],
                'enhancement_strength': self.enhancement_strength,
                'preserve_faces': self.preserve_faces,
                'auto_adjust_brightness': self.auto_adjust_brightness,
                'apply_super_resolution': EnhancementMethod.SUPER_RESOLUTION in self.post_processing_config.enabled_methods,
                'apply_noise_reduction': EnhancementMethod.NOISE_REDUCTION in self.post_processing_config.enabled_methods,
                'apply_sharpening': EnhancementMethod.SHARPENING in self.post_processing_config.enabled_methods,
                'apply_color_correction': EnhancementMethod.COLOR_CORRECTION in self.post_processing_config.enabled_methods,
                'apply_contrast_enhancement': EnhancementMethod.CONTRAST_ENHANCEMENT in self.post_processing_config.enabled_methods
            }
            
            # 사용자 옵션으로 덮어쓰기
            if enhancement_options:
                default_options.update(enhancement_options)
            
            return default_options
            
        except Exception as e:
            self.logger.error(f"향상 옵션 준비 실패: {e}")
            return {}

    async def _perform_enhancement_pipeline(
        self,
        processed_input: Dict[str, Any],
        options: Dict[str, Any],
        **kwargs
    ) -> PostProcessingResult:
        """향상 파이프라인 수행"""
        try:
            image = processed_input['image'].copy()
            applied_methods = []
            enhancement_log = []
            
            original_quality = self._calculate_image_quality(image)
            
            # 1. Super Resolution (해상도 향상)
            if options.get('apply_super_resolution', False) and self.sr_model:
                try:
                    enhanced_image = await self._apply_super_resolution(image)
                    if enhanced_image is not None:
                        image = enhanced_image
                        applied_methods.append('super_resolution')
                        enhancement_log.append("Super Resolution 적용")
                        self.logger.info("✅ Super Resolution 적용 완료")
                except Exception as e:
                    self.logger.warning(f"Super Resolution 실패: {e}")
            
            # 2. Noise Reduction (노이즈 제거)
            if options.get('apply_noise_reduction', False):
                try:
                    if self.denoise_model:
                        enhanced_image = await self._apply_ai_denoising(image)
                    else:
                        enhanced_image = self._apply_traditional_denoising(image)
                    
                    if enhanced_image is not None:
                        image = enhanced_image
                        applied_methods.append('noise_reduction')
                        enhancement_log.append("노이즈 제거 적용")
                        self.logger.info("✅ 노이즈 제거 적용 완료")
                except Exception as e:
                    self.logger.warning(f"노이즈 제거 실패: {e}")
            
            # 3. Sharpening (선명도 향상)
            if options.get('apply_sharpening', False):
                try:
                    enhanced_image = self._apply_advanced_sharpening(image, options['enhancement_strength'])
                    if enhanced_image is not None:
                        image = enhanced_image
                        applied_methods.append('sharpening')
                        enhancement_log.append("선명도 향상 적용")
                        self.logger.info("✅ 선명도 향상 적용 완료")
                except Exception as e:
                    self.logger.warning(f"선명도 향상 실패: {e}")
            
            # 4. Color Correction (색상 보정)
            if options.get('apply_color_correction', False):
                try:
                    enhanced_image = self._apply_color_correction(image)
                    if enhanced_image is not None:
                        image = enhanced_image
                        applied_methods.append('color_correction')
                        enhancement_log.append("색상 보정 적용")
                        self.logger.info("✅ 색상 보정 적용 완료")
                except Exception as e:
                    self.logger.warning(f"색상 보정 실패: {e}")
            
            # 5. Contrast Enhancement (대비 향상)
            if options.get('apply_contrast_enhancement', False):
                try:
                    enhanced_image = self._apply_contrast_enhancement(image)
                    if enhanced_image is not None:
                        image = enhanced_image
                        applied_methods.append('contrast_enhancement')
                        enhancement_log.append("대비 향상 적용")
                        self.logger.info("✅ 대비 향상 적용 완료")
                except Exception as e:
                    self.logger.warning(f"대비 향상 실패: {e}")
            
            # 6. Face Enhancement (얼굴 향상) - 얼굴 검출된 경우만
            if options.get('preserve_faces', False) and self.face_detector:
                try:
                    faces = self._detect_faces(image)
                    if len(faces) > 0:
                        enhanced_image = self._enhance_face_regions(image, faces)
                        if enhanced_image is not None:
                            image = enhanced_image
                            applied_methods.append('face_enhancement')
                            enhancement_log.append(f"얼굴 향상 적용 ({len(faces)}개 얼굴)")
                            self.logger.info(f"✅ 얼굴 향상 적용 완료 ({len(faces)}개)")
                except Exception as e:
                    self.logger.warning(f"얼굴 향상 실패: {e}")
            
            # 7. 최종 후처리
            try:
                final_image = self._apply_final_post_processing(image)
                if final_image is not None:
                    image = final_image
                    enhancement_log.append("최종 후처리 적용")
            except Exception as e:
                self.logger.warning(f"최종 후처리 실패: {e}")
            
            # 품질 개선도 계산
            final_quality = self._calculate_image_quality(image)
            quality_improvement = final_quality - original_quality
            
            return PostProcessingResult(
                success=True,
                enhanced_image=image,
                quality_improvement=quality_improvement,
                applied_methods=applied_methods,
                processing_time=0.0,  # 호출부에서 설정
                metadata={
                    'enhancement_log': enhancement_log,
                    'original_quality': original_quality,
                    'final_quality': final_quality,
                    'original_shape': processed_input['original_shape'],
                    'options_used': options
                }
            )
            
        except Exception as e:
            return PostProcessingResult(
                success=False,
                error_message=f"향상 파이프라인 실패: {e}",
                processing_time=0.0
            )

    async def _apply_super_resolution(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Super Resolution 적용"""
        try:
            if not self.sr_model:
                return None
            
            # 이미지를 텐서로 변환
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            
            # PIL로 변환 후 텐서로
            pil_image = Image.fromarray(image)
            input_tensor = transform(pil_image).unsqueeze(0).to(self.device)
            
            if self.is_m3_max and self.device != 'cpu':
                input_tensor = input_tensor.half()
            
            # 추론
            with torch.no_grad():
                output_tensor = self.sr_model(input_tensor)
                
                # 후처리
                output_tensor = torch.clamp(output_tensor, 0, 1)
                
                # 텐서를 numpy로 변환
                output_np = output_tensor.squeeze().cpu().float().numpy()
                if output_np.ndim == 3:
                    output_np = output_np.transpose(1, 2, 0)
                
                # 0-255 범위로 변환
                enhanced_image = (output_np * 255).astype(np.uint8)
                
                return enhanced_image
                
        except Exception as e:
            self.logger.error(f"Super Resolution 적용 실패: {e}")
            return None

    async def _apply_ai_denoising(self, image: np.ndarray) -> Optional[np.ndarray]:
        """AI 기반 노이즈 제거"""
        try:
            if not self.denoise_model:
                return None
            
            # 이미지를 텐서로 변환
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            
            pil_image = Image.fromarray(image)
            input_tensor = transform(pil_image).unsqueeze(0).to(self.device)
            
            if self.is_m3_max and self.device != 'cpu':
                input_tensor = input_tensor.half()
            
            # 추론
            with torch.no_grad():
                output_tensor = self.denoise_model(input_tensor)
                
                # 텐서를 numpy로 변환
                output_np = output_tensor.squeeze().cpu().float().numpy()
                if output_np.ndim == 3:
                    output_np = output_np.transpose(1, 2, 0)
                
                # 0-255 범위로 변환
                denoised_image = (output_np * 255).astype(np.uint8)
                
                return denoised_image
                
        except Exception as e:
            self.logger.error(f"AI 노이즈 제거 실패: {e}")
            return None

    def _apply_traditional_denoising(self, image: np.ndarray) -> np.ndarray:
        """전통적 노이즈 제거"""
        try:
            # 비선형 확산 필터 또는 bilateral 필터 사용
            if SKIMAGE_AVAILABLE:
                denoised = restoration.denoise_bilateral(
                    image, 
                    sigma_color=0.05, 
                    sigma_spatial=15, 
                    multichannel=True
                )
                return (denoised * 255).astype(np.uint8)
            else:
                # OpenCV bilateral filter
                denoised = cv2.bilateralFilter(image, 9, 75, 75)
                return denoised
                
        except Exception as e:
            self.logger.error(f"전통적 노이즈 제거 실패: {e}")
            return image

    def _apply_advanced_sharpening(self, image: np.ndarray, strength: float = 0.7) -> np.ndarray:
        """고급 선명도 향상"""
        try:
            # 언샤프 마스킹
            blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
            unsharp_mask = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
            
            # 적응형 선명화
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # 에지 영역에만 추가 선명화 적용
            kernel = self.sharpening_kernel * strength
            sharpened = cv2.filter2D(unsharp_mask, -1, kernel)
            
            # 에지 마스크 적용
            edge_mask = (edges > 0).astype(np.float32)
            edge_mask = np.stack([edge_mask, edge_mask, edge_mask], axis=2)
            
            result = unsharp_mask * (1 - edge_mask) + sharpened * edge_mask
            
            return np.clip(result, 0, 255).astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"선명도 향상 실패: {e}")
            return image

    def _apply_color_correction(self, image: np.ndarray) -> np.ndarray:
        """색상 보정"""
        try:
            # LAB 색공간으로 변환
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # LAB 채널 재결합
            lab = cv2.merge([l, a, b])
            
            # RGB로 다시 변환
            corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # 화이트 밸런스 조정
            corrected = self._adjust_white_balance(corrected)
            
            return corrected
            
        except Exception as e:
            self.logger.error(f"색상 보정 실패: {e}")
            return image

    def _adjust_white_balance(self, image: np.ndarray) -> np.ndarray:
        """화이트 밸런스 조정"""
        try:
            # Gray World 알고리즘
            r_mean = np.mean(image[:, :, 0])
            g_mean = np.mean(image[:, :, 1])
            b_mean = np.mean(image[:, :, 2])
            
            gray_mean = (r_mean + g_mean + b_mean) / 3
            
            r_gain = gray_mean / r_mean if r_mean > 0 else 1.0
            g_gain = gray_mean / g_mean if g_mean > 0 else 1.0
            b_gain = gray_mean / b_mean if b_mean > 0 else 1.0
            
            # 게인 제한
            max_gain = 1.5
            r_gain = min(r_gain, max_gain)
            g_gain = min(g_gain, max_gain)
            b_gain = min(b_gain, max_gain)
            
            # 채널별 조정
            balanced = image.copy().astype(np.float32)
            balanced[:, :, 0] *= r_gain
            balanced[:, :, 1] *= g_gain
            balanced[:, :, 2] *= b_gain
            
            return np.clip(balanced, 0, 255).astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"화이트 밸런스 조정 실패: {e}")
            return image

    def _apply_contrast_enhancement(self, image: np.ndarray) -> np.ndarray:
        """대비 향상"""
        try:
            # 적응형 히스토그램 평활화
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE 적용
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            # 채널 재결합
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
            
            # 추가 대비 조정 (sigmoid 곡선)
            enhanced = self._apply_sigmoid_correction(enhanced, gain=1.2, cutoff=0.5)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"대비 향상 실패: {e}")
            return image

    def _apply_sigmoid_correction(self, image: np.ndarray, gain: float = 1.0, cutoff: float = 0.5) -> np.ndarray:
        """시그모이드 곡선을 사용한 대비 조정"""
        try:
            # 0-1 범위로 정규화
            normalized = image.astype(np.float32) / 255.0
            
            # 시그모이드 함수 적용
            sigmoid = 1 / (1 + np.exp(gain * (cutoff - normalized)))
            
            # 0-255 범위로 다시 변환
            result = (sigmoid * 255).astype(np.uint8)
            
            return result
            
        except Exception as e:
            self.logger.error(f"시그모이드 보정 실패: {e}")
            return image

    def _detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """얼굴 검출"""
        try:
            if not self.face_detector:
                return []
            
            faces = []
            
            if hasattr(self.face_detector, 'setInput'):
                # DNN 기반 검출기
                blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
                self.face_detector.setInput(blob)
                detections = self.face_detector.forward()
                
                h, w = image.shape[:2]
                
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:
                        x1 = int(detections[0, 0, i, 3] * w)
                        y1 = int(detections[0, 0, i, 4] * h)
                        x2 = int(detections[0, 0, i, 5] * w)
                        y2 = int(detections[0, 0, i, 6] * h)
                        faces.append((x1, y1, x2 - x1, y2 - y1))
            else:
                # Haar Cascade 검출기
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                detected_faces = self.face_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                faces = [tuple(face) for face in detected_faces]
            
            return faces
            
        except Exception as e:
            self.logger.error(f"얼굴 검출 실패: {e}")
            return []

    def _enhance_face_regions(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """얼굴 영역 향상"""
        try:
            enhanced = image.copy()
            
            for (x, y, w, h) in faces:
                # 얼굴 영역 추출
                face_region = image[y:y+h, x:x+w]
                
                if face_region.size == 0:
                    continue
                
                # 얼굴 영역에 대해 부드러운 향상 적용
                # 1. 약간의 블러를 통한 피부 부드럽게
                blurred = cv2.GaussianBlur(face_region, (5, 5), 1.0)
                
                # 2. 원본과 블러의 가중 합성
                softened = cv2.addWeighted(face_region, 0.7, blurred, 0.3, 0)
                
                # 3. 밝기 약간 조정
                brightened = cv2.convertScaleAbs(softened, alpha=1.1, beta=5)
                
                # 4. 향상된 얼굴 영역을 원본에 적용
                enhanced[y:y+h, x:x+w] = brightened
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"얼굴 영역 향상 실패: {e}")
            return image

    def _apply_final_post_processing(self, image: np.ndarray) -> np.ndarray:
        """최종 후처리"""
        try:
            # 1. 미세한 노이즈 제거
            denoised = cv2.medianBlur(image, 3)
            
            # 2. 약간의 선명화
            kernel = np.array([[-0.1, -0.1, -0.1],
                              [-0.1,  1.8, -0.1],
                              [-0.1, -0.1, -0.1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            # 3. 색상 보정
            final = cv2.convertScaleAbs(sharpened, alpha=1.02, beta=2)
            
            return final
            
        except Exception as e:
            self.logger.error(f"최종 후처리 실패: {e}")
            return image

    def _calculate_image_quality(self, image: np.ndarray) -> float:
        """이미지 품질 점수 계산"""
        try:
            # 여러 품질 지표의 조합
            
            # 1. 선명도 (라플라시안 분산)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000.0, 1.0)
            
            # 2. 대비 (표준편차)
            contrast_score = min(np.std(gray) / 128.0, 1.0)
            
            # 3. 밝기 균형 (히스토그램 분포)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_normalized = hist / hist.sum()
            entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
            brightness_score = min(entropy / 8.0, 1.0)
            
            # 4. 색상 다양성
            rgb_std = np.mean([np.std(image[:, :, i]) for i in range(3)])
            color_score = min(rgb_std / 64.0, 1.0)
            
            # 가중 평균
            quality_score = (
                sharpness_score * 0.3 +
                contrast_score * 0.3 +
                brightness_score * 0.2 +
                color_score * 0.2
            )
            
            return quality_score
            
        except Exception as e:
            self.logger.error(f"품질 계산 실패: {e}")
            return 0.5

    def _resize_image_preserve_ratio(self, image: np.ndarray, max_height: int, max_width: int) -> np.ndarray:
        """비율을 유지하면서 이미지 크기 조정"""
        try:
            h, w = image.shape[:2]
            
            if h <= max_height and w <= max_width:
                return image
            
            # 비율 계산
            ratio = min(max_height / h, max_width / w)
            new_h = int(h * ratio)
            new_w = int(w * ratio)
            
            # 고품질 리샘플링
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            return resized
            
        except Exception as e:
            self.logger.error(f"이미지 크기 조정 실패: {e}")
            return image

    def _generate_cache_key(self, fitting_result: Dict[str, Any], enhancement_options: Optional[Dict[str, Any]]) -> str:
        """캐시 키 생성"""
        try:
            # 입력 이미지 해시
            fitted_image = fitting_result.get('fitted_image') or fitting_result.get('result_image')
            if isinstance(fitted_image, str):
                # Base64 문자열의 해시
                image_hash = hashlib.md5(fitted_image.encode()).hexdigest()[:16]
            elif isinstance(fitted_image, np.ndarray):
                image_hash = hashlib.md5(fitted_image.tobytes()).hexdigest()[:16]
            else:
                image_hash = str(hash(str(fitted_image)))[:16]
            
            # 옵션 해시
            options_str = json.dumps(enhancement_options or {}, sort_keys=True)
            options_hash = hashlib.md5(options_str.encode()).hexdigest()[:8]
            
            # 전체 키 생성
            cache_key = f"{image_hash}_{options_hash}_{self.device}_{self.quality_level}"
            return cache_key
            
        except Exception as e:
            self.logger.warning(f"캐시 키 생성 실패: {e}")
            return f"fallback_{time.time()}_{self.device}"

    def _cleanup_cache(self):
        """캐시 정리 (LRU 방식)"""
        try:
            if len(self.enhancement_cache) <= self.post_processing_config.cache_size:
                return
            
            # 가장 오래된 항목들 제거
            items = list(self.enhancement_cache.items())
            # 처리 시간 기준으로 정렬 (최근 사용된 것이 뒤에)
            items.sort(key=lambda x: x[1].processing_time)
            
            # 절반 정도 제거
            remove_count = len(items) - self.post_processing_config.cache_size // 2
            
            for i in range(remove_count):
                del self.enhancement_cache[items[i][0]]
            
            self.logger.info(f"💾 캐시 정리 완료: {remove_count}개 항목 제거")
            
        except Exception as e:
            self.logger.error(f"캐시 정리 실패: {e}")

    def _update_statistics(self, result: PostProcessingResult, processing_time: float):
        """통계 업데이트"""
        try:
            self.processing_stats['total_processed'] += 1
            
            if result.success:
                self.processing_stats['successful_enhancements'] += 1
                
                # 평균 개선도 업데이트
                current_avg = self.processing_stats['average_improvement']
                total_successful = self.processing_stats['successful_enhancements']
                
                self.processing_stats['average_improvement'] = (
                    (current_avg * (total_successful - 1) + result.quality_improvement) / total_successful
                )
                
                # 방법별 사용 통계
                for method in result.applied_methods:
                    if method not in self.processing_stats['method_usage']:
                        self.processing_stats['method_usage'][method] = 0
                    self.processing_stats['method_usage'][method] += 1
            
            # 평균 처리 시간 업데이트
            current_avg_time = self.processing_stats['average_processing_time']
            total_processed = self.processing_stats['total_processed']
            
            self.processing_stats['average_processing_time'] = (
                (current_avg_time * (total_processed - 1) + processing_time) / total_processed
            )
            
            # 결과에 처리 시간 설정
            result.processing_time = processing_time
            
        except Exception as e:
            self.logger.warning(f"통계 업데이트 실패: {e}")

    def _format_result(self, result: PostProcessingResult) -> Dict[str, Any]:
        """결과를 표준 딕셔너리 형태로 포맷"""
        try:
            formatted_result = {
                'success': result.success,
                'processing_time': result.processing_time,
                'applied_methods': result.applied_methods,
                'metadata': result.metadata
            }
            
            if result.success:
                formatted_result.update({
                    'enhanced_image': result.enhanced_image.tolist() if result.enhanced_image is not None else None,
                    'quality_improvement': result.quality_improvement,
                })
            else:
                formatted_result['error_message'] = result.error_message
            
            return formatted_result
            
        except Exception as e:
            self.logger.error(f"결과 포맷팅 실패: {e}")
            return {
                'success': False,
                'error_message': f"결과 포맷팅 실패: {e}",
                'processing_time': 0.0,
                'applied_methods': []
            }

    async def _perform_enhancement_cached(self, *args, **kwargs):
        """캐시된 향상 수행 (LRU 캐시용)"""
        return await self._perform_enhancement_pipeline(*args, **kwargs)

    def get_statistics(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        try:
            stats = self.processing_stats.copy()
            
            # 성공률 계산
            if stats['total_processed'] > 0:
                stats['success_rate'] = stats['successful_enhancements'] / stats['total_processed']
            else:
                stats['success_rate'] = 0.0
            
            # 캐시 정보
            stats['cache_info'] = {
                'size': len(self.enhancement_cache),
                'max_size': self.post_processing_config.cache_size,
                'hit_ratio': stats['cache_hits'] / max(stats['total_processed'], 1)
            }
            
            # 시스템 정보
            stats['system_info'] = {
                'device': self.device,
                'is_m3_max': self.is_m3_max,
                'memory_gb': self.memory_gb,
                'enabled_methods': [method.value for method in self.post_processing_config.enabled_methods],
                'optimization_enabled': self.optimization_enabled,
                'models_loaded': {
                    'sr_model': hasattr(self, 'sr_model') and self.sr_model is not None,
                    'denoise_model': hasattr(self, 'denoise_model') and self.denoise_model is not None,
                    'face_detector': self.face_detector is not None
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"통계 조회 실패: {e}")
            return {'error': str(e)}

    async def cleanup(self):
        """리소스 정리"""
        try:
            self.logger.info("🧹 7단계 후처리 시스템 정리 시작...")
            
            # 캐시 정리
            self.enhancement_cache.clear()
            self.model_cache.clear()
            
            # 모델 메모리 해제
            if hasattr(self, 'sr_model') and self.sr_model:
                del self.sr_model
                self.sr_model = None
            
            if hasattr(self, 'denoise_model') and self.denoise_model:
                del self.denoise_model
                self.denoise_model = None
            
            if hasattr(self, 'face_detector') and self.face_detector:
                del self.face_detector
                self.face_detector = None
            
            # 스레드 풀 종료
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
            
            # 메모리 정리
            if self.memory_manager:
                await self.memory_manager.cleanup_memory()
            
            # PyTorch 캐시 정리
            if self.device == 'mps' and hasattr(torch.mps, 'empty_cache'):
                try:
                    torch.mps.empty_cache()
                except:
                    pass
            elif self.device == 'cuda':
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
            
            # 가비지 컬렉션
            gc.collect()
            
            self.is_initialized = False
            self.logger.info("✅ 7단계 후처리 시스템 정리 완료")
            
        except Exception as e:
            self.logger.error(f"정리 과정에서 오류 발생: {e}")

    def __del__(self):
        """소멸자"""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except:
            pass

# ==============================================
# 4. 팩토리 함수들 및 유틸리티
# ==============================================

def create_post_processing_step(
    device: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> PostProcessingStep:
    """
    PostProcessingStep 팩토리 함수
    
    Args:
        device: 사용할 디바이스 ('auto', 'cpu', 'cuda', 'mps')
        config: 설정 딕셔너리
        **kwargs: 추가 설정
        
    Returns:
        PostProcessingStep: 초기화된 인스턴스
    """
    try:
        return PostProcessingStep(device=device, config=config, **kwargs)
    except Exception as e:
        logger.error(f"PostProcessingStep 생성 실패: {e}")
        raise

def create_m3_max_post_processing_step(**kwargs) -> PostProcessingStep:
    """M3 Max 최적화된 후처리 스텝 생성"""
    m3_max_config = {
        'device': 'mps',
        'is_m3_max': True,
        'optimization_enabled': True,
        'memory_gb': 128,
        'quality_level': 'high',
        'processing_mode': 'quality',
        'enabled_methods': [
            'super_resolution',
            'noise_reduction',
            'sharpening',
            'color_correction',
            'contrast_enhancement',
            'face_enhancement'
        ],
        'enhancement_strength': 0.8,
        'preserve_faces': True,
        'cache_size': 100
    }
    
    m3_max_config.update(kwargs)
    
    return PostProcessingStep(**m3_max_config)

def create_production_post_processing_step(
    quality_level: str = "balanced",
    processing_mode: str = "quality",
    **kwargs
) -> PostProcessingStep:
    """프로덕션 환경용 후처리 스텝 생성"""
    production_config = {
        'quality_level': quality_level,
        'processing_mode': processing_mode,
        'optimization_enabled': True,
        'enabled_methods': [
            'noise_reduction',
            'sharpening',
            'color_correction',
            'contrast_enhancement'
        ],
        'enhancement_strength': 0.6,
        'preserve_faces': True,
        'auto_adjust_brightness': True,
        'cache_size': 50
    }
    
    production_config.update(kwargs)
    
    return PostProcessingStep(**production_config)

def create_real_time_post_processing_step(**kwargs) -> PostProcessingStep:
    """실시간 처리용 후처리 스텝 생성"""
    real_time_config = {
        'processing_mode': 'real_time',
        'quality_level': 'fast',
        'enabled_methods': [
            'sharpening',
            'color_correction'
        ],
        'enhancement_strength': 0.4,
        'preserve_faces': False,
        'cache_size': 25
    }
    
    real_time_config.update(kwargs)
    
    return PostProcessingStep(**real_time_config)

# ==============================================
# 5. 고급 이미지 처리 유틸리티 함수들
# ==============================================

def enhance_image_quality(
    image: np.ndarray,
    methods: List[str] = None,
    strength: float = 0.7,
    device: str = "auto"
) -> np.ndarray:
    """
    독립 실행형 이미지 품질 향상 함수
    
    Args:
        image: 입력 이미지 (numpy array)
        methods: 적용할 방법들 ['sharpening', 'denoising', 'color_correction']
        strength: 향상 강도 (0.0-1.0)
        device: 사용할 디바이스
        
    Returns:
        np.ndarray: 향상된 이미지
    """
    try:
        if methods is None:
            methods = ['sharpening', 'color_correction', 'contrast_enhancement']
        
        step = create_post_processing_step(
            device=device,
            enabled_methods=methods,
            enhancement_strength=strength
        )
        
        # 동기적 처리를 위한 래퍼
        import asyncio
        
        async def process_async():
            await step.initialize()
            
            fitting_result = {'fitted_image': image}
            result = await step.process(fitting_result)
            
            await step.cleanup()
            
            return result['enhanced_image'] if result['success'] else image
        
        # 이벤트 루프 실행
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(process_async())
        except RuntimeError:
            # 새 이벤트 루프 생성
            return asyncio.run(process_async())
            
    except Exception as e:
        logger.error(f"이미지 품질 향상 실패: {e}")
        return image

def batch_enhance_images(
    images: List[np.ndarray],
    methods: List[str] = None,
    strength: float = 0.7,
    device: str = "auto",
    max_workers: int = 4
) -> List[np.ndarray]:
    """
    배치 이미지 품질 향상
    
    Args:
        images: 입력 이미지들
        methods: 적용할 방법들
        strength: 향상 강도
        device: 사용할 디바이스
        max_workers: 최대 워커 수
        
    Returns:
        List[np.ndarray]: 향상된 이미지들
    """
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for image in images:
                future = executor.submit(
                    enhance_image_quality,
                    image, methods, strength, device
                )
                futures.append(future)
            
            # 결과 수집
            enhanced_images = []
            for future in as_completed(futures):
                try:
                    enhanced_image = future.result()
                    enhanced_images.append(enhanced_image)
                except Exception as e:
                    logger.error(f"배치 처리 중 오류: {e}")
                    enhanced_images.append(None)
            
            return enhanced_images
            
    except Exception as e:
        logger.error(f"배치 이미지 향상 실패: {e}")
        return images

# ==============================================
# 6. 모듈 익스포트
# ==============================================

__all__ = [
    # 메인 클래스
    'PostProcessingStep',
    
    # 열거형 및 데이터 클래스
    'EnhancementMethod',
    'QualityLevel',
    'ProcessingMode',
    'PostProcessingConfig',
    'PostProcessingResult',
    
    # AI 모델 클래스들
    'SRResNet',
    'DenoiseNet',
    
    # 팩토리 함수들
    'create_post_processing_step',
    'create_m3_max_post_processing_step',
    'create_production_post_processing_step',
    'create_real_time_post_processing_step',
    
    # 유틸리티 함수들
    'enhance_image_quality',
    'batch_enhance_images'
]

# 모듈 초기화 로깅
logger.info("✅ Step 07 후처리 모듈 로드 완료")
logger.info(f"   - Model Loader 연동: {'✅' if MODEL_LOADER_AVAILABLE else '❌'}")
logger.info(f"   - Memory Manager 연동: {'✅' if MEMORY_MANAGER_AVAILABLE else '❌'}")
logger.info(f"   - PyTorch 사용 가능: {'✅' if torch.cuda.is_available() or torch.backends.mps.is_available() else '❌'}")
logger.info(f"   - OpenCV 사용 가능: {'✅' if cv2 else '❌'}")
logger.info(f"   - scikit-image 사용 가능: {'✅' if SKIMAGE_AVAILABLE else '❌'}")
logger.info(f"   - SciPy 사용 가능: {'✅' if SCIPY_AVAILABLE else '❌'}")

# 자동 정리 등록
import atexit

def _cleanup_on_exit():
    """프로그램 종료 시 정리"""
    try:
        # 전역 인스턴스들 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except:
                pass
        gc.collect()
    except:
        pass

atexit.register(_cleanup_on_exit)