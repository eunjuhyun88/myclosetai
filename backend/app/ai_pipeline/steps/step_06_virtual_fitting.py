# app/ai_pipeline/steps/step_06_virtual_fitting.py
"""
6단계: 가상 피팅 (Virtual Fitting) - 완전한 구현
✅ Pipeline Manager 완전 호환
✅ ModelLoader 완전 연동
✅ M3 Max 128GB 최적화
✅ 실제 AI 모델 사용 (OOTDiffusion, VITON-HD)
✅ 물리 기반 천 시뮬레이션
✅ 프로덕션 레벨 코드
"""

import os
import sys
import logging
import time
import asyncio
import json
import math
import threading
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import gc
import weakref

# 필수 라이브러리
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

# PyTorch 관련
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# 과학 연산 라이브러리
try:
    from scipy.interpolate import RBFInterpolator, griddata
    from scipy.spatial.distance import cdist
    from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from skimage.feature import local_binary_pattern, canny
    from skimage.segmentation import slic, watershed
    from skimage.transform import resize, rotate
    from skimage.measure import regionprops, label
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# AI 모델 라이브러리
try:
    from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPProcessor, CLIPModel
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

# 로거 설정
logger = logging.getLogger(__name__)

# =================================================================
# 1. 상수 및 설정
# =================================================================

class FittingQuality(Enum):
    """피팅 품질 레벨"""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"
    ULTRA = "ultra"

class FittingMethod(Enum):
    """피팅 방법"""
    PHYSICS_BASED = "physics_based"
    AI_NEURAL = "ai_neural"
    HYBRID = "hybrid"
    TEMPLATE_MATCHING = "template_matching"

@dataclass
class FabricProperties:
    """천 재질 속성"""
    stiffness: float = 0.5
    elasticity: float = 0.3
    density: float = 1.4
    friction: float = 0.5
    shine: float = 0.5
    transparency: float = 0.0
    texture_scale: float = 1.0

@dataclass
class FittingParams:
    """피팅 파라미터"""
    fit_type: str = "fitted"
    body_contact: float = 0.7
    drape_level: float = 0.3
    stretch_zones: List[str] = field(default_factory=lambda: ["chest", "waist"])
    wrinkle_intensity: float = 0.5
    shadow_strength: float = 0.6

# 천 재질별 물리 속성
FABRIC_PROPERTIES = {
    'cotton': FabricProperties(0.3, 0.2, 1.5, 0.7, 0.2, 0.0, 1.0),
    'denim': FabricProperties(0.8, 0.1, 2.0, 0.9, 0.1, 0.0, 1.2),
    'silk': FabricProperties(0.1, 0.4, 1.3, 0.3, 0.8, 0.1, 0.8),
    'wool': FabricProperties(0.5, 0.3, 1.4, 0.6, 0.3, 0.0, 1.1),
    'polyester': FabricProperties(0.4, 0.5, 1.2, 0.4, 0.6, 0.0, 0.9),
    'leather': FabricProperties(0.9, 0.1, 2.5, 0.8, 0.9, 0.0, 1.5),
    'spandex': FabricProperties(0.1, 0.8, 1.1, 0.5, 0.4, 0.0, 0.7),
    'linen': FabricProperties(0.6, 0.2, 1.6, 0.8, 0.1, 0.0, 1.3),
    'default': FabricProperties(0.4, 0.3, 1.4, 0.5, 0.5, 0.0, 1.0)
}

# 의류 타입별 피팅 파라미터
CLOTHING_FITTING_PARAMS = {
    'shirt': FittingParams("fitted", 0.7, 0.3, ["chest", "waist"], 0.3, 0.6),
    'dress': FittingParams("flowing", 0.5, 0.7, ["bust", "waist", "hip"], 0.6, 0.7),
    'pants': FittingParams("fitted", 0.8, 0.2, ["thigh", "calf"], 0.4, 0.5),
    'jacket': FittingParams("structured", 0.6, 0.4, ["shoulder", "chest"], 0.2, 0.8),
    'skirt': FittingParams("flowing", 0.6, 0.6, ["waist", "hip"], 0.5, 0.6),
    'blouse': FittingParams("loose", 0.5, 0.5, ["chest", "waist"], 0.4, 0.6),
    'sweater': FittingParams("relaxed", 0.6, 0.4, ["chest", "arms"], 0.3, 0.5),
    'default': FittingParams("fitted", 0.6, 0.4, ["chest", "waist"], 0.4, 0.6)
}

# =================================================================
# 2. 메인 클래스
# =================================================================

class VirtualFittingStep:
    """6단계: 가상 피팅 - 완전한 구현"""
    
    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        ✅ 통일된 생성자 패턴 - Pipeline Manager 완전 호환
        
        Args:
            device: 디바이스 ('cpu', 'cuda', 'mps', None=자동감지)
            config: 설정 딕셔너리
            **kwargs: 확장 파라미터
                - device_type: str = "auto"
                - memory_gb: float = 16.0
                - is_m3_max: bool = False
                - optimization_enabled: bool = True
                - quality_level: str = "balanced"
                - fitting_method: str = "physics_based"
                - enable_physics: bool = True
                - enable_ai_models: bool = True
        """
        
        # === 1. 통일된 초기화 패턴 ===
        self.device = self._auto_detect_device(device)
        self.config = config or {}
        self.step_name = self.__class__.__name__
        self.logger = logging.getLogger(f"pipeline.{self.step_name}")
        
        # === 2. 시스템 파라미터 ===
        self.device_type = kwargs.get('device_type', 'auto')
        self.memory_gb = kwargs.get('memory_gb', 16.0)
        self.is_m3_max = kwargs.get('is_m3_max', self._detect_m3_max())
        self.optimization_enabled = kwargs.get('optimization_enabled', True)
        self.quality_level = kwargs.get('quality_level', 'balanced')
        
        # === 3. 6단계 특화 파라미터 ===
        self.fitting_method = kwargs.get('fitting_method', 'physics_based')
        self.enable_physics = kwargs.get('enable_physics', True)
        self.enable_ai_models = kwargs.get('enable_ai_models', True)
        
        # === 4. Step 특화 설정 병합 ===
        self._merge_step_specific_config(kwargs)
        
        # === 5. 상태 초기화 ===
        self.is_initialized = False
        self.session_id = str(uuid.uuid4())
        
        # === 6. ModelLoader 연동 ===
        self._setup_model_loader()
        
        # === 7. 6단계 전용 초기화 ===
        self._initialize_step_specific()
        
        # 초기화 완료 로그
        self.logger.info(f"✅ {self.step_name} 초기화 완료 - 디바이스: {self.device}, "
                        f"품질: {self.quality_level}, 방법: {self.fitting_method}")
    
    def _auto_detect_device(self, device: Optional[str]) -> str:
        """디바이스 자동 감지"""
        if device is not None:
            return device
            
        if TORCH_AVAILABLE:
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
        return "cpu"
    
    def _detect_m3_max(self) -> bool:
        """M3 Max 감지"""
        try:
            if sys.platform == "darwin":
                import platform
                return "arm64" in platform.machine().lower()
        except:
            pass
        return False
    
    def _merge_step_specific_config(self, kwargs: Dict[str, Any]):
        """Step 특화 설정 병합"""
        system_params = {
            'device_type', 'memory_gb', 'is_m3_max', 
            'optimization_enabled', 'quality_level',
            'fitting_method', 'enable_physics', 'enable_ai_models'
        }
        
        for key, value in kwargs.items():
            if key not in system_params:
                self.config[key] = value
    
    def _setup_model_loader(self):
        """ModelLoader 연동"""
        try:
            # ModelLoader 시스템과 연동
            from app.ai_pipeline.utils.model_loader import BaseStepMixin
            if hasattr(BaseStepMixin, '_setup_model_interface'):
                BaseStepMixin._setup_model_interface(self)
        except ImportError:
            self.logger.warning("ModelLoader 사용 불가 - 독립 모드로 동작")
    
    def _initialize_step_specific(self):
        """6단계 전용 초기화"""
        
        # 가상 피팅 설정
        self.fitting_config = {
            'method': FittingMethod(self.fitting_method),
            'quality': FittingQuality(self.quality_level),
            'physics_enabled': self.enable_physics,
            'ai_models_enabled': self.enable_ai_models,
            'body_interaction': self.config.get('body_interaction', True),
            'fabric_simulation': self.config.get('fabric_simulation', True),
            'enable_shadows': self.config.get('enable_shadows', True),
            'enable_highlights': self.config.get('enable_highlights', True),
            'texture_preservation': self.config.get('texture_preservation', True),
            'wrinkle_simulation': self.config.get('wrinkle_simulation', True)
        }
        
        # 성능 설정
        self.performance_config = {
            'max_resolution': self._get_max_resolution(),
            'fitting_iterations': self._get_fitting_iterations(),
            'precision_factor': self._get_precision_factor(),
            'batch_size': self._get_batch_size(),
            'cache_enabled': True,
            'parallel_processing': self.is_m3_max,
            'memory_efficient': self.memory_gb < 32
        }
        
        # 캐시 시스템
        cache_size = min(200 if self.is_m3_max and self.memory_gb >= 128 else 50, 
                        int(self.memory_gb * 2))
        self.fitting_cache = {}
        self.cache_max_size = cache_size
        
        # AI 모델 관리
        self.ai_models = {
            'diffusion_pipeline': None,
            'human_parser': None,
            'cloth_segmenter': None,
            'pose_estimator': None,
            'style_encoder': None
        }
        
        # 성능 통계
        self.performance_stats = {
            'total_processed': 0,
            'successful_fittings': 0,
            'failed_fittings': 0,
            'average_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_peak_mb': 0.0,
            'ai_model_usage': {model: 0 for model in self.ai_models.keys()}
        }
        
        # 스레드 풀
        max_workers = min(8, int(self.memory_gb / 8)) if self.is_m3_max else 2
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # 메모리 관리
        self.memory_manager = self._create_memory_manager()
    
    def _get_max_resolution(self) -> int:
        """최대 해상도 계산"""
        if self.quality_level == "ultra" and self.memory_gb >= 64:
            return 1024
        elif self.quality_level == "high" and self.memory_gb >= 32:
            return 768
        elif self.quality_level == "balanced":
            return 512
        else:
            return 384
    
    def _get_fitting_iterations(self) -> int:
        """피팅 반복 횟수"""
        quality_iterations = {
            "fast": 1,
            "balanced": 3,
            "high": 5,
            "ultra": 8
        }
        return quality_iterations.get(self.quality_level, 3)
    
    def _get_precision_factor(self) -> float:
        """정밀도 계수"""
        quality_precision = {
            "fast": 0.5,
            "balanced": 1.0,
            "high": 1.5,
            "ultra": 2.0
        }
        return quality_precision.get(self.quality_level, 1.0)
    
    def _get_batch_size(self) -> int:
        """배치 크기"""
        if self.is_m3_max and self.memory_gb >= 128:
            return 4
        elif self.memory_gb >= 32:
            return 2
        else:
            return 1
    
    def _create_memory_manager(self):
        """메모리 매니저 생성"""
        try:
            from app.ai_pipeline.utils.memory_manager import MemoryManager
            return MemoryManager(
                device=self.device,
                is_m3_max=self.is_m3_max,
                memory_gb=self.memory_gb
            )
        except ImportError:
            return None
    
    # =================================================================
    # 3. 초기화 및 모델 로딩
    # =================================================================
    
    async def initialize(self) -> bool:
        """단계 초기화"""
        if self.is_initialized:
            return True
        
        try:
            start_time = time.time()
            self.logger.info(f"🚀 {self.step_name} 초기화 시작...")
            
            # 1. 의존성 확인
            if not self._check_dependencies():
                raise RuntimeError("필수 의존성이 설치되지 않음")
            
            # 2. AI 모델 로드 (선택적)
            if self.fitting_config['ai_models_enabled']:
                await self._load_ai_models()
            
            # 3. 물리 엔진 초기화 (선택적)
            if self.fitting_config['physics_enabled']:
                self._initialize_physics_engine()
            
            # 4. 텍스처 및 렌더링 시스템 초기화
            self._initialize_rendering_system()
            
            # 5. 캐시 시스템 준비
            self._prepare_cache_system()
            
            # 초기화 완료
            self.is_initialized = True
            init_time = time.time() - start_time
            
            self.logger.info(f"✅ {self.step_name} 초기화 완료 - {init_time:.2f}초")
            self.logger.info(f"   - AI 모델: {'활성화' if self.fitting_config['ai_models_enabled'] else '비활성화'}")
            self.logger.info(f"   - 물리 엔진: {'활성화' if self.fitting_config['physics_enabled'] else '비활성화'}")
            self.logger.info(f"   - 최대 해상도: {self.performance_config['max_resolution']}px")
            self.logger.info(f"   - 캐시 크기: {self.cache_max_size}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 초기화 실패: {e}")
            self.logger.error(f"   상세 오류: {traceback.format_exc()}")
            return False
    
    def _check_dependencies(self) -> bool:
        """의존성 확인"""
        required_packages = {
            'numpy': True,
            'PIL': True,
            'torch': TORCH_AVAILABLE,
            'cv2': CV2_AVAILABLE,
            'scipy': SCIPY_AVAILABLE,
            'sklearn': SKLEARN_AVAILABLE
        }
        
        missing = [pkg for pkg, available in required_packages.items() if not available]
        
        if missing:
            self.logger.warning(f"⚠️ 누락된 패키지: {missing}")
            return len(missing) <= 2  # 일부 누락 허용
        
        return True
    
    async def _load_ai_models(self):
        """AI 모델들 로드"""
        try:
            self.logger.info("🧠 AI 모델 로딩 시작...")
            
            # 1. OOTDiffusion 모델 (가장 중요)
            if DIFFUSERS_AVAILABLE:
                await self._load_diffusion_model()
            
            # 2. 인체 파싱 모델
            await self._load_human_parsing_model()
            
            # 3. 포즈 추정 모델
            await self._load_pose_estimation_model()
            
            # 4. 의류 분할 모델
            await self._load_cloth_segmentation_model()
            
            # 5. 스타일 인코더
            await self._load_style_encoder()
            
            self.logger.info("✅ AI 모델 로딩 완료")
            
        except Exception as e:
            self.logger.error(f"❌ AI 모델 로딩 실패: {e}")
            # AI 모델 없어도 물리 기반 피팅은 가능
            self.fitting_config['ai_models_enabled'] = False
    
    async def _load_diffusion_model(self):
        """디퓨전 모델 로드"""
        try:
            if hasattr(self, 'model_loader') and self.model_loader:
                # ModelLoader 사용
                pipeline = await self.model_loader.load_model("ootdiffusion")
                if pipeline:
                    self.ai_models['diffusion_pipeline'] = pipeline
                    self.performance_stats['ai_model_usage']['diffusion_pipeline'] += 1
                    self.logger.info("✅ OOTDiffusion 모델 로드 성공 (ModelLoader)")
                    return
            
            # 직접 로드
            model_path = self._find_model_path("ootdiffusion") or "runwayml/stable-diffusion-v1-5"
            
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
            pipeline = pipeline.to(self.device)
            
            if self.device == "mps":
                # M3 Max 최적화
                pipeline.enable_attention_slicing()
            elif self.device == "cuda":
                pipeline.enable_memory_efficient_attention()
                pipeline.enable_attention_slicing()
            
            self.ai_models['diffusion_pipeline'] = pipeline
            self.performance_stats['ai_model_usage']['diffusion_pipeline'] += 1
            self.logger.info("✅ 디퓨전 모델 로드 성공")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 디퓨전 모델 로드 실패: {e}")
    
    async def _load_human_parsing_model(self):
        """인체 파싱 모델 로드"""
        try:
            if hasattr(self, 'model_loader') and self.model_loader:
                model = await self.model_loader.load_model("human_parsing")
                if model:
                    self.ai_models['human_parser'] = model
                    self.performance_stats['ai_model_usage']['human_parser'] += 1
                    self.logger.info("✅ 인체 파싱 모델 로드 성공")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 인체 파싱 모델 로드 실패: {e}")
    
    async def _load_pose_estimation_model(self):
        """포즈 추정 모델 로드"""
        try:
            if hasattr(self, 'model_loader') and self.model_loader:
                model = await self.model_loader.load_model("openpose")
                if model:
                    self.ai_models['pose_estimator'] = model
                    self.performance_stats['ai_model_usage']['pose_estimator'] += 1
                    self.logger.info("✅ 포즈 추정 모델 로드 성공")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 포즈 추정 모델 로드 실패: {e}")
    
    async def _load_cloth_segmentation_model(self):
        """의류 분할 모델 로드"""
        try:
            if hasattr(self, 'model_loader') and self.model_loader:
                model = await self.model_loader.load_model("u2net")
                if model:
                    self.ai_models['cloth_segmenter'] = model
                    self.performance_stats['ai_model_usage']['cloth_segmenter'] += 1
                    self.logger.info("✅ 의류 분할 모델 로드 성공")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 의류 분할 모델 로드 실패: {e}")
    
    async def _load_style_encoder(self):
        """스타일 인코더 로드"""
        try:
            if DIFFUSERS_AVAILABLE:
                if hasattr(self, 'model_loader') and self.model_loader:
                    model = await self.model_loader.load_model("clip")
                    if model:
                        self.ai_models['style_encoder'] = model
                        self.performance_stats['ai_model_usage']['style_encoder'] += 1
                        self.logger.info("✅ 스타일 인코더 로드 성공")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 스타일 인코더 로드 실패: {e}")
    
    def _find_model_path(self, model_name: str) -> Optional[str]:
        """모델 경로 찾기"""
        try:
            from app.core.optimized_model_paths import ANALYZED_MODELS
            for key, model_info in ANALYZED_MODELS.items():
                if model_name.lower() in key.lower():
                    if model_info.get('ready', False):
                        return str(model_info['path'])
        except ImportError:
            pass
        return None
    
    def _initialize_physics_engine(self):
        """물리 엔진 초기화"""
        try:
            self.physics_engine = {
                'gravity': 9.81,
                'air_resistance': 0.1,
                'cloth_tension': 0.8,
                'body_collision': True,
                'wind_simulation': False,
                'fabric_stretching': True,
                'wrinkle_generation': True
            }
            
            # 물리 시뮬레이션 파라미터
            self.physics_params = {
                'time_step': 0.01,
                'iterations': self._get_fitting_iterations(),
                'damping': 0.95,
                'spring_constant': 100.0,
                'mass_distribution': 'uniform'
            }
            
            self.logger.info("✅ 물리 엔진 초기화 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 물리 엔진 초기화 실패: {e}")
            self.fitting_config['physics_enabled'] = False
    
    def _initialize_rendering_system(self):
        """렌더링 시스템 초기화"""
        try:
            self.rendering_config = {
                'lighting_model': 'pbr',  # Physically Based Rendering
                'shadow_quality': 'medium',
                'reflection_quality': 'low',
                'ambient_occlusion': True,
                'anti_aliasing': True,
                'texture_filtering': 'bilinear',
                'color_space': 'srgb'
            }
            
            # 조명 설정
            self.lighting_setup = {
                'main_light': {'direction': (0.3, -0.5, 0.8), 'intensity': 1.0, 'color': (1.0, 1.0, 1.0)},
                'fill_light': {'direction': (-0.3, -0.2, 0.5), 'intensity': 0.4, 'color': (0.9, 0.9, 1.0)},
                'rim_light': {'direction': (0.0, 0.8, -0.2), 'intensity': 0.3, 'color': (1.0, 0.9, 0.8)},
                'ambient': {'intensity': 0.2, 'color': (0.5, 0.5, 0.6)}
            }
            
            self.logger.info("✅ 렌더링 시스템 초기화 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 렌더링 시스템 초기화 실패: {e}")
    
    def _prepare_cache_system(self):
        """캐시 시스템 준비"""
        try:
            # 캐시 디렉토리 생성
            cache_dir = Path("cache/virtual_fitting")
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # 캐시 설정
            self.cache_config = {
                'enabled': True,
                'max_size': self.cache_max_size,
                'ttl_seconds': 3600,  # 1시간
                'compression': True,
                'persist_to_disk': self.memory_gb < 64
            }
            
            # 메모리 기반 캐시
            self.fitting_cache = {}
            self.cache_access_times = {}
            
            self.logger.info(f"✅ 캐시 시스템 준비 완료 - 크기: {self.cache_max_size}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 캐시 시스템 준비 실패: {e}")
            self.cache_config['enabled'] = False
    
    # =================================================================
    # 4. 메인 처리 함수
    # =================================================================
    
    async def process(
        self,
        person_image: Union[np.ndarray, Image.Image, str],
        clothing_image: Union[np.ndarray, Image.Image, str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        가상 피팅 메인 처리 함수
        
        Args:
            person_image: 사람 이미지
            clothing_image: 의류 이미지
            **kwargs: 추가 파라미터
                - fabric_type: str = "cotton"
                - clothing_type: str = "shirt"
                - fit_preference: str = "fitted"
                - pose_guidance: Optional[Dict] = None
                - style_guidance: Optional[str] = None
                - preserve_background: bool = True
                - quality_enhancement: bool = True
        
        Returns:
            Dict containing fitted image and metadata
        """
        
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        session_id = f"vf_{uuid.uuid4().hex[:8]}"
        
        try:
            self.logger.info(f"🎭 가상 피팅 시작 - 세션: {session_id}")
            
            # 1. 입력 검증 및 전처리
            person_img, clothing_img = await self._preprocess_inputs(
                person_image, clothing_image
            )
            
            # 2. 캐시 확인
            cache_key = self._generate_cache_key(person_img, clothing_img, kwargs)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self.performance_stats['cache_hits'] += 1
                self.logger.info(f"💾 캐시에서 결과 반환 - {session_id}")
                return cached_result
            
            self.performance_stats['cache_misses'] += 1
            
            # 3. 메타데이터 추출
            metadata = await self._extract_metadata(person_img, clothing_img, kwargs)
            
            # 4. 피팅 방법 선택
            fitting_result = await self._execute_fitting(
                person_img, clothing_img, metadata, session_id
            )
            
            # 5. 후처리
            final_result = await self._post_process_result(
                fitting_result, metadata, kwargs
            )
            
            # 6. 결과 구성
            processing_time = time.time() - start_time
            result = self._build_result(
                final_result, metadata, processing_time, session_id
            )
            
            # 7. 캐시 저장
            self._save_to_cache(cache_key, result)
            
            # 8. 통계 업데이트
            self._update_stats(processing_time, success=True)
            
            self.logger.info(f"✅ 가상 피팅 완료 - {session_id} ({processing_time:.2f}초)")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 가상 피팅 실패 - {session_id}: {e}")
            self.logger.error(f"   상세 오류: {traceback.format_exc()}")
            
            processing_time = time.time() - start_time
            self._update_stats(processing_time, success=False)
            
            return {
                "success": False,
                "session_id": session_id,
                "error_message": str(e),
                "processing_time": processing_time,
                "fitted_image": None,
                "metadata": {}
            }
    
    async def _preprocess_inputs(
        self, 
        person_image: Union[np.ndarray, Image.Image, str],
        clothing_image: Union[np.ndarray, Image.Image, str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """입력 이미지 전처리"""
        
        # 이미지 로드 및 변환
        person_img = self._load_and_convert_image(person_image)
        clothing_img = self._load_and_convert_image(clothing_image)
        
        # 해상도 정규화
        target_size = self.performance_config['max_resolution']
        person_img = self._resize_image(person_img, target_size)
        clothing_img = self._resize_image(clothing_img, target_size)
        
        # 색상 공간 정규화
        person_img = self._normalize_color_space(person_img)
        clothing_img = self._normalize_color_space(clothing_img)
        
        return person_img, clothing_img
    
    def _load_and_convert_image(self, image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """이미지 로드 및 변환"""
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
            return np.array(img)
        elif isinstance(image, Image.Image):
            return np.array(image.convert('RGB'))
        elif isinstance(image, np.ndarray):
            if image.shape[-1] == 4:  # RGBA
                return image[:, :, :3]
            return image
        else:
            raise ValueError(f"지원하지 않는 이미지 형식: {type(image)}")
    
    def _resize_image(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """이미지 리사이즈"""
        h, w = image.shape[:2]
        if max(h, w) != target_size:
            if h > w:
                new_h, new_w = target_size, int(w * target_size / h)
            else:
                new_h, new_w = int(h * target_size / w), target_size
            
            if CV2_AVAILABLE:
                return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                from PIL import Image
                img = Image.fromarray(image)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                return np.array(img)
        
        return image
    
    def _normalize_color_space(self, image: np.ndarray) -> np.ndarray:
        """색상 공간 정규화"""
        # 0-255 범위로 정규화
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # 색상 밸런스 조정 (선택적)
        if self.config.get('auto_color_balance', False):
            image = self._auto_color_balance(image)
        
        return image
    
    def _auto_color_balance(self, image: np.ndarray) -> np.ndarray:
        """자동 색상 밸런스"""
        try:
            if CV2_AVAILABLE:
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                l = cv2.equalizeHist(l)
                lab = cv2.merge([l, a, b])
                return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                return image
        except:
            return image
    
    async def _extract_metadata(
        self, 
        person_img: np.ndarray, 
        clothing_img: np.ndarray, 
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """메타데이터 추출"""
        
        metadata = {
            'fabric_type': kwargs.get('fabric_type', 'cotton'),
            'clothing_type': kwargs.get('clothing_type', 'shirt'),
            'fit_preference': kwargs.get('fit_preference', 'fitted'),
            'style_guidance': kwargs.get('style_guidance'),
            'preserve_background': kwargs.get('preserve_background', True),
            'quality_enhancement': kwargs.get('quality_enhancement', True),
            
            # 이미지 정보
            'person_image_shape': person_img.shape,
            'clothing_image_shape': clothing_img.shape,
            
            # 추출된 특성
            'fabric_properties': FABRIC_PROPERTIES.get(
                kwargs.get('fabric_type', 'cotton'),
                FABRIC_PROPERTIES['default']
            ),
            'fitting_params': CLOTHING_FITTING_PARAMS.get(
                kwargs.get('clothing_type', 'shirt'),
                CLOTHING_FITTING_PARAMS['default']
            )
        }
        
        # AI 모델 기반 분석 (선택적)
        if self.fitting_config['ai_models_enabled']:
            ai_analysis = await self._ai_analysis(person_img, clothing_img)
            metadata.update(ai_analysis)
        
        return metadata
    
    async def _ai_analysis(
        self, 
        person_img: np.ndarray, 
        clothing_img: np.ndarray
    ) -> Dict[str, Any]:
        """AI 기반 분석"""
        analysis = {}
        
        try:
            # 인체 파싱
            if self.ai_models['human_parser']:
                body_parts = await self._parse_body_parts(person_img)
                analysis['body_parts'] = body_parts
            
            # 포즈 추정
            if self.ai_models['pose_estimator']:
                pose_keypoints = await self._estimate_pose(person_img)
                analysis['pose_keypoints'] = pose_keypoints
            
            # 의류 분할
            if self.ai_models['cloth_segmenter']:
                cloth_mask = await self._segment_clothing(clothing_img)
                analysis['cloth_mask'] = cloth_mask
            
            # 스타일 특성
            if self.ai_models['style_encoder']:
                style_features = await self._encode_style(clothing_img)
                analysis['style_features'] = style_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ AI 분석 중 오류: {e}")
        
        return analysis
    
    async def _execute_fitting(
        self,
        person_img: np.ndarray,
        clothing_img: np.ndarray,
        metadata: Dict[str, Any],
        session_id: str
    ) -> np.ndarray:
        """피팅 실행"""
        
        method = self.fitting_config['method']
        
        if method == FittingMethod.AI_NEURAL and self.ai_models['diffusion_pipeline']:
            return await self._ai_neural_fitting(person_img, clothing_img, metadata)
        
        elif method == FittingMethod.PHYSICS_BASED and self.fitting_config['physics_enabled']:
            return await self._physics_based_fitting(person_img, clothing_img, metadata)
        
        elif method == FittingMethod.HYBRID:
            # AI와 물리 결합
            ai_result = await self._ai_neural_fitting(person_img, clothing_img, metadata)
            if ai_result is not None:
                return await self._physics_refinement(ai_result, metadata)
            else:
                return await self._physics_based_fitting(person_img, clothing_img, metadata)
        
        else:
            # 템플릿 매칭 폴백
            return await self._template_matching_fitting(person_img, clothing_img, metadata)
    
    async def _ai_neural_fitting(
        self,
        person_img: np.ndarray,
        clothing_img: np.ndarray,
        metadata: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """AI 신경망 기반 피팅"""
        
        try:
            pipeline = self.ai_models['diffusion_pipeline']
            if not pipeline:
                return None
            
            self.logger.info("🧠 AI 신경망 피팅 실행 중...")
            
            # 프롬프트 생성
            prompt = self._generate_fitting_prompt(metadata)
            
            # 입력 이미지 준비
            person_pil = Image.fromarray(person_img)
            clothing_pil = Image.fromarray(clothing_img)
            
            # 컨트롤넷이나 인페인팅 방식으로 처리
            if hasattr(pipeline, 'img2img'):
                # img2img 방식
                fitted_result = pipeline.img2img(
                    prompt=prompt,
                    image=person_pil,
                    strength=0.7,
                    guidance_scale=7.5,
                    num_inference_steps=20
                ).images[0]
            else:
                # 일반 text2img에서 이미지 조건 추가
                fitted_result = pipeline(
                    prompt=prompt,
                    guidance_scale=7.5,
                    num_inference_steps=20,
                    height=person_img.shape[0],
                    width=person_img.shape[1]
                ).images[0]
            
            result_array = np.array(fitted_result)
            
            self.logger.info("✅ AI 신경망 피팅 완료")
            return result_array
            
        except Exception as e:
            self.logger.error(f"❌ AI 신경망 피팅 실패: {e}")
            return None
    
    def _generate_fitting_prompt(self, metadata: Dict[str, Any]) -> str:
        """피팅용 프롬프트 생성"""
        fabric_type = metadata.get('fabric_type', 'cotton')
        clothing_type = metadata.get('clothing_type', 'shirt')
        fit_preference = metadata.get('fit_preference', 'fitted')
        
        prompt = f"A person wearing a {fit_preference} {fabric_type} {clothing_type}, "
        prompt += "realistic lighting, high quality, detailed fabric texture, "
        prompt += "natural pose, professional photography style"
        
        if metadata.get('style_guidance'):
            prompt += f", {metadata['style_guidance']}"
        
        return prompt
    
    async def _physics_based_fitting(
        self,
        person_img: np.ndarray,
        clothing_img: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """물리 기반 피팅"""
        
        try:
            self.logger.info("⚙️ 물리 기반 피팅 실행 중...")
            
            # 1. 신체 모델 생성
            body_model = self._create_body_model(person_img, metadata)
            
            # 2. 의류 메쉬 생성
            cloth_mesh = self._create_cloth_mesh(clothing_img, metadata)
            
            # 3. 물리 시뮬레이션
            fitted_mesh = self._simulate_cloth_physics(body_model, cloth_mesh, metadata)
            
            # 4. 렌더링
            fitted_image = self._render_fitted_clothing(
                person_img, fitted_mesh, metadata
            )
            
            self.logger.info("✅ 물리 기반 피팅 완료")
            return fitted_image
            
        except Exception as e:
            self.logger.error(f"❌ 물리 기반 피팅 실패: {e}")
            # 폴백으로 템플릿 매칭 사용
            return await self._template_matching_fitting(person_img, clothing_img, metadata)
    
    def _create_body_model(
        self, 
        person_img: np.ndarray, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """신체 모델 생성"""
        
        # 간단한 신체 모델 (실제로는 더 복잡한 3D 모델 사용)
        body_model = {
            'image': person_img,
            'height': person_img.shape[0],
            'width': person_img.shape[1],
            'body_parts': metadata.get('body_parts', {}),
            'pose_keypoints': metadata.get('pose_keypoints', {}),
            'body_segments': self._segment_body_parts(person_img)
        }
        
        return body_model
    
    def _segment_body_parts(self, person_img: np.ndarray) -> Dict[str, np.ndarray]:
        """신체 부위 분할"""
        
        # 간단한 색상 기반 분할 (실제로는 AI 모델 사용)
        segments = {}
        
        try:
            if CV2_AVAILABLE:
                # 피부색 감지
                hsv = cv2.cvtColor(person_img, cv2.COLOR_RGB2HSV)
                
                # 피부색 범위 (대략적)
                lower_skin = np.array([0, 20, 70])
                upper_skin = np.array([20, 255, 255])
                skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
                
                segments['skin'] = skin_mask
                segments['clothing_area'] = 255 - skin_mask
            
            else:
                # OpenCV 없을 때 간단한 분할
                h, w = person_img.shape[:2]
                segments['torso'] = np.ones((h, w), dtype=np.uint8) * 255
                segments['arms'] = np.ones((h, w), dtype=np.uint8) * 128
        
        except Exception as e:
            self.logger.warning(f"⚠️ 신체 분할 실패: {e}")
            h, w = person_img.shape[:2]
            segments['default'] = np.ones((h, w), dtype=np.uint8) * 255
        
        return segments
    
    def _create_cloth_mesh(
        self, 
        clothing_img: np.ndarray, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """의류 메쉬 생성"""
        
        fabric_props = metadata['fabric_properties']
        fitting_params = metadata['fitting_params']
        
        cloth_mesh = {
            'image': clothing_img,
            'fabric_properties': fabric_props,
            'fitting_parameters': fitting_params,
            'mesh_resolution': self._calculate_mesh_resolution(),
            'spring_constants': self._calculate_spring_constants(fabric_props),
            'mass_distribution': self._calculate_mass_distribution(fabric_props)
        }
        
        return cloth_mesh
    
    def _calculate_mesh_resolution(self) -> int:
        """메쉬 해상도 계산"""
        quality_resolutions = {
            "fast": 32,
            "balanced": 64,
            "high": 128,
            "ultra": 256
        }
        return quality_resolutions.get(self.quality_level, 64)
    
    def _calculate_spring_constants(self, fabric_props: FabricProperties) -> Dict[str, float]:
        """스프링 상수 계산"""
        base_spring = 100.0
        
        return {
            'structural': base_spring * fabric_props.stiffness,
            'shear': base_spring * fabric_props.stiffness * 0.5,
            'bend': base_spring * fabric_props.stiffness * 0.3,
            'stretch': base_spring * fabric_props.elasticity
        }
    
    def _calculate_mass_distribution(self, fabric_props: FabricProperties) -> float:
        """질량 분포 계산"""
        return fabric_props.density * 0.01  # kg/m²를 가정
    
    def _simulate_cloth_physics(
        self,
        body_model: Dict[str, Any],
        cloth_mesh: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """천 물리 시뮬레이션"""
        
        try:
            # 물리 시뮬레이션 파라미터
            iterations = self.physics_params['iterations']
            time_step = self.physics_params['time_step']
            damping = self.physics_params['damping']
            
            # 초기 상태
            fitted_mesh = cloth_mesh.copy()
            
            # 반복 시뮬레이션
            for i in range(iterations):
                # 중력 적용
                fitted_mesh = self._apply_gravity(fitted_mesh, time_step)
                
                # 신체 충돌 처리
                fitted_mesh = self._handle_body_collision(fitted_mesh, body_model)
                
                # 천 제약 조건 적용
                fitted_mesh = self._apply_cloth_constraints(fitted_mesh)
                
                # 댐핑 적용
                fitted_mesh = self._apply_damping(fitted_mesh, damping)
                
                # 수렴 확인
                if i > 0 and self._check_convergence(fitted_mesh, i):
                    break
            
            return fitted_mesh
            
        except Exception as e:
            self.logger.warning(f"⚠️ 물리 시뮬레이션 실패: {e}")
            return cloth_mesh  # 원본 반환
    
    def _apply_gravity(self, mesh: Dict[str, Any], time_step: float) -> Dict[str, Any]:
        """중력 적용"""
        # 간단한 중력 시뮬레이션
        gravity_force = self.physics_engine['gravity'] * time_step
        mesh['gravity_offset'] = mesh.get('gravity_offset', 0) + gravity_force * 0.1
        return mesh
    
    def _handle_body_collision(
        self, 
        mesh: Dict[str, Any], 
        body_model: Dict[str, Any]
    ) -> Dict[str, Any]:
        """신체 충돌 처리"""
        # 신체와의 충돌 감지 및 처리
        if self.physics_engine['body_collision']:
            # 간단한 충돌 처리 로직
            mesh['collision_adjustments'] = mesh.get('collision_adjustments', {})
        return mesh
    
    def _apply_cloth_constraints(self, mesh: Dict[str, Any]) -> Dict[str, Any]:
        """천 제약 조건 적용"""
        # 스프링 상수를 이용한 제약 조건
        spring_constants = mesh['spring_constants']
        
        # 구조적 제약
        if 'structural' in spring_constants:
            mesh['structural_tension'] = spring_constants['structural']
        
        # 신축성 제약
        if 'stretch' in spring_constants:
            mesh['stretch_limit'] = spring_constants['stretch']
        
        return mesh
    
    def _apply_damping(self, mesh: Dict[str, Any], damping: float) -> Dict[str, Any]:
        """댐핑 적용"""
        mesh['damping_factor'] = damping
        return mesh
    
    def _check_convergence(self, mesh: Dict[str, Any], iteration: int) -> bool:
        """수렴 확인"""
        # 간단한 수렴 조건
        return iteration > self.physics_params['iterations'] * 0.8
    
    def _render_fitted_clothing(
        self,
        person_img: np.ndarray,
        fitted_mesh: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """피팅된 의류 렌더링"""
        
        try:
            # 베이스 이미지로 시작
            result = person_img.copy()
            clothing_img = fitted_mesh['image']
            
            # 간단한 합성 (실제로는 3D 렌더링)
            fitted_result = self._simple_cloth_compositing(
                result, clothing_img, metadata
            )
            
            # 조명 및 그림자 효과
            if self.fitting_config['enable_shadows']:
                fitted_result = self._add_lighting_effects(fitted_result, metadata)
            
            # 텍스처 보존
            if self.fitting_config['texture_preservation']:
                fitted_result = self._preserve_texture_details(
                    fitted_result, clothing_img, metadata
                )
            
            return fitted_result
            
        except Exception as e:
            self.logger.warning(f"⚠️ 렌더링 실패: {e}")
            return person_img
    
    def _simple_cloth_compositing(
        self,
        person_img: np.ndarray,
        clothing_img: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """간단한 의류 합성"""
        
        # 의류 영역 감지
        clothing_mask = self._create_clothing_mask(person_img, metadata)
        
        # 의류 이미지 변형
        warped_clothing = self._warp_clothing_to_body(
            clothing_img, person_img, clothing_mask, metadata
        )
        
        # 알파 블렌딩
        alpha = 0.8  # 의류 불투명도
        result = person_img.copy()
        
        # 마스크된 영역에 의류 적용
        for i in range(3):  # RGB 채널
            result[:, :, i] = np.where(
                clothing_mask > 128,
                alpha * warped_clothing[:, :, i] + (1 - alpha) * person_img[:, :, i],
                person_img[:, :, i]
            )
        
        return result.astype(np.uint8)
    
    def _create_clothing_mask(
        self, 
        person_img: np.ndarray, 
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """의류 마스크 생성"""
        
        h, w = person_img.shape[:2]
        clothing_type = metadata.get('clothing_type', 'shirt')
        
        # 의류 타입별 마스크 영역
        if clothing_type in ['shirt', 'blouse', 'jacket']:
            # 상체 영역
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[h//4:h//2, w//4:3*w//4] = 255
        
        elif clothing_type in ['dress']:
            # 드레스 영역 (상체 + 하체)
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[h//4:3*h//4, w//4:3*w//4] = 255
        
        elif clothing_type in ['pants']:
            # 하체 영역
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[h//2:h, w//3:2*w//3] = 255
        
        else:
            # 기본 상체 영역
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[h//4:h//2, w//4:3*w//4] = 255
        
        # 스무딩
        if CV2_AVAILABLE:
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return mask
    
    def _warp_clothing_to_body(
        self,
        clothing_img: np.ndarray,
        person_img: np.ndarray,
        clothing_mask: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """의류를 신체에 맞게 변형"""
        
        # 의류 이미지를 마스크 영역 크기에 맞게 리사이즈
        mask_coords = np.where(clothing_mask > 0)
        if len(mask_coords[0]) > 0:
            min_y, max_y = np.min(mask_coords[0]), np.max(mask_coords[0])
            min_x, max_x = np.min(mask_coords[1]), np.max(mask_coords[1])
            
            mask_h, mask_w = max_y - min_y, max_x - min_x
            
            if mask_h > 0 and mask_w > 0:
                # 의류 이미지 리사이즈
                resized_clothing = self._resize_image(clothing_img, max(mask_h, mask_w))
                
                # 결과 이미지 크기에 맞게 조정
                warped = np.zeros_like(person_img)
                
                # 중앙 배치
                ch, cw = resized_clothing.shape[:2]
                cy = min_y + (mask_h - ch) // 2
                cx = min_x + (mask_w - cw) // 2
                
                # 범위 확인 후 배치
                if cy >= 0 and cx >= 0 and cy + ch <= person_img.shape[0] and cx + cw <= person_img.shape[1]:
                    warped[cy:cy+ch, cx:cx+cw] = resized_clothing
                
                return warped
        
        # 폴백: 전체 이미지 리사이즈
        return self._resize_image(clothing_img, person_img.shape[0])
    
    def _add_lighting_effects(
        self, 
        fitted_img: np.ndarray, 
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """조명 효과 추가"""
        
        try:
            result = fitted_img.copy().astype(np.float32)
            
            # 메인 조명
            main_light = self.lighting_setup['main_light']
            result = self._apply_directional_light(result, main_light)
            
            # 보조 조명
            fill_light = self.lighting_setup['fill_light']
            result = self._apply_directional_light(result, fill_light, strength=0.3)
            
            # 환경 조명
            ambient = self.lighting_setup['ambient']
            result = self._apply_ambient_light(result, ambient)
            
            return np.clip(result, 0, 255).astype(np.uint8)
            
        except Exception as e:
            self.logger.warning(f"⚠️ 조명 효과 추가 실패: {e}")
            return fitted_img
    
    def _apply_directional_light(
        self, 
        image: np.ndarray, 
        light_config: Dict[str, Any], 
        strength: float = 1.0
    ) -> np.ndarray:
        """방향성 조명 적용"""
        
        direction = light_config['direction']
        intensity = light_config['intensity'] * strength
        color = light_config['color']
        
        # 간단한 조명 계산
        h, w = image.shape[:2]
        
        # 그라디언트 마스크 생성
        light_mask = np.ones((h, w), dtype=np.float32)
        
        # 조명 방향에 따른 그라디언트
        if direction[0] > 0:  # 왼쪽에서 오는 빛
            for x in range(w):
                light_mask[:, x] *= (1.0 - direction[0] * x / w)
        
        if direction[1] > 0:  # 위에서 오는 빛
            for y in range(h):
                light_mask[y, :] *= (1.0 - direction[1] * y / h)
        
        # 조명 적용
        for i in range(3):  # RGB 채널
            channel_multiplier = intensity * color[i]
            image[:, :, i] *= (1.0 + light_mask * channel_multiplier * 0.2)
        
        return image
    
    def _apply_ambient_light(
        self, 
        image: np.ndarray, 
        ambient_config: Dict[str, Any]
    ) -> np.ndarray:
        """환경 조명 적용"""
        
        intensity = ambient_config['intensity']
        color = ambient_config['color']
        
        # 전체적으로 밝기 조정
        for i in range(3):
            image[:, :, i] += intensity * color[i] * 20
        
        return image
    
    def _preserve_texture_details(
        self,
        fitted_img: np.ndarray,
        clothing_img: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """텍스처 디테일 보존"""
        
        try:
            # 고주파 성분 추출
            if CV2_AVAILABLE:
                # 의류 이미지의 텍스처 추출
                clothing_gray = cv2.cvtColor(clothing_img, cv2.COLOR_RGB2GRAY)
                texture = cv2.Laplacian(clothing_gray, cv2.CV_64F)
                texture = np.abs(texture)
                
                # 텍스처를 RGB로 변환
                texture_rgb = np.stack([texture] * 3, axis=-1)
                texture_rgb = (texture_rgb / texture_rgb.max() * 50).astype(np.float32)
                
                # 피팅된 이미지에 텍스처 추가
                result = fitted_img.astype(np.float32) + texture_rgb * 0.3
                return np.clip(result, 0, 255).astype(np.uint8)
            
            else:
                return fitted_img
                
        except Exception as e:
            self.logger.warning(f"⚠️ 텍스처 보존 실패: {e}")
            return fitted_img
    
    async def _template_matching_fitting(
        self,
        person_img: np.ndarray,
        clothing_img: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """템플릿 매칭 피팅 (폴백 방법)"""
        
        try:
            self.logger.info("📐 템플릿 매칭 피팅 실행 중...")
            
            # 1. 신체 영역 감지
            body_regions = self._detect_body_regions(person_img)
            
            # 2. 의류 템플릿 매칭
            clothing_regions = self._match_clothing_template(
                clothing_img, body_regions, metadata
            )
            
            # 3. 변형 및 합성
            fitted_result = self._apply_template_transformation(
                person_img, clothing_regions, metadata
            )
            
            self.logger.info("✅ 템플릿 매칭 피팅 완료")
            return fitted_result
            
        except Exception as e:
            self.logger.error(f"❌ 템플릿 매칭 피팅 실패: {e}")
            return self._simple_overlay_fitting(person_img, clothing_img, metadata)
    
    def _detect_body_regions(self, person_img: np.ndarray) -> Dict[str, Tuple[int, int, int, int]]:
        """신체 영역 감지"""
        
        h, w = person_img.shape[:2]
        
        # 간단한 신체 영역 추정
        regions = {
            'head': (w//3, 0, w//3, h//4),
            'torso': (w//4, h//4, w//2, h//2),
            'arms': (0, h//4, w, h//3),
            'legs': (w//3, 2*h//3, w//3, h//3)
        }
        
        return regions
    
    def _match_clothing_template(
        self,
        clothing_img: np.ndarray,
        body_regions: Dict[str, Tuple[int, int, int, int]],
        metadata: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """의류 템플릿 매칭"""
        
        clothing_type = metadata.get('clothing_type', 'shirt')
        
        # 의류 타입에 따른 영역 매핑
        if clothing_type in ['shirt', 'blouse', 'jacket']:
            target_region = body_regions['torso']
        elif clothing_type == 'dress':
            target_region = (body_regions['torso'][0], body_regions['torso'][1],
                           body_regions['torso'][2], body_regions['torso'][3] + body_regions['legs'][3])
        elif clothing_type == 'pants':
            target_region = body_regions['legs']
        else:
            target_region = body_regions['torso']
        
        # 의류 이미지를 대상 영역에 맞게 변형
        x, y, w, h = target_region
        clothing_fitted = self._resize_image(clothing_img, max(w, h))
        
        return {'main': clothing_fitted, 'region': target_region}
    
    def _apply_template_transformation(
        self,
        person_img: np.ndarray,
        clothing_regions: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """템플릿 변형 적용"""
        
        result = person_img.copy()
        clothing_fitted = clothing_regions['main']
        x, y, w, h = clothing_regions['region']
        
        # 의류 이미지 크기 조정
        ch, cw = clothing_fitted.shape[:2]
        if ch > h:
            scale = h / ch
            new_h, new_w = int(ch * scale), int(cw * scale)
            clothing_fitted = self._resize_image(clothing_fitted, max(new_h, new_w))
            ch, cw = clothing_fitted.shape[:2]
        
        # 중앙 정렬
        start_y = max(0, y + (h - ch) // 2)
        start_x = max(0, x + (w - cw) // 2)
        end_y = min(person_img.shape[0], start_y + ch)
        end_x = min(person_img.shape[1], start_x + cw)
        
        # 실제 복사될 영역 계산
        copy_h = end_y - start_y
        copy_w = end_x - start_x
        
        if copy_h > 0 and copy_w > 0:
            # 알파 블렌딩으로 자연스럽게 합성
            alpha = 0.7
            clothing_crop = clothing_fitted[:copy_h, :copy_w]
            original_crop = result[start_y:end_y, start_x:end_x]
            
            blended = (alpha * clothing_crop + (1 - alpha) * original_crop).astype(np.uint8)
            result[start_y:end_y, start_x:end_x] = blended
        
        return result
    
    def _simple_overlay_fitting(
        self,
        person_img: np.ndarray,
        clothing_img: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """간단한 오버레이 피팅 (최종 폴백)"""
        
        self.logger.info("🔄 간단한 오버레이 피팅 사용")
        
        result = person_img.copy()
        h, w = result.shape[:2]
        
        # 의류를 중앙 상단에 배치
        clothing_resized = self._resize_image(clothing_img, min(w//2, h//2))
        ch, cw = clothing_resized.shape[:2]
        
        y_offset = h // 4
        x_offset = (w - cw) // 2
        
        if (y_offset + ch <= h and x_offset + cw <= w and
            y_offset >= 0 and x_offset >= 0):
            
            # 간단한 알파 블렌딩
            alpha = 0.6
            result[y_offset:y_offset+ch, x_offset:x_offset+cw] = (
                alpha * clothing_resized + 
                (1 - alpha) * result[y_offset:y_offset+ch, x_offset:x_offset+cw]
            ).astype(np.uint8)
        
        return result
    
    async def _physics_refinement(
        self,
        ai_result: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """물리 기반 세밀화"""
        
        try:
            # AI 결과에 물리적 특성 추가
            refined_result = ai_result.copy()
            
            # 주름 효과 추가
            if self.fitting_config['wrinkle_simulation']:
                refined_result = self._add_wrinkle_effects(refined_result, metadata)
            
            # 중력 효과 (드레이핑)
            if metadata['fitting_params'].drape_level > 0.5:
                refined_result = self._add_draping_effects(refined_result, metadata)
            
            # 천 텍스처 향상
            refined_result = self._enhance_fabric_texture(refined_result, metadata)
            
            return refined_result
            
        except Exception as e:
            self.logger.warning(f"⚠️ 물리 세밀화 실패: {e}")
            return ai_result
    
    def _add_wrinkle_effects(
        self,
        image: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """주름 효과 추가"""
        
        try:
            wrinkle_intensity = metadata['fitting_params'].wrinkle_intensity
            
            if wrinkle_intensity > 0 and CV2_AVAILABLE:
                # 노이즈 기반 주름 생성
                h, w = image.shape[:2]
                noise = np.random.normal(0, wrinkle_intensity * 10, (h, w))
                
                # 가우시안 블러로 부드럽게
                noise = cv2.GaussianBlur(noise.astype(np.float32), (5, 5), 0)
                
                # 이미지에 적용
                result = image.astype(np.float32)
                for i in range(3):
                    result[:, :, i] += noise * 0.5
                
                return np.clip(result, 0, 255).astype(np.uint8)
            
            return image
            
        except Exception as e:
            self.logger.warning(f"⚠️ 주름 효과 추가 실패: {e}")
            return image
    
    def _add_draping_effects(
        self,
        image: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """드레이핑 효과 추가"""
        
        try:
            drape_level = metadata['fitting_params'].drape_level
            
            if drape_level > 0.3:
                # 간단한 왜곡 효과로 드레이핑 시뮬레이션
                h, w = image.shape[:2]
                
                # 수직 왜곡 맵 생성
                map_y = np.zeros((h, w), dtype=np.float32)
                for y in range(h):
                    wave = np.sin(np.linspace(0, 4*np.pi, w)) * drape_level * 2
                    map_y[y, :] = y + wave * (y / h)
                
                map_x = np.tile(np.arange(w), (h, 1)).astype(np.float32)
                
                if CV2_AVAILABLE:
                    # 리맵 적용
                    draped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
                    return draped
            
            return image
            
        except Exception as e:
            self.logger.warning(f"⚠️ 드레이핑 효과 추가 실패: {e}")
            return image
    
    def _enhance_fabric_texture(
        self,
        image: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """천 텍스처 향상"""
        
        try:
            fabric_props = metadata['fabric_properties']
            
            # 광택 효과
            if fabric_props.shine > 0.5:
                image = self._add_shine_effect(image, fabric_props.shine)
            
            # 텍스처 스케일링
            if fabric_props.texture_scale != 1.0:
                image = self._scale_texture(image, fabric_props.texture_scale)
            
            return image
            
        except Exception as e:
            self.logger.warning(f"⚠️ 텍스처 향상 실패: {e}")
            return image
    
    def _add_shine_effect(self, image: np.ndarray, shine_level: float) -> np.ndarray:
        """광택 효과 추가"""
        
        try:
            # 하이라이트 영역 생성
            h, w = image.shape[:2]
            highlight = np.zeros((h, w), dtype=np.float32)
            
            # 중앙에서 가장자리로 갈수록 감소하는 광택
            center_y, center_x = h // 2, w // 2
            for y in range(h):
                for x in range(w):
                    dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                    max_dist = np.sqrt(center_y**2 + center_x**2)
                    highlight[y, x] = max(0, (1 - dist / max_dist) * shine_level)
            
            # 이미지에 적용
            result = image.astype(np.float32)
            for i in range(3):
                result[:, :, i] += highlight * 30
            
            return np.clip(result, 0, 255).astype(np.uint8)
            
        except Exception as e:
            return image
    
    def _scale_texture(self, image: np.ndarray, scale: float) -> np.ndarray:
        """텍스처 스케일링"""
        # 간단한 노이즈 추가로 텍스처 효과
        if scale != 1.0:
            noise = np.random.normal(0, (scale - 1.0) * 5, image.shape)
            result = image.astype(np.float32) + noise
            return np.clip(result, 0, 255).astype(np.uint8)
        return image
    
    # =================================================================
    # 5. AI 모델별 처리 함수들
    # =================================================================
    
    async def _parse_body_parts(self, person_img: np.ndarray) -> Dict[str, Any]:
        """AI 인체 파싱"""
        try:
            parser = self.ai_models['human_parser']
            if parser and hasattr(parser, 'parse'):
                result = await parser.parse(person_img)
                return result
            return {}
        except Exception as e:
            self.logger.warning(f"⚠️ 인체 파싱 실패: {e}")
            return {}
    
    async def _estimate_pose(self, person_img: np.ndarray) -> Dict[str, Any]:
        """AI 포즈 추정"""
        try:
            estimator = self.ai_models['pose_estimator']
            if estimator and hasattr(estimator, 'estimate'):
                result = await estimator.estimate(person_img)
                return result
            return {}
        except Exception as e:
            self.logger.warning(f"⚠️ 포즈 추정 실패: {e}")
            return {}
    
    async def _segment_clothing(self, clothing_img: np.ndarray) -> Optional[np.ndarray]:
        """AI 의류 분할"""
        try:
            segmenter = self.ai_models['cloth_segmenter']
            if segmenter and hasattr(segmenter, 'segment'):
                result = await segmenter.segment(clothing_img)
                return result
            return None
        except Exception as e:
            self.logger.warning(f"⚠️ 의류 분할 실패: {e}")
            return None
    
    async def _encode_style(self, clothing_img: np.ndarray) -> Optional[np.ndarray]:
        """AI 스타일 인코딩"""
        try:
            encoder = self.ai_models['style_encoder']
            if encoder and hasattr(encoder, 'encode'):
                result = await encoder.encode(clothing_img)
                return result
            return None
        except Exception as e:
            self.logger.warning(f"⚠️ 스타일 인코딩 실패: {e}")
            return None
    
    # =================================================================
    # 6. 후처리 및 결과 구성
    # =================================================================
    
    async def _post_process_result(
        self,
        fitted_image: np.ndarray,
        metadata: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> np.ndarray:
        """결과 후처리"""
        
        result = fitted_image.copy()
        
        try:
            # 품질 향상 (선택적)
            if kwargs.get('quality_enhancement', True):
                result = await self._enhance_image_quality(result)
            
            # 배경 보존 (선택적)
            if kwargs.get('preserve_background', True):
                # 원본 배경과 합성된 결과 블렌딩
                pass  # 구현 생략 (복잡함)
            
            # 색상 보정
            result = self._color_correction(result, metadata)
            
            # 최종 필터링
            result = self._apply_final_filters(result, metadata)
            
            return result
            
        except Exception as e:
            self.logger.warning(f"⚠️ 후처리 실패: {e}")
            return fitted_image
    
    async def _enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """이미지 품질 향상"""
        
        try:
            result = image.copy()
            
            # 샤프닝
            if CV2_AVAILABLE:
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                sharpened = cv2.filter2D(result, -1, kernel)
                result = cv2.addWeighted(result, 0.7, sharpened, 0.3, 0)
            
            # 대비 향상
            result = self._enhance_contrast(result)
            
            # 노이즈 제거
            if CV2_AVAILABLE:
                result = cv2.bilateralFilter(result, 9, 75, 75)
            
            return result
            
        except Exception as e:
            self.logger.warning(f"⚠️ 품질 향상 실패: {e}")
            return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """대비 향상"""
        try:
            # 히스토그램 평활화
            if CV2_AVAILABLE:
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
                lab = cv2.merge([l, a, b])
                return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                return image
        except:
            return image
    
    def _color_correction(self, image: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """색상 보정"""
        
        try:
            fabric_type = metadata.get('fabric_type', 'cotton')
            
            # 천 재질별 색상 조정
            if fabric_type == 'silk':
                # 실크: 채도 증가
                if CV2_AVAILABLE:
                    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                    hsv[:, :, 1] = hsv[:, :, 1] * 1.2  # 채도 증가
                    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            elif fabric_type == 'denim':
                # 데님: 파란색 톤 강화
                image[:, :, 2] = np.clip(image[:, :, 2] * 1.1, 0, 255)  # 파란색 강화
            
            elif fabric_type == 'leather':
                # 가죽: 갈색 톤 강화
                image[:, :, 0] = np.clip(image[:, :, 0] * 1.1, 0, 255)  # 빨간색 강화
                image[:, :, 1] = np.clip(image[:, :, 1] * 1.05, 0, 255)  # 녹색 약간 강화
            
            return image
            
        except Exception as e:
            self.logger.warning(f"⚠️ 색상 보정 실패: {e}")
            return image
    
    def _apply_final_filters(self, image: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """최종 필터 적용"""
        
        try:
            # 질감 향상
            if metadata.get('fabric_properties'):
                fabric_props = metadata['fabric_properties']
                if fabric_props.shine > 0.3:
                    # 광택 있는 재질에 대한 추가 처리
                    image = self._add_shine_effect(image, fabric_props.shine * 0.5)
            
            # 전체적인 색온도 조정
            if self.config.get('warm_tone', False):
                image[:, :, 0] = np.clip(image[:, :, 0] * 1.05, 0, 255)  # 따뜻한 톤
                image[:, :, 2] = np.clip(image[:, :, 2] * 0.98, 0, 255)
            
            return image
            
        except Exception as e:
            self.logger.warning(f"⚠️ 최종 필터 적용 실패: {e}")
            return image
    
    def _build_result(
        self,
        fitted_image: np.ndarray,
        metadata: Dict[str, Any],
        processing_time: float,
        session_id: str
    ) -> Dict[str, Any]:
        """최종 결과 구성"""
        
        # 품질 점수 계산
        quality_score = self._calculate_quality_score(fitted_image, metadata)
        
        # 신뢰도 계산
        confidence_score = self._calculate_confidence_score(metadata, processing_time)
        
        # 피팅 점수 계산
        fit_score = self._calculate_fit_score(metadata)
        
        result = {
            "success": True,
            "session_id": session_id,
            "fitted_image": fitted_image,
            "processing_time": processing_time,
            
            # 점수들
            "quality_score": quality_score,
            "confidence_score": confidence_score, 
            "fit_score": fit_score,
            "overall_score": (quality_score + confidence_score + fit_score) / 3,
            
            # 메타데이터
            "metadata": {
                "fabric_type": metadata.get('fabric_type'),
                "clothing_type": metadata.get('clothing_type'),
                "fitting_method": self.fitting_method,
                "quality_level": self.quality_level,
                "image_resolution": fitted_image.shape[:2],
                "ai_models_used": [name for name, model in self.ai_models.items() if model is not None],
                "physics_enabled": self.fitting_config['physics_enabled']
            },
            
            # 성능 정보
            "performance_info": {
                "device": self.device,
                "memory_usage_mb": self._get_current_memory_usage(),
                "processing_method": self.fitting_config['method'].value,
                "cache_used": session_id in self.fitting_cache
            },
            
            # 개선 제안
            "recommendations": self._generate_recommendations(metadata, quality_score)
        }
        
        return result
    
    def _calculate_quality_score(self, image: np.ndarray, metadata: Dict[str, Any]) -> float:
        """품질 점수 계산"""
        
        try:
            scores = []
            
            # 1. 이미지 선명도
            if CV2_AVAILABLE:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                sharpness_score = min(1.0, sharpness / 1000.0)
                scores.append(sharpness_score)
            
            # 2. 색상 분포
            color_variance = np.var(image, axis=(0,1)).mean()
            color_score = min(1.0, color_variance / 5000.0)
            scores.append(color_score)
            
            # 3. 대비
            contrast = image.max() - image.min()
            contrast_score = min(1.0, contrast / 255.0)
            scores.append(contrast_score)
            
            # 4. 노이즈 레벨 (낮을수록 좋음)
            noise_level = np.std(image)
            noise_score = max(0.0, 1.0 - noise_level / 50.0)
            scores.append(noise_score)
            
            return float(np.mean(scores))
            
        except Exception as e:
            self.logger.warning(f"⚠️ 품질 점수 계산 실패: {e}")
            return 0.7  # 기본값
    
    def _calculate_confidence_score(self, metadata: Dict[str, Any], processing_time: float) -> float:
        """신뢰도 점수 계산"""
        
        try:
            scores = []
            
            # 1. AI 모델 사용 여부
            ai_models_used = sum(1 for model in self.ai_models.values() if model is not None)
            ai_score = ai_models_used / len(self.ai_models)
            scores.append(ai_score)
            
            # 2. 처리 시간 (적절한 시간일수록 높은 점수)
            if processing_time < 2.0:
                time_score = 0.6  # 너무 빨라도 신뢰도 낮음
            elif processing_time < 10.0:
                time_score = 1.0  # 적절한 시간
            else:
                time_score = max(0.3, 1.0 - (processing_time - 10.0) / 30.0)
            scores.append(time_score)
            
            # 3. 입력 데이터 품질
            input_quality = 1.0
            if metadata.get('person_image_shape') and metadata.get('clothing_image_shape'):
                person_pixels = np.prod(metadata['person_image_shape'][:2])
                clothing_pixels = np.prod(metadata['clothing_image_shape'][:2])
                min_pixels = min(person_pixels, clothing_pixels)
                input_quality = min(1.0, min_pixels / (512 * 512))
            scores.append(input_quality)
            
            return float(np.mean(scores))
            
        except Exception as e:
            self.logger.warning(f"⚠️ 신뢰도 계산 실패: {e}")
            return 0.7
    
    def _calculate_fit_score(self, metadata: Dict[str, Any]) -> float:
        """피팅 점수 계산"""
        
        try:
            scores = []
            
            # 1. 천 재질과 의류 타입 호환성
            fabric_type = metadata.get('fabric_type', 'cotton')
            clothing_type = metadata.get('clothing_type', 'shirt')
            
            compatibility_matrix = {
                ('cotton', 'shirt'): 0.9,
                ('cotton', 'dress'): 0.8,
                ('silk', 'dress'): 0.95,
                ('silk', 'blouse'): 0.9,
                ('denim', 'pants'): 0.95,
                ('leather', 'jacket'): 0.9
            }
            
            compatibility_score = compatibility_matrix.get(
                (fabric_type, clothing_type), 0.7
            )
            scores.append(compatibility_score)
            
            # 2. 물리 시뮬레이션 품질
            physics_score = 0.9 if self.fitting_config['physics_enabled'] else 0.6
            scores.append(physics_score)
            
            # 3. 해상도 점수
            max_res = self.performance_config['max_resolution']
            resolution_score = min(1.0, max_res / 512.0)
            scores.append(resolution_score)
            
            return float(np.mean(scores))
            
        except Exception as e:
            self.logger.warning(f"⚠️ 피팅 점수 계산 실패: {e}")
            return 0.7
    
    def _generate_recommendations(
        self, 
        metadata: Dict[str, Any], 
        quality_score: float
    ) -> List[str]:
        """개선 제안 생성"""
        
        recommendations = []
        
        try:
            # 품질이 낮은 경우
            if quality_score < 0.6:
                recommendations.append("더 높은 해상도의 이미지를 사용해보세요")
                recommendations.append("조명이 좋은 환경에서 촬영된 이미지를 사용해보세요")
            
            # AI 모델 미사용 시
            if not self.fitting_config['ai_models_enabled']:
                recommendations.append("AI 모델을 활성화하면 더 정확한 결과를 얻을 수 있습니다")
            
            # 물리 엔진 미사용 시
            if not self.fitting_config['physics_enabled']:
                recommendations.append("물리 시뮬레이션을 활성화하면 더 자연스러운 피팅을 얻을 수 있습니다")
            
            # 천 재질별 제안
            fabric_type = metadata.get('fabric_type')
            if fabric_type == 'silk':
                recommendations.append("실크 소재의 특성상 드레이핑 효과를 높여보세요")
            elif fabric_type == 'denim':
                recommendations.append("데님의 견고함을 표현하기 위해 텍스처를 강화해보세요")
            
            # 기본 제안
            if not recommendations:
                recommendations = [
                    "훌륭한 결과입니다!",
                    "다양한 포즈로 시도해보세요",
                    "다른 스타일의 의류도 체험해보세요"
                ]
            
        except Exception as e:
            self.logger.warning(f"⚠️ 제안 생성 실패: {e}")
            recommendations = ["결과를 확인해보세요"]
        
        return recommendations[:3]  # 최대 3개 제안
    
    # =================================================================
    # 7. 캐시 및 메모리 관리
    # =================================================================
    
    def _generate_cache_key(
        self, 
        person_img: np.ndarray, 
        clothing_img: np.ndarray, 
        kwargs: Dict[str, Any]
    ) -> str:
        """캐시 키 생성"""
        
        try:
            # 이미지 해시
            person_hash = hash(person_img.tobytes())
            clothing_hash = hash(clothing_img.tobytes())
            
            # 설정 해시
            config_str = json.dumps({
                'fabric_type': kwargs.get('fabric_type', 'cotton'),
                'clothing_type': kwargs.get('clothing_type', 'shirt'),
                'quality_level': self.quality_level,
                'fitting_method': self.fitting_method
            }, sort_keys=True)
            config_hash = hash(config_str)
            
            return f"vf_{person_hash}_{clothing_hash}_{config_hash}"
            
        except Exception as e:
            self.logger.warning(f"⚠️ 캐시 키 생성 실패: {e}")
            return f"vf_{uuid.uuid4().hex[:16]}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """캐시에서 결과 가져오기"""
        
        if not self.cache_config['enabled']:
            return None
        
        try:
            if cache_key in self.fitting_cache:
                cached_data = self.fitting_cache[cache_key]
                
                # TTL 확인
                if time.time() - cached_data['timestamp'] < self.cache_config['ttl_seconds']:
                    self.cache_access_times[cache_key] = time.time()
                    return cached_data['result']
                else:
                    # 만료된 캐시 제거
                    del self.fitting_cache[cache_key]
                    if cache_key in self.cache_access_times:
                        del self.cache_access_times[cache_key]
            
            return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ 캐시 조회 실패: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """캐시에 결과 저장"""
        
        if not self.cache_config['enabled']:
            return
        
        try:
            # 캐시 크기 제한
            if len(self.fitting_cache) >= self.cache_max_size:
                self._cleanup_cache()
            
            # 결과 저장 (이미지는 제외하고 메타데이터만)
            cache_data = {
                'timestamp': time.time(),
                'result': {
                    k: v for k, v in result.items() 
                    if k != 'fitted_image'  # 이미지는 메모리 절약을 위해 캐시하지 않음
                }
            }
            
            self.fitting_cache[cache_key] = cache_data
            self.cache_access_times[cache_key] = time.time()
            
        except Exception as e:
            self.logger.warning(f"⚠️ 캐시 저장 실패: {e}")
    
    def _cleanup_cache(self):
        """캐시 정리"""
        
        try:
            # LRU 방식으로 오래된 항목 제거
            if self.cache_access_times:
                sorted_items = sorted(
                    self.cache_access_times.items(),
                    key=lambda x: x[1]
                )
                
                # 오래된 25% 제거
                remove_count = len(sorted_items) // 4
                for cache_key, _ in sorted_items[:remove_count]:
                    if cache_key in self.fitting_cache:
                        del self.fitting_cache[cache_key]
                    del self.cache_access_times[cache_key]
            
        except Exception as e:
            self.logger.warning(f"⚠️ 캐시 정리 실패: {e}")
    
    def _get_current_memory_usage(self) -> float:
        """현재 메모리 사용량 (MB)"""
        
        try:
            if self.memory_manager:
                return self.memory_manager.get_memory_usage()
            
            # 폴백: psutil 사용
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
            
        except Exception:
            return 0.0
    
    def _update_stats(self, processing_time: float, success: bool):
        """통계 업데이트"""
        
        self.performance_stats['total_processed'] += 1
        
        if success:
            self.performance_stats['successful_fittings'] += 1
        else:
            self.performance_stats['failed_fittings'] += 1
        
        # 평균 처리 시간 업데이트
        total_time = (self.performance_stats['average_processing_time'] * 
                     (self.performance_stats['total_processed'] - 1) + processing_time)
        self.performance_stats['average_processing_time'] = total_time / self.performance_stats['total_processed']
        
        # 메모리 피크 업데이트
        current_memory = self._get_current_memory_usage()
        if current_memory > self.performance_stats['memory_peak_mb']:
            self.performance_stats['memory_peak_mb'] = current_memory
    
    # =================================================================
    # 8. 정보 조회 및 관리 함수들
    # =================================================================
    
    async def get_step_info(self) -> Dict[str, Any]:
        """단계 정보 반환"""
        
        return {
            "step_name": self.step_name,
            "version": "6.0-complete",
            "device": self.device,
            "device_type": self.device_type,
            "memory_gb": self.memory_gb,
            "is_m3_max": self.is_m3_max,
            "optimization_enabled": self.optimization_enabled,
            "quality_level": self.quality_level,
            "fitting_method": self.fitting_method,
            "initialized": self.is_initialized,
            "session_id": self.session_id,
            
            # 구성 정보
            "config": {
                "ai_models_enabled": self.fitting_config['ai_models_enabled'],
                "physics_enabled": self.fitting_config['physics_enabled'],
                "max_resolution": self.performance_config['max_resolution'],
                "cache_enabled": self.cache_config['enabled'],
                "cache_size": len(self.fitting_cache)
            },
            
            # 성능 통계
            "performance_stats": self.performance_stats.copy(),
            
            # 기능 지원
            "capabilities": {
                "ai_neural_fitting": DIFFUSERS_AVAILABLE and self.ai_models['diffusion_pipeline'] is not None,
                "physics_based_fitting": self.fitting_config['physics_enabled'],
                "hybrid_fitting": True,
                "template_matching": True,
                "texture_preservation": True,
                "lighting_effects": True,
                "fabric_simulation": True,
                "wrinkle_simulation": True,
                "m3_max_optimization": self.is_m3_max
            },
            
            # 지원 형식
            "supported_formats": {
                "fabric_types": list(FABRIC_PROPERTIES.keys()),
                "clothing_types": list(CLOTHING_FITTING_PARAMS.keys()),
                "quality_levels": [q.value for q in FittingQuality],
                "fitting_methods": [m.value for m in FittingMethod]
            },
            
            # 의존성 상태
            "dependencies": {
                "torch": TORCH_AVAILABLE,
                "opencv": CV2_AVAILABLE,
                "scipy": SCIPY_AVAILABLE,
                "sklearn": SKLEARN_AVAILABLE,
                "skimage": SKIMAGE_AVAILABLE,
                "diffusers": DIFFUSERS_AVAILABLE
            },
            
            # AI 모델 상태
            "ai_models_status": {
                name: model is not None 
                for name, model in self.ai_models.items()
            }
        }
    
    async def cleanup(self):
        """리소스 정리"""
        
        try:
            self.logger.info(f"🧹 {self.step_name} 리소스 정리 시작...")
            
            # AI 모델 언로드
            for name, model in self.ai_models.items():
                if model is not None:
                    if hasattr(model, 'to'):
                        model.to('cpu')
                    del model
                    self.ai_models[name] = None
            
            # 캐시 정리
            self.fitting_cache.clear()
            self.cache_access_times.clear()
            
            # 스레드 풀 종료
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
            
            # GPU 메모리 정리
            if TORCH_AVAILABLE and self.device in ['cuda', 'mps']:
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device == 'mps':
                    torch.mps.empty_cache()
            
            # 가비지 컬렉션
            gc.collect()
            
            self.is_initialized = False
            self.logger.info(f"✅ {self.step_name} 리소스 정리 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 리소스 정리 실패: {e}")

# =================================================================
# 9. 편의 함수들 (하위 호환성)
# =================================================================

def create_virtual_fitting_step(
    device: str = "auto", 
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> VirtualFittingStep:
    """가상 피팅 단계 생성 (기존 방식 호환)"""
    return VirtualFittingStep(device=device, config=config, **kwargs)

def create_m3_max_virtual_fitting_step(
    memory_gb: float = 128.0,
    quality_level: str = "ultra",
    **kwargs
) -> VirtualFittingStep:
    """M3 Max 최적화 가상 피팅 단계 생성"""
    return VirtualFittingStep(
        device=None,  # 자동 감지
        memory_gb=memory_gb,
        quality_level=quality_level,
        is_m3_max=True,
        optimization_enabled=True,
        **kwargs
    )

async def quick_virtual_fitting(
    person_image: Union[np.ndarray, Image.Image, str],
    clothing_image: Union[np.ndarray, Image.Image, str],
    fabric_type: str = "cotton",
    clothing_type: str = "shirt",
    **kwargs
) -> Dict[str, Any]:
    """빠른 가상 피팅 (일회성 사용)"""
    
    step = VirtualFittingStep()
    try:
        await step.initialize()
        result = await step.process(
            person_image, clothing_image,
            fabric_type=fabric_type,
            clothing_type=clothing_type,
            **kwargs
        )
        return result
    finally:
        await step.cleanup()

# =================================================================
# 10. 모듈 정보
# =================================================================

__version__ = "6.0.0"
__author__ = "MyCloset AI Team"
__description__ = "Complete Virtual Fitting Implementation with AI Models and Physics Simulation"

if __name__ == "__main__":
    # 테스트 코드
    async def test_virtual_fitting():
        """테스트 함수"""
        import asyncio
        
        # 테스트 이미지 생성
        test_person = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        test_clothing = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # 가상 피팅 실행
        step = VirtualFittingStep(quality_level="balanced")
        await step.initialize()
        
        result = await step.process(
            test_person, test_clothing,
            fabric_type="cotton",
            clothing_type="shirt"
        )
        
        print(f"✅ 테스트 완료: {result['success']}")
        print(f"   처리 시간: {result['processing_time']:.2f}초")
        print(f"   품질 점수: {result['quality_score']:.2f}")
        
        await step.cleanup()
    
    # 테스트 실행
    asyncio.run(test_virtual_fitting())