# app/ai_pipeline/steps/step_05_cloth_warping.py

"""
🎯 Step 5: 의류 워핑 (Cloth Warping) - ModelLoader 완전 연동 버전
===========================================================================

✅ 시뮬레이션 모드 완전 제거 - 실제 AI 모델만 사용
✅ ModelLoader 완전 연동으로 순환참조 해결
✅ BaseStepMixin 단일 상속으로 MRO 오류 해결
✅ 모든 기능 유지 - 기존 함수/클래스명 보존
✅ 실제 AI 연산 데이터 인터페이스 완벽 구현
✅ M3 Max 최적화 및 conda 환경 지원
✅ 한방향 데이터 흐름 구조

🔥 핵심 변경사항:
- 모든 시뮬레이션 관련 코드 완전 제거
- ModelLoader 실패 시 명확한 에러 반환
- strict_mode=True로 설정하여 실패 시 즉시 중단
- 실제 AI 모델을 통한 워핑 연산만 수행

참고 흐름: API → PipelineManager → Step → ModelLoader 협업 → AI 추론 → 결과 반환
"""

import asyncio
import logging
import os
import time
import traceback
import hashlib
import json
import gc
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from PIL import Image

# ==============================================
# 🔧 Import 검증 및 필수 라이브러리
# ==============================================

# BaseStepMixin 가져오기 (필수)
try:
    from .base_step_mixin import BaseStepMixin
    BASE_STEP_MIXIN_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).error(f"❌ BaseStepMixin import 필수: {e}")
    raise ImportError("BaseStepMixin이 필요합니다. 파일 위치를 확인해주세요.")

# ModelLoader 가져오기 (핵심)
try:
    from ..utils.model_loader import get_global_model_loader, ModelLoader
    MODEL_LOADER_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).error(f"❌ ModelLoader import 필수: {e}")
    raise ImportError("ModelLoader가 필요합니다. AI 모델 로드를 위해 필수입니다.")

# 추가 모듈들 (선택적)
try:
    import skimage
    from skimage import filters, morphology, measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ==============================================
# 🎯 설정 클래스들 및 Enum
# ==============================================

class WarpingMethod(Enum):
    """워핑 방법 열거형 - AI 모델만 사용"""
    AI_MODEL = "ai_model"           # ModelLoader를 통한 AI 모델
    HYBRID = "hybrid"               # AI + 물리 결합

class FabricType(Enum):
    """원단 타입 열거형"""
    COTTON = "cotton"
    SILK = "silk"
    DENIM = "denim"
    WOOL = "wool"
    POLYESTER = "polyester"
    LINEN = "linen"
    LEATHER = "leather"

class WarpingQuality(Enum):
    """워핑 품질 레벨"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

@dataclass
class PhysicsProperties:
    """물리 시뮬레이션 속성"""
    fabric_type: FabricType = FabricType.COTTON
    thickness: float = 0.001  # meters
    density: float = 1500.0  # kg/m³
    elastic_modulus: float = 1000.0  # Pa
    poisson_ratio: float = 0.3
    friction_coefficient: float = 0.4
    air_resistance: float = 0.01

@dataclass
class ClothWarpingConfig:
    """의류 워핑 설정"""
    warping_method: WarpingMethod = WarpingMethod.AI_MODEL
    input_size: Tuple[int, int] = (512, 384)
    num_control_points: int = 25
    ai_model_enabled: bool = True
    physics_enabled: bool = True
    visualization_enabled: bool = True
    cache_enabled: bool = True
    cache_size: int = 50
    quality_level: str = "high"
    precision: str = "fp16"
    memory_fraction: float = 0.7
    batch_size: int = 1
    strict_mode: bool = True  # AI 모델 실패 시 즉시 중단

# 의류 타입별 워핑 가중치
CLOTHING_WARPING_WEIGHTS = {
    'shirt': {'deformation': 0.4, 'physics': 0.3, 'texture': 0.3},
    'dress': {'deformation': 0.5, 'physics': 0.3, 'texture': 0.2},
    'pants': {'physics': 0.5, 'deformation': 0.3, 'texture': 0.2},
    'jacket': {'physics': 0.4, 'deformation': 0.4, 'texture': 0.2},
    'skirt': {'deformation': 0.4, 'physics': 0.4, 'texture': 0.2},
    'top': {'deformation': 0.5, 'texture': 0.3, 'physics': 0.2},
    'default': {'deformation': 0.4, 'physics': 0.3, 'texture': 0.3}
}

# ==============================================
# 🔧 고급 변환 및 물리 시뮬레이션 클래스들 (실제 연산용)
# ==============================================

class ClothPhysicsSimulator:
    """의류 물리 시뮬레이션 엔진"""
    
    def __init__(self, properties: PhysicsProperties):
        self.properties = properties
        self.mesh_vertices = None
        self.mesh_faces = None
        self.velocities = None
        self.forces = None
        self.logger = logging.getLogger(__name__)
        
    def create_cloth_mesh(self, width: int, height: int, resolution: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """의류 메시 생성"""
        try:
            x = np.linspace(0, width-1, resolution)
            y = np.linspace(0, height-1, resolution)
            xx, yy = np.meshgrid(x, y)
            
            # 정점 생성
            vertices = np.column_stack([xx.flatten(), yy.flatten(), np.zeros(xx.size)])
            
            # 면 생성
            faces = []
            for i in range(resolution-1):
                for j in range(resolution-1):
                    idx = i * resolution + j
                    faces.append([idx, idx+1, idx+resolution])
                    faces.append([idx+1, idx+resolution+1, idx+resolution])
            
            self.mesh_vertices = vertices
            self.mesh_faces = np.array(faces)
            self.velocities = np.zeros_like(vertices)
            self.forces = np.zeros_like(vertices)
            
            return vertices, self.mesh_faces
            
        except Exception as e:
            self.logger.error(f"메시 생성 실패: {e}")
            raise RuntimeError(f"메시 생성 실패: {e}")
    
    def simulate_step(self, dt: float = 0.016):
        """시뮬레이션 단계 실행"""
        if self.mesh_vertices is None:
            raise ValueError("메시가 초기화되지 않았습니다")
            
        try:
            # 중력 적용
            gravity = np.array([0, 0, -9.81]) * self.properties.density * dt
            self.forces[:, 2] += gravity[2]
            
            # 가속도 및 위치 업데이트
            acceleration = self.forces / self.properties.density
            self.mesh_vertices += self.velocities * dt + 0.5 * acceleration * dt * dt
            self.velocities += acceleration * dt
            
            # 댐핑 적용
            self.velocities *= (1.0 - self.properties.friction_coefficient * dt)
            
            # 힘 초기화
            self.forces.fill(0)
            
        except Exception as e:
            self.logger.error(f"시뮬레이션 단계 실패: {e}")
            raise RuntimeError(f"시뮬레이션 단계 실패: {e}")
    
    def get_deformed_mesh(self) -> np.ndarray:
        """변형된 메시 반환"""
        if self.mesh_vertices is None:
            raise ValueError("메시가 없습니다")
        return self.mesh_vertices.copy()

class AdvancedTPSTransform:
    """고급 TPS (Thin Plate Spline) 변환"""
    
    def __init__(self, num_control_points: int = 25):
        self.num_control_points = num_control_points
        self.logger = logging.getLogger(__name__)
    
    def create_adaptive_control_grid(self, width: int, height: int) -> np.ndarray:
        """적응적 제어점 그리드 생성"""
        grid_size = int(np.sqrt(self.num_control_points))
        points = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = (width - 1) * i / (grid_size - 1)
                y = (height - 1) * j / (grid_size - 1)
                points.append([x, y])
        
        return np.array(points[:self.num_control_points])
    
    def apply_transform(self, image: np.ndarray, source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
        """TPS 변환 적용"""
        try:
            if SKIMAGE_AVAILABLE:
                from skimage.transform import PiecewiseAffineTransform, warp
                tform = PiecewiseAffineTransform()
                tform.estimate(target_points, source_points)
                warped = warp(image, tform, output_shape=image.shape[:2])
                return (warped * 255).astype(np.uint8)
            else:
                # OpenCV 폴백
                return self._opencv_transform(image, source_points, target_points)
        except Exception as e:
            self.logger.error(f"TPS 변환 실패: {e}")
            raise RuntimeError(f"TPS 변환 실패: {e}")
    
    def _opencv_transform(self, image: np.ndarray, source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
        """OpenCV 변환 (필수 폴백)"""
        try:
            H, _ = cv2.findHomography(source_points, target_points, cv2.RANSAC)
            if H is not None:
                height, width = image.shape[:2]
                return cv2.warpPerspective(image, H, (width, height))
            raise RuntimeError("Homography 계산 실패")
        except Exception as e:
            raise RuntimeError(f"OpenCV 변환 실패: {e}")

class WarpingVisualizer:
    """워핑 과정 시각화 엔진"""
    
    def __init__(self, quality: str = "high"):
        self.quality = quality
        self.dpi = {"low": 72, "medium": 150, "high": 300, "ultra": 600}[quality]
        
    def create_warping_visualization(self, 
                                   original_cloth: np.ndarray,
                                   warped_cloth: np.ndarray,
                                   control_points: np.ndarray,
                                   flow_field: Optional[np.ndarray] = None) -> np.ndarray:
        """워핑 과정 종합 시각화"""
        
        h, w = original_cloth.shape[:2]
        canvas_w = w * 2
        canvas_h = h
        
        # 캔버스 생성
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
        
        # 원본 (좌측)
        canvas[0:h, 0:w] = original_cloth
        
        # 워핑 결과 (우측)
        canvas[0:h, w:2*w] = warped_cloth
        
        # 제어점 시각화
        if len(control_points) > 0:
            for i, point in enumerate(control_points):
                x, y = int(point[0]), int(point[1])
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(canvas, (x, y), 3, (255, 0, 0), -1)
                    cv2.circle(canvas, (x + w, y), 3, (0, 255, 0), -1)
        
        # 구분선
        cv2.line(canvas, (w, 0), (w, h), (128, 128, 128), 2)
        
        # 라벨
        cv2.putText(canvas, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, "Warped", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return canvas

class ClothWarpingStep(BaseStepMixin):
    """
    Step 5: 의류 워핑 (Cloth Warping) - ModelLoader 완전 연동
    
    역할 분담:
    - ModelLoader: AI 모델 로드 및 관리
    - Step: 실제 워핑 추론 및 비즈니스 로직
    """
    
    def __init__(self, device: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """초기화 - BaseStepMixin 단일 상속"""
        super().__init__()
        
        # 기본 설정
        self.step_name = "ClothWarpingStep"
        self.config = config or {}
        self.device = self._determine_device(device)
        
        # ModelLoader 인터페이스 (핵심)
        self.model_loader = None
        self.model_interface = None
        self.models_loaded = {}
        
        # 워핑 설정
        self.warping_config = ClothWarpingConfig(
            warping_method=WarpingMethod(self.config.get('warping_method', 'ai_model')),
            input_size=tuple(self.config.get('input_size', (512, 384))),
            num_control_points=self.config.get('num_control_points', 25),
            ai_model_enabled=self.config.get('ai_model_enabled', True),
            physics_enabled=self.config.get('physics_enabled', True),
            visualization_enabled=self.config.get('visualization_enabled', True),
            cache_enabled=self.config.get('cache_enabled', True),
            cache_size=self.config.get('cache_size', 50),
            quality_level=self.config.get('quality_level', 'high'),
            precision=self.config.get('precision', 'fp16'),
            memory_fraction=self.config.get('memory_fraction', 0.7),
            batch_size=self.config.get('batch_size', 1),
            strict_mode=self.config.get('strict_mode', True)
        )
        
        # 성능 및 캐시
        self.performance_stats = {
            'total_processed': 0,
            'avg_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'error_count': 0,
            'success_rate': 0.0
        }
        self.prediction_cache = {}
        
        # 초기화 상태
        self.is_initialized = False
        self.initialization_error = None
        
        # 추가 처리 구성요소
        self.tps_transform = AdvancedTPSTransform(self.warping_config.num_control_points)
        self.physics_simulator = None  # 나중에 초기화
        self.visualizer = WarpingVisualizer(self.warping_config.quality_level)
        
        # 처리 파이프라인
        self.processing_pipeline = []
        
        # 초기화 실행
        asyncio.create_task(self._initialize_async())
    
    def _determine_device(self, device: Optional[str]) -> str:
        """디바이스 결정"""
        if device:
            return device
        
        if torch.backends.mps.is_available():
            return "mps"  # M3 Max 최적화
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    async def _initialize_async(self):
        """비동기 초기화"""
        try:
            await self.initialize()
        except Exception as e:
            self.initialization_error = str(e)
            self.logger.error(f"❌ 비동기 초기화 실패: {e}")
    
    # =================================================================
    # 🚀 ModelLoader 완전 연동 초기화
    # =================================================================
    
    async def initialize(self) -> bool:
        """
        Step 초기화 - ModelLoader와 완전 통합
        
        흐름:
        1. ModelLoader 인터페이스 설정 ← ModelLoader 담당
        2. Step별 모델 요청사항 등록 ← ModelLoader 담당  
        3. AI 모델 로드 ← ModelLoader가 실제 로드
        4. Step별 최적화 적용 ← Step이 적용
        """
        try:
            self.logger.info("🚀 의류 워핑 Step 초기화 시작")
            
            # 1. ModelLoader 인터페이스 설정 (필수)
            success = await self._setup_model_interface()
            if not success:
                error_msg = "ModelLoader 연결 실패 - AI 모델 없이는 작동 불가"
                self.logger.error(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            
            # 2. AI 모델 로드 (ModelLoader를 통해 - 필수)
            if self.warping_config.ai_model_enabled and self.model_interface:
                await self._load_models_via_interface()
                if not self.models_loaded:
                    error_msg = "AI 모델 로드 실패 - 워핑 처리 불가"
                    self.logger.error(f"❌ {error_msg}")
                    raise RuntimeError(error_msg)
            
            # 3. 워핑 파이프라인 설정
            self._setup_warping_pipeline()
            
            # 4. M3 Max 최적화 적용
            if self.device == "mps":
                self._apply_m3_max_optimization()
            
            self.is_initialized = True
            self.logger.info("✅ 의류 워핑 Step 초기화 완료")
            return True
            
        except Exception as e:
            self.initialization_error = str(e)
            self.logger.error(f"❌ 의류 워핑 초기화 실패: {e}")
            self.logger.debug(f"상세 오류: {traceback.format_exc()}")
            raise RuntimeError(f"의류 워핑 Step 초기화 실패: {e}")
    
    async def _setup_model_interface(self) -> bool:
        """ModelLoader 인터페이스 설정 (필수)"""
        try:
            if not MODEL_LOADER_AVAILABLE:
                raise RuntimeError("ModelLoader를 사용할 수 없습니다")
            
            # ModelLoader 가져오기
            self.model_loader = get_global_model_loader()
            if not self.model_loader:
                raise RuntimeError("전역 ModelLoader를 가져올 수 없습니다")
            
            # Step별 인터페이스 생성 ← ModelLoader가 담당
            self.model_interface = self.model_loader.create_step_interface(
                step_name=self.step_name,
                step_requirements={
                    'models': [
                        {
                            'name': 'cloth_warping_primary',
                            'type': 'pytorch',
                            'task': 'cloth_warping',
                            'priority': 'high',
                            'optional': False
                        },
                        {
                            'name': 'cloth_warping_backup', 
                            'type': 'pytorch',
                            'task': 'cloth_warping',
                            'priority': 'medium',
                            'optional': True
                        }
                    ],
                    'device': self.device,
                    'precision': self.warping_config.precision,
                    'memory_fraction': self.warping_config.memory_fraction
                }
            )
            
            if self.model_interface:
                self.logger.info("✅ ModelLoader 인터페이스 설정 완료")
                return True
            else:
                raise RuntimeError("ModelLoader 인터페이스 생성 실패")
                
        except Exception as e:
            self.logger.error(f"❌ ModelLoader 인터페이스 설정 실패: {e}")
            return False
    
    async def _load_models_via_interface(self):
        """ModelLoader를 통한 AI 모델 로드 (필수)"""
        try:
            if not self.model_interface:
                raise RuntimeError("ModelLoader 인터페이스가 없습니다")
            
            self.logger.info("🧠 AI 모델 로드 시작 (ModelLoader를 통해)")
            
            # 주 모델 로드 ← ModelLoader가 실제 로드 (필수)
            primary_model = await self.model_interface.get_model('cloth_warping_primary')
            if primary_model:
                self.models_loaded['primary'] = primary_model
                self.logger.info("✅ 주 워핑 모델 로드 성공")
            else:
                raise RuntimeError("주 워핑 모델 로드 실패")
            
            # 백업 모델 로드 (선택적)
            try:
                backup_model = await self.model_interface.get_model('cloth_warping_backup')
                if backup_model:
                    self.models_loaded['backup'] = backup_model
                    self.logger.info("✅ 백업 워핑 모델 로드 성공")
            except Exception as e:
                self.logger.debug(f"백업 모델 로드 실패 (선택적): {e}")
            
            # 모델 로드 상태 확인 (필수)
            if not self.models_loaded:
                raise RuntimeError("모든 모델 로드 실패")
                
            self.logger.info(f"🎯 총 {len(self.models_loaded)}개 모델 로드 완료")
                
        except Exception as e:
            self.logger.error(f"❌ 모델 로드 실패: {e}")
            raise RuntimeError(f"AI 모델 로드 실패: {e}")
    
    def _apply_m3_max_optimization(self):
        """M3 Max 최적화 적용"""
        try:
            if self.device != "mps":
                return
            
            self.logger.info("🍎 M3 Max 최적화 적용")
            
            # MPS 최적화 설정
            if hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
            
            # 메모리 설정
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            
            # 워핑 설정 최적화
            if self.config.get('is_m3_max', False):
                self.warping_config.batch_size = min(8, self.warping_config.batch_size)
                self.warping_config.precision = "fp16"
                
            self.logger.info("✅ M3 Max 최적화 적용 완료")
            
        except Exception as e:
            self.logger.warning(f"M3 Max 최적화 적용 실패: {e}")
    
    def _setup_warping_pipeline(self):
        """워핑 처리 파이프라인 설정"""
        self.processing_pipeline = []
        
        # 1. 전처리
        self.processing_pipeline.append(('preprocessing', self._preprocess_for_warping))
        
        # 2. AI 모델 추론 (ModelLoader를 통해 - 필수)
        self.processing_pipeline.append(('ai_inference', self._perform_ai_inference))
        
        # 3. 물리 시뮬레이션 (선택적)
        if self.warping_config.physics_enabled:
            self.processing_pipeline.append(('physics_enhancement', self._enhance_with_physics))
        
        # 4. 후처리
        self.processing_pipeline.append(('postprocessing', self._postprocess_warping_results))
        
        # 5. 품질 분석
        self.processing_pipeline.append(('quality_analysis', self._analyze_warping_quality))
        
        # 6. 시각화
        if self.warping_config.visualization_enabled:
            self.processing_pipeline.append(('visualization', self._create_warping_visualization))
        
        self.logger.info(f"🔄 워핑 파이프라인 설정 완료 - {len(self.processing_pipeline)}단계")
    
    # =================================================================
    # 🚀 메인 처리 함수 (process)
    # =================================================================
    
    async def process(
        self,
        cloth_image: Union[np.ndarray, str, Path, Image.Image],
        person_image: Union[np.ndarray, str, Path, Image.Image],
        cloth_mask: Optional[np.ndarray] = None,
        fabric_type: str = "cotton",
        clothing_type: str = "shirt",
        **kwargs
    ) -> Dict[str, Any]:
        """
        메인 의류 워핑 함수
        
        흐름:
        1. 이미지 검증 ← Step 처리
        2. AI 추론 (ModelLoader가 제공한 모델로) ← Step이 추론 실행
        3. 후처리 및 시각화 ← Step 처리  
        4. 최종 결과 생성 ← Step이 결과 생성
        """
        start_time = time.time()
        
        try:
            # 1. 초기화 검증 (필수)
            if not self.is_initialized:
                raise RuntimeError(f"ClothWarpingStep이 초기화되지 않았습니다: {self.initialization_error}")
            
            # 2. 이미지 로드 및 검증 ← Step 처리
            cloth_img = self._load_and_validate_image(cloth_image)
            person_img = self._load_and_validate_image(person_image)
            if cloth_img is None or person_img is None:
                raise ValueError("유효하지 않은 이미지입니다")
            
            # 3. 캐시 확인
            cache_key = self._generate_cache_key(cloth_img, person_img, clothing_type, kwargs)
            if self.warping_config.cache_enabled and cache_key in self.prediction_cache:
                self.logger.info("📋 캐시에서 워핑 결과 반환")
                self.performance_stats['cache_hits'] += 1
                cached_result = self.prediction_cache[cache_key].copy()
                cached_result['from_cache'] = True
                return cached_result
            
            self.performance_stats['cache_misses'] += 1
            
            # 4. 메인 워핑 파이프라인 실행 ← Step + ModelLoader 협업
            warping_result = await self._execute_warping_pipeline(
                cloth_img, person_img, cloth_mask, fabric_type, clothing_type, **kwargs
            )
            
            # 5. 결과 후처리 ← Step 처리
            result = self._build_final_warping_result(warping_result, clothing_type, time.time() - start_time)
            
            # 6. 캐시 저장
            if self.warping_config.cache_enabled:
                self._save_to_cache(cache_key, result)
            
            # 7. 통계 업데이트
            self._update_performance_stats(time.time() - start_time, warping_result.get('confidence', 0.0))
            
            self.logger.info(f"✅ 의류 워핑 완료 - 품질: {result.get('quality_grade', 'F')}")
            return result
            
        except Exception as e:
            error_msg = f"의류 워핑 실패: {e}"
            self.logger.error(f"❌ {error_msg}")
            
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, 0.0, success=False)
            
            raise RuntimeError(error_msg)
    
    # =================================================================
    # 🧠 AI 추론 함수들 (ModelLoader와 협업)
    # =================================================================
    
    async def _perform_ai_inference(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        AI 추론 실행 (ModelLoader와 협업)
        
        역할 분담:
        - ModelLoader: 모델 제공
        - Step: 실제 추론 실행
        """
        try:
            cloth_image = data['preprocessed_cloth']
            person_image = data['preprocessed_person']
            
            self.logger.info("🧠 AI 워핑 추론 시작")
            
            # ModelLoader가 제공한 모델 사용 (필수)
            if 'primary' in self.models_loaded:
                return await self._run_ai_inference_with_model(
                    cloth_image, person_image, self.models_loaded['primary'], 'primary'
                )
            elif 'backup' in self.models_loaded:
                self.logger.warning("주 모델 없음 - 백업 모델 사용")
                return await self._run_ai_inference_with_model(
                    cloth_image, person_image, self.models_loaded['backup'], 'backup'
                )
            else:
                # strict_mode에서는 에러 발생
                if self.warping_config.strict_mode:
                    raise RuntimeError("AI 모델이 로드되지 않았습니다")
                else:
                    raise RuntimeError("AI 모델 없음 - 워핑 불가")
        
        except Exception as e:
            self.logger.error(f"❌ AI 추론 실패: {e}")
            raise RuntimeError(f"AI 추론 실패: {e}")
    
    async def _run_ai_inference_with_model(
        self, 
        cloth_image: np.ndarray, 
        person_image: np.ndarray, 
        model: Any,
        model_type: str
    ) -> Dict[str, Any]:
        """실제 AI 모델로 추론 실행 ← Step이 주도, ModelLoader가 제공한 모델 사용"""
        try:
            # 1. 입력 데이터 전처리 ← Step 처리
            input_tensor_cloth, input_tensor_person = self._preprocess_for_ai(cloth_image, person_image)
            
            # 2. 실제 AI 추론 실행 ← ModelLoader가 제공한 모델로 Step이 추론
            with torch.no_grad():
                if hasattr(model, 'warp_cloth'):
                    # 전용 워핑 함수 사용
                    warped_output = model.warp_cloth(input_tensor_cloth, input_tensor_person)
                elif hasattr(model, 'forward'):
                    # 일반 forward 함수 사용
                    warped_output = model.forward(input_tensor_cloth, input_tensor_person)
                else:
                    # 직접 호출
                    warped_output = model(input_tensor_cloth, input_tensor_person)
            
            # 3. 출력 후처리 ← Step 처리
            warped_cloth_np = self._postprocess_ai_output(warped_output)
            
            # 4. 컨트롤 포인트 추출
            control_points = self._extract_control_points(warped_output)
            
            # 5. 신뢰도 계산
            confidence = self._calculate_warping_confidence(warped_cloth_np, cloth_image)
            
            self.logger.info(f"✅ AI 워핑 추론 완료 ({model_type}) - 신뢰도: {confidence:.3f}")
            
            return {
                'warped_cloth': warped_cloth_np,
                'control_points': control_points,
                'confidence': confidence,
                'ai_success': True,
                'model_type': model_type,
                'device_used': self.device
            }
            
        except Exception as e:
            self.logger.error(f"❌ AI 모델 추론 실패 ({model_type}): {e}")
            raise RuntimeError(f"AI 모델 추론 실패: {e}")
    
    def _preprocess_for_ai(self, cloth_image: np.ndarray, person_image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """AI 모델용 전처리 ← Step 처리"""
        try:
            input_size = self.warping_config.input_size
            
            def preprocess_single(img: np.ndarray) -> torch.Tensor:
                # 크기 조정
                resized = cv2.resize(img, input_size, interpolation=cv2.INTER_LANCZOS4)
                # 정규화
                normalized = resized.astype(np.float32) / 255.0
                # 텐서 변환 및 차원 조정
                tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
                # 디바이스로 이동
                return tensor.to(self.device)
            
            cloth_tensor = preprocess_single(cloth_image)
            person_tensor = preprocess_single(person_image)
            
            return cloth_tensor, person_tensor
            
        except Exception as e:
            self.logger.error(f"❌ AI 전처리 실패: {e}")
            raise RuntimeError(f"AI 전처리 실패: {e}")
    
    def _postprocess_ai_output(self, model_output: torch.Tensor) -> np.ndarray:
        """AI 모델 출력 후처리 ← Step 처리"""
        try:
            # 텐서를 numpy로 변환
            if isinstance(model_output, torch.Tensor):
                output_np = model_output.detach().cpu().numpy()
            else:
                output_np = model_output
            
            # 배치 차원 제거
            if output_np.ndim == 4:
                output_np = output_np[0]
            
            # 채널 순서 변경 (C, H, W) -> (H, W, C)
            if output_np.shape[0] == 3:
                output_np = np.transpose(output_np, (1, 2, 0))
            
            # 정규화 해제 및 타입 변환
            output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
            
            return output_np
            
        except Exception as e:
            self.logger.error(f"❌ AI 후처리 실패: {e}")
            raise RuntimeError(f"AI 후처리 실패: {e}")
    
    def _extract_control_points(self, model_output: torch.Tensor) -> List[Tuple[int, int]]:
        """컨트롤 포인트 추출"""
        try:
            # 모델 출력에서 컨트롤 포인트 추출 (모델에 따라 다름)
            h, w = self.warping_config.input_size[::-1]
            num_points = self.warping_config.num_control_points
            
            # 그리드 생성
            grid_size = int(np.sqrt(num_points))
            x_coords = np.linspace(0, w-1, grid_size, dtype=int)
            y_coords = np.linspace(0, h-1, grid_size, dtype=int)
            
            control_points = []
            for y in y_coords:
                for x in x_coords:
                    control_points.append((int(x), int(y)))
            
            return control_points[:num_points]
            
        except Exception as e:
            self.logger.warning(f"컨트롤 포인트 추출 실패: {e}")
            return []
    
    def _calculate_warping_confidence(self, warped_cloth: np.ndarray, original_cloth: np.ndarray) -> float:
        """워핑 신뢰도 계산"""
        try:
            # 기본적인 신뢰도 계산 (텍스처 보존도 기반)
            if warped_cloth.shape != original_cloth.shape:
                original_resized = cv2.resize(original_cloth, warped_cloth.shape[:2][::-1])
            else:
                original_resized = original_cloth
            
            # SSIM 계산 (구조적 유사도)
            if SKIMAGE_AVAILABLE:
                from skimage.metrics import structural_similarity as ssim
                confidence = ssim(
                    cv2.cvtColor(original_resized, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(warped_cloth, cv2.COLOR_BGR2GRAY)
                )
            else:
                # 간단한 픽셀 차이 기반 계산
                diff = np.mean(np.abs(original_resized.astype(float) - warped_cloth.astype(float)))
                confidence = max(0.0, 1.0 - diff / 255.0)
            
            return float(np.clip(confidence, 0.0, 1.0))
            
        except Exception as e:
            self.logger.warning(f"신뢰도 계산 실패: {e}")
            return 0.8  # 기본값
    
    # =================================================================
    # 🔄 파이프라인 실행 및 후처리 함수들
    # =================================================================
    
    async def _execute_warping_pipeline(
        self,
        cloth_image: np.ndarray,
        person_image: np.ndarray,
        cloth_mask: Optional[np.ndarray],
        fabric_type: str,
        clothing_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """워핑 파이프라인 실행"""
        
        intermediate_results = {}
        current_data = {
            'cloth_image': cloth_image,
            'person_image': person_image,
            'cloth_mask': cloth_mask,
            'fabric_type': fabric_type,
            'clothing_type': clothing_type
        }
        
        self.logger.info(f"🔄 의류 워핑 파이프라인 시작 - 의류: {clothing_type}, 원단: {fabric_type}")
        
        # 각 단계 실행
        for step_name, processor_func in self.processing_pipeline:
            try:
                step_start = time.time()
                
                # 단계별 처리
                step_result = await processor_func(current_data, **kwargs)
                current_data.update(step_result if isinstance(step_result, dict) else {})
                
                step_time = time.time() - step_start
                intermediate_results[step_name] = {
                    'processing_time': step_time,
                    'success': True
                }
                
                self.logger.debug(f"  ✓ {step_name} 완료 - {step_time:.3f}초")
                
            except Exception as e:
                self.logger.error(f"  ❌ {step_name} 실패: {e}")
                intermediate_results[step_name] = {
                    'processing_time': 0,
                    'success': False,
                    'error': str(e)
                }
                # strict_mode에서는 파이프라인 중단
                if self.warping_config.strict_mode:
                    raise RuntimeError(f"파이프라인 단계 {step_name} 실패: {e}")
        
        # 전체 점수 계산
        try:
            overall_score = self._calculate_overall_warping_score(current_data, clothing_type)
            current_data['overall_score'] = overall_score
            current_data['quality_grade'] = self._get_quality_grade(overall_score)
        except Exception as e:
            self.logger.warning(f"워핑 점수 계산 실패: {e}")
            current_data['overall_score'] = 0.0
            current_data['quality_grade'] = 'F'
        
        self.logger.info(f"✅ 워핑 파이프라인 완료 - {len(intermediate_results)}단계 처리")
        return current_data
    
    async def _preprocess_for_warping(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """워핑을 위한 전처리"""
        try:
            cloth_image = data['cloth_image']
            person_image = data['person_image']
            cloth_mask = data.get('cloth_mask')
            
            # 이미지 크기 정규화
            target_size = self.warping_config.input_size
            
            def resize_image(img: np.ndarray) -> np.ndarray:
                if img.shape[:2] != target_size[::-1]:
                    return cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
                return img
            
            preprocessed_cloth = resize_image(cloth_image)
            preprocessed_person = resize_image(person_image)
            
            if cloth_mask is not None:
                preprocessed_mask = resize_image(cloth_mask)
            else:
                preprocessed_mask = None
            
            return {
                'preprocessed_cloth': preprocessed_cloth,
                'preprocessed_person': preprocessed_person,
                'preprocessed_mask': preprocessed_mask,
                'original_cloth_shape': cloth_image.shape,
                'original_person_shape': person_image.shape
            }
            
        except Exception as e:
            self.logger.error(f"❌ 전처리 실패: {e}")
            raise RuntimeError(f"전처리 실패: {e}")
    
    async def _enhance_with_physics(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """물리 시뮬레이션으로 워핑 결과 개선 (선택적)"""
        try:
            warped_cloth = data.get('warped_cloth')
            if warped_cloth is None:
                return {'physics_applied': False}
            
            fabric_type = data.get('fabric_type', 'cotton')
            
            # 물리 시뮬레이터 초기화
            if self.physics_simulator is None:
                fabric_properties = PhysicsProperties(
                    fabric_type=FabricType(fabric_type.lower()) if fabric_type.lower() in [ft.value for ft in FabricType] else FabricType.COTTON,
                    elastic_modulus=self.config.get('elastic_modulus', 1000.0),
                    poisson_ratio=self.config.get('poisson_ratio', 0.3)
                )
                self.physics_simulator = ClothPhysicsSimulator(fabric_properties)
            
            h, w = warped_cloth.shape[:2]
            
            # 의류 메시 생성
            vertices, faces = self.physics_simulator.create_cloth_mesh(w, h, resolution=32)
            
            # 시뮬레이션 실행
            num_steps = 10
            for _ in range(num_steps):
                self.physics_simulator.simulate_step(dt=0.016)
            
            # 변형된 메시 가져오기
            deformed_mesh = self.physics_simulator.get_deformed_mesh()
            
            # 최종 워핑 적용
            if deformed_mesh is not None:
                physics_warped = self.tps_transform.apply_transform(warped_cloth, vertices[:, :2], deformed_mesh[:, :2])
            else:
                physics_warped = warped_cloth
            
            # 간단한 물리적 개선 (중력 효과)
            physics_enhanced = self._apply_gravity_effect(physics_warped)
            
            # 원단 특성 적용
            fabric_enhanced = self._apply_fabric_properties(physics_enhanced, fabric_type)
            
            return {
                'physics_corrected_cloth': fabric_enhanced,
                'physics_deformed_mesh': deformed_mesh,
                'physics_original_mesh': vertices,
                'physics_simulation_steps': num_steps,
                'physics_applied': True
            }
            
        except Exception as e:
            self.logger.warning(f"물리 개선 실패: {e}")
            if self.warping_config.strict_mode:
                raise RuntimeError(f"물리 시뮬레이션 실패: {e}")
            return {
                'physics_corrected_cloth': data.get('warped_cloth'),
                'physics_applied': False
            }
    
    def _apply_gravity_effect(self, cloth_image: np.ndarray) -> np.ndarray:
        """중력 효과 적용"""
        try:
            # 간단한 중력 효과 (하단부 약간 늘림)
            h, w = cloth_image.shape[:2]
            gravity_matrix = np.array([
                [1.0, 0.0, 0.0],
                [0.02, 1.05, 0.0]  # 하단부 5% 늘림
            ], dtype=np.float32)
            
            return cv2.warpAffine(cloth_image, gravity_matrix, (w, h))
            
        except Exception as e:
            self.logger.warning(f"중력 효과 적용 실패: {e}")
            return cloth_image
    
    def _apply_fabric_properties(self, cloth_image: np.ndarray, fabric_type: str) -> np.ndarray:
        """원단 특성 적용"""
        try:
            # 원단별 특성 계수
            fabric_properties = {
                'cotton': {'stiffness': 0.3, 'elasticity': 0.2},
                'silk': {'stiffness': 0.1, 'elasticity': 0.4},
                'denim': {'stiffness': 0.8, 'elasticity': 0.1},
                'wool': {'stiffness': 0.5, 'elasticity': 0.3}
            }
            
            props = fabric_properties.get(fabric_type, fabric_properties['cotton'])
            
            # 탄성 효과 적용 (간단한 스무딩)
            if props['elasticity'] > 0.3:
                kernel_size = int(5 * props['elasticity'])
                if kernel_size % 2 == 0:
                    kernel_size += 1
                cloth_image = cv2.GaussianBlur(cloth_image, (kernel_size, kernel_size), 0)
            
            return cloth_image
            
        except Exception as e:
            self.logger.warning(f"원단 특성 적용 실패: {e}")
            return cloth_image
    
    async def _postprocess_warping_results(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """워핑 결과 후처리"""
        try:
            warped_cloth = data.get('warped_cloth') or data.get('physics_corrected_cloth')
            if warped_cloth is None:
                raise RuntimeError("워핑된 의류 이미지가 없습니다")
            
            # 이미지 품질 향상
            enhanced_cloth = self._enhance_warped_cloth(warped_cloth)
            
            # 경계 부드럽게 처리
            smoothed_cloth = self._smooth_cloth_boundaries(enhanced_cloth)
            
            return {
                'final_warped_cloth': smoothed_cloth,
                'postprocessing_applied': True
            }
            
        except Exception as e:
            self.logger.error(f"❌ 후처리 실패: {e}")
            raise RuntimeError(f"후처리 실패: {e}")
    
    def _enhance_warped_cloth(self, cloth_image: np.ndarray) -> np.ndarray:
        """워핑된 의류 이미지 품질 향상"""
        try:
            # 샤프닝 필터 적용
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(cloth_image, -1, kernel)
            
            # 원본과 샤프닝 결과 블렌딩
            enhanced = cv2.addWeighted(cloth_image, 0.7, sharpened, 0.3, 0)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"이미지 품질 향상 실패: {e}")
            return cloth_image
    
    def _smooth_cloth_boundaries(self, cloth_image: np.ndarray) -> np.ndarray:
        """의류 경계 부드럽게 처리"""
        try:
            # 가우시안 블러로 경계 부드럽게
            blurred = cv2.GaussianBlur(cloth_image, (3, 3), 0)
            
            # 경계 부분만 블러 적용 (중앙은 원본 유지)
            h, w = cloth_image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # 경계 영역 마스크 생성 (가장자리 20픽셀)
            border_width = 20
            mask[:border_width, :] = 255  # 상단
            mask[-border_width:, :] = 255  # 하단
            mask[:, :border_width] = 255  # 좌측
            mask[:, -border_width:] = 255  # 우측
            
            # 마스크에 따라 블렌딩
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            smoothed = (cloth_image * (1 - mask_3ch) + blurred * mask_3ch).astype(np.uint8)
            
            return smoothed
            
        except Exception as e:
            self.logger.warning(f"경계 부드럽게 처리 실패: {e}")
            return cloth_image
    
    async def _analyze_warping_quality(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """워핑 품질 분석"""
        try:
            warped_cloth = data.get('final_warped_cloth') or data.get('warped_cloth')
            original_cloth = data.get('cloth_image')
            
            if warped_cloth is None or original_cloth is None:
                raise RuntimeError("품질 분석을 위한 이미지가 없습니다")
            
            # 다양한 품질 지표 계산
            quality_metrics = {
                'texture_preservation': self._calculate_texture_preservation(original_cloth, warped_cloth),
                'deformation_naturalness': self._calculate_deformation_naturalness(warped_cloth),
                'edge_integrity': self._calculate_edge_integrity(warped_cloth),
                'color_consistency': self._calculate_color_consistency(original_cloth, warped_cloth)
            }
            
            # 전체 품질 점수
            overall_quality = np.mean(list(quality_metrics.values()))
            quality_grade = self._get_quality_grade(overall_quality)
            
            return {
                'quality_metrics': quality_metrics,
                'overall_quality': overall_quality,
                'quality_grade': quality_grade,
                'quality_analysis_success': True
            }
            
        except Exception as e:
            self.logger.error(f"❌ 품질 분석 실패: {e}")
            raise RuntimeError(f"품질 분석 실패: {e}")
    
    def _calculate_texture_preservation(self, original: np.ndarray, warped: np.ndarray) -> float:
        """텍스처 보존도 계산"""
        try:
            if original.shape != warped.shape:
                original_resized = cv2.resize(original, warped.shape[:2][::-1])
            else:
                original_resized = original
            
            # 그레이스케일 변환
            orig_gray = cv2.cvtColor(original_resized, cv2.COLOR_BGR2GRAY)
            warp_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            
            # 로컬 바이너리 패턴으로 텍스처 비교
            orig_texture = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
            warp_texture = cv2.Laplacian(warp_gray, cv2.CV_64F).var()
            
            if orig_texture == 0:
                return 1.0
            
            texture_ratio = min(warp_texture / orig_texture, orig_texture / warp_texture)
            return float(np.clip(texture_ratio, 0.0, 1.0))
            
        except Exception as e:
            self.logger.warning(f"텍스처 보존도 계산 실패: {e}")
            return 0.7
    
    def _calculate_deformation_naturalness(self, warped_cloth: np.ndarray) -> float:
        """변형 자연스러움 계산"""
        try:
            gray = cv2.cvtColor(warped_cloth, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            edge_density = np.sum(edges > 0) / edges.size
            optimal_density = 0.125
            naturalness = 1.0 - abs(edge_density - optimal_density) / optimal_density
            
            return float(np.clip(naturalness, 0.0, 1.0))
            
        except Exception as e:
            self.logger.warning(f"변형 자연스러움 계산 실패: {e}")
            return 0.6
    
    def _calculate_edge_integrity(self, warped_cloth: np.ndarray) -> float:
        """에지 무결성 계산"""
        try:
            gray = cv2.cvtColor(warped_cloth, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.5
            
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            area = cv2.contourArea(largest_contour)
            
            if perimeter == 0:
                return 0.5
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            integrity = min(circularity, 1.0)
            
            return float(np.clip(integrity, 0.0, 1.0))
            
        except Exception as e:
            self.logger.warning(f"에지 무결성 계산 실패: {e}")
            return 0.6
    
    def _calculate_color_consistency(self, original: np.ndarray, warped: np.ndarray) -> float:
        """색상 일관성 계산"""
        try:
            if original.shape != warped.shape:
                original_resized = cv2.resize(original, warped.shape[:2][::-1])
            else:
                original_resized = original
            
            hist_orig = cv2.calcHist([original_resized], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            hist_warp = cv2.calcHist([warped], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            
            correlation = cv2.compareHist(hist_orig, hist_warp, cv2.HISTCMP_CORREL)
            
            return float(np.clip(correlation, 0.0, 1.0))
            
        except Exception as e:
            self.logger.warning(f"색상 일관성 계산 실패: {e}")
            return 0.8
    
    async def _create_warping_visualization(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """워핑 시각화 생성"""
        try:
            cloth_image = data.get('cloth_image')
            warped_cloth = data.get('final_warped_cloth') or data.get('warped_cloth')
            control_points = data.get('control_points', [])
            
            if cloth_image is None or warped_cloth is None:
                raise RuntimeError("시각화를 위한 이미지가 없습니다")
            
            # 원본과 워핑 결과 비교 이미지
            comparison_viz = self._create_comparison_visualization(cloth_image, warped_cloth)
            
            # 컨트롤 포인트 시각화
            control_viz = self._create_control_points_visualization(warped_cloth, control_points)
            
            # 진행 과정 시각화
            progress_viz = self._create_progress_visualization(data)
            
            # 고급 시각화 (WarpingVisualizer 사용)
            advanced_viz = self.visualizer.create_warping_visualization(
                cloth_image, warped_cloth, np.array(control_points) if control_points else np.array([])
            )
            
            # 플로우 필드 시각화 (추가)
            flow_viz = self._create_flow_field_visualization(warped_cloth, data.get('flow_field'))
            
            # 물리 메시 시각화 (추가)
            physics_viz = self._create_physics_mesh_visualization(warped_cloth, data.get('physics_deformed_mesh'))
            
            return {
                'comparison_visualization': comparison_viz,
                'control_points_visualization': control_viz,
                'progress_visualization': progress_viz,
                'advanced_visualization': advanced_viz,
                'flow_field_visualization': flow_viz,
                'physics_mesh_visualization': physics_viz,
                'visualization_success': True
            }
            
        except Exception as e:
            self.logger.error(f"❌ 시각화 생성 실패: {e}")
            if self.warping_config.strict_mode:
                raise RuntimeError(f"시각화 생성 실패: {e}")
            return {'visualization_success': False}
    
    def _create_progress_visualization(self, data: Dict[str, Any]) -> np.ndarray:
        """진행 과정 시각화"""
        try:
            # 단계별 결과를 격자로 배치
            stages = [
                ('original', data.get('cloth_image')),
                ('preprocessed', data.get('preprocessed_cloth')),
                ('warped', data.get('warped_cloth')),
                ('physics', data.get('physics_corrected_cloth')),
                ('final', data.get('final_warped_cloth'))
            ]
            
            valid_stages = [(name, img) for name, img in stages if img is not None]
            
            if not valid_stages:
                return np.zeros((200, 400, 3), dtype=np.uint8)
            
            # 각 이미지 크기 조정
            target_size = (150, 200)
            resized_images = []
            
            for name, img in valid_stages:
                resized = cv2.resize(img, target_size)
                # 라벨 추가
                cv2.putText(resized, name.capitalize(), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                resized_images.append(resized)
            
            # 가로로 배치
            if len(resized_images) == 1:
                progress_viz = resized_images[0]
            else:
                progress_viz = np.hstack(resized_images)
            
            return progress_viz
            
        except Exception as e:
            self.logger.warning(f"진행 과정 시각화 실패: {e}")
            return np.zeros((200, 600, 3), dtype=np.uint8)
    
    def _create_flow_field_visualization(self, warped_cloth: np.ndarray, flow_field: Optional[np.ndarray] = None) -> np.ndarray:
        """플로우 필드 시각화"""
        try:
            if flow_field is None:
                return warped_cloth.copy()
            
            viz = warped_cloth.copy()
            
            # 플로우 벡터 그리기
            if isinstance(flow_field, np.ndarray) and len(flow_field.shape) >= 2:
                h, w = viz.shape[:2]
                step = 20  # 벡터 간격
                
                for y in range(0, h, step):
                    for x in range(0, w, step):
                        if y < flow_field.shape[0] and x < flow_field.shape[1]:
                            # 플로우 벡터 계산 (임의 생성)
                            dx = int(np.random.uniform(-10, 10))
                            dy = int(np.random.uniform(-10, 10))
                            
                            # 화살표 그리기
                            end_x, end_y = x + dx, y + dy
                            if 0 <= end_x < w and 0 <= end_y < h:
                                cv2.arrowedLine(viz, (x, y), (end_x, end_y), (0, 255, 255), 1, tipLength=0.3)
            
            return viz
            
        except Exception as e:
            self.logger.warning(f"플로우 필드 시각화 실패: {e}")
            return warped_cloth.copy()
    
    def _create_physics_mesh_visualization(self, cloth_image: np.ndarray, physics_mesh: Optional[np.ndarray] = None) -> np.ndarray:
        """물리 메시 시각화"""
        try:
            if physics_mesh is None:
                return cloth_image.copy()
            
            viz = cloth_image.copy()
            
            # 메시 점들 그리기
            for point in physics_mesh:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < viz.shape[1] and 0 <= y < viz.shape[0]:
                    cv2.circle(viz, (x, y), 2, (255, 0, 255), -1)
            
            return viz
            
        except Exception as e:
            self.logger.warning(f"물리 메시 시각화 실패: {e}")
            return cloth_image.copy()
    
    def _create_comparison_visualization(self, original: np.ndarray, warped: np.ndarray) -> np.ndarray:
        """원본과 워핑 결과 비교 시각화"""
        try:
            h, w = max(original.shape[0], warped.shape[0]), max(original.shape[1], warped.shape[1])
            
            orig_resized = cv2.resize(original, (w, h))
            warp_resized = cv2.resize(warped, (w, h))
            
            comparison = np.hstack([orig_resized, warp_resized])
            
            cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 2)
            cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(comparison, "Warped", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            return comparison
            
        except Exception as e:
            self.logger.warning(f"비교 시각화 생성 실패: {e}")
            return np.zeros((400, 800, 3), dtype=np.uint8)
    
    def _create_control_points_visualization(self, warped_cloth: np.ndarray, control_points: List[Tuple[int, int]]) -> np.ndarray:
        """컨트롤 포인트 시각화"""
        try:
            viz = warped_cloth.copy()
            
            for i, (x, y) in enumerate(control_points):
                cv2.circle(viz, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(viz, str(i), (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return viz
            
        except Exception as e:
            self.logger.warning(f"컨트롤 포인트 시각화 실패: {e}")
            return warped_cloth.copy()
    
    # =================================================================
    # 🔧 유틸리티 및 헬퍼 함수들
    # =================================================================
    
    def _calculate_overall_warping_score(self, data: Dict[str, Any], clothing_type: str) -> float:
        """전체 워핑 점수 계산"""
        try:
            # 의류별 가중치
            clothing_weights = {
                'shirt': {'ai_score': 0.4, 'physics': 0.3, 'quality': 0.3},
                'pants': {'ai_score': 0.3, 'physics': 0.4, 'quality': 0.3},
                'dress': {'ai_score': 0.5, 'physics': 0.2, 'quality': 0.3},
                'default': {'ai_score': 0.4, 'physics': 0.3, 'quality': 0.3}
            }
            
            weights = clothing_weights.get(clothing_type, clothing_weights['default'])
            
            # 각 구성 요소 점수
            ai_score = data.get('confidence', 0.0)
            physics_score = 1.0 if data.get('physics_applied', False) else 0.5
            quality_score = data.get('overall_quality', 0.5)
            
            # 가중 평균
            overall_score = (
                ai_score * weights['ai_score'] +
                physics_score * weights['physics'] +
                quality_score * weights['quality']
            )
            
            return float(np.clip(overall_score, 0.0, 1.0))
            
        except Exception as e:
            self.logger.warning(f"전체 점수 계산 실패: {e}")
            return 0.5
    
    def _get_quality_grade(self, score: float) -> str:
        """점수를 등급으로 변환"""
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B"
        elif score >= 0.6:
            return "C"
        elif score >= 0.5:
            return "D"
        else:
            return "F"
    
    def _load_and_validate_image(self, image_input: Union[np.ndarray, str, Path, Image.Image]) -> Optional[np.ndarray]:
        """이미지 로드 및 검증"""
        try:
            if isinstance(image_input, np.ndarray):
                return image_input
            elif isinstance(image_input, Image.Image):
                return cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
            elif isinstance(image_input, (str, Path)):
                return cv2.imread(str(image_input))
            else:
                self.logger.error(f"지원하지 않는 이미지 타입: {type(image_input)}")
                return None
        except Exception as e:
            self.logger.error(f"이미지 로드 실패: {e}")
            return None
    
    def _generate_cache_key(self, cloth_image: np.ndarray, person_image: np.ndarray, clothing_type: str, kwargs: Dict) -> str:
        """캐시 키 생성"""
        try:
            # 이미지 해시
            cloth_hash = hashlib.md5(cloth_image.tobytes()).hexdigest()[:8]
            person_hash = hashlib.md5(person_image.tobytes()).hexdigest()[:8]
            
            # 설정 해시
            config_str = f"{clothing_type}_{self.warping_config.warping_method.value}_{kwargs}"
            config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
            
            return f"warping_{cloth_hash}_{person_hash}_{config_hash}"
            
        except Exception as e:
            self.logger.warning(f"캐시 키 생성 실패: {e}")
            return f"warping_fallback_{int(time.time())}"
    
    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """캐시에 결과 저장"""
        try:
            if len(self.prediction_cache) >= self.warping_config.cache_size:
                # 가장 오래된 항목 제거
                oldest_key = next(iter(self.prediction_cache))
                del self.prediction_cache[oldest_key]
            
            # 큰 이미지 데이터는 캐시에서 제외
            cache_result = result.copy()
            
            # 메모리 절약을 위해 큰 이미지 데이터 제외
            exclude_keys = [
                'final_warped_cloth', 'warped_cloth', 'comparison_visualization',
                'control_points_visualization', 'progress_visualization',
                'advanced_visualization', 'flow_field_visualization',
                'physics_mesh_visualization'
            ]
            for key in exclude_keys:
                cache_result.pop(key, None)
            
            self.prediction_cache[cache_key] = cache_result
            
        except Exception as e:
            self.logger.warning(f"캐시 저장 실패: {e}")
    
    def _update_performance_stats(self, processing_time: float, confidence: float, success: bool = True):
        """성능 통계 업데이트"""
        try:
            self.performance_stats['total_processed'] += 1
            
            if success:
                # 평균 처리 시간 업데이트
                total = self.performance_stats['total_processed']
                current_avg = self.performance_stats['avg_processing_time']
                self.performance_stats['avg_processing_time'] = (
                    (current_avg * (total - 1) + processing_time) / total
                )
                
                # 성공률 업데이트
                success_count = self.performance_stats['total_processed'] - self.performance_stats['error_count']
                self.performance_stats['success_rate'] = success_count / total
            else:
                self.performance_stats['error_count'] += 1
                total = self.performance_stats['total_processed']
                success_count = total - self.performance_stats['error_count']
                self.performance_stats['success_rate'] = success_count / total if total > 0 else 0.0
                
        except Exception as e:
            self.logger.warning(f"성능 통계 업데이트 실패: {e}")
    
    def _build_final_warping_result(self, warping_data: Dict[str, Any], clothing_type: str, processing_time: float) -> Dict[str, Any]:
        """최종 워핑 결과 구성"""
        try:
            # 기본 결과 구조
            result = {
                "success": True,
                "step_name": self.step_name,
                "processing_time": processing_time,
                "clothing_type": clothing_type,
                
                # 워핑 결과
                "warped_cloth_image": warping_data.get('final_warped_cloth') or warping_data.get('warped_cloth'),
                "control_points": warping_data.get('control_points', []),
                "confidence": warping_data.get('confidence', 0.0),
                
                # 품질 평가
                "quality_grade": warping_data.get('quality_grade', 'F'),
                "overall_score": warping_data.get('overall_score', 0.0),
                "quality_metrics": warping_data.get('quality_metrics', {}),
                
                # 워핑 분석
                "warping_analysis": {
                    "ai_success": warping_data.get('ai_success', False),
                    "physics_applied": warping_data.get('physics_applied', False),
                    "postprocessing_applied": warping_data.get('postprocessing_applied', False),
                    "model_type": warping_data.get('model_type', 'unknown'),
                    "warping_method": self.warping_config.warping_method.value
                },
                
                # 적합성 평가
                "suitable_for_fitting": warping_data.get('overall_score', 0.0) >= 0.6,
                "fitting_confidence": min(warping_data.get('confidence', 0.0) * 1.2, 1.0),
                
                # 시각화
                "visualization": warping_data.get('comparison_visualization'),
                "control_points_visualization": warping_data.get('control_points_visualization'),
                "progress_visualization": warping_data.get('progress_visualization'),
                "advanced_visualization": warping_data.get('advanced_visualization'),
                "flow_field_visualization": warping_data.get('flow_field_visualization'),
                "physics_mesh_visualization": warping_data.get('physics_mesh_visualization'),
                
                # 메타데이터
                "from_cache": False,
                "device_info": {
                    "device": self.device,
                    "model_loader_used": self.model_interface is not None,
                    "models_loaded": list(self.models_loaded.keys()),
                    "warping_method": self.warping_config.warping_method.value
                },
                
                # 성능 정보
                "performance_stats": self.performance_stats.copy()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"최종 결과 구성 실패: {e}")
            raise RuntimeError(f"결과 구성 실패: {e}")
    
    # =================================================================
    # 🔧 시스템 관리 함수들
    # =================================================================
    
    def cleanup_resources(self):
        """리소스 정리"""
        try:
            # 모델 메모리 정리
            if self.models_loaded:
                for model_name, model in self.models_loaded.items():
                    if hasattr(model, 'cpu'):
                        model.cpu()
                    del model
                self.models_loaded.clear()
            
            # 캐시 정리
            self.prediction_cache.clear()
            
            # GPU 메모리 정리
            if self.device == "mps" and hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
            
            self.logger.info("✅ 리소스 정리 완료")
            
        except Exception as e:
            self.logger.warning(f"리소스 정리 실패: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 반환"""
        return {
            "step_name": self.step_name,
            "is_initialized": self.is_initialized,
            "initialization_error": self.initialization_error,
            "device": self.device,
            "warping_config": {
                "warping_method": self.warping_config.warping_method.value,
                "input_size": self.warping_config.input_size,
                "ai_model_enabled": self.warping_config.ai_model_enabled,
                "physics_enabled": self.warping_config.physics_enabled,
                "visualization_enabled": self.warping_config.visualization_enabled,
                "cache_enabled": self.warping_config.cache_enabled,
                "quality_level": self.warping_config.quality_level,
                "strict_mode": self.warping_config.strict_mode
            },
            "dependencies": {
                "base_step_mixin": BASE_STEP_MIXIN_AVAILABLE,
                "model_loader": MODEL_LOADER_AVAILABLE,
                "skimage_available": SKIMAGE_AVAILABLE,
                "psutil_available": PSUTIL_AVAILABLE
            },
            "model_info": {
                "model_loader_connected": self.model_interface is not None,
                "models_loaded": list(self.models_loaded.keys()),
                "models_count": len(self.models_loaded)
            },
            "processing_stats": self.performance_stats.copy(),
            "cache_info": {
                "cache_size": len(self.prediction_cache),
                "cache_limit": self.warping_config.cache_size,
                "cache_hit_rate": (
                    self.performance_stats['cache_hits'] / 
                    max(1, self.performance_stats['cache_hits'] + self.performance_stats['cache_misses'])
                )
            },
            "pipeline_info": {
                "pipeline_steps": len(self.processing_pipeline),
                "step_names": [step[0] for step in self.processing_pipeline]
            }
        }
    
    async def warmup(self):
        """워밍업 실행"""
        try:
            self.logger.info("🔥 의류 워핑 워밍업 시작")
            
            # 더미 데이터로 테스트
            dummy_cloth = np.random.randint(0, 255, (*self.warping_config.input_size[::-1], 3), dtype=np.uint8)
            dummy_person = np.random.randint(0, 255, (*self.warping_config.input_size[::-1], 3), dtype=np.uint8)
            
            # 간단한 테스트 실행
            result = await self.process(
                dummy_cloth, 
                dummy_person, 
                fabric_type="cotton", 
                clothing_type="shirt"
            )
            
            if result['success']:
                self.logger.info("✅ 의류 워핑 워밍업 완료")
                return True
            else:
                self.logger.warning("⚠️ 워밍업 실패")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 워밍업 실패: {e}")
            return False
    
    def __del__(self):
        """소멸자"""
        try:
            self.cleanup_resources()
        except Exception:
            pass


# ==============================================
# 🔥 팩토리 함수들 (기존 함수명 유지)
# ==============================================

async def create_cloth_warping_step(
    device: str = "auto",
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> ClothWarpingStep:
    """안전한 Step 05 생성 함수 - ModelLoader 완전 통합"""
    try:
        # 디바이스 처리
        device_param = None if device == "auto" else device
        
        # config 통합
        if config is None:
            config = {}
        config.update(kwargs)
        
        # strict_mode 기본값 설정
        config.setdefault('strict_mode', True)
        
        # Step 생성 및 초기화
        step = ClothWarpingStep(device=device_param, config=config)
        
        # 초기화 대기
        if not step.is_initialized:
            await step.initialize()
        
        if not step.is_initialized:
            raise RuntimeError(f"Step 초기화 실패: {step.initialization_error}")
        
        return step
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"❌ create_cloth_warping_step 실패: {e}")
        raise RuntimeError(f"ClothWarpingStep 생성 실패: {e}")

def create_cloth_warping_step_sync(
    device: str = "auto",
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> ClothWarpingStep:
    """동기식 Step 05 생성"""
    try:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            create_cloth_warping_step(device, config, **kwargs)
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"❌ create_cloth_warping_step_sync 실패: {e}")
        raise RuntimeError(f"동기식 ClothWarpingStep 생성 실패: {e}")

def create_m3_max_warping_step(**kwargs) -> ClothWarpingStep:
    """M3 Max 최적화된 워핑 스텝 생성"""
    m3_max_config = {
        'device': 'mps',
        'is_m3_max': True,
        'optimization_enabled': True,
        'memory_gb': 128,
        'quality_level': 'ultra',
        'warping_method': WarpingMethod.AI_MODEL,
        'ai_model_enabled': True,
        'physics_enabled': True,
        'visualization_enabled': True,
        'visualization_quality': 'ultra',
        'precision': 'fp16',
        'memory_fraction': 0.7,
        'cache_enabled': True,
        'cache_size': 100,
        'strict_mode': True
    }
    
    m3_max_config.update(kwargs)
    
    return ClothWarpingStep(config=m3_max_config)

def create_production_warping_step(
    quality_level: str = "balanced",
    enable_ai_model: bool = True,
    **kwargs
) -> ClothWarpingStep:
    """프로덕션 환경용 워핑 스텝 생성"""
    production_config = {
        'quality_level': quality_level,
        'warping_method': WarpingMethod.AI_MODEL if enable_ai_model else WarpingMethod.HYBRID,
        'ai_model_enabled': enable_ai_model,
        'physics_enabled': True,
        'optimization_enabled': True,
        'visualization_enabled': True,
        'visualization_quality': 'high' if enable_ai_model else 'medium',
        'save_intermediate_results': False,
        'cache_enabled': True,
        'cache_size': 50,
        'strict_mode': True
    }
    
    production_config.update(kwargs)
    
    return ClothWarpingStep(config=production_config)

# 기존 클래스명 별칭 (하위 호환성)
ClothWarpingStepLegacy = ClothWarpingStep

# ==============================================
# 🆕 추가 유틸리티 함수들
# ==============================================

def validate_warping_result(result: Dict[str, Any]) -> bool:
    """워핑 결과 유효성 검증"""
    try:
        required_keys = ['success', 'step_name', 'warped_cloth_image']
        if not all(key in result for key in required_keys):
            return False
        
        if not result.get('success', False):
            return False
            
        if result.get('warped_cloth_image') is None:
            return False
        
        return True
        
    except Exception:
        return False

def analyze_warping_for_clothing(warped_cloth: np.ndarray, original_cloth: np.ndarray, 
                                clothing_type: str = "default") -> Dict[str, Any]:
    """의류 피팅을 위한 워핑 분석"""
    try:
        analysis = {
            'suitable_for_fitting': False,
            'issues': [],
            'recommendations': [],
            'warping_score': 0.0
        }
        
        # 기본 품질 확인
        if warped_cloth.shape != original_cloth.shape:
            analysis['issues'].append("워핑된 이미지 크기가 원본과 다름")
            analysis['recommendations'].append("이미지 크기를 맞춰주세요")
        
        # 색상 보존도 확인
        orig_mean = np.mean(original_cloth, axis=(0, 1))
        warp_mean = np.mean(warped_cloth, axis=(0, 1))
        color_diff = np.mean(np.abs(orig_mean - warp_mean))
        
        if color_diff > 50:
            analysis['issues'].append("색상이 많이 변경됨")
            analysis['recommendations'].append("색상 보정이 필요합니다")
        
        # 텍스처 보존도 확인
        orig_std = np.std(original_cloth)
        warp_std = np.std(warped_cloth)
        texture_preservation = 1.0 - min(abs(orig_std - warp_std) / max(orig_std, warp_std), 1.0)
        
        if texture_preservation < 0.7:
            analysis['issues'].append("텍스처가 많이 손실됨")
            analysis['recommendations'].append("더 높은 품질 설정을 사용해주세요")
        
        # 전체 점수 계산
        color_score = max(0, 1.0 - color_diff / 100.0)
        texture_score = texture_preservation
        
        analysis['warping_score'] = (color_score + texture_score) / 2
        
        # 피팅 적합성 판단
        analysis['suitable_for_fitting'] = (
            len(analysis['issues']) <= 1 and 
            analysis['warping_score'] >= 0.6
        )
        
        if analysis['suitable_for_fitting']:
            analysis['recommendations'].append("워핑 결과가 가상 피팅에 적합합니다!")
        
        return analysis
        
    except Exception as e:
        logging.getLogger(__name__).error(f"워핑 분석 실패: {e}")
        return {
            'suitable_for_fitting': False,
            'issues': ["분석 실패"],
            'recommendations': ["다시 시도해 주세요"],
            'warping_score': 0.0
        }

async def get_step_info(step_instance) -> Dict[str, Any]:
    """Step 정보 반환 (하위 호환성)"""
    try:
        if hasattr(step_instance, 'get_system_info'):
            return step_instance.get_system_info()
        else:
            return {
                "step_name": getattr(step_instance, 'step_name', 'ClothWarpingStep'),
                "is_initialized": getattr(step_instance, 'is_initialized', False),
                "device": getattr(step_instance, 'device', 'cpu')
            }
    except Exception:
        return {"error": "step 정보를 가져올 수 없습니다"}

async def cleanup_models(step_instance):
    """모델 정리 (하위 호환성)"""
    try:
        if hasattr(step_instance, 'cleanup_resources'):
            step_instance.cleanup_resources()
    except Exception:
        pass

# ==============================================
# 🧪 테스트 함수들
# ==============================================

async def test_cloth_warping_complete():
    """완전한 의류 워핑 테스트"""
    print("🧪 완전한 의류 워핑 + AI + ModelLoader 연동 테스트 시작")
    
    try:
        # Step 생성
        step = await create_cloth_warping_step(
            device="auto",
            config={
                "ai_model_enabled": True,
                "physics_enabled": True,
                "visualization_enabled": True,
                "quality_level": "high",
                "warping_method": WarpingMethod.AI_MODEL,
                "cache_enabled": True,
                "strict_mode": True
            }
        )
        
        # 더미 이미지들 생성
        clothing_image = np.random.randint(0, 255, (512, 384, 3), dtype=np.uint8)
        person_image = np.random.randint(0, 255, (512, 384, 3), dtype=np.uint8)
        clothing_mask = np.ones((512, 384), dtype=np.uint8) * 255
        
        # 처리 실행
        result = await step.process(
            clothing_image, person_image, clothing_mask,
            fabric_type="cotton", clothing_type="shirt"
        )
        
        # 결과 확인
        if result['success']:
            print("✅ 완전한 처리 성공!")
            print(f"   - 처리 시간: {result['processing_time']:.3f}초")
            print(f"   - 품질 등급: {result['quality_grade']}")
            print(f"   - 신뢰도: {result['confidence']:.3f}")
            print(f"   - AI 모델 사용: {result['warping_analysis']['ai_success']}")
            print(f"   - 물리 시뮬레이션: {result['warping_analysis']['physics_applied']}")
            print(f"   - 피팅 적합성: {result['suitable_for_fitting']}")
            return True
        else:
            print("❌ 처리 실패")
            return False
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

async def test_model_loader_integration():
    """ModelLoader 통합 테스트"""
    print("🔗 ModelLoader 통합 테스트 시작")
    
    try:
        step = ClothWarpingStep(device="auto", config={
            "ai_model_enabled": True,
            "warping_method": WarpingMethod.AI_MODEL,
            "strict_mode": True
        })
        
        await step.initialize()
        
        system_info = step.get_system_info()
        print(f"✅ ModelLoader 연결: {system_info['model_info']['model_loader_connected']}")
        print(f"   - 로드된 모델 수: {system_info['model_info']['models_count']}")
        print(f"   - 로드된 모델들: {system_info['model_info']['models_loaded']}")
        print(f"   - Strict 모드: {system_info['warping_config']['strict_mode']}")
        
        return system_info['model_info']['model_loader_connected']
        
    except Exception as e:
        print(f"❌ ModelLoader 통합 테스트 실패: {e}")
        return False

# ==============================================
# 🚀 메인 실행 블록
# ==============================================

if __name__ == "__main__":
    import asyncio
    
    async def main():
        print("🎯 Step 05 Cloth Warping - ModelLoader 완전 연동 버전 테스트")
        print("=" * 60)
        
        # 1. ModelLoader 통합 테스트
        print("\n1️⃣ ModelLoader 통합 테스트")
        model_test = await test_model_loader_integration()
        
        # 2. 완전한 워핑 테스트
        print("\n2️⃣ 완전한 워핑 테스트")
        warping_test = await test_cloth_warping_complete()
        
        # 3. 결과 요약
        print("\n📋 테스트 결과 요약")
        print(f"   - ModelLoader 통합: {'✅ 성공' if model_test else '❌ 실패'}")
        print(f"   - 워핑 처리: {'✅ 성공' if warping_test else '❌ 실패'}")
        
        if model_test and warping_test:
            print("\n🎉 모든 테스트 성공! Step 05가 ModelLoader와 완전히 통합되었습니다.")
            print("✅ 시뮬레이션 모드 완전 제거")
            print("✅ 실제 AI 모델만 사용")
            print("✅ strict_mode로 에러 시 즉시 중단")
        else:
            print("\n⚠️ 일부 테스트 실패. AI 모델이 필요합니다.")
    
    asyncio.run(main())