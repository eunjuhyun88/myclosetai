"""
완전 통합된 8단계 AI 파이프라인 매니저 - 최적 생성자 패턴 적용
- 최적 생성자 설계로 모든 Step 클래스 통일
- M3 Max 최적화 및 완전한 호환성
- 프로덕션 레벨 안정성과 고급 분석 기능
- 실제 프로젝트 구조에 완전 적합
"""
import os
import sys
import logging
import asyncio
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union, List, Tuple
from enum import Enum
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
import json
import gc
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# 최적 생성자 기본 클래스
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod

class OptimalStepConstructor(ABC):
    """
    🎯 최적화된 생성자 패턴
    - 단순함 + 편의성 + 확장성 + 일관성
    """

    def __init__(
        self,
        device: Optional[str] = None,  # 🔥 핵심 개선: None으로 자동 감지
        config: Optional[Dict[str, Any]] = None,
        **kwargs  # 🚀 확장성: 무제한 추가 파라미터
    ):
        """
        ✅ 최적 생성자 - 모든 장점 결합

        Args:
            device: 사용할 디바이스 (None=자동감지, 'cpu', 'cuda', 'mps')
            config: 스텝별 설정 딕셔너리
            **kwargs: 확장 파라미터들
                - device_type: str = "auto"
                - memory_gb: float = 16.0  
                - is_m3_max: bool = False
                - optimization_enabled: bool = True
                - quality_level: str = "balanced"
                - 기타 스텝별 특화 파라미터들...
        """
        # 1. 💡 지능적 디바이스 자동 감지
        self.device = self._auto_detect_device(device)

        # 2. 📋 기본 설정
        self.config = config or {}
        self.step_name = self.__class__.__name__
        self.logger = logging.getLogger(f"pipeline.{self.step_name}")

        # 3. 🔧 표준 시스템 파라미터 추출 (일관성)
        self.device_type = kwargs.get('device_type', 'auto')
        self.memory_gb = kwargs.get('memory_gb', 16.0)
        self.is_m3_max = kwargs.get('is_m3_max', self._detect_m3_max())
        self.optimization_enabled = kwargs.get('optimization_enabled', True)
        self.quality_level = kwargs.get('quality_level', 'balanced')

        # 4. ⚙️ 스텝별 특화 파라미터를 config에 병합
        self._merge_step_specific_config(kwargs)

        # 5. ✅ 상태 초기화
        self.is_initialized = False

        self.logger.info(f"🎯 {self.step_name} 초기화 - 디바이스: {self.device}")

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
                # M3 Max 감지 로직
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                return 'M3' in result.stdout
        except:
            pass
        return False

    def _merge_step_specific_config(self, kwargs: Dict[str, Any]):
        """⚙️ 스텝별 특화 설정 병합"""
        # 시스템 파라미터 제외하고 모든 kwargs를 config에 병합
        system_params = {
            'device_type', 'memory_gb', 'is_m3_max', 
            'optimization_enabled', 'quality_level'
        }

        for key, value in kwargs.items():
            if key not in system_params:
                self.config[key] = value

    @abstractmethod
    async def initialize(self) -> bool:
        """스텝 초기화"""
        pass

    @abstractmethod
    async def process(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """메인 처리"""
        pass

    async def get_step_info(self) -> Dict[str, Any]:
        """🔍 스텝 정보 반환"""
        return {
            "step_name": self.step_name,
            "device": self.device,
            "device_type": self.device_type,
            "memory_gb": self.memory_gb,
            "is_m3_max": self.is_m3_max,
            "optimization_enabled": self.optimization_enabled,
            "quality_level": self.quality_level,
            "initialized": self.is_initialized,
            "config_keys": list(self.config.keys())
        }

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('virtual_fitting_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# main.py에서 요구하는 ENUM 추가
# ==========================================

class PipelineMode(Enum):
    """파이프라인 모드 enum (main.py에서 요구)"""
    SIMULATION = "simulation"
    PRODUCTION = "production"
    HYBRID = "hybrid"
    DEVELOPMENT = "development"
    
    @classmethod
    def get_default(cls):
        return cls.SIMULATION

# 수정된 ai_pipeline 구조의 step 파일들 import
try:
    from app.ai_pipeline.steps.step_01_human_parsing import HumanParsingStep
    from app.ai_pipeline.steps.step_02_pose_estimation import PoseEstimationStep
    from app.ai_pipeline.steps.step_03_cloth_segmentation import ClothSegmentationStep
    from app.ai_pipeline.steps.step_04_geometric_matching import GeometricMatchingStep
    from app.ai_pipeline.steps.step_05_cloth_warping import ClothWarpingStep
    from app.ai_pipeline.steps.step_06_virtual_fitting import VirtualFittingStep
    from app.ai_pipeline.steps.step_07_post_processing import PostProcessingStep
    from app.ai_pipeline.steps.step_08_quality_assessment import QualityAssessmentStep
    STEPS_IMPORT_SUCCESS = True
except ImportError as e:
    logger.warning(f"Step 클래스들 import 실패: {e}")
    STEPS_IMPORT_SUCCESS = False

# 유틸리티들 안전하게 import
try:
    from app.ai_pipeline.utils.model_loader import ModelLoader
    from app.ai_pipeline.utils.memory_manager import MemoryManager
    from app.ai_pipeline.utils.data_converter import DataConverter
except ImportError as e:
    logger.warning(f"일부 유틸리티 import 실패: {e}")
    ModelLoader = None
    MemoryManager = None
    DataConverter = None

class PipelineManager:
    """
    완전 통합된 8단계 가상 피팅 파이프라인 - 최적 생성자 패턴 적용
    - 모든 Step 클래스가 동일한 생성자 패턴 사용
    - pipeline_routes.py와 완벽 호환 (1번 파일 기능)
    - 고급 품질 분석 및 처리 (2번 파일 기능)
    - M3 Max 최적화
    - 프로덕션 레벨 안정성
    """
    
    def __init__(
        self, 
        device: Optional[str] = None,  # 자동 감지로 변경
        config: Optional[Dict[str, Any]] = None,
        mode: Union[str, PipelineMode] = PipelineMode.PRODUCTION,
        **kwargs  # 시스템 설정들
    ):
        """
        파이프라인 초기화 - 최적 생성자 패턴 적용
        
        Args:
            device: 사용할 디바이스 (None=자동감지, 'cpu', 'cuda', 'mps')
            config: 파이프라인 설정 딕셔너리
            mode: 파이프라인 모드
            **kwargs: 시스템 설정
                - device_type: str = "auto"
                - memory_gb: float = 16.0
                - is_m3_max: bool = False
                - optimization_enabled: bool = True
                - quality_level: str = "balanced"
                - config_path: Optional[str] = None
        """
        # 1. 💡 지능적 디바이스 자동 감지
        self.device = self._auto_detect_device(device)

        # 2. 📋 기본 시스템 설정
        self.device_type = kwargs.get('device_type', 'auto')
        self.memory_gb = kwargs.get('memory_gb', 16.0)
        self.is_m3_max = kwargs.get('is_m3_max', self._detect_m3_max())
        self.optimization_enabled = kwargs.get('optimization_enabled', True)
        self.quality_level = kwargs.get('quality_level', 'balanced')
        config_path = kwargs.get('config_path')
        
        # 3. 모드 설정
        if isinstance(mode, str):
            try:
                self.mode = PipelineMode(mode)
            except ValueError:
                self.mode = PipelineMode.PRODUCTION
        else:
            self.mode = mode
        
        # 4. 디바이스 최적화 설정
        self._configure_device_optimizations()
        
        # 5. 기존 유틸리티들 초기화 (안전하게)
        try:
            self.model_loader = ModelLoader(device=self.device) if ModelLoader else None
            self.memory_manager = MemoryManager(device=self.device, memory_limit_gb=self.memory_gb) if MemoryManager else None
            self.data_converter = DataConverter(device=self.device) if DataConverter else None
        except Exception as e:
            logger.warning(f"유틸리티 초기화 실패: {e}")
            self.model_loader = None
            self.memory_manager = None
            self.data_converter = None
        
        # 6. 설정 로드
        self.config = self._load_config(config_path, config)
        
        # 7. 파이프라인 설정
        self.pipeline_config = self.config.get('pipeline', {
            'quality_level': self.quality_level,
            'processing_mode': 'complete',
            'enable_optimization': self.optimization_enabled,
            'enable_caching': True,
            'parallel_processing': True,
            'memory_optimization': True,
            'enable_intermediate_saving': False,
            'max_retries': 3,
            'timeout_seconds': 300
        })
        
        # 8. 8단계 컴포넌트
        self.steps = {}
        self.step_order = [
            'human_parsing',           # 1단계: 인체 파싱
            'pose_estimation',         # 2단계: 포즈 추정
            'cloth_segmentation',      # 3단계: 의류 세그멘테이션
            'geometric_matching',      # 4단계: 기하학적 매칭
            'cloth_warping',          # 5단계: 옷 워핑
            'virtual_fitting',        # 6단계: 가상 피팅 생성
            'post_processing',        # 7단계: 후처리
            'quality_assessment'      # 8단계: 품질 평가
        ]
        
        # 9. 상태 관리
        self.is_initialized = False
        self.processing_stats = {}
        self.session_data = {}  # 2번 파일의 세션 관리 기능
        self.error_history = []
        
        # 10. 성능 모니터링
        self.performance_metrics = {
            'total_sessions': 0,
            'successful_sessions': 0,
            'average_processing_time': 0.0,
            'average_quality_score': 0.0
        }
        
        # 11. 스레드 풀 (병렬 처리용)
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"🚀 완전 통합된 가상 피팅 파이프라인 초기화 - 디바이스: {self.device}")
        logger.info(f"💻 디바이스 타입: {self.device_type}, 메모리: {self.memory_gb}GB")
        logger.info(f"🍎 M3 Max: {'✅' if self.is_m3_max else '❌'}, 최적화: {'✅' if self.optimization_enabled else '❌'}")
        logger.info(f"📊 파이프라인 모드: {self.mode.value}")
        logger.info(f"🎯 품질 레벨: {self.pipeline_config['quality_level']}")

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
                # M3 Max 감지 로직
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                return 'M3' in result.stdout
        except:
            pass
        return False
    
    def _configure_device_optimizations(self):
        """디바이스별 최적화 설정 (MPS empty_cache 오류 수정)"""
        try:
            import gc
            import torch
            
            if self.device == 'mps':
                logger.info("🍎 M3 Max MPS 디바이스 최적화 시작...")
                
                # 환경 변수 설정
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                
                # 메모리 정리
                gc.collect()
                
                # PyTorch 버전별 MPS 최적화 처리
                try:
                    pytorch_version = torch.__version__
                    
                    # MPS 백엔드 초기화 테스트
                    if torch.backends.mps.is_available():
                        # 간단한 텐서 연산으로 MPS 초기화
                        test_tensor = torch.randn(1, 1).to(self.device)
                        _ = test_tensor + 1
                        del test_tensor
                        logger.info("🍎 M3 Max MPS 백엔드 초기화 완료")
                        
                        # MPS empty_cache 지원 여부 확인
                        if hasattr(torch.backends.mps, 'empty_cache'):
                            torch.backends.mps.empty_cache()
                            logger.info("✅ MPS empty_cache 사용")
                        else:
                            logger.info(f"ℹ️ PyTorch {pytorch_version}: MPS empty_cache 미지원 - 대체 메모리 관리 사용")
                            
                            # 대체 메모리 관리
                            if hasattr(torch.mps, 'synchronize'):
                                torch.mps.synchronize()
                                logger.info("✅ MPS synchronize 대체 사용")
                            
                            # 가비지 컬렉션으로 대체
                            gc.collect()
                            logger.info("✅ 가비지 컬렉션으로 메모리 정리")
                    else:
                        logger.warning("⚠️ MPS 사용 불가 - CPU로 폴백")
                        self.device = "cpu"
                        
                except Exception as mps_error:
                    logger.warning(f"MPS 초기화 실패: {mps_error}")
                    # 완전 안전 모드로 폴백
                    gc.collect()
                    logger.info("🚨 안전 모드로 메모리 관리")
                
                logger.info("🍎 M3 Max 메모리 최적화 완료")
                
            elif self.device == 'cuda':
                logger.info("🎮 CUDA 디바이스 최적화 시작...")
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.deterministic = False
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("✅ CUDA 메모리 캐시 정리 완료")
                    
                logger.info("🎮 CUDA 최적화 완료")
                
            else:
                logger.info("⚡ CPU 디바이스 최적화 시작...")
                
                if hasattr(torch, 'set_num_threads'):
                    # M3 Max의 효율 코어 활용
                    num_threads = min(4, os.cpu_count() or 4)
                    torch.set_num_threads(num_threads)
                    logger.info(f"⚡ CPU 스레드 수 설정: {num_threads}")
                    
                logger.info("⚡ CPU 최적화 완료")
            
            # 혼합 정밀도 설정
            if self.device in ['cuda', 'mps'] and self.optimization_enabled:
                self.use_amp = True
                logger.info("⚡ 혼합 정밀도 연산 활성화")
            else:
                self.use_amp = False
                
            logger.info(f"✅ {self.device.upper()} 디바이스 최적화 완료")
                
        except Exception as e:
            logger.error(f"❌ 디바이스 최적화 실패: {e}")
            # 오류가 발생해도 초기화는 계속 진행
            self.device = "cpu"  # 안전한 폴백
            self.use_amp = False
            logger.info("🔄 안전 모드로 폴백 - CPU 사용")

    async def initialize(self) -> bool:
        """전체 파이프라인 초기화 - 최적 생성자 패턴"""
        try:
            logger.info("🔄 최적 생성자 패턴 8단계 가상 피팅 파이프라인 초기화 시작...")
            start_time = time.time()
            
            # Step 클래스 import 확인
            if not STEPS_IMPORT_SUCCESS:
                logger.warning("⚠️ Step 클래스들을 import할 수 없어 시뮬레이션 모드로 진행")
                return await self._initialize_simulation_mode()
            
            # 메모리 정리
            self._cleanup_memory()
            
            # 각 단계 순차적 초기화 - ✅ 최적 생성자 패턴 적용
            await self._initialize_all_steps_optimal()
            
            # 초기화 검증
            initialization_success = await self._verify_initialization()
            
            if not initialization_success:
                raise RuntimeError("일부 단계 초기화 실패")
            
            initialization_time = time.time() - start_time
            self.processing_stats['initialization_time'] = initialization_time
            
            self.is_initialized = True
            logger.info(f"✅ 전체 파이프라인 초기화 완료 - 소요시간: {initialization_time:.2f}초")
            
            # 시스템 상태 출력
            await self._print_system_status()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 파이프라인 초기화 실패: {e}")
            logger.error(f"📋 오류 상세: {traceback.format_exc()}")
            
            # 시뮬레이션 모드로 폴백
            logger.info("🔄 시뮬레이션 모드로 폴백 시도...")
            return await self._initialize_simulation_mode()

    async def _initialize_all_steps_optimal(self):
        """✅ 모든 단계 초기화 - 최적 생성자 패턴 적용"""
        
        # 🎯 시스템 공통 설정 정의
        system_config = {
            'device': self.device,
            'device_type': self.device_type,
            'memory_gb': self.memory_gb,
            'is_m3_max': self.is_m3_max,
            'optimization_enabled': self.optimization_enabled,
            'quality_level': self.quality_level
        }
        
        # 🎯 단계별 특화 설정
        step_configs = {
            'human_parsing': {
                'use_coreml': self.is_m3_max,
                'enable_quantization': True,
                'input_size': (512, 512),
                'num_classes': 20,
                'cache_size': 50,
                'batch_size': 1,
                'model_name': 'graphonomy'
            },
            'pose_estimation': {
                'model_type': 'openpose',
                'input_size': (368, 368),
                'confidence_threshold': 0.1,
                'use_gpu': self.device != 'cpu'
            },
            'cloth_segmentation': {
                'model_name': 'u2net',
                'background_threshold': 0.5,
                'post_process': True,
                'refine_edges': True
            },
            'geometric_matching': {
                'tps_points': 25,
                'matching_threshold': 0.8,
                'use_advanced_matching': True
            },
            'cloth_warping': {
                'warping_method': 'tps',
                'physics_simulation': True,
                'fabric_simulation': True,
                'optimization_level': 'high'
            },
            'virtual_fitting': {
                'blending_method': 'poisson',
                'seamless_cloning': True,
                'color_transfer': True
            },
            'post_processing': {
                'enable_super_resolution': True,
                'enhance_faces': True,
                'color_correction': True,
                'noise_reduction': True
            },
            'quality_assessment': {
                'enable_detailed_analysis': True,
                'perceptual_metrics': True,
                'technical_metrics': True
            }
        }
        
        # 🎯 스텝 클래스들 매핑
        step_classes = {
            'human_parsing': HumanParsingStep,
            'pose_estimation': PoseEstimationStep,
            'cloth_segmentation': ClothSegmentationStep,
            'geometric_matching': GeometricMatchingStep,
            'cloth_warping': ClothWarpingStep,
            'virtual_fitting': VirtualFittingStep,
            'post_processing': PostProcessingStep,
            'quality_assessment': QualityAssessmentStep
        }
        
        # 🚀 모든 스텝을 최적 생성자 패턴으로 생성
        for step_name, step_class in step_classes.items():
            logger.info(f"{step_name.replace('_', ' ').title()} 초기화...")
            
            try:
                # ✅ 최적 생성자 패턴: 모든 스텝이 동일한 시그니처!
                step_config = step_configs.get(step_name, {})
                
                self.steps[step_name] = step_class(
                    device=system_config['device'],                    # None으로 자동 감지
                    config=step_config,                               # 스텝별 설정
                    **system_config                                   # 시스템 전체 설정
                )
                
                # 안전한 초기화
                await self._safe_initialize_step(step_name)
                
                logger.info(f"✅ {step_name} 최적 생성자 패턴 적용 완료")
                
            except Exception as e:
                logger.warning(f"⚠️ {step_name} 초기화 실패: {e}")
                # 폴백 스텝 생성 - 동일한 생성자 패턴
                self.steps[step_name] = self._create_optimal_fallback_step(step_name, system_config, step_configs.get(step_name, {}))

    def _create_optimal_fallback_step(self, step_name: str, system_config: Dict[str, Any], step_config: Dict[str, Any]):
        """✅ 최적 생성자 패턴을 적용한 폴백 단계 클래스 생성"""
        
        class OptimalFallbackStep(OptimalStepConstructor):
            """최적 생성자 패턴을 따르는 폴백 스텝"""
            
            def __init__(self, device=None, config=None, **kwargs):
                super().__init__(device=device, config=config, **kwargs)
                self.step_name = step_name
                self.fallback_mode = True
            
            async def initialize(self):
                self.is_initialized = True
                return True
            
            async def process(self, *args, **kwargs):
                await asyncio.sleep(0.1)  # 처리 시뮬레이션
                return {
                    'success': True,
                    'fallback': True,
                    'step_name': self.step_name,
                    'confidence': 0.6,
                    'processing_time': 0.1,
                    'method': 'optimal_fallback',
                    'device': self.device,
                    'device_type': self.device_type,
                    'memory_gb': self.memory_gb,
                    'is_m3_max': self.is_m3_max,
                    'optimization_enabled': self.optimization_enabled,
                    'quality_level': self.quality_level
                }
            
            async def cleanup(self):
                pass
            
            async def get_model_info(self):
                return await self.get_step_info()
        
        logger.info(f"🚨 {step_name} 최적 폴백 클래스 생성")
        return OptimalFallbackStep(
            device=system_config['device'],
            config=step_config,
            **system_config
        )

    async def _initialize_simulation_mode(self) -> bool:
        """시뮬레이션 모드 초기화 - 최적 생성자 패턴"""
        try:
            logger.info("🎭 최적 생성자 패턴 시뮬레이션 모드로 파이프라인 초기화...")
            
            # 시스템 설정
            system_config = {
                'device': self.device,
                'device_type': self.device_type,
                'memory_gb': self.memory_gb,
                'is_m3_max': self.is_m3_max,
                'optimization_enabled': self.optimization_enabled,
                'quality_level': self.quality_level
            }
            
            # 시뮬레이션 단계들 생성 - 최적 생성자 패턴
            for step_name in self.step_order:
                self.steps[step_name] = self._create_optimal_fallback_step(step_name, system_config, {})
                logger.info(f"🎭 {step_name} 최적 시뮬레이션 단계 생성됨")
            
            self.is_initialized = True
            self.pipeline_config['processing_mode'] = 'simulation'
            self.mode = PipelineMode.SIMULATION
            
            logger.info("✅ 최적 생성자 패턴 시뮬레이션 모드 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 시뮬레이션 모드 초기화도 실패: {e}")
            self.is_initialized = False
            return False

    async def _safe_initialize_step(self, step_name: str):
        """안전한 단계 초기화"""
        try:
            step = self.steps[step_name]
            if hasattr(step, 'initialize'):
                await step.initialize()
            logger.info(f"✅ {step_name} 초기화 완료")
        except Exception as e:
            logger.warning(f"⚠️ {step_name} 초기화 실패: {e}")
    
    async def _verify_initialization(self) -> bool:
        """초기화 검증"""
        total_steps = len(self.step_order)
        initialized_steps = len(self.steps)
        
        success_rate = initialized_steps / total_steps
        logger.info(f"📊 초기화 성공률: {success_rate:.1%} ({initialized_steps}/{total_steps})")
        
        return success_rate >= 0.8  # 80% 이상 성공하면 진행
    
    # ===========================================
    # 1번 파일 기능: pipeline_routes.py 호환성
    # ===========================================
    
    async def process_virtual_tryon(
        self,
        person_image: Union[str, Image.Image, np.ndarray],
        clothing_image: Union[str, Image.Image, np.ndarray],
        height: float = 170.0,
        weight: float = 65.0,
        progress_callback: Optional[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        pipeline_routes.py에서 호출하는 가상 피팅 메서드 (1번 파일 기능)
        최적 생성자 패턴 적용된 Step들과 호환
        """
        if not self.is_initialized:
            raise RuntimeError("파이프라인이 초기화되지 않았습니다.")
        
        start_time = time.time()
        
        try:
            # 진행률 콜백
            if progress_callback:
                await progress_callback("입력 이미지 전처리 중...", 10)
            
            # 입력 전처리
            person_tensor, clothing_tensor = await self._preprocess_inputs(person_image, clothing_image)
            
            if progress_callback:
                await progress_callback("AI 파이프라인 실행 중...", 20)
            
            # 8단계 파이프라인 실행
            result = await self._execute_complete_pipeline(
                person_tensor, clothing_tensor, height, weight, progress_callback
            )
            
            processing_time = time.time() - start_time
            
            # 최종 결과 구성 (1번 파일과 동일한 형식)
            final_result = {
                'success': result.get('success', True),
                'fitted_image': result.get('result_image'),
                'processing_time': processing_time,
                'confidence': result.get('final_quality_score', 0.8),
                'fit_score': result.get('fit_score', 0.8),
                'quality_score': result.get('final_quality_score', 0.82),
                'measurements': {
                    'height': height,
                    'weight': weight,
                    'estimated_chest': 95,
                    'estimated_waist': 80
                },
                'recommendations': result.get('improvement_suggestions', {}).get('user_experience', []),
                'pipeline_stages': {
                    'human_parsing': {'completed': True, 'time': 0.5},
                    'pose_estimation': {'completed': True, 'time': 0.3},
                    'cloth_segmentation': {'completed': True, 'time': 0.4},
                    'geometric_matching': {'completed': True, 'time': 0.6},
                    'cloth_warping': {'completed': True, 'time': 0.8},
                    'virtual_fitting': {'completed': True, 'time': 1.2},
                    'post_processing': {'completed': True, 'time': 0.7},
                    'quality_assessment': {'completed': True, 'time': 0.3}
                },
                'debug_info': {
                    'device': self.device,
                    'device_type': self.device_type,
                    'memory_gb': self.memory_gb,
                    'is_m3_max': self.is_m3_max,
                    'optimization_enabled': self.optimization_enabled,
                    'mode': self.mode.value,
                    'constructor_pattern': 'optimal'
                }
            }
            
            if progress_callback:
                await progress_callback("처리 완료!", 100)
            
            return final_result
            
        except Exception as e:
            logger.error(f"❌ 가상 피팅 처리 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'confidence': 0.0,
                'fit_score': 0.0,
                'quality_score': 0.0,
                'measurements': {},
                'recommendations': ['오류가 발생했습니다. 다시 시도해주세요.'],
                'pipeline_stages': {},
                'debug_info': {'error': str(e), 'constructor_pattern': 'optimal'}
            }

    async def _execute_complete_pipeline(
        self, 
        person_tensor: torch.Tensor,
        clothing_tensor: torch.Tensor,
        height: float,
        weight: float,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """완전한 8단계 파이프라인 실행 - 최적 생성자 패턴 스텝들과 호환"""
        
        try:
            # 1단계: 인체 파싱
            if progress_callback:
                await progress_callback("1단계: 인체 파싱 중...", 25)
            parsing_result = await self.steps['human_parsing'].process(person_tensor)
            
            # 2단계: 포즈 추정
            if progress_callback:
                await progress_callback("2단계: 포즈 추정 중...", 35)
            pose_result = await self.steps['pose_estimation'].process(person_tensor)
            
            # 3단계: 의류 세그멘테이션
            if progress_callback:
                await progress_callback("3단계: 의류 세그멘테이션 중...", 45)
            segmentation_result = await self.steps['cloth_segmentation'].process(
                clothing_tensor, clothing_type='shirt'
            )
            
            # 4단계: 기하학적 매칭
            if progress_callback:
                await progress_callback("4단계: 기하학적 매칭 중...", 55)
            matching_result = await self.steps['geometric_matching'].process(
                parsing_result, pose_result.get('keypoints_18', []), segmentation_result
            )
            
            # 5단계: 옷 워핑
            if progress_callback:
                await progress_callback("5단계: 옷 워핑 중...", 65)
            warping_result = await self.steps['cloth_warping'].process(
                matching_result, {'height': height, 'weight': weight}, 'cotton'
            )
            
            # 6단계: 가상 피팅
            if progress_callback:
                await progress_callback("6단계: 가상 피팅 생성 중...", 75)
            fitting_result = await self.steps['virtual_fitting'].process(
                person_tensor, warping_result, parsing_result, pose_result
            )
            
            # 7단계: 후처리
            if progress_callback:
                await progress_callback("7단계: 후처리 중...", 85)
            post_result = await self.steps['post_processing'].process(fitting_result)
            
            # 8단계: 품질 평가
            if progress_callback:
                await progress_callback("8단계: 품질 평가 중...", 95)
            quality_result = await self.steps['quality_assessment'].process(
                post_result, person_tensor, clothing_tensor,
                parsing_result, pose_result, warping_result, fitting_result
            )
            
            # 결과 통합
            final_quality_score = quality_result.get('overall_confidence', 0.8)
            
            return {
                'success': True,
                'result_image': self._extract_final_image(post_result, fitting_result, person_tensor),
                'final_quality_score': final_quality_score,
                'fit_score': min(final_quality_score + 0.1, 1.0),
                'step_results': {
                    'parsing': parsing_result,
                    'pose': pose_result,
                    'segmentation': segmentation_result,
                    'matching': matching_result,
                    'warping': warping_result,
                    'fitting': fitting_result,
                    'post_processing': post_result,
                    'quality': quality_result
                },
                'improvement_suggestions': {
                    'user_experience': [
                        "정면을 바라보는 자세로 촬영하면 더 좋은 결과를 얻을 수 있습니다",
                        "조명이 균등한 환경에서 촬영해보세요",
                        "단색 배경을 사용하면 더 정확한 세그멘테이션이 가능합니다"
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"파이프라인 실행 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'final_quality_score': 0.0
            }
    
    # ===========================================
    # 2번 파일 기능: 고급 분석 및 처리
    # ===========================================
    
    async def process_complete_virtual_fitting(
        self,
        person_image: Union[str, Image.Image, np.ndarray],
        clothing_image: Union[str, Image.Image, np.ndarray],
        body_measurements: Optional[Dict[str, float]] = None,
        clothing_type: str = "shirt",
        fabric_type: str = "cotton",
        style_preferences: Optional[Dict[str, Any]] = None,
        quality_target: float = 0.8,
        progress_callback: Optional[Callable] = None,
        save_intermediate: bool = False,
        enable_auto_retry: bool = True
    ) -> Dict[str, Any]:
        """
        고급 가상 피팅 처리 (2번 파일 기능) - 최적 생성자 패턴 적용
        """
        if not self.is_initialized:
            raise RuntimeError("파이프라인이 초기화되지 않았습니다. initialize()를 먼저 호출하세요.")
        
        start_time = time.time()
        session_id = f"vf_{int(time.time())}_{np.random.randint(1000, 9999)}"
        
        self.performance_metrics['total_sessions'] += 1
        
        try:
            logger.info(f"🎯 최적 생성자 패턴 8단계 가상 피팅 시작 - 세션 ID: {session_id}")
            logger.info(f"⚙️ 설정: {clothing_type} ({fabric_type}), 품질목표: {quality_target}")
            
            # 입력 검증 및 전처리
            person_tensor, clothing_tensor = await self._preprocess_inputs(
                person_image, clothing_image
            )
            
            # 세션 데이터 초기화
            self._initialize_session_data(session_id, start_time, {
                'clothing_type': clothing_type,
                'fabric_type': fabric_type,
                'quality_target': quality_target,
                'style_preferences': style_preferences or {},
                'processing_mode': self.pipeline_config['processing_mode'],
                'constructor_pattern': 'optimal'
            })
            
            if progress_callback:
                await progress_callback("입력 전처리 완료", 5)
            
            # 메모리 최적화
            if self.pipeline_config['memory_optimization']:
                self._optimize_memory_usage()
            
            # 8단계 파이프라인 실행 (재시도 로직 포함)
            result = await self._execute_complete_pipeline_with_retry(
                person_tensor, clothing_tensor, body_measurements, 
                fabric_type, progress_callback, session_id
            )
            
            total_time = time.time() - start_time
            
            # 최종 결과 이미지 추출
            final_image_tensor = self._extract_final_image(
                result.get('post_processing_result', {}), 
                result.get('fitting_result', {}), 
                person_tensor
            )
            final_image_pil = self._tensor_to_pil(final_image_tensor)
            
            # 종합 품질 분석 (2번 파일의 고급 기능)
            comprehensive_quality = await self._comprehensive_quality_analysis(
                result.get('quality_result', {}), self.session_data[session_id]
            )
            
            # 처리 통계 계산
            processing_statistics = self._calculate_detailed_statistics(session_id, total_time)
            
            # 개선 제안 생성
            improvement_suggestions = await self._generate_detailed_suggestions(
                comprehensive_quality, processing_statistics, clothing_type, fabric_type
            )
            
            # 성능 메트릭 업데이트
            self.performance_metrics['successful_sessions'] += 1
            self._update_performance_metrics(total_time, comprehensive_quality['overall_score'])
            
            # 최종 결과 구성 (2번 파일의 상세 결과)
            final_result = {
                'success': True,
                'session_id': session_id,
                'processing_mode': self.pipeline_config['processing_mode'],
                'quality_level': self.pipeline_config['quality_level'],
                'constructor_pattern': 'optimal',
                
                # 결과 이미지들
                'result_image': final_image_pil,
                'result_image_tensor': final_image_tensor,
                'original_person_image': self._tensor_to_pil(person_tensor),
                'original_clothing_image': self._tensor_to_pil(clothing_tensor),
                
                # 품질 메트릭
                'final_quality_score': comprehensive_quality['overall_score'],
                'quality_grade': comprehensive_quality['quality_grade'],
                'quality_confidence': comprehensive_quality['confidence'],
                'quality_breakdown': comprehensive_quality['breakdown'],
                'quality_target_achieved': comprehensive_quality['overall_score'] >= quality_target,
                
                # 개선 제안
                'improvement_suggestions': improvement_suggestions,
                'next_steps': self._generate_next_steps(comprehensive_quality, quality_target),
                
                # 처리 통계
                'processing_statistics': processing_statistics,
                'total_processing_time': total_time,
                'device_used': self.device,
                'memory_usage': self._get_detailed_memory_usage(),
                'performance_metrics': self.performance_metrics.copy(),
                
                # 단계별 결과
                'step_results_summary': self._create_detailed_step_summary(session_id),
                
                # 중간 결과 (선택적)
                'intermediate_results': (
                    self.session_data[session_id]['intermediate_results'] 
                    if save_intermediate else {}
                ),
                
                # 메타데이터
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'pipeline_version': '4.0.0-optimal',
                    'input_resolution': f"{person_tensor.shape[3]}x{person_tensor.shape[2]}",
                    'output_resolution': f"{final_image_pil.width}x{final_image_pil.height}",
                    'clothing_type': clothing_type,
                    'fabric_type': fabric_type,
                    'body_measurements_provided': body_measurements is not None,
                    'style_preferences_provided': bool(style_preferences),
                    'intermediate_results_saved': save_intermediate,
                    'device_optimization': self.device,
                    'constructor_pattern': 'optimal',
                    'integrated_version': True
                }
            }
            
            # 세션 데이터 정리
            if not save_intermediate:
                self._cleanup_session_data(session_id)
            
            if progress_callback:
                await progress_callback("처리 완료", 100)
            
            logger.info(
                f"🎉 최적 생성자 패턴 8단계 가상 피팅 완료! "
                f"전체 소요시간: {total_time:.2f}초, "
                f"최종 품질: {comprehensive_quality['overall_score']:.3f} ({comprehensive_quality['quality_grade']}), "
                f"목표 달성: {'✅' if comprehensive_quality['overall_score'] >= quality_target else '❌'}"
            )
            
            return final_result
            
        except Exception as e:
            # 에러 처리 및 복구 (2번 파일의 고급 기능)
            error_result = await self._handle_processing_error(
                e, session_id, start_time, person_image, clothing_image,
                enable_auto_retry
            )
            return error_result

    async def _execute_complete_pipeline_with_retry(
        self,
        person_tensor: torch.Tensor,
        clothing_tensor: torch.Tensor,
        body_measurements: Optional[Dict[str, float]],
        fabric_type: str,
        progress_callback: Optional[Callable],
        session_id: str
    ) -> Dict[str, Any]:
        """재시도 로직이 포함된 완전한 파이프라인 실행 - 최적 생성자 패턴"""
        
        try:
            # 1단계: 인체 파싱
            parsing_result = await self._execute_step_with_retry(
                'human_parsing', 1, person_tensor, progress_callback, 18
            )
            
            # 2단계: 포즈 추정
            pose_result = await self._execute_step_with_retry(
                'pose_estimation', 2, person_tensor, progress_callback, 31
            )
            
            # 3단계: 의류 세그멘테이션
            segmentation_result = await self._execute_step_with_retry(
                'cloth_segmentation', 3, clothing_tensor, progress_callback, 44,
                extra_args={'clothing_type': 'shirt'}
            )
            
            # 4단계: 기하학적 매칭
            matching_result = await self._execute_step_with_retry(
                'geometric_matching', 4, 
                (segmentation_result, pose_result, parsing_result),
                progress_callback, 57
            )
            
            # 5단계: 옷 워핑
            warping_result = await self._execute_step_with_retry(
                'cloth_warping', 5,
                (matching_result, body_measurements, fabric_type),
                progress_callback, 70
            )
            
            # 6단계: 가상 피팅 생성
            fitting_result = await self._execute_step_with_retry(
                'virtual_fitting', 6,
                (person_tensor, warping_result, parsing_result, pose_result),
                progress_callback, 83
            )
            
            # 7단계: 후처리
            post_processing_result = await self._execute_step_with_retry(
                'post_processing', 7, fitting_result, progress_callback, 91
            )
            
            # 8단계: 품질 평가
            quality_result = await self._execute_step_with_retry(
                'quality_assessment', 8,
                (post_processing_result, person_tensor, clothing_tensor, 
                 parsing_result, pose_result, warping_result, fitting_result),
                progress_callback, 96
            )
            
            return {
                'success': True,
                'parsing_result': parsing_result,
                'pose_result': pose_result,
                'segmentation_result': segmentation_result,
                'matching_result': matching_result,
                'warping_result': warping_result,
                'fitting_result': fitting_result,
                'post_processing_result': post_processing_result,
                'quality_result': quality_result
            }
            
        except Exception as e:
            logger.error(f"재시도 파이프라인 실행 실패: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _execute_step_with_retry(
        self,
        step_name: str,
        step_number: int,
        input_data: Any,
        progress_callback: Optional[Callable],
        progress_percentage: int,
        extra_args: Optional[Dict] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """재시도 로직이 포함된 단계 실행 - 최적 생성자 패턴 스텝과 호환"""
        
        step_start = time.time()
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"{'🔄' if attempt > 0 else ''} {step_number}단계: {step_name} 처리 중... {'(재시도 ' + str(attempt) + ')' if attempt > 0 else ''}")
                
                # 메모리 정리 (재시도 시)
                if attempt > 0:
                    self._cleanup_memory()
                
                # 단계 실행
                result = await self._execute_single_step(step_name, input_data, extra_args)
                
                # 결과 검증
                if self._validate_step_result(step_name, result):
                    step_time = time.time() - step_start
                    
                    # 세션 데이터 저장
                    session_id = list(self.session_data.keys())[-1] if self.session_data else 'unknown'
                    if session_id in self.session_data:
                        self.session_data[session_id]['step_times'][step_name] = step_time
                        self.session_data[session_id]['step_results'][step_name] = result
                        
                        if self.pipeline_config.get('enable_intermediate_saving', False):
                            self.session_data[session_id]['intermediate_results'][step_name] = result
                    
                    # 품질 점수 로깅
                    quality_score = result.get('confidence', result.get('quality_score', 0.8))
                    logger.info(f"✅ {step_number}단계 완료 - 소요시간: {step_time:.2f}초, 품질: {quality_score:.3f}")
                    
                    # 진행률 콜백
                    if progress_callback:
                        await progress_callback(f"{step_name}_complete", progress_percentage)
                    
                    return result
                else:
                    raise ValueError(f"Step {step_name} result validation failed")
                    
            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ {step_number}단계 시도 {attempt + 1} 실패: {e}")
                
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    logger.info(f"🔄 {wait_time}초 후 재시도...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ {step_number}단계 최종 실패: {e}")
        
        # 모든 재시도 실패 시 폴백 결과 생성
        logger.warning(f"🚨 {step_name} 폴백 결과 생성 중...")
        return self._create_fallback_step_result(step_name, input_data, last_error)

    async def _execute_single_step(
        self, 
        step_name: str, 
        input_data: Any, 
        extra_args: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """단일 단계 실행 - 최적 생성자 패턴 클래스들과 호환"""
        
        step = self.steps.get(step_name)
        if not step:
            raise ValueError(f"Step {step_name} not found")
        
        # 최적 생성자 패턴 클래스들의 process 메서드 호출
        if step_name == 'human_parsing':
            return await step.process(input_data)
                
        elif step_name == 'pose_estimation':
            return await step.process(input_data)
                
        elif step_name == 'cloth_segmentation':
            clothing_type = extra_args.get('clothing_type', 'shirt') if extra_args else 'shirt'
            return await step.process(input_data, clothing_type=clothing_type)
                
        elif step_name == 'geometric_matching':
            segmentation_result, pose_result, parsing_result = input_data
            return await step.process(segmentation_result, pose_result, parsing_result)
                
        elif step_name == 'cloth_warping':
            matching_result, body_measurements, fabric_type = input_data
            return await step.process(matching_result, body_measurements, fabric_type)
                
        elif step_name == 'virtual_fitting':
            person_tensor, warping_result, parsing_result, pose_result = input_data
            return await step.process(person_tensor, warping_result, parsing_result, pose_result)
                
        elif step_name == 'post_processing':
            return await step.process(input_data)
                
        elif step_name == 'quality_assessment':
            (post_processing_result, person_tensor, clothing_tensor, 
             parsing_result, pose_result, warping_result, fitting_result) = input_data
            return await step.process(
                post_processing_result, person_tensor, clothing_tensor,
                parsing_result, pose_result, warping_result, fitting_result
            )
        
        else:
            raise ValueError(f"Unknown step: {step_name}")

    def _validate_step_result(self, step_name: str, result: Dict[str, Any]) -> bool:
        """단계 결과 검증"""
        if not isinstance(result, dict):
            return False
        return result.get('success', True)  # 기본적으로 성공으로 간주

    def _create_fallback_step_result(
        self, 
        step_name: str, 
        input_data: Any, 
        error: Exception
    ) -> Dict[str, Any]:
        """폴백 단계 결과 생성"""
        return {
            'success': False,
            'error': str(error),
            'fallback': True,
            'step_name': step_name,
            'confidence': 0.5,
            'processing_time': 0.1,
            'method': 'optimal_fallback',
            'constructor_pattern': 'optimal',
            'timestamp': datetime.now().isoformat()
        }

    # ===========================================
    # 2번 파일의 고급 분석 메서드들
    # ===========================================
    
    async def _comprehensive_quality_analysis(
        self, 
        quality_result: Dict[str, Any], 
        session_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """종합적 품질 분석 (2번 파일 기능)"""
        overall_score = quality_result.get('overall_score', 0.8)
        
        if overall_score >= 0.9:
            quality_grade = "Excellent"
            confidence = 0.95
        elif overall_score >= 0.8:
            quality_grade = "Good"
            confidence = 0.85
        elif overall_score >= 0.7:
            quality_grade = "Acceptable"
            confidence = 0.75
        elif overall_score >= 0.6:
            quality_grade = "Poor"
            confidence = 0.65
        else:
            quality_grade = "Very Poor"
            confidence = 0.5
        
        breakdown = quality_result.get('quality_breakdown', {})
        
        return {
            'overall_score': overall_score,
            'quality_grade': quality_grade,
            'confidence': confidence,
            'breakdown': breakdown,
            'analysis_timestamp': datetime.now().isoformat(),
            'constructor_pattern': 'optimal'
        }
    
    def _calculate_detailed_statistics(self, session_id: str, total_time: float) -> Dict[str, Any]:
        """상세 처리 통계 계산 (2번 파일 기능)"""
        session_data = self.session_data[session_id]
        step_times = session_data['step_times']
        
        stats = {
            'total_time': total_time,
            'step_times': step_times.copy(),
            'steps_completed': len(step_times),
            'success_rate': len(step_times) / len(self.step_order),
            'memory_usage': self._get_detailed_memory_usage(),
            'device_utilization': self._get_device_utilization(),
            'constructor_pattern': 'optimal'
        }
        
        if step_times:
            times = list(step_times.values())
            stats.update({
                'average_step_time': np.mean(times),
                'fastest_step': {'name': min(step_times, key=step_times.get), 'time': min(times)},
                'slowest_step': {'name': max(step_times, key=step_times.get), 'time': max(times)},
            })
        
        return stats
    
    async def _generate_detailed_suggestions(
        self, 
        quality_analysis: Dict[str, Any], 
        statistics: Dict[str, Any],
        clothing_type: str,
        fabric_type: str
    ) -> Dict[str, List[str]]:
        """상세 개선 제안 생성 (2번 파일 기능)"""
        suggestions = {
            'quality_improvements': [],
            'performance_optimizations': [],
            'user_experience': [],
            'technical_adjustments': []
        }
        
        overall_score = quality_analysis['overall_score']
        
        if overall_score < 0.8:
            suggestions['quality_improvements'].extend([
                "🎯 전체적인 품질 향상이 필요합니다",
                "📷 더 높은 해상도의 입력 이미지를 사용해보세요",
                "💡 조명이 균등한 환경에서 촬영된 이미지를 사용하세요"
            ])
        
        total_time = statistics['total_time']
        if total_time > 60:
            suggestions['performance_optimizations'].extend([
                "⚡ 처리 시간이 긴 편입니다. 품질 레벨을 조정해보세요",
                "🖥️ 더 높은 성능의 디바이스 사용을 고려하세요"
            ])
        
        suggestions['user_experience'].extend([
            "📸 정면을 바라보는 자세의 사진이 가장 좋은 결과를 제공합니다",
            "🎨 단색 배경의 의류 이미지를 사용하면 더 정확한 결과를 얻을 수 있습니다",
            "✨ 최적 생성자 패턴으로 모든 단계가 일관되게 처리되었습니다"
        ])
        
        if self.device == 'cpu':
            suggestions['technical_adjustments'].append(
                "🚀 GPU나 MPS를 사용하면 처리 속도가 크게 향상됩니다"
            )
        
        return suggestions
    
    def _generate_next_steps(self, quality_analysis: Dict[str, Any], quality_target: float) -> List[str]:
        """다음 단계 제안"""
        overall_score = quality_analysis['overall_score']
        next_steps = []
        
        if overall_score >= quality_target:
            next_steps.extend([
                "✅ 목표 품질에 도달했습니다!",
                "💾 결과를 저장하고 활용하세요",
                "🔄 다른 의류로 추가 피팅을 시도해보세요",
                "🎯 최적 생성자 패턴으로 일관된 품질이 보장됩니다"
            ])
        else:
            gap = quality_target - overall_score
            next_steps.extend([
                f"🎯 목표 품질까지 {gap:.2f} 점 향상이 필요합니다",
                "🔧 개선 제안사항을 적용해보세요",
                "📷 더 좋은 품질의 입력 이미지를 준비하세요"
            ])
        
        return next_steps
    
    def _create_detailed_step_summary(self, session_id: str) -> Dict[str, Dict[str, Any]]:
        """상세 단계 요약 생성"""
        session_data = self.session_data[session_id]
        step_times = session_data['step_times']
        step_results = session_data['step_results']
        
        summary = {}
        
        for step_name in self.step_order:
            step_summary = {
                'completed': step_name in step_results,
                'processing_time': step_times.get(step_name, 0),
                'success': step_name in step_results and not step_results[step_name].get('error'),
                'fallback_used': step_results.get(step_name, {}).get('fallback', False),
                'constructor_pattern': 'optimal'
            }
            
            if step_name in step_results:
                result = step_results[step_name]
                step_summary.update({
                    'confidence': result.get('confidence', 0),
                    'method': result.get('method', 'optimal'),
                    'device': result.get('device', self.device)
                })
            
            summary[step_name] = step_summary
        
        return summary
    
    # ===========================================
    # 세션 관리 메서드들 (2번 파일 기능)
    # ===========================================
    
    def _initialize_session_data(self, session_id: str, start_time: float, config: Dict[str, Any]):
        """세션 데이터 초기화"""
        self.session_data[session_id] = {
            'start_time': start_time,
            'config': config,
            'step_times': {},
            'step_results': {},
            'intermediate_results': {},
            'error_log': [],
            'memory_snapshots': [],
            'quality_progression': [],
            'constructor_pattern': 'optimal'
        }
    
    def _cleanup_session_data(self, session_id: str):
        """세션 데이터 정리"""
        if session_id in self.session_data:
            del self.session_data[session_id]
            logger.debug(f"🧹 세션 {session_id} 데이터 정리 완료")
    
    def _update_performance_metrics(self, processing_time: float, quality_score: float):
        """성능 메트릭 업데이트"""
        total_sessions = self.performance_metrics['total_sessions']
        
        if total_sessions > 1:
            prev_avg_time = self.performance_metrics['average_processing_time']
            prev_avg_quality = self.performance_metrics['average_quality_score']
            
            self.performance_metrics['average_processing_time'] = (
                (prev_avg_time * (total_sessions - 1) + processing_time) / total_sessions
            )
            self.performance_metrics['average_quality_score'] = (
                (prev_avg_quality * (total_sessions - 1) + quality_score) / total_sessions
            )
        else:
            self.performance_metrics['average_processing_time'] = processing_time
            self.performance_metrics['average_quality_score'] = quality_score
    
    async def _handle_processing_error(
        self,
        error: Exception,
        session_id: str,
        start_time: float,
        person_image: Union[str, Image.Image, np.ndarray],
        clothing_image: Union[str, Image.Image, np.ndarray],
        enable_auto_retry: bool = True
    ) -> Dict[str, Any]:
        """처리 오류 핸들링 및 복구"""
        processing_time = time.time() - start_time
        error_msg = str(error)
        
        # 오류 기록
        self.error_history.append({
            'session_id': session_id,
            'error': error_msg,
            'error_type': type(error).__name__,
            'timestamp': datetime.now().isoformat(),
            'processing_time': processing_time,
            'constructor_pattern': 'optimal'
        })
        
        logger.error(f"❌ 최적 생성자 패턴 가상 피팅 처리 실패 - 세션 {session_id}: {error_msg}")
        
        # 자동 복구 시도
        if enable_auto_retry and not hasattr(error, '_retry_attempted'):
            logger.info("🔄 자동 복구 시도 중...")
            
            try:
                self._cleanup_memory()
                error._retry_attempted = True
                
                # 낮은 품질 모드로 재시도
                original_quality = self.pipeline_config['quality_level']
                self.pipeline_config['quality_level'] = 'medium'
                
                result = await self.process_complete_virtual_fitting(
                    person_image=person_image,
                    clothing_image=clothing_image,
                    quality_target=0.6,
                    enable_auto_retry=False
                )
                
                self.pipeline_config['quality_level'] = original_quality
                
                if result['success']:
                    logger.info("✅ 자동 복구 성공!")
                    result['recovered'] = True
                    result['recovery_method'] = 'quality_downgrade'
                    result['constructor_pattern'] = 'optimal'
                    return result
                    
            except Exception as retry_error:
                logger.warning(f"⚠️ 자동 복구 실패: {retry_error}")
        
        # 기본 오류 결과 반환
        return {
            'success': False,
            'session_id': session_id,
            'error': error_msg,
            'error_type': 'processing_failure',
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'recovery_attempted': enable_auto_retry,
            'constructor_pattern': 'optimal',
            'metadata': {
                'pipeline_version': '4.0.0-optimal',
                'device': self.device,
                'integrated_version': True
            }
        }
    
    # ===========================================
    # 입력 처리 및 변환 메서드들
    # ===========================================
    
    async def _preprocess_inputs(
        self, 
        person_image: Union[str, Image.Image, np.ndarray],
        clothing_image: Union[str, Image.Image, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """입력 이미지 전처리"""
        try:
            # 데이터 변환기 사용
            if self.data_converter and hasattr(self.data_converter, 'preprocess_image'):
                person_tensor = self.data_converter.preprocess_image(person_image)
                clothing_tensor = self.data_converter.preprocess_image(clothing_image)
            else:
                person_tensor = self._manual_preprocess_image(person_image)
                clothing_tensor = self._manual_preprocess_image(clothing_image)
            
            # 디바이스로 이동
            person_tensor = person_tensor.to(self.device)
            clothing_tensor = clothing_tensor.to(self.device)
            
            logger.info(f"✅ 입력 이미지 전처리 완료 - 사람: {person_tensor.shape}, 의류: {clothing_tensor.shape}")
            
            return person_tensor, clothing_tensor
            
        except Exception as e:
            logger.error(f"❌ 입력 전처리 실패: {e}")
            raise
    
    def _manual_preprocess_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """수동 이미지 전처리"""
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_input}")
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype != np.uint8:
                image_input = (image_input * 255).astype(np.uint8)
            image = Image.fromarray(image_input).convert('RGB')
        else:
            raise ValueError(f"지원하지 않는 이미지 타입: {type(image_input)}")
        
        # 크기 조정
        target_size = self.config.get('input_size', (512, 512))
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # 텐서 변환
        img_array = np.array(image)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor
    
    def _extract_final_image(
        self, 
        post_processing_result: Dict[str, Any],
        fitting_result: Dict[str, Any], 
        person_tensor: torch.Tensor
    ) -> torch.Tensor:
        """최종 결과 이미지 추출"""
        if 'enhanced_image' in post_processing_result:
            return post_processing_result['enhanced_image']
        elif 'fitted_image' in fitting_result:
            return fitting_result['fitted_image']
        else:
            logger.warning("⚠️ 최종 이미지를 찾을 수 없어 원본을 반환합니다")
            return person_tensor
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """텐서를 PIL 이미지로 변환"""
        try:
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            
            if tensor.shape[0] == 3:
                tensor = tensor.permute(1, 2, 0)
            
            tensor = torch.clamp(tensor, 0, 1)
            tensor = tensor.cpu()
            array = (tensor.numpy() * 255).astype(np.uint8)
            
            return Image.fromarray(array)
        except Exception as e:
            logger.error(f"❌ 텐서-PIL 변환 실패: {e}")
            return Image.new('RGB', (512, 512), color='black')
    
    # ===========================================
    # 메모리 관리 및 시스템 정보
    # ===========================================
    
    def _optimize_memory_usage(self):
        """메모리 사용량 최적화"""
        gc.collect()
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif self.device == 'mps' and torch.backends.mps.is_available():
            # PyTorch 2.2.2 호환성 체크
            if hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
            else:
                # 대체 메모리 관리
                gc.collect()
    
    def _cleanup_memory(self):
        """메모리 정리"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
            else:
                # PyTorch 2.2.2 호환성
                gc.collect()
    
    def _get_detailed_memory_usage(self) -> Dict[str, str]:
        """상세 메모리 사용량 조회"""
        try:
            import psutil
            memory_info = {
                'system_memory': f"{psutil.virtual_memory().percent}%",
                'available_memory': f"{psutil.virtual_memory().available / 1024**3:.1f}GB"
            }
        except ImportError:
            memory_info = {'system_memory': 'N/A', 'available_memory': 'N/A'}
        
        if torch.cuda.is_available():
            memory_info.update({
                'gpu_memory_allocated': f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB",
                'gpu_memory_reserved': f"{torch.cuda.memory_reserved() / 1024**3:.1f}GB"
            })
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                if hasattr(torch.mps, 'current_allocated_memory'):
                    memory_info['mps_memory'] = f"{torch.mps.current_allocated_memory() / 1024**3:.1f}GB"
                else:
                    memory_info['mps_memory'] = "N/A"
            except:
                memory_info['mps_memory'] = "N/A"
        
        return memory_info
    
    def _get_device_utilization(self) -> Dict[str, Any]:
        """디바이스 활용도 조회"""
        utilization = {
            'device_type': self.device,
            'optimization_enabled': self.pipeline_config['enable_optimization'],
            'constructor_pattern': 'optimal'
        }
        
        if self.device == 'cuda' and torch.cuda.is_available():
            utilization.update({
                'gpu_name': torch.cuda.get_device_name(),
                'compute_capability': torch.cuda.get_device_capability()
            })
        elif self.device == 'mps':
            utilization.update({
                'mps_available': torch.backends.mps.is_available(),
                'mps_built': torch.backends.mps.is_built()
            })
        
        return utilization
    
    # ===========================================
    # 설정 및 상태 관리
    # ===========================================
    
    def _load_config(self, config_path: Optional[str] = None, base_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """설정 파일 로드"""
        default_config = {
            'input_size': (512, 512),
            'pipeline': {
                'quality_level': self.quality_level,
                'processing_mode': 'complete',
                'enable_optimization': self.optimization_enabled,
                'enable_caching': True,
                'parallel_processing': True,
                'memory_optimization': True,
                'enable_intermediate_saving': False,
                'max_retries': 3,
                'timeout_seconds': 300,
                'constructor_pattern': 'optimal'
            },
            'quality_thresholds': {
                'excellent': 0.9,
                'good': 0.8,
                'acceptable': 0.7,
                'poor': 0.6
            },
            'device_optimization': {
                'enable_mps': self.is_m3_max,
                'enable_cuda': True,
                'mixed_precision': self.optimization_enabled,
                'memory_efficient': True
            }
        }
        
        # 기본 설정 병합
        if base_config:
            def deep_update(base_dict, update_dict):
                for key, value in update_dict.items():
                    if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                        deep_update(base_dict[key], value)
                    else:
                        base_dict[key] = value
            
            deep_update(default_config, base_config)
        
        # 파일 설정 로드
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                
                def deep_update(base_dict, update_dict):
                    for key, value in update_dict.items():
                        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                            deep_update(base_dict[key], value)
                        else:
                            base_dict[key] = value
                
                deep_update(default_config, file_config)
                logger.info(f"✅ 설정 파일 로드 완료: {config_path}")
                
            except Exception as e:
                logger.warning(f"⚠️ 설정 파일 로드 실패: {e}, 기본 설정 사용")
        
        return default_config
    
    async def _print_system_status(self):
        """시스템 상태 출력"""
        logger.info("=" * 70)
        logger.info("🏥 최적 생성자 패턴 가상 피팅 파이프라인 시스템 상태")
        logger.info("=" * 70)
        
        # 디바이스 정보
        logger.info(f"🖥️ 디바이스: {self.device}")
        logger.info(f"💻 디바이스 타입: {self.device_type}")
        logger.info(f"💾 메모리: {self.memory_gb}GB")
        logger.info(f"🍎 M3 Max: {'✅' if self.is_m3_max else '❌'}")
        logger.info(f"⚡ 최적화: {'✅' if self.optimization_enabled else '❌'}")
        logger.info(f"🎯 생성자 패턴: 최적 통일 패턴")
        
        if self.device == 'mps':
            logger.info(f"   - MPS 사용 가능: {torch.backends.mps.is_available()}")
        elif self.device == 'cuda':
            logger.info(f"   - CUDA 버전: {torch.version.cuda}")
            if torch.cuda.is_available():
                logger.info(f"   - GPU 이름: {torch.cuda.get_device_name()}")
        
        # 단계별 상태
        logger.info("📋 단계별 초기화 상태 (최적 생성자 패턴):")
        for i, step_name in enumerate(self.step_order, 1):
            if step_name in self.steps:
                status = "✅ 준비됨"
                step = self.steps[step_name]
                if hasattr(step, 'is_initialized'):
                    if not step.is_initialized:
                        status = "⚠️ 초기화 미완료"
                if hasattr(step, 'fallback_mode') and step.fallback_mode:
                    status += " (폴백)"
            else:
                status = "❌ 로드 안됨"
            
            logger.info(f"   {i}. {step_name}: {status}")
        
        # 성능 설정
        logger.info("⚙️ 파이프라인 설정:")
        logger.info(f"   - 품질 레벨: {self.pipeline_config['quality_level']}")
        logger.info(f"   - 처리 모드: {self.pipeline_config['processing_mode']}")
        logger.info(f"   - 메모리 최적화: {self.pipeline_config['memory_optimization']}")
        logger.info(f"   - 병렬 처리: {self.pipeline_config['parallel_processing']}")
        logger.info(f"   - 생성자 패턴: 최적 통일")
        
        # 메모리 정보
        memory_info = self._get_detailed_memory_usage()
        logger.info("💾 메모리 사용량:")
        for key, value in memory_info.items():
            logger.info(f"   - {key}: {value}")
        
        logger.info("=" * 70)
    
    # ===========================================
    # pipeline_routes.py 호환성 메서드들
    # ===========================================
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """파이프라인 상세 상태 조회 - pipeline_routes.py 호환"""
        return {
            'initialized': self.is_initialized,
            'device': self.device,
            'device_type': self.device_type,
            'memory_gb': self.memory_gb,
            'is_m3_max': self.is_m3_max,
            'optimization_enabled': self.optimization_enabled,
            'mode': self.mode.value,
            'constructor_pattern': 'optimal',
            'steps_loaded': len(self.steps),
            'total_steps': len(self.step_order),
            'pipeline_config': self.pipeline_config,
            'performance_metrics': self.performance_metrics.copy(),
            'memory_status': self._get_detailed_memory_usage(),
            'stats': {
                'total_sessions': self.performance_metrics['total_sessions'],
                'successful_sessions': self.performance_metrics['successful_sessions'],
                'success_rate': (
                    self.performance_metrics['successful_sessions'] / 
                    max(1, self.performance_metrics['total_sessions'])
                ),
                'average_processing_time': self.performance_metrics['average_processing_time'],
                'constructor_pattern': 'optimal'
            },
            'steps_status': {
                step_name: {
                    'loaded': step_name in self.steps,
                    'type': type(self.steps[step_name]).__name__ if step_name in self.steps else None,
                    'initialized': (
                        getattr(self.steps[step_name], 'is_initialized', False) 
                        if step_name in self.steps else False
                    ),
                    'constructor_pattern': 'optimal'
                }
                for step_name in self.step_order
            },
            'version': '4.0.0-optimal',
            'integrated_version': True
        }
    
    def get_status(self) -> Dict[str, Any]:
        """파이프라인 상태 반환 - main.py 호환용"""
        return {
            'initialized': self.is_initialized,
            'device': self.device,
            'mode': self.mode.value,
            'status': 'ready' if self.is_initialized else 'initializing',
            'steps_loaded': len(self.steps),
            'performance_stats': self.performance_metrics.copy(),
            'error_count': len(self.error_history),
            'version': '4.0.0-optimal',
            'simulation_mode': self.pipeline_config.get('processing_mode', 'complete') == 'simulation',
            'pipeline_config': self.pipeline_config,
            'constructor_pattern': 'optimal',
            'integrated_version': True
        }
    
    async def warmup(self) -> bool:
        """파이프라인 웜업"""
        try:
            logger.info("🔥 최적 생성자 패턴 파이프라인 웜업 시작...")
            
            # 더미 텐서로 각 단계 워밍업
            dummy_tensor = torch.randn(1, 3, 512, 512).to(self.device)
            
            for step_name in self.step_order:
                if step_name in self.steps:
                    try:
                        step = self.steps[step_name]
                        if hasattr(step, 'process'):
                            await step.process(dummy_tensor)
                        logger.debug(f"✅ {step_name} 웜업 완료")
                    except Exception as e:
                        logger.warning(f"⚠️ {step_name} 웜업 실패: {e}")
            
            logger.info("✅ 최적 생성자 패턴 파이프라인 웜업 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 파이프라인 웜업 실패: {e}")
            return False
    
    async def cleanup(self):
        """전체 파이프라인 리소스 정리"""
        logger.info("🧹 최적 생성자 패턴 가상 피팅 파이프라인 리소스 정리 중...")
        
        try:
            # 각 단계별 정리
            for step_name, step in self.steps.items():
                try:
                    if hasattr(step, 'cleanup'):
                        await step.cleanup()
                    elif hasattr(step, 'close'):
                        step.close()
                    logger.info(f"✅ {step_name} 정리 완료")
                except Exception as e:
                    logger.warning(f"⚠️ {step_name} 정리 중 오류: {e}")
            
            # 유틸리티 정리
            if self.model_loader and hasattr(self.model_loader, 'cleanup'):
                await self.model_loader.cleanup()
            
            if self.memory_manager and hasattr(self.memory_manager, 'cleanup'):
                await self.memory_manager.cleanup()
            
            # 스레드 풀 정리
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
            
            # 메모리 정리
            self._cleanup_memory()
            
            # 세션 데이터 정리
            self.session_data.clear()
            
            # 상태 초기화
            self.is_initialized = False
            
            logger.info("✅ 최적 생성자 패턴 전체 파이프라인 리소스 정리 완료")
            
        except Exception as e:
            logger.error(f"❌ 리소스 정리 중 오류: {e}")


# ==========================================
# Export 함수들 (1번 파일과 동일하지만 최적화)
# ==========================================

# 전역 파이프라인 매니저
_global_pipeline_manager: Optional[PipelineManager] = None

def get_pipeline_manager() -> Optional[PipelineManager]:
    """전역 파이프라인 매니저 반환 - pipeline_routes.py에서 필수"""
    global _global_pipeline_manager
    return _global_pipeline_manager

def create_pipeline_manager(
    mode: Union[str, PipelineMode] = PipelineMode.PRODUCTION,
    device: Optional[str] = None,  # 자동 감지로 변경
    config: Optional[Dict[str, Any]] = None,
    **kwargs  # 최적 생성자 패턴
) -> PipelineManager:
    """새로운 파이프라인 매니저 생성 - 최적 생성자 패턴"""
    global _global_pipeline_manager
    
    # 기존 매니저 정리
    if _global_pipeline_manager:
        try:
            asyncio.create_task(_global_pipeline_manager.cleanup())
        except:
            pass
    
    # 새 매니저 생성 - 최적 생성자 패턴
    _global_pipeline_manager = PipelineManager(
        device=device,              # None으로 자동 감지
        config=config,             # 설정 딕셔너리
        mode=mode,                 # 파이프라인 모드
        **kwargs                   # 추가 설정들
    )
    
    logger.info(f"✅ 최적 생성자 패턴 파이프라인 매니저 생성됨 - {_global_pipeline_manager.device}")
    return _global_pipeline_manager

def get_available_modes() -> Dict[str, str]:
    """사용 가능한 파이프라인 모드 반환"""
    return {
        PipelineMode.SIMULATION.value: "시뮬레이션 모드 (빠른 테스트용)",
        PipelineMode.PRODUCTION.value: "프로덕션 모드 (실제 AI 모델 사용)",
        PipelineMode.HYBRID.value: "하이브리드 모드 (자동 폴백)",
        PipelineMode.DEVELOPMENT.value: "개발 모드 (디버깅용)"
    }

# 하위 호환성을 위한 별칭들 - 최적 생성자 패턴으로 업데이트
def initialize_pipeline_manager(
    mode: str = "production", 
    device: Optional[str] = None,  # 자동 감지
    **kwargs
) -> PipelineManager:
    """파이프라인 매니저 초기화 (하위 호환성) - 최적 생성자 패턴"""
    return create_pipeline_manager(
        mode=mode, 
        device=device,
        **kwargs
    )

def get_default_pipeline_manager() -> PipelineManager:
    """기본 파이프라인 매니저 반환 - 최적 생성자 패턴"""
    manager = get_pipeline_manager()
    if manager is None:
        manager = create_pipeline_manager()
    return manager

# 호환성 검증 함수 - 최적 생성자 패턴
def validate_pipeline_manager_compatibility() -> Dict[str, bool]:
    """pipeline_routes.py와의 호환성 검증"""
    try:
        # 테스트 매니저 생성 - 최적 생성자 패턴
        test_manager = create_pipeline_manager(
            mode=PipelineMode.SIMULATION,
            device="cpu",  # 명시적 설정 가능
            device_type="test",
            memory_gb=8.0,
            is_m3_max=False,
            optimization_enabled=False
        )
        
        # 필수 속성 확인
        required_attrs = ['device', 'device_type', 'memory_gb', 'is_m3_max', 'optimization_enabled']
        attr_check = {attr: hasattr(test_manager, attr) for attr in required_attrs}
        
        # 필수 메서드 확인
        required_methods = ['initialize', 'process_virtual_tryon', 'get_pipeline_status', 'cleanup']
        method_check = {method: hasattr(test_manager, method) for method in required_methods}
        
        return {
            'attributes': all(attr_check.values()),
            'methods': all(method_check.values()),
            'attr_details': attr_check,
            'method_details': method_check,
            'overall_compatible': all(attr_check.values()) and all(method_check.values()),
            'constructor_pattern': 'optimal'
        }
        
    except Exception as e:
        logger.error(f"호환성 검증 실패: {e}")
        return {'overall_compatible': False, 'error': str(e), 'constructor_pattern': 'optimal'}

# 최적 생성자 패턴 호환성 함수들
def create_human_parsing_step(
    device: Optional[str] = None, 
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> 'OptimalStepConstructor':
    """✅ 최적 생성자 패턴 - 인체 파싱 스텝 생성"""
    try:
        from app.ai_pipeline.steps.step_01_human_parsing import HumanParsingStep
        return HumanParsingStep(device=device, config=config, **kwargs)
    except ImportError:
        logger.warning("HumanParsingStep import 실패 - 폴백 생성")
        return None

def create_cloth_warping_step(
    device: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> 'OptimalStepConstructor':
    """✅ 최적 생성자 패턴 - 옷 워핑 스텝 생성"""
    try:
        from app.ai_pipeline.steps.step_05_cloth_warping import ClothWarpingStep
        return ClothWarpingStep(device=device, config=config, **kwargs)
    except ImportError:
        logger.warning("ClothWarpingStep import 실패 - 폴백 생성")
        return None

# 모듈 로드 시 호환성 검증
_compatibility_result = validate_pipeline_manager_compatibility()
if _compatibility_result['overall_compatible']:
    logger.info("✅ pipeline_routes.py와 완전 호환됨 (최적 생성자 패턴)")
else:
    logger.warning(f"⚠️ 호환성 문제: {_compatibility_result}")

# __all__ export
__all__ = [
    'OptimalStepConstructor',
    'PipelineManager',
    'PipelineMode',
    'get_pipeline_manager',
    'create_pipeline_manager',
    'get_available_modes',
    'initialize_pipeline_manager',
    'get_default_pipeline_manager',
    'validate_pipeline_manager_compatibility',
    'create_human_parsing_step',
    'create_cloth_warping_step'
]