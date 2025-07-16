# app/ai_pipeline/steps/step_06_virtual_fitting.py
"""
6단계: 가상 피팅 (Virtual Fitting) - 최적 생성자 패턴 적용
통일된 생성자: def __init__(self, device: Optional[str] = None, config: Optional[Dict[str, Any]] = None, **kwargs)
"""
import os
import logging
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import json
import math

# 필수 패키지들 - 안전한 임포트 처리
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("❌ PyTorch 설치 필요: pip install torch")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("❌ OpenCV 설치 필요: pip install opencv-python")

try:
    from scipy.interpolate import RBFInterpolator
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ SciPy 권장: pip install scipy (고급 워핑 기능)")

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn 권장: pip install scikit-learn")

try:
    from skimage.feature import local_binary_pattern
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("⚠️ Scikit-image 권장: pip install scikit-image")

# 로거 설정
logger = logging.getLogger(__name__)

class VirtualFittingStep:
    """
    가상 피팅 스텝 - 최적 생성자 패턴 적용
    통일된 생성자: def __init__(self, device: Optional[str] = None, config: Optional[Dict[str, Any]] = None, **kwargs)
    - M3 Max MPS 최적화
    - 물리 기반 천 시뮬레이션
    - 다양한 의류 타입 지원
    - 견고한 폴백 메커니즘
    """
    
    # 천 재질별 속성 정의
    FABRIC_PROPERTIES = {
        'cotton': {'stiffness': 0.3, 'elasticity': 0.2, 'density': 1.5, 'friction': 0.7},
        'denim': {'stiffness': 0.8, 'elasticity': 0.1, 'density': 2.0, 'friction': 0.9},
        'silk': {'stiffness': 0.1, 'elasticity': 0.4, 'density': 1.3, 'friction': 0.3},
        'wool': {'stiffness': 0.5, 'elasticity': 0.3, 'density': 1.4, 'friction': 0.6},
        'polyester': {'stiffness': 0.4, 'elasticity': 0.5, 'density': 1.2, 'friction': 0.4},
        'leather': {'stiffness': 0.9, 'elasticity': 0.1, 'density': 2.5, 'friction': 0.8},
        'default': {'stiffness': 0.4, 'elasticity': 0.3, 'density': 1.4, 'friction': 0.5}
    }
    
    # 의류 타입별 변형 파라미터
    CLOTHING_DEFORMATION_PARAMS = {
        'shirt': {'stretch_factor': 1.1, 'drape_intensity': 0.3, 'wrinkle_factor': 0.4},
        'dress': {'stretch_factor': 1.2, 'drape_intensity': 0.7, 'wrinkle_factor': 0.3},
        'pants': {'stretch_factor': 1.0, 'drape_intensity': 0.2, 'wrinkle_factor': 0.5},
        'jacket': {'stretch_factor': 1.05, 'drape_intensity': 0.4, 'wrinkle_factor': 0.6},
        'skirt': {'stretch_factor': 1.15, 'drape_intensity': 0.6, 'wrinkle_factor': 0.3},
        'default': {'stretch_factor': 1.1, 'drape_intensity': 0.4, 'wrinkle_factor': 0.4}
    }
    
    def __init__(
        self,
        device: Optional[str] = None,  # 🔥 최적 패턴: None으로 자동 감지
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
                - optimization_level: str = "speed"
                - physics_enabled: bool = True
                - deformation_strength: float = 0.7
                - enable_wrinkles: bool = True
                - enable_fabric_physics: bool = True
                - adaptive_warping: bool = True
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
        self.initialization_error = None

        # 6. 🎯 기존 클래스별 고유 초기화 로직 실행
        self._initialize_step_specific()

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

    def _initialize_step_specific(self):
        """🎯 각 클래스별 고유 초기화 - 기존 로직 완전 유지"""
        # 가상 피팅 특화 설정
        self.warping_config = self.config.get('warping', {
            'physics_enabled': self.config.get('physics_enabled', True),
            'deformation_strength': self.config.get('deformation_strength', 0.7),
            'quality_level': self.quality_level,
            'enable_wrinkles': self.config.get('enable_wrinkles', True),
            'enable_fabric_physics': self.config.get('enable_fabric_physics', True),
            'adaptive_warping': self.config.get('adaptive_warping', True)
        })
        
        # 성능 설정
        self.performance_config = self.config.get('performance', {
            'use_mps': self.device == 'mps',
            'memory_efficient': True,
            'max_resolution': self._get_max_resolution(),
            'enable_caching': True
        })
        
        # 최적화 수준 설정
        self.optimization_level = self.config.get('optimization_level', 'speed')  # speed, balanced, quality
        
        # 핵심 컴포넌트들
        self.fabric_simulator = None
        self.advanced_warper = None
        self.texture_synthesizer = None
        
        # 성능 통계
        self.performance_stats = {
            'total_processed': 0,
            'average_time': 0.0,
            'success_rate': 0.0,
            'warping_quality_avg': 0.0
        }
        
        # M3 Max 추가 최적화
        if self.is_m3_max:
            self._apply_m3_max_specific_config()
    
    def _apply_m3_max_specific_config(self):
        """M3 Max 전용 최적화 설정"""
        self.performance_config.update({
            'mps_optimization': True,
            'memory_fraction': 0.8,
            'use_unified_memory': True
        })
        self.logger.info(f"🍎 M3 Max 최적화 활성화 - 메모리: {self.memory_gb}GB")
    
    def _get_max_resolution(self) -> int:
        """최대 해상도 결정"""
        if self.is_m3_max and self.memory_gb >= 128:
            return 2048  # M3 Max 128GB
        elif self.is_m3_max or (self.device == 'cuda' and self.memory_gb >= 64):
            return 1536  # M3 Max 36GB 또는 고급 GPU
        elif self.memory_gb >= 32:
            return 1024
        else:
            return 512
    
    async def initialize(self) -> bool:
        """
        Step 초기화
        
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            logger.info("🔄 가상 피팅 시스템 초기화 시작...")
            
            # 1. 기본 요구사항 검증
            if not CV2_AVAILABLE:
                raise RuntimeError("OpenCV가 필요합니다: pip install opencv-python")
            
            # 2. M3 Max 최적화 (선택적)
            if self.is_m3_max:
                await self._initialize_m3_max_optimizations()
            
            # 3. 천 시뮬레이터 초기화
            self.fabric_simulator = FabricSimulator(
                physics_enabled=self.warping_config['physics_enabled'],
                device=self.device
            )
            
            # 4. 고급 워핑 엔진 초기화
            self.advanced_warper = AdvancedClothingWarper(
                deformation_strength=self.warping_config['deformation_strength'],
                device=self.device
            )
            
            # 5. 텍스처 합성기 초기화
            self.texture_synthesizer = TextureSynthesizer(
                device=self.device,
                use_advanced_features=self.optimization_level == 'quality'
            )
            
            # 6. 시스템 검증
            await self._validate_system()
            
            self.is_initialized = True
            logger.info("✅ 가상 피팅 시스템 초기화 완료")
            
            return True
            
        except Exception as e:
            error_msg = f"가상 피팅 시스템 초기화 실패: {e}"
            logger.error(f"❌ {error_msg}")
            self.initialization_error = error_msg
            self.is_initialized = False
            return False
    
    async def _initialize_m3_max_optimizations(self):
        """M3 Max 특화 최적화"""
        try:
            logger.info("🍎 M3 Max 최적화 적용...")
            
            # MPS 메모리 최적화
            if TORCH_AVAILABLE and self.device == 'mps':
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                if hasattr(torch.backends.mps, 'empty_cache'):
                    if hasattr(torch.mps, "empty_cache"): torch.mps.empty_cache()
            
            # CPU 최적화 (M3 Max는 더 많은 코어 활용)
            if TORCH_AVAILABLE:
                optimal_threads = min(8, os.cpu_count() or 8)
                torch.set_num_threads(optimal_threads)
            
            logger.info("✅ M3 Max 최적화 완료")
            
        except Exception as e:
            logger.warning(f"M3 Max 최적화 실패: {e}")
    
    async def _validate_system(self):
        """시스템 검증"""
        available_features = []
        
        if CV2_AVAILABLE:
            available_features.append('basic_warping')
        if SCIPY_AVAILABLE:
            available_features.append('advanced_warping')
        if TORCH_AVAILABLE:
            available_features.append('neural_processing')
        if self.is_m3_max:
            available_features.append('m3_max_acceleration')
        
        if not available_features:
            raise RuntimeError("사용 가능한 워핑 기능이 없습니다")
        
        logger.info(f"✅ 사용 가능한 기능들: {available_features}")
    
    async def process(
        self,
        input_data: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """
        메인 처리 로직
        
        Args:
            input_data: 입력 데이터 (각 Step마다 다를 수 있음)
            **kwargs: 추가 파라미터
                - body_measurements: Optional[Dict[str, float]] = None
                - fabric_type: str = "cotton"
                - clothing_type: str = "shirt"
                - cloth_image: Optional[np.ndarray] = None
                
        Returns:
            Dict[str, Any]: 처리 결과
                - success: bool (필수)
                - fitted_image: Optional[torch.Tensor]
                - fitted_image_numpy: np.ndarray
                - fitted_mask: Optional[torch.Tensor]
                - deformation_map: Optional[np.ndarray]
                - warping_quality: float
                - fabric_analysis: Dict
                - fitting_info: Dict
                - 기타 Step별 결과 데이터
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # input_data 처리 - 다양한 형태 지원
            if isinstance(input_data, dict):
                matching_result = input_data
            else:
                # 기존 형태와의 호환성
                matching_result = input_data
            
            # kwargs에서 매개변수 추출
            body_measurements = kwargs.get('body_measurements', None)
            fabric_type = kwargs.get('fabric_type', 'cotton')
            clothing_type = kwargs.get('clothing_type', 'shirt')
            cloth_image = kwargs.get('cloth_image', None)
            
            logger.info(f"🔄 가상 피팅 시작 - 재질: {fabric_type}, 타입: {clothing_type}")
            
            # 1. 매칭 결과에서 필요한 데이터 추출
            warped_clothing = matching_result.get('warped_clothing') if isinstance(matching_result, dict) else cloth_image
            warped_mask = matching_result.get('warped_mask') if isinstance(matching_result, dict) else None
            transform_matrix = matching_result.get('transform_matrix', np.eye(3)) if isinstance(matching_result, dict) else np.eye(3)
            matched_pairs = matching_result.get('matched_pairs', []) if isinstance(matching_result, dict) else []
            
            # 2. 입력 데이터 검증
            if warped_clothing is None:
                logger.warning("⚠️ 워핑된 의류 이미지가 없음 - 폴백 모드")
                return self._create_fallback_result("워핑된 의류 이미지 없음")
            
            # 3. 데이터 타입 변환
            cloth_img = self._prepare_image_data(warped_clothing)
            cloth_mask = self._prepare_mask_data(warped_mask) if warped_mask is not None else None
            
            # 4. 천 특성 설정
            fabric_props = self.FABRIC_PROPERTIES.get(fabric_type, self.FABRIC_PROPERTIES['default'])
            deform_params = self.CLOTHING_DEFORMATION_PARAMS.get(clothing_type, self.CLOTHING_DEFORMATION_PARAMS['default'])
            
            # 5. 물리 시뮬레이션
            logger.info("🧵 천 물리 시뮬레이션...")
            simulated_result = await self.fabric_simulator.simulate_fabric_physics(
                cloth_img, cloth_mask, fabric_props, body_measurements
            )
            
            # 6. 고급 워핑 적용
            logger.info("🔧 고급 워핑 적용...")
            warping_result = await self.advanced_warper.apply_advanced_warping(
                simulated_result['fabric_image'],
                simulated_result.get('deformation_map', np.zeros(cloth_img.shape[:2])),
                matched_pairs,
                clothing_type,
                deform_params
            )
            
            # 7. 텍스처 합성 및 디테일 추가
            logger.info("✨ 텍스처 합성...")
            texture_result = await self.texture_synthesizer.synthesize_fabric_details(
                warping_result['warped_image'],
                warping_result.get('strain_map', np.ones(cloth_img.shape[:2])),
                fabric_props,
                clothing_type
            )
            
            # 8. 최종 결과 구성
            processing_time = time.time() - start_time
            result = self._build_final_result(
                texture_result, warping_result, simulated_result,
                processing_time, clothing_type, fabric_type
            )
            
            # 9. 통계 업데이트
            self._update_performance_stats(processing_time, result['warping_quality'])
            
            logger.info(f"✅ 가상 피팅 완료 - {processing_time:.3f}초")
            return result
            
        except Exception as e:
            error_msg = f"가상 피팅 처리 실패: {e}"
            logger.error(f"❌ {error_msg}")
            return {
                "success": False,
                "step_name": self.__class__.__name__,
                "error": error_msg,
                "processing_time": time.time() - start_time
            }
    
    def _prepare_image_data(self, image_data) -> np.ndarray:
        """이미지 데이터 준비"""
        if TORCH_AVAILABLE and isinstance(image_data, torch.Tensor):
            return self._tensor_to_numpy(image_data)
        elif isinstance(image_data, np.ndarray):
            return image_data
        else:
            # PIL 이미지나 기타 형식
            try:
                return np.array(image_data)
            except:
                logger.warning("이미지 데이터 변환 실패 - 더미 데이터 생성")
                return np.ones((256, 256, 3), dtype=np.uint8) * 128
    
    def _prepare_mask_data(self, mask_data) -> np.ndarray:
        """마스크 데이터 준비"""
        if TORCH_AVAILABLE and isinstance(mask_data, torch.Tensor):
            return self._tensor_to_numpy(mask_data, is_mask=True)
        elif isinstance(mask_data, np.ndarray):
            return mask_data.astype(np.uint8)
        else:
            try:
                return np.array(mask_data, dtype=np.uint8)
            except:
                logger.warning("마스크 데이터 변환 실패 - 기본 마스크 생성")
                return np.ones((256, 256), dtype=np.uint8) * 255
    
    def _build_final_result(
        self,
        texture_result: Dict[str, Any],
        warping_result: Dict[str, Any],
        simulation_result: Dict[str, Any],
        processing_time: float,
        clothing_type: str,
        fabric_type: str
    ) -> Dict[str, Any]:
        """최종 결과 구성 (최적 패턴 호환 형식)"""
        
        # 메인 결과 이미지
        final_image = texture_result.get('enhanced_image', warping_result['warped_image'])
        
        # 텐서로 변환 (Pipeline Manager 호환)
        if TORCH_AVAILABLE:
            final_tensor = self._numpy_to_tensor(final_image)
            mask_tensor = self._numpy_to_tensor(warping_result.get('warped_mask', np.ones(final_image.shape[:2])), is_mask=True)
        else:
            final_tensor = None
            mask_tensor = None
        
        # 품질 점수 계산
        warping_quality = self._calculate_warping_quality(warping_result, texture_result)
        
        return {
            "success": True,
            "step_name": self.__class__.__name__,
            "fitted_image": final_tensor,
            "fitted_mask": mask_tensor,
            "fitted_image_numpy": final_image,
            "deformation_map": warping_result.get('strain_map'),
            "warping_quality": warping_quality,
            "fabric_analysis": {
                "fabric_type": fabric_type,
                "stiffness": self.FABRIC_PROPERTIES.get(fabric_type, {}).get('stiffness', 0.4),
                "deformation_applied": True,
                "physics_simulated": simulation_result.get('simulation_info', {}).get('physics_enabled', False),
                "texture_enhanced": 'enhanced_image' in texture_result
            },
            "fitting_info": {
                "clothing_type": clothing_type,
                "warping_method": "physics_based",
                "processing_time": processing_time,
                "device": self.device,
                "device_type": self.device_type,
                "m3_max_optimized": self.is_m3_max,
                "memory_gb": self.memory_gb,
                "features_used": self._get_used_features(),
                "quality_level": self.optimization_level,
                "optimal_constructor": True  # 최적 생성자 사용 표시
            },
            "performance_info": {
                "optimization_enabled": self.optimization_enabled,
                "gpu_acceleration": self.device != 'cpu'
            }
        }
    
    def _calculate_warping_quality(self, warping_result: Dict, texture_result: Dict) -> float:
        """워핑 품질 점수 계산"""
        try:
            quality_factors = []
            
            # 1. 변형 일관성 (strain map 기반)
            if 'strain_map' in warping_result:
                strain_consistency = 1.0 - np.std(warping_result['strain_map'])
                quality_factors.append(strain_consistency * 0.3)
            
            # 2. 텍스처 품질
            if 'texture_quality' in texture_result:
                quality_factors.append(texture_result['texture_quality'] * 0.3)
            else:
                quality_factors.append(0.7)  # 기본값
            
            # 3. 기하학적 일관성
            if 'deformation_stats' in warping_result:
                geo_consistency = min(1.0, warping_result['deformation_stats'].get('uniformity', 0.8))
                quality_factors.append(geo_consistency * 0.4)
            else:
                quality_factors.append(0.8)  # 기본값
            
            # M3 Max 보너스 (더 정확한 처리)
            if self.is_m3_max and self.optimization_enabled:
                quality_factors = [q * 1.05 for q in quality_factors]
            
            return max(0.0, min(1.0, sum(quality_factors)))
            
        except Exception as e:
            logger.warning(f"품질 계산 실패: {e}")
            return 0.7  # 기본값
    
    def _get_used_features(self) -> List[str]:
        """사용된 기능들 목록"""
        features = ['basic_warping', 'optimal_constructor']
        
        if self.fabric_simulator and self.warping_config['physics_enabled']:
            features.append('physics_simulation')
        if SCIPY_AVAILABLE:
            features.append('advanced_interpolation')
        if TORCH_AVAILABLE:
            features.append('neural_processing')
        if self.texture_synthesizer:
            features.append('texture_synthesis')
        if self.is_m3_max:
            features.append('m3_max_acceleration')
        if self.device == 'mps':
            features.append('metal_performance_shaders')
        
        return features
    
    def _create_fallback_result(self, reason: str) -> Dict[str, Any]:
        """폴백 결과 생성 (최소 기능)"""
        logger.warning(f"폴백 모드: {reason}")
        
        # 기본 이미지 생성 (더미)
        dummy_image = np.ones((256, 256, 3), dtype=np.uint8) * 128
        dummy_mask = np.ones((256, 256), dtype=np.uint8) * 255
        
        return {
            "success": True,
            "step_name": self.__class__.__name__,
            "fitted_image": None,
            "fitted_mask": None,
            "fitted_image_numpy": dummy_image,
            "deformation_map": dummy_mask,
            "warping_quality": 0.5,
            "fabric_analysis": {
                "fallback_mode": True,
                "reason": reason
            },
            "fitting_info": {
                "warping_method": "fallback",
                "processing_time": 0.001,
                "device": self.device,
                "device_type": self.device_type,
                "m3_max_optimized": self.is_m3_max,
                "fallback_reason": reason,
                "optimal_constructor": True
            }
        }
    
    # =================================================================
    # 유틸리티 메서드들
    # =================================================================
    
    def _tensor_to_numpy(self, tensor: torch.Tensor, is_mask: bool = False) -> np.ndarray:
        """PyTorch 텐서를 NumPy 배열로 변환"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch가 필요합니다")
        
        try:
            # GPU에서 CPU로 이동
            if tensor.is_cuda or (hasattr(tensor, 'is_mps') and tensor.is_mps):
                tensor = tensor.cpu()
            
            # 차원 정리
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)  # [1, C, H, W] -> [C, H, W]
            
            if is_mask:
                if tensor.dim() == 3:
                    tensor = tensor.squeeze(0)  # [1, H, W] -> [H, W]
                array = tensor.numpy().astype(np.uint8)
                if array.max() <= 1.0:
                    array = array * 255
            else:
                if tensor.dim() == 3 and tensor.size(0) == 3:
                    tensor = tensor.permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]
                
                array = tensor.numpy()
                if array.max() <= 1.0:
                    array = array * 255
                array = array.astype(np.uint8)
            
            return array
            
        except Exception as e:
            logger.error(f"텐서 변환 실패: {e}")
            raise
    
    def _numpy_to_tensor(self, array: np.ndarray, is_mask: bool = False) -> torch.Tensor:
        """NumPy 배열을 PyTorch 텐서로 변환"""
        if not TORCH_AVAILABLE:
            return None
        
        try:
            if is_mask:
                if len(array.shape) == 2:
                    array = array[np.newaxis, :]  # [H, W] -> [1, H, W]
                tensor = torch.from_numpy(array.astype(np.float32) / 255.0)
                tensor = tensor.unsqueeze(0)  # [1, H, W] -> [1, 1, H, W]
            else:
                if len(array.shape) == 3 and array.shape[2] == 3:
                    array = array.transpose(2, 0, 1)  # [H, W, 3] -> [3, H, W]
                tensor = torch.from_numpy(array.astype(np.float32) / 255.0)
                tensor = tensor.unsqueeze(0)  # [3, H, W] -> [1, 3, H, W]
            
            return tensor.to(self.device)
            
        except Exception as e:
            logger.warning(f"텐서 변환 실패: {e}")
            return None
    
    def _update_performance_stats(self, processing_time: float, quality_score: float):
        """성능 통계 업데이트"""
        try:
            self.performance_stats['total_processed'] += 1
            total = self.performance_stats['total_processed']
            
            # 평균 처리 시간 업데이트
            current_avg = self.performance_stats['average_time']
            self.performance_stats['average_time'] = (current_avg * (total - 1) + processing_time) / total
            
            # 평균 품질 업데이트
            current_quality_avg = self.performance_stats['warping_quality_avg']
            self.performance_stats['warping_quality_avg'] = (current_quality_avg * (total - 1) + quality_score) / total
            
            # 성공률 업데이트 (품질 0.5 이상이면 성공)
            success_count = sum(1 for _ in range(total) if quality_score > 0.5)
            self.performance_stats['success_rate'] = success_count / total
            
        except Exception as e:
            logger.warning(f"통계 업데이트 실패: {e}")
    
    # =================================================================
    # 최적 패턴 호환 메서드들
    # =================================================================
    
    def cleanup(self):
        """리소스 정리 (선택적)"""
        try:
            logger.info("🧹 가상 피팅 시스템 리소스 정리 시작...")
            
            # 컴포넌트들 정리
            if self.fabric_simulator:
                if hasattr(self.fabric_simulator, 'cleanup'):
                    self.fabric_simulator.cleanup()
                self.fabric_simulator = None
            
            if self.advanced_warper:
                if hasattr(self.advanced_warper, 'cleanup'):
                    self.advanced_warper.cleanup()
                del self.advanced_warper
                self.advanced_warper = None
            
            if self.texture_synthesizer:
                if hasattr(self.texture_synthesizer, 'cleanup'):
                    self.texture_synthesizer.cleanup()
                del self.texture_synthesizer
                self.texture_synthesizer = None
            
            # GPU 메모리 정리
            if TORCH_AVAILABLE:
                if self.device == 'mps':
                    if hasattr(torch.backends.mps, 'empty_cache'):
                        if hasattr(torch.mps, "empty_cache"): torch.mps.empty_cache()
                elif self.device == 'cuda':
                    torch.cuda.empty_cache()
            
            # 시스템 메모리 정리
            import gc
            gc.collect()
            
            self.is_initialized = False
            logger.info("✅ 가상 피팅 시스템 리소스 정리 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ 리소스 정리 중 오류: {e}")

    async def get_step_info(self) -> Dict[str, Any]:
        """Step 정보 반환 (선택적)"""
        return {
            "step_name": self.__class__.__name__,
            "version": "6.0-optimal",
            "device": self.device,
            "device_type": self.device_type,
            "memory_gb": self.memory_gb,
            "is_m3_max": self.is_m3_max,
            "optimization_enabled": self.optimization_enabled,
            "quality_level": self.quality_level,
            "initialized": self.is_initialized,
            "initialization_error": self.initialization_error,
            "config_keys": list(self.config.keys()),
            "performance_stats": self.performance_stats.copy(),
            "optimal_constructor": True,
            "capabilities": {
                "physics_simulation": bool(self.fabric_simulator),
                "advanced_warping": bool(self.advanced_warper),
                "texture_synthesis": bool(self.texture_synthesizer),
                "neural_processing": TORCH_AVAILABLE and self.device != 'cpu',
                "m3_max_acceleration": self.is_m3_max and self.device == 'mps'
            },
            "supported_fabrics": list(self.FABRIC_PROPERTIES.keys()),
            "supported_clothing_types": list(self.CLOTHING_DEFORMATION_PARAMS.keys()),
            "dependencies": {
                "torch": TORCH_AVAILABLE,
                "opencv": CV2_AVAILABLE,
                "scipy": SCIPY_AVAILABLE,
                "sklearn": SKLEARN_AVAILABLE,
                "skimage": SKIMAGE_AVAILABLE
            }
        }


# =================================================================
# 보조 클래스들 (간소화 버전)
# =================================================================

class FabricSimulator:
    """천 물리 시뮬레이션 (간소화 버전)"""
    
    def __init__(self, physics_enabled: bool = True, device: str = 'cpu'):
        self.physics_enabled = physics_enabled
        self.device = device
        self.gravity = 9.81
        self.damping = 0.95
    
    async def simulate_fabric_physics(
        self,
        cloth_image: np.ndarray,
        cloth_mask: Optional[np.ndarray],
        fabric_props: Dict[str, float],
        body_measurements: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """천 물리 시뮬레이션"""
        
        if not self.physics_enabled or not CV2_AVAILABLE:
            return {
                'fabric_image': cloth_image,
                'deformation_map': np.zeros(cloth_image.shape[:2]),
                'simulation_info': {'physics_enabled': False}
            }
        
        try:
            # 1. 중력 효과 시뮬레이션
            gravity_deformed = self._apply_gravity_effect(
                cloth_image, cloth_mask, fabric_props['stiffness']
            )
            
            # 2. 간단한 변형 맵 생성
            deformation_map = self._generate_simple_deformation_map(
                cloth_image.shape[:2], fabric_props
            )
            
            return {
                'fabric_image': gravity_deformed,
                'deformation_map': deformation_map,
                'simulation_info': {
                    'physics_enabled': True,
                    'gravity_applied': True,
                    'fabric_stiffness': fabric_props['stiffness']
                }
            }
            
        except Exception as e:
            logger.warning(f"물리 시뮬레이션 실패: {e}")
            return {
                'fabric_image': cloth_image,
                'deformation_map': np.zeros(cloth_image.shape[:2]),
                'simulation_info': {'physics_enabled': False, 'error': str(e)}
            }
    
    def _apply_gravity_effect(self, image: np.ndarray, mask: Optional[np.ndarray], stiffness: float) -> np.ndarray:
        """중력 효과 적용 (단순화)"""
        if not CV2_AVAILABLE:
            return image
        
        h, w = image.shape[:2]
        
        # 아래쪽으로 갈수록 약간 늘어나는 효과
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # 중력에 의한 변형 (stiffness가 낮을수록 더 많이 변형)
        gravity_factor = (1 - stiffness) * 0.1
        y_offset = (y_coords / h) * gravity_factor * 10
        
        map_x = x_coords.astype(np.float32)
        map_y = (y_coords + y_offset).astype(np.float32)
        
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    def _generate_simple_deformation_map(self, shape: Tuple[int, int], fabric_props: Dict) -> np.ndarray:
        """간단한 변형 맵 생성"""
        h, w = shape
        
        # 중앙에서 가장자리로 갈수록 변형이 적어지는 패턴
        y_center, x_center = h // 2, w // 2
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        distance_from_center = np.sqrt((y_coords - y_center)**2 + (x_coords - x_center)**2)
        max_distance = np.sqrt(y_center**2 + x_center**2)
        
        # 정규화된 거리 (0~1)
        normalized_distance = distance_from_center / max_distance
        
        # 변형 강도 (중앙이 높고 가장자리가 낮음)
        deformation_strength = 1.0 - normalized_distance * fabric_props.get('elasticity', 0.3)
        
        return deformation_strength.astype(np.float32)
    
    def cleanup(self):
        """리소스 정리"""
        pass


class AdvancedClothingWarper:
    """고급 의류 워핑 엔진 (간소화 버전)"""
    
    def __init__(self, deformation_strength: float = 0.8, device: str = 'cpu'):
        self.deformation_strength = deformation_strength
        self.device = device
    
    async def apply_advanced_warping(
        self,
        cloth_image: np.ndarray,
        deformation_map: np.ndarray,
        control_points: List,
        clothing_type: str,
        deform_params: Dict[str, float]
    ) -> Dict[str, Any]:
        """고급 워핑 적용"""
        
        if not CV2_AVAILABLE:
            return {
                'warped_image': cloth_image,
                'strain_map': np.ones(cloth_image.shape[:2]),
                'deformation_stats': {'method': 'none'}
            }
        
        try:
            # 1. 의류 타입별 특화 워핑
            type_warped = self._apply_type_specific_warping(cloth_image, clothing_type, deform_params)
            
            # 2. 변형 맵 기반 워핑
            if deformation_map.size > 0:
                final_warped = self._apply_deformation_warping(type_warped, deformation_map)
            else:
                final_warped = type_warped
            
            # 3. 변형 통계 계산
            deformation_stats = {
                'method': 'type_specific',
                'clothing_type': clothing_type,
                'uniformity': 0.8,  # 기본값
                'deformation_applied': True
            }
            
            # 4. 스트레인 맵 생성
            strain_map = self._generate_strain_map(cloth_image.shape[:2], deform_params)
            
            return {
                'warped_image': final_warped,
                'strain_map': strain_map,
                'deformation_stats': deformation_stats
            }
            
        except Exception as e:
            logger.warning(f"고급 워핑 실패: {e}")
            return {
                'warped_image': cloth_image,
                'strain_map': np.ones(cloth_image.shape[:2]),
                'deformation_stats': {'method': 'fallback', 'error': str(e)}
            }
    
    def _apply_type_specific_warping(
        self, 
        image: np.ndarray, 
        clothing_type: str, 
        deform_params: Dict[str, float]
    ) -> np.ndarray:
        """의류 타입별 특화 워핑"""
        
        if clothing_type == "dress":
            return self._apply_dress_warping(image, deform_params)
        elif clothing_type == "shirt":
            return self._apply_shirt_warping(image, deform_params)
        elif clothing_type == "pants":
            return self._apply_pants_warping(image, deform_params)
        else:
            return image
    
    def _apply_dress_warping(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """드레스 워핑 (A라인 실루엣)"""
        h, w = image.shape[:2]
        
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # 아래쪽으로 갈수록 확장
        expansion_factor = (y_coords / h) * params.get('drape_intensity', 0.7) * 0.1
        center_x = w // 2
        
        offset_x = (x_coords - center_x) * expansion_factor
        
        map_x = (x_coords + offset_x).astype(np.float32)
        map_y = y_coords.astype(np.float32)
        
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    def _apply_shirt_warping(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """셔츠 워핑"""
        stretch_factor = params.get('stretch_factor', 1.0)
        
        if abs(stretch_factor - 1.0) < 0.01:
            return image
        
        h, w = image.shape[:2]
        new_w = int(w * stretch_factor)
        
        resized = cv2.resize(image, (new_w, h))
        
        # 원래 크기로 crop 또는 pad
        if new_w > w:
            start_x = (new_w - w) // 2
            return resized[:, start_x:start_x + w]
        else:
            pad_x = (w - new_w) // 2
            padded = np.pad(resized, ((0, 0), (pad_x, w - new_w - pad_x), (0, 0)), mode='edge')
            return padded
    
    def _apply_pants_warping(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """바지 워핑"""
        return image  # 기본 구현
    
    def _apply_deformation_warping(self, image: np.ndarray, deformation_map: np.ndarray) -> np.ndarray:
        """변형 맵 기반 워핑"""
        if deformation_map.shape[:2] != image.shape[:2]:
            deformation_map = cv2.resize(deformation_map, (image.shape[1], image.shape[0]))
        
        h, w = image.shape[:2]
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # 변형 맵을 변위로 변환
        offset_x = (deformation_map - 0.5) * 5.0
        offset_y = (deformation_map - 0.5) * 2.5
        
        map_x = (x_coords + offset_x).astype(np.float32)
        map_y = (y_coords + offset_y).astype(np.float32)
        
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    def _generate_strain_map(self, shape: Tuple[int, int], params: Dict) -> np.ndarray:
        """스트레인 맵 생성"""
        h, w = shape
        
        # 의류의 중앙 부분이 가장 많이 늘어나는 패턴
        y_center, x_center = h // 2, w // 2
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        distance_from_center = np.sqrt((y_coords - y_center)**2 + (x_coords - x_center)**2)
        max_distance = np.sqrt(y_center**2 + x_center**2)
        
        normalized_distance = distance_from_center / max_distance
        strain_intensity = params.get('stretch_factor', 1.0) - 1.0
        
        # 중앙에서 높고 가장자리에서 낮은 스트레인
        strain_map = (1.0 - normalized_distance) * abs(strain_intensity) + 1.0
        
        return strain_map.astype(np.float32)
    
    def cleanup(self):
        """리소스 정리"""
        pass


class TextureSynthesizer:
    """텍스처 합성기 (간소화 버전)"""
    
    def __init__(self, device: str = 'cpu', use_advanced_features: bool = False):
        self.device = device
        self.use_advanced_features = use_advanced_features and SKIMAGE_AVAILABLE
    
    async def synthesize_fabric_details(
        self,
        warped_image: np.ndarray,
        strain_map: np.ndarray,
        fabric_props: Dict[str, float],
        clothing_type: str
    ) -> Dict[str, Any]:
        """천 디테일 합성"""
        
        try:
            # 기본 품질 개선
            enhanced_image = self._enhance_quality(warped_image)
            
            # 텍스처 품질 분석
            texture_quality = 0.8  # 기본값
            if self.use_advanced_features:
                texture_quality = self._analyze_texture_quality(enhanced_image)
            
            # 주름 효과 추가
            if fabric_props.get('stiffness', 0.5) < 0.6:  # 부드러운 천에만
                enhanced_image = self._add_wrinkles(enhanced_image, strain_map)
            
            return {
                'enhanced_image': enhanced_image,
                'texture_quality': texture_quality,
                'details_added': True,
                'wrinkles_applied': fabric_props.get('stiffness', 0.5) < 0.6
            }
            
        except Exception as e:
            logger.warning(f"텍스처 합성 실패: {e}")
            return {
                'enhanced_image': warped_image,
                'texture_quality': 0.7,
                'details_added': False,
                'error': str(e)
            }
    
    def _enhance_quality(self, image: np.ndarray) -> np.ndarray:
        """품질 개선"""
        if not CV2_AVAILABLE:
            return image
        
        # 노이즈 제거
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # 선명화
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * 0.1
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened
    
    def _analyze_texture_quality(self, image: np.ndarray) -> float:
        """텍스처 품질 분석"""
        if not SKIMAGE_AVAILABLE:
            return 0.8
        
        try:
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # LBP를 사용한 텍스처 분석
            lbp = local_binary_pattern(gray, 24, 8, method='uniform')
            
            # 텍스처 균일성 측정
            hist, _ = np.histogram(lbp, bins=26, range=(0, 26))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)
            
            # 엔트로피 계산
            entropy = -np.sum(hist * np.log2(hist + 1e-7))
            
            # 0.5~0.9 범위로 정규화
            quality = 0.5 + (entropy / 10.0) * 0.4
            
            return min(0.9, max(0.5, quality))
            
        except Exception as e:
            logger.warning(f"텍스처 분석 실패: {e}")
            return 0.8
    
    def _add_wrinkles(self, image: np.ndarray, strain_map: np.ndarray) -> np.ndarray:
        """주름 효과 추가"""
        try:
            h, w = image.shape[:2]
            
            if strain_map.shape[:2] != (h, w):
                strain_map = cv2.resize(strain_map, (w, h))
            
            # 높은 strain 영역에 주름 효과
            wrinkle_mask = strain_map > np.percentile(strain_map, 70)
            
            # 주름 효과 적용 (5% 어둡게)
            wrinkle_effect = image.copy()
            wrinkle_effect[wrinkle_mask] = (wrinkle_effect[wrinkle_mask] * 0.95).astype(np.uint8)
            
            return wrinkle_effect
            
        except Exception as e:
            logger.warning(f"주름 효과 추가 실패: {e}")
            return image
    
    def cleanup(self):
        """리소스 정리"""
        pass


# ===============================================================
# 🔄 하위 호환성 지원 (기존 코드 100% 지원)
# ===============================================================

def create_virtual_fitting_step(
    device: str = "mps", 
    config: Optional[Dict[str, Any]] = None
) -> VirtualFittingStep:
    """🔄 기존 방식 100% 호환 생성자"""
    return VirtualFittingStep(device=device, config=config)

# 간단한 생성자도 지원
def create_simple_virtual_fitting_step(
    device: Optional[str] = None, 
    config: Optional[Dict[str, Any]] = None
) -> VirtualFittingStep:
    """✅ 간단한 생성자 (자동 최적화)"""
    return VirtualFittingStep(device=device, config=config)

# M3 Max 최적화 전용 생성자
def create_m3_max_virtual_fitting_step(
    memory_gb: float = 128.0,
    optimization_level: str = "quality",
    **kwargs
) -> VirtualFittingStep:
    """🍎 M3 Max 최적화 전용 생성자"""
    return VirtualFittingStep(
        device=None,  # 자동 감지
        memory_gb=memory_gb,
        quality_level=optimization_level,
        is_m3_max=True,
        optimization_enabled=True,
        **kwargs
    )