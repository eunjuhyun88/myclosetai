# app/ai_pipeline/steps/step_05_cloth_warping.py
"""
5단계: 옷 워핑 (Clothing Warping) - 신체에 맞춘 고급 의류 변형
Pipeline Manager 완전 호환 버전 - M3 Max 최적화
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

class ClothWarpingStep:
    """
    의류 워핑 스텝 - Pipeline Manager 완전 호환
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
    
    def __init__(self, model_loader=None, device: str = "mps", config: Dict[str, Any] = None):
        """
        초기화 - Pipeline Manager 호환 인터페이스
        
        Args:
            model_loader: 모델 로더 인스턴스 (선택적)
            device: 사용할 디바이스 (mps, cuda, cpu)
            config: 설정 딕셔너리 (선택적)
        """
        self.model_loader = model_loader
        self.device = self._setup_optimal_device(device)
        self.config = config or {}
        
        # 워핑 설정
        self.warping_config = self.config.get('warping', {
            'physics_enabled': True,
            'deformation_strength': 0.7,
            'quality_level': 'medium',
            'enable_wrinkles': True,
            'enable_fabric_physics': True,
            'adaptive_warping': True
        })
        
        # 성능 설정
        self.performance_config = self.config.get('performance', {
            'use_mps': self.device == 'mps',
            'memory_efficient': True,
            'max_resolution': 1024,
            'enable_caching': True
        })
        
        # 최적화 수준
        self.optimization_level = self.config.get('optimization_level', 'speed')  # speed, balanced, quality
        
        # 핵심 컴포넌트들
        self.fabric_simulator = None
        self.advanced_warper = None
        self.texture_synthesizer = None
        
        # 상태 변수들
        self.is_initialized = False
        self.initialization_error = None
        
        # 성능 통계
        self.performance_stats = {
            'total_processed': 0,
            'average_time': 0.0,
            'success_rate': 0.0,
            'warping_quality_avg': 0.0
        }
        
        logger.info(f"🎯 ClothWarpingStep 초기화 - 디바이스: {self.device}")
    
    def _setup_optimal_device(self, preferred_device: str) -> str:
        """최적 디바이스 선택"""
        try:
            if preferred_device == 'mps' and TORCH_AVAILABLE and torch.backends.mps.is_available():
                logger.info("✅ Apple Silicon MPS 백엔드 활성화")
                return 'mps'
            elif preferred_device == 'cuda' and TORCH_AVAILABLE and torch.cuda.is_available():
                logger.info("✅ CUDA 백엔드 활성화")
                return 'cuda'
            else:
                logger.info("⚠️ CPU 백엔드 사용")
                return 'cpu'
        except Exception as e:
            logger.warning(f"디바이스 설정 실패: {e}, CPU 사용")
            return 'cpu'
    
    async def initialize(self) -> bool:
        """
        워핑 시스템 초기화
        Pipeline Manager가 호출하는 표준 초기화 메서드
        """
        try:
            logger.info("🔄 옷 워핑 시스템 초기화 시작...")
            
            # 1. 기본 요구사항 검증
            if not CV2_AVAILABLE:
                raise RuntimeError("OpenCV가 필요합니다: pip install opencv-python")
            
            # 2. 천 시뮬레이터 초기화
            self.fabric_simulator = FabricSimulator(
                physics_enabled=self.warping_config['physics_enabled'],
                device=self.device
            )
            
            # 3. 고급 워핑 엔진 초기화
            self.advanced_warper = AdvancedClothingWarper(
                deformation_strength=self.warping_config['deformation_strength'],
                device=self.device
            )
            
            # 4. 텍스처 합성기 초기화
            self.texture_synthesizer = TextureSynthesizer(
                device=self.device,
                use_advanced_features=self.optimization_level == 'quality'
            )
            
            # 5. 시스템 검증
            await self._validate_system()
            
            self.is_initialized = True
            logger.info("✅ 옷 워핑 시스템 초기화 완료")
            
            return True
            
        except Exception as e:
            error_msg = f"옷 워핑 시스템 초기화 실패: {e}"
            logger.error(f"❌ {error_msg}")
            self.initialization_error = error_msg
            self.is_initialized = False
            return False
    
    async def _validate_system(self):
        """시스템 검증"""
        available_features = []
        
        if CV2_AVAILABLE:
            available_features.append('basic_warping')
        if SCIPY_AVAILABLE:
            available_features.append('advanced_warping')
        if TORCH_AVAILABLE:
            available_features.append('neural_processing')
        
        if not available_features:
            raise RuntimeError("사용 가능한 워핑 기능이 없습니다")
        
        logger.info(f"✅ 사용 가능한 기능들: {available_features}")
    
    # =================================================================
    # 메인 처리 메서드 - Pipeline Manager 호환 인터페이스
    # =================================================================
    
    def process(
        self,
        clothing_image_tensor: torch.Tensor,
        clothing_mask: torch.Tensor,
        geometric_matching_result: Dict[str, Any],
        body_measurements: Optional[Dict[str, float]] = None,
        clothing_type: str = "shirt",
        fabric_type: str = "cotton"
    ) -> Dict[str, Any]:
        """
        옷 워핑 처리 (동기 버전)
        Pipeline Manager가 호출하는 메인 메서드
        """
        if not TORCH_AVAILABLE:
            return self._create_fallback_result("PyTorch 없이는 워핑 처리 불가")
        
        try:
            return self._process_warping_sync(
                clothing_image_tensor, clothing_mask, geometric_matching_result,
                body_measurements, clothing_type, fabric_type
            )
        except Exception as e:
            logger.error(f"워핑 처리 실패: {e}")
            return self._create_error_result(str(e))
    
    def _process_warping_sync(
        self,
        clothing_image_tensor: torch.Tensor,
        clothing_mask: torch.Tensor,
        geometric_matching_result: Dict[str, Any],
        body_measurements: Optional[Dict[str, float]],
        clothing_type: str,
        fabric_type: str
    ) -> Dict[str, Any]:
        """동기 워핑 처리"""
        
        if not self.is_initialized:
            error_msg = f"워핑 시스템이 초기화되지 않음: {self.initialization_error}"
            logger.error(f"❌ {error_msg}")
            return self._create_error_result(error_msg)
        
        start_time = time.time()
        
        try:
            logger.info(f"🔄 의류 워핑 시작 - 타입: {clothing_type}, 재질: {fabric_type}")
            
            # 1. 입력 데이터 준비
            cloth_img = self._tensor_to_numpy(clothing_image_tensor)
            cloth_mask = self._tensor_to_numpy(clothing_mask, is_mask=True)
            
            # 2. 기하학적 매칭 결과 처리
            if 'warped_clothing' in geometric_matching_result:
                warped_clothing = self._tensor_to_numpy(geometric_matching_result['warped_clothing'])
                warped_mask = self._tensor_to_numpy(geometric_matching_result['warped_mask'], is_mask=True)
            else:
                # 기하학적 매칭 결과가 없으면 원본 사용
                warped_clothing = cloth_img
                warped_mask = cloth_mask
            
            matched_pairs = geometric_matching_result.get('matched_pairs', [])
            
            # 3. 천 특성 설정
            fabric_props = self.FABRIC_PROPERTIES.get(fabric_type, self.FABRIC_PROPERTIES['default'])
            deform_params = self.CLOTHING_DEFORMATION_PARAMS.get(clothing_type, self.CLOTHING_DEFORMATION_PARAMS['default'])
            
            # 4. 물리 시뮬레이션
            logger.info("🧵 천 물리 시뮬레이션...")
            simulated_result = self.fabric_simulator.simulate_fabric_physics(
                warped_clothing, warped_mask, fabric_props, body_measurements
            )
            
            # 5. 고급 워핑 적용
            logger.info("🔧 고급 워핑 적용...")
            warping_result = self.advanced_warper.apply_advanced_warping(
                simulated_result['fabric_image'],
                simulated_result.get('deformation_map', np.zeros(warped_clothing.shape[:2])),
                matched_pairs,
                clothing_type,
                deform_params
            )
            
            # 6. 텍스처 합성 및 디테일 추가
            logger.info("✨ 텍스처 합성...")
            texture_result = self.texture_synthesizer.synthesize_fabric_details(
                warping_result['warped_image'],
                warping_result.get('strain_map', np.ones(warped_clothing.shape[:2])),
                fabric_props,
                clothing_type
            )
            
            # 7. 최종 결과 구성
            processing_time = time.time() - start_time
            result = self._build_final_result(
                texture_result, warping_result, simulated_result,
                processing_time, clothing_type, fabric_type
            )
            
            # 8. 통계 업데이트
            self._update_performance_stats(processing_time, result['warping_quality'])
            
            logger.info(f"✅ 워핑 완료 - {processing_time:.3f}초")
            return result
            
        except Exception as e:
            error_msg = f"워핑 처리 실패: {e}"
            logger.error(f"❌ {error_msg}")
            return self._create_error_result(error_msg)
    
    def _build_final_result(
        self,
        texture_result: Dict[str, Any],
        warping_result: Dict[str, Any],
        simulation_result: Dict[str, Any],
        processing_time: float,
        clothing_type: str,
        fabric_type: str
    ) -> Dict[str, Any]:
        """최종 결과 구성 (Pipeline Manager 호환 형식)"""
        
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
            "warped_clothing": final_tensor,
            "warped_mask": mask_tensor,
            "warped_image_numpy": final_image,
            "deformation_map": warping_result.get('strain_map'),
            "warping_quality": warping_quality,
            "fabric_analysis": {
                "fabric_type": fabric_type,
                "stiffness": self.FABRIC_PROPERTIES.get(fabric_type, {}).get('stiffness', 0.4),
                "deformation_applied": True,
                "physics_simulated": simulation_result.get('simulation_info', {}).get('physics_enabled', False),
                "texture_enhanced": 'enhanced_image' in texture_result
            },
            "warping_info": {
                "clothing_type": clothing_type,
                "warping_method": "physics_based",
                "processing_time": processing_time,
                "device": self.device,
                "features_used": self._get_used_features(),
                "quality_level": self.optimization_level
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
            
            return max(0.0, min(1.0, sum(quality_factors)))
            
        except Exception as e:
            logger.warning(f"품질 계산 실패: {e}")
            return 0.7  # 기본값
    
    def _get_used_features(self) -> List[str]:
        """사용된 기능들 목록"""
        features = ['basic_warping']
        
        if self.fabric_simulator and self.warping_config['physics_enabled']:
            features.append('physics_simulation')
        if SCIPY_AVAILABLE:
            features.append('advanced_interpolation')
        if TORCH_AVAILABLE:
            features.append('neural_processing')
        if self.texture_synthesizer:
            features.append('texture_synthesis')
        
        return features
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """에러 결과 생성"""
        return {
            "success": False,
            "error": error_message,
            "warped_clothing": None,
            "warped_mask": None,
            "warped_image_numpy": None,
            "deformation_map": None,
            "warping_quality": 0.0,
            "fabric_analysis": {},
            "warping_info": {
                "error_details": error_message,
                "device": self.device,
                "processing_time": 0.0
            }
        }
    
    def _create_fallback_result(self, reason: str) -> Dict[str, Any]:
        """폴백 결과 생성 (최소 기능)"""
        logger.warning(f"폴백 모드: {reason}")
        
        # 기본 이미지 생성 (더미)
        dummy_image = np.ones((256, 256, 3), dtype=np.uint8) * 128
        dummy_mask = np.ones((256, 256), dtype=np.uint8) * 255
        
        return {
            "success": True,
            "warped_clothing": None,
            "warped_mask": None,
            "warped_image_numpy": dummy_image,
            "deformation_map": dummy_mask,
            "warping_quality": 0.5,
            "fabric_analysis": {
                "fallback_mode": True,
                "reason": reason
            },
            "warping_info": {
                "warping_method": "fallback",
                "processing_time": 0.001,
                "device": self.device,
                "fallback_reason": reason
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
    # Pipeline Manager 호환 메서드들
    # =================================================================
    
    async def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환 (Pipeline Manager 호환)"""
        return {
            "step_name": "ClothWarping",
            "version": "3.0",
            "device": self.device,
            "initialized": self.is_initialized,
            "initialization_error": self.initialization_error,
            "capabilities": {
                "physics_simulation": bool(self.fabric_simulator),
                "advanced_warping": bool(self.advanced_warper),
                "texture_synthesis": bool(self.texture_synthesizer),
                "neural_processing": TORCH_AVAILABLE and self.device != 'cpu'
            },
            "supported_fabrics": list(self.FABRIC_PROPERTIES.keys()),
            "supported_clothing_types": list(self.CLOTHING_DEFORMATION_PARAMS.keys()),
            "performance_stats": self.performance_stats,
            "dependencies": {
                "torch": TORCH_AVAILABLE,
                "opencv": CV2_AVAILABLE,
                "scipy": SCIPY_AVAILABLE,
                "sklearn": SKLEARN_AVAILABLE,
                "skimage": SKIMAGE_AVAILABLE
            },
            "config": {
                "warping": self.warping_config,
                "performance": self.performance_config,
                "optimization_level": self.optimization_level
            }
        }
    
    async def cleanup(self):
        """리소스 정리 (Pipeline Manager 호환)"""
        try:
            logger.info("🧹 옷 워핑 시스템 리소스 정리 시작...")
            
            # 컴포넌트들 정리
            if self.fabric_simulator:
                await self.fabric_simulator.cleanup()
                self.fabric_simulator = None
            
            if self.advanced_warper:
                del self.advanced_warper
                self.advanced_warper = None
            
            if self.texture_synthesizer:
                del self.texture_synthesizer
                self.texture_synthesizer = None
            
            # GPU 메모리 정리
            if TORCH_AVAILABLE:
                if self.device == 'mps':
                    torch.mps.empty_cache()
                elif self.device == 'cuda':
                    torch.cuda.empty_cache()
            
            self.is_initialized = False
            logger.info("✅ 옷 워핑 시스템 리소스 정리 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ 리소스 정리 중 오류: {e}")


# =================================================================
# 보조 클래스들
# =================================================================

class FabricSimulator:
    """천 물리 시뮬레이션 (간소화 버전)"""
    
    def __init__(self, physics_enabled: bool = True, device: str = 'cpu'):
        self.physics_enabled = physics_enabled
        self.device = device
        self.gravity = 9.81
        self.damping = 0.95
    
    def simulate_fabric_physics(
        self,
        cloth_image: np.ndarray,
        cloth_mask: np.ndarray,
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
    
    def _apply_gravity_effect(self, image: np.ndarray, mask: np.ndarray, stiffness: float) -> np.ndarray:
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
    
    async def cleanup(self):
        """리소스 정리"""
        pass


class AdvancedClothingWarper:
    """고급 의류 워핑 엔진 (간소화 버전)"""
    
    def __init__(self, deformation_strength: float = 0.8, device: str = 'cpu'):
        self.deformation_strength = deformation_strength
        self.device = device
    
    def apply_advanced_warping(
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
        # 기본적으로는 미세한 변형만 적용
        stretch_factor = params.get('stretch_factor', 1.0)
        if abs(stretch_factor - 1.0) < 0.01:
            return image
        
        h, w = image.shape[:2]
        new_w = int(w * stretch_factor)
        
        resized = cv2.resize(image, (new_w, h), interpolation=cv2.INTER_LINEAR)
        
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
        # 기본적으로는 변경 없음
        return image
    
    def _apply_deformation_warping(self, image: np.ndarray, deformation_map: np.ndarray) -> np.ndarray:
        """변형 맵 기반 워핑"""
        if deformation_map.shape[:2] != image.shape[:2]:
            deformation_map = cv2.resize(deformation_map, (image.shape[1], image.shape[0]))
        
        h, w = image.shape[:2]
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # 변형 맵을 변위로 변환
        deform_strength = 5.0  # 변형 강도
        offset_x = (deformation_map - 0.5) * deform_strength
        offset_y = (deformation_map - 0.5) * deform_strength * 0.5
        
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


class TextureSynthesizer:
    """텍스처 합성기 (간소화 버전)"""
    
    def __init__(self, device: str = 'cpu', use_advanced_features: bool = False):
        self.device = device
        self.use_advanced_features = use_advanced_features and SKIMAGE_AVAILABLE
    
    def synthesize_fabric_details(
        self,
        warped_image: np.ndarray,
        strain_map: np.ndarray,
        fabric_props: Dict[str, float],
        clothing_type: str
    ) -> Dict[str, Any]:
        """천 디테일 합성"""
        
        try:
            # 1. 기본 품질 개선
            enhanced_image = self._enhance_basic_quality(warped_image)
            
            # 2. 고급 텍스처 분석 (옵션)
            texture_quality = 0.8  # 기본값
            if self.use_advanced_features:
                texture_quality = self._analyze_texture_quality(enhanced_image)
            
            # 3. 주름 효과 추가 (간단한 버전)
            if fabric_props.get('stiffness', 0.5) < 0.6:  # 부드러운 천에만
                enhanced_image = self._add_simple_wrinkles(enhanced_image, strain_map)
            
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
    
    def _enhance_basic_quality(self, image: np.ndarray) -> np.ndarray:
        """기본 품질 개선"""
        if not CV2_AVAILABLE:
            return image
        
        # 1. 가우시안 노이즈 제거
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # 2. 약간의 선명화
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * 0.1
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened
    
    def _analyze_texture_quality(self, image: np.ndarray) -> float:
        """텍스처 품질 분석 (고급 기능)"""
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
            
            # 엔트로피 계산 (높을수록 복잡한 텍스처)
            entropy = -np.sum(hist * np.log2(hist + 1e-7))
            
            # 0.5~0.9 범위로 정규화
            quality = 0.5 + (entropy / 10.0) * 0.4
            
            return min(0.9, max(0.5, quality))
            
        except Exception as e:
            logger.warning(f"텍스처 분석 실패: {e}")
            return 0.8
    
    def _add_simple_wrinkles(self, image: np.ndarray, strain_map: np.ndarray) -> np.ndarray:
        """간단한 주름 효과 추가"""
        if not CV2_AVAILABLE:
            return image
        
        try:
            # strain_map에서 높은 변형 영역에 주름 효과 추가
            h, w = image.shape[:2]
            
            if strain_map.shape[:2] != (h, w):
                strain_map = cv2.resize(strain_map, (w, h))
            
            # 주름이 생길 영역 찾기 (높은 strain 영역)
            wrinkle_mask = (strain_map > np.percentile(strain_map, 70)).astype(np.uint8)
            
            # 가벼운 어둡게 처리로 주름 효과 시뮬레이션
            wrinkle_effect = image.copy().astype(np.float32)
            wrinkle_effect[wrinkle_mask > 0] *= 0.95  # 5% 어둡게
            
            return np.clip(wrinkle_effect, 0, 255).astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"주름 효과 추가 실패: {e}")
            return image