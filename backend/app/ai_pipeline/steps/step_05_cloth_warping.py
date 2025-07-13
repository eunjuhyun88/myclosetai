"""
5단계: 옷 워핑 (Clothing Warping) - 신체에 맞춘 고급 의류 변형
M3 Max 최적화 버전 (물리 시뮬레이션 + 천 특성 고려)
"""
import os
import logging
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import cv2
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import json
import math

logger = logging.getLogger(__name__)

class ClothingWarpingStep:
    """옷 워핑 스텝 - 신체에 맞춘 고급 의류 변형"""
    
    def __init__(self, model_loader, device: str, config: Dict[str, Any] = None):
        """
        Args:
            model_loader: 모델 로더 인스턴스
            device: 사용할 디바이스 ('cpu', 'cuda', 'mps')
            config: 설정 딕셔너리
        """
        self.model_loader = model_loader
        self.device = device
        self.config = config or {}
        
        # 워핑 설정
        self.warping_config = self.config.get('warping', {
            'deformation_strength': 0.8,
            'physics_enabled': True,
            'fabric_stiffness': 0.5,
            'gravity_factor': 0.1,
            'wind_effect': 0.0,
            'wrinkle_simulation': True
        })
        
        # 천 물리 특성 (재질별)
        self.fabric_properties = {
            'cotton': {'stiffness': 0.6, 'elasticity': 0.3, 'thickness': 0.5},
            'denim': {'stiffness': 0.9, 'elasticity': 0.1, 'thickness': 0.8},
            'silk': {'stiffness': 0.2, 'elasticity': 0.4, 'thickness': 0.2},
            'wool': {'stiffness': 0.7, 'elasticity': 0.2, 'thickness': 0.7},
            'polyester': {'stiffness': 0.4, 'elasticity': 0.6, 'thickness': 0.3},
            'leather': {'stiffness': 0.95, 'elasticity': 0.05, 'thickness': 0.9},
            'default': {'stiffness': 0.5, 'elasticity': 0.5, 'thickness': 0.5}
        }
        
        # 성능 최적화 (M3 Max)
        self.use_mps = device == 'mps' and torch.backends.mps.is_available()
        self.optimization_level = self.config.get('optimization_level', 'balanced')  # fast, balanced, quality
        
        # 워핑 컴포넌트들
        self.fabric_simulator = None
        self.advanced_warper = None
        self.texture_synthesizer = None
        
        self.is_initialized = False
        
        logger.info(f"🎯 옷 워핑 스텝 초기화 - 디바이스: {device}, MPS: {self.use_mps}")
    
    async def initialize(self) -> bool:
        """초기화"""
        try:
            logger.info("🔄 옷 워핑 시스템 초기화 중...")
            
            # 천 시뮬레이터 초기화
            self.fabric_simulator = FabricSimulator(
                physics_enabled=self.warping_config['physics_enabled'],
                device=self.device
            )
            
            # 고급 워핑 엔진 초기화
            self.advanced_warper = AdvancedClothingWarper(
                deformation_strength=self.warping_config['deformation_strength'],
                device=self.device
            )
            
            # 텍스처 합성기 초기화
            self.texture_synthesizer = TextureSynthesizer(
                device=self.device,
                use_neural_synthesis=self.optimization_level == 'quality'
            )
            
            self.is_initialized = True
            logger.info("✅ 옷 워핑 시스템 초기화 완료")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 옷 워핑 시스템 초기화 실패: {e}")
            self.is_initialized = False
            return False
    
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
        옷 워핑 처리
        
        Args:
            clothing_image_tensor: 의류 이미지 텐서 [1, 3, H, W]
            clothing_mask: 의류 마스크 텐서 [1, 1, H, W]
            geometric_matching_result: 4단계 기하학적 매칭 결과
            body_measurements: 신체 치수 정보
            clothing_type: 의류 타입 (shirt, pants, dress, etc.)
            fabric_type: 천 재질 (cotton, denim, silk, etc.)
            
        Returns:
            처리 결과 딕셔너리
        """
        if not self.is_initialized:
            raise RuntimeError("옷 워핑 시스템이 초기화되지 않았습니다.")
        
        start_time = time.time()
        
        try:
            # 입력 데이터 변환
            cloth_img = self._tensor_to_numpy(clothing_image_tensor)
            cloth_mask = self._tensor_to_numpy(clothing_mask, is_mask=True)
            
            # 기하학적 매칭 결과 추출
            warped_clothing = self._tensor_to_numpy(geometric_matching_result['warped_clothing'])
            warped_mask = self._tensor_to_numpy(geometric_matching_result['warped_mask'], is_mask=True)
            matched_pairs = geometric_matching_result['matched_pairs']
            
            # 천 특성 설정
            fabric_props = self.fabric_properties.get(fabric_type, self.fabric_properties['default'])
            
            # 1. 고급 천 시뮬레이션
            logger.info("🧵 1단계: 천 물리 시뮬레이션...")
            simulated_clothing = self.fabric_simulator.simulate_fabric_physics(
                warped_clothing, warped_mask, fabric_props, body_measurements
            )
            
            # 2. 세밀한 변형 적용
            logger.info("🔧 2단계: 세밀한 의류 변형...")
            refined_warping = self.advanced_warper.apply_advanced_warping(
                simulated_clothing['fabric_image'],
                simulated_clothing['deformation_map'],
                matched_pairs,
                clothing_type
            )
            
            # 3. 주름 및 디테일 합성
            logger.info("✨ 3단계: 주름 및 디테일 합성...")
            detailed_clothing = self.texture_synthesizer.synthesize_fabric_details(
                refined_warping['warped_image'],
                refined_warping['strain_map'],
                fabric_props,
                clothing_type
            )
            
            # 4. 최종 품질 개선
            logger.info("🎨 4단계: 최종 품질 개선...")
            final_result = self._apply_final_enhancements(
                detailed_clothing['enhanced_image'],
                detailed_clothing['detail_mask'],
                fabric_props
            )
            
            # 5. 결과 평가
            quality_metrics = self._evaluate_warping_quality(
                cloth_img, final_result['final_image'], fabric_props
            )
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "warped_clothing": self._numpy_to_tensor(final_result['final_image']),
                "warped_mask": self._numpy_to_tensor(final_result['final_mask'], is_mask=True),
                "deformation_map": final_result['deformation_visualization'],
                "strain_map": detailed_clothing['strain_map'],
                "wrinkle_map": detailed_clothing.get('wrinkle_map', None),
                "fabric_properties_used": fabric_props,
                "quality_metrics": quality_metrics,
                "simulation_details": {
                    "physics_simulation": simulated_clothing['simulation_info'],
                    "deformation_stats": refined_warping['deformation_stats'],
                    "texture_synthesis": detailed_clothing['synthesis_info']
                },
                "processing_time": processing_time,
                "optimization_level": self.optimization_level
            }
            
            logger.info(f"✅ 옷 워핑 완료 - 처리시간: {processing_time:.3f}초, 품질: {quality_metrics['overall_quality']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 옷 워핑 처리 실패: {e}")
            raise
    
    def _tensor_to_numpy(self, tensor: torch.Tensor, is_mask: bool = False) -> np.ndarray:
        """텐서를 numpy 배열로 변환"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        if is_mask:
            if tensor.dim() == 3:
                tensor = tensor.squeeze(0)
            return (tensor.cpu().numpy() * 255).astype(np.uint8)
        else:
            if tensor.shape[0] == 3:
                tensor = tensor.permute(1, 2, 0)
            
            if tensor.max() <= 1.0:
                tensor = tensor * 255
            
            return tensor.cpu().numpy().astype(np.uint8)
    
    def _numpy_to_tensor(self, array: np.ndarray, is_mask: bool = False) -> torch.Tensor:
        """numpy 배열을 텐서로 변환"""
        if is_mask:
            if array.ndim == 2:
                tensor = torch.from_numpy(array / 255.0).float()
                return tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        else:
            if array.ndim == 3:
                tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
                return tensor.unsqueeze(0).to(self.device)
        
        return torch.from_numpy(array).to(self.device)
    
    def _apply_final_enhancements(
        self,
        enhanced_image: np.ndarray,
        detail_mask: np.ndarray,
        fabric_props: Dict[str, float]
    ) -> Dict[str, np.ndarray]:
        """최종 품질 개선"""
        
        final_image = enhanced_image.copy()
        
        # 1. 색상 보정
        final_image = self._enhance_color_consistency(final_image)
        
        # 2. 엣지 샤프닝 (천 재질에 따라)
        if fabric_props['stiffness'] > 0.7:
            # 딱딱한 재질은 선명하게
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            final_image = cv2.filter2D(final_image, -1, kernel * 0.1)
        
        # 3. 노이즈 제거
        final_image = cv2.bilateralFilter(final_image, 5, 50, 50)
        
        # 4. 마스크 정제
        final_mask = cv2.morphologyEx(detail_mask, cv2.MORPH_CLOSE, 
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        
        # 5. 변형 시각화
        deformation_viz = self._create_deformation_visualization(final_image, detail_mask)
        
        return {
            'final_image': final_image,
            'final_mask': final_mask,
            'deformation_visualization': deformation_viz
        }
    
    def _enhance_color_consistency(self, image: np.ndarray) -> np.ndarray:
        """색상 일관성 개선"""
        # LAB 색공간에서 색상 균형 조정
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE를 L 채널에 적용
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # 재결합
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _create_deformation_visualization(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> np.ndarray:
        """변형 시각화 생성"""
        
        # 그리드 패턴으로 변형 표시
        h, w = image.shape[:2]
        viz = image.copy()
        
        # 격자 그리기
        grid_spacing = 20
        for i in range(0, h, grid_spacing):
            cv2.line(viz, (0, i), (w, i), (0, 255, 0), 1)
        for j in range(0, w, grid_spacing):
            cv2.line(viz, (j, 0), (j, h), (0, 255, 0), 1)
        
        # 마스크 영역만 표시
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255
        viz = viz * mask_3ch + image * (1 - mask_3ch)
        
        return viz.astype(np.uint8)
    
    def _evaluate_warping_quality(
        self,
        original_cloth: np.ndarray,
        warped_cloth: np.ndarray,
        fabric_props: Dict[str, float]
    ) -> Dict[str, float]:
        """워핑 품질 평가"""
        
        metrics = {}
        
        # 1. 텍스처 보존도
        metrics['texture_preservation'] = self._calculate_texture_similarity(
            original_cloth, warped_cloth
        )
        
        # 2. 변형 자연스러움
        metrics['deformation_naturalness'] = self._assess_deformation_naturalness(
            warped_cloth, fabric_props
        )
        
        # 3. 엣지 품질
        metrics['edge_quality'] = self._evaluate_edge_quality(warped_cloth)
        
        # 4. 색상 일관성
        metrics['color_consistency'] = self._measure_color_consistency(warped_cloth)
        
        # 5. 전체 품질 점수
        metrics['overall_quality'] = (
            metrics['texture_preservation'] * 0.3 +
            metrics['deformation_naturalness'] * 0.4 +
            metrics['edge_quality'] * 0.2 +
            metrics['color_consistency'] * 0.1
        )
        
        return metrics
    
    def _calculate_texture_similarity(
        self, 
        original: np.ndarray, 
        warped: np.ndarray
    ) -> float:
        """텍스처 유사도 계산"""
        try:
            # 크기 맞추기
            if original.shape != warped.shape:
                warped_resized = cv2.resize(warped, (original.shape[1], original.shape[0]))
            else:
                warped_resized = warped
            
            # 그레이스케일 변환
            if len(original.shape) == 3:
                orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                warp_gray = cv2.cvtColor(warped_resized, cv2.COLOR_BGR2GRAY)
            else:
                orig_gray = original
                warp_gray = warped_resized
            
            # 텍스처 특징 추출 (LBP 패턴)
            from skimage.feature import local_binary_pattern
            
            orig_lbp = local_binary_pattern(orig_gray, 24, 8, method='uniform')
            warp_lbp = local_binary_pattern(warp_gray, 24, 8, method='uniform')
            
            # 히스토그램 비교
            orig_hist = np.histogram(orig_lbp, bins=26)[0]
            warp_hist = np.histogram(warp_lbp, bins=26)[0]
            
            # 정규화
            orig_hist = orig_hist / (orig_hist.sum() + 1e-7)
            warp_hist = warp_hist / (warp_hist.sum() + 1e-7)
            
            # 교집합 계산
            similarity = np.sum(np.minimum(orig_hist, warp_hist))
            
            return similarity
            
        except Exception:
            # 간단한 히스토그램 비교로 fallback
            orig_hist = cv2.calcHist([original], [0], None, [256], [0, 256])
            warp_hist = cv2.calcHist([warped], [0], None, [256], [0, 256])
            return cv2.compareHist(orig_hist, warp_hist, cv2.HISTCMP_CORREL)
    
    def _assess_deformation_naturalness(
        self, 
        warped_cloth: np.ndarray, 
        fabric_props: Dict[str, float]
    ) -> float:
        """변형 자연스러움 평가"""
        
        # 그래디언트 분석
        gray = cv2.cvtColor(warped_cloth, cv2.COLOR_BGR2GRAY) if len(warped_cloth.shape) == 3 else warped_cloth
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 변형의 부드러움 측정
        smoothness = 1.0 / (1.0 + np.std(gradient_magnitude))
        
        # 천 재질 특성 고려
        stiffness_factor = fabric_props['stiffness']
        expected_smoothness = 0.5 + stiffness_factor * 0.3  # 딱딱한 재질일수록 더 균등한 변형
        
        naturalness = 1.0 - abs(smoothness - expected_smoothness)
        
        return max(0.0, min(1.0, naturalness))
    
    def _evaluate_edge_quality(self, image: np.ndarray) -> float:
        """엣지 품질 평가"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Canny 엣지 검출
        edges = cv2.Canny(gray, 50, 150)
        
        # 엣지의 연결성 평가
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.5
        
        # 가장 큰 윤곽선의 연속성 평가
        main_contour = max(contours, key=cv2.contourArea)
        
        # 윤곽선의 부드러움 측정 (곡률 변화)
        if len(main_contour) < 5:
            return 0.5
        
        # 근사화 후 원본과의 차이
        epsilon = 0.02 * cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        
        approximation_quality = len(approx) / len(main_contour)
        edge_quality = 1.0 - approximation_quality  # 적게 근사화될수록 부드러운 엣지
        
        return max(0.0, min(1.0, edge_quality))
    
    def _measure_color_consistency(self, image: np.ndarray) -> float:
        """색상 일관성 측정"""
        if len(image.shape) != 3:
            return 1.0
        
        # 각 채널별 분산 계산
        b_var = np.var(image[:, :, 0])
        g_var = np.var(image[:, :, 1])
        r_var = np.var(image[:, :, 2])
        
        # 채널 간 분산의 균등성
        variances = [b_var, g_var, r_var]
        mean_var = np.mean(variances)
        
        if mean_var == 0:
            return 1.0
        
        consistency = 1.0 - (np.std(variances) / mean_var)
        
        return max(0.0, min(1.0, consistency))
    
    async def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "step_name": "ClothingWarping",
            "version": "1.0",
            "device": self.device,
            "use_mps": self.use_mps,
            "initialized": self.is_initialized,
            "optimization_level": self.optimization_level,
            "warping_config": self.warping_config,
            "supported_fabrics": list(self.fabric_properties.keys()),
            "supported_clothing_types": ["shirt", "pants", "dress", "jacket", "skirt"],
            "features": [
                "physics_simulation", 
                "texture_synthesis", 
                "wrinkle_generation",
                "fabric_property_modeling"
            ]
        }
    
    async def cleanup(self):
        """리소스 정리"""
        if self.fabric_simulator:
            await self.fabric_simulator.cleanup()
            self.fabric_simulator = None
        
        if self.advanced_warper:
            del self.advanced_warper
            self.advanced_warper = None
        
        if self.texture_synthesizer:
            del self.texture_synthesizer
            self.texture_synthesizer = None
        
        self.is_initialized = False
        logger.info("🧹 옷 워핑 스텝 리소스 정리 완료")


class FabricSimulator:
    """천 물리 시뮬레이션"""
    
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
        
        if not self.physics_enabled:
            return {
                'fabric_image': cloth_image,
                'deformation_map': np.zeros(cloth_image.shape[:2]),
                'simulation_info': {'physics_enabled': False}
            }
        
        # 1. 중력 효과 시뮬레이션
        gravity_deformed = self._apply_gravity_effect(
            cloth_image, cloth_mask, fabric_props['stiffness']
        )
        
        # 2. 신체 압력에 의한 변형
        pressure_deformed = self._apply_body_pressure(
            gravity_deformed, fabric_props, body_measurements
        )
        
        # 3. 변형 맵 생성
        deformation_map = self._calculate_deformation_map(
            cloth_image, pressure_deformed
        )
        
        return {
            'fabric_image': pressure_deformed,
            'deformation_map': deformation_map,
            'simulation_info': {
                'physics_enabled': True,
                'gravity_applied': True,
                'pressure_applied': True,
                'fabric_stiffness': fabric_props['stiffness']
            }
        }
    
    def _apply_gravity_effect(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        stiffness: float
    ) -> np.ndarray:
        """중력 효과 적용"""
        
        # 딱딱한 재질일수록 중력 효과 적음
        gravity_strength = (1.0 - stiffness) * 0.1
        
        if gravity_strength < 0.01:
            return image
        
        # 수직 방향 워핑 생성
        h, w = image.shape[:2]
        
        # 중력에 의한 변형 맵
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # 아래쪽으로 갈수록 더 많이 늘어짐
        deformation = gravity_strength * (y_coords / h) * mask / 255.0
        
        # 워핑 적용
        map_y = (y_coords + deformation).astype(np.float32)
        map_x = x_coords.astype(np.float32)
        
        warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return warped
    
    def _apply_body_pressure(
        self,
        image: np.ndarray,
        fabric_props: Dict[str, float],
        body_measurements: Optional[Dict[str, float]]
    ) -> np.ndarray:
        """신체 압력에 의한 변형"""
        
        if not body_measurements:
            return image
        
        # 탄성에 따른 압력 반응
        elasticity = fabric_props['elasticity']
        
        # BMI가 높을수록 더 많은 압력
        bmi = body_measurements.get('bmi', 22.0)
        pressure_factor = max(0.0, (bmi - 18.5) / 10.0) * elasticity
        
        if pressure_factor < 0.01:
            return image
        
        # 중앙 부분에 더 많은 압력 (배, 가슴 부위)
        h, w = image.shape[:2]
        y_center, x_center = h // 2, w // 2
        
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # 가우시안 압력 분포
        distance = np.sqrt((x_coords - x_center)**2 + (y_coords - y_center)**2)
        pressure_map = np.exp(-distance**2 / (2 * (min(h, w) * 0.3)**2))
        
        # 외곽으로 밀어내는 효과
        deform_x = pressure_factor * pressure_map * np.sign(x_coords - x_center) * 5
        deform_y = pressure_factor * pressure_map * np.sign(y_coords - y_center) * 3
        
        map_x = (x_coords + deform_x).astype(np.float32)
        map_y = (y_coords + deform_y).astype(np.float32)
        
        warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return warped
    
    def _calculate_deformation_map(
        self,
        original: np.ndarray,
        deformed: np.ndarray
    ) -> np.ndarray:
        """변형 맵 계산"""
        
        # 차이 계산
        diff = cv2.absdiff(original, deformed)
        
        # 그레이스케일 변환
        if len(diff.shape) == 3:
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        else:
            diff_gray = diff
        
        # 정규화
        deformation_map = diff_gray / 255.0
        
        return deformation_map
    
    async def cleanup(self):
        """리소스 정리"""
        pass


class AdvancedClothingWarper:
    """고급 의류 워핑 엔진"""
    
    def __init__(self, deformation_strength: float = 0.8, device: str = 'cpu'):
        self.deformation_strength = deformation_strength
        self.device = device
    
    def apply_advanced_warping(
        self,
        cloth_image: np.ndarray,
        deformation_map: np.ndarray,
        control_points: List[Tuple[np.ndarray, np.ndarray]],
        clothing_type: str
    ) -> Dict[str, Any]:
        """고급 워핑 적용"""
        
        # 1. 타입별 특화 워핑
        type_warped = self._apply_type_specific_warping(
            cloth_image, clothing_type
        )
        
        # 2. 변형 맵 기반 워핑
        deformation_warped = self._apply_deformation_warping(
            type_warped, deformation_map
        )
        
        # 3. 제어점 기반 정밀 워핑
        if control_points:
            final_warped = self._apply_control_point_warping(
                deformation_warped, control_points
            )
        else:
            final_warped = deformation_warped
        
        # 4. 변형 통계 계산
        deformation_stats = self._calculate_deformation_stats(
            cloth_image, final_warped
        )
        
        # 5. 스트레인 맵 생성
        strain_map = self._generate_strain_map(deformation_map, deformation_stats)
        
        return {
            'warped_image': final_warped,
            'strain_map': strain_map,
            'deformation_stats': deformation_stats
        }
    
    def _apply_type_specific_warping(
        self,
        image: np.ndarray,
        clothing_type: str
    ) -> np.ndarray:
        """의류 타입별 특화 워핑"""
        
        if clothing_type == "dress":
            # 드레스: 아래쪽으로 갈수록 더 넓어짐
            return self._apply_dress_warping(image)
        elif clothing_type == "shirt":
            # 셔츠: 어깨 부분 강조
            return self._apply_shirt_warping(image)
        elif clothing_type == "pants":
            # 바지: 다리 형태에 맞춤
            return self._apply_pants_warping(image)
        else:
            return image
    
    def _apply_dress_warping(self, image: np.ndarray) -> np.ndarray:
        """드레스 워핑"""
        h, w = image.shape[:2]
        
        # A라인 실루엣 생성
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # 아래쪽으로 갈수록 확장
        expansion_factor = (y_coords / h) * 0.1
        center_x = w // 2
        
        offset_x = (x_coords - center_x) * expansion_factor
        
        map_x = (x_coords + offset_x).astype(np.float32)
        map_y = y_coords.astype(np.float32)
        
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    def _apply_shirt_warping(self, image: np.ndarray) -> np.ndarray:
        """셔츠 워핑"""
        # 어깨 라인 강조를 위한 미세 조정
        return image  # 기본적으로는 변경 없음
    
    def _apply_pants_warping(self, image: np.ndarray) -> np.ndarray:
        """바지 워핑"""
        # 다리 부분 분리 처리
        return image  # 기본적으로는 변경 없음
    
    def _apply_deformation_warping(
        self,
        image: np.ndarray,
        deformation_map: np.ndarray
    ) -> np.ndarray:
        """변형 맵 기반 워핑"""
        
        # 변형 맵을 워핑 필드로 변환
        h, w = image.shape[:2]
        
        if deformation_map.shape != (h, w):
            deformation_map = cv2.resize(deformation_map, (w, h))
        
        # 변형 강도 적용
        deformation_scaled = deformation_map * self.deformation_strength * 10
        
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # 변형 적용 (방향성 고려)
        map_x = (x_coords + deformation_scaled * np.cos(deformation_scaled * np.pi)).astype(np.float32)
        map_y = (y_coords + deformation_scaled * np.sin(deformation_scaled * np.pi)).astype(np.float32)
        
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    def _apply_control_point_warping(
        self,
        image: np.ndarray,
        control_points: List[Tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        """제어점 기반 정밀 워핑"""
        
        if len(control_points) < 3:
            return image
        
        # RBF 기반 워핑
        source_points = np.array([pair[0] for pair in control_points])
        target_points = np.array([pair[1] for pair in control_points])
        
        # RBF 보간기 생성
        rbf_x = RBFInterpolator(source_points, target_points[:, 0], 
                               kernel='thin_plate_spline', smoothing=0.01)
        rbf_y = RBFInterpolator(source_points, target_points[:, 1], 
                               kernel='thin_plate_spline', smoothing=0.01)
        
        h, w = image.shape[:2]
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        grid_points = np.stack([x_coords.ravel(), y_coords.ravel()], axis=-1)
        
        # 변환 적용
        mapped_x = rbf_x(grid_points).reshape(h, w)
        mapped_y = rbf_y(grid_points).reshape(h, w)
        
        return cv2.remap(image, mapped_x.astype(np.float32), mapped_y.astype(np.float32), 
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    def _calculate_deformation_stats(
        self,
        original: np.ndarray,
        warped: np.ndarray
    ) -> Dict[str, float]:
        """변형 통계 계산"""
        
        # 크기 맞추기
        if original.shape != warped.shape:
            warped_resized = cv2.resize(warped, (original.shape[1], original.shape[0]))
        else:
            warped_resized = warped
        
        # 변형량 계산
        diff = cv2.absdiff(original, warped_resized)
        
        if len(diff.shape) == 3:
            diff_magnitude = np.sqrt(np.sum(diff**2, axis=2))
        else:
            diff_magnitude = diff
        
        return {
            'max_deformation': float(np.max(diff_magnitude)),
            'mean_deformation': float(np.mean(diff_magnitude)),
            'std_deformation': float(np.std(diff_magnitude)),
            'deformation_area': float(np.sum(diff_magnitude > 10) / diff_magnitude.size)
        }
    
    def _generate_strain_map(
        self,
        deformation_map: np.ndarray,
        deformation_stats: Dict[str, float]
    ) -> np.ndarray:
        """스트레인 맵 생성"""
        
        # 변형률을 색상으로 매핑
        normalized_deformation = deformation_map / (deformation_stats['max_deformation'] + 1e-7)
        
        # 컬러맵 적용 (파란색: 낮은 변형, 빨간색: 높은 변형)
        strain_colored = cv2.applyColorMap(
            (normalized_deformation * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        return strain_colored


class TextureSynthesizer:
    """텍스처 합성 및 디테일 생성"""
    
    def __init__(self, device: str = 'cpu', use_neural_synthesis: bool = False):
        self.device = device
        self.use_neural_synthesis = use_neural_synthesis
    
    def synthesize_fabric_details(
        self,
        warped_image: np.ndarray,
        strain_map: np.ndarray,
        fabric_props: Dict[str, float],
        clothing_type: str
    ) -> Dict[str, Any]:
        """천 디테일 합성"""
        
        # 1. 주름 생성
        wrinkle_map = self._generate_wrinkles(
            warped_image, strain_map, fabric_props
        )
        
        # 2. 텍스처 디테일 강화
        enhanced_texture = self._enhance_texture_details(
            warped_image, fabric_props
        )
        
        # 3. 세밀한 음영 추가
        detailed_shading = self._add_fabric_shading(
            enhanced_texture, strain_map, fabric_props
        )
        
        # 4. 최종 결합
        enhanced_image = self._combine_details(
            detailed_shading, wrinkle_map, fabric_props
        )
        
        # 5. 디테일 마스크 생성
        detail_mask = self._create_detail_mask(wrinkle_map, strain_map)
        
        return {
            'enhanced_image': enhanced_image,
            'detail_mask': detail_mask,
            'wrinkle_map': wrinkle_map,
            'synthesis_info': {
                'neural_synthesis_used': self.use_neural_synthesis,
                'wrinkles_generated': True,
                'texture_enhanced': True,
                'fabric_type': fabric_props
            }
        }
    
    def _generate_wrinkles(
        self,
        image: np.ndarray,
        strain_map: np.ndarray,
        fabric_props: Dict[str, float]
    ) -> np.ndarray:
        """주름 생성"""
        
        # 딱딱한 재질은 주름이 적음
        wrinkle_intensity = (1.0 - fabric_props['stiffness']) * 0.3
        
        if wrinkle_intensity < 0.05:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 스트레인이 높은 곳에 주름 생성
        if len(strain_map.shape) == 3:
            strain_gray = cv2.cvtColor(strain_map, cv2.COLOR_BGR2GRAY)
        else:
            strain_gray = strain_map
        
        # 높은 변형 영역 찾기
        high_strain = strain_gray > 100
        
        # 주름 패턴 생성 (가우시안 노이즈 기반)
        h, w = image.shape[:2]
        noise = np.random.normal(0, wrinkle_intensity * 50, (h, w))
        
        # 주름 방향성 추가 (수직 방향 선호)
        noise = cv2.GaussianBlur(noise, (3, 7), 0)
        
        # 변형 영역에만 적용
        wrinkles = noise * high_strain * wrinkle_intensity * 255
        wrinkles = np.clip(wrinkles, 0, 255).astype(np.uint8)
        
        return wrinkles
    
    def _enhance_texture_details(
        self,
        image: np.ndarray,
        fabric_props: Dict[str, float]
    ) -> np.ndarray:
        """텍스처 디테일 강화"""
        
        # 언샤프 마스킹으로 디테일 강화
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        unsharp_strength = fabric_props['thickness'] * 0.5
        
        enhanced = cv2.addWeighted(image, 1.0 + unsharp_strength, gaussian, -unsharp_strength, 0)
        
        return enhanced
    
    def _add_fabric_shading(
        self,
        image: np.ndarray,
        strain_map: np.ndarray,
        fabric_props: Dict[str, float]
    ) -> np.ndarray:
        """천 음영 추가"""
        
        if len(strain_map.shape) == 3:
            strain_gray = cv2.cvtColor(strain_map, cv2.COLOR_BGR2GRAY)
        else:
            strain_gray = strain_map
        
        # 변형에 따른 음영 생성
        shading_strength = fabric_props['thickness'] * 0.2
        shading = strain_gray.astype(np.float32) / 255.0 * shading_strength
        
        # 이미지에 음영 적용
        if len(image.shape) == 3:
            shading_3ch = np.stack([shading] * 3, axis=2)
            shaded = image.astype(np.float32) * (1.0 - shading_3ch) + image.astype(np.float32) * shading_3ch
        else:
            shaded = image.astype(np.float32) * (1.0 - shading) + image.astype(np.float32) * shading
        
        return np.clip(shaded, 0, 255).astype(np.uint8)
    
    def _combine_details(
        self,
        base_image: np.ndarray,
        wrinkle_map: np.ndarray,
        fabric_props: Dict[str, float]
    ) -> np.ndarray:
        """디테일 결합"""
        
        if wrinkle_map.size == 0:
            return base_image
        
        # 주름을 밝기 변화로 적용
        wrinkle_effect = wrinkle_map.astype(np.float32) / 255.0 * 0.1
        
        if len(base_image.shape) == 3:
            wrinkle_3ch = np.stack([wrinkle_effect] * 3, axis=2)
            combined = base_image.astype(np.float32) - wrinkle_3ch * 30  # 주름 부분을 어둡게
        else:
            combined = base_image.astype(np.float32) - wrinkle_effect * 30
        
        return np.clip(combined, 0, 255).astype(np.uint8)
    
    def _create_detail_mask(
        self,
        wrinkle_map: np.ndarray,
        strain_map: np.ndarray
    ) -> np.ndarray:
        """디테일 마스크 생성"""
        
        # 주름과 변형 영역 결합
        if wrinkle_map.size > 0:
            detail_mask = wrinkle_map
        else:
            detail_mask = np.zeros(strain_map.shape[:2], dtype=np.uint8)
        
        if len(strain_map.shape) == 3:
            strain_gray = cv2.cvtColor(strain_map, cv2.COLOR_BGR2GRAY)
        else:
            strain_gray = strain_map
        
        # 높은 변형 영역 추가
        high_strain_mask = (strain_gray > 50).astype(np.uint8) * 255
        
        combined_mask = cv2.bitwise_or(detail_mask, high_strain_mask)
        
        return combined_mask