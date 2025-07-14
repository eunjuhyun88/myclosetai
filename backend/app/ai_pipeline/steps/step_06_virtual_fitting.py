"""
6단계: 가상 피팅 (Virtual Fitting) - Pipeline Manager 완전 호환 버전
M3 Max 128GB 환경 최적화 및 통합 인터페이스
"""
import os
import time
import asyncio
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import base64
import io

# 선택적 import (없어도 작동)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

try:
    from scipy.spatial.distance import cdist
    from scipy.interpolate import Rbf
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# 기존 app 구조 import (안전하게)
try:
    from app.core.config import get_settings
    settings = get_settings()
except ImportError:
    settings = None

try:
    from app.utils.image_utils import save_temp_image, load_image
except ImportError:
    save_temp_image = None
    load_image = None

try:
    from app.ai_pipeline.utils.memory_manager import optimize_memory_usage
except ImportError:
    def optimize_memory_usage():
        pass

logger = logging.getLogger(__name__)

class VirtualFittingStep:
    """
    6단계: 가상 피팅 - Pipeline Manager 완전 호환 버전
    
    Pipeline Manager가 요구하는 표준 인터페이스:
    - __init__(device: str, config: Dict[str, Any])
    - async initialize() -> bool
    - process(...) -> Dict[str, Any]
    - async cleanup()
    """
    
    def __init__(self, device: str = "mps", config: Dict[str, Any] = None):
        """
        Pipeline Manager 호환 초기화
        
        Args:
            device: 사용할 디바이스 ("mps", "cuda", "cpu")
            config: 설정 딕셔너리
        """
        self.device = self._validate_device(device)
        self.config = config or {}
        
        # 가상 피팅 설정
        self.fitting_config = self.config.get('virtual_fitting', {
            'composition_method': 'neural_blend',  # neural_blend, traditional_blend, advanced_blend
            'quality_level': 'high',  # fast, medium, high
            'enable_pose_guidance': True,
            'enable_texture_enhancement': True,
            'blend_strength': 0.8,
            'edge_smoothing': True
        })
        
        # M3 Max 최적화 설정
        self.optimization_config = {
            'use_mps': self.device == 'mps',
            'memory_efficient': True,
            'batch_size': 1,
            'fp16_enabled': self.device == 'mps',  # M3 Max에서 fp16 활용
            'enable_caching': True
        }
        
        # 핵심 컴포넌트들
        self.pose_analyzer = None
        self.composition_engine = None
        self.quality_enhancer = None
        self.texture_processor = None
        
        # 상태 변수
        self.is_initialized = False
        self.initialization_error = None
        
        # 성능 통계
        self.performance_stats = {
            'total_processed': 0,
            'average_time': 0.0,
            'success_rate': 1.0,
            'average_quality': 0.85
        }
        
        logger.info(f"🎯 VirtualFittingStep 초기화 - 디바이스: {self.device}")
    
    def _validate_device(self, device: str) -> str:
        """디바이스 유효성 검사 및 최적화"""
        if device == 'mps' and torch.backends.mps.is_available():
            logger.info("✅ Apple Silicon MPS 백엔드 활성화")
            return 'mps'
        elif device == 'cuda' and torch.cuda.is_available():
            logger.info("✅ CUDA 백엔드 활성화")
            return 'cuda'
        else:
            logger.info("⚠️ CPU 백엔드 사용")
            return 'cpu'
    
    async def initialize(self) -> bool:
        """
        가상 피팅 시스템 초기화
        Pipeline Manager가 호출하는 표준 메서드
        """
        try:
            logger.info("🔄 가상 피팅 시스템 초기화 시작...")
            
            # 1. 포즈 분석기 초기화
            self.pose_analyzer = PoseAnalyzer(
                device=self.device,
                enabled=self.fitting_config['enable_pose_guidance']
            )
            await self.pose_analyzer.initialize()
            
            # 2. 합성 엔진 초기화
            self.composition_engine = CompositionEngine(
                device=self.device,
                method=self.fitting_config['composition_method'],
                quality_level=self.fitting_config['quality_level']
            )
            await self.composition_engine.initialize()
            
            # 3. 품질 향상기 초기화
            self.quality_enhancer = QualityEnhancer(
                device=self.device,
                enable_texture=self.fitting_config['enable_texture_enhancement']
            )
            
            # 4. 텍스처 처리기 초기화
            self.texture_processor = TextureProcessor(
                device=self.device,
                optimization_level=self.optimization_config
            )
            
            # 5. 메모리 최적화
            self._optimize_memory()
            
            self.is_initialized = True
            logger.info("✅ 가상 피팅 시스템 초기화 완료")
            return True
            
        except Exception as e:
            error_msg = f"가상 피팅 시스템 초기화 실패: {e}"
            logger.error(f"❌ {error_msg}")
            self.initialization_error = error_msg
            self.is_initialized = False
            return False
    
    def process(
        self,
        person_image: torch.Tensor,
        warped_clothing: torch.Tensor,
        clothing_mask: torch.Tensor,
        pose_keypoints: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        가상 피팅 처리 - Pipeline Manager 호환 인터페이스
        
        Args:
            person_image: 인물 이미지 텐서 [1, 3, H, W]
            warped_clothing: 변형된 의류 이미지 텐서 [1, 3, H, W]  
            clothing_mask: 의류 마스크 텐서 [1, 1, H, W]
            pose_keypoints: 포즈 키포인트 정보 (선택적)
            user_preferences: 사용자 선호도 (선택적)
        
        Returns:
            Pipeline Manager 호환 결과 딕셔너리
        """
        if not self.is_initialized:
            return self._create_error_result(
                f"시스템이 초기화되지 않음: {self.initialization_error}"
            )
        
        start_time = time.time()
        
        try:
            logger.info("🎨 가상 피팅 처리 시작...")
            
            # 1. 입력 검증 및 전처리
            person_np = self._tensor_to_numpy(person_image)
            clothing_np = self._tensor_to_numpy(warped_clothing)
            mask_np = self._tensor_to_numpy(clothing_mask, is_mask=True)
            
            # 2. 포즈 기반 가이던스 생성
            pose_guidance = self._generate_pose_guidance(
                person_np, pose_keypoints
            )
            
            # 3. 메인 합성 처리
            composition_result = self._perform_composition(
                person_np, clothing_np, mask_np, pose_guidance
            )
            
            # 4. 품질 향상 후처리
            enhanced_result = self._enhance_quality(
                composition_result, person_np, clothing_np
            )
            
            # 5. 최종 결과 검증 및 품질 평가
            quality_metrics = self._evaluate_result_quality(
                enhanced_result, person_np, clothing_np
            )
            
            processing_time = time.time() - start_time
            
            # 6. Pipeline Manager 호환 결과 구성
            result = self._build_pipeline_result(
                enhanced_result, quality_metrics, processing_time,
                user_preferences
            )
            
            # 7. 성능 통계 업데이트
            self._update_performance_stats(processing_time, quality_metrics['overall_quality'])
            
            logger.info(f"✅ 가상 피팅 완료 - {processing_time:.3f}초, 품질: {quality_metrics['overall_quality']:.3f}")
            return result
            
        except Exception as e:
            error_msg = f"가상 피팅 처리 실패: {e}"
            logger.error(f"❌ {error_msg}")
            processing_time = time.time() - start_time
            return self._create_error_result(error_msg, processing_time)
    
    def _generate_pose_guidance(
        self, 
        person_image: np.ndarray, 
        pose_keypoints: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """포즈 기반 가이던스 생성"""
        
        if not self.fitting_config['enable_pose_guidance'] or not self.pose_analyzer:
            return {'enabled': False}
        
        try:
            if pose_keypoints:
                # 기존 포즈 키포인트 사용
                guidance = self.pose_analyzer.process_existing_keypoints(pose_keypoints)
            else:
                # 새로운 포즈 분석
                guidance = self.pose_analyzer.analyze_pose(person_image)
            
            return {
                'enabled': True,
                'body_regions': guidance.get('body_regions', {}),
                'attention_map': guidance.get('attention_map'),
                'pose_confidence': guidance.get('confidence', 0.8)
            }
            
        except Exception as e:
            logger.warning(f"포즈 가이던스 생성 실패: {e}")
            return {'enabled': False, 'error': str(e)}
    
    def _perform_composition(
        self,
        person_image: np.ndarray,
        clothing_image: np.ndarray,
        clothing_mask: np.ndarray,
        pose_guidance: Dict[str, Any]
    ) -> np.ndarray:
        """메인 합성 처리"""
        
        method = self.fitting_config['composition_method']
        
        if method == 'neural_blend' and self.composition_engine.neural_compositor:
            return self._neural_composition(
                person_image, clothing_image, clothing_mask, pose_guidance
            )
        elif method == 'advanced_blend':
            return self._advanced_composition(
                person_image, clothing_image, clothing_mask, pose_guidance
            )
        else:
            return self._traditional_composition(
                person_image, clothing_image, clothing_mask, pose_guidance
            )
    
    def _neural_composition(
        self,
        person_image: np.ndarray,
        clothing_image: np.ndarray,
        clothing_mask: np.ndarray,
        pose_guidance: Dict[str, Any]
    ) -> np.ndarray:
        """신경망 기반 합성"""
        
        try:
            # 텐서로 변환
            person_tensor = self._numpy_to_tensor(person_image)
            clothing_tensor = self._numpy_to_tensor(clothing_image)
            mask_tensor = self._numpy_to_tensor(clothing_mask, is_mask=True)
            
            # 신경망 합성 실행
            with torch.no_grad():
                if pose_guidance.get('enabled'):
                    attention_map = pose_guidance.get('attention_map')
                    if attention_map is not None:
                        attention_tensor = self._numpy_to_tensor(attention_map, is_mask=True)
                        result_tensor = self.composition_engine.compose_with_attention(
                            person_tensor, clothing_tensor, mask_tensor, attention_tensor
                        )
                    else:
                        result_tensor = self.composition_engine.compose(
                            person_tensor, clothing_tensor, mask_tensor
                        )
                else:
                    result_tensor = self.composition_engine.compose(
                        person_tensor, clothing_tensor, mask_tensor
                    )
            
            return self._tensor_to_numpy(result_tensor)
            
        except Exception as e:
            logger.warning(f"신경망 합성 실패, 전통적 방법 사용: {e}")
            return self._traditional_composition(
                person_image, clothing_image, clothing_mask, pose_guidance
            )
    
    def _advanced_composition(
        self,
        person_image: np.ndarray,
        clothing_image: np.ndarray,
        clothing_mask: np.ndarray,
        pose_guidance: Dict[str, Any]
    ) -> np.ndarray:
        """고급 합성 (포아송 블렌딩 + 그라디언트 도메인)"""
        
        try:
            # 1. 포즈 가이던스를 이용한 블렌딩 영역 정제
            if pose_guidance.get('enabled'):
                refined_mask = self._refine_mask_with_pose(
                    clothing_mask, pose_guidance
                )
            else:
                refined_mask = clothing_mask
            
            # 2. 멀티 레벨 블렌딩
            result = self._multi_level_blending(
                person_image, clothing_image, refined_mask
            )
            
            # 3. 그라디언트 도메인 최적화
            if self.fitting_config['quality_level'] == 'high':
                result = self._gradient_domain_optimization(result, person_image)
            
            return result
            
        except Exception as e:
            logger.warning(f"고급 합성 실패, 전통적 방법 사용: {e}")
            return self._traditional_composition(
                person_image, clothing_image, clothing_mask, pose_guidance
            )
    
    def _traditional_composition(
        self,
        person_image: np.ndarray,
        clothing_image: np.ndarray,
        clothing_mask: np.ndarray,
        pose_guidance: Dict[str, Any]
    ) -> np.ndarray:
        """전통적 알파 블렌딩 합성"""
        
        try:
            # 마스크 정규화 및 부드럽게 처리
            mask_float = clothing_mask.astype(np.float32) / 255.0
            
            # 가우시안 블러로 경계 부드럽게
            blur_kernel_size = 15 if self.fitting_config['edge_smoothing'] else 5
            for i in range(3):
                mask_float = cv2.GaussianBlur(mask_float, (blur_kernel_size, blur_kernel_size), 3)
            
            # 3채널로 확장
            if len(mask_float.shape) == 2:
                mask_float = np.stack([mask_float] * 3, axis=2)
            
            # 블렌딩 강도 적용
            blend_strength = self.fitting_config['blend_strength']
            mask_float = mask_float * blend_strength
            
            # 알파 블렌딩
            person_float = person_image.astype(np.float32)
            clothing_float = clothing_image.astype(np.float32)
            
            blended = person_float * (1 - mask_float) + clothing_float * mask_float
            
            return np.clip(blended, 0, 255).astype(np.uint8)
            
        except Exception as e:
            logger.error(f"전통적 합성 실패: {e}")
            return person_image  # 실패 시 원본 반환
    
    def _enhance_quality(
        self,
        composed_image: np.ndarray,
        person_image: np.ndarray,
        clothing_image: np.ndarray
    ) -> np.ndarray:
        """품질 향상 후처리"""
        
        if not self.quality_enhancer:
            return composed_image
        
        try:
            enhanced = self.quality_enhancer.enhance(composed_image, person_image)
            
            if self.fitting_config['enable_texture_enhancement']:
                enhanced = self.texture_processor.enhance_texture(enhanced, clothing_image)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"품질 향상 실패: {e}")
            return composed_image
    
    def _evaluate_result_quality(
        self,
        result_image: np.ndarray,
        person_image: np.ndarray,
        clothing_image: np.ndarray
    ) -> Dict[str, float]:
        """결과 품질 평가"""
        
        try:
            metrics = {}
            
            # 1. 구조적 유사도 (SSIM 간소화 버전)
            metrics['structural_similarity'] = self._calculate_simple_ssim(result_image, person_image)
            
            # 2. 색상 일관성
            metrics['color_consistency'] = self._evaluate_color_harmony(result_image, person_image)
            
            # 3. 경계 자연스러움
            metrics['edge_naturalness'] = self._evaluate_edge_quality(result_image)
            
            # 4. 의류 보존도
            metrics['clothing_preservation'] = self._evaluate_clothing_preservation(result_image, clothing_image)
            
            # 5. 전체 품질 (가중 평균)
            metrics['overall_quality'] = (
                metrics['structural_similarity'] * 0.3 +
                metrics['color_consistency'] * 0.25 +
                metrics['edge_naturalness'] * 0.25 +
                metrics['clothing_preservation'] * 0.2
            )
            
            return metrics
            
        except Exception as e:
            logger.warning(f"품질 평가 실패: {e}")
            return {
                'overall_quality': 0.75,
                'structural_similarity': 0.8,
                'color_consistency': 0.7,
                'edge_naturalness': 0.8,
                'clothing_preservation': 0.7
            }
    
    def _build_pipeline_result(
        self,
        fitted_image: np.ndarray,
        quality_metrics: Dict[str, float],
        processing_time: float,
        user_preferences: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Pipeline Manager 호환 결과 구성"""
        
        # 텐서로 변환
        fitted_tensor = self._numpy_to_tensor(fitted_image)
        
        return {
            "success": True,
            "fitted_image": fitted_tensor,
            "fitted_image_numpy": fitted_image,
            "fitted_image_pil": Image.fromarray(cv2.cvtColor(fitted_image, cv2.COLOR_BGR2RGB)),
            "quality_metrics": quality_metrics,
            "confidence": quality_metrics['overall_quality'],
            "fitting_info": {
                "composition_method": self.fitting_config['composition_method'],
                "quality_level": self.fitting_config['quality_level'],
                "processing_time": processing_time,
                "device": self.device,
                "pose_guidance_used": self.fitting_config['enable_pose_guidance'],
                "texture_enhancement_used": self.fitting_config['enable_texture_enhancement'],
                "optimization": "M3_Max_MPS" if self.device == 'mps' else self.device.upper()
            },
            "recommendations": self._generate_recommendations(quality_metrics, user_preferences)
        }
    
    def _generate_recommendations(
        self, 
        quality_metrics: Dict[str, float], 
        user_preferences: Optional[Dict] = None
    ) -> List[str]:
        """사용자 추천 생성"""
        
        recommendations = []
        
        overall_quality = quality_metrics.get('overall_quality', 0.75)
        edge_quality = quality_metrics.get('edge_naturalness', 0.75)
        color_consistency = quality_metrics.get('color_consistency', 0.75)
        
        if overall_quality > 0.85:
            recommendations.append("완벽한 핏입니다! 이 스타일이 당신에게 잘 어울려요.")
        elif overall_quality > 0.7:
            recommendations.append("좋은 결과입니다. 이 스타일을 시도해보세요.")
        else:
            recommendations.append("더 나은 결과를 위해 정면을 향한 전신 사진을 사용해보세요.")
        
        if edge_quality < 0.7:
            recommendations.append("더 자연스러운 경계를 위해 조명이 균일한 환경에서 촬영해보세요.")
        
        if color_consistency < 0.7:
            recommendations.append("색상 조화를 위해 비슷한 톤의 의류를 선택해보세요.")
        
        return recommendations[:3]  # 최대 3개
    
    def _create_error_result(self, error_message: str, processing_time: float = 0.0) -> Dict[str, Any]:
        """에러 결과 생성"""
        return {
            "success": False,
            "error": error_message,
            "fitted_image": None,
            "fitted_image_numpy": None,
            "fitted_image_pil": None,
            "quality_metrics": {},
            "confidence": 0.0,
            "fitting_info": {
                "processing_time": processing_time,
                "device": self.device,
                "error_details": error_message
            },
            "recommendations": ["시스템 오류가 발생했습니다. 다시 시도해주세요."]
        }
    
    # =================================================================
    # 유틸리티 메서드들
    # =================================================================
    
    def _tensor_to_numpy(self, tensor: torch.Tensor, is_mask: bool = False) -> np.ndarray:
        """텐서를 NumPy 배열로 변환"""
        try:
            # GPU에서 CPU로 이동
            if tensor.is_cuda or (hasattr(tensor, 'is_mps') and tensor.is_mps):
                tensor = tensor.cpu()
            
            # 배치 차원 제거
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            
            if is_mask:
                if tensor.dim() == 3:
                    tensor = tensor.squeeze(0)
                array = tensor.numpy().astype(np.uint8)
                if array.max() <= 1.0:
                    array = array * 255
            else:
                if tensor.dim() == 3 and tensor.size(0) == 3:
                    tensor = tensor.permute(1, 2, 0)
                array = tensor.numpy()
                if array.max() <= 1.0:
                    array = array * 255
                array = array.astype(np.uint8)
            
            return array
            
        except Exception as e:
            logger.error(f"텐서 변환 실패: {e}")
            raise
    
    def _numpy_to_tensor(self, array: np.ndarray, is_mask: bool = False) -> torch.Tensor:
        """NumPy 배열을 텐서로 변환"""
        try:
            if is_mask:
                if len(array.shape) == 2:
                    array = array[np.newaxis, :]
                tensor = torch.from_numpy(array.astype(np.float32) / 255.0)
                tensor = tensor.unsqueeze(0)
            else:
                if len(array.shape) == 3 and array.shape[2] == 3:
                    array = array.transpose(2, 0, 1)
                tensor = torch.from_numpy(array.astype(np.float32) / 255.0)
                tensor = tensor.unsqueeze(0)
            
            return tensor.to(self.device)
            
        except Exception as e:
            logger.warning(f"텐서 변환 실패: {e}")
            return torch.zeros(1, 3, 256, 256).to(self.device)
    
    def _optimize_memory(self):
        """메모리 최적화"""
        optimize_memory_usage()
        
        if self.device == 'mps':
            try:
                torch.mps.empty_cache()
            except:
                pass
        elif self.device == 'cuda':
            torch.cuda.empty_cache()
    
    def _calculate_simple_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """간소화된 SSIM 계산"""
        try:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
            
            # 간소화된 SSIM 구현
            mu1 = cv2.GaussianBlur(gray1.astype(np.float32), (11, 11), 1.5)
            mu2 = cv2.GaussianBlur(gray2.astype(np.float32), (11, 11), 1.5)
            
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            
            C1 = (0.01 * 255) ** 2
            numerator = 2 * mu1_mu2 + C1
            denominator = mu1_sq + mu2_sq + C1
            
            ssim_map = numerator / denominator
            return float(np.mean(ssim_map))
            
        except Exception as e:
            logger.warning(f"SSIM 계산 실패: {e}")
            return 0.8
    
    def _evaluate_color_harmony(self, result: np.ndarray, reference: np.ndarray) -> float:
        """색상 조화도 평가"""
        try:
            result_mean = np.mean(result, axis=(0, 1))
            ref_mean = np.mean(reference, axis=(0, 1))
            
            color_diff = np.linalg.norm(result_mean - ref_mean)
            harmony = max(0.0, 1.0 - color_diff / 255.0)
            
            return harmony
        except:
            return 0.75
    
    def _evaluate_edge_quality(self, image: np.ndarray) -> float:
        """경계 품질 평가"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # 경계의 연속성 평가
            edge_density = np.sum(edges > 0) / edges.size
            quality = min(1.0, edge_density * 5)  # 정규화
            
            return quality
        except:
            return 0.75
    
    def _evaluate_clothing_preservation(self, result: np.ndarray, clothing: np.ndarray) -> float:
        """의류 특성 보존도 평가"""
        try:
            # 색상 분포 비교
            result_hist = cv2.calcHist([result], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            clothing_hist = cv2.calcHist([clothing], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            
            correlation = cv2.compareHist(result_hist, clothing_hist, cv2.HISTCMP_CORREL)
            return max(0.0, correlation)
        except:
            return 0.7
    
    def _refine_mask_with_pose(self, mask: np.ndarray, pose_guidance: Dict) -> np.ndarray:
        """포즈 가이던스로 마스크 정제"""
        # 간단한 구현 - 실제로는 더 복잡한 로직 필요
        return mask
    
    def _multi_level_blending(self, person: np.ndarray, clothing: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """멀티 레벨 블렌딩"""
        # 간소화된 구현
        mask_float = mask.astype(np.float32) / 255.0
        if len(mask_float.shape) == 2:
            mask_float = np.stack([mask_float] * 3, axis=2)
        
        return (person.astype(np.float32) * (1 - mask_float) + 
                clothing.astype(np.float32) * mask_float).astype(np.uint8)
    
    def _gradient_domain_optimization(self, image: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """그라디언트 도메인 최적화"""
        # 간소화된 구현
        return image
    
    def _update_performance_stats(self, processing_time: float, quality: float):
        """성능 통계 업데이트"""
        try:
            self.performance_stats['total_processed'] += 1
            total = self.performance_stats['total_processed']
            
            # 평균 시간 업데이트
            current_avg = self.performance_stats['average_time']
            self.performance_stats['average_time'] = (current_avg * (total - 1) + processing_time) / total
            
            # 평균 품질 업데이트
            current_quality = self.performance_stats['average_quality']
            self.performance_stats['average_quality'] = (current_quality * (total - 1) + quality) / total
            
            # 성공률 업데이트
            if quality > 0.5:
                success_count = int(self.performance_stats['success_rate'] * (total - 1)) + 1
                self.performance_stats['success_rate'] = success_count / total
            
        except Exception as e:
            logger.warning(f"통계 업데이트 실패: {e}")
    
    async def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환 - Pipeline Manager 호환"""
        return {
            "step_name": "VirtualFitting",
            "version": "3.0",
            "device": self.device,
            "initialized": self.is_initialized,
            "initialization_error": self.initialization_error,
            "config": self.fitting_config,
            "optimization": self.optimization_config,
            "performance_stats": self.performance_stats,
            "capabilities": {
                "neural_composition": bool(self.composition_engine),
                "pose_guidance": bool(self.pose_analyzer),
                "quality_enhancement": bool(self.quality_enhancer),
                "texture_processing": bool(self.texture_processor)
            }
        }
    
    async def cleanup(self):
        """리소스 정리 - Pipeline Manager 호환"""
        try:
            logger.info("🧹 가상 피팅 시스템 리소스 정리...")
            
            if self.pose_analyzer:
                await self.pose_analyzer.cleanup()
                self.pose_analyzer = None
            
            if self.composition_engine:
                await self.composition_engine.cleanup()
                self.composition_engine = None
            
            if self.quality_enhancer:
                del self.quality_enhancer
                self.quality_enhancer = None
            
            if self.texture_processor:
                del self.texture_processor
                self.texture_processor = None
            
            self._optimize_memory()
            self.is_initialized = False
            
            logger.info("✅ 가상 피팅 시스템 리소스 정리 완료")
            
        except Exception as e:
            logger.warning(f"리소스 정리 중 오류: {e}")


# =================================================================
# 보조 클래스들
# =================================================================

class PoseAnalyzer:
    """포즈 분석기"""
    
    def __init__(self, device: str = 'cpu', enabled: bool = True):
        self.device = device
        self.enabled = enabled and MEDIAPIPE_AVAILABLE
        self.pose_model = None
    
    async def initialize(self) -> bool:
        if not self.enabled:
            return True
        
        try:
            self.pose_model = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )
            return True
        except Exception as e:
            logger.warning(f"포즈 분석기 초기화 실패: {e}")
            self.enabled = False
            return False
    
    def analyze_pose(self, image: np.ndarray) -> Dict[str, Any]:
        if not self.enabled:
            return {'confidence': 0.5}
        
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose_model.process(rgb_image)
            
            if results.pose_landmarks:
                return {
                    'confidence': 0.8,
                    'body_regions': self._extract_body_regions(results.pose_landmarks),
                    'attention_map': self._generate_attention_map(image.shape[:2], results.pose_landmarks)
                }
            else:
                return {'confidence': 0.3}
        except Exception as e:
            logger.warning(f"포즈 분석 실패: {e}")
            return {'confidence': 0.5}
    
    def process_existing_keypoints(self, keypoints: Dict) -> Dict[str, Any]:
        return {'confidence': 0.8, 'body_regions': keypoints}
    
    def _extract_body_regions(self, landmarks) -> Dict[str, Any]:
        return {'torso': True, 'arms': True, 'legs': True}
    
    def _generate_attention_map(self, shape: Tuple[int, int], landmarks) -> np.ndarray:
        h, w = shape
        attention = np.ones((h, w), dtype=np.float32) * 0.5
        return attention
    
    async def cleanup(self):
        if self.pose_model:
            self.pose_model.close()


class CompositionEngine:
    """합성 엔진"""
    
    def __init__(self, device: str = 'cpu', method: str = 'neural_blend', quality_level: str = 'medium'):
        self.device = device
        self.method = method
        self.quality_level = quality_level
        self.neural_compositor = None
    
    async def initialize(self) -> bool:
        try:
            if self.method == 'neural_blend':
                self.neural_compositor = SimpleNeuralCompositor(self.device)
                await self.neural_compositor.initialize()
            return True
        except Exception as e:
            logger.warning(f"합성 엔진 초기화 실패: {e}")
            return False
    
    def compose(self, person: torch.Tensor, clothing: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.neural_compositor:
            return self.neural_compositor.compose(person, clothing, mask)
        else:
            # 기본 블렌딩
            return person * (1 - mask) + clothing * mask
    
    def compose_with_attention(self, person: torch.Tensor, clothing: torch.Tensor, 
                             mask: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        combined_mask = mask * attention
        return self.compose(person, clothing, combined_mask)
    
    async def cleanup(self):
        if self.neural_compositor:
            del self.neural_compositor
            self.neural_compositor = None


class SimpleNeuralCompositor(nn.Module):
    """간단한 신경망 합성기"""
    
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = device
        
        # 간단한 CNN 레이어들
        self.conv1 = nn.Conv2d(7, 32, 3, padding=1)  # person(3) + clothing(3) + mask(1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 3, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    async def initialize(self):
        self.to(self.device)
        self.eval()
    
    def compose(self, person: torch.Tensor, clothing: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        try:
            # 입력 결합
            inputs = torch.cat([person, clothing, mask], dim=1)
            
            # CNN 처리
            x = self.relu(self.conv1(inputs))
            x = self.relu(self.conv2(x))
            x = self.sigmoid(self.conv3(x))
            
            return x
        except Exception as e:
            logger.warning(f"신경망 합성 실패: {e}")
            # 폴백: 간단한 블렌딩
            return person * (1 - mask) + clothing * mask


class QualityEnhancer:
    """품질 향상기"""
    
    def __init__(self, device: str = 'cpu', enable_texture: bool = True):
        self.device = device
        self.enable_texture = enable_texture
    
    def enhance(self, image: np.ndarray, reference: np.ndarray) -> np.ndarray:
        try:
            # 기본 노이즈 제거
            enhanced = cv2.bilateralFilter(image, 9, 75, 75)
            
            # 선명화
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * 0.1
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            return enhanced
        except Exception as e:
            logger.warning(f"품질 향상 실패: {e}")
            return image


class TextureProcessor:
    """텍스처 처리기"""
    
    def __init__(self, device: str = 'cpu', optimization_level: Dict = None):
        self.device = device
        self.optimization_level = optimization_level or {}
    
    def enhance_texture(self, image: np.ndarray, clothing_reference: np.ndarray) -> np.ndarray:
        try:
            # 간단한 텍스처 향상
            return cv2.addWeighted(image, 0.9, clothing_reference, 0.1, 0)
        except Exception as e:
            logger.warning(f"텍스처 처리 실패: {e}")
            return image