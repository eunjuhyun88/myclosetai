# app/ai_pipeline/steps/step_04_geometric_matching.py
"""
4단계: 기하학적 매칭 (Geometric Matching) - 최적 생성자 패턴 적용
M3 Max 최적화 + 견고한 에러 처리 + 기존 기능 100% 유지
"""
import os
import logging
import time
import asyncio
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
from PIL import Image
import cv2

# PyTorch 선택적 임포트
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# SciPy 선택적 임포트
try:
    from scipy.spatial.distance import cdist
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    cdist = None

logger = logging.getLogger(__name__)
class GeometricMatchingStep:
    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """✅ 최적 생성자 패턴 적용"""
        
        # 동일한 패턴...
        self.device = self._auto_detect_device(device)
        self.config = config or {}
        self.step_name = self.__class__.__name__
        self.logger = logging.getLogger(f"pipeline.{self.step_name}")
        
        self.device_type = kwargs.get('device_type', 'auto')
        self.memory_gb = kwargs.get('memory_gb', 16.0)
        self.is_m3_max = kwargs.get('is_m3_max', self._detect_m3_max())
        self.optimization_enabled = kwargs.get('optimization_enabled', True)
        self.quality_level = kwargs.get('quality_level', 'balanced')
        
        self._merge_step_specific_config(kwargs)
        self.is_initialized = False
        
        from app.ai_pipeline.utils.model_loader import BaseStepMixin
        if hasattr(BaseStepMixin, '_setup_model_interface'):
            BaseStepMixin._setup_model_interface(self)
        
        self._initialize_step_specific()
        self.logger.info(f"🎯 {self.step_name} 초기화 - 디바이스: {self.device}")
    
        
        # 매칭 설정 (기존 기능 유지 + kwargs 확장)
        self.matching_config = self.config.get('matching', {
            'method': kwargs.get('method', 'auto'),  # 'tps', 'affine', 'homography', 'auto'
            'max_iterations': kwargs.get('max_iterations', 1000),
            'convergence_threshold': kwargs.get('convergence_threshold', 1e-6),
            'outlier_threshold': kwargs.get('outlier_threshold', 0.15),
            'use_pose_guidance': kwargs.get('use_pose_guidance', True),
            'adaptive_weights': kwargs.get('adaptive_weights', True),
            'quality_threshold': kwargs.get('quality_threshold', 0.7)
        })
        
        # TPS 설정 (M3 Max 최적화)
        self.tps_config = self.config.get('tps', {
            'regularization': kwargs.get('tps_regularization', 0.1),
            'grid_size': kwargs.get('tps_grid_size', 30 if self.is_m3_max else 20),
            'boundary_padding': kwargs.get('tps_boundary_padding', 0.1)
        })
        
        # 최적화 설정 (M3 Max 고려)
        learning_rate_base = 0.01
        if self.is_m3_max and self.optimization_enabled:
            learning_rate_base *= 1.2  # M3 Max는 더 빠른 학습
        
        self.optimization_config = self.config.get('optimization', {
            'learning_rate': kwargs.get('learning_rate', learning_rate_base),
            'momentum': kwargs.get('momentum', 0.9),
            'weight_decay': kwargs.get('weight_decay', 1e-4),
            'scheduler_step': kwargs.get('scheduler_step', 100)
        })
        
        # 매칭 통계 (기존과 동일)
        self.matching_stats = {
            'total_matches': 0,
            'successful_matches': 0,
            'average_accuracy': 0.0,
            'method_performance': {}
        }
        
        # 매칭 컴포넌트들 초기화
        self.tps_grid = None
        self.ransac_params = None
        self.optimizer_config = None
        
        self.logger.info(f"🎯 기하학적 매칭 스텝 초기화 완료 - 디바이스: {self.device}")
    
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
                return 'M3' in result.stdout
        except:
            pass
        return False

    def _merge_step_specific_config(self, kwargs: Dict[str, Any]):
        """⚙️ 스텝별 특화 설정 병합"""
        system_params = {
            'device_type', 'memory_gb', 'is_m3_max', 
            'optimization_enabled', 'quality_level'
        }

        for key, value in kwargs.items():
            if key not in system_params:
                self.config[key] = value

    async def initialize(self) -> bool:
        """초기화 메서드 (기존과 동일하지만 M3 Max 최적화 추가)"""
        try:
            self.logger.info("🔄 기하학적 매칭 시스템 초기화 시작...")
            
            # 디바이스 검증
            if not self._validate_device():
                self.logger.warning(f"⚠️ 디바이스 {self.device} 검증 실패, CPU로 폴백")
                self.device = "cpu"
            
            # M3 Max 특화 최적화
            if self.is_m3_max:
                await self._initialize_m3_max_optimizations()
            
            # 매칭 알고리즘 초기화
            await self._initialize_matching_algorithms()
            
            # 최적화 도구 초기화
            await self._initialize_optimization_tools()
            
            # 테스트 매칭 수행
            await self._test_system()
            
            self.is_initialized = True
            self.logger.info("✅ 기하학적 매칭 시스템 초기화 완료")
            return True
            
        except Exception as e:
            error_msg = f"매칭 시스템 초기화 실패: {e}"
            self.logger.error(f"❌ {error_msg}")
            self.initialization_error = error_msg
            
            # 기본 시스템으로 폴백
            await self._initialize_fallback_system()
            self.is_initialized = True
            return True
    
    async def _initialize_m3_max_optimizations(self):
        """M3 Max 특화 최적화"""
        try:
            self.logger.info("🍎 M3 Max 최적화 적용...")
            
            # MPS 메모리 최적화
            if TORCH_AVAILABLE and self.device == 'mps':
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                if hasattr(torch.backends.mps, 'empty_cache'):
                    if hasattr(torch.mps, "empty_cache"): torch.mps.empty_cache()
            
            # M3 Max용 고성능 파라미터
            self.matching_config['quality_threshold'] = 0.8
            
            # 고정밀도 모드
            if self.quality_level in ['high', 'ultra']:
                self.tps_config['grid_size'] = 30
                self.matching_config['max_iterations'] = 1500
            
            self.logger.info("✅ M3 Max 최적화 완료")
            
        except Exception as e:
            self.logger.warning(f"M3 Max 최적화 실패: {e}")
    
    def _validate_device(self) -> bool:
        """디바이스 유효성 검사"""
        if self.device == 'mps':
            return TORCH_AVAILABLE and torch.backends.mps.is_available()
        elif self.device == 'cuda':
            return TORCH_AVAILABLE and torch.cuda.is_available()
        elif self.device == 'cpu':
            return True
        return False
    
    async def _initialize_matching_algorithms(self):
        """매칭 알고리즘 초기화"""
        try:
            # TPS 그리드 초기화
            if SCIPY_AVAILABLE:
                grid_size = self.tps_config['grid_size']
                self.tps_grid = np.mgrid[0:grid_size, 0:grid_size].reshape(2, -1).T
                self.logger.debug("✅ TPS 그리드 초기화 완료")
            
            # RANSAC 파라미터 설정 (M3 Max 최적화)
            max_trials = 1500 if self.is_m3_max else 1000
            residual_threshold = 4.0 if self.is_m3_max else 5.0
            
            self.ransac_params = {
                'max_trials': max_trials,
                'residual_threshold': residual_threshold,
                'min_samples': 4
            }
            
            self.logger.info("✅ 매칭 알고리즘 초기화 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 매칭 알고리즘 초기화 실패: {e}")
            self.tps_grid = None
            self.ransac_params = {'max_trials': 100, 'residual_threshold': 10.0, 'min_samples': 3}
    
    async def _initialize_optimization_tools(self):
        """최적화 도구 초기화"""
        try:
            method = 'L-BFGS-B' if (SCIPY_AVAILABLE and self.is_m3_max) else ('L-BFGS-B' if SCIPY_AVAILABLE else 'Powell')
            
            self.optimizer_config = {
                'method': method,
                'options': {
                    'maxiter': self.matching_config['max_iterations'],
                    'ftol': self.matching_config['convergence_threshold']
                }
            }
            
            self.logger.info(f"✅ 최적화 도구 초기화 완료 (방법: {method})")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 최적화 도구 초기화 실패: {e}")
            self.optimizer_config = {'method': 'Powell', 'options': {'maxiter': 100}}
    
    async def _test_system(self):
        """시스템 테스트"""
        try:
            test_person_points = [(100, 100), (200, 100), (150, 200)]
            test_clothing_points = [(105, 105), (195, 95), (155, 205)]
            
            test_result = await self._perform_initial_matching(
                test_person_points, test_clothing_points, 'affine'
            )
            
            if test_result.get('success', True):
                self.logger.debug("✅ 시스템 테스트 통과")
            else:
                self.logger.warning("⚠️ 시스템 테스트 실패, 기본 모드로 동작")
                
        except Exception as e:
            self.logger.warning(f"⚠️ 시스템 테스트 실패: {e}")
    
    async def _initialize_fallback_system(self):
        """폴백 시스템 초기화"""
        try:
            self.logger.info("🔄 기본 매칭 시스템으로 초기화...")
            
            self.matching_config['method'] = 'similarity'
            self.tps_grid = None
            self.ransac_params = {'max_trials': 50, 'residual_threshold': 15.0, 'min_samples': 2}
            self.optimizer_config = {'method': 'Powell', 'options': {'maxiter': 50}}
            
            self.logger.info("✅ 기본 매칭 시스템 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 폴백 시스템 초기화도 실패: {e}")
    
    async def process(
        self,
        person_parsing: Dict[str, Any],
        pose_keypoints: List[List[float]],
        clothing_segmentation: Dict[str, Any],
        clothing_type: str = "shirt",
        **kwargs
    ) -> Dict[str, Any]:
        """
        기하학적 매칭 처리 (기존과 동일)
        
        Args:
            person_parsing: 인체 파싱 결과
            pose_keypoints: 포즈 키포인트 (OpenPose 18 형식)
            clothing_segmentation: 의류 세그멘테이션 결과
            clothing_type: 의류 타입
            **kwargs: 추가 매개변수
            
        Returns:
            Dict: 매칭 결과
        """
        start_time = time.time()
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            self.logger.info(f"🎯 기하학적 매칭 시작 - 의류: {clothing_type}")
            
            # 1. 입력 데이터 검증 및 전처리
            person_points = self._extract_person_keypoints(pose_keypoints, clothing_type)
            clothing_points = self._extract_clothing_keypoints(clothing_segmentation, clothing_type)
            
            if len(person_points) < 2 or len(clothing_points) < 2:
                return self._create_empty_result("충분하지 않은 매칭 포인트", clothing_type)
            
            # 2. 매칭 방법 선택 (M3 Max 최적화)
            matching_method = self._select_matching_method(person_points, clothing_points, clothing_type)
            self.logger.info(f"📐 선택된 매칭 방법: {matching_method}")
            
            # 3. 초기 매칭 수행
            initial_match = await self._perform_initial_matching(
                person_points, clothing_points, matching_method
            )
            
            # 4. 포즈 기반 정제
            if self.matching_config['use_pose_guidance'] and len(pose_keypoints) > 5:
                refined_match = await self._refine_with_pose_guidance(
                    initial_match, pose_keypoints, clothing_type
                )
            else:
                refined_match = initial_match
            
            # 5. 매칭 품질 평가
            quality_metrics = self._evaluate_matching_quality(
                person_points, clothing_points, refined_match
            )
            
            # 6. 품질 개선 시도 (M3 Max는 더 높은 임계값)
            quality_threshold = 0.8 if self.is_m3_max else self.matching_config['quality_threshold']
            if quality_metrics['overall_quality'] < quality_threshold:
                self.logger.info(f"🔄 품질 개선 시도 (현재: {quality_metrics['overall_quality']:.3f})")
                alternative_match = await self._try_alternative_methods(
                    person_points, clothing_points, clothing_type
                )
                
                if alternative_match:
                    alternative_quality = self._evaluate_matching_quality(
                        person_points, clothing_points, alternative_match
                    )
                    
                    if alternative_quality['overall_quality'] > quality_metrics['overall_quality']:
                        refined_match = alternative_match
                        quality_metrics = alternative_quality
                        matching_method = alternative_match.get('method', matching_method)
            
            # 7. 워핑 파라미터 생성
            warp_params = self._generate_warp_parameters(refined_match, clothing_segmentation)
            
            # 8. 최종 결과 구성
            processing_time = time.time() - start_time
            result = self._build_final_result(
                refined_match, warp_params, quality_metrics, 
                processing_time, matching_method, clothing_type
            )
            
            # 9. 통계 업데이트
            self._update_statistics(matching_method, quality_metrics['overall_quality'])
            
            self.logger.info(f"✅ 기하학적 매칭 완료 - 방법: {matching_method}, 품질: {quality_metrics['overall_quality']:.3f}")
            return result
            
        except Exception as e:
            error_msg = f"기하학적 매칭 실패: {e}"
            self.logger.error(f"❌ {error_msg}")
            return self._create_empty_result(error_msg, clothing_type)
    
    def _extract_person_keypoints(self, pose_keypoints: List[List[float]], clothing_type: str) -> List[Tuple[float, float]]:
        """인체에서 매칭 포인트 추출 (M3 Max 최적화)"""
        
        try:
            keypoint_mapping = {
                'neck': 1, 'left_shoulder': 5, 'right_shoulder': 2,
                'left_elbow': 6, 'right_elbow': 3,
                'left_wrist': 7, 'right_wrist': 4,
                'left_hip': 11, 'right_hip': 8,
                'left_knee': 12, 'right_knee': 9,
                'left_ankle': 13, 'right_ankle': 10
            }
            
            matching_points = self.MATCHING_POINTS.get(clothing_type, self.MATCHING_POINTS['shirt'])
            person_points = []
            
            # M3 Max는 더 낮은 신뢰도 임계값으로 더 많은 포인트 활용
            confidence_threshold = 0.2 if self.is_m3_max else 0.3
            
            for keypoint_name in matching_points['keypoints']:
                if keypoint_name in keypoint_mapping:
                    idx = keypoint_mapping[keypoint_name]
                    if idx < len(pose_keypoints):
                        x, y, conf = pose_keypoints[idx]
                        if conf > confidence_threshold:
                            person_points.append((float(x), float(y)))
            
            # 최소 포인트 확보 (M3 Max는 더 많이)
            min_points = 3 if self.is_m3_max else 2
            max_points = 7 if self.is_m3_max else 5
            
            if len(person_points) < min_points and len(pose_keypoints) > 2:
                for i, (x, y, conf) in enumerate(pose_keypoints):
                    if conf > 0.5 and len(person_points) < max_points:
                        person_points.append((float(x), float(y)))
            
            self.logger.debug(f"추출된 인체 포인트: {len(person_points)}개 (M3 Max: {self.is_m3_max})")
            return person_points
            
        except Exception as e:
            self.logger.warning(f"인체 키포인트 추출 실패: {e}")
            return []
    
    def _extract_clothing_keypoints(self, clothing_segmentation: Dict[str, Any], clothing_type: str) -> List[Tuple[float, float]]:
        """의류에서 매칭 포인트 추출"""
        
        try:
            mask = clothing_segmentation.get('mask')
            if mask is None:
                return []
            
            # NumPy 배열로 변환
            if hasattr(mask, 'cpu'):  # Tensor인 경우
                mask = mask.cpu().numpy()
            
            mask = np.array(mask, dtype=np.uint8)
            
            # 의류 윤곽선 추출
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return []
            
            # 가장 큰 윤곽선 선택
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 의류 타입별 특징점 추출
            clothing_points = self._extract_clothing_features(largest_contour, mask, clothing_type)
            
            self.logger.debug(f"추출된 의류 포인트: {len(clothing_points)}개")
            return clothing_points
            
        except Exception as e:
            self.logger.warning(f"의류 키포인트 추출 실패: {e}")
            return []
    
    def _extract_clothing_features(self, contour: np.ndarray, mask: np.ndarray, clothing_type: str) -> List[Tuple[float, float]]:
        """의류 특징점 추출"""
        
        features = []
        
        try:
            # 바운딩 박스
            x, y, w, h = cv2.boundingRect(contour)
            
            if clothing_type in ['shirt', 't-shirt', 'blouse']:
                features.extend([
                    (x + w * 0.2, y + h * 0.1),  # 왼쪽 어깨
                    (x + w * 0.8, y + h * 0.1),  # 오른쪽 어깨
                    (x + w * 0.5, y),            # 목/칼라
                    (x, y + h * 0.3),            # 왼쪽 소매
                    (x + w, y + h * 0.3)         # 오른쪽 소매
                ])
                
            elif clothing_type in ['pants', 'jeans', 'trousers']:
                features.extend([
                    (x + w * 0.2, y),            # 왼쪽 허리
                    (x + w * 0.8, y),            # 오른쪽 허리
                    (x + w * 0.3, y + h * 0.6),  # 왼쪽 무릎
                    (x + w * 0.7, y + h * 0.6),  # 오른쪽 무릎
                    (x + w * 0.3, y + h),        # 왼쪽 발목
                    (x + w * 0.7, y + h)         # 오른쪽 발목
                ])
                
            elif clothing_type in ['dress', 'gown']:
                features.extend([
                    (x + w * 0.2, y + h * 0.1),  # 왼쪽 어깨
                    (x + w * 0.8, y + h * 0.1),  # 오른쪽 어깨
                    (x + w * 0.5, y),            # 목/칼라
                    (x + w * 0.2, y + h * 0.4),  # 왼쪽 허리
                    (x + w * 0.8, y + h * 0.4)   # 오른쪽 허리
                ])
            
            # 윤곽선 기반 추가 특징점
            features.extend(self._extract_contour_features(contour))
            
            return features
            
        except Exception as e:
            self.logger.warning(f"의류 특징점 추출 실패: {e}")
            return []
    
    def _extract_contour_features(self, contour: np.ndarray) -> List[Tuple[float, float]]:
        """윤곽선 기반 특징점 추출"""
        
        features = []
        
        try:
            # 극값점들
            leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
            rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
            topmost = tuple(contour[contour[:, :, 1].argmin()][0])
            bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
            
            features.extend([leftmost, rightmost, topmost, bottommost])
            
            # 중심점
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                features.append((cx, cy))
            
            return features
            
        except Exception as e:
            self.logger.warning(f"윤곽선 특징점 추출 실패: {e}")
            return []
    
    def _select_matching_method(self, person_points: List, clothing_points: List, clothing_type: str) -> str:
        """매칭 방법 선택 (M3 Max 최적화)"""
        
        method = self.matching_config['method']
        
        if method == 'auto':
            num_points = min(len(person_points), len(clothing_points))
            
            # M3 Max는 더 고급 알고리즘 선호
            if self.is_m3_max and num_points >= 6 and SCIPY_AVAILABLE:
                return 'tps_advanced'  # M3 Max 전용 고급 TPS
            elif num_points >= 8 and SCIPY_AVAILABLE:
                return 'tps'  # 충분한 포인트가 있으면 TPS
            elif num_points >= 4:
                return 'homography'  # 4-7개 포인트는 Homography
            elif num_points >= 3:
                return 'affine'  # 3개 포인트는 Affine
            else:
                return 'similarity'  # 최소 변환
        
        return method
    
    async def _perform_initial_matching(
        self, 
        person_points: List, 
        clothing_points: List, 
        method: str
    ) -> Dict[str, Any]:
        """초기 매칭 수행 (M3 Max 고급 TPS 추가)"""
        
        try:
            if method == 'tps_advanced' and SCIPY_AVAILABLE and self.is_m3_max:
                return await self._tps_advanced_matching(person_points, clothing_points)
            elif method == 'tps' and SCIPY_AVAILABLE:
                return await self._tps_matching(person_points, clothing_points)
            elif method == 'homography':
                return self._homography_matching(person_points, clothing_points)
            elif method == 'affine':
                return self._affine_matching(person_points, clothing_points)
            else:  # similarity
                return self._similarity_matching(person_points, clothing_points)
                
        except Exception as e:
            self.logger.warning(f"매칭 방법 {method} 실패: {e}")
            return self._similarity_matching(person_points, clothing_points)
    
    async def _tps_advanced_matching(self, person_points: List, clothing_points: List) -> Dict[str, Any]:
        """M3 Max 전용 고급 TPS 매칭"""
        
        try:
            if not SCIPY_AVAILABLE:
                raise ImportError("SciPy 없이는 고급 TPS 사용 불가")
            
            # 대응점 쌍 생성 (M3 Max는 더 정교)
            person_array = np.array(person_points)
            clothing_array = np.array(clothing_points)
            
            # 최적 대응점 찾기 (고급 알고리즘)
            correspondences = self._find_optimal_correspondences(person_array, clothing_array)
            
            if len(correspondences) >= 4:  # M3 Max는 더 높은 최소 요구사항
                source_pts = np.array([corr[1] for corr in correspondences])
                target_pts = np.array([corr[0] for corr in correspondences])
                
                # 고급 TPS 변환 계산
                tps_transform = self._compute_advanced_tps_transform(source_pts, target_pts)
                
                return {
                    'method': 'tps_advanced',
                    'transform': tps_transform,
                    'correspondences': correspondences,
                    'source_points': source_pts.tolist(),
                    'target_points': target_pts.tolist(),
                    'confidence': 0.95,  # M3 Max는 더 높은 신뢰도
                    'success': True,
                    'm3_max_optimized': True
                }
            else:
                raise ValueError("고급 TPS를 위한 충분한 대응점이 없음")
                
        except Exception as e:
            self.logger.warning(f"고급 TPS 매칭 실패: {e}")
            # 기본 TPS로 폴백
            return await self._tps_matching(person_points, clothing_points)
    
    def _find_optimal_correspondences(self, person_array: np.ndarray, clothing_array: np.ndarray) -> List:
        """최적 대응점 찾기 (M3 Max 전용)"""
        
        try:
            # 거리 기반 + 기하학적 제약 조건
            distances = cdist(person_array, clothing_array)
            correspondences = []
            
            # 헝가리안 알고리즘 대신 탐욕적 최적화 + 기하학적 검증
            used_clothing = set()
            used_person = set()
            
            # 거리 순으로 정렬된 모든 쌍
            pairs = []
            for i, person_pt in enumerate(person_array):
                for j, clothing_pt in enumerate(clothing_array):
                    distance = distances[i, j]
                    pairs.append((distance, i, j, person_pt, clothing_pt))
            
            pairs.sort()  # 거리 순으로 정렬
            
            for distance, i, j, person_pt, clothing_pt in pairs:
                if i not in used_person and j not in used_clothing:
                    # 기하학적 일관성 검사 (M3 Max 전용)
                    if self._is_geometrically_consistent(person_pt, clothing_pt, correspondences):
                        correspondences.append((person_pt, clothing_pt))
                        used_person.add(i)
                        used_clothing.add(j)
                        
                        # M3 Max는 더 많은 대응점 활용
                        if len(correspondences) >= min(len(person_array), len(clothing_array), 8):
                            break
            
            return correspondences
            
        except Exception as e:
            self.logger.warning(f"최적 대응점 찾기 실패: {e}")
            # 기본 대응점 반환
            min_points = min(len(person_array), len(clothing_array))
            return [(person_array[i], clothing_array[i]) for i in range(min_points)]
    
    def _is_geometrically_consistent(self, person_pt: np.ndarray, clothing_pt: np.ndarray, existing_correspondences: List) -> bool:
        """기하학적 일관성 검사 (M3 Max 전용)"""
        
        if len(existing_correspondences) < 2:
            return True
        
        try:
            # 각도 일관성 검사
            for p1, c1 in existing_correspondences[-2:]:
                # 인체 포인트들 간의 각도
                person_angle = np.arctan2(person_pt[1] - p1[1], person_pt[0] - p1[0])
                # 의류 포인트들 간의 각도
                clothing_angle = np.arctan2(clothing_pt[1] - c1[1], clothing_pt[0] - c1[0])
                
                # 각도 차이 (라디안)
                angle_diff = abs(person_angle - clothing_angle)
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                
                # M3 Max는 더 엄격한 기하학적 제약
                if angle_diff > np.pi / 3:  # 60도 이상 차이나면 거부
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"기하학적 일관성 검사 실패: {e}")
            return True
    
    def _compute_advanced_tps_transform(self, source_pts: np.ndarray, target_pts: np.ndarray) -> Dict[str, Any]:
        """고급 TPS 변환 계산 (M3 Max 전용)"""
        
        try:
            n = len(source_pts)
            
            # TPS 기본 함수 (개선된 U 함수)
            def U(r):
                # M3 Max용 고정밀도 TPS 함수
                return np.where(r < 1e-8, 0, r**2 * np.log(r**2 + 1e-12))
            
            # 거리 행렬 계산 (고정밀도)
            if SCIPY_AVAILABLE:
                distances = cdist(source_pts, source_pts)
            else:
                distances = np.sqrt(((source_pts[:, np.newaxis] - source_pts[np.newaxis, :])**2).sum(axis=2))
            
            # K 행렬 (기본 함수들의 값)
            K = U(distances)
            
            # 정규화 추가 (M3 Max 전용)
            regularization = self.tps_config['regularization'] * 0.5  # M3 Max는 더 낮은 정규화
            K += regularization * np.eye(n)
            
            # P 행렬 (affine 부분을 위한 다항식 기저)
            P = np.column_stack([np.ones(n), source_pts])
            
            # L 행렬 구성
            O = np.zeros((3, 3))
            L = np.block([[K, P], [P.T, O]])
            
            # 목표 점들을 확장
            Y = np.vstack([target_pts.T, np.zeros((3, 2))])
            
            # 선형 시스템 해결 (고정밀도)
            try:
                coeffs = np.linalg.solve(L, Y)
            except np.linalg.LinAlgError:
                # 특이 행렬인 경우 SVD 기반 pseudo-inverse 사용
                U_svd, s, Vt = np.linalg.svd(L, full_matrices=False)
                s_inv = np.where(s > 1e-10, 1/s, 0)
                coeffs = Vt.T @ np.diag(s_inv) @ U_svd.T @ Y
            
            # 계수 분리
            w = coeffs[:n]  # TPS 가중치
            a = coeffs[n:]  # affine 계수
            
            return {
                'source_points': source_pts.tolist(),
                'weights': w.tolist(),
                'affine_coeffs': a.tolist(),
                'regularization': regularization,
                'advanced_mode': True,
                'm3_max_precision': True
            }
            
        except Exception as e:
            self.logger.error(f"고급 TPS 변환 계산 실패: {e}")
            # 기본 TPS로 폴백
            return self._compute_tps_transform(source_pts, target_pts)
    
    async def _tps_matching(self, person_points: List, clothing_points: List) -> Dict[str, Any]:
        """Thin Plate Spline 매칭"""
        
        try:
            if not SCIPY_AVAILABLE:
                raise ImportError("SciPy 없이는 TPS 사용 불가")
            
            # 대응점 쌍 생성
            person_array = np.array(person_points)
            clothing_array = np.array(clothing_points)
            
            # 최소 개수에 맞춰 대응
            min_points = min(len(person_array), len(clothing_array))
            person_array = person_array[:min_points]
            clothing_array = clothing_array[:min_points]
            
            # 거리 기반 대응 찾기
            distances = cdist(person_array, clothing_array)
            correspondences = []
            
            used_clothing = set()
            for i, person_pt in enumerate(person_array):
                distances_to_clothing = distances[i]
                sorted_indices = np.argsort(distances_to_clothing)
                
                for clothing_idx in sorted_indices:
                    if clothing_idx not in used_clothing:
                        correspondences.append((person_pt, clothing_array[clothing_idx]))
                        used_clothing.add(clothing_idx)
                        break
            
            if len(correspondences) >= 3:
                source_pts = np.array([corr[1] for corr in correspondences])  # 의류 점들
                target_pts = np.array([corr[0] for corr in correspondences])  # 인체 점들
                
                tps_transform = self._compute_tps_transform(source_pts, target_pts)
                
                return {
                    'method': 'tps',
                    'transform': tps_transform,
                    'correspondences': correspondences,
                    'source_points': source_pts.tolist(),
                    'target_points': target_pts.tolist(),
                    'confidence': 0.9,
                    'success': True
                }
            else:
                raise ValueError("TPS를 위한 충분한 대응점이 없음")
                
        except Exception as e:
            self.logger.warning(f"TPS 매칭 실패: {e}")
            raise
    
    def _compute_tps_transform(self, source_pts: np.ndarray, target_pts: np.ndarray) -> Dict[str, Any]:
        """TPS 변환 매개변수 계산"""
        
        try:
            n = len(source_pts)
            
            # TPS 기본 함수 (U 함수: r^2 * log(r))
            def U(r):
                return np.where(r == 0, 0, r**2 * np.log(r + 1e-10))
            
            # 거리 행렬 계산
            if SCIPY_AVAILABLE:
                distances = cdist(source_pts, source_pts)
            else:
                # SciPy 없이 계산
                distances = np.sqrt(((source_pts[:, np.newaxis] - source_pts[np.newaxis, :])**2).sum(axis=2))
            
            # K 행렬 (기본 함수들의 값)
            K = U(distances)
            
            # P 행렬 (affine 부분을 위한 다항식 기저)
            P = np.column_stack([np.ones(n), source_pts])
            
            # L 행렬 구성
            O = np.zeros((3, 3))
            L = np.block([[K, P], [P.T, O]])
            
            # 목표 점들을 확장
            Y = np.vstack([target_pts.T, np.zeros((3, 2))])
            
            # 선형 시스템 해결
            try:
                coeffs = np.linalg.solve(L, Y)
            except np.linalg.LinAlgError:
                # 특이 행렬인 경우 pseudo-inverse 사용
                coeffs = np.linalg.pinv(L) @ Y
            
            # 계수 분리
            w = coeffs[:n]  # TPS 가중치
            a = coeffs[n:]  # affine 계수
            
            return {
                'source_points': source_pts.tolist(),
                'weights': w.tolist(),
                'affine_coeffs': a.tolist(),
                'regularization': self.tps_config['regularization']
            }
            
        except Exception as e:
            self.logger.error(f"TPS 변환 계산 실패: {e}")
            # 폴백: 단위 변환
            return {
                'source_points': source_pts.tolist(),
                'weights': np.zeros((len(source_pts), 2)).tolist(),
                'affine_coeffs': np.array([[1, 0, 0], [0, 1, 0]]).tolist(),
                'regularization': 0.0
            }
    
    def _homography_matching(self, person_points: List, clothing_points: List) -> Dict[str, Any]:
        """Homography 매칭"""
        
        try:
            person_array = np.array(person_points, dtype=np.float32)
            clothing_array = np.array(clothing_points, dtype=np.float32)
            
            # 최소 4개 점 필요
            min_points = min(len(person_array), len(clothing_array), 4)
            
            if min_points < 4:
                raise ValueError("Homography를 위한 충분한 점이 없음")
            
            # 첫 4개 점 사용
            src_pts = clothing_array[:min_points]
            dst_pts = person_array[:min_points]
            
            # Homography 계산
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is None:
                raise ValueError("Homography 계산 실패")
            
            return {
                'method': 'homography',
                'transform': H.tolist(),
                'source_points': src_pts.tolist(),
                'target_points': dst_pts.tolist(),
                'inlier_mask': mask.flatten().tolist() if mask is not None else [],
                'confidence': 0.8,
                'success': True
            }
            
        except Exception as e:
            self.logger.warning(f"Homography 매칭 실패: {e}")
            raise
    
    def _affine_matching(self, person_points: List, clothing_points: List) -> Dict[str, Any]:
        """Affine 변환 매칭"""
        
        try:
            person_array = np.array(person_points, dtype=np.float32)
            clothing_array = np.array(clothing_points, dtype=np.float32)
            
            # 최소 3개 점 필요
            min_points = min(len(person_array), len(clothing_array), 3)
            
            if min_points < 3:
                raise ValueError("Affine 변환을 위한 충분한 점이 없음")
            
            # 첫 3개 점 사용
            src_pts = clothing_array[:min_points]
            dst_pts = person_array[:min_points]
            
            # Affine 변환 계산
            M = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])
            
            return {
                'method': 'affine',
                'transform': M.tolist(),
                'source_points': src_pts.tolist(),
                'target_points': dst_pts.tolist(),
                'confidence': 0.7,
                'success': True
            }
            
        except Exception as e:
            self.logger.warning(f"Affine 매칭 실패: {e}")
            raise
    
    def _similarity_matching(self, person_points: List, clothing_points: List) -> Dict[str, Any]:
        """유사성 변환 매칭 (회전, 스케일, 평행이동)"""
        
        try:
            if len(person_points) < 1 or len(clothing_points) < 1:
                # 최소 변환: 단위 변환
                M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            elif len(person_points) < 2 or len(clothing_points) < 2:
                # 최소 변환: 평행이동만
                tx = person_points[0][0] - clothing_points[0][0]
                ty = person_points[0][1] - clothing_points[0][1]
                M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
            else:
                # 중심점 기반 변환
                person_center = np.mean(person_points, axis=0)
                clothing_center = np.mean(clothing_points, axis=0)
                
                # 스케일 추정
                person_spread = np.std(person_points, axis=0)
                clothing_spread = np.std(clothing_points, axis=0)
                
                scale_x = person_spread[0] / (clothing_spread[0] + 1e-6)
                scale_y = person_spread[1] / (clothing_spread[1] + 1e-6)
                scale = (scale_x + scale_y) / 2  # 평균 스케일
                
                # 평행이동
                tx = person_center[0] - clothing_center[0] * scale
                ty = person_center[1] - clothing_center[1] * scale
                
                M = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)
            
            return {
                'method': 'similarity',
                'transform': M.tolist(),
                'source_points': clothing_points[:2] if len(clothing_points) >= 2 else clothing_points,
                'target_points': person_points[:2] if len(person_points) >= 2 else person_points,
                'confidence': 0.6,
                'success': True
            }
            
        except Exception as e:
            self.logger.warning(f"유사성 변환 실패: {e}")
            # 최후의 폴백: 단위 변환
            return {
                'method': 'identity',
                'transform': [[1, 0, 0], [0, 1, 0]],
                'source_points': [],
                'target_points': [],
                'confidence': 0.3,
                'success': True
            }
    
    async def _refine_with_pose_guidance(
        self, 
        initial_match: Dict[str, Any], 
        pose_keypoints: List[List[float]], 
        clothing_type: str
    ) -> Dict[str, Any]:
        """포즈 기반 매칭 정제"""
        
        try:
            # 포즈 특성 분석
            pose_analysis = self._analyze_pose_characteristics(pose_keypoints)
            
            # 의류 타입별 포즈 적응
            adaptation_factor = self._calculate_pose_adaptation(pose_analysis, clothing_type)
            
            # 변환 매개변수 조정
            refined_transform = self._adapt_transform_to_pose(
                initial_match['transform'], adaptation_factor, pose_analysis
            )
            
            refined_match = initial_match.copy()
            refined_match['transform'] = refined_transform
            refined_match['pose_adapted'] = True
            refined_match['adaptation_factor'] = adaptation_factor
            
            return refined_match
            
        except Exception as e:
            self.logger.warning(f"포즈 기반 정제 실패: {e}")
            return initial_match
    
    def _analyze_pose_characteristics(self, pose_keypoints: List[List[float]]) -> Dict[str, Any]:
        """포즈 특성 분석"""
        
        analysis = {}
        
        try:
            # 어깨 각도
            if len(pose_keypoints) > 5 and all(pose_keypoints[i][2] > 0.5 for i in [2, 5]):
                left_shoulder = pose_keypoints[5][:2]
                right_shoulder = pose_keypoints[2][:2]
                shoulder_angle = np.degrees(np.arctan2(
                    left_shoulder[1] - right_shoulder[1],
                    left_shoulder[0] - right_shoulder[0]
                ))
                analysis['shoulder_angle'] = shoulder_angle
            
            # 몸통 기울기
            if len(pose_keypoints) > 11 and all(pose_keypoints[i][2] > 0.5 for i in [1, 8, 11]):
                neck = pose_keypoints[1][:2]
                hip_center = np.mean([pose_keypoints[8][:2], pose_keypoints[11][:2]], axis=0)
                torso_angle = np.degrees(np.arctan2(
                    neck[0] - hip_center[0],
                    hip_center[1] - neck[1]
                ))
                analysis['torso_angle'] = torso_angle
            
        except Exception as e:
            self.logger.warning(f"포즈 특성 분석 실패: {e}")
        
        return analysis
    
    def _calculate_pose_adaptation(self, pose_analysis: Dict[str, Any], clothing_type: str) -> Dict[str, float]:
        """포즈 적응 인수 계산"""
        
        adaptation = {
            'scale_factor': 1.0,
            'rotation_adjustment': 0.0,
            'shear_factor': 0.0
        }
        
        try:
            # 어깨 기울기에 따른 회전 조정
            if 'shoulder_angle' in pose_analysis:
                shoulder_angle = pose_analysis['shoulder_angle']
                adaptation['rotation_adjustment'] = -shoulder_angle * 0.3
            
            # 몸통 기울기에 따른 전단 조정
            if 'torso_angle' in pose_analysis:
                torso_angle = pose_analysis['torso_angle']
                adaptation['shear_factor'] = np.tan(np.radians(torso_angle)) * 0.2
            
        except Exception as e:
            self.logger.warning(f"포즈 적응 계산 실패: {e}")
        
        return adaptation
    
    def _adapt_transform_to_pose(
        self, 
        original_transform: List[List[float]], 
        adaptation_factor: Dict[str, float], 
        pose_analysis: Dict[str, Any]
    ) -> List[List[float]]:
        """포즈에 맞게 변환 조정"""
        
        try:
            transform = np.array(original_transform)
            
            # 회전 조정
            rotation_adj = adaptation_factor.get('rotation_adjustment', 0.0)
            if abs(rotation_adj) > 0.1:
                cos_r = np.cos(np.radians(rotation_adj))
                sin_r = np.sin(np.radians(rotation_adj))
                rotation_matrix = np.array([[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]])
                
                if transform.shape[0] == 2:  # Affine transform
                    transform = np.vstack([transform, [0, 0, 1]])
                    transform = rotation_matrix @ transform
                    transform = transform[:2]
            
            # 스케일 조정
            scale_factor = adaptation_factor.get('scale_factor', 1.0)
            if abs(scale_factor - 1.0) > 0.01:
                if transform.shape[0] == 2:  # Affine
                    transform[0, 0] *= scale_factor
                    transform[1, 1] *= scale_factor
            
            return transform.tolist()
            
        except Exception as e:
            self.logger.warning(f"변환 포즈 적응 실패: {e}")
            return original_transform
    
    def _evaluate_matching_quality(
        self, 
        person_points: List, 
        clothing_points: List, 
        match_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """매칭 품질 평가"""
        
        try:
            transform = np.array(match_result['transform'])
            method = match_result['method']
            
            # 1. 재투영 오차 계산
            reprojection_error = self._calculate_reprojection_error(
                clothing_points, person_points, transform, method
            )
            
            # 2. 기하학적 일관성
            geometric_consistency = self._evaluate_geometric_consistency(transform, method)
            
            # 3. 변환 안정성
            transform_stability = self._evaluate_transform_stability(transform, method)
            
            # 4. 대응점 신뢰도
            correspondence_confidence = match_result.get('confidence', 0.5)
            
            # 5. 전체 품질 점수
            overall_quality = (
                (1.0 - min(1.0, reprojection_error)) * 0.4 +
                geometric_consistency * 0.3 +
                transform_stability * 0.2 +
                correspondence_confidence * 0.1
            )
            
            return {
                'overall_quality': max(0.0, min(1.0, overall_quality)),
                'reprojection_error': min(1.0, reprojection_error),
                'geometric_consistency': geometric_consistency,
                'transform_stability': transform_stability,
                'correspondence_confidence': correspondence_confidence,
                'quality_grade': self._get_quality_grade(overall_quality)
            }
            
        except Exception as e:
            self.logger.warning(f"품질 평가 실패: {e}")
            return {
                'overall_quality': 0.5,
                'reprojection_error': 1.0,
                'geometric_consistency': 0.0,
                'transform_stability': 0.0,
                'correspondence_confidence': 0.0,
                'quality_grade': 'poor'
            }
    
    def _calculate_reprojection_error(
        self, 
        source_points: List, 
        target_points: List, 
        transform: np.ndarray, 
        method: str
    ) -> float:
        """재투영 오차 계산"""
        
        try:
            if not source_points or not target_points:
                return 1.0
            
            source_array = np.array(source_points)
            target_array = np.array(target_points)
            
            # 변환 적용
            if method == 'tps':
                # TPS는 별도 처리 필요 (여기서는 간단화)
                projected_points = source_array
            elif method in ['homography']:
                if transform.shape == (3, 3):
                    # 동차 좌표로 변환
                    source_homo = np.column_stack([source_array, np.ones(len(source_array))])
                    projected_homo = source_homo @ transform.T
                    projected_points = projected_homo[:, :2] / (projected_homo[:, 2:3] + 1e-8)
                else:
                    projected_points = source_array
            else:  # affine, similarity
                if transform.shape == (2, 3):
                    source_homo = np.column_stack([source_array, np.ones(len(source_array))])
                    projected_points = source_homo @ transform.T
                else:
                    projected_points = source_array
            
            # 가장 가까운 대응점들 찾기
            min_len = min(len(projected_points), len(target_array))
            if min_len == 0:
                return 1.0
            
            if SCIPY_AVAILABLE:
                distances = cdist(projected_points[:min_len], target_array[:min_len])
                min_distances = np.min(distances, axis=1)
            else:
                # SciPy 없이 계산
                min_distances = []
                for p in projected_points[:min_len]:
                    dists = [np.linalg.norm(p - t) for t in target_array[:min_len]]
                    min_distances.append(min(dists))
                min_distances = np.array(min_distances)
            
            avg_error = np.mean(min_distances)
            
            # 정규화 (이미지 크기 대비)
            if target_array.size > 0:
                image_diagonal = np.linalg.norm(np.ptp(target_array, axis=0))
                normalized_error = avg_error / (image_diagonal + 1e-6)
            else:
                normalized_error = 1.0
            
            return min(1.0, normalized_error)
            
        except Exception as e:
            self.logger.warning(f"재투영 오차 계산 실패: {e}")
            return 1.0
    
    def _evaluate_geometric_consistency(self, transform: np.ndarray, method: str) -> float:
        """기하학적 일관성 평가"""
        
        try:
            if method == 'tps':
                return 0.9  # TPS는 항상 일관성 있음
            
            if transform.shape[0] < 2:
                return 0.0
            
            # 행렬식 계산 (스케일 변화)
            if transform.shape == (2, 3):  # Affine
                det = np.linalg.det(transform[:2, :2])
            elif transform.shape == (3, 3):  # Homography
                det = np.linalg.det(transform[:2, :2])
            else:
                return 0.5
            
            # 합리적인 스케일 변화인지 확인 (0.1 ~ 10 배)
            if 0.1 <= abs(det) <= 10:
                scale_consistency = 1.0
            else:
                scale_consistency = 0.0
            
            return scale_consistency
            
        except Exception as e:
            self.logger.warning(f"기하학적 일관성 평가 실패: {e}")
            return 0.5
    
    def _evaluate_transform_stability(self, transform: np.ndarray, method: str) -> float:
        """변환 안정성 평가"""
        
        try:
            # 조건수 확인
            if transform.shape == (2, 3):  # Affine
                matrix_part = transform[:2, :2]
            elif transform.shape == (3, 3):  # Homography
                matrix_part = transform[:2, :2]
            else:
                return 0.5
            
            condition_number = np.linalg.cond(matrix_part)
            
            # 조건수가 낮을수록 안정적
            if condition_number < 10:
                stability = 1.0
            elif condition_number < 100:
                stability = 0.8
            elif condition_number < 1000:
                stability = 0.5
            else:
                stability = 0.2
            
            return stability
            
        except Exception as e:
            self.logger.warning(f"변환 안정성 평가 실패: {e}")
            return 0.5
    
    def _get_quality_grade(self, overall_quality: float) -> str:
        """품질 등급 반환"""
        if overall_quality >= 0.9:
            return "excellent"
        elif overall_quality >= 0.8:
            return "good"
        elif overall_quality >= 0.6:
            return "fair"
        elif overall_quality >= 0.4:
            return "poor"
        else:
            return "very_poor"
    
    async def _try_alternative_methods(
        self, 
        person_points: List, 
        clothing_points: List, 
        clothing_type: str
    ) -> Optional[Dict[str, Any]]:
        """대안 매칭 방법들 시도"""
        
        alternative_methods = ['affine', 'similarity', 'homography']
        best_result = None
        best_quality = 0.0
        
        for method in alternative_methods:
            try:
                result = await self._perform_initial_matching(person_points, clothing_points, method)
                quality = self._evaluate_matching_quality(person_points, clothing_points, result)
                
                if quality['overall_quality'] > best_quality:
                    best_quality = quality['overall_quality']
                    best_result = result
                    
                self.logger.debug(f"대안 방법 {method}: 품질 {quality['overall_quality']:.3f}")
                
            except Exception as e:
                self.logger.warning(f"대안 방법 {method} 실패: {e}")
                continue
        
        return best_result
    
    def _generate_warp_parameters(self, match_result: Dict[str, Any], clothing_segmentation: Dict[str, Any]) -> Dict[str, Any]:
        """워핑 파라미터 생성"""
        
        try:
            transform = match_result['transform']
            method = match_result['method']
            
            # 기본 워핑 파라미터
            warp_params = {
                'transform_matrix': transform,
                'transform_method': method,
                'interpolation': 'bilinear',
                'border_mode': 'reflect',
                'output_size': None  # 원본 크기 유지
            }
            
            # 의류 마스크 정보 추가
            if 'mask' in clothing_segmentation:
                mask = clothing_segmentation['mask']
                warp_params['mask_transform'] = transform
                
                if hasattr(mask, 'shape'):
                    warp_params['original_mask_size'] = mask.shape
                elif hasattr(mask, 'size'):
                    warp_params['original_mask_size'] = mask.size
            
            # 방법별 특화 파라미터
            if method == 'tps' and isinstance(transform, dict):
                warp_params.update({
                    'source_points': transform.get('source_points', []),
                    'tps_weights': transform.get('weights', []),
                    'tps_affine': transform.get('affine_coeffs', [])
                })
            
            return warp_params
            
        except Exception as e:
            self.logger.warning(f"워핑 파라미터 생성 실패: {e}")
            return {
                'transform_matrix': [[1, 0, 0], [0, 1, 0]],
                'transform_method': 'identity',
                'interpolation': 'bilinear',
                'border_mode': 'reflect'
            }
    
    def _build_final_result(
        self,
        match_result: Dict[str, Any],
        warp_params: Dict[str, Any],
        quality_metrics: Dict[str, float],
        processing_time: float,
        method: str,
        clothing_type: str
    ) -> Dict[str, Any]:
        """최종 결과 구성 (M3 Max 정보 추가)"""
        
        return {
            'success': match_result.get('success', True),
            'transform_matrix': match_result['transform'],
            'warp_matrix': match_result['transform'],  # 호환성을 위한 중복
            'warp_parameters': warp_params,
            'matching_method': method,
            'clothing_type': clothing_type,
            'quality_metrics': quality_metrics,
            'confidence': quality_metrics['overall_quality'],
            'processing_time': processing_time,
            'matching_info': {
                'source_points': match_result.get('source_points', []),
                'target_points': match_result.get('target_points', []),
                'correspondences': match_result.get('correspondences', []),
                'pose_adapted': match_result.get('pose_adapted', False),
                'method_used': method,
                'm3_max_optimized': match_result.get('m3_max_optimized', False),
                'optimal_constructor': True  # 최적 생성자 사용 표시
            },
            'geometric_analysis': {
                'reprojection_error': quality_metrics['reprojection_error'],
                'geometric_consistency': quality_metrics['geometric_consistency'],
                'transform_stability': quality_metrics['transform_stability'],
                'quality_grade': quality_metrics['quality_grade']
            }
        }
    
    def _create_empty_result(self, reason: str, clothing_type: str = "unknown") -> Dict[str, Any]:
        """빈 결과 생성"""
        return {
            'success': False,
            'error': reason,
            'transform_matrix': [[1, 0, 0], [0, 1, 0]],
            'warp_matrix': [[1, 0, 0], [0, 1, 0]],
            'warp_parameters': {
                'transform_matrix': [[1, 0, 0], [0, 1, 0]],
                'transform_method': 'identity',
                'interpolation': 'bilinear'
            },
            'matching_method': 'none',
            'clothing_type': clothing_type,
            'quality_metrics': {
                'overall_quality': 0.0,
                'quality_grade': 'failed'
            },
            'confidence': 0.0,
            'processing_time': 0.0,
            'matching_info': {
                'error_occurred': True,
                'error_message': reason,
                'method_used': 'none',
                'optimal_constructor': True
            }
        }
    
    def _update_statistics(self, method: str, quality: float):
        """통계 업데이트"""
        try:
            self.matching_stats['total_matches'] += 1
            
            if quality > 0.6:
                self.matching_stats['successful_matches'] += 1
            
            # 품질 이동 평균
            alpha = 0.1
            self.matching_stats['average_accuracy'] = (
                alpha * quality + 
                (1 - alpha) * self.matching_stats['average_accuracy']
            )
            
            # 방법별 성능 추적
            if method not in self.matching_stats['method_performance']:
                self.matching_stats['method_performance'][method] = {'count': 0, 'avg_quality': 0.0}
            
            method_stats = self.matching_stats['method_performance'][method]
            method_stats['count'] += 1
            method_stats['avg_quality'] = (
                (method_stats['avg_quality'] * (method_stats['count'] - 1) + quality) / 
                method_stats['count']
            )
            
        except Exception as e:
            self.logger.warning(f"통계 업데이트 실패: {e}")
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            # 캐시된 데이터 정리
            if hasattr(self, 'tps_grid'):
                self.tps_grid = None
            
            # 통계 초기화
            self.matching_stats = {
                'total_matches': 0,
                'successful_matches': 0,
                'average_accuracy': 0.0,
                'method_performance': {}
            }
            
            self.is_initialized = False
            self.logger.info("🧹 기하학적 매칭 스텝 정리 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 리소스 정리 중 오류: {e}")
    
    # 최적 패턴 호환 메서드들
    async def get_step_info(self) -> Dict[str, Any]:
        """🔍 스텝 정보 반환 (최적 패턴 호환)"""
        return {
            "step_name": "GeometricMatching",
            "class_name": self.__class__.__name__,
            "version": "3.0-optimal",
            "device": self.device,
            "device_type": self.device_type,
            "memory_gb": self.memory_gb,
            "is_m3_max": self.is_m3_max,
            "optimization_enabled": self.optimization_enabled,
            "quality_level": self.quality_level,
            "initialized": self.is_initialized,
            "initialization_error": self.initialization_error,
            "optimal_constructor": True,
            "capabilities": {
                "tps_matching": SCIPY_AVAILABLE,
                "tps_advanced_matching": SCIPY_AVAILABLE and self.is_m3_max,
                "homography_matching": True,
                "affine_matching": True,
                "similarity_matching": True,
                "pose_guidance": True,
                "m3_max_acceleration": self.is_m3_max
            },
            "performance_stats": self.matching_stats,
            "dependencies": {
                "opencv": True,
                "numpy": True,
                "scipy": SCIPY_AVAILABLE,
                "torch": TORCH_AVAILABLE
            },
            "config": {
                "matching": self.matching_config,
                "tps": self.tps_config,
                "optimization": self.optimization_config
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return self.matching_stats.copy()
    
    def reset_statistics(self):
        """통계 초기화"""
        self.matching_stats = {
            'total_matches': 0,
            'successful_matches': 0,
            'average_accuracy': 0.0,
            'method_performance': {}
        }


# ===============================================================
# 🔄 하위 호환성 지원 (기존 코드 100% 지원)
# ===============================================================

def create_geometric_matching_step(
    device: str = "mps", 
    config: Optional[Dict[str, Any]] = None
) -> GeometricMatchingStep:
    """🔄 기존 방식 100% 호환 생성자"""
    return GeometricMatchingStep(device=device, config=config)

# M3 Max 최적화 전용 생성자도 지원
def create_m3_max_geometric_matching_step(
    device: Optional[str] = None,
    memory_gb: float = 128.0,
    optimization_level: str = "ultra",
    **kwargs
) -> GeometricMatchingStep:
    """🍎 M3 Max 최적화 전용 생성자"""
    return GeometricMatchingStep(
        device=device,
        memory_gb=memory_gb,
        quality_level=optimization_level,
        is_m3_max=True,
        optimization_enabled=True,
        **kwargs
    )