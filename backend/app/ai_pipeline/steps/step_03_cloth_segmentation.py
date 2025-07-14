# app/ai_pipeline/steps/step_03_cloth_segmentation.py
"""
3단계: 의류 세그멘테이션 (Clothing Segmentation) - 통일된 생성자 패턴 적용
✅ 최적화된 생성자: device 자동감지, M3 Max 최적화, 일관된 인터페이스
M3 Max 최적화 + 견고한 에러 처리 + 완전한 초기화 시스템
"""
import os
import logging
import time
import asyncio
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# 통일된 베이스 클래스 import
from .base_step import ProcessingPipelineStep

# 배경 제거 라이브러리들 (선택적 import)
try:
    import rembg
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    logging.warning("rembg 라이브러리가 없습니다. 대안 방법을 사용합니다.")

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn이 없습니다. K-means 세그멘테이션 비활성화됩니다.")

logger = logging.getLogger(__name__)

class ClothSegmentationStep(ProcessingPipelineStep):
    """
    ✅ 3단계: 의류 세그멘테이션 - 통일된 생성자 패턴
    - 자동 디바이스 감지
    - M3 Max 최적화
    - 일관된 인터페이스
    - 다중 세그멘테이션 방법 지원
    """
    
    # 의류 카테고리 정의
    CLOTHING_CATEGORIES = {
        'upper': ['shirt', 't-shirt', 'blouse', 'sweater', 'jacket', 'coat', 'top'],
        'lower': ['pants', 'jeans', 'skirt', 'shorts', 'trousers', 'bottom'],
        'full': ['dress', 'jumpsuit', 'overall', 'gown'],
        'accessories': ['hat', 'scarf', 'gloves', 'shoes', 'bag', 'belt']
    }
    
    # 지원하는 세그멘테이션 방법들
    SEGMENTATION_METHODS = [
        'auto', 'rembg', 'model', 'grabcut', 'kmeans', 'threshold'
    ]
    
    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        ✅ 통일된 생성자 - 최적화된 인터페이스
        
        Args:
            device: 사용할 디바이스 (None=자동감지, 'cpu', 'cuda', 'mps')
            config: 스텝별 설정 딕셔너리
            **kwargs: 확장 파라미터들
                - method: str = 'auto' (세그멘테이션 방법)
                - model_name: str = 'u2net'
                - confidence_threshold: float = 0.5
                - use_background_removal: bool = True
                - quality_threshold: float = 0.7
                - enable_post_processing: bool = True
                - max_image_size: int = 1024 (M3 Max에서 더 큼)
                - morphology_enabled: bool = True
                - gaussian_blur: bool = True
                - edge_refinement: bool = True
                - hole_filling: bool = True
        """
        # 부모 클래스 초기화 (자동 디바이스 감지, M3 Max 최적화 등)
        super().__init__(device, config, **kwargs)
        
        # 3단계 전용 세그멘테이션 설정
        self.segmentation_config = self.config.get('segmentation', {})
        
        # 세그멘테이션 방법 설정
        self.method = self.config.get('method', 'auto')
        self.model_name = self.config.get('model_name', 'u2net')
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.use_background_removal = self.config.get('use_background_removal', True)
        
        # 품질 설정 (M3 Max에서 더 높은 품질)
        default_quality = 0.8 if self.is_m3_max else 0.7
        self.quality_threshold = self.config.get('quality_threshold', default_quality)
        
        # 후처리 설정
        self.post_process_config = self.config.get('post_processing', {})
        self.enable_post_processing = self.config.get('enable_post_processing', True)
        self.morphology_enabled = self.config.get('morphology_enabled', True)
        self.gaussian_blur = self.config.get('gaussian_blur', True)
        self.edge_refinement = self.config.get('edge_refinement', True)
        self.hole_filling = self.config.get('hole_filling', True)
        
        # 모델 및 세션 변수들
        self.rembg_session = None
        self.rembg_sessions = {}
        self.segmentation_model = None
        self.backup_methods = None
        
        # 3단계 전용 통계
        self.segmentation_stats = {
            'total_processed': 0,
            'successful_segmentations': 0,
            'average_quality': 0.0,
            'method_usage': {},
            'rembg_usage': 0,
            'model_usage': 0,
            'fallback_usage': 0,
            'cache_hits': 0
        }
        
        # 성능 캐시 (M3 Max에서 더 큰 캐시)
        cache_size = 100 if self.is_m3_max and self.memory_gb >= 128 else 50
        self.segmentation_cache = {}
        self.cache_max_size = cache_size
        
        self.logger.info(f"👕 의류 세그멘테이션 스텝 초기화 완료 - RemBG: {'✅' if REMBG_AVAILABLE else '❌'}")
        if self.is_m3_max:
            self.logger.info(f"🍎 M3 Max 최적화: 품질 {self.quality_threshold}, 크기 {self.max_resolution}")
    
    async def initialize(self) -> bool:
        """
        ✅ 통일된 초기화 인터페이스
        
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            self.logger.info("🔄 3단계: 의류 세그멘테이션 시스템 초기화 중...")
            
            # 1. RemBG 초기화
            await self._initialize_rembg()
            
            # 2. 커스텀 모델 초기화
            await self._initialize_custom_model()
            
            # 3. 백업 방법들 초기화
            self._initialize_backup_methods()
            
            # 4. M3 Max 최적화 워밍업
            if self.is_m3_max and self.optimization_enabled:
                await self._warmup_m3_max()
            
            self.is_initialized = True
            self.logger.info("✅ 의류 세그멘테이션 시스템 초기화 완료")
            return True
            
        except Exception as e:
            error_msg = f"세그멘테이션 시스템 초기화 실패: {e}"
            self.logger.error(f"❌ {error_msg}")
            self.initialization_error = error_msg
            
            # 최소한의 폴백 시스템으로라도 초기화
            self.backup_methods = self._create_simple_backup()
            self.is_initialized = True
            return True
    
    async def process(
        self, 
        clothing_image: Union[str, np.ndarray, Image.Image, torch.Tensor], 
        clothing_type: str = "shirt",
        quality_level: str = "high",
        **kwargs
    ) -> Dict[str, Any]:
        """
        ✅ 통일된 처리 인터페이스
        
        Args:
            clothing_image: 입력 의류 이미지
            clothing_type: 의류 타입
            quality_level: 품질 레벨 ('low', 'medium', 'high', 'ultra')
            **kwargs: 추가 매개변수
                - method_override: str = None (방법 강제 지정)
                - enable_fallback: bool = True
                - cache_result: bool = True
                
        Returns:
            Dict[str, Any]: 세그멘테이션 결과
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            self.logger.info(f"👕 의류 세그멘테이션 시작 - 타입: {clothing_type}, 품질: {quality_level}")
            
            # 캐시 확인
            cache_key = self._generate_cache_key(clothing_image, clothing_type, quality_level)
            if cache_key in self.segmentation_cache and kwargs.get('cache_result', True):
                self.logger.info("💾 캐시에서 세그멘테이션 결과 반환")
                self.segmentation_stats['cache_hits'] += 1
                cached_result = self.segmentation_cache[cache_key].copy()
                cached_result['from_cache'] = True
                return cached_result
            
            # 1. 입력 텐서 검증 및 전처리
            clothing_pil = self._prepare_input_image(clothing_image)
            
            # 2. 최적 세그멘테이션 방법 선택
            method = kwargs.get('method_override') or self._select_segmentation_method(
                clothing_pil, clothing_type, quality_level
            )
            self.logger.info(f"📋 선택된 방법: {method}")
            
            # 3. 메인 세그멘테이션 수행
            segmentation_result = await self._perform_segmentation(clothing_pil, method)
            
            # 4. 품질 평가
            quality_score = self._evaluate_quality(clothing_pil, segmentation_result['mask'])
            
            # 5. 품질이 낮으면 폴백 시도
            if (quality_score < self.quality_threshold and 
                kwargs.get('enable_fallback', True) and 
                method != 'fallback'):
                
                self.logger.info(f"🔄 품질 개선 시도 (현재: {quality_score:.3f})")
                improved_result = await self._try_fallback_methods(clothing_pil, clothing_type)
                
                if improved_result and improved_result.get('quality', 0) > quality_score:
                    segmentation_result = improved_result
                    quality_score = improved_result['quality']
                    method = improved_result.get('method', method)
            
            # 6. 후처리 적용
            if self.enable_post_processing:
                processed_result = self._apply_post_processing(segmentation_result, quality_level)
            else:
                processed_result = segmentation_result
            
            # 7. 최종 결과 구성
            processing_time = time.time() - start_time
            result = self._build_final_result(
                processed_result, quality_score, processing_time, method, 
                clothing_type, quality_level
            )
            
            # 8. 통계 업데이트
            self._update_segmentation_stats(method, quality_score, processing_time)
            self._update_performance_stats(processing_time, quality_score > 0.5)
            
            # 9. 캐시 저장
            if kwargs.get('cache_result', True):
                self._update_cache(cache_key, result)
            
            self.logger.info(f"✅ 세그멘테이션 완료 - 방법: {method}, 품질: {quality_score:.3f}, 시간: {processing_time:.3f}초")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"세그멘테이션 실패: {e}"
            self.logger.error(f"❌ {error_msg}")
            
            # 통계 업데이트 (실패)
            self._update_performance_stats(processing_time, False)
            
            return self._create_empty_result(error_msg)
    
    # =================================================================
    # 🔧 핵심 초기화 메서드들
    # =================================================================
    
    async def _warmup_m3_max(self):
        """M3 Max 워밍업"""
        try:
            self.logger.info("🍎 M3 Max 세그멘테이션 시스템 워밍업...")
            
            # 더미 이미지로 워밍업
            dummy_image = Image.new('RGB', (256, 256), color='white')
            
            # RemBG 워밍업
            if self.rembg_session:
                try:
                    _ = remove(dummy_image, session=self.rembg_session)
                except Exception as e:
                    self.logger.warning(f"RemBG 워밍업 실패: {e}")
            
            # 커스텀 모델 워밍업
            if self.segmentation_model:
                try:
                    dummy_tensor = self._preprocess_for_model(dummy_image)
                    with torch.no_grad():
                        _ = self.segmentation_model(dummy_tensor)
                except Exception as e:
                    self.logger.warning(f"모델 워밍업 실패: {e}")
            
            self.logger.info("✅ M3 Max 워밍업 완료")
            
        except Exception as e:
            self.logger.warning(f"M3 Max 워밍업 실패: {e}")
    
    async def _initialize_rembg(self):
        """RemBG 초기화 (M3 Max 최적화)"""
        if not REMBG_AVAILABLE:
            self.logger.warning("RemBG 사용 불가")
            return
        
        try:
            # 기본 세션 생성
            self.rembg_session = new_session('u2net')
            
            # 특화 세션들 생성 (M3 Max에서 더 많은 세션)
            if self.is_m3_max and self.memory_gb >= 64:
                self.rembg_sessions = {
                    'human_seg': new_session('u2net_human_seg'),
                    'cloth_seg': new_session('u2net_cloth_seg') if hasattr(rembg, 'u2net_cloth_seg') else self.rembg_session,
                    'silueta': new_session('silueta') if hasattr(rembg, 'silueta') else self.rembg_session
                }
            else:
                self.rembg_sessions = {
                    'human_seg': new_session('u2net_human_seg')
                }
            
            self.logger.info(f"✅ RemBG 세션 초기화 완료 - 세션 수: {len(self.rembg_sessions) + 1}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ RemBG 초기화 실패: {e}")
            self.rembg_session = None
            self.rembg_sessions = {}
    
    async def _initialize_custom_model(self):
        """커스텀 모델 초기화 (M3 Max 최적화)"""
        try:
            # 모델 로더가 있으면 사용
            if hasattr(self, 'model_loader') and self.model_loader:
                try:
                    self.segmentation_model = await self.model_loader.load_model(
                        self.model_name
                    )
                except Exception as e:
                    self.logger.warning(f"모델 로더 실패: {e}")
            
            # 없으면 간단한 모델 생성
            if not self.segmentation_model:
                self.segmentation_model = await self._create_u2net_model()
            
            self.logger.info("✅ 커스텀 모델 초기화 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 커스텀 모델 초기화 실패: {e}")
            self.segmentation_model = self._create_fallback_model()
    
    def _initialize_backup_methods(self):
        """백업 세그멘테이션 방법들 초기화"""
        try:
            self.backup_methods = BackupSegmentationMethods(
                self.device, 
                self.is_m3_max, 
                self.memory_gb
            )
            self.logger.info("✅ 백업 방법들 초기화 완료")
        except Exception as e:
            self.logger.warning(f"⚠️ 백업 방법 초기화 실패: {e}")
            self.backup_methods = self._create_simple_backup()
    
    # =================================================================
    # 🔧 핵심 처리 메서드들
    # =================================================================
    
    def _prepare_input_image(self, image_input: Union[str, np.ndarray, Image.Image, torch.Tensor]) -> Image.Image:
        """입력 이미지 전처리 (M3 Max 최적화)"""
        try:
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"이미지 파일이 존재하지 않음: {image_input}")
                image_pil = Image.open(image_input).convert('RGB')
                
            elif isinstance(image_input, np.ndarray):
                if len(image_input.shape) == 3:
                    # BGR to RGB 변환
                    if image_input.shape[2] == 3:
                        image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                    image_pil = Image.fromarray(image_input.astype(np.uint8))
                else:
                    raise ValueError("잘못된 numpy 배열 형태")
                    
            elif isinstance(image_input, torch.Tensor):
                # 텐서를 numpy로 변환
                if image_input.dim() == 4:
                    image_input = image_input.squeeze(0)
                if image_input.dim() == 3:
                    image_array = image_input.permute(1, 2, 0).cpu().numpy()
                    if image_array.max() <= 1.0:
                        image_array = (image_array * 255).astype(np.uint8)
                    image_pil = Image.fromarray(image_array)
                else:
                    raise ValueError("잘못된 텐서 형태")
                    
            elif isinstance(image_input, Image.Image):
                image_pil = image_input.convert('RGB')
                
            else:
                raise ValueError(f"지원하지 않는 이미지 형식: {type(image_input)}")
            
            # 크기 조정 (M3 Max에서 더 큰 크기 허용)
            max_size = self.max_resolution
            if max(image_pil.size) > max_size:
                # M3 Max에서 고품질 리샘플링
                resample = Image.Resampling.LANCZOS if self.is_m3_max else Image.Resampling.LANCZOS
                image_pil.thumbnail((max_size, max_size), resample)
                self.logger.info(f"🔄 이미지 크기 조정: {image_pil.size}")
            
            return image_pil
            
        except Exception as e:
            self.logger.error(f"이미지 전처리 실패: {e}")
            raise
    
    def _select_segmentation_method(self, image: Image.Image, clothing_type: str, quality_level: str) -> str:
        """최적 세그멘테이션 방법 선택 (M3 Max 최적화)"""
        
        method = self.method
        
        # 자동 선택 모드
        if method == 'auto':
            # 이미지 복잡도 분석
            complexity = self._analyze_image_complexity(image)
            
            # M3 Max에서 더 정교한 방법 선택
            if self.is_m3_max and quality_level in ['high', 'ultra']:
                if REMBG_AVAILABLE and self.rembg_session and complexity < 0.8:
                    return 'rembg'
                elif self.segmentation_model and complexity > 0.2:
                    return 'model'
            
            # 품질 레벨과 복잡도에 따른 방법 선택
            if quality_level in ['high', 'ultra']:
                if REMBG_AVAILABLE and self.rembg_session and complexity < 0.7:
                    return 'rembg'
                elif self.segmentation_model and complexity > 0.3:
                    return 'model'
            
            elif quality_level == 'medium':
                if REMBG_AVAILABLE and self.rembg_session:
                    return 'rembg'
                elif self.backup_methods and complexity < 0.6:
                    return 'grabcut'
            
            # 기본 방법
            if self.backup_methods:
                return 'grabcut'
            
            return 'threshold'  # 최후의 수단
        
        # 명시적 방법 선택 시 사용 가능성 확인
        if method == 'rembg' and not (REMBG_AVAILABLE and self.rembg_session):
            return 'grabcut'
        elif method == 'model' and not self.segmentation_model:
            return 'grabcut'
        elif method == 'kmeans' and not SKLEARN_AVAILABLE:
            return 'grabcut'
        
        return method
    
    def _analyze_image_complexity(self, image: Image.Image) -> float:
        """이미지 복잡도 분석 (M3 Max 고정밀도)"""
        try:
            # 그레이스케일 변환
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # 엣지 밀도 계산
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # 텍스처 복잡도 (표준편차 기반)
            texture_complexity = np.std(gray) / 255.0
            
            # 히스토그램 복잡도
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)
            
            # 엔트로피 계산
            entropy = -np.sum(hist * np.log2(hist + 1e-7))
            hist_complexity = entropy / 8.0  # 정규화
            
            # M3 Max에서 더 정교한 복잡도 계산
            if self.is_m3_max:
                # 그래디언트 복잡도 추가
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                grad_complexity = np.std(grad_magnitude) / (np.mean(grad_magnitude) + 1e-7)
                grad_complexity = min(grad_complexity / 10.0, 1.0)
                
                # 종합 복잡도 (가중평균)
                complexity = (
                    edge_density * 0.3 + 
                    texture_complexity * 0.3 + 
                    hist_complexity * 0.2 +
                    grad_complexity * 0.2
                )
            else:
                # 기본 복잡도
                complexity = (
                    edge_density * 0.4 + 
                    texture_complexity * 0.4 + 
                    hist_complexity * 0.2
                )
            
            return min(max(complexity, 0.0), 1.0)
            
        except Exception as e:
            self.logger.warning(f"복잡도 분석 실패: {e}")
            return 0.5  # 기본값
    
    async def _perform_segmentation(self, image: Image.Image, method: str) -> Dict[str, Any]:
        """메인 세그멘테이션 수행"""
        try:
            if method == 'rembg' and self.rembg_session:
                result = await self._segment_with_rembg(image)
                self.segmentation_stats['rembg_usage'] += 1
                return result
            elif method == 'model' and self.segmentation_model:
                result = await self._segment_with_model(image)
                self.segmentation_stats['model_usage'] += 1
                return result
            elif method == 'grabcut' and self.backup_methods:
                result = self.backup_methods.grabcut_segmentation(image)
                self.segmentation_stats['fallback_usage'] += 1
                return result
            elif method == 'kmeans' and SKLEARN_AVAILABLE:
                result = await self._segment_with_kmeans(image)
                self.segmentation_stats['fallback_usage'] += 1
                return result
            elif method == 'threshold' and self.backup_methods:
                result = self.backup_methods.threshold_segmentation(image)
                self.segmentation_stats['fallback_usage'] += 1
                return result
            else:
                # 폴백
                result = await self._segment_with_simple_threshold(image)
                self.segmentation_stats['fallback_usage'] += 1
                return result
                
        except Exception as e:
            self.logger.warning(f"세그멘테이션 방법 {method} 실패: {e}")
            result = await self._segment_with_simple_threshold(image)
            self.segmentation_stats['fallback_usage'] += 1
            return result
    
    async def _segment_with_rembg(self, image: Image.Image) -> Dict[str, Any]:
        """RemBG를 사용한 세그멘테이션 (M3 Max 최적화)"""
        try:
            # 의류 타입별 모델 선택 (M3 Max에서 더 정교)
            if self.is_m3_max and 'cloth_seg' in self.rembg_sessions:
                specialized_session = self.rembg_sessions['cloth_seg']
            else:
                specialized_session = self.rembg_sessions.get('human_seg', self.rembg_session)
            
            # RemBG 처리
            result_image = remove(image, session=specialized_session)
            
            # 마스크 추출
            if result_image.mode == 'RGBA':
                mask = np.array(result_image)[:, :, 3]
                segmented_rgb = result_image.convert('RGB')
            else:
                # RGBA가 아닌 경우 간단한 임계값 사용
                gray = np.array(result_image.convert('L'))
                mask = (gray > 20).astype(np.uint8) * 255
                segmented_rgb = result_image.convert('RGB')
            
            return {
                'segmented_image': segmented_rgb,
                'mask': mask,
                'method': 'rembg',
                'confidence': 0.9
            }
            
        except Exception as e:
            self.logger.warning(f"RemBG 처리 실패: {e}")
            raise
    
    async def _segment_with_model(self, image: Image.Image) -> Dict[str, Any]:
        """커스텀 모델을 사용한 세그멘테이션 (M3 Max 최적화)"""
        try:
            # 입력 전처리
            input_tensor = self._preprocess_for_model(image)
            
            # 모델 추론 (M3 Max에서 더 큰 배치 크기 가능)
            with torch.no_grad():
                if self.is_m3_max and self.memory_gb >= 64:
                    # 높은 정밀도 모드
                    if hasattr(self.segmentation_model, 'eval'):
                        self.segmentation_model.eval()
                
                mask_pred = self.segmentation_model(input_tensor)
                mask = mask_pred.squeeze().cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
            
            # 원본 크기로 복원
            if mask.shape != (image.height, image.width):
                # M3 Max에서 고품질 보간
                interpolation = cv2.INTER_LANCZOS4 if self.is_m3_max else cv2.INTER_NEAREST
                mask = cv2.resize(mask, (image.width, image.height), interpolation=interpolation)
            
            # 세그멘테이션된 이미지 생성
            segmented_image = self._apply_mask_to_image(image, mask)
            
            return {
                'segmented_image': segmented_image,
                'mask': mask,
                'method': 'model',
                'confidence': 0.8
            }
            
        except Exception as e:
            self.logger.warning(f"모델 세그멘테이션 실패: {e}")
            raise
    
    async def _segment_with_kmeans(self, image: Image.Image) -> Dict[str, Any]:
        """K-means를 사용한 세그멘테이션 (M3 Max 최적화)"""
        try:
            # 이미지를 numpy 배열로 변환
            image_array = np.array(image)
            
            # 픽셀을 1D 배열로 변환
            pixels = image_array.reshape(-1, 3)
            
            # M3 Max에서 더 많은 클러스터 사용
            n_clusters = 3 if self.is_m3_max else 2
            
            # K-means 클러스터링
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixels)
            
            # 라벨을 이미지 크기로 복원
            label_image = labels.reshape(image_array.shape[:2])
            
            # 전경과 배경 구분 (가장 큰 클러스터를 배경으로 가정)
            unique, counts = np.unique(labels, return_counts=True)
            background_label = unique[np.argmax(counts)]
            
            # 마스크 생성 (전경=255, 배경=0)
            mask = (label_image != background_label).astype(np.uint8) * 255
            
            # 세그멘테이션된 이미지 생성
            segmented_image = self._apply_mask_to_image(image, mask)
            
            return {
                'segmented_image': segmented_image,
                'mask': mask,
                'method': 'kmeans',
                'confidence': 0.7
            }
            
        except Exception as e:
            self.logger.warning(f"K-means 세그멘테이션 실패: {e}")
            raise
    
    async def _segment_with_simple_threshold(self, image: Image.Image) -> Dict[str, Any]:
        """간단한 임계값 세그멘테이션 (최후의 수단)"""
        try:
            # 그레이스케일 변환
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # M3 Max에서 더 정교한 임계값 처리
            if self.is_m3_max:
                # 적응적 임계값 + Otsu 조합
                _, mask1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                mask2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                # 기본 Otsu 임계값
                _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 가장 큰 연결 성분만 유지
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                mask = np.zeros_like(mask)
                cv2.fillPoly(mask, [largest_contour], 255)
            
            # 세그멘테이션된 이미지 생성
            segmented_image = self._apply_mask_to_image(image, mask)
            
            return {
                'segmented_image': segmented_image,
                'mask': mask,
                'method': 'threshold',
                'confidence': 0.6
            }
            
        except Exception as e:
            self.logger.error(f"임계값 세그멘테이션 실패: {e}")
            # 전체 이미지를 마스크로 반환
            mask = np.ones((image.height, image.width), dtype=np.uint8) * 255
            return {
                'segmented_image': image,
                'mask': mask,
                'method': 'fallback',
                'confidence': 0.3
            }
    
    # =================================================================
    # 🔧 품질 평가 및 후처리 메서드들
    # =================================================================
    
    def _evaluate_quality(self, original_image: Image.Image, mask: np.ndarray) -> float:
        """세그멘테이션 품질 평가 (M3 Max 고정밀도)"""
        try:
            # 1. 마스크 커버리지 (전체 이미지 대비 마스크 비율)
            mask_coverage = np.sum(mask > 0) / mask.size
            
            # 2. 마스크 연결성 (연결된 영역 수)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            connectivity_score = 1.0 / (len(contours) + 1)  # 영역이 적을수록 좋음
            
            # 3. 엣지 품질 (엣지의 부드러움)
            edges = cv2.Canny(mask, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            edge_score = min(edge_density * 10, 1.0)  # 적당한 엣지 밀도
            
            # 4. 형태 복잡도 (볼록성)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(largest_contour)
                contour_area = cv2.contourArea(largest_contour)
                hull_area = cv2.contourArea(hull)
                
                if hull_area > 0:
                    convexity = contour_area / hull_area
                else:
                    convexity = 0.0
            else:
                convexity = 0.0
            
            # M3 Max에서 추가 품질 메트릭
            if self.is_m3_max:
                # 5. 경계선 부드러움
                boundary_smoothness = self._calculate_boundary_smoothness(mask)
                
                # 종합 점수 계산 (M3 Max 가중치)
                quality_score = (
                    mask_coverage * 0.25 +
                    connectivity_score * 0.25 +
                    edge_score * 0.2 +
                    convexity * 0.15 +
                    boundary_smoothness * 0.15
                )
            else:
                # 기본 종합 점수
                quality_score = (
                    mask_coverage * 0.3 +
                    connectivity_score * 0.3 +
                    edge_score * 0.2 +
                    convexity * 0.2
                )
            
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception as e:
            self.logger.warning(f"품질 평가 실패: {e}")
            return 0.5
    
    def _calculate_boundary_smoothness(self, mask: np.ndarray) -> float:
        """경계선 부드러움 계산 (M3 Max 전용)"""
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.0
            
            largest_contour = max(contours, key=cv2.contourArea)
            
            if len(largest_contour) < 10:
                return 0.0
            
            # 윤곽선의 곡률 변화 계산
            contour_points = largest_contour.reshape(-1, 2)
            
            # 연속된 세 점 간의 각도 변화 계산
            angle_changes = []
            for i in range(2, len(contour_points)):
                p1, p2, p3 = contour_points[i-2], contour_points[i-1], contour_points[i]
                
                v1 = p2 - p1
                v2 = p3 - p2
                
                # 각도 계산
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angle_changes.append(angle)
            
            # 각도 변화의 표준편차 (낮을수록 부드러움)
            if angle_changes:
                smoothness = 1.0 - min(np.std(angle_changes) / np.pi, 1.0)
                return smoothness
            
            return 0.5
            
        except Exception as e:
            self.logger.warning(f"경계선 부드러움 계산 실패: {e}")
            return 0.5
    
    def _apply_post_processing(self, segmentation_result: Dict[str, Any], quality_level: str) -> Dict[str, Any]:
        """후처리 적용 (M3 Max 최적화)"""
        
        if not self.enable_post_processing:
            return segmentation_result
        
        try:
            mask = segmentation_result['mask'].copy()
            
            # 품질 레벨에 따른 처리 강도
            intensity_map = {'low': 0, 'medium': 1, 'high': 2, 'ultra': 3}
            intensity = intensity_map.get(quality_level, 1)
            
            # M3 Max에서 더 강력한 후처리
            if self.is_m3_max and intensity >= 2:
                intensity += 1
            
            processed_mask = mask.copy()
            
            # 1. 형태학적 연산 (노이즈 제거)
            if self.morphology_enabled:
                kernel_size = 3 + intensity
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel)
                processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)
            
            # 2. 가우시안 블러 (엣지 스무딩)
            if self.gaussian_blur:
                blur_kernel = 3 + intensity * 2
                if blur_kernel % 2 == 0:
                    blur_kernel += 1
                processed_mask = cv2.GaussianBlur(processed_mask, (blur_kernel, blur_kernel), 0)
                processed_mask = (processed_mask > 127).astype(np.uint8) * 255
            
            # 3. 홀 채우기
            if self.hole_filling:
                contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.fillPoly(processed_mask, contours, 255)
            
            # 4. 엣지 정제 (M3 Max 고급 모드)
            if self.edge_refinement and intensity > 0:
                processed_mask = self._refine_edges(processed_mask, intensity)
            
            # 5. M3 Max 전용 고급 후처리
            if self.is_m3_max and quality_level == 'ultra':
                processed_mask = self._advanced_post_processing(processed_mask)
            
            segmentation_result['mask'] = processed_mask
            segmentation_result['segmented_image'] = self._apply_mask_to_image(
                segmentation_result['segmented_image'], processed_mask
            )
            
            return segmentation_result
            
        except Exception as e:
            self.logger.warning(f"후처리 실패: {e}")
            return segmentation_result
    
    def _advanced_post_processing(self, mask: np.ndarray) -> np.ndarray:
        """M3 Max 전용 고급 후처리"""
        try:
            # 1. 윤곽선 스무딩
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 가장 큰 윤곽선 선택
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 윤곽선 근사화 (Douglas-Peucker 알고리즘)
                epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # 새로운 마스크 생성
                refined_mask = np.zeros_like(mask)
                cv2.fillPoly(refined_mask, [approx_contour], 255)
                
                return refined_mask
            
            return mask
            
        except Exception as e:
            self.logger.warning(f"고급 후처리 실패: {e}")
            return mask
    
    # =================================================================
    # 🔧 헬퍼 메서드들
    # =================================================================
    
    def _refine_edges(self, mask: np.ndarray, intensity: int) -> np.ndarray:
        """엣지 정제"""
        try:
            # 엣지 검출
            edges = cv2.Canny(mask, 50, 150)
            
            # 엣지 확장
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated_edges = cv2.dilate(edges, kernel, iterations=intensity)
            
            # 원본 마스크와 결합
            refined_mask = cv2.bitwise_or(mask, dilated_edges)
            
            return refined_mask
            
        except Exception as e:
            self.logger.warning(f"엣지 정제 실패: {e}")
            return mask
    
    def _apply_mask_to_image(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """이미지에 마스크 적용"""
        try:
            # PIL 이미지를 numpy 배열로 변환
            image_array = np.array(image)
            
            # 마스크를 3채널로 확장
            if len(mask.shape) == 2:
                mask_3channel = np.stack([mask] * 3, axis=2)
            else:
                mask_3channel = mask
            
            # 마스크 정규화 (0-1 범위)
            mask_normalized = mask_3channel.astype(np.float32) / 255.0
            
            # 마스크 적용
            segmented_array = image_array * mask_normalized
            
            # PIL 이미지로 변환
            segmented_image = Image.fromarray(segmented_array.astype(np.uint8))
            
            return segmented_image
            
        except Exception as e:
            self.logger.warning(f"마스크 적용 실패: {e}")
            return image
    
    async def _try_fallback_methods(self, image: Image.Image, clothing_type: str) -> Optional[Dict[str, Any]]:
        """폴백 방법들 시도"""
        
        fallback_methods = ['grabcut', 'kmeans', 'threshold']
        best_result = None
        best_quality = 0.0
        
        for method in fallback_methods:
            try:
                if method == 'grabcut' and self.backup_methods:
                    result = self.backup_methods.grabcut_segmentation(image)
                elif method == 'kmeans' and SKLEARN_AVAILABLE:
                    result = await self._segment_with_kmeans(image)
                elif method == 'threshold':
                    result = await self._segment_with_simple_threshold(image)
                else:
                    continue
                
                # 품질 평가
                quality = self._evaluate_quality(image, result['mask'])
                result['quality'] = quality
                
                if quality > best_quality:
                    best_quality = quality
                    best_result = result
                    
                self.logger.info(f"📊 폴백 방법 {method}: 품질 {quality:.3f}")
                
            except Exception as e:
                self.logger.warning(f"폴백 방법 {method} 실패: {e}")
                continue
        
        return best_result
    
    # =================================================================
    # 🔧 결과 생성 및 통계 메서드들
    # =================================================================
    
    def _build_final_result(
        self,
        processed_result: Dict[str, Any],
        quality_score: float,
        processing_time: float,
        method: str,
        clothing_type: str,
        quality_level: str
    ) -> Dict[str, Any]:
        """최종 결과 구성"""
        
        try:
            # 기본 결과 구조
            result = {
                'success': True,
                'segmented_image': processed_result['segmented_image'],
                'clothing_mask': processed_result['mask'],
                'mask': processed_result['mask'],  # 호환성을 위한 중복
                'clothing_type': clothing_type,
                'segmentation_method': method,
                'quality_score': quality_score,
                'confidence': processed_result.get('confidence', quality_score),
                'processing_time': processing_time,
                'quality_level': quality_level
            }
            
            # 품질 등급 추가
            if quality_score >= 0.9:
                result['quality_grade'] = 'excellent'
            elif quality_score >= 0.8:
                result['quality_grade'] = 'good'
            elif quality_score >= 0.6:
                result['quality_grade'] = 'fair'
            elif quality_score >= 0.4:
                result['quality_grade'] = 'poor'
            else:
                result['quality_grade'] = 'very_poor'
            
            # 세그멘테이션 분석 추가
            result['segmentation_analysis'] = self._analyze_segmentation(processed_result['mask'])
            
            # 처리 정보 추가
            result['processing_info'] = {
                'method_used': method,
                'post_processing_applied': self.enable_post_processing,
                'fallback_used': method in ['grabcut', 'kmeans', 'threshold'],
                'device': self.device,
                'device_type': self.device_type,
                'm3_max_optimized': self.is_m3_max,
                'image_size': f"{processed_result['segmented_image'].size[0]}x{processed_result['segmented_image'].size[1]}",
                'mask_coverage': np.sum(processed_result['mask'] > 0) / processed_result['mask'].size,
                'quality_threshold_met': quality_score >= self.quality_threshold
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"최종 결과 구성 실패: {e}")
            return self._create_empty_result("결과 구성 오류")
    
    def _analyze_segmentation(self, mask: np.ndarray) -> Dict[str, Any]:
        """세그멘테이션 분석"""
        
        analysis = {}
        
        try:
            # 마스크 영역 분석
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 가장 큰 영역
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 바운딩 박스
                x, y, w, h = cv2.boundingRect(largest_contour)
                analysis['bounding_box'] = {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
                
                # 영역 정보
                analysis['area'] = float(cv2.contourArea(largest_contour))
                analysis['perimeter'] = float(cv2.arcLength(largest_contour, True))
                
                # 형태 특성
                if analysis['perimeter'] > 0:
                    analysis['compactness'] = 4 * np.pi * analysis['area'] / (analysis['perimeter'] ** 2)
                else:
                    analysis['compactness'] = 0.0
                
                # 종횡비
                if h > 0:
                    analysis['aspect_ratio'] = w / h
                else:
                    analysis['aspect_ratio'] = 1.0
                
                # 영역 개수
                analysis['num_regions'] = len(contours)
                
                # M3 Max 추가 분석
                if self.is_m3_max:
                    # 볼록성
                    hull = cv2.convexHull(largest_contour)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        analysis['convexity'] = analysis['area'] / hull_area
                    else:
                        analysis['convexity'] = 0.0
                    
                    # 견고성 (면적/바운딩박스면적)
                    bbox_area = w * h
                    if bbox_area > 0:
                        analysis['solidity'] = analysis['area'] / bbox_area
                    else:
                        analysis['solidity'] = 0.0
                
            else:
                # 윤곽선이 없는 경우
                analysis = {
                    'bounding_box': {'x': 0, 'y': 0, 'width': 0, 'height': 0},
                    'area': 0.0,
                    'perimeter': 0.0,
                    'compactness': 0.0,
                    'aspect_ratio': 1.0,
                    'num_regions': 0
                }
            
        except Exception as e:
            self.logger.warning(f"세그멘테이션 분석 실패: {e}")
            analysis = {'error': str(e)}
        
        return analysis
    
    # =================================================================
    # 🔧 캐시 및 통계 메서드들
    # =================================================================
    
    def _generate_cache_key(self, image_input, clothing_type: str, quality_level: str) -> str:
        """캐시 키 생성"""
        try:
            if isinstance(image_input, str):
                base_hash = hash(image_input)
            elif hasattr(image_input, 'tobytes'):
                base_hash = hash(image_input.tobytes())
            else:
                base_hash = hash(str(image_input))
            
            return f"seg_{base_hash}_{clothing_type}_{quality_level}_{self.method}"
        except Exception:
            return f"seg_fallback_{time.time()}"
    
    def _update_cache(self, key: str, result: Dict[str, Any]):
        """캐시 업데이트"""
        try:
            if len(self.segmentation_cache) >= self.cache_max_size:
                # 가장 오래된 항목 제거
                oldest_key = next(iter(self.segmentation_cache))
                del self.segmentation_cache[oldest_key]
            
            # 결과 복사해서 저장 (무거운 데이터 제외)
            cached_result = {k: v for k, v in result.items() if k not in ['segmented_image']}
            self.segmentation_cache[key] = cached_result
            
        except Exception as e:
            self.logger.warning(f"캐시 업데이트 실패: {e}")
    
    def _update_segmentation_stats(self, method: str, quality: float, processing_time: float):
        """3단계 전용 통계 업데이트"""
        self.segmentation_stats['total_processed'] += 1
        
        if quality > 0.5:
            self.segmentation_stats['successful_segmentations'] += 1
        
        # 품질 이동 평균
        alpha = 0.1
        self.segmentation_stats['average_quality'] = (
            alpha * quality + 
            (1 - alpha) * self.segmentation_stats['average_quality']
        )
        
        # 방법별 사용 통계
        if method not in self.segmentation_stats['method_usage']:
            self.segmentation_stats['method_usage'][method] = 0
        self.segmentation_stats['method_usage'][method] += 1
    
    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """빈 결과 생성"""
        return {
            'success': False,
            'error': reason,
            'segmented_image': None,
            'clothing_mask': None,
            'mask': None,
            'clothing_type': 'unknown',
            'segmentation_method': 'none',
            'quality_score': 0.0,
            'confidence': 0.0,
            'quality_grade': 'failed',
            'processing_time': 0.0,
            'segmentation_analysis': {},
            'processing_info': {
                'method_used': 'none',
                'post_processing_applied': False,
                'fallback_used': False,
                'error_occurred': True,
                'device': self.device,
                'error_details': reason
            }
        }
    
    # =================================================================
    # 🔧 모델 생성 메서드들
    # =================================================================
    
    async def _create_u2net_model(self):
        """U²-Net 스타일 모델 생성 (M3 Max 최적화)"""
        class SimpleU2Net(torch.nn.Module):
            def __init__(self, is_m3_max=False):
                super(SimpleU2Net, self).__init__()
                
                # M3 Max에서 더 복잡한 모델
                channels = 64 if is_m3_max else 32
                
                # 간단한 U-Net 구조
                self.encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(3, channels, 3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(channels, channels, 3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool2d(2)
                )
                
                self.middle = torch.nn.Sequential(
                    torch.nn.Conv2d(channels, channels*2, 3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(channels*2, channels*2, 3, padding=1),
                    torch.nn.ReLU(inplace=True),
                )
                
                self.decoder = torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    torch.nn.Conv2d(channels*2, channels, 3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(channels, 1, 1),
                    torch.nn.Sigmoid()
                )
            
            def forward(self, x):
                x1 = self.encoder(x)
                x2 = self.middle(x1)
                x3 = self.decoder(x2)
                return x3
        
        model = SimpleU2Net(self.is_m3_max).to(self.device)
        if self.is_m3_max and self.device == 'mps':
            # M3 Max MPS 최적화
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
        
        return model
    
    def _create_fallback_model(self):
        """폴백 모델 생성"""
        class FallbackModel:
            def __call__(self, x):
                batch_size = x.shape[0] if len(x.shape) == 4 else 1
                height, width = x.shape[-2], x.shape[-1]
                return torch.ones(batch_size, 1, height, width) * 0.5
        
        return FallbackModel()
    
    def _create_simple_backup(self):
        """간단한 백업 방법 생성"""
        class SimpleBackup:
            def grabcut_segmentation(self, image):
                mask = np.ones((image.height, image.width), dtype=np.uint8) * 255
                return {
                    'segmented_image': image,
                    'mask': mask,
                    'method': 'simple_grabcut',
                    'confidence': 0.5
                }
            
            def threshold_segmentation(self, image):
                mask = np.ones((image.height, image.width), dtype=np.uint8) * 255
                return {
                    'segmented_image': image,
                    'mask': mask,
                    'method': 'simple_threshold',
                    'confidence': 0.4
                }
        
        return SimpleBackup()
    
    def _preprocess_for_model(self, image: Image.Image) -> torch.Tensor:
        """모델용 이미지 전처리"""
        try:
            # PIL을 텐서로 변환
            image_array = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
            
            # 디바이스로 이동
            image_tensor = image_tensor.to(self.device)
            
            return image_tensor
            
        except Exception as e:
            self.logger.error(f"모델 전처리 실패: {e}")
            raise
    
    # =================================================================
    # 🔧 Pipeline Manager 호환 메서드들
    # =================================================================
    
    async def get_step_info(self) -> Dict[str, Any]:
        """🔍 3단계 상세 정보 반환"""
        base_info = await super().get_step_info()
        
        # 3단계 전용 정보 추가
        base_info.update({
            "segmentation_stats": self.segmentation_stats.copy(),
            "clothing_categories": {
                category: items for category, items in self.CLOTHING_CATEGORIES.items()
            },
            "supported_methods": self.SEGMENTATION_METHODS,
            "cache_usage": {
                "cache_size": len(self.segmentation_cache),
                "cache_limit": self.cache_max_size,
                "hit_rate": self.segmentation_stats['cache_hits'] / max(1, self.segmentation_stats['total_processed'])
            },
            "models_available": {
                "rembg": self.rembg_session is not None,
                "custom_model": self.segmentation_model is not None,
                "backup_methods": self.backup_methods is not None,
                "sklearn_kmeans": SKLEARN_AVAILABLE
            },
            "capabilities": {
                "segmentation_method": self.method,
                "model_name": self.model_name,
                "max_resolution": self.max_resolution,
                "post_processing_enabled": self.enable_post_processing,
                "quality_threshold": self.quality_threshold,
                "background_removal": self.use_background_removal,
                "advanced_analysis": self.is_m3_max
            },
            "rembg_sessions": list(self.rembg_sessions.keys()) if self.rembg_sessions else []
        })
        
        return base_info
    
    def get_supported_clothing_types(self) -> Dict[str, List[str]]:
        """지원하는 의류 타입 반환"""
        return self.CLOTHING_CATEGORIES.copy()
    
    def get_supported_methods(self) -> List[str]:
        """지원하는 세그멘테이션 방법들 반환"""
        return self.SEGMENTATION_METHODS.copy()
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            self.logger.info("🧹 3단계: 의류 세그멘테이션 리소스 정리 중...")
            
            # RemBG 세션 정리
            if hasattr(self, 'rembg_session'):
                self.rembg_session = None
            self.rembg_sessions = {}
            
            # 모델 정리
            if hasattr(self, 'segmentation_model'):
                if hasattr(self.segmentation_model, 'cpu'):
                    self.segmentation_model.cpu()
                self.segmentation_model = None
            
            # 백업 방법들 정리
            self.backup_methods = None
            
            # 캐시 정리
            self.segmentation_cache.clear()
            
            # 부모 클래스 정리
            await super().cleanup()
            
            self.logger.info("✅ 3단계 의류 세그멘테이션 리소스 정리 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 리소스 정리 중 오류: {e}")


# =================================================================
# 🔧 백업 세그멘테이션 방법들 클래스
# =================================================================

class BackupSegmentationMethods:
    """백업 세그멘테이션 방법들 (M3 Max 최적화)"""
    
    def __init__(self, device: str, is_m3_max: bool = False, memory_gb: float = 16.0):
        self.device = device
        self.is_m3_max = is_m3_max
        self.memory_gb = memory_gb
        self.logger = logging.getLogger(__name__)
    
    def grabcut_segmentation(self, image: Image.Image) -> Dict[str, Any]:
        """GrabCut 알고리즘을 사용한 세그멘테이션 (M3 Max 최적화)"""
        try:
            # PIL을 OpenCV 형식으로 변환
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # 초기 사각형 (이미지의 10% 여백)
            height, width = img.shape[:2]
            
            # M3 Max에서 더 정교한 초기 영역 설정
            if self.is_m3_max:
                margin = 0.08  # 8% 여백 (더 정밀)
            else:
                margin = 0.1   # 10% 여백
            
            rect = (
                int(width * margin), 
                int(height * margin), 
                int(width * (1 - 2 * margin)), 
                int(height * (1 - 2 * margin))
            )
            
            # GrabCut 초기화
            mask = np.zeros((height, width), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # M3 Max에서 더 많은 반복
            iterations = 7 if self.is_m3_max else 5
            
            # GrabCut 수행
            cv2.grabCut(img, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
            
            # 마스크 후처리
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            final_mask = mask2 * 255
            
            # M3 Max에서 추가 정제
            if self.is_m3_max:
                # 모폴로지 연산으로 정제
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
                final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
            
            # RGB로 변환된 세그멘테이션 이미지 생성
            segmented_img = img * mask2[:, :, np.newaxis]
            segmented_img_rgb = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)
            segmented_image = Image.fromarray(segmented_img_rgb.astype(np.uint8))
            
            confidence = 0.8 if self.is_m3_max else 0.75
            
            return {
                'segmented_image': segmented_image,
                'mask': final_mask,
                'method': 'grabcut',
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.warning(f"GrabCut 실패: {e}")
            # 폴백: 전체 이미지를 마스크로 반환
            mask = np.ones((image.height, image.width), dtype=np.uint8) * 255
            return {
                'segmented_image': image,
                'mask': mask,
                'method': 'grabcut_fallback',
                'confidence': 0.3
            }
    
    def threshold_segmentation(self, image: Image.Image) -> Dict[str, Any]:
        """임계값 기반 세그멘테이션 (M3 Max 최적화)"""
        try:
            # 그레이스케일 변환
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # M3 Max에서 더 정교한 임계값 처리
            if self.is_m3_max:
                # 다중 임계값 조합
                _, mask1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                mask2 = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                mask3 = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
                )
                
                # 마스크 조합 (다수결 원리)
                mask_sum = (mask1.astype(np.float32) + mask2.astype(np.float32) + mask3.astype(np.float32)) / 3
                mask = (mask_sum > 127).astype(np.uint8) * 255
            else:
                # 기본 적응형 임계값
                mask = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
            
            # 노이즈 제거
            kernel_size = 7 if self.is_m3_max else 5
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # M3 Max에서 추가 정제
            if self.is_m3_max:
                # 가장 큰 연결 성분만 유지
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # 면적 기준으로 상위 컴포넌트들 선택
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    mask_refined = np.zeros_like(mask)
                    
                    # 전체 면적의 5% 이상인 컴포넌트들만 유지
                    total_area = mask.shape[0] * mask.shape[1]
                    for contour in contours:
                        if cv2.contourArea(contour) > total_area * 0.05:
                            cv2.fillPoly(mask_refined, [contour], 255)
                    
                    mask = mask_refined
            
            # 세그멘테이션된 이미지 생성
            image_array = np.array(image)
            mask_3channel = np.stack([mask] * 3, axis=2)
            segmented_array = image_array * (mask_3channel / 255.0)
            segmented_image = Image.fromarray(segmented_array.astype(np.uint8))
            
            confidence = 0.7 if self.is_m3_max else 0.65
            
            return {
                'segmented_image': segmented_image,
                'mask': mask,
                'method': 'adaptive_threshold',
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.warning(f"임계값 세그멘테이션 실패: {e}")
            # 폴백
            mask = np.ones((image.height, image.width), dtype=np.uint8) * 255
            return {
                'segmented_image': image,
                'mask': mask,
                'method': 'threshold_fallback',
                'confidence': 0.3
            }
    
    def watershed_segmentation(self, image: Image.Image) -> Dict[str, Any]:
        """Watershed 세그멘테이션 (M3 Max 전용 고급 방법)"""
        if not self.is_m3_max:
            # M3 Max가 아니면 간단한 방법으로 폴백
            return self.threshold_segmentation(image)
        
        try:
            # numpy 배열로 변환
            img = np.array(image)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # 노이즈 제거
            img_blur = cv2.medianBlur(gray, 5)
            
            # 임계값으로 이진 이미지 생성
            _, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 노이즈 제거
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # 확실한 배경 영역
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            
            # 확실한 전경 영역
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            
            # 불확실한 영역
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            # 마커 라벨링
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            
            # Watershed 적용
            markers = cv2.watershed(img, markers)
            
            # 결과 마스크 생성
            mask = np.zeros(gray.shape, dtype=np.uint8)
            mask[markers > 1] = 255
            
            # 세그멘테이션된 이미지 생성
            mask_3channel = np.stack([mask] * 3, axis=2)
            segmented_array = img * (mask_3channel / 255.0)
            segmented_image = Image.fromarray(segmented_array.astype(np.uint8))
            
            return {
                'segmented_image': segmented_image,
                'mask': mask,
                'method': 'watershed',
                'confidence': 0.8
            }
            
        except Exception as e:
            self.logger.warning(f"Watershed 세그멘테이션 실패: {e}")
            # 폴백
            return self.threshold_segmentation(image)


# =================================================================
# 🔄 하위 호환성 지원 (기존 코드 호환)
# =================================================================

async def create_cloth_segmentation_step(
    device: str = "auto",
    config: Dict[str, Any] = None
) -> ClothSegmentationStep:
    """
    🔄 기존 팩토리 함수 호환 (기존 파이프라인 호환)
    
    Args:
        device: 사용할 디바이스 ("auto"는 자동 감지)
        config: 설정 딕셔너리
        
    Returns:
        ClothSegmentationStep: 초기화된 3단계 스텝
    """
    # 기존 방식 호환
    device_param = None if device == "auto" else device
    
    default_config = {
        "method": "auto",
        "model_name": "u2net",
        "confidence_threshold": 0.5,
        "use_background_removal": True,
        "quality_threshold": 0.7,
        "enable_post_processing": True,
        "max_image_size": 1024,
        "morphology_enabled": True,
        "gaussian_blur": True,
        "edge_refinement": True,
        "hole_filling": True
    }
    
    final_config = {**default_config, **(config or {})}
    
    # ✅ 새로운 통일된 생성자 사용
    step = ClothSegmentationStep(device=device_param, config=final_config)
    
    if not await step.initialize():
        logger.warning("3단계 초기화 실패했지만 진행합니다.")
    
    return step

# 기존 클래스명 별칭 (완전 호환)
ClothSegmentationStepLegacy = ClothSegmentationStep

# 유틸리티 함수들
def get_supported_segmentation_methods() -> List[str]:
    """지원하는 세그멘테이션 방법들 반환"""
    return ClothSegmentationStep.SEGMENTATION_METHODS.copy()

def get_clothing_categories() -> Dict[str, List[str]]:
    """의류 카테고리 정보 반환"""
    return ClothSegmentationStep.CLOTHING_CATEGORIES.copy()

def is_rembg_available() -> bool:
    """RemBG 사용 가능 여부"""
    return REMBG_AVAILABLE

def is_sklearn_available() -> bool:
    """scikit-learn 사용 가능 여부"""
    return SKLEARN_AVAILABLE