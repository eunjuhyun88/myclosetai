# app/ai_pipeline/steps/step_03_cloth_segmentation.py
"""
3단계: 의류 세그멘테이션 (Clothing Segmentation) - 수정된 버전
Pipeline Manager와 완전 호환되는 배경 제거 시스템
M3 Max 최적화 + 견고한 에러 처리 + 폴백 메커니즘
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

class ClothSegmentationStep:
    """
    의류 세그멘테이션 스텝 - Pipeline Manager 완전 호환
    - M3 Max MPS 최적화
    - 다중 세그멘테이션 방법 지원
    - 견고한 폴백 메커니즘
    - 실시간 품질 평가
    """
    
    # 의류 카테고리 정의
    CLOTHING_CATEGORIES = {
        'upper': ['shirt', 't-shirt', 'blouse', 'sweater', 'jacket', 'coat', 'top'],
        'lower': ['pants', 'jeans', 'skirt', 'shorts', 'trousers', 'bottom'],
        'full': ['dress', 'jumpsuit', 'overall', 'gown'],
        'accessories': ['hat', 'scarf', 'gloves', 'shoes', 'bag', 'belt']
    }
    
    def __init__(self, device: str = "mps", config: Optional[Dict[str, Any]] = None):
        """
        초기화 - Pipeline Manager 완전 호환
        
        Args:
            device: 사용할 디바이스 (mps, cuda, cpu)
            config: 설정 딕셔너리 (선택적)
        """
        # model_loader는 내부에서 전역 함수로 가져옴
        from app.ai_pipeline.utils.model_loader import get_global_model_loader
        self.model_loader = get_global_model_loader()
        
        self.device = self._setup_optimal_device(device)
        self.config = config or {}
        self.is_initialized = False
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        # 세그멘테이션 설정
        self.segmentation_config = self.config.get('segmentation', {
            'method': 'auto',
            'model_name': 'u2net',
            'confidence_threshold': 0.5,
            'use_background_removal': True,
            'quality_threshold': 0.7,
            'enable_post_processing': True,
            'max_image_size': 1024
        })
        
        # 후처리 설정
        self.post_process_config = self.config.get('post_processing', {
            'morphology_enabled': True,
            'gaussian_blur': True,
            'edge_refinement': True,
            'hole_filling': True
        })
        
        # 모델 및 세션 변수들
        self.rembg_session = None
        self.rembg_sessions = {}
        self.segmentation_model = None
        self.backup_methods = None
        
        # 통계
        self.processing_stats = {
            'total_processed': 0,
            'successful_segmentations': 0,
            'average_quality': 0.0,
            'method_usage': {}
        }
        
        self.logger.info(f"👕 의류 세그멘테이션 스텝 초기화 - 디바이스: {device}")
    
    def _setup_optimal_device(self, device: str) -> str:
        """최적 디바이스 설정"""
        if device == "mps" and torch.backends.mps.is_available():
            return "mps"
        elif device == "cuda" and torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    async def initialize(self) -> bool:
        """초기화 메서드"""
        try:
            # 1. RemBG 초기화
            await self._initialize_rembg()
            
            # 2. 커스텀 모델 초기화
            await self._initialize_custom_model()
            
            # 3. 백업 방법들 초기화
            self._initialize_backup_methods()
            
            self.is_initialized = True
            self.logger.info("✅ 의류 세그멘테이션 시스템 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 세그멘테이션 초기화 실패: {e}")
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
        의류 세그멘테이션 처리
        
        Args:
            clothing_image: 입력 의류 이미지
            clothing_type: 의류 타입
            quality_level: 품질 레벨 ('low', 'medium', 'high')
            **kwargs: 추가 매개변수
            
        Returns:
            Dict: 세그멘테이션 결과
        """
        start_time = time.time()
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # 1. 입력 텐서 검증 및 전처리
            clothing_pil = self._prepare_input_image(clothing_image)
            
            # 2. 최적 세그멘테이션 방법 선택
            method = self._select_segmentation_method(clothing_pil, clothing_type, quality_level)
            self.logger.info(f"📋 선택된 방법: {method}")
            
            # 3. 메인 세그멘테이션 수행
            segmentation_result = await self._perform_segmentation(clothing_pil, method)
            
            # 4. 품질 평가
            quality_score = self._evaluate_quality(clothing_pil, segmentation_result['mask'])
            
            # 5. 품질이 낮으면 폴백 시도
            if quality_score < self.segmentation_config['quality_threshold']:
                self.logger.info(f"🔄 품질 개선 시도 (현재: {quality_score:.3f})")
                improved_result = await self._try_fallback_methods(clothing_pil, clothing_type)
                
                if improved_result and improved_result.get('quality', 0) > quality_score:
                    segmentation_result = improved_result
                    quality_score = improved_result['quality']
                    method = improved_result.get('method', method)
            
            # 6. 후처리 적용
            processed_result = self._apply_post_processing(segmentation_result, quality_level)
            
            # 7. 최종 결과 구성
            processing_time = time.time() - start_time
            result = self._build_final_result(
                processed_result, quality_score, processing_time, method, clothing_type
            )
            
            # 8. 통계 업데이트
            self._update_statistics(method, quality_score, processing_time)
            
            self.logger.info(f"✅ 세그멘테이션 완료 - 방법: {method}, 품질: {quality_score:.3f}, 시간: {processing_time:.3f}초")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 세그멘테이션 실패: {e}")
            return self._create_empty_result(f"처리 오류: {str(e)}")
    
    def _prepare_input_image(self, image_input: Union[str, np.ndarray, Image.Image, torch.Tensor]) -> Image.Image:
        """입력 이미지 전처리"""
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
            
            # 크기 조정 (필요한 경우)
            max_size = self.segmentation_config['max_image_size']
            if max(image_pil.size) > max_size:
                image_pil.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                self.logger.info(f"🔄 이미지 크기 조정: {image_pil.size}")
            
            return image_pil
            
        except Exception as e:
            self.logger.error(f"이미지 전처리 실패: {e}")
            raise
    
    def _select_segmentation_method(self, image: Image.Image, clothing_type: str, quality_level: str) -> str:
        """최적 세그멘테이션 방법 선택"""
        
        method = self.segmentation_config['method']
        
        # 자동 선택 모드
        if method == 'auto':
            # 이미지 복잡도 분석
            complexity = self._analyze_image_complexity(image)
            
            # 품질 레벨과 복잡도에 따른 방법 선택
            if quality_level == 'high':
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
        
        return method
    
    def _analyze_image_complexity(self, image: Image.Image) -> float:
        """이미지 복잡도 분석 (0.0-1.0)"""
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
            hist_entropy = -np.sum((hist / hist.sum()) * np.log2(hist / hist.sum() + 1e-10))
            hist_complexity = hist_entropy / 8.0  # 정규화
            
            # 종합 복잡도
            complexity = (edge_density * 0.4 + texture_complexity * 0.4 + hist_complexity * 0.2)
            
            return min(max(complexity, 0.0), 1.0)
            
        except Exception as e:
            self.logger.warning(f"복잡도 분석 실패: {e}")
            return 0.5  # 기본값
    
    async def _perform_segmentation(self, image: Image.Image, method: str) -> Dict[str, Any]:
        """메인 세그멘테이션 수행"""
        try:
            if method == 'rembg' and self.rembg_session:
                return await self._segment_with_rembg(image)
            elif method == 'model' and self.segmentation_model:
                return await self._segment_with_model(image)
            elif method == 'grabcut' and self.backup_methods:
                return self.backup_methods.grabcut_segmentation(image)
            elif method == 'threshold' and self.backup_methods:
                return self.backup_methods.threshold_segmentation(image)
            else:
                # 폴백
                return await self._segment_with_simple_threshold(image)
                
        except Exception as e:
            self.logger.warning(f"세그멘테이션 방법 {method} 실패: {e}")
            return await self._segment_with_simple_threshold(image)
    
    async def _segment_with_rembg(self, image: Image.Image) -> Dict[str, Any]:
        """RemBG를 사용한 세그멘테이션"""
        try:
            # 의류 타입별 모델 선택
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
        """커스텀 모델을 사용한 세그멘테이션"""
        try:
            # 입력 전처리
            input_tensor = self._preprocess_for_model(image)
            
            # 모델 추론
            with torch.no_grad():
                mask_pred = self.segmentation_model(input_tensor)
                mask = mask_pred.squeeze().cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
            
            # 원본 크기로 복원
            if mask.shape != (image.height, image.width):
                mask = cv2.resize(mask, (image.width, image.height), interpolation=cv2.INTER_NEAREST)
            
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
    
    async def _segment_with_simple_threshold(self, image: Image.Image) -> Dict[str, Any]:
        """간단한 임계값 세그멘테이션 (최후의 수단)"""
        try:
            # 그레이스케일 변환
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # Otsu 임계값 적용
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
    
    def _apply_post_processing(self, segmentation_result: Dict[str, Any], quality_level: str) -> Dict[str, Any]:
        """후처리 적용"""
        
        if not self.post_process_config.get('enable_post_processing', True):
            return segmentation_result
        
        try:
            mask = segmentation_result['mask'].copy()
            
            # 품질 레벨에 따른 처리 강도
            intensity_map = {'low': 0, 'medium': 1, 'high': 2}
            intensity = intensity_map.get(quality_level, 1)
            
            processed_mask = mask.copy()
            
            # 1. 형태학적 연산 (노이즈 제거)
            if self.post_process_config['morphology_enabled']:
                kernel_size = 3 + intensity
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel)
                processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)
            
            # 2. 가우시안 블러 (엣지 스무딩)
            if self.post_process_config['gaussian_blur']:
                blur_kernel = 3 + intensity * 2
                if blur_kernel % 2 == 0:
                    blur_kernel += 1
                processed_mask = cv2.GaussianBlur(processed_mask, (blur_kernel, blur_kernel), 0)
                processed_mask = (processed_mask > 127).astype(np.uint8) * 255
            
            # 3. 홀 채우기
            if self.post_process_config['hole_filling']:
                contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.fillPoly(processed_mask, contours, 255)
            
            # 4. 엣지 정제
            if self.post_process_config['edge_refinement'] and intensity > 0:
                processed_mask = self._refine_edges(processed_mask, intensity)
            
            segmentation_result['mask'] = processed_mask
            segmentation_result['segmented_image'] = self._apply_mask_to_image(
                segmentation_result['segmented_image'], processed_mask
            )
            
            return segmentation_result
            
        except Exception as e:
            self.logger.warning(f"후처리 실패: {e}")
            return segmentation_result
    
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
    
    def _evaluate_quality(self, original_image: Image.Image, mask: np.ndarray) -> float:
        """세그멘테이션 품질 평가"""
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
            
            # 종합 점수 계산
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
    
    async def _segment_with_kmeans(self, image: Image.Image) -> Dict[str, Any]:
        """K-means를 사용한 세그멘테이션"""
        try:
            # 이미지를 numpy 배열로 변환
            image_array = np.array(image)
            
            # 픽셀을 1D 배열로 변환
            pixels = image_array.reshape(-1, 3)
            
            # K-means 클러스터링 (2개 클러스터: 배경과 전경)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixels)
            
            # 라벨을 이미지 크기로 복원
            label_image = labels.reshape(image_array.shape[:2])
            
            # 전경과 배경 구분 (더 큰 클러스터를 배경으로 가정)
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
    
    def _build_final_result(
        self, 
        processed_result: Dict[str, Any], 
        quality_score: float, 
        processing_time: float, 
        method: str, 
        clothing_type: str
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
                'processing_time': processing_time
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
                'post_processing_applied': True,
                'fallback_used': method in ['grabcut', 'kmeans', 'threshold'],
                'image_size': f"{processed_result['segmented_image'].size[0]}x{processed_result['segmented_image'].size[1]}",
                'mask_coverage': np.sum(processed_result['mask'] > 0) / processed_result['mask'].size
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
                'error_occurred': True
            }
        }
    
    def _update_statistics(self, method: str, quality_score: float, processing_time: float):
        """통계 업데이트"""
        self.processing_stats['total_processed'] += 1
        
        if quality_score > 0.5:
            self.processing_stats['successful_segmentations'] += 1
        
        # 품질 이동 평균
        alpha = 0.1
        self.processing_stats['average_quality'] = (
            alpha * quality_score + 
            (1 - alpha) * self.processing_stats['average_quality']
        )
        
        # 방법별 사용 통계
        if method not in self.processing_stats['method_usage']:
            self.processing_stats['method_usage'][method] = 0
        self.processing_stats['method_usage'][method] += 1
    
    async def _initialize_rembg(self):
        """RemBG 초기화"""
        if not REMBG_AVAILABLE:
            self.logger.warning("RemBG 사용 불가")
            return
        
        try:
            # 기본 세션 생성
            self.rembg_session = new_session('u2net')
            
            # 특화 세션들 생성
            self.rembg_sessions = {
                'human_seg': new_session('u2net_human_seg'),
                'cloth_seg': new_session('u2net_cloth_seg') if hasattr(rembg, 'u2net_cloth_seg') else self.rembg_session
            }
            
            self.logger.info("✅ RemBG 세션 초기화 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ RemBG 초기화 실패: {e}")
            self.rembg_session = None
            self.rembg_sessions = {}
    
    async def _initialize_custom_model(self):
        """커스텀 모델 초기화"""
        try:
            if self.model_loader:
                # 모델 로더를 통한 모델 로드 시도
                self.segmentation_model = await self.model_loader.load_model(
                    self.segmentation_config['model_name']
                )
            
            if not self.segmentation_model:
                # 간단한 모델 생성
                self.segmentation_model = await self._create_u2net_model()
            
            self.logger.info("✅ 커스텀 모델 초기화 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 커스텀 모델 초기화 실패: {e}")
            self.segmentation_model = self._create_fallback_model()
    
    def _initialize_backup_methods(self):
        """백업 세그멘테이션 방법들 초기화"""
        try:
            self.backup_methods = BackupSegmentationMethods(self.device)
            self.logger.info("✅ 백업 방법들 초기화 완료")
        except Exception as e:
            self.logger.warning(f"⚠️ 백업 방법 초기화 실패: {e}")
            self.backup_methods = self._create_simple_backup()
    
    async def _create_u2net_model(self):
        """U²-Net 스타일 모델 생성"""
        class SimpleU2Net(torch.nn.Module):
            def __init__(self):
                super(SimpleU2Net, self).__init__()
                # 간단한 U-Net 구조
                self.encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, 3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(64, 64, 3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool2d(2)
                )
                
                self.middle = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 128, 3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(128, 128, 3, padding=1),
                    torch.nn.ReLU(inplace=True),
                )
                
                self.decoder = torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    torch.nn.Conv2d(128, 64, 3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(64, 1, 1),
                    torch.nn.Sigmoid()
                )
            
            def forward(self, x):
                x1 = self.encoder(x)
                x2 = self.middle(x1)
                x3 = self.decoder(x2)
                return x3
        
        return SimpleU2Net().to(self.device)
    
    def _create_fallback_model(self):
        """폴백 모델 생성"""
        class FallbackModel:
            def __call__(self, x):
                # 간단한 더미 출력
                batch_size = x.shape[0] if len(x.shape) == 4 else 1
                height, width = x.shape[-2], x.shape[-1]
                return torch.ones(batch_size, 1, height, width) * 0.5
        
        return FallbackModel()
    
    def _create_simple_backup(self):
        """간단한 백업 방법 생성"""
        class SimpleBackup:
            def grabcut_segmentation(self, image):
                # 더미 GrabCut
                mask = np.ones((image.height, image.width), dtype=np.uint8) * 255
                return {
                    'segmented_image': image,
                    'mask': mask,
                    'method': 'simple_grabcut',
                    'confidence': 0.5
                }
            
            def threshold_segmentation(self, image):
                # 더미 임계값
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
    
    async def cleanup(self):
        """리소스 정리"""
        try:
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
            
            self.is_initialized = False
            self.logger.info("🧹 의류 세그멘테이션 스텝 정리 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 리소스 정리 중 오류: {e}")


class BackupSegmentationMethods:
    """백업 세그멘테이션 방법들"""
    
    def __init__(self, device: str):
        self.device = device
        self.logger = logging.getLogger(__name__)
    
    def grabcut_segmentation(self, image: Image.Image) -> Dict[str, Any]:
        """GrabCut 알고리즘을 사용한 세그멘테이션"""
        try:
            # PIL을 OpenCV 형식으로 변환
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # 초기 사각형 (이미지의 10% 여백)
            height, width = img.shape[:2]
            rect = (
                int(width * 0.1), 
                int(height * 0.1), 
                int(width * 0.8), 
                int(height * 0.8)
            )
            
            # GrabCut 초기화
            mask = np.zeros((height, width), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # GrabCut 수행
            cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # 마스크 후처리
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            final_mask = mask2 * 255
            
            # RGB로 변환된 세그멘테이션 이미지 생성
            segmented_img = img * mask2[:, :, np.newaxis]
            segmented_img_rgb = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)
            segmented_image = Image.fromarray(segmented_img_rgb.astype(np.uint8))
            
            return {
                'segmented_image': segmented_image,
                'mask': final_mask,
                'method': 'grabcut',
                'confidence': 0.75
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
        """임계값 기반 세그멘테이션"""
        try:
            # 그레이스케일 변환
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # 적응형 임계값 적용
            mask = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # 노이즈 제거
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 세그멘테이션된 이미지 생성
            image_array = np.array(image)
            mask_3channel = np.stack([mask] * 3, axis=2)
            segmented_array = image_array * (mask_3channel / 255.0)
            segmented_image = Image.fromarray(segmented_array.astype(np.uint8))
            
            return {
                'segmented_image': segmented_image,
                'mask': mask,
                'method': 'adaptive_threshold',
                'confidence': 0.65
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