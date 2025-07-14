# app/ai_pipeline/steps/step_03_cloth_segmentation.py
"""
3단계: 의류 세그멘테이션 (Clothing Segmentation) - 배경 제거
Pipeline Manager와 완전 호환되는 수정된 버전
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
        
        # 세그멘테이션 설정
        self.segmentation_config = self.config.get('segmentation', {
            'method': 'auto',
            'model_name': 'u2net',
            'post_processing': True,
            'edge_refinement': True,
            'quality_threshold': 0.6,
            'fallback_methods': ['rembg', 'grabcut', 'threshold'],
            'use_ensemble': False
        })
        
        # 후처리 설정
        self.post_process_config = self.config.get('post_processing', {
            'morphology_enabled': True,
            'gaussian_blur': True,
            'edge_smoothing': True,
            'noise_removal': True,
            'bilateral_filter': False  # 속도 최적화
        })
        
        # 성능 설정
        self.performance_config = self.config.get('performance', {
            'use_mps': self.device == 'mps',
            'memory_efficient': True,
            'async_processing': True,
            'max_resolution': 1024,
            'cache_models': True
        })
        
        # 모델 인스턴스들
        self.rembg_session = None
        self.rembg_sessions = {}
        self.segmentation_model = None
        self.backup_methods = None
        
        # 통계 및 상태
        self.is_initialized = False
        self.initialization_error = None
        self.performance_stats = {
            'total_processed': 0,
            'average_time': 0.0,
            'success_rate': 0.0,
            'method_usage': {}
        }
        
        # 스레드 풀 (비동기 처리용)
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
        logger.info(f"🎯 ClothSegmentationStep 초기화 - 디바이스: {self.device}")
    
    def _setup_optimal_device(self, preferred_device: str) -> str:
        """최적 디바이스 선택 (M3 Max 우선)"""
        try:
            if preferred_device == 'mps' and torch.backends.mps.is_available():
                logger.info("✅ Apple Silicon MPS 백엔드 활성화")
                return 'mps'
            elif preferred_device == 'cuda' and torch.cuda.is_available():
                logger.info("✅ CUDA 백엔드 활성화")
                return 'cuda'
            else:
                logger.info("⚠️ CPU 백엔드 사용 (속도가 느릴 수 있음)")
                return 'cpu'
        except Exception as e:
            logger.warning(f"디바이스 설정 실패: {e}, CPU 사용")
            return 'cpu'
    
    async def initialize(self) -> bool:
        """
        세그멘테이션 시스템 초기화
        Pipeline Manager가 호출하는 표준 초기화 메서드
        """
        try:
            logger.info("🔄 의류 세그멘테이션 시스템 초기화 시작...")
            
            # 1. RemBG 모델 초기화 (우선순위)
            await self._initialize_rembg_models()
            
            # 2. 커스텀 세그멘테이션 모델 초기화
            await self._initialize_custom_models()
            
            # 3. 백업 방법들 초기화
            self._initialize_backup_methods()
            
            # 4. 시스템 검증
            await self._validate_system()
            
            # 5. 모델 워밍업
            await self._warmup_models()
            
            self.is_initialized = True
            logger.info("✅ 의류 세그멘테이션 시스템 초기화 완료")
            
            return True
            
        except Exception as e:
            error_msg = f"의류 세그멘테이션 초기화 실패: {e}"
            logger.error(f"❌ {error_msg}")
            self.initialization_error = error_msg
            self.is_initialized = False
            return False
    
    async def _initialize_rembg_models(self):
        """RemBG 모델들 초기화"""
        if not REMBG_AVAILABLE:
            logger.warning("⚠️ RemBG 사용 불가 - 대안 방법 사용")
            return
        
        try:
            model_name = self.segmentation_config['model_name']
            logger.info(f"📦 RemBG 모델 로딩: {model_name}")
            
            # 메인 모델 로드
            self.rembg_session = new_session(model_name)
            self.rembg_sessions[model_name] = self.rembg_session
            
            # 의류별 특화 모델들 (리소스가 허용하는 경우)
            specialized_models = {
                'human_seg': 'u2net_human_seg',
                'cloth': 'silueta'
            }
            
            for name, model in specialized_models.items():
                try:
                    if model != model_name:  # 중복 로드 방지
                        session = new_session(model)
                        self.rembg_sessions[name] = session
                        logger.info(f"✅ 특화 모델 로드: {name} ({model})")
                except Exception as e:
                    logger.warning(f"⚠️ 특화 모델 {name} 로드 실패: {e}")
            
            logger.info("✅ RemBG 모델 초기화 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ RemBG 초기화 실패: {e}")
            self.rembg_session = None
    
    async def _initialize_custom_models(self):
        """커스텀 세그멘테이션 모델 초기화"""
        try:
            model_type = self.config.get('model_type', 'simple')
            
            if model_type == 'u2net':
                self.segmentation_model = await self._create_u2net_model()
            else:
                self.segmentation_model = self._create_simple_model()
            
            logger.info("✅ 커스텀 모델 초기화 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ 커스텀 모델 초기화 실패: {e}")
            self.segmentation_model = self._create_fallback_model()
    
    def _initialize_backup_methods(self):
        """백업 세그멘테이션 방법들 초기화"""
        try:
            self.backup_methods = BackupSegmentationMethods(self.device)
            logger.info("✅ 백업 방법들 초기화 완료")
        except Exception as e:
            logger.warning(f"⚠️ 백업 방법 초기화 실패: {e}")
            self.backup_methods = None
    
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
                out = self.decoder(x2)
                return out
        
        model = SimpleU2Net().to(self.device)
        model.eval()
        
        # MPS 최적화
        if self.device == 'mps':
            try:
                model = torch.jit.optimize_for_inference(model)
            except:
                pass  # 실패해도 계속 진행
        
        return model
    
    def _create_simple_model(self):
        """간단한 세그멘테이션 모델"""
        class SimpleSegModel(torch.nn.Module):
            def __init__(self):
                super(SimpleSegModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = torch.nn.Conv2d(32, 16, 3, padding=1)
                self.conv3 = torch.nn.Conv2d(16, 1, 1)
                
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = torch.sigmoid(self.conv3(x))
                return x
        
        model = SimpleSegModel().to(self.device)
        model.eval()
        return model
    
    def _create_fallback_model(self):
        """최소 기능 폴백 모델"""
        class FallbackModel(torch.nn.Module):
            def forward(self, x):
                # 간단한 밝기 기반 마스크 생성
                gray = torch.mean(x, dim=1, keepdim=True)
                mask = (gray > 0.3).float()
                return mask
        
        return FallbackModel().to(self.device)
    
    async def _validate_system(self):
        """시스템 검증"""
        available_methods = []
        
        if self.rembg_session:
            available_methods.append('rembg')
        if self.segmentation_model:
            available_methods.append('model')
        if self.backup_methods:
            available_methods.append('backup')
        
        if not available_methods:
            raise RuntimeError("사용 가능한 세그멘테이션 방법이 없습니다")
        
        logger.info(f"✅ 사용 가능한 방법들: {available_methods}")
    
    async def _warmup_models(self):
        """모델 워밍업 (성능 최적화)"""
        try:
            # 더미 입력 생성
            dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
            
            # 커스텀 모델 워밍업
            if self.segmentation_model:
                with torch.no_grad():
                    _ = self.segmentation_model(dummy_input)
            
            logger.info("🔥 모델 워밍업 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ 모델 워밍업 실패: {e}")
    
    # =================================================================
    # 메인 처리 메서드 - Pipeline Manager 호환 인터페이스
    # =================================================================
    
    def process(
        self, 
        clothing_image_tensor: torch.Tensor,
        clothing_type: str = "shirt",
        quality_level: str = "medium"
    ) -> Dict[str, Any]:
        """
        의류 세그멘테이션 처리 (동기 버전)
        Pipeline Manager가 호출하는 메인 메서드
        """
        return asyncio.run(self.process_async(clothing_image_tensor, clothing_type, quality_level))
    
    async def process_async(
        self, 
        clothing_image_tensor: torch.Tensor,
        clothing_type: str = "shirt",
        quality_level: str = "medium"
    ) -> Dict[str, Any]:
        """
        의류 세그멘테이션 비동기 처리
        
        Args:
            clothing_image_tensor: 의류 이미지 텐서 [1, 3, H, W] 또는 [3, H, W]
            clothing_type: 의류 타입 (shirt, pants, dress 등)
            quality_level: 품질 레벨 (low, medium, high)
            
        Returns:
            세그멘테이션 결과 딕셔너리
        """
        if not self.is_initialized:
            error_msg = f"세그멘테이션 시스템이 초기화되지 않음: {self.initialization_error}"
            logger.error(f"❌ {error_msg}")
            return self._create_error_result(error_msg)
        
        start_time = time.time()
        
        try:
            logger.info(f"🔍 의류 세그멘테이션 시작 - 타입: {clothing_type}, 품질: {quality_level}")
            
            # 1. 입력 텐서 검증 및 전처리
            clothing_pil = self._prepare_input_image(clothing_image_tensor)
            
            # 2. 최적 세그멘테이션 방법 선택
            method = self._select_segmentation_method(clothing_pil, clothing_type, quality_level)
            logger.info(f"📋 선택된 방법: {method}")
            
            # 3. 메인 세그멘테이션 수행
            segmentation_result = await self._perform_segmentation(clothing_pil, method)
            
            # 4. 품질 평가
            quality_score = self._evaluate_quality(clothing_pil, segmentation_result['mask'])
            
            # 5. 품질이 낮으면 폴백 시도
            if quality_score < self.segmentation_config['quality_threshold']:
                logger.info(f"🔄 품질 개선 시도 (현재: {quality_score:.3f})")
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
            self._update_performance_stats(method, processing_time, quality_score)
            
            logger.info(f"✅ 세그멘테이션 완료 - {processing_time:.3f}초, 품질: {quality_score:.3f}")
            return result
            
        except Exception as e:
            error_msg = f"세그멘테이션 처리 실패: {e}"
            logger.error(f"❌ {error_msg}")
            return self._create_error_result(error_msg)
    
    def _prepare_input_image(self, tensor: torch.Tensor) -> Image.Image:
        """입력 텐서를 PIL 이미지로 변환"""
        try:
            # 텐서 차원 정규화
            if tensor.dim() == 4 and tensor.size(0) == 1:
                tensor = tensor.squeeze(0)  # [1, 3, H, W] -> [3, H, W]
            elif tensor.dim() == 3 and tensor.size(0) == 3:
                pass  # [3, H, W] - 올바른 형태
            else:
                raise ValueError(f"지원하지 않는 텐서 형태: {tensor.shape}")
            
            # [3, H, W] -> [H, W, 3]
            tensor = tensor.permute(1, 2, 0)
            
            # 값 범위 정규화
            if tensor.max() <= 1.0:
                tensor = tensor * 255
            
            # NumPy 배열로 변환
            array = tensor.cpu().numpy().astype(np.uint8)
            
            # PIL 이미지로 변환
            pil_image = Image.fromarray(array)
            
            # 크기 제한 (메모리 효율성)
            max_size = self.performance_config['max_resolution']
            if max(pil_image.size) > max_size:
                pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            return pil_image
            
        except Exception as e:
            logger.error(f"입력 이미지 준비 실패: {e}")
            raise ValueError(f"입력 텐서 처리 실패: {e}")
    
    def _select_segmentation_method(self, image: Image.Image, clothing_type: str, quality_level: str) -> str:
        """최적 세그멘테이션 방법 선택"""
        method = self.segmentation_config['method']
        
        if method == 'auto':
            # 이미지 복잡도 기반 자동 선택
            complexity = self._analyze_image_complexity(image)
            
            if REMBG_AVAILABLE and self.rembg_session:
                if quality_level == 'high' or complexity > 0.7:
                    return 'rembg'
                elif complexity < 0.3:
                    return 'rembg'
            
            if self.segmentation_model and complexity > 0.4:
                return 'model'
            
            return 'grabcut'  # 기본 백업 방법
        
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
            logger.warning(f"복잡도 분석 실패: {e}")
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
            logger.warning(f"세그멘테이션 방법 {method} 실패: {e}")
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
            logger.warning(f"RemBG 처리 실패: {e}")
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
            logger.warning(f"모델 세그멘테이션 실패: {e}")
            raise
    
    async def _segment_with_simple_threshold(self, image: Image.Image) -> Dict[str, Any]:
        """간단한 임계값 세그멘테이션 (최후의 수단)"""
        try:
            # 그레이스케일 변환
            gray = np.array(image.convert('L'))
            
            # Otsu 임계값
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 배경이 밝은 경우 반전
            if np.mean(mask) > 127:
                mask = 255 - mask
            
            # 세그멘테이션된 이미지 생성
            segmented_image = self._apply_mask_to_image(image, mask)
            
            return {
                'segmented_image': segmented_image,
                'mask': mask,
                'method': 'threshold',
                'confidence': 0.6
            }
            
        except Exception as e:
            logger.error(f"간단한 임계값 세그멘테이션 실패: {e}")
            # 최후의 폴백 - 전체 이미지를 전경으로
            h, w = image.size
            mask = np.ones((w, h), dtype=np.uint8) * 255
            return {
                'segmented_image': image,
                'mask': mask,
                'method': 'fallback',
                'confidence': 0.3
            }
    
    async def _try_fallback_methods(self, image: Image.Image, clothing_type: str) -> Optional[Dict[str, Any]]:
        """폴백 방법들 시도"""
        fallback_methods = self.segmentation_config['fallback_methods']
        best_result = None
        best_quality = 0.0
        
        for method in fallback_methods:
            try:
                result = await self._perform_segmentation(image, method)
                if result:
                    quality = self._evaluate_quality(image, result['mask'])
                    result['quality'] = quality
                    
                    if quality > best_quality:
                        best_quality = quality
                        best_result = result
                        
            except Exception as e:
                logger.warning(f"폴백 방법 {method} 실패: {e}")
                continue
        
        return best_result
    
    def _evaluate_quality(self, original: Image.Image, mask: np.ndarray) -> float:
        """세그멘테이션 품질 평가 (0.0-1.0)"""
        try:
            # 기본 검증
            if mask is None or mask.size == 0:
                return 0.0
            
            # 연결성 분석
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.0
            
            # 가장 큰 연결 요소
            main_contour = max(contours, key=cv2.contourArea)
            main_area = cv2.contourArea(main_contour)
            total_mask_area = np.sum(mask > 0)
            
            # 연결성 점수
            connectivity_score = main_area / (total_mask_area + 1e-6)
            
            # 크기 적절성
            image_area = original.width * original.height
            size_ratio = total_mask_area / image_area
            size_score = 1.0 if 0.05 <= size_ratio <= 0.8 else max(0.0, 1.0 - abs(size_ratio - 0.4) * 2)
            
            # 엣지 품질
            edge_score = self._evaluate_edge_quality(mask)
            
            # 종합 점수
            quality = (connectivity_score * 0.4 + size_score * 0.3 + edge_score * 0.3)
            
            return min(max(quality, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"품질 평가 실패: {e}")
            return 0.5
    
    def _evaluate_edge_quality(self, mask: np.ndarray) -> float:
        """엣지 품질 평가"""
        try:
            # 엣지 검출
            edges = cv2.Canny(mask, 50, 150)
            
            # 엣지 부드러움 측정
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.5
            
            main_contour = max(contours, key=cv2.contourArea)
            
            # 윤곽선 근사화로 부드러움 측정
            epsilon = 0.02 * cv2.arcLength(main_contour, True)
            approx = cv2.approxPolyDP(main_contour, epsilon, True)
            
            if len(main_contour) > 0:
                smoothness = 1.0 - (len(approx) / len(main_contour))
            else:
                smoothness = 0.5
            
            return max(0.0, min(1.0, smoothness))
            
        except Exception as e:
            logger.warning(f"엣지 품질 평가 실패: {e}")
            return 0.5
    
    def _apply_post_processing(self, segmentation_result: Dict[str, Any], quality_level: str) -> Dict[str, Any]:
        """후처리 적용"""
        try:
            mask = segmentation_result['mask']
            
            # 품질 레벨에 따른 처리 강도
            intensity_map = {'low': 1, 'medium': 2, 'high': 3}
            intensity = intensity_map.get(quality_level, 2)
            
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
            
            # 3. 노이즈 제거 (작은 연결 요소 제거)
            if self.post_process_config['noise_removal']:
                processed_mask = self._remove_small_components(processed_mask, intensity)
            
            # 마스크를 텐서로 변환 (Pipeline Manager 호환)
            mask_tensor = torch.from_numpy(processed_mask).float() / 255.0
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
            
            # 처리된 세그멘테이션 이미지 생성
            processed_segmented = self._apply_mask_to_image(
                segmentation_result['segmented_image'], processed_mask
            )
            
            return {
                'segmented_image': processed_segmented,
                'mask_tensor': mask_tensor,
                'binary_mask': processed_mask,
                'confidence_map': self._generate_confidence_map(processed_mask)
            }
            
        except Exception as e:
            logger.warning(f"후처리 실패: {e}")
            # 원본 결과 반환
            mask = segmentation_result['mask']
            mask_tensor = torch.from_numpy(mask).float() / 255.0
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
            
            return {
                'segmented_image': segmentation_result['segmented_image'],
                'mask_tensor': mask_tensor,
                'binary_mask': mask,
                'confidence_map': None
            }
    
    def _remove_small_components(self, mask: np.ndarray, intensity: int) -> np.ndarray:
        """작은 연결 요소 제거"""
        try:
            # 연결 성분 분석
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            if num_labels <= 1:
                return mask
            
            # 가장 큰 연결 성분 찾기
            largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            
            # 최소 크기 임계값
            min_area = stats[largest_component, cv2.CC_STAT_AREA] * 0.05 / intensity
            
            # 큰 성분들만 유지
            cleaned_mask = np.zeros_like(mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    cleaned_mask[labels == i] = 255
            
            return cleaned_mask
            
        except Exception as e:
            logger.warning(f"작은 성분 제거 실패: {e}")
            return mask
    
    def _generate_confidence_map(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """신뢰도 맵 생성"""
        try:
            # 거리 변환으로 중심부일수록 높은 신뢰도
            dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            
            # 정규화
            if dist_transform.max() > 0:
                confidence_map = dist_transform / dist_transform.max()
            else:
                confidence_map = np.zeros_like(mask, dtype=np.float32)
            
            return confidence_map
            
        except Exception as e:
            logger.warning(f"신뢰도 맵 생성 실패: {e}")
            return None
    
    def _build_final_result(
        self,
        processed_result: Dict[str, Any],
        quality_score: float,
        processing_time: float,
        method: str,
        clothing_type: str
    ) -> Dict[str, Any]:
        """최종 결과 구성 (Pipeline Manager 호환 형식)"""
        
        return {
            "success": True,
            "segmented_image": processed_result['segmented_image'],
            "clothing_mask": processed_result['mask_tensor'],
            "binary_mask": processed_result['binary_mask'],
            "confidence_map": processed_result.get('confidence_map'),
            "segmentation_quality": quality_score,
            "clothing_analysis": {
                "dominant_colors": self._extract_dominant_colors(processed_result['segmented_image']),
                "clothing_area": self._calculate_clothing_area(processed_result['binary_mask']),
                "edge_complexity": self._calculate_edge_complexity(processed_result['binary_mask']),
                "background_removed": True,
                "clothing_type": clothing_type
            },
            "processing_info": {
                "method_used": method,
                "processing_time": processing_time,
                "device": self.device,
                "post_processing_applied": True,
                "quality_level": "good" if quality_score > 0.7 else "medium" if quality_score > 0.5 else "low"
            }
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """에러 결과 생성"""
        return {
            "success": False,
            "error": error_message,
            "segmented_image": None,
            "clothing_mask": None,
            "binary_mask": None,
            "segmentation_quality": 0.0,
            "clothing_analysis": {},
            "processing_info": {
                "method_used": "error",
                "processing_time": 0.0,
                "device": self.device,
                "error_details": error_message
            }
        }
    
    # =================================================================
    # 유틸리티 메서드들
    # =================================================================
    
    def _preprocess_for_model(self, image: Image.Image) -> torch.Tensor:
        """모델 입력을 위한 전처리"""
        try:
            # 크기 조정
            input_size = (256, 256)
            resized = image.resize(input_size, Image.Resampling.LANCZOS)
            
            # 텐서로 변환
            tensor = torch.from_numpy(np.array(resized)).float() / 255.0
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            
            # 정규화 (ImageNet 표준)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            tensor = (tensor - mean) / std
            
            return tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"모델 전처리 실패: {e}")
            raise
    
    def _apply_mask_to_image(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """마스크를 이미지에 적용하여 배경 제거"""
        try:
            img_array = np.array(image)
            
            # 3채널 마스크 생성
            if len(mask.shape) == 2:
                mask_3ch = np.stack([mask] * 3, axis=2) / 255.0
            else:
                mask_3ch = mask / 255.0
            
            # 배경을 흰색으로 설정
            background = np.ones_like(img_array) * 255
            
            # 마스크 적용
            result = img_array * mask_3ch + background * (1 - mask_3ch)
            
            return Image.fromarray(result.astype(np.uint8))
            
        except Exception as e:
            logger.warning(f"마스크 적용 실패: {e}")
            return image
    
    def _extract_dominant_colors(self, image: Image.Image) -> List[List[int]]:
        """주요 색상 추출"""
        try:
            if not SKLEARN_AVAILABLE:
                return [[128, 128, 128]]  # 기본 회색
            
            img_array = np.array(image.resize((100, 100)))  # 속도 최적화
            pixels = img_array.reshape(-1, 3)
            
            # 배경색 제거 (흰색 근처)
            non_white_pixels = pixels[np.sum(pixels, axis=1) < 700]
            
            if len(non_white_pixels) < 10:
                return [[128, 128, 128]]
            
            # K-means로 주요 색상 추출
            n_colors = min(3, len(non_white_pixels) // 50)
            if n_colors < 1:
                n_colors = 1
                
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(non_white_pixels)
            
            return kmeans.cluster_centers_.astype(int).tolist()
            
        except Exception as e:
            logger.warning(f"색상 추출 실패: {e}")
            return [[128, 128, 128]]
    
    def _calculate_clothing_area(self, mask: np.ndarray) -> Dict[str, float]:
        """의류 영역 계산"""
        try:
            total_pixels = mask.size
            clothing_pixels = np.sum(mask > 0)
            
            return {
                'total_pixels': float(total_pixels),
                'clothing_pixels': float(clothing_pixels),
                'coverage_ratio': float(clothing_pixels / total_pixels),
                'area_score': min(1.0, clothing_pixels / 40000)  # 정규화
            }
            
        except Exception as e:
            logger.warning(f"영역 계산 실패: {e}")
            return {'total_pixels': 0.0, 'clothing_pixels': 0.0, 'coverage_ratio': 0.0, 'area_score': 0.0}
    
    def _calculate_edge_complexity(self, mask: np.ndarray) -> Dict[str, float]:
        """엣지 복잡도 계산"""
        try:
            # 엣지 검출
            edges = cv2.Canny(mask, 50, 150)
            edge_pixels = np.sum(edges > 0)
            
            # 윤곽선 분석
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                perimeter = cv2.arcLength(main_contour, True)
                area = cv2.contourArea(main_contour)
                
                # 복잡도 계산
                if area > 0:
                    complexity = (perimeter * perimeter) / (4 * np.pi * area)
                else:
                    complexity = 0.0
            else:
                perimeter = 0.0
                complexity = 0.0
            
            return {
                'edge_pixels': float(edge_pixels),
                'perimeter': float(perimeter),
                'complexity_ratio': float(complexity),
                'edge_density': float(edge_pixels / mask.size)
            }
            
        except Exception as e:
            logger.warning(f"복잡도 계산 실패: {e}")
            return {'edge_pixels': 0.0, 'perimeter': 0.0, 'complexity_ratio': 0.0, 'edge_density': 0.0}
    
    def _update_performance_stats(self, method: str, processing_time: float, quality_score: float):
        """성능 통계 업데이트"""
        try:
            self.performance_stats['total_processed'] += 1
            
            # 평균 처리 시간 업데이트
            total = self.performance_stats['total_processed']
            current_avg = self.performance_stats['average_time']
            self.performance_stats['average_time'] = (current_avg * (total - 1) + processing_time) / total
            
            # 방법별 사용 통계
            if method not in self.performance_stats['method_usage']:
                self.performance_stats['method_usage'][method] = 0
            self.performance_stats['method_usage'][method] += 1
            
            # 성공률 업데이트
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
            "step_name": "ClothSegmentation",
            "version": "3.0",
            "device": self.device,
            "initialized": self.is_initialized,
            "initialization_error": self.initialization_error,
            "available_methods": ["rembg", "model", "grabcut", "threshold"],
            "rembg_available": REMBG_AVAILABLE and bool(self.rembg_session),
            "custom_model_available": bool(self.segmentation_model),
            "backup_methods_available": bool(self.backup_methods),
            "performance_stats": self.performance_stats,
            "config": {
                "segmentation": self.segmentation_config,
                "post_processing": self.post_process_config,
                "performance": self.performance_config
            }
        }
    
    async def cleanup(self):
        """리소스 정리 (Pipeline Manager 호환)"""
        try:
            logger.info("🧹 의류 세그멘테이션 리소스 정리 시작...")
            
            # 모델들 정리
            if self.segmentation_model:
                del self.segmentation_model
                self.segmentation_model = None
            
            # RemBG 세션들 정리
            for session in self.rembg_sessions.values():
                try:
                    del session
                except:
                    pass
            self.rembg_sessions.clear()
            self.rembg_session = None
            
            # 백업 방법들 정리
            if self.backup_methods:
                del self.backup_methods
                self.backup_methods = None
            
            # 스레드 풀 종료
            self.thread_pool.shutdown(wait=True)
            
            # GPU 메모리 정리
            if self.device == 'mps':
                torch.mps.empty_cache()
            elif self.device == 'cuda':
                torch.cuda.empty_cache()
            
            self.is_initialized = False
            logger.info("✅ 의류 세그멘테이션 리소스 정리 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ 리소스 정리 중 오류: {e}")


# =================================================================
# 백업 세그멘테이션 방법들
# =================================================================

class BackupSegmentationMethods:
    """백업 세그멘테이션 방법들 (RemBG 없이도 동작)"""
    
    def __init__(self, device: str):
        self.device = device
    
    def grabcut_segmentation(self, image: Image.Image) -> Dict[str, Any]:
        """GrabCut 세그멘테이션"""
        try:
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            h, w = img_cv.shape[:2]
            
            # 초기 사각형 (중앙 80% 영역)
            margin_h, margin_w = int(h * 0.1), int(w * 0.1)
            rect = (margin_w, margin_h, w - 2*margin_w, h - 2*margin_h)
            
            # GrabCut 초기화
            mask = np.zeros((h, w), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # GrabCut 실행
            cv2.grabCut(img_cv, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # 마스크 후처리
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') * 255
            
            # 결과 이미지
            result_cv = img_cv.copy()
            result_cv[mask2 == 0] = [255, 255, 255]
            result_rgb = cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)
            
            return {
                'segmented_image': result_pil,
                'mask': mask2,
                'method': 'grabcut',
                'confidence': 0.7
            }
            
        except Exception as e:
            logger.warning(f"GrabCut 실패: {e}")
            return self.threshold_segmentation(image)
    
    def threshold_segmentation(self, image: Image.Image) -> Dict[str, Any]:
        """임계값 기반 세그멘테이션 (최후의 수단)"""
        try:
            # 그레이스케일 변환
            gray = np.array(image.convert('L'))
            
            # Otsu 임계값
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 배경이 밝은 경우 반전
            if np.mean(binary) > 127:
                binary = 255 - binary
            
            # 형태학적 정리
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # 결과 이미지
            img_array = np.array(image)
            result = img_array.copy()
            result[binary == 0] = [255, 255, 255]
            result_pil = Image.fromarray(result)
            
            return {
                'segmented_image': result_pil,
                'mask': binary,
                'method': 'threshold',
                'confidence': 0.5
            }
            
        except Exception as e:
            logger.error(f"임계값 세그멘테이션 실패: {e}")
            # 최후의 폴백
            h, w = image.size
            mask = np.ones((w, h), dtype=np.uint8) * 255
            return {
                'segmented_image': image,
                'mask': mask,
                'method': 'emergency_fallback',
                'confidence': 0.3
            }