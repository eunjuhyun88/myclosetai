# app/ai_pipeline/steps/step_03_cloth_segmentation.py
"""
3단계: 의류 세그멘테이션 (Clothing Segmentation) - 배경 제거
기존 코드를 통합하고 실제 작동하도록 개선한 버전
U²-Net + RemBG + 백업 방법들을 통합하여 M3 Max에서 최적 성능 제공
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

# 배경 제거 라이브러리들
try:
    import rembg
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    logging.warning("rembg 설치 필요: pip install rembg")

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

class ClothingSegmentationStep:
    """
    의류 세그멘테이션 스텝 - 기존 두 파일의 장점을 통합
    - M3 Max MPS 최적화
    - RemBG + U²-Net + 백업 방법들 통합
    - 품질 기반 자동 선택 및 폴백
    - 실시간 성능 모니터링
    """
    
    # 의류 카테고리 (기존 코드 유지)
    CLOTHING_CATEGORIES = {
        'upper': ['shirt', 't-shirt', 'blouse', 'sweater', 'jacket', 'coat', 'dress'],
        'lower': ['pants', 'jeans', 'skirt', 'shorts', 'trousers'],
        'full': ['dress', 'jumpsuit', 'overall'],
        'accessories': ['hat', 'scarf', 'gloves', 'shoes', 'bag']
    }
    
    def __init__(self, model_loader, device: str, config: Dict[str, Any] = None):
        """
        Args:
            model_loader: 모델 로더 인스턴스 (기존 코드와 호환)
            device: 사용할 디바이스
            config: 설정 딕셔너리
        """
        self.model_loader = model_loader
        self.device = self._setup_optimal_device(device)
        self.config = config or {}
        
        # 세그멘테이션 설정 (두 파일의 설정 통합)
        self.segmentation_config = self.config.get('segmentation', {
            'method': 'auto',  # auto, rembg, u2net, grabcut, kmeans, threshold
            'model_name': 'u2net',  # rembg 모델명
            'post_processing': True,
            'edge_refinement': True,
            'multi_scale': False,
            'quality_threshold': 0.7,
            'fallback_methods': ['rembg', 'u2net', 'grabcut', 'threshold'],
            'use_ensemble': False  # 고품질 모드에서만 활성화
        })
        
        # 후처리 설정 (기존 코드 확장)
        self.post_process_config = self.config.get('post_processing', {
            'morphology_enabled': True,
            'gaussian_blur': True,
            'edge_smoothing': True,
            'noise_removal': True,
            'bilateral_filter': True,
            'alpha_matting': False  # 고급 기능
        })
        
        # 성능 최적화 설정 (M3 Max 특화)
        self.performance_config = self.config.get('performance', {
            'use_mps': self.device == 'mps',
            'batch_processing': True,
            'memory_efficient': True,
            'async_processing': True,
            'cache_models': True,
            'max_resolution': 1024
        })
        
        # 모델들 (기존 코드 구조 유지)
        self.rembg_session = None
        self.rembg_sessions = {}  # 다중 모델 지원
        self.segmentation_model = None  # U²-Net 모델
        self.backup_segmenter = None
        
        # 성능 통계
        self.performance_stats = {
            'total_processed': 0,
            'average_time': 0.0,
            'method_success_rate': {},
            'quality_distribution': []
        }
        
        # 스레드 풀
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
        self.is_initialized = False
        
        logger.info(f"🎯 통합 의류 세그멘테이션 시스템 초기화 - 디바이스: {self.device}")
    
    def _setup_optimal_device(self, preferred_device: str) -> str:
        """M3 Max 최적화된 디바이스 설정"""
        if preferred_device == 'mps' and torch.backends.mps.is_available():
            logger.info("✅ M3 Max MPS 백엔드 사용")
            return 'mps'
        elif preferred_device == 'cuda' and torch.cuda.is_available():
            logger.info("✅ CUDA 백엔드 사용") 
            return 'cuda'
        else:
            logger.info("⚠️ CPU 백엔드 사용")
            return 'cpu'
    
    async def initialize(self) -> bool:
        """세그멘테이션 시스템 초기화 (기존 코드 구조 유지)"""
        try:
            logger.info("🔄 통합 의류 세그멘테이션 시스템 초기화 중...")
            
            # 1. RemBG 모델들 초기화
            await self._initialize_rembg_models()
            
            # 2. U²-Net 모델 초기화 (기존 코드 방식)
            await self._initialize_segmentation_models()
            
            # 3. 백업 방법들 초기화
            self.backup_segmenter = BackupSegmentationMethods(self.device)
            
            # 4. 모델 워밍업
            await self._warmup_models()
            
            self.is_initialized = True
            logger.info("✅ 통합 의류 세그멘테이션 시스템 초기화 완료")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 의류 세그멘테이션 시스템 초기화 실패: {e}")
            self.is_initialized = False
            return False
    
    async def _initialize_rembg_models(self):
        """RemBG 모델들 초기화"""
        if not REMBG_AVAILABLE:
            logger.warning("RemBG 사용 불가 - 백업 방법 사용")
            return
        
        try:
            # 기본 모델 먼저 로드
            model_name = self.segmentation_config['model_name']
            logger.info(f"📦 RemBG 기본 모델 로딩: {model_name}")
            self.rembg_session = new_session(model_name)
            self.rembg_sessions[model_name] = self.rembg_session
            
            # 추가 모델들 (의류별 특화)
            additional_models = ['u2net_human_seg', 'silueta']
            for model in additional_models:
                if model != model_name:
                    try:
                        session = new_session(model)
                        self.rembg_sessions[model] = session
                        logger.info(f"✅ 추가 모델 로드: {model}")
                    except Exception as e:
                        logger.warning(f"⚠️ 추가 모델 {model} 로드 실패: {e}")
                        
            logger.info("✅ RemBG 모델들 로드 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ RemBG 초기화 실패: {e}")
            self.rembg_session = None
    
    async def _initialize_segmentation_models(self):
        """세그멘테이션 모델들 초기화 (기존 코드 방식 유지)"""
        try:
            model_type = self.config.get('model_type', 'rembg')
            
            if model_type == 'u2net' or not REMBG_AVAILABLE:
                # U²-Net 모델 초기화
                self.segmentation_model = await self._initialize_u2net()
            elif model_type == 'custom':
                # 커스텀 모델 초기화
                self.segmentation_model = await self._initialize_custom_model()
            
            logger.info("✅ 세그멘테이션 모델 초기화 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ 세그멘테이션 모델 초기화 실패: {e}")
            self.segmentation_model = None
    
    async def _initialize_u2net(self):
        """U²-Net 모델 초기화 (기존 코드 개선)"""
        try:
            # U²-Net 아키텍처 생성
            model = self._create_u2net_model()
            
            # 모델 가중치 로드 시도
            model_path = self._get_u2net_model_path()
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(f"✅ U²-Net 가중치 로드: {model_path}")
            else:
                logger.warning(f"⚠️ U²-Net 가중치 파일 없음: {model_path}")
            
            # 모델 최적화 (model_loader 사용)
            if self.model_loader:
                model = self.model_loader.optimize_model(model, 'cloth_segmentation')
            
            model = model.to(self.device)
            model.eval()
            
            # M3 Max 최적화
            if self.device == 'mps':
                model = torch.jit.optimize_for_inference(model)
            
            return model
            
        except Exception as e:
            logger.warning(f"U²-Net 초기화 실패: {e}")
            return self._create_demo_segmentation_model()
    
    async def _initialize_custom_model(self):
        """커스텀 세그멘테이션 모델 초기화 (기존 코드)"""
        return self._create_demo_segmentation_model()
    
    def _create_u2net_model(self):
        """U²-Net 모델 아키텍처 생성 (기존 코드 개선)"""
        class U2NetSegmentation(torch.nn.Module):
            def __init__(self, in_ch=3, out_ch=1):
                super(U2NetSegmentation, self).__init__()
                
                # 인코더 (더 효율적인 구조)
                self.encoder1 = self._conv_block(in_ch, 64)
                self.encoder2 = self._conv_block(64, 128)
                self.encoder3 = self._conv_block(128, 256)
                self.encoder4 = self._conv_block(256, 512)
                
                # 중간 레이어
                self.middle = self._conv_block(512, 1024)
                
                # 디코더
                self.decoder4 = self._conv_block(1024 + 512, 512)
                self.decoder3 = self._conv_block(512 + 256, 256)
                self.decoder2 = self._conv_block(256 + 128, 128)
                self.decoder1 = self._conv_block(128 + 64, 64)
                
                # 출력 레이어
                self.final = torch.nn.Conv2d(64, out_ch, 1)
                
                # 풀링 및 업샘플링
                self.pool = torch.nn.MaxPool2d(2)
                self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                
                # 드롭아웃 (정규화)
                self.dropout = torch.nn.Dropout2d(0.2)
            
            def _conv_block(self, in_ch, out_ch):
                return torch.nn.Sequential(
                    torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    torch.nn.BatchNorm2d(out_ch),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    torch.nn.BatchNorm2d(out_ch),
                    torch.nn.ReLU(inplace=True)
                )
            
            def forward(self, x):
                # 인코더
                e1 = self.encoder1(x)
                e2 = self.encoder2(self.pool(e1))
                e3 = self.encoder3(self.pool(e2))
                e4 = self.encoder4(self.pool(e3))
                
                # 중간
                m = self.middle(self.pool(e4))
                m = self.dropout(m)
                
                # 디코더
                d4 = self.decoder4(torch.cat([self.upsample(m), e4], dim=1))
                d3 = self.decoder3(torch.cat([self.upsample(d4), e3], dim=1))
                d2 = self.decoder2(torch.cat([self.upsample(d3), e2], dim=1))
                d1 = self.decoder1(torch.cat([self.upsample(d2), e1], dim=1))
                
                # 출력
                output = torch.sigmoid(self.final(d1))
                
                return output
        
        return U2NetSegmentation().to(self.device)
    
    def _create_demo_segmentation_model(self):
        """데모용 세그멘테이션 모델 (기존 코드)"""
        class DemoSegmentationModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = torch.nn.Conv2d(64, 32, 3, padding=1)
                self.conv3 = torch.nn.Conv2d(32, 1, 1)
                
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = torch.sigmoid(self.conv3(x))
                return x
        
        return DemoSegmentationModel().to(self.device)
    
    def _get_u2net_model_path(self) -> str:
        """U²-Net 모델 파일 경로 (기존 코드)"""
        model_dir = self.config.get('model_dir', 'app/ai_pipeline/models')
        model_file = self.config.get('model_file', 'u2net_cloth.pth')
        return os.path.join(model_dir, model_file)
    
    async def _warmup_models(self):
        """모델 워밍업"""
        logger.info("🔥 모델 워밍업 중...")
        
        try:
            # 더미 입력으로 워밍업
            dummy_input = torch.randn(1, 3, 320, 320).to(self.device)
            
            if self.segmentation_model:
                with torch.no_grad():
                    _ = self.segmentation_model(dummy_input)
            
            logger.info("✅ 모델 워밍업 완료")
        except Exception as e:
            logger.warning(f"⚠️ 모델 워밍업 실패: {e}")
    
    def process(
        self, 
        clothing_image_tensor: torch.Tensor,
        clothing_type: str = "shirt",
        quality_level: str = "high"
    ) -> Dict[str, Any]:
        """
        의류 세그멘테이션 처리 (기존 인터페이스 유지)
        
        Args:
            clothing_image_tensor: 의류 이미지 텐서 [1, 3, H, W]
            clothing_type: 의류 타입 (shirt, pants, dress, etc.)
            quality_level: 품질 레벨 (low, medium, high)
            
        Returns:
            처리 결과 딕셔너리
        """
        if not self.is_initialized:
            raise RuntimeError("의류 세그멘테이션 시스템이 초기화되지 않았습니다.")
        
        start_time = time.time()
        
        try:
            # 1. 텐서를 PIL 이미지로 변환
            clothing_pil = self._tensor_to_pil(clothing_image_tensor)
            
            # 2. 최적 세그멘테이션 방법 선택
            method = self._select_best_method(clothing_pil, clothing_type, quality_level)
            
            # 3. 메인 세그멘테이션 수행
            logger.info(f"🔍 세그멘테이션 수행: {method}")
            segmentation_result = self._perform_main_segmentation(clothing_pil, method)
            
            # 4. 세그멘테이션 품질 평가
            logger.info("📊 품질 평가 중...")
            quality_score = self._evaluate_segmentation_quality(
                clothing_pil, segmentation_result['mask']
            )
            
            # 5. 품질이 낮으면 폴백 방법 시도
            if quality_score < self.segmentation_config['quality_threshold']:
                logger.info(f"🔄 품질 개선 시도 (현재: {quality_score:.3f})")
                improved_result = self._try_fallback_methods(clothing_pil, clothing_type)
                if improved_result and improved_result.get('quality', 0) > quality_score:
                    segmentation_result = improved_result
                    quality_score = improved_result['quality']
            
            # 6. 후처리 적용
            logger.info("✨ 후처리 적용 중...")
            processed_result = self._apply_post_processing(
                segmentation_result, quality_level
            )
            
            # 7. 최종 결과 구성
            processing_time = time.time() - start_time
            
            result = self._build_result(
                processed_result, quality_score, processing_time, 
                method, clothing_type
            )
            
            # 8. 통계 업데이트
            self._update_stats(method, processing_time, quality_score)
            
            logger.info(f"✅ 의류 세그멘테이션 완료 - {processing_time:.3f}초, 품질: {quality_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 의류 세그멘테이션 처리 실패: {e}")
            raise
    
    def _select_best_method(self, image: Image.Image, clothing_type: str, quality_level: str) -> str:
        """최적 세그멘테이션 방법 선택"""
        method = self.segmentation_config['method']
        
        if method == 'auto':
            # 이미지 복잡도 분석
            complexity = self._analyze_image_complexity(image)
            
            # 품질 레벨과 복잡도에 따른 방법 선택
            if quality_level == 'high' and complexity > 0.7:
                return 'ensemble' if self.segmentation_config['use_ensemble'] else 'rembg'
            elif REMBG_AVAILABLE and complexity < 0.5:
                return 'rembg'
            elif self.segmentation_model and complexity > 0.3:
                return 'u2net'
            else:
                return 'grabcut'
        
        return method
    
    def _analyze_image_complexity(self, image: Image.Image) -> float:
        """이미지 복잡도 분석"""
        try:
            # 그레이스케일 변환
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # 엣지 밀도
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # 텍스처 복잡도 (LBP 간소화 버전)
            texture_var = np.var(gray)
            normalized_texture = min(texture_var / 10000, 1.0)
            
            # 복잡도 점수
            complexity = edge_density * 0.6 + normalized_texture * 0.4
            
            return min(complexity, 1.0)
            
        except Exception as e:
            logger.warning(f"복잡도 분석 실패: {e}")
            return 0.5  # 기본값
    
    def _perform_main_segmentation(self, image: Image.Image, method: str) -> Dict[str, Any]:
        """메인 세그멘테이션 수행"""
        
        if method == 'ensemble':
            return self._ensemble_segmentation(image)
        elif method == 'rembg' and self.rembg_session:
            return self._segment_with_rembg(image)
        elif method == 'u2net' and self.segmentation_model:
            return self._segment_with_u2net_model(image)
        elif method == 'grabcut':
            return self._segment_with_grabcut(image)
        elif method == 'kmeans':
            return self._segment_with_kmeans(image)
        else:
            return self._segment_with_threshold(image)
    
    def _ensemble_segmentation(self, image: Image.Image) -> Dict[str, Any]:
        """앙상블 세그멘테이션 (여러 방법 조합)"""
        results = []
        methods = ['rembg', 'u2net', 'grabcut']
        
        for method in methods:
            try:
                if method == 'rembg' and self.rembg_session:
                    result = self._segment_with_rembg(image)
                elif method == 'u2net' and self.segmentation_model:
                    result = self._segment_with_u2net_model(image)
                else:
                    result = self.backup_segmenter.grabcut_segmentation(image)
                
                if result:
                    results.append(result)
            except Exception as e:
                logger.warning(f"앙상블 방법 {method} 실패: {e}")
        
        if not results:
            return self._segment_with_threshold(image)
        
        # 최고 품질 결과 선택
        best_result = max(results, key=lambda x: x.get('confidence', 0))
        best_result['method'] = 'ensemble'
        
        return best_result
    
    def _segment_with_rembg(self, image: Image.Image) -> Dict[str, Any]:
        """RemBG를 사용한 세그멘테이션 (기존 코드 개선)"""
        try:
            # 의류별 모델 선택
            model_selection = {
                'dress': 'u2net_human_seg',
                'upper': 'u2net',
                'lower': 'silueta'
            }
            
            # 적절한 모델 선택
            preferred_model = model_selection.get('upper', 'u2net')  # 기본값
            session = self.rembg_sessions.get(preferred_model, self.rembg_session)
            
            # RemBG로 배경 제거
            result_image = remove(image, session=session)
            
            # 알파 채널에서 마스크 추출
            if result_image.mode == 'RGBA':
                mask = np.array(result_image)[:, :, 3]
                segmented_rgb = result_image.convert('RGB')
            else:
                # 알파 채널이 없으면 간단한 임계값 사용
                gray = np.array(result_image.convert('L'))
                mask = (gray > 10).astype(np.uint8) * 255
                segmented_rgb = result_image.convert('RGB')
            
            return {
                'segmented_image': segmented_rgb,
                'mask': mask,
                'method': f'rembg_{preferred_model}',
                'confidence': 0.9
            }
            
        except Exception as e:
            logger.warning(f"RemBG 세그멘테이션 실패: {e}")
            return self._segment_with_threshold(image)
    
    def _segment_with_u2net_model(self, image: Image.Image) -> Dict[str, Any]:
        """U²-Net 모델을 사용한 세그멘테이션 (기존 코드 개선)"""
        try:
            # 입력 전처리
            input_tensor = self._preprocess_for_model(image)
            
            # 모델 추론
            with torch.no_grad():
                if self.device == 'mps':
                    # M3 Max 최적화
                    mask_pred = self.segmentation_model(input_tensor)
                else:
                    mask_pred = self.segmentation_model(input_tensor)
                
                # 마스크 후처리
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
                'method': 'u2net',
                'confidence': 0.8
            }
            
        except Exception as e:
            logger.warning(f"U²-Net 세그멘테이션 실패: {e}")
            return self._segment_with_threshold(image)
    
    def _segment_with_grabcut(self, image: Image.Image) -> Dict[str, Any]:
        """GrabCut을 사용한 세그멘테이션 (기존 코드)"""
        return self.backup_segmenter.grabcut_segmentation(image)
    
    def _segment_with_kmeans(self, image: Image.Image) -> Dict[str, Any]:
        """K-means 클러스터링을 사용한 세그멘테이션 (기존 코드)"""
        return self.backup_segmenter.kmeans_segmentation(image)
    
    def _segment_with_threshold(self, image: Image.Image) -> Dict[str, Any]:
        """간단한 임계값을 사용한 세그멘테이션 (기존 코드)"""
        return self.backup_segmenter.threshold_segmentation(image)
    
    def _try_fallback_methods(self, image: Image.Image, clothing_type: str) -> Optional[Dict[str, Any]]:
        """폴백 세그멘테이션 방법들 시도 (기존 코드 개선)"""
        fallback_methods = self.segmentation_config['fallback_methods']
        best_result = None
        best_quality = 0.0
        
        for method in fallback_methods:
            try:
                if method == 'rembg' and self.rembg_session:
                    result = self._segment_with_rembg(image)
                elif method == 'u2net' and self.segmentation_model:
                    result = self._segment_with_u2net_model(image)
                elif method == 'grabcut':
                    result = self.backup_segmenter.grabcut_segmentation(image)
                elif method == 'kmeans':
                    result = self.backup_segmenter.kmeans_segmentation(image)
                else:
                    result = self.backup_segmenter.threshold_segmentation(image)
                
                if result:
                    quality = self._evaluate_segmentation_quality(image, result['mask'])
                    result['quality'] = quality
                    
                    if quality > best_quality:
                        best_quality = quality
                        best_result = result
                        
            except Exception as e:
                logger.warning(f"폴백 방법 {method} 실패: {e}")
                continue
        
        return best_result
    
    def _apply_post_processing(
        self, 
        segmentation_result: Dict[str, Any], 
        quality_level: str
    ) -> Dict[str, Any]:
        """후처리 적용 (기존 코드 확장)"""
        
        mask = segmentation_result['mask']
        
        # 품질 레벨에 따른 처리 강도
        intensity_map = {'low': 1, 'medium': 2, 'high': 3}
        intensity = intensity_map.get(quality_level, 2)
        
        processed_mask = mask.copy()
        
        # 1. 형태학적 연산
        if self.post_process_config['morphology_enabled']:
            kernel_size = 3 + intensity
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel)
            processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)
        
        # 2. 가우시안 블러
        if self.post_process_config['gaussian_blur']:
            blur_kernel = 3 + intensity * 2
            if blur_kernel % 2 == 0:
                blur_kernel += 1
            processed_mask = cv2.GaussianBlur(processed_mask, (blur_kernel, blur_kernel), 0)
            processed_mask = (processed_mask > 127).astype(np.uint8) * 255
        
        # 3. 엣지 스무딩
        if self.post_process_config['edge_smoothing']:
            processed_mask = self._smooth_edges(processed_mask, intensity)
        
        # 4. 노이즈 제거
        if self.post_process_config['noise_removal']:
            processed_mask = self._remove_noise(processed_mask, intensity)
        
        # 마스크를 텐서로 변환
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
    
    def _smooth_edges(self, mask: np.ndarray, intensity: int) -> np.ndarray:
        """엣지 스무딩"""
        # 거리 변환
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # 임계값으로 부드러운 경계 생성
        threshold = intensity * 2
        smooth_mask = (dist_transform > threshold).astype(np.uint8) * 255
        
        return smooth_mask
    
    def _remove_noise(self, mask: np.ndarray, intensity: int) -> np.ndarray:
        """노이즈 제거"""
        # 연결 성분 분석
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_labels <= 1:
            return mask
        
        # 가장 큰 연결 성분 찾기
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        # 작은 성분들 제거
        min_area = stats[largest_component, cv2.CC_STAT_AREA] * 0.1 / intensity
        
        cleaned_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned_mask[labels == i] = 255
        
        return cleaned_mask
    
    def _generate_confidence_map(self, mask: np.ndarray) -> np.ndarray:
        """신뢰도 맵 생성"""
        # 거리 변환으로 중심부일수록 높은 신뢰도
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # 정규화
        if dist_transform.max() > 0:
            confidence_map = dist_transform / dist_transform.max()
        else:
            confidence_map = np.zeros_like(mask, dtype=np.float32)
        
        return confidence_map
    
    def _build_result(
        self,
        processed_result: Dict[str, Any],
        quality_score: float,
        processing_time: float,
        method: str,
        clothing_type: str
    ) -> Dict[str, Any]:
        """최종 결과 구성 (기존 인터페이스 유지)"""
        
        mask = processed_result['binary_mask']
        segmented_image = processed_result['segmented_image']
        
        return {
            "success": True,
            "segmented_image": segmented_image,
            "clothing_mask": processed_result['mask_tensor'],
            "binary_mask": mask,
            "confidence_map": processed_result.get('confidence_map', None),
            "segmentation_quality": quality_score,
            "clothing_analysis": {
                "dominant_colors": self._extract_dominant_colors(segmented_image),
                "clothing_area": self._calculate_clothing_area(mask),
                "edge_complexity": self._calculate_edge_complexity(mask),
                "background_removed": True
            },
            "processing_info": {
                "method_used": method,
                "processing_time": processing_time,
                "post_processing_applied": True,
                "device": self.device
            }
        }
    
    def _extract_dominant_colors(self, image: Image.Image) -> List[List[int]]:
        """주요 색상 추출"""
        if not SKLEARN_AVAILABLE:
            return [[128, 128, 128]]  # 기본 회색
        
        img_array = np.array(image)
        pixels = img_array.reshape(-1, 3)
        
        # 배경색 제거 (흰색 근처)
        non_white_pixels = pixels[np.sum(pixels, axis=1) < 700]
        
        if len(non_white_pixels) < 10:
            return [[128, 128, 128]]
        
        # K-means로 주요 색상 추출
        n_colors = min(5, len(non_white_pixels) // 100)
        if n_colors < 1:
            n_colors = 1
            
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(non_white_pixels)
        
        dominant_colors = kmeans.cluster_centers_.astype(int).tolist()
        
        return dominant_colors
    
    def _calculate_clothing_area(self, mask: np.ndarray) -> Dict[str, float]:
        """의류 영역 계산"""
        total_pixels = mask.size
        clothing_pixels = np.sum(mask > 0)
        
        return {
            'total_pixels': float(total_pixels),
            'clothing_pixels': float(clothing_pixels),
            'coverage_ratio': float(clothing_pixels / total_pixels),
            'area_score': min(1.0, clothing_pixels / 50000)  # 정규화
        }
    
    def _calculate_edge_complexity(self, mask: np.ndarray) -> Dict[str, float]:
        """엣지 복잡도 계산"""
        # 엣지 검출
        edges = cv2.Canny(mask, 50, 150)
        edge_pixels = np.sum(edges > 0)
        
        # 윤곽선 분석
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(main_contour, True)
            area = cv2.contourArea(main_contour)
            
            # 복잡도 계산 (둘레²/면적 비율)
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
    
    def _evaluate_segmentation_quality(self, original: Image.Image, mask: np.ndarray) -> float:
        """세그멘테이션 품질 평가 (기존 코드)"""
        # 마스크 연결성 평가
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # 가장 큰 연결 요소
        main_contour = max(contours, key=cv2.contourArea)
        main_area = cv2.contourArea(main_contour)
        total_mask_area = np.sum(mask > 0)
        
        # 연결성 점수 (주요 영역이 전체에서 차지하는 비율)
        connectivity_score = main_area / (total_mask_area + 1e-6)
        
        # 엣지 품질 평가
        edge_quality = self._evaluate_edge_quality(mask)
        
        # 크기 적절성 (전체 이미지에서 의류가 차지하는 비율)
        image_area = original.width * original.height
        size_ratio = total_mask_area / image_area
        size_score = 1.0 if 0.1 <= size_ratio <= 0.8 else max(0.0, 1.0 - abs(size_ratio - 0.4) * 2)
        
        # 종합 품질 점수
        quality = (connectivity_score * 0.4 + edge_quality * 0.4 + size_score * 0.2)
        
        return quality
    
    def _evaluate_edge_quality(self, mask: np.ndarray) -> float:
        """엣지 품질 평가"""
        # 엣지 검출
        edges = cv2.Canny(mask, 50, 150)
        
        # 엣지의 부드러움 측정
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.5
        
        main_contour = max(contours, key=cv2.contourArea)
        
        # 윤곽선 근사화
        epsilon = 0.02 * cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        
        # 부드러움 점수 (근사화 후 점의 개수가 적을수록 부드러움)
        if len(main_contour) > 0:
            smoothness = 1.0 - (len(approx) / len(main_contour))
        else:
            smoothness = 0.5
        
        return max(0.0, min(1.0, smoothness))
    
    def _update_stats(self, method: str, processing_time: float, quality_score: float):
        """성능 통계 업데이트"""
        self.performance_stats['total_processed'] += 1
        
        # 평균 처리 시간 업데이트
        total = self.performance_stats['total_processed']
        current_avg = self.performance_stats['average_time']
        self.performance_stats['average_time'] = (current_avg * (total - 1) + processing_time) / total
        
        # 방법별 성공률
        if method not in self.performance_stats['method_success_rate']:
            self.performance_stats['method_success_rate'][method] = {'success': 0, 'total': 0}
        
        self.performance_stats['method_success_rate'][method]['total'] += 1
        if quality_score > 0.5:  # 성공 기준
            self.performance_stats['method_success_rate'][method]['success'] += 1
        
        # 품질 분포
        self.performance_stats['quality_distribution'].append(quality_score)
        if len(self.performance_stats['quality_distribution']) > 100:
            self.performance_stats['quality_distribution'] = self.performance_stats['quality_distribution'][-100:]
    
    # 유틸리티 메서드들 (기존 코드)
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """PyTorch 텐서를 PIL 이미지로 변환"""
        # [1, 3, H, W] -> [3, H, W]
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # [3, H, W] -> [H, W, 3]
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        
        # 정규화 해제
        if tensor.max() <= 1.0:
            tensor = tensor * 255
        
        # NumPy로 변환
        array = tensor.cpu().numpy().astype(np.uint8)
        
        # PIL 이미지로 변환
        return Image.fromarray(array)
    
    def _preprocess_for_model(self, image: Image.Image) -> torch.Tensor:
        """모델 입력을 위한 전처리"""
        # 크기 조정
        input_size = (320, 320)
        resized = image.resize(input_size, Image.Resampling.LANCZOS)
        
        # 텐서로 변환
        tensor = torch.from_numpy(np.array(resized)).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        
        # 정규화 (ImageNet)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        
        return tensor.to(self.device)
    
    def _apply_mask_to_image(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """마스크를 이미지에 적용하여 배경 제거"""
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
    
    async def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환 (기존 코드)"""
        return {
            "step_name": "ClothingSegmentation",
            "version": "2.0",
            "device": self.device,
            "use_mps": self.device == 'mps',
            "initialized": self.is_initialized,
            "segmentation_config": self.segmentation_config,
            "post_process_config": self.post_process_config,
            "available_methods": ["rembg", "u2net", "grabcut", "kmeans", "threshold"],
            "rembg_available": REMBG_AVAILABLE,
            "sklearn_available": SKLEARN_AVAILABLE,
            "supported_models": list(self.rembg_sessions.keys()) if REMBG_AVAILABLE else [],
            "performance_stats": self.performance_stats
        }
    
    async def cleanup(self):
        """리소스 정리 (기존 코드)"""
        # 모델들 정리
        if self.segmentation_model:
            del self.segmentation_model
            self.segmentation_model = None
        
        # RemBG 세션들 정리
        for session in self.rembg_sessions.values():
            del session
        self.rembg_sessions.clear()
        self.rembg_session = None
        
        # 백업 세그멘터 정리
        if self.backup_segmenter:
            del self.backup_segmenter
            self.backup_segmenter = None
        
        # 스레드 풀 종료
        self.thread_pool.shutdown(wait=True)
        
        # MPS 캐시 정리 (M3 Max)
        if self.device == 'mps':
            torch.mps.empty_cache()
        elif self.device == 'cuda':
            torch.cuda.empty_cache()
        
        self.is_initialized = False
        logger.info("🧹 의류 세그멘테이션 스텝 리소스 정리 완료")


class BackupSegmentationMethods:
    """백업 세그멘테이션 방법들"""
    
    def __init__(self, device: str):
        self.device = device
    
    def grabcut_segmentation(self, image: Image.Image) -> Dict[str, Any]:
        """GrabCut 세그멘테이션"""
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]
        
        # 초기 사각형 (중앙 80% 영역)
        margin_h, margin_w = int(h * 0.1), int(w * 0.1)
        rect = (margin_w, margin_h, w - 2*margin_w, h - 2*margin_h)
        
        # GrabCut 초기화
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
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
    
    def kmeans_segmentation(self, image: Image.Image) -> Dict[str, Any]:
        """K-means 클러스터링 세그멘테이션"""
        if not SKLEARN_AVAILABLE:
            return self.threshold_segmentation(image)
        
        img_array = np.array(image)
        h, w, c = img_array.shape
        
        # 픽셀 데이터 준비
        pixel_data = img_array.reshape((-1, 3))
        
        try:
            # K-means (3개 클러스터)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixel_data)
            
            # 가장 큰 클러스터를 배경으로 간주
            unique_labels, counts = np.unique(labels, return_counts=True)
            background_label = unique_labels[np.argmax(counts)]
            
            # 마스크 생성
            mask = (labels != background_label).reshape((h, w)).astype(np.uint8) * 255
            
            # 결과 이미지
            result = img_array.copy()
            result[mask == 0] = [255, 255, 255]
            result_pil = Image.fromarray(result)
            
            return {
                'segmented_image': result_pil,
                'mask': mask,
                'method': 'kmeans',
                'confidence': 0.6
            }
        except Exception as e:
            logger.warning(f"K-means 실패: {e}")
            return self.threshold_segmentation(image)
    
    def threshold_segmentation(self, image: Image.Image) -> Dict[str, Any]:
        """임계값 기반 세그멘테이션"""
        try:
            # 그레이스케일 변환
            gray = np.array(image.convert('L'))
            
            # Otsu 임계값
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 배경이 밝은 경우 반전
            if np.mean(binary) > 127:
                binary = 255 - binary
            
            # 형태학적 연산
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
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
            # 최후의 수단: 전체 이미지를 전경으로
            h, w = image.size
            mask = np.ones((w, h), dtype=np.uint8) * 255
            return {
                'segmented_image': image,
                'mask': mask,
                'method': 'fallback',
                'confidence': 0.3
            }