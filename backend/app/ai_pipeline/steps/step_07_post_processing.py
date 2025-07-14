# app/ai_pipeline/steps/step_07_post_processing.py
"""
7단계: 후처리 (Post Processing) - 품질 향상
MyCloset AI 가상 피팅 파이프라인의 최종 단계 - model_loader 수정 버전

🎯 주요 기능:
- Real-ESRGAN: Super Resolution (2x, 4x 해상도 향상)
- GFPGAN: 얼굴 품질 향상 및 복원
- CodeFormer: 전체적인 이미지 복원  
- 색상 보정, 노이즈 제거, 엣지 향상
- M3 Max Metal Performance Shaders 최적화
- 실시간 품질 평가 및 피드백

🚀 M3 Max 최적화:
- 128GB RAM 대용량 배치 처리
- 14코어 CPU 병렬 처리
- Metal GPU 가속
- 메모리 효율적 타일 처리
"""

import os
import logging
import time
import asyncio
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image, ImageFilter, ImageEnhance
from concurrent.futures import ThreadPoolExecutor
import json

logger = logging.getLogger(__name__)

class PostProcessingStep:
    """
    Step 7: 후처리 품질 향상 
    기존 ai_pipeline 구조에 맞춘 통합 후처리 시스템
    """
    
    def __init__(self, device: str = "mps", config: Dict[str, Any] = None):
        """
        Args:
            device: 사용할 디바이스 (mps, cuda, cpu)
            config: 설정 딕셔너리
        """
        self.config = config or {}
        
        # 디바이스 설정 (M3 Max 최적화)
        self.device = self._get_optimal_device(device)
        
        # model_loader는 내부에서 생성하거나 전역에서 가져옴
        try:
            from app.ai_pipeline.utils.model_loader import get_global_model_loader
            self.model_loader = get_global_model_loader()
        except ImportError:
            logger.warning("ModelLoader 초기화 실패 - 기본 처리로 진행")
            self.model_loader = None
        
        # 메모리 관리 초기화  
        try:
            from app.ai_pipeline.utils.memory_manager import MemoryManager
            self.memory_manager = MemoryManager()
        except ImportError:
            logger.warning("MemoryManager 초기화 실패")
            self.memory_manager = None
        
        # 후처리 설정
        self.enhancement_config = self.config.get('post_processing', {
            'super_resolution': True,    # Real-ESRGAN
            'face_enhancement': True,    # GFPGAN
            'image_restoration': True,   # CodeFormer
            'color_correction': True,    # 색상 보정
            'noise_reduction': True,     # 노이즈 제거
            'edge_enhancement': True,    # 엣지 향상
            'quality_level': 'high'      # 품질 수준
        })
        
        # M3 Max 최적화 설정
        self.use_mps = self.device == 'mps' and torch.backends.mps.is_available()
        self.batch_size = self.config.get('batch_size', 1)
        self.tile_size = self.config.get('tile_size', 512)
        
        # 모델 인스턴스들
        self.real_esrgan = None
        self.gfpgan = None
        self.codeformer = None
        
        # 전통적 처리 도구들
        self.color_enhancer = None
        self.noise_reducer = None
        self.edge_enhancer = None
        
        # 초기화 상태
        self.is_initialized = False
        
        # 성능 통계
        self.processing_stats = {
            'total_images': 0,
            'average_time': 0.0,
            'enhancement_success_rate': 0.0
        }
        
        logger.info(f"🎨 Step 7 후처리 초기화 - 디바이스: {self.device}")
    
    def _get_optimal_device(self, preferred_device: str) -> str:
        """최적 디바이스 선택"""
        if preferred_device == 'auto':
            if torch.backends.mps.is_available():
                return 'mps'  # M3 Max Metal Performance Shaders
            elif torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return preferred_device
    
    async def initialize(self) -> bool:
        """후처리 모델들 초기화"""
        try:
            logger.info("🔄 Step 7 후처리 모델 로드 중...")
            
            initialization_tasks = []
            
            # AI 모델 비동기 초기화
            if self.enhancement_config.get('super_resolution', True):
                initialization_tasks.append(self._init_real_esrgan())
            
            if self.enhancement_config.get('face_enhancement', True):
                initialization_tasks.append(self._init_gfpgan())
            
            if self.enhancement_config.get('image_restoration', True):
                initialization_tasks.append(self._init_codeformer())
            
            # 전통적 도구 초기화
            initialization_tasks.extend([
                self._init_color_enhancer(),
                self._init_noise_reducer(),
                self._init_edge_enhancer()
            ])
            
            # 병렬 초기화 실행
            results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
            
            # 초기화 결과 확인
            success_count = sum(1 for result in results if result is True)
            total_count = len(results)
            
            if success_count >= total_count // 2:  # 절반 이상 성공시 OK
                self.is_initialized = True
                logger.info(f"✅ Step 7 후처리 모델 로드 완료 ({success_count}/{total_count})")
                return True
            else:
                logger.error(f"❌ Step 7 후처리 모델 로드 실패 ({success_count}/{total_count})")
                return False
                
        except Exception as e:
            logger.error(f"❌ Step 7 후처리 초기화 실패: {e}")
            self.is_initialized = False
            return False
    
    async def _init_real_esrgan(self) -> bool:
        """Real-ESRGAN Super Resolution 모델 초기화"""
        try:
            model_path = self._get_model_path('real_esrgan', 'RealESRGAN_x4plus.pth')
            
            if os.path.exists(model_path) and self.model_loader:
                self.real_esrgan = await self.model_loader.load_model(
                    'real_esrgan', 
                    model_path, 
                    device=self.device
                )
                logger.info("✅ Real-ESRGAN 모델 로드 완료")
                return True
            else:
                # 폴백: 기본 업스케일링
                self.real_esrgan = BasicUpscaler()
                logger.info("📄 Real-ESRGAN 폴백 모드 사용")
                return True
                
        except Exception as e:
            logger.warning(f"Real-ESRGAN 초기화 실패: {e}")
            self.real_esrgan = BasicUpscaler()
            return True
    
    async def _init_gfpgan(self) -> bool:
        """GFPGAN 얼굴 향상 모델 초기화"""
        try:
            model_path = self._get_model_path('gfpgan', 'GFPGANv1.4.pth')
            
            if os.path.exists(model_path) and self.model_loader:
                self.gfpgan = await self.model_loader.load_model(
                    'gfpgan', 
                    model_path, 
                    device=self.device
                )
                logger.info("✅ GFPGAN 모델 로드 완료")
                return True
            else:
                # 폴백: 기본 얼굴 향상
                self.gfpgan = BasicFaceEnhancer()
                logger.info("📄 GFPGAN 폴백 모드 사용")
                return True
                
        except Exception as e:
            logger.warning(f"GFPGAN 초기화 실패: {e}")
            self.gfpgan = BasicFaceEnhancer()
            return True
    
    async def _init_codeformer(self) -> bool:
        """CodeFormer 이미지 복원 모델 초기화"""
        try:
            model_path = self._get_model_path('codeformer', 'codeformer.pth')
            
            if os.path.exists(model_path) and self.model_loader:
                self.codeformer = await self.model_loader.load_model(
                    'codeformer', 
                    model_path, 
                    device=self.device
                )
                logger.info("✅ CodeFormer 모델 로드 완료")
                return True
            else:
                # 폴백: 기본 이미지 복원
                self.codeformer = BasicImageRestorer()
                logger.info("📄 CodeFormer 폴백 모드 사용")
                return True
                
        except Exception as e:
            logger.warning(f"CodeFormer 초기화 실패: {e}")
            self.codeformer = BasicImageRestorer()
            return True
    
    async def _init_color_enhancer(self) -> bool:
        """색상 향상기 초기화"""
        try:
            self.color_enhancer = ColorEnhancer()
            return True
        except Exception as e:
            logger.warning(f"색상 향상기 초기화 실패: {e}")
            return False
    
    async def _init_noise_reducer(self) -> bool:
        """노이즈 제거기 초기화"""
        try:
            self.noise_reducer = NoiseReducer()
            return True
        except Exception as e:
            logger.warning(f"노이즈 제거기 초기화 실패: {e}")
            return False
    
    async def _init_edge_enhancer(self) -> bool:
        """엣지 향상기 초기화"""
        try:
            self.edge_enhancer = EdgeEnhancer()
            return True
        except Exception as e:
            logger.warning(f"엣지 향상기 초기화 실패: {e}")
            return False
    
    def _get_model_path(self, model_type: str, filename: str) -> str:
        """모델 파일 경로 반환"""
        base_path = self.config.get('models_base_path', 'app/ai_pipeline/models/ai_models')
        return os.path.join(base_path, model_type, filename)
    
    async def process(
        self, 
        input_image: Union[np.ndarray, torch.Tensor, str],
        enhancement_options: Optional[Dict[str, Any]] = None,
        quality_target: float = 0.8
    ) -> Dict[str, Any]:
        """
        후처리 메인 처리 함수
        
        Args:
            input_image: 입력 이미지
            enhancement_options: 향상 옵션
            quality_target: 목표 품질 (0.0-1.0)
            
        Returns:
            후처리 결과
        """
        if not self.is_initialized:
            raise RuntimeError("후처리 시스템이 초기화되지 않았습니다.")
        
        start_time = time.time()
        
        try:
            # 1. 입력 전처리
            image_tensor = await self._preprocess_input(input_image)
            original_shape = image_tensor.shape
            
            logger.info(f"🎨 후처리 시작 - 크기: {original_shape}")
            
            # 2. 향상 옵션 설정
            options = {**self.enhancement_config, **(enhancement_options or {})}
            
            # 3. 순차적 향상 처리
            enhanced_image = image_tensor.clone()
            processing_log = []
            
            # Super Resolution (해상도 향상)
            if options.get('super_resolution', True) and self.real_esrgan:
                logger.info("🔍 Super Resolution 적용 중...")
                enhanced_image, sr_metrics = await self._apply_super_resolution(enhanced_image)
                processing_log.append({'step': 'super_resolution', 'metrics': sr_metrics})
            
            # Face Enhancement (얼굴 향상)
            if options.get('face_enhancement', True) and self.gfpgan:
                logger.info("👤 얼굴 향상 적용 중...")
                enhanced_image, face_metrics = await self._apply_face_enhancement(enhanced_image)
                processing_log.append({'step': 'face_enhancement', 'metrics': face_metrics})
            
            # Image Restoration (전체 복원)
            if options.get('image_restoration', True) and self.codeformer:
                logger.info("🔧 이미지 복원 적용 중...")
                enhanced_image, restoration_metrics = await self._apply_image_restoration(enhanced_image)
                processing_log.append({'step': 'image_restoration', 'metrics': restoration_metrics})
            
            # Color Correction (색상 보정)
            if options.get('color_correction', True) and self.color_enhancer:
                logger.info("🌈 색상 보정 적용 중...")
                enhanced_image, color_metrics = await self._apply_color_correction(enhanced_image)
                processing_log.append({'step': 'color_correction', 'metrics': color_metrics})
            
            # Noise Reduction (노이즈 제거)
            if options.get('noise_reduction', True) and self.noise_reducer:
                logger.info("🔇 노이즈 제거 적용 중...")
                enhanced_image, noise_metrics = await self._apply_noise_reduction(enhanced_image)
                processing_log.append({'step': 'noise_reduction', 'metrics': noise_metrics})
            
            # Edge Enhancement (엣지 향상)
            if options.get('edge_enhancement', True) and self.edge_enhancer:
                logger.info("📐 엣지 향상 적용 중...")
                enhanced_image, edge_metrics = await self._apply_edge_enhancement(enhanced_image)
                processing_log.append({'step': 'edge_enhancement', 'metrics': edge_metrics})
            
            # 4. 후처리 및 품질 평가
            final_image = await self._postprocess_output(enhanced_image)
            quality_score = await self._evaluate_enhancement_quality(
                original=image_tensor, 
                enhanced=final_image
            )
            
            # 5. 결과 구성
            processing_time = time.time() - start_time
            
            result = {
                'success': True,
                'enhanced_image': final_image,
                'original_shape': original_shape,
                'final_shape': final_image.shape,
                'quality_score': quality_score,
                'processing_time': processing_time,
                'enhancement_log': processing_log,
                'applied_enhancements': [log['step'] for log in processing_log],
                'target_achieved': quality_score >= quality_target,
                'device_used': self.device,
                'config_used': options
            }
            
            # 6. 통계 업데이트
            self._update_processing_stats(processing_time, quality_score)
            
            logger.info(f"✅ 후처리 완료 - 품질: {quality_score:.3f}, 시간: {processing_time:.2f}초")
            
            return result
            
        except Exception as e:
            error_msg = f"후처리 실패: {str(e)}"
            logger.error(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'processing_time': time.time() - start_time
            }
    
    async def _preprocess_input(self, input_image: Union[np.ndarray, torch.Tensor, str]) -> torch.Tensor:
        """입력 전처리"""
        try:
            if isinstance(input_image, str):
                # Base64 디코딩
                import base64
                import io
                from PIL import Image
                
                if input_image.startswith('data:image'):
                    header, data = input_image.split(',', 1)
                    image_data = base64.b64decode(data)
                else:
                    image_data = base64.b64decode(input_image)
                
                pil_image = Image.open(io.BytesIO(image_data))
                image_np = np.array(pil_image)
                
            elif isinstance(input_image, np.ndarray):
                image_np = input_image.copy()
                
            elif isinstance(input_image, torch.Tensor):
                return input_image.to(self.device)
                
            else:
                raise ValueError(f"지원하지 않는 입력 타입: {type(input_image)}")
            
            # NumPy를 PyTorch 텐서로 변환
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                # HWC -> CHW
                tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
                
                # 정규화 (0-255 -> 0-1)
                if tensor.max() > 1.0:
                    tensor = tensor / 255.0
                
                # 배치 차원 추가
                if len(tensor.shape) == 3:
                    tensor = tensor.unsqueeze(0)
                
                return tensor.to(self.device)
            
            else:
                raise ValueError(f"지원하지 않는 이미지 형태: {image_np.shape}")
                
        except Exception as e:
            logger.error(f"입력 전처리 실패: {e}")
            # 기본 더미 텐서 반환
            return torch.zeros(1, 3, 512, 512, device=self.device)
    
    async def _apply_super_resolution(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Super Resolution 적용"""
        try:
            start_time = time.time()
            
            if hasattr(self.real_esrgan, 'enhance'):
                # 실제 Real-ESRGAN 모델
                enhanced = await asyncio.to_thread(self.real_esrgan.enhance, image)
            else:
                # 폴백: 기본 업스케일링
                enhanced = await asyncio.to_thread(self.real_esrgan.upscale, image)
            
            processing_time = time.time() - start_time
            
            # 품질 메트릭 계산
            metrics = {
                'processing_time': processing_time,
                'scale_factor': enhanced.shape[-1] / image.shape[-1],
                'improvement_score': self._calculate_sharpness_improvement(image, enhanced)
            }
            
            return enhanced, metrics
            
        except Exception as e:
            logger.warning(f"Super Resolution 실패: {e}")
            return image, {'error': str(e)}
    
    async def _apply_face_enhancement(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """얼굴 향상 적용"""
        try:
            start_time = time.time()
            
            if hasattr(self.gfpgan, 'enhance'):
                enhanced = await asyncio.to_thread(self.gfpgan.enhance, image)
            else:
                enhanced = await asyncio.to_thread(self.gfpgan.process, image)
            
            processing_time = time.time() - start_time
            
            metrics = {
                'processing_time': processing_time,
                'face_regions_processed': self._count_face_regions(image),
                'enhancement_strength': 0.7  # 기본값
            }
            
            return enhanced, metrics
            
        except Exception as e:
            logger.warning(f"얼굴 향상 실패: {e}")
            return image, {'error': str(e)}
    
    async def _apply_image_restoration(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """이미지 복원 적용"""
        try:
            start_time = time.time()
            
            if hasattr(self.codeformer, 'restore'):
                enhanced = await asyncio.to_thread(self.codeformer.restore, image)
            else:
                enhanced = await asyncio.to_thread(self.codeformer.process, image)
            
            processing_time = time.time() - start_time
            
            metrics = {
                'processing_time': processing_time,
                'artifacts_removed': self._estimate_artifacts_removed(image, enhanced),
                'detail_preservation': self._calculate_detail_preservation(image, enhanced)
            }
            
            return enhanced, metrics
            
        except Exception as e:
            logger.warning(f"이미지 복원 실패: {e}")
            return image, {'error': str(e)}
    
    async def _apply_color_correction(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """색상 보정 적용"""
        try:
            start_time = time.time()
            
            enhanced = await asyncio.to_thread(self.color_enhancer.correct_colors, image)
            
            processing_time = time.time() - start_time
            
            metrics = {
                'processing_time': processing_time,
                'color_balance_improvement': self._calculate_color_balance_improvement(image, enhanced),
                'saturation_adjustment': self._calculate_saturation_change(image, enhanced)
            }
            
            return enhanced, metrics
            
        except Exception as e:
            logger.warning(f"색상 보정 실패: {e}")
            return image, {'error': str(e)}
    
    async def _apply_noise_reduction(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """노이즈 제거 적용"""
        try:
            start_time = time.time()
            
            enhanced = await asyncio.to_thread(self.noise_reducer.reduce_noise, image)
            
            processing_time = time.time() - start_time
            
            metrics = {
                'processing_time': processing_time,
                'noise_reduction_amount': self._calculate_noise_reduction(image, enhanced),
                'detail_preservation': self._calculate_detail_preservation(image, enhanced)
            }
            
            return enhanced, metrics
            
        except Exception as e:
            logger.warning(f"노이즈 제거 실패: {e}")
            return image, {'error': str(e)}
    
    async def _apply_edge_enhancement(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """엣지 향상 적용"""
        try:
            start_time = time.time()
            
            enhanced = await asyncio.to_thread(self.edge_enhancer.enhance_edges, image)
            
            processing_time = time.time() - start_time
            
            metrics = {
                'processing_time': processing_time,
                'edge_strength_improvement': self._calculate_edge_improvement(image, enhanced),
                'sharpness_gain': self._calculate_sharpness_improvement(image, enhanced)
            }
            
            return enhanced, metrics
            
        except Exception as e:
            logger.warning(f"엣지 향상 실패: {e}")
            return image, {'error': str(e)}
    
    async def _postprocess_output(self, image: torch.Tensor) -> torch.Tensor:
        """출력 후처리"""
        try:
            # 텐서 정규화 및 클리핑
            image = torch.clamp(image, 0.0, 1.0)
            
            # 최종 품질 조정
            if self.enhancement_config.get('final_adjustment', True):
                image = self._apply_final_adjustments(image)
            
            return image
            
        except Exception as e:
            logger.warning(f"출력 후처리 실패: {e}")
            return image
    
    def _apply_final_adjustments(self, image: torch.Tensor) -> torch.Tensor:
        """최종 조정 적용"""
        try:
            # 약간의 선명도 향상
            if self.enhancement_config.get('final_sharpening', True):
                image = self._apply_unsharp_mask(image, strength=0.2)
            
            # 색상 미세 조정
            if self.enhancement_config.get('final_color_boost', True):
                image = self._boost_colors(image, factor=0.1)
            
            return image
            
        except Exception as e:
            logger.warning(f"최종 조정 실패: {e}")
            return image
    
    def _apply_unsharp_mask(self, image: torch.Tensor, strength: float = 0.2) -> torch.Tensor:
        """언샵 마스크 적용"""
        try:
            # 가우시안 블러
            blurred = F.conv2d(
                image,
                self._get_gaussian_kernel(5, 1.0).to(image.device),
                padding=2,
                groups=image.shape[1]
            )
            
            # 언샵 마스크
            unsharp = image + strength * (image - blurred)
            
            return torch.clamp(unsharp, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"언샵 마스크 실패: {e}")
            return image
    
    def _boost_colors(self, image: torch.Tensor, factor: float = 0.1) -> torch.Tensor:
        """색상 부스트"""
        try:
            # RGB를 HSV로 변환 (근사)
            # 단순화된 채도 증가
            mean_brightness = torch.mean(image, dim=1, keepdim=True)
            color_deviation = image - mean_brightness
            
            # 채도 증가
            boosted = image + factor * color_deviation
            
            return torch.clamp(boosted, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"색상 부스트 실패: {e}")
            return image
    
    def _get_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """가우시안 커널 생성"""
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        # 2D 커널
        kernel = g[:, None] * g[None, :]
        
        # 3채널용으로 확장
        kernel = kernel.expand(3, 1, size, size)
        
        return kernel
    
    async def _evaluate_enhancement_quality(self, original: torch.Tensor, enhanced: torch.Tensor) -> float:
        """향상 품질 평가"""
        try:
            # 여러 품질 메트릭 조합
            
            # 1. 선명도 개선
            sharpness_gain = self._calculate_sharpness_improvement(original, enhanced)
            
            # 2. 디테일 보존
            detail_preservation = self._calculate_detail_preservation(original, enhanced)
            
            # 3. 색상 자연스러움
            color_naturalness = self._calculate_color_naturalness(enhanced)
            
            # 4. 아티팩트 수준
            artifact_level = self._estimate_artifact_level(enhanced)
            
            # 종합 점수
            quality_score = (
                sharpness_gain * 0.3 +
                detail_preservation * 0.25 +
                color_naturalness * 0.25 +
                (1.0 - artifact_level) * 0.2
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.warning(f"품질 평가 실패: {e}")
            return 0.5
    
    def _calculate_sharpness_improvement(self, original: torch.Tensor, enhanced: torch.Tensor) -> float:
        """선명도 개선 계산"""
        try:
            # 라플라시안 분산으로 선명도 측정
            laplacian_kernel = torch.tensor([[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]], 
                                          dtype=torch.float32, device=original.device)
            
            orig_sharpness = self._calculate_laplacian_variance(original, laplacian_kernel)
            enhanced_sharpness = self._calculate_laplacian_variance(enhanced, laplacian_kernel)
            
            if orig_sharpness > 0:
                improvement = (enhanced_sharpness - orig_sharpness) / orig_sharpness
                return max(0.0, min(1.0, improvement + 0.5))  # 0.5 기준점
            
            return 0.5
            
        except Exception as e:
            logger.warning(f"선명도 개선 계산 실패: {e}")
            return 0.5
    
    def _calculate_laplacian_variance(self, image: torch.Tensor, kernel: torch.Tensor) -> float:
        """라플라시안 분산 계산"""
        try:
            # 그레이스케일 변환
            if image.shape[1] == 3:
                gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
            else:
                gray = image
            
            # 라플라시안 적용
            laplacian = F.conv2d(gray, kernel.unsqueeze(0), padding=1)
            
            return float(torch.var(laplacian))
            
        except Exception as e:
            logger.warning(f"라플라시안 분산 계산 실패: {e}")
            return 0.0
    
    def _calculate_detail_preservation(self, original: torch.Tensor, enhanced: torch.Tensor) -> float:
        """디테일 보존도 계산"""
        try:
            # 고주파 성분 비교
            orig_details = self._extract_high_frequency(original)
            enhanced_details = self._extract_high_frequency(enhanced)
            
            # 상관관계 계산
            correlation = F.cosine_similarity(
                orig_details.flatten(), 
                enhanced_details.flatten(), 
                dim=0
            )
            
            return float((correlation + 1.0) / 2.0)  # -1~1을 0~1로 변환
            
        except Exception as e:
            logger.warning(f"디테일 보존도 계산 실패: {e}")
            return 0.5
    
    def _extract_high_frequency(self, image: torch.Tensor) -> torch.Tensor:
        """고주파 성분 추출"""
        try:
            # 고주파 필터 (라플라시안)
            kernel = torch.tensor([[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]], 
                                dtype=torch.float32, device=image.device)
            
            if image.shape[1] == 3:
                # 그레이스케일 변환
                gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
            else:
                gray = image
            
            high_freq = F.conv2d(gray, kernel.unsqueeze(0), padding=1)
            
            return high_freq
            
        except Exception as e:
            logger.warning(f"고주파 성분 추출 실패: {e}")
            return torch.zeros_like(image[:, 0:1])
    
    def _calculate_color_naturalness(self, image: torch.Tensor) -> float:
        """색상 자연스러움 계산"""
        try:
            # RGB 값 분포 분석
            r_mean = torch.mean(image[:, 0])
            g_mean = torch.mean(image[:, 1])
            b_mean = torch.mean(image[:, 2])
            
            # 색상 균형 검사 (자연스러운 이미지는 적절한 균형을 가짐)
            color_balance = 1.0 - torch.std(torch.tensor([r_mean, g_mean, b_mean]))
            
            # 채도 검사 (과도한 채도는 부자연스러움)
            saturation = torch.std(image, dim=1).mean()
            saturation_score = 1.0 - torch.clamp(saturation - 0.2, 0, 1)
            
            # 조합
            naturalness = (color_balance * 0.6 + saturation_score * 0.4)
            
            return float(torch.clamp(naturalness, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"색상 자연스러움 계산 실패: {e}")
            return 0.5
    
    def _estimate_artifact_level(self, image: torch.Tensor) -> float:
        """아티팩트 수준 추정"""
        try:
            # 블로킹 아티팩트 검출
            blocking_score = self._detect_blocking_artifacts(image)
            
            # 링잉 아티팩트 검출
            ringing_score = self._detect_ringing_artifacts(image)
            
            # 전체 아티팩트 수준
            artifact_level = (blocking_score + ringing_score) / 2.0
            
            return float(torch.clamp(artifact_level, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"아티팩트 수준 추정 실패: {e}")
            return 0.3
    
    def _detect_blocking_artifacts(self, image: torch.Tensor) -> torch.Tensor:
        """블로킹 아티팩트 검출"""
        try:
            # 8x8 블록 경계 불연속성 검사
            b, c, h, w = image.shape
            
            # 수직 경계 검사
            vertical_diff = torch.abs(image[:, :, :, 8::8] - image[:, :, :, 7::8])
            
            # 수평 경계 검사  
            horizontal_diff = torch.abs(image[:, :, 8::8, :] - image[:, :, 7::8, :])
            
            # 평균 불연속성
            blocking_level = (torch.mean(vertical_diff) + torch.mean(horizontal_diff)) / 2.0
            
            return blocking_level
            
        except Exception as e:
            logger.warning(f"블로킹 아티팩트 검출 실패: {e}")
            return torch.tensor(0.0)
    
    def _detect_ringing_artifacts(self, image: torch.Tensor) -> torch.Tensor:
        """링잉 아티팩트 검출"""
        try:
            # 라플라시안으로 엣지 주변 진동 검출
            laplacian_kernel = torch.tensor([[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]], 
                                          dtype=torch.float32, device=image.device)
            
            # 그레이스케일 변환
            if image.shape[1] == 3:
                gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
            else:
                gray = image
            
            # 라플라시안 적용
            laplacian = F.conv2d(gray, laplacian_kernel.unsqueeze(0), padding=1)
            
            # 링잉은 라플라시안의 과도한 변동으로 나타남
            ringing_level = torch.std(laplacian)
            
            return ringing_level / 10.0  # 정규화
            
        except Exception as e:
            logger.warning(f"링잉 아티팩트 검출 실패: {e}")
            return torch.tensor(0.0)
    
    def _count_face_regions(self, image: torch.Tensor) -> int:
        """얼굴 영역 카운트 (간단 버전)"""
        try:
            # 간단한 얼굴 영역 추정
            # 실제로는 얼굴 검출기를 사용해야 함
            return 1  # 기본값
            
        except Exception as e:
            logger.warning(f"얼굴 영역 카운트 실패: {e}")
            return 0
    
    def _estimate_artifacts_removed(self, original: torch.Tensor, enhanced: torch.Tensor) -> float:
        """제거된 아티팩트 추정"""
        try:
            orig_artifacts = self._estimate_artifact_level(original)
            enhanced_artifacts = self._estimate_artifact_level(enhanced)
            
            artifacts_removed = float(orig_artifacts - enhanced_artifacts)
            
            return max(0.0, artifacts_removed)
            
        except Exception as e:
            logger.warning(f"아티팩트 제거 추정 실패: {e}")
            return 0.0
    
    def _calculate_color_balance_improvement(self, original: torch.Tensor, enhanced: torch.Tensor) -> float:
        """색상 균형 개선 계산"""
        try:
            orig_balance = self._calculate_color_naturalness(original)
            enhanced_balance = self._calculate_color_naturalness(enhanced)
            
            improvement = enhanced_balance - orig_balance
            
            return float(max(0.0, improvement))
            
        except Exception as e:
            logger.warning(f"색상 균형 개선 계산 실패: {e}")
            return 0.0
    
    def _calculate_saturation_change(self, original: torch.Tensor, enhanced: torch.Tensor) -> float:
        """채도 변화 계산"""
        try:
            orig_saturation = torch.std(original, dim=1).mean()
            enhanced_saturation = torch.std(enhanced, dim=1).mean()
            
            saturation_change = float(enhanced_saturation - orig_saturation)
            
            return saturation_change
            
        except Exception as e:
            logger.warning(f"채도 변화 계산 실패: {e}")
            return 0.0
    
    def _calculate_noise_reduction(self, original: torch.Tensor, enhanced: torch.Tensor) -> float:
        """노이즈 감소량 계산"""
        try:
            # 고주파 노이즈 비교
            orig_noise = torch.std(self._extract_high_frequency(original))
            enhanced_noise = torch.std(self._extract_high_frequency(enhanced))
            
            noise_reduction = float(orig_noise - enhanced_noise)
            
            return max(0.0, noise_reduction)
            
        except Exception as e:
            logger.warning(f"노이즈 감소량 계산 실패: {e}")
            return 0.0
    
    def _calculate_edge_improvement(self, original: torch.Tensor, enhanced: torch.Tensor) -> float:
        """엣지 개선 계산"""
        try:
            # 엣지 강도 비교
            orig_edges = self._calculate_edge_strength(original)
            enhanced_edges = self._calculate_edge_strength(enhanced)
            
            edge_improvement = float(enhanced_edges - orig_edges)
            
            return max(0.0, edge_improvement)
            
        except Exception as e:
            logger.warning(f"엣지 개선 계산 실패: {e}")
            return 0.0
    
    def _calculate_edge_strength(self, image: torch.Tensor) -> torch.Tensor:
        """엣지 강도 계산"""
        try:
            # Sobel 필터
            sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], 
                                 dtype=torch.float32, device=image.device)
            sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], 
                                 dtype=torch.float32, device=image.device)
            
            # 그레이스케일 변환
            if image.shape[1] == 3:
                gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
            else:
                gray = image
            
            # Sobel 적용
            edge_x = F.conv2d(gray, sobel_x.unsqueeze(0), padding=1)
            edge_y = F.conv2d(gray, sobel_y.unsqueeze(0), padding=1)
            
            # 엣지 크기
            edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)
            
            return torch.mean(edge_magnitude)
            
        except Exception as e:
            logger.warning(f"엣지 강도 계산 실패: {e}")
            return torch.tensor(0.0)
    
    def _update_processing_stats(self, processing_time: float, quality_score: float):
        """처리 통계 업데이트"""
        try:
            self.processing_stats['total_images'] += 1
            
            # 평균 시간 업데이트
            total_images = self.processing_stats['total_images']
            old_avg_time = self.processing_stats['average_time']
            self.processing_stats['average_time'] = (
                (old_avg_time * (total_images - 1) + processing_time) / total_images
            )
            
            # 성공률 업데이트 (품질 점수 0.6 이상을 성공으로 간주)
            success_count = self.processing_stats.get('success_count', 0)
            if quality_score >= 0.6:
                success_count += 1
            
            self.processing_stats['success_count'] = success_count
            self.processing_stats['enhancement_success_rate'] = success_count / total_images
            
        except Exception as e:
            logger.warning(f"통계 업데이트 실패: {e}")
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        return {
            **self.processing_stats,
            'device': self.device,
            'models_loaded': {
                'real_esrgan': self.real_esrgan is not None,
                'gfpgan': self.gfpgan is not None,
                'codeformer': self.codeformer is not None,
                'color_enhancer': self.color_enhancer is not None,
                'noise_reducer': self.noise_reducer is not None,
                'edge_enhancer': self.edge_enhancer is not None
            },
            'is_initialized': self.is_initialized
        }
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            # 모델 정리
            if self.real_esrgan and hasattr(self.real_esrgan, 'cleanup'):
                await self.real_esrgan.cleanup()
            
            if self.gfpgan and hasattr(self.gfpgan, 'cleanup'):
                await self.gfpgan.cleanup()
            
            if self.codeformer and hasattr(self.codeformer, 'cleanup'):
                await self.codeformer.cleanup()
            
            # 메모리 정리
            if self.memory_manager:
                await self.memory_manager.cleanup()
            
            # GPU 메모리 정리
            if self.use_mps:
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_initialized = False
            logger.info("✅ Step 7 후처리 리소스 정리 완료")
            
        except Exception as e:
            logger.warning(f"리소스 정리 중 오류: {e}")


# === 폴백 클래스들 ===

class BasicUpscaler:
    """기본 업스케일러 (Real-ESRGAN 폴백)"""
    
    def __init__(self, scale_factor: int = 2):
        self.scale_factor = scale_factor
    
    def upscale(self, image: torch.Tensor) -> torch.Tensor:
        """기본 바이큐빅 업스케일링"""
        try:
            b, c, h, w = image.shape
            new_h, new_w = h * self.scale_factor, w * self.scale_factor
            
            upscaled = F.interpolate(
                image, 
                size=(new_h, new_w), 
                mode='bicubic', 
                align_corners=False
            )
            
            return upscaled
            
        except Exception as e:
            logger.warning(f"기본 업스케일링 실패: {e}")
            return image


class BasicFaceEnhancer:
    """기본 얼굴 향상기 (GFPGAN 폴백)"""
    
    def process(self, image: torch.Tensor) -> torch.Tensor:
        """기본 얼굴 향상"""
        try:
            # 간단한 선명화 적용
            kernel = torch.tensor([[[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]], 
                                dtype=torch.float32, device=image.device) / 8.0
            
            if image.shape[1] == 3:
                enhanced = F.conv2d(image, kernel.repeat(3, 1, 1, 1), padding=1, groups=3)
            else:
                enhanced = F.conv2d(image, kernel, padding=1)
            
            return torch.clamp(enhanced, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"기본 얼굴 향상 실패: {e}")
            return image


class BasicImageRestorer:
    """기본 이미지 복원기 (CodeFormer 폴백)"""
    
    def process(self, image: torch.Tensor) -> torch.Tensor:
        """기본 이미지 복원"""
        try:
            # 가우시안 노이즈 제거
            kernel_size = 3
            sigma = 0.5
            
            # 가우시안 커널 생성
            coords = torch.arange(kernel_size, dtype=torch.float32, device=image.device)
            coords -= kernel_size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g /= g.sum()
            
            kernel = g[:, None] * g[None, :]
            kernel = kernel.expand(image.shape[1], 1, kernel_size, kernel_size)
            
            # 적용
            restored = F.conv2d(image, kernel, padding=kernel_size//2, groups=image.shape[1])
            
            return restored
            
        except Exception as e:
            logger.warning(f"기본 이미지 복원 실패: {e}")
            return image


class ColorEnhancer:
    """색상 향상기"""
    
    def correct_colors(self, image: torch.Tensor) -> torch.Tensor:
        """색상 보정"""
        try:
            # 간단한 색상 균형 조정
            r_mean = torch.mean(image[:, 0])
            g_mean = torch.mean(image[:, 1])
            b_mean = torch.mean(image[:, 2])
            
            overall_mean = (r_mean + g_mean + b_mean) / 3.0
            
            # 각 채널 조정
            r_factor = overall_mean / (r_mean + 1e-8)
            g_factor = overall_mean / (g_mean + 1e-8)
            b_factor = overall_mean / (b_mean + 1e-8)
            
            # 부드러운 조정 (너무 급격하지 않게)
            r_factor = 1.0 + 0.1 * (r_factor - 1.0)
            g_factor = 1.0 + 0.1 * (g_factor - 1.0)
            b_factor = 1.0 + 0.1 * (b_factor - 1.0)
            
            corrected = image.clone()
            corrected[:, 0] *= r_factor
            corrected[:, 1] *= g_factor
            corrected[:, 2] *= b_factor
            
            return torch.clamp(corrected, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"색상 보정 실패: {e}")
            return image


class NoiseReducer:
    """노이즈 제거기"""
    
    def reduce_noise(self, image: torch.Tensor) -> torch.Tensor:
        """노이즈 제거"""
        try:
            # 양방향 필터 근사 (가우시안 블러)
            kernel_size = 5
            sigma = 1.0
            
            # 가우시안 커널 생성
            coords = torch.arange(kernel_size, dtype=torch.float32, device=image.device)
            coords -= kernel_size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g /= g.sum()
            
            kernel = g[:, None] * g[None, :]
            kernel = kernel.expand(image.shape[1], 1, kernel_size, kernel_size)
            
            # 노이즈 제거
            denoised = F.conv2d(image, kernel, padding=kernel_size//2, groups=image.shape[1])
            
            # 원본과 블렌딩 (디테일 보존)
            blending_factor = 0.7
            result = blending_factor * denoised + (1 - blending_factor) * image
            
            return result
            
        except Exception as e:
            logger.warning(f"노이즈 제거 실패: {e}")
            return image


class EdgeEnhancer:
    """엣지 향상기"""
    
    def enhance_edges(self, image: torch.Tensor) -> torch.Tensor:
        """엣지 향상"""
        try:
            # 언샵 마스크 적용
            # 1. 가우시안 블러
            kernel_size = 5
            sigma = 1.0
            
            coords = torch.arange(kernel_size, dtype=torch.float32, device=image.device)
            coords -= kernel_size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g /= g.sum()
            
            kernel = g[:, None] * g[None, :]
            kernel = kernel.expand(image.shape[1], 1, kernel_size, kernel_size)
            
            blurred = F.conv2d(image, kernel, padding=kernel_size//2, groups=image.shape[1])
            
            # 2. 언샵 마스크
            strength = 0.3
            enhanced = image + strength * (image - blurred)
            
            return torch.clamp(enhanced, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"엣지 향상 실패: {e}")
            return image


# === 테스트 함수 ===
async def test_post_processing():
    """후처리 테스트"""
    
    print("🎨 Step 7 후처리 테스트 시작")
    print("=" * 50)
    
    # 1. 후처리 시스템 초기화
    post_processor = PostProcessingStep(
        device='cpu',  # 또는 'mps'
        config={
            'post_processing': {
                'super_resolution': True,
                'face_enhancement': True,
                'image_restoration': True,
                'color_correction': True,
                'noise_reduction': True,
                'edge_enhancement': True,
                'quality_level': 'high'
            }
        }
    )
    
    # 2. 초기화
    success = await post_processor.initialize()
    if not success:
        print("❌ 후처리 시스템 초기화 실패")
        return
    
    print("✅ 후처리 시스템 초기화 완료")
    
    # 3. 테스트 이미지 생성
    print("📝 테스트 이미지 생성 중...")
    test_image = torch.randn(1, 3, 256, 256)  # 더미 이미지
    
    # 4. 후처리 실행
    print("🚀 후처리 시작...")
    
    result = await post_processor.process(
        input_image=test_image,
        enhancement_options={
            'super_resolution': True,
            'face_enhancement': False,  # 얼굴이 없는 더미 이미지
            'color_correction': True
        },
        quality_target=0.8
    )
    
    # 5. 결과 출력
    if result['success']:
        print("\n" + "="*50)
        print("🎨 후처리 결과")
        print("="*50)
        
        print(f"✅ 처리 성공!")
        print(f"⏱️ 처리 시간: {result['processing_time']:.2f}초")
        print(f"📊 품질 점수: {result['quality_score']:.3f}")
        print(f"🎯 목표 달성: {'예' if result['target_achieved'] else '아니오'}")
        print(f"🔧 사용된 디바이스: {result['device_used']}")
        
        print(f"\n🔄 적용된 향상:")
        for enhancement in result['applied_enhancements']:
            print(f"  • {enhancement.replace('_', ' ').title()}")
        
        print(f"\n📋 세부 로그:")
        for i, log_entry in enumerate(result['enhancement_log'], 1):
            step = log_entry['step'].replace('_', ' ').title()
            metrics = log_entry['metrics']
            
            print(f"  {i}. {step}:")
            if 'processing_time' in metrics:
                print(f"     - 처리 시간: {metrics['processing_time']:.3f}초")
            
            for key, value in metrics.items():
                if key != 'processing_time' and not key.startswith('error'):
                    if isinstance(value, (int, float)):
                        print(f"     - {key.replace('_', ' ').title()}: {value:.3f}")
                    else:
                        print(f"     - {key.replace('_', ' ').title()}: {value}")
        
        # 6. 통계 정보
        stats = await post_processor.get_processing_stats()
        print(f"\n📊 처리 통계:")
        print(f"  • 총 처리 이미지: {stats['total_images']}")
        print(f"  • 평균 처리 시간: {stats['average_time']:.2f}초")
        print(f"  • 향상 성공률: {stats['enhancement_success_rate']:.1%}")
        
        print(f"\n🤖 로드된 모델:")
        for model_name, loaded in stats['models_loaded'].items():
            status = "✅" if loaded else "❌"
            print(f"  {status} {model_name.replace('_', ' ').title()}")
        
    else:
        print(f"❌ 후처리 실패: {result['error']}")
    
    # 7. 정리
    await post_processor.cleanup()
    print("\n✅ 테스트 완료")


if __name__ == "__main__":
    print("🎨 실제 후처리 시스템 테스트")
    print("=" * 50)
    
    # 테스트 실행
    asyncio.run(test_post_processing())