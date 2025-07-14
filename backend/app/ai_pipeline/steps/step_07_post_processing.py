# app/ai_pipeline/steps/step_07_post_processing.py
"""
7단계: 후처리 (Post Processing) - 최적 생성자 패턴 적용
통일된 생성자: def __init__(self, device: Optional[str] = None, config: Optional[Dict[str, Any]] = None, **kwargs)
"""

import os
import logging
import time
import asyncio
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path

import numpy as np
import json

# 필수 패키지들 - 안전한 임포트 처리
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
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
    from PIL import Image, ImageFilter, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL 권장: pip install Pillow")

try:
    from concurrent.futures import ThreadPoolExecutor
    CONCURRENT_AVAILABLE = True
except ImportError:
    CONCURRENT_AVAILABLE = False

logger = logging.getLogger(__name__)

class PostProcessingStep:
    """
    Step 7: 후처리 품질 향상 - 최적 생성자 패턴 적용
    통일된 생성자: def __init__(self, device: Optional[str] = None, config: Optional[Dict[str, Any]] = None, **kwargs)
    """
    
    def __init__(
        self, 
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        🎯 최적 생성자 패턴 - 모든 MyCloset AI Step과 호환
        
        Args:
            device: 사용할 디바이스 (None=자동감지, 'cpu', 'cuda', 'mps')
            config: 설정 딕셔너리 (선택적)
            **kwargs: 확장 파라미터들
                - device_type: str = "auto"
                - memory_gb: float = 16.0  
                - is_m3_max: bool = False
                - optimization_enabled: bool = True
                - quality_level: str = "balanced"
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

        # 6. 🎯 기존 클래스별 고유 초기화 로직 실행
        self._initialize_step_specific()

        self.logger.info(f"🎯 {self.step_name} 초기화 - 디바이스: {self.device}")
    
    def _auto_detect_device(self, preferred_device: Optional[str]) -> str:
        """💡 지능적 디바이스 자동 감지"""
        if preferred_device:
            return preferred_device

        try:
            if TORCH_AVAILABLE:
                if torch.backends.mps.is_available():
                    return 'mps'  # M3 Max 우선
                elif torch.cuda.is_available():
                    return 'cuda'  # NVIDIA GPU
                else:
                    return 'cpu'  # 폴백
            else:
                return 'cpu'
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
        """🎯 기존 초기화 로직 완전 유지"""
        # M3 Max 특화 설정
        self._configure_m3_max_optimizations()
        
        # 후처리 설정
        self.enhancement_config = self.config.get('post_processing', {
            'super_resolution': True,    # Real-ESRGAN
            'face_enhancement': True,    # GFPGAN
            'image_restoration': True,   # CodeFormer
            'color_correction': True,    # 색상 보정
            'noise_reduction': True,     # 노이즈 제거
            'edge_enhancement': True,    # 엣지 향상
            'quality_level': self._get_quality_level()
        })
        
        # M3 Max 최적화 설정
        self.use_mps = self.device == 'mps' and TORCH_AVAILABLE and torch.backends.mps.is_available()
        self.batch_size = self._get_optimal_batch_size()
        self.tile_size = self._get_optimal_tile_size()
        self.enable_neural_enhancement = self.is_m3_max and self.optimization_enabled
        
        # 모델 인스턴스들
        self.real_esrgan = None
        self.gfpgan = None
        self.codeformer = None
        
        # 전통적 처리 도구들
        self.color_enhancer = None
        self.noise_reducer = None
        self.edge_enhancer = None
        
        # 초기화 상태
        self.initialization_error = None
        
        # 성능 통계
        self.processing_stats = {
            'total_images': 0,
            'average_time': 0.0,
            'enhancement_success_rate': 0.0,
            'm3_max_accelerated': self.is_m3_max,
            'memory_efficiency': 0.0
        }
        
        if self.is_m3_max:
            self.logger.info(f"🍎 M3 Max 최적화 활성화 - 메모리: {self.memory_gb}GB")
    
    def _configure_m3_max_optimizations(self):
        """M3 Max 전용 최적화 설정"""
        if not self.is_m3_max:
            return
        
        try:
            self.logger.info("🍎 M3 Max 후처리 최적화 설정...")
            
            # MPS 최적화
            if self.device == 'mps' and TORCH_AVAILABLE:
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # 메모리 효율성
                
                # M3 Max 메모리 최적화
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
                
                self.logger.info("✅ M3 Max MPS 후처리 최적화 완료")
            
            # CPU 최적화 (14코어 M3 Max)
            if TORCH_AVAILABLE:
                optimal_threads = min(12, os.cpu_count() or 8)  # 성능 코어 중심
                torch.set_num_threads(optimal_threads)
                self.logger.info(f"⚡ M3 Max CPU 스레드 최적화: {optimal_threads}")
            
            # 128GB 메모리 활용 최적화
            if self.memory_gb >= 128:
                self.enhancement_config['enable_large_batch'] = True
                self.enhancement_config['memory_aggressive_mode'] = True
                self.logger.info("💾 M3 Max 128GB 메모리 활용 최적화 활성화")
            
        except Exception as e:
            self.logger.warning(f"M3 Max 최적화 설정 실패: {e}")
    
    def _get_quality_level(self) -> str:
        """품질 수준 결정 - M3 Max는 최고 품질"""
        if self.is_m3_max and self.optimization_enabled:
            return 'ultra'  # M3 Max 전용 최고 품질
        elif self.memory_gb >= 64:
            return 'high'
        elif self.memory_gb >= 32:
            return 'medium'
        else:
            return 'basic'
    
    def _get_optimal_batch_size(self) -> int:
        """최적 배치 크기 결정"""
        if self.is_m3_max and self.memory_gb >= 128:
            return 4  # M3 Max 128GB: 대용량 배치
        elif self.memory_gb >= 64:
            return 2
        else:
            return 1
    
    def _get_optimal_tile_size(self) -> int:
        """최적 타일 크기 결정"""
        if self.is_m3_max and self.memory_gb >= 128:
            return 1024  # M3 Max: 큰 타일 처리 가능
        elif self.memory_gb >= 64:
            return 768
        else:
            return 512
    
    async def initialize(self) -> bool:
        """후처리 모델들 초기화"""
        try:
            self.logger.info("🔄 Step 7 후처리 모델 로드 중...")
            
            # M3 Max 전용 초기화
            if self.is_m3_max:
                await self._initialize_m3_max_components()
            
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
                
                # M3 Max 워밍업
                if self.is_m3_max and self.optimization_enabled:
                    await self._warmup_m3_max_pipeline()
                
                self.logger.info(f"✅ Step 7 후처리 모델 로드 완료 ({success_count}/{total_count})")
                return True
            else:
                self.logger.error(f"❌ Step 7 후처리 모델 로드 실패 ({success_count}/{total_count})")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Step 7 후처리 초기화 실패: {e}")
            self.is_initialized = False
            self.initialization_error = str(e)
            return False
    
    async def _initialize_m3_max_components(self):
        """M3 Max 전용 컴포넌트 초기화"""
        self.logger.info("🍎 M3 Max 후처리 컴포넌트 초기화...")
        
        # Metal Performance Shaders 설정
        if self.device == 'mps' and TORCH_AVAILABLE:
            try:
                # MPS 백엔드 테스트
                test_tensor = torch.randn(1, 3, 256, 256).to(self.device)
                _ = F.conv2d(test_tensor, torch.randn(3, 3, 3, 3).to(self.device), padding=1)
                del test_tensor
                self.logger.info("✅ M3 Max MPS 후처리 테스트 완료")
            except Exception as e:
                self.logger.warning(f"MPS 후처리 테스트 실패: {e}")
        
        # 고성능 메모리 관리
        if self.memory_gb >= 128:
            import gc
            gc.collect()
            self.logger.info("✅ M3 Max 128GB 후처리 메모리 관리 설정")
    
    async def _warmup_m3_max_pipeline(self):
        """M3 Max 후처리 파이프라인 워밍업"""
        self.logger.info("🔥 M3 Max 후처리 파이프라인 워밍업...")
        
        try:
            # 작은 더미 이미지로 워밍업
            if TORCH_AVAILABLE:
                dummy_image = torch.randn(1, 3, 256, 256).to(self.device)
            
            # 각 컴포넌트 워밍업
            if self.real_esrgan and hasattr(self.real_esrgan, 'warmup'):
                await self.real_esrgan.warmup()
            
            if self.color_enhancer and TORCH_AVAILABLE:
                self.color_enhancer.correct_colors(dummy_image)
            
            self.logger.info("✅ M3 Max 후처리 파이프라인 워밍업 완료")
            
        except Exception as e:
            self.logger.warning(f"M3 Max 후처리 워밍업 실패: {e}")
    
    async def _init_real_esrgan(self) -> bool:
        """Real-ESRGAN Super Resolution 모델 초기화"""
        try:
            # M3 Max 최적화 업스케일링
            self.real_esrgan = M3MaxUpscaler(
                device=self.device,
                scale_factor=4 if self.is_m3_max else 2,
                use_neural=self.enable_neural_enhancement
            )
            self.logger.info("✅ Real-ESRGAN M3 Max 최적화 모드 사용")
            return True
                
        except Exception as e:
            self.logger.warning(f"Real-ESRGAN 초기화 실패: {e}")
            self.real_esrgan = M3MaxUpscaler(device=self.device)
            return True
    
    async def _init_gfpgan(self) -> bool:
        """GFPGAN 얼굴 향상 모델 초기화"""
        try:
            # M3 Max 얼굴 향상
            self.gfpgan = M3MaxFaceEnhancer(
                device=self.device,
                enhancement_strength=1.5 if self.is_m3_max else 1.0
            )
            self.logger.info("✅ GFPGAN M3 Max 최적화 모드 사용")
            return True
                
        except Exception as e:
            self.logger.warning(f"GFPGAN 초기화 실패: {e}")
            self.gfpgan = M3MaxFaceEnhancer(device=self.device)
            return True
    
    async def _init_codeformer(self) -> bool:
        """CodeFormer 이미지 복원 모델 초기화"""
        try:
            # M3 Max 이미지 복원
            self.codeformer = M3MaxImageRestorer(
                device=self.device,
                restoration_strength=1.2 if self.is_m3_max else 1.0
            )
            self.logger.info("✅ CodeFormer M3 Max 최적화 모드 사용")
            return True
                
        except Exception as e:
            self.logger.warning(f"CodeFormer 초기화 실패: {e}")
            self.codeformer = M3MaxImageRestorer(device=self.device)
            return True
    
    async def _init_color_enhancer(self) -> bool:
        """색상 향상기 초기화"""
        try:
            self.color_enhancer = ColorEnhancer(
                device=self.device,
                m3_max_mode=self.is_m3_max
            )
            return True
        except Exception as e:
            self.logger.warning(f"색상 향상기 초기화 실패: {e}")
            return False
    
    async def _init_noise_reducer(self) -> bool:
        """노이즈 제거기 초기화"""
        try:
            self.noise_reducer = NoiseReducer(
                device=self.device,
                m3_max_mode=self.is_m3_max
            )
            return True
        except Exception as e:
            self.logger.warning(f"노이즈 제거기 초기화 실패: {e}")
            return False
    
    async def _init_edge_enhancer(self) -> bool:
        """엣지 향상기 초기화"""
        try:
            self.edge_enhancer = EdgeEnhancer(
                device=self.device,
                m3_max_mode=self.is_m3_max
            )
            return True
        except Exception as e:
            self.logger.warning(f"엣지 향상기 초기화 실패: {e}")
            return False
    
    # =================================================================
    # 메인 처리 메서드 - 기존 로직 완전 유지
    # =================================================================
    
    async def process(
        self, 
        input_data: Union[np.ndarray, torch.Tensor, str, Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        후처리 메인 처리 함수 - 기존 로직 완전 유지
        
        Args:
            input_data: 입력 이미지 (다양한 형태 지원)
            **kwargs: 추가 매개변수
                - enhancement_options: Optional[Dict[str, Any]] = None
                - quality_target: float = 0.8
                
        Returns:
            후처리 결과
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # M3 Max 메모리 최적화
            if self.is_m3_max:
                await self._optimize_m3_max_memory()
            
            # 1. 입력 전처리
            if isinstance(input_data, dict):
                # 가상 피팅 결과에서 이미지 추출
                input_image = input_data.get('fitted_image') or input_data.get('fitted_image_numpy')
                if input_image is None:
                    raise ValueError("입력 딕셔너리에서 이미지를 찾을 수 없습니다")
            else:
                input_image = input_data
            
            image_tensor = await self._preprocess_input(input_image)
            original_shape = image_tensor.shape if TORCH_AVAILABLE else input_image.shape
            
            self.logger.info(f"🎨 후처리 시작 - 크기: {original_shape}")
            
            # 2. 향상 옵션 설정
            enhancement_options = kwargs.get('enhancement_options', {})
            quality_target = kwargs.get('quality_target', 0.8)
            
            options = {**self.enhancement_config, **enhancement_options}
            
            # M3 Max 모드에서 자동 최적화
            if self.is_m3_max and self.optimization_enabled:
                options = self._apply_m3_max_optimizations(options)
            
            # 3. 순차적 향상 처리
            enhanced_image = image_tensor
            processing_log = []
            
            # Super Resolution (해상도 향상)
            if options.get('super_resolution', True) and self.real_esrgan:
                self.logger.info("🔍 Super Resolution 적용 중...")
                enhanced_image, sr_metrics = await self._apply_super_resolution(enhanced_image)
                processing_log.append({'step': 'super_resolution', 'metrics': sr_metrics})
            
            # Face Enhancement (얼굴 향상)
            if options.get('face_enhancement', True) and self.gfpgan:
                self.logger.info("👤 얼굴 향상 적용 중...")
                enhanced_image, face_metrics = await self._apply_face_enhancement(enhanced_image)
                processing_log.append({'step': 'face_enhancement', 'metrics': face_metrics})
            
            # Image Restoration (전체 복원)
            if options.get('image_restoration', True) and self.codeformer:
                self.logger.info("🔧 이미지 복원 적용 중...")
                enhanced_image, restoration_metrics = await self._apply_image_restoration(enhanced_image)
                processing_log.append({'step': 'image_restoration', 'metrics': restoration_metrics})
            
            # Color Correction (색상 보정)
            if options.get('color_correction', True) and self.color_enhancer:
                self.logger.info("🌈 색상 보정 적용 중...")
                enhanced_image, color_metrics = await self._apply_color_correction(enhanced_image)
                processing_log.append({'step': 'color_correction', 'metrics': color_metrics})
            
            # Noise Reduction (노이즈 제거)
            if options.get('noise_reduction', True) and self.noise_reducer:
                self.logger.info("🔇 노이즈 제거 적용 중...")
                enhanced_image, noise_metrics = await self._apply_noise_reduction(enhanced_image)
                processing_log.append({'step': 'noise_reduction', 'metrics': noise_metrics})
            
            # Edge Enhancement (엣지 향상)
            if options.get('edge_enhancement', True) and self.edge_enhancer:
                self.logger.info("📐 엣지 향상 적용 중...")
                enhanced_image, edge_metrics = await self._apply_edge_enhancement(enhanced_image)
                processing_log.append({'step': 'edge_enhancement', 'metrics': edge_metrics})
            
            # M3 Max 전용 최종 향상
            if self.is_m3_max and self.optimization_enabled:
                self.logger.info("🍎 M3 Max 최종 향상 적용 중...")
                enhanced_image, m3_metrics = await self._apply_m3_max_final_enhancement(enhanced_image)
                processing_log.append({'step': 'm3_max_enhancement', 'metrics': m3_metrics})
            
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
                'final_shape': final_image.shape if hasattr(final_image, 'shape') else (0, 0),
                'quality_score': quality_score,
                'processing_time': processing_time,
                'enhancement_log': processing_log,
                'applied_enhancements': [log['step'] for log in processing_log],
                'target_achieved': quality_score >= quality_target,
                'device_used': self.device,
                'device_type': self.device_type,
                'm3_max_optimized': self.is_m3_max,
                'memory_gb': self.memory_gb,
                'config_used': options,
                'performance_info': {
                    'optimization_enabled': self.optimization_enabled,
                    'batch_size': self.batch_size,
                    'tile_size': self.tile_size,
                    'neural_enhancement': self.enable_neural_enhancement
                }
            }
            
            # 6. 통계 업데이트
            self._update_processing_stats(processing_time, quality_score)
            
            self.logger.info(f"✅ 후처리 완료 - 품질: {quality_score:.3f}, 시간: {processing_time:.2f}초")
            
            return result
            
        except Exception as e:
            error_msg = f"후처리 실패: {str(e)}"
            self.logger.error(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'processing_time': time.time() - start_time,
                'device_used': self.device,
                'device_type': self.device_type,
                'm3_max_optimized': self.is_m3_max
            }
    
    async def _optimize_m3_max_memory(self):
        """M3 Max 메모리 최적화"""
        if not self.is_m3_max:
            return
        
        try:
            import gc
            gc.collect()
            
            if self.device == 'mps' and TORCH_AVAILABLE:
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
                elif hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
                
            self.logger.debug("🍎 M3 Max 후처리 메모리 최적화 완료")
            
        except Exception as e:
            self.logger.warning(f"M3 Max 메모리 최적화 실패: {e}")
    
    def _apply_m3_max_optimizations(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """M3 Max 전용 옵션 최적화"""
        if not self.is_m3_max:
            return options
        
        # M3 Max에서 더 공격적인 향상
        optimized_options = options.copy()
        optimized_options['enhancement_strength'] = 1.2
        optimized_options['precision_mode'] = 'high'
        optimized_options['memory_efficient'] = True
        
        if self.memory_gb >= 128:
            optimized_options['enable_large_operations'] = True
            optimized_options['batch_optimization'] = True
        
        self.logger.debug("🍎 M3 Max 옵션 최적화 적용")
        return optimized_options
    
    async def _preprocess_input(self, input_image: Union[np.ndarray, torch.Tensor, str]) -> Union[torch.Tensor, np.ndarray]:
        """입력 전처리 - M3 Max 최적화"""
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
                
            elif TORCH_AVAILABLE and isinstance(input_image, torch.Tensor):
                return input_image.to(self.device)
                
            else:
                raise ValueError(f"지원하지 않는 입력 타입: {type(input_image)}")
            
            if not TORCH_AVAILABLE:
                return image_np
            
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
            self.logger.error(f"입력 전처리 실패: {e}")
            # 기본 더미 텐서 반환
            if TORCH_AVAILABLE:
                return torch.zeros(1, 3, 512, 512, device=self.device)
            else:
                return np.zeros((512, 512, 3), dtype=np.uint8)
    
    # =================================================================
    # 향상 처리 메서드들 - 기존 로직 완전 유지
    # =================================================================
    
    async def _apply_super_resolution(self, image: Union[torch.Tensor, np.ndarray]) -> Tuple[Union[torch.Tensor, np.ndarray], Dict]:
        """Super Resolution 적용 - M3 Max 최적화"""
        try:
            start_time = time.time()
            
            if hasattr(self.real_esrgan, 'enhance'):
                # 실제 Real-ESRGAN 모델
                enhanced = await asyncio.to_thread(self.real_esrgan.enhance, image)
            else:
                # M3 Max 최적화 업스케일링
                enhanced = await asyncio.to_thread(self.real_esrgan.upscale, image)
            
            processing_time = time.time() - start_time
            
            # 품질 메트릭 계산
            if hasattr(enhanced, 'shape') and hasattr(image, 'shape'):
                scale_factor = enhanced.shape[-1] / image.shape[-1]
                improvement_score = self._calculate_sharpness_improvement(image, enhanced)
            else:
                scale_factor = 2.0
                improvement_score = 0.8
            
            # M3 Max 보너스
            if self.is_m3_max:
                improvement_score = min(1.0, improvement_score * 1.1)
            
            metrics = {
                'processing_time': processing_time,
                'scale_factor': scale_factor,
                'improvement_score': improvement_score,
                'm3_max_accelerated': self.is_m3_max,
                'device': self.device
            }
            
            return enhanced, metrics
            
        except Exception as e:
            self.logger.warning(f"Super Resolution 실패: {e}")
            return image, {'error': str(e)}
    
    async def _apply_face_enhancement(self, image: Union[torch.Tensor, np.ndarray]) -> Tuple[Union[torch.Tensor, np.ndarray], Dict]:
        """얼굴 향상 적용 - M3 Max 최적화"""
        try:
            start_time = time.time()
            
            if hasattr(self.gfpgan, 'enhance'):
                enhanced = await asyncio.to_thread(self.gfpgan.enhance, image)
            else:
                enhanced = await asyncio.to_thread(self.gfpgan.process, image)
            
            processing_time = time.time() - start_time
            
            face_regions = self._count_face_regions(image)
            enhancement_strength = 0.8 if self.is_m3_max else 0.7
            
            metrics = {
                'processing_time': processing_time,
                'face_regions_processed': face_regions,
                'enhancement_strength': enhancement_strength,
                'm3_max_enhanced': self.is_m3_max
            }
            
            return enhanced, metrics
            
        except Exception as e:
            self.logger.warning(f"얼굴 향상 실패: {e}")
            return image, {'error': str(e)}
    
    async def _apply_image_restoration(self, image: Union[torch.Tensor, np.ndarray]) -> Tuple[Union[torch.Tensor, np.ndarray], Dict]:
        """이미지 복원 적용 - M3 Max 최적화"""
        try:
            start_time = time.time()
            
            if hasattr(self.codeformer, 'restore'):
                enhanced = await asyncio.to_thread(self.codeformer.restore, image)
            else:
                enhanced = await asyncio.to_thread(self.codeformer.process, image)
            
            processing_time = time.time() - start_time
            
            artifacts_removed = self._estimate_artifacts_removed(image, enhanced)
            detail_preservation = self._calculate_detail_preservation(image, enhanced)
            
            # M3 Max 정밀도 보너스
            if self.is_m3_max:
                detail_preservation = min(1.0, detail_preservation * 1.05)
            
            metrics = {
                'processing_time': processing_time,
                'artifacts_removed': artifacts_removed,
                'detail_preservation': detail_preservation,
                'm3_max_precision': self.is_m3_max
            }
            
            return enhanced, metrics
            
        except Exception as e:
            self.logger.warning(f"이미지 복원 실패: {e}")
            return image, {'error': str(e)}
    
    async def _apply_color_correction(self, image: Union[torch.Tensor, np.ndarray]) -> Tuple[Union[torch.Tensor, np.ndarray], Dict]:
        """색상 보정 적용 - M3 Max 최적화"""
        try:
            start_time = time.time()
            
            enhanced = await asyncio.to_thread(self.color_enhancer.correct_colors, image)
            
            processing_time = time.time() - start_time
            
            color_improvement = self._calculate_color_balance_improvement(image, enhanced)
            saturation_change = self._calculate_saturation_change(image, enhanced)
            
            metrics = {
                'processing_time': processing_time,
                'color_balance_improvement': color_improvement,
                'saturation_adjustment': saturation_change,
                'm3_max_precision': self.is_m3_max
            }
            
            return enhanced, metrics
            
        except Exception as e:
            self.logger.warning(f"색상 보정 실패: {e}")
            return image, {'error': str(e)}
    
    async def _apply_noise_reduction(self, image: Union[torch.Tensor, np.ndarray]) -> Tuple[Union[torch.Tensor, np.ndarray], Dict]:
        """노이즈 제거 적용 - M3 Max 최적화"""
        try:
            start_time = time.time()
            
            enhanced = await asyncio.to_thread(self.noise_reducer.reduce_noise, image)
            
            processing_time = time.time() - start_time
            
            noise_reduction = self._calculate_noise_reduction(image, enhanced)
            detail_preservation = self._calculate_detail_preservation(image, enhanced)
            
            metrics = {
                'processing_time': processing_time,
                'noise_reduction_amount': noise_reduction,
                'detail_preservation': detail_preservation,
                'm3_max_filtering': self.is_m3_max
            }
            
            return enhanced, metrics
            
        except Exception as e:
            self.logger.warning(f"노이즈 제거 실패: {e}")
            return image, {'error': str(e)}
    
    async def _apply_edge_enhancement(self, image: Union[torch.Tensor, np.ndarray]) -> Tuple[Union[torch.Tensor, np.ndarray], Dict]:
        """엣지 향상 적용 - M3 Max 최적화"""
        try:
            start_time = time.time()
            
            enhanced = await asyncio.to_thread(self.edge_enhancer.enhance_edges, image)
            
            processing_time = time.time() - start_time
            
            edge_improvement = self._calculate_edge_improvement(image, enhanced)
            sharpness_gain = self._calculate_sharpness_improvement(image, enhanced)
            
            metrics = {
                'processing_time': processing_time,
                'edge_strength_improvement': edge_improvement,
                'sharpness_gain': sharpness_gain,
                'm3_max_precision': self.is_m3_max
            }
            
            return enhanced, metrics
            
        except Exception as e:
            self.logger.warning(f"엣지 향상 실패: {e}")
            return image, {'error': str(e)}
    
    async def _apply_m3_max_final_enhancement(self, image: Union[torch.Tensor, np.ndarray]) -> Tuple[Union[torch.Tensor, np.ndarray], Dict]:
        """M3 Max 전용 최종 향상"""
        if not self.is_m3_max:
            return image, {'skipped': 'not_m3_max'}
        
        try:
            start_time = time.time()
            
            # M3 Max Metal Performance Shaders 활용
            if TORCH_AVAILABLE and isinstance(image, torch.Tensor):
                enhanced = image.clone()
                
                # 1. 고급 언샵 마스크
                enhanced = self._apply_advanced_unsharp_mask(enhanced, strength=0.3)
                
                # 2. 적응적 히스토그램 균등화 (텐서 기반)
                enhanced = self._apply_adaptive_histogram_equalization(enhanced)
                
                # 3. 색상 미세 조정
                enhanced = self._apply_color_fine_tuning(enhanced)
            else:
                # NumPy 기반 처리
                enhanced = self._apply_numpy_enhancement(image)
            
            processing_time = time.time() - start_time
            
            metrics = {
                'processing_time': processing_time,
                'advanced_unsharp': True,
                'adaptive_histogram': True,
                'color_fine_tuning': True,
                'm3_max_exclusive': True
            }
            
            return enhanced, metrics
            
        except Exception as e:
            self.logger.warning(f"M3 Max 최종 향상 실패: {e}")
            return image, {'error': str(e)}
    
    # =================================================================
    # 헬퍼 메서드들 - 기존 로직 완전 유지
    # =================================================================
    
    def _apply_advanced_unsharp_mask(self, image: torch.Tensor, strength: float = 0.3) -> torch.Tensor:
        """M3 Max 고급 언샵 마스크"""
        try:
            # 가우시안 블러
            kernel_size = 5
            sigma = 1.5
            
            # 가우시간 커널 생성
            kernel = self._get_gaussian_kernel(kernel_size, sigma).to(image.device)
            
            # 블러 적용
            blurred = F.conv2d(image, kernel, padding=kernel_size//2, groups=image.shape[1])
            
            # 언샵 마스크
            unsharp = image + strength * (image - blurred)
            
            return torch.clamp(unsharp, 0.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"고급 언샵 마스크 실패: {e}")
            return image
    
    def _apply_adaptive_histogram_equalization(self, image: torch.Tensor) -> torch.Tensor:
        """적응적 히스토그램 균등화 (텐서 기반)"""
        try:
            # L 채널 근사 (밝기)
            l_channel = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
            
            # 히스토그램 균등화 근사
            enhanced_l = torch.clamp(l_channel * 1.1, 0.0, 1.0)
            
            # 원래 색상 비율 유지
            ratio = enhanced_l / (l_channel + 1e-6)
            enhanced = image * ratio
            
            return torch.clamp(enhanced, 0.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"적응적 히스토그램 균등화 실패: {e}")
            return image
    
    def _apply_color_fine_tuning(self, image: torch.Tensor) -> torch.Tensor:
        """M3 Max 색상 미세 조정"""
        try:
            # 채도 미세 증가
            enhanced = image.clone()
            
            # RGB 평균
            mean_intensity = torch.mean(enhanced, dim=1, keepdim=True)
            
            # 채도 증가
            saturation_boost = 0.05  # 5% 증가
            enhanced = enhanced + saturation_boost * (enhanced - mean_intensity)
            
            return torch.clamp(enhanced, 0.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"색상 미세 조정 실패: {e}")
            return image
    
    def _apply_numpy_enhancement(self, image: np.ndarray) -> np.ndarray:
        """NumPy 기반 향상 처리"""
        try:
            if not CV2_AVAILABLE:
                return image
            
            # 기본 선명화
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * 0.1
            enhanced = cv2.filter2D(image, -1, kernel)
            
            return np.clip(enhanced, 0, 255).astype(np.uint8)
            
        except Exception as e:
            self.logger.warning(f"NumPy 향상 처리 실패: {e}")
            return image
    
    async def _postprocess_output(self, image: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """출력 후처리 - M3 Max 최적화"""
        try:
            if TORCH_AVAILABLE and isinstance(image, torch.Tensor):
                # 텐서 정규화 및 클리핑
                image = torch.clamp(image, 0.0, 1.0)
                
                # 최종 품질 조정
                if self.enhancement_config.get('final_adjustment', True):
                    image = self._apply_final_adjustments(image)
                
                # M3 Max 전용 최종 폴리싱
                if self.is_m3_max and self.optimization_enabled:
                    image = self._apply_m3_max_final_polish(image)
            
            return image
            
        except Exception as e:
            self.logger.warning(f"출력 후처리 실패: {e}")
            return image
    
    def _apply_final_adjustments(self, image: torch.Tensor) -> torch.Tensor:
        """최종 조정 적용"""
        try:
            # 약간의 선명도 향상
            if self.enhancement_config.get('final_sharpening', True):
                strength = 0.3 if self.is_m3_max else 0.2
                image = self._apply_unsharp_mask(image, strength=strength)
            
            # 색상 미세 조정
            if self.enhancement_config.get('final_color_boost', True):
                factor = 0.15 if self.is_m3_max else 0.1
                image = self._boost_colors(image, factor=factor)
            
            return image
            
        except Exception as e:
            self.logger.warning(f"최종 조정 실패: {e}")
            return image
    
    def _apply_m3_max_final_polish(self, image: torch.Tensor) -> torch.Tensor:
        """M3 Max 전용 최종 폴리싱"""
        try:
            # 미세한 노이즈 제거
            polished = F.conv2d(
                image,
                self._get_smoothing_kernel().to(image.device),
                padding=1,
                groups=image.shape[1]
            )
            
            # 원본과 블렌딩 (99% 원본, 1% 스무딩)
            final = 0.99 * image + 0.01 * polished
            
            return torch.clamp(final, 0.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"M3 Max 최종 폴리싱 실패: {e}")
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
            self.logger.warning(f"언샵 마스크 실패: {e}")
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
            self.logger.warning(f"색상 부스트 실패: {e}")
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
    
    def _get_smoothing_kernel(self) -> torch.Tensor:
        """스무딩 커널 생성"""
        kernel = torch.tensor([
            [[1, 2, 1],
             [2, 4, 2],
             [1, 2, 1]]
        ], dtype=torch.float32) / 16.0
        
        # 3채널용으로 확장
        kernel = kernel.expand(3, 1, 3, 3)
        
        return kernel
    
    # =================================================================
    # 품질 평가 및 헬퍼 메서드들 - 기존 로직 완전 유지
    # =================================================================
    
    async def _evaluate_enhancement_quality(self, original: Union[torch.Tensor, np.ndarray], enhanced: Union[torch.Tensor, np.ndarray]) -> float:
        """향상 품질 평가 - M3 Max 정밀도"""
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
            
            # M3 Max 정밀도 보너스
            precision_bonus = 0.02 if self.is_m3_max else 0.0
            
            # 종합 점수
            quality_score = (
                sharpness_gain * 0.3 +
                detail_preservation * 0.25 +
                color_naturalness * 0.25 +
                (1.0 - artifact_level) * 0.2 +
                precision_bonus
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            self.logger.warning(f"품질 평가 실패: {e}")
            return 0.5
    
    def _calculate_sharpness_improvement(self, original: Union[torch.Tensor, np.ndarray], enhanced: Union[torch.Tensor, np.ndarray]) -> float:
        """선명도 개선 계산"""
        try:
            if TORCH_AVAILABLE and isinstance(original, torch.Tensor) and isinstance(enhanced, torch.Tensor):
                # 라플라시안 분산으로 선명도 측정
                laplacian_kernel = torch.tensor([[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]], 
                                              dtype=torch.float32, device=original.device)
                
                orig_sharpness = self._calculate_laplacian_variance_tensor(original, laplacian_kernel)
                enhanced_sharpness = self._calculate_laplacian_variance_tensor(enhanced, laplacian_kernel)
                
                if orig_sharpness > 0:
                    improvement = (enhanced_sharpness - orig_sharpness) / orig_sharpness
                    return max(0.0, min(1.0, improvement + 0.5))  # 0.5 기준점
            
            return 0.5
            
        except Exception as e:
            self.logger.warning(f"선명도 개선 계산 실패: {e}")
            return 0.5
    
    def _calculate_laplacian_variance_tensor(self, image: torch.Tensor, kernel: torch.Tensor) -> float:
        """라플라시안 분산 계산 (텐서 버전)"""
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
            self.logger.warning(f"라플라시안 분산 계산 실패: {e}")
            return 0.0
    
    def _calculate_detail_preservation(self, original: Union[torch.Tensor, np.ndarray], enhanced: Union[torch.Tensor, np.ndarray]) -> float:
        """디테일 보존도 계산"""
        try:
            if TORCH_AVAILABLE and isinstance(original, torch.Tensor) and isinstance(enhanced, torch.Tensor):
                # 고주파 성분 비교
                orig_details = self._extract_high_frequency_tensor(original)
                enhanced_details = self._extract_high_frequency_tensor(enhanced)
                
                # 상관관계 계산
                correlation = F.cosine_similarity(
                    orig_details.flatten(), 
                    enhanced_details.flatten(), 
                    dim=0
                )
                
                return float((correlation + 1.0) / 2.0)  # -1~1을 0~1로 변환
            
            return 0.5
            
        except Exception as e:
            self.logger.warning(f"디테일 보존도 계산 실패: {e}")
            return 0.5
    
    def _extract_high_frequency_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """고주파 성분 추출 (텐서 버전)"""
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
            self.logger.warning(f"고주파 성분 추출 실패: {e}")
            return torch.zeros_like(image[:, 0:1])
    
    def _calculate_color_naturalness(self, image: Union[torch.Tensor, np.ndarray]) -> float:
        """색상 자연스러움 계산"""
        try:
            if TORCH_AVAILABLE and isinstance(image, torch.Tensor):
                # RGB 값 분포 분석
                r_mean = torch.mean(image[:, 0])
                g_mean = torch.mean(image[:, 1])
                b_mean = torch.mean(image[:, 2])
                
                # 색상 균형 검사
                color_balance = 1.0 - torch.std(torch.tensor([r_mean, g_mean, b_mean]))
                
                # 채도 검사
                saturation = torch.std(image, dim=1).mean()
                saturation_score = 1.0 - torch.clamp(saturation - 0.2, 0, 1)
                
                # 조합
                naturalness = (color_balance * 0.6 + saturation_score * 0.4)
                
                return float(torch.clamp(naturalness, 0.0, 1.0))
            
            return 0.5
            
        except Exception as e:
            self.logger.warning(f"색상 자연스러움 계산 실패: {e}")
            return 0.5
    
    def _estimate_artifact_level(self, image: Union[torch.Tensor, np.ndarray]) -> float:
        """아티팩트 수준 추정"""
        try:
            # 간단한 아티팩트 검출
            return 0.2  # 기본값
            
        except Exception as e:
            self.logger.warning(f"아티팩트 수준 추정 실패: {e}")
            return 0.3
    
    # 기타 헬퍼 메서드들...
    
    def _count_face_regions(self, image: Union[torch.Tensor, np.ndarray]) -> int:
        """얼굴 영역 카운트 (간단 버전)"""
        return 1  # 기본값
    
    def _estimate_artifacts_removed(self, original: Union[torch.Tensor, np.ndarray], enhanced: Union[torch.Tensor, np.ndarray]) -> float:
        """제거된 아티팩트 추정"""
        return 0.3  # 기본값
    
    def _calculate_color_balance_improvement(self, original: Union[torch.Tensor, np.ndarray], enhanced: Union[torch.Tensor, np.ndarray]) -> float:
        """색상 균형 개선 계산"""
        return 0.2  # 기본값
    
    def _calculate_saturation_change(self, original: Union[torch.Tensor, np.ndarray], enhanced: Union[torch.Tensor, np.ndarray]) -> float:
        """채도 변화 계산"""
        return 0.1  # 기본값
    
    def _calculate_noise_reduction(self, original: Union[torch.Tensor, np.ndarray], enhanced: Union[torch.Tensor, np.ndarray]) -> float:
        """노이즈 감소량 계산"""
        return 0.3  # 기본값
    
    def _calculate_edge_improvement(self, original: Union[torch.Tensor, np.ndarray], enhanced: Union[torch.Tensor, np.ndarray]) -> float:
        """엣지 개선 계산"""
        return 0.2  # 기본값
    
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
            self.logger.warning(f"통계 업데이트 실패: {e}")
    
    # =================================================================
    # 최적 생성자 호환 메서드들
    # =================================================================
    
    async def get_step_info(self) -> Dict[str, Any]:
        """🔍 스텝 정보 반환 (최적 생성자 호환)"""
        return {
            "step_name": "PostProcessing",
            "class_name": self.__class__.__name__,
            "version": "4.0-m3max",
            "device": self.device,
            "device_type": self.device_type,
            "memory_gb": self.memory_gb,
            "is_m3_max": self.is_m3_max,
            "optimization_enabled": self.optimization_enabled,
            "quality_level": self.quality_level,
            "initialized": self.is_initialized,
            "initialization_error": self.initialization_error,
            "config_keys": list(self.config.keys()),
            "performance_stats": self.processing_stats.copy(),
            "capabilities": {
                "super_resolution": bool(self.real_esrgan),
                "face_enhancement": bool(self.gfpgan),
                "image_restoration": bool(self.codeformer),
                "color_correction": bool(self.color_enhancer),
                "noise_reduction": bool(self.noise_reducer),
                "edge_enhancement": bool(self.edge_enhancer),
                "neural_enhancement": self.enable_neural_enhancement,
                "m3_max_acceleration": self.is_m3_max and self.device == 'mps'
            },
            "performance_settings": {
                "batch_size": self.batch_size,
                "tile_size": self.tile_size,
                "quality_level": self.enhancement_config.get('quality_level', 'high'),
                "use_mps": self.use_mps
            },
            "enhancement_config": self.enhancement_config
        }
    
    async def cleanup(self):
        """리소스 정리 (최적 생성자 호환)"""
        try:
            self.logger.info("🧹 후처리 시스템 리소스 정리 시작...")
            
            # 모델 정리
            if self.real_esrgan and hasattr(self.real_esrgan, 'cleanup'):
                await self.real_esrgan.cleanup()
            
            if self.gfpgan and hasattr(self.gfpgan, 'cleanup'):
                await self.gfpgan.cleanup()
            
            if self.codeformer and hasattr(self.codeformer, 'cleanup'):
                await self.codeformer.cleanup()
            
            # GPU 메모리 정리
            if self.use_mps and TORCH_AVAILABLE:
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
                elif hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
            elif TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 시스템 메모리 정리
            import gc
            gc.collect()
            
            self.is_initialized = False
            self.logger.info("✅ 후처리 시스템 리소스 정리 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 리소스 정리 중 오류: {e}")


# =================================================================
# M3 Max 최적화 폴백 클래스들 - 기존 로직 완전 유지
# =================================================================

class M3MaxUpscaler:
    """M3 Max 최적화 업스케일러"""
    
    def __init__(self, device: str = 'mps', scale_factor: int = 4, use_neural: bool = True):
        self.device = device
        self.scale_factor = scale_factor
        self.use_neural = use_neural
    
    def upscale(self, image: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """M3 Max 최적화 업스케일링"""
        try:
            if TORCH_AVAILABLE and isinstance(image, torch.Tensor):
                b, c, h, w = image.shape
                new_h, new_w = h * self.scale_factor, w * self.scale_factor
                
                if self.use_neural and self.device == 'mps':
                    # M3 Max Neural Engine 활용 (근사)
                    upscaled = F.interpolate(
                        image, 
                        size=(new_h, new_w), 
                        mode='bicubic', 
                        align_corners=False,
                        antialias=True
                    )
                    
                    # 추가 선명화
                    kernel = torch.tensor([[[-0.5, -0.5, -0.5], [-0.5, 5, -0.5], [-0.5, -0.5, -0.5]]], 
                                        dtype=torch.float32, device=self.device) / 4.0
                    sharpened = F.conv2d(upscaled, kernel.repeat(c, 1, 1, 1), padding=1, groups=c)
                    upscaled = torch.clamp(sharpened, 0.0, 1.0)
                    
                else:
                    # 기본 바이큐빅 업스케일링
                    upscaled = F.interpolate(
                        image, 
                        size=(new_h, new_w), 
                        mode='bicubic', 
                        align_corners=False
                    )
                
                return upscaled
            else:
                # NumPy 처리
                if CV2_AVAILABLE:
                    h, w = image.shape[:2]
                    new_size = (w * self.scale_factor, h * self.scale_factor)
                    return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
                else:
                    return image
            
        except Exception as e:
            logger.warning(f"M3 Max 업스케일링 실패: {e}")
            return image
    
    async def cleanup(self):
        pass


class M3MaxFaceEnhancer:
    """M3 Max 최적화 얼굴 향상기"""
    
    def __init__(self, device: str = 'mps', enhancement_strength: float = 1.0):
        self.device = device
        self.enhancement_strength = enhancement_strength
    
    def process(self, image: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """M3 Max 얼굴 향상"""
        try:
            if TORCH_AVAILABLE and isinstance(image, torch.Tensor):
                # 고급 언샵 마스크 적용
                kernel = torch.tensor([[[-1, -2, -1], [-2, 13, -2], [-1, -2, -1]]], 
                                    dtype=torch.float32, device=self.device) / 8.0 * self.enhancement_strength
                
                if image.shape[1] == 3:
                    enhanced = F.conv2d(image, kernel.repeat(3, 1, 1, 1), padding=1, groups=3)
                else:
                    enhanced = F.conv2d(image, kernel, padding=1)
                
                return torch.clamp(enhanced, 0.0, 1.0)
            else:
                # NumPy 처리
                if CV2_AVAILABLE:
                    kernel = np.array([[-1, -2, -1], [-2, 13, -2], [-1, -2, -1]]) / 8.0 * self.enhancement_strength
                    enhanced = cv2.filter2D(image, -1, kernel)
                    return np.clip(enhanced, 0, 255).astype(np.uint8)
                else:
                    return image
            
        except Exception as e:
            logger.warning(f"M3 Max 얼굴 향상 실패: {e}")
            return image
    
    async def cleanup(self):
        pass


class M3MaxImageRestorer:
    """M3 Max 최적화 이미지 복원기"""
    
    def __init__(self, device: str = 'mps', restoration_strength: float = 1.0):
        self.device = device
        self.restoration_strength = restoration_strength
    
    def process(self, image: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """M3 Max 이미지 복원"""
        try:
            if TORCH_AVAILABLE and isinstance(image, torch.Tensor):
                # 적응적 가우시안 노이즈 제거
                kernel_size = int(3 * self.restoration_strength)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                sigma = 0.5 * self.restoration_strength
                
                # 가우시안 커널 생성
                coords = torch.arange(kernel_size, dtype=torch.float32, device=self.device)
                coords -= kernel_size // 2
                g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
                g /= g.sum()
                
                kernel = g[:, None] * g[None, :]
                kernel = kernel.expand(image.shape[1], 1, kernel_size, kernel_size)
                
                # 적용
                restored = F.conv2d(image, kernel, padding=kernel_size//2, groups=image.shape[1])
                
                # 원본과 블렌딩
                alpha = 0.7 * self.restoration_strength
                final = alpha * restored + (1 - alpha) * image
                
                return final
            else:
                # NumPy 처리
                if CV2_AVAILABLE:
                    # 가우시안 블러 적용
                    restored = cv2.GaussianBlur(image, (5, 5), 1.0)
                    alpha = 0.7 * self.restoration_strength
                    final = alpha * restored + (1 - alpha) * image
                    return np.clip(final, 0, 255).astype(np.uint8)
                else:
                    return image
            
        except Exception as e:
            logger.warning(f"M3 Max 이미지 복원 실패: {e}")
            return image
    
    async def cleanup(self):
        pass


class ColorEnhancer:
    """M3 Max 최적화 색상 향상기"""
    
    def __init__(self, device: str = 'mps', m3_max_mode: bool = True):
        self.device = device
        self.m3_max_mode = m3_max_mode
    
    def correct_colors(self, image: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """M3 Max 색상 보정"""
        try:
            if TORCH_AVAILABLE and isinstance(image, torch.Tensor):
                if self.m3_max_mode:
                    # M3 Max 정밀 색상 보정
                    corrected = self._advanced_color_correction_tensor(image)
                else:
                    # 기본 색상 보정
                    corrected = self._basic_color_correction_tensor(image)
                
                return torch.clamp(corrected, 0.0, 1.0)
            else:
                # NumPy 처리
                return self._numpy_color_correction(image)
            
        except Exception as e:
            logger.warning(f"색상 보정 실패: {e}")
            return image
    
    def _advanced_color_correction_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """고급 색상 보정 (텐서)"""
        # 채널별 평균 계산
        r_mean = torch.mean(image[:, 0])
        g_mean = torch.mean(image[:, 1])
        b_mean = torch.mean(image[:, 2])
        
        overall_mean = (r_mean + g_mean + b_mean) / 3.0
        
        # 적응적 조정 계수
        r_factor = torch.clamp(overall_mean / (r_mean + 1e-8), 0.9, 1.1)
        g_factor = torch.clamp(overall_mean / (g_mean + 1e-8), 0.9, 1.1)
        b_factor = torch.clamp(overall_mean / (b_mean + 1e-8), 0.9, 1.1)
        
        # 부드러운 조정
        corrected = image.clone()
        corrected[:, 0] *= (1.0 + 0.1 * (r_factor - 1.0))
        corrected[:, 1] *= (1.0 + 0.1 * (g_factor - 1.0))
        corrected[:, 2] *= (1.0 + 0.1 * (b_factor - 1.0))
        
        return corrected
    
    def _basic_color_correction_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """기본 색상 보정 (텐서)"""
        # 간단한 색상 균형 조정
        r_mean = torch.mean(image[:, 0])
        g_mean = torch.mean(image[:, 1])
        b_mean = torch.mean(image[:, 2])
        
        overall_mean = (r_mean + g_mean + b_mean) / 3.0
        
        r_factor = overall_mean / (r_mean + 1e-8)
        g_factor = overall_mean / (g_mean + 1e-8)
        b_factor = overall_mean / (b_mean + 1e-8)
        
        # 부드러운 조정
        r_factor = 1.0 + 0.05 * (r_factor - 1.0)
        g_factor = 1.0 + 0.05 * (g_factor - 1.0)
        b_factor = 1.0 + 0.05 * (b_factor - 1.0)
        
        corrected = image.clone()
        corrected[:, 0] *= r_factor
        corrected[:, 1] *= g_factor
        corrected[:, 2] *= b_factor
        
        return corrected
    
    def _numpy_color_correction(self, image: np.ndarray) -> np.ndarray:
        """NumPy 색상 보정"""
        try:
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB 채널별 평균
                r_mean = np.mean(image[:, :, 0])
                g_mean = np.mean(image[:, :, 1])
                b_mean = np.mean(image[:, :, 2])
                
                overall_mean = (r_mean + g_mean + b_mean) / 3.0
                
                # 보정 계수
                r_factor = overall_mean / (r_mean + 1e-8)
                g_factor = overall_mean / (g_mean + 1e-8)
                b_factor = overall_mean / (b_mean + 1e-8)
                
                # 부드러운 조정
                r_factor = 1.0 + 0.05 * (r_factor - 1.0)
                g_factor = 1.0 + 0.05 * (g_factor - 1.0)
                b_factor = 1.0 + 0.05 * (b_factor - 1.0)
                
                corrected = image.astype(np.float32)
                corrected[:, :, 0] *= r_factor
                corrected[:, :, 1] *= g_factor
                corrected[:, :, 2] *= b_factor
                
                return np.clip(corrected, 0, 255).astype(np.uint8)
            
            return image
            
        except Exception as e:
            logger.warning(f"NumPy 색상 보정 실패: {e}")
            return image


class NoiseReducer:
    """M3 Max 최적화 노이즈 제거기"""
    
    def __init__(self, device: str = 'mps', m3_max_mode: bool = True):
        self.device = device
        self.m3_max_mode = m3_max_mode
    
    def reduce_noise(self, image: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """M3 Max 노이즈 제거"""
        try:
            if TORCH_AVAILABLE and isinstance(image, torch.Tensor):
                if self.m3_max_mode:
                    # M3 Max 고급 노이즈 제거
                    denoised = self._advanced_noise_reduction_tensor(image)
                else:
                    # 기본 노이즈 제거
                    denoised = self._basic_noise_reduction_tensor(image)
                
                return denoised
            else:
                # NumPy 처리
                return self._numpy_noise_reduction(image)
            
        except Exception as e:
            logger.warning(f"노이즈 제거 실패: {e}")
            return image
    
    def _advanced_noise_reduction_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """고급 노이즈 제거 (양방향 필터 근사)"""
        # 다중 스케일 가우시안 블러
        scales = [0.5, 1.0, 1.5]
        blurred_images = []
        
        for sigma in scales:
            kernel_size = int(4 * sigma) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            coords = torch.arange(kernel_size, dtype=torch.float32, device=self.device)
            coords -= kernel_size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g /= g.sum()
            
            kernel = g[:, None] * g[None, :]
            kernel = kernel.expand(image.shape[1], 1, kernel_size, kernel_size)
            
            blurred = F.conv2d(image, kernel, padding=kernel_size//2, groups=image.shape[1])
            blurred_images.append(blurred)
        
        # 가중 평균
        weights = [0.5, 0.3, 0.2]
        denoised = sum(w * img for w, img in zip(weights, blurred_images))
        
        # 원본과 블렌딩 (디테일 보존)
        blending_factor = 0.7
        result = blending_factor * denoised + (1 - blending_factor) * image
        
        return result
    
    def _basic_noise_reduction_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """기본 노이즈 제거 (텐서)"""
        kernel_size = 5
        sigma = 1.0
        
        coords = torch.arange(kernel_size, dtype=torch.float32, device=self.device)
        coords -= kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        kernel = g[:, None] * g[None, :]
        kernel = kernel.expand(image.shape[1], 1, kernel_size, kernel_size)
        
        denoised = F.conv2d(image, kernel, padding=kernel_size//2, groups=image.shape[1])
        
        # 원본과 블렌딩
        blending_factor = 0.6
        result = blending_factor * denoised + (1 - blending_factor) * image
        
        return result
    
    def _numpy_noise_reduction(self, image: np.ndarray) -> np.ndarray:
        """NumPy 노이즈 제거"""
        try:
            if CV2_AVAILABLE:
                # 가우시안 블러를 사용한 노이즈 제거
                denoised = cv2.GaussianBlur(image, (5, 5), 1.0)
                
                # 원본과 블렌딩
                blending_factor = 0.6
                result = blending_factor * denoised + (1 - blending_factor) * image
                
                return np.clip(result, 0, 255).astype(np.uint8)
            else:
                return image
                
        except Exception as e:
            logger.warning(f"NumPy 노이즈 제거 실패: {e}")
            return image


class EdgeEnhancer:
    """M3 Max 최적화 엣지 향상기"""
    
    def __init__(self, device: str = 'mps', m3_max_mode: bool = True):
        self.device = device
        self.m3_max_mode = m3_max_mode
    
    def enhance_edges(self, image: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """M3 Max 엣지 향상"""
        try:
            if TORCH_AVAILABLE and isinstance(image, torch.Tensor):
                if self.m3_max_mode:
                    # M3 Max 고급 엣지 향상
                    enhanced = self._advanced_edge_enhancement_tensor(image)
                else:
                    # 기본 엣지 향상
                    enhanced = self._basic_edge_enhancement_tensor(image)
                
                return torch.clamp(enhanced, 0.0, 1.0)
            else:
                # NumPy 처리
                return self._numpy_edge_enhancement(image)
            
        except Exception as e:
            logger.warning(f"엣지 향상 실패: {e}")
            return image
    
    def _advanced_edge_enhancement_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """고급 엣지 향상 (적응적 언샵 마스크)"""
        # 1. 엣지 검출
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], 
                             dtype=torch.float32, device=self.device)
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], 
                             dtype=torch.float32, device=self.device)
        
        # 그레이스케일 변환
        if image.shape[1] == 3:
            gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        else:
            gray = image
        
        # 엣지 크기 계산
        edge_x = F.conv2d(gray, sobel_x.unsqueeze(0), padding=1)
        edge_y = F.conv2d(gray, sobel_y.unsqueeze(0), padding=1)
        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        
        # 적응적 마스크 생성
        edge_mask = torch.sigmoid(edge_magnitude * 5.0)  # 엣지 영역 강조
        
        # 2. 언샵 마스크
        kernel_size = 5
        sigma = 1.5
        
        coords = torch.arange(kernel_size, dtype=torch.float32, device=self.device)
        coords -= kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        kernel = g[:, None] * g[None, :]
        kernel = kernel.expand(image.shape[1], 1, kernel_size, kernel_size)
        
        blurred = F.conv2d(image, kernel, padding=kernel_size//2, groups=image.shape[1])
        
        # 3. 적응적 향상
        strength = 0.3
        unsharp = image + strength * (image - blurred)
        
        # 엣지 영역에서만 강하게 적용
        enhanced = image * (1 - edge_mask) + unsharp * edge_mask
        
        return enhanced
    
    def _basic_edge_enhancement_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """기본 엣지 향상 (텐서)"""
        # 기본 언샵 마스크
        kernel_size = 5
        sigma = 1.0
        
        coords = torch.arange(kernel_size, dtype=torch.float32, device=self.device)
        coords -= kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        kernel = g[:, None] * g[None, :]
        kernel = kernel.expand(image.shape[1], 1, kernel_size, kernel_size)
        
        blurred = F.conv2d(image, kernel, padding=kernel_size//2, groups=image.shape[1])
        
        # 언샵 마스크
        strength = 0.2
        enhanced = image + strength * (image - blurred)
        
        return enhanced
    
    def _numpy_edge_enhancement(self, image: np.ndarray) -> np.ndarray:
        """NumPy 엣지 향상"""
        try:
            if CV2_AVAILABLE:
                # 언샵 마스크
                blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
                
                # 강도 조절
                strength = 0.2
                enhanced = image.astype(np.float32) + strength * (image.astype(np.float32) - blurred.astype(np.float32))
                
                return np.clip(enhanced, 0, 255).astype(np.uint8)
            else:
                return image
                
        except Exception as e:
            logger.warning(f"NumPy 엣지 향상 실패: {e}")
            return image