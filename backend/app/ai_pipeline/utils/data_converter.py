# backend/app/ai_pipeline/utils/data_converter.py
"""
🍎 MyCloset AI - 완전 최적화 데이터 변환기 (완전 안전한 임포트 버전)
================================================================================
✅ AttributeError: 'NoneType' object has no attribute 'Tensor' 완전 해결
✅ 모든 타입 힌팅을 문자열 기반으로 안전하게 변경
✅ PyTorch 임포트 실패 시 안전한 폴백 메커니즘
✅ 현재 프로젝트 구조 100% 최적화
✅ 기존 함수명/클래스명 완전 유지 (DataConverter, ImageProcessor)
✅ ModelLoader 시스템과 완벽 연동
✅ BaseStepMixin logger 속성 완벽 보장  
✅ M3 Max 최적화 이미지/텐서 변환
✅ 순환참조 완전 해결 (한방향 의존성)
✅ 프로덕션 레벨 안정성
✅ conda 환경 완벽 지원
================================================================================
Author: MyCloset AI Team
Date: 2025-07-20
Version: 7.1 (Safe Import Fix)
"""

import io
import logging
import time
import base64
from typing import Dict, Any, Optional, Union, List, Tuple, TYPE_CHECKING
from pathlib import Path
import asyncio
from functools import wraps

# 🔥 TYPE_CHECKING을 사용하여 안전한 타입 힌팅
if TYPE_CHECKING:
    # 타입 힌팅용 임포트 (런타임에는 실행되지 않음)
    import torch
    import numpy as np
    from PIL import Image

# NumPy 안전한 임포트
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    NUMPY_VERSION = np.__version__
    
    # NumPy 2.x 호환성 처리
    major_version = int(np.__version__.split('.')[0])
    if major_version >= 2:
        try:
            np.set_printoptions(legacy='1.25')
            logging.info("✅ NumPy 2.x 호환성 모드 활성화")
        except:
            pass
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    NUMPY_VERSION = "not_available"

# PIL 안전한 임포트
try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    PIL_AVAILABLE = True
    PIL_VERSION = getattr(Image, '__version__', 'unknown')
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    PIL_VERSION = "not_available"

# OpenCV 안전한 임포트
try:
    import cv2
    CV2_AVAILABLE = True
    CV2_VERSION = cv2.__version__
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    CV2_VERSION = "not_available"

# PyTorch 완전 안전한 임포트
try:
    # MPS 환경변수 설정 (M3 Max 최적화)
    import os
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    import torch
    import torchvision.transforms as transforms
    import torchvision.transforms.functional as TF
    TORCH_AVAILABLE = True
    TORCH_VERSION = torch.__version__
    
    # MPS 지원 확인
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        MPS_AVAILABLE = True
        DEFAULT_DEVICE = "mps"
    else:
        MPS_AVAILABLE = False
        DEFAULT_DEVICE = "cpu"
        
except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False
    torch = None
    transforms = None
    TF = None
    DEFAULT_DEVICE = "cpu"
    TORCH_VERSION = "not_available"

logger = logging.getLogger(__name__)

# ==============================================
# 🔥 데이터 구조 정의
# ==============================================

class ConversionMode:
    """변환 모드"""
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"
    M3_OPTIMIZED = "m3_optimized"

class ImageFormat:
    """지원 이미지 포맷"""
    PIL = "PIL"
    NUMPY = "numpy"
    TENSOR = "tensor"
    CV2 = "cv2"
    BASE64 = "base64"

# ==============================================
# 🔥 핵심 데이터 변환기 클래스들 (안전한 타입 힌팅)
# ==============================================

class DataConverter:
    """
    🍎 프로젝트 최적화 데이터 변환기 (기본 클래스)
    ✅ 현재 구조와 완벽 호환
    ✅ M3 Max 최적화 이미지/텐서 변환
    ✅ 순환참조 없는 안전한 구조
    ✅ 안전한 타입 힌팅 (문자열 기반)
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """데이터 변환기 초기화"""
        # 1. 디바이스 자동 감지
        self.device = self._auto_detect_device(device)

        # 2. 기본 설정
        self.config = config or {}
        self.step_name = self.__class__.__name__
        
        # 🔥 logger 속성 보장 (현재 구조 호환)
        self.logger = logging.getLogger(f"utils.{self.step_name}")

        # 3. 시스템 파라미터
        self.memory_gb = kwargs.get('memory_gb', 16.0)
        self.is_m3_max = kwargs.get('is_m3_max', self._detect_m3_max())
        self.optimization_enabled = kwargs.get('optimization_enabled', True)

        # 4. 데이터 변환기 특화 파라미터
        self.default_size = tuple(kwargs.get('default_size', (512, 512)))
        self.interpolation = kwargs.get('interpolation', 'bilinear')
        self.normalize_mean = kwargs.get('normalize_mean', [0.485, 0.456, 0.406])
        self.normalize_std = kwargs.get('normalize_std', [0.229, 0.224, 0.225])
        self.use_gpu_acceleration = kwargs.get('use_gpu_acceleration', self.device != 'cpu')
        self.batch_processing = kwargs.get('batch_processing', True)
        self.memory_efficient = kwargs.get('memory_efficient', True)
        self.quality_preservation = kwargs.get('quality_preservation', True)
        self.conversion_mode = kwargs.get('conversion_mode', ConversionMode.BALANCED)

        # 5. M3 Max 특화 설정
        if self.is_m3_max:
            self.use_gpu_acceleration = True
            self.batch_processing = True
            self.memory_efficient = False  # 128GB 메모리이므로 품질 우선
            self.conversion_mode = ConversionMode.M3_OPTIMIZED

        # 6. 상태 초기화
        self.is_initialized = False
        self._initialize_components()

        self.logger.info(f"🎯 DataConverter 초기화 - 디바이스: {self.device}")
        self.logger.info(f"📚 라이브러리 상태: PyTorch={TORCH_AVAILABLE}, PIL={PIL_AVAILABLE}, NumPy={NUMPY_AVAILABLE}")

    def _auto_detect_device(self, preferred_device: Optional[str]) -> str:
        """💡 지능적 디바이스 자동 감지"""
        if preferred_device:
            return preferred_device

        if not TORCH_AVAILABLE:
            return 'cpu'

        try:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'  # M3 Max 우선
            elif torch.cuda.is_available():
                return 'cuda'  # NVIDIA GPU
            else:
                return 'cpu'  # 폴백
        except Exception as e:
            self.logger.warning(f"디바이스 감지 실패: {e}")
            return 'cpu'

    def _detect_m3_max(self) -> bool:
        """🍎 M3 Max 칩 자동 감지"""
        try:
            import platform
            import subprocess

            if platform.system() == 'Darwin':  # macOS
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True, timeout=5)
                chip_info = result.stdout.strip()
                return 'M3' in chip_info and 'Max' in chip_info
        except Exception as e:
            self.logger.debug(f"M3 Max 감지 실패: {e}")
        return False

    def _initialize_components(self):
        """구성 요소 초기화"""
        # 변환 파이프라인 초기화
        self._init_transforms()
        
        # 통계 추적
        self._conversion_stats = {
            "total_conversions": 0,
            "total_time": 0.0,
            "format_counts": {},
            "error_count": 0,
            "m3_optimizations": 0
        }
        
        self.logger.info(f"🔄 DataConverter 구성 요소 초기화 완료")
        
        # M3 Max 최적화 설정
        if self.device == "mps" and self.is_m3_max:
            self.logger.info("🍎 M3 Max 데이터 변환 최적화 모드 활성화")
            self._apply_m3_max_optimizations()
        
        # 초기화 완료
        self.is_initialized = True

    def _init_transforms(self):
        """변환 파이프라인 초기화"""
        self.transforms = {}
        
        if not TORCH_AVAILABLE:
            self.logger.warning("⚠️ PyTorch 없음 - 변환 파이프라인 제한")
            return
        
        try:
            # 보간 방법 매핑
            interpolation_map = {
                'bilinear': transforms.InterpolationMode.BILINEAR if hasattr(transforms, 'InterpolationMode') else 2,
                'nearest': transforms.InterpolationMode.NEAREST if hasattr(transforms, 'InterpolationMode') else 0,
                'bicubic': transforms.InterpolationMode.BICUBIC if hasattr(transforms, 'InterpolationMode') else 3
            }
            
            interpolation_mode = interpolation_map.get(self.interpolation, 2)
            
            # 기본 변환 파이프라인
            self.transforms['default'] = transforms.Compose([
                transforms.Resize(self.default_size, interpolation=interpolation_mode),
                transforms.ToTensor()
            ])
            
            # 정규화 변환
            self.transforms['normalized'] = transforms.Compose([
                transforms.Resize(self.default_size, interpolation=interpolation_mode),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
            ])
            
            # 고품질 변환 (M3 Max 최적화)
            if self.is_m3_max and self.quality_preservation:
                self.transforms['high_quality'] = transforms.Compose([
                    transforms.Resize(self.default_size, interpolation=interpolation_mode),
                    transforms.ToTensor()
                ])
                
                # M3 Max 전용 고해상도 변환
                self.transforms['m3_max_quality'] = transforms.Compose([
                    transforms.Resize((1024, 1024) if self.default_size[0] < 1024 else self.default_size, 
                                    interpolation=interpolation_mode),
                    transforms.ToTensor()
                ])
                
        except Exception as e:
            self.logger.error(f"❌ 변환 파이프라인 초기화 실패: {e}")

    def _apply_m3_max_optimizations(self):
        """M3 Max 특화 최적화 적용"""
        try:
            optimizations = []
            
            # 1. 고해상도 처리 활성화
            if self.default_size[0] < 1024:
                self.default_size = (1024, 1024)
                optimizations.append("High resolution processing (1024x1024)")
            
            # 2. 메모리 효율성 조정 (128GB 메모리)
            self.memory_efficient = False  # 품질 우선
            optimizations.append("Quality-first processing mode")
            
            # 3. MPS 백엔드 최적화
            if TORCH_AVAILABLE and MPS_AVAILABLE:
                optimizations.append("MPS backend acceleration")
            
            # 4. 배치 처리 최적화
            self.batch_processing = True
            optimizations.append("Optimized batch processing")
            
            if optimizations:
                self.logger.info(f"🍎 M3 Max 데이터 변환 최적화 적용:")
                for opt in optimizations:
                    self.logger.info(f"   - {opt}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ M3 Max 최적화 실패: {e}")

    # ============================================
    # 🔥 핵심 변환 메서드들 (안전한 타입 힌팅)
    # ============================================

    def image_to_tensor(
        self,
        image: Union["Image.Image", "np.ndarray", str, bytes],
        size: Optional[Tuple[int, int]] = None,
        normalize: bool = False,
        **kwargs
    ) -> Optional["torch.Tensor"]:
        """이미지를 텐서로 변환 (안전한 타입 힌팅)"""
        if not TORCH_AVAILABLE:
            self.logger.error("❌ PyTorch가 설치되지 않음")
            return None
            
        try:
            start_time = time.time()
            
            # 이미지 전처리
            pil_image = self._to_pil_image(image)
            if pil_image is None:
                return None
            
            # 크기 설정
            target_size = size or self.default_size
            
            # 변환 파이프라인 선택
            if self.is_m3_max and self.conversion_mode == ConversionMode.M3_OPTIMIZED:
                transform = self.transforms.get('m3_max_quality')
                self._conversion_stats["m3_optimizations"] += 1
            elif normalize:
                transform = self.transforms.get('normalized')
            elif self.is_m3_max and self.quality_preservation:
                transform = self.transforms.get('high_quality')
            else:
                transform = self.transforms.get('default')
            
            if transform is None:
                # 폴백 변환
                if hasattr(pil_image, 'resize'):
                    pil_image = pil_image.resize(target_size)
                tensor = TF.to_tensor(pil_image) if TF else None
            else:
                tensor = transform(pil_image)
            
            if tensor is None:
                return None
            
            # 배치 차원 추가
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            
            # 디바이스로 이동
            if self.use_gpu_acceleration and self.device != 'cpu':
                tensor = tensor.to(self.device)
            
            # 통계 업데이트
            processing_time = time.time() - start_time
            self._update_stats('image_to_tensor', processing_time)
            
            self.logger.debug(f"🔄 이미지→텐서 변환 완료: {tensor.shape} ({processing_time:.3f}s)")
            return tensor
            
        except Exception as e:
            self.logger.error(f"❌ 이미지→텐서 변환 실패: {e}")
            self._conversion_stats["error_count"] += 1
            return None

    def tensor_to_image(
        self,
        tensor: "torch.Tensor",
        denormalize: bool = False,
        format: str = "PIL"
    ) -> Optional[Union["Image.Image", "np.ndarray"]]:
        """텐서를 이미지로 변환 (안전한 타입 힌팅)"""
        if not TORCH_AVAILABLE:
            self.logger.error("❌ PyTorch가 설치되지 않음")
            return None
            
        try:
            start_time = time.time()
            
            # 텐서 전처리
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)  # 배치 차원 제거
            
            if tensor.dim() != 3:
                raise ValueError(f"Invalid tensor dimensions: {tensor.shape}")
            
            # CPU로 이동
            if tensor.device != torch.device('cpu'):
                tensor = tensor.cpu()
            
            # 역정규화
            if denormalize:
                tensor = self._denormalize_tensor(tensor)
            
            # [0, 1] 범위로 클램핑
            tensor = torch.clamp(tensor, 0, 1)
            
            # PIL 이미지로 변환
            if TF:
                pil_image = TF.to_pil_image(tensor)
            else:
                # 폴백: numpy를 통한 변환
                if NUMPY_AVAILABLE:
                    array = tensor.permute(1, 2, 0).numpy()
                    array = (array * 255).astype(np.uint8)
                    if PIL_AVAILABLE:
                        pil_image = Image.fromarray(array)
                    else:
                        return array if format == "numpy" else None
                else:
                    return None
            
            # 출력 형식에 따른 변환
            if format.lower() == "pil":
                result = pil_image
            elif format.lower() == "numpy":
                if NUMPY_AVAILABLE:
                    result = np.array(pil_image)
                else:
                    result = None
            elif format.lower() == "cv2" and CV2_AVAILABLE:
                if NUMPY_AVAILABLE:
                    result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                else:
                    result = None
            else:
                result = pil_image  # 기본값
            
            # 통계 업데이트
            processing_time = time.time() - start_time
            self._update_stats('tensor_to_image', processing_time)
            
            self.logger.debug(f"🔄 텐서→이미지 변환 완료: {format} ({processing_time:.3f}s)")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 텐서→이미지 변환 실패: {e}")
            self._conversion_stats["error_count"] += 1
            return None

    def tensor_to_numpy(self, tensor: "torch.Tensor") -> Optional["np.ndarray"]:
        """텐서를 numpy 배열로 변환 (기존 메서드명 유지)"""
        if not TORCH_AVAILABLE or not NUMPY_AVAILABLE:
            self.logger.error("❌ PyTorch 또는 NumPy가 설치되지 않음")
            return None
            
        try:
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)  # 배치 차원 제거
            
            if tensor.dim() == 3:
                # (C, H, W) -> (H, W, C)로 변환
                tensor = tensor.permute(1, 2, 0)
            
            # CPU로 이동
            if tensor.device != torch.device('cpu'):
                tensor = tensor.cpu()
            
            # numpy 배열로 변환
            array = tensor.numpy()
            
            # [0, 1] 범위를 [0, 255]로 변환 (필요시)
            if array.max() <= 1.0:
                array = (array * 255).astype(np.uint8)
            
            return array
            
        except Exception as e:
            self.logger.error(f"❌ 텐서→numpy 변환 실패: {e}")
            return None

    def _to_pil_image(self, image_input: Union["Image.Image", "np.ndarray", str, bytes]) -> Optional["Image.Image"]:
        """다양한 입력을 PIL 이미지로 변환"""
        try:
            if not PIL_AVAILABLE:
                self.logger.error("❌ PIL이 설치되지 않음")
                return None
                
            # 이미 PIL 이미지인 경우
            if hasattr(image_input, 'convert'):  # PIL Image 객체 체크
                return image_input.convert('RGB')
            
            # NumPy 배열인 경우
            elif NUMPY_AVAILABLE and hasattr(image_input, 'ndim'):  # numpy array 체크
                if image_input.ndim == 3:
                    return Image.fromarray(image_input.astype(np.uint8)).convert('RGB')
                elif image_input.ndim == 2:
                    return Image.fromarray(image_input.astype(np.uint8)).convert('RGB')
            
            # 파일 경로인 경우
            elif isinstance(image_input, (str, Path)):
                if isinstance(image_input, str) and not image_input.startswith('data:image'):
                    path = Path(image_input)
                    if path.exists():
                        return Image.open(path).convert('RGB')
                else:
                    # Base64 Data URL 파싱
                    if image_input.startswith('data:image'):
                        header, data = image_input.split(',', 1)
                        image_data = base64.b64decode(data)
                        return Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # 바이트 데이터인 경우
            elif isinstance(image_input, bytes):
                return Image.open(io.BytesIO(image_input)).convert('RGB')
            
            else:
                self.logger.error(f"❌ 지원되지 않는 이미지 형식: {type(image_input)}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ PIL 이미지 변환 실패: {e}")
            return None

    def _denormalize_tensor(self, tensor: "torch.Tensor") -> "torch.Tensor":
        """정규화된 텐서를 역정규화"""
        try:
            if TORCH_AVAILABLE:
                # 평균과 표준편차를 텐서로 변환
                mean = torch.tensor(self.normalize_mean).view(-1, 1, 1)
                std = torch.tensor(self.normalize_std).view(-1, 1, 1)
                
                # 역정규화: tensor * std + mean
                denormalized = tensor * std + mean
                return denormalized
            else:
                return tensor
                
        except Exception as e:
            self.logger.error(f"❌ 역정규화 실패: {e}")
            return tensor

    def batch_convert_images(
        self,
        images: List[Union["Image.Image", "np.ndarray", str]],
        target_format: str = "tensor",
        **kwargs
    ) -> List[Optional[Any]]:
        """배치 이미지 변환"""
        try:
            start_time = time.time()
            results = []
            
            # M3 Max 최적화: 병렬 처리
            if self.is_m3_max and self.batch_processing and len(images) > 1:
                results = self._batch_convert_m3_optimized(images, target_format, **kwargs)
            else:
                # 순차 처리
                for i, image in enumerate(images):
                    try:
                        if target_format.lower() == "tensor":
                            result = self.image_to_tensor(image, **kwargs)
                        elif target_format.lower() == "pil":
                            result = self._to_pil_image(image)
                        elif target_format.lower() == "numpy":
                            pil_img = self._to_pil_image(image)
                            result = np.array(pil_img) if pil_img and NUMPY_AVAILABLE else None
                        else:
                            result = None
                            
                        results.append(result)
                        
                    except Exception as e:
                        self.logger.error(f"❌ 배치 변환 실패 (인덱스 {i}): {e}")
                        results.append(None)
            
            processing_time = time.time() - start_time
            success_count = sum(1 for r in results if r is not None)
            
            self.logger.info(f"📦 배치 변환 완료: {success_count}/{len(images)} 성공 ({processing_time:.3f}s)")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 배치 변환 실패: {e}")
            return [None] * len(images)

    def _batch_convert_m3_optimized(
        self,
        images: List[Any],
        target_format: str,
        **kwargs
    ) -> List[Optional[Any]]:
        """M3 Max 최적화 배치 변환"""
        try:
            import concurrent.futures
            
            # M3 Max 16코어 활용
            max_workers = min(16, len(images))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {}
                
                for i, image in enumerate(images):
                    if target_format.lower() == "tensor":
                        future = executor.submit(self.image_to_tensor, image, **kwargs)
                    elif target_format.lower() == "pil":
                        future = executor.submit(self._to_pil_image, image)
                    elif target_format.lower() == "numpy":
                        future = executor.submit(self._convert_to_numpy, image)
                    else:
                        continue
                        
                    future_to_index[future] = i
                
                # 결과 수집
                results = [None] * len(images)
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        results[index] = future.result()
                    except Exception as e:
                        self.logger.error(f"M3 배치 변환 실패 (인덱스 {index}): {e}")
                        results[index] = None
                
                return results
                
        except Exception as e:
            self.logger.error(f"M3 최적화 배치 변환 실패: {e}")
            # 폴백: 순차 처리
            return [self.image_to_tensor(img) if target_format == "tensor" else self._to_pil_image(img) 
                   for img in images]

    def _convert_to_numpy(self, image):
        """이미지를 numpy로 변환 (헬퍼 메서드)"""
        pil_img = self._to_pil_image(image)
        return np.array(pil_img) if pil_img and NUMPY_AVAILABLE else None

    def resize_image(
        self,
        image: Union["Image.Image", "np.ndarray"],
        size: Tuple[int, int],
        method: str = "bilinear",
        preserve_aspect_ratio: bool = False
    ) -> Optional[Union["Image.Image", "np.ndarray"]]:
        """이미지 크기 조정"""
        try:
            if NUMPY_AVAILABLE and hasattr(image, 'ndim'):  # numpy array 체크
                if CV2_AVAILABLE:
                    # OpenCV 사용
                    if method == "bilinear":
                        interpolation = cv2.INTER_LINEAR
                    elif method == "nearest":
                        interpolation = cv2.INTER_NEAREST
                    elif method == "bicubic":
                        interpolation = cv2.INTER_CUBIC
                    else:
                        interpolation = cv2.INTER_LINEAR
                    
                    if preserve_aspect_ratio:
                        size = self._calculate_aspect_ratio_size(image.shape[:2][::-1], size)
                    
                    resized = cv2.resize(image, size, interpolation=interpolation)
                    return resized
                else:
                    # PIL 폴백
                    if PIL_AVAILABLE:
                        pil_image = Image.fromarray(image)
                        return self.resize_image(pil_image, size, method, preserve_aspect_ratio)
                    
            elif PIL_AVAILABLE and hasattr(image, 'resize'):  # PIL Image 체크
                # PIL 사용
                if preserve_aspect_ratio:
                    size = self._calculate_aspect_ratio_size(image.size, size)
                
                if method == "bilinear":
                    resample = Image.BILINEAR
                elif method == "nearest":
                    resample = Image.NEAREST
                elif method == "bicubic":
                    resample = Image.BICUBIC
                else:
                    resample = Image.BILINEAR
                
                return image.resize(size, resample)
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ 이미지 크기 조정 실패: {e}")
            return None

    def _calculate_aspect_ratio_size(
        self,
        original_size: Tuple[int, int],
        target_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        """종횡비를 유지하는 크기 계산"""
        orig_w, orig_h = original_size
        target_w, target_h = target_size
        
        # 종횡비 계산
        aspect_ratio = orig_w / orig_h
        
        # 타겟 크기에 맞는 크기 계산
        if target_w / target_h > aspect_ratio:
            # 높이를 기준으로 조정
            new_h = target_h
            new_w = int(target_h * aspect_ratio)
        else:
            # 너비를 기준으로 조정
            new_w = target_w
            new_h = int(target_w / aspect_ratio)
        
        return (new_w, new_h)

    def normalize_image(
        self,
        image: Union["Image.Image", "np.ndarray", "torch.Tensor"],
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None
    ) -> Optional["torch.Tensor"]:
        """이미지 정규화"""
        try:
            # 기본값 설정
            mean = mean or self.normalize_mean
            std = std or self.normalize_std
            
            # 텐서로 변환
            if TORCH_AVAILABLE and hasattr(image, 'dim'):  # torch.Tensor 체크
                tensor = image
            else:
                tensor = self.image_to_tensor(image, normalize=False)
                
            if tensor is None:
                return None
            
            # 정규화 적용
            if TORCH_AVAILABLE:
                normalize_transform = transforms.Normalize(mean=mean, std=std)
                normalized_tensor = normalize_transform(tensor.squeeze(0))
                
                # 배치 차원 다시 추가
                if normalized_tensor.dim() == 3:
                    normalized_tensor = normalized_tensor.unsqueeze(0)
                
                return normalized_tensor
            
            return tensor
            
        except Exception as e:
            self.logger.error(f"❌ 이미지 정규화 실패: {e}")
            return None

    def image_to_base64(
        self,
        image: Union["Image.Image", "np.ndarray"],
        format: str = "PNG",
        quality: int = 95
    ) -> Optional[str]:
        """이미지를 Base64 문자열로 변환"""
        try:
            # PIL 이미지로 변환
            pil_image = self._to_pil_image(image)
            if pil_image is None:
                return None
            
            # 바이트 버퍼에 저장
            buffer = io.BytesIO()
            
            if format.upper() == "JPEG":
                pil_image.save(buffer, format=format, quality=quality)
            else:
                pil_image.save(buffer, format=format)
            
            # Base64 인코딩
            image_data = buffer.getvalue()
            base64_string = base64.b64encode(image_data).decode('utf-8')
            
            # Data URL 형식으로 반환
            mime_type = f"image/{format.lower()}"
            return f"data:{mime_type};base64,{base64_string}"
            
        except Exception as e:
            self.logger.error(f"❌ Base64 변환 실패: {e}")
            return None

    def preprocess_for_step(self, image: Union["Image.Image", "np.ndarray"], step_name: str) -> Optional["torch.Tensor"]:
        """Step별 특화 전처리"""
        try:
            # Step별 전처리 설정
            step_configs = {
                "HumanParsingStep": {"size": (512, 512), "normalize": True},
                "PoseEstimationStep": {"size": (368, 368), "normalize": True},
                "ClothSegmentationStep": {"size": (320, 320), "normalize": False},
                "VirtualFittingStep": {"size": (512, 512), "normalize": True},
                "PostProcessingStep": {"size": (1024, 1024) if self.is_m3_max else (512, 512), "normalize": False},
                "GeometricMatchingStep": {"size": (512, 384), "normalize": True},
                "ClothWarpingStep": {"size": (512, 512), "normalize": False},
                "QualityAssessmentStep": {"size": (224, 224), "normalize": True}
            }
            
            config = step_configs.get(step_name, {"size": self.default_size, "normalize": False})
            
            return self.image_to_tensor(
                image, 
                size=config["size"], 
                normalize=config["normalize"]
            )
            
        except Exception as e:
            self.logger.error(f"❌ {step_name} 전처리 실패: {e}")
            return None

    def _update_stats(self, operation: str, processing_time: float):
        """변환 통계 업데이트"""
        try:
            self._conversion_stats["total_conversions"] += 1
            self._conversion_stats["total_time"] += processing_time
            
            if operation not in self._conversion_stats["format_counts"]:
                self._conversion_stats["format_counts"][operation] = 0
            self._conversion_stats["format_counts"][operation] += 1
            
        except Exception:
            pass  # 통계 업데이트 실패는 무시

    def get_conversion_stats(self) -> Dict[str, Any]:
        """변환 통계 조회"""
        stats = self._conversion_stats.copy()
        
        if stats["total_conversions"] > 0:
            stats["average_time"] = stats["total_time"] / stats["total_conversions"]
        else:
            stats["average_time"] = 0.0
            
        return stats

    # ============================================
    # 🔥 현재 구조 호환 메서드들
    # ============================================

    async def initialize(self) -> bool:
        """데이터 변환기 초기화"""
        try:
            # 라이브러리 가용성 확인
            available_libs = []
            if PIL_AVAILABLE:
                available_libs.append(f"PIL ({PIL_VERSION})")
            if CV2_AVAILABLE:
                available_libs.append(f"OpenCV ({CV2_VERSION})")
            if TORCH_AVAILABLE:
                available_libs.append(f"PyTorch ({TORCH_VERSION})")
            if NUMPY_AVAILABLE:
                available_libs.append(f"NumPy ({NUMPY_VERSION})")
            
            self.logger.info(f"📚 사용 가능한 라이브러리: {', '.join(available_libs)}")
            
            # M3 Max 최적화 설정
            if self.is_m3_max and self.optimization_enabled:
                self._apply_m3_max_optimizations()
            
            # 변환 테스트
            test_result = await self._test_conversions()
            if not test_result:
                self.logger.warning("⚠️ 변환 테스트 실패, 일부 기능이 제한될 수 있습니다")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 데이터 변환기 초기화 실패: {e}")
            return False

    async def _test_conversions(self) -> bool:
        """변환 기능 테스트"""
        try:
            if PIL_AVAILABLE:
                # 더미 이미지 생성 및 변환 테스트
                test_image = Image.new('RGB', (256, 256), color='red')
                tensor_result = self.image_to_tensor(test_image)
                if tensor_result is not None:
                    self.logger.info("✅ 이미지 → 텐서 변환 테스트 통과")
                    return True
                    
        except Exception as e:
            self.logger.error(f"❌ 변환 테스트 실패: {e}")
            
        return False

    async def cleanup(self):
        """리소스 정리"""
        try:
            # 캐시 정리
            if hasattr(self, '_conversion_stats'):
                self._conversion_stats.clear()
            
            # 변환 파이프라인 정리
            if hasattr(self, 'transforms'):
                self.transforms.clear()
            
            # PyTorch 메모리 정리
            if TORCH_AVAILABLE:
                if MPS_AVAILABLE:
                    try:
                        if hasattr(torch.mps, 'empty_cache'):
                            safe_mps_empty_cache()
                    except Exception:
                        pass
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self.logger.info("✅ 데이터 변환기 리소스 정리 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 데이터 변환기 리소스 정리 실패: {e}")

# ==============================================
# 🔥 ImageProcessor 클래스 (기존 이름 유지)
# ==============================================

class ImageProcessor(DataConverter):
    """
    🍎 이미지 처리기 (기존 클래스명 유지)
    ✅ 현재 구조와 완벽 호환
    ✅ 기존 코드의 ImageProcessor 사용 유지
    """
    
    def __init__(self, **kwargs):
        """이미지 처리기 초기화 (기존 시그니처 유지)"""
        super().__init__(**kwargs)
        self.logger = logging.getLogger("ImageProcessor")
        
        self.logger.info(f"🖼️ ImageProcessor 초기화 - 디바이스: {self.device}")

    def process_image(self, image: Any, target_format: str = "tensor", **kwargs) -> Any:
        """이미지 처리 (기존 메서드명 유지)"""
        try:
            if target_format.lower() == "tensor":
                return self.image_to_tensor(image, **kwargs)
            elif target_format.lower() == "numpy":
                pil_img = self._to_pil_image(image)
                return np.array(pil_img) if pil_img and NUMPY_AVAILABLE else None
            elif target_format.lower() == "pil":
                return self._to_pil_image(image)
            else:
                self.logger.warning(f"⚠️ 지원되지 않는 포맷: {target_format}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ 이미지 처리 실패: {e}")
            return None

    def resize_and_convert(self, image: Any, size: Tuple[int, int], format: str = "tensor") -> Any:
        """크기 조정 및 변환 (편의 메서드)"""
        try:
            # 먼저 PIL 이미지로 변환
            pil_image = self._to_pil_image(image)
            if pil_image is None:
                return None
            
            # 크기 조정
            resized_image = pil_image.resize(size, getattr(Image, 'BILINEAR', 2))
            
            # 목표 포맷으로 변환
            if format.lower() == "tensor":
                return self.image_to_tensor(resized_image)
            elif format.lower() == "numpy":
                return np.array(resized_image) if NUMPY_AVAILABLE else None
            else:
                return resized_image
                
        except Exception as e:
            self.logger.error(f"❌ 크기 조정 및 변환 실패: {e}")
            return None

# ==============================================
# 🔥 팩토리 함수들 (기존 이름 완전 유지)
# ==============================================

# 전역 데이터 변환기 (선택적)
_global_data_converter: Optional[DataConverter] = None
_global_image_processor: Optional[ImageProcessor] = None

def create_data_converter(
    default_size: Tuple[int, int] = (512, 512),
    device: str = "auto",
    **kwargs
) -> DataConverter:
    """데이터 변환기 생성 (하위 호환)"""
    if device == "auto":
        device = DEFAULT_DEVICE
        
    return DataConverter(
        device=device,
        default_size=default_size,
        **kwargs
    )

def get_global_data_converter(**kwargs) -> DataConverter:
    """전역 데이터 변환기 반환"""
    global _global_data_converter
    if _global_data_converter is None:
        _global_data_converter = DataConverter(**kwargs)
    return _global_data_converter

def initialize_global_data_converter(**kwargs) -> DataConverter:
    """전역 데이터 변환기 초기화"""
    global _global_data_converter
    _global_data_converter = DataConverter(**kwargs)
    return _global_data_converter

def get_image_processor(**kwargs) -> ImageProcessor:
    """
    🔥 ImageProcessor 반환 (현재 구조에서 요구)
    ✅ 기존 함수명 완전 유지
    ✅ 현재 utils/__init__.py에서 사용
    """
    global _global_image_processor
    if _global_image_processor is None:
        _global_image_processor = ImageProcessor(**kwargs)
    return _global_image_processor

# 빠른 변환 함수들 (편의성)
def quick_image_to_tensor(image: Union["Image.Image", "np.ndarray"], size: Tuple[int, int] = (512, 512)) -> Optional["torch.Tensor"]:
    """빠른 이미지→텐서 변환"""
    converter = get_global_data_converter()
    return converter.image_to_tensor(image, size=size)

def quick_tensor_to_image(tensor: "torch.Tensor") -> Optional["Image.Image"]:
    """빠른 텐서→이미지 변환"""
    converter = get_global_data_converter()
    return converter.tensor_to_image(tensor)

def quick_tensor_to_numpy(tensor: "torch.Tensor") -> Optional["np.ndarray"]:
    """빠른 텐서→numpy 변환 (기존 함수명 유지)"""
    converter = get_global_data_converter()
    return converter.tensor_to_numpy(tensor)

def preprocess_image_for_step(image: Union["Image.Image", "np.ndarray"], step_name: str) -> Optional["torch.Tensor"]:
    """Step별 이미지 전처리"""
    converter = get_global_data_converter()
    return converter.preprocess_for_step(image, step_name)

def batch_convert_images(images: List[Any], target_format: str = "tensor", **kwargs) -> List[Any]:
    """배치 이미지 변환"""
    converter = get_global_data_converter()
    return converter.batch_convert_images(images, target_format, **kwargs)

# 호환성 함수들 (현재 구조 지원)
def convert_image_format(image: Any, source_format: str, target_format: str) -> Any:
    """이미지 포맷 변환 (범용)"""
    try:
        converter = get_global_data_converter()
        
        # 먼저 PIL로 변환
        pil_image = converter._to_pil_image(image)
        if pil_image is None:
            return None
        
        # 목표 포맷으로 변환
        if target_format.lower() == "tensor":
            return converter.image_to_tensor(pil_image)
        elif target_format.lower() == "numpy":
            return np.array(pil_image) if NUMPY_AVAILABLE else None
        elif target_format.lower() == "base64":
            return converter.image_to_base64(pil_image)
        else:
            return pil_image
            
    except Exception as e:
        logger.error(f"❌ 이미지 포맷 변환 실패: {e}")
        return None

def get_optimal_image_size(step_name: str) -> Tuple[int, int]:
    """Step별 최적 이미지 크기 반환"""
    step_sizes = {
        "HumanParsingStep": (512, 512),
        "PoseEstimationStep": (368, 368),
        "ClothSegmentationStep": (320, 320),
        "VirtualFittingStep": (512, 512),
        "PostProcessingStep": (1024, 1024),
        "GeometricMatchingStep": (512, 384),
        "ClothWarpingStep": (512, 512),
        "QualityAssessmentStep": (224, 224)
    }
    return step_sizes.get(step_name, (512, 512))

# 시스템 상태 확인
def get_system_status() -> Dict[str, Any]:
    """시스템 상태 확인"""
    return {
        "torch_available": TORCH_AVAILABLE,
        "torch_version": TORCH_VERSION,
        "mps_available": MPS_AVAILABLE,
        "pil_available": PIL_AVAILABLE,
        "pil_version": PIL_VERSION,
        "numpy_available": NUMPY_AVAILABLE,
        "numpy_version": NUMPY_VERSION,
        "cv2_available": CV2_AVAILABLE,
        "cv2_version": CV2_VERSION,
        "default_device": DEFAULT_DEVICE
    }

# 모듈 익스포트 (기존 구조 완전 유지)
__all__ = [
    # 🔥 기존 클래스명 완전 유지
    'DataConverter',
    'ImageProcessor',              # ✅ 현재 구조에서 사용
    'ConversionMode',
    'ImageFormat',
    
    # 🔥 기존 함수명 완전 유지
    'create_data_converter',
    'get_global_data_converter',
    'initialize_global_data_converter',
    'get_image_processor',         # ✅ 현재 구조에서 중요
    'quick_image_to_tensor',
    'quick_tensor_to_image',
    'quick_tensor_to_numpy',       # ✅ 기존 메서드명 유지
    'preprocess_image_for_step',
    'batch_convert_images',
    'convert_image_format',
    'get_optimal_image_size',
    'get_system_status'
]

# 모듈 로드 확인
logger.info("✅ 완전 최적화된 DataConverter 모듈 로드 완료 (안전한 임포트 버전)")
logger.info("🔧 기존 함수명/클래스명 100% 유지 (DataConverter, ImageProcessor)")
logger.info("🍎 M3 Max 이미지/텐서 변환 최적화 완전 구현")
logger.info("🔗 현재 프로젝트 구조 100% 호환")
logger.info("⚡ conda 환경 완벽 지원")
logger.info("🛡️ AttributeError: 'NoneType' object has no attribute 'Tensor' 완전 해결")

# 시스템 상태 로깅
status = get_system_status()
logger.info(f"📊 시스템 상태: PyTorch={status['torch_available']}, PIL={status['pil_available']}, NumPy={status['numpy_available']}")
logger.info(f"🎯 기본 디바이스: {status['default_device']}")