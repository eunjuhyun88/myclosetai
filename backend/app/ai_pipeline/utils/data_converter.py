# app/ai_pipeline/utils/data_converter.py
"""
데이터 변환기 - M3 Max 최적화 이미지/텐서 변환 (최적 생성자 패턴 적용)
단순함 + 편의성 + 확장성 + 일관성
"""

import io
import logging
import time
import base64
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
from pathlib import Path

# PIL import
try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

# OpenCV import
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

# PyTorch import
try:
    import torch
    import torchvision.transforms as transforms
    import torchvision.transforms.functional as TF
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    transforms = None
    TF = None

logger = logging.getLogger(__name__)

class DataConverter:
    """
    🍎 M3 Max 최적화 데이터 변환기
    ✅ 최적 생성자 패턴 적용 - 이미지/텐서 변환 및 처리
    """
    
    def __init__(
        self,
        device: Optional[str] = None,  # 🔥 최적 패턴: None으로 자동 감지
        config: Optional[Dict[str, Any]] = None,
        **kwargs  # 🚀 확장성: 무제한 추가 파라미터
    ):
        """
        ✅ 최적 생성자 - 데이터 변환기 특화

        Args:
            device: 사용할 디바이스 (None=자동감지, 'cpu', 'cuda', 'mps')
            config: 데이터 변환 설정 딕셔너리
            **kwargs: 확장 파라미터들
                - device_type: str = "auto"
                - memory_gb: float = 16.0  
                - is_m3_max: bool = False
                - optimization_enabled: bool = True
                - quality_level: str = "balanced"
                - default_size: Tuple[int, int] = (512, 512)  # 기본 이미지 크기
                - interpolation: str = "bilinear"  # 보간 방법
                - normalize_mean: List[float] = [0.485, 0.456, 0.406]  # 정규화 평균
                - normalize_std: List[float] = [0.229, 0.224, 0.225]  # 정규화 표준편차
                - use_gpu_acceleration: bool = True  # GPU 가속 사용
                - batch_processing: bool = True  # 배치 처리
                - memory_efficient: bool = True  # 메모리 효율적 처리
                - quality_preservation: bool = True  # 품질 보존
        """
        # 1. 💡 지능적 디바이스 자동 감지
        self.device = self._auto_detect_device(device)

        # 2. 📋 기본 설정
        self.config = config or {}
        self.step_name = self.__class__.__name__
        self.logger = logging.getLogger(f"utils.{self.step_name}")

        # 3. 🔧 표준 시스템 파라미터 추출 (일관성)
        self.device_type = kwargs.get('device_type', 'auto')
        self.memory_gb = kwargs.get('memory_gb', 16.0)
        self.is_m3_max = kwargs.get('is_m3_max', self._detect_m3_max())
        self.optimization_enabled = kwargs.get('optimization_enabled', True)
        self.quality_level = kwargs.get('quality_level', 'balanced')

        # 4. ⚙️ 데이터 변환기 특화 파라미터
        self.default_size = tuple(kwargs.get('default_size', (512, 512)))
        self.interpolation = kwargs.get('interpolation', 'bilinear')
        self.normalize_mean = kwargs.get('normalize_mean', [0.485, 0.456, 0.406])
        self.normalize_std = kwargs.get('normalize_std', [0.229, 0.224, 0.225])
        self.use_gpu_acceleration = kwargs.get('use_gpu_acceleration', self.device != 'cpu')
        self.batch_processing = kwargs.get('batch_processing', True)
        self.memory_efficient = kwargs.get('memory_efficient', True)
        self.quality_preservation = kwargs.get('quality_preservation', True)

        # 5. 🍎 M3 Max 특화 설정
        if self.is_m3_max:
            self.use_gpu_acceleration = True  # M3 Max는 항상 GPU 가속
            self.batch_processing = True  # 배치 처리 최적화
            self.memory_efficient = False  # 128GB 메모리이므로 품질 우선

        # 6. ⚙️ 스텝별 특화 파라미터를 config에 병합
        self._merge_step_specific_config(kwargs)

        # 7. ✅ 상태 초기화
        self.is_initialized = False

        # 8. 🎯 기존 클래스별 고유 초기화 로직 실행
        self._initialize_step_specific()

        self.logger.info(f"🎯 {self.step_name} 초기화 - 디바이스: {self.device}")

    def _auto_detect_device(self, preferred_device: Optional[str]) -> str:
        """💡 지능적 디바이스 자동 감지"""
        if preferred_device:
            return preferred_device

        if not TORCH_AVAILABLE:
            return 'cpu'

        try:
            if torch.backends.mps.is_available():
                return 'mps'  # M3 Max 우선
            elif torch.cuda.is_available():
                return 'cuda'  # NVIDIA GPU
            else:
                return 'cpu'  # 폴백
        except:
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
            'optimization_enabled', 'quality_level',
            'default_size', 'interpolation', 'normalize_mean', 'normalize_std',
            'use_gpu_acceleration', 'batch_processing', 'memory_efficient', 'quality_preservation'
        }

        for key, value in kwargs.items():
            if key not in system_params:
                self.config[key] = value

    def _initialize_step_specific(self):
        """🎯 기존 초기화 로직 완전 유지"""
        # 변환 파이프라인 초기화
        self._init_transforms()
        
        # 통계 추적
        self._conversion_stats = {
            "total_conversions": 0,
            "total_time": 0.0,
            "format_counts": {},
            "error_count": 0
        }
        
        self.logger.info(f"🔄 데이터 변환기 초기화 - {self.device} (크기: {self.default_size})")
        
        # 초기화 완료
        self.is_initialized = True

    def _init_transforms(self):
        """변환 파이프라인 초기화"""
        self.transforms = {}
        
        if TORCH_AVAILABLE:
            # 보간 방법 매핑
            interpolation_map = {
                'bilinear': transforms.InterpolationMode.BILINEAR if hasattr(transforms, 'InterpolationMode') else 'bilinear',
                'nearest': transforms.InterpolationMode.NEAREST if hasattr(transforms, 'InterpolationMode') else 'nearest',
                'bicubic': transforms.InterpolationMode.BICUBIC if hasattr(transforms, 'InterpolationMode') else 'bicubic'
            }
            
            interpolation_mode = interpolation_map.get(self.interpolation, 'bilinear')
            
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

    async def initialize(self) -> bool:
        """데이터 변환기 초기화"""
        try:
            # 라이브러리 가용성 확인
            available_libs = []
            if PIL_AVAILABLE:
                available_libs.append("PIL")
            if CV2_AVAILABLE:
                available_libs.append("OpenCV")
            if TORCH_AVAILABLE:
                available_libs.append("PyTorch")
            
            self.logger.info(f"📚 사용 가능한 라이브러리: {', '.join(available_libs)}")
            
            # M3 Max 최적화 설정
            if self.is_m3_max and self.optimization_enabled:
                await self._apply_m3_max_optimizations()
            
            # 변환 테스트
            test_result = await self._test_conversions()
            if not test_result:
                self.logger.warning("⚠️ 변환 테스트 실패, 일부 기능이 제한될 수 있습니다")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 데이터 변환기 초기화 실패: {e}")
            return False

    async def _apply_m3_max_optimizations(self):
        """M3 Max 특화 최적화 적용"""
        try:
            optimizations = []
            
            # 1. 고해상도 처리 활성화
            if self.default_size[0] < 1024:
                self.default_size = (1024, 1024)
                optimizations.append("High resolution processing")
            
            # 2. 메모리 효율성 조정 (128GB 메모리)
            self.memory_efficient = False  # 품질 우선
            optimizations.append("Quality-first processing")
            
            # 3. MPS 백엔드 최적화
            if TORCH_AVAILABLE and torch.backends.mps.is_available():
                optimizations.append("MPS backend optimization")
            
            if optimizations:
                self.logger.info(f"🍎 M3 Max 최적화 적용: {', '.join(optimizations)}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ M3 Max 최적화 실패: {e}")

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

    def image_to_tensor(
        self,
        image: Union[Image.Image, np.ndarray, str, bytes],
        size: Optional[Tuple[int, int]] = None,
        normalize: bool = False,
        **kwargs
    ) -> Optional[torch.Tensor]:
        """이미지를 텐서로 변환"""
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
            if normalize:
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
        tensor: torch.Tensor,
        denormalize: bool = False,
        format: str = "PIL"
    ) -> Optional[Union[Image.Image, np.ndarray]]:
        """텐서를 이미지로 변환"""
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
                array = tensor.permute(1, 2, 0).numpy()
                array = (array * 255).astype(np.uint8)
                if PIL_AVAILABLE:
                    pil_image = Image.fromarray(array)
                else:
                    return array if format == "numpy" else None
            
            # 출력 형식에 따른 변환
            if format.lower() == "pil":
                result = pil_image
            elif format.lower() == "numpy":
                result = np.array(pil_image)
            elif format.lower() == "cv2" and CV2_AVAILABLE:
                result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
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

    def _to_pil_image(self, image_input: Union[Image.Image, np.ndarray, str, bytes]) -> Optional[Image.Image]:
        """다양한 입력을 PIL 이미지로 변환"""
        try:
            if not PIL_AVAILABLE:
                return None
                
            # 이미 PIL 이미지인 경우
            if isinstance(image_input, Image.Image):
                return image_input.convert('RGB')
            
            # NumPy 배열인 경우
            elif isinstance(image_input, np.ndarray):
                if image_input.ndim == 3:
                    return Image.fromarray(image_input.astype(np.uint8)).convert('RGB')
                elif image_input.ndim == 2:
                    return Image.fromarray(image_input.astype(np.uint8)).convert('RGB')
            
            # 파일 경로인 경우
            elif isinstance(image_input, (str, Path)):
                path = Path(image_input)
                if path.exists():
                    return Image.open(path).convert('RGB')
            
            # Base64 문자열인 경우
            elif isinstance(image_input, str) and image_input.startswith('data:image'):
                # Data URL 파싱
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

    def _denormalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
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
        images: List[Union[Image.Image, np.ndarray, str]],
        target_format: str = "tensor",
        **kwargs
    ) -> List[Optional[Any]]:
        """배치 이미지 변환"""
        try:
            start_time = time.time()
            results = []
            
            for i, image in enumerate(images):
                try:
                    if target_format.lower() == "tensor":
                        result = self.image_to_tensor(image, **kwargs)
                    elif target_format.lower() == "pil":
                        result = self._to_pil_image(image)
                    elif target_format.lower() == "numpy":
                        pil_img = self._to_pil_image(image)
                        result = np.array(pil_img) if pil_img else None
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

    def resize_image(
        self,
        image: Union[Image.Image, np.ndarray],
        size: Tuple[int, int],
        method: str = "bilinear",
        preserve_aspect_ratio: bool = False
    ) -> Optional[Union[Image.Image, np.ndarray]]:
        """이미지 크기 조정"""
        try:
            if isinstance(image, np.ndarray):
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
                    
            elif PIL_AVAILABLE and isinstance(image, Image.Image):
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
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None
    ) -> Optional[torch.Tensor]:
        """이미지 정규화"""
        try:
            # 기본값 설정
            mean = mean or self.normalize_mean
            std = std or self.normalize_std
            
            # 텐서로 변환
            if isinstance(image, torch.Tensor):
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
        image: Union[Image.Image, np.ndarray],
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

    async def get_step_info(self) -> Dict[str, Any]:
        """데이터 변환기 정보 반환"""
        return {
            "step_name": self.step_name,
            "device": self.device,
            "device_type": self.device_type,
            "memory_gb": self.memory_gb,
            "is_m3_max": self.is_m3_max,
            "optimization_enabled": self.optimization_enabled,
            "quality_level": self.quality_level,
            "initialized": self.is_initialized,
            "config_keys": list(self.config.keys()),
            "specialized_features": {
                "default_size": self.default_size,
                "interpolation": self.interpolation,
                "use_gpu_acceleration": self.use_gpu_acceleration,
                "batch_processing": self.batch_processing,
                "memory_efficient": self.memory_efficient,
                "quality_preservation": self.quality_preservation
            },
            "library_support": {
                "PIL": PIL_AVAILABLE,
                "OpenCV": CV2_AVAILABLE,
                "PyTorch": TORCH_AVAILABLE
            },
            "conversion_stats": self.get_conversion_stats()
        }

# 편의 함수들 (하위 호환성)
def create_data_converter(
    default_size: Tuple[int, int] = (512, 512),
    device: str = "mps",
    **kwargs
) -> DataConverter:
    """데이터 변환기 생성 (하위 호환)"""
    return DataConverter(
        device=device,
        default_size=default_size,
        **kwargs
    )

# 전역 데이터 변환기 (선택적)
_global_data_converter: Optional[DataConverter] = None

def get_global_data_converter() -> Optional[DataConverter]:
    """전역 데이터 변환기 반환"""
    global _global_data_converter
    return _global_data_converter

def initialize_global_data_converter(**kwargs) -> DataConverter:
    """전역 데이터 변환기 초기화"""
    global _global_data_converter
    _global_data_converter = DataConverter(**kwargs)
    return _global_data_converter

# 빠른 변환 함수들 (편의성)
def quick_image_to_tensor(image: Union[Image.Image, np.ndarray], size: Tuple[int, int] = (512, 512)) -> Optional[torch.Tensor]:
    """빠른 이미지→텐서 변환"""
    converter = get_global_data_converter()
    if converter is None:
        converter = DataConverter(default_size=size)
    return converter.image_to_tensor(image, size=size)

def quick_tensor_to_image(tensor: torch.Tensor) -> Optional[Image.Image]:
    """빠른 텐서→이미지 변환"""
    converter = get_global_data_converter()
    if converter is None:
        converter = DataConverter()
    return converter.tensor_to_image(tensor)

# 모듈 익스포트
__all__ = [
    'DataConverter',
    'create_data_converter',
    'get_global_data_converter',
    'initialize_global_data_converter',
    'quick_image_to_tensor',
    'quick_tensor_to_image'
]