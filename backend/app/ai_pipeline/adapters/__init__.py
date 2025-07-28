# app/ai_pipeline/adapters/__init__.py
"""어댑터 패키지"""

from .model_adapter import ModelLoaderAdapter, StepInterfaceAdapter
from .memory_adapter import MemoryManagerAdapter
from .data_adapter import DataConverterAdapter

__all__ = [
    'ModelLoaderAdapter',
    'StepInterfaceAdapter', 
    'MemoryManagerAdapter',
    'DataConverterAdapter'
]

# ==============================================
# app/ai_pipeline/adapters/model_adapter.py
# ==============================================
"""
🔥 모델 어댑터 구현
==================

✅ 기존 ModelLoader를 IModelLoader 인터페이스로 래핑
✅ 순환참조 방지
✅ 안전한 지연 로딩
"""

import logging
from typing import Any, Dict, Optional, List
from ..interface.model_interface import IModelLoader, IStepInterface

logger = logging.getLogger(__name__)

class StepInterfaceAdapter(IStepInterface):
    """Step 인터페이스 어댑터"""
    
    def __init__(self, real_interface: Any = None, step_name: str = "unknown"):
        self.real_interface = real_interface
        self.step_name = step_name
        self.logger = logging.getLogger(f"{__name__}.StepInterfaceAdapter.{step_name}")
        
    def get_model(self, model_name: Optional[str] = None) -> Optional[Any]:
        """동기 모델 조회"""
        try:
            if self.real_interface and hasattr(self.real_interface, 'get_model_sync'):
                return self.real_interface.get_model_sync(model_name)
            elif self.real_interface and hasattr(self.real_interface, 'get_model'):
                result = self.real_interface.get_model(model_name)
                # coroutine 객체 처리
                if hasattr(result, '__await__'):
                    self.logger.warning(f"⚠️ 비동기 결과를 동기로 변환: {model_name}")
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # 이미 실행 중인 루프에서는 새 태스크 생성
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(asyncio.run, result)
                                return future.result(timeout=30)
                        else:
                            return asyncio.run(result)
                    except Exception as e:
                        self.logger.error(f"❌ 비동기 변환 실패: {e}")
                        return self._create_mock_model(model_name or "default")
                return result
            else:
                return self._create_mock_model(model_name or "default")
        except Exception as e:
            self.logger.error(f"❌ 모델 조회 실패 {model_name}: {e}")
            return self._create_mock_model(model_name or "error")
    
    async def get_model_async(self, model_name: Optional[str] = None) -> Optional[Any]:
        """비동기 모델 조회"""
        try:
            if self.real_interface and hasattr(self.real_interface, 'get_model'):
                result = self.real_interface.get_model(model_name)
                if hasattr(result, '__await__'):
                    return await result
                else:
                    return result
            else:
                return self._create_mock_model(model_name or "default")
        except Exception as e:
            self.logger.error(f"❌ 비동기 모델 조회 실패 {model_name}: {e}")
            return self._create_mock_model(model_name or "error")
    
    def list_available_models(self) -> List[str]:
        """사용 가능한 모델 목록"""
        try:
            if self.real_interface and hasattr(self.real_interface, 'list_available_models'):
                return self.real_interface.list_available_models()
            else:
                return [f"{self.step_name}_default", f"{self.step_name}_backup"]
        except Exception as e:
            self.logger.error(f"❌ 모델 목록 조회 실패: {e}")
            return []
    
    def _create_mock_model(self, model_name: str) -> Any:
        """Mock 모델 생성"""
        class MockModel:
            def __init__(self, name: str, step: str):
                self.name = name
                self.step = step
                self.device = "cpu"
                
            def __call__(self, *args, **kwargs):
                return {
                    'status': 'success',
                    'model_name': self.name,
                    'step_name': self.step,
                    'result': f'mock_result_for_{self.name}',
                    'type': 'mock_adapter'
                }
            
            def to(self, device):
                self.device = str(device)
                return self
            
            def eval(self):
                return self
        
        return MockModel(model_name, self.step_name)

class ModelLoaderAdapter(IModelLoader):
    """ModelLoader 어댑터"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ModelLoaderAdapter")
        self._model_loader = None
        self._initialized = False
        self._initialization_attempted = False
        
    def _ensure_initialized(self):
        """지연 초기화"""
        if self._initialization_attempted:
            return self._model_loader is not None
            
        self._initialization_attempted = True
        
        try:
            # 실제 ModelLoader 가져오기
            from ..utils.model_loader import get_global_model_loader
            self._model_loader = get_global_model_loader()
            self._initialized = True
            self.logger.info("✅ 실제 ModelLoader 연결 성공")
            return True
        except ImportError as e:
            self.logger.warning(f"⚠️ ModelLoader import 실패: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ ModelLoader 초기화 실패: {e}")
            return False
    
    def get_model(self, model_name: str, **kwargs) -> Optional[Any]:
        """동기 모델 로드"""
        if self._ensure_initialized() and self._model_loader:
            try:
                if hasattr(self._model_loader, 'load_model_sync'):
                    return self._model_loader.load_model_sync(model_name, **kwargs)
                elif hasattr(self._model_loader, 'get_model'):
                    return self._model_loader.get_model(model_name, **kwargs)
                else:
                    self.logger.warning(f"⚠️ ModelLoader에 적절한 메서드 없음")
                    return None
            except Exception as e:
                self.logger.error(f"❌ 모델 로드 실패 {model_name}: {e}")
                return None
        
        # 폴백 처리
        self.logger.warning(f"⚠️ 폴백 모델 생성: {model_name}")
        return self._create_fallback_model(model_name)
    
    async def get_model_async(self, model_name: str, **kwargs) -> Optional[Any]:
        """비동기 모델 로드"""
        if self._ensure_initialized() and self._model_loader:
            try:
                if hasattr(self._model_loader, 'load_model_async'):
                    return await self._model_loader.load_model_async(model_name, **kwargs)
                elif hasattr(self._model_loader, 'get_model_async'):
                    return await self._model_loader.get_model_async(model_name, **kwargs)
                elif hasattr(self._model_loader, 'get_model'):
                    # 동기 메서드를 비동기로 실행
                    import asyncio
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None, 
                        lambda: self._model_loader.get_model(model_name, **kwargs)
                    )
            except Exception as e:
                self.logger.error(f"❌ 비동기 모델 로드 실패 {model_name}: {e}")
                return None
        
        # 폴백 처리
        return self._create_fallback_model(model_name)
    
    def create_step_interface(self, step_name: str, **kwargs) -> IStepInterface:
        """Step 인터페이스 생성"""
        if self._ensure_initialized() and self._model_loader:
            try:
                if hasattr(self._model_loader, 'create_step_interface'):
                    real_interface = self._model_loader.create_step_interface(step_name, **kwargs)
                    return StepInterfaceAdapter(real_interface, step_name)
                else:
                    self.logger.warning(f"⚠️ create_step_interface 메서드 없음")
            except Exception as e:
                self.logger.error(f"❌ Step 인터페이스 생성 실패 {step_name}: {e}")
        
        # 폴백: Mock 인터페이스
        return StepInterfaceAdapter(None, step_name)
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """모델 목록 조회"""
        if self._ensure_initialized() and self._model_loader:
            try:
                if hasattr(self._model_loader, 'list_models'):
                    return self._model_loader.list_models()
            except Exception as e:
                self.logger.error(f"❌ 모델 목록 조회 실패: {e}")
        
        # 폴백
        return {
            'default_model': {
                'name': 'default_model',
                'status': 'fallback',
                'source': 'adapter'
            }
        }
    
    def cleanup(self):
        """리소스 정리"""
        if self._model_loader and hasattr(self._model_loader, 'cleanup'):
            try:
                self._model_loader.cleanup()
                self.logger.info("✅ ModelLoader 정리 완료")
            except Exception as e:
                self.logger.error(f"❌ ModelLoader 정리 실패: {e}")
        
        self._model_loader = None
        self._initialized = False
        self._initialization_attempted = False
    
    def _create_fallback_model(self, model_name: str) -> Any:
        """폴백 모델 생성"""
        class FallbackModel:
            def __init__(self, name: str):
                self.name = name
                self.device = "cpu"
                
            def __call__(self, *args, **kwargs):
                return {
                    'status': 'success',
                    'model_name': self.name,
                    'result': f'fallback_result_for_{self.name}',
                    'type': 'fallback_adapter'
                }
            
            def to(self, device):
                self.device = str(device)
                return self
            
            def eval(self):
                return self
        
        return FallbackModel(model_name)

# ==============================================
# app/ai_pipeline/adapters/memory_adapter.py
# ==============================================
"""
🔥 메모리 관리자 어댑터
======================
"""

import logging
import gc
import time
from typing import Dict, Any
from ..interface.memory_interface import IMemoryManager

class MemoryManagerAdapter(IMemoryManager):
    """메모리 관리자 어댑터"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.MemoryManagerAdapter")
        self._memory_manager = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """지연 초기화"""
        if self._initialized:
            return self._memory_manager is not None
            
        try:
            from ..utils.memory_manager import get_global_memory_manager
            self._memory_manager = get_global_memory_manager()
            self._initialized = True
            self.logger.info("✅ 실제 MemoryManager 연결 성공")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ MemoryManager 초기화 실패: {e}")
            self._initialized = True
            return False
    
    def optimize_memory(self, **kwargs) -> Dict[str, Any]:
        """동기 메모리 최적화"""
        if self._ensure_initialized() and self._memory_manager:
            try:
                if hasattr(self._memory_manager, 'optimize_memory'):
                    return self._memory_manager.optimize_memory(**kwargs)
            except Exception as e:
                self.logger.error(f"❌ 메모리 최적화 실패: {e}")
        
        # 폴백: 기본 최적화
        return self._basic_memory_optimization()
    
    async def optimize_memory_async(self, **kwargs) -> Dict[str, Any]:
        """비동기 메모리 최적화"""
        if self._ensure_initialized() and self._memory_manager:
            try:
                if hasattr(self._memory_manager, 'optimize_memory_async'):
                    return await self._memory_manager.optimize_memory_async(**kwargs)
                else:
                    # 동기 메서드를 비동기로 실행
                    import asyncio
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None, 
                        lambda: self.optimize_memory(**kwargs)
                    )
            except Exception as e:
                self.logger.error(f"❌ 비동기 메모리 최적화 실패: {e}")
        
        # 폴백
        return self._basic_memory_optimization()
    
    def get_memory_status(self) -> Dict[str, Any]:
        """메모리 상태 조회"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / 1024**3,
                'available_gb': memory.available / 1024**3,
                'percent_used': memory.percent,
                'adapter': True
            }
        except Exception as e:
            self.logger.error(f"❌ 메모리 상태 조회 실패: {e}")
            return {'error': str(e), 'adapter': True}
    
    def cleanup_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """메모리 정리"""
        try:
            before_objects = len(gc.get_objects())
            gc.collect()
            after_objects = len(gc.get_objects())
            
            # PyTorch 메모리 정리
            try:
                import torch
                if torch.backends.mps.is_available():
                    if hasattr(torch.mps, 'empty_cache'):
                        safe_mps_empty_cache()
                    elif hasattr(torch.backends.mps, 'empty_cache'):
                        torch.backends.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            return {
                'success': True,
                'objects_freed': before_objects - after_objects,
                'aggressive': aggressive,
                'adapter': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e), 'adapter': True}
    
    def _basic_memory_optimization(self) -> Dict[str, Any]:
        """기본 메모리 최적화"""
        try:
            start_time = time.time()
            
            before_objects = len(gc.get_objects())
            gc.collect()
            after_objects = len(gc.get_objects())
            
            duration = time.time() - start_time
            
            return {
                'success': True,
                'duration': duration,
                'objects_freed': before_objects - after_objects,
                'method': 'basic_fallback',
                'adapter': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e), 'adapter': True}

# ==============================================
# app/ai_pipeline/adapters/data_adapter.py
# ==============================================
"""
🔥 데이터 변환기 어댑터
======================
"""

import logging
from typing import Any, Tuple, Union
from PIL import Image
import numpy as np
from ..interface.data_interface import IDataConverter

class DataConverterAdapter(IDataConverter):
    """데이터 변환기 어댑터"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DataConverterAdapter")
        self._data_converter = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """지연 초기화"""
        if self._initialized:
            return self._data_converter is not None
            
        try:
            from ..utils.data_converter import DataConverter
            self._data_converter = DataConverter()
            self._initialized = True
            self.logger.info("✅ 실제 DataConverter 연결 성공")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ DataConverter 초기화 실패: {e}")
            self._initialized = True
            return False
    
    def convert_image(self, image: Any, target_format: str = "tensor", **kwargs) -> Any:
        """이미지 변환"""
        if self._ensure_initialized() and self._data_converter:
            try:
                if hasattr(self._data_converter, 'convert_image'):
                    return self._data_converter.convert_image(image, target_format, **kwargs)
            except Exception as e:
                self.logger.error(f"❌ 이미지 변환 실패: {e}")
        
        # 폴백 처리
        return self._basic_image_conversion(image, target_format, **kwargs)
    
    def preprocess_image(self, image: Any, size: Tuple[int, int] = (512, 512), **kwargs) -> Any:
        """이미지 전처리"""
        if self._ensure_initialized() and self._data_converter:
            try:
                if hasattr(self._data_converter, 'preprocess_image'):
                    return self._data_converter.preprocess_image(image, size, **kwargs)
            except Exception as e:
                self.logger.error(f"❌ 이미지 전처리 실패: {e}")
        
        # 폴백 처리
        return self._basic_image_preprocessing(image, size, **kwargs)
    
    def postprocess_result(self, result: Any, output_format: str = "image", **kwargs) -> Any:
        """결과 후처리"""
        if self._ensure_initialized() and self._data_converter:
            try:
                if hasattr(self._data_converter, 'postprocess_result'):
                    return self._data_converter.postprocess_result(result, output_format, **kwargs)
            except Exception as e:
                self.logger.error(f"❌ 결과 후처리 실패: {e}")
        
        # 폴백 처리
        return result
    
    def tensor_to_image(self, tensor: Any, **kwargs) -> Image.Image:
        """텐서를 이미지로 변환"""
        try:
            # 기본적인 텐서-이미지 변환
            if hasattr(tensor, 'cpu'):
                tensor = tensor.cpu().numpy()
            
            if isinstance(tensor, np.ndarray):
                # 정규화 (0-1 → 0-255)
                if tensor.max() <= 1.0:
                    tensor = (tensor * 255).astype(np.uint8)
                
                # 차원 조정
                if tensor.ndim == 4:  # (B, C, H, W)
                    tensor = tensor[0]
                if tensor.ndim == 3 and tensor.shape[0] in [1, 3]:  # (C, H, W)
                    tensor = tensor.transpose(1, 2, 0)
                if tensor.ndim == 3 and tensor.shape[2] == 1:  # (H, W, 1)
                    tensor = tensor.squeeze(2)
                
                return Image.fromarray(tensor)
            
            # 폴백: 기본 이미지 생성
            return Image.new('RGB', (512, 512), (128, 128, 128))
            
        except Exception as e:
            self.logger.error(f"❌ 텐서-이미지 변환 실패: {e}")
            return Image.new('RGB', (512, 512), (128, 128, 128))
    
    def image_to_tensor(self, image: Union[Image.Image, np.ndarray], **kwargs) -> Any:
        """이미지를 텐서로 변환"""
        try:
            # PIL Image → NumPy
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            # 정규화 (0-255 → 0-1)
            if image_array.max() > 1.0:
                image_array = image_array.astype(np.float32) / 255.0
            
            # 차원 조정 (H, W, C) → (C, H, W)
            if image_array.ndim == 3:
                image_array = image_array.transpose(2, 0, 1)
            elif image_array.ndim == 2:
                image_array = image_array[np.newaxis, :, :]
            
            # 배치 차원 추가 (C, H, W) → (1, C, H, W)
            image_array = image_array[np.newaxis, :, :, :]
            
            # PyTorch 텐서로 변환 (사용 가능한 경우)
            try:
                import torch
                return torch.from_numpy(image_array)
            except ImportError:
                return image_array
                
        except Exception as e:
            self.logger.error(f"❌ 이미지-텐서 변환 실패: {e}")
            # 폴백: 기본 텐서
            try:
                import torch
                return torch.zeros(1, 3, 512, 512)
            except ImportError:
                return np.zeros((1, 3, 512, 512), dtype=np.float32)
    
    def _basic_image_conversion(self, image: Any, target_format: str, **kwargs) -> Any:
        """기본 이미지 변환 (폴백)"""
        try:
            if target_format == "tensor":
                return self.image_to_tensor(image, **kwargs)
            elif target_format == "image":
                if isinstance(image, Image.Image):
                    return image
                else:
                    return self.tensor_to_image(image, **kwargs)
            elif target_format == "numpy":
                if isinstance(image, Image.Image):
                    return np.array(image)
                elif hasattr(image, 'numpy'):
                    return image.numpy()
                else:
                    return image
            else:
                return image
        except Exception as e:
            self.logger.error(f"❌ 기본 이미지 변환 실패: {e}")
            return image
    
    def _basic_image_preprocessing(self, image: Any, size: Tuple[int, int], **kwargs) -> Any:
        """기본 이미지 전처리 (폴백)"""
        try:
            # PIL Image로 변환
            if not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                else:
                    # 텐서인 경우
                    image = self.tensor_to_image(image)
            
            # 리사이즈
            image = image.resize(size, Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
            
            # 요청된 형식으로 변환
            target_format = kwargs.get('output_format', 'image')
            if target_format == 'tensor':
                return self.image_to_tensor(image)
            elif target_format == 'numpy':
                return np.array(image)
            else:
                return image
                
        except Exception as e:
            self.logger.error(f"❌ 기본 이미지 전처리 실패: {e}")
            return image