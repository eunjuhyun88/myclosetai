"""
Step 공통 기능 Mixin
"""
import logging
import time
import gc
import traceback
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path


class StepMixin:
    """Step 공통 기능 Mixin"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._performance_metrics = {}
        self._error_count = 0
        
    def log_performance(self, operation: str, duration: float, **kwargs):
        """성능 로깅"""
        self._performance_metrics[operation] = {
            'duration': duration,
            'timestamp': time.time(),
            **kwargs
        }
        self.logger.info(f"📊 {operation}: {duration:.2f}s")
    
    def log_error(self, error: Exception, context: str = ""):
        """에러 로깅"""
        self._error_count += 1
        error_msg = f"❌ {context}: {str(error)}" if context else f"❌ {str(error)}"
        self.logger.error(error_msg)
        self.logger.debug(f"🔍 상세 에러: {traceback.format_exc()}")
    
    def validate_file_path(self, file_path: str) -> bool:
        """파일 경로 검증"""
        try:
            path = Path(file_path)
            return path.exists() and path.is_file()
        except Exception:
            return False
    
    def safe_model_load(self, model_loader_func, *args, **kwargs) -> Optional[Any]:
        """안전한 모델 로딩"""
        try:
            start_time = time.time()
            model = model_loader_func(*args, **kwargs)
            duration = time.time() - start_time
            self.log_performance("model_loading", duration, model_type=type(model).__name__)
            return model
        except Exception as e:
            self.log_error(e, "모델 로딩 실패")
            return None
    
    def cleanup_memory(self):
        """메모리 정리"""
        try:
            gc.collect()
            self.logger.debug("🧹 메모리 정리 완료")
        except Exception as e:
            self.log_error(e, "메모리 정리 실패")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 반환"""
        return {
            'metrics': self._performance_metrics.copy(),
            'error_count': self._error_count,
            'total_operations': len(self._performance_metrics)
        }
    
    def validate_config(self, required_keys: List[str], config: Dict[str, Any]) -> bool:
        """설정 검증"""
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            self.logger.error(f"❌ 누락된 설정 키: {missing_keys}")
            return False
        return True
    
    def safe_execute(self, func, *args, **kwargs) -> Tuple[bool, Any]:
        """안전한 함수 실행"""
        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            self.log_performance(func.__name__, duration)
            return True, result
        except Exception as e:
            self.log_error(e, f"{func.__name__} 실행 실패")
            return False, None
