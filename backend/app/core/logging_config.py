"""
MyCloset AI - 완전한 로깅 설정 시스템
backend/app/core/logging_config.py

✅ 완전한 로깅 설정 시스템
✅ 파일 및 콘솔 로깅 지원
✅ 레벨별 로그 분리
✅ 로그 회전 및 압축
✅ 성능 최적화
"""

import os
import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json

# ===============================================================
# 🔧 로깅 설정 클래스
# ===============================================================

class LoggingConfig:
    """로깅 설정 관리 클래스"""
    
    def __init__(self):
        self.log_dir = Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 로그 파일 경로
        self.log_file = self.log_dir / f"mycloset-ai-{datetime.now().strftime('%Y%m%d')}.log"
        self.error_log_file = self.log_dir / f"error-{datetime.now().strftime('%Y%m%d')}.log"
        
        # 로그 레벨 설정
        self.log_level = self._get_log_level()
        
        # 로그 포맷 설정
        self.log_format = self._get_log_format()
        self.console_format = self._get_console_format()
        
        # 로깅 설정 완료 플래그
        self.is_configured = False
    
    def _get_log_level(self) -> str:
        """로그 레벨 결정"""
        level = os.getenv('LOG_LEVEL', 'INFO').upper()
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        return level if level in valid_levels else 'INFO'
    
    def _get_log_format(self) -> str:
        """파일 로그 포맷 설정"""
        return (
            "%(asctime)s | %(levelname)s | %(name)s | %(process)d | %(message)s"
        )
    
    def _get_console_format(self) -> str:
        """콘솔 로그 포맷 설정"""
        return (
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
    
    def setup_logging(self) -> bool:
        """로깅 시스템 설정"""
        try:
            # 기존 핸들러 제거
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
            
            # 루트 로거 설정
            root_logger.setLevel(getattr(logging, self.log_level))
            
            # 파일 핸들러 설정
            self._setup_file_handlers(root_logger)
            
            # 콘솔 핸들러 설정
            self._setup_console_handler(root_logger)
            
            # 특정 로거 설정
            self._setup_specific_loggers()
            
            self.is_configured = True
            
            # 로깅 설정 완료 메시지
            logger = logging.getLogger("mycloset.logging")
            logger.info("🔧 로깅 시스템 초기화 완료")
            logger.info(f"📝 로그 레벨: {self.log_level}")
            logger.info(f"📁 로그 파일: {self.log_file}")
            logger.info(f"❌ 에러 로그: {self.error_log_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ 로깅 설정 실패: {e}")
            return False
    
    def _setup_file_handlers(self, root_logger: logging.Logger):
        """파일 핸들러 설정"""
        # 일반 로그 파일 핸들러
        file_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(self.log_format))
        root_logger.addHandler(file_handler)
        
        # 에러 로그 파일 핸들러
        error_handler = logging.handlers.RotatingFileHandler(
            filename=self.error_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(self.log_format))
        root_logger.addHandler(error_handler)
    
    def _setup_console_handler(self, root_logger: logging.Logger):
        """콘솔 핸들러 설정"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.log_level))
        
        # 컬러 포맷터 설정
        console_formatter = ColoredFormatter(self.console_format)
        console_handler.setFormatter(console_formatter)
        
        root_logger.addHandler(console_handler)
    
    def _setup_specific_loggers(self):
        """특정 로거들 설정"""
        # FastAPI 관련 로거 설정
        fastapi_logger = logging.getLogger("fastapi")
        fastapi_logger.setLevel(logging.WARNING)
        
        # Uvicorn 관련 로거 설정
        uvicorn_logger = logging.getLogger("uvicorn")
        uvicorn_logger.setLevel(logging.INFO)
        
        # PyTorch 관련 로거 설정
        torch_logger = logging.getLogger("torch")
        torch_logger.setLevel(logging.WARNING)
        
        # MyCloset AI 관련 로거 설정
        mycloset_logger = logging.getLogger("mycloset")
        mycloset_logger.setLevel(logging.INFO)
    
    def get_logger(self, name: str) -> logging.Logger:
        """로거 반환"""
        return logging.getLogger(name)
    
    def log_system_info(self):
        """시스템 정보 로깅"""
        logger = logging.getLogger("mycloset.system")
        
        logger.info("🖥️ 시스템 정보:")
        logger.info(f"  - Python: {sys.version.split()[0]}")
        logger.info(f"  - Platform: {sys.platform}")
        logger.info(f"  - 프로세스 ID: {os.getpid()}")
        logger.info(f"  - 작업 디렉토리: {os.getcwd()}")
        logger.info(f"  - 로그 디렉토리: {self.log_dir.absolute()}")

# ===============================================================
# 🎨 컬러 포맷터
# ===============================================================

class ColoredFormatter(logging.Formatter):
    """컬러 로그 포맷터"""
    
    # ANSI 컬러 코드
    COLORS = {
        'DEBUG': '\033[36m',    # 청록색
        'INFO': '\033[32m',     # 녹색
        'WARNING': '\033[33m',  # 노란색
        'ERROR': '\033[31m',    # 빨간색
        'CRITICAL': '\033[35m', # 자주색
        'RESET': '\033[0m'      # 리셋
    }
    
    def format(self, record):
        # 레벨별 컬러 적용
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # 원본 레벨명 저장
        original_levelname = record.levelname
        
        # 컬러 적용
        record.levelname = f"{level_color}{record.levelname}{reset_color}"
        
        # 포맷 적용
        formatted = super().format(record)
        
        # 원본 레벨명 복원
        record.levelname = original_levelname
        
        return formatted

# ===============================================================
# 🔧 전역 로깅 설정
# ===============================================================

# 전역 로깅 설정 인스턴스
_logging_config = LoggingConfig()

def setup_logging() -> bool:
    """로깅 시스템 설정 (전역 함수)"""
    global _logging_config
    
    if _logging_config.is_configured:
        return True
    
    success = _logging_config.setup_logging()
    
    if success:
        # 시스템 정보 로깅
        _logging_config.log_system_info()
    
    return success

def get_logger(name: str) -> logging.Logger:
    """로거 반환 (전역 함수)"""
    return _logging_config.get_logger(name)

def get_logging_config() -> LoggingConfig:
    """로깅 설정 인스턴스 반환"""
    return _logging_config

# ===============================================================
# 🔧 성능 로깅 유틸리티
# ===============================================================

class PerformanceLogger:
    """성능 로깅 유틸리티"""
    
    def __init__(self, logger_name: str = "mycloset.performance"):
        self.logger = logging.getLogger(logger_name)
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """타이머 시작"""
        import time
        self.start_times[operation] = time.time()
        self.logger.info(f"⏱️ {operation} 시작")
    
    def end_timer(self, operation: str, details: Optional[Dict[str, Any]] = None):
        """타이머 종료"""
        import time
        
        if operation not in self.start_times:
            self.logger.warning(f"⚠️ {operation} 시작 시간을 찾을 수 없음")
            return
        
        elapsed = time.time() - self.start_times[operation]
        del self.start_times[operation]
        
        detail_str = ""
        if details:
            detail_str = f" ({', '.join(f'{k}={v}' for k, v in details.items())})"
        
        self.logger.info(f"✅ {operation} 완료: {elapsed:.3f}초{detail_str}")
    
    def log_memory_usage(self, operation: str):
        """메모리 사용량 로깅"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.logger.info(f"💾 {operation} 메모리 사용량: {memory_info.rss / 1024 / 1024:.1f}MB")
        except ImportError:
            self.logger.warning("psutil이 설치되지 않아 메모리 사용량을 측정할 수 없습니다.")
        except Exception as e:
            self.logger.warning(f"메모리 사용량 측정 실패: {e}")

# ===============================================================
# 🔧 GPU 로깅 유틸리티
# ===============================================================

class GPULogger:
    """GPU 로깅 유틸리티"""
    
    def __init__(self, logger_name: str = "mycloset.gpu"):
        self.logger = logging.getLogger(logger_name)
    
    def log_gpu_memory(self, device: str, operation: str):
        """GPU 메모리 로깅"""
        try:
            import torch
            
            if device == "mps":
                # MPS 메모리 정보 (제한적)
                try:
                    if hasattr(torch.mps, 'current_allocated_memory'):
                        allocated = torch.mps.current_allocated_memory()
                        self.logger.info(f"🍎 {operation} MPS 메모리: {allocated / 1024 / 1024:.1f}MB")
                    else:
                        self.logger.info(f"🍎 {operation} MPS 메모리 정보 제한됨")
                except:
                    self.logger.info(f"🍎 {operation} MPS 메모리 정보 없음")
            
            elif device == "cuda" and torch.cuda.is_available():
                # CUDA 메모리 정보
                allocated = torch.cuda.memory_allocated() / 1024 / 1024
                reserved = torch.cuda.memory_reserved() / 1024 / 1024
                
                self.logger.info(f"🚀 {operation} CUDA 메모리: {allocated:.1f}MB 할당됨, {reserved:.1f}MB 예약됨")
            
            else:
                self.logger.info(f"💻 {operation} CPU 모드")
        
        except Exception as e:
            self.logger.warning(f"GPU 메모리 로깅 실패: {e}")
    
    def log_gpu_utilization(self, device: str):
        """GPU 사용률 로깅"""
        try:
            if device == "cuda":
                import torch
                if torch.cuda.is_available():
                    gpu_util = torch.cuda.utilization()
                    self.logger.info(f"🚀 GPU 사용률: {gpu_util}%")
            else:
                self.logger.info(f"🖥️ {device} 사용률 정보 없음")
        
        except Exception as e:
            self.logger.warning(f"GPU 사용률 로깅 실패: {e}")

# ===============================================================
# 🔧 API 로깅 유틸리티
# ===============================================================

class APILogger:
    """API 로깅 유틸리티"""
    
    def __init__(self, logger_name: str = "mycloset.api"):
        self.logger = logging.getLogger(logger_name)
    
    def log_request(self, method: str, path: str, client_ip: str = None):
        """API 요청 로깅"""
        client_info = f" from {client_ip}" if client_ip else ""
        self.logger.info(f"📥 {method} {path}{client_info}")
    
    def log_response(self, method: str, path: str, status_code: int, duration: float):
        """API 응답 로깅"""
        status_emoji = "✅" if status_code < 400 else "❌"
        self.logger.info(f"📤 {method} {path} - {status_code} ({duration:.3f}s) {status_emoji}")
    
    def log_error(self, method: str, path: str, error: Exception):
        """API 에러 로깅"""
        self.logger.error(f"💥 {method} {path} - {type(error).__name__}: {str(error)}")

# ===============================================================
# 🔧 전역 유틸리티 인스턴스
# ===============================================================

# 전역 로깅 유틸리티 인스턴스들
performance_logger = PerformanceLogger()
gpu_logger = GPULogger()
api_logger = APILogger()

# ===============================================================
# 🔧 Export 리스트
# ===============================================================

__all__ = [
    # 주요 함수들
    'setup_logging',
    'get_logger',
    'get_logging_config',
    
    # 클래스들
    'LoggingConfig',
    'ColoredFormatter',
    'PerformanceLogger',
    'GPULogger',
    'APILogger',
    
    # 전역 인스턴스들
    'performance_logger',
    'gpu_logger',
    'api_logger'
]

# ===============================================================
# 🔧 자동 초기화 (모듈 로드 시)
# ===============================================================

# 모듈 로드 시 자동으로 로깅 설정
if not _logging_config.is_configured:
    setup_logging()