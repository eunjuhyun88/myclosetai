"""
MyCloset AI - 개선된 로깅 시스템 (깔끔하고 핵심적)
backend/app/core/logging_config.py

✅ 깔끔한 콘솔 출력
✅ 핵심 문제만 표시
✅ 개발/프로덕션 모드 분리
✅ 스마트 로그 레벨
✅ 중요한 에러만 강조
"""

import os
import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

# ===============================================================
# 🔧 로깅 모드 정의
# ===============================================================

class LogMode(Enum):
    """로깅 모드"""
    MINIMAL = "minimal"      # 최소한의 로그만 (에러, 중요 정보)
    CLEAN = "clean"          # 깔끔한 로그 (기본값)
    DETAILED = "detailed"    # 상세한 로그 (개발용)
    DEBUG = "debug"          # 모든 로그 (디버깅용)

class LoggingConfig:
    """개선된 로깅 설정 관리 클래스"""
    
    def __init__(self):
        self.log_dir = Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 로깅 모드 결정
        self.mode = self._get_log_mode()
        self.log_level = self._get_log_level()
        
        # 환경별 설정
        self.is_production = os.getenv('ENVIRONMENT', 'development').lower() == 'production'
        self.show_startup_info = not self.is_production
        
        # 로그 파일 경로
        date_str = datetime.now().strftime('%Y%m%d')
        self.log_file = self.log_dir / f"mycloset-{date_str}.log"
        self.error_log_file = self.log_dir / f"error-{date_str}.log"
        
        # 포맷 설정
        self.file_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        self.console_format = self._get_console_format()
        
        self.is_configured = False
    
    def _get_log_mode(self) -> LogMode:
        """로깅 모드 결정"""
        mode_str = os.getenv('LOG_MODE', 'clean').lower()
        try:
            return LogMode(mode_str)
        except ValueError:
            return LogMode.CLEAN
    
    def _get_log_level(self) -> str:
        """로그 레벨 결정 (모드에 따라)"""
        env_level = os.getenv('LOG_LEVEL', '').upper()
        
        # 환경변수가 설정된 경우 우선 적용
        if env_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            return env_level
        
        # 모드별 기본 레벨
        mode_levels = {
            LogMode.MINIMAL: 'ERROR',
            LogMode.CLEAN: 'WARNING', 
            LogMode.DETAILED: 'INFO',
            LogMode.DEBUG: 'DEBUG'
        }
        
        return mode_levels.get(self.mode, 'WARNING')
    
    def _get_console_format(self) -> str:
        """콘솔 포맷 설정 (모드별)"""
        if self.mode == LogMode.MINIMAL:
            return "%(levelname)s: %(message)s"
        elif self.mode == LogMode.CLEAN:
            return "%(asctime)s | %(levelname)s | %(message)s"
        else:
            return "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    
    def setup_logging(self) -> bool:
        """로깅 시스템 설정"""
        try:
            # 기존 핸들러 제거
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
            
            # 루트 로거 설정
            root_logger.setLevel(getattr(logging, self.log_level))
            
            # 파일 핸들러 설정 (항상 유지)
            self._setup_file_handlers(root_logger)
            
            # 콘솔 핸들러 설정
            self._setup_console_handler(root_logger)
            
            # 외부 라이브러리 로거 제어
            self._setup_external_loggers()
            
            # MyCloset AI 로거 설정
            self._setup_app_loggers()
            
            self.is_configured = True
            
            # 시작 메시지 (깔끔하게)
            if self.show_startup_info:
                self._log_startup_info()
            
            return True
            
        except Exception as e:
            print(f"❌ 로깅 설정 실패: {e}")
            return False
    
    def _setup_file_handlers(self, root_logger: logging.Logger):
        """파일 핸들러 설정"""
        # 일반 로그 파일
        file_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=3,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(self.file_format))
        root_logger.addHandler(file_handler)
        
        # 에러 로그 파일
        error_handler = logging.handlers.RotatingFileHandler(
            filename=self.error_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=2,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(self.file_format))
        root_logger.addHandler(error_handler)
    
    def _setup_console_handler(self, root_logger: logging.Logger):
        """콘솔 핸들러 설정"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.log_level))
        
        # 컬러 포맷터 적용
        console_formatter = CleanColoredFormatter(self.console_format)
        console_handler.setFormatter(console_formatter)
        
        root_logger.addHandler(console_handler)
    
    def _setup_external_loggers(self):
        """외부 라이브러리 로거 제어"""
        # 시끄러운 라이브러리들 조용하게 만들기
        noisy_loggers = [
            'urllib3',
            'requests',
            'PIL',
            'matplotlib',
            'tensorflow',
            'torch',
            'transformers',
            'diffusers',
            'timm',
            'coremltools',
            'watchfiles',
            'multipart'
        ]
        
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        
        # FastAPI/Uvicorn 로거 설정
        if self.mode in [LogMode.MINIMAL, LogMode.CLEAN]:
            logging.getLogger("fastapi").setLevel(logging.WARNING)
            logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
            logging.getLogger("uvicorn.error").setLevel(logging.INFO)
        else:
            logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    
    def _setup_app_loggers(self):
        """MyCloset AI 앱 로거 설정"""
        # 앱 로거는 모드에 따라 다르게 설정
        app_logger = logging.getLogger("app")
        
        if self.mode == LogMode.MINIMAL:
            app_logger.setLevel(logging.ERROR)
        elif self.mode == LogMode.CLEAN:
            app_logger.setLevel(logging.WARNING)
        else:
            app_logger.setLevel(logging.INFO)
    
    def _log_startup_info(self):
        """시작 정보 로깅 (깔끔하게)"""
        logger = logging.getLogger("mycloset.startup")
        
        if self.mode == LogMode.MINIMAL:
            logger.info("🚀 MyCloset AI 시작")
        elif self.mode == LogMode.CLEAN:
            logger.info("🚀 MyCloset AI 시작 중...")
            logger.info(f"📋 로그 모드: {self.mode.value}")
        else:
            logger.info("🚀 MyCloset AI 시작 중...")
            logger.info(f"📋 로그 모드: {self.mode.value}")
            logger.info(f"📝 로그 레벨: {self.log_level}")
            logger.info(f"📁 로그 파일: {self.log_file.name}")

# ===============================================================
# 🎨 개선된 컬러 포맷터
# ===============================================================

class CleanColoredFormatter(logging.Formatter):
    """깔끔한 컬러 로그 포맷터"""
    
    # 심플한 컬러 스키마
    COLORS = {
        'DEBUG': '\033[90m',    # 회색
        'INFO': '\033[36m',     # 청록색
        'WARNING': '\033[33m',  # 노란색
        'ERROR': '\033[91m',    # 밝은 빨간색
        'CRITICAL': '\033[95m', # 밝은 자주색
        'RESET': '\033[0m'
    }
    
    # 레벨별 이모지
    EMOJIS = {
        'DEBUG': '🔍',
        'INFO': '✅',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨'
    }
    
    def format(self, record):
        # 시간 포맷 간소화
        if hasattr(record, 'asctime'):
            # 이미 asctime이 있으면 시간만 추출
            time_part = record.asctime.split()[1] if ' ' in record.asctime else record.asctime
        else:
            time_part = datetime.now().strftime('%H:%M:%S')
        
        # 컬러와 이모지 적용
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        emoji = self.EMOJIS.get(record.levelname, '')
        
        # 원본 데이터 백업
        original_levelname = record.levelname
        original_asctime = getattr(record, 'asctime', None)
        
        # 수정된 데이터 적용
        record.levelname = f"{level_color}{emoji} {record.levelname}{reset_color}"
        record.asctime = time_part
        
        # 포맷 적용
        formatted = super().format(record)
        
        # 원본 데이터 복원
        record.levelname = original_levelname
        if original_asctime:
            record.asctime = original_asctime
        
        return formatted

# ===============================================================
# 🔧 스마트 로거 클래스들
# ===============================================================

class SmartLogger:
    """스마트 로거 - 중요한 것만 로깅"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.mode = LogMode(os.getenv('LOG_MODE', 'clean'))
    
    def startup(self, message: str, details: Optional[Dict] = None):
        """시작 관련 로그"""
        if self.mode == LogMode.MINIMAL:
            return
        
        if details and self.mode in [LogMode.DETAILED, LogMode.DEBUG]:
            detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
            self.logger.info(f"{message} ({detail_str})")
        else:
            self.logger.info(message)
    
    def success(self, message: str):
        """성공 메시지"""
        if self.mode != LogMode.MINIMAL:
            self.logger.info(message)
    
    def problem(self, message: str, suggestion: str = None):
        """문제 상황 (해결책 포함)"""
        self.logger.warning(message)
        if suggestion and self.mode != LogMode.MINIMAL:
            self.logger.info(f"💡 해결책: {suggestion}")
    
    def critical_error(self, message: str, action: str = None):
        """심각한 에러"""
        self.logger.error(message)
        if action:
            self.logger.error(f"🔧 필요한 조치: {action}")
    
    def progress(self, step: str, total: int = None, current: int = None):
        """진행 상황"""
        if self.mode in [LogMode.DETAILED, LogMode.DEBUG]:
            if total and current:
                self.logger.info(f"📊 {step} ({current}/{total})")
            else:
                self.logger.info(f"📊 {step}")

# ===============================================================
# 🔧 전역 설정 및 유틸리티
# ===============================================================

# 전역 로깅 설정 인스턴스
_logging_config = LoggingConfig()

def setup_logging() -> bool:
    """로깅 시스템 설정"""
    global _logging_config
    
    if _logging_config.is_configured:
        return True
    
    return _logging_config.setup_logging()

def get_smart_logger(name: str) -> SmartLogger:
    """스마트 로거 반환"""
    return SmartLogger(name)

def get_logger(name: str) -> logging.Logger:
    """일반 로거 반환"""
    return logging.getLogger(name)

def set_log_mode(mode: str):
    """런타임에 로그 모드 변경"""
    os.environ['LOG_MODE'] = mode
    # 재설정 필요시 여기에 추가

# ===============================================================
# 🔧 컨텍스트 매니저 (임시 로그 레벨 변경용)
# ===============================================================

from contextlib import contextmanager

@contextmanager
def quiet_logging():
    """임시로 조용한 로깅"""
    original_level = logging.getLogger().level
    try:
        logging.getLogger().setLevel(logging.ERROR)
        yield
    finally:
        logging.getLogger().setLevel(original_level)

@contextmanager
def verbose_logging():
    """임시로 상세한 로깅"""
    original_level = logging.getLogger().level
    try:
        logging.getLogger().setLevel(logging.DEBUG)
        yield
    finally:
        logging.getLogger().setLevel(original_level)

# ===============================================================
# 🔧 Export 리스트
# ===============================================================

__all__ = [
    'setup_logging',
    'get_smart_logger', 
    'get_logger',
    'set_log_mode',
    'LogMode',
    'SmartLogger',
    'quiet_logging',
    'verbose_logging'
]

# ===============================================================
# 🔧 자동 초기화
# ===============================================================

if not _logging_config.is_configured:
    setup_logging()