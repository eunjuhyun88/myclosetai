# backend/app/core/logging_config.py
"""
MyCloset AI Backend - 로깅 설정 (수정된 버전)
구조화된 로깅 및 M3 Max 성능 모니터링
LOGS_DIR 에러 해결
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    import structlog
    from structlog.stdlib import LoggerFactory
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

def get_settings_safe():
    """안전한 settings 가져오기"""
    try:
        from app.core.config import settings
        return settings
    except Exception:
        # 폴백 설정
        class FallbackSettings:
            LOG_LEVEL = "INFO"
            LOGS_DIR = Path("logs")
            PROJECT_ROOT = Path(".")
        
        return FallbackSettings()

def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_json: bool = False
) -> None:
    """로깅 시스템 설정 - 안전한 버전"""
    
    try:
        settings = get_settings_safe()
        
        # 로그 레벨 설정
        level = log_level or getattr(settings, 'LOG_LEVEL', 'INFO')
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        
        # 로그 디렉토리 생성 - 안전한 방식
        try:
            log_dir = getattr(settings, 'LOGS_DIR', None)
            if log_dir is None:
                # LOGS_DIR이 없으면 생성
                project_root = getattr(settings, 'PROJECT_ROOT', '.')
                log_dir = Path(project_root) / "logs"
            elif isinstance(log_dir, str):
                log_dir = Path(log_dir)
            
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            # 폴백: 현재 디렉토리에 logs 폴더
            log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            print(f"⚠️ 로그 디렉토리 설정 실패, 기본값 사용: {e}")
        
        # 로그 파일 경로 설정
        if log_file:
            log_file_path = log_dir / log_file
        else:
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file_path = log_dir / f"mycloset-ai-{timestamp}.log"
        
        # 기본 로깅 설정
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[]
        )
        
        # 루트 로거 설정
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # 콘솔 핸들러 (컬러 출력)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        if enable_json:
            console_formatter = JsonFormatter()
        else:
            console_formatter = ColoredFormatter(
                '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
                datefmt='%H:%M:%S'
            )
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # 파일 핸들러 (로테이션) - 안전한 처리
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(numeric_level)
            
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)-20s | %(process)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            
            # 에러 전용 파일 핸들러
            error_file_path = log_dir / f"error-{datetime.now().strftime('%Y%m%d')}.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_file_path,
                maxBytes=5 * 1024 * 1024,  # 5MB
                backupCount=3,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(file_formatter)
            root_logger.addHandler(error_handler)
            
        except Exception as e:
            print(f"⚠️ 파일 핸들러 설정 실패: {e}")
        
        # Structlog 설정 (고급 로깅) - 선택적
        if STRUCTLOG_AVAILABLE:
            try:
                structlog.configure(
                    processors=[
                        structlog.stdlib.filter_by_level,
                        structlog.stdlib.add_logger_name,
                        structlog.stdlib.add_log_level,
                        structlog.stdlib.PositionalArgumentsFormatter(),
                        structlog.processors.TimeStamper(fmt="ISO"),
                        structlog.processors.StackInfoRenderer(),
                        structlog.processors.format_exc_info,
                        structlog.processors.UnicodeDecoder(),
                        structlog.processors.JSONRenderer() if enable_json else structlog.dev.ConsoleRenderer(),
                    ],
                    context_class=dict,
                    logger_factory=LoggerFactory(),
                    wrapper_class=structlog.stdlib.BoundLogger,
                    cache_logger_on_first_use=True,
                )
            except Exception as e:
                print(f"⚠️ Structlog 설정 실패: {e}")
        
        # 외부 라이브러리 로그 레벨 조정
        logging.getLogger('uvicorn').setLevel(logging.INFO)
        logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
        logging.getLogger('torch').setLevel(logging.WARNING)
        logging.getLogger('transformers').setLevel(logging.WARNING)
        logging.getLogger('diffusers').setLevel(logging.WARNING)
        
        # 초기 로그 메시지
        logger = logging.getLogger("mycloset.logging")
        logger.info(f"🔧 로깅 시스템 초기화 완료")
        logger.info(f"📝 로그 레벨: {level}")
        logger.info(f"📁 로그 파일: {log_file_path}")
        
    except Exception as e:
        # 최종 폴백: 기본 로깅만
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        print(f"❌ 로깅 설정 실패, 기본 설정 사용: {e}")


class ColoredFormatter(logging.Formatter):
    """컬러 로그 포매터 - 안전한 버전"""
    
    # ANSI 컬러 코드
    COLORS = {
        'DEBUG': '\033[36m',     # 청록색
        'INFO': '\033[32m',      # 초록색
        'WARNING': '\033[33m',   # 노란색
        'ERROR': '\033[31m',     # 빨간색
        'CRITICAL': '\033[35m',  # 자주색
    }
    RESET = '\033[0m'
    
    def format(self, record):
        try:
            # 로그 레벨에 따른 컬러 적용
            if record.levelname in self.COLORS:
                record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            
            # 로거 이름 단축
            if len(record.name) > 20:
                record.name = record.name.split('.')[-1][:20]
            
            return super().format(record)
        except Exception as e:
            # 포맷팅 실패 시 기본 포맷 사용
            return f"{record.levelname}: {record.getMessage()}"


class JsonFormatter(logging.Formatter):
    """JSON 로그 포매터 - 안전한 버전"""
    
    def format(self, record):
        try:
            import json
            
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': getattr(record, 'module', 'unknown'),
                'function': getattr(record, 'funcName', 'unknown'),
                'line': getattr(record, 'lineno', 0),
            }
            
            # 예외 정보 추가
            if record.exc_info:
                try:
                    log_entry['exception'] = self.formatException(record.exc_info)
                except:
                    log_entry['exception'] = "Exception formatting failed"
            
            # 추가 필드
            if hasattr(record, 'user_id'):
                log_entry['user_id'] = record.user_id
            
            if hasattr(record, 'request_id'):
                log_entry['request_id'] = record.request_id
            
            return json.dumps(log_entry, ensure_ascii=False)
        except Exception as e:
            # JSON 포맷팅 실패 시 기본 포맷
            return f"{record.levelname}: {record.getMessage()}"


class PerformanceLogger:
    """성능 모니터링 로거 - 안전한 버전"""
    
    def __init__(self, name: str = "performance"):
        self.logger = logging.getLogger(f"mycloset.{name}")
        self.start_time = None
        self.operation = None
    
    def start_timer(self, operation: str):
        """타이머 시작"""
        try:
            import time
            self.start_time = time.time()
            self.operation = operation
            self.logger.info(f"⏱️ {operation} 시작")
        except Exception as e:
            self.logger.warning(f"타이머 시작 실패: {e}")
    
    def end_timer(self, success: bool = True, **kwargs):
        """타이머 종료"""
        try:
            if self.start_time is None:
                return 0
            
            import time
            duration = time.time() - self.start_time
            
            status = "✅ 완료" if success else "❌ 실패"
            self.logger.info(f"{status} {self.operation}: {duration:.3f}s", extra=kwargs)
            
            self.start_time = None
            return duration
        except Exception as e:
            self.logger.warning(f"타이머 종료 실패: {e}")
            return 0


class GPUMonitorLogger:
    """GPU 모니터링 로거 - 안전한 버전"""
    
    def __init__(self):
        self.logger = logging.getLogger("mycloset.gpu")
        self.enabled = True
    
    def log_gpu_usage(self, operation: str = "GPU 연산"):
        """GPU 사용량 로깅"""
        if not self.enabled:
            return
        
        try:
            import torch
            
            gpu_info = {
                "operation": operation,
            }
            
            # 시스템 메모리 (선택적)
            try:
                import psutil
                memory = psutil.virtual_memory()
                gpu_info.update({
                    "system_memory_percent": memory.percent,
                    "system_memory_available_gb": round(memory.available / (1024**3), 1)
                })
            except ImportError:
                pass
            
            # GPU별 정보
            if torch.backends.mps.is_available():
                gpu_info.update({
                    "device": "mps",
                    "backend": "Metal Performance Shaders"
                })
            elif torch.cuda.is_available():
                try:
                    gpu_info.update({
                        "device": "cuda",
                        "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / (1024*1024), 1),
                        "memory_reserved_mb": round(torch.cuda.memory_reserved(0) / (1024*1024), 1)
                    })
                except:
                    gpu_info["device"] = "cuda"
            else:
                gpu_info["device"] = "cpu"
            
            self.logger.debug(f"🖥️ GPU 사용량: {operation}", extra=gpu_info)
            
        except Exception as e:
            self.logger.warning(f"GPU 모니터링 실패: {e}")
    
    def enable(self):
        """모니터링 활성화"""
        self.enabled = True
        self.logger.info("🔍 GPU 모니터링 활성화")
    
    def disable(self):
        """모니터링 비활성화"""
        self.enabled = False
        self.logger.info("🔇 GPU 모니터링 비활성화")


# 전역 인스턴스 - 안전한 생성
try:
    performance_logger = PerformanceLogger()
    gpu_monitor = GPUMonitorLogger()
except Exception as e:
    print(f"⚠️ 로거 인스턴스 생성 실패: {e}")
    performance_logger = None
    gpu_monitor = None

def get_logger(name: str) -> logging.Logger:
    """로거 팩토리 함수"""
    try:
        return logging.getLogger(f"mycloset.{name}")
    except Exception:
        return logging.getLogger(name)

def log_request_info(request_id: str, method: str, path: str, **kwargs):
    """요청 정보 로깅"""
    try:
        logger = get_logger("api")
        logger.info(
            f"📥 {method} {path}",
            extra={"request_id": request_id, **kwargs}
        )
    except Exception as e:
        print(f"요청 로깅 실패: {e}")

def log_model_operation(model_name: str, operation: str, duration: float, success: bool = True):
    """모델 연산 로깅"""
    try:
        logger = get_logger("model")
        status = "✅" if success else "❌"
        logger.info(f"{status} {model_name}: {operation} ({duration:.3f}s)")
    except Exception as e:
        print(f"모델 로깅 실패: {e}")

# 자동 초기화 제거 - 명시적 호출만 허용
# 모듈 임포트 시 자동으로 setup_logging()을 호출하지 않음

def safe_setup_logging():
    """안전한 로깅 설정 - 에러가 발생해도 프로그램이 중단되지 않음"""
    try:
        setup_logging()
        return True
    except Exception as e:
        print(f"⚠️ 로깅 설정 실패: {e}")
        # 최소한의 로깅이라도 설정
        logging.basicConfig(level=logging.INFO)
        return False

# 모듈 레벨에서 자동 호출하지 않음 - 필요한 곳에서 명시적으로 호출