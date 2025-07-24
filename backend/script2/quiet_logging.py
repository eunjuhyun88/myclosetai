import logging
import sys

def setup_quiet_logging():
    """로그를 최소한으로 줄이는 설정"""
    
    # 모든 기존 핸들러 제거
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 로그 레벨을 WARNING 이상으로 설정
    logging.root.setLevel(logging.WARNING)
    
    # 콘솔 핸들러만 간단하게
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    
    # 간단한 포맷
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    
    logging.root.addHandler(console_handler)
    
    # 특정 로거들 완전 억제
    noisy_loggers = [
        'uvicorn.access', 'uvicorn.error', 'app.ai_pipeline',
        'pipeline', 'app.core', 'app.services', 'app.api',
        'torch', 'transformers', 'PIL', 'matplotlib',
        'urllib3', 'requests', 'diffusers'
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)
        logging.getLogger(logger_name).disabled = True
    
    print("✅ 조용한 로그 모드 활성화")
