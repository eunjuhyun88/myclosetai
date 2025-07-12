# backend/run_server.py
#!/usr/bin/env python3
"""
MyCloset AI Backend Server
M3 Max 최적화 실행 스크립트
"""

import sys
import os
import uvicorn
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/server.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

def check_environment():
    """환경 체크"""
    logger.info("🔍 환경 체크 시작...")
    
    # Python 버전 체크
    python_version = sys.version_info
    if python_version < (3, 9):
        logger.error(f"❌ Python 3.9+ 필요. 현재: {python_version}")
        sys.exit(1)
    
    logger.info(f"✅ Python {python_version.major}.{python_version.minor}")
    
    # 필수 디렉토리 생성
    directories = [
        'static/uploads',
        'static/results', 
        'logs',
        'ai_models/checkpoints',
        'ai_models/temp'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    logger.info("✅ 디렉토리 구조 확인 완료")
    
    # GPU 체크
    try:
        import torch
        if torch.backends.mps.is_available():
            logger.info("✅ Apple M3 Max GPU (Metal) 사용 가능")
        elif torch.cuda.is_available():
            logger.info("✅ NVIDIA GPU (CUDA) 사용 가능")
        else:
            logger.info("ℹ️ CPU 모드로 실행")
    except ImportError:
        logger.warning("⚠️ PyTorch가 설치되지 않았습니다")

def main():
    """메인 실행 함수"""
    
    # 환경 체크
    check_environment()
    
    # 설정값 로드
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8000))
    debug = os.getenv('DEBUG', 'true').lower() == 'true'
    
    logger.info("🚀 MyCloset AI Backend 서버 시작...")
    logger.info(f"📍 서버 주소: http://{host}:{port}")
    logger.info(f"🔧 디버그 모드: {debug}")
    logger.info(f"📚 API 문서: http://{host}:{port}/docs")
    logger.info(f"❤️ 헬스체크: http://{host}:{port}/health")
    
    try:
        # 서버 실행
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=debug,
            log_level="info" if debug else "warning",
            access_log=debug,
            reload_dirs=["app"] if debug else None,
            reload_excludes=["static", "logs", "ai_models"] if debug else None
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 서버 종료 요청")
    except Exception as e:
        logger.error(f"❌ 서버 실행 오류: {e}")
        sys.exit(1)
    finally:
        logger.info("👋 MyCloset AI Backend 서버 종료")

if __name__ == "__main__":
    main()