#!/usr/bin/env python3
"""
MyCloset AI 백엔드 서버 실행 스크립트
"""
import sys
import os
import uvicorn
from pathlib import Path

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.environ['PYTHONPATH'] = str(project_root)

# 환경변수 직접 설정 (dotenv 의존성 제거)
os.environ.setdefault('APP_NAME', 'MyCloset AI Backend')
os.environ.setdefault('DEBUG', 'true')
os.environ.setdefault('DEVICE', 'mps')
os.environ.setdefault('HOST', '0.0.0.0')
os.environ.setdefault('PORT', '8000')

if __name__ == "__main__":
    # 필요한 디렉토리 생성
    dirs_to_create = [
        "static/uploads",
        "static/results", 
        "static/temp",
        "logs"
    ]
    
    for dir_path in dirs_to_create:
        (project_root / dir_path).mkdir(parents=True, exist_ok=True)
        # .gitkeep 파일 생성
        gitkeep_file = project_root / dir_path / ".gitkeep"
        if not gitkeep_file.exists():
            gitkeep_file.touch()
    
    print("🚀 MyCloset AI 백엔드 서버 시작...")
    print(f"📁 프로젝트 루트: {project_root}")
    print(f"🌐 서버 주소: http://localhost:8000")
    print(f"📖 API 문서: http://localhost:8000/docs")
    print(f"🔧 디바이스: {os.getenv('DEVICE', 'cpu')}")
    
    try:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 서버를 종료합니다...")
    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")
        sys.exit(1)
