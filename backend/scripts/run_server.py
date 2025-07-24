#!/usr/bin/env python3
"""
🍎 MyCloset AI Backend - 서버 실행 스크립트
✅ M3 Max 최적화 설정
✅ 개발/프로덕션 환경 대응
✅ 자동 의존성 체크
✅ 상세 로깅
"""

import os
import sys
import time
import subprocess
import platform
from pathlib import Path

# 색상 출력을 위한 ANSI 코드
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(message: str, color: str = Colors.OKGREEN):
    """색상 메시지 출력"""
    print(f"{color}{message}{Colors.ENDC}")

def print_header(message: str):
    """헤더 메시지 출력"""
    print_colored(f"\n🚀 {message}", Colors.HEADER + Colors.BOLD)
    print_colored("=" * (len(message) + 3), Colors.HEADER)

def print_success(message: str):
    """성공 메시지 출력"""
    print_colored(f"✅ {message}", Colors.OKGREEN)

def print_info(message: str):
    """정보 메시지 출력"""
    print_colored(f"ℹ️  {message}", Colors.OKBLUE)

def print_warning(message: str):
    """경고 메시지 출력"""
    print_colored(f"⚠️  {message}", Colors.WARNING)

def print_error(message: str):
    """에러 메시지 출력"""
    print_colored(f"❌ {message}", Colors.FAIL)

def check_python_version():
    """Python 버전 확인"""
    version = sys.version_info
    required_major, required_minor = 3, 9
    
    if version.major < required_major or (version.major == required_major and version.minor < required_minor):
        print_error(f"Python {required_major}.{required_minor}+ 필요. 현재: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print_success(f"Python {version.major}.{version.minor}.{version.micro}")
    return True

def detect_system():
    """시스템 정보 감지"""
    system = platform.system()
    machine = platform.machine()
    
    print_info(f"운영체제: {system}")
    print_info(f"아키텍처: {machine}")
    
    # M3 Max 감지
    is_m3_max = False
    if system == "Darwin" and machine == "arm64":
        try:
            # macOS에서 CPU 정보 확인
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if "Apple M3 Max" in result.stdout:
                is_m3_max = True
                print_success("🍎 Apple M3 Max 감지됨!")
            else:
                print_info("🍎 Apple Silicon 감지됨 (M3 Max 아님)")
        except:
            print_warning("시스템 정보 확인 실패")
    
    return {
        "system": system,
        "machine": machine,
        "is_m3_max": is_m3_max
    }

def check_virtual_env():
    """가상환경 확인"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        venv_path = sys.prefix
        print_success(f"가상환경 활성화됨: {venv_path}")
        return True
    else:
        print_warning("가상환경이 활성화되지 않았습니다")
        return False

def check_dependencies():
    """필수 의존성 확인"""
    dependencies = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("torch", "PyTorch"),
        ("PIL", "Pillow"),
        ("psutil", "psutil")
    ]
    
    missing = []
    for module, name in dependencies:
        try:
            __import__(module)
            print_success(f"{name} 설치됨")
        except ImportError:
            print_error(f"{name} 설치되지 않음")
            missing.append(name)
    
    if missing:
        print_error(f"누락된 패키지: {', '.join(missing)}")
        print_info("설치 명령어: pip install -r requirements.txt")
        return False
    
    return True

def check_pytorch_mps():
    """PyTorch MPS 지원 확인"""
    try:
        import torch
        print_success(f"PyTorch {torch.__version__}")
        
        if torch.backends.mps.is_available():
            print_success("🍎 MPS (Metal Performance Shaders) 사용 가능")
            return True
        else:
            print_warning("MPS 사용 불가 - CPU 모드로 실행")
            return False
    except ImportError:
        print_error("PyTorch가 설치되지 않았습니다")
        return False

def setup_environment(system_info):
    """환경 변수 설정"""
    env_vars = {}
    
    # 기본 설정
    env_vars.update({
        "APP_NAME": "MyCloset AI Backend",
        "APP_VERSION": "3.0.0",
        "DEBUG": "True",
        "HOST": "0.0.0.0",
        "PORT": "8000"
    })
    
    # M3 Max 최적화 설정
    if system_info["is_m3_max"]:
        env_vars.update({
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
            "OMP_NUM_THREADS": "16",
            "MKL_NUM_THREADS": "16",
            "DEVICE_TYPE": "mps"
        })
        print_success("🍎 M3 Max 최적화 환경 변수 설정됨")
    else:
        env_vars.update({
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "DEVICE_TYPE": "cpu"
        })
        print_info("CPU 최적화 환경 변수 설정됨")
    
    # 환경 변수 적용
    for key, value in env_vars.items():
        os.environ[key] = value
    
    return env_vars

def find_main_module():
    """main.py 모듈 경로 찾기"""
    possible_paths = [
        "app.main:app",
        "main:app",
        "app:app"
    ]
    
    for path in possible_paths:
        module_path = path.split(":")[0].replace(".", "/") + ".py"
        if Path(module_path).exists():
            print_success(f"메인 모듈 발견: {path}")
            return path
    
    # 기본값
    main_path = "app.main:app"
    print_info(f"기본 모듈 사용: {main_path}")
    return main_path

def get_server_config(system_info):
    """서버 설정 가져오기"""
    config = {
        "host": os.getenv("HOST", "0.0.0.0"),
        "port": int(os.getenv("PORT", 8000)),
        "reload": os.getenv("DEBUG", "True").lower() == "true",
        "workers": 1,  # M3 Max GPU 메모리 공유 이슈 방지
        "log_level": "info",
        "access_log": True,
        "timeout_keep_alive": 30,
        "limit_concurrency": 1000,
        "limit_max_requests": 10000
    }
    
    # M3 Max 특화 설정
    if system_info["is_m3_max"]:
        config.update({
            "workers": 1,  # 단일 워커 (GPU 메모리 최적화)
            "limit_concurrency": 500,  # 동시 연결 제한 (메모리 고려)
        })
        print_info("🍎 M3 Max 최적화 서버 설정 적용")
    
    return config

def run_server():
    """서버 실행"""
    print_header("MyCloset AI Backend 서버 시작")
    
    # 1. Python 버전 확인
    if not check_python_version():
        sys.exit(1)
    
    # 2. 시스템 정보 감지
    system_info = detect_system()
    
    # 3. 가상환경 확인
    venv_active = check_virtual_env()
    if not venv_active:
        print_warning("가상환경 활성화를 권장합니다")
        response = input("계속 진행하시겠습니까? (y/N): ")
        if response.lower() != 'y':
            print_info("가상환경 활성화 후 다시 시도해주세요")
            print_info("명령어: source venv/bin/activate")
            sys.exit(1)
    
    # 4. 의존성 확인
    if not check_dependencies():
        sys.exit(1)
    
    # 5. PyTorch MPS 확인
    mps_available = check_pytorch_mps()
    
    # 6. 환경 변수 설정
    env_vars = setup_environment(system_info)
    
    # 7. 메인 모듈 확인
    main_module = find_main_module()
    
    # 8. 서버 설정
    config = get_server_config(system_info)
    
    # 9. 서버 시작 정보 출력
    print_header("서버 실행 정보")
    print_info(f"📍 주소: http://{config['host']}:{config['port']}")
    print_info(f"📖 API 문서: http://{config['host']}:{config['port']}/docs")
    print_info(f"🔄 자동 리로드: {'✅' if config['reload'] else '❌'}")
    print_info(f"👥 워커 수: {config['workers']}")
    print_info(f"🎯 디바이스: {'MPS' if mps_available else 'CPU'}")
    print_info(f"🍎 M3 Max 최적화: {'✅' if system_info['is_m3_max'] else '❌'}")
    
    # 10. 서버 실행
    try:
        import uvicorn
        
        print_header("서버 시작 중...")
        print_info("종료하려면 Ctrl+C를 누르세요")
        
        time.sleep(1)  # 메시지 출력 대기
        
        uvicorn.run(
            main_module,
            host=config["host"],
            port=config["port"],
            reload=config["reload"],
            workers=config["workers"],
            log_level=config["log_level"],
            access_log=config["access_log"],
            timeout_keep_alive=config["timeout_keep_alive"],
            limit_concurrency=config["limit_concurrency"],
            limit_max_requests=config["limit_max_requests"],
            loop="auto"
        )
        
    except KeyboardInterrupt:
        print_success("\n🛑 서버가 안전하게 종료되었습니다")
    except ImportError:
        print_error("Uvicorn이 설치되지 않았습니다")
        print_info("설치 명령어: pip install uvicorn")
        sys.exit(1)
    except Exception as e:
        print_error(f"서버 실행 실패: {e}")
        sys.exit(1)

def main():
    """메인 함수"""
    # 작업 디렉토리를 스크립트 위치로 변경
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print_colored("""
🍎 MyCloset AI Backend Server
=============================
M3 Max 128GB 최적화 버전
8단계 AI 파이프라인 지원
""", Colors.HEADER + Colors.BOLD)
    
    run_server()

if __name__ == "__main__":
    main()