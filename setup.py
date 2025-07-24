#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MyCloset AI - 패키지 설치 자동화 스크립트
===========================================

🚀 기능:
- conda 환경 설정 자동화
- Python 패키지 설치
- AI 모델 다운로드
- 환경 검증 및 최적화
- M3 Max 특화 설정

💡 사용법:
python setup.py install --user-install
python setup.py develop --conda-env=mycloset-ai-clean
"""

import os
import sys
import platform
import subprocess
import argparse
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop


# ============================================================================
# 📋 프로젝트 메타데이터
# ============================================================================

__version__ = "3.0.0"
__author__ = "MyCloset AI Team"
__email__ = "contact@mycloset-ai.com"
__description__ = "🍎 AI-Powered Virtual Try-On System with M3 Max Optimization"

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.absolute()
BACKEND_ROOT = PROJECT_ROOT / "backend"
FRONTEND_ROOT = PROJECT_ROOT / "frontend"

# ============================================================================
# 🔧 시스템 환경 감지
# ============================================================================

def detect_system_info():
    """시스템 정보 감지"""
    system_info = {
        'platform': platform.system(),
        'machine': platform.machine(),
        'python_version': platform.python_version(),
        'is_m3_max': False,
        'is_conda': False,
        'conda_env': None
    }
    
    # M3 Max 감지
    if system_info['platform'] == 'Darwin' and 'arm64' in system_info['machine']:
        try:
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                 capture_output=True, text=True, timeout=5)
            if 'M3' in result.stdout:
                system_info['is_m3_max'] = True
        except:
            pass
    
    # conda 환경 감지
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        system_info['is_conda'] = True
        system_info['conda_env'] = conda_env
    
    return system_info

def print_system_info(info):
    """시스템 정보 출력"""
    print("🔍 시스템 정보:")
    print(f"   OS: {info['platform']} ({info['machine']})")
    print(f"   Python: {info['python_version']}")
    print(f"   M3 Max: {'✅' if info['is_m3_max'] else '❌'}")
    print(f"   conda: {'✅' if info['is_conda'] else '❌'} ({info['conda_env'] or 'none'})")

# ============================================================================
# 🐍 conda 환경 관리
# ============================================================================

def create_conda_environment(env_name="mycloset-ai-clean"):
    """conda 환경 생성"""
    print(f"🐍 conda 환경 생성: {env_name}")
    
    env_file = PROJECT_ROOT / "environment.yml"
    if not env_file.exists():
        raise FileNotFoundError("environment.yml 파일을 찾을 수 없습니다.")
    
    try:
        # 기존 환경 제거 (선택적)
        subprocess.run(['conda', 'env', 'remove', '-n', env_name, '-y'], 
                      capture_output=True)
        
        # 새 환경 생성
        cmd = ['conda', 'env', 'create', '-f', str(env_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ conda 환경 생성 실패:\n{result.stderr}")
            return False
        
        print(f"✅ conda 환경 '{env_name}' 생성 완료")
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ conda 환경 생성 시간 초과")
        return False
    except Exception as e:
        print(f"❌ conda 환경 생성 오류: {e}")
        return False

def verify_conda_environment(env_name="mycloset-ai-clean"):
    """conda 환경 검증"""
    try:
        result = subprocess.run(['conda', 'env', 'list'], 
                              capture_output=True, text=True)
        return env_name in result.stdout
    except:
        return False

# ============================================================================
# 📦 패키지 설치
# ============================================================================

def install_backend_packages():
    """백엔드 패키지 설치"""
    print("📦 백엔드 패키지 설치 중...")
    
    requirements_file = BACKEND_ROOT / "requirements.txt"
    if not requirements_file.exists():
        print("❌ requirements.txt 파일을 찾을 수 없습니다.")
        return False
    
    try:
        cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ 패키지 설치 실패:\n{result.stderr}")
            return False
        
        print("✅ 백엔드 패키지 설치 완료")
        return True
        
    except Exception as e:
        print(f"❌ 패키지 설치 오류: {e}")
        return False

def install_frontend_packages():
    """프론트엔드 패키지 설치"""
    print("🌐 프론트엔드 패키지 설치 중...")
    
    package_json = FRONTEND_ROOT / "package.json"
    if not package_json.exists():
        print("❌ package.json 파일을 찾을 수 없습니다.")
        return False
    
    try:
        # npm install
        cmd = ['npm', 'install']
        result = subprocess.run(cmd, cwd=FRONTEND_ROOT, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ npm 설치 실패:\n{result.stderr}")
            return False
        
        print("✅ 프론트엔드 패키지 설치 완료")
        return True
        
    except Exception as e:
        print(f"❌ npm 설치 오류: {e}")
        return False

# ============================================================================
# 🤖 AI 모델 다운로드
# ============================================================================

def download_ai_models():
    """AI 모델 다운로드"""
    print("🤖 AI 모델 다운로드 시작...")
    
    try:
        # install_models.py 스크립트 실행
        script_path = PROJECT_ROOT / "install_models.py"
        if script_path.exists():
            cmd = [sys.executable, str(script_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"⚠️ AI 모델 다운로드 실패 (수동 설치 필요):\n{result.stderr}")
                return False
            
            print("✅ AI 모델 다운로드 완료")
            return True
        else:
            print("⚠️ install_models.py 스크립트를 찾을 수 없습니다.")
            return False
            
    except Exception as e:
        print(f"❌ AI 모델 다운로드 오류: {e}")
        return False

# ============================================================================
# 🔧 환경 최적화
# ============================================================================

def setup_m3_max_optimization():
    """M3 Max 최적화 설정"""
    print("🍎 M3 Max 최적화 설정 중...")
    
    # 환경 변수 설정
    env_vars = {
        'PYTORCH_ENABLE_MPS_FALLBACK': '1',
        'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',
        'DEVICE': 'mps',
        'OMP_NUM_THREADS': '8',
        'MKL_NUM_THREADS': '8'
    }
    
    # .env 파일 생성
    env_file = PROJECT_ROOT / ".env"
    with open(env_file, 'w') as f:
        f.write("# MyCloset AI - M3 Max 최적화 환경 변수\n")
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    print("✅ M3 Max 최적화 설정 완료")
    return True

def verify_installation():
    """설치 검증"""
    print("🔍 설치 검증 중...")
    
    verification_script = f"""
import sys
print(f"Python: {{sys.version}}")

try:
    import torch
    print(f"PyTorch: {{torch.__version__}}")
    print(f"MPS Available: {{torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}}")
except ImportError as e:
    print(f"❌ PyTorch 없음: {{e}}")

try:
    import fastapi
    print(f"FastAPI: {{fastapi.__version__}}")
except ImportError as e:
    print(f"❌ FastAPI 없음: {{e}}")

try:
    import numpy as np
    print(f"NumPy: {{np.__version__}}")
except ImportError as e:
    print(f"❌ NumPy 없음: {{e}}")

try:
    from PIL import Image
    print(f"Pillow: Available")
except ImportError as e:
    print(f"❌ Pillow 없음: {{e}}")
"""
    
    try:
        result = subprocess.run([sys.executable, '-c', verification_script], 
                              capture_output=True, text=True)
        print(result.stdout)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 검증 오류: {e}")
        return False

# ============================================================================
# 🛠️ 커스텀 설치 명령어
# ============================================================================

class CustomInstallCommand(install):
    """커스텀 설치 명령어"""
    
    user_options = install.user_options + [
        ('user-install', None, 'Install for current user only'),
        ('skip-models', None, 'Skip AI model download'),
        ('conda-env=', None, 'Conda environment name'),
    ]
    
    def initialize_options(self):
        install.initialize_options(self)
        self.user_install = False
        self.skip_models = False
        self.conda_env = "mycloset-ai-clean"
    
    def finalize_options(self):
        install.finalize_options(self)
    
    def run(self):
        # 시스템 정보 출력
        system_info = detect_system_info()
        print_system_info(system_info)
        
        print("\n🚀 MyCloset AI 설치 시작...")
        
        # conda 환경 설정
        if not system_info['is_conda']:
            print("⚠️ conda 환경이 활성화되지 않았습니다.")
            if input("conda 환경을 생성하시겠습니까? (y/N): ").lower() == 'y':
                create_conda_environment(self.conda_env)
                print(f"🔄 다음 명령어로 환경을 활성화하세요: conda activate {self.conda_env}")
                return
        
        # 패키지 설치
        success = True
        success &= install_backend_packages()
        success &= install_frontend_packages()
        
        # AI 모델 다운로드
        if not self.skip_models:
            download_ai_models()
        
        # M3 Max 최적화
        if system_info['is_m3_max']:
            setup_m3_max_optimization()
        
        # 설치 검증
        verify_installation()
        
        if success:
            print("\n🎉 MyCloset AI 설치 완료!")
            print("\n📝 다음 단계:")
            print("1. 백엔드 실행: cd backend && python app/main.py")
            print("2. 프론트엔드 실행: cd frontend && npm run dev")
            print("3. 브라우저에서 http://localhost:5173 접속")
        else:
            print("\n❌ 설치 중 오류가 발생했습니다. 로그를 확인하세요.")
        
        # 기본 설치 실행
        install.run(self)

class CustomDevelopCommand(develop):
    """커스텀 개발 명령어"""
    
    user_options = develop.user_options + [
        ('conda-env=', None, 'Conda environment name'),
    ]
    
    def initialize_options(self):
        develop.initialize_options(self)
        self.conda_env = "mycloset-ai-clean"
    
    def finalize_options(self):
        develop.finalize_options(self)
    
    def run(self):
        print("🔧 개발 환경 설정 중...")
        
        # 시스템 정보
        system_info = detect_system_info()
        print_system_info(system_info)
        
        # 개발 환경 설정
        if not verify_conda_environment(self.conda_env):
            create_conda_environment(self.conda_env)
        
        # 패키지 설치 (개발 모드)
        install_backend_packages()
        install_frontend_packages()
        
        print("✅ 개발 환경 설정 완료!")
        
        # 기본 개발 설치 실행
        develop.run(self)

# ============================================================================
# 📦 setuptools 설정
# ============================================================================

# 긴 설명 (README.md에서 읽기)
long_description = "MyCloset AI - AI-Powered Virtual Try-On System"
readme_file = PROJECT_ROOT / "README.md"
if readme_file.exists():
    with open(readme_file, 'r', encoding='utf-8') as f:
        long_description = f.read()

# requirements.txt에서 의존성 읽기
install_requires = []
requirements_file = BACKEND_ROOT / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        install_requires = [line.strip() for line in f 
                          if line.strip() and not line.startswith('#')]

setup(
    name="mycloset-ai",
    version=__version__,
    author=__author__,
    author_email=__email__,
    description=__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eunjuhyun88/mycloset-ai",
    project_urls={
        "Bug Tracker": "https://github.com/eunjuhyun88/mycloset-ai/issues",
        "Documentation": "https://github.com/eunjuhyun88/mycloset-ai/blob/main/README.md",
        "Source Code": "https://github.com/eunjuhyun88/mycloset-ai",
    },
    packages=find_packages(where="backend"),
    package_dir={"": "backend"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "black>=23.10.1",
            "isort>=5.12.0",
            "mypy>=1.7.0",
            "flake8>=6.1.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yml", "*.yaml", "*.json", "*.md", "*.txt"],
    },
    cmdclass={
        'install': CustomInstallCommand,
        'develop': CustomDevelopCommand,
    },
    entry_points={
        "console_scripts": [
            "mycloset-ai=app.main:main",
            "mycloset-setup=setup:main",
        ],
    },
    keywords="ai computer-vision virtual-tryon fashion pytorch fastapi react m3-max",
    zip_safe=False,
)

# ============================================================================
# 🚀 메인 함수 (직접 실행시)
# ============================================================================

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="MyCloset AI 설치 도구")
    parser.add_argument('--conda-env', default='mycloset-ai-clean', 
                       help='Conda 환경 이름')
    parser.add_argument('--skip-models', action='store_true', 
                       help='AI 모델 다운로드 건너뛰기')
    parser.add_argument('--user-install', action='store_true', 
                       help='사용자 설치만')
    
    args = parser.parse_args()
    
    print("🚀 MyCloset AI 설정 도구")
    print("=" * 50)
    
    # 시스템 정보
    system_info = detect_system_info()
    print_system_info(system_info)
    
    # conda 환경 생성
    if not system_info['is_conda']:
        if input("\nconda 환경을 생성하시겠습니까? (y/N): ").lower() == 'y':
            create_conda_environment(args.conda_env)
            print(f"\n✅ 다음 명령어로 환경을 활성화하세요:")
            print(f"conda activate {args.conda_env}")
            print(f"python setup.py install")
            return
    
    print("\n계속하려면 Enter를 누르세요...")
    input()

if __name__ == "__main__":
    main()