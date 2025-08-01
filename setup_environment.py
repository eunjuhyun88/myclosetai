#!/usr/bin/env python3
"""
🍎 MyCloset AI 환경 설치 및 설정 스크립트 v7.1 (의존성 순서 해결)
================================================================

✅ conda 환경 우선 최적화
✅ M3 Max 128GB 메모리 완전 활용
✅ 의존성 순서 문제 해결 (torch -> xformers)
✅ 단계별 안전한 패키지 설치
✅ 자동 시스템 감지 및 설정
✅ Python Path 자동 설정
✅ 환경 변수 자동 구성

사용법:
    python setup_environment.py
    python setup_environment.py --reinstall
    python setup_environment.py --conda-only
    python setup_environment.py --safe-mode  # 문제 패키지 제외
"""

import os
import sys
import subprocess
import platform
import shutil
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class SystemDetector:
    """시스템 정보 감지 클래스"""
    
    def __init__(self):
        self.platform = platform.system()
        self.architecture = platform.architecture()[0]
        self.python_version = platform.python_version()
        self.cpu_count = os.cpu_count() or 4
        self.is_conda = self._detect_conda()
        self.conda_env = os.getenv('CONDA_DEFAULT_ENV', 'none')
        self.is_m3_max = self._detect_m3_max()
        self.memory_gb = self._get_memory_gb()
        self.has_miniconda = self._has_miniconda()
    
    def _detect_conda(self) -> bool:
        """conda 환경 감지"""
        indicators = [
            'CONDA_DEFAULT_ENV' in os.environ,
            'CONDA_PREFIX' in os.environ,
            'conda' in sys.executable.lower(),
            'miniconda' in sys.executable.lower()
        ]
        return any(indicators)
    
    def _detect_m3_max(self) -> bool:
        """M3 Max 칩 감지"""
        if self.platform != 'Darwin':
            return False
        
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'], 
                capture_output=True, text=True, timeout=5
            )
            chip_info = result.stdout.strip()
            return 'M3' in chip_info and 'Max' in chip_info
        except:
            return False
    
    def _get_memory_gb(self) -> float:
        """메모리 크기 감지"""
        try:
            if self.platform == 'Darwin':
                result = subprocess.run(
                    ['sysctl', '-n', 'hw.memsize'],
                    capture_output=True, text=True
                )
                bytes_memory = int(result.stdout.strip())
                return round(bytes_memory / (1024**3), 1)
            else:
                # Linux/Windows는 psutil 사용
                import psutil
                return round(psutil.virtual_memory().total / (1024**3), 1)
        except:
            return 16.0
    
    def _has_miniconda(self) -> bool:
        """Miniconda 설치 여부 확인"""
        conda_paths = [
            os.path.expanduser('~/miniconda3/bin/conda'),
            os.path.expanduser('~/anaconda3/bin/conda'),
            '/opt/miniconda3/bin/conda',
            '/opt/anaconda3/bin/conda'
        ]
        
        for path in conda_paths:
            if os.path.exists(path):
                return True
        
        # PATH에서 conda 확인
        return shutil.which('conda') is not None
    
    def get_info(self) -> Dict[str, Any]:
        """시스템 정보 딕셔너리 반환"""
        return {
            'platform': self.platform,
            'architecture': self.architecture,
            'python_version': self.python_version,
            'cpu_count': self.cpu_count,
            'is_conda': self.is_conda,
            'conda_env': self.conda_env,
            'is_m3_max': self.is_m3_max,
            'memory_gb': self.memory_gb,
            'has_miniconda': self.has_miniconda
        }

class EnvironmentSetup:
    """환경 설정 및 설치 클래스"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.system = SystemDetector()
        self.project_root = project_root or Path(__file__).parent.absolute()
        self.backend_dir = self.project_root / 'backend'
        self.app_dir = self.backend_dir / 'app'
        self.config_dir = self.backend_dir / 'configs'
        
        # 의존성 순서 정의
        self.dependency_stages = {
            'core': [
                'numpy==1.24.3',
                'packaging',
                'wheel',
                'setuptools'
            ],
            'pytorch': [
                'torch==2.2.2',
                'torchvision==0.17.2', 
                'torchaudio==2.2.2'
            ],
            'ai_base': [
                'transformers==4.35.0',
                'diffusers==0.21.4',
                'accelerate==0.24.0',
                'tokenizers==0.14.1',
                'huggingface-hub==0.17.3',
                'safetensors==0.4.0'
            ],
            'web': [
                'fastapi==0.104.1',
                'uvicorn[standard]==0.24.0',
                'python-multipart==0.0.6',
                'pydantic==2.5.0',
                'pydantic-settings==2.1.0'
            ],
            'image': [
                'Pillow==10.0.1',
                'opencv-python==4.8.0.76',
                'scikit-image==0.21.0',
                'imageio==2.31.5',
                'imageio-ffmpeg==0.4.9',
                'albumentations==1.3.1'
            ],
            'optional': [
                'xformers==0.0.22',  # torch 설치 후에
                'segment-anything==1.0',
                'timm==0.9.8',
                'controlnet-aux==0.0.7',
                'clip-by-openai==1.0',
                'open-clip-torch==2.23.0',
                'onnxruntime==1.16.1'
            ]
        }
        
    def print_system_info(self):
        """시스템 정보 출력"""
        info = self.system.get_info()
        logger.info("🔍 시스템 정보 감지 결과:")
        logger.info(f"  🖥️  플랫폼: {info['platform']} ({info['architecture']})")
        logger.info(f"  🐍 Python: {info['python_version']}")
        logger.info(f"  💻 CPU: {info['cpu_count']}코어")
        logger.info(f"  💾 메모리: {info['memory_gb']}GB")
        logger.info(f"  🍎 M3 Max: {'✅' if info['is_m3_max'] else '❌'}")
        logger.info(f"  🐍 conda: {'✅' if info['is_conda'] else '❌'} ({info['conda_env']})")
        logger.info(f"  📦 Miniconda: {'✅' if info['has_miniconda'] else '❌'}")
    
    def setup_python_path(self):
        """Python Path 자동 설정"""
        logger.info("🔧 Python Path 설정 중...")
        
        paths_to_add = [
            str(self.backend_dir),
            str(self.app_dir),
            str(self.project_root)
        ]
        
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)
        
        # 환경 변수 설정
        env_vars = {
            'PROJECT_ROOT': str(self.project_root),
            'BACKEND_ROOT': str(self.backend_dir),
            'APP_ROOT': str(self.app_dir),
            'PYTHONPATH': f"{self.backend_dir}:{os.environ.get('PYTHONPATH', '')}"
        }
        
        os.environ.update(env_vars)
        
        # 작업 디렉토리 변경
        try:
            os.chdir(self.backend_dir)
            logger.info(f"✅ 작업 디렉토리: {os.getcwd()}")
        except OSError as e:
            logger.warning(f"⚠️ 작업 디렉토리 변경 실패: {e}")
        
        logger.info("✅ Python Path 설정 완료")
    
    def install_miniconda(self):
        """Miniconda 자동 설치 (M3 Max용)"""
        if self.system.has_miniconda:
            logger.info("✅ Miniconda가 이미 설치되어 있습니다.")
            return True
        
        if not self.system.is_m3_max:
            logger.warning("⚠️ M3 Max가 아닌 시스템에서는 수동 설치를 권장합니다.")
            return False
        
        logger.info("📦 Miniconda 설치 중...")
        
        # M3 Max용 Miniconda 다운로드
        installer_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        installer_path = self.project_root / "Miniconda3-latest-MacOSX-arm64.sh"
        
        try:
            # 다운로드
            subprocess.run([
                'curl', '-o', str(installer_path), installer_url
            ], check=True)
            
            # 설치
            subprocess.run([
                'bash', str(installer_path), '-b', '-p', 
                os.path.expanduser('~/miniconda3')
            ], check=True)
            
            # 초기화
            subprocess.run([
                os.path.expanduser('~/miniconda3/bin/conda'), 'init', 'bash'
            ], check=True)
            
            # 설치 파일 정리
            installer_path.unlink()
            
            logger.info("✅ Miniconda 설치 완료")
            logger.info("🔄 새 터미널을 열어서 conda 명령을 사용하세요.")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Miniconda 설치 실패: {e}")
            return False
    
    def create_conda_environment(self, env_name: str = "mycloset-ai-clean"):
        """conda 환경 생성"""
        if not self.system.has_miniconda:
            logger.error("❌ conda가 설치되어 있지 않습니다.")
            return False
        
        logger.info(f"🐍 conda 환경 '{env_name}' 생성 중...")
        
        try:
            # 환경 존재 확인
            result = subprocess.run([
                'conda', 'env', 'list'
            ], capture_output=True, text=True)
            
            if env_name in result.stdout:
                logger.info(f"✅ conda 환경 '{env_name}'이 이미 존재합니다.")
                return True
            
            # Python 3.10.18 환경 생성
            subprocess.run([
                'conda', 'create', '-n', env_name, 'python=3.10.18', '-y'
            ], check=True)
            
            logger.info(f"✅ conda 환경 '{env_name}' 생성 완료")
            logger.info(f"🔄 다음 명령으로 활성화: conda activate {env_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ conda 환경 생성 실패: {e}")
            return False
    
    def install_pytorch_via_conda(self):
        """conda를 통한 PyTorch 설치 (M3 Max 최적화)"""
        if not self.system.has_miniconda or not self.system.is_conda:
            logger.warning("⚠️ conda 환경이 아니므로 conda PyTorch 설치를 건너뜁니다.")
            return False
        
        logger.info("🔥 conda를 통한 PyTorch 설치...")
        
        try:
            # PyTorch 설치 여부 확인
            result = subprocess.run([
                sys.executable, '-c', 'import torch; print(torch.__version__)'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ PyTorch가 이미 설치되어 있습니다: {result.stdout.strip()}")
                return True
            
            if self.system.is_m3_max:
                # M3 Max용 PyTorch 설치
                cmd = [
                    'conda', 'install', '-y',
                    'pytorch==2.2.2', 
                    'torchvision==0.17.2', 
                    'torchaudio==2.2.2', 
                    '-c', 'pytorch'
                ]
            else:
                # CPU 전용 PyTorch 설치
                cmd = [
                    'conda', 'install', '-y',
                    'pytorch', 'torchvision', 'torchaudio', 
                    'cpuonly', '-c', 'pytorch'
                ]
            
            logger.info("⏳ PyTorch 설치 중... (시간이 걸릴 수 있습니다)")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            
            if result.returncode == 0:
                logger.info("✅ conda PyTorch 설치 완료")
                return True
            else:
                logger.warning(f"⚠️ conda PyTorch 설치 실패: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ conda PyTorch 설치 시간 초과")
            return False
        except Exception as e:
            logger.error(f"❌ conda PyTorch 설치 중 오류: {e}")
            return False
    
    def install_packages_by_stages(self, force_reinstall: bool = False, safe_mode: bool = False):
        """단계별 패키지 설치 (의존성 순서 고려)"""
        logger.info("📦 단계별 패키지 설치 시작...")
        
        success_count = 0
        total_stages = len(self.dependency_stages)
        
        # 각 단계별로 설치
        for stage_name, packages in self.dependency_stages.items():
            if safe_mode and stage_name == 'optional':
                logger.info(f"🛡️ 안전 모드: '{stage_name}' 단계 건너뛰기")
                continue
                
            logger.info(f"🔥 {stage_name.upper()} 단계 설치 중...")
            
            if self._install_package_stage(packages, stage_name, force_reinstall):
                success_count += 1
                logger.info(f"✅ {stage_name.upper()} 단계 완료")
            else:
                logger.warning(f"⚠️ {stage_name.upper()} 단계 일부 실패")
            
            # 잠시 대기 (안정성)
            time.sleep(1)
        
        logger.info(f"📊 패키지 설치 완료: {success_count}/{total_stages} 단계 성공")
        return success_count > 0
    
    def _install_package_stage(self, packages: List[str], stage_name: str, force_reinstall: bool = False) -> bool:
        """단계별 패키지 설치 실행"""
        success_count = 0
        
        for package in packages:
            try:
                # 이미 설치된 패키지 확인 (force_reinstall이 False인 경우)
                if not force_reinstall and self._is_package_installed(package):
                    logger.info(f"  ✅ {package} (이미 설치됨)")
                    success_count += 1
                    continue
                
                # 패키지 설치
                if self._install_single_package(package, force_reinstall):
                    success_count += 1
                else:
                    logger.warning(f"  ⚠️ {package} 설치 실패")
                    
            except Exception as e:
                logger.warning(f"  ❌ {package} 설치 중 오류: {e}")
        
        return success_count > 0
    
    def _is_package_installed(self, package: str) -> bool:
        """패키지 설치 여부 확인"""
        package_name = package.split('==')[0].split('>=')[0].split('<=')[0]
        
        try:
            result = subprocess.run([
                sys.executable, '-c', f'import {package_name}'
            ], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except:
            # import 이름이 다른 경우들 처리
            name_mapping = {
                'opencv-python': 'cv2',
                'pillow': 'PIL',
                'scikit-image': 'skimage',
                'python-multipart': 'multipart',
                'clip-by-openai': 'clip'
            }
            
            if package_name.lower() in name_mapping:
                try:
                    result = subprocess.run([
                        sys.executable, '-c', f'import {name_mapping[package_name.lower()]}'
                    ], capture_output=True, text=True, timeout=10)
                    return result.returncode == 0
                except:
                    pass
            
            return False
    
    def _install_single_package(self, package: str, force_reinstall: bool = False) -> bool:
        """단일 패키지 설치"""
        try:
            cmd = [sys.executable, '-m', 'pip', 'install', package]
            
            if force_reinstall:
                cmd.extend(['--force-reinstall'])
            
            # xformers 특별 처리
            if 'xformers' in package.lower():
                cmd.extend(['--no-deps'])  # 의존성 무시하고 설치
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"  ✅ {package}")
                return True
            else:
                logger.warning(f"  ❌ {package}: {result.stderr.split(chr(10))[0] if result.stderr else 'Unknown error'}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.warning(f"  ⏰ {package}: 설치 시간 초과")
            return False
        except Exception as e:
            logger.warning(f"  ❌ {package}: {str(e)}")
            return False
    
    def setup_environment_variables(self):
        """환경 변수 설정"""
        logger.info("🔧 환경 변수 설정 중...")
        
        env_vars = {
            # PyTorch 설정
            'PYTORCH_ENABLE_MPS_FALLBACK': '1',
            'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',
            
            # 디바이스 설정
            'DEVICE': 'mps' if self.system.is_m3_max else 'cpu',
            
            # MyCloset AI 설정
            'MYCLOSET_AI_MODELS_PATH': str(self.backend_dir / 'ai_models'),
            'MYCLOSET_CORE_CONFIG_PATH': str(self.app_dir / 'core'),
            'MYCLOSET_CONDA_OPTIMIZED': 'true' if self.system.is_conda else 'false',
            'MYCLOSET_PACKAGE_MANAGER': 'conda' if self.system.is_conda else 'pip',
            
            # 성능 최적화
            'OMP_NUM_THREADS': str(max(1, self.system.cpu_count // 2)),
            'MKL_NUM_THREADS': str(max(1, self.system.cpu_count // 2)),
            'NUMEXPR_NUM_THREADS': str(max(1, self.system.cpu_count // 2))
        }
        
        # M3 Max 특화 설정
        if self.system.is_m3_max:
            env_vars.update({
                'MYCLOSET_M3_MAX_OPTIMIZED': 'true',
                'MYCLOSET_MEMORY_POOL_GB': '24' if self.system.is_conda else '32',
                'MYCLOSET_BATCH_SIZE': '4' if self.system.is_conda else '6',
                'MYCLOSET_MAX_WORKERS': '8' if self.system.is_conda else '12'
            })
        
        # 환경 변수 적용
        os.environ.update(env_vars)
        
        # .env 파일 생성
        env_file = self.backend_dir / '.env'
        with env_file.open('w') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        
        logger.info(f"✅ 환경 변수 설정 완료 (.env 파일 생성: {env_file})")
    
    def verify_installation(self):
        """설치 검증"""
        logger.info("🔍 설치 검증 중...")
        
        checks = []
        
        # Python 버전 확인
        python_version = tuple(map(int, platform.python_version().split('.')))
        checks.append({
            'name': 'Python 버전',
            'status': python_version >= (3, 10),
            'detail': f"Python {platform.python_version()}"
        })
        
        # 핵심 라이브러리 확인
        core_imports = {
            'torch': 'PyTorch',
            'torchvision': 'TorchVision', 
            'fastapi': 'FastAPI',
            'numpy': 'NumPy',
            'PIL': 'Pillow',
            'cv2': 'OpenCV',
            'transformers': 'Transformers',
            'diffusers': 'Diffusers'
        }
        
        for import_name, display_name in core_imports.items():
            try:
                module = __import__(import_name)
                version = getattr(module, '__version__', 'Unknown')
                checks.append({
                    'name': display_name,
                    'status': True,
                    'detail': f"{display_name} {version}"
                })
            except ImportError:
                checks.append({
                    'name': display_name,
                    'status': False,
                    'detail': f'{display_name}가 설치되지 않음'
                })
        
        # PyTorch 디바이스 확인
        try:
            import torch
            if self.system.is_m3_max:
                device_available = torch.backends.mps.is_available()
                device_name = 'MPS (M3 Max)'
            else:
                device_available = torch.cuda.is_available()
                device_name = 'CUDA' if device_available else 'CPU'
            
            checks.append({
                'name': '디바이스',
                'status': device_available or device_name == 'CPU',
                'detail': device_name
            })
        except ImportError:
            checks.append({
                'name': '디바이스',
                'status': False,
                'detail': 'PyTorch가 설치되지 않음'
            })
        
        # 결과 출력
        logger.info("📋 설치 검증 결과:")
        all_passed = True
        critical_failed = False
        
        for check in checks:
            status_icon = "✅" if check['status'] else "❌"
            logger.info(f"  {status_icon} {check['name']}: {check['detail']}")
            
            if not check['status']:
                all_passed = False
                # 중요한 패키지들
                if check['name'] in ['Python 버전', 'PyTorch', 'FastAPI', 'NumPy']:
                    critical_failed = True
        
        if all_passed:
            logger.info("🎉 모든 검증을 통과했습니다!")
        elif not critical_failed:
            logger.info("✅ 핵심 패키지는 정상 설치되었습니다!")
        else:
            logger.warning("⚠️ 중요한 패키지 설치에 실패했습니다.")
        
        return all_passed or not critical_failed
    
    def run_setup(self, args):
        """전체 설정 실행"""
        logger.info("🚀 MyCloset AI 환경 설정을 시작합니다...")
        
        # 시스템 정보 출력
        self.print_system_info()
        
        # Python Path 설정
        self.setup_python_path()
        
        # Miniconda 설치 (필요한 경우)
        if not args.conda_only and not self.system.has_miniconda and self.system.is_m3_max:
            if input("Miniconda를 설치하시겠습니까? (y/N): ").lower() == 'y':
                self.install_miniconda()
        
        # conda 환경 생성 (conda가 사용 가능한 경우)
        if self.system.has_miniconda:
            self.create_conda_environment()
        
        # PyTorch conda 설치 시도 (M3 Max conda 환경)
        if not args.conda_only and self.system.is_m3_max and self.system.is_conda:
            self.install_pytorch_via_conda()
        
        # 패키지 설치
        if not args.conda_only:
            self.install_packages_by_stages(args.reinstall, args.safe_mode)
        
        # 환경 변수 설정
        self.setup_environment_variables()
        
        # 설치 검증
        if not args.conda_only:
            success = self.verify_installation()
            
            if not success and not args.safe_mode:
                logger.info("🛡️ 안전 모드로 다시 시도해보세요: python setup_environment.py --safe-mode")
        
        logger.info("🎉 환경 설정이 완료되었습니다!")
        
        # 다음 단계 안내
        if self.system.has_miniconda and not self.system.is_conda:
            logger.info("📌 다음 단계:")
            logger.info("  1. 새 터미널을 열어주세요")
            logger.info("  2. 다음 명령을 실행하세요: conda activate mycloset-ai-clean")
            logger.info("  3. 애플리케이션을 실행하세요: python main.py")
        else:
            logger.info("📌 다음 단계:")
            logger.info("  애플리케이션을 실행하세요: python main.py")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='MyCloset AI 환경 설정')
    parser.add_argument('--reinstall', action='store_true', 
                       help='패키지 강제 재설치')
    parser.add_argument('--conda-only', action='store_true',
                       help='conda 환경만 설정 (패키지 설치 제외)')
    parser.add_argument('--safe-mode', action='store_true',
                       help='안전 모드 (문제 패키지 제외 설치)')
    
    args = parser.parse_args()
    
    try:
        setup = EnvironmentSetup()
        setup.run_setup(args)
    except KeyboardInterrupt:
        logger.info("\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"❌ 환경 설정 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()