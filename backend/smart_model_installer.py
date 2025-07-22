#!/usr/bin/env python3
"""
🔥 MyCloset AI - 스마트 모델 설치 시스템 (라이브러리 활용)
===============================================================================
✅ pip 라이브러리를 활용한 자동 모델 설치
✅ rembg, ultralytics, transformers 등 검증된 패키지 사용
✅ conda 환경 최적화
✅ 에러 없는 안정적인 설치
===============================================================================
"""

import os
import sys
import subprocess
import importlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ==============================================
# 🔥 1. 패키지 기반 모델 정의
# ==============================================

MODEL_PACKAGES = {
    # Step 01: Human Parsing
    "human_parsing": {
        "pip_packages": ["rembg[new]", "segment-anything"],
        "conda_packages": ["pillow", "opencv"],
        "description": "RemBG + SAM 기반 인체 분할",
        "models_downloaded": ["u2net", "u2net_human_seg", "silueta"],
        "step_folders": ["step_01_human_parsing"],
        "priority": 1,
        "test_command": "python -c 'import rembg; print(\"Human parsing OK\")'"
    },
    
    # Step 02: Pose Estimation  
    "pose_estimation": {
        "pip_packages": ["ultralytics", "mediapipe"],
        "conda_packages": [],
        "description": "YOLOv8 Pose + MediaPipe 기반 포즈 추정",
        "models_downloaded": ["yolov8n-pose.pt", "pose_landmarker_heavy"],
        "step_folders": ["step_02_pose_estimation"],
        "priority": 1,
        "test_command": "python -c 'from ultralytics import YOLO; print(\"Pose estimation OK\")'"
    },
    
    # Step 03: Cloth Segmentation
    "cloth_segmentation": {
        "pip_packages": ["rembg[new]", "transformers"],
        "conda_packages": ["pillow"],
        "description": "RemBG + Transformers 기반 의류 분할",
        "models_downloaded": ["u2net_cloth_seg", "u2netp", "cloth-segm"],
        "step_folders": ["step_03_cloth_segmentation"],
        "priority": 1,
        "test_command": "python -c 'import rembg; from transformers import pipeline; print(\"Cloth segmentation OK\")'"
    },
    
    # Step 06: Virtual Fitting (이미 있는 모델 활용)
    "virtual_fitting": {
        "pip_packages": ["diffusers", "transformers", "accelerate"],
        "conda_packages": [],
        "description": "Stable Diffusion 기반 가상 피팅",
        "models_downloaded": ["existing_models"],
        "step_folders": ["step_06_virtual_fitting"],
        "priority": 2,
        "test_command": "python -c 'from diffusers import StableDiffusionPipeline; print(\"Virtual fitting OK\")'"
    },
    
    # Step 08: Quality Assessment (이미 있는 모델 활용)
    "quality_assessment": {
        "pip_packages": ["transformers", "torch-fidelity"],
        "conda_packages": [],
        "description": "CLIP 기반 품질 평가",
        "models_downloaded": ["existing_models"],
        "step_folders": ["step_08_quality_assessment"],
        "priority": 2,
        "test_command": "python -c 'from transformers import CLIPModel; print(\"Quality assessment OK\")'"
    }
}

# ==============================================
# 🔥 2. 스마트 설치 관리자
# ==============================================

class SmartModelInstaller:
    """스마트 모델 설치 관리자"""
    
    def __init__(self):
        self.project_root = self._find_project_root()
        self.ai_models_dir = self.project_root / "backend" / "ai_models"
        self.conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
        self.installation_log = []
        
        logger.info(f"🏠 프로젝트 루트: {self.project_root}")
        logger.info(f"🤖 AI 모델 경로: {self.ai_models_dir}")
        logger.info(f"🐍 conda 환경: {self.conda_env}")
    
    def _find_project_root(self) -> Path:
        """프로젝트 루트 찾기"""
        current = Path(__file__).resolve()
        
        for _ in range(10):
            if current.name == 'backend':
                return current.parent
            if current.parent == current:
                break
            current = current.parent
        
        return Path.cwd()
    
    def check_environment(self) -> Dict[str, Any]:
        """환경 상태 체크"""
        env_info = {
            "conda_active": bool(self.conda_env),
            "conda_env": self.conda_env,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "pip_available": self._check_command("pip"),
            "conda_available": self._check_command("conda"),
            "git_available": self._check_command("git"),
            "package_status": {},
            "missing_packages": []
        }
        
        # 핵심 패키지 체크
        core_packages = ["torch", "torchvision", "numpy", "pillow", "opencv-python"]
        for package in core_packages:
            try:
                importlib.import_module(package.replace("-", "_"))
                env_info["package_status"][package] = "✅ 설치됨"
            except ImportError:
                env_info["package_status"][package] = "❌ 누락"
                env_info["missing_packages"].append(package)
        
        return env_info
    
    def _check_command(self, command: str) -> bool:
        """명령어 사용 가능 여부 체크"""
        try:
            subprocess.run([command, "--version"], 
                         capture_output=True, check=True, timeout=10)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def install_package_set(self, package_name: str, force: bool = False) -> bool:
        """패키지 세트 설치"""
        if package_name not in MODEL_PACKAGES:
            logger.error(f"❌ 알 수 없는 패키지: {package_name}")
            return False
        
        package_info = MODEL_PACKAGES[package_name]
        logger.info(f"📦 {package_name} 설치 시작: {package_info['description']}")
        
        success_count = 0
        total_operations = len(package_info.get('conda_packages', [])) + len(package_info.get('pip_packages', []))
        
        # 1. conda 패키지 설치
        conda_packages = package_info.get('conda_packages', [])
        if conda_packages and self._check_command("conda"):
            logger.info(f"🐍 conda 패키지 설치: {', '.join(conda_packages)}")
            if self._install_conda_packages(conda_packages):
                success_count += len(conda_packages)
            else:
                logger.warning("⚠️ conda 패키지 설치 일부 실패")
        
        # 2. pip 패키지 설치
        pip_packages = package_info.get('pip_packages', [])
        if pip_packages:
            logger.info(f"📦 pip 패키지 설치: {', '.join(pip_packages)}")
            if self._install_pip_packages(pip_packages):
                success_count += len(pip_packages)
            else:
                logger.warning("⚠️ pip 패키지 설치 일부 실패")
        
        # 3. 디렉토리 생성
        step_folders = package_info.get('step_folders', [])
        for folder in step_folders:
            folder_path = self.ai_models_dir / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 디렉토리 생성: {folder_path}")
        
        # 4. 설치 테스트
        test_command = package_info.get('test_command')
        if test_command:
            logger.info(f"🧪 설치 테스트 실행...")
            if self._test_installation(test_command):
                logger.info(f"✅ {package_name} 설치 및 테스트 완료!")
                self.installation_log.append(f"✅ {package_name}: 성공")
                return True
            else:
                logger.error(f"❌ {package_name} 테스트 실패")
                self.installation_log.append(f"❌ {package_name}: 테스트 실패")
                return False
        
        logger.info(f"✅ {package_name} 설치 완료 (테스트 없음)")
        self.installation_log.append(f"✅ {package_name}: 설치 완료")
        return True
    
    def _install_conda_packages(self, packages: List[str]) -> bool:
        """conda 패키지 설치"""
        try:
            cmd = ["conda", "install", "-y"] + packages + ["-c", "conda-forge"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ conda 패키지 설치 성공")
                return True
            else:
                logger.error(f"❌ conda 설치 실패: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ conda 설치 중 오류: {e}")
            return False
    
    def _install_pip_packages(self, packages: List[str]) -> bool:
        """pip 패키지 설치"""
        try:
            for package in packages:
                logger.info(f"  📦 설치 중: {package}")
                cmd = [sys.executable, "-m", "pip", "install", package, "--upgrade"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    logger.error(f"❌ {package} 설치 실패: {result.stderr}")
                    return False
                else:
                    logger.info(f"  ✅ {package} 설치 완료")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ pip 설치 중 오류: {e}")
            return False
    
    def _test_installation(self, test_command: str) -> bool:
        """설치 테스트"""
        try:
            result = subprocess.run(test_command, shell=True, 
                                  capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"❌ 테스트 실행 실패: {e}")
            return False
    
    def install_all_priority_packages(self, max_priority: int = 1) -> Dict[str, bool]:
        """우선순위별 패키지 일괄 설치"""
        results = {}
        
        # 우선순위에 따라 정렬
        priority_packages = [
            (name, info) for name, info in MODEL_PACKAGES.items()
            if info.get('priority', 3) <= max_priority
        ]
        priority_packages.sort(key=lambda x: x[1].get('priority', 3))
        
        logger.info(f"🚀 우선순위 {max_priority} 이하 패키지 설치 시작")
        logger.info(f"   대상 패키지: {[name for name, _ in priority_packages]}")
        
        for package_name, package_info in priority_packages:
            logger.info(f"\n{'='*50}")
            logger.info(f"📦 {package_name} 설치 중... (우선순위: {package_info.get('priority', 3)})")
            
            try:
                success = self.install_package_set(package_name)
                results[package_name] = success
                
                if success:
                    logger.info(f"✅ {package_name} 설치 성공!")
                else:
                    logger.error(f"❌ {package_name} 설치 실패")
                
            except Exception as e:
                logger.error(f"❌ {package_name} 설치 중 예외: {e}")
                results[package_name] = False
        
        return results
    
    def create_test_script(self) -> Path:
        """테스트 스크립트 생성"""
        test_script_content = '''#!/usr/bin/env python3
"""
MyCloset AI - 모델 테스트 스크립트
"""

import sys
import traceback

def test_human_parsing():
    """인체 파싱 테스트"""
    try:
        import rembg
        from PIL import Image
        import numpy as np
        
        # RemBG 세션 생성
        session = rembg.new_session('u2net_human_seg')
        
        # 더미 이미지로 테스트
        test_image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        result = rembg.remove(test_image, session=session)
        
        print("✅ Step 01 - Human Parsing: OK")
        return True
    except Exception as e:
        print(f"❌ Step 01 - Human Parsing: {e}")
        return False

def test_pose_estimation():
    """포즈 추정 테스트"""
    try:
        from ultralytics import YOLO
        import numpy as np
        
        # YOLOv8 포즈 모델 로드
        model = YOLO('yolov8n-pose.pt')
        
        # 더미 이미지로 테스트  
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(test_image, verbose=False)
        
        print("✅ Step 02 - Pose Estimation: OK")
        return True
    except Exception as e:
        print(f"❌ Step 02 - Pose Estimation: {e}")
        return False

def test_cloth_segmentation():
    """의류 분할 테스트"""
    try:
        import rembg
        from PIL import Image
        import numpy as np
        
        # RemBG 의류 세션 생성
        session = rembg.new_session('u2netp')
        
        # 더미 이미지로 테스트
        test_image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        result = rembg.remove(test_image, session=session)
        
        print("✅ Step 03 - Cloth Segmentation: OK")
        return True
    except Exception as e:
        print(f"❌ Step 03 - Cloth Segmentation: {e}")
        return False

def test_virtual_fitting():
    """가상 피팅 테스트"""
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        
        # 간단한 파이프라인 체크
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Device: {device}")
        
        print("✅ Step 06 - Virtual Fitting: Libraries OK")
        return True
    except Exception as e:
        print(f"❌ Step 06 - Virtual Fitting: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🧪 MyCloset AI - 모델 테스트 시작")
    print("="*50)
    
    tests = [
        ("인체 파싱", test_human_parsing),
        ("포즈 추정", test_pose_estimation),
        ("의류 분할", test_cloth_segmentation),
        ("가상 피팅", test_virtual_fitting)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\\n🧪 {test_name} 테스트 중...")
        try:
            if test_func():
                passed += 1
            else:
                print(f"   실패: {test_name}")
        except Exception as e:
            print(f"   예외 발생: {test_name} - {e}")
    
    print(f"\\n📊 테스트 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("🎉 모든 테스트 통과! MyCloset AI 준비 완료!")
        return 0
    else:
        print("⚠️ 일부 테스트 실패. 설치를 확인하세요.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''
        
        test_script_path = self.ai_models_dir / "test_models.py"
        test_script_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(test_script_path, 'w', encoding='utf-8') as f:
            f.write(test_script_content)
        
        # 실행 권한 부여
        os.chmod(test_script_path, 0o755)
        
        logger.info(f"📝 테스트 스크립트 생성: {test_script_path}")
        return test_script_path
    
    def print_installation_summary(self):
        """설치 요약 출력"""
        print("\n" + "="*70)
        print("📊 MyCloset AI - 스마트 모델 설치 요약")
        print("="*70)
        
        print("📋 설치 로그:")
        for log_entry in self.installation_log:
            print(f"   {log_entry}")
        
        print(f"\n🏠 프로젝트 경로: {self.project_root}")
        print(f"🤖 AI 모델 경로: {self.ai_models_dir}")
        print(f"🐍 conda 환경: {self.conda_env}")
        
        # 다음 단계 안내
        print(f"\n🚀 다음 단계:")
        test_script = self.ai_models_dir / "test_models.py"
        if test_script.exists():
            print(f"   1. 테스트 실행: python {test_script}")
        print("   2. auto_model_detector.py 업데이트")
        print("   3. 기본 파이프라인 테스트")
        
        print("="*70)

# ==============================================
# 🔥 3. CLI 인터페이스
# ==============================================

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MyCloset AI - 스마트 모델 설치 도구')
    parser.add_argument('--check-env', action='store_true', help='환경 상태 확인')
    parser.add_argument('--install-core', action='store_true', help='핵심 모델 설치 (우선순위 1)')
    parser.add_argument('--install-all', action='store_true', help='모든 모델 설치')
    parser.add_argument('--install-package', type=str, help='특정 패키지 설치')
    parser.add_argument('--create-test', action='store_true', help='테스트 스크립트 생성')
    parser.add_argument('--test', action='store_true', help='설치 테스트 실행')
    
    args = parser.parse_args()
    
    installer = SmartModelInstaller()
    
    # 환경 체크
    if args.check_env:
        env_info = installer.check_environment()
        print("\n🔍 환경 상태 체크")
        print("-"*50)
        print(f"conda 활성화: {'✅' if env_info['conda_active'] else '❌'}")
        print(f"conda 환경: {env_info['conda_env'] or 'None'}")
        print(f"Python 버전: {env_info['python_version']}")
        
        print("\n📦 패키지 상태:")
        for package, status in env_info['package_status'].items():
            print(f"   {package}: {status}")
        
        if env_info['missing_packages']:
            print(f"\n⚠️ 누락 패키지: {', '.join(env_info['missing_packages'])}")
            print("   다음 명령어로 설치: --install-core")
        
        return
    
    # 테스트 스크립트 생성
    if args.create_test:
        test_script = installer.create_test_script()
        print(f"✅ 테스트 스크립트 생성 완료: {test_script}")
        return
    
    # 테스트 실행
    if args.test:
        test_script = installer.ai_models_dir / "test_models.py"
        if test_script.exists():
            subprocess.run([sys.executable, str(test_script)])
        else:
            print("❌ 테스트 스크립트가 없습니다. --create-test로 먼저 생성하세요.")
        return
    
    # 핵심 모델 설치
    if args.install_core:
        print("🚀 핵심 모델 설치 시작...")
        results = installer.install_all_priority_packages(max_priority=1)
        
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        print(f"\n📊 설치 완료: {success_count}/{total_count}")
        installer.print_installation_summary()
        
        if success_count == total_count:
            print("🎉 핵심 모델 설치 완료!")
            installer.create_test_script()
            return 0
        else:
            print("⚠️ 일부 설치 실패")
            return 1
    
    # 모든 모델 설치
    if args.install_all:
        print("🚀 모든 모델 설치 시작...")
        results = installer.install_all_priority_packages(max_priority=3)
        
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        installer.print_installation_summary()
        return 0 if success_count == total_count else 1
    
    # 특정 패키지 설치
    if args.install_package:
        package_name = args.install_package
        if installer.install_package_set(package_name):
            print(f"✅ {package_name} 설치 완료!")
            return 0
        else:
            print(f"❌ {package_name} 설치 실패")
            return 1
    
    # 기본 도움말
    print("💡 사용법:")
    print("   python smart_model_installer.py --check-env     # 환경 상태 확인")
    print("   python smart_model_installer.py --install-core  # 핵심 모델 설치")
    print("   python smart_model_installer.py --create-test   # 테스트 스크립트 생성")
    print("   python smart_model_installer.py --test          # 설치 테스트")

if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("\n⚠️ 사용자가 중단했습니다.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)