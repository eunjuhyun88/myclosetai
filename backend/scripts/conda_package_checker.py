#!/usr/bin/env python3
"""
🔍 Conda 환경 패키지 체크 및 누락 패키지 설치 스크립트
- conda list로 설치된 패키지 확인
- 누락된 패키지만 정확히 설치
- pip vs conda 설치 방식 구분
"""

import subprocess
import sys
import json
import re
from typing import Dict, List, Tuple, Optional

def log_info(msg: str):
    print(f"ℹ️  {msg}")

def log_success(msg: str):
    print(f"✅ {msg}")

def log_error(msg: str):
    print(f"❌ {msg}")

def log_warning(msg: str):
    print(f"⚠️  {msg}")

def run_command(cmd: str, shell: bool = True) -> Tuple[bool, str, str]:
    """명령어 실행"""
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True, timeout=30)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "명령어 실행 시간 초과"
    except Exception as e:
        return False, "", str(e)

class CondaPackageChecker:
    """Conda 패키지 체크 및 관리"""
    
    def __init__(self):
        self.conda_packages = {}
        self.pip_packages = {}
        self.required_packages = {
            # 패키지명: (conda 채널, pip 패키지명, 테스트 import)
            "pytorch": ("pytorch", "torch", "torch"),
            "torchvision": ("pytorch", "torchvision", "torchvision"),
            "torchaudio": ("pytorch", "torchaudio", "torchaudio"),
            "transformers": ("conda-forge", "transformers", "transformers"),
            "opencv": ("conda-forge", "opencv-python", "cv2"),
            "numpy": ("conda-forge", "numpy", "numpy"),
            "pillow": ("conda-forge", "pillow", "PIL"),
            "pyyaml": ("conda-forge", "PyYAML", "yaml"),
            "psutil": ("conda-forge", "psutil", "psutil"),
            "onnxruntime": ("conda-forge", "onnxruntime", "onnxruntime"),
            "mediapipe": ("conda-forge", "mediapipe", "mediapipe"),
            "diffusers": ("conda-forge", "diffusers", "diffusers"),
            "safetensors": ("conda-forge", "safetensors", "safetensors"),
            "huggingface_hub": ("conda-forge", "huggingface_hub", "huggingface_hub"),
            "tokenizers": ("conda-forge", "tokenizers", "tokenizers"),
            "scipy": ("conda-forge", "scipy", "scipy"),
            "matplotlib": ("conda-forge", "matplotlib", "matplotlib"),
            "scikit-image": ("conda-forge", "scikit-image", "skimage"),
            "tqdm": ("conda-forge", "tqdm", "tqdm"),
            "requests": ("conda-forge", "requests", "requests")
        }
    
    def check_conda_environment(self) -> bool:
        """conda 환경 확인"""
        log_info("Conda 환경 확인 중...")
        
        success, stdout, stderr = run_command("conda info --json")
        if not success:
            log_error("conda 명령어를 찾을 수 없습니다")
            return False
        
        try:
            conda_info = json.loads(stdout)
            active_env = conda_info.get("active_prefix", "")
            env_name = conda_info.get("active_prefix_name", "")
            
            if "mycloset-ai" in active_env or "mycloset-ai" in env_name:
                log_success(f"mycloset-ai 환경 활성화됨: {active_env}")
                return True
            else:
                log_warning(f"현재 환경: {env_name}")
                log_warning("mycloset-ai 환경이 활성화되지 않음")
                return False
                
        except json.JSONDecodeError:
            log_error("conda 정보 파싱 실패")
            return False
    
    def get_conda_packages(self) -> Dict[str, dict]:
        """conda로 설치된 패키지 목록 가져오기"""
        log_info("Conda 패키지 목록 조회 중...")
        
        success, stdout, stderr = run_command("conda list --json")
        if not success:
            log_error(f"conda list 실패: {stderr}")
            return {}
        
        try:
            packages = json.loads(stdout)
            conda_packages = {}
            
            for pkg in packages:
                name = pkg.get("name", "")
                version = pkg.get("version", "")
                channel = pkg.get("channel", "")
                
                conda_packages[name] = {
                    "version": version,
                    "channel": channel,
                    "is_pip": channel == "pypi"
                }
            
            log_success(f"총 {len(conda_packages)}개 패키지 발견")
            return conda_packages
            
        except json.JSONDecodeError:
            log_error("conda 패키지 목록 파싱 실패")
            return {}
    
    def test_import_packages(self) -> Dict[str, dict]:
        """실제 import 테스트"""
        log_info("패키지 import 테스트 중...")
        
        import_results = {}
        
        for pkg_name, (conda_channel, pip_name, import_name) in self.required_packages.items():
            try:
                # import 테스트
                exec(f"import {import_name}")
                
                # 버전 확인
                version = "unknown"
                try:
                    version_cmd = f"import {import_name}; print({import_name}.__version__)"
                    exec(version_cmd)
                except:
                    try:
                        version_cmd = f"import {import_name}; print({import_name}.version)"
                        exec(version_cmd)
                    except:
                        pass
                
                import_results[pkg_name] = {
                    "status": "success",
                    "version": version,
                    "import_name": import_name
                }
                
            except ImportError as e:
                import_results[pkg_name] = {
                    "status": "missing",
                    "error": str(e),
                    "import_name": import_name
                }
            except Exception as e:
                import_results[pkg_name] = {
                    "status": "error",
                    "error": str(e),
                    "import_name": import_name
                }
        
        return import_results
    
    def analyze_packages(self) -> Dict[str, List[str]]:
        """패키지 분석 및 분류"""
        log_info("패키지 상태 분석 중...")
        
        conda_packages = self.get_conda_packages()
        import_results = self.test_import_packages()
        
        analysis = {
            "conda_installed": [],
            "pip_installed": [],
            "missing": [],
            "broken": []
        }
        
        for pkg_name, (conda_channel, pip_name, import_name) in self.required_packages.items():
            import_status = import_results.get(pkg_name, {})
            
            if import_status.get("status") == "success":
                # 설치 방식 확인
                if any(pkg_name in name or pip_name.lower() in name.lower() 
                      for name in conda_packages.keys()):
                    # conda 패키지 확인
                    for conda_name, conda_info in conda_packages.items():
                        if (pkg_name in conda_name.lower() or 
                            pip_name.lower() in conda_name.lower() or
                            import_name in conda_name.lower()):
                            
                            if conda_info["is_pip"]:
                                analysis["pip_installed"].append(f"{pkg_name} ({conda_info['version']})")
                            else:
                                analysis["conda_installed"].append(f"{pkg_name} ({conda_info['version']})")
                            break
                    else:
                        analysis["conda_installed"].append(f"{pkg_name} (버전 확인 불가)")
                else:
                    analysis["pip_installed"].append(f"{pkg_name} (설치 방식 불명)")
                    
            elif import_status.get("status") == "missing":
                analysis["missing"].append(pkg_name)
            else:
                analysis["broken"].append(f"{pkg_name}: {import_status.get('error', '')}")
        
        return analysis
    
    def install_missing_packages(self, missing_packages: List[str]) -> bool:
        """누락된 패키지 설치"""
        if not missing_packages:
            log_success("설치할 패키지가 없습니다")
            return True
        
        log_info(f"누락된 패키지 설치 시작: {missing_packages}")
        
        success_count = 0
        
        for pkg_name in missing_packages:
            if pkg_name not in self.required_packages:
                continue
                
            conda_channel, pip_name, import_name = self.required_packages[pkg_name]
            
            # 1. conda로 설치 시도
            log_info(f"{pkg_name} conda 설치 시도...")
            conda_cmd = f"conda install -c {conda_channel} {pkg_name} -y"
            success, stdout, stderr = run_command(conda_cmd)
            
            if success:
                log_success(f"{pkg_name} conda 설치 완료")
                success_count += 1
                continue
            
            # 2. pip으로 설치 시도
            log_info(f"{pkg_name} pip 설치 시도...")
            pip_cmd = f"pip install {pip_name}"
            success, stdout, stderr = run_command(pip_cmd)
            
            if success:
                log_success(f"{pkg_name} pip 설치 완료")
                success_count += 1
            else:
                log_error(f"{pkg_name} 설치 실패: {stderr}")
        
        log_info(f"설치 완료: {success_count}/{len(missing_packages)}")
        return success_count == len(missing_packages)
    
    def fix_opencv_mediapipe(self) -> bool:
        """OpenCV와 MediaPipe 호환성 문제 해결"""
        log_info("OpenCV/MediaPipe 호환성 문제 해결 중...")
        
        # 현재 OpenCV 상태 확인
        try:
            import cv2
            if hasattr(cv2, 'cvtColor'):
                log_success("OpenCV cvtColor 함수 정상 동작")
                return True
        except:
            pass
        
        # OpenCV 재설치
        log_info("OpenCV 재설치 시도...")
        
        # 1. conda로 opencv 설치
        success, stdout, stderr = run_command("conda install -c conda-forge opencv -y")
        if success:
            log_success("conda opencv 설치 완료")
            return True
        
        # 2. pip으로 특정 버전 설치
        log_info("pip opencv 설치 시도...")
        success, stdout, stderr = run_command("pip install opencv-python==4.8.1.78")
        if success:
            log_success("pip opencv 설치 완료")
            return True
        
        log_error("OpenCV 설치 실패")
        return False
    
    def generate_report(self) -> str:
        """설치 상태 리포트 생성"""
        analysis = self.analyze_packages()
        
        report = []
        report.append("🔍 MyCloset AI - Conda 패키지 상태 리포트")
        report.append("=" * 50)
        
        if analysis["conda_installed"]:
            report.append(f"\n✅ Conda로 설치된 패키지 ({len(analysis['conda_installed'])}개):")
            for pkg in analysis["conda_installed"]:
                report.append(f"   {pkg}")
        
        if analysis["pip_installed"]:
            report.append(f"\n📦 Pip으로 설치된 패키지 ({len(analysis['pip_installed'])}개):")
            for pkg in analysis["pip_installed"]:
                report.append(f"   {pkg}")
        
        if analysis["missing"]:
            report.append(f"\n❌ 누락된 패키지 ({len(analysis['missing'])}개):")
            for pkg in analysis["missing"]:
                report.append(f"   {pkg}")
        
        if analysis["broken"]:
            report.append(f"\n🔧 문제 있는 패키지 ({len(analysis['broken'])}개):")
            for pkg in analysis["broken"]:
                report.append(f"   {pkg}")
        
        report.append(f"\n📊 요약:")
        report.append(f"   정상: {len(analysis['conda_installed']) + len(analysis['pip_installed'])}")
        report.append(f"   누락: {len(analysis['missing'])}")
        report.append(f"   문제: {len(analysis['broken'])}")
        
        return "\n".join(report)

def main():
    """메인 실행"""
    print("🔍 MyCloset AI - Conda 패키지 체크")
    print("=" * 50)
    
    checker = CondaPackageChecker()
    
    # 1. conda 환경 확인
    if not checker.check_conda_environment():
        log_error("conda 환경 문제 발견")
        print("\n💡 해결 방법:")
        print("   conda activate mycloset-ai")
        return
    
    # 2. 패키지 분석
    analysis = checker.analyze_packages()
    
    # 3. 리포트 출력
    print(checker.generate_report())
    
    # 4. 누락 패키지 설치
    if analysis["missing"]:
        print(f"\n🔧 누락된 패키지 자동 설치를 시작하시겠습니까? (y/n): ", end="")
        if input().lower() == 'y':
            checker.install_missing_packages(analysis["missing"])
        else:
            print("수동 설치 명령어:")
            for pkg in analysis["missing"]:
                conda_channel, pip_name, _ = checker.required_packages[pkg]
                print(f"   conda install -c {conda_channel} {pkg} -y")
    
    # 5. OpenCV/MediaPipe 문제 해결
    if "opencv" in analysis["missing"] or "mediapipe" in analysis["missing"]:
        print(f"\n🔧 OpenCV/MediaPipe 호환성 문제를 해결하시겠습니까? (y/n): ", end="")
        if input().lower() == 'y':
            checker.fix_opencv_mediapipe()
    
    # 6. 최종 테스트
    print(f"\n🚀 설치 완료 후 다음 명령어로 테스트:")
    print("   python3 advanced_model_test.py")

if __name__ == "__main__":
    main()