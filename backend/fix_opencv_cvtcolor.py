#!/usr/bin/env python3
"""
🔧 OpenCV cv2.cvtColor 문제 완전 해결 스크립트
- OpenCV 버전 호환성 문제 해결
- conda 환경에서 안전한 재설치
- MediaPipe 호환성 보장
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def log_info(msg: str):
    print(f"ℹ️  {msg}")

def log_success(msg: str):
    print(f"✅ {msg}")

def log_error(msg: str):
    print(f"❌ {msg}")

def log_warning(msg: str):
    print(f"⚠️  {msg}")

def run_command(cmd: str) -> tuple:
    """명령어 실행"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "시간 초과"
    except Exception as e:
        return False, "", str(e)

class OpenCVFixer:
    """OpenCV cv2.cvtColor 문제 해결"""
    
    def __init__(self):
        self.opencv_packages = [
            "opencv-python",
            "opencv-contrib-python", 
            "opencv-python-headless",
            "libopencv",
            "py-opencv",
            "opencv"
        ]
        
    def diagnose_opencv_problem(self):
        """OpenCV 문제 진단"""
        log_info("OpenCV 문제 진단 중...")
        
        # 1. OpenCV 설치 확인
        try:
            import cv2
            log_success("OpenCV 모듈 import 성공")
            
            # 버전 확인
            version = getattr(cv2, '__version__', 'unknown')
            log_info(f"OpenCV 버전: {version}")
            
            # 함수 확인
            if hasattr(cv2, 'cvtColor'):
                log_success("cv2.cvtColor 함수 존재")
                
                # 실제 동작 테스트
                import numpy as np
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                try:
                    result = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
                    log_success("cv2.cvtColor 함수 정상 동작")
                    return True
                except Exception as e:
                    log_error(f"cv2.cvtColor 함수 실행 오류: {e}")
                    return False
            else:
                log_error("cv2.cvtColor 함수 없음")
                return False
                
        except ImportError as e:
            log_error(f"OpenCV import 실패: {e}")
            return False
    
    def get_installed_opencv_packages(self):
        """설치된 OpenCV 패키지 확인"""
        log_info("설치된 OpenCV 패키지 확인 중...")
        
        # conda list로 확인
        success, stdout, stderr = run_command("conda list | grep -i opencv")
        conda_opencv = []
        if success and stdout:
            conda_opencv = [line.strip() for line in stdout.split('\n') if line.strip()]
        
        # pip list로 확인  
        success, stdout, stderr = run_command("pip list | grep -i opencv")
        pip_opencv = []
        if success and stdout:
            pip_opencv = [line.strip() for line in stdout.split('\n') if line.strip()]
        
        print("\n📦 설치된 OpenCV 패키지:")
        if conda_opencv:
            print("   Conda 패키지:")
            for pkg in conda_opencv:
                print(f"     {pkg}")
        if pip_opencv:
            print("   Pip 패키지:")
            for pkg in pip_opencv:
                print(f"     {pkg}")
        
        return conda_opencv, pip_opencv
    
    def clean_opencv_packages(self):
        """기존 OpenCV 패키지 완전 제거"""
        log_info("기존 OpenCV 패키지 완전 제거 중...")
        
        # conda 패키지 제거
        for pkg in self.opencv_packages:
            log_info(f"conda {pkg} 제거 시도...")
            success, stdout, stderr = run_command(f"conda remove {pkg} -y")
            if success:
                log_success(f"conda {pkg} 제거 완료")
        
        # pip 패키지 제거
        for pkg in self.opencv_packages:
            log_info(f"pip {pkg} 제거 시도...")
            success, stdout, stderr = run_command(f"pip uninstall {pkg} -y")
            if success:
                log_success(f"pip {pkg} 제거 완료")
        
        # 캐시 정리
        run_command("conda clean -a -y")
        run_command("pip cache purge")
    
    def install_opencv_conda_forge(self):
        """conda-forge에서 OpenCV 설치"""
        log_info("conda-forge에서 OpenCV 설치 중...")
        
        # 방법 1: conda-forge opencv
        success, stdout, stderr = run_command("conda install -c conda-forge opencv -y")
        if success:
            log_success("conda-forge opencv 설치 완료")
            return self.test_opencv_installation()
        
        # 방법 2: conda-forge libopencv + py-opencv
        log_info("대안 방법: libopencv + py-opencv")
        success, stdout, stderr = run_command("conda install -c conda-forge libopencv py-opencv -y")
        if success:
            log_success("conda-forge libopencv + py-opencv 설치 완료")
            return self.test_opencv_installation()
        
        return False
    
    def install_opencv_pip_specific(self):
        """특정 버전 OpenCV pip 설치"""
        log_info("특정 버전 OpenCV pip 설치 중...")
        
        # MediaPipe와 호환되는 OpenCV 버전들
        opencv_versions = [
            "4.8.1.78",   # 안정적인 버전
            "4.9.0.80",   # 최신 안정 버전
            "4.7.1.84",   # 구버전 안정
        ]
        
        for version in opencv_versions:
            log_info(f"OpenCV {version} 설치 시도...")
            success, stdout, stderr = run_command(f"pip install opencv-python=={version}")
            if success:
                if self.test_opencv_installation():
                    log_success(f"OpenCV {version} 설치 및 테스트 성공")
                    return True
                else:
                    log_warning(f"OpenCV {version} 설치되었지만 테스트 실패")
        
        return False
    
    def install_opencv_headless(self):
        """OpenCV headless 버전 설치"""
        log_info("OpenCV headless 버전 설치 중...")
        
        success, stdout, stderr = run_command("pip install opencv-python-headless")
        if success:
            log_success("OpenCV headless 설치 완료")
            return self.test_opencv_installation()
        
        return False
    
    def test_opencv_installation(self):
        """OpenCV 설치 테스트"""
        try:
            import cv2
            import numpy as np
            
            # 기본 함수 테스트
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            result = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
            
            # 추가 함수 테스트
            gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            
            log_success("OpenCV 설치 테스트 통과")
            return True
            
        except Exception as e:
            log_error(f"OpenCV 테스트 실패: {e}")
            return False
    
    def fix_mediapipe_opencv_compatibility(self):
        """MediaPipe-OpenCV 호환성 보장"""
        log_info("MediaPipe-OpenCV 호환성 확인 중...")
        
        try:
            import mediapipe as mp
            import cv2
            import numpy as np
            
            # MediaPipe 포즈 모델 테스트
            mp_pose = mp.solutions.pose
            
            # 더미 이미지 생성
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5
            ) as pose:
                # BGR to RGB 변환 테스트
                rgb_image = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_image)
                
                log_success("MediaPipe-OpenCV 호환성 확인 완료")
                return True
                
        except Exception as e:
            log_error(f"MediaPipe-OpenCV 호환성 문제: {e}")
            return False
    
    def create_opencv_test_script(self):
        """OpenCV 테스트 스크립트 생성"""
        test_script = '''#!/usr/bin/env python3
"""OpenCV 기능 테스트 스크립트"""

import cv2
import numpy as np
import sys

def test_opencv_functions():
    """OpenCV 주요 함수 테스트"""
    try:
        print(f"OpenCV 버전: {cv2.__version__}")
        
        # 이미지 생성
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[:, :, 0] = 255  # 파란색
        
        # 색상 변환 테스트
        rgb_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        hsv_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2HSV)
        
        print("✅ cv2.cvtColor 함수 정상 동작")
        
        # 기본 함수들 테스트
        resized = cv2.resize(test_img, (50, 50))
        blurred = cv2.GaussianBlur(test_img, (5, 5), 0)
        
        print("✅ cv2.resize, cv2.GaussianBlur 정상 동작")
        
        # MediaPipe 호환성 테스트
        try:
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            
            with mp_pose.Pose(static_image_mode=True) as pose:
                results = pose.process(rgb_img)
                
            print("✅ MediaPipe 호환성 확인")
            
        except ImportError:
            print("⚠️ MediaPipe 설치되지 않음")
        except Exception as e:
            print(f"❌ MediaPipe 호환성 문제: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ OpenCV 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    if test_opencv_functions():
        print("🎉 OpenCV 모든 테스트 통과!")
        sys.exit(0)
    else:
        print("💥 OpenCV 테스트 실패!")
        sys.exit(1)
'''
        
        with open("test_opencv.py", "w") as f:
            f.write(test_script)
        
        os.chmod("test_opencv.py", 0o755)
        log_success("OpenCV 테스트 스크립트 생성: test_opencv.py")

def main():
    """메인 실행 함수"""
    print("🔧 OpenCV cv2.cvtColor 문제 완전 해결")
    print("=" * 50)
    
    fixer = OpenCVFixer()
    
    # 1. 현재 상태 진단
    print("\n1️⃣ 현재 OpenCV 상태 진단")
    print("=" * 30)
    
    if fixer.diagnose_opencv_problem():
        log_success("OpenCV가 정상적으로 동작합니다!")
        return
    
    # 2. 설치된 패키지 확인
    print("\n2️⃣ 설치된 OpenCV 패키지 확인")
    print("=" * 30)
    fixer.get_installed_opencv_packages()
    
    # 3. 기존 패키지 제거
    print("\n3️⃣ 기존 OpenCV 패키지 제거")
    print("=" * 30)
    fixer.clean_opencv_packages()
    
    # 4. 새로 설치 시도
    print("\n4️⃣ OpenCV 새로 설치")
    print("=" * 30)
    
    install_methods = [
        ("conda-forge 방식", fixer.install_opencv_conda_forge),
        ("pip 특정 버전", fixer.install_opencv_pip_specific),
        ("headless 버전", fixer.install_opencv_headless)
    ]
    
    for method_name, method_func in install_methods:
        log_info(f"{method_name} 시도...")
        if method_func():
            log_success(f"{method_name} 성공!")
            break
    else:
        log_error("모든 설치 방법 실패")
        return
    
    # 5. MediaPipe 호환성 확인
    print("\n5️⃣ MediaPipe 호환성 확인")
    print("=" * 30)
    if fixer.fix_mediapipe_opencv_compatibility():
        log_success("MediaPipe-OpenCV 호환성 확인 완료!")
    else:
        log_warning("MediaPipe 호환성 문제 있음")
    
    # 6. 테스트 스크립트 생성
    print("\n6️⃣ 테스트 스크립트 생성")
    print("=" * 30)
    fixer.create_opencv_test_script()
    
    # 7. 최종 테스트 실행
    print("\n7️⃣ 최종 테스트 실행")
    print("=" * 30)
    success, stdout, stderr = run_command("python3 test_opencv.py")
    if success:
        print(stdout)
        log_success("🎉 OpenCV 완전 복구 완료!")
        print("\n🚀 이제 다음 명령어로 테스트하세요:")
        print("   python3 advanced_model_test.py")
    else:
        print(stderr)
        log_error("최종 테스트 실패")
        print("\n💡 수동 해결 방법:")
        print("   1. conda activate mycloset-ai")
        print("   2. conda install -c conda-forge opencv -y")
        print("   3. python3 test_opencv.py")

if __name__ == "__main__":
    main()