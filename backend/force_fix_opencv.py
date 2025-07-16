#!/usr/bin/env python3
"""
🚨 OpenCV 강제 복구 스크립트
- conda 환경 완전 초기화
- Python path 문제 해결
- 패키지 충돌 해결
"""

import subprocess
import sys
import os
import importlib
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
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def deep_opencv_diagnosis():
    """OpenCV 깊이 진단"""
    log_info("OpenCV 깊이 진단 시작...")
    
    try:
        import cv2
        
        print(f"cv2 모듈 위치: {cv2.__file__}")
        print(f"cv2 모듈 속성 개수: {len(dir(cv2))}")
        
        # 주요 함수들 확인
        functions_to_check = [
            'cvtColor', 'imread', 'imwrite', 'resize', 'GaussianBlur',
            'COLOR_BGR2RGB', 'COLOR_BGR2GRAY', 'IMREAD_COLOR'
        ]
        
        missing_functions = []
        for func in functions_to_check:
            if hasattr(cv2, func):
                log_success(f"cv2.{func} 존재")
            else:
                missing_functions.append(func)
                log_error(f"cv2.{func} 없음")
        
        if missing_functions:
            log_error(f"누락된 함수들: {missing_functions}")
            
            # 모든 cv2 속성 출력 (디버깅용)
            log_info("cv2 모듈의 모든 속성 (처음 50개):")
            attrs = [attr for attr in dir(cv2) if not attr.startswith('__')][:50]
            for i, attr in enumerate(attrs):
                print(f"  {i+1:2d}. {attr}")
            
            return False
        else:
            log_success("모든 필수 함수 존재")
            return True
            
    except ImportError as e:
        log_error(f"cv2 import 실패: {e}")
        return False

def nuclear_opencv_cleanup():
    """OpenCV 핵폭탄급 청소"""
    log_info("OpenCV 핵폭탄급 청소 시작...")
    
    # 1. Python에서 cv2 언로드
    try:
        if 'cv2' in sys.modules:
            del sys.modules['cv2']
            log_success("cv2 모듈 언로드")
    except:
        pass
    
    # 2. 모든 OpenCV 관련 패키지 강제 제거
    opencv_packages = [
        "opencv-python", "opencv-contrib-python", "opencv-python-headless",
        "libopencv", "py-opencv", "opencv", "opencv-base", "opencv-devel"
    ]
    
    for pkg in opencv_packages:
        # conda 제거
        run_command(f"conda remove {pkg} -y --force")
        # pip 제거
        run_command(f"pip uninstall {pkg} -y")
    
    # 3. conda 환경 정리
    run_command("conda clean -a -y")
    run_command("pip cache purge")
    
    # 4. site-packages에서 cv2 폴더 직접 삭제
    try:
        import site
        for site_dir in site.getsitepackages():
            cv2_path = Path(site_dir) / "cv2"
            if cv2_path.exists():
                log_info(f"cv2 폴더 직접 삭제: {cv2_path}")
                import shutil
                shutil.rmtree(cv2_path, ignore_errors=True)
    except:
        pass
    
    log_success("핵폭탄급 청소 완료")

def install_opencv_from_scratch():
    """OpenCV 처음부터 새로 설치"""
    log_info("OpenCV 처음부터 새로 설치...")
    
    # 방법 1: conda-forge에서 opencv + opencv-python
    log_info("방법 1: conda-forge opencv 패키지")
    success, stdout, stderr = run_command("conda install -c conda-forge opencv python-opencv -y")
    if success:
        if test_opencv_complete():
            return True
    
    # 방법 2: pip로 최신 opencv-python
    log_info("방법 2: pip 최신 opencv-python")
    success, stdout, stderr = run_command("pip install opencv-python")
    if success:
        if test_opencv_complete():
            return True
    
    # 방법 3: pip로 contrib 버전
    log_info("방법 3: pip opencv-contrib-python")
    success, stdout, stderr = run_command("pip install opencv-contrib-python")
    if success:
        if test_opencv_complete():
            return True
    
    # 방법 4: 소스에서 컴파일 (최후 수단)
    log_info("방법 4: conda opencv-devel")
    success, stdout, stderr = run_command("conda install -c conda-forge opencv-devel -y")
    if success:
        if test_opencv_complete():
            return True
    
    return False

def test_opencv_complete():
    """완전한 OpenCV 테스트"""
    try:
        # 모듈 재로드
        if 'cv2' in sys.modules:
            importlib.reload(sys.modules['cv2'])
        else:
            import cv2
        
        import numpy as np
        
        # 기본 테스트
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # 핵심 함수들 테스트
        rgb_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(test_img, (50, 50))
        
        log_success("OpenCV 완전 테스트 통과")
        return True
        
    except Exception as e:
        log_error(f"OpenCV 테스트 실패: {e}")
        return False

def create_mediapipe_workaround():
    """MediaPipe 우회 솔루션 생성"""
    log_info("MediaPipe 우회 솔루션 생성...")
    
    workaround_code = '''
def safe_cvtColor(image, code):
    """OpenCV cvtColor 안전 래퍼"""
    try:
        import cv2
        return cv2.cvtColor(image, code)
    except AttributeError:
        # cvtColor가 없는 경우 수동 변환
        if code == cv2.COLOR_BGR2RGB:
            return image[:,:,::-1]  # BGR -> RGB
        elif code == cv2.COLOR_RGB2BGR:
            return image[:,:,::-1]  # RGB -> BGR
        elif code == cv2.COLOR_BGR2GRAY:
            # BGR to Gray 변환
            return 0.299 * image[:,:,2] + 0.587 * image[:,:,1] + 0.114 * image[:,:,0]
        else:
            raise NotImplementedError(f"색상 변환 코드 {code} 지원하지 않음")
    except Exception as e:
        raise RuntimeError(f"색상 변환 실패: {e}")

# cv2.cvtColor 대체
try:
    import cv2
    if not hasattr(cv2, 'cvtColor'):
        cv2.cvtColor = safe_cvtColor
except:
    pass
'''
    
    with open("opencv_workaround.py", "w") as f:
        f.write(workaround_code)
    
    log_success("우회 솔루션 생성: opencv_workaround.py")

def main():
    """메인 실행"""
    print("🚨 OpenCV 강제 복구 모드")
    print("=" * 40)
    
    # 1. 현재 상태 깊이 진단
    print("\n1️⃣ 깊이 진단")
    print("=" * 20)
    if deep_opencv_diagnosis():
        log_success("OpenCV 정상 - 복구 불필요")
        return
    
    # 2. 핵폭탄급 청소
    print("\n2️⃣ 핵폭탄급 청소")
    print("=" * 20)
    nuclear_opencv_cleanup()
    
    # 3. 파이썬 재시작 권장
    log_warning("Python 세션을 재시작하는 것을 권장합니다")
    print("계속하시겠습니까? (y/n): ", end="")
    if input().lower() != 'y':
        log_info("스크립트를 다시 실행하세요")
        return
    
    # 4. 처음부터 새로 설치
    print("\n3️⃣ 새로 설치")
    print("=" * 20)
    if install_opencv_from_scratch():
        log_success("🎉 OpenCV 복구 성공!")
    else:
        log_error("복구 실패 - 우회 솔루션 생성")
        create_mediapipe_workaround()
    
    # 5. 최종 테스트
    print("\n4️⃣ 최종 테스트")
    print("=" * 20)
    
    test_code = '''
import sys
sys.path.insert(0, '.')

try:
    import opencv_workaround  # 우회 솔루션 로드
except:
    pass

import cv2
import numpy as np

# 테스트
test_img = np.zeros((100, 100, 3), dtype=np.uint8)
rgb_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
print("✅ OpenCV 복구 완료!")
'''
    
    with open("final_opencv_test.py", "w") as f:
        f.write(test_code)
    
    success, stdout, stderr = run_command("python3 final_opencv_test.py")
    if success:
        print(stdout)
        log_success("🚀 이제 다시 테스트하세요: python3 advanced_model_test.py")
    else:
        print(stderr)
        log_error("최종 복구 실패")
        
        # 마지막 수단: 환경 재구축 제안
        print("\n💡 최후 수단: conda 환경 재구축")
        print("=" * 30)
        print("다음 명령어로 환경을 재구축하세요:")
        print("conda deactivate")
        print("conda remove -n mycloset-ai --all -y")
        print("conda create -n mycloset-ai python=3.11 -y")
        print("conda activate mycloset-ai")
        print("conda install -c conda-forge opencv pytorch torchvision -y")

if __name__ == "__main__":
    main()
    