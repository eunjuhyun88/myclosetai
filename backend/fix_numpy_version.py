#!/usr/bin/env python3
"""
🔧 NumPy 버전 호환성 문제 해결
- NumPy 2.x와 1.x 충돌 해결
- MediaPipe와 Transformers 호환성 보장
"""

import subprocess
import sys
import os

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

def check_current_versions():
    """현재 버전 확인"""
    log_info("현재 패키지 버전 확인...")
    
    try:
        import numpy as np
        import torch
        import transformers
        import mediapipe as mp
        
        print(f"NumPy: {np.__version__}")
        print(f"PyTorch: {torch.__version__}")
        print(f"Transformers: {transformers.__version__}")
        print(f"MediaPipe: {mp.__version__}")
        
        # NumPy 버전 체크
        numpy_version = np.__version__
        if numpy_version.startswith("1."):
            log_warning("NumPy 1.x 버전 - 호환성 문제 가능성")
            return False
        elif numpy_version.startswith("2."):
            log_success("NumPy 2.x 버전")
            return True
        else:
            log_warning(f"알 수 없는 NumPy 버전: {numpy_version}")
            return False
            
    except ImportError as e:
        log_error(f"패키지 import 실패: {e}")
        return False

def fix_numpy_compatibility():
    """NumPy 호환성 문제 해결"""
    log_info("NumPy 호환성 문제 해결 중...")
    
    # 방법 1: NumPy 2.x로 업그레이드 (가장 확실)
    log_info("방법 1: NumPy 2.x 업그레이드")
    success, stdout, stderr = run_command("pip install 'numpy>=2.0,<3.0'")
    if success:
        log_success("NumPy 2.x 업그레이드 완료")
        if test_imports():
            return True
    
    # 방법 2: 모든 패키지 강제 재설치
    log_info("방법 2: 핵심 패키지 강제 재설치")
    packages_to_reinstall = [
        "numpy>=2.0",
        "torch",
        "transformers", 
        "safetensors",
        "tokenizers"
    ]
    
    for package in packages_to_reinstall:
        log_info(f"{package} 재설치 중...")
        success, stdout, stderr = run_command(f"pip install --force-reinstall {package}")
        if not success:
            log_error(f"{package} 재설치 실패: {stderr}")
    
    if test_imports():
        return True
    
    # 방법 3: MediaPipe 호환 버전으로 NumPy 다운그레이드 + transformers 재컴파일
    log_info("방법 3: 호환성 우선 다운그레이드")
    success, stdout, stderr = run_command("pip install 'numpy>=1.21,<2.0'")
    if success:
        # transformers를 NumPy 1.x와 호환되도록 재설치
        run_command("pip uninstall transformers -y")
        run_command("pip install transformers --no-cache-dir")
        
        if test_imports():
            return True
    
    return False

def fix_mediapipe_numpy():
    """MediaPipe NumPy 호환성 특별 처리"""
    log_info("MediaPipe NumPy 호환성 특별 처리...")
    
    # MediaPipe 재설치 (NumPy 2.x 호환 버전)
    success, stdout, stderr = run_command("pip uninstall mediapipe -y")
    success, stdout, stderr = run_command("pip install mediapipe --no-deps")
    
    # 의존성 수동 설치
    deps = [
        "numpy>=2.0",
        "opencv-python", 
        "protobuf",
        "attrs"
    ]
    
    for dep in deps:
        run_command(f"pip install {dep}")
    
    return test_imports()

def test_imports():
    """패키지 import 테스트"""
    log_info("패키지 import 테스트...")
    
    try:
        # NumPy 테스트
        import numpy as np
        test_array = np.zeros((10, 10))
        log_success(f"NumPy {np.__version__} 정상")
        
        # PyTorch 테스트
        import torch
        test_tensor = torch.zeros(10, 10)
        log_success(f"PyTorch {torch.__version__} 정상")
        
        # Transformers 테스트
        from transformers import SegformerImageProcessor
        log_success("Transformers 정상")
        
        # MediaPipe 테스트
        import mediapipe as mp
        import cv2
        mp_pose = mp.solutions.pose
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        with mp_pose.Pose(static_image_mode=True) as pose:
            rgb_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_img)
        log_success(f"MediaPipe {mp.__version__} 정상")
        
        return True
        
    except Exception as e:
        log_error(f"Import 테스트 실패: {e}")
        return False

def create_compatibility_test():
    """호환성 테스트 스크립트 생성"""
    test_script = '''#!/usr/bin/env python3
"""NumPy/PyTorch/Transformers/MediaPipe 호환성 테스트"""

def test_all_packages():
    try:
        # NumPy 테스트
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        # PyTorch 테스트  
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        if torch.backends.mps.is_available():
            print("✅ MPS 지원 가능")
        
        # Transformers 테스트
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        print("✅ Transformers 정상")
        
        # CLIP 테스트
        from transformers import CLIPModel, CLIPProcessor
        print("✅ CLIP 정상")
        
        # MediaPipe 테스트
        import mediapipe as mp
        import cv2
        
        mp_pose = mp.solutions.pose
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        with mp_pose.Pose(static_image_mode=True) as pose:
            rgb_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_img)
        
        print(f"✅ MediaPipe {mp.__version__}")
        
        print("\\n🎉 모든 패키지 호환성 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    test_all_packages()
'''
    
    with open("test_compatibility.py", "w") as f:
        f.write(test_script)
    
    os.chmod("test_compatibility.py", 0o755)
    log_success("호환성 테스트 스크립트 생성: test_compatibility.py")

def main():
    """메인 실행"""
    print("🔧 NumPy 호환성 문제 해결")
    print("=" * 40)
    
    # 1. 현재 상태 확인
    print("\n1️⃣ 현재 버전 확인")
    print("=" * 20)
    if check_current_versions():
        log_success("NumPy 버전 정상")
        if test_imports():
            log_success("모든 패키지 정상 - 수정 불필요")
            return
    
    # 2. 호환성 문제 해결
    print("\n2️⃣ 호환성 문제 해결")
    print("=" * 20)
    if fix_numpy_compatibility():
        log_success("NumPy 호환성 문제 해결 완료")
    else:
        log_warning("표준 방법 실패 - 특별 처리 시도")
        if fix_mediapipe_numpy():
            log_success("MediaPipe 특별 처리 완료")
        else:
            log_error("모든 방법 실패")
    
    # 3. 테스트 스크립트 생성
    print("\n3️⃣ 테스트 스크립트 생성")
    print("=" * 20)
    create_compatibility_test()
    
    # 4. 최종 테스트
    print("\n4️⃣ 최종 테스트")
    print("=" * 20)
    success, stdout, stderr = run_command("python3 test_compatibility.py")
    if success:
        print(stdout)
        log_success("🚀 이제 다시 테스트하세요: python3 advanced_model_test.py")
    else:
        print(stderr)
        log_error("최종 테스트 실패")

if __name__ == "__main__":
    main()