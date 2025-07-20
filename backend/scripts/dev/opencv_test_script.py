#!/usr/bin/env python3
"""
OpenCV cv2.data 속성 완전 해결 스크립트
conda-forge OpenCV의 cv2.data 누락 문제 해결
"""

import os
import sys
import subprocess
from pathlib import Path

def create_opencv_data_fix():
    """cv2.data 속성 직접 패치"""
    
    # backend/app/utils 디렉토리 생성
    utils_dir = Path(__file__).parent.parent / "app" / "utils"
    utils_dir.mkdir(parents=True, exist_ok=True)
    
    # __init__.py 파일 생성
    (utils_dir / "__init__.py").touch()
    
    # opencv_data_patch.py 생성
    patch_content = '''"""
OpenCV cv2.data 속성 패치 모듈
conda-forge OpenCV에서 누락된 cv2.data 속성을 복구합니다.
"""

import cv2
import os
from pathlib import Path
from typing import Optional

def find_opencv_data_path() -> Optional[str]:
    """OpenCV 데이터 파일 경로 찾기"""
    
    # 1. conda 환경에서 찾기
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        conda_data_paths = [
            Path(conda_prefix) / "share" / "opencv4" / "haarcascades",
            Path(conda_prefix) / "share" / "OpenCV" / "haarcascades",
            Path(conda_prefix) / "lib" / "python3.11" / "site-packages" / "cv2" / "data",
        ]
        
        for path in conda_data_paths:
            if path.exists() and (path / "haarcascade_frontalface_default.xml").exists():
                return str(path) + "/"
    
    # 2. 시스템 경로에서 찾기 (M3 Mac)
    system_paths = [
        "/opt/homebrew/share/opencv4/haarcascades/",
        "/usr/local/share/opencv4/haarcascades/",
        "/usr/share/opencv4/haarcascades/",
    ]
    
    for path_str in system_paths:
        path = Path(path_str)
        if path.exists() and (path / "haarcascade_frontalface_default.xml").exists():
            return path_str
    
    return None

def download_opencv_data():
    """OpenCV 데이터 파일 직접 다운로드"""
    try:
        import urllib.request
        
        # 데이터 저장 디렉토리
        data_dir = Path.home() / ".mycloset_opencv_data"
        data_dir.mkdir(exist_ok=True)
        
        # 필요한 cascade 파일 URL들
        base_url = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
        files_to_download = [
            "haarcascade_frontalface_default.xml",
            "haarcascade_eye.xml",
            "haarcascade_smile.xml"
        ]
        
        for filename in files_to_download:
            file_path = data_dir / filename
            if not file_path.exists():
                print(f"다운로드 중: {filename}")
                urllib.request.urlretrieve(f"{base_url}{filename}", file_path)
        
        return str(data_dir) + "/"
        
    except Exception as e:
        print(f"다운로드 실패: {e}")
        return None

class OpenCVDataPatch:
    """cv2.data 속성 패치 클래스"""
    
    def __init__(self):
        self._haarcascades_path = None
        self._find_data_path()
    
    def _find_data_path(self):
        """데이터 경로 찾기"""
        # 1. 설치된 경로에서 찾기
        path = find_opencv_data_path()
        if path:
            self._haarcascades_path = path
            return
        
        # 2. 직접 다운로드
        path = download_opencv_data()
        if path:
            self._haarcascades_path = path
            return
        
        # 3. 기본 경로 (파일이 없어도)
        self._haarcascades_path = "/tmp/opencv_data/"
        print("⚠️ OpenCV 데이터 파일을 찾을 수 없습니다. 기본 경로를 사용합니다.")
    
    @property
    def haarcascades(self) -> str:
        """haarcascades 경로 반환"""
        return self._haarcascades_path

# cv2.data 속성 패치 적용
def patch_cv2_data():
    """cv2.data 속성 패치"""
    if not hasattr(cv2, 'data'):
        cv2.data = OpenCVDataPatch()
        print("✅ cv2.data 속성 패치 적용됨")
    else:
        print("✅ cv2.data 속성이 이미 존재합니다")

# 자동 패치 (모듈 import 시 실행)
patch_cv2_data()

# 유틸리티 함수들
def get_cascade_path(cascade_name: str = "haarcascade_frontalface_default.xml") -> str:
    """Cascade 파일 경로 반환"""
    if hasattr(cv2, 'data'):
        return cv2.data.haarcascades + cascade_name
    else:
        # 패치 적용
        patch_cv2_data()
        return cv2.data.haarcascades + cascade_name

def test_opencv_data():
    """OpenCV data 작동 테스트"""
    try:
        # cv2.data 테스트
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        
        # 파일 존재 확인
        if os.path.exists(path):
            print(f"✅ cv2.data 완전 복구: {path}")
            return True
        else:
            print(f"⚠️ cv2.data 경로 설정됨: {path} (파일 다운로드 필요)")
            return False
            
    except Exception as e:
        print(f"❌ cv2.data 테스트 실패: {e}")
        return False

# 테스트 실행
if __name__ == "__main__":
    print("🔧 OpenCV cv2.data 패치 테스트")
    test_opencv_data()
'''
    
    # 파일 저장
    patch_file = utils_dir / "opencv_data_patch.py"
    with open(patch_file, 'w', encoding='utf-8') as f:
        f.write(patch_content)
    
    print(f"✅ OpenCV 패치 파일 생성: {patch_file}")
    return str(patch_file)

def apply_opencv_fix():
    """OpenCV 문제 완전 해결"""
    print("🔧 OpenCV cv2.data 문제 완전 해결 시작...")
    
    # 1. 패치 파일 생성
    patch_file = create_opencv_data_fix()
    
    # 2. 패치 적용 테스트
    print("\n🧪 패치 적용 테스트...")
    test_code = '''
# 패치 모듈 import
from app.utils.opencv_data_patch import patch_cv2_data, test_opencv_data
import cv2

# 패치 강제 적용
patch_cv2_data()

# 테스트 실행
success = test_opencv_data()
print(f"패치 성공: {success}")

# 실제 사용 테스트
try:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    print(f"✅ 최종 테스트 성공: {cascade_path}")
except Exception as e:
    print(f"❌ 최종 테스트 실패: {e}")
'''
    
    # 테스트 스크립트 저장
    test_file = Path("test_opencv_fix.py")
    with open(test_file, 'w') as f:
        f.write(test_code)
    
    print(f"\n📋 다음 단계:")
    print(f"1. 패치 테스트: python {test_file}")
    print(f"2. 프로젝트에서 사용: from app.utils.opencv_data_patch import patch_cv2_data")
    print(f"3. 앱 시작 시 한 번 호출: patch_cv2_data()")
    
    return True

if __name__ == "__main__":
    apply_opencv_fix()