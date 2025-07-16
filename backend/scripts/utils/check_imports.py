#!/usr/bin/env python3
"""
MyCloset AI 의존성 및 임포트 검증 스크립트
모든 필수 패키지의 설치 상태와 임포트 가능 여부를 확인합니다.
"""

import sys
import subprocess
import importlib
from typing import Dict, List, Tuple

class ImportChecker:
    """패키지 임포트 상태 검증기"""
    
    def __init__(self):
        # 제공된 코드에서 사용하는 주요 임포트들
        self.critical_imports = {
            # 포즈 추정 (step_02) 관련
            'mediapipe': 'MediaPipe 포즈 추정',
            'cv2': 'OpenCV 이미지 처리',
            'numpy': 'NumPy 수치 연산',
            'torch': 'PyTorch 딥러닝',
            'PIL': 'Pillow 이미지 처리',
            
            # 옷 워핑 (step_05) 관련
            'scipy': 'SciPy 과학 연산',
            'scipy.interpolate': 'SciPy 보간',
            'scipy.spatial.distance': 'SciPy 거리 계산',
            'sklearn': 'Scikit-learn 머신러닝',
            'sklearn.cluster': 'Scikit-learn 클러스터링',
            'skimage': 'Scikit-image 이미지 처리',
            'skimage.feature': 'Scikit-image 특징 추출',
            
            # 웹 프레임워크 관련
            'fastapi': 'FastAPI 웹 프레임워크',
            'uvicorn': 'Uvicorn ASGI 서버',
            'pydantic': 'Pydantic 데이터 검증',
            
            # 유틸리티
            'asyncio': '비동기 처리 (내장)',
            'logging': '로깅 (내장)',
            'json': 'JSON 처리 (내장)',
            'base64': 'Base64 인코딩 (내장)',
            'io': '입출력 (내장)',
            'time': '시간 처리 (내장)',
            'math': '수학 함수 (내장)',
            'os': '운영체제 인터페이스 (내장)',
            'pathlib': '경로 처리 (내장)'
        }
        
        # 설치 명령어 매핑
        self.install_commands = {
            'mediapipe': 'pip install mediapipe',
            'cv2': 'pip install opencv-python',
            'numpy': 'pip install numpy',
            'torch': 'pip install torch torchvision',
            'PIL': 'pip install Pillow',
            'scipy': 'pip install scipy',
            'sklearn': 'pip install scikit-learn',
            'skimage': 'pip install scikit-image',
            'fastapi': 'pip install fastapi',
            'uvicorn': 'pip install uvicorn[standard]',
            'pydantic': 'pip install pydantic'
        }
        
        self.results = {}
    
    def check_python_version(self) -> bool:
        """Python 버전 확인"""
        print("🐍 Python 버전 확인...")
        
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        print(f"   현재 버전: {version_str}")
        
        if version.major == 3 and version.minor >= 9:
            print("   ✅ Python 3.9+ 요구사항 충족")
            return True
        else:
            print("   ❌ Python 3.9 이상이 필요합니다")
            return False
    
    def check_package_installation(self, package_name: str) -> bool:
        """패키지 설치 여부 확인"""
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'show', package_name.split('.')[0]],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except:
            return False
    
    def check_import(self, module_name: str) -> Tuple[bool, str]:
        """모듈 임포트 테스트"""
        try:
            if '.' in module_name:
                # 서브모듈인 경우
                parent_module = module_name.split('.')[0]
                importlib.import_module(parent_module)
                importlib.import_module(module_name)
            else:
                importlib.import_module(module_name)
            return True, "성공"
        
        except ImportError as e:
            return False, f"ImportError: {str(e)}"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def check_specific_functionality(self) -> Dict[str, bool]:
        """특정 기능 테스트"""
        tests = {}
        
        # MediaPipe 포즈 검출 테스트
        try:
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose()
            tests['mediapipe_pose'] = True
            pose.close()  # 리소스 정리
        except Exception as e:
            tests['mediapipe_pose'] = False
            print(f"   ⚠️ MediaPipe 포즈 초기화 실패: {e}")
        
        # PyTorch MPS (M3 Max) 테스트
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                tests['pytorch_mps'] = True
                print("   🚀 PyTorch MPS (M3 Max GPU) 사용 가능")
            else:
                tests['pytorch_mps'] = False
                print("   ⚠️ PyTorch MPS 사용 불가 (CPU 모드)")
        except Exception:
            tests['pytorch_mps'] = False
        
        # OpenCV 기능 테스트
        try:
            import cv2
            import numpy as np
            
            # 더미 이미지로 기본 기능 테스트
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            tests['opencv_basic'] = True
        except Exception as e:
            tests['opencv_basic'] = False
            print(f"   ⚠️ OpenCV 기본 기능 테스트 실패: {e}")
        
        # Scikit-image LBP 기능 테스트 (옷 워핑에서 사용)
        try:
            from skimage.feature import local_binary_pattern
            import numpy as np
            
            test_img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
            lbp = local_binary_pattern(test_img, 8, 1, method='uniform')
            tests['skimage_lbp'] = True
        except Exception as e:
            tests['skimage_lbp'] = False
            print(f"   ⚠️ Scikit-image LBP 기능 실패: {e}")
        
        return tests
    
    def run_full_check(self) -> Dict[str, any]:
        """전체 검증 실행"""
        print("🔍 MyCloset AI 의존성 검증 시작...\n")
        
        # 1. Python 버전 확인
        python_ok = self.check_python_version()
        print()
        
        # 2. 패키지별 임포트 테스트
        print("📦 패키지 임포트 테스트...")
        
        failed_imports = []
        success_count = 0
        
        for module_name, description in self.critical_imports.items():
            success, error_msg = self.check_import(module_name)
            
            if success:
                print(f"   ✅ {module_name:<25} - {description}")
                success_count += 1
            else:
                print(f"   ❌ {module_name:<25} - {description}")
                print(f"      └─ {error_msg}")
                failed_imports.append(module_name)
                
                # 설치 명령어 제안
                if module_name in self.install_commands:
                    print(f"      └─ 설치: {self.install_commands[module_name]}")
        
        print()
        
        # 3. 특정 기능 테스트
        print("🧪 핵심 기능 테스트...")
        functionality_tests = self.check_specific_functionality()
        
        for test_name, result in functionality_tests.items():
            status = "✅" if result else "❌"
            print(f"   {status} {test_name}")
        
        print()
        
        # 4. 결과 요약
        total_packages = len(self.critical_imports)
        success_rate = (success_count / total_packages) * 100
        
        print("📊 검증 결과 요약:")
        print(f"   📦 패키지 성공률: {success_count}/{total_packages} ({success_rate:.1f}%)")
        
        if failed_imports:
            print(f"   ❌ 실패한 패키지: {', '.join(failed_imports)}")
            print("\n🔧 설치 명령어:")
            
            for failed_module in failed_imports:
                if failed_module in self.install_commands:
                    print(f"   {self.install_commands[failed_module]}")
        
        else:
            print("   🎉 모든 필수 패키지가 정상적으로 임포트됩니다!")
        
        # 5. 권장사항
        print("\n💡 권장사항:")
        
        if not functionality_tests.get('pytorch_mps', False):
            print("   • PyTorch MPS 활성화로 M3 Max GPU 성능 향상 가능")
        
        if not functionality_tests.get('mediapipe_pose', False):
            print("   • MediaPipe 재설치 필요: pip uninstall mediapipe && pip install mediapipe")
        
        if success_rate < 80:
            print("   • 가상환경 재생성 권장: python -m venv fresh_env")
        
        return {
            'python_version_ok': python_ok,
            'success_rate': success_rate,
            'failed_imports': failed_imports,
            'functionality_tests': functionality_tests,
            'ready_for_development': python_ok and success_rate >= 80
        }
    
    def generate_requirements_txt(self) -> str:
        """실제 설치된 패키지 버전으로 requirements.txt 생성"""
        print("\n📝 현재 환경 기반 requirements.txt 생성...")
        
        requirements = []
        
        # 성공적으로 임포트된 패키지들의 버전 확인
        for module_name in self.critical_imports.keys():
            if module_name in ['asyncio', 'logging', 'json', 'base64', 'io', 'time', 'math', 'os', 'pathlib']:
                continue  # 내장 모듈 스킵
            
            success, _ = self.check_import(module_name)
            if success:
                try:
                    # 패키지명 매핑
                    package_map = {
                        'cv2': 'opencv-python',
                        'PIL': 'Pillow',
                        'sklearn': 'scikit-learn',
                        'skimage': 'scikit-image'
                    }
                    
                    package_name = package_map.get(module_name.split('.')[0], module_name.split('.')[0])
                    
                    result = subprocess.run(
                        [sys.executable, '-m', 'pip', 'show', package_name],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        for line in result.stdout.split('\n'):
                            if line.startswith('Version:'):
                                version = line.split(':')[1].strip()
                                requirements.append(f"{package_name}=={version}")
                                break
                
                except Exception:
                    requirements.append(package_name)  # 버전 없이 추가
        
        requirements_content = '\n'.join(sorted(requirements))
        
        try:
            with open('requirements_current.txt', 'w') as f:
                f.write(requirements_content)
            print("   ✅ requirements_current.txt 파일 생성됨")
        except Exception as e:
            print(f"   ❌ 파일 생성 실패: {e}")
        
        return requirements_content


def main():
    """메인 실행 함수"""
    checker = ImportChecker()
    
    try:
        results = checker.run_full_check()
        
        # requirements.txt 생성
        if results['success_rate'] > 50:
            checker.generate_requirements_txt()
        
        # 종료 상태
        if results['ready_for_development']:
            print("\n🎉 개발 환경이 준비되었습니다!")
            sys.exit(0)
        else:
            print("\n⚠️ 일부 패키지 설치가 필요합니다.")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n❌ 검증이 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()