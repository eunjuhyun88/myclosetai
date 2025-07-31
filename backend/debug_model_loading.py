#!/usr/bin/env python3
"""
🔥 MyCloset AI 프로젝트 구조 디버깅 스크립트 v1.0
================================================================================
✅ 프로젝트 구조 자동 감지 및 검증
✅ 각 단계별 디버깅 및 파일 로딩 체크
✅ AI 모델 파일 존재 여부 및 크기 확인
✅ conda 환경 및 Python 패키지 검증
✅ 깔끔한 로그 출력 및 진행 상황 표시
✅ 문제 발견 시 해결 방안 제시
"""

import os
import sys
import json
import time
import logging
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class CheckStatus(Enum):
    """체크 상태"""
    PASS = "✅"
    FAIL = "❌" 
    WARN = "⚠️"
    INFO = "ℹ️"
    PROCESSING = "🔄"

@dataclass
class CheckResult:
    """체크 결과"""
    name: str
    status: CheckStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None

class ProjectDebugger:
    """프로젝트 구조 디버깅 클래스"""
    
    def __init__(self):
        self.project_root = None
        self.backend_root = None
        self.ai_models_root = None
        self.results: List[CheckResult] = []
        self.start_time = time.time()
        
        # 프로젝트 구조 자동 감지
        self._detect_project_structure()
        
    def _detect_project_structure(self):
        """프로젝트 구조 자동 감지"""
        current_dir = Path.cwd().resolve()
        
        # 현재 디렉토리부터 상위로 올라가며 mycloset-ai 찾기
        for path in [current_dir] + list(current_dir.parents):
            if path.name == 'mycloset-ai':
                self.project_root = path
                self.backend_root = path / 'backend'
                self.ai_models_root = path / 'backend' / 'ai_models'
                break
            elif (path / 'backend').exists() and (path / 'frontend').exists():
                self.project_root = path
                self.backend_root = path / 'backend'
                self.ai_models_root = path / 'backend' / 'ai_models'
                break
        
        # 폴백: 현재 위치에서 추정
        if not self.project_root:
            if 'backend' in str(current_dir):
                # backend 내부에서 실행 중인 경우
                temp = current_dir
                while temp.parent != temp:
                    if temp.name == 'backend':
                        self.project_root = temp.parent
                        self.backend_root = temp
                        self.ai_models_root = temp / 'ai_models'
                        break
                    temp = temp.parent
            else:
                # 프로젝트 루트로 추정
                self.project_root = current_dir
                self.backend_root = current_dir / 'backend'
                self.ai_models_root = current_dir / 'backend' / 'ai_models'

    def print_header(self):
        """헤더 출력"""
        print("=" * 80)
        print("🔥 MyCloset AI 프로젝트 구조 디버깅 스크립트 v1.0")
        print("=" * 80)
        print(f"🕒 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📍 실행 위치: {Path.cwd()}")
        print(f"🐍 Python 버전: {platform.python_version()}")
        print(f"🖥️ 운영체제: {platform.system()} {platform.release()}")
        print("=" * 80)

    def print_progress(self, step: int, total: int, message: str):
        """진행 상황 출력"""
        progress = int((step / total) * 30)
        bar = "█" * progress + "░" * (30 - progress)
        percentage = (step / total) * 100
        print(f"\r{CheckStatus.PROCESSING.value} [{bar}] {percentage:5.1f}% | {message}", end="", flush=True)

    def add_result(self, result: CheckResult):
        """결과 추가 및 출력"""
        self.results.append(result)
        print(f"\n{result.status.value} {result.name}: {result.message}")
        
        if result.details:
            for key, value in result.details.items():
                if isinstance(value, (int, float)):
                    print(f"   📊 {key}: {value}")
                else:
                    print(f"   📋 {key}: {value}")
        
        if result.recommendations:
            print("   💡 권장사항:")
            for rec in result.recommendations:
                print(f"      • {rec}")

    def check_01_project_structure(self):
        """1단계: 프로젝트 구조 확인"""
        print(f"\n{'='*50}")
        print("🏗️  1단계: 프로젝트 구조 확인")
        print(f"{'='*50}")
        
        # 프로젝트 루트 확인
        if self.project_root and self.project_root.exists():
            self.add_result(CheckResult(
                name="프로젝트 루트",
                status=CheckStatus.PASS,
                message=f"감지됨: {self.project_root}",
                details={"디렉토리": str(self.project_root)}
            ))
        else:
            self.add_result(CheckResult(
                name="프로젝트 루트",
                status=CheckStatus.FAIL,
                message="프로젝트 루트를 찾을 수 없습니다",
                recommendations=["mycloset-ai 디렉토리 내에서 스크립트를 실행하세요"]
            ))
            return
        
        # 핵심 디렉토리 확인
        core_dirs = {
            "백엔드": self.backend_root,
            "프론트엔드": self.project_root / "frontend",
            "AI 파이프라인": self.backend_root / "app" / "ai_pipeline",
            "AI 모델": self.ai_models_root
        }
        
        for name, path in core_dirs.items():
            if path and path.exists():
                self.add_result(CheckResult(
                    name=f"{name} 디렉토리",
                    status=CheckStatus.PASS,
                    message="존재함",
                    details={"경로": str(path)}
                ))
            else:
                status = CheckStatus.WARN if name == "AI 모델" else CheckStatus.FAIL
                self.add_result(CheckResult(
                    name=f"{name} 디렉토리",
                    status=status,
                    message="존재하지 않음",
                    details={"예상 경로": str(path) if path else "알 수 없음"},
                    recommendations=[f"{name} 디렉토리를 생성하거나 경로를 확인하세요"]
                ))

    def check_02_python_environment(self):
        """2단계: Python 환경 확인"""
        print(f"\n{'='*50}")
        print("🐍 2단계: Python 환경 확인")
        print(f"{'='*50}")
        
        # Python 버전 확인
        python_version = platform.python_version()
        major, minor = python_version.split('.')[:2]
        
        if int(major) >= 3 and int(minor) >= 8:
            self.add_result(CheckResult(
                name="Python 버전",
                status=CheckStatus.PASS,
                message=f"호환 가능: {python_version}",
                details={"버전": python_version}
            ))
        else:
            self.add_result(CheckResult(
                name="Python 버전",
                status=CheckStatus.WARN,
                message=f"권장 버전 미만: {python_version}",
                details={"현재": python_version, "권장": "3.8+"},
                recommendations=["Python 3.8 이상으로 업그레이드하세요"]
            ))
        
        # conda 환경 확인
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'none')
        if conda_env != 'none':
            status = CheckStatus.PASS if conda_env == 'mycloset-ai-clean' else CheckStatus.INFO
            self.add_result(CheckResult(
                name="conda 환경",
                status=status,
                message=f"활성화됨: {conda_env}",
                details={"환경명": conda_env}
            ))
        else:
            self.add_result(CheckResult(
                name="conda 환경",
                status=CheckStatus.WARN,
                message="conda 환경이 활성화되지 않음",
                recommendations=["conda activate mycloset-ai-clean 실행하세요"]
            ))
        
        # 가상환경 확인
        virtual_env = os.environ.get('VIRTUAL_ENV')
        if virtual_env:
            self.add_result(CheckResult(
                name="가상환경",
                status=CheckStatus.INFO,
                message=f"활성화됨: {Path(virtual_env).name}",
                details={"경로": virtual_env}
            ))

    def check_03_required_packages(self):
        """3단계: 필수 패키지 확인"""
        print(f"\n{'='*50}")
        print("📦 3단계: 필수 패키지 확인")
        print(f"{'='*50}")
        
        required_packages = [
            ("torch", "PyTorch"),
            ("torchvision", "TorchVision"),
            ("numpy", "NumPy"),
            ("PIL", "Pillow"),
            ("cv2", "OpenCV"),
            ("fastapi", "FastAPI"),
            ("pydantic", "Pydantic")
        ]
        
        for package, display_name in required_packages:
            try:
                if package == "PIL":
                    import PIL
                    version = PIL.__version__
                elif package == "cv2":
                    import cv2
                    version = cv2.__version__
                else:
                    module = __import__(package)
                    version = getattr(module, '__version__', 'Unknown')
                
                self.add_result(CheckResult(
                    name=f"{display_name} 패키지",
                    status=CheckStatus.PASS,
                    message="설치됨",
                    details={"버전": version}
                ))
            except ImportError:
                self.add_result(CheckResult(
                    name=f"{display_name} 패키지",
                    status=CheckStatus.FAIL,
                    message="설치되지 않음",
                    recommendations=[f"pip install {package} 또는 conda install {package} 실행하세요"]
                ))

    def check_04_ai_pipeline_structure(self):
        """4단계: AI 파이프라인 구조 확인"""
        print(f"\n{'='*50}")
        print("🤖 4단계: AI 파이프라인 구조 확인")
        print(f"{'='*50}")
        
        if not self.backend_root:
            self.add_result(CheckResult(
                name="AI 파이프라인 구조",
                status=CheckStatus.FAIL,
                message="백엔드 루트를 찾을 수 없어 확인 불가"
            ))
            return
        
        pipeline_dirs = [
            "app/ai_pipeline",
            "app/ai_pipeline/steps",
            "app/ai_pipeline/utils", 
            "app/ai_pipeline/interface",
            "app/core",
            "app/services"
        ]
        
        for dir_path in pipeline_dirs:
            full_path = self.backend_root / dir_path
            if full_path.exists():
                # 디렉토리 내 Python 파일 수 확인
                py_files = list(full_path.glob("*.py"))
                self.add_result(CheckResult(
                    name=f"파이프라인 디렉토리: {dir_path}",
                    status=CheckStatus.PASS,
                    message="존재함",
                    details={
                        "Python 파일 수": len(py_files),
                        "파일들": [f.name for f in py_files[:5]]  # 최대 5개만 표시
                    }
                ))
            else:
                self.add_result(CheckResult(
                    name=f"파이프라인 디렉토리: {dir_path}",
                    status=CheckStatus.WARN,
                    message="존재하지 않음",
                    details={"예상 경로": str(full_path)}
                ))

    def check_05_step_files(self):
        """5단계: Step 파일들 확인"""
        print(f"\n{'='*50}")
        print("📋 5단계: Step 파일들 확인")
        print(f"{'='*50}")
        
        if not self.backend_root:
            return
        
        steps_dir = self.backend_root / "app" / "ai_pipeline" / "steps"
        if not steps_dir.exists():
            self.add_result(CheckResult(
                name="Steps 디렉토리",
                status=CheckStatus.FAIL,
                message="steps 디렉토리를 찾을 수 없음",
                details={"경로": str(steps_dir)}
            ))
            return
        
        expected_steps = [
            "step_01_human_parsing.py",
            "step_02_pose_estimation.py", 
            "step_03_cloth_segmentation.py",
            "step_04_geometric_matching.py",
            "step_05_cloth_warping.py",
            "step_06_virtual_fitting.py",
            "step_07_post_processing.py",
            "step_08_quality_assessment.py",
            "__init__.py",
            "base_step_mixin.py"
        ]
        
        for step_file in expected_steps:
            file_path = steps_dir / step_file
            if file_path.exists():
                # 파일 크기 및 기본 구문 검사
                file_size = file_path.stat().st_size
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 기본 Python 구문 검사
                    try:
                        compile(content, str(file_path), 'exec')
                        syntax_ok = True
                    except SyntaxError as e:
                        syntax_ok = False
                        syntax_error = str(e)
                    
                    status = CheckStatus.PASS if syntax_ok else CheckStatus.WARN
                    message = "정상" if syntax_ok else f"구문 오류: {syntax_error}"
                    
                    self.add_result(CheckResult(
                        name=f"Step 파일: {step_file}",
                        status=status,
                        message=message,
                        details={
                            "파일 크기": f"{file_size:,} bytes",
                            "라인 수": len(content.split('\n')),
                            "구문 검사": "통과" if syntax_ok else "실패"
                        }
                    ))
                except Exception as e:
                    self.add_result(CheckResult(
                        name=f"Step 파일: {step_file}",
                        status=CheckStatus.WARN,
                        message=f"읽기 실패: {e}",
                        details={"파일 크기": f"{file_size:,} bytes"}
                    ))
            else:
                self.add_result(CheckResult(
                    name=f"Step 파일: {step_file}",
                    status=CheckStatus.FAIL,
                    message="파일이 존재하지 않음",
                    details={"예상 경로": str(file_path)}
                ))

    def check_06_ai_models(self):
        """6단계: AI 모델 파일들 확인"""
        print(f"\n{'='*50}")
        print("🧠 6단계: AI 모델 파일들 확인")
        print(f"{'='*50}")
        
        if not self.ai_models_root or not self.ai_models_root.exists():
            self.add_result(CheckResult(
                name="AI 모델 디렉토리",
                status=CheckStatus.WARN,
                message="AI 모델 디렉토리를 찾을 수 없음",
                details={"예상 경로": str(self.ai_models_root) if self.ai_models_root else "알 수 없음"},
                recommendations=["AI 모델을 다운로드하고 올바른 위치에 배치하세요"]
            ))
            return
        
        # AI 모델 파일 스캔
        model_extensions = ['.pth', '.pt', '.ckpt', '.safetensors', '.bin', '.onnx']
        model_files = []
        total_size = 0
        
        for ext in model_extensions:
            files = list(self.ai_models_root.rglob(f"*{ext}"))
            for file in files:
                if file.is_file():
                    size = file.stat().st_size
                    model_files.append((file, size))
                    total_size += size
        
        if model_files:
            total_size_gb = total_size / (1024**3)
            large_models = [f for f, s in model_files if s > 100 * 1024 * 1024]  # 100MB 이상
            
            self.add_result(CheckResult(
                name="AI 모델 파일들",
                status=CheckStatus.PASS,
                message=f"{len(model_files)}개 모델 파일 발견",
                details={
                    "총 파일 수": len(model_files),
                    "총 크기": f"{total_size_gb:.1f} GB",
                    "대형 모델 (100MB+)": len(large_models),
                    "가장 큰 파일": max(model_files, key=lambda x: x[1])[0].name if model_files else "없음"
                }
            ))
            
            # Step별 모델 분포 확인
            step_dirs = {}
            for file, size in model_files:
                parts = file.parts
                for part in parts:
                    if 'step_' in part.lower():
                        step_dirs[part] = step_dirs.get(part, 0) + 1
                        break
            
            if step_dirs:
                self.add_result(CheckResult(
                    name="Step별 모델 분포",
                    status=CheckStatus.INFO,
                    message=f"{len(step_dirs)}개 Step에 모델 분포",
                    details=step_dirs
                ))
        else:
            self.add_result(CheckResult(
                name="AI 모델 파일들",
                status=CheckStatus.WARN,
                message="AI 모델 파일을 찾을 수 없음",
                recommendations=[
                    "AI 모델을 다운로드하세요",
                    "모델 파일이 올바른 위치에 있는지 확인하세요"
                ]
            ))

    def check_07_import_test(self):
        """7단계: 주요 모듈 Import 테스트"""
        print(f"\n{'='*50}")
        print("🔄 7단계: 주요 모듈 Import 테스트")
        print(f"{'='*50}")
        
        if not self.backend_root:
            return
        
        # sys.path에 백엔드 루트 추가
        backend_str = str(self.backend_root)
        if backend_str not in sys.path:
            sys.path.insert(0, backend_str)
        
        test_imports = [
            ("app.core.model_paths", "모델 경로 설정"),
            ("app.ai_pipeline.steps", "AI 파이프라인 Steps"),
            ("app.services.step_service", "Step 서비스"),
            ("app.ai_pipeline.interface.step_interface", "Step 인터페이스")
        ]
        
        for module_name, description in test_imports:
            try:
                __import__(module_name)
                self.add_result(CheckResult(
                    name=f"Import 테스트: {description}",
                    status=CheckStatus.PASS,
                    message="성공",
                    details={"모듈": module_name}
                ))
            except ImportError as e:
                self.add_result(CheckResult(
                    name=f"Import 테스트: {description}",
                    status=CheckStatus.FAIL,
                    message=f"실패: {str(e)}",
                    details={"모듈": module_name, "오류": str(e)},
                    recommendations=["모듈 경로와 의존성을 확인하세요"]
                ))
            except Exception as e:
                self.add_result(CheckResult(
                    name=f"Import 테스트: {description}",
                    status=CheckStatus.WARN,
                    message=f"경고: {str(e)}",
                    details={"모듈": module_name, "오류": str(e)}
                ))

    def check_08_configuration_files(self):
        """8단계: 설정 파일들 확인"""
        print(f"\n{'='*50}")
        print("⚙️  8단계: 설정 파일들 확인")
        print(f"{'='*50}")
        
        if not self.project_root:
            return
        
        config_files = [
            ("requirements.txt", "Python 패키지 의존성"),
            ("backend/requirements.txt", "백엔드 의존성"),
            (".gitignore", "Git 무시 파일"),
            ("README.md", "프로젝트 문서"),
            ("backend/app/core/config.py", "백엔드 설정"),
            ("frontend/package.json", "프론트엔드 의존성")
        ]
        
        for file_path, description in config_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                file_size = full_path.stat().st_size
                self.add_result(CheckResult(
                    name=f"설정 파일: {description}",
                    status=CheckStatus.PASS,
                    message="존재함",
                    details={
                        "파일": file_path,
                        "크기": f"{file_size:,} bytes"
                    }
                ))
            else:
                status = CheckStatus.WARN if file_path in ["README.md", "frontend/package.json"] else CheckStatus.FAIL
                self.add_result(CheckResult(
                    name=f"설정 파일: {description}",
                    status=status,
                    message="존재하지 않음",
                    details={"예상 경로": str(full_path)}
                ))

    def print_summary(self):
        """결과 요약 출력"""
        print(f"\n{'='*80}")
        print("📊 디버깅 결과 요약")
        print(f"{'='*80}")
        
        total_checks = len(self.results)
        passed = len([r for r in self.results if r.status == CheckStatus.PASS])
        failed = len([r for r in self.results if r.status == CheckStatus.FAIL])
        warnings = len([r for r in self.results if r.status == CheckStatus.WARN])
        infos = len([r for r in self.results if r.status == CheckStatus.INFO])
        
        print(f"✅ 통과: {passed}개")
        print(f"❌ 실패: {failed}개")
        print(f"⚠️ 경고: {warnings}개")
        print(f"ℹ️ 정보: {infos}개")
        print(f"📊 전체: {total_checks}개")
        
        # 성공률 계산
        success_rate = (passed / total_checks) * 100 if total_checks > 0 else 0
        print(f"🎯 성공률: {success_rate:.1f}%")
        
        # 실행 시간
        elapsed_time = time.time() - self.start_time
        print(f"⏱️ 실행 시간: {elapsed_time:.2f}초")
        
        # 전체 상태 판정
        if failed == 0 and warnings <= 2:
            status_emoji = "🎉"
            status_msg = "프로젝트 상태가 양호합니다!"
        elif failed <= 2:
            status_emoji = "⚠️"
            status_msg = "일부 문제가 있지만 개발 가능합니다."
        else:
            status_emoji = "🚨"
            status_msg = "심각한 문제가 있습니다. 해결이 필요합니다."
        
        print(f"\n{status_emoji} {status_msg}")
        
        # 주요 권장사항 요약
        all_recommendations = []
        for result in self.results:
            if result.recommendations:
                all_recommendations.extend(result.recommendations)
        
        if all_recommendations:
            print(f"\n💡 주요 권장사항:")
            unique_recommendations = list(set(all_recommendations))
            for i, rec in enumerate(unique_recommendations[:5], 1):
                print(f"   {i}. {rec}")
        
        print(f"\n{'='*80}")
        print(f"🕒 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")

    def save_results_to_file(self):
        """결과를 JSON 파일로 저장"""
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root) if self.project_root else None,
            "backend_root": str(self.backend_root) if self.backend_root else None,
            "ai_models_root": str(self.ai_models_root) if self.ai_models_root else None,
            "execution_time": time.time() - self.start_time,
            "results": []
        }
        
        for result in self.results:
            results_data["results"].append({
                "name": result.name,
                "status": result.status.name,
                "message": result.message,
                "details": result.details,
                "recommendations": result.recommendations
            })
        
        output_file = Path("debug_results.json")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2)
            print(f"📄 상세 결과가 {output_file}에 저장되었습니다.")
        except Exception as e:
            print(f"❌ 결과 파일 저장 실패: {e}")

    def run_all_checks(self):
        """모든 체크 실행"""
        self.print_header()
        
        checks = [
            self.check_01_project_structure,
            self.check_02_python_environment,
            self.check_03_required_packages,
            self.check_04_ai_pipeline_structure,
            self.check_05_step_files,
            self.check_06_ai_models,
            self.check_07_import_test,
            self.check_08_configuration_files
        ]
        
        total_checks = len(checks)
        for i, check_func in enumerate(checks, 1):
            self.print_progress(i-1, total_checks, f"실행 중: {check_func.__name__}")
            try:
                check_func()
            except Exception as e:
                self.add_result(CheckResult(
                    name=f"체크 실행: {check_func.__name__}",
                    status=CheckStatus.FAIL,
                    message=f"예외 발생: {str(e)}",
                    recommendations=["체크 함수에 버그가 있을 수 있습니다"]
                ))
        
        print("\n")  # 진행 표시줄 완료 후 줄바꿈
        self.print_summary()
        self.save_results_to_file()

def main():
    """메인 함수"""
    debugger = ProjectDebugger()
    debugger.run_all_checks()

if __name__ == "__main__":
    main()