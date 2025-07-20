#!/usr/bin/env python3
"""
🔍 MyCloset AI - AI 모델 검증 및 수정 스크립트
현재 conda 환경에서 실행 가능한 모델 검증 도구

✅ PIL.Image VERSION 오류 해결
✅ MemoryManagerAdapter 메서드 누락 수정
✅ conda 환경 정보 정확한 감지
✅ M3 Max 128GB 최적화 검증
✅ AI 모델 경로 검증 및 수정
"""

import os
import sys
import time
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# 프로젝트 루트 설정
project_root = Path(__file__).parent
backend_root = project_root / "backend"
sys.path.insert(0, str(backend_root))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("🔍 MyCloset AI - AI 모델 검증 및 수정 스크립트")
print("=" * 60)

class ModelVerifier:
    """AI 모델 검증 및 수정 클래스"""
    
    def __init__(self):
        self.project_root = project_root
        self.backend_root = backend_root
        self.results = {}
        self.fixes_applied = []
        
    def check_system_environment(self) -> Dict[str, Any]:
        """시스템 환경 검증"""
        print("\n🔧 시스템 환경 검증 중...")
        
        env_info = {}
        
        # Python 버전 확인
        env_info['python_version'] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        print(f"🐍 Python: {env_info['python_version']}")
        
        # conda 환경 확인 (정확한 감지)
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        env_info['conda_env'] = conda_env
        env_info['conda_prefix'] = conda_prefix
        env_info['in_conda'] = conda_env != 'base'
        
        print(f"🐍 Conda 환경: {conda_env} ({'✅' if env_info['in_conda'] else '❌'})")
        
        # 메모리 정보
        try:
            import psutil
            memory = psutil.virtual_memory()
            env_info['memory_total_gb'] = round(memory.total / (1024**3), 1)
            env_info['memory_available_gb'] = round(memory.available / (1024**3), 1)
            print(f"💾 메모리: {env_info['memory_total_gb']}GB (사용가능: {env_info['memory_available_gb']}GB)")
        except ImportError:
            env_info['memory_total_gb'] = 'unknown'
            print("💾 메모리: psutil 없음")
        
        # M3 Max 감지
        try:
            import platform
            if platform.system() == 'Darwin' and 'arm64' in platform.machine():
                env_info['is_m3_max'] = env_info['memory_total_gb'] >= 64
            else:
                env_info['is_m3_max'] = False
        except:
            env_info['is_m3_max'] = False
            
        print(f"🍎 M3 Max: {'✅' if env_info['is_m3_max'] else '❌'}")
        
        return env_info
    
    def check_dependencies(self) -> Dict[str, Any]:
        """의존성 라이브러리 검증"""
        print("\n📚 의존성 라이브러리 검증 중...")
        
        deps = {}
        
        # PIL/Pillow 검증
        try:
            from PIL import Image
            # PIL.Image.VERSION → PIL.__version__ 수정 검증
            pil_version = getattr(Image, '__version__', 'unknown')
            if hasattr(Image, 'VERSION'):
                deps['pil_version'] = Image.VERSION
                deps['pil_issue'] = False
            else:
                deps['pil_version'] = pil_version
                deps['pil_issue'] = True
            
            print(f"🖼️ PIL/Pillow: {deps['pil_version']} ({'✅' if not deps['pil_issue'] else '⚠️ VERSION 속성 없음'})")
        except ImportError:
            deps['pil_version'] = None
            deps['pil_issue'] = True
            print("🖼️ PIL/Pillow: ❌ 설치되지 않음")
        
        # NumPy 검증
        try:
            import numpy as np
            deps['numpy_version'] = np.__version__
            deps['numpy_compatible'] = int(np.__version__.split('.')[0]) < 2
            print(f"🔢 NumPy: {deps['numpy_version']} ({'✅' if deps['numpy_compatible'] else '⚠️ 2.x 버전'})")
        except ImportError:
            deps['numpy_version'] = None
            deps['numpy_compatible'] = False
            print("🔢 NumPy: ❌ 설치되지 않음")
        
        # PyTorch 검증
        try:
            import torch
            deps['torch_version'] = torch.__version__
            deps['mps_available'] = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
            deps['mps_built'] = torch.backends.mps.is_built() if hasattr(torch.backends, 'mps') else False
            
            print(f"🔥 PyTorch: {deps['torch_version']}")
            print(f"🍎 MPS: {'✅ 사용가능' if deps['mps_available'] else '❌ 사용불가'}")
        except ImportError:
            deps['torch_version'] = None
            deps['mps_available'] = False
            print("🔥 PyTorch: ❌ 설치되지 않음")
        
        return deps
    
    def check_ai_models_paths(self) -> Dict[str, Any]:
        """AI 모델 경로 검증"""
        print("\n🤖 AI 모델 경로 검증 중...")
        
        models_info = {}
        
        # 기본 AI 모델 디렉토리
        ai_models_dir = self.backend_root / "ai_models"
        models_info['ai_models_exists'] = ai_models_dir.exists()
        
        if models_info['ai_models_exists']:
            print(f"📁 AI 모델 디렉토리: ✅ {ai_models_dir}")
            
            # HuggingFace 캐시 확인
            hf_cache = ai_models_dir / "huggingface_cache"
            models_info['hf_cache_exists'] = hf_cache.exists()
            
            if models_info['hf_cache_exists']:
                print(f"📁 HuggingFace 캐시: ✅ {hf_cache}")
                
                # OOTDiffusion 모델 확인
                ootd_path = hf_cache / "models--levihsu--OOTDiffusion"
                models_info['ootdiffusion_exists'] = ootd_path.exists()
                
                if models_info['ootdiffusion_exists']:
                    print(f"🎯 OOTDiffusion: ✅ {ootd_path}")
                    
                    # 스냅샷 디렉토리 확인
                    snapshots = list(ootd_path.glob("snapshots/*"))
                    models_info['ootd_snapshots'] = len(snapshots)
                    print(f"📸 스냅샷: {models_info['ootd_snapshots']}개")
                    
                    if snapshots:
                        latest_snapshot = snapshots[0]
                        unet_path = latest_snapshot / "checkpoints" / "ootd" / "ootd_dc" / "checkpoint-36000" / "unet_vton"
                        models_info['unet_vton_exists'] = unet_path.exists()
                        
                        if models_info['unet_vton_exists']:
                            print(f"🧠 UNet VTON: ✅ {unet_path}")
                        else:
                            print(f"🧠 UNet VTON: ❌ {unet_path}")
                else:
                    print("🎯 OOTDiffusion: ❌ 없음")
            else:
                print(f"📁 HuggingFace 캐시: ❌ {hf_cache}")
        else:
            print(f"📁 AI 모델 디렉토리: ❌ {ai_models_dir}")
        
        # 개별 체크포인트 파일들 확인
        checkpoint_files = [
            "exp-schp-201908301523-atr.pth",
            "openpose.pth",
            "u2net.pth"
        ]
        
        models_info['checkpoints'] = {}
        for ckpt in checkpoint_files:
            ckpt_path = ai_models_dir / ckpt
            exists = ckpt_path.exists()
            models_info['checkpoints'][ckpt] = {
                'exists': exists,
                'path': str(ckpt_path),
                'size_mb': round(ckpt_path.stat().st_size / (1024*1024), 1) if exists else 0
            }
            
            status = "✅" if exists else "❌"
            size_info = f"({models_info['checkpoints'][ckpt]['size_mb']}MB)" if exists else ""
            print(f"⚙️ {ckpt}: {status} {size_info}")
        
        return models_info
    
    def check_backend_issues(self) -> Dict[str, Any]:
        """백엔드 코드 문제점 검증"""
        print("\n🔧 백엔드 코드 문제점 검증 중...")
        
        issues = {}
        
        # MemoryManagerAdapter 검증
        memory_manager_path = self.backend_root / "app" / "ai_pipeline" / "utils" / "memory_manager.py"
        if memory_manager_path.exists():
            try:
                with open(memory_manager_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # optimize_memory 메서드 존재 확인
                has_optimize_memory = 'def optimize_memory' in content
                issues['memory_manager_optimize_method'] = has_optimize_memory
                
                print(f"🧠 MemoryManagerAdapter.optimize_memory: {'✅' if has_optimize_memory else '❌ 메서드 누락'}")
            except Exception as e:
                issues['memory_manager_optimize_method'] = False
                print(f"🧠 MemoryManagerAdapter: ❌ 읽기 실패 - {e}")
        else:
            issues['memory_manager_optimize_method'] = False
            print(f"🧠 MemoryManagerAdapter: ❌ 파일 없음")
        
        # PIL.Image.VERSION 사용 확인
        utils_files = list((self.backend_root / "app").rglob("*.py"))
        pil_version_usage = []
        
        for file_path in utils_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if 'PIL.Image.VERSION' in content or 'Image.VERSION' in content:
                    pil_version_usage.append(str(file_path.relative_to(self.backend_root)))
            except:
                continue
        
        issues['pil_version_usage'] = pil_version_usage
        if pil_version_usage:
            print(f"🖼️ PIL.Image.VERSION 사용: ❌ {len(pil_version_usage)}개 파일")
            for file in pil_version_usage:
                print(f"   📄 {file}")
        else:
            print("🖼️ PIL.Image.VERSION 사용: ✅ 없음")
        
        return issues
    
    def apply_immediate_fixes(self, env_info: Dict, deps: Dict, issues: Dict) -> List[str]:
        """즉시 적용 가능한 수정사항들"""
        print("\n🛠️ 즉시 수정 적용 중...")
        
        fixes = []
        
        # 1. PIL.Image.VERSION 수정
        if issues.get('pil_version_usage'):
            for file_rel_path in issues['pil_version_usage']:
                file_path = self.backend_root / file_rel_path
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # PIL.Image.VERSION → PIL.__version__ 변경
                    modified_content = content.replace('PIL.Image.VERSION', 'PIL.__version__')
                    modified_content = modified_content.replace('Image.VERSION', 'Image.__version__')
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(modified_content)
                    
                    fixes.append(f"PIL.Image.VERSION 수정: {file_rel_path}")
                    print(f"✅ {file_rel_path} - PIL.Image.VERSION 수정")
                except Exception as e:
                    print(f"❌ {file_rel_path} - 수정 실패: {e}")
        
        # 2. MemoryManagerAdapter.optimize_memory 메서드 추가
        if not issues.get('memory_manager_optimize_method'):
            memory_manager_path = self.backend_root / "app" / "ai_pipeline" / "utils" / "memory_manager.py"
            if memory_manager_path.exists():
                try:
                    with open(memory_manager_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # MemoryManagerAdapter 클래스 찾기
                    if 'class MemoryManagerAdapter' in content:
                        optimize_method = '''
    async def optimize_memory(self, aggressive: bool = False):
        """메모리 최적화 - M3 Max 128GB 최적화"""
        try:
            import gc
            gc.collect()
            
            if hasattr(self, 'device') and self.device == "mps":
                try:
                    import torch
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                    if hasattr(torch.mps, 'synchronize'):
                        torch.mps.synchronize()
                except (AttributeError, RuntimeError):
                    pass  # MPS 기능 없는 경우 무시
            
            if aggressive:
                # 추가적인 메모리 정리
                for _ in range(3):
                    gc.collect()
                    
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"메모리 최적화 실패: {e}")
'''
                        
                        # 클래스 끝 부분에 메서드 추가
                        lines = content.split('\n')
                        new_lines = []
                        in_class = False
                        class_indent = 0
                        
                        for line in lines:
                            new_lines.append(line)
                            
                            if 'class MemoryManagerAdapter' in line:
                                in_class = True
                                class_indent = len(line) - len(line.lstrip())
                            elif in_class and line.strip() and not line.startswith(' ' * (class_indent + 1)):
                                # 클래스 끝
                                new_lines.insert(-1, optimize_method)
                                in_class = False
                        
                        if in_class:
                            # 파일 끝에서 클래스가 끝나는 경우
                            new_lines.append(optimize_method)
                        
                        with open(memory_manager_path, 'w', encoding='utf-8') as f:
                            f.write('\n'.join(new_lines))
                        
                        fixes.append("MemoryManagerAdapter.optimize_memory 메서드 추가")
                        print("✅ MemoryManagerAdapter.optimize_memory 메서드 추가")
                    else:
                        print("❌ MemoryManagerAdapter 클래스를 찾을 수 없음")
                except Exception as e:
                    print(f"❌ MemoryManagerAdapter 수정 실패: {e}")
        
        return fixes
    
    def generate_conda_fix_script(self, env_info: Dict) -> str:
        """Conda 환경 수정 스크립트 생성"""
        script_content = f"""#!/bin/bash
# MyCloset AI - Conda 환경 수정 스크립트 (Python {env_info['python_version']})

echo "🔧 MyCloset AI Conda 환경 수정"
echo "현재 환경: {env_info['conda_env']}"
echo "Python: {env_info['python_version']}"
echo ""

# 현재 환경 활성화
conda activate {env_info['conda_env']}

# NumPy 호환성 해결 (Python 3.12 버전)
echo "🔢 NumPy 호환성 수정 중..."
pip install numpy==1.24.4

# PyTorch M3 Max 최적화 버전 설치
echo "🔥 PyTorch M3 Max 최적화 설치 중..."
pip install torch torchvision torchaudio

# 기타 필수 패키지 업데이트
echo "📚 필수 패키지 업데이트 중..."
pip install --upgrade fastapi uvicorn pydantic

echo "✅ Conda 환경 수정 완료"
echo "🚀 서버 실행: cd backend && python app/main.py"
"""
        
        script_path = self.project_root / "fix_conda_env.sh"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        return str(script_path)
    
    def run_verification(self) -> Dict[str, Any]:
        """전체 검증 실행"""
        print("🔍 MyCloset AI 전체 검증 시작...")
        start_time = time.time()
        
        # 1. 시스템 환경 검증
        env_info = self.check_system_environment()
        
        # 2. 의존성 검증
        deps = self.check_dependencies()
        
        # 3. AI 모델 경로 검증
        models_info = self.check_ai_models_paths()
        
        # 4. 백엔드 코드 문제점 검증
        issues = self.check_backend_issues()
        
        # 5. 즉시 수정 적용
        fixes = self.apply_immediate_fixes(env_info, deps, issues)
        
        # 6. Conda 수정 스크립트 생성
        conda_script = self.generate_conda_fix_script(env_info)
        
        elapsed_time = time.time() - start_time
        
        # 결과 정리
        results = {
            'verification_time': round(elapsed_time, 2),
            'environment': env_info,
            'dependencies': deps,
            'models': models_info,
            'issues': issues,
            'fixes_applied': fixes,
            'conda_fix_script': conda_script,
            'summary': self.generate_summary(env_info, deps, models_info, issues, fixes)
        }
        
        return results
    
    def generate_summary(self, env_info: Dict, deps: Dict, models_info: Dict, issues: Dict, fixes: List[str]) -> Dict[str, Any]:
        """검증 결과 요약"""
        summary = {
            'status': 'healthy',
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Critical Issues
        if not deps.get('pil_version'):
            summary['critical_issues'].append("PIL/Pillow not installed")
            summary['status'] = 'critical'
        
        if not deps.get('torch_version'):
            summary['critical_issues'].append("PyTorch not installed")
            summary['status'] = 'critical'
        
        if not issues.get('memory_manager_optimize_method'):
            summary['critical_issues'].append("MemoryManagerAdapter.optimize_memory missing")
            summary['status'] = 'critical'
        
        # Warnings
        if deps.get('pil_issue'):
            summary['warnings'].append("PIL.Image.VERSION attribute issue")
        
        if not deps.get('mps_available'):
            summary['warnings'].append("MPS not available for M3 Max")
        
        if not models_info.get('ootdiffusion_exists'):
            summary['warnings'].append("OOTDiffusion model not found")
        
        # Recommendations
        if env_info.get('is_m3_max'):
            summary['recommendations'].append("Enable M3 Max 128GB optimization")
        
        if not deps.get('numpy_compatible'):
            summary['recommendations'].append("Downgrade NumPy to 1.x for compatibility")
        
        return summary

def main():
    """메인 실행 함수"""
    verifier = ModelVerifier()
    
    try:
        results = verifier.run_verification()
        
        print("\n" + "="*60)
        print("🎯 검증 결과 요약")
        print("="*60)
        
        summary = results['summary']
        print(f"📊 전체 상태: {summary['status'].upper()}")
        print(f"⏱️ 검증 시간: {results['verification_time']}초")
        print(f"🛠️ 적용된 수정: {len(results['fixes_applied'])}개")
        
        if summary['critical_issues']:
            print(f"\n🚨 중요 문제: {len(summary['critical_issues'])}개")
            for issue in summary['critical_issues']:
                print(f"   ❌ {issue}")
        
        if summary['warnings']:
            print(f"\n⚠️ 경고 사항: {len(summary['warnings'])}개")
            for warning in summary['warnings']:
                print(f"   ⚠️ {warning}")
        
        if summary['recommendations']:
            print(f"\n💡 권장 사항: {len(summary['recommendations'])}개")
            for rec in summary['recommendations']:
                print(f"   💡 {rec}")
        
        if results['fixes_applied']:
            print(f"\n✅ 적용된 수정 사항:")
            for fix in results['fixes_applied']:
                print(f"   ✅ {fix}")
        
        print(f"\n📝 Conda 수정 스크립트: {results['conda_fix_script']}")
        print("\n🚀 다음 단계:")
        print("1. ./fix_conda_env.sh 실행")
        print("2. cd backend && python app/main.py")
        
        # 결과를 JSON 파일로 저장
        results_file = project_root / "verification_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📄 상세 결과: {results_file}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 검증 실패: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()