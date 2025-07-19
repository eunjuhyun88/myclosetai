#!/usr/bin/env python3
"""
🔥 즉시 작동하는 AI 모델 스캐너 - 수정된 버전
==============================================

MyCloset AI 프로젝트 구조에 맞춰 실제 모델 위치를 정확히 찾습니다.

사용법:
    python quick_scanner.py                 # 현재 위치에서 스캔
    python quick_scanner.py --organize      # 스캔 + 정리
    python quick_scanner.py --verbose       # 상세 출력
"""

import os
import sys
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import re
import hashlib

@dataclass
class ModelInfo:
    name: str
    path: str
    size_mb: float
    framework: str
    step_candidate: str
    confidence: float
    is_valid: bool

class QuickModelScanner:
    """즉시 작동하는 모델 스캐너"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.current_dir = Path.cwd()
        self.found_models = []
        
        # 프로젝트 루트 자동 탐지
        self.project_root = self._find_project_root()
        self.ai_models_dir = self._find_ai_models_dir()
        
        print(f"🔍 프로젝트 루트: {self.project_root}")
        print(f"🤖 AI 모델 디렉토리: {self.ai_models_dir}")
        
        # 모델 확장자
        self.model_extensions = {
            '.pth', '.pt', '.bin', '.safetensors', '.ckpt', 
            '.h5', '.pb', '.onnx', '.pkl', '.model'
        }
        
        # MyCloset AI 8단계 패턴
        self.step_patterns = {
            'step_01_human_parsing': [
                'human.*pars', 'graphonomy', 'schp', 'atr', 'self.*correction'
            ],
            'step_02_pose_estimation': [
                'pose', 'openpose', 'dwpose', 'keypoint'
            ],
            'step_03_cloth_segmentation': [
                'cloth.*seg', 'u2net', 'sam', 'segment'
            ],
            'step_04_geometric_matching': [
                'geometric', 'gmm', 'matching', 'tps'
            ],
            'step_05_cloth_warping': [
                'warp', 'tom', 'viton.*warp', 'deformation'
            ],
            'step_06_virtual_fitting': [
                'virtual', 'ootdiff', 'viton', 'hrviton', 'diffusion'
            ],
            'step_07_post_processing': [
                'post.*process', 'esrgan', 'super.*resolution'
            ],
            'step_08_quality_assessment': [
                'clip', 'quality', 'assessment', 'metric'
            ]
        }
    
    def _find_project_root(self) -> Path:
        """프로젝트 루트 찾기"""
        current = self.current_dir
        
        # 현재 디렉토리가 mycloset-ai인 경우
        if current.name == 'mycloset-ai':
            return current
        
        # 부모 디렉토리들 검사
        for parent in current.parents:
            if parent.name == 'mycloset-ai':
                return parent
            # backend 또는 frontend 디렉토리가 있는 곳 찾기
            if (parent / 'backend').exists() and (parent / 'frontend').exists():
                return parent
        
        # 기본값으로 현재 디렉토리
        return current
    
    def _find_ai_models_dir(self) -> Optional[Path]:
        """AI 모델 디렉토리 찾기"""
        candidates = [
            self.project_root / "backend" / "ai_models",
            self.project_root / "ai_models", 
            self.current_dir / "backend" / "ai_models",
            self.current_dir / "ai_models"
        ]
        
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
        
        return None
    
    def scan_models(self) -> List[ModelInfo]:
        """모델 스캔 실행"""
        print("🚀 AI 모델 스캔 시작...")
        print("=" * 60)
        
        if not self.ai_models_dir:
            print("❌ AI 모델 디렉토리를 찾을 수 없습니다.")
            self._suggest_locations()
            return []
        
        # 1. 기본 AI 모델 디렉토리 스캔
        print(f"📁 메인 디렉토리 스캔: {self.ai_models_dir}")
        self._scan_directory(self.ai_models_dir)
        
        # 2. 추가 위치 스캔
        additional_paths = [
            self.project_root / "models",
            self.project_root / "checkpoints", 
            Path.home() / "Downloads",
            Path.home() / ".cache" / "huggingface",
            Path.home() / ".cache" / "torch"
        ]
        
        for path in additional_paths:
            if path.exists():
                print(f"📂 추가 스캔: {path}")
                self._scan_directory(path)
        
        # 3. 결과 정리 및 출력
        self._process_results()
        
        return self.found_models
    
    def _scan_directory(self, directory: Path, max_depth: int = 5):
        """디렉토리 스캔"""
        try:
            if max_depth <= 0:
                return
            
            for item in directory.iterdir():
                if item.is_file() and item.suffix.lower() in self.model_extensions:
                    try:
                        size_mb = item.stat().st_size / (1024 * 1024)
                        if size_mb > 0.1:  # 0.1MB 이상만
                            model_info = self._analyze_model(item, size_mb)
                            if model_info:
                                self.found_models.append(model_info)
                                if self.verbose:
                                    print(f"  ✅ {item.name} ({size_mb:.1f}MB)")
                    except Exception as e:
                        if self.verbose:
                            print(f"  ⚠️ 스캔 실패 {item.name}: {e}")
                
                elif item.is_dir() and not self._should_skip_directory(item):
                    self._scan_directory(item, max_depth - 1)
                    
        except PermissionError:
            if self.verbose:
                print(f"  ⚠️ 권한 없음: {directory}")
        except Exception as e:
            if self.verbose:
                print(f"  ❌ 오류: {directory} - {e}")
    
    def _should_skip_directory(self, directory: Path) -> bool:
        """건너뛸 디렉토리인지 확인"""
        skip_patterns = {
            '__pycache__', '.git', 'node_modules', '.cache',
            '.DS_Store', 'temp', 'tmp'
        }
        return directory.name in skip_patterns or directory.name.startswith('.')
    
    def _analyze_model(self, file_path: Path, size_mb: float) -> Optional[ModelInfo]:
        """모델 파일 분석"""
        try:
            # 프레임워크 분류
            framework = self._classify_framework(file_path)
            
            # Step 분류
            step_candidate, confidence = self._classify_step(file_path)
            
            # 유효성 검사
            is_valid = self._validate_model(file_path, framework)
            
            return ModelInfo(
                name=file_path.name,
                path=str(file_path.absolute()),
                size_mb=size_mb,
                framework=framework,
                step_candidate=step_candidate,
                confidence=confidence,
                is_valid=is_valid
            )
            
        except Exception as e:
            if self.verbose:
                print(f"    ⚠️ 분석 실패 {file_path.name}: {e}")
            return None
    
    def _classify_framework(self, file_path: Path) -> str:
        """프레임워크 분류"""
        ext = file_path.suffix.lower()
        
        if ext in ['.pth', '.pt']:
            return 'pytorch'
        elif ext == '.safetensors':
            return 'safetensors'
        elif ext in ['.pb', '.h5']:
            return 'tensorflow'
        elif ext == '.onnx':
            return 'onnx'
        elif ext == '.bin':
            # 내용으로 판단
            path_str = str(file_path).lower()
            if 'pytorch' in path_str or 'transformers' in path_str:
                return 'pytorch'
            return 'binary'
        else:
            return 'unknown'
    
    def _classify_step(self, file_path: Path) -> tuple:
        """MyCloset AI 8단계 분류"""
        path_str = str(file_path).lower()
        name_str = file_path.name.lower()
        
        best_step = "unknown"
        best_confidence = 0.0
        
        for step_name, patterns in self.step_patterns.items():
            confidence = 0.0
            
            for pattern in patterns:
                if re.search(pattern, path_str):
                    confidence = max(confidence, 0.9)
                elif re.search(pattern, name_str):
                    confidence = max(confidence, 0.8)
                elif pattern.replace('.*', '') in path_str:
                    confidence = max(confidence, 0.6)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_step = step_name
        
        return best_step, best_confidence
    
    def _validate_model(self, file_path: Path, framework: str) -> bool:
        """모델 유효성 검사"""
        try:
            if file_path.stat().st_size < 1024:  # 1KB 미만
                return False
            
            # 간단한 헤더 검사
            with open(file_path, 'rb') as f:
                header = f.read(100)
            
            if framework == 'pytorch' and (b'PK' in header or b'\x80' in header):
                return True
            elif framework == 'safetensors' and b'{' in header:
                return True
            elif framework == 'tensorflow' and len(header) > 10:
                return True
            else:
                return True
        except:
            return False
    
    def _process_results(self):
        """결과 처리 및 출력"""
        print("\n" + "=" * 80)
        print("🎯 AI 모델 스캔 결과")
        print("=" * 80)
        
        if not self.found_models:
            print("❌ AI 모델을 찾을 수 없습니다.")
            self._suggest_debug_steps()
            return
        
        # 통계
        total_models = len(self.found_models)
        total_size = sum(m.size_mb for m in self.found_models)
        valid_models = [m for m in self.found_models if m.is_valid]
        
        print(f"📊 총 발견: {total_models}개 모델")
        print(f"💾 총 크기: {total_size:.1f}MB ({total_size/1024:.2f}GB)")
        print(f"✅ 유효한 모델: {len(valid_models)}개")
        
        # 프레임워크별 분포
        frameworks = {}
        for model in self.found_models:
            fw = model.framework
            frameworks[fw] = frameworks.get(fw, 0) + 1
        
        print(f"\n🔧 프레임워크별 분포:")
        for fw, count in sorted(frameworks.items(), key=lambda x: x[1], reverse=True):
            fw_size = sum(m.size_mb for m in self.found_models if m.framework == fw)
            print(f"  - {fw}: {count}개 ({fw_size:.1f}MB)")
        
        # Step별 분포 (신뢰도 0.5 이상)
        steps = {}
        for model in self.found_models:
            if model.confidence >= 0.5:
                step = model.step_candidate
                steps[step] = steps.get(step, 0) + 1
        
        if steps:
            print(f"\n🎯 MyCloset AI Step별 분포:")
            step_names = {
                'step_01_human_parsing': '1️⃣ Human Parsing',
                'step_02_pose_estimation': '2️⃣ Pose Estimation', 
                'step_03_cloth_segmentation': '3️⃣ Cloth Segmentation',
                'step_04_geometric_matching': '4️⃣ Geometric Matching',
                'step_05_cloth_warping': '5️⃣ Cloth Warping',
                'step_06_virtual_fitting': '6️⃣ Virtual Fitting',
                'step_07_post_processing': '7️⃣ Post Processing',
                'step_08_quality_assessment': '8️⃣ Quality Assessment'
            }
            
            for step, count in sorted(steps.items()):
                display_name = step_names.get(step, step)
                step_size = sum(m.size_mb for m in self.found_models 
                              if m.step_candidate == step and m.confidence >= 0.5)
                print(f"  {display_name}: {count}개 ({step_size:.1f}MB)")
        
        # 상위 모델들
        print(f"\n🏆 발견된 주요 모델들:")
        sorted_models = sorted(self.found_models, key=lambda x: x.size_mb, reverse=True)
        
        for i, model in enumerate(sorted_models[:15], 1):
            step_info = ""
            if model.confidence >= 0.5:
                step_num = model.step_candidate.split('_')[1] if '_' in model.step_candidate else '?'
                step_info = f" | 🎯 Step {step_num}"
            
            confidence_icon = "🟢" if model.confidence >= 0.8 else "🟡" if model.confidence >= 0.5 else "🔴"
            validity_icon = "✅" if model.is_valid else "⚠️"
            
            print(f"  {i:2d}. {model.name}")
            print(f"      📁 {model.path}")
            print(f"      📊 {model.size_mb:.1f}MB | {model.framework} | "
                  f"{validity_icon} | {confidence_icon} {model.confidence:.2f}{step_info}")
    
    def _suggest_debug_steps(self):
        """디버그 단계 제안"""
        print("\n🔍 디버그 단계:")
        print("1. 실제 AI 모델 위치 확인:")
        
        # 현재 프로젝트 구조 출력
        if self.ai_models_dir and self.ai_models_dir.exists():
            print(f"   📁 AI 모델 디렉토리 존재: {self.ai_models_dir}")
            try:
                items = list(self.ai_models_dir.iterdir())
                print(f"   📄 내용 ({len(items)}개):")
                for item in items[:10]:
                    if item.is_dir():
                        try:
                            sub_count = len(list(item.iterdir()))
                            print(f"     📁 {item.name}/ ({sub_count}개 파일)")
                        except:
                            print(f"     📁 {item.name}/ (접근 불가)")
                    else:
                        size_mb = item.stat().st_size / (1024 * 1024)
                        print(f"     📄 {item.name} ({size_mb:.1f}MB)")
                
                if len(items) > 10:
                    print(f"     ... 외 {len(items) - 10}개")
            except Exception as e:
                print(f"   ❌ 디렉토리 읽기 실패: {e}")
        else:
            print(f"   ❌ AI 모델 디렉토리 없음: {self.ai_models_dir}")
        
        print("\n2. 수동 확인 명령어:")
        print(f"   find {self.project_root} -name '*.pth' -o -name '*.pt' -o -name '*.bin' -o -name '*.safetensors'")
        print(f"   ls -la {self.ai_models_dir}/ 2>/dev/null")
        
        print("\n3. 예상 위치들:")
        expected_locations = [
            self.project_root / "backend" / "ai_models",
            self.project_root / "ai_models",
            Path.home() / "Downloads",
            Path.home() / ".cache" / "huggingface"
        ]
        
        for location in expected_locations:
            exists = "✅" if location.exists() else "❌"
            print(f"   {exists} {location}")
    
    def _suggest_locations(self):
        """모델 위치 제안"""
        print("\n💡 AI 모델 디렉토리를 찾을 수 없습니다.")
        print("\n예상 위치:")
        print("  - ./ai_models/")
        print("  - ./backend/ai_models/") 
        print("  - ~/Downloads/")
        print("  - ~/.cache/huggingface/")
        
    def generate_model_config(self, output_file: str = "model_scan_result.json"):
        """모델 설정 파일 생성"""
        if not self.found_models:
            print("❌ 설정 파일을 생성할 모델이 없습니다.")
            return
        
        config_data = {
            "scan_info": {
                "timestamp": datetime.now().isoformat(),
                "project_root": str(self.project_root),
                "ai_models_dir": str(self.ai_models_dir),
                "total_models": len(self.found_models)
            },
            "models": {}
        }
        
        for i, model in enumerate(self.found_models):
            model_key = f"model_{i+1:03d}"
            config_data["models"][model_key] = {
                "name": model.name,
                "path": model.path,
                "size_mb": model.size_mb,
                "framework": model.framework,
                "step_candidate": model.step_candidate,
                "confidence": model.confidence,
                "is_valid": model.is_valid
            }
        
        # JSON 파일 저장
        output_path = self.project_root / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 설정 파일 저장: {output_path}")
        
        # Python 설정 파일도 생성
        self._generate_python_config()
    
    def _generate_python_config(self):
        """Python 설정 파일 생성"""
        config_content = f'''#!/usr/bin/env python3
"""
MyCloset AI 모델 경로 설정 - 자동 생성됨
생성 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
발견된 모델: {len(self.found_models)}개
"""

from pathlib import Path

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent
AI_MODELS_ROOT = PROJECT_ROOT / "ai_models"

# 발견된 모델 경로들
SCANNED_MODELS = {{
'''
        
        for i, model in enumerate(self.found_models):
            safe_name = model.name.replace('.', '_').replace('-', '_')
            config_content += f'''    "{safe_name}": {{
        "name": "{model.name}",
        "path": Path("{model.path}"),
        "framework": "{model.framework}",
        "step": "{model.step_candidate}",
        "confidence": {model.confidence:.3f},
        "size_mb": {model.size_mb:.1f}
    }},
'''
        
        config_content += '''}

# Step별 모델 매핑
STEP_MODELS = {
'''
        
        # Step별 모델 그룹화
        step_models = {}
        for model in self.found_models:
            if model.confidence >= 0.5:
                step = model.step_candidate
                if step not in step_models:
                    step_models[step] = []
                step_models[step].append(model.name.replace('.', '_').replace('-', '_'))
        
        for step, models in step_models.items():
            config_content += f'    "{step}": {models},\n'
        
        config_content += '''}

def get_model_path(model_name: str) -> Path:
    """모델 경로 반환"""
    for key, info in SCANNED_MODELS.items():
        if model_name in key or model_name in info["name"]:
            return info["path"]
    raise KeyError(f"Model not found: {model_name}")

def get_step_models(step: str) -> list:
    """Step별 모델 목록"""
    return STEP_MODELS.get(step, [])

def list_available_models() -> dict:
    """사용 가능한 모델 목록"""
    available = {}
    for key, info in SCANNED_MODELS.items():
        if info["path"].exists():
            available[key] = info
    return available

if __name__ == "__main__":
    print("🤖 MyCloset AI 모델 설정")
    print("=" * 40)
    available = list_available_models()
    print(f"사용 가능한 모델: {len(available)}개")
    
    for key, info in available.items():
        print(f"  ✅ {info['name']} ({info['framework']}, {info['size_mb']:.1f}MB)")
'''
        
        config_path = self.project_root / "model_paths_config.py"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"🐍 Python 설정 파일: {config_path}")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="즉시 작동하는 AI 모델 스캐너")
    parser.add_argument('--organize', action='store_true', help='스캔 후 설정 파일 생성')
    parser.add_argument('--verbose', action='store_true', help='상세 출력')
    parser.add_argument('--output', type=str, default='model_scan_result.json', help='출력 파일명')
    
    args = parser.parse_args()
    
    try:
        scanner = QuickModelScanner(verbose=args.verbose)
        models = scanner.scan_models()
        
        if args.organize and models:
            scanner.generate_model_config(args.output)
        
        print(f"\n✅ 스캔 완료! 발견된 모델: {len(models)}개")
        return 0 if models else 1
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())