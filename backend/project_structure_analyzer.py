#!/usr/bin/env python3
"""
MyCloset AI 프로젝트 구조 분석 및 수정 스크립트
====================================================

컴퓨터 스펙에 맞춰서 제안하고, 실제 GitHub 구조를 파악한 후
하나씩 수정할 항목들을 분석합니다.

Author: MyCloset AI Team
Date: 2025-07-16
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import subprocess
import shutil
from datetime import datetime
import platform

class ProjectStructureAnalyzer:
    """프로젝트 구조 분석 및 수정 제안 클래스"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.backend_root = self.project_root / "backend"
        self.analysis_report = {}
        self.required_structure = self._get_required_structure()
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 수집"""
        try:
            # CPU 정보
            cpu_info = platform.processor()
            if "M3" in cpu_info or "Apple" in cpu_info:
                cpu_type = "Apple Silicon M3"
                gpu_type = "Apple M3 Max GPU"
                optimization = "MPS"
            else:
                cpu_type = cpu_info
                gpu_type = "Unknown"
                optimization = "CPU/CUDA"
            
            # Python 버전
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            
            # 메모리 정보 (대략적)
            try:
                import psutil
                memory_gb = round(psutil.virtual_memory().total / (1024**3))
            except ImportError:
                memory_gb = "Unknown"
            
            return {
                "platform": platform.system(),
                "cpu": cpu_type,
                "gpu": gpu_type,
                "optimization": optimization,
                "python_version": python_version,
                "memory_gb": memory_gb,
                "architecture": platform.machine()
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_required_structure(self) -> Dict[str, Any]:
        """필요한 프로젝트 구조 정의"""
        return {
            "backend": {
                "directories": {
                    "app": {
                        "api": ["__init__.py", "pipeline_routes.py", "health.py", "models.py"],
                        "core": ["__init__.py", "config.py", "gpu_config.py", "logging_config.py", "model_paths.py"],
                        "models": ["__init__.py", "schemas.py", "ootd_model.py"],
                        "services": ["__init__.py", "ai_models.py", "model_manager.py", "virtual_fitter.py", 
                                   "human_analysis.py", "cloth_analysis.py"],
                        "utils": ["__init__.py", "file_manager.py", "image_utils.py"],
                        "ai_pipeline": {
                            "steps": ["__init__.py", "step_01_human_parsing.py", "step_02_pose_estimation.py",
                                    "step_03_cloth_segmentation.py", "step_04_geometric_matching.py",
                                    "step_05_cloth_warping.py", "step_06_virtual_fitting.py",
                                    "step_07_post_processing.py", "step_08_quality_assessment.py"],
                            "utils": ["__init__.py", "memory_manager.py", "data_converter.py", "model_loader.py"]
                        }
                    },
                    "ai_models": {
                        "checkpoints": {},
                        "configs": ["models_config.yaml"],
                        "clip-vit-base-patch32": ["config.json", "model.safetensors"]
                    },
                    "static": {
                        "uploads": [".gitkeep"],
                        "results": [".gitkeep"]
                    },
                    "tests": ["__init__.py", "test_api.py", "test_models.py"],
                    "scripts": {
                        "test": ["test_final_models.py", "simple_model_test.py"],
                        "utils": ["check_imports.py"],
                        "download": ["model_downloader.py"]
                    },
                    "logs": [".gitkeep"]
                },
                "files": ["requirements.txt", "run_server.py", "Makefile", "README.md", ".env.example"]
            },
            "frontend": {
                "directories": {
                    "src": {
                        "components": {
                            "ui": [],
                            "features": []
                        },
                        "pages": [],
                        "hooks": [],
                        "types": [],
                        "utils": []
                    },
                    "public": []
                },
                "files": ["package.json", "tsconfig.json", "vite.config.ts", "index.html"]
            }
        }
    
    def analyze_current_structure(self) -> Dict[str, Any]:
        """현재 프로젝트 구조 분석"""
        print("🔍 프로젝트 구조 분석 시작...")
        print("=" * 60)
        
        current_structure = {}
        missing_items = []
        existing_items = []
        problematic_items = []
        
        # 백엔드 구조 분석
        backend_analysis = self._analyze_backend()
        current_structure["backend"] = backend_analysis
        
        # 프론트엔드 구조 분석 (있는 경우)
        frontend_analysis = self._analyze_frontend()
        current_structure["frontend"] = frontend_analysis
        
        # AI 모델 분석
        ai_models_analysis = self._analyze_ai_models()
        current_structure["ai_models"] = ai_models_analysis
        
        # 누락 항목 및 문제 항목 식별
        missing_items, problematic_items = self._identify_issues()
        
        return {
            "system_info": self.system_info,
            "current_structure": current_structure,
            "missing_items": missing_items,
            "problematic_items": problematic_items,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _analyze_backend(self) -> Dict[str, Any]:
        """백엔드 구조 분석"""
        backend_info = {
            "exists": self.backend_root.exists(),
            "directories": {},
            "files": {},
            "critical_issues": []
        }
        
        if not self.backend_root.exists():
            backend_info["critical_issues"].append("backend 디렉토리가 존재하지 않음")
            return backend_info
        
        # 주요 디렉토리 확인
        key_dirs = ["app", "ai_models", "static", "tests", "scripts", "logs"]
        for dir_name in key_dirs:
            dir_path = self.backend_root / dir_name
            backend_info["directories"][dir_name] = {
                "exists": dir_path.exists(),
                "contents": list(dir_path.iterdir()) if dir_path.exists() else []
            }
        
        # 주요 파일 확인
        key_files = ["requirements.txt", "run_server.py", "Makefile", "README.md"]
        for file_name in key_files:
            file_path = self.backend_root / file_name
            backend_info["files"][file_name] = {
                "exists": file_path.exists(),
                "size": file_path.stat().st_size if file_path.exists() else 0
            }
        
        # app 디렉토리 세부 분석
        if (self.backend_root / "app").exists():
            backend_info["app_structure"] = self._analyze_app_structure()
        
        return backend_info
    
    def _analyze_app_structure(self) -> Dict[str, Any]:
        """app 디렉토리 구조 세부 분석"""
        app_root = self.backend_root / "app"
        app_structure = {
            "api": self._analyze_directory(app_root / "api"),
            "core": self._analyze_directory(app_root / "core"),
            "models": self._analyze_directory(app_root / "models"),
            "services": self._analyze_directory(app_root / "services"),
            "utils": self._analyze_directory(app_root / "utils"),
            "ai_pipeline": self._analyze_directory(app_root / "ai_pipeline")
        }
        return app_structure
    
    def _analyze_directory(self, dir_path: Path) -> Dict[str, Any]:
        """디렉토리 분석"""
        if not dir_path.exists():
            return {"exists": False, "files": [], "subdirs": []}
        
        files = [f.name for f in dir_path.iterdir() if f.is_file()]
        subdirs = [d.name for d in dir_path.iterdir() if d.is_dir()]
        
        return {
            "exists": True,
            "files": files,
            "subdirs": subdirs,
            "total_files": len(files),
            "python_files": len([f for f in files if f.endswith('.py')])
        }
    
    def _analyze_frontend(self) -> Dict[str, Any]:
        """프론트엔드 구조 분석"""
        frontend_root = self.project_root / "frontend"
        return {
            "exists": frontend_root.exists(),
            "package_json": (frontend_root / "package.json").exists() if frontend_root.exists() else False,
            "src_exists": (frontend_root / "src").exists() if frontend_root.exists() else False
        }
    
    def _analyze_ai_models(self) -> Dict[str, Any]:
        """AI 모델 구조 분석"""
        ai_models_root = self.backend_root / "ai_models"
        
        if not ai_models_root.exists():
            return {"exists": False, "critical": "AI 모델 디렉토리 누락"}
        
        models_info = {
            "exists": True,
            "clip_model": self._check_clip_model(),
            "checkpoints": self._analyze_directory(ai_models_root / "checkpoints"),
            "ootdiffusion": self._check_ootdiffusion(),
            "stable_diffusion": self._check_stable_diffusion(),
            "total_size_gb": self._calculate_models_size()
        }
        
        return models_info
    
    def _check_clip_model(self) -> Dict[str, Any]:
        """CLIP 모델 확인"""
        clip_path = self.backend_root / "ai_models" / "clip-vit-base-patch32"
        if not clip_path.exists():
            return {"exists": False, "issue": "CLIP 모델 디렉토리 없음"}
        
        required_files = ["config.json", "model.safetensors"]
        existing_files = [f.name for f in clip_path.iterdir() if f.is_file()]
        
        return {
            "exists": True,
            "required_files": required_files,
            "existing_files": existing_files,
            "complete": all(f in existing_files for f in required_files)
        }
    
    def _check_ootdiffusion(self) -> Dict[str, Any]:
        """OOTDiffusion 모델 확인"""
        ootd_paths = [
            self.backend_root / "ai_models" / "OOTDiffusion",
            self.backend_root / "ai_models" / "oot_diffusion",
            self.backend_root / "ai_models" / "checkpoints" / "ootdiffusion"
        ]
        
        for path in ootd_paths:
            if path.exists():
                return {"exists": True, "path": str(path)}
        
        return {"exists": False, "checked_paths": [str(p) for p in ootd_paths]}
    
    def _check_stable_diffusion(self) -> Dict[str, Any]:
        """Stable Diffusion 모델 확인"""
        sd_path = self.backend_root / "ai_models" / "checkpoints" / "stable-diffusion-v1-5"
        return {"exists": sd_path.exists(), "path": str(sd_path)}
    
    def _calculate_models_size(self) -> float:
        """모델 총 크기 계산 (GB)"""
        ai_models_path = self.backend_root / "ai_models"
        if not ai_models_path.exists():
            return 0.0
        
        total_size = 0
        for root, dirs, files in os.walk(ai_models_path):
            for file in files:
                file_path = Path(root) / file
                try:
                    total_size += file_path.stat().st_size
                except (OSError, FileNotFoundError):
                    pass
        
        return round(total_size / (1024**3), 2)
    
    def _identify_issues(self) -> Tuple[List[Dict], List[Dict]]:
        """누락 항목 및 문제 항목 식별"""
        missing_items = []
        problematic_items = []
        
        # 필수 파일 확인
        critical_files = [
            ("backend/app/__init__.py", "앱 패키지 초기화"),
            ("backend/app/api/pipeline_routes.py", "핵심 API 라우트"),
            ("backend/app/models/schemas.py", "데이터 스키마"),
            ("backend/app/services/model_manager.py", "모델 관리자"),
            ("backend/run_server.py", "서버 실행 파일"),
            ("backend/requirements.txt", "의존성 목록")
        ]
        
        for file_path, description in critical_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_items.append({
                    "type": "critical_file",
                    "path": file_path,
                    "description": description,
                    "priority": "HIGH"
                })
        
        # 모델 파일 확인
        model_checks = [
            ("backend/ai_models/clip-vit-base-patch32/model.safetensors", "CLIP 모델"),
            ("backend/ai_models/checkpoints", "AI 모델 체크포인트 디렉토리")
        ]
        
        for model_path, description in model_checks:
            full_path = self.project_root / model_path
            if not full_path.exists():
                missing_items.append({
                    "type": "model_file",
                    "path": model_path,
                    "description": description,
                    "priority": "MEDIUM"
                })
        
        return missing_items, problematic_items
    
    def generate_fixes(self) -> Dict[str, List[str]]:
        """수정 방안 생성"""
        analysis = self.analyze_current_structure()
        fixes = {
            "immediate": [],  # 즉시 수정 가능
            "download_required": [],  # 다운로드 필요
            "manual_review": []  # 수동 검토 필요
        }
        
        # 즉시 수정 가능한 항목들
        for item in analysis["missing_items"]:
            if item["type"] == "critical_file":
                if item["path"].endswith("__init__.py"):
                    fixes["immediate"].append(f"touch {item['path']}")
                elif "schemas.py" in item["path"]:
                    fixes["immediate"].append(f"생성: {item['path']} - 데이터 스키마 파일")
                elif "pipeline_routes.py" in item["path"]:
                    fixes["manual_review"].append(f"검토 필요: {item['path']} - API 라우트 복구")
            
            elif item["type"] == "model_file":
                if "clip" in item["path"].lower():
                    fixes["download_required"].append("CLIP 모델 다운로드 필요")
                else:
                    fixes["download_required"].append(f"모델 다운로드: {item['description']}")
        
        return fixes
    
    def create_fix_script(self, output_file: str = "fix_project_structure.sh") -> str:
        """수정 스크립트 생성"""
        analysis = self.analyze_current_structure()
        fixes = self.generate_fixes()
        
        script_content = f"""#!/bin/bash
# MyCloset AI 프로젝트 구조 수정 스크립트
# 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# 시스템: {self.system_info.get('platform', 'Unknown')} / {self.system_info.get('cpu', 'Unknown')}

set -e  # 오류 시 스크립트 중단

echo "🔧 MyCloset AI 프로젝트 구조 수정 시작..."
echo "=================================================="

# 색상 정의
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
NC='\\033[0m' # No Color

log_info() {{ echo -e "${{BLUE}}ℹ️  $1${{NC}}"; }}
log_success() {{ echo -e "${{GREEN}}✅ $1${{NC}}"; }}
log_warning() {{ echo -e "${{YELLOW}}⚠️  $1${{NC}}"; }}
log_error() {{ echo -e "${{RED}}❌ $1${{NC}}"; }}

# 1. 필수 디렉토리 생성
log_info "Step 1: 필수 디렉토리 생성"
mkdir -p backend/app/{{api,core,models,services,utils}}
mkdir -p backend/app/ai_pipeline/{{steps,utils}}
mkdir -p backend/{{ai_models,static,tests,scripts,logs}}
mkdir -p backend/static/{{uploads,results}}
mkdir -p backend/scripts/{{test,utils,download}}

# .gitkeep 파일 생성
touch backend/static/uploads/.gitkeep
touch backend/static/results/.gitkeep
touch backend/logs/.gitkeep

log_success "디렉토리 구조 생성 완료"

# 2. 필수 __init__.py 파일 생성
log_info "Step 2: Python 패키지 초기화 파일 생성"
"""

        # __init__.py 파일들 생성
        init_files = [
            "backend/app/__init__.py",
            "backend/app/api/__init__.py",
            "backend/app/core/__init__.py",
            "backend/app/models/__init__.py",
            "backend/app/services/__init__.py",
            "backend/app/utils/__init__.py",
            "backend/app/ai_pipeline/__init__.py",
            "backend/app/ai_pipeline/steps/__init__.py",
            "backend/app/ai_pipeline/utils/__init__.py",
            "backend/tests/__init__.py"
        ]
        
        for init_file in init_files:
            script_content += f'touch {init_file}\n'
        
        script_content += f"""
log_success "__init__.py 파일들 생성 완료"

# 3. 시스템별 최적화 설정
log_info "Step 3: 시스템 최적화 설정"
"""

        # 시스템별 설정
        if "M3" in self.system_info.get("cpu", ""):
            script_content += """
# M3 Max 최적화 설정
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
log_success "M3 Max GPU 최적화 설정 완료"
"""
        else:
            script_content += """
# 일반 시스템 설정
export CUDA_VISIBLE_DEVICES=0
log_info "CUDA 설정 완료"
"""

        script_content += f"""
# 4. 모델 다운로드 체크
log_info "Step 4: AI 모델 상태 확인"

if [ ! -f "backend/ai_models/clip-vit-base-patch32/model.safetensors" ]; then
    log_warning "CLIP 모델이 없습니다. 다운로드가 필요합니다."
    echo "다음 명령어로 CLIP 모델을 다운로드하세요:"
    echo "python3 -c \\"from transformers import CLIPModel, CLIPProcessor; model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); model.save_pretrained('./backend/ai_models/clip-vit-base-patch32')\\"" 
else
    log_success "CLIP 모델 확인됨"
fi

# 5. 권한 설정
log_info "Step 5: 파일 권한 설정"
chmod +x backend/run_server.py 2>/dev/null || log_warning "run_server.py 권한 설정 실패"
chmod +x backend/scripts/test/*.py 2>/dev/null || log_info "테스트 스크립트 권한 설정"

log_success "프로젝트 구조 수정 완료!"
echo ""
echo "📊 수정 완료 상태:"
echo "=================================="
"""

        # 수정 완료 후 상태 요약
        for category, items in fixes.items():
            if items:
                category_name = {
                    "immediate": "즉시 수정됨",
                    "download_required": "다운로드 필요", 
                    "manual_review": "수동 검토 필요"
                }[category]
                
                script_content += f'echo "📋 {category_name}:"\n'
                for item in items:
                    script_content += f'echo "   - {item}"\n'

        script_content += """
echo ""
echo "🚀 다음 단계:"
echo "1. python3 backend/scripts/test/test_final_models.py  # 모델 테스트"
echo "2. python3 backend/run_server.py  # 서버 시작"
echo "3. 브라우저에서 http://localhost:8000 접속"
"""
        
        # 스크립트 파일 생성
        script_path = self.project_root / output_file
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # 실행 권한 부여
        os.chmod(script_path, 0o755)
        
        return str(script_path)
    
    def print_analysis_report(self) -> None:
        """분석 보고서 출력"""
        analysis = self.analyze_current_structure()
        
        print(f"🔍 MyCloset AI 프로젝트 구조 분석 보고서")
        print("=" * 60)
        print(f"📅 분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💻 시스템: {self.system_info.get('platform')} / {self.system_info.get('cpu')}")
        print(f"🔧 최적화: {self.system_info.get('optimization')}")
        print()
        
        # 백엔드 상태
        backend = analysis["current_structure"]["backend"]
        print("🏗️ 백엔드 구조 상태:")
        print(f"   ✅ 백엔드 존재: {'예' if backend['exists'] else '❌ 아니오'}")
        
        if backend["exists"]:
            for dir_name, info in backend["directories"].items():
                status = "✅" if info["exists"] else "❌"
                print(f"   {status} {dir_name}/: {'존재' if info['exists'] else '누락'}")
        
        # AI 모델 상태
        ai_models = analysis["current_structure"]["ai_models"]
        print("\n🤖 AI 모델 상태:")
        if ai_models["exists"]:
            print(f"   ✅ AI 모델 디렉토리: 존재")
            print(f"   📊 총 크기: {ai_models['total_size_gb']} GB")
            
            clip = ai_models["clip_model"]
            if clip["exists"]:
                status = "✅ 완전" if clip["complete"] else "⚠️ 불완전"
                print(f"   {status} CLIP 모델: {len(clip['existing_files'])}개 파일")
            else:
                print(f"   ❌ CLIP 모델: 누락")
        else:
            print(f"   ❌ AI 모델 디렉토리: 누락")
        
        # 누락 항목
        missing = analysis["missing_items"]
        if missing:
            print(f"\n⚠️ 누락된 중요 항목 ({len(missing)}개):")
            for item in missing:
                priority_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
                icon = priority_icon.get(item["priority"], "🔹")
                print(f"   {icon} {item['description']}")
                print(f"      경로: {item['path']}")
        else:
            print("\n✅ 모든 필수 항목이 존재합니다!")
        
        # 수정 방안
        fixes = self.generate_fixes()
        print(f"\n🔧 수정 방안:")
        for category, items in fixes.items():
            if items:
                category_names = {
                    "immediate": "즉시 수정 가능",
                    "download_required": "다운로드 필요",
                    "manual_review": "수동 검토 필요"
                }
                print(f"   📋 {category_names[category]} ({len(items)}개):")
                for item in items[:3]:  # 최대 3개만 표시
                    print(f"      • {item}")
                if len(items) > 3:
                    print(f"      ... 외 {len(items)-3}개")
        
        print("\n" + "=" * 60)
        print("💡 다음 단계 권장사항:")
        print("1. ./fix_project_structure.sh 실행으로 자동 수정")
        print("2. AI 모델 다운로드 (필요시)")
        print("3. python backend/scripts/test/test_final_models.py 로 테스트")
        print("4. python backend/run_server.py 로 서버 시작")
        
    def save_analysis_json(self, output_file: str = "project_analysis.json") -> str:
        """분석 결과를 JSON으로 저장"""
        analysis = self.analyze_current_structure()
        
        json_path = self.project_root / output_file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        
        return str(json_path)

def main():
    """메인 실행 함수"""
    print("🚀 MyCloset AI 프로젝트 구조 분석기")
    print("=" * 60)
    
    # 프로젝트 루트 찾기
    current_dir = Path.cwd()
    project_root = current_dir
    
    # backend 디렉토리가 있는 곳을 프로젝트 루트로 설정
    if (current_dir / "backend").exists():
        project_root = current_dir
    elif (current_dir.parent / "backend").exists():
        project_root = current_dir.parent
    elif (current_dir / "mycloset-ai" / "backend").exists():
        project_root = current_dir / "mycloset-ai"
    
    print(f"📁 프로젝트 루트: {project_root}")
    
    # 분석기 초기화
    analyzer = ProjectStructureAnalyzer(str(project_root))
    
    # 분석 실행
    analyzer.print_analysis_report()
    
    # JSON 보고서 저장
    json_file = analyzer.save_analysis_json()
    print(f"\n💾 상세 분석 결과 저장: {json_file}")
    
    # 수정 스크립트 생성
    script_file = analyzer.create_fix_script()
    print(f"🔧 수정 스크립트 생성: {script_file}")
    
    print(f"\n✨ 분석 완료! 수정을 위해 다음 명령어를 실행하세요:")
    print(f"   chmod +x {script_file}")
    print(f"   ./{script_file}")

if __name__ == "__main__":
    main()