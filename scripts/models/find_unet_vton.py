#!/usr/bin/env python3
"""
🔍 MyCloset AI - unet_vton 모델 찾기 및 배치 스크립트
누락된 unet_vton 모델을 찾아서 올바른 위치에 배치

기능:
- 전체 시스템에서 unet_vton 관련 파일/폴더 검색
- OOTDiffusion 관련 모델 탐지
- 자동 복사 및 배치
- 백업 생성

사용법:
python find_unet_vton.py                    # 검색만
python find_unet_vton.py --copy             # 검색 후 복사
python find_unet_vton.py --verify           # 복사 후 검증
"""

import os
import sys
import shutil
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
import subprocess

# 안전한 import (conda 환경 호환)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️ tqdm 없음. 진행률 표시 불가")

class UnetVtonFinder:
    """unet_vton 모델 찾기 및 배치 도구"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.target_dir = self.project_root / "backend" / "app" / "ai_pipeline" / "models" / "checkpoints" / "step_06_virtual_fitting"
        self.found_candidates: List[Dict] = []
        
        # unet_vton 관련 검색 패턴들
        self.search_patterns = [
            "*unet*vton*",
            "*vton*unet*",
            "*unet_vton*",
            "*unet*",
            "*ootd*unet*",
            "*diffusion*unet*"
        ]
        
        # 폴더 패턴들
        self.folder_patterns = [
            "unet_vton",
            "unet",
            "OOTDiffusion",
            "ootdiffusion", 
            "checkpoints",
            "models"
        ]
        
        # 검색할 경로들
        self.search_paths = self._get_search_paths()
        
        print(f"🎯 프로젝트 루트: {self.project_root}")
        print(f"📁 타겟 디렉토리: {self.target_dir}")
        
    def _get_search_paths(self) -> List[Path]:
        """검색 경로 목록 생성"""
        paths = []
        
        # 1. 프로젝트 내부 경로들
        project_paths = [
            self.project_root,
            self.project_root / "ai_models",
            self.project_root / "backend",
            self.project_root.parent,
        ]
        
        # 2. 시스템 캐시 경로들
        home = Path.home()
        cache_paths = [
            home / ".cache" / "huggingface" / "hub",
            home / ".cache" / "huggingface" / "transformers", 
            home / ".cache" / "torch" / "hub",
            home / "Downloads",
            home / "Desktop",
            home / "Documents"
        ]
        
        # 3. conda 환경 경로
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if conda_env:
            conda_base = os.environ.get('CONDA_PREFIX', home / "anaconda3")
            cache_paths.append(Path(conda_base) / "envs" / conda_env)
        
        # 모든 경로 병합
        all_paths = project_paths + cache_paths
        
        # 존재하고 접근 가능한 경로만 필터링
        valid_paths = []
        for path in all_paths:
            try:
                if path.exists() and path.is_dir() and os.access(path, os.R_OK):
                    valid_paths.append(path)
            except (OSError, PermissionError):
                continue
                
        return valid_paths
    
    def search_unet_vton(self) -> List[Dict]:
        """unet_vton 관련 파일/폴더 검색"""
        print("🔍 unet_vton 모델 검색 시작...")
        
        candidates = []
        
        for search_path in self.search_paths:
            print(f"📂 검색 중: {search_path}")
            
            try:
                # 패턴별 검색
                for pattern in self.search_patterns:
                    for item in search_path.rglob(pattern):
                        if self._is_valid_candidate(item):
                            candidate = self._analyze_candidate(item)
                            candidates.append(candidate)
                
                # 특별 검색: OOTDiffusion 관련 폴더들
                for ootd_path in search_path.rglob("*OOTDiffusion*"):
                    if ootd_path.is_dir():
                        unet_paths = list(ootd_path.rglob("*unet*"))
                        for unet_path in unet_paths:
                            if self._is_valid_candidate(unet_path):
                                candidate = self._analyze_candidate(unet_path)
                                candidate["source"] = "OOTDiffusion"
                                candidates.append(candidate)
                                
            except (PermissionError, OSError) as e:
                print(f"⚠️ 접근 불가: {search_path} - {e}")
                continue
        
        # 중복 제거 (경로 기준)
        unique_candidates = []
        seen_paths = set()
        for candidate in candidates:
            path_str = str(candidate["path"])
            if path_str not in seen_paths:
                unique_candidates.append(candidate)
                seen_paths.add(path_str)
        
        self.found_candidates = unique_candidates
        print(f"✅ {len(unique_candidates)}개 후보 발견!")
        
        return unique_candidates
    
    def _is_valid_candidate(self, path: Path) -> bool:
        """유효한 unet_vton 후보인지 판단"""
        try:
            # 기본 체크
            if not path.exists():
                return False
            
            path_str = str(path).lower()
            name_str = path.name.lower()
            
            # unet 관련 키워드 확인
            unet_keywords = ["unet", "diffusion", "ootd", "vton"]
            has_unet_keyword = any(keyword in path_str for keyword in unet_keywords)
            
            if not has_unet_keyword:
                return False
            
            # 폴더인 경우
            if path.is_dir():
                # 내부에 모델 파일이 있는지 확인
                model_files = list(path.rglob("*.pth")) + list(path.rglob("*.safetensors")) + list(path.rglob("*.bin"))
                return len(model_files) > 0
            
            # 파일인 경우
            elif path.is_file():
                # 모델 파일 확장자 확인
                valid_extensions = {'.pth', '.pt', '.bin', '.safetensors', '.ckpt'}
                if path.suffix.lower() not in valid_extensions:
                    return False
                
                # 파일 크기 확인 (최소 1MB)
                stat = path.stat()
                size_mb = stat.st_size / (1024 * 1024)
                return size_mb >= 1.0
            
            return False
            
        except Exception:
            return False
    
    def _analyze_candidate(self, path: Path) -> Dict:
        """후보 파일/폴더 분석"""
        try:
            candidate = {
                "path": path,
                "name": path.name,
                "type": "folder" if path.is_dir() else "file",
                "size_mb": 0.0,
                "confidence": 0.0,
                "reason": [],
                "source": "search",
                "files_count": 0
            }
            
            # 크기 계산
            if path.is_file():
                candidate["size_mb"] = path.stat().st_size / (1024 * 1024)
                candidate["files_count"] = 1
            else:
                # 폴더인 경우 내부 파일들 크기 합계
                total_size = 0
                file_count = 0
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        try:
                            total_size += file_path.stat().st_size
                            file_count += 1
                        except OSError:
                            pass
                candidate["size_mb"] = total_size / (1024 * 1024)
                candidate["files_count"] = file_count
            
            # 신뢰도 계산
            path_str = str(path).lower()
            name_str = path.name.lower()
            
            # unet_vton 정확한 이름 매칭
            if "unet_vton" in name_str:
                candidate["confidence"] += 0.8
                candidate["reason"].append("정확한 이름 매칭")
            elif "unet" in name_str and "vton" in name_str:
                candidate["confidence"] += 0.6
                candidate["reason"].append("이름에 unet, vton 포함")
            elif "unet" in name_str:
                candidate["confidence"] += 0.4
                candidate["reason"].append("이름에 unet 포함")
            
            # 경로 기반 점수
            if "ootdiffusion" in path_str:
                candidate["confidence"] += 0.3
                candidate["reason"].append("OOTDiffusion 경로")
            elif "virtual_fitting" in path_str:
                candidate["confidence"] += 0.2
                candidate["reason"].append("Virtual Fitting 경로")
            elif "checkpoints" in path_str:
                candidate["confidence"] += 0.1
                candidate["reason"].append("Checkpoints 경로")
            
            # 크기 기반 점수
            if 100 < candidate["size_mb"] < 5000:  # 100MB-5GB 적정 범위
                candidate["confidence"] += 0.2
                candidate["reason"].append("적정 크기 범위")
            elif candidate["size_mb"] > 5000:
                candidate["confidence"] += 0.1
                candidate["reason"].append("대용량 모델")
            
            # 파일 개수 기반 점수 (폴더인 경우)
            if path.is_dir() and candidate["files_count"] > 0:
                candidate["confidence"] += 0.1
                candidate["reason"].append(f"{candidate['files_count']}개 파일 포함")
            
            return candidate
            
        except Exception as e:
            return {
                "path": path,
                "name": path.name,
                "type": "unknown",
                "size_mb": 0.0,
                "confidence": 0.0,
                "reason": [f"분석 오류: {e}"],
                "source": "error",
                "files_count": 0
            }
    
    def print_candidates(self):
        """발견된 후보들 출력"""
        if not self.found_candidates:
            print("❌ unet_vton 관련 파일을 찾을 수 없습니다.")
            return
        
        print(f"\n🔍 발견된 unet_vton 후보들:")
        print("=" * 70)
        
        # 신뢰도순 정렬
        sorted_candidates = sorted(self.found_candidates, key=lambda x: x["confidence"], reverse=True)
        
        for i, candidate in enumerate(sorted_candidates, 1):
            confidence_emoji = "🟢" if candidate["confidence"] >= 0.7 else "🟡" if candidate["confidence"] >= 0.4 else "🔴"
            type_emoji = "📁" if candidate["type"] == "folder" else "📄"
            
            print(f"\n{i}. {confidence_emoji} {type_emoji} {candidate['name']}")
            print(f"   📍 경로: {candidate['path']}")
            print(f"   📊 신뢰도: {candidate['confidence']:.2f}")
            print(f"   💾 크기: {candidate['size_mb']:.1f}MB")
            if candidate["type"] == "folder":
                print(f"   📁 파일 수: {candidate['files_count']}개")
            print(f"   💡 이유: {', '.join(candidate['reason'])}")
    
    def copy_best_candidate(self, candidate_index: Optional[int] = None) -> bool:
        """최적 후보를 타겟 위치로 복사"""
        if not self.found_candidates:
            print("❌ 복사할 후보가 없습니다.")
            return False
        
        # 후보 선택
        if candidate_index is not None:
            if 0 <= candidate_index < len(self.found_candidates):
                candidate = self.found_candidates[candidate_index]
            else:
                print(f"❌ 잘못된 인덱스: {candidate_index}")
                return False
        else:
            # 가장 높은 신뢰도 후보 선택
            sorted_candidates = sorted(self.found_candidates, key=lambda x: x["confidence"], reverse=True)
            candidate = sorted_candidates[0]
        
        source_path = candidate["path"]
        
        print(f"📋 복사 대상:")
        print(f"   원본: {source_path}")
        print(f"   신뢰도: {candidate['confidence']:.2f}")
        print(f"   크기: {candidate['size_mb']:.1f}MB")
        
        # 타겟 경로 결정
        if candidate["type"] == "folder":
            target_path = self.target_dir / "unet_vton"
        else:
            target_path = self.target_dir / f"unet_vton{source_path.suffix}"
        
        print(f"   타겟: {target_path}")
        
        try:
            # 타겟 디렉토리 생성
            self.target_dir.mkdir(parents=True, exist_ok=True)
            
            # 기존 파일/폴더가 있다면 백업
            if target_path.exists():
                backup_path = target_path.parent / f"{target_path.name}_backup_{int(time.time())}"
                print(f"⚠️ 기존 파일 백업: {backup_path}")
                shutil.move(str(target_path), str(backup_path))
            
            # 복사 실행
            if candidate["type"] == "folder":
                print("📁 폴더 복사 중...")
                shutil.copytree(str(source_path), str(target_path))
            else:
                print("📄 파일 복사 중...")
                shutil.copy2(str(source_path), str(target_path))
            
            print(f"✅ 복사 완료: {target_path}")
            return True
            
        except Exception as e:
            print(f"❌ 복사 실패: {e}")
            return False
    
    def verify_placement(self) -> bool:
        """배치된 unet_vton 검증"""
        print("🔍 unet_vton 배치 검증 중...")
        
        # 가능한 unet_vton 경로들
        possible_paths = [
            self.target_dir / "unet_vton",
            self.target_dir / "unet_vton.pth",
            self.target_dir / "unet_vton.safetensors",
            self.target_dir / "unet_vton.bin"
        ]
        
        found_paths = []
        for path in possible_paths:
            if path.exists():
                found_paths.append(path)
        
        if not found_paths:
            print("❌ unet_vton이 배치되지 않았습니다.")
            return False
        
        print(f"✅ unet_vton 발견: {len(found_paths)}개")
        for path in found_paths:
            if path.is_dir():
                file_count = len(list(path.rglob("*")))
                print(f"   📁 {path.name} - {file_count}개 파일")
            else:
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"   📄 {path.name} - {size_mb:.1f}MB")
        
        return True
    
    def run_verification_script(self) -> bool:
        """검증 스크립트 실행"""
        verify_script = self.project_root / "verify_models.py"
        if not verify_script.exists():
            print("⚠️ verify_models.py 스크립트를 찾을 수 없습니다.")
            return False
        
        try:
            print("🔍 모델 검증 스크립트 실행 중...")
            result = subprocess.run([
                "python", str(verify_script), "--step", "6"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                print("✅ 검증 스크립트 실행 성공!")
                # Virtual Fitting 부분만 출력
                output_lines = result.stdout.split('\n')
                in_virtual_fitting = False
                for line in output_lines:
                    if "Virtual Fitting" in line:
                        in_virtual_fitting = True
                    elif in_virtual_fitting and line.startswith('✅') or line.startswith('⚠️') or line.startswith('❌'):
                        if not line.strip().startswith(' '):
                            in_virtual_fitting = False
                    
                    if in_virtual_fitting:
                        print(line)
                
                return True
            else:
                print(f"❌ 검증 스크립트 실행 실패: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ 검증 스크립트 실행 오류: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="unet_vton 모델 찾기 및 배치 도구")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="프로젝트 루트 디렉토리")
    parser.add_argument("--copy", action="store_true", help="검색 후 최적 후보를 자동 복사")
    parser.add_argument("--copy-index", type=int, help="특정 인덱스 후보를 복사 (0부터 시작)")
    parser.add_argument("--verify", action="store_true", help="복사 후 검증 실행")
    parser.add_argument("--interactive", action="store_true", help="대화형 모드")
    
    args = parser.parse_args()
    
    print("🔍 MyCloset AI - unet_vton 찾기 도구")
    print("=" * 50)
    
    # 검색기 초기화
    finder = UnetVtonFinder(project_root=args.project_root)
    
    # unet_vton 검색
    candidates = finder.search_unet_vton()
    
    # 후보 출력
    finder.print_candidates()
    
    if not candidates:
        print("\n💡 해결 방법:")
        print("   1. OOTDiffusion 모델을 HuggingFace에서 다운로드")
        print("   2. 다른 Virtual Fitting 모델 사용")
        print("   3. unet_vton 대신 다른 UNet 모델 활용")
        return
    
    # 대화형 모드
    if args.interactive and not args.copy:
        print(f"\n선택할 후보 번호를 입력하세요 (1-{len(candidates)}, Enter=최고 신뢰도): ", end="")
        try:
            user_input = input().strip()
            if user_input:
                candidate_index = int(user_input) - 1
            else:
                candidate_index = None
        except (ValueError, KeyboardInterrupt):
            print("❌ 취소됨")
            return
        
        args.copy = True
        args.copy_index = candidate_index
    
    # 복사 실행
    if args.copy:
        print("\n" + "=" * 50)
        success = finder.copy_best_candidate(args.copy_index)
        
        if success and args.verify:
            print("\n" + "=" * 50)
            finder.verify_placement()
            finder.run_verification_script()

if __name__ == "__main__":
    main()