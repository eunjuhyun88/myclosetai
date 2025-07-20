#!/usr/bin/env python3
"""
🔍 MyCloset AI - 전체 시스템 AI 모델 검색 스크립트
macOS 전체에서 AI 모델 파일들을 찾아내는 강력한 검색 도구

기능:
- 전체 시스템 스캔 (/, /Users, /Applications 등)
- 특정 모델 검색 (unet_vton, stable-diffusion 등)
- 크기/날짜 필터링
- 중복 제거 및 분석
- 안전한 복사 및 배치

사용법:
python comprehensive_finder.py                           # 전체 AI 모델 검색
python comprehensive_finder.py --model unet_vton         # 특정 모델만 검색
python comprehensive_finder.py --deep-scan               # 깊은 검색 (오래 걸림)
python comprehensive_finder.py --downloads-only          # 다운로드 폴더만
python comprehensive_finder.py --copy-best               # 최고 후보 자동 복사
"""

import os
import sys
import hashlib
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import argparse
import json

# 안전한 import (conda 환경 호환)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️ tqdm 없음. 진행률 표시 불가")

@dataclass
class ModelCandidate:
    """모델 후보 정보"""
    path: Path
    name: str
    size_mb: float
    model_type: str
    confidence: float
    reason: List[str] = field(default_factory=list)
    checksum: str = ""
    last_modified: float = 0.0
    file_count: int = 0
    source_category: str = "unknown"

class ComprehensiveModelFinder:
    """전체 시스템 AI 모델 검색기"""
    
    def __init__(self, target_model: str = "unet_vton", deep_scan: bool = False):
        self.target_model = target_model.lower()
        self.deep_scan = deep_scan
        self.found_candidates: List[ModelCandidate] = []
        
        # AI 모델 확장자들
        self.model_extensions = {
            '.pth', '.pt', '.bin', '.safetensors', '.ckpt', '.h5', 
            '.pb', '.onnx', '.tflite', '.pkl', '.pickle', '.model',
            '.weights', '.params', '.caffemodel', '.prototxt'
        }
        
        # 검색 경로들 (우선순위별)
        self.search_paths = self._get_comprehensive_paths()
        
        # 모델별 검색 패턴
        self.model_patterns = {
            "unet_vton": [
                "*unet*vton*", "*vton*unet*", "*unet_vton*", 
                "*ootd*unet*", "*diffusion*unet*"
            ],
            "stable_diffusion": [
                "*stable*diffusion*", "*sd*", "*runwayml*",
                "*stabilityai*", "*stable-diffusion*"
            ],
            "openpose": [
                "*openpose*", "*pose*model*", "*body*pose*"
            ],
            "u2net": [
                "*u2net*", "*background*removal*", "*salient*"
            ],
            "sam": [
                "*sam*vit*", "*segment*anything*", "*mobile*sam*"
            ],
            "clip": [
                "*clip*vit*", "*openai*clip*", "*vision*transformer*"
            ]
        }
        
        print(f"🎯 검색 대상: {target_model}")
        print(f"🔍 검색 모드: {'깊은 검색' if deep_scan else '빠른 검색'}")
        print(f"📁 검색 경로: {len(self.search_paths)}개")
    
    def _get_comprehensive_paths(self) -> List[Path]:
        """포괄적 검색 경로 목록 생성"""
        paths = []
        home = Path.home()
        
        # 1. 사용자 주요 폴더들
        user_paths = [
            home / "Downloads",
            home / "Desktop", 
            home / "Documents",
            home / "Documents" / "AI_Models",
            home / "Documents" / "models",
            home / "Library" / "Application Support",
            home / "Applications",
        ]
        
        # 2. 개발 관련 폴더들
        dev_paths = [
            home / "anaconda3",
            home / "miniconda3", 
            home / "miniforge3",
            home / "opt" / "homebrew",
            home / "Developer",
            home / "Projects",
            home / "GitHub",
            home / "git",
            home / "code",
            home / "workspace"
        ]
        
        # 3. 캐시 폴더들
        cache_paths = [
            home / ".cache",
            home / ".cache" / "huggingface",
            home / ".cache" / "torch", 
            home / ".cache" / "transformers",
            home / ".cache" / "diffusers",
            home / ".torch",
            home / ".transformers_cache",
            home / ".huggingface"
        ]
        
        # 4. conda/pip 환경들
        conda_paths = []
        if os.environ.get('CONDA_PREFIX'):
            conda_base = Path(os.environ['CONDA_PREFIX']).parent.parent
            conda_paths.extend([
                conda_base / "envs",
                conda_base / "pkgs",
                conda_base / "lib"
            ])
        
        # 5. 시스템 경로들 (deep_scan인 경우만)
        system_paths = []
        if self.deep_scan:
            system_paths.extend([
                Path("/opt"),
                Path("/usr/local"), 
                Path("/Applications"),
                Path("/Library"),
                Path("/tmp")
            ])
        
        # 6. 현재 프로젝트 관련
        current_dir = Path.cwd()
        project_paths = [
            current_dir,
            current_dir.parent,
            current_dir / "models",
            current_dir / "checkpoints",
            current_dir / "ai_models"
        ]
        
        # 모든 경로 병합
        all_paths = user_paths + dev_paths + cache_paths + conda_paths + system_paths + project_paths
        
        # 존재하고 접근 가능한 경로만 필터링
        valid_paths = []
        for path in all_paths:
            try:
                if path.exists() and path.is_dir() and os.access(path, os.R_OK):
                    valid_paths.append(path)
            except (OSError, PermissionError):
                continue
        
        # 중복 제거
        unique_paths = []
        seen_paths = set()
        for path in valid_paths:
            resolved = path.resolve()
            if str(resolved) not in seen_paths:
                unique_paths.append(path)
                seen_paths.add(str(resolved))
        
        return unique_paths
    
    def search_with_find_command(self, search_pattern: str, max_depth: int = 10) -> List[Path]:
        """macOS find 명령어를 사용한 빠른 검색"""
        found_paths = []
        
        for search_path in self.search_paths:
            try:
                # find 명령어 실행
                cmd = [
                    "find", str(search_path),
                    "-maxdepth", str(max_depth),
                    "-name", search_pattern,
                    "-type", "f", "-o", "-type", "d"
                ]
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=30
                )
                
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            path = Path(line)
                            if path.exists():
                                found_paths.append(path)
                                
            except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
                continue
        
        return found_paths
    
    def search_with_python(self, patterns: List[str]) -> List[Path]:
        """Python을 사용한 세밀한 검색"""
        found_paths = []
        
        for search_path in self.search_paths:
            try:
                for pattern in patterns:
                    for path in search_path.rglob(pattern):
                        if path.exists():
                            found_paths.append(path)
            except (OSError, PermissionError):
                continue
        
        return found_paths
    
    def comprehensive_search(self) -> List[ModelCandidate]:
        """포괄적 모델 검색"""
        print("🔍 포괄적 AI 모델 검색 시작...")
        
        candidates = []
        patterns = self.model_patterns.get(self.target_model, [f"*{self.target_model}*"])
        
        # 1. find 명령어로 빠른 검색
        print("⚡ find 명령어로 빠른 검색...")
        all_found = []
        for pattern in patterns:
            found = self.search_with_find_command(pattern)
            all_found.extend(found)
            print(f"   패턴 '{pattern}': {len(found)}개 발견")
        
        # 2. Python으로 추가 검색
        print("🐍 Python으로 세밀한 검색...")
        python_found = self.search_with_python(patterns)
        all_found.extend(python_found)
        print(f"   Python 검색: {len(python_found)}개 추가 발견")
        
        # 3. 특별 검색 (HuggingFace, conda 등)
        print("🔍 특별 위치 검색...")
        special_found = self._search_special_locations()
        all_found.extend(special_found)
        print(f"   특별 위치: {len(special_found)}개 발견")
        
        # 중복 제거
        unique_paths = []
        seen_paths = set()
        for path in all_found:
            try:
                resolved = str(path.resolve())
                if resolved not in seen_paths:
                    unique_paths.append(path)
                    seen_paths.add(resolved)
            except OSError:
                continue
        
        print(f"✅ 총 {len(unique_paths)}개 고유 경로 발견")
        
        # 4. 각 경로 분석
        print("📊 후보 분석 중...")
        iterator = tqdm(unique_paths, desc="모델 분석") if TQDM_AVAILABLE else unique_paths
        
        for path in iterator:
            if self._is_valid_candidate(path):
                candidate = self._analyze_candidate(path)
                candidates.append(candidate)
        
        # 5. 신뢰도순 정렬
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        self.found_candidates = candidates
        print(f"🎯 최종 {len(candidates)}개 유효 후보 발견!")
        
        return candidates
    
    def _search_special_locations(self) -> List[Path]:
        """특별 위치들 검색"""
        special_paths = []
        
        # HuggingFace 모델들
        hf_patterns = [
            "~/.cache/huggingface/hub/models--*",
            "~/.cache/huggingface/transformers/*",
            "*/huggingface_cache/models--*"
        ]
        
        for pattern in hf_patterns:
            expanded = Path(pattern).expanduser()
            try:
                if expanded.exists():
                    for path in expanded.parent.rglob(expanded.name):
                        if self.target_model in str(path).lower():
                            special_paths.append(path)
            except (OSError, PermissionError):
                continue
        
        # PyTorch Hub
        torch_hub = Path.home() / ".cache" / "torch" / "hub"
        if torch_hub.exists():
            try:
                for path in torch_hub.rglob("*"):
                    if self.target_model in str(path).lower():
                        special_paths.append(path)
            except (OSError, PermissionError):
                pass
        
        return special_paths
    
    def _is_valid_candidate(self, path: Path) -> bool:
        """유효한 후보인지 판단"""
        try:
            if not path.exists():
                return False
            
            path_str = str(path).lower()
            name_str = path.name.lower()
            
            # 대상 모델 키워드 포함 확인
            if self.target_model not in path_str and self.target_model not in name_str:
                return False
            
            # 제외할 패턴들
            exclude_patterns = [
                '__pycache__', '.git', '.svn', 'node_modules',
                '.DS_Store', 'Thumbs.db', '.tmp', '.temp'
            ]
            
            if any(pattern in path_str for pattern in exclude_patterns):
                return False
            
            # 폴더인 경우
            if path.is_dir():
                # 내부에 모델 파일이 있는지 확인
                try:
                    model_files = []
                    for ext in self.model_extensions:
                        model_files.extend(list(path.rglob(f"*{ext}")))
                    return len(model_files) > 0
                except OSError:
                    return False
            
            # 파일인 경우
            elif path.is_file():
                # 모델 파일 확장자 확인
                if path.suffix.lower() not in self.model_extensions:
                    return False
                
                # 최소 크기 확인 (1MB)
                try:
                    size_mb = path.stat().st_size / (1024 * 1024)
                    return size_mb >= 1.0
                except OSError:
                    return False
            
            return False
            
        except Exception:
            return False
    
    def _analyze_candidate(self, path: Path) -> ModelCandidate:
        """후보 상세 분석"""
        try:
            candidate = ModelCandidate(
                path=path,
                name=path.name,
                size_mb=0.0,
                model_type=self.target_model,
                confidence=0.0,
                reason=[],
                last_modified=0.0,
                file_count=0,
                source_category=self._categorize_source(path)
            )
            
            # 크기 및 파일 수 계산
            if path.is_file():
                stat = path.stat()
                candidate.size_mb = stat.st_size / (1024 * 1024)
                candidate.last_modified = stat.st_mtime
                candidate.file_count = 1
            else:
                # 폴더인 경우
                total_size = 0
                file_count = 0
                latest_time = 0
                
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        try:
                            stat = file_path.stat()
                            total_size += stat.st_size
                            file_count += 1
                            latest_time = max(latest_time, stat.st_mtime)
                        except OSError:
                            pass
                
                candidate.size_mb = total_size / (1024 * 1024)
                candidate.file_count = file_count
                candidate.last_modified = latest_time
            
            # 신뢰도 계산
            candidate.confidence = self._calculate_confidence(candidate)
            
            return candidate
            
        except Exception as e:
            return ModelCandidate(
                path=path,
                name=path.name,
                size_mb=0.0,
                model_type="error",
                confidence=0.0,
                reason=[f"분석 오류: {e}"]
            )
    
    def _categorize_source(self, path: Path) -> str:
        """소스 카테고리 분류"""
        path_str = str(path).lower()
        
        if "downloads" in path_str:
            return "downloads"
        elif "huggingface" in path_str or "hf_" in path_str:
            return "huggingface_cache"
        elif "torch" in path_str and "hub" in path_str:
            return "pytorch_hub"
        elif "conda" in path_str or "anaconda" in path_str or "miniconda" in path_str:
            return "conda_env"
        elif "desktop" in path_str:
            return "desktop"
        elif "documents" in path_str:
            return "documents"
        elif "applications" in path_str:
            return "applications"
        elif "cache" in path_str:
            return "cache"
        elif "git" in path_str or "github" in path_str:
            return "git_repo"
        else:
            return "other"
    
    def _calculate_confidence(self, candidate: ModelCandidate) -> float:
        """신뢰도 점수 계산"""
        confidence = 0.0
        
        path_str = str(candidate.path).lower()
        name_str = candidate.name.lower()
        
        # 이름 매칭 점수
        if candidate.name.lower() == self.target_model:
            confidence += 1.0
            candidate.reason.append("정확한 이름 매칭")
        elif self.target_model in name_str:
            confidence += 0.7
            candidate.reason.append("이름에 대상 포함")
        elif self.target_model in path_str:
            confidence += 0.5
            candidate.reason.append("경로에 대상 포함")
        
        # 크기 점수
        if 100 <= candidate.size_mb <= 5000:  # 100MB-5GB
            confidence += 0.3
            candidate.reason.append("적정 크기 범위")
        elif candidate.size_mb > 5000:
            confidence += 0.1
            candidate.reason.append("대용량 모델")
        elif candidate.size_mb < 10:
            confidence -= 0.2
            candidate.reason.append("크기가 작음")
        
        # 소스 점수
        source_scores = {
            "huggingface_cache": 0.3,
            "pytorch_hub": 0.2,
            "downloads": 0.2,
            "git_repo": 0.1,
            "conda_env": 0.1,
            "desktop": 0.05,
            "documents": 0.05
        }
        
        if candidate.source_category in source_scores:
            confidence += source_scores[candidate.source_category]
            candidate.reason.append(f"{candidate.source_category} 경로")
        
        # 최신성 점수 (최근 1개월 내)
        if candidate.last_modified > time.time() - (30 * 24 * 3600):
            confidence += 0.1
            candidate.reason.append("최근 파일")
        
        # 파일 수 점수 (폴더인 경우)
        if candidate.file_count > 1:
            confidence += 0.1
            candidate.reason.append(f"{candidate.file_count}개 파일 포함")
        
        return round(confidence, 2)
    
    def print_results(self, limit: int = 20):
        """검색 결과 출력"""
        if not self.found_candidates:
            print("❌ 후보를 찾을 수 없습니다.")
            return
        
        print(f"\n🔍 발견된 '{self.target_model}' 후보들:")
        print("=" * 100)
        
        # 소스별 통계
        source_stats = defaultdict(int)
        for candidate in self.found_candidates:
            source_stats[candidate.source_category] += 1
        
        print(f"\n📊 소스별 분포:")
        for source, count in sorted(source_stats.items()):
            print(f"   {source}: {count}개")
        
        print(f"\n🏆 상위 {min(limit, len(self.found_candidates))}개 후보:")
        
        for i, candidate in enumerate(self.found_candidates[:limit], 1):
            confidence_emoji = "🟢" if candidate.confidence >= 1.0 else "🟡" if candidate.confidence >= 0.5 else "🔴"
            type_emoji = "📁" if candidate.path.is_dir() else "📄"
            
            print(f"\n{i}. {confidence_emoji} {type_emoji} {candidate.name}")
            print(f"   📍 경로: {candidate.path}")
            print(f"   📊 신뢰도: {candidate.confidence}")
            print(f"   💾 크기: {candidate.size_mb:.1f}MB")
            print(f"   🏷️ 소스: {candidate.source_category}")
            if candidate.path.is_dir():
                print(f"   📁 파일 수: {candidate.file_count}개")
            print(f"   🕒 수정일: {time.strftime('%Y-%m-%d %H:%M', time.localtime(candidate.last_modified))}")
            print(f"   💡 이유: {', '.join(candidate.reason)}")
    
    def save_results(self, output_file: Path):
        """결과를 JSON 파일로 저장"""
        results = {
            "search_info": {
                "target_model": self.target_model,
                "deep_scan": self.deep_scan,
                "timestamp": time.time(),
                "total_candidates": len(self.found_candidates)
            },
            "candidates": []
        }
        
        for candidate in self.found_candidates:
            results["candidates"].append({
                "path": str(candidate.path),
                "name": candidate.name,
                "size_mb": candidate.size_mb,
                "confidence": candidate.confidence,
                "source_category": candidate.source_category,
                "file_count": candidate.file_count,
                "last_modified": candidate.last_modified,
                "reason": candidate.reason
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 결과 저장: {output_file}")
    
    def copy_best_candidate(self, target_dir: Path) -> bool:
        """최고 후보를 타겟 디렉토리로 복사"""
        if not self.found_candidates:
            print("❌ 복사할 후보가 없습니다.")
            return False
        
        best_candidate = self.found_candidates[0]
        
        print(f"📋 최고 후보 복사:")
        print(f"   원본: {best_candidate.path}")
        print(f"   신뢰도: {best_candidate.confidence}")
        print(f"   크기: {best_candidate.size_mb:.1f}MB")
        
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            
            if best_candidate.path.is_dir():
                target_path = target_dir / best_candidate.name
                if target_path.exists():
                    backup_path = target_dir / f"{best_candidate.name}_backup_{int(time.time())}"
                    print(f"⚠️ 기존 폴더 백업: {backup_path}")
                    target_path.rename(backup_path)
                
                import shutil
                shutil.copytree(str(best_candidate.path), str(target_path))
                print(f"✅ 폴더 복사 완료: {target_path}")
            else:
                target_path = target_dir / best_candidate.name
                if target_path.exists():
                    backup_path = target_dir / f"{best_candidate.path.stem}_backup_{int(time.time())}{best_candidate.path.suffix}"
                    print(f"⚠️ 기존 파일 백업: {backup_path}")
                    target_path.rename(backup_path)
                
                import shutil
                shutil.copy2(str(best_candidate.path), str(target_path))
                print(f"✅ 파일 복사 완료: {target_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ 복사 실패: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="전체 시스템 AI 모델 검색 도구")
    parser.add_argument("--model", default="unet_vton", help="검색할 모델명")
    parser.add_argument("--deep-scan", action="store_true", help="깊은 검색 활성화")
    parser.add_argument("--downloads-only", action="store_true", help="다운로드 폴더만 검색")
    parser.add_argument("--copy-best", action="store_true", help="최고 후보 자동 복사")
    parser.add_argument("--target-dir", type=Path, help="복사 대상 디렉토리")
    parser.add_argument("--output", type=Path, help="결과 JSON 저장 파일")
    parser.add_argument("--limit", type=int, default=20, help="출력할 최대 후보 수")
    
    args = parser.parse_args()
    
    print("🔍 MyCloset AI - 전체 시스템 모델 검색기")
    print("=" * 60)
    
    # 검색기 초기화
    finder = ComprehensiveModelFinder(
        target_model=args.model,
        deep_scan=args.deep_scan
    )
    
    # 다운로드 폴더만 검색하는 경우
    if args.downloads_only:
        finder.search_paths = [
            Path.home() / "Downloads",
            Path.home() / "Desktop",
            Path.home() / "Documents" / "Downloads"
        ]
        finder.search_paths = [p for p in finder.search_paths if p.exists()]
        print(f"📁 다운로드 폴더만 검색: {len(finder.search_paths)}개 경로")
    
    # 검색 실행
    candidates = finder.comprehensive_search()
    
    # 결과 출력
    finder.print_results(limit=args.limit)
    
    # 결과 저장
    if args.output:
        finder.save_results(args.output)
    
    # 최고 후보 복사
    if args.copy_best and candidates:
        if args.target_dir:
            target_dir = args.target_dir
        else:
            target_dir = Path.cwd() / "backend" / "app" / "ai_pipeline" / "models" / "checkpoints" / f"step_06_virtual_fitting"
        
        print(f"\n🚀 최고 후보를 {target_dir}로 복사...")
        success = finder.copy_best_candidate(target_dir)
        
        if success:
            print("✅ 복사 완료! 검증을 실행하세요.")
            print(f"   python verify_models.py --step 6")

if __name__ == "__main__":
    main()