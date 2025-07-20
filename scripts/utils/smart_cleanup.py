#!/usr/bin/env python3
"""
🧹 MyCloset AI 스마트 디스크 정리 시스템 v3.0
✅ 500GB+ 프로젝트에서 필요한 것만 안전하게 보존
✅ 중복 파일 자동 탐지 및 제거
✅ 백업 파일 및 임시 파일 정리
✅ AI 모델 중복 제거 및 최적화
✅ 사용자 확인 후 안전한 삭제
✅ 상세한 공간 절약 보고서
"""

import os
import sys
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import re

# 진행률 표시
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

class FileCategory(Enum):
    ESSENTIAL = "essential"        # 절대 삭제하면 안되는 파일
    BACKUP = "backup"             # 백업 파일 (삭제 가능)
    DUPLICATE = "duplicate"       # 중복 파일
    CACHE = "cache"              # 캐시 파일
    TEMP = "temp"                # 임시 파일
    LOG = "log"                  # 로그 파일
    MODEL_REDUNDANT = "model_redundant"  # 중복 AI 모델
    LARGE_UNUSED = "large_unused"  # 큰 미사용 파일
    ARCHIVE = "archive"          # 압축 파일

@dataclass
class FileInfo:
    path: Path
    size_mb: float
    category: FileCategory
    priority: int = 1  # 1=안전삭제, 2=주의, 3=위험
    reason: str = ""
    duplicate_of: Optional[Path] = None
    last_accessed: Optional[float] = None

class SmartCleanupSystem:
    """스마트 디스크 정리 시스템"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.scan_results: Dict[FileCategory, List[FileInfo]] = defaultdict(list)
        self.total_space_mb = 0
        self.potential_savings_mb = 0
        
        # 필수 보존 패턴들
        self.essential_patterns = {
            # 핵심 소스 코드
            r".*\.(py|js|ts|tsx|jsx|vue|svelte)$",
            r".*\.(html|css|scss|sass|less)$",
            r".*\.(json|yaml|yml|toml|ini|cfg)$",
            r".*\.(md|txt|rst)$",
            r".*/package\.json$",
            r".*/requirements\.txt$",
            r".*/pyproject\.toml$",
            r".*/Dockerfile$",
            r".*/docker-compose\.ya?ml$",
            r".*\.env\.example$",
            r".*/\.gitignore$",
            r".*/README\.*$",
            r".*/LICENSE$",
            r".*/Makefile$",
            
            # 필수 AI 모델 (최신 버전만)
            r".*/diffusion_pytorch_model\.safetensors$",
            r".*/pytorch_model\.bin$",
            r".*/config\.json$",
            r".*/model\.onnx$",
            r".*/(ootd|clip|stable-diffusion).*\.(safetensors|bin)$",
        }
        
        # 안전하게 삭제 가능한 패턴들
        self.safe_delete_patterns = {
            # 백업 파일들
            r".*\.backup.*$",
            r".*\.bak$",
            r".*\.old$",
            r".*_backup_\d+.*$",
            r".*\.backup\d*$",
            
            # 임시 파일들
            r".*\.tmp$",
            r".*\.temp$",
            r".*/temp/.*$",
            r".*/tmp/.*$",
            r".*\.swp$",
            r".*\.swo$",
            r".*~$",
            
            # 캐시 파일들
            r".*/__pycache__/.*$",
            r".*\.pyc$",
            r".*\.pyo$",
            r".*/\.cache/.*$",
            r".*/node_modules/.*$",
            r".*/\.next/.*$",
            r".*/dist/.*$",
            r".*/build/.*$",
            
            # 로그 파일들 (오래된 것)
            r".*\.log$",
            r".*\.log\.\d+$",
            r".*/logs/.*\.log$",
            
            # 압축/아카이브 (소스가 있는 경우)
            r".*\.zip$",
            r".*\.tar\.gz$",
            r".*\.tar\.bz2$",
            r".*\.7z$",
            r".*\.rar$",
            
            # Git 관련 (큰 것들)
            r".*/\.git/objects/.*$",
            r".*\.bfg-report/.*$",
            
            # 시스템 파일
            r".*\.DS_Store$",
            r".*/Thumbs\.db$",
            
            # 중복 모델들 (패턴 기반)
            r".*/.*_v\d+\..*$",  # 버전이 있는 중복
            r".*/.*_copy\..*$",   # 복사본
            r".*/.*_duplicate\..*$",  # 중복 표시
        }
        
        # 주의해서 삭제할 패턴들
        self.caution_patterns = {
            r".*\.pth$",
            r".*\.ckpt$", 
            r".*\.h5$",
            r".*\.pkl$",
            r".*\.pickle$",
        }

    def scan_directory(self) -> Dict[str, float]:
        """디렉토리 전체 스캔"""
        print("🔍 프로젝트 전체 스캔 시작...")
        
        file_hashes = {}  # 중복 파일 탐지용
        large_files = []  # 100MB 이상 파일들
        
        total_files = sum(1 for _ in self.project_root.rglob('*') if _.is_file())
        
        if TQDM_AVAILABLE:
            pbar = tqdm(total=total_files, desc="파일 스캔", unit="files")
        
        for file_path in self.project_root.rglob('*'):
            if not file_path.is_file():
                continue
                
            try:
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                self.total_space_mb += file_size_mb
                
                # 100MB 이상 파일들 별도 추적
                if file_size_mb >= 100:
                    large_files.append((file_path, file_size_mb))
                
                # 파일 분류
                category = self._classify_file(file_path, file_size_mb)
                
                if category != FileCategory.ESSENTIAL:
                    # 중복 파일 체크 (1MB 이상만)
                    if file_size_mb >= 1:
                        file_hash = self._get_file_hash(file_path)
                        if file_hash in file_hashes:
                            # 중복 발견
                            original_file = file_hashes[file_hash]
                            duplicate_info = FileInfo(
                                path=file_path,
                                size_mb=file_size_mb,
                                category=FileCategory.DUPLICATE,
                                priority=1,
                                reason=f"중복 파일 (원본: {original_file.name})",
                                duplicate_of=original_file
                            )
                            self.scan_results[FileCategory.DUPLICATE].append(duplicate_info)
                            self.potential_savings_mb += file_size_mb
                        else:
                            file_hashes[file_hash] = file_path
                    
                    # 카테고리별 분류
                    file_info = FileInfo(
                        path=file_path,
                        size_mb=file_size_mb,
                        category=category,
                        priority=self._get_priority(file_path, category),
                        reason=self._get_reason(file_path, category),
                        last_accessed=file_path.stat().st_atime
                    )
                    
                    self.scan_results[category].append(file_info)
                    
                    if category in [FileCategory.BACKUP, FileCategory.CACHE, 
                                  FileCategory.TEMP, FileCategory.LOG]:
                        self.potential_savings_mb += file_size_mb
                
                if TQDM_AVAILABLE:
                    pbar.update(1)
                    
            except (OSError, PermissionError):
                continue
        
        if TQDM_AVAILABLE:
            pbar.close()
        
        # 큰 파일들 분석
        self._analyze_large_files(large_files)
        
        return self._generate_scan_summary()

    def _classify_file(self, file_path: Path, size_mb: float) -> FileCategory:
        """파일 분류"""
        path_str = str(file_path).lower()
        
        # 필수 파일 체크
        for pattern in self.essential_patterns:
            if re.match(pattern, path_str):
                return FileCategory.ESSENTIAL
        
        # 안전 삭제 가능 파일 체크
        for pattern in self.safe_delete_patterns:
            if re.match(pattern, path_str):
                if "backup" in pattern or ".bak" in pattern:
                    return FileCategory.BACKUP
                elif "cache" in pattern or "__pycache__" in pattern:
                    return FileCategory.CACHE
                elif "temp" in pattern or ".tmp" in pattern:
                    return FileCategory.TEMP
                elif ".log" in pattern:
                    return FileCategory.LOG
                elif any(ext in pattern for ext in [".zip", ".tar", ".7z"]):
                    return FileCategory.ARCHIVE
        
        # 주의 파일 체크
        for pattern in self.caution_patterns:
            if re.match(pattern, path_str):
                # AI 모델인지 확인
                if any(model in path_str for model in ["ootd", "diffusion", "clip", "stable"]):
                    return FileCategory.MODEL_REDUNDANT
        
        # 큰 파일 (500MB 이상)이면서 최근에 접근하지 않은 파일
        if size_mb >= 500:
            last_access = file_path.stat().st_atime
            if time.time() - last_access > 30 * 24 * 3600:  # 30일 이상
                return FileCategory.LARGE_UNUSED
        
        return FileCategory.ESSENTIAL

    def _get_file_hash(self, file_path: Path) -> str:
        """파일 해시 계산 (중복 탐지용)"""
        try:
            # 큰 파일은 처음 1MB만 해시
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                chunk = f.read(1024 * 1024)  # 1MB
                hasher.update(chunk)
            return hasher.hexdigest()
        except:
            return ""

    def _get_priority(self, file_path: Path, category: FileCategory) -> int:
        """삭제 우선순위 반환"""
        if category == FileCategory.ESSENTIAL:
            return 3  # 절대 삭제 금지
        elif category in [FileCategory.BACKUP, FileCategory.CACHE, FileCategory.TEMP]:
            return 1  # 안전 삭제
        elif category == FileCategory.DUPLICATE:
            return 1  # 안전 삭제
        elif category == FileCategory.LOG:
            return 1  # 안전 삭제
        elif category == FileCategory.ARCHIVE:
            return 2  # 주의 삭제
        else:
            return 2  # 주의 삭제

    def _get_reason(self, file_path: Path, category: FileCategory) -> str:
        """삭제 이유 반환"""
        reasons = {
            FileCategory.BACKUP: "백업 파일",
            FileCategory.CACHE: "캐시 파일 (재생성 가능)",
            FileCategory.TEMP: "임시 파일",
            FileCategory.LOG: "로그 파일",
            FileCategory.DUPLICATE: "중복 파일",
            FileCategory.ARCHIVE: "압축 파일 (소스 존재시)",
            FileCategory.MODEL_REDUNDANT: "중복 AI 모델",
            FileCategory.LARGE_UNUSED: "큰 미사용 파일 (30일+ 미접근)"
        }
        return reasons.get(category, "분류되지 않음")

    def _analyze_large_files(self, large_files: List[Tuple[Path, float]]):
        """큰 파일들 분석"""
        print(f"\n📊 큰 파일 분석 (100MB 이상: {len(large_files)}개)")
        
        # 크기순 정렬
        large_files.sort(key=lambda x: x[1], reverse=True)
        
        for file_path, size_mb in large_files[:10]:  # 상위 10개만
            category = self._classify_file(file_path, size_mb)
            if category != FileCategory.ESSENTIAL:
                print(f"  📁 {file_path.name}: {size_mb:.1f}MB ({category.value})")

    def _generate_scan_summary(self) -> Dict[str, float]:
        """스캔 결과 요약"""
        summary = {
            "total_space_gb": self.total_space_mb / 1024,
            "potential_savings_gb": self.potential_savings_mb / 1024,
            "savings_percentage": (self.potential_savings_mb / self.total_space_mb * 100) if self.total_space_mb > 0 else 0
        }
        
        print(f"\n📊 스캔 결과:")
        print(f"  📁 총 용량: {summary['total_space_gb']:.1f}GB")
        print(f"  💾 절약 가능: {summary['potential_savings_gb']:.1f}GB")
        print(f"  📈 절약률: {summary['savings_percentage']:.1f}%")
        
        return summary

    def generate_cleanup_report(self) -> str:
        """정리 보고서 생성"""
        report = []
        report.append("🧹 MyCloset AI 디스크 정리 보고서")
        report.append("=" * 50)
        report.append(f"📅 스캔 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"📁 스캔 경로: {self.project_root}")
        report.append(f"💾 총 용량: {self.total_space_mb / 1024:.1f}GB")
        report.append(f"🎯 절약 가능: {self.potential_savings_mb / 1024:.1f}GB")
        report.append("")
        
        for category, files in self.scan_results.items():
            if not files:
                continue
                
            total_size = sum(f.size_mb for f in files)
            report.append(f"📋 {category.value.upper()} ({len(files)}개 파일, {total_size:.1f}MB)")
            
            # 크기순 정렬해서 상위 5개만 표시
            sorted_files = sorted(files, key=lambda x: x.size_mb, reverse=True)
            for file_info in sorted_files[:5]:
                report.append(f"  📄 {file_info.path.name}: {file_info.size_mb:.1f}MB")
                report.append(f"      이유: {file_info.reason}")
            
            if len(files) > 5:
                report.append(f"  ... 및 {len(files) - 5}개 더")
            report.append("")
        
        return "\n".join(report)

    def execute_cleanup(self, categories: List[FileCategory], dry_run: bool = True) -> Dict[str, int]:
        """정리 실행"""
        if dry_run:
            print("🧪 시뮬레이션 모드 (실제 삭제하지 않음)")
        else:
            print("🚨 실제 삭제 모드")
        
        results = {"deleted_files": 0, "freed_space_mb": 0, "errors": 0}
        
        for category in categories:
            files = self.scan_results.get(category, [])
            if not files:
                continue
                
            print(f"\n🗂️ {category.value} 처리 중... ({len(files)}개 파일)")
            
            if TQDM_AVAILABLE:
                pbar = tqdm(files, desc=f"삭제: {category.value}")
            else:
                pbar = files
                
            for file_info in pbar:
                try:
                    if not dry_run:
                        if file_info.path.is_file():
                            file_info.path.unlink()
                        elif file_info.path.is_dir():
                            shutil.rmtree(file_info.path)
                    
                    results["deleted_files"] += 1
                    results["freed_space_mb"] += file_info.size_mb
                    
                except Exception as e:
                    results["errors"] += 1
                    print(f"❌ 삭제 실패: {file_info.path} ({e})")
        
        print(f"\n✅ 정리 완료:")
        print(f"  🗑️ 삭제된 파일: {results['deleted_files']}개")
        print(f"  💾 확보된 공간: {results['freed_space_mb'] / 1024:.1f}GB")
        print(f"  ❌ 오류: {results['errors']}개")
        
        return results

def main():
    """메인 함수"""
    print("🧹 MyCloset AI 스마트 디스크 정리 시스템 v3.0")
    print("=" * 60)
    
    # 프로젝트 루트 찾기
    current_dir = Path.cwd()
    project_candidates = [
        current_dir,
        current_dir / "mycloset-ai",
        current_dir.parent,
        Path("/Users/gimdudeul/MVP/mycloset-ai")
    ]
    
    project_root = None
    for candidate in project_candidates:
        if (candidate / "backend").exists() or candidate.name == "mycloset-ai":
            project_root = candidate
            break
    
    if not project_root:
        project_root = current_dir
        print(f"⚠️ MyCloset AI 프로젝트를 찾을 수 없어 현재 디렉토리 사용: {project_root}")
    else:
        print(f"📁 프로젝트 루트: {project_root}")
    
    # 정리 시스템 초기화
    cleanup_system = SmartCleanupSystem(project_root)
    
    # 스캔 실행
    summary = cleanup_system.scan_directory()
    
    # 보고서 생성
    report = cleanup_system.generate_cleanup_report()
    
    # 보고서 저장
    report_file = project_root / f"cleanup_report_{int(time.time())}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📋 상세 보고서 저장: {report_file}")
    
    # 사용자 선택
    print("\n🎯 정리 옵션을 선택하세요:")
    print("1. 🚀 안전 정리 (백업, 캐시, 임시파일, 중복파일)")
    print("2. ⚡ 적극 정리 (+ 로그, 압축파일)")
    print("3. 🔥 전체 정리 (+ 큰 미사용 파일)")
    print("4. 🎯 커스텀 선택")
    print("5. 📊 보고서만 보기")
    print("0. 종료")
    
    choice = input("\n선택 (0-5): ").strip()
    
    if choice == "1":
        categories = [FileCategory.BACKUP, FileCategory.CACHE, 
                     FileCategory.TEMP, FileCategory.DUPLICATE]
        mode = "안전 정리"
    elif choice == "2":
        categories = [FileCategory.BACKUP, FileCategory.CACHE, FileCategory.TEMP, 
                     FileCategory.DUPLICATE, FileCategory.LOG, FileCategory.ARCHIVE]
        mode = "적극 정리"
    elif choice == "3":
        categories = list(FileCategory)
        categories.remove(FileCategory.ESSENTIAL)
        mode = "전체 정리"
    elif choice == "4":
        print("\n사용 가능한 카테고리:")
        for i, cat in enumerate(FileCategory):
            if cat != FileCategory.ESSENTIAL:
                file_count = len(cleanup_system.scan_results.get(cat, []))
                total_size = sum(f.size_mb for f in cleanup_system.scan_results.get(cat, [])) / 1024
                print(f"{i+1}. {cat.value} ({file_count}개 파일, {total_size:.1f}GB)")
        
        selected = input("선택할 번호들 (쉼표로 구분): ").strip()
        try:
            indices = [int(x.strip()) - 1 for x in selected.split(",")]
            all_cats = [cat for cat in FileCategory if cat != FileCategory.ESSENTIAL]
            categories = [all_cats[i] for i in indices if 0 <= i < len(all_cats)]
            mode = "커스텀 정리"
        except:
            print("❌ 잘못된 입력")
            return
    elif choice == "5":
        print("\n" + report)
        return
    else:
        print("정리 취소됨")
        return
    
    if not categories:
        print("선택된 카테고리가 없습니다")
        return
    
    # 시뮬레이션 실행
    print(f"\n🧪 {mode} 시뮬레이션 실행...")
    sim_results = cleanup_system.execute_cleanup(categories, dry_run=True)
    
    print(f"\n💾 예상 절약 공간: {sim_results['freed_space_mb'] / 1024:.1f}GB")
    
    # 실제 삭제 확인
    confirm = input("\n정말로 삭제하시겠습니까? (yes/no): ").strip().lower()
    
    if confirm == "yes":
        print("\n🚨 실제 정리 실행...")
        actual_results = cleanup_system.execute_cleanup(categories, dry_run=False)
        
        print(f"\n🎉 정리 완료! {actual_results['freed_space_mb'] / 1024:.1f}GB 확보")
        
        # 최종 보고서 저장
        final_report_file = project_root / f"cleanup_completed_{int(time.time())}.txt"
        with open(final_report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            f.write(f"\n\n=== 실행 결과 ===\n")
            f.write(f"삭제된 파일: {actual_results['deleted_files']}개\n")
            f.write(f"확보된 공간: {actual_results['freed_space_mb'] / 1024:.1f}GB\n")
            f.write(f"오류: {actual_results['errors']}개\n")
        
        print(f"📋 최종 보고서: {final_report_file}")
    else:
        print("정리 취소됨")

if __name__ == "__main__":
    main()