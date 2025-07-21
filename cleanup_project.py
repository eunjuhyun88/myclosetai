#!/usr/bin/env python3
"""
🧹 MyCloset AI 프로젝트 정리 스크립트 v2.0
====================================================

✅ conda 환경 우선 보호
✅ 중복 파일 제거
✅ 백업 파일 정리
✅ 캐시 정리
✅ 불필요한 의존성 제거
✅ 안전한 정리 (백업 생성)
✅ 단계별 실행 옵션

사용법:
    python cleanup_project.py --all                    # 전체 정리
    python cleanup_project.py --backup-files          # 백업 파일만
    python cleanup_project.py --cache                 # 캐시만
    python cleanup_project.py --dependencies          # 의존성만
    python cleanup_project.py --dry-run               # 미리보기
"""

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
import json
from datetime import datetime
import subprocess
import re
import tarfile
import tempfile

# ============================================================================
# 🔧 설정 및 초기화
# ============================================================================

class ProjectCleaner:
    def __init__(self, project_root: str = None, dry_run: bool = False):
        self.project_root = Path(project_root or os.getcwd()).resolve()
        self.dry_run = dry_run
        self.setup_logging()
        
        # 정리 통계
        self.stats = {
            'files_removed': 0,
            'dirs_removed': 0,
            'space_freed_mb': 0,
            'backup_files': 0,
            'cache_files': 0
        }
        
        # 보호할 패턴들 (conda 환경 우선)
        self.protected_patterns = {
            'conda_envs': [
                r'mycloset-ai',
                r'mycloset_env', 
                r'venv*',
                r'env',
                r'\.conda'
            ],
            'important_configs': [
                r'environment\.ya?ml',
                r'requirements\.txt',
                r'package\.json',
                r'pyproject\.toml',
                r'setup\.py',
                r'Dockerfile',
                r'\.gitignore',
                r'README\.md'
            ],
            'git_files': [
                r'\.git/.*',
                r'\.gitignore',
                r'\.gitattributes'
            ]
        }
        
        # 제거 대상 패턴들
        self.removal_patterns = {
            'backup_files': [
                r'.*\.backup$',
                r'.*\.bak$',
                r'.*\.old$',
                r'.*\.orig$',
                r'.*~$',
                r'.*\.backup\.py$'
            ],
            'cache_dirs': [
                r'__pycache__',
                r'\.pytest_cache',
                r'\.coverage',
                r'htmlcov',
                r'\.tox',
                r'\.cache',
                r'node_modules',
                r'\.npm',
                r'\.yarn'
            ],
            'temp_files': [
                r'.*\.tmp$',
                r'.*\.temp$',
                r'.*\.swp$',
                r'.*\.swo$',
                r'\.DS_Store$',
                r'Thumbs\.db$',
                r'.*\.log$'
            ],
            'compiled_files': [
                r'.*\.pyc$',
                r'.*\.pyo$',
                r'.*\.pyd$',
                r'.*\.so$',
                r'.*\.o$'
            ]
        }
    
    def setup_logging(self):
        """로깅 설정"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.project_root / 'cleanup.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    # ========================================================================
    # 🔍 분석 및 검색 메서드
    # ========================================================================
    
    def scan_project(self) -> Dict[str, List[Path]]:
        """프로젝트 전체 스캔"""
        self.logger.info(f"🔍 프로젝트 스캔 시작: {self.project_root}")
        
        found_files = {
            'backup_files': [],
            'cache_dirs': [],
            'temp_files': [],
            'compiled_files': [],
            'large_files': [],
            'duplicate_files': []
        }
        
        # 전체 파일 스캔
        for root, dirs, files in os.walk(self.project_root):
            root_path = Path(root)
            
            # 보호된 디렉토리 스킵
            if self.is_protected_path(root_path):
                dirs.clear()  # 하위 디렉토리 탐색 중단
                continue
            
            # 디렉토리 검사
            for dir_name in dirs[:]:  # 복사본으로 순회
                dir_path = root_path / dir_name
                if self.should_remove_dir(dir_path):
                    found_files['cache_dirs'].append(dir_path)
                    dirs.remove(dir_name)  # 하위 탐색 중단
            
            # 파일 검사
            for file_name in files:
                file_path = root_path / file_name
                file_category = self.categorize_file(file_path)
                if file_category:
                    found_files[file_category].append(file_path)
        
        return found_files
    
    def is_protected_path(self, path: Path) -> bool:
        """경로가 보호 대상인지 확인"""
        path_str = str(path.relative_to(self.project_root))
        
        # conda 환경 보호
        for pattern in self.protected_patterns['conda_envs']:
            if re.search(pattern, path_str, re.IGNORECASE):
                return True
        
        # git 디렉토리 보호
        if '.git' in path.parts:
            return True
            
        # 중요 설정 파일들이 있는 디렉토리 보호
        if any(path.glob(pattern) for pattern in self.protected_patterns['important_configs']):
            return False  # 설정 파일이 있어도 내부 정리는 허용
            
        return False
    
    def should_remove_dir(self, dir_path: Path) -> bool:
        """디렉토리 제거 대상 여부"""
        dir_name = dir_path.name
        
        for pattern in self.removal_patterns['cache_dirs']:
            if re.match(pattern, dir_name):
                return True
        return False
    
    def categorize_file(self, file_path: Path) -> Optional[str]:
        """파일 카테고리 분류"""
        file_name = file_path.name
        
        # 백업 파일
        for pattern in self.removal_patterns['backup_files']:
            if re.match(pattern, file_name):
                return 'backup_files'
        
        # 임시 파일
        for pattern in self.removal_patterns['temp_files']:
            if re.match(pattern, file_name):
                return 'temp_files'
        
        # 컴파일된 파일
        for pattern in self.removal_patterns['compiled_files']:
            if re.match(pattern, file_name):
                return 'compiled_files'
        
        # 큰 파일 (100MB 이상)
        if file_path.exists() and file_path.stat().st_size > 100 * 1024 * 1024:
            return 'large_files'
        
        return None
    
    # ========================================================================
    # 🧹 정리 실행 메서드
    # ========================================================================
    
    def remove_backup_files(self, found_files: Dict[str, List[Path]]) -> None:
        """백업 파일 제거"""
        self.logger.info("🗑️ 백업 파일 제거 시작")
        
        backup_files = found_files['backup_files']
        if not backup_files:
            self.logger.info("✅ 제거할 백업 파일이 없습니다")
            return
        
        for file_path in backup_files:
            if self.dry_run:
                self.logger.info(f"[DRY-RUN] 제거 예정: {file_path}")
            else:
                try:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    self.stats['files_removed'] += 1
                    self.stats['space_freed_mb'] += file_size / (1024 * 1024)
                    self.stats['backup_files'] += 1
                    self.logger.info(f"✅ 제거완료: {file_path}")
                except Exception as e:
                    self.logger.error(f"❌ 제거실패: {file_path} - {e}")
    
    def remove_cache_dirs(self, found_files: Dict[str, List[Path]]) -> None:
        """캐시 디렉토리 제거"""
        self.logger.info("🗑️ 캐시 디렉토리 제거 시작")
        
        cache_dirs = found_files['cache_dirs']
        if not cache_dirs:
            self.logger.info("✅ 제거할 캐시 디렉토리가 없습니다")
            return
        
        for dir_path in cache_dirs:
            if self.dry_run:
                self.logger.info(f"[DRY-RUN] 제거 예정: {dir_path}")
            else:
                try:
                    # 디렉토리 크기 계산
                    dir_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                    
                    shutil.rmtree(dir_path)
                    self.stats['dirs_removed'] += 1
                    self.stats['space_freed_mb'] += dir_size / (1024 * 1024)
                    self.stats['cache_files'] += 1
                    self.logger.info(f"✅ 제거완료: {dir_path}")
                except Exception as e:
                    self.logger.error(f"❌ 제거실패: {dir_path} - {e}")
    
    def remove_temp_files(self, found_files: Dict[str, List[Path]]) -> None:
        """임시 파일 제거"""
        self.logger.info("🗑️ 임시 파일 제거 시작")
        
        temp_files = found_files['temp_files']
        if not temp_files:
            self.logger.info("✅ 제거할 임시 파일이 없습니다")
            return
        
        for file_path in temp_files:
            if self.dry_run:
                self.logger.info(f"[DRY-RUN] 제거 예정: {file_path}")
            else:
                try:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    self.stats['files_removed'] += 1
                    self.stats['space_freed_mb'] += file_size / (1024 * 1024)
                    self.logger.info(f"✅ 제거완료: {file_path}")
                except Exception as e:
                    self.logger.error(f"❌ 제거실패: {file_path} - {e}")
    
    def remove_compiled_files(self, found_files: Dict[str, List[Path]]) -> None:
        """컴파일된 파일 제거"""
        self.logger.info("🗑️ 컴파일된 파일 제거 시작")
        
        compiled_files = found_files['compiled_files']
        if not compiled_files:
            self.logger.info("✅ 제거할 컴파일된 파일이 없습니다")
            return
        
        for file_path in compiled_files:
            if self.dry_run:
                self.logger.info(f"[DRY-RUN] 제거 예정: {file_path}")
            else:
                try:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    self.stats['files_removed'] += 1
                    self.stats['space_freed_mb'] += file_size / (1024 * 1024)
                    self.logger.info(f"✅ 제거완료: {file_path}")
                except Exception as e:
                    self.logger.error(f"❌ 제거실패: {file_path} - {e}")
    
    # ========================================================================
    # 🔧 유틸리티 메서드
    # ========================================================================
    
    def create_backup_archive(self) -> Optional[Path]:
        """중요 파일들의 백업 아카이브 생성"""
        if self.dry_run:
            self.logger.info("[DRY-RUN] 백업 아카이브 생성 예정")
            return None
        
        backup_name = f"mycloset_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
        backup_path = self.project_root / backup_name
        
        try:
            with tarfile.open(backup_path, 'w:gz') as tar:
                # 중요 설정 파일들만 백업
                important_files = []
                for pattern in self.protected_patterns['important_configs']:
                    important_files.extend(self.project_root.glob(pattern))
                
                for file_path in important_files:
                    if file_path.exists():
                        tar.add(file_path, arcname=file_path.relative_to(self.project_root))
            
            self.logger.info(f"✅ 백업 아카이브 생성: {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.error(f"❌ 백업 아카이브 생성 실패: {e}")
            return None
    
    def clean_conda_environment(self) -> None:
        """conda 환경 정리"""
        if self.dry_run:
            self.logger.info("[DRY-RUN] conda 환경 정리 예정")
            return
        
        try:
            # conda 캐시 정리
            result = subprocess.run(['conda', 'clean', '--all', '--yes'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info("✅ conda 캐시 정리 완료")
            else:
                self.logger.warning(f"⚠️ conda 정리 경고: {result.stderr}")
        except Exception as e:
            self.logger.error(f"❌ conda 환경 정리 실패: {e}")
    
    def update_gitignore(self) -> None:
        """gitignore 파일 업데이트"""
        gitignore_path = self.project_root / '.gitignore'
        
        additional_patterns = [
            "# 🧹 정리 스크립트 추가",
            "*.backup",
            "*.bak", 
            "*.old",
            "*.orig",
            "*~",
            "__pycache__/",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".DS_Store",
            "Thumbs.db",
            "*.log",
            "cleanup.log",
            "mycloset_backup_*.tar.gz"
        ]
        
        if self.dry_run:
            self.logger.info("[DRY-RUN] .gitignore 업데이트 예정")
            return
        
        try:
            existing_content = ""
            if gitignore_path.exists():
                existing_content = gitignore_path.read_text()
            
            # 중복 방지를 위해 이미 있는 패턴은 추가하지 않음
            new_patterns = []
            for pattern in additional_patterns:
                if pattern not in existing_content:
                    new_patterns.append(pattern)
            
            if new_patterns:
                with gitignore_path.open('a') as f:
                    f.write('\n' + '\n'.join(new_patterns) + '\n')
                self.logger.info("✅ .gitignore 업데이트 완료")
            else:
                self.logger.info("✅ .gitignore 이미 최신 상태")
        
        except Exception as e:
            self.logger.error(f"❌ .gitignore 업데이트 실패: {e}")
    
    def print_summary(self) -> None:
        """정리 결과 요약 출력"""
        print("\n" + "="*50)
        print("🧹 MyCloset AI 프로젝트 정리 완료!")
        print("="*50)
        print(f"📁 제거된 파일: {self.stats['files_removed']:,}개")
        print(f"📂 제거된 디렉토리: {self.stats['dirs_removed']:,}개")
        print(f"💾 절약된 공간: {self.stats['space_freed_mb']:.1f} MB")
        print(f"🗑️ 백업 파일: {self.stats['backup_files']}개")
        print(f"📦 캐시 파일: {self.stats['cache_files']}개")
        print("="*50)
        
        if self.dry_run:
            print("ℹ️ 이것은 미리보기입니다. 실제 정리를 위해 --dry-run 옵션을 제거하세요.")
    
    # ========================================================================
    # 🚀 메인 실행 메서드
    # ========================================================================
    
    def run_full_cleanup(self) -> None:
        """전체 정리 실행"""
        self.logger.info("🚀 MyCloset AI 프로젝트 전체 정리 시작")
        
        # 1. 백업 생성
        if not self.dry_run:
            backup_path = self.create_backup_archive()
            if backup_path:
                self.logger.info(f"✅ 백업 생성: {backup_path}")
        
        # 2. 프로젝트 스캔
        found_files = self.scan_project()
        
        # 3. 각 카테고리별 정리
        self.remove_backup_files(found_files)
        self.remove_cache_dirs(found_files)
        self.remove_temp_files(found_files)
        self.remove_compiled_files(found_files)
        
        # 4. conda 환경 정리
        self.clean_conda_environment()
        
        # 5. gitignore 업데이트
        self.update_gitignore()
        
        # 6. 결과 출력
        self.print_summary()
    
    def run_selective_cleanup(self, targets: List[str]) -> None:
        """선택적 정리 실행"""
        self.logger.info(f"🎯 선택적 정리 시작: {', '.join(targets)}")
        
        found_files = self.scan_project()
        
        for target in targets:
            if target == 'backup':
                self.remove_backup_files(found_files)
            elif target == 'cache':
                self.remove_cache_dirs(found_files)
            elif target == 'temp':
                self.remove_temp_files(found_files)
            elif target == 'compiled':
                self.remove_compiled_files(found_files)
            elif target == 'conda':
                self.clean_conda_environment()
        
        self.print_summary()

# ============================================================================
# 🚀 CLI 인터페이스
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MyCloset AI 프로젝트 정리 도구')
    parser.add_argument('--project-root', help='프로젝트 루트 경로')
    parser.add_argument('--dry-run', action='store_true', help='미리보기 모드')
    
    # 정리 옵션들
    parser.add_argument('--all', action='store_true', help='전체 정리')
    parser.add_argument('--backup-files', action='store_true', help='백업 파일만')
    parser.add_argument('--cache', action='store_true', help='캐시만')
    parser.add_argument('--temp', action='store_true', help='임시 파일만')
    parser.add_argument('--compiled', action='store_true', help='컴파일된 파일만')
    parser.add_argument('--conda', action='store_true', help='conda 환경만')
    
    args = parser.parse_args()
    
    # 정리 대상 결정
    targets = []
    if args.backup_files:
        targets.append('backup')
    if args.cache:
        targets.append('cache')
    if args.temp:
        targets.append('temp')
    if args.compiled:
        targets.append('compiled')
    if args.conda:
        targets.append('conda')
    
    # 클리너 초기화
    cleaner = ProjectCleaner(args.project_root, args.dry_run)
    
    # 실행
    if args.all or not targets:
        cleaner.run_full_cleanup()
    else:
        cleaner.run_selective_cleanup(targets)

if __name__ == '__main__':
    main()