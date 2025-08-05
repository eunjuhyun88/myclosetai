#!/usr/bin/env python3
"""
세션 정리 스크립트
오래된 세션들을 정리하여 메모리 사용량을 줄입니다.
"""

import os
import shutil
import time
import json
import logging
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SessionCleaner:
    """세션 정리 클래스"""
    
    def __init__(self, sessions_dir: str = "sessions", max_age_hours: int = 24):
        self.sessions_dir = Path(sessions_dir)
        self.max_age_hours = max_age_hours
        self.current_time = time.time()
        
    def get_session_info(self, session_path: Path) -> Dict[str, Any]:
        """세션 정보를 가져옵니다."""
        try:
            session_data_file = session_path / "session_data.json"
            if session_data_file.exists():
                with open(session_data_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                # 세션 생성 시간 추출
                created_time = session_data.get('created_time', 0)
                last_accessed = session_data.get('last_accessed', created_time)
                
                return {
                    'session_id': session_path.name,
                    'created_time': created_time,
                    'last_accessed': last_accessed,
                    'age_hours': (self.current_time - created_time) / 3600,
                    'size_mb': self.get_directory_size(session_path),
                    'files_count': len(list(session_path.rglob('*')))
                }
            else:
                # session_data.json이 없는 경우 디렉토리 생성 시간 사용
                created_time = session_path.stat().st_ctime
                return {
                    'session_id': session_path.name,
                    'created_time': created_time,
                    'last_accessed': created_time,
                    'age_hours': (self.current_time - created_time) / 3600,
                    'size_mb': self.get_directory_size(session_path),
                    'files_count': len(list(session_path.rglob('*')))
                }
        except Exception as e:
            logger.warning(f"세션 정보 읽기 실패 {session_path}: {e}")
            return None
    
    def get_file_session_info(self, file_path: Path) -> Dict[str, Any]:
        """파일 기반 세션 정보를 가져옵니다."""
        try:
            # 파일명에서 세션 ID 추출 (예: session_1754416024_98d0fe59_clothing.jpg)
            filename = file_path.name
            match = re.match(r'session_(\d+)_([a-f0-9]+)_', filename)
            
            if match:
                timestamp = int(match.group(1))
                session_id = match.group(2)
                created_time = timestamp
                
                return {
                    'session_id': session_id,
                    'created_time': created_time,
                    'last_accessed': created_time,
                    'age_hours': (self.current_time - created_time) / 3600,
                    'size_mb': file_path.stat().st_size / (1024 * 1024),
                    'files_count': 1,
                    'file_path': file_path
                }
            else:
                # 파일명 패턴이 맞지 않는 경우 파일 생성 시간 사용
                created_time = file_path.stat().st_ctime
                return {
                    'session_id': file_path.stem,
                    'created_time': created_time,
                    'last_accessed': created_time,
                    'age_hours': (self.current_time - created_time) / 3600,
                    'size_mb': file_path.stat().st_size / (1024 * 1024),
                    'files_count': 1,
                    'file_path': file_path
                }
        except Exception as e:
            logger.warning(f"파일 세션 정보 읽기 실패 {file_path}: {e}")
            return None
    
    def get_directory_size(self, path: Path) -> float:
        """디렉토리 크기를 MB 단위로 반환합니다."""
        try:
            total_size = 0
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)  # MB로 변환
        except Exception:
            return 0.0
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """모든 세션 정보를 가져옵니다."""
        sessions = []
        
        if not self.sessions_dir.exists():
            logger.warning(f"세션 디렉토리가 존재하지 않습니다: {self.sessions_dir}")
            return sessions
        
        # 디렉토리 기반 세션 확인
        for session_path in self.sessions_dir.iterdir():
            if session_path.is_dir():
                session_info = self.get_session_info(session_path)
                if session_info:
                    sessions.append(session_info)
        
        # 파일 기반 세션 확인
        for file_path in self.sessions_dir.iterdir():
            if file_path.is_file() and file_path.name.startswith('session_'):
                session_info = self.get_file_session_info(file_path)
                if session_info:
                    sessions.append(session_info)
        
        # 세션 ID별로 그룹화하여 중복 제거
        session_groups = {}
        for session in sessions:
            session_id = session['session_id']
            if session_id not in session_groups:
                session_groups[session_id] = session
            else:
                # 기존 세션과 병합 (크기와 파일 수 합산)
                existing = session_groups[session_id]
                existing['size_mb'] += session['size_mb']
                existing['files_count'] += session['files_count']
                # 더 오래된 시간 사용
                if session['created_time'] < existing['created_time']:
                    existing['created_time'] = session['created_time']
                    existing['age_hours'] = session['age_hours']
        
        sessions = list(session_groups.values())
        
        # 나이순으로 정렬 (오래된 것부터)
        sessions.sort(key=lambda x: x['age_hours'], reverse=True)
        return sessions
    
    def cleanup_old_sessions(self, dry_run: bool = True) -> Dict[str, Any]:
        """오래된 세션들을 정리합니다."""
        sessions = self.get_all_sessions()
        
        if not sessions:
            logger.info("정리할 세션이 없습니다.")
            return {'cleaned': 0, 'total_size_mb': 0, 'sessions': []}
        
        logger.info(f"총 {len(sessions)}개 세션 발견")
        
        # 정리 대상 세션 찾기
        sessions_to_clean = []
        total_size_to_clean = 0
        
        for session in sessions:
            if session['age_hours'] > self.max_age_hours:
                sessions_to_clean.append(session)
                total_size_to_clean += session['size_mb']
        
        logger.info(f"정리 대상: {len(sessions_to_clean)}개 세션 (총 {total_size_to_clean:.2f}MB)")
        
        if dry_run:
            logger.info("🔍 DRY RUN 모드 - 실제 삭제하지 않습니다")
            for session in sessions_to_clean[:10]:  # 상위 10개만 표시
                logger.info(f"  - {session['session_id']}: {session['age_hours']:.1f}시간 전, {session['size_mb']:.2f}MB")
            if len(sessions_to_clean) > 10:
                logger.info(f"  ... 및 {len(sessions_to_clean) - 10}개 더")
        else:
            logger.info("🗑️ 실제 세션 정리 시작")
            cleaned_count = 0
            
            for session in sessions_to_clean:
                try:
                    if 'file_path' in session:
                        # 파일 기반 세션 삭제
                        file_path = session['file_path']
                        if file_path.exists():
                            file_path.unlink()
                            cleaned_count += 1
                            logger.info(f"✅ 파일 삭제됨: {file_path.name} ({session['size_mb']:.2f}MB)")
                    else:
                        # 디렉토리 기반 세션 삭제
                        session_path = self.sessions_dir / session['session_id']
                        if session_path.exists():
                            shutil.rmtree(session_path)
                            cleaned_count += 1
                            logger.info(f"✅ 디렉토리 삭제됨: {session['session_id']} ({session['size_mb']:.2f}MB)")
                except Exception as e:
                    logger.error(f"❌ 삭제 실패 {session['session_id']}: {e}")
            
            logger.info(f"✅ 정리 완료: {cleaned_count}개 세션 삭제됨")
        
        return {
            'cleaned': len(sessions_to_clean) if not dry_run else 0,
            'total_size_mb': total_size_to_clean,
            'sessions': sessions_to_clean
        }
    
    def cleanup_by_count(self, keep_count: int = 50, dry_run: bool = True) -> Dict[str, Any]:
        """세션 개수 기준으로 정리합니다."""
        sessions = self.get_all_sessions()
        
        if len(sessions) <= keep_count:
            logger.info(f"세션 수가 {keep_count}개 이하입니다. 정리할 필요가 없습니다.")
            return {'cleaned': 0, 'total_size_mb': 0, 'sessions': []}
        
        sessions_to_clean = sessions[keep_count:]  # 오래된 것부터 제거
        total_size_to_clean = sum(s['size_mb'] for s in sessions_to_clean)
        
        logger.info(f"정리 대상: {len(sessions_to_clean)}개 세션 (총 {total_size_to_clean:.2f}MB)")
        
        if dry_run:
            logger.info("🔍 DRY RUN 모드 - 실제 삭제하지 않습니다")
            for session in sessions_to_clean[:10]:
                logger.info(f"  - {session['session_id']}: {session['age_hours']:.1f}시간 전, {session['size_mb']:.2f}MB")
            if len(sessions_to_clean) > 10:
                logger.info(f"  ... 및 {len(sessions_to_clean) - 10}개 더")
        else:
            logger.info("🗑️ 실제 세션 정리 시작")
            cleaned_count = 0
            
            for session in sessions_to_clean:
                try:
                    if 'file_path' in session:
                        # 파일 기반 세션 삭제
                        file_path = session['file_path']
                        if file_path.exists():
                            file_path.unlink()
                            cleaned_count += 1
                            logger.info(f"✅ 파일 삭제됨: {file_path.name} ({session['size_mb']:.2f}MB)")
                    else:
                        # 디렉토리 기반 세션 삭제
                        session_path = self.sessions_dir / session['session_id']
                        if session_path.exists():
                            shutil.rmtree(session_path)
                            cleaned_count += 1
                            logger.info(f"✅ 디렉토리 삭제됨: {session['session_id']} ({session['size_mb']:.2f}MB)")
                except Exception as e:
                    logger.error(f"❌ 삭제 실패 {session['session_id']}: {e}")
            
            logger.info(f"✅ 정리 완료: {cleaned_count}개 세션 삭제됨")
        
        return {
            'cleaned': len(sessions_to_clean) if not dry_run else 0,
            'total_size_mb': total_size_to_clean,
            'sessions': sessions_to_clean
        }
    
    def show_session_stats(self):
        """세션 통계를 표시합니다."""
        sessions = self.get_all_sessions()
        
        if not sessions:
            logger.info("세션이 없습니다.")
            return
        
        total_size = sum(s['size_mb'] for s in sessions)
        avg_age = sum(s['age_hours'] for s in sessions) / len(sessions)
        
        logger.info("📊 세션 통계:")
        logger.info(f"  - 총 세션 수: {len(sessions)}개")
        logger.info(f"  - 총 크기: {total_size:.2f}MB")
        logger.info(f"  - 평균 나이: {avg_age:.1f}시간")
        logger.info(f"  - 가장 오래된 세션: {sessions[0]['age_hours']:.1f}시간 전")
        logger.info(f"  - 가장 최근 세션: {sessions[-1]['age_hours']:.1f}시간 전")
        
        # 나이별 분포
        age_groups = {
            '1시간 이내': 0,
            '1-6시간': 0,
            '6-24시간': 0,
            '24시간 이상': 0
        }
        
        for session in sessions:
            age = session['age_hours']
            if age <= 1:
                age_groups['1시간 이내'] += 1
            elif age <= 6:
                age_groups['1-6시간'] += 1
            elif age <= 24:
                age_groups['6-24시간'] += 1
            else:
                age_groups['24시간 이상'] += 1
        
        logger.info("  - 나이별 분포:")
        for group, count in age_groups.items():
            logger.info(f"    {group}: {count}개")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='세션 정리 스크립트')
    parser.add_argument('--sessions-dir', default='sessions', help='세션 디렉토리 경로')
    parser.add_argument('--max-age', type=int, default=24, help='최대 보관 시간 (시간)')
    parser.add_argument('--keep-count', type=int, default=50, help='보관할 세션 수')
    parser.add_argument('--dry-run', action='store_true', help='실제 삭제하지 않고 미리보기만')
    parser.add_argument('--mode', choices=['age', 'count', 'stats'], default='stats', 
                       help='정리 모드: age(나이 기준), count(개수 기준), stats(통계만)')
    
    args = parser.parse_args()
    
    cleaner = SessionCleaner(args.sessions_dir, args.max_age)
    
    if args.mode == 'stats':
        cleaner.show_session_stats()
    elif args.mode == 'age':
        cleaner.cleanup_old_sessions(dry_run=args.dry_run)
    elif args.mode == 'count':
        cleaner.cleanup_by_count(args.keep_count, dry_run=args.dry_run)

if __name__ == "__main__":
    main() 