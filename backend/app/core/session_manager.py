    # backend/app/core/session_manager.py - 신규 생성
import json
import time
import uuid
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from PIL import Image
import aiofiles
import logging

logger = logging.getLogger(__name__)

class SessionData:
    """세션 데이터 클래스"""
    def __init__(self, session_id: str, session_dir: Path):
        self.session_id = session_id
        self.session_dir = session_dir
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.metadata = {}
        self.step_results = {}
        
    def update_access_time(self):
        self.last_accessed = datetime.now()
        
    def is_expired(self, max_age_hours: int = 24) -> bool:
        """세션 만료 확인"""
        age = datetime.now() - self.created_at
        return age > timedelta(hours=max_age_hours)

class SessionManager:
    """세션 기반 이미지 및 데이터 관리"""
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path("static/sessions")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.sessions: Dict[str, SessionData] = {}
        self.max_sessions = 100
        self.session_max_age_hours = 24
        
        # 세션 정리 태스크
        self._cleanup_task = None
        
    async def create_session(
        self, 
        person_image: Image.Image,
        clothing_image: Image.Image,
        measurements: Dict[str, Any]
    ) -> str:
        """새 세션 생성 및 이미지 저장"""
        try:
            # 세션 ID 생성
            session_id = f"session_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            session_dir = self.base_path / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            
            # 이미지 저장 (고품질)
            person_path = session_dir / "person_image.jpg"
            clothing_path = session_dir / "clothing_image.jpg"
            
            person_image.save(person_path, "JPEG", quality=95, optimize=True)
            clothing_image.save(clothing_path, "JPEG", quality=95, optimize=True)
            
            # 세션 데이터 생성
            session_data = SessionData(session_id, session_dir)
            session_data.metadata = {
                "session_id": session_id,
                "created_at": session_data.created_at.isoformat(),
                "measurements": measurements,
                "image_paths": {
                    "person": str(person_path),
                    "clothing": str(clothing_path)
                },
                "image_info": {
                    "person_size": person_image.size,
                    "clothing_size": clothing_image.size,
                    "person_mode": person_image.mode,
                    "clothing_mode": clothing_image.mode
                }
            }
            
            # 메타데이터 저장
            await self._save_session_metadata(session_data)
            
            # 세션 등록
            self.sessions[session_id] = session_data
            
            # 세션 수 제한
            await self._cleanup_old_sessions()
            
            logger.info(f"✅ 새 세션 생성: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"❌ 세션 생성 실패: {e}")
            raise
    
    async def get_session_images(self, session_id: str) -> Tuple[Image.Image, Image.Image]:
        """세션에서 이미지 로드"""
        session_data = self.sessions.get(session_id)
        if not session_data:
            raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
        
        session_data.update_access_time()
        
        try:
            person_path = session_data.metadata["image_paths"]["person"]
            clothing_path = session_data.metadata["image_paths"]["clothing"]
            
            person_image = Image.open(person_path)
            clothing_image = Image.open(clothing_path)
            
            return person_image, clothing_image
            
        except Exception as e:
            logger.error(f"❌ 세션 이미지 로드 실패 {session_id}: {e}")
            raise
    
    async def save_step_result(
        self, 
        session_id: str, 
        step_id: int,
        result: Dict[str, Any],
        result_image: Optional[Image.Image] = None
    ):
        """단계별 결과 저장"""
        session_data = self.sessions.get(session_id)
        if not session_data:
            logger.warning(f"⚠️ 세션 없음 - 결과 저장 실패: {session_id}")
            return
        
        session_data.update_access_time()
        
        try:
            # 결과 이미지 저장
            if result_image:
                results_dir = session_data.session_dir / "results"
                results_dir.mkdir(exist_ok=True)
                
                result_image_path = results_dir / f"step_{step_id}_result.jpg"
                result_image.save(result_image_path, "JPEG", quality=90)
                result["result_image_path"] = str(result_image_path)
            
            # 결과 데이터 저장
            session_data.step_results[str(step_id)] = {
                **result,
                "timestamp": datetime.now().isoformat(),
                "step_id": step_id
            }
            
            # 메타데이터 업데이트
            await self._save_session_metadata(session_data)
            
            logger.info(f"✅ Step {step_id} 결과 저장: {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Step {step_id} 결과 저장 실패 {session_id}: {e}")
    
    async def get_session_data(self, session_id: str) -> Optional[SessionData]:
        """세션 데이터 조회"""
        session_data = self.sessions.get(session_id)
        if session_data:
            session_data.update_access_time()
        return session_data
    
    async def cleanup_session(self, session_id: str):
        """특정 세션 정리"""
        session_data = self.sessions.pop(session_id, None)
        if session_data:
            try:
                # 디렉토리 삭제
                import shutil
                shutil.rmtree(session_data.session_dir)
                logger.info(f"🧹 세션 정리 완료: {session_id}")
            except Exception as e:
                logger.error(f"❌ 세션 정리 실패 {session_id}: {e}")
    
    async def cleanup_all_sessions(self):
        """모든 세션 정리"""
        session_ids = list(self.sessions.keys())
        for session_id in session_ids:
            await self.cleanup_session(session_id)
        
        logger.info(f"🧹 전체 세션 정리 완료: {len(session_ids)}개")
    
    async def _save_session_metadata(self, session_data: SessionData):
        """세션 메타데이터 저장"""
        metadata_path = session_data.session_dir / "session_metadata.json"
        metadata = {
            **session_data.metadata,
            "step_results": session_data.step_results,
            "last_accessed": session_data.last_accessed.isoformat()
        }
        
        async with aiofiles.open(metadata_path, 'w') as f:
            await f.write(json.dumps(metadata, indent=2, ensure_ascii=False))
    
    async def _cleanup_old_sessions(self):
        """오래된 세션 정리"""
        if len(self.sessions) <= self.max_sessions:
            return
        
        # 만료된 세션 찾기
        expired_sessions = [
            session_id for session_id, session_data in self.sessions.items()
            if session_data.is_expired(self.session_max_age_hours)
        ]
        
        # 만료된 세션 정리
        for session_id in expired_sessions:
            await self.cleanup_session(session_id)
        
        # 여전히 너무 많으면 오래된 세션부터 정리
        if len(self.sessions) > self.max_sessions:
            sorted_sessions = sorted(
                self.sessions.items(),
                key=lambda x: x[1].last_accessed
            )
            
            sessions_to_remove = sorted_sessions[:len(self.sessions) - self.max_sessions]
            for session_id, _ in sessions_to_remove:
                await self.cleanup_session(session_id)
        
        logger.info(f"🧹 세션 정리: {len(expired_sessions)}개 만료 세션 제거")

# 전역 세션 매니저 인스턴스
_session_manager_instance: Optional[SessionManager] = None

def get_session_manager() -> SessionManager:
    """세션 매니저 싱글톤 인스턴스 반환"""
    global _session_manager_instance
    if _session_manager_instance is None:
        _session_manager_instance = SessionManager()
    return _session_manager_instance