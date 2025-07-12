import os
import uuid
import aiofiles
from pathlib import Path
from typing import Optional
from fastapi import UploadFile

from app.core.config import settings

class FileManager:
    """File management utilities"""
    
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.results_dir = Path(settings.RESULTS_DIR)
        
        # 디렉토리 생성
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_image(self, file: UploadFile) -> bool:
        """Validate uploaded image file"""
        # 파일 크기 확인
        if hasattr(file, 'size') and file.size > settings.MAX_UPLOAD_SIZE:
            return False
        
        # 확장자 확인
        if file.filename:
            ext = file.filename.split('.')[-1].lower()
            if ext not in settings.ALLOWED_EXTENSIONS:
                return False
        
        # MIME 타입 확인
        if file.content_type and not file.content_type.startswith('image/'):
            return False
        
        return True
    
    async def save_upload_file(
        self, 
        file: UploadFile, 
        session_id: str, 
        file_type: str
    ) -> str:
        """Save uploaded file"""
        # 파일명 생성
        ext = file.filename.split('.')[-1] if file.filename else 'jpg'
        filename = f"{session_id}_{file_type}.{ext}"
        file_path = self.upload_dir / filename
        
        # 파일 저장
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        return str(file_path)
    
    async def cleanup_session_files(self, session_id: str):
        """Clean up session files"""
        # 업로드 파일 정리
        for pattern in [f"{session_id}_person.*", f"{session_id}_clothing.*"]:
            for file_path in self.upload_dir.glob(pattern):
                try:
                    file_path.unlink()
                except:
                    pass
        
        # 결과 파일 정리
        for pattern in [f"{session_id}_result.*"]:
            for file_path in self.results_dir.glob(pattern):
                try:
                    file_path.unlink()
                except:
                    pass
