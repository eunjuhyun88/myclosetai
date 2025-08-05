"""
파일 업로드 서비스
"""

import os
import traceback
from typing import Optional, Tuple
from fastapi import UploadFile
import logging

logger = logging.getLogger(__name__)


async def process_uploaded_file(file: UploadFile) -> tuple[bool, str, Optional[bytes]]:
    """업로드된 파일을 처리하는 함수"""
    try:
        logger.info(f"🔄 파일 업로드 처리 시작: {file.filename}")
        logger.info(f"🔍 파일 크기: {file.size} bytes")
        logger.info(f"🔍 파일 타입: {file.content_type}")
        
        # 파일 크기 검증 (10MB 제한)
        max_size = 10 * 1024 * 1024  # 10MB
        if file.size and file.size > max_size:
            logger.warning(f"⚠️ 파일 크기 초과: {file.size} bytes > {max_size} bytes")
            return False, f"파일 크기가 너무 큽니다. 최대 {max_size // (1024*1024)}MB까지 허용됩니다.", None
        
        # 파일 타입 검증
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
        if file.content_type not in allowed_types:
            logger.warning(f"⚠️ 지원하지 않는 파일 타입: {file.content_type}")
            return False, f"지원하지 않는 파일 타입입니다. {', '.join(allowed_types)} 형식만 허용됩니다.", None
        
        # 파일 내용 읽기
        content = await file.read()
        if not content:
            logger.warning("⚠️ 빈 파일 업로드")
            return False, "빈 파일은 업로드할 수 없습니다.", None
        
        logger.info(f"✅ 파일 업로드 처리 완료: {file.filename} ({len(content)} bytes)")
        return True, "파일 업로드 성공", content
        
    except Exception as e:
        logger.error(f"❌ 파일 업로드 처리 실패: {e}")
        logger.error(f"❌ 상세 오류: {traceback.format_exc()}")
        return False, f"파일 업로드 처리 중 오류가 발생했습니다: {str(e)}", None 