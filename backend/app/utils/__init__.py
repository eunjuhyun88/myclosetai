"""
MyCloset AI - Utils 모듈 초기화
pipeline_routes.py에서 필요한 모든 유틸리티 함수/클래스 제공
✅ 실제 구현 + 폴백 지원
✅ 함수명/클래스명 유지
"""

import logging

logger = logging.getLogger(__name__)

# file_manager.py에서 FileManager import
try:
    from .file_manager import FileManager
    logger.info("✅ FileManager import 성공")
except ImportError as e:
    logger.warning(f"⚠️ FileManager import 실패: {e}")
    # 폴백 구현
    class FileManager:
        @staticmethod
        async def save_upload_file(file, directory):
            return f"{directory}/{file.filename}"

# image_utils.py에서 ImageProcessor import
try:
    from .image_utils import ImageProcessor
    logger.info("✅ ImageProcessor import 성공")
except ImportError as e:
    logger.warning(f"⚠️ ImageProcessor import 실패: {e}")
    # 폴백 구현
    class ImageProcessor:
        @staticmethod
        def enhance_image(image):
            return image

__all__ = ['FileManager', 'ImageProcessor']

logger.info("✅ Utils 모듈 로드 완료")