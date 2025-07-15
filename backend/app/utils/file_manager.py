"""
MyCloset AI - 완전한 파일 관리 유틸리티
✅ M3 Max 최적화
✅ 업로드 파일 처리
✅ 안전한 파일 저장
✅ 검증 기능 포함
✅ 기존 코드와 완전 호환
"""

import os
import uuid
import aiofiles
import asyncio
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from datetime import datetime, timedelta
from fastapi import UploadFile, HTTPException
from PIL import Image
import io

logger = logging.getLogger(__name__)

# 설정 상수들
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB (M3 Max 처리 능력 고려)
MIN_IMAGE_SIZE = (100, 100)
MAX_IMAGE_SIZE = (4096, 4096)  # M3 Max 고해상도 처리 가능
ALLOWED_MIME_TYPES = [
    "image/jpeg", "image/jpg", "image/png", 
    "image/webp", "image/bmp", "image/tiff"
]

class FileManager:
    """
    완전한 파일 관리 유틸리티 클래스
    ✅ 기존 함수명 완전 유지
    ✅ M3 Max 최적화
    ✅ 안전한 파일 처리
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """파일 매니저 초기화"""
        # 기본 디렉토리 설정
        if base_dir is None:
            base_dir = os.getcwd()
        
        self.base_dir = Path(base_dir)
        self.upload_dir = self.base_dir / "uploads"
        self.results_dir = self.base_dir / "results" 
        self.temp_dir = self.base_dir / "temp"
        self.static_dir = self.base_dir / "static"
        
        # 디렉토리 생성
        self._ensure_directories()
        
        # M3 Max 최적화 설정
        self.is_m3_max = self._detect_m3_max()
        self.max_concurrent_ops = 8 if self.is_m3_max else 4
        
        logger.info(f"📁 FileManager 초기화 - M3 Max: {self.is_m3_max}")

    def _detect_m3_max(self) -> bool:
        """M3 Max 감지"""
        try:
            import platform
            import subprocess
            
            if platform.system() == 'Darwin':
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True, timeout=5)
                chip_info = result.stdout.strip()
                return 'M3' in chip_info and 'Max' in chip_info
        except:
            pass
        return False

    def _ensure_directories(self):
        """필요한 디렉토리들 생성"""
        directories = [
            self.upload_dir, self.results_dir, 
            self.temp_dir, self.static_dir,
            self.static_dir / "results",
            self.static_dir / "uploads"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.debug("📁 모든 디렉토리 생성 완료")

    @staticmethod
    async def save_upload_file(
        file: UploadFile, 
        directory: Union[str, Path],
        filename: Optional[str] = None,
        max_size: Optional[int] = None
    ) -> str:
        """
        업로드 파일을 안전하게 저장
        ✅ 기존 함수명 유지 (static method)
        """
        try:
            # 파일 크기 검증
            if max_size is None:
                max_size = MAX_FILE_SIZE
            
            if hasattr(file, 'size') and file.size and file.size > max_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"파일 크기가 {max_size // (1024*1024)}MB를 초과합니다"
                )
            
            # 디렉토리 생성
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            
            # 파일명 생성
            if filename is None:
                file_ext = Path(file.filename).suffix if file.filename else ".tmp"
                filename = f"{uuid.uuid4().hex}{file_ext}"
            
            file_path = directory / filename
            
            # 파일 저장
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                
                # 추가 크기 검증
                if len(content) > max_size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"파일 크기가 {max_size // (1024*1024)}MB를 초과합니다"
                    )
                
                await f.write(content)
            
            logger.info(f"📁 파일 저장 완료: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ 파일 저장 실패: {e}")
            raise

    async def save_session_file(
        self, 
        file: UploadFile, 
        session_id: str, 
        file_type: str = "upload"
    ) -> str:
        """세션별 파일 저장"""
        try:
            # 파일 검증
            if not self.validate_image(file):
                raise HTTPException(
                    status_code=400,
                    detail="지원되지 않는 이미지 형식입니다"
                )
            
            # 파일명 생성
            ext = self.get_file_extension(file.filename)
            filename = f"{session_id}_{file_type}_{uuid.uuid4().hex[:8]}.{ext}"
            
            # 저장 경로 결정
            if file_type in ["person", "clothing", "upload"]:
                save_dir = self.upload_dir
            else:
                save_dir = self.results_dir
            
            file_path = await self.save_upload_file(file, save_dir, filename)
            
            logger.info(f"📁 세션 파일 저장: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"❌ 세션 파일 저장 실패: {e}")
            raise

    def validate_image(self, file: UploadFile) -> bool:
        """
        이미지 파일 검증
        ✅ 기존 함수와 완전 호환
        """
        try:
            # 파일명 검증
            if not file.filename:
                logger.warning("⚠️ 파일명이 없습니다")
                return False
            
            # 확장자 검증
            extension = self.get_file_extension(file.filename)
            if extension not in ALLOWED_EXTENSIONS:
                logger.warning(f"⚠️ 지원되지 않는 확장자: {extension}")
                return False
            
            # MIME 타입 검증
            if not file.content_type or file.content_type not in ALLOWED_MIME_TYPES:
                logger.warning(f"⚠️ 지원되지 않는 MIME 타입: {file.content_type}")
                return False
            
            # 파일 크기 검증
            if hasattr(file, 'size') and file.size:
                if file.size > MAX_FILE_SIZE:
                    logger.warning(f"⚠️ 파일 크기 초과: {file.size} bytes")
                    return False
                if file.size < 1024:  # 1KB 미만
                    logger.warning(f"⚠️ 파일이 너무 작습니다: {file.size} bytes")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 이미지 검증 실패: {e}")
            return False

    @staticmethod
    def validate_measurements(height: float, weight: float) -> bool:
        """
        신체 측정값 검증
        ✅ 기존 함수와 완전 호환
        """
        try:
            # 키 검증 (cm)
            if not (100 <= height <= 250):
                logger.warning(f"⚠️ 키 범위 초과: {height}cm")
                return False
            
            # 체중 검증 (kg)
            if not (30 <= weight <= 300):
                logger.warning(f"⚠️ 체중 범위 초과: {weight}kg")
                return False
            
            # BMI 검증
            bmi = weight / ((height / 100) ** 2)
            if not (10 <= bmi <= 50):
                logger.warning(f"⚠️ BMI 범위 초과: {bmi}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 측정값 검증 실패: {e}")
            return False

    async def validate_image_content(self, image_bytes: bytes) -> bool:
        """
        이미지 내용 검증
        ✅ 기존 함수와 완전 호환 (인스턴스 메서드로 확장)
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            # 이미지 크기 검증
            width, height = image.size
            
            # 최소 크기 검증
            if width < MIN_IMAGE_SIZE[0] or height < MIN_IMAGE_SIZE[1]:
                logger.warning(f"⚠️ 이미지가 너무 작습니다: {width}x{height}")
                return False
            
            # 최대 크기 검증 (M3 Max는 더 큰 이미지 처리 가능)
            max_width, max_height = MAX_IMAGE_SIZE
            if self.is_m3_max:
                max_width *= 2  # M3 Max는 8K 이미지까지 처리
                max_height *= 2
            
            if width > max_width or height > max_height:
                logger.warning(f"⚠️ 이미지가 너무 큽니다: {width}x{height}")
                return False
            
            # 이미지 형식 검증
            if image.format not in ['JPEG', 'PNG', 'WEBP', 'BMP', 'TIFF']:
                logger.warning(f"⚠️ 지원되지 않는 이미지 형식: {image.format}")
                return False
            
            # 색상 모드 검증
            if image.mode not in ['RGB', 'RGBA', 'L']:
                logger.warning(f"⚠️ 지원되지 않는 색상 모드: {image.mode}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 이미지 내용 검증 실패: {e}")
            return False

    @staticmethod
    async def validate_image_content_static(image_bytes: bytes) -> bool:
        """
        정적 메서드 버전 - 기존 함수와 완전 호환
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            width, height = image.size
            
            if width < MIN_IMAGE_SIZE[0] or height < MIN_IMAGE_SIZE[1]:
                return False
                
            if width > MAX_IMAGE_SIZE[0] or height > MAX_IMAGE_SIZE[1]:
                return False
            
            return True
        except:
            return False

    def get_file_extension(self, filename: str) -> str:
        """파일 확장자 추출"""
        if not filename:
            return ""
        return filename.split(".")[-1].lower()

    def get_safe_filename(self, filename: str) -> str:
        """안전한 파일명 생성"""
        import re
        # 특수문자 제거
        safe_name = re.sub(r'[^\w\-_\.]', '_', filename)
        return safe_name[:100]  # 길이 제한

    async def save_result_image(
        self, 
        image: Union[Image.Image, bytes], 
        session_id: str,
        result_type: str = "final"
    ) -> str:
        """결과 이미지 저장"""
        try:
            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{session_id}_{result_type}_{timestamp}.jpg"
            file_path = self.results_dir / filename
            
            # 이미지 처리
            if isinstance(image, bytes):
                pil_image = Image.open(io.BytesIO(image))
            else:
                pil_image = image
            
            # RGB 변환
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # M3 Max 최적화된 저장
            quality = 95 if self.is_m3_max else 90
            pil_image.save(file_path, "JPEG", quality=quality, optimize=True)
            
            # 정적 파일용 복사
            static_path = self.static_dir / "results" / filename
            shutil.copy2(file_path, static_path)
            
            logger.info(f"📁 결과 이미지 저장: {file_path}")
            return str(static_path)
            
        except Exception as e:
            logger.error(f"❌ 결과 이미지 저장 실패: {e}")
            raise

    async def cleanup_session_files(self, session_id: str):
        """세션 파일들 정리"""
        try:
            cleaned_count = 0
            
            # 업로드 파일 정리
            for pattern in [f"{session_id}_*"]:
                for file_path in self.upload_dir.glob(pattern):
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"⚠️ 파일 삭제 실패: {file_path} - {e}")
            
            # 임시 파일 정리
            for pattern in [f"{session_id}_*"]:
                for file_path in self.temp_dir.glob(pattern):
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"⚠️ 임시 파일 삭제 실패: {file_path} - {e}")
            
            # 오래된 결과 파일 정리 (24시간 후)
            cutoff_time = datetime.now() - timedelta(hours=24)
            for file_path in self.results_dir.glob(f"{session_id}_*"):
                try:
                    if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_time:
                        file_path.unlink()
                        cleaned_count += 1
                except Exception as e:
                    logger.warning(f"⚠️ 오래된 파일 삭제 실패: {file_path} - {e}")
            
            logger.info(f"🧹 세션 {session_id} 파일 정리 완료: {cleaned_count}개")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"❌ 세션 파일 정리 실패: {e}")
            return 0

    async def cleanup_temp_files(self, max_age_hours: int = 24):
        """임시 파일들 정리"""
        try:
            if not self.temp_dir.exists():
                return 0
            
            current_time = datetime.now()
            max_age_seconds = max_age_hours * 3600
            cleaned_count = 0
            
            for file_path in self.temp_dir.glob("*"):
                if file_path.is_file():
                    try:
                        file_age = current_time.timestamp() - file_path.stat().st_mtime
                        if file_age > max_age_seconds:
                            file_path.unlink()
                            cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"⚠️ 임시 파일 삭제 실패: {file_path} - {e}")
            
            logger.info(f"🧹 임시 파일 정리 완료: {cleaned_count}개")
            return cleaned_count
                        
        except Exception as e:
            logger.error(f"❌ 임시 파일 정리 실패: {e}")
            return 0

    async def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """파일 정보 조회"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return {"exists": False}
            
            stat = file_path.stat()
            
            info = {
                "exists": True,
                "name": file_path.name,
                "size": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "extension": file_path.suffix.lower(),
                "is_image": file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
            }
            
            # 이미지인 경우 추가 정보
            if info["is_image"]:
                try:
                    with Image.open(file_path) as img:
                        info.update({
                            "width": img.width,
                            "height": img.height,
                            "format": img.format,
                            "mode": img.mode,
                            "resolution": f"{img.width}x{img.height}"
                        })
                except:
                    pass
            
            return info
            
        except Exception as e:
            logger.error(f"❌ 파일 정보 조회 실패: {e}")
            return {"exists": False, "error": str(e)}

    async def batch_process_files(
        self, 
        files: List[UploadFile], 
        session_id: str,
        operation: str = "validate"
    ) -> List[Dict[str, Any]]:
        """배치 파일 처리 (M3 Max 병렬 최적화)"""
        try:
            results = []
            
            # M3 Max 병렬 처리
            semaphore = asyncio.Semaphore(self.max_concurrent_ops)
            
            async def process_single_file(file: UploadFile, index: int):
                async with semaphore:
                    try:
                        if operation == "validate":
                            is_valid = self.validate_image(file)
                            return {
                                "index": index,
                                "filename": file.filename,
                                "valid": is_valid,
                                "size": getattr(file, 'size', 0)
                            }
                        elif operation == "save":
                            file_path = await self.save_session_file(
                                file, session_id, f"batch_{index}"
                            )
                            return {
                                "index": index,
                                "filename": file.filename,
                                "saved_path": file_path,
                                "success": True
                            }
                    except Exception as e:
                        return {
                            "index": index,
                            "filename": file.filename,
                            "error": str(e),
                            "success": False
                        }
            
            # 병렬 처리 실행
            tasks = [
                process_single_file(file, i) 
                for i, file in enumerate(files)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 예외 처리
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    processed_results.append({
                        "success": False,
                        "error": str(result)
                    })
                else:
                    processed_results.append(result)
            
            logger.info(f"📁 배치 처리 완료: {len(processed_results)}개 파일")
            return processed_results
            
        except Exception as e:
            logger.error(f"❌ 배치 파일 처리 실패: {e}")
            return []

    def get_storage_stats(self) -> Dict[str, Any]:
        """저장소 통계 조회"""
        try:
            def get_dir_size(directory: Path) -> int:
                total = 0
                if directory.exists():
                    for file_path in directory.rglob('*'):
                        if file_path.is_file():
                            try:
                                total += file_path.stat().st_size
                            except:
                                pass
                return total
            
            upload_size = get_dir_size(self.upload_dir)
            results_size = get_dir_size(self.results_dir)
            temp_size = get_dir_size(self.temp_dir)
            static_size = get_dir_size(self.static_dir)
            
            stats = {
                "directories": {
                    "upload": {
                        "path": str(self.upload_dir),
                        "size_bytes": upload_size,
                        "size_mb": round(upload_size / (1024 * 1024), 2),
                        "files": len(list(self.upload_dir.glob('*'))) if self.upload_dir.exists() else 0
                    },
                    "results": {
                        "path": str(self.results_dir),
                        "size_bytes": results_size,
                        "size_mb": round(results_size / (1024 * 1024), 2),
                        "files": len(list(self.results_dir.glob('*'))) if self.results_dir.exists() else 0
                    },
                    "temp": {
                        "path": str(self.temp_dir),
                        "size_bytes": temp_size,
                        "size_mb": round(temp_size / (1024 * 1024), 2),
                        "files": len(list(self.temp_dir.glob('*'))) if self.temp_dir.exists() else 0
                    },
                    "static": {
                        "path": str(self.static_dir),
                        "size_bytes": static_size,
                        "size_mb": round(static_size / (1024 * 1024), 2),
                        "files": len(list(self.static_dir.rglob('*'))) if self.static_dir.exists() else 0
                    }
                },
                "total": {
                    "size_bytes": upload_size + results_size + temp_size + static_size,
                    "size_mb": round((upload_size + results_size + temp_size + static_size) / (1024 * 1024), 2),
                    "size_gb": round((upload_size + results_size + temp_size + static_size) / (1024 * 1024 * 1024), 2)
                },
                "limits": {
                    "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
                    "allowed_extensions": ALLOWED_EXTENSIONS,
                    "max_image_size": MAX_IMAGE_SIZE,
                    "min_image_size": MIN_IMAGE_SIZE
                },
                "optimization": {
                    "is_m3_max": self.is_m3_max,
                    "max_concurrent_ops": self.max_concurrent_ops
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ 저장소 통계 조회 실패: {e}")
            return {"error": str(e)}

# ============================================
# 전역 유틸리티 함수들 (기존 호환성)
# ============================================

# 전역 파일 매니저 인스턴스
_global_file_manager = None

def get_file_manager() -> FileManager:
    """전역 파일 매니저 인스턴스 반환"""
    global _global_file_manager
    if _global_file_manager is None:
        _global_file_manager = FileManager()
    return _global_file_manager

# 기존 함수들과의 호환성을 위한 래퍼들
def validate_image(file: UploadFile) -> bool:
    """기존 validate_image 함수와 호환"""
    return get_file_manager().validate_image(file)

def validate_measurements(height: float, weight: float) -> bool:
    """기존 validate_measurements 함수와 호환"""
    return FileManager.validate_measurements(height, weight)

async def validate_image_content(image_bytes: bytes) -> bool:
    """기존 validate_image_content 함수와 호환"""
    return await FileManager.validate_image_content_static(image_bytes)

# ============================================
# 추가 유틸리티 함수들
# ============================================

def get_file_size_str(size_bytes: int) -> str:
    """파일 크기를 읽기 쉬운 문자열로 변환"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"

def is_image_file(filename: str) -> bool:
    """파일명으로 이미지 파일 여부 확인"""
    if not filename:
        return False
    ext = filename.split(".")[-1].lower()
    return ext in ALLOWED_EXTENSIONS

def generate_unique_filename(original_filename: str, prefix: str = "") -> str:
    """유니크한 파일명 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    
    if original_filename:
        name, ext = os.path.splitext(original_filename)
        safe_name = get_file_manager().get_safe_filename(name)
        return f"{prefix}{safe_name}_{timestamp}_{unique_id}{ext}"
    else:
        return f"{prefix}file_{timestamp}_{unique_id}.jpg"

async def save_base64_image(
    base64_data: str, 
    save_path: Union[str, Path],
    max_size: Optional[int] = None
) -> bool:
    """Base64 이미지 데이터를 파일로 저장"""
    try:
        import base64
        
        # data:image/... 프리픽스 제거
        if base64_data.startswith('data:image'):
            header, data = base64_data.split(',', 1)
        else:
            data = base64_data
        
        # 디코딩
        image_bytes = base64.b64decode(data)
        
        # 크기 검증
        if max_size and len(image_bytes) > max_size:
            logger.warning(f"⚠️ Base64 이미지 크기 초과: {len(image_bytes)} bytes")
            return False
        
        # 이미지 검증
        if not await validate_image_content(image_bytes):
            logger.warning("⚠️ Base64 이미지 내용 검증 실패")
            return False
        
        # 파일 저장
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(save_path, 'wb') as f:
            await f.write(image_bytes)
        
        logger.info(f"📁 Base64 이미지 저장 완료: {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Base64 이미지 저장 실패: {e}")
        return False

logger.info("✅ FileManager 모듈 로드 완료 - 모든 기능 포함")