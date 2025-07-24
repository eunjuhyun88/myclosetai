"""
MyCloset AI - 통합 파일 관리 유틸리티 v3.0
================================================
✅ 기존 FileManager 모든 기능 유지
✅ 스마트 백업 정책 통합
✅ 백업 파일 자동 정리
✅ M3 Max 최적화 유지
✅ conda 환경 완벽 지원
✅ 기존 API 100% 호환
✅ 프로덕션 레벨 안정성
✅ .bak 파일 생성 방지 및 정리
"""

import os
import uuid
import aiofiles
import asyncio
import logging
import tempfile
import shutil
import hashlib
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from fastapi import UploadFile, HTTPException
from PIL import Image
import io

logger = logging.getLogger(__name__)

# =============================================================================
# 🔥 백업 정책 설정 (스마트 백업 시스템)
# =============================================================================

class BackupPolicy(Enum):
    """백업 정책 정의"""
    NONE = "none"              # 백업 안함 (추천 - .bak 파일 생성 방지)
    SMART = "smart"            # 스마트 백업 (중요한 것만)
    TIMESTAMP = "timestamp"    # 타임스탬프 기반
    VERSION = "version"        # 버전 기반
    SESSION = "session"        # 세션 기반

@dataclass
class BackupConfig:
    """백업 설정"""
    policy: BackupPolicy = BackupPolicy.NONE  # 기본값: 백업 안함 (.bak 방지)
    max_backups_per_file: int = 2  # 최대 백업 수 제한
    max_backup_age_days: int = 3   # 백업 보관 기간 단축
    auto_cleanup: bool = True      # 자동 정리 활성화
    cleanup_interval_hours: int = 12  # 정리 주기 단축
    backup_important_only: bool = True  # 중요한 파일만 백업
    preserve_original: bool = True      # 원본 보존
    use_hidden_backup_dir: bool = True  # 숨김 디렉토리 사용

# 설정 상수들 (기존 유지)
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB (M3 Max 처리 능력 고려)
MIN_IMAGE_SIZE = (100, 100)
MAX_IMAGE_SIZE = (4096, 4096)  # M3 Max 고해상도 처리 가능
ALLOWED_MIME_TYPES = [
    "image/jpeg", "image/jpg", "image/png", 
    "image/webp", "image/bmp", "image/tiff"
]

# 중요한 파일 패턴들 (백업이 필요한 경우에만)
IMPORTANT_FILE_PATTERNS = {
    "config_files": ["*config*.yaml", "*config*.yml", "main.py", "__init__.py"],
    "requirements": ["requirements*.txt", "environment*.yml"],
    "critical_scripts": ["main.py", "app.py", "server.py"],
}

# 백업하지 않을 파일 패턴들 (.bak 방지용)
EXCLUDE_BACKUP_PATTERNS = [
    "*.pyc", "*.pyo", "__pycache__/*", "*.log", "*.tmp", "*.temp",
    "*.bak", "*.backup", "*~", ".DS_Store", "Thumbs.db",
    "*.pid", "*.lock", "*.cache"
]

class UnifiedFileManager:
    """
    통합 파일 관리자 - 기존 기능 + 스마트 백업
    ✅ 모든 기존 메서드 유지
    ✅ 스마트 백업 추가
    ✅ .bak 파일 생성 방지
    ✅ 자동 정리 시스템
    """
    def __init__(self, base_dir: Optional[str] = None, backup_config: Optional[BackupConfig] = None):
        """초기화 - 기존 FileManager와 완전 호환 + backend/backend 문제 해결"""
        
        # 🔥 기본 디렉토리 설정 - backend/backend 문제 완전 해결
        if base_dir is None:            
            # ✅ 해결된 코드: 파일 위치 기반으로 backend 경로 계산
            current_file = Path(__file__).absolute()  # /path/to/backend/app/utils/file_manager.py
            backend_root = current_file.parent.parent.parent  # /path/to/backend/
            base_dir = str(backend_root)
            
            print(f"🔧 UnifiedFileManager 자동 경로 설정: {base_dir}")
        
        self.base_dir = Path(base_dir)
        
        # 디렉토리 구조를 static 하위로 정리 (더 깔끔한 구조)
        self.upload_dir = self.base_dir / "static" / "uploads"
        self.results_dir = self.base_dir / "static" / "results" 
        self.temp_dir = self.base_dir / "temp"
        self.static_dir = self.base_dir / "static"
        
        # 백업 설정 (기본값: 백업 안함)
        self.backup_config = backup_config or BackupConfig()
        
        # 백업 디렉토리 (숨김 디렉토리로 설정)
        if self.backup_config.use_hidden_backup_dir:
            self.backup_dir = self.base_dir / ".smart_backups"
        else:
            self.backup_dir = self.base_dir / "backups"
        
        # 디렉토리 생성
        self._ensure_directories()
        
        # M3 Max 최적화 설정 (기존 유지)
        self.is_m3_max = self._detect_m3_max()
        self.max_concurrent_ops = 8 if self.is_m3_max else 4
        
        # 백업 메타데이터 추적
        self.backup_metadata: Dict[str, Dict] = {}
        self._last_cleanup = datetime.now()
        
        # 시작 시 기존 백업 파일 정리
        asyncio.create_task(self._initial_cleanup())
        
        logger.info(f"📁 UnifiedFileManager 초기화 완료")
        logger.info(f"   Base 경로: {self.base_dir}")
        logger.info(f"   Upload 경로: {self.upload_dir}")  
        logger.info(f"   Results 경로: {self.results_dir}")
        logger.info(f"   M3 Max: {self.is_m3_max}")
        logger.info(f"   백업 정책: {self.backup_config.policy.value}")


    # 🎯 핵심 변경사항 요약:
    # 2. uploads → static/uploads (기존 프로젝트 구조와 일치)
    # 3. results → static/results (기존 프로젝트 구조와 일치)
    # 4. 상세한 로그 출력으로 경로 확인 가능


    def _ensure_directories(self):
        """필요한 디렉토리들 생성 (기존 + 백업)"""
        directories = [
            self.upload_dir, self.results_dir, self.temp_dir, 
            self.static_dir, 
            self.static_dir / "results",
            self.static_dir / "uploads"
        ]
        
        # 백업 정책이 NONE이 아닌 경우에만 백업 디렉토리 생성
        if self.backup_config.policy != BackupPolicy.NONE:
            directories.append(self.backup_dir)
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
            # .gitkeep 파일 생성 (빈 디렉토리 보존)
            gitkeep = directory / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.touch()
        
        logger.debug("📁 모든 디렉토리 생성 완료")

    def _detect_m3_max(self) -> bool:
        """M3 Max 감지 (기존 로직 유지)"""
        try:
            import platform
            import subprocess
            
            if platform.system() == 'Darwin':
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True, timeout=5)
                chip_info = result.stdout.strip()
                return 'M3' in chip_info and ('Max' in chip_info or 'Pro' in chip_info)
        except:
            pass
        return False

    async def _initial_cleanup(self):
        """초기화 시 기존 .bak 파일들 정리"""
        try:
            await asyncio.sleep(1)  # 초기화 완료 후 실행
            
            bak_files = list(self.base_dir.rglob("*.bak"))
            backup_files = list(self.base_dir.rglob("*.backup"))
            
            if bak_files or backup_files:
                logger.info(f"🧹 기존 백업 파일 정리 시작: .bak({len(bak_files)}개), .backup({len(backup_files)}개)")
                
                cleaned_count = 0
                for file_path in bak_files + backup_files:
                    try:
                        if file_path.exists() and file_path.is_file():
                            file_path.unlink()
                            cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"⚠️ 백업 파일 삭제 실패: {file_path} - {e}")
                
                logger.info(f"🧹 기존 백업 파일 정리 완료: {cleaned_count}개 삭제")
                
        except Exception as e:
            logger.warning(f"⚠️ 초기 정리 실패: {e}")

    def _should_backup_file(self, file_path: Path) -> bool:
        """파일이 백업이 필요한지 판단"""
        if self.backup_config.policy == BackupPolicy.NONE:
            return False
        
        # 백업 제외 패턴 확인
        file_str = str(file_path).lower()
        for pattern in EXCLUDE_BACKUP_PATTERNS:
            if file_path.match(pattern.lower()):
                return False
        
        if not self.backup_config.backup_important_only:
            return True
        
        # 중요한 파일만 백업
        for category, patterns in IMPORTANT_FILE_PATTERNS.items():
            for pattern in patterns:
                if file_path.match(pattern.lower()):
                    return True
        
        return False

    async def _create_smart_backup(self, file_path: Path, session_id: Optional[str] = None) -> Optional[Path]:
        """스마트 백업 생성 (조건부)"""
        try:
            if not file_path.exists() or not self._should_backup_file(file_path):
                return None
            
            # 백업 경로 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if session_id:
                backup_name = f"{file_path.stem}_{session_id}_{timestamp}{file_path.suffix}"
            else:
                backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            
            backup_path = self.backup_dir / backup_name
            
            # 기존 백업 정리
            await self._cleanup_old_backups_for_file(file_path)
            
            # 파일 복사
            await asyncio.to_thread(shutil.copy2, file_path, backup_path)
            
            # 메타데이터 저장
            self.backup_metadata[str(backup_path)] = {
                "original_path": str(file_path),
                "created_at": datetime.now().isoformat(),
                "session_id": session_id,
                "file_size": backup_path.stat().st_size
            }
            
            logger.debug(f"📁 스마트 백업 생성: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.warning(f"⚠️ 백업 생성 실패: {e}")
            return None

    async def _cleanup_old_backups_for_file(self, original_path: Path):
        """특정 파일의 오래된 백업 정리"""
        if not self.backup_config.auto_cleanup or self.backup_config.policy == BackupPolicy.NONE:
            return
        
        try:
            pattern = f"{original_path.stem}_*{original_path.suffix}"
            existing_backups = list(self.backup_dir.glob(pattern))
            
            # 개수 제한
            if len(existing_backups) >= self.backup_config.max_backups_per_file:
                existing_backups.sort(key=lambda p: p.stat().st_mtime)
                to_delete = existing_backups[:-self.backup_config.max_backups_per_file + 1]
                
                for backup_file in to_delete:
                    try:
                        backup_file.unlink()
                        if str(backup_file) in self.backup_metadata:
                            del self.backup_metadata[str(backup_file)]
                        logger.debug(f"🗑️ 오래된 백업 삭제: {backup_file}")
                    except Exception as e:
                        logger.warning(f"⚠️ 백업 삭제 실패: {e}")
        except Exception as e:
            logger.warning(f"⚠️ 백업 정리 실패: {e}")

    # =============================================================================
    # 🔥 기존 FileManager API - 모든 메서드 유지 (백업 로직 통합)
    # =============================================================================

    @staticmethod
    async def save_upload_file(
        file: UploadFile, 
        directory: Union[str, Path],
        filename: Optional[str] = None,
        max_size: Optional[int] = None
    ) -> str:
        """
        업로드 파일을 안전하게 저장 (기존 API 유지)
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
        """세션별 파일 저장 (스마트 백업 통합)"""
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
            
            # 스마트 백업 (필요한 경우에만)
            if self.backup_config.policy != BackupPolicy.NONE:
                await self._create_smart_backup(Path(file_path), session_id)
            
            logger.info(f"📁 세션 파일 저장: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"❌ 세션 파일 저장 실패: {e}")
            raise

    def validate_image(self, file: UploadFile) -> bool:
        """이미지 파일 검증 (기존 API 유지)"""
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
        """신체 측정값 검증 (기존 API 유지)"""
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
        """이미지 내용 검증 (기존 API 유지)"""
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
                max_width *= 2
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
        """정적 메서드 버전 - 기존 함수와 완전 호환"""
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
        """파일 확장자 추출 (기존 API 유지)"""
        if not filename:
            return ""
        return filename.split(".")[-1].lower()

    def get_safe_filename(self, filename: str) -> str:
        """안전한 파일명 생성 (기존 API 유지)"""
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
        """결과 이미지 저장 (기존 API 유지)"""
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
            
            # 스마트 백업 (결과 이미지는 백업하지 않음)
            
            logger.info(f"📁 결과 이미지 저장: {file_path}")
            return str(static_path)
            
        except Exception as e:
            logger.error(f"❌ 결과 이미지 저장 실패: {e}")
            raise

    async def cleanup_session_files(self, session_id: str):
        """세션 파일들 정리 (기존 API 유지)"""
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
            
            # 세션 관련 백업 정리
            if self.backup_config.policy != BackupPolicy.NONE:
                await self._cleanup_session_backups(session_id)
            
            logger.info(f"🧹 세션 {session_id} 파일 정리 완료: {cleaned_count}개")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"❌ 세션 파일 정리 실패: {e}")
            return 0

    async def _cleanup_session_backups(self, session_id: str):
        """세션 관련 백업 정리"""
        try:
            if not self.backup_dir.exists():
                return
            
            pattern = f"*_{session_id}_*"
            session_backups = list(self.backup_dir.glob(pattern))
            
            for backup_file in session_backups:
                try:
                    backup_file.unlink()
                    if str(backup_file) in self.backup_metadata:
                        del self.backup_metadata[str(backup_file)]
                except Exception as e:
                    logger.warning(f"⚠️ 세션 백업 삭제 실패: {e}")
                    
        except Exception as e:
            logger.warning(f"⚠️ 세션 백업 정리 실패: {e}")

    async def cleanup_temp_files(self, max_age_hours: int = 24):
        """임시 파일들 정리 (기존 API 유지)"""
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
        """파일 정보 조회 (기존 API 유지)"""
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
        """배치 파일 처리 (M3 Max 병렬 최적화) - 기존 API 유지"""
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
        """저장소 통계 조회 (기존 API 유지 + 백업 통계 추가)"""
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
            backup_size = get_dir_size(self.backup_dir) if self.backup_dir.exists() else 0
            
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
                    },
                    "backups": {
                        "path": str(self.backup_dir),
                        "size_bytes": backup_size,
                        "size_mb": round(backup_size / (1024 * 1024), 2),
                        "files": len(list(self.backup_dir.glob('*'))) if self.backup_dir.exists() else 0,
                        "policy": self.backup_config.policy.value
                    }
                },
                "total": {
                    "size_bytes": upload_size + results_size + temp_size + static_size + backup_size,
                    "size_mb": round((upload_size + results_size + temp_size + static_size + backup_size) / (1024 * 1024), 2),
                    "size_gb": round((upload_size + results_size + temp_size + static_size + backup_size) / (1024 * 1024 * 1024), 2)
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
                },
                "backup_config": {
                    "policy": self.backup_config.policy.value,
                    "max_backups_per_file": self.backup_config.max_backups_per_file,
                    "max_age_days": self.backup_config.max_backup_age_days,
                    "auto_cleanup": self.backup_config.auto_cleanup
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ 저장소 통계 조회 실패: {e}")
            return {"error": str(e)}

    # =============================================================================
    # 🔥 추가 백업 관리 메서드들
    # =============================================================================

    async def auto_cleanup_all_backups(self):
        """전체 백업 자동 정리"""
        try:
            if self.backup_config.policy == BackupPolicy.NONE:
                return 0
            
            now = datetime.now()
            
            # 정리 간격 체크
            if (now - self._last_cleanup).total_seconds() < self.backup_config.cleanup_interval_hours * 3600:
                return 0
            
            cutoff_date = now - timedelta(days=self.backup_config.max_backup_age_days)
            cleaned_count = 0
            
            if self.backup_dir.exists():
                for backup_file in self.backup_dir.glob("*"):
                    try:
                        if backup_file.is_file() and datetime.fromtimestamp(backup_file.stat().st_mtime) < cutoff_date:
                            backup_file.unlink()
                            if str(backup_file) in self.backup_metadata:
                                del self.backup_metadata[str(backup_file)]
                            cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"⚠️ 백업 정리 실패: {e}")
            
            self._last_cleanup = now
            
            if cleaned_count > 0:
                logger.info(f"🧹 자동 백업 정리 완료: {cleaned_count}개 파일")
            
            return cleaned_count
                
        except Exception as e:
            logger.error(f"❌ 백업 정리 실패: {e}")
            return 0

    def change_backup_policy(self, new_policy: BackupPolicy):
        """백업 정책 변경"""
        old_policy = self.backup_config.policy
        self.backup_config.policy = new_policy
        
        logger.info(f"📁 백업 정책 변경: {old_policy.value} → {new_policy.value}")
        
        # NONE으로 변경 시 기존 백업 정리 제안
        if new_policy == BackupPolicy.NONE and self.backup_dir.exists():
            asyncio.create_task(self._cleanup_all_existing_backups())

    async def _cleanup_all_existing_backups(self):
        """모든 기존 백업 정리 (정책이 NONE으로 변경될 때)"""
        try:
            if not self.backup_dir.exists():
                return
            
            backup_files = list(self.backup_dir.glob("*"))
            cleaned_count = 0
            
            for backup_file in backup_files:
                try:
                    if backup_file.is_file():
                        backup_file.unlink()
                        cleaned_count += 1
                except Exception as e:
                    logger.warning(f"⚠️ 백업 삭제 실패: {e}")
            
            # 백업 디렉토리 제거 (비어있다면)
            try:
                if not any(self.backup_dir.iterdir()):
                    self.backup_dir.rmdir()
            except:
                pass
            
            self.backup_metadata.clear()
            
            logger.info(f"🧹 모든 백업 파일 정리 완료: {cleaned_count}개")
            
        except Exception as e:
            logger.warning(f"⚠️ 전체 백업 정리 실패: {e}")

# =============================================================================
# 🔥 전역 인스턴스 및 호환성 함수들 (기존 API 완전 호환)
# =============================================================================

# 통합 전역 파일 매니저
_global_unified_file_manager = None

def get_file_manager() -> UnifiedFileManager:
    """전역 파일 매니저 인스턴스 반환 (기존 함수명 유지)"""
    global _global_unified_file_manager
    if _global_unified_file_manager is None:
        # 기본값: 백업 안함 (.bak 파일 생성 방지)
        backup_config = BackupConfig(policy=BackupPolicy.NONE)
        _global_unified_file_manager = UnifiedFileManager(backup_config=backup_config)
    return _global_unified_file_manager

def get_smart_file_manager() -> UnifiedFileManager:
    """스마트 백업 활성화된 파일 매니저 반환"""
    backup_config = BackupConfig(policy=BackupPolicy.SMART)
    return UnifiedFileManager(backup_config=backup_config)

# 기존 함수들과의 호환성 래퍼들 (모든 기존 코드가 그대로 작동)
def validate_image(file: UploadFile) -> bool:
    """기존 validate_image 함수와 호환"""
    return get_file_manager().validate_image(file)

def validate_measurements(height: float, weight: float) -> bool:
    """기존 validate_measurements 함수와 호환"""
    return UnifiedFileManager.validate_measurements(height, weight)

async def validate_image_content(image_bytes: bytes) -> bool:
    """기존 validate_image_content 함수와 호환"""
    return await UnifiedFileManager.validate_image_content_static(image_bytes)

# =============================================================================
# 🔥 추가 유틸리티 함수들 (기존 유지)
# =============================================================================

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

# =============================================================================
# 🔥 백업 파일 정리 유틸리티 함수들
# =============================================================================

async def cleanup_all_bak_files(base_dir: Optional[str] = None) -> int:
    """프로젝트 전체의 .bak 파일들 정리"""
    try:
        # ✅ 수정된 코드
        if base_dir is None:
            # 파일 위치 기반으로 backend 경로 자동 계산
            current_file = Path(__file__).absolute()  # file_manager.py 위치
            backend_root = current_file.parent.parent.parent  # backend/ 경로
            base_dir = str(backend_root)
            print(f"🔧 UnifiedFileManager 경로 고정: {base_dir}")
        base_path = Path(base_dir)
        cleaned_count = 0
        
        # .bak 및 .backup 파일들 찾기
        patterns = ["*.bak", "*.backup", "*~"]
        
        for pattern in patterns:
            for file_path in base_path.rglob(pattern):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                        cleaned_count += 1
                        logger.debug(f"🗑️ 백업 파일 삭제: {file_path}")
                except Exception as e:
                    logger.warning(f"⚠️ 파일 삭제 실패: {file_path} - {e}")
        
        logger.info(f"🧹 전체 백업 파일 정리 완료: {cleaned_count}개")
        return cleaned_count
        
    except Exception as e:
        logger.error(f"❌ 백업 파일 정리 실패: {e}")
        return 0

def enable_smart_backup():
    """스마트 백업 정책으로 전환"""
    manager = get_file_manager()
    manager.change_backup_policy(BackupPolicy.SMART)
    logger.info("✅ 스마트 백업 정책 활성화")

def disable_all_backup():
    """모든 백업 비활성화 (.bak 파일 생성 방지)"""
    manager = get_file_manager()
    manager.change_backup_policy(BackupPolicy.NONE)
    logger.info("✅ 모든 백업 비활성화 - .bak 파일 생성 방지")

logger.info("✅ UnifiedFileManager v3.0 로드 완료")
logger.info("   🔧 기존 FileManager 모든 기능 유지")
logger.info("   🧹 스마트 백업 정책 통합 (.bak 방지)")
logger.info("   🚀 M3 Max 최적화 및 conda 환경 지원")
logger.info("   📁 기본 설정: 백업 비활성화 (NONE 정책)")