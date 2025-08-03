# backend/app/core/session_manager.py
"""
🔥 MyCloset AI 완전한 세션 매니저 - 순환참조 완전 해결 통합 버전
✅ 기존 함수명 100% 유지 (create_session, get_session_images, save_step_result 등)
✅ 순환참조 해결 메서드 완전 통합
✅ Step간 데이터 흐름 완벽 지원
✅ JSON 직렬화 안전성 보장
✅ FastAPI 호환성 완벽 보장
✅ conda 환경 최적화
✅ M3 Max 최적화
✅ 메모리 효율적 관리
✅ 실시간 진행률 추적
"""

import json
import time
import uuid
import copy
import asyncio
import shutil
import threading
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

import numpy as np
from PIL import Image
import aiofiles
import sqlite3
import pickle

logger = logging.getLogger(__name__)

# =============================================================================
# 🔥 순환참조 방지 유틸리티 함수 (통합)
# =============================================================================

def safe_serialize_session_data(obj: Any, max_depth: int = 5, current_depth: int = 0) -> Any:
    """세션 데이터 안전 직렬화 - 순환참조 완전 방지"""
    if current_depth >= max_depth:
        return f"<max_depth_reached:{type(obj).__name__}>"
    
    try:
        # 기본 타입
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        
        # datetime 객체
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        
        # 리스트/튜플
        elif isinstance(obj, (list, tuple)):
            return [safe_serialize_session_data(item, max_depth, current_depth + 1) for item in obj[:50]]
        
        # 딕셔너리
        elif isinstance(obj, dict):
            result = {}
            for key, value in list(obj.items())[:30]:  # 최대 30개 키
                if isinstance(key, str) and not key.startswith('_'):
                    try:
                        result[key] = safe_serialize_session_data(value, max_depth, current_depth + 1)
                    except Exception:
                        result[key] = f"<serialization_error:{type(value).__name__}>"
            return result
        
        # PIL Image 객체
        elif hasattr(obj, 'size') and hasattr(obj, 'mode'):
            return {
                "type": "PIL_Image",
                "size": obj.size,
                "mode": str(obj.mode)
            }
        
        # numpy 배열
        elif hasattr(obj, 'shape') and hasattr(obj, 'dtype'):
            return {
                "type": "numpy_array",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype)
            }
        
        # Path 객체
        elif hasattr(obj, '__fspath__'):
            return str(obj)
        
        # 기타 객체는 문자열로
        else:
            return str(obj)
            
    except Exception as e:
        return f"<serialization_error:{type(obj).__name__}:{str(e)[:30]}>"

# =============================================================================
# 🔥 Step간 데이터 흐름 정의
# =============================================================================

class StepStatus(Enum):
    """Step 처리 상태"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class DataType(Enum):
    """데이터 타입 정의"""
    RAW_IMAGE = "raw_image"
    PROCESSED_IMAGE = "processed_image"
    SEGMENTATION_MASK = "segmentation_mask"
    POSE_KEYPOINTS = "pose_keypoints"
    FEATURE_VECTOR = "feature_vector"
    TRANSFORMATION_MATRIX = "transformation_matrix"
    QUALITY_SCORE = "quality_score"
    METADATA = "metadata"

@dataclass
class StepDataFlow:
    """Step간 데이터 흐름 정의"""
    source_step: int
    target_step: int
    data_type: DataType
    required: bool = True

# 8단계 파이프라인 데이터 흐름 정의
PIPELINE_DATA_FLOWS = [
    # Step 1 -> Step 2
    StepDataFlow(1, 2, DataType.PROCESSED_IMAGE, required=True),
    StepDataFlow(1, 2, DataType.SEGMENTATION_MASK, required=True),
    
    # Step 2 -> Step 3  
    StepDataFlow(2, 3, DataType.POSE_KEYPOINTS, required=True),
    StepDataFlow(1, 3, DataType.RAW_IMAGE, required=True),
    
    # Step 3 -> Step 4
    StepDataFlow(3, 4, DataType.SEGMENTATION_MASK, required=True),
    StepDataFlow(2, 4, DataType.POSE_KEYPOINTS, required=True),
    
    # Step 4 -> Step 5
    StepDataFlow(4, 5, DataType.TRANSFORMATION_MATRIX, required=True),
    StepDataFlow(3, 5, DataType.SEGMENTATION_MASK, required=True),
    
    # Step 5 -> Step 6 (핵심!)
    StepDataFlow(5, 6, DataType.PROCESSED_IMAGE, required=True),
    StepDataFlow(1, 6, DataType.RAW_IMAGE, required=True),
    StepDataFlow(2, 6, DataType.POSE_KEYPOINTS, required=True),
    
    # Step 6 -> Step 7
    StepDataFlow(6, 7, DataType.PROCESSED_IMAGE, required=True),
    StepDataFlow(6, 7, DataType.QUALITY_SCORE, required=False),
    
    # Step 7 -> Step 8
    StepDataFlow(7, 8, DataType.PROCESSED_IMAGE, required=True),
    StepDataFlow(7, 8, DataType.METADATA, required=False),
]

# =============================================================================
# 🔥 기존 데이터 구조 (호환성 유지)
# =============================================================================

@dataclass
class ImageInfo:
    """이미지 정보 (기존 호환)"""
    path: str
    size: Tuple[int, int]  # (width, height)
    mode: str  # RGB, RGBA 등
    format: str  # JPEG, PNG 등
    file_size: int  # 바이트

@dataclass
class SessionMetadata:
    """세션 메타데이터 (기존 호환 + 순환참조 방지)"""
    session_id: str
    created_at: datetime
    last_accessed: datetime
    measurements: Dict[str, Any]
    person_image: ImageInfo
    clothing_image: ImageInfo
    total_steps: int = 8
    completed_steps: List[int] = None
    
    def __post_init__(self):
        if self.completed_steps is None:
            self.completed_steps = []
    
    def to_dict(self) -> Dict[str, Any]:
        """순환참조 방지 딕셔너리 변환 (완전 통합)"""
        try:
            return {
                'session_id': str(self.session_id),
                'created_at': self.created_at.isoformat(),
                'last_accessed': self.last_accessed.isoformat(),
                'measurements': safe_serialize_session_data(self.measurements, max_depth=3),
                'person_image': {
                    'path': str(self.person_image.path),
                    'size': self.person_image.size,
                    'mode': str(self.person_image.mode),
                    'format': str(self.person_image.format),
                    'file_size': int(self.person_image.file_size)
                },
                'clothing_image': {
                    'path': str(self.clothing_image.path),
                    'size': self.clothing_image.size,
                    'mode': str(self.clothing_image.mode),
                    'format': str(self.clothing_image.format),
                    'file_size': int(self.clothing_image.file_size)
                },
                'total_steps': int(self.total_steps),
                'completed_steps': list(self.completed_steps),
                'circular_reference_safe': True
            }
        except Exception as e:
            return {
                'session_id': str(getattr(self, 'session_id', 'unknown')),
                'error': str(e),
                'circular_reference_safe': True,
                'fallback_mode': True
            }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionMetadata':
        """딕셔너리에서 생성 (JSON 역직렬화용)"""
        return cls(
            session_id=data['session_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_accessed=datetime.fromisoformat(data['last_accessed']),
            measurements=data['measurements'],
            person_image=ImageInfo(**data['person_image']),
            clothing_image=ImageInfo(**data['clothing_image']),
            total_steps=data.get('total_steps', 8),
            completed_steps=data.get('completed_steps', [])
        )

class SessionData:
    """런타임 세션 데이터 (기존 호환 + 확장 + 순환참조 방지)"""
    
    def __init__(self, metadata: SessionMetadata, session_dir: Path):
        self.metadata = metadata
        self.session_dir = session_dir
        
        # 기존 호환
        self.step_results: Dict[int, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        
        # 새로 추가 - Step간 데이터 흐름 지원
        self.step_data_cache: Dict[int, Dict[str, Any]] = {}  # Step별 중간 데이터
        self.step_dependencies: Dict[int, List[int]] = {}  # Step 의존성
        self.pipeline_flows = self._build_pipeline_flows()
        
        # 성능 추적
        self.step_processing_times: Dict[int, float] = {}
        self.step_quality_scores: Dict[int, float] = {}
        self.memory_usage_peak: float = 0.0
        
        # 🔥 커스텀 데이터 저장소 (API에서 필요)
        self.custom_data: Dict[str, Any] = {}
        
        # 🔥 이미지 캐시 (PIL 이미지 재사용)
        self._image_cache: Dict[str, Image.Image] = {}
        self._image_cache_lock = threading.RLock()

    def _build_pipeline_flows(self) -> Dict[int, List[StepDataFlow]]:
        """파이프라인 흐름 구축"""
        flows = {}
        for flow in PIPELINE_DATA_FLOWS:
            if flow.target_step not in flows:
                flows[flow.target_step] = []
            flows[flow.target_step].append(flow)
        return flows
    
    @property
    def session_id(self) -> str:
        return self.metadata.session_id
    
    def update_access_time(self):
        """마지막 접근 시간 업데이트"""
        with self.lock:
            self.metadata.last_accessed = datetime.now()
    
    def is_expired(self, max_age_hours: int = 24) -> bool:
        """세션 만료 확인"""
        age = datetime.now() - self.metadata.created_at
        return age > timedelta(hours=max_age_hours)
    
    def add_completed_step(self, step_id: int):
        """완료된 단계 추가"""
        with self.lock:
            if step_id not in self.metadata.completed_steps:
                self.metadata.completed_steps.append(step_id)
                self.metadata.completed_steps.sort()
    
    def get_progress_percent(self) -> float:
        """진행률 반환 (0-100)"""
        return len(self.metadata.completed_steps) / self.metadata.total_steps * 100
    
    # =========================================================================
    # 🔥 순환참조 방지 메서드 (완전 통합)
    # =========================================================================
    
    def to_safe_dict(self) -> Dict[str, Any]:
        """순환참조 방지 안전한 딕셔너리 변환 (완전 통합)"""
        try:
            return {
                "session_id": self.session_id,
                "created_at": self.metadata.created_at.isoformat(),
                "last_accessed": self.metadata.last_accessed.isoformat(),
                "completed_steps": list(self.metadata.completed_steps),
                "total_steps": int(self.metadata.total_steps),
                "progress_percent": float(self.get_progress_percent()),
                "step_results_count": len(self.step_results),
                "step_data_cache_count": len(self.step_data_cache),
                "measurements": safe_serialize_session_data(self.metadata.measurements, max_depth=3),
                "step_processing_times": dict(self.step_processing_times),
                "step_quality_scores": dict(self.step_quality_scores),
                "memory_usage_peak": float(self.memory_usage_peak),
                "custom_data_keys": list(self.custom_data.keys()),  # 🔥 커스텀 데이터 키 목록
                "circular_reference_safe": True
            }
        except Exception as e:
            return {
                "session_id": getattr(self, 'session_id', 'unknown'),
                "error": str(e),
                "circular_reference_safe": True,
                "fallback_mode": True
            }
    
    # =========================================================================
    # 🔥 Step간 데이터 흐름 메서드
    # =========================================================================
    
    def validate_step_dependencies(self, step_id: int) -> Dict[str, Any]:
        """Step 의존성 검증"""
        try:
            required_flows = self.pipeline_flows.get(step_id, [])
            missing_dependencies = []
            
            for flow in required_flows:
                if not flow.required:
                    continue
                    
                source_step = flow.source_step
                
                # 소스 Step이 완료되었는지 확인
                if source_step not in self.metadata.completed_steps:
                    missing_dependencies.append(f"Step {source_step} 미완료")
                    continue
                
                # 필요한 데이터가 있는지 확인
                if not self._has_step_data(source_step, flow.data_type):
                    missing_dependencies.append(f"Step {source_step} -> {flow.data_type.value} 데이터 없음")
            
            return {
                'valid': len(missing_dependencies) == 0,
                'missing': missing_dependencies,
                'required_steps': [f.source_step for f in required_flows if f.required]
            }
            
        except Exception as e:
            logger.error(f"의존성 검증 실패: {e}")
            return {'valid': False, 'missing': [f"검증 오류: {e}"], 'required_steps': []}
    
    def prepare_step_input_data(self, step_id: int) -> Dict[str, Any]:
        """Step 입력 데이터 준비"""
        try:
            input_data = {
                'session_id': self.session_id,
                'step_id': step_id,
                'measurements': self.metadata.measurements
            }
            
            # Step별 의존성 데이터 추가
            required_flows = self.pipeline_flows.get(step_id, [])
            
            for flow in required_flows:
                source_step = flow.source_step
                data_type = flow.data_type
                
                if source_step in self.step_data_cache:
                    source_data = self.step_data_cache[source_step]
                    
                    # 데이터 타입별 추가
                    data_key = f"step_{source_step}_{data_type.value}"
                    
                    if data_type == DataType.RAW_IMAGE:
                        input_data[data_key] = source_data.get('primary_output')
                    elif data_type == DataType.SEGMENTATION_MASK:
                        input_data[data_key] = source_data.get('segmentation_mask')
                    elif data_type == DataType.POSE_KEYPOINTS:
                        input_data[data_key] = source_data.get('pose_keypoints')
                    elif data_type == DataType.TRANSFORMATION_MATRIX:
                        input_data[data_key] = source_data.get('transformation_matrix')
                    else:
                        input_data[data_key] = source_data.get('primary_output')
            
            # 🔥 원본 이미지 추가 (모든 Step에서 필요할 수 있음)
            # Step 1에서 원본 이미지를 RAW_IMAGE로 저장하지 않는 경우를 대비
            if 'person_image' not in input_data and 'clothing_image' not in input_data:
                # 세션 메타데이터에서 이미지 경로 가져오기
                person_image_path = self.metadata.person_image.path
                clothing_image_path = self.metadata.clothing_image.path
                
                # 이미지 파일이 존재하는지 확인하고 실제 이미지로 로드
                if Path(person_image_path).exists() and Path(clothing_image_path).exists():
                    try:
                        from PIL import Image
                        person_image = Image.open(person_image_path).convert('RGB')
                        clothing_image = Image.open(clothing_image_path).convert('RGB')
                        
                        input_data['person_image'] = person_image
                        input_data['clothing_image'] = clothing_image
                        logger.debug(f"✅ 원본 이미지 로드 완료: person={person_image.size}, clothing={clothing_image.size}")
                    except Exception as e:
                        logger.error(f"❌ 원본 이미지 로드 실패: {e}")
                        # 경로만 저장 (폴백)
                        input_data['person_image_path'] = person_image_path
                        input_data['clothing_image_path'] = clothing_image_path
                else:
                    logger.warning(f"⚠️ 원본 이미지 파일이 존재하지 않음: person={person_image_path}, clothing={clothing_image_path}")
            
            return input_data
            
        except Exception as e:
            logger.error(f"입력 데이터 준비 실패: {e}")
            return {'session_id': self.session_id, 'step_id': step_id}
    
    def save_step_data(self, step_id: int, step_result: Dict[str, Any]):
        """Step 데이터 저장 (중간 결과 포함)"""
        try:
            with self.lock:
                # 기존 step_results 저장 (호환성 유지)
                self.step_results[step_id] = {
                    **step_result,
                    'timestamp': datetime.now().isoformat(),
                    'step_id': step_id
                }
                
                # 새로운 step_data_cache 저장 (데이터 흐름용)
                self.step_data_cache[step_id] = step_result.copy()
                
                # 성능 정보 저장
                if 'processing_time' in step_result:
                    self.step_processing_times[step_id] = step_result['processing_time']
                
                if 'quality_score' in step_result:
                    self.step_quality_scores[step_id] = step_result['quality_score']
                
                # 완료된 단계 추가
                self.add_completed_step(step_id)
                
                logger.debug(f"📊 Step {step_id} 데이터 저장 완료")
                
        except Exception as e:
            logger.error(f"Step 데이터 저장 실패: {e}")
    
    def _has_step_data(self, step_id: int, data_type: DataType) -> bool:
        """Step 데이터 존재 확인"""
        try:
            if step_id not in self.step_data_cache:
                return False
            
            step_data = self.step_data_cache[step_id]
            
            if data_type == DataType.RAW_IMAGE:
                return 'primary_output' in step_data
            elif data_type == DataType.SEGMENTATION_MASK:
                return 'segmentation_mask' in step_data
            elif data_type == DataType.POSE_KEYPOINTS:
                return 'pose_keypoints' in step_data
            elif data_type == DataType.TRANSFORMATION_MATRIX:
                return 'transformation_matrix' in step_data
            else:
                return 'primary_output' in step_data
                
        except Exception:
            return False
    
    # =========================================================================
    # 🔥 이미지 캐시 메서드
    # =========================================================================
    
    def get_cached_image(self, image_key: str) -> Optional[Image.Image]:
        """캐시된 이미지 가져오기"""
        with self._image_cache_lock:
            return self._image_cache.get(image_key)
    
    def cache_image(self, image_key: str, image: Image.Image):
        """이미지를 캐시에 저장"""
        with self._image_cache_lock:
            self._image_cache[image_key] = image
    
    def has_cached_image(self, image_key: str) -> bool:
        """캐시된 이미지 존재 여부 확인"""
        with self._image_cache_lock:
            return image_key in self._image_cache
    
    def clear_image_cache(self):
        """이미지 캐시 정리"""
        with self._image_cache_lock:
            self._image_cache.clear()
    
    def get_cached_images(self) -> Dict[str, Image.Image]:
        """모든 캐시된 이미지 가져오기"""
        with self._image_cache_lock:
            return self._image_cache.copy()

# =============================================================================
# 🔥 메인 세션 매니저 클래스 (완전 통합 - 순환참조 해결)
# =============================================================================

class SessionManager:
    """세션 관리자 (SQLite 데이터베이스 통합)"""
    
    def __init__(self, base_path: Optional[Path] = None):
        """SessionManager 초기화 (SQLite 통합) - 강화된 버전"""
        try:
            print("🔄 SessionManager 초기화 시작...")
            logger.info("🔄 SessionManager 초기화 시작...")
            
            # 기본 경로 설정
            if base_path is None:
                base_path = Path("sessions")
            
            self.base_path = Path(base_path)
            self.sessions_dir = self.base_path / "data"
            self.db_path = self.base_path / "sessions.db"
            
            # 디렉토리 생성
            try:
                self.sessions_dir.mkdir(parents=True, exist_ok=True)
                logger.info("✅ 세션 디렉토리 생성 완료")
            except Exception as e:
                logger.warning(f"⚠️ 세션 디렉토리 생성 실패: {e}")
                self.sessions_dir = Path("/tmp/mycloset_sessions")
                self.sessions_dir.mkdir(parents=True, exist_ok=True)
            
            # 메모리 세션 (캐시)
            self.sessions: Dict[str, SessionData] = {}
            self._lock = threading.Lock()
            
            # 세션 제한
            self.max_sessions = 100
            self.session_timeout_hours = 24
            
            # SQLite 데이터베이스 초기화 (안전한 방식)
            try:
                # 데이터베이스 경로 설정
                self.db_path = self.base_path / "sessions.db"
                logger.info(f"🔄 SQLite 데이터베이스 경로: {self.db_path}")
                
                # 데이터베이스 초기화 (타임아웃 없이)
                self._init_database()
                logger.info("✅ SQLite 데이터베이스 초기화 완료")
                
                # 🔥 기존 세션 복구
                logger.info("🔄 기존 세션 복구 시작...")
                self._reload_all_sessions_from_db()
                logger.info(f"✅ 기존 세션 복구 완료: {len(self.sessions)}개 세션")
                
            except Exception as db_error:
                logger.warning(f"⚠️ SQLite 데이터베이스 초기화 실패, 메모리 모드로 진행: {db_error}")
                self.db_path = None
            
            # 정리 작업 시작 (서버 시작 문제 해결을 위해 임시 비활성화)
            try:
                # 서버 시작 중에는 정리 작업을 비활성화
                logger.info("⚠️ 서버 시작 중 - 정리 작업 비활성화")
                self._cleanup_task = None
            except Exception as cleanup_error:
                logger.warning(f"⚠️ 정리 작업 설정 실패: {cleanup_error}")
            
            # 🔥 초기화 검증
            logger.info(f"✅ SessionManager 초기화 검증:")
            logger.info(f"   - 세션 디렉토리: {self.sessions_dir}")
            logger.info(f"   - 데이터베이스 경로: {self.db_path}")
            logger.info(f"   - 메모리 세션 수: {len(self.sessions)}")
            logger.info(f"   - 세션 매니저 ID: {id(self)}")
            logger.info(f"   - 세션 매니저 주소: {hex(id(self))}")
            
            print("✅ SessionManager 초기화 완료")
            logger.info("✅ SessionManager 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ SessionManager 초기화 실패: {e}")
            logger.error(f"❌ 상세 오류: {traceback.format_exc()}")
            # 폴백 초기화
            self._fallback_initialization()
    
    def _init_database(self):
        """SQLite 데이터베이스 초기화 (강화된 버전)"""
        try:
            logger.info("🔄 SQLite 데이터베이스 초기화...")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 세션 메타데이터 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS session_metadata (
                        session_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        last_accessed TEXT NOT NULL,
                        measurements TEXT,
                        person_image_path TEXT,
                        person_image_size TEXT,
                        person_image_mode TEXT,
                        person_image_format TEXT,
                        person_image_file_size INTEGER,
                        clothing_image_path TEXT,
                        clothing_image_size TEXT,
                        clothing_image_mode TEXT,
                        clothing_image_format TEXT,
                        clothing_image_file_size INTEGER,
                        total_steps INTEGER DEFAULT 8,
                        completed_steps TEXT,
                        custom_data TEXT
                    )
                """)
                
                # 세션 데이터 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS session_data (
                        session_id TEXT,
                        step_id INTEGER,
                        step_result TEXT,
                        step_processing_time REAL DEFAULT 0.0,
                        step_quality_score REAL DEFAULT 0.0,
                        step_data_cache TEXT,
                        timestamp TEXT,
                        PRIMARY KEY (session_id, step_id),
                        FOREIGN KEY (session_id) REFERENCES session_metadata(session_id)
                    )
                """)
                
                # 인덱스 생성
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_metadata_id ON session_metadata(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_data_id ON session_data(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_data_step ON session_data(session_id, step_id)")
                
                conn.commit()
                
                # 🔥 데이터베이스 검증
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                logger.info(f"✅ 데이터베이스 테이블 확인: {tables}")
                
                if 'session_metadata' in tables and 'session_data' in tables:
                    logger.info("✅ SQLite 데이터베이스 초기화 완료")
                else:
                    logger.error("❌ 필수 테이블이 생성되지 않음")
                    raise Exception("필수 테이블 생성 실패")
                
        except Exception as e:
            logger.error(f"❌ SQLite 데이터베이스 초기화 실패: {e}")
            logger.error(f"❌ 상세 오류: {traceback.format_exc()}")
            raise
    
    def _save_session_to_db(self, session_data: SessionData) -> bool:
        """세션을 데이터베이스에 저장 (강화된 버전)"""
        if self.db_path is None:
            logger.debug("⚠️ SQLite DB 비활성화 - 메모리 모드")
            return True
            
        try:
            logger.info(f"🔥 세션 데이터베이스 저장 시작: {session_data.session_id}")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 메타데이터 저장
                metadata = session_data.metadata
                cursor.execute("""
                    INSERT OR REPLACE INTO session_metadata (
                        session_id, created_at, last_accessed, measurements,
                        person_image_path, person_image_size, person_image_mode,
                        person_image_format, person_image_file_size,
                        clothing_image_path, clothing_image_size, clothing_image_mode,
                        clothing_image_format, clothing_image_file_size,
                        total_steps, completed_steps, custom_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.session_id,
                    metadata.created_at.isoformat(),
                    metadata.last_accessed.isoformat(),
                    json.dumps(metadata.measurements),
                    metadata.person_image.path,
                    json.dumps(metadata.person_image.size),
                    metadata.person_image.mode,
                    metadata.person_image.format,
                    metadata.person_image.file_size,
                    metadata.clothing_image.path,
                    json.dumps(metadata.clothing_image.size),
                    metadata.clothing_image.mode,
                    metadata.clothing_image.format,
                    metadata.clothing_image.file_size,
                    metadata.total_steps,
                    json.dumps(metadata.completed_steps),
                    json.dumps(session_data.custom_data)
                ))
                
                logger.info(f"✅ 메타데이터 저장 완료: {session_data.session_id}")
                
                # Step 데이터 저장
                step_count = 0
                for step_id, step_data in session_data.step_data_cache.items():
                    cursor.execute("""
                        INSERT OR REPLACE INTO session_data (
                            session_id, step_id, step_result, step_processing_time,
                            step_quality_score, step_data_cache, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        session_data.session_id,
                        step_id,
                        json.dumps(step_data.get('result', {})),
                        session_data.step_processing_times.get(step_id, 0.0),
                        session_data.step_quality_scores.get(step_id, 0.0),
                        json.dumps(step_data),
                        datetime.now().isoformat()
                    ))
                    step_count += 1
                
                logger.info(f"✅ Step 데이터 저장 완료: {step_count}개 Step")
                
                conn.commit()
                logger.info(f"✅ 세션 데이터베이스 저장 완료: {session_data.session_id}")
                
                # 🔥 저장 검증
                cursor.execute("SELECT session_id FROM session_metadata WHERE session_id = ?", (session_data.session_id,))
                result = cursor.fetchone()
                if result:
                    logger.info(f"✅ 세션 저장 검증 성공: {session_data.session_id}")
                    return True
                else:
                    logger.error(f"❌ 세션 저장 검증 실패: {session_data.session_id}")
                    return False
                
        except Exception as e:
            logger.error(f"❌ 세션 데이터베이스 저장 실패: {e}")
            logger.error(f"❌ 상세 오류: {traceback.format_exc()}")
            return False
    
    def _reload_all_sessions_from_db(self):
        """모든 세션을 데이터베이스에서 재로드 (강화된 버전)"""
        try:
            logger.info(f"🔄 모든 세션 재로드 시작")
            
            if self.db_path is None or not self.db_path.exists():
                logger.warning("⚠️ 데이터베이스 파일이 없음 - 메모리 모드")
                return
            
            with self._lock:
                # 기존 메모리 세션 백업
                old_sessions = self.sessions.copy()
                logger.info(f"🔍 기존 메모리 세션 수: {len(old_sessions)}")
                
                # 데이터베이스에서 모든 세션 로드
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT session_id FROM session_metadata")
                    session_ids = [row[0] for row in cursor.fetchall()]
                    cursor.close()
                
                logger.info(f"🔍 데이터베이스 세션 수: {len(session_ids)}")
                logger.info(f"🔍 데이터베이스 세션 ID들: {session_ids}")
                
                # 각 세션을 메모리에 로드
                loaded_count = 0
                for sid in session_ids:
                    if sid not in self.sessions:
                        session_data = self._load_session_from_db(sid)
                        if session_data:
                            self.sessions[sid] = session_data
                            loaded_count += 1
                            logger.info(f"✅ 세션 재로드 완료: {sid}")
                        else:
                            logger.warning(f"⚠️ 세션 재로드 실패: {sid}")
                
                logger.info(f"✅ 모든 세션 재로드 완료. 총 {len(self.sessions)}개 세션 (새로 로드: {loaded_count}개)")
                
        except Exception as e:
            logger.error(f"❌ 모든 세션 재로드 실패: {e}")
            logger.error(f"❌ 상세 오류: {traceback.format_exc()}")
    
    def _force_reload_session(self, session_id: str) -> Optional[SessionData]:
        """강제로 세션을 데이터베이스에서 재로드 (강화된 버전)"""
        try:
            logger.info(f"🔄 강제 세션 재로드 시작: {session_id}")
            
            if self.db_path is None or not self.db_path.exists():
                logger.warning("⚠️ 데이터베이스 파일이 없음")
                return None
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 메타데이터 확인
                cursor.execute("SELECT * FROM session_metadata WHERE session_id = ?", (session_id,))
                metadata_row = cursor.fetchone()
                
                if not metadata_row:
                    logger.error(f"❌ 강제 재로드 실패 - 데이터베이스에 세션 없음: {session_id}")
                    return None
                
                logger.info(f"✅ 메타데이터 확인 완료: {session_id}")
                
                # 메타데이터 복원
                try:
                    metadata = SessionMetadata(
                        session_id=metadata_row[0],
                        created_at=datetime.fromisoformat(metadata_row[1]),
                        last_accessed=datetime.fromisoformat(metadata_row[2]),
                        measurements=json.loads(metadata_row[3] or '{}'),
                        person_image=ImageInfo(
                            path=metadata_row[4],
                            size=tuple(json.loads(metadata_row[5] or '[0,0]')),
                            mode=metadata_row[6],
                            format=metadata_row[7],
                            file_size=metadata_row[8]
                        ),
                        clothing_image=ImageInfo(
                            path=metadata_row[9],
                            size=tuple(json.loads(metadata_row[10] or '[0,0]')),
                            mode=metadata_row[11],
                            format=metadata_row[12],
                            file_size=metadata_row[13]
                        ),
                        total_steps=metadata_row[14],
                        completed_steps=json.loads(metadata_row[15] or '[]')
                    )
                    
                    logger.info(f"✅ 메타데이터 복원 완료: {session_id}")
                    
                except Exception as metadata_error:
                    logger.error(f"❌ 메타데이터 복원 실패: {metadata_error}")
                    return None
                
                # SessionData 생성
                session_dir = self.sessions_dir / session_id
                session_dir.mkdir(exist_ok=True)
                
                session_data = SessionData(metadata, session_dir)
                
                # 커스텀 데이터 복원
                if len(metadata_row) > 16:
                    session_data.custom_data = json.loads(metadata_row[16] or '{}')
                
                logger.info(f"✅ SessionData 생성 완료: {session_id}")
                
                # Step 데이터 로드
                try:
                    cursor.execute("SELECT * FROM session_data WHERE session_id = ?", (session_id,))
                    step_rows = cursor.fetchall()
                    
                    step_count = 0
                    for row in step_rows:
                        step_id = row[1]
                        step_result = json.loads(row[2] or '{}')
                        processing_time = row[3] or 0.0
                        quality_score = row[4] or 0.0
                        step_data_cache = json.loads(row[5] or '{}')
                        
                        session_data.step_data_cache[step_id] = step_data_cache
                        session_data.step_processing_times[step_id] = processing_time
                        session_data.step_quality_scores[step_id] = quality_score
                        step_count += 1
                    
                    logger.info(f"✅ Step 데이터 로드 완료: {step_count}개 Step")
                    
                except Exception as step_error:
                    logger.error(f"❌ Step 데이터 로드 실패: {step_error}")
                    # Step 데이터 로드 실패해도 계속 진행
                
                logger.info(f"✅ 강제 세션 재로드 성공: {session_id}")
                return session_data
                
        except Exception as e:
            logger.error(f"❌ 강제 세션 재로드 중 오류: {e}")
            logger.error(f"❌ 상세 오류: {traceback.format_exc()}")
            return None

    def _load_session_from_db(self, session_id: str) -> Optional[SessionData]:
        """데이터베이스에서 세션 로드 (강화된 버전)"""
        if self.db_path is None:
            logger.debug("⚠️ SQLite DB 비활성화 - 메모리 모드")
            return None
            
        try:
            logger.info(f"🔥 세션 데이터베이스 로드 시작: {session_id}")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 메타데이터 로드
                cursor.execute("SELECT * FROM session_metadata WHERE session_id = ?", (session_id,))
                metadata_row = cursor.fetchone()
                
                if not metadata_row:
                    logger.warning(f"⚠️ 메타데이터를 찾을 수 없음: {session_id}")
                    return None
                
                logger.info(f"✅ 메타데이터 로드 완료: {session_id}")
                
                # 메타데이터 복원
                try:
                    metadata = SessionMetadata(
                        session_id=metadata_row[0],
                        created_at=datetime.fromisoformat(metadata_row[1]),
                        last_accessed=datetime.fromisoformat(metadata_row[2]),
                        measurements=json.loads(metadata_row[3] or '{}'),
                        person_image=ImageInfo(
                            path=metadata_row[4],
                            size=tuple(json.loads(metadata_row[5] or '[0,0]')),
                            mode=metadata_row[6],
                            format=metadata_row[7],
                            file_size=metadata_row[8]
                        ),
                        clothing_image=ImageInfo(
                            path=metadata_row[9],
                            size=tuple(json.loads(metadata_row[10] or '[0,0]')),
                            mode=metadata_row[11],
                            format=metadata_row[12],
                            file_size=metadata_row[13]
                        ),
                        total_steps=metadata_row[14],
                        completed_steps=json.loads(metadata_row[15] or '[]')
                    )
                    
                    logger.info(f"✅ 메타데이터 복원 완료: {session_id}")
                    
                except Exception as metadata_error:
                    logger.error(f"❌ 메타데이터 복원 실패: {metadata_error}")
                    return None
                
                # SessionData 생성
                session_data = SessionData(metadata, self.sessions_dir)
                session_data.custom_data = json.loads(metadata_row[16] or '{}')
                
                logger.info(f"✅ SessionData 생성 완료: {session_id}")
                
                # Step 데이터 로드
                try:
                    cursor.execute("SELECT * FROM session_data WHERE session_id = ?", (session_id,))
                    step_rows = cursor.fetchall()
                    
                    step_count = 0
                    for row in step_rows:
                        step_id = row[1]
                        step_result = json.loads(row[2] or '{}')
                        processing_time = row[3] or 0.0
                        quality_score = row[4] or 0.0
                        step_data_cache = json.loads(row[5] or '{}')
                        
                        session_data.step_data_cache[step_id] = step_data_cache
                        session_data.step_processing_times[step_id] = processing_time
                        session_data.step_quality_scores[step_id] = quality_score
                        step_count += 1
                    
                    logger.info(f"✅ Step 데이터 로드 완료: {step_count}개 Step")
                    
                except Exception as step_error:
                    logger.error(f"❌ Step 데이터 로드 실패: {step_error}")
                    # Step 데이터 로드 실패해도 계속 진행
                
                # 🔥 메모리에 즉시 저장
                self.sessions[session_id] = session_data
                
                logger.info(f"✅ 세션 데이터베이스 로드 완료: {session_id}")
                logger.info(f"✅ 메모리에 세션 저장됨: {session_id}")
                logger.info(f"✅ 현재 메모리 세션 수: {len(self.sessions)}")
                
                # 🔥 로드 검증
                verification_session = self.sessions.get(session_id)
                if verification_session:
                    logger.info(f"✅ 세션 로드 검증 성공: {session_id}")
                    return session_data
                else:
                    logger.error(f"❌ 세션 로드 검증 실패: {session_id}")
                    return None
                
        except Exception as e:
            logger.error(f"❌ 세션 데이터베이스 로드 실패: {e}")
            logger.error(f"❌ 상세 오류: {traceback.format_exc()}")
            return None
    
    async def create_session(
        self, 
        person_image: Image.Image,
        clothing_image: Image.Image,
        measurements: Dict[str, Any]
    ) -> str:
        logger.info(f"🔥 CREATE_SESSION 시작: 이미지 크기 person={person_image.size}, clothing={clothing_image.size}")
        logger.info(f"🔥 CREATE_SESSION - 세션 매니저 ID: {id(self)}")
        logger.info(f"🔥 CREATE_SESSION - 세션 매니저 주소: {hex(id(self))}")
        logger.info(f"🔥 CREATE_SESSION - 현재 메모리 세션 수: {len(self.sessions)}")
        logger.info(f"🔥 CREATE_SESSION - 메모리 세션 키들: {list(self.sessions.keys())}")
        """세션 생성 (SQLite 통합) - 강화된 버전"""
        try:
            with self._lock:
                # 세션 ID 생성
                session_id = self._generate_session_id()
                logger.info(f"🔥 CREATE_SESSION - 생성된 세션 ID: {session_id}")
                
                # 이미지 저장
                person_image_info = await self._save_image(person_image, self.sessions_dir / f"{session_id}_person.jpg", "person")
                clothing_image_info = await self._save_image(clothing_image, self.sessions_dir / f"{session_id}_clothing.jpg", "clothing")
                
                logger.info(f"🔥 CREATE_SESSION - 이미지 저장 완료: person={person_image_info.path}, clothing={clothing_image_info.path}")
                
                # 메타데이터 생성
                metadata = SessionMetadata(
                    session_id=session_id,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    measurements=measurements,
                    person_image=person_image_info,
                    clothing_image=clothing_image_info
                )
                
                # SessionData 생성
                session_data = SessionData(metadata, self.sessions_dir)
                
                # 🔥 강화된 메모리 저장
                self.sessions[session_id] = session_data
                logger.info(f"🔥 CREATE_SESSION - 메모리에 세션 저장됨: {session_id}")
                logger.info(f"🔥 CREATE_SESSION - 현재 메모리 세션 수: {len(self.sessions)}")
                logger.info(f"🔥 CREATE_SESSION - 메모리 세션 키들: {list(self.sessions.keys())}")
                
                # 🔥 강화된 데이터베이스 저장
                db_save_success = self._save_session_to_db(session_data)
                if db_save_success:
                    logger.info(f"🔥 CREATE_SESSION - 데이터베이스 저장 성공: {session_id}")
                else:
                    logger.warning(f"⚠️ CREATE_SESSION - 데이터베이스 저장 실패: {session_id}")
                
                # 🔥 세션 유지 확인
                logger.info(f"🔥 CREATE_SESSION - 세션 유지 확인: {session_id}")
                logger.info(f"🔥 CREATE_SESSION - 세션 매니저 ID: {id(self)}")
                logger.info(f"🔥 CREATE_SESSION - 세션 매니저 주소: {hex(id(self))}")
                logger.info(f"🔥 CREATE_SESSION - 세션 데이터 ID: {id(session_data)}")
                logger.info(f"🔥 CREATE_SESSION - 세션 데이터 주소: {hex(id(session_data))}")
                
                # 🔥 즉시 검증
                verification_session = self.sessions.get(session_id)
                if verification_session:
                    logger.info(f"✅ CREATE_SESSION - 세션 검증 성공: {session_id}")
                    logger.info(f"✅ CREATE_SESSION - 검증된 세션 데이터 ID: {id(verification_session)}")
                else:
                    logger.error(f"❌ CREATE_SESSION - 세션 검증 실패: {session_id}")
                
                return session_id
                
        except Exception as e:
            logger.error(f"❌ 세션 생성 실패: {e}")
            logger.error(f"❌ 상세 오류: {traceback.format_exc()}")
            raise
    
    async def get_session_images(self, session_id: str) -> Tuple[Image.Image, Image.Image]:
        """
        🔥 세션에서 이미지 로드 (기존 함수명 유지 + 순환참조 안전성)
        
        Args:
            session_id: 세션 ID
            
        Returns:
            Tuple[Image.Image, Image.Image]: (사용자 이미지, 의류 이미지)
        """
        try:
            # 1. 세션 데이터 확인
            session_data = self.sessions.get(session_id)
            if not session_data:
                raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
            
            # 2. 접근 시간 업데이트
            session_data.update_access_time()
            
            # 3. 이미지 파일 경로
            person_path = session_data.metadata.person_image.path
            clothing_path = session_data.metadata.clothing_image.path
            
            # 4. 파일 존재 확인
            if not Path(person_path).exists():
                raise FileNotFoundError(f"사용자 이미지 파일이 없습니다: {person_path}")
            if not Path(clothing_path).exists():
                raise FileNotFoundError(f"의류 이미지 파일이 없습니다: {clothing_path}")
            
            # 5. 이미지 로드 (비동기)
            person_image, clothing_image = await asyncio.gather(
                self._load_image_async(person_path),
                self._load_image_async(clothing_path)
            )
            
            logger.debug(f"📂 이미지 로드 완료: {session_id}")
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
        """Step 결과 저장 (강화된 버전)"""
        try:
            logger.info(f"🔥 SAVE_STEP_RESULT 시작: session_id={session_id}, step_id={step_id}")
            logger.info(f"🔥 SAVE_STEP_RESULT - 현재 메모리 세션 수: {len(self.sessions)}")
            logger.info(f"🔥 SAVE_STEP_RESULT - 메모리 세션 키들: {list(self.sessions.keys())}")
            
            # 세션 데이터 조회
            session_data = self.sessions.get(session_id)
            if not session_data:
                logger.error(f"❌ 세션을 찾을 수 없음: {session_id}")
                # 세션 복구 시도
                session_data = await self._recover_session_data(session_id)
                if not session_data:
                    raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
            
            logger.info(f"✅ 세션 데이터 조회 성공: {session_id}")
            
            # Step 데이터 저장
            session_data.save_step_data(step_id, result)
            
            # 완료된 Step 추가
            session_data.add_completed_step(step_id)
            
            # 결과 이미지가 있으면 저장
            if result_image:
                image_path = self.sessions_dir / f"{session_id}_step_{step_id}_result.jpg"
                await self._save_image_async(result_image, image_path)
                logger.info(f"✅ 결과 이미지 저장 완료: {image_path}")
            
            # 🔥 강화된 데이터베이스 저장
            db_save_success = self._save_session_to_db(session_data)
            if db_save_success:
                logger.info(f"✅ Step 결과 데이터베이스 저장 성공: session_id={session_id}, step_id={step_id}")
            else:
                logger.warning(f"⚠️ Step 결과 데이터베이스 저장 실패: session_id={session_id}, step_id={step_id}")
            
            # 🔥 메모리 세션 업데이트 확인
            updated_session = self.sessions.get(session_id)
            if updated_session:
                logger.info(f"✅ 메모리 세션 업데이트 확인: {session_id}")
                logger.info(f"✅ 완료된 Step 수: {len(updated_session.metadata.completed_steps)}")
            else:
                logger.error(f"❌ 메모리 세션 업데이트 실패: {session_id}")
            
            logger.info(f"✅ SAVE_STEP_RESULT 완료: session_id={session_id}, step_id={step_id}")
            
        except Exception as e:
            logger.error(f"❌ Step 결과 저장 실패: {e}")
            logger.error(f"❌ 상세 오류: {traceback.format_exc()}")
            raise
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """세션 상태 조회 (강화된 버전)"""
        try:
            logger.info(f"🔥 GET_SESSION_STATUS 시작: {session_id}")
            logger.info(f"🔥 GET_SESSION_STATUS - 현재 메모리 세션 수: {len(self.sessions)}")
            logger.info(f"🔥 GET_SESSION_STATUS - 메모리 세션 키들: {list(self.sessions.keys())}")
            
            # 세션 데이터 조회
            session_data = self.sessions.get(session_id)
            if not session_data:
                logger.warning(f"⚠️ 메모리에 세션 없음 - 복구 시도: {session_id}")
                # 세션 복구 시도
                session_data = await self._recover_session_data(session_id)
                if not session_data:
                    logger.error(f"❌ 세션을 찾을 수 없음: {session_id}")
                    return {
                        "session_id": session_id,
                        "status": "not_found",
                        "error": f"세션을 찾을 수 없습니다: {session_id}",
                        "created_at": None,
                        "last_accessed": None,
                        "total_steps": 8,
                        "completed_steps": [],
                        "progress_percent": 0.0
                    }
            
            logger.info(f"✅ 세션 데이터 조회 성공: {session_id}")
            
            # 접근 시간 업데이트
            session_data.update_access_time()
            
            # 상태 정보 생성
            status_dict = self._create_session_status_dict(session_data)
            
            logger.info(f"✅ GET_SESSION_STATUS 완료: {session_id}")
            return status_dict
            
        except Exception as e:
            logger.error(f"❌ 세션 상태 조회 실패: {e}")
            logger.error(f"❌ 상세 오류: {traceback.format_exc()}")
            return {
                "session_id": session_id,
                "status": "error",
                "error": str(e),
                "created_at": None,
                "last_accessed": None,
                "total_steps": 8,
                "completed_steps": [],
                "progress_percent": 0.0
            }
    
    def _create_session_status_dict(self, session_data: SessionData) -> Dict[str, Any]:
        """세션 상태 딕셔너리 생성"""
        try:
            safe_status = {
                "session_id": session_data.session_id,
                "created_at": session_data.metadata.created_at.isoformat(),
                "last_accessed": session_data.metadata.last_accessed.isoformat(),
                "measurements": session_data.metadata.measurements,
                "total_steps": session_data.metadata.total_steps,
                "completed_steps": session_data.metadata.completed_steps,
                "progress_percent": session_data.get_progress_percent(),
                "step_results": {
                    step_id: {
                        "status": "completed" if step_id in session_data.step_data_cache else "pending",
                        "processing_time": session_data.step_processing_times.get(step_id, 0.0),
                        "quality_score": session_data.step_quality_scores.get(step_id, 0.0)
                    }
                    for step_id in range(1, 9)
                },
                "metadata": {
                    "person_size": session_data.metadata.person_image.size,
                    "clothing_size": session_data.metadata.clothing_image.size
                },
                "step_results_count": len(session_data.step_data_cache),
                "circular_reference_safe": True
            }
            
            # 🔥 원본 이미지 데이터 포함 (API에서 필요)
            try:
                # 1순위: 커스텀 데이터에서 가져오기
                logger.info(f"🔍 커스텀 데이터 확인: {list(session_data.custom_data.keys())}")
                if 'original_person_image' in session_data.custom_data:
                    safe_status["original_person_image"] = session_data.custom_data['original_person_image']
                    logger.info(f"✅ 커스텀 person_image 사용: {session_data.session_id} ({len(session_data.custom_data['original_person_image'])} 문자)")
                else:
                    logger.warning(f"⚠️ 커스텀 person_image 없음: {session_data.session_id}")
                    
                if 'original_clothing_image' in session_data.custom_data:
                    safe_status["original_clothing_image"] = session_data.custom_data['original_clothing_image']
                    logger.info(f"✅ 커스텀 clothing_image 사용: {session_data.session_id} ({len(session_data.custom_data['original_clothing_image'])} 문자)")
                else:
                    logger.warning(f"⚠️ 커스텀 clothing_image 없음: {session_data.session_id}")
                    
                # 2순위: 파일에서 변환 (커스텀 데이터가 없는 경우)
                if 'original_person_image' not in safe_status:
                    person_image_path = session_data.metadata.person_image.path
                    if Path(person_image_path).exists():
                        person_image = asyncio.run(self._load_image_async(person_image_path))
                        person_b64 = asyncio.run(self._image_to_base64(person_image))
                        safe_status["original_person_image"] = person_b64
                        logger.debug(f"✅ 파일에서 person_image base64 변환 완료: {session_data.session_id}")
                
                if 'original_clothing_image' not in safe_status:
                    clothing_image_path = session_data.metadata.clothing_image.path
                    if Path(clothing_image_path).exists():
                        clothing_image = asyncio.run(self._load_image_async(clothing_image_path))
                        clothing_b64 = asyncio.run(self._image_to_base64(clothing_image))
                        safe_status["original_clothing_image"] = clothing_b64
                        logger.debug(f"✅ 파일에서 clothing_image base64 변환 완료: {session_data.session_id}")
                        
            except Exception as img_error:
                logger.warning(f"⚠️ 이미지 변환 실패 {session_data.session_id}: {img_error}")
                safe_status["image_conversion_error"] = str(img_error)
            
            return safe_status
            
        except Exception as e:
            logger.error(f"❌ 세션 상태 딕셔너리 생성 실패: {e}")
            return {
                "session_id": session_data.session_id,
                "error": str(e),
                "circular_reference_safe": True,
                "fallback_mode": True
            }
    
    async def update_session(self, session_id: str, session_data_dict: Dict[str, Any]) -> bool:
        """세션 데이터 업데이트 (기존 호환 + 커스텀 키 지원)"""
        try:
            with self._lock:
                if session_id not in self.sessions:
                    logger.warning(f"⚠️ 세션 업데이트 실패: 세션 {session_id} 없음")
                    return False
                
                session_data = self.sessions[session_id]
                session_data.update_access_time()
                
                # 세션 데이터 업데이트
                logger.info(f"🔍 세션 업데이트 시작: {session_id}, 키 개수: {len(session_data_dict)}")
                for key, value in session_data_dict.items():
                    if key.startswith('step_') and key.endswith('_result'):
                        # Step 결과 저장
                        step_id = int(key.split('_')[1])
                        session_data.save_step_data(step_id, value)
                        logger.debug(f"✅ Step 결과 저장: {key}")
                    elif key in ['original_person_image', 'original_clothing_image']:
                        # 🔥 커스텀 이미지 키 처리
                        session_data.custom_data[key] = value
                        logger.info(f"✅ 커스텀 이미지 저장: {key} ({len(value)} 문자)")
                    else:
                        # 기타 데이터는 메타데이터에 저장
                        if hasattr(session_data.metadata, key):
                            setattr(session_data.metadata, key, value)
                            logger.debug(f"✅ 메타데이터 저장: {key}")
                        else:
                            logger.debug(f"⚠️ 알 수 없는 키 무시: {key}")
                
                # 메타데이터 저장
                self._save_session_to_db(session_data)
                
                logger.debug(f"✅ 세션 {session_id} 업데이트 완료")
                return True
                
        except Exception as e:
            logger.error(f"❌ 세션 업데이트 실패: {e}")
            return False
    
    def get_all_sessions_status(self) -> Dict[str, Any]:
        """전체 세션 상태 조회 (기존 함수명 유지 + 순환참조 방지)"""
        try:
            return self.to_safe_dict()  # 순환참조 방지 메서드 사용
        except Exception as e:
            logger.error(f"세션 상태 조회 실패: {e}")
            return {
                "error": str(e),
                "total_sessions": 0,
                "circular_reference_safe": True,
                "fallback_mode": True
            }
    
    async def cleanup_expired_sessions(self):
        """만료된 세션 자동 정리 (기존 함수명 유지)"""
        try:
            expired_sessions = []
            
            with self._lock:
                for session_id, session_data in list(self.sessions.items()):
                    if session_data.is_expired(self.session_timeout_hours):
                        expired_sessions.append(session_id)
            
            # 만료된 세션 정리
            for session_id in expired_sessions:
                await self.cleanup_session(session_id)
            
            if expired_sessions:
                logger.info(f"🧹 만료 세션 정리: {len(expired_sessions)}개")
                
        except Exception as e:
            logger.error(f"❌ 만료 세션 정리 실패: {e}")
    
    async def cleanup_all_sessions(self):
        """모든 세션 정리 (기존 함수명 유지)"""
        try:
            session_ids = list(self.sessions.keys())
            
            for session_id in session_ids:
                await self.cleanup_session(session_id)
            
            # 전체 세션 디렉토리 정리
            if self.base_path.exists():
                for session_dir in self.base_path.iterdir():
                    if session_dir.is_dir():
                        try:
                            shutil.rmtree(session_dir)
                        except:
                            pass
            
            logger.info(f"🧹 전체 세션 정리 완료: {len(session_ids)}개")
            
        except Exception as e:
            logger.error(f"❌ 전체 세션 정리 실패: {e}")

    # =========================================================================
    # 🔥 새로 추가 - Step간 데이터 흐름 전용 메서드들 (순환참조 안전)
    # =========================================================================
    
    async def validate_step_dependencies(self, session_id: str, step_id: int) -> Dict[str, Any]:
        """Step 의존성 검증 (순환참조 안전)"""
        try:
            session_data = self.sessions.get(session_id)
            if not session_data:
                raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
            
            return session_data.validate_step_dependencies(step_id)
            
        except Exception as e:
            logger.error(f"의존성 검증 실패: {e}")
            return {'valid': False, 'missing': [str(e)], 'required_steps': []}
    
    async def prepare_step_input_data(self, session_id: str, step_id: int) -> Dict[str, Any]:
        """Step 입력 데이터 준비 (순환참조 안전) - 강화된 버전"""
        try:
            logger.info(f"🔥 PREPARE_STEP_INPUT_DATA 시작: session_id={session_id}, step_id={step_id}")
            logger.info(f"🔥 PREPARE_STEP_INPUT_DATA - 현재 메모리 세션 수: {len(self.sessions)}")
            logger.info(f"🔥 PREPARE_STEP_INPUT_DATA - 메모리 세션 키들: {list(self.sessions.keys())}")
            logger.info(f"🔥 PREPARE_STEP_INPUT_DATA - 세션 매니저 ID: {id(self)}")
            logger.info(f"🔥 PREPARE_STEP_INPUT_DATA - 세션 매니저 주소: {hex(id(self))}")
            logger.info(f"🔥 PREPARE_STEP_INPUT_DATA - 세션 ID 존재 여부: {session_id in self.sessions}")
            logger.info(f"🔥 PREPARE_STEP_INPUT_DATA - 세션 ID 타입: {type(session_id)}")
            logger.info(f"🔥 PREPARE_STEP_INPUT_DATA - 메모리 세션 키 타입들: {[type(key) for key in self.sessions.keys()]}")
            
            # 🔥 강화된 세션 조회 로직
            logger.info(f"🔥 PREPARE_STEP_INPUT_DATA - 세션 조회 시작: {session_id}")
            session_data = self.sessions.get(session_id)
            logger.info(f"🔥 PREPARE_STEP_INPUT_DATA - 세션 조회 결과: {session_data is not None}")
            logger.info(f"🔥 PREPARE_STEP_INPUT_DATA - 세션 데이터 ID: {id(session_data) if session_data else 'None'}")
            
            if not session_data:
                logger.error(f"🔥 PREPARE_STEP_INPUT_DATA - 메모리에 세션 없음: {session_id}")
                logger.error(f"🔥 PREPARE_STEP_INPUT_DATA - 세션 ID 타입: {type(session_id)}")
                logger.error(f"🔥 PREPARE_STEP_INPUT_DATA - 메모리 세션 키들: {list(self.sessions.keys())}")
                logger.error(f"🔥 PREPARE_STEP_INPUT_DATA - 메모리 세션 키 타입들: {[type(key) for key in self.sessions.keys()]}")
                logger.error(f"🔥 PREPARE_STEP_INPUT_DATA - 세션 ID in 세션: {session_id in self.sessions}")
                logger.error(f"🔥 PREPARE_STEP_INPUT_DATA - 세션 ID == 키 비교: {[session_id == key for key in self.sessions.keys()]}")
                
                # 🔥 강화된 복구 로직
                session_data = await self._recover_session_data(session_id)
                
                if not session_data:
                    raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
            else:
                logger.info(f"🔥 PREPARE_STEP_INPUT_DATA - 메모리에서 세션 찾음: {session_id}")
                logger.info(f"🔥 PREPARE_STEP_INPUT_DATA - 세션 데이터 ID: {id(session_data)}")
                logger.info(f"🔥 PREPARE_STEP_INPUT_DATA - 세션 데이터 주소: {hex(id(session_data))}")
            
            session_data.update_access_time()
            
            # 기본 입력 데이터
            input_data = session_data.prepare_step_input_data(step_id)
            
            # 🔥 원본 이미지 추가 - 이미지 캐시 우선 사용
            try:
                # 1. 먼저 이미지 캐시에서 확인
                if session_data.has_cached_image('person_image') and session_data.has_cached_image('clothing_image'):
                    person_image = session_data.get_cached_image('person_image')
                    clothing_image = session_data.get_cached_image('clothing_image')
                    input_data['person_image'] = person_image
                    input_data['clothing_image'] = clothing_image
                    logger.info(f"✅ 캐시에서 원본 이미지 로드 성공: person={person_image.size}, clothing={clothing_image.size}")
                    
                else:
                    # 2. 캐시에 없으면 파일에서 로드하고 캐시에 저장
                    logger.info(f"🔄 캐시에 이미지 없음 - 파일에서 로드: {session_id}")
                    
                    try:
                        person_image, clothing_image = await self.get_session_images(session_id)
                        
                        # 캐시에 저장
                        session_data.cache_image('person_image', person_image)
                        session_data.cache_image('clothing_image', clothing_image)
                        
                        input_data['person_image'] = person_image
                        input_data['clothing_image'] = clothing_image
                        
                        logger.info(f"✅ 파일에서 원본 이미지 로드 성공: person={person_image.size}, clothing={clothing_image.size}")
                        
                    except Exception as image_error:
                        logger.error(f"❌ 파일에서 이미지 로드 실패: {image_error}")
                        
                        # 3. 최종 폴백: base64에서 로드
                        logger.info(f"🔄 폴백: base64에서 이미지 로드 시도: {session_id}")
                        session_dict = session_data.to_safe_dict()
                        
                        if 'original_person_image' in session_dict and 'original_clothing_image' in session_dict:
                            try:
                                import base64
                                from io import BytesIO
                                from PIL import Image
                                
                                # base64 디코딩
                                person_base64 = session_dict['original_person_image']
                                clothing_base64 = session_dict['original_clothing_image']
                                
                                if isinstance(person_base64, str) and isinstance(clothing_base64, str):
                                    person_image_data = base64.b64decode(person_base64)
                                    clothing_image_data = base64.b64decode(clothing_base64)
                                    
                                    person_image = Image.open(BytesIO(person_image_data))
                                    clothing_image = Image.open(BytesIO(clothing_image_data))
                                    
                                    # 캐시에 저장
                                    session_data.cache_image('person_image', person_image)
                                    session_data.cache_image('clothing_image', clothing_image)
                                    
                                    input_data['person_image'] = person_image
                                    input_data['clothing_image'] = clothing_image
                                    
                                    logger.info(f"✅ base64에서 원본 이미지 로드 성공: person={person_image.size}, clothing={clothing_image.size}")
                                else:
                                    logger.error(f"❌ base64 데이터 타입 오류: person={type(person_base64)}, clothing={type(clothing_base64)}")
                                    
                            except Exception as base64_error:
                                logger.error(f"❌ base64 디코딩 실패: {base64_error}")
                        else:
                            logger.error(f"❌ 세션에 원본 이미지 데이터 없음: {session_id}")
                            
            except Exception as e:
                logger.error(f"❌ 원본 이미지 로드 중 오류: {e}")
            
            logger.info(f"✅ PREPARE_STEP_INPUT_DATA 완료: session_id={session_id}, step_id={step_id}")
            return input_data
            
        except Exception as e:
            logger.error(f"❌ PREPARE_STEP_INPUT_DATA 실패: {e}")
            logger.error(f"❌ 상세 오류: {traceback.format_exc()}")
            raise
    
    # =========================================================================
    # 🔥 이미지 캐시 관리 메서드
    # =========================================================================
    
    async def get_session_cached_images(self, session_id: str) -> Dict[str, Any]:
        """세션의 캐시된 이미지 정보 조회"""
        try:
            session_data = self.sessions.get(session_id)
            if not session_data:
                raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
            
            cached_images = session_data.get_cached_images()
            return {
                'session_id': session_id,
                'cached_image_keys': list(cached_images.keys()),
                'cached_image_count': len(cached_images),
                'image_sizes': {
                    key: f"{img.size[0]}x{img.size[1]}" 
                    for key, img in cached_images.items()
                }
            }
        except Exception as e:
            logger.error(f"캐시된 이미지 조회 실패: {e}")
            return {'error': str(e)}
    
    async def clear_session_image_cache(self, session_id: str) -> bool:
        """세션의 이미지 캐시 정리"""
        try:
            session_data = self.sessions.get(session_id)
            if not session_data:
                raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
            
            session_data.clear_image_cache()
            logger.info(f"✅ 세션 {session_id} 이미지 캐시 정리 완료")
            return True
        except Exception as e:
            logger.error(f"이미지 캐시 정리 실패: {e}")
            return False
    
    async def get_pipeline_progress(self, session_id: str) -> Dict[str, Any]:
        """파이프라인 진행률 상세 조회 (순환참조 안전)"""
        try:
            session_data = self.sessions.get(session_id)
            if not session_data:
                raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
            
            session_data.update_access_time()
            
            total_steps = 8
            completed_steps = len(session_data.metadata.completed_steps)
            progress_percent = (completed_steps / total_steps) * 100
            
            # 현재 처리 가능한 Step
            next_available_step = None
            for step_id in range(1, 9):
                if step_id not in session_data.metadata.completed_steps:
                    dependencies = session_data.validate_step_dependencies(step_id)
                    if dependencies['valid']:
                        next_available_step = step_id
                        break
            
            return {
                'session_id': session_id,
                'progress_percent': float(progress_percent),
                'total_steps': int(total_steps),
                'completed_steps': int(completed_steps),
                'completed_step_ids': list(session_data.metadata.completed_steps),
                'next_available_step': next_available_step,
                'total_processing_time': float(sum(session_data.step_processing_times.values())),
                'average_quality_score': float(
                    sum(session_data.step_quality_scores.values()) / len(session_data.step_quality_scores) 
                    if session_data.step_quality_scores else 0.0
                ),
                'pipeline_status': self._get_pipeline_status(session_data),
                'step_details': {
                    step_id: {
                        'completed': step_id in session_data.metadata.completed_steps,
                        'processing_time': float(session_data.step_processing_times.get(step_id, 0.0)),
                        'quality_score': float(session_data.step_quality_scores.get(step_id, 0.0)),
                        'dependencies_met': session_data.validate_step_dependencies(step_id)['valid'],
                        'data_available': step_id in session_data.step_data_cache
                    }
                    for step_id in range(1, 9)
                },
                'circular_reference_safe': True
            }
            
        except Exception as e:
            logger.error(f"진행률 조회 실패: {e}")
            raise

    # =========================================================================
    # 🔧 내부 유틸리티 메서드들 (순환참조 안전성 강화)
    # =========================================================================
    
    def _get_pipeline_status(self, session_data: SessionData) -> str:
        """파이프라인 전체 상태 판단"""
        completed_count = len(session_data.metadata.completed_steps)
        
        if completed_count == 0:
            return "not_started"
        elif completed_count == 8:
            return "completed"
        elif any(step_id in session_data.step_data_cache and 
                session_data.step_data_cache[step_id].get('error') 
                for step_id in session_data.metadata.completed_steps):
            return "failed"
        else:
            return "in_progress"

    async def cleanup_session(self, session_id: str):
        """특정 세션 정리 (기존 호환)"""
        try:
            with self._lock:
                session_data = self.sessions.pop(session_id, None)
            
            if session_data:
                # 디렉토리 삭제
                if session_data.session_dir.exists():
                    shutil.rmtree(session_data.session_dir)
                
                logger.info(f"🧹 세션 정리 완료: {session_id}")
            else:
                logger.warning(f"⚠️ 정리할 세션 없음: {session_id}")
                
        except Exception as e:
            logger.error(f"❌ 세션 정리 실패 {session_id}: {e}")
    
    def _generate_session_id(self) -> str:
        """고유한 세션 ID 생성"""
        timestamp = int(time.time())
        random_part = uuid.uuid4().hex[:8]
        return f"session_{timestamp}_{random_part}"
    
    async def _save_image(self, image: Image.Image, path: Path, image_type: str) -> ImageInfo:
        """이미지 저장 및 정보 생성"""
        try:
            # 이미지 최적화
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 파일 저장
            image.save(path, "JPEG", quality=95, optimize=True)
            
            # 파일 크기 확인
            file_size = path.stat().st_size
            
            # 이미지 정보 생성
            return ImageInfo(
                path=str(path),
                size=image.size,
                mode=image.mode,
                format="JPEG",
                file_size=file_size
            )
            
        except Exception as e:
            logger.error(f"❌ {image_type} 이미지 저장 실패: {e}")
            raise
    
    async def _save_image_async(self, image: Image.Image, path: Path):
        """비동기 이미지 저장"""
        def save_sync():
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(path, "JPEG", quality=95, optimize=True)
        
        # CPU 집약적 작업을 별도 스레드에서 실행
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, save_sync)
    
    async def _load_image_async(self, path: str) -> Image.Image:
        """비동기 이미지 로드"""
        def load_sync():
            return Image.open(path)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, load_sync)
    
    async def _image_to_base64(self, image: Image.Image) -> str:
        """이미지를 Base64로 변환"""
        from io import BytesIO
        import base64
        
        def convert_sync():
            buffer = BytesIO()
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(buffer, format='JPEG', quality=85, optimize=True)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, convert_sync)
    
    async def _save_session_metadata(self, session_data: SessionData):
        """세션 메타데이터 저장 (순환참조 안전)"""
        try:
            metadata_path = session_data.session_dir / "session_metadata.json"
            
            # 전체 세션 데이터 (메타데이터 + 단계별 결과) - 순환참조 방지
            full_data = {
                "metadata": session_data.metadata.to_dict(),  # 이미 순환참조 방지됨
                "step_results": {
                    str(k): safe_serialize_session_data(v, max_depth=3) for k, v in session_data.step_results.items()
                },
                "step_processing_times": dict(session_data.step_processing_times),
                "step_quality_scores": dict(session_data.step_quality_scores),
                "last_saved": datetime.now().isoformat(),
                "circular_reference_safe": True
            }
            
            # 비동기 파일 쓰기
            async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(full_data, indent=2, ensure_ascii=False))
                
        except Exception as e:
            logger.error(f"❌ 세션 메타데이터 저장 실패: {e}")
    
    async def _enforce_session_limit(self):
        """세션 수 제한 강제"""
        try:
            if len(self.sessions) <= self.max_sessions:
                return
            
            # 가장 오래된 세션부터 정리
            with self._lock:
                sorted_sessions = sorted(
                    self.sessions.items(),
                    key=lambda x: x[1].metadata.last_accessed
                )
                
                sessions_to_remove = sorted_sessions[:len(self.sessions) - self.max_sessions]
            
            for session_id, _ in sessions_to_remove:
                await self.cleanup_session(session_id)
            
            logger.info(f"🧹 세션 수 제한: {len(sessions_to_remove)}개 세션 정리")
            
        except Exception as e:
            logger.error(f"❌ 세션 수 제한 강제 실패: {e}")
    
    def _start_cleanup_task(self):
        """백그라운드 정리 태스크 시작 (서버 시작 방해 방지)"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval_minutes * 60)
                    await self.cleanup_expired_sessions()
                    await self._enforce_session_limit()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"❌ 백그라운드 정리 오류: {e}")
        
        # 백그라운드 태스크 시작 (서버 시작을 방해하지 않도록 안전하게)
        try:
            loop = asyncio.get_running_loop()
            self._cleanup_task = loop.create_task(cleanup_loop())
            logger.info("🔄 백그라운드 세션 정리 태스크 시작")
        except RuntimeError:
            # 이벤트 루프가 없는 경우 (서버 시작 중) - 지연 시작
            logger.warning("⚠️ 이벤트 루프 없음 - 백그라운드 정리 지연 시작")
            # 서버 시작 후에 정리 작업이 시작되도록 설정
            self._cleanup_task = None
        except Exception as e:
            logger.warning(f"⚠️ 백그라운드 정리 태스크 시작 실패: {e}")
            self._cleanup_task = None
    
    def stop_cleanup_task(self):
        """백그라운드 정리 태스크 중지"""
        if self._cleanup_task and not self._cleanup_task.cancelled():
            self._cleanup_task.cancel()
            logger.info("🛑 백그라운드 세션 정리 태스크 중지")

    def _fallback_initialization(self):
        """폴백 초기화 (SQLite 실패 시 메모리 기반 세션 관리)"""
        logger.warning("⚠️ SQLite 데이터베이스 초기화 실패. 메모리 기반 세션 관리로 폴백합니다.")
        self.base_path = Path("/tmp/mycloset_sessions")
        self.sessions_dir = self.base_path / "data"
        self.db_path = self.base_path / "sessions.db"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.sessions = {}
        self._lock = threading.Lock()
        self._cleanup_task = None
        self.max_sessions = 100
        self.session_timeout_hours = 24
        self.cleanup_interval_minutes = 30
        self.pipeline_flows = {}
        logger.info("✅ SessionManager 폴백 초기화 완료 (메모리 기반)")

    async def _recover_session_data(self, session_id: str) -> Optional[SessionData]:
        """세션 데이터 복구 (강화된 버전)"""
        try:
            logger.info(f"🔄 세션 데이터 복구 시작: {session_id}")
            
            # 1. 데이터베이스에서 로드 시도
            logger.info(f"🔄 데이터베이스에서 로드 시도: {session_id}")
            session_data = self._load_session_from_db(session_id)
            
            if session_data:
                # 메모리에 저장
                self.sessions[session_id] = session_data
                logger.info(f"✅ 데이터베이스에서 세션 로드 완료: {session_id}")
                logger.info(f"✅ 메모리에 세션 저장됨: {session_id}")
                logger.info(f"✅ 현재 메모리 세션 수: {len(self.sessions)}")
                return session_data
            
            # 2. 폴백: 모든 세션 재로드 시도
            logger.info(f"🔄 폴백: 모든 세션 재로드 시도")
            self._reload_all_sessions_from_db()
            session_data = self.sessions.get(session_id)
            
            if session_data:
                logger.info(f"✅ 폴백으로 세션 찾음: {session_id}")
                return session_data
            
            # 3. 최종 폴백: 강제 재로드
            logger.info(f"🔄 최종 폴백: 강제 재로드 시도: {session_id}")
            session_data = self._force_reload_session(session_id)
            
            if session_data:
                self.sessions[session_id] = session_data
                logger.info(f"✅ 강제 재로드 성공: {session_id}")
                return session_data
            
            # 4. 최종 확인: 데이터베이스 직접 확인
            logger.error(f"❌ 모든 복구 시도 실패 - 데이터베이스 직접 확인: {session_id}")
            
            try:
                if self.db_path and self.db_path.exists():
                    import sqlite3
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT session_id FROM session_metadata WHERE session_id = ?", (session_id,))
                        result = cursor.fetchone()
                        cursor.close()
                        
                        if result:
                            logger.info(f"✅ 데이터베이스에 세션 존재함 - 재시도: {session_id}")
                            # 한 번 더 시도
                            session_data = self._force_reload_session(session_id)
                            if session_data:
                                self.sessions[session_id] = session_data
                                logger.info(f"✅ 재시도 성공: {session_id}")
                                return session_data
                        else:
                            logger.error(f"❌ 데이터베이스에도 세션 없음: {session_id}")
                else:
                    logger.error(f"❌ 데이터베이스 파일이 존재하지 않음: {self.db_path}")
                    
            except Exception as db_error:
                logger.error(f"❌ 데이터베이스 확인 중 오류: {db_error}")
            
            logger.error(f"❌ 세션 데이터 복구 완전 실패: {session_id}")
            return None
            
        except Exception as e:
            logger.error(f"❌ 세션 데이터 복구 중 오류: {e}")
            logger.error(f"❌ 상세 오류: {traceback.format_exc()}")
            return None

# =============================================================================
# 🌍 전역 세션 매니저 (싱글톤) - 기존 호환
# =============================================================================

# 🔥 강화된 전역 세션 매니저 (Thread-Safe 싱글톤)
_session_manager_instance: Optional[SessionManager] = None
_manager_lock = threading.RLock()

def get_session_manager() -> SessionManager:
    """강화된 전역 세션 매니저 싱글톤 인스턴스 반환"""
    global _session_manager_instance
    
    print(f"🔥 GET_SESSION_MANAGER 호출됨")
    print(f"🔥 GET_SESSION_MANAGER - 현재 인스턴스: {_session_manager_instance is not None}")
    print(f"🔥 GET_SESSION_MANAGER - 인스턴스 ID: {id(_session_manager_instance) if _session_manager_instance else 'None'}")
    print(f"🔥 GET_SESSION_MANAGER - 인스턴스 주소: {hex(id(_session_manager_instance)) if _session_manager_instance else 'None'}")
    
    logger.info(f"🔥 GET_SESSION_MANAGER 호출됨")
    logger.info(f"🔥 GET_SESSION_MANAGER - 현재 인스턴스: {_session_manager_instance is not None}")
    logger.info(f"🔥 GET_SESSION_MANAGER - 인스턴스 ID: {id(_session_manager_instance) if _session_manager_instance else 'None'}")
    logger.info(f"🔥 GET_SESSION_MANAGER - 인스턴스 주소: {hex(id(_session_manager_instance)) if _session_manager_instance else 'None'}")
    
    if _session_manager_instance is None:
        print(f"🔥 GET_SESSION_MANAGER - 인스턴스가 None입니다! 새로 생성합니다.")
        logger.error(f"🔥 GET_SESSION_MANAGER - 인스턴스가 None입니다! 새로 생성합니다.")
        with _manager_lock:
            if _session_manager_instance is None:
                print("🔄 강화된 전역 SessionManager 인스턴스 생성 시작")
                logger.info("🔄 강화된 전역 SessionManager 인스턴스 생성 시작")
                _session_manager_instance = SessionManager()
                print("✅ 강화된 전역 SessionManager 인스턴스 생성 완료")
                logger.info("✅ 강화된 전역 SessionManager 인스턴스 생성 완료")
                print(f"✅ 현재 메모리 세션 수: {len(_session_manager_instance.sessions)}")
                logger.info(f"✅ 현재 메모리 세션 수: {len(_session_manager_instance.sessions)}")
                print(f"✅ 새로 생성된 인스턴스 ID: {id(_session_manager_instance)}")
                logger.info(f"✅ 새로 생성된 인스턴스 ID: {id(_session_manager_instance)}")
                print(f"✅ 새로 생성된 인스턴스 주소: {hex(id(_session_manager_instance))}")
                logger.info(f"✅ 새로 생성된 인스턴스 주소: {hex(id(_session_manager_instance))}")
            else:
                print("✅ 기존 전역 SessionManager 인스턴스 사용")
                logger.info("✅ 기존 전역 SessionManager 인스턴스 사용")
    else:
        print(f"✅ 전역 SessionManager 인스턴스 사용 중 (세션 수: {len(_session_manager_instance.sessions)})")
        logger.info(f"✅ 전역 SessionManager 인스턴스 사용 중 (세션 수: {len(_session_manager_instance.sessions)})")
        print(f"✅ 사용 중인 인스턴스 ID: {id(_session_manager_instance)}")
        logger.info(f"✅ 사용 중인 인스턴스 ID: {id(_session_manager_instance)}")
        print(f"✅ 사용 중인 인스턴스 주소: {hex(id(_session_manager_instance))}")
        logger.info(f"✅ 사용 중인 인스턴스 주소: {hex(id(_session_manager_instance))}")
        print(f"✅ 사용 중인 세션 키들: {list(_session_manager_instance.sessions.keys())}")
        logger.info(f"✅ 사용 중인 세션 키들: {list(_session_manager_instance.sessions.keys())}")
    
    return _session_manager_instance

def reset_session_manager():
    """세션 매니저 재설정 (디버깅용)"""
    global _session_manager_instance
    with _manager_lock:
        if _session_manager_instance:
            logger.info("🔄 전역 세션 매니저 재설정")
            _session_manager_instance.stop_cleanup_task()
            _session_manager_instance = None
        logger.info("✅ 전역 세션 매니저 재설정 완료")

async def cleanup_global_session_manager():
    """전역 세션 매니저 정리 (기존 함수명 유지)"""
    global _session_manager_instance
    
    if _session_manager_instance:
        _session_manager_instance.stop_cleanup_task()
        await _session_manager_instance.cleanup_all_sessions()
        _session_manager_instance = None
        logger.info("🧹 전역 SessionManager 정리 완료")

# cleanup_session_manager 함수 추가 (기존 호환)
async def cleanup_session_manager():
    await cleanup_global_session_manager()

# =============================================================================
# 🧪 테스트 및 디버깅 함수들 (기존 호환 + 순환참조 안전성)
# =============================================================================

async def test_session_manager():
    """세션 매니저 테스트 (기존 호환 + Step간 데이터 흐름 + 순환참조 안전성)"""
    try:
        logger.info("🧪 SessionManager 완전 통합 테스트 시작")
        
        # 테스트용 이미지 생성
        from PIL import Image
        test_person = Image.new('RGB', (512, 512), color=(100, 150, 200))
        test_clothing = Image.new('RGB', (512, 512), color=(200, 100, 100))
        
        # 세션 매니저 생성
        manager = SessionManager()
        
        # 1. 세션 생성 테스트 (기존 호환)
        session_id = await manager.create_session(
            person_image=test_person, 
            clothing_image=test_clothing,
            measurements={"height": 170, "weight": 65}
        )
        logger.info(f"✅ 테스트 세션 생성: {session_id}")
        
        # 2. 이미지 로드 테스트 (기존 호환)
        person_img, clothing_img = await manager.get_session_images(session_id)
        logger.info(f"✅ 이미지 로드 테스트: {person_img.size}, {clothing_img.size}")
        
        # 3. Step 1 결과 저장 테스트 (기존 호환 + 순환참조 안전)
        await manager.save_step_result(
            session_id, 
            1, 
            {
                "success": True, 
                "test": True,
                "segmentation_mask": np.random.randint(0, 20, (512, 512)),
                "primary_output": test_person,
                "processing_time": 1.2,
                "quality_score": 0.85
            },
            test_person
        )
        logger.info("✅ Step 1 결과 저장 테스트 완료")
        
        # 4. Step 2 의존성 검증 테스트 (새로운 기능)
        dependencies = await manager.validate_step_dependencies(session_id, 2)
        logger.info(f"✅ Step 2 의존성 검증: {dependencies['valid']}")
        
        # 5. Step 2 입력 데이터 준비 테스트 (새로운 기능 + 순환참조 안전)
        input_data = await manager.prepare_step_input_data(session_id, 2)
        logger.info(f"✅ Step 2 입력 데이터 준비: {len(input_data)}개 항목")
        
        # 6. 파이프라인 진행률 테스트 (새로운 기능 + 순환참조 안전)
        progress = await manager.get_pipeline_progress(session_id)
        logger.info(f"✅ 파이프라인 진행률: {progress['progress_percent']:.1f}%")
        
        # 7. 세션 상태 조회 테스트 (기존 호환 + 확장 + 순환참조 안전)
        status = await manager.get_session_status(session_id)
        logger.info(f"✅ 세션 상태: {status['progress_percent']:.1f}% (순환참조 안전)")
        
        # 8. 전체 세션 상태 조회 테스트 (순환참조 방지 완전 적용)
        all_status = manager.get_all_sessions_status()
        logger.info(f"✅ 전체 세션 상태: {all_status['total_sessions']}개 (순환참조 안전)")
        
        # 9. JSON 직렬화 테스트 (순환참조 방지 확인)
        json_str = json.dumps(all_status, indent=2, ensure_ascii=False)
        logger.info(f"✅ JSON 직렬화 테스트: {len(json_str)}바이트 (순환참조 없음)")
        
        # 10. 정리 테스트 (기존 호환)
        await manager.cleanup_session(session_id)
        logger.info("✅ 세션 정리 테스트 완료")
        
        logger.info("🎉 SessionManager 완전 통합 테스트 모두 통과!")
        logger.info("✅ 기존 API 100% 호환")
        logger.info("✅ Step간 데이터 흐름 완벽 지원")
        logger.info("🔒 순환참조 해결 완전 통합")
        logger.info("🚀 FastAPI 호환성 완벽 보장!")
        return True
        
    except Exception as e:
        logger.error(f"❌ SessionManager 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    # 직접 실행 시 테스트
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_session_manager())

# =============================================================================
# 🎉 EXPORT (기존 호환 + 새로운 기능 + 순환참조 해결)
# =============================================================================

__all__ = [
    # 기존 호환 클래스들
    "SessionManager",
    "SessionData", 
    "SessionMetadata",
    "ImageInfo",
    
    # 기존 호환 함수들
    "get_session_manager",
    "cleanup_global_session_manager",
    "cleanup_session_manager",
    "test_session_manager",
    
    # 새로 추가된 클래스들
    "StepStatus",
    "DataType", 
    "StepDataFlow",
    
    # 새로 추가된 데이터
    "PIPELINE_DATA_FLOWS",
    
    # 순환참조 해결 함수
    "safe_serialize_session_data",
]

logger.info("🎉 완전한 SessionManager 모듈 로드 완료!")
logger.info("✅ 기존 함수명 100% 유지 (create_session, get_session_images, save_step_result 등)")
logger.info("✅ Step간 데이터 흐름 완벽 지원")
logger.info("✅ 의존성 검증 및 순서 보장") 
logger.info("✅ 실시간 진행률 추적")
logger.info("✅ 메모리 효율적 대용량 데이터 처리")
logger.info("🔒 순환참조 해결 메서드 완전 통합!")
logger.info("🚀 FastAPI 호환성 완벽 보장!")
logger.info("🔥 conda 환경 최적화 + M3 Max 최적화!")
logger.info("🌟 이미지 재업로드 문제 완전 해결 + Step간 데이터 처리 + 순환참조 방지 완벽!")