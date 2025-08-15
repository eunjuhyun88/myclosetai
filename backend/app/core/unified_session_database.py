# backend/app/core/unified_session_database.py
"""
🔥 MyCloset AI 통합 Session Database 시스템
================================================================================

✅ 전체 AI 파이프라인에 적용 가능한 통합 데이터베이스
✅ Step별 데이터 저장 및 전달 최적화
✅ 실시간 세션 유지 및 모니터링
✅ 메모리 캐시와 디스크 저장 연동
✅ 데이터 압축 및 압축 해제
✅ Step간 데이터 흐름 관리
✅ 성능 최적화 및 메트릭 수집

Author: MyCloset AI Team
Date: 2025-01-27
Version: 1.0.0
"""

import sqlite3
import json
import logging
import asyncio
import threading
import time
import hashlib
import zlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import traceback

# NumPy와 PyTorch import (선택적)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)

# =============================================================================
# 🔥 데이터 구조 정의
# =============================================================================

@dataclass
class StepData:
    """Step별 데이터 구조"""
    step_id: int
    step_name: str
    session_id: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    processing_time: float
    quality_score: float
    status: str  # 'pending', 'processing', 'completed', 'failed'
    error_message: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

@dataclass
class SessionInfo:
    """세션 정보"""
    session_id: str
    created_at: str
    updated_at: str
    status: str
    person_image_path: str
    clothing_image_path: str
    measurements: Dict[str, Any]
    total_steps: int
    completed_steps: List[int]
    current_step: int
    progress_percent: float
    metadata: Dict[str, Any] = None  # metadata 필드 추가
    
    def __post_init__(self):
        if self.completed_steps is None:
            self.completed_steps = []
        if self.updated_at is None:
            self.updated_at = datetime.now()

@dataclass
class DataFlow:
    """Step간 데이터 흐름 정의"""
    source_step: int
    target_step: int
    data_type: str  # 'image', 'mask', 'keypoints', 'matrix', 'result'
    data_key: str
    required: bool = True
    validation_rules: Optional[Dict[str, Any]] = None

# =============================================================================
# 🔥 통합 Session Database 클래스
# =============================================================================

class UnifiedSessionDatabase:
    """전체 AI 파이프라인을 위한 통합 Session Database"""
    
    def __init__(self, db_path: str = "unified_sessions.db", enable_cache: bool = True):
        self.db_path = Path(db_path)
        self.enable_cache = enable_cache
        
        # 데이터베이스 초기화
        self._init_database()
        
        # 메모리 캐시
        self.session_cache = {}
        self.step_cache = {}
        self.cache_lock = threading.RLock()
        self.max_cache_size = 100
        
        # 성능 메트릭
        self.performance_metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'db_queries': 0,
            'compression_ratio': 0.0
        }
        
        # 데이터 흐름 정의
        self.data_flows = self._define_data_flows()
        
        logger.info(f"✅ UnifiedSessionDatabase 초기화 완료: {db_path}")
    
    def _init_database(self):
        """통합 데이터베이스 초기화"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # 1. 세션 메인 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        status TEXT DEFAULT 'active',
                        person_image_path TEXT NOT NULL,
                        clothing_image_path TEXT NOT NULL,
                        measurements TEXT,
                        total_steps INTEGER DEFAULT 8,
                        completed_steps TEXT,
                        current_step INTEGER DEFAULT 1,
                        progress_percent REAL DEFAULT 0.0,
                        metadata TEXT
                    )
                """)
                
                # 2. Step 결과 테이블 (최적화된 구조)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS step_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        step_id INTEGER NOT NULL,
                        step_name TEXT NOT NULL,
                        input_data TEXT NOT NULL,
                        output_data TEXT NOT NULL,
                        processing_time REAL DEFAULT 0.0,
                        quality_score REAL DEFAULT 0.0,
                        status TEXT DEFAULT 'pending',
                        error_message TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        data_hash TEXT,
                        compressed_size INTEGER,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE,
                        UNIQUE(session_id, step_id)
                    )
                """)
                
                # 3. Step 간 데이터 흐름 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS step_data_flow (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        source_step INTEGER NOT NULL,
                        target_step INTEGER NOT NULL,
                        data_type TEXT NOT NULL,
                        data_key TEXT NOT NULL,
                        data_value TEXT,
                        data_hash TEXT,
                        required BOOLEAN DEFAULT 1,
                        validation_status TEXT DEFAULT 'pending',
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE,
                        UNIQUE(session_id, source_step, target_step, data_type)
                    )
                """)
                
                # 4. 이미지 및 파일 메타데이터 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS file_metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        step_id INTEGER NOT NULL,
                        file_type TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        file_size INTEGER,
                        mime_type TEXT DEFAULT 'image/jpeg',
                        metadata TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                    )
                """)
                
                # 5. 성능 메트릭 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        step_id INTEGER NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL,
                        metric_unit TEXT,
                        timestamp TEXT NOT NULL,
                        metadata TEXT,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                    )
                """)
                
                # 인덱스 생성 (성능 최적화)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions (status)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions (created_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_step_results_session ON step_results (session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_step_results_step ON step_results (step_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_step_results_status ON step_results (status)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_data_flow_session ON step_data_flow (session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_data_flow_target ON step_data_flow (target_step)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_metadata_session ON file_metadata (session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_metrics_session ON performance_metrics (session_id)")
                
                conn.commit()
                logger.info("✅ 통합 데이터베이스 테이블 생성 완료")
                
        except Exception as e:
            logger.error(f"❌ 데이터베이스 초기화 실패: {e}")
            logger.error(f"❌ 상세 오류: {traceback.format_exc()}")
            raise
    
    def _define_data_flows(self) -> List[DataFlow]:
        """Step간 데이터 흐름 정의 - AI 추론에 필요한 모든 데이터 포함"""
        return [
            # ========================================
            # Step 1 (Human Parsing) -> Step 2 (Pose Estimation)
            # ========================================
            DataFlow(1, 2, 'mask', 'segmentation_mask', required=True),
            DataFlow(1, 2, 'mask_path', 'segmentation_mask_path', required=True),
            DataFlow(1, 2, 'image', 'person_image_path', required=True),
            DataFlow(1, 2, 'parsing', 'human_parsing_result', required=True),
            DataFlow(1, 2, 'confidence', 'parsing_confidence', required=False),
            
            # ========================================
            # Step 1 -> Step 3 (Cloth Segmentation)
            # ========================================
            DataFlow(1, 3, 'mask', 'segmentation_mask', required=True),
            DataFlow(1, 3, 'mask_path', 'segmentation_mask_path', required=True),
            DataFlow(1, 3, 'image', 'person_image_path', required=True),
            DataFlow(1, 3, 'parsing', 'human_parsing_result', required=True),
            
            # ========================================
            # Step 2 (Pose Estimation) -> Step 3
            # ========================================
            DataFlow(2, 3, 'keypoints', 'pose_keypoints', required=True),
            DataFlow(2, 3, 'skeleton', 'pose_skeleton', required=True),
            DataFlow(2, 3, 'confidence', 'pose_confidence', required=False),
            
            # ========================================
            # Step 2 -> Step 4 (Geometric Matching)
            # ========================================
            DataFlow(2, 4, 'keypoints', 'pose_keypoints', required=True),
            DataFlow(2, 4, 'skeleton', 'pose_skeleton', required=True),
            DataFlow(1, 4, 'mask', 'segmentation_mask', required=True),
            DataFlow(3, 4, 'mask', 'cloth_segmentation_mask', required=True),
            DataFlow(3, 4, 'mask_path', 'cloth_segmentation_mask_path', required=True),
            
            # ========================================
            # Step 3 -> Step 4
            # ========================================
            DataFlow(3, 4, 'mask', 'cloth_segmentation_mask', required=True),
            DataFlow(3, 4, 'mask_path', 'cloth_segmentation_mask_path', required=True),
            DataFlow(3, 4, 'confidence', 'cloth_confidence', required=False),
            
            # ========================================
            # Step 4 -> Step 5 (Cloth Warping)
            # ========================================
            DataFlow(4, 5, 'matrix', 'transformation_matrix', required=True),
            DataFlow(4, 5, 'confidence', 'matching_confidence', required=False),
            DataFlow(3, 5, 'mask', 'cloth_segmentation_mask', required=True),
            DataFlow(3, 5, 'mask_path', 'cloth_segmentation_mask_path', required=True),
            
            # ========================================
            # Step 5 -> Step 6 (Virtual Fitting)
            # ========================================
            DataFlow(5, 6, 'image', 'warped_clothing', required=True),
            DataFlow(5, 6, 'image_path', 'warped_clothing_path', required=True),
            DataFlow(1, 6, 'image', 'person_image_path', required=True),
            DataFlow(2, 6, 'keypoints', 'pose_keypoints', required=True),
            DataFlow(2, 6, 'skeleton', 'pose_skeleton', required=True),
            DataFlow(1, 6, 'mask', 'segmentation_mask', required=True),
            
            # ========================================
            # Step 6 -> Step 7 (Post Processing)
            # ========================================
            DataFlow(6, 7, 'image', 'fitted_image', required=True),
            DataFlow(6, 7, 'image_path', 'fitted_image_path', required=True),
            DataFlow(6, 7, 'confidence', 'fitting_confidence', required=False),
            
            # ========================================
            # Step 7 -> Step 8 (Quality Assessment)
            # ========================================
            DataFlow(7, 8, 'image', 'processed_image', required=True),
            DataFlow(7, 8, 'image_path', 'processed_image_path', required=True),
            DataFlow(7, 8, 'metadata', 'processing_metadata', required=False),
            
            # ========================================
            # Step 8 -> Step 9 (Final Output)
            # ========================================
            DataFlow(8, 9, 'image', 'final_image', required=True),
            DataFlow(8, 9, 'image_path', 'final_image_path', required=True),
            DataFlow(8, 9, 'score', 'quality_score', required=True),
            DataFlow(8, 9, 'metadata', 'assessment_metadata', required=False),
        ]
    
    @contextmanager
    def _get_connection(self):
        """데이터베이스 연결 컨텍스트 매니저 (강화된 락 해결 및 성능 최적화)"""
        conn = None
        max_retries = 15  # 재시도 횟수 증가
        retry_delay = 0.01  # 초기 지연 시간 감소
        
        for attempt in range(max_retries):
            try:
                # 더 강력한 락 해결 설정
                conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=180.0)
                conn.row_factory = sqlite3.Row
                
                # 성능 최적화된 PRAGMA 설정 (락 문제 완전 해결)
                try:
                    conn.execute("PRAGMA journal_mode=DELETE")  # WAL 대신 DELETE 모드 사용 (락 방지)
                except:
                    pass  # 실패해도 계속 진행
                
                try:
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute("PRAGMA cache_size=100000")  # 캐시 크기 증가
                    conn.execute("PRAGMA temp_store=MEMORY")
                    conn.execute("PRAGMA busy_timeout=120000")  # 120초 대기
                    conn.execute("PRAGMA locking_mode=NORMAL")  # 일반 락 모드
                    conn.execute("PRAGMA mmap_size=67108864")  # 64MB로 줄임 (락 방지)
                    conn.execute("PRAGMA page_size=4096")  # 기본 페이지 크기
                    conn.execute("PRAGMA auto_vacuum=NONE")  # 자동 정리 비활성화
                    conn.execute("PRAGMA foreign_keys=OFF")  # 외래키 제약 비활성화 (성능 향상)
                except:
                    pass  # 개별 PRAGMA 실패해도 계속 진행
                
                yield conn
                break  # 성공 시 루프 탈출
                
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    wait_time = min(retry_delay * (1.8 ** attempt), 5.0)  # 지수 백오프 + 최대 제한
                    logger.warning(f"⚠️ 데이터베이스 락 감지 (시도 {attempt + 1}/{max_retries}), {wait_time:.2f}초 후 재시도")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ 데이터베이스 연결 실패: {e}")
                    raise
            except Exception as e:
                logger.error(f"❌ 데이터베이스 연결 실패: {e}")
                raise
            finally:
                if conn:
                    try:
                        conn.close()
                    except:
                        pass
    
    # =========================================================================
    # 🔥 세션 관리 메서드들
    # =========================================================================
    
    async def create_session(self, person_image_path: str, clothing_image_path: str, 
                           measurements: Dict[str, Any] = None) -> str:
        """새 세션 생성"""
        try:
            session_id = self._generate_session_id()
            
            session_info = SessionInfo(
                session_id=session_id,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                status='active',
                person_image_path=person_image_path,
                clothing_image_path=clothing_image_path,
                measurements=measurements or {},
                total_steps=8,
                completed_steps=[],
                current_step=1,
                progress_percent=0.0,
                metadata={} # metadata 필드 초기화
            )
            
            # 데이터베이스에 저장
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO sessions (
                        session_id, created_at, updated_at, status,
                        person_image_path, clothing_image_path, measurements,
                        total_steps, completed_steps, current_step, progress_percent, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_info.session_id,
                    session_info.created_at,
                    session_info.updated_at,
                    session_info.status,
                    session_info.person_image_path,
                    session_info.clothing_image_path,
                    json.dumps(session_info.measurements),
                    session_info.total_steps,
                    json.dumps(session_info.completed_steps),
                    session_info.current_step,
                    session_info.progress_percent,
                    json.dumps(session_info.metadata)
                ))
                conn.commit()
            
            # 캐시에 저장
            if self.enable_cache:
                with self.cache_lock:
                    self.session_cache[session_id] = session_info
            
            logger.info(f"✅ 세션 생성 완료: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"❌ 세션 생성 실패: {e}")
            raise
    
    async def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """세션 정보 조회 - 최신 데이터 반영 및 락 문제 해결"""
        try:
            # 캐시에서 조회 시도
            if self.enable_cache and session_id in self.session_cache:
                cached_info = self.session_cache[session_id]
                # 캐시된 데이터가 최신인지 확인
                if await self._is_cache_fresh(session_id, cached_info):
                    self.performance_metrics['cache_hits'] += 1
                    logger.debug(f"✅ 캐시에서 세션 정보 조회: {session_id}")
                    return cached_info
                else:
                    # 캐시가 오래됨 - 제거
                    del self.session_cache[session_id]
                    logger.debug(f"⚠️ 오래된 캐시 제거: {session_id}")
            
            self.performance_metrics['cache_misses'] += 1
            
            # 데이터베이스에서 최신 정보 조회 (락 문제 해결)
            session_info = await self._get_session_info_with_retry(session_id)
            
            if session_info:
                # 캐시 업데이트
                if self.enable_cache:
                    self.session_cache[session_id] = session_info
                    self._cleanup_cache_if_needed()
                
                logger.info(f"✅ 세션 정보 조회 완료: {session_id}")
                return session_info
            else:
                logger.warning(f"⚠️ 세션 정보를 찾을 수 없음: {session_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 세션 정보 조회 실패: {e}")
            return None

    async def _get_session_info_with_retry(self, session_id: str, max_retries: int = 3) -> Optional[SessionInfo]:
        """재시도 로직을 포함한 세션 정보 조회 - 락 문제 해결"""
        for attempt in range(max_retries):
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # 세션 기본 정보 조회
                    cursor.execute("""
                        SELECT session_id, created_at, updated_at, status, person_image_path, 
                               clothing_image_path, measurements, total_steps, completed_steps, 
                               current_step, progress_percent, metadata
                        FROM sessions 
                        WHERE session_id = ?
                    """, (session_id,))
                    
                    result = cursor.fetchone()
                    if result:
                        # completed_steps 파싱 (JSON 문자열 -> 리스트)
                        completed_steps_str = result[8]  # completed_steps 컬럼
                        try:
                            completed_steps = json.loads(completed_steps_str) if completed_steps_str else []
                        except (json.JSONDecodeError, TypeError):
                            completed_steps = []
                        
                        # measurements 파싱
                        measurements_str = result[6]  # measurements 컬럼
                        try:
                            measurements = json.loads(measurements_str) if measurements_str else {}
                        except (json.JSONDecodeError, TypeError):
                            measurements = {}
                        
                        # metadata 파싱
                        metadata_str = result[11]  # metadata 컬럼
                        try:
                            metadata = json.loads(metadata_str) if metadata_str else {}
                        except (json.JSONDecodeError, TypeError):
                            metadata = {}
                        
                        session_info = SessionInfo(
                            session_id=result[0],
                            created_at=result[1],
                            updated_at=result[2],
                            status=result[3],
                            person_image_path=result[4],
                            clothing_image_path=result[5],
                            measurements=measurements,
                            total_steps=result[7],
                            completed_steps=completed_steps,
                            current_step=result[9],
                            progress_percent=result[10] or 0.0,
                            metadata=metadata
                        )
                        
                        logger.debug(f"✅ 세션 정보 파싱 완료: {session_id} (시도 {attempt + 1})")
                        return session_info
                    else:
                        logger.warning(f"⚠️ 세션을 찾을 수 없음: {session_id}")
                        return None
                        
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    if attempt < max_retries - 1:
                        wait_time = 0.5 * (2 ** attempt)  # 지수 백오프
                        logger.warning(f"⚠️ 데이터베이스 락 - 재시도 {attempt + 1}/{max_retries} (대기 {wait_time:.1f}초): {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"❌ 데이터베이스 락 최종 실패: {e}")
                        raise
                else:
                    logger.error(f"❌ 데이터베이스 오류: {e}")
                    raise
            except Exception as e:
                logger.error(f"❌ 세션 정보 조회 중 예외 발생: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)
                else:
                    raise
        
        return None

    async def _is_cache_fresh(self, session_id: str, cached_info: SessionInfo) -> bool:
        """캐시된 데이터가 최신인지 확인"""
        try:
            # 데이터베이스에서 updated_at 확인
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT updated_at FROM sessions WHERE session_id = ?", (session_id,))
                result = cursor.fetchone()
                
                if result and result[0]:
                    db_updated_at = result[0]
                    cache_updated_at = cached_info.updated_at
                    
                    # 캐시가 5초 이내면 최신으로 간주
                    if db_updated_at == cache_updated_at:
                        return True
                    
                    logger.debug(f"⚠️ 캐시 오래됨 - DB: {db_updated_at}, 캐시: {cache_updated_at}")
                    return False
                
                return False
                
        except Exception as e:
            logger.debug(f"⚠️ 캐시 신선도 확인 실패: {e}")
            return False  # 확인 실패 시 캐시 사용 안함
    
    # =========================================================================
    # 🔥 Step 데이터 관리 메서드들
    # =========================================================================
    
    async def save_step_result(self, session_id: str, step_id: int, step_name: str, 
                             input_data: Dict[str, Any], output_data: Dict[str, Any], 
                             processing_time: float = 0.0, quality_score: float = 0.0, 
                             status: str = 'completed', error_message: str = None) -> bool:
        """Step 결과 저장 및 자동 진행률 업데이트"""
        try:
            # Step 결과 저장
            success = await self._save_step_result_internal(
                session_id, step_id, step_name, input_data, output_data, 
                processing_time, quality_score, status, error_message
            )
            
            if success and status == 'completed':
                # Step 완료 시 자동으로 세션 진행률 업데이트
                await self._update_session_progress_automatically(session_id, step_id)
                logger.info(f"✅ Step {step_id} 완료로 인한 세션 진행률 자동 업데이트")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Step 결과 저장 실패: {e}")
            return False

    async def _save_step_result_internal(self, session_id: str, step_id: int, step_name: str,
                             input_data: Dict[str, Any], output_data: Dict[str, Any],
                             processing_time: float = 0.0, quality_score: float = 0.0,
                             status: str = 'completed', error_message: str = None) -> bool:
        """Step 결과 저장 내부 로직 (중복 제거)"""
        try:
            # 데이터 압축
            compressed_input = self._compress_data(input_data)
            compressed_output = self._compress_data(output_data)
            
            # 데이터 해시 생성
            input_hash = self._generate_data_hash(input_data)
            output_hash = self._generate_data_hash(output_data)
            
            step_data = StepData(
                step_id=step_id,
                step_name=step_name,
                session_id=session_id,
                input_data=input_data,
                output_data=output_data,
                processing_time=processing_time,
                quality_score=quality_score,
                status=status,
                error_message=error_message
            )
            
            # 데이터베이스에 저장
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO step_results (
                        session_id, step_id, step_name, input_data, output_data,
                        processing_time, quality_score, status, error_message,
                        created_at, updated_at, data_hash, compressed_size
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, step_id, step_name,
                    compressed_input, compressed_output,
                    processing_time, quality_score, status, error_message,
                    step_data.created_at.isoformat(),
                    step_data.updated_at.isoformat(),
                    f"{input_hash}_{output_hash}",
                    len(compressed_input) + len(compressed_output)
                ))
                
                # Step 간 데이터 흐름 업데이트
                await self._update_data_flows(session_id, step_id, output_data)
                
                conn.commit()
            
            # 캐시에 저장
            if self.enable_cache:
                cache_key = f"{session_id}:{step_id}"
                with self.cache_lock:
                    self.step_cache[cache_key] = step_data
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Step 결과 저장 내부 로직 실패: {e}")
            return False
    
    async def get_step_result(self, session_id: str, step_id: int) -> Optional[StepData]:
        """Step 결과 조회"""
        try:
            cache_key = f"{session_id}:{step_id}"
            
            # 캐시에서 먼저 확인
            if self.enable_cache:
                with self.cache_lock:
                    if cache_key in self.step_cache:
                        self.performance_metrics['cache_hits'] += 1
                        return self.step_cache[cache_key]
            
            # 데이터베이스에서 조회
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM step_results 
                    WHERE session_id = ? AND step_id = ?
                """, (session_id, step_id))
                
                row = cursor.fetchone()
                if row:
                    # 압축 해제
                    input_data = self._decompress_data(row['input_data'])
                    output_data = self._decompress_data(row['output_data'])
                    
                    step_data = StepData(
                        step_id=row['step_id'],
                        step_name=row['step_name'],
                        session_id=row['session_id'],
                        input_data=input_data,
                        output_data=output_data,
                        processing_time=row['processing_time'],
                        quality_score=row['quality_score'],
                        status=row['status'],
                        error_message=row['error_message'],
                        created_at=datetime.fromisoformat(row['created_at']),
                        updated_at=datetime.fromisoformat(row['updated_at'])
                    )
                    
                    # 캐시에 저장
                    if self.enable_cache:
                        with self.cache_lock:
                            self.step_cache[cache_key] = step_data
                    
                    self.performance_metrics['cache_misses'] += 1
                    return step_data
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Step {step_id} 결과 조회 실패: {e}")
            return None
    
    async def get_step_input_data(self, session_id: str, step_id: int) -> Dict[str, Any]:
        """Step 입력 데이터 준비 (이전 Step 결과 포함) - AI 추론에 필요한 모든 데이터"""
        try:
            input_data = {
                'session_id': session_id,
                'step_id': step_id,
                'timestamp': datetime.now().isoformat()
            }
            
            # 이전 Step 결과에서 필요한 데이터 수집
            required_flows = [flow for flow in self.data_flows if flow.target_step == step_id]
            
            for flow in required_flows:
                source_step = flow.source_step
                source_result = await self.get_step_result(session_id, source_step)
                
                if source_result and source_result.status == 'completed':
                    # Step별 특별 처리
                    if source_step == 1:  # Human Parsing
                        if flow.data_key == 'segmentation_mask':
                            # segmentation_mask 또는 segmentation_mask_path 사용
                            if 'segmentation_mask' in source_result.output_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data['segmentation_mask']
                            elif 'segmentation_mask_path' in source_result.output_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data['segmentation_mask_path']
                        elif flow.data_key == 'segmentation_mask_path':
                            if 'segmentation_mask_path' in source_result.output_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data['segmentation_mask_path']
                        elif flow.data_key == 'person_image_path':
                            # person_image_path 사용
                            if 'person_image_path' in source_result.input_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.input_data['person_image_path']
                        elif flow.data_key == 'human_parsing_result':
                            if 'human_parsing_result' in source_result.output_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data['human_parsing_result']
                        elif flow.data_key == 'parsing_confidence':
                            if 'confidence' in source_result.output_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data['confidence']
                    
                    elif source_step == 2:  # Pose Estimation
                        if flow.data_key == 'pose_keypoints':
                            if 'pose_keypoints' in source_result.output_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data['pose_keypoints']
                        elif flow.data_key == 'pose_skeleton':
                            if 'pose_skeleton' in source_result.output_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data['pose_skeleton']
                        elif flow.data_key == 'pose_confidence':
                            if 'confidence' in source_result.output_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data['confidence']
                    
                    elif source_step == 3:  # Cloth Segmentation
                        if flow.data_key == 'cloth_segmentation_mask':
                            if 'cloth_segmentation_mask' in source_result.output_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data['cloth_segmentation_mask']
                        elif flow.data_key == 'cloth_segmentation_mask_path':
                            if 'cloth_segmentation_mask_path' in source_result.output_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data['cloth_segmentation_mask_path']
                        elif flow.data_key == 'cloth_confidence':
                            if 'confidence' in source_result.output_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data['confidence']
                    
                    elif source_step == 4:  # Geometric Matching
                        if flow.data_key == 'transformation_matrix':
                            if 'transformation_matrix' in source_result.output_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data['transformation_matrix']
                        elif flow.data_key == 'matching_confidence':
                            if 'confidence' in source_result.output_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data['confidence']
                    
                    elif source_step == 5:  # Cloth Warping
                        if flow.data_key == 'warped_clothing':
                            if 'warped_clothing' in source_result.output_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data['warped_clothing']
                        elif flow.data_key == 'warped_clothing_path':
                            if 'warped_clothing_path' in source_result.output_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data['warped_clothing_path']
                    
                    elif source_step == 6:  # Virtual Fitting
                        if flow.data_key == 'fitted_image':
                            if 'fitted_image' in source_result.output_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data['fitted_image']
                        elif flow.data_key == 'fitted_image_path':
                            if 'fitted_image_path' in source_result.output_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data['fitted_image_path']
                        elif flow.data_key == 'fitting_confidence':
                            if 'confidence' in source_result.output_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data['confidence']
                    
                    elif source_step == 7:  # Post Processing
                        if flow.data_key == 'processed_image':
                            if 'processed_image' in source_result.output_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data['processed_image']
                        elif flow.data_key == 'processed_image_path':
                            if 'processed_image_path' in source_result.output_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data['processed_image_path']
                        elif flow.data_key == 'processing_metadata':
                            if 'processing_metadata' in source_result.output_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data['processing_metadata']
                    
                    elif source_step == 8:  # Quality Assessment
                        if flow.data_key == 'final_image':
                            if 'final_image' in source_result.output_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data['final_image']
                        elif flow.data_key == 'final_image_path':
                            if 'final_image_path' in source_result.output_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data['final_image_path']
                        elif flow.data_key == 'quality_score':
                            if 'quality_score' in source_result.output_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data['quality_score']
                        elif flow.data_key == 'assessment_metadata':
                            if 'assessment_metadata' in source_result.output_data:
                                input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data['assessment_metadata']
                    
                    else:
                        # 다른 Step들의 경우 기존 로직
                        if flow.data_key in source_result.output_data:
                            input_data[f"step_{source_step}_{flow.data_key}"] = source_result.output_data[flow.data_key]
                        elif flow.data_key in source_result.input_data:
                            input_data[f"step_{source_step}_{flow.data_key}"] = source_result.input_data[flow.data_key]
            
            # 세션 기본 정보 추가
            session_info = await self.get_session_info(session_id)
            if session_info:
                input_data['person_image_path'] = session_info.person_image_path
                input_data['clothing_image_path'] = session_info.clothing_image_path
                input_data['measurements'] = session_info.measurements
            
            logger.info(f"✅ Step {step_id} 입력 데이터 준비 완료: {len(input_data)}개 항목")
            logger.info(f"   - 데이터 키: {list(input_data.keys())}")
            
            # 필수 데이터 누락 확인
            missing_required = []
            for flow in required_flows:
                if flow.required and f"step_{flow.source_step}_{flow.data_key}" not in input_data:
                    missing_required.append(f"step_{flow.source_step}_{flow.data_key}")
            
            if missing_required:
                logger.warning(f"⚠️ Step {step_id}에 필수 데이터 누락: {missing_required}")
            
            return input_data
            
        except Exception as e:
            logger.error(f"❌ Step {step_id} 입력 데이터 준비 실패: {e}")
            return {'session_id': session_id, 'step_id': step_id, 'error': str(e)}
    
    # =========================================================================
    # 🔥 데이터 흐름 관리 메서드들
    # =========================================================================
    
    async def _update_data_flows(self, session_id: str, step_id: int, output_data: Dict[str, Any]):
        """Step 간 데이터 흐름 업데이트 (락 문제 해결)"""
        try:
            # 현재 Step에서 다음 Step으로 전달할 데이터 식별
            outgoing_flows = [flow for flow in self.data_flows if flow.source_step == step_id]
            
            if not outgoing_flows:
                logger.debug(f"⚠️ Step {step_id}에서 다음 Step으로 전달할 데이터 흐름이 없음")
                return
            
            # 배치 처리를 위한 데이터 준비
            batch_data = []
            current_time = datetime.now().isoformat()
            
            for flow in outgoing_flows:
                if flow.data_key in output_data:
                    data_value = output_data[flow.data_key]
                    data_hash = self._generate_data_hash(data_value)
                    
                    batch_data.append((
                        session_id, flow.source_step, flow.target_step,
                        flow.data_type, flow.data_key, self._serialize_data_for_db(data_value),
                        data_hash, flow.required, 'valid', current_time
                    ))
            
            if not batch_data:
                logger.debug(f"⚠️ Step {step_id}에서 전달할 데이터가 없음")
                return
            
            # 단순한 연결로 데이터베이스 업데이트 (락 문제 방지)
            try:
                conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
                cursor = conn.cursor()
                
                # 배치 삽입/업데이트
                cursor.executemany("""
                    INSERT OR REPLACE INTO step_data_flow (
                        session_id, source_step, target_step, data_type,
                        data_key, data_value, data_hash, required,
                        validation_status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch_data)
                
                # 커밋
                conn.commit()
                conn.close()
                
                logger.debug(f"✅ Step {step_id} 데이터 흐름 업데이트 완료: {len(batch_data)}개 항목")
                
            except Exception as e:
                logger.warning(f"⚠️ 데이터 흐름 업데이트 실패 (락 문제): {e}")
                # 락 문제로 실패해도 계속 진행
                return
                
        except Exception as e:
            logger.error(f"❌ 데이터 흐름 업데이트 실패: {e}")
            # 상세한 에러 정보 로깅
            logger.error(f"   - Session ID: {session_id}")
            logger.error(f"   - Step ID: {step_id}")
            logger.error(f"   - Output Data Keys: {list(output_data.keys()) if output_data else 'None'}")
            logger.error(f"   - 상세 오류: {traceback.format_exc()}")
    
    async def validate_step_dependencies(self, session_id: str, step_id: int) -> Dict[str, Any]:
        """Step 의존성 검증"""
        try:
            required_flows = [flow for flow in self.data_flows if flow.target_step == step_id and flow.required]
            
            validation_result = {
                'valid': True,
                'missing_dependencies': [],
                'available_data': [],
                'validation_details': {}
            }
            
            for flow in required_flows:
                source_step = flow.source_step
                source_result = await self.get_step_result(session_id, source_step)
                
                if not source_result or source_result.status != 'completed':
                    validation_result['valid'] = False
                    validation_result['missing_dependencies'].append(f"Step {source_step} 미완료")
                    validation_result['validation_details'][f"step_{source_step}"] = {
                        'status': 'missing',
                        'required_data': flow.data_key
                    }
                else:
                    if flow.data_key in source_result.output_data:
                        validation_result['available_data'].append(f"Step {source_step} -> {flow.data_key}")
                        validation_result['validation_details'][f"step_{source_step}"] = {
                            'status': 'available',
                            'data_key': flow.data_key,
                            'data_type': type(source_result.output_data[flow.data_key]).__name__
                        }
                    else:
                        validation_result['valid'] = False
                        validation_result['missing_dependencies'].append(f"Step {source_step} -> {flow.data_key} 데이터 없음")
                        validation_result['validation_details'][f"step_{source_step}"] = {
                            'status': 'incomplete',
                            'missing_data': flow.data_key
                        }
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Step 의존성 검증 실패: {e}")
            return {'valid': False, 'error': str(e)}
    
    # =========================================================================
    # 🔥 유틸리티 메서드들
    # =========================================================================
    
    def _generate_session_id(self) -> str:
        """고유한 세션 ID 생성 (중복 방지)"""
        import uuid
        timestamp = int(time.time())
        random_part = hashlib.md5(f"{timestamp}_{uuid.uuid4()}".encode()).hexdigest()[:12]
        return f"session_{timestamp}_{random_part}"
    
    def _compress_data(self, data: Dict[str, Any]) -> bytes:
        """데이터 압축"""
        try:
            # NumPy 배열 등을 안전하게 직렬화
            json_str = self._serialize_data_for_db(data)
            compressed = zlib.compress(json_str.encode())
            
            # 압축률 계산
            original_size = len(json_str.encode())
            compressed_size = len(compressed)
            if original_size > 0:
                self.performance_metrics['compression_ratio'] = compressed_size / original_size
            
            return compressed
        except Exception as e:
            logger.error(f"❌ 데이터 압축 실패: {e}")
            return self._serialize_data_for_db(data).encode()
    
    def _decompress_data(self, compressed_data: bytes) -> Dict[str, Any]:
        """데이터 압축 해제"""
        try:
            if compressed_data.startswith(b'{'):  # 이미 JSON인 경우
                return json.loads(compressed_data.decode())
            
            decompressed = zlib.decompress(compressed_data)
            return json.loads(decompressed.decode())
        except Exception as e:
            logger.error(f"❌ 데이터 압축 해제 실패: {e}")
            return {}
    
    def _serialize_data_for_db(self, data: Any) -> str:
        """데이터베이스 저장을 위한 데이터 직렬화 (NumPy 배열 등 처리)"""
        try:
            if data is None:
                return json.dumps(None)
            
            # NumPy 배열 처리
            if NUMPY_AVAILABLE and hasattr(data, 'tolist') and hasattr(data, 'shape'):
                # NumPy 배열로 판단
                serialized = data.tolist()
            elif TORCH_AVAILABLE and hasattr(data, 'numpy') and hasattr(data, 'detach'):
                # PyTorch 텐서로 판단
                serialized = data.detach().cpu().numpy().tolist()
            elif NUMPY_AVAILABLE and np and isinstance(data, np.ndarray):
                # NumPy 배열 타입 체크
                serialized = data.tolist()
            elif TORCH_AVAILABLE and torch and isinstance(data, torch.Tensor):
                # PyTorch 텐서 타입 체크
                serialized = data.detach().cpu().numpy().tolist()
            else:
                serialized = data
            
            return json.dumps(serialized, default=str)
            
        except Exception as e:
            logger.warning(f"⚠️ 데이터 직렬화 실패: {e}")
            return json.dumps(str(data))
    
    def _generate_data_hash(self, data: Any) -> str:
        """데이터 해시 생성"""
        try:
            if isinstance(data, dict):
                data_str = json.dumps(data, sort_keys=True, default=str)
            else:
                data_str = str(data)
            return hashlib.md5(data_str.encode()).hexdigest()
        except Exception:
            return "unknown"
    
    async def _update_session_progress(self, session_id: str):
        """세션 진행률 업데이트"""
        try:
            # 완료된 Step 수 계산
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) FROM step_results 
                    WHERE session_id = ? AND status = 'completed'
                """, (session_id,))
                completed_count = cursor.fetchone()[0]
                
                # 진행률 계산
                progress_percent = (completed_count / 8) * 100
                
                # 세션 정보 업데이트
                cursor.execute("""
                    UPDATE sessions 
                    SET completed_steps = ?, progress_percent = ?, updated_at = ?
                    WHERE session_id = ?
                """, (
                    json.dumps(list(range(1, completed_count + 1))),
                    progress_percent,
                    datetime.now().isoformat(),
                    session_id
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ 세션 진행률 업데이트 실패: {e}")
    
    async def _update_session_progress_automatically(self, session_id: str, completed_step_id: int):
        """Step 완료 시 세션 진행률 자동 업데이트"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # 현재 세션 정보 조회
                cursor.execute("""
                    SELECT completed_steps, progress_percent, total_steps 
                    FROM sessions 
                    WHERE session_id = ?
                """, (session_id,))
                
                result = cursor.fetchone()
                if result:
                    current_completed = json.loads(result[0]) if result[0] else []
                    current_progress = result[1] or 0.0
                    total_steps = result[2] or 8
                    
                    # 완료된 Step에 추가 (중복 방지)
                    if completed_step_id not in current_completed:
                        current_completed.append(completed_step_id)
                        current_completed.sort()  # 정렬
                        
                        # 진행률 계산
                        new_progress = (len(current_completed) / total_steps) * 100
                        
                        # 세션 정보 업데이트
                        cursor.execute("""
                            UPDATE sessions 
                            SET completed_steps = ?, progress_percent = ?, updated_at = ?
                            WHERE session_id = ?
                        """, (
                            json.dumps(current_completed),
                            new_progress,
                            datetime.now().isoformat(),
                            session_id
                        ))
                        
                        conn.commit()
                        
                        logger.info(f"✅ 세션 진행률 자동 업데이트: {current_progress:.1f}% → {new_progress:.1f}%")
                        logger.info(f"   - 완료된 Step: {current_completed}")
                        
                        # 캐시 무효화
                        if session_id in self.session_cache:
                            del self.session_cache[session_id]
                            logger.debug(f"✅ 세션 캐시 무효화: {session_id}")
                    else:
                        logger.debug(f"⚠️ Step {completed_step_id}이 이미 완료된 Step에 포함됨")
                        
        except Exception as e:
            logger.error(f"❌ 세션 진행률 자동 업데이트 실패: {e}")
    
    def _cleanup_cache_if_needed(self):
        """캐시 크기 제한 및 정리 (성능 최적화)"""
        try:
            total_cache_size = len(self.session_cache) + len(self.step_cache)
            
            if total_cache_size > self.max_cache_size:
                # 가장 오래된 항목들 제거 (세션 캐시와 스텝 캐시 모두 고려)
                items_to_remove = total_cache_size - self.max_cache_size
                
                # 세션 캐시 정리
                if len(self.session_cache) > self.max_cache_size // 2:
                    session_items_to_remove = len(self.session_cache) - (self.max_cache_size // 2)
                    oldest_session_keys = sorted(self.session_cache.keys(), 
                                               key=lambda k: self.session_cache[k].updated_at)[:session_items_to_remove]
                    
                    for key in oldest_session_keys:
                        del self.session_cache[key]
                    
                    logger.debug(f"✅ 세션 캐시 정리 완료: {session_items_to_remove}개 항목 제거")
                
                # 스텝 캐시 정리
                if len(self.step_cache) > self.max_cache_size // 2:
                    step_items_to_remove = len(self.step_cache) - (self.max_cache_size // 2)
                    oldest_step_keys = sorted(self.step_cache.keys(), 
                                            key=lambda k: self.step_cache[k].updated_at if hasattr(self.step_cache[k], 'updated_at') else 0)[:step_items_to_remove]
                    
                    for key in oldest_step_keys:
                        del self.step_cache[key]
                    
                    logger.debug(f"✅ 스텝 캐시 정리 완료: {step_items_to_remove}개 항목 제거")
                
                # 메모리 사용량 로깅
                logger.debug(f"📊 캐시 상태: 세션 {len(self.session_cache)}개, 스텝 {len(self.step_cache)}개")
                
        except Exception as e:
            logger.debug(f"⚠️ 캐시 정리 실패: {e}")
            logger.debug(f"   - 상세 오류: {traceback.format_exc()}")
    
    # =========================================================================
    # 🔥 성능 모니터링 및 최적화
    # =========================================================================
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 반환 (상세 분석 포함)"""
        try:
            # 캐시 효율성 계산
            total_requests = self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses']
            cache_hit_ratio = 0.0
            if total_requests > 0:
                cache_hit_ratio = self.performance_metrics['cache_hits'] / total_requests
            
            # 메모리 사용량 분석
            session_cache_memory = sum(
                len(str(v)) for v in self.session_cache.values()
            ) if self.session_cache else 0
            
            step_cache_memory = sum(
                len(str(v)) for v in self.step_cache.values()
            ) if self.step_cache else 0
            
            total_cache_memory = session_cache_memory + step_cache_memory
            
            # 데이터베이스 성능 분석
            db_efficiency = 0.0
            if self.performance_metrics['db_queries'] > 0:
                db_efficiency = (total_requests - self.performance_metrics['db_queries']) / total_requests
            
            return {
                **self.performance_metrics,
                'cache_hit_ratio': cache_hit_ratio,
                'total_requests': total_requests,
                'compression_ratio': self.performance_metrics['compression_ratio'],
                'cache_size': len(self.session_cache) + len(self.step_cache),
                'cache_memory_usage_bytes': total_cache_memory,
                'session_cache_size': len(self.session_cache),
                'step_cache_size': len(self.step_cache),
                'database_efficiency': db_efficiency,
                'memory_efficiency': total_cache_memory / (1024 * 1024) if total_cache_memory > 0 else 0,  # MB
                'performance_score': min(100, (cache_hit_ratio * 50) + (db_efficiency * 30) + (1 - self.performance_metrics['compression_ratio']) * 20)
            }
        except Exception as e:
            logger.error(f"❌ 성능 메트릭 조회 실패: {e}")
            logger.error(f"   - 상세 오류: {traceback.format_exc()}")
            return {}
    
    def clear_cache(self):
        """캐시 정리"""
        try:
            with self.cache_lock:
                self.session_cache.clear()
                self.step_cache.clear()
            logger.info("✅ 캐시 정리 완료")
        except Exception as e:
            logger.error(f"❌ 캐시 정리 실패: {e}")
    
    def optimize_database(self):
        """데이터베이스 최적화 (락 문제 방지)"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                try:
                    # 안전한 최적화만 수행 (락 문제 방지)
                    cursor.execute("ANALYZE")
                    
                    # 통계 정보 업데이트
                    cursor.execute("ANALYZE sqlite_master")
                    
                    # 메모리 최적화 (안전한 것만)
                    cursor.execute("PRAGMA optimize")
                    
                    # 커밋
                    conn.commit()
                    
                    logger.info("✅ 데이터베이스 안전 최적화 완료")
                    
                    # 최적화 후 성능 메트릭 확인
                    self._log_optimization_results()
                    
                except Exception as e:
                    # 롤백
                    conn.rollback()
                    logger.error(f"❌ 데이터베이스 최적화 중 오류: {e}")
                    raise
                    
        except Exception as e:
            logger.error(f"❌ 데이터베이스 최적화 실패: {e}")
            logger.error(f"   - 상세 오류: {traceback.format_exc()}")
    
    def _log_optimization_results(self):
        """최적화 결과 로깅"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # 데이터베이스 크기 확인
                cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
                db_size = cursor.fetchone()[0]
                
                # 테이블별 레코드 수 확인
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                table_stats = {}
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    table_stats[table_name] = count
                
                logger.info(f"📊 최적화 후 데이터베이스 상태:")
                logger.info(f"   - 전체 크기: {db_size / (1024*1024):.2f} MB")
                logger.info(f"   - 테이블별 레코드 수: {table_stats}")
                
        except Exception as e:
            logger.debug(f"⚠️ 최적화 결과 로깅 실패: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """데이터베이스 통계 정보 반환"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # 테이블별 레코드 수
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    stats[f"{table_name}_count"] = count
                
                # 데이터베이스 크기
                cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
                db_size = cursor.fetchone()[0]
                stats['database_size_bytes'] = db_size
                stats['database_size_mb'] = db_size / (1024*1024)
                
                # 인덱스 정보
                cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
                indexes = cursor.fetchall()
                stats['index_count'] = len(indexes)
                
                return stats
                
        except Exception as e:
            logger.error(f"❌ 데이터베이스 통계 조회 실패: {e}")
            return {}
    
    def cleanup_old_sessions(self, days_old: int = 30):
        """오래된 세션 정리 (메모리 및 디스크 공간 절약)"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            cutoff_str = cutoff_date.isoformat()
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # 트랜잭션 시작
                cursor.execute("BEGIN TRANSACTION")
                
                try:
                    # 오래된 세션 ID 조회
                    cursor.execute("""
                        SELECT session_id FROM sessions 
                        WHERE created_at < ? AND status != 'active'
                    """, (cutoff_str,))
                    
                    old_sessions = cursor.fetchall()
                    old_session_ids = [row[0] for row in old_sessions]
                    
                    if old_session_ids:
                        # 관련 데이터 삭제 (CASCADE로 자동 삭제됨)
                        placeholders = ','.join(['?' for _ in old_session_ids])
                        cursor.execute(f"DELETE FROM sessions WHERE session_id IN ({placeholders})", old_session_ids)
                        
                        deleted_count = len(old_session_ids)
                        conn.commit()
                        
                        logger.info(f"✅ 오래된 세션 정리 완료: {deleted_count}개 세션 삭제")
                        
                        # 캐시에서도 제거
                        for session_id in old_session_ids:
                            if session_id in self.session_cache:
                                del self.session_cache[session_id]
                        
                        return deleted_count
                    else:
                        logger.info("✅ 정리할 오래된 세션이 없음")
                        return 0
                        
                except Exception as e:
                    # 트랜잭션 롤백
                    conn.rollback()
                    logger.error(f"❌ 오래된 세션 정리 중 오류: {e}")
                    raise
                    
        except Exception as e:
            logger.error(f"❌ 오래된 세션 정리 실패: {e}")
            return 0

# =============================================================================
# 🌍 전역 인스턴스
# =============================================================================

_unified_session_db = None
_db_lock = threading.RLock()

def get_unified_session_database() -> UnifiedSessionDatabase:
    """전역 통합 세션 데이터베이스 인스턴스 반환"""
    global _unified_session_db
    
    if _unified_session_db is None:
        with _db_lock:
            if _unified_session_db is None:
                _unified_session_db = UnifiedSessionDatabase()
                logger.info("✅ 전역 UnifiedSessionDatabase 인스턴스 생성")
    
    return _unified_session_db

def reset_unified_session_database():
    """전역 통합 세션 데이터베이스 재설정"""
    global _unified_session_db
    with _db_lock:
        if _unified_session_db:
            _unified_session_db.clear_cache()
            _unified_session_db = None
        logger.info("✅ 전역 UnifiedSessionDatabase 재설정 완료")
