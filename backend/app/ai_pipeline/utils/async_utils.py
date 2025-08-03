# backend/app/ai_pipeline/utils/async_utils.py
"""
🔄 MyCloset AI - 비동기 유틸리티 시스템 v1.0
==============================================
✅ 비동기 작업 관리 및 스케줄링
✅ 동시성 제어 및 제한
✅ 작업 큐 및 배치 처리
✅ 비동기 컨텍스트 관리
✅ 순환참조 방지 - 독립적 모듈
✅ conda 환경 우선 지원

Author: MyCloset AI Team
Date: 2025-07-21
Version: 1.0 (분리된 비동기 유틸리티 시스템)
"""

import asyncio
import logging
import time
import threading
import weakref
from typing import Any, Dict, List, Optional, Callable, Union, Coroutine, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import asynccontextmanager
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

# ==============================================
# 🔥 1. 비동기 작업 관련 열거형
# ==============================================

class TaskPriority(Enum):
    """작업 우선순위"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class TaskStatus(Enum):
    """작업 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ConcurrencyStrategy(Enum):
    """동시성 전략"""
    UNLIMITED = "unlimited"
    LIMITED = "limited"
    SEMAPHORE = "semaphore"
    QUEUE = "queue"

# ==============================================
# 🔥 2. 비동기 작업 데이터 구조
# ==============================================

@dataclass
class AsyncTask:
    """비동기 작업 정보"""
    id: str
    name: str
    coro: Coroutine
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConcurrencyConfig:
    """동시성 설정"""
    max_concurrent_tasks: int = 10
    max_workers: int = 4
    strategy: ConcurrencyStrategy = ConcurrencyStrategy.LIMITED
    enable_task_queue: bool = True
    queue_size: int = 100
    default_timeout: float = 300.0  # 5분
    enable_retries: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0

# ==============================================
# 🔥 3. 비동기 작업 스케줄러
# ==============================================

class AsyncTaskScheduler:
    """비동기 작업 스케줄러"""
    
    def __init__(self, config: Optional[ConcurrencyConfig] = None):
        self.config = config or ConcurrencyConfig()
        self.logger = logging.getLogger(f"{__name__}.AsyncTaskScheduler")
        
        # 작업 관리
        self.tasks: Dict[str, AsyncTask] = {}
        self.task_queue = deque()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # 동시성 제어
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        self.task_counter = 0
        
        # 스레드 풀
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # 통계
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "cancelled_tasks": 0,
            "average_execution_time": 0.0
        }
        
        # 락
        self._lock = asyncio.Lock()
        self._running = False
        
        self.logger.info(f"🔄 비동기 작업 스케줄러 초기화: {self.config.strategy.value}")
    
    async def schedule_task(
        self,
        coro: Coroutine,
        name: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None
    ) -> str:
        """작업 스케줄링"""
        # 에러 컨텍스트 준비
        error_context = {
            'task_name': name or f"async_task_{self.task_counter + 1}",
            'priority': priority.value,
            'timeout': timeout or self.config.default_timeout,
            'max_retries': max_retries or self.config.max_retries,
            'scheduler_running': self._running,
            'total_tasks': self.stats["total_tasks"],
            'running_tasks': len(self.running_tasks),
            'queue_size': len(self.task_queue)
        }
        
        try:
            async with self._lock:
                self.task_counter += 1
                task_id = f"task_{self.task_counter}_{int(time.time())}"
                
                task = AsyncTask(
                    id=task_id,
                    name=name or f"async_task_{self.task_counter}",
                    coro=coro,
                    priority=priority,
                    timeout=timeout or self.config.default_timeout,
                    max_retries=max_retries or self.config.max_retries
                )
                
                self.tasks[task_id] = task
                self.stats["total_tasks"] += 1
                
                # 우선순위 기반 큐에 추가
                self._add_to_queue(task)
                
                self.logger.debug(f"📝 작업 스케줄링: {task_id} ({task.name})")
                
                # 스케줄러 시작
                if not self._running:
                    asyncio.create_task(self._run_scheduler())
                
                return task_id
                
        except Exception as e:
            # exceptions.py의 커스텀 예외로 변환
            from app.core.exceptions import (
                convert_to_mycloset_exception,
                PipelineError,
                ConfigurationError
            )
            
            # 에러 타입별 커스텀 예외 변환
            if isinstance(e, (ValueError, TypeError)):
                custom_error = ConfigurationError(
                    f"작업 스케줄링 중 설정 오류: {e}",
                    "TASK_SCHEDULING_CONFIG_ERROR",
                    error_context
                )
            else:
                custom_error = PipelineError(
                    f"작업 스케줄링 실패: {e}",
                    "TASK_SCHEDULING_FAILED",
                    error_context
                )
            
            self.logger.error(f"❌ 작업 스케줄링 실패: {custom_error}")
            raise custom_error
    
    def _add_to_queue(self, task: AsyncTask):
        """우선순위 기반 큐에 작업 추가"""
        # 우선순위에 따라 적절한 위치에 삽입
        inserted = False
        for i, queued_task in enumerate(self.task_queue):
            if task.priority.value > queued_task.priority.value:
                self.task_queue.insert(i, task)
                inserted = True
                break
        
        if not inserted:
            self.task_queue.append(task)
    
    async def _run_scheduler(self):
        """스케줄러 실행"""
        try:
            self._running = True
            
            while self.task_queue or self.running_tasks:
                # 새 작업 시작
                await self._start_pending_tasks()
                
                # 완료된 작업 정리
                await self._cleanup_completed_tasks()
                
                # 잠시 대기
                await asyncio.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"❌ 스케줄러 실행 오류: {e}")
        finally:
            self._running = False
    
    async def _start_pending_tasks(self):
        """대기 중인 작업들 시작"""
        try:
            while (self.task_queue and 
                   len(self.running_tasks) < self.config.max_concurrent_tasks):
                
                task = self.task_queue.popleft()
                
                # 동시성 제어
                if self.config.strategy == ConcurrencyStrategy.SEMAPHORE:
                    await self.semaphore.acquire()
                
                # 작업 시작
                asyncio_task = asyncio.create_task(self._execute_task(task))
                self.running_tasks[task.id] = asyncio_task
                
                task.status = TaskStatus.RUNNING
                task.started_at = time.time()
                
                self.logger.debug(f"🚀 작업 시작: {task.id}")
                
        except Exception as e:
            self.logger.error(f"❌ 작업 시작 오류: {e}")
    
    async def _execute_task(self, task: AsyncTask):
        """작업 실행"""
        # 에러 컨텍스트 준비
        error_context = {
            'task_id': task.id,
            'task_name': task.name,
            'priority': task.priority.value,
            'timeout': task.timeout,
            'retry_count': task.retry_count,
            'max_retries': task.max_retries,
            'strategy': self.config.strategy.value,
            'started_at': task.started_at
        }
        
        try:
            # 타임아웃 설정
            if task.timeout:
                result = await asyncio.wait_for(task.coro, timeout=task.timeout)
            else:
                result = await task.coro
            
            # 성공 처리
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            
            self.stats["completed_tasks"] += 1
            self._update_average_execution_time(task)
            
            self.logger.debug(f"✅ 작업 완료: {task.id}")
            
        except asyncio.TimeoutError as e:
            # 타임아웃 처리
            from app.core.exceptions import TimeoutError as MyClosetTimeoutError
            
            task.error = MyClosetTimeoutError(
                f"작업 타임아웃: {task.id}",
                "TASK_EXECUTION_TIMEOUT",
                error_context
            )
            await self._handle_task_failure(task)
            
        except Exception as e:
            # 오류 처리
            from app.core.exceptions import (
                convert_to_mycloset_exception,
                PipelineError,
                ModelInferenceError
            )
            
            # 에러 타입별 커스텀 예외 변환
            if isinstance(e, (ValueError, TypeError)):
                task.error = PipelineError(
                    f"작업 실행 중 데이터 오류: {e}",
                    "TASK_EXECUTION_DATA_ERROR",
                    error_context
                )
            elif isinstance(e, (OSError, IOError)):
                task.error = PipelineError(
                    f"작업 실행 중 시스템 오류: {e}",
                    "TASK_EXECUTION_SYSTEM_ERROR",
                    error_context
                )
            else:
                task.error = convert_to_mycloset_exception(e, error_context)
            
            await self._handle_task_failure(task)
            
        finally:
            # 정리
            if self.config.strategy == ConcurrencyStrategy.SEMAPHORE:
                self.semaphore.release()
    
    async def _handle_task_failure(self, task: AsyncTask):
        """작업 실패 처리"""
        # 에러 컨텍스트 준비
        error_context = {
            'task_id': task.id,
            'task_name': task.name,
            'retry_count': task.retry_count,
            'max_retries': task.max_retries,
            'enable_retries': self.config.enable_retries,
            'retry_delay': self.config.retry_delay,
            'original_error': str(task.error) if task.error else None,
            'error_type': type(task.error).__name__ if task.error else None
        }
        
        try:
            task.retry_count += 1
            
            if (self.config.enable_retries and 
                task.retry_count <= task.max_retries):
                
                # 재시도
                self.logger.warning(f"⚠️ 작업 재시도: {task.id} ({task.retry_count}/{task.max_retries})")
                
                # 재시도 지연
                await asyncio.sleep(self.config.retry_delay)
                
                # 큐에 다시 추가
                task.status = TaskStatus.PENDING
                self._add_to_queue(task)
                
            else:
                # 최대 재시도 초과 또는 재시도 비활성화
                task.status = TaskStatus.FAILED
                task.completed_at = time.time()
                self.stats["failed_tasks"] += 1
                
                # exceptions.py의 커스텀 예외로 변환
                from app.core.exceptions import (
                    convert_to_mycloset_exception,
                    PipelineError,
                    ModelInferenceError
                )
                
                # 원본 에러가 이미 커스텀 예외인 경우 그대로 사용
                if hasattr(task.error, 'error_code'):
                    final_error = task.error
                else:
                    # 일반 예외를 커스텀 예외로 변환
                    final_error = PipelineError(
                        f"작업 최종 실패: {task.id}",
                        "TASK_FINAL_FAILURE",
                        error_context
                    )
                
                self.logger.error(f"❌ 작업 실패: {task.id} - {final_error}")
                
        except Exception as e:
            # 실패 처리 중 발생한 에러
            from app.core.exceptions import PipelineError
            
            failure_error = PipelineError(
                f"작업 실패 처리 중 오류: {e}",
                "TASK_FAILURE_HANDLING_ERROR",
                error_context
            )
            
            self.logger.error(f"❌ 작업 실패 처리 오류: {failure_error}")
            
            # 원본 작업을 실패로 표시
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            self.stats["failed_tasks"] += 1
    
    async def _cleanup_completed_tasks(self):
        """완료된 작업 정리"""
        try:
            completed_task_ids = []
            
            for task_id, asyncio_task in self.running_tasks.items():
                if asyncio_task.done():
                    completed_task_ids.append(task_id)
            
            for task_id in completed_task_ids:
                del self.running_tasks[task_id]
                
        except Exception as e:
            self.logger.error(f"❌ 작업 정리 오류: {e}")
    
    def _update_average_execution_time(self, task: AsyncTask):
        """평균 실행 시간 업데이트"""
        if task.started_at and task.completed_at:
            execution_time = task.completed_at - task.started_at
            current_avg = self.stats["average_execution_time"]
            completed_count = self.stats["completed_tasks"]
            
            self.stats["average_execution_time"] = (
                (current_avg * (completed_count - 1) + execution_time) / completed_count
            )
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """특정 작업 완료 대기"""
        # 에러 컨텍스트 준비
        error_context = {
            'task_id': task_id,
            'timeout': timeout,
            'task_exists': task_id in self.tasks,
            'total_tasks': len(self.tasks),
            'running_tasks': len(self.running_tasks)
        }
        
        try:
            if task_id not in self.tasks:
                from app.core.exceptions import DataValidationError
                raise DataValidationError(
                    f"작업을 찾을 수 없습니다: {task_id}",
                    "TASK_NOT_FOUND",
                    error_context
                )
            
            task = self.tasks[task_id]
            start_time = time.time()
            
            # 에러 컨텍스트 업데이트
            error_context.update({
                'task_name': task.name,
                'task_status': task.status.value,
                'task_priority': task.priority.value
            })
            
            while task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                if timeout and (time.time() - start_time) > timeout:
                    from app.core.exceptions import TimeoutError as MyClosetTimeoutError
                    raise MyClosetTimeoutError(
                        f"작업 대기 타임아웃: {task_id}",
                        "TASK_WAIT_TIMEOUT",
                        error_context
                    )
                
                await asyncio.sleep(0.1)
            
            if task.status == TaskStatus.COMPLETED:
                return task.result
            elif task.status == TaskStatus.FAILED:
                # 실패한 작업의 에러를 그대로 전파
                if task.error:
                    raise task.error
                else:
                    from app.core.exceptions import PipelineError
                    raise PipelineError(
                        f"작업이 실패했습니다: {task_id}",
                        "TASK_FAILED",
                        error_context
                    )
            elif task.status == TaskStatus.CANCELLED:
                from app.core.exceptions import PipelineError
                raise PipelineError(
                    f"작업이 취소되었습니다: {task_id}",
                    "TASK_CANCELLED",
                    error_context
                )
            
        except Exception as e:
            # 이미 커스텀 예외인 경우 그대로 전파
            if hasattr(e, 'error_code'):
                self.logger.error(f"❌ 작업 대기 실패 {task_id}: {e}")
                raise
            
            # 일반 예외를 커스텀 예외로 변환
            from app.core.exceptions import (
                convert_to_mycloset_exception,
                PipelineError
            )
            
            custom_error = convert_to_mycloset_exception(e, error_context)
            self.logger.error(f"❌ 작업 대기 실패 {task_id}: {custom_error}")
            raise custom_error
    
    async def cancel_task(self, task_id: str) -> bool:
        """작업 취소"""
        try:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            
            # 실행 중인 작업 취소
            if task_id in self.running_tasks:
                asyncio_task = self.running_tasks[task_id]
                asyncio_task.cancel()
                del self.running_tasks[task_id]
            
            # 큐에서 제거
            self.task_queue = deque([t for t in self.task_queue if t.id != task_id])
            
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            self.stats["cancelled_tasks"] += 1
            
            self.logger.info(f"🚫 작업 취소: {task_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 작업 취소 실패 {task_id}: {e}")
            return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """작업 상태 조회"""
        try:
            if task_id not in self.tasks:
                return None
            
            task = self.tasks[task_id]
            
            return {
                "id": task.id,
                "name": task.name,
                "status": task.status.value,
                "priority": task.priority.value,
                "retry_count": task.retry_count,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "execution_time": (
                    task.completed_at - task.started_at 
                    if task.started_at and task.completed_at 
                    else None
                ),
                "error": str(task.error) if task.error else None,
                "metadata": task.metadata
            }
            
        except Exception as e:
            self.logger.error(f"❌ 작업 상태 조회 실패 {task_id}: {e}")
            return None
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """스케줄러 통계"""
        try:
            return {
                **self.stats,
                "pending_tasks": len(self.task_queue),
                "running_tasks": len(self.running_tasks),
                "total_managed_tasks": len(self.tasks),
                "scheduler_running": self._running,
                "config": {
                    "max_concurrent_tasks": self.config.max_concurrent_tasks,
                    "strategy": self.config.strategy.value,
                    "enable_retries": self.config.enable_retries
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def shutdown(self):
        """스케줄러 종료"""
        try:
            self.logger.info("🔄 스케줄러 종료 시작")
            
            # 대기 중인 작업들 취소
            while self.task_queue:
                task = self.task_queue.popleft()
                task.status = TaskStatus.CANCELLED
                self.stats["cancelled_tasks"] += 1
            
            # 실행 중인 작업들 취소
            for task_id, asyncio_task in self.running_tasks.items():
                asyncio_task.cancel()
                self.tasks[task_id].status = TaskStatus.CANCELLED
                self.stats["cancelled_tasks"] += 1
            
            # 모든 작업 완료 대기
            if self.running_tasks:
                await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
            
            # 스레드 풀 종료
            self.thread_pool.shutdown(wait=True)
            
            self._running = False
            self.logger.info("✅ 스케줄러 종료 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 스케줄러 종료 실패: {e}")

# ==============================================
# 🔥 4. 비동기 배치 처리기
# ==============================================

class AsyncBatchProcessor:
    """비동기 배치 처리기"""
    
    def __init__(
        self,
        batch_size: int = 10,
        max_concurrent_batches: int = 3,
        flush_interval: float = 5.0
    ):
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.flush_interval = flush_interval
        self.logger = logging.getLogger(f"{__name__}.AsyncBatchProcessor")
        
        # 배치 관리
        self.pending_items = []
        self.processing_batches = {}
        self.batch_counter = 0
        
        # 동시성 제어
        self.semaphore = asyncio.Semaphore(max_concurrent_batches)
        
        # 자동 플러시
        self._flush_task = None
        self._shutdown = False
        
        # 통계
        self.stats = {
            "total_items": 0,
            "processed_items": 0,
            "failed_items": 0,
            "total_batches": 0,
            "processed_batches": 0,
            "failed_batches": 0
        }
        
        self._lock = asyncio.Lock()
        
        self.logger.info(f"📦 배치 처리기 초기화: batch_size={batch_size}")
    
    async def add_item(self, item: Any, metadata: Optional[Dict] = None) -> str:
        """아이템 추가"""
        # 에러 컨텍스트 준비
        error_context = {
            'item_type': type(item).__name__,
            'metadata': metadata or {},
            'batch_size': self.batch_size,
            'pending_items': len(self.pending_items),
            'processing_batches': len(self.processing_batches),
            'shutdown': self._shutdown
        }
        
        try:
            async with self._lock:
                item_id = f"item_{len(self.pending_items)}_{int(time.time())}"
                
                self.pending_items.append({
                    "id": item_id,
                    "data": item,
                    "metadata": metadata or {},
                    "added_at": time.time()
                })
                
                self.stats["total_items"] += 1
                
                # 배치 크기에 도달하면 즉시 처리
                if len(self.pending_items) >= self.batch_size:
                    await self._flush_batch()
                
                # 자동 플러시 스케줄링
                if self._flush_task is None and not self._shutdown:
                    self._flush_task = asyncio.create_task(self._auto_flush())
                
                return item_id
                
        except Exception as e:
            # exceptions.py의 커스텀 예외로 변환
            from app.core.exceptions import (
                convert_to_mycloset_exception,
                PipelineError,
                DataValidationError
            )
            
            # 에러 타입별 커스텀 예외 변환
            if isinstance(e, (ValueError, TypeError)):
                custom_error = DataValidationError(
                    f"아이템 추가 중 데이터 오류: {e}",
                    "BATCH_ITEM_DATA_ERROR",
                    error_context
                )
            else:
                custom_error = PipelineError(
                    f"아이템 추가 실패: {e}",
                    "BATCH_ITEM_ADD_FAILED",
                    error_context
                )
            
            self.logger.error(f"❌ 아이템 추가 실패: {custom_error}")
            raise custom_error
    
    async def _flush_batch(self):
        """배치 플러시"""
        try:
            if not self.pending_items:
                return
            
            # 배치 생성
            self.batch_counter += 1
            batch_id = f"batch_{self.batch_counter}"
            
            batch_items = self.pending_items[:self.batch_size]
            self.pending_items = self.pending_items[self.batch_size:]
            
            self.stats["total_batches"] += 1
            
            # 비동기로 배치 처리
            asyncio.create_task(self._process_batch(batch_id, batch_items))
            
            self.logger.debug(f"📦 배치 플러시: {batch_id} ({len(batch_items)}개 아이템)")
            
        except Exception as e:
            self.logger.error(f"❌ 배치 플러시 실패: {e}")
    
    async def _process_batch(self, batch_id: str, items: List[Dict]):
        """배치 처리"""
        # 에러 컨텍스트 준비
        error_context = {
            'batch_id': batch_id,
            'item_count': len(items),
            'batch_size': self.batch_size,
            'max_concurrent_batches': self.max_concurrent_batches,
            'processing_batches': len(self.processing_batches)
        }
        
        try:
            async with self.semaphore:
                self.processing_batches[batch_id] = {
                    "items": items,
                    "started_at": time.time(),
                    "status": "processing"
                }
                
                # 실제 배치 처리 (오버라이드 필요)
                results = await self._process_batch_items(items)
                
                # 결과 처리
                successful_count = sum(1 for r in results if r.get("success", False))
                failed_count = len(results) - successful_count
                
                self.stats["processed_items"] += successful_count
                self.stats["failed_items"] += failed_count
                self.stats["processed_batches"] += 1
                
                # 배치 완료
                self.processing_batches[batch_id]["status"] = "completed"
                self.processing_batches[batch_id]["completed_at"] = time.time()
                self.processing_batches[batch_id]["results"] = results
                
                self.logger.debug(f"✅ 배치 처리 완료: {batch_id} (성공: {successful_count}, 실패: {failed_count})")
                
        except Exception as e:
            # exceptions.py의 커스텀 예외로 변환
            from app.core.exceptions import (
                convert_to_mycloset_exception,
                PipelineError,
                ModelInferenceError
            )
            
            # 에러 타입별 커스텀 예외 변환
            if isinstance(e, (ValueError, TypeError)):
                custom_error = PipelineError(
                    f"배치 처리 중 데이터 오류: {e}",
                    "BATCH_PROCESSING_DATA_ERROR",
                    error_context
                )
            elif isinstance(e, (OSError, IOError)):
                custom_error = PipelineError(
                    f"배치 처리 중 시스템 오류: {e}",
                    "BATCH_PROCESSING_SYSTEM_ERROR",
                    error_context
                )
            else:
                custom_error = convert_to_mycloset_exception(e, error_context)
            
            self.logger.error(f"❌ 배치 처리 실패 {batch_id}: {custom_error}")
            
            self.stats["failed_batches"] += 1
            self.stats["failed_items"] += len(items)
            
            if batch_id in self.processing_batches:
                self.processing_batches[batch_id]["status"] = "failed"
                self.processing_batches[batch_id]["error"] = str(custom_error)
    
    async def _process_batch_items(self, items: List[Dict]) -> List[Dict]:
        """배치 아이템 처리 (오버라이드 필요)"""
        # 기본 구현: 모든 아이템을 성공으로 처리
        results = []
        for item in items:
            results.append({
                "item_id": item["id"],
                "success": True,
                "result": f"processed_{item['id']}",
                "processing_time": 0.1
            })
        
        # 처리 시뮬레이션
        await asyncio.sleep(0.1)
        
        return results
    
    async def _auto_flush(self):
        """자동 플러시"""
        try:
            while not self._shutdown:
                await asyncio.sleep(self.flush_interval)
                
                async with self._lock:
                    if self.pending_items:
                        await self._flush_batch()
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"❌ 자동 플러시 오류: {e}")
        finally:
            self._flush_task = None
    
    async def flush_all(self):
        """모든 대기 중인 아이템 플러시"""
        try:
            async with self._lock:
                while self.pending_items:
                    await self._flush_batch()
                    
        except Exception as e:
            self.logger.error(f"❌ 전체 플러시 실패: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 조회"""
        return {
            **self.stats,
            "pending_items": len(self.pending_items),
            "processing_batches": len(self.processing_batches),
            "batch_size": self.batch_size,
            "max_concurrent_batches": self.max_concurrent_batches
        }
    
    async def shutdown(self):
        """배치 처리기 종료"""
        try:
            self.logger.info("📦 배치 처리기 종료 시작")
            
            self._shutdown = True
            
            # 자동 플러시 중단
            if self._flush_task:
                self._flush_task.cancel()
                try:
                    await self._flush_task
                except asyncio.CancelledError:
                    pass
            
            # 남은 아이템 처리
            await self.flush_all()
            
            # 진행 중인 배치 완료 대기
            while self.processing_batches:
                await asyncio.sleep(0.1)
                # 완료된 배치 정리
                completed_batches = [
                    batch_id for batch_id, batch_info in self.processing_batches.items()
                    if batch_info["status"] in ["completed", "failed"]
                ]
                for batch_id in completed_batches:
                    del self.processing_batches[batch_id]
            
            self.logger.info("✅ 배치 처리기 종료 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 배치 처리기 종료 실패: {e}")

# ==============================================
# 🔥 5. 비동기 컨텍스트 관리자들
# ==============================================

@asynccontextmanager
async def async_timeout(timeout: float):
    """비동기 타임아웃 컨텍스트 매니저"""
    try:
        async with asyncio.timeout(timeout):
            yield
    except asyncio.TimeoutError:
        logger.warning(f"⏰ 비동기 작업 타임아웃: {timeout}초")
        raise

@asynccontextmanager
async def async_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """비동기 재시도 컨텍스트 매니저"""
    for attempt in range(max_retries + 1):
        try:
            yield attempt
            break
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"❌ 최대 재시도 초과: {max_retries}")
                raise
            
            wait_time = delay * (backoff ** attempt)
            logger.warning(f"⚠️ 재시도 {attempt + 1}/{max_retries + 1}: {wait_time:.1f}초 후 재시도")
            await asyncio.sleep(wait_time)

@asynccontextmanager
async def async_concurrency_limit(semaphore: asyncio.Semaphore):
    """비동기 동시성 제한 컨텍스트 매니저"""
    await semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()

# ==============================================
# 🔥 6. 비동기 데코레이터들
# ==============================================

def async_retry_decorator(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """비동기 재시도 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"❌ {func.__name__} 최대 재시도 초과: {max_retries}")
                        raise
                    
                    wait_time = delay * (backoff ** attempt)
                    logger.warning(f"⚠️ {func.__name__} 재시도 {attempt + 1}/{max_retries + 1}: {wait_time:.1f}초 후")
                    await asyncio.sleep(wait_time)
            
        return wrapper
    return decorator

def async_timeout_decorator(timeout: float):
    """비동기 타임아웃 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            except asyncio.TimeoutError:
                logger.error(f"⏰ {func.__name__} 타임아웃: {timeout}초")
                raise
        
        return wrapper
    return decorator

def async_rate_limit(calls_per_second: float):
    """비동기 속도 제한 데코레이터"""
    min_interval = 1.0 / calls_per_second
    last_called = {}
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            func_key = f"{func.__module__}.{func.__name__}"
            now = time.time()
            
            if func_key in last_called:
                elapsed = now - last_called[func_key]
                if elapsed < min_interval:
                    sleep_time = min_interval - elapsed
                    await asyncio.sleep(sleep_time)
            
            last_called[func_key] = time.time()
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

# ==============================================
# 🔥 7. 비동기 유틸리티 함수들
# ==============================================

async def gather_with_limit(
    *coros: Awaitable,
    limit: int = 10,
    return_exceptions: bool = False
) -> List[Any]:
    """제한된 동시성으로 여러 코루틴 실행"""
    # 에러 컨텍스트 준비
    error_context = {
        'coro_count': len(coros),
        'limit': limit,
        'return_exceptions': return_exceptions
    }
    
    try:
        semaphore = asyncio.Semaphore(limit)
        
        async def limited_coro(coro):
            async with semaphore:
                return await coro
        
        limited_coros = [limited_coro(coro) for coro in coros]
        return await asyncio.gather(*limited_coros, return_exceptions=return_exceptions)
        
    except Exception as e:
        # exceptions.py의 커스텀 예외로 변환
        from app.core.exceptions import (
            convert_to_mycloset_exception,
            PipelineError
        )
        
        custom_error = convert_to_mycloset_exception(e, error_context)
        logger.error(f"❌ gather_with_limit 실패: {custom_error}")
        raise custom_error

async def run_with_timeout(coro: Awaitable, timeout: float, default=None):
    """타임아웃과 함께 코루틴 실행"""
    # 에러 컨텍스트 준비
    error_context = {
        'timeout': timeout,
        'default_value': default,
        'coro_type': type(coro).__name__
    }
    
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        # 타임아웃은 정상적인 상황이므로 경고만 로깅
        logger.warning(f"⏰ 코루틴 타임아웃: {timeout}초")
        return default
    except Exception as e:
        # 기타 예외는 커스텀 예외로 변환
        from app.core.exceptions import (
            convert_to_mycloset_exception,
            PipelineError
        )
        
        custom_error = convert_to_mycloset_exception(e, error_context)
        logger.error(f"❌ run_with_timeout 실패: {custom_error}")
        raise custom_error

async def retry_async(
    coro_factory: Callable[[], Awaitable],
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple = (Exception,)
) -> Any:
    """비동기 함수 재시도"""
    # 에러 컨텍스트 준비
    error_context = {
        'max_retries': max_retries,
        'delay': delay,
        'backoff': backoff,
        'exceptions': [exc.__name__ for exc in exceptions]
    }
    
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except exceptions as e:
            if attempt == max_retries:
                # 최대 재시도 초과 시 커스텀 예외로 변환
                from app.core.exceptions import (
                    convert_to_mycloset_exception,
                    PipelineError
                )
                
                error_context['final_attempt'] = attempt
                error_context['final_error'] = str(e)
                
                custom_error = convert_to_mycloset_exception(e, error_context)
                logger.error(f"❌ retry_async 최대 재시도 초과: {custom_error}")
                raise custom_error
            
            wait_time = delay * (backoff ** attempt)
            logger.warning(f"⚠️ 재시도 {attempt + 1}/{max_retries + 1}: {wait_time:.1f}초 후")
            await asyncio.sleep(wait_time)

async def async_map(
    func: Callable,
    items: List[Any],
    max_concurrency: int = 10
) -> List[Any]:
    """비동기 맵 함수"""
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process_item(item):
        async with semaphore:
            if asyncio.iscoroutinefunction(func):
                return await func(item)
            else:
                return func(item)
    
    tasks = [process_item(item) for item in items]
    return await asyncio.gather(*tasks)

async def async_filter(
    predicate: Callable,
    items: List[Any],
    max_concurrency: int = 10
) -> List[Any]:
    """비동기 필터 함수"""
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def check_item(item):
        async with semaphore:
            if asyncio.iscoroutinefunction(predicate):
                return await predicate(item), item
            else:
                return predicate(item), item
    
    tasks = [check_item(item) for item in items]
    results = await asyncio.gather(*tasks)
    
    return [item for passed, item in results if passed]

# ==============================================
# 🔥 8. 비동기 작업 풀
# ==============================================

class AsyncWorkerPool:
    """비동기 작업자 풀"""
    
    def __init__(self, worker_count: int = 5, queue_size: int = 100):
        self.worker_count = worker_count
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.workers = []
        self.running = False
        self.logger = logging.getLogger(f"{__name__}.AsyncWorkerPool")
        
        # 통계
        self.stats = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "active_workers": 0
        }
    
    async def start(self):
        """작업자 풀 시작"""
        if self.running:
            return
        
        self.running = True
        self.workers = []
        
        for i in range(self.worker_count):
            worker = asyncio.create_task(self._worker(f"worker_{i}"))
            self.workers.append(worker)
        
        self.logger.info(f"🔄 작업자 풀 시작: {self.worker_count}개 작업자")
    
    async def _worker(self, worker_name: str):
        """작업자 루프"""
        self.stats["active_workers"] += 1
        
        try:
            while self.running:
                try:
                    # 작업 대기 (타임아웃 포함)
                    task_data = await asyncio.wait_for(
                        self.queue.get(), 
                        timeout=1.0
                    )
                    
                    # 작업 실행
                    await self._execute_task(worker_name, task_data)
                    
                except asyncio.TimeoutError:
                    # 타임아웃은 정상 (큐가 비어있음)
                    continue
                except Exception as e:
                    self.logger.error(f"❌ {worker_name} 오류: {e}")
                    self.stats["tasks_failed"] += 1
                    
        finally:
            self.stats["active_workers"] -= 1
            self.logger.debug(f"🔄 {worker_name} 종료")
    
    async def _execute_task(self, worker_name: str, task_data: Dict):
        """작업 실행"""
        try:
            func = task_data["func"]
            args = task_data.get("args", ())
            kwargs = task_data.get("kwargs", {})
            callback = task_data.get("callback")
            
            # 함수 실행
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # 콜백 실행
            if callback:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            
            self.stats["tasks_processed"] += 1
            self.logger.debug(f"✅ {worker_name} 작업 완료")
            
        except Exception as e:
            self.logger.error(f"❌ {worker_name} 작업 실행 실패: {e}")
            self.stats["tasks_failed"] += 1
            
            # 에러 콜백
            error_callback = task_data.get("error_callback")
            if error_callback:
                try:
                    if asyncio.iscoroutinefunction(error_callback):
                        await error_callback(e)
                    else:
                        error_callback(e)
                except Exception:
                    pass
    
    async def submit(
        self,
        func: Callable,
        *args,
        callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None,
        **kwargs
    ):
        """작업 제출"""
        task_data = {
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "callback": callback,
            "error_callback": error_callback
        }
        
        await self.queue.put(task_data)
    
    def submit_nowait(
        self,
        func: Callable,
        *args,
        callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None,
        **kwargs
    ):
        """작업 즉시 제출 (블로킹 없음)"""
        task_data = {
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "callback": callback,
            "error_callback": error_callback
        }
        
        try:
            self.queue.put_nowait(task_data)
            return True
        except asyncio.QueueFull:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 조회"""
        return {
            **self.stats,
            "queue_size": self.queue.qsize(),
            "worker_count": self.worker_count,
            "running": self.running
        }
    
    async def stop(self):
        """작업자 풀 중지"""
        self.running = False
        
        # 모든 작업자 완료 대기
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        self.logger.info("✅ 작업자 풀 중지 완료")

# ==============================================
# 🔥 9. 전역 비동기 관리자
# ==============================================

class GlobalAsyncManager:
    """전역 비동기 관리자"""
    
    def __init__(self):
        self.scheduler = None
        self.worker_pool = None
        self.batch_processor = None
        self.logger = logging.getLogger(f"{__name__}.GlobalAsyncManager")
        
        # 설정
        self.concurrency_config = ConcurrencyConfig()
        
        # 상태
        self.initialized = False
    
    async def initialize(
        self,
        concurrency_config: Optional[ConcurrencyConfig] = None,
        worker_count: int = 5,
        batch_size: int = 10
    ):
        """비동기 관리자 초기화"""
        try:
            if self.initialized:
                return
            
            self.concurrency_config = concurrency_config or ConcurrencyConfig()
            
            # 스케줄러 초기화
            self.scheduler = AsyncTaskScheduler(self.concurrency_config)
            
            # 작업자 풀 초기화
            self.worker_pool = AsyncWorkerPool(worker_count)
            await self.worker_pool.start()
            
            # 배치 처리기 초기화
            self.batch_processor = AsyncBatchProcessor(batch_size)
            
            self.initialized = True
            self.logger.info("🔄 전역 비동기 관리자 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 비동기 관리자 초기화 실패: {e}")
            raise
    
    async def shutdown(self):
        """비동기 관리자 종료"""
        try:
            if not self.initialized:
                return
            
            # 각 컴포넌트 종료
            if self.scheduler:
                await self.scheduler.shutdown()
            
            if self.worker_pool:
                await self.worker_pool.stop()
            
            if self.batch_processor:
                await self.batch_processor.shutdown()
            
            self.initialized = False
            self.logger.info("✅ 전역 비동기 관리자 종료 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 비동기 관리자 종료 실패: {e}")

# 전역 인스턴스
_global_async_manager: Optional[GlobalAsyncManager] = None
_manager_lock = threading.Lock()

def get_global_async_manager() -> GlobalAsyncManager:
    """전역 비동기 관리자 인스턴스 반환"""
    global _global_async_manager
    
    with _manager_lock:
        if _global_async_manager is None:
            _global_async_manager = GlobalAsyncManager()
        
        return _global_async_manager

async def initialize_async_system(
    concurrency_config: Optional[ConcurrencyConfig] = None,
    worker_count: int = 5,
    batch_size: int = 10
):
    """비동기 시스템 초기화"""
    manager = get_global_async_manager()
    await manager.initialize(concurrency_config, worker_count, batch_size)

async def shutdown_async_system():
    """비동기 시스템 종료"""
    manager = get_global_async_manager()
    await manager.shutdown()

# ==============================================
# 🔥 10. 모듈 내보내기
# ==============================================

__all__ = [
    # 핵심 클래스들
    'AsyncTaskScheduler',
    'AsyncBatchProcessor',
    'AsyncWorkerPool',
    'GlobalAsyncManager',
    
    # 설정 클래스들
    'AsyncTask',
    'ConcurrencyConfig',
    'TaskPriority',
    'TaskStatus',
    'ConcurrencyStrategy',
    
    # 컨텍스트 매니저들
    'async_timeout',
    'async_retry',
    'async_concurrency_limit',
    
    # 데코레이터들
    'async_retry_decorator',
    'async_timeout_decorator',
    'async_rate_limit',
    
    # 유틸리티 함수들
    'gather_with_limit',
    'run_with_timeout',
    'retry_async',
    'async_map',
    'async_filter',
    
    # 전역 함수들
    'get_global_async_manager',
    'initialize_async_system',
    'shutdown_async_system'
]

logger.info("✅ 비동기 유틸리티 시스템 v1.0 로드 완료")
logger.info("🔄 비동기 작업 관리 및 스케줄링")
logger.info("⚡ 동시성 제어 및 제한")
logger.info("📦 작업 큐 및 배치 처리")
logger.info("🎯 비동기 컨텍스트 관리")
logger.info("🔗 순환참조 방지 - 독립적 모듈")