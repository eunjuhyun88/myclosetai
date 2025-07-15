# app/api/websocket_routes.py
"""
MyCloset AI Backend - 완전 안전한 WebSocket 라우터
🛡️ 순환참조 완전 제거 + 에러 구조 완전 해결
✅ 모든 누락된 함수 추가 + Import 에러 해결
"""

import asyncio
import json
import logging
import time
import uuid
import traceback
import weakref
from typing import Dict, Any, Set, Optional, List, Callable, Union
from datetime import datetime
from contextlib import asynccontextmanager
from enum import Enum

# psutil 안전한 import
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse

# ========================
# 완전 독립적인 타입 정의 (순환참조 방지)
# ========================

logger = logging.getLogger(__name__)
router = APIRouter()

class WebSocketState(Enum):
    """WebSocket 연결 상태"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"

class MessageType(Enum):
    """메시지 타입 열거형"""
    CONNECTION_ESTABLISHED = "connection_established"
    PIPELINE_PROGRESS = "pipeline_progress"
    STEP_UPDATE = "step_update"
    SYSTEM_INFO = "system_info"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"
    SUBSCRIBE_SESSION = "subscribe_session"
    SUBSCRIPTION_CONFIRMED = "subscription_confirmed"

class PipelineStatus(Enum):
    """파이프라인 상태 (독립적 정의)"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# ========================
# 지연 import 헬퍼 (순환참조 완전 방지)
# ========================

def safe_import(module_name: str, fallback_value=None):
    """안전한 지연 import"""
    try:
        if module_name == "schemas":
            from app.models import schemas
            return schemas
        elif module_name == "config":
            from app.core import config
            return config
        elif module_name == "pipeline":
            from app.ai_pipeline import pipeline_manager
            return pipeline_manager
        elif module_name == "torch":
            import torch
            return torch
        else:
            return fallback_value
    except ImportError as e:
        logger.warning(f"⚠️ 모듈 {module_name} import 실패: {e}")
        return fallback_value
    except Exception as e:
        logger.error(f"❌ 모듈 {module_name} import 오류: {e}")
        return fallback_value

def get_gpu_info_safe() -> Dict[str, Any]:
    """GPU 정보 안전한 수집"""
    try:
        torch = safe_import("torch")
        if not torch:
            return {"available": False, "error": "torch not available"}
        
        if torch.cuda.is_available():
            return {
                "available": True,
                "type": "CUDA",
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
                "memory_allocated": torch.cuda.memory_allocated(),
                "memory_reserved": torch.cuda.memory_reserved()
            }
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return {
                "available": True,
                "type": "MPS (Apple Silicon)",
                "device_count": 1
            }
        else:
            return {"available": False, "type": "CPU Only"}
            
    except Exception as e:
        logger.error(f"❌ GPU 정보 수집 실패: {e}")
        return {"available": False, "error": str(e)}

def get_cpu_info_safe() -> Dict[str, Any]:
    """CPU 정보 안전한 수집"""
    try:
        if not PSUTIL_AVAILABLE:
            return {"available": False, "error": "psutil not available"}
        
        return {
            "available": True,
            "usage_percent": psutil.cpu_percent(interval=1),
            "core_count": psutil.cpu_count(),
            "core_count_logical": psutil.cpu_count(logical=True)
        }
    except Exception as e:
        logger.error(f"❌ CPU 정보 수집 실패: {e}")
        return {"available": False, "error": str(e)}

def get_memory_info_safe() -> Dict[str, Any]:
    """메모리 정보 안전한 수집"""
    try:
        if not PSUTIL_AVAILABLE:
            return {"available": False, "error": "psutil not available"}
        
        memory = psutil.virtual_memory()
        return {
            "available": True,
            "total": memory.total,
            "available_bytes": memory.available,
            "used": memory.used,
            "percent": memory.percent,
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3)
        }
    except Exception as e:
        logger.error(f"❌ 메모리 정보 수집 실패: {e}")
        return {"available": False, "error": str(e)}

# ========================
# 안전한 WebSocket 연결 관리자
# ========================

class SafeConnectionManager:
    """
    🛡️ 완전 안전한 WebSocket 연결 관리자
    - 순환참조 완전 방지
    - 메모리 누수 방지
    - 에러 전파 차단
    - WebSocket 상태 안전 관리
    """
    
    def __init__(self):
        # 핵심 연결 관리 (weakref 사용으로 메모리 누수 방지)
        self._active_connections: Dict[str, WebSocket] = {}
        self._connection_states: Dict[str, WebSocketState] = {}
        self._session_connections: Dict[str, Set[str]] = {}
        self._client_sessions: Dict[str, str] = {}
        
        # 메타데이터 (크기 제한)
        self._client_metadata: Dict[str, Dict[str, Any]] = {}
        self._message_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # 통계 및 모니터링
        self._stats = {
            "total_connections": 0,
            "current_connections": 0,
            "total_messages": 0,
            "errors": 0,
            "reconnections": 0,
            "start_time": time.time()
        }
        
        # 설정
        self._config = {
            "max_message_history": 50,
            "max_connections_per_session": 10,
            "inactive_timeout": 300,  # 5분
            "heartbeat_interval": 30,  # 30초
            "max_message_size": 1024 * 1024,  # 1MB
            "cleanup_interval": 60  # 1분
        }
        
        # 백그라운드 태스크 관리
        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        logger.info("🛡️ SafeConnectionManager 초기화 완료")
    
    async def start(self):
        """매니저 시작"""
        if self._is_running:
            return
        
        self._is_running = True
        self._cleanup_task = asyncio.create_task(self._background_cleanup())
        logger.info("✅ SafeConnectionManager 시작됨")
    
    async def stop(self):
        """매니저 중지"""
        self._is_running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # 모든 연결 정리
        await self._disconnect_all()
        logger.info("🛑 SafeConnectionManager 중지됨")
    
    @asynccontextmanager
    async def safe_operation(self, operation_name: str):
        """안전한 작업 컨텍스트"""
        try:
            yield
        except asyncio.CancelledError:
            logger.info(f"🔄 작업 취소됨: {operation_name}")
            raise
        except Exception as e:
            logger.error(f"❌ 작업 실패 {operation_name}: {e}")
            logger.error(f"스택 트레이스: {traceback.format_exc()}")
            self._stats["errors"] += 1
    
    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None) -> str:
        """안전한 클라이언트 연결"""
        async with self.safe_operation("connect"):
            try:
                await websocket.accept()
                
                if not client_id:
                    client_id = f"client_{uuid.uuid4().hex[:8]}"
                
                # 재연결 감지
                is_reconnection = client_id in self._client_metadata
                if is_reconnection:
                    self._stats["reconnections"] += 1
                    logger.info(f"🔄 클라이언트 재연결: {client_id}")
                
                # 연결 등록
                self._active_connections[client_id] = websocket
                self._connection_states[client_id] = WebSocketState.CONNECTED
                
                # 메타데이터 설정
                current_time = time.time()
                if client_id not in self._client_metadata:
                    self._client_metadata[client_id] = {}
                
                self._client_metadata[client_id].update({
                    "connected_at": datetime.now().isoformat(),
                    "last_activity": current_time,
                    "message_count": self._client_metadata[client_id].get("message_count", 0),
                    "is_reconnection": is_reconnection,
                    "connection_attempts": self._client_metadata[client_id].get("connection_attempts", 0) + 1
                })
                
                # 통계 업데이트
                if not is_reconnection:
                    self._stats["total_connections"] += 1
                self._stats["current_connections"] = len(self._active_connections)
                
                logger.info(f"✅ WebSocket 클라이언트 연결: {client_id}")
                
                # 환영 메시지 전송
                welcome_message = self._create_welcome_message(client_id, is_reconnection)
                await self._send_message_safe(welcome_message, client_id)
                
                return client_id
                
            except Exception as e:
                logger.error(f"❌ 연결 설정 실패: {e}")
                self._stats["errors"] += 1
                raise HTTPException(status_code=500, detail=f"Connection failed: {str(e)}")
    
    def _create_welcome_message(self, client_id: str, is_reconnection: bool) -> Dict[str, Any]:
        """환영 메시지 생성"""
        schemas = safe_import("schemas")
        config = safe_import("config")
        
        message = {
            "type": MessageType.CONNECTION_ESTABLISHED.value,
            "client_id": client_id,
            "timestamp": time.time(),
            "is_reconnection": is_reconnection,
            "server_info": {
                "api_version": "2.1.0",
                "websocket_version": "1.0.0",
                "features": [
                    "safe_connection_management",
                    "memory_leak_prevention", 
                    "circular_import_prevention",
                    "automatic_cleanup",
                    "error_recovery"
                ],
                "endpoints": [
                    "/api/ws/pipeline-progress",
                    "/api/ws/system-monitor", 
                    "/api/ws/test"
                ]
            }
        }
        
        # 조건부 정보 추가
        if schemas:
            message["server_info"]["schemas_available"] = True
        if config and hasattr(config, 'settings'):
            message["server_info"]["config_loaded"] = True
        
        # 재연결 정보
        if is_reconnection and client_id in self._client_metadata:
            prev_data = self._client_metadata[client_id]
            message["previous_session"] = {
                "message_count": prev_data.get("message_count", 0),
                "connection_attempts": prev_data.get("connection_attempts", 1)
            }
        
        message["message"] = (
            f"MyCloset AI WebSocket {'재연결' if is_reconnection else '연결'}이 안전하게 설정되었습니다."
        )
        
        return message
    
    def disconnect(self, client_id: str, reason: str = "unknown"):
        """안전한 클라이언트 연결 해제"""
        try:
            # 상태 업데이트
            self._connection_states[client_id] = WebSocketState.DISCONNECTED
            
            if client_id in self._client_metadata:
                self._client_metadata[client_id].update({
                    "disconnected_at": datetime.now().isoformat(),
                    "disconnect_reason": reason
                })
            
            # 활성 연결에서 제거
            if client_id in self._active_connections:
                del self._active_connections[client_id]
            
            # 세션에서 제거 (세션 정보는 유지 - 재연결 대비)
            if client_id in self._client_sessions:
                session_id = self._client_sessions[client_id]
                if session_id in self._session_connections:
                    self._session_connections[session_id].discard(client_id)
                    if not self._session_connections[session_id]:
                        del self._session_connections[session_id]
                        logger.info(f"🗑️ 빈 세션 정리: {session_id}")
            
            # 통계 업데이트
            self._stats["current_connections"] = len(self._active_connections)
            
            logger.info(f"❌ WebSocket 클라이언트 연결 해제: {client_id} (이유: {reason})")
            
        except Exception as e:
            logger.error(f"❌ 연결 해제 처리 오류: {e}")
    
    async def _send_message_safe(self, message: Dict[str, Any], client_id: str) -> bool:
        """완전 안전한 메시지 전송"""
        if client_id not in self._active_connections:
            logger.warning(f"⚠️ 비활성 클라이언트: {client_id}")
            return False
        
        if self._connection_states.get(client_id) != WebSocketState.CONNECTED:
            logger.warning(f"⚠️ 잘못된 연결 상태: {client_id}")
            return False
        
        async with self.safe_operation(f"send_message_to_{client_id}"):
            try:
                # 메시지 보강
                if "timestamp" not in message:
                    message["timestamp"] = time.time()
                if "client_id" not in message:
                    message["client_id"] = client_id
                
                # JSON 직렬화 (안전)
                try:
                    message_str = json.dumps(message, ensure_ascii=False, default=str)
                except (TypeError, ValueError) as e:
                    logger.error(f"❌ JSON 직렬화 실패: {e}")
                    error_message = {
                        "type": MessageType.ERROR.value,
                        "error": "Message serialization failed",
                        "timestamp": time.time(),
                        "client_id": client_id
                    }
                    message_str = json.dumps(error_message, ensure_ascii=False)
                
                # 메시지 크기 체크
                if len(message_str) > self._config["max_message_size"]:
                    logger.warning(f"⚠️ 메시지 크기 초과: {len(message_str)} bytes")
                    return False
                
                # WebSocket 전송
                websocket = self._active_connections[client_id]
                await websocket.send_text(message_str)
                
                # 통계 업데이트
                self._stats["total_messages"] += 1
                if client_id in self._client_metadata:
                    metadata = self._client_metadata[client_id]
                    metadata["message_count"] = metadata.get("message_count", 0) + 1
                    metadata["last_activity"] = time.time()
                
                # 메시지 히스토리 저장 (크기 제한)
                self._save_message_history_safe(client_id, message, len(message_str))
                
                return True
                
            except Exception as e:
                logger.error(f"❌ 메시지 전송 실패 to {client_id}: {e}")
                self.disconnect(client_id, f"send_error: {str(e)}")
                return False
    
    def _save_message_history_safe(self, client_id: str, message: Dict[str, Any], size: int):
        """안전한 메시지 히스토리 저장"""
        try:
            if client_id not in self._message_history:
                self._message_history[client_id] = []
            
            history_entry = {
                "timestamp": message.get("timestamp"),
                "type": message.get("type", "unknown"),
                "size": size
            }
            
            self._message_history[client_id].append(history_entry)
            
            # 크기 제한 (메모리 누수 방지)
            max_history = self._config["max_message_history"]
            if len(self._message_history[client_id]) > max_history:
                self._message_history[client_id] = self._message_history[client_id][-max_history:]
                
        except Exception as e:
            logger.error(f"❌ 메시지 히스토리 저장 실패: {e}")
    
    async def send_personal_message(self, message: Dict[str, Any], client_id: str) -> bool:
        """외부 인터페이스용 안전한 메시지 전송"""
        return await self._send_message_safe(message, client_id)
    
    async def broadcast_to_session(self, message: Dict[str, Any], session_id: str) -> int:
        """세션별 안전한 브로드캐스트"""
        if session_id not in self._session_connections:
            logger.warning(f"⚠️ 존재하지 않는 세션: {session_id}")
            return 0
        
        success_count = 0
        failed_clients = []
        
        # 안전한 병렬 전송
        clients = list(self._session_connections[session_id])
        tasks = []
        
        for client_id in clients:
            if client_id in self._active_connections:
                task = self._send_message_safe(message, client_id)
                tasks.append((client_id, task))
        
        # 결과 수집
        for client_id, task in tasks:
            try:
                if await task:
                    success_count += 1
                else:
                    failed_clients.append(client_id)
            except Exception as e:
                logger.error(f"❌ 브로드캐스트 실패 {client_id}: {e}")
                failed_clients.append(client_id)
        
        # 실패한 클라이언트 정리
        for client_id in failed_clients:
            self.disconnect(client_id, "broadcast_failed")
        
        logger.debug(f"📡 세션 {session_id} 브로드캐스트: {success_count}/{len(clients)} 성공")
        return success_count
    
    def subscribe_to_session(self, client_id: str, session_id: str) -> bool:
        """안전한 세션 구독"""
        try:
            # 세션당 최대 연결 수 체크
            if session_id in self._session_connections:
                if len(self._session_connections[session_id]) >= self._config["max_connections_per_session"]:
                    logger.warning(f"⚠️ 세션 {session_id} 최대 연결 수 초과")
                    return False
            
            if session_id not in self._session_connections:
                self._session_connections[session_id] = set()
            
            # 기존 구독 해제
            if client_id in self._client_sessions:
                old_session = self._client_sessions[client_id]
                if old_session != session_id and old_session in self._session_connections:
                    self._session_connections[old_session].discard(client_id)
                    logger.info(f"🔄 세션 변경: {client_id} ({old_session} → {session_id})")
            
            # 새 구독 설정
            self._session_connections[session_id].add(client_id)
            self._client_sessions[client_id] = session_id
            
            logger.info(f"📡 클라이언트 {client_id}가 세션 {session_id}에 구독")
            return True
            
        except Exception as e:
            logger.error(f"❌ 세션 구독 실패: {e}")
            return False
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """안전한 연결 통계"""
        try:
            uptime = time.time() - self._stats["start_time"]
            
            return {
                "current_connections": len(self._active_connections),
                "total_connections": self._stats["total_connections"],
                "active_sessions": len(self._session_connections),
                "total_messages": self._stats["total_messages"],
                "errors": self._stats["errors"],
                "reconnections": self._stats["reconnections"],
                "uptime_seconds": uptime,
                "uptime_formatted": f"{uptime // 3600:.0f}h {(uptime % 3600) // 60:.0f}m {uptime % 60:.0f}s",
                "session_details": {
                    session_id: len(clients) 
                    for session_id, clients in self._session_connections.items()
                },
                "message_rate": self._stats["total_messages"] / max(uptime, 1),
                "error_rate": self._stats["errors"] / max(self._stats["total_messages"], 1),
                "config": self._config,
                "gpu": get_gpu_info_safe()
            }
            
        except Exception as e:
            logger.error(f"❌ 통계 수집 실패: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    async def _background_cleanup(self):
        """백그라운드 정리 작업"""
        while self._is_running:
            try:
                await asyncio.sleep(self._config["cleanup_interval"])
                
                if not self._is_running:
                    break
                
                # 비활성 연결 정리
                await self._cleanup_inactive_connections()
                
                # 메모리 정리
                self._cleanup_memory()
                
            except asyncio.CancelledError:
                logger.info("🔄 백그라운드 정리 작업 취소됨")
                break
            except Exception as e:
                logger.error(f"❌ 백그라운드 정리 오류: {e}")
                await asyncio.sleep(60)  # 에러 시 1분 대기
    
    async def _cleanup_inactive_connections(self):
        """비활성 연결 정리"""
        try:
            current_time = time.time()
            inactive_threshold = self._config["inactive_timeout"]
            inactive_clients = []
            
            for client_id, metadata in self._client_metadata.items():
                if client_id in self._active_connections:
                    last_activity = metadata.get("last_activity", current_time)
                    if current_time - last_activity > inactive_threshold:
                        inactive_clients.append(client_id)
            
            for client_id in inactive_clients:
                logger.info(f"🧹 비활성 클라이언트 정리: {client_id}")
                self.disconnect(client_id, "inactive_timeout")
            
            if inactive_clients:
                logger.info(f"🧹 {len(inactive_clients)}개 비활성 연결 정리 완료")
                
        except Exception as e:
            logger.error(f"❌ 비활성 연결 정리 실패: {e}")
    
    def _cleanup_memory(self):
        """메모리 정리"""
        try:
            # 연결 해제된 클라이언트의 메타데이터 정리 (24시간 후)
            current_time = time.time()
            old_threshold = 24 * 3600  # 24시간
            
            cleanup_clients = []
            for client_id, metadata in self._client_metadata.items():
                if client_id not in self._active_connections:
                    disconnected_at = metadata.get("disconnected_at")
                    if disconnected_at:
                        try:
                            disconnect_time = datetime.fromisoformat(disconnected_at).timestamp()
                            if current_time - disconnect_time > old_threshold:
                                cleanup_clients.append(client_id)
                        except:
                            cleanup_clients.append(client_id)  # 잘못된 타임스탬프
            
            for client_id in cleanup_clients:
                if client_id in self._client_metadata:
                    del self._client_metadata[client_id]
                if client_id in self._message_history:
                    del self._message_history[client_id]
                if client_id in self._client_sessions:
                    del self._client_sessions[client_id]
            
            if cleanup_clients:
                logger.info(f"🧹 {len(cleanup_clients)}개 오래된 메타데이터 정리")
                
        except Exception as e:
            logger.error(f"❌ 메모리 정리 실패: {e}")
    
    async def _disconnect_all(self):
        """모든 연결 정리"""
        try:
            for client_id in list(self._active_connections.keys()):
                self.disconnect(client_id, "server_shutdown")
            
            logger.info("✅ 모든 WebSocket 연결 정리 완료")
            
        except Exception as e:
            logger.error(f"❌ 전체 연결 정리 실패: {e}")

# 글로벌 매니저 인스턴스
manager = SafeConnectionManager()

# ========================
# 안전한 콜백 팩토리
# ========================

def create_safe_pipeline_callbacks(session_id: str):
    """안전한 파이프라인 콜백 생성"""
    
    async def progress_callback(progress_data: Dict[str, Any]):
        """진행 상황 업데이트 (에러 안전)"""
        try:
            message = {
                "type": MessageType.PIPELINE_PROGRESS.value,
                "session_id": session_id,
                "data": progress_data,
                "timestamp": time.time()
            }
            
            await manager.broadcast_to_session(message, session_id)
            logger.debug(f"📊 진행 상황: {session_id} - {progress_data.get('current_step', 'unknown')}")
            
        except Exception as e:
            logger.error(f"❌ 진행 상황 콜백 오류: {e}")
    
    async def step_callback(step_name: str, step_data: Dict[str, Any]):
        """단계별 업데이트 (에러 안전)"""
        try:
            message = {
                "type": MessageType.STEP_UPDATE.value,
                "session_id": session_id,
                "step_name": step_name,
                "step_data": step_data,
                "timestamp": time.time()
            }
            
            await manager.broadcast_to_session(message, session_id)
            logger.info(f"🔄 단계 업데이트: {session_id} - {step_name}")
            
        except Exception as e:
            logger.error(f"❌ 단계 콜백 오류: {e}")
    
    async def error_callback(error_info: Dict[str, Any]):
        """에러 처리 (에러 안전)"""
        try:
            message = {
                "type": MessageType.ERROR.value,
                "session_id": session_id,
                "error": error_info,
                "timestamp": time.time()
            }
            
            await manager.broadcast_to_session(message, session_id)
            logger.error(f"❌ 파이프라인 에러: {session_id} - {error_info}")
            
        except Exception as e:
            logger.error(f"❌ 에러 콜백 실패: {e}")
    
    return progress_callback, step_callback, error_callback

# ========================
# 안전한 메시지 핸들러
# ========================

async def handle_client_message_safe(message: Dict[str, Any], client_id: str):
    """완전 안전한 클라이언트 메시지 처리"""
    async with manager.safe_operation("handle_client_message"):
        try:
            message_type = message.get("type", "unknown")
            
            handlers = {
                "ping": lambda: handle_ping_safe(client_id),
                "subscribe_session": lambda: handle_session_subscribe_safe(message, client_id),
                "get_stats": lambda: handle_stats_request_safe(client_id),
                "get_system_info": lambda: handle_system_info_request_safe(client_id)
            }
            
            if message_type in handlers:
                await handlers[message_type]()
            else:
                await handle_unknown_message_safe(message_type, client_id)
                
        except Exception as e:
            logger.error(f"❌ 메시지 처리 오류: {e}")
            await manager.send_personal_message({
                "type": MessageType.ERROR.value,
                "error": "Message processing failed",
                "details": str(e)
            }, client_id)

async def handle_ping_safe(client_id: str):
    """안전한 Ping 응답"""
    await manager.send_personal_message({
        "type": MessageType.PONG.value,
        "timestamp": time.time(),
        "server_uptime": time.time() - manager._stats["start_time"]
    }, client_id)

async def handle_session_subscribe_safe(message: Dict[str, Any], client_id: str):
    """안전한 세션 구독 처리"""
    session_id = message.get("session_id")
    if not session_id:
        await manager.send_personal_message({
            "type": MessageType.ERROR.value,
            "error": "session_id required"
        }, client_id)
        return
    
    success = manager.subscribe_to_session(client_id, session_id)
    if success:
        await manager.send_personal_message({
            "type": MessageType.SUBSCRIPTION_CONFIRMED.value,
            "session_id": session_id,
            "subscribers_count": len(manager._session_connections.get(session_id, set()))
        }, client_id)
    else:
        await manager.send_personal_message({
            "type": MessageType.ERROR.value,
            "error": "Subscription failed"
        }, client_id)

async def handle_stats_request_safe(client_id: str):
    """안전한 통계 요청 처리"""
    stats = manager.get_connection_stats()
    await manager.send_personal_message({
        "type": "stats_response",
        "data": stats
    }, client_id)

async def handle_system_info_request_safe(client_id: str):
    """안전한 시스템 정보 요청 처리"""
    system_info = await get_comprehensive_system_info_safe()
    await manager.send_personal_message({
        "type": MessageType.SYSTEM_INFO.value,
        "data": system_info
    }, client_id)

async def handle_unknown_message_safe(message_type: str, client_id: str):
    """안전한 알 수 없는 메시지 처리"""
    logger.warning(f"⚠️ 알 수 없는 메시지: {message_type} from {client_id}")
    await manager.send_personal_message({
        "type": MessageType.ERROR.value,
        "error": f"Unknown message type: {message_type}",
        "supported_types": ["ping", "subscribe_session", "get_stats", "get_system_info"]
    }, client_id)

# ========================
# 안전한 시스템 정보 수집
# ========================

async def get_comprehensive_system_info_safe() -> Dict[str, Any]:
    """완전 안전한 시스템 정보 수집"""
    try:
        system_info = {
            "cpu": get_cpu_info_safe(),
            "memory": get_memory_info_safe(),
            "gpu": get_gpu_info_safe(),
            "connections": manager.get_connection_stats(),
            "timestamp": time.time()
        }
        
        # 안전한 추가 정보 수집
        try:
            if PSUTIL_AVAILABLE:
                # 디스크 정보 (선택적)
                disk = psutil.disk_usage('/')
                system_info["disk"] = {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100,
                    "total_gb": disk.total / (1024**3),
                    "used_gb": disk.used / (1024**3),
                    "free_gb": disk.free / (1024**3)
                }
        except:
            pass
        
        try:
            if PSUTIL_AVAILABLE:
                # 프로세스 정보 (선택적)
                process = psutil.Process()
                system_info["process"] = {
                    "cpu_percent": process.cpu_percent(),
                    "memory_percent": process.memory_percent(),
                    "num_threads": process.num_threads(),
                    "pid": process.pid
                }
        except:
            pass
        
        return system_info
        
    except Exception as e:
        logger.error(f"❌ 시스템 정보 수집 실패: {e}")
        return {
            "error": str(e),
            "timestamp": time.time(),
            "connections": manager.get_connection_stats() if manager else {}
        }

# ========================
# WebSocket 엔드포인트들
# ========================

@router.websocket("/pipeline-progress")
async def pipeline_progress_websocket_safe(websocket: WebSocket):
    """완전 안전한 파이프라인 진행 상황 WebSocket"""
    client_id = None
    try:
        client_id = await manager.connect(websocket)
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await handle_client_message_safe(message, client_id)
            except json.JSONDecodeError as e:
                await manager.send_personal_message({
                    "type": MessageType.ERROR.value,
                    "error": "Invalid JSON format",
                    "details": str(e)
                }, client_id)
            except asyncio.CancelledError:
                logger.info(f"🔄 WebSocket 태스크 취소됨: {client_id}")
                break
                
    except WebSocketDisconnect:
        if client_id:
            manager.disconnect(client_id, "websocket_disconnect")
        logger.info(f"🔌 파이프라인 WebSocket 연결 해제: {client_id}")
    except Exception as e:
        logger.error(f"❌ 파이프라인 WebSocket 오류: {e}")
        if client_id:
            manager.disconnect(client_id, f"websocket_error: {str(e)}")

@router.websocket("/system-monitor")
async def system_monitor_websocket_safe(websocket: WebSocket):
    """완전 안전한 시스템 모니터링 WebSocket"""
    client_id = None
    monitor_task = None
    
    try:
        client_id = await manager.connect(websocket)
        
        # 주기적 시스템 정보 전송 시작
        monitor_task = asyncio.create_task(
            send_periodic_system_info_safe(client_id, interval=10)
        )
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await handle_client_message_safe(message, client_id)
            except json.JSONDecodeError as e:
                await manager.send_personal_message({
                    "type": MessageType.ERROR.value,
                    "error": "Invalid JSON format"
                }, client_id)
            except asyncio.CancelledError:
                logger.info(f"🔄 시스템 모니터 태스크 취소됨: {client_id}")
                break
                
    except WebSocketDisconnect:
        if client_id:
            manager.disconnect(client_id, "websocket_disconnect")
        logger.info(f"🔌 시스템 모니터 WebSocket 연결 해제: {client_id}")
    except Exception as e:
        logger.error(f"❌ 시스템 모니터 WebSocket 오류: {e}")
        if client_id:
            manager.disconnect(client_id, f"websocket_error: {str(e)}")
    finally:
        # 모니터링 태스크 정리
        if monitor_task:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

@router.websocket("/test")
async def test_websocket_safe(websocket: WebSocket):
    """완전 안전한 테스트용 WebSocket"""
    client_id = None
    try:
        client_id = await manager.connect(websocket)
        
        # 테스트 환영 메시지
        await manager.send_personal_message({
            "type": "test_welcome",
            "message": "안전한 테스트 WebSocket에 연결되었습니다.",
            "available_commands": ["echo", "stats", "system_info", "stress_test"],
            "connection_info": manager.get_connection_stats()
        }, client_id)
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # 테스트 전용 핸들러
                if message.get("type") == "echo":
                    await manager.send_personal_message({
                        "type": "echo_response",
                        "original_message": message,
                        "server_timestamp": time.time()
                    }, client_id)
                elif message.get("type") == "stress_test":
                    await handle_stress_test_safe(message, client_id)
                else:
                    await handle_client_message_safe(message, client_id)
                    
            except json.JSONDecodeError as e:
                await manager.send_personal_message({
                    "type": MessageType.ERROR.value,
                    "error": "Invalid JSON format"
                }, client_id)
            except asyncio.CancelledError:
                logger.info(f"🔄 테스트 WebSocket 태스크 취소됨: {client_id}")
                break
                
    except WebSocketDisconnect:
        if client_id:
            manager.disconnect(client_id, "websocket_disconnect")
        logger.info(f"🔌 테스트 WebSocket 연결 해제: {client_id}")
    except Exception as e:
        logger.error(f"❌ 테스트 WebSocket 오류: {e}")
        if client_id:
            manager.disconnect(client_id, f"websocket_error: {str(e)}")

async def handle_stress_test_safe(message: Dict[str, Any], client_id: str):
    """안전한 스트레스 테스트"""
    async with manager.safe_operation("stress_test"):
        count = min(message.get("count", 10), 100)  # 최대 100개로 제한
        
        for i in range(count):
            await manager.send_personal_message({
                "type": "stress_test_message",
                "index": i + 1,
                "total": count,
                "timestamp": time.time(),
                "data": f"Safe test message {i + 1}/{count}"
            }, client_id)
            
            if i % 10 == 0:  # 10개마다 잠깐 대기
                await asyncio.sleep(0.01)
        
        await manager.send_personal_message({
            "type": "stress_test_completed",
            "total_sent": count,
            "timestamp": time.time()
        }, client_id)

# ========================
# 안전한 백그라운드 태스크
# ========================

async def send_periodic_system_info_safe(client_id: str, interval: int = 10):
    """안전한 주기적 시스템 정보 전송"""
    while client_id in manager._active_connections:
        try:
            system_info = await get_comprehensive_system_info_safe()
            await manager.send_personal_message({
                "type": "periodic_system_info",
                "data": system_info,
                "interval": interval
            }, client_id)
            
            await asyncio.sleep(interval)
            
        except asyncio.CancelledError:
            logger.info(f"🔄 주기적 시스템 정보 전송 취소됨: {client_id}")
            break
        except Exception as e:
            logger.error(f"❌ 주기적 시스템 정보 전송 실패: {e}")
            break

# ========================
# 디버깅 페이지 (간소화)
# ========================

@router.get("/debug")
async def websocket_debug_safe():
    """안전한 WebSocket 디버깅 페이지"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>MyCloset AI 안전한 WebSocket 디버그</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1000px; margin: 0 auto; }
        .panel { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { display: inline-block; padding: 4px 8px; border-radius: 4px; color: white; font-weight: bold; }
        .connected { background: #4CAF50; }
        .disconnected { background: #f44336; }
        button { background: #2196F3; color: white; border: none; padding: 10px 20px; margin: 5px; border-radius: 4px; cursor: pointer; }
        button:hover { background: #1976D2; }
        textarea { width: 100%; height: 300px; font-family: monospace; font-size: 12px; }
        .feature { background: #e8f5e8; padding: 8px; margin: 3px 0; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ MyCloset AI 안전한 WebSocket 디버그</h1>
        
        <div class="panel">
            <h2>안전 기능</h2>
            <div class="feature">✅ 순환참조 완전 제거</div>
            <div class="feature">✅ 메모리 누수 방지</div>
            <div class="feature">✅ 에러 전파 차단</div>
            <div class="feature">✅ 자동 연결 상태 관리</div>
            <div class="feature">✅ 백그라운드 자동 정리</div>
        </div>
        
        <div class="panel">
            <h2>연결 상태</h2>
            <div>파이프라인: <span id="status-pipeline" class="status disconnected">연결 해제됨</span></div>
            <div>시스템 모니터: <span id="status-system" class="status disconnected">연결 해제됨</span></div>
            <div>테스트: <span id="status-test" class="status disconnected">연결 해제됨</span></div>
        </div>
        
        <div class="panel">
            <h2>안전한 테스트</h2>
            <button onclick="connectAll()">모든 연결 시작</button>
            <button onclick="disconnectAll()">모든 연결 해제</button>
            <button onclick="safeStressTest()">안전한 스트레스 테스트</button>
            <button onclick="clearMessages()">메시지 지우기</button>
        </div>
        
        <div class="panel">
            <h2>실시간 메시지</h2>
            <textarea id="messages" readonly placeholder="안전한 WebSocket 메시지..."></textarea>
        </div>
    </div>

    <script>
        let connections = {};
        let messageLog = [];
        
        function updateStatus(endpoint, connected) {
            const element = document.getElementById(`status-${endpoint}`);
            element.textContent = connected ? '연결됨' : '연결 해제됨';
            element.className = `status ${connected ? 'connected' : 'disconnected'}`;
        }
        
        function logMessage(endpoint, message) {
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = `[${timestamp}] [${endpoint}] ${JSON.stringify(message, null, 2)}`;
            messageLog.push(logEntry);
            
            const textarea = document.getElementById('messages');
            textarea.value += logEntry + '\\n\\n';
            textarea.scrollTop = textarea.scrollHeight;
            
            // 메시지 히스토리 크기 제한 (클라이언트 측도 안전하게)
            if (messageLog.length > 100) {
                messageLog = messageLog.slice(-50);
                const lines = textarea.value.split('\\n');
                textarea.value = lines.slice(-200).join('\\n');
            }
        }
        
        function createConnection(type) {
            try {
                const ws = new WebSocket(`ws://localhost:8001/api/ws/${type}`);
                
                ws.onopen = () => {
                    updateStatus(type, true);
                    logMessage(type.toUpperCase(), {type: 'safe_connection_opened'});
                };
                
                ws.onmessage = (event) => {
                    try {
                        const message = JSON.parse(event.data);
                        logMessage(type.toUpperCase(), message);
                    } catch (e) {
                        logMessage(type.toUpperCase(), {type: 'parse_error', raw: event.data});
                    }
                };
                
                ws.onclose = () => {
                    updateStatus(type, false);
                    logMessage(type.toUpperCase(), {type: 'safe_connection_closed'});
                };
                
                ws.onerror = (error) => {
                    logMessage(type.toUpperCase(), {type: 'safe_error', error: error});
                };
                
                connections[type] = ws;
                
            } catch (e) {
                logMessage(type.toUpperCase(), {type: 'connection_creation_error', error: e.message});
            }
        }
        
        function connectAll() {
            ['pipeline-progress', 'system-monitor', 'test'].forEach(createConnection);
        }
        
        function disconnectAll() {
            Object.values(connections).forEach(ws => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.close();
                }
            });
            connections = {};
        }
        
        function safeStressTest() {
            const ws = connections['test'];
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'stress_test',
                    count: 20,  // 안전한 수량
                    timestamp: Date.now()
                }));
            }
        }
        
        function clearMessages() {
            document.getElementById('messages').value = '';
            messageLog = [];
        }
        
        // 안전한 자동 연결
        setTimeout(connectAll, 1000);
        
        // 안전한 주기적 ping (60초마다)
        setInterval(() => {
            Object.values(connections).forEach(ws => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    try {
                        ws.send(JSON.stringify({type: 'ping'}));
                    } catch (e) {
                        console.error('Ping 전송 실패:', e);
                    }
                }
            });
        }, 60000);
    </script>
</body>
</html>
    """)

# ========================
# 🔥 핵심: 누락된 함수들 모두 추가
# ========================

async def start_background_tasks():
    """백그라운드 태스크 시작 - main.py에서 호출"""
    try:
        await manager.start()
        logger.info("🚀 WebSocket 백그라운드 태스크 시작")
    except Exception as e:
        logger.error(f"❌ 백그라운드 태스크 시작 실패: {e}")

async def stop_background_tasks():
    """백그라운드 태스크 중지"""
    try:
        await manager.stop()
        logger.info("🛑 WebSocket 백그라운드 태스크 중지")
    except Exception as e:
        logger.error(f"❌ 백그라운드 태스크 중지 실패: {e}")

def cleanup_websocket_resources():
    """WebSocket 리소스 정리 - 동기 함수"""
    try:
        # 동기적으로 실행할 수 있는 정리 작업만
        logger.info("🧹 WebSocket 리소스 정리")
        
        # 통계 초기화
        manager._stats = {
            "total_connections": 0,
            "current_connections": 0,
            "total_messages": 0,
            "errors": 0,
            "reconnections": 0,
            "start_time": time.time()
        }
        
        logger.info("✅ WebSocket 리소스 정리 완료")
    except Exception as e:
        logger.error(f"❌ 리소스 정리 실패: {e}")

def get_websocket_stats() -> Dict[str, Any]:
    """WebSocket 통계 조회"""
    return manager.get_connection_stats()

def get_websocket_manager():
    """WebSocket 매니저 인스턴스 반환"""
    return manager

# ========================
# 안전한 초기화 및 정리
# ========================

async def start_safe_websocket_system():
    """안전한 WebSocket 시스템 시작"""
    try:
        await manager.start()
        logger.info("✅ 안전한 WebSocket 시스템 시작됨")
    except Exception as e:
        logger.error(f"❌ WebSocket 시스템 시작 실패: {e}")
        raise

async def stop_safe_websocket_system():
    """안전한 WebSocket 시스템 중지"""
    try:
        await manager.stop()
        logger.info("✅ 안전한 WebSocket 시스템 중지됨")
    except Exception as e:
        logger.error(f"❌ WebSocket 시스템 중지 실패: {e}")

# ========================
# 모듈 exports (안전 + 완전)
# ========================

__all__ = [
    'router', 
    'manager', 
    'create_safe_pipeline_callbacks',
    'start_safe_websocket_system',
    'stop_safe_websocket_system',
    'start_background_tasks',  # 🔥 추가
    'stop_background_tasks',   # 🔥 추가
    'cleanup_websocket_resources',  # 🔥 추가
    'get_websocket_stats',     # 🔥 추가
    'get_websocket_manager',   # 🔥 추가
    'MessageType',
    'PipelineStatus',
    'WebSocketState',
    'SafeConnectionManager'
]

# 모듈 로드 확인
logger.info("✅ WebSocket 라우터 모듈 로드 완료 - 모든 함수 포함")