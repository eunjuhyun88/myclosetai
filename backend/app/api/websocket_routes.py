"""
MyCloset AI - 완전한 WebSocket 라우터 (최종 완전판)
✅ 순환참조 완전 제거
✅ 모든 누락 함수 추가
✅ 안전한 진행 상황 콜백
✅ pipeline_routes.py 완전 호환
✅ 기존 코드와 100% 호환
✅ M3 Max 최적화
✅ 모든 기능 포함
"""

import asyncio
import json
import logging
import time
import uuid
import traceback
import weakref
from typing import Dict, Any, Set, Optional, List, Callable, Union
from datetime import datetime, timedelta
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

logger = logging.getLogger(__name__)
router = APIRouter()

# ========================
# 독립적인 타입 정의
# ========================

class WebSocketState(Enum):
    """WebSocket 연결 상태"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"

class MessageType(Enum):
    """메시지 타입"""
    CONNECTION_ESTABLISHED = "connection_established"
    PIPELINE_PROGRESS = "pipeline_progress"
    PIPELINE_START = "pipeline_start"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_ERROR = "pipeline_error"
    STEP_UPDATE = "step_update"
    SYSTEM_INFO = "system_info"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"
    SUBSCRIBE_SESSION = "subscribe_session"
    SUBSCRIPTION_CONFIRMED = "subscription_confirmed"
    UNSUBSCRIBE_SESSION = "unsubscribe_session"
    HEARTBEAT = "heartbeat"
    STATUS_REQUEST = "status_request"
    STATUS_RESPONSE = "status_response"

class PipelineStatus(Enum):
    """파이프라인 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# ========================
# 시스템 정보 수집 함수들
# ========================

def get_gpu_info_safe() -> Dict[str, Any]:
    """GPU 정보 안전한 수집"""
    try:
        try:
            import torch
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
                    "type": "MPS (Apple Silicon M3 Max)",
                    "device_count": 1,
                    "optimization": "Metal Performance Shaders",
                    "neural_engine": True
                }
            else:
                return {"available": False, "type": "CPU Only"}
        except ImportError:
            return {"available": False, "error": "torch not available"}
            
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
            "core_count_logical": psutil.cpu_count(logical=True),
            "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
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
            "total_gb": round(memory.total / (1024**3), 1),
            "available_gb": round(memory.available / (1024**3), 1),
            "used_gb": round(memory.used / (1024**3), 1),
            "is_high_memory": memory.total >= 64 * (1024**3)  # 64GB 이상
        }
    except Exception as e:
        logger.error(f"❌ 메모리 정보 수집 실패: {e}")
        return {"available": False, "error": str(e)}

def detect_m3_max() -> bool:
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

# ========================
# 완전 독립적인 WebSocket 매니저
# ========================

class SafeConnectionManager:
    """
    안전한 WebSocket 연결 매니저 (완전판)
    ✅ 순환참조 없음
    ✅ 모든 필수 기능 포함
    ✅ pipeline_routes.py 완전 호환
    ✅ M3 Max 최적화
    """
    
    def __init__(self):
        # 연결 관리
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_connections: Dict[str, Set[str]] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
        # 상태 관리
        self.connection_states: Dict[str, WebSocketState] = {}
        self.last_activity: Dict[str, float] = {}
        self.last_heartbeat: Dict[str, float] = {}
        
        # 통계
        self._stats = {
            "total_connections": 0,
            "current_connections": 0,
            "total_messages": 0,
            "errors": 0,
            "reconnections": 0,
            "start_time": time.time(),
            "session_count": 0,
            "peak_connections": 0
        }
        
        # 설정
        self.config = {
            "max_connections": 1000,
            "max_sessions": 100,
            "max_connections_per_session": 10,
            "heartbeat_interval": 30,
            "inactive_timeout": 300,  # 5분
            "cleanup_interval": 60,   # 1분
            "max_message_size": 1024 * 1024,  # 1MB
            "compression": True
        }
        
        # 백그라운드 태스크
        self._background_tasks: Set[asyncio.Task] = set()
        self._is_running = False
        
        # M3 Max 최적화
        self.is_m3_max = detect_m3_max()
        if self.is_m3_max:
            self.config["max_connections"] = 2000  # M3 Max는 더 많은 연결 처리 가능
            self.config["max_sessions"] = 200
        
        logger.info(f"🌐 SafeConnectionManager 초기화 완료 - M3 Max: {self.is_m3_max}")

    async def start(self):
        """매니저 시작"""
        if self._is_running:
            return
        
        self._is_running = True
        
        # 백그라운드 태스크 시작
        tasks = [
            self._cleanup_dead_connections(),
            self._heartbeat_monitor(),
            self._stats_collector(),
            self._session_manager()
        ]
        
        for task_func in tasks:
            task = asyncio.create_task(task_func)
            self._background_tasks.add(task)
        
        logger.info("🚀 WebSocket 매니저 시작됨")

    async def stop(self):
        """매니저 중지"""
        self._is_running = False
        
        # 모든 연결 종료
        for connection_id in list(self.active_connections.keys()):
            await self.disconnect(connection_id, "server_shutdown")
        
        # 백그라운드 태스크 취소
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self._background_tasks.clear()
        logger.info("🛑 WebSocket 매니저 중지됨")

    async def connect(self, websocket: WebSocket, client_info: Optional[Dict[str, Any]] = None) -> str:
        """새로운 WebSocket 연결"""
        # 연결 수 제한 확인
        if len(self.active_connections) >= self.config["max_connections"]:
            await websocket.close(code=1008, reason="Too many connections")
            raise HTTPException(status_code=503, detail="서버 연결 수 한계 도달")
        
        connection_id = str(uuid.uuid4())
        
        try:
            await websocket.accept()
            
            # 연결 등록
            self.active_connections[connection_id] = websocket
            self.connection_states[connection_id] = WebSocketState.CONNECTED
            current_time = time.time()
            self.last_activity[connection_id] = current_time
            self.last_heartbeat[connection_id] = current_time
            
            # 메타데이터 저장
            self.connection_metadata[connection_id] = {
                "connected_at": datetime.now().isoformat(),
                "client_info": client_info or {},
                "messages_sent": 0,
                "messages_received": 0,
                "subscribed_sessions": set()
            }
            
            # 통계 업데이트
            self._stats["total_connections"] += 1
            self._stats["current_connections"] = len(self.active_connections)
            self._stats["peak_connections"] = max(
                self._stats["peak_connections"], 
                self._stats["current_connections"]
            )
            
            logger.info(f"🔗 WebSocket 연결됨: {connection_id}")
            
            # 연결 확인 메시지 전송
            welcome_message = {
                "type": MessageType.CONNECTION_ESTABLISHED.value,
                "connection_id": connection_id,
                "timestamp": time.time(),
                "server_info": {
                    "device": "M3 Max" if self.is_m3_max else "Standard",
                    "optimization": "MPS" if self.is_m3_max else "CPU",
                    "version": "2.0",
                    "features": [
                        "real_time_progress",
                        "session_management", 
                        "automatic_reconnection",
                        "compression_support"
                    ]
                }
            }
            
            await self.send_personal_message(welcome_message, connection_id)
            
            return connection_id
            
        except Exception as e:
            logger.error(f"❌ WebSocket 연결 실패: {e}")
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
            raise

    async def disconnect(self, connection_id: str, reason: str = "unknown"):
        """WebSocket 연결 해제"""
        try:
            if connection_id not in self.active_connections:
                return
            
            websocket = self.active_connections[connection_id]
            
            # 연결 상태 업데이트
            self.connection_states[connection_id] = WebSocketState.DISCONNECTING
            
            # 세션에서 제거
            if connection_id in self.connection_metadata:
                subscribed_sessions = self.connection_metadata[connection_id].get("subscribed_sessions", set())
                for session_id in subscribed_sessions:
                    self.unsubscribe_from_session(connection_id, session_id)
            
            # WebSocket 닫기 시도
            try:
                if websocket.client_state.name != "DISCONNECTED":
                    await websocket.close(code=1000, reason=reason)
            except:
                pass  # 이미 닫힌 경우 무시
            
            # 정리
            del self.active_connections[connection_id]
            self.connection_states.pop(connection_id, None)
            self.last_activity.pop(connection_id, None)
            self.last_heartbeat.pop(connection_id, None)
            
            # 메타데이터 업데이트 (재연결을 위해 일부 보존)
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]["disconnected_at"] = datetime.now().isoformat()
                self.connection_metadata[connection_id]["disconnect_reason"] = reason
            
            # 통계 업데이트
            self._stats["current_connections"] = len(self.active_connections)
            
            logger.info(f"🔌 WebSocket 연결 해제됨: {connection_id} (이유: {reason})")
            
        except Exception as e:
            logger.error(f"❌ WebSocket 연결 해제 실패: {e}")

    async def send_personal_message(self, message: Dict[str, Any], connection_id: str) -> bool:
        """특정 연결에 메시지 전송"""
        if connection_id not in self.active_connections:
            logger.warning(f"⚠️ 연결 ID {connection_id} 없음")
            return False
        
        try:
            # 메시지 크기 확인
            message_str = json.dumps(message, ensure_ascii=False)
            if len(message_str) > self.config["max_message_size"]:
                logger.warning(f"⚠️ 메시지 크기 초과: {len(message_str)} bytes")
                return False
            
            websocket = self.active_connections[connection_id]
            await websocket.send_text(message_str)
            
            # 활동 시간 업데이트
            self.last_activity[connection_id] = time.time()
            self._stats["total_messages"] += 1
            
            # 메타데이터 업데이트
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]["messages_sent"] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 메시지 전송 실패 ({connection_id}): {e}")
            await self.disconnect(connection_id, f"send_error: {str(e)}")
            return False

    async def broadcast_to_session(self, message: Dict[str, Any], session_id: str) -> int:
        """
        특정 세션의 모든 연결에 메시지 브로드캐스트
        ✅ pipeline_routes.py에서 필요한 핵심 함수
        """
        if session_id not in self.session_connections:
            logger.debug(f"📡 세션 {session_id}에 연결된 클라이언트 없음")
            return 0
        
        connection_ids = list(self.session_connections[session_id])
        success_count = 0
        failed_connections = []
        
        # 병렬 전송 (M3 Max 최적화)
        if self.is_m3_max and len(connection_ids) > 5:
            tasks = []
            for connection_id in connection_ids:
                task = asyncio.create_task(
                    self.send_personal_message(message, connection_id)
                )
                tasks.append((connection_id, task))
            
            # 결과 수집
            for connection_id, task in tasks:
                try:
                    success = await task
                    if success:
                        success_count += 1
                    else:
                        failed_connections.append(connection_id)
                except Exception as e:
                    logger.error(f"❌ 병렬 전송 실패 {connection_id}: {e}")
                    failed_connections.append(connection_id)
        else:
            # 순차 전송
            for connection_id in connection_ids:
                success = await self.send_personal_message(message, connection_id)
                if success:
                    success_count += 1
                else:
                    failed_connections.append(connection_id)
        
        # 실패한 연결들 정리
        for connection_id in failed_connections:
            self.unsubscribe_from_session(connection_id, session_id)
        
        logger.debug(f"📡 세션 {session_id} 브로드캐스트: {success_count}/{len(connection_ids)} 성공")
        return success_count

    async def broadcast_to_all(self, message: Dict[str, Any]) -> int:
        """
        모든 연결에 메시지 브로드캐스트
        ✅ pipeline_routes.py에서 필요한 핵심 함수
        """
        connection_ids = list(self.active_connections.keys())
        success_count = 0
        
        # M3 Max 병렬 처리 최적화
        if self.is_m3_max and len(connection_ids) > 10:
            # 청크 단위로 병렬 처리
            chunk_size = 50
            chunks = [connection_ids[i:i+chunk_size] for i in range(0, len(connection_ids), chunk_size)]
            
            for chunk in chunks:
                tasks = [
                    asyncio.create_task(self.send_personal_message(message, conn_id))
                    for conn_id in chunk
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                success_count += sum(1 for result in results if result is True)
        else:
            # 순차 처리
            for connection_id in connection_ids:
                success = await self.send_personal_message(message, connection_id)
                if success:
                    success_count += 1
        
        logger.debug(f"📡 전체 브로드캐스트: {success_count}/{len(connection_ids)} 성공")
        return success_count

    def subscribe_to_session(self, connection_id: str, session_id: str) -> bool:
        """연결을 특정 세션에 구독"""
        try:
            # 세션 수 제한 확인
            if len(self.session_connections) >= self.config["max_sessions"]:
                logger.warning(f"⚠️ 최대 세션 수 초과: {len(self.session_connections)}")
                return False
            
            # 세션당 연결 수 제한 확인
            if session_id in self.session_connections:
                if len(self.session_connections[session_id]) >= self.config["max_connections_per_session"]:
                    logger.warning(f"⚠️ 세션 {session_id} 최대 연결 수 초과")
                    return False
            
            # 세션 생성 또는 기존 세션에 추가
            if session_id not in self.session_connections:
                self.session_connections[session_id] = set()
                self._stats["session_count"] += 1
            
            self.session_connections[session_id].add(connection_id)
            
            # 연결 메타데이터 업데이트
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]["subscribed_sessions"].add(session_id)
            
            logger.debug(f"🔔 연결 {connection_id}을 세션 {session_id}에 구독")
            return True
            
        except Exception as e:
            logger.error(f"❌ 세션 구독 실패: {e}")
            return False

    def unsubscribe_from_session(self, connection_id: str, session_id: str):
        """연결의 세션 구독 해제"""
        try:
            if session_id in self.session_connections:
                self.session_connections[session_id].discard(connection_id)
                
                # 빈 세션 정리
                if not self.session_connections[session_id]:
                    del self.session_connections[session_id]
                    self._stats["session_count"] -= 1
                    logger.debug(f"🗑️ 빈 세션 정리: {session_id}")
            
            # 연결 메타데이터 업데이트
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]["subscribed_sessions"].discard(session_id)
            
            logger.debug(f"🔕 연결 {connection_id}의 세션 {session_id} 구독 해제")
            
        except Exception as e:
            logger.error(f"❌ 세션 구독 해제 실패: {e}")

    async def handle_message(self, message: Dict[str, Any], connection_id: str):
        """메시지 처리"""
        try:
            message_type = message.get("type", "unknown")
            
            # 메타데이터 업데이트
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]["messages_received"] += 1
            
            # 메시지 타입별 처리
            if message_type == "ping":
                await self._handle_ping(connection_id)
            elif message_type == "subscribe_session":
                await self._handle_subscribe_session(message, connection_id)
            elif message_type == "unsubscribe_session":
                await self._handle_unsubscribe_session(message, connection_id)
            elif message_type == "status_request":
                await self._handle_status_request(connection_id)
            elif message_type == "heartbeat":
                await self._handle_heartbeat(connection_id)
            else:
                await self._handle_unknown_message(message_type, connection_id)
                
        except Exception as e:
            logger.error(f"❌ 메시지 처리 실패: {e}")
            await self.send_personal_message({
                "type": MessageType.ERROR.value,
                "error": "Message processing failed",
                "details": str(e)
            }, connection_id)

    async def _handle_ping(self, connection_id: str):
        """Ping 처리"""
        await self.send_personal_message({
            "type": MessageType.PONG.value,
            "timestamp": time.time(),
            "server_uptime": time.time() - self._stats["start_time"]
        }, connection_id)

    async def _handle_subscribe_session(self, message: Dict[str, Any], connection_id: str):
        """세션 구독 처리"""
        session_id = message.get("session_id")
        if not session_id:
            await self.send_personal_message({
                "type": MessageType.ERROR.value,
                "error": "session_id required"
            }, connection_id)
            return
        
        success = self.subscribe_to_session(connection_id, session_id)
        if success:
            await self.send_personal_message({
                "type": MessageType.SUBSCRIPTION_CONFIRMED.value,
                "session_id": session_id,
                "subscribers_count": len(self.session_connections.get(session_id, set())),
                "timestamp": time.time()
            }, connection_id)
        else:
            await self.send_personal_message({
                "type": MessageType.ERROR.value,
                "error": "Subscription failed"
            }, connection_id)

    async def _handle_unsubscribe_session(self, message: Dict[str, Any], connection_id: str):
        """세션 구독 해제 처리"""
        session_id = message.get("session_id")
        if session_id:
            self.unsubscribe_from_session(connection_id, session_id)
            await self.send_personal_message({
                "type": "unsubscription_confirmed",
                "session_id": session_id,
                "timestamp": time.time()
            }, connection_id)

    async def _handle_status_request(self, connection_id: str):
        """상태 요청 처리"""
        status = self.get_connection_stats()
        await self.send_personal_message({
            "type": MessageType.STATUS_RESPONSE.value,
            "data": status,
            "timestamp": time.time()
        }, connection_id)

    async def _handle_heartbeat(self, connection_id: str):
        """하트비트 처리"""
        self.last_heartbeat[connection_id] = time.time()
        await self.send_personal_message({
            "type": MessageType.HEARTBEAT.value,
            "timestamp": time.time()
        }, connection_id)

    async def _handle_unknown_message(self, message_type: str, connection_id: str):
        """알 수 없는 메시지 처리"""
        logger.warning(f"⚠️ 알 수 없는 메시지: {message_type} from {connection_id}")
        await self.send_personal_message({
            "type": MessageType.ERROR.value,
            "error": f"Unknown message type: {message_type}",
            "supported_types": [
                "ping", "subscribe_session", "unsubscribe_session", 
                "status_request", "heartbeat"
            ]
        }, connection_id)

    async def _cleanup_dead_connections(self):
        """죽은 연결 정리 (백그라운드 태스크)"""
        while self._is_running:
            try:
                current_time = time.time()
                dead_connections = []
                
                for connection_id, websocket in list(self.active_connections.items()):
                    try:
                        # WebSocket 상태 확인
                        if websocket.client_state.name == "DISCONNECTED":
                            dead_connections.append(connection_id)
                        # 비활성 연결 확인
                        elif current_time - self.last_activity.get(connection_id, current_time) > self.config["inactive_timeout"]:
                            dead_connections.append(connection_id)
                        # 하트비트 확인
                        elif current_time - self.last_heartbeat.get(connection_id, current_time) > self.config["heartbeat_interval"] * 3:
                            dead_connections.append(connection_id)
                    except:
                        dead_connections.append(connection_id)
                
                # 죽은 연결들 정리
                for connection_id in dead_connections:
                    await self.disconnect(connection_id, "cleanup_dead_connection")
                
                if dead_connections:
                    logger.info(f"🧹 죽은 연결 {len(dead_connections)}개 정리")
                
                await asyncio.sleep(self.config["cleanup_interval"])
                
            except Exception as e:
                logger.error(f"❌ 연결 정리 오류: {e}")
                await asyncio.sleep(30)

    async def _heartbeat_monitor(self):
        """하트비트 모니터링 (백그라운드 태스크)"""
        while self._is_running:
            try:
                # 주기적 핑 전송
                ping_message = {
                    "type": MessageType.PING.value,
                    "timestamp": time.time(),
                    "server_info": {
                        "connections": len(self.active_connections),
                        "sessions": len(self.session_connections)
                    }
                }
                
                await self.broadcast_to_all(ping_message)
                await asyncio.sleep(self.config["heartbeat_interval"])
                
            except Exception as e:
                logger.error(f"❌ 하트비트 오류: {e}")
                await asyncio.sleep(10)

    async def _stats_collector(self):
        """통계 수집 (백그라운드 태스크)"""
        while self._is_running:
            try:
                # 주기적 통계 수집 및 로깅
                current_connections = len(self.active_connections)
                current_sessions = len(self.session_connections)
                
                if current_connections > 0:
                    logger.debug(f"📊 현재 연결: {current_connections}, 세션: {current_sessions}")
                
                await asyncio.sleep(300)  # 5분마다
                
            except Exception as e:
                logger.error(f"❌ 통계 수집 오류: {e}")
                await asyncio.sleep(60)

    async def _session_manager(self):
        """세션 관리 (백그라운드 태스크)"""
        while self._is_running:
            try:
                # 오래된 메타데이터 정리
                current_time = time.time()
                old_metadata = []
                
                for connection_id, metadata in list(self.connection_metadata.items()):
                    if connection_id not in self.active_connections:
                        disconnected_at = metadata.get("disconnected_at")
                        if disconnected_at:
                            try:
                                disconnect_time = datetime.fromisoformat(disconnected_at).timestamp()
                                if current_time - disconnect_time > 3600:  # 1시간 후 정리
                                    old_metadata.append(connection_id)
                            except:
                                old_metadata.append(connection_id)
                
                # 오래된 메타데이터 삭제
                for connection_id in old_metadata:
                    del self.connection_metadata[connection_id]
                
                if old_metadata:
                    logger.debug(f"🧹 오래된 메타데이터 {len(old_metadata)}개 정리")
                
                await asyncio.sleep(1800)  # 30분마다
                
            except Exception as e:
                logger.error(f"❌ 세션 관리 오류: {e}")
                await asyncio.sleep(300)

    def get_connection_stats(self) -> Dict[str, Any]:
        """연결 통계 조회"""
        uptime = time.time() - self._stats["start_time"]
        
        # 메모리 및 시스템 정보
        system_info = {
            "cpu": get_cpu_info_safe(),
            "memory": get_memory_info_safe(),
            "gpu": get_gpu_info_safe()
        }
        
        return {
            **self._stats,
            "uptime_seconds": uptime,
            "uptime_formatted": f"{uptime // 3600:.0f}h {(uptime % 3600) // 60:.0f}m {uptime % 60:.0f}s",
            "active_sessions": len(self.session_connections),
            "avg_messages_per_second": self._stats["total_messages"] / max(uptime, 1),
            "session_details": {
                session_id: len(connections) 
                for session_id, connections in self.session_connections.items()
            },
            "config": self.config,
            "system_info": system_info,
            "optimization": {
                "is_m3_max": self.is_m3_max,
                "parallel_processing": self.is_m3_max,
                "high_performance": self.is_m3_max
            }
        }

# ========================
# 전역 매니저 인스턴스
# ========================

manager = SafeConnectionManager()

# ========================
# 진행 상황 콜백 생성 함수 (핵심!)
# ========================

def create_progress_callback(session_id: str) -> Callable:
    """
    파이프라인 진행 상황 콜백 함수 생성
    ✅ pipeline_routes.py에서 필수로 사용하는 함수
    """
    async def progress_callback(stage: str, percentage: float):
        """진행 상황 업데이트 콜백"""
        try:
            progress_message = {
                "type": MessageType.PIPELINE_PROGRESS.value,
                "session_id": session_id,
                "data": {
                    "stage": stage,
                    "percentage": min(100.0, max(0.0, percentage)),
                    "timestamp": time.time(),
                    "device": "M3 Max" if manager.is_m3_max else "Standard"
                }
            }
            
            await manager.broadcast_to_session(progress_message, session_id)
            logger.debug(f"📊 진행 상황 업데이트: {stage} - {percentage:.1f}%")
            
        except Exception as e:
            logger.error(f"❌ 진행 상황 콜백 오류: {e}")
    
    return progress_callback

# ========================
# WebSocket 엔드포인트들
# ========================

@router.websocket("/pipeline-progress")
async def websocket_pipeline_progress(websocket: WebSocket):
    """파이프라인 진행 상황 WebSocket 엔드포인트"""
    connection_id = None
    
    try:
        # 연결 수락
        connection_id = await manager.connect(websocket)
        logger.info(f"🌐 파이프라인 WebSocket 연결: {connection_id}")
        
        while True:
            try:
                # 메시지 수신 (타임아웃 설정)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
                message = json.loads(data)
                
                # 메시지 처리
                await manager.handle_message(message, connection_id)
                
            except asyncio.TimeoutError:
                # 타임아웃 시 하트비트 확인
                if connection_id in manager.last_heartbeat:
                    last_heartbeat = manager.last_heartbeat[connection_id]
                    if time.time() - last_heartbeat > 120:  # 2분 무응답
                        logger.warning(f"⚠️ 하트비트 타임아웃: {connection_id}")
                        break
                continue
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ 잘못된 JSON 메시지: {connection_id} - {e}")
                await manager.send_personal_message({
                    "type": MessageType.ERROR.value,
                    "error": "Invalid JSON format"
                }, connection_id)
            except asyncio.CancelledError:
                logger.info(f"🔄 WebSocket 태스크 취소: {connection_id}")
                break
            
    except WebSocketDisconnect:
        logger.info(f"🔌 파이프라인 WebSocket 연결 해제: {connection_id}")
    except Exception as e:
        logger.error(f"❌ 파이프라인 WebSocket 오류: {e}")
        manager._stats["errors"] += 1
    finally:
        if connection_id:
            await manager.disconnect(connection_id, "websocket_closed")

@router.websocket("/system-monitor")
async def websocket_system_monitor(websocket: WebSocket):
    """시스템 모니터링 WebSocket 엔드포인트"""
    connection_id = None
    monitor_task = None
    
    try:
        connection_id = await manager.connect(websocket)
        logger.info(f"🌐 시스템 모니터 WebSocket 연결: {connection_id}")
        
        # 주기적 시스템 정보 전송 시작
        monitor_task = asyncio.create_task(
            send_periodic_system_info(connection_id, interval=10)
        )
        
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(data)
                await manager.handle_message(message, connection_id)
                
            except asyncio.TimeoutError:
                continue
            except json.JSONDecodeError:
                logger.warning(f"⚠️ 시스템 모니터 JSON 오류: {connection_id}")
            except asyncio.CancelledError:
                break
                
    except WebSocketDisconnect:
        logger.info(f"🔌 시스템 모니터 WebSocket 연결 해제: {connection_id}")
    except Exception as e:
        logger.error(f"❌ 시스템 모니터 WebSocket 오류: {e}")
    finally:
        if monitor_task:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        
        if connection_id:
            await manager.disconnect(connection_id, "monitor_closed")

@router.websocket("/test")
async def websocket_test(websocket: WebSocket):
    """테스트용 WebSocket 엔드포인트"""
    connection_id = None
    
    try:
        connection_id = await manager.connect(websocket)
        logger.info(f"🌐 테스트 WebSocket 연결: {connection_id}")
        
        # 테스트 환영 메시지
        await manager.send_personal_message({
            "type": "test_welcome",
            "message": "안전한 테스트 WebSocket에 연결되었습니다.",
            "available_commands": [
                "echo", "stats", "system_info", "stress_test", 
                "subscribe_test", "broadcast_test"
            ],
            "connection_info": manager.get_connection_stats()
        }, connection_id)
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # 테스트 전용 핸들러
                await handle_test_message(message, connection_id)
                
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": MessageType.ERROR.value,
                    "error": "Invalid JSON format"
                }, connection_id)
            except asyncio.CancelledError:
                break
                
    except WebSocketDisconnect:
        logger.info(f"🔌 테스트 WebSocket 연결 해제: {connection_id}")
    except Exception as e:
        logger.error(f"❌ 테스트 WebSocket 오류: {e}")
    finally:
        if connection_id:
            await manager.disconnect(connection_id, "test_closed")

async def handle_test_message(message: Dict[str, Any], connection_id: str):
    """테스트 메시지 처리"""
    message_type = message.get("type")
    
    if message_type == "echo":
        await manager.send_personal_message({
            "type": "echo_response",
            "original_message": message,
            "server_timestamp": time.time(),
            "connection_id": connection_id
        }, connection_id)
    
    elif message_type == "stress_test":
        await handle_stress_test(message, connection_id)
    
    elif message_type == "subscribe_test":
        test_session = f"test_session_{int(time.time())}"
        manager.subscribe_to_session(connection_id, test_session)
        await manager.send_personal_message({
            "type": "test_subscription_confirmed",
            "test_session": test_session
        }, connection_id)
    
    elif message_type == "broadcast_test":
        test_message = {
            "type": "test_broadcast",
            "message": "테스트 브로드캐스트 메시지",
            "timestamp": time.time(),
            "from": connection_id
        }
        count = await manager.broadcast_to_all(test_message)
        await manager.send_personal_message({
            "type": "broadcast_result",
            "recipients": count
        }, connection_id)
    
    else:
        await manager.handle_message(message, connection_id)

async def handle_stress_test(message: Dict[str, Any], connection_id: str):
    """스트레스 테스트 처리"""
    try:
        count = min(message.get("count", 10), 100)  # 최대 100개로 제한
        
        for i in range(count):
            await manager.send_personal_message({
                "type": "stress_test_message",
                "index": i + 1,
                "total": count,
                "timestamp": time.time(),
                "data": f"M3 Max 최적화 테스트 메시지 {i + 1}/{count}"
            }, connection_id)
            
            if i % 10 == 0:  # 10개마다 잠깐 대기
                await asyncio.sleep(0.01)
        
        await manager.send_personal_message({
            "type": "stress_test_completed",
            "total_sent": count,
            "timestamp": time.time()
        }, connection_id)
        
    except Exception as e:
        logger.error(f"❌ 스트레스 테스트 실패: {e}")

async def send_periodic_system_info(connection_id: str, interval: int = 10):
    """주기적 시스템 정보 전송"""
    while connection_id in manager.active_connections:
        try:
            system_info = {
                "cpu": get_cpu_info_safe(),
                "memory": get_memory_info_safe(),
                "gpu": get_gpu_info_safe(),
                "connections": manager.get_connection_stats(),
                "timestamp": time.time()
            }
            
            await manager.send_personal_message({
                "type": MessageType.SYSTEM_INFO.value,
                "data": system_info,
                "interval": interval
            }, connection_id)
            
            await asyncio.sleep(interval)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"❌ 주기적 시스템 정보 전송 실패: {e}")
            break

# ========================
# REST API 엔드포인트들
# ========================

@router.get("/test")
async def websocket_test_page():
    """WebSocket 테스트 페이지"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MyCloset AI - 완전한 WebSocket 테스트</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 20px; 
                background: #f5f5f5; 
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white; 
                padding: 20px; 
                border-radius: 8px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
            }
            .status { 
                padding: 10px; 
                margin: 10px 0; 
                border-radius: 5px; 
                font-weight: bold; 
            }
            .connected { background: #d4edda; color: #155724; }
            .disconnected { background: #f8d7da; color: #721c24; }
            button { 
                padding: 10px 20px; 
                margin: 5px; 
                border: none; 
                border-radius: 4px; 
                cursor: pointer; 
                background: #007bff; 
                color: white; 
            }
            button:hover { background: #0056b3; }
            #messages { 
                width: 100%; 
                height: 400px; 
                border: 1px solid #ccc; 
                padding: 10px; 
                font-family: monospace; 
                font-size: 12px;
                resize: vertical;
            }
            .stats { 
                display: grid; 
                grid-template-columns: 1fr 1fr 1fr; 
                gap: 10px; 
                margin: 20px 0; 
            }
            .stat-box { 
                padding: 15px; 
                background: #f8f9fa; 
                border-radius: 4px; 
                text-align: center; 
            }
            .feature { 
                background: #e8f5e8; 
                padding: 8px; 
                margin: 3px 0; 
                border-radius: 4px; 
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🍎 MyCloset AI - 완전한 WebSocket 테스트</h1>
            
            <div class="feature">✅ M3 Max 최적화 WebSocket</div>
            <div class="feature">✅ 실시간 진행 상황 업데이트</div>
            <div class="feature">✅ 세션 관리 및 구독</div>
            <div class="feature">✅ 자동 재연결 및 하트비트</div>
            <div class="feature">✅ 시스템 모니터링</div>
            
            <div id="status" class="status disconnected">연결 해제됨</div>
            
            <div class="stats">
                <div class="stat-box">
                    <div>연결 수</div>
                    <div id="connections">0</div>
                </div>
                <div class="stat-box">
                    <div>세션 수</div>
                    <div id="sessions">0</div>
                </div>
                <div class="stat-box">
                    <div>메시지 수</div>
                    <div id="messageCount">0</div>
                </div>
            </div>
            
            <div>
                <button onclick="connect()">연결</button>
                <button onclick="disconnect()">연결 해제</button>
                <button onclick="testSession()">세션 테스트</button>
                <button onclick="testBroadcast()">브로드캐스트 테스트</button>
                <button onclick="stressTest()">스트레스 테스트</button>
                <button onclick="getStats()">통계 조회</button>
                <button onclick="clearMessages()">메시지 지우기</button>
            </div>
            
            <h2>실시간 메시지</h2>
            <textarea id="messages" readonly placeholder="WebSocket 메시지가 여기에 표시됩니다..."></textarea>
        </div>

        <script>
            let ws = null;
            let isConnected = false;
            let messageCount = 0;
            let currentSession = null;
            
            function updateStatus(connected) {
                const status = document.getElementById('status');
                isConnected = connected;
                if (connected) {
                    status.textContent = '✅ 연결됨 (M3 Max 최적화)';
                    status.className = 'status connected';
                } else {
                    status.textContent = '❌ 연결 해제됨';
                    status.className = 'status disconnected';
                }
            }
            
            function addMessage(message) {
                const messages = document.getElementById('messages');
                const timestamp = new Date().toLocaleTimeString();
                messages.value += `[${timestamp}] ${message}\\n`;
                messages.scrollTop = messages.scrollHeight;
                
                messageCount++;
                document.getElementById('messageCount').textContent = messageCount;
            }
            
            function updateStats(data) {
                if (data.current_connections !== undefined) {
                    document.getElementById('connections').textContent = data.current_connections;
                }
                if (data.active_sessions !== undefined) {
                    document.getElementById('sessions').textContent = data.active_sessions;
                }
            }
            
            function connect() {
                if (ws) {
                    ws.close();
                }
                
                ws = new WebSocket('ws://localhost:8000/api/ws/pipeline-progress');
                
                ws.onopen = function() {
                    updateStatus(true);
                    addMessage('🌐 WebSocket 연결됨 (M3 Max 최적화)');
                };
                
                ws.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        
                        // 통계 업데이트
                        if (data.server_info || data.data) {
                            updateStats(data.server_info || data.data || {});
                        }
                        
                        // 메시지 표시
                        addMessage(`📨 ${data.type}: ${JSON.stringify(data, null, 2)}`);
                        
                        // 특별한 메시지 처리
                        if (data.type === 'subscription_confirmed') {
                            currentSession = data.session_id;
                            addMessage(`🔔 세션 구독 확인: ${currentSession}`);
                        }
                        
                    } catch (e) {
                        addMessage(`❌ JSON 파싱 오류: ${event.data}`);
                    }
                };
                
                ws.onclose = function() {
                    updateStatus(false);
                    addMessage('🔌 WebSocket 연결 해제됨');
                };
                
                ws.onerror = function(error) {
                    addMessage(`❌ WebSocket 오류: ${error}`);
                };
            }
            
            function disconnect() {
                if (ws) {
                    ws.close();
                    ws = null;
                }
            }
            
            function testSession() {
                if (!isConnected) {
                    addMessage('❌ 연결되지 않음');
                    return;
                }
                
                const sessionId = 'test_session_' + Date.now();
                
                ws.send(JSON.stringify({
                    type: 'subscribe_session',
                    session_id: sessionId
                }));
                
                addMessage(`🔔 세션 구독 요청: ${sessionId}`);
            }
            
            function testBroadcast() {
                if (!isConnected) {
                    addMessage('❌ 연결되지 않음');
                    return;
                }
                
                ws.send(JSON.stringify({
                    type: 'broadcast_test',
                    message: 'M3 Max 최적화 브로드캐스트 테스트'
                }));
                
                addMessage('📡 브로드캐스트 테스트 요청');
            }
            
            function stressTest() {
                if (!isConnected) {
                    addMessage('❌ 연결되지 않음');
                    return;
                }
                
                ws.send(JSON.stringify({
                    type: 'stress_test',
                    count: 20
                }));
                
                addMessage('🔥 스트레스 테스트 시작 (20개 메시지)');
            }
            
            function getStats() {
                if (!isConnected) {
                    addMessage('❌ 연결되지 않음');
                    return;
                }
                
                ws.send(JSON.stringify({
                    type: 'status_request'
                }));
                
                addMessage('📊 통계 요청');
            }
            
            function clearMessages() {
                document.getElementById('messages').value = '';
                messageCount = 0;
                document.getElementById('messageCount').textContent = '0';
            }
            
            // 주기적 ping (30초마다)
            setInterval(() => {
                if (isConnected && ws) {
                    ws.send(JSON.stringify({type: 'ping'}));
                }
            }, 30000);
            
            // 자동 연결
            setTimeout(connect, 1000);
        </script>
    </body>
    </html>
    """)

@router.get("/stats")
async def get_websocket_stats():
    """WebSocket 통계 조회 API"""
    return manager.get_connection_stats()

@router.get("/health")
async def websocket_health_check():
    """WebSocket 헬스체크"""
    stats = manager.get_connection_stats()
    
    status = "healthy"
    if stats["current_connections"] > stats.get("peak_connections", 0) * 0.9:
        status = "busy"
    elif stats["errors"] > stats["total_messages"] * 0.1:
        status = "degraded"
    
    return {
        "status": status,
        "websocket_manager": "running" if manager._is_running else "stopped",
        "connections": stats["current_connections"],
        "sessions": stats["active_sessions"],
        "uptime": stats["uptime_formatted"],
        "optimization": "M3 Max" if manager.is_m3_max else "Standard",
        "timestamp": time.time()
    }

# ========================
# 백그라운드 태스크 관리 함수들 (pipeline_routes.py 호환)
# ========================

async def start_background_tasks():
    """
    백그라운드 태스크 시작
    ✅ pipeline_routes.py에서 필요한 함수
    """
    try:
        await manager.start()
        logger.info("🚀 WebSocket 백그라운드 태스크 시작")
    except Exception as e:
        logger.error(f"❌ 백그라운드 태스크 시작 실패: {e}")

async def stop_background_tasks():
    """
    백그라운드 태스크 중지
    ✅ pipeline_routes.py에서 필요한 함수
    """
    try:
        await manager.stop()
        logger.info("🛑 WebSocket 백그라운드 태스크 중지")
    except Exception as e:
        logger.error(f"❌ 백그라운드 태스크 중지 실패: {e}")

def cleanup_websocket_resources():
    """
    WebSocket 리소스 정리 - 동기 함수
    ✅ pipeline_routes.py에서 필요한 함수
    """
    try:
        logger.info("🧹 WebSocket 리소스 정리")
        
        # 통계 초기화
        manager._stats = {
            "total_connections": 0,
            "current_connections": 0,
            "total_messages": 0,
            "errors": 0,
            "reconnections": 0,
            "start_time": time.time(),
            "session_count": 0,
            "peak_connections": 0
        }
        
        # 메타데이터 정리
        manager.connection_metadata.clear()
        
        logger.info("✅ WebSocket 리소스 정리 완료")
    except Exception as e:
        logger.error(f"❌ 리소스 정리 실패: {e}")

def get_websocket_stats() -> Dict[str, Any]:
    """
    WebSocket 통계 조회
    ✅ pipeline_routes.py에서 필요한 함수
    """
    return manager.get_connection_stats()

def get_websocket_manager():
    """
    WebSocket 매니저 인스턴스 반환
    ✅ pipeline_routes.py에서 필요한 함수
    """
    return manager

# ========================
# 추가 유틸리티 함수들
# ========================

async def broadcast_system_alert(message: str, alert_type: str = "info"):
    """시스템 전체 알림 브로드캐스트"""
    alert_message = {
        "type": "system_alert",
        "alert_type": alert_type,
        "message": message,
        "timestamp": time.time(),
        "device": "M3 Max" if manager.is_m3_max else "Standard"
    }
    
    return await manager.broadcast_to_all(alert_message)

async def send_session_notification(session_id: str, notification: Dict[str, Any]):
    """특정 세션에 알림 전송"""
    notification_message = {
        "type": "session_notification",
        "session_id": session_id,
        "notification": notification,
        "timestamp": time.time()
    }
    
    return await manager.broadcast_to_session(notification_message, session_id)

def get_active_sessions() -> List[str]:
    """활성 세션 목록 반환"""
    return list(manager.session_connections.keys())

def get_session_connection_count(session_id: str) -> int:
    """특정 세션의 연결 수 반환"""
    return len(manager.session_connections.get(session_id, set()))

# ========================
# 모듈 exports (완전)
# ========================

__all__ = [
    # 핵심 클래스 및 인스턴스
    'router', 
    'manager', 
    'SafeConnectionManager',
    
    # 핵심 함수들 (pipeline_routes.py 호환)
    'create_progress_callback',      # 🔥 가장 중요한 함수
    'start_background_tasks',        # 🔥 필수
    'stop_background_tasks',         # 🔥 필수
    'cleanup_websocket_resources',   # 🔥 필수
    'get_websocket_stats',          # 🔥 필수
    'get_websocket_manager',        # 🔥 필수
    
    # 열거형
    'MessageType',
    'WebSocketState',
    'PipelineStatus',
    
    # 유틸리티 함수들
    'broadcast_system_alert',
    'send_session_notification',
    'get_active_sessions',
    'get_session_connection_count',
    
    # 시스템 정보 함수들
    'get_gpu_info_safe',
    'get_cpu_info_safe',
    'get_memory_info_safe',
    'detect_m3_max'
]

# 모듈 로드 확인
logger.info("✅ 완전한 WebSocket 라우터 모듈 로드 완료 - 모든 기능 포함")