# app/api/websocket_routes.py
"""
WebSocket API 라우터 - M3 Max 최적화 (최적 생성자 패턴 적용)
실시간 통신 및 진행 상황 업데이트 - FIXED VERSION
"""

import json
import time
import logging
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Set
from collections import defaultdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse

class OptimalRouterConstructor:
    """최적화된 라우터 생성자 패턴 베이스 클래스"""
    
    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # 1. 지능적 디바이스 자동 감지
        self.device = self._auto_detect_device(device)
        
        # 2. 기본 설정
        self.config = config or {}
        self.router_name = self.__class__.__name__
        self.logger = logging.getLogger(f"api.{self.router_name}")
        
        # 3. 표준 시스템 파라미터 추출
        self.device_type = kwargs.get('device_type', 'auto')
        self.memory_gb = kwargs.get('memory_gb', 16.0)
        self.is_m3_max = kwargs.get('is_m3_max', self._detect_m3_max())
        self.optimization_enabled = kwargs.get('optimization_enabled', True)
        
        # 4. 상태 초기화
        self.is_initialized = False

    def _auto_detect_device(self, preferred_device: Optional[str]) -> str:
        """지능적 디바이스 자동 감지"""
        if preferred_device:
            return preferred_device

        try:
            import torch
            if torch.backends.mps.is_available():
                return 'mps'
            elif torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        except:
            return 'cpu'

    def _detect_m3_max(self) -> bool:
        """M3 Max 칩 자동 감지"""
        try:
            import platform
            import subprocess

            if platform.system() == 'Darwin':
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                return 'M3' in result.stdout
        except:
            pass
        return False

class ConnectionManager:
    """
    🍎 M3 Max 최적화 WebSocket 연결 매니저 - FIXED VERSION
    최적 생성자 패턴 적용
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        ✅ 최적 생성자 - 연결 매니저 특화

        Args:
            device: 사용할 디바이스 (None=자동감지)
            config: 연결 매니저 설정
            **kwargs: 확장 파라미터들
                - max_connections: int = 1000
                - heartbeat_interval: float = 30.0
                - message_queue_size: int = 100
                - enable_message_history: bool = True
                - compression_enabled: bool = True
        """
        # 기본 설정
        self.device = device or self._auto_detect_device()
        self.config = config or {}
        self.logger = logging.getLogger("websocket.ConnectionManager")
        
        # 연결 매니저 특화 설정
        self.max_connections = kwargs.get('max_connections', 1000)
        self.heartbeat_interval = kwargs.get('heartbeat_interval', 30.0)
        self.message_queue_size = kwargs.get('message_queue_size', 100)
        self.enable_message_history = kwargs.get('enable_message_history', True)
        self.compression_enabled = kwargs.get('compression_enabled', True)
        
        # M3 Max 감지 및 최적화
        self.is_m3_max = kwargs.get('is_m3_max', self._detect_m3_max())
        if self.is_m3_max:
            self.max_connections = 2000
            self.heartbeat_interval = 15.0
        
        # 연결 관리
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_connections: Dict[str, Set[str]] = defaultdict(set)
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
        # 메시지 관리
        self.message_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.message_queues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # 통계
        self.stats = {
            "total_connections": 0,
            "current_connections": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "disconnections": 0,
            "errors": 0
        }
        
        # 백그라운드 태스크 관리
        self._background_tasks: Set[asyncio.Task] = set()
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        self.logger.info(f"🔗 WebSocket 연결 매니저 초기화 - {self.device} (M3 Max: {self.is_m3_max})")

    def _auto_detect_device(self) -> str:
        """디바이스 자동 감지"""
        try:
            import torch
            if torch.backends.mps.is_available():
                return 'mps'
            elif torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        except:
            return 'cpu'

    def _detect_m3_max(self) -> bool:
        """M3 Max 칩 자동 감지"""
        try:
            import platform
            import subprocess

            if platform.system() == 'Darwin':
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                return 'M3' in result.stdout
        except:
            pass
        return False

    async def connect(self, websocket: WebSocket, connection_id: str, session_id: Optional[str] = None) -> bool:
        """WebSocket 연결 - FIXED VERSION"""
        try:
            # 🔥 1. 연결 수 제한 확인 (accept 전에)
            if len(self.active_connections) >= self.max_connections:
                await websocket.close(code=1008, reason="Connection limit exceeded")
                self.logger.warning(f"❌ 연결 제한 초과: {connection_id}")
                return False
            
            # 🔥 2. WebSocket 연결 수락 (반드시 먼저!)
            await websocket.accept()
            self.logger.info(f"🔗 WebSocket accept 완료: {connection_id}")
            
            # 🔥 3. 연결 등록 (accept 후에)
            self.active_connections[connection_id] = websocket
            
            # 4. 세션 연결 매핑
            if session_id:
                self.session_connections[session_id].add(connection_id)
            
            # 5. 메타데이터 저장
            self.connection_metadata[connection_id] = {
                "session_id": session_id,
                "connected_at": time.time(),
                "last_ping": time.time(),
                "message_count": 0,
                "device": self.device,
                "m3_max_optimized": self.is_m3_max
            }
            
            # 6. 통계 업데이트
            self.stats["total_connections"] += 1
            self.stats["current_connections"] = len(self.active_connections)
            
            # 🔥 7. 연결 완료 메시지 전송 (짧은 지연 후)
            await asyncio.sleep(0.1)  # WebSocket 안정화 대기
            
            welcome_message = {
                "type": "connection_established",
                "connection_id": connection_id,
                "session_id": session_id,
                "message": "WebSocket 연결이 성공적으로 설정되었습니다",
                "device": self.device,
                "m3_max_optimized": self.is_m3_max,
                "features": {
                    "realtime_updates": True,
                    "message_history": self.enable_message_history,
                    "compression": self.compression_enabled
                },
                "timestamp": time.time()
            }
            
            # 안전한 메시지 전송
            success = await self._safe_send_message(websocket, welcome_message)
            if not success:
                await self.disconnect(connection_id)
                return False
            
            self.logger.info(f"✅ WebSocket 연결 성공: {connection_id} (세션: {session_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ WebSocket 연결 실패: {connection_id} - {e}")
            self.stats["errors"] += 1
            
            # 실패 시 정리
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
            
            try:
                await websocket.close(code=1011, reason="Internal server error")
            except:
                pass
            
            return False

    async def _safe_send_message(self, websocket: WebSocket, message: Dict[str, Any]) -> bool:
        """안전한 메시지 전송"""
        try:
            await websocket.send_text(json.dumps(message))
            return True
        except WebSocketDisconnect:
            self.logger.warning("WebSocket 연결이 클라이언트에 의해 종료됨")
            return False
        except Exception as e:
            self.logger.error(f"메시지 전송 실패: {e}")
            return False

    async def disconnect(self, connection_id: str):
        """WebSocket 연결 해제 - IMPROVED VERSION"""
        try:
            if connection_id in self.active_connections:
                websocket = self.active_connections[connection_id]
                
                # 연결 제거 (먼저)
                del self.active_connections[connection_id]
                
                # 메타데이터에서 세션 ID 확인
                metadata = self.connection_metadata.get(connection_id, {})
                session_id = metadata.get("session_id")
                
                # 세션 연결에서 제거
                if session_id and session_id in self.session_connections:
                    self.session_connections[session_id].discard(connection_id)
                    if not self.session_connections[session_id]:
                        del self.session_connections[session_id]
                
                # 메타데이터 정리
                if connection_id in self.connection_metadata:
                    self.connection_metadata[connection_id]["disconnected_at"] = time.time()
                    # 즉시 삭제하지 않고 나중에 정리
                
                # 메시지 히스토리 정리 (선택적)
                if not self.enable_message_history and connection_id in self.message_history:
                    del self.message_history[connection_id]
                
                # 통계 업데이트
                self.stats["disconnections"] += 1
                self.stats["current_connections"] = len(self.active_connections)
                
                # WebSocket 연결 정리 시도 (에러 무시)
                try:
                    if websocket.client_state != 3:  # not CLOSED
                        await websocket.close()
                except:
                    pass
                
                self.logger.info(f"🔌 WebSocket 연결 해제: {connection_id}")
            
        except Exception as e:
            self.logger.error(f"❌ WebSocket 연결 해제 실패: {connection_id} - {e}")

    async def send_personal_message(self, message: Dict[str, Any], connection_id: str) -> bool:
        """개인 메시지 전송 - IMPROVED VERSION"""
        try:
            if connection_id not in self.active_connections:
                return False
            
            websocket = self.active_connections[connection_id]
            
            # 안전한 메시지 전송
            success = await self._safe_send_message(websocket, message)
            
            if success:
                # 통계 및 히스토리 업데이트
                self.stats["messages_sent"] += 1
                
                if connection_id in self.connection_metadata:
                    self.connection_metadata[connection_id]["message_count"] += 1
                
                if self.enable_message_history:
                    self.message_history[connection_id].append({
                        **message,
                        "sent_at": time.time(),
                        "type": "outbound"
                    })
                    
                    # 히스토리 크기 제한
                    if len(self.message_history[connection_id]) > self.message_queue_size:
                        self.message_history[connection_id] = self.message_history[connection_id][-self.message_queue_size:]
            else:
                # 전송 실패시 연결 해제
                await self.disconnect(connection_id)
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ 개인 메시지 전송 실패: {connection_id} - {e}")
            self.stats["errors"] += 1
            await self.disconnect(connection_id)
            return False

    async def send_to_session(self, message: Dict[str, Any], session_id: str) -> int:
        """세션의 모든 연결에 메시지 전송"""
        try:
            if session_id not in self.session_connections:
                return 0
            
            success_count = 0
            failed_connections = []
            
            # 연결 ID 복사 (iteration 중 수정 방지)
            connection_ids = list(self.session_connections[session_id])
            
            for connection_id in connection_ids:
                success = await self.send_personal_message(message, connection_id)
                if success:
                    success_count += 1
                else:
                    failed_connections.append(connection_id)
            
            # 실패한 연결들은 이미 disconnect에서 정리됨
            return success_count
            
        except Exception as e:
            self.logger.error(f"❌ 세션 메시지 전송 실패: {session_id} - {e}")
            return 0

    async def broadcast(self, message: Dict[str, Any]) -> int:
        """모든 연결에 브로드캐스트"""
        try:
            success_count = 0
            failed_connections = []
            
            # 연결 ID 복사 (iteration 중 수정 방지)
            connection_ids = list(self.active_connections.keys())
            
            for connection_id in connection_ids:
                success = await self.send_personal_message(message, connection_id)
                if success:
                    success_count += 1
                else:
                    failed_connections.append(connection_id)
            
            return success_count
            
        except Exception as e:
            self.logger.error(f"❌ 브로드캐스트 실패: {e}")
            return 0

    async def receive_message(self, message: str, connection_id: str) -> Dict[str, Any]:
        """메시지 수신 처리 - IMPROVED VERSION"""
        try:
            # JSON 파싱
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                return {"type": "error", "message": "Invalid JSON format"}
            
            # 통계 업데이트
            self.stats["messages_received"] += 1
            
            # 메시지 히스토리 저장
            if self.enable_message_history:
                self.message_history[connection_id].append({
                    **data,
                    "received_at": time.time(),
                    "type": "inbound"
                })
            
            # 메시지 타입별 처리
            message_type = data.get("type", "unknown")
            
            if message_type == "ping":
                # 하트비트 업데이트
                if connection_id in self.connection_metadata:
                    self.connection_metadata[connection_id]["last_ping"] = time.time()
                
                return {
                    "type": "pong",
                    "timestamp": time.time(),
                    "connection_id": connection_id,
                    "server_time": datetime.now().isoformat()
                }
            
            elif message_type == "subscribe":
                # 세션 구독
                session_id = data.get("session_id")
                if session_id:
                    self.session_connections[session_id].add(connection_id)
                    if connection_id in self.connection_metadata:
                        self.connection_metadata[connection_id]["session_id"] = session_id
                    
                    return {
                        "type": "subscribed",
                        "session_id": session_id,
                        "message": f"세션 {session_id}에 구독되었습니다"
                    }
                else:
                    return {
                        "type": "error",
                        "message": "session_id가 필요합니다"
                    }
            
            elif message_type == "unsubscribe":
                # 세션 구독 해제
                session_id = data.get("session_id")
                if session_id and session_id in self.session_connections:
                    self.session_connections[session_id].discard(connection_id)
                    
                    if connection_id in self.connection_metadata:
                        self.connection_metadata[connection_id]["session_id"] = None
                    
                    return {
                        "type": "unsubscribed",
                        "session_id": session_id,
                        "message": f"세션 {session_id}에서 구독 해제되었습니다"
                    }
                else:
                    return {
                        "type": "error",
                        "message": "유효하지 않은 session_id입니다"
                    }
            
            elif message_type == "get_status":
                # 연결 상태 조회
                metadata = self.connection_metadata.get(connection_id, {})
                return {
                    "type": "status",
                    "connection_id": connection_id,
                    "metadata": metadata,
                    "stats": self.get_connection_stats(connection_id)
                }
            
            else:
                # 에코 응답
                return {
                    "type": "echo",
                    "original_message": data,
                    "processed_at": time.time(),
                    "message": f"메시지를 수신했습니다: {message_type}"
                }
            
        except Exception as e:
            self.logger.error(f"❌ 메시지 수신 처리 실패: {connection_id} - {e}")
            return {
                "type": "error",
                "message": f"메시지 처리 중 오류 발생: {str(e)}",
                "timestamp": time.time()
            }

    def get_connection_stats(self, connection_id: str) -> Dict[str, Any]:
        """연결 통계 조회"""
        metadata = self.connection_metadata.get(connection_id, {})
        connected_at = metadata.get("connected_at", time.time())
        
        return {
            "connected_at": connected_at,
            "uptime_seconds": time.time() - connected_at,
            "message_count": metadata.get("message_count", 0),
            "last_ping": metadata.get("last_ping"),
            "session_id": metadata.get("session_id"),
            "device": metadata.get("device"),
            "m3_max_optimized": metadata.get("m3_max_optimized", False)
        }

    def get_global_stats(self) -> Dict[str, Any]:
        """전역 통계 조회"""
        return {
            **self.stats,
            "active_sessions": len(self.session_connections),
            "total_message_history": sum(len(history) for history in self.message_history.values()),
            "device": self.device,
            "m3_max_optimized": self.is_m3_max,
            "connection_manager_features": {
                "max_connections": self.max_connections,
                "heartbeat_interval": self.heartbeat_interval,
                "message_history": self.enable_message_history,
                "compression": self.compression_enabled
            }
        }

    async def start_background_tasks(self):
        """백그라운드 태스크 시작"""
        try:
            # 하트비트 태스크
            if not self._heartbeat_task or self._heartbeat_task.done():
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                self._background_tasks.add(self._heartbeat_task)
            
            # 정리 태스크
            cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._background_tasks.add(cleanup_task)
            
            self.logger.info("🔄 WebSocket 백그라운드 태스크 시작됨")
            
        except Exception as e:
            self.logger.error(f"❌ 백그라운드 태스크 시작 실패: {e}")

    async def _heartbeat_loop(self):
        """하트비트 루프"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                current_time = time.time()
                disconnected_connections = []
                
                # 비활성 연결 확인
                for connection_id, metadata in list(self.connection_metadata.items()):
                    if connection_id not in self.active_connections:
                        continue
                        
                    last_ping = metadata.get("last_ping", metadata.get("connected_at", current_time))
                    
                    # 하트비트 간격의 3배 이상 응답 없으면 연결 해제
                    if current_time - last_ping > self.heartbeat_interval * 3:
                        disconnected_connections.append(connection_id)
                
                # 비활성 연결 정리
                for connection_id in disconnected_connections:
                    await self.disconnect(connection_id)
                    self.logger.info(f"⏰ 비활성 연결 정리: {connection_id}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ 하트비트 루프 오류: {e}")

    async def _cleanup_loop(self):
        """정리 루프"""
        while True:
            try:
                await asyncio.sleep(300)  # 5분마다 정리
                
                current_time = time.time()
                
                # 메타데이터 정리 (연결 끊어진지 1시간 후)
                expired_metadata = []
                for connection_id, metadata in self.connection_metadata.items():
                    if connection_id not in self.active_connections:
                        disconnect_time = metadata.get("disconnected_at", current_time)
                        if current_time - disconnect_time > 3600:  # 1시간
                            expired_metadata.append(connection_id)
                
                for connection_id in expired_metadata:
                    del self.connection_metadata[connection_id]
                    if connection_id in self.message_history:
                        del self.message_history[connection_id]
                
                if expired_metadata:
                    self.logger.info(f"🧹 메타데이터 정리: {len(expired_metadata)}개 항목")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ 정리 루프 오류: {e}")

class WebSocketRouter(OptimalRouterConstructor):
    """
    🍎 M3 Max 최적화 WebSocket 라우터 - FIXED VERSION
    최적 생성자 패턴 적용
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        ✅ 최적 생성자 - WebSocket 라우터 특화

        Args:
            device: 사용할 디바이스 (None=자동감지)
            config: WebSocket 라우터 설정
            **kwargs: 확장 파라미터들
        """
        super().__init__(device=device, config=config, **kwargs)
        
        # 연결 매니저 생성
        self.manager = ConnectionManager(
            device=self.device,
            config=self.config.get('connection_manager', {}),
            **kwargs
        )
        
        # FastAPI 라우터 생성
        self.router = APIRouter()
        self._setup_routes()
        
        self.logger.info(f"🔗 WebSocket 라우터 초기화 - {self.device}")
        
        # 초기화 완료
        self.is_initialized = True

    def _setup_routes(self):
        """라우터 엔드포인트 설정 - FIXED VERSION"""
        
        @self.router.websocket("/ws/pipeline-progress")
        async def websocket_pipeline_progress_legacy(websocket: WebSocket):
            """파이프라인 진행 상황 WebSocket - 기존 경로 호환"""
            connection_id = f"pipeline_{uuid.uuid4().hex[:8]}"
            
            if await self.manager.connect(websocket, connection_id):
                try:
                    while True:
                        data = await websocket.receive_text()
                        response = await self.manager.receive_message(data, connection_id)
                        await self.manager.send_personal_message(response, connection_id)
                        
                except WebSocketDisconnect:
                    self.logger.info(f"🔌 파이프라인 WebSocket 정상 종료: {connection_id}")
                except Exception as e:
                    self.logger.error(f"❌ 파이프라인 WebSocket 오류: {connection_id} - {e}")
                finally:
                    await self.manager.disconnect(connection_id)
        
        @self.router.websocket("/ws/pipeline-progress/{session_id}")
        async def websocket_pipeline_progress_with_session(websocket: WebSocket, session_id: str):
            """파이프라인 진행 상황 WebSocket - 세션별 (새 기능)"""
            connection_id = f"pipeline_{session_id}_{uuid.uuid4().hex[:8]}"
            
            if await self.manager.connect(websocket, connection_id, session_id):
                try:
                    while True:
                        data = await websocket.receive_text()
                        response = await self.manager.receive_message(data, connection_id)
                        await self.manager.send_personal_message(response, connection_id)
                        
                except WebSocketDisconnect:
                    self.logger.info(f"🔌 파이프라인 WebSocket 정상 종료: {connection_id}")
                except Exception as e:
                    self.logger.error(f"❌ 파이프라인 WebSocket 오류: {connection_id} - {e}")
                finally:
                    await self.manager.disconnect(connection_id)
        
        @self.router.websocket("/ws/system-monitor")
        async def websocket_system_monitor(websocket: WebSocket):
            """시스템 모니터링 WebSocket"""
            connection_id = f"monitor_{uuid.uuid4().hex[:8]}"
            
            if await self.manager.connect(websocket, connection_id):
                try:
                    # 주기적으로 시스템 상태 전송
                    while True:
                        system_stats = self.manager.get_global_stats()
                        await self.manager.send_personal_message({
                            "type": "system_stats",
                            "data": system_stats,
                            "timestamp": time.time()
                        }, connection_id)
                        
                        await asyncio.sleep(5)  # 5초마다 업데이트
                        
                except WebSocketDisconnect:
                    self.logger.info(f"🔌 시스템 모니터 WebSocket 정상 종료: {connection_id}")
                except Exception as e:
                    self.logger.error(f"❌ 시스템 모니터 WebSocket 오류: {connection_id} - {e}")
                finally:
                    await self.manager.disconnect(connection_id)
        
        @self.router.websocket("/ws/test")
        async def websocket_test(websocket: WebSocket):
            """WebSocket 테스트 엔드포인트"""
            connection_id = f"test_{uuid.uuid4().hex[:8]}"
            
            if await self.manager.connect(websocket, connection_id):
                try:
                    while True:
                        data = await websocket.receive_text()
                        
                        # 메시지 처리
                        response = await self.manager.receive_message(data, connection_id)
                        
                        # 테스트용 추가 정보
                        response.update({
                            "test_mode": True,
                            "device": self.device,
                            "m3_max_optimized": self.is_m3_max,
                            "connection_id": connection_id
                        })
                        
                        await self.manager.send_personal_message(response, connection_id)
                        
                except WebSocketDisconnect:
                    self.logger.info(f"🔌 테스트 WebSocket 정상 종료: {connection_id}")
                except Exception as e:
                    self.logger.error(f"❌ 테스트 WebSocket 오류: {connection_id} - {e}")
                finally:
                    await self.manager.disconnect(connection_id)
        
        @self.router.get("/debug")
        async def websocket_debug_page():
            """WebSocket 디버깅 페이지"""
            return HTMLResponse(content=self._get_debug_html())
        
        @self.router.get("/stats")
        async def get_websocket_stats():
            """WebSocket 통계"""
            return {
                "success": True,
                "stats": self.manager.get_global_stats(),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.router.post("/broadcast")
        async def broadcast_message(message: Dict[str, Any]):
            """브로드캐스트 메시지 전송"""
            try:
                sent_count = await self.manager.broadcast(message)
                return {
                    "success": True,
                    "message": "브로드캐스트 완료",
                    "sent_to": sent_count,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }

    def _get_debug_html(self) -> str:
        """WebSocket 디버깅 HTML 페이지 - IMPROVED VERSION"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>WebSocket Debug - M3 Max Optimized (Fixed)</title>
            <meta charset="utf-8">
            <style>
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; 
                    margin: 20px; 
                    background: #f5f5f7;
                }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ 
                    text-align: center; 
                    margin-bottom: 30px; 
                    padding: 20px; 
                    background: white; 
                    border-radius: 12px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .section {{ 
                    margin: 20px 0; 
                    padding: 20px; 
                    background: white;
                    border-radius: 12px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .status {{ 
                    padding: 15px; 
                    margin: 10px 0; 
                    border-radius: 8px; 
                    font-weight: 600;
                    text-align: center;
                }}
                .connected {{ background: #d4edda; color: #155724; border: 2px solid #28a745; }}
                .disconnected {{ background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }}
                .connecting {{ background: #fff3cd; color: #856404; border: 2px solid #ffc107; }}
                button {{ 
                    padding: 10px 20px; 
                    margin: 5px; 
                    border: none; 
                    border-radius: 8px; 
                    cursor: pointer; 
                    font-weight: 600;
                    transition: all 0.2s;
                }}
                .btn-primary {{ background: #007bff; color: white; }}
                .btn-primary:hover {{ background: #0056b3; transform: translateY(-1px); }}
                .btn-success {{ background: #28a745; color: white; }}
                .btn-success:hover {{ background: #1e7e34; transform: translateY(-1px); }}
                .btn-danger {{ background: #dc3545; color: white; }}
                .btn-danger:hover {{ background: #c82333; transform: translateY(-1px); }}
                .btn-warning {{ background: #ffc107; color: #212529; }}
                .btn-warning:hover {{ background: #e0a800; transform: translateY(-1px); }}
                textarea {{ 
                    width: 100%; 
                    height: 120px; 
                    margin: 10px 0; 
                    padding: 12px;
                    border: 1px solid #ced4da; 
                    border-radius: 8px;
                    font-family: 'Monaco', 'Menlo', monospace;
                    resize: vertical;
                }}
                .log {{ 
                    background: #f8f9fa; 
                    padding: 15px; 
                    height: 300px; 
                    overflow-y: scroll; 
                    font-family: 'Monaco', 'Menlo', monospace; 
                    border: 1px solid #dee2e6;
                    border-radius: 8px;
                    font-size: 13px;
                    line-height: 1.4;
                }}
                .m3-badge {{ 
                    background: linear-gradient(45deg, #ff6b6b, #ffa726); 
                    padding: 5px 15px; 
                    border-radius: 20px; 
                    color: white; 
                    font-size: 0.9em; 
                    font-weight: 600;
                    text-shadow: 0 1px 2px rgba(0,0,0,0.2);
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 15px 0;
                }}
                .stat-card {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                }}
                .stat-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #007bff;
                }}
                .stat-label {{
                    font-size: 12px;
                    color: #6c757d;
                    margin-top: 5px;
                }}
                .log-entry {{
                    margin: 3px 0;
                    padding: 2px 0;
                }}
                .log-send {{ color: #28a745; }}
                .log-receive {{ color: #007bff; }}
                .log-error {{ color: #dc3545; }}
                .log-success {{ color: #155724; }}
                .log-warn {{ color: #856404; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>
                        🔗 WebSocket Debug Console (Fixed Version)
                        {'<span class="m3-badge">🍎 M3 Max Optimized</span>' if self.is_m3_max else ''}
                    </h1>
                    <p>실시간 WebSocket 연결 테스트 및 디버깅</p>
                </div>
                
                <div class="section">
                    <h3>🔌 연결 상태</h3>
                    <div id="connectionStatus" class="status disconnected">연결되지 않음</div>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div id="statConnections" class="stat-value">0</div>
                            <div class="stat-label">활성 연결</div>
                        </div>
                        <div class="stat-card">
                            <div id="statMessages" class="stat-value">0</div>
                            <div class="stat-label">전송된 메시지</div>
                        </div>
                        <div class="stat-card">
                            <div id="statUptime" class="stat-value">0s</div>
                            <div class="stat-label">연결 시간</div>
                        </div>
                    </div>
                    <button class="btn-primary" onclick="connectWebSocket()">🔗 연결</button>
                    <button class="btn-danger" onclick="disconnectWebSocket()">🔌 연결 해제</button>
                    <button class="btn-warning" onclick="reconnectWebSocket()">🔄 재연결</button>
                </div>
                
                <div class="section">
                    <h3>📤 메시지 전송</h3>
                    <textarea id="messageInput" placeholder='{{
    "type": "test",
    "message": "Hello WebSocket!",
    "data": {{
        "test": true,
        "timestamp": "2025-01-15"
    }}
}}'></textarea>
                    <br>
                    <button class="btn-success" onclick="sendMessage()">📤 메시지 전송</button>
                    <button class="btn-primary" onclick="sendPing()">🏓 Ping</button>
                    <button class="btn-primary" onclick="getStatus()">📊 상태 조회</button>
                    <button class="btn-warning" onclick="sendSubscribe()">📡 세션 구독</button>
                </div>
                
                <div class="section">
                    <h3>📋 메시지 로그</h3>
                    <div id="messageLog" class="log"></div>
                    <button class="btn-danger" onclick="clearLog()">🗑️ 로그 지우기</button>
                    <button class="btn-primary" onclick="exportLog()">💾 로그 내보내기</button>
                </div>
                
                <div class="section">
                    <h3>ℹ️ 시스템 정보</h3>
                    <p><strong>디바이스:</strong> {self.device}</p>
                    <p><strong>M3 Max 최적화:</strong> {'✅ 활성화' if self.is_m3_max else '❌ 비활성화'}</p>
                    <p><strong>현재 시간:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>연결 관리자:</strong> 최적 생성자 패턴 적용</p>
                </div>
            </div>
            
            <script>
                let ws = null;
                let connectionId = null;
                let connectTime = null;
                let messageCount = 0;
                let uptimeInterval = null;
                
                function addLog(message, type = 'info') {{
                    const log = document.getElementById('messageLog');
                    const timestamp = new Date().toLocaleTimeString();
                    const logEntry = document.createElement('div');
                    logEntry.className = `log-entry log-${{type}}`;
                    logEntry.innerHTML = `[${{timestamp}}] [${{type.toUpperCase()}}] ${{message}}`;
                    log.appendChild(logEntry);
                    log.scrollTop = log.scrollHeight;
                }}
                
                function updateStatus(connected) {{
                    const status = document.getElementById('connectionStatus');
                    if (connected) {{
                        status.className = 'status connected';
                        status.textContent = '연결됨' + (connectionId ? ` (ID: ${{connectionId}})` : '');
                        connectTime = Date.now();
                        startUptimeCounter();
                    }} else {{
                        status.className = 'status disconnected';
                        status.textContent = '연결되지 않음';
                        connectionId = null;
                        connectTime = null;
                        stopUptimeCounter();
                        updateStats(0, messageCount, '0s');
                    }}
                }}
                
                function updateStats(connections, messages, uptime) {{
                    document.getElementById('statConnections').textContent = connections;
                    document.getElementById('statMessages').textContent = messages;
                    document.getElementById('statUptime').textContent = uptime;
                }}
                
                function startUptimeCounter() {{
                    if (uptimeInterval) clearInterval(uptimeInterval);
                    uptimeInterval = setInterval(() => {{
                        if (connectTime) {{
                            const uptime = Math.floor((Date.now() - connectTime) / 1000);
                            const minutes = Math.floor(uptime / 60);
                            const seconds = uptime % 60;
                            updateStats(1, messageCount, `${{minutes}}m ${{seconds}}s`);
                        }}
                    }}, 1000);
                }}
                
                function stopUptimeCounter() {{
                    if (uptimeInterval) {{
                        clearInterval(uptimeInterval);
                        uptimeInterval = null;
                    }}
                }}
                
                function connectWebSocket() {{
                    if (ws && ws.readyState === WebSocket.OPEN) {{
                        addLog('이미 연결되어 있습니다', 'warn');
                        return;
                    }}
                    
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${{protocol}}//${{window.location.host}}/api/ws/test`;
                    
                    addLog(`연결 시도: ${{wsUrl}}`, 'info');
                    updateStatus(false);
                    document.getElementById('connectionStatus').className = 'status connecting';
                    document.getElementById('connectionStatus').textContent = '연결 중...';
                    
                    ws = new WebSocket(wsUrl);
                    
                    ws.onopen = function(event) {{
                        addLog('WebSocket 연결 성공! 🎉', 'success');
                        updateStatus(true);
                    }};
                    
                    ws.onmessage = function(event) {{
                        try {{
                            const data = JSON.parse(event.data);
                            if (data.type === 'connection_established') {{
                                connectionId = data.connection_id;
                                updateStatus(true);
                                addLog(`연결 설정 완료 - Connection ID: ${{connectionId}}`, 'success');
                            }}
                            addLog(`수신: ${{JSON.stringify(data, null, 2)}}`, 'receive');
                        }} catch (e) {{
                            addLog(`수신 (원시): ${{event.data}}`, 'receive');
                        }}
                    }};
                    
                    ws.onclose = function(event) {{
                        addLog(`WebSocket 연결 종료 (코드: ${{event.code}}, 이유: ${{event.reason || '없음'}})`, 'warn');
                        updateStatus(false);
                    }};
                    
                    ws.onerror = function(error) {{
                        addLog(`WebSocket 오류: ${{error}}`, 'error');
                        updateStatus(false);
                    }};
                }}
                
                function disconnectWebSocket() {{
                    if (ws) {{
                        ws.close();
                        addLog('연결 해제 요청', 'info');
                    }} else {{
                        addLog('연결된 WebSocket이 없습니다', 'warn');
                    }}
                }}
                
                function reconnectWebSocket() {{
                    addLog('재연결 시도...', 'info');
                    disconnectWebSocket();
                    setTimeout(connectWebSocket, 1000);
                }}
                
                function sendMessage() {{
                    if (!ws || ws.readyState !== WebSocket.OPEN) {{
                        addLog('WebSocket이 연결되지 않음', 'error');
                        return;
                    }}
                    
                    const input = document.getElementById('messageInput');
                    const message = input.value.trim();
                    
                    if (!message) {{
                        addLog('메시지가 비어있습니다', 'warn');
                        return;
                    }}
                    
                    try {{
                        JSON.parse(message); // JSON 유효성 검사
                        ws.send(message);
                        messageCount++;
                        addLog(`전송: ${{message}}`, 'send');
                    }} catch (e) {{
                        addLog(`잘못된 JSON 형식: ${{e.message}}`, 'error');
                    }}
                }}
                
                function sendPing() {{
                    if (!ws || ws.readyState !== WebSocket.OPEN) {{
                        addLog('WebSocket이 연결되지 않음', 'error');
                        return;
                    }}
                    
                    const pingMessage = {{
                        type: 'ping',
                        timestamp: Date.now(),
                        client_time: new Date().toISOString()
                    }};
                    
                    ws.send(JSON.stringify(pingMessage));
                    messageCount++;
                    addLog('🏓 Ping 전송', 'send');
                }}
                
                function getStatus() {{
                    if (!ws || ws.readyState !== WebSocket.OPEN) {{
                        addLog('WebSocket이 연결되지 않음', 'error');
                        return;
                    }}
                    
                    const statusMessage = {{
                        type: 'get_status'
                    }};
                    
                    ws.send(JSON.stringify(statusMessage));
                    messageCount++;
                    addLog('📊 상태 조회 요청', 'send');
                }}
                
                function sendSubscribe() {{
                    if (!ws || ws.readyState !== WebSocket.OPEN) {{
                        addLog('WebSocket이 연결되지 않음', 'error');
                        return;
                    }}
                    
                    const sessionId = `test_session_${{Date.now()}}`;
                    const subscribeMessage = {{
                        type: 'subscribe',
                        session_id: sessionId
                    }};
                    
                    ws.send(JSON.stringify(subscribeMessage));
                    messageCount++;
                    addLog(`📡 세션 구독 요청: ${{sessionId}}`, 'send');
                }}
                
                function clearLog() {{
                    document.getElementById('messageLog').innerHTML = '';
                    addLog('로그가 지워졌습니다', 'info');
                }}
                
                function exportLog() {{
                    const log = document.getElementById('messageLog').innerText;
                    const blob = new Blob([log], {{ type: 'text/plain' }});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `websocket_log_${{new Date().toISOString().split('T')[0]}}.txt`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                    addLog('로그 파일 다운로드됨', 'success');
                }}
                
                // 페이지 로드 시 초기화
                window.onload = function() {{
                    addLog('🚀 WebSocket 디버그 콘솔 시작 (Fixed Version)', 'info');
                    updateStats(0, 0, '0s');
                }};
                
                // 페이지 언로드 시 정리
                window.onbeforeunload = function() {{
                    if (ws) {{
                        ws.close();
                    }}
                }};
            </script>
        </body>
        </html>
        """

# WebSocket 라우터 인스턴스 생성 (최적 생성자 패턴)
websocket_router = WebSocketRouter()
router = websocket_router.router
manager = websocket_router.manager

# 백그라운드 태스크 시작 함수
async def start_background_tasks():
    """WebSocket 백그라운드 태스크 시작"""
    await manager.start_background_tasks()

# 편의 함수들 (하위 호환성)
def create_websocket_router(
    device: Optional[str] = None,
    max_connections: int = 1000,
    **kwargs
) -> WebSocketRouter:
    """WebSocket 라우터 생성 (하위 호환)"""
    return WebSocketRouter(
        device=device,
        max_connections=max_connections,
        **kwargs
    )

# 모듈 익스포트
__all__ = [
    'router',
    'manager',
    'WebSocketRouter',
    'ConnectionManager',
    'OptimalRouterConstructor',
    'create_websocket_router',
    'start_background_tasks'
]