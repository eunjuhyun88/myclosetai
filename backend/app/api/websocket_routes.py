# app/api/websocket_routes.py
"""
WebSocket API 라우터 - M3 Max 최적화 (최적 생성자 패턴 적용)
실시간 통신 및 진행 상황 업데이트
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
    🍎 M3 Max 최적화 WebSocket 연결 매니저
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
                - max_connections: int = 1000  # 최대 동시 연결 수
                - heartbeat_interval: float = 30.0  # 하트비트 간격
                - message_queue_size: int = 100  # 메시지 큐 크기
                - enable_message_history: bool = True  # 메시지 히스토리
                - compression_enabled: bool = True  # 압축 활성화
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
            self.max_connections = 2000  # M3 Max는 더 많은 연결 처리 가능
            self.heartbeat_interval = 15.0  # 더 빈번한 하트비트
        
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
        """WebSocket 연결"""
        try:
            # 연결 수 제한 확인
            if len(self.active_connections) >= self.max_connections:
                await websocket.close(code=1008, reason="Connection limit exceeded")
                return False
            
            # WebSocket 연결 수락
            await websocket.accept()
            
            # 연결 등록
            self.active_connections[connection_id] = websocket
            
            # 세션 연결 매핑
            if session_id:
                self.session_connections[session_id].add(connection_id)
            
            # 메타데이터 저장
            self.connection_metadata[connection_id] = {
                "session_id": session_id,
                "connected_at": time.time(),
                "last_ping": time.time(),
                "message_count": 0,
                "device": self.device,
                "m3_max_optimized": self.is_m3_max
            }
            
            # 통계 업데이트
            self.stats["total_connections"] += 1
            self.stats["current_connections"] = len(self.active_connections)
            
            # 환영 메시지 전송
            await self.send_personal_message({
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
            }, connection_id)
            
            self.logger.info(f"🔗 WebSocket 연결 성공: {connection_id} (세션: {session_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ WebSocket 연결 실패: {e}")
            self.stats["errors"] += 1
            return False

    async def disconnect(self, connection_id: str):
        """WebSocket 연결 해제"""
        try:
            if connection_id in self.active_connections:
                # 연결 제거
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
                    del self.connection_metadata[connection_id]
                
                # 메시지 히스토리 정리 (선택적)
                if not self.enable_message_history and connection_id in self.message_history:
                    del self.message_history[connection_id]
                
                # 통계 업데이트
                self.stats["disconnections"] += 1
                self.stats["current_connections"] = len(self.active_connections)
                
                self.logger.info(f"🔌 WebSocket 연결 해제: {connection_id}")
            
        except Exception as e:
            self.logger.error(f"❌ WebSocket 연결 해제 실패: {e}")

    async def send_personal_message(self, message: Dict[str, Any], connection_id: str) -> bool:
        """개인 메시지 전송"""
        try:
            if connection_id not in self.active_connections:
                return False
            
            websocket = self.active_connections[connection_id]
            
            # 메시지 전송
            await websocket.send_text(json.dumps(message))
            
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
            
            return True
            
        except WebSocketDisconnect:
            await self.disconnect(connection_id)
            return False
        except Exception as e:
            self.logger.error(f"❌ 개인 메시지 전송 실패: {e}")
            self.stats["errors"] += 1
            return False

    async def send_to_session(self, message: Dict[str, Any], session_id: str) -> int:
        """세션의 모든 연결에 메시지 전송"""
        try:
            if session_id not in self.session_connections:
                return 0
            
            success_count = 0
            failed_connections = []
            
            for connection_id in self.session_connections[session_id]:
                success = await self.send_personal_message(message, connection_id)
                if success:
                    success_count += 1
                else:
                    failed_connections.append(connection_id)
            
            # 실패한 연결들 정리
            for connection_id in failed_connections:
                await self.disconnect(connection_id)
            
            return success_count
            
        except Exception as e:
            self.logger.error(f"❌ 세션 메시지 전송 실패: {e}")
            return 0

    async def broadcast(self, message: Dict[str, Any]) -> int:
        """모든 연결에 브로드캐스트"""
        try:
            success_count = 0
            failed_connections = []
            
            for connection_id in list(self.active_connections.keys()):
                success = await self.send_personal_message(message, connection_id)
                if success:
                    success_count += 1
                else:
                    failed_connections.append(connection_id)
            
            # 실패한 연결들 정리
            for connection_id in failed_connections:
                await self.disconnect(connection_id)
            
            return success_count
            
        except Exception as e:
            self.logger.error(f"❌ 브로드캐스트 실패: {e}")
            return 0

    async def broadcast_to_session(self, message: Dict[str, Any], session_id: str) -> int:
        """세션별 브로드캐스트 (편의 메서드)"""
        return await self.send_to_session(message, session_id)

    async def receive_message(self, message: str, connection_id: str) -> Dict[str, Any]:
        """메시지 수신 처리"""
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
                    "connection_id": connection_id
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
            
            elif message_type == "unsubscribe":
                # 세션 구독 해제
                session_id = data.get("session_id")
                if session_id and session_id in self.session_connections:
                    self.session_connections[session_id].discard(connection_id)
                    
                    return {
                        "type": "unsubscribed",
                        "session_id": session_id,
                        "message": f"세션 {session_id}에서 구독 해제되었습니다"
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
                return {
                    "type": "echo",
                    "original_message": data,
                    "timestamp": time.time()
                }
            
        except Exception as e:
            self.logger.error(f"❌ 메시지 수신 처리 실패: {e}")
            return {
                "type": "error",
                "message": f"메시지 처리 중 오류 발생: {str(e)}"
            }

    def get_connection_stats(self, connection_id: str) -> Dict[str, Any]:
        """연결 통계 조회"""
        metadata = self.connection_metadata.get(connection_id, {})
        
        return {
            "connected_at": metadata.get("connected_at"),
            "uptime_seconds": time.time() - metadata.get("connected_at", time.time()),
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
                for connection_id, metadata in self.connection_metadata.items():
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
                
                # 메시지 히스토리 정리
                if self.enable_message_history:
                    current_time = time.time()
                    
                    for connection_id in list(self.message_history.keys()):
                        # 연결이 끊어진 경우의 히스토리 정리
                        if connection_id not in self.active_connections:
                            # 1시간 후 히스토리 삭제
                            metadata = self.connection_metadata.get(connection_id)
                            if metadata:
                                disconnect_time = metadata.get("disconnected_at", current_time)
                                if current_time - disconnect_time > 3600:
                                    del self.message_history[connection_id]
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ 정리 루프 오류: {e}")

class WebSocketRouter(OptimalRouterConstructor):
    """
    🍎 M3 Max 최적화 WebSocket 라우터
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
        """라우터 엔드포인트 설정"""
        
        @self.router.websocket("/pipeline-progress")
        async def websocket_pipeline_progress(websocket: WebSocket):
            """파이프라인 진행 상황 WebSocket"""
            connection_id = f"pipeline_{uuid.uuid4().hex[:8]}"
            
            if await self.manager.connect(websocket, connection_id):
                try:
                    while True:
                        data = await websocket.receive_text()
                        response = await self.manager.receive_message(data, connection_id)
                        await self.manager.send_personal_message(response, connection_id)
                        
                except WebSocketDisconnect:
                    await self.manager.disconnect(connection_id)
        
        @self.router.websocket("/system-monitor")
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
                    await self.manager.disconnect(connection_id)
        
        @self.router.websocket("/test")
        async def websocket_test(websocket: WebSocket):
            """WebSocket 테스트 엔드포인트"""
            connection_id = f"test_{uuid.uuid4().hex[:8]}"
            
            if await self.manager.connect(websocket, connection_id):
                try:
                    while True:
                        data = await websocket.receive_text()
                        
                        # 에코 응답
                        response = {
                            "type": "test_echo",
                            "original_data": data,
                            "connection_id": connection_id,
                            "timestamp": time.time(),
                            "device": self.device,
                            "m3_max_optimized": self.is_m3_max
                        }
                        
                        await self.manager.send_personal_message(response, connection_id)
                        
                except WebSocketDisconnect:
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
        """WebSocket 디버깅 HTML 페이지"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>WebSocket Debug - M3 Max Optimized</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
                .status {{ padding: 10px; margin: 10px 0; border-radius: 4px; }}
                .connected {{ background: #d4edda; color: #155724; }}
                .disconnected {{ background: #f8d7da; color: #721c24; }}
                button {{ padding: 8px 16px; margin: 5px; border: none; border-radius: 4px; cursor: pointer; }}
                .btn-primary {{ background: #007bff; color: white; }}
                .btn-success {{ background: #28a745; color: white; }}
                .btn-danger {{ background: #dc3545; color: white; }}
                textarea {{ width: 100%; height: 100px; margin: 10px 0; }}
                .log {{ background: #f8f9fa; padding: 10px; height: 200px; overflow-y: scroll; font-family: monospace; }}
                .m3-badge {{ background: linear-gradient(45deg, #ff6b6b, #ffa726); padding: 3px 10px; border-radius: 15px; color: white; font-size: 0.8em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>
                    🔗 WebSocket Debug Console
                    {'<span class="m3-badge">🍎 M3 Max Optimized</span>' if self.is_m3_max else ''}
                </h1>
                
                <div class="section">
                    <h3>연결 상태</h3>
                    <div id="connectionStatus" class="status disconnected">연결되지 않음</div>
                    <button class="btn-primary" onclick="connectWebSocket()">연결</button>
                    <button class="btn-danger" onclick="disconnectWebSocket()">연결 해제</button>
                </div>
                
                <div class="section">
                    <h3>메시지 전송</h3>
                    <textarea id="messageInput" placeholder='{{\"type\": \"test\", \"message\": \"Hello WebSocket!\"}}'></textarea>
                    <br>
                    <button class="btn-success" onclick="sendMessage()">메시지 전송</button>
                    <button class="btn-primary" onclick="sendPing()">Ping</button>
                    <button class="btn-primary" onclick="getStatus()">상태 조회</button>
                </div>
                
                <div class="section">
                    <h3>메시지 로그</h3>
                    <div id="messageLog" class="log"></div>
                    <button class="btn-danger" onclick="clearLog()">로그 지우기</button>
                </div>
                
                <div class="section">
                    <h3>시스템 정보</h3>
                    <p><strong>디바이스:</strong> {self.device}</p>
                    <p><strong>M3 Max 최적화:</strong> {'✅ 활성화' if self.is_m3_max else '❌ 비활성화'}</p>
                    <p><strong>현재 시간:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
            
            <script>
                let ws = null;
                let connectionId = null;
                
                function addLog(message, type = 'info') {{
                    const log = document.getElementById('messageLog');
                    const timestamp = new Date().toLocaleTimeString();
                    const logEntry = document.createElement('div');
                    logEntry.innerHTML = `[${{timestamp}}] [${{type.toUpperCase()}}] ${{message}}`;
                    log.appendChild(logEntry);
                    log.scrollTop = log.scrollHeight;
                }}
                
                function updateStatus(connected) {{
                    const status = document.getElementById('connectionStatus');
                    if (connected) {{
                        status.className = 'status connected';
                        status.textContent = '연결됨 (Connection ID: ' + connectionId + ')';
                    }} else {{
                        status.className = 'status disconnected';
                        status.textContent = '연결되지 않음';
                        connectionId = null;
                    }}
                }}
                
                function connectWebSocket() {{
                    if (ws && ws.readyState === WebSocket.OPEN) {{
                        addLog('이미 연결되어 있습니다', 'warn');
                        return;
                    }}
                    
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${{protocol}}//${{window.location.host}}/api/ws/test`;
                    
                    ws = new WebSocket(wsUrl);
                    
                    ws.onopen = function(event) {{
                        addLog('WebSocket 연결 성공', 'success');
                        updateStatus(true);
                    }};
                    
                    ws.onmessage = function(event) {{
                        try {{
                            const data = JSON.parse(event.data);
                            if (data.type === 'connection_established') {{
                                connectionId = data.connection_id;
                                updateStatus(true);
                            }}
                            addLog('수신: ' + JSON.stringify(data, null, 2), 'receive');
                        }} catch (e) {{
                            addLog('수신 (원시): ' + event.data, 'receive');
                        }}
                    }};
                    
                    ws.onclose = function(event) {{
                        addLog('WebSocket 연결 종료 (코드: ' + event.code + ')', 'warn');
                        updateStatus(false);
                    }};
                    
                    ws.onerror = function(error) {{
                        addLog('WebSocket 오류: ' + error, 'error');
                    }};
                }}
                
                function disconnectWebSocket() {{
                    if (ws) {{
                        ws.close();
                        addLog('연결 해제 요청', 'info');
                    }}
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
                        addLog('전송: ' + message, 'send');
                    }} catch (e) {{
                        addLog('잘못된 JSON 형식: ' + e.message, 'error');
                    }}
                }}
                
                function sendPing() {{
                    if (!ws || ws.readyState !== WebSocket.OPEN) {{
                        addLog('WebSocket이 연결되지 않음', 'error');
                        return;
                    }}
                    
                    const pingMessage = {{
                        type: 'ping',
                        timestamp: Date.now()
                    }};
                    
                    ws.send(JSON.stringify(pingMessage));
                    addLog('Ping 전송', 'send');
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
                    addLog('상태 조회 요청', 'send');
                }}
                
                function clearLog() {{
                    document.getElementById('messageLog').innerHTML = '';
                }}
                
                // 페이지 로드 시 자동 연결
                window.onload = function() {{
                    addLog('WebSocket 디버그 콘솔 시작', 'info');
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