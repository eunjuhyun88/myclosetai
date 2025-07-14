"""
MyCloset AI Backend - 완전한 WebSocket 라우터 구현
프론트엔드 usePipeline과 완벽 호환
실시간 파이프라인 진행 상황 및 시스템 모니터링
"""

import asyncio
import json
import logging
import time
import uuid
import traceback
from typing import Dict, Any, Set, Optional, List, Callable
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
import psutil
import torch

# 프로젝트 내부 import
try:
    from app.models.schemas import (
        ProcessingStep, ProcessingStatusEnum, VirtualTryOnRequest,
        VirtualTryOnResponse, PipelineProgress, SystemHealth, PerformanceMetrics
    )
    from app.core.config import settings
    from app.ai_pipeline.pipeline_manager import PipelineManager
except ImportError as e:
    logging.warning(f"일부 모듈 import 실패: {e}")

logger = logging.getLogger(__name__)
router = APIRouter()

# ========================
# 연결 관리자 클래스
# ========================

class WebSocketConnectionManager:
    """
    고급 WebSocket 연결 관리자
    - 세션별 클라이언트 그룹화
    - 실시간 진행 상황 브로드캐스트
    - 연결 상태 모니터링
    - 자동 재연결 지원
    """
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_connections: Dict[str, Set[str]] = {}  # session_id -> client_ids
        self.client_sessions: Dict[str, str] = {}  # client_id -> session_id
        self.client_metadata: Dict[str, Dict[str, Any]] = {}  # 클라이언트 메타데이터
        self.message_history: Dict[str, List[Dict[str, Any]]] = {}  # 메시지 히스토리
        
        # 활성 파이프라인 추적
        self.active_pipelines: Dict[str, Dict[str, Any]] = {}
        
        # 연결 통계
        self.connection_stats = {
            "total_connections": 0,
            "current_connections": 0,
            "total_messages": 0,
            "start_time": time.time()
        }
    
    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None) -> str:
        """클라이언트 연결 설정"""
        await websocket.accept()
        
        if not client_id:
            client_id = str(uuid.uuid4())
        
        # 연결 등록
        self.active_connections[client_id] = websocket
        self.client_metadata[client_id] = {
            "connected_at": datetime.now().isoformat(),
            "last_activity": time.time(),
            "message_count": 0,
            "user_agent": "unknown",
            "ip_address": "unknown"
        }
        
        # 통계 업데이트
        self.connection_stats["total_connections"] += 1
        self.connection_stats["current_connections"] = len(self.active_connections)
        
        logger.info(f"✅ WebSocket 클라이언트 연결: {client_id}")
        
        # 환영 메시지 전송
        welcome_message = {
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": time.time(),
            "server_info": {
                "api_version": "2.0.0",
                "pipeline_version": "2.0.0",
                "supported_features": [
                    "real_time_progress",
                    "system_monitoring", 
                    "session_tracking",
                    "error_recovery",
                    "message_history"
                ],
                "available_endpoints": [
                    "/api/ws/pipeline-progress",
                    "/api/ws/system-monitor",
                    "/api/ws/test"
                ]
            },
            "message": "MyCloset AI WebSocket 연결이 설정되었습니다."
        }
        
        await self.send_personal_message(welcome_message, client_id)
        return client_id
    
    def disconnect(self, client_id: str):
        """클라이언트 연결 해제"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        # 세션에서 제거
        if client_id in self.client_sessions:
            session_id = self.client_sessions[client_id]
            if session_id in self.session_connections:
                self.session_connections[session_id].discard(client_id)
                if not self.session_connections[session_id]:
                    del self.session_connections[session_id]
            del self.client_sessions[client_id]
        
        # 메타데이터 정리
        if client_id in self.client_metadata:
            del self.client_metadata[client_id]
        
        # 통계 업데이트
        self.connection_stats["current_connections"] = len(self.active_connections)
        
        logger.info(f"❌ WebSocket 클라이언트 연결 해제: {client_id}")
    
    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        """개별 클라이언트에게 메시지 전송"""
        if client_id not in self.active_connections:
            logger.warning(f"⚠️ 비활성 클라이언트에게 메시지 시도: {client_id}")
            return False
        
        try:
            # 메시지에 타임스탬프 추가
            if "timestamp" not in message:
                message["timestamp"] = time.time()
            
            # 전송
            await self.active_connections[client_id].send_text(json.dumps(message))
            
            # 통계 및 메타데이터 업데이트
            self.connection_stats["total_messages"] += 1
            if client_id in self.client_metadata:
                self.client_metadata[client_id]["message_count"] += 1
                self.client_metadata[client_id]["last_activity"] = time.time()
            
            # 메시지 히스토리 저장 (최근 100개만)
            if client_id not in self.message_history:
                self.message_history[client_id] = []
            
            self.message_history[client_id].append({
                "timestamp": message.get("timestamp"),
                "type": message.get("type", "unknown"),
                "size": len(json.dumps(message))
            })
            
            # 히스토리 크기 제한
            if len(self.message_history[client_id]) > 100:
                self.message_history[client_id] = self.message_history[client_id][-100:]
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 메시지 전송 실패 to {client_id}: {e}")
            self.disconnect(client_id)
            return False
    
    async def broadcast_to_session(self, message: Dict[str, Any], session_id: str):
        """세션별 브로드캐스트"""
        if session_id not in self.session_connections:
            logger.warning(f"⚠️ 존재하지 않는 세션: {session_id}")
            return 0
        
        success_count = 0
        failed_clients = []
        
        for client_id in list(self.session_connections[session_id]):
            if await self.send_personal_message(message, client_id):
                success_count += 1
            else:
                failed_clients.append(client_id)
        
        # 실패한 클라이언트 정리
        for client_id in failed_clients:
            self.disconnect(client_id)
        
        logger.debug(f"📡 세션 {session_id} 브로드캐스트: {success_count}명 성공")
        return success_count
    
    async def broadcast_to_all(self, message: Dict[str, Any]):
        """전체 브로드캐스트"""
        success_count = 0
        failed_clients = []
        
        for client_id in list(self.active_connections.keys()):
            if await self.send_personal_message(message, client_id):
                success_count += 1
            else:
                failed_clients.append(client_id)
        
        # 실패한 클라이언트 정리
        for client_id in failed_clients:
            self.disconnect(client_id)
        
        logger.debug(f"📡 전체 브로드캐스트: {success_count}명 성공")
        return success_count
    
    def subscribe_to_session(self, client_id: str, session_id: str):
        """세션 구독"""
        if session_id not in self.session_connections:
            self.session_connections[session_id] = set()
        
        # 기존 구독 해제
        if client_id in self.client_sessions:
            old_session = self.client_sessions[client_id]
            if old_session in self.session_connections:
                self.session_connections[old_session].discard(client_id)
        
        # 새 구독 설정
        self.session_connections[session_id].add(client_id)
        self.client_sessions[client_id] = session_id
        
        logger.info(f"📡 클라이언트 {client_id}가 세션 {session_id}에 구독")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """연결 통계 조회"""
        uptime = time.time() - self.connection_stats["start_time"]
        
        return {
            "current_connections": len(self.active_connections),
            "total_connections": self.connection_stats["total_connections"],
            "active_sessions": len(self.session_connections),
            "total_messages": self.connection_stats["total_messages"],
            "uptime_seconds": uptime,
            "uptime_formatted": f"{uptime // 3600:.0f}h {(uptime % 3600) // 60:.0f}m {uptime % 60:.0f}s",
            "session_details": {
                session_id: len(clients) 
                for session_id, clients in self.session_connections.items()
            },
            "active_pipelines": len(self.active_pipelines)
        }

# 글로벌 연결 관리자
manager = WebSocketConnectionManager()

# ========================
# 진행 상황 콜백 함수들
# ========================

def create_progress_callback(session_id: str) -> Callable[[Dict[str, Any]], None]:
    """파이프라인 진행 상황 콜백 생성"""
    
    async def progress_callback(progress_data: Dict[str, Any]):
        """진행 상황 업데이트 브로드캐스트"""
        try:
            # 진행 상황 메시지 구성
            message = {
                "type": "pipeline_progress",
                "session_id": session_id,
                "data": progress_data,
                "timestamp": time.time()
            }
            
            # 세션 구독자들에게 브로드캐스트
            await manager.broadcast_to_session(message, session_id)
            
            logger.debug(f"📊 진행 상황 업데이트: {session_id} - {progress_data.get('current_step', 'unknown')}")
            
        except Exception as e:
            logger.error(f"❌ 진행 상황 콜백 오류: {e}")
    
    return progress_callback

def create_step_callback(session_id: str) -> Callable[[str, Dict[str, Any]], None]:
    """단계별 진행 상황 콜백 생성"""
    
    async def step_callback(step_name: str, step_data: Dict[str, Any]):
        """단계별 진행 상황 브로드캐스트"""
        try:
            message = {
                "type": "step_update",
                "session_id": session_id,
                "step_name": step_name,
                "step_data": step_data,
                "timestamp": time.time()
            }
            
            await manager.broadcast_to_session(message, session_id)
            
            logger.info(f"🔄 단계 업데이트: {session_id} - {step_name}")
            
        except Exception as e:
            logger.error(f"❌ 단계 콜백 오류: {e}")
    
    return step_callback

# ========================
# 메시지 핸들러
# ========================

async def handle_client_message(message: Dict[str, Any], client_id: str):
    """클라이언트 메시지 처리"""
    try:
        message_type = message.get("type", "unknown")
        
        if message_type == "ping":
            await handle_ping(client_id)
        
        elif message_type == "subscribe_session":
            session_id = message.get("session_id")
            if session_id:
                manager.subscribe_to_session(client_id, session_id)
                await manager.send_personal_message({
                    "type": "subscription_confirmed",
                    "session_id": session_id
                }, client_id)
        
        elif message_type == "get_stats":
            stats = manager.get_connection_stats()
            await manager.send_personal_message({
                "type": "stats_response",
                "data": stats
            }, client_id)
        
        elif message_type == "get_system_info":
            system_info = await get_system_info()
            await manager.send_personal_message({
                "type": "system_info_response",
                "data": system_info
            }, client_id)
        
        else:
            logger.warning(f"⚠️ 알 수 없는 메시지 타입: {message_type} from {client_id}")
            await manager.send_personal_message({
                "type": "error",
                "error": f"Unknown message type: {message_type}"
            }, client_id)
            
    except Exception as e:
        logger.error(f"❌ 메시지 처리 오류: {e}")
        await manager.send_personal_message({
            "type": "error",
            "error": "Message processing failed"
        }, client_id)

async def handle_ping(client_id: str):
    """Ping 응답"""
    await manager.send_personal_message({
        "type": "pong",
        "timestamp": time.time()
    }, client_id)

# ========================
# 시스템 정보 수집
# ========================

async def get_system_info() -> Dict[str, Any]:
    """시스템 정보 수집"""
    try:
        # CPU 정보
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # 메모리 정보
        memory = psutil.virtual_memory()
        
        # GPU 정보 (가능한 경우)
        gpu_info = {}
        try:
            if torch.cuda.is_available():
                gpu_info = {
                    "available": True,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(),
                    "memory_allocated": torch.cuda.memory_allocated(),
                    "memory_reserved": torch.cuda.memory_reserved()
                }
            elif torch.backends.mps.is_available():
                gpu_info = {
                    "available": True,
                    "type": "MPS (Apple Silicon)",
                    "device_count": 1
                }
            else:
                gpu_info = {"available": False}
        except:
            gpu_info = {"available": False, "error": "GPU info unavailable"}
        
        return {
            "cpu": {
                "usage_percent": cpu_percent,
                "core_count": cpu_count
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            },
            "gpu": gpu_info,
            "connections": manager.get_connection_stats(),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ 시스템 정보 수집 실패: {e}")
        return {"error": str(e), "timestamp": time.time()}

# ========================
# WebSocket 엔드포인트들
# ========================

@router.websocket("/pipeline-progress")
async def pipeline_progress_websocket(websocket: WebSocket):
    """파이프라인 진행 상황 WebSocket"""
    client_id = None
    try:
        client_id = await manager.connect(websocket)
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            await handle_client_message(message, client_id)
            
    except WebSocketDisconnect:
        if client_id:
            manager.disconnect(client_id)
        logger.info(f"🔌 파이프라인 WebSocket 연결 해제: {client_id}")
    except Exception as e:
        logger.error(f"❌ 파이프라인 WebSocket 오류: {e}")
        if client_id:
            manager.disconnect(client_id)

@router.websocket("/system-monitor")
async def system_monitor_websocket(websocket: WebSocket):
    """시스템 모니터링 WebSocket"""
    client_id = None
    try:
        client_id = await manager.connect(websocket)
        
        # 시스템 정보 주기적 전송 시작
        asyncio.create_task(send_periodic_system_info(client_id))
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            await handle_client_message(message, client_id)
            
    except WebSocketDisconnect:
        if client_id:
            manager.disconnect(client_id)
        logger.info(f"🔌 시스템 모니터 WebSocket 연결 해제: {client_id}")
    except Exception as e:
        logger.error(f"❌ 시스템 모니터 WebSocket 오류: {e}")
        if client_id:
            manager.disconnect(client_id)

@router.websocket("/test")
async def test_websocket(websocket: WebSocket):
    """테스트용 WebSocket"""
    client_id = None
    try:
        client_id = await manager.connect(websocket)
        
        # 테스트 메시지 전송
        await manager.send_personal_message({
            "type": "test_message",
            "message": "WebSocket 테스트 연결이 성공했습니다!",
            "connection_info": manager.get_connection_stats()
        }, client_id)
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 에코 응답
            echo_response = {
                "type": "echo_response",
                "original_message": message,
                "server_timestamp": time.time()
            }
            await manager.send_personal_message(echo_response, client_id)
            
    except WebSocketDisconnect:
        if client_id:
            manager.disconnect(client_id)
        logger.info(f"🔌 테스트 WebSocket 연결 해제: {client_id}")
    except Exception as e:
        logger.error(f"❌ 테스트 WebSocket 오류: {e}")
        if client_id:
            manager.disconnect(client_id)

# ========================
# 백그라운드 태스크
# ========================

async def send_periodic_system_info(client_id: str, interval: int = 10):
    """주기적 시스템 정보 전송"""
    while client_id in manager.active_connections:
        try:
            system_info = await get_system_info()
            await manager.send_personal_message({
                "type": "periodic_system_info",
                "data": system_info
            }, client_id)
            
            await asyncio.sleep(interval)
            
        except Exception as e:
            logger.error(f"❌ 주기적 시스템 정보 전송 실패: {e}")
            break

async def system_monitor():
    """전역 시스템 모니터링"""
    while True:
        try:
            # 비활성 연결 정리
            inactive_clients = []
            current_time = time.time()
            
            for client_id, metadata in manager.client_metadata.items():
                if current_time - metadata["last_activity"] > 300:  # 5분 이상 비활성
                    inactive_clients.append(client_id)
            
            for client_id in inactive_clients:
                logger.info(f"🧹 비활성 클라이언트 정리: {client_id}")
                manager.disconnect(client_id)
            
            # 30초마다 체크
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"❌ 시스템 모니터링 오류: {e}")
            await asyncio.sleep(60)

# ========================
# 디버깅 및 관리 엔드포인트
# ========================

@router.get("/debug")
async def websocket_debug():
    """WebSocket 디버깅 페이지"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>MyCloset AI WebSocket Debug</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .panel { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { display: inline-block; padding: 4px 8px; border-radius: 4px; color: white; font-weight: bold; }
        .connected { background: #4CAF50; }
        .disconnected { background: #f44336; }
        button { background: #2196F3; color: white; border: none; padding: 10px 20px; margin: 5px; border-radius: 4px; cursor: pointer; }
        button:hover { background: #1976D2; }
        textarea { width: 100%; height: 300px; font-family: monospace; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .endpoint { background: #e3f2fd; padding: 10px; margin: 5px 0; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 MyCloset AI WebSocket 디버깅 도구</h1>
        
        <div class="panel">
            <h2>연결 상태</h2>
            <div>
                <span>파이프라인 진행 상황: </span>
                <span id="status-pipeline" class="status disconnected">연결 해제됨</span>
            </div>
            <div>
                <span>시스템 모니터: </span>
                <span id="status-system" class="status disconnected">연결 해제됨</span>
            </div>
            <div>
                <span>테스트: </span>
                <span id="status-test" class="status disconnected">연결 해제됨</span>
            </div>
        </div>
        
        <div class="panel">
            <h2>사용 가능한 엔드포인트</h2>
            <div class="endpoint">
                <strong>파이프라인 진행 상황:</strong> ws://localhost:8000/api/ws/pipeline-progress
            </div>
            <div class="endpoint">
                <strong>시스템 모니터:</strong> ws://localhost:8000/api/ws/system-monitor
            </div>
            <div class="endpoint">
                <strong>테스트:</strong> ws://localhost:8000/api/ws/test
            </div>
        </div>
        
        <div class="grid">
            <div class="panel">
                <h2>연결 관리</h2>
                <button onclick="connectAll()">모든 연결 시작</button>
                <button onclick="disconnectAll()">모든 연결 해제</button>
                <button onclick="sendTestMessage()">테스트 메시지 전송</button>
                <button onclick="clearMessages()">메시지 지우기</button>
                <button onclick="exportLogs()">로그 내보내기</button>
            </div>
            
            <div class="panel">
                <h2>시뮬레이션</h2>
                <button onclick="simulatePipelineProgress()">파이프라인 진행 시뮬레이션</button>
                <button onclick="subscribeToSession()">세션 구독 (test-session)</button>
                <button onclick="getStats()">연결 통계 요청</button>
                <button onclick="getSystemInfo()">시스템 정보 요청</button>
            </div>
        </div>
        
        <div class="panel">
            <h2>실시간 메시지</h2>
            <textarea id="messages" readonly placeholder="WebSocket 메시지가 여기에 표시됩니다..."></textarea>
        </div>
    </div>

    <script>
        let ws_pipeline = null;
        let ws_system = null;
        let ws_test = null;
        let messageLog = [];
        
        function updateStatus(endpoint, connected) {
            const element = document.getElementById(`status-${endpoint}`);
            if (connected) {
                element.textContent = '연결됨';
                element.className = 'status connected';
            } else {
                element.textContent = '연결 해제됨';
                element.className = 'status disconnected';
            }
        }
        
        function logMessage(endpoint, message) {
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = `[${timestamp}] [${endpoint}] ${JSON.stringify(message, null, 2)}`;
            messageLog.push(logEntry);
            
            const textarea = document.getElementById('messages');
            textarea.value += logEntry + '\\n\\n';
            textarea.scrollTop = textarea.scrollHeight;
        }
        
        function connectPipeline() {
            ws_pipeline = new WebSocket('ws://localhost:8000/api/ws/pipeline-progress');
            
            ws_pipeline.onopen = () => {
                updateStatus('pipeline', true);
                logMessage('PIPELINE', {type: 'connection_opened'});
            };
            
            ws_pipeline.onmessage = (event) => {
                const message = JSON.parse(event.data);
                logMessage('PIPELINE', message);
            };
            
            ws_pipeline.onclose = () => {
                updateStatus('pipeline', false);
                logMessage('PIPELINE', {type: 'connection_closed'});
            };
            
            ws_pipeline.onerror = (error) => {
                logMessage('PIPELINE', {type: 'error', error: error});
            };
        }
        
        function connectSystem() {
            ws_system = new WebSocket('ws://localhost:8000/api/ws/system-monitor');
            
            ws_system.onopen = () => {
                updateStatus('system', true);
                logMessage('SYSTEM', {type: 'connection_opened'});
            };
            
            ws_system.onmessage = (event) => {
                const message = JSON.parse(event.data);
                logMessage('SYSTEM', message);
            };
            
            ws_system.onclose = () => {
                updateStatus('system', false);
                logMessage('SYSTEM', {type: 'connection_closed'});
            };
        }
        
        function connectTest() {
            ws_test = new WebSocket('ws://localhost:8000/api/ws/test');
            
            ws_test.onopen = () => {
                updateStatus('test', true);
                logMessage('TEST', {type: 'connection_opened'});
            };
            
            ws_test.onmessage = (event) => {
                const message = JSON.parse(event.data);
                logMessage('TEST', message);
            };
            
            ws_test.onclose = () => {
                updateStatus('test', false);
                logMessage('TEST', {type: 'connection_closed'});
            };
        }
        
        function connectAll() {
            connectPipeline();
            connectSystem();
            connectTest();
        }
        
        function disconnectAll() {
            if (ws_pipeline) ws_pipeline.close();
            if (ws_system) ws_system.close();
            if (ws_test) ws_test.close();
        }
        
        function sendTestMessage() {
            if (ws_test && ws_test.readyState === WebSocket.OPEN) {
                const message = {
                    type: 'test',
                    data: 'Hello from debug client!',
                    timestamp: Date.now()
                };
                ws_test.send(JSON.stringify(message));
                logMessage('TEST', {type: 'sent', message: message});
            }
        }
        
        function subscribeToSession() {
            if (ws_pipeline && ws_pipeline.readyState === WebSocket.OPEN) {
                const message = {
                    type: 'subscribe_session',
                    session_id: 'test-session'
                };
                ws_pipeline.send(JSON.stringify(message));
                logMessage('PIPELINE', {type: 'sent', message: message});
            }
        }
        
        function getStats() {
            if (ws_pipeline && ws_pipeline.readyState === WebSocket.OPEN) {
                const message = {type: 'get_stats'};
                ws_pipeline.send(JSON.stringify(message));
                logMessage('PIPELINE', {type: 'sent', message: message});
            }
        }
        
        function getSystemInfo() {
            if (ws_system && ws_system.readyState === WebSocket.OPEN) {
                const message = {type: 'get_system_info'};
                ws_system.send(JSON.stringify(message));
                logMessage('SYSTEM', {type: 'sent', message: message});
            }
        }
        
        function clearMessages() {
            document.getElementById('messages').value = '';
            messageLog = [];
        }
        
        function exportLogs() {
            const logs = messageLog.join('\\n');
            const blob = new Blob([logs], {type: 'text/plain'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `websocket_debug_logs_${new Date().getTime()}.txt`;
            a.click();
            URL.revokeObjectURL(url);
        }
        
        // 자동 연결
        setTimeout(connectAll, 1000);
    </script>
</body>
</html>
    """)

# ========================
# 백그라운드 태스크 시작
# ========================

async def start_background_tasks():
    """백그라운드 태스크 시작"""
    asyncio.create_task(system_monitor())
    logger.info("✅ WebSocket 백그라운드 태스크 시작됨")

# ========================
# 모듈 exports
# ========================

__all__ = [
    'router', 
    'manager', 
    'create_progress_callback', 
    'create_step_callback',
    'start_background_tasks'
]