"""
MyCloset AI Backend - 실시간 WebSocket 라우터
통합된 스키마와 pipeline_manager 완전 호환
8단계 파이프라인의 실시간 진행 상황을 클라이언트에 전송
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, Set, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from app.models.schemas import (
    ProcessingStep, ProcessingStatusEnum, create_processing_steps, 
    update_processing_step_status, SystemHealth, PerformanceMetrics
)
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# ========================
# WebSocket 연결 관리자
# ========================

class ConnectionManager:
    """WebSocket 연결 관리자 - 통합된 스키마 사용"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_connections: Dict[str, Set[str]] = {}  # session_id -> client_ids
        self.client_sessions: Dict[str, str] = {}  # client_id -> session_id
        
    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None) -> str:
        """클라이언트 연결"""
        await websocket.accept()
        
        if not client_id:
            client_id = str(uuid.uuid4())
        
        self.active_connections[client_id] = websocket
        logger.info(f"✅ WebSocket 클라이언트 연결: {client_id}")
        
        # 연결 확인 메시지 전송
        await self.send_personal_message({
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": time.time(),
            "message": "WebSocket 연결이 설정되었습니다.",
            "server_info": {
                "api_version": "1.0.0",
                "pipeline_version": "2.0.0",
                "supported_features": ["real_time_progress", "system_monitoring", "session_tracking"]
            }
        }, client_id)
        
        return client_id
    
    def disconnect(self, client_id: str):
        """클라이언트 연결 해제"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
        # 세션 연결에서도 제거
        if client_id in self.client_sessions:
            session_id = self.client_sessions[client_id]
            if session_id in self.session_connections:
                self.session_connections[session_id].discard(client_id)
                if not self.session_connections[session_id]:
                    del self.session_connections[session_id]
            del self.client_sessions[client_id]
        
        logger.info(f"❌ WebSocket 클라이언트 연결 해제: {client_id}")
    
    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        """특정 클라이언트에게 메시지 전송"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"❌ 메시지 전송 실패 to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast_to_session(self, message: Dict[str, Any], session_id: str):
        """특정 세션을 구독하는 모든 클라이언트에게 브로드캐스트"""
        if session_id in self.session_connections:
            disconnected_clients = []
            
            for client_id in list(self.session_connections[session_id]):
                try:
                    await self.send_personal_message(message, client_id)
                except Exception as e:
                    logger.error(f"❌ 브로드캐스트 실패 to {client_id}: {e}")
                    disconnected_clients.append(client_id)
            
            # 연결 끊어진 클라이언트 제거
            for client_id in disconnected_clients:
                self.disconnect(client_id)
    
    async def broadcast_to_all(self, message: Dict[str, Any]):
        """모든 활성 클라이언트에게 브로드캐스트"""
        disconnected_clients = []
        
        for client_id in list(self.active_connections.keys()):
            try:
                await self.send_personal_message(message, client_id)
            except Exception as e:
                logger.error(f"❌ 전체 브로드캐스트 실패 to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # 연결 끊어진 클라이언트 제거
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    def subscribe_to_session(self, client_id: str, session_id: str):
        """클라이언트를 특정 세션에 구독"""
        if session_id not in self.session_connections:
            self.session_connections[session_id] = set()
        
        # 기존 구독 정리
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
        return {
            "active_connections": len(self.active_connections),
            "active_sessions": len(self.session_connections),
            "session_subscribers": {
                session_id: len(clients) 
                for session_id, clients in self.session_connections.items()
            },
            "total_subscribers": sum(len(clients) for clients in self.session_connections.values())
        }

# 글로벌 연결 관리자
manager = ConnectionManager()

# ========================
# WebSocket 엔드포인트
# ========================

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket 연결 엔드포인트"""
    try:
        actual_client_id = await manager.connect(websocket, client_id)
        
        while True:
            # 클라이언트로부터 메시지 수신
            data = await websocket.receive_text()
            message = json.loads(data)
            
            await handle_client_message(message, actual_client_id)
            
    except WebSocketDisconnect:
        manager.disconnect(actual_client_id)
        logger.info(f"🔌 클라이언트 {actual_client_id} 정상 연결 해제")
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON 파싱 오류 from {client_id}: {e}")
        await manager.send_personal_message({
            "type": "error",
            "error": "Invalid JSON format",
            "timestamp": time.time()
        }, client_id)
    except Exception as e:
        logger.error(f"❌ WebSocket 오류 {client_id}: {e}")
        manager.disconnect(client_id)

async def handle_client_message(message: Dict[str, Any], client_id: str):
    """클라이언트 메시지 처리 - 통합된 스키마 사용"""
    message_type = message.get("type")
    
    try:
        if message_type == "subscribe_session":
            # 세션 구독
            session_id = message.get("session_id")
            if session_id:
                manager.subscribe_to_session(client_id, session_id)
                await manager.send_personal_message({
                    "type": "subscription_confirmed",
                    "session_id": session_id,
                    "timestamp": time.time(),
                    "message": f"세션 {session_id}에 구독되었습니다"
                }, client_id)
            else:
                await manager.send_personal_message({
                    "type": "error",
                    "error": "session_id is required",
                    "timestamp": time.time()
                }, client_id)
        
        elif message_type == "get_system_status":
            # 시스템 상태 요청
            status = await get_system_status()
            await manager.send_personal_message({
                "type": "system_status",
                "data": status,
                "timestamp": time.time()
            }, client_id)
        
        elif message_type == "get_session_status":
            # 세션 상태 요청
            session_id = message.get("session_id")
            if session_id:
                # 실제 세션 상태를 가져와야 함 (메인 앱에서)
                from app.main import processing_sessions
                
                if session_id in processing_sessions:
                    session_data = processing_sessions[session_id]
                    await manager.send_personal_message({
                        "type": "session_status",
                        "session_id": session_id,
                        "data": session_data,
                        "timestamp": time.time()
                    }, client_id)
                else:
                    await manager.send_personal_message({
                        "type": "error",
                        "error": f"Session {session_id} not found",
                        "timestamp": time.time()
                    }, client_id)
        
        elif message_type == "ping":
            # Ping 응답
            await manager.send_personal_message({
                "type": "pong",
                "timestamp": time.time(),
                "server_time": time.time()
            }, client_id)
        
        elif message_type == "get_connection_stats":
            # 연결 통계 요청
            stats = manager.get_connection_stats()
            await manager.send_personal_message({
                "type": "connection_stats",
                "data": stats,
                "timestamp": time.time()
            }, client_id)
        
        else:
            await manager.send_personal_message({
                "type": "error",
                "error": f"Unknown message type: {message_type}",
                "timestamp": time.time()
            }, client_id)
            
    except Exception as e:
        logger.error(f"❌ 메시지 처리 오류: {e}")
        await manager.send_personal_message({
            "type": "error",
            "error": f"Message processing failed: {str(e)}",
            "timestamp": time.time()
        }, client_id)

async def get_system_status() -> Dict[str, Any]:
    """시스템 상태 조회 - 통합된 스키마 사용"""
    try:
        # 메인 앱에서 파이프라인 매니저 가져오기
        from app.main import pipeline_manager, processing_sessions
        
        if pipeline_manager:
            pipeline_status = await pipeline_manager.get_pipeline_status()
            memory_usage = pipeline_manager._get_detailed_memory_usage()
        else:
            pipeline_status = {"initialized": False, "error": "Pipeline not available"}
            memory_usage = {}
        
        system_health = SystemHealth(
            overall_status="healthy" if pipeline_manager and pipeline_manager.is_initialized else "degraded",
            pipeline_initialized=pipeline_manager.is_initialized if pipeline_manager else False,
            device_available=True,
            memory_usage=memory_usage,
            active_sessions=len(processing_sessions),
            error_rate=0.0,  # 계산 필요
            uptime=time.time(),  # 시작 시간부터 계산 필요
            pipeline_ready=pipeline_manager.is_initialized if pipeline_manager else False
        )
        
        return {
            "pipeline_status": pipeline_status,
            "system_health": system_health.dict(),
            "connection_stats": manager.get_connection_stats(),
            "processing_sessions": len(processing_sessions),
            "device": pipeline_manager.device if pipeline_manager else "unknown"
        }
        
    except Exception as e:
        logger.error(f"❌ 시스템 상태 조회 실패: {e}")
        return {
            "error": str(e),
            "timestamp": time.time(),
            "status": "error"
        }

# ========================
# 진행 상황 콜백 함수
# ========================

def create_progress_callback(session_id: str):
    """세션별 진행 상황 콜백 함수 생성 - 통합된 스키마 사용"""
    
    # 초기 처리 단계들 생성
    processing_steps = create_processing_steps()
    
    async def progress_callback(stage: str, percentage: int):
        """진행 상황 WebSocket으로 전송"""
        try:
            # 현재 단계 확인
            current_step = None
            for step in processing_steps:
                if stage.lower().replace(" ", "_").replace("-", "_") in step.id:
                    current_step = step
                    break
            
            # 단계 상태 업데이트
            if current_step:
                processing_steps = update_processing_step_status(
                    processing_steps, 
                    current_step.id, 
                    "processing" if percentage < 100 else "completed",
                    percentage
                )
            
            # WebSocket으로 진행 상황 전송
            progress_message = {
                "type": "pipeline_progress",
                "session_id": session_id,
                "stage": stage,
                "percentage": percentage,
                "current_step": current_step.dict() if current_step else None,
                "all_steps": [step.dict() for step in processing_steps],
                "timestamp": time.time()
            }
            
            await manager.broadcast_to_session(progress_message, session_id)
            
            # 로그 출력
            logger.info(f"📊 세션 {session_id} 진행: {stage} - {percentage}%")
            
        except Exception as e:
            logger.error(f"❌ 진행 상황 전송 실패: {e}")
    
    return progress_callback

def create_step_callback(session_id: str, step_id: str):
    """특정 단계별 콜백 함수 생성"""
    
    async def step_callback(status: str, progress: int = 0, error_message: str = None):
        """단계별 상태 업데이트"""
        try:
            step_message = {
                "type": "step_update",
                "session_id": session_id,
                "step_id": step_id,
                "status": status,
                "progress": progress,
                "error_message": error_message,
                "timestamp": time.time()
            }
            
            await manager.broadcast_to_session(step_message, session_id)
            
        except Exception as e:
            logger.error(f"❌ 단계 업데이트 전송 실패: {e}")
    
    return step_callback

# ========================
# 시스템 모니터링
# ========================

async def system_monitor():
    """시스템 상태 주기적 브로드캐스트"""
    while True:
        try:
            if manager.active_connections:
                status = await get_system_status()
                
                # 시스템 모니터링 메시지 브로드캐스트
                await manager.broadcast_to_all({
                    "type": "system_monitor",
                    "data": status,
                    "timestamp": time.time(),
                    "interval": 30  # 30초마다
                })
            
            await asyncio.sleep(30)  # 30초마다 업데이트
            
        except Exception as e:
            logger.error(f"❌ 시스템 모니터링 오류: {e}")
            await asyncio.sleep(10)

# ========================
# REST API 엔드포인트들
# ========================

@router.get("/connections/status")
async def get_connections_status():
    """현재 WebSocket 연결 상태"""
    stats = manager.get_connection_stats()
    return {
        "success": True,
        "data": stats,
        "timestamp": time.time()
    }

@router.post("/notify/session/{session_id}")
async def notify_session_subscribers(session_id: str, message: Dict[str, Any]):
    """특정 세션 구독자들에게 알림 전송"""
    try:
        notification = {
            "type": "manual_notification",
            "session_id": session_id,
            "data": message,
            "timestamp": time.time()
        }
        
        await manager.broadcast_to_session(notification, session_id)
        
        subscriber_count = len(manager.session_connections.get(session_id, set()))
        
        return {
            "success": True, 
            "notified_clients": subscriber_count,
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"❌ 세션 알림 전송 실패: {e}")
        return {"success": False, "error": str(e)}

@router.post("/broadcast")
async def broadcast_message(message: Dict[str, Any]):
    """모든 클라이언트에게 브로드캐스트"""
    try:
        notification = {
            "type": "broadcast",
            "data": message,
            "timestamp": time.time()
        }
        
        await manager.broadcast_to_all(notification)
        
        return {
            "success": True,
            "notified_clients": len(manager.active_connections)
        }
        
    except Exception as e:
        logger.error(f"❌ 브로드캐스트 실패: {e}")
        return {"success": False, "error": str(e)}

# ========================
# 테스트 페이지
# ========================

@router.get("/test")
async def websocket_test_page():
    """WebSocket 테스트 페이지 - 통합된 스키마 호환"""
    return HTMLResponse(f"""
<!DOCTYPE html>
<html>
<head>
    <title>MyCloset AI WebSocket Test</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif; margin: 20px; background: #f8f9fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .status {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }}
        .message {{ padding: 8px 12px; margin: 4px 0; border-radius: 4px; font-size: 14px; }}
        .success {{ background: #d4edda; color: #155724; border-left: 4px solid #28a745; }}
        .error {{ background: #f8d7da; color: #721c24; border-left: 4px solid #dc3545; }}
        .info {{ background: #d1ecf1; color: #0c5460; border-left: 4px solid #17a2b8; }}
        .progress {{ background: #fff3cd; color: #856404; border-left: 4px solid #ffc107; }}
        button {{ padding: 8px 16px; margin: 4px; cursor: pointer; border: none; border-radius: 4px; background: #007bff; color: white; }}
        button:hover {{ background: #0056b3; }}
        button:disabled {{ background: #6c757d; cursor: not-allowed; }}
        input {{ padding: 8px; margin: 4px; border: 1px solid #ced4da; border-radius: 4px; width: 200px; }}
        #messages {{ height: 400px; overflow-y: auto; border: 1px solid #dee2e6; padding: 10px; background: white; border-radius: 4px; }}
        .step-progress {{ margin: 5px 0; padding: 8px; background: #f8f9fa; border-radius: 4px; }}
        .progress-bar {{ width: 100%; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background: #28a745; transition: width 0.3s ease; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 MyCloset AI WebSocket Test</h1>
            <p>실시간 가상 피팅 진행 상황 테스트</p>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>🔌 연결 상태</h3>
                <div class="status">
                    <p><strong>Status:</strong> <span id="status">Disconnected</span></p>
                    <p><strong>Client ID:</strong> <span id="clientId">-</span></p>
                    <p><strong>Server:</strong> ws://localhost:{settings.PORT}</p>
                </div>
                <button onclick="connect()">Connect</button>
                <button onclick="disconnect()">Disconnect</button>
                <button onclick="getSystemStatus()">시스템 상태</button>
            </div>
            
            <div class="card">
                <h3>🎮 테스트 명령</h3>
                <div>
                    <input type="text" id="sessionId" placeholder="Session ID" value="test_session_123">
                    <button onclick="subscribeToSession()">세션 구독</button>
                </div>
                <div>
                    <button onclick="ping()">Ping Test</button>
                    <button onclick="getConnectionStats()">연결 통계</button>
                    <button onclick="simulateProgress()">진행률 시뮬레이션</button>
                </div>
            </div>
        </div>
        
        <div class="card" style="margin-top: 20px;">
            <h3>📊 가상 피팅 진행 상황</h3>
            <div id="progressSteps"></div>
        </div>
        
        <div class="card" style="margin-top: 20px;">
            <h3>📨 실시간 메시지</h3>
            <div id="messages"></div>
            <div style="margin-top: 10px;">
                <button onclick="clearMessages()">Clear Messages</button>
                <button onclick="exportLogs()">Export Logs</button>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let clientId = null;
        let messageLog = [];
        
        const steps = [
            {{ id: 'upload', name: '이미지 업로드', progress: 0 }},
            {{ id: 'human_parsing', name: '인체 분석', progress: 0 }},
            {{ id: 'pose_estimation', name: '포즈 추정', progress: 0 }},
            {{ id: 'cloth_segmentation', name: '의류 분석', progress: 0 }},
            {{ id: 'geometric_matching', name: '기하학적 매칭', progress: 0 }},
            {{ id: 'cloth_warping', name: '의류 변형', progress: 0 }},
            {{ id: 'virtual_fitting', name: '가상 피팅', progress: 0 }},
            {{ id: 'post_processing', name: '품질 향상', progress: 0 }},
            {{ id: 'quality_assessment', name: '품질 평가', progress: 0 }}
        ];
        
        function connect() {{
            const wsUrl = `ws://localhost:{settings.PORT}/api/ws/${{Date.now()}}_test`;
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function(event) {{
                document.getElementById('status').textContent = 'Connected';
                document.getElementById('status').style.color = '#28a745';
                addMessage('🟢 WebSocket 연결됨', 'success');
            }};
            
            ws.onmessage = function(event) {{
                const message = JSON.parse(event.data);
                handleMessage(message);
            }};
            
            ws.onclose = function(event) {{
                document.getElementById('status').textContent = 'Disconnected';
                document.getElementById('status').style.color = '#dc3545';
                addMessage('🔴 WebSocket 연결 종료', 'error');
            }};
            
            ws.onerror = function(error) {{
                addMessage('❌ WebSocket 오류: ' + error, 'error');
            }};
        }}
        
        function disconnect() {{
            if (ws) {{
                ws.close();
            }}
        }}
        
        function handleMessage(message) {{
            messageLog.push({{ ...message, clientTimestamp: Date.now() }});
            const type = message.type;
            
            if (type === 'connection_established') {{
                clientId = message.client_id;
                document.getElementById('clientId').textContent = clientId;
                addMessage(`✅ 연결 설정됨 - Client ID: ${{clientId}}`, 'success');
            }} else if (type === 'pipeline_progress') {{
                addMessage(`⚡ [${{message.percentage.toFixed(1)}}%] ${{message.stage}}`, 'progress');
                updateProgressSteps(message.all_steps || []);
            }} else if (type === 'step_update') {{
                addMessage(`🔄 단계 업데이트: ${{message.step_id}} - ${{message.status}} (${{message.progress}}%)`, 'progress');
            }} else if (type === 'system_status') {{
                const status = message.data;
                addMessage(`📊 시스템: Pipeline ${{status.pipeline_status?.initialized ? '✅' : '❌'}}, 활성 세션: ${{status.processing_sessions}}`, 'info');
            }} else if (type === 'error') {{
                addMessage(`❌ 오류: ${{message.error}}`, 'error');
            }} else {{
                addMessage(`📨 ${{type}}: ${{JSON.stringify(message, null, 2)}}`, 'info');
            }}
        }}
        
        function updateProgressSteps(stepData) {{
            const container = document.getElementById('progressSteps');
            container.innerHTML = '';
            
            stepData.forEach((step, index) => {{
                const div = document.createElement('div');
                div.className = 'step-progress';
                div.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span>${{index + 1}}. ${{step.name}}</span>
                        <span>${{step.progress}}% - ${{step.status}}</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${{step.progress}}%"></div>
                    </div>
                `;
                container.appendChild(div);
            }});
        }}
        
        function addMessage(text, type = 'info') {{
            const messages = document.getElementById('messages');
            const div = document.createElement('div');
            div.className = `message ${{type}}`;
            div.innerHTML = `<strong>[${{new Date().toLocaleTimeString()}}]</strong> ${{text}}`;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
        }}
        
        function sendMessage(message) {{
            if (ws && ws.readyState === WebSocket.OPEN) {{
                ws.send(JSON.stringify(message));
            }} else {{
                addMessage('❌ WebSocket이 연결되지 않음', 'error');
            }}
        }}
        
        function subscribeToSession() {{
            const sessionId = document.getElementById('sessionId').value;
            sendMessage({{
                type: 'subscribe_session',
                session_id: sessionId
            }});
        }}
        
        function getSystemStatus() {{
            sendMessage({{ type: 'get_system_status' }});
        }}
        
        function ping() {{
            sendMessage({{ type: 'ping' }});
        }}
        
        function getConnectionStats() {{
            sendMessage({{ type: 'get_connection_stats' }});
        }}
        
        function simulateProgress() {{
            const sessionId = document.getElementById('sessionId').value;
            let step = 0;
            const stepNames = [
                '이미지 업로드', '인체 분석', '포즈 추정', '의류 분석',
                '기하학적 매칭', '의류 변형', '가상 피팅', '품질 향상', '품질 평가'
            ];
            
            const interval = setInterval(() => {{
                if (step >= stepNames.length) {{
                    clearInterval(interval);
                    return;
                }}
                
                const progress = ((step + 1) / stepNames.length) * 100;
                const simulatedMessage = {{
                    type: 'pipeline_progress',
                    session_id: sessionId,
                    stage: stepNames[step],
                    percentage: progress,
                    all_steps: steps.map((s, i) => ({{
                        ...s,
                        status: i < step ? 'completed' : i === step ? 'processing' : 'pending',
                        progress: i < step ? 100 : i === step ? 50 : 0
                    }})),
                    timestamp: Date.now() / 1000
                }};
                
                handleMessage(simulatedMessage);
                step++;
            }}, 1000);
        }}
        
        function clearMessages() {{
            document.getElementById('messages').innerHTML = '';
            messageLog = [];
        }}
        
        function exportLogs() {{
            const logs = JSON.stringify(messageLog, null, 2);
            const blob = new Blob([logs], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `websocket_logs_${{new Date().getTime()}}.json`;
            a.click();
            URL.revokeObjectURL(url);
        }}
        
        // 자동 연결
        connect();
        
        // 주기적 ping (30초마다)
        setInterval(() => {{
            if (ws && ws.readyState === WebSocket.OPEN) {{
                ping();
            }}
        }}, 30000);
    </script>
</body>
</html>
    """)

# ========================
# 백그라운드 태스크
# ========================

async def start_background_tasks():
    """백그라운드 태스크 시작"""
    asyncio.create_task(system_monitor())
    logger.info("✅ WebSocket 백그라운드 태스크 시작됨")

# 연결 관리자 및 진행 콜백 내보내기
__all__ = [
    'router', 
    'manager', 
    'create_progress_callback', 
    'create_step_callback',
    'start_background_tasks'
]