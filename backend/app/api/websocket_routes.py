"""
실시간 처리 상태 WebSocket 라우터
8단계 파이프라인의 실시간 진행 상황을 클라이언트에 전송
"""
import asyncio
import json
import logging
import time
from typing import Dict, Any, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uuid

from ..ai_pipeline.pipeline_manager import get_pipeline_manager
from ..core.gpu_config import GPUConfig

logger = logging.getLogger(__name__)
router = APIRouter()

# 연결된 WebSocket 클라이언트들 관리
class ConnectionManager:
    """WebSocket 연결 관리자"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.process_connections: Dict[str, Set[str]] = {}  # process_id -> client_ids
        
    async def connect(self, websocket: WebSocket) -> str:
        """클라이언트 연결"""
        await websocket.accept()
        client_id = str(uuid.uuid4())
        self.active_connections[client_id] = websocket
        logger.info(f"✅ WebSocket 클라이언트 연결: {client_id}")
        return client_id
    
    def disconnect(self, client_id: str):
        """클라이언트 연결 해제"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
        # 프로세스 연결에서도 제거
        for process_id, client_ids in self.process_connections.items():
            client_ids.discard(client_id)
        
        logger.info(f"❌ WebSocket 클라이언트 연결 해제: {client_id}")
    
    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        """특정 클라이언트에게 메시지 전송"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"메시지 전송 실패 to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast_to_process(self, message: Dict[str, Any], process_id: str):
        """특정 프로세스를 구독하는 모든 클라이언트에게 브로드캐스트"""
        if process_id in self.process_connections:
            disconnected_clients = []
            
            for client_id in self.process_connections[process_id]:
                try:
                    await self.send_personal_message(message, client_id)
                except:
                    disconnected_clients.append(client_id)
            
            # 연결 끊어진 클라이언트 제거
            for client_id in disconnected_clients:
                self.disconnect(client_id)
    
    def subscribe_to_process(self, client_id: str, process_id: str):
        """클라이언트를 특정 프로세스에 구독"""
        if process_id not in self.process_connections:
            self.process_connections[process_id] = set()
        self.process_connections[process_id].add(client_id)
        logger.info(f"📡 클라이언트 {client_id}가 프로세스 {process_id}에 구독")

# 글로벌 연결 관리자
manager = ConnectionManager()

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket 연결 엔드포인트"""
    client_id = await manager.connect(websocket)
    
    try:
        # 초기 연결 확인 메시지
        await manager.send_personal_message({
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": time.time(),
            "message": "WebSocket 연결이 설정되었습니다."
        }, client_id)
        
        while True:
            # 클라이언트로부터 메시지 수신
            data = await websocket.receive_text()
            message = json.loads(data)
            
            await handle_client_message(message, client_id)
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket 오류: {e}")
        manager.disconnect(client_id)

async def handle_client_message(message: Dict[str, Any], client_id: str):
    """클라이언트 메시지 처리"""
    message_type = message.get("type")
    
    if message_type == "subscribe_process":
        # 프로세스 구독
        process_id = message.get("process_id")
        if process_id:
            manager.subscribe_to_process(client_id, process_id)
            await manager.send_personal_message({
                "type": "subscription_confirmed",
                "process_id": process_id,
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
    
    elif message_type == "ping":
        # Ping 응답
        await manager.send_personal_message({
            "type": "pong",
            "timestamp": time.time()
        }, client_id)

async def get_system_status() -> Dict[str, Any]:
    """시스템 상태 조회"""
    try:
        pipeline_manager = get_pipeline_manager()
        gpu_config = GPUConfig()
        
        return {
            "pipeline": await pipeline_manager.get_pipeline_status(),
            "memory": gpu_config.get_memory_info(),
            "active_connections": len(manager.active_connections),
            "active_processes": len(manager.process_connections)
        }
    except Exception as e:
        logger.error(f"시스템 상태 조회 실패: {e}")
        return {"error": str(e)}

# 진행 상황 콜백 함수 생성기
def create_progress_callback(process_id: str):
    """프로세스별 진행 상황 콜백 함수 생성"""
    
    async def progress_callback(step: int, message: str, progress: float):
        """진행 상황 WebSocket으로 전송"""
        try:
            await manager.broadcast_to_process({
                "type": "pipeline_progress",
                "process_id": process_id,
                "step": step,
                "step_name": get_step_name(step),
                "message": message,
                "progress": progress,
                "timestamp": time.time()
            }, process_id)
        except Exception as e:
            logger.error(f"진행 상황 전송 실패: {e}")
    
    return progress_callback

def get_step_name(step: int) -> str:
    """스텝 번호를 이름으로 변환"""
    step_names = {
        1: "인체 파싱 (20개 부위)",
        2: "포즈 추정 (18개 키포인트)",
        3: "의류 세그멘테이션 (배경 제거)",
        4: "기하학적 매칭 (TPS 변환)",
        5: "옷 워핑 (신체에 맞춰 변형)",
        6: "가상 피팅 생성 (HR-VITON/ACGPN)",
        7: "후처리 (품질 향상)",
        8: "품질 평가 (자동 스코어링)"
    }
    return step_names.get(step, f"단계 {step}")

# 시스템 모니터링 백그라운드 태스크
async def system_monitor():
    """시스템 상태 주기적 브로드캐스트"""
    while True:
        try:
            if manager.active_connections:
                status = await get_system_status()
                
                # 모든 활성 클라이언트에게 시스템 상태 전송
                for client_id in list(manager.active_connections.keys()):
                    await manager.send_personal_message({
                        "type": "system_monitor",
                        "data": status,
                        "timestamp": time.time()
                    }, client_id)
            
            await asyncio.sleep(10)  # 10초마다 업데이트
            
        except Exception as e:
            logger.error(f"시스템 모니터링 오류: {e}")
            await asyncio.sleep(5)

# REST API 엔드포인트들
@router.get("/status/connections")
async def get_connections_status():
    """현재 WebSocket 연결 상태"""
    return {
        "active_connections": len(manager.active_connections),
        "active_processes": len(manager.process_connections),
        "process_subscribers": {
            process_id: len(clients) 
            for process_id, clients in manager.process_connections.items()
        }
    }

@router.post("/notify/process/{process_id}")
async def notify_process_subscribers(process_id: str, message: Dict[str, Any]):
    """특정 프로세스 구독자들에게 알림 전송"""
    await manager.broadcast_to_process({
        "type": "manual_notification",
        "process_id": process_id,
        "data": message,
        "timestamp": time.time()
    }, process_id)
    
    return {"success": True, "notified_clients": len(manager.process_connections.get(process_id, set()))}

@router.get("/test")
async def websocket_test_page():
    """WebSocket 테스트 페이지"""
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>MyCloset AI WebSocket Test</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        .status { background: #f0f0f0; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .message { background: #e8f5e8; padding: 10px; margin: 5px 0; border-radius: 3px; }
        .error { background: #ffe8e8; }
        .progress { background: #e8f0ff; }
        button { padding: 10px 15px; margin: 5px; cursor: pointer; }
        input { padding: 8px; margin: 5px; width: 200px; }
        #messages { height: 400px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 MyCloset AI WebSocket Test</h1>
        
        <div class="status">
            <h3>연결 상태</h3>
            <p>Status: <span id="status">Disconnected</span></p>
            <p>Client ID: <span id="clientId">-</span></p>
            <button onclick="connect()">Connect</button>
            <button onclick="disconnect()">Disconnect</button>
        </div>
        
        <div class="status">
            <h3>테스트 명령</h3>
            <button onclick="getSystemStatus()">시스템 상태 조회</button>
            <button onclick="ping()">Ping</button>
            <br>
            <input type="text" id="processId" placeholder="Process ID" value="test_process_123">
            <button onclick="subscribeToProcess()">프로세스 구독</button>
        </div>
        
        <div class="status">
            <h3>📨 메시지</h3>
            <div id="messages"></div>
            <button onclick="clearMessages()">Clear</button>
        </div>
    </div>

    <script>
        let ws = null;
        let clientId = null;
        
        function connect() {
            const wsUrl = `ws://localhost:8000/api/ws/test_client`;
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function(event) {
                document.getElementById('status').textContent = 'Connected';
                addMessage('🟢 WebSocket 연결됨', 'success');
            };
            
            ws.onmessage = function(event) {
                const message = JSON.parse(event.data);
                handleMessage(message);
            };
            
            ws.onclose = function(event) {
                document.getElementById('status').textContent = 'Disconnected';
                addMessage('🔴 WebSocket 연결 종료', 'error');
            };
            
            ws.onerror = function(error) {
                addMessage('❌ WebSocket 오류: ' + error, 'error');
            };
        }
        
        function disconnect() {
            if (ws) {
                ws.close();
            }
        }
        
        function handleMessage(message) {
            const type = message.type;
            const timestamp = new Date(message.timestamp * 1000).toLocaleTimeString();
            
            if (type === 'connection_established') {
                clientId = message.client_id;
                document.getElementById('clientId').textContent = clientId;
                addMessage(`✅ 연결 설정됨 - Client ID: ${clientId}`, 'success');
            } else if (type === 'pipeline_progress') {
                addMessage(`⚡ [${message.step}/8] ${message.step_name}: ${message.message} (${message.progress.toFixed(1)}%)`, 'progress');
            } else if (type === 'system_status') {
                addMessage(`📊 시스템 상태: Pipeline ${message.data.pipeline?.initialized ? '✅' : '❌'}, Memory: ${JSON.stringify(message.data.memory?.system_memory)}`, 'info');
            } else {
                addMessage(`📨 ${type}: ${JSON.stringify(message)}`, 'info');
            }
        }
        
        function addMessage(text, type = 'info') {
            const messages = document.getElementById('messages');
            const div = document.createElement('div');
            div.className = `message ${type}`;
            div.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
        }
        
        function sendMessage(message) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify(message));
            } else {
                addMessage('❌ WebSocket이 연결되지 않음', 'error');
            }
        }
        
        function getSystemStatus() {
            sendMessage({type: 'get_system_status'});
        }
        
        function ping() {
            sendMessage({type: 'ping'});
        }
        
        function subscribeToProcess() {
            const processId = document.getElementById('processId').value;
            sendMessage({
                type: 'subscribe_process',
                process_id: processId
            });
        }
        
        function clearMessages() {
            document.getElementById('messages').innerHTML = '';
        }
        
        // 자동 연결
        connect();
    </script>
</body>
</html>
    """)

# 백그라운드 태스크 시작 함수
async def start_background_tasks():
    """백그라운드 태스크 시작"""
    asyncio.create_task(system_monitor())

# 연결 관리자 및 진행 콜백 내보내기
__all__ = ['router', 'manager', 'create_progress_callback', 'start_background_tasks']