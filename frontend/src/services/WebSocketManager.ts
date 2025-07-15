"""
MyCloset AI WebSocket 관리자
실시간 진행률 업데이트 및 세션 관리
파일 경로: frontend/src/services/WebSocketManager.ts
"""
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from fastapi import WebSocket, WebSocketDisconnect
from uuid import uuid4
import time

logger = logging.getLogger(__name__)

class WebSocketConnectionManager:
    """
    WebSocket 연결 및 세션 관리자
    """
    
    def __init__(self):
        # 활성 연결들
        self.active_connections: Dict[str, WebSocket] = {}
        
        # 세션별 연결 매핑
        self.session_connections: Dict[str, List[str]] = {}
        
        # 연결별 메타데이터
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
        # 통계
        self.stats = {
            'total_connections': 0,
            'current_connections': 0,
            'messages_sent': 0,
            'messages_received': 0
        }
    
    async def connect(self, websocket: WebSocket, session_id: Optional[str] = None) -> str:
        """WebSocket 연결 수락 및 등록"""
        await websocket.accept()
        
        # 고유 연결 ID 생성
        connection_id = f"ws_{uuid4().hex[:12]}"
        
        # 연결 등록
        self.active_connections[connection_id] = websocket
        
        # 메타데이터 저장
        self.connection_metadata[connection_id] = {
            'session_id': session_id,
            'connected_at': time.time(),
            'last_activity': time.time(),
            'message_count': 0
        }
        
        # 세션 매핑 (세션 ID가 있는 경우)
        if session_id:
            if session_id not in self.session_connections:
                self.session_connections[session_id] = []
            self.session_connections[session_id].append(connection_id)
        
        # 통계 업데이트
        self.stats['total_connections'] += 1
        self.stats['current_connections'] = len(self.active_connections)
        
        logger.info(f"WebSocket 연결됨: {connection_id} (세션: {session_id})")
        
        # 연결 확인 메시지 전송
        await self.send_to_connection(connection_id, {
            "type": "connection_established",
            "connection_id": connection_id,
            "session_id": session_id,
            "timestamp": time.time()
        })
        
        return connection_id
    
    def disconnect(self, connection_id: str):
        """WebSocket 연결 해제"""
        if connection_id in self.active_connections:
            # 연결 제거
            del self.active_connections[connection_id]
            
            # 메타데이터에서 세션 ID 가져오기
            metadata = self.connection_metadata.get(connection_id, {})
            session_id = metadata.get('session_id')
            
            # 세션 매핑에서 제거
            if session_id and session_id in self.session_connections:
                self.session_connections[session_id] = [
                    conn_id for conn_id in self.session_connections[session_id] 
                    if conn_id != connection_id
                ]
                # 빈 세션 제거
                if not self.session_connections[session_id]:
                    del self.session_connections[session_id]
            
            # 메타데이터 제거
            if connection_id in self.connection_metadata:
                del self.connection_metadata[connection_id]
            
            # 통계 업데이트
            self.stats['current_connections'] = len(self.active_connections)
            
            logger.info(f"WebSocket 연결 해제됨: {connection_id} (세션: {session_id})")
    
    async def send_to_connection(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """특정 연결에 메시지 전송"""
        if connection_id not in self.active_connections:
            logger.warning(f"연결 ID가 존재하지 않음: {connection_id}")
            return False
        
        try:
            websocket = self.active_connections[connection_id]
            await websocket.send_text(json.dumps(message))
            
            # 통계 및 메타데이터 업데이트
            self.stats['messages_sent'] += 1
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]['last_activity'] = time.time()
                self.connection_metadata[connection_id]['message_count'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"메시지 전송 실패 ({connection_id}): {e}")
            # 연결이 끊어진 경우 정리
            self.disconnect(connection_id)
            return False
    
    async def broadcast_to_session(self, message: Dict[str, Any], session_id: str) -> int:
        """특정 세션의 모든 연결에 브로드캐스트"""
        if session_id not in self.session_connections:
            logger.warning(f"세션이 존재하지 않음: {session_id}")
            return 0
        
        connection_ids = self.session_connections[session_id].copy()
        sent_count = 0
        
        for connection_id in connection_ids:
            success = await self.send_to_connection(connection_id, message)
            if success:
                sent_count += 1
        
        logger.debug(f"세션 {session_id}에 메시지 브로드캐스트: {sent_count}/{len(connection_ids)}")
        return sent_count
    
    async def broadcast_to_all(self, message: Dict[str, Any]) -> int:
        """모든 활성 연결에 브로드캐스트"""
        connection_ids = list(self.active_connections.keys())
        sent_count = 0
        
        for connection_id in connection_ids:
            success = await self.send_to_connection(connection_id, message)
            if success:
                sent_count += 1
        
        logger.debug(f"전체 브로드캐스트: {sent_count}/{len(connection_ids)}")
        return sent_count
    
    def get_session_connections(self, session_id: str) -> List[str]:
        """세션의 연결 ID 목록 반환"""
        return self.session_connections.get(session_id, [])
    
    def get_connection_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """연결 정보 반환"""
        if connection_id not in self.connection_metadata:
            return None
        
        metadata = self.connection_metadata[connection_id].copy()
        metadata['is_active'] = connection_id in self.active_connections
        return metadata
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return {
            **self.stats,
            'active_sessions': len(self.session_connections),
            'connections_per_session': {
                session_id: len(connections) 
                for session_id, connections in self.session_connections.items()
            }
        }

# 전역 WebSocket 매니저 인스턴스
manager = WebSocketConnectionManager()

def create_progress_callback(session_id: str) -> Callable:
    """진행률 콜백 함수 생성"""
    async def progress_callback(stage: str, percentage: float, **kwargs):
        """진행률 업데이트 콜백"""
        message = {
            "type": "pipeline_progress",
            "session_id": session_id,
            "data": {
                "stage": stage,
                "percentage": min(max(percentage, 0), 100),  # 0-100 범위 제한
                "timestamp": time.time(),
                **kwargs
            }
        }
        
        await manager.broadcast_to_session(message, session_id)
        logger.debug(f"진행률 업데이트: {session_id} - {stage} ({percentage:.1f}%)")
    
    return progress_callback

async def send_pipeline_start(session_id: str, pipeline_info: Dict[str, Any]):
    """파이프라인 시작 알림"""
    message = {
        "type": "pipeline_start",
        "session_id": session_id,
        "data": {
            "message": "가상 피팅 처리를 시작합니다...",
            "pipeline_info": pipeline_info,
            "timestamp": time.time()
        }
    }
    await manager.broadcast_to_session(message, session_id)

async def send_pipeline_complete(session_id: str, result: Dict[str, Any]):
    """파이프라인 완료 알림"""
    message = {
        "type": "pipeline_completed",
        "session_id": session_id,
        "data": {
            "message": "가상 피팅 처리가 완료되었습니다!",
            "result_summary": {
                "processing_time": result.get("processing_time", 0),
                "quality_score": result.get("quality_score", 0),
                "success": result.get("success", False)
            },
            "timestamp": time.time()
        }
    }
    await manager.broadcast_to_session(message, session_id)

async def send_pipeline_error(session_id: str, error: str):
    """파이프라인 에러 알림"""
    message = {
        "type": "pipeline_error",
        "session_id": session_id,
        "data": {
            "message": "가상 피팅 처리 중 오류가 발생했습니다.",
            "error": error,
            "timestamp": time.time()
        }
    }
    await manager.broadcast_to_session(message, session_id)

# 편의 함수들
def get_manager() -> WebSocketConnectionManager:
    """전역 WebSocket 매니저 반환"""
    return manager

async def handle_websocket_connection(websocket: WebSocket, session_id: Optional[str] = None):
    """WebSocket 연결 핸들러"""
    connection_id = await manager.connect(websocket, session_id)
    
    try:
        while True:
            # 클라이언트 메시지 수신
            try:
                data = await websocket.receive_text()
                manager.stats['messages_received'] += 1
                
                # JSON 파싱 시도
                try:
                    message = json.loads(data)
                    await handle_client_message(connection_id, message)
                except json.JSONDecodeError:
                    logger.warning(f"잘못된 JSON 형식: {data}")
                    
            except WebSocketDisconnect:
                logger.info(f"클라이언트 연결 해제: {connection_id}")
                break
            except Exception as e:
                logger.error(f"메시지 수신 오류: {e}")
                break
                
    finally:
        manager.disconnect(connection_id)

async def handle_client_message(connection_id: str, message: Dict[str, Any]):
    """클라이언트 메시지 처리"""
    message_type = message.get("type")
    
    if message_type == "ping":
        # Ping-Pong 응답
        await manager.send_to_connection(connection_id, {
            "type": "pong",
            "timestamp": time.time()
        })
    
    elif message_type == "subscribe":
        # 세션 구독 (동적으로 세션 변경)
        new_session_id = message.get("session_id")
        if new_session_id:
            # 기존 세션에서 제거
            old_metadata = manager.connection_metadata.get(connection_id, {})
            old_session_id = old_metadata.get('session_id')
            
            if old_session_id and old_session_id in manager.session_connections:
                manager.session_connections[old_session_id] = [
                    conn_id for conn_id in manager.session_connections[old_session_id] 
                    if conn_id != connection_id
                ]
            
            # 새 세션에 추가
            if new_session_id not in manager.session_connections:
                manager.session_connections[new_session_id] = []
            manager.session_connections[new_session_id].append(connection_id)
            
            # 메타데이터 업데이트
            manager.connection_metadata[connection_id]['session_id'] = new_session_id
            
            await manager.send_to_connection(connection_id, {
                "type": "subscription_confirmed",
                "session_id": new_session_id,
                "timestamp": time.time()
            })
    
    elif message_type == "get_stats":
        # 통계 정보 요청
        stats = manager.get_stats()
        await manager.send_to_connection(connection_id, {
            "type": "stats_response",
            "data": stats,
            "timestamp": time.time()
        })
    
    else:
        logger.debug(f"알 수 없는 메시지 타입: {message_type}")