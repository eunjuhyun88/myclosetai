# backend/app/shared/websocket_manager.py
"""
🔥 MyCloset AI WebSocket Manager
================================================================================

웹소켓 관리를 위한 공통 모듈입니다.

- WebSocketManager: 웹소켓 연결 관리
- broadcast_to_session: 세션별 브로드캐스트
- broadcast_to_all: 전체 브로드캐스트

Author: MyCloset AI Team
Date: 2025-08-01
Version: 1.0
"""

import logging
import json
import asyncio
from typing import Dict, Set, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime

logger = logging.getLogger(__name__)


class WebSocketManager:
    """웹소켓 연결 관리자"""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.connection_info: Dict[WebSocket, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, session_id: str, client_info: Optional[Dict[str, Any]] = None):
        """웹소켓 연결 추가"""
        try:
            await websocket.accept()
            
            async with self._lock:
                if session_id not in self.active_connections:
                    self.active_connections[session_id] = set()
                
                self.active_connections[session_id].add(websocket)
                self.connection_info[websocket] = {
                    'session_id': session_id,
                    'connected_at': datetime.now().isoformat(),
                    'client_info': client_info or {},
                    'last_activity': datetime.now().isoformat()
                }
            
            logger.info(f"✅ 웹소켓 연결 추가: 세션 {session_id}, 총 연결 수: {len(self.active_connections[session_id])}")
            
            # 연결 확인 메시지 전송
            await self.send_personal_message(
                websocket,
                {
                    "type": "connection_established",
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "message": "웹소켓 연결이 성공적으로 설정되었습니다."
                }
            )
            
        except Exception as e:
            logger.error(f"❌ 웹소켓 연결 실패: {e}")
            raise
    
    async def disconnect(self, websocket: WebSocket):
        """웹소켓 연결 제거"""
        try:
            async with self._lock:
                # 연결 정보에서 제거
                if websocket in self.connection_info:
                    session_id = self.connection_info[websocket]['session_id']
                    
                    # 세션에서 제거
                    if session_id in self.active_connections:
                        self.active_connections[session_id].discard(websocket)
                        
                        # 세션이 비어있으면 세션도 제거
                        if not self.active_connections[session_id]:
                            del self.active_connections[session_id]
                    
                    # 연결 정보 제거
                    del self.connection_info[websocket]
                    
                    logger.info(f"✅ 웹소켓 연결 제거: 세션 {session_id}")
            
        except Exception as e:
            logger.error(f"❌ 웹소켓 연결 제거 실패: {e}")
    
    async def send_personal_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """개별 웹소켓에 메시지 전송"""
        try:
            if websocket.client_state.value == 1:  # 연결 상태 확인
                await websocket.send_text(json.dumps(message, ensure_ascii=False))
                
                # 마지막 활동 시간 업데이트
                if websocket in self.connection_info:
                    self.connection_info[websocket]['last_activity'] = datetime.now().isoformat()
                    
        except WebSocketDisconnect:
            await self.disconnect(websocket)
        except Exception as e:
            logger.error(f"❌ 개별 메시지 전송 실패: {e}")
            await self.disconnect(websocket)
    
    async def broadcast_to_session(self, session_id: str, message: Dict[str, Any]):
        """특정 세션의 모든 연결에 메시지 브로드캐스트"""
        try:
            if session_id not in self.active_connections:
                logger.warning(f"⚠️ 세션 {session_id}에 활성 연결이 없습니다.")
                return
            
            disconnected_websockets = set()
            
            for websocket in self.active_connections[session_id].copy():
                try:
                    await self.send_personal_message(websocket, message)
                except Exception as e:
                    logger.error(f"❌ 세션 브로드캐스트 실패: {e}")
                    disconnected_websockets.add(websocket)
            
            # 연결이 끊어진 웹소켓들 정리
            for websocket in disconnected_websockets:
                await self.disconnect(websocket)
            
            logger.info(f"✅ 세션 {session_id} 브로드캐스트 완료: {len(self.active_connections.get(session_id, set()))}개 연결")
            
        except Exception as e:
            logger.error(f"❌ 세션 브로드캐스트 중 에러: {e}")
    
    async def broadcast_to_all(self, message: Dict[str, Any]):
        """모든 연결에 메시지 브로드캐스트"""
        try:
            all_websockets = set()
            for session_websockets in self.active_connections.values():
                all_websockets.update(session_websockets)
            
            disconnected_websockets = set()
            
            for websocket in all_websockets:
                try:
                    await self.send_personal_message(websocket, message)
                except Exception as e:
                    logger.error(f"❌ 전체 브로드캐스트 실패: {e}")
                    disconnected_websockets.add(websocket)
            
            # 연결이 끊어진 웹소켓들 정리
            for websocket in disconnected_websockets:
                await self.disconnect(websocket)
            
            logger.info(f"✅ 전체 브로드캐스트 완료: {len(all_websockets)}개 연결")
            
        except Exception as e:
            logger.error(f"❌ 전체 브로드캐스트 중 에러: {e}")
    
    async def send_progress_update(self, session_id: str, step_id: int, progress: float, message: str = ""):
        """진행상황 업데이트 전송"""
        progress_message = {
            "type": "progress_update",
            "session_id": session_id,
            "step_id": step_id,
            "progress_percentage": progress,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.broadcast_to_session(session_id, progress_message)
    
    async def send_step_completion(self, session_id: str, step_id: int, result: Dict[str, Any]):
        """스텝 완료 알림 전송"""
        completion_message = {
            "type": "step_completion",
            "session_id": session_id,
            "step_id": step_id,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.broadcast_to_session(session_id, completion_message)
    
    async def send_error_notification(self, session_id: str, step_id: int, error_message: str):
        """에러 알림 전송"""
        error_message_data = {
            "type": "error_notification",
            "session_id": session_id,
            "step_id": step_id,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.broadcast_to_session(session_id, error_message_data)
    
    def get_connection_count(self, session_id: Optional[str] = None) -> int:
        """연결 수 반환"""
        if session_id:
            return len(self.active_connections.get(session_id, set()))
        else:
            return sum(len(connections) for connections in self.active_connections.values())
    
    def get_session_connections(self, session_id: str) -> Set[WebSocket]:
        """특정 세션의 연결들 반환"""
        return self.active_connections.get(session_id, set()).copy()
    
    def get_all_sessions(self) -> Set[str]:
        """모든 세션 ID 반환"""
        return set(self.active_connections.keys())
    
    async def cleanup_inactive_connections(self, max_inactive_minutes: int = 30):
        """비활성 연결 정리"""
        try:
            current_time = datetime.now()
            inactive_websockets = set()
            
            for websocket, info in self.connection_info.items():
                last_activity = datetime.fromisoformat(info['last_activity'])
                inactive_minutes = (current_time - last_activity).total_seconds() / 60
                
                if inactive_minutes > max_inactive_minutes:
                    inactive_websockets.add(websocket)
            
            # 비활성 연결 제거
            for websocket in inactive_websockets:
                await self.disconnect(websocket)
            
            if inactive_websockets:
                logger.info(f"✅ 비활성 연결 정리 완료: {len(inactive_websockets)}개 연결 제거")
            
        except Exception as e:
            logger.error(f"❌ 비활성 연결 정리 실패: {e}")


# 전역 웹소켓 매니저 인스턴스
websocket_manager = WebSocketManager()


# 편의 함수들
async def broadcast_to_session(session_id: str, message: Dict[str, Any]):
    """세션 브로드캐스트 편의 함수"""
    await websocket_manager.broadcast_to_session(session_id, message)


async def broadcast_to_all(message: Dict[str, Any]):
    """전체 브로드캐스트 편의 함수"""
    await websocket_manager.broadcast_to_all(message)


async def send_progress_update(session_id: str, step_id: int, progress: float, message: str = ""):
    """진행상황 업데이트 편의 함수"""
    await websocket_manager.send_progress_update(session_id, step_id, progress, message)


async def send_step_completion(session_id: str, step_id: int, result: Dict[str, Any]):
    """스텝 완료 알림 편의 함수"""
    await websocket_manager.send_step_completion(session_id, step_id, result)


async def send_error_notification(session_id: str, step_id: int, error_message: str):
    """에러 알림 편의 함수"""
    await websocket_manager.send_error_notification(session_id, step_id, error_message) 