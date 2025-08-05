"""
WebSocket 라우트
"""

import logging
import json
import asyncio
from typing import Dict, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket 엔드포인트 - 실시간 세션 상태 업데이트"""
    try:
        await websocket.accept()
        logger.info(f"🔌 WebSocket 연결 수락: session_id={session_id}")
        
        # 연결 성공 메시지 전송
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "message": "WebSocket 연결이 성공적으로 설정되었습니다."
        }))
        
        # 연결 유지 및 메시지 수신 대기
        while True:
            try:
                # 클라이언트로부터 메시지 수신
                data = await websocket.receive_text()
                message = json.loads(data)
                
                logger.info(f"📨 WebSocket 메시지 수신: {message.get('type', 'unknown')}")
                
                # 메시지 타입에 따른 처리
                if message.get('type') == 'ping':
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat()
                    }))
                elif message.get('type') == 'get_status':
                    # 세션 상태 조회
                    status = await get_session_status_websocket(session_id)
                    await websocket.send_text(json.dumps({
                        "type": "status_update",
                        "session_id": session_id,
                        "data": status,
                        "timestamp": datetime.now().isoformat()
                    }))
                elif message.get('type') == 'subscribe_updates':
                    # 실시간 업데이트 구독
                    await websocket.send_text(json.dumps({
                        "type": "subscription_confirmed",
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat(),
                        "message": "실시간 업데이트 구독이 확인되었습니다."
                    }))
                else:
                    # 알 수 없는 메시지 타입
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "session_id": session_id,
                        "error": "알 수 없는 메시지 타입입니다.",
                        "timestamp": datetime.now().isoformat()
                    }))
                    
            except WebSocketDisconnect:
                logger.info(f"🔌 WebSocket 연결 해제: session_id={session_id}")
                break
            except json.JSONDecodeError:
                logger.warning(f"⚠️ WebSocket JSON 파싱 오류: session_id={session_id}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "session_id": session_id,
                    "error": "잘못된 JSON 형식입니다.",
                    "timestamp": datetime.now().isoformat()
                }))
            except Exception as e:
                logger.error(f"❌ WebSocket 처리 오류: session_id={session_id}, error={e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "session_id": session_id,
                    "error": f"서버 오류가 발생했습니다: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }))
                
    except Exception as e:
        logger.error(f"❌ WebSocket 연결 설정 실패: session_id={session_id}, error={e}")


async def get_session_status_websocket(session_id: str) -> Dict[str, Any]:
    """WebSocket용 세션 상태 조회"""
    try:
        # Central Hub에서 SessionManager 조회
        from app.api.central_hub import _get_session_manager
        session_manager = _get_session_manager()
        
        if not session_manager:
            return {
                "status": "error",
                "message": "SessionManager를 찾을 수 없습니다.",
                "session_id": session_id
            }
        
        # 세션 상태 조회
        session_status = await session_manager.get_session_status(session_id)
        
        return {
            "status": "success",
            "session_id": session_id,
            "data": session_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ WebSocket 세션 상태 조회 실패: session_id={session_id}, error={e}")
        return {
            "status": "error",
            "message": f"세션 상태 조회 실패: {str(e)}",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }


async def broadcast_session_update(session_id: str, update_data: Dict[str, Any]):
    """세션 업데이트를 WebSocket으로 브로드캐스트"""
    try:
        # WebSocket 매니저에서 연결된 클라이언트들에게 브로드캐스트
        from app.api.central_hub import _get_websocket_manager
        websocket_manager = _get_websocket_manager()
        
        if websocket_manager and hasattr(websocket_manager, 'broadcast_to_session'):
            await websocket_manager.broadcast_to_session(session_id, {
                "type": "session_update",
                "session_id": session_id,
                "data": update_data,
                "timestamp": datetime.now().isoformat()
            })
            logger.info(f"📡 WebSocket 브로드캐스트 완료: session_id={session_id}")
        else:
            logger.warning(f"⚠️ WebSocket 매니저를 찾을 수 없습니다: session_id={session_id}")
            
    except Exception as e:
        logger.error(f"❌ WebSocket 브로드캐스트 실패: session_id={session_id}, error={e}")