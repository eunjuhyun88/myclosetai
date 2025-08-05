# backend/app/shared/__init__.py
"""
🔥 MyCloset AI Shared Modules
================================================================================

공통으로 사용되는 모듈들을 모아둔 패키지입니다.

- response_formatter: API 응답 포맷팅
- error_handler: 에러 처리
- validation_service: 검증 서비스
- websocket_manager: 웹소켓 관리

Author: MyCloset AI Team
Date: 2025-08-01
Version: 1.0
"""

from .response_formatter import (
    format_step_api_response, 
    format_api_response,
    enhance_step_result_for_frontend,
    get_bmi_category
)
from .error_handler import (
    handle_api_error, 
    create_error_response,
    handle_validation_error,
    handle_session_error,
    handle_processing_error
)
from .validation_service import (
    validate_upload_file, 
    validate_measurements,
    validate_session_id,
    validate_step_parameters
)
from .websocket_manager import (
    WebSocketManager,
    websocket_manager,
    broadcast_to_session,
    broadcast_to_all,
    send_progress_update,
    send_step_completion,
    send_error_notification
)

__all__ = [
    # Response Formatter
    'format_step_api_response',
    'format_api_response',
    'enhance_step_result_for_frontend',
    'get_bmi_category',
    
    # Error Handler
    'handle_api_error',
    'create_error_response',
    'handle_validation_error',
    'handle_session_error',
    'handle_processing_error',
    
    # Validation Service
    'validate_upload_file',
    'validate_measurements',
    'validate_session_id',
    'validate_step_parameters',
    
    # WebSocket Manager
    'WebSocketManager',
    'websocket_manager',
    'broadcast_to_session',
    'broadcast_to_all',
    'send_progress_update',
    'send_step_completion',
    'send_error_notification'
] 