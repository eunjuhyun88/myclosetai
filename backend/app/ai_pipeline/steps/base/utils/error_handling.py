#!/usr/bin/env python3
"""
🔥 MyCloset AI - Error Handling Mixin
======================================

에러 처리 관련 기능을 담당하는 Mixin 클래스
에러 로깅, 복구, 사용자 친화적 메시지 등을 담당

Author: MyCloset AI Team
Date: 2025-08-14
Version: 2.0
"""

import logging
import traceback
import time
from typing import Dict, Any, Optional, Callable
from functools import wraps

class ErrorHandlingMixin:
    """에러 처리 관련 기능을 제공하는 Mixin"""
    
    def _create_error_response(self, error_message: str, error_code: str = None, 
                             suggestion: str = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """에러 응답 생성"""
        try:
            error_response = {
                'success': False,
                'error': error_message,
                'timestamp': time.time(),
                'step_name': getattr(self, 'step_name', 'Unknown'),
                'step_id': getattr(self, 'step_name', 0)
            }
            
            if error_code:
                error_response['error_code'] = error_code
            
            if suggestion:
                error_response['suggestion'] = suggestion
            
            if context:
                error_response['context'] = context
            
            # 성능 메트릭 업데이트
            if hasattr(self, 'performance_metrics'):
                self.performance_metrics.error_count += 1
            
            return error_response
            
        except Exception as e:
            # 에러 응답 생성 중 오류 발생 시 기본 응답 반환
            return {
                'success': False,
                'error': f"에러 응답 생성 실패: {str(e)}",
                'original_error': error_message,
                'timestamp': time.time()
            }

    def _log_error_with_context(self, error: Exception, operation: str = "unknown", 
                               context: Dict[str, Any] = None, level: str = "error"):
        """컨텍스트와 함께 에러 로깅"""
        try:
            if not hasattr(self, 'logger'):
                return
            
            # 에러 메시지 구성
            error_msg = f"❌ {operation} 중 오류 발생: {str(error)}"
            
            # 컨텍스트 정보 추가
            if context:
                context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
                error_msg += f" | 컨텍스트: {context_str}"
            
            # 스택 트레이스 추가
            stack_trace = traceback.format_exc()
            
            # 로그 레벨에 따른 출력
            if level == "debug":
                self.logger.debug(error_msg)
                self.logger.debug(f"스택 트레이스:\n{stack_trace}")
            elif level == "info":
                self.logger.info(error_msg)
            elif level == "warning":
                self.logger.warning(error_msg)
            else:  # error
                self.logger.error(error_msg)
                self.logger.error(f"스택 트레이스:\n{stack_trace}")
            
            # 성능 메트릭 업데이트
            if hasattr(self, 'performance_metrics'):
                self.performance_metrics.error_count += 1
            
        except Exception as e:
            # 로깅 중 오류 발생 시 기본 출력
            print(f"에러 로깅 실패: {e}")
            print(f"원본 에러: {error}")

    def _handle_step_initialization_error(self, error: Exception, context: Dict[str, Any] = None):
        """Step 초기화 에러 처리"""
        try:
            error_response = self._create_error_response(
                error_message=f"Step 초기화 실패: {str(error)}",
                error_code="STEP_INIT_ERROR",
                suggestion="의존성 주입 상태를 확인하고 Central Hub 연결을 점검하세요",
                context=context or {}
            )
            
            self._log_error_with_context(error, "Step 초기화", context)
            
            # 기본 속성들만 설정하여 최소한의 동작 보장
            self._setup_minimal_attributes()
            
            return error_response
            
        except Exception as e:
            return self._create_error_response(
                error_message=f"에러 처리 중 추가 오류 발생: {str(e)}",
                error_code="ERROR_HANDLING_FAILED"
            )

    def _handle_dependency_injection_error(self, error: Exception, service_name: str, context: Dict[str, Any] = None):
        """의존성 주입 에러 처리"""
        try:
            error_response = self._create_error_response(
                error_message=f"의존성 주입 실패 ({service_name}): {str(error)}",
                error_code="DEPENDENCY_INJECTION_ERROR",
                suggestion=f"{service_name} 서비스가 Central Hub에 등록되어 있는지 확인하세요",
                context=context or {}
            )
            
            self._log_error_with_context(error, f"의존성 주입 ({service_name})", context)
            
            return error_response
            
        except Exception as e:
            return self._create_error_response(
                error_message=f"의존성 주입 에러 처리 중 추가 오류 발생: {str(e)}",
                error_code="ERROR_HANDLING_FAILED"
            )

    def _handle_data_conversion_error(self, error: Exception, conversion_type: str, 
                                    data_info: Dict[str, Any] = None, context: Dict[str, Any] = None):
        """데이터 변환 에러 처리"""
        try:
            error_response = self._create_error_response(
                error_message=f"데이터 변환 실패 ({conversion_type}): {str(error)}",
                error_code="DATA_CONVERSION_ERROR",
                suggestion="입력 데이터 형식과 API 매핑 설정을 확인하세요",
                context={
                    'conversion_type': conversion_type,
                    'data_info': data_info or {},
                    **(context or {})
                }
            )
            
            self._log_error_with_context(error, f"데이터 변환 ({conversion_type})", context)
            
            return error_response
            
        except Exception as e:
            return self._create_error_response(
                error_message=f"데이터 변환 에러 처리 중 추가 오류 발생: {str(e)}",
                error_code="ERROR_HANDLING_FAILED"
            )

    def _handle_central_hub_error(self, error: Exception, operation: str, context: Dict[str, Any] = None):
        """Central Hub 에러 처리"""
        try:
            error_response = self._create_error_response(
                error_message=f"Central Hub {operation} 실패: {str(error)}",
                error_code="CENTRAL_HUB_ERROR",
                suggestion="Central Hub 서비스 상태와 네트워크 연결을 확인하세요",
                context=context or {}
            )
            
            self._log_error_with_context(error, f"Central Hub {operation}", context)
            
            # Central Hub 연결 재시도
            self._retry_central_hub_connection()
            
            return error_response
            
        except Exception as e:
            return self._create_error_response(
                error_message=f"Central Hub 에러 처리 중 추가 오류 발생: {str(e)}",
                error_code="ERROR_HANDLING_FAILED"
            )

    def _setup_minimal_attributes(self):
        """최소한의 속성들 설정 (에러 상황에서의 폴백)"""
        try:
            # 기본 속성들만 설정
            if not hasattr(self, 'step_name'):
                self.step_name = getattr(self, '__class__', type(self)).__name__
            
            if not hasattr(self, 'step_id'):
                self.step_id = 0
            
            if not hasattr(self, 'device'):
                self.device = 'cpu'
            
            if not hasattr(self, 'is_initialized'):
                self.is_initialized = False
            
            if not hasattr(self, 'is_ready'):
                self.is_ready = False
            
            if not hasattr(self, 'dependencies_injected'):
                self.dependencies_injected = {}
            
            if not hasattr(self, 'performance_stats'):
                self.performance_stats = {}
            
            # 로거 설정
            if not hasattr(self, 'logger'):
                self.logger = logging.getLogger(f"steps.{self.step_name}")
                if not self.logger.handlers:
                    handler = logging.StreamHandler()
                    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                    handler.setFormatter(formatter)
                    self.logger.addHandler(handler)
                    self.logger.setLevel(logging.INFO)
            
        except Exception as e:
            print(f"최소 속성 설정 실패: {e}")

    def _retry_central_hub_connection(self, max_retries: int = 3, delay: float = 1.0):
        """Central Hub 연결 재시도"""
        try:
            if not hasattr(self, 'logger'):
                return
            
            self.logger.info(f"🔄 Central Hub 연결 재시도 시작 (최대 {max_retries}회)")
            
            for attempt in range(max_retries):
                try:
                    time.sleep(delay * (attempt + 1))  # 지수 백오프
                    
                    container = self._get_central_hub_container()
                    if container:
                        self.set_central_hub_container(container)
                        self.logger.info(f"✅ Central Hub 연결 재시도 성공 (시도 {attempt + 1})")
                        return True
                    
                    self.logger.debug(f"Central Hub 연결 재시도 {attempt + 1}/{max_retries} 실패")
                    
                except Exception as e:
                    self.logger.debug(f"Central Hub 연결 재시도 {attempt + 1}/{max_retries} 중 오류: {e}")
            
            self.logger.warning(f"⚠️ Central Hub 연결 재시도 {max_retries}회 모두 실패")
            return False
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Central Hub 연결 재시도 중 오류: {e}")
            return False

    def _create_fallback_api_response(self, original_result: Dict[str, Any], 
                                    error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """폴백 API 응답 생성"""
        try:
            fallback_response = {
                'success': False,
                'error': f"처리 실패: {str(error)}",
                'fallback_mode': True,
                'step_name': getattr(self, 'step_name', 'Unknown'),
                'step_id': getattr(self, 'step_name', 0),
                'timestamp': time.time(),
                'suggestion': '서비스 상태를 확인하고 잠시 후 다시 시도하세요'
            }
            
            if context:
                fallback_response['context'] = context
            
            # 원본 결과에서 안전한 데이터만 포함
            if original_result:
                safe_keys = ['step_name', 'step_id', 'processing_time']
                for key in safe_keys:
                    if key in original_result:
                        fallback_response[key] = original_result[key]
            
            return fallback_response
            
        except Exception as e:
            return {
                'success': False,
                'error': f"폴백 응답 생성 실패: {str(e)}",
                'fallback_mode': True,
                'timestamp': time.time()
            }

    def error_handler(self, operation: str = "unknown", context: Dict[str, Any] = None):
        """에러 핸들러 데코레이터"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # 에러 로깅
                    self._log_error_with_context(e, operation, context)
                    
                    # 에러 응답 생성
                    error_response = self._create_error_response(
                        error_message=f"{operation} 실패: {str(e)}",
                        error_code=f"{operation.upper()}_ERROR",
                        context=context or {}
                    )
                    
                    return error_response
            
            return wrapper
        return decorator

    def _log_step_performance(self, operation: str, start_time: float, success: bool, error: Exception = None):
        """Step 성능 로깅"""
        try:
            if not hasattr(self, 'logger'):
                return
            
            processing_time = time.time() - start_time
            
            if success:
                if hasattr(self, 'logger'):
                    self.logger.debug(f"✅ {operation} 완료 ({processing_time:.3f}s)")
            else:
                if hasattr(self, 'logger'):
                    self.logger.error(f"❌ {operation} 실패 ({processing_time:.3f}s): {error}")
            
            # 성능 통계 업데이트
            if hasattr(self, 'performance_stats'):
                self.performance_stats['total_requests'] = self.performance_stats.get('total_requests', 0) + 1
                self.performance_stats['total_processing_time'] = self.performance_stats.get('total_processing_time', 0.0) + processing_time
                
                if success:
                    self.performance_stats['successful_requests'] = self.performance_stats.get('successful_requests', 0) + 1
                else:
                    self.performance_stats['failed_requests'] = self.performance_stats.get('failed_requests', 0) + 1
                
                # 평균 처리 시간 계산
                total_requests = self.performance_stats['total_requests']
                if total_requests > 0:
                    self.performance_stats['average_processing_time'] = (
                        self.performance_stats['total_processing_time'] / total_requests
                    )
            
        except Exception as e:
            print(f"성능 로깅 실패: {e}")

    def _create_step_error_response(self, step_name: str, error: Exception, operation: str = "unknown") -> Dict[str, Any]:
        """Step 에러 응답 생성"""
        try:
            return {
                'success': False,
                'error': str(error),
                'step_name': step_name,
                'operation': operation,
                'timestamp': time.time(),
                'suggestion': self._get_error_suggestion(error, operation)
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"에러 응답 생성 실패: {str(e)}",
                'original_error': str(error),
                'timestamp': time.time()
            }

    def _get_error_suggestion(self, error: Exception, operation: str) -> str:
        """에러에 따른 제안사항 반환"""
        try:
            error_str = str(error).lower()
            
            if "connection" in error_str or "network" in error_str:
                return "네트워크 연결 상태를 확인하세요"
            elif "memory" in error_str or "out of memory" in error_str:
                return "메모리 사용량을 확인하고 불필요한 프로세스를 종료하세요"
            elif "permission" in error_str or "access" in error_str:
                return "파일/폴더 접근 권한을 확인하세요"
            elif "import" in error_str or "module" in error_str:
                return "필요한 패키지가 설치되어 있는지 확인하세요"
            elif "timeout" in error_str:
                return "요청 시간이 초과되었습니다. 잠시 후 다시 시도하세요"
            else:
                return "로그를 확인하여 자세한 오류 내용을 파악하세요"
                
        except Exception:
            return "로그를 확인하여 자세한 오류 내용을 파악하세요"
