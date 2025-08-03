"""
🔥 MyCloset AI 커스텀 예외 클래스 정의
Central Hub DI Container v7.0 기반 구체적 예외 처리
목업 데이터 진단 및 품질 관리 시스템 포함
"""

import traceback
import threading
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
import numpy as np
import cv2


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ErrorContext:
    """에러 컨텍스트 정보"""
    step_name: str
    step_id: int
    session_id: str
    timestamp: str
    input_data_info: Dict[str, Any] = None
    model_info: Dict[str, Any] = None
    system_info: Dict[str, Any] = None
    performance_metrics: Dict[str, Any] = None


class MyClosetAIException(Exception):
    """MyCloset AI 기본 예외 클래스"""
    
    def __init__(self, message: str, error_code: str = None, context: dict = None):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.timestamp = datetime.now().isoformat()
        self.error_context: Optional[ErrorContext] = None
        super().__init__(self.message)
    
    def __str__(self):
        return f"[{self.error_code}] {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """예외를 딕셔너리로 변환"""
        result = {
            'error_code': self.error_code,
            'message': self.message,
            'context': self.context,
            'timestamp': self.timestamp,
            'exception_type': self.__class__.__name__
        }
        
        if self.error_context:
            result['error_context'] = asdict(self.error_context)
        
        return result


# 목업 데이터 진단 관련 예외 클래스들
class MockDataDetectionError(MyClosetAIException):
    """목업 데이터 감지 오류"""
    pass


class DataQualityError(MyClosetAIException):
    """데이터 품질 관련 오류"""
    pass


class ModelInferenceError(MyClosetAIException):
    """모델 추론 관련 오류"""
    pass


class ModelLoadingError(MyClosetAIException):
    """AI 모델 로딩 관련 오류"""
    pass


class ImageProcessingError(MyClosetAIException):
    """이미지 처리 관련 오류"""
    pass


class SessionError(MyClosetAIException):
    """세션 관리 관련 오류"""
    pass


class DependencyInjectionError(MyClosetAIException):
    """의존성 주입 관련 오류"""
    pass


class APIResponseError(MyClosetAIException):
    """API 응답 생성 관련 오류"""
    pass


class VirtualFittingError(MyClosetAIException):
    """가상 피팅 처리 관련 오류"""
    pass


class DataValidationError(MyClosetAIException):
    """데이터 검증 관련 오류"""
    pass


class MemoryError(MyClosetAIException):
    """메모리 관련 오류"""
    pass


class FileOperationError(MyClosetAIException):
    """파일 작업 관련 오류"""
    pass


class NetworkError(MyClosetAIException):
    """네트워크 관련 오류"""
    pass


class ConfigurationError(MyClosetAIException):
    """설정 관련 오류"""
    pass


class AuthenticationError(MyClosetAIException):
    """인증 관련 오류"""
    pass


class RateLimitError(MyClosetAIException):
    """속도 제한 관련 오류"""
    pass


class TimeoutError(MyClosetAIException):
    """타임아웃 관련 오류"""
    pass


class DatabaseError(MyClosetAIException):
    """데이터베이스 관련 오류"""
    pass


class CacheError(MyClosetAIException):
    """캐시 관련 오류"""
    pass


class WebSocketError(MyClosetAIException):
    """WebSocket 관련 오류"""
    pass


class PipelineError(MyClosetAIException):
    """파이프라인 처리 관련 오류"""
    pass


class QualityAssessmentError(MyClosetAIException):
    """품질 평가 관련 오류"""
    pass


class GeometricMatchingError(MyClosetAIException):
    """기하학적 매칭 관련 오류"""
    pass


class PoseEstimationError(MyClosetAIException):
    """포즈 추정 관련 오류"""
    pass


class HumanParsingError(MyClosetAIException):
    """인간 파싱 관련 오류"""
    pass


class ClothingAnalysisError(MyClosetAIException):
    """의류 분석 관련 오류"""
    pass


class MeasurementValidationError(MyClosetAIException):
    """측정값 검증 관련 오류"""
    pass


class UploadValidationError(MyClosetAIException):
    """업로드 검증 관련 오류"""
    pass


class ResultAnalysisError(MyClosetAIException):
    """결과 분석 관련 오류"""
    pass


class MockDataDetector:
    """목업 데이터 감지 및 진단 시스템"""
    
    def __init__(self):
        self.mock_patterns = {
            'image': {
                'uniform_color': self._detect_uniform_color,
                'test_pattern': self._detect_test_pattern,
                'placeholder': self._detect_placeholder_image,
                'default_size': self._detect_default_size
            },
            'text': {
                'placeholder': self._detect_placeholder_text,
                'lorem_ipsum': self._detect_lorem_ipsum,
                'default_values': self._detect_default_values
            },
            'data': {
                'empty_arrays': self._detect_empty_arrays,
                'constant_values': self._detect_constant_values,
                'test_data': self._detect_test_data
            }
        }
    
    def detect_mock_data(self, data: Any, data_type: str = "auto") -> Dict[str, Any]:
        """목업 데이터 감지"""
        detection_result = {
            'is_mock': False,
            'confidence': 0.0,
            'detected_patterns': [],
            'suggestions': [],
            'data_quality_score': 0.0
        }
        
        try:
            if data_type == "auto":
                data_type = self._infer_data_type(data)
            
            if data_type in self.mock_patterns:
                patterns = self.mock_patterns[data_type]
                detected_patterns = []
                total_confidence = 0.0
                
                for pattern_name, detector_func in patterns.items():
                    try:
                        result = detector_func(data)
                        if result['detected']:
                            detected_patterns.append({
                                'pattern': pattern_name,
                                'confidence': result['confidence'],
                                'description': result['description']
                            })
                            total_confidence += result['confidence']
                    except Exception as e:
                        logger.warning(f"패턴 감지 중 오류: {pattern_name} - {e}")
                
                detection_result['detected_patterns'] = detected_patterns
                detection_result['confidence'] = min(total_confidence, 1.0)
                detection_result['is_mock'] = total_confidence > 0.7
                detection_result['data_quality_score'] = 1.0 - total_confidence
                
                # 제안사항 생성
                if detection_result['is_mock']:
                    detection_result['suggestions'] = self._generate_suggestions(detected_patterns, data_type)
            
        except Exception as e:
            logger.error(f"목업 데이터 감지 중 오류: {e}")
            detection_result['error'] = str(e)
        
        return detection_result
    
    def _infer_data_type(self, data: Any) -> str:
        """데이터 타입 추론"""
        if isinstance(data, (np.ndarray, list)) and len(data) > 0:
            if isinstance(data[0], (int, float)):
                return "data"
            elif isinstance(data[0], str):
                return "text"
        elif isinstance(data, str):
            return "text"
        elif isinstance(data, (np.ndarray, list)) and len(data) > 0:
            if hasattr(data[0], 'shape'):  # 이미지 배열
                return "image"
        return "data"
    
    def _detect_uniform_color(self, image_data) -> Dict[str, Any]:
        """균일한 색상 감지"""
        try:
            if isinstance(image_data, np.ndarray):
                # 이미지가 균일한 색상인지 확인
                std_dev = np.std(image_data)
                return {
                    'detected': std_dev < 5.0,
                    'confidence': max(0, (10.0 - std_dev) / 10.0),
                    'description': f"균일한 색상 감지 (표준편차: {std_dev:.2f})"
                }
        except:
            pass
        return {'detected': False, 'confidence': 0.0, 'description': ''}
    
    def _detect_test_pattern(self, image_data) -> Dict[str, Any]:
        """테스트 패턴 감지"""
        try:
            if isinstance(image_data, np.ndarray):
                # 체크보드 패턴이나 테스트 패턴 감지
                if image_data.shape[0] == image_data.shape[1] and image_data.shape[0] in [64, 128, 256]:
                    return {
                        'detected': True,
                        'confidence': 0.8,
                        'description': f"테스트 패턴 감지 (크기: {image_data.shape})"
                    }
        except:
            pass
        return {'detected': False, 'confidence': 0.0, 'description': ''}
    
    def _detect_placeholder_image(self, image_data) -> Dict[str, Any]:
        """플레이스홀더 이미지 감지"""
        try:
            if isinstance(image_data, np.ndarray):
                # 특정 크기의 기본 이미지 패턴 감지
                if image_data.shape in [(224, 224, 3), (256, 256, 3), (512, 512, 3)]:
                    return {
                        'detected': True,
                        'confidence': 0.6,
                        'description': f"플레이스홀더 이미지 감지 (크기: {image_data.shape})"
                    }
        except:
            pass
        return {'detected': False, 'confidence': 0.0, 'description': ''}
    
    def _detect_default_size(self, image_data) -> Dict[str, Any]:
        """기본 크기 감지"""
        try:
            if isinstance(image_data, np.ndarray):
                # 일반적인 기본 크기들
                default_sizes = [(64, 64), (128, 128), (224, 224), (256, 256), (512, 512)]
                if image_data.shape[:2] in default_sizes:
                    return {
                        'detected': True,
                        'confidence': 0.5,
                        'description': f"기본 크기 감지 (크기: {image_data.shape})"
                    }
        except:
            pass
        return {'detected': False, 'confidence': 0.0, 'description': ''}
    
    def _detect_placeholder_text(self, text_data) -> Dict[str, Any]:
        """플레이스홀더 텍스트 감지"""
        placeholder_patterns = [
            'placeholder', 'test', 'sample', 'example', 'dummy', 'mock',
            'undefined', 'null', 'none', 'empty', 'temp', 'tmp'
        ]
        
        if isinstance(text_data, str):
            text_lower = text_data.lower()
            for pattern in placeholder_patterns:
                if pattern in text_lower:
                    return {
                        'detected': True,
                        'confidence': 0.8,
                        'description': f"플레이스홀더 텍스트 감지: {pattern}"
                    }
        
        return {'detected': False, 'confidence': 0.0, 'description': ''}
    
    def _detect_lorem_ipsum(self, text_data) -> Dict[str, Any]:
        """Lorem ipsum 텍스트 감지"""
        if isinstance(text_data, str):
            lorem_patterns = ['lorem ipsum', 'dolor sit', 'amet consectetur']
            text_lower = text_data.lower()
            for pattern in lorem_patterns:
                if pattern in text_lower:
                    return {
                        'detected': True,
                        'confidence': 0.9,
                        'description': "Lorem ipsum 텍스트 감지"
                    }
        
        return {'detected': False, 'confidence': 0.0, 'description': ''}
    
    def _detect_default_values(self, text_data) -> Dict[str, Any]:
        """기본값 감지"""
        default_values = ['0', '1', 'true', 'false', 'null', 'undefined', 'none']
        
        if isinstance(text_data, str) and text_data.lower() in default_values:
            return {
                'detected': True,
                'confidence': 0.7,
                'description': f"기본값 감지: {text_data}"
            }
        
        return {'detected': False, 'confidence': 0.0, 'description': ''}
    
    def _detect_empty_arrays(self, data) -> Dict[str, Any]:
        """빈 배열 감지"""
        if isinstance(data, (list, np.ndarray)) and len(data) == 0:
            return {
                'detected': True,
                'confidence': 0.9,
                'description': "빈 배열 감지"
            }
        return {'detected': False, 'confidence': 0.0, 'description': ''}
    
    def _detect_constant_values(self, data) -> Dict[str, Any]:
        """상수값 감지"""
        try:
            if isinstance(data, (list, np.ndarray)) and len(data) > 1:
                if isinstance(data, list):
                    unique_values = set(data)
                else:
                    unique_values = set(data.flatten())
                
                if len(unique_values) == 1:
                    return {
                        'detected': True,
                        'confidence': 0.8,
                        'description': f"상수값 감지: {list(unique_values)[0]}"
                    }
        except:
            pass
        return {'detected': False, 'confidence': 0.0, 'description': ''}
    
    def _detect_test_data(self, data) -> Dict[str, Any]:
        """테스트 데이터 감지"""
        try:
            if isinstance(data, (list, np.ndarray)):
                # 테스트 데이터 패턴들
                if len(data) > 0:
                    first_item = data[0] if isinstance(data, list) else data.flatten()[0]
                    if first_item in [0, 1, -1, 255, 0.0, 1.0]:
                        return {
                            'detected': True,
                            'confidence': 0.6,
                            'description': f"테스트 데이터 패턴 감지: {first_item}"
                        }
        except:
            pass
        return {'detected': False, 'confidence': 0.0, 'description': ''}
    
    def _generate_suggestions(self, detected_patterns: List[Dict], data_type: str) -> List[str]:
        """제안사항 생성"""
        suggestions = []
        
        for pattern in detected_patterns:
            if pattern['pattern'] == 'uniform_color':
                suggestions.append("실제 이미지 데이터를 사용하세요")
            elif pattern['pattern'] == 'test_pattern':
                suggestions.append("실제 사용자 이미지를 업로드하세요")
            elif pattern['pattern'] == 'placeholder':
                suggestions.append("실제 데이터로 교체하세요")
            elif pattern['pattern'] == 'empty_arrays':
                suggestions.append("데이터가 비어있습니다. 입력을 확인하세요")
            elif pattern['pattern'] == 'constant_values':
                suggestions.append("다양한 값을 가진 실제 데이터를 사용하세요")
        
        return suggestions


class ErrorTracker:
    """에러 추적 및 통계 관리"""
    
    def __init__(self):
        self.error_counts = {}
        self.error_details = []
        self._lock = threading.RLock()
        self.max_error_details = 1000  # 최대 저장할 에러 상세 정보 수
        self.mock_detector = MockDataDetector()
    
    def track_error(self, error: Exception, context: dict = None, step_id: int = None):
        """에러 추적"""
        with self._lock:
            error_type = type(error).__name__
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            
            error_detail = {
                'timestamp': datetime.now().isoformat(),
                'error_type': error_type,
                'message': str(error),
                'context': context or {},
                'step_id': step_id
            }
            
            if isinstance(error, MyClosetAIException):
                error_detail['error_code'] = error.error_code
                error_detail['custom_context'] = error.context
                error_detail['is_custom_exception'] = True
            else:
                error_detail['is_custom_exception'] = False
            
            # 스택 트레이스 추가 (최대 10줄)
            try:
                tb_lines = traceback.format_exc().split('\n')
                error_detail['stack_trace'] = tb_lines[:10]
            except:
                error_detail['stack_trace'] = []
            
            self.error_details.append(error_detail)
            
            # 최대 개수 유지
            if len(self.error_details) > self.max_error_details:
                self.error_details = self.error_details[-self.max_error_details:]
    
    def detect_mock_data_in_context(self, context: dict) -> Dict[str, Any]:
        """컨텍스트에서 목업 데이터 감지"""
        mock_detection_results = {}
        
        if context:
            for key, value in context.items():
                if value is not None:
                    detection_result = self.mock_detector.detect_mock_data(value)
                    if detection_result['is_mock']:
                        mock_detection_results[key] = detection_result
        
        return mock_detection_results
    
    def get_error_summary(self) -> dict:
        """에러 요약 정보"""
        with self._lock:
            total_errors = sum(self.error_counts.values())
            most_common = max(self.error_counts.items(), key=lambda x: x[1]) if self.error_counts else None
            
            # 최근 에러들 (최대 20개)
            recent_errors = self.error_details[-20:] if self.error_details else []
            
            # 커스텀 예외 통계
            custom_exceptions = [e for e in self.error_details if e.get('is_custom_exception', False)]
            custom_exception_count = len(custom_exceptions)
            
            # 목업 데이터 관련 에러 통계
            mock_related_errors = [e for e in self.error_details 
                                 if 'mock' in e.get('message', '').lower() or 
                                 'test' in e.get('message', '').lower()]
            
            return {
                'total_errors': total_errors,
                'error_types': dict(self.error_counts),
                'most_common_error': most_common,
                'recent_errors': recent_errors,
                'custom_exception_count': custom_exception_count,
                'custom_exception_ratio': custom_exception_count / total_errors if total_errors > 0 else 0,
                'mock_related_errors': len(mock_related_errors),
                'tracking_started': self.error_details[0]['timestamp'] if self.error_details else None,
                'last_error': self.error_details[-1] if self.error_details else None
            }
    
    def get_errors_by_step(self, step_id: int) -> list:
        """특정 단계의 에러들 조회"""
        with self._lock:
            return [e for e in self.error_details if e.get('step_id') == step_id]
    
    def get_errors_by_type(self, error_type: str) -> list:
        """특정 타입의 에러들 조회"""
        with self._lock:
            return [e for e in self.error_details if e['error_type'] == error_type]
    
    def get_mock_data_analysis(self) -> dict:
        """목업 데이터 분석 결과"""
        with self._lock:
            mock_errors = []
            for error in self.error_details:
                if 'mock' in error.get('message', '').lower():
                    mock_errors.append(error)
            
            return {
                'total_mock_errors': len(mock_errors),
                'mock_error_details': mock_errors[-10:],  # 최근 10개
                'mock_error_types': list(set([e['error_type'] for e in mock_errors]))
            }
    
    def clear_old_errors(self, days: int = 7):
        """오래된 에러들 정리"""
        with self._lock:
            cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
            self.error_details = [
                e for e in self.error_details 
                if datetime.fromisoformat(e['timestamp']).timestamp() > cutoff_date
            ]
    
    def reset(self):
        """에러 추적기 초기화"""
        with self._lock:
            self.error_counts.clear()
            self.error_details.clear()


# 전역 에러 트래커 인스턴스
error_tracker = ErrorTracker()


def track_exception(error: Exception, context: dict = None, step_id: int = None):
    """전역 에러 트래커에 예외 등록"""
    error_tracker.track_error(error, context, step_id)


def get_error_summary() -> dict:
    """전역 에러 요약 조회"""
    return error_tracker.get_error_summary()


def detect_mock_data(data: Any, data_type: str = "auto") -> Dict[str, Any]:
    """목업 데이터 감지"""
    return error_tracker.mock_detector.detect_mock_data(data, data_type)


def create_exception_response(
    error: Exception, 
    step_name: str = "Unknown", 
    step_id: int = None,
    session_id: str = "unknown"
) -> dict:
    """예외를 API 응답 형식으로 변환"""
    
    # 에러 추적
    track_exception(error, {
        'step_name': step_name,
        'step_id': step_id,
        'session_id': session_id
    }, step_id)
    
    # 커스텀 예외인 경우
    if isinstance(error, MyClosetAIException):
        return {
            'success': False,
            'message': error.message,
            'error': error.error_code,
            'error_details': error.context,
            'step_name': step_name,
            'step_id': step_id,
            'session_id': session_id,
            'timestamp': error.timestamp,
            'exception_type': 'custom'
        }
    
    # 일반 예외인 경우
    return {
        'success': False,
        'message': f"처리 중 오류가 발생했습니다: {type(error).__name__}",
        'error': type(error).__name__,
        'error_details': {
            'original_message': str(error),
            'exception_type': type(error).__name__
        },
        'step_name': step_name,
        'step_id': step_id,
        'session_id': session_id,
        'timestamp': datetime.now().isoformat(),
        'exception_type': 'system'
    }


def create_mock_data_diagnosis_response(
    data: Any,
    step_name: str = "Unknown",
    step_id: int = None,
    session_id: str = "unknown"
) -> dict:
    """목업 데이터 진단 응답 생성"""
    
    try:
        # 목업 데이터 감지
        detection_result = detect_mock_data(data)
        
        # 에러 컨텍스트 생성
        error_context = ErrorContext(
            step_name=step_name,
            step_id=step_id,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            input_data_info={
                'data_type': type(data).__name__,
                'data_shape': getattr(data, 'shape', None) if hasattr(data, 'shape') else None,
                'data_length': len(data) if hasattr(data, '__len__') else None
            }
        )
        
        if detection_result['is_mock']:
            # 목업 데이터 감지된 경우
            mock_error = MockDataDetectionError(
                message="목업 데이터가 감지되었습니다",
                error_code="MOCK_DATA_DETECTED",
                context={
                    'detection_result': detection_result,
                    'step_name': step_name,
                    'step_id': step_id,
                    'session_id': session_id
                }
            )
            mock_error.error_context = error_context
            
            # 에러 추적
            track_exception(mock_error, {
                'step_name': step_name,
                'step_id': step_id,
                'session_id': session_id,
                'detection_result': detection_result
            }, step_id)
            
            return {
                'success': False,
                'message': "목업 데이터가 감지되었습니다. 실제 데이터를 사용하세요.",
                'error': 'MOCK_DATA_DETECTED',
                'error_details': {
                    'detection_result': detection_result,
                    'suggestions': detection_result.get('suggestions', []),
                    'data_quality_score': detection_result.get('data_quality_score', 0.0)
                },
                'step_name': step_name,
                'step_id': step_id,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'exception_type': 'mock_detection'
            }
        else:
            # 정상 데이터인 경우
            return {
                'success': True,
                'message': "데이터 품질 검증 완료",
                'data_quality_score': detection_result.get('data_quality_score', 1.0),
                'step_name': step_name,
                'step_id': step_id,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }
    
    except Exception as e:
        logger.error(f"목업 데이터 진단 중 오류: {e}")
        return {
            'success': False,
            'message': f"데이터 진단 중 오류 발생: {str(e)}",
            'error': 'DIAGNOSIS_ERROR',
            'step_name': step_name,
            'step_id': step_id,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        }


# 예외 변환 헬퍼 함수들
def convert_to_mycloset_exception(error: Exception, context: dict = None) -> MyClosetAIException:
    """일반 예외를 MyCloset AI 예외로 변환"""
    
    if isinstance(error, MyClosetAIException):
        return error
    
    error_message = str(error)
    error_type = type(error).__name__
    
    # 예외 타입별 변환
    if isinstance(error, FileNotFoundError):
        return FileOperationError(f"파일을 찾을 수 없습니다: {error_message}", "FILE_NOT_FOUND", context)
    
    elif isinstance(error, PermissionError):
        return FileOperationError(f"파일 접근 권한이 없습니다: {error_message}", "PERMISSION_DENIED", context)
    
    elif isinstance(error, MemoryError):
        return MemoryError(f"메모리 부족: {error_message}", "MEMORY_INSUFFICIENT", context)
    
    elif isinstance(error, ValueError):
        return DataValidationError(f"잘못된 값: {error_message}", "INVALID_VALUE", context)
    
    elif isinstance(error, TypeError):
        return DataValidationError(f"잘못된 타입: {error_message}", "INVALID_TYPE", context)
    
    elif isinstance(error, ImportError):
        return ConfigurationError(f"모듈 import 실패: {error_message}", "IMPORT_FAILED", context)
    
    elif isinstance(error, TimeoutError):
        return TimeoutError(f"타임아웃: {error_message}", "TIMEOUT", context)
    
    elif isinstance(error, ConnectionError):
        return NetworkError(f"네트워크 연결 오류: {error_message}", "CONNECTION_FAILED", context)
    
    else:
        # 기본적으로 VirtualFittingError로 변환
        return VirtualFittingError(f"처리 중 오류 발생: {error_message}", "UNEXPECTED_ERROR", context)


# 에러 코드 상수 정의
class ErrorCodes:
    """에러 코드 상수"""
    
    # 모델 관련
    MODEL_LOADING_FAILED = "MODEL_LOADING_FAILED"
    MODEL_FILE_NOT_FOUND = "MODEL_FILE_NOT_FOUND"
    MODEL_CORRUPTED = "MODEL_CORRUPTED"
    MODEL_VERSION_MISMATCH = "MODEL_VERSION_MISMATCH"
    
    # 이미지 처리 관련
    IMAGE_PROCESSING_FAILED = "IMAGE_PROCESSING_FAILED"
    IMAGE_FORMAT_INVALID = "IMAGE_FORMAT_INVALID"
    IMAGE_SIZE_INVALID = "IMAGE_SIZE_INVALID"
    BASE64_CONVERSION_FAILED = "BASE64_CONVERSION_FAILED"
    
    # 세션 관련
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    SESSION_CREATION_FAILED = "SESSION_CREATION_FAILED"
    
    # 의존성 주입 관련
    DI_CONTAINER_ERROR = "DI_CONTAINER_ERROR"
    SERVICE_NOT_FOUND = "SERVICE_NOT_FOUND"
    DEPENDENCY_CIRCULAR = "DEPENDENCY_CIRCULAR"
    
    # API 관련
    API_RESPONSE_ERROR = "API_RESPONSE_ERROR"
    INVALID_REQUEST = "INVALID_REQUEST"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    
    # 가상 피팅 관련
    VIRTUAL_FITTING_FAILED = "VIRTUAL_FITTING_FAILED"
    AI_INFERENCE_FAILED = "AI_INFERENCE_FAILED"
    GPU_MEMORY_INSUFFICIENT = "GPU_MEMORY_INSUFFICIENT"
    MPS_ERROR = "MPS_ERROR"
    
    # 파일 관련
    FILE_UPLOAD_FAILED = "FILE_UPLOAD_FAILED"
    FILE_DOWNLOAD_FAILED = "FILE_DOWNLOAD_FAILED"
    FILE_PERMISSION_DENIED = "FILE_PERMISSION_DENIED"
    
    # 메모리 관련
    MEMORY_INSUFFICIENT = "MEMORY_INSUFFICIENT"
    CACHE_FULL = "CACHE_FULL"
    
    # 네트워크 관련
    NETWORK_TIMEOUT = "NETWORK_TIMEOUT"
    CONNECTION_REFUSED = "CONNECTION_REFUSED"
    
    # 목업 데이터 관련
    MOCK_DATA_DETECTED = "MOCK_DATA_DETECTED"
    DATA_QUALITY_ISSUE = "DATA_QUALITY_ISSUE"
    TEST_DATA_USED = "TEST_DATA_USED"
    
    # 기타
    UNEXPECTED_ERROR = "UNEXPECTED_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"


# 진단 헬퍼 함수들
def diagnose_pipeline_issue(step_name: str, step_id: int, data: Any = None) -> dict:
    """파이프라인 문제 진단"""
    diagnosis = {
        'step_name': step_name,
        'step_id': step_id,
        'timestamp': datetime.now().isoformat(),
        'issues': [],
        'recommendations': []
    }
    
    try:
        # 에러 요약 조회
        error_summary = get_error_summary()
        
        # 해당 단계의 에러들 조회
        step_errors = error_tracker.get_errors_by_step(step_id)
        
        if step_errors:
            diagnosis['issues'].append({
                'type': 'step_errors',
                'count': len(step_errors),
                'details': step_errors[-5:]  # 최근 5개 에러
            })
        
        # 목업 데이터 분석
        if data is not None:
            mock_analysis = error_tracker.get_mock_data_analysis()
            if mock_analysis['total_mock_errors'] > 0:
                diagnosis['issues'].append({
                    'type': 'mock_data',
                    'count': mock_analysis['total_mock_errors'],
                    'details': mock_analysis['mock_error_details']
                })
        
        # 권장사항 생성
        if diagnosis['issues']:
            diagnosis['recommendations'].extend([
                "로그를 확인하여 구체적인 에러 원인을 파악하세요",
                "모델 파일들이 올바르게 로드되었는지 확인하세요",
                "입력 데이터의 형식과 크기를 확인하세요"
            ])
        
    except Exception as e:
        logger.error(f"진단 중 오류: {e}")
        diagnosis['error'] = str(e)
    
    return diagnosis


def log_detailed_error(error: Exception, context: dict = None, step_id: int = None):
    """상세한 에러 로깅"""
    try:
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'step_id': step_id,
            'context': context or {}
        }
        
        if isinstance(error, MyClosetAIException):
            error_info['error_code'] = error.error_code
            error_info['custom_context'] = error.context
        
        # 스택 트레이스 추가
        error_info['stack_trace'] = traceback.format_exc()
        
        # 로그 파일에 저장
        logger.error(f"상세 에러 정보: {json.dumps(error_info, indent=2, ensure_ascii=False)}")
        
        # 에러 추적
        track_exception(error, context, step_id)
        
    except Exception as e:
        logger.error(f"에러 로깅 중 오류: {e}")


# 목업 데이터 감지 데코레이터
def detect_mock_data_decorator(func):
    """함수 실행 전 목업 데이터 감지 데코레이터"""
    def wrapper(*args, **kwargs):
        try:
            # 입력 데이터에서 목업 데이터 감지
            for arg in args:
                if arg is not None:
                    detection_result = detect_mock_data(arg)
                    if detection_result['is_mock']:
                        logger.warning(f"목업 데이터 감지됨: {detection_result}")
                        # 에러 발생
                        raise MockDataDetectionError(
                            message="목업 데이터가 감지되었습니다",
                            error_code="MOCK_DATA_DETECTED",
                            context={'detection_result': detection_result}
                        )
            
            # 함수 실행
            return func(*args, **kwargs)
            
        except MockDataDetectionError:
            raise
        except Exception as e:
            # 다른 에러는 원래대로 처리
            raise e
    
    return wrapper


# ==============================================
# 🔥 BaseStepMixin 전용 에러 처리 헬퍼 함수들
# ==============================================

# ==============================================
# 🔥 Virtual Fitting 전용 에러 처리 헬퍼 함수들
# ==============================================

# ==============================================
# 🔥 Human Parsing 전용 에러 처리 헬퍼 함수들
# ==============================================

def handle_human_parsing_model_loading_error(model_name: str, error: Exception, checkpoint_path: str = None) -> dict:
    """Human Parsing 모델 로딩 에러 처리"""
    error_context = {
        'model_name': model_name,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'checkpoint_path': checkpoint_path,
        'timestamp': datetime.now().isoformat()
    }
    
    # 에러 추적
    track_exception(error, error_context)
    
    # 에러 타입별 처리
    if isinstance(error, (OSError, IOError)):
        return {
            'success': False,
            'error': 'PARSING_MODEL_FILE_ERROR',
            'message': f"{model_name} 체크포인트 파일 읽기 실패: {str(error)}",
            'error_context': error_context,
            'suggestion': f'{model_name} 체크포인트 파일이 올바른 경로에 있는지 확인하세요'
        }
    elif isinstance(error, (KeyError, ValueError)):
        return {
            'success': False,
            'error': 'PARSING_MODEL_FORMAT_ERROR',
            'message': f"{model_name} 체크포인트 형식 오류: {str(error)}",
            'error_context': error_context,
            'suggestion': f'{model_name} 체크포인트 파일이 손상되었을 수 있습니다. 재다운로드를 시도하세요'
        }
    elif isinstance(error, RuntimeError):
        return {
            'success': False,
            'error': 'PARSING_MODEL_LOADING_ERROR',
            'message': f"{model_name} 모델 로딩 실패: {str(error)}",
            'error_context': error_context,
            'suggestion': f'{model_name} 모델이 현재 환경과 호환되는지 확인하세요'
        }
    else:
        return {
            'success': False,
            'error': 'PARSING_MODEL_ERROR',
            'message': f"{model_name} 모델 오류: {str(error)}",
            'error_context': error_context,
            'suggestion': '로그를 확인하여 구체적인 오류 원인을 파악하세요'
        }


def handle_human_parsing_inference_error(model_name: str, error: Exception, inference_params: dict = None) -> dict:
    """Human Parsing 추론 에러 처리"""
    error_context = {
        'model_name': model_name,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'inference_params': inference_params or {},
        'timestamp': datetime.now().isoformat()
    }
    
    # 에러 추적
    track_exception(error, error_context)
    
    # 에러 타입별 처리
    if isinstance(error, (ValueError, TypeError)):
        return {
            'success': False,
            'error': 'PARSING_INFERENCE_INPUT_ERROR',
            'message': f"{model_name} 추론 입력 데이터 오류: {str(error)}",
            'error_context': error_context,
            'suggestion': '입력 이미지의 형식과 크기가 올바른지 확인하세요'
        }
    elif isinstance(error, RuntimeError):
        return {
            'success': False,
            'error': 'PARSING_INFERENCE_RUNTIME_ERROR',
            'message': f"{model_name} 추론 실행 오류: {str(error)}",
            'error_context': error_context,
            'suggestion': 'GPU 메모리가 충분한지 확인하거나 CPU 모드로 전환하세요'
        }
    elif isinstance(error, MemoryError):
        return {
            'success': False,
            'error': 'PARSING_INFERENCE_MEMORY_ERROR',
            'message': f"{model_name} 추론 메모리 부족: {str(error)}",
            'error_context': error_context,
            'suggestion': '입력 이미지 크기를 줄이거나 배치 크기를 줄여보세요'
        }
    else:
        return {
            'success': False,
            'error': 'PARSING_INFERENCE_ERROR',
            'message': f"{model_name} 추론 오류: {str(error)}",
            'error_context': error_context,
            'suggestion': '로그를 확인하여 구체적인 오류 원인을 파악하세요'
        }


def handle_image_preprocessing_error(operation: str, error: Exception, image_info: dict = None) -> dict:
    """이미지 전처리 에러 처리"""
    error_context = {
        'operation': operation,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'image_info': image_info or {},
        'timestamp': datetime.now().isoformat()
    }
    
    # 에러 추적
    track_exception(error, error_context)
    
    # 에러 타입별 처리
    if isinstance(error, (ValueError, TypeError)):
        return {
            'success': False,
            'error': 'IMAGE_PREPROCESSING_FORMAT_ERROR',
            'message': f"이미지 전처리 {operation} 형식 오류: {str(error)}",
            'error_context': error_context,
            'suggestion': '이미지 파일이 손상되지 않았는지 확인하세요'
        }
    elif isinstance(error, (OSError, IOError)):
        return {
            'success': False,
            'error': 'IMAGE_PREPROCESSING_IO_ERROR',
            'message': f"이미지 전처리 {operation} 입출력 오류: {str(error)}",
            'error_context': error_context,
            'suggestion': '이미지 파일에 접근 권한이 있는지 확인하세요'
        }
    elif isinstance(error, MemoryError):
        return {
            'success': False,
            'error': 'IMAGE_PREPROCESSING_MEMORY_ERROR',
            'message': f"이미지 전처리 {operation} 메모리 부족: {str(error)}",
            'error_context': error_context,
            'suggestion': '이미지 크기를 줄여보세요'
        }
    else:
        return {
            'success': False,
            'error': 'IMAGE_PREPROCESSING_ERROR',
            'message': f"이미지 전처리 {operation} 오류: {str(error)}",
            'error_context': error_context,
            'suggestion': '로그를 확인하여 구체적인 오류 원인을 파악하세요'
        }


def create_human_parsing_error_response(step_name: str, error: Exception, operation: str = "unknown", context: dict = None) -> dict:
    """Human Parsing 에러 응답 생성"""
    error_context = {
        'step_name': step_name,
        'operation': operation,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context or {},
        'timestamp': datetime.now().isoformat()
    }
    
    # 에러 추적
    track_exception(error, error_context)
    
    # 에러 타입별 처리
    if isinstance(error, (ImportError, ModuleNotFoundError)):
        return handle_step_initialization_error(step_name, error, {'operation': operation})
    elif isinstance(error, (OSError, IOError)):
        return handle_human_parsing_model_loading_error("Unknown", error)
    elif isinstance(error, (ValueError, TypeError)):
        return handle_human_parsing_inference_error("Unknown", error)
    elif isinstance(error, RuntimeError):
        return handle_human_parsing_inference_error("Unknown", error)
    elif isinstance(error, MemoryError):
        return handle_human_parsing_inference_error("Unknown", error)
    else:
        return {
            'success': False,
            'error': 'HUMAN_PARSING_ERROR',
            'message': f"{step_name} {operation} 실패: {str(error)}",
            'error_context': error_context,
            'suggestion': '로그를 확인하여 구체적인 오류 원인을 파악하세요'
        }


def validate_human_parsing_environment() -> dict:
    """Human Parsing 환경 검증"""
    validation_result = {
        'success': True,
        'checks': {},
        'timestamp': datetime.now().isoformat()
    }
    
    # PyTorch 사용 가능성 확인
    try:
        import torch
        validation_result['checks']['pytorch_available'] = True
        validation_result['checks']['pytorch_version'] = torch.__version__
        
        # CUDA 사용 가능성 확인
        if torch.cuda.is_available():
            validation_result['checks']['cuda_available'] = True
            validation_result['checks']['cuda_device_count'] = torch.cuda.device_count()
        else:
            validation_result['checks']['cuda_available'] = False
            
        # MPS 사용 가능성 확인
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            validation_result['checks']['mps_available'] = True
        else:
            validation_result['checks']['mps_available'] = False
            
    except ImportError:
        validation_result['checks']['pytorch_available'] = False
        validation_result['success'] = False
    
    # PIL 사용 가능성 확인
    try:
        from PIL import Image
        validation_result['checks']['pil_available'] = True
    except ImportError:
        validation_result['checks']['pil_available'] = False
        validation_result['success'] = False
    
    # OpenCV 사용 가능성 확인
    try:
        import cv2
        validation_result['checks']['opencv_available'] = True
        validation_result['checks']['opencv_version'] = cv2.__version__
    except ImportError:
        validation_result['checks']['opencv_available'] = False
    
    # NumPy 사용 가능성 확인
    try:
        import numpy as np
        validation_result['checks']['numpy_available'] = True
        validation_result['checks']['numpy_version'] = np.__version__
    except ImportError:
        validation_result['checks']['numpy_available'] = False
        validation_result['success'] = False
    
    # 메모리 사용량 확인
    try:
        import psutil
        memory = psutil.virtual_memory()
        validation_result['checks']['memory_available_gb'] = memory.available / (1024**3)
        validation_result['checks']['memory_total_gb'] = memory.total / (1024**3)
    except ImportError:
        validation_result['checks']['memory_info_available'] = False
    
    return validation_result


def log_human_parsing_performance(step_name: str, model_name: str, operation: str, start_time: float, success: bool, error: Exception = None, inference_params: dict = None) -> dict:
    """Human Parsing 성능 로깅"""
    end_time = time.time()
    duration = end_time - start_time
    
    performance_data = {
        'step_name': step_name,
        'model_name': model_name,
        'operation': operation,
        'duration_seconds': duration,
        'success': success,
        'inference_params': inference_params or {},
        'timestamp': datetime.now().isoformat()
    }
    
    if error:
        performance_data['error'] = {
            'type': type(error).__name__,
            'message': str(error)
        }
    
    # 성능 메트릭 업데이트
    if success:
        logger.info(f"✅ {step_name} {model_name} {operation} 완료 ({duration:.3f}초)")
    else:
        logger.error(f"❌ {step_name} {model_name} {operation} 실패 ({duration:.3f}초): {error}")
    
    return performance_data

def handle_virtual_fitting_model_loading_error(model_name: str, error: Exception, checkpoint_path: str = None) -> dict:
    """Virtual Fitting 모델 로딩 에러 처리"""
    error_context = {
        'model_name': model_name,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'checkpoint_path': checkpoint_path,
        'timestamp': datetime.now().isoformat()
    }
    
    # 에러 추적
    track_exception(error, error_context)
    
    # 에러 타입별 처리
    if isinstance(error, (OSError, IOError)):
        return {
            'success': False,
            'error': 'MODEL_FILE_ERROR',
            'message': f"{model_name} 체크포인트 파일 읽기 실패: {str(error)}",
            'error_context': error_context,
            'suggestion': f'{model_name} 체크포인트 파일이 올바른 경로에 있는지 확인하세요'
        }
    elif isinstance(error, (KeyError, ValueError)):
        return {
            'success': False,
            'error': 'MODEL_FORMAT_ERROR',
            'message': f"{model_name} 체크포인트 형식 오류: {str(error)}",
            'error_context': error_context,
            'suggestion': f'{model_name} 체크포인트 파일이 손상되었을 수 있습니다. 재다운로드를 시도하세요'
        }
    elif isinstance(error, RuntimeError):
        return {
            'success': False,
            'error': 'MODEL_LOADING_ERROR',
            'message': f"{model_name} 모델 로딩 실패: {str(error)}",
            'error_context': error_context,
            'suggestion': f'{model_name} 모델이 현재 환경과 호환되는지 확인하세요'
        }
    else:
        return {
            'success': False,
            'error': 'MODEL_ERROR',
            'message': f"{model_name} 모델 오류: {str(error)}",
            'error_context': error_context,
            'suggestion': '로그를 확인하여 구체적인 오류 원인을 파악하세요'
        }


def handle_virtual_fitting_inference_error(model_name: str, error: Exception, inference_params: dict = None) -> dict:
    """Virtual Fitting 추론 에러 처리"""
    error_context = {
        'model_name': model_name,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'inference_params': inference_params or {},
        'timestamp': datetime.now().isoformat()
    }
    
    # 에러 추적
    track_exception(error, error_context)
    
    # 에러 타입별 처리
    if isinstance(error, (ValueError, TypeError)):
        return {
            'success': False,
            'error': 'INFERENCE_INPUT_ERROR',
            'message': f"{model_name} 추론 입력 데이터 오류: {str(error)}",
            'error_context': error_context,
            'suggestion': '입력 이미지의 형식과 크기가 올바른지 확인하세요'
        }
    elif isinstance(error, RuntimeError):
        return {
            'success': False,
            'error': 'INFERENCE_RUNTIME_ERROR',
            'message': f"{model_name} 추론 실행 오류: {str(error)}",
            'error_context': error_context,
            'suggestion': 'GPU 메모리가 충분한지 확인하거나 CPU 모드로 전환하세요'
        }
    elif isinstance(error, MemoryError):
        return {
            'success': False,
            'error': 'INFERENCE_MEMORY_ERROR',
            'message': f"{model_name} 추론 메모리 부족: {str(error)}",
            'error_context': error_context,
            'suggestion': '입력 이미지 크기를 줄이거나 배치 크기를 줄여보세요'
        }
    else:
        return {
            'success': False,
            'error': 'INFERENCE_ERROR',
            'message': f"{model_name} 추론 오류: {str(error)}",
            'error_context': error_context,
            'suggestion': '로그를 확인하여 구체적인 오류 원인을 파악하세요'
        }


def handle_session_data_error(operation: str, error: Exception, session_id: str = None) -> dict:
    """세션 데이터 에러 처리"""
    error_context = {
        'operation': operation,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'session_id': session_id,
        'timestamp': datetime.now().isoformat()
    }
    
    # 에러 추적
    track_exception(error, error_context)
    
    # 에러 타입별 처리
    if isinstance(error, (KeyError, ValueError)):
        return {
            'success': False,
            'error': 'SESSION_DATA_ERROR',
            'message': f"세션 데이터 {operation} 오류: {str(error)}",
            'error_context': error_context,
            'suggestion': '세션 ID가 올바른지 확인하세요'
        }
    elif isinstance(error, TimeoutError):
        return {
            'success': False,
            'error': 'SESSION_TIMEOUT_ERROR',
            'message': f"세션 {operation} 타임아웃: {str(error)}",
            'error_context': error_context,
            'suggestion': '네트워크 연결을 확인하고 다시 시도하세요'
        }
    elif isinstance(error, RuntimeError):
        return {
            'success': False,
            'error': 'SESSION_RUNTIME_ERROR',
            'message': f"세션 {operation} 실행 오류: {str(error)}",
            'error_context': error_context,
            'suggestion': '세션 매니저가 올바르게 초기화되었는지 확인하세요'
        }
    else:
        return {
            'success': False,
            'error': 'SESSION_ERROR',
            'message': f"세션 {operation} 오류: {str(error)}",
            'error_context': error_context,
            'suggestion': '로그를 확인하여 구체적인 오류 원인을 파악하세요'
        }


def handle_image_processing_error(operation: str, error: Exception, image_info: dict = None) -> dict:
    """이미지 처리 에러 처리"""
    error_context = {
        'operation': operation,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'image_info': image_info or {},
        'timestamp': datetime.now().isoformat()
    }
    
    # 에러 추적
    track_exception(error, error_context)
    
    # 에러 타입별 처리
    if isinstance(error, (ValueError, TypeError)):
        return {
            'success': False,
            'error': 'IMAGE_FORMAT_ERROR',
            'message': f"이미지 {operation} 형식 오류: {str(error)}",
            'error_context': error_context,
            'suggestion': '이미지 파일이 손상되지 않았는지 확인하세요'
        }
    elif isinstance(error, (OSError, IOError)):
        return {
            'success': False,
            'error': 'IMAGE_IO_ERROR',
            'message': f"이미지 {operation} 입출력 오류: {str(error)}",
            'error_context': error_context,
            'suggestion': '이미지 파일에 접근 권한이 있는지 확인하세요'
        }
    elif isinstance(error, MemoryError):
        return {
            'success': False,
            'error': 'IMAGE_MEMORY_ERROR',
            'message': f"이미지 {operation} 메모리 부족: {str(error)}",
            'error_context': error_context,
            'suggestion': '이미지 크기를 줄여보세요'
        }
    else:
        return {
            'success': False,
            'error': 'IMAGE_PROCESSING_ERROR',
            'message': f"이미지 {operation} 오류: {str(error)}",
            'error_context': error_context,
            'suggestion': '로그를 확인하여 구체적인 오류 원인을 파악하세요'
        }


def create_virtual_fitting_error_response(step_name: str, error: Exception, operation: str = "unknown", context: dict = None) -> dict:
    """Virtual Fitting 에러 응답 생성"""
    error_context = {
        'step_name': step_name,
        'operation': operation,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context or {},
        'timestamp': datetime.now().isoformat()
    }
    
    # 에러 추적
    track_exception(error, error_context)
    
    # 에러 타입별 처리
    if isinstance(error, (ImportError, ModuleNotFoundError)):
        return handle_step_initialization_error(step_name, error, {'operation': operation})
    elif isinstance(error, (OSError, IOError)):
        return handle_virtual_fitting_model_loading_error("Unknown", error)
    elif isinstance(error, (ValueError, TypeError)):
        return handle_virtual_fitting_inference_error("Unknown", error)
    elif isinstance(error, RuntimeError):
        return handle_virtual_fitting_inference_error("Unknown", error)
    elif isinstance(error, MemoryError):
        return handle_virtual_fitting_inference_error("Unknown", error)
    elif isinstance(error, TimeoutError):
        return handle_session_data_error(operation, error)
    else:
        return {
            'success': False,
            'error': 'VIRTUAL_FITTING_ERROR',
            'message': f"{step_name} {operation} 실패: {str(error)}",
            'error_context': error_context,
            'suggestion': '로그를 확인하여 구체적인 오류 원인을 파악하세요'
        }


def validate_virtual_fitting_environment() -> dict:
    """Virtual Fitting 환경 검증"""
    validation_result = {
        'success': True,
        'checks': {},
        'timestamp': datetime.now().isoformat()
    }
    
    # PyTorch 사용 가능성 확인
    try:
        import torch
        validation_result['checks']['pytorch_available'] = True
        validation_result['checks']['pytorch_version'] = torch.__version__
        
        # CUDA 사용 가능성 확인
        if torch.cuda.is_available():
            validation_result['checks']['cuda_available'] = True
            validation_result['checks']['cuda_device_count'] = torch.cuda.device_count()
        else:
            validation_result['checks']['cuda_available'] = False
            
        # MPS 사용 가능성 확인
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            validation_result['checks']['mps_available'] = True
        else:
            validation_result['checks']['mps_available'] = False
            
    except ImportError:
        validation_result['checks']['pytorch_available'] = False
        validation_result['success'] = False
    
    # Diffusers 사용 가능성 확인
    try:
        import diffusers
        validation_result['checks']['diffusers_available'] = True
        validation_result['checks']['diffusers_version'] = diffusers.__version__
    except ImportError:
        validation_result['checks']['diffusers_available'] = False
    
    # PIL 사용 가능성 확인
    try:
        from PIL import Image
        validation_result['checks']['pil_available'] = True
    except ImportError:
        validation_result['checks']['pil_available'] = False
        validation_result['success'] = False
    
    # OpenCV 사용 가능성 확인
    try:
        import cv2
        validation_result['checks']['opencv_available'] = True
        validation_result['checks']['opencv_version'] = cv2.__version__
    except ImportError:
        validation_result['checks']['opencv_available'] = False
    
    # 메모리 사용량 확인
    try:
        import psutil
        memory = psutil.virtual_memory()
        validation_result['checks']['memory_available_gb'] = memory.available / (1024**3)
        validation_result['checks']['memory_total_gb'] = memory.total / (1024**3)
    except ImportError:
        validation_result['checks']['memory_info_available'] = False
    
    return validation_result


def log_virtual_fitting_performance(step_name: str, model_name: str, operation: str, start_time: float, success: bool, error: Exception = None, inference_params: dict = None) -> dict:
    """Virtual Fitting 성능 로깅"""
    end_time = time.time()
    duration = end_time - start_time
    
    performance_data = {
        'step_name': step_name,
        'model_name': model_name,
        'operation': operation,
        'duration_seconds': duration,
        'success': success,
        'inference_params': inference_params or {},
        'timestamp': datetime.now().isoformat()
    }
    
    if error:
        performance_data['error'] = {
            'type': type(error).__name__,
            'message': str(error)
        }
    
    # 성능 메트릭 업데이트
    if success:
        logger.info(f"✅ {step_name} {model_name} {operation} 완료 ({duration:.3f}초)")
    else:
        logger.error(f"❌ {step_name} {model_name} {operation} 실패 ({duration:.3f}초): {error}")
    
    return performance_data


# ==============================================
# 🔥 BaseStepMixin 전용 에러 처리 헬퍼 함수들
# ==============================================

def handle_step_initialization_error(step_name: str, error: Exception, context: dict = None) -> dict:
    """Step 초기화 에러 처리"""
    error_context = {
        'step_name': step_name,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context or {},
        'timestamp': datetime.now().isoformat()
    }
    
    # 에러 추적
    track_exception(error, error_context)
    
    # 에러 타입별 처리
    if isinstance(error, ImportError):
        return {
            'success': False,
            'error': 'IMPORT_ERROR',
            'message': f"{step_name} 모듈 import 실패: {str(error)}",
            'error_context': error_context,
            'suggestion': '필요한 패키지가 설치되었는지 확인하세요'
        }
    elif isinstance(error, AttributeError):
        return {
            'success': False,
            'error': 'ATTRIBUTE_ERROR',
            'message': f"{step_name} 속성 접근 실패: {str(error)}",
            'error_context': error_context,
            'suggestion': 'Step 클래스의 필수 속성이 정의되었는지 확인하세요'
        }
    elif isinstance(error, TypeError):
        return {
            'success': False,
            'error': 'TYPE_ERROR',
            'message': f"{step_name} 타입 오류: {str(error)}",
            'error_context': error_context,
            'suggestion': '메서드 호출 시 올바른 타입의 인자를 전달했는지 확인하세요'
        }
    else:
        return {
            'success': False,
            'error': 'INITIALIZATION_ERROR',
            'message': f"{step_name} 초기화 실패: {str(error)}",
            'error_context': error_context,
            'suggestion': '로그를 확인하여 구체적인 오류 원인을 파악하세요'
        }


def handle_dependency_injection_error(step_name: str, service_name: str, error: Exception) -> dict:
    """의존성 주입 에러 처리"""
    error_context = {
        'step_name': step_name,
        'service_name': service_name,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'timestamp': datetime.now().isoformat()
    }
    
    # 에러 추적
    track_exception(error, error_context)
    
    return {
        'success': False,
        'error': 'DEPENDENCY_INJECTION_ERROR',
        'message': f"{step_name} {service_name} 의존성 주입 실패: {str(error)}",
        'error_context': error_context,
        'suggestion': f'{service_name} 서비스가 Central Hub Container에 등록되었는지 확인하세요'
    }


def handle_data_conversion_error(step_name: str, conversion_type: str, error: Exception, data_info: dict = None) -> dict:
    """데이터 변환 에러 처리"""
    error_context = {
        'step_name': step_name,
        'conversion_type': conversion_type,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'data_info': data_info or {},
        'timestamp': datetime.now().isoformat()
    }
    
    # 에러 추적
    track_exception(error, error_context)
    
    return {
        'success': False,
        'error': 'DATA_CONVERSION_ERROR',
        'message': f"{step_name} {conversion_type} 변환 실패: {str(error)}",
        'error_context': error_context,
        'suggestion': '입력 데이터의 형식과 크기가 올바른지 확인하세요'
    }


def handle_central_hub_error(step_name: str, operation: str, error: Exception) -> dict:
    """Central Hub 연동 에러 처리"""
    error_context = {
        'step_name': step_name,
        'operation': operation,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'timestamp': datetime.now().isoformat()
    }
    
    # 에러 추적
    track_exception(error, error_context)
    
    return {
        'success': False,
        'error': 'CENTRAL_HUB_ERROR',
        'message': f"{step_name} Central Hub {operation} 실패: {str(error)}",
        'error_context': error_context,
        'suggestion': 'Central Hub DI Container가 올바르게 초기화되었는지 확인하세요'
    }


def create_step_error_response(step_name: str, error: Exception, operation: str = "unknown") -> dict:
    """Step 에러 응답 생성"""
    error_context = {
        'step_name': step_name,
        'operation': operation,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'timestamp': datetime.now().isoformat()
    }
    
    # 에러 추적
    track_exception(error, error_context)
    
    # 에러 타입별 처리
    if isinstance(error, (ImportError, ModuleNotFoundError)):
        return handle_step_initialization_error(step_name, error, {'operation': operation})
    elif isinstance(error, AttributeError):
        return handle_step_initialization_error(step_name, error, {'operation': operation})
    elif isinstance(error, TypeError):
        return handle_step_initialization_error(step_name, error, {'operation': operation})
    elif isinstance(error, ValueError):
        return handle_data_conversion_error(step_name, "validation", error)
    elif isinstance(error, (FileNotFoundError, OSError)):
        return {
            'success': False,
            'error': 'FILE_OPERATION_ERROR',
            'message': f"{step_name} 파일 작업 실패: {str(error)}",
            'error_context': error_context,
            'suggestion': '필요한 파일이 올바른 경로에 있는지 확인하세요'
        }
    else:
        return {
            'success': False,
            'error': 'STEP_ERROR',
            'message': f"{step_name} {operation} 실패: {str(error)}",
            'error_context': error_context,
            'suggestion': '로그를 확인하여 구체적인 오류 원인을 파악하세요'
        }


def validate_step_environment(step_name: str) -> dict:
    """Step 환경 검증"""
    validation_result = {
        'success': True,
        'step_name': step_name,
        'checks': {},
        'timestamp': datetime.now().isoformat()
    }
    
    # PyTorch 사용 가능성 확인
    try:
        import torch
        validation_result['checks']['pytorch_available'] = True
        validation_result['checks']['pytorch_version'] = torch.__version__
        
        # MPS 사용 가능성 확인
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            validation_result['checks']['mps_available'] = True
        else:
            validation_result['checks']['mps_available'] = False
            
    except ImportError:
        validation_result['checks']['pytorch_available'] = False
        validation_result['success'] = False
    
    # NumPy 사용 가능성 확인
    try:
        import numpy as np
        validation_result['checks']['numpy_available'] = True
        validation_result['checks']['numpy_version'] = np.__version__
    except ImportError:
        validation_result['checks']['numpy_available'] = False
        validation_result['success'] = False
    
    # PIL 사용 가능성 확인
    try:
        from PIL import Image
        validation_result['checks']['pil_available'] = True
    except ImportError:
        validation_result['checks']['pil_available'] = False
        validation_result['success'] = False
    
    # OpenCV 사용 가능성 확인
    try:
        import cv2
        validation_result['checks']['opencv_available'] = True
        validation_result['checks']['opencv_version'] = cv2.__version__
    except ImportError:
        validation_result['checks']['opencv_available'] = False
    
    # 메모리 사용량 확인
    try:
        import psutil
        memory = psutil.virtual_memory()
        validation_result['checks']['memory_available_gb'] = memory.available / (1024**3)
        validation_result['checks']['memory_total_gb'] = memory.total / (1024**3)
    except ImportError:
        validation_result['checks']['memory_info_available'] = False
    
    return validation_result


def log_step_performance(step_name: str, operation: str, start_time: float, success: bool, error: Exception = None) -> dict:
    """Step 성능 로깅"""
    end_time = time.time()
    duration = end_time - start_time
    
    performance_data = {
        'step_name': step_name,
        'operation': operation,
        'duration_seconds': duration,
        'success': success,
        'timestamp': datetime.now().isoformat()
    }
    
    if error:
        performance_data['error'] = {
            'type': type(error).__name__,
            'message': str(error)
        }
    
    # 성능 메트릭 업데이트
    if success:
        logger.info(f"✅ {step_name} {operation} 완료 ({duration:.3f}초)")
    else:
        logger.error(f"❌ {step_name} {operation} 실패 ({duration:.3f}초): {error}")
    
    return performance_data 