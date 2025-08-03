"""
🔥 MyCloset AI 목업 데이터 진단 시스템
파이프라인 각 단계에서 목업 데이터를 감지하고 문제를 진단하는 전용 도구
"""

import json
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import numpy as np
import cv2
from pathlib import Path

from .exceptions import (
    MockDataDetectionError, 
    DataQualityError, 
    ModelInferenceError,
    error_tracker,
    detect_mock_data,
    log_detailed_error
)

logger = logging.getLogger(__name__)


class MockDataDiagnostic:
    """목업 데이터 진단 시스템"""
    
    def __init__(self):
        self.diagnostic_history = []
        self.max_history = 100
        self.diagnostic_config = {
            'image_quality_threshold': 0.3,
            'text_quality_threshold': 0.5,
            'data_quality_threshold': 0.4,
            'enable_detailed_logging': True,
            'save_diagnostic_reports': True
        }
    
    def diagnose_pipeline_step(
        self, 
        step_name: str, 
        step_id: int, 
        input_data: Any = None,
        output_data: Any = None,
        model_info: Dict[str, Any] = None,
        session_id: str = "unknown"
    ) -> Dict[str, Any]:
        """파이프라인 단계별 진단"""
        
        diagnostic_result = {
            'step_name': step_name,
            'step_id': step_id,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'input_diagnosis': {},
            'output_diagnosis': {},
            'model_diagnosis': {},
            'overall_health': 'unknown',
            'issues': [],
            'recommendations': []
        }
        
        try:
            # 입력 데이터 진단
            if input_data is not None:
                diagnostic_result['input_diagnosis'] = self._diagnose_data(
                    input_data, f"{step_name}_input", step_id
                )
            
            # 출력 데이터 진단
            if output_data is not None:
                diagnostic_result['output_diagnosis'] = self._diagnose_data(
                    output_data, f"{step_name}_output", step_id
                )
            
            # 모델 정보 진단
            if model_info is not None:
                diagnostic_result['model_diagnosis'] = self._diagnose_model(model_info, step_id)
            
            # 전체 건강도 평가
            diagnostic_result['overall_health'] = self._evaluate_overall_health(diagnostic_result)
            
            # 문제점 식별
            diagnostic_result['issues'] = self._identify_issues(diagnostic_result)
            
            # 권장사항 생성
            diagnostic_result['recommendations'] = self._generate_recommendations(
                diagnostic_result['issues'], step_name
            )
            
            # 진단 결과 저장
            self._save_diagnostic_result(diagnostic_result)
            
            # 로깅
            if self.diagnostic_config['enable_detailed_logging']:
                self._log_diagnostic_result(diagnostic_result)
            
        except Exception as e:
            logger.error(f"진단 중 오류 발생: {e}")
            diagnostic_result['error'] = str(e)
            diagnostic_result['overall_health'] = 'error'
        
        return diagnostic_result
    
    def _diagnose_data(self, data: Any, data_name: str, step_id: int) -> Dict[str, Any]:
        """데이터 진단"""
        diagnosis = {
            'data_name': data_name,
            'data_type': type(data).__name__,
            'mock_detection': {},
            'quality_assessment': {},
            'shape_info': {},
            'value_analysis': {},
            'issues': []
        }
        
        try:
            # 목업 데이터 감지
            mock_result = detect_mock_data(data)
            diagnosis['mock_detection'] = mock_result
            
            # 데이터 품질 평가
            diagnosis['quality_assessment'] = self._assess_data_quality(data)
            
            # 형태 정보 분석
            diagnosis['shape_info'] = self._analyze_shape_info(data)
            
            # 값 분석
            diagnosis['value_analysis'] = self._analyze_values(data)
            
            # 문제점 식별
            if mock_result['is_mock']:
                diagnosis['issues'].append({
                    'type': 'mock_data',
                    'severity': 'high',
                    'description': '목업 데이터가 감지되었습니다',
                    'details': mock_result
                })
            
            # 품질 문제 식별
            quality_score = diagnosis['quality_assessment'].get('overall_score', 1.0)
            if quality_score < 0.5:
                diagnosis['issues'].append({
                    'type': 'low_quality',
                    'severity': 'medium',
                    'description': f'데이터 품질이 낮습니다 (점수: {quality_score:.2f})',
                    'details': diagnosis['quality_assessment']
                })
            
        except Exception as e:
            logger.error(f"데이터 진단 중 오류: {e}")
            diagnosis['error'] = str(e)
        
        return diagnosis
    
    def _diagnose_model(self, model_info: Dict[str, Any], step_id: int) -> Dict[str, Any]:
        """모델 정보 진단"""
        diagnosis = {
            'model_status': 'unknown',
            'model_type': model_info.get('model_type', 'unknown'),
            'model_path': model_info.get('model_path', 'unknown'),
            'model_loaded': model_info.get('loaded', False),
            'model_size': model_info.get('size', 0),
            'issues': []
        }
        
        try:
            # 모델 로딩 상태 확인
            if not model_info.get('loaded', False):
                diagnosis['issues'].append({
                    'type': 'model_not_loaded',
                    'severity': 'high',
                    'description': '모델이 로드되지 않았습니다'
                })
                diagnosis['model_status'] = 'not_loaded'
            else:
                diagnosis['model_status'] = 'loaded'
            
            # 모델 경로 확인
            model_path = model_info.get('model_path', '')
            if model_path and not Path(model_path).exists():
                diagnosis['issues'].append({
                    'type': 'model_file_missing',
                    'severity': 'high',
                    'description': f'모델 파일이 존재하지 않습니다: {model_path}'
                })
            
            # 모델 크기 확인
            model_size = model_info.get('size', 0)
            if model_size == 0:
                diagnosis['issues'].append({
                    'type': 'model_size_zero',
                    'severity': 'medium',
                    'description': '모델 크기가 0입니다'
                })
            
        except Exception as e:
            logger.error(f"모델 진단 중 오류: {e}")
            diagnosis['error'] = str(e)
        
        return diagnosis
    
    def _assess_data_quality(self, data: Any) -> Dict[str, Any]:
        """데이터 품질 평가"""
        quality_assessment = {
            'overall_score': 1.0,
            'diversity_score': 1.0,
            'complexity_score': 1.0,
            'validity_score': 1.0
        }
        
        try:
            if isinstance(data, np.ndarray):
                # 이미지 데이터 품질 평가
                if len(data.shape) >= 2:
                    # 다양성 점수 (표준편차 기반)
                    std_dev = np.std(data)
                    quality_assessment['diversity_score'] = min(1.0, std_dev / 50.0)
                    
                    # 복잡성 점수 (엔트로피 기반)
                    if data.dtype in [np.uint8, np.float32, np.float64]:
                        hist, _ = np.histogram(data.flatten(), bins=256)
                        hist = hist[hist > 0]
                        if len(hist) > 0:
                            entropy = -np.sum(hist * np.log2(hist + 1e-10))
                            quality_assessment['complexity_score'] = min(1.0, entropy / 8.0)
                    
                    # 유효성 점수 (NaN, Inf 체크)
                    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                        quality_assessment['validity_score'] = 0.0
                    
                    # 전체 점수 계산
                    quality_assessment['overall_score'] = (
                        quality_assessment['diversity_score'] * 0.4 +
                        quality_assessment['complexity_score'] * 0.4 +
                        quality_assessment['validity_score'] * 0.2
                    )
            
            elif isinstance(data, list):
                # 리스트 데이터 품질 평가
                if len(data) == 0:
                    quality_assessment['overall_score'] = 0.0
                else:
                    # 다양성 점수
                    unique_count = len(set(data))
                    quality_assessment['diversity_score'] = min(1.0, unique_count / len(data))
                    
                    # 전체 점수
                    quality_assessment['overall_score'] = quality_assessment['diversity_score']
            
            elif isinstance(data, str):
                # 텍스트 데이터 품질 평가
                if not data or data.strip() == '':
                    quality_assessment['overall_score'] = 0.0
                else:
                    # 텍스트 길이와 다양성 평가
                    quality_assessment['diversity_score'] = min(1.0, len(set(data)) / len(data))
                    quality_assessment['overall_score'] = quality_assessment['diversity_score']
        
        except Exception as e:
            logger.error(f"데이터 품질 평가 중 오류: {e}")
            quality_assessment['overall_score'] = 0.0
        
        return quality_assessment
    
    def _analyze_shape_info(self, data: Any) -> Dict[str, Any]:
        """형태 정보 분석"""
        shape_info = {
            'shape': None,
            'dimensions': 0,
            'size': 0,
            'is_empty': False
        }
        
        try:
            if isinstance(data, np.ndarray):
                shape_info['shape'] = data.shape
                shape_info['dimensions'] = len(data.shape)
                shape_info['size'] = data.size
                shape_info['is_empty'] = data.size == 0
            
            elif isinstance(data, list):
                shape_info['shape'] = (len(data),)
                shape_info['dimensions'] = 1
                shape_info['size'] = len(data)
                shape_info['is_empty'] = len(data) == 0
            
            elif isinstance(data, str):
                shape_info['shape'] = (len(data),)
                shape_info['dimensions'] = 1
                shape_info['size'] = len(data)
                shape_info['is_empty'] = len(data) == 0
        
        except Exception as e:
            logger.error(f"형태 정보 분석 중 오류: {e}")
        
        return shape_info
    
    def _analyze_values(self, data: Any) -> Dict[str, Any]:
        """값 분석"""
        value_analysis = {
            'min_value': None,
            'max_value': None,
            'mean_value': None,
            'std_value': None,
            'unique_count': 0,
            'has_nan': False,
            'has_inf': False
        }
        
        try:
            if isinstance(data, np.ndarray):
                if data.size > 0:
                    value_analysis['min_value'] = float(np.min(data))
                    value_analysis['max_value'] = float(np.max(data))
                    value_analysis['mean_value'] = float(np.mean(data))
                    value_analysis['std_value'] = float(np.std(data))
                    value_analysis['unique_count'] = len(np.unique(data))
                    value_analysis['has_nan'] = np.any(np.isnan(data))
                    value_analysis['has_inf'] = np.any(np.isinf(data))
            
            elif isinstance(data, list):
                if len(data) > 0:
                    numeric_data = [x for x in data if isinstance(x, (int, float))]
                    if numeric_data:
                        value_analysis['min_value'] = min(numeric_data)
                        value_analysis['max_value'] = max(numeric_data)
                        value_analysis['mean_value'] = sum(numeric_data) / len(numeric_data)
                        value_analysis['unique_count'] = len(set(data))
        
        except Exception as e:
            logger.error(f"값 분석 중 오류: {e}")
        
        return value_analysis
    
    def _evaluate_overall_health(self, diagnostic_result: Dict[str, Any]) -> str:
        """전체 건강도 평가"""
        issues = diagnostic_result.get('issues', [])
        
        # 심각한 문제가 있는지 확인
        high_severity_issues = [issue for issue in issues if issue.get('severity') == 'high']
        if high_severity_issues:
            return 'critical'
        
        # 중간 수준의 문제가 있는지 확인
        medium_severity_issues = [issue for issue in issues if issue.get('severity') == 'medium']
        if medium_severity_issues:
            return 'warning'
        
        # 문제가 없으면 정상
        return 'healthy'
    
    def _identify_issues(self, diagnostic_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """문제점 식별"""
        issues = []
        
        # 입력 데이터 문제
        input_diagnosis = diagnostic_result.get('input_diagnosis', {})
        if input_diagnosis.get('issues'):
            issues.extend(input_diagnosis['issues'])
        
        # 출력 데이터 문제
        output_diagnosis = diagnostic_result.get('output_diagnosis', {})
        if output_diagnosis.get('issues'):
            issues.extend(output_diagnosis['issues'])
        
        # 모델 문제
        model_diagnosis = diagnostic_result.get('model_diagnosis', {})
        if model_diagnosis.get('issues'):
            issues.extend(model_diagnosis['issues'])
        
        return issues
    
    def _generate_recommendations(self, issues: List[Dict[str, Any]], step_name: str) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        for issue in issues:
            issue_type = issue.get('type', '')
            
            if issue_type == 'mock_data':
                recommendations.append("실제 사용자 데이터를 사용하세요")
                recommendations.append("테스트 데이터 대신 실제 이미지를 업로드하세요")
            
            elif issue_type == 'low_quality':
                recommendations.append("입력 데이터의 품질을 확인하세요")
                recommendations.append("이미지 해상도와 형식을 확인하세요")
            
            elif issue_type == 'model_not_loaded':
                recommendations.append("모델 파일이 올바르게 로드되었는지 확인하세요")
                recommendations.append("모델 경로와 파일 존재 여부를 확인하세요")
            
            elif issue_type == 'model_file_missing':
                recommendations.append("모델 파일을 다운로드하거나 경로를 수정하세요")
                recommendations.append("모델 파일의 권한을 확인하세요")
        
        # 단계별 특화 권장사항
        if step_name == 'human_parsing':
            recommendations.append("사람이 포함된 이미지를 사용하세요")
        elif step_name == 'pose_estimation':
            recommendations.append("전신이 보이는 이미지를 사용하세요")
        elif step_name == 'cloth_segmentation':
            recommendations.append("의류가 명확히 보이는 이미지를 사용하세요")
        
        return list(set(recommendations))  # 중복 제거
    
    def _save_diagnostic_result(self, diagnostic_result: Dict[str, Any]):
        """진단 결과 저장"""
        if not self.diagnostic_config['save_diagnostic_reports']:
            return
        
        try:
            # 진단 히스토리에 추가
            self.diagnostic_history.append(diagnostic_result)
            
            # 최대 개수 유지
            if len(self.diagnostic_history) > self.max_history:
                self.diagnostic_history = self.diagnostic_history[-self.max_history:]
            
            # 파일로 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"diagnostic_report_{timestamp}.json"
            
            # logs 디렉토리 생성
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            filepath = log_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(diagnostic_result, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"진단 결과 저장됨: {filepath}")
        
        except Exception as e:
            logger.error(f"진단 결과 저장 중 오류: {e}")
    
    def _log_diagnostic_result(self, diagnostic_result: Dict[str, Any]):
        """진단 결과 로깅"""
        try:
            step_name = diagnostic_result.get('step_name', 'unknown')
            step_id = diagnostic_result.get('step_id', 'unknown')
            health = diagnostic_result.get('overall_health', 'unknown')
            issues_count = len(diagnostic_result.get('issues', []))
            
            log_message = f"진단 결과 - 단계: {step_name} (ID: {step_id}), 건강도: {health}, 문제수: {issues_count}"
            
            if health == 'critical':
                logger.error(log_message)
            elif health == 'warning':
                logger.warning(log_message)
            else:
                logger.info(log_message)
            
            # 상세 정보 로깅
            if issues_count > 0:
                for issue in diagnostic_result.get('issues', []):
                    logger.warning(f"  - {issue.get('type', 'unknown')}: {issue.get('description', '')}")
        
        except Exception as e:
            logger.error(f"진단 결과 로깅 중 오류: {e}")
    
    def get_diagnostic_summary(self) -> Dict[str, Any]:
        """진단 요약 정보"""
        if not self.diagnostic_history:
            return {'message': '진단 기록이 없습니다'}
        
        try:
            # 전체 통계
            total_diagnoses = len(self.diagnostic_history)
            health_counts = {}
            issue_types = {}
            
            for diagnosis in self.diagnostic_history:
                health = diagnosis.get('overall_health', 'unknown')
                health_counts[health] = health_counts.get(health, 0) + 1
                
                for issue in diagnosis.get('issues', []):
                    issue_type = issue.get('type', 'unknown')
                    issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
            
            # 최근 진단들
            recent_diagnoses = self.diagnostic_history[-10:] if self.diagnostic_history else []
            
            return {
                'total_diagnoses': total_diagnoses,
                'health_distribution': health_counts,
                'issue_type_distribution': issue_types,
                'recent_diagnoses': recent_diagnoses,
                'diagnostic_period': {
                    'start': self.diagnostic_history[0]['timestamp'] if self.diagnostic_history else None,
                    'end': self.diagnostic_history[-1]['timestamp'] if self.diagnostic_history else None
                }
            }
        
        except Exception as e:
            logger.error(f"진단 요약 생성 중 오류: {e}")
            return {'error': str(e)}
    
    def clear_diagnostic_history(self):
        """진단 히스토리 초기화"""
        self.diagnostic_history.clear()
        logger.info("진단 히스토리가 초기화되었습니다")


# 전역 진단 인스턴스
mock_data_diagnostic = MockDataDiagnostic()


def diagnose_step_data(
    step_name: str,
    step_id: int,
    input_data: Any = None,
    output_data: Any = None,
    model_info: Dict[str, Any] = None,
    session_id: str = "unknown"
) -> Dict[str, Any]:
    """단계별 데이터 진단"""
    return mock_data_diagnostic.diagnose_pipeline_step(
        step_name, step_id, input_data, output_data, model_info, session_id
    )


def get_diagnostic_summary() -> Dict[str, Any]:
    """진단 요약 조회"""
    return mock_data_diagnostic.get_diagnostic_summary()


def clear_diagnostic_history():
    """진단 히스토리 초기화"""
    mock_data_diagnostic.clear_diagnostic_history()


# 진단 데코레이터
def diagnostic_decorator(step_name: str = None):
    """진단 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            step_id = kwargs.get('step_id', 0)
            session_id = kwargs.get('session_id', 'unknown')
            
            # 함수 실행 전 진단
            try:
                # 입력 데이터 진단
                if args:
                    input_data = args[0] if args else None
                    diagnose_step_data(
                        step_name or func.__name__,
                        step_id,
                        input_data=input_data,
                        session_id=session_id
                    )
                
                # 함수 실행
                result = func(*args, **kwargs)
                
                # 출력 데이터 진단
                diagnose_step_data(
                    step_name or func.__name__,
                    step_id,
                    output_data=result,
                    session_id=session_id
                )
                
                return result
            
            except Exception as e:
                # 에러 발생 시 진단
                log_detailed_error(e, {
                    'step_name': step_name or func.__name__,
                    'step_id': step_id,
                    'session_id': session_id
                }, step_id)
                raise
        
        return wrapper
    return decorator 