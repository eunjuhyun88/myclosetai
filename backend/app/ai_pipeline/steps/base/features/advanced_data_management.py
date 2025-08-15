#!/usr/bin/env python3
"""
🔥 MyCloset AI - Advanced Data Management Mixin
===============================================

고급 데이터 관리 기능을 담당하는 Mixin 클래스
- DetailedDataSpec 관리
- 데이터 완전성 검증
- 메모리 최적화
- 데이터 흐름 분석

Author: MyCloset AI Team
Date: 2025-08-14
Version: 2.0
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union

# 선택적 import
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

class AdvancedDataManagementMixin:
    """고급 데이터 관리 기능을 담당하는 Mixin"""

    def _load_detailed_data_spec_from_kwargs(self, **kwargs):
        """StepFactory에서 주입받은 DetailedDataSpec 정보 로딩"""
        try:
            from .central_hub import DetailedDataSpecConfig
            
            config = DetailedDataSpecConfig(
                # 입력 사양
                input_data_types=kwargs.get('input_data_types', []),
                input_shapes=kwargs.get('input_shapes', {}),
                input_value_ranges=kwargs.get('input_value_ranges', {}),
                preprocessing_required=kwargs.get('preprocessing_required', []),
                
                # 출력 사양
                output_data_types=kwargs.get('output_data_types', []),
                output_shapes=kwargs.get('output_shapes', {}),
                output_value_ranges=kwargs.get('output_value_ranges', {}),
                postprocessing_required=kwargs.get('postprocessing_required', []),
                
                # API 호환성
                api_input_mapping=kwargs.get('api_input_mapping', {}),
                api_output_mapping=kwargs.get('api_output_mapping', {}),
                
                # Step 간 연동
                step_input_schema=kwargs.get('step_input_schema', {}),
                step_output_schema=kwargs.get('step_output_schema', {}),
                
                # 전처리/후처리 요구사항
                normalization_mean=kwargs.get('normalization_mean', (0.485, 0.456, 0.406)),
                normalization_std=kwargs.get('normalization_std', (0.229, 0.224, 0.225)),
                preprocessing_steps=kwargs.get('preprocessing_steps', []),
                postprocessing_steps=kwargs.get('postprocessing_steps', []),
                
                # Step 간 데이터 전달 스키마
                accepts_from_previous_step=kwargs.get('accepts_from_previous_step', {}),
                provides_to_next_step=kwargs.get('provides_to_next_step', {})
            )
            
            self.detailed_data_spec = config
            self.logger.debug(f"✅ {self.step_name} DetailedDataSpec 로딩 완료")
            return config
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} DetailedDataSpec 로딩 실패: {e}")
            return self._create_emergency_detailed_data_spec()

    def _validate_data_conversion_readiness(self) -> bool:
        """데이터 변환 준비 상태 검증 (워닝 방지)"""
        try:
            # DetailedDataSpec 존재 확인 및 자동 생성
            if not hasattr(self, 'detailed_data_spec') or not self.detailed_data_spec:
                self._create_emergency_detailed_data_spec()
                self.logger.debug(f"✅ {self.step_name} DetailedDataSpec 기본값 자동 생성")
            
            # 필수 필드 존재 확인 및 자동 보완
            missing_fields = []
            required_fields = ['input_data_types', 'output_data_types', 'api_input_mapping', 'api_output_mapping']
            
            for field in required_fields:
                if not hasattr(self.detailed_data_spec, field):
                    missing_fields.append(field)
                else:
                    value = getattr(self.detailed_data_spec, field)
                    if not value:
                        missing_fields.append(field)
            
            # 누락된 필드 자동 보완
            if missing_fields:
                self._fill_missing_fields(missing_fields)
                self.logger.debug(f"{self.step_name} DetailedDataSpec 필드 보완: {missing_fields}")
            
            # dependency_manager 상태 업데이트
            if hasattr(self, 'dependency_manager') and self.dependency_manager:
                self.dependency_manager.dependency_status.detailed_data_spec_loaded = True
                self.dependency_manager.dependency_status.data_conversion_ready = True
            
            self.logger.debug(f"✅ {self.step_name} DetailedDataSpec 데이터 변환 준비 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 데이터 변환 준비 상태 검증 실패: {e}")
            return False

    def _create_emergency_detailed_data_spec(self):
        """응급 DetailedDataSpec 생성"""
        try:
            if not hasattr(self, 'detailed_data_spec') or not self.detailed_data_spec:
                class EmergencyDataSpec:
                    def __init__(self):
                        self.input_data_types = {
                            'person_image': 'PIL.Image.Image',
                            'clothing_image': 'PIL.Image.Image',
                            'data': 'Any'
                        }
                        self.output_data_types = {
                            'result': 'numpy.ndarray',
                            'confidence': 'float',
                            'metadata': 'dict'
                        }
                        self.api_input_mapping = {
                            'person_image': 'person_image',
                            'clothing_image': 'clothing_image'
                        }
                        self.api_output_mapping = {
                            'result': 'result',
                            'confidence': 'confidence'
                        }
                        self.preprocessing_steps = ['resize', 'normalize']
                        self.postprocessing_steps = ['denormalize', 'to_numpy']
                
                self.detailed_data_spec = EmergencyDataSpec()
                self.logger.info(f"✅ {self.step_name} 응급 DetailedDataSpec 생성 완료")
                
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 응급 DetailedDataSpec 생성 실패: {e}")

    def _fill_missing_fields(self, missing_fields: List[str]):
        """누락된 필드 자동 보완"""
        try:
            for field in missing_fields:
                if field == 'input_data_types':
                    self.detailed_data_spec.input_data_types = {
                        'image': 'PIL.Image.Image',
                        'data': 'Any'
                    }
                elif field == 'output_data_types':
                    self.detailed_data_spec.output_data_types = {
                        'result': 'numpy.ndarray',
                        'confidence': 'float'
                    }
                elif field == 'api_input_mapping':
                    self.detailed_data_spec.api_input_mapping = {
                        'image': 'image',
                        'data': 'data'
                    }
                elif field == 'api_output_mapping':
                    self.detailed_data_spec.api_output_mapping = {
                        'result': 'result',
                        'confidence': 'confidence'
                    }
            
            self.logger.debug(f"✅ {self.step_name} 누락된 필드 보완 완료: {missing_fields}")
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 누락된 필드 보완 실패: {e}")

    def _optimize_memory_usage(self, data: Dict[str, Any], target_device: str = None) -> Dict[str, Any]:
        """메모리 사용량 최적화"""
        optimized_data = {}
        memory_saved_mb = 0.0
        
        try:
            target_device = target_device or getattr(self, 'device', 'cpu')
            
            for key, value in data.items():
                try:
                    # 텐서 최적화
                    if TORCH_AVAILABLE and torch.is_tensor(value):
                        original_size = value.element_size() * value.nelement() / (1024 * 1024)
                        
                        # 디바이스 최적화
                        if target_device == "cpu" and value.device.type != "cpu":
                            value = value.cpu()
                        elif target_device == "mps" and value.device.type != "mps":
                            value = value.to("mps")
                        
                        # FP16 변환 (메모리 절약)
                        if hasattr(self, 'config') and getattr(self.config, 'use_fp16', False):
                            if value.dtype == torch.float32:
                                value = value.half()
                        
                        optimized_size = value.element_size() * value.nelement() / (1024 * 1024)
                        memory_saved_mb += (original_size - optimized_size)
                        
                    # NumPy 배열 최적화
                    elif NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                        original_size = value.nbytes / (1024 * 1024)
                        
                        # 불필요한 복사 방지
                        if value.flags['C_CONTIGUOUS']:
                            optimized_value = value
                        else:
                            optimized_value = np.ascontiguousarray(value)
                        
                        # 데이터 타입 최적화
                        if hasattr(self, 'config') and getattr(self.config, 'use_fp16', False):
                            if value.dtype == np.float64:
                                optimized_value = optimized_value.astype(np.float16)
                        
                        optimized_size = optimized_value.nbytes / (1024 * 1024)
                        memory_saved_mb += (original_size - optimized_size)
                        value = optimized_value
                    
                    optimized_data[key] = value
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ {key} 메모리 최적화 실패: {e}")
                    optimized_data[key] = value
            
            if memory_saved_mb > 0:
                self.logger.info(f"💾 메모리 최적화 완료: {memory_saved_mb:.2f}MB 절약")
                if hasattr(self, 'performance_metrics'):
                    self.performance_metrics.memory_optimizations += 1
            
            return optimized_data
            
        except Exception as e:
            self.logger.error(f"❌ 메모리 최적화 실패: {e}")
            return data

    def _analyze_di_container_data_flow(self, step_result: Dict[str, Any], step_id: int) -> Dict[str, Any]:
        """DI Container 데이터 흐름 분석"""
        analysis_result = {
            'di_container_used': False,
            'services_accessed': [],
            'data_flow_path': [],
            'memory_optimizations': 0,
            'data_transfers': 0,
            'errors': []
        }
        
        try:
            # DI Container 사용 여부 확인
            if hasattr(self, 'di_container') and self.di_container:
                analysis_result['di_container_used'] = True
                
                # Central Hub 서비스 접근 확인
                central_hub_services = ['memory_manager', 'model_loader', 'data_converter']
                for service_name in central_hub_services:
                    try:
                        service = self.get_service(service_name)
                        if service:
                            analysis_result['services_accessed'].append(service_name)
                    except Exception as e:
                        analysis_result['errors'].append(f"서비스 접근 실패 ({service_name}): {e}")
                
                # 데이터 흐름 경로 분석
                if hasattr(self, 'detailed_data_spec'):
                    provides_to_next = getattr(self.detailed_data_spec, 'provides_to_next_step', {})
                    for next_step, data_mapping in provides_to_next.items():
                        analysis_result['data_flow_path'].append({
                            'from_step': step_id,
                            'to_step': next_step,
                            'data_keys': list(data_mapping.keys())
                        })
                        analysis_result['data_transfers'] += 1
                
                # 메모리 최적화 확인
                if hasattr(self, 'performance_metrics'):
                    analysis_result['memory_optimizations'] = getattr(
                        self.performance_metrics, 'memory_optimizations', 0
                    )
            
            # 로깅
            if analysis_result['di_container_used']:
                self.logger.info(f"🔗 DI Container 데이터 흐름 분석 완료")
                self.logger.debug(f"   - 사용된 서비스: {analysis_result['services_accessed']}")
                self.logger.debug(f"   - 데이터 전달 경로: {len(analysis_result['data_flow_path'])}개")
            else:
                self.logger.warning(f"⚠️ DI Container 미사용")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"❌ DI Container 데이터 흐름 분석 실패: {e}")
            analysis_result['errors'].append(f"분석 오류: {e}")
            return analysis_result

    def _create_data_transfer_report(self, step_id: int, step_result: Dict[str, Any], 
                                   processing_time: float) -> Dict[str, Any]:
        """데이터 전달 종합 리포트 생성"""
        report = {
            'step_id': step_id,
            'step_name': getattr(self, 'step_name', 'Unknown'),
            'processing_time': processing_time,
            'timestamp': time.time(),
            'data_completeness': {},
            'memory_usage': {},
            'di_container_analysis': {},
            'performance_metrics': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            # 1. 데이터 완전성 검증
            if hasattr(self, 'detailed_data_spec'):
                expected_outputs = getattr(self.detailed_data_spec, 'provides_to_next_step', {})
                all_expected_keys = []
                for data_mapping in expected_outputs.values():
                    all_expected_keys.extend(data_mapping.keys())
                
                if all_expected_keys:
                    report['data_completeness'] = self._validate_data_completeness(
                        step_result, all_expected_keys
                    )
            
            # 2. 메모리 사용량 분석
            if hasattr(self, 'performance_metrics'):
                report['memory_usage'] = {
                    'peak_memory_mb': getattr(self.performance_metrics, 'peak_memory_usage_mb', 0),
                    'average_memory_mb': getattr(self.performance_metrics, 'average_memory_usage_mb', 0),
                    'optimizations_count': getattr(self.performance_metrics, 'memory_optimizations', 0)
                }
            
            # 3. DI Container 분석
            report['di_container_analysis'] = self._analyze_di_container_data_flow(step_result, step_id)
            
            # 4. 성능 메트릭
            if hasattr(self, 'performance_metrics'):
                report['performance_metrics'] = {
                    'data_conversions': getattr(self.performance_metrics, 'data_conversions', 0),
                    'step_data_transfers': getattr(self.performance_metrics, 'step_data_transfers', 0),
                    'validation_failures': getattr(self.performance_metrics, 'validation_failures', 0),
                    'api_conversions': getattr(self.performance_metrics, 'api_conversions', 0)
                }
            
            # 5. 경고 및 오류 수집
            if not report['data_completeness'].get('is_complete', True):
                report['warnings'].append("데이터 완전성 검증 실패")
            
            if report['di_container_analysis'].get('errors'):
                report['errors'].extend(report['di_container_analysis']['errors'])
            
            # 6. 리포트 로깅
            self.logger.info(f"📊 Step {step_id} 데이터 전달 리포트 생성 완료")
            if report['warnings']:
                self.logger.warning(f"⚠️ 경고: {len(report['warnings'])}개")
            if report['errors']:
                self.logger.error(f"❌ 오류: {len(report['errors'])}개")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ 데이터 전달 리포트 생성 실패: {e}")
            report['errors'].append(f"리포트 생성 오류: {e}")
            return report

    def _validate_data_completeness(self, step_result: Dict[str, Any], expected_keys: List[str]) -> Dict[str, Any]:
        """데이터 완전성 검증"""
        try:
            present_keys = []
            missing_keys = []
            
            for key in expected_keys:
                if key in step_result and step_result[key] is not None:
                    present_keys.append(key)
                else:
                    missing_keys.append(key)
            
            is_complete = len(missing_keys) == 0
            
            return {
                'is_complete': is_complete,
                'present_keys': present_keys,
                'missing_keys': missing_keys,
                'completeness_ratio': len(present_keys) / len(expected_keys) if expected_keys else 1.0
            }
            
        except Exception as e:
            self.logger.error(f"❌ 데이터 완전성 검증 실패: {e}")
            return {
                'is_complete': False,
                'present_keys': [],
                'missing_keys': expected_keys,
                'completeness_ratio': 0.0,
                'error': str(e)
            }

    def _log_comprehensive_process_report(self, process_report: Dict[str, Any], processing_time: float):
        """종합 프로세스 리포트 로깅"""
        try:
            step_name = process_report.get('step_name', 'Unknown')
            step_id = process_report.get('step_id', 0)
            
            # 🔥 단계별 성능 분석
            total_stage_time = 0
            stage_details = []
            
            for stage in process_report.get('processing_stages', []):
                stage_time = stage.get('duration', 0)
                total_stage_time += stage_time
                stage_details.append(f"{stage['stage']}: {stage_time:.3f}s")
                
                if not stage.get('success', True):
                    self.logger.warning(f"⚠️ {stage['stage']} 단계 실패: {stage.get('error', 'Unknown error')}")
            
            # 🔥 데이터 전달 분석
            data_report = process_report.get('data_transfer_report', {})
            data_completeness = data_report.get('data_completeness', {})
            memory_usage = data_report.get('memory_usage', {})
            
            # 🔥 DI Container 분석
            di_analysis = process_report.get('di_container_analysis', {})
            
            # 🔥 종합 로깅
            self.logger.info(f"📊 {step_name} (Step {step_id}) 종합 리포트")
            self.logger.info(f"   - 총 처리 시간: {processing_time:.3f}초")
            if stage_details:
                self.logger.info(f"   - 단계별 시간: {' | '.join(stage_details)}")
            
            # 데이터 완전성
            if data_completeness:
                completeness = data_completeness.get('is_complete', False)
                missing_count = len(data_completeness.get('missing_keys', []))
                present_count = len(data_completeness.get('present_keys', []))
                
                if completeness:
                    self.logger.info(f"   - 데이터 완전성: ✅ ({present_count}개 포함)")
                else:
                    self.logger.warning(f"   - 데이터 완전성: ❌ ({missing_count}개 누락)")
            
            # 메모리 사용량
            if memory_usage:
                peak_memory = memory_usage.get('peak_memory_mb', 0)
                optimizations = memory_usage.get('optimizations_count', 0)
                self.logger.info(f"   - 메모리 사용량: {peak_memory:.2f}MB (최적화: {optimizations}회)")
            
            # DI Container 상태
            if di_analysis:
                di_used = di_analysis.get('di_container_used', False)
                services_accessed = di_analysis.get('services_accessed', [])
                data_transfers = di_analysis.get('data_transfers', 0)
                
                if di_used:
                    self.logger.info(f"   - DI Container: ✅ ({len(services_accessed)}개 서비스, {data_transfers}개 전달)")
                else:
                    self.logger.warning(f"   - DI Container: ❌ 미사용")
            
            # 경고 및 오류 요약
            warnings_count = len(process_report.get('warnings', []))
            errors_count = len(process_report.get('errors', []))
            
            if warnings_count > 0:
                self.logger.warning(f"   - 경고: {warnings_count}개")
            if errors_count > 0:
                self.logger.error(f"   - 오류: {errors_count}개")
            
        except Exception as e:
            self.logger.error(f"❌ 종합 리포트 로깅 실패: {e}")
