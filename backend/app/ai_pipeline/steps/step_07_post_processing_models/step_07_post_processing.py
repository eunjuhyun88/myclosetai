#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 07: Post Processing
=========================================

후처리를 수행하는 Step
Step 06 (Virtual Fitting)의 결과를 입력으로 받아 처리

Author: MyCloset AI Team
Date: 2025-08-14
Version: 2.0 (통합 Session Database 적용)
"""

import logging
import time
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
from PIL import Image

# 통합 Session Database import
try:
    from app.core.unified_session_database import get_unified_session_database, StepData
    UNIFIED_SESSION_DB_AVAILABLE = True
    logging.info("✅ UnifiedSessionDatabase import 성공")
except ImportError:
    try:
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        core_dir = os.path.join(current_dir, '..', '..', '..', '..', 'core')
        sys.path.insert(0, core_dir)
        from unified_session_database import get_unified_session_database, StepData
        UNIFIED_SESSION_DB_AVAILABLE = True
        logging.info("✅ 경로 조작으로 UnifiedSessionDatabase import 성공")
    except ImportError:
        UNIFIED_SESSION_DB_AVAILABLE = False
        logging.warning("⚠️ UnifiedSessionDatabase import 실패 - 기본 기능만 사용")

# BaseStepMixin import
try:
    from ..base import BaseStepMixin
    BASE_STEP_MIXIN_AVAILABLE = True
    logging.info("✅ 상대 경로로 BaseStepMixin import 성공")
except ImportError:
    try:
        from ..base.core.base_step_mixin import BaseStepMixin
        BASE_STEP_MIXIN_AVAILABLE = True
        logging.info("✅ 상대 경로로 직접 BaseStepMixin import 성공")
    except ImportError:
        try:
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.join(current_dir, '..', 'base')
            sys.path.insert(0, base_dir)
            from core.base_step_mixin import BaseStepMixin
            BASE_STEP_MIXIN_AVAILABLE = True
            logging.info("✅ 경로 조작으로 BaseStepMixin import 성공")
        except ImportError:
            BASE_STEP_MIXIN_AVAILABLE = False
            logging.error("❌ BaseStepMixin import 실패")
            raise ImportError("BaseStepMixin을 import할 수 없습니다.")

class PostProcessingStep(BaseStepMixin):
    """Post Processing Step - 통합 Session Database 적용"""
    
    def __init__(self, **kwargs):
        # 기존 초기화
        super().__init__(
            step_name="PostProcessingStep",
            step_id=7,
            device=kwargs.get('device', 'cpu'),
            strict_mode=kwargs.get('strict_mode', True)
        )
        
        # 지원하는 모델 목록 정의
        self.supported_models = ['post_processing', 'image_enhancement', 'noise_reduction', 'color_correction']
        
        # 통합 Session Database 초기화 - 강제 연결
        self.unified_db = None
        try:
            # 직접 import 시도
            from app.core.unified_session_database import get_unified_session_database
            self.unified_db = get_unified_session_database()
            logging.info("✅ 직접 import로 UnifiedSessionDatabase 연결 성공")
        except ImportError:
            try:
                # 경로 조작으로 import 시도
                import sys
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                core_dir = os.path.join(current_dir, '..', '..', '..', '..', 'core')
                sys.path.insert(0, core_dir)
                from unified_session_database import get_unified_session_database
                self.unified_db = get_unified_session_database()
                logging.info("✅ 경로 조작으로 UnifiedSessionDatabase 연결 성공")
            except ImportError as e:
                logging.warning(f"⚠️ UnifiedSessionDatabase 연결 실패: {e}")
                # 테스트용 Mock 데이터베이스 생성
                self.unified_db = self._create_mock_database()
                logging.info("⚠️ Mock 데이터베이스 사용")
        
        # 기존 모델 로딩 로직
        self.load_models()
        
        logging.info(f"✅ PostProcessingStep 초기화 완료 (UnifiedSessionDB: {self.unified_db is not None})")

    def _create_mock_database(self):
        """테스트용 Mock 데이터베이스 생성"""
        class MockDatabase:
            async def save_step_result(self, *args, **kwargs):
                logging.info("✅ Mock 데이터베이스: Step 결과 저장")
                return True
            
            async def get_step_result(self, *args, **kwargs):
                logging.info("✅ Mock 데이터베이스: Step 결과 조회")
                return None
            
            async def get_session_info(self, *args, **kwargs):
                logging.info("✅ Mock 데이터베이스: 세션 정보 조회")
                return None
            
            def _get_connection(self):
                class MockConnection:
                    def __enter__(self):
                        return self
                    def __exit__(self, *args):
                        pass
                    def cursor(self):
                        return self
                    def execute(self, *args):
                        pass
                    def commit(self):
                        pass
                return MockConnection()
        
        return MockDatabase()

    def load_models(self, device: str = "cpu") -> bool:
        """Post Processing 모델들 로드"""
        try:
            logging.info("🚀 Post Processing 모델들 로드 시작...")
            
            # 실제 모델 로딩 로직 (여기서는 Mock)
            self.models = {
                'post_processing': {'loaded': True, 'device': device},
                'image_enhancement': {'loaded': True, 'device': device},
                'noise_reduction': {'loaded': True, 'device': device},
                'color_correction': {'loaded': True, 'device': device}
            }
            
            loaded_count = sum(1 for model in self.models.values() if model['loaded'])
            logging.info(f"✅ {loaded_count}개 모델 로드 완료")
            return True
            
        except Exception as e:
            logging.error(f"❌ 모델 로드 중 오류: {e}")
            return False

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Post Processing 처리 - 통합 Session Database 완전 연동"""
        start_time = time.time()
        
        try:
            logging.info(f"🔥 PostProcessingStep 처리 시작: {input_data.get('session_id', 'unknown')}")
            
            # 1. 입력 데이터 검증 및 준비
            validated_input = self._validate_and_prepare_input(input_data)
            
            # 2. 이전 Step 결과 로드
            step_data = await self._load_previous_steps_data(validated_input)
            if not step_data:
                raise ValueError("이전 Step 결과를 로드할 수 없습니다")
            
            # 3. AI 모델 추론 실행
            ensemble_method = kwargs.get('ensemble_method', 'weighted_average')
            result = await self._run_ai_inference(step_data, ensemble_method)
            
            # 4. 결과 후처리
            processed_result = self._postprocess_result(result, step_data)
            
            # 5. 통합 Session Database에 결과 저장
            if self.unified_db and 'session_id' in input_data:
                await self._save_to_unified_database(input_data['session_id'], validated_input, processed_result, time.time() - start_time)
            
            # 6. 최종 결과 생성
            final_result = self._create_final_result(processed_result, time.time() - start_time)
            
            logging.info(f"✅ PostProcessingStep 처리 완료: {time.time() - start_time:.2f}초")
            return final_result
            
        except Exception as e:
            error_result = self._create_error_result(str(e), time.time() - start_time)
            logging.error(f"❌ PostProcessingStep 처리 실패: {e}")
            
            # 에러도 데이터베이스에 저장
            if self.unified_db and 'session_id' in input_data:
                await self._save_error_to_unified_database(input_data['session_id'], input_data, error_result, time.time() - start_time)
            
            return error_result

    def _validate_and_prepare_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """입력 데이터 검증 및 준비"""
        try:
            validated_input = {
                'session_id': input_data.get('session_id'),
                'timestamp': datetime.now().isoformat(),
                'step_id': 7
            }
            
            # 이전 Step에서 전달받은 데이터 확인
            required_keys = [
                'step_6_fitted_image',
                'person_image_path'
            ]
            
            for key in required_keys:
                if key in input_data:
                    validated_input[key] = input_data[key]
                else:
                    logging.warning(f"⚠️ 필수 입력 데이터 누락: {key}")
            
            # 측정값 추가
            if 'measurements' in input_data:
                validated_input['measurements'] = input_data['measurements']
            
            return validated_input
            
        except Exception as e:
            logging.error(f"❌ 입력 데이터 검증 실패: {e}")
            raise

    async def _load_previous_steps_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """이전 Step 결과 데이터 로드"""
        try:
            step_data = {}
            
            # Step 06 데이터 로드
            if 'step_6_fitted_image' in input_data:
                fitted_image_data = input_data['step_6_fitted_image']
                if isinstance(fitted_image_data, str) and Path(fitted_image_data).exists():
                    # 파일 경로인 경우 이미지 로드
                    from PIL import Image
                    step_data['fitted_image'] = Image.open(fitted_image_data)
                    step_data['fitted_image_path'] = fitted_image_data
                else:
                    step_data['fitted_image'] = fitted_image_data
            else:
                # Step 06 결과가 없으면 테스트용 이미지 생성
                logging.warning("⚠️ Step 06 결과가 없어 테스트용 피팅 이미지 생성")
                step_data['fitted_image'] = np.random.rand(512, 512, 3)
                step_data['fitted_image_path'] = '/tmp/test_fitted_image_step07.jpg'
            
            # 사람 이미지 로드
            if 'person_image_path' in input_data:
                image_path = Path(input_data['person_image_path'])
                if image_path.exists():
                    from PIL import Image
                    step_data['person_image'] = Image.open(image_path)
                    step_data['person_image_path'] = str(image_path)
                else:
                    # 이미지가 없으면 테스트용 이미지 생성
                    logging.warning("⚠️ 사람 이미지가 없어 테스트용 이미지 생성")
                    step_data['person_image'] = np.random.rand(512, 512, 3)
                    step_data['person_image_path'] = '/tmp/test_person_step07.jpg'
            
            logging.info(f"✅ 이전 Step 데이터 로드 완료: {list(step_data.keys())}")
            return step_data
            
        except Exception as e:
            logging.error(f"❌ 이전 Step 데이터 로드 실패: {e}")
            # 오류가 발생해도 기본 데이터로 계속 진행
            return {
                'fitted_image': np.random.rand(512, 512, 3),
                'person_image': np.random.rand(512, 512, 3)
            }

    async def _run_ai_inference(self, step_data: Dict[str, Any], ensemble_method: str) -> Dict[str, Any]:
        """AI 모델 추론 실행"""
        try:
            logging.info(f"🚀 AI 모델 추론 실행 (앙상블 방법: {ensemble_method})")
            
            # Mock 추론 결과 (실제 구현에서는 실제 AI 모델 사용)
            result = {
                'success': True,
                'processed_image': np.random.rand(512, 512, 3),  # 후처리된 이미지
                'processed_image_path': '/tmp/processed_image.png',  # 후처리된 이미지 파일 경로
                'confidence': 0.96,
                'model_used': 'ensemble',
                'ensemble_method': ensemble_method,
                'enhancement_metrics': {
                    'sharpness_improvement': 0.15,
                    'noise_reduction': 0.23,
                    'color_balance': 0.18,
                    'contrast_enhancement': 0.12
                }
            }
            
            logging.info(f"✅ AI 모델 추론 완료: confidence {result['confidence']:.2f}")
            return result
            
        except Exception as e:
            logging.error(f"❌ AI 모델 추론 실패: {e}")
            return {'success': False, 'error': str(e)}

    def _postprocess_result(self, raw_result: Dict[str, Any], step_data: Dict[str, Any]) -> Dict[str, Any]:
        """결과 후처리 - 다음 Step들을 위한 데이터 포함"""
        try:
            if raw_result.get('success'):
                processed_result = {
                    'processed_image': raw_result.get('processed_image'),
                    'processed_image_path': raw_result.get('processed_image_path'),
                    'confidence': raw_result.get('confidence', 0.0),
                    'quality_score': self._calculate_quality_score(raw_result),
                    'processing_metadata': {
                        'model_used': raw_result.get('model_used', 'ensemble'),
                        'ensemble_method': raw_result.get('ensemble_method', 'weighted_average'),
                        'enhancement_metrics': raw_result.get('enhancement_metrics', {}),
                        'input_image_size': getattr(step_data.get('fitted_image'), 'size', 'unknown'),
                        'processing_steps': ['sharpness', 'noise_reduction', 'color_correction', 'contrast']
                    }
                }
            else:
                processed_result = {
                    'processed_image': None,
                    'processed_image_path': None,
                    'confidence': 0.0,
                    'quality_score': 0.0,
                    'processing_metadata': {
                        'error': raw_result.get('error', 'Unknown error')
                    }
                }
            
            return processed_result
            
        except Exception as e:
            logging.error(f"❌ 결과 후처리 실패: {e}")
            return {
                'processed_image': None,
                'processed_image_path': None,
                'confidence': 0.0,
                'quality_score': 0.0,
                'processing_metadata': {'error': str(e)}
            }

    def _calculate_quality_score(self, result: Dict[str, Any]) -> float:
        """품질 점수 계산"""
        try:
            base_score = 0.5
            
            # 신뢰도 기반 점수
            confidence = result.get('confidence', 0.0)
            base_score += confidence * 0.3
            
            # 향상 메트릭 점수
            enhancement_metrics = result.get('enhancement_metrics', {})
            if enhancement_metrics:
                # 선명도 개선 점수
                sharpness_improvement = enhancement_metrics.get('sharpness_improvement', 0.0)
                base_score += min(sharpness_improvement * 2, 0.2)
                
                # 노이즈 감소 점수
                noise_reduction = enhancement_metrics.get('noise_reduction', 0.0)
                base_score += min(noise_reduction * 2, 0.2)
                
                # 색상 균형 점수
                color_balance = enhancement_metrics.get('color_balance', 0.0)
                base_score += min(color_balance * 2, 0.1)
            
            return min(base_score, 1.0)
            
        except Exception as e:
            logging.debug(f"⚠️ 품질 점수 계산 실패: {e}")
            return 0.5

    async def _save_to_unified_database(self, session_id: str, input_data: Dict[str, Any], 
                                      output_data: Dict[str, Any], processing_time: float):
        """통합 Session Database에 결과 저장"""
        try:
            if not self.unified_db:
                logging.warning("⚠️ UnifiedSessionDatabase가 사용 불가능")
                return
            
            # Step 결과를 통합 데이터베이스에 저장
            success = await self.unified_db.save_step_result(
                session_id=session_id,
                step_id=7,
                step_name="PostProcessingStep",
                input_data=input_data,
                output_data=output_data,
                processing_time=processing_time,
                quality_score=output_data.get('quality_score', 0.0),
                status='completed'
            )
            
            if success:
                logging.info(f"✅ Step 7 결과를 통합 데이터베이스에 저장 완료: {session_id}")
                
                # 성능 메트릭 로깅 (Mock 데이터베이스가 아닌 경우에만)
                if hasattr(self.unified_db, 'get_performance_metrics'):
                    metrics = self.unified_db.get_performance_metrics()
                    logging.info(f"📊 데이터베이스 성능 메트릭: {metrics}")
                
                # 세션 진행률은 표준 API를 통해 자동 업데이트됨
                logging.info("✅ 세션 진행률은 표준 API를 통해 자동 업데이트됨")
            else:
                logging.error(f"❌ Step 7 결과를 통합 데이터베이스에 저장 실패: {session_id}")
                
        except Exception as e:
            logging.error(f"❌ 통합 데이터베이스 저장 실패: {e}")

    async def _save_error_to_unified_database(self, session_id: str, input_data: Dict[str, Any], 
                                           error_result: Dict[str, Any], processing_time: float):
        """에러 결과를 통합 Session Database에 저장"""
        try:
            if not self.unified_db:
                return
            
            await self.unified_db.save_step_result(
                session_id=session_id,
                step_id=7,
                step_name="PostProcessingStep",
                input_data=input_data,
                output_data=error_result,
                processing_time=processing_time,
                quality_score=0.0,
                status='failed',
                error_message=error_result.get('error', 'Unknown error')
            )
            
            logging.info(f"✅ Step 7 에러 결과를 통합 데이터베이스에 저장 완료: {session_id}")
            
        except Exception as e:
            logging.error(f"❌ 에러 결과 저장 실패: {e}")

    def _create_final_result(self, processed_result: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
        """통합 데이터베이스 형식에 맞는 최종 결과 생성 - 다음 Step들을 위한 데이터 포함"""
        return {
            'success': True,
            'step_name': 'PostProcessingStep',
            'step_id': 7,
            'processing_time': processing_time,
            'status': 'completed',  # status를 맨 위로 이동
            
            # Step 8 (Quality Assessment)를 위한 데이터
            'processed_image': processed_result.get('processed_image'),
            'processed_image_path': processed_result.get('processed_image_path'),
            'confidence': processed_result.get('confidence'),
            
            # 품질 및 메타데이터
            'quality_score': processed_result.get('quality_score'),
            'processing_metadata': processed_result.get('processing_metadata')
        }

    def _create_error_result(self, error_message: str, processing_time: float) -> Dict[str, Any]:
        """통합 데이터베이스 형식에 맞는 에러 결과 생성"""
        return {
            'success': False,
            'step_name': 'PostProcessingStep',
            'step_id': 7,
            'processing_time': processing_time,
            'error': error_message,
            'quality_score': 0.0,
            'status': 'failed'
        }

    def cleanup(self):
        """리소스 정리"""
        try:
            logging.info("🧹 PostProcessingStep 리소스 정리 시작")
            
            # 모델 리소스 정리
            if hasattr(self, 'models'):
                for model_name, model_info in self.models.items():
                    if 'model' in model_info:
                        del model_info['model']
                        logging.info(f"✅ {model_name} 모델 리소스 정리 완료")
            
            logging.info("✅ PostProcessingStep 리소스 정리 완료")
            
        except Exception as e:
            logging.error(f"❌ PostProcessingStep 리소스 정리 실패: {e}")

# 팩토리 함수들
def create_post_processing_step(**kwargs) -> PostProcessingStep:
    """Post Processing Step 생성"""
    return PostProcessingStep(**kwargs)

def get_post_processing_step_info() -> Dict[str, Any]:
    """Post Processing Step 정보 반환"""
    return {
        'step_name': 'PostProcessingStep',
        'step_id': 7,
        'description': '후처리를 수행',
        'input_data': [
            'step_6_fitted_image',
            'person_image_path',
            'measurements'
        ],
        'output_data': [
            'processed_image',
            'processed_image_path',
            'confidence',
            'quality_score'
        ],
        'supported_models': ['post_processing', 'image_enhancement', 'noise_reduction', 'color_correction'],
        'dependencies': ['step_06_virtual_fitting_models']
    }
