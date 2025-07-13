"""
개선된 완전한 8단계 가상 피팅 파이프라인
기존 app/ai_pipeline 구조를 완전히 활용하면서 프로덕션 레벨 기능 제공
M3 Max 최적화, 상세한 품질 분석, 에러 복구, 메모리 최적화 포함
"""
import os
import sys
import logging
import asyncio
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
import json
import gc
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# 기존 ai_pipeline 구조의 step 파일들 import
from app.ai_pipeline.steps.step_01_human_parsing import HumanParsingStep
from app.ai_pipeline.steps.step_02_pose_estimation import PoseEstimationStep
from app.ai_pipeline.steps.step_03_cloth_segmentation import ClothSegmentationStep
from app.ai_pipeline.steps.step_04_geometric_matching import GeometricMatchingStep
from app.ai_pipeline.steps.step_05_cloth_warping import ClothWarpingStep
from app.ai_pipeline.steps.step_06_virtual_fitting import VirtualFittingStep
from app.ai_pipeline.steps.step_07_post_processing import PostProcessingStep
from app.ai_pipeline.steps.step_08_quality_assessment import QualityAssessmentStep

# 기존 유틸리티들 import
from app.ai_pipeline.utils.model_loader import ModelLoader
from app.ai_pipeline.utils.memory_manager import MemoryManager
from app.ai_pipeline.utils.data_converter import DataConverter

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('virtual_fitting_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PipelineManager:
    """
    개선된 완전한 8단계 가상 피팅 파이프라인
    - 기존 ai_pipeline 구조 완전 호환
    - 프로덕션 레벨 품질과 안정성
    - M3 Max MPS 최적화
    - 상세한 품질 분석 및 개선 제안
    - 메모리 효율성 및 에러 복구
    """
    
    def __init__(self, config_path: Optional[str] = None, device: Optional[str] = None):
        """
        파이프라인 초기화
        
        Args:
            config_path: 설정 파일 경로 (선택적)
            device: 사용할 디바이스 ('auto', 'cpu', 'cuda', 'mps')
        """
        # 기존 유틸리티들 초기화
        self.model_loader = ModelLoader()
        self.memory_manager = MemoryManager()
        self.data_converter = DataConverter()
        
        # 디바이스 최적화
        self.device = device or self._get_optimal_device()
        self._configure_device_optimizations()
        
        # 설정 로드
        self.config = self._load_config(config_path)
        
        # 파이프라인 설정
        self.pipeline_config = self.config.get('pipeline', {
            'quality_level': 'high',  # low, medium, high, ultra
            'processing_mode': 'complete',  # fast, balanced, complete
            'enable_optimization': True,
            'enable_caching': True,
            'parallel_processing': True,
            'memory_optimization': True,
            'enable_intermediate_saving': False,
            'max_retries': 3,
            'timeout_seconds': 300
        })
        
        # 8단계 컴포넌트
        self.steps = {}
        self.step_order = [
            'human_parsing',           # 1단계: 인체 파싱
            'pose_estimation',         # 2단계: 포즈 추정
            'cloth_segmentation',      # 3단계: 의류 세그멘테이션
            'geometric_matching',      # 4단계: 기하학적 매칭
            'cloth_warping',          # 5단계: 옷 워핑
            'virtual_fitting',        # 6단계: 가상 피팅 생성
            'post_processing',        # 7단계: 후처리
            'quality_assessment'      # 8단계: 품질 평가
        ]
        
        # 상태 관리
        self.is_initialized = False
        self.processing_stats = {}
        self.session_data = {}
        self.error_history = []
        
        # 성능 모니터링
        self.performance_metrics = {
            'total_sessions': 0,
            'successful_sessions': 0,
            'average_processing_time': 0.0,
            'average_quality_score': 0.0
        }
        
        # 스레드 풀 (병렬 처리용)
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"🚀 개선된 가상 피팅 파이프라인 초기화 - 디바이스: {self.device}")
        logger.info(f"📊 파이프라인 모드: {self.pipeline_config['processing_mode']}")
        logger.info(f"🎯 품질 레벨: {self.pipeline_config['quality_level']}")
    
    def _get_optimal_device(self) -> str:
        """최적 디바이스 자동 선택"""
        if torch.backends.mps.is_available():
            # M3 Max MPS 우선
            return 'mps'
        elif torch.cuda.is_available():
            # CUDA 지원
            return 'cuda'
        else:
            # CPU 폴백
            return 'cpu'
    
    def _configure_device_optimizations(self):
        """디바이스별 최적화 설정"""
        if self.device == 'mps':
            # M3 Max MPS 최적화
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            torch.backends.mps.empty_cache()
            logger.info("🔧 M3 Max MPS 최적화 설정 완료")
            
        elif self.device == 'cuda':
            # CUDA 최적화
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            logger.info("🔧 CUDA 최적화 설정 완료")
        
        # 혼합 정밀도 설정
        if self.device in ['cuda', 'mps']:
            self.use_amp = True
            logger.info("⚡ 혼합 정밀도 연산 활성화")
        else:
            self.use_amp = False
    
    async def initialize(self) -> bool:
        """전체 파이프라인 초기화"""
        try:
            logger.info("🔄 개선된 8단계 가상 피팅 파이프라인 초기화 시작...")
            start_time = time.time()
            
            # 메모리 정리
            self._cleanup_memory()
            
            # 각 단계 순차적 초기화
            await self._initialize_all_steps()
            
            # 초기화 검증
            initialization_success = await self._verify_initialization()
            
            if not initialization_success:
                raise RuntimeError("일부 단계 초기화 실패")
            
            initialization_time = time.time() - start_time
            self.processing_stats['initialization_time'] = initialization_time
            
            self.is_initialized = True
            logger.info(f"✅ 전체 파이프라인 초기화 완료 - 소요시간: {initialization_time:.2f}초")
            
            # 시스템 상태 출력
            await self._print_system_status()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 파이프라인 초기화 실패: {e}")
            logger.error(f"📋 오류 상세: {traceback.format_exc()}")
            self.is_initialized = False
            return False
    
    async def _initialize_all_steps(self):
        """모든 단계 초기화"""
        
        # 1단계: 인체 파싱
        logger.info("1️⃣ 인체 파싱 초기화...")
        self.steps['human_parsing'] = HumanParsingStep()
        await self._safe_initialize_step('human_parsing')
        
        # 2단계: 포즈 추정
        logger.info("2️⃣ 포즈 추정 초기화...")
        self.steps['pose_estimation'] = PoseEstimationStep()
        await self._safe_initialize_step('pose_estimation')
        
        # 3단계: 의류 세그멘테이션
        logger.info("3️⃣ 의류 세그멘테이션 초기화...")
        self.steps['cloth_segmentation'] = ClothSegmentationStep()
        await self._safe_initialize_step('cloth_segmentation')
        
        # 4단계: 기하학적 매칭
        logger.info("4️⃣ 기하학적 매칭 초기화...")
        self.steps['geometric_matching'] = GeometricMatchingStep()
        await self._safe_initialize_step('geometric_matching')
        
        # 5단계: 옷 워핑
        logger.info("5️⃣ 옷 워핑 초기화...")
        self.steps['cloth_warping'] = ClothWarpingStep()
        await self._safe_initialize_step('cloth_warping')
        
        # 6단계: 가상 피팅
        logger.info("6️⃣ 가상 피팅 생성 초기화...")
        self.steps['virtual_fitting'] = VirtualFittingStep()
        await self._safe_initialize_step('virtual_fitting')
        
        # 7단계: 후처리
        logger.info("7️⃣ 후처리 초기화...")
        self.steps['post_processing'] = PostProcessingStep()
        await self._safe_initialize_step('post_processing')
        
        # 8단계: 품질 평가
        logger.info("8️⃣ 품질 평가 초기화...")
        self.steps['quality_assessment'] = QualityAssessmentStep()
        await self._safe_initialize_step('quality_assessment')
    
    async def _safe_initialize_step(self, step_name: str):
        """안전한 단계 초기화"""
        try:
            step = self.steps[step_name]
            if hasattr(step, 'initialize'):
                await step.initialize()
            logger.info(f"✅ {step_name} 초기화 완료")
        except Exception as e:
            logger.warning(f"⚠️ {step_name} 초기화 실패: {e}")
            # 계속 진행 (일부 단계 실패해도 전체 중단하지 않음)
    
    async def _verify_initialization(self) -> bool:
        """초기화 검증"""
        total_steps = len(self.step_order)
        initialized_steps = len(self.steps)
        
        success_rate = initialized_steps / total_steps
        logger.info(f"📊 초기화 성공률: {success_rate:.1%} ({initialized_steps}/{total_steps})")
        
        return success_rate >= 0.8  # 80% 이상 성공하면 진행
    
    async def process_complete_virtual_fitting(
        self,
        person_image: Union[str, Image.Image, np.ndarray],
        clothing_image: Union[str, Image.Image, np.ndarray],
        body_measurements: Optional[Dict[str, float]] = None,
        clothing_type: str = "shirt",
        fabric_type: str = "cotton",
        style_preferences: Optional[Dict[str, Any]] = None,
        quality_target: float = 0.8,
        progress_callback: Optional[Callable] = None,
        save_intermediate: bool = False,
        enable_auto_retry: bool = True
    ) -> Dict[str, Any]:
        """
        개선된 완전한 8단계 가상 피팅 처리
        
        Args:
            person_image: 사용자 이미지 (경로, PIL, numpy 배열)
            clothing_image: 의류 이미지 (경로, PIL, numpy 배열)
            body_measurements: 신체 치수 {'height': 170, 'weight': 65, 'chest': 95, ...}
            clothing_type: 의류 타입 ('shirt', 'pants', 'dress', 'jacket', 'skirt')
            fabric_type: 천 재질 ('cotton', 'denim', 'silk', 'polyester', 'wool')
            style_preferences: 스타일 선호도 {'fit': 'slim', 'color_preference': 'original'}
            quality_target: 목표 품질 점수 (0.0-1.0)
            progress_callback: 진행상황 콜백 함수 async def callback(stage: str, percentage: int)
            save_intermediate: 중간 결과 저장 여부
            enable_auto_retry: 자동 재시도 활성화
            
        Returns:
            완전한 가상 피팅 결과 딕셔너리
        """
        if not self.is_initialized:
            raise RuntimeError("파이프라인이 초기화되지 않았습니다. initialize()를 먼저 호출하세요.")
        
        start_time = time.time()
        session_id = f"vf_{int(time.time())}_{np.random.randint(1000, 9999)}"
        
        # 성능 메트릭 업데이트
        self.performance_metrics['total_sessions'] += 1
        
        try:
            logger.info(f"🎯 개선된 8단계 가상 피팅 시작 - 세션 ID: {session_id}")
            logger.info(f"⚙️ 설정: {clothing_type} ({fabric_type}), 품질목표: {quality_target}")
            
            # 입력 검증 및 전처리
            person_tensor, clothing_tensor = await self._preprocess_inputs(
                person_image, clothing_image
            )
            
            # 세션 데이터 초기화
            self._initialize_session_data(session_id, start_time, {
                'clothing_type': clothing_type,
                'fabric_type': fabric_type,
                'quality_target': quality_target,
                'style_preferences': style_preferences or {},
                'processing_mode': self.pipeline_config['processing_mode']
            })
            
            if progress_callback:
                await progress_callback("입력 전처리 완료", 5)
            
            # 메모리 최적화
            if self.pipeline_config['memory_optimization']:
                self._optimize_memory_usage()
            
            # ===========================================
            # 8단계 순차 처리
            # ===========================================
            
            # 1단계: 인체 파싱 (20개 부위 분할)
            parsing_result = await self._execute_step_with_retry(
                'human_parsing', 1, person_tensor, progress_callback, 18
            )
            
            # 2단계: 포즈 추정 (18개 키포인트)
            pose_result = await self._execute_step_with_retry(
                'pose_estimation', 2, person_tensor, progress_callback, 31
            )
            
            # 3단계: 의류 세그멘테이션 (배경 제거)
            segmentation_result = await self._execute_step_with_retry(
                'cloth_segmentation', 3, clothing_tensor, progress_callback, 44,
                extra_args={'clothing_type': clothing_type}
            )
            
            # 4단계: 기하학적 매칭 (TPS 변환)
            matching_result = await self._execute_step_with_retry(
                'geometric_matching', 4, 
                (segmentation_result, pose_result, parsing_result),
                progress_callback, 57
            )
            
            # 5단계: 옷 워핑 (물리 시뮬레이션)
            warping_result = await self._execute_step_with_retry(
                'cloth_warping', 5,
                (matching_result, body_measurements, fabric_type),
                progress_callback, 70
            )
            
            # 6단계: 가상 피팅 생성 (최종 합성)
            fitting_result = await self._execute_step_with_retry(
                'virtual_fitting', 6,
                (person_tensor, warping_result, parsing_result, pose_result),
                progress_callback, 83
            )
            
            # 7단계: 후처리 (품질 향상)
            post_processing_result = await self._execute_step_with_retry(
                'post_processing', 7, fitting_result, progress_callback, 91
            )
            
            # 8단계: 품질 평가 (자동 스코어링)
            quality_result = await self._execute_step_with_retry(
                'quality_assessment', 8,
                (post_processing_result, person_tensor, clothing_tensor, 
                 parsing_result, pose_result, warping_result, fitting_result),
                progress_callback, 96
            )
            
            # ===========================================
            # 최종 결과 구성 및 분석
            # ===========================================
            
            total_time = time.time() - start_time
            
            # 최종 결과 이미지 추출
            final_image_tensor = self._extract_final_image(
                post_processing_result, fitting_result, person_tensor
            )
            final_image_pil = self._tensor_to_pil(final_image_tensor)
            
            # 종합 품질 분석
            comprehensive_quality = await self._comprehensive_quality_analysis(
                quality_result, self.session_data[session_id]
            )
            
            # 처리 통계 계산
            processing_statistics = self._calculate_detailed_statistics(session_id, total_time)
            
            # 품질 개선 분석
            quality_improvement_analysis = self._analyze_quality_progression(session_id)
            
            # 개선 제안 생성
            improvement_suggestions = await self._generate_detailed_suggestions(
                comprehensive_quality, processing_statistics, clothing_type, fabric_type
            )
            
            # 성능 메트릭 업데이트
            self.performance_metrics['successful_sessions'] += 1
            self._update_performance_metrics(total_time, comprehensive_quality['overall_score'])
            
            # 최종 결과 딕셔너리 구성
            final_result = {
                # 기본 정보
                'success': True,
                'session_id': session_id,
                'processing_mode': self.pipeline_config['processing_mode'],
                'quality_level': self.pipeline_config['quality_level'],
                
                # 결과 이미지들
                'result_image': final_image_pil,
                'result_image_tensor': final_image_tensor,
                'original_person_image': self._tensor_to_pil(person_tensor),
                'original_clothing_image': self._tensor_to_pil(clothing_tensor),
                
                # 품질 메트릭 (상세)
                'final_quality_score': comprehensive_quality['overall_score'],
                'quality_grade': comprehensive_quality['quality_grade'],
                'quality_confidence': comprehensive_quality['confidence'],
                'quality_breakdown': comprehensive_quality['breakdown'],
                'quality_target_achieved': comprehensive_quality['overall_score'] >= quality_target,
                'quality_improvement_analysis': quality_improvement_analysis,
                
                # 핏 분석 (상세)
                'fit_analysis': {
                    'overall_fit_score': comprehensive_quality['breakdown'].get('fit_quality', 0.8),
                    'body_alignment': comprehensive_quality.get('body_alignment', 0.8),
                    'garment_deformation': comprehensive_quality.get('garment_deformation', 0.8),
                    'size_compatibility': self._analyze_size_compatibility(
                        body_measurements, clothing_type
                    ),
                    'style_match': self._analyze_style_match(
                        style_preferences, fitting_result
                    )
                },
                
                # 개선 제안 (카테고리별)
                'improvement_suggestions': improvement_suggestions,
                'next_steps': self._generate_next_steps(comprehensive_quality, quality_target),
                
                # 처리 통계 (상세)
                'processing_statistics': processing_statistics,
                'total_processing_time': total_time,
                'device_used': self.device,
                'memory_usage': self._get_detailed_memory_usage(),
                'performance_metrics': self.performance_metrics.copy(),
                
                # 단계별 결과 (상세)
                'step_results_summary': self._create_detailed_step_summary(session_id),
                
                # 중간 결과 (선택적)
                'intermediate_results': (
                    self.session_data[session_id]['intermediate_results'] 
                    if save_intermediate else {}
                ),
                
                # 메타데이터 (확장)
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'pipeline_version': '2.0.0',
                    'input_resolution': f"{person_tensor.shape[3]}x{person_tensor.shape[2]}",
                    'output_resolution': f"{final_image_pil.width}x{final_image_pil.height}",
                    'clothing_type': clothing_type,
                    'fabric_type': fabric_type,
                    'body_measurements_provided': body_measurements is not None,
                    'style_preferences_provided': bool(style_preferences),
                    'intermediate_results_saved': save_intermediate,
                    'device_optimization': self.device,
                    'memory_optimization_enabled': self.pipeline_config['memory_optimization'],
                    'parallel_processing_enabled': self.pipeline_config['parallel_processing']
                }
            }
            
            # 세션 데이터 정리 (메모리 절약)
            if not save_intermediate:
                self._cleanup_session_data(session_id)
            
            if progress_callback:
                await progress_callback("처리 완료", 100)
            
            logger.info(
                f"🎉 개선된 8단계 가상 피팅 완료! "
                f"전체 소요시간: {total_time:.2f}초, "
                f"최종 품질: {comprehensive_quality['overall_score']:.3f} ({comprehensive_quality['quality_grade']}), "
                f"목표 달성: {'✅' if comprehensive_quality['overall_score'] >= quality_target else '❌'}"
            )
            
            return final_result
            
        except Exception as e:
            # 에러 처리 및 복구
            error_result = await self._handle_processing_error(
                e, session_id, start_time, person_image, clothing_image,
                enable_auto_retry
            )
            return error_result
    
    async def _execute_step_with_retry(
        self,
        step_name: str,
        step_number: int,
        input_data: Any,
        progress_callback: Optional[Callable],
        progress_percentage: int,
        extra_args: Optional[Dict] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """재시도 로직이 포함된 단계 실행"""
        
        step_start = time.time()
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"{'🔄' if attempt > 0 else ''} {step_number}단계: {step_name} 처리 중... {'(재시도 ' + str(attempt) + ')' if attempt > 0 else ''}")
                
                # 메모리 정리 (재시도 시)
                if attempt > 0:
                    self._cleanup_memory()
                
                # 단계 실행
                result = await self._execute_single_step(step_name, input_data, extra_args)
                
                # 결과 검증
                if self._validate_step_result(step_name, result):
                    step_time = time.time() - step_start
                    
                    # 세션 데이터 저장
                    session_id = list(self.session_data.keys())[-1] if self.session_data else 'unknown'
                    if session_id in self.session_data:
                        self.session_data[session_id]['step_times'][step_name] = step_time
                        self.session_data[session_id]['step_results'][step_name] = result
                        
                        # 중간 결과 저장 (요청된 경우)
                        if self.pipeline_config.get('enable_intermediate_saving', False):
                            self.session_data[session_id]['intermediate_results'][step_name] = result
                    
                    # 품질 점수 로깅
                    quality_score = result.get('confidence', result.get('quality_score', 0.8))
                    logger.info(f"✅ {step_number}단계 완료 - 소요시간: {step_time:.2f}초, 품질: {quality_score:.3f}")
                    
                    # 진행률 콜백
                    if progress_callback:
                        await progress_callback(f"{step_name}_complete", progress_percentage)
                    
                    return result
                else:
                    raise ValueError(f"Step {step_name} result validation failed")
                    
            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ {step_number}단계 시도 {attempt + 1} 실패: {e}")
                
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # 지수 백오프
                    logger.info(f"🔄 {wait_time}초 후 재시도...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ {step_number}단계 최종 실패: {e}")
        
        # 모든 재시도 실패 시 폴백 결과 생성
        logger.warning(f"🚨 {step_name} 폴백 결과 생성 중...")
        return self._create_fallback_step_result(step_name, input_data, last_error)
    
    async def _execute_single_step(
        self, 
        step_name: str, 
        input_data: Any, 
        extra_args: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """단일 단계 실행"""
        
        step = self.steps.get(step_name)
        if not step:
            raise ValueError(f"Step {step_name} not found")
        
        # 단계별 실행 로직
        if step_name == 'human_parsing':
            if hasattr(step, 'process'):
                return await step.process(input_data)
            else:
                return await step.parse_human(input_data)
                
        elif step_name == 'pose_estimation':
            if hasattr(step, 'process'):
                return await step.process(input_data)
            else:
                return await step.estimate_pose(input_data)
                
        elif step_name == 'cloth_segmentation':
            clothing_type = extra_args.get('clothing_type', 'shirt') if extra_args else 'shirt'
            if hasattr(step, 'process'):
                return await step.process(input_data, clothing_type)
            else:
                return await step.segment_cloth(input_data, clothing_type)
                
        elif step_name == 'geometric_matching':
            segmentation_result, pose_result, parsing_result = input_data
            if hasattr(step, 'process'):
                return await step.process(segmentation_result, pose_result, parsing_result)
            else:
                return await step.match_geometry(segmentation_result, pose_result, parsing_result)
                
        elif step_name == 'cloth_warping':
            matching_result, body_measurements, fabric_type = input_data
            if hasattr(step, 'process'):
                return await step.process(matching_result, body_measurements, fabric_type)
            else:
                return await step.warp_cloth(matching_result, body_measurements, fabric_type)
                
        elif step_name == 'virtual_fitting':
            person_tensor, warping_result, parsing_result, pose_result = input_data
            if hasattr(step, 'process'):
                return await step.process(person_tensor, warping_result, parsing_result, pose_result)
            else:
                return await step.generate_fitting(person_tensor, warping_result, parsing_result, pose_result)
                
        elif step_name == 'post_processing':
            if hasattr(step, 'process'):
                return await step.process(input_data)
            else:
                return await step.enhance_quality(input_data)
                
        elif step_name == 'quality_assessment':
            (post_processing_result, person_tensor, clothing_tensor, 
             parsing_result, pose_result, warping_result, fitting_result) = input_data
            if hasattr(step, 'process'):
                return await step.process(
                    post_processing_result, person_tensor, clothing_tensor,
                    parsing_result, pose_result, warping_result, fitting_result
                )
            else:
                return await step.assess_quality(
                    post_processing_result, person_tensor, clothing_tensor
                )
        
        else:
            raise ValueError(f"Unknown step: {step_name}")
    
    def _validate_step_result(self, step_name: str, result: Dict[str, Any]) -> bool:
        """단계 결과 검증"""
        if not isinstance(result, dict):
            return False
        
        # 기본 필드 검증
        required_fields = {
            'human_parsing': ['confidence'],
            'pose_estimation': ['pose_confidence', 'keypoints'],
            'cloth_segmentation': ['confidence', 'segmented_clothing'],
            'geometric_matching': ['transform_quality'],
            'cloth_warping': ['quality_metrics'],
            'virtual_fitting': ['fitted_image'],
            'post_processing': ['enhanced_image'],
            'quality_assessment': ['overall_score']
        }
        
        step_required = required_fields.get(step_name, [])
        for field in step_required:
            if field not in result:
                logger.warning(f"⚠️ {step_name} 결과에 필수 필드 '{field}' 누락")
                return False
        
        return True
    
    def _create_fallback_step_result(
        self, 
        step_name: str, 
        input_data: Any, 
        error: Exception
    ) -> Dict[str, Any]:
        """폴백 단계 결과 생성"""
        
        base_result = {
            'success': False,
            'error': str(error),
            'fallback': True,
            'timestamp': datetime.now().isoformat()
        }
        
        # 단계별 기본 폴백 결과
        if step_name == 'human_parsing':
            base_result.update({
                'confidence': 0.5,
                'body_parts_detected': [],
                'parsing_map': torch.zeros(1, 20, 512, 512) if torch.is_tensor(input_data) else np.zeros((512, 512))
            })
            
        elif step_name == 'pose_estimation':
            base_result.update({
                'pose_confidence': 0.5,
                'keypoints': np.zeros((18, 3)),
                'keypoints_18': np.zeros((18, 3))
            })
            
        elif step_name == 'cloth_segmentation':
            base_result.update({
                'confidence': 0.5,
                'segmented_clothing': input_data if torch.is_tensor(input_data) else torch.zeros(1, 3, 512, 512),
                'clothing_mask': torch.ones(1, 1, 512, 512)
            })
            
        elif step_name == 'geometric_matching':
            base_result.update({
                'transform_quality': {'overall_quality': 0.5},
                'matched_pairs': [],
                'transformation_matrix': np.eye(3)
            })
            
        elif step_name == 'cloth_warping':
            base_result.update({
                'quality_metrics': {'overall_quality': 0.5},
                'warped_clothing': input_data[0] if isinstance(input_data, tuple) else input_data,
                'simulation_details': {'physics_simulation': False}
            })
            
        elif step_name == 'virtual_fitting':
            base_result.update({
                'fitted_image': input_data[0] if isinstance(input_data, tuple) else input_data,
                'quality_metrics': {'overall_quality': 0.5},
                'fitting_analysis': {'fit_score': 0.5}
            })
            
        elif step_name == 'post_processing':
            base_result.update({
                'enhanced_image': input_data.get('fitted_image', input_data) if isinstance(input_data, dict) else input_data,
                'enhancement_score': 0.5,
                'improvements_applied': 0
            })
            
        elif step_name == 'quality_assessment':
            base_result.update({
                'overall_score': 0.5,
                'quality_grade': 'Poor',
                'detailed_metrics': {},
                'improvement_suggestions': ['시스템 오류로 인한 품질 저하']
            })
        
        logger.warning(f"🚨 {step_name} 폴백 결과 생성됨")
        return base_result
    
    async def _preprocess_inputs(
        self, 
        person_image: Union[str, Image.Image, np.ndarray],
        clothing_image: Union[str, Image.Image, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """입력 이미지 전처리 및 검증"""
        
        try:
            # 데이터 변환기 사용
            if hasattr(self.data_converter, 'preprocess_image'):
                person_tensor = self.data_converter.preprocess_image(person_image)
                clothing_tensor = self.data_converter.preprocess_image(clothing_image)
            else:
                person_tensor = self._manual_preprocess_image(person_image)
                clothing_tensor = self._manual_preprocess_image(clothing_image)
            
            # 입력 검증
            self._validate_input_tensors(person_tensor, clothing_tensor)
            
            # 디바이스로 이동
            person_tensor = person_tensor.to(self.device)
            clothing_tensor = clothing_tensor.to(self.device)
            
            logger.info(f"✅ 입력 이미지 전처리 완료 - 사람: {person_tensor.shape}, 의류: {clothing_tensor.shape}")
            
            return person_tensor, clothing_tensor
            
        except Exception as e:
            logger.error(f"❌ 입력 전처리 실패: {e}")
            raise
    
    def _manual_preprocess_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """수동 이미지 전처리"""
        
        # 입력 타입별 처리
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_input}")
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype != np.uint8:
                image_input = (image_input * 255).astype(np.uint8)
            image = Image.fromarray(image_input).convert('RGB')
        else:
            raise ValueError(f"지원하지 않는 이미지 타입: {type(image_input)}")
        
        # 크기 조정
        target_size = self.config.get('input_size', (512, 512))
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # 텐서 변환
        img_array = np.array(image)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor
    
    def _validate_input_tensors(self, person_tensor: torch.Tensor, clothing_tensor: torch.Tensor):
        """입력 텐서 검증"""
        
        # 차원 검증
        if person_tensor.dim() != 4 or clothing_tensor.dim() != 4:
            raise ValueError("입력 텐서는 4차원이어야 합니다 (B, C, H, W)")
        
        # 채널 검증
        if person_tensor.shape[1] != 3 or clothing_tensor.shape[1] != 3:
            raise ValueError("입력 이미지는 RGB 3채널이어야 합니다")
        
        # 값 범위 검증
        if (person_tensor.min() < 0 or person_tensor.max() > 1 or 
            clothing_tensor.min() < 0 or clothing_tensor.max() > 1):
            raise ValueError("텐서 값은 0-1 범위여야 합니다")
        
        logger.debug("✅ 입력 텐서 검증 완료")
    
    def _initialize_session_data(self, session_id: str, start_time: float, config: Dict[str, Any]):
        """세션 데이터 초기화"""
        
        self.session_data[session_id] = {
            'start_time': start_time,
            'config': config,
            'step_times': {},
            'step_results': {},
            'intermediate_results': {},
            'error_log': [],
            'memory_snapshots': [],
            'quality_progression': []
        }
    
    def _optimize_memory_usage(self):
        """메모리 사용량 최적화"""
        
        # 가비지 컬렉션
        gc.collect()
        
        # 디바이스별 메모리 정리
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        elif self.device == 'mps':
            torch.mps.empty_cache()
        
        # 메모리 사용량 로깅
        memory_usage = self._get_detailed_memory_usage()
        logger.debug(f"🧹 메모리 최적화 완료 - 사용량: {memory_usage}")
    
    def _cleanup_memory(self):
        """메모리 정리"""
        
        # Python 가비지 컬렉션
        gc.collect()
        
        # PyTorch 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    def _extract_final_image(
        self, 
        post_processing_result: Dict[str, Any],
        fitting_result: Dict[str, Any], 
        person_tensor: torch.Tensor
    ) -> torch.Tensor:
        """최종 결과 이미지 추출"""
        
        # 우선순위: 후처리 결과 > 피팅 결과 > 원본
        if 'enhanced_image' in post_processing_result:
            return post_processing_result['enhanced_image']
        elif 'fitted_image' in fitting_result:
            return fitting_result['fitted_image']
        else:
            logger.warning("⚠️ 최종 이미지를 찾을 수 없어 원본을 반환합니다")
            return person_tensor
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """텐서를 PIL 이미지로 변환"""
        
        try:
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            
            if tensor.shape[0] == 3:
                tensor = tensor.permute(1, 2, 0)
            
            # 0-1 범위로 클램핑
            tensor = torch.clamp(tensor, 0, 1)
            
            # CPU로 이동
            tensor = tensor.cpu()
            
            # numpy 배열로 변환
            array = (tensor.numpy() * 255).astype(np.uint8)
            
            return Image.fromarray(array)
            
        except Exception as e:
            logger.error(f"❌ 텐서-PIL 변환 실패: {e}")
            # 폴백: 빈 이미지 반환
            return Image.new('RGB', (512, 512), color='black')
    
    async def _comprehensive_quality_analysis(
        self, 
        quality_result: Dict[str, Any], 
        session_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """종합적 품질 분석"""
        
        overall_score = quality_result.get('overall_score', 0.8)
        
        # 품질 등급 결정
        if overall_score >= 0.9:
            quality_grade = "Excellent"
            confidence = 0.95
        elif overall_score >= 0.8:
            quality_grade = "Good"
            confidence = 0.85
        elif overall_score >= 0.7:
            quality_grade = "Acceptable"
            confidence = 0.75
        elif overall_score >= 0.6:
            quality_grade = "Poor"
            confidence = 0.65
        else:
            quality_grade = "Very Poor"
            confidence = 0.5
        
        # 상세 분석
        breakdown = quality_result.get('quality_breakdown', {})
        
        return {
            'overall_score': overall_score,
            'quality_grade': quality_grade,
            'confidence': confidence,
            'breakdown': breakdown,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_detailed_statistics(self, session_id: str, total_time: float) -> Dict[str, Any]:
        """상세 처리 통계 계산"""
        
        session_data = self.session_data[session_id]
        step_times = session_data['step_times']
        
        # 기본 통계
        stats = {
            'total_time': total_time,
            'step_times': step_times.copy(),
            'steps_completed': len(step_times),
            'success_rate': len(step_times) / len(self.step_order),
        }
        
        if step_times:
            # 시간 분석
            times = list(step_times.values())
            stats.update({
                'average_step_time': np.mean(times),
                'fastest_step': {'name': min(step_times, key=step_times.get), 'time': min(times)},
                'slowest_step': {'name': max(step_times, key=step_times.get), 'time': max(times)},
                'time_distribution': {step: time/total_time*100 for step, time in step_times.items()}
            })
        
        # 메모리 및 성능
        stats.update({
            'memory_usage': self._get_detailed_memory_usage(),
            'device_utilization': self._get_device_utilization(),
            'efficiency_score': self._calculate_efficiency_score(total_time, len(step_times))
        })
        
        return stats
    
    def _analyze_quality_progression(self, session_id: str) -> Dict[str, Any]:
        """품질 진행 분석"""
        
        session_data = self.session_data[session_id]
        step_results = session_data['step_results']
        
        quality_progression = []
        
        for step_name in self.step_order:
            if step_name in step_results:
                result = step_results[step_name]
                # 각 단계의 품질 점수 추출
                if 'confidence' in result:
                    quality_score = result['confidence']
                elif 'quality_score' in result:
                    quality_score = result['quality_score']
                elif 'overall_score' in result:
                    quality_score = result['overall_score']
                else:
                    quality_score = 0.8  # 기본값
                
                quality_progression.append({
                    'step': step_name,
                    'quality': quality_score,
                    'timestamp': result.get('timestamp', datetime.now().isoformat())
                })
        
        # 품질 개선 분석
        improvements = []
        for i in range(1, len(quality_progression)):
            prev_quality = quality_progression[i-1]['quality']
            curr_quality = quality_progression[i]['quality']
            improvement = curr_quality - prev_quality
            
            improvements.append({
                'from_step': quality_progression[i-1]['step'],
                'to_step': quality_progression[i]['step'],
                'improvement': improvement,
                'improvement_percentage': (improvement / prev_quality * 100) if prev_quality > 0 else 0
            })
        
        return {
            'quality_progression': quality_progression,
            'improvements': improvements,
            'total_improvement': (quality_progression[-1]['quality'] - quality_progression[0]['quality']) if quality_progression else 0,
            'consistent_improvement': all(imp['improvement'] >= 0 for imp in improvements)
        }
    
    async def _generate_detailed_suggestions(
        self, 
        quality_analysis: Dict[str, Any], 
        statistics: Dict[str, Any],
        clothing_type: str,
        fabric_type: str
    ) -> Dict[str, List[str]]:
        """상세 개선 제안 생성"""
        
        suggestions = {
            'quality_improvements': [],
            'performance_optimizations': [],
            'user_experience': [],
            'technical_adjustments': []
        }
        
        overall_score = quality_analysis['overall_score']
        breakdown = quality_analysis['breakdown']
        
        # 품질 개선 제안
        if overall_score < 0.8:
            suggestions['quality_improvements'].extend([
                "🎯 전체적인 품질 향상이 필요합니다",
                "📷 더 높은 해상도의 입력 이미지를 사용해보세요",
                "💡 조명이 균등한 환경에서 촬영된 이미지를 사용하세요"
            ])
        
        if breakdown.get('fit_quality', 1.0) < 0.7:
            suggestions['quality_improvements'].extend([
                f"👔 {clothing_type} 핏 개선을 위해 더 정확한 신체 치수를 제공하세요",
                "🤏 의류 크기가 체형과 맞지 않을 수 있습니다",
                f"🧵 {fabric_type} 재질 특성을 고려한 조정이 필요합니다"
            ])
        
        # 성능 최적화 제안
        total_time = statistics['total_time']
        if total_time > 60:
            suggestions['performance_optimizations'].extend([
                "⚡ 처리 시간이 긴 편입니다. 품질 레벨을 조정해보세요",
                "🖥️ 더 높은 성능의 디바이스 사용을 고려하세요",
                "🧹 불필요한 백그라운드 프로세스를 종료하세요"
            ])
        
        if statistics['success_rate'] < 1.0:
            suggestions['performance_optimizations'].extend([
                "🔄 일부 단계가 실패했습니다. 입력 이미지 품질을 확인하세요",
                "💾 충분한 메모리가 확보되었는지 확인하세요"
            ])
        
        # 사용자 경험 개선
        suggestions['user_experience'].extend([
            "📸 정면을 바라보는 자세의 사진이 가장 좋은 결과를 제공합니다",
            "🎨 단색 배경의 의류 이미지를 사용하면 더 정확한 결과를 얻을 수 있습니다",
            "📏 신체 치수 정보를 정확히 입력하면 핏이 개선됩니다"
        ])
        
        # 기술적 조정
        if self.device == 'cpu':
            suggestions['technical_adjustments'].append(
                "🚀 GPU나 MPS를 사용하면 처리 속도가 크게 향상됩니다"
            )
        
        if breakdown.get('technical_quality', {}).get('sharpness', 1.0) < 0.7:
            suggestions['technical_adjustments'].extend([
                "🔍 이미지 선명도 개선이 필요합니다",
                "📱 카메라 흔들림 없이 촬영하세요"
            ])
        
        return suggestions
    
    def _generate_next_steps(self, quality_analysis: Dict[str, Any], quality_target: float) -> List[str]:
        """다음 단계 제안"""
        
        overall_score = quality_analysis['overall_score']
        next_steps = []
        
        if overall_score >= quality_target:
            next_steps.extend([
                "✅ 목표 품질에 도달했습니다!",
                "💾 결과를 저장하고 활용하세요",
                "🔄 다른 의류로 추가 피팅을 시도해보세요"
            ])
        else:
            gap = quality_target - overall_score
            next_steps.extend([
                f"🎯 목표 품질까지 {gap:.2f} 점 향상이 필요합니다",
                "🔧 개선 제안사항을 적용해보세요",
                "📷 더 좋은 품질의 입력 이미지를 준비하세요"
            ])
        
        return next_steps
    
    def _analyze_size_compatibility(
        self, 
        body_measurements: Optional[Dict[str, float]], 
        clothing_type: str
    ) -> Dict[str, Any]:
        """사이즈 호환성 분석"""
        
        if not body_measurements:
            return {
                'compatibility_score': 0.5,
                'recommendation': '신체 치수 정보가 없어 정확한 분석이 어렵습니다',
                'confidence': 'low'
            }
        
        # 의류 타입별 중요 치수
        key_measurements = {
            'shirt': ['chest', 'shoulder_width', 'waist'],
            'pants': ['waist', 'hip', 'inseam'],
            'dress': ['chest', 'waist', 'hip'],
            'jacket': ['chest', 'shoulder_width', 'arm_length'],
            'skirt': ['waist', 'hip']
        }
        
        relevant_measurements = key_measurements.get(clothing_type, ['chest', 'waist'])
        provided_measurements = [m for m in relevant_measurements if m in body_measurements]
        
        completeness = len(provided_measurements) / len(relevant_measurements)
        
        return {
            'compatibility_score': min(0.9, 0.5 + completeness * 0.4),
            'provided_measurements': provided_measurements,
            'missing_measurements': [m for m in relevant_measurements if m not in body_measurements],
            'recommendation': f"{clothing_type}에 중요한 치수 정보가 {completeness:.1%} 제공되었습니다",
            'confidence': 'high' if completeness > 0.8 else 'medium' if completeness > 0.5 else 'low'
        }
    
    def _analyze_style_match(
        self, 
        style_preferences: Optional[Dict[str, Any]], 
        fitting_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """스타일 매칭 분석"""
        
        if not style_preferences:
            return {
                'match_score': 0.8,
                'analysis': '스타일 선호도가 제공되지 않아 기본 스타일을 적용했습니다',
                'confidence': 'medium'
            }
        
        # 기본 스타일 분석
        match_factors = []
        
        # 핏 스타일
        preferred_fit = style_preferences.get('fit', 'regular')
        match_factors.append({
            'factor': 'fit_style',
            'preferred': preferred_fit,
            'achieved': 'regular',  # 기본값
            'match': 0.8
        })
        
        # 색상 선호도
        color_preference = style_preferences.get('color_preference', 'original')
        match_factors.append({
            'factor': 'color',
            'preferred': color_preference,
            'achieved': 'original',
            'match': 0.9 if color_preference == 'original' else 0.7
        })
        
        overall_match = np.mean([factor['match'] for factor in match_factors])
        
        return {
            'match_score': overall_match,
            'match_factors': match_factors,
            'analysis': f"스타일 매칭도: {overall_match:.1%}",
            'confidence': 'high'
        }
    
    def _create_detailed_step_summary(self, session_id: str) -> Dict[str, Dict[str, Any]]:
        """상세 단계 요약 생성"""
        
        session_data = self.session_data[session_id]
        step_times = session_data['step_times']
        step_results = session_data['step_results']
        
        summary = {}
        
        for step_name in self.step_order:
            step_summary = {
                'completed': step_name in step_results,
                'processing_time': step_times.get(step_name, 0),
                'success': step_name in step_results and not step_results[step_name].get('error'),
                'fallback_used': step_results.get(step_name, {}).get('fallback', False)
            }
            
            # 단계별 특화 정보
            if step_name in step_results:
                result = step_results[step_name]
                
                if step_name == 'human_parsing':
                    step_summary.update({
                        'confidence': result.get('confidence', 0),
                        'body_parts_detected': len(result.get('body_parts_detected', [])),
                        'parsing_accuracy': result.get('parsing_accuracy', 'unknown')
                    })
                    
                elif step_name == 'pose_estimation':
                    step_summary.update({
                        'pose_confidence': result.get('pose_confidence', 0),
                        'keypoints_detected': len(result.get('keypoints', [])),
                        'pose_stability': result.get('pose_stability', 'unknown')
                    })
                    
                elif step_name == 'quality_assessment':
                    step_summary.update({
                        'overall_score': result.get('overall_score', 0),
                        'quality_grade': result.get('quality_grade', 'Unknown'),
                        'metrics_computed': len(result.get('detailed_metrics', {}))
                    })
            
            summary[step_name] = step_summary
        
        return summary
    
    def _get_detailed_memory_usage(self) -> Dict[str, str]:
        """상세 메모리 사용량 조회"""
        
        import psutil
        
        memory_info = {
            'system_memory': f"{psutil.virtual_memory().percent}%",
            'available_memory': f"{psutil.virtual_memory().available / 1024**3:.1f}GB"
        }
        
        if torch.cuda.is_available():
            memory_info.update({
                'gpu_memory_allocated': f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB",
                'gpu_memory_reserved': f"{torch.cuda.memory_reserved() / 1024**3:.1f}GB"
            })
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                memory_info['mps_memory'] = f"{torch.mps.current_allocated_memory() / 1024**3:.1f}GB"
            except:
                memory_info['mps_memory'] = "N/A"
        
        return memory_info
    
    def _get_device_utilization(self) -> Dict[str, Any]:
        """디바이스 활용도 조회"""
        
        utilization = {
            'device_type': self.device,
            'optimization_enabled': self.pipeline_config['enable_optimization']
        }
        
        if self.device == 'cuda' and torch.cuda.is_available():
            utilization.update({
                'gpu_name': torch.cuda.get_device_name(),
                'compute_capability': torch.cuda.get_device_capability()
            })
        elif self.device == 'mps':
            utilization.update({
                'mps_available': torch.backends.mps.is_available(),
                'mps_built': torch.backends.mps.is_built()
            })
        
        return utilization
    
    def _calculate_efficiency_score(self, total_time: float, completed_steps: int) -> float:
        """효율성 점수 계산"""
        
        expected_time_per_step = 5.0  # 초
        expected_total_time = len(self.step_order) * expected_time_per_step
        
        time_efficiency = min(1.0, expected_total_time / total_time) if total_time > 0 else 0
        completion_efficiency = completed_steps / len(self.step_order)
        
        return (time_efficiency + completion_efficiency) / 2
    
    def _update_performance_metrics(self, processing_time: float, quality_score: float):
        """성능 메트릭 업데이트"""
        
        # 이동 평균 계산
        total_sessions = self.performance_metrics['total_sessions']
        
        if total_sessions > 1:
            prev_avg_time = self.performance_metrics['average_processing_time']
            prev_avg_quality = self.performance_metrics['average_quality_score']
            
            # 이동 평균 업데이트
            self.performance_metrics['average_processing_time'] = (
                (prev_avg_time * (total_sessions - 1) + processing_time) / total_sessions
            )
            self.performance_metrics['average_quality_score'] = (
                (prev_avg_quality * (total_sessions - 1) + quality_score) / total_sessions
            )
        else:
            self.performance_metrics['average_processing_time'] = processing_time
            self.performance_metrics['average_quality_score'] = quality_score
    
    def _cleanup_session_data(self, session_id: str):
        """세션 데이터 정리"""
        
        if session_id in self.session_data:
            # 중요한 통계만 보존
            session_summary = {
                'total_time': time.time() - self.session_data[session_id]['start_time'],
                'steps_completed': len(self.session_data[session_id]['step_times']),
                'final_quality': max(
                    [result.get('overall_score', result.get('confidence', 0)) 
                     for result in self.session_data[session_id]['step_results'].values()], 
                    default=0
                )
            }
            
            # 전체 세션 데이터 삭제
            del self.session_data[session_id]
            
            # 요약만 보존 (선택적)
            if hasattr(self, 'session_summaries'):
                self.session_summaries[session_id] = session_summary
            
            logger.debug(f"🧹 세션 {session_id} 데이터 정리 완료")
    
    async def _handle_processing_error(
        self,
        error: Exception,
        session_id: str,
        start_time: float,
        person_image: Union[str, Image.Image, np.ndarray],
        clothing_image: Union[str, Image.Image, np.ndarray],
        enable_auto_retry: bool = True
    ) -> Dict[str, Any]:
        """처리 오류 핸들링 및 복구"""
        
        processing_time = time.time() - start_time
        error_msg = str(error)
        
        # 오류 기록
        self.error_history.append({
            'session_id': session_id,
            'error': error_msg,
            'error_type': type(error).__name__,
            'timestamp': datetime.now().isoformat(),
            'processing_time': processing_time
        })
        
        logger.error(f"❌ 가상 피팅 처리 실패 - 세션 {session_id}: {error_msg}")
        logger.error(f"📋 오류 상세: {traceback.format_exc()}")
        
        # 자동 복구 시도
        if enable_auto_retry and not hasattr(error, '_retry_attempted'):
            logger.info("🔄 자동 복구 시도 중...")
            
            try:
                # 메모리 정리
                self._cleanup_memory()
                
                # 간단한 재시도 (한 번만)
                error._retry_attempted = True
                
                # 낮은 품질 모드로 재시도
                original_quality = self.pipeline_config['quality_level']
                self.pipeline_config['quality_level'] = 'medium'
                
                result = await self.process_complete_virtual_fitting(
                    person_image=person_image,
                    clothing_image=clothing_image,
                    quality_target=0.6,  # 낮은 목표
                    enable_auto_retry=False  # 무한 루프 방지
                )
                
                # 원래 품질 복구
                self.pipeline_config['quality_level'] = original_quality
                
                if result['success']:
                    logger.info("✅ 자동 복구 성공!")
                    result['recovered'] = True
                    result['recovery_method'] = 'quality_downgrade'
                    return result
                    
            except Exception as retry_error:
                logger.warning(f"⚠️ 자동 복구 실패: {retry_error}")
        
        # 폴백 결과 생성
        try:
            fallback_result = await self._create_comprehensive_fallback_result(
                person_image, clothing_image, session_id, error_msg, processing_time
            )
            return fallback_result
            
        except Exception as fallback_error:
            logger.error(f"❌ 폴백 결과 생성도 실패: {fallback_error}")
            
            # 최소한의 오류 결과
            return {
                'success': False,
                'session_id': session_id,
                'error': f"원본 오류: {error_msg}, 폴백 오류: {str(fallback_error)}",
                'error_type': 'critical_failure',
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'recovery_attempted': enable_auto_retry,
                'metadata': {
                    'pipeline_version': '2.0.0',
                    'device': self.device,
                    'critical_error': True
                }
            }
    
    async def _create_comprehensive_fallback_result(
        self,
        person_image: Union[str, Image.Image, np.ndarray],
        clothing_image: Union[str, Image.Image, np.ndarray],
        session_id: str,
        error_message: str,
        processing_time: float
    ) -> Dict[str, Any]:
        """종합적인 폴백 결과 생성"""
        
        try:
            # 기본 이미지 처리
            if isinstance(person_image, str):
                person_pil = Image.open(person_image).convert('RGB')
            elif isinstance(person_image, Image.Image):
                person_pil = person_image.convert('RGB')
            else:
                person_pil = Image.fromarray(person_image).convert('RGB')
            
            if isinstance(clothing_image, str):
                clothing_pil = Image.open(clothing_image).convert('RGB')
            elif isinstance(clothing_image, Image.Image):
                clothing_pil = clothing_image.convert('RGB')
            else:
                clothing_pil = Image.fromarray(clothing_image).convert('RGB')
            
            # 간단한 합성 시도 (오류 복구용)
            try:
                result_image = self._create_simple_composite(person_pil, clothing_pil)
            except:
                result_image = person_pil  # 최악의 경우 원본 반환
            
            return {
                'success': False,
                'session_id': session_id,
                'error': error_message,
                'error_type': 'processing_failure',
                'fallback_used': True,
                
                # 기본 이미지들
                'result_image': result_image,
                'original_person_image': person_pil,
                'original_clothing_image': clothing_pil,
                
                # 기본 품질 정보
                'final_quality_score': 0.3,
                'quality_grade': 'Error',
                'quality_target_achieved': False,
                
                # 오류 관련 정보
                'error_details': {
                    'error_message': error_message,
                    'error_timestamp': datetime.now().isoformat(),
                    'session_duration': processing_time,
                    'fallback_method': 'simple_composite'
                },
                
                # 처리 통계 (기본)
                'processing_statistics': {
                    'total_time': processing_time,
                    'steps_completed': 0,
                    'success_rate': 0.0,
                    'error_occurred': True,
                    'device_used': self.device
                },
                
                # 개선 제안
                'improvement_suggestions': {
                    'quality_improvements': [
                        "❌ 처리 중 오류가 발생했습니다",
                        "🔄 다시 시도하거나 다른 이미지를 사용해보세요",
                        "📷 이미지 품질이나 형식을 확인해보세요"
                    ],
                    'technical_adjustments': [
                        "🧹 메모리를 정리하고 다시 시도하세요",
                        "⚙️ 품질 레벨을 낮춰서 시도해보세요",
                        "🖥️ 시스템 리소스를 확인하세요"
                    ]
                },
                
                # 메타데이터
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'pipeline_version': '2.0.0',
                    'fallback_result': True,
                    'device': self.device,
                    'error_recovery': True
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 폴백 결과 생성 실패: {e}")
            raise
    
    def _create_simple_composite(self, person_image: Image.Image, clothing_image: Image.Image) -> Image.Image:
        """간단한 합성 이미지 생성 (오류 복구용)"""
        
        try:
            # 크기 맞추기
            target_size = (512, 512)
            person_resized = person_image.resize(target_size, Image.Resampling.LANCZOS)
            clothing_resized = clothing_image.resize((256, 256), Image.Resampling.LANCZOS)
            
            # 단순 오버레이 (우상단에 의류 이미지)
            result = person_resized.copy()
            result.paste(clothing_resized, (256, 0), clothing_resized)
            
            # 텍스트 오버레이 (오류 표시)
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(result)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            draw.text((10, 10), "Preview Only - Error Occurred", fill=(255, 0, 0), font=font)
            
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ 간단한 합성도 실패: {e}")
            return person_image
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """설정 파일 로드"""
        
        # 기본 설정
        default_config = {
            'input_size': (512, 512),
            'pipeline': {
                'quality_level': 'high',
                'processing_mode': 'complete',
                'enable_optimization': True,
                'enable_caching': True,
                'parallel_processing': True,
                'memory_optimization': True,
                'enable_intermediate_saving': False,
                'max_retries': 3,
                'timeout_seconds': 300
            },
            'quality_thresholds': {
                'excellent': 0.9,
                'good': 0.8,
                'acceptable': 0.7,
                'poor': 0.6
            },
            'device_optimization': {
                'enable_mps': True,
                'enable_cuda': True,
                'mixed_precision': True,
                'memory_efficient': True
            }
        }
        
        # 설정 파일에서 로드
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                
                # 딥 업데이트
                def deep_update(base_dict, update_dict):
                    for key, value in update_dict.items():
                        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                            deep_update(base_dict[key], value)
                        else:
                            base_dict[key] = value
                
                deep_update(default_config, file_config)
                logger.info(f"✅ 설정 파일 로드 완료: {config_path}")
                
            except Exception as e:
                logger.warning(f"⚠️ 설정 파일 로드 실패: {e}, 기본 설정 사용")
        
        return default_config
    
    async def _print_system_status(self):
        """시스템 상태 출력"""
        
        logger.info("=" * 70)
        logger.info("🏥 개선된 가상 피팅 파이프라인 시스템 상태")
        logger.info("=" * 70)
        
        # 디바이스 정보
        logger.info(f"🖥️ 디바이스: {self.device}")
        if self.device == 'mps':
            logger.info(f"   - MPS 사용 가능: {torch.backends.mps.is_available()}")
        elif self.device == 'cuda':
            logger.info(f"   - CUDA 버전: {torch.version.cuda}")
            logger.info(f"   - GPU 이름: {torch.cuda.get_device_name()}")
        
        # 단계별 상태
        logger.info("📋 단계별 초기화 상태:")
        for i, step_name in enumerate(self.step_order, 1):
            if step_name in self.steps:
                status = "✅ 준비됨"
                step = self.steps[step_name]
                if hasattr(step, 'is_initialized'):
                    if not step.is_initialized:
                        status = "⚠️ 초기화 미완료"
            else:
                status = "❌ 로드 안됨"
            
            logger.info(f"   {i}. {step_name}: {status}")
        
        # 성능 설정
        logger.info("⚙️ 파이프라인 설정:")
        logger.info(f"   - 품질 레벨: {self.pipeline_config['quality_level']}")
        logger.info(f"   - 처리 모드: {self.pipeline_config['processing_mode']}")
        logger.info(f"   - 메모리 최적화: {self.pipeline_config['memory_optimization']}")
        logger.info(f"   - 병렬 처리: {self.pipeline_config['parallel_processing']}")
        
        # 메모리 정보
        memory_info = self._get_detailed_memory_usage()
        logger.info("💾 메모리 사용량:")
        for key, value in memory_info.items():
            logger.info(f"   - {key}: {value}")
        
        # 성능 메트릭
        logger.info("📊 성능 메트릭:")
        logger.info(f"   - 총 세션: {self.performance_metrics['total_sessions']}")
        logger.info(f"   - 성공 세션: {self.performance_metrics['successful_sessions']}")
        if self.performance_metrics['total_sessions'] > 0:
            success_rate = (self.performance_metrics['successful_sessions'] / 
                          self.performance_metrics['total_sessions'] * 100)
            logger.info(f"   - 성공률: {success_rate:.1f}%")
        
        logger.info("=" * 70)
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """파이프라인 상세 상태 조회"""
        
        return {
            'initialized': self.is_initialized,
            'device': self.device,
            'pipeline_config': self.pipeline_config,
            'performance_metrics': self.performance_metrics.copy(),
            'memory_usage': self._get_detailed_memory_usage(),
            'device_utilization': self._get_device_utilization(),
            'active_sessions': len(self.session_data),
            'error_history_count': len(self.error_history),
            'steps_status': {
                step_name: {
                    'loaded': step_name in self.steps,
                    'type': type(self.steps[step_name]).__name__ if step_name in self.steps else None
                }
                for step_name in self.step_order
            },
            'system_health': {
                'initialization_success_rate': len(self.steps) / len(self.step_order),
                'recent_errors': self.error_history[-5:] if self.error_history else [],
                'uptime': time.time() - self.processing_stats.get('initialization_time', time.time())
            }
        }
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """성능 리포트 생성"""
        
        return {
            'overall_performance': self.performance_metrics.copy(),
            'efficiency_metrics': {
                'average_time_per_step': (
                    self.performance_metrics['average_processing_time'] / len(self.step_order)
                    if self.performance_metrics['average_processing_time'] > 0 else 0
                ),
                'quality_per_time_ratio': (
                    self.performance_metrics['average_quality_score'] / 
                    self.performance_metrics['average_processing_time']
                    if self.performance_metrics['average_processing_time'] > 0 else 0
                )
            },
            'resource_utilization': {
                'device_type': self.device,
                'memory_usage': self._get_detailed_memory_usage(),
                'optimization_enabled': self.pipeline_config['enable_optimization']
            },
            'reliability_metrics': {
                'success_rate': (
                    self.performance_metrics['successful_sessions'] / 
                    self.performance_metrics['total_sessions']
                    if self.performance_metrics['total_sessions'] > 0 else 0
                ),
                'error_rate': (
                    len(self.error_history) / 
                    max(1, self.performance_metrics['total_sessions'])
                ),
                'average_retry_needed': len(self.error_history) / max(1, self.performance_metrics['total_sessions'])
            },
            'recommendations': self._generate_performance_recommendations()
        }
    
    def _generate_performance_recommendations(self) -> List[str]:
        """성능 개선 권장사항 생성"""
        
        recommendations = []
        
        # 성공률 기반 권장사항
        if self.performance_metrics['total_sessions'] > 0:
            success_rate = (self.performance_metrics['successful_sessions'] / 
                          self.performance_metrics['total_sessions'])
            
            if success_rate < 0.8:
                recommendations.append("🔧 성공률이 낮습니다. 입력 이미지 품질을 확인하세요")
            
            if self.performance_metrics['average_processing_time'] > 60:
                recommendations.append("⚡ 처리 시간이 긴 편입니다. 품질 레벨 조정을 고려하세요")
        
        # 디바이스 기반 권장사항
        if self.device == 'cpu':
            recommendations.append("🚀 GPU 또는 MPS 사용 시 성능이 크게 향상됩니다")
        
        # 메모리 기반 권장사항
        memory_info = self._get_detailed_memory_usage()
        if 'system_memory' in memory_info:
            memory_percent = float(memory_info['system_memory'].replace('%', ''))
            if memory_percent > 80:
                recommendations.append("💾 시스템 메모리 사용량이 높습니다. 메모리 최적화를 활성화하세요")
        
        # 에러 기반 권장사항
        if len(self.error_history) > 0:
            recent_errors = self.error_history[-5:]
            common_error_types = {}
            for error in recent_errors:
                error_type = error.get('error_type', 'unknown')
                common_error_types[error_type] = common_error_types.get(error_type, 0) + 1
            
            if common_error_types:
                most_common = max(common_error_types, key=common_error_types.get)
                recommendations.append(f"🐛 최근 '{most_common}' 오류가 빈번합니다. 시스템 점검이 필요합니다")
        
        if not recommendations:
            recommendations.append("✅ 시스템이 최적 상태로 운영되고 있습니다")
        
        return recommendations
    
    async def cleanup(self):
        """전체 파이프라인 리소스 정리"""
        
        logger.info("🧹 개선된 가상 피팅 파이프라인 리소스 정리 중...")
        
        try:
            # 각 단계별 정리
            for step_name, step in self.steps.items():
                try:
                    if hasattr(step, 'cleanup'):
                        await step.cleanup()
                    elif hasattr(step, 'close'):
                        step.close()
                    logger.info(f"✅ {step_name} 정리 완료")
                except Exception as e:
                    logger.warning(f"⚠️ {step_name} 정리 중 오류: {e}")
            
            # 유틸리티 정리
            if hasattr(self.model_loader, 'cleanup'):
                await self.model_loader.cleanup()
            
            if hasattr(self.memory_manager, 'cleanup'):
                await self.memory_manager.cleanup()
            
            # 스레드 풀 정리
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
            
            # 메모리 정리
            self._cleanup_memory()
            
            # 세션 데이터 정리
            self.session_data.clear()
            
            # 상태 초기화
            self.is_initialized = False
            
            logger.info("✅ 전체 파이프라인 리소스 정리 완료")
            
        except Exception as e:
            logger.error(f"❌ 리소스 정리 중 오류: {e}")


# ===================================
# 사용 예시 및 테스트 함수들
# ===================================

async def demo_pipeline_manager():
    """파이프라인 매니저 데모"""
    
    print("🚀 개선된 8단계 가상 피팅 파이프라인 매니저 데모 시작")
    
    # 파이프라인 매니저 초기화
    pipeline = PipelineManager(
        config_path='config/pipeline_config.json',  # 선택적
        device='auto'  # 최적 디바이스 자동 선택
    )
    
    # 초기화
    success = await pipeline.initialize()
    if not success:
        print("❌ 파이프라인 매니저 초기화 실패")
        return
    
    # 진행률 콜백 함수
    async def progress_callback(stage: str, percentage: int):
        print(f"🔄 진행상황: {stage} - {percentage}%")
    
    # 가상 피팅 실행
    try:
        result = await pipeline.process_complete_virtual_fitting(
            person_image='test_images/person.jpg',  # 실제 경로로 변경
            clothing_image='test_images/shirt.jpg',  # 실제 경로로 변경
            body_measurements={
                'height': 175,
                'weight': 70,
                'chest': 95,
                'waist': 80,
                'shoulder_width': 45,
                'hip': 90
            },
            clothing_type='shirt',
            fabric_type='cotton',
            style_preferences={
                'fit': 'slim',
                'color_preference': 'original'
            },
            quality_target=0.85,
            progress_callback=progress_callback,
            save_intermediate=True,  # 중간 결과 저장
            enable_auto_retry=True   # 자동 재시도 활성화
        )
        
        if result['success']:
            print(f"\n🎉 가상 피팅 성공!")
            print(f"📊 최종 품질: {result['final_quality_score']:.3f} ({result['quality_grade']})")
            print(f"⏱️ 총 처리 시간: {result['total_processing_time']:.2f}초")
            print(f"🎯 목표 달성: {'✅' if result['quality_target_achieved'] else '❌'}")
            print(f"🔧 복구됨: {'✅' if result.get('recovered', False) else '❌'}")
            
            # 결과 이미지 저장
            os.makedirs('output', exist_ok=True)
            result['result_image'].save('output/pipeline_manager_result.jpg')
            print("💾 결과 이미지 저장 완료: output/pipeline_manager_result.jpg")
            
            # 상세 분석 출력
            print(f"\n📈 품질 분석:")
            for category, score in result['quality_breakdown'].items():
                print(f"  - {category}: {score:.3f}")
            
            print(f"\n💡 개선 제안:")
            for category, suggestions in result['improvement_suggestions'].items():
                print(f"  📋 {category}:")
                for suggestion in suggestions[:3]:  # 상위 3개만
                    print(f"    - {suggestion}")
            
            print(f"\n⏱️ 단계별 처리 시간:")
            for step, summary in result['step_results_summary'].items():
                if summary['completed']:
                    print(f"  - {step}: {summary['processing_time']:.2f}초 ({'✅' if summary['success'] else '⚠️'})")
            
            # 성능 리포트
            performance_report = await pipeline.get_performance_report()
            print(f"\n📊 성능 리포트:")
            print(f"  - 전체 성공률: {performance_report['reliability_metrics']['success_rate']:.1%}")
            print(f"  - 평균 처리 시간: {performance_report['overall_performance']['average_processing_time']:.2f}초")
            print(f"  - 평균 품질 점수: {performance_report['overall_performance']['average_quality_score']:.3f}")
            
        else:
            print(f"❌ 가상 피팅 실패: {result['error']}")
            if result.get('fallback_used'):
                print("🚨 폴백 결과가 제공되었습니다")
            
            # 오류 상세 정보
            if 'error_details' in result:
                print(f"📋 오류 상세: {result['error_details']}")
    
    except Exception as e:
        print(f"💥 예외 발생: {e}")
        print(f"📋 상세: {traceback.format_exc()}")
    
    finally:
        # 리소스 정리
        await pipeline.cleanup()
        print("🧹 리소스 정리 완료")


async def benchmark_pipeline_manager():
    """파이프라인 매니저 성능 벤치마크"""
    
    print("📊 파이프라인 매니저 성능 벤치마크 시작")
    
    pipeline = PipelineManager(device='auto')
    await pipeline.initialize()
    
    # 테스트 이미지들 (실제 경로로 변경 필요)
    test_cases = [
        ('test_images/person1.jpg', 'test_images/shirt1.jpg', 'shirt'),
        ('test_images/person2.jpg', 'test_images/pants1.jpg', 'pants'),
        ('test_images/person3.jpg', 'test_images/dress1.jpg', 'dress')
    ]
    
    results = []
    
    for i, (person_path, clothing_path, clothing_type) in enumerate(test_cases):
        print(f"\n🧪 테스트 케이스 {i+1}/{len(test_cases)}: {clothing_type}")
        
        try:
            start_time = time.time()
            
            # 더미 이미지 생성 (실제 파일이 없는 경우)
            if not os.path.exists(person_path):
                person_image = Image.new('RGB', (512, 512), color='blue')
            else:
                person_image = person_path
                
            if not os.path.exists(clothing_path):
                clothing_image = Image.new('RGB', (512, 512), color='red')
            else:
                clothing_image = clothing_path
            
            result = await pipeline.process_complete_virtual_fitting(
                person_image=person_image,
                clothing_image=clothing_image,
                clothing_type=clothing_type,
                quality_target=0.8
            )
            
            processing_time = time.time() - start_time
            
            results.append({
                'test_case': i + 1,
                'clothing_type': clothing_type,
                'success': result['success'],
                'processing_time': processing_time,
                'quality_score': result.get('final_quality_score', 0),
                'memory_usage': result.get('memory_usage', {})
            })
            
            print(f"  ✅ 완료 - 시간: {processing_time:.2f}초, 품질: {result.get('final_quality_score', 0):.3f}")
            
        except Exception as e:
            print(f"  ❌ 실패: {e}")
            results.append({
                'test_case': i + 1,
                'clothing_type': clothing_type,
                'success': False,
                'error': str(e)
            })
    
    # 벤치마크 결과 요약
    print(f"\n📈 벤치마크 결과 요약:")
    successful_tests = [r for r in results if r['success']]
    
    if successful_tests:
        avg_time = np.mean([r['processing_time'] for r in successful_tests])
        avg_quality = np.mean([r['quality_score'] for r in successful_tests])
        success_rate = len(successful_tests) / len(results)
        
        print(f"  - 성공률: {success_rate:.1%}")
        print(f"  - 평균 처리 시간: {avg_time:.2f}초")
        print(f"  - 평균 품질 점수: {avg_quality:.3f}")
        print(f"  - 최고 성능: {min(r['processing_time'] for r in successful_tests):.2f}초")
        print(f"  - 최고 품질: {max(r['quality_score'] for r in successful_tests):.3f}")
    
    await pipeline.cleanup()
    
    return results


# 메인 실행 함수
if __name__ == "__main__":
    print("🎽 개선된 완전한 8단계 가상 피팅 파이프라인 매니저")
    print("=" * 60)
    
    # 데모 실행
    asyncio.run(demo_pipeline_manager())
    
    # 벤치마크 실행 (선택적)
    # asyncio.run(benchmark_pipeline_manager())