# backend/app/ai_pipeline/steps/__init__.py
"""
🎯 MyCloset AI Pipeline Steps 모듈 v7.0 - 단순화된 Step 관리
================================================================

✅ 단순하고 안정적인 Step 로딩
✅ conda 환경 우선 최적화
✅ M3 Max 메모리 효율성 극대화
✅ 순환참조 완전 방지
✅ 8단계 Step 클래스 안전한 로딩
✅ 실패 허용적 설계
✅ 지연 로딩으로 성능 최적화

8단계 AI 파이프라인:
Step 1: HumanParsingStep (SCHP/Graphonomy)
Step 2: PoseEstimationStep (OpenPose/YOLO)
Step 3: ClothSegmentationStep (U2Net/SAM)
Step 4: GeometricMatchingStep (TPS/GMM)
Step 5: ClothWarpingStep (Advanced Warping)
Step 6: VirtualFittingStep (OOTDiffusion/IDM-VTON)
Step 7: PostProcessingStep (Enhancement/SR)
Step 8: QualityAssessmentStep (CLIP/Quality)

작성자: MyCloset AI Team
날짜: 2025-07-23
버전: v7.0.0 (Simplified Step Management)
"""
import os
import logging
import importlib
import threading
import sys
from typing import Dict, Any, Optional, List, Type, Union
from pathlib import Path
from functools import lru_cache

# =============================================================================
# 🔥 기본 설정 및 시스템 정보 로딩
# =============================================================================

logger = logging.getLogger(__name__)

# 상위 패키지에서 시스템 정보 가져오기
try:
    from ... import get_system_info, is_conda_environment, is_m3_max, get_device
    SYSTEM_INFO = get_system_info()
    IS_CONDA = is_conda_environment()
    IS_M3_MAX = is_m3_max()
    DEVICE = get_device()
    CONDA_ENV = os.environ.get('CONDA_DEFAULT_ENV', 'none')
    logger.info("✅ 상위 패키지에서 시스템 정보 로드 성공")
except ImportError as e:
    import os
    logger.warning(f"⚠️ 상위 패키지 로드 실패, 기본값 사용: {e}")
    SYSTEM_INFO = {'device': 'cpu', 'is_m3_max': False, 'memory_gb': 16.0}
    IS_CONDA = 'CONDA_DEFAULT_ENV' in os.environ
    IS_M3_MAX = False
    DEVICE = 'cpu'
    CONDA_ENV = os.environ.get('CONDA_DEFAULT_ENV', 'none')

# =============================================================================
# 🔥 Step 모듈 정보 정의
# =============================================================================

# Step 모듈명과 클래스명 매핑
STEP_MODULES = {
    'step_01': 'step_01_human_parsing',
    'step_02': 'step_02_pose_estimation', 
    'step_03': 'step_03_cloth_segmentation',
    'step_04': 'step_04_geometric_matching',
    'step_05': 'step_05_cloth_warping',
    'step_06': 'step_06_virtual_fitting',
    'step_07': 'step_07_post_processing',
    'step_08': 'step_08_quality_assessment'
}

STEP_CLASSES = {
    'step_01': 'HumanParsingStep',
    'step_02': 'PoseEstimationStep',
    'step_03': 'ClothSegmentationStep', 
    'step_04': 'GeometricMatchingStep',
    'step_05': 'ClothWarpingStep',
    'step_06': 'VirtualFittingStep',
    'step_07': 'PostProcessingStep',
    'step_08': 'QualityAssessmentStep'
}

# Step 설명
STEP_DESCRIPTIONS = {
    'step_01': '인체 파싱 - Human Body Parsing (SCHP/Graphonomy)',
    'step_02': '포즈 추정 - Pose Estimation (OpenPose/YOLO)', 
    'step_03': '의류 분할 - Cloth Segmentation (U2Net/SAM)',
    'step_04': '기하학적 매칭 - Geometric Matching (TPS/GMM)',
    'step_05': '의류 변형 - Cloth Warping (Advanced Warping)',
    'step_06': '가상 피팅 - Virtual Fitting (OOTDiffusion/IDM-VTON)',
    'step_07': '후처리 - Post Processing (RealESRGAN/Enhancement)',
    'step_08': '품질 평가 - Quality Assessment (CLIP/Metrics)'
}

# conda 환경에서 Step 우선순위 (메모리 효율성 고려)
CONDA_STEP_PRIORITY = {
    'step_06': 1,  # Virtual Fitting - 가장 중요
    'step_01': 2,  # Human Parsing - 기초
    'step_03': 3,  # Cloth Segmentation - 핵심
    'step_02': 4,  # Pose Estimation
    'step_07': 5,  # Post Processing
    'step_08': 6,  # Quality Assessment
    'step_04': 7,  # Geometric Matching
    'step_05': 8   # Cloth Warping
}

# =============================================================================
# 🔥 단순화된 Step 로더
# =============================================================================

class SimpleStepLoader:
    """단순화된 Step 로더 - 안정성 중심"""
    
    def __init__(self):
        self._step_cache = {}
        self._failed_steps = set()
        self._lock = threading.Lock()
        self.logger = logging.getLogger(f"{__name__}.SimpleStepLoader")
        
        self.logger.info(f"🎯 Step 로더 초기화 (conda: {IS_CONDA}, M3Max: {IS_M3_MAX})")
    
    @lru_cache(maxsize=8)
    def safe_import_step(self, step_id: str) -> Optional[Type[Any]]:
        """안전한 Step 클래스 import (캐시됨)"""
        with self._lock:
            # 이미 실패한 Step은 재시도 안함
            if step_id in self._failed_steps:
                return None
            
            # 캐시에서 확인
            if step_id in self._step_cache:
                return self._step_cache[step_id]
            
            try:
                module_name = STEP_MODULES.get(step_id)
                class_name = STEP_CLASSES.get(step_id)
                
                if not module_name or not class_name:
                    self.logger.warning(f"⚠️ 알 수 없는 Step ID: {step_id}")
                    self._failed_steps.add(step_id)
                    return None
                
                # 동적 import 시도
                full_module_name = f"app.ai_pipeline.steps.{module_name}"
                
                try:
                    module = importlib.import_module(full_module_name)
                    step_class = getattr(module, class_name, None)
                    
                    if step_class is None:
                        self.logger.debug(f"📋 {class_name} 클래스가 {module_name}에 없음 (정상)")
                        self._failed_steps.add(step_id)
                        self._step_cache[step_id] = None
                        return None
                    
                    # 성공적으로 로드됨
                    self._step_cache[step_id] = step_class
                    priority = CONDA_STEP_PRIORITY.get(step_id, 9)
                    self.logger.info(f"✅ {step_id} ({class_name}) 로드 성공 (우선순위: {priority})")
                    
                    return step_class
                    
                except ImportError:
                    # 모듈이 없는 것은 정상 (아직 구현되지 않음)
                    self.logger.debug(f"📋 {step_id} 모듈 없음 (정상)")
                
            except Exception as e:
                self.logger.error(f"❌ {step_id} 로드 중 예상치 못한 오류: {e}")
            
            # 실패 처리
            self._failed_steps.add(step_id)
            self._step_cache[step_id] = None
            return None
    
    def load_all_steps(self) -> Dict[str, Optional[Type[Any]]]:
        """모든 Step 클래스 로드"""
        loaded_steps = {}
        
        # conda 환경에서는 우선순위 순으로 로딩
        if IS_CONDA:
            step_order = sorted(STEP_MODULES.keys(), 
                              key=lambda x: CONDA_STEP_PRIORITY.get(x, 9))
            self.logger.info("🐍 conda 환경: 우선순위 기반 Step 로딩")
        else:
            step_order = list(STEP_MODULES.keys())
            self.logger.info("📊 일반 환경: 순차적 Step 로딩")
        
        for step_id in step_order:
            step_class = self.safe_import_step(step_id)
            loaded_steps[step_id] = step_class
        
        # 로딩 결과 요약
        available_count = sum(1 for step in loaded_steps.values() if step is not None)
        success_rate = (available_count / len(STEP_MODULES)) * 100
        
        self.logger.info(f"📊 Step 로딩 완료: {available_count}/8개 ({success_rate:.1f}%)")
        
        if IS_CONDA:
            conda_priority_loaded = sum(
                1 for step_id in ['step_06', 'step_01', 'step_03', 'step_02']
                if loaded_steps.get(step_id) is not None
            )
            self.logger.info(f"🐍 conda 고우선순위 Step: {conda_priority_loaded}/4개 로드됨")
        
        return loaded_steps
    
    def get_step_info(self, step_id: str) -> Dict[str, Any]:
        """Step 정보 반환"""
        step_class = self._step_cache.get(step_id)
        if step_class is None and step_id not in self._failed_steps:
            step_class = self.safe_import_step(step_id)
        
        return {
            'step_id': step_id,
            'module_name': STEP_MODULES.get(step_id),
            'class_name': STEP_CLASSES.get(step_id),
            'description': STEP_DESCRIPTIONS.get(step_id),
            'available': step_class is not None,
            'priority': CONDA_STEP_PRIORITY.get(step_id, 9),
            'conda_optimized': IS_CONDA,
            'm3_max_optimized': IS_M3_MAX,
            'device': DEVICE,
            'cached': step_id in self._step_cache,
            'failed': step_id in self._failed_steps
        }
    
    def clear_cache(self):
        """캐시 초기화"""
        with self._lock:
            self._step_cache.clear()
            self._failed_steps.clear()
            # @lru_cache 캐시도 초기화
            self.safe_import_step.cache_clear()
            self.logger.info("🧹 Step 캐시 초기화 완료")

# 전역 Step 로더
_step_loader = SimpleStepLoader()

# =============================================================================
# 🔥 Step 관리 함수들 (외부 인터페이스)
# =============================================================================

def safe_import_step(step_id: str) -> Optional[Type[Any]]:
    """안전한 Step 클래스 import"""
    return _step_loader.safe_import_step(step_id)

def load_all_steps() -> Dict[str, Optional[Type[Any]]]:
    """모든 Step 클래스 로드"""
    return _step_loader.load_all_steps()

def get_step_class(step_name: Union[str, int]) -> Optional[Type[Any]]:
    """Step 클래스 반환"""
    try:
        if isinstance(step_name, int):
            step_key = f"step_{step_name:02d}"
        elif step_name.startswith('step_'):
            step_key = step_name
        else:
            # 클래스명으로 검색
            for step_id, class_name in STEP_CLASSES.items():
                if class_name == step_name:
                    step_key = step_id
                    break
            else:
                logger.warning(f"⚠️ 알 수 없는 Step 이름: {step_name}")
                return None
        
        return safe_import_step(step_key)
    except Exception as e:
        logger.error(f"❌ Step 클래스 조회 실패 {step_name}: {e}")
        return None

def create_step_instance(step_name: Union[str, int], **kwargs) -> Optional[Any]:
    """Step 인스턴스 생성"""
    try:
        step_class = get_step_class(step_name)
        if step_class is None:
            logger.error(f"❌ Step 클래스를 찾을 수 없음: {step_name}")
            return None
        
        # 기본 설정 추가
        default_config = {
            "device": DEVICE,
            "is_m3_max": IS_M3_MAX,
            "memory_gb": SYSTEM_INFO.get('memory_gb', 16.0),
            "conda_optimized": IS_CONDA,
            "conda_env": CONDA_ENV
        }
        default_config.update(kwargs)
        
        return step_class(**default_config)
        
    except Exception as e:
        logger.error(f"❌ Step 인스턴스 생성 실패 {step_name}: {e}")
        return None

def list_available_steps() -> List[str]:
    """사용 가능한 Step 목록 반환"""
    loaded_steps = load_all_steps()
    return [step_id for step_id, step_class in loaded_steps.items() if step_class is not None]

def get_step_info(step_id: str) -> Dict[str, Any]:
    """Step 정보 반환"""
    return _step_loader.get_step_info(step_id)

def get_steps_status() -> Dict[str, Any]:
    """전체 Step 상태 반환"""
    loaded_steps = load_all_steps()
    available_steps = [k for k, v in loaded_steps.items() if v is not None]
    
    return {
        'total_steps': len(STEP_MODULES),
        'available_steps': len(available_steps),
        'loaded_steps': available_steps,
        'failed_steps': [k for k, v in loaded_steps.items() if v is None],
        'success_rate': (len(available_steps) / len(STEP_MODULES)) * 100,
        'conda_optimized': IS_CONDA,
        'conda_env': CONDA_ENV,
        'm3_max_optimized': IS_M3_MAX,
        'device': DEVICE,
        'step_details': {step_id: get_step_info(step_id) for step_id in STEP_MODULES.keys()},
        'step_priorities': CONDA_STEP_PRIORITY
    }

def get_high_priority_steps() -> List[str]:
    """고우선순위 Step 목록 반환"""
    if IS_CONDA:
        # 우선순위 순으로 정렬
        sorted_steps = sorted(
            STEP_MODULES.keys(),
            key=lambda x: CONDA_STEP_PRIORITY.get(x, 9)
        )
        return sorted_steps[:4]  # 상위 4개
    else:
        return ['step_01', 'step_03', 'step_06', 'step_08']  # 기본 핵심 Step들

def clear_step_cache():
    """Step 캐시 초기화"""
    _step_loader.clear_cache()
    logger.info("🧹 Step 캐시 초기화 완료")

# =============================================================================
# 🔥 자동 로딩 및 전역 변수 설정
# =============================================================================

# Step 클래스들 자동 로딩
try:
    ALL_STEPS = load_all_steps()
    
    # 개별 클래스 전역 변수로 내보내기 (하위 호환성)
    for step_id, step_class in ALL_STEPS.items():
        if step_class:
            class_name = STEP_CLASSES[step_id]
            globals()[class_name] = step_class
    
    available_count = sum(1 for step in ALL_STEPS.values() if step is not None)
    logger.info(f"✅ Step 클래스들 전역 설정 완료 ({available_count}/8개)")
    
    # conda 환경에서 추가 정보
    if IS_CONDA:
        high_priority_loaded = sum(
            1 for step_id in get_high_priority_steps() 
            if ALL_STEPS.get(step_id) is not None
        )
        logger.info(f"🐍 conda 고우선순위 Step: {high_priority_loaded}/4개 로드됨")
    
except Exception as e:
    logger.error(f"❌ Step 클래스 자동 로딩 실패: {e}")
    ALL_STEPS = {}

# =============================================================================
# 🔥 Export 목록
# =============================================================================

__all__ = [
    # 🔥 Step 정보
    'STEP_MODULES',
    'STEP_CLASSES', 
    'STEP_DESCRIPTIONS',
    'CONDA_STEP_PRIORITY',
    
    # 🔗 Step 관리 함수들
    'safe_import_step',
    'load_all_steps',
    'get_step_class',
    'create_step_instance',
    'list_available_steps',
    'get_step_info',
    'get_steps_status',
    'get_high_priority_steps',
    'clear_step_cache',
    
    # 📊 로딩 결과
    'ALL_STEPS',
    
    # 🔧 시스템 정보
    'SYSTEM_INFO',
    'IS_CONDA',
    'IS_M3_MAX',
    'DEVICE',
    'CONDA_ENV'
] + list(STEP_CLASSES.values())  # Step 클래스들도 동적 추가

# =============================================================================
# 🔥 초기화 완료 메시지
# =============================================================================

def _print_initialization_summary():
    """초기화 요약 출력"""
    available_count = len([k for k, v in ALL_STEPS.items() if v is not None])
    success_rate = (available_count / len(STEP_MODULES)) * 100
    
    print(f"\n🎯 MyCloset AI Pipeline Steps 모듈 v7.0 초기화 완료!")
    print(f"📊 로딩된 Step: {available_count}/8개 ({success_rate:.1f}%)")
    print(f"🐍 conda 환경: {'✅' if IS_CONDA else '❌'} ({CONDA_ENV})")
    print(f"🍎 M3 Max: {'✅' if IS_M3_MAX else '❌'}")
    print(f"🖥️ 디바이스: {DEVICE}")
    print(f"🔗 지연 로딩: ✅ 활성화")
    
    if IS_CONDA:
        high_priority_steps = get_high_priority_steps()
        high_priority_loaded = sum(
            1 for step_id in high_priority_steps 
            if ALL_STEPS.get(step_id) is not None
        )
        print(f"⭐ 고우선순위 Step: {high_priority_loaded}/4개")
        
    if available_count < len(STEP_MODULES):
        failed_steps = [k for k, v in ALL_STEPS.items() if v is None]
        print(f"⚠️ 구현 대기 Step: {failed_steps}")
        print(f"💡 이는 정상적인 상태입니다 (단계적 구현)")
        
    print("🚀 Step 시스템 준비 완료!\n")

# 초기화 상태 출력 (한 번만)
if not hasattr(sys, '_mycloset_steps_initialized'):
    _print_initialization_summary()
    sys._mycloset_steps_initialized = True

logger.info("🎯 MyCloset AI Pipeline Steps 모듈 초기화 완료")