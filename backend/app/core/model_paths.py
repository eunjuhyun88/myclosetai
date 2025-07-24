# backend/app/core/model_paths.py
"""
MyCloset AI 통합 모델 경로 관리 (완전 통합 버전)
================================================================================
✅ 모든 중복 파일 통합 (corrected, downloaded, relocated, actual)
✅ 실제 프로젝트 구조 기반 경로 매핑
✅ 안전한 Path 객체 처리 ('str' object has no attribute 'exists' 오류 해결)
✅ 8단계 AI 파이프라인 완전 지원
✅ 폴백 메커니즘 및 오류 처리 강화
✅ conda 환경 + M3 Max 최적화 호환
"""

from pathlib import Path
from typing import Dict, Optional, List, Union
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# 🔥 1. 기본 경로 설정
# =============================================================================

# 프로젝트 루트 및 AI 모델 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent.parent
AI_MODELS_DIR = PROJECT_ROOT / "ai_models"

# =============================================================================
# 🔥 2. 실제 모델 경로 매핑 (프로젝트 구조 기반)
# =============================================================================

# 🎯 8단계 AI 파이프라인 모델 경로
STEP_MODEL_PATHS = {
    # Step 1: Human Parsing
    "human_parsing_schp": AI_MODELS_DIR / "Self-Correction-Human-Parsing" / "exp-schp-201908261155-atr.pth",
    "human_parsing_graphonomy": AI_MODELS_DIR / "Graphonomy" / "graphonomy.pth", 
    "human_parsing_checkpoints": AI_MODELS_DIR / "checkpoints" / "step_01_human_parsing",
    
    # Step 2: Pose Estimation  
    "pose_estimation_openpose": AI_MODELS_DIR / "openpose" / "body_pose_model.pth",
    "pose_estimation_checkpoints": AI_MODELS_DIR / "checkpoints" / "step_02_pose_estimation",
    
    # Step 3: Cloth Segmentation
    "cloth_segmentation_sam": AI_MODELS_DIR / "checkpoints" / "sam" / "sam_vit_h_4b8939.pth",
    "cloth_segmentation_u2net": AI_MODELS_DIR / "checkpoints" / "step_03_cloth_segmentation" / "u2net.pth",
    
    # Step 4: Geometric Matching
    "geometric_matching_gmm": AI_MODELS_DIR / "checkpoints" / "step_04_geometric_matching" / "gmm_final.pth",
    
    # Step 5: Cloth Warping  
    "cloth_warping_tom": AI_MODELS_DIR / "checkpoints" / "step_05_cloth_warping" / "tom_final.pth",
    
    # Step 6: Virtual Fitting (핵심)
    "virtual_fitting_ootd": AI_MODELS_DIR / "checkpoints" / "ootdiffusion",
    "virtual_fitting_hrviton": AI_MODELS_DIR / "HR-VITON",
    "virtual_fitting_vitonhd": AI_MODELS_DIR / "VITON-HD",
    
    # Step 7: Post Processing
    "post_processing_esrgan": AI_MODELS_DIR / "checkpoints" / "step_07_post_processing" / "esrgan.pth",
    
    # Step 8: Quality Assessment
    "quality_assessment_clip": AI_MODELS_DIR / "checkpoints" / "clip-vit-large-patch14"
}

# 🔥 추가 보조 모델 경로
ADDITIONAL_MODEL_PATHS = {
    "stable_diffusion": AI_MODELS_DIR / "checkpoints" / "stable_diffusion",
    "clip_vit_base": AI_MODELS_DIR / "checkpoints" / "clip-vit-base-patch32",
    "controlnet_openpose": AI_MODELS_DIR / "checkpoints" / "controlnet_openpose"
}

# 통합 모델 경로 딕셔너리
ALL_MODEL_PATHS = {**STEP_MODEL_PATHS, **ADDITIONAL_MODEL_PATHS}

# =============================================================================
# 🔥 3. 안전한 경로 처리 함수들
# =============================================================================

def safe_path_conversion(path_input: Union[str, Path]) -> Path:
    """안전한 Path 객체 변환 ('str' object has no attribute 'exists' 오류 해결)"""
    try:
        if isinstance(path_input, str):
            return Path(path_input)
        elif isinstance(path_input, Path):
            return path_input
        else:
            # 예상치 못한 타입인 경우 문자열로 변환 시도
            return Path(str(path_input))
    except Exception as e:
        logger.warning(f"⚠️ 경로 변환 실패: {path_input} - {e}")
        return Path(".")

def get_model_path(model_name: str) -> Optional[Path]:
    """모델 경로 가져오기 (안전한 처리)"""
    try:
        if model_name in ALL_MODEL_PATHS:
            raw_path = ALL_MODEL_PATHS[model_name]
            return safe_path_conversion(raw_path)
        
        # 🔥 동적 매칭 시도 (파일명 기반)
        for key, path in ALL_MODEL_PATHS.items():
            if model_name.lower() in key.lower():
                return safe_path_conversion(path)
        
        # 🔥 폴백: 단계별 디렉토리에서 찾기
        step_mapping = {
            "human_parsing": "step_01_human_parsing",
            "pose_estimation": "step_02_pose_estimation", 
            "cloth_segmentation": "step_03_cloth_segmentation",
            "geometric_matching": "step_04_geometric_matching",
            "cloth_warping": "step_05_cloth_warping",
            "virtual_fitting": "step_06_virtual_fitting",
            "post_processing": "step_07_post_processing",
            "quality_assessment": "step_08_quality_assessment"
        }
        
        for key, step_dir in step_mapping.items():
            if key in model_name.lower():
                fallback_path = AI_MODELS_DIR / "checkpoints" / step_dir
                return safe_path_conversion(fallback_path)
        
        logger.warning(f"⚠️ 모델 경로를 찾을 수 없음: {model_name}")
        return None
        
    except Exception as e:
        logger.error(f"❌ 모델 경로 조회 실패: {model_name} - {e}")
        return None

def is_model_available(model_name: str) -> bool:
    """모델 사용 가능 여부 확인 (안전한 처리)"""
    try:
        path = get_model_path(model_name)
        if path is None:
            return False
        
        # 🔥 안전한 Path 객체 변환 및 존재 확인
        path_obj = safe_path_conversion(path)
        return path_obj.exists()
        
    except Exception as e:
        logger.warning(f"⚠️ 모델 가용성 확인 실패: {model_name} - {e}")
        return False

def get_all_available_models() -> Dict[str, str]:
    """사용 가능한 모든 모델 반환"""
    available = {}
    
    try:
        for model_name, raw_path in ALL_MODEL_PATHS.items():
            try:
                path_obj = safe_path_conversion(raw_path)
                if path_obj.exists():
                    available[model_name] = str(path_obj.absolute())
            except Exception as e:
                logger.debug(f"모델 확인 실패: {model_name} - {e}")
                continue
        
        logger.info(f"✅ 사용 가능한 모델: {len(available)}개")
        return available
        
    except Exception as e:
        logger.error(f"❌ 모델 목록 조회 실패: {e}")
        return {}

# =============================================================================
# 🔥 4. Step별 모델 그룹핑 함수들
# =============================================================================

def get_step_models(step_id: int) -> List[str]:
    """단계별 모델 목록 반환"""
    step_patterns = {
        1: ["human_parsing"],
        2: ["pose_estimation"], 
        3: ["cloth_segmentation"],
        4: ["geometric_matching"],
        5: ["cloth_warping"],
        6: ["virtual_fitting"],
        7: ["post_processing"], 
        8: ["quality_assessment"]
    }
    
    if step_id not in step_patterns:
        return []
    
    pattern = step_patterns[step_id][0]
    return [key for key in ALL_MODEL_PATHS.keys() if pattern in key]

def get_virtual_tryon_models() -> List[str]:
    """가상 피팅 모델 목록 (Step 6)"""
    return [key for key in ALL_MODEL_PATHS.keys() if "virtual_fitting" in key]

def get_human_parsing_models() -> List[str]:
    """Human Parsing 모델 목록 (Step 1)"""
    return [key for key in ALL_MODEL_PATHS.keys() if "human_parsing" in key]

# =============================================================================
# 🔥 5. 특화 경로 함수들
# =============================================================================

def get_primary_ootd_path() -> Optional[Path]:
    """메인 OOTDiffusion 경로 반환"""
    ootd_path = get_model_path("virtual_fitting_ootd")
    if ootd_path and ootd_path.exists():
        return ootd_path
    
    # 폴백: checkpoints/ootdiffusion
    fallback = AI_MODELS_DIR / "checkpoints" / "ootdiffusion"
    return safe_path_conversion(fallback) if fallback.exists() else None

def get_sam_path(model_size: str = "vit_h") -> Optional[Path]:
    """SAM 모델 경로 반환"""
    sam_base = get_model_path("cloth_segmentation_sam")
    if not sam_base or not sam_base.exists():
        sam_base = AI_MODELS_DIR / "checkpoints" / "sam"
    
    if model_size == "vit_h":
        sam_file = sam_base / "sam_vit_h_4b8939.pth"
    elif model_size == "vit_b":
        sam_file = sam_base / "sam_vit_b_01ec64.pth"
    else:
        return None
    
    return safe_path_conversion(sam_file) if sam_file.exists() else None

def get_stable_diffusion_path() -> Optional[Path]:
    """Stable Diffusion 경로 반환"""
    return get_model_path("stable_diffusion")

# =============================================================================
# 🔥 6. 빠른 접근 클래스
# =============================================================================

class ModelPaths:
    """모델 경로 빠른 접근 클래스"""
    
    @property
    def ai_models_dir(self) -> Path:
        return AI_MODELS_DIR
    
    @property
    def ootd_path(self) -> Optional[Path]:
        return get_primary_ootd_path()
    
    @property
    def sam_large(self) -> Optional[Path]:
        return get_sam_path("vit_h")
    
    @property
    def sam_base(self) -> Optional[Path]:
        return get_sam_path("vit_b")
    
    @property
    def stable_diffusion(self) -> Optional[Path]:
        return get_stable_diffusion_path()
    
    def get_step_path(self, step_id: int) -> Optional[Path]:
        """단계별 체크포인트 디렉토리 경로"""
        step_dir = f"step_{step_id:02d}"
        step_names = {
            1: "step_01_human_parsing",
            2: "step_02_pose_estimation",
            3: "step_03_cloth_segmentation", 
            4: "step_04_geometric_matching",
            5: "step_05_cloth_warping",
            6: "step_06_virtual_fitting",
            7: "step_07_post_processing",
            8: "step_08_quality_assessment"
        }
        
        if step_id in step_names:
            path = AI_MODELS_DIR / "checkpoints" / step_names[step_id]
            return safe_path_conversion(path)
        
        return None

# =============================================================================
# 🔥 7. 전역 인스턴스 및 초기화
# =============================================================================

# 전역 모델 경로 인스턴스
model_paths = ModelPaths()

def initialize_model_paths():
    """모델 경로 초기화 및 검증"""
    try:
        logger.info("🔄 모델 경로 초기화 시작...")
        
        # AI 모델 디렉토리 생성
        AI_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # 단계별 체크포인트 디렉토리 생성
        for step_id in range(1, 9):
            step_path = model_paths.get_step_path(step_id)
            if step_path:
                step_path.mkdir(parents=True, exist_ok=True)
        
        available_models = get_all_available_models()
        logger.info(f"✅ 모델 경로 초기화 완료: {len(available_models)}개 모델 발견")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 모델 경로 초기화 실패: {e}")
        return False

# =============================================================================
# 🔥 8. 하위 호환성 함수들 (기존 코드 지원)
# =============================================================================

# 기존 함수명 지원
def get_model_info(model_key: str) -> Optional[Dict]:
    """모델 정보 반환 (하위 호환성)"""
    path = get_model_path(model_key)
    if path and path.exists():
        return {
            "name": model_key,
            "path": path,
            "exists": True,
            "size_mb": path.stat().st_size / (1024 * 1024) if path.is_file() else 0
        }
    return None

def get_models_by_type(model_type: str) -> List[str]:
    """타입별 모델 목록 반환 (하위 호환성)"""
    return [key for key in ALL_MODEL_PATHS.keys() if model_type in key]

# =============================================================================
# 🔥 9. 모듈 초기화
# =============================================================================

# 모듈 로드 시 자동 초기화
try:
    initialize_model_paths()
    logger.info("✅ 통합 모델 경로 시스템 초기화 완료")
except Exception as e:
    logger.warning(f"⚠️ 모델 경로 초기화 실패: {e}")

# 내보낼 함수들
__all__ = [
    # 핵심 함수들
    'get_model_path',
    'is_model_available', 
    'get_all_available_models',
    
    # Step별 함수들
    'get_step_models',
    'get_virtual_tryon_models',
    'get_human_parsing_models',
    
    # 특화 경로 함수들
    'get_primary_ootd_path',
    'get_sam_path',
    'get_stable_diffusion_path',
    
    # 클래스 및 상수
    'ModelPaths',
    'model_paths',
    'AI_MODELS_DIR',
    'PROJECT_ROOT',
    
    # 하위 호환성
    'get_model_info',
    'get_models_by_type',
    
    # 초기화
    'initialize_model_paths'
]