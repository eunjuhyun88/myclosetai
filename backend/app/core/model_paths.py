# backend/app/core/model_paths.py
"""
MyCloset AI 통합 모델 경로 관리 (backend/backend 문제 완전 해결)
================================================================================
✅ backend/backend 중복 생성 문제 완전 해결
✅ 안전한 경로 계산 및 Path 객체 처리
✅ 실제 프로젝트 구조 기반 정확한 경로 매핑
✅ conda 환경 + M3 Max 최적화 호환
✅ 폴백 메커니즘 강화
"""

from pathlib import Path
from typing import Dict, Optional, List, Union
import logging
import os

logger = logging.getLogger(__name__)

# =============================================================================
# 🔥 1. 안전한 프로젝트 경로 계산 (backend/backend 문제 해결)
# =============================================================================

def _get_safe_project_root() -> Path:
    """
    안전한 프로젝트 루트 디렉토리 계산
    backend/backend 중복 생성 문제 완전 해결
    """
    current_file = Path(__file__).absolute()
    logger.debug(f"🔍 현재 파일: {current_file}")
    
    # 현재 파일: backend/app/core/model_paths.py
    current = current_file.parent  # core/
    
    # backend/ 디렉토리를 찾을 때까지 상위로 이동
    for level in range(10):  # 최대 10단계
        logger.debug(f"  레벨 {level}: {current} (이름: {current.name})")
        
        if current.name == 'backend':
            # backend/ 디렉토리를 찾았으면 그 부모가 프로젝트 루트
            project_root = current.parent
            logger.info(f"✅ 프로젝트 루트 발견: {project_root}")
            return project_root
            
        elif current.name == 'mycloset-ai':
            # 직접 프로젝트 루트에 도달한 경우
            logger.info(f"✅ 프로젝트 루트 직접 발견: {current}")
            return current
            
        elif current.parent == current:
            # 파일시스템 루트에 도달
            logger.warning("⚠️ 파일시스템 루트에 도달 - 폴백 사용")
            break
            
        current = current.parent
    
    # 폴백: 하드코딩된 상대 경로 사용
    fallback_root = current_file.parents[3]  # backend/app/core에서 3단계 위
    logger.warning(f"⚠️ 프로젝트 루트 폴백 사용: {fallback_root}")
    return fallback_root

def _get_safe_backend_root() -> Path:
    """
    안전한 백엔드 루트 디렉토리 계산
    """
    current_file = Path(__file__).absolute()
    current = current_file.parent  # core/
    
    # backend/ 디렉토리를 찾을 때까지 상위로 이동
    for level in range(10):
        if current.name == 'backend':
            logger.info(f"✅ 백엔드 루트 발견: {current}")
            return current
        elif current.parent == current:
            break
        current = current.parent
    
    # 폴백: 프로젝트 루트에서 backend 추가
    project_root = _get_safe_project_root()
    backend_root = project_root / 'backend'
    logger.warning(f"⚠️ 백엔드 루트 폴백 사용: {backend_root}")
    return backend_root

def _get_safe_ai_models_dir() -> Path:
    """
    안전한 AI 모델 디렉토리 계산
    backend/backend 생성 방지
    """
    backend_root = _get_safe_backend_root()
    ai_models_dir = backend_root / "ai_models"
    
    # 🔥 backend/backend 패턴 검사 및 수정
    ai_models_str = str(ai_models_dir)
    if "backend/backend" in ai_models_str:
        corrected_path = Path(ai_models_str.replace("backend/backend", "backend"))
        logger.warning(f"⚠️ backend/backend 패턴 감지 및 수정: {ai_models_dir} → {corrected_path}")
        ai_models_dir = corrected_path
    
    logger.info(f"📁 AI 모델 디렉토리: {ai_models_dir}")
    return ai_models_dir

# 🔥 안전한 경로 계산 실행
PROJECT_ROOT = _get_safe_project_root()
BACKEND_ROOT = _get_safe_backend_root()
AI_MODELS_DIR = _get_safe_ai_models_dir()

# 디렉토리 존재 확인 및 안전한 생성
def _ensure_directories_exist():
    """필요한 디렉토리들을 안전하게 생성"""
    try:
        # AI 모델 디렉토리 생성
        if not AI_MODELS_DIR.exists():
            AI_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 AI 모델 디렉토리 생성: {AI_MODELS_DIR}")
        else:
            logger.debug(f"📁 AI 모델 디렉토리 존재 확인: {AI_MODELS_DIR}")
            
        # 기본 체크포인트 디렉토리들 생성
        checkpoints_dir = AI_MODELS_DIR / "checkpoints"
        if not checkpoints_dir.exists():
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 체크포인트 디렉토리 생성: {checkpoints_dir}")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ 디렉토리 생성 실패: {e}")
        return False

# 초기화 실행
_ensure_directories_exist()

# =============================================================================
# 🔥 2. 실제 모델 경로 매핑 (backend/backend 방지)
# =============================================================================

# 🎯 8단계 AI 파이프라인 모델 경로
STEP_MODEL_PATHS = {
    # Step 1: Human Parsing
    "human_parsing_schp_atr": AI_MODELS_DIR / "Self-Correction-Human-Parsing" / "exp-schp-201908261155-atr.pth",
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
# 🔥 3. 안전한 경로 처리 함수들 (backend/backend 방지)
# =============================================================================

def safe_path_conversion(path_input: Union[str, Path, None]) -> Path:
    """
    안전한 Path 객체 변환
    backend/backend 패턴 자동 수정 포함
    """
    try:
        if path_input is None:
            return Path(".")
            
        if isinstance(path_input, str):
            # 🔥 backend/backend 패턴 자동 수정
            if "backend/backend" in path_input:
                corrected_path = path_input.replace("backend/backend", "backend")
                logger.info(f"✅ backend/backend 자동 수정: {path_input} → {corrected_path}")
                path_input = corrected_path
            return Path(path_input)
            
        elif isinstance(path_input, Path):
            # 🔥 Path 객체에서도 backend/backend 패턴 검사
            path_str = str(path_input)
            if "backend/backend" in path_str:
                corrected_path = Path(path_str.replace("backend/backend", "backend"))
                logger.info(f"✅ Path 객체 backend/backend 자동 수정: {path_input} → {corrected_path}")
                return corrected_path
            return path_input
            
        else:
            # 예상치 못한 타입인 경우 문자열로 변환 시도
            converted = str(path_input)
            if "backend/backend" in converted:
                converted = converted.replace("backend/backend", "backend")
            return Path(converted)
            
    except Exception as e:
        logger.warning(f"⚠️ 경로 변환 실패: {path_input} - {e}")
        return Path(".")

def get_model_path(model_name: str) -> Optional[Path]:
    """
    모델 경로 가져오기
    backend/backend 자동 수정 포함
    """
    try:
        if model_name in ALL_MODEL_PATHS:
            raw_path = ALL_MODEL_PATHS[model_name]
            safe_path = safe_path_conversion(raw_path)
            logger.debug(f"📝 모델 경로 반환: {model_name} → {safe_path}")
            return safe_path
        
        # 🔥 동적 매칭 시도 (파일명 기반)
        for key, path in ALL_MODEL_PATHS.items():
            if model_name.lower() in key.lower():
                safe_path = safe_path_conversion(path)
                logger.debug(f"🔍 동적 매칭: {model_name} → {key} → {safe_path}")
                return safe_path
        
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
                safe_path = safe_path_conversion(fallback_path)
                logger.debug(f"🔄 폴백 경로: {model_name} → {safe_path}")
                return safe_path
        
        logger.warning(f"⚠️ 모델 경로를 찾을 수 없음: {model_name}")
        return None
        
    except Exception as e:
        logger.error(f"❌ 모델 경로 조회 실패: {model_name} - {e}")
        return None

def is_model_available(model_name: str) -> bool:
    """모델 사용 가능 여부 확인"""
    try:
        path = get_model_path(model_name)
        if path is None:
            return False
        
        # 안전한 Path 객체 변환 및 존재 확인
        path_obj = safe_path_conversion(path)
        exists = path_obj.exists()
        logger.debug(f"📊 모델 가용성: {model_name} → {exists}")
        return exists
        
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
                    logger.debug(f"✅ 사용 가능한 모델: {model_name}")
            except Exception as e:
                logger.debug(f"❌ 모델 확인 실패: {model_name} - {e}")
                continue
        
        logger.info(f"📊 사용 가능한 모델: {len(available)}개")
        return available
        
    except Exception as e:
        logger.error(f"❌ 모델 목록 조회 실패: {e}")
        return {}

# =============================================================================
# 🔥 4. backend/backend 문제 진단 및 수정 함수
# =============================================================================

def diagnose_backend_duplication() -> Dict[str, any]:
    """backend/backend 중복 문제 진단"""
    diagnosis = {
        "has_duplication": False,
        "affected_paths": [],
        "current_structure": {},
        "recommendations": []
    }
    
    try:
        # 현재 경로 구조 분석
        current_dir = Path.cwd()
        diagnosis["current_working_directory"] = str(current_dir)
        
        # backend/backend 패턴 검사
        for model_name, path in ALL_MODEL_PATHS.items():
            path_str = str(path)
            if "backend/backend" in path_str:
                diagnosis["has_duplication"] = True
                diagnosis["affected_paths"].append({
                    "model": model_name,
                    "problematic_path": path_str,
                    "corrected_path": path_str.replace("backend/backend", "backend")
                })
        
        # 실제 파일시스템 검사
        if current_dir.name == "backend":
            backend_subdir = current_dir / "backend"
            if backend_subdir.exists():
                diagnosis["filesystem_duplication"] = True
                diagnosis["recommendations"].append("rm -rf backend/backend 실행 필요")
            else:
                diagnosis["filesystem_duplication"] = False
        
        # 권장사항 생성
        if diagnosis["has_duplication"]:
            diagnosis["recommendations"].extend([
                "model_paths.py의 경로 계산 로직 수정 필요",
                "ModelLoader의 폴백 디렉토리 설정 검토 필요",
                "경로 변환 함수들에 backend/backend 수정 로직 추가"
            ])
        
        return diagnosis
        
    except Exception as e:
        logger.error(f"❌ backend/backend 진단 실패: {e}")
        diagnosis["error"] = str(e)
        return diagnosis

def fix_backend_duplication() -> bool:
    """backend/backend 중복 문제 자동 수정"""
    try:
        logger.info("🔧 backend/backend 중복 문제 자동 수정 시작...")
        
        # 1. 현재 작업 디렉토리에서 중복 제거
        current_dir = Path.cwd()
        if current_dir.name == "backend":
            duplicate_backend = current_dir / "backend"
            if duplicate_backend.exists():
                import shutil
                shutil.rmtree(duplicate_backend)
                logger.info(f"✅ 중복 디렉토리 제거: {duplicate_backend}")
        
        # 2. 모든 모델 경로 재계산 및 수정
        global ALL_MODEL_PATHS
        corrected_paths = {}
        
        for model_name, path in ALL_MODEL_PATHS.items():
            corrected_path = safe_path_conversion(path)
            corrected_paths[model_name] = corrected_path
        
        ALL_MODEL_PATHS = corrected_paths
        
        # 3. 디렉토리 재생성
        _ensure_directories_exist()
        
        logger.info("✅ backend/backend 중복 문제 자동 수정 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ backend/backend 자동 수정 실패: {e}")
        return False

# =============================================================================
# 🔥 5. 추가 유틸리티 및 하위 호환성
# =============================================================================

# (이전 코드와 동일하게 유지)
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

class ModelPaths:
    """모델 경로 빠른 접근 클래스 (backend/backend 문제 해결 포함)"""
    
    @property
    def ai_models_dir(self) -> Path:
        return safe_path_conversion(AI_MODELS_DIR)
    
    @property
    def project_root(self) -> Path:
        return safe_path_conversion(PROJECT_ROOT)
    
    @property
    def backend_root(self) -> Path:
        return safe_path_conversion(BACKEND_ROOT)
    
    def diagnose_duplication(self) -> Dict[str, any]:
        return diagnose_backend_duplication()
    
    def fix_duplication(self) -> bool:
        return fix_backend_duplication()

# 전역 인스턴스
model_paths = ModelPaths()

# =============================================================================
# 🔥 6. 모듈 초기화 및 자동 진단
# =============================================================================

def initialize_model_paths() -> bool:
    """모델 경로 초기화 및 backend/backend 문제 자동 해결"""
    try:
        logger.info("🔄 모델 경로 초기화 및 문제 진단 시작...")
        
        # 1. backend/backend 문제 진단
        diagnosis = diagnose_backend_duplication()
        
        if diagnosis.get("has_duplication", False):
            logger.warning("⚠️ backend/backend 중복 문제 감지됨")
            logger.info("🔧 자동 수정 시도...")
            
            if fix_backend_duplication():
                logger.info("✅ backend/backend 문제 자동 수정 완료")
            else:
                logger.error("❌ 자동 수정 실패 - 수동 개입 필요")
                return False
        
        # 2. 디렉토리 구조 확인 및 생성
        success = _ensure_directories_exist()
        
        if success:
            available_models = get_all_available_models()
            logger.info(f"✅ 모델 경로 초기화 완료: {len(available_models)}개 모델 발견")
            return True
        else:
            logger.error("❌ 디렉토리 생성 실패")
            return False
        
    except Exception as e:
        logger.error(f"❌ 모델 경로 초기화 실패: {e}")
        return False

# 자동 초기화 실행
if __name__ != "__main__":
    try:
        initialize_model_paths()
        logger.info("✅ 통합 모델 경로 시스템 초기화 완료 (backend/backend 문제 해결 포함)")
    except Exception as e:
        logger.warning(f"⚠️ 모델 경로 초기화 실패: {e}")

# =============================================================================
# 🔥 7. 내보내기 목록
# =============================================================================

__all__ = [
    # 핵심 함수들
    'get_model_path',
    'is_model_available', 
    'get_all_available_models',
    'safe_path_conversion',
    
    # 문제 해결 함수들
    'diagnose_backend_duplication',
    'fix_backend_duplication',
    
    # Step별 함수들
    'get_step_models',
    
    # 클래스 및 상수
    'ModelPaths',
    'model_paths',
    'AI_MODELS_DIR',
    'PROJECT_ROOT',
    'BACKEND_ROOT',
    
    # 초기화
    'initialize_model_paths'
]