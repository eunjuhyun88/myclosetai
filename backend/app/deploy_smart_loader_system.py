# backend/app/ai_pipeline/utils/smart_model_mapper.py
"""
🔥 SmartModelPathMapper - 통합 모델 경로 매핑 시스템 v1.0
================================================================================
✅ ModelLoader + AutoDetector 완전 통합
✅ 동적 경로 탐지로 워닝 제거
✅ BaseStepMixin v18.0 완전 호환
✅ 실제 AI 모델 파일 (229GB) 완전 활용
✅ conda 환경 + M3 Max 최적화
================================================================================
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import threading

logger = logging.getLogger(__name__)

@dataclass
class ModelMappingInfo:
    """모델 매핑 정보"""
    name: str
    actual_path: Optional[Path] = None
    step_class: Optional[str] = None
    ai_class: Optional[str] = None
    size_mb: float = 0.0
    priority_score: float = 0.0
    is_loaded: bool = False
    load_method: str = "load_models"
    search_patterns: List[str] = field(default_factory=list)
    fallback_paths: List[str] = field(default_factory=list)

class SmartModelPathMapper:
    """🔥 통합 스마트 모델 경로 매핑 시스템"""
    
    def __init__(self, ai_models_root: Union[str, Path]):
        self.ai_models_root = Path(ai_models_root)
        self.logger = logging.getLogger(f"{__name__}.SmartModelPathMapper")
        
        # 경로 검증 및 수정
        self.ai_models_root = self._validate_and_fix_path(self.ai_models_root)
        
        # 캐시 시스템
        self.path_cache: Dict[str, ModelMappingInfo] = {}
        self.last_scan_time = 0.0
        self.cache_valid_duration = 300  # 5분
        
        # 🔥 실제 파일 구조 기반 통합 매핑 테이블
        self.unified_model_mappings = self._create_unified_mappings()
        
        # 통계
        self.mapping_stats = {
            "total_patterns": len(self.unified_model_mappings),
            "successful_mappings": 0,
            "failed_mappings": 0,
            "cache_hits": 0,
            "dynamic_discoveries": 0
        }
        
        self._lock = threading.RLock()
        
        self.logger.info(f"🧠 SmartModelPathMapper 초기화 완료")
        self.logger.info(f"📁 AI 모델 루트: {self.ai_models_root}")
        self.logger.info(f"🎯 통합 매핑 패턴: {len(self.unified_model_mappings)}개")
    
    def _validate_and_fix_path(self, path: Path) -> Path:
        """경로 검증 및 자동 수정"""
        try:
            if path.exists():
                return path.resolve()
            
                    # backend 중복 패턴 자동 수정
        path_str = str(path)
        backend_pattern = "backend" + "/" + "backend"
        if backend_pattern in path_str:
            corrected = Path(path_str.replace(backend_pattern, "backend"))
                if corrected.exists():
                    self.logger.info(f"✅ 경로 자동 수정: {path} → {corrected}")
                    return corrected.resolve()
            
            # 상대 경로에서 절대 경로 찾기
            current = Path(__file__).resolve()
            for i in range(10):
                potential = current / "ai_models"
                if potential.exists():
                    self.logger.info(f"✅ 동적 경로 발견: {potential}")
                    return potential
                
                if current.name == "backend":
                    backend_ai_models = current / "ai_models"
                    if backend_ai_models.exists():
                        return backend_ai_models
                    
                if current.parent == current:
                    break
                current = current.parent
            
            # 최종 폴백
            self.logger.warning(f"⚠️ 경로를 찾을 수 없음, 생성 시도: {path}")
            path.mkdir(parents=True, exist_ok=True)
            return path.resolve()
            
        except Exception as e:
            self.logger.error(f"❌ 경로 검증 실패: {e}")
            return path
    
    def _create_unified_mappings(self) -> Dict[str, Dict[str, Any]]:
        """🔥 통합 모델 매핑 테이블 생성"""
        return {
            # Step 01: Human Parsing
            "graphonomy": {
                "step_class": "HumanParsingStep",
                "ai_class": "RealGraphonomyModel", 
                "search_patterns": [
                    "graphonomy.pth",
                    "*/graphonomy.pth",
                    "step_01_human_parsing/graphonomy.pth",
                    "Graphonomy/*/graphonomy.pth"
                ],
                "fallback_patterns": [
                    "**/graphonomy*.pth",
                    "**/Graphonomy*.pth"
                ],
                "min_size_mb": 1000,
                "priority": 1
            },
            
            "human_parsing_schp": {
                "step_class": "HumanParsingStep", 
                "ai_class": "RealSCHPModel",
                "search_patterns": [
                    "exp-schp-*-atr.pth",
                    "step_01_human_parsing/exp-schp-*-atr.pth",
                    "Self-Correction-Human-Parsing/exp-schp-*-atr.pth"
                ],
                "fallback_patterns": ["**/exp-schp*atr*.pth"],
                "min_size_mb": 250,
                "priority": 2
            },
            
            # Step 02: Pose Estimation  
            "yolov8": {
                "step_class": "PoseEstimationStep",
                "ai_class": "RealYOLOv8PoseModel",
                "search_patterns": [
                    "yolov8n-pose.pt",
                    "step_02_pose_estimation/yolov8*.pt"
                ],
                "fallback_patterns": ["**/yolov8*pose*.pt"],
                "min_size_mb": 5,
                "priority": 1
            },
            
            "openpose": {
                "step_class": "PoseEstimationStep",
                "ai_class": "RealOpenPoseModel", 
                "search_patterns": [
                    "openpose.pth",
                    "step_02_pose_estimation/openpose.pth",
                    "body_pose_model.pth"
                ],
                "fallback_patterns": ["**/openpose*.pth", "**/body_pose*.pth"],
                "min_size_mb": 90,
                "priority": 2
            },
            
            # Step 03: Cloth Segmentation
            "sam_vit_h": {
                "step_class": "ClothSegmentationStep",
                "ai_class": "RealSAMModel",
                "search_patterns": [
                    "sam_vit_h_4b8939.pth",
                    "step_03_cloth_segmentation/sam_vit_h_4b8939.pth",
                    "step_04_geometric_matching/sam_vit_h_4b8939.pth"
                ],
                "fallback_patterns": ["**/sam_vit_h*.pth"],
                "min_size_mb": 2400,
                "priority": 1
            },
            
            "u2net": {
                "step_class": "ClothSegmentationStep",
                "ai_class": "RealU2NetModel",
                "search_patterns": [
                    "u2net.pth",
                    "step_03_cloth_segmentation/u2net.pth"
                ],
                "fallback_patterns": ["**/u2net*.pth"],
                "min_size_mb": 160,
                "priority": 2
            },
            
            # Step 04: Geometric Matching
            "gmm": {
                "step_class": "GeometricMatchingStep", 
                "ai_class": "RealGMMModel",
                "search_patterns": [
                    "gmm_final.pth",
                    "step_04_geometric_matching/gmm_final.pth"
                ],
                "fallback_patterns": ["**/gmm*.pth"],
                "min_size_mb": 10,
                "priority": 1
            },
            
            # Step 05: Cloth Warping
            "realvis_xl": {
                "step_class": "ClothWarpingStep",
                "ai_class": "RealVisXLModel", 
                "search_patterns": [
                    "RealVisXL_V4.0.safetensors",
                    "step_05_cloth_warping/RealVisXL_V4.0.safetensors"
                ],
                "fallback_patterns": ["**/RealVis*.safetensors", "**/realvis*.safetensors"],
                "min_size_mb": 6500,
                "priority": 1
            },
            
            "vgg16_warping": {
                "step_class": "ClothWarpingStep",
                "ai_class": "RealVGGModel",
                "search_patterns": [
                    "vgg16_warping.pth",
                    "step_05_cloth_warping/vgg16_warping.pth"
                ],
                "fallback_patterns": ["**/vgg16*warp*.pth"],
                "min_size_mb": 500,
                "priority": 2
            },
            
            "vgg19_warping": {
                "step_class": "ClothWarpingStep", 
                "ai_class": "RealVGGModel",
                "search_patterns": [
                    "vgg19_warping.pth",
                    "step_05_cloth_warping/vgg19_warping.pth"
                ],
                "fallback_patterns": ["**/vgg19*warp*.pth"],
                "min_size_mb": 500,
                "priority": 3
            },
            
            "densenet121": {
                "step_class": "ClothWarpingStep",
                "ai_class": "RealDenseNetModel", 
                "search_patterns": [
                    "densenet121_warping.pth",
                    "step_05_cloth_warping/densenet121_warping.pth"
                ],
                "fallback_patterns": ["**/densenet121*.pth"],
                "min_size_mb": 30,
                "priority": 4
            },
            
            # Step 06: Virtual Fitting
            "ootdiffusion": {
                "step_class": "VirtualFittingStep",
                "ai_class": "RealOOTDDiffusionModel",
                "search_patterns": [
                    "diffusion_pytorch_model.bin",
                    "diffusion_pytorch_model.safetensors", 
                    "step_06_virtual_fitting/*/diffusion_pytorch_model.*"
                ],
                "fallback_patterns": ["**/diffusion_pytorch_model.*"],
                "min_size_mb": 3100,
                "priority": 1
            },
            
            # Step 07: Post Processing
            "post_processing_model": {
                "step_class": "PostProcessingStep",
                "ai_class": "RealGFPGANModel",
                "search_patterns": [
                    "sr_model.pth",
                    "step_07_post_processing/sr_model.pth",
                    "GFPGANv1.4.pth",
                    "Real-ESRGAN_x4plus.pth"
                ],
                "fallback_patterns": ["**/GFPGAN*.pth", "**/ESRGAN*.pth", "**/sr_model*.pth"],
                "min_size_mb": 300,
                "priority": 1
            },
            
            # Step 08: Quality Assessment
            "quality_assessment_clip": {
                "step_class": "QualityAssessmentStep",
                "ai_class": "RealCLIPModel",
                "search_patterns": [
                    "open_clip_pytorch_model.bin",
                    "step_08_quality_assessment/open_clip_pytorch_model.bin",
                    "ViT-L-14.pt"
                ],
                "fallback_patterns": ["**/open_clip*.bin", "**/ViT-L-14*.pt"],
                "min_size_mb": 5100,
                "priority": 1
            }
        }
    
    def get_model_path(self, model_name: str, force_refresh: bool = False) -> Optional[ModelMappingInfo]:
        """🔥 모델 경로 가져오기 (캐시 + 동적 탐지)"""
        try:
            with self._lock:
                current_time = time.time()
                
                # 캐시 확인
                if (not force_refresh and 
                    model_name in self.path_cache and 
                    current_time - self.last_scan_time < self.cache_valid_duration):
                    
                    self.mapping_stats["cache_hits"] += 1
                    return self.path_cache[model_name]
                
                # 새로 탐지
                mapping_info = self._discover_model_path(model_name)
                
                if mapping_info and mapping_info.actual_path:
                    self.path_cache[model_name] = mapping_info
                    self.mapping_stats["successful_mappings"] += 1
                    
                    self.logger.info(f"✅ 모델 경로 매핑 성공: {model_name} → {mapping_info.actual_path} ({mapping_info.size_mb:.1f}MB)")
                    return mapping_info
                else:
                    self.mapping_stats["failed_mappings"] += 1
                    self.logger.warning(f"⚠️ 모델 경로 찾기 실패: {model_name}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"❌ 모델 경로 가져오기 실패 {model_name}: {e}")
            return None
    
    def _discover_model_path(self, model_name: str) -> Optional[ModelMappingInfo]:
        """모델 경로 동적 탐지"""
        try:
            # 통합 매핑에서 패턴 가져오기
            if model_name not in self.unified_model_mappings:
                # 부분 매칭 시도
                matching_key = self._find_partial_match(model_name)
                if not matching_key:
                    return self._fallback_discovery(model_name)
                model_name = matching_key
            
            mapping_config = self.unified_model_mappings[model_name]
            
            # 우선순위 검색 패턴
            candidates = []
            
            # 1단계: 정확한 패턴 검색
            for pattern in mapping_config["search_patterns"]:
                found_files = self._search_by_pattern(pattern)
                candidates.extend(found_files)
            
            # 2단계: 폴백 패턴 검색
            if not candidates:
                for pattern in mapping_config.get("fallback_patterns", []):
                    found_files = self._search_by_pattern(pattern)
                    candidates.extend(found_files)
            
            # 3단계: 최적 후보 선택
            if candidates:
                best_candidate = self._select_best_candidate(candidates, mapping_config)
                
                if best_candidate:
                    return ModelMappingInfo(
                        name=model_name,
                        actual_path=best_candidate[0],
                        step_class=mapping_config.get("step_class"),
                        ai_class=mapping_config.get("ai_class"),
                        size_mb=best_candidate[1],
                        priority_score=self._calculate_priority(best_candidate[1], mapping_config),
                        search_patterns=mapping_config["search_patterns"],
                        fallback_paths=mapping_config.get("fallback_patterns", [])
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ 모델 경로 탐지 실패 {model_name}: {e}")
            return None
    
    def _search_by_pattern(self, pattern: str) -> List[Tuple[Path, float]]:
        """패턴으로 파일 검색"""
        try:
            candidates = []
            
            # glob 패턴 검색
            if '*' in pattern:
                for file_path in self.ai_models_root.glob(pattern):
                    if file_path.is_file():
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        candidates.append((file_path, size_mb))
            else:
                # 직접 경로 검색
                direct_path = self.ai_models_root / pattern
                if direct_path.exists() and direct_path.is_file():
                    size_mb = direct_path.stat().st_size / (1024 * 1024) 
                    candidates.append((direct_path, size_mb))
            
            return candidates
            
        except Exception as e:
            self.logger.debug(f"패턴 검색 실패 {pattern}: {e}")
            return []
    
    def _find_partial_match(self, model_name: str) -> Optional[str]:
        """부분 매칭으로 키 찾기"""
        model_lower = model_name.lower()
        
        # 직접 매칭
        for key in self.unified_model_mappings.keys():
            if key.lower() == model_lower:
                return key
        
        # 부분 매칭
        for key in self.unified_model_mappings.keys():
            if key.lower() in model_lower or model_lower in key.lower():
                return key
        
        # 키워드 매칭
        keywords = model_lower.split('_')
        for key in self.unified_model_mappings.keys():
            key_keywords = key.lower().split('_')
            if any(kw in key_keywords for kw in keywords):
                return key
        
        return None
    
    def _select_best_candidate(self, candidates: List[Tuple[Path, float]], config: Dict[str, Any]) -> Optional[Tuple[Path, float]]:
        """최적 후보 선택"""
        if not candidates:
            return None
        
        min_size = config.get("min_size_mb", 0)
        
        # 크기 필터링
        valid_candidates = [(path, size) for path, size in candidates if size >= min_size]
        
        if not valid_candidates:
            # 크기 조건을 만족하지 않아도 가장 큰 것 선택
            valid_candidates = candidates
        
        # 크기 우선순위로 정렬 (큰 것부터)
        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return valid_candidates[0]
    
    def _calculate_priority(self, size_mb: float, config: Dict[str, Any]) -> float:
        """우선순위 점수 계산"""
        score = 0.0
        
        # 크기 기반 점수
        import math
        if size_mb > 0:
            score += math.log10(max(size_mb, 1)) * 100
        
        # 설정 기반 우선순위
        base_priority = config.get("priority", 5)
        score += (6 - base_priority) * 50
        
        # 크기 임계값 보너스
        min_size = config.get("min_size_mb", 0)
        if size_mb >= min_size:
            score += 100
        
        return score
    
    def _fallback_discovery(self, model_name: str) -> Optional[ModelMappingInfo]:
        """폴백 탐지 (키워드 기반)"""
        try:
            self.mapping_stats["dynamic_discoveries"] += 1
            
            keywords = model_name.lower().split('_')
            model_extensions = ['.pth', '.bin', '.safetensors', '.pt']
            
            candidates = []
            
            for ext in model_extensions:
                for file_path in self.ai_models_root.rglob(f"*{ext}"):
                    if file_path.is_file():
                        filename_lower = file_path.name.lower()
                        
                        # 키워드 매칭 점수
                        score = sum(1 for kw in keywords if kw in filename_lower)
                        if score > 0:
                            size_mb = file_path.stat().st_size / (1024 * 1024)
                            if size_mb >= 50:  # 최소 50MB
                                candidates.append((file_path, size_mb, score))
            
            if candidates:
                # 매칭 점수 우선, 크기 차선으로 정렬
                candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
                best = candidates[0]
                
                self.logger.info(f"🔍 동적 탐지 성공: {model_name} → {best[0]} (점수: {best[2]})")
                
                return ModelMappingInfo(
                    name=model_name,
                    actual_path=best[0], 
                    size_mb=best[1],
                    priority_score=best[1] * best[2],
                    ai_class="BaseRealAIModel"
                )
            
            return None
            
        except Exception as e:
            self.logger.debug(f"폴백 탐지 실패: {e}")
            return None
    
    def get_step_models(self, step_class: str) -> List[ModelMappingInfo]:
        """Step별 모델 목록 가져오기"""
        try:
            step_models = []
            
            for model_name, config in self.unified_model_mappings.items():
                if config.get("step_class") == step_class:
                    mapping_info = self.get_model_path(model_name)
                    if mapping_info:
                        step_models.append(mapping_info)
            
            # 우선순위 점수로 정렬
            step_models.sort(key=lambda x: x.priority_score, reverse=True)
            
            return step_models
            
        except Exception as e:
            self.logger.error(f"❌ Step 모델 조회 실패 {step_class}: {e}")
            return []
    
    def refresh_cache(self) -> Dict[str, Any]:
        """캐시 새로고침"""
        try:
            with self._lock:
                old_cache_size = len(self.path_cache)
                self.path_cache.clear()
                self.last_scan_time = 0.0
                
                # 모든 모델 재탐지
                for model_name in self.unified_model_mappings.keys():
                    self.get_model_path(model_name, force_refresh=True)
                
                new_cache_size = len(self.path_cache)
                
                self.logger.info(f"✅ 캐시 새로고침 완료: {old_cache_size} → {new_cache_size}")
                
                return {
                    "cache_refreshed": True,
                    "old_cache_size": old_cache_size,
                    "new_cache_size": new_cache_size,
                    "mapping_stats": self.mapping_stats
                }
                
        except Exception as e:
            self.logger.error(f"❌ 캐시 새로고침 실패: {e}")
            return {"cache_refreshed": False, "error": str(e)}
    
    def get_mapping_statistics(self) -> Dict[str, Any]:
        """매핑 통계 조회"""
        try:
            total_cache_entries = len(self.path_cache)
            successful_mappings = sum(1 for info in self.path_cache.values() if info.actual_path)
            
            return {
                "ai_models_root": str(self.ai_models_root),
                "ai_models_root_exists": self.ai_models_root.exists(),
                "total_patterns": len(self.unified_model_mappings),
                "cache_entries": total_cache_entries,
                "successful_mappings": successful_mappings,
                "success_rate": successful_mappings / max(1, total_cache_entries),
                "last_scan_time": self.last_scan_time,
                "cache_valid": time.time() - self.last_scan_time < self.cache_valid_duration,
                "mapping_stats": self.mapping_stats.copy(),
                "step_classes": list(set(config.get("step_class") for config in self.unified_model_mappings.values())),
                "ai_classes": list(set(config.get("ai_class") for config in self.unified_model_mappings.values()))
            }
            
        except Exception as e:
            self.logger.error(f"❌ 통계 조회 실패: {e}")
            return {"error": str(e)}

# ==============================================
# 🔥 전역 인스턴스 관리
# ==============================================

_global_mapper: Optional[SmartModelPathMapper] = None
_mapper_lock = threading.Lock()

def get_global_smart_mapper(ai_models_root: Optional[Union[str, Path]] = None) -> SmartModelPathMapper:
    """전역 SmartModelPathMapper 인스턴스"""
    global _global_mapper
    
    with _mapper_lock:
        if _global_mapper is None or ai_models_root is not None:
            if ai_models_root is None:
                # 자동 경로 탐지
                current_file = Path(__file__)
                backend_root = current_file.parents[3]  # backend/
                ai_models_root = backend_root / "ai_models"
            
            _global_mapper = SmartModelPathMapper(ai_models_root)
            
        return _global_mapper

def resolve_model_path(model_name: str) -> Optional[Path]:
    """모델 경로 해결 (간편 함수)"""
    mapper = get_global_smart_mapper()
    mapping_info = mapper.get_model_path(model_name)
    return mapping_info.actual_path if mapping_info else None

def get_step_model_paths(step_class: str) -> Dict[str, Path]:
    """Step별 모델 경로들"""
    mapper = get_global_smart_mapper()
    step_models = mapper.get_step_models(step_class)
    
    return {
        model.name: model.actual_path 
        for model in step_models 
        if model.actual_path
    }

# Export
__all__ = [
    'SmartModelPathMapper',
    'ModelMappingInfo', 
    'get_global_smart_mapper',
    'resolve_model_path',
    'get_step_model_paths'
]

logger.info("✅ SmartModelPathMapper 통합 시스템 로드 완료")
logger.info("🔥 동적 경로 탐지로 워닝 제거")
logger.info("🎯 ModelLoader + AutoDetector 완전 통합")