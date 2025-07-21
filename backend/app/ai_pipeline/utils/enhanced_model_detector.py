#!/usr/bin/env python3
"""
🔧 MyCloset AI - 모델 탐지기 패치 v2.0
✅ 실제 파일 구조에 맞는 경로 설정
✅ 229GB 모델 파일 완전 활용
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# 실제 구조 기반 검색 경로
ENHANCED_SEARCH_PATHS = ['/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/step_01_human_parsing', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/step_02_pose_estimation', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/step_03_cloth_segmentation', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/step_04_geometric_matching', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/step_05_cloth_warping', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/step_06_virtual_fitting', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/step_07_post_processing', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/step_08_quality_assessment', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/checkpoints/step_01_human_parsing', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/checkpoints/step_02_pose_estimation', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/checkpoints/step_03_cloth_segmentation', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/checkpoints/step_04_geometric_matching', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/checkpoints/step_05_cloth_warping', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/checkpoints/step_06_virtual_fitting', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/checkpoints/step_07_post_processing', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/checkpoints/step_08_quality_assessment', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/organized/step_01_human_parsing', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/organized/step_02_pose_estimation', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/organized/step_03_cloth_segmentation', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/organized/step_04_geometric_matching', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/organized/step_05_cloth_warping', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/organized/step_06_virtual_fitting', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/organized/step_07_post_processing', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/organized/step_08_quality_assessment', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/ai_models2/step_01_human_parsing', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/ai_models2/step_02_pose_estimation', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/ai_models2/step_03_cloth_segmentation', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/ai_models2/step_04_geometric_matching', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/ai_models2/step_05_cloth_warping', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/ai_models2/step_06_virtual_fitting', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/ai_models2/step_07_post_processing', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/ai_models2/step_08_quality_assessment', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/Graphonomy', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/openpose', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/OOTDiffusion', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/HR-VITON', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/u2net', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/clip_vit_large', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/idm_vton', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/fashion_clip', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/sam2_large', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/huggingface_cache', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/cache/huggingface', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/auxiliary_models', '/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/individual_models']

class EnhancedModelDetector:
    """향상된 모델 탐지기"""
    
    def __init__(self):
        self.search_paths = [Path(p) for p in ENHANCED_SEARCH_PATHS if Path(p).exists()]
        logger.info(f"🔍 유효한 검색 경로: {len(self.search_paths)}개")
        
    def scan_all_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """모든 모델 스캔"""
        detected_models = {}
        
        for search_path in self.search_paths:
            try:
                models = self._scan_directory(search_path)
                
                # Step 이름 추출
                step_name = self._extract_step_name(search_path)
                if step_name not in detected_models:
                    detected_models[step_name] = []
                
                detected_models[step_name].extend(models)
                
                if models:
                    logger.info(f"📁 {search_path.name}: {len(models)}개 모델 발견")
                    
            except Exception as e:
                logger.warning(f"⚠️ 스캔 실패 {search_path}: {e}")
        
        # 중복 제거 및 정렬
        for step_name in detected_models:
            detected_models[step_name] = self._deduplicate_models(detected_models[step_name])
        
        return detected_models
    
    def _scan_directory(self, directory: Path) -> List[Dict[str, Any]]:
        """디렉토리 스캔"""
        models = []
        
        # 모델 파일 확장자
        model_extensions = {'.pth', '.pt', '.bin', '.safetensors', '.onnx', '.pkl'}
        
        try:
            # 하위 디렉토리까지 재귀적으로 스캔
            for file_path in directory.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in model_extensions:
                    try:
                        stat = file_path.stat()
                        size_mb = stat.st_size / (1024 * 1024)
                        
                        # 최소 크기 필터 (0.1MB 이상)
                        if size_mb >= 0.1:
                            model_info = {
                                'file_path': str(file_path),
                                'file_name': file_path.name,
                                'size_mb': round(size_mb, 1),
                                'last_modified': stat.st_mtime,
                                'confidence': self._calculate_confidence(file_path.name, size_mb),
                                'relative_path': str(file_path.relative_to(directory))
                            }
                            models.append(model_info)
                            
                    except Exception as e:
                        logger.debug(f"파일 처리 실패 {file_path}: {e}")
                        
        except Exception as e:
            logger.warning(f"디렉토리 스캔 실패 {directory}: {e}")
        
        # 크기 순으로 정렬 (큰 파일 우선)
        models.sort(key=lambda x: x['size_mb'], reverse=True)
        
        return models
    
    def _extract_step_name(self, path: Path) -> str:
        """경로에서 Step 이름 추출"""
        path_str = str(path).lower()
        
        # Step 패턴 매칭
        step_patterns = {
            'human_parsing': ['step_01', 'human_parsing', 'graphonomy', 'schp'],
            'pose_estimation': ['step_02', 'pose_estimation', 'openpose', 'hrnet'],
            'cloth_segmentation': ['step_03', 'cloth_segmentation', 'u2net', 'rembg'],
            'geometric_matching': ['step_04', 'geometric_matching', 'gmm', 'tps'],
            'cloth_warping': ['step_05', 'cloth_warping', 'warping'],
            'virtual_fitting': ['step_06', 'virtual_fitting', 'viton', 'ootd', 'diffusion'],
            'post_processing': ['step_07', 'post_processing', 'enhancement', 'super_resolution'],
            'quality_assessment': ['step_08', 'quality_assessment', 'clip', 'aesthetic']
        }
        
        for step_name, keywords in step_patterns.items():
            if any(keyword in path_str for keyword in keywords):
                return step_name
        
        return 'unknown'
    
    def _calculate_confidence(self, filename: str, size_mb: float) -> float:
        """신뢰도 계산"""
        confidence = 0.0
        filename_lower = filename.lower()
        
        # 파일명 기반 점수
        high_confidence_keywords = ['schp', 'openpose', 'u2net', 'viton', 'ootd', 'clip']
        if any(keyword in filename_lower for keyword in high_confidence_keywords):
            confidence += 0.6
        
        medium_confidence_keywords = ['model', 'checkpoint', 'pytorch', 'final', 'best']
        if any(keyword in filename_lower for keyword in medium_confidence_keywords):
            confidence += 0.3
        
        # 크기 기반 점수
        if 10 <= size_mb <= 5000:  # 10MB ~ 5GB (적정 범위)
            confidence += 0.4
        elif size_mb > 5000:  # 5GB 이상 (대용량 모델)
            confidence += 0.2
        elif size_mb >= 1:  # 1MB 이상
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _deduplicate_models(self, models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """중복 모델 제거"""
        seen_names = set()
        unique_models = []
        
        for model in models:
            name = model['file_name']
            if name not in seen_names:
                seen_names.add(name)
                unique_models.append(model)
        
        return unique_models

# 전역 탐지기 인스턴스
enhanced_detector = EnhancedModelDetector()

def scan_enhanced_models() -> Dict[str, List[Dict[str, Any]]]:
    """향상된 모델 스캔 실행"""
    return enhanced_detector.scan_all_models()

def patch_existing_detector(detector_instance):
    """기존 탐지기 패치"""
    if hasattr(detector_instance, 'search_paths'):
        detector_instance.search_paths = enhanced_detector.search_paths
        logger.info("✅ 기존 탐지기 검색 경로 업데이트 완료")
    
    return detector_instance
