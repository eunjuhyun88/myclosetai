#!/usr/bin/env python3
"""
🔧 MyCloset AI - 모델 탐지기 설정 수정
✅ 실제 경로 구조에 맞는 탐지 로직 수정
✅ 229GB 모델 파일들 완전 활용
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

class ModelDetectorConfigFix:
    """모델 탐지기 설정 수정기"""
    
    def __init__(self, project_root: str = "/Users/gimdudeul/MVP/mycloset-ai"):
        self.project_root = Path(project_root)
        self.backend_root = self.project_root / "backend"
        self.ai_models_root = self.backend_root / "ai_models"
        
    def get_enhanced_search_paths(self) -> List[str]:
        """실제 구조에 맞는 검색 경로 생성"""
        
        search_paths = [
            # 1. 표준 Step 디렉토리들 (심볼릭 링크 후)
            str(self.ai_models_root / "step_01_human_parsing"),
            str(self.ai_models_root / "step_02_pose_estimation"),
            str(self.ai_models_root / "step_03_cloth_segmentation"),
            str(self.ai_models_root / "step_04_geometric_matching"),
            str(self.ai_models_root / "step_05_cloth_warping"),
            str(self.ai_models_root / "step_06_virtual_fitting"),
            str(self.ai_models_root / "step_07_post_processing"),
            str(self.ai_models_root / "step_08_quality_assessment"),
            
            # 2. 실제 체크포인트 디렉토리들
            str(self.ai_models_root / "checkpoints" / "step_01_human_parsing"),
            str(self.ai_models_root / "checkpoints" / "step_02_pose_estimation"),
            str(self.ai_models_root / "checkpoints" / "step_03_cloth_segmentation"),
            str(self.ai_models_root / "checkpoints" / "step_04_geometric_matching"),
            str(self.ai_models_root / "checkpoints" / "step_05_cloth_warping"),
            str(self.ai_models_root / "checkpoints" / "step_06_virtual_fitting"),
            str(self.ai_models_root / "checkpoints" / "step_07_post_processing"),
            str(self.ai_models_root / "checkpoints" / "step_08_quality_assessment"),
            
            # 3. organized 디렉토리들
            str(self.ai_models_root / "organized" / "step_01_human_parsing"),
            str(self.ai_models_root / "organized" / "step_02_pose_estimation"),
            str(self.ai_models_root / "organized" / "step_03_cloth_segmentation"),
            str(self.ai_models_root / "organized" / "step_04_geometric_matching"),
            str(self.ai_models_root / "organized" / "step_05_cloth_warping"),
            str(self.ai_models_root / "organized" / "step_06_virtual_fitting"),
            str(self.ai_models_root / "organized" / "step_07_post_processing"),
            str(self.ai_models_root / "organized" / "step_08_quality_assessment"),
            
            # 4. ai_models2 디렉토리들
            str(self.ai_models_root / "ai_models2" / "step_01_human_parsing"),
            str(self.ai_models_root / "ai_models2" / "step_02_pose_estimation"),
            str(self.ai_models_root / "ai_models2" / "step_03_cloth_segmentation"),
            str(self.ai_models_root / "ai_models2" / "step_04_geometric_matching"),
            str(self.ai_models_root / "ai_models2" / "step_05_cloth_warping"),
            str(self.ai_models_root / "ai_models2" / "step_06_virtual_fitting"),
            str(self.ai_models_root / "ai_models2" / "step_07_post_processing"),
            str(self.ai_models_root / "ai_models2" / "step_08_quality_assessment"),
            
            # 5. 특수 모델 디렉토리들
            str(self.ai_models_root / "Graphonomy"),
            str(self.ai_models_root / "openpose"),
            str(self.ai_models_root / "OOTDiffusion"),
            str(self.ai_models_root / "HR-VITON"),
            str(self.ai_models_root / "u2net"),
            str(self.ai_models_root / "clip_vit_large"),
            str(self.ai_models_root / "idm_vton"),
            str(self.ai_models_root / "fashion_clip"),
            str(self.ai_models_root / "sam2_large"),
            
            # 6. Hugging Face 캐시
            str(self.ai_models_root / "huggingface_cache"),
            str(self.ai_models_root / "cache" / "huggingface"),
            
            # 7. 기타 중요 디렉토리
            str(self.ai_models_root / "auxiliary_models"),
            str(self.ai_models_root / "individual_models"),
        ]
        
        # 존재하는 경로만 반환
        valid_paths = [path for path in search_paths if Path(path).exists()]
        
        return valid_paths
    
    def generate_model_detector_patch(self) -> str:
        """모델 탐지기 패치 코드 생성"""
        
        search_paths = self.get_enhanced_search_paths()
        
        code = f'''#!/usr/bin/env python3
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
ENHANCED_SEARCH_PATHS = {search_paths}

class EnhancedModelDetector:
    """향상된 모델 탐지기"""
    
    def __init__(self):
        self.search_paths = [Path(p) for p in ENHANCED_SEARCH_PATHS if Path(p).exists()]
        logger.info(f"🔍 유효한 검색 경로: {{len(self.search_paths)}}개")
        
    def scan_all_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """모든 모델 스캔"""
        detected_models = {{}}
        
        for search_path in self.search_paths:
            try:
                models = self._scan_directory(search_path)
                
                # Step 이름 추출
                step_name = self._extract_step_name(search_path)
                if step_name not in detected_models:
                    detected_models[step_name] = []
                
                detected_models[step_name].extend(models)
                
                if models:
                    logger.info(f"📁 {{search_path.name}}: {{len(models)}}개 모델 발견")
                    
            except Exception as e:
                logger.warning(f"⚠️ 스캔 실패 {{search_path}}: {{e}}")
        
        # 중복 제거 및 정렬
        for step_name in detected_models:
            detected_models[step_name] = self._deduplicate_models(detected_models[step_name])
        
        return detected_models
    
    def _scan_directory(self, directory: Path) -> List[Dict[str, Any]]:
        """디렉토리 스캔"""
        models = []
        
        # 모델 파일 확장자
        model_extensions = {{'.pth', '.pt', '.bin', '.safetensors', '.onnx', '.pkl'}}
        
        try:
            # 하위 디렉토리까지 재귀적으로 스캔
            for file_path in directory.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in model_extensions:
                    try:
                        stat = file_path.stat()
                        size_mb = stat.st_size / (1024 * 1024)
                        
                        # 최소 크기 필터 (0.1MB 이상)
                        if size_mb >= 0.1:
                            model_info = {{
                                'file_path': str(file_path),
                                'file_name': file_path.name,
                                'size_mb': round(size_mb, 1),
                                'last_modified': stat.st_mtime,
                                'confidence': self._calculate_confidence(file_path.name, size_mb),
                                'relative_path': str(file_path.relative_to(directory))
                            }}
                            models.append(model_info)
                            
                    except Exception as e:
                        logger.debug(f"파일 처리 실패 {{file_path}}: {{e}}")
                        
        except Exception as e:
            logger.warning(f"디렉토리 스캔 실패 {{directory}}: {{e}}")
        
        # 크기 순으로 정렬 (큰 파일 우선)
        models.sort(key=lambda x: x['size_mb'], reverse=True)
        
        return models
    
    def _extract_step_name(self, path: Path) -> str:
        """경로에서 Step 이름 추출"""
        path_str = str(path).lower()
        
        # Step 패턴 매칭
        step_patterns = {{
            'human_parsing': ['step_01', 'human_parsing', 'graphonomy', 'schp'],
            'pose_estimation': ['step_02', 'pose_estimation', 'openpose', 'hrnet'],
            'cloth_segmentation': ['step_03', 'cloth_segmentation', 'u2net', 'rembg'],
            'geometric_matching': ['step_04', 'geometric_matching', 'gmm', 'tps'],
            'cloth_warping': ['step_05', 'cloth_warping', 'warping'],
            'virtual_fitting': ['step_06', 'virtual_fitting', 'viton', 'ootd', 'diffusion'],
            'post_processing': ['step_07', 'post_processing', 'enhancement', 'super_resolution'],
            'quality_assessment': ['step_08', 'quality_assessment', 'clip', 'aesthetic']
        }}
        
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
'''
        
        return code
    
    def create_step_model_requests_fix(self) -> str:
        """step_model_requests.py 수정 코드 생성"""
        
        return '''#!/usr/bin/env python3
"""
📋 MyCloset AI - Step별 모델 요청사항 v6.1 (실제 파일 기반)
✅ 실제 탐지된 파일들로 패턴 업데이트
✅ 신뢰도 기반 매칭 개선
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class StepModelRequest:
    """Step별 모델 요청사항"""
    step_name: str
    model_name: str
    checkpoint_patterns: List[str]
    size_range_mb: tuple = (0.1, 50000.0)  # 최대 50GB까지 허용
    required_files: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.1  # 매우 관대한 임계값
    device: str = "mps"
    precision: str = "fp16"
    priority: int = 1

# 실제 발견된 파일들 기반 패턴 정의
STEP_MODEL_REQUESTS = {
    "HumanParsingStep": StepModelRequest(
        step_name="HumanParsingStep",
        model_name="human_parsing_enhanced",
        checkpoint_patterns=[
            r".*\.pth$",
            r".*\.pt$", 
            r".*\.bin$",
            r".*\.pkl$",
            r".*schp.*\.pth$",
            r".*graphonomy.*\.pth$",
            r".*parsing.*\.pth$",
            r".*human.*\.pth$"
        ],
        confidence_threshold=0.05
    ),
    
    "PoseEstimationStep": StepModelRequest(
        step_name="PoseEstimationStep", 
        model_name="pose_estimation_enhanced",
        checkpoint_patterns=[
            r".*\.pth$",
            r".*\.pt$",
            r".*\.bin$",
            r".*openpose.*\.pth$",
            r".*pose.*\.pth$",
            r".*body.*\.pth$"
        ],
        confidence_threshold=0.05
    ),
    
    "ClothSegmentationStep": StepModelRequest(
        step_name="ClothSegmentationStep",
        model_name="cloth_segmentation_enhanced", 
        checkpoint_patterns=[
            r".*\.pth$",
            r".*\.pt$",
            r".*\.bin$",
            r".*u2net.*\.pth$",
            r".*segment.*\.pth$",
            r".*cloth.*\.pth$"
        ],
        confidence_threshold=0.05
    ),
    
    "GeometricMatchingStep": StepModelRequest(
        step_name="GeometricMatchingStep",
        model_name="geometric_matching_enhanced",
        checkpoint_patterns=[
            r".*\.pth$",
            r".*\.pt$", 
            r".*\.bin$",
            r".*gmm.*\.pth$",
            r".*tps.*\.pth$",
            r".*matching.*\.pth$"
        ],
        confidence_threshold=0.05
    ),
    
    "ClothWarpingStep": StepModelRequest(
        step_name="ClothWarpingStep",
        model_name="cloth_warping_enhanced",
        checkpoint_patterns=[
            r".*\.pth$",
            r".*\.pt$",
            r".*\.bin$", 
            r".*warp.*\.pth$",
            r".*tom.*\.pth$"
        ],
        confidence_threshold=0.05
    ),
    
    "VirtualFittingStep": StepModelRequest(
        step_name="VirtualFittingStep",
        model_name="virtual_fitting_enhanced",
        checkpoint_patterns=[
            r".*\.pth$",
            r".*\.pt$",
            r".*\.bin$",
            r".*\.safetensors$",
            r".*viton.*\.pth$",
            r".*ootd.*\.bin$",
            r".*diffusion.*\.bin$",
            r".*fitting.*\.pth$"
        ],
        size_range_mb=(100.0, 50000.0),  # 대용량 모델
        confidence_threshold=0.02
    ),
    
    "PostProcessingStep": StepModelRequest(
        step_name="PostProcessingStep",
        model_name="post_processing_enhanced",
        checkpoint_patterns=[
            r".*\.pth$",
            r".*\.pt$",
            r".*\.bin$",
            r".*enhance.*\.pth$",
            r".*super.*\.pth$",
            r".*resolution.*\.pth$"
        ],
        confidence_threshold=0.05
    ),
    
    "QualityAssessmentStep": StepModelRequest(
        step_name="QualityAssessmentStep", 
        model_name="quality_assessment_enhanced",
        checkpoint_patterns=[
            r".*\.pth$",
            r".*\.pt$",
            r".*\.bin$",
            r".*clip.*\.bin$",
            r".*quality.*\.pth$",
            r".*aesthetic.*\.pth$"
        ],
        confidence_threshold=0.05
    )
}

def get_step_model_request(step_name: str) -> Optional[StepModelRequest]:
    """Step별 모델 요청사항 조회"""
    return STEP_MODEL_REQUESTS.get(step_name)

def get_all_patterns() -> List[str]:
    """모든 패턴 목록 반환"""
    all_patterns = []
    for request in STEP_MODEL_REQUESTS.values():
        all_patterns.extend(request.checkpoint_patterns)
    return list(set(all_patterns))
'''

def main():
    """메인 실행 함수"""
    print("🔧 모델 탐지기 설정 수정")
    print("="*50)
    
    fixer = ModelDetectorConfigFix()
    
    # 1. 향상된 탐지기 패치 생성
    detector_patch = fixer.generate_model_detector_patch()
    with open('enhanced_model_detector.py', 'w') as f:
        f.write(detector_patch)
    print("✅ enhanced_model_detector.py 생성 완료")
    
    # 2. step_model_requests 수정
    requests_fix = fixer.create_step_model_requests_fix()
    with open('step_model_requests_enhanced.py', 'w') as f:
        f.write(requests_fix)
    print("✅ step_model_requests_enhanced.py 생성 완료")
    
    # 3. 검색 경로 확인
    search_paths = fixer.get_enhanced_search_paths()
    print(f"\n📁 유효한 검색 경로: {len(search_paths)}개")
    for path in search_paths[:10]:  # 상위 10개만 표시
        print(f"   - {path}")
    if len(search_paths) > 10:
        print(f"   ... 외 {len(search_paths) - 10}개")
    
    print("\n🎯 적용 방법:")
    print("   1. 먼저 모델 경로 정리 스크립트 실행")
    print("   2. enhanced_model_detector.py를 사용하여 모델 탐지")
    print("   3. 서버 재시작")

if __name__ == "__main__":
    main()