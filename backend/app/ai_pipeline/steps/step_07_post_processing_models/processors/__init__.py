#!/usr/bin/env python3
"""
🔥 MyCloset AI - Post Processing Processors
==========================================

🎯 후처리 프로세서들
✅ 배치 처리기
✅ 품질 향상기
✅ 아티팩트 제거기
✅ 해상도 향상기
✅ 색상 보정기
✅ 최종 출력 최적화기
"""

import logging

# 로거 설정
logger = logging.getLogger(__name__)

try:
    from .batch_processor import PostProcessingBatchProcessor
    from .quality_enhancer import PostProcessingQualityEnhancer
    from .artifact_remover import PostProcessingArtifactRemover
    from .resolution_enhancer import PostProcessingResolutionEnhancer
    from .color_corrector import PostProcessingColorCorrector
    from .final_output_optimizer import PostProcessingFinalOutputOptimizer
    
    __all__ = [
        "PostProcessingBatchProcessor",
        "PostProcessingQualityEnhancer",
        "PostProcessingArtifactRemover",
        "PostProcessingResolutionEnhancer",
        "PostProcessingColorCorrector",
        "PostProcessingFinalOutputOptimizer"
    ]
    
except ImportError as e:
    logger.error(f"프로세서 모듈 로드 실패: {e}")
    raise ImportError(f"프로세서 모듈을 로드할 수 없습니다: {e}")

logger.info("✅ Post Processing 프로세서 모듈 로드 완료")
