#!/bin/bash
# ================================================================
# MyCloset AI - 2단계: AI 모델 연결 최적화 실행
# ================================================================

set -e

echo "🚀 MyCloset AI - 2단계: AI 모델 연결 최적화"
echo "=================================================================="

# 환경 확인
if [[ "$CONDA_DEFAULT_ENV" != "mycloset-ai" ]]; then
    echo "⚠️ conda 환경 활성화 필요"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate mycloset-ai
fi

echo "✅ 환경: $CONDA_DEFAULT_ENV"
echo "✅ Python: $(python --version)"

# ================================================================
# 2단계-1: 모델 디렉토리 스캔 및 선별
# ================================================================

echo ""
echo "📋 2단계-1: 모델 스캔 및 선별"

python3 -c "
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional

print('🔍 AI 모델 디렉토리 스캔 중...')

# 모델 디렉토리 찾기
model_paths = [
    './ai_models',
    '../ai_models', 
    './backend/ai_models',
    '../backend/ai_models',
    './app/ai_pipeline/models/ai_models'
]

found_models = {}
total_size = 0

for path_str in model_paths:
    path = Path(path_str)
    if path.exists():
        print(f'📁 발견: {path.resolve()}')
        
        # 모델 파일 스캔
        extensions = ['.pth', '.bin', '.safetensors', '.ckpt']
        for ext in extensions:
            for model_file in path.rglob(f'*{ext}'):
                size_mb = model_file.stat().st_size / (1024 * 1024)
                total_size += size_mb
                
                # 8단계 분류
                name_lower = model_file.name.lower()
                if 'human' in name_lower or 'parsing' in name_lower:
                    step = 'step_01_human_parsing'
                elif 'pose' in name_lower or 'openpose' in name_lower:
                    step = 'step_02_pose_estimation'
                elif 'cloth' in name_lower or 'segment' in name_lower or 'u2net' in name_lower:
                    step = 'step_03_cloth_segmentation'
                elif 'geometric' in name_lower or 'gmm' in name_lower:
                    step = 'step_04_geometric_matching'
                elif 'warp' in name_lower or 'tom' in name_lower:
                    step = 'step_05_cloth_warping'
                elif 'diffusion' in name_lower or 'virtual' in name_lower or 'oot' in name_lower:
                    step = 'step_06_virtual_fitting'
                elif 'post' in name_lower or 'esrgan' in name_lower:
                    step = 'step_07_post_processing'
                elif 'clip' in name_lower or 'quality' in name_lower:
                    step = 'step_08_quality_assessment'
                else:
                    step = 'unknown'
                
                found_models[model_file.name] = {
                    'path': str(model_file),
                    'size_mb': round(size_mb, 1),
                    'step': step,
                    'priority': 'high' if size_mb > 500 else 'medium' if size_mb > 100 else 'low'
                }

print(f'\\n📊 스캔 결과:')
print(f'  - 총 모델: {len(found_models)}개')
print(f'  - 총 크기: {total_size/1024:.1f}GB')

# 단계별 분류
step_counts = {}
for model_info in found_models.values():
    step = model_info['step']
    step_counts[step] = step_counts.get(step, 0) + 1

print(f'\\n📋 단계별 분류:')
for step, count in sorted(step_counts.items()):
    print(f'  - {step}: {count}개')

# 핵심 모델 선별 (각 단계별 최대 2개)
essential_models = {}
for step in ['step_01_human_parsing', 'step_02_pose_estimation', 'step_03_cloth_segmentation', 
             'step_04_geometric_matching', 'step_05_cloth_warping', 'step_06_virtual_fitting']:
    
    step_models = [(name, info) for name, info in found_models.items() if info['step'] == step]
    step_models.sort(key=lambda x: x[1]['size_mb'], reverse=True)  # 큰 것부터 (보통 더 좋음)
    
    for i, (name, info) in enumerate(step_models[:2]):  # 최대 2개
        essential_models[name] = info
        print(f'✅ 선별: {step} -> {name} ({info[\"size_mb\"]}MB)')

# 설정 저장
config = {
    'scan_results': {
        'total_models': len(found_models),
        'total_size_gb': round(total_size/1024, 1),
        'step_counts': step_counts
    },
    'selected_models': essential_models,
    'optimization_target': {
        'memory_limit_gb': 32,
        'device': 'mps',
        'precision': 'float32'
    }
}

with open('model_scan_results.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f'\\n✅ 스캔 완료: model_scan_results.json 저장')
print(f'📊 선별된 핵심 모델: {len(essential_models)}개')
"

echo ""
echo "✅ 2단계-1 완료: 모델 스캔 및 선별"

# ================================================================
# 2단계-2: 모델 경로 매핑 해결
# ================================================================

echo ""
echo "📋 2단계-2: 모델 경로 매핑 해결"

python3 -c "
import json
from pathlib import Path

print('🗺️ 모델 경로 매핑 해결 중...')

# 스캔 결과 로드
with open('model_scan_results.json', 'r') as f:
    config = json.load(f)

selected_models = config['selected_models']

# 경로 매핑 생성
path_mappings = {}
duplicates_removed = 0

for model_name, model_info in selected_models.items():
    real_path = Path(model_info['path']).resolve()
    
    if real_path.exists():
        # 상대 경로로 변환 (프로젝트 루트 기준)
        try:
            rel_path = real_path.relative_to(Path.cwd())
            path_mappings[model_name] = str(rel_path)
        except ValueError:
            path_mappings[model_name] = str(real_path)
        
        print(f'✅ 매핑: {model_name} -> {path_mappings[model_name]}')
    else:
        print(f'❌ 경로 없음: {model_name} -> {model_info[\"path\"]}')

# 중복 제거 (같은 파일을 가리키는 모델들)
path_to_models = {}
for model_name, path in path_mappings.items():
    real_path = str(Path(path).resolve())
    if real_path not in path_to_models:
        path_to_models[real_path] = []
    path_to_models[real_path].append(model_name)

# 중복된 경우 가장 간단한 이름만 유지
final_mappings = {}
for real_path, model_names in path_to_models.items():
    if len(model_names) > 1:
        # 가장 간단한 이름 선택
        best_name = min(model_names, key=len)
        final_mappings[best_name] = path_mappings[best_name]
        duplicates_removed += len(model_names) - 1
        print(f'🔄 중복 제거: {model_names} -> {best_name}')
    else:
        model_name = model_names[0]
        final_mappings[model_name] = path_mappings[model_name]

# 업데이트된 설정 저장
config['path_mappings'] = final_mappings
config['optimization_stats'] = {
    'original_models': len(selected_models),
    'final_models': len(final_mappings),
    'duplicates_removed': duplicates_removed
}

with open('model_scan_results.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f'\\n✅ 경로 매핑 완료:')
print(f'  - 최종 모델: {len(final_mappings)}개')
print(f'  - 중복 제거: {duplicates_removed}개')
"

echo ""
echo "✅ 2단계-2 완료: 경로 매핑 해결"

# ================================================================
# 2단계-3: 통합 모델 로더 생성
# ================================================================

echo ""
echo "📋 2단계-3: 통합 모델 로더 생성"

# 통합 모델 로더 클래스 생성
cat > optimized_model_loader.py << 'EOF'
"""
MyCloset AI - 최적화된 통합 모델 로더
M3 Max 128GB 전용 최적화 버전
"""

import json
import torch
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedModelLoader:
    """M3 Max 최적화 모델 로더"""
    
    def __init__(self, config_path: str = "model_scan_results.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.loaded_models = {}
        self.load_times = {}
        
        logger.info(f"🍎 M3 Max 모델 로더 초기화: {self.device}")
        logger.info(f"📦 관리 대상 모델: {len(self.config['path_mappings'])}개")
    
    async def load_model_async(self, model_name: str) -> Optional[torch.nn.Module]:
        """비동기 모델 로딩"""
        if model_name in self.loaded_models:
            logger.info(f"🔄 캐시된 모델 반환: {model_name}")
            return self.loaded_models[model_name]
        
        if model_name not in self.config['path_mappings']:
            logger.error(f"❌ 모델 경로 없음: {model_name}")
            return None
        
        model_path = self.config['path_mappings'][model_name]
        
        try:
            start_time = time.time()
            logger.info(f"🔄 모델 로딩 시작: {model_name}")
            
            # M3 Max MPS 최적화 로딩
            if self.device.type == 'mps':
                # CPU에서 로드 후 MPS로 이동 (안전한 방법)
                model = torch.load(model_path, map_location='cpu', weights_only=False)
                if hasattr(model, 'to'):
                    model = model.to(self.device)
            else:
                model = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # 평가 모드 설정
            if hasattr(model, 'eval'):
                model.eval()
            
            # 메모리 최적화
            if hasattr(model, 'half') and self.device.type == 'mps':
                # MPS에서 float16 사용 (메모리 절약)
                try:
                    model = model.half()
                except:
                    logger.warning(f"⚠️ float16 변환 실패, float32 유지: {model_name}")
            
            load_time = time.time() - start_time
            self.loaded_models[model_name] = model
            self.load_times[model_name] = load_time
            
            logger.info(f"✅ 모델 로딩 완료: {model_name} ({load_time:.2f}초)")
            return model
            
        except Exception as e:
            logger.error(f"❌ 모델 로딩 실패 {model_name}: {e}")
            return None
    
    def unload_model(self, model_name: str):
        """모델 언로드 (메모리 절약)"""
        if model_name in self.loaded_models:
            model = self.loaded_models[model_name]
            if hasattr(model, 'cpu'):
                model.cpu()
            del self.loaded_models[model_name]
            logger.info(f"🗑️ 모델 언로드: {model_name}")
    
    def get_step_model(self, step: str) -> Optional[str]:
        """단계별 권장 모델 반환"""
        for model_name, model_info in self.config['selected_models'].items():
            if model_info['step'] == step:
                return model_name
        return None
    
    def get_memory_usage(self) -> Dict[str, float]:
        """메모리 사용량 조회"""
        if self.device.type == 'mps':
            try:
                total_memory = torch.mps.current_allocated_memory() / (1024**2)  # MB
                return {"mps_memory_mb": total_memory}
            except:
                return {"mps_memory_mb": 0}
        return {"cpu_memory": "N/A"}
    
    def get_statistics(self) -> Dict[str, Any]:
        """로더 통계"""
        return {
            "device": str(self.device),
            "loaded_models": len(self.loaded_models),
            "available_models": len(self.config['path_mappings']),
            "load_times": self.load_times,
            "memory_usage": self.get_memory_usage()
        }

# 전역 로더 인스턴스
_global_loader = None

def get_model_loader() -> OptimizedModelLoader:
    """전역 모델 로더 반환"""
    global _global_loader
    if _global_loader is None:
        _global_loader = OptimizedModelLoader()
    return _global_loader

async def load_essential_models():
    """필수 모델들 사전 로딩"""
    loader = get_model_loader()
    
    essential_steps = [
        'step_01_human_parsing',
        'step_02_pose_estimation', 
        'step_03_cloth_segmentation',
        'step_06_virtual_fitting'
    ]
    
    loaded_count = 0
    for step in essential_steps:
        model_name = loader.get_step_model(step)
        if model_name:
            model = await loader.load_model_async(model_name)
            if model:
                loaded_count += 1
    
    logger.info(f"✅ 필수 모델 로딩 완료: {loaded_count}/{len(essential_steps)}")
    return loaded_count

if __name__ == "__main__":
    # 테스트 실행
    async def test_loader():
        loader = get_model_loader()
        print("📊 모델 로더 통계:")
        stats = loader.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n🔄 필수 모델 로딩 테스트:")
        loaded_count = await load_essential_models()
        
        print(f"\n✅ 테스트 완료: {loaded_count}개 모델 로딩 성공")
    
    asyncio.run(test_loader())
EOF

echo ""
echo "✅ 2단계-3 완료: 통합 모델 로더 생성"

# ================================================================
# 2단계-4: 테스트 및 검증
# ================================================================

echo ""
echo "📋 2단계-4: 모델 로더 테스트"

python3 optimized_model_loader.py

echo ""
echo "✅ 2단계-4 완료: 테스트 및 검증"

# ================================================================
# 2단계 완료 요약
# ================================================================

echo ""
echo "🎉 2단계: AI 모델 연결 최적화 완료!"
echo "=================================================================="

echo ""
echo "📊 완료된 작업:"
echo "  ✅ 모델 스캔 및 선별"
echo "  ✅ 경로 매핑 해결"  
echo "  ✅ 통합 모델 로더 생성"
echo "  ✅ 테스트 및 검증"

echo ""
echo "📁 생성된 파일:"
echo "  - model_scan_results.json: 모델 스캔 결과 및 설정"
echo "  - optimized_model_loader.py: 통합 모델 로더"

echo ""
echo "🚀 다음 단계 (3단계 - 서버 통합):"
echo "  1. 기존 서버에 모델 로더 통합"
echo "  2. API 엔드포인트 연결"
echo "  3. 전체 시스템 테스트"

echo ""
echo "💡 사용법:"
echo "  from optimized_model_loader import get_model_loader"
echo "  loader = get_model_loader()"
echo "  model = await loader.load_model_async('model_name')"

echo ""
echo "📊 모델 현황 확인:"
echo "  cat model_scan_results.json | jq '.scan_results'"