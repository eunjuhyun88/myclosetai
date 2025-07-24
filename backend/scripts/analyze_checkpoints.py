#!/usr/bin/env python3
"""
🔍 checkpoints 디렉토리 상세 분석
80GB의 실제 체크포인트들을 분석하고 사용 가능한 모델들 식별
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CheckpointAnalyzer:
    """체크포인트 상세 분석기"""
    
    def __init__(self, checkpoints_dir: str = "ai_models/checkpoints"):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.analyzed_models = {}
        
        logger.info(f"📁 체크포인트 디렉토리: {self.checkpoints_dir.absolute()}")
        
        if not self.checkpoints_dir.exists():
            logger.error(f"❌ 체크포인트 디렉토리를 찾을 수 없습니다: {self.checkpoints_dir}")
            raise FileNotFoundError(f"체크포인트 디렉토리 없음: {self.checkpoints_dir}")
    
    def analyze_all_checkpoints(self) -> Dict[str, Any]:
        """모든 체크포인트 상세 분석"""
        logger.info("🔍 체크포인트 상세 분석 시작...")
        logger.info(f"💾 총 용량: 80.3GB - 매우 큰 규모!")
        
        # 각 서브디렉토리 분석
        for subdir in sorted(self.checkpoints_dir.iterdir()):
            if subdir.is_dir():
                self._analyze_model_directory(subdir)
        
        # 루트 레벨 파일들 분석
        self._analyze_root_files()
        
        # 분석 결과 요약
        self._show_analysis_summary()
        
        return self.analyzed_models
    
    def _analyze_model_directory(self, model_dir: Path):
        """개별 모델 디렉토리 분석"""
        model_name = model_dir.name
        logger.info(f"\n📦 분석 중: {model_name}")
        
        # 파일 통계
        all_files = list(model_dir.rglob("*"))
        model_files = [f for f in all_files if f.is_file()]
        
        # 파일 타입별 분류
        file_types = {}
        checkpoint_files = []
        config_files = []
        
        for file_path in model_files:
            ext = file_path.suffix.lower()
            size_mb = file_path.stat().st_size / (1024 * 1024)
            
            # 파일 타입 카운트
            file_types[ext] = file_types.get(ext, 0) + 1
            
            # 체크포인트 파일 식별
            if ext in ['.pth', '.pt', '.bin', '.ckpt', '.pkl', '.caffemodel']:
                checkpoint_files.append({
                    'name': file_path.name,
                    'path': str(file_path.relative_to(model_dir)),
                    'size_mb': round(size_mb, 1),
                    'type': ext
                })
            
            # 설정 파일 식별
            elif ext in ['.json', '.yaml', '.yml', '.txt'] or 'config' in file_path.name.lower():
                config_files.append({
                    'name': file_path.name,
                    'path': str(file_path.relative_to(model_dir)),
                    'size_mb': round(size_mb, 1)
                })
        
        # 총 크기 계산
        total_size_mb = sum(f.stat().st_size for f in model_files) / (1024 * 1024)
        
        # 모델 타입 추정
        model_type = self._estimate_model_type(model_name, checkpoint_files, config_files)
        
        # 사용 가능성 판단
        is_ready = len(checkpoint_files) > 0 and total_size_mb > 1  # 1MB 이상
        
        self.analyzed_models[model_name] = {
            'name': model_name,
            'path': str(model_dir),
            'type': model_type,
            'ready': is_ready,
            'total_files': len(model_files),
            'total_size_mb': round(total_size_mb, 1),
            'file_types': file_types,
            'checkpoints': checkpoint_files,
            'configs': config_files,
            'step': self._map_to_pipeline_step(model_type),
            'priority': self._get_priority(model_type, total_size_mb)
        }
        
        # 로그 출력
        status = "✅" if is_ready else "⚠️"
        logger.info(f"   {status} {model_name}: {len(checkpoint_files)}개 체크포인트, {total_size_mb:.1f}MB")
        if checkpoint_files:
            for ckpt in checkpoint_files[:3]:  # 상위 3개만 표시
                logger.info(f"      - {ckpt['name']} ({ckpt['size_mb']}MB)")
            if len(checkpoint_files) > 3:
                logger.info(f"      ... 및 {len(checkpoint_files) - 3}개 더")
    
    def _analyze_root_files(self):
        """루트 레벨 체크포인트 파일들 분석"""
        root_files = []
        
        for file_path in self.checkpoints_dir.glob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in ['.pth', '.pt', '.bin', '.ckpt', '.pkl']:
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    root_files.append({
                        'name': file_path.name,
                        'size_mb': round(size_mb, 1),
                        'type': ext
                    })
        
        if root_files:
            logger.info(f"\n📄 루트 레벨 체크포인트 파일들:")
            self.analyzed_models['_root_files'] = {
                'name': 'Root Level Files',
                'type': 'mixed',
                'ready': True,
                'files': root_files,
                'total_size_mb': sum(f['size_mb'] for f in root_files),
                'step': 'auxiliary',
                'priority': 99
            }
            
            for file_info in root_files:
                logger.info(f"   📄 {file_info['name']} ({file_info['size_mb']}MB)")
    
    def _estimate_model_type(self, model_name: str, checkpoints: List, configs: List) -> str:
        """모델 타입 추정"""
        name_lower = model_name.lower()
        
        # 이름 기반 추정
        if any(keyword in name_lower for keyword in ['ootd', 'diffusion', 'stable']):
            return 'diffusion'
        elif any(keyword in name_lower for keyword in ['viton', 'tryon', 'fitting']):
            return 'virtual_tryon'
        elif any(keyword in name_lower for keyword in ['parsing', 'segmentation', 'human']):
            return 'human_parsing'
        elif any(keyword in name_lower for keyword in ['pose', 'openpose', 'keypoint']):
            return 'pose_estimation'
        elif any(keyword in name_lower for keyword in ['u2net', 'background', 'removal']):
            return 'cloth_segmentation'
        elif any(keyword in name_lower for keyword in ['gmm', 'geometric', 'matching']):
            return 'geometric_matching'
        elif any(keyword in name_lower for keyword in ['cloth', 'garment', 'warping']):
            return 'cloth_warping'
        elif any(keyword in name_lower for keyword in ['detectron', 'rcnn', 'detection']):
            return 'detection'
        elif any(keyword in name_lower for keyword in ['clip', 'vit', 'text']):
            return 'text_image'
        else:
            # 파일 크기로 추정
            total_size = sum(c['size_mb'] for c in checkpoints)
            if total_size > 3000:  # 3GB 이상
                return 'diffusion'
            elif total_size > 500:  # 500MB 이상
                return 'virtual_tryon'
            elif total_size > 100:  # 100MB 이상
                return 'human_parsing'
            else:
                return 'auxiliary'
    
    def _map_to_pipeline_step(self, model_type: str) -> str:
        """모델 타입을 파이프라인 단계로 매핑"""
        mapping = {
            'human_parsing': 'step_01_human_parsing',
            'pose_estimation': 'step_02_pose_estimation',
            'cloth_segmentation': 'step_03_cloth_segmentation',
            'geometric_matching': 'step_04_geometric_matching',
            'cloth_warping': 'step_05_cloth_warping',
            'virtual_tryon': 'step_06_virtual_fitting',
            'diffusion': 'step_06_virtual_fitting',
            'detection': 'auxiliary',
            'text_image': 'auxiliary'
        }
        return mapping.get(model_type, 'auxiliary')
    
    def _get_priority(self, model_type: str, size_mb: float) -> int:
        """모델 우선순위 계산"""
        base_priority = {
            'diffusion': 1,
            'virtual_tryon': 2,
            'human_parsing': 3,
            'pose_estimation': 4,
            'cloth_segmentation': 5,
            'geometric_matching': 6,
            'cloth_warping': 7,
            'detection': 8,
            'text_image': 9
        }.get(model_type, 99)
        
        # 크기가 클수록 우선순위 높음 (더 완성도 높은 모델로 간주)
        if size_mb > 5000:  # 5GB 이상
            return base_priority
        elif size_mb > 1000:  # 1GB 이상
            return base_priority + 1
        elif size_mb > 100:  # 100MB 이상
            return base_priority + 2
        else:
            return base_priority + 3
    
    def _show_analysis_summary(self):
        """분석 결과 요약"""
        logger.info(f"\n📊 체크포인트 분석 결과 요약:")
        logger.info("=" * 60)
        
        total_models = len(self.analyzed_models)
        ready_models = sum(1 for model in self.analyzed_models.values() if model.get('ready', False))
        total_size = sum(model.get('total_size_mb', 0) for model in self.analyzed_models.values())
        
        logger.info(f"📦 분석된 모델/디렉토리: {total_models}개")
        logger.info(f"✅ 사용 가능한 모델: {ready_models}개")
        logger.info(f"💾 총 크기: {total_size:.1f} MB ({total_size/1024:.1f} GB)")
        
        # 타입별 분류
        type_summary = {}
        for model_info in self.analyzed_models.values():
            model_type = model_info.get('type', 'unknown')
            if model_type not in type_summary:
                type_summary[model_type] = {'count': 0, 'size_mb': 0, 'ready': 0}
            type_summary[model_type]['count'] += 1
            type_summary[model_type]['size_mb'] += model_info.get('total_size_mb', 0)
            if model_info.get('ready', False):
                type_summary[model_type]['ready'] += 1
        
        logger.info(f"\n📋 타입별 분류:")
        for model_type, stats in type_summary.items():
            logger.info(f"   {model_type}: {stats['ready']}/{stats['count']}개 사용가능, {stats['size_mb']:.1f}MB")
        
        # 우선순위별 정렬
        logger.info(f"\n🎯 우선순위별 사용 가능한 모델들:")
        ready_models = [(name, info) for name, info in self.analyzed_models.items() 
                       if info.get('ready', False)]
        ready_models.sort(key=lambda x: x[1].get('priority', 99))
        
        for name, info in ready_models[:10]:  # 상위 10개만
            step = info.get('step', 'auxiliary')
            size = info.get('total_size_mb', 0)
            priority = info.get('priority', 99)
            logger.info(f"   {priority:2d}. {name} ({size:.1f}MB) - {step}")
    
    def create_optimized_model_config(self):
        """최적화된 모델 설정 파일 생성"""
        logger.info("📝 최적화된 모델 설정 파일 생성 중...")
        
        # 사용 가능한 모델들만 필터링
        ready_models = {name: info for name, info in self.analyzed_models.items() 
                       if info.get('ready', False)}
        
        # 단계별 최고 우선순위 모델 선택
        step_best_models = {}
        for model_name, model_info in ready_models.items():
            step = model_info.get('step', 'auxiliary')
            priority = model_info.get('priority', 99)
            
            if step not in step_best_models or priority < step_best_models[step]['priority']:
                step_best_models[step] = {
                    'model_name': model_name,
                    'priority': priority,
                    'info': model_info
                }
        
        # Python 설정 파일 생성
        config_content = '''# app/core/optimized_model_paths.py
"""
최적화된 AI 모델 경로 설정 - 체크포인트 분석 기반
실제 사용 가능한 체크포인트들로만 구성
"""

from pathlib import Path
from typing import Dict, Optional, List, Any

# 기본 경로
AI_MODELS_ROOT = Path(__file__).parent.parent.parent / "ai_models"
CHECKPOINTS_ROOT = AI_MODELS_ROOT / "checkpoints"

# 분석된 체크포인트 모델들
ANALYZED_MODELS = {
'''
        
        for model_name, model_info in ready_models.items():
            if model_name.startswith('_'):  # 특수 항목 제외
                continue
            config_content += f'''    "{model_name}": {{
        "name": "{model_info['name']}",
        "type": "{model_info['type']}",
        "step": "{model_info['step']}",
        "path": CHECKPOINTS_ROOT / "{model_name}",
        "ready": {model_info['ready']},
        "size_mb": {model_info['total_size_mb']},
        "priority": {model_info['priority']},
        "checkpoints": {model_info['checkpoints'][:3]},  # 상위 3개만
        "total_checkpoints": {len(model_info['checkpoints'])}
    }},
'''
        
        config_content += '''}

# 단계별 최적 모델 매핑
STEP_OPTIMAL_MODELS = {
'''
        
        for step, best_model in step_best_models.items():
            config_content += f'''    "{step}": "{best_model['model_name']}",
'''
        
        config_content += '''}

def get_optimal_model_for_step(step: str) -> Optional[str]:
    """단계별 최적 모델 반환"""
    return STEP_OPTIMAL_MODELS.get(step)

def get_model_checkpoints(model_name: str) -> List[Dict]:
    """모델의 체크포인트 목록 반환"""
    if model_name in ANALYZED_MODELS:
        return ANALYZED_MODELS[model_name]["checkpoints"]
    return []

def get_largest_checkpoint(model_name: str) -> Optional[str]:
    """모델의 가장 큰 체크포인트 반환 (보통 메인 모델)"""
    checkpoints = get_model_checkpoints(model_name)
    if not checkpoints:
        return None
    
    largest = max(checkpoints, key=lambda x: x['size_mb'])
    return largest['path']

def get_ready_models_by_type(model_type: str) -> List[str]:
    """타입별 사용 가능한 모델들"""
    return [name for name, info in ANALYZED_MODELS.items() 
            if info["type"] == model_type and info["ready"]]

def get_diffusion_models() -> List[str]:
    """Diffusion 모델들 (OOTD 등)"""
    return get_ready_models_by_type("diffusion")

def get_virtual_tryon_models() -> List[str]:
    """가상 피팅 모델들 (HR-VITON 등)"""
    return get_ready_models_by_type("virtual_tryon")

def get_human_parsing_models() -> List[str]:
    """인체 파싱 모델들"""
    return get_ready_models_by_type("human_parsing")

def get_model_info(model_name: str) -> Optional[Dict]:
    """모델 상세 정보 반환"""
    return ANALYZED_MODELS.get(model_name)

def list_all_ready_models() -> Dict[str, Dict]:
    """모든 사용 가능한 모델 정보"""
    return ANALYZED_MODELS.copy()

# 빠른 접근 함수들
def get_best_diffusion_model() -> Optional[str]:
    """최고 성능 Diffusion 모델"""
    return get_optimal_model_for_step("step_06_virtual_fitting")

def get_best_human_parsing_model() -> Optional[str]:
    """최고 성능 인체 파싱 모델"""  
    return get_optimal_model_for_step("step_01_human_parsing")

def get_model_path(model_name: str) -> Optional[Path]:
    """모델 디렉토리 경로 반환"""
    if model_name in ANALYZED_MODELS:
        return ANALYZED_MODELS[model_name]["path"]
    return None

def get_checkpoint_path(model_name: str, checkpoint_name: Optional[str] = None) -> Optional[Path]:
    """특정 체크포인트 파일 경로 반환"""
    model_path = get_model_path(model_name)
    if not model_path:
        return None
    
    if checkpoint_name:
        return model_path / checkpoint_name
    else:
        # 가장 큰 체크포인트 반환
        largest_ckpt = get_largest_checkpoint(model_name)
        return model_path / largest_ckpt if largest_ckpt else None

# 사용 통계
ANALYSIS_STATS = {
    "total_models": len(ANALYZED_MODELS),
    "total_size_gb": sum(info["size_mb"] for info in ANALYZED_MODELS.values()) / 1024,
    "models_by_step": {step: len([m for m in ANALYZED_MODELS.values() if m["step"] == step]) 
                      for step in set(info["step"] for info in ANALYZED_MODELS.values())},
    "largest_model": max(ANALYZED_MODELS.items(), key=lambda x: x[1]["size_mb"])[0] if ANALYZED_MODELS else None
}
'''
        
        # 파일 저장
        config_path = Path("app/core/optimized_model_paths.py")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        logger.info(f"✅ 최적화된 Python 설정 파일 생성: {config_path}")
        
        # ModelLoader 연동 파일도 생성
        self._create_modelloader_integration()
    
    def _create_modelloader_integration(self):
        """ModelLoader 완전 연동 파일 생성"""
        logger.info("🔧 ModelLoader 완전 연동 파일 생성 중...")
        
        integration_content = '''# app/ai_pipeline/utils/checkpoint_model_loader.py
"""
체크포인트 분석 기반 ModelLoader 완전 연동
실제 다운로드된 80GB 체크포인트들 활용
"""

from app.ai_pipeline.utils.model_loader import ModelLoader, ModelConfig, ModelType
from app.core.optimized_model_paths import (
    ANALYZED_MODELS, get_optimal_model_for_step, 
    get_checkpoint_path, get_largest_checkpoint
)
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CheckpointModelLoader(ModelLoader):
    """체크포인트 분석 기반 확장 ModelLoader"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._register_analyzed_models()
    
    def _register_analyzed_models(self):
        """분석된 체크포인트 모델들 자동 등록"""
        logger.info("📦 분석된 체크포인트 모델들 등록 중...")
        
        registered_count = 0
        
        for model_name, model_info in ANALYZED_MODELS.items():
            if not model_info["ready"]:
                continue
                
            try:
                # ModelType 매핑
                model_type = self._map_to_model_type(model_info["type"])
                if not model_type:
                    continue
                
                # 가장 큰 체크포인트 경로
                main_checkpoint = get_largest_checkpoint(model_name)
                checkpoint_path = get_checkpoint_path(model_name, main_checkpoint) if main_checkpoint else None
                
                # 모델 설정 생성
                model_config = ModelConfig(
                    name=model_info["name"],
                    model_type=model_type,
                    model_class=self._get_model_class(model_info["type"]),
                    checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
                    input_size=(512, 512),
                    device=self.device
                )
                
                # 모델 등록
                self.register_model(model_name, model_config)
                registered_count += 1
                
                logger.info(f"   ✅ {model_name}: {model_info['name']}")
                
            except Exception as e:
                logger.warning(f"   ⚠️ {model_name} 등록 실패: {e}")
        
        logger.info(f"📦 총 {registered_count}개 체크포인트 모델 등록 완료")
    
    def _map_to_model_type(self, analysis_type: str) -> Optional[ModelType]:
        """분석 타입을 ModelType으로 매핑"""
        mapping = {
            'diffusion': ModelType.DIFFUSION,
            'virtual_tryon': ModelType.VIRTUAL_FITTING,
            'human_parsing': ModelType.HUMAN_PARSING,
            'pose_estimation': ModelType.POSE_ESTIMATION,
            'cloth_segmentation': ModelType.CLOTH_SEGMENTATION,
            'geometric_matching': ModelType.GEOMETRIC_MATCHING,
            'cloth_warping': ModelType.CLOTH_WARPING,
            'detection': ModelType.SEGMENTATION,
            'text_image': ModelType.DIFFUSION
        }
        return mapping.get(analysis_type)
    
    def _get_model_class(self, analysis_type: str) -> str:
        """분석 타입에서 모델 클래스명 추출"""
        mapping = {
            'diffusion': 'StableDiffusionPipeline',
            'virtual_tryon': 'HRVITONModel',
            'human_parsing': 'GraphonomyModel',
            'pose_estimation': 'OpenPoseModel',
            'cloth_segmentation': 'U2NetModel',
            'geometric_matching': 'GeometricMatchingModel',
            'cloth_warping': 'HRVITONModel',
            'detection': 'DetectronModel',
            'text_image': 'CLIPModel'
        }
        return mapping.get(analysis_type, 'BaseModel')
    
    async def load_optimal_model_for_step(self, step: str, **kwargs):
        """단계별 최적 모델 로드"""
        optimal_model = get_optimal_model_for_step(step)
        if not optimal_model:
            logger.warning(f"⚠️ {step}에 대한 최적 모델을 찾을 수 없음")
            return None
        
        logger.info(f"🎯 {step} 최적 모델 로드: {optimal_model}")
        return await self.load_model(optimal_model, **kwargs)

# 전역 체크포인트 모델 로더
_global_checkpoint_loader: Optional[CheckpointModelLoader] = None

def get_checkpoint_model_loader(**kwargs) -> CheckpointModelLoader:
    """전역 체크포인트 모델 로더 반환"""
    global _global_checkpoint_loader
    if _global_checkpoint_loader is None:
        _global_checkpoint_loader = CheckpointModelLoader(**kwargs)
    return _global_checkpoint_loader

async def load_best_model_for_step(step: str, **kwargs):
    """단계별 최고 성능 모델 로드"""
    loader = get_checkpoint_model_loader()
    return await loader.load_optimal_model_for_step(step, **kwargs)

# 빠른 접근 함수들
async def load_best_diffusion_model(**kwargs):
    """최고 성능 Diffusion 모델 로드"""
    return await load_best_model_for_step("step_06_virtual_fitting", **kwargs)

async def load_best_human_parsing_model(**kwargs):
    """최고 성능 인체 파싱 모델 로드"""
    return await load_best_model_for_step("step_01_human_parsing", **kwargs)

async def load_best_pose_model(**kwargs):
    """최고 성능 포즈 추정 모델 로드"""
    return await load_best_model_for_step("step_02_pose_estimation", **kwargs)
'''
        
        # 파일 저장
        integration_path = Path("app/ai_pipeline/utils/checkpoint_model_loader.py")
        integration_path.parent.mkdir(parents=True, exist_ok=True)
        with open(integration_path, 'w', encoding='utf-8') as f:
            f.write(integration_content)
        
        logger.info(f"✅ ModelLoader 완전 연동 파일 생성: {integration_path}")
    
    def create_analysis_report(self):
        """상세 분석 보고서 생성"""
        logger.info("📋 상세 분석 보고서 생성 중...")
        
        report = {
            "analysis_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_size_gb": sum(model.get('total_size_mb', 0) for model in self.analyzed_models.values()) / 1024,
            "analyzed_models": self.analyzed_models,
            "summary": {
                "total_models": len(self.analyzed_models),
                "ready_models": sum(1 for model in self.analyzed_models.values() if model.get('ready', False)),
                "largest_model": max(self.analyzed_models.items(), 
                                   key=lambda x: x[1].get('total_size_mb', 0))[0] if self.analyzed_models else None,
                "models_by_type": {}
            }
        }
        
        # 타입별 통계
        for model_info in self.analyzed_models.values():
            model_type = model_info.get('type', 'unknown')
            if model_type not in report["summary"]["models_by_type"]:
                report["summary"]["models_by_type"][model_type] = {
                    "count": 0, "ready": 0, "total_size_mb": 0
                }
            
            report["summary"]["models_by_type"][model_type]["count"] += 1
            report["summary"]["models_by_type"][model_type]["total_size_mb"] += model_info.get('total_size_mb', 0)
            if model_info.get('ready', False):
                report["summary"]["models_by_type"][model_type]["ready"] += 1
        
        # 보고서 저장
        report_path = self.checkpoints_dir / "checkpoint_analysis_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 상세 분석 보고서 생성: {report_path}")

def main():
    """메인 함수"""
    print("🔍 MyCloset AI - 체크포인트 상세 분석")
    print("=" * 50)
    
    try:
        analyzer = CheckpointAnalyzer("ai_models/checkpoints")
        
        # 체크포인트 분석
        analyzed_models = analyzer.analyze_all_checkpoints()
        
        if not analyzed_models:
            logger.warning("⚠️ 분석할 체크포인트를 찾을 수 없습니다.")
            return False
        
        # 최적화된 설정 파일 생성
        analyzer.create_optimized_model_config()
        analyzer.create_analysis_report()
        
        print(f"\n🎉 체크포인트 분석 완료!")
        print(f"📊 총 {len(analyzed_models)}개 항목 분석")
        ready_count = sum(1 for m in analyzed_models.values() if m.get('ready', False))
        print(f"✅ 사용 가능한 모델: {ready_count}개")
        
        print(f"\n📝 생성된 파일들:")
        print(f"   - app/core/optimized_model_paths.py (최적화된 경로 설정)")
        print(f"   - app/ai_pipeline/utils/checkpoint_model_loader.py (ModelLoader 연동)")
        print(f"   - ai_models/checkpoints/checkpoint_analysis_report.json (상세 보고서)")
        
        print(f"\n🚀 사용 방법:")
        print(f"   from app.ai_pipeline.utils.checkpoint_model_loader import load_best_diffusion_model")
        print(f"   model = await load_best_diffusion_model()")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 분석 실패: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)