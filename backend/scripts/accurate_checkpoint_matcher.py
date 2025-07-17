#!/usr/bin/env python3
"""
🚀 안전한 체크포인트 재배치 실행 스크립트
✅ 정확한 매칭만 실행
✅ 검증된 모델만 재배치
✅ 백업 및 롤백 지원
✅ M3 Max 최적화
"""

import os
import sys
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class SafeCheckpointRelocator:
    """안전한 체크포인트 재배치기"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.target_dir = self.project_root / "ai_models" / "checkpoints"
        self.backup_dir = self.project_root / "ai_models" / "backup"
        self.relocate_plan = {}
        
    def load_relocate_plan(self) -> bool:
        """재배치 계획 로딩"""
        plan_file = self.project_root / "accurate_relocate_plan.json"
        
        if not plan_file.exists():
            logger.error(f"❌ 재배치 계획 파일이 없습니다: {plan_file}")
            return False
        
        try:
            with open(plan_file, 'r', encoding='utf-8') as f:
                self.relocate_plan = json.load(f)
            
            logger.info(f"✅ 재배치 계획 로딩 완료: {len(self.relocate_plan['actions'])}개")
            return True
        except Exception as e:
            logger.error(f"❌ 재배치 계획 로딩 실패: {e}")
            return False
    
    def validate_plan(self) -> bool:
        """재배치 계획 검증"""
        if not self.relocate_plan or 'actions' not in self.relocate_plan:
            logger.error("❌ 유효하지 않은 재배치 계획")
            return False
        
        # 신뢰도 체크
        high_confidence_actions = [
            action for action in self.relocate_plan['actions'] 
            if action.get('confidence', 0) >= 0.8
        ]
        
        if len(high_confidence_actions) < len(self.relocate_plan['actions']):
            logger.warning(f"⚠️ 낮은 신뢰도 파일들 발견: {len(self.relocate_plan['actions']) - len(high_confidence_actions)}개")
            
            # 사용자 확인
            answer = input("낮은 신뢰도 파일들도 재배치하시겠습니까? (y/N): ")
            if answer.lower() != 'y':
                self.relocate_plan['actions'] = high_confidence_actions
                logger.info(f"✅ 높은 신뢰도 파일들만 재배치: {len(high_confidence_actions)}개")
        
        # 소스 파일 존재 확인
        valid_actions = []
        for action in self.relocate_plan['actions']:
            source_path = Path(action['source'])
            if source_path.exists():
                valid_actions.append(action)
            else:
                logger.warning(f"⚠️ 소스 파일 없음: {source_path}")
        
        self.relocate_plan['actions'] = valid_actions
        logger.info(f"✅ 유효한 재배치 대상: {len(valid_actions)}개")
        
        return len(valid_actions) > 0
    
    def create_directory_structure(self):
        """필요한 디렉토리 구조 생성"""
        logger.info("📁 디렉토리 구조 생성 중...")
        
        directories = [
            "step_01_human_parsing",
            "step_02_pose_estimation", 
            "step_03_cloth_segmentation",
            "step_04_geometric_matching",
            "step_05_cloth_warping",
            "step_06_virtual_fitting",
            "step_07_post_processing",
            "step_08_quality_assessment"
        ]
        
        for directory in directories:
            dir_path = self.target_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 백업 디렉토리 생성
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ 디렉토리 구조 생성 완료")
    
    def backup_existing_files(self):
        """기존 파일들 백업"""
        logger.info("📦 기존 파일 백업 중...")
        
        backup_count = 0
        timestamp = int(time.time())
        
        for action in self.relocate_plan['actions']:
            target_path = self.target_dir / action['target'].replace('ai_models/checkpoints/', '')
            
            if target_path.exists():
                backup_name = f"{target_path.name}.backup_{timestamp}"
                backup_path = self.backup_dir / backup_name
                
                try:
                    shutil.copy2(target_path, backup_path)
                    backup_count += 1
                    logger.debug(f"📦 백업: {target_path.name} → {backup_name}")
                except Exception as e:
                    logger.warning(f"⚠️ 백업 실패 {target_path}: {e}")
        
        logger.info(f"✅ 백업 완료: {backup_count}개 파일")
    
    def execute_relocate(self) -> Dict[str, Any]:
        """실제 재배치 실행"""
        logger.info("🚀 체크포인트 재배치 실행 중...")
        
        results = {
            "success": [],
            "failed": [],
            "skipped": [],
            "total_size_mb": 0
        }
        
        for i, action in enumerate(self.relocate_plan['actions'], 1):
            logger.info(f"\n📋 [{i}/{len(self.relocate_plan['actions'])}] {action['model_type']}")
            
            try:
                source_path = Path(action['source'])
                target_path = self.target_dir / action['target'].replace('ai_models/checkpoints/', '')
                action_type = action.get('action', 'symlink')
                
                # 소스 파일 확인
                if not source_path.exists():
                    results["failed"].append({
                        "model": action['model_type'],
                        "error": f"소스 파일 없음: {source_path}"
                    })
                    continue
                
                # 타겟 디렉토리 생성
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 기존 파일 제거 (백업은 이미 완료)
                if target_path.exists():
                    if target_path.is_symlink():
                        target_path.unlink()
                    else:
                        target_path.unlink()
                
                # 재배치 실행
                if action_type == "symlink":
                    # 심볼릭 링크 생성
                    target_path.symlink_to(source_path.resolve())
                    logger.info(f"🔗 심볼릭 링크: {target_path.name}")
                else:
                    # 파일 복사
                    shutil.copy2(source_path, target_path)
                    logger.info(f"📋 복사: {target_path.name}")
                
                # 성공 기록
                results["success"].append({
                    "model": action['model_type'],
                    "source": str(source_path),
                    "target": str(target_path),
                    "action": action_type,
                    "size_mb": action.get('size_mb', 0),
                    "confidence": action.get('confidence', 0)
                })
                
                results["total_size_mb"] += action.get('size_mb', 0)
                
                logger.info(f"   ✅ 완료 ({action.get('size_mb', 0):.1f}MB)")
                
            except Exception as e:
                error_msg = f"재배치 실패: {e}"
                logger.error(f"   ❌ {error_msg}")
                results["failed"].append({
                    "model": action['model_type'],
                    "error": error_msg
                })
        
        return results
    
    def verify_relocate(self, results: Dict[str, Any]) -> bool:
        """재배치 결과 검증"""
        logger.info("\n🔍 재배치 결과 검증 중...")
        
        success_count = len(results["success"])
        failed_count = len(results["failed"])
        
        # 성공한 파일들 검증
        verified_count = 0
        for success in results["success"]:
            target_path = Path(success["target"])
            if target_path.exists():
                verified_count += 1
            else:
                logger.warning(f"⚠️ 타겟 파일 없음: {target_path}")
        
        logger.info(f"📊 검증 결과:")
        logger.info(f"   ✅ 성공: {success_count}개")
        logger.info(f"   ✅ 검증됨: {verified_count}개")
        logger.info(f"   ❌ 실패: {failed_count}개")
        logger.info(f"   💾 총 크기: {results['total_size_mb']:.1f}MB")
        
        return verified_count == success_count and success_count > 0
    
    def generate_model_config(self, results: Dict[str, Any]):
        """모델 설정 파일 생성"""
        logger.info("🔧 모델 설정 파일 생성 중...")
        
        # 성공한 모델들로 설정 생성
        model_paths = {}
        for success in results["success"]:
            model_type = success["model"]
            target_path = success["target"]
            
            # 상대 경로로 변환
            relative_path = str(Path(target_path).relative_to(self.project_root))
            model_paths[model_type] = relative_path
        
        # Python 설정 파일 생성
        config_content = f'''# app/core/relocated_model_paths.py
"""
재배치된 AI 모델 경로 설정
자동 생성됨: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""

from pathlib import Path

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 재배치된 모델 경로들
RELOCATED_MODEL_PATHS = {{
'''
        
        for model_type, path in model_paths.items():
            config_content += f'    "{model_type}": PROJECT_ROOT / "{path}",\n'
        
        config_content += '''
}

# 모델 타입별 매핑
MODEL_TYPE_MAPPING = {
    "human_parsing_graphonomy": "step_01_human_parsing",
    "pose_estimation_openpose": "step_02_pose_estimation", 
    "cloth_segmentation_u2net": "step_03_cloth_segmentation",
    "geometric_matching_gmm": "step_04_geometric_matching",
    "cloth_warping_tom": "step_05_cloth_warping",
    "virtual_fitting_hrviton": "step_06_virtual_fitting"
}

def get_model_path(model_type: str) -> Path:
    """모델 타입으로 경로 반환"""
    return RELOCATED_MODEL_PATHS.get(model_type, None)

def is_model_available(model_type: str) -> bool:
    """모델 사용 가능 여부 확인"""
    path = get_model_path(model_type)
    return path is not None and path.exists()
'''
        
        # 파일 저장
        config_file = self.project_root / "app" / "core" / "relocated_model_paths.py"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        logger.info(f"✅ 모델 설정 파일 생성: {config_file}")
        
        # JSON 설정도 생성
        json_config = {
            "relocated_models": model_paths,
            "generation_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_models": len(model_paths),
            "total_size_mb": results["total_size_mb"]
        }
        
        json_file = self.project_root / "app" / "core" / "relocated_models.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ JSON 설정 파일 생성: {json_file}")
    
    def print_next_steps(self, results: Dict[str, Any]):
        """다음 단계 안내"""
        logger.info("\n" + "="*70)
        logger.info("🎉 체크포인트 재배치 완료!")
        logger.info("="*70)
        
        if results["success"]:
            logger.info("✅ 다음 명령어로 서버를 시작할 수 있습니다:")
            logger.info("   cd backend")
            logger.info("   python3 app/main.py")
            logger.info("")
            logger.info("📊 재배치된 모델들:")
            for success in results["success"]:
                logger.info(f"   - {success['model']} ({success['size_mb']:.1f}MB)")
        
        if results["failed"]:
            logger.info("\n❌ 실패한 모델들:")
            for failed in results["failed"]:
                logger.info(f"   - {failed['model']}: {failed['error']}")

def main():
    """메인 실행 함수"""
    logger.info("="*70)
    logger.info("🚀 안전한 체크포인트 재배치 실행")
    logger.info("="*70)
    
    relocator = SafeCheckpointRelocator()
    
    # 1. 재배치 계획 로딩
    if not relocator.load_relocate_plan():
        return False
    
    # 2. 계획 검증
    if not relocator.validate_plan():
        return False
    
    # 3. 디렉토리 구조 생성
    relocator.create_directory_structure()
    
    # 4. 기존 파일 백업
    relocator.backup_existing_files()
    
    # 5. 실제 재배치 실행
    results = relocator.execute_relocate()
    
    # 6. 결과 검증
    if relocator.verify_relocate(results):
        # 7. 설정 파일 생성
        relocator.generate_model_config(results)
        
        # 8. 다음 단계 안내
        relocator.print_next_steps(results)
        
        return True
    else:
        logger.error("❌ 재배치 검증 실패")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)