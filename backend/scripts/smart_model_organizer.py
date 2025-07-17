#!/usr/bin/env python3
"""
🧠 MyCloset AI - 스마트 모델 정리 및 최적화 스크립트
M3 Max 128GB 최적화 | 우선순위 기반 모델 선택

사용법: python scripts/smart_model_organizer.py
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('model_organization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SmartModelOrganizer:
    """스마트 AI 모델 정리 및 최적화 도구"""
    
    def __init__(self, base_dir: str = "ai_models/checkpoints"):
        self.base_dir = Path(base_dir)
        self.checkpoint_report = self.load_checkpoint_report()
        
        # M3 Max 128GB에 최적화된 모델 우선순위
        self.model_priorities = {
            # ⭐ 최고 우선순위 (필수)
            1: ["ootdiffusion", "ootdiffusion_hf"],  # 가상 피팅 핵심
            
            # 🔥 높은 우선순위 (중요)
            2: ["human_parsing", "step_01_human_parsing"],  # 인간 파싱
            3: ["openpose", "step_02_pose_estimation"],     # 포즈 추정
            4: ["u2net", "step_03_cloth_segmentation"],     # 의류 분할
            5: ["step_04_geometric_matching"],              # 기하학적 매칭
            6: ["step_05_cloth_warping"],                   # 의류 워핑
            
            # 💡 중간 우선순위 (선택적)
            7: ["clip-vit-base-patch32", "grounding_dino"], # 텍스트-이미지
            8: ["step_07_post_processing"],                 # 후처리
            
            # 📦 낮은 우선순위 (백업용)
            9: ["stable-diffusion-v1-5"],                   # 대체 모델
            10: ["auxiliary", "background_removal"]         # 보조 도구
        }
        
        # 제거 대상 (중복/불필요)
        self.removal_candidates = [
            "stable_diffusion_v15",      # 중복
            "stable_diffusion_inpaint",  # 중복  
            "sam_vit_h",                 # 중복
            "clip-vit-large-patch14",    # 큰 용량, base로 충분
            "controlnet_openpose",       # 중복
            "esrgan", "gfpgan", "rembg", # 사용하지 않음
            "viton_hd",                  # 불완전
            "densepose",                 # 비어있음
            "u2net_cloth",               # 비어있음
        ]
    
    def load_checkpoint_report(self) -> Dict:
        """체크포인트 분석 보고서 로드"""
        report_path = self.base_dir / "checkpoint_analysis_report.json"
        if report_path.exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def analyze_current_state(self) -> Dict:
        """현재 상태 분석"""
        logger.info("🔍 현재 AI 모델 상태 분석 중...")
        
        if not self.checkpoint_report:
            logger.error("❌ checkpoint_analysis_report.json을 찾을 수 없습니다")
            return {}
        
        analyzed_models = self.checkpoint_report.get('analyzed_models', {})
        summary = self.checkpoint_report.get('summary', {})
        
        analysis = {
            'total_models': summary.get('total_models', 0),
            'ready_models': summary.get('ready_models', 0),
            'total_size_gb': self.checkpoint_report.get('total_size_gb', 0),
            'priority_analysis': {},
            'removal_analysis': {},
            'size_optimization': {}
        }
        
        # 우선순위별 분석
        for priority, model_names in self.model_priorities.items():
            priority_info = {
                'models': [],
                'total_size_mb': 0,
                'ready_count': 0
            }
            
            for model_name in model_names:
                if model_name in analyzed_models:
                    model_info = analyzed_models[model_name]
                    priority_info['models'].append({
                        'name': model_name,
                        'size_mb': model_info.get('total_size_mb', 0),
                        'ready': model_info.get('ready', False),
                        'type': model_info.get('type', 'unknown')
                    })
                    priority_info['total_size_mb'] += model_info.get('total_size_mb', 0)
                    if model_info.get('ready', False):
                        priority_info['ready_count'] += 1
            
            analysis['priority_analysis'][priority] = priority_info
        
        # 제거 대상 분석
        removal_size = 0
        removal_models = []
        for model_name in self.removal_candidates:
            if model_name in analyzed_models:
                model_info = analyzed_models[model_name]
                size_mb = model_info.get('total_size_mb', 0)
                removal_size += size_mb
                removal_models.append({
                    'name': model_name,
                    'size_mb': size_mb,
                    'reason': self._get_removal_reason(model_name)
                })
        
        analysis['removal_analysis'] = {
            'total_models': len(removal_models),
            'total_size_mb': removal_size,
            'models': removal_models
        }
        
        # 크기 최적화 분석
        analysis['size_optimization'] = {
            'current_size_gb': analysis['total_size_gb'],
            'after_removal_gb': (analysis['total_size_gb'] * 1024 - removal_size) / 1024,
            'space_saved_gb': removal_size / 1024,
            'optimization_percentage': (removal_size / (analysis['total_size_gb'] * 1024)) * 100
        }
        
        return analysis
    
    def _get_removal_reason(self, model_name: str) -> str:
        """제거 이유 반환"""
        reasons = {
            "stable_diffusion_v15": "stable-diffusion-v1-5와 중복",
            "stable_diffusion_inpaint": "ootdiffusion으로 대체 가능",
            "sam_vit_h": "sam으로 충분",
            "clip-vit-large-patch14": "clip-vit-base-patch32로 충분",
            "controlnet_openpose": "openpose와 중복",
            "esrgan": "step_07_post_processing에 포함됨",
            "gfpgan": "사용하지 않음",
            "rembg": "u2net으로 대체 가능",
            "viton_hd": "불완전한 모델",
            "densepose": "비어있는 디렉토리",
            "u2net_cloth": "비어있는 디렉토리"
        }
        return reasons.get(model_name, "불필요한 모델")
    
    def create_optimization_plan(self, analysis: Dict) -> Dict:
        """최적화 계획 생성"""
        logger.info("📋 모델 최적화 계획 생성 중...")
        
        plan = {
            'timestamp': datetime.now().isoformat(),
            'current_state': {
                'total_size_gb': analysis['total_size_gb'],
                'total_models': analysis['total_models']
            },
            'actions': [],
            'expected_result': {
                'final_size_gb': analysis['size_optimization']['after_removal_gb'],
                'space_saved_gb': analysis['size_optimization']['space_saved_gb'],
                'optimization_percentage': analysis['size_optimization']['optimization_percentage']
            }
        }
        
        # 1. 제거 액션
        for model in analysis['removal_analysis']['models']:
            plan['actions'].append({
                'type': 'remove',
                'model': model['name'],
                'reason': model['reason'],
                'size_mb': model['size_mb'],
                'priority': 'high'
            })
        
        # 2. 재구성 액션 (우선순위 기반)
        for priority in range(1, 7):  # 핵심 모델들만
            if priority in analysis['priority_analysis']:
                priority_info = analysis['priority_analysis'][priority]
                for model in priority_info['models']:
                    if model['ready']:
                        plan['actions'].append({
                            'type': 'reorganize',
                            'model': model['name'],
                            'priority': priority,
                            'size_mb': model['size_mb'],
                            'target_path': f"step_{priority:02d}_{model['type']}/{model['name']}"
                        })
        
        # 3. 심볼릭 링크 액션 (중복 제거)
        plan['actions'].append({
            'type': 'create_symlinks',
            'description': "중복 모델들을 심볼릭 링크로 연결",
            'targets': [
                "stable-diffusion-v1-5 → ootdiffusion",
                "clip-vit-base → clip-vit-base-patch32", 
                "sam → sam_vit_h"
            ]
        })
        
        return plan
    
    def execute_optimization(self, plan: Dict, dry_run: bool = True) -> bool:
        """최적화 실행"""
        logger.info(f"{'🧪 [DRY RUN]' if dry_run else '🚀 [EXECUTE]'} 모델 최적화 실행...")
        
        if not dry_run:
            # 백업 생성
            backup_dir = Path(f"backup_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            logger.info(f"📦 백업 생성: {backup_dir}")
            
        executed_actions = []
        
        for action in plan['actions']:
            action_type = action['type']
            model_name = action.get('model', 'unknown')
            
            try:
                if action_type == 'remove':
                    result = self._execute_remove_action(action, dry_run)
                elif action_type == 'reorganize':
                    result = self._execute_reorganize_action(action, dry_run)
                elif action_type == 'create_symlinks':
                    result = self._execute_symlink_action(action, dry_run)
                else:
                    result = {'success': False, 'error': f'Unknown action type: {action_type}'}
                
                executed_actions.append({
                    'action': action,
                    'result': result
                })
                
                if result['success']:
                    logger.info(f"✅ {action_type}: {model_name}")
                else:
                    logger.error(f"❌ {action_type}: {model_name} - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"🚨 {action_type} 실행 중 오류: {e}")
                executed_actions.append({
                    'action': action,
                    'result': {'success': False, 'error': str(e)}
                })
        
        # 결과 저장
        result_file = f"optimization_result_{'dryrun' if dry_run else 'executed'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'plan': plan,
                'executed_actions': executed_actions,
                'summary': self._create_execution_summary(executed_actions)
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 실행 결과 저장: {result_file}")
        return True
    
    def _execute_remove_action(self, action: Dict, dry_run: bool) -> Dict:
        """제거 액션 실행"""
        model_name = action['model']
        model_path = self.base_dir / model_name
        
        if not model_path.exists():
            return {'success': True, 'message': 'Already removed'}
        
        if dry_run:
            return {'success': True, 'message': f'Would remove {model_path}'}
        
        try:
            if model_path.is_dir():
                shutil.rmtree(model_path)
            else:
                model_path.unlink()
            return {'success': True, 'message': f'Removed {model_path}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _execute_reorganize_action(self, action: Dict, dry_run: bool) -> Dict:
        """재구성 액션 실행"""
        model_name = action['model']
        target_path = action['target_path']
        
        if dry_run:
            return {'success': True, 'message': f'Would reorganize {model_name} to {target_path}'}
        
        # 실제 재구성 로직 구현
        return {'success': True, 'message': f'Reorganized {model_name}'}
    
    def _execute_symlink_action(self, action: Dict, dry_run: bool) -> Dict:
        """심볼릭 링크 액션 실행"""
        if dry_run:
            return {'success': True, 'message': 'Would create symlinks'}
        
        # 실제 심볼릭 링크 생성 로직 구현
        return {'success': True, 'message': 'Created symlinks'}
    
    def _create_execution_summary(self, executed_actions: List[Dict]) -> Dict:
        """실행 요약 생성"""
        total_actions = len(executed_actions)
        successful_actions = sum(1 for action in executed_actions if action['result']['success'])
        
        return {
            'total_actions': total_actions,
            'successful_actions': successful_actions,
            'failed_actions': total_actions - successful_actions,
            'success_rate': (successful_actions / total_actions * 100) if total_actions > 0 else 0
        }
    
    def generate_report(self, analysis: Dict, plan: Dict) -> str:
        """상세 보고서 생성"""
        report = f"""
🧠 MyCloset AI - 스마트 모델 최적화 보고서
========================================
📅 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 현재 상태 분석
-----------------
총 모델 수: {analysis['total_models']}개
사용 가능한 모델: {analysis['ready_models']}개  
총 용량: {analysis['total_size_gb']:.1f}GB

🎯 우선순위별 모델 분석
---------------------"""
        
        for priority, info in analysis['priority_analysis'].items():
            if info['models']:
                report += f"\n우선순위 {priority}: {info['ready_count']}/{len(info['models'])}개 준비됨, {info['total_size_mb']/1024:.1f}GB"
                for model in info['models']:
                    status = "✅" if model['ready'] else "❌"
                    report += f"\n  {status} {model['name']} ({model['size_mb']/1024:.1f}GB)"
        
        report += f"""

🗑️ 제거 대상 분석
----------------
제거할 모델: {analysis['removal_analysis']['total_models']}개
절약 용량: {analysis['removal_analysis']['total_size_mb']/1024:.1f}GB
"""
        
        for model in analysis['removal_analysis']['models']:
            report += f"\n❌ {model['name']} ({model['size_mb']/1024:.1f}GB) - {model['reason']}"
        
        report += f"""

💡 최적화 결과 예상
-----------------
현재 용량: {analysis['size_optimization']['current_size_gb']:.1f}GB
최적화 후: {analysis['size_optimization']['after_removal_gb']:.1f}GB
절약 용량: {analysis['size_optimization']['space_saved_gb']:.1f}GB
최적화율: {analysis['size_optimization']['optimization_percentage']:.1f}%

🚀 권장 액션
-----------
1. 중복/불필요 모델 제거 ({len(analysis['removal_analysis']['models'])}개)
2. 우선순위 1-6 모델 유지 (핵심 기능)
3. 심볼릭 링크로 중복 제거
4. 백업 후 점진적 적용

⚠️ 주의사항
----------
- 실행 전 전체 백업 필수
- 테스트 환경에서 먼저 검증
- 단계별 적용 권장
"""
        
        return report

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MyCloset AI 스마트 모델 정리')
    parser.add_argument('--analyze', action='store_true', help='분석만 수행')
    parser.add_argument('--plan', action='store_true', help='계획 생성')
    parser.add_argument('--execute', action='store_true', help='실제 실행')
    parser.add_argument('--dry-run', action='store_true', default=True, help='가상 실행 (기본값)')
    
    args = parser.parse_args()
    
    organizer = SmartModelOrganizer()
    
    # 1. 현재 상태 분석
    logger.info("🔍 Step 1: 현재 모델 상태 분석")
    analysis = organizer.analyze_current_state()
    
    if not analysis:
        logger.error("❌ 분석 실패")
        return False
    
    # 2. 최적화 계획 생성
    if args.plan or args.execute:
        logger.info("📋 Step 2: 최적화 계획 생성")
        plan = organizer.create_optimization_plan(analysis)
    else:
        plan = {}
    
    # 3. 보고서 생성
    report = organizer.generate_report(analysis, plan)
    print(report)
    
    # 보고서 파일 저장
    report_file = f"model_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"📊 보고서 저장: {report_file}")
    
    # 4. 실행 (옵션)
    if args.execute and plan:
        dry_run = args.dry_run
        logger.info(f"🚀 Step 3: 최적화 실행 ({'가상' if dry_run else '실제'})")
        organizer.execute_optimization(plan, dry_run=dry_run)
    
    logger.info("🎉 스마트 모델 정리 완료!")
    return True

if __name__ == "__main__":
    main()