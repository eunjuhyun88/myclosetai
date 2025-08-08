#!/usr/bin/env python3
"""
🔥 Step별 모델 적용 도구
========================

각 Step에 실제로 적용할 수 있는 모델들을 찾고 적용하는 도구

Author: MyCloset AI Team
Date: 2025-08-08
Version: 1.0
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# PyTorch 관련
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# SafeTensors 관련
try:
    import safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

logger = logging.getLogger(__name__)

class StepModelApplicator:
    """Step별 모델 적용 도구"""
    
    def __init__(self):
        self.step_models = {}
        self.applied_models = {}
        
        # Step별 모델 요구사항 정의
        self.step_requirements = {
            'step_01': {
                'name': 'Human Parsing',
                'required_models': ['graphonomy', 'u2net', 'deeplabv3plus'],
                'model_paths': {
                    'graphonomy': 'backend/ai_models/step_01_human_parsing/graphonomy.pth',
                    'u2net': 'backend/ai_models/step_01_human_parsing/u2net.pth',
                    'deeplabv3plus': 'backend/ai_models/step_01_human_parsing/deeplabv3plus.pth'
                },
                'fallback_models': [
                    'backend/ai_models/Graphonomy/inference.pth',
                    'backend/ai_models/Graphonomy/model.safetensors',
                    'backend/ai_models/Self-Correction-Human-Parsing/exp-schp-201908261155-atr.pth'
                ]
            },
            'step_02': {
                'name': 'Pose Estimation',
                'required_models': ['hrnet', 'openpose', 'yolo'],
                'model_paths': {
                    'hrnet': 'backend/ai_models/step_02_pose_estimation/hrnet_w48_coco_384x288.pth',
                    'openpose': 'backend/ai_models/step_02_pose_estimation/openpose.pth',
                    'yolo': 'backend/ai_models/step_02_pose_estimation/yolov8n-pose.pt'
                },
                'fallback_models': [
                    'backend/ai_models/step_02_pose_estimation/hrnet_w48_coco_256x192.pth',
                    'backend/ai_models/step_02_pose_estimation/body_pose_model.pth',
                    'backend/ai_models/openpose.pth'
                ]
            },
            'step_03': {
                'name': 'Cloth Segmentation',
                'required_models': ['sam', 'u2net', 'deeplabv3'],
                'model_paths': {
                    'sam': 'backend/ai_models/step_03_cloth_segmentation/sam_vit_h_4b8939.pth',
                    'u2net': 'backend/ai_models/step_03_cloth_segmentation/u2net.pth',
                    'deeplabv3': 'backend/ai_models/step_03_cloth_segmentation/deeplabv3_resnet101_ultra.pth'
                },
                'fallback_models': [
                    'backend/ai_models/step_03_cloth_segmentation/sam_vit_l_0b3195.pth',
                    'backend/ai_models/step_03_cloth_segmentation/mobile_sam.pt',
                    'backend/ai_models/step_03_cloth_segmentation/deeplabv3_resnet101_coco.pth'
                ]
            },
            'step_04': {
                'name': 'Geometric Matching',
                'required_models': ['gmm', 'tps', 'raft'],
                'model_paths': {
                    'gmm': 'backend/ai_models/step_04_geometric_matching/gmm_final.pth',
                    'tps': 'backend/ai_models/step_04_geometric_matching/tps_network.pth',
                    'raft': 'backend/ai_models/step_04_geometric_matching/raft-things.pth'
                },
                'fallback_models': [
                    'backend/ai_models/step_04_geometric_matching/raft-small.pth',
                    'backend/ai_models/step_04_geometric_matching/raft-sintel.pth',
                    'backend/ai_models/step_04_geometric_matching/raft-kitti.pth'
                ]
            },
            'step_05': {
                'name': 'Cloth Warping',
                'required_models': ['tom', 'viton_hd', 'dpt'],
                'model_paths': {
                    'tom': 'backend/ai_models/step_05_cloth_warping/tom_final.pth',
                    'viton_hd': 'backend/ai_models/step_05_cloth_warping/viton_hd_warping.pth',
                    'dpt': 'backend/ai_models/step_05_cloth_warping/dpt_hybrid_midas.pth'
                },
                'fallback_models': [
                    'backend/ai_models/step_05_cloth_warping/tps_transformation.pth',
                    'backend/ai_models/step_05_cloth_warping/vgg19_warping.pth',
                    'backend/ai_models/dpt_hybrid-midas-501f0c75.pt'
                ]
            },
            'step_06': {
                'name': 'Virtual Fitting',
                'required_models': ['stable_diffusion', 'ootd', 'viton_hd'],
                'model_paths': {
                    'stable_diffusion': 'backend/ai_models/step_06_virtual_fitting/stable_diffusion_4.8gb.pth',
                    'ootd': 'backend/ai_models/step_06_virtual_fitting/ootd_3.2gb.pth',
                    'viton_hd': 'backend/ai_models/step_06_virtual_fitting/viton_hd_2.1gb.pth'
                },
                'fallback_models': [
                    'backend/ai_models/step_06_virtual_fitting/hrviton_final.pth',
                    'backend/ai_models/step_06_virtual_fitting/ootd_checkpoint.pth',
                    'backend/ai_models/step_06_virtual_fitting/validated_checkpoints/ootd_checkpoint.pth'
                ]
            },
            'step_07': {
                'name': 'Post Processing',
                'required_models': ['real_esrgan', 'swinir', 'gfpgan'],
                'model_paths': {
                    'real_esrgan': 'backend/ai_models/step_07_post_processing/RealESRGAN_x4plus.pth',
                    'swinir': 'backend/ai_models/step_07_post_processing/swinir_real_sr_x4_large.pth',
                    'gfpgan': 'backend/ai_models/step_07_post_processing/GFPGAN.pth'
                },
                'fallback_models': [
                    'backend/ai_models/step_07_post_processing/densenet161_enhance.pth',
                    'backend/ai_models/step_07_post_processing/RealESRGAN_x2plus.pth',
                    'backend/ai_models/step_07_post_processing/ESRGAN_x8.pth'
                ]
            },
            'step_08': {
                'name': 'Quality Assessment',
                'required_models': ['clip', 'lpips', 'alex'],
                'model_paths': {
                    'clip': 'backend/ai_models/step_08_quality_assessment/clip_vit_b32.pth',
                    'lpips': 'backend/ai_models/step_08_quality_assessment/lpips_alex.pth',
                    'alex': 'backend/ai_models/step_08_quality_assessment/alex.pth'
                },
                'fallback_models': [
                    'backend/ai_models/step_08_quality_assessment/ViT-L-14.pt',
                    'backend/ai_models/step_08_quality_assessment/ViT-B-32.pt',
                    'backend/ai_models/step_08_quality_assessment/open_clip_pytorch_model.bin'
                ]
            }
        }
    
    def find_available_models(self):
        """사용 가능한 모델들 찾기"""
        print("🔍 사용 가능한 모델들 검색 중...")
        
        for step_key, step_info in self.step_requirements.items():
            print(f"\n🎯 {step_info['name']} ({step_key})")
            
            available_models = {}
            
            # 필수 모델 확인
            for model_name, model_path in step_info['model_paths'].items():
                if Path(model_path).exists():
                    available_models[model_name] = {
                        'path': model_path,
                        'type': 'required',
                        'valid': self._validate_model(model_path)
                    }
                    print(f"   ✅ {model_name}: {Path(model_path).name}")
                else:
                    print(f"   ❌ {model_name}: 파일 없음")
            
            # 대체 모델 확인
            for fallback_path in step_info['fallback_models']:
                if Path(fallback_path).exists():
                    model_name = Path(fallback_path).stem
                    available_models[model_name] = {
                        'path': fallback_path,
                        'type': 'fallback',
                        'valid': self._validate_model(fallback_path)
                    }
                    print(f"   🔄 {model_name}: {Path(fallback_path).name} (대체)")
            
            self.step_models[step_key] = available_models
    
    def _validate_model(self, model_path: str) -> bool:
        """모델 유효성 검증"""
        try:
            if model_path.endswith('.safetensors') and SAFETENSORS_AVAILABLE:
                with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
                    keys = list(f.keys())
                return len(keys) > 0
            else:
                model_data = torch.load(model_path, map_location='cpu', weights_only=True)
                if isinstance(model_data, dict):
                    if 'state_dict' in model_data:
                        return len(model_data['state_dict']) > 0
                    else:
                        return len(model_data) > 0
                else:
                    return True
        except Exception as e:
            return False
    
    def apply_models_to_steps(self):
        """각 Step에 모델 적용"""
        print("\n🔧 각 Step에 모델 적용 중...")
        
        for step_key, step_info in self.step_requirements.items():
            print(f"\n🎯 {step_info['name']} ({step_key}) 적용")
            
            available_models = self.step_models.get(step_key, {})
            
            if not available_models:
                print(f"   ⚠️ 사용 가능한 모델이 없습니다")
                continue
            
            # 유효한 모델들만 필터링
            valid_models = {name: info for name, info in available_models.items() if info['valid']}
            
            if not valid_models:
                print(f"   ❌ 유효한 모델이 없습니다")
                continue
            
            # 모델 적용
            applied_models = []
            for model_name, model_info in valid_models.items():
                if self._apply_model_to_step(step_key, model_name, model_info):
                    applied_models.append(model_name)
            
            self.applied_models[step_key] = applied_models
            print(f"   ✅ 적용 완료: {', '.join(applied_models)}")
    
    def _apply_model_to_step(self, step_key: str, model_name: str, model_info: Dict[str, Any]) -> bool:
        """개별 모델을 Step에 적용"""
        try:
            model_path = model_info['path']
            
            # Step별 디렉토리 생성
            step_dir = Path(f"backend/ai_models/{step_key}")
            step_dir.mkdir(parents=True, exist_ok=True)
            
            # 모델 파일 복사
            target_path = step_dir / f"{model_name}.pth"
            
            if model_path.endswith('.safetensors'):
                # SafeTensors를 PyTorch로 변환
                if SAFETENSORS_AVAILABLE:
                    with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
                        keys = list(f.keys())
                        state_dict = {key: f.get_tensor(key) for key in keys}
                    
                    torch.save({'state_dict': state_dict}, target_path)
                    print(f"   🔄 {model_name}: SafeTensors → PyTorch 변환 완료")
                    return True
            else:
                # PyTorch 모델 복사
                shutil.copy2(model_path, target_path)
                print(f"   📋 {model_name}: 모델 복사 완료")
                return True
            
        except Exception as e:
            print(f"   ❌ {model_name}: 적용 실패 - {e}")
            return False
    
    def create_step_configs(self):
        """Step별 설정 파일 생성"""
        print("\n📋 Step별 설정 파일 생성 중...")
        
        for step_key, step_info in self.step_requirements.items():
            print(f"\n🎯 {step_info['name']} ({step_key}) 설정 생성")
            
            applied_models = self.applied_models.get(step_key, [])
            
            if not applied_models:
                print(f"   ⚠️ 적용된 모델이 없습니다")
                continue
            
            # 설정 파일 생성
            config = {
                'step_name': step_info['name'],
                'step_key': step_key,
                'applied_models': applied_models,
                'model_paths': {},
                'created_at': datetime.now().isoformat()
            }
            
            # 모델 경로 추가
            for model_name in applied_models:
                model_path = f"backend/ai_models/{step_key}/{model_name}.pth"
                config['model_paths'][model_name] = model_path
            
            # 설정 파일 저장
            config_path = f"backend/ai_models/{step_key}/step_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ 설정 파일 생성: {config_path}")
    
    def update_pipeline_config(self):
        """파이프라인 설정 업데이트"""
        print("\n🔧 파이프라인 설정 업데이트 중...")
        
        pipeline_config = {
            'pipeline_name': 'MyCloset AI Virtual Try-On Pipeline',
            'version': '1.0',
            'steps': {},
            'created_at': datetime.now().isoformat()
        }
        
        for step_key, step_info in self.step_requirements.items():
            applied_models = self.applied_models.get(step_key, [])
            
            pipeline_config['steps'][step_key] = {
                'name': step_info['name'],
                'applied_models': applied_models,
                'status': 'ready' if applied_models else 'missing_models'
            }
        
        # 파이프라인 설정 저장
        config_path = "backend/ai_models/pipeline_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(pipeline_config, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ 파이프라인 설정 저장: {config_path}")
    
    def generate_application_report(self):
        """적용 리포트 생성"""
        report = []
        report.append("🔥 Step별 모델 적용 리포트")
        report.append("=" * 80)
        report.append(f"📅 적용 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        total_applied = 0
        total_required = 0
        
        for step_key, step_info in self.step_requirements.items():
            applied_models = self.applied_models.get(step_key, [])
            required_models = list(step_info['model_paths'].keys())
            
            total_applied += len(applied_models)
            total_required += len(required_models)
            
            status = "✅ 완료" if applied_models else "❌ 미완료"
            report.append(f"🎯 {step_info['name']} ({step_key}): {status}")
            
            if applied_models:
                report.append(f"   ✅ 적용된 모델: {', '.join(applied_models)}")
            else:
                report.append(f"   ❌ 필요한 모델: {', '.join(required_models)}")
            
            report.append("")
        
        # 전체 통계
        report.append("📊 전체 통계")
        report.append("-" * 50)
        report.append(f"   🔍 총 필요 모델: {total_required}개")
        report.append(f"   ✅ 적용 완료: {total_applied}개")
        report.append(f"   📈 적용률: {(total_applied/total_required*100):.1f}%" if total_required > 0 else "   📈 적용률: 0%")
        report.append("")
        
        return "\n".join(report)

def main():
    """메인 함수"""
    print("🔥 Step별 모델 적용 도구")
    print("=" * 80)
    
    # 적용기 초기화
    applicator = StepModelApplicator()
    
    # 1. 사용 가능한 모델들 찾기
    print("\n📋 1단계: 사용 가능한 모델들 검색")
    applicator.find_available_models()
    
    # 2. 각 Step에 모델 적용
    print("\n🔧 2단계: 각 Step에 모델 적용")
    applicator.apply_models_to_steps()
    
    # 3. Step별 설정 파일 생성
    print("\n📋 3단계: Step별 설정 파일 생성")
    applicator.create_step_configs()
    
    # 4. 파이프라인 설정 업데이트
    print("\n🔧 4단계: 파이프라인 설정 업데이트")
    applicator.update_pipeline_config()
    
    # 5. 적용 리포트 생성
    print("\n📋 5단계: 적용 리포트 생성")
    report = applicator.generate_application_report()
    print(report)
    
    # 6. 리포트 저장
    with open("step_model_application_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n💾 적용 리포트 저장: step_model_application_report.txt")
    print("\n🎉 Step별 모델 적용 완료!")

if __name__ == "__main__":
    main()
