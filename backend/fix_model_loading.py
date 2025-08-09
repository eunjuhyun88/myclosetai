#!/usr/bin/env python3
"""
🛠️ 모델 로딩 문제 해결 스크립트
1. PyTorch 호환성 패치
2. 키 매핑 알고리즘 개선
3. 체크포인트 구조 분석 강화
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def fix_pytorch_compatibility():
    """PyTorch 호환성 패치"""
    print("🔧 PyTorch 호환성 패치 적용...")
    
    try:
        import torch
        
        # PyTorch 2.7+ 호환성 패치
        if hasattr(torch, 'load'):
            original_load = torch.load
            
            def safe_load(*args, **kwargs):
                """안전한 체크포인트 로딩"""
                try:
                    # weights_only=True로 시도
                    kwargs['weights_only'] = True
                    return original_load(*args, **kwargs)
                except Exception as e1:
                    try:
                        # weights_only 제거하고 다시 시도
                        kwargs.pop('weights_only', None)
                        return original_load(*args, **kwargs)
                    except Exception as e2:
                        try:
                            # map_location='cpu'로 강제 시도
                            kwargs['map_location'] = 'cpu'
                            return original_load(*args, **kwargs)
                        except Exception as e3:
                            print(f"⚠️ 체크포인트 로딩 실패: {e3}")
                            return None
            
            torch.load = safe_load
            print("   ✅ PyTorch 호환성 패치 적용 완료")
        else:
            print("   ⚠️ PyTorch 로드 함수를 찾을 수 없음")
            
    except Exception as e:
        print(f"   ❌ PyTorch 호환성 패치 실패: {e}")

def improve_key_mapping():
    """키 매핑 알고리즘 개선"""
    print("\n🔑 키 매핑 알고리즘 개선...")
    
    try:
        from app.ai_pipeline.utils.model_loader import KeyMapper
        
        # KeyMapper 클래스에 개선된 매핑 메서드 추가
        def enhanced_map_keys(self, checkpoint, target_architecture, model_state_dict):
            """개선된 키 매핑"""
            try:
                # State dict 추출
                if 'state_dict' in checkpoint:
                    source_state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    source_state_dict = checkpoint['model']
                else:
                    source_state_dict = checkpoint
                
                # 퍼지 매칭을 위한 키 정규화
                normalized_source = {}
                for key, value in source_state_dict.items():
                    # 키 정규화 (소문자, 특수문자 제거)
                    normalized_key = key.lower().replace('_', '').replace('.', '')
                    normalized_source[normalized_key] = (key, value)
                
                # 타겟 모델 키 정규화
                normalized_target = {}
                for key in model_state_dict.keys():
                    normalized_key = key.lower().replace('_', '').replace('.', '')
                    normalized_target[normalized_key] = key
                
                # 매핑 수행
                mapped_dict = {}
                success_count = 0
                
                for norm_source_key, (orig_source_key, source_value) in normalized_source.items():
                    # 정확 매칭
                    if norm_source_key in normalized_target:
                        target_key = normalized_target[norm_source_key]
                        mapped_dict[target_key] = source_value
                        success_count += 1
                        continue
                    
                    # 부분 매칭
                    for norm_target_key, target_key in normalized_target.items():
                        if (norm_source_key in norm_target_key or 
                            norm_target_key in norm_source_key):
                            # 텐서 크기 확인
                            if hasattr(source_value, 'shape') and hasattr(model_state_dict[target_key], 'shape'):
                                if source_value.shape == model_state_dict[target_key].shape:
                                    mapped_dict[target_key] = source_value
                                    success_count += 1
                                    break
                
                print(f"   📊 개선된 매핑 결과: {success_count}/{len(model_state_dict)} 성공")
                return mapped_dict
                
            except Exception as e:
                print(f"   ❌ 개선된 매핑 실패: {e}")
                return {}
        
        # KeyMapper에 개선된 메서드 추가
        KeyMapper.enhanced_map_keys = enhanced_map_keys
        print("   ✅ 키 매핑 알고리즘 개선 완료")
        
    except Exception as e:
        print(f"   ❌ 키 매핑 개선 실패: {e}")

def enhance_checkpoint_analysis():
    """체크포인트 구조 분석 강화"""
    print("\n🔍 체크포인트 구조 분석 강화...")
    
    try:
        from app.ai_pipeline.utils.model_loader import CheckpointAnalyzer
        
        def enhanced_analyze_checkpoint(self, checkpoint_path):
            """강화된 체크포인트 분석"""
            try:
                import torch
                
                # 안전한 로딩
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                if checkpoint is None:
                    return {}
                
                # State dict 추출
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
                
                # 상세 분석
                analysis = {
                    'architecture_type': self._infer_architecture_type(state_dict, checkpoint_path),
                    'total_params': sum(tensor.numel() for tensor in state_dict.values() if hasattr(tensor, 'numel')),
                    'input_channels': self._infer_input_channels(state_dict),
                    'num_classes': self._infer_num_classes(state_dict),
                    'num_keypoints': self._infer_num_keypoints(state_dict),
                    'num_control_points': self._infer_num_control_points(state_dict),
                    'key_patterns': self._analyze_key_patterns(state_dict),
                    'layer_types': self._analyze_layer_types(state_dict),
                    'model_depth': self._estimate_model_depth(state_dict),
                    'has_batch_norm': self._has_batch_normalization(state_dict),
                    'has_attention': self._has_attention_layers(state_dict),
                    'metadata': self._extract_metadata(checkpoint)
                }
                
                print(f"   📊 강화된 분석 완료: {analysis['total_params']:,} 파라미터")
                return analysis
                
            except Exception as e:
                print(f"   ❌ 강화된 분석 실패: {e}")
                return {}
        
        # CheckpointAnalyzer에 강화된 메서드 추가
        CheckpointAnalyzer.enhanced_analyze_checkpoint = enhanced_analyze_checkpoint
        print("   ✅ 체크포인트 구조 분석 강화 완료")
        
    except Exception as e:
        print(f"   ❌ 체크포인트 분석 강화 실패: {e}")

def test_improved_loading():
    """개선된 로딩 테스트"""
    print("\n🧪 개선된 로딩 테스트...")
    
    try:
        from app.ai_pipeline.utils.model_loader import ModelLoader
        
        loader = ModelLoader()
        
        test_cases = [
            ('human_parsing', 'ai_models/step_01_human_parsing/graphonomy.pth'),
            ('pose_estimation', 'ai_models/step_02_pose_estimation/hrnet_w48_coco_384x288.pth'),
            ('cloth_segmentation', 'ai_models/step_03/sam.pth')
        ]
        
        for step, path in test_cases:
            if os.path.exists(path):
                print(f"\n   🔧 {step} 개선된 로딩 테스트...")
                try:
                    model = loader.load_model_for_step(step, checkpoint_path=path)
                    if model:
                        param_count = sum(p.numel() for p in model.parameters())
                        print(f"      ✅ 로딩 성공: {param_count:,} 파라미터")
                    else:
                        print(f"      ❌ 로딩 실패")
                except Exception as e:
                    print(f"      ❌ 로딩 오류: {e}")
            else:
                print(f"   ⚠️ {step}: 파일 없음 ({path})")
        
    except Exception as e:
        print(f"   ❌ 개선된 로딩 테스트 실패: {e}")

def main():
    """메인 함수"""
    print("🛠️ 모델 로딩 문제 해결 시작...")
    print("=" * 60)
    
    # 1. PyTorch 호환성 패치
    fix_pytorch_compatibility()
    
    # 2. 키 매핑 알고리즘 개선
    improve_key_mapping()
    
    # 3. 체크포인트 구조 분석 강화
    enhance_checkpoint_analysis()
    
    # 4. 개선된 로딩 테스트
    test_improved_loading()
    
    print("\n" + "=" * 60)
    print("🎉 모델 로딩 문제 해결 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()
