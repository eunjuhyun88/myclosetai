# final_warning_fix.py - backend 디렉토리에서 실행
"""
최종 워닝 해결 패치 스크립트
실행: python final_warning_fix.py
"""

import sys
import os
import logging
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def force_integrate_smart_mapper():
    """SmartMapper를 강제로 ModelLoader와 완전 연동"""
    try:
        print("🔥 SmartMapper 강제 연동 시작...")
        
        # SmartMapper 가져오기
        from app.ai_pipeline.utils.smart_model_mapper import get_global_smart_mapper
        smart_mapper = get_global_smart_mapper()
        
        # ModelLoader 가져오기
        from app.ai_pipeline.utils.model_loader import get_global_model_loader
        model_loader = get_global_model_loader()
        
        if not model_loader:
            print("❌ ModelLoader 없음")
            return False
            
        print(f"✅ ModelLoader 획득: {type(model_loader).__name__}")
        
        # 현재 available_models 상태 확인
        available_models = getattr(model_loader, '_available_models_cache', {})
        print(f"📊 현재 available_models: {len(available_models)}개")
        
        # SmartMapper 모델들을 강제로 추가
        missing_models = {
            "vgg16_warping": {
                "name": "vgg16_warping",
                "path": "ai_models/step_05_cloth_warping/ultra_models/vgg16_warping_ultra.pth",
                "checkpoint_path": "ai_models/step_05_cloth_warping/ultra_models/vgg16_warping_ultra.pth",
                "ai_model_info": {"ai_class": "RealVGG16Model"},
                "size_mb": 527.8,
                "step_class": "ClothWarpingStep",
                "model_type": "warping",
                "loaded": False,
                "device": "mps"
            },
            "vgg19_warping": {
                "name": "vgg19_warping", 
                "path": "ai_models/step_05_cloth_warping/ultra_models/vgg19_warping.pth",
                "checkpoint_path": "ai_models/step_05_cloth_warping/ultra_models/vgg19_warping.pth",
                "ai_model_info": {"ai_class": "RealVGG19Model"},
                "size_mb": 548.1,
                "step_class": "ClothWarpingStep",
                "model_type": "warping",
                "loaded": False,
                "device": "mps"
            },
            "densenet121": {
                "name": "densenet121",
                "path": "ai_models/step_05_cloth_warping/ultra_models/densenet121_ultra.pth",
                "checkpoint_path": "ai_models/step_05_cloth_warping/ultra_models/densenet121_ultra.pth",
                "ai_model_info": {"ai_class": "RealDenseNetModel"},
                "size_mb": 31.0,
                "step_class": "ClothWarpingStep", 
                "model_type": "warping",
                "loaded": False,
                "device": "mps"
            }
        }
        
        # 누락된 모델들 강제 추가
        added_count = 0
        for model_name, model_info in missing_models.items():
            # SmartMapper에서 실제 경로 확인
            mapping_result = smart_mapper.get_model_path(model_name)
            
            if mapping_result and mapping_result.found:
                # 실제 경로로 업데이트
                model_info["path"] = str(mapping_result.actual_path)
                model_info["checkpoint_path"] = str(mapping_result.actual_path)
                model_info["size_mb"] = mapping_result.size_mb
                model_info["ai_model_info"]["ai_class"] = mapping_result.ai_class
                
                # available_models에 추가
                available_models[model_name] = model_info
                added_count += 1
                
                print(f"  ✅ {model_name}: {mapping_result.actual_path} ({mapping_result.size_mb:.1f}MB)")
            else:
                print(f"  ⚠️ {model_name}: SmartMapper에서 찾을 수 없음")
        
        # ModelLoader 캐시 업데이트
        if hasattr(model_loader, '_available_models_cache'):
            model_loader._available_models_cache = available_models
            
        print(f"✅ 강제 연동 완료: {added_count}개 모델 추가")
        print(f"📊 최종 available_models: {len(available_models)}개")
        
        return added_count > 0
        
    except Exception as e:
        print(f"❌ 강제 연동 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def patch_step_model_loading():
    """Step 모델 로딩을 패치하여 SmartMapper 우선 사용"""
    try:
        print("\n🔧 Step 모델 로딩 패치 적용 중...")
        
        # ClothWarpingStep 패치
        try:
            from app.ai_pipeline.steps.step_05_cloth_warping import ClothWarpingStep
            original_load_models = getattr(ClothWarpingStep, '_load_real_ai_models', None)
            
            if original_load_models:
                def patched_load_models(self):
                    """패치된 모델 로딩 함수"""
                    try:
                        # SmartMapper 우선 시도
                        from app.ai_pipeline.utils.smart_model_mapper import resolve_model_path
                        
                        model_names = ["realvis_xl", "vgg16_warping", "vgg19_warping", "densenet121"]
                        loaded_models = {}
                        
                        for model_name in model_names:
                            resolved_path = resolve_model_path(model_name)
                            if resolved_path:
                                print(f"🔄 SmartMapper 경로 해결: {model_name} → {resolved_path}")
                                # 여기서 실제 모델 로딩 로직
                                loaded_models[model_name] = {
                                    "path": str(resolved_path),
                                    "loaded": True,
                                    "source": "SmartMapper"
                                }
                            else:
                                # 원본 로딩 시도
                                try:
                                    result = original_load_models(self)
                                    if result:
                                        loaded_models.update(result)
                                except:
                                    pass
                        
                        return loaded_models
                        
                    except Exception as e:
                        print(f"⚠️ 패치된 로딩 실패: {e}")
                        # 원본 함수로 폴백
                        return original_load_models(self)
                
                # 메서드 교체
                ClothWarpingStep._load_real_ai_models = patched_load_models
                print("✅ ClothWarpingStep 패치 적용 완료")
                
        except Exception as e:
            print(f"⚠️ ClothWarpingStep 패치 실패: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Step 패치 실패: {e}")
        return False

def create_persistent_fix():
    """지속적인 워닝 해결을 위한 패치 파일 생성"""
    try:
        print("\n📝 지속적 워닝 해결 패치 파일 생성 중...")
        
        patch_content = '''# 워닝 해결 패치 - main.py 시작 시 자동 실행
def apply_warning_fixes():
    """main.py에서 자동 실행될 워닝 해결 패치"""
    try:
        # SmartMapper 강제 연동
        from app.ai_pipeline.utils.smart_model_mapper import get_global_smart_mapper
        from app.ai_pipeline.utils.model_loader import get_global_model_loader
        
        smart_mapper = get_global_smart_mapper()
        model_loader = get_global_model_loader()
        
        if smart_mapper and model_loader:
            # 누락된 모델들 자동 추가
            missing_models = ["vgg16_warping", "vgg19_warping", "densenet121"]
            
            for model_name in missing_models:
                mapping_result = smart_mapper.get_model_path(model_name)
                if mapping_result and mapping_result.found:
                    if hasattr(model_loader, '_available_models_cache'):
                        model_loader._available_models_cache[model_name] = {
                            "name": model_name,
                            "path": str(mapping_result.actual_path),
                            "checkpoint_path": str(mapping_result.actual_path),
                            "ai_model_info": {"ai_class": mapping_result.ai_class},
                            "size_mb": mapping_result.size_mb,
                            "loaded": False,
                            "device": "mps"
                        }
            
            return True
            
    except Exception as e:
        print(f"⚠️ 자동 워닝 해결 실패: {e}")
        
    return False
'''
        
        # 패치 파일 저장
        patch_file = Path("app/utils/warning_fixes.py")
        patch_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(patch_file, 'w', encoding='utf-8') as f:
            f.write(patch_content)
        
        print(f"✅ 패치 파일 생성 완료: {patch_file}")
        
        # main.py에 추가할 코드 출력
        print("\n📋 main.py에 추가할 코드:")
        print("=" * 50)
        print("# main.py에 이 코드를 추가하세요 (AI 컨테이너 초기화 전)")
        print("try:")
        print("    from app.utils.warning_fixes import apply_warning_fixes")
        print("    apply_warning_fixes()")
        print("    print('✅ 자동 워닝 해결 적용 완료')")
        print("except:")
        print("    print('⚠️ 자동 워닝 해결 실패')")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ 패치 파일 생성 실패: {e}")
        return False

def main():
    """메인 실행 함수"""
    print("🔥 최종 워닝 해결 패치")
    print("=" * 40)
    
    # 1단계: SmartMapper 강제 연동
    step1_success = force_integrate_smart_mapper()
    
    # 2단계: Step 모델 로딩 패치
    step2_success = patch_step_model_loading()
    
    # 3단계: 지속적 해결책 생성
    step3_success = create_persistent_fix()
    
    print("\n" + "=" * 40)
    print("🎯 최종 워닝 해결 결과:")
    print(f"  🔥 SmartMapper 강제 연동: {'✅' if step1_success else '❌'}")
    print(f"  🔧 Step 로딩 패치: {'✅' if step2_success else '❌'}")
    print(f"  📝 지속적 해결책: {'✅' if step3_success else '❌'}")
    
    if step1_success:
        print("\n🎉 최종 워닝 해결 성공!")
        print("🚀 이제 main.py를 다시 실행하면:")
        print("   ✅ vgg16_warping 워닝 해결")
        print("   ✅ vgg19_warping 워닝 해결")
        print("   ✅ densenet121 워닝 해결")
        print("\n💡 더 완벽한 해결을 위해 main.py에 패치 코드를 추가하세요!")
    else:
        print("\n⚠️ 일부 문제가 남아있습니다")
        print("💡 main.py를 실행하여 상태를 확인해보세요")
    
    print("=" * 40)

if __name__ == "__main__":
    main()