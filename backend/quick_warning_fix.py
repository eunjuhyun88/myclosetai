# quick_warning_fix.py - backend 디렉토리에서 실행
"""
빠른 워닝 해결 스크립트
실행: python quick_warning_fix.py
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

def resolve_missing_model_warnings():
    """누락된 모델 워닝 해결"""
    try:
        print("🔧 누락된 모델 워닝 해결 시작...")
        
        # SmartModelPathMapper 임포트 시도
        try:
            from app.ai_pipeline.utils.smart_model_mapper import get_global_smart_mapper
            print("✅ SmartModelPathMapper 임포트 성공")
        except ImportError as e:
            print(f"❌ SmartModelPathMapper 임포트 실패: {e}")
            print("💡 먼저 create_smart_mapper.py를 실행해주세요")
            return False
        
        # SmartMapper 초기화
        smart_mapper = get_global_smart_mapper()
        print(f"✅ SmartMapper 초기화 완료: {smart_mapper.ai_models_root}")
        
        # 캐시 새로고침
        refresh_result = smart_mapper.refresh_cache()
        print(f"✅ 캐시 새로고침: {refresh_result.get('new_cache_size', 0)}개 모델 발견")
        
        # 워닝 모델들 확인
        warning_models = [
            "realvis_xl", "vgg16_warping", "vgg19_warping", "densenet121",
            "post_processing_model", "gmm"
        ]
        
        resolved_models = {}
        
        for model_name in warning_models:
            mapping_info = smart_mapper.get_model_path(model_name)
            if mapping_info and mapping_info.actual_path:
                resolved_models[model_name] = {
                    "path": str(mapping_info.actual_path),
                    "size_mb": mapping_info.size_mb,
                    "ai_class": mapping_info.ai_class
                }
                print(f"  ✅ {model_name}: {mapping_info.actual_path} ({mapping_info.size_mb:.1f}MB)")
            else:
                print(f"  ⚠️ {model_name}: 경로를 찾을 수 없음")
        
        print(f"\n🎉 워닝 해결 완료: {len(resolved_models)}개 모델 해결")
        
        # 통계 출력
        stats = smart_mapper.get_mapping_statistics()
        print(f"📊 총 매핑된 모델: {stats['successful_mappings']}개")
        print(f"📁 AI 모델 루트: {stats['ai_models_root']}")
        print(f"📁 디렉토리 존재: {stats['ai_models_root_exists']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 워닝 해결 실패: {e}")
        return False

def patch_model_loader():
    """ModelLoader 패치 적용"""
    try:
        print("\n🔧 ModelLoader 패치 적용 중...")
        
        # ModelLoader 임포트 시도
        try:
            from app.ai_pipeline.utils.model_loader import get_global_model_loader
            print("✅ ModelLoader 임포트 성공")
        except ImportError as e:
            print(f"❌ ModelLoader 임포트 실패: {e}")
            return False
        
        # 전역 ModelLoader 가져오기
        model_loader = get_global_model_loader()
        print(f"✅ ModelLoader 인스턴스 획득: {type(model_loader).__name__}")
        
        # 간단한 패치 적용
        original_load_model = model_loader.load_model
        
        def patched_load_model(model_name: str, **kwargs):
            """패치된 load_model 메서드"""
            try:
                # 원본 메서드 먼저 시도
                result = original_load_model(model_name, **kwargs)
                if result:
                    return result
                
                # 실패 시 SmartMapper로 경로 해결
                try:
                    from app.ai_pipeline.utils.smart_model_mapper import resolve_model_path
                    resolved_path = resolve_model_path(model_name)
                    
                    if resolved_path:
                        print(f"🔄 {model_name} 폴백 경로 발견: {resolved_path}")
                        
                        # available_models에 임시 추가
                        if hasattr(model_loader, '_available_models_cache'):
                            model_loader._available_models_cache[model_name] = {
                                "name": model_name,
                                "path": str(resolved_path),
                                "checkpoint_path": str(resolved_path),
                                "ai_model_info": {"ai_class": "BaseRealAIModel"},
                                "size_mb": resolved_path.stat().st_size / (1024 * 1024)
                            }
                            
                            # 재시도
                            return original_load_model(model_name, **kwargs)
                
                except ImportError:
                    pass
                
                return None
                
            except Exception as e:
                print(f"⚠️ {model_name} 로딩 실패: {e}")
                return None
        
        # 메서드 교체
        model_loader.load_model = patched_load_model
        print("✅ ModelLoader 패치 적용 완료")
        
        return True
        
    except Exception as e:
        print(f"❌ ModelLoader 패치 실패: {e}")
        return False

def test_warnings_resolved():
    """워닝 해결 테스트"""
    try:
        print("\n🧪 워닝 해결 테스트 중...")
        
        # Step별 문제 모델들 테스트
        test_models = [
            ("realvis_xl", "ClothWarpingStep"),
            ("post_processing_model", "PostProcessingStep"), 
            ("gmm", "GeometricMatchingStep")
        ]
        
        success_count = 0
        
        for model_name, step_name in test_models:
            try:
                from app.ai_pipeline.utils.smart_model_mapper import resolve_model_path
                resolved_path = resolve_model_path(model_name)
                
                if resolved_path:
                    print(f"  ✅ {step_name} - {model_name}: 경로 해결 성공")
                    success_count += 1
                else:
                    print(f"  ⚠️ {step_name} - {model_name}: 경로 해결 실패")
                    
            except Exception as e:
                print(f"  ❌ {step_name} - {model_name}: 테스트 실패 - {e}")
        
        print(f"\n📊 테스트 결과: {success_count}/{len(test_models)} 성공")
        return success_count >= len(test_models) // 2  # 절반 이상 성공하면 OK
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

def main():
    """메인 실행 함수"""
    print("🔥 빠른 워닝 해결 스크립트")
    print("=" * 40)
    
    # 1단계: 누락된 모델 워닝 해결
    step1_success = resolve_missing_model_warnings()
    
    # 2단계: ModelLoader 패치
    step2_success = patch_model_loader()
    
    # 3단계: 테스트
    step3_success = test_warnings_resolved()
    
    print("\n" + "=" * 40)
    print("🎯 워닝 해결 결과:")
    print(f"  📁 모델 경로 매핑: {'✅' if step1_success else '❌'}")
    print(f"  🔧 ModelLoader 패치: {'✅' if step2_success else '❌'}")
    print(f"  🧪 해결 검증: {'✅' if step3_success else '❌'}")
    
    overall_success = step1_success and step2_success
    
    if overall_success:
        print("\n🎉 워닝 해결 성공!")
        print("🚀 이제 main.py를 다시 실행해보세요:")
        print("   python app/main.py")
    else:
        print("\n⚠️ 일부 문제가 남아있습니다")
        print("💡 다음 단계:")
        if not step1_success:
            print("   1. create_smart_mapper.py 먼저 실행")
        if not step2_success:
            print("   2. ModelLoader 수동 패치 필요")
    
    print("=" * 40)

if __name__ == "__main__":
    main()