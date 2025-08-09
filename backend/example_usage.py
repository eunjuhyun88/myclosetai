#!/usr/bin/env python3
"""
MyCloset-AI 새로운 아키텍처 사용 예시
=====================================

이 파일은 새로 구현된 model_architectures.py의 모델들을
실제 MyCloset-AI 시스템에서 사용하는 방법을 보여줍니다.

사용법:
    python example_usage.py
"""

import sys
import os
sys.path.append('.')

def example_1_basic_model_loading():
    """예시 1: 기본 모델 로딩"""
    print("🔧 예시 1: 기본 모델 로딩")
    print("=" * 50)
    
    try:
        from app.ai_pipeline.utils.model_loader import ModelLoader
        from app.ai_pipeline.utils.model_architectures import ModelArchitectureFactory
        
        # ModelLoader 초기화
        loader = ModelLoader()
        
        # Step별 모델 로딩
        steps = ['step_01', 'step_02', 'step_03', 'step_04', 'step_05', 'step_06', 'step_07', 'step_08']
        
        for step in steps:
            print(f"\n📁 {step} 모델 로딩:")
            try:
                model = loader.load_model_for_step(step)
                if model is not None:
                    print(f"   ✅ {step} 모델 로딩 성공")
                    # 모델 정보 출력
                    total_params = sum(p.numel() for p in model.parameters())
                    print(f"   📊 총 파라미터: {total_params:,}개")
                else:
                    print(f"   ❌ {step} 모델 로딩 실패")
            except Exception as e:
                print(f"   ❌ {step} 오류: {e}")
                
    except Exception as e:
        print(f"❌ 기본 모델 로딩 실패: {e}")

def example_2_direct_architecture_usage():
    """예시 2: 직접 아키텍처 사용"""
    print("\n🔧 예시 2: 직접 아키텍처 사용")
    print("=" * 50)
    
    try:
        from app.ai_pipeline.utils.model_architectures import (
            GMMModel, OpenPoseModel, TPSModel, RAFTModel,
            SAMModel, U2NetModel, RealESRGANModel, TOMModel,
            OOTDModel, CLIPModel, LPIPSModel, DeepLabV3PlusModel,
            MobileSAMModel, VITONHDModel, GFPGANModel
        )
        import torch
        
        # 테스트할 모델들
        test_models = [
            ('GMMModel', GMMModel()),
            ('OpenPoseModel', OpenPoseModel()),
            ('TPSModel', TPSModel()),
            ('RAFTModel', RAFTModel()),
            ('SAMModel', SAMModel()),
            ('U2NetModel', U2NetModel()),
            ('RealESRGANModel', RealESRGANModel()),
            ('TOMModel', TOMModel()),
            ('OOTDModel', OOTDModel()),
            ('CLIPModel', CLIPModel()),
            ('LPIPSModel', LPIPSModel()),
            ('DeepLabV3PlusModel', DeepLabV3PlusModel()),
            ('MobileSAMModel', MobileSAMModel()),
            ('VITONHDModel', VITONHDModel()),
            ('GFPGANModel', GFPGANModel())
        ]
        
        for model_name, model in test_models:
            try:
                print(f"\n🔍 {model_name} 테스트:")
                
                # 모델 정보
                total_params = sum(p.numel() for p in model.parameters())
                print(f"   📊 총 파라미터: {total_params:,}개")
                
                # Forward pass 테스트
                if model_name == 'LPIPSModel':
                    x = torch.randn(1, 3, 256, 256)
                    y = torch.randn(1, 3, 256, 256)
                    output = model(x, y)
                    print(f"   ✅ Forward pass: {output.shape}")
                elif model_name == 'VITONHDModel':
                    person = torch.randn(1, 3, 256, 256)
                    clothing = torch.randn(1, 3, 256, 256)
                    output = model(person, clothing)
                    print(f"   ✅ Forward pass: {type(output)} (딕셔너리)")
                elif model_name == 'GFPGANModel':
                    x = torch.randn(1, 3, 256, 256)
                    output = model(x)
                    print(f"   ✅ Forward pass: {output.shape}")
                else:
                    x = torch.randn(1, 3, 256, 256)
                    output = model(x)
                    print(f"   ✅ Forward pass: {output.shape}")
                    
            except Exception as e:
                print(f"   ❌ {model_name} 테스트 실패: {e}")
                
    except Exception as e:
        print(f"❌ 직접 아키텍처 사용 실패: {e}")

def example_3_checkpoint_loading():
    """예시 3: 체크포인트 로딩"""
    print("\n🔧 예시 3: 체크포인트 로딩")
    print("=" * 50)
    
    try:
        from app.ai_pipeline.utils.model_loader import ModelLoader
        import torch
        
        # ModelLoader 초기화
        loader = ModelLoader()
        
        # 실제 체크포인트 파일들
        checkpoint_tests = [
            ('step_04/gmm.pth', 'step_04'),
            ('step_02/openpose.pth', 'step_02'),
            ('step_04/tps.pth', 'step_04'),
            ('step_04/raft.pth', 'step_04')
        ]
        
        for checkpoint_path, step_type in checkpoint_tests:
            full_path = f"ai_models/{checkpoint_path}"
            if os.path.exists(full_path):
                print(f"\n📁 {checkpoint_path} 체크포인트 로딩:")
                try:
                    # 체크포인트 분석
                    analysis = loader.analyzer.analyze_checkpoint(full_path)
                    print(f"   📊 아키텍처 타입: {analysis.get('architecture_type', 'unknown')}")
                    print(f"   📊 모델 이름: {analysis.get('model_name', 'unknown')}")
                    
                    # 모델 생성
                    model = loader.creator.create_model_from_checkpoint(full_path, step_type)
                    if model is not None:
                        print(f"   ✅ 모델 생성 성공")
                        
                        # 체크포인트 로딩
                        checkpoint = torch.load(full_path, map_location='cpu')
                        if 'model' in checkpoint:
                            state_dict = checkpoint['model']
                        elif 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        else:
                            state_dict = checkpoint
                        
                        # 가중치 로딩 시도
                        try:
                            model.load_state_dict(state_dict, strict=False)
                            print(f"   ✅ 가중치 로딩 성공 (strict=False)")
                        except Exception as e:
                            print(f"   ⚠️ 가중치 로딩 실패: {e}")
                    else:
                        print(f"   ❌ 모델 생성 실패")
                        
                except Exception as e:
                    print(f"   ❌ 체크포인트 로딩 실패: {e}")
            else:
                print(f"\n📁 {checkpoint_path} 파일 없음")
                
    except Exception as e:
        print(f"❌ 체크포인트 로딩 실패: {e}")

def example_4_step_integration():
    """예시 4: Step 클래스와 통합"""
    print("\n🔧 예시 4: Step 클래스와 통합")
    print("=" * 50)
    
    try:
        from app.ai_pipeline.utils.model_loader import ModelLoader, StepModelInterface
        
        # ModelLoader 초기화
        loader = ModelLoader()
        
        # Step별 인터페이스 생성
        steps = ['step_01', 'step_02', 'step_03', 'step_04']
        
        for step in steps:
            print(f"\n📁 {step} Step 인터페이스:")
            try:
                # Step 인터페이스 생성
                step_interface = loader.create_step_interface(step)
                
                # 모델 로딩
                success = step_interface.load_primary_model()
                if success:
                    print(f"   ✅ {step} 모델 로딩 성공")
                    
                    # 모델 가져오기
                    model = step_interface.get_model()
                    if model is not None:
                        total_params = sum(p.numel() for p in model.parameters())
                        print(f"   📊 총 파라미터: {total_params:,}개")
                        
                        # 추론 테스트
                        try:
                            import torch
                            x = torch.randn(1, 3, 256, 256)
                            result = step_interface.run_inference(x)
                            print(f"   ✅ 추론 성공: {type(result)}")
                        except Exception as e:
                            print(f"   ⚠️ 추론 실패: {e}")
                    else:
                        print(f"   ❌ 모델 가져오기 실패")
                else:
                    print(f"   ❌ {step} 모델 로딩 실패")
                    
            except Exception as e:
                print(f"   ❌ {step} Step 인터페이스 실패: {e}")
                
    except Exception as e:
        print(f"❌ Step 통합 실패: {e}")

def main():
    """메인 함수"""
    print("🚀 MyCloset-AI 새로운 아키텍처 사용 예시")
    print("=" * 60)
    
    # 예시 실행
    example_1_basic_model_loading()
    example_2_direct_architecture_usage()
    example_3_checkpoint_loading()
    example_4_step_integration()
    
    print("\n🎉 모든 예시 실행 완료!")
    print("\n💡 사용 팁:")
    print("   1. ModelLoader를 사용하여 Step별 모델 로딩")
    print("   2. ModelArchitectureFactory로 직접 모델 생성")
    print("   3. 체크포인트 파일과 함께 사용")
    print("   4. Step 클래스와 완전 통합")

if __name__ == "__main__":
    main()
