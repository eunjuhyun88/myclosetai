#!/usr/bin/env python3
"""
MyCloset-AI 새로운 아키텍처 간단 사용 예시
==========================================

실제 프로젝트에서 바로 사용할 수 있는 간단한 예시들입니다.
"""

import sys
import os
sys.path.append('.')

def example_1_direct_model_usage():
    """예시 1: 직접 모델 사용 (가장 간단)"""
    print("🔧 예시 1: 직접 모델 사용")
    print("=" * 40)
    
    try:
        # 성공한 모델들만 사용
        from app.ai_pipeline.utils.model_architectures import (
            OpenPoseModel, RAFTModel, SAMModel, U2NetModel,
            RealESRGANModel, LPIPSModel, DeepLabV3PlusModel,
            VITONHDModel, GFPGANModel
        )
        import torch
        
        # 테스트할 모델들 (성공한 것들만)
        working_models = [
            ('OpenPoseModel', OpenPoseModel()),
            ('RAFTModel', RAFTModel()),
            ('SAMModel', SAMModel()),
            ('U2NetModel', U2NetModel()),
            ('RealESRGANModel', RealESRGANModel()),
            ('LPIPSModel', LPIPSModel()),
            ('DeepLabV3PlusModel', DeepLabV3PlusModel()),
            ('VITONHDModel', VITONHDModel()),
            ('GFPGANModel', GFPGANModel())
        ]
        
        for model_name, model in working_models:
            try:
                print(f"\n🔍 {model_name}:")
                
                # 모델 정보
                total_params = sum(p.numel() for p in model.parameters())
                print(f"   📊 파라미터: {total_params:,}개")
                
                # Forward pass 테스트
                if model_name == 'LPIPSModel':
                    x = torch.randn(1, 3, 256, 256)
                    y = torch.randn(1, 3, 256, 256)
                    output = model(x, y)
                    print(f"   ✅ 출력: {output.shape}")
                elif model_name == 'VITONHDModel':
                    person = torch.randn(1, 3, 256, 256)
                    clothing = torch.randn(1, 3, 256, 256)
                    output = model(person, clothing)
                    print(f"   ✅ 출력: {type(output)} (딕셔너리)")
                else:
                    x = torch.randn(1, 3, 256, 256)
                    output = model(x)
                    print(f"   ✅ 출력: {output.shape}")
                    
            except Exception as e:
                print(f"   ❌ 오류: {e}")
                
    except Exception as e:
        print(f"❌ 직접 모델 사용 실패: {e}")

def example_2_step_integration():
    """예시 2: Step 클래스에 통합"""
    print("\n🔧 예시 2: Step 클래스에 통합")
    print("=" * 40)
    
    try:
        # Step 클래스에서 모델 사용 예시
        from app.ai_pipeline.utils.model_architectures import (
            OpenPoseModel, SAMModel, U2NetModel, RealESRGANModel
        )
        import torch
        
        class ExampleStep:
            def __init__(self):
                # 모델 초기화
                self.pose_model = OpenPoseModel()
                self.segmentation_model = SAMModel()
                self.enhancement_model = RealESRGANModel()
                
            def process_image(self, image):
                """이미지 처리 파이프라인"""
                print("🔄 이미지 처리 시작...")
                
                # 1. 포즈 추정
                pose_result = self.pose_model(image)
                print(f"   📍 포즈 추정 완료: {pose_result.shape}")
                
                # 2. 세그멘테이션
                seg_result = self.segmentation_model(image)
                print(f"   🎯 세그멘테이션 완료: {seg_result.shape}")
                
                # 3. 이미지 향상
                enhanced_result = self.enhancement_model(image)
                print(f"   ✨ 이미지 향상 완료: {enhanced_result.shape}")
                
                return {
                    'pose': pose_result,
                    'segmentation': seg_result,
                    'enhanced': enhanced_result
                }
        
        # 사용 예시
        step = ExampleStep()
        test_image = torch.randn(1, 3, 256, 256)
        result = step.process_image(test_image)
        
        print(f"\n✅ Step 처리 완료!")
        print(f"   📊 결과 키: {list(result.keys())}")
        
    except Exception as e:
        print(f"❌ Step 통합 실패: {e}")

def example_3_checkpoint_loading():
    """예시 3: 체크포인트 로딩"""
    print("\n🔧 예시 3: 체크포인트 로딩")
    print("=" * 40)
    
    try:
        from app.ai_pipeline.utils.model_architectures import OpenPoseModel
        import torch
        
        # 모델 생성
        model = OpenPoseModel()
        
        # 체크포인트 파일 경로
        checkpoint_path = "ai_models/step_02/openpose.pth"
        
        if os.path.exists(checkpoint_path):
            print(f"📁 체크포인트 로딩: {checkpoint_path}")
            
            # 체크포인트 로드
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # state_dict 추출
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 가중치 로딩
            try:
                model.load_state_dict(state_dict, strict=False)
                print("   ✅ 가중치 로딩 성공!")
                
                # 테스트 추론
                test_input = torch.randn(1, 3, 256, 256)
                with torch.no_grad():
                    output = model(test_input)
                print(f"   ✅ 추론 성공: {output.shape}")
                
            except Exception as e:
                print(f"   ⚠️ 가중치 로딩 실패: {e}")
        else:
            print(f"📁 체크포인트 파일 없음: {checkpoint_path}")
            
    except Exception as e:
        print(f"❌ 체크포인트 로딩 실패: {e}")

def example_4_api_integration():
    """예시 4: API에 통합"""
    print("\n🔧 예시 4: API에 통합")
    print("=" * 40)
    
    try:
        from app.ai_pipeline.utils.model_architectures import (
            OpenPoseModel, SAMModel, RealESRGANModel
        )
        import torch
        
        class AIProcessor:
            def __init__(self):
                self.models = {
                    'pose': OpenPoseModel(),
                    'segmentation': SAMModel(),
                    'enhancement': RealESRGANModel()
                }
                
            def process_request(self, request_type, image):
                """API 요청 처리"""
                if request_type == 'pose':
                    return self.models['pose'](image)
                elif request_type == 'segmentation':
                    return self.models['segmentation'](image)
                elif request_type == 'enhancement':
                    return self.models['enhancement'](image)
                else:
                    raise ValueError(f"지원하지 않는 요청 타입: {request_type}")
        
        # API 사용 예시
        processor = AIProcessor()
        test_image = torch.randn(1, 3, 256, 256)
        
        # 포즈 추정 요청
        pose_result = processor.process_request('pose', test_image)
        print(f"📍 포즈 추정 결과: {pose_result.shape}")
        
        # 세그멘테이션 요청
        seg_result = processor.process_request('segmentation', test_image)
        print(f"🎯 세그멘테이션 결과: {seg_result.shape}")
        
        # 이미지 향상 요청
        enhanced_result = processor.process_request('enhancement', test_image)
        print(f"✨ 이미지 향상 결과: {enhanced_result.shape}")
        
    except Exception as e:
        print(f"❌ API 통합 실패: {e}")

def main():
    """메인 함수"""
    print("🚀 MyCloset-AI 새로운 아키텍처 간단 사용 예시")
    print("=" * 60)
    
    # 예시 실행
    example_1_direct_model_usage()
    example_2_step_integration()
    example_3_checkpoint_loading()
    example_4_api_integration()
    
    print("\n🎉 모든 예시 실행 완료!")
    print("\n💡 실제 사용 방법:")
    print("   1. from app.ai_pipeline.utils.model_architectures import 모델명")
    print("   2. model = 모델명()")
    print("   3. output = model(input)")
    print("   4. 체크포인트가 있다면 model.load_state_dict() 사용")

if __name__ == "__main__":
    main()
