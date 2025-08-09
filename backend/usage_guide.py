#!/usr/bin/env python3
"""
🎯 MyCloset-AI Model Architectures 사용 가이드
================================================================================
✅ 기본 모델 사용법
✅ 체크포인트 로딩
✅ 파이프라인 실행
✅ 성능 모니터링
✅ 모델 관리
================================================================================
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
import psutil

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent))

def basic_model_usage():
    """기본 모델 사용법 - 전처리 강화"""
    print("\n" + "="*60)
    print("🎯 기본 모델 사용법 (전처리 강화)")
    print("="*60)
    
    try:
        from app.ai_pipeline.utils.model_architectures import (
            OpenPoseModel, CompleteModelWrapper, OpenPosePreprocessor
        )
        
        # 1. 직접 모델 사용 (전처리 포함)
        print("\n📌 1. 직접 모델 사용 (전처리 포함)")
        
        # 더미 이미지 생성 (HWC 형태)
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"   입력 이미지 형태: {dummy_image.shape}")
        
        # 전처리기 생성
        preprocessor = OpenPosePreprocessor()
        
        # 이미지 전처리
        processed_tensor = preprocessor(dummy_image)
        print(f"   전처리 후 텐서 형태: {processed_tensor.shape}")
        
        # 모델 생성 및 추론
        model = OpenPoseModel()
        with torch.no_grad():
            output = model(processed_tensor)
        
        print(f"   모델 출력 형태: {output.shape}")
        print("   ✅ 직접 모델 사용 성공")
        
        # 2. CompleteModelWrapper 사용
        print("\n📌 2. CompleteModelWrapper 사용")
        
        # 래퍼 생성
        wrapper = CompleteModelWrapper(model, 'openpose')
        
        # 다양한 입력 형태로 테스트
        test_inputs = [
            dummy_image,  # NumPy 배열
            "test_image.jpg",  # 파일 경로 (실제 파일이 없으므로 예외 발생 예상)
        ]
        
        for i, test_input in enumerate(test_inputs):
            try:
                if isinstance(test_input, str):
                    print(f"   테스트 {i+1}: 파일 경로 입력 (예외 발생 예상)")
                else:
                    print(f"   테스트 {i+1}: NumPy 배열 입력")
                
                result = wrapper(test_input)
                print(f"   결과 형태: {type(result)}")
                if isinstance(result, dict):
                    print(f"   결과 키: {list(result.keys())}")
                print("   ✅ CompleteModelWrapper 사용 성공")
                
            except Exception as e:
                if isinstance(test_input, str):
                    print(f"   ⚠️ 예상된 오류 (파일 없음): {e}")
                else:
                    print(f"   ❌ 오류: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 기본 모델 사용법 실패: {e}")
        return False

def checkpoint_loading_usage():
    """2. 체크포인트 로딩 사용법"""
    print("\n🎯 2. 체크포인트 로딩 사용법")
    print("=" * 50)
    
    try:
        from app.ai_pipeline.utils.model_architectures import (
            ModelArchitectureFactory,
            AdvancedKeyMapper
        )
        
        # 2-1. 체크포인트 분석 및 모델 생성
        print("\n📌 2-1. 체크포인트 분석 및 모델 생성")
        
        # 체크포인트 파일 경로 (실제 파일이 있다면)
        checkpoint_path = "./models/openpose_checkpoint.pth"
        
        # 체크포인트가 있는 경우에만 실행
        if os.path.exists(checkpoint_path):
            # 체크포인트 분석
            analysis = {
                'architecture_type': 'openpose',
                'model_name': 'openpose',
                'checkpoint_path': checkpoint_path
            }
            
            # 모델 생성
            model = ModelArchitectureFactory.create_model_from_analysis(analysis)
            
            if model:
                print(f"✅ 체크포인트 로딩 성공")
                print(f"   - 모델 타입: {analysis['architecture_type']}")
                print(f"   - 모델 이름: {analysis['model_name']}")
            else:
                print(f"⚠️ 체크포인트 파일이 없어서 더미 분석으로 진행")
        else:
            print(f"⚠️ 체크포인트 파일이 없어서 더미 분석으로 진행")
        
        # 2-2. 고급 키 매핑 사용
        print("\n📌 2-2. 고급 키 매핑 사용")
        key_mapper = AdvancedKeyMapper()
        
        # 더미 체크포인트 생성
        dummy_checkpoint = {
            'model_state_dict': {
                'conv1.weight': torch.randn(64, 3, 7, 7),
                'conv1.bias': torch.randn(64),
                'bn1.weight': torch.randn(64),
                'bn1.bias': torch.randn(64)
            }
        }
        
        # 더미 모델 생성
        from app.ai_pipeline.utils.model_architectures import OpenPoseModel
        dummy_model = OpenPoseModel()
        
        # 키 매핑 테스트
        mapping_success = key_mapper.map_checkpoint(dummy_checkpoint, dummy_model, 'openpose')
        
        print(f"✅ 키 매핑 테스트 완료")
        print(f"   - 매핑 성공: {mapping_success}")
        
        return True
        
    except Exception as e:
        print(f"❌ 체크포인트 로딩 실패: {e}")
        return False

def pipeline_usage():
    """파이프라인 실행 - 데이터 검증 강화"""
    print("\n" + "="*60)
    print("🎯 파이프라인 실행 (데이터 검증 강화)")
    print("="*60)
    
    try:
        from app.ai_pipeline.utils.model_architectures import (
            IntegratedInferenceEngine, CompleteModelWrapper, 
            OpenPoseModel, HRNetPoseModel, GraphonomyModel
        )
        
        # 엔진 생성
        engine = IntegratedInferenceEngine()
        
        # 더미 모델들 생성 및 등록
        print("\n📌 모델 등록")
        
        # OpenPose 모델
        openpose_model = CompleteModelWrapper(OpenPoseModel(), 'openpose')
        engine.register_model('pose_estimation', openpose_model)
        print("   ✅ pose_estimation 모델 등록")
        
        # HRNet 모델
        hrnet_model = CompleteModelWrapper(HRNetPoseModel(), 'hrnet')
        engine.register_model('pose_estimation_hrnet', hrnet_model)
        print("   ✅ pose_estimation_hrnet 모델 등록")
        
        # Graphonomy 모델
        graphonomy_model = CompleteModelWrapper(GraphonomyModel(), 'graphonomy')
        engine.register_model('human_parsing', graphonomy_model)
        print("   ✅ human_parsing 모델 등록")
        
        # 파이프라인 생성
        print("\n📌 파이프라인 생성")
        
        # 단일 모델 파이프라인
        engine.create_pipeline('single_pose', ['pose_estimation'])
        print("   ✅ single_pose 파이프라인 생성")
        
        # 복합 파이프라인
        engine.create_pipeline('pose_and_parsing', ['pose_estimation', 'human_parsing'])
        print("   ✅ pose_and_parsing 파이프라인 생성")
        
        # 더미 이미지 생성
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 파이프라인 실행 테스트
        print("\n📌 파이프라인 실행 테스트")
        
        # 1. 단일 모델 파이프라인
        print("\n   🔄 single_pose 파이프라인 실행")
        try:
            result = engine.run_pipeline('single_pose', dummy_image)
            print(f"   결과: {result['success']}")
            if result['success']:
                print(f"   실행 시간: {result['total_time']:.2f}초")
                print(f"   결과 키: {list(result['results'].keys())}")
            else:
                print(f"   오류: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"   ❌ 실행 오류: {e}")
        
        # 2. 복합 파이프라인 (데이터 검증 강화)
        print("\n   🔄 pose_and_parsing 파이프라인 실행")
        try:
            result = engine.run_pipeline('pose_and_parsing', dummy_image)
            print(f"   결과: {result['success']}")
            if result['success']:
                print(f"   실행 시간: {result['total_time']:.2f}초")
                print(f"   단계별 결과:")
                for model_name, step_result in result['results'].items():
                    print(f"     - {model_name}: {type(step_result)}")
            else:
                print(f"   오류: {result.get('error', 'Unknown error')}")
                print(f"   실패 단계: {result.get('failed_step', 'Unknown')}")
        except Exception as e:
            print(f"   ❌ 실행 오류: {e}")
        
        # 3. 잘못된 입력 테스트 (검증 강화 확인)
        print("\n   🔄 잘못된 입력 테스트 (검증 강화)")
        try:
            result = engine.run_pipeline('single_pose', None)  # None 입력
            print(f"   결과: {result['success']}")
            if not result['success']:
                print(f"   예상된 오류: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"   ✅ 검증 오류 (예상됨): {e}")
        
        # 성능 리포트
        print("\n📌 성능 리포트")
        report = engine.get_performance_report()
        print(f"   등록된 모델: {len(report['registered_models'])}")
        print(f"   등록된 파이프라인: {len(report['available_pipelines'])}")
        print(f"   캐시 크기: {report['cache_size']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 파이프라인 실행 실패: {e}")
        return False

def monitoring_usage():
    """4. 성능 모니터링 사용법"""
    print("\n🎯 4. 성능 모니터링 사용법")
    print("=" * 50)
    
    try:
        from app.ai_pipeline.utils.model_architectures import RealTimePerformanceMonitor
        
        # 4-1. 성능 모니터 생성
        print("\n📌 4-1. 성능 모니터 생성")
        monitor = RealTimePerformanceMonitor()
        
        print(f"✅ 성능 모니터 생성 완료")
        print(f"   - 기본 임계값: {monitor.thresholds}")
        
        # 4-2. 모니터링 시작
        print("\n📌 4-2. 모니터링 시작")
        monitor_id = monitor.start_monitoring('test_model', 'inference')
        
        print(f"✅ 모니터링 시작 완료")
        print(f"   - 모니터 ID: {monitor_id}")
        
        # 4-3. 메트릭 업데이트
        print("\n📌 4-3. 메트릭 업데이트")
        monitor.update_metrics(monitor_id, accuracy=0.85, execution_time=2.5)
        
        print(f"✅ 메트릭 업데이트 완료")
        
        # 4-4. 모니터링 종료
        print("\n📌 4-4. 모니터링 종료")
        final_result = monitor.stop_monitoring(monitor_id, {'accuracy': 0.85})
        
        print(f"✅ 모니터링 종료 완료")
        print(f"   - 실행 시간: {final_result['execution_time']:.2f}초")
        print(f"   - 메모리 사용량: {final_result['memory_usage']:.1f}%")
        print(f"   - CPU 사용량: {final_result['cpu_usage']:.1f}%")
        
        # 4-5. 성능 요약 확인
        print("\n📌 4-5. 성능 요약 확인")
        summary = monitor.get_performance_summary('test_model')
        
        print(f"✅ 성능 요약 생성 완료")
        print(f"   - 총 실행 횟수: {summary['total_runs']}")
        print(f"   - 평균 실행 시간: {summary['avg_execution_time']:.2f}초")
        print(f"   - 평균 메모리 사용량: {summary['avg_memory_usage']:.1f}%")
        
        # 4-6. 시스템 상태 확인
        print("\n📌 4-6. 시스템 상태 확인")
        system_status = monitor.get_system_status()
        
        print(f"✅ 시스템 상태 확인 완료")
        print(f"   - 메모리 사용량: {system_status['memory']['percent']:.1f}%")
        print(f"   - CPU 사용량: {system_status['cpu']['percent']:.1f}%")
        print(f"   - 디스크 사용량: {system_status['disk']['percent']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ 성능 모니터링 실패: {e}")
        return False

def model_management_usage():
    """5. 모델 관리 사용법"""
    print("\n🎯 5. 모델 관리 사용법")
    print("=" * 50)
    
    try:
        from app.ai_pipeline.utils.model_architectures import AdvancedModelManager
        
        # 5-1. 모델 관리자 생성
        print("\n📌 5-1. 모델 관리자 생성")
        manager = AdvancedModelManager("./models")
        
        print(f"✅ 모델 관리자 생성 완료")
        print(f"   - 기본 경로: {manager.base_path}")
        print(f"   - 자동 관리 설정: {manager.auto_management}")
        
        # 5-2. 모델 등록
        print("\n📌 5-2. 모델 등록")
        model_info = manager.register_model(
            'openpose_model',
            './models/openpose.pth',
            '1.0.0',
            dependencies=['torch', 'numpy'],
            metadata={'description': 'OpenPose 포즈 추정 모델', 'author': 'CMU'}
        )
        
        print(f"✅ 모델 등록 완료")
        print(f"   - 모델 이름: {model_info['name']}")
        print(f"   - 버전: {model_info['version']}")
        print(f"   - 상태: {model_info['state']}")
        print(f"   - 의존성: {model_info['dependencies']}")
        
        # 5-3. 모델 정보 조회
        print("\n📌 5-3. 모델 정보 조회")
        retrieved_info = manager.get_model('openpose_model')
        
        print(f"✅ 모델 정보 조회 완료")
        print(f"   - 사용 횟수: {retrieved_info['usage_count']}")
        print(f"   - 마지막 사용: {retrieved_info['last_used']}")
        
        # 5-4. 백업 생성
        print("\n📌 5-4. 백업 생성")
        backup_info = manager.create_backup('openpose_model', 'initial_backup')
        
        print(f"✅ 백업 생성 완료")
        print(f"   - 백업 이름: {backup_info['backup_name']}")
        print(f"   - 원본 버전: {backup_info['original_version']}")
        
        # 5-5. 모델 업데이트
        print("\n📌 5-5. 모델 업데이트")
        updated_info = manager.update_model(
            'openpose_model',
            './models/openpose_v2.pth',
            '2.0.0',
            changelog='성능 개선 및 정확도 향상'
        )
        
        print(f"✅ 모델 업데이트 완료")
        print(f"   - 새 버전: {updated_info['version']}")
        print(f"   - 변경 로그: {updated_info['changelog']}")
        
        # 5-6. 버전 히스토리 확인
        print("\n📌 5-6. 버전 히스토리 확인")
        versions = manager.get_model_versions('openpose_model')
        
        print(f"✅ 버전 히스토리 확인 완료")
        print(f"   - 버전 목록: {versions}")
        
        # 5-7. 모델 통계 확인
        print("\n📌 5-7. 모델 통계 확인")
        stats = manager.get_model_statistics('openpose_model')
        
        print(f"✅ 모델 통계 확인 완료")
        print(f"   - 사용 횟수: {stats['usage_count']}")
        print(f"   - 백업 수: {stats['backup_count']}")
        print(f"   - 버전 수: {stats['version_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 관리 실패: {e}")
        return False

def complete_workflow_example():
    """완전한 워크플로우 예제 - 통합 시스템"""
    print("\n" + "="*60)
    print("🎯 완전한 워크플로우 예제 (통합 시스템)")
    print("="*60)
    
    try:
        from app.ai_pipeline.utils.model_architectures import (
            IntegratedInferenceEngine, RealTimePerformanceMonitor, 
            AdvancedModelManager, CompleteModelWrapper, OpenPoseModel
        )
        
        # 1. 시스템 초기화
        print("\n📌 1. 시스템 초기화")
        
        # 엔진 생성
        engine = IntegratedInferenceEngine()
        
        # 모니터 생성
        monitor = RealTimePerformanceMonitor()
        
        # 모델 관리자 생성
        manager = AdvancedModelManager("./models")
        
        print("   ✅ 시스템 초기화 완료")
        
        # 2. 모델 등록 및 관리
        print("\n📌 2. 모델 등록 및 관리")
        
        # 모델 등록
        openpose_model = CompleteModelWrapper(OpenPoseModel(), 'openpose')
        engine.register_model('pose_estimation', openpose_model)
        
        # 모델 관리자에 등록
        manager.register_model(
            'pose_estimation', 
            './models/openpose_model.pth', 
            '1.0.0',
            dependencies=['torch', 'numpy'],
            metadata={'type': 'pose_estimation', 'framework': 'pytorch'}
        )
        
        print("   ✅ 모델 등록 및 관리 완료")
        
        # 3. 파이프라인 생성
        print("\n📌 3. 파이프라인 생성")
        
        engine.create_pipeline('fashion_analysis', ['pose_estimation'])
        print("   ✅ fashion_analysis 파이프라인 생성")
        
        # 4. 성능 모니터링 시작
        print("\n📌 4. 성능 모니터링 시작")
        
        monitor_id = monitor.start_monitoring('fashion_analysis', 'pipeline_execution')
        print(f"   모니터링 ID: {monitor_id}")
        
        # 5. 파이프라인 실행
        print("\n📌 5. 파이프라인 실행")
        
        # 더미 이미지 생성
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 실행
        result = engine.run_pipeline('fashion_analysis', dummy_image)
        
        # 6. 성능 모니터링 종료
        print("\n📌 6. 성능 모니터링 종료")
        
        final_metrics = monitor.stop_monitoring(monitor_id, {
            'execution_time': result.get('total_time', 0),
            'success': result.get('success', False),
            'memory_usage': psutil.virtual_memory().percent / 100
        })
        
        print(f"   최종 메트릭: {final_metrics}")
        
        # 7. 결과 분석
        print("\n📌 7. 결과 분석")
        
        print(f"   파이프라인 성공: {result.get('success', False)}")
        if result.get('success'):
            print(f"   총 실행 시간: {result.get('total_time', 0):.2f}초")
            print(f"   결과 키: {list(result.get('results', {}).keys())}")
        
        # 8. 시스템 상태 확인
        print("\n📌 8. 시스템 상태 확인")
        
        # 성능 요약
        performance_summary = monitor.get_performance_summary()
        print(f"   성능 요약: {len(performance_summary)} 항목")
        
        # 시스템 상태
        system_status = monitor.get_system_status()
        print(f"   시스템 상태: CPU {system_status['cpu']['percent']:.1f}%, 메모리 {system_status['memory']['percent']:.1f}%")
        
        # 🔥 ModelLoader 통합 상태 확인
        print("\n📌 ModelLoader 통합 상태 확인")
        try:
            from app.ai_pipeline.utils.model_loader import get_global_model_loader, initialize_global_model_loader
            
            # ModelLoader 초기화
            success = initialize_global_model_loader()
            if success:
                model_loader = get_global_model_loader()
                if model_loader:
                    print("   ✅ ModelLoader Central Hub 통합 성공")
                    
                    # 사용 가능한 모델 확인
                    if hasattr(model_loader, 'list_available_models'):
                        available_models = model_loader.list_available_models()
                        print(f"   - 사용 가능한 모델: {len(available_models)}개")
                        
                        # 모델별 분류
                        step_models = {}
                        for model in available_models:
                            step_class = model.get('step_class', 'Unknown')
                            if step_class not in step_models:
                                step_models[step_class] = 0
                            step_models[step_class] += 1
                        
                        print("   - Step별 모델 분포:")
                        for step_class, count in step_models.items():
                            print(f"     * {step_class}: {count}개")
                    
                    print(f"   - 디바이스: {model_loader.device}")
                else:
                    print("   ❌ ModelLoader 인스턴스 생성 실패")
            else:
                print("   ❌ ModelLoader 초기화 실패")
                
        except Exception as e:
            print(f"   ❌ ModelLoader 통합 확인 실패: {e}")
        
        # 모델 통계
        model_stats = manager.get_model_statistics()
        print(f"   등록된 모델: {len(model_stats)}")
        
        print("\n🎉 완전한 워크플로우 성공!")
        return True
        
    except Exception as e:
        print(f"❌ 완전한 워크플로우 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 실행 함수"""
    print("🚀 MyCloset-AI Model Architectures 사용 가이드")
    print("=" * 60)
    
    # 각 사용법 테스트
    tests = [
        ("기본 모델 사용법", basic_model_usage),
        ("체크포인트 로딩", checkpoint_loading_usage),
        ("파이프라인 실행", pipeline_usage),
        ("성능 모니터링", monitoring_usage),
        ("모델 관리", model_management_usage),
        ("완전한 워크플로우", complete_workflow_example)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 실행 중 오류: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 사용 가이드 실행 결과")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ 성공" if result else "❌ 실패"
        print(f"   {test_name}: {status}")
    
    success_count = sum(1 for _, result in results if result)
    total_count = len(results)
    
    print(f"\n🎯 전체 결과: {success_count}/{total_count} 성공")
    
    if success_count == total_count:
        print("🎉 모든 사용법이 성공적으로 실행되었습니다!")
    else:
        print("⚠️ 일부 사용법에서 문제가 발생했습니다.")

if __name__ == "__main__":
    main()
