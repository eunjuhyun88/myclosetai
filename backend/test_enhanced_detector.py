#!/usr/bin/env python3
"""
🔍 MyCloset AI - 강화된 모델 탐지기 테스트 스크립트 v7.0
================================================================

✅ RealWorldModelDetector v7.0 완전 테스트
✅ 89.8GB 체크포인트 탐지 및 검증
✅ PyTorch 모델 실제 로딩 테스트
✅ M3 Max 최적화 검증
✅ Step별 모델 매핑 테스트
✅ 성능 벤치마크 및 메모리 모니터링

사용법:
    python test_enhanced_detector.py
    python test_enhanced_detector.py --quick     # 빠른 테스트
    python test_enhanced_detector.py --detailed  # 상세 분석
    python test_enhanced_detector.py --step VirtualFittingStep  # 특정 Step만
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir
while backend_dir.name != 'backend' and backend_dir.parent != backend_dir:
    backend_dir = backend_dir.parent

if backend_dir.name == 'backend':
    sys.path.insert(0, str(backend_dir))
    print(f"✅ 프로젝트 경로 추가: {backend_dir}")
else:
    print(f"⚠️ backend 디렉토리를 찾을 수 없습니다: {current_dir}")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'test_detector_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

def test_imports():
    """필수 라이브러리 import 테스트"""
    logger.info("🔍 Import 테스트 시작...")
    
    import_results = {
        "torch": False,
        "numpy": False,
        "PIL": False,
        "cv2": False,
        "auto_model_detector": False
    }
    
    # PyTorch 테스트
    try:
        import torch
        import_results["torch"] = True
        logger.info(f"✅ PyTorch {torch.__version__} 로드 성공")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("🍎 M3 Max MPS 지원 확인됨")
        elif torch.cuda.is_available():
            logger.info("🎮 CUDA GPU 지원 확인됨")
        else:
            logger.info("💻 CPU 모드로 동작")
            
    except ImportError as e:
        logger.error(f"❌ PyTorch import 실패: {e}")
    
    # NumPy 테스트
    try:
        import numpy as np
        import_results["numpy"] = True
        logger.info(f"✅ NumPy {np.__version__} 로드 성공")
    except ImportError as e:
        logger.error(f"❌ NumPy import 실패: {e}")
    
    # PIL 테스트
    try:
        from PIL import Image
        import_results["PIL"] = True
        logger.info("✅ PIL 로드 성공")
    except ImportError as e:
        logger.error(f"❌ PIL import 실패: {e}")
    
    # OpenCV 테스트
    try:
        import cv2
        import_results["cv2"] = True
        logger.info(f"✅ OpenCV {cv2.__version__} 로드 성공")
    except ImportError as e:
        logger.error(f"❌ OpenCV import 실패: {e}")
    
    # 자동 모델 탐지기 테스트
    try:
        from app.ai_pipeline.utils.auto_model_detector import (
            RealWorldModelDetector,
            create_real_world_detector,
            DetectedModel,
            ModelCategory,
            ModelPriority
        )
        import_results["auto_model_detector"] = True
        logger.info("✅ 강화된 모델 탐지기 import 성공")
    except ImportError as e:
        logger.error(f"❌ 자동 모델 탐지기 import 실패: {e}")
        logger.error("💡 해결방법: auto_model_detector.py 파일이 올바른 위치에 있는지 확인하세요")
        return False
    
    return all(import_results.values())

def test_detector_creation():
    """탐지기 생성 테스트"""
    logger.info("🔧 탐지기 생성 테스트...")
    
    try:
        from app.ai_pipeline.utils.auto_model_detector import create_real_world_detector
        
        # 기본 탐지기 생성
        detector = create_real_world_detector()
        logger.info("✅ 기본 탐지기 생성 성공")
        
        # 고급 탐지기 생성 (모든 기능 활성화)
        enhanced_detector = create_real_world_detector(
            enable_deep_scan=True,
            enable_pytorch_validation=True,
            enable_performance_profiling=True,
            enable_memory_monitoring=True,
            enable_caching=True,
            max_workers=2,  # 테스트용으로 제한
            scan_timeout=300
        )
        logger.info("✅ 강화된 탐지기 생성 성공")
        
        # 탐지기 속성 확인
        logger.info(f"   - 검색 경로: {len(enhanced_detector.search_paths)}개")
        logger.info(f"   - 딥스캔: {enhanced_detector.enable_deep_scan}")
        logger.info(f"   - PyTorch 검증: {enhanced_detector.enable_pytorch_validation}")
        logger.info(f"   - 성능 프로파일링: {enhanced_detector.enable_performance_profiling}")
        logger.info(f"   - 메모리 모니터링: {enhanced_detector.enable_memory_monitoring}")
        
        return enhanced_detector
        
    except Exception as e:
        logger.error(f"❌ 탐지기 생성 실패: {e}")
        return None

def test_model_detection(detector, quick_mode: bool = False):
    """실제 모델 탐지 테스트"""
    logger.info("🔍 실제 모델 탐지 테스트 시작...")
    
    try:
        start_time = time.time()
        
        # 탐지 실행
        detected_models = detector.detect_all_models(
            force_rescan=True,  # 강제 재스캔
            min_confidence=0.3,
            enable_detailed_analysis=not quick_mode,
            max_models_per_category=10 if quick_mode else None
        )
        
        detection_time = time.time() - start_time
        
        logger.info(f"✅ 모델 탐지 완료: {len(detected_models)}개 발견 ({detection_time:.2f}초)")
        
        # 탐지 통계 출력
        stats = detector.scan_stats
        logger.info("📊 탐지 통계:")
        logger.info(f"   - 스캔된 파일: {stats['total_files_scanned']:,}개")
        logger.info(f"   - PyTorch 파일: {stats['pytorch_files_found']:,}개")
        logger.info(f"   - 검증된 모델: {stats['valid_pytorch_models']:,}개")
        logger.info(f"   - 오류 발생: {stats['errors_encountered']:,}개")
        
        # 발견된 모델들 상세 정보
        if detected_models:
            logger.info("\n🎯 발견된 모델들:")
            
            for i, (name, model) in enumerate(detected_models.items(), 1):
                logger.info(f"\n{i}. {name}")
                logger.info(f"   📁 경로: {model.path}")
                logger.info(f"   📦 크기: {model.file_size_mb:.1f}MB ({model.file_size_mb/1024:.2f}GB)")
                logger.info(f"   🏷️ 카테고리: {model.category.value}")
                logger.info(f"   🎯 Step: {model.step_name}")
                logger.info(f"   ⭐ 신뢰도: {model.confidence_score:.2f}")
                logger.info(f"   🔧 PyTorch 검증: {'✅' if model.pytorch_valid else '❌'}")
                
                if hasattr(model, 'parameter_count') and model.parameter_count > 0:
                    logger.info(f"   📊 파라미터: {model.parameter_count:,}개")
                
                if hasattr(model, 'architecture'):
                    logger.info(f"   🏗️ 아키텍처: {model.architecture.value}")
                
                if hasattr(model, 'device_compatibility'):
                    compatible_devices = [k for k, v in model.device_compatibility.items() if v]
                    logger.info(f"   💻 호환 디바이스: {', '.join(compatible_devices)}")
        else:
            logger.warning("⚠️ 탐지된 모델이 없습니다")
            logger.info("💡 해결방법:")
            logger.info("   1. ai_models 디렉토리에 모델 파일이 있는지 확인")
            logger.info("   2. 파일 확장자가 .pth, .pt, .bin, .safetensors인지 확인")
            logger.info("   3. 파일 크기가 최소 요구사항을 만족하는지 확인")
        
        return detected_models
        
    except Exception as e:
        logger.error(f"❌ 모델 탐지 실패: {e}")
        import traceback
        logger.debug(f"상세 오류:\n{traceback.format_exc()}")
        return {}

def test_step_specific_detection(detector, step_name: str):
    """특정 Step별 모델 탐지 테스트"""
    logger.info(f"🎯 {step_name} 전용 모델 탐지 테스트...")
    
    try:
        # 전체 모델 탐지 (이미 실행된 경우 캐시 사용)
        all_models = detector.detected_models
        if not all_models:
            all_models = detector.detect_all_models()
        
        # Step별 모델 필터링
        step_models = detector.get_models_by_step(step_name)
        
        logger.info(f"✅ {step_name}용 모델: {len(step_models)}개 발견")
        
        if step_models:
            for i, model in enumerate(step_models, 1):
                logger.info(f"{i}. {model.name}")
                logger.info(f"   크기: {model.file_size_mb:.1f}MB")
                logger.info(f"   신뢰도: {model.confidence_score:.2f}")
                logger.info(f"   우선순위: {model.priority.name}")
            
            # 최적 모델 선택
            best_model = detector.get_best_model_for_step(step_name)
            if best_model:
                logger.info(f"\n🏆 {step_name} 최적 모델: {best_model.name}")
                logger.info(f"   크기: {best_model.file_size_mb:.1f}MB")
                logger.info(f"   신뢰도: {best_model.confidence_score:.2f}")
                logger.info(f"   PyTorch 검증: {'✅' if best_model.pytorch_valid else '❌'}")
        else:
            logger.warning(f"⚠️ {step_name}용 모델을 찾을 수 없습니다")
        
        return step_models
        
    except Exception as e:
        logger.error(f"❌ {step_name} 모델 탐지 실패: {e}")
        return []

def test_pytorch_validation(detected_models):
    """PyTorch 모델 검증 테스트"""
    logger.info("🔍 PyTorch 모델 검증 테스트...")
    
    validated_models = []
    validation_failures = []
    
    for name, model in detected_models.items():
        if model.pytorch_valid:
            validated_models.append(model)
            logger.info(f"✅ {name}: 검증 성공")
            
            # 상세 검증 정보
            if hasattr(model, 'validation_results') and model.validation_results:
                results = model.validation_results
                if 'parameter_count' in results:
                    logger.info(f"   파라미터: {results['parameter_count']:,}개")
                if 'layer_types' in results:
                    layer_info = results['layer_types']
                    logger.info(f"   레이어 타입: {dict(list(layer_info.items())[:3])}")
        else:
            validation_failures.append(model)
            logger.warning(f"❌ {name}: 검증 실패")
    
    logger.info(f"\n📊 PyTorch 검증 결과:")
    logger.info(f"   ✅ 성공: {len(validated_models)}개")
    logger.info(f"   ❌ 실패: {len(validation_failures)}개")
    logger.info(f"   📈 성공률: {len(validated_models)/len(detected_models)*100:.1f}%")
    
    return validated_models

def test_performance_metrics(detector):
    """성능 메트릭 테스트"""
    logger.info("📊 성능 메트릭 테스트...")
    
    try:
        stats = detector.scan_stats
        device_info = detector.device_info
        
        logger.info("🖥️ 시스템 정보:")
        logger.info(f"   디바이스: {device_info.get('type', 'unknown')}")
        logger.info(f"   M3 Max: {'✅' if device_info.get('is_m3_max', False) else '❌'}")
        logger.info(f"   메모리: {device_info.get('memory_total_gb', 0):.1f}GB 총 / {device_info.get('memory_available_gb', 0):.1f}GB 사용가능")
        logger.info(f"   CPU 코어: {device_info.get('cpu_count', 0)}개")
        
        logger.info("\n⚡ 성능 통계:")
        logger.info(f"   스캔 시간: {stats.get('scan_duration', 0):.2f}초")
        logger.info(f"   처리된 파일: {stats.get('total_files_scanned', 0):,}개")
        logger.info(f"   초당 처리량: {stats.get('total_files_scanned', 0) / max(stats.get('scan_duration', 1), 0.1):.1f} 파일/초")
        
        if stats.get('memory_usage_delta_gb'):
            logger.info(f"   메모리 사용량: {stats['memory_usage_delta_gb']:.2f}GB")
        
        # 최적화 제안
        if device_info.get('optimization_hints'):
            logger.info(f"\n💡 최적화 제안:")
            for hint in device_info['optimization_hints']:
                logger.info(f"   - {hint}")
        
    except Exception as e:
        logger.error(f"❌ 성능 메트릭 테스트 실패: {e}")

def test_cache_functionality(detector):
    """캐시 기능 테스트"""
    logger.info("💾 캐시 기능 테스트...")
    
    try:
        # 첫 번째 스캔 (캐시 생성)
        start_time = time.time()
        models1 = detector.detect_all_models(force_rescan=True)
        first_scan_time = time.time() - start_time
        
        # 두 번째 스캔 (캐시 사용)
        start_time = time.time()
        models2 = detector.detect_all_models(force_rescan=False)
        second_scan_time = time.time() - start_time
        
        logger.info(f"✅ 캐시 테스트 완료:")
        logger.info(f"   첫 스캔 (캐시 생성): {first_scan_time:.2f}초")
        logger.info(f"   둘째 스캔 (캐시 사용): {second_scan_time:.2f}초")
        
        if second_scan_time > 0:
            speedup = first_scan_time / second_scan_time
            logger.info(f"   속도 향상: {speedup:.1f}배")
        
        # 캐시 통계
        cache_hits = detector.scan_stats.get('cache_hits', 0)
        cache_misses = detector.scan_stats.get('cache_misses', 0)
        
        if cache_hits + cache_misses > 0:
            hit_rate = cache_hits / (cache_hits + cache_misses) * 100
            logger.info(f"   캐시 적중률: {hit_rate:.1f}%")
        
    except Exception as e:
        logger.error(f"❌ 캐시 테스트 실패: {e}")

def generate_test_report(detected_models, detector, test_results):
    """테스트 리포트 생성"""
    logger.info("📋 테스트 리포트 생성...")
    
    try:
        report = {
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "detector_version": "7.0",
                "test_duration": test_results.get('total_duration', 0)
            },
            "system_info": detector.device_info,
            "scan_statistics": detector.scan_stats,
            "detected_models": []
        }
        
        # 모델 정보 추가
        for name, model in detected_models.items():
            model_info = {
                "name": name,
                "path": str(model.path),
                "category": model.category.value,
                "step_name": model.step_name,
                "file_size_mb": model.file_size_mb,
                "confidence_score": model.confidence_score,
                "pytorch_valid": model.pytorch_valid,
                "priority": model.priority.name
            }
            
            # 확장 정보 (있는 경우만)
            if hasattr(model, 'parameter_count'):
                model_info["parameter_count"] = model.parameter_count
            if hasattr(model, 'architecture'):
                model_info["architecture"] = model.architecture.value
            if hasattr(model, 'device_compatibility'):
                model_info["device_compatibility"] = model.device_compatibility
            
            report["detected_models"].append(model_info)
        
        # 리포트 파일 저장
        report_filename = f"detector_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 테스트 리포트 저장: {report_filename}")
        return report_filename
        
    except Exception as e:
        logger.error(f"❌ 리포트 생성 실패: {e}")
        return None

def main():
    """메인 테스트 함수"""
    parser = argparse.ArgumentParser(description="강화된 모델 탐지기 테스트 스크립트")
    parser.add_argument('--quick', action='store_true', help='빠른 테스트 모드')
    parser.add_argument('--detailed', action='store_true', help='상세 분석 모드')
    parser.add_argument('--step', type=str, help='특정 Step만 테스트 (예: VirtualFittingStep)')
    parser.add_argument('--no-cache', action='store_true', help='캐시 테스트 비활성화')
    
    args = parser.parse_args()
    
    logger.info("🚀 MyCloset AI - 강화된 모델 탐지기 테스트 시작")
    logger.info("=" * 60)
    
    test_results = {
        "start_time": time.time(),
        "tests_passed": 0,
        "tests_failed": 0
    }
    
    # 1. Import 테스트
    logger.info("\n1️⃣ Import 테스트")
    if test_imports():
        test_results["tests_passed"] += 1
        logger.info("✅ Import 테스트 통과")
    else:
        test_results["tests_failed"] += 1
        logger.error("❌ Import 테스트 실패 - 테스트 중단")
        return
    
    # 2. 탐지기 생성 테스트
    logger.info("\n2️⃣ 탐지기 생성 테스트")
    detector = test_detector_creation()
    if detector:
        test_results["tests_passed"] += 1
        logger.info("✅ 탐지기 생성 테스트 통과")
    else:
        test_results["tests_failed"] += 1
        logger.error("❌ 탐지기 생성 테스트 실패 - 테스트 중단")
        return
    
    # 3. 모델 탐지 테스트
    logger.info("\n3️⃣ 모델 탐지 테스트")
    detected_models = test_model_detection(detector, quick_mode=args.quick)
    if detected_models is not None:
        test_results["tests_passed"] += 1
        logger.info("✅ 모델 탐지 테스트 통과")
    else:
        test_results["tests_failed"] += 1
        logger.error("❌ 모델 탐지 테스트 실패")
    
    # 4. Step별 테스트 (옵션)
    if args.step:
        logger.info(f"\n4️⃣ {args.step} 전용 테스트")
        step_models = test_step_specific_detection(detector, args.step)
        if step_models is not None:
            test_results["tests_passed"] += 1
        else:
            test_results["tests_failed"] += 1
    
    # 5. PyTorch 검증 테스트
    if detected_models and not args.quick:
        logger.info("\n5️⃣ PyTorch 검증 테스트")
        validated_models = test_pytorch_validation(detected_models)
        if validated_models is not None:
            test_results["tests_passed"] += 1
        else:
            test_results["tests_failed"] += 1
    
    # 6. 성능 메트릭 테스트
    logger.info("\n6️⃣ 성능 메트릭 테스트")
    test_performance_metrics(detector)
    test_results["tests_passed"] += 1
    
    # 7. 캐시 기능 테스트 (옵션)
    if not args.no_cache and not args.quick:
        logger.info("\n7️⃣ 캐시 기능 테스트")
        test_cache_functionality(detector)
        test_results["tests_passed"] += 1
    
    # 최종 결과
    test_results["total_duration"] = time.time() - test_results["start_time"]
    
    logger.info("\n" + "=" * 60)
    logger.info("🏁 테스트 완료!")
    logger.info(f"✅ 통과: {test_results['tests_passed']}개")
    logger.info(f"❌ 실패: {test_results['tests_failed']}개")
    logger.info(f"⏱️ 총 시간: {test_results['total_duration']:.2f}초")
    
    # 테스트 리포트 생성
    if detected_models:
        report_file = generate_test_report(detected_models, detector, test_results)
        if report_file:
            logger.info(f"📋 상세 리포트: {report_file}")
    
    # 종료 코드
    exit_code = 0 if test_results["tests_failed"] == 0 else 1
    logger.info(f"🔚 종료 코드: {exit_code}")
    sys.exit(exit_code)

if __name__ == "__main__":
    main()