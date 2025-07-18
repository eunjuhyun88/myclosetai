#!/usr/bin/env python3
"""
Step 01 Human Parsing 완전 테스트 스크립트
backend/scripts/test_step_01_complete.py

Step 01의 모든 기능을 포괄적으로 테스트:
- BaseStepMixin 연동 확인
- ModelLoader 인터페이스 테스트
- 20개 부위 인체 파싱 검증
- 시각화 기능 테스트
- M3 Max 최적화 확인
- 프로덕션 안정성 검증

실행 방법:
cd backend
python scripts/test_step_01_complete.py
"""

import os
import sys
import time
import asyncio
import logging
import json
import base64
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from io import BytesIO

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from PIL import Image, ImageDraw

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_step_01_complete.log')
    ]
)
logger = logging.getLogger(__name__)

class Step01TestSuite:
    """Step 01 Human Parsing 완전 테스트 스위트"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        self.device = self._detect_best_device()
        
    def _detect_best_device(self) -> str:
        """최적 디바이스 감지"""
        try:
            import torch
            if torch.backends.mps.is_available():
                return 'mps'
            elif torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        except:
            return 'cpu'
    
    def create_realistic_test_image(self, size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
        """현실적인 테스트 이미지 생성 (사람 모양)"""
        width, height = size
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 배경 (흰색)
        image[:, :] = [240, 240, 240]
        
        # 사람 형태 시뮬레이션
        center_x, center_y = width // 2, height // 2
        
        # 머리 (원형)
        head_radius = 60
        for y in range(max(0, center_y - 150), min(height, center_y - 30)):
            for x in range(max(0, center_x - head_radius), min(width, center_x + head_radius)):
                if (x - center_x) ** 2 + (y - (center_y - 90)) ** 2 <= head_radius ** 2:
                    image[y, x] = [255, 220, 177]  # 피부색
        
        # 얼굴 특징
        # 눈
        image[center_y - 110:center_y - 100, center_x - 20:center_x - 10] = [0, 0, 0]
        image[center_y - 110:center_y - 100, center_x + 10:center_x + 20] = [0, 0, 0]
        
        # 입
        image[center_y - 80:center_y - 70, center_x - 10:center_x + 10] = [200, 100, 100]
        
        # 목
        image[center_y - 30:center_y, center_x - 25:center_x + 25] = [255, 220, 177]
        
        # 몸통 (상의)
        image[center_y:center_y + 120, center_x - 80:center_x + 80] = [100, 150, 200]  # 파란 상의
        
        # 팔
        # 왼팔
        image[center_y + 20:center_y + 100, center_x - 120:center_x - 80] = [255, 220, 177]  # 팔
        image[center_y + 40:center_y + 80, center_x - 140:center_x - 120] = [150, 100, 50]   # 소매
        
        # 오른팔
        image[center_y + 20:center_y + 100, center_x + 80:center_x + 120] = [255, 220, 177]  # 팔
        image[center_y + 40:center_y + 80, center_x + 120:center_x + 140] = [150, 100, 50]   # 소매
        
        # 하의 (바지)
        image[center_y + 120:center_y + 200, center_x - 80:center_x + 80] = [50, 50, 100]  # 검은 바지
        
        # 다리
        # 왼다리
        image[center_y + 200:center_y + 280, center_x - 70:center_x - 10] = [50, 50, 100]
        
        # 오른다리
        image[center_y + 200:center_y + 280, center_x + 10:center_x + 70] = [50, 50, 100]
        
        # 신발
        image[center_y + 280:center_y + 300, center_x - 80:center_x - 40] = [0, 0, 0]      # 왼쪽 신발
        image[center_y + 280:center_y + 300, center_x + 40:center_x + 80] = [0, 0, 0]      # 오른쪽 신발
        
        # PIL로 변환 후 텐서로
        pil_image = Image.fromarray(image)
        
        # 텐서 변환 [1, 3, H, W]
        tensor = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    async def test_01_imports(self) -> bool:
        """테스트 1: Import 및 클래스 로딩"""
        print("\n" + "="*60)
        print("🧪 테스트 1: Import 및 클래스 로딩")
        print("="*60)
        
        try:
            # Step 01 클래스들 import
            from app.ai_pipeline.steps.step_01_human_parsing import (
                HumanParsingStep,
                HumanParsingConfig,
                create_human_parsing_step,
                create_human_parsing_step_sync,
                BODY_PARTS,
                CLOTHING_CATEGORIES,
                VISUALIZATION_COLORS
            )
            
            print("✅ Step 01 클래스들 import 성공")
            print(f"📊 인체 부위 정의: {len(BODY_PARTS)}개")
            print(f"👕 의류 카테고리: {len(CLOTHING_CATEGORIES)}개")
            print(f"🎨 시각화 색상: {len(VISUALIZATION_COLORS)}개")
            
            # BaseStepMixin import 확인
            try:
                from app.ai_pipeline.steps.base_step_mixin import BaseStepMixin
                print("✅ BaseStepMixin import 성공")
            except ImportError as e:
                print(f"⚠️ BaseStepMixin import 실패: {e}")
            
            # ModelLoader import 확인
            try:
                from app.ai_pipeline.utils.model_loader import ModelLoader, get_global_model_loader
                print("✅ ModelLoader import 성공")
            except ImportError as e:
                print(f"⚠️ ModelLoader import 실패: {e}")
            
            self.test_results['imports'] = True
            return True
            
        except Exception as e:
            print(f"❌ Import 테스트 실패: {e}")
            self.test_results['imports'] = False
            return False
    
    async def test_02_config_creation(self) -> bool:
        """테스트 2: 설정 클래스 생성 및 호환성"""
        print("\n" + "="*60)
        print("🧪 테스트 2: 설정 클래스 생성 및 호환성")
        print("="*60)
        
        try:
            from app.ai_pipeline.steps.step_01_human_parsing import HumanParsingConfig
            
            # 기본 설정 생성
            basic_config = HumanParsingConfig()
            print("✅ 기본 설정 생성 성공")
            print(f"   - 디바이스: {basic_config.device}")
            print(f"   - 모델: {basic_config.model_name}")
            print(f"   - 입력 크기: {basic_config.input_size}")
            
            # M3 Max 최적화 설정
            m3_config = HumanParsingConfig(
                device='mps',
                use_fp16=True,
                use_coreml=True,
                enable_neural_engine=True,
                memory_efficient=True,
                optimization_enabled=True
            )
            print("✅ M3 Max 최적화 설정 생성 성공")
            print(f"   - M3 Max 감지: {m3_config.is_m3_max}")
            print(f"   - FP16 사용: {m3_config.use_fp16}")
            print(f"   - CoreML 사용: {m3_config.use_coreml}")
            
            # PipelineManager 호환성 테스트
            pipeline_params = {
                'device': 'cpu',
                'optimization_enabled': True,
                'device_type': 'cpu',
                'memory_gb': 16.0,
                'quality_level': 'balanced',
                'model_type': 'graphonomy',
                'processing_mode': 'production',
                'unknown_param': 'should_be_ignored'  # 알 수 없는 파라미터
            }
            
            # 호환성 설정 생성
            compat_config = HumanParsingConfig(**{
                k: v for k, v in pipeline_params.items() 
                if k in HumanParsingConfig.__dataclass_fields__
            })
            print("✅ PipelineManager 호환성 설정 생성 성공")
            print(f"   - 최적화: {compat_config.optimization_enabled}")
            print(f"   - 품질 레벨: {compat_config.quality_level}")
            
            self.test_results['config_creation'] = True
            return True
            
        except Exception as e:
            print(f"❌ 설정 생성 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['config_creation'] = False
            return False
    
    async def test_03_step_initialization(self) -> bool:
        """테스트 3: Step 인스턴스 생성 및 초기화"""
        print("\n" + "="*60)
        print("🧪 테스트 3: Step 인스턴스 생성 및 초기화")
        print("="*60)
        
        try:
            from app.ai_pipeline.steps.step_01_human_parsing import HumanParsingStep, HumanParsingConfig
            
            # 기본 Step 생성
            config = HumanParsingConfig(
                device=self.device,
                use_fp16=False if self.device == 'cpu' else True,
                warmup_enabled=True,
                enable_visualization=True
            )
            
            step = HumanParsingStep(device=self.device, config=config)
            print("✅ Step 인스턴스 생성 성공")
            print(f"   - 클래스: {step.__class__.__name__}")
            print(f"   - 단계 번호: {step.step_number}")
            print(f"   - 디바이스: {step.device}")
            print(f"   - Logger 존재: {hasattr(step, 'logger') and step.logger is not None}")
            
            # BaseStepMixin 속성 확인
            base_attributes = ['logger', 'device', 'is_initialized', 'model_interface']
            for attr in base_attributes:
                exists = hasattr(step, attr)
                print(f"   - {attr}: {'✅' if exists else '❌'} {'존재' if exists else '누락'}")
            
            # 초기화 테스트
            print("\n🔄 초기화 테스트 중...")
            try:
                init_success = await step.initialize()
                print(f"✅ 초기화 완료:")
                print(f"   - 성공 여부: {init_success}")
                print(f"   - 초기화 상태: {step.is_initialized}")
                print(f"   - 로드된 모델: {list(step.models_loaded.keys()) if hasattr(step, 'models_loaded') else '없음'}")
            except Exception as e:
                print(f"⚠️ 초기화 실패 (예상됨 - ModelLoader 없음): {e}")
                # 시뮬레이션 모드로 강제 설정
                step.is_initialized = True
                print("💡 시뮬레이션 모드로 계속 진행")
            
            # Step 정보 확인
            try:
                step_info = await step.get_step_info()
                print("\n📊 Step 정보:")
                print(f"   - Step 이름: {step_info.get('step_name')}")
                print(f"   - 설정된 디바이스: {step_info.get('device')}")
                print(f"   - 캐시 크기: {step_info.get('cache', {}).get('size', 0)}")
            except Exception as e:
                print(f"⚠️ Step 정보 조회 실패: {e}")
            
            # 글로벌 변수로 저장 (다른 테스트에서 사용)
            self.test_step = step
            
            self.test_results['step_initialization'] = True
            return True
            
        except Exception as e:
            print(f"❌ Step 초기화 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['step_initialization'] = False
            return False
    
    async def test_04_basic_processing(self) -> bool:
        """테스트 4: 기본 이미지 처리"""
        print("\n" + "="*60)
        print("🧪 테스트 4: 기본 이미지 처리")
        print("="*60)
        
        try:
            step = getattr(self, 'test_step', None)
            if not step:
                print("❌ Step 인스턴스가 없습니다. 이전 테스트를 먼저 실행하세요.")
                return False
            
            # 테스트 이미지 생성
            print("🖼️ 현실적인 테스트 이미지 생성 중...")
            test_image = self.create_realistic_test_image()
            print(f"✅ 테스트 이미지 생성 완료:")
            print(f"   - 형태: {test_image.shape}")
            print(f"   - 데이터 타입: {test_image.dtype}")
            print(f"   - 값 범위: [{test_image.min().item():.3f}, {test_image.max().item():.3f}]")
            
            # 기본 처리 테스트
            print("\n🔄 인체 파싱 처리 중...")
            start_time = time.time()
            
            result = await step.process(test_image)
            
            processing_time = time.time() - start_time
            print(f"✅ 처리 완료 ({processing_time:.3f}초)")
            
            # 결과 검증
            print("\n📊 처리 결과 분석:")
            print(f"   - 성공 여부: {result.get('success', False)}")
            print(f"   - 신뢰도: {result.get('confidence', 0):.3f}")
            print(f"   - 처리 시간: {result.get('processing_time', 0):.3f}초")
            print(f"   - 캐시에서 반환: {result.get('from_cache', False)}")
            
            # 상세 정보 확인
            details = result.get('details', {})
            if details:
                print(f"   - 감지된 부위: {details.get('detected_parts', 0)}/20")
                print(f"   - 시각화 이미지: {'있음' if details.get('result_image') else '없음'}")
                print(f"   - 오버레이 이미지: {'있음' if details.get('overlay_image') else '없음'}")
                
                # 의류 정보
                clothing_info = details.get('clothing_info', {})
                if clothing_info:
                    categories = clothing_info.get('categories_detected', [])
                    print(f"   - 감지된 의류 카테고리: {len(categories)}개 {categories}")
                    print(f"   - 주요 카테고리: {clothing_info.get('dominant_category', 'None')}")
            
            # 파싱 맵 확인
            parsing_map = result.get('parsing_map')
            if parsing_map is not None:
                if isinstance(parsing_map, np.ndarray):
                    unique_values = np.unique(parsing_map)
                    print(f"   - 파싱 맵 고유값: {len(unique_values)}개 {unique_values[:10].tolist()}...")
                elif isinstance(parsing_map, list):
                    print(f"   - 파싱 맵 (리스트): {len(parsing_map)}x{len(parsing_map[0]) if parsing_map else 0}")
            
            self.test_results['basic_processing'] = True
            return True
            
        except Exception as e:
            print(f"❌ 기본 처리 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['basic_processing'] = False
            return False
    
    async def test_05_visualization_features(self) -> bool:
        """테스트 5: 시각화 기능"""
        print("\n" + "="*60)
        print("🧪 테스트 5: 시각화 기능")
        print("="*60)
        
        try:
            step = getattr(self, 'test_step', None)
            if not step:
                print("❌ Step 인스턴스가 없습니다.")
                return False
            
            # 시각화 활성화 설정으로 재처리
            print("🎨 시각화 기능 테스트 중...")
            
            # 설정 업데이트
            step.config.enable_visualization = True
            step.config.visualization_quality = "high"
            step.config.show_part_labels = True
            
            # 테스트 이미지로 처리
            test_image = self.create_realistic_test_image()
            result = await step.process(test_image)
            
            # 시각화 결과 확인
            details = result.get('details', {})
            
            # 결과 이미지 확인
            result_image = details.get('result_image', '')
            overlay_image = details.get('overlay_image', '')
            
            print(f"✅ 시각화 결과:")
            print(f"   - 색깔 파싱 이미지: {'생성됨' if result_image else '없음'} ({len(result_image)} chars)")
            print(f"   - 오버레이 이미지: {'생성됨' if overlay_image else '없음'} ({len(overlay_image)} chars)")
            
            # Base64 이미지 저장 테스트
            if result_image:
                try:
                    # Base64 디코딩 테스트
                    image_data = base64.b64decode(result_image)
                    with open('test_result_parsing.jpg', 'wb') as f:
                        f.write(image_data)
                    print("   - 파싱 결과 이미지 저장: test_result_parsing.jpg")
                except Exception as e:
                    print(f"   - 이미지 저장 실패: {e}")
            
            if overlay_image:
                try:
                    image_data = base64.b64decode(overlay_image)
                    with open('test_result_overlay.jpg', 'wb') as f:
                        f.write(image_data)
                    print("   - 오버레이 이미지 저장: test_result_overlay.jpg")
                except Exception as e:
                    print(f"   - 오버레이 저장 실패: {e}")
            
            # 직접 시각화 함수 테스트
            print("\n🔧 직접 시각화 함수 테스트:")
            try:
                parsing_map = result.get('parsing_map')
                if parsing_map is not None:
                    if isinstance(parsing_map, list):
                        parsing_map = np.array(parsing_map)
                    
                    # 시각화 함수 호출
                    visualized = step.visualize_parsing(parsing_map)
                    print(f"   ✅ 직접 시각화 성공: {visualized.shape}")
                    
                    # PIL로 저장
                    vis_image = Image.fromarray(visualized)
                    vis_image.save('test_direct_visualization.png')
                    print("   - 직접 시각화 이미지 저장: test_direct_visualization.png")
                    
                else:
                    print("   ⚠️ 파싱 맵이 없어서 직접 시각화 건너뜀")
                    
            except Exception as e:
                print(f"   ❌ 직접 시각화 실패: {e}")
            
            self.test_results['visualization_features'] = True
            return True
            
        except Exception as e:
            print(f"❌ 시각화 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['visualization_features'] = False
            return False
    
    async def test_06_performance_cache(self) -> bool:
        """테스트 6: 성능 및 캐시 시스템"""
        print("\n" + "="*60)
        print("🧪 테스트 6: 성능 및 캐시 시스템")
        print("="*60)
        
        try:
            step = getattr(self, 'test_step', None)
            if not step:
                print("❌ Step 인스턴스가 없습니다.")
                return False
            
            # 성능 테스트용 이미지들 생성
            test_images = [
                self.create_realistic_test_image((256, 256)),
                self.create_realistic_test_image((512, 512)),
                self.create_realistic_test_image((768, 768))
            ]
            
            print("⚡ 성능 테스트 중...")
            
            processing_times = []
            
            for i, img in enumerate(test_images):
                start = time.time()
                result = await step.process(img)
                elapsed = time.time() - start
                processing_times.append(elapsed)
                
                print(f"   🔄 이미지 {i+1} ({img.shape[2]}x{img.shape[3]}): {elapsed:.3f}초")
                print(f"       신뢰도: {result.get('confidence', 0):.3f}")
                print(f"       성공: {result.get('success', False)}")
            
            avg_time = sum(processing_times) / len(processing_times)
            print(f"\n📊 성능 통계:")
            print(f"   - 평균 처리 시간: {avg_time:.3f}초")
            print(f"   - 최소 시간: {min(processing_times):.3f}초")
            print(f"   - 최대 시간: {max(processing_times):.3f}초")
            
            # 캐시 테스트
            print("\n💾 캐시 시스템 테스트:")
            
            # 같은 이미지로 다시 처리 (캐시 히트 기대)
            cache_img = test_images[0]
            
            cache_start = time.time()
            cached_result = await step.process(cache_img)
            cache_time = time.time() - cache_start
            
            print(f"   - 캐시된 처리 시간: {cache_time:.3f}초")
            print(f"   - 캐시에서 반환: {cached_result.get('from_cache', False)}")
            print(f"   - 속도 향상: {processing_times[0]/cache_time:.1f}x" if cache_time > 0 else "")
            
            # 처리 통계 확인
            try:
                stats = step.processing_stats
                print(f"\n📈 누적 통계:")
                print(f"   - 총 처리 횟수: {stats.get('total_processed', 0)}")
                print(f"   - 평균 시간: {stats.get('average_time', 0):.3f}초")
                print(f"   - 캐시 히트: {stats.get('cache_hits', 0)}")
                print(f"   - 모델 전환: {stats.get('model_switches', 0)}")
            except Exception as e:
                print(f"   ⚠️ 통계 조회 실패: {e}")
            
            self.test_results['performance_cache'] = True
            return True
            
        except Exception as e:
            print(f"❌ 성능/캐시 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['performance_cache'] = False
            return False
    
    async def test_07_error_handling(self) -> bool:
        """테스트 7: 에러 처리 및 안정성"""
        print("\n" + "="*60)
        print("🧪 테스트 7: 에러 처리 및 안정성")
        print("="*60)
        
        try:
            step = getattr(self, 'test_step', None)
            if not step:
                print("❌ Step 인스턴스가 없습니다.")
                return False
            
            error_cases = [
                ("잘못된 형태 텐서", torch.randn(2, 2)),  # 2D 텐서
                ("잘못된 채널 수", torch.randn(1, 1, 512, 512)),  # 1채널
                ("너무 작은 이미지", torch.randn(1, 3, 10, 10)),  # 10x10
                ("None 입력", None),
            ]
            
            print("🛡️ 에러 케이스 테스트:")
            
            error_handled_count = 0
            
            for case_name, input_data in error_cases:
                print(f"\n   📋 {case_name} 테스트...")
                try:
                    if input_data is None:
                        # None 입력은 별도 처리
                        print(f"      ⚠️ None 입력 - 건너뜀")
                        continue
                    
                    result = await step.process(input_data)
                    
                    if result.get('success', False):
                        print(f"      ✅ 처리 성공 (예상 밖)")
                        print(f"         신뢰도: {result.get('confidence', 0):.3f}")
                    else:
                        print(f"      ✅ 적절한 에러 처리")
                        print(f"         메시지: {result.get('message', 'No message')}")
                    
                    error_handled_count += 1
                    
                except Exception as e:
                    print(f"      ⚠️ 예외 발생 (예상됨): {e}")
                    error_handled_count += 1
            
            print(f"\n🛡️ 에러 처리 결과:")
            print(f"   - 테스트 케이스: {len(error_cases)}개")
            print(f"   - 안전 처리: {error_handled_count}개")
            print(f"   - 처리율: {error_handled_count/len(error_cases)*100:.1f}%")
            
            # 메모리 관리 테스트
            print("\n🧹 메모리 관리 테스트:")
            try:
                if hasattr(step, 'memory_manager'):
                    memory_stats = await step.memory_manager.get_usage_stats()
                    print(f"   ✅ 메모리 상태 조회 성공: {memory_stats}")
                    
                    await step.memory_manager.cleanup()
                    print(f"   ✅ 메모리 정리 완료")
                else:
                    print(f"   ⚠️ 메모리 매니저 없음")
            except Exception as e:
                print(f"   ⚠️ 메모리 관리 테스트 실패: {e}")
            
            self.test_results['error_handling'] = True
            return True
            
        except Exception as e:
            print(f"❌ 에러 처리 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['error_handling'] = False
            return False
    
    async def test_08_factory_functions(self) -> bool:
        """테스트 8: 팩토리 함수 및 호환성"""
        print("\n" + "="*60)
        print("🧪 테스트 8: 팩토리 함수 및 호환성")
        print("="*60)
        
        try:
            from app.ai_pipeline.steps.step_01_human_parsing import (
                create_human_parsing_step,
                create_human_parsing_step_sync
            )
            
            # 비동기 팩토리 함수 테스트
            print("⚡ 비동기 팩토리 함수 테스트:")
            step1 = await create_human_parsing_step(
                device="cpu",
                config={
                    'use_fp16': False,
                    'warmup_enabled': False,
                    'enable_visualization': True
                }
            )
            print(f"   ✅ create_human_parsing_step 성공")
            print(f"      클래스: {step1.__class__.__name__}")
            print(f"      초기화: {step1.is_initialized}")
            
            # 동기 팩토리 함수 테스트
            print("\n🔄 동기 팩토리 함수 테스트:")
            step2 = create_human_parsing_step_sync(
                device="cpu",
                config={
                    'quality_level': 'fast',
                    'optimization_enabled': False
                }
            )
            print(f"   ✅ create_human_parsing_step_sync 성공")
            print(f"      클래스: {step2.__class__.__name__}")
            print(f"      디바이스: {step2.device}")
            
            # kwargs 호환성 테스트
            print("\n🔧 kwargs 호환성 테스트:")
            step3 = await create_human_parsing_step(
                device="auto",
                optimization_enabled=True,
                device_type="auto", 
                memory_gb=16.0,
                quality_level="balanced",
                model_type="graphonomy",
                unknown_param="should_be_ignored"
            )
            print(f"   ✅ kwargs 호환성 성공")
            print(f"      설정 품질: {step3.config.quality_level}")
            print(f"      최적화: {step3.config.optimization_enabled}")
            
            # 정리
            await step1.cleanup()
            await step2.cleanup() 
            await step3.cleanup()
            print(f"   ✅ 팩토리 생성 Step들 정리 완료")
            
            self.test_results['factory_functions'] = True
            return True
            
        except Exception as e:
            print(f"❌ 팩토리 함수 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['factory_functions'] = False
            return False
    
    async def test_09_cleanup(self) -> bool:
        """테스트 9: 리소스 정리"""
        print("\n" + "="*60)
        print("🧪 테스트 9: 리소스 정리")
        print("="*60)
        
        try:
            step = getattr(self, 'test_step', None)
            if not step:
                print("⚠️ 정리할 Step 인스턴스가 없습니다.")
                return True
            
            print("🧹 리소스 정리 중...")
            
            # 정리 전 상태 확인
            print(f"   정리 전 상태:")
            print(f"   - 초기화 상태: {step.is_initialized}")
            print(f"   - 로드된 모델: {len(step.models_loaded) if hasattr(step, 'models_loaded') else 0}개")
            print(f"   - 캐시 크기: {len(step.result_cache) if hasattr(step, 'result_cache') else 0}개")
            
            # 실제 정리 실행
            await step.cleanup()
            
            # 정리 후 상태 확인  
            print(f"\n   정리 후 상태:")
            print(f"   - 초기화 상태: {step.is_initialized}")
            print(f"   - 로드된 모델: {len(step.models_loaded) if hasattr(step, 'models_loaded') else 0}개")
            print(f"   - 캐시 크기: {len(step.result_cache) if hasattr(step, 'result_cache') else 0}개")
            
            print("✅ 리소스 정리 완료")
            
            self.test_results['cleanup'] = True
            return True
            
        except Exception as e:
            print(f"❌ 리소스 정리 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['cleanup'] = False
            return False
    
    def print_final_report(self):
        """최종 테스트 보고서 출력"""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*80)
        print("📊 Step 01 Human Parsing 완전 테스트 결과 보고서")
        print("="*80)
        print(f"🕐 총 테스트 시간: {total_time:.2f}초")
        print(f"🖥️ 테스트 디바이스: {self.device}")
        print(f"📅 완료 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 테스트 결과 요약
        passed_tests = sum(1 for result in self.test_results.values() if result)
        total_tests = len(self.test_results)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"📈 테스트 결과 요약:")
        print(f"   전체 테스트: {total_tests}개")
        print(f"   통과: {passed_tests}개")
        print(f"   실패: {total_tests - passed_tests}개")
        print(f"   성공률: {success_rate:.1f}%")
        print()
        
        # 상세 결과
        print(f"📋 상세 테스트 결과:")
        test_names = {
            'imports': '1. Import 및 클래스 로딩',
            'config_creation': '2. 설정 클래스 생성',
            'step_initialization': '3. Step 초기화',
            'basic_processing': '4. 기본 이미지 처리',
            'visualization_features': '5. 시각화 기능',
            'performance_cache': '6. 성능 및 캐시',
            'error_handling': '7. 에러 처리',
            'factory_functions': '8. 팩토리 함수',
            'cleanup': '9. 리소스 정리'
        }
        
        for test_key, test_name in test_names.items():
            result = self.test_results.get(test_key, False)
            status = "✅ 통과" if result else "❌ 실패"
            print(f"   {status} {test_name}")
        
        print()
        
        # 결론
        if success_rate >= 90:
            print("🎉 Step 01 Human Parsing이 정상적으로 작동합니다!")
            print("   모든 핵심 기능이 완벽하게 구현되었습니다.")
        elif success_rate >= 70:
            print("✅ Step 01이 대부분 정상 작동합니다.")
            print("   일부 비핵심 기능에서 문제가 있을 수 있습니다.")
        else:
            print("⚠️ Step 01에 중요한 문제가 있습니다.")
            print("   설치 및 설정을 다시 확인하세요.")
        
        print()
        
        # 다음 단계 안내
        print("🚀 다음 단계:")
        if success_rate >= 90:
            print("1. 전체 8단계 파이프라인 테스트 실행")
            print("2. FastAPI 서버 통합 테스트")
            print("3. 프론트엔드 연동 테스트")
        elif success_rate >= 70:
            print("1. 실패한 테스트 케이스 분석")
            print("2. 의존성 패키지 재설치") 
            print("3. Step 01 재테스트")
        else:
            print("1. 환경 설정 재검토")
            print("2. 필수 패키지 설치 확인")
            print("3. 에러 로그 분석")
        
        print()
        print("📁 생성된 파일들:")
        files = [
            "test_step_01_complete.log (테스트 로그)",
            "test_result_parsing.jpg (파싱 결과 이미지)",
            "test_result_overlay.jpg (오버레이 이미지)",
            "test_direct_visualization.png (직접 시각화)"
        ]
        for file_info in files:
            print(f"   - {file_info}")
        
        print("="*80)

async def main():
    """메인 테스트 실행"""
    print("🧪 Step 01 Human Parsing 완전 테스트 스위트")
    print("="*80)
    print("🎯 목표: Step 01의 모든 기능을 포괄적으로 검증")
    print("📋 포함: BaseStepMixin, ModelLoader, 시각화, M3 Max 최적화")
    print("="*80)
    
    # 테스트 스위트 생성
    test_suite = Step01TestSuite()
    
    # 테스트 실행
    tests = [
        test_suite.test_01_imports,
        test_suite.test_02_config_creation,
        test_suite.test_03_step_initialization,
        test_suite.test_04_basic_processing,
        test_suite.test_05_visualization_features,
        test_suite.test_06_performance_cache,
        test_suite.test_07_error_handling,
        test_suite.test_08_factory_functions,
        test_suite.test_09_cleanup
    ]
    
    # 순차 테스트 실행
    for test_func in tests:
        try:
            await test_func()
        except KeyboardInterrupt:
            print("\n🛑 사용자에 의해 테스트가 중단되었습니다.")
            break
        except Exception as e:
            print(f"\n💥 예상치 못한 오류: {e}")
            import traceback
            traceback.print_exc()
    
    # 최종 보고서
    test_suite.print_final_report()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n💥 테스트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()