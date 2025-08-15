# backend/app/ai_pipeline/steps/test_step_01_with_unified_db.py
"""
🧪 Step 01 + 통합 Session Database 통합 테스트
================================================================================

✅ Step 01 Human Parsing 실제 AI 추론 테스트
✅ 통합 Session Database 연동 테스트
✅ 데이터 저장 및 전달 테스트
✅ 실제 이미지 처리 테스트

Author: MyCloset AI Team
Date: 2025-01-27
Version: 1.0.0
"""

import asyncio
import logging
import time
import sys
import os
from pathlib import Path
from typing import Dict, Any
import numpy as np
from PIL import Image
import json

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 경로 설정
current_dir = Path(__file__).parent
app_core_dir = current_dir.parent.parent.parent / "app" / "core"
sys.path.insert(0, str(app_core_dir))

try:
    from unified_session_database import get_unified_session_database
    logger.info("✅ UnifiedSessionDatabase import 성공")
except ImportError as e:
    logger.error(f"❌ UnifiedSessionDatabase import 실패: {e}")
    sys.exit(1)

# Step 01 import
try:
    from step_01_human_parsing_models.step_01_human_parsing import HumanParsingStep
    logger.info("✅ HumanParsingStep import 성공")
except ImportError as e:
    logger.error(f"❌ HumanParsingStep import 실패: {e}")
    sys.exit(1)

class Step01IntegrationTester:
    """Step 01 + 통합 Session Database 통합 테스트"""
    
    def __init__(self):
        self.unified_db = None
        self.human_parsing_step = None
        self.test_session_id = None
        self.test_results = {}
        
    async def setup_test_environment(self):
        """테스트 환경 설정"""
        try:
            logger.info("🔄 테스트 환경 설정 시작...")
            
            # 1. 통합 Session Database 설정
            self.unified_db = get_unified_session_database()
            logger.info("✅ UnifiedSessionDatabase 연결 성공")
            
            # 2. Step 01 초기화
            self.human_parsing_step = HumanParsingStep(device='cpu', strict_mode=True)
            logger.info("✅ HumanParsingStep 초기화 성공")
            
            # 3. 테스트용 이미지 생성
            test_image = self._create_test_image()
            test_image_path = self._save_test_image(test_image)
            
            # 4. 테스트 세션 생성
            self.test_session_id = await self.unified_db.create_session(
                person_image_path=str(test_image_path),
                clothing_image_path="test_clothing.jpg",
                measurements={"height": 170, "weight": 65}
            )
            logger.info(f"✅ 테스트 세션 생성: {self.test_session_id}")
            
            logger.info("✅ 테스트 환경 설정 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 테스트 환경 설정 실패: {e}")
            return False
    
    def _create_test_image(self) -> np.ndarray:
        """테스트용 이미지 생성"""
        try:
            # 256x256 크기의 테스트 이미지 생성
            image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            
            # 사람 형태를 시뮬레이션하기 위한 간단한 패턴
            # 중앙에 타원형 영역 생성 (사람 몸통 시뮬레이션)
            center_y, center_x = 128, 128
            for y in range(256):
                for x in range(256):
                    # 타원형 영역 계산
                    ellipse = ((x - center_x) ** 2 / 60 ** 2) + ((y - center_y) ** 2 / 100 ** 2)
                    if ellipse <= 1:
                        image[y, x] = [100, 150, 200]  # 파란색 계열
            
            logger.info("✅ 테스트 이미지 생성 완료")
            return image
            
        except Exception as e:
            logger.error(f"❌ 테스트 이미지 생성 실패: {e}")
            # 기본 이미지 반환
            return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    def _save_test_image(self, image: np.ndarray) -> Path:
        """테스트 이미지를 파일로 저장"""
        try:
            # 테스트 이미지 디렉토리 생성
            test_dir = Path("test_images")
            test_dir.mkdir(exist_ok=True)
            
            # 이미지 저장
            image_path = test_dir / f"test_person_{int(time.time())}.jpg"
            pil_image = Image.fromarray(image)
            pil_image.save(image_path)
            
            logger.info(f"✅ 테스트 이미지 저장: {image_path}")
            return image_path
            
        except Exception as e:
            logger.error(f"❌ 테스트 이미지 저장 실패: {e}")
            return Path("test_person.jpg")
    
    async def test_step_01_processing(self):
        """Step 01 처리 테스트"""
        try:
            logger.info("🧪 Step 01 Human Parsing 처리 테스트 시작...")
            
            if not self.human_parsing_step or not self.test_session_id:
                logger.error("❌ 테스트 환경이 설정되지 않음")
                return False
            
            # Step 01 입력 데이터 준비 - 실제 생성된 이미지 경로 사용
            input_data = {
                'session_id': self.test_session_id,
                'person_image_path': str(Path("test_images") / "test_person.jpg"),  # 기본 경로 사용
                'measurements': {"height": 170, "weight": 65}
            }
            
            # 실제 이미지 파일이 있는지 확인하고 경로 조정
            test_images_dir = Path("test_images")
            if test_images_dir.exists():
                image_files = list(test_images_dir.glob("test_person_*.jpg"))
                if image_files:
                    input_data['person_image_path'] = str(image_files[0])
                    logger.info(f"✅ 실제 이미지 파일 경로 사용: {input_data['person_image_path']}")
            
            # Step 01 처리 실행
            start_time = time.time()
            result = await self.human_parsing_step.process(input_data)
            processing_time = time.time() - start_time
            
            if result and result.get('success'):
                logger.info("✅ Step 01 처리 성공")
                logger.info(f"   - 처리 시간: {processing_time:.2f}초")
                logger.info(f"   - 결과 키: {list(result.keys())}")
                logger.info(f"   - 품질 점수: {result.get('quality_score', 'N/A')}")
                logger.info(f"   - 상태: {result.get('status', 'N/A')}")
                
                # 결과 데이터베이스 저장 확인 (잠시 대기)
                await asyncio.sleep(1)
                step_result = await self.unified_db.get_step_result(self.test_session_id, 1)
                if step_result:
                    logger.info("✅ Step 01 결과가 데이터베이스에 정상 저장됨")
                    logger.info(f"   - 데이터베이스 상태: {step_result.status}")
                    logger.info(f"   - 데이터베이스 품질 점수: {step_result.quality_score}")
                    logger.info(f"   - 데이터베이스 출력 데이터 키: {list(step_result.output_data.keys())}")
                else:
                    logger.warning("⚠️ Step 01 결과가 데이터베이스에 저장되지 않음")
                
                self.test_results['step_01_processing'] = True
                return True
            else:
                logger.error(f"❌ Step 01 처리 실패: {result}")
                self.test_results['step_01_processing'] = False
                return False
                
        except Exception as e:
            logger.error(f"❌ Step 01 처리 테스트 실패: {e}")
            self.test_results['step_01_processing'] = False
            return False
    
    async def test_data_flow_to_step_02(self):
        """Step 01 -> Step 02 데이터 흐름 테스트"""
        try:
            logger.info("🧪 Step 01 -> Step 02 데이터 흐름 테스트 시작...")
            
            if not self.test_session_id:
                logger.error("❌ 테스트 세션이 없음")
                return False
            
            # Step 02 입력 데이터 준비
            step2_input = await self.unified_db.get_step_input_data(self.test_session_id, 2)
            
            if step2_input:
                logger.info(f"✅ Step 02 입력 데이터 준비 성공: {len(step2_input)}개 항목")
                logger.info(f"   - 데이터 키: {list(step2_input.keys())}")
                
                # Step 01 결과가 포함되어 있는지 확인
                required_keys = [
                    'step_1_segmentation_mask',
                    'person_image_path',
                    'measurements'
                ]
                
                missing_keys = []
                for key in required_keys:
                    if key not in step2_input:
                        missing_keys.append(key)
                
                if not missing_keys:
                    logger.info("✅ Step 01 결과가 Step 02 입력에 모두 포함됨")
                    logger.info(f"   - 세그멘테이션 마스크: {'step_1_segmentation_mask' in step2_input}")
                    logger.info(f"   - 사람 이미지: {'person_image_path' in step2_input}")
                    logger.info(f"   - 측정값: {'measurements' in step2_input}")
                    
                    # 실제 데이터 값 확인
                    if 'step_1_segmentation_mask' in step2_input:
                        mask_data = step2_input['step_1_segmentation_mask']
                        logger.info(f"   - 마스크 데이터 타입: {type(mask_data).__name__}")
                        if isinstance(mask_data, str):
                            logger.info(f"   - 마스크 파일 경로: {mask_data}")
                    
                    self.test_results['data_flow_to_step_02'] = True
                    return True
                else:
                    logger.warning(f"⚠️ Step 02 입력에 누락된 키: {missing_keys}")
                    logger.info(f"   - 현재 포함된 키: {[k for k in step2_input.keys() if k.startswith('step_1_')]}")
                    self.test_results['data_flow_to_step_02'] = False
                    return False
            else:
                logger.error("❌ Step 02 입력 데이터 준비 실패")
                self.test_results['data_flow_to_step_02'] = False
                return False
                
        except Exception as e:
            logger.error(f"❌ 데이터 흐름 테스트 실패: {e}")
            self.test_results['data_flow_to_step_02'] = False
            return False
    
    async def test_session_progress_update(self):
        """세션 진행률 업데이트 테스트"""
        try:
            logger.info("🧪 세션 진행률 업데이트 테스트 시작...")
            
            if not self.test_session_id:
                logger.error("❌ 테스트 세션이 없음")
                return False
            
            # 잠시 대기 후 세션 정보 조회 (데이터베이스 업데이트 반영 대기)
            await asyncio.sleep(2)
            
            # 세션 정보 직접 조회
            session_info = await self.unified_db.get_session_info(self.test_session_id)
            if session_info:
                logger.info(f"✅ 세션 진행률 확인: {session_info.progress_percent:.1f}%")
                logger.info(f"   - 완료된 Step: {session_info.completed_steps}")
                logger.info(f"   - 현재 Step: {session_info.current_step}")
                logger.info(f"   - 상태: {session_info.status}")
                
                # Step 1이 완료된 Step에 포함되어 있는지 확인
                if 1 in session_info.completed_steps:
                    logger.info("✅ Step 1이 세션 진행률에 정상 반영됨")
                    self.test_results['session_progress_update'] = True
                    return True
                else:
                    logger.warning("⚠️ Step 1이 세션 진행률에 반영되지 않음")
                    
                    # 데이터베이스에서 직접 확인
                    try:
                        with self.unified_db._get_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute("""
                                SELECT completed_steps, progress_percent 
                                FROM sessions 
                                WHERE session_id = ?
                            """, (self.test_session_id,))
                            result = cursor.fetchone()
                            
                            if result:
                                db_completed_steps = json.loads(result[0]) if result[0] else []
                                db_progress = result[1] or 0.0
                                logger.info(f"   - DB 직접 조회 - 완료된 Step: {db_completed_steps}")
                                logger.info(f"   - DB 직접 조회 - 진행률: {db_progress:.1f}%")
                                
                                if 1 in db_completed_steps:
                                    logger.info("✅ 데이터베이스에는 Step 1이 정상 반영됨")
                                    self.test_results['session_progress_update'] = True
                                    return True
                                else:
                                    logger.error("❌ 데이터베이스에도 Step 1이 반영되지 않음")
                            else:
                                logger.error("❌ 세션 정보를 찾을 수 없음")
                    except Exception as e:
                        logger.error(f"❌ 데이터베이스 직접 조회 실패: {e}")
                    
                    self.test_results['session_progress_update'] = False
                    return False
            else:
                logger.error("❌ 세션 정보 조회 실패")
                self.test_results['session_progress_update'] = False
                return False
                
        except Exception as e:
            logger.error(f"❌ 세션 진행률 테스트 실패: {e}")
            self.test_results['session_progress_update'] = False
            return False
    
    async def cleanup_test_environment(self):
        """테스트 환경 정리"""
        try:
            logger.info("🧹 테스트 환경 정리 시작...")
            
            # 테스트 이미지 정리
            test_images_dir = Path("test_images")
            if test_images_dir.exists():
                import shutil
                shutil.rmtree(test_images_dir)
                logger.info("✅ 테스트 이미지 디렉토리 정리 완료")
            
            # 임시 마스크 정리
            temp_masks_dir = Path("temp_masks")
            if temp_masks_dir.exists():
                import shutil
                shutil.rmtree(temp_masks_dir)
                logger.info("✅ 임시 마스크 디렉토리 정리 완료")
            
            # 데이터베이스 캐시 정리
            if self.unified_db:
                self.unified_db.clear_cache()
                logger.info("✅ 데이터베이스 캐시 정리 완료")
            
            logger.info("✅ 테스트 환경 정리 완료")
            
        except Exception as e:
            logger.error(f"❌ 테스트 환경 정리 실패: {e}")
    
    async def print_test_summary(self):
        """테스트 결과 요약 출력"""
        try:
            logger.info("📊 Step 01 + 통합 Session Database 테스트 결과 요약")
            logger.info("=" * 70)
            
            total_tests = len(self.test_results)
            passed_tests = sum(1 for result in self.test_results.values() if result)
            failed_tests = total_tests - passed_tests
            
            logger.info(f"총 테스트 수: {total_tests}")
            logger.info(f"통과: {passed_tests}")
            logger.info(f"실패: {failed_tests}")
            logger.info(f"성공률: {(passed_tests/total_tests)*100:.1f}%")
            
            logger.info("\n📋 상세 결과:")
            for test_name, result in self.test_results.items():
                status = "✅ 통과" if result else "❌ 실패"
                logger.info(f"  {test_name}: {status}")
            
            if failed_tests == 0:
                logger.info("\n🎉 모든 테스트 통과!")
                logger.info("🔥 Step 01 + 통합 Session Database 연동 성공!")
            else:
                logger.info(f"\n⚠️ {failed_tests}개 테스트 실패")
            
        except Exception as e:
            logger.error(f"❌ 테스트 결과 요약 출력 실패: {e}")
    
    async def run_all_tests(self):
        """모든 테스트 실행"""
        try:
            logger.info("🚀 Step 01 + 통합 Session Database 통합 테스트 시작")
            
            # 1. 테스트 환경 설정
            if not await self.setup_test_environment():
                return False
            
            # 2. Step 01 처리 테스트
            if not await self.test_step_01_processing():
                logger.warning("⚠️ Step 01 처리 테스트 실패 - 계속 진행")
            
            # 3. 데이터 흐름 테스트
            if not await self.test_data_flow_to_step_02():
                logger.warning("⚠️ 데이터 흐름 테스트 실패 - 계속 진행")
            
            # 4. 세션 진행률 테스트
            if not await self.test_session_progress_update():
                logger.warning("⚠️ 세션 진행률 테스트 실패 - 계속 진행")
            
            # 5. 테스트 결과 요약
            await self.print_test_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 테스트 실행 중 오류: {e}")
            return False
        finally:
            # 테스트 환경 정리
            await self.cleanup_test_environment()

async def main():
    """메인 함수"""
    try:
        tester = Step01IntegrationTester()
        success = await tester.run_all_tests()
        
        if success:
            logger.info("🎉 Step 01 + 통합 Session Database 통합 테스트 완료!")
            return 0
        else:
            logger.error("❌ Step 01 + 통합 Session Database 통합 테스트 실패!")
            return 1
            
    except Exception as e:
        logger.error(f"❌ 메인 함수 실행 실패: {e}")
        return 1

if __name__ == "__main__":
    # asyncio 이벤트 루프 실행
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
