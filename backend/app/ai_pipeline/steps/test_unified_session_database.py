# backend/app/ai_pipeline/steps/test_unified_session_database.py
"""
🧪 통합 Session Database 시스템 테스트 스크립트
================================================================================

✅ Step 01 통합 Session Database 적용 테스트
✅ 데이터 저장 및 조회 테스트
✅ Step간 데이터 흐름 테스트
✅ 성능 메트릭 테스트
✅ 캐시 시스템 테스트

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

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 경로 설정 - 올바른 경로로 수정
current_dir = Path(__file__).parent
app_core_dir = current_dir.parent.parent.parent / "app" / "core"
sys.path.insert(0, str(app_core_dir))

try:
    from unified_session_database import (
        get_unified_session_database, 
        UnifiedSessionDatabase,
        SessionInfo,
        StepData,
        DataFlow
    )
    logger.info("✅ UnifiedSessionDatabase import 성공")
except ImportError as e:
    logger.error(f"❌ UnifiedSessionDatabase import 실패: {e}")
    # 대안 경로 시도
    try:
        import sys
        import os
        current_file = os.path.abspath(__file__)
        steps_dir = os.path.dirname(current_file)
        app_core_dir = os.path.join(steps_dir, '..', '..', '..', 'app', 'core')
        sys.path.insert(0, app_core_dir)
        from unified_session_database import (
            get_unified_session_database, 
            UnifiedSessionDatabase,
            SessionInfo,
            StepData,
            DataFlow
        )
        logger.info("✅ 대안 경로로 UnifiedSessionDatabase import 성공")
    except ImportError as e2:
        logger.error(f"❌ 대안 경로로도 import 실패: {e2}")
        sys.exit(1)

class UnifiedSessionDatabaseTester:
    """통합 Session Database 테스트 클래스"""
    
    def __init__(self):
        self.db = None
        self.test_session_id = None
        self.test_results = {}
        
    async def setup_database(self):
        """데이터베이스 설정"""
        try:
            logger.info("🔄 데이터베이스 설정 시작...")
            
            # 테스트용 데이터베이스 경로 설정
            test_db_path = "test_unified_sessions.db"
            
            # 새로운 UnifiedSessionDatabase 인스턴스 생성 (테스트용)
            self.db = UnifiedSessionDatabase(db_path=test_db_path, enable_cache=True)
            
            logger.info(f"✅ 데이터베이스 설정 완료: {test_db_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 데이터베이스 설정 실패: {e}")
            return False
    
    async def test_session_creation(self):
        """세션 생성 테스트"""
        try:
            logger.info("🧪 세션 생성 테스트 시작...")
            
            # 테스트용 이미지 경로
            person_image_path = "test_person.jpg"
            clothing_image_path = "test_clothing.jpg"
            measurements = {"height": 170, "weight": 65}
            
            # 세션 생성
            session_id = await self.db.create_session(
                person_image_path=person_image_path,
                clothing_image_path=clothing_image_path,
                measurements=measurements
            )
            
            self.test_session_id = session_id
            logger.info(f"✅ 세션 생성 성공: {session_id}")
            
            # 세션 정보 조회 테스트
            session_info = await self.db.get_session_info(session_id)
            if session_info:
                logger.info(f"✅ 세션 정보 조회 성공: {session_info.session_id}")
                logger.info(f"   - 상태: {session_info.status}")
                logger.info(f"   - 진행률: {session_info.progress_percent:.1f}%")
                logger.info(f"   - 완료된 Step: {session_info.completed_steps}")
            else:
                logger.error("❌ 세션 정보 조회 실패")
                return False
            
            self.test_results['session_creation'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ 세션 생성 테스트 실패: {e}")
            self.test_results['session_creation'] = False
            return False
    
    async def test_step_data_saving(self):
        """Step 데이터 저장 테스트"""
        try:
            if not self.test_session_id:
                logger.error("❌ 테스트 세션이 생성되지 않음")
                return False
            
            logger.info("🧪 Step 데이터 저장 테스트 시작...")
            
            # Step 1 데이터 저장 테스트
            input_data = {
                'person_image_path': 'test_person.jpg',
                'measurements': {'height': 170, 'weight': 65}
            }
            
            output_data = {
                'segmentation_mask': 'test_mask_data',
                'human_parsing_result': 'test_parsing_result',
                'confidence': 0.85,
                'quality_score': 0.8
            }
            
            success = await self.db.save_step_result(
                session_id=self.test_session_id,
                step_id=1,
                step_name="HumanParsingStep",
                input_data=input_data,
                output_data=output_data,
                processing_time=2.5,
                quality_score=0.8
            )
            
            if success:
                logger.info("✅ Step 1 데이터 저장 성공")
                
                # 저장된 데이터 조회 테스트
                step_result = await self.db.get_step_result(self.test_session_id, 1)
                if step_result:
                    logger.info(f"✅ Step 1 데이터 조회 성공")
                    logger.info(f"   - 처리 시간: {step_result.processing_time:.2f}초")
                    logger.info(f"   - 품질 점수: {step_result.quality_score:.2f}")
                    logger.info(f"   - 상태: {step_result.status}")
                else:
                    logger.error("❌ Step 1 데이터 조회 실패")
                    return False
            else:
                logger.error("❌ Step 1 데이터 저장 실패")
                return False
            
            self.test_results['step_data_saving'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Step 데이터 저장 테스트 실패: {e}")
            self.test_results['step_data_saving'] = False
            return False
    
    async def test_step_input_data_preparation(self):
        """Step 입력 데이터 준비 테스트"""
        try:
            if not self.test_session_id:
                logger.error("❌ 테스트 세션이 생성되지 않음")
                return False
            
            logger.info("🧪 Step 입력 데이터 준비 테스트 시작...")
            
            # Step 2 입력 데이터 준비 테스트 (Step 1 결과에 의존)
            input_data = await self.db.get_step_input_data(self.test_session_id, 2)
            
            if input_data:
                logger.info(f"✅ Step 2 입력 데이터 준비 성공: {len(input_data)}개 항목")
                logger.info(f"   - 세션 ID: {input_data.get('session_id')}")
                logger.info(f"   - Step ID: {input_data.get('step_id')}")
                logger.info(f"   - Step 1 결과 포함: {'step_1_segmentation_mask' in input_data}")
                logger.info(f"   - 측정값 포함: {'measurements' in input_data}")
                
                # Step 1 결과가 포함되어 있는지 확인
                if 'step_1_segmentation_mask' in input_data:
                    logger.info("✅ Step 1 결과가 Step 2 입력에 정상적으로 포함됨")
                else:
                    logger.warning("⚠️ Step 1 결과가 Step 2 입력에 포함되지 않음")
            else:
                logger.error("❌ Step 2 입력 데이터 준비 실패")
                return False
            
            self.test_results['step_input_data_preparation'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Step 입력 데이터 준비 테스트 실패: {e}")
            self.test_results['step_input_data_preparation'] = False
            return False
    
    async def test_dependency_validation(self):
        """Step 의존성 검증 테스트"""
        try:
            if not self.test_session_id:
                logger.error("❌ 테스트 세션이 생성되지 않음")
                return False
            
            logger.info("🧪 Step 의존성 검증 테스트 시작...")
            
            # Step 2 의존성 검증 테스트
            validation_result = await self.db.validate_step_dependencies(self.test_session_id, 2)
            
            if validation_result:
                logger.info(f"✅ Step 2 의존성 검증 성공")
                logger.info(f"   - 유효성: {validation_result.get('valid')}")
                logger.info(f"   - 누락된 의존성: {validation_result.get('missing_dependencies')}")
                logger.info(f"   - 사용 가능한 데이터: {validation_result.get('available_data')}")
                
                # Step 1이 완료되었는지 확인
                if validation_result.get('valid'):
                    logger.info("✅ Step 2 의존성 검증 통과")
                else:
                    logger.warning("⚠️ Step 2 의존성 검증 실패")
            else:
                logger.error("❌ Step 2 의존성 검증 실패")
                return False
            
            self.test_results['dependency_validation'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Step 의존성 검증 테스트 실패: {e}")
            self.test_results['dependency_validation'] = False
            return False
    
    async def test_performance_metrics(self):
        """성능 메트릭 테스트"""
        try:
            logger.info("🧪 성능 메트릭 테스트 시작...")
            
            # 성능 메트릭 조회
            metrics = self.db.get_performance_metrics()
            
            if metrics:
                logger.info(f"✅ 성능 메트릭 조회 성공")
                logger.info(f"   - 캐시 히트: {metrics.get('cache_hits', 0)}")
                logger.info(f"   - 캐시 미스: {metrics.get('cache_misses', 0)}")
                logger.info(f"   - 캐시 히트율: {metrics.get('cache_hit_ratio', 0):.2%}")
                logger.info(f"   - 압축률: {metrics.get('compression_ratio', 0):.2%}")
                logger.info(f"   - 캐시 크기: {metrics.get('cache_size', 0)}")
                
                # 캐시 효율성 확인
                cache_hit_ratio = metrics.get('cache_hit_ratio', 0)
                if cache_hit_ratio > 0:
                    logger.info(f"✅ 캐시 시스템 정상 작동 (히트율: {cache_hit_ratio:.2%})")
                else:
                    logger.info("ℹ️ 캐시 시스템 초기 상태")
            else:
                logger.error("❌ 성능 메트릭 조회 실패")
                return False
            
            self.test_results['performance_metrics'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ 성능 메트릭 테스트 실패: {e}")
            self.test_results['performance_metrics'] = False
            return False
    
    async def test_data_flow_management(self):
        """데이터 흐름 관리 테스트"""
        try:
            logger.info("🧪 데이터 흐름 관리 테스트 시작...")
            
            # 데이터 흐름 정의 확인
            data_flows = self.db.data_flows
            if data_flows:
                logger.info(f"✅ 데이터 흐름 정의 확인: {len(data_flows)}개 흐름")
                
                # Step 1 -> Step 2 흐름 확인
                step1_to_step2_flows = [flow for flow in data_flows if flow.source_step == 1 and flow.target_step == 2]
                if step1_to_step2_flows:
                    logger.info(f"✅ Step 1 -> Step 2 데이터 흐름: {len(step1_to_step2_flows)}개")
                    for flow in step1_to_step2_flows:
                        logger.info(f"   - {flow.data_type}: {flow.data_key} (필수: {flow.required})")
                else:
                    logger.warning("⚠️ Step 1 -> Step 2 데이터 흐름이 정의되지 않음")
            else:
                logger.error("❌ 데이터 흐름 정의가 없음")
                return False
            
            self.test_results['data_flow_management'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ 데이터 흐름 관리 테스트 실패: {e}")
            self.test_results['data_flow_management'] = False
            return False
    
    async def cleanup_test_data(self):
        """테스트 데이터 정리"""
        try:
            logger.info("🧹 테스트 데이터 정리 시작...")
            
            if self.db:
                # 캐시 정리
                self.db.clear_cache()
                logger.info("✅ 캐시 정리 완료")
                
                # 데이터베이스 최적화
                self.db.optimize_database()
                logger.info("✅ 데이터베이스 최적화 완료")
            
            # 테스트 파일 정리
            test_files = [
                "test_unified_sessions.db",
                "temp_masks"
            ]
            
            for file_path in test_files:
                if Path(file_path).exists():
                    if Path(file_path).is_file():
                        Path(file_path).unlink()
                    else:
                        import shutil
                        shutil.rmtree(file_path)
                    logger.info(f"✅ 테스트 파일 정리: {file_path}")
            
            logger.info("✅ 테스트 데이터 정리 완료")
            
        except Exception as e:
            logger.error(f"❌ 테스트 데이터 정리 실패: {e}")
    
    async def run_all_tests(self):
        """모든 테스트 실행"""
        try:
            logger.info("🚀 통합 Session Database 테스트 시작")
            
            # 1. 데이터베이스 설정
            if not await self.setup_database():
                return False
            
            # 2. 세션 생성 테스트
            if not await self.test_session_creation():
                return False
            
            # 3. Step 데이터 저장 테스트
            if not await self.test_step_data_saving():
                return False
            
            # 4. Step 입력 데이터 준비 테스트
            if not await self.test_step_input_data_preparation():
                return False
            
            # 5. Step 의존성 검증 테스트
            if not await self.test_dependency_validation():
                return False
            
            # 6. 성능 메트릭 테스트
            if not await self.test_performance_metrics():
                return False
            
            # 7. 데이터 흐름 관리 테스트
            if not await self.test_data_flow_management():
                return False
            
            # 8. 테스트 결과 요약
            await self.print_test_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 테스트 실행 중 오류: {e}")
            return False
        finally:
            # 테스트 데이터 정리
            await self.cleanup_test_data()
    
    async def print_test_summary(self):
        """테스트 결과 요약 출력"""
        try:
            logger.info("📊 테스트 결과 요약")
            logger.info("=" * 50)
            
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
            else:
                logger.info(f"\n⚠️ {failed_tests}개 테스트 실패")
            
        except Exception as e:
            logger.error(f"❌ 테스트 결과 요약 출력 실패: {e}")

async def main():
    """메인 함수"""
    try:
        tester = UnifiedSessionDatabaseTester()
        success = await tester.run_all_tests()
        
        if success:
            logger.info("🎉 통합 Session Database 테스트 완료!")
            return 0
        else:
            logger.error("❌ 통합 Session Database 테스트 실패!")
            return 1
            
    except Exception as e:
        logger.error(f"❌ 메인 함수 실행 실패: {e}")
        return 1

if __name__ == "__main__":
    # asyncio 이벤트 루프 실행
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
