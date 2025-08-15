#!/usr/bin/env python3
"""
🧪 MyCloset AI - 전체 파이프라인 통합 테스트
============================================

모든 Step들이 통합 Session Database와 올바르게 연동되는지 테스트
Step 01 → Step 02 → Step 03 → Step 04 → Step 05 → Step 06 → Step 07 → Step 08 → Step 09

Author: MyCloset AI Team
Date: 2025-08-14
Version: 1.0
"""

import asyncio
import logging
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 통합 Session Database import
try:
    from ...core.unified_session_database import get_unified_session_database
    UNIFIED_SESSION_DB_AVAILABLE = True
    logging.info("✅ UnifiedSessionDatabase import 성공")
except ImportError:
    try:
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        core_dir = os.path.join(current_dir, '..', '..', '..', 'core')
        sys.path.insert(0, core_dir)
        from unified_session_database import get_unified_session_database
        UNIFIED_SESSION_DB_AVAILABLE = True
        logging.info("✅ 경로 조작으로 UnifiedSessionDatabase import 성공")
    except ImportError:
        UNIFIED_SESSION_DB_AVAILABLE = False
        logging.error("❌ UnifiedSessionDatabase import 실패")

# Step들 import - 개별 상태 추적 및 실제 import 확인
STEPS_AVAILABLE = {}
STEP_IMPORTS = {}

try:
    from .step_01_human_parsing_models.step_01_human_parsing import create_human_parsing_step
    STEPS_AVAILABLE['step_01'] = True
    STEP_IMPORTS['step_01'] = create_human_parsing_step
    logging.info("✅ Step 01 import 성공")
except ImportError as e:
    STEPS_AVAILABLE['step_01'] = False
    logging.warning(f"⚠️ Step 01 import 실패: {e}")

try:
    from .step_02_pose_estimation_models.step_02_pose_estimation import create_pose_estimation_step
    STEPS_AVAILABLE['step_02'] = True
    STEP_IMPORTS['step_02'] = create_pose_estimation_step
    logging.info("✅ Step 02 import 성공")
except ImportError as e:
    STEPS_AVAILABLE['step_02'] = False
    logging.warning(f"⚠️ Step 02 import 실패: {e}")

try:
    from .step_03_cloth_segmentation_models.step_03_cloth_segmentation import create_cloth_segmentation_step
    STEPS_AVAILABLE['step_03'] = True
    STEP_IMPORTS['step_03'] = create_cloth_segmentation_step
    logging.info("✅ Step 03 import 성공")
except ImportError as e:
    STEPS_AVAILABLE['step_03'] = False
    logging.warning(f"⚠️ Step 03 import 실패: {e}")

try:
    from .step_04_geometric_matching_models.step_04_geometric_matching import create_geometric_matching_step
    STEPS_AVAILABLE['step_04'] = True
    STEP_IMPORTS['step_04'] = create_geometric_matching_step
    logging.info("✅ Step 04 import 성공")
except ImportError as e:
    STEPS_AVAILABLE['step_04'] = False
    logging.warning(f"⚠️ Step 04 import 실패: {e}")

try:
    from .step_05_cloth_warping_models.step_05_cloth_warping import create_cloth_warping_step
    STEPS_AVAILABLE['step_05'] = True
    STEP_IMPORTS['step_05'] = create_cloth_warping_step
    logging.info("✅ Step 05 import 성공")
except ImportError as e:
    STEPS_AVAILABLE['step_05'] = False
    logging.warning(f"⚠️ Step 05 import 실패: {e}")

try:
    from .step_06_virtual_fitting_models.step_06_virtual_fitting import create_virtual_fitting_step
    STEPS_AVAILABLE['step_06'] = True
    STEP_IMPORTS['step_06'] = create_virtual_fitting_step
    logging.info("✅ Step 06 import 성공")
except ImportError as e:
    STEPS_AVAILABLE['step_06'] = False
    logging.warning(f"⚠️ Step 06 import 실패: {e}")

try:
    from .step_07_post_processing_models.step_07_post_processing import create_post_processing_step
    STEPS_AVAILABLE['step_07'] = True
    STEP_IMPORTS['step_07'] = create_post_processing_step
    logging.info("✅ Step 07 import 성공")
except ImportError as e:
    STEPS_AVAILABLE['step_07'] = False
    logging.warning(f"⚠️ Step 07 import 실패: {e}")

try:
    from .step_08_quality_assessment_models.step_08_quality_assessment import create_quality_assessment_step
    STEPS_AVAILABLE['step_08'] = True
    STEP_IMPORTS['step_08'] = create_quality_assessment_step
    logging.info("✅ Step 08 import 성공")
except ImportError as e:
    STEPS_AVAILABLE['step_08'] = False
    logging.warning(f"⚠️ Step 08 import 실패: {e}")

try:
    from .step_09_final_output_models.step_09_final_output import create_final_output_step
    STEPS_AVAILABLE['step_09'] = True
    STEP_IMPORTS['step_09'] = create_final_output_step
    logging.info("✅ Step 09 import 성공")
except ImportError as e:
    STEPS_AVAILABLE['step_09'] = False
    logging.warning(f"⚠️ Step 09 import 실패: {e}")

# 전체 Step 가용성 확인
available_steps = [step for step, available in STEPS_AVAILABLE.items() if available]
logging.info(f"📊 사용 가능한 Steps: {len(available_steps)}/{len(STEPS_AVAILABLE)} ({', '.join(available_steps)})")

class FullPipelineIntegrationTest:
    """전체 AI 파이프라인 통합 테스트"""
    
    def __init__(self):
        self.unified_db = None
        self.test_session_id = None
        self.test_results = {}
        
        if UNIFIED_SESSION_DB_AVAILABLE:
            self.unified_db = get_unified_session_database()
            logging.info("✅ 통합 Session Database 연결 성공")
        else:
            logging.error("❌ 통합 Session Database 연결 실패")
    
    def create_test_image(self, width: int = 1024, height: int = 1024, filename: str = "test_image.jpg") -> str:
        """테스트용 이미지 생성"""
        try:
            # PIL을 사용하여 테스트 이미지 생성
            from PIL import Image, ImageDraw
            
            # 빈 이미지 생성
            image = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(image)
            
            # 간단한 도형 그리기 (사람 모양)
            # 머리 (원)
            head_center = (width // 2, height // 4)
            head_radius = min(width, height) // 8
            draw.ellipse([
                head_center[0] - head_radius, 
                head_center[1] - head_radius,
                head_center[0] + head_radius, 
                head_center[1] + head_radius
            ], fill='pink', outline='black', width=2)
            
            # 몸통 (사각형)
            body_top = head_center[1] + head_radius
            body_bottom = height * 3 // 4
            body_left = width // 2 - width // 6
            body_right = width // 2 + width // 6
            draw.rectangle([body_left, body_top, body_right, body_bottom], 
                         fill='lightblue', outline='black', width=2)
            
            # 팔 (선)
            arm_y = body_top + (body_bottom - body_top) // 3
            draw.line([body_left, arm_y, body_left - width // 4, arm_y], 
                     fill='pink', width=3)  # 왼팔
            draw.line([body_right, arm_y, body_right + width // 4, arm_y], 
                     fill='pink', width=3)  # 오른팔
            
            # 다리 (선)
            leg_y = body_bottom
            draw.line([body_left + width // 12, leg_y, body_left - width // 8, height], 
                     fill='black', width=3)  # 왼다리
            draw.line([body_right - width // 12, leg_y, body_right + width // 8, height], 
                     fill='black', width=3)  # 오른다리
            
            # 이미지 저장
            temp_dir = Path("/tmp")
            temp_dir.mkdir(exist_ok=True)
            image_path = temp_dir / filename
            image.save(image_path, "JPEG", quality=95)
            
            logging.info(f"✅ 테스트 이미지 생성 완료: {image_path}")
            return str(image_path)
            
        except Exception as e:
            logging.error(f"❌ 테스트 이미지 생성 실패: {e}")
            return None

    def create_test_clothing_image(self, width: int = 512, height: int = 512, filename: str = "test_clothing.jpg") -> str:
        """테스트용 의류 이미지 생성"""
        try:
            from PIL import Image, ImageDraw
            
            # 의류 이미지 생성
            image = Image.new('RGB', (width, height), color='lightgreen')
            draw = ImageDraw.Draw(image)
            
            # 간단한 셔츠 모양
            # 셔츠 본체
            shirt_top = height // 4
            shirt_bottom = height * 3 // 4
            shirt_left = width // 4
            shirt_right = width * 3 // 4
            draw.rectangle([shirt_left, shirt_top, shirt_right, shirt_bottom], 
                         fill='lightblue', outline='darkblue', width=2)
            
            # 소매
            sleeve_width = width // 6
            draw.rectangle([0, shirt_top, sleeve_width, shirt_bottom], 
                         fill='lightblue', outline='darkblue', width=2)  # 왼소매
            draw.rectangle([width - sleeve_width, shirt_top, width, shirt_bottom], 
                         fill='lightblue', outline='darkblue', width=2)  # 오른소매
            
            # 목 부분
            neck_width = width // 8
            neck_height = height // 8
            neck_x = width // 2 - neck_width // 2
            neck_y = shirt_top - neck_height
            draw.rectangle([neck_x, neck_y, neck_x + neck_width, neck_y + neck_height], 
                         fill='white', outline='darkblue', width=2)
            
            # 이미지 저장
            temp_dir = Path("/tmp")
            temp_dir.mkdir(exist_ok=True)
            image_path = temp_dir / filename
            image.save(image_path, "JPEG", quality=95)
            
            logging.info(f"✅ 테스트 의류 이미지 생성 완료: {image_path}")
            return str(image_path)
            
        except Exception as e:
            logging.error(f"❌ 테스트 의류 이미지 생성 실패: {e}")
            return None
    
    async def create_test_session(self) -> str:
        """테스트 세션 생성"""
        try:
            if not self.unified_db:
                raise RuntimeError("통합 Session Database가 연결되지 않음")
            
            # 세션 정보 생성
            session_info = {
                'user_id': 'test_user',
                'measurements': {
                    'height': 170,
                    'weight': 65,
                    'chest': 90,
                    'waist': 75,
                    'hip': 95
                },
                'metadata': {
                    'test_mode': True,
                    'pipeline_version': '2.0'
                }
            }
            
            # 실제 세션 생성 (session_id 반환)
            session_id = await self.unified_db.create_session(
                person_image_path="test_image.png",
                clothing_image_path="test_clothing.png",
                measurements=session_info['measurements']
            )
            
            if session_id:
                self.test_session_id = session_id
                logging.info(f"✅ 테스트 세션 생성 성공: {session_id}")
                return session_id
            else:
                raise RuntimeError("세션 생성 실패")
                
        except Exception as e:
            logging.error(f"❌ 테스트 세션 생성 실패: {e}")
            return None
    
    async def test_step_01(self, session_id: str) -> bool:
        """Step 01 (Human Parsing) 테스트"""
        try:
            logging.info("🧪 Step 01 (Human Parsing) 테스트 시작")
            
            # Step 01 가용성 확인
            if 'step_01' not in STEPS_AVAILABLE or not STEPS_AVAILABLE['step_01']:
                logging.error("❌ Step 01을 사용할 수 없습니다")
                return False
            
            # 테스트 이미지 생성
            person_image_path = self.create_test_image(512, 512, "test_person_step01.jpg")
            clothing_image_path = self.create_test_clothing_image(256, 256, "test_clothing_step01.jpg")
            
            if not person_image_path or not clothing_image_path:
                logging.error("❌ 테스트 이미지 생성 실패")
                return False
            
            # Step 01 생성
            step_01 = STEP_IMPORTS['step_01']()
            
            # 테스트 입력 데이터
            test_input = {
                'session_id': session_id,
                'person_image_path': person_image_path,
                'clothing_image_path': clothing_image_path
            }
            
            # Step 01 실행
            result = await step_01.process(test_input)
            
            # 결과 형식 확인 및 처리
            if result:
                if hasattr(result, 'status'):
                    # StepResult 객체인 경우
                    if result.status == 'completed':
                        logging.info("✅ Step 01 테스트 성공")
                        return True
                    else:
                        logging.error(f"❌ Step 01 실행 실패: {result.status}")
                        return False
                elif isinstance(result, dict):
                    # 딕셔너리인 경우
                    if result.get('success') or result.get('status') == 'completed':
                        logging.info("✅ Step 01 테스트 성공")
                        return True
                    else:
                        logging.error(f"❌ Step 01 실행 실패: {result}")
                        return False
                else:
                    logging.error(f"❌ Step 01 결과 형식 오류: {type(result)}")
                    return False
            else:
                logging.error("❌ Step 01 실행 실패: 결과가 None")
                return False
                
        except Exception as e:
            logging.error(f"❌ Step 01 테스트 중 오류: {e}")
            return False
    
    async def test_step_02(self, session_id: str) -> bool:
        """Step 02 (Pose Estimation) 테스트"""
        try:
            logging.info("🧪 Step 02 (Pose Estimation) 테스트 시작")
            
            # Step 02 가용성 확인
            if 'step_02' not in STEPS_AVAILABLE or not STEPS_AVAILABLE['step_02']:
                logging.error("❌ Step 02를 사용할 수 없습니다")
                return False
            
            # 테스트 이미지 생성
            person_image_path = self.create_test_image(512, 512, "test_person_step02.jpg")
            segmentation_mask_path = self.create_test_image(512, 512, "test_mask_step02.jpg")
            
            if not person_image_path or not segmentation_mask_path:
                logging.error("❌ 테스트 이미지 생성 실패")
                return False
            
            # Step 02 생성
            step_02 = STEP_IMPORTS['step_02']()
            
            # 테스트 입력 데이터
            test_input = {
                'session_id': session_id,
                'person_image_path': person_image_path,
                'segmentation_mask_path': segmentation_mask_path
            }
            
            # Step 02 실행
            result = await step_02.process(test_input)
            
            # 결과 형식 확인 및 처리
            if result:
                if hasattr(result, 'status'):
                    if result.status == 'completed':
                        logging.info("✅ Step 02 테스트 성공")
                        return True
                    else:
                        logging.error(f"❌ Step 02 실행 실패: {result.status}")
                        return False
                elif isinstance(result, dict):
                    if result.get('success') or result.get('status') == 'completed':
                        logging.info("✅ Step 02 테스트 성공")
                        return True
                    else:
                        logging.error(f"❌ Step 02 실행 실패: {result}")
                        return False
                else:
                    logging.error(f"❌ Step 02 결과 형식 오류: {type(result)}")
                    return False
            else:
                logging.error("❌ Step 02 실행 실패: 결과가 None")
                return False
                
        except Exception as e:
            logging.error(f"❌ Step 02 테스트 중 오류: {e}")
            return False

    async def test_step_03(self, session_id: str) -> bool:
        """Step 03 (Cloth Segmentation) 테스트"""
        try:
            logging.info("🧪 Step 03 (Cloth Segmentation) 테스트 시작")
            
            # Step 03 가용성 확인
            if 'step_03' not in STEPS_AVAILABLE or not STEPS_AVAILABLE['step_03']:
                logging.error("❌ Step 03을 사용할 수 없습니다")
                return False
            
            # 테스트 이미지 생성
            person_image_path = self.create_test_image(512, 512, "test_person_step03.jpg")
            segmentation_mask_path = self.create_test_image(512, 512, "test_mask_step03.jpg")
            
            if not person_image_path or not segmentation_mask_path:
                logging.error("❌ 테스트 이미지 생성 실패")
                return False
            
            # Step 03 생성
            step_03 = STEP_IMPORTS['step_03']()
            
            # 테스트 입력 데이터
            test_input = {
                'session_id': session_id,
                'person_image_path': person_image_path,
                'segmentation_mask_path': segmentation_mask_path
            }
            
            # Step 03 실행
            result = await step_03.process(test_input)
            
            # 결과 형식 확인 및 처리
            if result:
                if hasattr(result, 'status'):
                    if result.status == 'completed':
                        logging.info("✅ Step 03 테스트 성공")
                        return True
                    else:
                        logging.error(f"❌ Step 03 실행 실패: {result.status}")
                        return False
                elif isinstance(result, dict):
                    if result.get('success') or result.get('status') == 'completed':
                        logging.info("✅ Step 03 테스트 성공")
                        return True
                    else:
                        logging.error(f"❌ Step 03 실행 실패: {result}")
                        return False
                else:
                    logging.error(f"❌ Step 03 결과 형식 오류: {type(result)}")
                    return False
            else:
                logging.error("❌ Step 03 실행 실패: 결과가 None")
                return False
                
        except Exception as e:
            logging.error(f"❌ Step 03 테스트 중 오류: {e}")
            return False
    
    async def test_step_04(self, session_id: str) -> bool:
        """Step 04 (Geometric Matching) 테스트"""
        try:
            logging.info("🧪 Step 04 (Geometric Matching) 테스트 시작")
            
            # Step 04 가용성 확인
            if 'step_04' not in STEPS_AVAILABLE or not STEPS_AVAILABLE['step_04']:
                logging.error("❌ Step 04를 사용할 수 없습니다")
                return False
            
            # 테스트 이미지 생성
            person_image_path = self.create_test_image(512, 512, "test_person_step04.jpg")
            segmentation_mask_path = self.create_test_image(512, 512, "test_mask_step04.jpg")
            cloth_segmentation_mask_path = self.create_test_image(512, 512, "test_cloth_mask_step04.jpg")
            
            if not person_image_path or not segmentation_mask_path or not cloth_segmentation_mask_path:
                logging.error("❌ 테스트 이미지 생성 실패")
                return False
            
            # Step 04 생성
            step_04 = STEP_IMPORTS['step_04']()
            
            # 테스트 입력 데이터
            test_input = {
                'session_id': session_id,
                'person_image_path': person_image_path,
                'segmentation_mask_path': segmentation_mask_path,
                'cloth_segmentation_mask_path': cloth_segmentation_mask_path
            }
            
            # Step 04 실행
            result = await step_04.process(test_input)
            
            # 결과 형식 확인 및 처리
            if result:
                if hasattr(result, 'status'):
                    if result.status == 'completed':
                        logging.info("✅ Step 04 테스트 성공")
                        return True
                    else:
                        logging.error(f"❌ Step 04 실행 실패: {result.status}")
                        return False
                elif isinstance(result, dict):
                    if result.get('success') or result.get('status') == 'completed':
                        logging.info("✅ Step 04 테스트 성공")
                        return True
                    else:
                        logging.error(f"❌ Step 04 실행 실패: {result}")
                        return False
                else:
                    logging.error(f"❌ Step 04 결과 형식 오류: {type(result)}")
                    return False
            else:
                logging.error("❌ Step 04 실행 실패: 결과가 None")
                return False
                
        except Exception as e:
            logging.error(f"❌ Step 04 테스트 중 오류: {e}")
            return False

    async def test_step_05(self, session_id: str) -> bool:
        """Step 05 (Cloth Warping) 테스트"""
        try:
            logging.info("🧪 Step 05 (Cloth Warping) 테스트 시작")
            
            # Step 05 가용성 확인
            if 'step_05' not in STEPS_AVAILABLE or not STEPS_AVAILABLE['step_05']:
                logging.error("❌ Step 05를 사용할 수 없습니다")
                return False
            
            # 테스트 이미지 생성
            person_image_path = self.create_test_image(512, 512, "test_person_step05.jpg")
            segmentation_mask_path = self.create_test_image(512, 512, "test_mask_step05.jpg")
            cloth_segmentation_mask_path = self.create_test_image(512, 512, "test_cloth_mask_step05.jpg")
            
            if not person_image_path or not segmentation_mask_path or not cloth_segmentation_mask_path:
                logging.error("❌ 테스트 이미지 생성 실패")
                return False
            
            # Step 05 생성
            step_05 = STEP_IMPORTS['step_05']()
            
            # 테스트 입력 데이터
            test_input = {
                'session_id': session_id,
                'person_image_path': person_image_path,
                'segmentation_mask_path': segmentation_mask_path,
                'cloth_segmentation_mask_path': cloth_segmentation_mask_path,
                'transformation_matrix': np.eye(3)
            }
            
            # Step 05 실행
            result = await step_05.process(test_input)
            
            # 결과 형식 확인 및 처리
            if result:
                if hasattr(result, 'status'):
                    if result.status == 'completed':
                        logging.info("✅ Step 05 테스트 성공")
                        return True
                    else:
                        logging.error(f"❌ Step 05 실행 실패: {result.status}")
                        return False
                elif isinstance(result, dict):
                    if result.get('success') or result.get('status') == 'completed':
                        logging.info("✅ Step 05 테스트 성공")
                        return True
                    else:
                        logging.error(f"❌ Step 05 실행 실패: {result}")
                        return False
                else:
                    logging.error(f"❌ Step 05 결과 형식 오류: {type(result)}")
                    return False
            else:
                logging.error("❌ Step 05 실행 실패: 결과가 None")
                return False
                
        except Exception as e:
            logging.error(f"❌ Step 05 테스트 중 오류: {e}")
            return False

    async def test_step_06(self, session_id: str) -> bool:
        """Step 06 (Virtual Fitting) 테스트"""
        try:
            logging.info("🧪 Step 06 (Virtual Fitting) 테스트 시작")
            
            # Step 06 가용성 확인
            if 'step_06' not in STEPS_AVAILABLE or not STEPS_AVAILABLE['step_06']:
                logging.error("❌ Step 06을 사용할 수 없습니다")
                return False
            
            # Step 06 생성
            step_06 = STEP_IMPORTS['step_06']()
            
            # 테스트 입력 데이터
            test_input = {
                'session_id': session_id,
                'person_image_path': 'test_person.jpg',
                'segmentation_mask_path': 'test_mask.jpg',
                'warped_clothing_path': 'test_warped_clothing.jpg'
            }
            
            # Step 06 실행
            result = await step_06.process(test_input)
            
            # 결과 형식 확인 및 처리
            if result:
                if hasattr(result, 'status'):
                    # StepResult 객체인 경우
                    if result.status == 'completed':
                        logging.info("✅ Step 06 테스트 성공")
                        return True
                    else:
                        logging.error(f"❌ Step 06 실행 실패: {result.status}")
                        return False
                elif isinstance(result, dict):
                    # 딕셔너리인 경우
                    if result.get('success') or result.get('status') == 'completed':
                        logging.info("✅ Step 06 테스트 성공")
                        return True
                    else:
                        logging.error(f"❌ Step 06 실행 실패: {result}")
                        return False
                else:
                    logging.error(f"❌ Step 06 결과 형식 오류: {type(result)}")
                    return False
            else:
                logging.error("❌ Step 06 실행 실패: 결과가 None")
                return False
                
        except Exception as e:
            logging.error(f"❌ Step 06 테스트 중 오류: {e}")
            return False

    async def test_step_07(self, session_id: str) -> bool:
        """Step 07 (Post Processing) 테스트"""
        try:
            logging.info("🧪 Step 07 (Post Processing) 테스트 시작")
            
            # Step 07 가용성 확인
            if 'step_07' not in STEPS_AVAILABLE or not STEPS_AVAILABLE['step_07']:
                logging.error("❌ Step 07을 사용할 수 없습니다")
                return False
            
            # Step 07 생성
            step_07 = STEP_IMPORTS['step_07']()
            
            # 테스트 입력 데이터
            test_input = {
                'session_id': session_id,
                'fitted_image_path': 'test_fitted_image.jpg'
            }
            
            # Step 07 실행
            result = await step_07.process(test_input)
            
            # 결과 형식 확인 및 처리
            if result:
                if hasattr(result, 'status'):
                    # StepResult 객체인 경우
                    if result.status == 'completed':
                        logging.info("✅ Step 07 테스트 성공")
                        return True
                    else:
                        logging.error(f"❌ Step 07 실행 실패: {result.status}")
                        return False
                elif isinstance(result, dict):
                    # 딕셔너리인 경우
                    if result.get('success') or result.get('status') == 'completed':
                        logging.info("✅ Step 07 테스트 성공")
                        return True
                    else:
                        logging.error(f"❌ Step 07 실행 실패: {result}")
                        return False
                else:
                    logging.error(f"❌ Step 07 결과 형식 오류: {type(result)}")
                    return False
            else:
                logging.error("❌ Step 07 실행 실패: 결과가 None")
                return False
                
        except Exception as e:
            logging.error(f"❌ Step 07 테스트 중 오류: {e}")
            return False

    async def test_step_08(self, session_id: str) -> bool:
        """Step 08 (Quality Assessment) 테스트"""
        try:
            logging.info("🧪 Step 08 (Quality Assessment) 테스트 시작")
            
            # Step 08 가용성 확인
            if 'step_08' not in STEPS_AVAILABLE or not STEPS_AVAILABLE['step_08']:
                logging.error("❌ Step 08을 사용할 수 없습니다")
                return False
            
            # Step 08 생성
            step_08 = STEP_IMPORTS['step_08']()
            
            # 테스트 입력 데이터
            test_input = {
                'session_id': session_id,
                'processed_image_path': 'test_processed_image.jpg'
            }
            
            # Step 08 실행
            result = await step_08.process(test_input)
            
            # 결과 형식 확인 및 처리
            if result:
                if hasattr(result, 'status'):
                    # StepResult 객체인 경우
                    if result.status == 'completed':
                        logging.info("✅ Step 08 테스트 성공")
                        return True
                    else:
                        logging.error(f"❌ Step 08 실행 실패: {result.status}")
                        return False
                elif isinstance(result, dict):
                    # 딕셔너리인 경우
                    if result.get('success') or result.get('status') == 'completed':
                        logging.info("✅ Step 08 테스트 성공")
                        return True
                    else:
                        logging.error(f"❌ Step 08 실행 실패: {result}")
                        return False
                else:
                    logging.error(f"❌ Step 08 결과 형식 오류: {type(result)}")
                    return False
            else:
                logging.error("❌ Step 08 실행 실패: 결과가 None")
                return False
                
        except Exception as e:
            logging.error(f"❌ Step 08 테스트 중 오류: {e}")
            return False

    async def test_step_09(self, session_id: str) -> bool:
        """Step 09 (Final Output) 테스트"""
        try:
            logging.info("🧪 Step 09 (Final Output) 테스트 시작")
            
            # Step 09 가용성 확인
            if 'step_09' not in STEPS_AVAILABLE or not STEPS_AVAILABLE['step_09']:
                logging.error("❌ Step 09를 사용할 수 없습니다")
                return False
            
            # Step 09 생성
            step_09 = STEP_IMPORTS['step_09']()
            
            # 테스트 입력 데이터
            test_input = {
                'session_id': session_id,
                'final_image_path': 'test_final_image.jpg'
            }
            
            # Step 09 실행
            result = await step_09.process(test_input)
            
            # 결과 형식 확인 및 처리
            if result:
                if hasattr(result, 'status'):
                    # StepResult 객체인 경우
                    if result.status == 'completed':
                        logging.info("✅ Step 09 테스트 성공")
                        return True
                    else:
                        logging.error(f"❌ Step 09 실행 실패: {result.status}")
                        return False
                elif isinstance(result, dict):
                    # 딕셔너리인 경우
                    if result.get('success') or result.get('status') == 'completed':
                        logging.info("✅ Step 09 테스트 성공")
                        return True
                    else:
                        logging.error(f"❌ Step 09 실행 실패: {result}")
                        return False
                else:
                    logging.error(f"❌ Step 09 결과 형식 오류: {type(result)}")
                    return False
            else:
                logging.error("❌ Step 09 실행 실패: 결과가 None")
                return False
                
        except Exception as e:
            logging.error(f"❌ Step 09 테스트 중 오류: {e}")
            return False
    
    async def test_session_progress_tracking(self) -> bool:
        """세션 진행률 추적 테스트"""
        try:
            logging.info("🧪 세션 진행률 추적 테스트 시작")
            
            if not self.unified_db:
                raise RuntimeError("통합 Session Database가 연결되지 않음")
            
            # 세션 정보 조회
            await asyncio.sleep(2)  # DB 업데이트 대기
            session_info = await self.unified_db.get_session_info(self.test_session_id)
            
            if session_info:
                completed_steps = session_info.completed_steps
                progress_percent = session_info.progress_percent
                
                logging.info(f"📊 세션 진행률: {completed_steps}/9 Steps ({progress_percent:.1f}%)")
                
                # 모든 Step이 완료되었는지 확인
                if len(completed_steps) == 9 and progress_percent >= 100.0:
                    logging.info("✅ 모든 Step이 완료되고 진행률이 100%에 도달함")
                    return True
                else:
                    logging.warning(f"⚠️ 일부 Step이 완료되지 않음: {len(completed_steps)}/9")
                    return False
            else:
                logging.error("❌ 세션 정보를 조회할 수 없음")
                return False
                
        except Exception as e:
            logging.error(f"❌ 세션 진행률 추적 테스트 중 오류: {e}")
            return False
    
    async def test_data_flow_between_steps(self) -> bool:
        """Step 간 데이터 흐름 테스트"""
        try:
            logging.info("🧪 Step 간 데이터 흐름 테스트 시작")
            
            # 각 Step의 출력 데이터가 다음 Step의 입력 데이터로 올바르게 전달되었는지 확인
            data_flow_issues = []
            
            # Step 01 → Step 02 데이터 흐름 확인
            step_01_output = self.test_results.get('step_01', {})
            step_02_input_required = ['segmentation_mask', 'segmentation_mask_path', 'person_image_path']
            
            for key in step_02_input_required:
                if key not in step_01_output:
                    data_flow_issues.append(f"Step 01 → Step 02: {key} 누락")
            
            # Step 02 → Step 03 데이터 흐름 확인
            step_02_output = self.test_results.get('step_02', {})
            step_03_input_required = ['pose_keypoints']
            
            for key in step_03_input_required:
                if key not in step_02_output:
                    data_flow_issues.append(f"Step 02 → Step 03: {key} 누락")
            
            # Step 03 → Step 04 데이터 흐름 확인
            step_03_output = self.test_results.get('step_03', {})
            step_04_input_required = ['cloth_segmentation_mask']
            
            for key in step_04_input_required:
                if key not in step_03_output:
                    data_flow_issues.append(f"Step 03 → Step 04: {key} 누락")
            
            # Step 04 → Step 05 데이터 흐름 확인
            step_04_output = self.test_results.get('step_04', {})
            step_05_input_required = ['transformation_matrix']
            
            for key in step_05_input_required:
                if key not in step_04_output:
                    data_flow_issues.append(f"Step 04 → Step 05: {key} 누락")
            
            # Step 05 → Step 06 데이터 흐름 확인
            step_05_output = self.test_results.get('step_05', {})
            step_06_input_required = ['warped_clothing']
            
            for key in step_06_input_required:
                if key not in step_05_output:
                    data_flow_issues.append(f"Step 05 → Step 06: {key} 누락")
            
            # Step 06 → Step 07 데이터 흐름 확인
            step_06_output = self.test_results.get('step_06', {})
            step_07_input_required = ['fitted_image']
            
            for key in step_07_input_required:
                if key not in step_06_output:
                    data_flow_issues.append(f"Step 06 → Step 07: {key} 누락")
            
            # Step 07 → Step 08 데이터 흐름 확인
            step_07_output = self.test_results.get('step_07', {})
            step_08_input_required = ['processed_image']
            
            for key in step_08_input_required:
                if key not in step_07_output:
                    data_flow_issues.append(f"Step 07 → Step 08: {key} 누락")
            
            # Step 08 → Step 09 데이터 흐름 확인
            step_08_output = self.test_results.get('step_08', {})
            step_09_input_required = ['final_image']
            
            for key in step_09_input_required:
                if key not in step_08_output:
                    data_flow_issues.append(f"Step 08 → Step 09: {key} 누락")
            
            if data_flow_issues:
                logging.error(f"❌ 데이터 흐름 문제 발견: {len(data_flow_issues)}개")
                for issue in data_flow_issues:
                    logging.error(f"  - {issue}")
                return False
            else:
                logging.info("✅ 모든 Step 간 데이터 흐름이 올바름")
                return True
                
        except Exception as e:
            logging.error(f"❌ 데이터 흐름 테스트 중 오류: {e}")
            return False
    
    async def run_full_pipeline_test(self) -> Dict[str, Any]:
        """전체 파이프라인 테스트 실행"""
        try:
            logging.info("🚀 전체 AI 파이프라인 통합 테스트 시작")
            
            test_results = {
                'session_creation': False,
                'step_01': False,
                'step_02': False,
                'step_03': False,
                'step_04': False,
                'step_05': False,
                'step_06': False,
                'step_07': False,
                'step_08': False,
                'step_09': False,
                'progress_tracking': False,
                'data_flow': False
            }
            
            # 1. 테스트 세션 생성
            session_id = await self.create_test_session()
            if session_id:
                test_results['session_creation'] = True
                logging.info("✅ 테스트 세션 생성 성공")
            else:
                logging.error("❌ 테스트 세션 생성 실패")
                return test_results
            
            # 2. 각 Step 순차 실행
            test_results['step_01'] = await self.test_step_01(session_id)
            if not test_results['step_01']:
                logging.error("❌ Step 01 실패로 테스트 중단")
                return test_results
            
            test_results['step_02'] = await self.test_step_02(session_id)
            if not test_results['step_02']:
                logging.error("❌ Step 02 실패로 테스트 중단")
                return test_results
            
            test_results['step_03'] = await self.test_step_03(session_id)
            if not test_results['step_03']:
                logging.error("❌ Step 03 실패로 테스트 중단")
                return test_results
            
            test_results['step_04'] = await self.test_step_04(session_id)
            if not test_results['step_04']:
                logging.error("❌ Step 04 실패로 테스트 중단")
                return test_results
            
            test_results['step_05'] = await self.test_step_05(session_id)
            if not test_results['step_05']:
                logging.error("❌ Step 05 실패로 테스트 중단")
                return test_results
            
            test_results['step_06'] = await self.test_step_06(session_id)
            if not test_results['step_06']:
                logging.error("❌ Step 06 실패로 테스트 중단")
                return test_results
            
            test_results['step_07'] = await self.test_step_07(session_id)
            if not test_results['step_07']:
                logging.error("❌ Step 07 실패로 테스트 중단")
                return test_results
            
            test_results['step_08'] = await self.test_step_08(session_id)
            if not test_results['step_08']:
                logging.error("❌ Step 08 실패로 테스트 중단")
                return test_results
            
            test_results['step_09'] = await self.test_step_09(session_id)
            if not test_results['step_09']:
                logging.error("❌ Step 09 실패로 테스트 중단")
                return test_results
            
            # 3. 세션 진행률 추적 테스트
            test_results['progress_tracking'] = await self.test_session_progress_tracking()
            
            # 4. 데이터 흐름 테스트
            test_results['data_flow'] = await self.test_data_flow_between_steps()
            
            # 5. 최종 결과 요약
            success_count = sum(test_results.values())
            total_tests = len(test_results)
            
            logging.info(f"🎯 테스트 결과 요약: {success_count}/{total_tests} 성공")
            
            if success_count == total_tests:
                logging.info("🎉 모든 테스트가 성공적으로 완료되었습니다!")
            else:
                failed_tests = [key for key, value in test_results.items() if not value]
                logging.error(f"❌ 실패한 테스트: {failed_tests}")
            
            return test_results
            
        except Exception as e:
            logging.error(f"❌ 전체 파이프라인 테스트 중 오류: {e}")
            return test_results
    
    def cleanup(self):
        """테스트 리소스 정리"""
        try:
            logging.info("🧹 테스트 리소스 정리 시작")
            
            # 임시 파일들 정리
            for step_result in self.test_results.values():
                if isinstance(step_result, dict):
                    # 이미지 파일 경로가 있는 경우 정리
                    for key, value in step_result.items():
                        if 'path' in key and isinstance(value, str) and Path(value).exists():
                            try:
                                Path(value).unlink()
                                logging.info(f"✅ 임시 파일 삭제: {value}")
                            except Exception as e:
                                logging.warning(f"⚠️ 임시 파일 삭제 실패: {value} - {e}")
            
            logging.info("✅ 테스트 리소스 정리 완료")
            
        except Exception as e:
            logging.error(f"❌ 테스트 리소스 정리 실패: {e}")

async def main():
    """메인 테스트 실행 함수"""
    try:
        logging.info("🧪 MyCloset AI 전체 파이프라인 통합 테스트 시작")
        
        # 테스트 인스턴스 생성
        test_instance = FullPipelineIntegrationTest()
        
        # 전체 파이프라인 테스트 실행
        test_results = await test_instance.run_full_pipeline_test()
        
        # 테스트 결과 출력
        print("\n" + "="*60)
        print("🎯 전체 파이프라인 통합 테스트 결과")
        print("="*60)
        
        for test_name, result in test_results.items():
            status = "✅ 성공" if result else "❌ 실패"
            print(f"{test_name:20} : {status}")
        
        success_count = sum(test_results.values())
        total_tests = len(test_results)
        print(f"\n📊 최종 결과: {success_count}/{total_tests} 성공")
        
        if success_count == total_tests:
            print("🎉 모든 테스트가 성공적으로 완료되었습니다!")
        else:
            print("⚠️ 일부 테스트가 실패했습니다.")
        
        print("="*60)
        
        # 리소스 정리
        test_instance.cleanup()
        
    except Exception as e:
        logging.error(f"❌ 메인 테스트 실행 중 오류: {e}")
        print(f"❌ 테스트 실행 실패: {e}")

if __name__ == "__main__":
    # 테스트 실행
    asyncio.run(main())

# 모듈로 실행할 때를 위한 함수
def run_test():
    """모듈로 실행할 때 사용하는 함수"""
    return asyncio.run(main())
