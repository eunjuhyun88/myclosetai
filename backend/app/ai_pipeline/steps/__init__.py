"""
AI Pipeline Steps - 순환 참조 방지 및 안전한 import
"""

__all__ = [
    'HumanParsingStep',
    'PoseEstimationStep', 
    'ClothSegmentationStep',
    'GeometricMatchingStep',
    'ClothWarpingStep',
    'VirtualFittingStep',
    'PostProcessingStep',
    'QualityAssessmentStep'
]

# 안전한 import - 실패 시 None 반환
try:
    from .step_01_human_parsing import HumanParsingStep
except ImportError as e:
    print(f"⚠️ HumanParsingStep import 실패: {e}")
    HumanParsingStep = None

try:
    from .step_02_pose_estimation import PoseEstimationStep
except ImportError as e:
    print(f"⚠️ PoseEstimationStep import 실패: {e}")
    PoseEstimationStep = None

try:
    from .step_03_cloth_segmentation import ClothSegmentationStep
except ImportError as e:
    print(f"⚠️ ClothSegmentationStep import 실패: {e}")
    ClothSegmentationStep = None

try:
    from .step_04_geometric_matching import GeometricMatchingStep
except ImportError as e:
    print(f"⚠️ GeometricMatchingStep import 실패: {e}")
    GeometricMatchingStep = None

try:
    from .step_05_cloth_warping import ClothWarpingStep
except ImportError as e:
    print(f"⚠️ ClothWarpingStep import 실패: {e}")
    ClothWarpingStep = None

try:
    from .step_06_virtual_fitting import VirtualFittingStep
except ImportError as e:
    print(f"⚠️ VirtualFittingStep import 실패: {e}")
    VirtualFittingStep = None

try:
    from .step_07_post_processing import PostProcessingStep
except ImportError as e:
    print(f"⚠️ PostProcessingStep import 실패: {e}")
    PostProcessingStep = None

try:
    from .step_08_quality_assessment import QualityAssessmentStep
except ImportError as e:
    print(f"⚠️ QualityAssessmentStep import 실패: {e}")
    QualityAssessmentStep = None

# import 성공 여부 체크
def check_imports():
    """모든 클래스가 성공적으로 import되었는지 확인"""
    classes = [
        HumanParsingStep, PoseEstimationStep, ClothSegmentationStep,
        GeometricMatchingStep, ClothWarpingStep, VirtualFittingStep,
        PostProcessingStep, QualityAssessmentStep
    ]
    
    success_count = sum(1 for cls in classes if cls is not None)
    total_count = len(classes)
    
    return {
        'success_count': success_count,
        'total_count': total_count,
        'success_rate': success_count / total_count,
        'all_imported': success_count == total_count
    }

# 지연 로딩 함수 (필요시 사용)
def get_available_steps():
    """사용 가능한 step 클래스들만 반환"""
    available = {}
    
    if HumanParsingStep is not None:
        available['HumanParsingStep'] = HumanParsingStep
    if PoseEstimationStep is not None:
        available['PoseEstimationStep'] = PoseEstimationStep
    if ClothSegmentationStep is not None:
        available['ClothSegmentationStep'] = ClothSegmentationStep
    if GeometricMatchingStep is not None:
        available['GeometricMatchingStep'] = GeometricMatchingStep
    if ClothWarpingStep is not None:
        available['ClothWarpingStep'] = ClothWarpingStep
    if VirtualFittingStep is not None:
        available['VirtualFittingStep'] = VirtualFittingStep
    if PostProcessingStep is not None:
        available['PostProcessingStep'] = PostProcessingStep
    if QualityAssessmentStep is not None:
        available['QualityAssessmentStep'] = QualityAssessmentStep
    
    return available

# 초기화 시 상태 출력
import_status = check_imports()
print(f"📊 AI Pipeline Steps Import 상태: {import_status['success_count']}/{import_status['total_count']} "
      f"({'✅' if import_status['all_imported'] else '⚠️'})")

if not import_status['all_imported']:
    available_steps = get_available_steps()
    print(f"✅ 사용 가능한 Step 클래스들: {list(available_steps.keys())}")