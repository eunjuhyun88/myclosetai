
# 패치 모듈 import
from app.utils.opencv_data_patch import patch_cv2_data, test_opencv_data
import cv2

# 패치 강제 적용
patch_cv2_data()

# 테스트 실행
success = test_opencv_data()
print(f"패치 성공: {success}")

# 실제 사용 테스트
try:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    print(f"✅ 최종 테스트 성공: {cascade_path}")
except Exception as e:
    print(f"❌ 최종 테스트 실패: {e}")
