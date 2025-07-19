#!/bin/bash
# 🚨 ModelLoader callable 오류 즉시 해결 스크립트

set -e
cd backend

echo "🔧 1. ModelLoader callable 오류 수정 중..."

# StepModelInterface 클래스의 load_model_async 호출 문제 수정
python3 << 'EOF'
import re

# model_loader.py 수정
with open('app/ai_pipeline/utils/model_loader.py', 'r') as f:
    content = f.read()

# load_model_async 메서드 파라미터 수정
content = re.sub(
    r'async def load_model_async\(self, model_name: str, \*\*kwargs\)',
    'async def load_model_async(self, model_name: str)',
    content
)

# _load_model_sync_wrapper 호출 방식 수정
content = re.sub(
    r'return sync_wrapper_func\(model_name, \{\}\)',
    'return sync_wrapper_func(model_name)',
    content
)

# _load_model_sync_wrapper 메서드 시그니처 수정
content = re.sub(
    r'def _load_model_sync_wrapper\(self, model_name: str, kwargs: Dict\)',
    'def _load_model_sync_wrapper(self, model_name: str)',
    content
)

with open('app/ai_pipeline/utils/model_loader.py', 'w') as f:
    f.write(content)

print("✅ model_loader.py 수정 완료")
EOF

echo "🔧 2. Step 클래스들의 디바이스 설정 오류 수정 중..."

# BaseStepMixin의 _auto_detect_device 메서드 호출 문제 수정
python3 << 'EOF'
import os
import re

step_files = [
    'app/ai_pipeline/steps/step_02_pose_estimation.py',
    'app/ai_pipeline/steps/step_03_cloth_segmentation.py'
]

for file_path in step_files:
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            content = f.read()
        
        # _auto_detect_device 호출 시 파라미터 추가
        content = re.sub(
            r'self\._auto_detect_device\(\)',
            'self._auto_detect_device(device="auto")',
            content
        )
        
        # 또는 파라미터 없이 호출하도록 수정
        content = re.sub(
            r'PoseEstimationStep\._auto_detect_device\(\) missing 1 required positional argument',
            '',
            content
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"✅ {file_path} 수정 완료")
EOF

echo "🔧 3. 'dict' object is not callable 오류 수정 중..."

# 워밍업 함수 호출 문제 수정
python3 << 'EOF'
import os
import re

base_step_file = 'app/ai_pipeline/steps/base_step_mixin.py'
if os.path.exists(base_step_file):
    with open(base_step_file, 'r') as f:
        content = f.read()
    
    # warmup_functions 딕셔너리 호출 문제 해결
    warmup_fix = '''
    def _setup_warmup_functions(self):
        """워밍업 함수들 안전하게 설정"""
        try:
            # 실제 함수 객체로 설정
            self.warmup_functions = {
                'model_warmup': self._safe_model_warmup,
                'device_warmup': self._safe_device_warmup,
                'memory_warmup': self._safe_memory_warmup,
                'pipeline_warmup': self._safe_pipeline_warmup
            }
            
            if hasattr(self, 'logger'):
                self.logger.debug("워밍업 함수들 설정 완료")
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"워밍업 함수 설정 실패: {e}")
            self.warmup_functions = {}
    '''
    
    # 기존 _setup_warmup_functions 메서드 교체
    content = re.sub(
        r'def _setup_warmup_functions\(self\):.*?(?=\n    def|\nclass|\n# )',
        warmup_fix,
        content,
        flags=re.DOTALL
    )
    
    with open(base_step_file, 'w') as f:
        f.write(content)
    
    print("✅ base_step_mixin.py 워밍업 함수 수정 완료")
EOF

echo "🔧 4. Step 클래스 logger 속성 추가..."

# 모든 Step 클래스에 logger 속성 추가
find app/ai_pipeline/steps -name "step_*.py" -exec python3 -c "
import sys
import re

file_path = sys.argv[1]
with open(file_path, 'r') as f:
    content = f.read()

# logger 속성이 없는 경우 추가
if 'self.logger =' not in content:
    # __init__ 메서드 찾아서 logger 추가
    content = re.sub(
        r'(def __init__\(self[^)]*\):.*?)(super\(\).__init__\(\)|pass)',
        r'\1\2\n        if not hasattr(self, \"logger\"):\n            self.logger = logging.getLogger(f\"pipeline.{self.__class__.__name__}\")',
        content,
        flags=re.DOTALL
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    print(f'✅ {file_path}에 logger 추가')
" {} \;

echo "✅ 모든 Step 클래스에 logger 속성 추가 완료"

echo "🧹 5. 캐시 정리..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

echo ""
echo "🎉 callable 오류 수정 완료!"
echo "🚀 이제 서버를 다시 시작해보세요:"
echo "   python app/main.py"