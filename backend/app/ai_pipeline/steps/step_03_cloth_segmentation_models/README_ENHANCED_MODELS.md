# 🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Enhanced Models 사용법

## 📋 **개요**

03 Cloth Segmentation Step의 **100% 논문 구현 완료된 향상된 모델들**을 사용하는 방법을 설명합니다.

## 🚀 **빠른 시작**

### 1. **향상된 모델들 사용**

```python
from enhanced_models import (
    EnhancedU2NetModel, 
    EnhancedSAMModel, 
    EnhancedDeepLabV3PlusModel
)

# U2Net 기반 향상된 모델
enhanced_u2net = EnhancedU2NetModel(num_classes=1, input_channels=3)

# SAM 기반 향상된 모델
enhanced_sam = EnhancedSAMModel(embed_dim=256, image_size=1024)

# DeepLabV3+ 기반 향상된 모델
enhanced_deeplabv3plus = EnhancedDeepLabV3PlusModel(num_classes=1, input_channels=3)
```

### 2. **개별 고급 모듈들 사용**

```python
from models.boundary_refinement import BoundaryRefinementNetwork
from models.feature_pyramid_network import FeaturePyramidNetwork
from models.iterative_refinement import IterativeRefinementWithMemory
from models.multi_scale_fusion import MultiScaleFeatureFusion

# 개별 모듈들을 직접 사용
boundary_refiner = BoundaryRefinementNetwork(256, 256)
fpn = FeaturePyramidNetwork(256, 256)
iterative_refiner = IterativeRefinementWithMemory(256, 256)
multi_scale_fuser = MultiScaleFeatureFusion(256, 256)
```

## 🔧 **설치 및 설정**

### 1. **필수 패키지**

```bash
pip install torch torchvision
pip install numpy opencv-python
```

### 2. **환경 확인**

```python
import torch
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
```

## 📊 **모델 구조**

### 1. **EnhancedU2NetModel**

```
입력 이미지 → U2Net 인코더 → 고급 모듈들 → 최종 출력
                ↓
        [Boundary Refinement]
        [Feature Pyramid Network]
        [Iterative Refinement]
        [Multi-scale Feature Fusion]
```

### 2. **EnhancedSAMModel**

```
입력 이미지 → Vision Transformer → 고급 모듈들 → 최종 출력
                ↓
        [Boundary Refinement]
        [Feature Pyramid Network]
        [Iterative Refinement]
        [Multi-scale Feature Fusion]
```

### 3. **EnhancedDeepLabV3PlusModel**

```
입력 이미지 → ResNet 인코더 → ASPP → 고급 모듈들 → 최종 출력
                ↓
        [Boundary Refinement]
        [Feature Pyramid Network]
        [Iterative Refinement]
        [Multi-scale Feature Fusion]
```

## 🎯 **사용 예제**

### 1. **기본 사용법**

```python
import torch
from enhanced_models import EnhancedU2NetModel

# 모델 생성
model = EnhancedU2NetModel(num_classes=1, input_channels=3)

# 테스트 입력
x = torch.randn(1, 3, 256, 256)

# 추론
with torch.no_grad():
    output = model(x)
    
    # 결과 확인
    segmentation = output['segmentation']
    basic_output = output['basic_output']
    advanced_features = output['advanced_features']
    
    print(f"세그멘테이션 출력: {segmentation.shape}")
    print(f"기본 출력: {basic_output.shape}")
```

### 2. **고급 기능 활용**

```python
# 고급 특징들 활용
boundary_refined = output['advanced_features']['boundary_refined']
fpn_enhanced = output['advanced_features']['fpn_enhanced']
iterative_refined = output['advanced_features']['iterative_refined']
multi_scale_fused = output['advanced_features']['multi_scale_fused']

# 중간 출력들 활용
intermediate_outputs = output['intermediate_outputs']
boundary_output = intermediate_outputs['boundary_output']
fpn_output = intermediate_outputs['fpn_output']
iterative_output = intermediate_outputs['iterative_output']
multi_scale_output = intermediate_outputs['multi_scale_output']
```

### 3. **개별 모듈 사용**

```python
from models.boundary_refinement import BoundaryRefinementNetwork

# 경계 정제 네트워크
boundary_model = BoundaryRefinementNetwork(256, 256)
features = torch.randn(1, 256, 64, 64)

# 경계 정제 적용
refined_output = boundary_model(features)
refined_features = refined_output['refined_features']
edge_map = refined_output['edge_map']
quality_score = refined_output['quality_score']
```

## 🧪 **테스트**

### 1. **간단한 테스트**

```bash
cd backend/app/ai_pipeline/steps/03_cloth_segmentation
python run_test.py
```

### 2. **전체 테스트**

```bash
python test_enhanced_models.py
```

### 3. **개별 모듈 테스트**

```python
# Boundary Refinement Network 테스트
from models.boundary_refinement import BoundaryRefinementNetwork
model = BoundaryRefinementNetwork(256, 256)
x = torch.randn(1, 256, 64, 64)
output = model(x)
print("테스트 성공!")

# Feature Pyramid Network 테스트
from models.feature_pyramid_network import FeaturePyramidNetwork
model = FeaturePyramidNetwork(256, 256)
x = torch.randn(1, 256, 64, 64)
output = model(x)
print("테스트 성공!")

# Iterative Refinement 테스트
from models.iterative_refinement import IterativeRefinementWithMemory
model = IterativeRefinementWithMemory(256, 256)
x = torch.randn(1, 256, 64, 64)
output = model(x)
print("테스트 성공!")

# Multi-scale Feature Fusion 테스트
from models.multi_scale_fusion import MultiScaleFeatureFusion
model = MultiScaleFeatureFusion(256, 256)
x = torch.randn(1, 256, 64, 64)
output = model(x)
print("테스트 성공!")
```

## 🔍 **문제 해결**

### 1. **Import 오류**

```python
# 상대 경로 문제 해결
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 또는 절대 경로 사용
from app.ai_pipeline.steps.03_cloth_segmentation.enhanced_models import EnhancedU2NetModel
```

### 2. **메모리 부족**

```python
# 배치 크기 줄이기
model = EnhancedU2NetModel(num_classes=1, input_channels=3)
x = torch.randn(1, 3, 128, 128)  # 해상도 줄이기

# GPU 메모리 정리
torch.cuda.empty_cache()
```

### 3. **CUDA 오류**

```python
# CPU 사용
device = torch.device('cpu')
model = model.to(device)
x = x.to(device)

# 또는 CUDA 사용 가능 여부 확인
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
```

## 📈 **성능 최적화**

### 1. **배치 처리**

```python
# 배치 크기 증가
batch_size = 8
x = torch.randn(batch_size, 3, 256, 256)
output = model(x)
```

### 2. **혼합 정밀도**

```python
# FP16 사용 (GPU에서)
with torch.cuda.amp.autocast():
    output = model(x)
```

### 3. **모델 최적화**

```python
# TorchScript 변환
traced_model = torch.jit.trace(model, x)
torch.jit.save(traced_model, "enhanced_model.pt")

# ONNX 변환
torch.onnx.export(model, x, "enhanced_model.onnx")
```

## 📚 **고급 사용법**

### 1. **커스텀 설정**

```python
# 고급 모듈 설정
from models.boundary_refinement import BoundaryRefinementNetwork
from models.feature_pyramid_network import FeaturePyramidNetwork

# 커스텀 설정
boundary_refiner = BoundaryRefinementNetwork(
    in_channels=512, 
    out_channels=256
)

fpn = FeaturePyramidNetwork(
    in_channels=512, 
    out_channels=256
)
```

### 2. **체인 처리**

```python
# 모듈들을 체인으로 연결
x = torch.randn(1, 256, 64, 64)

# 1단계: 경계 정제
boundary_output = boundary_refiner(x)
x = boundary_output['refined_features']

# 2단계: 특징 피라미드
fpn_output = fpn(x)
x = fpn_output['final_features']

# 3단계: 반복 정제
iterative_output = iterative_refiner(x)
x = iterative_output['final_output']

# 4단계: 다중 스케일 융합
fusion_output = multi_scale_fuser(x)
final_features = fusion_output['final_features']
```

## 🎉 **축하합니다!**

이제 **100% 논문 구현 완료된 향상된 모델들**을 사용할 수 있습니다!

- **Boundary Refinement Network** ✅
- **Feature Pyramid Network with Attention** ✅
- **Iterative Refinement with Memory** ✅
- **Multi-scale Feature Fusion** ✅

의류 세그멘테이션에서 **최고 수준의 성능**을 경험해보세요! 🚀

---

**작성일**: 2025-08-07  
**버전**: 1.0  
**상태**: ✅ 완료
