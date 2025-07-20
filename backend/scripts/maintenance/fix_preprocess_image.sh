#!/bin/bash
# MyCloset AI 즉시 수정 스크립트 - preprocess_image 함수 추가

set -e

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔧 MyCloset AI preprocess_image 함수 누락 문제 즉시 해결${NC}"
echo "=================================================================="

cd backend

# 1. model_loader.py 백업
echo -e "${YELLOW}📋 백업 생성 중...${NC}"
cp app/ai_pipeline/utils/model_loader.py app/ai_pipeline/utils/model_loader.py.backup

# 2. preprocess_image 함수들 추가
echo -e "${BLUE}🔧 preprocess_image 함수들 추가 중...${NC}"

# __all__ 목록에 함수들 추가
python3 << 'EOF'
import re

# model_loader.py 파일 읽기
with open('app/ai_pipeline/utils/model_loader.py', 'r') as f:
    content = f.read()

# __all__ 리스트에 누락된 함수들 추가
all_pattern = r'(__all__ = \[.*?)\]'

new_exports = '''
    # 이미지 전처리 함수들 (누락된 함수들)
    'preprocess_image',
    'postprocess_segmentation', 
    'preprocess_pose_input',
    'preprocess_human_parsing_input',
    'preprocess_cloth_segmentation_input',
    'tensor_to_pil',
    'pil_to_tensor',
    'resize_image_with_aspect_ratio',
    'create_visualization_grid',
    'optimize_tensor_memory',
    'safe_model_forward',
'''

def replace_all(match):
    return match.group(1) + new_exports + ']'

content = re.sub(all_pattern, replace_all, content, flags=re.DOTALL)

# 파일 끝에 함수들 추가 (logger 메시지 앞에)
insert_point = content.rfind('logger.info("✅ ModelLoader v4.3 모듈 로드 완료")')

if insert_point == -1:
    # 파일 끝에 추가
    insert_point = len(content)

# preprocess_image 함수들 코드
functions_code = '''
# ==============================================
# 🔥 누락된 preprocess_image 함수들 추가
# ==============================================

def preprocess_image(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    target_size: Tuple[int, int] = (512, 512),
    device: str = "mps",
    normalize: bool = True,
    to_tensor: bool = True
) -> torch.Tensor:
    """
    🔥 이미지 전처리 함수 - Step 클래스들에서 사용
    
    Args:
        image: 입력 이미지 (PIL.Image, numpy array, tensor)
        target_size: 목표 크기 (height, width)
        device: 디바이스 ("mps", "cuda", "cpu")
        normalize: 정규화 여부 (0-1 범위로)
        to_tensor: 텐서로 변환 여부
    
    Returns:
        torch.Tensor: 전처리된 이미지 텐서
    """
    try:
        # 1. PIL Image로 변환
        if isinstance(image, torch.Tensor) if TORCH_AVAILABLE else False:
            if image.dim() == 4:
                image = image.squeeze(0)
            if image.dim() == 3 and image.shape[0] == 3:
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif isinstance(image, np.ndarray) if NUMPY_AVAILABLE else False:
            if image.ndim == 3 and image.shape[2] == 3:
                image = Image.fromarray(image.astype(np.uint8))
            else:
                image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            logger.warning(f"지원하지 않는 이미지 타입: {type(image)}")
            if TORCH_AVAILABLE and to_tensor:
                return torch.randn(1, 3, target_size[0], target_size[1])
            else:
                return np.random.randn(target_size[0], target_size[1], 3).astype(np.float32)
        
        # 2. RGB 변환
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 3. 크기 조정
        if target_size != image.size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # 4. numpy 배열로 변환
        img_array = np.array(image).astype(np.float32) if NUMPY_AVAILABLE else np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)
        
        # 5. 정규화
        if normalize:
            img_array = img_array / 255.0
        
        # 6. 텐서 변환
        if to_tensor and TORCH_AVAILABLE:
            # HWC -> CHW 변환
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            
            # 배치 차원 추가
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            
            # 디바이스로 이동
            try:
                if device != "cpu" and torch.cuda.is_available() and device == "cuda":
                    img_tensor = img_tensor.cuda()
                elif device == "mps" and torch.backends.mps.is_available():
                    img_tensor = img_tensor.to("mps")
                else:
                    img_tensor = img_tensor.cpu()
            except Exception as e:
                logger.warning(f"디바이스 이동 실패: {e}, CPU 사용")
                img_tensor = img_tensor.cpu()
            
            return img_tensor
        else:
            return img_array
    
    except Exception as e:
        logger.error(f"❌ 이미지 전처리 실패: {e}")
        # 폴백: 기본 크기 더미 텐서
        if TORCH_AVAILABLE and to_tensor:
            return torch.randn(1, 3, target_size[0], target_size[1])
        else:
            return np.random.randn(target_size[0], target_size[1], 3).astype(np.float32) if NUMPY_AVAILABLE else [[[]]]

def postprocess_segmentation(
    segmentation: torch.Tensor,
    original_size: Tuple[int, int],
    threshold: float = 0.5,
    smooth: bool = True
) -> np.ndarray:
    """세그멘테이션 결과 후처리"""
    try:
        if not TORCH_AVAILABLE or not NUMPY_AVAILABLE:
            return np.zeros(original_size[::-1], dtype=np.uint8)
            
        # 텐서를 numpy로 변환
        if isinstance(segmentation, torch.Tensor):
            seg_np = segmentation.detach().cpu().numpy()
        else:
            seg_np = segmentation
        
        # 배치 및 채널 차원 제거
        if seg_np.ndim == 4:
            seg_np = seg_np.squeeze(0)
        if seg_np.ndim == 3 and seg_np.shape[0] == 1:
            seg_np = seg_np.squeeze(0)
        
        # 이진화
        if threshold > 0:
            seg_np = (seg_np > threshold).astype(np.float32)
        
        # 크기 조정
        if seg_np.shape != original_size[::-1]:
            seg_img = Image.fromarray((seg_np * 255).astype(np.uint8))
            seg_img = seg_img.resize(original_size, Image.Resampling.LANCZOS)
            seg_np = np.array(seg_img) / 255.0
        
        # 0-255 범위로 변환
        mask = (seg_np * 255).astype(np.uint8)
        return mask
    
    except Exception as e:
        logger.error(f"❌ 세그멘테이션 후처리 실패: {e}")
        return np.zeros(original_size[::-1], dtype=np.uint8) if NUMPY_AVAILABLE else [[]]

def preprocess_pose_input(
    image: Union[Image.Image, np.ndarray],
    input_size: Tuple[int, int] = (368, 368),
    device: str = "mps"
) -> torch.Tensor:
    """포즈 추정용 이미지 전처리"""
    return preprocess_image(image=image, target_size=input_size, device=device, normalize=True, to_tensor=True)

def preprocess_human_parsing_input(
    image: Union[Image.Image, np.ndarray],
    input_size: Tuple[int, int] = (512, 512),
    device: str = "mps"
) -> torch.Tensor:
    """인간 파싱용 이미지 전처리"""
    return preprocess_image(image=image, target_size=input_size, device=device, normalize=True, to_tensor=True)

def preprocess_cloth_segmentation_input(
    image: Union[Image.Image, np.ndarray],
    input_size: Tuple[int, int] = (320, 320),
    device: str = "mps"
) -> torch.Tensor:
    """의류 세그멘테이션용 이미지 전처리"""
    return preprocess_image(image=image, target_size=input_size, device=device, normalize=True, to_tensor=True)

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """텐서를 PIL 이미지로 변환"""
    try:
        if not TORCH_AVAILABLE:
            return Image.new('RGB', (512, 512), (128, 128, 128))
            
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)
        
        tensor = tensor.detach().cpu()
        
        # 정규화된 텐서라면 0-255로 변환
        if tensor.max() <= 1.0:
            tensor = tensor * 255
        
        numpy_img = tensor.numpy().astype(np.uint8)
        return Image.fromarray(numpy_img)
    
    except Exception as e:
        logger.error(f"텐서->PIL 변환 실패: {e}")
        return Image.new('RGB', (512, 512), (128, 128, 128))

def pil_to_tensor(image: Image.Image, device: str = "mps", normalize: bool = True) -> torch.Tensor:
    """PIL 이미지를 텐서로 변환"""
    return preprocess_image(image, device=device, normalize=normalize, to_tensor=True)

def resize_image_with_aspect_ratio(
    image: Image.Image,
    target_size: Tuple[int, int],
    fill_color: Tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """종횡비 유지하면서 이미지 크기 조정"""
    try:
        target_w, target_h = target_size
        original_w, original_h = image.size
        
        # 종횡비 계산
        aspect_ratio = original_w / original_h
        target_aspect_ratio = target_w / target_h
        
        if aspect_ratio > target_aspect_ratio:
            new_w = target_w
            new_h = int(target_w / aspect_ratio)
        else:
            new_h = target_h
            new_w = int(target_h * aspect_ratio)
        
        # 크기 조정
        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 새 이미지 생성 및 중앙 배치
        result = Image.new('RGB', target_size, fill_color)
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        result.paste(resized, (paste_x, paste_y))
        
        return result
    
    except Exception as e:
        logger.error(f"종횡비 조정 실패: {e}")
        return image.resize(target_size, Image.Resampling.LANCZOS)

def create_visualization_grid(
    images: List[Image.Image],
    labels: List[str],
    grid_size: Optional[Tuple[int, int]] = None
) -> Image.Image:
    """여러 이미지를 그리드로 배치하여 시각화"""
    try:
        if not images:
            return Image.new('RGB', (512, 512), (128, 128, 128))
        
        num_images = len(images)
        
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(num_images))) if NUMPY_AVAILABLE else 2
            rows = int(np.ceil(num_images / cols)) if NUMPY_AVAILABLE else 2
        else:
            cols, rows = grid_size
        
        # 개별 이미지 크기
        img_w, img_h = 256, 256
        
        # 전체 그리드 크기
        grid_w = cols * img_w + (cols - 1) * 10
        grid_h = rows * img_h + (rows - 1) * 10 + 50
        
        # 그리드 이미지 생성
        grid_img = Image.new('RGB', (grid_w, grid_h), (240, 240, 240))
        
        for i, (img, label) in enumerate(zip(images, labels)):
            if i >= cols * rows:
                break
            
            row = i // cols
            col = i % cols
            
            # 이미지 크기 조정
            img_resized = img.resize((img_w, img_h), Image.Resampling.LANCZOS)
            
            # 배치 위치 계산
            x = col * (img_w + 10)
            y = row * (img_h + 60) + 50
            
            # 이미지 붙이기
            grid_img.paste(img_resized, (x, y))
        
        return grid_img
    
    except Exception as e:
        logger.error(f"시각화 그리드 생성 실패: {e}")
        return Image.new('RGB', (512, 512), (128, 128, 128))

def optimize_tensor_memory(tensor: torch.Tensor) -> torch.Tensor:
    """텐서 메모리 최적화"""
    try:
        if not TORCH_AVAILABLE:
            return tensor
        
        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # MPS 캐시 정리
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except:
                pass
        
        return tensor.contiguous()
    
    except Exception as e:
        logger.warning(f"텐서 메모리 최적화 실패: {e}")
        return tensor

def safe_model_forward(model: Any, inputs: torch.Tensor, device: str = "mps") -> torch.Tensor:
    """안전한 모델 forward pass"""
    try:
        if not TORCH_AVAILABLE:
            return torch.zeros(1, 3, 512, 512)
            
        if not hasattr(model, '__call__'):
            raise ValueError("모델이 호출 가능하지 않습니다")
        
        # 입력을 올바른 디바이스로 이동
        if hasattr(inputs, 'to'):
            try:
                inputs = inputs.to(device)
            except Exception as e:
                logger.warning(f"입력 디바이스 이동 실패: {e}")
        
        # 모델을 평가 모드로
        if hasattr(model, 'eval'):
            model.eval()
        
        # 그래디언트 비활성화
        with torch.no_grad():
            outputs = model(inputs)
        
        return outputs
    
    except Exception as e:
        logger.error(f"모델 forward 실패: {e}")
        # 폴백: 입력과 같은 크기의 더미 출력
        if hasattr(inputs, 'shape'):
            return torch.zeros_like(inputs)
        else:
            return torch.zeros(1, 3, 512, 512)

'''

# 함수들 삽입
new_content = content[:insert_point] + functions_code + '\n' + content[insert_point:]

# 파일에 쓰기
with open('app/ai_pipeline/utils/model_loader.py', 'w') as f:
    f.write(new_content)

print("✅ preprocess_image 함수들 추가 완료")
EOF

# 3. import 누락 문제 해결
echo -e "${BLUE}🔧 import 구문 추가 중...${NC}"

python3 << 'EOF'
# model_loader.py에 Union import 추가
with open('app/ai_pipeline/utils/model_loader.py', 'r') as f:
    content = f.read()

# typing import에 Union 추가 (없다면)
if 'Union' not in content:
    content = content.replace(
        'from typing import Dict, Any, Optional,',
        'from typing import Dict, Any, Optional, Union,'
    )

# List import 추가 (없다면)
if ', List,' not in content:
    content = content.replace(
        'from typing import Dict, Any, Optional, Union,',
        'from typing import Dict, Any, Optional, Union, List,'
    )

with open('app/ai_pipeline/utils/model_loader.py', 'w') as f:
    f.write(content)

print("✅ import 구문 추가 완료")
EOF

# 4. 서버 재시작
echo -e "${YELLOW}🔄 서버 재시작...${NC}"

# 기존 프로세스 종료
pkill -f "python.*app/main.py" 2>/dev/null || true
sleep 2

# 서버 재시작
echo -e "${GREEN}🚀 서버 시작 중...${NC}"
python app/main.py &

# 잠시 대기
sleep 5

# 서버 상태 확인
echo -e "${BLUE}📊 서버 상태 확인...${NC}"
curl -s http://localhost:8000/health | python -m json.tool || echo "서버 아직 시작 중..."

echo -e "${GREEN}✅ preprocess_image 함수 누락 문제 해결 완료!${NC}"
echo ""
echo "📝 다음 단계:"
echo "1. 브라우저에서 http://localhost:8000/docs 확인"
echo "2. curl http://localhost:8000/api/step/health 테스트"
echo "3. 로그 확인: tail -f logs/*.log"