"""
MyCloset AI - AI 모델 매니저
M3 Max GPU에 최적화된 AI 모델 로딩 및 관리 서비스
"""

import logging
import torch
from typing import Dict, Optional, Any
from pathlib import Path
import psutil
import gc

logger = logging.getLogger("mycloset.services.model_manager")

class ModelManager:
    """M3 Max 최적화 AI 모델 매니저"""
    
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.loaded_models: Dict[str, Any] = {}
        self.model_configs: Dict[str, Dict] = {}
        self.memory_threshold = 0.85  # 85% 메모리 사용시 경고
        self.available_models: Dict[str, bool] = {}  # 캐시된 사용 가능 모델 목록
        
        logger.info(f"🤖 ModelManager 초기화: device={self.device}")
        self._setup_device_optimization()
    
    def _setup_device_optimization(self):
        """M3 Max 디바이스 최적화 설정"""
        if self.device == "mps":
            # Metal Performance Shaders 최적화 (버전 호환성 체크)
            if hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
                logger.info("✅ MPS 캐시 정리 완료")
            else:
                logger.info("ℹ️ MPS empty_cache 미지원 (PyTorch 2.5.1)")
            
            # 메모리 최적화 설정
            if hasattr(torch.backends.mps, 'set_per_process_memory_fraction'):
                torch.backends.mps.set_per_process_memory_fraction(0.8)
                logger.info("✅ MPS 메모리 할당 제한: 80%")
            else:
                logger.info("ℹ️ MPS 메모리 제한 설정 미지원")
    
    def get_available_models(self) -> Dict[str, bool]:
        """사용 가능한 모델 목록 조회"""
        try:
            from app.core.model_paths import DETECTED_MODELS, is_model_available
            
            available_models = {}
            for model_key in DETECTED_MODELS.keys():
                available_models[model_key] = is_model_available(model_key)
            
            logger.info(f"📋 사용 가능한 모델: {sum(available_models.values())}/{len(available_models)}")
            return available_models
            
        except ImportError:
            logger.warning("⚠️ 모델 경로 정보를 불러올 수 없습니다")
            return {}
        except Exception as e:
            logger.error(f"❌ 모델 목록 조회 오류: {e}")
            return {}
    
    async def load_models(self) -> bool:
        """모든 사용 가능한 모델 로드 (비동기)"""
        logger.info("🔄 모든 모델 로드 시작...")
        
        try:
            available_models = self.get_available_models()
            
            if not available_models:
                logger.warning("⚠️ 사용 가능한 모델이 없습니다")
                return False
            
            # 우선순위 높은 모델들만 자동 로드
            priority_models = ["ootdiffusion", "sam", "stable_diffusion"]
            loaded_count = 0
            
            for model_key in priority_models:
                if available_models.get(model_key, False):
                    if self.load_model(model_key):
                        loaded_count += 1
                        logger.info(f"✅ 자동 로드 완료: {model_key}")
                    else:
                        logger.warning(f"⚠️ 자동 로드 실패: {model_key}")
            
            logger.info(f"✅ 모델 자동 로드 완료: {loaded_count}개")
            return loaded_count > 0
            
        except Exception as e:
            logger.error(f"❌ 모델 로드 오류: {e}")
            return False
    
    async def unload_models(self) -> bool:
        """모든 로드된 모델 언로드 (비동기)"""
        logger.info("🔄 모든 모델 언로드 시작...")
        
        try:
            model_keys = list(self.loaded_models.keys())
            unloaded_count = 0
            
            for model_key in model_keys:
                if self.unload_model(model_key):
                    unloaded_count += 1
            
            logger.info(f"✅ 모든 모델 언로드 완료: {unloaded_count}개")
            return True
            
        except Exception as e:
            logger.error(f"❌ 모델 언로드 오류: {e}")
            return False
    
    def load_model(self, model_key: str, **kwargs) -> bool:
        """모델 로드 (M3 Max 최적화)"""
        logger.info(f"🔄 모델 로드 시작: {model_key}")
        
        try:
            # 메모리 체크
            memory_info = self._check_memory()
            if memory_info["usage_percent"] > self.memory_threshold * 100:
                logger.warning(f"⚠️ 메모리 사용량 높음: {memory_info['usage_percent']:.1f}%")
                self._cleanup_memory()
            
            # 이미 로드된 모델 확인
            if model_key in self.loaded_models:
                logger.info(f"✅ 모델 이미 로드됨: {model_key}")
                return True
            
            # 모델 타입별 로딩 전략
            model_info = self._get_model_info(model_key)
            if not model_info:
                logger.error(f"❌ 모델 정보를 찾을 수 없습니다: {model_key}")
                return False
            
            # 실제 모델 로딩 (현재는 시뮬레이션)
            success = self._load_model_by_type(model_key, model_info, **kwargs)
            
            if success:
                logger.info(f"✅ 모델 로드 완료: {model_key}")
                return True
            else:
                logger.error(f"❌ 모델 로드 실패: {model_key}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 모델 로드 오류 {model_key}: {e}")
            return False
    
    def unload_model(self, model_key: str) -> bool:
        """모델 언로드"""
        logger.info(f"🔄 모델 언로드: {model_key}")
        
        try:
            if model_key in self.loaded_models:
                # 모델 메모리 해제
                del self.loaded_models[model_key]
                
                # GPU 메모리 정리
                if self.device == "mps":
                    if hasattr(torch.backends.mps, 'empty_cache'):
                        torch.backends.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
                
                # Python 가비지 컬렉션
                gc.collect()
                
                logger.info(f"✅ 모델 언로드 완료: {model_key}")
                return True
            else:
                logger.warning(f"⚠️ 로드되지 않은 모델: {model_key}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 모델 언로드 오류 {model_key}: {e}")
            return False
    
    def get_model_status(self) -> Dict[str, Any]:
        """현재 모델 상태 조회"""
        memory_info = self._check_memory()
        
        status = {
            "loaded_models": list(self.loaded_models.keys()),
            "loaded_count": len(self.loaded_models),
            "available_models": self.get_available_models(),
            "memory_info": memory_info,
            "device": self.device,
            "device_available": torch.backends.mps.is_available() if self.device == "mps" else True
        }
        
        return status
    
    def _load_model_by_type(self, model_key: str, model_info: Dict, **kwargs) -> bool:
        """모델 타입별 로딩 로직"""
        model_type = model_info.get("type", "unknown")
        
        # 현재는 시뮬레이션 (실제 모델 로딩 로직은 추후 구현)
        logger.info(f"🎯 모델 타입: {model_type}")
        
        if model_type == "virtual_tryon":
            return self._load_virtual_tryon_model(model_key, model_info, **kwargs)
        elif model_type == "segmentation":
            return self._load_segmentation_model(model_key, model_info, **kwargs)
        elif model_type == "base_diffusion":
            return self._load_diffusion_model(model_key, model_info, **kwargs)
        else:
            return self._load_generic_model(model_key, model_info, **kwargs)
    
    def _load_virtual_tryon_model(self, model_key: str, model_info: Dict, **kwargs) -> bool:
        """가상 피팅 모델 로딩 (시뮬레이션)"""
        logger.info(f"👗 가상 피팅 모델 로딩: {model_key}")
        
        # 시뮬레이션: 실제로는 OOTDiffusion, VITON 등을 로드
        self.loaded_models[model_key] = {
            "type": "virtual_tryon",
            "model": "시뮬레이션 모델",
            "device": self.device,
            "memory_usage": "2.1GB",
            "status": "ready"
        }
        
        return True
    
    def _load_segmentation_model(self, model_key: str, model_info: Dict, **kwargs) -> bool:
        """세그멘테이션 모델 로딩 (시뮬레이션)"""
        logger.info(f"✂️ 세그멘테이션 모델 로딩: {model_key}")
        
        # 시뮬레이션: 실제로는 SAM 등을 로드
        self.loaded_models[model_key] = {
            "type": "segmentation",
            "model": "시뮬레이션 모델",
            "device": self.device,
            "memory_usage": "1.8GB",
            "status": "ready"
        }
        
        return True
    
    def _load_diffusion_model(self, model_key: str, model_info: Dict, **kwargs) -> bool:
        """디퓨전 모델 로딩 (시뮬레이션)"""
        logger.info(f"🎨 디퓨전 모델 로딩: {model_key}")
        
        # 시뮬레이션: 실제로는 Stable Diffusion 등을 로드
        self.loaded_models[model_key] = {
            "type": "base_diffusion",
            "model": "시뮬레이션 모델",
            "device": self.device,
            "memory_usage": "3.5GB",
            "status": "ready"
        }
        
        return True
    
    def _load_generic_model(self, model_key: str, model_info: Dict, **kwargs) -> bool:
        """일반 모델 로딩 (시뮬레이션)"""
        logger.info(f"🔧 일반 모델 로딩: {model_key}")
        
        self.loaded_models[model_key] = {
            "type": model_info.get("type", "generic"),
            "model": "시뮬레이션 모델",
            "device": self.device,
            "memory_usage": "1.2GB",
            "status": "ready"
        }
        
        return True
    
    def _get_model_info(self, model_key: str) -> Optional[Dict]:
        """모델 정보 조회"""
        try:
            from app.core.model_paths import get_model_info
            return get_model_info(model_key)
        except ImportError:
            logger.warning("⚠️ 모델 경로 정보를 불러올 수 없습니다")
            return None
        except Exception as e:
            logger.error(f"❌ 모델 정보 조회 오류: {e}")
            return None
    
    def _check_memory(self) -> Dict[str, float]:
        """메모리 상태 확인"""
        memory = psutil.virtual_memory()
        
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "usage_percent": memory.percent
        }
    
    def _cleanup_memory(self):
        """메모리 정리"""
        logger.info("🧹 메모리 정리 시작...")
        
        # Python 가비지 컬렉션
        gc.collect()
        
        # GPU 메모리 정리
        if self.device == "mps":
            if hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        
        logger.info("✅ 메모리 정리 완료")

# 전역 모델 매니저 인스턴스
model_manager = ModelManager()

# 편의 함수들
def load_model(model_key: str, **kwargs) -> bool:
    """모델 로드"""
    return model_manager.load_model(model_key, **kwargs)

def unload_model(model_key: str) -> bool:
    """모델 언로드"""
    return model_manager.unload_model(model_key)

def get_model_status() -> Dict[str, Any]:
    """모델 상태 조회"""
    return model_manager.get_model_status()

def get_available_models() -> Dict[str, bool]:
    """사용 가능한 모델 목록"""
    return model_manager.get_available_models()