import torch
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class CheckpointKeyMapper:
    """
    Intelligent checkpoint key mapping utility for different model architectures.
    Analyzes checkpoint and model structures to create proper key mappings.
    """
    
    def __init__(self):
        self.mapping_cache = {}
    
    def analyze_checkpoint_structure(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Analyze checkpoint structure and return detailed information.
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Analyze key patterns
            key_patterns = {}
            for key in checkpoint.keys():
                parts = key.split('.')
                if len(parts) >= 2:
                    pattern = f"{parts[0]}.{parts[1]}"
                    if pattern not in key_patterns:
                        key_patterns[pattern] = []
                    key_patterns[pattern].append(key)
            
            return {
                'total_keys': len(checkpoint),
                'key_patterns': key_patterns,
                'sample_keys': list(checkpoint.keys())[:20],
                'checkpoint': checkpoint
            }
        except Exception as e:
            logger.error(f"체크포인트 분석 실패: {e}")
            return {}
    
    def analyze_model_structure(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Analyze model structure and return detailed information.
        """
        try:
            state_dict = model.state_dict()
            
            # Analyze key patterns
            key_patterns = {}
            for key in state_dict.keys():
                parts = key.split('.')
                if len(parts) >= 2:
                    pattern = f"{parts[0]}.{parts[1]}"
                    if pattern not in key_patterns:
                        key_patterns[pattern] = []
                    key_patterns[pattern].append(key)
            
            return {
                'total_keys': len(state_dict),
                'key_patterns': key_patterns,
                'sample_keys': list(state_dict.keys())[:20],
                'state_dict': state_dict
            }
        except Exception as e:
            logger.error(f"모델 구조 분석 실패: {e}")
            return {}
    
    def create_ootd_mapping(self, checkpoint_path: str, model: torch.nn.Module) -> Dict[str, str]:
        """
        Create intelligent key mapping for OOTD model.
        """
        logger.info("🔍 OOTD 키 매핑 생성 시작...")
        
        # Analyze structures
        checkpoint_info = self.analyze_checkpoint_structure(checkpoint_path)
        model_info = self.analyze_model_structure(model)
        
        if not checkpoint_info or not model_info:
            logger.error("체크포인트 또는 모델 분석 실패")
            return {}
        
        logger.info(f"📊 체크포인트 키 수: {checkpoint_info['total_keys']}")
        logger.info(f"📊 모델 키 수: {model_info['total_keys']}")
        
        # Create mapping based on architectural analysis
        mapping = {}
        
        # Get all keys
        checkpoint_keys = list(checkpoint_info['checkpoint'].keys())
        model_keys = list(model_info['state_dict'].keys())
        
        # 1. Direct key mapping (exact matches)
        for ckpt_key in checkpoint_keys:
            if ckpt_key in model_keys:
                mapping[ckpt_key] = ckpt_key
                logger.debug(f"✅ 직접 매핑: {ckpt_key}")
        
        # 2. Pattern-based mapping for OOTD
        checkpoint_patterns = checkpoint_info['key_patterns']
        model_patterns = model_info['key_patterns']
        
        # Map time embedding
        if 'time_embedding' in checkpoint_patterns and 'time_embedding' in model_patterns:
            ckpt_time_keys = checkpoint_patterns['time_embedding']
            model_time_keys = model_patterns['time_embedding']
            
            # Map linear layers
            for i, ckpt_key in enumerate(ckpt_time_keys):
                if 'linear_1' in ckpt_key:
                    if i < len(model_time_keys):
                        mapping[ckpt_key] = model_time_keys[i]
                elif 'linear_2' in ckpt_key:
                    if i < len(model_time_keys):
                        mapping[ckpt_key] = model_time_keys[i]
        
        # 3. Advanced pattern matching
        # Map down_blocks to encoder_blocks
        if 'down_blocks' in checkpoint_patterns and 'encoder_blocks' in model_patterns:
            ckpt_down_keys = checkpoint_patterns['down_blocks']
            model_encoder_keys = model_patterns['encoder_blocks']
            
            # Create block-to-block mapping
            for ckpt_key in ckpt_down_keys:
                if ckpt_key not in mapping:  # Skip if already mapped
                    parts = ckpt_key.split('.')
                    if len(parts) >= 3:
                        block_num = parts[1]
                        
                        # Find corresponding encoder block
                        for model_key in model_encoder_keys:
                            if f"encoder_blocks.{block_num}" in model_key:
                                # Check if they have similar structure
                                if self._keys_similar(ckpt_key, model_key):
                                    mapping[ckpt_key] = model_key
                                    logger.debug(f"✅ 블록 매핑: {ckpt_key} → {model_key}")
                                    break
        
        # Map up_blocks to decoder_blocks
        if 'up_blocks' in checkpoint_patterns and 'decoder_blocks' in model_patterns:
            ckpt_up_keys = checkpoint_patterns['up_blocks']
            model_decoder_keys = model_patterns['decoder_blocks']
            
            for ckpt_key in ckpt_up_keys:
                if ckpt_key not in mapping:  # Skip if already mapped
                    parts = ckpt_key.split('.')
                    if len(parts) >= 3:
                        block_num = parts[1]
                        
                        for model_key in model_decoder_keys:
                            if f"decoder_blocks.{block_num}" in model_key:
                                if self._keys_similar(ckpt_key, model_key):
                                    mapping[ckpt_key] = model_key
                                    logger.debug(f"✅ 블록 매핑: {ckpt_key} → {model_key}")
                                    break
        
        # 4. Attention mapping
        if 'cross_attention' in model_patterns:
            model_attn_keys = model_patterns['cross_attention']
            
            for ckpt_key in checkpoint_keys:
                if ckpt_key not in mapping and 'attentions' in ckpt_key and 'transformer_blocks' in ckpt_key:
                    for model_key in model_attn_keys:
                        if 'to_' in model_key and self._keys_similar(ckpt_key, model_key):
                            mapping[ckpt_key] = model_key
                            logger.debug(f"✅ 어텐션 매핑: {ckpt_key} → {model_key}")
                            break
        
        # 5. Cloth encoder mapping
        if 'cloth_encoder' in model_patterns:
            model_cloth_keys = model_patterns['cloth_encoder']
            
            for ckpt_key in checkpoint_keys:
                if ckpt_key not in mapping and 'down_blocks' in ckpt_key and 'resnets' in ckpt_key:
                    for model_key in model_cloth_keys:
                        if 'encoder' in model_key and self._keys_similar(ckpt_key, model_key):
                            mapping[ckpt_key] = model_key
                            logger.debug(f"✅ 클로스 인코더 매핑: {ckpt_key} → {model_key}")
                            break
        
        # 6. Conv mapping
        if 'conv_in' in checkpoint_patterns:
            ckpt_conv_in = checkpoint_patterns['conv_in']
            for ckpt_key in ckpt_conv_in:
                if ckpt_key not in mapping:
                    # Map to first encoder block
                    for model_key in model_keys:
                        if 'encoder_blocks.0' in model_key and self._keys_similar(ckpt_key, model_key):
                            mapping[ckpt_key] = model_key
                            logger.debug(f"✅ Conv In 매핑: {ckpt_key} → {model_key}")
                            break
        
        if 'conv_out' in checkpoint_patterns:
            ckpt_conv_out = checkpoint_patterns['conv_out']
            for ckpt_key in ckpt_conv_out:
                if ckpt_key not in mapping:
                    # Map to last decoder block
                    for model_key in model_keys:
                        if 'decoder_blocks' in model_key and self._keys_similar(ckpt_key, model_key):
                            mapping[ckpt_key] = model_key
                            logger.debug(f"✅ Conv Out 매핑: {ckpt_key} → {model_key}")
                            break
        
        # 7. Fallback: Similar key matching
        if len(mapping) == 0:
            logger.warning("⚠️ 패턴 매핑이 실패했습니다. 유사 키 매핑을 시도합니다...")
            
            for ckpt_key in checkpoint_keys:
                if ckpt_key not in mapping:
                    best_match = None
                    best_score = 0
                    
                    for model_key in model_keys:
                        if model_key not in mapping.values():  # Avoid duplicate mappings
                            score = self._calculate_key_similarity(ckpt_key, model_key)
                            if score > best_score and score > 0.5:  # Minimum similarity threshold
                                best_score = score
                                best_match = model_key
                    
                    if best_match:
                        mapping[ckpt_key] = best_match
                        logger.debug(f"✅ 유사 키 매핑: {ckpt_key} → {best_match} (점수: {best_score:.2f})")
        
        logger.info(f"✅ OOTD 키 매핑 생성 완료: {len(mapping)}개 매핑")
        
        # Log some mappings for verification
        sample_mappings = list(mapping.items())[:10]
        for ckpt_key, model_key in sample_mappings:
            logger.info(f"  {ckpt_key} → {model_key}")
        
        return mapping
    
    def create_viton_hd_mapping(self, checkpoint_path: str, model: torch.nn.Module) -> Dict[str, str]:
        """
        Create intelligent key mapping for VITON-HD model.
        """
        logger.info("🔍 VITON-HD 키 매핑 생성 시작...")
        
        # Analyze structures
        checkpoint_info = self.analyze_checkpoint_structure(checkpoint_path)
        model_info = self.analyze_model_structure(model)
        
        if not checkpoint_info or not model_info:
            logger.error("체크포인트 또는 모델 분석 실패")
            return {}
        
        # Create mapping based on VITON-HD architecture
        mapping = {}
        
        # VITON-HD typically has simpler structure
        # Map keys directly if they match
        checkpoint_keys = set(checkpoint_info['checkpoint'].keys())
        model_keys = set(model_info['state_dict'].keys())
        
        # Direct mapping for matching keys
        for ckpt_key in checkpoint_keys:
            if ckpt_key in model_keys:
                mapping[ckpt_key] = ckpt_key
        
        # Pattern-based mapping for non-matching keys
        for ckpt_key in checkpoint_keys:
            if ckpt_key not in mapping:
                # Try to find similar keys
                for model_key in model_keys:
                    if self._keys_similar(ckpt_key, model_key):
                        mapping[ckpt_key] = model_key
                        break
        
        logger.info(f"✅ VITON-HD 키 매핑 생성 완료: {len(mapping)}개 매핑")
        return mapping
    
    def create_stable_diffusion_mapping(self, checkpoint_path: str, model: torch.nn.Module) -> Dict[str, str]:
        """
        Create intelligent key mapping for Stable Diffusion model.
        """
        logger.info("🔍 Stable Diffusion 키 매핑 생성 시작...")
        
        # Analyze structures
        checkpoint_info = self.analyze_checkpoint_structure(checkpoint_path)
        model_info = self.analyze_model_structure(model)
        
        if not checkpoint_info or not model_info:
            logger.error("체크포인트 또는 모델 분석 실패")
            return {}
        
        # Create mapping based on Stable Diffusion architecture
        mapping = {}
        
        # Stable Diffusion typically has complex transformer structure
        checkpoint_keys = set(checkpoint_info['checkpoint'].keys())
        model_keys = set(model_info['state_dict'].keys())
        
        # Direct mapping for matching keys
        for ckpt_key in checkpoint_keys:
            if ckpt_key in model_keys:
                mapping[ckpt_key] = ckpt_key
        
        # Pattern-based mapping for non-matching keys
        for ckpt_key in checkpoint_keys:
            if ckpt_key not in mapping:
                # Try to find similar keys
                for model_key in model_keys:
                    if self._keys_similar(ckpt_key, model_key):
                        mapping[ckpt_key] = model_key
                        break
        
        logger.info(f"✅ Stable Diffusion 키 매핑 생성 완료: {len(mapping)}개 매핑")
        return mapping
    
    def _keys_similar(self, key1: str, key2: str) -> bool:
        """
        Check if two keys are similar enough for mapping.
        """
        parts1 = key1.split('.')
        parts2 = key2.split('.')
        
        # Check if they have similar structure
        if len(parts1) != len(parts2):
            return False
        
        # Check if they end with similar suffixes
        if parts1[-1] != parts2[-1]:  # weight, bias, etc.
            return False
        
        # Check if they have similar prefixes
        if len(parts1) >= 2 and len(parts2) >= 2:
            if parts1[0] != parts2[0]:
                return False
        
        return True
    
    def _calculate_key_similarity(self, key1: str, key2: str) -> float:
        """
        Calculate similarity score between two keys (0.0 to 1.0).
        """
        parts1 = key1.split('.')
        parts2 = key2.split('.')
        
        # Base similarity
        if parts1 == parts2:
            return 1.0
        
        # Check suffix similarity (weight, bias, etc.)
        suffix_similarity = 1.0 if parts1[-1] == parts2[-1] else 0.0
        
        # Check prefix similarity
        prefix_similarity = 0.0
        min_len = min(len(parts1), len(parts2))
        if min_len > 0:
            matching_prefix = 0
            for i in range(min_len):
                if parts1[i] == parts2[i]:
                    matching_prefix += 1
                else:
                    break
            prefix_similarity = matching_prefix / min_len
        
        # Check structural similarity
        structural_similarity = 1.0 - abs(len(parts1) - len(parts2)) / max(len(parts1), len(parts2))
        
        # Check keyword similarity
        keywords1 = set(parts1)
        keywords2 = set(parts2)
        keyword_similarity = len(keywords1 & keywords2) / len(keywords1 | keywords2) if keywords1 | keywords2 else 0.0
        
        # Weighted combination
        similarity = (
            suffix_similarity * 0.3 +
            prefix_similarity * 0.3 +
            structural_similarity * 0.2 +
            keyword_similarity * 0.2
        )
        
        return similarity
    
    def apply_mapping(self, checkpoint: Dict[str, torch.Tensor], mapping: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """
        Apply key mapping to checkpoint.
        """
        mapped_checkpoint = {}
        
        for ckpt_key, model_key in mapping.items():
            if ckpt_key in checkpoint:
                mapped_checkpoint[model_key] = checkpoint[ckpt_key]
        
        logger.info(f"✅ 키 매핑 적용 완료: {len(mapped_checkpoint)}개 키")
        return mapped_checkpoint
    
    def validate_mapping(self, mapped_checkpoint: Dict[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, Any]:
        """
        Validate the mapped checkpoint against the model.
        """
        model_state_dict = model.state_dict()
        
        # Check coverage
        model_keys = set(model_state_dict.keys())
        mapped_keys = set(mapped_checkpoint.keys())
        
        coverage = len(mapped_keys & model_keys) / len(model_keys) * 100
        
        # Check shape compatibility
        shape_matches = 0
        shape_mismatches = []
        
        for key in mapped_keys & model_keys:
            if key in mapped_checkpoint and key in model_state_dict:
                if mapped_checkpoint[key].shape == model_state_dict[key].shape:
                    shape_matches += 1
                else:
                    shape_mismatches.append({
                        'key': key,
                        'checkpoint_shape': mapped_checkpoint[key].shape,
                        'model_shape': model_state_dict[key].shape
                    })
        
        return {
            'coverage_percentage': coverage,
            'total_model_keys': len(model_keys),
            'mapped_keys': len(mapped_keys),
            'shape_matches': shape_matches,
            'shape_mismatches': shape_mismatches,
            'missing_keys': list(model_keys - mapped_keys),
            'extra_keys': list(mapped_keys - model_keys)
        } 