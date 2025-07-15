def _load_external_config(self, config_path: str):
        """외부 설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.json'):
                    external_config = json.load(f)
                elif config_path.endswith(('.yml', '.yaml')):
                    try:
                        import yaml
                        external_config = yaml.safe_load(f)
                    except ImportError:
                        logger.warning("⚠️ PyYAML not installed, skipping YAML config")
                        return
                else:
                    logger.warning(f"⚠️ Unsupported config file format: {config_path}")
                    return
            
            # 딥 머지
            self._deep_merge(self.config, external_config)
            logger.info(f"✅ 외부 설정 로드 완료: {config_path}")
            
        except FileNotFoundError:
            logger.warning(f"⚠️ 설정 파일 없음: {config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON 파싱 오류: {e}")
        except Exception as e:
            logger.error(f"❌ 외부 설정 로드 실패: {e}")
    
    def _apply_environment_overrides(self):
        """환경변수 기반 설정 오버라이드"""
        
        # 품질 레벨
        quality = os.getenv("PIPELINE_QUALITY_LEVEL", self.quality_level)
        if quality != self.quality_level:
            self.quality_level = quality
            self.config["pipeline"]["quality_level"] = quality
            logger.info(f"🔄 환경변수: 품질 레벨 = {quality}")
        
        # 디바이스 설정
        device_override = os.getenv("PIPELINE_DEVICE")
        if device_override and device_override != self.device:
            self.device = device_override
            self.config["optimization"]["device"] = device_override
            self.config["system"]["device"] = device_override
            logger.info(f"🔄 환경변수: 디바이스 = {device_override}")
        
        # 메모리 제한
        memory_limit = os.getenv("PIPELINE_MEMORY_LIMIT")
        if memory_limit:
            self.config["memory"]["max_memory_usage"] = memory_limit
            logger.info(f"🔄 환경변수: 메모리 제한 = {memory_limit}")
        
        # 동시 처리 수
        max_concurrent = os.getenv("PIPELINE_MAX_CONCURRENT")
        if max_concurrent:
            try:
                concurrent_val = int(max_concurrent)
                self.max_concurrent_requests = concurrent_val
                self.config["pipeline"]["max_concurrent_requests"] = concurrent_val
                logger.info(f"🔄 환경변수: 동시 처리 = {concurrent_val}")
            except ValueError:
                logger.warning(f"⚠️ 잘못된 동시 처리 값: {max_concurrent}")
        
        # 타임아웃
        timeout = os.getenv("PIPELINE_TIMEOUT")
        if timeout:
            try:
                timeout_val = int(timeout)
                self.timeout_seconds = timeout_val
                self.config["pipeline"]["timeout_seconds"] = timeout_val
                logger.info(f"🔄 환경변수: 타임아웃 = {timeout_val}초")
            except ValueError:
                logger.warning(f"⚠️ 잘못된 타임아웃 값: {timeout}")
        
        # 디버그 모드
        debug_mode = os.getenv("PIPELINE_DEBUG", "false").lower() == "true"
        if debug_mode != self.debug_mode:
            self.debug_mode = debug_mode
            self.config["logging"]["debug_mode"] = debug_mode
            self.config["logging"]["level"] = "DEBUG" if debug_mode else "INFO"
            self.config["logging"]["save_intermediate"] = debug_mode
            logger.info(f"🔄 환경변수: 디버그 모드 = {debug_mode}")
        
        # 최적화 활성화/비활성화
        optimization_override = os.getenv("PIPELINE_OPTIMIZATION")
        if optimization_override:
            enable_opt = optimization_override.lower() == "true"
            if enable_opt != self.optimization_enabled:
                self.optimization_enabled = enable_opt
                self.config["optimization"]["optimization_enabled"] = enable_opt
                self.config["system"]["optimization_enabled"] = enable_opt
                logger.info(f"🔄 환경변수: 최적화 = {enable_opt}")
        
        # 캐싱 설정
        caching_override = os.getenv("PIPELINE_CACHING")
        if caching_override:
            enable_cache = caching_override.lower() == "true"
            if enable_cache != self.enable_caching:
                self.enable_caching = enable_cache
                self.config["pipeline"]["enable_caching"] = enable_cache
                self.config["optimization"]["caching"]["enabled"] = enable_cache
                logger.info(f"🔄 환경변수: 캐싱 = {enable_cache}")
    
    def _apply_device_optimizations(self):
        """디바이스별 최적화 적용"""
        
        if self.device == "mps":
            # M3 Max MPS 최적화
            self.config["optimization"].update({
                "mixed_precision": self.optimization_enabled,
                "memory_efficient_attention": True,
                "compile_models": False,  # MPS에서는 컴파일 비활성화
                "batch_processing": {
                    "enabled": self.batch_processing,
                    "max_batch_size": 2 if self.is_m3_max else 1,  # M3 Max 메모리 제한
                    "dynamic_batching": False  # 안정성을 위해 비활성화
                }
            })
            
            # M3 Max 전용 최적화
            if self.is_m3_max:
                self.config["memory"]["max_memory_usage"] = "70%"  # 안전 마진
                self.config["optimization"]["model_offloading"]["keep_active"] = 3
                logger.info("🍎 M3 Max MPS 최적화 적용")
            
            # 이미지 크기 조정 (메모리 효율성)
            if self.quality_level in ["fast", "balanced"]:
                self.config["image"]["input_size"] = (512, 512)
                self.config["image"]["max_resolution"] = 1024
            
        elif self.device == "cuda":
            # CUDA 최적화
            self.config["optimization"].update({
                "mixed_precision": self.optimization_enabled,
                "gradient_checkpointing": True,
                "compile_models": self.optimization_enabled,
                "batch_processing": {
                    "enabled": self.batch_processing,
                    "max_batch_size": 8 if self.memory_gb >= 16 else 4,
                    "dynamic_batching": self.dynamic_batching
                }
            })
            
            # CUDA 메모리 관리
            self.config["memory"]["aggressive_cleanup"] = True
            self.config["memory"]["model_offloading"]["enabled"] = self.memory_gb < 24
            logger.info("🔥 CUDA 최적화 적용")
            
        else:
            # CPU 최적화
            self.config["optimization"].update({
                "mixed_precision": False,
                "compile_models": False,
                "batch_processing": {
                    "enabled": False,
                    "max_batch_size": 1
                }
            })
            
            # CPU에서는 더 작은 모델 사용
            self.config["steps"]["virtual_fitting"]["num_inference_steps"] = 20
            self.config["steps"]["post_processing"]["super_resolution"]["enabled"] = False
            self.config["steps"]["post_processing"]["face_enhancement"]["enabled"] = False
            logger.info("💻 CPU 최적화 적용")
    
    def _apply_quality_preset(self, quality_level: str):
        """품질 레벨에 따른 프리셋 적용"""
        
        quality_presets = {
            "low": {
                "image_size": (256, 256),
                "inference_steps": 15,
                "super_resolution": False,
                "face_enhancement": False,
                "physics_simulation": False,
                "advanced_post_processing": False,
                "timeout": 60,
                "batch_size": 1
            },
            "medium": {
                "image_size": (384, 384),
                "inference_steps": 25,
                "super_resolution": False,
                "face_enhancement": False,
                "physics_simulation": True,
                "advanced_post_processing": False,
                "timeout": 120,
                "batch_size": 1 if self.device == 'cpu' else 2
            },
            "high": {
                "image_size": (512, 512),
                "inference_steps": 50,
                "super_resolution": self.optimization_enabled,
                "face_enhancement": self.optimization_enabled,
                "physics_simulation": True,
                "advanced_post_processing": True,
                "timeout": 300,
                "batch_size": 1 if self.device == 'cpu' else 2
            },
            "ultra": {
                "image_size": (768, 768) if self.is_m3_max else (512, 512),
                "inference_steps": 100,
                "super_resolution": self.optimization_enabled,
                "face_enhancement": self.optimization_enabled,
                "physics_simulation": True,
                "advanced_post_processing": True,
                "timeout": 600,
                "batch_size": 1
            }
        }
        
        preset = quality_presets.get(quality_level, quality_presets["high"])
        
        # 이미지 크기
        self.config["image"]["input_size"] = preset["image_size"]
        self.config["image"]["output_size"] = preset["image_size"]
        
        # 추론 단계
        self.config["steps"]["virtual_fitting"]["num_inference_steps"] = preset["inference_steps"]
        
        # 후처리 설정
        self.config["steps"]["post_processing"]["super_resolution"]["enabled"] = preset["super_resolution"]
        self.config["steps"]["post_processing"]["face_enhancement"]["enabled"] = preset["face_enhancement"]
        self.config["steps"]["post_processing"]["detail_enhancement"]["enabled"] = preset["advanced_post_processing"]
        self.config["steps"]["post_processing"]["edge_enhancement"]["enabled"] = preset["advanced_post_processing"]
        
        # 물리 시뮬레이션
        self.config["steps"]["cloth_warping"]["physics_enabled"] = preset["physics_simulation"]
        
        # 배치 크기
        for step_name in ["human_parsing"]:
            if step_name in self.config["steps"]:
                self.config["steps"][step_name]["batch_size"] = preset["batch_size"]
        
        # 타임아웃
        self.config["pipeline"]["timeout_seconds"] = preset["timeout"]
        
        logger.info(f"🎯 품질 프리셋 적용: {quality_level} (해상도: {preset['image_size']})")
    
    def _apply_custom_configurations(self):
        """사용자 정의 설정 적용"""
        
        # 커스텀 단계 설정
        if self.custom_step_configs:
            for step_name, step_config in self.custom_step_configs.items():
                if step_name in self.config["steps"]:
                    self._deep_merge(self.config["steps"][step_name], step_config)
                    logger.info(f"🔧 커스텀 단계 설정 적용: {step_name}")
        
        # 커스텀 모델 경로
        if self.custom_model_paths:
            self.config["model_paths"]["custom_paths"].update(self.custom_model_paths)
            logger.info(f"📁 커스텀 모델 경로 적용: {len(self.custom_model_paths)}개")
        
        # 실험적 기능 적용
        if self.experimental_features:
            # 신경망 스타일 전송
            if self.experimental_features.get('neural_style_transfer'):
                self.config["steps"]["virtual_fitting"]["style_transfer"]["enable"] = True
                logger.info("🧪 실험적 기능: 신경망 스타일 전송 활성화")
            
            # 고급 물리 시뮬레이션
            if self.experimental_features.get('advanced_physics'):
                self.config["steps"]["cloth_warping"]["gravity_simulation"] = True
                self.config["steps"]["cloth_warping"]["wind_simulation"] = True
                logger.info("🧪 실험적 기능: 고급 물리 시뮬레이션 활성화")
            
            # 다중 인물 지원
            if self.experimental_features.get('multi_person_support'):
                self.config["steps"]["pose_estimation"]["multi_person"] = True
                self.config["steps"]["human_parsing"]["multi_person"] = True
                logger.info("🧪 실험적 기능: 다중 인물 지원 활성화")
    
    def _deep_merge(self, base_dict: Dict, update_dict: Dict):
        """딕셔너리 딥 머지 (개선된 버전)"""
        for key, value in update_dict.items():
            if key in base_dict:
                if isinstance(base_dict[key], dict) and isinstance(value, dict):
                    self._deep_merge(base_dict[key], value)
                elif isinstance(base_dict[key], list) and isinstance(value, list):
                    # 리스트 병합 (중복 제거)
                    base_dict[key] = list(set(base_dict[key] + value))
                else:
                    base_dict[key] = value
            else:
                base_dict[key] = value
    
    # ===============================================================
    # ✅ 최적 생성자 패턴 - 설정 접근 메서드들 (완전 개선)
    # ===============================================================
    
    def get_step_config(self, step_name: str) -> Dict[str, Any]:
        """특정 단계 설정 반환"""
        step_config = self.config["steps"].get(step_name, {})
        if not step_config:
            logger.warning(f"⚠️ 단계 설정 없음: {step_name}")
        return step_config
    
    def get_model_path(self, model_name: str) -> str:
        """모델 파일 경로 반환 (개선된 버전)"""
        # 커스텀 경로 우선 확인
        if model_name in self.config["model_paths"]["custom_paths"]:
            return self.config["model_paths"]["custom_paths"][model_name]
        
        # 기본 경로 확인
        base_dir = self.config["model_paths"]["base_dir"]
        checkpoint_path = self.config["model_paths"]["checkpoints"].get(model_name)
        
        if checkpoint_path:
            full_path = os.path.join(base_dir, checkpoint_path)
            return full_path
        else:
            # 기본 경로 생성
            default_path = os.path.join(base_dir, model_name)
            logger.warning(f"⚠️ 모델 경로 추정: {model_name} -> {default_path}")
            return default_path
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """최적화 설정 반환"""
        return self.config["optimization"]
    
    def get_memory_config(self) -> Dict[str, Any]:
        """메모리 설정 반환"""
        return self.config["memory"]
    
    def get_image_config(self) -> Dict[str, Any]:
        """이미지 처리 설정 반환"""
        return self.config["image"]
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """파이프라인 전역 설정 반환"""
        return self.config["pipeline"]
    
    def get_system_config(self) -> Dict[str, Any]:
        """✅ 최적 생성자 패턴 - 시스템 설정 반환"""
        return {
            "device": self.device,
            "device_type": self.device_type,
            "memory_gb": self.memory_gb,
            "cpu_cores": self.system_config.cpu_cores,
            "is_m3_max": self.is_m3_max,
            "platform": self.system_config.platform,
            "architecture": self.system_config.architecture,
            "optimization_enabled": self.optimization_enabled,
            "quality_level": self.quality_level,
            "torch_available": self.system_config.torch_available,
            "constructor_pattern": "optimal"
        }
    
    def get_experimental_config(self) -> Dict[str, Any]:
        """실험적 기능 설정 반환"""
        return self.config["experimental"]
    
    def get_logging_config(self) -> Dict[str, Any]:
        """로깅 설정 반환"""
        return self.config["logging"]
    
    # ===============================================================
    # ✅ 최적 생성자 패턴 - 동적 설정 변경 메서드들 (완전 개선)
    # ===============================================================
    
    def update_quality_level(self, quality_level: str):
        """품질 레벨 동적 변경"""
        valid_levels = ["low", "medium", "high", "ultra"]
        if quality_level not in valid_levels:
            logger.warning(f"⚠️ 잘못된 품질 레벨: {quality_level}, 유효값: {valid_levels}")
            return False
        
        if quality_level != self.quality_level:
            old_level = self.quality_level
            self.quality_level = quality_level
            self._apply_quality_preset(quality_level)
            logger.info(f"🔄 품질 레벨 변경: {old_level} -> {quality_level}")
            return True
        return False
    
    def update_device(self, device: str):
        """디바이스 동적 변경"""
        valid_devices = ["auto", "cpu", "cuda", "mps"]
        if device not in valid_devices:
            logger.warning(f"⚠️ 잘못된 디바이스: {device}, 유효값: {valid_devices}")
            return False
        
        if device != self.device:
            old_device = self.device
            
            if device == "auto":
                self.device = self._determine_optimal_device(None, self.is_m3_max, 
                                                           self.system_config.mps_available, 
                                                           self.system_config.cuda_available)
            else:
                self.device = device
            
            self.config["optimization"]["device"] = self.device
            self.config["system"]["device"] = self.device
            self._apply_device_optimizations()
            logger.info(f"🔄 디바이스 변경: {old_device} -> {self.device}")
            return True
        return False
    
    def update_memory_limit(self, memory_gb: float):
        """✅ 최적 생성자 패턴 - 메모리 제한 동적 변경"""
        if memory_gb <= 0:
            logger.warning(f"⚠️ 잘못된 메모리 크기: {memory_gb}GB")
            return False
        
        old_memory = self.memory_gb
        self.memory_gb = memory_gb
        self.config["memory"]["memory_gb"] = memory_gb
        self.config["system"]["memory_gb"] = memory_gb
        self.config["memory"]["max_memory_usage"] = f"{min(80, int(memory_gb * 0.8))}%"
        
        # 메모리 크기에 따른 배치 크기 조정
        if memory_gb >= 32:
            max_batch = 4
        elif memory_gb >= 16:
            max_batch = 2
        else:
            max_batch = 1
        
        if self.device != 'cpu':
            self.config["optimization"]["batch_processing"]["max_batch_size"] = max_batch
        
        logger.info(f"🔄 메모리 제한 변경: {old_memory}GB -> {memory_gb}GB")
        return True
    
    def toggle_optimization(self, enabled: bool):
        """✅ 최적 생성자 패턴 - 최적화 토글"""
        if enabled == self.optimization_enabled:
            return False
        
        self.optimization_enabled = enabled
        self.config["optimization"]["optimization_enabled"] = enabled
        self.config["system"]["optimization_enabled"] = enabled
        
        # 관련 설정들 업데이트
        self.config["optimization"]["mixed_precision"] = enabled
        self.config["steps"]["post_processing"]["super_resolution"]["enabled"] = (
            enabled and self.super_resolution_enabled
        )
        self.config["steps"]["human_parsing"]["enable_quantization"] = enabled
        
        logger.info(f"🔄 최적화 모드: {'활성화' if enabled else '비활성화'}")
        return True
    
    def toggle_caching(self, enabled: bool):
        """캐싱 토글"""
        if enabled == self.enable_caching:
            return False
        
        self.enable_caching = enabled
        self.config["pipeline"]["enable_caching"] = enabled
        self.config["optimization"]["caching"]["enabled"] = enabled
        
        # 단계별 캐싱 설정 업데이트
        for step_config in self.config["steps"].values():
            if "cache_enabled" in step_config:
                step_config["cache_enabled"] = enabled
        
        logger.info(f"🔄 캐싱: {'활성화' if enabled else '비활성화'}")
        return True
    
    def toggle_debug_mode(self, enabled: bool):
        """디버그 모드 토글"""
        if enabled == self.debug_mode:
            return False
        
        self.debug_mode = enabled
        self.save_debug_info = enabled
        self.config["logging"]["debug_mode"] = enabled
        self.config["logging"]["save_debug_info"] = enabled
        self.config["logging"]["save_intermediate"] = enabled
        self.config["logging"]["level"] = "DEBUG" if enabled else "INFO"
        self.config["pipeline"]["save_debug_info"] = enabled
        
        logger.info(f"🔄 디버그 모드: {'활성화' if enabled else '비활성화'}")
        return True
    
    def update_concurrent_requests(self, max_requests: int):
        """동시 요청 수 변경"""
        if max_requests <= 0 or max_requests > 32:
            logger.warning(f"⚠️ 잘못된 동시 요청 수: {max_requests} (1-32 사이여야 함)")
            return False
        
        old_max = self.max_concurrent_requests
        self.max_concurrent_requests = max_requests
        self.config["pipeline"]["max_concurrent_requests"] = max_requests
        
        logger.info(f"🔄 동시 요청 수 변경: {old_max} -> {max_requests}")
        return True
    
    def update_timeout(self, timeout_seconds: int):
        """타임아웃 변경"""
        if timeout_seconds <= 0 or timeout_seconds > 3600:
            logger.warning(f"⚠️ 잘못된 타임아웃: {timeout_seconds}초 (1-3600 사이여야 함)")
            return False
        
        old_timeout = self.timeout_seconds
        self.timeout_seconds = timeout_seconds
        self.config["pipeline"]["timeout_seconds"] = timeout_seconds
        
        logger.info(f"🔄 타임아웃 변경: {old_timeout}초 -> {timeout_seconds}초")
        return True
    
    def enable_experimental_feature(self, feature_name: str, enabled: bool = True):
        """실험적 기능 활성화/비활성화"""
        if feature_name not in self.experimental_features:
            self.experimental_features[feature_name] = enabled
        else:
            self.experimental_features[feature_name] = enabled
        
        self.config["experimental"]["features"][feature_name] = enabled
        self._apply_custom_configurations()  # 실험적 기능 재적용
        
        logger.info(f"🧪 실험적 기능 {feature_name}: {'활성화' if enabled else '비활성화'}")
        return True
    
    def add_custom_model_path(self, model_name: str, model_path: str):
        """커스텀 모델 경로 추가"""
        if not os.path.exists(model_path):
            logger.warning(f"⚠️ 모델 경로가 존재하지 않음: {model_path}")
            return False
        
        self.custom_model_paths[model_name] = model_path
        self.config["model_paths"]["custom_paths"][model_name] = model_path
        
        logger.info(f"📁 커스텀 모델 경로 추가: {model_name} -> {model_path}")
        return True
    
    def update_step_config(self, step_name: str, step_config: Dict[str, Any]):
        """단계별 설정 업데이트"""
        if step_name not in self.config["steps"]:
            logger.warning(f"⚠️ 존재하지 않는 단계: {step_name}")
            return False
        
        self._deep_merge(self.config["steps"][step_name], step_config)
        logger.info(f"🔧 단계 설정 업데이트: {step_name}")
        return True
    
    # ===============================================================
    # ✅ 최적 생성자 패턴 - 검증 및 진단 메서드들 (완전 개선)
    # ===============================================================
    
    def validate_config(self) -> Dict[str, Any]:
        """설정 유효성 검사 (완전 개선 버전)"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": [],
            "constructor_pattern": "optimal",
            "validation_timestamp": self._get_timestamp()
        }
        
        # 필수 단계 확인
        required_steps = [
            "human_parsing", "pose_estimation", "cloth_segmentation",
            "geometric_matching", "cloth_warping", "virtual_fitting",
            "post_processing", "quality_assessment"
        ]
        
        for step in required_steps:
            if step not in self.config["steps"]:
                validation_result["errors"].append(f"필수 파이프라인 단계 누락: {step}")
                validation_result["valid"] = False
        
        # 모델 경로 확인
        missing_models = []
        for model_name, checkpoint_path in self.config["model_paths"]["checkpoints"].items():
            full_path = self.get_model_path(model_name)
            model_dir = os.path.dirname(full_path)
            if not os.path.exists(model_dir):
                missing_models.append(model_name)
        
        if missing_models:
            validation_result["warnings"].append(f"모델 디렉토리 없음: {', '.join(missing_models)}")
        
        # 디바이스 호환성 확인
        if self.device == "mps" and not self.system_config.mps_available:
            validation_result["errors"].append("MPS가 요청되었지만 사용할 수 없습니다")
            validation_result["valid"] = False
        
        if self.device == "cuda" and not self.system_config.cuda_available:
            validation_result["errors"].append("CUDA가 요청되었지만 사용할 수 없습니다")
            validation_result["valid"] = False
        
        # 메모리 설정 확인
        max_memory = self.config["memory"]["max_memory_usage"]
        if isinstance(max_memory, str) and max_memory.endswith("%"):
            try:
                percent = float(max_memory[:-1])
                if not (10 <= percent <= 95):
                    validation_result["errors"].append(f"메모리 사용률 범위 오류: {max_memory} (10-95% 사이여야 함)")
                    validation_result["valid"] = False
            except ValueError:
                validation_result["errors"].append(f"잘못된 메모리 형식: {max_memory}")
                validation_result["valid"] = False
        
        # 시스템 리소스 확인
        if self.memory_gb < 8:
            validation_result["warnings"].append(f"메모리 부족: {self.memory_gb}GB < 8GB (권장)")
        
        if self.memory_gb < 4:
            validation_result["errors"].append(f"메모리 심각 부족: {self.memory_gb}GB < 4GB (최소)")
            validation_result["valid"] = False
        
        # 품질 설정과 시스템 성능 호환성
        if self.quality_level == "ultra" and self.memory_gb < 16:
            validation_result["warnings"].append("Ultra 품질에는 16GB 이상 메모리 권장")
        
        if self.quality_level in ["high", "ultra"] and self.device == "cpu":
            validation_result["warnings"].append("High/Ultra 품질에는 GPU 사용 권장")
        
        # 최적 생성자 패턴 검증
        required_system_params = [
            "device", "device_type", "memory_gb", "is_m3_max", 
            "optimization_enabled", "quality_level"
        ]
        for param in required_system_params:
            if not hasattr(self, param):
                validation_result["errors"].append(f"필수 시스템 파라미터 누락: {param}")
                validation_result["valid"] = False
        
        # 실험적 기능 충돌 확인
        if self.experimental_features.get('multi_person_support') and self.quality_level == "ultra":
            validation_result["warnings"].append("다중 인물 지원과 Ultra 품질은 성능에 영향을 줄 수 있음")
        
        # 정보성 메시지
        validation_result["info"].append(f"파이프라인 버전: {self.config['pipeline']['version']}")
        validation_result["info"].append(f"생성자 패턴: {self.config['pipeline']['constructor_pattern']}")
        validation_result["info"].append(f"활성화된 실험적 기능: {len([k for k, v in self.experimental_features.items() if v])}개")
        
        return validation_result
    
    def get_system_info(self) -> Dict[str, Any]:
        """✅ 최적 생성자 패턴 - 시스템 정보 반환 (완전 개선 버전)"""
        base_info = super().get_system_info()
        
        # PipelineConfig 특화 정보 추가
        base_info.update({
            "pipeline_version": self.config["pipeline"]["version"],
            "quality_level": self.quality_level,
            "config_path": self.config_path,
            "device_info": self.device_info,
            "memory_config": self.get_memory_config(),
            "optimization_config": self.get_optimization_config(),
            "torch_version": torch.__version__ if TORCH_AVAILABLE else "N/A",
            "config_valid": self.validate_config()["valid"],
            "pipeline_mode": self.config["pipeline"]["processing_mode"],
            "constructor_pattern": "optimal",
            "total_steps": len(self.config["steps"]),
            "experimental_features_count": len([k for k, v in self.experimental_features.items() if v]),
            "custom_models_count": len(self.custom_model_paths),
            "performance_config": self.get_performance_config()
        })
        
        return base_info
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보"""
        return {
            "quality_level": self.quality_level,
            "device": self.device,
            "optimization_enabled": self.optimization_enabled,
            "expected_memory_usage": self.config["memory"]["max_memory_usage"],
            "max_concurrent_requests": self.max_concurrent_requests,
            "timeout_seconds": self.timeout_seconds,
            "caching_enabled": self.enable_caching,
            "parallel_processing": self.enable_parallel,
            "batch_processing": self.batch_processing,
            "m3_max_optimized": self.is_m3_max,
            "experimental_features_active": len([k for k, v in self.experimental_features.items() if v])
        }
    
    def export_config(self, file_path: str, include_system_info: bool = True):
        """설정을 파일로 내보내기 (완전 개선 버전)"""
        try:
            export_data = {
                "config": self.config,
                "export_metadata": {
                    "export_timestamp": self._get_timestamp(),
                    "exported_from": self.class_name,
                    "constructor_pattern": "optimal",
                    "pipeline_version": self.config["pipeline"]["version"]
                }
            }
            
            if include_system_info:
                export_data.update({
                    "system_info": self.get_system_info(),
                    "performance_summary": self.get_performance_summary(),
                    "validation_result": self.validate_config()
                })
            
            file_path_obj = Path(file_path)
            file_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"💾 설정 내보내기 완료: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 설정 내보내기 실패: {e}")
            return False
    
    def import_config(self, file_path: str, merge_mode: bool = True):
        """설정을 파일에서 가져오기"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            if "config" in import_data:
                imported_config = import_data["config"]
                
                if merge_mode:
                    self._deep_merge(self.config, imported_config)
                else:
                    self.config = imported_config
                
                logger.info(f"📥 설정 가져오기 완료: {file_path} ({'병합' if merge_mode else '교체'} 모드)")
                return True
            else:
                logger.error("❌ 잘못된 설정 파일 형식")
                return False
                
        except Exception as e:
            logger.error(f"❌ 설정 가져오기 실패: {e}")
            return False
    
    def create_config_backup(self, backup_dir: str = "config_backups"):
        """설정 백업 생성"""
        try:
            backup_path = Path(backup_dir)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = self._get_timestamp().replace(":", "-").replace(".", "-")
            backup_file = backup_path / f"pipeline_config_backup_{timestamp}.json"
            
            return self.export_config(str(backup_file), include_system_info=True)
            
        except Exception as e:
            logger.error(f"❌ 설정 백업 실패: {e}")
            return False
    
    def __repr__(self):
        return (f"PipelineConfig(device={self.device}, quality={self.quality_level}, "
                f"memory={self.memory_gb}GB, m3_max={self.is_m3_max}, "
                f"optimization={self.optimization_enabled}, constructor='optimal', "
                f"steps={len(self.config.get('steps', {}))}, "
                f"experimental_features={len([k for k, v in self.experimental_features.items() if v])})")


# ===============================================================
# ✅ 최적 생성자 패턴 - 전역 파이프라인 설정 팩토리 함수들
# ===============================================================

@lru_cache()
def get_pipeline_config(
    quality_level: str = "high",
    device: Optional[str] = None,    # 🔥 자동 감지
    **kwargs  # 🚀 확장성
) -> PipelineConfig:
    """✅ 최적 생성자 패턴 - 파이프라인 설정 인스턴스 반환 (캐시됨)"""
    return PipelineConfig(
        device=device,
        quality_level=quality_level,
        **kwargs
    )

@lru_cache()
def get_step_configs() -> Dict[str, Dict[str, Any]]:
    """모든 단계 설정 반환 (캐시됨)"""
    config = get_pipeline_config()
    return config.config["steps"]

@lru_cache()
def get_model_paths() -> Dict[str, str]:
    """모든 모델 경로 반환 (캐시됨)"""
    config = get_pipeline_config()
    return {
        model_name: config.get_model_path(model_name)
        for model_name in config.config["model_paths"]["checkpoints"].keys()
    }

def create_custom_config(
    quality_level: str = "high",
    device: Optional[str] = None,      # 🔥 자동 감지
    custom_settings: Optional[Dict[str, Any]] = None,
    **kwargs  # 🚀 확장성
) -> PipelineConfig:
    """✅ 최적 생성자 패턴 - 커스텀 파이프라인 설정 생성"""
    
    # 커스텀 설정을 kwargs에 병합
    if custom_settings:
        kwargs.update(custom_settings)
    
    config = PipelineConfig(
        device=device,
        quality_level=quality_level,
        **kwargs
    )
    
    return config

def create_optimized_config(
    device: Optional[str] = None,
    optimization_level: str = "balanced",  # conservative, balanced, aggressive
    **kwargs
) -> PipelineConfig:
    """최적화 레벨에 따른 설정 생성"""
    
    optimization_presets = {
        "conservative": {
            "optimization_enabled": False,
            "enable_caching": False,
            "batch_processing": False,
            "memory_optimization": False,
            "quality_level": "medium"
        },
        "balanced": {
            "optimization_enabled": True,
            "enable_caching": True,
            "batch_processing": True,
            "memory_optimization": True,
            "quality_level": "high"
        },
        "aggressive": {
            "optimization_enabled": True,
            "enable_caching": True,
            "batch_processing": True,
            "dynamic_batching": True,
            "memory_optimization": True,
            "super_resolution_enabled": True,
            "face_enhancement_enabled": True,
            "physics_simulation_enabled": True,
            "quality_level": "ultra"
        }
    }
    
    preset = optimization_presets.get(optimization_level, optimization_presets["balanced"])
    preset.update(kwargs)
    
    return PipelineConfig(device=device, **preset)

# ===============================================================
# ✅ 최적 생성자 패턴 - 하위 호환성 보장 함수들
# ===============================================================

def create_optimal_pipeline_config(
    device: Optional[str] = None,      # 🔥 자동 감지
    config: Optional[Dict[str, Any]] = None,
    **kwargs  # 🚀 확장성
) -> PipelineConfig:
    """✅ 최적 생성자 패턴 - 새로운 최적 방식"""
    return PipelineConfig(
        device=device,
        config=config,
        **kwargs
    )

def create_legacy_pipeline_config(
    config_path: Optional[str] = None, 
    quality_level: str = "high"
) -> PipelineConfig:
    """기존 방식 호환 (최적 생성자 패턴으로 내부 처리)"""
    return PipelineConfig(
        config_path=config_path,
        quality_level=quality_level
    )

def create_advanced_pipeline_config(
    device: Optional[str] = None,
    quality_level: str = "high",
    experimental_features: Optional[Dict[str, bool]] = None,
    custom_model_paths: Optional[Dict[str, str]] = None,
    custom_step_configs: Optional[Dict[str, Dict]] = None,
    **kwargs
) -> PipelineConfig:
    """고급 파이프라인 설정 생성"""
    
    return PipelineConfig(
        device=device,
        quality_level=quality_level,
        experimental_features=experimental_features or {},
        custom_model_paths=custom_model_paths or {},
        custom_step_configs=custom_step_configs or {},
        **kwargs
    )

# ===============================================================
# 환경별 설정 함수들 - 최적 생성자 패턴 (완전 개선)
# ===============================================================

def configure_for_development(**kwargs):
    """개발 환경 설정 - 최적 생성자 패턴"""
    dev_config = {
        "quality_level": "medium",
        "optimization_enabled": False,
        "enable_caching": False,
        "enable_intermediate_saving": True,
        "debug_mode": True,
        "save_debug_info": True,
        "profiling_enabled": True,
        "timeout_seconds": 120,
        "max_concurrent_requests": 2,
        **kwargs
    }
    
    config = create_optimal_pipeline_config(**dev_config)
    logger.info("🔧 개발 환경 설정 적용 (최적 생성자 패턴)")
    return config

def configure_for_production(**kwargs):
    """프로덕션 환경 설정 - 최적 생성자 패턴"""
    prod_config = {
        "quality_level": "high",
        "optimization_enabled": True,
        "enable_caching": True,
        "memory_optimization": True,
        "debug_mode": False,
        "save_debug_info": False,
        "profiling_enabled": False,
        "timeout_seconds": 300,
        "max_concurrent_requests": 8,
        "auto_retry": True,
        "max_retries": 2,
        **kwargs
    }
    
    config = create_optimal_pipeline_config(**prod_config)
    logger.info("🔧 프로덕션 환경 설정 적용 (최적 생성자 패턴)")
    return config

def configure_for_testing(**kwargs):
    """테스트 환경 설정 - 최적 생성자 패턴"""
    test_config = {
        "quality_level": "low",
        "max_concurrent_requests": 1,
        "timeout_seconds": 60,
        "optimization_enabled": False,
        "enable_caching": False,
        "batch_processing": False,
        "debug_mode": True,
        "save_debug_info": True,
        **kwargs
    }
    
    config = create_optimal_pipeline_config(**test_config)
    logger.info("🔧 테스트 환경 설정 적용 (최적 생성자 패턴)")
    return config

def configure_for_m3_max(**kwargs):
    """✅ M3 Max 최적화 설정 - 최적 생성자 패턴"""
    m3_config = {
        "device": "mps",
        "quality_level": "high",
        "memory_gb": 128.0,
        "is_m3_max": True,
        "optimization_enabled": True,
        "enable_caching": True,
        "memory_optimization": True,
        "batch_processing": True,
        "super_resolution_enabled": True,
        "face_enhancement_enabled": True,
        "physics_simulation_enabled": True,
        "advanced_post_processing": True,
        "max_concurrent_requests": 6,
        "timeout_seconds": 300,
        **kwargs
    }
    
    config = create_optimal_pipeline_config(**m3_config)
    logger.info("🔧 M3 Max 최적화 설정 적용 (최적 생성자 패턴)")
    return config

def configure_for_low_memory(**kwargs):
    """저메모리 환경 설정"""
    low_mem_config = {
        "quality_level": "low",
        "memory_gb": 8.0,
        "optimization_enabled": False,
        "enable_caching": False,
        "batch_processing": False,
        "super_resolution_enabled": False,
        "face_enhancement_enabled": False,
        "physics_simulation_enabled": False,
        "max_concurrent_requests": 1,
        "timeout_seconds": 180,
        **kwargs
    }
    
    config = create_optimal_pipeline_config(**low_mem_config)
    logger.info("🔧 저메모리 환경 설정 적용")
    return config

def configure_for_high_performance(**kwargs):
    """고성능 환경 설정"""
    high_perf_config = {
        "quality_level": "ultra",
        "optimization_enabled": True,
        "enable_caching": True,
        "batch_processing": True,
        "dynamic_batching": True,
        "memory_optimization": True,
        "super_resolution_enabled": True,
        "face_enhancement_enabled": True,
        "physics_simulation_enabled": True,
        "advanced_post_processing": True,
        "max_concurrent_requests": 12,
        "timeout_seconds": 600,
        "experimental_features": {
            "advanced_physics": True,
            "neural_style_transfer": True,
            "advanced_lighting": True
        },
        **kwargs
    }
    
    config = create_optimal_pipeline_config(**high_perf_config)
    logger.info("🔧 고성능 환경 설정 적용")
    return config

# ===============================================================
# 설정 관리 유틸리티 함수들
# ===============================================================

def compare_configs(config1: PipelineConfig, config2: PipelineConfig) -> Dict[str, Any]:
    """두 설정 비교"""
    def deep_diff(dict1, dict2, path=""):
        """딕셔너리 깊은 비교"""
        differences = []
        
        all_keys = set(dict1.keys()) | set(dict2.keys())
        for key in all_keys:
            current_path = f"{path}.{key}" if path else key
            
            if key not in dict1:
                differences.append(f"+ {current_path}: {dict2[key]}")
            elif key not in dict2:
                differences.append(f"- {current_path}: {dict1[key]}")
            elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                differences.extend(deep_diff(dict1[key], dict2[key], current_path))
            elif dict1[key] != dict2[key]:
                differences.append(f"~ {current_path}: {dict1[key]} -> {dict2[key]}")
        
        return differences
    
    return {
        "config1_name": f"{config1.class_name}({config1.quality_level})",
        "config2_name": f"{config2.class_name}({config2.quality_level})",
        "differences": deep_diff(config1.config, config2.config),
        "system_differences": deep_diff(config1.get_system_config(), config2.get_system_config())
    }

def merge_configs(base_config: PipelineConfig, override_config: PipelineConfig) -> PipelineConfig:
    """두 설정 병합"""
    merged_config = PipelineConfig(
        device=override_config.device,
        quality_level=override_config.quality_level,
        optimization_enabled=override_config.optimization_enabled
    )
    
    # 기본 설정으로 시작
    merged_config.config = base_config.config.copy()
    
    # 오버라이드 설정 병합
    merged_config._deep_merge(merged_config.config, override_config.config)
    
    logger.info(f"🔀 설정 병합 완료: {base_config.quality_level} + {override_config.quality_level}")
    return merged_config

def create_config_profile(profile_name: str, **kwargs) -> PipelineConfig:
    """프로필 기반 설정 생성"""
    profiles = {
        "ultra_quality": {
            "quality_level": "ultra",
            "optimization_enabled": True,
            "super_resolution_enabled": True,
            "face_enhancement_enabled": True,
            "physics_simulation_enabled": True,
            "advanced_post_processing": True
        },
        "speed_optimized": {
            "quality_level": "low",
            "optimization_enabled": True,
            "enable_caching": True,
            "batch_processing": True,
            "timeout_seconds": 60
        },
        "memory_efficient": {
            "quality_level": "medium",
            "memory_optimization": True,
            "batch_processing": False,
            "super_resolution_enabled": False,
            "face_enhancement_enabled": False
        },
        "experimental": {
            "quality_level": "high",
            "experimental_features": {
                "neural_style_transfer": True,
                "advanced_physics": True,
                "multi_person_support": True,
                "3d_pose_estimation": True
            }
        }
    }
    
    profile_config = profiles.get(profile_name, {})
    profile_config.update(kwargs)
    
    config = create_optimal_pipeline_config(**profile_config)
    logger.info(f"👤 프로필 설정 생성: {profile_name}")
    return config

# ===============================================================
# 초기화 및 검증 (최적 생성자 패턴)
# ===============================================================

# 기본 설정 생성 (자동 감지)
try:
    _default_config = get_pipeline_config()
    _validation_result = _default_config.validate_config()

    if not _validation_result["valid"]:
        for error in _validation_result["errors"]:
            logger.error(f"❌ 설정 오류: {error}")
        
        # 경고는 로깅만
        for warning in _validation_result["warnings"]:
            logger.warning(f"⚠️ 설정 경고: {warning}")
    else:
        logger.info("✅ 기본 파이프라인 설정 검증 완료")

    logger.info(f"🔧 최적 생성자 패턴 파이프라인 설정 초기화 완료 - 디바이스: {DEVICE}")

    # 시스템 정보 로깅
    _system_info = _default_config.get_system_info()
    logger.info(f"💻 시스템: {_system_info['device']} ({_system_info['quality_level']}) - 최적 생성자 패턴")
    logger.info(f"🎯 메모리: {_system_info['memory_gb']}GB, M3 Max: {'✅' if _system_info['is_m3_max'] else '❌'}")
    logger.info(f"🔧 파이프라인 단계: {_system_info['total_steps']}개")
    logger.info(f"🧪 실험적 기능: {_system_info['experimental_features_count']}개 활성화")

except Exception as e:
    logger.error(f"❌ 기본 설정 초기화 실패: {e}")
    # 최소한의 폴백 설정
    _default_config = None

# ===============================================================
# 최적 생성자 패턴 호환성 검증 (완전 개선)
# ===============================================================

def validate_optimal_constructor_compatibility() -> Dict[str, bool]:
    """최적 생성자 패턴 호환성 검증 (완전 개선 버전)"""
    try:
        # 테스트 설정 생성 - 최적 생성자 패턴
        test_config = create_optimal_pipeline_config(
            device="cpu",  # 명시적 설정
            quality_level="medium",
            device_type="test",
            memory_gb=16.0,
            is_m3_max=False,
            optimization_enabled=True,
            enable_caching=True,
            batch_processing=True,
            experimental_features={"test_feature": True},
            custom_model_paths={"test_model": "/test/path"},
            custom_step_configs={"test_step": {"test_param": "test_value"}},
            custom_param="test_value"  # 확장 파라미터
        )
        
        # 필수 속성 확인
        required_attrs = [
            'device', 'device_type', 'memory_gb', 'is_m3_max', 
            'optimization_enabled', 'quality_level', 'enable_caching',
            'batch_processing', 'experimental_features', 'custom_model_paths'
        ]
        attr_check = {attr: hasattr(test_config, attr) for attr in required_attrs}
        
        # 필수 메서드 확인
        required_methods = [
            'get_step_config', 'get_model_path', 'get_system_config',
            'update_quality_level', 'update_device', 'validate_config',
            'toggle_optimization', 'toggle_caching', 'enable_experimental_feature',
            'export_config', 'get_performance_summary'
        ]
        method_check = {method: hasattr(test_config, method) and callable(getattr(test_config, method)) 
                       for method in required_methods}
        
        # 확장 파라미터 확인
        extension_check = test_config.config.get('custom_param') == 'test_value'
        
        # 실험적 기능 확인
        experimental_check = test_config.experimental_features.get('test_feature') == True
        
        # 커스텀 모델 경로 확인
        custom_model_check = test_config.custom_model_paths.get('test_model') == '/test/path'
        
        # 설정 검증 확인
        validation_result = test_config.validate_config()
        validation_check = isinstance(validation_result, dict) and 'valid' in validation_result
        
        # 동적 변경 테스트
        dynamic_test_passed = True
        try:
            test_config.update_quality_level("low")
            test_config.toggle_optimization(False)
            test_config.toggle_caching(False)
            test_config.enable_experimental_feature("new_feature", True)
        except Exception:
            dynamic_test_passed = False
        
        return {
            'attributes': all(attr_check.values()),
            'methods': all(method_check.values()),
            'extensions': extension_check,
            'experimental_features': experimental_check,
            'custom_models': custom_model_check,
            'validation': validation_check,
            'dynamic_changes': dynamic_test_passed,
            'attr_details': attr_check,
            'method_details': method_check,
            'overall_compatible': (
                all(attr_check.values()) and 
                all(method_check.values()) and 
                extension_check and
                experimental_check and
                custom_model_check and
                validation_check and
                dynamic_test_passed
            ),
            'constructor_pattern': 'optimal',
            'test_config_info': test_config.get_system_info()
        }
        
    except Exception as e:
        logger.error(f"최적 생성자 패턴 호환성 검증 실패: {e}")
        return {
            'overall_compatible': False, 
            'error': str(e), 
            'constructor_pattern': 'optimal'
        }

# 모듈 로드 시 호환성 검증
_compatibility_result = validate_optimal_constructor_compatibility()
if _compatibility_result.get('overall_compatible'):
    logger.info("✅ 최적 생성자 패턴 호환성 검증 완료 (완전 개선 버전)")
    if _compatibility_result.get('test_config_info'):
        test_info = _compatibility_result['test_config_info']
        logger.info(f"🧪 테스트 설정: {test_info.get('pipeline_version', 'unknown')} "
                   f"({test_info.get('total_steps', 0)}단계)")
else:
    logger.warning(f"⚠️ 호환성 문제 발견: {_compatibility_result}")

# ===============================================================
# 설정 템플릿 및 예제
# ===============================================================

def get_config_templates() -> Dict[str, Dict[str, Any]]:
    """설정 템플릿 반환"""
    return {
        "minimal": {
            "quality_level": "low",
            "optimization_enabled": False,
            "enable_caching": False
        },
        "standard": {
            "quality_level": "high",
            "optimization_enabled": True,
            "enable_caching": True,
            "batch_processing": True
        },
        "professional": {
            "quality_level": "ultra",
            "optimization_enabled": True,
            "enable_caching": True,
            "batch_processing": True,
            "super_resolution_enabled": True,
            "face_enhancement_enabled": True,
            "physics_simulation_enabled": True,
            "advanced_post_processing": True
        },
        "m3_max": {
            "device": "mps",
            "quality_level": "high",
            "optimization_enabled": True,
            "memory_gb": 128.0,
            "is_m3_max": True,
            "enable_caching": True,
            "batch_processing": True,
            "super_resolution_enabled": True
        }
    }

def create_config_from_template(template_name: str, **overrides) -> PipelineConfig:
    """템플릿에서 설정 생성"""
    templates = get_config_templates()
    
    if template_name not in templates:
        available = ", ".join(templates.keys())
        raise ValueError(f"Unknown template: {template_name}. Available: {available}")
    
    template_config = templates[template_name].copy()
    template_config.update(overrides)
    
    config = create_optimal_pipeline_config(**template_config)
    logger.info(f"📋 템플릿 설정 생성: {template_name}")
    return config

# 모듈 레벨 exports (완전 개선)
__all__ = [
    # 핵심 클래스들
    "OptimalConstructorBase",         # 최적 생성자 베이스
    "PipelineConfig",                 # 메인 설정 클래스
    "SystemConfig",                   # 시스템 설정 데이터 클래스
    
    # 팩토리 함수들 (기본)
    "get_pipeline_config", 
    "get_step_configs",
    "get_model_paths",
    "create_custom_config",
    "create_optimized_config",
    
    # 생성자 패턴 함수들
    "create_optimal_pipeline_config", # 새로운 최적 방식
    "create_legacy_pipeline_config",  # 기존 호환성
    "create_advanced_pipeline_config", # 고급 설정
    
    # 환경별 설정 함수들
    "configure_for_development",
    "configure_for_production", 
    "configure_for_testing",
    "configure_for_m3_max",
    "configure_for_low_memory",
    "configure_for_high_performance",
    
    # 설정 관리 유틸리티
    "compare_configs",
    "merge_configs",
    "create_config_profile",
    "get_config_templates",
    "create_config_from_template",
    
    # 검증 및 호환성
    "validate_optimal_constructor_compatibility"
]

logger.info("🎯 최적 생성자 패턴 PipelineConfig 초기화 완료 (완전 개선 버전) - 모든 기능 활성화")
logger.info(f"📚 내보내기 함수: {len(__all__)}개")
logger.info(f"🔧 GPU Config 사용 가능: {'✅' if GPU_CONFIG_AVAILABLE else '❌'}")
logger.info(f"🐍 PyTorch 사용 가능: {'✅' if TORCH_AVAILABLE else '❌'}")# backend/app/core/pipeline_config.py
"""
🎯 MyCloset AI Backend 최적화된 파이프라인 설정 관리
- 최적 생성자 패턴 (Optimal Constructor Pattern) 적용
- 8단계 가상 피팅 파이프라인의 전체 설정 관리
- 기존 app/ 구조와 완전 호환
- M3 Max 최적화 및 자동 감지 포함
- 무제한 확장성과 하위 호환성 보장

✅ 최적 생성자 패턴 장점:
- 지능적 디바이스 자동 감지 (None으로 설정 시)
- **kwargs로 무제한 확장성
- 표준 시스템 파라미터 통일
- 하위 호환성 100% 보장
- 타입 안전성 및 검증 기능
"""

import os
import json
import logging
import platform
import subprocess
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from functools import lru_cache
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# PyTorch 안전 import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# 기존 gpu_config import (안전한 import)
try:
    from .gpu_config import gpu_config, DEVICE, DEVICE_INFO
    GPU_CONFIG_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("gpu_config import 실패 - 기본값 사용")
    
    # 기본값 설정
    DEVICE = "mps" if TORCH_AVAILABLE and torch.backends.mps.is_available() else (
        "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    )
    DEVICE_INFO = {"device": DEVICE, "available": True}
    
    class DummyGPUConfig:
        def __init__(self):
            self.device = DEVICE
            self.device_type = "auto"
    
    gpu_config = DummyGPUConfig()
    GPU_CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)

# ===============================================================
# 🎯 최적 생성자 베이스 클래스 (통합 개선 버전)
# ===============================================================

@dataclass
class SystemConfig:
    """시스템 설정 데이터 클래스"""
    device: str = "auto"
    device_type: str = "auto"
    memory_gb: float = 16.0
    cpu_cores: int = 4
    is_m3_max: bool = False
    platform: str = "unknown"
    architecture: str = "unknown"
    torch_available: bool = False
    mps_available: bool = False
    cuda_available: bool = False

class OptimalConstructorBase(ABC):
    """
    🎯 최적화된 생성자 패턴 베이스 클래스
    - 단순함 + 편의성 + 확장성 + 일관성 통합
    - 지능적 자동 감지와 완전한 확장성
    """

    def __init__(
        self,
        device: Optional[str] = None,           # 🔥 핵심: None으로 자동 감지
        config: Optional[Dict[str, Any]] = None,
        **kwargs  # 🚀 확장성: 무제한 추가 파라미터
    ):
        """
        ✅ 최적 생성자 - 모든 장점 결합

        Args:
            device: 사용할 디바이스 (None=자동감지, 'cpu', 'cuda', 'mps')
            config: 설정 딕셔너리
            **kwargs: 확장 파라미터들
                - device_type: str = "auto"
                - memory_gb: float = 16.0
                - is_m3_max: bool = False (자동 감지)
                - optimization_enabled: bool = True
                - quality_level: str = "balanced"
                - enable_caching: bool = True
                - enable_parallel: bool = True
                - memory_optimization: bool = True
                - max_concurrent_requests: int = 4
                - timeout_seconds: int = 300
                - debug_mode: bool = False
                - 기타 모든 확장 파라미터들...
        """
        # 1. 🔧 시스템 설정 초기화
        self.system_config = self._detect_system_config(device, **kwargs)
        
        # 2. 📋 기본 속성 설정
        self.config = config or {}
        self.class_name = self.__class__.__name__
        self.logger = logging.getLogger(f"pipeline.{self.class_name}")
        
        # 3. 🎯 표준 파라미터 추출 (최적 생성자 패턴)
        self.device = self.system_config.device
        self.device_type = kwargs.get('device_type', self.system_config.device_type)
        self.memory_gb = kwargs.get('memory_gb', self.system_config.memory_gb)
        self.is_m3_max = kwargs.get('is_m3_max', self.system_config.is_m3_max)
        self.optimization_enabled = kwargs.get('optimization_enabled', True)
        self.quality_level = kwargs.get('quality_level', 'balanced')
        
        # 4. 🔧 추가 설정 파라미터들
        self.enable_caching = kwargs.get('enable_caching', True)
        self.enable_parallel = kwargs.get('enable_parallel', True)
        self.memory_optimization = kwargs.get('memory_optimization', True)
        self.max_concurrent_requests = kwargs.get('max_concurrent_requests', 4)
        self.timeout_seconds = kwargs.get('timeout_seconds', 300)
        self.debug_mode = kwargs.get('debug_mode', False)
        
        # 5. ⚙️ 클래스별 특화 파라미터를 config에 병합
        self._merge_class_specific_config(kwargs)
        
        # 6. ✅ 상태 초기화
        self.is_initialized = False
        self.creation_timestamp = self._get_timestamp()
        
        self.logger.info(f"🎯 {self.class_name} 최적 생성자 초기화 - "
                        f"디바이스: {self.device}, 품질: {self.quality_level}")

    def _detect_system_config(self, preferred_device: Optional[str], **kwargs) -> SystemConfig:
        """💡 지능적 시스템 설정 감지"""
        try:
            # 기본 시스템 정보
            platform_name = platform.system()
            architecture = platform.machine()
            
            # 메모리 정보
            try:
                import psutil
                memory_bytes = psutil.virtual_memory().total
                memory_gb = round(memory_bytes / (1024**3), 1)
                cpu_cores = psutil.cpu_count(logical=False) or 4
            except ImportError:
                memory_gb = kwargs.get('memory_gb', 16.0)
                cpu_cores = kwargs.get('cpu_cores', 4)
            
            # PyTorch 지원 확인
            torch_available = TORCH_AVAILABLE
            mps_available = torch_available and torch.backends.mps.is_available()
            cuda_available = torch_available and torch.cuda.is_available()
            
            # M3 Max 감지
            is_m3_max = self._detect_m3_max(platform_name, architecture, memory_gb, mps_available)
            
            # 최적 디바이스 결정
            device = self._determine_optimal_device(preferred_device, is_m3_max, mps_available, cuda_available)
            
            return SystemConfig(
                device=device,
                device_type=device,
                memory_gb=memory_gb,
                cpu_cores=cpu_cores,
                is_m3_max=is_m3_max,
                platform=platform_name,
                architecture=architecture,
                torch_available=torch_available,
                mps_available=mps_available,
                cuda_available=cuda_available
            )
            
        except Exception as e:
            self.logger.warning(f"⚠️ 시스템 감지 실패: {e}")
            return SystemConfig()  # 기본값 사용

    def _detect_m3_max(self, platform_name: str, architecture: str, memory_gb: float, mps_available: bool) -> bool:
        """🍎 M3 Max 정밀 감지"""
        try:
            # macOS ARM64 확인
            if platform_name != 'Darwin' or architecture != 'arm64':
                return False
            
            # 메모리 크기 확인 (M3 Max는 보통 64GB 이상)
            if memory_gb < 64:
                return False
            
            # MPS 백엔드 지원 확인
            if not mps_available:
                return False
            
            # 추가 검증: CPU 브랜드 확인
            try:
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'], 
                    capture_output=True, text=True, timeout=5
                )
                if 'M3' in result.stdout:
                    return True
            except Exception:
                pass
            
            # 메모리 크기로만 판단 (128GB 모델)
            return memory_gb >= 118  # 실제 사용 가능 메모리
            
        except Exception as e:
            self.logger.debug(f"M3 Max 감지 실패: {e}")
            return False

    def _determine_optimal_device(self, preferred: Optional[str], is_m3_max: bool, 
                                mps_available: bool, cuda_available: bool) -> str:
        """💡 최적 디바이스 결정"""
        if preferred and preferred != "auto":
            return preferred
        
        # M3 Max 우선
        if is_m3_max and mps_available:
            return "mps"
        
        # CUDA 다음
        if cuda_available:
            return "cuda"
        
        # 일반 MPS
        if mps_available:
            return "mps"
        
        # CPU 폴백
        return "cpu"

    def _merge_class_specific_config(self, kwargs: Dict[str, Any]):
        """⚙️ 클래스별 특화 설정 병합"""
        # 표준 시스템 파라미터 제외하고 모든 kwargs를 config에 병합
        standard_params = {
            'device_type', 'memory_gb', 'is_m3_max', 'optimization_enabled', 
            'quality_level', 'enable_caching', 'enable_parallel', 'memory_optimization',
            'max_concurrent_requests', 'timeout_seconds', 'debug_mode', 'cpu_cores'
        }

        for key, value in kwargs.items():
            if key not in standard_params:
                self.config[key] = value

    def _get_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        from datetime import datetime
        return datetime.now().isoformat()

    def get_system_info(self) -> Dict[str, Any]:
        """🔍 시스템 정보 반환"""
        return {
            "class_name": self.class_name,
            "device": self.device,
            "device_type": self.device_type,
            "memory_gb": self.memory_gb,
            "cpu_cores": self.system_config.cpu_cores,
            "is_m3_max": self.is_m3_max,
            "platform": self.system_config.platform,
            "architecture": self.system_config.architecture,
            "optimization_enabled": self.optimization_enabled,
            "quality_level": self.quality_level,
            "torch_available": self.system_config.torch_available,
            "mps_available": self.system_config.mps_available,
            "cuda_available": self.system_config.cuda_available,
            "initialized": self.is_initialized,
            "config_keys": list(self.config.keys()),
            "constructor_pattern": "optimal",
            "creation_timestamp": self.creation_timestamp
        }

    def get_performance_config(self) -> Dict[str, Any]:
        """성능 설정 정보 반환"""
        return {
            "enable_caching": self.enable_caching,
            "enable_parallel": self.enable_parallel,
            "memory_optimization": self.memory_optimization,
            "max_concurrent_requests": self.max_concurrent_requests,
            "timeout_seconds": self.timeout_seconds,
            "optimization_enabled": self.optimization_enabled
        }

# ===============================================================
# 🎯 최적화된 PipelineConfig 클래스 (완전 개선 버전)
# ===============================================================

class PipelineConfig(OptimalConstructorBase):
    """
    🎯 최적 생성자 패턴이 적용된 8단계 AI 파이프라인 설정 관리 (완전 개선 버전)
    
    ✅ 최적 생성자 패턴 장점:
    - 지능적 디바이스 자동 감지 (None으로 설정 시)
    - **kwargs로 무제한 확장성
    - 표준 시스템 파라미터 통일
    - 기존 코드와 100% 하위 호환성
    - 타입 안전성 및 검증 기능
    
    완전한 8단계 가상 피팅 파이프라인:
    1. Human Parsing (인체 파싱)
    2. Pose Estimation (포즈 추정)
    3. Cloth Segmentation (의류 세그멘테이션)
    4. Geometric Matching (기하학적 매칭)
    5. Cloth Warping (옷 워핑)
    6. Virtual Fitting (가상 피팅)
    7. Post Processing (후처리)
    8. Quality Assessment (품질 평가)
    """
    
    def __init__(
        self, 
        device: Optional[str] = None,                    # 🔥 자동 감지로 변경
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,               # 기존 호환성
        quality_level: str = "high",                     # 기존 호환성
        **kwargs  # 🚀 무제한 확장성
    ):
        """
        ✅ 최적 생성자 - PipelineConfig 완전 개선 버전
        
        Args:
            device: 사용할 디바이스 (None=자동감지, 'cpu', 'cuda', 'mps')
            config: 기본 설정 딕셔너리
            config_path: 설정 파일 경로 (기존 호환성)
            quality_level: 품질 레벨 (low, medium, high, ultra)
            **kwargs: 확장 파라미터들
                # 시스템 파라미터
                - device_type: str = "auto"
                - memory_gb: float = 16.0
                - is_m3_max: bool = False (자동 감지)
                - optimization_enabled: bool = True
                
                # 성능 파라미터
                - enable_caching: bool = True
                - enable_parallel: bool = True
                - memory_optimization: bool = True
                - max_concurrent_requests: int = 4
                - timeout_seconds: int = 300
                
                # 처리 파라미터
                - enable_intermediate_saving: bool = False
                - auto_retry: bool = True
                - max_retries: int = 3
                - batch_processing: bool = True
                - dynamic_batching: bool = True
                
                # 품질 파라미터
                - super_resolution_enabled: bool = True
                - face_enhancement_enabled: bool = True
                - physics_simulation_enabled: bool = True
                - advanced_post_processing: bool = True
                
                # 개발자 파라미터
                - debug_mode: bool = False
                - save_debug_info: bool = False
                - benchmark_mode: bool = False
                - profiling_enabled: bool = False
                
                # 사용자 정의 파라미터
                - custom_model_paths: Dict[str, str] = None
                - custom_step_configs: Dict[str, Dict] = None
                - experimental_features: Dict[str, bool] = None
        """
        
        # kwargs에서 품질 레벨 덮어쓰기 확인
        if 'quality_level_override' in kwargs:
            quality_level = kwargs.pop('quality_level_override')
        
        # 부모 클래스 초기화 (최적 생성자 패턴)
        super().__init__(
            device=device,
            config=config,
            quality_level=quality_level,
            **kwargs
        )
        
        # PipelineConfig 특화 속성들
        self.quality_level = quality_level
        self.config_path = config_path or kwargs.get('config_path')
        self.device_info = DEVICE_INFO if GPU_CONFIG_AVAILABLE else {"device": self.device}
        
        # 확장 파라미터들 추가 추출
        self.enable_intermediate_saving = kwargs.get('enable_intermediate_saving', False)
        self.auto_retry = kwargs.get('auto_retry', True)
        self.max_retries = kwargs.get('max_retries', 3)
        self.batch_processing = kwargs.get('batch_processing', True)
        self.dynamic_batching = kwargs.get('dynamic_batching', True)
        
        self.super_resolution_enabled = kwargs.get('super_resolution_enabled', True)
        self.face_enhancement_enabled = kwargs.get('face_enhancement_enabled', True)
        self.physics_simulation_enabled = kwargs.get('physics_simulation_enabled', True)
        self.advanced_post_processing = kwargs.get('advanced_post_processing', True)
        
        self.save_debug_info = kwargs.get('save_debug_info', False)
        self.benchmark_mode = kwargs.get('benchmark_mode', False)
        self.profiling_enabled = kwargs.get('profiling_enabled', False)
        
        self.custom_model_paths = kwargs.get('custom_model_paths', {})
        self.custom_step_configs = kwargs.get('custom_step_configs', {})
        self.experimental_features = kwargs.get('experimental_features', {})
        
        # 기본 설정 로드 (최적 생성자 패턴과 통합)
        self.config = self._load_default_config_optimal()
        
        # 외부 설정 파일 로드 (있는 경우)
        if self.config_path and os.path.exists(self.config_path):
            self._load_external_config(self.config_path)
        
        # 환경변수 기반 오버라이드
        self._apply_environment_overrides()
        
        # 디바이스별 최적화 적용
        self._apply_device_optimizations()
        
        # 품질 레벨 프리셋 적용
        self._apply_quality_preset(quality_level)
        
        # 사용자 정의 설정 적용
        self._apply_custom_configurations()
        
        # 초기화 완료
        self.is_initialized = True
        
        logger.info(f"🔧 최적 생성자 패턴 파이프라인 설정 완료")
        logger.info(f"   - 품질: {quality_level}, 디바이스: {self.device}")
        logger.info(f"   - 시스템: {self.device_type}, 메모리: {self.memory_gb}GB")
        logger.info(f"   - M3 Max: {'✅' if self.is_m3_max else '❌'}")
        logger.info(f"   - 확장 설정: {len(self.config)} 항목")

    def _load_default_config_optimal(self) -> Dict[str, Any]:
        """✅ 최적 생성자 패턴과 통합된 완전한 기본 설정 로드"""
        
        return {
            # =======================================
            # 🎯 전역 파이프라인 설정
            # =======================================
            "pipeline": {
                "name": "mycloset_virtual_fitting",
                "version": "4.1.0-optimal-enhanced",
                "constructor_pattern": "optimal",
                "quality_level": self.quality_level,
                "processing_mode": "complete",  # fast, balanced, complete
                "enable_optimization": self.optimization_enabled,
                "enable_caching": self.enable_caching,
                "enable_parallel": self.enable_parallel,
                "memory_optimization": self.memory_optimization,
                "max_concurrent_requests": self.max_concurrent_requests,
                "timeout_seconds": self.timeout_seconds,
                "enable_intermediate_saving": self.enable_intermediate_saving,
                "auto_retry": self.auto_retry,
                "max_retries": self.max_retries,
                "batch_processing": self.batch_processing,
                "dynamic_batching": self.dynamic_batching,
                "debug_mode": self.debug_mode,
                "save_debug_info": self.save_debug_info,
                "benchmark_mode": self.benchmark_mode,
                "profiling_enabled": self.profiling_enabled
            },
            
            # =======================================
            # 💻 시스템 정보 (최적 생성자 패턴)
            # =======================================
            "system": {
                "device": self.device,
                "device_type": self.device_type,
                "memory_gb": self.memory_gb,
                "cpu_cores": self.system_config.cpu_cores,
                "is_m3_max": self.is_m3_max,
                "platform": self.system_config.platform,
                "architecture": self.system_config.architecture,
                "optimization_enabled": self.optimization_enabled,
                "torch_available": self.system_config.torch_available,
                "mps_available": self.system_config.mps_available,
                "cuda_available": self.system_config.cuda_available,
                "constructor_pattern": "optimal"
            },
            
            # =======================================
            # 🖼️ 이미지 처리 설정
            # =======================================
            "image": {
                "input_size": (512, 512),
                "output_size": (512, 512),
                "max_resolution": 1024,
                "supported_formats": ["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
                "quality": 95,
                "color_space": "RGB",
                "bit_depth": 8,
                "preprocessing": {
                    "normalize": True,
                    "resize_mode": "lanczos",
                    "center_crop": True,
                    "background_removal": True,
                    "noise_reduction": True,
                    "contrast_enhancement": True
                },
                "postprocessing": {
                    "color_correction": True,
                    "gamma_correction": True,
                    "sharpening": True,
                    "edge_enhancement": True
                }
            },
            
            # =======================================
            # 🔄 8단계 파이프라인 개별 설정
            # =======================================
            "steps": {
                # 1단계: 인체 파싱 (Human Parsing)
                "human_parsing": {
                    "model_name": "graphonomy",
                    "model_path": "app/ai_pipeline/models/ai_models/graphonomy",
                    "fallback_models": ["detectron2", "deeplabv3"],
                    "num_classes": 20,
                    "confidence_threshold": 0.7,
                    "input_size": (512, 512),
                    "batch_size": 1 if self.device == 'cpu' else 2,
                    "cache_enabled": self.enable_caching,
                    "use_coreml": self.is_m3_max,
                    "enable_quantization": self.optimization_enabled,
                    "model_precision": "float16" if self.optimization_enabled else "float32",
                    "preprocessing": {
                        "normalize": True,
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                        "resize_method": "bilinear",
                        "padding_mode": "reflect"
                    },
                    "postprocessing": {
                        "morphology_cleanup": True,
                        "smooth_edges": True,
                        "fill_holes": True,
                        "remove_noise": True,
                        "edge_refinement": True
                    },
                    "validation": {
                        "min_body_area": 0.1,
                        "max_fragmentation": 0.3,
                        "require_torso": True
                    }
                },
                
                # 2단계: 포즈 추정 (Pose Estimation)
                "pose_estimation": {
                    "model_name": "mediapipe",
                    "fallback_models": ["openpose", "hrnet", "alphapose"],
                    "model_complexity": 2,
                    "min_detection_confidence": 0.5,
                    "min_tracking_confidence": 0.5,
                    "static_image_mode": True,
                    "enable_segmentation": True,
                    "smooth_landmarks": True,
                    "keypoints_format": "openpose_18",
                    "use_gpu": self.device != 'cpu',
                    "enable_face_landmarks": True,
                    "enable_hand_landmarks": False,
                    "pose_validation": {
                        "min_keypoints": 8,
                        "visibility_threshold": 0.3,
                        "symmetry_check": True,
                        "anatomical_constraints": True
                    },
                    "filtering": {
                        "temporal_smoothing": True,
                        "outlier_removal": True,
                        "confidence_weighting": True
                    }
                },
                
                # 3단계: 의류 세그멘테이션 (Cloth Segmentation)
                "cloth_segmentation": {
                    "model_name": "u2net",
                    "model_path": "app/ai_pipeline/models/ai_models/u2net",
                    "fallback_models": ["rembg", "backgroundmattingv2", "modnet"],
                    "background_removal": True,
                    "edge_refinement": True,
                    "background_threshold": 0.5,
                    "trimap_generation": True,
                    "alpha_matting": True,
                    "multi_scale_processing": True,
                    "cloth_categories": {
                        "upper_body": ["shirt", "blouse", "jacket", "sweater", "hoodie"],
                        "lower_body": ["pants", "jeans", "skirt", "shorts"],
                        "full_body": ["dress", "jumpsuit", "overall"],
                        "accessories": ["hat", "scarf", "belt", "bag"]
                    },
                    "preprocessing": {
                        "contrast_enhancement": True,
                        "edge_detection": True,
                        "texture_analysis": True
                    },
                    "postprocessing": {
                        "morphology_enabled": True,
                        "gaussian_blur": True,
                        "edge_smoothing": True,
                        "noise_removal": True,
                        "hole_filling": True,
                        "boundary_refinement": True
                    },
                    "quality_assessment": {
                        "enable": True,
                        "min_quality": 0.6,
                        "auto_retry": self.auto_retry,
                        "quality_metrics": ["edge_sharpness", "mask_completeness", "boundary_smoothness"]
                    }
                },
                
                # 4단계: 기하학적 매칭 (Geometric Matching)
                "geometric_matching": {
                    "algorithm": "tps_hybrid",  # tps, affine, tps_hybrid, thin_plate_spline
                    "num_control_points": 20,
                    "regularization": 0.001,
                    "matching_method": "hungarian",
                    "tps_points": 25,
                    "matching_threshold": 0.8,
                    "use_advanced_matching": True,
                    "multi_scale_matching": True,
                    "iterative_refinement": True,
                    "keypoint_extraction": {
                        "method": "contour_based",
                        "num_points": 50,
                        "adaptive_sampling": True,
                        "curvature_based": True,
                        "corner_detection": True
                    },
                    "feature_matching": {
                        "descriptor": "sift",  # sift, surf, orb, brief
                        "matcher": "flann",    # brute_force, flann
                        "cross_check": True,
                        "ratio_test": 0.7
                    },
                    "validation": {
                        "min_matched_points": 4,
                        "outlier_threshold": 2.0,
                        "quality_threshold": 0.7,
                        "geometric_consistency": True
                    },
                    "optimization": {
                        "max_iterations": 100,
                        "convergence_threshold": 1e-6,
                        "adaptive_learning": True
                    }
                },
                
                # 5단계: 옷 워핑 (Cloth Warping)
                "cloth_warping": {
                    "physics_enabled": self.physics_simulation_enabled,
                    "fabric_simulation": True,
                    "deformation_strength": 0.8,
                    "wrinkle_simulation": True,
                    "gravity_simulation": False,
                    "wind_simulation": False,
                    "warping_method": "tps",  # tps, grid, mesh, physics
                    "optimization_level": "high",
                    "mesh_resolution": 64,
                    "simulation_steps": 50,
                    "convergence_threshold": 0.001,
                    "fabric_properties": {
                        "cotton": {
                            "stiffness": 0.6, 
                            "elasticity": 0.3, 
                            "thickness": 0.5,
                            "damping": 0.1,
                            "friction": 0.4
                        },
                        "denim": {
                            "stiffness": 0.9, 
                            "elasticity": 0.1, 
                            "thickness": 0.8,
                            "damping": 0.2,
                            "friction": 0.6
                        },
                        "silk": {
                            "stiffness": 0.2, 
                            "elasticity": 0.4, 
                            "thickness": 0.2,
                            "damping": 0.05,
                            "friction": 0.2
                        },
                        "wool": {
                            "stiffness": 0.7, 
                            "elasticity": 0.2, 
                            "thickness": 0.7,
                            "damping": 0.15,
                            "friction": 0.5
                        },
                        "polyester": {
                            "stiffness": 0.4, 
                            "elasticity": 0.6, 
                            "thickness": 0.3,
                            "damping": 0.08,
                            "friction": 0.3
                        }
                    },
                    "collision_detection": {
                        "enable": True,
                        "method": "continuous",
                        "resolution": "high"
                    },
                    "constraints": {
                        "anatomical": True,
                        "fabric_stretch": True,
                        "seam_preservation": True
                    }
                },
                
                # 6단계: 가상 피팅 (Virtual Fitting)
                "virtual_fitting": {
                    "model_name": "hr_viton",
                    "model_path": "app/ai_pipeline/models/ai_models/hr_viton",
                    "fallback_models": ["ootd", "viton_hd", "cagan"],
                    "guidance_scale": 7.5,
                    "num_inference_steps": 50,
                    "strength": 0.8,
                    "eta": 0.0,
                    "composition_method": "neural_blending",
                    "fallback_method": "traditional_blending",
                    "blending_method": "poisson",  # poisson, laplacian, multiband
                    "seamless_cloning": True,
                    "color_transfer": True,
                    "lighting_preservation": True,
                    "shadow_generation": True,
                    "reflection_handling": True,
                    "style_transfer": {
                        "enable": False,
                        "strength": 0.3,
                        "preserve_structure": True
                    },
                    "quality_enhancement": {
                        "color_matching": True,
                        "lighting_adjustment": True,
                        "texture_preservation": True,
                        "edge_smoothing": True,
                        "detail_enhancement": True,
                        "artifact_removal": True
                    },
                    "advanced_features": {
                        "multi_pose_support": True,
                        "dynamic_lighting": True,
                        "fabric_physics": self.physics_simulation_enabled,
                        "realistic_draping": True
                    }
                },
                
                # 7단계: 후처리 (Post Processing)
                "post_processing": {
                    "enable_super_resolution": self.super_resolution_enabled and self.optimization_enabled,
                    "enhance_faces": self.face_enhancement_enabled,
                    "color_correction": True,
                    "noise_reduction": True,
                    "detail_enhancement": self.advanced_post_processing,
                    "artifact_removal": True,
                    "super_resolution": {
                        "enabled": self.super_resolution_enabled and self.optimization_enabled,
                        "model": "real_esrgan",
                        "fallback_models": ["esrgan", "srcnn", "edsr"],
                        "scale_factor": 2,
                        "model_path": "app/ai_pipeline/models/ai_models/real_esrgan",
                        "tile_processing": True,
                        "overlap": 32,
                        "precision": "float16" if self.optimization_enabled else "float32"
                    },
                    "face_enhancement": {
                        "enabled": self.face_enhancement_enabled,
                        "model": "gfpgan",
                        "fallback_models": ["codeformer", "restoreformer"],
                        "strength": 0.8,
                        "model_path": "app/ai_pipeline/models/ai_models/gfpgan",
                        "detect_faces": True,
                        "preserve_identity": True,
                        "enhance_eyes": True,
                        "enhance_mouth": True
                    },
                    "color_correction": {
                        "enabled": True,
                        "method": "histogram_matching",
                        "strength": 0.6,
                        "preserve_skin_tone": True,
                        "white_balance": True,
                        "exposure_correction": True
                    },
                    "noise_reduction": {
                        "enabled": True,
                        "method": "bilateral_filter",
                        "strength": 0.7,
                        "preserve_edges": True,
                        "adaptive": True
                    },
                    "edge_enhancement": {
                        "enabled": self.advanced_post_processing,
                        "method": "unsharp_mask",
                        "strength": 0.5,
                        "radius": 1.0,
                        "threshold": 0.05
                    },
                    "detail_enhancement": {
                        "enabled": self.advanced_post_processing,
                        "texture_sharpening": True,
                        "clarity_boost": True,
                        "micro_contrast": True
                    },
                    "final_optimization": {
                        "compression_quality": 95,
                        "format_optimization": True,
                        "metadata_preservation": False
                    }
                },
                
                # 8단계: 품질 평가 (Quality Assessment)
                "quality_assessment": {
                    "enabled": True,
                    "comprehensive_analysis": True,
                    "generate_suggestions": True,
                    "save_report": self.save_debug_info,
                    "metrics": {
                        "perceptual": ["ssim", "lpips", "ms_ssim"],
                        "technical": ["psnr", "mse", "mae"],
                        "aesthetic": ["brisque", "niqe"],
                        "semantic": ["face_quality", "cloth_fit", "realism"]
                    },
                    "thresholds": {
                        "ssim_min": 0.7,
                        "lpips_max": 0.3,
                        "psnr_min": 20.0,
                        "brisque_max": 50.0,
                        "overall_min": 0.7
                    },
                    "detailed_analysis": {
                        "face_region": True,
                        "cloth_region": True,
                        "background_region": True,
                        "edge_quality": True,
                        "color_consistency": True,
                        "lighting_coherence": True
                    },
                    "benchmarking": {
                        "enabled": self.benchmark_mode,
                        "reference_dataset": None,
                        "save_results": self.benchmark_mode,
                        "performance_tracking": True
                    },
                    "auto_improvement": {
                        "enabled": True,
                        "retry_on_low_quality": self.auto_retry,
                        "parameter_adjustment": True,
                        "fallback_models": True
                    }
                }
            },
            
            # =======================================
            # 📁 모델 경로 설정
            # =======================================
            "model_paths": {
                "base_dir": "app/ai_pipeline/models/ai_models",
                "cache_dir": "app/ai_pipeline/cache",
                "checkpoints": {
                    "graphonomy": "graphonomy/checkpoints/graphonomy.pth",
                    "hr_viton": "hr_viton/checkpoints/hr_viton.pth",
                    "u2net": "u2net/checkpoints/u2net.pth",
                    "real_esrgan": "real_esrgan/checkpoints/RealESRGAN_x4plus.pth",
                    "gfpgan": "gfpgan/checkpoints/GFPGANv1.4.pth",
                    "openpose": "openpose/checkpoints/pose_iter_440000.caffemodel",
                    "mediapipe": "mediapipe/models/pose_landmarker.task",
                    "ootd": "ootd/checkpoints/ootd_diffusion.pth",
                    "viton_hd": "viton_hd/checkpoints/viton_hd.pth"
                },
                "custom_paths": self.custom_model_paths
            },
            
            # =======================================
            # ⚡ 성능 최적화 설정
            # =======================================
            "optimization": {
                "device": self.device,
                "device_type": self.device_type,
                "memory_gb": self.memory_gb,
                "is_m3_max": self.is_m3_max,
                "optimization_enabled": self.optimization_enabled,
                "mixed_precision": self.optimization_enabled,
                "gradient_checkpointing": False,
                "memory_efficient_attention": True,
                "compile_models": False,  # PyTorch 2.0 compile
                "constructor_pattern": "optimal",
                "model_offloading": {
                    "enabled": True,
                    "strategy": "lru",  # lru, priority, manual
                    "keep_active": 2,
                    "offload_to": "cpu"
                },
                "batch_processing": {
                    "enabled": self.batch_processing,
                    "max_batch_size": 4 if self.device != 'cpu' else 1,
                    "dynamic_batching": self.dynamic_batching,
                    "batch_timeout": 5.0
                },
                "caching": {
                    "enabled": self.enable_caching,
                    "ttl": 3600,  # 1시간
                    "max_size": "2GB",
                    "cache_intermediate": self.enable_intermediate_saving,
                    "compression": True,
                    "cache_strategy": "lru"
                },
                "parallel_processing": {
                    "enabled": self.enable_parallel,
                    "max_workers": min(4, self.system_config.cpu_cores),
                    "thread_pool": True,
                    "async_pipeline": True
                }
            },
            
            # =======================================
            # 💾 메모리 관리 설정
            # =======================================
            "memory": {
                "max_memory_usage": f"{min(80, int(self.memory_gb * 0.8))}%" if self.memory_gb else "80%",
                "memory_gb": self.memory_gb,
                "cleanup_interval": 300,  # 5분
                "aggressive_cleanup": self.memory_optimization,
                "optimization": self.memory_optimization,
                "monitoring": {
                    "enabled": True,
                    "warning_threshold": 0.85,
                    "critical_threshold": 0.95,
                    "auto_cleanup": True
                },
                "model_offloading": {
                    "enabled": True,
                    "offload_to": "cpu",
                    "keep_in_memory": ["human_parsing", "pose_estimation"] if self.memory_gb < 32 else [],
                    "intelligent_preloading": True
                },
                "garbage_collection": {
                    "aggressive": self.memory_optimization,
                    "frequency": "auto",
                    "force_after_step": True
                }
            },
            
            # =======================================
            # 📊 로깅 및 모니터링
            # =======================================
            "logging": {
                "level": "DEBUG" if self.debug_mode else "INFO",
                "detailed_timing": True,
                "performance_metrics": True,
                "save_intermediate": self.enable_intermediate_saving,
                "debug_mode": self.debug_mode,
                "save_debug_info": self.save_debug_info,
                "constructor_pattern": "optimal",
                "profiling": {
                    "enabled": self.profiling_enabled,
                    "memory_profiling": self.profiling_enabled,
                    "time_profiling": self.profiling_enabled,
                    "gpu_profiling": self.profiling_enabled and self.device != 'cpu'
                },
                "monitoring": {
                    "system_metrics": True,
                    "pipeline_metrics": True,
                    "error_tracking": True,
                    "performance_alerts": True
                }
            },
            
            # =======================================
            # 🧪 실험적 기능
            # =======================================
            "experimental": {
                "features": self.experimental_features,
                "neural_style_transfer": self.experimental_features.get('neural_style_transfer', False),
                "advanced_physics": self.experimental_features.get('advanced_physics', False),
                "ai_fabric_detection": self.experimental_features.get('ai_fabric_detection', False),
                "real_time_processing": self.experimental_features.get('real_time_processing', False),
                "multi_person_support": self.experimental_features.get('multi_person_support', False),
                "3d_pose_estimation": self.experimental_features.get('3d_pose_estimation', False),
                "advanced_lighting": self.experimental_features.get('advanced_lighting', False)
            }
        }