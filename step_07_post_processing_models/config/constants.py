"""
Post Processing Models Constants

후처리 모델들의 상수들을 정의합니다.
"""

# 모델 타입 상수
MODEL_TYPE_SWINIR = 'swinir'
MODEL_TYPE_REALESRGAN = 'realesrgan'
MODEL_TYPE_GFPGAN = 'gfpgan'
MODEL_TYPE_CODEFORMER = 'codeformer'

# 지원하는 모델 타입들
SUPPORTED_MODEL_TYPES = [
    MODEL_TYPE_SWINIR,
    MODEL_TYPE_REALESRGAN,
    MODEL_TYPE_GFPGAN,
    MODEL_TYPE_CODEFORMER
]

# 모델별 기본 체크포인트 파일명
DEFAULT_CHECKPOINT_FILES = {
    MODEL_TYPE_SWINIR: 'swinir_checkpoint.pth',
    MODEL_TYPE_REALESRGAN: 'realesrgan_checkpoint.pth',
    MODEL_TYPE_GFPGAN: 'gfpgan_checkpoint.pth',
    MODEL_TYPE_CODEFORMER: 'codeformer_checkpoint.pth'
}

# 모델별 기본 설정
DEFAULT_MODEL_CONFIGS = {
    MODEL_TYPE_SWINIR: {
        'img_size': 64,
        'patch_size': 1,
        'in_chans': 3,
        'embed_dim': 96,
        'depths': [6, 6, 6, 6, 6, 6],
        'num_heads': [6, 6, 6, 6, 6, 6],
        'window_size': 7,
        'mlp_ratio': 4.0,
        'qkv_bias': True,
        'qk_scale': None,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.1,
        'ape': False,
        'patch_norm': True,
        'use_checkpoint': False,
        'upscale': 4,
        'img_range': 1.0,
        'upsampler': 'nearest+conv',
        'resi_connection': '1conv'
    },
    MODEL_TYPE_REALESRGAN: {
        'num_feat': 64,
        'num_block': 23,
        'upscale': 4,
        'num_in_ch': 3,
        'num_out_ch': 3,
        'task': 'realesrgan'
    },
    MODEL_TYPE_GFPGAN: {
        'out_size': 512,
        'num_style_feat': 512,
        'channel_multiplier': 2,
        'decoder_load_path': None,
        'fix_decoder': True,
        'num_mlp': 8,
        'input_is_latent': True,
        'different_w': True,
        'narrow': 1,
        'sft_half': True
    },
    MODEL_TYPE_CODEFORMER: {
        'dim_embd': 512,
        'n_head': 8,
        'n_layers': 9,
        'codebook_size': 1024,
        'latent_dim': 256,
        'channels': [64, 128, 256, 512],
        'img_size': 256
    }
}

# 추론 설정 상수
DEFAULT_INFERENCE_CONFIG = {
    'tile_size': 400,
    'tile_pad': 10,
    'pre_pad': 0,
    'half': True,
    'bg_upsampler': True,
    'bg_tile': 400,
    'suffix': None,
    'only_center_face': False,
    'aligned': False,
    'background_enhance': True,
    'face_upsample': True,
    'upscale': 2,
    'codeformer_fidelity': 0.7
}

# 이미지 처리 상수
IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
MAX_IMAGE_SIZE = 4096  # 최대 이미지 크기
MIN_IMAGE_SIZE = 64    # 최소 이미지 크기

# 메모리 관리 상수
MAX_MEMORY_USAGE = 0.8  # GPU 메모리의 80%까지 사용
MEMORY_CLEANUP_THRESHOLD = 0.9  # 90% 이상 사용 시 메모리 정리

# 배치 처리 상수
DEFAULT_BATCH_SIZE = 1
MAX_BATCH_SIZE = 8
BATCH_TIMEOUT = 300  # 배치 처리 타임아웃 (초)

# 로깅 상수
LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
DEFAULT_LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# 에러 코드 상수
ERROR_CODES = {
    'SUCCESS': 0,
    'MODEL_NOT_FOUND': 1001,
    'CHECKPOINT_NOT_FOUND': 1002,
    'INVALID_INPUT': 1003,
    'INFERENCE_FAILED': 1004,
    'MEMORY_ERROR': 1005,
    'CONFIG_ERROR': 1006,
    'UNKNOWN_ERROR': 9999
}

# 성능 메트릭 상수
PERFORMANCE_METRICS = [
    'inference_time',
    'memory_usage',
    'gpu_utilization',
    'throughput',
    'quality_score'
]

# 품질 평가 상수
QUALITY_METRICS = [
    'psnr',      # Peak Signal-to-Noise Ratio
    'ssim',      # Structural Similarity Index
    'lpips',     # Learned Perceptual Image Patch Similarity
    'fid',       # Fréchet Inception Distance
    'lpips_alex' # LPIPS with AlexNet backbone
]

# 모델 가중치 초기화 상수
WEIGHT_INIT_METHODS = [
    'xavier_uniform',
    'xavier_normal',
    'kaiming_uniform',
    'kaiming_normal',
    'orthogonal',
    'sparse'
]

# 활성화 함수 상수
ACTIVATION_FUNCTIONS = [
    'relu',
    'leaky_relu',
    'gelu',
    'swish',
    'mish',
    'elu',
    'selu'
]

# 옵티마이저 상수
OPTIMIZER_TYPES = [
    'adam',
    'adamw',
    'sgd',
    'rmsprop',
    'adagrad',
    'adamax'
]

# 스케줄러 상수
SCHEDULER_TYPES = [
    'step',
    'cosine',
    'exponential',
    'plateau',
    'one_cycle',
    'cosine_annealing'
]

# 데이터 증강 상수
AUGMENTATION_METHODS = [
    'horizontal_flip',
    'vertical_flip',
    'rotation',
    'translation',
    'scaling',
    'noise',
    'blur',
    'sharpening'
]

# 체크포인트 저장 상수
CHECKPOINT_SAVE_FREQUENCY = 1000  # N 스텝마다 체크포인트 저장
MAX_CHECKPOINT_FILES = 5  # 최대 보관할 체크포인트 파일 수

# 모델 앙상블 상수
ENSEMBLE_METHODS = [
    'average',
    'weighted_average',
    'voting',
    'stacking',
    'bagging'
]

# 하드웨어 가속 상수
HARDWARE_ACCELERATION = [
    'cuda',
    'mps',      # Apple Silicon
    'rocm',     # AMD GPU
    'cpu'
]

# 모델 압축 상수
COMPRESSION_METHODS = [
    'pruning',
    'quantization',
    'knowledge_distillation',
    'low_rank_approximation'
]

# 실험 추적 상수
EXPERIMENT_TRACKING = [
    'tensorboard',
    'wandb',
    'mlflow',
    'tensorboardx'
]

# 배포 모델 형식 상수
DEPLOYMENT_FORMATS = [
    'torchscript',
    'onnx',
    'tensorrt',
    'openvino',
    'coreml'
]

# 모니터링 상수
MONITORING_METRICS = [
    'gpu_memory',
    'gpu_utilization',
    'cpu_usage',
    'memory_usage',
    'inference_latency',
    'throughput'
]

# 알림 상수
NOTIFICATION_TYPES = [
    'email',
    'slack',
    'discord',
    'webhook'
]

# 보안 상수
SECURITY_LEVELS = [
    'public',
    'internal',
    'confidential',
    'restricted'
]

# 라이센스 상수
LICENSE_TYPES = [
    'mit',
    'apache2',
    'gpl3',
    'bsd',
    'proprietary'
]

# 버전 관리 상수
VERSION_FORMAT = 'semver'  # semantic versioning
MINIMUM_PYTHON_VERSION = '3.8'
MINIMUM_TORCH_VERSION = '1.8.0'

# 환경 변수 상수
ENVIRONMENT_VARIABLES = {
    'CHECKPOINT_DIR': 'POST_PROCESSING_CHECKPOINT_DIR',
    'LOG_LEVEL': 'POST_PROCESSING_LOG_LEVEL',
    'DEVICE': 'POST_PROCESSING_DEVICE',
    'MAX_MEMORY_USAGE': 'POST_PROCESSING_MAX_MEMORY_USAGE',
    'BATCH_SIZE': 'POST_PROCESSING_BATCH_SIZE'
}

# 파일 경로 상수
DEFAULT_PATHS = {
    'checkpoints': './checkpoints',
    'logs': './logs',
    'outputs': './outputs',
    'temp': './temp',
    'configs': './configs'
}

# 파일 확장자 상수
FILE_EXTENSIONS = {
    'checkpoint': '.pth',
    'config': '.yaml',
    'log': '.log',
    'image': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
    'video': ['.mp4', '.avi', '.mov', '.mkv']
}

# 네트워크 상수
NETWORK_CONFIG = {
    'timeout': 30,
    'retry_count': 3,
    'max_connections': 10,
    'chunk_size': 8192
}

# 캐시 상수
CACHE_CONFIG = {
    'max_size': 1000,
    'ttl': 3600,  # 1시간
    'cleanup_interval': 300  # 5분
}

# 스레드 풀 상수
THREAD_POOL_CONFIG = {
    'max_workers': 4,
    'thread_name_prefix': 'post_processing_worker'
}

# 비동기 처리 상수
ASYNC_CONFIG = {
    'max_concurrent_tasks': 4,
    'task_timeout': 300,
    'retry_delay': 1
}

# 성능 프로파일링 상수
PROFILING_CONFIG = {
    'enable_profiling': False,
    'profile_memory': True,
    'profile_time': True,
    'profile_cpu': False
}

# 디버깅 상수
DEBUG_CONFIG = {
    'enable_debug_mode': False,
    'save_intermediate_results': False,
    'verbose_logging': False,
    'assertion_checks': True
}

# 테스트 상수
TEST_CONFIG = {
    'test_data_dir': './test_data',
    'test_output_dir': './test_outputs',
    'test_timeout': 60,
    'test_retry_count': 2
}

# 문서화 상수
DOCUMENTATION_CONFIG = {
    'generate_api_docs': True,
    'generate_examples': True,
    'include_type_hints': True,
    'docstring_format': 'google'
}

# 국제화 상수
I18N_CONFIG = {
    'default_language': 'ko',
    'supported_languages': ['ko', 'en'],
    'fallback_language': 'en'
}

# 접근성 상수
ACCESSIBILITY_CONFIG = {
    'high_contrast_mode': False,
    'screen_reader_support': True,
    'keyboard_navigation': True
}

# 백업 상수
BACKUP_CONFIG = {
    'enable_auto_backup': True,
    'backup_frequency': 24,  # 시간
    'max_backup_files': 10,
    'backup_compression': True
}

# 동기화 상수
SYNC_CONFIG = {
    'enable_auto_sync': False,
    'sync_interval': 300,  # 5분
    'conflict_resolution': 'latest_wins'
}

# 검증 상수
VALIDATION_CONFIG = {
    'validate_inputs': True,
    'validate_outputs': True,
    'validate_configs': True,
    'strict_mode': False
}

# 오류 복구 상수
RECOVERY_CONFIG = {
    'enable_auto_recovery': True,
    'max_recovery_attempts': 3,
    'recovery_delay': 5,  # 초
    'fallback_strategy': 'use_default'
}

# 성능 튜닝 상수
PERFORMANCE_TUNING = {
    'enable_auto_tuning': False,
    'tuning_interval': 1000,  # 스텝
    'tuning_metrics': ['throughput', 'latency'],
    'tuning_strategy': 'greedy'
}

# 리소스 관리 상수
RESOURCE_MANAGEMENT = {
    'enable_resource_monitoring': True,
    'resource_cleanup_interval': 60,  # 초
    'max_resource_usage': 0.9,
    'resource_cleanup_threshold': 0.8
}

# 보안 설정 상수
SECURITY_CONFIG = {
    'enable_input_validation': True,
    'enable_output_sanitization': True,
    'max_file_size': 100 * 1024 * 1024,  # 100MB
    'allowed_file_types': IMAGE_FORMATS
}

# 로깅 설정 상수
LOGGING_CONFIG = {
    'log_rotation': True,
    'max_log_file_size': 10 * 1024 * 1024,  # 10MB
    'max_log_files': 5,
    'log_compression': True
}

# 모니터링 설정 상수
MONITORING_CONFIG = {
    'enable_health_checks': True,
    'health_check_interval': 30,  # 초
    'enable_metrics_collection': True,
    'metrics_export_interval': 60  # 초
}
