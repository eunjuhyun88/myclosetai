/**
 * MyCloset AI 프론트엔드 완전한 타입 정의 (최종 통합 버전)
 * 실제 백엔드 API 구조와 100% 호환되는 TypeScript 타입들
 * - 8단계 가상 피팅 파이프라인
 * - WebSocket 실시간 진행률
 * - 백엔드 스키마 완전 호환
 * - App.tsx와 usePipeline Hook 완전 호환
 */

// =================================================================
// 🔧 기본 공통 타입들
// =================================================================

export type ProcessingStatusEnum = 'pending' | 'processing' | 'completed' | 'error' | 'cancelled';

export type DeviceType = 'auto' | 'cpu' | 'cuda' | 'mps' | 'mac' | 'pc';

export type QualityLevel = 'fast' | 'balanced' | 'quality' | 'ultra';

export type ClothingCategory = 'shirt' | 'dress' | 'pants' | 'skirt' | 'jacket' | 'sweater' | 'coat' | 'top' | 'bottom';

export type FabricType = 'cotton' | 'silk' | 'wool' | 'polyester' | 'linen' | 'denim' | 'leather' | 'synthetic';

export type StylePreference = 'tight' | 'slim' | 'regular' | 'loose' | 'oversized';

// =================================================================
// 🔧 Hook 설정 타입들 (완전 통합)
// =================================================================

export interface UsePipelineOptions {
  // 기본 연결 설정
  baseURL?: string;
  wsURL?: string;
  apiKey?: string;
  
  // WebSocket 연결 설정
  autoReconnect?: boolean;
  maxReconnectAttempts?: number;
  reconnectInterval?: number;
  heartbeatInterval?: number;
  connectionTimeout?: number;
  enableWebSocket?: boolean;
  
  // 시스템 설정 (백엔드 호환)
  device?: DeviceType;
  device_type?: DeviceType;
  memory_gb?: number;
  is_m3_max?: boolean;
  optimization_enabled?: boolean;
  quality_level?: QualityLevel;
  max_batch_size?: number;
  use_fp16?: boolean;
  enable_xformers?: boolean;
  
  // Hook 전용 설정
  autoHealthCheck?: boolean;
  healthCheckInterval?: number;
  persistSession?: boolean;
  enableDetailedProgress?: boolean;
  enableRetry?: boolean;
  maxRetryAttempts?: number;
  retryDelay?: number;
  
  // 기능 플래그
  enableTaskTracking?: boolean;
  enableBrandIntegration?: boolean;
  enableCaching?: boolean;
  enableOfflineMode?: boolean;
  enableDebugMode?: boolean;
  
  // 성능 설정
  requestTimeout?: number;
  uploadTimeout?: number;
  maxFileSize?: number;
  compressionEnabled?: boolean;
  compressionQuality?: number;
  
  // 에러 처리
  enableErrorReporting?: boolean;
  errorReportingURL?: string;
  enableFallbackMode?: boolean;
  
  // 실험적 기능
  enableExperimentalFeatures?: boolean;
  experimentalFlags?: string[];
  
  // 8단계 파이프라인 전용 설정 (추가됨)
  enableStepTracking?: boolean;
  enableRealTimeUpdates?: boolean;
  stepTimeout?: number;
  autoRetrySteps?: boolean;
  maxStepRetries?: number;
  
  // WebSocket 프로토콜 설정 (추가됨)
  wsProtocols?: string[];
  messageQueueSize?: number;
  enableCompression?: boolean;
  cacheTimeout?: number;
}

// =================================================================
// 🔧 사용자 측정값 타입들
// =================================================================

export interface UserMeasurements {
  height: number;
  weight: number;
  chest?: number;
  waist?: number;
  hip?: number;
  shoulder_width?: number;
  arm_length?: number;
  leg_length?: number;
  neck?: number;
  wrist?: number;
  ankle?: number;
}

// =================================================================
// 🔧 요청/응답 타입들 (백엔드 스키마 완전 호환)
// =================================================================

export interface VirtualTryOnRequest {
  // 필수 입력
  person_image: File;
  clothing_image: File;
  height: number;
  weight: number;
  
  // 선택적 측정값
  chest?: number;
  waist?: number;
  hip?: number;
  shoulder_width?: number;
  
  // 의류 정보
  clothing_type?: ClothingCategory;
  fabric_type?: FabricType;
  style_preference?: StylePreference;
  clothing_size?: string;
  brand?: string;
  
  // 처리 옵션
  quality_mode?: QualityLevel;
  session_id?: string;
  task_id?: string;
  enable_realtime?: boolean;
  save_intermediate?: boolean;
  
  // 고급 옵션
  pose_adjustment?: boolean;
  color_preservation?: boolean;
  texture_enhancement?: boolean;
  background_removal?: boolean;
  lighting_adjustment?: boolean;
  
  // 메타데이터
  user_id?: string;
  client_version?: string;
  platform?: string;
  timestamp?: string;
  
  // 8단계 파이프라인 전용 (추가됨)
  enable_step_tracking?: boolean;
  experimental_features?: string[];
  custom_parameters?: Record<string, any>;
}

export interface ClothingAnalysis {
  category: ClothingCategory;
  style: string;
  dominant_color: number[];
  secondary_colors?: number[][];
  material?: FabricType;
  pattern?: string;
  texture_description?: string;
  brand_detected?: string;
  size_detected?: string;
  confidence: number;
  garment_boundaries?: number[][];
  style_attributes?: Record<string, any>;
}

export interface BodyAnalysis {
  body_type: string;
  pose_landmarks: number[][];
  body_measurements: UserMeasurements;
  pose_confidence: number;
  body_parts_segmentation?: number[][];
  skin_tone?: number[];
  estimated_size?: Record<string, string>;
  fit_preferences?: string[];
}

export interface QualityMetrics {
  // 기본 품질 지표
  ssim: number;
  lpips: number;
  fid?: number;
  inception_score?: number;
  
  // 피팅 품질 지표
  fit_overall: number;
  fit_coverage?: number;
  fit_shape_consistency?: number;
  fit_size_accuracy?: number;
  
  // 시각적 품질 지표
  color_preservation?: number;
  texture_quality?: number;
  boundary_naturalness?: number;
  lighting_consistency?: number;
  shadow_realism?: number;
  
  // 기술적 지표
  resolution_preservation?: number;
  noise_level?: number;
  artifact_score?: number;
}

export interface ProcessingDiagnostics {
  step_times: Record<string, number>;
  memory_usage: Record<string, number>;
  model_performance: Record<string, any>;
  error_details?: Record<string, any>;
  warning_messages?: string[];
  debug_info?: Record<string, any>;
}

export interface VirtualTryOnResponse {
  // 기본 결과
  success: boolean;
  fitted_image?: string; // base64
  processing_time: number;
  confidence: number;
  
  // 분석 결과
  measurements: UserMeasurements;
  clothing_analysis: ClothingAnalysis;
  body_analysis?: BodyAnalysis;
  
  // 품질 평가
  fit_score: number;
  quality_metrics: QualityMetrics;
  recommendations: string[];
  
  // 메타데이터
  session_id?: string;
  task_id?: string;
  model_version?: string;
  processing_pipeline?: string;
  
  // 중간 결과 (디버그용)
  intermediate_results?: Record<string, string>;
  processing_diagnostics?: ProcessingDiagnostics;
  
  // 에러 정보
  error_message?: string;
  error_code?: string;
  error_details?: Record<string, any>;
  
  // 추가 결과
  alternative_results?: string[];
  size_recommendations?: Record<string, string>;
  styling_suggestions?: string[];
  similar_products?: any[];
  
  // 8단계 파이프라인 전용 (추가됨)
  experimental_results?: Record<string, any>;
  debug_data?: Record<string, any>;
}

// =================================================================
// 🔧 파이프라인 진행률 타입들 (완전 통합)
// =================================================================

export interface PipelineStep {
  id: number;
  name: string;
  korean?: string; // 한국어 이름 (App.tsx 호환)
  description?: string;
  status: ProcessingStatusEnum | 'skipped' | 'timeout'; // 추가 상태들
  progress: number;
  start_time?: string;
  end_time?: string;
  duration?: number;
  error_message?: string;
  sub_steps?: PipelineStep[];
  metadata?: Record<string, any>;
}

export interface PipelineProgress {
  // 기본 정보
  type: 'connection_established' | 'pipeline_progress' | 'step_start' | 'step_progress' | 'step_complete' | 'step_error' | 'pipeline_completed' | 'pipeline_error' | 'step_update' | 'completed' | 'error' | 'warning' | 'debug';
  session_id?: string;
  task_id?: string;
  timestamp: number;
  
  // 진행률 정보
  progress: number;
  message: string;
  step_name?: string;
  step_id?: number;
  steps?: PipelineStep[];
  
  // 상세 정보
  current_stage?: string;
  eta?: number;
  processing_speed?: number;
  memory_usage?: Record<string, number>;
  
  // 결과 데이터 (추가됨)
  result?: any;
  processing_time?: number;
  
  // 메타데이터
  debug_info?: Record<string, any>;
  performance_metrics?: Record<string, number>;
  client_id?: string;
}

// =================================================================
// 🔧 시스템 상태 타입들
// =================================================================

export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  uptime: number;
  version: string;
  environment: string;
  
  services: {
    api: boolean;
    websocket: boolean;
    database?: boolean;
    redis?: boolean;
    ai_pipeline: boolean;
  };
  
  performance: {
    cpu_usage: number;
    memory_usage: number;
    gpu_usage?: number;
    disk_usage: number;
    network_latency?: number;
  };
  
  ai_models: {
    loaded_models: string[];
    model_status: Record<string, 'loaded' | 'loading' | 'error'>;
    gpu_memory_usage?: number;
    model_versions?: Record<string, string>;
  };
}

export interface PipelineStatus {
  status: 'idle' | 'processing' | 'error' | 'maintenance';
  device: DeviceType;
  current_quality_level: QualityLevel;
  
  memory_usage: {
    total: number;
    used: number;
    free: number;
    gpu_total?: number;
    gpu_used?: number;
    gpu_free?: number;
  };
  
  models_loaded: string[];
  active_connections: number;
  queue_length: number;
  
  performance_stats: {
    average_processing_time: number;
    requests_per_minute: number;
    success_rate: number;
    error_rate: number;
  };
  
  maintenance_info?: {
    scheduled_maintenance?: string;
    estimated_downtime?: number;
    maintenance_reason?: string;
  };
}

export interface SystemStats {
  // 기본 통계
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  
  // 성능 통계
  average_processing_time: number;
  median_processing_time: number;
  p95_processing_time: number;
  average_quality_score: number;
  
  // 메모리 통계
  peak_memory_usage: number;
  average_memory_usage: number;
  current_memory_usage: number;
  
  // 시간 정보
  uptime: number;
  last_restart: string;
  last_request: string;
  
  // 세부 통계
  step_performance: Record<string, {
    average_time: number;
    success_rate: number;
    error_count: number;
  }>;
  
  quality_distribution: {
    excellent: number; // > 0.9
    good: number;      // 0.7-0.9
    fair: number;      // 0.5-0.7
    poor: number;      // < 0.5
  };
  
  error_breakdown: Record<string, number>;
  device_usage_stats: Record<DeviceType, number>;
}

// =================================================================
// 🔧 Task 관리 타입들
// =================================================================

export interface TaskInfo {
  task_id: string;
  session_id?: string;
  user_id?: string;
  
  status: ProcessingStatusEnum;
  created_at: string;
  updated_at: string;
  completed_at?: string;
  
  progress_percentage: number;
  current_stage: string;
  eta?: number;
  
  request_data?: Partial<VirtualTryOnRequest>;
  result_data?: Partial<VirtualTryOnResponse>;
  
  error_info?: {
    error_code: string;
    error_message: string;
    error_details?: Record<string, any>;
    retry_count: number;
  };
  
  performance_metrics?: {
    processing_time?: number;
    memory_peak?: number;
    gpu_usage?: number;
    queue_wait_time?: number;
  };
  
  metadata?: Record<string, any>;
}

export interface ProcessingStatus {
  task_id: string;
  status: ProcessingStatusEnum;
  progress_percentage: number;
  current_stage: string;
  eta?: number;
  message?: string;
  
  steps_completed: number;
  total_steps: number;
  current_step_detail?: PipelineStep;
  
  error_info?: {
    error_code: string;
    error_message: string;
    retry_available: boolean;
  };
  
  performance_info?: {
    elapsed_time: number;
    estimated_remaining: number;
    processing_speed: number;
  };
}

// =================================================================
// 🔧 브랜드/사이즈 관련 타입들
// =================================================================

export interface BrandSizeData {
  brand: string;
  country: string;
  size_system: 'US' | 'EU' | 'UK' | 'JP' | 'KR' | 'CN';
  
  sizes: Record<string, {
    measurements: UserMeasurements;
    fit_type: StylePreference;
    size_label: string;
    size_numeric?: number;
  }>;
  
  size_chart_url?: string;
  fit_guide?: string[];
  brand_specific_notes?: string[];
}

export interface SizeRecommendation {
  recommended_size: string;
  confidence: number;
  fit_type: StylePreference;
  
  alternatives: Array<{
    size: string;
    confidence: number;
    fit_type: StylePreference;
    notes?: string;
  }>;
  
  size_comparison: {
    too_small_probability: number;
    perfect_fit_probability: number;
    too_large_probability: number;
  };
  
  recommendations: string[];
  brand_specific_advice?: string[];
}

// =================================================================
// 🔧 Hook 상태 타입들 (완전 통합)
// =================================================================

export interface UsePipelineState {
  // 처리 상태
  isProcessing: boolean;
  progress: number;
  progressMessage: string;
  currentStep: string;
  stepProgress: number;
  
  // 결과 상태
  result: VirtualTryOnResponse | null;
  error: string | null;
  
  // 연결 상태
  isConnected: boolean;
  isHealthy: boolean;
  connectionAttempts: number;
  lastConnectionAttempt: Date | null;
  
  // 시스템 상태
  pipelineStatus: PipelineStatus | null;
  systemStats: SystemStats | null;
  systemHealth?: SystemHealth | null;
  
  // 세션 관리
  sessionId: string | null;
  currentTaskId?: string | null;
  
  // 상세 진행률
  steps: PipelineStep[];
  activeTask?: TaskInfo | null;
  
  // 8단계 파이프라인 확장 (추가됨)
  currentPipelineStep: number;
  pipelineSteps: PipelineStep[];
  stepResults: { [stepId: number]: any };
  sessionActive: boolean;
  
  // 메타데이터
  lastProcessingTime?: number;
  totalRequestsCount?: number;
  successfulRequestsCount?: number;
  
  // 캐시 상태
  cachedResults?: Map<string, VirtualTryOnResponse>;
  
  // 브랜드 데이터
  brandSizeData?: Map<string, BrandSizeData>;
}

export interface UsePipelineActions {
  // 메인 처리 액션
  processVirtualTryOn: (request: VirtualTryOnRequest) => Promise<VirtualTryOnResponse | void>;
  
  // 결과 관리
  clearResult: () => void;
  clearError: () => void;
  reset: () => void;
  
  // 연결 관리
  connect: () => Promise<boolean>;
  disconnect: () => void;
  reconnect: () => Promise<boolean>;
  
  // 상태 조회
  checkHealth: () => Promise<boolean>;
  getPipelineStatus: () => Promise<void>;
  getSystemStats: () => Promise<void>;
  getSystemHealth?: () => Promise<void>;
  
  // 파이프라인 관리
  warmupPipeline: (qualityMode?: QualityLevel | string) => Promise<void>;
  testConnection: () => Promise<void>;
  
  // Task 관리
  getTaskStatus?: (taskId: string) => Promise<ProcessingStatus | null>;
  cancelTask?: (taskId: string) => Promise<boolean>;
  retryTask?: (taskId: string) => Promise<boolean>;
  getTaskHistory?: () => TaskInfo[];
  
  // 브랜드/사이즈 기능
  getBrandSizes?: (brand: string) => Promise<BrandSizeData | null>;
  getSizeRecommendation?: (measurements: UserMeasurements, brand: string, item: string) => Promise<SizeRecommendation | null>;
  
  // 유틸리티
  sendHeartbeat: () => void;
  exportLogs: () => void;
  clearCache?: () => void;
  
  // 실험적 기능
  enableExperimentalFeature?: (feature: string) => void;
  disableExperimentalFeature?: (feature: string) => void;
}

// =================================================================
// 🔧 에러 타입들
// =================================================================

export interface APIError {
  code: string;
  message: string;
  details?: Record<string, any>;
  timestamp: string;
  request_id?: string;
  retry_after?: number;
}

export interface ValidationError extends APIError {
  field_errors: Record<string, string[]>;
}

export interface RateLimitError extends APIError {
  limit: number;
  remaining: number;
  reset_time: string;
}

// =================================================================
// 🔧 이벤트 타입들
// =================================================================

export interface PipelineEvent {
  type: string;
  data: any;
  timestamp: number;
  source: 'api' | 'websocket' | 'hook';
}

export type PipelineEventHandler = (event: PipelineEvent) => void;

// =================================================================
// 🔧 브라우저 호환성 타입들
// =================================================================

export interface BrowserCompatibility {
  websocket: boolean;
  fileApi: boolean;
  formData: boolean;
  fetch: boolean;
  webgl?: boolean;
  webassembly?: boolean;
  overall: boolean;
}

export interface SystemRequirements {
  minMemory: number;
  recommendedMemory: number;
  supportedBrowsers: string[];
  requiredFeatures: string[];
}

// =================================================================
// 🔧 유틸리티 타입들 (완전 통합)
// =================================================================

// 진행률 콜백 타입
export type ProgressCallback = (progress: PipelineProgress) => void;
export type ErrorCallback = (error: APIError) => void;
export type CompleteCallback = (result: VirtualTryOnResponse) => void;

// Hook 합성 타입
export type PipelineHookConfig = UsePipelineOptions;
export type PipelineHookState = UsePipelineState & UsePipelineActions;

// 요청 타입 확장
export type ExtendedVirtualTryOnRequest = VirtualTryOnRequest & {
  experimental_features?: string[];
  custom_parameters?: Record<string, any>;
};

// 응답 타입 확장  
export type ExtendedVirtualTryOnResponse = VirtualTryOnResponse & {
  experimental_results?: Record<string, any>;
  debug_data?: Record<string, any>;
};

// =================================================================
// 🔧 모든 타입 내보내기 (완전 통합)
// =================================================================

export type {
  // 기본 타입들
  ProcessingStatusEnum,
  DeviceType,
  QualityLevel,
  ClothingCategory,
  FabricType,
  StylePreference,
  
  // 주요 인터페이스들
  VirtualTryOnRequest,
  VirtualTryOnResponse,
  PipelineProgress,
  PipelineStep,
  PipelineStatus,
  SystemStats,
  TaskInfo,
  ProcessingStatus,
  
  // Hook 관련
  UsePipelineOptions,
  UsePipelineState,
  UsePipelineActions,
  
  // 유틸리티 타입들
  ProgressCallback,
  ErrorCallback,
  CompleteCallback,
  PipelineEvent,
  PipelineEventHandler,
  
  // 확장 타입들
  ExtendedVirtualTryOnRequest,
  ExtendedVirtualTryOnResponse,
  PipelineHookConfig,
  PipelineHookState,
  
  // 시스템 타입들
  SystemHealth,
  BrowserCompatibility,
  SystemRequirements,
  UserMeasurements,
  ClothingAnalysis,
  BodyAnalysis,
  QualityMetrics,
  ProcessingDiagnostics,
  BrandSizeData,
  SizeRecommendation,
  APIError,
  ValidationError,
  RateLimitError,
};

// 기본 내보내기
export default UsePipelineOptions;