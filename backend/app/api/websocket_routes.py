
export interface UserMeasurements {
  height: number;
  weight: number;
  chest?: number;
  waist?: number;
  hip?: number;
  shoulder_width?: number;
  bmi?: number;
  body_type?: string;
}

export interface ClothingAnalysis {
  category: string;
  style: string;
  dominant_color: number[];
  material?: string;
  confidence?: number;
  size_recommendation?: string;
  style_match?: string;
}

export interface QualityMetrics {
  ssim: number;
  lpips: number;
  fid?: number;
  fit_overall: number;
  fit_coverage?: number;
  color_preservation?: number;
  boundary_naturalness?: number;
  texture_quality?: number;
  pose_accuracy?: number;
}

export interface VirtualTryOnRequest {
  person_image: File;
  clothing_image: File;
  height: number;
  weight: number;
  chest?: number;
  waist?: number;
  hip?: number;
  clothing_type?: string;
  fabric_type?: string;
  style_preference?: string;
  quality_mode?: 'fast' | 'balanced' | 'quality';
  session_id?: string;
  enable_realtime?: boolean;
}

// MyCloset AI 특화 응답 인터페이스 - 프로젝트 지식 기반
export interface VirtualTryOnResponse {
  success: boolean;
  process_id?: string;
  task_id?: string;
  fitted_image?: string; // base64
  processing_time: number;
  confidence: number;
  fit_score: number;
  quality_score?: number;
  
  // MyCloset AI 특화 필드들
  platform: string;
  version: string;
  pipeline_info: {
    total_steps: number;
    completed_steps: number;
    optimization: string;
    ai_models: {
      human_parsing: string;
      pose_estimation: string;
      cloth_segmentation: string;
      geometric_matching: string;
      cloth_warping: string;
      virtual_fitting: string;
      post_processing: string;
      quality_assessment: string;
    };
  };
  
  measurements: UserMeasurements;
  clothing_analysis: ClothingAnalysis;
  recommendations: string[];
  quality_metrics?: QualityMetrics;
  pipeline_stages?: Record<string, any>;
  step_times?: Record<string, number>;
  memory_usage?: Record<string, any>;
  device_info?: {
    device: string;
    optimization: string;
    memory_usage: string;
  };
  error?: string;
}

// 실제 백엔드 WebSocket 메시지 형식 - 프로젝트 지식 기반
export interface PipelineProgress {
  type: 'pipeline_progress' | 'pipeline_started' | 'step_update' | 'completed' | 'error' | 'connection_established';
  session_id?: string;
  step_id?: number;
  step_name?: string;
  progress: number;
  message: string;
  timestamp: number;
  data?: {
    step_name?: string;
    progress?: number;
    message?: string;
    status?: string;
    processing_time?: number;
    fit_score?: number;
    quality_score?: number;
    platform?: string;
    pipeline_steps?: number;
    steps?: string[];
    estimated_time?: number;
  };
  status?: 'pending' | 'processing' | 'completed' | 'error';
}

// MyCloset AI 파이프라인 상태 - 프로젝트 지식 반영
export interface PipelineStatus {
  initialized: boolean;
  platform: string;
  version: string;
  device: string;
  device_type?: string;
  memory_gb?: number;
  is_m3_max?: boolean;
  optimization_enabled?: boolean;
  steps_loaded: number;
  total_steps: number;
  memory_status: Record<string, any>;
  stats: Record<string, any>;
  performance_metrics?: Record<string, any>;
  pipeline_config?: Record<string, any>;
  pipeline_ready?: boolean;
  ai_pipeline_steps?: string[];
  optimization_features?: string[];
}

export interface BodyAnalysisResponse {
  success: boolean;
  measurements: UserMeasurements;
  body_type: string;
  recommendations: string[];
  processing_time: number;
}

export interface ClothingAnalysisResponse {
  success: boolean;
  analysis: ClothingAnalysis;
  tags: string[];
  processing_time: number;
}

export interface ProcessingStatus {
  task_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';
  progress_percentage: number;
  current_stage: string;
  estimated_remaining_time?: number;
  created_at: string;
  updated_at: string;
}

export interface SystemStats {
  total_requests: number;
  successful_requests: number;
  average_processing_time: number;
  average_quality_score: number;
  peak_memory_usage: number;
  uptime: number;
  active_connections?: number;
  pipeline_health?: string;
}

export interface TaskInfo {
  task_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';
  progress_percentage: number;
  current_stage: string;
  created_at: string;
  updated_at: string;
}

export interface BrandSizeData {
  brand: string;
  sizes: Record<string, any>;
  size_chart: any[];
  fit_guide: string[];
}

export interface SizeRecommendation {
  recommended_size: string;
  confidence: number;
  alternatives: string[];
  fit_notes: string[];
}

// 연결 설정 인터페이스
export interface ConnectionConfig {
  baseURL?: string;
  wsURL?: string;
  autoReconnect?: boolean;
  maxReconnectAttempts?: number;
  reconnectInterval?: number;
  connectionTimeout?: number;
  device_type?: string;
  memory_gb?: number;
  is_m3_max?: boolean;
  optimization_enabled?: boolean;
  quality_level?: string;
  enableRetry?: boolean;
  maxRetryAttempts?: number;
  enableTaskTracking?: boolean;
  enableBrandIntegration?: boolean;
}

// ============================================
// 🛠️ 유틸리티 클래스
// ============================================

export class PipelineUtils {
  /**
   * 이미지 파일을 base64로 변환
   */
  static async fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        resolve(result.split(',')[1]); // "data:image/...;base64," 부분 제거
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  /**
   * base64를 이미지 URL로 변환
   */
  static base64ToImageURL(base64: string): string {
    return `data:image/png;base64,${base64}`;
  }

  /**
   * 파일 크기 검증 (프로젝트 지식: 15MB)
   */
  static validateFileSize(file: File, maxSizeMB: number = 15): boolean {
    const maxSizeBytes = maxSizeMB * 1024 * 1024;
    return file.size <= maxSizeBytes;
  }

  /**
   * 이미지 파일 타입 검증
   */
  static validateImageType(file: File): boolean {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    return allowedTypes.includes(file.type);
  }

  /**
   * 처리 시간을 사용자 친화적 형식으로 변환
   */
  static formatProcessingTime(seconds: number): string {
    if (seconds < 1) {
      return `${Math.round(seconds * 1000)}ms`;
    } else if (seconds < 60) {
      return `${seconds.toFixed(1)}초`;
    } else {
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = Math.round(seconds % 60);
      return `${minutes}분 ${remainingSeconds}초`;
    }
  }

  /**
   * 품질 점수를 등급으로 변환
   */
  static getQualityGrade(score: number): {
    grade: string;
    color: string;
    description: string;
  } {
    if (score >= 0.9) {
      return { grade: 'Excellent', color: 'text-green-600', description: '완벽한 품질' };
    } else if (score >= 0.8) {
      return { grade: 'Good', color: 'text-blue-600', description: '우수한 품질' };
    } else if (score >= 0.6) {
      return { grade: 'Fair', color: 'text-yellow-600', description: '양호한 품질' };
    } else {
      return { grade: 'Poor', color: 'text-red-600', description: '개선 필요' };
    }
  }

  /**
   * 메모리 사용량을 사용자 친화적 형식으로 변환
   */
  static formatMemoryUsage(bytes: number): string {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }

    return `${size.toFixed(1)} ${units[unitIndex]}`;
  }

  /**
   * 세션 ID 생성
   */
  static generateSessionId(): string {
    return `mycloset_session_${Date.now()}_${Math.random().toString(36).substring(7)}`;
  }

  /**
   * 진행률을 백분율 문자열로 변환
   */
  static formatProgress(progress: number): string {
    return `${Math.round(Math.max(0, Math.min(100, progress)))}%`;
  }

  /**
   * 디바이스 자동 감지
   */
  static autoDetectDevice(): string {
    // 브라우저에서 디바이스 감지 로직
    const userAgent = navigator.userAgent.toLowerCase();
    if (userAgent.includes('mac')) {
      return 'mps'; // M3 Max 우선
    }
    return 'auto';
  }

  /**
   * 디바이스 타입 감지
   */
  static autoDetectDeviceType(): string {
    const userAgent = navigator.userAgent.toLowerCase();
    if (userAgent.includes('mac')) {
      return 'apple_silicon';
    }
    return 'auto';
  }

  /**
   * M3 Max 감지
   */
  static detectM3Max(): boolean {
    const userAgent = navigator.userAgent.toLowerCase();
    return userAgent.includes('mac');
  }

  /**
   * 로그 출력
   */
  static log(level: 'info' | 'warn' | 'error', message: string, data?: any): void {
    const timestamp = new Date().toISOString();
    const prefix = `[${timestamp}] MyCloset AI:`;
    
    switch (level) {
      case 'info':
        console.log(`${prefix} ${message}`, data || '');
        break;
      case 'warn':
        console.warn(`${prefix} ${message}`, data || '');
        break;
      case 'error':
        console.error(`${prefix} ${message}`, data || '');
        break;
    }
  }

  /**
   * 성능 타이머 생성
   */
  static createPerformanceTimer(label: string): { end: () => number } {
    const startTime = performance.now();
    return {
      end: () => {
        const endTime = performance.now();
        const duration = endTime - startTime;
        PipelineUtils.log('info', `⏱️ ${label}: ${duration.toFixed(2)}ms`);
        return duration;
      }
    };
  }

  /**
   * HTTP 에러 메시지 변환
   */
  static getHTTPErrorMessage(status: number): string {
    const messages: Record<number, string> = {
      400: '잘못된 요청입니다.',
      401: '인증이 필요합니다.',
      403: '접근이 거부되었습니다.',
      404: '요청한 리소스를 찾을 수 없습니다.',
      413: '파일 크기가 너무 큽니다.',
      500: '서버 내부 오류가 발생했습니다.',
      503: '서비스를 사용할 수 없습니다.',
    };
    return messages[status] || `HTTP 오류 (${status})`;
  }

  /**
   * 시스템 파라미터 목록
   */
  static getSystemParams(): Set<string> {
    return new Set([
      'device_type', 'memory_gb', 'is_m3_max', 
      'optimization_enabled', 'quality_level'
    ]);
  }
}

// ============================================
// 🚀 메인 API 클라이언트 클래스
// ============================================

export class PipelineAPIClient {
  private baseURL: string;
  private wsURL: string;
  private device: string;
  private step_name: string;
  private config: ConnectionConfig;
  private defaultHeaders: Record<string, string>;
  private activeTasks: Map<string, TaskInfo> = new Map();
  
  // WebSocket 관련
  private currentWS: WebSocket | null = null;
  private connectionAttempts: number = 0;
  private heartbeatTimer: NodeJS.Timeout | null = null;

  constructor(config: ConnectionConfig = {}) {
    this.baseURL = config.baseURL || 'http://localhost:8000';
    this.wsURL = config.wsURL || this.baseURL.replace('http', 'ws');
    this.device = PipelineUtils.autoDetectDevice();
    this.step_name = 'PipelineAPIClient';
    
    this.config = {
      autoReconnect: true,
      maxReconnectAttempts: 5,
      reconnectInterval: 3000,
      connectionTimeout: 10000,
      device_type: PipelineUtils.autoDetectDeviceType(),
      memory_gb: 128.0,
      is_m3_max: PipelineUtils.detectM3Max(),
      optimization_enabled: true,
      quality_level: 'balanced',
      enableRetry: true,
      maxRetryAttempts: 3,
      enableTaskTracking: true,
      enableBrandIntegration: true,
      ...config,
    };

    this.defaultHeaders = {
      'Accept': 'application/json',
    };

    PipelineUtils.log('info', `🎯 ${this.step_name} 초기화 - 디바이스: ${this.device}`);
  }

  // ============================================
  // 🔧 기본 HTTP 요청 메서드들
  // ============================================

  private async request<T = any>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    const config: RequestInit = {
      ...options,
      headers: {
        ...this.defaultHeaders,
        ...options.headers,
      },
    };

    PipelineUtils.log('info', `🌐 API 요청: ${config.method || 'GET'} ${endpoint}`);

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        let errorMessage = `HTTP ${response.status}`;
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.error || errorMessage;
        } catch {
          // JSON 파싱 실패 시 기본 메시지 사용
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();
      PipelineUtils.log('info', `✅ API 응답 성공: ${endpoint}`);
      return data;

    } catch (error) {
      PipelineUtils.log('error', `❌ API 요청 실패: ${endpoint}`, error);
      throw error;
    }
  }

  // ============================================
  // 🎯 메인 가상 피팅 API - 프로젝트 지식 완벽 반영
  // ============================================

  async processVirtualTryOn(
    request: VirtualTryOnRequest,
    onProgress?: (progress: PipelineProgress) => void
  ): Promise<VirtualTryOnResponse> {
    const timer = PipelineUtils.createPerformanceTimer('가상 피팅 API 요청');
    
    try {
      // 입력 검증
      this.validateRequest(request);

      // 세션 ID 생성
      const sessionId = request.session_id || PipelineUtils.generateSessionId();

      // WebSocket 연결 설정 (진행률 업데이트용)
      let wsConnected = false;
      if (onProgress) {
        try {
          wsConnected = await this.setupProgressWebSocket(sessionId, onProgress);
          if (!wsConnected) {
            PipelineUtils.log('warn', '⚠️ WebSocket 연결 실패, 진행률 업데이트 없이 계속 진행');
          }
        } catch (error) {
          PipelineUtils.log('warn', '⚠️ WebSocket 설정 실패:', error);
        }
      }

      // FormData 준비 - 프로젝트 지식 기반 파라미터
      const formData = new FormData();
      formData.append('person_image', request.person_image);
      formData.append('clothing_image', request.clothing_image);
      formData.append('height', request.height.toString());
      formData.append('weight', request.weight.toString());
      formData.append('quality_mode', request.quality_mode || 'balanced');
      formData.append('enable_realtime', wsConnected ? 'true' : 'false');
      formData.append('session_id', sessionId);

      // 선택적 파라미터들
      if (request.chest) formData.append('chest', request.chest.toString());
      if (request.waist) formData.append('waist', request.waist.toString());
      if (request.hip) formData.append('hip', request.hip.toString());
      if (request.clothing_type) formData.append('clothing_type', request.clothing_type);
      if (request.fabric_type) formData.append('fabric_type', request.fabric_type);
      if (request.style_preference) formData.append('style_preference', request.style_preference);

      PipelineUtils.log('info', '🚀 MyCloset AI 가상 피팅 API 요청 시작', {
        sessionId,
        qualityMode: request.quality_mode,
        wsConnected
      });

      // ✅ 실제 백엔드 엔드포인트 사용 (프로젝트 지식 확인)
      const response = await fetch(`${this.baseURL}/virtual-tryon`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errorMessage = `HTTP ${response.status}`;
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.error || errorMessage;
        } catch {
          // JSON 파싱 실패 시 기본 메시지 사용
        }
        throw new Error(errorMessage);
      }

      const result: VirtualTryOnResponse = await response.json();
      
      // WebSocket 연결 정리
      this.closeProgressWebSocket();
      
      timer.end();
      PipelineUtils.log('info', '✅ MyCloset AI 가상 피팅 API 응답 성공:', {
        success: result.success,
        processingTime: result.processing_time,
        fitScore: result.fit_score,
        processId: result.process_id,
        platform: result.platform
      });

      // Task 추적
      if (this.config.enableTaskTracking && result.task_id) {
        this.activeTasks.set(result.task_id, {
          task_id: result.task_id,
          status: 'completed',
          progress_percentage: 100,
          current_stage: 'completed',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        });
      }

      return result;

    } catch (error) {
      this.closeProgressWebSocket();
      timer.end();
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      PipelineUtils.log('error', '❌ MyCloset AI 가상 피팅 API 오류:', errorMessage);
      throw new Error(`가상 피팅 처리 실패: ${errorMessage}`);
    }
  }

  // ============================================
  // 🔗 WebSocket 연결 관리 - 실제 백엔드 구조 반영
  // ============================================

  private async setupProgressWebSocket(
    sessionId: string,
    onProgress: (progress: PipelineProgress) => void
  ): Promise<boolean> {
    return new Promise((resolve, reject) => {
      try {
        // ✅ 실제 백엔드 WebSocket 엔드포인트 (프로젝트 지식 기반)
        const clientId = `${sessionId}_${Date.now()}`;
        const wsUrl = `${this.wsURL}/api/ws/${clientId}`;
        
        PipelineUtils.log('info', '🔗 WebSocket 연결 시도:', wsUrl);
        
        const ws = new WebSocket(wsUrl);
        let connectionResolved = false;

        // 연결 타임아웃 설정
        const connectionTimer = setTimeout(() => {
          if (!connectionResolved) {
            connectionResolved = true;
            ws.close();
            PipelineUtils.log('error', '❌ WebSocket 연결 타임아웃');
            resolve(false);
          }
        }, this.config.connectionTimeout || 10000);

        ws.onopen = () => {
          if (!connectionResolved) {
            connectionResolved = true;
            clearTimeout(connectionTimer);
            this.currentWS = ws;
            this.connectionAttempts = 0;
            
            PipelineUtils.log('info', '✅ WebSocket 연결 성공');
            
            // ✅ 실제 백엔드 세션 구독 메시지 형식 (프로젝트 지식 기반)
            try {
              ws.send(JSON.stringify({
                type: 'subscribe_session',
                session_id: sessionId
              }));
              PipelineUtils.log('info', '📡 세션 구독:', sessionId);
            } catch (error) {
              PipelineUtils.log('warn', '⚠️ 세션 구독 실패:', error);
            }

            // 하트비트 시작
            this.startHeartbeat();
            
            resolve(true);
          }
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            
            // ✅ 실제 백엔드 메시지 타입들 처리 (프로젝트 지식 기반)
            if (data.type === 'pipeline_progress' || 
                data.type === 'pipeline_started' ||
                data.type === 'step_update' || 
                data.type === 'completed' || 
                data.type === 'error') {
              
              PipelineUtils.log('info', '📊 진행률 업데이트:', {
                type: data.type,
                step: data.step_name || data.data?.step_name,
                progress: data.progress || data.data?.progress,
                message: data.message || data.data?.message
              });
              
              // 프론트엔드 형식으로 변환
              const progress: PipelineProgress = {
                type: data.type,
                session_id: data.session_id || sessionId,
                step_id: data.step_id || data.data?.step_id || 0,
                step_name: data.step_name || data.data?.step_name || '',
                progress: data.progress || data.data?.progress || 0,
                message: data.message || data.data?.message || '',
                timestamp: data.timestamp || Date.now() / 1000,
                status: data.data?.status || 'processing',
                data: data.data
              };
              
              onProgress(progress);
            } else if (data.type === 'connection_established') {
              PipelineUtils.log('info', '🤝 WebSocket 연결 확립:', data.client_id);
            } else {
              PipelineUtils.log('info', '📨 기타 WebSocket 메시지:', data.type);
            }
          } catch (error) {
            PipelineUtils.log('error', '❌ WebSocket 메시지 파싱 오류:', error);
          }
        };

        ws.onerror = (error) => {
          if (!connectionResolved) {
            connectionResolved = true;
            clearTimeout(connectionTimer);
            PipelineUtils.log('error', '❌ WebSocket 연결 오류:', error);
            resolve(false);
          }
        };

        ws.onclose = (event) => {
          if (!connectionResolved) {
            connectionResolved = true;
            clearTimeout(connectionTimer);
          }
          
          this.stopHeartbeat();
          this.currentWS = null;
          
          PipelineUtils.log('info', '🔌 WebSocket 연결 종료:', event.code, event.reason);
          
          // 자동 재연결 시도
          if (this.config.autoReconnect && 
              this.connectionAttempts < (this.config.maxReconnectAttempts || 5) && 
              event.code !== 1000) {
            this.scheduleReconnect(sessionId, onProgress);
          }
        };

      } catch (error) {
        PipelineUtils.log('error', '❌ WebSocket 생성 실패:', error);
        resolve(false);
      }
    });
  }

  private scheduleReconnect(sessionId: string, onProgress: (progress: PipelineProgress) => void): void {
    this.connectionAttempts++;
    const delay = (this.config.reconnectInterval || 3000) * this.connectionAttempts;
    
    PipelineUtils.log('info', `🔄 WebSocket 재연결 시도 ${this.connectionAttempts}/${this.config.maxReconnectAttempts} (${delay}ms 후)`);
    
    setTimeout(() => {
      this.setupProgressWebSocket(sessionId, onProgress);
    }, delay);
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.currentWS?.readyState === WebSocket.OPEN) {
        try {
          this.currentWS.send(JSON.stringify({
            type: 'ping',
            timestamp: Date.now()
          }));
        } catch (error) {
          PipelineUtils.log('warn', '⚠️ 하트비트 전송 실패:', error);
        }
      }
    }, 30000);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private closeProgressWebSocket(): void {
    this.config.autoReconnect = false;
    this.stopHeartbeat();
    
    if (this.currentWS) {
      this.currentWS.close(1000, 'Client disconnect');
      this.currentWS = null;
    }
  }

  // ============================================
  // 🔧 추가 API 메서드들 - 프로젝트 지식 기반
  // ============================================

  /**
   * 신체 분석 API
   */
  async analyzeBody(image: File): Promise<BodyAnalysisResponse> {
    try {
      if (!PipelineUtils.validateImageType(image)) {
        throw new Error('지원되지 않는 이미지 형식입니다.');
      }

      const formData = new FormData();
      formData.append('image', image);

      const result = await this.request<BodyAnalysisResponse>(
        '/analyze-body',
        {
          method: 'POST',
          body: formData,
        }
      );

      PipelineUtils.log('info', '✅ 신체 분석 완료');
      return result;

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      PipelineUtils.log('error', '❌ 신체 분석 실패', errorMessage);
      throw new Error(`신체 분석 실패: ${errorMessage}`);
    }
  }

  /**
   * 의류 분석 API
   */
  async analyzeClothing(image: File): Promise<ClothingAnalysisResponse> {
    try {
      if (!PipelineUtils.validateImageType(image)) {
        throw new Error('지원되지 않는 이미지 형식입니다.');
      }

      const formData = new FormData();
      formData.append('image', image);

      const result = await this.request<ClothingAnalysisResponse>(
        '/analyze-clothing',
        {
          method: 'POST',
          body: formData,
        }
      );

      PipelineUtils.log('info', '✅ 의류 분석 완료');
      return result;

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      PipelineUtils.log('error', '❌ 의류 분석 실패', errorMessage);
      throw new Error(`의류 분석 실패: ${errorMessage}`);
    }
  }

  /**
   * 처리 상태 조회
   */
  async getProcessingStatus(taskId: string): Promise<ProcessingStatus> {
    try {
      const result = await this.request<ProcessingStatus>(
        `/processing-status/${taskId}`
      );

      // 로컬 Task 정보 업데이트
      if (this.activeTasks.has(taskId)) {
        const taskInfo = this.activeTasks.get(taskId)!;
        taskInfo.status = result.status;
        taskInfo.progress_percentage = result.progress_percentage;
        taskInfo.current_stage = result.current_stage;
        taskInfo.updated_at = new Date().toISOString();
        this.activeTasks.set(taskId, taskInfo);
      }

      return result;

    } catch (error) {
      PipelineUtils.log('error', '❌ 처리 상태 조회 실패', error);
      throw error;
    }
  }

  /**
   * 파이프라인 상태 조회
   */
  async getPipelineStatus(): Promise<PipelineStatus> {
    try {
      return await this.request<PipelineStatus>('/status');
    } catch (error) {
      PipelineUtils.log('error', '❌ 파이프라인 상태 조회 실패', error);
      throw error;
    }
  }

  /**
   * 파이프라인 초기화
   */
  async initializePipeline(): Promise<boolean> {
    try {
      const response = await this.request('/initialize', {
        method: 'POST',
      });
      return response.initialized || false;
    } catch (error) {
      PipelineUtils.log('error', '❌ 파이프라인 초기화 실패', error);
      return false;
    }
  }

  /**
   * 파이프라인 워밍업
   */
  async warmupPipeline(qualityMode: string = 'balanced'): Promise<boolean> {
    try {
      const formData = new FormData();
      formData.append('quality_mode', qualityMode);

      const response = await this.request('/warmup', {
        method: 'POST',
        body: formData,
      });

      return response.success || false;
    } catch (error) {
      PipelineUtils.log('error', '❌ 파이프라인 워밍업 실패', error);
      return false;
    }
  }

  /**
   * 메모리 상태 조회
   */
  async getMemoryStatus(): Promise<any> {
    try {
      return await this.request('/memory');
    } catch (error) {
      PipelineUtils.log('error', '❌ 메모리 상태 조회 실패', error);
      throw error;
    }
  }

  /**
   * 메모리 정리
   */
  async cleanupMemory(): Promise<boolean> {
    try {
      await this.request('/cleanup', {
        method: 'POST',
      });
      return true;
    } catch (error) {
      PipelineUtils.log('error', '❌ 메모리 정리 실패', error);
      return false;
    }
  }

  /**
   * 모델 정보 조회
   */
  async getModelsInfo(): Promise<any> {
    try {
      return await this.request('/models/info');
    } catch (error) {
      PipelineUtils.log('error', '❌ 모델 정보 조회 실패', error);
      throw error;
    }
  }

  /**
   * 헬스 체크
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.request('/health');
      return response.status === 'healthy';
    } catch {
      return false;
    }
  }

  /**
   * 시스템 통계 조회
   */
  async getSystemStats(): Promise<SystemStats> {
    try {
      return await this.request<SystemStats>('/stats');
    } catch (error) {
      PipelineUtils.log('error', '❌ 시스템 통계 조회 실패', error);
      throw error;
    }
  }

  // ============================================
  // 🔧 브랜드 사이즈 관련 API들
  // ============================================

  async getBrandSizes(brand: string): Promise<BrandSizeData> {
    try {
      return await this.request<BrandSizeData>(
        `/brands/${encodeURIComponent(brand)}/sizes`
      );
    } catch (error) {
      PipelineUtils.log('error', '❌ 브랜드 사이즈 조회 실패', error);
      throw error;
    }
  }

  async getSizeRecommendation(
    measurements: any,
    brand: string, 
    item: string
  ): Promise<SizeRecommendation> {
    try {
      return await this.request<SizeRecommendation>(
        '/size-recommendation',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ measurements, brand, item }),
        }
      );
    } catch (error) {
      PipelineUtils.log('error', '❌ 사이즈 추천 실패', error);
      throw error;
    }
  }

  // ============================================
  // 🔧 헬퍼 메서드들
  // ============================================

  private validateRequest(request: VirtualTryOnRequest): void {
    if (!request.person_image || !request.clothing_image) {
      throw new Error('사용자 이미지와 의류 이미지는 필수입니다.');
    }

    if (request.height <= 0 || request.weight <= 0) {
      throw new Error('유효한 키와 몸무게를 입력해주세요.');
    }

    // 파일 크기 검증 (15MB)
    if (!PipelineUtils.validateFileSize(request.person_image, 15)) {
      throw new Error('사용자 이미지가 15MB를 초과합니다.');
    }
    if (!PipelineUtils.validateFileSize(request.clothing_image, 15)) {
      throw new Error('의류 이미지가 15MB를 초과합니다.');
    }

    // 파일 타입 검증
    if (!PipelineUtils.validateImageType(request.person_image)) {
      throw new Error('사용자 이미지는 JPG, PNG, WebP 형식만 지원됩니다.');
    }
    if (!PipelineUtils.validateImageType(request.clothing_image)) {
      throw new Error('의류 이미지는 JPG, PNG, WebP 형식만 지원됩니다.');
    }
  }

  /**
   * Task 기록 조회
   */
  getTaskHistory(): TaskInfo[] {
    return Array.from(this.activeTasks.values()).sort(
      (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    );
  }

  /**
   * WebSocket 연결 상태 확인
   */
  isWebSocketConnected(): boolean {
    return this.currentWS?.readyState === WebSocket.OPEN;
  }

  /**
   * 연결 통계 조회
   */
  getConnectionStats(): any {
    return {
      wsConnected: this.isWebSocketConnected(),
      connectionAttempts: this.connectionAttempts,
      maxReconnectAttempts: this.config.maxReconnectAttempts,
      autoReconnect: this.config.autoReconnect,
      baseURL: this.baseURL,
      wsURL: this.wsURL,
      activeTasks: this.activeTasks.size,
      device: this.device,
      config: this.config
    };
  }

  /**
   * 설정 업데이트
   */
  updateConfig(newConfig: Partial<ConnectionConfig>): void {
    this.config = { ...this.config, ...newConfig };
    
    if (newConfig.baseURL) {
      this.baseURL = newConfig.baseURL;
    }
    
    if (newConfig.wsURL) {
      this.wsURL = newConfig.wsURL;
    }
    
    PipelineUtils.log('info', '⚙️ PipelineAPIClient 설정 업데이트');
  }

  /**
   * 정리
   */
  async cleanup(): Promise<void> {
    PipelineUtils.log('info', '🧹 PipelineAPIClient: 리소스 정리 중...');
    
    this.closeProgressWebSocket();
    this.activeTasks.clear();
    
    PipelineUtils.log('info', '✅ PipelineAPIClient 리소스 정리 완료');
  }
}

// ============================================
// 🔗 React Hook 형태로 사용할 수 있는 함수
// ============================================

export const usePipelineAPI = (config?: ConnectionConfig) => {
  const apiClient = new PipelineAPIClient(config);

  return {
    // 주요 API 메서드들
    processVirtualTryOn: apiClient.processVirtualTryOn.bind(apiClient),
    analyzeBody: apiClient.analyzeBody.bind(apiClient),
    analyzeClothing: apiClient.analyzeClothing.bind(apiClient),
    getProcessingStatus: apiClient.getProcessingStatus.bind(apiClient),
    
    // 파이프라인 관리
    getPipelineStatus: apiClient.getPipelineStatus.bind(apiClient),
    initializePipeline: apiClient.initializePipeline.bind(apiClient),
    warmupPipeline: apiClient.warmupPipeline.bind(apiClient),
    
    // 메모리 관리
    getMemoryStatus: apiClient.getMemoryStatus.bind(apiClient),
    cleanupMemory: apiClient.cleanupMemory.bind(apiClient),
    
    // 정보 조회
    getModelsInfo: apiClient.getModelsInfo.bind(apiClient),
    healthCheck: apiClient.healthCheck.bind(apiClient),
    getSystemStats: apiClient.getSystemStats.bind(apiClient),
    
    // 브랜드 사이즈
    getBrandSizes: apiClient.getBrandSizes.bind(apiClient),
    getSizeRecommendation: apiClient.getSizeRecommendation.bind(apiClient),
    
    // Task 관리
    getTaskHistory: apiClient.getTaskHistory.bind(apiClient),
    
    // 상태 조회
    isWebSocketConnected: apiClient.isWebSocketConnected.bind(apiClient),
    getConnectionStats: apiClient.getConnectionStats.bind(apiClient),
    
    // 설정 관리
    updateConfig: apiClient.updateConfig.bind(apiClient),
    cleanup: apiClient.cleanup.bind(apiClient),
  };
};

// 기본 export
export default PipelineAPIClient;

// 추가 export들
export { PipelineUtils };