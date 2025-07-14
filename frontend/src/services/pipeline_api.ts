/**
 * MyCloset AI 프론트엔드 API 서비스 - 완전 수정 버전
 * 백엔드 pipeline_routes.py와 완벽 호환
 * WebSocket 연결 안정화 및 에러 처리 강화
 */

// 타입 정의 개선
export interface UserMeasurements {
  height: number;
  weight: number;
  chest?: number;
  waist?: number;
  hip?: number;
}

export interface ClothingAnalysis {
  category: string;
  style: string;
  dominant_color: number[];
  material?: string;
  confidence?: number;
}

export interface QualityMetrics {
  ssim: number;
  lpips: number;
  fid?: number;
  fit_overall: number;
  fit_coverage?: number;
  fit_shape_consistency?: number;
  color_preservation?: number;
  boundary_naturalness?: number;
}

export interface VirtualTryOnRequest {
  person_image: File;
  clothing_image: File;
  height: number;
  weight: number;
  quality_mode?: 'fast' | 'balanced' | 'quality';
  session_id?: string;
}

export interface VirtualTryOnResponse {
  success: boolean;
  process_id?: string;
  fitted_image?: string; // base64
  processing_time: number;
  confidence: number;
  measurements: Record<string, number>;
  clothing_analysis: ClothingAnalysis;
  fit_score: number;
  quality_score?: number;
  recommendations: string[];
  quality_metrics?: QualityMetrics;
  pipeline_stages?: Record<string, any>;
  debug_info?: Record<string, any>;
  memory_usage?: Record<string, number>;
  step_times?: Record<string, number>;
  error?: string;
}

export interface PipelineProgress {
  type: 'pipeline_progress' | 'step_update' | 'completed' | 'error' | 'connection_established';
  session_id?: string;
  step_id?: number;
  step_name?: string;
  progress: number;
  message: string;
  timestamp: number;
  data?: any;
  status?: 'pending' | 'processing' | 'completed' | 'error';
}

export interface PipelineStatus {
  initialized: boolean;
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
}

export interface SystemStats {
  total_requests: number;
  successful_requests: number;
  average_processing_time: number;
  average_quality_score: number;
  peak_memory_usage: number;
  uptime: number;
}

// 연결 설정 인터페이스
export interface ConnectionConfig {
  baseURL?: string;
  wsURL?: string;
  autoReconnect?: boolean;
  maxReconnectAttempts?: number;
  reconnectInterval?: number;
  connectionTimeout?: number;
}

// API 클라이언트 클래스 - 완전 개선
class PipelineAPIClient {
  private baseURL: string;
  private wsURL: string;
  private currentWS: WebSocket | null = null;
  private connectionAttempts: number = 0;
  private maxReconnectAttempts: number = 5;
  private reconnectInterval: number = 3000;
  private connectionTimeout: number = 10000;
  private autoReconnect: boolean = true;
  private heartbeatTimer: NodeJS.Timeout | null = null;

  constructor(config: ConnectionConfig = {}) {
    this.baseURL = config.baseURL || 'http://localhost:8000';
    this.wsURL = config.wsURL || this.baseURL.replace('http', 'ws');
    this.autoReconnect = config.autoReconnect ?? true;
    this.maxReconnectAttempts = config.maxReconnectAttempts || 5;
    this.reconnectInterval = config.reconnectInterval || 3000;
    this.connectionTimeout = config.connectionTimeout || 10000;
  }

  /**
   * 가상 피팅 처리 요청 - 백엔드 API와 완벽 호환
   */
  async processVirtualTryOn(
    request: VirtualTryOnRequest,
    onProgress?: (progress: PipelineProgress) => void
  ): Promise<VirtualTryOnResponse> {
    try {
      // 입력 검증
      this.validateRequest(request);

      // 세션 ID 생성 (없으면)
      const sessionId = request.session_id || `session_${Date.now()}_${Math.random().toString(36).substring(7)}`;

      // WebSocket 연결 설정 (진행률 업데이트용)
      let wsConnected = false;
      if (onProgress) {
        try {
          wsConnected = await this.setupProgressWebSocket(sessionId, onProgress);
          if (!wsConnected) {
            console.warn('⚠️ WebSocket 연결 실패, 진행률 업데이트 없이 계속 진행');
          }
        } catch (error) {
          console.warn('⚠️ WebSocket 설정 실패:', error);
        }
      }

      // FormData 준비 - 백엔드 API 형식에 맞춤
      const formData = new FormData();
      formData.append('person_image', request.person_image);
      formData.append('clothing_image', request.clothing_image);
      formData.append('height', request.height.toString());
      formData.append('weight', request.weight.toString());
      formData.append('quality_mode', request.quality_mode || 'balanced');
      formData.append('enable_realtime', wsConnected ? 'true' : 'false');
      formData.append('session_id', sessionId);

      console.log('🚀 가상 피팅 API 요청 시작...', {
        sessionId,
        qualityMode: request.quality_mode,
        wsConnected
      });

      // API 요청
      const response = await fetch(`${this.baseURL}/api/pipeline/virtual-tryon`, {
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
      
      console.log('✅ 가상 피팅 API 응답 성공:', {
        success: result.success,
        processingTime: result.processing_time,
        fitScore: result.fit_score,
        processId: result.process_id
      });

      return result;

    } catch (error) {
      this.closeProgressWebSocket();
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('❌ 가상 피팅 API 오류:', errorMessage);
      throw new Error(`가상 피팅 처리 실패: ${errorMessage}`);
    }
  }

  /**
   * 입력 검증
   */
  private validateRequest(request: VirtualTryOnRequest): void {
    if (!request.person_image || !request.clothing_image) {
      throw new Error('사용자 이미지와 의류 이미지는 필수입니다.');
    }

    if (request.height <= 0 || request.weight <= 0) {
      throw new Error('유효한 키와 몸무게를 입력해주세요.');
    }

    // 파일 크기 검증 (10MB)
    const maxSize = 10 * 1024 * 1024;
    if (request.person_image.size > maxSize) {
      throw new Error('사용자 이미지가 10MB를 초과합니다.');
    }
    if (request.clothing_image.size > maxSize) {
      throw new Error('의류 이미지가 10MB를 초과합니다.');
    }

    // 파일 타입 검증
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (!allowedTypes.includes(request.person_image.type)) {
      throw new Error('사용자 이미지는 JPG, PNG, WebP 형식만 지원됩니다.');
    }
    if (!allowedTypes.includes(request.clothing_image.type)) {
      throw new Error('의류 이미지는 JPG, PNG, WebP 형식만 지원됩니다.');
    }
  }

  /**
   * 진행률 WebSocket 설정 - 향상된 안정성
   */
  private async setupProgressWebSocket(
    sessionId: string,
    onProgress: (progress: PipelineProgress) => void
  ): Promise<boolean> {
    return new Promise((resolve, reject) => {
      try {
        const wsUrl = `${this.wsURL}/api/ws/pipeline-progress`;
        console.log('🔗 WebSocket 연결 시도:', wsUrl);
        
        const ws = new WebSocket(wsUrl);
        let connectionResolved = false;

        // 연결 타임아웃 설정
        const connectionTimer = setTimeout(() => {
          if (!connectionResolved) {
            connectionResolved = true;
            ws.close();
            console.error('❌ WebSocket 연결 타임아웃');
            resolve(false); // 타임아웃 시 false 반환 (에러가 아님)
          }
        }, this.connectionTimeout);

        ws.onopen = () => {
          if (!connectionResolved) {
            connectionResolved = true;
            clearTimeout(connectionTimer);
            this.currentWS = ws;
            this.connectionAttempts = 0;
            
            console.log('✅ WebSocket 연결 성공');
            
            // 세션 구독 메시지 전송
            try {
              ws.send(JSON.stringify({
                type: 'subscribe_session',
                session_id: sessionId
              }));
              console.log('📡 세션 구독:', sessionId);
            } catch (error) {
              console.warn('⚠️ 세션 구독 실패:', error);
            }

            // 하트비트 시작
            this.startHeartbeat();
            
            resolve(true);
          }
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            
            // 진행률 메시지만 콜백으로 전달
            if (data.type === 'pipeline_progress' || 
                data.type === 'step_update' || 
                data.type === 'completed' || 
                data.type === 'error') {
              
              console.log('📊 진행률 업데이트:', {
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
                status: data.data?.status || 'processing'
              };
              
              onProgress(progress);
            } else {
              console.log('📨 기타 WebSocket 메시지:', data.type);
            }
          } catch (error) {
            console.error('❌ WebSocket 메시지 파싱 오류:', error);
          }
        };

        ws.onerror = (error) => {
          if (!connectionResolved) {
            connectionResolved = true;
            clearTimeout(connectionTimer);
            console.error('❌ WebSocket 연결 오류:', error);
            resolve(false); // 에러 시 false 반환
          }
        };

        ws.onclose = (event) => {
          if (!connectionResolved) {
            connectionResolved = true;
            clearTimeout(connectionTimer);
          }
          
          this.stopHeartbeat();
          this.currentWS = null;
          
          console.log('🔌 WebSocket 연결 종료:', event.code, event.reason);
          
          // 자동 재연결 시도
          if (this.autoReconnect && 
              this.connectionAttempts < this.maxReconnectAttempts && 
              event.code !== 1000) { // 정상 종료가 아닌 경우
            this.scheduleReconnect(sessionId, onProgress);
          }
        };

      } catch (error) {
        console.error('❌ WebSocket 생성 실패:', error);
        resolve(false);
      }
    });
  }

  /**
   * 재연결 스케줄링
   */
  private scheduleReconnect(sessionId: string, onProgress: (progress: PipelineProgress) => void): void {
    this.connectionAttempts++;
    const delay = this.reconnectInterval * this.connectionAttempts;
    
    console.log(`🔄 WebSocket 재연결 시도 ${this.connectionAttempts}/${this.maxReconnectAttempts} (${delay}ms 후)`);
    
    setTimeout(() => {
      this.setupProgressWebSocket(sessionId, onProgress);
    }, delay);
  }

  /**
   * 하트비트 시작
   */
  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.currentWS?.readyState === WebSocket.OPEN) {
        try {
          this.currentWS.send(JSON.stringify({
            type: 'ping',
            timestamp: Date.now()
          }));
        } catch (error) {
          console.warn('⚠️ 하트비트 전송 실패:', error);
        }
      }
    }, 30000); // 30초마다
  }

  /**
   * 하트비트 중지
   */
  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  /**
   * WebSocket 연결 종료
   */
  private closeProgressWebSocket(): void {
    this.autoReconnect = false; // 수동 종료 시 재연결 비활성화
    this.stopHeartbeat();
    
    if (this.currentWS) {
      this.currentWS.close(1000, 'Client disconnect');
      this.currentWS = null;
    }
  }

  /**
   * 파이프라인 상태 조회
   */
  async getPipelineStatus(): Promise<PipelineStatus> {
    try {
      const response = await fetch(`${this.baseURL}/api/pipeline/status`);
      
      if (!response.ok) {
        throw new Error(`상태 조회 실패: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('❌ 파이프라인 상태 조회 실패:', error);
      throw error;
    }
  }

  /**
   * 파이프라인 초기화
   */
  async initializePipeline(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseURL}/api/pipeline/initialize`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`초기화 실패: ${response.status}`);
      }

      const result = await response.json();
      return result.initialized || false;
    } catch (error) {
      console.error('❌ 파이프라인 초기화 실패:', error);
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

      const response = await fetch(`${this.baseURL}/api/pipeline/warmup`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || '워밍업 실패');
      }

      const result = await response.json();
      return result.success || false;
    } catch (error) {
      console.error('❌ 파이프라인 워밍업 실패:', error);
      return false;
    }
  }

  /**
   * 메모리 상태 조회
   */
  async getMemoryStatus(): Promise<any> {
    try {
      const response = await fetch(`${this.baseURL}/api/pipeline/memory`);
      
      if (!response.ok) {
        throw new Error(`메모리 상태 조회 실패: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('❌ 메모리 상태 조회 실패:', error);
      throw error;
    }
  }

  /**
   * 메모리 정리
   */
  async cleanupMemory(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseURL}/api/pipeline/cleanup`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`메모리 정리 실패: ${response.status}`);
      }

      return true;
    } catch (error) {
      console.error('❌ 메모리 정리 실패:', error);
      return false;
    }
  }

  /**
   * 모델 정보 조회
   */
  async getModelsInfo(): Promise<any> {
    try {
      const response = await fetch(`${this.baseURL}/api/pipeline/models/info`);
      
      if (!response.ok) {
        throw new Error(`모델 정보 조회 실패: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('❌ 모델 정보 조회 실패:', error);
      throw error;
    }
  }

  /**
   * 헬스 체크
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseURL}/api/pipeline/health`);
      return response.ok;
    } catch (error) {
      console.error('❌ 헬스체크 실패:', error);
      return false;
    }
  }

  /**
   * 실시간 업데이트 테스트
   */
  async testRealtimeUpdates(
    processId?: string,
    onProgress?: (progress: PipelineProgress) => void
  ): Promise<any> {
    try {
      const testProcessId = processId || `test_${Date.now()}`;
      
      // WebSocket 연결
      if (onProgress) {
        await this.setupProgressWebSocket(testProcessId, onProgress);
      }

      const response = await fetch(`${this.baseURL}/api/pipeline/test/realtime/${testProcessId}`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`테스트 실패: ${response.status}`);
      }

      const result = await response.json();
      
      // 잠시 후 WebSocket 연결 정리
      setTimeout(() => {
        this.closeProgressWebSocket();
      }, 10000); // 10초 후

      return result;
    } catch (error) {
      this.closeProgressWebSocket();
      console.error('❌ 실시간 테스트 실패:', error);
      throw error;
    }
  }

  /**
   * 디버그 정보 조회
   */
  async getDebugConfig(): Promise<any> {
    try {
      const response = await fetch(`${this.baseURL}/api/pipeline/debug/config`);
      
      if (!response.ok) {
        throw new Error(`디버그 정보 조회 실패: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('❌ 디버그 정보 조회 실패:', error);
      throw error;
    }
  }

  /**
   * 파이프라인 재시작 (개발용)
   */
  async restartPipeline(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseURL}/api/pipeline/dev/restart`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`재시작 실패: ${response.status}`);
      }

      const result = await response.json();
      return result.success || false;
    } catch (error) {
      console.error('❌ 파이프라인 재시작 실패:', error);
      return false;
    }
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
      maxReconnectAttempts: this.maxReconnectAttempts,
      autoReconnect: this.autoReconnect,
      baseURL: this.baseURL,
      wsURL: this.wsURL
    };
  }
}

// 유틸리티 함수들
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
   * 파일 크기 검증
   */
  static validateFileSize(file: File, maxSizeMB: number = 10): boolean {
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
      return {
        grade: 'Excellent',
        color: 'text-green-600',
        description: '완벽한 품질'
      };
    } else if (score >= 0.8) {
      return {
        grade: 'Good',
        color: 'text-blue-600',
        description: '우수한 품질'
      };
    } else if (score >= 0.6) {
      return {
        grade: 'Fair',
        color: 'text-yellow-600',
        description: '양호한 품질'
      };
    } else {
      return {
        grade: 'Poor',
        color: 'text-red-600',
        description: '개선 필요'
      };
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
   * 에러 메시지를 사용자 친화적으로 변환
   */
  static getUserFriendlyError(error: string): string {
    const errorMappings: Record<string, string> = {
      'Network Error': '네트워크 연결을 확인해주세요.',
      'timeout': '처리 시간이 초과되었습니다. 다시 시도해주세요.',
      'invalid image': '지원되지 않는 이미지 형식입니다.',
      'file too large': '파일 크기가 너무 큽니다. 10MB 이하로 업로드해주세요.',
      'server error': '서버에 일시적인 문제가 발생했습니다. 잠시 후 다시 시도해주세요.',
      'connection failed': 'WebSocket 연결에 실패했습니다.',
      'pipeline not ready': '파이프라인이 준비되지 않았습니다.',
      'initialization failed': '파이프라인 초기화에 실패했습니다.',
    };

    const lowerError = error.toLowerCase();
    for (const [key, message] of Object.entries(errorMappings)) {
      if (lowerError.includes(key)) {
        return message;
      }
    }

    return '알 수 없는 오류가 발생했습니다. 지원팀에 문의해주세요.';
  }

  /**
   * 진행률을 백분율 문자열로 변환
   */
  static formatProgress(progress: number): string {
    return `${Math.round(Math.max(0, Math.min(100, progress)))}%`;
  }

  /**
   * 세션 ID 생성
   */
  static generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substring(7)}`;
  }

  /**
   * 파일 정보 요약
   */
  static getFileInfo(file: File): {
    name: string;
    size: string;
    type: string;
    valid: boolean;
  } {
    return {
      name: file.name,
      size: PipelineUtils.formatMemoryUsage(file.size),
      type: file.type,
      valid: PipelineUtils.validateImageType(file) && PipelineUtils.validateFileSize(file)
    };
  }

  /**
   * 처리 결과 요약
   */
  static summarizeResult(result: VirtualTryOnResponse): {
    success: boolean;
    processingTime: string;
    qualityGrade: ReturnType<typeof PipelineUtils.getQualityGrade>;
    fitScore: string;
    recommendationCount: number;
  } {
    const qualityScore = result.quality_score || result.fit_score || 0;
    
    return {
      success: result.success,
      processingTime: PipelineUtils.formatProcessingTime(result.processing_time),
      qualityGrade: PipelineUtils.getQualityGrade(qualityScore),
      fitScore: PipelineUtils.formatProgress(result.fit_score * 100),
      recommendationCount: result.recommendations?.length || 0
    };
  }
}

// React Hook 형태로 사용할 수 있는 함수 (선택사항)
export const usePipelineAPI = (config?: ConnectionConfig) => {
  const apiClient = new PipelineAPIClient(config);

  return {
    // 주요 API 메서드들
    processVirtualTryOn: apiClient.processVirtualTryOn.bind(apiClient),
    getPipelineStatus: apiClient.getPipelineStatus.bind(apiClient),
    initializePipeline: apiClient.initializePipeline.bind(apiClient),
    warmupPipeline: apiClient.warmupPipeline.bind(apiClient),
    
    // 메모리 관리
    getMemoryStatus: apiClient.getMemoryStatus.bind(apiClient),
    cleanupMemory: apiClient.cleanupMemory.bind(apiClient),
    
    // 정보 조회
    getModelsInfo: apiClient.getModelsInfo.bind(apiClient),
    healthCheck: apiClient.healthCheck.bind(apiClient),
    getDebugConfig: apiClient.getDebugConfig.bind(apiClient),
    
    // 테스트 및 개발
    testRealtimeUpdates: apiClient.testRealtimeUpdates.bind(apiClient),
    restartPipeline: apiClient.restartPipeline.bind(apiClient),
    
    // 상태 조회
    isWebSocketConnected: apiClient.isWebSocketConnected.bind(apiClient),
    getConnectionStats: apiClient.getConnectionStats.bind(apiClient),
  };
};

// 기본 export
export default PipelineAPIClient;