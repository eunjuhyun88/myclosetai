/**
 * MyCloset AI 파이프라인 API 클라이언트 (완전한 기능형 버전)
 * ✅ 백엔드 API 완전 호환 (모든 엔드포인트 지원)
 * ✅ 8단계 파이프라인 완전 지원
 * ✅ 실시간 진행률 추적
 * ✅ 파일 업로드 및 검증
 * ✅ 브랜드/사이즈 추천 시스템
 * ✅ 메모리 관리 및 최적화
 * ✅ 캐싱 및 재시도 로직
 * ✅ WebSocket 실시간 통신
 * ✅ 에러 처리 및 복구
 */

import type {
  VirtualTryOnRequest,
  VirtualTryOnResponse,
  PipelineStatus,
  SystemStats,
  SystemHealth,
  UsePipelineOptions,
  TaskInfo,
  ProcessingStatus,
  BrandSizeData,
  SizeRecommendation,
  APIError,
  QualityLevel,
  DeviceType,
  PipelineProgress,
  ClothingCategory,
  FabricType,
  StylePreference
} from '../types/pipeline';
import { PipelineUtils } from '../utils/pipelineUtils';

// =================================================================
// 🔧 완전한 API 클라이언트 설정 타입들
// =================================================================

export interface APIClientConfig {
  baseURL: string;
  wsURL?: string;
  apiKey?: string;
  timeout: number;
  retryAttempts: number;
  retryDelay: number;
  enableCaching: boolean;
  cacheTimeout: number;
  enableCompression: boolean;
  enableRetry: boolean;
  uploadChunkSize: number;
  maxConcurrentRequests: number;
  requestQueueSize: number;
  enableMetrics: boolean;
  enableDebug: boolean;
  enableWebSocket: boolean;
  heartbeatInterval: number;
  reconnectInterval: number;
  maxReconnectAttempts: number;
}

export interface RequestMetrics {
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  averageResponseTime: number;
  totalBytesTransferred: number;
  cacheHitRate: number;
  retryRate: number;
  errorBreakdown: Record<string, number>;
  uptime: number;
  lastError?: string;
  lastErrorTime?: number;
}

export interface CacheEntry<T = any> {
  data: T;
  timestamp: number;
  expiry: number;
  size: number;
  hits: number;
  etag?: string;
}

export interface QueuedRequest {
  id: string;
  url: string;
  options: RequestInit;
  resolve: (value: any) => void;
  reject: (reason: any) => void;
  priority: number;
  attempts: number;
  timestamp: number;
  maxRetries: number;
}

// =================================================================
// 🔧 완전한 WebSocket 관리자 클래스
// =================================================================

class EnhancedWebSocketManager {
  private ws: WebSocket | null = null;
  private url: string;
  private protocols?: string[];
  private isConnecting = false;
  private isDestroyed = false;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private connectionTimeout: NodeJS.Timeout | null = null;
  private messageQueue: Array<{ data: any; timestamp: number }> = [];
  private subscriptions = new Set<string>();
  
  // 콜백 관리
  private messageHandlers = new Map<string, Function[]>();
  private eventHandlers = new Map<string, Function[]>();
  
  // 연결 품질 추적
  private latencyMeasurements: number[] = [];
  private lastPingTime = 0;
  private connectionQuality = 0;
  private totalReconnects = 0;
  private lastDisconnectTime = 0;

  constructor(url: string, options: Partial<APIClientConfig> = {}) {
    this.url = url;
    this.protocols = options.wsURL ? [options.wsURL] : undefined;
    this.maxReconnectAttempts = options.maxReconnectAttempts || 10;
    
    console.log('🔧 EnhancedWebSocketManager 생성:', url);
  }

  // 메시지 핸들러 등록
  onMessage(type: string, handler: Function): void {
    if (!this.messageHandlers.has(type)) {
      this.messageHandlers.set(type, []);
    }
    this.messageHandlers.get(type)!.push(handler);
  }

  // 이벤트 핸들러 등록
  onEvent(event: string, handler: Function): void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, []);
    }
    this.eventHandlers.get(event)!.push(handler);
  }

  // 연결 관리
  async connect(): Promise<boolean> {
    if (this.isDestroyed) return false;
    if (this.isConnected()) return true;
    if (this.isConnecting) return false;

    this.isConnecting = true;
    console.log('🔗 WebSocket 연결 시도:', this.url);

    try {
      this.ws = new WebSocket(this.url, this.protocols);
      
      return new Promise((resolve) => {
        if (!this.ws || this.isDestroyed) {
          this.isConnecting = false;
          resolve(false);
          return;
        }

        this.connectionTimeout = setTimeout(() => {
          console.log('⏰ WebSocket 연결 타임아웃');
          this.ws?.close();
          this.isConnecting = false;
          resolve(false);
        }, 15000);

        this.ws.onopen = () => {
          if (this.isDestroyed) return;
          
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          
          if (this.connectionTimeout) {
            clearTimeout(this.connectionTimeout);
            this.connectionTimeout = null;
          }
          
          console.log('✅ WebSocket 연결 성공');
          this.startHeartbeat();
          this.processMessageQueue();
          this.resubscribeAll();
          this.emitEvent('connected');
          resolve(true);
        };

        this.ws.onmessage = (event) => {
          if (this.isDestroyed) return;
          this.handleMessage(event);
        };

        this.ws.onclose = (event) => {
          this.isConnecting = false;
          this.stopHeartbeat();
          this.lastDisconnectTime = Date.now();
          
          if (this.connectionTimeout) {
            clearTimeout(this.connectionTimeout);
            this.connectionTimeout = null;
          }
          
          if (!this.isDestroyed) {
            console.log('🔌 WebSocket 연결 종료:', event.code, event.reason);
            this.emitEvent('disconnected', { code: event.code, reason: event.reason });
            
            if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
              this.scheduleReconnect();
            }
          }
        };

        this.ws.onerror = (error) => {
          this.isConnecting = false;
          
          if (!this.isDestroyed) {
            console.error('❌ WebSocket 오류:', error);
            this.emitEvent('error', error);
          }
          resolve(false);
        };
      });
    } catch (error) {
      this.isConnecting = false;
      console.error('❌ WebSocket 연결 실패:', error);
      return false;
    }
  }

  private handleMessage(event: MessageEvent): void {
    try {
      const data = JSON.parse(event.data);
      
      // 핑퐁 처리
      if (data.type === 'pong') {
        this.handlePong();
        return;
      }
      
      // 메시지 타입별 핸들러 실행
      const handlers = this.messageHandlers.get(data.type) || [];
      handlers.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error('❌ 메시지 핸들러 오류:', error);
        }
      });
      
      // 일반 메시지 핸들러들
      const allHandlers = this.messageHandlers.get('*') || [];
      allHandlers.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error('❌ 일반 메시지 핸들러 오류:', error);
        }
      });
      
    } catch (error) {
      console.error('❌ WebSocket 메시지 파싱 오류:', error);
    }
  }

  private scheduleReconnect(): void {
    if (this.isDestroyed || this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('🚫 재연결 중단');
      return;
    }

    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
    this.reconnectAttempts++;
    this.totalReconnects++;
    
    console.log(`🔄 ${delay}ms 후 재연결 시도 (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
    
    this.reconnectTimer = setTimeout(() => {
      if (!this.isDestroyed) {
        this.connect();
      }
    }, delay);
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    
    this.heartbeatTimer = setInterval(() => {
      if (this.isConnected()) {
        this.lastPingTime = performance.now();
        this.send({ type: 'ping', timestamp: Date.now() });
      }
    }, 30000);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private handlePong(): void {
    if (this.lastPingTime > 0) {
      const latency = performance.now() - this.lastPingTime;
      this.latencyMeasurements.push(latency);
      
      if (this.latencyMeasurements.length > 10) {
        this.latencyMeasurements.shift();
      }
      
      // 연결 품질 계산
      const avgLatency = this.latencyMeasurements.reduce((a, b) => a + b, 0) / this.latencyMeasurements.length;
      this.connectionQuality = Math.max(0, Math.min(100, 100 - (avgLatency / 10)));
      
      this.lastPingTime = 0;
    }
  }

  private processMessageQueue(): void {
    const messages = [...this.messageQueue];
    this.messageQueue = [];
    
    messages.forEach(({ data }) => {
      this.send(data);
    });
  }

  private resubscribeAll(): void {
    this.subscriptions.forEach(sessionId => {
      this.send({
        type: 'subscribe',
        session_id: sessionId,
        timestamp: Date.now()
      });
    });
  }

  private emitEvent(event: string, data?: any): void {
    const handlers = this.eventHandlers.get(event) || [];
    handlers.forEach(handler => {
      try {
        handler(data);
      } catch (error) {
        console.error('❌ 이벤트 핸들러 오류:', error);
      }
    });
  }

  // 공개 메서드들
  isConnected(): boolean {
    return !this.isDestroyed && this.ws?.readyState === WebSocket.OPEN;
  }

  send(data: any): boolean {
    if (!this.isConnected()) {
      // 큐에 추가
      this.messageQueue.push({ data, timestamp: Date.now() });
      return false;
    }

    try {
      this.ws!.send(JSON.stringify(data));
      return true;
    } catch (error) {
      console.error('❌ WebSocket 메시지 전송 실패:', error);
      return false;
    }
  }

  subscribe(sessionId: string): void {
    this.subscriptions.add(sessionId);
    this.send({
      type: 'subscribe',
      session_id: sessionId,
      timestamp: Date.now()
    });
  }

  unsubscribe(sessionId: string): void {
    this.subscriptions.delete(sessionId);
    this.send({
      type: 'unsubscribe',
      session_id: sessionId,
      timestamp: Date.now()
    });
  }

  getConnectionStats(): any {
    return {
      connected: this.isConnected(),
      quality: this.connectionQuality,
      latency: this.latencyMeasurements.length > 0 
        ? Math.round(this.latencyMeasurements.reduce((a, b) => a + b, 0) / this.latencyMeasurements.length)
        : 0,
      reconnectAttempts: this.reconnectAttempts,
      totalReconnects: this.totalReconnects,
      queueSize: this.messageQueue.length,
      subscriptions: this.subscriptions.size,
      uptime: this.lastDisconnectTime > 0 ? Date.now() - this.lastDisconnectTime : 0
    };
  }

  disconnect(): void {
    console.log('🔌 WebSocket 연결 해제');
    
    this.stopHeartbeat();
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.ws && this.ws.readyState !== WebSocket.CLOSED) {
      this.ws.close(1000, 'Normal closure');
    }
    
    this.ws = null;
  }

  cleanup(): void {
    console.log('🧹 EnhancedWebSocketManager 정리 시작');
    
    this.isDestroyed = true;
    this.disconnect();
    
    this.messageHandlers.clear();
    this.eventHandlers.clear();
    this.subscriptions.clear();
    this.messageQueue = [];
    
    console.log('✅ EnhancedWebSocketManager 정리 완료');
  }
}

// =================================================================
// 🔧 메인 PipelineAPIClient 클래스 (완전한 기능형)
// =================================================================

export default class PipelineAPIClient {
  private config: APIClientConfig;
  private defaultHeaders: Record<string, string>;
  private metrics: RequestMetrics;
  private cache: Map<string, CacheEntry> = new Map();
  private requestQueue: QueuedRequest[] = [];
  private activeRequests: Set<string> = new Set();
  private abortControllers: Map<string, AbortController> = new Map();
  private wsManager: EnhancedWebSocketManager | null = null;
  
  // 재시도 및 백오프 관리
  private retryDelays: number[] = [1000, 2000, 4000, 8000, 16000];
  private circuitBreakerFailures = 0;
  private circuitBreakerLastFailure = 0;
  private readonly circuitBreakerThreshold = 5;
  private readonly circuitBreakerTimeout = 60000;
  
  // 업로드 진행률 추적
  private uploadProgressCallbacks: Map<string, (progress: number) => void> = new Map();
  
  // 성능 모니터링
  private startTime = Date.now();
  private lastMetricsUpdate = Date.now();

  constructor(options: UsePipelineOptions = {}, ...kwargs: any[]) {
    this.config = {
      baseURL: options.baseURL || 'http://localhost:8000',
      wsURL: options.wsURL || options.baseURL?.replace('http', 'ws') || 'ws://localhost:8000',
      apiKey: options.apiKey,
      timeout: options.requestTimeout || 60000,
      retryAttempts: options.maxRetryAttempts || 3,
      retryDelay: options.retryDelay || 1000,
      enableCaching: options.enableCaching ?? true,
      cacheTimeout: options.cacheTimeout || 300000,
      enableCompression: options.compressionEnabled ?? true,
      enableRetry: options.enableRetry ?? true,
      uploadChunkSize: 1024 * 1024,
      maxConcurrentRequests: options.maxConcurrentRequests || 3,
      requestQueueSize: 100,
      enableMetrics: true,
      enableDebug: options.enableDebugMode ?? false,
      enableWebSocket: options.enableRealTimeUpdates ?? true,
      heartbeatInterval: options.heartbeatInterval || 30000,
      reconnectInterval: options.reconnectInterval || 3000,
      maxReconnectAttempts: options.maxReconnectAttempts || 10,
    };

    this.defaultHeaders = {
      'Accept': 'application/json',
      'User-Agent': `MyClosetAI-Client/2.0.0 (${navigator.userAgent})`,
      'X-Client-Version': '2.0.0',
      'X-Client-Platform': navigator.platform,
      'X-Request-ID': this.generateRequestId(),
      'X-Session-ID': this.generateSessionId(),
    };

    if (this.config.apiKey) {
      this.defaultHeaders['Authorization'] = `Bearer ${this.config.apiKey}`;
    }

    if (this.config.enableCompression) {
      this.defaultHeaders['Accept-Encoding'] = 'gzip, deflate, br';
    }

    this.metrics = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      averageResponseTime: 0,
      totalBytesTransferred: 0,
      cacheHitRate: 0,
      retryRate: 0,
      errorBreakdown: {},
      uptime: 0,
    };

    this.mergeAdditionalConfig(kwargs);

    PipelineUtils.info('🔧 PipelineAPIClient 초기화', {
      baseURL: this.config.baseURL,
      enableWebSocket: this.config.enableWebSocket,
      enableCaching: this.config.enableCaching,
      timeout: this.config.timeout
    });

    this.startBackgroundTasks();
  }

  // =================================================================
  // 🔧 초기화 및 백그라운드 작업
  // =================================================================

  private mergeAdditionalConfig(kwargs: any[]): void {
    for (const kwarg of kwargs) {
      if (typeof kwarg === 'object' && kwarg !== null) {
        Object.assign(this.config, kwarg);
      }
    }
  }

  private startBackgroundTasks(): void {
    // 주기적 캐시 정리
    setInterval(() => this.cleanupExpiredCache(), 60000);
    
    // 요청 큐 처리
    setInterval(() => this.processRequestQueue(), 100);
    
    // 메트릭 업데이트
    setInterval(() => this.updateMetrics(), 5000);
    
    // WebSocket 초기화 (옵션이 활성화된 경우)
    if (this.config.enableWebSocket) {
      this.initializeWebSocket();
    }
  }

  private initializeWebSocket(): void {
    if (!this.wsManager) {
      const wsUrl = `${this.config.wsURL}/api/ws/pipeline-progress`;
      this.wsManager = new EnhancedWebSocketManager(wsUrl, this.config);
      
      // 기본 메시지 핸들러들 등록
      this.wsManager.onMessage('pipeline_progress', (data: PipelineProgress) => {
        PipelineUtils.emitEvent('pipeline:progress', data);
      });
      
      this.wsManager.onMessage('step_start', (data: any) => {
        PipelineUtils.emitEvent('pipeline:step_start', data);
      });
      
      this.wsManager.onMessage('step_complete', (data: any) => {
        PipelineUtils.emitEvent('pipeline:step_complete', data);
      });
      
      this.wsManager.onMessage('step_error', (data: any) => {
        PipelineUtils.emitEvent('pipeline:step_error', data);
      });
      
      this.wsManager.onEvent('connected', () => {
        PipelineUtils.info('✅ WebSocket 연결됨');
      });
      
      this.wsManager.onEvent('disconnected', () => {
        PipelineUtils.warn('❌ WebSocket 연결 해제됨');
      });
    }
  }

  async initialize(): Promise<boolean> {
    PipelineUtils.info('🔄 PipelineAPIClient 초기화 중...');
    
    try {
      // 헬스체크
      const isHealthy = await this.healthCheck();
      if (!isHealthy) {
        PipelineUtils.error('❌ 서버 헬스체크 실패');
        return false;
      }
      
      // WebSocket 연결 (옵션이 활성화된 경우)
      if (this.config.enableWebSocket && this.wsManager) {
        await this.wsManager.connect();
      }
      
      // 시스템 정보 로드
      try {
        const [serverInfo, features, supportedModels] = await Promise.all([
          this.getServerInfo(),
          this.getSupportedFeatures(),
          this.getModelsInfo()
        ]);
        
        PipelineUtils.info('✅ PipelineAPIClient 초기화 완료', {
          serverVersion: serverInfo.version,
          supportedFeatures: features.length,
          loadedModels: supportedModels.model_info?.currently_loaded || 0
        });
      } catch (error) {
        PipelineUtils.warn('⚠️ 서버 정보 로드 실패', error);
      }
      
      return true;
    } catch (error) {
      PipelineUtils.error('❌ PipelineAPIClient 초기화 중 오류', error);
      return false;
    }
  }

  // =================================================================
  // 🔧 핵심 HTTP 요청 메서드들
  // =================================================================

  private async request<T = any>(
    endpoint: string,
    options: RequestInit = {},
    skipCache: boolean = false
  ): Promise<T> {
    const url = this.buildURL(endpoint);
    const cacheKey = this.generateCacheKey(url, options);
    
    if (this.isCircuitBreakerOpen()) {
      throw this.createAPIError('circuit_breaker_open', 'Circuit breaker is open');
    }

    if (!skipCache && this.config.enableCaching && options.method !== 'POST') {
      const cached = this.getFromCache<T>(cacheKey);
      if (cached) {
        this.updateCacheMetrics(true);
        return cached;
      }
    }

    this.updateCacheMetrics(false);

    if (this.activeRequests.size >= this.config.maxConcurrentRequests) {
      return this.queueRequest<T>(url, options);
    }

    return this.executeRequest<T>(url, options, cacheKey);
  }

  private async executeRequest<T>(
    url: string,
    options: RequestInit,
    cacheKey: string,
    attemptNum: number = 1
  ): Promise<T> {
    const requestId = this.generateRequestId();
    const timer = PipelineUtils.createPerformanceTimer(`API Request: ${url}`);
    
    try {
      this.activeRequests.add(requestId);
      this.metrics.totalRequests++;

      const abortController = new AbortController();
      this.abortControllers.set(requestId, abortController);
      
      const timeoutId = setTimeout(() => {
        abortController.abort();
        PipelineUtils.warn('⏰ 요청 타임아웃', { url, timeout: this.config.timeout });
      }, this.config.timeout);

      const requestOptions: RequestInit = {
        ...options,
        headers: {
          ...this.defaultHeaders,
          ...options.headers,
          'X-Request-ID': requestId,
          'X-Attempt-Number': attemptNum.toString(),
          'X-Timestamp': Date.now().toString(),
        },
        signal: abortController.signal,
      };

      if (options.body instanceof FormData) {
        delete requestOptions.headers!['Content-Type'];
      }

      if (this.config.enableDebug) {
        PipelineUtils.debug('🌐 API 요청 시작', {
          url,
          method: requestOptions.method || 'GET',
          requestId,
          attempt: attemptNum
        });
      }

      const response = await fetch(url, requestOptions);
      clearTimeout(timeoutId);

      const duration = timer.end();
      this.updateResponseTimeMetrics(duration);

      const result = await this.processResponse<T>(response, requestId);
      
      if (this.config.enableCaching && requestOptions.method !== 'POST') {
        this.saveToCache(cacheKey, result, this.calculateCacheSize(result), response.headers.get('etag'));
      }

      this.metrics.successfulRequests++;
      this.resetCircuitBreaker();

      if (this.config.enableDebug) {
        PipelineUtils.debug('✅ API 요청 성공', {
          url,
          requestId,
          duration: `${duration}ms`
        });
      }

      return result;

    } catch (error: any) {
      const duration = timer.end();
      this.updateResponseTimeMetrics(duration);
      
      return this.handleRequestError<T>(error, url, options, cacheKey, attemptNum);
      
    } finally {
      this.activeRequests.delete(requestId);
      this.abortControllers.delete(requestId);
      this.processRequestQueue();
    }
  }

  private async processResponse<T>(response: Response, requestId: string): Promise<T> {
    const contentLength = response.headers.get('content-length');
    if (contentLength) {
      this.metrics.totalBytesTransferred += parseInt(contentLength);
    }

    if (!response.ok) {
      const errorData = await this.parseErrorResponse(response);
      
      PipelineUtils.error('❌ HTTP 오류 응답', {
        status: response.status,
        statusText: response.statusText,
        url: response.url,
        errorData,
        requestId
      });

      throw this.createAPIError(
        `http_${response.status}`,
        errorData.message || response.statusText,
        errorData,
        this.getRetryAfter(response)
      );
    }

    const contentType = response.headers.get('content-type');
    if (contentType?.includes('application/json')) {
      return await response.json();
    } else if (contentType?.includes('text/')) {
      return await response.text() as unknown as T;
    } else {
      return await response.blob() as unknown as T;
    }
  }

  // =================================================================
  // 🔧 메인 API 메서드들 (모든 백엔드 엔드포인트 지원)
  // =================================================================

  // ===== 가상 피팅 API =====
  async processVirtualTryOn(
    request: VirtualTryOnRequest,
    onProgress?: (progress: PipelineProgress) => void
  ): Promise<VirtualTryOnResponse> {
    const timer = PipelineUtils.createPerformanceTimer('가상 피팅 API 전체 처리');

    try {
      this.validateVirtualTryOnRequest(request);

      const formData = this.buildVirtualTryOnFormData(request);
      const requestId = this.generateRequestId();
      
      if (onProgress) {
        this.uploadProgressCallbacks.set(requestId, (progress: number) => {
          onProgress({
            type: 'upload_progress',
            progress,
            message: `업로드 중... ${progress}%`,
            timestamp: Date.now()
          });
        });
      }

      // WebSocket 세션 구독 (WebSocket이 연결된 경우)
      if (this.wsManager && this.wsManager.isConnected()) {
        this.wsManager.subscribe(request.session_id || requestId);
      }

      const result = await this.uploadWithProgress<VirtualTryOnResponse>(
        '/api/virtual-tryon',
        formData,
        requestId,
        onProgress
      );

      const duration = timer.end();
      
      PipelineUtils.info('✅ 가상 피팅 API 성공', {
        processingTime: duration / 1000,
        fitScore: result.fit_score,
        confidence: result.confidence
      });

      return result;

    } catch (error: any) {
      timer.end();
      const friendlyError = PipelineUtils.getUserFriendlyError(error);
      PipelineUtils.error('❌ 가상 피팅 API 실패', friendlyError);
      throw error;
    }
  }

  private buildVirtualTryOnFormData(request: VirtualTryOnRequest): FormData {
    const formData = new FormData();
    
    // 필수 파일들
    formData.append('person_image', request.person_image);
    formData.append('clothing_image', request.clothing_image);
    
    // 신체 측정값
    formData.append('height', request.height.toString());
    formData.append('weight', request.weight.toString());
    
    // 선택적 측정값들
    if (request.chest) formData.append('chest', request.chest.toString());
    if (request.waist) formData.append('waist', request.waist.toString());
    if (request.hip) formData.append('hip', request.hip.toString());
    if (request.shoulder_width) formData.append('shoulder_width', request.shoulder_width.toString());
    
    // 의류 정보
    formData.append('clothing_type', request.clothing_type || 'upper_body');
    formData.append('fabric_type', request.fabric_type || 'cotton');
    formData.append('style_preference', request.style_preference || 'regular');
    
    // 처리 옵션
    formData.append('quality_mode', request.quality_mode || 'balanced');
    formData.append('session_id', request.session_id || this.generateSessionId());
    formData.append('enable_realtime', String(request.enable_realtime || false));
    formData.append('save_intermediate', String(request.save_intermediate || false));
    
    // 고급 옵션들
    if (request.pose_adjustment !== undefined) {
      formData.append('pose_adjustment', String(request.pose_adjustment));
    }
    if (request.color_preservation !== undefined) {
      formData.append('color_preservation', String(request.color_preservation));
    }
    if (request.texture_enhancement !== undefined) {
      formData.append('texture_enhancement', String(request.texture_enhancement));
    }
    
    // 시스템 파라미터
    const systemParams = PipelineUtils.getSystemParams();
    for (const [key, value] of systemParams) {
      formData.append(key, String(value));
    }
    
    // 메타데이터
    formData.append('client_version', '2.0.0');
    formData.append('platform', navigator.platform);
    formData.append('timestamp', new Date().toISOString());
    formData.append('user_agent', navigator.userAgent);
    
    return formData;
  }

  // ===== 개별 분석 API들 =====
  async analyzeBody(image: File): Promise<any> {
    this.validateImageFile(image);

    const formData = new FormData();
    formData.append('image', image);
    formData.append('analysis_type', 'body_parsing');
    formData.append('detail_level', 'high');

    return await this.request('/api/analyze-body', {
      method: 'POST',
      body: formData,
    });
  }

  async analyzeClothing(image: File): Promise<any> {
    this.validateImageFile(image);

    const formData = new FormData();
    formData.append('image', image);
    formData.append('analysis_type', 'clothing_segmentation');
    formData.append('extract_features', 'true');

    return await this.request('/api/analyze-clothing', {
      method: 'POST',
      body: formData,
    });
  }

  async analyzePose(image: File): Promise<any> {
    this.validateImageFile(image);

    const formData = new FormData();
    formData.append('image', image);
    formData.append('pose_model', 'openpose');
    formData.append('keypoints', '18');

    return await this.request('/api/analyze-pose', {
      method: 'POST',
      body: formData,
    });
  }

  async extractBackground(image: File): Promise<any> {
    this.validateImageFile(image);

    const formData = new FormData();
    formData.append('image', image);
    formData.append('model', 'u2net');
    formData.append('output_format', 'png');

    return await this.request('/api/extract-background', {
      method: 'POST',
      body: formData,
    });
  }

  // ===== Task 관리 API들 =====
  async getTaskStatus(taskId: string): Promise<ProcessingStatus> {
    return await this.request(`/api/tasks/${taskId}/status`);
  }

  async cancelTask(taskId: string): Promise<boolean> {
    try {
      await this.request(`/api/tasks/${taskId}/cancel`, {
        method: 'POST',
      });
      return true;
    } catch (error) {
      PipelineUtils.error('❌ Task 취소 실패', { taskId, error });
      return false;
    }
  }

  async retryTask(taskId: string): Promise<boolean> {
    try {
      await this.request(`/api/tasks/${taskId}/retry`, {
        method: 'POST',
      });
      return true;
    } catch (error) {
      PipelineUtils.error('❌ Task 재시도 실패', { taskId, error });
      return false;
    }
  }

  async getTaskHistory(limit: number = 50, status?: string): Promise<TaskInfo[]> {
    const params = new URLSearchParams({
      limit: limit.toString()
    });
    
    if (status) {
      params.append('status', status);
    }
    
    return await this.request(`/api/tasks/history?${params.toString()}`);
  }

  async getProcessingQueue(): Promise<any> {
    return await this.request('/api/tasks/queue');
  }

  async clearTaskHistory(): Promise<boolean> {
    try {
      await this.request('/api/tasks/clear-history', {
        method: 'POST',
      });
      return true;
    } catch (error) {
      PipelineUtils.error('❌ Task 히스토리 정리 실패', error);
      return false;
    }
  }

  // ===== 시스템 상태 API들 =====
  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.request('/health', {}, true);
      return response.status === 'healthy';
    } catch (error) {
      PipelineUtils.debug('❌ 헬스체크 실패', error);
      return false;
    }
  }

  async getSystemHealth(): Promise<SystemHealth> {
    return await this.request('/health/detailed', {}, true);
  }

  async getSystemStats(): Promise<SystemStats> {
    return await this.request('/stats');
  }

  async getPipelineStatus(): Promise<PipelineStatus> {
    return await this.request('/api/pipeline/status', {}, true);
  }

  async getServerInfo(): Promise<any> {
    return await this.request('/info');
  }

  async getSystemLogs(level?: string, limit?: number): Promise<any> {
    const params = new URLSearchParams();
    if (level) params.append('level', level);
    if (limit) params.append('limit', limit.toString());
    
    return await this.request(`/api/system/logs?${params.toString()}`);
  }

  // ===== 파이프라인 관리 API들 =====
  async initializePipeline(): Promise<boolean> {
    try {
      const response = await this.request('/api/pipeline/initialize', {
        method: 'POST',
      });
      return response.success || false;
    } catch (error) {
      PipelineUtils.error('❌ 파이프라인 초기화 실패', error);
      return false;
    }
  }

  async warmupPipeline(qualityMode: QualityLevel = 'balanced'): Promise<boolean> {
    try {
      const formData = new FormData();
      formData.append('quality_mode', qualityMode);
      
      const systemParams = PipelineUtils.getSystemParams();
      for (const [key, value] of systemParams) {
        formData.append(key, String(value));
      }

      const response = await this.request('/api/pipeline/warmup', {
        method: 'POST',
        body: formData,
      });

      return response.success || false;
    } catch (error) {
      PipelineUtils.error('❌ 파이프라인 워밍업 실패', error);
      return false;
    }
  }

  async getMemoryStatus(): Promise<any> {
    return await this.request('/api/pipeline/memory');
  }

  async cleanupMemory(): Promise<boolean> {
    try {
      await this.request('/api/pipeline/cleanup', {
        method: 'POST',
      });
      return true;
    } catch (error) {
      PipelineUtils.error('❌ 메모리 정리 실패', error);
      return false;
    }
  }

  async getModelsInfo(): Promise<any> {
    return await this.request('/api/pipeline/models');
  }

  async loadModel(modelName: string): Promise<boolean> {
    try {
      const response = await this.request('/api/pipeline/load-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: modelName }),
      });
      return response.success || false;
    } catch (error) {
      PipelineUtils.error('❌ 모델 로드 실패', { modelName, error });
      return false;
    }
  }

  async unloadModel(modelName: string): Promise<boolean> {
    try {
      const response = await this.request('/api/pipeline/unload-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: modelName }),
      });
      return response.success || false;
    } catch (error) {
      PipelineUtils.error('❌ 모델 언로드 실패', { modelName, error });
      return false;
    }
  }

  async getSupportedFeatures(): Promise<string[]> {
    const response = await this.request('/api/features');
    return response.features || [];
  }

  async getPerformanceMetrics(): Promise<any> {
    return await this.request('/api/pipeline/performance');
  }

  // ===== 브랜드 및 사이즈 API들 =====
  async getBrandSizes(brand: string): Promise<BrandSizeData> {
    return await this.request(`/api/brands/${encodeURIComponent(brand)}/sizes`);
  }

  async getSizeRecommendation(
    measurements: any,
    brand: string,
    item: string
  ): Promise<SizeRecommendation> {
    return await this.request('/api/size-recommendation', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        measurements,
        brand,
        item
      }),
    });
  }

  async getBrandCompatibility(measurements: any): Promise<any> {
    return await this.request('/api/brand-compatibility', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ measurements }),
    });
  }

  async searchBrands(query: string): Promise<string[]> {
    const response = await this.request(`/api/brands/search?q=${encodeURIComponent(query)}`);
    return response.brands || [];
  }

  async getAllBrands(): Promise<string[]> {
    const response = await this.request('/api/brands');
    return response.brands || [];
  }

  async addCustomBrand(brandData: any): Promise<boolean> {
    try {
      await this.request('/api/brands', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(brandData),
      });
      return true;
    } catch (error) {
      PipelineUtils.error('❌ 브랜드 추가 실패', error);
      return false;
    }
  }

  // ===== 파일 및 미디어 관리 =====
  async uploadFile(file: File, type: string = 'image'): Promise<any> {
    this.validateImageFile(file);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('type', type);
    formData.append('timestamp', Date.now().toString());

    return await this.request('/api/files/upload', {
      method: 'POST',
      body: formData,
    });
  }

  async getFileInfo(fileId: string): Promise<any> {
    return await this.request(`/api/files/${fileId}`);
  }

  async deleteFile(fileId: string): Promise<boolean> {
    try {
      await this.request(`/api/files/${fileId}`, {
        method: 'DELETE',
      });
      return true;
    } catch (error) {
      PipelineUtils.error('❌ 파일 삭제 실패', { fileId, error });
      return false;
    }
  }

  async getFileList(type?: string, limit?: number): Promise<any[]> {
    const params = new URLSearchParams();
    if (type) params.append('type', type);
    if (limit) params.append('limit', limit.toString());
    
    const response = await this.request(`/api/files?${params.toString()}`);
    return response.files || [];
  }

  // ===== 설정 및 프로필 관리 =====
  async getUserProfile(): Promise<any> {
    return await this.request('/api/user/profile');
  }

  async updateUserProfile(profile: any): Promise<boolean> {
    try {
      await this.request('/api/user/profile', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(profile),
      });
      return true;
    } catch (error) {
      PipelineUtils.error('❌ 프로필 업데이트 실패', error);
      return false;
    }
  }

  async getUserPreferences(): Promise<any> {
    return await this.request('/api/user/preferences');
  }

  async updateUserPreferences(preferences: any): Promise<boolean> {
    try {
      await this.request('/api/user/preferences', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(preferences),
      });
      return true;
    } catch (error) {
      PipelineUtils.error('❌ 설정 업데이트 실패', error);
      return false;
    }
  }

  // ===== 피드백 및 평가 시스템 =====
  async submitFeedback(feedback: any): Promise<boolean> {
    try {
      await this.request('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...feedback,
          timestamp: Date.now(),
          client_version: '2.0.0'
        }),
      });
      return true;
    } catch (error) {
      PipelineUtils.error('❌ 피드백 제출 실패', error);
      return false;
    }
  }

  async rateFitResult(sessionId: string, rating: number, comments?: string): Promise<boolean> {
    try {
      await this.request('/api/feedback/rating', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          rating,
          comments,
          timestamp: Date.now()
        }),
      });
      return true;
    } catch (error) {
      PipelineUtils.error('❌ 평가 제출 실패', error);
      return false;
    }
  }

  async getFeedbackHistory(): Promise<any[]> {
    const response = await this.request('/api/feedback/history');
    return response.feedback || [];
  }

  // =================================================================
  // 🔧 유틸리티 메서드들 (완전한 기능형)
  // =================================================================

  private async uploadWithProgress<T>(
    endpoint: string,
    formData: FormData,
    requestId: string,
    onProgress?: (progress: PipelineProgress) => void
  ): Promise<T> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const url = this.buildURL(endpoint);

      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          const progress = Math.round((event.loaded / event.total) * 100);
          const callback = this.uploadProgressCallbacks.get(requestId);
          if (callback) {
            callback(progress);
          }
          
          if (this.config.enableDebug) {
            PipelineUtils.debug('📤 업로드 진행률', {
              requestId,
              progress,
              loaded: PipelineUtils.formatBytes(event.loaded),
              total: PipelineUtils.formatBytes(event.total)
            });
          }
        }
      });

      xhr.addEventListener('load', () => {
        this.uploadProgressCallbacks.delete(requestId);
        
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const result = JSON.parse(xhr.responseText);
            
            if (this.config.enableDebug) {
              PipelineUtils.debug('✅ 업로드 응답 수신', {
                requestId,
                status: xhr.status,
                responseSize: xhr.responseText.length
              });
            }
            
            resolve(result);
          } catch (error) {
            PipelineUtils.error('❌ JSON 파싱 실패', {
              requestId,
              responseText: xhr.responseText.substring(0, 200) + '...',
              error
            });
            reject(new Error('Invalid JSON response'));
          }
        } else {
          try {
            const errorData = JSON.parse(xhr.responseText);
            PipelineUtils.error('❌ 업로드 HTTP 오류', {
              requestId,
              status: xhr.status,
              errorData
            });
            reject(this.createAPIError(
              `http_${xhr.status}`,
              errorData.message || xhr.statusText,
              errorData
            ));
          } catch {
            reject(this.createAPIError(`http_${xhr.status}`, xhr.statusText));
          }
        }
      });

      xhr.addEventListener('error', (event) => {
        this.uploadProgressCallbacks.delete(requestId);
        PipelineUtils.error('❌ 업로드 네트워크 오류', { requestId, event });
        reject(new Error('Network error during upload'));
      });

      xhr.addEventListener('timeout', () => {
        this.uploadProgressCallbacks.delete(requestId);
        PipelineUtils.error('❌ 업로드 타임아웃', { requestId, timeout: this.config.timeout });
        reject(new Error('Upload timeout'));
      });

      xhr.addEventListener('abort', () => {
        this.uploadProgressCallbacks.delete(requestId);
        PipelineUtils.warn('⚠️ 업로드 취소됨', { requestId });
        reject(new Error('Upload aborted'));
      });

      xhr.timeout = this.config.timeout;
      xhr.open('POST', url);

      for (const [key, value] of Object.entries(this.defaultHeaders)) {
        if (key !== 'Content-Type') {
          xhr.setRequestHeader(key, value);
        }
      }
      xhr.setRequestHeader('X-Request-ID', requestId);

      if (this.config.enableDebug) {
        PipelineUtils.debug('📤 업로드 시작', {
          requestId,
          url,
          timeout: this.config.timeout
        });
      }

      xhr.send(formData);
    });
  }

  private validateVirtualTryOnRequest(request: VirtualTryOnRequest): void {
    if (!request.person_image || !request.clothing_image) {
      throw this.createAPIError('validation_error', '사용자 이미지와 의류 이미지는 필수입니다.');
    }

    this.validateImageFile(request.person_image, '사용자 이미지');
    this.validateImageFile(request.clothing_image, '의류 이미지');

    if (request.height <= 0 || request.height > 300) {
      throw this.createAPIError('validation_error', '키는 1-300cm 범위여야 합니다.');
    }

    if (request.weight <= 0 || request.weight > 500) {
      throw this.createAPIError('validation_error', '몸무게는 1-500kg 범위여야 합니다.');
    }
  }

  private validateImageFile(file: File, fieldName: string = '이미지'): void {
    if (!PipelineUtils.validateImageType(file)) {
      throw this.createAPIError('invalid_file', `${fieldName}: 지원되지 않는 파일 형식입니다. JPG, PNG, WebP 파일을 사용해주세요.`);
    }

    if (!PipelineUtils.validateFileSize(file, 50)) {
      throw this.createAPIError('file_too_large', `${fieldName}: 파일 크기가 너무 큽니다. 50MB 이하의 파일을 사용해주세요.`);
    }
  }

  // =================================================================
  // 🔧 캐싱 시스템 (완전한 기능형)
  // =================================================================

  private generateCacheKey(url: string, options: RequestInit): string {
    const method = options.method || 'GET';
    const headers = JSON.stringify(options.headers || {});
    const body = options.body instanceof FormData ? 'FormData' : JSON.stringify(options.body || '');
    return `${method}:${url}:${headers}:${body}`;
  }

  private getFromCache<T>(key: string): T | null {
    const entry = this.cache.get(key);
    if (!entry) return null;

    if (Date.now() > entry.expiry) {
      this.cache.delete(key);
      return null;
    }

    entry.hits++;
    return entry.data;
  }

  private saveToCache<T>(key: string, data: T, size: number, etag?: string | null): void {
    const expiry = Date.now() + this.config.cacheTimeout;
    
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      expiry,
      size,
      hits: 0,
      etag: etag || undefined
    });

    if (this.cache.size > 200) {
      this.evictOldestCacheEntries();
    }
  }

  private evictOldestCacheEntries(): void {
    const entries = Array.from(this.cache.entries());
    entries.sort((a, b) => {
      // LRU: 히트 수와 타임스탬프를 고려
      const scoreA = a[1].hits / (Date.now() - a[1].timestamp);
      const scoreB = b[1].hits / (Date.now() - b[1].timestamp);
      return scoreA - scoreB;
    });
    
    const toRemove = Math.min(50, entries.length);
    for (let i = 0; i < toRemove; i++) {
      this.cache.delete(entries[i][0]);
    }
  }

  private cleanupExpiredCache(): void {
    const now = Date.now();
    const expiredKeys: string[] = [];
    
    for (const [key, entry] of this.cache.entries()) {
      if (now > entry.expiry) {
        expiredKeys.push(key);
      }
    }
    
    for (const key of expiredKeys) {
      this.cache.delete(key);
    }
    
    if (expiredKeys.length > 0) {
      PipelineUtils.debug('🗑️ 만료된 캐시 항목 정리됨', { count: expiredKeys.length });
    }
  }

  clearCache(): void {
    this.cache.clear();
    PipelineUtils.info('🗑️ 캐시 전체 정리됨');
  }

  getCacheStats(): {
    size: number;
    hitRate: number;
    totalSize: number;
    oldestEntry: number;
    newestEntry: number;
    memoryUsage: number;
  } {
    let totalSize = 0;
    let oldestTimestamp = Date.now();
    let newestTimestamp = 0;
    let totalHits = 0;
    let totalRequests = 0;

    for (const entry of this.cache.values()) {
      totalSize += entry.size;
      totalHits += entry.hits;
      totalRequests += entry.hits + 1;
      oldestTimestamp = Math.min(oldestTimestamp, entry.timestamp);
      newestTimestamp = Math.max(newestTimestamp, entry.timestamp);
    }

    return {
      size: this.cache.size,
      hitRate: totalRequests > 0 ? totalHits / totalRequests : 0,
      totalSize,
      oldestEntry: oldestTimestamp,
      newestEntry: newestTimestamp,
      memoryUsage: totalSize
    };
  }

  // =================================================================
  // 🔧 요청 큐잉 시스템 (완전한 기능형)
  // =================================================================

  private async queueRequest<T>(url: string, options: RequestInit): Promise<T> {
    return new Promise((resolve, reject) => {
      const queuedRequest: QueuedRequest = {
        id: this.generateRequestId(),
        url,
        options,
        resolve,
        reject,
        priority: this.getRequestPriority(url),
        attempts: 0,
        timestamp: Date.now(),
        maxRetries: this.config.retryAttempts
      };

      if (this.requestQueue.length >= this.config.requestQueueSize) {
        reject(new Error('Request queue is full'));
        return;
      }

      this.requestQueue.push(queuedRequest);
      this.requestQueue.sort((a, b) => b.priority - a.priority);

      PipelineUtils.debug('📥 요청이 큐에 추가됨', {
        requestId: queuedRequest.id,
        queueSize: this.requestQueue.length,
        priority: queuedRequest.priority
      });
    });
  }

  private getRequestPriority(url: string): number {
    if (url.includes('/health')) return 10;
    if (url.includes('/virtual-tryon')) return 9;
    if (url.includes('/pipeline/status')) return 8;
    if (url.includes('/tasks/')) return 7;
    if (url.includes('/analyze-')) return 6;
    if (url.includes('/brands')) return 4;
    if (url.includes('/stats')) return 3;
    if (url.includes('/files')) return 2;
    return 5;
  }

  private processRequestQueue(): void {
    while (
      this.requestQueue.length > 0 && 
      this.activeRequests.size < this.config.maxConcurrentRequests
    ) {
      const queuedRequest = this.requestQueue.shift()!;
      
      this.executeRequest(
        queuedRequest.url,
        queuedRequest.options,
        this.generateCacheKey(queuedRequest.url, queuedRequest.options)
      )
        .then(queuedRequest.resolve)
        .catch(queuedRequest.reject);
    }
  }

  getQueueStats(): {
    queueSize: number;
    activeRequests: number;
    averageWaitTime: number;
    priorityDistribution: Record<number, number>;
    totalProcessed: number;
  } {
    const now = Date.now();
    let totalWaitTime = 0;
    const priorityDistribution: Record<number, number> = {};

    for (const request of this.requestQueue) {
      totalWaitTime += now - request.timestamp;
      priorityDistribution[request.priority] = (priorityDistribution[request.priority] || 0) + 1;
    }

    return {
      queueSize: this.requestQueue.length,
      activeRequests: this.activeRequests.size,
      averageWaitTime: this.requestQueue.length > 0 ? totalWaitTime / this.requestQueue.length : 0,
      priorityDistribution,
      totalProcessed: this.metrics.totalRequests
    };
  }

  // =================================================================
  // 🔧 서킷 브레이커 패턴 (완전한 기능형)
  // =================================================================

  private isCircuitBreakerOpen(): boolean {
    const now = Date.now();
    
    if (this.circuitBreakerFailures >= this.circuitBreakerThreshold) {
      if (now - this.circuitBreakerLastFailure < this.circuitBreakerTimeout) {
        return true;
      } else {
        this.resetCircuitBreaker();
        return false;
      }
    }
    
    return false;
  }

  private incrementCircuitBreakerFailures(): void {
    this.circuitBreakerFailures++;
    this.circuitBreakerLastFailure = Date.now();
    
    if (this.circuitBreakerFailures >= this.circuitBreakerThreshold) {
      PipelineUtils.warn('⚠️ 서킷 브레이커 활성화됨', {
        failures: this.circuitBreakerFailures,
        threshold: this.circuitBreakerThreshold
      });
    }
  }

  private resetCircuitBreaker(): void {
    if (this.circuitBreakerFailures > 0) {
      PipelineUtils.info('✅ 서킷 브레이커 리셋됨');
      this.circuitBreakerFailures = 0;
      this.circuitBreakerLastFailure = 0;
    }
  }

  getCircuitBreakerStatus(): {
    isOpen: boolean;
    failures: number;
    threshold: number;
    timeUntilReset: number;
    lastFailureTime: number;
  } {
    const now = Date.now();
    const timeUntilReset = this.isCircuitBreakerOpen() 
      ? this.circuitBreakerTimeout - (now - this.circuitBreakerLastFailure)
      : 0;

    return {
      isOpen: this.isCircuitBreakerOpen(),
      failures: this.circuitBreakerFailures,
      threshold: this.circuitBreakerThreshold,
      timeUntilReset: Math.max(0, timeUntilReset),
      lastFailureTime: this.circuitBreakerLastFailure
    };
  }

  // =================================================================
  // 🔧 재시도 로직 (완전한 기능형)
  // =================================================================

  private shouldRetry(error: any, attemptNum: number): boolean {
    if (!this.config.enableRetry || attemptNum >= this.config.retryAttempts) {
      return false;
    }

    const errorCode = this.getErrorCode(error);
    
    const nonRetryableErrors = [
      'http_400', 'http_401', 'http_403', 'http_404', 
      'http_422', 'validation_error', 'invalid_file'
    ];
    
    if (nonRetryableErrors.includes(errorCode)) {
      return false;
    }

    const retryableErrors = [
      'http_500', 'http_502', 'http_503', 'http_504',
      'network_error', 'timeout', 'connection_failed'
    ];
    
    return retryableErrors.includes(errorCode) || error.name === 'AbortError';
  }

  private calculateRetryDelay(attemptNum: number): number {
    const baseDelay = this.config.retryDelay;
    const exponentialDelay = Math.min(
      baseDelay * Math.pow(2, attemptNum - 1),
      this.retryDelays[Math.min(attemptNum - 1, this.retryDelays.length - 1)]
    );
    
    const jitter = exponentialDelay * 0.25 * (Math.random() - 0.5);
    return Math.max(1000, exponentialDelay + jitter);
  }

  private async handleRequestError<T>(
    error: any,
    url: string,
    options: RequestInit,
    cacheKey: string,
    attemptNum: number
  ): Promise<T> {
    this.metrics.failedRequests++;
    this.incrementCircuitBreakerFailures();

    const errorCode = this.getErrorCode(error);
    this.updateErrorMetrics(errorCode);

    PipelineUtils.error('❌ API 요청 실패', {
      url,
      error: error.message,
      attempt: attemptNum,
      errorCode
    });

    if (this.shouldRetry(error, attemptNum)) {
      const delay = this.calculateRetryDelay(attemptNum);
      
      PipelineUtils.info(`🔄 재시도 예약됨 (${attemptNum}/${this.config.retryAttempts})`, {
        delay,
        url
      });

      this.metrics.retryRate = (this.metrics.retryRate * (this.metrics.totalRequests - 1) + 1) / this.metrics.totalRequests;
      
      await PipelineUtils.sleep(delay);
      return this.executeRequest<T>(url, options, cacheKey, attemptNum + 1);
    }

    throw error;
  }

  // =================================================================
  // 🔧 메트릭 및 성능 추적 (완전한 기능형)
  // =================================================================

  private updateResponseTimeMetrics(duration: number): void {
    const currentAvg = this.metrics.averageResponseTime;
    const totalRequests = this.metrics.totalRequests;
    
    this.metrics.averageResponseTime = 
      (currentAvg * (totalRequests - 1) + duration) / totalRequests;
  }

  private updateCacheMetrics(isHit: boolean): void {
    const totalCacheRequests = this.metrics.successfulRequests + this.metrics.failedRequests;
    const currentHitRate = this.metrics.cacheHitRate;
    
    if (totalCacheRequests > 0) {
      this.metrics.cacheHitRate = 
        (currentHitRate * (totalCacheRequests - 1) + (isHit ? 1 : 0)) / totalCacheRequests;
    }
  }

  private updateErrorMetrics(errorCode: string): void {
    this.metrics.errorBreakdown[errorCode] = (this.metrics.errorBreakdown[errorCode] || 0) + 1;
    this.metrics.lastError = errorCode;
    this.metrics.lastErrorTime = Date.now();
  }

  private updateMetrics(): void {
    this.metrics.uptime = Date.now() - this.startTime;
    this.lastMetricsUpdate = Date.now();
  }

  getMetrics(): RequestMetrics & {
    cacheStats: any;
    queueStats: any;
    circuitBreakerStatus: any;
    websocketStats?: any;
  } {
    const result: any = {
      ...this.metrics,
      cacheStats: this.getCacheStats(),
      queueStats: this.getQueueStats(),
      circuitBreakerStatus: this.getCircuitBreakerStatus()
    };

    if (this.wsManager) {
      result.websocketStats = this.wsManager.getConnectionStats();
    }

    return result;
  }

  resetMetrics(): void {
    this.metrics = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      averageResponseTime: 0,
      totalBytesTransferred: 0,
      cacheHitRate: 0,
      retryRate: 0,
      errorBreakdown: {},
      uptime: 0,
    };
    
    PipelineUtils.info('📊 API 메트릭 리셋됨');
  }

  // =================================================================
  // 🔧 WebSocket 관련 메서드들
  // =================================================================

  connectWebSocket(): Promise<boolean> {
    if (!this.wsManager) {
      this.initializeWebSocket();
    }
    return this.wsManager?.connect() || Promise.resolve(false);
  }

  disconnectWebSocket(): void {
    this.wsManager?.disconnect();
  }

  isWebSocketConnected(): boolean {
    return this.wsManager?.isConnected() || false;
  }

  subscribeToSession(sessionId: string): void {
    this.wsManager?.subscribe(sessionId);
  }

  unsubscribeFromSession(sessionId: string): void {
    this.wsManager?.unsubscribe(sessionId);
  }

  onWebSocketMessage(type: string, handler: Function): void {
    this.wsManager?.onMessage(type, handler);
  }

  onWebSocketEvent(event: string, handler: Function): void {
    this.wsManager?.onEvent(event, handler);
  }

  // =================================================================
  // 🔧 요청 취소 및 중단 (완전한 기능형)
  // =================================================================

  cancelRequest(requestId: string): boolean {
    const abortController = this.abortControllers.get(requestId);
    if (abortController) {
      abortController.abort();
      this.abortControllers.delete(requestId);
      PipelineUtils.info('🚫 요청 취소됨', { requestId });
      return true;
    }
    return false;
  }

  cancelAllRequests(): void {
    const cancelledCount = this.abortControllers.size;
    
    for (const [requestId, controller] of this.abortControllers.entries()) {
      controller.abort();
    }
    
    this.abortControllers.clear();
    this.requestQueue = [];
    
    PipelineUtils.info('🚫 모든 요청 취소됨', { cancelledCount });
  }

  pauseRequestProcessing(): void {
    // 새로운 요청 처리 일시 중지
    this.config.maxConcurrentRequests = 0;
    PipelineUtils.info('⏸️ 요청 처리 일시 중지됨');
  }

  resumeRequestProcessing(): void {
    // 요청 처리 재개
    this.config.maxConcurrentRequests = 3;
    this.processRequestQueue();
    PipelineUtils.info('▶️ 요청 처리 재개됨');
  }

  // =================================================================
  // 🔧 유틸리티 메서드들 (완전한 기능형)
  // =================================================================

  private buildURL(endpoint: string): string {
    const baseURL = this.config.baseURL.replace(/\/$/, '');
    const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
    return `${baseURL}${cleanEndpoint}`;
  }

  private generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  private generateSessionId(): string {
    return `ses_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  private getErrorCode(error: any): string {
    if (error?.code) return error.code;
    if (error?.name === 'AbortError') return 'timeout';
    if (error?.message?.includes('fetch')) return 'network_error';
    if (error?.message?.includes('timeout')) return 'timeout';
    if (error?.message?.includes('JSON')) return 'parse_error';
    return 'unknown_error';
  }

  private getRetryAfter(response: Response): number | undefined {
    const retryAfter = response.headers.get('retry-after');
    if (retryAfter) {
      const seconds = parseInt(retryAfter);
      return isNaN(seconds) ? undefined : seconds * 1000;
    }
    return undefined;
  }

  private createAPIError(
    code: string,
    message: string,
    details?: any,
    retryAfter?: number
  ): APIError {
    return {
      code,
      message,
      details,
      timestamp: new Date().toISOString(),
      request_id: this.generateRequestId(),
      retry_after: retryAfter
    };
  }

  private calculateCacheSize(data: any): number {
    try {
      return JSON.stringify(data).length;
    } catch {
      return 0;
    }
  }

  private async parseErrorResponse(response: Response): Promise<any> {
    try {
      const contentType = response.headers.get('content-type');
      if (contentType?.includes('application/json')) {
        return await response.json();
      } else {
        const text = await response.text();
        return { message: text || response.statusText };
      }
    } catch (parseError) {
      PipelineUtils.warn('⚠️ 에러 응답 파싱 실패', parseError);
      return { message: response.statusText };
    }
  }

  // =================================================================
  // 🔧 설정 관리 (완전한 기능형)
  // =================================================================

  updateConfig(newConfig: Partial<APIClientConfig>): void {
    Object.assign(this.config, newConfig);
    
    if (newConfig.apiKey) {
      this.defaultHeaders['Authorization'] = `Bearer ${newConfig.apiKey}`;
    }
    
    PipelineUtils.info('⚙️ API 클라이언트 설정 업데이트', newConfig);
  }

  getConfig(): APIClientConfig {
    return { ...this.config };
  }

  setAuthToken(token: string): void {
    this.defaultHeaders['Authorization'] = `Bearer ${token}`;
    PipelineUtils.info('🔑 인증 토큰 설정됨');
  }

  removeAuthToken(): void {
    delete this.defaultHeaders['Authorization'];
    PipelineUtils.info('🔑 인증 토큰 제거됨');
  }

  setDefaultHeaders(headers: Record<string, string>): void {
    Object.assign(this.defaultHeaders, headers);
    PipelineUtils.info('📝 기본 헤더 업데이트됨', Object.keys(headers));
  }

  removeDefaultHeader(key: string): void {
    delete this.defaultHeaders[key];
    PipelineUtils.info('🗑️ 기본 헤더 제거됨', { key });
  }

  // =================================================================
  // 🔧 백엔드 호환 메서드들 (완전한 기능형)
  // =================================================================

  async process(data: any, ...kwargs: any[]): Promise<{ success: boolean; [key: string]: any }> {
    const timer = PipelineUtils.createPerformanceTimer('API 통합 처리');
    
    try {
      let result: any;
      
      if (this.isVirtualTryOnRequest(data)) {
        result = await this.processVirtualTryOn(data, ...kwargs);
      } else if (this.isTaskRequest(data)) {
        result = await this.getTaskStatus(data.task_id);
      } else if (this.isAnalysisRequest(data)) {
        result = await this.processAnalysisRequest(data);
      } else {
        result = await this.request('/api/process', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data),
        });
      }
      
      const processingTime = timer.end();
      
      return {
        success: true,
        step_name: 'PipelineAPIClient',
        result,
        processing_time: processingTime / 1000,
        device: PipelineUtils.autoDetectDevice(),
        device_type: PipelineUtils.autoDetectDeviceType(),
        timestamp: new Date().toISOString()
      };
      
    } catch (error) {
      const processingTime = timer.end();
      PipelineUtils.error('❌ API 통합 처리 실패', error);
      
      return {
        success: false,
        step_name: 'PipelineAPIClient',
        error: this.extractErrorMessage(error),
        processing_time: processingTime / 1000,
        device: PipelineUtils.autoDetectDevice(),
        timestamp: new Date().toISOString()
      };
    }
  }

  private isVirtualTryOnRequest(data: any): boolean {
    return data && 
           data.person_image instanceof File && 
           data.clothing_image instanceof File &&
           typeof data.height === 'number' &&
           typeof data.weight === 'number';
  }

  private isTaskRequest(data: any): boolean {
    return data && typeof data.task_id === 'string';
  }

  private isAnalysisRequest(data: any): boolean {
    return data && data.analysis_type && data.image instanceof File;
  }

  private async processAnalysisRequest(data: any): Promise<any> {
    switch (data.analysis_type) {
      case 'body':
        return await this.analyzeBody(data.image);
      case 'clothing':
        return await this.analyzeClothing(data.image);
      case 'pose':
        return await this.analyzePose(data.image);
      case 'background':
        return await this.extractBackground(data.image);
      default:
        throw new Error(`Unsupported analysis type: ${data.analysis_type}`);
    }
  }

  private extractErrorMessage(error: any): string {
    if (error && typeof error === 'object') {
      if ('message' in error) return error.message;
      if ('detail' in error) return error.detail;
      if ('error' in error) return error.error;
    }
    
    return error instanceof Error ? error.message : 'Unknown error';
  }

  // =================================================================
  // 🔧 정보 조회 메서드들 (완전한 기능형)
  // =================================================================

  getClientInfo(): any {
    return {
      step_name: 'PipelineAPIClient',
      device: PipelineUtils.autoDetectDevice(),
      device_type: PipelineUtils.autoDetectDeviceType(),
      baseURL: this.config.baseURL,
      version: '2.0.0',
      
      configuration: {
        enableRetry: this.config.enableRetry,
        maxRetryAttempts: this.config.retryAttempts,
        enableCaching: this.config.enableCaching,
        timeout: this.config.timeout,
        maxConcurrentRequests: this.config.maxConcurrentRequests,
        enableCompression: this.config.enableCompression,
        enableWebSocket: this.config.enableWebSocket,
      },
      
      capabilities: {
        virtual_tryon: true,
        body_analysis: true,
        clothing_analysis: true,
        pose_analysis: true,
        background_extraction: true,
        task_tracking: true,
        brand_integration: true,
        file_upload: true,
        progress_tracking: true,
        caching: this.config.enableCaching,
        retry_logic: this.config.enableRetry,
        circuit_breaker: true,
        request_queuing: true,
        metrics_collection: this.config.enableMetrics,
        websocket_support: this.config.enableWebSocket,
        feedback_system: true,
        user_profiles: true,
      },
      
      runtime_info: {
        active_requests: this.activeRequests.size,
        queue_size: this.requestQueue.length,
        cache_size: this.cache.size,
        circuit_breaker_failures: this.circuitBreakerFailures,
        total_requests: this.metrics.totalRequests,
        success_rate: this.metrics.totalRequests > 0 
          ? this.metrics.successfulRequests / this.metrics.totalRequests 
          : 0,
        uptime: this.metrics.uptime,
        websocket_connected: this.isWebSocketConnected(),
      },
      
      browser_info: {
        user_agent: navigator.userAgent,
        platform: navigator.platform,
        language: navigator.language,
        online: navigator.onLine,
        hardware_concurrency: navigator.hardwareConcurrency,
        device_memory: (navigator as any).deviceMemory,
        connection: (navigator as any).connection?.effectiveType,
      }
    };
  }

  // =================================================================
  // 🔧 디버그 및 개발 지원 메서드들 (완전한 기능형)
  // =================================================================

  enableDebugMode(enable: boolean = true): void {
    this.config.enableDebug = enable;
    PipelineUtils.info(`🐛 디버그 모드 ${enable ? '활성화' : '비활성화'}됨`);
  }

  exportDebugInfo(): string {
    const debugInfo = {
      config: this.getConfig(),
      metrics: this.getMetrics(),
      clientInfo: this.getClientInfo(),
      headers: this.defaultHeaders,
      cacheEntries: Array.from(this.cache.entries()).map(([key, entry]) => ({
        key: key.substring(0, 100) + '...',
        size: entry.size,
        hits: entry.hits,
        age: Date.now() - entry.timestamp,
        etag: entry.etag
      })),
      activeRequests: Array.from(this.activeRequests),
      requestQueue: this.requestQueue.map(req => ({
        id: req.id,
        url: req.url,
        priority: req.priority,
        attempts: req.attempts,
        age: Date.now() - req.timestamp
      })),
      timestamp: new Date().toISOString()
    };
    
    return JSON.stringify(debugInfo, null, 2);
  }

  async testEndpoint(endpoint: string, options: RequestInit = {}): Promise<any> {
    PipelineUtils.info('🧪 엔드포인트 테스트', { endpoint });
    
    try {
      const result = await this.request(endpoint, {
        ...options,
        method: options.method || 'GET'
      }, true);
      
      PipelineUtils.info('✅ 엔드포인트 테스트 성공', { endpoint, result });
      return result;
    } catch (error) {
      PipelineUtils.error('❌ 엔드포인트 테스트 실패', { endpoint, error });
      throw error;
    }
  }

  async benchmarkEndpoint(endpoint: string, iterations: number = 10): Promise<any> {
    const results: number[] = [];
    
    for (let i = 0; i < iterations; i++) {
      const timer = PipelineUtils.createPerformanceTimer(`Benchmark ${i + 1}`);
      try {
        await this.request(endpoint, {}, true);
        results.push(timer.end());
      } catch (error) {
        timer.end();
        PipelineUtils.warn(`❌ 벤치마크 반복 ${i + 1} 실패`, error);
      }
    }

    const avgTime = results.reduce((a, b) => a + b, 0) / results.length;
    const minTime = Math.min(...results);
    const maxTime = Math.max(...results);

    return {
      endpoint,
      iterations: results.length,
      averageTime: avgTime,
      minTime,
      maxTime,
      successRate: results.length / iterations,
      results
    };
  }

  // =================================================================
  // 🔧 정리 및 종료 (완전한 기능형)
  // =================================================================

  async cleanup(): Promise<void> {
    PipelineUtils.info('🧹 PipelineAPIClient: 리소스 정리 중...');
    
    try {
      // WebSocket 정리
      if (this.wsManager) {
        this.wsManager.cleanup();
        this.wsManager = null;
      }
      
      // 진행 중인 모든 요청 취소
      this.cancelAllRequests();
      
      // 업로드 진행률 콜백 정리
      this.uploadProgressCallbacks.clear();
      
      // 캐시 정리
      this.clearCache();
      
      // 메트릭 정리
      this.resetMetrics();
      
      // 서킷 브레이커 리셋
      this.resetCircuitBreaker();
      
      // 타이머들 정리
      // (백그라운드 타이머들은 setInterval로 생성되어 명시적 정리가 어려움)
      
      PipelineUtils.info('✅ PipelineAPIClient 리소스 정리 완료');
    } catch (error) {
      PipelineUtils.warn('⚠️ PipelineAPIClient 리소스 정리 중 오류', error);
    }
  }

  // =================================================================
  // 🔧 고급 기능들 (완전한 기능형)
  // =================================================================

  async bulkOperation(operations: Array<{
    endpoint: string;
    method?: string;
    data?: any;
    priority?: number;
  }>): Promise<any[]> {
    const results = await Promise.allSettled(
      operations.map(async (op, index) => {
        const options: RequestInit = {
          method: op.method || 'GET'
        };
        
        if (op.data) {
          if (op.data instanceof FormData) {
            options.body = op.data;
          } else {
            options.headers = { 'Content-Type': 'application/json' };
            options.body = JSON.stringify(op.data);
          }
        }
        
        return await this.request(op.endpoint, options);
      })
    );

    return results.map((result, index) => ({
      index,
      operation: operations[index],
      success: result.status === 'fulfilled',
      data: result.status === 'fulfilled' ? result.value : null,
      error: result.status === 'rejected' ? result.reason : null
    }));
  }

  async streamResponse(endpoint: string, onChunk: (chunk: any) => void): Promise<void> {
    const response = await fetch(this.buildURL(endpoint), {
      headers: this.defaultHeaders
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    const decoder = new TextDecoder();

    try {
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        
        try {
          const data = JSON.parse(chunk);
          onChunk(data);
        } catch {
          // 부분적인 JSON일 수 있으므로 무시
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  async uploadFileInChunks(
    file: File,
    endpoint: string,
    chunkSize: number = this.config.uploadChunkSize,
    onProgress?: (progress: number) => void
  ): Promise<any> {
    const totalChunks = Math.ceil(file.size / chunkSize);
    const uploadId = this.generateRequestId();
    
    for (let i = 0; i < totalChunks; i++) {
      const start = i * chunkSize;
      const end = Math.min(start + chunkSize, file.size);
      const chunk = file.slice(start, end);
      
      const formData = new FormData();
      formData.append('chunk', chunk);
      formData.append('upload_id', uploadId);
      formData.append('chunk_index', i.toString());
      formData.append('total_chunks', totalChunks.toString());
      
      if (i === 0) {
        formData.append('filename', file.name);
        formData.append('total_size', file.size.toString());
      }
      
      await this.request(`${endpoint}/chunk`, {
        method: 'POST',
        body: formData
      });
      
      const progress = Math.round(((i + 1) / totalChunks) * 100);
      onProgress?.(progress);
    }
    
    // 업로드 완료 알림
    return await this.request(`${endpoint}/complete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ upload_id: uploadId })
    });
  }
}