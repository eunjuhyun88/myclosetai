/**
 * MyCloset AI 파이프라인 API 클라이언트 (완전한 수정 버전)
 * 실제 백엔드 API 구조와 100% 호환되는 프로덕션 수준 HTTP 클라이언트
 * - 완전한 에러 처리 및 재시도 로직
 * - 파일 업로드 및 진행률 추적
 * - 캐싱 및 성능 최적화
 * - 백엔드 8단계 파이프라인 완전 지원
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
} from '../types/pipeline';
import { PipelineUtils } from '../utils/pipelineUtils';

// =================================================================
// 🔧 API 클라이언트 설정 타입들
// =================================================================

export interface APIClientConfig {
  baseURL: string;
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
}

export interface CacheEntry<T = any> {
  data: T;
  timestamp: number;
  expiry: number;
  size: number;
  hits: number;
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
}

// =================================================================
// 🔧 메인 PipelineAPIClient 클래스
// =================================================================

export default class PipelineAPIClient {
  private config: APIClientConfig;
  private defaultHeaders: Record<string, string>;
  private metrics: RequestMetrics;
  private cache: Map<string, CacheEntry> = new Map();
  private requestQueue: QueuedRequest[] = [];
  private activeRequests: Set<string> = new Set();
  private abortControllers: Map<string, AbortController> = new Map();
  
  // 재시도 및 백오프 관리
  private retryDelays: number[] = [1000, 2000, 4000, 8000, 16000];
  private circuitBreakerFailures = 0;
  private circuitBreakerLastFailure = 0;
  private readonly circuitBreakerThreshold = 5;
  private readonly circuitBreakerTimeout = 60000; // 1분
  
  // 업로드 진행률 추적
  private uploadProgressCallbacks: Map<string, (progress: number) => void> = new Map();

  constructor(options: UsePipelineOptions = {}, ...kwargs: any[]) {
    this.config = {
      baseURL: options.baseURL || 'http://localhost:8000',
      apiKey: options.apiKey,
      timeout: options.requestTimeout || 30000,
      retryAttempts: options.maxRetryAttempts || 3,
      retryDelay: options.retryDelay || 1000,
      enableCaching: options.enableCaching ?? true,
      cacheTimeout: options.cacheTimeout || 300000, // 5분
      enableCompression: options.compressionEnabled ?? true,
      enableRetry: options.enableRetry ?? true,
      uploadChunkSize: 1024 * 1024, // 1MB
      maxConcurrentRequests: options.maxConcurrentRequests || 3,
      requestQueueSize: 100,
      enableMetrics: true,
      enableDebug: options.enableDebugMode ?? false,
    };

    this.defaultHeaders = {
      'Accept': 'application/json',
      'User-Agent': `MyClosetAI-Client/1.0.0 (${navigator.userAgent})`,
      'X-Client-Version': '1.0.0',
      'X-Client-Platform': navigator.platform,
      'X-Request-ID': this.generateRequestId(),
    };

    // API Key 설정
    if (this.config.apiKey) {
      this.defaultHeaders['Authorization'] = `Bearer ${this.config.apiKey}`;
    }

    // 압축 지원
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
    };

    // 추가 설정 병합
    this.mergeAdditionalConfig(kwargs);

    PipelineUtils.info('🔧 PipelineAPIClient 초기화', {
      baseURL: this.config.baseURL,
      enableCaching: this.config.enableCaching,
      enableRetry: this.config.enableRetry,
      timeout: this.config.timeout
    });

    // 주기적 캐시 정리
    this.startCacheCleanup();
    
    // 요청 큐 처리 시작
    this.startRequestQueueProcessor();
  }

  // =================================================================
  // 🔧 설정 관리
  // =================================================================

  private mergeAdditionalConfig(kwargs: any[]): void {
    for (const kwarg of kwargs) {
      if (typeof kwarg === 'object' && kwarg !== null) {
        Object.assign(this.config, kwarg);
      }
    }
  }

  updateConfig(newConfig: Partial<APIClientConfig>): void {
    Object.assign(this.config, newConfig);
    
    if (newConfig.apiKey) {
      this.defaultHeaders['Authorization'] = `Bearer ${newConfig.apiKey}`;
    }
    
    PipelineUtils.info('⚙️ API 클라이언트 설정 업데이트', newConfig);
  }

  setAuthToken(token: string): void {
    this.defaultHeaders['Authorization'] = `Bearer ${token}`;
    PipelineUtils.info('🔑 인증 토큰 설정됨');
  }

  removeAuthToken(): void {
    delete this.defaultHeaders['Authorization'];
    PipelineUtils.info('🔑 인증 토큰 제거됨');
  }

  // =================================================================
  // 🔧 핵심 HTTP 요청 메서드들 (수정됨)
  // =================================================================

  private async request<T = any>(
    endpoint: string,
    options: RequestInit = {},
    skipCache: boolean = false
  ): Promise<T> {
    const url = this.buildURL(endpoint);
    const cacheKey = this.generateCacheKey(url, options);
    
    // 서킷 브레이커 체크
    if (this.isCircuitBreakerOpen()) {
      throw this.createAPIError('circuit_breaker_open', 'Circuit breaker is open');
    }

    // 캐시 확인
    if (!skipCache && this.config.enableCaching && options.method !== 'POST') {
      const cached = this.getFromCache<T>(cacheKey);
      if (cached) {
        this.updateCacheMetrics(true);
        return cached;
      }
    }

    this.updateCacheMetrics(false);

    // 동시 요청 수 제한
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

      // AbortController 설정
      const abortController = new AbortController();
      this.abortControllers.set(requestId, abortController);
      
      const timeoutId = setTimeout(() => {
        abortController.abort();
      }, this.config.timeout);

      // 🔧 수정: FormData인 경우 Content-Type 헤더 제거
      const requestOptions: RequestInit = {
        ...options,
        headers: {
          ...this.defaultHeaders,
          ...options.headers,
          'X-Request-ID': requestId,
          'X-Attempt-Number': attemptNum.toString(),
        },
        signal: abortController.signal,
      };

      // FormData인 경우 Content-Type 헤더 제거 (브라우저가 자동 설정)
      if (options.body instanceof FormData) {
        delete (requestOptions.headers as any)['Content-Type'];
        delete (requestOptions.headers as any)['Accept'];
        (requestOptions.headers as any)['Accept'] = '*/*';
      }

      if (this.config.enableDebug) {
        PipelineUtils.debug('🌐 API 요청 시작', {
          url,
          method: requestOptions.method || 'GET',
          requestId,
          attempt: attemptNum,
          isFormData: options.body instanceof FormData
        });
      }

      // 실제 요청 수행
      const response = await fetch(url, requestOptions);
      clearTimeout(timeoutId);

      const duration = timer.end();
      this.updateResponseTimeMetrics(duration);

      // 응답 처리
      const result = await this.processResponse<T>(response, requestId);
      
      // 성공 시 캐시에 저장
      if (this.config.enableCaching && requestOptions.method !== 'POST') {
        this.saveToCache(cacheKey, result, this.calculateCacheSize(result));
      }

      this.metrics.successfulRequests++;
      this.resetCircuitBreaker();

      return result;

    } catch (error) {
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
    // 응답 크기 추적
    const contentLength = response.headers.get('content-length');
    if (contentLength) {
      this.metrics.totalBytesTransferred += parseInt(contentLength);
    }

    // 상태 코드 확인
    if (!response.ok) {
      const errorData = await this.parseErrorResponse(response);
      throw this.createAPIError(
        `http_${response.status}`,
        errorData.message || errorData.detail || response.statusText,
        errorData,
        this.getRetryAfter(response)
      );
    }

    // Content-Type 확인
    const contentType = response.headers.get('content-type');
    if (contentType?.includes('application/json')) {
      return await response.json();
    } else if (contentType?.includes('text/')) {
      return await response.text() as unknown as T;
    } else {
      return await response.blob() as unknown as T;
    }
  }

  private async parseErrorResponse(response: Response): Promise<any> {
    try {
      const contentType = response.headers.get('content-type');
      if (contentType?.includes('application/json')) {
        return await response.json();
      } else {
        return { message: await response.text() };
      }
    } catch {
      return { message: response.statusText };
    }
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

    // 재시도 가능한 오류인지 확인
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
  // 🔧 메인 API 메서드들 (백엔드 완전 호환) - 수정됨
  // =================================================================

  async processVirtualTryOn(
    request: VirtualTryOnRequest,
    onProgress?: (progress: number) => void
  ): Promise<VirtualTryOnResponse> {
    const timer = PipelineUtils.createPerformanceTimer('가상 피팅 API 전체 처리');

    try {
      this.validateVirtualTryOnRequest(request);

      // FormData 구성
      const formData = this.buildVirtualTryOnFormData(request);

      // 업로드 진행률 콜백 등록
      const requestId = this.generateRequestId();
      if (onProgress) {
        this.uploadProgressCallbacks.set(requestId, onProgress);
      }

      // XMLHttpRequest를 사용한 업로드 (진행률 추적을 위해)
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

    } catch (error) {
      timer.end();
      const friendlyError = PipelineUtils.getUserFriendlyError(error);
      PipelineUtils.error('❌ 가상 피팅 API 실패', friendlyError);
      throw error;
    }
  }

  private buildVirtualTryOnFormData(request: VirtualTryOnRequest): FormData {
    const formData = new FormData();
    
    // 필수 파일들
    formData.append('person_image', request.person_image, request.person_image.name);
    formData.append('clothing_image', request.clothing_image, request.clothing_image.name);
    
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
    formData.append('session_id', request.session_id || PipelineUtils.generateSessionId());
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
    formData.append('client_version', '1.0.0');
    formData.append('platform', navigator.platform);
    formData.append('timestamp', new Date().toISOString());
    
    return formData;
  }

  private async uploadWithProgress<T>(
    endpoint: string,
    formData: FormData,
    requestId: string,
    onProgress?: (progress: number) => void
  ): Promise<T> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const url = this.buildURL(endpoint);

      // 업로드 진행률 추적
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          const progress = Math.round((event.loaded / event.total) * 100);
          onProgress?.(progress);
          
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

      // 응답 처리
      xhr.addEventListener('load', () => {
        this.uploadProgressCallbacks.delete(requestId);
        
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const result = JSON.parse(xhr.responseText);
            resolve(result);
          } catch (error) {
            reject(new Error('Invalid JSON response'));
          }
        } else {
          try {
            const errorData = JSON.parse(xhr.responseText);
            reject(this.createAPIError(
              `http_${xhr.status}`,
              errorData.message || errorData.detail || xhr.statusText,
              errorData
            ));
          } catch {
            reject(this.createAPIError(`http_${xhr.status}`, xhr.statusText));
          }
        }
      });

      // 오류 처리
      xhr.addEventListener('error', () => {
        this.uploadProgressCallbacks.delete(requestId);
        reject(new Error('Network error during upload'));
      });

      // 타임아웃 처리
      xhr.addEventListener('timeout', () => {
        this.uploadProgressCallbacks.delete(requestId);
        reject(new Error('Upload timeout'));
      });

      // 취소 처리
      xhr.addEventListener('abort', () => {
        this.uploadProgressCallbacks.delete(requestId);
        reject(new Error('Upload aborted'));
      });

      // 요청 설정
      xhr.timeout = this.config.timeout;
      xhr.open('POST', url);

      // 헤더 설정 (FormData는 Content-Type을 자동 설정하므로 제외)
      for (const [key, value] of Object.entries(this.defaultHeaders)) {
        if (key !== 'Content-Type') {
          xhr.setRequestHeader(key, value);
        }
      }
      xhr.setRequestHeader('X-Request-ID', requestId);

      // 요청 전송
      xhr.send(formData);
    });
  }

  // =================================================================
  // 🔧 개별 분석 API들
  // =================================================================

  async analyzeBody(image: File): Promise<any> {
    this.validateImageFile(image);

    const formData = new FormData();
    formData.append('image', image);

    return await this.request('/api/analyze-body', {
      method: 'POST',
      body: formData,
    });
  }

  async analyzeClothing(image: File): Promise<any> {
    this.validateImageFile(image);

    const formData = new FormData();
    formData.append('image', image);

    return await this.request('/api/analyze-clothing', {
      method: 'POST',
      body: formData,
    });
  }

  // =================================================================
  // 🔧 Task 관리 API들
  // =================================================================

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

  async getTaskHistory(limit: number = 50): Promise<TaskInfo[]> {
    return await this.request(`/api/tasks/history?limit=${limit}`);
  }

  async getProcessingQueue(): Promise<any> {
    return await this.request('/api/tasks/queue');
  }

  // =================================================================
  // 🔧 시스템 상태 API들
  // =================================================================

  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.request('/health', {}, true); // 캐시 무시
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

  // =================================================================
  // 🔧 파이프라인 관리 API들
  // =================================================================

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
      
      // 시스템 파라미터 추가
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

  async getSupportedFeatures(): Promise<string[]> {
    const response = await this.request('/api/features');
    return response.features || [];
  }

  // =================================================================
  // 🔧 브랜드 및 사이즈 API들
  // =================================================================

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

  // =================================================================
  // 🔧 캐싱 시스템
  // =================================================================

  private generateCacheKey(url: string, options: RequestInit): string {
    const method = options.method || 'GET';
    const body = options.body ? JSON.stringify(options.body) : '';
    return `${method}:${url}:${body}`;
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

  private saveToCache<T>(key: string, data: T, size: number): void {
    const expiry = Date.now() + this.config.cacheTimeout;
    
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      expiry,
      size,
      hits: 0
    });

    // 캐시 크기 제한 (최대 100개 항목)
    if (this.cache.size > 100) {
      this.evictOldestCacheEntries();
    }
  }

  private evictOldestCacheEntries(): void {
    const entries = Array.from(this.cache.entries());
    entries.sort((a, b) => a[1].timestamp - b[1].timestamp);
    
    // 가장 오래된 20개 항목 제거
    for (let i = 0; i < Math.min(20, entries.length); i++) {
      this.cache.delete(entries[i][0]);
    }
  }

  private calculateCacheSize(data: any): number {
    try {
      return JSON.stringify(data).length;
    } catch {
      return 0;
    }
  }

  private startCacheCleanup(): void {
    setInterval(() => {
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
    }, 60000); // 1분마다 실행
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
  } {
    let totalSize = 0;
    let oldestTimestamp = Date.now();
    let newestTimestamp = 0;
    let totalHits = 0;
    let totalRequests = 0;

    for (const entry of this.cache.values()) {
      totalSize += entry.size;
      totalHits += entry.hits;
      totalRequests += entry.hits + 1; // +1 for initial store
      oldestTimestamp = Math.min(oldestTimestamp, entry.timestamp);
      newestTimestamp = Math.max(newestTimestamp, entry.timestamp);
    }

    return {
      size: this.cache.size,
      hitRate: totalRequests > 0 ? totalHits / totalRequests : 0,
      totalSize,
      oldestEntry: oldestTimestamp,
      newestEntry: newestTimestamp
    };
  }

  // =================================================================
  // 🔧 요청 큐잉 시스템
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
        timestamp: Date.now()
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
    // 요청 타입에 따른 우선순위 설정
    if (url.includes('/health')) return 10;
    if (url.includes('/virtual-tryon')) return 8;
    if (url.includes('/pipeline/status')) return 7;
    if (url.includes('/tasks/')) return 6;
    if (url.includes('/stats')) return 3;
    return 5; // 기본 우선순위
  }

  private startRequestQueueProcessor(): void {
    setInterval(() => {
      this.processRequestQueue();
    }, 100); // 100ms마다 큐 처리
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
      priorityDistribution
    };
  }

  // =================================================================
  // 🔧 서킷 브레이커 패턴
  // =================================================================

  private isCircuitBreakerOpen(): boolean {
    const now = Date.now();
    
    if (this.circuitBreakerFailures >= this.circuitBreakerThreshold) {
      if (now - this.circuitBreakerLastFailure < this.circuitBreakerTimeout) {
        return true;
      } else {
        // 타임아웃 후 서킷 브레이커 리셋
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
  } {
    const now = Date.now();
    const timeUntilReset = this.isCircuitBreakerOpen() 
      ? this.circuitBreakerTimeout - (now - this.circuitBreakerLastFailure)
      : 0;

    return {
      isOpen: this.isCircuitBreakerOpen(),
      failures: this.circuitBreakerFailures,
      threshold: this.circuitBreakerThreshold,
      timeUntilReset: Math.max(0, timeUntilReset)
    };
  }

  // =================================================================
  // 🔧 재시도 로직
  // =================================================================

  private shouldRetry(error: any, attemptNum: number): boolean {
    if (!this.config.enableRetry || attemptNum >= this.config.retryAttempts) {
      return false;
    }

    const errorCode = this.getErrorCode(error);
    
    // 재시도 불가능한 오류들
    const nonRetryableErrors = [
      'http_400', 'http_401', 'http_403', 'http_404', 
      'http_422', 'validation_error', 'invalid_file'
    ];
    
    if (nonRetryableErrors.includes(errorCode)) {
      return false;
    }

    // 재시도 가능한 오류들
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
    
    // 지터 추가 (±25%)
    const jitter = exponentialDelay * 0.25 * (Math.random() - 0.5);
    
    return Math.max(1000, exponentialDelay + jitter);
  }

  // =================================================================
  // 🔧 메트릭 및 성능 추적
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
  }

  getMetrics(): RequestMetrics & {
    cacheStats: any;
    queueStats: any;
    circuitBreakerStatus: any;
  } {
    return {
      ...this.metrics,
      cacheStats: this.getCacheStats(),
      queueStats: this.getQueueStats(),
      circuitBreakerStatus: this.getCircuitBreakerStatus()
    };
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
    };
    
    PipelineUtils.info('📊 API 메트릭 리셋됨');
  }

  // =================================================================
  // 🔧 검증 및 유틸리티 메서드들
  // =================================================================

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

    if (!PipelineUtils.validateFileSize(file, 10)) {
      throw this.createAPIError('file_too_large', `${fieldName}: 파일 크기가 너무 큽니다. 10MB 이하의 파일을 사용해주세요.`);
    }
  }

  private buildURL(endpoint: string): string {
    const baseURL = this.config.baseURL.replace(/\/$/, '');
    const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
    return `${baseURL}${cleanEndpoint}`;
  }

  private generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  private getErrorCode(error: any): string {
    if (error?.code) return error.code;
    if (error?.name === 'AbortError') return 'timeout';
    if (error?.message?.includes('fetch')) return 'network_error';
    if (error?.message?.includes('timeout')) return 'timeout';
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

  // =================================================================
  // 🔧 요청 취소 및 중단
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

  // =================================================================
  // 🔧 백엔드 패턴 호환 메서드들
  // =================================================================

  async initialize(): Promise<boolean> {
    PipelineUtils.info('🔄 PipelineAPIClient 초기화 중...');
    
    try {
      const isHealthy = await this.healthCheck();
      
      if (isHealthy) {
        // 시스템 정보 로드
        try {
          const [serverInfo, features] = await Promise.all([
            this.getServerInfo(),
            this.getSupportedFeatures()
          ]);
          
          PipelineUtils.info('✅ PipelineAPIClient 초기화 완료', {
            serverVersion: serverInfo.version,
            supportedFeatures: features.length
          });
        } catch (error) {
          PipelineUtils.warn('⚠️ 서버 정보 로드 실패', error);
        }
        
        return true;
      } else {
        PipelineUtils.error('❌ PipelineAPIClient 초기화 실패 - 서버 비정상');
        return false;
      }
    } catch (error) {
      PipelineUtils.error('❌ PipelineAPIClient 초기화 중 오류', error);
      return false;
    }
  }

  async process(data: any, ...kwargs: any[]): Promise<{ success: boolean; [key: string]: any }> {
    const timer = PipelineUtils.createPerformanceTimer('API 통합 처리');
    
    try {
      let result: any;
      
      // 데이터 타입에 따른 적절한 처리 메서드 선택
      if (this.isVirtualTryOnRequest(data)) {
        result = await this.processVirtualTryOn(data, ...kwargs);
      } else if (this.isTaskRequest(data)) {
        result = await this.getTaskStatus(data.task_id);
      } else {
        // 일반 API 요청
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

  private extractErrorMessage(error: any): string {
    if (error && typeof error === 'object') {
      if ('message' in error) return error.message;
      if ('detail' in error) return error.detail;
      if ('error' in error) return error.error;
    }
    
    return error instanceof Error ? error.message : 'Unknown error';
  }

  // =================================================================
  // 🔧 정리 및 종료
  // =================================================================

  async cleanup(): Promise<void> {
    PipelineUtils.info('🧹 PipelineAPIClient: 리소스 정리 중...');
    
    try {
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
      
      PipelineUtils.info('✅ PipelineAPIClient 리소스 정리 완료');
    } catch (error) {
      PipelineUtils.warn('⚠️ PipelineAPIClient 리소스 정리 중 오류', error);
    }
  }

  // =================================================================
  // 🔧 정보 조회 메서드들
  // =================================================================

  getClientInfo(): any {
    return {
      step_name: 'PipelineAPIClient',
      device: PipelineUtils.autoDetectDevice(),
      device_type: PipelineUtils.autoDetectDeviceType(),
      baseURL: this.config.baseURL,
      version: '1.0.0',
      
      configuration: {
        enableRetry: this.config.enableRetry,
        maxRetryAttempts: this.config.retryAttempts,
        enableCaching: this.config.enableCaching,
        timeout: this.config.timeout,
        maxConcurrentRequests: this.config.maxConcurrentRequests,
        enableCompression: this.config.enableCompression,
      },
      
      capabilities: {
        virtual_tryon: true,
        body_analysis: true,
        clothing_analysis: true,
        task_tracking: true,
        brand_integration: true,
        file_upload: true,
        progress_tracking: true,
        caching: this.config.enableCaching,
        retry_logic: this.config.enableRetry,
        circuit_breaker: true,
        request_queuing: true,
        metrics_collection: this.config.enableMetrics,
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
      },
      
      browser_info: {
        user_agent: navigator.userAgent,
        platform: navigator.platform,
        language: navigator.language,
        online: navigator.onLine,
        hardware_concurrency: navigator.hardwareConcurrency,
        device_memory: (navigator as any).deviceMemory,
      }
    };
  }

  getConfig(): APIClientConfig {
    return { ...this.config };
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
  // 🔧 디버그 및 개발 지원 메서드들
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
        key,
        size: entry.size,
        hits: entry.hits,
        age: Date.now() - entry.timestamp
      })),
      timestamp: new Date().toISOString()
    };
    
    return JSON.stringify(debugInfo, null, 2);
  }

  // 개발 환경에서 API 엔드포인트 테스트
  async testEndpoint(endpoint: string, options: RequestInit = {}): Promise<any> {
    PipelineUtils.info('🧪 엔드포인트 테스트', { endpoint });
    
    try {
      const result = await this.request(endpoint, {
        ...options,
        method: options.method || 'GET'
      }, true); // 캐시 무시
      
      PipelineUtils.info('✅ 엔드포인트 테스트 성공', { endpoint, result });
      return result;
    } catch (error) {
      PipelineUtils.error('❌ 엔드포인트 테스트 실패', { endpoint, error });
      throw error;
    }
  }
}