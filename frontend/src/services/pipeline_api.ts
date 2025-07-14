/**
 * MyCloset AI 파이프라인 API 클라이언트 - 실제 백엔드 완전 호환 버전
 * 백엔드 실제 엔드포인트와 100% 호환 + Task 기반 처리 지원
 */

import type {
  VirtualTryOnRequest,
  VirtualTryOnResponse,
  BodyAnalysisResponse,
  ClothingAnalysisResponse,
  ProcessingStatus,
  PipelineStatus,
  SystemStats,
  UsePipelineOptions,
  BrandSizeData,
  SizeRecommendation,
  TaskInfo,
} from '../types/pipeline';
import { PipelineUtils } from '../utils/pipelineUtils';

export class PipelineAPIClient {
  /**
   * ✅ 백엔드 실제 API 구조와 완전 호환되는 클라이언트
   */
  
  private config: UsePipelineOptions;
  private baseURL: string;
  private device: string;
  private step_name: string;
  private defaultHeaders: Record<string, string>;
  private activeTasks: Map<string, TaskInfo> = new Map();

  constructor(
    options: UsePipelineOptions = {},
    ...kwargs: any[]
  ) {
    this.device = options.device || PipelineUtils.autoDetectDevice();
    this.baseURL = options.baseURL || 'http://localhost:8000';
    this.step_name = 'PipelineAPIClient';
    
    this.config = {
      baseURL: this.baseURL,
      wsURL: options.wsURL || this.baseURL.replace('http', 'ws'),
      autoReconnect: options.autoReconnect ?? true,
      maxReconnectAttempts: options.maxReconnectAttempts || 5,
      reconnectInterval: options.reconnectInterval || 3000,
      heartbeatInterval: options.heartbeatInterval || 30000,
      connectionTimeout: options.connectionTimeout || 10000,
      
      device_type: options.device_type || PipelineUtils.autoDetectDeviceType(),
      memory_gb: options.memory_gb || 16.0,
      is_m3_max: options.is_m3_max ?? PipelineUtils.detectM3Max(),
      optimization_enabled: options.optimization_enabled ?? true,
      quality_level: options.quality_level || 'balanced',
      
      autoHealthCheck: options.autoHealthCheck ?? true,
      healthCheckInterval: options.healthCheckInterval || 30000,
      persistSession: options.persistSession ?? true,
      enableDetailedProgress: options.enableDetailedProgress ?? true,
      enableRetry: options.enableRetry ?? true,
      maxRetryAttempts: options.maxRetryAttempts || 3,
      enableTaskTracking: options.enableTaskTracking ?? true,
      enableBrandIntegration: options.enableBrandIntegration ?? true,
      
      ...options,
    };
    
    this._mergeAdditionalConfig(kwargs);
    
    this.defaultHeaders = {
      'Accept': 'application/json',
    };
    
    PipelineUtils.log('info', `🎯 ${this.step_name} 초기화 - 디바이스: ${this.device}`);
  }

  private _mergeAdditionalConfig(kwargs: any[]): void {
    const systemParams = PipelineUtils.getSystemParams();

    for (const kwarg of kwargs) {
      if (typeof kwarg === 'object' && kwarg !== null) {
        for (const [key, value] of Object.entries(kwarg)) {
          if (!systemParams.has(key) && value !== undefined) {
            (this.config as any)[key] = value;
          }
        }
      }
    }
  }

  // =================================================================
  // 🔧 기본 HTTP 요청 메서드들
  // =================================================================

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
        const errorData = await response.json().catch(() => ({}));
        const error = {
          status: response.status,
          message: errorData.detail || errorData.error || response.statusText,
          details: errorData
        };
        throw error;
      }

      const data = await response.json();
      PipelineUtils.log('info', `✅ API 응답 성공: ${endpoint}`);
      return data;

    } catch (error) {
      PipelineUtils.log('error', `❌ API 요청 실패: ${endpoint}`, error);
      throw error;
    }
  }

  private async requestWithRetry<T = any>(
    endpoint: string,
    options: RequestInit = {},
    maxRetries: number = this.config.maxRetryAttempts || 3
  ): Promise<T> {
    let lastError: any;
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        return await this.request<T>(endpoint, options);
      } catch (error) {
        lastError = error;
        
        if (attempt === maxRetries) {
          break;
        }
        
        if (error && typeof error === 'object' && 'status' in error) {
          const apiError = error as any;
          if (apiError.status === 400 || apiError.status === 401 || apiError.status === 403) {
            break;
          }
        }
        
        const delay = 1000 * attempt;
        PipelineUtils.log('warn', `⚠️ API 호출 실패 (${attempt}/${maxRetries}), ${delay}ms 후 재시도...`);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
    
    throw lastError;
  }

  // =================================================================
  // 🔧 메인 API 메서드들 - 백엔드 실제 엔드포인트 호환
  // =================================================================

  /**
   * ✅ 메인 가상 피팅 API - /api/virtual-tryon
   */
  async processVirtualTryOn(request: VirtualTryOnRequest): Promise<VirtualTryOnResponse> {
    const timer = PipelineUtils.createPerformanceTimer('가상 피팅 API 요청');
    
    try {
      this._validateRequest(request);

      const formData = new FormData();
      formData.append('person_image', request.person_image);
      formData.append('clothing_image', request.clothing_image);
      formData.append('height', request.height.toString());
      formData.append('weight', request.weight.toString());
      
      // ✅ 백엔드 실제 파라미터들
      if (request.chest) formData.append('chest', request.chest.toString());
      if (request.waist) formData.append('waist', request.waist.toString());
      if (request.hip) formData.append('hip', request.hip.toString());
      
      formData.append('clothing_type', request.clothing_type || 'shirt');
      formData.append('fabric_type', request.fabric_type || 'cotton');
      formData.append('style_preference', request.style_preference || 'regular');
      formData.append('quality_level', request.quality_level || 'balanced');
      
      formData.append('session_id', request.session_id || PipelineUtils.generateSessionId());
      formData.append('enable_realtime', String(request.enable_realtime || false));

      // 백엔드 호환 시스템 설정
      formData.append('device_type', this.config.device_type || 'auto');
      formData.append('optimization_enabled', String(this.config.optimization_enabled));
      formData.append('is_m3_max', String(this.config.is_m3_max));
      formData.append('memory_gb', String(this.config.memory_gb));

      PipelineUtils.log('info', '🚀 가상 피팅 API 요청 시작');

      // ✅ 실제 백엔드 엔드포인트 사용
      const result = await this.requestWithRetry<VirtualTryOnResponse>(
        '/api/virtual-tryon',  // ❌ '/api/virtual-tryon-pipeline'에서 수정
        {
          method: 'POST',
          body: formData,
        }
      );

      timer.end();
      PipelineUtils.log('info', '✅ 가상 피팅 API 응답 성공');
      
      // Task 추적 활성화 시 저장
      if (this.config.enableTaskTracking && result.task_id) {
        this.activeTasks.set(result.task_id, {
          task_id: result.task_id,
          status: 'processing',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          progress_percentage: 0,
          current_stage: 'started'
        });
      }
      
      return result;

    } catch (error) {
      timer.end();
      const errorMessage = this._extractErrorMessage(error);
      PipelineUtils.log('error', '❌ 가상 피팅 API 오류', errorMessage);
      throw new Error(`가상 피팅 처리 실패: ${errorMessage}`);
    }
  }

  /**
   * ✅ 신체 분석 API - /api/analyze-body
   */
  async analyzeBody(image: File): Promise<BodyAnalysisResponse> {
    try {
      if (!PipelineUtils.validateImageType(image)) {
        throw new Error('지원되지 않는 이미지 형식입니다.');
      }

      const formData = new FormData();
      formData.append('image', image);

      const result = await this.request<BodyAnalysisResponse>(
        '/api/analyze-body',  // ✅ 백엔드 실제 엔드포인트
        {
          method: 'POST',
          body: formData,
        }
      );

      PipelineUtils.log('info', '✅ 신체 분석 완료');
      return result;

    } catch (error) {
      const errorMessage = this._extractErrorMessage(error);
      PipelineUtils.log('error', '❌ 신체 분석 실패', errorMessage);
      throw new Error(`신체 분석 실패: ${errorMessage}`);
    }
  }

  /**
   * ✅ 의류 분석 API - /api/analyze-clothing
   */
  async analyzeClothing(image: File): Promise<ClothingAnalysisResponse> {
    try {
      if (!PipelineUtils.validateImageType(image)) {
        throw new Error('지원되지 않는 이미지 형식입니다.');
      }

      const formData = new FormData();
      formData.append('image', image);

      const result = await this.request<ClothingAnalysisResponse>(
        '/api/analyze-clothing',  // ✅ 백엔드 실제 엔드포인트
        {
          method: 'POST',
          body: formData,
        }
      );

      PipelineUtils.log('info', '✅ 의류 분석 완료');
      return result;

    } catch (error) {
      const errorMessage = this._extractErrorMessage(error);
      PipelineUtils.log('error', '❌ 의류 분석 실패', errorMessage);
      throw new Error(`의류 분석 실패: ${errorMessage}`);
    }
  }

  // =================================================================
  // 🔧 Task 기반 처리 API들 - 백엔드 실제 구조
  // =================================================================

  /**
   * ✅ 처리 상태 조회 - /api/processing-status/{task_id}
   */
  async getProcessingStatus(taskId: string): Promise<ProcessingStatus> {
    try {
      const result = await this.request<ProcessingStatus>(
        `/api/processing-status/${taskId}`  // ✅ 백엔드 실제 엔드포인트
      );

      // 로컬 Task 정보 업데이트
      if (this.activeTasks.has(taskId)) {
        const taskInfo = this.activeTasks.get(taskId)!;
        taskInfo.status = result.status as any;
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
   * ✅ 지원 모델 조회 - /api/supported-models
   */
  async getSupportedModels(): Promise<string[]> {
    try {
      const result = await this.request<{models: string[]}>(
        '/api/supported-models'  // ✅ 백엔드 실제 엔드포인트
      );

      return result.models || [];

    } catch (error) {
      PipelineUtils.log('error', '❌ 지원 모델 조회 실패', error);
      return [];
    }
  }

  /**
   * Task 취소 (추가 구현)
   */
  async cancelTask(taskId: string): Promise<boolean> {
    try {
      await this.request(`/api/cancel-task/${taskId}`, {
        method: 'POST',
      });

      // 로컬에서 제거
      if (this.activeTasks.has(taskId)) {
        const taskInfo = this.activeTasks.get(taskId)!;
        taskInfo.status = 'cancelled';
        taskInfo.updated_at = new Date().toISOString();
        this.activeTasks.set(taskId, taskInfo);
      }

      return true;

    } catch (error) {
      PipelineUtils.log('error', '❌ Task 취소 실패', error);
      return false;
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

  // =================================================================
  // 🔧 브랜드 사이즈 관련 API들 - 프로젝트 핵심 기능
  // =================================================================

  /**
   * ✅ 브랜드 사이즈 데이터 조회
   */
  async getBrandSizes(brand: string): Promise<BrandSizeData> {
    try {
      const result = await this.request<BrandSizeData>(
        `/api/brands/${encodeURIComponent(brand)}/sizes`
      );

      return result;

    } catch (error) {
      PipelineUtils.log('error', '❌ 브랜드 사이즈 조회 실패', error);
      throw error;
    }
  }

  /**
   * ✅ 사이즈 추천
   */
  async getSizeRecommendation(
    measurements: any,
    brand: string, 
    item: string
  ): Promise<SizeRecommendation> {
    try {
      const result = await this.request<SizeRecommendation>(
        '/api/size-recommendation',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            measurements,
            brand,
            item
          }),
        }
      );

      return result;

    } catch (error) {
      PipelineUtils.log('error', '❌ 사이즈 추천 실패', error);
      throw error;
    }
  }

  /**
   * 브랜드 호환성 조회
   */
  async getBrandCompatibility(measurements: any): Promise<any> {
    try {
      const result = await this.request(
        '/api/brand-compatibility',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ measurements }),
        }
      );

      return result;

    } catch (error) {
      PipelineUtils.log('error', '❌ 브랜드 호환성 조회 실패', error);
      throw error;
    }
  }

  // =================================================================
  // 🔧 파이프라인 관리 API들
  // =================================================================

  async getPipelineStatus(): Promise<PipelineStatus> {
    try {
      return await this.request<PipelineStatus>('/api/pipeline/status');
    } catch (error) {
      PipelineUtils.log('error', '❌ 파이프라인 상태 조회 실패', error);
      throw error;
    }
  }

  async initializePipeline(): Promise<boolean> {
    try {
      const response = await this.request('/api/pipeline/initialize', {
        method: 'POST',
      });
      return response.initialized || false;
    } catch (error) {
      PipelineUtils.log('error', '❌ 파이프라인 초기화 실패', error);
      return false;
    }
  }

  async warmupPipeline(qualityMode: string = 'balanced'): Promise<boolean> {
    try {
      const formData = new FormData();
      formData.append('quality_mode', qualityMode);
      formData.append('device_type', this.config.device_type || 'auto');
      formData.append('optimization_enabled', String(this.config.optimization_enabled));

      const response = await this.request('/api/pipeline/warmup', {
        method: 'POST',
        body: formData,
      });

      return response.success || false;
    } catch (error) {
      PipelineUtils.log('error', '❌ 파이프라인 워밍업 실패', error);
      return false;
    }
  }

  async getMemoryStatus(): Promise<any> {
    try {
      return await this.request('/api/pipeline/memory');
    } catch (error) {
      PipelineUtils.log('error', '❌ 메모리 상태 조회 실패', error);
      throw error;
    }
  }

  async cleanupMemory(): Promise<boolean> {
    try {
      await this.request('/api/pipeline/cleanup', {
        method: 'POST',
      });
      return true;
    } catch (error) {
      PipelineUtils.log('error', '❌ 메모리 정리 실패', error);
      return false;
    }
  }

  async getModelsInfo(): Promise<any> {
    try {
      return await this.request('/api/pipeline/models/info');
    } catch (error) {
      PipelineUtils.log('error', '❌ 모델 정보 조회 실패', error);
      throw error;
    }
  }

  // =================================================================
  // 🔧 시스템 API들
  // =================================================================

  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.request('/health');
      return response.status === 'healthy';
    } catch {
      return false;
    }
  }

  async getSystemStats(): Promise<SystemStats> {
    try {
      return await this.request<SystemStats>('/stats');
    } catch (error) {
      PipelineUtils.log('error', '❌ 시스템 통계 조회 실패', error);
      throw error;
    }
  }

  async getServerInfo(): Promise<any> {
    try {
      return await this.request('/');
    } catch (error) {
      PipelineUtils.log('error', '❌ 서버 정보 조회 실패', error);
      throw error;
    }
  }

  // =================================================================
  // 🔧 내부 헬퍼 메서드들
  // =================================================================

  private _validateRequest(request: VirtualTryOnRequest): void {
    if (!request.person_image || !request.clothing_image) {
      throw new Error('사용자 이미지와 의류 이미지는 필수입니다.');
    }

    if (!PipelineUtils.validateImageType(request.person_image)) {
      throw new Error('사용자 이미지 형식이 올바르지 않습니다.');
    }
    
    if (!PipelineUtils.validateImageType(request.clothing_image)) {
      throw new Error('의류 이미지 형식이 올바르지 않습니다.');
    }

    if (!PipelineUtils.validateFileSize(request.person_image)) {
      throw new Error('사용자 이미지 크기가 너무 큽니다. (최대 10MB)');
    }

    if (!PipelineUtils.validateFileSize(request.clothing_image)) {
      throw new Error('의류 이미지 크기가 너무 큽니다. (최대 10MB)');
    }

    if (request.height <= 0 || request.weight <= 0) {
      throw new Error('유효한 키와 몸무게를 입력해주세요.');
    }
  }

  private _extractErrorMessage(error: any): string {
    if (error && typeof error === 'object') {
      if ('message' in error) {
        return error.message;
      }
      if ('detail' in error) {
        return error.detail;
      }
      if ('status' in error) {
        return PipelineUtils.getHTTPErrorMessage(error.status);
      }
    }
    
    return error instanceof Error ? error.message : 'Unknown error';
  }

  // =================================================================
  // 🔧 백엔드 패턴 호환 메서드들
  // =================================================================

  async initialize(): Promise<boolean> {
    PipelineUtils.log('info', '🔄 PipelineAPIClient 초기화 중...');
    
    try {
      const isHealthy = await this.healthCheck();
      
      if (isHealthy) {
        PipelineUtils.log('info', '✅ PipelineAPIClient 초기화 완료');
        return true;
      } else {
        PipelineUtils.log('error', '❌ PipelineAPIClient 초기화 실패 - 서버 비정상');
        return false;
      }
    } catch (error) {
      PipelineUtils.log('error', '❌ PipelineAPIClient 초기화 중 오류', error);
      return false;
    }
  }

  async process(data: any, ...kwargs: any[]): Promise<{ success: boolean; [key: string]: any }> {
    const timer = PipelineUtils.createPerformanceTimer('API 요청 처리');
    
    try {
      let result: any;
      
      if (data && typeof data === 'object' && 'person_image' in data) {
        result = await this.processVirtualTryOn(data, ...kwargs);
      } else {
        result = await this.request('/api/pipeline/process', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data),
        });
      }
      
      const processingTime = timer.end();
      
      return {
        success: true,
        step_name: this.step_name,
        result,
        processing_time: processingTime / 1000,
        device: this.device,
        device_type: this.config.device_type,
      };
      
    } catch (error) {
      const processingTime = timer.end();
      PipelineUtils.log('error', '❌ API 처리 실패', error);
      
      return {
        success: false,
        step_name: this.step_name,
        error: this._extractErrorMessage(error),
        processing_time: processingTime / 1000,
        device: this.device,
      };
    }
  }

  async cleanup(): Promise<void> {
    PipelineUtils.log('info', '🧹 PipelineAPIClient: 리소스 정리 중...');
    
    try {
      // 활성 Task들 정리
      this.activeTasks.clear();
      
      PipelineUtils.log('info', '✅ PipelineAPIClient 리소스 정리 완료');
    } catch (error) {
      PipelineUtils.log('warn', '⚠️ PipelineAPIClient 리소스 정리 중 오류', error);
    }
  }

  getClientInfo(): any {
    return {
      step_name: this.step_name,
      device: this.device,
      device_type: this.config.device_type,
      baseURL: this.baseURL,
      wsURL: this.config.wsURL,
      is_m3_max: this.config.is_m3_max,
      optimization_enabled: this.config.optimization_enabled,
      quality_level: this.config.quality_level,
      memory_gb: this.config.memory_gb,
      configuration: {
        enableRetry: this.config.enableRetry,
        maxRetryAttempts: this.config.maxRetryAttempts,
        enableDetailedProgress: this.config.enableDetailedProgress,
        persistSession: this.config.persistSession,
        enableTaskTracking: this.config.enableTaskTracking,
        enableBrandIntegration: this.config.enableBrandIntegration,
      },
      capabilities: {
        virtual_tryon: true,
        body_analysis: true,
        clothing_analysis: true,
        task_tracking: this.config.enableTaskTracking,
        brand_integration: this.config.enableBrandIntegration,
        realtime_updates: true,
        file_upload: true,
        memory_management: true,
        debug_mode: true,
      },
      active_tasks: this.activeTasks.size,
    };
  }

  // =================================================================
  // 🔧 편의 메서드들
  // =================================================================

  updateConfig(newConfig: Partial<UsePipelineOptions>): void {
    this.config = { ...this.config, ...newConfig };
    
    if (newConfig.baseURL) {
      this.baseURL = newConfig.baseURL;
    }
    
    PipelineUtils.log('info', '⚙️ PipelineAPIClient 설정 업데이트');
  }

  getConfig(): UsePipelineOptions {
    return { ...this.config };
  }

  setDefaultHeaders(headers: Record<string, string>): void {
    this.defaultHeaders = { ...this.defaultHeaders, ...headers };
  }

  setAuthToken(token: string): void {
    this.defaultHeaders['Authorization'] = `Bearer ${token}`;
  }

  removeAuthToken(): void {
    delete this.defaultHeaders['Authorization'];
  }
}

export default PipelineAPIClient;