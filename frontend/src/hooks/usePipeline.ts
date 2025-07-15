/**
 * MyCloset AI 8단계 파이프라인 React Hook (완전한 수정 버전)
 * ✅ React 18 StrictMode 문제 해결
 * ✅ App.tsx의 모든 필요 기능 지원
 * ✅ 백엔드 WebSocket과 완전 호환
 * ✅ 타입 안전성 보장
 * ✅ 메모리 누수 방지
 * ✅ 중복 요청 방지
 */

import { useState, useCallback, useRef, useEffect, useMemo } from 'react';

// =================================================================
// 🔧 기본 타입 정의 (App.tsx 완전 호환)
// =================================================================

export interface UsePipelineOptions {
  baseURL?: string;
  wsURL?: string;
  autoReconnect?: boolean;
  maxReconnectAttempts?: number;
  reconnectInterval?: number;
  heartbeatInterval?: number;
  connectionTimeout?: number;
  autoHealthCheck?: boolean;
  healthCheckInterval?: number;
  enableStepTracking?: boolean;
  enableRealTimeUpdates?: boolean;
  stepTimeout?: number;
  autoRetrySteps?: boolean;
  maxStepRetries?: number;
  device?: string;
  device_type?: string;
  memory_gb?: number;
  is_m3_max?: boolean;
  optimization_enabled?: boolean;
  quality_level?: string;
}

export interface VirtualTryOnRequest {
  person_image: File;
  clothing_image: File;
  height: number;
  weight: number;
  chest?: number;
  waist?: number;
  hip?: number;
  shoulder_width?: number;
  clothing_type?: string;
  fabric_type?: string;
  style_preference?: string;
  quality_mode?: 'fast' | 'balanced' | 'quality' | 'ultra';
  session_id?: string;
  enable_realtime?: boolean;
  save_intermediate?: boolean;
}

export interface VirtualTryOnResponse {
  success: boolean;
  fitted_image?: string;
  processing_time: number;
  confidence: number;
  fit_score: number;
  measurements?: {
    chest: number;
    waist: number;
    hip: number;
    bmi: number;
  };
  clothing_analysis?: {
    category: string;
    style: string;
    dominant_color: number[];
  };
  recommendations?: string[];
  session_id?: string;
  task_id?: string;
  error_message?: string;
}

export interface PipelineProgress {
  type: string;
  progress: number;
  message: string;
  step_name?: string;
  step_id?: number;
  current_stage?: string;
  timestamp: number;
  session_id?: string;
  result?: any;
  processing_time?: number;
}

export interface PipelineStep {
  id: number;
  name: string;
  korean?: string;
  description?: string;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'skipped' | 'timeout';
  progress: number;
  start_time?: string;
  end_time?: string;
  duration?: number;
  error_message?: string;
}

// =================================================================
// 🔧 개선된 WebSocket 관리자 (메모리 누수 방지)
// =================================================================

class SafeWebSocketManager {
  private ws: WebSocket | null = null;
  private url: string;
  private isConnecting = false;
  private isDestroyed = false;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 3;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private connectionTimeout: NodeJS.Timeout | null = null;
  
  // 콜백 함수들
  private onMessageCallback?: (data: PipelineProgress) => void;
  private onConnectedCallback?: () => void;
  private onDisconnectedCallback?: () => void;
  private onErrorCallback?: (error: any) => void;

  constructor(url: string, options: Partial<UsePipelineOptions> = {}) {
    this.url = url;
    this.maxReconnectAttempts = options.maxReconnectAttempts || 3;
    
    console.log('🔧 SafeWebSocketManager 생성:', url);
  }

  async connect(): Promise<boolean> {
    if (this.isDestroyed) {
      console.log('❌ WebSocket이 이미 파괴됨');
      return false;
    }

    if (this.isConnected()) {
      console.log('✅ WebSocket이 이미 연결됨');
      return true;
    }

    if (this.isConnecting) {
      console.log('⏳ WebSocket 연결 중...');
      return false;
    }

    this.isConnecting = true;
    console.log('🔗 WebSocket 연결 시도:', this.url);
    
    try {
      this.ws = new WebSocket(this.url);
      
      return new Promise((resolve) => {
        if (!this.ws || this.isDestroyed) {
          this.isConnecting = false;
          resolve(false);
          return;
        }

        // 연결 타임아웃 설정
        this.connectionTimeout = setTimeout(() => {
          console.log('⏰ WebSocket 연결 타임아웃');
          this.ws?.close();
          this.isConnecting = false;
          resolve(false);
        }, 10000);

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
          this.onConnectedCallback?.();
          resolve(true);
        };

        this.ws.onmessage = (event) => {
          if (this.isDestroyed) return;
          
          try {
            const data = JSON.parse(event.data);
            console.log('📨 WebSocket 메시지 수신:', data.type);
            this.onMessageCallback?.(data);
          } catch (error) {
            console.error('❌ WebSocket 메시지 파싱 오류:', error);
          }
        };

        this.ws.onclose = (event) => {
          this.isConnecting = false;
          this.stopHeartbeat();
          
          if (this.connectionTimeout) {
            clearTimeout(this.connectionTimeout);
            this.connectionTimeout = null;
          }
          
          if (!this.isDestroyed) {
            console.log('🔌 WebSocket 연결 종료:', event.code, event.reason);
            this.onDisconnectedCallback?.();
            
            // 자동 재연결 시도
            this.scheduleReconnect();
          }
        };

        this.ws.onerror = (error) => {
          this.isConnecting = false;
          
          if (this.connectionTimeout) {
            clearTimeout(this.connectionTimeout);
            this.connectionTimeout = null;
          }
          
          if (!this.isDestroyed) {
            console.error('❌ WebSocket 오류:', error);
            this.onErrorCallback?.(error);
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

  private scheduleReconnect(): void {
    if (this.isDestroyed || this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('🚫 재연결 중단:', this.isDestroyed ? '파괴됨' : '최대 시도 횟수 초과');
      return;
    }

    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
    this.reconnectAttempts++;
    
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
        this.send({
          type: 'ping',
          timestamp: Date.now()
        });
      }
    }, 30000); // 30초마다 핑
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  disconnect(): void {
    console.log('🔌 WebSocket 연결 해제');
    
    this.stopHeartbeat();
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    
    if (this.connectionTimeout) {
      clearTimeout(this.connectionTimeout);
      this.connectionTimeout = null;
    }

    if (this.ws && this.ws.readyState !== WebSocket.CLOSED) {
      this.ws.close(1000, 'Normal closure');
    }
    
    this.ws = null;
  }

  isConnected(): boolean {
    return !this.isDestroyed && this.ws?.readyState === WebSocket.OPEN;
  }

  send(data: any): boolean {
    if (!this.isConnected()) {
      console.warn('⚠️ WebSocket이 연결되지 않음');
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

  subscribeToSession(sessionId: string): boolean {
    return this.send({
      type: 'subscribe_session',
      session_id: sessionId,
      timestamp: Date.now()
    });
  }

  // 콜백 설정 메서드들
  setOnMessage(callback: (data: PipelineProgress) => void): void {
    if (!this.isDestroyed) {
      this.onMessageCallback = callback;
    }
  }

  setOnConnected(callback: () => void): void {
    if (!this.isDestroyed) {
      this.onConnectedCallback = callback;
    }
  }

  setOnDisconnected(callback: () => void): void {
    if (!this.isDestroyed) {
      this.onDisconnectedCallback = callback;
    }
  }

  setOnError(callback: (error: any) => void): void {
    if (!this.isDestroyed) {
      this.onErrorCallback = callback;
    }
  }

  cleanup(): void {
    console.log('🧹 SafeWebSocketManager 정리 시작');
    
    this.isDestroyed = true;
    this.disconnect();
    
    // 콜백 정리
    this.onMessageCallback = undefined;
    this.onConnectedCallback = undefined;
    this.onDisconnectedCallback = undefined;
    this.onErrorCallback = undefined;
    
    console.log('✅ SafeWebSocketManager 정리 완료');
  }

  getStatus(): any {
    return {
      connected: this.isConnected(),
      connecting: this.isConnecting,
      destroyed: this.isDestroyed,
      reconnectAttempts: this.reconnectAttempts,
      url: this.url
    };
  }
}

// =================================================================
// 🔧 개선된 API 클라이언트 (재시도 및 캐싱)
// =================================================================

class SafeAPIClient {
  private baseURL: string;
  private abortController: AbortController | null = null;
  private cache = new Map<string, { data: any; timestamp: number }>();
  private cacheTimeout = 30000; // 30초

  constructor(baseURL: string = 'http://localhost:8000') {
    this.baseURL = baseURL.replace(/\/$/, '');
    console.log('🔧 SafeAPIClient 생성:', this.baseURL);
  }

  private getCacheKey(url: string, options?: any): string {
    return `${url}_${JSON.stringify(options || {})}`;
  }

  private getCached(key: string): any | null {
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return cached.data;
    }
    this.cache.delete(key);
    return null;
  }

  private setCache(key: string, data: any): void {
    this.cache.set(key, { data, timestamp: Date.now() });
  }

  private async fetchWithRetry(
    url: string, 
    options: RequestInit = {}, 
    maxRetries = 3
  ): Promise<Response> {
    let lastError: Error | null = null;

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        // 이전 요청 취소
        if (this.abortController) {
          this.abortController.abort();
        }
        
        this.abortController = new AbortController();
        
        const response = await fetch(url, {
          ...options,
          signal: this.abortController.signal,
          headers: {
            'Content-Type': 'application/json',
            ...options.headers
          }
        });

        if (response.ok) {
          return response;
        }

        // HTTP 오류도 재시도
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);

      } catch (error: any) {
        lastError = error;
        
        if (error.name === 'AbortError') {
          throw error; // 취소된 요청은 재시도하지 않음
        }

        if (attempt < maxRetries - 1) {
          const delay = Math.min(1000 * Math.pow(2, attempt), 5000);
          console.log(`🔄 API 재시도 ${attempt + 1}/${maxRetries} (${delay}ms 후)`);
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }

    throw lastError || new Error('Unknown fetch error');
  }

  async processVirtualTryOn(
    request: VirtualTryOnRequest,
    onProgress?: (progress: PipelineProgress) => void
  ): Promise<VirtualTryOnResponse> {
    console.log('🎯 가상 피팅 처리 시작');

    const formData = new FormData();
    formData.append('person_image', request.person_image);
    formData.append('clothing_image', request.clothing_image);
    formData.append('height', request.height.toString());
    formData.append('weight', request.weight.toString());
    formData.append('quality_mode', request.quality_mode || 'balanced');
    
    if (request.session_id) {
      formData.append('session_id', request.session_id);
    }

    if (request.enable_realtime) {
      formData.append('enable_realtime', 'true');
    }

    try {
      const response = await this.fetchWithRetry(`${this.baseURL}/api/virtual-tryon`, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      console.log('✅ 가상 피팅 처리 완료');
      return result;

    } catch (error) {
      console.error('❌ 가상 피팅 처리 실패:', error);
      throw error;
    }
  }

  async healthCheck(): Promise<boolean> {
    const cacheKey = this.getCacheKey('/health');
    const cached = this.getCached(cacheKey);
    
    if (cached !== null) {
      return cached;
    }

    try {
      const response = await this.fetchWithRetry(`${this.baseURL}/health`, {}, 2);
      const isHealthy = response.ok;
      
      this.setCache(cacheKey, isHealthy);
      return isHealthy;
    } catch (error) {
      console.error('❌ 헬스체크 실패:', error);
      return false;
    }
  }

  async warmupPipeline(qualityMode: string = 'balanced'): Promise<void> {
    console.log('🔥 파이프라인 워밍업 시작');

    try {
      const response = await this.fetchWithRetry(`${this.baseURL}/api/pipeline/warmup`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `quality_mode=${qualityMode}`,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || '워밍업 실패');
      }

      console.log('✅ 파이프라인 워밍업 완료');
    } catch (error) {
      console.error('❌ 파이프라인 워밍업 실패:', error);
      throw error;
    }
  }

  async getPipelineStatus(): Promise<any> {
    try {
      const response = await this.fetchWithRetry(`${this.baseURL}/api/pipeline/status`);
      return await response.json();
    } catch (error) {
      console.error('❌ 파이프라인 상태 조회 실패:', error);
      throw error;
    }
  }

  async getSystemStats(): Promise<any> {
    try {
      const response = await this.fetchWithRetry(`${this.baseURL}/stats`);
      return await response.json();
    } catch (error) {
      console.error('❌ 시스템 통계 조회 실패:', error);
      throw error;
    }
  }

  cancelCurrentRequest(): void {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
  }

  clearCache(): void {
    this.cache.clear();
    console.log('🗑️ API 캐시 정리됨');
  }

  cleanup(): void {
    console.log('🧹 SafeAPIClient 정리 시작');
    
    this.cancelCurrentRequest();
    this.clearCache();
    
    console.log('✅ SafeAPIClient 정리 완료');
  }
}

// =================================================================
// 🔧 8단계 파이프라인 정의
// =================================================================

const PIPELINE_STEPS = [
  { id: 1, name: 'human_parsing', description: '인체 파싱 (20개 부위)', korean: '인체 파싱' },
  { id: 2, name: 'pose_estimation', description: '포즈 추정 (18개 키포인트)', korean: '포즈 추정' },
  { id: 3, name: 'cloth_segmentation', description: '의류 세그멘테이션', korean: '의류 분석' },
  { id: 4, name: 'geometric_matching', description: '기하학적 매칭', korean: '매칭 분석' },
  { id: 5, name: 'cloth_warping', description: '옷 워핑', korean: '의류 변형' },
  { id: 6, name: 'virtual_fitting', description: '가상 피팅 생성', korean: '가상 피팅' },
  { id: 7, name: 'post_processing', description: '후처리', korean: '품질 향상' },
  { id: 8, name: 'quality_assessment', description: '품질 평가', korean: '품질 검증' }
];

// =================================================================
// 🔧 메인 usePipeline Hook (완전한 수정 버전)
// =================================================================

export const usePipeline = (options: UsePipelineOptions = {}) => {
  const [mounted, setMounted] = useState(true);
  
  // 기본 상태 (App.tsx에서 사용하는 모든 상태)
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const [currentStep, setCurrentStep] = useState('');
  const [stepProgress, setStepProgress] = useState(0);
  const [result, setResult] = useState<VirtualTryOnResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isHealthy, setIsHealthy] = useState(false);
  const [connectionAttempts, setConnectionAttempts] = useState(0);
  const [lastConnectionAttempt, setLastConnectionAttempt] = useState<Date | null>(null);

  // 8단계 파이프라인 확장 상태
  const [currentPipelineStep, setCurrentPipelineStep] = useState(0);
  const [pipelineSteps, setPipelineSteps] = useState<PipelineStep[]>(
    PIPELINE_STEPS.map(step => ({
      id: step.id,
      name: step.name,
      korean: step.korean,
      description: step.description,
      status: 'pending',
      progress: 0
    }))
  );
  const [stepResults, setStepResults] = useState<{ [stepId: number]: any }>({});
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionActive, setSessionActive] = useState(false);

  // 서비스 참조 (useRef로 안정적 관리)
  const wsManager = useRef<SafeWebSocketManager | null>(null);
  const apiClient = useRef<SafeAPIClient | null>(null);
  const healthCheckInterval = useRef<NodeJS.Timeout | null>(null);
  const initializationRef = useRef<boolean>(false);

  // 설정 메모이제이션
  const config = useMemo(() => ({
    baseURL: options.baseURL || 'http://localhost:8000',
    wsURL: options.wsURL || options.baseURL?.replace('http', 'ws') || 'ws://localhost:8000',
    autoHealthCheck: options.autoHealthCheck ?? true,
    healthCheckInterval: options.healthCheckInterval || 30000,
    autoReconnect: options.autoReconnect ?? true,
    maxReconnectAttempts: options.maxReconnectAttempts || 3,
    ...options
  }), [options]);

  // =================================================================
  // 🔧 WebSocket 메시지 핸들러
  // =================================================================

  const handleWebSocketMessage = useCallback((data: PipelineProgress) => {
    if (!mounted) return;

    console.log('📨 WebSocket 메시지 수신:', data.type, data);

    switch (data.type) {
      case 'connection_established':
        setIsConnected(true);
        setSessionId(prev => data.session_id || prev);
        setError(null);
        break;

      case 'pipeline_progress':
        setProgress(data.progress);
        setProgressMessage(data.message);
        setCurrentStep(data.step_name || '');
        break;

      case 'step_start':
        if (data.step_id) {
          setCurrentPipelineStep(data.step_id);
          setPipelineSteps(prev => prev.map(step => 
            step.id === data.step_id 
              ? { ...step, status: 'processing', progress: 0 }
              : step
          ));
          setProgressMessage(data.message || `${PIPELINE_STEPS.find(s => s.id === data.step_id)?.korean} 처리 시작`);
        }
        break;

      case 'step_progress':
        if (data.step_id) {
          setStepProgress(data.progress);
          setPipelineSteps(prev => prev.map(step => 
            step.id === data.step_id 
              ? { ...step, progress: data.progress }
              : step
          ));
        }
        break;

      case 'step_complete':
        if (data.step_id) {
          const stepResult = data.result || { success: true, step_id: data.step_id };
          setStepResults(prev => ({ ...prev, [data.step_id!]: stepResult }));
          setPipelineSteps(prev => prev.map(step => 
            step.id === data.step_id 
              ? { ...step, status: 'completed', progress: 100 }
              : step
          ));
        }
        break;

      case 'step_error':
        if (data.step_id) {
          setPipelineSteps(prev => prev.map(step => 
            step.id === data.step_id 
              ? { ...step, status: 'failed', progress: 0 }
              : step
          ));
          setError(data.message || `단계 ${data.step_id} 처리 실패`);
        }
        break;

      case 'pipeline_completed':
        setIsProcessing(false);
        setProgress(100);
        setProgressMessage('8단계 파이프라인 완료!');
        setSessionActive(false);
        break;

      case 'pipeline_error':
        setIsProcessing(false);
        setError(data.message || '파이프라인 처리 중 오류가 발생했습니다.');
        setProgress(0);
        setProgressMessage('');
        setSessionActive(false);
        break;

      case 'pong':
        // 하트비트 응답 - 연결 상태 확인
        console.log('💓 WebSocket 하트비트 응답');
        break;

      default:
        console.log('❓ 알 수 없는 메시지 타입:', data.type);
    }
  }, [mounted]);

  // =================================================================
  // 🔧 서비스 초기화 (한 번만 실행)
  // =================================================================

  const initializeServices = useCallback(() => {
    if (initializationRef.current || !mounted) {
      return;
    }

    console.log('🚀 usePipeline 서비스 초기화 시작');
    initializationRef.current = true;

    // API 클라이언트 초기화
    if (!apiClient.current) {
      apiClient.current = new SafeAPIClient(config.baseURL);
    }

    // WebSocket 매니저 초기화
    if (!wsManager.current) {
      const wsUrl = `${config.wsURL}/api/ws/pipeline-progress`;
      wsManager.current = new SafeWebSocketManager(wsUrl, config);
      
      wsManager.current.setOnMessage(handleWebSocketMessage);
      wsManager.current.setOnConnected(() => {
        if (mounted) {
          setIsConnected(true);
          setError(null);
          setConnectionAttempts(prev => prev + 1);
          setLastConnectionAttempt(new Date());
          console.log('✅ WebSocket 연결됨');
        }
      });
      wsManager.current.setOnDisconnected(() => {
        if (mounted) {
          setIsConnected(false);
          console.log('❌ WebSocket 연결 해제됨');
        }
      });
      wsManager.current.setOnError((error) => {
        if (mounted) {
          setIsConnected(false);
          console.error('❌ WebSocket 오류:', error);
        }
      });
    }

    console.log('✅ usePipeline 서비스 초기화 완료');
  }, [config.baseURL, config.wsURL, handleWebSocketMessage, mounted, config]);

  // =================================================================
  // 🔧 메인 API 함수들 (App.tsx 완전 호환)
  // =================================================================

  const processVirtualTryOn = useCallback(async (request: VirtualTryOnRequest): Promise<VirtualTryOnResponse | void> => {
    if (!mounted) return;

    try {
      initializeServices();

      // 새 세션 시작
      const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      setSessionId(newSessionId);
      setSessionActive(true);

      // WebSocket 연결 확인
      if (!wsManager.current?.isConnected()) {
        console.log('🔄 WebSocket 재연결 시도...');
        await wsManager.current?.connect();
      }

      // 처리 시작 상태 설정
      setIsProcessing(true);
      setProgress(0);
      setProgressMessage('8단계 AI 파이프라인을 시작합니다...');
      setResult(null);
      setError(null);
      setCurrentPipelineStep(1);

      // 파이프라인 단계 초기화
      setPipelineSteps(PIPELINE_STEPS.map(step => ({
        id: step.id,
        name: step.name,
        korean: step.korean,
        description: step.description,
        status: 'pending',
        progress: 0
      })));
      setStepResults({});

      console.log('🎯 8단계 가상 피팅 처리 시작', { sessionId: newSessionId });

      // 세션 구독
      wsManager.current?.subscribeToSession(newSessionId);

      // API 처리
      const response = await apiClient.current!.processVirtualTryOn({
        ...request,
        session_id: newSessionId,
        enable_realtime: true
      });

      if (mounted) {
        setResult(response);
        setIsProcessing(false);
        setProgress(100);
        setProgressMessage('8단계 파이프라인 완료!');
        setSessionActive(false);
      }

      console.log('✅ 8단계 가상 피팅 처리 완료');
      return response;

    } catch (error) {
      if (!mounted) return;

      const errorMessage = error instanceof Error ? error.message : '알 수 없는 오류가 발생했습니다.';
      
      setIsProcessing(false);
      setError(errorMessage);
      setProgress(0);
      setProgressMessage('');
      setSessionActive(false);

      console.error('❌ 8단계 가상 피팅 처리 실패:', error);
      throw error;
    }
  }, [initializeServices, mounted]);

  const clearResult = useCallback(() => {
    if (!mounted) return;

    setResult(null);
    setProgress(0);
    setProgressMessage('');
    setPipelineSteps(PIPELINE_STEPS.map(step => ({
      id: step.id,
      name: step.name,
      korean: step.korean,
      description: step.description,
      status: 'pending',
      progress: 0
    })));
    setStepResults({});
  }, [mounted]);

  const clearError = useCallback(() => {
    if (mounted) {
      setError(null);
    }
  }, [mounted]);

  const reset = useCallback(() => {
    if (!mounted) return;

    setIsProcessing(false);
    setProgress(0);
    setProgressMessage('');
    setCurrentStep('');
    setStepProgress(0);
    setResult(null);
    setError(null);
    setCurrentPipelineStep(0);
    setPipelineSteps(PIPELINE_STEPS.map(step => ({
      id: step.id,
      name: step.name,
      korean: step.korean,
      description: step.description,
      status: 'pending',
      progress: 0
    })));
    setStepResults({});
    setSessionId(null);
    setSessionActive(false);
  }, [mounted]);

  const connect = useCallback(async (): Promise<boolean> => {
    if (!mounted) return false;

    initializeServices();
    setConnectionAttempts(prev => prev + 1);
    setLastConnectionAttempt(new Date());
    
    const connected = await wsManager.current?.connect();
    return connected || false;
  }, [initializeServices, mounted]);

  const disconnect = useCallback(() => {
    wsManager.current?.disconnect();
    if (mounted) {
      setIsConnected(false);
    }
  }, [mounted]);

  const reconnect = useCallback(async (): Promise<boolean> => {
    if (!mounted) return false;

    disconnect();
    await new Promise(resolve => setTimeout(resolve, 1000));
    return await connect();
  }, [connect, disconnect, mounted]);

  const checkHealth = useCallback(async (): Promise<boolean> => {
    if (!mounted) return false;

    try {
      initializeServices();
      const healthy = await apiClient.current!.healthCheck();
      if (mounted) {
        setIsHealthy(healthy);
      }
      return healthy;
    } catch (error) {
      console.error('❌ 헬스체크 실패:', error);
      if (mounted) {
        setIsHealthy(false);
      }
      return false;
    }
  }, [initializeServices, mounted]);

  const testConnection = useCallback(async () => {
    if (!mounted) return;

    try {
      setIsProcessing(true);
      setProgressMessage('연결 테스트 중...');

      const wsConnected = await connect();
      const healthOk = await checkHealth();

      if (!wsConnected) {
        throw new Error('WebSocket 연결 실패');
      }

      if (!healthOk) {
        throw new Error('API 헬스체크 실패');
      }

      if (mounted) {
        setError(null);
        setProgressMessage('연결 테스트 완료');
      }
      console.log('✅ 연결 테스트 완료');

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '연결 테스트 실패';
      if (mounted) {
        setError(errorMessage);
      }
      console.error('❌ 연결 테스트 실패:', error);
    } finally {
      if (mounted) {
        setIsProcessing(false);
      }
    }
  }, [connect, checkHealth, mounted]);

  const warmupPipeline = useCallback(async (qualityMode: string = 'balanced') => {
    if (!mounted) return;

    try {
      setIsProcessing(true);
      setProgressMessage('파이프라인 워밍업 중...');

      initializeServices();
      await apiClient.current!.warmupPipeline(qualityMode);
      
      if (mounted) {
        setError(null);
        setProgressMessage('워밍업 완료');
      }
      console.log('✅ 파이프라인 워밍업 완료');

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '워밍업 실패';
      if (mounted) {
        setError(errorMessage);
      }
      console.error('❌ 파이프라인 워밍업 실패:', error);
    } finally {
      if (mounted) {
        setIsProcessing(false);
      }
    }
  }, [initializeServices, mounted]);

  const getPipelineStatus = useCallback(async () => {
    if (!mounted) return;

    try {
      initializeServices();
      const status = await apiClient.current!.getPipelineStatus();
      console.log('📊 파이프라인 상태:', status);
      return status;
    } catch (error) {
      console.error('❌ 파이프라인 상태 조회 실패:', error);
      return null;
    }
  }, [initializeServices, mounted]);

  const getSystemStats = useCallback(async () => {
    if (!mounted) return;

    try {
      initializeServices();
      const stats = await apiClient.current!.getSystemStats();
      console.log('📈 시스템 통계:', stats);
      return stats;
    } catch (error) {
      console.error('❌ 시스템 통계 조회 실패:', error);
      return null;
    }
  }, [initializeServices, mounted]);

  // =================================================================
  // 🔧 생명주기 관리 (React 18 StrictMode 완전 대응)
  // =================================================================

  // 컴포넌트 마운트 상태 추적
  useEffect(() => {
    setMounted(true);
    console.log('🔧 usePipeline Hook 마운트됨');

    return () => {
      setMounted(false);
      console.log('🔧 usePipeline Hook 언마운트됨');
    };
  }, []);

  // 서비스 초기화 및 연결
  useEffect(() => {
    if (!mounted) return;

    let isMounted = true;
    
    const initAndConnect = async () => {
      if (!isMounted || !mounted) return;
      
      initializeServices();
      
      if (config.autoReconnect && isMounted && mounted) {
        // 지연을 두고 연결 (React 18 Strict Mode 대응)
        setTimeout(() => {
          if (isMounted && mounted) {
            connect();
          }
        }, 500);
      }
    };

    initAndConnect();

    return () => {
      isMounted = false;
    };
  }, [mounted]); // config 의존성 제거

  // 컴포넌트 언마운트 시 정리
  useEffect(() => {
    return () => {
      console.log('🧹 usePipeline 최종 정리 시작');
      
      try {
        // WebSocket 정리
        if (wsManager.current) {
          wsManager.current.cleanup();
          wsManager.current = null;
        }
        
        // API 클라이언트 정리
        if (apiClient.current) {
          apiClient.current.cleanup();
          apiClient.current = null;
        }
        
        // 헬스체크 인터벌 정리
        if (healthCheckInterval.current) {
          clearInterval(healthCheckInterval.current);
          healthCheckInterval.current = null;
        }
        
        // 초기화 플래그 리셋
        initializationRef.current = false;
        
        console.log('✅ usePipeline 최종 정리 완료');
      } catch (error) {
        console.warn('⚠️ usePipeline 정리 중 오류:', error);
      }
    };
  }, []);

  // 자동 헬스체크
  useEffect(() => {
    if (!config.autoHealthCheck || !mounted) return;

    let isMounted = true;
    let intervalId: NodeJS.Timeout | null = null;

    const startHealthCheck = async () => {
      if (!isMounted || !mounted) return;

      // 초기 헬스체크
      await checkHealth();

      if (isMounted && mounted) {
        intervalId = setInterval(() => {
          if (isMounted && mounted) {
            checkHealth();
          }
        }, config.healthCheckInterval);
      }
    };

    // 약간의 지연 후 시작
    const timer = setTimeout(startHealthCheck, 1000);

    return () => {
      isMounted = false;
      clearTimeout(timer);
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [config.autoHealthCheck, config.healthCheckInterval, mounted]);

  // =================================================================
  // 🔧 Hook 반환값 (App.tsx 완전 호환)
  // =================================================================

  return {
    // App.tsx에서 사용하는 모든 상태
    isProcessing,
    progress,
    progressMessage,
    currentStep,
    stepProgress,
    result,
    error,
    isConnected,
    isHealthy,
    connectionAttempts,
    lastConnectionAttempt,

    // 8단계 파이프라인 확장 상태
    currentPipelineStep,
    pipelineSteps,
    stepResults,
    sessionId,
    sessionActive,

    // App.tsx에서 사용하는 모든 액션
    processVirtualTryOn,
    clearResult,
    clearError,
    reset,
    connect,
    disconnect,
    reconnect,
    checkHealth,
    testConnection,
    warmupPipeline,
    getPipelineStatus,
    getSystemStats,

    // 추가 유틸리티 함수들
    sendHeartbeat: () => wsManager.current?.send({ type: 'ping', timestamp: Date.now() }),
    getConnectionStatus: () => wsManager.current?.getStatus() || null,
    clearCache: () => apiClient.current?.clearCache(),
    exportLogs: () => {
      const logs = {
        isProcessing,
        progress,
        result,
        error,
        pipelineSteps,
        stepResults,
        config,
        mounted,
        timestamp: new Date().toISOString()
      };
      console.log('📋 usePipeline 상태 로그:', logs);
      return logs;
    }
  };
};

// =================================================================
// 🔧 헬스체크 전용 Hook (App.tsx 호환)
// =================================================================

export const usePipelineHealth = (options: UsePipelineOptions = {}) => {
  const [mounted, setMounted] = useState(true);
  const [isHealthy, setIsHealthy] = useState(false);
  const [isChecking, setIsChecking] = useState(false);
  const [lastCheck, setLastCheck] = useState<Date | null>(null);

  const apiClient = useMemo(() => new SafeAPIClient(options.baseURL), [options.baseURL]);

  const checkHealth = useCallback(async () => {
    if (!mounted) return;

    setIsChecking(true);

    try {
      const healthy = await apiClient.healthCheck();
      if (mounted) {
        setIsHealthy(healthy);
        setLastCheck(new Date());
      }
    } catch (error) {
      if (mounted) {
        setIsHealthy(false);
      }
      console.error('❌ 헬스체크 실패:', error);
    } finally {
      if (mounted) {
        setIsChecking(false);
      }
    }
  }, [apiClient, mounted]);

  useEffect(() => {
    setMounted(true);
    return () => setMounted(false);
  }, []);

  useEffect(() => {
    if (!mounted) return;

    checkHealth();

    if (options.autoHealthCheck) {
      const interval = options.healthCheckInterval || 30000;
      const intervalId = setInterval(() => {
        if (mounted) {
          checkHealth();
        }
      }, interval);
      return () => clearInterval(intervalId);
    }
  }, [checkHealth, options.autoHealthCheck, options.healthCheckInterval, mounted]);

  useEffect(() => {
    return () => {
      apiClient.cleanup();
    };
  }, [apiClient]);

  return {
    isHealthy,
    isChecking,
    lastCheck,
    checkHealth
  };
};

export default usePipeline;