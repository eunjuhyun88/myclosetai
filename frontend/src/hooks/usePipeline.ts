/**
 * MyCloset AI 개선된 파이프라인 React Hook
 * 완전한 WebSocket 통합과 향상된 에러 처리
 */

import { useState, useCallback, useRef, useEffect, useMemo } from 'react';

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
  quality_mode: 'fast' | 'balanced' | 'quality';
  session_id?: string;
}

export interface VirtualTryOnResponse {
  success: boolean;
  fitted_image?: string; // base64
  processing_time: number;
  confidence: number;
  measurements: Record<string, number>;
  clothing_analysis: ClothingAnalysis;
  fit_score: number;
  recommendations: string[];
  quality_metrics: QualityMetrics;
  memory_usage?: Record<string, number>;
  step_times?: Record<string, number>;
  error_message?: string;
  session_id?: string;
}

export interface PipelineProgress {
  type: 'pipeline_progress' | 'step_update' | 'connection_established' | 'error' | 'completed';
  session_id?: string;
  step_id?: number;
  step_name?: string;
  progress: number;
  message: string;
  timestamp: number;
  data?: any;
  steps?: Array<{
    id: number;
    name: string;
    status: 'pending' | 'processing' | 'completed' | 'error';
    progress: number;
    error?: string;
  }>;
}

export interface PipelineStatus {
  status: string;
  device: string;
  memory_usage: Record<string, number>;
  models_loaded: string[];
  active_connections: number;
  pipeline_ready: boolean;
}

export interface SystemStats {
  total_requests: number;
  successful_requests: number;
  average_processing_time: number;
  average_quality_score: number;
  peak_memory_usage: number;
  uptime: number;
  current_connections: number;
}

export interface ConnectionConfig {
  baseURL?: string;
  wsURL?: string;
  autoReconnect?: boolean;
  maxReconnectAttempts?: number;
  reconnectInterval?: number;
  heartbeatInterval?: number;
  connectionTimeout?: number;
}

export interface UsePipelineOptions extends ConnectionConfig {
  autoHealthCheck?: boolean;
  healthCheckInterval?: number;
  persistSession?: boolean;
  enableDetailedProgress?: boolean;
}

export interface UsePipelineState {
  // 처리 상태
  isProcessing: boolean;
  progress: number;
  progressMessage: string;
  currentStep: string;
  stepProgress: number;
  
  // 결과 및 에러
  result: VirtualTryOnResponse | null;
  error: string | null;
  
  // 연결 상태
  isConnected: boolean;
  isHealthy: boolean;
  connectionAttempts: number;
  lastConnectionAttempt: Date | null;
  
  // 시스템 정보
  pipelineStatus: PipelineStatus | null;
  systemStats: SystemStats | null;
  
  // 세션 정보
  sessionId: string | null;
  
  // 상세 진행 정보
  steps: Array<{
    id: number;
    name: string;
    status: 'pending' | 'processing' | 'completed' | 'error';
    progress: number;
    error?: string;
    duration?: number;
  }>;
}

export interface UsePipelineActions {
  // 주요 기능
  processVirtualTryOn: (request: VirtualTryOnRequest) => Promise<void>;
  
  // 상태 관리
  clearResult: () => void;
  clearError: () => void;
  reset: () => void;
  
  // 연결 관리
  connect: () => Promise<boolean>;
  disconnect: () => void;
  reconnect: () => Promise<boolean>;
  
  // 정보 조회
  checkHealth: () => Promise<boolean>;
  getPipelineStatus: () => Promise<void>;
  getSystemStats: () => Promise<void>;
  
  // 파이프라인 관리
  warmupPipeline: (qualityMode?: string) => Promise<void>;
  testConnection: () => Promise<void>;
  
  // 유틸리티
  exportLogs: () => void;
  sendHeartbeat: () => void;
}

// WebSocket 연결 관리 클래스
class WebSocketManager {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts: number;
  private reconnectInterval: number;
  private heartbeatInterval: number;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private connectionTimeout: number;
  private autoReconnect: boolean;
  
  private onMessageCallback?: (data: any) => void;
  private onConnectedCallback?: () => void;
  private onDisconnectedCallback?: () => void;
  private onErrorCallback?: (error: Event) => void;

  constructor(url: string, config: ConnectionConfig = {}) {
    this.url = url;
    this.maxReconnectAttempts = config.maxReconnectAttempts || 5;
    this.reconnectInterval = config.reconnectInterval || 3000;
    this.heartbeatInterval = config.heartbeatInterval || 30000;
    this.connectionTimeout = config.connectionTimeout || 10000;
    this.autoReconnect = config.autoReconnect ?? true;
  }

  connect(): Promise<boolean> {
    return new Promise((resolve) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve(true);
        return;
      }

      try {
        this.ws = new WebSocket(this.url);
        
        const connectionTimer = setTimeout(() => {
          if (this.ws?.readyState !== WebSocket.OPEN) {
            this.ws?.close();
            console.error('WebSocket 연결 타임아웃');
            resolve(false);
          }
        }, this.connectionTimeout);

        this.ws.onopen = () => {
          clearTimeout(connectionTimer);
          this.reconnectAttempts = 0;
          console.log('✅ WebSocket 연결 성공:', this.url);
          
          this.startHeartbeat();
          this.onConnectedCallback?.();
          resolve(true);
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.onMessageCallback?.(data);
          } catch (error) {
            console.error('WebSocket 메시지 파싱 오류:', error);
          }
        };

        this.ws.onclose = (event) => {
          clearTimeout(connectionTimer);
          this.stopHeartbeat();
          console.log('🔌 WebSocket 연결 종료:', event.code, event.reason);
          
          this.onDisconnectedCallback?.();
          
          if (this.autoReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
          
          if (this.reconnectAttempts === 0) {
            resolve(false);
          }
        };

        this.ws.onerror = (error) => {
          clearTimeout(connectionTimer);
          console.error('❌ WebSocket 오류:', error);
          this.onErrorCallback?.(error);
          
          if (this.reconnectAttempts === 0) {
            resolve(false);
          }
        };

      } catch (error) {
        console.error('WebSocket 생성 실패:', error);
        resolve(false);
      }
    });
  }

  private scheduleReconnect(): void {
    this.reconnectAttempts++;
    console.log(`🔄 재연결 시도 ${this.reconnectAttempts}/${this.maxReconnectAttempts} (${this.reconnectInterval}ms 후)`);
    
    setTimeout(() => {
      this.connect();
    }, this.reconnectInterval * this.reconnectAttempts);
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      this.send({ type: 'ping', timestamp: Date.now() });
    }, this.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  send(data: any): boolean {
    if (this.ws?.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify(data));
        return true;
      } catch (error) {
        console.error('WebSocket 메시지 전송 실패:', error);
        return false;
      }
    }
    return false;
  }

  disconnect(): void {
    this.autoReconnect = false;
    this.stopHeartbeat();
    
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  setOnMessage(callback: (data: any) => void): void {
    this.onMessageCallback = callback;
  }

  setOnConnected(callback: () => void): void {
    this.onConnectedCallback = callback;
  }

  setOnDisconnected(callback: () => void): void {
    this.onDisconnectedCallback = callback;
  }

  setOnError(callback: (error: Event) => void): void {
    this.onErrorCallback = callback;
  }

  getReconnectAttempts(): number {
    return this.reconnectAttempts;
  }
}

// 유틸리티 클래스
export class PipelineUtils {
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
   * 에러 메시지를 사용자 친화적으로 변환
   */
  static getUserFriendlyError(error: string): string {
    const errorMappings: Record<string, string> = {
      'Network Error': '네트워크 연결을 확인해주세요.',
      'timeout': '처리 시간이 초과되었습니다. 다시 시도해주세요.',
      'invalid image': '지원되지 않는 이미지 형식입니다.',
      'file too large': '파일 크기가 너무 큽니다. 10MB 이하로 업로드해주세요.',
      'server error': '서버에 일시적인 문제가 발생했습니다.',
      'connection failed': 'WebSocket 연결에 실패했습니다.',
      'pipeline not ready': '파이프라인이 준비되지 않았습니다.',
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
}

// 메인 Hook
export const usePipeline = (options: UsePipelineOptions = {}): UsePipelineState & UsePipelineActions => {
  // 기본 설정
  const config = useMemo(() => ({
    baseURL: options.baseURL || 'http://localhost:8000',
    wsURL: options.wsURL || options.baseURL?.replace('http', 'ws') || 'ws://localhost:8000',
    autoReconnect: options.autoReconnect ?? true,
    maxReconnectAttempts: options.maxReconnectAttempts || 5,
    reconnectInterval: options.reconnectInterval || 3000,
    heartbeatInterval: options.heartbeatInterval || 30000,
    connectionTimeout: options.connectionTimeout || 10000,
    autoHealthCheck: options.autoHealthCheck ?? true,
    healthCheckInterval: options.healthCheckInterval || 30000,
    persistSession: options.persistSession ?? true,
    enableDetailedProgress: options.enableDetailedProgress ?? true,
  }), [options]);

  // 상태 관리
  const [state, setState] = useState<UsePipelineState>({
    isProcessing: false,
    progress: 0,
    progressMessage: '',
    currentStep: '',
    stepProgress: 0,
    result: null,
    error: null,
    isConnected: false,
    isHealthy: false,
    connectionAttempts: 0,
    lastConnectionAttempt: null,
    pipelineStatus: null,
    systemStats: null,
    sessionId: null,
    steps: []
  });

  // WebSocket 관리자
  const wsManager = useRef<WebSocketManager | null>(null);
  const healthCheckTimer = useRef<NodeJS.Timeout | null>(null);
  const messageLog = useRef<any[]>([]);

  // 상태 업데이트 헬퍼
  const updateState = useCallback((updates: Partial<UsePipelineState>) => {
    setState(prev => ({ ...prev, ...updates }));
  }, []);

  // WebSocket 메시지 핸들러
  const handleWebSocketMessage = useCallback((data: PipelineProgress) => {
    console.log('📨 WebSocket 메시지:', data);
    messageLog.current.push({ ...data, receivedAt: new Date() });

    switch (data.type) {
      case 'connection_established':
        updateState({
          isConnected: true,
          sessionId: data.session_id || state.sessionId,
          error: null
        });
        break;

      case 'pipeline_progress':
        updateState({
          progress: data.progress,
          progressMessage: data.message,
          currentStep: data.step_name || state.currentStep
        });
        break;

      case 'step_update':
        updateState({
          currentStep: data.step_name || '',
          stepProgress: data.progress,
          steps: data.steps || state.steps
        });
        break;

      case 'completed':
        updateState({
          isProcessing: false,
          progress: 100,
          progressMessage: '처리 완료!'
        });
        break;

      case 'error':
        updateState({
          isProcessing: false,
          error: PipelineUtils.getUserFriendlyError(data.message),
          progress: 0,
          progressMessage: ''
        });
        break;

      default:
        console.log('알 수 없는 메시지 타입:', data.type);
    }
  }, [updateState, state.sessionId, state.currentStep, state.steps]);

  // WebSocket 연결 설정
  const connect = useCallback(async (): Promise<boolean> => {
    if (wsManager.current?.isConnected()) {
      return true;
    }

    updateState({
      connectionAttempts: state.connectionAttempts + 1,
      lastConnectionAttempt: new Date()
    });

    try {
      const wsUrl = `${config.wsURL}/api/ws/pipeline-progress`;
      wsManager.current = new WebSocketManager(wsUrl, config);

      wsManager.current.setOnMessage(handleWebSocketMessage);
      wsManager.current.setOnConnected(() => {
        updateState({ isConnected: true, error: null });
      });
      wsManager.current.setOnDisconnected(() => {
        updateState({ isConnected: false });
      });
      wsManager.current.setOnError((error) => {
        updateState({
          isConnected: false,
          error: PipelineUtils.getUserFriendlyError('connection failed')
        });
      });

      const connected = await wsManager.current.connect();
      
      if (connected && state.sessionId) {
        // 세션 구독
        wsManager.current.send({
          type: 'subscribe_session',
          session_id: state.sessionId
        });
      }

      return connected;
    } catch (error) {
      console.error('WebSocket 연결 실패:', error);
      updateState({
        isConnected: false,
        error: PipelineUtils.getUserFriendlyError('connection failed')
      });
      return false;
    }
  }, [config, handleWebSocketMessage, state.connectionAttempts, state.sessionId, updateState]);

  // WebSocket 연결 해제
  const disconnect = useCallback(() => {
    wsManager.current?.disconnect();
    updateState({ isConnected: false });
  }, [updateState]);

  // 재연결
  const reconnect = useCallback(async (): Promise<boolean> => {
    disconnect();
    await new Promise(resolve => setTimeout(resolve, 1000)); // 1초 대기
    return await connect();
  }, [disconnect, connect]);

  // 가상 피팅 처리
  const processVirtualTryOn = useCallback(async (request: VirtualTryOnRequest) => {
    try {
      // 입력 검증
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

      // WebSocket 연결 확인
      if (!wsManager.current?.isConnected()) {
        console.log('🔄 WebSocket 재연결 시도...');
        const connected = await connect();
        if (!connected) {
          throw new Error('WebSocket 연결에 실패했습니다.');
        }
      }

      // 세션 ID 생성 또는 재사용
      const sessionId = request.session_id || 
                       (config.persistSession && state.sessionId) || 
                       `session_${Date.now()}_${Math.random().toString(36).substring(7)}`;

      // 처리 시작
      updateState({
        isProcessing: true,
        progress: 0,
        progressMessage: '처리를 시작합니다...',
        result: null,
        error: null,
        sessionId,
        steps: config.enableDetailedProgress ? [
          { id: 1, name: 'Human Parsing', status: 'pending', progress: 0 },
          { id: 2, name: 'Pose Estimation', status: 'pending', progress: 0 },
          { id: 3, name: 'Cloth Segmentation', status: 'pending', progress: 0 },
          { id: 4, name: 'Geometric Matching', status: 'pending', progress: 0 },
          { id: 5, name: 'Cloth Warping', status: 'pending', progress: 0 },
          { id: 6, name: 'Virtual Fitting', status: 'pending', progress: 0 },
          { id: 7, name: 'Post Processing', status: 'pending', progress: 0 },
          { id: 8, name: 'Quality Assessment', status: 'pending', progress: 0 }
        ] : []
      });

      console.log('🎯 가상 피팅 처리 시작:', { ...request, sessionId });

      // 세션 구독
      wsManager.current?.send({
        type: 'subscribe_session',
        session_id: sessionId
      });

      // FormData 준비
      const formData = new FormData();
      formData.append('person_image', request.person_image);
      formData.append('clothing_image', request.clothing_image);
      formData.append('height', request.height.toString());
      formData.append('weight', request.weight.toString());
      formData.append('quality_mode', request.quality_mode);
      formData.append('session_id', sessionId);

      // API 요청
      const response = await fetch(`${config.baseURL}/api/virtual-tryon-pipeline`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const result: VirtualTryOnResponse = await response.json();

      updateState({
        isProcessing: false,
        result,
        progress: 100,
        progressMessage: '완료!',
        sessionId: result.session_id || sessionId
      });

      console.log('✅ 가상 피팅 처리 완료:', result);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '알 수 없는 오류가 발생했습니다.';
      const friendlyError = PipelineUtils.getUserFriendlyError(errorMessage);
      
      updateState({
        isProcessing: false,
        error: friendlyError,
        progress: 0,
        progressMessage: ''
      });

      console.error('❌ 가상 피팅 처리 실패:', error);
    }
  }, [config, connect, state.sessionId, updateState]);

  // 기타 액션들
  const clearResult = useCallback(() => {
    updateState({
      result: null,
      progress: 0,
      progressMessage: '',
      steps: []
    });
  }, [updateState]);

  const clearError = useCallback(() => {
    updateState({ error: null });
  }, [updateState]);

  const reset = useCallback(() => {
    updateState({
      isProcessing: false,
      progress: 0,
      progressMessage: '',
      currentStep: '',
      stepProgress: 0,
      result: null,
      error: null,
      steps: []
    });
  }, [updateState]);

  const checkHealth = useCallback(async (): Promise<boolean> => {
    try {
      const response = await fetch(`${config.baseURL}/health`);
      const isHealthy = response.ok;
      updateState({ isHealthy });
      return isHealthy;
    } catch (error) {
      console.error('헬스체크 실패:', error);
      updateState({ isHealthy: false });
      return false;
    }
  }, [config.baseURL, updateState]);

  const getPipelineStatus = useCallback(async () => {
    try {
      const response = await fetch(`${config.baseURL}/api/pipeline/status`);
      if (response.ok) {
        const pipelineStatus = await response.json();
        updateState({ pipelineStatus });
      }
    } catch (error) {
      console.error('파이프라인 상태 조회 실패:', error);
    }
  }, [config.baseURL, updateState]);

  const getSystemStats = useCallback(async () => {
    try {
      const response = await fetch(`${config.baseURL}/stats`);
      if (response.ok) {
        const systemStats = await response.json();
        updateState({ systemStats });
      }
    } catch (error) {
      console.error('시스템 통계 조회 실패:', error);
    }
  }, [config.baseURL, updateState]);

  const warmupPipeline = useCallback(async (qualityMode: string = 'balanced') => {
    try {
      updateState({
        isProcessing: true,
        progressMessage: '파이프라인 워밍업 중...'
      });

      const response = await fetch(`${config.baseURL}/api/pipeline/warmup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: `quality_mode=${qualityMode}`,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || '워밍업 실패');
      }
      
      updateState({
        isProcessing: false,
        progressMessage: '워밍업 완료'
      });

      console.log('✅ 파이프라인 워밍업 완료');

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '워밍업 실패';
      updateState({
        isProcessing: false,
        error: errorMessage,
        progressMessage: ''
      });

      console.error('❌ 파이프라인 워밍업 실패:', error);
    }
  }, [config.baseURL, updateState]);

  const testConnection = useCallback(async () => {
    try {
      updateState({
        isProcessing: true,
        progressMessage: '연결 테스트 중...'
      });

      // WebSocket 테스트
      const wsConnected = await connect();
      if (!wsConnected) {
        throw new Error('WebSocket 연결 실패');
      }

      // API 테스트
      const healthOk = await checkHealth();
      if (!healthOk) {
        throw new Error('API 헬스체크 실패');
      }

      updateState({
        isProcessing: false,
        progressMessage: '연결 테스트 완료',
        error: null
      });

      console.log('✅ 연결 테스트 완료');

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '연결 테스트 실패';
      updateState({
        isProcessing: false,
        error: errorMessage,
        progressMessage: ''
      });

      console.error('❌ 연결 테스트 실패:', error);
    }
  }, [connect, checkHealth, updateState]);

  const sendHeartbeat = useCallback(() => {
    wsManager.current?.send({ type: 'ping', timestamp: Date.now() });
  }, []);

  const exportLogs = useCallback(() => {
    const logs = {
      state,
      messageLog: messageLog.current,
      connectionAttempts: wsManager.current?.getReconnectAttempts() || 0,
      timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(logs, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pipeline_logs_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [state]);

  // 자동 연결
  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  // 자동 헬스체크
  useEffect(() => {
    if (config.autoHealthCheck) {
      checkHealth();

      healthCheckTimer.current = setInterval(checkHealth, config.healthCheckInterval);

      return () => {
        if (healthCheckTimer.current) {
          clearInterval(healthCheckTimer.current);
        }
      };
    }
  }, [config.autoHealthCheck, config.healthCheckInterval, checkHealth]);

  // 정리
  useEffect(() => {
    return () => {
      if (healthCheckTimer.current) {
        clearInterval(healthCheckTimer.current);
      }
      disconnect();
    };
  }, [disconnect]);

  return {
    // 상태
    ...state,
    
    // 액션
    processVirtualTryOn,
    clearResult,
    clearError,
    reset,
    connect,
    disconnect,
    reconnect,
    checkHealth,
    getPipelineStatus,
    getSystemStats,
    warmupPipeline,
    testConnection,
    sendHeartbeat,
    exportLogs
  };
};

// 편의 Hook들
export const usePipelineStatus = (options: UsePipelineOptions = {}) => {
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const baseURL = options.baseURL || 'http://localhost:8000';

  const fetchStatus = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${baseURL}/api/pipeline/status`);
      if (!response.ok) {
        throw new Error(`상태 조회 실패: ${response.status}`);
      }
      const pipelineStatus = await response.json();
      setStatus(pipelineStatus);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '상태 조회 실패';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [baseURL]);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  return {
    status,
    isLoading,
    error,
    refetch: fetchStatus
  };
};

export const usePipelineHealth = (options: UsePipelineOptions = {}) => {
  const [isHealthy, setIsHealthy] = useState<boolean>(false);
  const [isChecking, setIsChecking] = useState(false);
  const [lastCheck, setLastCheck] = useState<Date | null>(null);

  const baseURL = options.baseURL || 'http://localhost:8000';

  const checkHealth = useCallback(async () => {
    setIsChecking(true);

    try {
      const response = await fetch(`${baseURL}/health`);
      const healthy = response.ok;
      setIsHealthy(healthy);
      setLastCheck(new Date());
    } catch (error) {
      setIsHealthy(false);
      console.error('헬스체크 실패:', error);
    } finally {
      setIsChecking(false);
    }
  }, [baseURL]);

  useEffect(() => {
    checkHealth();

    if (options.autoHealthCheck) {
      const interval = options.healthCheckInterval || 30000;
      const intervalId = setInterval(checkHealth, interval);
      return () => clearInterval(intervalId);
    }
  }, [checkHealth, options.autoHealthCheck, options.healthCheckInterval]);

  return {
    isHealthy,
    isChecking,
    lastCheck,
    checkHealth
  };
};

export default usePipeline;