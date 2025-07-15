/**
 * MyCloset AI 8단계 파이프라인 React Hook
 * 백엔드 pipeline router와 완전 호환
 * - 단계별 처리 상태 관리
 * - 실시간 WebSocket 연결
 * - 8단계 파이프라인 추적
 */

import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import type {
  UsePipelineOptions,
  UsePipelineState,
  UsePipelineActions,
  VirtualTryOnRequest,
  VirtualTryOnResponse,
  PipelineProgress,
  PipelineStatus,
  SystemStats,
  PipelineStep,
} from '../types/pipeline';

// 서비스 import
import WebSocketManager from '../services/WebSocketManager';
import PipelineAPIClient from '../services/PipelineAPIClient';
import { PipelineUtils } from '../utils/pipelineUtils';

// 8단계 파이프라인 정의
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

// Extended State Interface for 8-step pipeline
interface ExtendedUsePipelineState extends UsePipelineState {
  // 8단계 파이프라인 상태
  currentPipelineStep: number;
  pipelineSteps: PipelineStep[];
  stepResults: { [stepId: number]: any };
  stepProgress: { [stepId: number]: number };
  
  // 세션 관리
  sessionId: string | null;
  sessionActive: boolean;
  
  // WebSocket 상태
  wsConnected: boolean;
  wsConnectionAttempts: number;
  
  // 성능 메트릭
  startTime: number | null;
  stepTimings: { [stepId: number]: number };
}

interface ExtendedUsePipelineActions extends UsePipelineActions {
  // 8단계 파이프라인 액션들
  processStep: (stepId: number, data?: any) => Promise<any>;
  skipToStep: (stepId: number) => void;
  restartPipeline: () => void;
  
  // 세션 관리
  startNewSession: () => string;
  endSession: () => void;
  
  // WebSocket 관리
  connectWebSocket: () => Promise<boolean>;
  disconnectWebSocket: () => void;
  
  // 개별 단계 처리
  processHumanParsing: (image: File) => Promise<any>;
  processPoseEstimation: (image: File) => Promise<any>;
  processClothingAnalysis: (image: File) => Promise<any>;
  processGeometricMatching: (personData: any, clothingData: any) => Promise<any>;
  processClothWarping: (matchingData: any) => Promise<any>;
  processVirtualFitting: (allData: any) => Promise<any>;
  processPostProcessing: (fittingResult: any) => Promise<any>;
  processQualityAssessment: (finalResult: any) => Promise<any>;
}

export const usePipeline = (
  options: UsePipelineOptions = {},
  ...kwargs: any[]
): ExtendedUsePipelineState & ExtendedUsePipelineActions => {
  
  // 기본 설정
  const config = useMemo(() => {
    const baseConfig: UsePipelineOptions = {
      baseURL: options.baseURL || 'http://localhost:8000',
      wsURL: options.wsURL || options.baseURL?.replace('http', 'ws') || 'ws://localhost:8000',
      autoReconnect: options.autoReconnect ?? true,
      maxReconnectAttempts: options.maxReconnectAttempts || 5,
      reconnectInterval: options.reconnectInterval || 3000,
      heartbeatInterval: options.heartbeatInterval || 30000,
      connectionTimeout: options.connectionTimeout || 10000,
      
      // 8단계 파이프라인 설정
      enableStepTracking: options.enableStepTracking ?? true,
      enableRealTimeUpdates: options.enableRealTimeUpdates ?? true,
      stepTimeout: options.stepTimeout || 60000,
      autoRetrySteps: options.autoRetrySteps ?? true,
      maxStepRetries: options.maxStepRetries || 3,
      
      // 백엔드 호환 시스템 파라미터
      device: options.device || PipelineUtils.autoDetectDevice(),
      device_type: options.device_type || PipelineUtils.autoDetectDeviceType(),
      memory_gb: options.memory_gb || 16.0,
      is_m3_max: options.is_m3_max ?? PipelineUtils.detectM3Max(),
      optimization_enabled: options.optimization_enabled ?? true,
      quality_level: options.quality_level || 'balanced',
      
      ...options,
    };
    
    return PipelineUtils.mergeStepSpecificConfig(
      baseConfig,
      kwargs.reduce((acc, kwarg) => ({ ...acc, ...kwarg }), {}),
      PipelineUtils.getSystemParams()
    );
  }, [options, kwargs]);

  // Extended State Management
  const [state, setState] = useState<ExtendedUsePipelineState>({
    // 기본 상태
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
    steps: PIPELINE_STEPS.map(step => ({
      id: step.id,
      name: step.name,
      status: 'pending',
      progress: 0,
      korean: step.korean,
      description: step.description
    })),
    
    // 8단계 파이프라인 확장 상태
    currentPipelineStep: 0,
    pipelineSteps: PIPELINE_STEPS.map(step => ({
      id: step.id,
      name: step.name,
      status: 'pending',
      progress: 0,
      korean: step.korean,
      description: step.description
    })),
    stepResults: {},
    stepProgress: {},
    
    // 세션 관리
    sessionActive: false,
    
    // WebSocket 상태
    wsConnected: false,
    wsConnectionAttempts: 0,
    
    // 성능 메트릭
    startTime: null,
    stepTimings: {}
  });

  // 서비스 인스턴스들
  const wsManager = useRef<WebSocketManager | null>(null);
  const apiClient = useRef<PipelineAPIClient | null>(null);
  const healthCheckTimer = useRef<NodeJS.Timeout | null>(null);
  const stepTimeouts = useRef<{ [stepId: number]: NodeJS.Timeout }>({});
  const isInitialized = useRef<boolean>(false);

  // 상태 업데이트 헬퍼
  const updateState = useCallback((updates: Partial<ExtendedUsePipelineState>) => {
    setState(prev => ({ ...prev, ...updates }));
  }, []);

  // =================================================================
  // 🔧 서비스 초기화
  // =================================================================

  const initializeAPIClient = useCallback(() => {
    if (!apiClient.current) {
      apiClient.current = new PipelineAPIClient(config, ...kwargs);
      PipelineUtils.log('info', '✅ API 클라이언트 초기화 완료');
    }
  }, [config, kwargs]);

  const initializeWebSocketManager = useCallback(() => {
    if (!wsManager.current) {
      const wsUrl = `${config.wsURL}/api/ws/pipeline-progress`;
      wsManager.current = new WebSocketManager(wsUrl, config, ...kwargs);
      PipelineUtils.log('info', '✅ WebSocket 관리자 초기화 완료');
    }
  }, [config, kwargs]);

  // =================================================================
  // 🔧 8단계 파이프라인 WebSocket 메시지 처리
  // =================================================================

  const handleWebSocketMessage = useCallback((data: PipelineProgress) => {
    PipelineUtils.log('info', '📨 8단계 파이프라인 메시지 수신', data.type);

    switch (data.type) {
      case 'connection_established':
        updateState({
          wsConnected: true,
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

      case 'step_start':
        if (data.step_id) {
          updateState({
            currentPipelineStep: data.step_id,
            pipelineSteps: state.pipelineSteps.map(step => 
              step.id === data.step_id 
                ? { ...step, status: 'processing', progress: 0 }
                : step
            ),
            progressMessage: data.message || `${data.step_name} 처리 시작`
          });
        }
        break;

      case 'step_progress':
        if (data.step_id) {
          updateState({
            stepProgress: {
              ...state.stepProgress,
              [data.step_id]: data.progress
            },
            pipelineSteps: state.pipelineSteps.map(step => 
              step.id === data.step_id 
                ? { ...step, progress: data.progress }
                : step
            )
          });
        }
        break;

      case 'step_complete':
        if (data.step_id) {
          const stepResult = data.result || { success: true, step_id: data.step_id };
          updateState({
            stepResults: {
              ...state.stepResults,
              [data.step_id]: stepResult
            },
            pipelineSteps: state.pipelineSteps.map(step => 
              step.id === data.step_id 
                ? { ...step, status: 'completed', progress: 100 }
                : step
            ),
            stepTimings: {
              ...state.stepTimings,
              [data.step_id]: data.processing_time || 0
            }
          });
        }
        break;

      case 'step_error':
        if (data.step_id) {
          updateState({
            pipelineSteps: state.pipelineSteps.map(step => 
              step.id === data.step_id 
                ? { ...step, status: 'failed', progress: 0 }
                : step
            ),
            error: data.message || `단계 ${data.step_id} 처리 실패`
          });
        }
        break;

      case 'pipeline_completed':
        updateState({
          isProcessing: false,
          progress: 100,
          progressMessage: '8단계 파이프라인 완료!',
          sessionActive: false
        });
        break;

      case 'pipeline_error':
        updateState({
          isProcessing: false,
          error: PipelineUtils.getUserFriendlyError(data.message),
          progress: 0,
          progressMessage: '',
          sessionActive: false
        });
        break;

      default:
        PipelineUtils.log('warn', '알 수 없는 8단계 파이프라인 메시지 타입', data.type);
    }
  }, [updateState, state]);

  // =================================================================
  // 🔧 8단계 파이프라인 메인 처리 함수
  // =================================================================

  const processVirtualTryOn = useCallback(async (request: VirtualTryOnRequest) => {
    const timer = PipelineUtils.createPerformanceTimer('8단계 가상 피팅 전체 처리');
    
    try {
      // 서비스 초기화
      initializeAPIClient();
      
      // 새 세션 시작
      const sessionId = startNewSession();
      
      // WebSocket 연결 확인
      if (!wsManager.current?.isConnected()) {
        PipelineUtils.log('info', '🔄 WebSocket 재연결 시도...');
        await connectWebSocket();
      }

      // 처리 시작 상태 설정
      updateState({
        isProcessing: true,
        progress: 0,
        progressMessage: '8단계 AI 파이프라인을 시작합니다...',
        result: null,
        error: null,
        sessionId,
        sessionActive: true,
        currentPipelineStep: 1,
        startTime: Date.now(),
        pipelineSteps: PIPELINE_STEPS.map(step => ({
          id: step.id,
          name: step.name,
          status: 'pending',
          progress: 0,
          korean: step.korean,
          description: step.description
        })),
        stepResults: {},
        stepProgress: {},
        stepTimings: {}
      });

      PipelineUtils.log('info', '🎯 8단계 가상 피팅 처리 시작', { sessionId });

      // 세션 구독
      wsManager.current?.subscribeToSession(sessionId);

      // API 처리 (백엔드 8단계 파이프라인 호출)
      const result = await apiClient.current!.processVirtualTryOn({
        ...request,
        session_id: sessionId,
        enable_step_tracking: config.enableStepTracking,
        enable_realtime: config.enableRealTimeUpdates
      }, ...kwargs);

      const processingTime = timer.end();

      updateState({
        isProcessing: false,
        result,
        progress: 100,
        progressMessage: '8단계 파이프라인 완료!',
        sessionId: result.session_id || sessionId,
        sessionActive: false
      });

      PipelineUtils.log('info', '✅ 8단계 가상 피팅 처리 완료', { 
        processingTime: processingTime / 1000 
      });

      return result;

    } catch (error) {
      timer.end();
      const errorMessage = error instanceof Error ? error.message : '알 수 없는 오류가 발생했습니다.';
      const friendlyError = PipelineUtils.getUserFriendlyError(errorMessage);
      
      updateState({
        isProcessing: false,
        error: friendlyError,
        progress: 0,
        progressMessage: '',
        sessionActive: false
      });

      PipelineUtils.log('error', '❌ 8단계 가상 피팅 처리 실패', error);
      throw error;
    }
  }, [config, initializeAPIClient, connectWebSocket, updateState, kwargs]);

  // =================================================================
  // 🔧 개별 단계 처리 함수들
  // =================================================================

  const processStep = useCallback(async (stepId: number, data?: any) => {
    const step = PIPELINE_STEPS.find(s => s.id === stepId);
    if (!step) {
      throw new Error(`잘못된 단계 ID: ${stepId}`);
    }

    try {
      initializeAPIClient();

      // 단계 시작 알림
      updateState({
        currentPipelineStep: stepId,
        pipelineSteps: state.pipelineSteps.map(s => 
          s.id === stepId 
            ? { ...s, status: 'processing', progress: 0 }
            : s
        ),
        progressMessage: `${step.korean} 처리 중...`
      });

      // 단계별 API 호출
      let result;
      switch (step.name) {
        case 'human_parsing':
          result = await processHumanParsing(data);
          break;
        case 'pose_estimation':
          result = await processPoseEstimation(data);
          break;
        case 'cloth_segmentation':
          result = await processClothingAnalysis(data);
          break;
        case 'geometric_matching':
          result = await processGeometricMatching(data?.personData, data?.clothingData);
          break;
        case 'cloth_warping':
          result = await processClothWarping(data);
          break;
        case 'virtual_fitting':
          result = await processVirtualFitting(data);
          break;
        case 'post_processing':
          result = await processPostProcessing(data);
          break;
        case 'quality_assessment':
          result = await processQualityAssessment(data);
          break;
        default:
          throw new Error(`지원되지 않는 단계: ${step.name}`);
      }

      // 단계 완료 업데이트
      updateState({
        stepResults: {
          ...state.stepResults,
          [stepId]: result
        },
        pipelineSteps: state.pipelineSteps.map(s => 
          s.id === stepId 
            ? { ...s, status: 'completed', progress: 100 }
            : s
        ),
        progressMessage: `${step.korean} 완료`
      });

      return result;

    } catch (error) {
      updateState({
        pipelineSteps: state.pipelineSteps.map(s => 
          s.id === stepId 
            ? { ...s, status: 'failed', progress: 0 }
            : s
        ),
        error: `${step.korean} 처리 실패: ${error instanceof Error ? error.message : '알 수 없는 오류'}`
      });
      throw error;
    }
  }, [state.pipelineSteps, state.stepResults, updateState, initializeAPIClient]);

  // 개별 단계 처리 함수들 구현
  const processHumanParsing = useCallback(async (image: File) => {
    if (!apiClient.current) throw new Error('API 클라이언트가 초기화되지 않았습니다.');
    
    const formData = new FormData();
    formData.append('image', image);
    
    return await apiClient.current.request('/api/analyze-body', {
      method: 'POST',
      body: formData,
    });
  }, []);

  const processPoseEstimation = useCallback(async (image: File) => {
    if (!apiClient.current) throw new Error('API 클라이언트가 초기화되지 않았습니다.');
    
    const formData = new FormData();
    formData.append('image', image);
    
    return await apiClient.current.request('/api/pose-estimation', {
      method: 'POST',
      body: formData,
    });
  }, []);

  const processClothingAnalysis = useCallback(async (image: File) => {
    if (!apiClient.current) throw new Error('API 클라이언트가 초기화되지 않았습니다.');
    
    return await apiClient.current.analyzeClothing(image);
  }, []);

  const processGeometricMatching = useCallback(async (personData: any, clothingData: any) => {
    if (!apiClient.current) throw new Error('API 클라이언트가 초기화되지 않았습니다.');
    
    return await apiClient.current.request('/api/geometric-matching', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ personData, clothingData }),
    });
  }, []);

  const processClothWarping = useCallback(async (matchingData: any) => {
    if (!apiClient.current) throw new Error('API 클라이언트가 초기화되지 않았습니다.');
    
    return await apiClient.current.request('/api/cloth-warping', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ matchingData }),
    });
  }, []);

  const processVirtualFitting = useCallback(async (allData: any) => {
    if (!apiClient.current) throw new Error('API 클라이언트가 초기화되지 않았습니다.');
    
    return await apiClient.current.request('/api/virtual-fitting', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ allData }),
    });
  }, []);

  const processPostProcessing = useCallback(async (fittingResult: any) => {
    if (!apiClient.current) throw new Error('API 클라이언트가 초기화되지 않았습니다.');
    
    return await apiClient.current.request('/api/post-processing', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fittingResult }),
    });
  }, []);

  const processQualityAssessment = useCallback(async (finalResult: any) => {
    if (!apiClient.current) throw new Error('API 클라이언트가 초기화되지 않았습니다.');
    
    return await apiClient.current.request('/api/quality-assessment', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ finalResult }),
    });
  }, []);

  // =================================================================
  // 🔧 세션 및 WebSocket 관리
  // =================================================================

  const startNewSession = useCallback(() => {
    const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    updateState({
      sessionId,
      sessionActive: true,
      startTime: Date.now()
    });
    PipelineUtils.log('info', '🚀 새 8단계 파이프라인 세션 시작', { sessionId });
    return sessionId;
  }, [updateState]);

  const endSession = useCallback(() => {
    updateState({
      sessionActive: false,
      isProcessing: false,
      currentPipelineStep: 0
    });
    
    // 타이머 정리
    Object.values(stepTimeouts.current).forEach(timeout => clearTimeout(timeout));
    stepTimeouts.current = {};
    
    PipelineUtils.log('info', '🛑 8단계 파이프라인 세션 종료');
  }, [updateState]);

  const connectWebSocket = useCallback(async (): Promise<boolean> => {
    if (wsManager.current?.isConnected()) {
      return true;
    }

    updateState({
      wsConnectionAttempts: state.wsConnectionAttempts + 1
    });

    try {
      initializeWebSocketManager();

      if (!wsManager.current) {
        throw new Error('WebSocket 관리자 초기화 실패');
      }

      wsManager.current.setOnMessage(handleWebSocketMessage);
      wsManager.current.setOnConnected(() => {
        updateState({ wsConnected: true, error: null });
      });
      wsManager.current.setOnDisconnected(() => {
        updateState({ wsConnected: false });
      });
      wsManager.current.setOnError((error) => {
        updateState({
          wsConnected: false,
          error: PipelineUtils.getUserFriendlyError('WebSocket 연결 실패')
        });
      });

      const connected = await wsManager.current.connect();
      
      if (connected && state.sessionId) {
        wsManager.current.subscribeToSession(state.sessionId);
      }

      return connected;
    } catch (error) {
      PipelineUtils.log('error', '❌ WebSocket 연결 실패', error);
      updateState({
        wsConnected: false,
        error: PipelineUtils.getUserFriendlyError('WebSocket 연결 실패')
      });
      return false;
    }
  }, [initializeWebSocketManager, handleWebSocketMessage, state.wsConnectionAttempts, state.sessionId, updateState]);

  const disconnectWebSocket = useCallback(() => {
    wsManager.current?.disconnect();
    updateState({ wsConnected: false });
  }, [updateState]);

  // =================================================================
  // 🔧 파이프라인 제어 함수들
  // =================================================================

  const skipToStep = useCallback((stepId: number) => {
    if (stepId < 1 || stepId > 8) {
      throw new Error('단계 ID는 1-8 사이여야 합니다.');
    }
    
    updateState({
      currentPipelineStep: stepId,
      pipelineSteps: state.pipelineSteps.map(step => ({
        ...step,
        status: step.id < stepId ? 'skipped' : step.id === stepId ? 'processing' : 'pending'
      }))
    });
  }, [state.pipelineSteps, updateState]);

  const restartPipeline = useCallback(() => {
    endSession();
    
    // 모든 상태 초기화
    updateState({
      currentPipelineStep: 0,
      pipelineSteps: PIPELINE_STEPS.map(step => ({
        id: step.id,
        name: step.name,
        status: 'pending',
        progress: 0,
        korean: step.korean,
        description: step.description
      })),
      stepResults: {},
      stepProgress: {},
      stepTimings: {},
      progress: 0,
      progressMessage: '',
      result: null,
      error: null,
      startTime: null
    });
    
    PipelineUtils.log('info', '🔄 8단계 파이프라인 재시작');
  }, [endSession, updateState]);

  // =================================================================
  // 🔧 기존 Hook API 호환성 메서드들
  // =================================================================

  const clearResult = useCallback(() => {
    updateState({
      result: null,
      progress: 0,
      progressMessage: '',
      pipelineSteps: PIPELINE_STEPS.map(step => ({
        id: step.id,
        name: step.name,
        status: 'pending',
        progress: 0,
        korean: step.korean,
        description: step.description
      })),
      stepResults: {},
      stepProgress: {}
    });
  }, [updateState]);

  const clearError = useCallback(() => {
    updateState({ error: null });
  }, [updateState]);

  const reset = useCallback(() => {
    restartPipeline();
  }, [restartPipeline]);

  // =================================================================
  // 🔧 정보 조회 액션들
  // =================================================================

  const checkHealth = useCallback(async (): Promise<boolean> => {
    try {
      initializeAPIClient();
      const isHealthy = await apiClient.current!.healthCheck();
      updateState({ isHealthy });
      return isHealthy;
    } catch (error) {
      PipelineUtils.log('error', '헬스체크 실패', error);
      updateState({ isHealthy: false });
      return false;
    }
  }, [updateState, initializeAPIClient]);

  const getPipelineStatus = useCallback(async () => {
    try {
      initializeAPIClient();
      const pipelineStatus = await apiClient.current!.getPipelineStatus();
      updateState({ pipelineStatus });
    } catch (error) {
      PipelineUtils.log('error', '파이프라인 상태 조회 실패', error);
    }
  }, [updateState, initializeAPIClient]);

  const getSystemStats = useCallback(async () => {
    try {
      initializeAPIClient();
      const systemStats = await apiClient.current!.getSystemStats();
      updateState({ systemStats });
    } catch (error) {
      PipelineUtils.log('error', '시스템 통계 조회 실패', error);
    }
  }, [updateState, initializeAPIClient]);

  // =================================================================
  // 🔧 파이프라인 관리 액션들
  // =================================================================

  const warmupPipeline = useCallback(async (qualityMode: string = 'balanced') => {
    try {
      updateState({
        isProcessing: true,
        progressMessage: '8단계 파이프라인 워밍업 중...'
      });

      initializeAPIClient();
      const success = await apiClient.current!.warmupPipeline(qualityMode);
      
      updateState({
        isProcessing: false,
        progressMessage: success ? '워밍업 완료' : '워밍업 실패'
      });

      PipelineUtils.log(
        success ? 'info' : 'error', 
        success ? '✅ 8단계 파이프라인 워밍업 완료' : '❌ 8단계 파이프라인 워밍업 실패'
      );

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '워밍업 실패';
      updateState({
        isProcessing: false,
        error: errorMessage,
        progressMessage: ''
      });

      PipelineUtils.log('error', '❌ 8단계 파이프라인 워밍업 실패', error);
    }
  }, [updateState, initializeAPIClient]);

  const testConnection = useCallback(async () => {
    try {
      updateState({
        isProcessing: true,
        progressMessage: '8단계 파이프라인 연결 테스트 중...'
      });

      const wsConnected = await connectWebSocket();
      const healthOk = await checkHealth();

      if (!wsConnected) {
        throw new Error('WebSocket 연결 실패');
      }

      if (!healthOk) {
        throw new Error('API 헬스체크 실패');
      }

      updateState({
        isProcessing: false,
        progressMessage: '8단계 파이프라인 연결 테스트 완료',
        error: null
      });

      PipelineUtils.log('info', '✅ 8단계 파이프라인 연결 테스트 완료');

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '연결 테스트 실패';
      updateState({
        isProcessing: false,
        error: errorMessage,
        progressMessage: ''
      });

      PipelineUtils.log('error', '❌ 8단계 파이프라인 연결 테스트 실패', error);
    }
  }, [connectWebSocket, checkHealth, updateState]);

  // =================================================================
  // 🔧 유틸리티 액션들
  // =================================================================

  const sendHeartbeat = useCallback(() => {
    wsManager.current?.send({ 
      type: 'ping', 
      timestamp: Date.now(),
      session_id: state.sessionId 
    });
  }, [state.sessionId]);

  const exportLogs = useCallback(() => {
    const logs = {
      state,
      pipelineSteps: state.pipelineSteps,
      stepResults: state.stepResults,
      stepTimings: state.stepTimings,
      connectionAttempts: wsManager.current?.getReconnectAttempts() || 0,
      wsInfo: wsManager.current?.getWebSocketInfo(),
      clientInfo: apiClient.current?.getClientInfo(),
      config,
      timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(logs, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `8step_pipeline_logs_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [state, config]);

  // =================================================================
  // 🔧 생명주기 관리
  // =================================================================

  const initializeHook = useCallback(async () => {
    if (isInitialized.current) return;
    
    PipelineUtils.log('info', '🔄 8단계 파이프라인 Hook 초기화 중...');
    
    try {
      // 브라우저 호환성 확인
      const compatibility = PipelineUtils.checkBrowserCompatibility();
      if (!compatibility.overall) {
        PipelineUtils.log('error', '❌ 브라우저 호환성 부족', compatibility);
        updateState({
          error: '브라우저가 필요한 기능을 지원하지 않습니다.'
        });
        return;
      }
      
      // 서비스들 초기화
      initializeAPIClient();
      initializeWebSocketManager();
      
      // 자동 연결
      if (config.autoReconnect) {
        await connectWebSocket();
      }
      
      isInitialized.current = true;
      PipelineUtils.log('info', '✅ 8단계 파이프라인 Hook 초기화 완료');
      
    } catch (error) {
      PipelineUtils.log('error', '❌ 8단계 파이프라인 Hook 초기화 실패', error);
      updateState({
        error: PipelineUtils.getUserFriendlyError('8단계 파이프라인 초기화 실패')
      });
    }
  }, [config.autoReconnect, connectWebSocket, initializeAPIClient, initializeWebSocketManager, updateState]);

  const cleanupHook = useCallback(async () => {
    PipelineUtils.log('info', '🧹 8단계 파이프라인 Hook: 리소스 정리 중...');
    
    try {
      // 활성 세션 종료
      if (state.sessionActive) {
        endSession();
      }
      
      // 타이머들 정리
      if (healthCheckTimer.current) {
        clearInterval(healthCheckTimer.current);
        healthCheckTimer.current = null;
      }
      
      Object.values(stepTimeouts.current).forEach(timeout => clearTimeout(timeout));
      stepTimeouts.current = {};
      
      // 서비스들 정리
      await wsManager.current?.cleanup();
      await apiClient.current?.cleanup();
      
      // 연결 해제
      disconnectWebSocket();
      
      // 상태 초기화
      isInitialized.current = false;
      
      PipelineUtils.log('info', '✅ 8단계 파이프라인 Hook 리소스 정리 완료');
    } catch (error) {
      PipelineUtils.log('warn', '⚠️ 8단계 파이프라인 Hook 리소스 정리 중 오류', error);
    }
  }, [state.sessionActive, endSession, disconnectWebSocket]);

  // =================================================================
  // 🔧 Effect들
  // =================================================================

  // 초기화 Effect
  useEffect(() => {
    initializeHook();
    
    return () => {
      cleanupHook();
    };
  }, [initializeHook, cleanupHook]);

  // 자동 헬스체크 Effect
  useEffect(() => {
    if (config.autoHealthCheck) {
      checkHealth();

      healthCheckTimer.current = setInterval(checkHealth, config.healthCheckInterval);

      return () => {
        if (healthCheckTimer.current) {
          clearInterval(healthCheckTimer.current);
          healthCheckTimer.current = null;
        }
      };
    }
  }, [config.autoHealthCheck, config.healthCheckInterval, checkHealth]);

  // 단계별 타이머 관리 Effect
  useEffect(() => {
    if (state.currentPipelineStep > 0 && config.stepTimeout) {
      const stepId = state.currentPipelineStep;
      
      stepTimeouts.current[stepId] = setTimeout(() => {
        updateState({
          pipelineSteps: state.pipelineSteps.map(step => 
            step.id === stepId 
              ? { ...step, status: 'timeout' }
              : step
          ),
          error: `단계 ${stepId} 처리 시간 초과`
        });
      }, config.stepTimeout);
      
      return () => {
        if (stepTimeouts.current[stepId]) {
          clearTimeout(stepTimeouts.current[stepId]);
          delete stepTimeouts.current[stepId];
        }
      };
    }
  }, [state.currentPipelineStep, config.stepTimeout, state.pipelineSteps, updateState]);

  // =================================================================
  // 🔧 Hook 반환값 (확장된 인터페이스)
  // =================================================================

  return {
    // 기본 상태
    ...state,
    
    // 기본 액션들
    processVirtualTryOn,
    clearResult,
    clearError,
    reset,
    checkHealth,
    getPipelineStatus,
    getSystemStats,
    warmupPipeline,
    testConnection,
    sendHeartbeat,
    exportLogs,
    
    // 8단계 파이프라인 전용 액션들
    processStep,
    skipToStep,
    restartPipeline,
    
    // 세션 관리
    startNewSession,
    endSession,
    
    // WebSocket 관리
    connectWebSocket,
    disconnectWebSocket,
    
    // 개별 단계 처리
    processHumanParsing,
    processPoseEstimation,
    processClothingAnalysis,
    processGeometricMatching,
    processClothWarping,
    processVirtualFitting,
    processPostProcessing,
    processQualityAssessment
  };
};

// =================================================================
// 🔧 편의 Hook들 (8단계 파이프라인 특화)
// =================================================================

export const usePipelineStatus = (options: UsePipelineOptions = {}) => {
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const apiClient = useMemo(() => new PipelineAPIClient(options), [options]);

  const fetchStatus = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const pipelineStatus = await apiClient.getPipelineStatus();
      setStatus(pipelineStatus);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '8단계 파이프라인 상태 조회 실패';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [apiClient]);

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

  const apiClient = useMemo(() => new PipelineAPIClient(options), [options]);

  const checkHealth = useCallback(async () => {
    setIsChecking(true);

    try {
      const healthy = await apiClient.healthCheck();
      setIsHealthy(healthy);
      setLastCheck(new Date());
    } catch (error) {
      setIsHealthy(false);
      PipelineUtils.log('error', '8단계 파이프라인 헬스체크 실패', error);
    } finally {
      setIsChecking(false);
    }
  }, [apiClient]);

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

// 8단계 파이프라인 진행 상황 추적 Hook
export const usePipelineProgress = () => {
  const [stepProgress, setStepProgress] = useState<{ [stepId: number]: number }>({});
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);
  const [failedSteps, setFailedSteps] = useState<number[]>([]);

  const updateStepProgress = useCallback((stepId: number, progress: number) => {
    setStepProgress(prev => ({ ...prev, [stepId]: progress }));
    
    if (progress === 100 && !completedSteps.includes(stepId)) {
      setCompletedSteps(prev => [...prev, stepId]);
    }
  }, [completedSteps]);

  const markStepFailed = useCallback((stepId: number) => {
    if (!failedSteps.includes(stepId)) {
      setFailedSteps(prev => [...prev, stepId]);
    }
  }, [failedSteps]);

  const resetProgress = useCallback(() => {
    setStepProgress({});
    setCurrentStep(0);
    setCompletedSteps([]);
    setFailedSteps([]);
  }, []);

  const getOverallProgress = useCallback(() => {
    const totalSteps = 8;
    const completed = completedSteps.length;
    return (completed / totalSteps) * 100;
  }, [completedSteps]);

  return {
    stepProgress,
    currentStep,
    completedSteps,
    failedSteps,
    updateStepProgress,
    markStepFailed,
    resetProgress,
    getOverallProgress,
    setCurrentStep
  };
};

// =================================================================
// 🔄 하위 호환성 지원
// =================================================================

export const createPipelineHook = (
  config: UsePipelineOptions = {}
) => {
  return () => usePipeline(config);
};

// 8단계 파이프라인 팩토리
export const create8StepPipelineHook = (
  config: UsePipelineOptions = {}
) => {
  const enhancedConfig = {
    ...config,
    enableStepTracking: true,
    enableRealTimeUpdates: true,
    stepTimeout: 60000,
    autoRetrySteps: true,
    maxStepRetries: 3,
  };
  
  return () => usePipeline(enhancedConfig);
};

export default usePipeline;