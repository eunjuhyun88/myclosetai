/**
 * MyCloset AI 8단계 파이프라인 React Hook (단계별 호출 전용 버전)
 * ✅ processVirtualTryOn 제거
 * ✅ 개별 단계별 호출만 지원
 * ✅ 백엔드 API와 완전 호환
 * ✅ conda 환경 최적화
 */

import { useState, useCallback, useRef, useEffect, useMemo } from 'react';

// =================================================================
// 🔧 기본 타입 정의
// =================================================================

export interface UsePipelineOptions {
  baseURL?: string;
  wsURL?: string;
  autoReconnect?: boolean;
  maxReconnectAttempts?: number;
  requestTimeout?: number;
  enableDebugMode?: boolean;
  enableCaching?: boolean;
  device?: string;
  quality_level?: string;
}

export interface UserMeasurements {
  height: number;
  weight: number;
  chest?: number;
  waist?: number;
  hip?: number;
  shoulder_width?: number;
}

export interface StepResult {
  success: boolean;
  message: string;
  step_name: string;
  step_id: number;
  session_id: string;
  processing_time: number;
  details?: any;
  visualization?: {
    [key: string]: string; // base64 이미지들
  };
}

export interface PipelineStep {
  id: number;
  name: string;
  korean: string;
  description: string;
  endpoint: string;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'skipped';
  progress: number;
  start_time?: string;
  end_time?: string;
  duration?: number;
  error_message?: string;
  result?: StepResult;
}

// =================================================================
// 🔧 8단계 파이프라인 정의 (백엔드와 동일)
// =================================================================

const PIPELINE_STEPS: Omit<PipelineStep, 'status' | 'progress'>[] = [
  {
    id: 1,
    name: 'human_parsing',
    korean: '인체 파싱',
    description: 'AI가 신체 부위를 20개 영역으로 분석합니다',
    endpoint: '/api/step/1/human-parsing'
  },
  {
    id: 2,
    name: 'pose_estimation',
    korean: '포즈 추정',
    description: '18개 키포인트로 자세를 분석합니다',
    endpoint: '/api/step/2/pose-estimation'
  },
  {
    id: 3,
    name: 'cloth_segmentation',
    korean: '의류 세그멘테이션',
    description: '의류 영역을 정확히 분할합니다',
    endpoint: '/api/step/3/cloth-segmentation'
  },
  {
    id: 4,
    name: 'geometric_matching',
    korean: '기하학적 매칭',
    description: '신체와 의류를 정확히 매칭합니다',
    endpoint: '/api/step/4/geometric-matching'
  },
  {
    id: 5,
    name: 'cloth_warping',
    korean: '의류 워핑',
    description: '의류를 신체에 맞게 변형합니다',
    endpoint: '/api/step/5/cloth-warping'
  },
  {
    id: 6,
    name: 'virtual_fitting',
    korean: '가상 피팅',
    description: 'AI로 가상 착용 결과를 생성합니다',
    endpoint: '/api/step/6/virtual-fitting'
  },
  {
    id: 7,
    name: 'post_processing',
    korean: '후처리',
    description: '결과 이미지를 최적화합니다',
    endpoint: '/api/step/7/post-processing'
  },
  {
    id: 8,
    name: 'quality_assessment',
    korean: '품질 평가',
    description: '최종 결과를 평가하고 저장합니다',
    endpoint: '/api/step/8/quality-assessment'
  }
];

// =================================================================
// 🔧 안전한 API 클라이언트 (단계별 호출 전용)
// =================================================================

class StepAPIClient {
  private baseURL: string;
  private abortController: AbortController | null = null;
  private requestTimeout = 60000; // 60초

  constructor(baseURL: string = 'http://localhost:8000') {
    this.baseURL = baseURL.replace(/\/$/, '');
    console.log('🔧 StepAPIClient 생성:', this.baseURL);
  }

  private async fetchWithTimeout(
    url: string, 
    options: RequestInit = {}
  ): Promise<Response> {
    // 이전 요청 취소
    if (this.abortController) {
      this.abortController.abort();
    }
    
    this.abortController = new AbortController();
    
    const timeoutId = setTimeout(() => {
      this.abortController?.abort();
    }, this.requestTimeout);

    try {
      const response = await fetch(url, {
        ...options,
        signal: this.abortController.signal,
      });

      clearTimeout(timeoutId);
      return response;
    } catch (error: any) {
      clearTimeout(timeoutId);
      if (error.name === 'AbortError') {
        throw new Error('요청 시간이 초과되었습니다.');
      }
      throw error;
    }
  }

  async callStepAPI(stepId: number, formData: FormData): Promise<StepResult> {
    const stepConfig = PIPELINE_STEPS.find(s => s.id === stepId);
    if (!stepConfig) {
      throw new Error(`존재하지 않는 단계 ID: ${stepId}`);
    }

    const url = `${this.baseURL}${stepConfig.endpoint}`;
    
    try {
      console.log(`🚀 Step ${stepId} API 호출:`, url);
      console.log(`🔍 Step ${stepId} 설정:`, stepConfig);
      console.log(`📡 Step ${stepId} URL:`, url);
      
      const response = await this.fetchWithTimeout(url, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage;
        
        try {
          const errorJson = JSON.parse(errorText);
          errorMessage = errorJson.detail || errorJson.error || errorJson.message || `HTTP ${response.status}`;
        } catch {
          errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        }
        
        throw new Error(errorMessage);
      }

      const result: StepResult = await response.json();
      console.log(`✅ Step ${stepId} 완료:`, result);
      return result;
      
    } catch (error) {
      console.error(`❌ Step ${stepId} 실패:`, error);
      throw error;
    }
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.fetchWithTimeout(`${this.baseURL}/health`);
      return response.ok;
    } catch (error) {
      console.error('❌ 헬스체크 실패:', error);
      return false;
    }
  }

  cancelCurrentRequest(): void {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
      console.log('🚫 현재 요청 취소됨');
    }
  }

  cleanup(): void {
    console.log('🧹 StepAPIClient 정리');
    this.cancelCurrentRequest();
  }
}

// =================================================================
// 🔧 메인 usePipeline Hook (단계별 호출 전용)
// =================================================================

export const usePipeline = (options: UsePipelineOptions = {}) => {
  const [mounted, setMounted] = useState(true);
  
  // 기본 상태
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isHealthy, setIsHealthy] = useState(false);

  // 단계별 상태
  const [currentStep, setCurrentStep] = useState(0);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [pipelineSteps, setPipelineSteps] = useState<PipelineStep[]>(
    PIPELINE_STEPS.map(step => ({
      ...step,
      status: 'pending',
      progress: 0
    }))
  );
  const [stepResults, setStepResults] = useState<{ [stepId: number]: StepResult }>({});
  
  // 최종 결과
  const [finalResult, setFinalResult] = useState<any>(null);

  // API 클라이언트
  const apiClient = useRef<StepAPIClient | null>(null);

  // 설정
  const config = useMemo(() => ({
    baseURL: options.baseURL || 'http://localhost:8000',
    requestTimeout: options.requestTimeout || 60000,
    enableDebugMode: options.enableDebugMode ?? false,
    ...options
  }), [options]);

  // =================================================================
  // 🔧 서비스 초기화
  // =================================================================

  const initializeAPI = useCallback(() => {
    if (!apiClient.current) {
      apiClient.current = new StepAPIClient(config.baseURL);
      console.log('✅ StepAPIClient 초기화됨');
    }
  }, [config.baseURL]);

  // =================================================================
  // 🔧 개별 단계 호출 함수들
  // =================================================================

  const callStep = useCallback(async (
    stepId: number, 
    personImage?: File, 
    clothingImage?: File, 
    measurements?: UserMeasurements
  ): Promise<StepResult> => {
    if (!mounted) throw new Error('컴포넌트가 언마운트됨');

    initializeAPI();

    // 단계 상태 업데이트 - 처리 시작
    setPipelineSteps(prev => prev.map(step => 
      step.id === stepId 
        ? { 
            ...step, 
            status: 'processing', 
            progress: 0, 
            start_time: new Date().toISOString() 
          }
        : step
    ));

    setCurrentStep(stepId);
    setProgressMessage(`${PIPELINE_STEPS.find(s => s.id === stepId)?.korean} 처리 중...`);

    try {
      // FormData 구성
      const formData = new FormData();
      
      if (sessionId) {
        formData.append('session_id', sessionId);
      }

      // Step 0,1,2: 이미지와 측정값 필요 (세션 생성 및 AI 처리 단계)
      if (stepId <= 2) {
        if (personImage) formData.append('person_image', personImage);
        if (clothingImage) formData.append('clothing_image', clothingImage);
        if (measurements) {
          formData.append('height', measurements.height.toString());
          formData.append('weight', measurements.weight.toString());
          if (measurements.chest) formData.append('chest', measurements.chest.toString());
          if (measurements.waist) formData.append('waist', measurements.waist.toString());
          if (measurements.hip) formData.append('hip', measurements.hip.toString());
          if (measurements.shoulder_width) formData.append('shoulder_width', measurements.shoulder_width.toString());
        }
      }

      // API 호출
      const result = await apiClient.current!.callStepAPI(stepId, formData);

      // 세션 ID 업데이트 (Step 0,1,2에서 세션 생성)
      if (result.session_id && (stepId <= 2 || !sessionId)) {
        setSessionId(result.session_id);
      }

      // 단계 상태 업데이트 - 완료
      setPipelineSteps(prev => prev.map(step => 
        step.id === stepId 
          ? { 
              ...step, 
              status: 'completed', 
              progress: 100,
              end_time: new Date().toISOString(),
              duration: result.processing_time || 0,
              result: result
            }
          : step
      ));

      // 결과 저장
      setStepResults(prev => ({ ...prev, [stepId]: result }));

      // Step 9 완료 시 최종 결과 설정
      if (stepId === 9) {
        setFinalResult(result);
      }

      return result;

    } catch (error: any) {
      // 단계 상태 업데이트 - 실패
      setPipelineSteps(prev => prev.map(step => 
        step.id === stepId 
          ? { 
              ...step, 
              status: 'failed', 
              progress: 0,
              error_message: error.message,
              end_time: new Date().toISOString()
            }
          : step
      ));

      throw error;
    }
  }, [mounted, sessionId, initializeAPI]);

 
  // =================================================================
  // 🔧 개별 단계 실행 함수들 (App.tsx 호환용)
  // =================================================================

  const processStep1 = useCallback(async (personImage: File, clothingImage: File, measurements: UserMeasurements) => {
    return await callStep(1, personImage, clothingImage, measurements);
  }, [callStep]);

  const processStep2 = useCallback(async () => {
    return await callStep(2);
  }, [callStep]);

  const processStep3 = useCallback(async () => {
    return await callStep(3);
  }, [callStep]);

  const processStep4 = useCallback(async () => {
    return await callStep(4);
  }, [callStep]);

  const processStep5 = useCallback(async () => {
    return await callStep(5);
  }, [callStep]);

  const processStep6 = useCallback(async () => {
    return await callStep(6);
  }, [callStep]);

  const processStep7 = useCallback(async () => {
    return await callStep(7);
  }, [callStep]);

  const processStep8 = useCallback(async () => {
    return await callStep(8);
  }, [callStep]);

  // =================================================================
  // 🔧 유틸리티 함수들
  // =================================================================

  const reset = useCallback(() => {
    if (!mounted) return;

    apiClient.current?.cancelCurrentRequest();

    setIsProcessing(false);
    setProgress(0);
    setProgressMessage('');
    setError(null);
    setCurrentStep(0);
    setSessionId(null);
    setFinalResult(null);
    setPipelineSteps(PIPELINE_STEPS.map(step => ({
      ...step,
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

  const checkHealth = useCallback(async (): Promise<boolean> => {
    if (!mounted) return false;

    try {
      initializeAPI();
      const healthy = await apiClient.current!.healthCheck();
      setIsHealthy(healthy);
      return healthy;
    } catch (error) {
      console.error('❌ 헬스체크 실패:', error);
      setIsHealthy(false);
      return false;
    }
  }, [mounted, initializeAPI]);

  const getStepVisualization = useCallback((stepId: number): { [key: string]: string } => {
    const stepResult = stepResults[stepId];
    return stepResult?.visualization || {};
  }, [stepResults]);

  const getStepStatus = useCallback((stepId: number): PipelineStep['status'] => {
    const step = pipelineSteps.find(s => s.id === stepId);
    return step?.status || 'pending';
  }, [pipelineSteps]);

  // =================================================================
  // 🔧 생명주기 관리
  // =================================================================

  useEffect(() => {
    setMounted(true);
    console.log('🔧 usePipeline Hook (단계별) 마운트됨');

    return () => {
      setMounted(false);
      console.log('🔧 usePipeline Hook (단계별) 언마운트됨');
    };
  }, []);

  useEffect(() => {
    return () => {
      console.log('🧹 usePipeline 정리 시작');
      if (apiClient.current) {
        apiClient.current.cleanup();
        apiClient.current = null;
      }
      console.log('✅ usePipeline 정리 완료');
    };
  }, []);

  // 자동 헬스체크
  useEffect(() => {
    if (!mounted) return;

    checkHealth();

    const intervalId = setInterval(() => {
      if (mounted) {
        checkHealth();
      }
    }, 30000); // 30초마다

    return () => clearInterval(intervalId);
  }, [mounted, checkHealth]);

  // =================================================================
  // 🔧 Hook 반환값 (App.tsx 완전 호환)
  // =================================================================

  return {
    // 기본 상태
    isProcessing,
    progress,
    progressMessage,
    error,
    isHealthy,

    // 단계별 상태
    currentStep,
    sessionId,
    pipelineSteps,
    stepResults,
    finalResult,

    // 전체 파이프라인 실행

    // 개별 단계 실행 (App.tsx 호환)
    processStep1,
    processStep2, 
    processStep3,
    processStep4,
    processStep5,
    processStep6,
    processStep7,
    processStep8,

    // 범용 단계 호출
    callStep,

    // 유틸리티
    reset,
    clearError,
    checkHealth,
    getStepVisualization,
    getStepStatus,

    // 디버그/모니터링
    exportLogs: () => ({
      isProcessing,
      progress,
      currentStep,
      sessionId,
      pipelineSteps,
      stepResults,
      finalResult,
      timestamp: new Date().toISOString()
    }),

    // 취소
    cancelCurrentRequest: () => apiClient.current?.cancelCurrentRequest()
  };
};

export default usePipeline;