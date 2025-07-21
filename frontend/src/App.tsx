import React, { useState, useRef, useEffect, useCallback } from 'react';

// ===============================================================
// 🔧 백엔드 완전 호환 타입 정의들
// ===============================================================

interface UserMeasurements {
  height: number;
  weight: number;
}

interface StepResult {
  success: boolean;
  message: string;
  processing_time: number;
  confidence: number;
  error?: string;
  details?: {
    session_id?: string;
    result_image?: string;
    visualization?: string;
    overlay_image?: string;
    detected_parts?: number;
    total_parts?: number;
    detected_keypoints?: number;
    total_keypoints?: number;
    category?: string;
    style?: string;
    clothing_info?: {
      category: string;
      style: string;
      colors: string[];
    };
    body_parts?: string[];
    pose_confidence?: number;
    matching_score?: number;
    alignment_points?: number;
    fitting_quality?: string;
  };
  fitted_image?: string;
  fit_score?: number;
  recommendations?: string[];
}

interface TryOnResult {
  success: boolean;
  message: string;
  processing_time: number;
  confidence: number;
  session_id: string;
  fitted_image?: string;
  fit_score: number;
  measurements: {
    chest: number;
    waist: number;
    hip: number;
    bmi: number;
  };
  clothing_analysis: {
    category: string;
    style: string;
    dominant_color: number[];
    color_name?: string;
    material?: string;
    pattern?: string;
  };
  recommendations: string[];
}

interface SystemInfo {
  app_name: string;
  app_version: string;
  device: string;
  device_name: string;
  is_m3_max: boolean;
  total_memory_gb: number;
  available_memory_gb: number;
  timestamp: number;
}

interface PipelineStep {
  id: number;
  name: string;
  description: string;
  endpoint: string;
  processing_time: number;
}

// 🔥 백엔드와 완전 동일한 8단계 정의 (URL 수정됨)
const PIPELINE_STEPS: PipelineStep[] = [
  {
    id: 1,
    name: "이미지 업로드 검증",
    description: "사용자 사진과 의류 이미지를 검증합니다",
    endpoint: "/api/step/1/upload-validation", // 🔥 수정됨
    processing_time: 0.5
  },
  {
    id: 2,
    name: "신체 측정값 검증",
    description: "키와 몸무게 등 신체 정보를 검증합니다",
    endpoint: "/api/step/2/measurements-validation", // 🔥 수정됨
    processing_time: 0.3
  },
  {
    id: 3,
    name: "인체 파싱",
    description: "AI가 신체 부위를 20개 영역으로 분석합니다",
    endpoint: "/api/step/3/human-parsing", // 🔥 수정됨
    processing_time: 1.2
  },
  {
    id: 4,
    name: "포즈 추정",
    description: "18개 키포인트로 자세를 분석합니다",
    endpoint: "/api/step/4/pose-estimation", // 🔥 수정됨
    processing_time: 0.8
  },
  {
    id: 5,
    name: "의류 분석",
    description: "의류 스타일과 색상을 분석합니다",
    endpoint: "/api/step/5/clothing-analysis", // 🔥 수정됨
    processing_time: 0.6
  },
  {
    id: 6,
    name: "기하학적 매칭",
    description: "신체와 의류를 정확히 매칭합니다",
    endpoint: "/api/step/6/geometric-matching", // 🔥 수정됨
    processing_time: 1.5
  },
  {
    id: 7,
    name: "가상 피팅",
    description: "AI로 가상 착용 결과를 생성합니다",
    endpoint: "/api/step/7/virtual-fitting", // 🔥 수정됨
    processing_time: 2.5
  },
  {
    id: 8,
    name: "결과 분석",
    description: "최종 결과를 확인하고 저장합니다",
    endpoint: "/api/step/8/result-analysis", // 🔥 수정됨
    processing_time: 0.3
  }
];

// ===============================================================
// 🔧 백엔드 완전 호환 API 클라이언트 (수정됨)
// ===============================================================

class APIClient {
  private baseURL: string;
  private currentSessionId: string | null = null;
  private websocket: WebSocket | null = null;
  private progressCallback: ((step: number, progress: number, message: string) => void) | null = null;

  constructor(baseURL: string = 'http://localhost:8000') { // 🔥 포트 8000으로 통일
    this.baseURL = baseURL;
  }

  setSessionId(sessionId: string) {
    this.currentSessionId = sessionId;
  }

  getSessionId(): string | null {
    return this.currentSessionId;
  }

  setProgressCallback(callback: (step: number, progress: number, message: string) => void) {
    this.progressCallback = callback;
  }

  // WebSocket 연결 (백엔드 완전 호환 - 수정됨)
  connectWebSocket(sessionId: string): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        const wsURL = `ws://localhost:8000/api/ws/ai-pipeline`; // 🔥 수정됨
        this.websocket = new WebSocket(wsURL);

        this.websocket.onopen = () => {
          console.log('🔗 WebSocket 연결됨');
          if (this.websocket) {
            this.websocket.send(JSON.stringify({ 
              type: 'subscribe', 
              session_id: sessionId 
            }));
          }
          resolve();
        };

        this.websocket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            
            if (data.type === 'ai_progress' && this.progressCallback) {
              this.progressCallback(data.step || 0, data.progress || 0, data.message || '');
            }
            
            console.log('📡 WebSocket 메시지:', data);
          } catch (error) {
            console.error('WebSocket 메시지 파싱 오류:', error);
          }
        };

        this.websocket.onerror = (error) => {
          console.error('WebSocket 오류:', error);
          reject(error);
        };

        this.websocket.onclose = () => {
          console.log('🔌 WebSocket 연결 해제됨');
          this.websocket = null;
        };

        setTimeout(() => {
          if (this.websocket?.readyState !== WebSocket.OPEN) {
            reject(new Error('WebSocket 연결 타임아웃'));
          }
        }, 5000);

      } catch (error) {
        reject(error);
      }
    });
  }

  disconnectWebSocket() {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
  }

  // 헬스체크 (백엔드 완전 호환)
  async healthCheck(): Promise<{ success: boolean; data?: any; error?: string }> {
    try {
      const response = await fetch(`${this.baseURL}/health`);
      const data = await response.json();
      return { success: response.ok, data };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Network error' 
      };
    }
  }

  // 시스템 정보 조회 (백엔드 완전 호환)
  async getSystemInfo(): Promise<SystemInfo> {
    const response = await fetch(`${this.baseURL}/api/system/info`);
    if (!response.ok) {
      throw new Error(`시스템 정보 조회 실패: ${response.status}`);
    }
    return await response.json();
  }

  // 🔥 개별 단계 API 호출 (완전 수정됨)
  async callStepAPI(stepId: number, formData: FormData): Promise<StepResult> {
    const step = PIPELINE_STEPS.find(s => s.id === stepId);
    if (!step) {
      throw new Error(`Invalid step ID: ${stepId}`);
    }

    // 세션 ID가 있으면 FormData에 추가
    if (this.currentSessionId) {
      formData.append('session_id', this.currentSessionId);
    }

    try {
      console.log(`🚀 Step ${stepId} API 호출: ${step.endpoint}`);
      
      const response = await fetch(`${this.baseURL}${step.endpoint}`, {
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
      
    // 세션 ID 업데이트 (1단계에서 반환됨)
    if (stepId === 1 && result.session_id) {  // ✅ 수정됨
      this.setSessionId(result.session_id);
    }

      console.log(`✅ Step ${stepId} 완료:`, result);
      return result;
      
    } catch (error) {
      console.error(`❌ Step ${stepId} 실패:`, error);
      throw error;
    }
  }

  // 전체 파이프라인 실행 (백엔드 완전 호환 - 수정됨)
  async runCompletePipeline(
    personImage: File, 
    clothingImage: File, 
    measurements: UserMeasurements
  ): Promise<TryOnResult> {
    const formData = new FormData();
    formData.append('person_image', personImage);
    formData.append('clothing_image', clothingImage);
    formData.append('height', measurements.height.toString());
    formData.append('weight', measurements.weight.toString());
    
    if (this.currentSessionId) {
      formData.append('session_id', this.currentSessionId);
    }

    try {
      console.log('🚀 전체 파이프라인 실행 시작');
      
      const response = await fetch(`${this.baseURL}/api/step/complete`, { // 🔥 수정됨
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Pipeline failed: ${response.status} - ${errorText}`);
      }

      const result: TryOnResult = await response.json();
      
      if (result.session_id) {
        this.setSessionId(result.session_id);
      }

      console.log('✅ 전체 파이프라인 완료:', result);
      return result;
      
    } catch (error) {
      console.error('❌ 전체 파이프라인 실패:', error);
      throw error;
    }
  }
}

// ===============================================================
// 🔧 유틸리티 함수들
// ===============================================================

const fileUtils = {
  validateImageFile: (file: File) => {
    const maxSize = 50 * 1024 * 1024; // 50MB
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    
    if (file.size > maxSize) {
      return { valid: false, error: '파일 크기가 50MB를 초과합니다.' };
    }
    
    if (!allowedTypes.includes(file.type)) {
      return { valid: false, error: '지원되지 않는 파일 형식입니다.' };
    }
    
    return { valid: true };
  },
  
  formatFileSize: (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  },

  createImagePreview: (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target?.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }
};

// ===============================================================
// 🔧 메인 App 컴포넌트
// ===============================================================

const App: React.FC = () => {
  // API 클라이언트
  const [apiClient] = useState(() => new APIClient());

  // 현재 단계 관리
  const [currentStep, setCurrentStep] = useState(1);
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);
  
  // 파일 상태
  const [personImage, setPersonImage] = useState<File | null>(null);
  const [clothingImage, setClothingImage] = useState<File | null>(null);
  const [personImagePreview, setPersonImagePreview] = useState<string | null>(null);
  const [clothingImagePreview, setClothingImagePreview] = useState<string | null>(null);
  
  // 측정값
  const [measurements, setMeasurements] = useState<UserMeasurements>({
    height: 170,
    weight: 65
  });

  // 단계별 결과 저장
  const [stepResults, setStepResults] = useState<{[key: number]: StepResult}>({});
  
  // 최종 결과
  const [result, setResult] = useState<TryOnResult | null>(null);
  
  // 파일 검증 에러
  const [fileErrors, setFileErrors] = useState<{
    person?: string;
    clothing?: string;
  }>({});

  // 처리 상태
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const [error, setError] = useState<string | null>(null);

  // 서버 상태
  const [isServerHealthy, setIsServerHealthy] = useState(true);
  const [isCheckingHealth, setIsCheckingHealth] = useState(false);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);

  // 반응형 상태
  const [isMobile, setIsMobile] = useState(false);

  // 파일 참조
  const personImageRef = useRef<HTMLInputElement>(null);
  const clothingImageRef = useRef<HTMLInputElement>(null);

  // 🔥 Step 2 완료 후 자동 실행
  const [autoProcessing, setAutoProcessing] = useState(false);

  // ===============================================================
  // 🔧 이펙트들
  // ===============================================================

  // 반응형 처리
  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768);
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // 🔥 디버깅용 - 현재 설정 출력
  useEffect(() => {
    console.log('🔧 MyCloset AI 설정 (완전 수정됨):');
    console.log('  백엔드 URL:', 'http://localhost:8000');
    console.log('  WebSocket URL:', 'ws://localhost:8000/api/ws/ai-pipeline');
    console.log('  API 엔드포인트들:', PIPELINE_STEPS.map(step => step.endpoint));
  }, []);

  // 서버 헬스체크
  useEffect(() => {
    const checkHealth = async () => {
      setIsCheckingHealth(true);
      try {
        const result = await apiClient.healthCheck();
        setIsServerHealthy(result.success);
        
        if (result.success && result.data) {
          console.log('✅ 서버 상태:', result.data);
        } else {
          console.log('❌ 서버 연결 실패:', result.error);
        }
      } catch (error) {
        console.error('❌ 헬스체크 실패:', error);
        setIsServerHealthy(false);
      } finally {
        setIsCheckingHealth(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [apiClient]);

  // 🔥 Step 2 완료 후 자동으로 Step 3-8 실행
  useEffect(() => {
    if (completedSteps.includes(2) && currentStep === 2 && !isProcessing && !autoProcessing) {
      console.log('🚀 Step 2 완료됨 - Step 3-8 자동 시작!');
      autoProcessRemainingSteps();
    }
  }, [completedSteps, currentStep, isProcessing, autoProcessing]);

  // 시스템 정보 조회
  useEffect(() => {
    const fetchSystemInfo = async () => {
      try {
        const info = await apiClient.getSystemInfo();
        setSystemInfo(info);
        console.log('📊 시스템 정보:', info);
      } catch (error) {
        console.error('시스템 정보 조회 실패:', error);
      }
    };

    if (isServerHealthy) {
      fetchSystemInfo();
    }
  }, [isServerHealthy, apiClient]);

  // 진행률 콜백 설정
  useEffect(() => {
    apiClient.setProgressCallback((step, progressValue, message) => {
      setProgress(progressValue);
      setProgressMessage(message);
      console.log(`📊 Step ${step}: ${progressValue}% - ${message}`);
    });
  }, [apiClient]);

  // 컴포넌트 언마운트 시 WebSocket 정리
  useEffect(() => {
    return () => {
      apiClient.disconnectWebSocket();
    };
  }, [apiClient]);

  // ===============================================================
  // 🔧 핵심 처리 함수들
  // ===============================================================

  // 🔥 Step 3-8 자동 처리 함수 (백엔드 완전 호환)
  const autoProcessRemainingSteps = async () => {
    if (!stepResults[1]?.details?.session_id) {
      setError('세션 ID가 없습니다. Step 1부터 다시 시작해주세요.');
      return;
    }

    setAutoProcessing(true);
    setIsProcessing(true);
    const sessionId = stepResults[1].details.session_id;

    try {
      // WebSocket 연결 시도
      try {
        await apiClient.connectWebSocket(sessionId);
      } catch (error) {
        console.warn('WebSocket 연결 실패, HTTP 폴링으로 진행:', error);
      }

      // Step 3: 인체 파싱
      setCurrentStep(3);
      setProgress(20);
      setProgressMessage('Step 3: AI 인체 파싱 중...');
      
      const formData3 = new FormData();
      formData3.append('session_id', sessionId);
      
      const step3Result = await apiClient.callStepAPI(3, formData3);
      setStepResults(prev => ({ ...prev, 3: step3Result }));
      setCompletedSteps(prev => [...prev, 3]);
      
      // Step 4: 포즈 추정
      setCurrentStep(4);
      setProgress(35);
      setProgressMessage('Step 4: AI 포즈 추정 중...');
      
      const formData4 = new FormData();
      formData4.append('session_id', sessionId);
      
      const step4Result = await apiClient.callStepAPI(4, formData4);
      setStepResults(prev => ({ ...prev, 4: step4Result }));
      setCompletedSteps(prev => [...prev, 4]);
      
      // Step 5: 의류 분석
      setCurrentStep(5);
      setProgress(50);
      setProgressMessage('Step 5: AI 의류 분석 중...');
      
      const formData5 = new FormData();
      formData5.append('session_id', sessionId);
      
      const step5Result = await apiClient.callStepAPI(5, formData5);
      setStepResults(prev => ({ ...prev, 5: step5Result }));
      setCompletedSteps(prev => [...prev, 5]);
      
      // Step 6: 기하학적 매칭
      setCurrentStep(6);
      setProgress(65);
      setProgressMessage('Step 6: AI 기하학적 매칭 중...');
      
      const formData6 = new FormData();
      formData6.append('session_id', sessionId);
      
      const step6Result = await apiClient.callStepAPI(6, formData6);
      setStepResults(prev => ({ ...prev, 6: step6Result }));
      setCompletedSteps(prev => [...prev, 6]);
      
      // Step 7: 가상 피팅 (핵심!)
      setCurrentStep(7);
      setProgress(80);
      setProgressMessage('Step 7: AI 가상 피팅 생성 중...');
      
      const formData7 = new FormData();
      formData7.append('session_id', sessionId);
      
      const step7Result = await apiClient.callStepAPI(7, formData7);
      setStepResults(prev => ({ ...prev, 7: step7Result }));
      setCompletedSteps(prev => [...prev, 7]);
      
      // 🔥 가상 피팅 결과를 TryOnResult로 변환
      if (step7Result.success && step7Result.fitted_image) {
        const newResult: TryOnResult = {
          success: true,
          message: step7Result.message,
          processing_time: step7Result.processing_time,
          confidence: step7Result.confidence,
          session_id: sessionId,
          fitted_image: step7Result.fitted_image,
          fit_score: step7Result.fit_score || 0.88,
          measurements: {
            chest: measurements.height * 0.5,
            waist: measurements.height * 0.45,
            hip: measurements.height * 0.55,
            bmi: measurements.weight / ((measurements.height / 100) ** 2)
          },
          clothing_analysis: {
            category: step5Result?.details?.category || "상의",
            style: step5Result?.details?.style || "캐주얼",
            dominant_color: step5Result?.details?.clothing_info?.colors?.map(c => parseInt(c)) || [100, 150, 200],
            color_name: step5Result?.details?.clothing_info?.colors?.[0] || "블루",
            material: "코튼",
            pattern: "솔리드"
          },
          recommendations: step7Result.recommendations || [
            "색상이 잘 어울립니다",
            "사이즈가 적절합니다",
            "스타일이 매우 잘 맞습니다"
          ]
        };
        
        setResult(newResult);
      }
      
      // Step 8: 결과 분석
      setCurrentStep(8);
      setProgress(95);
      setProgressMessage('Step 8: 최종 결과 분석 중...');
      
      const formData8 = new FormData();
      formData8.append('session_id', sessionId);
      if (step7Result.fitted_image) {
        formData8.append('fitted_image_base64', step7Result.fitted_image);
      }
      formData8.append('fit_score', (step7Result.fit_score || 0.88).toString());
      
      const step8Result = await apiClient.callStepAPI(8, formData8);
      setStepResults(prev => ({ ...prev, 8: step8Result }));
      setCompletedSteps(prev => [...prev, 8]);
      
      // 최종 완료
      setProgress(100);
      setProgressMessage('🎉 모든 단계 완료!');
      
      setTimeout(() => {
        setIsProcessing(false);
        setAutoProcessing(false);
      }, 1500);
      
    } catch (error: any) {
      console.error('❌ 자동 처리 중 오류:', error);
      setError(`자동 처리 실패: ${error.message}`);
      setIsProcessing(false);
      setAutoProcessing(false);
    } finally {
      apiClient.disconnectWebSocket();
    }
  };

  // ===============================================================
  // 🔧 이벤트 핸들러들
  // ===============================================================

  // 파일 업로드 핸들러
  const handleImageUpload = useCallback(async (file: File, type: 'person' | 'clothing') => {
    const validation = fileUtils.validateImageFile(file);
    
    if (!validation.valid) {
      setFileErrors(prev => ({
        ...prev,
        [type]: validation.error
      }));
      return;
    }

    setFileErrors(prev => ({
      ...prev,
      [type]: undefined
    }));

    try {
      const preview = await fileUtils.createImagePreview(file);
      
      if (type === 'person') {
        setPersonImage(file);
        setPersonImagePreview(preview);
        console.log('✅ 사용자 이미지 업로드:', {
          name: file.name,
          size: fileUtils.formatFileSize(file.size),
          type: file.type
        });
      } else {
        setClothingImage(file);
        setClothingImagePreview(preview);
        console.log('✅ 의류 이미지 업로드:', {
          name: file.name,
          size: fileUtils.formatFileSize(file.size),
          type: file.type
        });
      }
      
      setError(null);
    } catch (error) {
      console.error('이미지 미리보기 생성 실패:', error);
      setFileErrors(prev => ({
        ...prev,
        [type]: '이미지 미리보기를 생성할 수 없습니다.'
      }));
    }
  }, []);

  // 드래그 앤 드롭
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent, type: 'person' | 'clothing') => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0 && files[0].type.startsWith('image/')) {
      handleImageUpload(files[0], type);
    }
  }, [handleImageUpload]);

  // 다음/이전 단계 이동
  const goToNextStep = useCallback(() => {
    if (currentStep < 8) {
      setCompletedSteps(prev => [...prev, currentStep]);
      setCurrentStep(prev => prev + 1);
    }
  }, [currentStep]);

  const goToPreviousStep = useCallback(() => {
    if (currentStep > 1) {
      setCurrentStep(prev => prev - 1);
      setCompletedSteps(prev => prev.filter(step => step < currentStep - 1));
    }
  }, [currentStep]);

  // 리셋
  const reset = useCallback(() => {
    setCurrentStep(1);
    setCompletedSteps([]);
    setPersonImage(null);
    setClothingImage(null);
    setPersonImagePreview(null);
    setClothingImagePreview(null);
    setStepResults({});
    setResult(null);
    setFileErrors({});
    setError(null);
    setIsProcessing(false);
    setAutoProcessing(false);
    setProgress(0);
    setProgressMessage('');
    apiClient.disconnectWebSocket();
    apiClient.setSessionId('');
  }, [apiClient]);

  const clearError = useCallback(() => setError(null), []);

  // ===============================================================
  // 🔧 단계별 처리 함수들
  // ===============================================================

  // 1단계: 이미지 업로드 검증
  const processStep1 = useCallback(async () => {
    if (!personImage || !clothingImage) {
      setError('사용자 이미지와 의류 이미지를 모두 업로드해주세요.');
      return;
    }

    setIsProcessing(true);
    setProgress(10);
    setProgressMessage('이미지 검증 중...');

    try {
      const formData = new FormData();
      formData.append('person_image', personImage);
      formData.append('clothing_image', clothingImage);
      
      setProgress(50);
      const stepResult = await apiClient.callStepAPI(1, formData);
      
      if (!stepResult.success) {
        throw new Error(stepResult.error || '1단계 검증 실패');
      }
      
      setStepResults(prev => ({ ...prev, 1: stepResult }));
      setProgress(100);
      setProgressMessage('이미지 검증 완료!');
      
      setTimeout(() => {
        setIsProcessing(false);
        goToNextStep();
      }, 1500);
      
    } catch (error: any) {
      console.error('❌ 1단계 실패:', error);
      setError(`1단계 실패: ${error.message}`);
      setIsProcessing(false);
      setProgress(0);
    }
  }, [personImage, clothingImage, apiClient, goToNextStep]);

  // 2단계: 신체 측정값 검증
 // 2단계: 신체 측정값 검증 (수정된 버전)
const processStep2 = useCallback(async () => {
  if (measurements.height <= 0 || measurements.weight <= 0) {
    setError('올바른 키와 몸무게를 입력해주세요.');
    return;
  }

  // 🔥 세션 ID 확인 (1단계에서 생성되어야 함)
  const sessionId = stepResults[1]?.details?.session_id || apiClient.getSessionId();
  
  if (!sessionId) {
    setError('세션 ID가 없습니다. 1단계부터 다시 시작해주세요.');
    return;
  }

  setIsProcessing(true);
  setProgress(10);
  setProgressMessage('신체 측정값 검증 중...');

  try {
    const formData = new FormData();
    
    // 🔥 필수 필드들 (백엔드 스키마와 일치)
    formData.append('height', measurements.height.toString());
    formData.append('weight', measurements.weight.toString());
    formData.append('session_id', sessionId); // 🔥 세션 ID 필수!
    
    // 🔥 선택적 필드들 (기본값 제공)
    formData.append('chest', '0'); // 가슴둘레 (선택적, 0으로 기본값)
    formData.append('waist', '0'); // 허리둘레 (선택적, 0으로 기본값)
    formData.append('hips', '0');  // 엉덩이둘레 (선택적, 0으로 기본값)
    
    console.log('🚀 Step 2 요청 데이터:', {
      height: measurements.height,
      weight: measurements.weight,
      session_id: sessionId,
      chest: 0,
      waist: 0,
      hips: 0
    });
    
    setProgress(50);
    const stepResult = await apiClient.callStepAPI(2, formData);
    
    if (!stepResult.success) {
      throw new Error(stepResult.error || '2단계 검증 실패');
    }
    
    setStepResults(prev => ({ ...prev, 2: stepResult }));
    setProgress(100);
    setProgressMessage('신체 측정값 검증 완료!');
    
    setTimeout(() => {
      setIsProcessing(false);
      goToNextStep();
    }, 1500);
    
  } catch (error: any) {
    console.error('❌ 2단계 실패:', error);
    
    // 🔥 구체적인 에러 메시지 제공
    let errorMessage = error.message;
    if (error.message.includes('422')) {
      errorMessage = '입력 데이터 형식이 올바르지 않습니다. 키와 몸무게를 다시 확인해주세요.';
    } else if (error.message.includes('404')) {
      errorMessage = '세션을 찾을 수 없습니다. 1단계부터 다시 시작해주세요.';
    }
    
    setError(`2단계 실패: ${errorMessage}`);
    setIsProcessing(false);
    setProgress(0);
  }
}, [measurements, apiClient, goToNextStep, stepResults]);

  // 유효성 검사 함수들
  const canProceedToNext = useCallback(() => {
    switch (currentStep) {
      case 1:
        return personImage && clothingImage && 
               !fileErrors.person && !fileErrors.clothing;
      case 2:
        return measurements.height > 0 && measurements.weight > 0 &&
               measurements.height >= 100 && measurements.height <= 250 &&
               measurements.weight >= 30 && measurements.weight <= 300;
      case 3:
      case 4:
      case 5:
      case 6:
        return stepResults[currentStep]?.success;
      case 7:
        return result?.success;
      case 8:
        return true;
      default:
        return false;
    }
  }, [currentStep, personImage, clothingImage, fileErrors, measurements, stepResults, result]);

  // 서버 상태 관련
  const getServerStatusColor = useCallback(() => {
    if (isCheckingHealth) return '#f59e0b';
    return isServerHealthy ? '#4ade80' : '#ef4444';
  }, [isCheckingHealth, isServerHealthy]);

  const getServerStatusText = useCallback(() => {
    if (isCheckingHealth) return 'Checking...';
    return isServerHealthy ? 'Server Online' : 'Server Offline';
  }, [isCheckingHealth, isServerHealthy]);

  // 개발 도구 함수들
  const handleTestConnection = useCallback(async () => {
    try {
      const result = await apiClient.healthCheck();
      console.log('연결 테스트 결과:', result);
      alert(result.success ? '✅ 연결 성공!' : `❌ 연결 실패: ${result.error}`);
    } catch (error) {
      console.error('연결 테스트 실패:', error);
      alert(`❌ 연결 테스트 실패: ${error}`);
    }
  }, [apiClient]);

  const handleSystemInfo = useCallback(async () => {
    try {
      const info = await apiClient.getSystemInfo();
      console.log('시스템 정보:', info);
      alert(`✅ ${info.app_name} v${info.app_version}\n🎯 ${info.device_name}\n💾 ${info.available_memory_gb}GB 사용가능`);
    } catch (error) {
      console.error('시스템 정보 실패:', error);
      alert(`❌ 시스템 정보 조회 실패: ${error}`);
    }
  }, [apiClient]);

  const handleCompletePipeline = useCallback(async () => {
    if (!personImage || !clothingImage) {
      setError('이미지를 먼저 업로드해주세요.');
      return;
    }
    
    setIsProcessing(true);
    setProgress(10);
    setProgressMessage('전체 파이프라인 실행 중...');
    
    try {
      const result = await apiClient.runCompletePipeline(personImage, clothingImage, measurements);
      console.log('전체 파이프라인 결과:', result);
      setResult(result);
      setProgress(100);
      setProgressMessage('전체 파이프라인 완료!');
      
      setTimeout(() => {
        setIsProcessing(false);
        setCurrentStep(8);
        setCompletedSteps([1, 2, 3, 4, 5, 6, 7]);
      }, 1500);
      
    } catch (error: any) {
      console.error('전체 파이프라인 실패:', error);
      setError(`전체 파이프라인 실패: ${error.message}`);
      setIsProcessing(false);
      setProgress(0);
    }
  }, [personImage, clothingImage, measurements, apiClient]);

  // 요청 취소
  const handleCancelRequest = useCallback(() => {
    if (isProcessing) {
      setIsProcessing(false);
      setAutoProcessing(false);
      setProgress(0);
      setProgressMessage('');
      apiClient.disconnectWebSocket();
    }
  }, [isProcessing, apiClient]);

  // 단계별 처리 함수 매핑
  const processCurrentStep = useCallback(async () => {
    const processors = {
      1: processStep1,
      2: processStep2
    };

    const processor = processors[currentStep as keyof typeof processors];
    if (processor) {
      await processor();
    }
  }, [currentStep, processStep1, processStep2]);

  // ===============================================================
  // 🔧 렌더링 함수들 (기존과 동일)
  // ===============================================================

  const renderImageUploadStep = () => (
    <div style={{ 
      display: 'grid', 
      gridTemplateColumns: isMobile ? '1fr' : 'repeat(2, 1fr)', 
      gap: isMobile ? '1rem' : '1.5rem', 
      marginBottom: '2rem' 
    }}>
      {/* Person Upload */}
      <div style={{ 
        backgroundColor: '#ffffff', 
        borderRadius: isMobile ? '0.5rem' : '0.75rem', 
        border: fileErrors.person ? '2px solid #ef4444' : '1px solid #e5e7eb', 
        padding: isMobile ? '1rem' : '1.5rem' 
      }}>
        <h3 style={{ 
          fontSize: isMobile ? '1rem' : '1.125rem', 
          fontWeight: '500', 
          color: '#111827', 
          marginBottom: '1rem' 
        }}>Your Photo</h3>
        {personImagePreview ? (
          <div style={{ position: 'relative' }}>
            <img
              src={personImagePreview}
              alt="Person"
              style={{ 
                width: '100%', 
                height: isMobile ? '12rem' : '16rem', 
                objectFit: 'cover', 
                borderRadius: '0.5rem' 
              }}
            />
            <button
              onClick={() => personImageRef.current?.click()}
              style={{ 
                position: 'absolute', 
                top: '0.5rem', 
                right: '0.5rem', 
                backgroundColor: '#ffffff', 
                borderRadius: '50%', 
                padding: isMobile ? '0.375rem' : '0.5rem', 
                boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)', 
                border: 'none',
                cursor: 'pointer'
              }}
            >
              <svg style={{ 
                width: isMobile ? '0.875rem' : '1rem', 
                height: isMobile ? '0.875rem' : '1rem', 
                color: '#4b5563' 
              }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 002 2v12a2 2 0 002 2z" />
              </svg>
            </button>
            <div style={{ 
              position: 'absolute', 
              bottom: '0.5rem', 
              left: '0.5rem', 
              backgroundColor: 'rgba(0,0,0,0.5)', 
              color: '#ffffff', 
              fontSize: isMobile ? '0.625rem' : '0.75rem', 
              padding: '0.25rem 0.5rem', 
              borderRadius: '0.25rem',
              maxWidth: '70%',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap'
            }}>
              {personImage?.name} ({personImage && fileUtils.formatFileSize(personImage.size)})
            </div>
          </div>
        ) : (
          <div 
            onClick={() => personImageRef.current?.click()}
            onDragOver={handleDragOver}
            onDrop={(e) => handleDrop(e, 'person')}
            style={{ 
              border: '2px dashed #d1d5db', 
              borderRadius: '0.5rem', 
              padding: isMobile ? '2rem' : '3rem', 
              textAlign: 'center', 
              cursor: 'pointer'
            }}
          >
            <svg style={{ 
              margin: '0 auto', 
              height: isMobile ? '2rem' : '3rem', 
              width: isMobile ? '2rem' : '3rem', 
              color: '#9ca3af', 
              marginBottom: '1rem' 
            }} stroke="currentColor" fill="none" viewBox="0 0 48 48">
              <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <p style={{ fontSize: isMobile ? '0.75rem' : '0.875rem', color: '#4b5563', marginBottom: '0.25rem', margin: 0 }}>Upload your photo</p>
            <p style={{ fontSize: isMobile ? '0.625rem' : '0.75rem', color: '#6b7280', margin: 0 }}>PNG, JPG, WebP up to 50MB</p>
            {!isMobile && <p style={{ fontSize: '0.75rem', color: '#6b7280', margin: 0 }}>또는 드래그 앤 드롭</p>}
          </div>
        )}
        {fileErrors.person && (
          <div style={{ 
            marginTop: '0.5rem', 
            padding: '0.5rem', 
            backgroundColor: '#fef2f2', 
            border: '1px solid #fecaca', 
            borderRadius: '0.25rem', 
            fontSize: isMobile ? '0.75rem' : '0.875rem', 
            color: '#b91c1c' 
          }}>
            {fileErrors.person}
          </div>
        )}
        <input
          ref={personImageRef}
          type="file"
          accept="image/*"
          onChange={(e) => e.target.files?.[0] && handleImageUpload(e.target.files[0], 'person')}
          style={{ display: 'none' }}
        />
      </div>

      {/* Clothing Upload - 동일한 구조 */}
      <div style={{ 
        backgroundColor: '#ffffff', 
        borderRadius: isMobile ? '0.5rem' : '0.75rem', 
        border: fileErrors.clothing ? '2px solid #ef4444' : '1px solid #e5e7eb', 
        padding: isMobile ? '1rem' : '1.5rem' 
      }}>
        <h3 style={{ 
          fontSize: isMobile ? '1rem' : '1.125rem', 
          fontWeight: '500', 
          color: '#111827', 
          marginBottom: '1rem' 
        }}>Clothing Item</h3>
        {clothingImagePreview ? (
          <div style={{ position: 'relative' }}>
            <img
              src={clothingImagePreview}
              alt="Clothing"
              style={{ 
                width: '100%', 
                height: isMobile ? '12rem' : '16rem', 
                objectFit: 'cover', 
                borderRadius: '0.5rem' 
              }}
            />
            <button
              onClick={() => clothingImageRef.current?.click()}
              style={{ 
                position: 'absolute', 
                top: '0.5rem', 
                right: '0.5rem', 
                backgroundColor: '#ffffff', 
                borderRadius: '50%', 
                padding: isMobile ? '0.375rem' : '0.5rem', 
                boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)', 
                border: 'none',
                cursor: 'pointer'
              }}
            >
              <svg style={{ 
                width: isMobile ? '0.875rem' : '1rem', 
                height: isMobile ? '0.875rem' : '1rem', 
                color: '#4b5563' 
              }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 002 2v12a2 2 0 002 2z" />
              </svg>
            </button>
            <div style={{ 
              position: 'absolute', 
              bottom: '0.5rem', 
              left: '0.5rem', 
              backgroundColor: 'rgba(0,0,0,0.5)', 
              color: '#ffffff', 
              fontSize: isMobile ? '0.625rem' : '0.75rem', 
              padding: '0.25rem 0.5rem', 
              borderRadius: '0.25rem',
              maxWidth: '70%',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap'
            }}>
              {clothingImage?.name} ({clothingImage && fileUtils.formatFileSize(clothingImage.size)})
            </div>
          </div>
        ) : (
          <div 
            onClick={() => clothingImageRef.current?.click()}
            onDragOver={handleDragOver}
            onDrop={(e) => handleDrop(e, 'clothing')}
            style={{ 
              border: '2px dashed #d1d5db', 
              borderRadius: '0.5rem', 
              padding: isMobile ? '2rem' : '3rem', 
              textAlign: 'center', 
              cursor: 'pointer'
            }}
          >
            <svg style={{ 
              margin: '0 auto', 
              height: isMobile ? '2rem' : '3rem', 
              width: isMobile ? '2rem' : '3rem', 
              color: '#9ca3af', 
              marginBottom: '1rem' 
            }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zM21 5a2 2 0 00-2-2h-4a2 2 0 00-2 2v12a4 4 0 004 4h4a4 4 0 004-4V5z" />
            </svg>
            <p style={{ fontSize: isMobile ? '0.75rem' : '0.875rem', color: '#4b5563', marginBottom: '0.25rem', margin: 0 }}>Upload clothing item</p>
            <p style={{ fontSize: isMobile ? '0.625rem' : '0.75rem', color: '#6b7280', margin: 0 }}>PNG, JPG, WebP up to 50MB</p>
            {!isMobile && <p style={{ fontSize: '0.75rem', color: '#6b7280', margin: 0 }}>또는 드래그 앤 드롭</p>}
          </div>
        )}
        {fileErrors.clothing && (
          <div style={{ 
            marginTop: '0.5rem', 
            padding: '0.5rem', 
            backgroundColor: '#fef2f2', 
            border: '1px solid #fecaca', 
            borderRadius: '0.25rem', 
            fontSize: isMobile ? '0.75rem' : '0.875rem', 
            color: '#b91c1c' 
          }}>
            {fileErrors.clothing}
          </div>
        )}
        <input
          ref={clothingImageRef}
          type="file"
          accept="image/*"
          onChange={(e) => e.target.files?.[0] && handleImageUpload(e.target.files[0], 'clothing')}
          style={{ display: 'none' }}
        />
      </div>
    </div>
  );

  const renderMeasurementsStep = () => (
    <div style={{ 
      backgroundColor: '#ffffff', 
      borderRadius: isMobile ? '0.5rem' : '0.75rem', 
      border: '1px solid #e5e7eb', 
      padding: isMobile ? '1rem' : '1.5rem', 
      maxWidth: isMobile ? '100%' : '28rem',
      margin: '0 auto'
    }}>
      <h3 style={{ 
        fontSize: isMobile ? '1rem' : '1.125rem', 
        fontWeight: '500', 
        color: '#111827', 
        marginBottom: '1rem' 
      }}>Body Measurements</h3>
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: isMobile ? '1fr' : 'repeat(2, 1fr)', 
        gap: '1rem' 
      }}>
        <div>
          <label style={{ 
            display: 'block', 
            fontSize: isMobile ? '0.75rem' : '0.875rem', 
            fontWeight: '500', 
            color: '#374151', 
            marginBottom: '0.5rem' 
          }}>Height (cm)</label>
          <input
            type="number"
            value={measurements.height}
            onChange={(e) => setMeasurements(prev => ({ ...prev, height: Number(e.target.value) }))}
            style={{ 
              width: '100%', 
              padding: isMobile ? '0.75rem' : '0.5rem 0.75rem', 
              border: '1px solid #d1d5db', 
              borderRadius: '0.5rem', 
              fontSize: isMobile ? '1rem' : '0.875rem',
              outline: 'none'
            }}
            min="100"
            max="250"
            placeholder="170"
          />
          <div style={{ 
            fontSize: isMobile ? '0.625rem' : '0.75rem', 
            color: '#6b7280', 
            marginTop: '0.25rem' 
          }}>
            100-250cm 범위
          </div>
        </div>
        <div>
          <label style={{ 
            display: 'block', 
            fontSize: isMobile ? '0.75rem' : '0.875rem', 
            fontWeight: '500', 
            color: '#374151', 
            marginBottom: '0.5rem' 
          }}>Weight (kg)</label>
          <input
            type="number"
            value={measurements.weight}
            onChange={(e) => setMeasurements(prev => ({ ...prev, weight: Number(e.target.value) }))}
            style={{ 
              width: '100%', 
              padding: isMobile ? '0.75rem' : '0.5rem 0.75rem', 
              border: '1px solid #d1d5db', 
              borderRadius: '0.5rem', 
              fontSize: isMobile ? '1rem' : '0.875rem',
              outline: 'none'
            }}
            min="30"
            max="300"
            placeholder="65"
          />
          <div style={{ 
            fontSize: isMobile ? '0.625rem' : '0.75rem', 
            color: '#6b7280', 
            marginTop: '0.25rem' 
          }}>
            30-300kg 범위
          </div>
        </div>
      </div>
      
      {/* BMI 계산 표시 */}
      {measurements.height > 0 && measurements.weight > 0 && (
        <div style={{ 
          marginTop: '1rem', 
          padding: '0.75rem', 
          backgroundColor: '#f9fafb', 
          borderRadius: '0.5rem' 
        }}>
          <div style={{ 
            fontSize: isMobile ? '0.75rem' : '0.875rem', 
            color: '#4b5563' 
          }}>
            BMI: {(measurements.weight / Math.pow(measurements.height / 100, 2)).toFixed(1)}
          </div>
        </div>
      )}
    </div>
  );

  // 나머지 렌더링 함수들은 기존과 동일하므로 생략...
  // (renderProcessingStep, renderVirtualFittingStep, renderResultStep)

  const renderProcessingStep = () => {
    const stepData = PIPELINE_STEPS[currentStep - 1];
    const stepResult = stepResults[currentStep];

    return (
      <div style={{ 
        textAlign: 'center', 
        maxWidth: isMobile ? '100%' : '40rem', 
        margin: '0 auto' 
      }}>
        <div style={{ 
          backgroundColor: '#ffffff', 
          borderRadius: isMobile ? '0.5rem' : '0.75rem', 
          border: '1px solid #e5e7eb', 
          padding: isMobile ? '1.5rem' : '2rem' 
        }}>
          <div style={{ marginBottom: '1.5rem' }}>
            <div style={{ 
              width: isMobile ? '3rem' : '4rem', 
              height: isMobile ? '3rem' : '4rem', 
              margin: '0 auto', 
              backgroundColor: '#eff6ff', 
              borderRadius: '50%', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center', 
              marginBottom: '1rem' 
            }}>
              {stepResult?.success ? (
                <svg style={{ 
                  width: isMobile ? '1.5rem' : '2rem', 
                  height: isMobile ? '1.5rem' : '2rem', 
                  color: '#22c55e' 
                }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              ) : autoProcessing ? (
                <div style={{ 
                  width: isMobile ? '1.5rem' : '2rem', 
                  height: isMobile ? '1.5rem' : '2rem', 
                  border: '4px solid #3b82f6', 
                  borderTop: '4px solid transparent', 
                  borderRadius: '50%',
                  animation: 'spin 1s linear infinite'
                }}></div>
              ) : (
                <span style={{ 
                  fontSize: isMobile ? '1rem' : '1.25rem', 
                  fontWeight: '600', 
                  color: '#3b82f6' 
                }}>{currentStep}</span>
              )}
            </div>
            <h3 style={{ 
              fontSize: isMobile ? '1.125rem' : '1.25rem', 
              fontWeight: '600', 
              color: '#111827' 
            }}>{stepData.name}</h3>
            <p style={{ 
              color: '#4b5563', 
              marginTop: '0.5rem',
              fontSize: isMobile ? '0.875rem' : '1rem'
            }}>{stepData.description}</p>
          </div>

          {/* 자동 처리 중 표시 */}
          {autoProcessing && !stepResult && (
            <div style={{ 
              marginTop: '1rem', 
              padding: '1rem', 
              backgroundColor: '#fef3c7', 
              borderRadius: '0.5rem',
              border: '1px solid #f59e0b'
            }}>
              <p style={{ 
                fontSize: isMobile ? '0.75rem' : '0.875rem', 
                color: '#92400e', 
                margin: 0 
              }}>
                {progressMessage}
              </p>
              <div style={{ 
                width: '100%', 
                backgroundColor: '#f3f4f6', 
                borderRadius: '0.5rem', 
                height: '0.5rem',
                marginTop: '0.5rem'
              }}>
                <div 
                  style={{ 
                    backgroundColor: '#f59e0b', 
                    height: '0.5rem', 
                    borderRadius: '0.5rem', 
                    transition: 'width 0.3s',
                    width: `${progress}%`
                  }}
                ></div>
              </div>
            </div>
          )}

          {/* API 처리 완료 후 결과 표시 */}
          {stepResult && (
            <div style={{ 
              marginTop: '1rem', 
              padding: '1rem', 
              backgroundColor: stepResult.success ? '#f0fdf4' : '#fef2f2', 
              borderRadius: '0.5rem',
              border: stepResult.success ? '1px solid #22c55e' : '1px solid #ef4444'
            }}>
              <p style={{ 
                fontSize: isMobile ? '0.75rem' : '0.875rem', 
                color: stepResult.success ? '#15803d' : '#dc2626',
                margin: '0 0 0.5rem 0',
                fontWeight: '500'
              }}>
                {stepResult.success ? '✅ ' : '❌ '}{stepResult.message}
              </p>
              
              {stepResult.success && (
                <>
                  <p style={{ 
                    fontSize: isMobile ? '0.625rem' : '0.75rem', 
                    color: '#16a34a', 
                    margin: '0 0 0.5rem 0' 
                  }}>
                    신뢰도: {(stepResult.confidence * 100).toFixed(1)}% | 
                    처리시간: {stepResult.processing_time.toFixed(2)}초
                  </p>
                </>
              )}
              
              {stepResult.error && (
                <p style={{ 
                  fontSize: isMobile ? '0.625rem' : '0.75rem', 
                  color: '#dc2626', 
                  margin: '0.25rem 0 0 0' 
                }}>
                  오류: {stepResult.error}
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderVirtualFittingStep = () => (
    <div style={{ 
      textAlign: 'center', 
      maxWidth: isMobile ? '100%' : '28rem', 
      margin: '0 auto' 
    }}>
      <div style={{ 
        backgroundColor: '#ffffff', 
        borderRadius: isMobile ? '0.5rem' : '0.75rem', 
        border: '1px solid #e5e7eb', 
        padding: isMobile ? '1.5rem' : '2rem' 
      }}>
        <div style={{ marginBottom: '1.5rem' }}>
          <div style={{ 
            width: isMobile ? '3rem' : '4rem', 
            height: isMobile ? '3rem' : '4rem', 
            margin: '0 auto', 
            backgroundColor: '#f3e8ff', 
            borderRadius: '50%', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center', 
            marginBottom: '1rem' 
          }}>
            {result?.success ? (
              <svg style={{ 
                width: isMobile ? '1.5rem' : '2rem', 
                height: isMobile ? '1.5rem' : '2rem', 
                color: '#22c55e' 
              }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            ) : (
              <div style={{ 
                width: isMobile ? '1.5rem' : '2rem', 
                height: isMobile ? '1.5rem' : '2rem', 
                border: '4px solid #7c3aed', 
                borderTop: '4px solid transparent', 
                borderRadius: '50%',
                animation: 'spin 1s linear infinite'
              }}></div>
            )}
          </div>
          <h3 style={{ 
            fontSize: isMobile ? '1.125rem' : '1.25rem', 
            fontWeight: '600', 
            color: '#111827' 
          }}>AI 가상 피팅 생성</h3>
          <p style={{ 
            color: '#4b5563', 
            marginTop: '0.5rem',
            fontSize: isMobile ? '0.875rem' : '1rem'
          }}>딥러닝 모델이 최종 결과를 생성하고 있습니다</p>
        </div>

        {autoProcessing && (
          <div style={{ marginTop: '1rem' }}>
            <div style={{ 
              width: '100%', 
              backgroundColor: '#f3f4f6', 
              borderRadius: '0.5rem', 
              height: '0.75rem',
              marginBottom: '0.5rem'
            }}>
              <div 
                style={{ 
                  backgroundColor: '#7c3aed', 
                  height: '0.75rem', 
                  borderRadius: '0.5rem', 
                  transition: 'width 0.3s',
                  width: `${progress}%`
                }}
              ></div>
            </div>
            <p style={{ 
              fontSize: isMobile ? '0.75rem' : '0.875rem', 
              color: '#4b5563' 
            }}>{progressMessage}</p>
            
            {/* 취소 버튼 */}
            <button
              onClick={handleCancelRequest}
              style={{
                marginTop: '1rem',
                padding: isMobile ? '0.75rem 1rem' : '0.5rem 1rem',
                backgroundColor: '#ef4444',
                color: '#ffffff',
                borderRadius: '0.5rem',
                border: 'none',
                cursor: 'pointer',
                fontSize: isMobile ? '0.875rem' : '0.875rem',
                width: isMobile ? '100%' : 'auto'
              }}
            >
              취소
            </button>
          </div>
        )}

        {result && (
          <div style={{ 
            marginTop: '1rem', 
            padding: '1rem', 
            backgroundColor: '#f0fdf4', 
            borderRadius: '0.5rem' 
          }}>
            <p style={{ 
              fontSize: isMobile ? '0.75rem' : '0.875rem', 
              color: '#15803d' 
            }}>가상 피팅 완성!</p>
            <p style={{ 
              fontSize: isMobile ? '0.625rem' : '0.75rem', 
              color: '#16a34a', 
              marginTop: '0.25rem' 
            }}>
              품질 점수: {(result.fit_score * 100).toFixed(1)}% | 
              처리시간: {result.processing_time.toFixed(1)}초
            </p>
          </div>
        )}
      </div>
    </div>
  );

  const renderResultStep = () => {
    if (!result) return null;

    return (
      <div style={{ 
        maxWidth: isMobile ? '100%' : '64rem', 
        margin: '0 auto' 
      }}>
        <div style={{ 
          backgroundColor: '#ffffff', 
          borderRadius: isMobile ? '0.5rem' : '0.75rem', 
          border: '1px solid #e5e7eb', 
          padding: isMobile ? '1rem' : '1.5rem' 
        }}>
          <h3 style={{ 
            fontSize: isMobile ? '1.125rem' : '1.25rem', 
            fontWeight: '600', 
            color: '#111827', 
            marginBottom: '1.5rem', 
            textAlign: 'center' 
          }}>🎉 가상 피팅 결과</h3>
          
          <div style={{ 
            display: 'flex', 
            flexDirection: isMobile ? 'column' : 'row', 
            gap: isMobile ? '1.5rem' : '2rem' 
          }}>
            {/* Result Image */}
            <div style={{ 
              flex: isMobile ? 'none' : '1',
              display: 'flex', 
              flexDirection: 'column', 
              gap: '1rem' 
            }}>
              {result.fitted_image ? (
                <img
                  src={`data:image/jpeg;base64,${result.fitted_image}`}
                  alt="Virtual try-on result"
                  style={{ 
                    width: '100%', 
                    borderRadius: '0.5rem', 
                    boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
                    maxHeight: isMobile ? '24rem' : '32rem',
                    objectFit: 'cover',
                    border: '3px solid #22c55e'
                  }}
                />
              ) : (
                <div style={{ 
                  width: '100%', 
                  height: isMobile ? '16rem' : '20rem', 
                  backgroundColor: '#f3f4f6', 
                  borderRadius: '0.5rem', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  color: '#6b7280'
                }}>
                  결과 이미지 없음
                </div>
              )}
              
              <div style={{ 
                display: 'flex', 
                flexDirection: isMobile ? 'column' : 'row',
                gap: '0.75rem' 
              }}>
                <button 
                  onClick={() => {
                    if (result.fitted_image) {
                      const link = document.createElement('a');
                      link.href = `data:image/jpeg;base64,${result.fitted_image}`;
                      link.download = 'virtual-tryon-result.jpg';
                      link.click();
                    }
                  }}
                  disabled={!result.fitted_image}
                  style={{ 
                    flex: 1, 
                    backgroundColor: result.fitted_image ? '#f3f4f6' : '#e5e7eb', 
                    color: '#374151', 
                    padding: isMobile ? '0.75rem 1rem' : '0.5rem 1rem', 
                    borderRadius: '0.5rem', 
                    fontWeight: '500', 
                    border: 'none',
                    cursor: result.fitted_image ? 'pointer' : 'not-allowed',
                    fontSize: isMobile ? '0.875rem' : '0.875rem'
                  }}
                >
                  📥 Download
                </button>
                <button 
                  onClick={() => {
                    if (navigator.share && result.fitted_image) {
                      navigator.share({
                        title: 'MyCloset AI Virtual Try-On',
                        text: '가상 피팅 결과를 확인해보세요!',
                        url: window.location.href
                      });
                    } else {
                      alert('공유 기능은 모바일에서 지원됩니다.');
                    }
                  }}
                  style={{ 
                    flex: 1, 
                    backgroundColor: '#000000', 
                    color: '#ffffff', 
                    padding: isMobile ? '0.75rem 1rem' : '0.5rem 1rem', 
                    borderRadius: '0.5rem', 
                    fontWeight: '500', 
                    border: 'none',
                    cursor: 'pointer',
                    fontSize: isMobile ? '0.875rem' : '0.875rem'
                  }}
                >
                  📤 Share
                </button>
              </div>
            </div>

            {/* Analysis */}
            <div style={{ 
              flex: isMobile ? 'none' : '1',
              display: 'flex', 
              flexDirection: 'column', 
              gap: '1.5rem' 
            }}>
              {/* Fit Scores */}
              <div>
                <h4 style={{ 
                  fontSize: isMobile ? '0.875rem' : '0.875rem', 
                  fontWeight: '500', 
                  color: '#111827', 
                  marginBottom: '1rem' 
                }}>🎯 Fit Analysis</h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  <div>
                    <div style={{ 
                      display: 'flex', 
                      justifyContent: 'space-between', 
                      fontSize: isMobile ? '0.75rem' : '0.875rem', 
                      marginBottom: '0.25rem' 
                    }}>
                      <span style={{ color: '#4b5563' }}>Fit Score</span>
                      <span style={{ fontWeight: '500' }}>{Math.round(result.fit_score * 100)}%</span>
                    </div>
                    <div style={{ 
                      width: '100%', 
                      backgroundColor: '#e5e7eb', 
                      borderRadius: '9999px', 
                      height: '0.5rem' 
                    }}>
                      <div 
                        style={{ 
                          backgroundColor: '#22c55e', 
                          height: '0.5rem', 
                          borderRadius: '9999px', 
                          transition: 'width 0.5s',
                          width: `${result.fit_score * 100}%`
                        }}
                      ></div>
                    </div>
                  </div>
                  <div>
                    <div style={{ 
                      display: 'flex', 
                      justifyContent: 'space-between', 
                      fontSize: isMobile ? '0.75rem' : '0.875rem', 
                      marginBottom: '0.25rem' 
                    }}>
                      <span style={{ color: '#4b5563' }}>Confidence</span>
                      <span style={{ fontWeight: '500' }}>{Math.round(result.confidence * 100)}%</span>
                    </div>
                    <div style={{ 
                      width: '100%', 
                      backgroundColor: '#e5e7eb', 
                      borderRadius: '9999px', 
                      height: '0.5rem' 
                    }}>
                      <div 
                        style={{ 
                          backgroundColor: '#3b82f6', 
                          height: '0.5rem', 
                          borderRadius: '9999px', 
                          transition: 'width 0.5s',
                          width: `${result.confidence * 100}%`
                        }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Details */}
              <div>
                <h4 style={{ 
                  fontSize: isMobile ? '0.875rem' : '0.875rem', 
                  fontWeight: '500', 
                  color: '#111827', 
                  marginBottom: '1rem' 
                }}>📊 Details</h4>
                <div style={{ 
                  backgroundColor: '#f9fafb', 
                  borderRadius: '0.5rem', 
                  padding: '1rem', 
                  display: 'flex', 
                  flexDirection: 'column', 
                  gap: '0.5rem' 
                }}>
                  <div style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    fontSize: isMobile ? '0.75rem' : '0.875rem' 
                  }}>
                    <span style={{ color: '#4b5563' }}>Category</span>
                    <span style={{ fontWeight: '500', textTransform: 'capitalize' }}>
                      {result?.clothing_analysis?.category || 'Unknown'}
                    </span>
                  </div>
                  <div style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    fontSize: isMobile ? '0.75rem' : '0.875rem' 
                  }}>
                    <span style={{ color: '#4b5563' }}>Style</span>
                    <span style={{ fontWeight: '500', textTransform: 'capitalize' }}>
                      {result?.clothing_analysis?.style || 'Unknown'}
                    </span>
                  </div>
                  <div style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    fontSize: isMobile ? '0.75rem' : '0.875rem' 
                  }}>
                    <span style={{ color: '#4b5563' }}>Processing Time</span>
                    <span style={{ fontWeight: '500' }}>{result?.processing_time?.toFixed(1) || 0}s</span>
                  </div>
                  <div style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    fontSize: isMobile ? '0.75rem' : '0.875rem' 
                  }}>
                    <span style={{ color: '#4b5563' }}>BMI</span>
                    <span style={{ fontWeight: '500' }}>{result?.measurements?.bmi?.toFixed(1) || 0}</span>
                  </div>
                </div>
              </div>

              {/* AI Recommendations */}
              {result.recommendations && result.recommendations.length > 0 && (
                <div>
                  <h4 style={{ 
                    fontSize: isMobile ? '0.875rem' : '0.875rem', 
                    fontWeight: '500', 
                    color: '#111827', 
                    marginBottom: '1rem' 
                  }}>🤖 AI Recommendations</h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {result.recommendations.map((rec, index) => (
                      <div key={index} style={{ 
                        backgroundColor: '#eff6ff', 
                        border: '1px solid #bfdbfe', 
                        borderRadius: '0.5rem', 
                        padding: '0.75rem' 
                      }}>
                        <p style={{ 
                          fontSize: isMobile ? '0.75rem' : '0.875rem', 
                          color: '#1e40af', 
                          margin: 0 
                        }}>• {rec}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* 🔥 추가 액션 버튼들 */}
              <div style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '0.75rem'
              }}>
                <button
                  onClick={() => {
                    // 새로운 이미지로 다시 시도
                    reset();
                  }}
                  style={{
                    width: '100%',
                    padding: isMobile ? '0.75rem' : '0.5rem',
                    backgroundColor: '#3b82f6',
                    color: '#ffffff',
                    borderRadius: '0.5rem',
                    border: 'none',
                    cursor: 'pointer',
                    fontSize: isMobile ? '0.875rem' : '0.875rem',
                    fontWeight: '500'
                  }}
                >
                  🔄 Try Another Outfit
                </button>
                
                {systemInfo?.is_m3_max && (
                  <div style={{
                    padding: '0.75rem',
                    backgroundColor: '#f0f9ff',
                    borderRadius: '0.5rem',
                    border: '1px solid #bfdbfe'
                  }}>
                    <p style={{
                      fontSize: isMobile ? '0.75rem' : '0.875rem',
                      color: '#1e40af',
                      margin: 0,
                      textAlign: 'center'
                    }}>
                      🍎 Powered by Apple M3 Max Neural Engine
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // 단계별 컨텐츠 렌더링
  const renderStepContent = () => {
    switch (currentStep) {
      case 1:
        return renderImageUploadStep();
      case 2:
        return renderMeasurementsStep();
      case 3:
      case 4:
      case 5:
      case 6:
        return renderProcessingStep();
      case 7:
        return renderVirtualFittingStep();
      case 8:
        return renderResultStep();
      default:
        return null;
    }
  };

  // ===============================================================
  // 🔧 메인 렌더링
  // ===============================================================

  // 컴포넌트 마운트 시 개발 도구 정보 출력
  useEffect(() => {
    console.log('🛠️ MyCloset AI App 시작됨 (완전 수정 버전)');
    console.log('📋 수정 사항:');
    console.log('  - 백엔드 URL: http://localhost:8000 (포트 통일)');
    console.log('  - API 엔드포인트: /api/step/* (prefix 수정)');
    console.log('  - WebSocket URL: ws://localhost:8000/api/ws/ai-pipeline');
    console.log('  - 완전한 에러 처리 및 디버깅 강화');

    // 전역에 개발 도구 등록
    (window as any).apiClient = apiClient;
    (window as any).PIPELINE_STEPS = PIPELINE_STEPS;
  }, [apiClient]);

  return (
    <div style={{ 
      minHeight: '100vh', 
      backgroundColor: '#f9fafb', 
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", "Fira Sans", "Droid Sans", "Helvetica Neue", sans-serif' 
    }}>
      {/* Header */}
      <header style={{ 
        backgroundColor: '#ffffff', 
        borderBottom: '1px solid #e5e7eb',
        position: 'sticky',
        top: 0,
        zIndex: 50
      }}>
        <div style={{ 
          maxWidth: '80rem', 
          margin: '0 auto', 
          padding: isMobile ? '0 0.75rem' : '0 1rem' 
        }}>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center', 
            height: isMobile ? '3.5rem' : '4rem' 
          }}>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <div style={{ flexShrink: 0 }}>
                <div style={{ 
                  width: isMobile ? '1.75rem' : '2rem', 
                  height: isMobile ? '1.75rem' : '2rem', 
                  backgroundColor: '#000000', 
                  borderRadius: '0.5rem', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center' 
                }}>
                  <svg style={{ 
                    width: isMobile ? '1rem' : '1.25rem', 
                    height: isMobile ? '1rem' : '1.25rem', 
                    color: '#ffffff' 
                  }} fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2L2 7v10c0 5.55 3.84 10 9 10s9-4.45 9-10V7l-10-5z"/>
                  </svg>
                </div>
              </div>
              <div style={{ marginLeft: '0.75rem' }}>
                <h1 style={{ 
                  fontSize: isMobile ? '1.125rem' : '1.25rem', 
                  fontWeight: '600', 
                  color: '#111827', 
                  margin: 0 
                }}>MyCloset AI</h1>
                <p style={{ 
                  fontSize: isMobile ? '0.625rem' : '0.75rem', 
                  color: '#6b7280', 
                  margin: 0 
                }}>
                  {systemInfo ? 
                    `${systemInfo.device_name} ${systemInfo.is_m3_max ? '🍎' : ''}` : 
                    '완전 수정 버전 (포트 8000)'
                  }
                </p>
              </div>
            </div>
            
            {/* 서버 상태 및 개발 도구 */}
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: isMobile ? '0.5rem' : '1rem' 
            }}>
              {/* 개발 도구 버튼들 - 데스크톱에서만 표시 */}
              {!isMobile && (
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button
                    onClick={handleTestConnection}
                    disabled={isProcessing}
                    style={{
                      padding: '0.25rem 0.5rem',
                      fontSize: '0.75rem',
                      backgroundColor: '#e5e7eb',
                      color: '#374151',
                      border: 'none',
                      borderRadius: '0.25rem',
                      cursor: isProcessing ? 'not-allowed' : 'pointer',
                      opacity: isProcessing ? 0.5 : 1
                    }}
                  >
                    Test
                  </button>
                  <button
                    onClick={handleSystemInfo}
                    disabled={isProcessing}
                    style={{
                      padding: '0.25rem 0.5rem',
                      fontSize: '0.75rem',
                      backgroundColor: '#e5e7eb',
                      color: '#374151',
                      border: 'none',
                      borderRadius: '0.25rem',
                      cursor: isProcessing ? 'not-allowed' : 'pointer',
                      opacity: isProcessing ? 0.5 : 1
                    }}
                  >
                    System
                  </button>
                  <button
                    onClick={handleCompletePipeline}
                    disabled={isProcessing || !personImage || !clothingImage}
                    style={{
                      padding: '0.25rem 0.5rem',
                      fontSize: '0.75rem',
                      backgroundColor: isProcessing || !personImage || !clothingImage ? '#d1d5db' : '#3b82f6',
                      color: '#ffffff',
                      border: 'none',
                      borderRadius: '0.25rem',
                      cursor: isProcessing || !personImage || !clothingImage ? 'not-allowed' : 'pointer',
                      opacity: isProcessing || !personImage || !clothingImage ? 0.5 : 1
                    }}
                  >
                    Complete
                  </button>
                </div>
              )}

              {/* 진행률 표시 (처리 중일 때) */}
              {isProcessing && (
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '0.5rem' 
                }}>
                  <div style={{ 
                    width: isMobile ? '0.625rem' : '0.75rem', 
                    height: isMobile ? '0.625rem' : '0.75rem', 
                    border: '2px solid #3b82f6', 
                    borderTop: '2px solid transparent', 
                    borderRadius: '50%',
                    animation: 'spin 1s linear infinite'
                  }}></div>
                  <span style={{ 
                    fontSize: isMobile ? '0.75rem' : '0.875rem', 
                    color: '#4b5563' 
                  }}>
                    {progress}%
                  </span>
                </div>
              )}

              {/* 서버 상태 */}
              <div style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '0.5rem' 
              }}>
                <div style={{ 
                  height: '0.5rem', 
                  width: '0.5rem', 
                  backgroundColor: getServerStatusColor(),
                  borderRadius: '50%',
                  transition: 'background-color 0.3s'
                }}></div>
                {!isMobile && (
                  <span style={{ 
                    fontSize: '0.875rem', 
                    color: '#4b5563' 
                  }}>
                    {getServerStatusText()}
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main style={{ 
        maxWidth: '80rem', 
        margin: '0 auto', 
        padding: isMobile ? '1rem 0.75rem' : '2rem 1rem' 
      }}>
        {/* Progress Bar */}
        <div style={{ marginBottom: isMobile ? '1.5rem' : '2rem' }}>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'space-between', 
            marginBottom: '1rem',
            flexDirection: isMobile ? 'column' : 'row',
            gap: isMobile ? '0.5rem' : '0'
          }}>
            <h2 style={{ 
              fontSize: isMobile ? '1.5rem' : '1.875rem', 
              fontWeight: '700', 
              color: '#111827', 
              margin: 0 
            }}>AI Virtual Try-On</h2>
            <span style={{ 
              fontSize: isMobile ? '0.75rem' : '0.875rem', 
              color: '#4b5563' 
            }}>Step {currentStep} of 8</span>
          </div>
          
          {/* Step Progress - 모바일에서는 컴팩트하게 */}
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: isMobile ? '0.5rem' : '1rem', 
            marginBottom: '1.5rem',
            overflowX: 'auto',
            paddingBottom: isMobile ? '0.5rem' : '0'
          }}>
            {PIPELINE_STEPS.map((step, index) => (
              <div key={step.id} style={{ 
                display: 'flex', 
                alignItems: 'center',
                flexShrink: 0
              }}>
                <div 
                  style={{
                    width: isMobile ? '1.5rem' : '2rem', 
                    height: isMobile ? '1.5rem' : '2rem', 
                    borderRadius: '50%', 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center', 
                    fontSize: isMobile ? '0.75rem' : '0.875rem', 
                    fontWeight: '500',
                    backgroundColor: completedSteps.includes(step.id) 
                      ? '#22c55e' 
                      : currentStep === step.id 
                        ? '#3b82f6' 
                        : '#e5e7eb',
                    color: completedSteps.includes(step.id) || currentStep === step.id 
                      ? '#ffffff' 
                      : '#4b5563'
                  }}
                >
                  {completedSteps.includes(step.id) ? (
                    <svg style={{ 
                      width: isMobile ? '0.75rem' : '1rem', 
                      height: isMobile ? '0.75rem' : '1rem' 
                    }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    step.id
                  )}
                </div>
                {index < PIPELINE_STEPS.length - 1 && (
                  <div 
                    style={{
                      width: isMobile ? '1.5rem' : '3rem', 
                      height: '2px', 
                      marginLeft: isMobile ? '0.25rem' : '0.5rem', 
                      marginRight: isMobile ? '0.25rem' : '0.5rem',
                      backgroundColor: completedSteps.includes(step.id) ? '#22c55e' : '#e5e7eb'
                    }}
                  ></div>
                )}
              </div>
            ))}
          </div>

          {/* Current Step Info */}
          <div style={{ 
            backgroundColor: '#eff6ff', 
            border: '1px solid #bfdbfe', 
            borderRadius: '0.5rem', 
            padding: isMobile ? '0.75rem' : '1rem' 
          }}>
            <h3 style={{ 
              fontWeight: '600', 
              color: '#1e40af', 
              margin: 0,
              fontSize: isMobile ? '0.875rem' : '1rem'
            }}>{PIPELINE_STEPS[currentStep - 1]?.name}</h3>
            <p style={{ 
              color: '#1d4ed8', 
              fontSize: isMobile ? '0.75rem' : '0.875rem', 
              marginTop: '0.25rem', 
              margin: 0 
            }}>{PIPELINE_STEPS[currentStep - 1]?.description}</p>
          </div>
        </div>

        {/* Step Content */}
        <div style={{ marginBottom: isMobile ? '1.5rem' : '2rem' }}>
          {renderStepContent()}
        </div>

        {/* Navigation Buttons */}
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between',
          flexDirection: isMobile ? 'column' : 'row',
          gap: isMobile ? '1rem' : '0'
        }}>
          <button
            onClick={goToPreviousStep}
            disabled={currentStep === 1 || isProcessing || autoProcessing}
            style={{
              padding: isMobile ? '0.875rem 1.5rem' : '0.75rem 1.5rem',
              backgroundColor: '#f3f4f6',
              color: '#374151',
              borderRadius: '0.5rem',
              fontWeight: '500',
              border: 'none',
              cursor: (currentStep === 1 || isProcessing || autoProcessing) ? 'not-allowed' : 'pointer',
              opacity: (currentStep === 1 || isProcessing || autoProcessing) ? 0.5 : 1,
              transition: 'all 0.2s',
              order: isMobile ? 2 : 1,
              width: isMobile ? '100%' : 'auto'
            }}
          >
            이전 단계
          </button>

          <div style={{ 
            display: 'flex', 
            gap: '0.75rem',
            order: isMobile ? 1 : 2,
            flexDirection: isMobile ? 'column' : 'row'
          }}>
            {/* 리셋 버튼 (처리 중이 아닐 때만) */}
            {!isProcessing && !autoProcessing && (currentStep > 1 || result) && (
              <button
                onClick={reset}
                style={{
                  padding: isMobile ? '0.875rem 1.5rem' : '0.75rem 1.5rem',
                  backgroundColor: '#6b7280',
                  color: '#ffffff',
                  borderRadius: '0.5rem',
                  fontWeight: '500',
                  border: 'none',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  width: isMobile ? '100%' : 'auto'
                }}
              >
                처음부터
              </button>
            )}

            {currentStep <= 2 && (
              <button
                onClick={processCurrentStep}
                disabled={!canProceedToNext() || isProcessing || autoProcessing}
                style={{
                  padding: isMobile ? '0.875rem 1.5rem' : '0.75rem 1.5rem',
                  backgroundColor: (!canProceedToNext() || isProcessing || autoProcessing) ? '#d1d5db' : '#3b82f6',
                  color: '#ffffff',
                  borderRadius: '0.5rem',
                  fontWeight: '500',
                  border: 'none',
                  cursor: (!canProceedToNext() || isProcessing || autoProcessing) ? 'not-allowed' : 'pointer',
                  transition: 'all 0.2s',
                  width: isMobile ? '100%' : 'auto'
                }}
              >
                {currentStep === 1 ? '다음 단계' : 
                 currentStep === 2 ? '🚀 AI 처리 시작' : 
                 isProcessing || autoProcessing ? '처리 중...' : '처리 시작'}
              </button>
            )}
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div style={{ 
            marginTop: '1.5rem',
            backgroundColor: '#fef2f2', 
            border: '1px solid #fecaca', 
            borderRadius: '0.5rem', 
            padding: isMobile ? '0.875rem' : '1rem'
          }}>
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'flex-start',
              gap: '0.75rem'
            }}>
              <div style={{ display: 'flex', flex: 1 }}>
                <svg style={{ 
                  flexShrink: 0, 
                  height: '1.25rem', 
                  width: '1.25rem', 
                  color: '#f87171' 
                }} viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
                <div style={{ marginLeft: '0.75rem', flex: 1 }}>
                  <h3 style={{ 
                    fontSize: isMobile ? '0.75rem' : '0.875rem', 
                    fontWeight: '500', 
                    color: '#991b1b', 
                    margin: 0 
                  }}>Error</h3>
                  <p style={{ 
                    fontSize: isMobile ? '0.75rem' : '0.875rem', 
                    color: '#b91c1c', 
                    marginTop: '0.25rem', 
                    margin: 0,
                    wordBreak: 'break-word'
                  }}>{error}</p>
                </div>
              </div>
              <button
                onClick={clearError}
                style={{
                  backgroundColor: 'transparent',
                  border: 'none',
                  color: '#991b1b',
                  cursor: 'pointer',
                  padding: '0.25rem',
                  flexShrink: 0
                }}
              >
                ✕
              </button>
            </div>
          </div>
        )}

        {/* Instructions (첫 번째 단계에서만 표시) */}
        {currentStep === 1 && !personImage && !clothingImage && (
          <div style={{ 
            marginTop: isMobile ? '1.5rem' : '2rem',
            backgroundColor: '#ffffff', 
            borderRadius: isMobile ? '0.5rem' : '0.75rem', 
            border: '1px solid #e5e7eb', 
            padding: isMobile ? '1rem' : '1.5rem' 
          }}>
            <h3 style={{ 
              fontSize: isMobile ? '1rem' : '1.125rem', 
              fontWeight: '500', 
              color: '#111827', 
              marginBottom: '1rem' 
            }}>How it works</h3>
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: isMobile ? '1fr' : 'repeat(3, 1fr)', 
              gap: isMobile ? '1rem' : '1.5rem' 
            }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ 
                  width: isMobile ? '2.5rem' : '3rem', 
                  height: isMobile ? '2.5rem' : '3rem', 
                  backgroundColor: '#f3f4f6', 
                  borderRadius: '0.5rem', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center', 
                  margin: '0 auto 0.75rem' 
                }}>
                  <span style={{ 
                    fontSize: isMobile ? '1rem' : '1.125rem', 
                    fontWeight: '600', 
                    color: '#4b5563' 
                  }}>1</span>
                </div>
                <h4 style={{ 
                  fontWeight: '500', 
                  color: '#111827', 
                  marginBottom: '0.5rem',
                  fontSize: isMobile ? '0.875rem' : '1rem'
                }}>Upload Photos</h4>
                <p style={{ 
                  fontSize: isMobile ? '0.75rem' : '0.875rem', 
                  color: '#4b5563' 
                }}>Upload a clear photo of yourself and the clothing item you want to try on.</p>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ 
                  width: isMobile ? '2.5rem' : '3rem', 
                  height: isMobile ? '2.5rem' : '3rem', 
                  backgroundColor: '#f3f4f6', 
                  borderRadius: '0.5rem', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center', 
                  margin: '0 auto 0.75rem' 
                }}>
                  <span style={{ 
                    fontSize: isMobile ? '1rem' : '1.125rem', 
                    fontWeight: '600', 
                    color: '#4b5563' 
                  }}>2</span>
                </div>
                <h4 style={{ 
                  fontWeight: '500', 
                  color: '#111827', 
                  marginBottom: '0.5rem',
                  fontSize: isMobile ? '0.875rem' : '1rem'
                }}>Add Measurements</h4>
                <p style={{ 
                  fontSize: isMobile ? '0.75rem' : '0.875rem', 
                  color: '#4b5563' 
                }}>Enter your height and weight for accurate size matching.</p>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ 
                  width: isMobile ? '2.5rem' : '3rem', 
                  height: isMobile ? '2.5rem' : '3rem', 
                  backgroundColor: '#f3f4f6', 
                  borderRadius: '0.5rem', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center', 
                  margin: '0 auto 0.75rem' 
                }}>
                  <span style={{ 
                    fontSize: isMobile ? '1rem' : '1.125rem', 
                    fontWeight: '600', 
                    color: '#4b5563' 
                  }}>3</span>
                </div>
                <h4 style={{ 
                  fontWeight: '500', 
                  color: '#111827', 
                  marginBottom: '0.5rem',
                  fontSize: isMobile ? '0.875rem' : '1rem'
                }}>Get Results</h4>
                <p style={{ 
                  fontSize: isMobile ? '0.75rem' : '0.875rem', 
                  color: '#4b5563' 
                }}>See how the clothing looks on you with AI-powered fitting analysis.</p>
              </div>
            </div>
            
            {/* 🔥 수정된 시스템 정보 */}
            <div style={{ 
              marginTop: '1.5rem', 
              padding: isMobile ? '0.75rem' : '1rem', 
              backgroundColor: '#f0f9ff', 
              borderRadius: '0.5rem',
              fontSize: isMobile ? '0.75rem' : '0.875rem',
              color: '#1e40af',
              border: '1px solid #bfdbfe'
            }}>
              <p style={{ margin: 0, fontWeight: '500' }}>
                🔧 완전 수정 버전 (모든 API 연동 문제 해결):
              </p>
              <p style={{ margin: '0.25rem 0 0 0' }}>
                🎯 백엔드: http://localhost:8000 (포트 통일) | 
                WebSocket: ws://localhost:8000/api/ws/ai-pipeline
              </p>
              <p style={{ margin: '0.25rem 0 0 0' }}>
                📡 API 엔드포인트: /api/step/* (이중 prefix 수정) | 완전한 에러 처리
              </p>
              {systemInfo && (
                <p style={{ margin: '0.25rem 0 0 0' }}>
                  💻 {systemInfo.app_name} v{systemInfo.app_version} | 
                  {systemInfo.device_name} {systemInfo.is_m3_max ? '🍎' : ''} | 
                  💾 {systemInfo.available_memory_gb}GB 사용가능
                </p>
              )}
              {!isMobile && (
                <p style={{ margin: '0.25rem 0 0 0' }}>
                  🔧 헤더의 "Test", "System", "Complete" 버튼으로 기능 테스트 가능
                </p>
              )}
            </div>
          </div>
        )}

        {/* 모바일 개발 도구 (하단에 표시) */}
        {isMobile && (
          <div style={{
            position: 'fixed',
            bottom: '1rem',
            right: '1rem',
            zIndex: 40
          }}>
            <button
              onClick={() => {
                const devMenu = document.getElementById('mobile-dev-menu');
                if (devMenu) {
                  devMenu.style.display = devMenu.style.display === 'none' ? 'block' : 'none';
                }
              }}
              style={{
                width: '3rem',
                height: '3rem',
                borderRadius: '50%',
                backgroundColor: '#3b82f6',
                color: '#ffffff',
                border: 'none',
                boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              ⚙️
            </button>
            <div
              id="mobile-dev-menu"
              style={{
                display: 'none',
                position: 'absolute',
                bottom: '3.5rem',
                right: '0',
                backgroundColor: '#ffffff',
                border: '1px solid #e5e7eb',
                borderRadius: '0.5rem',
                padding: '0.5rem',
                minWidth: '10rem',
                boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)'
              }}
            >
              <button
                onClick={handleTestConnection}
                disabled={isProcessing}
                style={{
                  width: '100%',
                  padding: '0.5rem',
                  fontSize: '0.75rem',
                  backgroundColor: 'transparent',
                  border: 'none',
                  textAlign: 'left',
                  cursor: 'pointer',
                  borderRadius: '0.25rem'
                }}
              >
                Test Connection
              </button>
              <button
                onClick={handleSystemInfo}
                disabled={isProcessing}
                style={{
                  width: '100%',
                  padding: '0.5rem',
                  fontSize: '0.75rem',
                  backgroundColor: 'transparent',
                  border: 'none',
                  textAlign: 'left',
                  cursor: 'pointer',
                  borderRadius: '0.25rem'
                }}
              >
                System Info
              </button>
              <button
                onClick={handleCompletePipeline}
                disabled={isProcessing || !personImage || !clothingImage}
                style={{
                  width: '100%',
                  padding: '0.5rem',
                  fontSize: '0.75rem',
                  backgroundColor: 'transparent',
                  border: 'none',
                  textAlign: 'left',
                  cursor: 'pointer',
                  borderRadius: '0.25rem',
                  opacity: isProcessing || !personImage || !clothingImage ? 0.5 : 1
                }}
              >
                Complete Pipeline
              </button>
            </div>
          </div>
        )}
      </main>

      {/* CSS Animation */}
      <style>{`
        @keyframes spin {
          from {
            transform: rotate(0deg);
          }
          to {
            transform: rotate(360deg);
          }
        }
        
        code {
          background-color: #f3f4f6;
          padding: 0.125rem 0.25rem;
          border-radius: 0.25rem;
          font-family: 'Courier New', monospace;
          font-size: 0.8em;
        }

        /* 모바일 최적화 스크롤바 */
        ::-webkit-scrollbar {
          width: 4px;
          height: 4px;
        }
        
        ::-webkit-scrollbar-track {
          background: transparent;
        }
        
        ::-webkit-scrollbar-thumb {
          background: #d1d5db;
          border-radius: 2px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
          background: #9ca3af;
        }

        /* 터치 디바이스 최적화 */
        @media (hover: none) and (pointer: coarse) {
          button:hover {
            background-color: inherit !important;
          }
        }

        /* 모바일 뷰포트 최적화 */
        @media screen and (max-width: 768px) {
          /* 터치 영역 최적화 */
          button {
            min-height: 44px;
            min-width: 44px;
          }
          
          /* 텍스트 가독성 향상 */
          body {
            -webkit-text-size-adjust: 100%;
            text-size-adjust: 100%;
          }
          
          /* 가로 스크롤 방지 */
          * {
            max-width: 100%;
            overflow-wrap: break-word;
          }
        }
      `}</style>
    </div>
  );
};

export default App;