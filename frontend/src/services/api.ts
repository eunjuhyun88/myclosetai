// frontend/src/services/api.ts
/**
 * MyCloset AI 백엔드 API 통신 서비스
 * 실제 AI 모델과 연동하는 API 클라이언트
 */

import axios, { AxiosProgressEvent } from 'axios';

// API 기본 설정
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2분 (AI 처리 시간 고려)
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

// 요청/응답 인터셉터
apiClient.interceptors.request.use(
  (config) => {
    console.log(`🚀 API 요청: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('❌ API 요청 오류:', error);
    return Promise.reject(error);
  }
);

apiClient.interceptors.response.use(
  (response) => {
    console.log(`✅ API 응답: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('❌ API 응답 오류:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// 타입 정의
export interface VirtualTryOnRequest {
  personImage: File;
  clothingImage: File;
  height: number;
  weight: number;
  chest?: number;
  waist?: number;
  hips?: number;
}

export interface VirtualTryOnResponse {
  task_id: string;
  status: 'processing' | 'completed' | 'error';
  message: string;
  estimated_time?: string;
}

export interface TaskStatus {
  status: 'processing' | 'completed' | 'error';
  progress: number;
  current_step: string;
  steps: Array<{
    id: string;
    name: string;
    status: 'pending' | 'processing' | 'completed' | 'error';
  }>;
  result?: TryOnResult;
  error?: string;
  created_at: number;
  completed_at?: number;
}

export interface TryOnResult {
  fitted_image: string; // base64 인코딩된 이미지
  confidence: number;
  processing_time: number;
  body_analysis: {
    measurements: Record<string, number>;
    pose_keypoints: number[][];
    body_type: string;
  };
  clothing_analysis: {
    category: string;
    style: string;
    colors: string[];
    pattern: string;
  };
  fit_score: number;
  recommendations: string[];
  model_used: string;
  image_specs: {
    resolution: [number, number];
    format: string;
    quality: number;
  };
}

export interface ModelInfo {
  available_models: string[];
  model_info: Record<string, any>;
  device: string;
  initialized: boolean;
}

// API 서비스 클래스
export class MyClosetAPI {
  
  /**
   * 서버 헬스체크
   */
  static async healthCheck(): Promise<boolean> {
    try {
      const response = await apiClient.get('/health');
      return response.status === 200;
    } catch (error) {
      console.error('헬스체크 실패:', error);
      return false;
    }
  }

  /**
   * 가상 피팅 요청
   */
  static async requestVirtualTryOn(
    data: VirtualTryOnRequest,
    onProgress?: (progress: AxiosProgressEvent) => void
  ): Promise<VirtualTryOnResponse> {
    const formData = new FormData();
    formData.append('person_image', data.personImage);
    formData.append('clothing_image', data.clothingImage);
    formData.append('height', data.height.toString());
    formData.append('weight', data.weight.toString());
    
    if (data.chest) formData.append('chest', data.chest.toString());
    if (data.waist) formData.append('waist', data.waist.toString());
    if (data.hips) formData.append('hips', data.hips.toString());

    const response = await apiClient.post('/api/virtual-tryon', formData, {
      onUploadProgress: onProgress,
    });

    return response.data;
  }

  /**
   * 태스크 상태 조회
   */
  static async getTaskStatus(taskId: string): Promise<TaskStatus> {
    const response = await apiClient.get(`/api/status/${taskId}`);
    return response.data;
  }

  /**
   * 태스크 결과 조회
   */
  static async getTaskResult(taskId: string): Promise<TryOnResult> {
    const response = await apiClient.get(`/api/result/${taskId}`);
    return response.data;
  }

  /**
   * 신체 분석만 수행
   */
  static async analyzeBody(image: File): Promise<any> {
    const formData = new FormData();
    formData.append('image', image);

    const response = await apiClient.post('/api/analyze-body', formData);
    return response.data;
  }

  /**
   * 의류 분석만 수행
   */
  static async analyzeClothing(image: File): Promise<any> {
    const formData = new FormData();
    formData.append('image', image);

    const response = await apiClient.post('/api/analyze-clothing', formData);
    return response.data;
  }

  /**
   * 사용 가능한 모델 목록 조회
   */
  static async getAvailableModels(): Promise<ModelInfo> {
    const response = await apiClient.get('/api/models');
    return response.data;
  }

  /**
   * 실시간 진행상황 WebSocket 연결
   */
  static connectWebSocket(
    taskId: string,
    onMessage: (data: any) => void,
    onError?: (error: Event) => void,
    onClose?: (event: CloseEvent) => void
  ): WebSocket {
    const wsUrl = API_BASE_URL.replace('http', 'ws') + `/ws/fitting/${taskId}`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('WebSocket 메시지 파싱 오류:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket 오류:', error);
      onError?.(error);
    };

    ws.onclose = (event) => {
      console.log('WebSocket 연결 종료:', event);
      onClose?.(event);
    };

    return ws;
  }
}

// frontend/src/hooks/useVirtualTryOn.ts
/**
 * 가상 피팅 커스텀 훅
 * 실제 AI 백엔드와 연동하는 비즈니스 로직
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { MyClosetAPI, VirtualTryOnRequest, TaskStatus, TryOnResult } from '../services/api';

interface UseVirtualTryOnReturn {
  // 상태
  isProcessing: boolean;
  progress: number;
  currentStep: string;
  result: TryOnResult | null;
  error: string | null;
  
  // 단계별 상태
  steps: TaskStatus['steps'];
  
  // 함수
  startVirtualTryOn: (data: VirtualTryOnRequest) => Promise<void>;
  resetState: () => void;
  
  // 추가 정보
  processingTime: number;
  taskId: string | null;
}

export const useVirtualTryOn = (): UseVirtualTryOnReturn => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [steps, setSteps] = useState<TaskStatus['steps']>([]);
  const [result, setResult] = useState<TryOnResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState(0);
  const [taskId, setTaskId] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const startTimeRef = useRef<number>(0);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const resetState = useCallback(() => {
    setIsProcessing(false);
    setProgress(0);
    setCurrentStep('');
    setSteps([]);
    setResult(null);
    setError(null);
    setProcessingTime(0);
    setTaskId(null);
    
    // WebSocket 정리
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    // 타이머 정리
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const pollTaskStatus = useCallback(async (taskId: string) => {
    try {
      const status = await MyClosetAPI.getTaskStatus(taskId);
      
      setProgress(status.progress);
      setCurrentStep(status.current_step);
      setSteps(status.steps);
      
      if (status.status === 'completed' && status.result) {
        setResult(status.result);
        setIsProcessing(false);
        
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      } else if (status.status === 'error') {
        setError(status.error || '처리 중 오류가 발생했습니다.');
        setIsProcessing(false);
        
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      }
    } catch (err) {
      console.error('상태 조회 실패:', err);
    }
  }, []);

  const startVirtualTryOn = useCallback(async (data: VirtualTryOnRequest) => {
    resetState();
    setIsProcessing(true);
    setError(null);
    startTimeRef.current = Date.now();
    
    try {
      // 1. 가상 피팅 요청
      console.log('🎨 가상 피팅 요청 시작...');
      const response = await MyClosetAPI.requestVirtualTryOn(data, (progressEvent) => {
        const uploadProgress = Math.round((progressEvent.loaded * 100) / (progressEvent.total || 1));
        setProgress(Math.min(uploadProgress * 0.1, 10)); // 업로드는 전체의 10%
      });
      
      setTaskId(response.task_id);
      console.log(`✅ 태스크 생성됨: ${response.task_id}`);
      
      // 2. WebSocket 연결 시도
      try {
        wsRef.current = MyClosetAPI.connectWebSocket(
          response.task_id,
          (data) => {
            console.log('📡 실시간 업데이트:', data);
            if (data.progress !== undefined) setProgress(data.progress);
            if (data.current_step) setCurrentStep(data.current_step);
            if (data.steps) setSteps(data.steps);
            
            if (data.status === 'completed' && data.result) {
              setResult(data.result);
              setIsProcessing(false);
            } else if (data.status === 'error') {
              setError(data.error || '처리 중 오류가 발생했습니다.');
              setIsProcessing(false);
            }
          },
          (error) => {
            console.warn('⚠️ WebSocket 오류, 폴링으로 대체:', error);
            // WebSocket 실패 시 폴링으로 대체
            intervalRef.current = setInterval(() => {
              pollTaskStatus(response.task_id);
            }, 2000);
          }
        );
      } catch (wsError) {
        console.warn('⚠️ WebSocket 연결 실패, 폴링 사용:', wsError);
        // WebSocket 실패 시 폴링으로 대체
        intervalRef.current = setInterval(() => {
          pollTaskStatus(response.task_id);
        }, 2000);
      }
      
    } catch (err: any) {
      console.error('❌ 가상 피팅 요청 실패:', err);
      setError(err.response?.data?.detail || err.message || '가상 피팅 요청에 실패했습니다.');
      setIsProcessing(false);
    }
  }, [pollTaskStatus, resetState]);

  // 처리 시간 업데이트
  useEffect(() => {
    let timer: NodeJS.Timeout;
    
    if (isProcessing && startTimeRef.current) {
      timer = setInterval(() => {
        setProcessingTime(Date.now() - startTimeRef.current);
      }, 1000);
    }
    
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isProcessing]);

  // 컴포넌트 언마운트 시 정리
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  return {
    isProcessing,
    progress,
    currentStep,
    result,
    error,
    steps,
    startVirtualTryOn,
    resetState,
    processingTime,
    taskId,
  };
};

// frontend/src/components/VirtualTryOnInterface.tsx
/**
 * 실제 AI 백엔드와 연동되는 가상 피팅 인터페이스
 */

import React, { useState, useRef } from 'react';
import { Upload, Zap, Download, Share2, RefreshCw, AlertCircle, CheckCircle } from 'lucide-react';
import { useVirtualTryOn } from '../hooks/useVirtualTryOn';

interface UserMeasurements {
  height: number;
  weight: number;
  chest?: number;
  waist?: number;
  hips?: number;
}

const VirtualTryOnInterface: React.FC = () => {
  const [personImage, setPersonImage] = useState<File | null>(null);
  const [clothingImage, setClothingImage] = useState<File | null>(null);
  const [measurements, setMeasurements] = useState<UserMeasurements>({
    height: 170,
    weight: 65,
  });

  const personInputRef = useRef<HTMLInputElement>(null);
  const clothingInputRef = useRef<HTMLInputElement>(null);

  const {
    isProcessing,
    progress,
    currentStep,
    result,
    error,
    steps,
    startVirtualTryOn,
    resetState,
    processingTime,
  } = useVirtualTryOn();

  const handleImageUpload = (file: File, type: 'person' | 'clothing') => {
    if (type === 'person') {
      setPersonImage(file);
    } else {
      setClothingImage(file);
    }
  };

  const handleStartFitting = async () => {
    if (!personImage || !clothingImage) {
      alert('사람 사진과 의류 사진을 모두 업로드해주세요.');
      return;
    }

    await startVirtualTryOn({
      personImage,
      clothingImage,
      ...measurements,
    });
  };

  const formatTime = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    return `${Math.floor(seconds / 60)}:${(seconds % 60).toString().padStart(2, '0')}`;
  };

  const getStepIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'processing':
        return <RefreshCw className="w-4 h-4 text-blue-500 animate-spin" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return <div className="w-4 h-4 bg-gray-300 rounded-full" />;
    }
  };

  const downloadResult = () => {
    if (result?.fitted_image) {
      const link = document.createElement('a');
      link.href = `data:image/jpeg;base64,${result.fitted_image}`;
      link.download = `mycloset-fitted-${Date.now()}.jpg`;
      link.click();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50 p-4">
      <div className="max-w-6xl mx-auto">
        {/* 헤더 */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            MyCloset AI 가상 피팅
          </h1>
          <p className="text-gray-600">
            실제 AI 모델로 구현된 고품질 가상 피팅 시스템
          </p>
        </div>

        {/* 메인 컨텐츠 */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* 이미지 업로드 섹션 */}
          <div className="lg:col-span-2 space-y-6">
            
            {/* 이미지 업로드 */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* 사람 이미지 */}
              <div className="bg-white p-6 rounded-lg shadow-lg">
                <h3 className="text-lg font-semibold mb-4">사람 사진</h3>
                {personImage ? (
                  <div className="relative">
                    <img
                      src={URL.createObjectURL(personImage)}
                      alt="Person"
                      className="w-full h-64 object-cover rounded-lg"
                    />
                    <button
                      onClick={() => personInputRef.current?.click()}
                      className="absolute top-2 right-2 bg-blue-500 text-white p-2 rounded-full hover:bg-blue-600"
                    >
                      <Upload className="w-4 h-4" />
                    </button>
                  </div>
                ) : (
                  <div
                    onClick={() => personInputRef.current?.click()}
                    className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-blue-400 transition-colors"
                  >
                    <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-600">사람 사진을 업로드하세요</p>
                    <p className="text-sm text-gray-400 mt-2">JPG, PNG 최대 10MB</p>
                  </div>
                )}
                <input
                  ref={personInputRef}
                  type="file"
                  accept="image/*"
                  onChange={(e) => e.target.files?.[0] && handleImageUpload(e.target.files[0], 'person')}
                  className="hidden"
                />
              </div>

              {/* 의류 이미지 */}
              <div className="bg-white p-6 rounded-lg shadow-lg">
                <h3 className="text-lg font-semibold mb-4">의류 사진</h3>
                {clothingImage ? (
                  <div className="relative">
                    <img
                      src={URL.createObjectURL(clothingImage)}
                      alt="Clothing"
                      className="w-full h-64 object-cover rounded-lg"
                    />
                    <button
                      onClick={() => clothingInputRef.current?.click()}
                      className="absolute top-2 right-2 bg-blue-500 text-white p-2 rounded-full hover:bg-blue-600"
                    >
                      <Upload className="w-4 h-4" />
                    </button>
                  </div>
                ) : (
                  <div
                    onClick={() => clothingInputRef.current?.click()}
                    className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-blue-400 transition-colors"
                  >
                    <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-600">의류 사진을 업로드하세요</p>
                    <p className="text-sm text-gray-400 mt-2">JPG, PNG 최대 10MB</p>
                  </div>
                )}
                <input
                  ref={clothingInputRef}
                  type="file"
                  accept="image/*"
                  onChange={(e) => e.target.files?.[0] && handleImageUpload(e.target.files[0], 'clothing')}
                  className="hidden"
                />
              </div>
            </div>

            {/* 신체 측정값 */}
            <div className="bg-white p-6 rounded-lg shadow-lg">
              <h3 className="text-lg font-semibold mb-4">신체 측정값</h3>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    키 (cm)
                  </label>
                  <input
                    type="number"
                    value={measurements.height}
                    onChange={(e) => setMeasurements({...measurements, height: Number(e.target.value)})}
                    className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    min="140"
                    max="220"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    몸무게 (kg)
                  </label>
                  <input
                    type="number"
                    value={measurements.weight}
                    onChange={(e) => setMeasurements({...measurements, weight: Number(e.target.value)})}
                    className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    min="30"
                    max="200"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    가슴둘레 (cm)
                  </label>
                  <input
                    type="number"
                    value={measurements.chest || ''}
                    onChange={(e) => setMeasurements({...measurements, chest: e.target.value ? Number(e.target.value) : undefined})}
                    className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="선택사항"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    허리둘레 (cm)
                  </label>
                  <input
                    type="number"
                    value={measurements.waist || ''}
                    onChange={(e) => setMeasurements({...measurements, waist: e.target.value ? Number(e.target.value) : undefined})}
                    className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="선택사항"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    엉덩이둘레 (cm)
                  </label>
                  <input
                    type="number"
                    value={measurements.hips || ''}
                    onChange={(e) => setMeasurements({...measurements, hips: e.target.value ? Number(e.target.value) : undefined})}
                    className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="선택사항"
                  />
                </div>
              </div>
            </div>

            {/* 가상 피팅 버튼 */}
            <div className="bg-white p-6 rounded-lg shadow-lg">
              <button
                onClick={handleStartFitting}
                disabled={!personImage || !clothingImage || isProcessing}
                className={`w-full py-4 px-6 rounded-lg font-semibold text-lg transition-all ${
                  !personImage || !clothingImage || isProcessing
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-blue-500 text-white hover:bg-blue-600 hover:shadow-lg'
                }`}
              >
                {isProcessing ? (
                  <div className="flex items-center justify-center space-x-2">
                    <RefreshCw className="w-5 h-5 animate-spin" />
                    <span>AI 가상 피팅 처리 중... ({formatTime(processingTime)})</span>
                  </div>
                ) : (
                  <div className="flex items-center justify-center space-x-2">
                    <Zap className="w-5 h-5" />
                    <span>AI 가상 피팅 시작</span>
                  </div>
                )}
              </button>
            </div>
          </div>

          {/* 결과 및 진행상황 섹션 */}
          <div className="space-y-6">
            
            {/* 진행상황 */}
            {(isProcessing || result || error) && (
              <div className="bg-white p-6 rounded-lg shadow-lg">
                <h3 className="text-lg font-semibold mb-4">진행상황</h3>
                
                {/* 진행률 바 */}
                <div className="mb-4">
                  <div className="flex justify-between text-sm text-gray-600 mb-1">
                    <span>전체 진행률</span>
                    <span>{progress}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>

                {/* 현재 단계 */}
                {currentStep && (
                  <div className="mb-4 p-3 bg-blue-50 rounded-lg">
                    <p className="text-sm font-medium text-blue-800">
                      현재 단계: {currentStep}
                    </p>
                  </div>
                )}

                {/* 단계별 상태 */}
                {steps.length > 0 && (
                  <div className="space-y-2">
                    {steps.map((step) => (
                      <div key={step.id} className="flex items-center space-x-3">
                        {getStepIcon(step.status)}
                        <span className={`text-sm ${
                          step.status === 'completed' ? 'text-green-700' :
                          step.status === 'processing' ? 'text-blue-700' :
                          step.status === 'error' ? 'text-red-700' : 'text-gray-600'
                        }`}>
                          {step.name}
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                {/* 에러 메시지 */}
                {error && (
                  <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <AlertCircle className="w-5 h-5 text-red-500" />
                      <p className="text-red-700 font-medium">처리 중 오류 발생</p>
                    </div>
                    <p className="text-red-600 text-sm mt-1">{error}</p>
                    <button
                      onClick={resetState}
                      className="mt-2 text-red-600 text-sm underline hover:no-underline"
                    >
                      다시 시도
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* 결과 이미지 */}
            {result && (
              <div className="bg-white p-6 rounded-lg shadow-lg">
                <h3 className="text-lg font-semibold mb-4">가상 피팅 결과</h3>
                
                {/* 결과 이미지 */}
                <div className="mb-4">
                  <img
                    src={`data:image/jpeg;base64,${result.fitted_image}`}
                    alt="Virtual Try-On Result"
                    className="w-full rounded-lg shadow-md"
                  />
                </div>

                {/* 결과 정보 */}
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">품질 점수:</span>
                    <span className="font-semibold">{(result.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">처리 시간:</span>
                    <span className="font-semibold">{result.processing_time.toFixed(1)}초</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">핏 점수:</span>
                    <span className="font-semibold">{(result.fit_score * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">사용된 모델:</span>
                    <span className="font-semibold">{result.model_used}</span>
                  </div>
                </div>

                {/* 추천사항 */}
                {result.recommendations.length > 0 && (
                  <div className="mt-4 p-3 bg-green-50 rounded-lg">
                    <h4 className="text-sm font-semibold text-green-800 mb-2">추천사항</h4>
                    <ul className="text-sm text-green-700 space-y-1">
                      {result.recommendations.map((rec, index) => (
                        <li key={index}>• {rec}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* 액션 버튼 */}
                <div className="flex space-x-3 mt-6">
                  <button
                    onClick={downloadResult}
                    className="flex-1 bg-green-500 text-white py-2 px-4 rounded-lg hover:bg-green-600 transition-colors flex items-center justify-center space-x-2"
                  >
                    <Download className="w-4 h-4" />
                    <span>다운로드</span>
                  </button>
                  <button
                    onClick={() => {
                      if (navigator.share) {
                        navigator.share({
                          title: 'MyCloset AI 가상 피팅 결과',
                          text: '내 가상 피팅 결과를 확인해보세요!',
                        });
                      }
                    }}
                    className="flex-1 bg-blue-500 text-white py-2 px-4 rounded-lg hover:bg-blue-600 transition-colors flex items-center justify-center space-x-2"
                  >
                    <Share2 className="w-4 h-4" />
                    <span>공유</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default VirtualTryOnInterface;