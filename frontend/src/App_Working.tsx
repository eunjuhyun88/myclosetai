import React, { useState, useRef, useEffect, useCallback } from 'react';

// ===============================================================
// 🔧 타입 정의들 (백엔드 완전 호환)
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
  details?: any;
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

// 8단계 정의 (백엔드와 완전 동일)
const PIPELINE_STEPS: PipelineStep[] = [
  {
    id: 1,
    name: "이미지 업로드 검증",
    description: "사용자 사진과 의류 이미지를 검증합니다",
    endpoint: "/api/step/1/upload-validation",
    processing_time: 0.5
  },
  {
    id: 2,
    name: "신체 측정값 검증",
    description: "키와 몸무게 등 신체 정보를 검증합니다",
    endpoint: "/api/step/2/measurements-validation",
    processing_time: 0.3
  },
  {
    id: 3,
    name: "인체 파싱",
    description: "AI가 신체 부위를 20개 영역으로 분석합니다",
    endpoint: "/api/step/3/human-parsing",
    processing_time: 1.2
  },
  {
    id: 4,
    name: "포즈 추정",
    description: "18개 키포인트로 자세를 분석합니다",
    endpoint: "/api/step/4/pose-estimation",
    processing_time: 0.8
  },
  {
    id: 5,
    name: "의류 분석",
    description: "의류 스타일과 색상을 분석합니다",
    endpoint: "/api/step/5/clothing-analysis",
    processing_time: 0.6
  },
  {
    id: 6,
    name: "기하학적 매칭",
    description: "신체와 의류를 정확히 매칭합니다",
    endpoint: "/api/step/6/geometric-matching",
    processing_time: 1.5
  },
  {
    id: 7,
    name: "가상 피팅",
    description: "AI로 가상 착용 결과를 생성합니다",
    endpoint: "/api/step/7/virtual-fitting",
    processing_time: 2.5
  },
  {
    id: 8,
    name: "결과 분석",
    description: "최종 결과를 확인하고 저장합니다",
    endpoint: "/api/step/8/result-analysis",
    processing_time: 0.3
  }
];

// ===============================================================
// 🔧 API 클라이언트 (완전 간소화된 버전)
// ===============================================================

class SimpleAPIClient {
  private baseURL: string;

  constructor(baseURL: string = 'http://localhost:8000') {
    this.baseURL = baseURL;
  }

  // 헬스체크
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

  // 시스템 정보 조회
  async getSystemInfo(): Promise<SystemInfo> {
    const response = await fetch(`${this.baseURL}/api/system/info`);
    if (!response.ok) {
      throw new Error(`시스템 정보 조회 실패: ${response.status}`);
    }
    return await response.json();
  }

  // 완전한 파이프라인 실행 (핵심!)
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

    try {
      console.log('🚀 완전한 파이프라인 실행 시작');
      
      const response = await fetch(`${this.baseURL}/api/pipeline/complete`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Pipeline failed: ${response.status} - ${errorText}`);
      }

      const result: TryOnResult = await response.json();
      console.log('✅ 완전한 파이프라인 완료:', result);
      return result;
      
    } catch (error) {
      console.error('❌ 완전한 파이프라인 실패:', error);
      throw error;
    }
  }

  // 시뮬레이션 모드 (백엔드가 없을 때)
  async simulateCompletePipeline(
    personImage: File,
    clothingImage: File,
    measurements: UserMeasurements
  ): Promise<TryOnResult> {
    console.log('🎭 시뮬레이션 모드 실행 중...');
    
    // 실제 처리 시뮬레이션 (6초)
    await new Promise(resolve => setTimeout(resolve, 6000));
    
    // 가짜 base64 이미지 (실제로는 백엔드에서 생성됨)
    const fakeImage = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';
    
    return {
      success: true,
      message: "시뮬레이션 완료 - 실제 백엔드 연결 시 진짜 결과가 나옵니다",
      processing_time: 6.0,
      confidence: 0.92,
      session_id: `sim_${Date.now()}`,
      fitted_image: fakeImage,
      fit_score: 0.88,
      measurements: {
        chest: 88 + (measurements.weight - 65) * 0.9,
        waist: 74 + (measurements.weight - 65) * 0.7,
        hip: 94 + (measurements.weight - 65) * 0.8,
        bmi: measurements.weight / ((measurements.height / 100) ** 2)
      },
      clothing_analysis: {
        category: '상의',
        style: '캐주얼',
        dominant_color: [95, 145, 195],
        color_name: '블루',
        material: '코튼',
        pattern: '솔리드'
      },
      recommendations: [
        '멋진 선택입니다! 이 스타일이 잘 어울려요.',
        '약간 더 큰 사이즈를 고려해보세요.',
        '이 색상이 피부톤과 잘 맞습니다.'
      ]
    };
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
  const [apiClient] = useState(() => new SimpleAPIClient());

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
  const [useSimulation, setUseSimulation] = useState(false);

  // 반응형 상태
  const [isMobile, setIsMobile] = useState(false);

  // 파일 참조
  const personImageRef = useRef<HTMLInputElement>(null);
  const clothingImageRef = useRef<HTMLInputElement>(null);

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

  // 서버 헬스체크
  useEffect(() => {
    const checkHealth = async () => {
      setIsCheckingHealth(true);
      try {
        const result = await apiClient.healthCheck();
        setIsServerHealthy(result.success);
        
        if (result.success && result.data) {
          console.log('✅ 서버 상태:', result.data);
          setUseSimulation(false);
        } else {
          console.log('⚠️ 서버 연결 불가 - 시뮬레이션 모드 활성화');
          setUseSimulation(true);
        }
      } catch {
        setIsServerHealthy(false);
        setUseSimulation(true);
      } finally {
        setIsCheckingHealth(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000); // 30초마다
    return () => clearInterval(interval);
  }, [apiClient]);

  // 시스템 정보 조회
  useEffect(() => {
    const fetchSystemInfo = async () => {
      if (!useSimulation) {
        try {
          const info = await apiClient.getSystemInfo();
          setSystemInfo(info);
          console.log('📊 시스템 정보:', info);
        } catch (error) {
          console.error('시스템 정보 조회 실패:', error);
        }
      }
    };

    if (isServerHealthy && !useSimulation) {
      fetchSystemInfo();
    }
  }, [isServerHealthy, useSimulation, apiClient]);

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

  // 리셋
  const reset = useCallback(() => {
    setPersonImage(null);
    setClothingImage(null);
    setPersonImagePreview(null);
    setClothingImagePreview(null);
    setResult(null);
    setFileErrors({});
    setError(null);
    setIsProcessing(false);
    setProgress(0);
    setProgressMessage('');
  }, []);

  // 에러 클리어
  const clearError = useCallback(() => setError(null), []);

  // ===============================================================
  // 🔧 메인 처리 함수
  // ===============================================================

  // 완전한 파이프라인 실행
  const handleRunPipeline = useCallback(async () => {
    if (!personImage || !clothingImage) {
      alert('사용자 이미지와 의류 이미지를 모두 업로드해주세요.');
      return;
    }

    if (measurements.height <= 0 || measurements.weight <= 0) {
      alert('올바른 키와 몸무게를 입력해주세요.');
      return;
    }

    setIsProcessing(true);
    setProgress(0);
    setError(null);
    setResult(null);

    try {
      // 진행률 시뮬레이션
      const progressSteps = [
        { progress: 10, message: '이미지 검증 중...' },
        { progress: 25, message: '신체 측정값 검증 중...' },
        { progress: 40, message: '인체 파싱 중...' },
        { progress: 55, message: '포즈 추정 중...' },
        { progress: 70, message: '의류 분석 중...' },
        { progress: 85, message: '가상 피팅 생성 중...' },
        { progress: 95, message: '결과 분석 중...' },
        { progress: 100, message: '완료!' }
      ];

      // 진행률 업데이트
      for (const step of progressSteps) {
        setProgress(step.progress);
        setProgressMessage(step.message);
        await new Promise(resolve => setTimeout(resolve, 800));
      }

      // 실제 파이프라인 실행
      let result: TryOnResult;
      
      if (useSimulation) {
        result = await apiClient.simulateCompletePipeline(personImage, clothingImage, measurements);
      } else {
        result = await apiClient.runCompletePipeline(personImage, clothingImage, measurements);
      }

      if (result.success) {
        setResult(result);
        setProgress(100);
        setProgressMessage('🎉 가상 피팅 완료!');
        
        setTimeout(() => {
          setIsProcessing(false);
          alert('🎉 가상 피팅이 완료되었습니다! 결과를 확인해보세요.');
        }, 1500);
      } else {
        throw new Error(result.message || '파이프라인 처리 실패');
      }
      
    } catch (error: any) {
      console.error('❌ 파이프라인 실패:', error);
      setError(error.message);
      setIsProcessing(false);
      setProgress(0);
      setProgressMessage('');
    }
  }, [personImage, clothingImage, measurements, apiClient, useSimulation]);

  // 요청 취소 핸들러
  const handleCancelRequest = useCallback(() => {
    if (isProcessing) {
      setIsProcessing(false);
      setProgress(0);
      setProgressMessage('');
      alert('요청이 취소되었습니다.');
    }
  }, [isProcessing]);

  // ===============================================================
  // 🔧 개발 도구 함수들
  // ===============================================================

  const handleTestConnection = useCallback(async () => {
    try {
      const result = await apiClient.healthCheck();
      console.log('연결 테스트 결과:', result);
      alert(result.success ? '✅ 서버 연결 성공!' : '❌ 서버 연결 실패 - 시뮬레이션 모드로 진행');
    } catch (error) {
      console.error('연결 테스트 실패:', error);
      alert('❌ 연결 테스트 실패 - 시뮬레이션 모드로 진행');
    }
  }, [apiClient]);

  const handleSystemInfo = useCallback(async () => {
    if (useSimulation) {
      alert('📊 시뮬레이션 모드\n🎭 실제 백엔드 연결 시 시스템 정보 표시');
      return;
    }

    try {
      const info = await apiClient.getSystemInfo();
      console.log('시스템 정보:', info);
      alert(`✅ ${info.app_name} v${info.app_version}\n🎯 ${info.device_name}\n💾 ${info.available_memory_gb}GB 사용가능`);
    } catch (error) {
      console.error('시스템 정보 실패:', error);
      alert('❌ 시스템 정보 조회 실패');
    }
  }, [apiClient, useSimulation]);

  // ===============================================================
  // 🔧 유효성 검사 함수들
  // ===============================================================

  const canRunPipeline = useCallback(() => {
    return personImage && clothingImage && 
           !fileErrors.person && !fileErrors.clothing &&
           measurements.height > 0 && measurements.weight > 0 &&
           measurements.height >= 100 && measurements.height <= 250 &&
           measurements.weight >= 30 && measurements.weight <= 300;
  }, [personImage, clothingImage, fileErrors, measurements]);

  // 서버 상태 색상/텍스트
  const getServerStatusColor = useCallback(() => {
    if (isCheckingHealth) return '#f59e0b';
    if (useSimulation) return '#8b5cf6';
    return isServerHealthy ? '#22c55e' : '#ef4444';
  }, [isCheckingHealth, isServerHealthy, useSimulation]);

  const getServerStatusText = useCallback(() => {
    if (isCheckingHealth) return 'Checking...';
    if (useSimulation) return 'Simulation Mode';
    return isServerHealthy ? 'Server Online' : 'Server Offline';
  }, [isCheckingHealth, isServerHealthy, useSimulation]);

  // ===============================================================
  // 🔧 렌더링 함수들
  // ===============================================================

  const renderImageUploadSection = () => (
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
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
            >
              📷
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
              cursor: 'pointer',
              transition: 'border-color 0.2s'
            }}
          >
            <div style={{ fontSize: isMobile ? '2rem' : '3rem', marginBottom: '1rem' }}>👤</div>
            <p style={{ fontSize: isMobile ? '0.75rem' : '0.875rem', color: '#4b5563', marginBottom: '0.25rem', margin: 0 }}>Upload your photo</p>
            <p style={{ fontSize: isMobile ? '0.625rem' : '0.75rem', color: '#6b7280', margin: 0 }}>PNG, JPG, WebP up to 50MB</p>
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

      {/* Clothing Upload */}
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
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
            >
              👕
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
              cursor: 'pointer',
              transition: 'border-color 0.2s'
            }}
          >
            <div style={{ fontSize: isMobile ? '2rem' : '3rem', marginBottom: '1rem' }}>👕</div>
            <p style={{ fontSize: isMobile ? '0.75rem' : '0.875rem', color: '#4b5563', marginBottom: '0.25rem', margin: 0 }}>Upload clothing item</p>
            <p style={{ fontSize: isMobile ? '0.625rem' : '0.75rem', color: '#6b7280', margin: 0 }}>PNG, JPG, WebP up to 50MB</p>
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

  const renderMeasurementsSection = () => (
    <div style={{ 
      backgroundColor: '#ffffff', 
      borderRadius: isMobile ? '0.5rem' : '0.75rem', 
      border: '1px solid #e5e7eb', 
      padding: isMobile ? '1rem' : '1.5rem', 
      maxWidth: isMobile ? '100%' : '28rem',
      margin: '0 auto 2rem'
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
        </div>
      </div>
      
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

  const renderProcessingSection = () => (
    <div style={{ 
      textAlign: 'center', 
      maxWidth: isMobile ? '100%' : '32rem', 
      margin: '0 auto 2rem' 
    }}>
      <div style={{ 
        backgroundColor: '#ffffff', 
        borderRadius: isMobile ? '0.5rem' : '0.75rem', 
        border: '1px solid #e5e7eb', 
        padding: isMobile ? '1.5rem' : '2rem' 
      }}>
        <div style={{ marginBottom: '1.5rem' }}>
          <div style={{ 
            width: isMobile ? '4rem' : '5rem', 
            height: isMobile ? '4rem' : '5rem', 
            margin: '0 auto', 
            backgroundColor: '#f3e8ff', 
            borderRadius: '50%', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center', 
            marginBottom: '1rem' 
          }}>
            <div style={{ 
              width: isMobile ? '2rem' : '2.5rem', 
              height: isMobile ? '2rem' : '2.5rem', 
              border: '4px solid #7c3aed', 
              borderTop: '4px solid transparent', 
              borderRadius: '50%',
              animation: 'spin 1s linear infinite'
            }}></div>
          </div>
          <h3 style={{ 
            fontSize: isMobile ? '1.25rem' : '1.5rem', 
            fontWeight: '600', 
            color: '#111827' 
          }}>AI 가상 피팅 진행 중</h3>
          <p style={{ 
            color: '#4b5563', 
            marginTop: '0.5rem',
            fontSize: isMobile ? '0.875rem' : '1rem'
          }}>8단계 AI 파이프라인을 실행하고 있습니다</p>
        </div>

        <div style={{ marginTop: '1rem' }}>
          <div style={{ 
            width: '100%', 
            backgroundColor: '#f3f4f6', 
            borderRadius: '0.5rem', 
            height: '0.75rem',
            marginBottom: '0.75rem'
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
            fontSize: isMobile ? '0.875rem' : '1rem', 
            color: '#4b5563',
            fontWeight: '500'
          }}>{progressMessage}</p>
          <p style={{ 
            fontSize: isMobile ? '0.75rem' : '0.875rem', 
            color: '#6b7280',
            marginTop: '0.5rem'
          }}>{progress}% 완료</p>
          
          {/* 취소 버튼 */}
          <button
            onClick={handleCancelRequest}
            style={{
              marginTop: '1.5rem',
              padding: isMobile ? '0.75rem 1.5rem' : '0.5rem 1.5rem',
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
      </div>
    </div>
  );

  const renderResultSection = () => {
    if (!result) return null;

    return (
      <div style={{ 
        maxWidth: isMobile ? '100%' : '64rem', 
        margin: '0 auto 2rem' 
      }}>
        <div style={{ 
          backgroundColor: '#ffffff', 
          borderRadius: isMobile ? '0.5rem' : '0.75rem', 
          border: '1px solid #e5e7eb', 
          padding: isMobile ? '1rem' : '1.5rem' 
        }}>
          <h3 style={{ 
            fontSize: isMobile ? '1.25rem' : '1.5rem', 
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
                    objectFit: 'cover'
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
                  color: '#6b7280',
                  fontSize: isMobile ? '1rem' : '1.125rem'
                }}>
                  🎭 {useSimulation ? '시뮬레이션 모드 - 실제 백엔드 연결 시 진짜 결과 표시' : '결과 이미지 없음'}
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
                    } else {
                      alert('다운로드할 이미지가 없습니다. 실제 백엔드 연결 시 이용 가능합니다.');
                    }
                  }}
                  style={{ 
                    flex: 1, 
                    backgroundColor: '#f3f4f6', 
                    color: '#374151', 
                    padding: isMobile ? '0.75rem 1rem' : '0.5rem 1rem', 
                    borderRadius: '0.5rem', 
                    fontWeight: '500', 
                    border: 'none',
                    cursor: 'pointer',
                    transition: 'background-color 0.2s',
                    fontSize: isMobile ? '0.875rem' : '0.875rem'
                  }}
                >
                  📥 Download
                </button>
                <button style={{ 
                  flex: 1, 
                  backgroundColor: '#000000', 
                  color: '#ffffff', 
                  padding: isMobile ? '0.75rem 1rem' : '0.5rem 1rem', 
                  borderRadius: '0.5rem', 
                  fontWeight: '500', 
                  border: 'none',
                  cursor: 'pointer',
                  transition: 'background-color 0.2s',
                  fontSize: isMobile ? '0.875rem' : '0.875rem'
                }}>
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
                  fontSize: isMobile ? '1rem' : '1.125rem', 
                  fontWeight: '500', 
                  color: '#111827', 
                  marginBottom: '1rem' 
                }}>📊 Fit Analysis</h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  <div>
                    <div style={{ 
                      display: 'flex', 
                      justifyContent: 'space-between', 
                      fontSize: isMobile ? '0.875rem' : '1rem', 
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
                      fontSize: isMobile ? '0.875rem' : '1rem', 
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
                  fontSize: isMobile ? '1rem' : '1.125rem', 
                  fontWeight: '500', 
                  color: '#111827', 
                  marginBottom: '1rem' 
                }}>📋 Details</h4>
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
                    fontSize: isMobile ? '0.875rem' : '1rem' 
                  }}>
                    <span style={{ color: '#4b5563' }}>Category</span>
                    <span style={{ fontWeight: '500', textTransform: 'capitalize' }}>
                      {result?.clothing_analysis?.category || 'Unknown'}
                    </span>
                  </div>
                  <div style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    fontSize: isMobile ? '0.875rem' : '1rem' 
                  }}>
                    <span style={{ color: '#4b5563' }}>Style</span>
                    <span style={{ fontWeight: '500', textTransform: 'capitalize' }}>
                      {result?.clothing_analysis?.style || 'Unknown'}
                    </span>
                  </div>
                  <div style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    fontSize: isMobile ? '0.875rem' : '1rem' 
                  }}>
                    <span style={{ color: '#4b5563' }}>Processing Time</span>
                    <span style={{ fontWeight: '500' }}>{result?.processing_time?.toFixed(1) || 0}s</span>
                  </div>
                  <div style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    fontSize: isMobile ? '0.875rem' : '1rem' 
                  }}>
                    <span style={{ color: '#4b5563' }}>BMI</span>
                    <span style={{ fontWeight: '500' }}>{result?.measurements?.bmi?.toFixed(1) || 0}</span>
                  </div>
                  <div style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    fontSize: isMobile ? '0.875rem' : '1rem' 
                  }}>
                    <span style={{ color: '#4b5563' }}>Session ID</span>
                    <span style={{ fontWeight: '500', fontSize: '0.75rem', fontFamily: 'monospace' }}>
                      {result.session_id}
                    </span>
                  </div>
                </div>
              </div>

              {/* Recommendations */}
              {result.recommendations && result.recommendations.length > 0 && (
                <div>
                  <h4 style={{ 
                    fontSize: isMobile ? '1rem' : '1.125rem', 
                    fontWeight: '500', 
                    color: '#111827', 
                    marginBottom: '1rem' 
                  }}>💡 AI Recommendations</h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {result.recommendations.map((rec, index) => (
                      <div key={index} style={{ 
                        backgroundColor: '#eff6ff', 
                        border: '1px solid #bfdbfe', 
                        borderRadius: '0.5rem', 
                        padding: '0.75rem' 
                      }}>
                        <p style={{ 
                          fontSize: isMobile ? '0.875rem' : '1rem', 
                          color: '#1e40af', 
                          margin: 0 
                        }}>{rec}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* 시뮬레이션 모드 안내 */}
              {useSimulation && (
                <div style={{ 
                  backgroundColor: '#f3e8ff', 
                  border: '1px solid #c4b5fd', 
                  borderRadius: '0.5rem', 
                  padding: '1rem' 
                }}>
                  <p style={{ 
                    fontSize: isMobile ? '0.875rem' : '1rem', 
                    color: '#7c3aed', 
                    margin: 0,
                    fontWeight: '500'
                  }}>
                    🎭 시뮬레이션 모드입니다
                  </p>
                  <p style={{ 
                    fontSize: isMobile ? '0.75rem' : '0.875rem', 
                    color: '#8b5cf6', 
                    margin: '0.25rem 0 0 0' 
                  }}>
                    실제 백엔드 서버 연결 시 진짜 AI 가상 피팅 결과를 볼 수 있습니다.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  };

  // ===============================================================
  // 🔧 메인 렌더링
  // ===============================================================

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
                  <span style={{ 
                    color: '#ffffff',
                    fontSize: isMobile ? '1rem' : '1.25rem'
                  }}>🎯</span>
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
                    'Virtual Try-On System'
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
                    border: '2px solid #7c3aed', 
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
        {/* Title */}
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
            }}>🎭 AI Virtual Try-On</h2>
            {useSimulation && (
              <span style={{ 
                fontSize: isMobile ? '0.75rem' : '0.875rem', 
                color: '#7c3aed',
                backgroundColor: '#f3e8ff',
                padding: '0.25rem 0.5rem',
                borderRadius: '0.25rem',
                fontWeight: '500'
              }}>시뮬레이션 모드</span>
            )}
          </div>
          
          <div style={{ 
            backgroundColor: '#eff6ff', 
            border: '1px solid #bfdbfe', 
            borderRadius: '0.5rem', 
            padding: isMobile ? '0.75rem' : '1rem' 
          }}>
            <p style={{ 
              fontSize: isMobile ? '0.875rem' : '1rem', 
              color: '#1e40af', 
              margin: 0,
              fontWeight: '500'
            }}>
              🚀 완전한 8단계 AI 파이프라인으로 가상 피팅을 체험해보세요
            </p>
            <p style={{ 
              fontSize: isMobile ? '0.75rem' : '0.875rem', 
              color: '#1d4ed8', 
              marginTop: '0.25rem', 
              margin: 0 
            }}>
              이미지 업로드 → 신체 측정 → AI 처리 → 최종 결과 확인
            </p>
          </div>
        </div>

        {/* Content */}
        {!isProcessing && !result && (
          <>
            {renderImageUploadSection()}
            {renderMeasurementsSection()}
          </>
        )}

        {isProcessing && renderProcessingSection()}
        {result && renderResultSection()}

        {/* Action Buttons */}
        {!isProcessing && (
          <div style={{ 
            display: 'flex', 
            justifyContent: 'center',
            flexDirection: isMobile ? 'column' : 'row',
            gap: '1rem',
            marginTop: '2rem'
          }}>
            {result && (
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
                  width: isMobile ? '100%' : 'auto',
                  fontSize: isMobile ? '1rem' : '0.875rem'
                }}
              >
                🔄 새로운 피팅 시작
              </button>
            )}

            {!result && (
              <button
                onClick={handleRunPipeline}
                disabled={!canRunPipeline()}
                style={{
                  padding: isMobile ? '0.875rem 2rem' : '0.75rem 2rem',
                  backgroundColor: !canRunPipeline() ? '#d1d5db' : '#7c3aed',
                  color: '#ffffff',
                  borderRadius: '0.5rem',
                  fontWeight: '500',
                  border: 'none',
                  cursor: !canRunPipeline() ? 'not-allowed' : 'pointer',
                  transition: 'all 0.2s',
                  width: isMobile ? '100%' : 'auto',
                  fontSize: isMobile ? '1rem' : '0.875rem',
                  opacity: !canRunPipeline() ? 0.5 : 1
                }}
                onMouseEnter={(e) => {
                  const target = e.target as HTMLButtonElement;
                  if (!target.disabled) {
                    target.style.backgroundColor = '#6d28d9';
                  }
                }}
                onMouseLeave={(e) => {
                  const target = e.target as HTMLButtonElement;
                  if (!target.disabled) {
                    target.style.backgroundColor = '#7c3aed';
                  }
                }}
              >
                🚀 AI 가상 피팅 시작
              </button>
            )}
          </div>
        )}

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
                <span style={{ fontSize: '1.25rem', marginRight: '0.75rem' }}>❌</span>
                <div style={{ flex: 1 }}>
                  <h3 style={{ 
                    fontSize: isMobile ? '0.875rem' : '1rem', 
                    fontWeight: '500', 
                    color: '#991b1b', 
                    margin: 0 
                  }}>처리 중 오류가 발생했습니다</h3>
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
                  flexShrink: 0,
                  fontSize: '1rem'
                }}
              >
                ✕
              </button>
            </div>
          </div>
        )}

        {/* Instructions (처음 로드 시에만 표시) */}
        {!personImage && !clothingImage && !isProcessing && !result && (
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
            }}>🎯 사용 방법</h3>
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: isMobile ? '1fr' : 'repeat(3, 1fr)', 
              gap: isMobile ? '1rem' : '1.5rem' 
            }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ 
                  width: isMobile ? '3rem' : '4rem', 
                  height: isMobile ? '3rem' : '4rem', 
                  backgroundColor: '#f3f4f6', 
                  borderRadius: '0.5rem', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center', 
                  margin: '0 auto 0.75rem',
                  fontSize: isMobile ? '1.5rem' : '2rem'
                }}>
                  📸
                </div>
                <h4 style={{ 
                  fontWeight: '500', 
                  color: '#111827', 
                  marginBottom: '0.5rem',
                  fontSize: isMobile ? '0.875rem' : '1rem'
                }}>1. 이미지 업로드</h4>
                <p style={{ 
                  fontSize: isMobile ? '0.75rem' : '0.875rem', 
                  color: '#4b5563' 
                }}>사용자 사진과 의류 이미지를 업로드하세요</p>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ 
                  width: isMobile ? '3rem' : '4rem', 
                  height: isMobile ? '3rem' : '4rem', 
                  backgroundColor: '#f3f4f6', 
                  borderRadius: '0.5rem', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center', 
                  margin: '0 auto 0.75rem',
                  fontSize: isMobile ? '1.5rem' : '2rem'
                }}>
                  📏
                </div>
                <h4 style={{ 
                  fontWeight: '500', 
                  color: '#111827', 
                  marginBottom: '0.5rem',
                  fontSize: isMobile ? '0.875rem' : '1rem'
                }}>2. 신체 측정값</h4>
                <p style={{ 
                  fontSize: isMobile ? '0.75rem' : '0.875rem', 
                  color: '#4b5563' 
                }}>키와 몸무게를 정확히 입력하세요</p>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ 
                  width: isMobile ? '3rem' : '4rem', 
                  height: isMobile ? '3rem' : '4rem', 
                  backgroundColor: '#f3f4f6', 
                  borderRadius: '0.5rem', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center', 
                  margin: '0 auto 0.75rem',
                  fontSize: isMobile ? '1.5rem' : '2rem'
                }}>
                  🎭
                </div>
                <h4 style={{ 
                  fontWeight: '500', 
                  color: '#111827', 
                  marginBottom: '0.5rem',
                  fontSize: isMobile ? '0.875rem' : '1rem'
                }}>3. AI 가상 피팅</h4>
                <p style={{ 
                  fontSize: isMobile ? '0.75rem' : '0.875rem', 
                  color: '#4b5563' 
                }}>8단계 AI 파이프라인으로 결과를 생성합니다</p>
              </div>
            </div>
            
            {/* 시스템 정보 */}
            <div style={{ 
              marginTop: '1.5rem', 
              padding: isMobile ? '0.75rem' : '1rem', 
              backgroundColor: '#f9fafb', 
              borderRadius: '0.5rem',
              fontSize: isMobile ? '0.75rem' : '0.875rem',
              color: '#4b5563'
            }}>
              <p style={{ margin: 0, fontWeight: '500' }}>
                🛠️ 시스템 정보:
              </p>
              {systemInfo && !useSimulation && (
                <p style={{ margin: '0.25rem 0 0 0' }}>
                  🎯 {systemInfo.app_name} v{systemInfo.app_version} | 
                  {systemInfo.device_name} {systemInfo.is_m3_max ? '🍎' : ''} | 
                  💾 {systemInfo.available_memory_gb}GB 사용가능
                </p>
              )}
              <p style={{ margin: '0.25rem 0 0 0' }}>
                {useSimulation ? 
                  '🎭 시뮬레이션 모드 - 실제 백엔드 연결 시 진짜 AI 처리' : 
                  '🚀 실시간 8단계 AI 파이프라인 | M3 Max 최적화 | WebSocket 통신'
                }
              </p>
              {!isMobile && !useSimulation && (
                <p style={{ margin: '0.25rem 0 0 0' }}>
                  🔧 헤더의 "Test", "System" 버튼으로 연결 상태 확인 가능
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
                backgroundColor: '#7c3aed',
                color: '#ffffff',
                border: 'none',
                boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '1.25rem'
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
                🔌 연결 테스트
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
                📊 시스템 정보
              </button>
              <button
                onClick={() => {
                  if (useSimulation) {
                    alert('🎭 현재 시뮬레이션 모드입니다.\n실제 백엔드 서버를 시작하고 다시 시도해주세요.');
                  } else {
                    alert('✅ 실제 백엔드 연결됨\n정상적으로 AI 처리가 가능합니다.');
                  }
                }}
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
                🎭 모드 확인
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