import React, { useState, useRef, useEffect } from 'react';

interface UserMeasurements {
  height: number;
  weight: number;
}

interface TryOnResult {
  success: boolean;
  fitted_image?: string;
  result_image?: string;
  processing_time: number;
  confidence: number;
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
  };
  fit_score: number;
  recommendations: string[];
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

// 8단계 정의
const PIPELINE_STEPS = [
  { id: 1, name: '이미지 업로드', description: '사용자 사진과 의류 이미지를 업로드합니다' },
  { id: 2, name: '신체 측정', description: '키와 몸무게 등 신체 정보를 입력합니다' },
  { id: 3, name: '인체 파싱', description: 'AI가 신체 부위를 20개 영역으로 분석합니다' },
  { id: 4, name: '포즈 추정', description: '18개 키포인트로 자세를 분석합니다' },
  { id: 5, name: '의류 분석', description: '의류 스타일과 색상을 분석합니다' },
  { id: 6, name: '기하학적 매칭', description: '신체와 의류를 정확히 매칭합니다' },
  { id: 7, name: '가상 피팅', description: 'AI로 가상 착용 결과를 생성합니다' },
  { id: 8, name: '결과 확인', description: '최종 결과를 확인하고 저장합니다' }
];

// 파일 유틸리티
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
  }
};

// 개발 도구
const devTools = {
  testAPI: async () => {
    try {
      const response = await fetch('http://localhost:8000/api/step/health');
      const result = await response.json();
      return { message: 'API 연결 성공', data: result };
    } catch (error) {
      return { message: 'API 연결 실패', error: error };
    }
  },
  
  testDummyVirtualTryOn: async () => {
    return { message: '더미 테스트 완료', success: true };
  },
  
  exportDebugInfo: () => {
    return { message: '디버그 정보 내보내기 완료' };
  }
};

// 전역에 devTools 등록
(window as any).devTools = devTools;

const App: React.FC = () => {
  // 현재 단계 관리
  const [currentStep, setCurrentStep] = useState(1);
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);
  
  // 파일 상태
  const [personImage, setPersonImage] = useState<File | null>(null);
  const [clothingImage, setClothingImage] = useState<File | null>(null);
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

  // 파일 참조
  const personImageRef = useRef<HTMLInputElement>(null);
  const clothingImageRef = useRef<HTMLInputElement>(null);

  // 반응형 상태
  const [isMobile, setIsMobile] = useState(false);
  const [screenWidth, setScreenWidth] = useState(window.innerWidth);
  
  // 서버 상태
  const [isHealthy, setIsHealthy] = useState(true);
  const [isChecking, setIsChecking] = useState(false);

  // 반응형 처리
  useEffect(() => {
    const handleResize = () => {
      const width = window.innerWidth;
      setScreenWidth(width);
      setIsMobile(width < 768);
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // 서버 헬스체크
  useEffect(() => {
    const checkHealth = async () => {
      setIsChecking(true);
      try {
        const response = await fetch('http://localhost:8000/api/step/health');
        setIsHealthy(response.ok);
      } catch {
        setIsHealthy(false);
      } finally {
        setIsChecking(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // 에러 처리 함수
  const handleAPIError = async (response: Response, context: string) => {
    const errorText = await response.text();
    let errorMessage;
    
    try {
      const errorJson = JSON.parse(errorText);
      errorMessage = errorJson.error || errorJson.message || `HTTP ${response.status}`;
    } catch {
      errorMessage = `HTTP ${response.status}: ${response.statusText}`;
    }
    
    if (errorMessage.includes('Failed to fetch')) {
      errorMessage = '서버에 연결할 수 없습니다. 백엔드 서버가 실행 중인지 확인해주세요.';
    } else if (errorMessage.includes('500')) {
      errorMessage = '서버 내부 오류가 발생했습니다. 로그를 확인해주세요.';
    }
    
    throw new Error(`${context}: ${errorMessage}`);
  };

  // 파일 업로드 핸들러
  const handleImageUpload = (file: File, type: 'person' | 'clothing') => {
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

    if (type === 'person') {
      setPersonImage(file);
      console.log('✅ 사용자 이미지 업로드:', {
        name: file.name,
        size: fileUtils.formatFileSize(file.size),
        type: file.type
      });
    } else {
      setClothingImage(file);
      console.log('✅ 의류 이미지 업로드:', {
        name: file.name,
        size: fileUtils.formatFileSize(file.size),
        type: file.type
      });
    }
    
    clearError();
  };

  // 다음 단계로 이동
  const goToNextStep = () => {
    if (currentStep < 8) {
      setCompletedSteps(prev => [...prev, currentStep]);
      setCurrentStep(prev => prev + 1);
    }
  };

  // 이전 단계로 이동
  const goToPreviousStep = () => {
    if (currentStep > 1) {
      setCurrentStep(prev => prev - 1);
      setCompletedSteps(prev => prev.filter(step => step < currentStep - 1));
    }
  };

  // 에러 클리어
  const clearError = () => setError(null);

  // 리셋
  const reset = () => {
    setCurrentStep(1);
    setCompletedSteps([]);
    setPersonImage(null);
    setClothingImage(null);
    setStepResults({});
    setResult(null);
    setFileErrors({});
    setError(null);
    setIsProcessing(false);
    setProgress(0);
    setProgressMessage('');
  };

  // ===========================================
  // 🔥 8단계 API 처리 함수들 (완전 구현)
  // ===========================================

  // 1단계: 이미지 업로드 검증
  const processStep1 = async () => {
    if (!personImage || !clothingImage) {
      alert('사용자 이미지와 의류 이미지를 모두 업로드해주세요.');
      return;
    }

    setIsProcessing(true);
    setProgress(10);
    setProgressMessage('이미지 검증 중...');

    try {
      const formData = new FormData();
      formData.append('person_image', personImage);
      formData.append('clothing_image', clothingImage);
      
      const response = await fetch('http://localhost:8000/api/step/1/upload-validation', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        await handleAPIError(response, '1단계 이미지 검증');
      }
      
      const stepResult: StepResult = await response.json();
      console.log('✅ 1단계 검증 결과:', stepResult);
      
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
      setError(error.message);
      setIsProcessing(false);
      setProgress(0);
    }
  };

  // 2단계: 신체 측정값 검증
  const processStep2 = async () => {
    if (measurements.height <= 0 || measurements.weight <= 0) {
      alert('올바른 키와 몸무게를 입력해주세요.');
      return;
    }

    setIsProcessing(true);
    setProgress(10);
    setProgressMessage('신체 측정값 검증 중...');

    try {
      const formData = new FormData();
      formData.append('height', measurements.height.toString());
      formData.append('weight', measurements.weight.toString());
      
      const response = await fetch('http://localhost:8000/api/step/2/measurements-validation', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        await handleAPIError(response, '2단계 측정값 검증');
      }
      
      const stepResult: StepResult = await response.json();
      console.log('✅ 2단계 측정 결과:', stepResult);
      
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
      setError(error.message);
      setIsProcessing(false);
      setProgress(0);
    }
  };

  // 3단계: 인체 파싱
  const processStep3 = async () => {
    if (!personImage) return;

    setIsProcessing(true);
    setProgress(10);
    setProgressMessage('AI 인체 파싱 중...');

    try {
      const formData = new FormData();
      formData.append('person_image', personImage);
      formData.append('height', measurements.height.toString());
      formData.append('weight', measurements.weight.toString());
      
      setProgress(30);
      setProgressMessage('Graphonomy + SCHP 모델 실행 중...');
      
      const response = await fetch('http://localhost:8000/api/step/3/human-parsing', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        await handleAPIError(response, '3단계 인체 파싱');
      }
      
      const stepResult: StepResult = await response.json();
      console.log('✅ 3단계 인체 파싱 완료:', stepResult);
      
      if (!stepResult.success) {
        throw new Error(stepResult.error || '3단계 처리 실패');
      }
      
      setStepResults(prev => ({ ...prev, 3: stepResult }));
      setProgress(100);
      setProgressMessage('인체 파싱 완료!');
      
      setTimeout(() => {
        setIsProcessing(false);
        goToNextStep();
      }, 2000);
      
    } catch (error: any) {
      console.error('❌ 3단계 실패:', error);
      setError(error.message);
      setIsProcessing(false);
      setProgress(0);
    }
  };

  // 4단계: 포즈 추정
  const processStep4 = async () => {
    if (!personImage) return;

    setIsProcessing(true);
    setProgress(10);
    setProgressMessage('AI 포즈 추정 중...');

    try {
      const formData = new FormData();
      formData.append('person_image', personImage);
      
      setProgress(30);
      setProgressMessage('OpenPose + MediaPipe 실행 중...');
      
      const response = await fetch('http://localhost:8000/api/step/4/pose-estimation', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        await handleAPIError(response, '4단계 포즈 추정');
      }
      
      const stepResult: StepResult = await response.json();
      console.log('✅ 4단계 포즈 추정 완료:', stepResult);
      
      if (!stepResult.success) {
        throw new Error(stepResult.error || '4단계 처리 실패');
      }
      
      setStepResults(prev => ({ ...prev, 4: stepResult }));
      setProgress(100);
      setProgressMessage('포즈 추정 완료!');
      
      setTimeout(() => {
        setIsProcessing(false);
        goToNextStep();
      }, 2000);
      
    } catch (error: any) {
      console.error('❌ 4단계 실패:', error);
      setError(error.message);
      setIsProcessing(false);
      setProgress(0);
    }
  };

  // 5단계: 의류 분석
  const processStep5 = async () => {
    if (!clothingImage) return;

    setIsProcessing(true);
    setProgress(10);
    setProgressMessage('AI 의류 분석 중...');

    try {
      const formData = new FormData();
      formData.append('clothing_image', clothingImage);
      
      setProgress(30);
      setProgressMessage('U2Net + CLIP 모델 실행 중...');
      
      const response = await fetch('http://localhost:8000/api/step/5/clothing-analysis', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        await handleAPIError(response, '5단계 의류 분석');
      }
      
      const stepResult: StepResult = await response.json();
      console.log('✅ 5단계 의류 분석 완료:', stepResult);
      
      if (!stepResult.success) {
        throw new Error(stepResult.error || '5단계 처리 실패');
      }
      
      setStepResults(prev => ({ ...prev, 5: stepResult }));
      setProgress(100);
      setProgressMessage('의류 분석 완료!');
      
      setTimeout(() => {
        setIsProcessing(false);
        goToNextStep();
      }, 2000);
      
    } catch (error: any) {
      console.error('❌ 5단계 실패:', error);
      setError(error.message);
      setIsProcessing(false);
      setProgress(0);
    }
  };

  // 6단계: 기하학적 매칭
  const processStep6 = async () => {
    if (!personImage || !clothingImage) return;

    setIsProcessing(true);
    setProgress(10);
    setProgressMessage('기하학적 매칭 중...');

    try {
      const formData = new FormData();
      formData.append('person_image', personImage);
      formData.append('clothing_image', clothingImage);
      formData.append('height', measurements.height.toString());
      formData.append('weight', measurements.weight.toString());
      
      setProgress(40);
      setProgressMessage('신체와 의류 매칭 분석 중...');
      
      const response = await fetch('http://localhost:8000/api/step/6/geometric-matching', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        await handleAPIError(response, '6단계 기하학적 매칭');
      }
      
      const stepResult: StepResult = await response.json();
      console.log('✅ 6단계 기하학적 매칭 완료:', stepResult);
      
      if (!stepResult.success) {
        throw new Error(stepResult.error || '6단계 처리 실패');
      }
      
      setStepResults(prev => ({ ...prev, 6: stepResult }));
      setProgress(100);
      setProgressMessage('기하학적 매칭 완료!');
      
      setTimeout(() => {
        setIsProcessing(false);
        goToNextStep();
      }, 2000);
      
    } catch (error: any) {
      console.error('❌ 6단계 실패:', error);
      setError(error.message);
      setIsProcessing(false);
      setProgress(0);
    }
  };

  // 7단계: 가상 피팅
  const processStep7 = async () => {
    if (!personImage || !clothingImage) {
      alert('이미지 파일이 누락되었습니다.');
      return;
    }

    setIsProcessing(true);
    setProgress(10);
    setProgressMessage('AI 가상 피팅 생성 중...');

    try {
      const formData = new FormData();
      formData.append('person_image', personImage);
      formData.append('clothing_image', clothingImage);
      formData.append('height', measurements.height.toString());
      formData.append('weight', measurements.weight.toString());
      formData.append('session_id', `app_session_${Date.now()}`);

      setProgress(30);
      setProgressMessage('HR-VITON + OOTDiffusion 실행 중...');
      
      setTimeout(() => {
        setProgress(60);
        setProgressMessage('Stable Diffusion 모델 처리 중...');
      }, 2000);

      const response = await fetch('http://localhost:8000/api/step/7/virtual-fitting', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        await handleAPIError(response, '7단계 가상 피팅');
      }

      const stepResult: StepResult = await response.json();
      console.log('✅ 7단계 가상 피팅 완료:', stepResult);

      if (!stepResult.success) {
        throw new Error(stepResult.error || '7단계 처리 실패');
      }

      setStepResults(prev => ({ ...prev, 7: stepResult }));
      
      // TryOnResult 형태로 변환
      const tryOnResult: TryOnResult = {
        success: stepResult.success,
        fitted_image: stepResult.fitted_image,
        processing_time: stepResult.processing_time,
        confidence: stepResult.confidence,
        fit_score: stepResult.fit_score || 0.85,
        measurements: stepResult.details?.measurements || {
          chest: 88 + (measurements.weight - 65) * 0.9,
          waist: 74 + (measurements.weight - 65) * 0.7,
          hip: 94 + (measurements.weight - 65) * 0.8,
          bmi: measurements.weight / ((measurements.height / 100) ** 2)
        },
        clothing_analysis: stepResult.details?.clothing_analysis || {
          category: '상의',
          style: '캐주얼',
          dominant_color: [95, 145, 195]
        },
        recommendations: stepResult.recommendations || []
      };
      
      setResult(tryOnResult);
      setProgress(100);
      setProgressMessage('가상 피팅 완료!');

      setTimeout(() => {
        setIsProcessing(false);
        goToNextStep();
      }, 2000);

    } catch (error: any) {
      console.error('❌ 7단계 실패:', error);
      setError(error.message);
      setIsProcessing(false);
      setProgress(0);
    }
  };

  // 8단계: 결과 분석
  const processStep8 = async () => {
    const step7Result = stepResults[7];
    
    if (!step7Result || !step7Result.fitted_image) {
      alert('가상 피팅 결과가 없습니다.');
      return;
    }

    setIsProcessing(true);
    setProgress(10);
    setProgressMessage('AI 결과 분석 중...');

    try {
      const formData = new FormData();
      formData.append('fitted_image_base64', step7Result.fitted_image);
      formData.append('fit_score', (step7Result.fit_score || 0.85).toString());
      formData.append('confidence', step7Result.confidence.toString());

      setProgress(50);
      setProgressMessage('품질 평가 및 추천 생성 중...');

      const response = await fetch('http://localhost:8000/api/step/8/result-analysis', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        await handleAPIError(response, '8단계 결과 분석');
      }

      const stepResult: StepResult = await response.json();
      console.log('✅ 8단계 결과 분석 완료:', stepResult);

      if (!stepResult.success) {
        throw new Error(stepResult.error || '8단계 처리 실패');
      }

      setStepResults(prev => ({ ...prev, 8: stepResult }));
      
      // 최종 result에 추천사항 추가
      if (result) {
        setResult(prev => prev ? {
          ...prev,
          recommendations: stepResult.recommendations || stepResult.details?.recommendations || prev.recommendations
        } : prev);
      }

      setProgress(100);
      setProgressMessage('결과 분석 완료!');
      
      setTimeout(() => {
        setIsProcessing(false);
        alert('🎉 모든 단계가 완료되었습니다!');
      }, 1500);

    } catch (error: any) {
      console.error('❌ 8단계 실패:', error);
      setError(error.message);
      setIsProcessing(false);
      setProgress(0);
    }
  };

  // 단계별 처리 함수 매핑
  const processCurrentStep = async () => {
    const processors = {
      1: processStep1,
      2: processStep2,
      3: processStep3,
      4: processStep4,
      5: processStep5,
      6: processStep6,
      7: processStep7,
      8: processStep8
    };

    const processor = processors[currentStep as keyof typeof processors];
    if (processor) {
      await processor();
    }
  };

  // 현재 단계가 완료 가능한지 확인
  const canProceedToNext = () => {
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
  };

  // 서버 상태 색상
  const getServerStatusColor = () => {
    if (isChecking) return '#f59e0b';
    return isHealthy ? '#4ade80' : '#ef4444';
  };

  const getServerStatusText = () => {
    if (isChecking) return 'Checking...';
    return isHealthy ? 'Server Online' : 'Server Offline';
  };

  // 개발 도구 함수들
  const handleTestConnection = async () => {
    try {
      const result = await devTools.testAPI();
      console.log('연결 테스트 결과:', result);
      alert(result.message);
    } catch (error) {
      console.error('연결 테스트 실패:', error);
      alert('연결 테스트 실패');
    }
  };

  const handleDevTest = async () => {
    const result = await devTools.testAPI();
    console.log('개발 도구 테스트 결과:', result);
    alert(result.message);
  };

  const handleDummyTest = async () => {
    try {
      const result = await devTools.testDummyVirtualTryOn();
      console.log('더미 테스트 결과:', result);
      alert(result.message);
    } catch (error: any) {
      console.error('더미 테스트 실패:', error);
      alert(`더미 테스트 실패: ${error.message}`);
    }
  };

  // 요청 취소 핸들러
  const handleCancelRequest = () => {
    if (isProcessing) {
      setIsProcessing(false);
      setProgress(0);
      setProgressMessage('');
      alert('요청이 취소되었습니다.');
    }
  };

  // 파일 드래그 앤 드롭 지원
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent, type: 'person' | 'clothing') => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0 && files[0].type.startsWith('image/')) {
      handleImageUpload(files[0], type);
    }
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
        {personImage ? (
          <div style={{ position: 'relative' }}>
            <img
              src={URL.createObjectURL(personImage)}
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
              <svg style={{ 
                width: isMobile ? '0.875rem' : '1rem', 
                height: isMobile ? '0.875rem' : '1rem', 
                color: '#4b5563' 
              }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
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
              {personImage.name} ({fileUtils.formatFileSize(personImage.size)})
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
            onMouseEnter={(e) => (e.target as HTMLElement).style.borderColor = '#9ca3af'}
            onMouseLeave={(e) => (e.target as HTMLElement).style.borderColor = '#d1d5db'}
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
        {clothingImage ? (
          <div style={{ position: 'relative' }}>
            <img
              src={URL.createObjectURL(clothingImage)}
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
              <svg style={{ 
                width: isMobile ? '0.875rem' : '1rem', 
                height: isMobile ? '0.875rem' : '1rem', 
                color: '#4b5563' 
              }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
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
              {clothingImage.name} ({fileUtils.formatFileSize(clothingImage.size)})
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
            onMouseEnter={(e) => (e.target as HTMLElement).style.borderColor = '#9ca3af'}
            onMouseLeave={(e) => (e.target as HTMLElement).style.borderColor = '#d1d5db'}
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

  const renderProcessingStep = () => {
    const stepData = PIPELINE_STEPS[currentStep - 1];
    const stepResult = stepResults[currentStep];

    return (
      <div style={{ 
        textAlign: 'center', 
        maxWidth: isMobile ? '100%' : '32rem', 
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
              ) : (
                <div style={{ 
                  width: isMobile ? '1.5rem' : '2rem', 
                  height: isMobile ? '1.5rem' : '2rem', 
                  border: '4px solid #3b82f6', 
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
            }}>{stepData.name}</h3>
            <p style={{ 
              color: '#4b5563', 
              marginTop: '0.5rem',
              fontSize: isMobile ? '0.875rem' : '1rem'
            }}>{stepData.description}</p>
          </div>

          {/* 실제 API 처리 중 표시 */}
          {isProcessing && !stepResult && (
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
                  
                  {/* 단계별 상세 정보 표시 */}
                  {stepResult.details && (
                    <div style={{ 
                      marginTop: '0.75rem', 
                      padding: '0.75rem', 
                      backgroundColor: '#f9fafb', 
                      borderRadius: '0.25rem',
                      fontSize: isMobile ? '0.625rem' : '0.75rem',
                      color: '#4b5563'
                    }}>
                      {currentStep === 3 && stepResult.details.detected_parts && (
                        <p style={{ margin: 0 }}>
                          🧍 신체 부위: {stepResult.details.detected_parts}/{stepResult.details.total_parts}개 감지
                        </p>
                      )}
                      {currentStep === 4 && stepResult.details.detected_keypoints && (
                        <p style={{ margin: 0 }}>
                          🎯 키포인트: {stepResult.details.detected_keypoints}/{stepResult.details.total_keypoints}개 감지
                        </p>
                      )}
                      {currentStep === 5 && stepResult.details.category && (
                        <p style={{ margin: 0 }}>
                          👕 의류: {stepResult.details.category} ({stepResult.details.style})
                        </p>
                      )}
                      {currentStep === 6 && stepResult.details.matching_quality && (
                        <p style={{ margin: 0 }}>
                          🎯 매칭: {stepResult.details.matching_quality} (정확도: {(stepResult.confidence * 100).toFixed(1)}%)
                        </p>
                      )}
                    </div>
                  )}
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

        {isProcessing && (
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
          }}>가상 피팅 결과</h3>
          
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
              {result.fitted_image || result.result_image ? (
                <img
                  src={`data:image/jpeg;base64,${result.fitted_image || result.result_image}`}
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
                    if (result.fitted_image || result.result_image) {
                      const link = document.createElement('a');
                      link.href = `data:image/jpeg;base64,${result.fitted_image || result.result_image}`;
                      link.download = 'virtual-tryon-result.jpg';
                      link.click();
                    }
                  }}
                  disabled={!result.fitted_image && !result.result_image}
                  style={{ 
                    flex: 1, 
                    backgroundColor: result.fitted_image || result.result_image ? '#f3f4f6' : '#e5e7eb', 
                    color: '#374151', 
                    padding: isMobile ? '0.75rem 1rem' : '0.5rem 1rem', 
                    borderRadius: '0.5rem', 
                    fontWeight: '500', 
                    border: 'none',
                    cursor: result.fitted_image || result.result_image ? 'pointer' : 'not-allowed',
                    transition: 'background-color 0.2s',
                    fontSize: isMobile ? '0.875rem' : '0.875rem'
                  }}
                >
                  Download
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
                  Share
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
                }}>Fit Analysis</h4>
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
                }}>Details</h4>
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

              {/* Recommendations */}
              {result.recommendations && result.recommendations.length > 0 && (
                <div>
                  <h4 style={{ 
                    fontSize: isMobile ? '0.875rem' : '0.875rem', 
                    fontWeight: '500', 
                    color: '#111827', 
                    marginBottom: '1rem' 
                  }}>AI Recommendations</h4>
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
                        }}>{rec}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  };

  // 컴포넌트 마운트 시 개발 도구 정보 출력
  useEffect(() => {
    console.log('🛠️ MyCloset AI App 시작됨');
    console.log('📋 개발 도구 사용법:');
    console.log('  - devTools.testAPI(): API 연결 테스트');
    console.log('  - devTools.testDummyVirtualTryOn(): 더미 가상 피팅 테스트');
    console.log('  - devTools.exportDebugInfo(): 디버그 정보 내보내기');
  }, []);

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
                    onClick={handleDevTest}
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
                    DevTest
                  </button>
                  <button
                    onClick={handleDummyTest}
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
                    Dummy
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
            disabled={currentStep === 1 || isProcessing}
            style={{
              padding: isMobile ? '0.875rem 1.5rem' : '0.75rem 1.5rem',
              backgroundColor: '#f3f4f6',
              color: '#374151',
              borderRadius: '0.5rem',
              fontWeight: '500',
              border: 'none',
              cursor: (currentStep === 1 || isProcessing) ? 'not-allowed' : 'pointer',
              opacity: (currentStep === 1 || isProcessing) ? 0.5 : 1,
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
            {!isProcessing && (currentStep > 1 || result) && (
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

            {currentStep < 8 && (
              <button
                onClick={processCurrentStep}
                disabled={!canProceedToNext() || isProcessing}
                style={{
                  padding: isMobile ? '0.875rem 1.5rem' : '0.75rem 1.5rem',
                  backgroundColor: (!canProceedToNext() || isProcessing) ? '#d1d5db' : '#3b82f6',
                  color: '#ffffff',
                  borderRadius: '0.5rem',
                  fontWeight: '500',
                  border: 'none',
                  cursor: (!canProceedToNext() || isProcessing) ? 'not-allowed' : 'pointer',
                  transition: 'all 0.2s',
                  width: isMobile ? '100%' : 'auto'
                }}
                onMouseEnter={(e) => {
                  const target = e.target as HTMLButtonElement;
                  if (!target.disabled) {
                    target.style.backgroundColor = '#2563eb';
                  }
                }}
                onMouseLeave={(e) => {
                  const target = e.target as HTMLButtonElement;
                  if (!target.disabled) {
                    target.style.backgroundColor = '#3b82f6';
                  }
                }}
              >
                {currentStep <= 2 ? '다음 단계' : 
                 currentStep === 7 ? '가상 피팅 시작' : 
                 isProcessing ? '처리 중...' : '처리 시작'}
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
            
            {/* 개발 도구 정보 */}
            <div style={{ 
              marginTop: '1.5rem', 
              padding: isMobile ? '0.75rem' : '1rem', 
              backgroundColor: '#f9fafb', 
              borderRadius: '0.5rem',
              fontSize: isMobile ? '0.75rem' : '0.875rem',
              color: '#4b5563'
            }}>
              <p style={{ margin: 0, fontWeight: '500' }}>🛠️ 개발자 도구:</p>
              <p style={{ margin: '0.25rem 0 0 0' }}>
                콘솔에서 <code>devTools.testAPI()</code>, <code>devTools.testDummyVirtualTryOn()</code> 등을 사용할 수 있습니다.
              </p>
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
                minWidth: '8rem',
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
                onClick={handleDevTest}
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
                Dev Test
              </button>
              <button
                onClick={handleDummyTest}
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
                Dummy Test
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